import os
import gc
import json
import random
import time
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config, data_dir
from data.dataset import DigitsDataset
from models import create_model
from losses.classification import LabelSmoothEntropy
from losses.attention import AttentionSupervisionLoss, attention_diversity_loss, spatial_ordering_loss
from losses.augmentation import cutmix_data
from trainer.base import BaseTrainer, ModelEMA


def _compute_joint_acc(pred_heads, labels, true_lengths, num_heads):
    head_correct = [(pred_heads[h].argmax(1) == labels[:, h]) for h in range(num_heads)]
    temp = t.stack(head_correct, dim=1)
    valid_head_mask = t.stack([(true_lengths > h).float() for h in range(num_heads)], dim=1)
    correct_mask = (temp | (valid_head_mask == 0))
    return t.all(correct_mask, dim=1).sum().item()


def _compute_char_acc(pred_heads, labels, true_lengths, num_heads):
    corrects = 0
    total_chars = true_lengths.sum().item()
    for h in range(num_heads):
        valid_mask = (true_lengths > h).float()
        if valid_mask.sum() > 0:
            corrects += ((pred_heads[h].argmax(1) == labels[:, h]) * valid_mask).sum().item()
    return corrects, total_chars


class MultiHeadTrainer(BaseTrainer):
    _checkpoint_prefix = 'resnet101'
    _eval_label = 'Evaluation'

    def __init__(self, val=True, model_type=None):
        super().__init__()
        print(f'Using device: {self.device}')
        self._model_type = model_type or config.model_type
        self.train_set = DigitsDataset(mode='train', aug=True,
                                       input_size=(config.input_height, config.input_width))
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True)
        if val:
            self.val_set = DigitsDataset(mode='val', aug=False,
                                         input_size=(config.input_height, config.input_width))
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False)
        else:
            self.val_loader = None

        self.model = create_model(self._model_type).to(self.device)
        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        if config.use_torch_compile and t.cuda.is_available():
            try:
                self.model = t.compile(self.model, mode=config.compile_mode)
                print(f'torch.compile enabled (mode={config.compile_mode})')
            except Exception as e:
                print(f'torch.compile failed: {e}, falling back to eager mode')
                config.use_torch_compile = False

        self.attn_supervision = AttentionSupervisionLoss()

        class_weights = self._compute_class_weights()

        self.head_criteria = nn.ModuleList()
        for h in range(config.num_heads):
            self.head_criteria.append(LabelSmoothEntropy(smooth=config.smooth, class_weights=class_weights))

        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() if not n.startswith('backbone.')]
        self.optimizer = self._setup_optimizer(backbone_params, other_params)
        self.lr_scheduler = self._setup_scheduler()
        self.scaler = self._setup_scaler()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_init(self._model_type, self.device, total_params, trainable_params)

        if config.pretrained is not None:
            self.load_model(config.pretrained, save_opt=False)
            print(f'Load model from {config.pretrained}')
            print('Warning: Optimizer and scheduler NOT restored. Using new config.')
            if 'best_acc' in t.load(config.pretrained, map_location='cpu', weights_only=False):
                self.best_acc = t.load(config.pretrained, map_location='cpu', weights_only=False)['best_acc']
                print(f'Restored best_acc: {self.best_acc * 100:.2f}%')

    def _compute_class_weights(self):
        class_counts = t.zeros(config.class_num)
        with open(data_dir['train_label'], 'r') as f:
            labels = json.load(f)
        for img_name, mark in labels.items():
            digits = mark['label'][:config.num_heads]
            for d in digits:
                class_counts[d] += 1
            for _ in range(config.num_heads - len(digits)):
                class_counts[10] += 1
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights * config.class_num / class_weights.sum()
        class_weights = class_weights.to(self.device)
        print(f'Computed class weights from JSON: {class_weights.cpu().numpy()}')
        print(f'   Class 10 (empty) weight: {class_weights[10].item():.3f}')
        return class_weights

    def _make_loader(self, dataset, batch_size, shuffle=False, drop_last=False):
        kwargs = dict(
            batch_size=batch_size, shuffle=shuffle,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            drop_last=drop_last, prefetch_factor=config.prefetch_factor,
        )
        if config.num_workers > 0:
            kwargs['persistent_workers'] = config.persistent_workers
        if config.multiprocessing_context is not None:
            kwargs['multiprocessing_context'] = config.multiprocessing_context
        return DataLoader(dataset, **kwargs)

    def _pre_epoch_hook(self, epoch):
        if hasattr(self.model, 'set_roi_gt_prob'):
            if epoch < config.warmup_epochs:
                self.model.set_roi_gt_prob(1.0)
            else:
                decay_end = int(config.epoches * 0.6)
                if epoch >= decay_end:
                    self.model.set_roi_gt_prob(0.0)
                else:
                    progress = (epoch - config.warmup_epochs) / max(decay_end - config.warmup_epochs, 1)
                    self.model.set_roi_gt_prob(1.0 - progress)

    def _cleanup_dataloader(self, loader):
        if loader is not None:
            if hasattr(loader, '_iterator'):
                try:
                    del loader._iterator
                except Exception:
                    pass
            gc.collect()

    def _rebuild_dataloaders(self):
        self._cleanup_dataloader(self.train_loader)
        self._cleanup_dataloader(self.val_loader)
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True)
        if self.val_loader is not None:
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False)

    def _train_epoch(self, epoch):
        total_loss = 0
        joint_corrects = 0
        joint_total = 0
        char_corrects = 0
        total_chars = 0
        batch_start = time.time()
        tbar = tqdm(self.train_loader)
        self.model.train()

        for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            bbox_target = bbox_target.to(self.device)
            bbox_mask = bbox_mask.to(self.device)

            if config.cutmix_prob > 0 and random.random() < config.cutmix_prob and config.cutmix_alpha > 0:
                img, label_a, label_b, bbox_a, bbox_b, mask_a, mask_b, lam = cutmix_data(
                    img, label, bbox_target, bbox_mask, alpha=config.cutmix_alpha)
                use_cutmix = True
            else:
                label_a = label
                label_b = label
                bbox_a = bbox_target
                bbox_b = bbox_target
                mask_a = bbox_mask
                mask_b = bbox_mask
                lam = 1.0
                use_cutmix = False

            if (i + 1) % config.grad_accum_steps == 1:
                self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.use_amp):
                pred, pred_bboxes, attn_maps, head_cls_outs = self.model.forward_with_attn(img, gt_bboxes=bbox_target)

                true_lengths = bbox_mask.sum(dim=1).long()

                cls_loss = t.tensor(0.0, device=self.device)
                for h in range(config.num_heads):
                    valid_mask = (true_lengths > h).float()
                    if valid_mask.sum() > 0:
                        if use_cutmix:
                            head_loss_a = self.head_criteria[h](pred[h], label_a[:, h])
                            head_loss_b = self.head_criteria[h](pred[h], label_b[:, h])
                            head_loss = lam * head_loss_a + (1 - lam) * head_loss_b
                        else:
                            head_loss = self.head_criteria[h](pred[h], label[:, h])
                        cls_loss = cls_loss + (head_loss * valid_mask).sum() / valid_mask.sum()

                div_loss = attention_diversity_loss(attn_maps)
                ord_loss = spatial_ordering_loss(attn_maps, bbox_preds=pred_bboxes, bbox_mask=bbox_mask)
                attn_sup_loss = self.attn_supervision(attn_maps, bbox_target, bbox_mask)
                bbox_loss = t.tensor(0.0, device=self.device)
                valid_bbox_sum = (bbox_target * bbox_mask.unsqueeze(-1)).sum(dim=1)
                valid_bbox_count = bbox_mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean_bbox = valid_bbox_sum / valid_bbox_count
                for h in range(config.num_heads):
                    mask = bbox_mask[:, h]
                    if mask.sum() > 0:
                        if use_cutmix:
                            bbox_loss_a = F.smooth_l1_loss(
                                pred_bboxes[h][mask_a[:, h] > 0], bbox_a[:, h, :][mask_a[:, h] > 0])
                            bbox_loss_b = F.smooth_l1_loss(
                                pred_bboxes[h][mask_b[:, h] > 0], bbox_b[:, h, :][mask_b[:, h] > 0])
                            bbox_loss_h = lam * bbox_loss_a + (1 - lam) * bbox_loss_b
                        else:
                            bbox_loss_h = F.smooth_l1_loss(
                                pred_bboxes[h][mask > 0], bbox_target[:, h, :][mask > 0])
                        bbox_loss = bbox_loss + bbox_loss_h
                    else:
                        empty_mask = (mask == 0)
                        if empty_mask.sum() > 0:
                            bbox_loss = bbox_loss + F.smooth_l1_loss(
                                pred_bboxes[h][empty_mask], mean_bbox[empty_mask].detach()) * 0.3
                loss = (cls_loss + config.bbox_loss_weight * bbox_loss
                        + config.attn_diversity_weight * div_loss
                        + config.ordering_loss_weight * ord_loss
                        + config.attn_supervision_weight * attn_sup_loss)

                aux_loss = t.tensor(0.0, device=self.device)
                if len(head_cls_outs) > 0:
                    for h in range(config.num_heads):
                        valid_mask = (true_lengths > h).float()
                        if valid_mask.sum() > 0:
                            aux_loss_h = self.head_criteria[h](head_cls_outs[h], label[:, h])
                            aux_loss = aux_loss + (aux_loss_h * valid_mask).sum() / valid_mask.sum()
                loss = loss + config.aux_loss_weight * aux_loss
                loss = loss / config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(tbar):
                self.scaler.unscale_(self.optimizer)
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update(self.model)
            total_loss += loss.item()
            batch_time = time.time() - batch_start
            batch_start = time.time()

            joint_total += img.size(0)
            joint_corrects += _compute_joint_acc(pred, label, true_lengths, config.num_heads)
            c_corrects, c_total = _compute_char_acc(pred, label, true_lengths, config.num_heads)
            char_corrects += c_corrects
            total_chars += c_total

            tbar.set_description(
                'Epoch %d, loss: %.3f, joint: %.3f, char: %.3f' % (
                    epoch + 1, total_loss / (i + 1),
                    joint_corrects * 100 / max(joint_total, 1),
                    char_corrects * 100 / max(total_chars, 1)))

            if (i + 1) % config.print_interval == 0:
                acc_str = f'joint={joint_corrects * 100 / max(joint_total, 1):.2f}% char={char_corrects * 100 / max(total_chars, 1):.2f}%'
                self.logger.log_batch(epoch, i, len(self.train_loader),
                                      total_loss / (i + 1), self.optimizer.param_groups[0]['lr'],
                                      acc_str,
                                      cls_loss=cls_loss.item(), bbox_loss=bbox_loss.item(),
                                      div_loss=div_loss.item() if isinstance(div_loss, t.Tensor) else div_loss,
                                      ord_loss=ord_loss.item() if isinstance(ord_loss, t.Tensor) else ord_loss,
                                      attn_loss=attn_sup_loss.item() if isinstance(attn_sup_loss, t.Tensor) else attn_sup_loss,
                                      batch_time=batch_time)

        self._last_train_joint_acc = joint_corrects * 100 / max(joint_total, 1)
        self._last_train_char_acc = char_corrects * 100 / max(total_chars, 1)
        return self._last_train_joint_acc

    def _eval(self):
        if self.ema is not None:
            model = self.ema.to_device(self.device)
        else:
            model = self.model
        model.eval()
        eval_bs = config.eval_batch_size
        max_retries = 3
        for attempt in range(max_retries):
            char_corrects = 0
            total_chars = 0
            joint_corrects = 0
            joint_total = 0
            oom_hit = False
            with t.no_grad():
                tbar = tqdm(self.val_loader)
                for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
                    try:
                        img = img.to(self.device)
                        label = label.to(self.device)
                        bbox_mask = bbox_mask.to(self.device)
                        pred_cls, _ = model(img)

                        true_lengths = bbox_mask.sum(dim=1).long()
                        c_corrects, c_total = _compute_char_acc(pred_cls, label, true_lengths, config.num_heads)
                        char_corrects += c_corrects
                        total_chars += c_total

                        joint_corrects += _compute_joint_acc(pred_cls, label, true_lengths, config.num_heads)
                        joint_total += img.size(0)

                        tbar.set_description('Val Char: %.2f%% Joint: %.2f%%' % (
                            char_corrects * 100 / max(total_chars, 1),
                            joint_corrects * 100 / max(joint_total, 1)))

                        del img, label, pred_cls
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            t.cuda.empty_cache()
                            oom_hit = True
                            if eval_bs > 16:
                                eval_bs = max(eval_bs // 2, 16)
                                self.logger.logger.warning(
                                    f'[OOM-EVAL] Reducing eval_batch_size to {eval_bs} (attempt {attempt + 1})')
                                self.val_loader = self._make_loader(
                                self.val_set, batch_size=eval_bs,
                                shuffle=False, drop_last=False)
                            break
                        raise
            if oom_hit:
                continue
            break
        t.cuda.empty_cache()
        self.model.train()

        char_acc = char_corrects / max(total_chars, 1)
        joint_acc = joint_corrects / max(joint_total, 1)
        self._last_val_char_acc = char_acc
        self._last_val_joint_acc = joint_acc
        print(f'  Char Acc: {char_acc * 100:.2f}%  |  Joint Acc: {joint_acc * 100:.2f}%')

        return joint_acc

    def eval_detailed(self):
        if self.ema is not None:
            model = self.ema.to_device(self.device)
        else:
            model = self.model
        model.eval()
        head_corrects = [0] * config.num_heads
        head_totals = [0] * config.num_heads
        char_corrects = 0
        total_chars = 0
        joint_corrects = 0
        joint_total = 0
        with t.no_grad():
            for img, label, bbox_target, bbox_mask in tqdm(self.val_loader, desc='Detailed Eval'):
                img = img.to(self.device)
                label = label.to(self.device)
                bbox_mask = bbox_mask.to(self.device)
                pred_cls, _ = model(img)

                true_lengths = bbox_mask.sum(dim=1).long()

                for h in range(config.num_heads):
                    valid_mask = (true_lengths > h).float()
                    head_corrects[h] += ((pred_cls[h].argmax(1) == label[:, h]) * valid_mask).sum().item()
                    head_totals[h] += valid_mask.sum().item()

                c_corrects, c_total = _compute_char_acc(pred_cls, label, true_lengths, config.num_heads)
                char_corrects += c_corrects
                total_chars += c_total

                joint_corrects += _compute_joint_acc(pred_cls, label, true_lengths, config.num_heads)
                joint_total += img.size(0)

                del img, label, pred_cls

        for h in range(config.num_heads):
            if head_totals[h] > 0:
                print(f'  Head {h+1} Char Acc: {head_corrects[h] / head_totals[h] * 100:.2f}% ({head_corrects[h]}/{head_totals[h]})')

        char_acc = char_corrects / max(total_chars, 1)
        joint_acc = joint_corrects / max(joint_total, 1)
        print(f'  Overall Char Acc: {char_acc * 100:.2f}%  |  Joint Acc: {joint_acc * 100:.2f}%')

        t.cuda.empty_cache()
        self.model.train()

        return joint_acc

    def eval_tta(self):
        if self.ema is not None:
            model = self.ema.to_device(self.device)
        else:
            model = self.model
        model.eval()

        all_probs = [t.zeros(len(self.val_set), config.class_num, device='cpu') for _ in range(config.num_heads)]
        all_labels = t.zeros(len(self.val_set), config.num_heads, dtype=t.long, device='cpu')
        all_bbox_mask = t.zeros(len(self.val_set), config.num_heads, dtype=t.float, device='cpu')

        for tta_size in config.tta_sizes:
            val_set_tta = DigitsDataset(mode='val', aug=False,
                                        input_size=(tta_size, tta_size))
            val_loader_tta = self._make_loader(val_set_tta, batch_size=config.eval_batch_size,
                                               shuffle=False, drop_last=False)
            sample_idx = 0
            with t.no_grad():
                for img, label, _, bbox_mask in tqdm(val_loader_tta, desc=f'TTA size={tta_size}'):
                    img = img.to(self.device)
                    probs = model.forward_with_probs(img)
                    bs = img.size(0)
                    for h in range(config.num_heads):
                        all_probs[h][sample_idx:sample_idx + bs] += probs[h].cpu()
                    if tta_size == config.tta_sizes[0]:
                        all_labels[sample_idx:sample_idx + bs] = label.cpu()
                        all_bbox_mask[sample_idx:sample_idx + bs] = bbox_mask.cpu()
                    sample_idx += bs
                    del img, probs
                t.cuda.empty_cache()

        pred_heads = [all_probs[h].argmax(1) for h in range(config.num_heads)]
        true_lengths = all_bbox_mask.sum(dim=1).long()

        char_corrects, total_chars = _compute_char_acc(pred_heads, all_labels, true_lengths, config.num_heads)
        char_acc = char_corrects / max(total_chars, 1)

        joint_corrects = _compute_joint_acc(pred_heads, all_labels, true_lengths, config.num_heads)
        joint_acc = joint_corrects / len(self.val_set)

        print(f'TTA Char Acc: {char_acc * 100:.2f}%  |  Joint Acc: {joint_acc * 100:.2f}%')
        self.model.train()
        return joint_acc
