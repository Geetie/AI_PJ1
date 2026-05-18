import os
import gc
import json
import random
import time
import threading
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
from utils.compile_utils import (
    try_compile_model, warmup_model, get_raw_model, configure_dynamo_cache,
    CompileLogger, configure_compile_cache
)


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

        self._gpu_sanity_check()

        self.model = create_model(self._model_type).to(self.device)

        if config.pretrained is not None:
            ckpt = t.load(config.pretrained, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
            if 'model_type' in ckpt:
                self._model_type = ckpt['model_type']
            print(f'Load model from {config.pretrained}')

        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        self._compile_logger = CompileLogger.get_instance()
        if config.use_torch_compile and t.cuda.is_available():
            configure_dynamo_cache()
            configure_compile_cache()
            self._compile_logger.log_compile_config({
                'use_torch_compile': True,
                'compile_mode': config.compile_mode,
                'compile_dynamic': config.compile_dynamic,
                'compile_fullgraph': config.compile_fullgraph,
                'model_type': self._model_type,
                'batch_size': config.batch_size,
                'input_size': f'{config.input_height}x{config.input_width}',
            })
            with self._compile_logger.phase('compile_multihead'):
                self.model, compile_ok = try_compile_model(
                    self.model, mode=config.compile_mode,
                    dynamic=config.compile_dynamic, fullgraph=config.compile_fullgraph)
            if not compile_ok:
                config.use_torch_compile = False
        else:
            self._compile_logger.log_compile_config({
                'use_torch_compile': False,
                'reason': 'no_cuda' if not t.cuda.is_available() else 'disabled',
            })

        self.attn_supervision = AttentionSupervisionLoss()

        class_weights = self._compute_class_weights()

        self.head_criteria = nn.ModuleList()
        for h in range(config.num_heads):
            self.head_criteria.append(LabelSmoothEntropy(smooth=config.smooth, class_weights=class_weights))

        backbone_params = list(get_raw_model(self.model).backbone.parameters())
        other_params = [p for n, p in get_raw_model(self.model).named_parameters()
                        if not n.startswith('backbone.')]
        self.optimizer = self._setup_optimizer(backbone_params, other_params)
        self.lr_scheduler = self._setup_scheduler()
        self.scaler = self._setup_scaler()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_init(self._model_type, self.device, total_params, trainable_params)

        if config.pretrained is not None:
            if 'best_acc' in ckpt:
                self.best_acc = ckpt['best_acc']
            if 'epoch' in ckpt:
                config.start_epoch = ckpt['epoch']
                self._current_epoch = ckpt['epoch']
            if 'train_log' in ckpt:
                self.train_log = ckpt['train_log']
            print(f'Restored best_acc: {self.best_acc * 100:.2f}%')
            print('Warning: Optimizer and scheduler NOT restored. Using new config.')

        self._gpu_warmup()

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

        self._diagnose_dataloader()

    def _diagnose_dataloader(self):
        print('[DIAG] Testing DataLoader first batch load...')
        diag_start = time.time()
        try:
            sample_batch = next(iter(self.train_loader))
            diag_time = time.time() - diag_start
            img_shape = sample_batch[0].shape
            print(f'[DIAG] First batch loaded in {diag_time:.2f}s, img_shape={img_shape}')
            if diag_time > 30:
                print(f'[DIAG] WARNING: Data loading is very slow ({diag_time:.1f}s). '
                      f'Consider reducing num_workers or checking disk I/O.')
            del sample_batch
        except Exception as e:
            diag_time = time.time() - diag_start
            print(f'[DIAG] DataLoader test FAILED after {diag_time:.2f}s: {e}')
            if config.num_workers > 0:
                print(f'[DIAG] Falling back to num_workers=0 due to DataLoader failure')
                config.num_workers = 0
                config.prefetch_factor = None
                config.persistent_workers = False
                config.multiprocessing_context = None
                self._original_num_workers = config.num_workers
                self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                                      shuffle=True, drop_last=True)
                if self.val_loader is not None:
                    self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                        shuffle=False, drop_last=False)

    def _gpu_warmup(self):
        if not t.cuda.is_available():
            return
        warmup_bs = config.batch_size
        compile_info = " (torch.compile)" if config.use_torch_compile else ""
        print(f'[WARMUP] Starting GPU warmup with bs={warmup_bs}{compile_info}...')
        if config.use_torch_compile:
            print(f'[WARMUP] Kernel compilation in progress (may take 5-15 min on first run)')
            print(f'[WARMUP] Subsequent runs will use cached kernels and be much faster')

        heartbeat_stop = threading.Event()
        warmup_start = time.time()

        def _heartbeat():
            elapsed = 0
            while not heartbeat_stop.wait(30):
                elapsed += 30
                print(f'[WARMUP] Still compiling kernels... ({elapsed}s elapsed)')

        ht = threading.Thread(target=_heartbeat, daemon=True)
        ht.start()

        try:
            with self._compile_logger.phase('warmup_inference'):
                dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                print(f'[WARMUP] Running inference pass (bs={warmup_bs})...')
                with t.no_grad(), autocast(self.device.type, enabled=self.use_amp):
                    _ = self.model(dummy)
                t.cuda.synchronize()
                del dummy
                t.cuda.empty_cache()

            warmup_time = time.time() - warmup_start
            self._compile_logger.log_warmup_summary(warmup_time, 1, 1)
            print(f'[WARMUP] Primary warmup completed in {warmup_time:.1f}s (inference with bs={warmup_bs})')

            if config.use_torch_compile:
                with self._compile_logger.phase('warmup_tta_shapes'):
                    self._warmup_tta_shapes(warmup_bs)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'[WARMUP] OOM with bs={warmup_bs}, trying bs={warmup_bs // 4}...')
                t.cuda.empty_cache()
                try:
                    warmup_bs = warmup_bs // 4
                    dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                    with t.no_grad():
                        _ = self.model(dummy)
                    t.cuda.synchronize()
                    del dummy
                    t.cuda.empty_cache()
                    warmup_time = time.time() - warmup_start
                    self._compile_logger.log_warmup_summary(warmup_time, 1, 1)
                    print(f'[WARMUP] Fallback warmup completed in {warmup_time:.1f}s (inference with bs={warmup_bs})')
                except Exception as e2:
                    print(f'[WARMUP] Fallback warmup also failed: {e2}')
                    t.cuda.empty_cache()
            else:
                print(f'[WARMUP] GPU warmup failed: {e}')
                t.cuda.empty_cache()
        except Exception as e:
            err_str = str(e).lower()
            if config.use_torch_compile and ('compile' in err_str or 'triton' in err_str or 'inductor' in err_str):
                print(f'[WARMUP] torch.compile failed during warmup: {e}')
                print(f'[WARMUP] Disabling torch.compile and falling back to eager mode')
                self._compile_logger.logger.warning(
                    f'[WARMUP] torch.compile failed, falling back to eager: {e}')
                config.use_torch_compile = False
                self.model = self._get_raw_model()
                if self.ema is not None:
                    from utils.compile_utils import get_raw_model as _grm
                    self.ema = ModelEMA(_grm(self.model), decay=config.ema_decay)
                t.cuda.empty_cache()
                try:
                    dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                    with t.no_grad():
                        _ = self.model(dummy)
                    t.cuda.synchronize()
                    del dummy
                    t.cuda.empty_cache()
                    warmup_time = time.time() - warmup_start
                    print(f'[WARMUP] Eager mode warmup completed in {warmup_time:.1f}s')
                except Exception as e2:
                    print(f'[WARMUP] Eager mode warmup also failed: {e2}')
                    t.cuda.empty_cache()
            else:
                print(f'[WARMUP] GPU warmup failed: {e}')
                t.cuda.empty_cache()
        finally:
            heartbeat_stop.set()
            ht.join(timeout=5)

        self._compile_logger.log_dynamo_stats()

    def _warmup_tta_shapes(self, warmup_bs):
        if not config.use_torch_compile or not t.cuda.is_available():
            return
        tta_shapes = [(s, s) for s in config.tta_sizes]
        eval_bs = min(config.eval_batch_size, warmup_bs)
        print(f'[WARMUP] Warming up TTA shapes: {tta_shapes} with bs={eval_bs}...')
        tta_start = time.time()
        for h, w in tta_shapes:
            try:
                dummy = t.randn(eval_bs, 3, h, w, device=self.device)
                with t.no_grad(), autocast(self.device.type, enabled=self.use_amp):
                    _ = self.model(dummy)
                t.cuda.synchronize()
                del dummy
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    t.cuda.empty_cache()
                    try:
                        small_bs = max(eval_bs // 4, 4)
                        dummy = t.randn(small_bs, 3, h, w, device=self.device)
                        with t.no_grad():
                            _ = self.model(dummy)
                        t.cuda.synchronize()
                        del dummy
                    except Exception:
                        pass
                    t.cuda.empty_cache()
                else:
                    print(f'[WARMUP] TTA shape ({h},{w}) warmup failed: {e}')
        t.cuda.empty_cache()
        tta_time = time.time() - tta_start
        print(f'[WARMUP] TTA shape warmup completed in {tta_time:.1f}s')

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
        class_weights[10] = 0.0
        active = class_weights[:10]
        class_weights[:10] = active * config.class_num / active.sum()
        class_weights = class_weights.to(self.device)
        print(f'Computed class weights from JSON: {class_weights.cpu().numpy()}')
        print(f'   Class 10 (empty) weight: {class_weights[10].item():.3f} (excluded from loss)')
        return class_weights

    def _make_loader(self, dataset, batch_size, shuffle=False, drop_last=False, collate_fn=None):
        kwargs = dict(
            batch_size=batch_size, shuffle=shuffle,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            drop_last=drop_last,
        )
        if config.num_workers > 0:
            kwargs['prefetch_factor'] = config.prefetch_factor
            kwargs['persistent_workers'] = config.persistent_workers
        if config.multiprocessing_context is not None and config.num_workers > 0:
            kwargs['multiprocessing_context'] = config.multiprocessing_context
        if collate_fn is not None:
            kwargs['collate_fn'] = collate_fn
        print(f'[DataLoader] batch={batch_size}, workers={config.num_workers}, '
              f'pin_mem={config.pin_memory}, ctx={config.multiprocessing_context}, '
              f'persistent={config.persistent_workers}, dataset={len(dataset)}')
        return DataLoader(dataset, **kwargs)

    def _pre_epoch_hook(self, epoch):
        raw_model = self._get_raw_model()
        if hasattr(raw_model, 'set_roi_gt_prob'):
            if epoch < config.warmup_epochs:
                raw_model.set_roi_gt_prob(1.0)
            else:
                decay_end = int(config.epoches * 0.6)
                if epoch >= decay_end:
                    raw_model.set_roi_gt_prob(0.0)
                else:
                    progress = (epoch - config.warmup_epochs) / max(decay_end - config.warmup_epochs, 1)
                    raw_model.set_roi_gt_prob(1.0 - progress)

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
        self.model.train()
        first_batch = True
        data_load_time = 0.0

        print(f'[EPOCH {epoch+1}] Waiting for first batch from DataLoader...')
        t_load_start = time.time()
        tbar = tqdm(self.train_loader)
        data_load_time = time.time() - t_load_start
        print(f'[EPOCH {epoch+1}] DataLoader iterator created in {data_load_time:.2f}s')

        for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
            t_data = time.time()
            if first_batch:
                print(f'[BATCH0] data load time: {t_data - batch_start:.2f}s')

            t0 = time.time()
            img = img.to(self.device)
            label = label.to(self.device)
            bbox_target = bbox_target.to(self.device)
            bbox_mask = bbox_mask.to(self.device)
            t_gpu = time.time()

            t1 = time.time()

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
                if first_batch:
                    t.cuda.synchronize()
                    print(f'[BATCH0] forward: {time.time()-t1:.2f}s, gpu_transfer: {t_gpu-t0:.2f}s, img={img.shape}')

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
            if first_batch:
                t.cuda.synchronize()
                t_backward = time.time()
                print(f'[BATCH0] backward+step: {t_backward-t1:.2f}s, loss={loss.item():.4f}')
                print(f'[BATCH0] TOTAL batch time: {t_backward-batch_start:.2f}s '
                      f'(data_load={t_data-batch_start:.2f}s gpu_transfer={t_gpu-t0:.2f}s '
                      f'forward={t1-t_gpu:.2f}s backward={t_backward-t1:.2f}s)')
                first_batch = False
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
            model = self._get_raw_model()
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
            model = self._get_raw_model()
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
            model = self._get_raw_model()
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
