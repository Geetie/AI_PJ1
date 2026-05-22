import os
import gc
import json
import random
import time
import threading
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config, data_dir
from data.dataset import DigitsDataset
from models import create_model
from losses.classification import LabelSmoothEntropy, FocalLoss
from losses.attention import AttentionSupervisionLoss, attention_diversity_loss, spatial_ordering_loss
from losses.augmentation import cutmix_data
from trainer.base import BaseTrainer, ModelEMA, _load_state_dict_compat
from utils.compile_utils import (
    try_compile_model, warmup_model, get_raw_model, configure_dynamo_cache,
    CompileLogger, configure_compile_cache
)
from utils.seed import make_epoch_generator, set_epoch_seed


def _compute_joint_acc(pred_heads, labels, true_lengths, num_heads):
    head_correct = [(pred_heads[h].argmax(1) == labels[:, h]) for h in range(num_heads)]
    temp = t.stack(head_correct, dim=1)
    valid_head_mask = t.stack([(true_lengths > h) for h in range(num_heads)], dim=1)
    empty_correct = (pred_heads[h].argmax(1) == 10 for h in range(num_heads))
    empty_correct = t.stack(list(empty_correct), dim=1)
    correct_mask = t.where(valid_head_mask, temp, empty_correct)
    return t.all(correct_mask, dim=1).sum().item()


def _compute_char_acc(pred_heads, labels, true_lengths, num_heads):
    corrects = 0
    total_chars = 0
    digit_corrects = 0
    digit_total = 0
    for h in range(num_heads):
        valid_mask = (true_lengths > h)
        head_pred = pred_heads[h].argmax(1)
        if valid_mask.sum() > 0:
            head_correct = (head_pred == labels[:, h]) * valid_mask
            corrects += head_correct.sum().item()
            digit_corrects += head_correct.sum().item()
        empty_mask = ~valid_mask
        if empty_mask.sum() > 0:
            corrects += ((head_pred == 10) * empty_mask).sum().item()
        total_chars += len(labels)
        digit_total += valid_mask.sum().item()
    return corrects, total_chars, digit_corrects, digit_total


def _apply_length_mask(pred_cls, length_logits, num_heads):
    pred_length = length_logits.argmax(dim=1)
    length_conf = F.softmax(length_logits, dim=1).max(dim=1)[0]
    for h in range(num_heads):
        mask = (pred_length <= h) & (length_conf >= 0.3)
        if mask.any():
            pred_cls[h][mask, 10] = pred_cls[h][mask].amax(dim=1) + 100.0


class MultiHeadTrainer(BaseTrainer):
    _checkpoint_prefix = 'resnet101'
    _eval_label = 'Evaluation'

    def __init__(self, val=True, model_type=None):
        super().__init__()
        self.logger.logger.info(f'Using device: {self.device}')
        self._model_type = model_type or config.model_type

        self._gpu_sanity_check()

        self.model = create_model(self._model_type).to(self.device)

        self._bn_protection = None
        raw_model_for_bn = self.model
        if hasattr(raw_model_for_bn, 'setup_bn_protection'):
            self._bn_protection = raw_model_for_bn.setup_bn_protection(
                warmup_epochs=config.warmup_epochs,
                freeze_epochs=0,
                bn_grad_max_norm=1.0,
                auto_fix=True,
            )
            self.logger.logger.info('[BN-PROT] BN protection installed: grad_clip=1.0, warmup=%d epochs',
                                    config.warmup_epochs)

        self._loaded_from_checkpoint = False
        self._optimizer_state_lost = False
        self._nan_skip_count = 0
        self._nan_skip_threshold = 3
        self._nan_lr_reduced = False
        
        self._loss_weight_history = []
        self._grad_clip_max_norm = config.grad_clip_max_norm
        self._last_train_joint_acc = 0.0
        self._last_train_char_acc = 0.0
        self._last_train_digit_acc = 0.0
        self._last_train_raw_joint_acc = 0.0
        self._last_epoch_avg_loss = float('inf')
        self._last_val_joint_acc = 0.0
        self._last_val_char_acc = 0.0
        self._last_val_digit_acc = 0.0
        self._last_val_raw_joint_acc = 0.0

        ckpt = None
        if config.pretrained is not None:
            ckpt = t.load(config.pretrained, map_location=self.device, weights_only=False)
            self._loaded_from_checkpoint = True
            if config.resume_weights_only:
                _, skipped = _load_state_dict_compat(self.model, ckpt['model'])
                if skipped:
                    self.logger.logger.info(f'[CKPT] resume_weights_only: skipped {len(skipped)} shape-mismatched keys')
            elif 'train_model' in ckpt:
                _, skipped = _load_state_dict_compat(self.model, ckpt['train_model'])
                self.logger.logger.info('[CKPT] Loaded train_model weights (not EMA) for continued training')
                if skipped:
                    self.logger.logger.info(f'[CKPT] Skipped {len(skipped)} shape-mismatched keys in train_model')
            else:
                _, skipped = _load_state_dict_compat(self.model, ckpt['model'])
                if skipped:
                    self.logger.logger.info(f'[CKPT] Skipped {len(skipped)} shape-mismatched keys')
            if 'model_type' in ckpt:
                self._model_type = ckpt['model_type']
            if config.resume_weights_only:
                self.logger.logger.info(f'[CKPT] resume_weights_only=True: loading model weights only, '
                                       f'using config optimizer_type={config.optimizer_type} '
                                       f'scheduler_type={config.scheduler_type}')
            else:
                if 'optimizer_type' in ckpt:
                    ckpt_opt_type = ckpt['optimizer_type']
                    if ckpt_opt_type != config.optimizer_type:
                        self.logger.logger.info(f'[CKPT] Checkpoint optimizer_type={ckpt_opt_type} '
                                               f'overrides config optimizer_type={config.optimizer_type}')
                        config.optimizer_type = ckpt_opt_type
                elif 'opt' in ckpt:
                    opt_keys = set(ckpt['opt'].get('param_groups', [{}])[0].keys())
                    if 'amsgrad' in opt_keys or 'max_lr' in opt_keys:
                        config.optimizer_type = 'adamw'
                        self.logger.logger.info('[CKPT] Detected AdamW optimizer from checkpoint state_dict')
                    else:
                        config.optimizer_type = 'sgd'
                        self.logger.logger.info('[CKPT] Detected SGD optimizer from checkpoint state_dict')
                if 'scheduler_type' in ckpt:
                    ckpt_sched_type = ckpt['scheduler_type']
                    if ckpt_sched_type != config.scheduler_type:
                        self.logger.logger.info(f'[CKPT] Checkpoint scheduler_type={ckpt_sched_type} '
                                               f'but keeping config scheduler_type={config.scheduler_type}')
            self.logger.logger.info(f'Load model from {config.pretrained}')

        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        if ckpt is not None and not config.resume_weights_only:
            if 'model' in ckpt:
                try:
                    _, skipped = _load_state_dict_compat(self.ema.ema, ckpt['model'])
                    self.logger.logger.info('[CKPT] Restored EMA shadow model from checkpoint')
                    if skipped:
                        self.logger.logger.info(f'[CKPT] EMA shadow: skipped {len(skipped)} shape-mismatched keys')
                except Exception as e:
                    self.logger.logger.warning(f'[CKPT] Failed to restore EMA shadow model: {e}')
            if 'best_checkpoint_path' in ckpt:
                self.best_checkpoint_path = ckpt['best_checkpoint_path']
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
        length_weights = self._compute_length_weights()
        self.length_criterion = FocalLoss(gamma=2.0, weight=length_weights)

        class_weights = self._compute_class_weights()

        self.head_criteria = nn.ModuleList()
        for h in range(config.num_heads):
            self.head_criteria.append(LabelSmoothEntropy(smooth=config.smooth, class_weights=class_weights, size_average='none'))

        backbone_params = list(get_raw_model(self.model).backbone.parameters())
        other_params = [p for n, p in get_raw_model(self.model).named_parameters()
                        if not n.startswith('backbone.')]
        self.optimizer = self._setup_optimizer(backbone_params, other_params)
        self.lr_scheduler = self._setup_scheduler()
        self.scaler = self._setup_scaler()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_init(self._model_type, self.device, total_params, trainable_params)

        if ckpt is not None:
            if 'best_acc' in ckpt:
                self.best_acc = ckpt['best_acc']
            import re
            fname_acc = re.search(r'acc-([\d.]+)\.pth', config.pretrained)
            if fname_acc:
                fname_acc_val = float(fname_acc.group(1)) / 100.0
                if fname_acc_val > self.best_acc:
                    self.logger.logger.warning(f'Checkpoint filename acc={fname_acc_val*100:.2f}% > '
                                              f'stored best_acc={self.best_acc*100:.2f}%. '
                                              f'Correcting to filename value.')
                    self.best_acc = fname_acc_val
            if 'train_log' in ckpt:
                self.train_log = ckpt['train_log']
            if config.resume_weights_only:
                config.start_epoch = 0
                self._current_epoch = 0
                self.patience_counter = 0
                self.best_acc = 0
                self.train_log = []
                if 'best_checkpoint_path' in ckpt:
                    self.best_checkpoint_path = ckpt['best_checkpoint_path']
                self.logger.logger.info(f'[CKPT] resume_weights_only: best_acc reset to 0%, '
                                       f'starting from epoch 1 with fresh '
                                       f'{config.optimizer_type} optimizer and {config.scheduler_type} scheduler')
            else:
                if 'epoch' in ckpt:
                    config.start_epoch = ckpt['epoch']
                    self._current_epoch = ckpt['epoch']
                if 'opt' in ckpt:
                    if not self._restore_optimizer_robust(ckpt):
                        self._optimizer_state_lost = True
                if 'lr_scheduler' in ckpt:
                    try:
                        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                        self.logger.logger.info('Restored lr_scheduler state from checkpoint')
                    except Exception as e:
                        self.logger.logger.warning(f'Failed to restore lr_scheduler: {e}. Using new scheduler.')
                if 'scaler' in ckpt:
                    try:
                        self.scaler.load_state_dict(ckpt['scaler'])
                        self.logger.logger.info('Restored scaler state from checkpoint')
                    except Exception as e:
                        self.logger.logger.warning(f'Failed to restore scaler: {e}. Using new scaler.')
                if 'patience_counter' in ckpt:
                    self.patience_counter = ckpt['patience_counter']
                if 'best_checkpoint_path' in ckpt:
                    self.best_checkpoint_path = ckpt['best_checkpoint_path']
                self.logger.logger.info(f'Restored best_acc: {self.best_acc * 100:.2f}%, '
                                       f'start_epoch: {config.start_epoch}, '
                                       f'patience: {self.patience_counter}/{config.early_stopping_patience}')

        if self._optimizer_state_lost:
            self._post_reset_warmup_epochs = 10
            self._warmup_target_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            self._warmup_start_lrs = [pg['lr'] * 0.05 for pg in self.optimizer.param_groups]
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.05
            self.logger.logger.info(
                f'[CKPT] Optimizer state lost, scheduling {self._post_reset_warmup_epochs}-epoch warmup '
                f'from LR={self.optimizer.param_groups[0]["lr"]:.8f}')

        self._gpu_warmup()

        self._base_seed = 42
        self._train_generator = make_epoch_generator(self._base_seed, epoch=0)

        self.train_set = DigitsDataset(mode='train', aug=True,
                                       input_size=(config.input_height, config.input_width))
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True,
                                              generator=self._train_generator)
        if val:
            self.val_set = DigitsDataset(mode='val', aug=False,
                                         input_size=(config.input_height, config.input_width))
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False)
        else:
            self.val_set = None
            self.val_loader = None

        self._diagnose_dataloader()

    def _compute_adaptive_loss_weights(self, epoch):
        warmup = config.warmup_epochs

        if not config.gradient_balance.get('enabled', True):
            if epoch < warmup // 2:
                return {'cls': 1.0, 'bbox': 0.0, 'div': 0.0,
                        'ord': 0.0, 'attn': 0.0, 'aux': 0.5, 'length': 0.0}
            elif epoch < warmup:
                progress = (epoch - warmup // 2) / (warmup // 2)
                return {'cls': 1.0, 'bbox': progress, 'div': 0.0,
                        'ord': 0.0, 'attn': 0.0, 'aux': 0.5, 'length': progress * 0.5}
            else:
                progress = min((epoch - warmup) / 30.0, 1.0)
                return {
                    'cls': 1.0,
                    'bbox': 1.0,
                    'div': 0.5 + 0.5 * progress,
                    'ord': 0.3 + 0.7 * progress,
                    'attn': 0.3 + 0.7 * progress,
                    'aux': 0.5 + 0.5 * progress,
                    'length': 0.5 + 0.5 * progress,
                }

        bbox_compensation = config.gradient_balance.get('bbox_norm_factor', 50.0)
        length_compensation = config.gradient_balance.get('length_norm_factor', 200.0)

        if epoch < warmup // 2:
            return {'cls': 1.0, 'bbox': 0.0, 'div': 0.0,
                    'ord': 0.0, 'attn': 0.0, 'aux': 0.5, 'length': 0.0}
        elif epoch < warmup:
            progress = (epoch - warmup // 2) / (warmup // 2)
            return {
                'cls': 1.0,
                'bbox': bbox_compensation * progress * 0.1,
                'div': 0.0,
                'ord': 0.0,
                'attn': 0.0,
                'aux': 0.5,
                'length': length_compensation * progress * 0.05,
            }

        transition_epochs = 30
        progress = min((epoch - warmup) / transition_epochs, 1.0)
        cos_factor = 0.5 * (1 + t.cos(t.tensor(progress * t.pi)))

        return {
            'cls': 1.0,
            'bbox': bbox_compensation * (0.3 + 0.7 * cos_factor),
            'div': 0.5 + 0.5 * cos_factor,
            'ord': bbox_compensation * (0.2 + 0.8 * cos_factor),
            'attn': bbox_compensation * (0.2 + 0.8 * cos_factor),
            'aux': 0.3 + 0.7 * cos_factor,
            'length': length_compensation * (0.2 + 0.8 * cos_factor),
        }

    def _diagnose_dataloader(self):
        self.logger.logger.info('[DIAG] Dataset sizes: train=%d, val=%s',
                                len(self.train_set),
                                len(self.val_set) if self.val_set is not None else 'N/A')
        self.logger.logger.info('[DIAG] DataLoader config: batch=%d, workers=%d, persistent=%s',
                                config.batch_size, config.num_workers, config.persistent_workers)

    def _gpu_warmup(self):
        if not t.cuda.is_available():
            return
        warmup_bs = config.batch_size
        compile_info = " (torch.compile)" if config.use_torch_compile else ""
        self._compile_logger.logger.info(f'[WARMUP] Starting GPU warmup with bs={warmup_bs}{compile_info}...')
        if config.use_torch_compile:
            self._compile_logger.logger.info('[WARMUP] Kernel compilation in progress (may take 5-15 min on first run)')
            self._compile_logger.logger.info('[WARMUP] Subsequent runs will use cached kernels and be much faster')

        heartbeat_stop = threading.Event()
        warmup_start = time.time()

        def _heartbeat():
            elapsed = 0
            while not heartbeat_stop.wait(30):
                elapsed += 30
                self._compile_logger.logger.info(f'[WARMUP] Still compiling kernels... ({elapsed}s elapsed)')

        ht = threading.Thread(target=_heartbeat, daemon=True)
        ht.start()

        try:
            with self._compile_logger.phase('warmup_inference'):
                dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                self._compile_logger.logger.info(f'[WARMUP] Running inference pass (bs={warmup_bs})...')
                with t.no_grad(), autocast(self.device.type, enabled=self.use_amp, dtype=t.bfloat16 if self.use_bf16 else t.float16):
                    _ = self.model(dummy)
                t.cuda.synchronize()
                del dummy
                t.cuda.empty_cache()

            warmup_time = time.time() - warmup_start
            self._compile_logger.log_warmup_summary(warmup_time, 1, 1)
            self._compile_logger.logger.info(f'[WARMUP] Primary warmup completed in {warmup_time:.1f}s (inference with bs={warmup_bs})')

            if config.use_torch_compile:
                with self._compile_logger.phase('warmup_tta_shapes'):
                    self._warmup_tta_shapes(warmup_bs)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self._compile_logger.logger.warning(f'[WARMUP] OOM with bs={warmup_bs}, trying bs={warmup_bs // 4}...')
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
                    self._compile_logger.logger.info(f'[WARMUP] Fallback warmup completed in {warmup_time:.1f}s (inference with bs={warmup_bs})')
                except Exception as e2:
                    self._compile_logger.logger.error(f'[WARMUP] Fallback warmup also failed: {e2}')
                    t.cuda.empty_cache()
            else:
                self._compile_logger.logger.error(f'[WARMUP] GPU warmup failed: {e}')
                t.cuda.empty_cache()
        except Exception as e:
            err_str = str(e).lower()
            if config.use_torch_compile and ('compile' in err_str or 'triton' in err_str or 'inductor' in err_str):
                self._compile_logger.logger.error(f'[WARMUP] torch.compile failed during warmup: {e}')
                self._compile_logger.logger.warning('[WARMUP] Disabling torch.compile and falling back to eager mode')
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
                    self._compile_logger.logger.info(f'[WARMUP] Eager mode warmup completed in {warmup_time:.1f}s')
                except Exception as e2:
                    self._compile_logger.logger.error(f'[WARMUP] Eager mode warmup also failed: {e2}')
                    t.cuda.empty_cache()
            else:
                self._compile_logger.logger.error(f'[WARMUP] GPU warmup failed: {e}')
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
        self._compile_logger.logger.info(f'[WARMUP] Warming up TTA shapes: {tta_shapes} with bs={eval_bs}...')
        tta_start = time.time()
        for h, w in tta_shapes:
            try:
                dummy = t.randn(eval_bs, 3, h, w, device=self.device)
                with t.no_grad(), autocast(self.device.type, enabled=self.use_amp, dtype=t.bfloat16 if self.use_bf16 else t.float16):
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
                    self._compile_logger.logger.error(f'[WARMUP] TTA shape ({h},{w}) warmup failed: {e}')
        t.cuda.empty_cache()
        tta_time = time.time() - tta_start
        self._compile_logger.logger.info(f'[WARMUP] TTA shape warmup completed in {tta_time:.1f}s')

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
        total = class_counts.sum()
        class_weights = total / (class_counts * config.class_num + 1e-6)
        class_weights = class_weights / class_weights.mean()
        class_weights[10] = max(class_weights[10].item(), 0.3)
        class_weights = class_weights.to(self.device)
        self.logger.logger.info(f'Computed class weights from JSON: {class_weights.cpu().numpy()}')
        self.logger.logger.info(f'   Class 10 (empty) weight: {class_weights[10].item():.4f} (floored at 0.3)')
        return class_weights

    def _compute_length_weights(self):
        length_counts = t.zeros(config.num_heads + 1)
        with open(data_dir['train_label'], 'r') as f:
            labels = json.load(f)
        for img_name, mark in labels.items():
            length = min(len(mark['label']), config.num_heads)
            length_counts[length] += 1
        total = length_counts.sum()
        
        mask = length_counts > 0
        length_weights = t.zeros_like(length_counts)
        
        if mask.any():
            length_weights[mask] = total / (length_counts[mask] * (config.num_heads + 1))
            length_weights[mask] = length_weights[mask] / length_weights[mask].mean()
        
        length_weights = length_weights.to(self.device)
        self.logger.logger.info(f'Computed length weights from JSON: {length_weights.cpu().numpy()}')
        for l in range(config.num_heads + 1):
            count = int(length_counts[l].item())
            pct = count / total.item() * 100
            self.logger.logger.info(f'   Length {l}: count={count} ({pct:.1f}%), weight={length_weights[l].item():.4f}')
        return length_weights

    @staticmethod
    def _worker_init_fn(worker_id):
        worker_seed = t.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _make_loader(self, dataset, batch_size, shuffle=False, drop_last=False, collate_fn=None, generator=None, persistent_override=None):
        kwargs = dict(
            batch_size=batch_size, shuffle=shuffle,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            drop_last=drop_last,
        )
        if shuffle and generator is not None:
            kwargs['generator'] = generator
        pw = persistent_override if persistent_override is not None else config.persistent_workers
        if config.num_workers > 0:
            kwargs['prefetch_factor'] = config.prefetch_factor
            kwargs['persistent_workers'] = pw
            kwargs['worker_init_fn'] = self._worker_init_fn
        if config.multiprocessing_context is not None and config.num_workers > 0:
            kwargs['multiprocessing_context'] = config.multiprocessing_context
        if collate_fn is not None:
            kwargs['collate_fn'] = collate_fn
        self.logger.logger.info(f'[DataLoader] batch={batch_size}, workers={config.num_workers}, '
              f'pin_mem={config.pin_memory}, ctx={config.multiprocessing_context}, '
              f'persistent={pw}, dataset={len(dataset)}, '
              f'generator={"epoch_seeded" if generator is not None else "global_rng"}')
        return DataLoader(dataset, **kwargs)

    def _pre_epoch_hook(self, epoch):
        epoch_seed = set_epoch_seed(self._base_seed, epoch)
        self._nan_lr_reduced = False
        self._nan_skip_count = 0

        if self._bn_protection is not None:
            momentum = self._bn_protection['momentum_scheduler'].step(epoch, config.epoches)
            self._bn_protection['safe_wrapper'].step(epoch)
            if epoch < 3 or epoch % 10 == 0:
                self.logger.logger.info('[BN-PROT] epoch=%d momentum=%.4f', epoch + 1, momentum)

        raw_model = self._get_raw_model()
        if hasattr(raw_model, 'set_roi_gt_prob'):
            if epoch < config.warmup_epochs:
                raw_model.set_roi_gt_prob(1.0)
            else:
                decay_end = int(config.epoches * config.roi_gt_decay_end_ratio)
                if epoch >= decay_end:
                    raw_model.set_roi_gt_prob(0.1)
                else:
                    progress = (epoch - config.warmup_epochs) / max(decay_end - config.warmup_epochs, 1)
                    import math
                    raw_model.set_roi_gt_prob(0.5 * (1 + math.cos(math.pi * progress)))

        self._cleanup_dataloader(self.train_loader)
        self._train_generator = make_epoch_generator(self._base_seed, epoch=epoch)
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True,
                                              generator=self._train_generator,
                                              persistent_override=False)
        self.logger.logger.info(
            f'[EPOCH-PRE] epoch={epoch+1} lr={self.optimizer.param_groups[0]["lr"]:.8f} '
            f'roi_gt_prob={raw_model.roi_gt_prob:.2f} seed={epoch_seed}')
        self.logger.logger.info(
            f'[EPOCH-PRE] DataLoader generator seed={epoch_seed}, ensuring unique random sequence for this epoch')

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
                                              shuffle=True, drop_last=True,
                                              generator=self._train_generator,
                                              persistent_override=False)
        if self.val_loader is not None:
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False)

    def _setup_grad_clipping(self):
        self._grad_clip_max_norm = config.grad_clip_max_norm
        self.logger.logger.info(f'[GRAD-CLIP] Using clip_grad_norm_ with max_grad_norm={self._grad_clip_max_norm}')

    def _update_grad_clip_hooks(self, max_grad_norm):
        self._grad_clip_max_norm = max_grad_norm

    def _do_grad_clip(self):
        """自适应梯度裁剪方法，在optimizer.step()前调用"""
        if not hasattr(self, '_grad_clip_max_norm'):
            return
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return
        grad_norm = t.norm(t.stack([g.norm() for g in grads]), 2)
        if grad_norm > self._grad_clip_max_norm:
            t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_max_norm)

    def _train_epoch(self, epoch):
        if not hasattr(self, '_grad_clip_max_norm'):
            self._setup_grad_clipping()
        joint_corrects = 0
        joint_total = 0
        raw_joint_corrects = 0
        char_corrects = 0
        total_chars = 0
        digit_corrects = 0
        digit_total = 0
        total_loss = 0.0
        batch_start = time.time()
        self.model.train()
        first_batch = True
        data_load_time = 0.0
        scaler_skip_count = 0

        print(f'[EPOCH {epoch+1}] Waiting for first batch from DataLoader...')
        t_load_start = time.time()
        tbar = tqdm(self.train_loader)
        data_load_time = time.time() - t_load_start
        print(f'[EPOCH {epoch+1}] DataLoader iterator created in {data_load_time:.2f}s')

        for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
            _batch_scaler_skipped = False
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

            if config.cutmix_prob > 0 and t.rand(1).item() < config.cutmix_prob and config.cutmix_alpha > 0:
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

            if i % config.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.use_amp, dtype=t.bfloat16 if self.use_bf16 else t.float16):
                pred, pred_bboxes, attn_maps, head_cls_outs, length_logits = self.model.forward_with_attn(img, gt_bboxes=bbox_target)
                if first_batch:
                    t.cuda.synchronize()
                    print(f'[BATCH0] forward: {time.time()-t1:.2f}s, gpu_transfer: {t_gpu-t0:.2f}s, img={img.shape}')

                true_lengths = bbox_mask.sum(dim=1).long()

                cls_loss = t.tensor(0.0, device=self.device)
                for h in range(config.num_heads):
                    if use_cutmix:
                        head_loss_a = self.head_criteria[h](pred[h], label_a[:, h])
                        head_loss_b = self.head_criteria[h](pred[h], label_b[:, h])
                        head_loss = lam * head_loss_a + (1 - lam) * head_loss_b
                    else:
                        head_loss = self.head_criteria[h](pred[h], label[:, h])
                    cls_loss = cls_loss + head_loss.mean()

                if use_cutmix:
                    div_loss = attention_diversity_loss(attn_maps)
                    attn_sup_loss = self.attn_supervision(attn_maps, bbox_a, mask_a)
                    ord_loss = spatial_ordering_loss(attn_maps, bbox_preds=pred_bboxes, bbox_mask=mask_a)
                    bbox_loss = t.tensor(0.0, device=self.device)
                    if lam > 0:
                        mixed_bbox_target = lam * bbox_a + (1 - lam) * bbox_b
                    else:
                        mixed_bbox_target = bbox_a
                    for h in range(config.num_heads):
                        mask_h = mask_a[:, h]
                        if mask_h.sum() > 0:
                            bbox_loss_h = F.smooth_l1_loss(
                                pred_bboxes[h][mask_h > 0], mixed_bbox_target[:, h, :][mask_h > 0])
                            bbox_loss = bbox_loss + bbox_loss_h
                else:
                    div_loss = attention_diversity_loss(attn_maps)
                    attn_sup_loss = self.attn_supervision(attn_maps, bbox_target, bbox_mask)
                    ord_loss = spatial_ordering_loss(attn_maps, bbox_preds=pred_bboxes, bbox_mask=bbox_mask)
                    bbox_loss = t.tensor(0.0, device=self.device)
                    valid_bbox_sum = (bbox_target * bbox_mask.unsqueeze(-1)).sum(dim=1)
                    valid_bbox_count = bbox_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    mean_bbox = valid_bbox_sum / valid_bbox_count
                    for h in range(config.num_heads):
                        mask = bbox_mask[:, h]
                        if mask.sum() > 0:
                            bbox_loss_h = F.smooth_l1_loss(
                                pred_bboxes[h][mask > 0], bbox_target[:, h, :][mask > 0])
                            bbox_loss = bbox_loss + bbox_loss_h
                        else:
                            empty_mask_h = (mask == 0)
                            if empty_mask_h.sum() > 0:
                                bbox_loss = bbox_loss + F.smooth_l1_loss(
                                    pred_bboxes[h][empty_mask_h], mean_bbox[empty_mask_h].detach()) * 0.3
                # 使用自适应损失权重，平衡各损失项的梯度贡献
                loss_weights = self._compute_adaptive_loss_weights(epoch)
                
                epoch_ratio = (epoch + 1) / config.epoches
                dynamic_ordering_weight = config.ordering_loss_weight * min(1.0, epoch_ratio * 2)
                dynamic_attn_weight = config.attn_supervision_weight * min(1.0, epoch_ratio * 1.5)
                
                cls_loss = cls_loss.clamp(max=10.0) if isinstance(cls_loss, t.Tensor) else cls_loss
                bbox_loss = bbox_loss.clamp(max=5.0) if isinstance(bbox_loss, t.Tensor) else bbox_loss
                div_loss = div_loss.clamp(max=5.0) if isinstance(div_loss, t.Tensor) else div_loss
                ord_loss_clamped = ord_loss.clamp(max=5.0) if isinstance(ord_loss, t.Tensor) else ord_loss
                attn_sup_loss = attn_sup_loss.clamp(max=5.0) if isinstance(attn_sup_loss, t.Tensor) else attn_sup_loss
                
                loss = (config.cls_loss_weight * cls_loss * loss_weights['cls'] + 
                        config.bbox_loss_weight * bbox_loss * loss_weights['bbox'] +
                        config.attn_diversity_weight * div_loss * loss_weights['div'] +
                        dynamic_ordering_weight * ord_loss_clamped * loss_weights['ord'] +
                        dynamic_attn_weight * attn_sup_loss * loss_weights['attn'])

                aux_loss = t.tensor(0.0, device=self.device)
                if len(head_cls_outs) > 0:
                    for h in range(config.num_heads):
                        if use_cutmix:
                            aux_loss_a = self.head_criteria[h](head_cls_outs[h], label_a[:, h])
                            aux_loss_b = self.head_criteria[h](head_cls_outs[h], label_b[:, h])
                            aux_loss_h = lam * aux_loss_a + (1 - lam) * aux_loss_b
                        else:
                            aux_loss_h = self.head_criteria[h](head_cls_outs[h], label[:, h])
                        aux_loss = aux_loss + aux_loss_h.mean()
                loss = loss + config.aux_loss_weight * aux_loss.clamp(max=10.0) * loss_weights['aux']
                length_target = true_lengths.clamp(max=config.num_heads)
                length_loss = self.length_criterion(length_logits, length_target)
                loss = loss + length_loss.clamp(max=10.0) * loss_weights['length']
                loss_for_accum = loss
                loss = loss / config.grad_accum_steps

                loss_value = loss.detach()
                if t.isnan(loss_value).any():
                    self.logger.logger.warning(
                        f'[TRAIN] Epoch {epoch+1} Batch {i+1}: NaN loss detected (loss={loss_value.item():.4f}), replacing with zero')
                    loss = loss.nan_to_num(0.0)
                elif t.isinf(loss_value).any():
                    self.logger.logger.warning(
                        f'[TRAIN] Epoch {epoch+1} Batch {i+1}: Inf loss detected (loss={loss_value.item():.4f}), clamping')
                    loss = loss.clamp(max=10.0, min=-10.0)

            if self.use_bf16:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()

            if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(tbar):
                if self.use_amp and not self.use_bf16:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm_before_clip = 0.0
                has_nan_grad = False
                need_grad_diag = (i % config.print_interval == 0) or (i + 1 == len(tbar))
                if need_grad_diag:
                    grad_total = 0
                    per_module_norms = {}
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            grad_total += 1
                            g_norm = p.grad.norm().item() ** 2
                            grad_norm_before_clip += g_norm
                            module_key = name.split('.')[0]
                            if module_key not in per_module_norms:
                                per_module_norms[module_key] = 0.0
                            per_module_norms[module_key] += g_norm
                            if t.isnan(p.grad).any() or t.isinf(p.grad).any():
                                has_nan_grad = True
                    grad_norm_before_clip = grad_norm_before_clip ** 0.5 if grad_total > 0 else 0.0
                    for k in per_module_norms:
                        per_module_norms[k] = per_module_norms[k] ** 0.5
                    if not has_nan_grad:
                        self.logger.logger.info(
                            f'[GRAD-DIAG] Epoch {epoch+1} Batch {i+1}: '
                            f'total_norm={grad_norm_before_clip:.4f}, '
                            f'per_module={{{", ".join(f"{k}={v:.4f}" for k, v in sorted(per_module_norms.items()))}}}')
                else:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            if t.isnan(p.grad).any() or t.isinf(p.grad).any():
                                has_nan_grad = True
                                break
                
                if has_nan_grad:
                    self._nan_skip_count += 1
                    _batch_scaler_skipped = True
                    self.scaler.update()
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            state = self.optimizer.state.get(p, None)
                            if state is not None:
                                if 'exp_avg' in state:
                                    state['exp_avg'].zero_()
                                if 'exp_avg_sq' in state:
                                    state['exp_avg_sq'].fill_(1e-6)
                    self.logger.logger.warning(
                        f'[TRAIN] Epoch {epoch+1} Batch {i+1}: NaN/Inf gradient detected, skipping optimizer step '
                        f'(consecutive={self._nan_skip_count}, scaler_scale={self.scaler.get_scale():.1f})')
                    if self._nan_skip_count >= self._nan_skip_threshold and not self._nan_lr_reduced:
                        for pg in self.optimizer.param_groups:
                            old_lr = pg['lr']
                            pg['lr'] = pg['lr'] * 0.2
                            self.logger.logger.warning(
                                f'[TRAIN] Epoch {epoch+1}: NaN/Inf gradient for {self._nan_skip_count} consecutive batches, '
                                f'reducing LR {old_lr:.8f} -> {pg["lr"]:.8f}')
                        self._nan_lr_reduced = True
                        self._nan_recovery_counter = 0
                        self._nan_skip_count = 0
                else:
                    self._nan_skip_count = 0
                    if self._nan_lr_reduced:
                        self._nan_recovery_counter = getattr(self, '_nan_recovery_counter', 0) + 1
                        if self._nan_recovery_counter >= 50:
                            for pg in self.optimizer.param_groups:
                                pg['lr'] = pg['lr'] * 5.0
                            self._nan_lr_reduced = False
                            self._nan_recovery_counter = 0
                            self.logger.logger.info(
                                f'[TRAIN] Epoch {epoch+1}: Recovered from NaN, restoring LR')
                    else:
                        self._nan_recovery_counter = 0
                    if self._bn_protection is not None:
                        self._bn_protection['grad_cliper'].clip(self.model)
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_max_norm)
                    
                    if not self.use_amp or self.use_bf16:
                        self.optimizer.step()
                        _batch_scaler_skipped = False
                        self.scaler.update()
                    else:
                        try:
                            step_result = self.scaler.step(self.optimizer)
                            if step_result is None:
                                _batch_scaler_skipped = True
                                self.logger.logger.warning(
                                    f'[TRAIN] Epoch {epoch+1} Batch {i+1}: scaler.step() overflow, '
                                    f'scaler_scale={self.scaler.get_scale():.1f} will auto-reduce')
                            else:
                                _batch_scaler_skipped = False
                        except Exception as e:
                            _batch_scaler_skipped = True
                            self.logger.logger.warning(
                                f'[TRAIN] Epoch {epoch+1} Batch {i+1}: scaler.step() failed with error: {e}')
                            try:
                                self.scaler.update()
                            except Exception:
                                pass
                        else:
                            self.scaler.update()
                    
                    if not _batch_scaler_skipped:
                        self.ema.update(self.model)
                    else:
                        scaler_skip_count += 1
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

            if not use_cutmix:
                joint_total += img.size(0)
                raw_joint_corrects += _compute_joint_acc([p.clone() for p in pred], label, true_lengths, config.num_heads)
                pred_masked = [p.clone() for p in pred]
                _apply_length_mask(pred_masked, length_logits, config.num_heads)
                joint_corrects += _compute_joint_acc(pred_masked, label, true_lengths, config.num_heads)
                c_corrects, c_total, d_corrects, d_total = _compute_char_acc(pred_masked, label, true_lengths, config.num_heads)
                char_corrects += c_corrects
                total_chars += c_total
                digit_corrects += d_corrects
                digit_total += d_total

            tbar.set_description(
                'Epoch %d, loss: %.3f, joint: %.3f, raw: %.3f, digit: %.3f' % (
                    epoch + 1, total_loss / (i + 1),
                    joint_corrects * 100 / max(joint_total, 1),
                    raw_joint_corrects * 100 / max(joint_total, 1),
                    digit_corrects * 100 / max(digit_total, 1)))

            if (i + 1) % config.print_interval == 0:
                acc_str = f'joint={joint_corrects * 100 / max(joint_total, 1):.2f}% raw={raw_joint_corrects * 100 / max(joint_total, 1):.2f}% char={char_corrects * 100 / max(total_chars, 1):.2f}% digit={digit_corrects * 100 / max(digit_total, 1):.2f}%'
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
        self._last_train_digit_acc = digit_corrects * 100 / max(digit_total, 1)
        self._last_train_raw_joint_acc = raw_joint_corrects * 100 / max(joint_total, 1)
        self._last_epoch_avg_loss = total_loss / max(len(self.train_loader), 1)
        if scaler_skip_count > 0:
            self.logger.logger.warning(f'[TRAIN] Epoch {epoch+1}: scaler.step() skipped {scaler_skip_count} time(s) '
                                       f'(inf/nan gradients), model may not be learning')
        return self._last_train_joint_acc

    def _eval(self):
        if self.ema is not None:
            model = self.ema.ema
        else:
            model = self._get_raw_model()
        model.eval()
        eval_bs = config.eval_batch_size
        max_retries = 3
        for attempt in range(max_retries):
            char_corrects = 0
            total_chars = 0
            digit_corrects = 0
            digit_total = 0
            joint_corrects = 0
            joint_total = 0
            raw_joint_corrects = 0
            length_corrects = 0
            length_total = 0
            oom_hit = False
            with t.no_grad():
                tbar = tqdm(self.val_loader)
                for i, (img, label, bbox_target, bbox_mask) in enumerate(tbar):
                    try:
                        img = img.to(self.device)
                        label = label.to(self.device)
                        bbox_mask = bbox_mask.to(self.device)
                        pred_cls, _, length_logits = model(img)

                        true_lengths = bbox_mask.sum(dim=1).long()
                        raw_joint_corrects += _compute_joint_acc([p.clone() for p in pred_cls], label, true_lengths, config.num_heads)

                        _apply_length_mask(pred_cls, length_logits, config.num_heads)

                        c_corrects, c_total, d_corrects, d_total = _compute_char_acc(pred_cls, label, true_lengths, config.num_heads)
                        char_corrects += c_corrects
                        total_chars += c_total
                        digit_corrects += d_corrects
                        digit_total += d_total

                        joint_corrects += _compute_joint_acc(pred_cls, label, true_lengths, config.num_heads)
                        joint_total += img.size(0)

                        pred_length = length_logits.argmax(dim=1)
                        clamped_true = true_lengths.clamp(max=config.num_heads)
                        length_corrects += (pred_length == clamped_true).sum().item()
                        length_total += img.size(0)

                        tbar.set_description('Val Char: %.2f%% Digit: %.2f%% Joint: %.2f%% Raw: %.2f%%' % (
                            char_corrects * 100 / max(total_chars, 1),
                            digit_corrects * 100 / max(digit_total, 1),
                            joint_corrects * 100 / max(joint_total, 1),
                            raw_joint_corrects * 100 / max(joint_total, 1)))

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
        digit_acc = digit_corrects / max(digit_total, 1)
        joint_acc = joint_corrects / max(joint_total, 1)
        raw_joint_acc = raw_joint_corrects / max(joint_total, 1)
        self._last_val_char_acc = char_acc
        self._last_val_digit_acc = digit_acc
        self._last_val_joint_acc = joint_acc
        self._last_val_raw_joint_acc = raw_joint_acc
        length_acc = length_corrects / max(length_total, 1)
        print(f'  Char Acc: {char_acc * 100:.2f}%  |  Digit Acc: {digit_acc * 100:.2f}%  |  Joint Acc: {joint_acc * 100:.2f}%  |  Raw Joint: {raw_joint_acc * 100:.2f}%  |  Length Acc: {length_acc * 100:.2f}%')
        self.logger.logger.info(f'[EVAL] length_acc={length_acc * 100:.2f}% ({length_corrects}/{length_total}) digit_acc={digit_acc * 100:.2f}% ({digit_corrects}/{digit_total}) raw_joint={raw_joint_acc * 100:.2f}% masked_joint={joint_acc * 100:.2f}%')

        return raw_joint_acc

    def eval_detailed(self):
        if self.ema is not None:
            model = self.ema.ema
        else:
            model = self._get_raw_model()
        model.eval()
        head_corrects = [0] * config.num_heads
        head_totals = [0] * config.num_heads
        char_corrects = 0
        total_chars = 0
        joint_corrects = 0
        joint_total = 0
        raw_joint_corrects = 0
        with t.no_grad():
            for img, label, bbox_target, bbox_mask in tqdm(self.val_loader, desc='Detailed Eval'):
                img = img.to(self.device)
                label = label.to(self.device)
                bbox_mask = bbox_mask.to(self.device)
                pred_cls, _, length_logits = model(img)

                true_lengths = bbox_mask.sum(dim=1).long()

                # 先计算 raw_joint (不带 length_mask)
                raw_joint_corrects += _compute_joint_acc([p.clone() for p in pred_cls], label, true_lengths, config.num_heads)

                for h in range(config.num_heads):
                    valid_mask = (true_lengths > h).float()
                    head_corrects[h] += ((pred_cls[h].argmax(1) == label[:, h]) * valid_mask).sum().item()
                    head_totals[h] += valid_mask.sum().item()

                c_corrects, c_total, d_corrects, d_total = _compute_char_acc(pred_cls, label, true_lengths, config.num_heads)
                char_corrects += c_corrects
                total_chars += c_total

                # 再计算 masked_joint
                pred_masked = [p.clone() for p in pred_cls]
                _apply_length_mask(pred_masked, length_logits, config.num_heads)
                joint_corrects += _compute_joint_acc(pred_masked, label, true_lengths, config.num_heads)
                joint_total += img.size(0)

                del img, label, pred_cls

        for h in range(config.num_heads):
            if head_totals[h] > 0:
                acc = head_corrects[h] / head_totals[h] * 100
                print(f'  Head {h+1} Char Acc: {acc:.2f}% ({head_corrects[h]}/{head_totals[h]})')
                self.logger.logger.info(f'[PER-POS] Head {h+1}: acc={acc:.2f}% ({head_corrects[h]}/{head_totals[h]})')

        if config.num_heads >= 3:
            length2_corrects = 0
            length2_totals = 0
            length3_corrects = 0
            length3_totals = 0
            for i, (img, label, bbox_target, bbox_mask) in enumerate(self.val_loader):
                if i > 0:
                    break
                img = img.to(self.device)
                label = label.to(self.device)
                bbox_mask = bbox_mask.to(self.device)
                pred_cls, _, length_logits = model(img)
                true_lengths = bbox_mask.sum(dim=1).long()
                pred_masked = [p.clone() for p in pred_cls]
                _apply_length_mask(pred_masked, length_logits, config.num_heads)
                for b in range(img.size(0)):
                    tlen = true_lengths[b].item()
                    if tlen == 2:
                        correct = all(pred_masked[h][b].argmax().item() == label[b, h].item() for h in range(2))
                        length2_corrects += int(correct)
                        length2_totals += 1
                    elif tlen == 3:
                        correct = all(pred_masked[h][b].argmax().item() == label[b, h].item() for h in range(3))
                        length3_corrects += int(correct)
                        length3_totals += 1
                del img, label, pred_cls
            if length2_totals > 0:
                print(f'  Length-2 Joint Acc: {length2_corrects / length2_totals * 100:.2f}% ({length2_corrects}/{length2_totals})')
                self.logger.logger.info(f'[PER-LEN] Length-2: acc={length2_corrects / length2_totals * 100:.2f}% ({length2_corrects}/{length2_totals})')
            if length3_totals > 0:
                print(f'  Length-3 Joint Acc: {length3_corrects / length3_totals * 100:.2f}% ({length3_corrects}/{length3_totals})')
                self.logger.logger.info(f'[PER-LEN] Length-3: acc={length3_corrects / length3_totals * 100:.2f}% ({length3_corrects}/{length3_totals})')

        char_acc = char_corrects / max(total_chars, 1)
        joint_acc = joint_corrects / max(joint_total, 1)
        raw_joint_acc = raw_joint_corrects / max(joint_total, 1)
        print(f'  Overall Char Acc: {char_acc * 100:.2f}%  |  Raw Joint: {raw_joint_acc * 100:.2f}%  |  Masked Joint: {joint_acc * 100:.2f}%')

        t.cuda.empty_cache()
        self.model.train()

        return raw_joint_acc

    def eval_tta(self):
        if self.ema is not None:
            model = self.ema.ema
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

        char_corrects, total_chars, digit_corrects, digit_total = _compute_char_acc(pred_heads, all_labels, true_lengths, config.num_heads)
        char_acc = char_corrects / max(total_chars, 1)
        digit_acc = digit_corrects / max(digit_total, 1)

        joint_corrects = _compute_joint_acc(pred_heads, all_labels, true_lengths, config.num_heads)
        joint_acc = joint_corrects / len(self.val_set)

        print(f'TTA Char Acc: {char_acc * 100:.2f}%  |  Digit Acc: {digit_acc * 100:.2f}%  |  Joint Acc: {joint_acc * 100:.2f}%')
        self.model.train()
        return joint_acc
