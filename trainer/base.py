import os
import gc
import copy
import time
import logging
import psutil
import signal
import glob as glob_mod
import torch as t
from tqdm.auto import tqdm
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, CosineAnnealingWarmRestarts
from torch.amp import GradScaler
from datetime import datetime
from config import config, SCRIPT_DIR, GPU_PLATFORM, TOTAL_VRAM_GB, NUM_PHYSICAL_CORES
from utils.compile_utils import get_raw_model as _get_raw_model_external
from utils.platform import is_hip_error, is_cuda_error, is_nvidia_cuda


def _convert_p1_channels(state_dict, model_sd):
    converted = {}
    converted_keys = []
    p1_layers = {
        'backbone.l1_reduce.0.weight',
        'backbone.l1_reduce.1.weight',
        'backbone.l1_reduce.1.bias',
        'backbone.l1_reduce.1.running_mean',
        'backbone.l1_reduce.1.running_var',
        'backbone.smooth_p1.0.weight',
        'backbone.smooth_p1.1.weight',
        'backbone.smooth_p1.1.bias',
        'backbone.smooth_p1.1.running_mean',
        'backbone.smooth_p1.1.running_var',
        'backbone.fuse.0.weight',
    }
    for k, v in state_dict.items():
        if k not in model_sd or v.shape == model_sd[k].shape:
            converted[k] = v
            continue
        if k not in p1_layers:
            converted[k] = v
            continue
        target_shape = model_sd[k].shape
        if k == 'backbone.l1_reduce.0.weight':
            converted[k] = v[:target_shape[0]]
            converted_keys.append(k)
        elif k.startswith('backbone.l1_reduce.1.'):
            converted[k] = v[:target_shape[0]]
            converted_keys.append(k)
        elif k == 'backbone.smooth_p1.0.weight':
            converted[k] = v[:target_shape[0], :target_shape[1]]
            converted_keys.append(k)
        elif k.startswith('backbone.smooth_p1.1.'):
            converted[k] = v[:target_shape[0]]
            converted_keys.append(k)
        elif k == 'backbone.fuse.0.weight':
            old_p1 = v.shape[1] - 768
            new_p1 = target_shape[1] - 768
            if old_p1 > new_p1:
                p1_part = v[:, :new_p1]
                rest_part = v[:, old_p1:]
                converted[k] = t.cat([p1_part, rest_part], dim=1)
            else:
                p1_part = v[:, :old_p1]
                rest_part = v[:, old_p1:]
                pad = t.zeros(target_shape[0], new_p1 - old_p1, *target_shape[2:])
                converted[k] = t.cat([p1_part, pad, rest_part], dim=1)
            converted_keys.append(k)
        else:
            converted[k] = v
    return converted, converted_keys


def _convert_head_interaction(state_dict, model_sd):
    hi_prefix = 'head_interaction.encoder.layers.'
    ckpt_layer_indices = set()
    for k in state_dict:
        if k.startswith(hi_prefix):
            parts = k[len(hi_prefix):].split('.')
            if parts[0].isdigit():
                ckpt_layer_indices.add(int(parts[0]))
    model_layer_indices = set()
    for k in model_sd:
        if k.startswith(hi_prefix):
            parts = k[len(hi_prefix):].split('.')
            if parts[0].isdigit():
                model_layer_indices.add(int(parts[0]))
    if not ckpt_layer_indices or not model_layer_indices:
        return state_dict, []
    ckpt_ffn_dim = None
    model_ffn_dim = None
    for k, v in state_dict.items():
        if k.startswith(hi_prefix) and 'linear1.weight' in k:
            ckpt_ffn_dim = v.shape[0]
            break
    for k, v in model_sd.items():
        if k.startswith(hi_prefix) and 'linear1.weight' in k:
            model_ffn_dim = v.shape[0]
            break
    ffn_changed = (ckpt_ffn_dim is not None and model_ffn_dim is not None
                   and ckpt_ffn_dim != model_ffn_dim)
    layers_changed = (ckpt_layer_indices != model_layer_indices)
    if not ffn_changed and not layers_changed:
        return state_dict, []
    converted = {}
    converted_keys = []
    ffn_subkeys = {'linear1.weight', 'linear1.bias', 'linear2.weight'}
    for k, v in state_dict.items():
        if not k.startswith(hi_prefix):
            converted[k] = v
            continue
        parts = k[len(hi_prefix):].split('.')
        layer_idx = int(parts[0])
        sub_key = '.'.join(parts[1:])
        if layer_idx not in model_layer_indices:
            continue
        if ffn_changed and sub_key in ffn_subkeys:
            target_shape = model_sd[k].shape
            if sub_key == 'linear1.weight':
                new_v = t.zeros(*target_shape)
                copy_rows = min(v.shape[0], target_shape[0])
                new_v[:copy_rows] = v[:copy_rows]
                if target_shape[0] > copy_rows:
                    new_v[copy_rows:] = t.randn(target_shape[0] - copy_rows, *target_shape[1:]) * 0.02
                converted[k] = new_v
                converted_keys.append(k)
            elif sub_key == 'linear1.bias':
                new_v = t.zeros(*target_shape)
                copy_size = min(v.shape[0], target_shape[0])
                new_v[:copy_size] = v[:copy_size]
                converted[k] = new_v
                converted_keys.append(k)
            elif sub_key == 'linear2.weight':
                new_v = t.zeros(*target_shape)
                copy_cols = min(v.shape[1], target_shape[1])
                new_v[:, :copy_cols] = v[:, :copy_cols]
                if target_shape[1] > copy_cols:
                    new_v[:, copy_cols:] = t.randn(target_shape[0], target_shape[1] - copy_cols) * 0.02
                converted[k] = new_v
                converted_keys.append(k)
        else:
            converted[k] = v
    return converted, converted_keys


def _load_state_dict_compat(model, state_dict, strict=False):
    model_sd = model.state_dict()
    converted, p1_keys = _convert_p1_channels(state_dict, model_sd)
    converted, hi_keys = _convert_head_interaction(converted, model_sd)
    converted_keys = p1_keys + hi_keys
    filtered = {}
    skipped_keys = []
    for k, v in converted.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped_keys.append((k, v.shape, model_sd[k].shape))
        else:
            if not strict:
                filtered[k] = v
    if converted_keys:
        import logging as _logging
        _logger = _logging.getLogger('checkpoint_compat')
        _logger.info(
            f'[CKPT-COMPAT] Converted {len(converted_keys)} keys '
            f'(P1-channel: {len(p1_keys)}, HeadInteraction: {len(hi_keys)})')
    if skipped_keys:
        import logging as _logging
        _logger = _logging.getLogger('checkpoint_compat')
        _logger.warning(
            f'[CKPT-COMPAT] Skipped {len(skipped_keys)} keys with shape mismatch:')
        for k, ckpt_shape, model_shape in skipped_keys[:10]:
            _logger.warning(f'  {k}: checkpoint {list(ckpt_shape)} vs model {list(model_shape)}')
        if len(skipped_keys) > 10:
            _logger.warning(f'  ... and {len(skipped_keys) - 10} more')
    incompatible = model.load_state_dict(filtered, strict=strict)
    return incompatible, skipped_keys


class TrainingLogger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join(SCRIPT_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'train_{timestamp}.log')
        latest_path = os.path.join(log_dir, 'latest.log')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        try:
            os.symlink(self.log_path, latest_path)
        except OSError:
            pass

        self.logger = logging.getLogger(f'TrainingLogger_{timestamp}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        fh = logging.FileHandler(self.log_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        self.epoch_start_time = None
        self.gpu_peak_mem = 0

    def log_init(self, model_type, device, total_params, trainable_params):
        self.logger.info(f'=== Training Initialization ===')
        self.logger.info(f'Model type: {model_type}')
        self.logger.info(f'Device: {device}')
        self.logger.info(f'GPU Platform: {GPU_PLATFORM.upper()}')
        self.logger.info(f'Total VRAM: {TOTAL_VRAM_GB:.1f} GB')
        self.logger.info(f'Physical CPU Cores: {NUM_PHYSICAL_CORES}')
        self.logger.info(f'Total parameters: {total_params:,}')
        self.logger.info(f'Trainable parameters: {trainable_params:,}')
        if t.cuda.is_available():
            gpu_name = t.cuda.get_device_name(0)
            self.logger.info(f'GPU: {gpu_name}')
        self.logger.info(f'Train batch size: {config.batch_size}')
        self.logger.info(f'Eval batch size: {config.eval_batch_size}')
        self.logger.info(f'Gradient Accumulation Steps: {config.grad_accum_steps}')
        self.logger.info(f'Equivalent Batch Size: {config.batch_size * config.grad_accum_steps}')
        self.logger.info(f'Learning rate: {config.lr}')
        self.logger.info(f'Epochs: {config.epoches}')
        self.logger.info(f'Use Torch Compile: {config.use_torch_compile}')
        if config.use_torch_compile:
            self.logger.info(f'Compile Mode: {config.compile_mode}')
        self.logger.info(f'Log file: {self.log_path}')

    def log_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.gpu_peak_mem = 0
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()
        self.logger.info(f'--- Epoch {epoch + 1}/{config.epoches} START ---')

    def log_batch(self, epoch, batch_idx, total_batches, loss, lr, acc_str,
                  cls_loss=None, bbox_loss=None, div_loss=None, ord_loss=None, attn_loss=None,
                  batch_time=None):
        msg = f'[TRAIN] Epoch={epoch + 1}/{config.epoches} Batch={batch_idx + 1}/{total_batches} ' \
              f'loss={loss:.4f}'
        if cls_loss is not None:
            msg += f' cls={cls_loss:.4f}'
        if bbox_loss is not None:
            msg += f' bbox={bbox_loss:.4f}'
        if div_loss is not None:
            msg += f' div={div_loss:.4f}'
        if ord_loss is not None:
            msg += f' ord={ord_loss:.4f}'
        if attn_loss is not None:
            msg += f' attn={attn_loss:.4f}'
        msg += f' lr={lr:.8f} {acc_str}'
        if t.cuda.is_available():
            alloc = t.cuda.memory_allocated() / (1024**3)
            peak = t.cuda.max_memory_allocated() / (1024**3)
            gpu_props = t.cuda.get_device_properties(0)
            total = getattr(gpu_props, 'total_mem', getattr(gpu_props, 'total_memory', 0)) / (1024**3)
            self.gpu_peak_mem = max(self.gpu_peak_mem, peak)
            msg += f' gpu_mem={alloc:.1f}/{total:.1f}GB peak={peak:.1f}GB'
        cpu_mem = psutil.virtual_memory()
        msg += f' cpu_mem={cpu_mem.used / (1024**3):.1f}/{cpu_mem.total / (1024**3):.1f}GB'
        if batch_time is not None:
            msg += f' batch_time={batch_time:.2f}s'
        self.logger.info(msg)

    def log_epoch_end(self, epoch, train_acc, val_acc, lr, is_best=False, patience_counter=0,
                      char_acc=None, digit_acc=None, raw_joint_acc=None):
        epoch_time = time.time() - self.epoch_start_time
        mins, secs = divmod(int(epoch_time), 60)
        msg = f'[EPOCH] Epoch={epoch + 1}/{config.epoches} ' \
              f'train_acc={train_acc:.2f}% val_joint={val_acc:.2f}%'
        if char_acc is not None:
            msg += f' val_char={char_acc:.2f}%'
        if digit_acc is not None:
            msg += f' val_digit={digit_acc:.2f}%'
        if raw_joint_acc is not None:
            msg += f' raw_joint={raw_joint_acc:.2f}%'
        msg += f' lr={lr:.8f} epoch_time={mins}m{secs:02d}s'
        if t.cuda.is_available():
            msg += f' gpu_peak={self.gpu_peak_mem:.1f}GB'
        msg += f' best={is_best} patience={patience_counter}/{config.early_stopping_patience}'
        self.logger.info(msg)

    def log_eval(self, acc, label='Evaluation'):
        self.logger.info(f'[EVAL] {label} acc={acc * 100:.2f}%')

    def log_early_stop(self, epoch, best_acc, best_path):
        self.logger.info(f'[STOP] Early stopping at epoch {epoch + 1}, '
                         f'best_acc={best_acc * 100:.2f}%, best_path={best_path}')

    def log_save(self, path, save_type='full'):
        self.logger.info(f'[SAVE] Model saved to {path} (type={save_type})')


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        if device is None:
            device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.device = device
        self.ema = copy.deepcopy(model).to(device)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with t.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data.to(self.device), alpha=1 - self.decay)

    def to_device(self, device=None):
        target_device = device or self.device
        if target_device != self.device:
            self.ema = self.ema.to(target_device)
            self.device = target_device
        return self.ema

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name in ('training', 'num_heads'):
            return getattr(self.ema, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class BaseTrainer:
    _checkpoint_prefix = 'model'
    _eval_label = 'Evaluation'

    def __init__(self):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        self.use_amp = getattr(config, 'use_amp', True) and self.device.type == 'cuda'
        self.best_acc = 0
        self.best_checkpoint_path = ''
        self.train_log = []
        self._current_epoch = 0
        self.patience_counter = 0
        self.early_stop_triggered = False
        self.logger = TrainingLogger()
        self._oom_retry_count = 0
        self._original_batch_size = config.batch_size
        self._original_effective_batch_size = config.batch_size * config.grad_accum_steps
        self._stable_batch_size = None
        self._stable_epoch_count = 0
        self._min_stable_epochs_for_recovery = 3
        self._pending_save = False
        self._periodic_checkpoints = []

        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._handle_save_signal)

    def _handle_save_signal(self, signum, frame):
        print(f"\n[EMERGENCY] Received SIGUSR1 signal, will save checkpoint after current batch...")
        self._pending_save = True

    def _gpu_sanity_check(self):
        if not t.cuda.is_available():
            return
        print('[GPU-CHECK] Running GPU sanity check...')
        try:
            test = t.randn(4, 3, 32, 32, device=self.device)
            _ = test + test
            t.cuda.synchronize()
            del test
            t.cuda.empty_cache()
        except RuntimeError as e:
            err_str = str(e).lower()
            if is_hip_error(err_str) or is_cuda_error(err_str):
                print(f'[GPU-CHECK] GPU sanity check FAILED (basic ops): {e}')
                self._handle_gpu_check_failure(e)
                return
            else:
                raise
        try:
            conv_weight = t.randn(16, 3, 3, 3, device=self.device)
            conv_input = t.randn(2, 3, 32, 32, device=self.device)
            _ = t.nn.functional.conv2d(conv_input, conv_weight, padding=1)
            t.cuda.synchronize()
            del conv_weight, conv_input
            t.cuda.empty_cache()
        except RuntimeError as e:
            err_str = str(e).lower()
            if is_hip_error(err_str) or is_cuda_error(err_str):
                print(f'[GPU-CHECK] GPU sanity check FAILED (conv2d): {e}')
                self._handle_gpu_check_failure(e)
                return
            else:
                raise
        try:
            mat_a = t.randn(64, 64, device=self.device)
            mat_b = t.randn(64, 64, device=self.device)
            _ = mat_a @ mat_b
            t.cuda.synchronize()
            del mat_a, mat_b
            t.cuda.empty_cache()
        except RuntimeError as e:
            err_str = str(e).lower()
            if is_hip_error(err_str) or is_cuda_error(err_str):
                print(f'[GPU-CHECK] GPU sanity check FAILED (matmul): {e}')
                self._handle_gpu_check_failure(e)
                return
            else:
                raise
        print('[GPU-CHECK] GPU sanity check passed (add+conv2d+matmul)')

    def _handle_gpu_check_failure(self, original_error):
        print('[GPU-CHECK] Attempting CUDA context reset...')
        try:
            t.cuda.empty_cache()
            t.cuda.synchronize()
        except Exception:
            pass
        try:
            test = t.randn(2, 3, 16, 16, device=self.device)
            _ = test + test
            t.cuda.synchronize()
            del test
            t.cuda.empty_cache()
            print('[GPU-CHECK] GPU recovered after reset')
        except Exception as e2:
            print(f'[GPU-CHECK] GPU unrecoverable: {e2}')
            print('[GPU-CHECK] Falling back to CPU')
            self.device = t.device('cpu')
            self.use_amp = False
            config.use_torch_compile = False
            config.multiprocessing_context = None

    def _setup_optimizer(self, backbone_params, other_params):
        if config.optimizer_type == 'adamw':
            from torch.optim import AdamW
            return AdamW([
                {'params': backbone_params, 'lr': config.lr * config.backbone_lr_factor},
                {'params': other_params, 'lr': config.lr},
            ], weight_decay=config.weights_decay)
        return SGD([
            {'params': backbone_params, 'lr': config.lr * config.backbone_lr_factor},
            {'params': other_params, 'lr': config.lr},
        ], momentum=config.momentum, weight_decay=config.weights_decay, nesterov=True)

    def _setup_scheduler(self):
        if config.scheduler_type == 'warm_restarts':
            warmup_scheduler = LinearLR(self.optimizer, start_factor=config.warmup_start_factor,
                                        total_iters=config.warmup_epochs)
            restart_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=config.scheduler_eta_min)
            return SequentialLR(self.optimizer,
                                schedulers=[warmup_scheduler, restart_scheduler],
                                milestones=[config.warmup_epochs])
        warmup_scheduler = LinearLR(self.optimizer, start_factor=config.warmup_start_factor,
                                    total_iters=config.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epoches - config.warmup_epochs,
                                             eta_min=config.scheduler_eta_min)
        return SequentialLR(self.optimizer,
                            schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[config.warmup_epochs])

    def _setup_scheduler_with_restarts(self):
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_T0,
            T_mult=config.scheduler_T_mult,
            eta_min=config.scheduler_eta_min
        )

    def _setup_scaler(self):
        return GradScaler(self.device.type, enabled=self.use_amp)

    def _pre_epoch_hook(self, epoch):
        pass

    def _rebuild_dataloaders(self):
        pass

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _eval(self):
        raise NotImplementedError

    def _cleanup_old_checkpoints(self):
        if len(self._periodic_checkpoints) <= config.max_checkpoints:
            return
        to_remove = self._periodic_checkpoints[:-config.max_checkpoints]
        for path in to_remove:
            if os.path.exists(path) and path != self.best_checkpoint_path:
                try:
                    os.remove(path)
                    self.logger.logger.info(f'[CLEANUP] Removed old checkpoint: {path}')
                except OSError as e:
                    self.logger.logger.warning(f'[CLEANUP] Failed to remove {path}: {e}')
        self._periodic_checkpoints = self._periodic_checkpoints[-config.max_checkpoints:]

    def _cleanup_dataloader(self, loader):
        if loader is not None:
            if hasattr(loader, '_iterator'):
                try:
                    del loader._iterator
                except Exception:
                    pass
            try:
                loader._IterableDataset_len_called = None
            except Exception:
                pass
            gc.collect()

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            if self.early_stop_triggered:
                self.logger.log_early_stop(epoch, self.best_acc, self.best_checkpoint_path)
                break

            self._current_epoch = epoch

            if epoch > 0:
                self.lr_scheduler.step()

            if hasattr(self, '_post_reset_warmup_epochs') and self._post_reset_warmup_epochs > 0:
                total_warmup = 10
                progress = 1.0 - (self._post_reset_warmup_epochs / total_warmup)
                for pg, start_lr, target_lr in zip(
                        self.optimizer.param_groups,
                        self._warmup_start_lrs, self._warmup_target_lrs):
                    pg['lr'] = start_lr + (target_lr - start_lr) * progress
                self._post_reset_warmup_epochs -= 1
                self.logger.logger.info(
                    f'[WARMUP-RESET] progress={progress:.2f} LR={self.optimizer.param_groups[0]["lr"]:.8f}')
                if self._post_reset_warmup_epochs == 0:
                    del self._post_reset_warmup_epochs
                    del self._warmup_start_lrs
                    del self._warmup_target_lrs

            self._pre_epoch_hook(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']

            if epoch == config.start_epoch or epoch % 10 == 0:
                raw = self._get_raw_model()
                param_norm = sum(p.data.norm().item() ** 2 for p in raw.parameters()) ** 0.5
                grad_norm = sum(p.grad.norm().item() ** 2 for p in raw.parameters() if p.grad is not None) ** 0.5 if epoch > config.start_epoch else 0.0
                self.logger.logger.info(
                    f'[DIAG] Epoch {epoch+1}: param_norm={param_norm:.2f}, '
                    f'grad_norm={grad_norm:.4f}, '
                    f'lr={self.optimizer.param_groups[0]["lr"]:.6f}, '
                    f'scaler_scale={self.scaler.get_scale():.1f}')

            self.logger.log_epoch_start(epoch)
            try:
                train_acc = self._train_epoch(epoch)
            except RuntimeError as e:
                err_msg = str(e).lower()
                if 'out of memory' in err_msg and self._oom_retry_count < 3:
                    self._oom_retry_count += 1
                    old_bs = config.batch_size
                    old_grad_accum = config.grad_accum_steps
                    t.cuda.empty_cache()
                    peak_gb = t.cuda.max_memory_allocated() / (1024**3)
                    gpu_props = t.cuda.get_device_properties(0)
                    total_gb = getattr(gpu_props, 'total_mem', getattr(gpu_props, 'total_memory', 0)) / (1024**3)
                    headroom = total_gb * config.oom_headroom_ratio
                    if peak_gb > 0 and total_gb > 0:
                        safe_ratio = (total_gb - headroom) / peak_gb
                        config.batch_size = max(int(config.batch_size * safe_ratio), 8)
                    else:
                        config.batch_size = max(int(config.batch_size * 0.75), 8)
                    effective_accum = max(-(-self._original_effective_batch_size // config.batch_size), 1)
                    config.grad_accum_steps = min(effective_accum, 32)
                    self._stable_batch_size = None
                    new_effective_bs = config.batch_size * config.grad_accum_steps
                    old_effective_bs = old_bs * old_grad_accum
                    lr_scale = new_effective_bs / old_effective_bs
                    self.logger.logger.warning(
                        f'[OOM] Reducing batch_size {old_bs} -> {config.batch_size}, '
                        f'peak={peak_gb:.1f}GB/{total_gb:.1f}GB, '
                        f'headroom={headroom:.1f}GB, '
                        f'grad_accum_steps={old_grad_accum}->{config.grad_accum_steps}, '
                        f'effective_bs={old_effective_bs}->{new_effective_bs} (retry {self._oom_retry_count})')
                    for pg in self.optimizer.param_groups:
                        old_lr = pg['lr']
                        pg['lr'] = pg['lr'] * lr_scale
                        self.logger.logger.info(
                            f'[OOM] Adjusted LR: {old_lr:.8f} -> {pg["lr"]:.8f} (scale={lr_scale:.3f})')
                    t.cuda.reset_peak_memory_stats()
                    self._cleanup_dataloader(self.train_loader)
                    self._rebuild_dataloaders()
                    continue
                elif 'shared memory' in err_msg and config.num_workers > 0:
                    config.num_workers = max(config.num_workers // 2, 0)
                    config.prefetch_factor = max(config.prefetch_factor - 1, 1) if config.num_workers > 0 else None
                    self.logger.logger.warning(
                        f'[SHM] Shared memory error. Reducing num_workers to {config.num_workers}, '
                        f'prefetch_factor to {config.prefetch_factor}')
                    self._cleanup_dataloader(self.train_loader)
                    self._rebuild_dataloaders()
                    continue
                else:
                    raise
            self._oom_retry_count = 0

            if hasattr(self, '_last_epoch_avg_loss') and hasattr(self, '_prev_epoch_avg_loss'):
                if self._last_epoch_avg_loss > self._prev_epoch_avg_loss * 3.0 and self._prev_epoch_avg_loss > 0:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = pg['lr'] * 0.5
                    self.logger.logger.warning(
                        f'[LOSS-SPIKE] Loss jumped from {self._prev_epoch_avg_loss:.4f} to '
                        f'{self._last_epoch_avg_loss:.4f}, halving LR to {self.optimizer.param_groups[0]["lr"]:.8f}')
            if hasattr(self, '_last_epoch_avg_loss'):
                self._prev_epoch_avg_loss = self._last_epoch_avg_loss
            self._stable_epoch_count += 1
            if self._stable_batch_size is None:
                self._stable_batch_size = config.batch_size
                self._stable_epoch_count = 0
                self.logger.logger.info(f'[STABLE] batch_size={config.batch_size} confirmed stable')
            if config.batch_size < self._stable_batch_size:
                if self._stable_epoch_count >= self._min_stable_epochs_for_recovery:
                    if t.cuda.is_available():
                        current_usage = t.cuda.max_memory_allocated() / (1024**3)
                        gpu_props = t.cuda.get_device_properties(0)
                        total_gb = getattr(gpu_props, 'total_mem', getattr(gpu_props, 'total_memory', 0)) / (1024**3)
                        headroom = total_gb * config.oom_headroom_ratio
                        usage_ratio = current_usage / (total_gb - headroom) if (total_gb - headroom) > 0 else 1.0
                    else:
                        usage_ratio = 0.5
                    if usage_ratio < 0.7:
                        recovery_step = max(16, int(self._stable_batch_size * 0.1))
                    else:
                        recovery_step = 8
                    new_bs = min(config.batch_size + recovery_step, self._stable_batch_size)
                    if new_bs != config.batch_size:
                        old_bs = config.batch_size
                        old_grad_accum = config.grad_accum_steps
                        config.batch_size = new_bs
                        config.grad_accum_steps = max(-(-self._original_effective_batch_size // config.batch_size), 1)
                        new_effective_bs = config.batch_size * config.grad_accum_steps
                        old_effective_bs = old_bs * old_grad_accum
                        lr_scale = new_effective_bs / old_effective_bs
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = pg['lr'] * lr_scale
                        self._stable_epoch_count = 0
                        self.logger.logger.info(
                            f'[RECOVER] Increasing batch_size to {config.batch_size} '
                            f'(usage_ratio={usage_ratio:.2f}, step={recovery_step}), '
                            f'grad_accum_steps={old_grad_accum}->{config.grad_accum_steps}, '
                            f'effective_bs={old_effective_bs}->{new_effective_bs}, '
                            f'lr_scale={lr_scale:.3f}')
                        self._cleanup_dataloader(self.train_loader)
                        self._rebuild_dataloaders()
                else:
                    self.logger.logger.info(
                        f'[WAIT] Waiting {self._min_stable_epochs_for_recovery - self._stable_epoch_count} more stable epochs before batch_size recovery')

            if (epoch + 1) % config.eval_interval == 0 or self._pending_save:
                acc = 0.0
                char_acc_val = None
                if self.val_loader is not None:
                    acc = self._eval()
                    if hasattr(self, '_last_val_char_acc'):
                        char_acc_val = self._last_val_char_acc
                is_best = acc > self.best_acc
                train_joint = train_acc
                train_char = getattr(self, '_last_train_char_acc', None)
                train_digit = getattr(self, '_last_train_digit_acc', None)
                val_digit = getattr(self, '_last_val_digit_acc', None)
                val_raw_joint = getattr(self, '_last_val_raw_joint_acc', None)
                self.train_log.append({
                    'epoch': epoch + 1,
                    'train_joint_acc': train_joint,
                    'train_char_acc': train_char,
                    'train_digit_acc': train_digit,
                    'val_joint_acc': acc * 100,
                    'val_char_acc': char_acc_val * 100 if char_acc_val is not None else None,
                    'val_digit_acc': val_digit * 100 if val_digit is not None else None,
                    'val_raw_joint_acc': val_raw_joint * 100 if val_raw_joint is not None else None,
                    'lr': current_lr
                })
                self._check_early_stopping(acc, epoch)
                self.logger.log_epoch_end(epoch, train_acc, acc * 100, current_lr, is_best,
                                          self.patience_counter,
                                          char_acc=char_acc_val * 100 if char_acc_val is not None else None,
                                          digit_acc=val_digit * 100 if val_digit is not None else None,
                                          raw_joint_acc=val_raw_joint * 100 if val_raw_joint is not None else None)
                os.makedirs(config.checkpoints, exist_ok=True)

                if is_best:
                    best_path = os.path.join(config.checkpoints,
                                             'best-%s-acc-%.2f.pth' % (self._checkpoint_prefix, acc * 100))
                    old_best = self.best_checkpoint_path
                    self.best_acc = acc
                    self.best_checkpoint_path = best_path
                    self.save_model(best_path, save_opt=True)
                    self.logger.log_save(best_path, save_type='best')
                    if old_best and os.path.exists(old_best) and old_best != best_path:
                        if old_best not in self._periodic_checkpoints:
                            try:
                                os.remove(old_best)
                                self.logger.logger.info(f'[CLEANUP] Removed old best: {old_best}')
                            except OSError:
                                pass

                if (epoch + 1) % config.checkpoint_interval == 0 or self._pending_save:
                    periodic_path = os.path.join(config.checkpoints,
                                                 'epoch-%s-%d-acc-%.2f.pth' % (self._checkpoint_prefix, epoch + 1, acc * 100))
                    self.save_model(periodic_path, save_opt=True)  # 保存完整训练状态，方便续训
                    self.logger.log_save(periodic_path, save_type='periodic')
                    self._periodic_checkpoints.append(periodic_path)
                    self._cleanup_old_checkpoints()

                if self._pending_save:
                    print(f"[EMERGENCY] Emergency save completed for epoch {epoch + 1}")
                    self._pending_save = False

    def save_model(self, save_path, save_opt=False, save_config=False):
        raw_model = self._get_raw_model()
        if self.ema is not None:
            dicts = {'model': self.ema.ema.state_dict(), 'train_model': raw_model.state_dict()}
        else:
            dicts = {'model': raw_model.state_dict()}
        dicts['model_type'] = self._model_type
        dicts['epoch'] = self._current_epoch + 1
        dicts['best_acc'] = self.best_acc
        dicts['best_checkpoint_path'] = self.best_checkpoint_path
        dicts['patience_counter'] = self.patience_counter
        dicts['optimizer_type'] = config.optimizer_type
        dicts['scheduler_type'] = config.scheduler_type
        if save_opt:
            opt_state = self.optimizer.state_dict()
            param_name_groups = []
            param_idx = 0
            all_named_params = list(raw_model.named_parameters())
            for pg in self.optimizer.param_groups:
                names = []
                for _ in pg['params']:
                    if param_idx < len(all_named_params):
                        names.append(all_named_params[param_idx][0])
                    param_idx += 1
                param_name_groups.append(names)
            opt_state['param_name_groups'] = param_name_groups
            dicts['opt'] = opt_state
            dicts['lr_scheduler'] = self.lr_scheduler.state_dict()
            dicts['scaler'] = self.scaler.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        dicts['train_log'] = self.train_log
        t.save(dicts, save_path)

    def _get_raw_model(self):
        return _get_raw_model_external(self.model)

    def _restore_optimizer_robust(self, ckpt):
        if not isinstance(ckpt, dict) or 'opt' not in ckpt:
            return False
        ckpt_opt_state = ckpt['opt']
        if not isinstance(ckpt_opt_state, dict) or 'state' not in ckpt_opt_state:
            return False

        try:
            self.optimizer.load_state_dict(ckpt_opt_state)
            self.logger.logger.info('Restored optimizer state from checkpoint')
            return True
        except Exception as e:
            self.logger.logger.warning(
                f'Direct optimizer restore failed: {e}. Attempting parameter-level matching...')

        ckpt_state = ckpt_opt_state.get('state', {})
        ckpt_param_groups = ckpt_opt_state.get('param_groups', [])
        ckpt_param_name_groups = ckpt_opt_state.get('param_name_groups', None)

        if ckpt_param_name_groups is None:
            self.logger.logger.warning(
                'Checkpoint optimizer has no param_name_groups, cannot do parameter-level matching. '
                'Using new optimizer.')
            return False

        ckpt_name_to_state_idx = {}
        global_idx = 0
        for gi, name_list in enumerate(ckpt_param_name_groups):
            for name in name_list:
                ckpt_name_to_state_idx[name] = global_idx
                global_idx += 1

        raw_model = self._get_raw_model()
        all_named_params = list(raw_model.named_parameters())

        current_param_idx = 0
        restored = 0
        for pg in self.optimizer.param_groups:
            for param in pg['params']:
                if current_param_idx < len(all_named_params):
                    name = all_named_params[current_param_idx][0]
                    if name in ckpt_name_to_state_idx:
                        ckpt_idx = ckpt_name_to_state_idx[name]
                        if ckpt_idx in ckpt_state:
                            try:
                                self.optimizer.state[param].update(ckpt_state[ckpt_idx])
                                restored += 1
                            except Exception:
                                pass
                current_param_idx += 1

        if restored > 0:
            self.logger.logger.info(
                f'Partially restored optimizer state: {restored}/{len(all_named_params)} params')
            return True
        else:
            self.logger.logger.warning('Failed to restore any optimizer state. Using new optimizer.')
            return False

    def load_model(self, load_path, skip_load_weights=False, save_opt=False, save_config=False):
        dicts = t.load(load_path, map_location=self.device, weights_only=False)
        raw_model = self._get_raw_model()
        if not skip_load_weights:
            incompatible, skipped = _load_state_dict_compat(raw_model, dicts['model'])
            if skipped:
                self.logger.logger.warning(
                    f'[LOAD] Skipped {len(skipped)} shape-mismatched keys during load')
            missing = incompatible.missing_keys
            unexpected = incompatible.unexpected_keys
            if missing or unexpected:
                self.logger.logger.warning(
                    f'[LOAD] Incompatible keys detected. '
                    f'Missing: {len(missing)}, '
                    f'Unexpected: {len(unexpected)}')
                if missing:
                    self.logger.logger.warning(
                        f'[LOAD] Missing keys (randomly initialized): '
                        f'{missing[:5]}...'
                        if len(missing) > 5
                        else f'[LOAD] Missing keys: {missing}')
            if self.ema is not None:
                _load_state_dict_compat(self.ema.ema, dicts['model'])
        if 'epoch' in dicts:
            config.start_epoch = dicts['epoch']
            self._current_epoch = dicts['epoch']
        elif 'train_log' in dicts and len(dicts['train_log']) > 0:
            config.start_epoch = dicts['train_log'][-1]['epoch']
            self._current_epoch = dicts['train_log'][-1]['epoch']
        if 'best_acc' in dicts:
            self.best_acc = dicts['best_acc']
        if 'best_checkpoint_path' in dicts:
            self.best_checkpoint_path = dicts['best_checkpoint_path']
        if 'patience_counter' in dicts:
            self.patience_counter = dicts['patience_counter']
        if 'train_log' in dicts:
            self.train_log = dicts['train_log']
        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])
            if 'lr_scheduler' in dicts:
                self.lr_scheduler.load_state_dict(dicts['lr_scheduler'])
            if 'scaler' in dicts:
                self.scaler.load_state_dict(dicts['scaler'])
        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)
        if not save_opt and config.start_epoch > 0:
            if 'lr_scheduler' in dicts:
                try:
                    self.lr_scheduler.load_state_dict(dicts['lr_scheduler'])
                    self.logger.logger.info(
                        f'[LOAD] Restored lr_scheduler state from checkpoint')
                except Exception as e:
                    self.logger.logger.warning(
                        f'[LOAD] Failed to restore lr_scheduler state: {e}, '
                        f'falling back to step-based advance')
                    for _ in range(config.start_epoch):
                        self.lr_scheduler.step()
            else:
                for _ in range(config.start_epoch):
                    self.lr_scheduler.step()
            self.logger.logger.info(
                f'[LOAD] lr_scheduler at epoch {config.start_epoch}, '
                f'lr={self.optimizer.param_groups[0]["lr"]:.8f}')

    def _check_early_stopping(self, acc, epoch):
        if acc > self.best_acc:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= config.early_stopping_patience:
                self.early_stop_triggered = True
