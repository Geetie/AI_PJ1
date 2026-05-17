import os
import copy
import time
import logging
import psutil
import torch as t
from tqdm.auto import tqdm
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler
from datetime import datetime
from config import config, SCRIPT_DIR


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
        self.logger.info(f'Total parameters: {total_params:,}')
        self.logger.info(f'Trainable parameters: {trainable_params:,}')
        if t.cuda.is_available():
            gpu_name = t.cuda.get_device_name(0)
            gpu_props = t.cuda.get_device_properties(0)
            gpu_total = getattr(gpu_props, 'total_mem', getattr(gpu_props, 'total_memory', 0)) / (1024**3)
            self.logger.info(f'GPU: {gpu_name} ({gpu_total:.1f}GB)')
        self.logger.info(f'Batch size: {config.batch_size}')
        self.logger.info(f'Learning rate: {config.lr}')
        self.logger.info(f'Epochs: {config.epoches}')
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

    def log_epoch_end(self, epoch, train_acc, val_acc, lr, is_best=False, patience_counter=0):
        epoch_time = time.time() - self.epoch_start_time
        mins, secs = divmod(int(epoch_time), 60)
        msg = f'[EPOCH] Epoch={epoch + 1}/{config.epoches} ' \
              f'train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% ' \
              f'lr={lr:.8f} epoch_time={mins}m{secs:02d}s'
        if t.cuda.is_available():
            msg += f' gpu_peak={self.gpu_peak_mem:.1f}GB'
        msg += f' best={is_best} patience={patience_counter}/{config.early_stopping_patience}'
        self.logger.info(msg)

    def log_eval(self, acc, label='Evaluation'):
        self.logger.info(f'[EVAL] {label} acc={acc * 100:.2f}%')

    def log_early_stop(self, epoch, best_acc, best_path):
        self.logger.info(f'[STOP] Early stopping at epoch {epoch + 1}, '
                         f'best_acc={best_acc * 100:.2f}%, best_path={best_path}')

    def log_save(self, path):
        self.logger.info(f'[SAVE] Model saved to {path}')


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with t.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


class BaseTrainer:
    _checkpoint_prefix = 'model'
    _eval_label = 'Evaluation'

    def __init__(self):
        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        self.use_amp = self.device.type == 'cuda'
        self.best_acc = 0
        self.best_checkpoint_path = ''
        self.train_log = []
        self._current_epoch = 0
        self.patience_counter = 0
        self.early_stop_triggered = False
        self.logger = TrainingLogger()
        self._oom_retry_count = 0
        self._original_batch_size = config.batch_size
        self._stable_batch_size = None

    def _setup_optimizer(self, backbone_params, other_params):
        return SGD([
            {'params': backbone_params, 'lr': config.lr * config.backbone_lr_factor},
            {'params': other_params, 'lr': config.lr},
        ], momentum=config.momentum, weight_decay=config.weights_decay, nesterov=True)

    def _setup_scheduler(self):
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01,
                                    total_iters=config.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epoches - config.warmup_epochs,
                                             eta_min=1e-6)
        return SequentialLR(self.optimizer,
                            schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[config.warmup_epochs])

    def _setup_scaler(self):
        return GradScaler('cuda', enabled=self.use_amp)

    def _pre_epoch_hook(self, epoch):
        pass

    def _rebuild_dataloaders(self):
        pass

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _eval(self):
        raise NotImplementedError

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            if self.early_stop_triggered:
                self.logger.log_early_stop(epoch, self.best_acc, self.best_checkpoint_path)
                break

            self._current_epoch = epoch

            self._pre_epoch_hook(epoch)
            self.logger.log_epoch_start(epoch)
            try:
                train_acc = self._train_epoch(epoch)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and self._oom_retry_count < 3:
                    self._oom_retry_count += 1
                    old_bs = config.batch_size
                    t.cuda.empty_cache()
                    peak_gb = t.cuda.max_memory_allocated() / (1024**3)
                    gpu_props = t.cuda.get_device_properties(0)
                    total_gb = getattr(gpu_props, 'total_mem', getattr(gpu_props, 'total_memory', 0)) / (1024**3)
                    if peak_gb > 0 and total_gb > 0:
                        safe_ratio = (total_gb * 0.85) / peak_gb
                        config.batch_size = max(int(config.batch_size * safe_ratio), 16)
                    else:
                        config.batch_size = max(int(config.batch_size * 0.75), 16)
                    config.grad_accum_steps = max(self._original_batch_size // config.batch_size, 1)
                    self._stable_batch_size = None
                    self.logger.logger.warning(
                        f'[OOM] Reducing batch_size {old_bs} -> {config.batch_size}, '
                        f'peak={peak_gb:.1f}GB/{total_gb:.1f}GB, '
                        f'grad_accum_steps={config.grad_accum_steps} (retry {self._oom_retry_count})')
                    t.cuda.reset_peak_memory_stats()
                    self._rebuild_dataloaders()
                    continue
                else:
                    raise
            self._oom_retry_count = 0
            if self._stable_batch_size is None:
                self._stable_batch_size = config.batch_size
                self.logger.logger.info(f'[STABLE] batch_size={config.batch_size} confirmed stable')
            if config.batch_size < self._stable_batch_size:
                new_bs = min(config.batch_size + 8, self._stable_batch_size)
                if new_bs != config.batch_size:
                    config.batch_size = new_bs
                    config.grad_accum_steps = max(self._original_batch_size // config.batch_size, 1)
                    self.logger.logger.info(
                        f'[RECOVER] Increasing batch_size to {config.batch_size}, '
                        f'grad_accum_steps={config.grad_accum_steps}')
                    self._rebuild_dataloaders()
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            if (epoch + 1) % config.eval_interval == 0:
                acc = 0.0
                if self.val_loader is not None:
                    acc = self._eval()
                is_best = acc > self.best_acc
                self.train_log.append({
                    'epoch': epoch + 1,
                    'train_acc': train_acc,
                    'val_acc': acc * 100,
                    'lr': current_lr
                })
                self._check_early_stopping(acc, epoch)
                self.logger.log_epoch_end(epoch, train_acc, acc * 100, current_lr, is_best, self.patience_counter)
                if is_best:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(config.checkpoints,
                                             'epoch-%s-%d-acc-%.2f.pth' % (self._checkpoint_prefix, epoch + 1, acc * 100))
                    self.save_model(save_path, save_opt=True)
                    self.logger.log_save(save_path)
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path

    def save_model(self, save_path, save_opt=False, save_config=False):
        if self.ema is not None:
            dicts = {'model': self.ema.ema.state_dict()}
        else:
            dicts = {'model': self.model.state_dict()}
        dicts['model_type'] = self._model_type
        dicts['epoch'] = self._current_epoch + 1
        dicts['best_acc'] = self.best_acc
        dicts['best_checkpoint_path'] = self.best_checkpoint_path
        dicts['patience_counter'] = self.patience_counter
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()
            dicts['lr_scheduler'] = self.lr_scheduler.state_dict()
            dicts['scaler'] = self.scaler.state_dict()
        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}
        dicts['train_log'] = self.train_log
        t.save(dicts, save_path)

    def load_model(self, load_path, skip_load_weights=False, save_opt=False, save_config=False):
        dicts = t.load(load_path, map_location=self.device, weights_only=False)
        if not skip_load_weights:
            self.model.load_state_dict(dicts['model'])
            if self.ema is not None:
                self.ema.ema.load_state_dict(dicts['model'])
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
            for _ in range(config.start_epoch):
                self.lr_scheduler.step()

    def _check_early_stopping(self, acc, epoch):
        if acc > self.best_acc:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= config.early_stopping_patience:
                self.early_stop_triggered = True
