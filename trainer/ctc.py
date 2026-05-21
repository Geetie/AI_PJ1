import gc
import random
import time
import threading
import numpy as np
import torch as t
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config
from data.dataset import CTCDataset, ctc_collate_fn, ctc_test_collate_fn
from models.ctc import CTCModel
from trainer.base import BaseTrainer, ModelEMA, _load_state_dict_compat
from inference.decode import ctc_greedy_decode, ctc_beam_decode
from utils.compile_utils import (
    try_compile_model, warmup_model, get_raw_model, CompileLogger, configure_compile_cache
)
from utils.seed import make_epoch_generator


class CTCTrainer(BaseTrainer):
    _checkpoint_prefix = 'ctc'
    _eval_label = 'CTC Evaluation'

    def __init__(self, val=True):
        super().__init__()
        print(f'CTC Model - Using device: {self.device}')
        self._model_type = 'ctc'

        self._gpu_sanity_check()

        self.model = CTCModel(num_classes=config.class_num).to(self.device)

        if config.pretrained is not None:
            ckpt = t.load(config.pretrained, map_location=self.device, weights_only=False)
            if config.resume_weights_only:
                _, skipped = _load_state_dict_compat(self.model, ckpt['model'])
                if skipped:
                    print(f'[CKPT] resume_weights_only: skipped {len(skipped)} shape-mismatched keys')
            elif 'train_model' in ckpt:
                _, skipped = _load_state_dict_compat(self.model, ckpt['train_model'])
                print(f'[CKPT] Loaded train_model weights (not EMA) for continued training')
                if skipped:
                    print(f'[CKPT] Skipped {len(skipped)} shape-mismatched keys in train_model')
            else:
                _, skipped = _load_state_dict_compat(self.model, ckpt['model'])
                if skipped:
                    print(f'[CKPT] Skipped {len(skipped)} shape-mismatched keys')
            print(f'Load model from {config.pretrained}')

        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        if config.pretrained is not None and not config.resume_weights_only:
            if 'model' in ckpt:
                try:
                    _, skipped = _load_state_dict_compat(self.ema.ema, ckpt['model'])
                    self.logger.logger.info('[CKPT] Restored EMA shadow model from checkpoint')
                    if skipped:
                        self.logger.logger.info(f'[CKPT] EMA shadow: skipped {len(skipped)} shape-mismatched keys')
                except Exception as e:
                    self.logger.logger.warning(f'[CKPT] Failed to restore EMA shadow model: {e}')
        self.criterion = nn.CTCLoss(blank=10, zero_infinity=True)

        self._compile_logger = CompileLogger.get_instance()
        if config.use_torch_compile and t.cuda.is_available():
            configure_compile_cache()
            self._compile_logger.log_compile_config({
                'use_torch_compile': True,
                'compile_mode': config.compile_mode,
                'compile_dynamic': config.compile_dynamic,
                'compile_fullgraph': config.compile_fullgraph,
                'model_type': 'ctc',
                'batch_size': config.batch_size,
                'input_size': f'{config.input_height}x{config.input_width}',
            })
            with self._compile_logger.phase('compile_ctc'):
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
            if 'best_checkpoint_path' in ckpt:
                self.best_checkpoint_path = ckpt['best_checkpoint_path']
            if config.resume_weights_only:
                config.start_epoch = 0
                self._current_epoch = 0
                self.patience_counter = 0
                print(f'[CKPT] resume_weights_only: best_acc={self.best_acc * 100:.2f}%, '
                      f'starting from epoch 1 with fresh optimizer')
            else:
                if 'epoch' in ckpt:
                    config.start_epoch = ckpt['epoch']
                    self._current_epoch = ckpt['epoch']
                if 'train_log' in ckpt:
                    self.train_log = ckpt['train_log']
                if 'patience_counter' in ckpt:
                    self.patience_counter = ckpt['patience_counter']
            if not config.resume_weights_only and 'opt' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['opt'])
                    self.logger.logger.info('Restored optimizer state from checkpoint')
                except Exception as e:
                    self.logger.logger.warning(f'Failed to restore optimizer: {e}. Using new optimizer.')
            if not config.resume_weights_only and 'lr_scheduler' in ckpt:
                try:
                    self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                    self.logger.logger.info('Restored lr_scheduler state from checkpoint')
                except Exception as e:
                    self.logger.logger.warning(f'Failed to restore lr_scheduler: {e}. Using new scheduler.')
            if not config.resume_weights_only and 'scaler' in ckpt:
                try:
                    self.scaler.load_state_dict(ckpt['scaler'])
                    self.logger.logger.info('Restored scaler state from checkpoint')
                except Exception as e:
                    self.logger.logger.warning(f'Failed to restore scaler: {e}. Using new scaler.')
            self.logger.logger.info(f'Restored best_acc: {self.best_acc * 100:.2f}%, '
                                   f'start_epoch: {config.start_epoch}, '
                                   f'patience: {self.patience_counter}/{config.early_stopping_patience}')

        self._gpu_warmup()

        self._base_seed = 42
        self._train_generator = make_epoch_generator(self._base_seed, epoch=0)

        self.train_set = CTCDataset(mode='train', aug=True,
                                    input_size=(config.input_height, config.input_width))
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True,
                                              collate_fn=ctc_collate_fn,
                                              generator=self._train_generator)
        if val:
            self.val_set = CTCDataset(mode='val', aug=False,
                                      input_size=(config.input_height, config.input_width))
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False,
                                                collate_fn=ctc_collate_fn)
        else:
            self.val_loader = None

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
        epoch_seed = self._base_seed + epoch * 1000
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        t.manual_seed(epoch_seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(epoch_seed)

        self._cleanup_dataloader(self.train_loader)
        self._train_generator = make_epoch_generator(self._base_seed, epoch=epoch)
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True,
                                              collate_fn=ctc_collate_fn,
                                              generator=self._train_generator,
                                              persistent_override=False)
        self.logger.logger.info(
            f'[EPOCH-PRE] epoch={epoch+1} lr={self.optimizer.param_groups[0]["lr"]:.8f} '
            f'generator_seed={self._base_seed + epoch}')

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
                                              collate_fn=ctc_collate_fn,
                                              generator=self._train_generator)
        if self.val_loader is not None:
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False,
                                                collate_fn=ctc_collate_fn)

    def _gpu_warmup(self):
        if not t.cuda.is_available():
            return
        warmup_bs = min(config.batch_size, 16)
        compile_info = " (torch.compile)" if config.use_torch_compile else ""
        print(f'[WARMUP-CTC] Starting GPU warmup with bs={warmup_bs}{compile_info}...')

        warmup_start = time.time()

        try:
            with self._compile_logger.phase('warmup_ctc_inference'):
                dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                with t.no_grad(), autocast(self.device.type, enabled=self.use_amp):
                    _ = self.model(dummy)
                t.cuda.synchronize()
                del dummy
                t.cuda.empty_cache()
            warmup_time = time.time() - warmup_start
            self._compile_logger.log_warmup_summary(warmup_time, 1, 1)
            print(f'[WARMUP-CTC] Inference warmup completed in {warmup_time:.1f}s')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                t.cuda.empty_cache()
                try:
                    warmup_bs = max(warmup_bs // 4, 4)
                    dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                    with t.no_grad():
                        _ = self.model(dummy)
                    t.cuda.synchronize()
                    del dummy
                    t.cuda.empty_cache()
                    print(f'[WARMUP-CTC] Fallback warmup completed with bs={warmup_bs}')
                except Exception as e2:
                    print(f'[WARMUP-CTC] Fallback warmup also failed: {e2}')
                    t.cuda.empty_cache()
            else:
                print(f'[WARMUP-CTC] GPU warmup failed: {e}')
                t.cuda.empty_cache()
        except Exception as e:
            err_str = str(e).lower()
            if config.use_torch_compile and ('compile' in err_str or 'triton' in err_str or 'inductor' in err_str):
                print(f'[WARMUP-CTC] torch.compile failed during warmup: {e}')
                print(f'[WARMUP-CTC] Disabling torch.compile and falling back to eager mode')
                self._compile_logger.logger.warning(
                    f'[WARMUP-CTC] torch.compile failed, falling back to eager: {e}')
                config.use_torch_compile = False
                self.model = get_raw_model(self.model)
                t.cuda.empty_cache()
            else:
                print(f'[WARMUP-CTC] GPU warmup failed: {e}')
                t.cuda.empty_cache()

        self._compile_logger.log_dynamo_stats()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        corrects = 0
        total = 0
        batch_start = time.time()
        tbar = tqdm(self.train_loader)

        for i, (img, label_concat, lengths) in enumerate(tbar):
            img = img.to(self.device)
            label_concat = label_concat.to(self.device)

            if i % config.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.use_amp):
                log_probs = self.model(img)
                T = log_probs.size(0)
                B = log_probs.size(1)
                input_lengths = t.full((B,), T, dtype=t.long)
                target_lengths = lengths.to(self.device)
                loss = self.criterion(log_probs, label_concat, input_lengths, target_lengths)
                loss = loss / config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == len(tbar):
                self.scaler.unscale_(self.optimizer)
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.grad_clip_max_norm)
                step_result = self.scaler.step(self.optimizer)
                self.scaler.update()
                if step_result is not None:
                    self.ema.update(self.model)

            total_loss += loss.item()
            batch_time = time.time() - batch_start
            batch_start = time.time()

            if i % config.train_eval_interval == 0:
                pred_strs = ctc_greedy_decode(log_probs.detach())
                offset = 0
                for b in range(B):
                    gt = label_concat[offset:offset + lengths[b]].tolist()
                    offset += lengths[b]
                    if pred_strs[b] == gt:
                        corrects += 1
                    total += 1

            tbar.set_description(
                'Epoch %d, loss: %.3f, acc: %.3f' % (epoch + 1, total_loss / (i + 1), corrects * 100 / max(total, 1)))
            if (i + 1) % config.print_interval == 0:
                acc_str = f'acc={corrects * 100 / max(total, 1):.2f}%'
                self.logger.log_batch(epoch, i, len(self.train_loader),
                                      total_loss / (i + 1), self.optimizer.param_groups[0]['lr'],
                                      acc_str, batch_time=batch_time)

        return corrects * 100 / max(total, 1)

    def _eval(self):
        if self.ema is not None:
            model = self.ema.to_device(self.device)
        else:
            model = self._get_raw_model()
        model.eval()
        eval_bs = config.eval_batch_size
        max_retries = 3
        for attempt in range(max_retries):
            corrects = 0
            total = 0
            oom_hit = False
            with t.no_grad():
                tbar = tqdm(self.val_loader)
                for img, label_concat, lengths in tbar:
                    try:
                        img = img.to(self.device)
                        log_probs = model(img)
                        pred_strs = ctc_beam_decode(log_probs.cpu())
                        offset = 0
                        for b in range(len(lengths)):
                            gt = label_concat[offset:offset + lengths[b]].tolist()
                            offset += lengths[b]
                            if pred_strs[b] == gt:
                                corrects += 1
                            total += 1
                        tbar.set_description('CTC Val Acc: %.2f' % (corrects * 100 / max(total, 1)))
                        del img, log_probs
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
                                    shuffle=False, drop_last=False,
                                    collate_fn=ctc_collate_fn)
                            break
                        raise
            if oom_hit:
                continue
            break
        t.cuda.empty_cache()
        self.model.train()
        return corrects / max(total, 1)
