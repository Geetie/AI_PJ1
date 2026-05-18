import gc
import time
import threading
import torch as t
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config
from data.dataset import CTCDataset, ctc_collate_fn, ctc_test_collate_fn
from models.ctc import CTCModel
from trainer.base import BaseTrainer, ModelEMA
from inference.decode import ctc_greedy_decode, ctc_beam_decode
from utils.compile_utils import (
    try_compile_model, warmup_model, get_raw_model, CompileLogger, configure_compile_cache
)


class CTCTrainer(BaseTrainer):
    _checkpoint_prefix = 'ctc'
    _eval_label = 'CTC Evaluation'

    def __init__(self, val=True):
        super().__init__()
        print(f'CTC Model - Using device: {self.device}')
        self._model_type = 'ctc'
        self.train_set = CTCDataset(mode='train', aug=True,
                                    input_size=(config.input_height, config.input_width))
        self.train_loader = self._make_loader(self.train_set, batch_size=config.batch_size,
                                              shuffle=True, drop_last=True,
                                              collate_fn=ctc_collate_fn)
        if val:
            self.val_set = CTCDataset(mode='val', aug=False,
                                      input_size=(config.input_height, config.input_width))
            self.val_loader = self._make_loader(self.val_set, batch_size=config.eval_batch_size,
                                                shuffle=False, drop_last=False,
                                                collate_fn=ctc_collate_fn)
        else:
            self.val_loader = None

        self.model = CTCModel(num_classes=config.class_num).to(self.device)
        self.ema = ModelEMA(self.model, decay=config.ema_decay)
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
            self.load_model(config.pretrained, save_opt=False)

        self._gpu_warmup()

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
                                              collate_fn=ctc_collate_fn)
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

        raw_model = get_raw_model(self.model)
        saved_state = None
        try:
            saved_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        except Exception:
            saved_state = None

        warmup_start = time.time()

        try:
            with self._compile_logger.phase('warmup_ctc_forward'):
                dummy = t.randn(warmup_bs, 3, config.input_height, config.input_width, device=self.device)
                with t.no_grad(), autocast(self.device.type, enabled=self.use_amp):
                    _ = self.model(dummy)
                t.cuda.synchronize()
                del dummy
                t.cuda.empty_cache()
            warmup_time = time.time() - warmup_start
            self._compile_logger.log_warmup_summary(warmup_time, 1, 1)
            print(f'[WARMUP-CTC] Forward warmup completed')
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

        if saved_state is not None:
            try:
                raw_model.load_state_dict(saved_state)
                if self.ema is not None:
                    self.ema.ema.load_state_dict(saved_state)
                del saved_state
                t.cuda.empty_cache()
                print(f'[WARMUP-CTC] Model weights restored after warmup')
            except Exception:
                del saved_state

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
            self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.use_amp):
                log_probs = self.model(img)
                T = log_probs.size(0)
                B = log_probs.size(1)
                input_lengths = t.full((B,), T, dtype=t.long)
                target_lengths = lengths.to(self.device)
                loss = self.criterion(log_probs, label_concat, input_lengths, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
        model = self.ema.ema if self.ema is not None else self.model
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
