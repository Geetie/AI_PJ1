import time
import torch as t
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config, NUM_WORKERS
from data.dataset import CTCDataset, ctc_collate_fn, ctc_test_collate_fn
from models.ctc import CTCModel
from trainer.base import BaseTrainer, ModelEMA
from inference.decode import ctc_greedy_decode, ctc_beam_decode


class CTCTrainer(BaseTrainer):
    _checkpoint_prefix = 'ctc'
    _eval_label = 'CTC Evaluation'

    def __init__(self, val=True):
        super().__init__()
        print(f'CTC Model - Using device: {self.device}')
        self._model_type = 'ctc'
        self.train_set = CTCDataset(mode='train', aug=True,
                                    input_size=(config.input_height, config.input_width))
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True,
                                       persistent_workers=NUM_WORKERS > 0,
                                       drop_last=True, prefetch_factor=2,
                                       collate_fn=ctc_collate_fn)
        if val:
            self.val_set = CTCDataset(mode='val', aug=False,
                                      input_size=(config.input_height, config.input_width))
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size,
                                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                                         persistent_workers=NUM_WORKERS > 0,
                                         prefetch_factor=2, collate_fn=ctc_collate_fn)
        else:
            self.val_loader = None

        self.model = CTCModel(num_classes=config.class_num).to(self.device)
        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        self.criterion = nn.CTCLoss(blank=10, zero_infinity=True)

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

    def _rebuild_dataloaders(self):
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True,
                                       persistent_workers=NUM_WORKERS > 0,
                                       drop_last=True, prefetch_factor=2,
                                       collate_fn=ctc_collate_fn)
        if self.val_loader is not None:
            self.val_loader = DataLoader(self.val_set, batch_size=config.batch_size,
                                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                                         persistent_workers=NUM_WORKERS > 0, prefetch_factor=2,
                                         collate_fn=ctc_collate_fn)

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
        corrects = 0
        total = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for img, label_concat, lengths in tbar:
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
        t.cuda.empty_cache()
        self.model.train()
        return corrects / max(total, 1)
