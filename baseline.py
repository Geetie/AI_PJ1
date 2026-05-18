import os
import sys

from utils.seed import set_seed
from config import config, BASE_DIR
from data.download import download_dataset
from utils.misc import find_latest_checkpoint
from trainer.multihead import MultiHeadTrainer
from trainer.ctc import CTCTrainer
from inference.predict import predicts, ctc_predict


set_seed(42)

import torch as t
if t.cuda.is_available():
    try:
        t.set_float32_matmul_precision('high')
        print('TF32 matmul precision enabled')
    except Exception:
        print('TF32 matmul precision not supported on this GPU')
    try:
        t.backends.cudnn.allow_tf32 = True
        t.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass


if __name__ == '__main__':
    download_dataset()
    latest_checkpoint = find_latest_checkpoint(config.checkpoints)
    if latest_checkpoint:
        print(f'Found latest checkpoint: {latest_checkpoint}')
        config.pretrained = latest_checkpoint
    else:
        print('No checkpoint found, starting from scratch.')

    model_type = sys.argv[1] if len(sys.argv) > 1 else config.model_type

    if model_type == 'ctc':
        print('\n========== Training CTC Model ==========')
        ctc_trainer = CTCTrainer()
        ctc_trainer.train()

        print('\n=== CTC Evaluation ===')
        ctc_acc = ctc_trainer._eval()
        print(f'CTC Best Val Acc: {ctc_trainer.best_acc * 100:.2f}')

        ctc_predict(ctc_trainer.best_checkpoint_path,
                    os.path.join(BASE_DIR, 'result_ctc.csv'), use_tta=False)
    elif model_type == 'transformer':
        print('\n========== Training Transformer Model ==========')
        trainer = MultiHeadTrainer(model_type='transformer')
        trainer.train()

        print('\n=== Standard Evaluation ===')
        val_acc = trainer._eval()
        print(f'Best Val Acc: {trainer.best_acc * 100:.2f}')

        print('\n=== Detailed Evaluation ===')
        trainer.eval_detailed()

        print('\n=== TTA Evaluation ===')
        tta_acc = trainer.eval_tta()
        if tta_acc > trainer.best_acc:
            print(f'TTA improved acc: {tta_acc * 100:.2f} > {trainer.best_acc * 100:.2f}')

        predicts(trainer.best_checkpoint_path, os.path.join(BASE_DIR, 'result.csv'), use_tta=True, model_type='transformer')
    else:
        print('\n========== Training FPN Multi-Head Model ==========')
        trainer = MultiHeadTrainer(model_type='fpn_multihead')
        trainer.train()

        print('\n=== Standard Evaluation ===')
        val_acc = trainer._eval()
        print(f'Best Val Acc: {trainer.best_acc * 100:.2f}')

        print('\n=== Detailed Evaluation ===')
        trainer.eval_detailed()

        print('\n=== TTA Evaluation ===')
        tta_acc = trainer.eval_tta()
        if tta_acc > trainer.best_acc:
            print(f'TTA improved acc: {tta_acc * 100:.2f} > {trainer.best_acc * 100:.2f}')

        predicts(trainer.best_checkpoint_path, os.path.join(BASE_DIR, 'result.csv'), use_tta=True, model_type='fpn_multihead')
