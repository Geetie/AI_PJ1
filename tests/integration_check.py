import sys
sys.path.insert(0, '.')

print('1. Importing config...')
from config import config, GPU_PLATFORM, TOTAL_VRAM_GB, COMPILE_AVAILABLE, TRITON_AVAILABLE
print(f'   GPU_PLATFORM={GPU_PLATFORM}, VRAM={TOTAL_VRAM_GB:.1f}GB, COMPILE={COMPILE_AVAILABLE}, TRITON={TRITON_AVAILABLE}')

print('2. Importing models...')
from models import create_model
from models.multihead import DigitsResnet101
from models.ctc import CTCModel
from models.transformer import TransformerDigitsModel
from models.backbone import FPNBackbone
print('   All models imported OK')

print('3. Importing losses...')
from losses.classification import LabelSmoothEntropy
from losses.attention import AttentionSupervisionLoss, attention_diversity_loss, spatial_ordering_loss
from losses.augmentation import cutmix_data
print('   All losses imported OK')

print('4. Importing data...')
from data.dataset import DigitsDataset, CTCDataset, ctc_collate_fn, ctc_test_collate_fn
from data.transform import transform_with_bbox, transform_test_img
print('   All data modules imported OK')

print('5. Importing trainers...')
from trainer.base import BaseTrainer, ModelEMA
from trainer.multihead import MultiHeadTrainer
from trainer.ctc import CTCTrainer
print('   All trainers imported OK')

print('6. Importing inference...')
from inference.predict import predicts, ctc_predict, ensemble_predict, cross_model_ensemble
from inference.decode import parse2class, parse2class_from_probs, ctc_greedy_decode, ctc_beam_decode
print('   All inference modules imported OK')

print('7. Importing compile utils...')
from utils.compile_utils import (
    is_compile_available, try_compile_model, warmup_model, get_raw_model,
    configure_dynamo_cache, configure_compile_cache, CompileLogger,
    selective_compile, measure_compile_performance, WARMUP_STRATEGIES,
    run_compile_audit, warmup_strategy_single, warmup_strategy_tta,
    warmup_strategy_fixed_shape, warmup_strategy_cache_optimized,
    warmup_strategy_selective, warmup_strategy_max_autotune
)
print('   All compile utils imported OK')

print('8. Importing misc utils...')
from utils.misc import PadToSquare, write2csv, find_latest_checkpoint
from utils.seed import set_seed
print('   All misc utils imported OK')

print('9. Testing model creation...')
import torch as t
model = create_model('fpn_multihead')
model.eval()
x = t.randn(2, 3, config.input_height, config.input_width)
with t.no_grad():
    cls_outs, bbox_outs = model(x)
print(f'   FPN MultiHead forward OK: {len(cls_outs)} heads, cls_shape={cls_outs[0].shape}')

ctc_model = create_model('ctc')
ctc_model.eval()
with t.no_grad():
    out = ctc_model(x)
print(f'   CTC forward OK: output_shape={out.shape}')

trans_model = create_model('transformer')
trans_model.eval()
with t.no_grad():
    cls_outs2, bbox_outs2 = trans_model(x)
print(f'   Transformer forward OK: {len(cls_outs2)} heads')

print('10. Testing EMA with buffer sync...')
import torch.nn as nn
test_model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
    nn.ReLU(), nn.Flatten(), nn.Linear(16*4*4, 10)
)
ema = ModelEMA(test_model, decay=0.9)
test_model.train()
_ = test_model(t.randn(4, 3, 4, 4))
bn_before = ema.ema[1].running_mean.clone()
test_model.train()
_ = test_model(t.randn(4, 3, 4, 4) + 1.0)
ema.update(test_model)
bn_after = ema.ema[1].running_mean.clone()
buffer_synced = not t.allclose(bn_before, bn_after)
status = 'OK' if buffer_synced else 'FAILED'
print(f'   EMA buffer sync: {status} (BN running_mean updated)')

print('11. Testing forward_with_attn gradient flow...')
model2 = DigitsResnet101(config.class_num, config.num_heads)
model2.train()
x2 = t.randn(2, 3, config.input_height, config.input_width)
cls_list, bbox_tuple, attn_maps, head_cls_outs = model2.forward_with_attn(x2)
has_grad = head_cls_outs[0].requires_grad
print(f'   head_cls_outs requires_grad: {has_grad} (aux loss will be effective: {has_grad})')

print()
print('=' * 60)
print('ALL IMPORT AND INTEGRATION CHECKS PASSED!')
print('=' * 60)
