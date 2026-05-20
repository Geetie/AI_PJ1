import sys
sys.path.insert(0, 'Street_Character_Recognition')
from config import config

required_attrs = [
    'cls_loss_weight', 'aux_loss_weight', 'grad_accum_steps',
    'eval_batch_size', 'pin_memory', 'persistent_workers',
    'prefetch_factor', 'multiprocessing_context', 'optimizer_type',
    'scheduler_type', 'compile_mode', 'compile_dynamic',
    'compile_fullgraph', 'max_checkpoints', 'oom_headroom_ratio',
    'use_gradient_checkpoint', 'resume_weights_only'
]

print("=" * 60)
print("Config Diagnostic Check")
print("=" * 60)

missing = []
for attr in required_attrs:
    if not hasattr(config, attr):
        missing.append(attr)
        print(f"❌ Missing: config.{attr}")
    else:
        value = getattr(config, attr)
        print(f"✅ config.{attr} = {value!r}")

print("\n" + "=" * 60)

if missing:
    print(f"⚠️  Found {len(missing)} missing attributes!")
    print(f"Missing list: {', '.join(missing)}")
    sys.exit(1)
else:
    print("✅ All configuration attributes present!")
    
print("\n" + "=" * 60)
print("Additional Configuration Summary")
print("=" * 60)
print(f"batch_size: {config.batch_size}")
print(f"lr: {config.lr}")
print(f"epoches: {config.epoches}")
print(f"warmup_epochs: {config.warmup_epochs}")
print(f"cutmix_prob: {config.cutmix_prob}")
print(f"erase_prob: {config.erase_prob}")
print(f"aug_rotation_degrees: {config.aug_rotation_degrees}")
print(f"aug_blur_prob: {config.aug_blur_prob}")
print(f"ordering_loss_weight: {config.ordering_loss_weight}")
print(f"attn_supervision_weight: {config.attn_supervision_weight}")
print(f"dropout: {config.dropout}")