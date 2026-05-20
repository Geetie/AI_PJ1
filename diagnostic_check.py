import sys
sys.path.insert(0, 'Street_Character_Recognition')
import config

required_attrs = [
    'cls_loss_weight', 'aux_loss_weight', 'grad_accum_steps',
    'eval_batch_size', 'pin_memory', 'persistent_workers',
    'prefetch_factor', 'multiprocessing_context', 'optimizer_type',
    'scheduler_type', 'compile_mode', 'compile_dynamic',
    'compile_fullgraph', 'max_checkpoints', 'oom_headroom_ratio',
    'use_gradient_checkpoint', 'resume_weights_only'
]

required_constants = [
    'config', 'BASE_DIR', 'data_dir', 'IS_MODELSCOPE', 'GPU_PLATFORM',
    'TOTAL_VRAM_GB', 'IS_NVIDIA', 'IS_AMD', 'NUM_PHYSICAL_CORES',
    'COMPILE_AVAILABLE', 'print_env_info'
]

print("=" * 60)
print("Config Diagnostic Check")
print("=" * 60)

# 检查 config 类的属性
missing_attrs = []
for attr in required_attrs:
    if not hasattr(config.config, attr):
        missing_attrs.append(attr)
        print(f"❌ Missing: config.{attr}")
    else:
        value = getattr(config.config, attr)
        print(f"✅ config.{attr} = {value!r}")

print("\n" + "-" * 60)
print("Top-Level Constants Check")
print("-" * 60)

# 检查顶层常量
missing_constants = []
for name in required_constants:
    if not hasattr(config, name):
        missing_constants.append(name)
        print(f"❌ Missing: {name}")
    else:
        value = getattr(config, name)
        if callable(value):
            print(f"✅ {name} = {value.__name__}()")
        else:
            print(f"✅ {name} = {value!r}")

print("\n" + "=" * 60)

if missing_attrs or missing_constants:
    print(f"⚠️  Found issues!")
    if missing_attrs:
        print(f"Missing attributes: {', '.join(missing_attrs)}")
    if missing_constants:
        print(f"Missing constants: {', '.join(missing_constants)}")
    sys.exit(1)
else:
    print("✅ All configuration attributes and constants present!")
    
print("\n" + "=" * 60)
print("Additional Configuration Summary")
print("=" * 60)
print(f"batch_size: {config.config.batch_size}")
print(f"lr: {config.config.lr}")
print(f"epoches: {config.config.epoches}")
print(f"warmup_epochs: {config.config.warmup_epochs}")
print(f"cutmix_prob: {config.config.cutmix_prob}")
print(f"erase_prob: {config.config.erase_prob}")
print(f"aug_rotation_degrees: {config.config.aug_rotation_degrees}")
print(f"aug_blur_prob: {config.config.aug_blur_prob}")
print(f"ordering_loss_weight: {config.config.ordering_loss_weight}")
print(f"attn_supervision_weight: {config.config.attn_supervision_weight}")
print(f"dropout: {config.config.dropout}")
print(f"\nPlatform Info:")
print(f"  GPU Platform: {config.GPU_PLATFORM}")
print(f"  VRAM: {config.TOTAL_VRAM_GB:.1f} GB")
print(f"  NVIDIA: {config.IS_NVIDIA}")
print(f"  AMD: {config.IS_AMD}")
print(f"  ModelScope: {config.IS_MODELSCOPE}")
print(f"  Compile Available: {config.COMPILE_AVAILABLE}")