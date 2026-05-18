import os
import sys
import multiprocessing
import torch as t

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/mnt/workspace' if os.path.exists('/mnt/workspace') else SCRIPT_DIR

IS_MODELSCOPE = os.path.exists('/mnt/workspace')

from utils.platform import (
    is_amd_rocm, is_nvidia_cuda, get_platform,
    get_total_vram_gb, get_gfx_arch, get_rocm_version,
    is_triton_available, is_compile_available, get_precision_config,
    get_cache_dirs, NUM_PHYSICAL_CORES as _NUM_PHYSICAL_CORES,
)

GPU_PLATFORM = get_platform()
TOTAL_VRAM_GB = get_total_vram_gb()
NUM_PHYSICAL_CORES = _NUM_PHYSICAL_CORES

IS_AMD = is_amd_rocm()
IS_NVIDIA = is_nvidia_cuda()

if IS_AMD:
    os.environ.setdefault('MIOPEN_DISABLE_CACHE', '0')
    os.environ.setdefault('MIOPEN_FIND_MODE', '2')
    cache_dirs = get_cache_dirs()
    os.environ.setdefault('MIOPEN_USER_DB_PATH', cache_dirs['miopen'])
    os.environ.setdefault('TRITON_CACHE_DIR', cache_dirs['triton'])
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dirs['inductor'])
    if IS_MODELSCOPE:
        os.environ.setdefault('TORCH_USE_HIP_DSA', '1')
        os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')

_GFX_VERSION_MAP = {
    'gfx900': '9.0.0',
    'gfx906': '9.0.6',
    'gfx908': '9.0.8',
    'gfx90a': '9.0.10',
    'gfx940': '9.4.0',
    'gfx941': '9.4.1',
    'gfx942': '9.4.2',
    'gfx1030': '10.3.0',
    'gfx1031': '10.3.0',
    'gfx1032': '10.3.0',
    'gfx1100': '11.0.0',
    'gfx1101': '11.0.0',
    'gfx1102': '11.0.0',
    'gfx1150': '11.5.0',
    'gfx1151': '11.5.1',
    'gfx1200': '12.0.0',
    'gfx1201': '12.0.1',
}


def _should_override_gfx(gfx_arch):
    if gfx_arch is None:
        return False
    try:
        supported = t.cuda.get_arch_list()
        if supported and gfx_arch in supported:
            return False
    except Exception:
        pass
    return gfx_arch in _GFX_VERSION_MAP


def _configure_hsa_override():
    if 'HSA_OVERRIDE_GFX_VERSION' in os.environ:
        return
    if not IS_MODELSCOPE:
        return
    if not IS_AMD:
        return
    gfx_arch = get_gfx_arch()
    if gfx_arch is not None:
        if not _should_override_gfx(gfx_arch):
            print(f'[HSA] GPU arch {gfx_arch} is natively supported by PyTorch, '
                  f'skipping HSA_OVERRIDE_GFX_VERSION')
            return
        version = _GFX_VERSION_MAP.get(gfx_arch)
        if version:
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = version
            print(f'[HSA] Set HSA_OVERRIDE_GFX_VERSION={version} for {gfx_arch}')
            return
    try:
        if t.cuda.is_available():
            props = t.cuda.get_device_properties(0)
            vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
            if vram > 100:
                os.environ['HSA_OVERRIDE_GFX_VERSION'] = '9.0.10'
                print(f'[HSA] Large VRAM GPU ({vram:.0f}GB), '
                      f'set HSA_OVERRIDE_GFX_VERSION=9.0.10 as fallback')
    except Exception:
        pass


_configure_hsa_override()


TRITON_AVAILABLE = is_triton_available()
COMPILE_AVAILABLE = is_compile_available()


ROCM_VERSION = get_rocm_version()


def _check_rocm_compatibility():
    if not IS_AMD:
        return
    if ROCM_VERSION is not None:
        major, minor = ROCM_VERSION
        if major == 6 and minor == 2:
            print(f'[ROCM-WARN] ROCm 6.2.x detected: Conv2d with certain out_channels '
                  f'may crash (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION). '
                  f'Consider upgrading to ROCm 6.3+.')
        if major == 7 and minor == 0:
            gfx = get_gfx_arch()
            if gfx and gfx.startswith('gfx12'):
                print(f'[ROCM-WARN] ROCm 7.0 + RDNA4 ({gfx}) detected: random memory access '
                      f'faults may occur due to expert scheduling mode. '
                      f'Consider adding amdgpu.cwsr_enable=0 to kernel boot params.')
    pytorch_ver = t.__version__
    if pytorch_ver.startswith('2.8.0'):
        try:
            import rocblas
        except ImportError:
            pass
        try:
            test_a = t.randn(2, 2, device='cuda')
            test_b = t.randn(2, 2, device='cuda')
            _ = test_a @ test_b
            t.cuda.synchronize()
            del test_a, test_b
        except Exception:
            print(f'[ROCM-WARN] PyTorch 2.8.0+rocm: matmul segfault detected, '
                  f'possible rocBLAS packaging bug. '
                  f'Consider using nightly wheels: pip install --pre torch '
                  f'--index-url https://download.pytorch.org/whl/nightly/rocm6.4')


_check_rocm_compatibility()


def _configure_rocm_blas_backend():
    if not IS_AMD:
        return
    try:
        pytorch_ver = t.__version__
        ver_parts = pytorch_ver.split('.')[:2]
        major_minor = tuple(int(x.split('+')[0]) for x in ver_parts)
    except Exception:
        return
    if major_minor < (2, 7):
        return
    try:
        if not hasattr(t.backends.cuda, 'preferred_blas_library'):
            return
        current = t.backends.cuda.preferred_blas_library()
        if current == 'cublaslt':
            t.backends.cuda.preferred_blas_library("cublas")
            print(f'[BLAS] Switched BLAS backend from hipBLASLt to rocBLAS '
                  f'(PyTorch {pytorch_ver}+ROCm: hipBLASLt lacks fp16_alt_impl '
                  f'for backward pass, which can silently flush subnormal gradients to zero)')
    except Exception as e:
        print(f'[BLAS] Failed to configure BLAS backend: {e}')


class GPUProfile:
    platform = 'any'
    batch_size = 32
    eval_batch_size = 64
    num_workers = 4
    prefetch_factor = 2
    persistent_workers = True
    multiprocessing_context = None
    input_height = 384
    input_width = 384
    resize_size = 416
    fc_hidden = 1024
    grad_accum_steps = 1
    use_torch_compile = False
    compile_mode = 'default'
    compile_dynamic = False
    compile_fullgraph = False
    use_gradient_checkpoint = True
    oom_headroom_ratio = 0.15
    max_checkpoints = 3
    pin_memory = True
    tta_sizes = [288, 320, 352, 384, 416]
    lr = 1e-3
    backbone_lr_factor = 0.1
    warmup_epochs = 5
    dropout = 0.2
    ema_decay = 0.999
    aux_loss_weight = 0.3
    bbox_loss_weight = 5.0
    attn_diversity_weight = 0.1
    attn_supervision_weight = 2.0
    ordering_loss_weight = 2.0
    multiscale_feat_dim = 512
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_feat_dim = 256
    transformer_heads = 4
    transformer_layers = 4
    head_interaction_layers = 2
    num_attn_channels = 8
    cutmix_alpha = 1.0
    cutmix_prob = 0.5
    erase_prob = 0.2
    smooth = 0.1
    aug_rotation_degrees = 10
    aug_blur_prob = 0.15
    roi_gt_prob = 0.8


class CPUProfile(GPUProfile):
    platform = 'cpu'
    batch_size = 32
    eval_batch_size = 32
    num_workers = 0
    prefetch_factor = 2
    input_height = 224
    input_width = 224
    resize_size = 256
    fc_hidden = 512
    grad_accum_steps = 4
    use_torch_compile = False
    pin_memory = False
    tta_sizes = [224, 256]


class A100Profile(GPUProfile):
    platform = 'nvidia_cuda'
    batch_size = 64
    eval_batch_size = 96
    num_workers = 6
    prefetch_factor = 2
    input_height = 384
    input_width = 384
    resize_size = 416
    fc_hidden = 1024
    grad_accum_steps = 2
    use_torch_compile = False
    compile_mode = 'default'
    compile_dynamic = False
    use_gradient_checkpoint = True
    oom_headroom_ratio = 0.15
    max_checkpoints = 3
    pin_memory = True
    tta_sizes = [384, 416]
    lr = 1e-3
    backbone_lr_factor = 0.1
    warmup_epochs = 5
    dropout = 0.2
    ema_decay = 0.999
    aux_loss_weight = 0.3
    bbox_loss_weight = 5.0
    attn_diversity_weight = 0.1
    attn_supervision_weight = 2.0
    ordering_loss_weight = 2.0
    multiscale_feat_dim = 512
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_feat_dim = 256
    transformer_heads = 4
    transformer_layers = 4
    head_interaction_layers = 2
    num_attn_channels = 8
    cutmix_alpha = 1.0
    cutmix_prob = 0.5
    erase_prob = 0.2
    smooth = 0.1
    aug_rotation_degrees = 10
    aug_blur_prob = 0.15
    roi_gt_prob = 0.8


class AMDMidProfile(GPUProfile):
    platform = 'amd_rocm'
    batch_size = 96
    eval_batch_size = 128
    num_workers = 12
    prefetch_factor = 3
    persistent_workers = True
    input_height = 384
    input_width = 384
    resize_size = 416
    fc_hidden = 1024
    grad_accum_steps = 2
    use_torch_compile = True
    compile_mode = 'default'
    compile_dynamic = False
    use_gradient_checkpoint = True
    oom_headroom_ratio = 0.12
    pin_memory = False
    tta_sizes = [320, 352, 384, 416]
    lr = 4e-3
    backbone_lr_factor = 0.1
    warmup_epochs = 6
    dropout = 0.15
    ema_decay = 0.9995
    aux_loss_weight = 0.35
    bbox_loss_weight = 5.0
    attn_diversity_weight = 0.1
    attn_supervision_weight = 2.0
    ordering_loss_weight = 2.0
    multiscale_feat_dim = 512
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_feat_dim = 256
    transformer_heads = 4
    transformer_layers = 4
    head_interaction_layers = 2
    num_attn_channels = 8
    cutmix_alpha = 1.0
    cutmix_prob = 0.5
    erase_prob = 0.15
    smooth = 0.1
    aug_rotation_degrees = 10
    aug_blur_prob = 0.15
    roi_gt_prob = 0.8


class AMDLargeProfile(GPUProfile):
    platform = 'amd_rocm'
    batch_size = 192
    eval_batch_size = 256
    num_workers = 20
    prefetch_factor = 4
    persistent_workers = True
    input_height = 416
    input_width = 416
    resize_size = 448
    fc_hidden = 1536
    grad_accum_steps = 1
    use_torch_compile = True
    compile_mode = 'default'
    compile_dynamic = False
    use_gradient_checkpoint = True
    oom_headroom_ratio = 0.08
    max_checkpoints = 3
    pin_memory = False
    tta_sizes = [384, 416]
    lr = 4e-3
    backbone_lr_factor = 0.1
    warmup_epochs = 6
    dropout = 0.15
    ema_decay = 0.9995
    aux_loss_weight = 0.35
    bbox_loss_weight = 5.0
    attn_diversity_weight = 0.1
    attn_supervision_weight = 2.0
    ordering_loss_weight = 2.0
    multiscale_feat_dim = 512
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_feat_dim = 256
    transformer_heads = 4
    transformer_layers = 4
    head_interaction_layers = 2
    num_attn_channels = 8
    cutmix_alpha = 1.0
    cutmix_prob = 0.5
    erase_prob = 0.15
    smooth = 0.1
    aug_rotation_degrees = 10
    aug_blur_prob = 0.15
    roi_gt_prob = 0.8


class AMDMI250Profile(GPUProfile):
    platform = 'amd_rocm'
    batch_size = 256
    eval_batch_size = 384
    num_workers = 24
    prefetch_factor = 6
    persistent_workers = True
    input_height = 448
    input_width = 448
    resize_size = 512
    fc_hidden = 1536
    grad_accum_steps = 1
    use_torch_compile = True
    compile_mode = 'default'
    compile_dynamic = False
    use_gradient_checkpoint = True
    oom_headroom_ratio = 0.05
    max_checkpoints = 3
    pin_memory = False
    tta_sizes = [384, 416]
    lr = 5e-3
    backbone_lr_factor = 0.1
    warmup_epochs = 5
    dropout = 0.15
    ema_decay = 0.9995
    aux_loss_weight = 0.35
    bbox_loss_weight = 5.0
    attn_diversity_weight = 0.1
    attn_supervision_weight = 2.0
    ordering_loss_weight = 2.0
    multiscale_feat_dim = 512
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_feat_dim = 256
    transformer_heads = 4
    transformer_layers = 4
    head_interaction_layers = 2
    num_attn_channels = 8
    cutmix_alpha = 1.0
    cutmix_prob = 0.5
    erase_prob = 0.15
    smooth = 0.1
    aug_rotation_degrees = 10
    aug_blur_prob = 0.15
    roi_gt_prob = 0.8


def _detect_gpu_profile():
    if GPU_PLATFORM == 'cpu':
        return CPUProfile()

    if IS_AMD:
        if TOTAL_VRAM_GB >= 180:
            profile = AMDMI250Profile()
        elif TOTAL_VRAM_GB >= 120:
            profile = AMDLargeProfile()
        elif TOTAL_VRAM_GB >= 48:
            profile = AMDMidProfile()
        else:
            profile = AMDMidProfile()
            profile.batch_size = 64
            profile.eval_batch_size = 96
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 2, 4), 8)
            profile.compile_dynamic = True
        profile.multiprocessing_context = 'spawn'
        _configure_rocm_blas_backend()

    elif IS_NVIDIA:
        if TOTAL_VRAM_GB >= 40:
            profile = GPUProfile()
            profile.batch_size = 96
            profile.eval_batch_size = 160
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 2, 4), 12)
            profile.prefetch_factor = 2
        elif TOTAL_VRAM_GB >= 22:
            profile = A100Profile()
        elif TOTAL_VRAM_GB >= 16:
            profile = GPUProfile()
            profile.batch_size = 32
            profile.eval_batch_size = 64
            profile.grad_accum_steps = 4
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 2, 4), 8)
        elif TOTAL_VRAM_GB >= 8:
            profile = GPUProfile()
            profile.batch_size = 16
            profile.eval_batch_size = 32
            profile.grad_accum_steps = 8
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 1, 2), 4)
        else:
            return CPUProfile()
        profile.multiprocessing_context = 'fork'

    else:
        return CPUProfile()

    if profile.use_torch_compile and not COMPILE_AVAILABLE:
        print(f'[COMPILE] torch.compile not available on this platform '
              f'(platform={sys.platform}, triton={TRITON_AVAILABLE}), disabling')
        profile.use_torch_compile = False

    return profile


ACTIVE_PROFILE = _detect_gpu_profile()


class Config:
    batch_size = ACTIVE_PROFILE.batch_size
    lr = ACTIVE_PROFILE.lr
    backbone_lr_factor = ACTIVE_PROFILE.backbone_lr_factor
    momentum = 0.9
    weights_decay = 5e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 5
    print_interval = 50
    checkpoints = os.path.join(SCRIPT_DIR, 'checkpoints')
    pretrained = None
    start_epoch = 0
    epoches = 120
    warmup_epochs = ACTIVE_PROFILE.warmup_epochs
    smooth = ACTIVE_PROFILE.smooth
    erase_prob = ACTIVE_PROFILE.erase_prob
    num_heads = 6
    input_height = ACTIVE_PROFILE.input_height
    input_width = ACTIVE_PROFILE.input_width
    resize_size = ACTIVE_PROFILE.resize_size
    tta_sizes = ACTIVE_PROFILE.tta_sizes
    dropout = ACTIVE_PROFILE.dropout
    fc_hidden = ACTIVE_PROFILE.fc_hidden
    ema_decay = ACTIVE_PROFILE.ema_decay
    cutmix_alpha = ACTIVE_PROFILE.cutmix_alpha
    cutmix_prob = ACTIVE_PROFILE.cutmix_prob
    train_eval_interval = 10
    use_torch_compile = ACTIVE_PROFILE.use_torch_compile
    compile_mode = ACTIVE_PROFILE.compile_mode
    compile_dynamic = ACTIVE_PROFILE.compile_dynamic
    compile_fullgraph = ACTIVE_PROFILE.compile_fullgraph
    attn_diversity_weight = ACTIVE_PROFILE.attn_diversity_weight
    multiscale_feat_dim = ACTIVE_PROFILE.multiscale_feat_dim
    bbox_loss_weight = ACTIVE_PROFILE.bbox_loss_weight
    pos_embed_channels = ACTIVE_PROFILE.pos_embed_channels
    feat_spatial_size = ACTIVE_PROFILE.feat_spatial_size
    ordering_loss_weight = ACTIVE_PROFILE.ordering_loss_weight
    roi_refine = True
    roi_feat_dim = ACTIVE_PROFILE.roi_feat_dim
    roi_teacher_forcing = True
    transformer_heads = ACTIVE_PROFILE.transformer_heads
    transformer_layers = ACTIVE_PROFILE.transformer_layers
    model_type = 'fpn_multihead'
    keep_aspect_ratio = True
    attn_supervision_weight = ACTIVE_PROFILE.attn_supervision_weight
    head_interaction_layers = ACTIVE_PROFILE.head_interaction_layers
    aug_rotation_degrees = ACTIVE_PROFILE.aug_rotation_degrees
    aug_blur_prob = ACTIVE_PROFILE.aug_blur_prob
    roi_gt_prob = ACTIVE_PROFILE.roi_gt_prob
    num_attn_channels = ACTIVE_PROFILE.num_attn_channels
    early_stopping_patience = 20
    aux_loss_weight = ACTIVE_PROFILE.aux_loss_weight
    grad_accum_steps = ACTIVE_PROFILE.grad_accum_steps
    eval_batch_size = ACTIVE_PROFILE.eval_batch_size
    max_checkpoints = ACTIVE_PROFILE.max_checkpoints
    oom_headroom_ratio = ACTIVE_PROFILE.oom_headroom_ratio
    pin_memory = ACTIVE_PROFILE.pin_memory
    prefetch_factor = ACTIVE_PROFILE.prefetch_factor
    num_workers = ACTIVE_PROFILE.num_workers
    use_gradient_checkpoint = ACTIVE_PROFILE.use_gradient_checkpoint
    persistent_workers = ACTIVE_PROFILE.persistent_workers
    multiprocessing_context = ACTIVE_PROFILE.multiprocessing_context


config = Config()


def _check_shm_and_adjust():
    if config.num_workers <= 0:
        return
    try:
        shm_stats = os.statvfs('/dev/shm')
        shm_gb = shm_stats.f_bsize * shm_stats.f_blocks / (1024**3)
        per_batch_mb = config.batch_size * 3 * config.input_height * config.input_width * 4 / (1024**2)
        needed_gb = config.num_workers * config.prefetch_factor * per_batch_mb / 1024
        if shm_gb < needed_gb * 1.5:
            safe_workers = max(int(shm_gb * 1024 / (per_batch_mb * config.prefetch_factor * 1.5)), 0)
            if safe_workers < config.num_workers:
                print(f'[SHM] /dev/shm={shm_gb:.1f}GB, need={needed_gb:.1f}GB. '
                      f'Reducing num_workers {config.num_workers}->{safe_workers}')
                config.num_workers = safe_workers
                if safe_workers == 0:
                    config.prefetch_factor = None
                    config.multiprocessing_context = None
    except Exception:
        if sys.platform == 'win32' and config.num_workers > 0:
            print('[SHM] /dev/shm check skipped on Windows. If DataLoader shared memory '
                  'errors occur, try reducing num_workers.')


_check_shm_and_adjust()

CSV_PATH = os.path.join(BASE_DIR, 'mchar_data_list_0515.csv')
dataset_path = os.path.join(BASE_DIR, 'dataset')

data_dir = {
    'train_data': os.path.join(dataset_path, 'mchar_train') + os.sep,
    'val_data': os.path.join(dataset_path, 'mchar_val') + os.sep,
    'test_data': os.path.join(dataset_path, 'mchar_test_a') + os.sep,
    'train_label': os.path.join(dataset_path, 'mchar_train.json'),
    'val_label': os.path.join(dataset_path, 'mchar_val.json'),
    'submit_file': os.path.join(dataset_path, 'mchar_sample_submit_A.csv'),
}

t.hub.set_dir(os.path.join(BASE_DIR, 'torch_hub'))


def make_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=None):
    from torch.utils.data import DataLoader
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


def print_env_info():
    print("=" * 80)
    print(f"GPU Platform: {GPU_PLATFORM.upper()}")
    print(f"Total VRAM: {TOTAL_VRAM_GB:.1f} GB")
    print(f"Physical CPU Cores: {NUM_PHYSICAL_CORES}")
    if IS_AMD:
        rocm_str = f"{ROCM_VERSION[0]}.{ROCM_VERSION[1]}" if ROCM_VERSION else "unknown"
        print(f"ROCm Version: {rocm_str}")
        hip_ver = getattr(t.version, 'hip', None)
        if hip_ver:
            print(f"HIP Runtime: {hip_ver}")
        try:
            blas_backend = t.backends.cuda.preferred_blas_library()
            print(f"BLAS Backend: {blas_backend}")
        except Exception:
            pass
        gfx = get_gfx_arch()
        if gfx:
            print(f"GPU Arch: {gfx}")
        hsa_ver = os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')
        print(f"HSA_OVERRIDE_GFX_VERSION: {hsa_ver}")
    elif IS_NVIDIA:
        cuda_ver = t.version.cuda if t.cuda.is_available() else 'N/A'
        print(f"CUDA Version: {cuda_ver}")
        if t.cuda.is_available():
            print(f"GPU Name: {t.cuda.get_device_name(0)}")
        precision_config = get_precision_config()
        print(f"TF32 matmul: {'enabled' if precision_config['tf32_enabled'] else 'disabled'}")
    print(f"Active Profile: {ACTIVE_PROFILE.__class__.__name__} (platform={ACTIVE_PROFILE.platform})")
    print(f"Data Loader Workers: {config.num_workers}")
    print(f"Train Batch Size: {config.batch_size}")
    print(f"Eval Batch Size: {config.eval_batch_size}")
    print(f"Gradient Accumulation Steps: {config.grad_accum_steps}")
    print(f"Equivalent Batch Size: {config.batch_size * config.grad_accum_steps}")
    print(f"Input Size: {config.input_height}x{config.input_width}")
    print(f"FC Hidden: {config.fc_hidden}")
    print(f"Use Torch Compile: {config.use_torch_compile}")
    if config.use_torch_compile:
        print(f"Compile Mode: {config.compile_mode}")
        print(f"Compile Dynamic: {config.compile_dynamic}")
        print(f"Compile Fullgraph: {config.compile_fullgraph}")
    print(f"Compile Available: {COMPILE_AVAILABLE}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    print(f"Gradient Checkpoint: {config.use_gradient_checkpoint}")
    print(f"Pin Memory: {config.pin_memory}")
    print(f"Prefetch Factor: {config.prefetch_factor}")
    print(f"OOM Headroom: {config.oom_headroom_ratio * 100:.0f}%")
    print(f"Max Checkpoints: {config.max_checkpoints}")
    print("=" * 80)
