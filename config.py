import os
import multiprocessing
import torch as t


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/mnt/workspace' if os.path.exists('/mnt/workspace') else SCRIPT_DIR

IS_MODELSCOPE = os.path.exists('/mnt/workspace')

os.environ.setdefault('MIOPEN_DISABLE_CACHE', '0')
os.environ.setdefault('MIOPEN_FIND_MODE', '2')
os.environ.setdefault('MIOPEN_USER_DB_PATH', os.path.join(BASE_DIR, 'miopen_cache'))


def _detect_gpu_platform():
    if not t.cuda.is_available():
        return 'cpu'
    if hasattr(t.version, 'hip') and t.version.hip is not None:
        return 'amd_rocm'
    try:
        props = t.cuda.get_device_properties(0)
        gpu_name = props.name.lower()
        vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024 ** 3)
        if 'amd' in gpu_name or 'radeon' in gpu_name or 'instinct' in gpu_name or 'mi2' in gpu_name:
            return 'amd_rocm'
        if not gpu_name.strip() and vram > 100:
            return 'amd_rocm'
    except Exception:
        pass
    return 'nvidia_cuda'


def _get_total_vram_gb():
    if not t.cuda.is_available():
        return 0
    try:
        props = t.cuda.get_device_properties(0)
        return getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024 ** 3)
    except Exception:
        return 0


GPU_PLATFORM = _detect_gpu_platform()
TOTAL_VRAM_GB = _get_total_vram_gb()
NUM_PHYSICAL_CORES = multiprocessing.cpu_count() or 2


class GPUProfile:
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
    use_gradient_checkpoint = True
    oom_headroom_ratio = 0.15
    max_checkpoints = 3
    pin_memory = True
    tta_sizes = [288, 320, 352, 384, 416]
    lr = 3e-3
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
    oom_headroom_ratio = 0.15
    max_checkpoints = 3
    pin_memory = True
    tta_sizes = [288, 320, 352, 384, 416]
    lr = 3e-3
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


class AMDLargeProfile(GPUProfile):
    batch_size = 256
    eval_batch_size = 384
    num_workers = 8
    prefetch_factor = 2
    persistent_workers = True
    multiprocessing_context = 'fork'
    input_height = 416
    input_width = 416
    resize_size = 448
    fc_hidden = 1536
    grad_accum_steps = 1
    use_torch_compile = False
    compile_mode = 'default'
    use_gradient_checkpoint = False
    oom_headroom_ratio = 0.10
    max_checkpoints = 3
    pin_memory = True
    tta_sizes = [320, 352, 384, 416, 448]
    lr = 5e-3
    backbone_lr_factor = 0.1
    warmup_epochs = 8
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


def _detect_gpu_profile():
    if GPU_PLATFORM == 'cpu':
        return CPUProfile()
    if GPU_PLATFORM == 'amd_rocm':
        if TOTAL_VRAM_GB >= 120:
            return AMDLargeProfile()
        elif TOTAL_VRAM_GB >= 48:
            profile = GPUProfile()
            profile.batch_size = 128
            profile.eval_batch_size = 192
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 4, 4), 12)
            profile.prefetch_factor = 3
            profile.pin_memory = False
            profile.use_torch_compile = True
            profile.compile_mode = 'default'
            return profile
        else:
            profile = GPUProfile()
            profile.batch_size = 64
            profile.eval_batch_size = 96
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 2, 4), 8)
            profile.pin_memory = False
            profile.use_torch_compile = True
            profile.compile_mode = 'default'
            return profile
    else:
        if TOTAL_VRAM_GB >= 40:
            profile = GPUProfile()
            profile.batch_size = 96
            profile.eval_batch_size = 160
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 2, 4), 12)
            profile.prefetch_factor = 2
            return profile
        elif TOTAL_VRAM_GB >= 24:
            return A100Profile()
        elif TOTAL_VRAM_GB >= 16:
            profile = GPUProfile()
            profile.batch_size = 32
            profile.eval_batch_size = 64
            profile.grad_accum_steps = 4
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 2, 4), 8)
            return profile
        elif TOTAL_VRAM_GB >= 8:
            profile = GPUProfile()
            profile.batch_size = 16
            profile.eval_batch_size = 32
            profile.grad_accum_steps = 8
            profile.num_workers = min(max(NUM_PHYSICAL_CORES - 1, 2), 4)
            return profile
        else:
            return CPUProfile()


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
        pass


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
    print(f"Active Profile: {ACTIVE_PROFILE.__class__.__name__}")
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
    print(f"Gradient Checkpoint: {config.use_gradient_checkpoint}")
    print(f"Pin Memory: {config.pin_memory}")
    print(f"Prefetch Factor: {config.prefetch_factor}")
    print(f"OOM Headroom: {config.oom_headroom_ratio * 100:.0f}%")
    print(f"Max Checkpoints: {config.max_checkpoints}")
    print("=" * 80)
