import os
import multiprocessing
import torch as t


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/mnt/workspace' if os.path.exists('/mnt/workspace') else SCRIPT_DIR

IS_MODELSCOPE = os.path.exists('/mnt/workspace')


def _detect_gpu_platform():
    """Detect GPU platform: NVIDIA CUDA or AMD ROCm"""
    if not t.cuda.is_available():
        return 'cpu'
    
    try:
        props = t.cuda.get_device_properties(0)
        gpu_name = props.name.lower()
        if 'amd' in gpu_name or 'radeon' in gpu_name:
            return 'amd_rocm'
        else:
            return 'nvidia_cuda'
    except Exception:
        return 'nvidia_cuda'


def _get_total_vram_gb():
    """Get total VRAM in GB"""
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


def _auto_batch_size():
    """Automatically determine batch size based on GPU platform and VRAM"""
    if not t.cuda.is_available():
        return 32
    
    if GPU_PLATFORM == 'amd_rocm':
        if TOTAL_VRAM_GB >= 180:
            return 256
        elif TOTAL_VRAM_GB >= 120:
            return 192
        elif TOTAL_VRAM_GB >= 90:
            return 128
        elif TOTAL_VRAM_GB >= 48:
            return 96
        elif TOTAL_VRAM_GB >= 24:
            return 64
        elif TOTAL_VRAM_GB >= 16:
            return 48
        elif TOTAL_VRAM_GB >= 8:
            return 32
        else:
            return 16
    else:
        if TOTAL_VRAM_GB >= 45:
            return 96
        elif TOTAL_VRAM_GB >= 24:
            return 64
        elif TOTAL_VRAM_GB >= 20:
            return 32
        elif TOTAL_VRAM_GB >= 16:
            return 32
        elif TOTAL_VRAM_GB >= 8:
            return 16
        elif TOTAL_VRAM_GB >= 4:
            return 8
        else:
            return 8


def _auto_num_workers():
    """Automatically determine number of data loader workers based on CPU cores"""
    if IS_MODELSCOPE:
        return min(max(NUM_PHYSICAL_CORES - 2, 4), 16)
    else:
        return min(max(NUM_PHYSICAL_CORES - 1, 4), 16) if os.name != 'nt' else 0


NUM_WORKERS = _auto_num_workers()
NUM_HEADS = 6


class Config:
    batch_size = _auto_batch_size()
    lr = 3e-3
    backbone_lr_factor = 0.1
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
    warmup_epochs = 5
    smooth = 0.1
    erase_prob = 0.2
    num_heads = NUM_HEADS
    input_height = 384
    input_width = 384
    resize_size = 416
    tta_sizes = [288, 320, 352, 384, 416]
    dropout = 0.2
    fc_hidden = 1024
    ema_decay = 0.999
    cutmix_alpha = 1.0
    cutmix_prob = 0.5
    train_eval_interval = 10
    use_torch_compile = GPU_PLATFORM == 'amd_rocm'
    attn_diversity_weight = 0.1
    multiscale_feat_dim = 512
    bbox_loss_weight = 5.0
    pos_embed_channels = 64
    feat_spatial_size = 40
    ordering_loss_weight = 2.0
    roi_refine = True
    roi_feat_dim = 256
    roi_teacher_forcing = True
    transformer_heads = 4
    transformer_layers = 4
    model_type = 'fpn_multihead'
    keep_aspect_ratio = True
    attn_supervision_weight = 2.0
    head_interaction_layers = 2
    aug_rotation_degrees = 10
    aug_blur_prob = 0.15
    roi_gt_prob = 0.8
    num_attn_channels = 8
    use_char_level_acc = True
    early_stopping_patience = 20
    aux_loss_weight = 0.3
    grad_accum_steps = max(1, 128 // batch_size)


config = Config()

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


def print_env_info():
    """Print environment configuration information"""
    print("=" * 80)
    print(f"GPU Platform: {GPU_PLATFORM.upper()}")
    print(f"Total VRAM: {TOTAL_VRAM_GB:.1f} GB")
    print(f"Physical CPU Cores: {NUM_PHYSICAL_CORES}")
    print(f"Data Loader Workers: {NUM_WORKERS}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Gradient Accumulation Steps: {config.grad_accum_steps}")
    print(f"Equivalent Batch Size: {config.batch_size * config.grad_accum_steps}")
    print(f"Use Torch Compile: {config.use_torch_compile}")
    print("=" * 80)
