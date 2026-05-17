import os
import multiprocessing
import torch as t


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/mnt/workspace' if os.path.exists('/mnt/workspace') else SCRIPT_DIR

IS_MODELSCOPE = os.path.exists('/mnt/workspace')

NUM_WORKERS = min(max(multiprocessing.cpu_count() - 2, 2), 8) if IS_MODELSCOPE else (min(multiprocessing.cpu_count() - 1, 8) if os.name != 'nt' else 0)

NUM_HEADS = 6


def _auto_batch_size():
    if not t.cuda.is_available():
        return 64
    try:
        props = t.cuda.get_device_properties(0)
        total_vram_mb = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024 * 1024)
    except RuntimeError:
        return 128
    if total_vram_mb >= 45000:
        return 192
    elif total_vram_mb >= 24000:
        return 144
    elif total_vram_mb >= 22000:
        return 80
    elif total_vram_mb >= 16000:
        return 48
    elif total_vram_mb >= 8000:
        return 64
    return 32


class Config:
    batch_size = _auto_batch_size()
    lr = 2e-3
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
    use_torch_compile = False
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
    early_stopping_patience = 15
    aux_loss_weight = 0.3
    grad_accum_steps = 4


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
