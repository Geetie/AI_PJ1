"""配置模块 - 包含所有超参数和全局设置"""
import os
import random
import numpy as np
import torch as t
import multiprocessing

# ==================== 随机种子设置 ====================
def set_seed(seed=42):
    """设置全局随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True


# 初始化随机种子
set_seed(42)

# TF32精度优化（如果可用）
if t.cuda.is_available():
    t.set_float32_matmul_precision('high')
    print('✅ TF32 matmul precision enabled')

# ==================== 路径配置 ====================
BASE_DIR = '/mnt/workspace' if os.path.exists('/mnt/workspace') else os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
t.hub.set_dir(os.path.join(SCRIPT_DIR, 'torch_hub'))

NUM_WORKERS = min(multiprocessing.cpu_count() - 1, 8) if os.name != 'nt' else 0

# 数据路径
CSV_PATH = os.path.join(BASE_DIR, 'mchar_data_list_0515.csv')
dataset_path = os.path.join(BASE_DIR, 'dataset')

data_dir = {
    'train_data': f'{dataset_path}/mchar_train/',
    'val_data': f'{dataset_path}/mchar_val/',
    'test_data': f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label': f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv'
}

# ==================== 常量定义 ====================
NUM_HEADS = 6  # 最大支持6个字符，但通过动态掩码处理可变长度

# ==================== 训练配置类 ====================
class Config:
    """所有超参数配置"""
    # 基础训练参数
    batch_size = 64
    eval_batch_size = 96
    lr = 2e-3
    backbone_lr_factor = 0.1
    momentum = 0.9
    weights_decay = 5e-4
    class_num = 11
    
    # 优化器与调度器
    optimizer_type = 'sgd'
    scheduler_type = 'warmup_cosine'
    
    # 梯度累积与裁剪
    grad_accum_steps = 2
    grad_clip_max_norm = 5.0
    
    # Loss权重
    cls_loss_weight = 1.0
    aux_loss_weight = 0.3
    bbox_loss_weight = 5.0
    attn_diversity_weight = 0.1
    ordering_loss_weight = 1.0
    attn_supervision_weight = 1.0
    
    # 训练流程控制
    eval_interval = 1
    checkpoint_interval = 5
    print_interval = 50
    checkpoints = os.path.join(SCRIPT_DIR, 'checkpoints')
    pretrained = None
    start_epoch = 0
    epoches = 120
    warmup_epochs = 5
    resume_weights_only = False
    
    # 数据加载
    num_workers = NUM_WORKERS
    pin_memory = True
    persistent_workers = False
    prefetch_factor = 2 if NUM_WORKERS > 0 else None
    multiprocessing_context = 'fork' if os.name != 'nt' and NUM_WORKERS > 0 else None
    
    # 数据增强（降低强度以改善训练不足问题）
    smooth = 0.1
    erase_prob = 0.1
    mixup_alpha = 0.0
    mixup_prob = 0.0
    cutmix_alpha = 1.0
    cutmix_prob = 0.3
    aug_rotation_degrees = 5
    aug_blur_prob = 0.1
    
    # 输入尺寸
    num_heads = NUM_HEADS
    input_height = 384
    input_width = 384
    resize_size = 416
    tta_sizes = [288, 320, 352, 384, 416]
    
    # 模型架构
    dropout = 0.15
    fc_hidden = 1024
    ema_decay = 0.999
    train_eval_interval = 10
    use_torch_compile = False
    use_gradient_checkpoint = True
    
    # Torch Compile配置
    compile_mode = 'default'
    compile_dynamic = False
    compile_fullgraph = False
    
    # FPN Multi-Head特有参数
    multiscale_feat_dim = 512
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_refine = True
    roi_feat_dim = 256
    roi_teacher_forcing = True
    head_interaction_layers = 2
    roi_gt_prob = 0.8
    num_attn_channels = 8
    
    # Transformer特有参数
    transformer_heads = 4
    transformer_layers = 4
    
    # 通用配置
    model_type = 'fpn_multihead'
    keep_aspect_ratio = True
    use_char_level_acc = True
    early_stopping_patience = 15
    
    # 系统配置
    max_checkpoints = 5
    oom_headroom_ratio = 0.15


# 全局配置实例
config = Config()
