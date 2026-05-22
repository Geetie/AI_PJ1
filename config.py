"""配置模块 - 包含所有超参数和全局设置"""
import os
import random
import numpy as np
import torch as t
import multiprocessing

# 导入平台相关的常量
from utils.platform import (
    GPU_PLATFORM, TOTAL_VRAM_GB, NUM_PHYSICAL_CORES,
    is_nvidia_cuda, is_amd_rocm, COMPILE_AVAILABLE,
    print_platform_info
)

# 导出平台相关的常量
def print_env_info():
    """打印环境信息的包装函数"""
    print_platform_info()

# ==================== 随机种子设置 ====================
def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = not deterministic
    t.backends.cudnn.deterministic = deterministic


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
NUM_HEADS = 3  # 减少到3以匹配94%样本长度≤3的数据分布，解决过拟合问题

# 平台相关常量导出
IS_NVIDIA = is_nvidia_cuda()
IS_AMD = is_amd_rocm()
IS_MODELSCOPE = os.path.exists('/mnt/workspace/.modelscope') or 'MODELSCOPE_CLOUD' in os.environ
COMPILE_AVAILABLE = COMPILE_AVAILABLE

# ==================== 训练配置类 ====================
class Config:
    """所有超参数配置"""
    # =========================================================
    # GPU 利用率优化：A10 22.2GB，batch_size=12 仅用 5.7GB (26%)
    # batch_size=32 + grad_accum_steps=8 → effective batch=256
    # =========================================================
    batch_size = 56
    eval_batch_size = 56
    lr = 3e-4
    backbone_lr_factor = 0.3  # 提高以加快预训练层的微调速度
    momentum = 0.9
    weights_decay = 1e-4
    class_num = 11
    
    optimizer_type = 'adamw'
    scheduler_type = 'cosine'
    
    grad_accum_steps = 4
    grad_clip_max_norm = 2.0  # 保守值，平衡梯度裁剪和有效梯度保留
    
    cls_loss_weight = 1.0
    aux_loss_weight = 0.1
    bbox_loss_weight = 5.0
    length_loss_weight = 3.0
    attn_diversity_weight = 0.02
    ordering_loss_weight = 0.02
    attn_supervision_weight = 0.05
    
    use_amp = True
    use_bf16 = False  # 默认禁用BF16，改用FP16以避免梯度下溢
    
    # 训练流程控制
    eval_interval = 1
    checkpoint_interval = 3
    print_interval = 50
    checkpoints = os.path.join(SCRIPT_DIR, 'checkpoints')
    pretrained = None
    start_epoch = 0
    epoches = 120
    warmup_epochs = 10
    resume_weights_only = False
    
    # 数据加载
    num_workers = NUM_WORKERS
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2 if NUM_WORKERS > 0 else None
    multiprocessing_context = 'fork' if os.name != 'nt' and NUM_WORKERS > 0 else None
    
    # 数据增强
    smooth = 0.1
    erase_prob = 0.1
    mixup_alpha = 0.0
    mixup_prob = 0.0
    cutmix_alpha = 1.0
    cutmix_prob = 0.05  # 大幅降低以避免BBox损失为0的问题
    aug_rotation_degrees = 10
    aug_blur_prob = 0.1
    
    # baseline.py特有参数
    train_eval_interval = 10
    
    # 输入尺寸
    num_heads = NUM_HEADS
    input_height = 384
    input_width = 384
    resize_size = 416
    tta_sizes = [288, 320, 352, 384, 416]
    
    # 学习率调度
    warmup_start_factor = 0.05
    scheduler_T0 = 5
    scheduler_T_mult = 2
    scheduler_eta_min = 1e-6

    # 模型架构
    dropout = 0.4  # 增强dropout以减少190M参数模型的过拟合
    fc_hidden = 1026  # ⚠️ 修改为能被 NUM_HEADS(3) 整除的值 (1024 % 3 = 1 ❌, 1026 % 3 = 0 ✅)
    ema_decay = 0.998
    use_torch_compile = False
    use_gradient_checkpoint = True
    gradient_checkpoint_with_bf16 = False  # BF16模式下禁用梯度检查点以避免数值不稳定
    
    # Torch Compile配置
    compile_mode = 'default'
    compile_dynamic = False
    compile_fullgraph = False
    
    # FPN Multi-Head特有参数
    multiscale_feat_dim = 512
    backbone_p1_channels = 128
    pos_embed_channels = 64
    feat_spatial_size = 40
    roi_refine = True
    roi_feat_dim = 256
    roi_teacher_forcing = True
    head_interaction_layers = 4
    roi_gt_prob = 0.8
    num_attn_channels = 8
    soft_attn_temperature = 0.5

    # 梯度平衡配置（补偿不同损失函数的固有梯度幅度差异）
    # 根据诊断结果：Classification梯度是BBox的242倍，是Length的1854倍
    # 使用保守补偿因子避免梯度爆炸
    gradient_balance = {
        'enabled': True,
        'cls_norm_factor': 1.0,
        'bbox_norm_factor': 50.0,
        'length_norm_factor': 200.0,
    }
    # ROI Teacher Forcing衰减配置
    roi_gt_decay_end_ratio = 0.5
    
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
    oom_headroom_ratio = 0.2


def make_dataloader(dataset, batch_size, shuffle=False, drop_last=False,
                    collate_fn=None, num_workers=None):
    from torch.utils.data import DataLoader
    nw = num_workers if num_workers is not None else NUM_WORKERS
    kwargs = dict(
        batch_size=batch_size, shuffle=shuffle,
        num_workers=nw, pin_memory=True,
        drop_last=drop_last,
    )
    if nw > 0:
        kwargs['prefetch_factor'] = 4
        kwargs['persistent_workers'] = True
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
    return DataLoader(dataset, **kwargs)


# 全局配置实例
config = Config()
