import random
import numpy as np
import torch as t


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    if hasattr(t.backends, 'cudnn'):
        t.backends.cudnn.benchmark = not deterministic
        t.backends.cudnn.deterministic = deterministic


def make_epoch_generator(base_seed=42, epoch=0):
    """创建指定epoch的随机数生成器
    
    确保每个epoch使用唯一的种子，保证数据加载顺序的随机性。
    种子计算公式与 set_epoch_seed 一致: base_seed + epoch * 1000，
    确保DataLoader生成器种子与全局种子同步。
    
    Args:
        base_seed: 基础种子值
        epoch: 当前epoch编号
        
    Returns:
        torch.Generator: 配置好种子的随机数生成器
    """
    generator = t.Generator()
    generator.manual_seed(base_seed + epoch * 1000)
    return generator


def set_epoch_seed(base_seed=42, epoch=0):
    """设置指定epoch的所有随机种子
    
    统一设置Python random、numpy和PyTorch的随机种子，
    确保数据增强等操作的随机性在每个epoch都不同。
    
    Args:
        base_seed: 基础种子值
        epoch: 当前epoch编号
    """
    epoch_seed = base_seed + epoch * 1000
    random.seed(epoch_seed)
    np.random.seed(epoch_seed)
    t.manual_seed(epoch_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(epoch_seed)
    return epoch_seed
