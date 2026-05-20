"""测试随机性修复的有效性"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as t
import numpy as np
import random


def test_seed_consistency():
    """测试不同epoch的种子是否唯一"""
    from utils.seed import set_epoch_seed
    
    seeds = []
    for epoch in range(10):
        epoch_seed = set_epoch_seed(42, epoch)
        seeds.append(epoch_seed)
    
    # 验证所有种子都不同
    assert len(set(seeds)) == len(seeds), f"种子重复！种子列表: {seeds}"
    print("✓ 测试1通过：不同epoch的种子都是唯一的")
    print(f"  种子序列: {seeds}")


def test_data_loader_randomness():
    """测试数据加载器的随机性"""
    from data.dataset import DigitsDataset
    from torch.utils.data import DataLoader
    from utils.seed import make_epoch_generator
    
    # 创建数据集（使用小批量以加快测试）
    dataset = DigitsDataset(mode='train', aug=True, input_size=(64, 64))
    
    # 使用不同种子创建生成器
    gen1 = make_epoch_generator(42, epoch=0)
    gen2 = make_epoch_generator(42, epoch=1)
    
    loader1 = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen1)
    loader2 = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen2)
    
    # 获取第一个batch
    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))
    
    # 验证两个batch的图像数据不同
    img1 = batch1[0]
    img2 = batch2[0]
    
    # 计算相似度
    similarity = t.mean((img1 == img2).float()).item()
    
    print(f"  Batch相似度: {similarity:.2%}")
    
    # 相似度应低于90%（随机打乱后不太可能高度相似）
    assert similarity < 0.9, f"数据加载顺序相同！相似度: {similarity:.2%}"
    print("✓ 测试2通过：不同epoch的数据加载顺序不同")


def test_generator_seed_independence():
    """测试生成器种子的独立性"""
    from utils.seed import make_epoch_generator
    
    # 使用相同种子创建两个生成器
    gen1 = make_epoch_generator(42, epoch=0)
    gen2 = make_epoch_generator(42, epoch=0)
    
    # 使用不同种子创建生成器
    gen3 = make_epoch_generator(42, epoch=1)
    
    # 生成随机数
    rand1 = [gen1.random() for _ in range(5)]
    rand2 = [gen2.random() for _ in range(5)]
    rand3 = [gen3.random() for _ in range(5)]
    
    # 相同种子应产生相同序列
    assert rand1 == rand2, "相同种子应产生相同随机序列"
    
    # 不同种子应产生不同序列
    assert rand1 != rand3, "不同种子应产生不同随机序列"
    
    print("✓ 测试3通过：生成器种子独立性验证通过")
    print(f"  种子42的序列: {[round(r, 4) for r in rand1]}")
    print(f"  种子43的序列: {[round(r, 4) for r in rand3]}")


def test_data_augmentation_randomness():
    """测试数据增强的随机性"""
    from data.dataset import DigitsDataset
    from torch.utils.data import DataLoader
    from utils.seed import set_epoch_seed, make_epoch_generator
    
    # 设置不同的种子
    set_epoch_seed(42, 0)
    gen1 = make_epoch_generator(42, epoch=0)
    
    set_epoch_seed(42, 1)
    gen2 = make_epoch_generator(42, epoch=1)
    
    dataset = DigitsDataset(mode='train', aug=True, input_size=(64, 64))
    
    loader1 = DataLoader(dataset, batch_size=1, shuffle=False, generator=gen1)
    loader2 = DataLoader(dataset, batch_size=1, shuffle=False, generator=gen2)
    
    # 获取同一个样本（索引0）在不同种子下的增强结果
    img1 = None
    img2 = None
    
    for i, batch in enumerate(loader1):
        if i == 0:
            img1 = batch[0].clone()
            break
    
    for i, batch in enumerate(loader2):
        if i == 0:
            img2 = batch[0].clone()
            break
    
    # 计算增强后图像的差异
    diff = t.abs(img1 - img2).mean().item()
    
    print(f"  增强差异: {diff:.4f}")
    
    # 增强后图像应该有明显差异
    assert diff > 0.01, f"数据增强未生效！差异: {diff:.4f}"
    print("✓ 测试4通过：数据增强的随机性验证通过")


def test_parameter_update():
    """测试模型参数更新"""
    import torch.nn as nn
    
    # 创建一个简单模型
    model = nn.Linear(10, 5)
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)
    
    # 保存初始参数
    initial_params = [p.detach().clone() for p in model.parameters()]
    
    # 执行一次前向和反向传播
    x = t.randn(3, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    # 检查参数是否更新
    param_diff = 0
    for p, initial_p in zip(model.parameters(), initial_params):
        param_diff += t.norm(p - initial_p).item()
    
    print(f"  参数更新幅度: {param_diff:.6f}")
    
    # 参数应该有明显更新
    assert param_diff > 1e-6, f"参数未更新！差异: {param_diff:.6f}"
    print("✓ 测试5通过：模型参数更新验证通过")


if __name__ == '__main__':
    print("=" * 60)
    print("随机性修复验证测试")
    print("=" * 60)
    
    # 设置初始种子
    from utils.seed import set_seed
    set_seed(42)
    
    test_seed_consistency()
    print()
    
    test_generator_seed_independence()
    print()
    
    test_data_loader_randomness()
    print()
    
    test_data_augmentation_randomness()
    print()
    
    test_parameter_update()
    print()
    
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    print()
    print("修复总结:")
    print("1. 统一了全局种子和数据加载器种子")
    print("2. 确保每个epoch使用唯一的种子值")
    print("3. 数据增强的随机性现在与epoch种子同步")
    print("4. 模型参数能够正常更新")