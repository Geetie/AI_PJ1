#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：检查 baseline.ipynb 关键配置和功能是否正常
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("街景字符识别项目 - 配置验证")
print("="*60)

try:
    # 1. 验证 config.py 导入和基本配置
    print("\n[1/6] 检查 config.py 配置...")
    from config import config, BASE_DIR, NUM_HEADS, IS_NVIDIA
    
    print(f"  ✅ config.py 导入成功")
    print(f"  ✅ BASE_DIR: {BASE_DIR}")
    print(f"  ✅ NUM_HEADS: {NUM_HEADS} (预期: 3)")
    print(f"  ✅ batch_size: {config.batch_size} (预期: 12)")
    print(f"  ✅ grad_accum_steps: {config.grad_accum_steps} (预期: 20)")
    print(f"  ✅ lr: {config.lr} (预期: 5e-5)")
    print(f"  ✅ dropout: {config.dropout} (预期: 0.4)")
    print(f"  ✅ attn_supervision_weight: {config.attn_supervision_weight} (预期: 0.0)")
    print(f"  ✅ cutmix_prob: {config.cutmix_prob} (预期: 0.05)")
    
    if NUM_HEADS != 3:
        print(f"  ⚠️ 警告: NUM_HEADS 当前为 {NUM_HEADS}，建议设置为 3")
    
except Exception as e:
    print(f"  ❌ config.py 导入失败: {e}")
    sys.exit(1)

try:
    # 2. 验证 set_seed 导入（v8更新要求）
    print("\n[2/6] 检查 set_seed 函数...")
    from config import set_seed
    set_seed(42)
    print(f"  ✅ set_seed 导入成功并可正常调用")
except Exception as e:
    print(f"  ⚠️ 警告: set_seed 函数导入有问题: {e}")

try:
    # 3. 验证数据模块
    print("\n[3/6] 检查数据模块...")
    from data.download import download_dataset
    print(f"  ✅ download_dataset 导入成功")
except Exception as e:
    print(f"  ⚠️ 警告: 数据模块导入有问题: {e}")

try:
    # 4. 验证模型模块
    print("\n[4/6] 检查模型模块...")
    from models import create_model
    model = create_model(config.model_type)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ create_model 导入成功")
    print(f"  ✅ 模型参数量: {total_params:,}")
except Exception as e:
    print(f"  ❌ 模型模块导入或创建失败: {e}")

try:
    # 5. 验证训练器模块
    print("\n[5/6] 检查训练器模块...")
    from trainer.multihead import MultiHeadTrainer
    print(f"  ✅ MultiHeadTrainer 导入成功")
except Exception as e:
    print(f"  ❌ 训练器模块导入失败: {e}")

try:
    # 6. 检查环境变量
    print("\n[6/6] 检查环境变量...")
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        print(f"  ⚠️ 警告: PYTORCH_CUDA_ALLOC_CONF 未设置")
        print(f"     建议在运行前设置: export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")
    else:
        print(f"  ✅ PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    # 检查 PyTorch 和 GPU
    import torch as t
    print(f"  ✅ PyTorch version: {t.__version__}")
    print(f"  ✅ CUDA available: {t.cuda.is_available()}")
    if t.cuda.is_available():
        print(f"  ✅ CUDA version: {t.version.cuda}")
        print(f"  ✅ GPU name: {t.cuda.get_device_name(0)}")
        props = t.cuda.get_device_properties(0)
        vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
        print(f"  ✅ GPU VRAM: {vram:.1f} GB")

except Exception as e:
    print(f"  ⚠️ 环境变量检查失败: {e}")


print("\n" + "="*60)
print("验证完成！")
print("="*60)

print("\n快速指南:")
print("1. 在终端中运行:")
print("   export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")
print("   python -m ipykernel install --user --name v8_ocr")
print("2. 打开 baseline.ipynb，从头开始执行")
print("3. 确保设置 TRAIN_FROM_SCRATCH = True")
print("4. 训练时观察显存使用和损失曲线")
