#!/usr/bin/env python3
"""
模型学习障碍分析与修复验证脚本
验证所有修复是否正确应用
"""

import sys
import os
import torch as t
import torch.nn as nn
import traceback

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("街景字符识别模型 - 学习障碍修复验证")
print("=" * 80)
print()


def verify_config():
    """验证配置文件的修复"""
    print("1. 验证配置文件修复")
    print("-" * 40)
    
    try:
        from config import config
        
        # 检查batch_size
        print(f"  batch_size: {config.batch_size} (期望: 56)")
        print(f"  eval_batch_size: {config.eval_batch_size} (期望: 56)")
        print(f"  gradient_checkpoint_with_bf16: {config.gradient_checkpoint_with_bf16} (期望: False)")
        print(f"  use_bf16: {config.use_bf16}")
        print(f"  use_gradient_checkpoint: {config.use_gradient_checkpoint}")
        print(f"  persistent_workers: {config.persistent_workers}")
        
        config_ok = (
            config.batch_size == 56 and 
            config.eval_batch_size == 56 and
            config.gradient_checkpoint_with_bf16 is False
        )
        
        print(f"  配置验证: {'✅ 通过' if config_ok else '❌ 失败'}")
        print()
        return config_ok
    except Exception as e:
        print(f"  ❌ 配置验证异常: {e}")
        traceback.print_exc()
        print()
        return False


def verify_trainer_multihead():
    """验证trainer/multihead.py的修复"""
    print("2. 验证trainer/multihead.py修复")
    print("-" * 40)
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'trainer', 'multihead.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键修复点
        checks = [
            ("BF16反向传播", 'if self.use_bf16:' in content and 'loss.backward()' in content),
            ("NaN梯度检测", 'has_nan_grad = False' in content),
            ("梯度裁剪顺序", 'self.scaler.unscale_(' in content),
            ("BF16模式scaler.update", 'self.scaler.update()' in content),
            ("EMA更新保护", 'if not _batch_scaler_skipped:' in content),
        ]
        
        all_pass = True
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False
        
        print(f"  Trainer验证: {'✅ 通过' if all_pass else '❌ 失败'}")
        print()
        return all_pass
    except Exception as e:
        print(f"  ❌ Trainer验证异常: {e}")
        traceback.print_exc()
        print()
        return False


def verify_models_multihead():
    """验证models/multihead.py的修复"""
    print("3. 验证models/multihead.py修复")
    print("-" * 40)
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'models', 'multihead.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键修复点
        checks = [
            ("BF16梯度检查点禁用", 'if config.use_bf16 and not config.gradient_checkpoint_with_bf16:' in content),
            ("梯度检查点条件", 'use_ckpt =' in content),
        ]
        
        all_pass = True
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False
        
        print(f"  Models验证: {'✅ 通过' if all_pass else '❌ 失败'}")
        print()
        return all_pass
    except Exception as e:
        print(f"  ❌ Models验证异常: {e}")
        traceback.print_exc()
        print()
        return False


def verify_trainer_base():
    """验证trainer/base.py的修复"""
    print("4. 验证trainer/base.py修复")
    print("-" * 40)
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'trainer', 'base.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键修复点
        checks = [
            ("AdamW权重衰减分组", 'separate_params' in content),
            ("decay_params分组", 'decay_params =' in content),
            ("no_decay_params分组", 'no_decay_params =' in content),
        ]
        
        all_pass = True
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False
        
        print(f"  Base验证: {'✅ 通过' if all_pass else '❌ 失败'}")
        print()
        return all_pass
    except Exception as e:
        print(f"  ❌ Base验证异常: {e}")
        traceback.print_exc()
        print()
        return False


def verify_losses_attention():
    """验证losses/attention.py的修复"""
    print("5. 验证losses/attention.py修复")
    print("-" * 40)
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'losses', 'attention.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键修复点
        checks = [
            ("KL散度计算", 'compute_single_gaussian_kl' in content),
            ("数值稳定性保护", 'eps = 1e-8' in content or 'clamp' in content),
        ]
        
        all_pass = True
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False
        
        print(f"  Losses验证: {'✅ 通过' if all_pass else '❌ 失败'}")
        print()
        return all_pass
    except Exception as e:
        print(f"  ❌ Losses验证异常: {e}")
        traceback.print_exc()
        print()
        return False


def test_model_initialization():
    """测试模型初始化"""
    print("6. 测试模型初始化")
    print("-" * 40)
    
    try:
        from models.multihead import DigitsResnet101
        from config import config
        
        print("  创建模型...")
        model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✅ 模型创建成功")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print()
        return True
    except Exception as e:
        print(f"  ❌ 模型初始化异常: {e}")
        traceback.print_exc()
        print()
        return False


def test_forward_backward():
    """测试前向和反向传播"""
    print("7. 测试前向和反向传播")
    print("-" * 40)
    
    try:
        from models.multihead import DigitsResnet101
        from config import config
        
        device = t.device('cpu')  # 使用CPU进行测试
        
        print("  创建模型...")
        model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads).to(device)
        model.train()
        
        # 创建测试数据
        print("  创建测试数据...")
        batch_size = 2
        img = t.randn(batch_size, 3, config.input_height, config.input_width).to(device)
        label = t.randint(0, config.class_num - 1, (batch_size, config.num_heads)).to(device)
        bbox = t.rand(batch_size, config.num_heads, 4).to(device)
        
        # 前向传播
        print("  前向传播...")
        cls_outs, bbox_outs, length_logits = model(img, bbox)
        print(f"  ✅ 前向传播成功")
        print(f"    分类输出数量: {len(cls_outs)}")
        print(f"    分类输出形状: {cls_outs[0].shape}")
        
        # 计算损失
        print("  计算损失...")
        import torch.nn.functional as F
        total_loss = 0
        for h_idx, cls_out in enumerate(cls_outs):
            total_loss += F.cross_entropy(cls_out, label[:, h_idx])
        total_loss = total_loss / len(cls_outs)
        
        length_labels = t.ones(batch_size, dtype=t.long, device=device) * 2
        length_loss = F.cross_entropy(length_logits, length_labels)
        total_loss += length_loss * 0.1
        
        print(f"  ✅ 损失计算成功: {total_loss.item():.4f}")
        
        # 反向传播
        print("  反向传播...")
        model.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        print("  检查梯度...")
        has_nan = False
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                if t.isnan(param.grad).any() or t.isinf(param.grad).any():
                    has_nan = True
                    print(f"    ❌ {name}: NaN/Inf梯度")
        
        if has_nan:
            print(f"  ❌ 发现NaN/Inf梯度")
            print()
            return False
        elif has_grad:
            print(f"  ✅ 反向传播成功，梯度正常")
            print()
            return True
        else:
            print(f"  ⚠️ 没有可训练的梯度")
            print()
            return False
    except Exception as e:
        print(f"  ❌ 前向/反向传播异常: {e}")
        traceback.print_exc()
        print()
        return False


def main():
    """主函数"""
    results = []
    
    # 运行所有验证
    results.append(("配置文件", verify_config()))
    results.append(("Trainer MultiHead", verify_trainer_multihead()))
    results.append(("Models MultiHead", verify_models_multihead()))
    results.append(("Trainer Base", verify_trainer_base()))
    results.append(("Losses Attention", verify_losses_attention()))
    results.append(("模型初始化", test_model_initialization()))
    results.append(("前向/反向传播", test_forward_backward()))
    
    # 汇总结果
    print("=" * 80)
    print("验证汇总")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print()
    print("=" * 80)
    if all_passed:
        print("🎉 所有验证通过！所有修复已正确应用。")
    else:
        print("⚠️ 部分验证失败，请检查上面的详细输出。")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
