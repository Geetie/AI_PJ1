#!/usr/bin/env python3
"""验证修复后的模型是否能够正常运行"""

import sys
import os
import torch as t
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, set_seed
from models.multihead import DigitsResnet101
from losses.classification import FocalLoss, LabelSmoothEntropy
from losses.attention import AttentionSupervisionLoss, attention_diversity_loss, spatial_ordering_loss


def test_model_init():
    """测试模型初始化"""
    print("=" * 60)
    print("1. Testing Model Initialization")
    print("=" * 60)
    
    try:
        set_seed(42)
        model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads)
        print(f"✅ Model created successfully")
        print(f"   Model type: {type(model).__name__}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("2. Testing Forward Pass")
    print("=" * 60)
    
    try:
        device = t.device('cpu')
        model = model.to(device)
        model.train()
        
        # 创建测试数据
        img = t.randn(2, 3, config.input_height, config.input_width).to(device)
        label = t.randint(0, config.class_num - 1, (2, config.num_heads)).to(device)
        bbox = t.rand(2, config.num_heads, 4).to(device)
        
        # 测试forward
        cls_outs, bbox_outs, length_logits = model(img, bbox)
        print(f"✅ Forward pass successful")
        print(f"   cls_outs: {len(cls_outs)} heads, shape: {cls_outs[0].shape}")
        print(f"   bbox_outs: {len(bbox_outs)} heads, shape: {bbox_outs[0].shape}")
        print(f"   length_logits shape: {length_logits.shape}")
        
        # 测试forward_with_attn
        cls_outs, bbox_outs, attn_maps, head_cls_outs, length_logits = model.forward_with_attn(img, bbox)
        print(f"✅ Forward with attention successful")
        print(f"   attn_maps: {len(attn_maps)} heads, shape: {attn_maps[0].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model):
    """测试反向传播"""
    print("\n" + "=" * 60)
    print("3. Testing Backward Pass")
    print("=" * 60)
    
    try:
        device = t.device('cpu')
        model = model.to(device)
        model.train()
        model.zero_grad()
        
        # 创建测试数据
        img = t.randn(2, 3, config.input_height, config.input_width).to(device)
        label = t.randint(0, config.class_num - 1, (2, config.num_heads)).to(device)
        bbox = t.rand(2, config.num_heads, 4).to(device)
        
        # 前向传播
        cls_outs, bbox_outs, attn_maps, head_cls_outs, length_logits = model.forward_with_attn(img, bbox)
        
        # 计算损失
        import torch.nn.functional as F
        total_loss = 0
        
        # 分类损失
        for h_idx, cls_out in enumerate(cls_outs):
            total_loss += F.cross_entropy(cls_out, label[:, h_idx])
        total_loss = total_loss / len(cls_outs)
        
        # 长度损失
        length_labels = t.ones(img.shape[0], dtype=t.long, device=device) * 2
        length_loss = F.cross_entropy(length_logits, length_labels)
        total_loss += length_loss * 0.1
        
        # 边界框损失
        for h_idx, bbox_out in enumerate(bbox_outs):
            bbox_target = bbox[:, h_idx, :]
            bbox_loss = F.mse_loss(bbox_out, bbox_target)
            total_loss += bbox_loss * 0.5
        
        print(f"✅ Loss computation successful: {total_loss.item():.4f}")
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        grad_ok = True
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if t.isnan(param.grad).any() or t.isinf(param.grad).any():
                    print(f"❌ NaN/Inf gradient detected in {name}")
                    grad_ok = False
        
        if grad_ok:
            print(f"✅ Backward pass successful")
            print(f"   Average grad norm: {sum(grad_norms) / len(grad_norms):.4f}")
            print(f"   Max grad norm: {max(grad_norms):.4f}")
            print(f"   Min grad norm: {min(grad_norms):.4f}")
        
        return grad_ok
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n" + "=" * 60)
    print("4. Testing Loss Functions")
    print("=" * 60)
    
    try:
        # 测试FocalLoss
        focal_loss = FocalLoss(gamma=2.0)
        test_preds = t.randn(4, 11)
        test_targets = t.randint(0, 10, (4,))
        loss_val = focal_loss(test_preds, test_targets)
        print(f"✅ FocalLoss: {loss_val.item():.4f}")
        
        # 测试LabelSmoothEntropy
        ls_loss = LabelSmoothEntropy(smooth=0.1)
        loss_val = ls_loss(test_preds, test_targets)
        print(f"✅ LabelSmoothEntropy: {loss_val.item():.4f}")
        
        # 测试AttentionSupervisionLoss
        attn_loss = AttentionSupervisionLoss()
        attn_maps = [t.rand(4, 1, 16, 16) for _ in range(3)]
        bbox_target = t.rand(4, 3, 4)
        bbox_mask = t.ones(4, 3)
        loss_val = attn_loss(attn_maps, bbox_target, bbox_mask)
        print(f"✅ AttentionSupervisionLoss: {loss_val.item():.4f}")
        
        # 测试attention_diversity_loss
        div_loss = attention_diversity_loss(attn_maps)
        print(f"✅ attention_diversity_loss: {div_loss.item():.4f}")
        
        # 测试spatial_ordering_loss
        ord_loss = spatial_ordering_loss(attn_maps, bbox_preds=[t.rand(4, 4) for _ in range(3)], bbox_mask=bbox_mask)
        print(f"✅ spatial_ordering_loss: {ord_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Verification of Gradient Fixes")
    print("=" * 60)
    
    all_passed = True
    
    # 测试模型初始化
    model = test_model_init()
    if model is None:
        all_passed = False
    
    # 测试前向传播
    if model is not None:
        if not test_forward_pass(model):
            all_passed = False
    
    # 测试反向传播
    if model is not None:
        if not test_backward_pass(model):
            all_passed = False
    
    # 测试损失函数
    if not test_loss_functions():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("   The gradient fixes have been successfully applied.")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())