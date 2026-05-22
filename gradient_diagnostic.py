#!/usr/bin/env python3
"""
梯度诊断和验证脚本
用于检测和验证街景字符识别模型的梯度学习障碍
"""

import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, set_seed
from models.multihead import DigitsResnet101
from losses.classification import FocalLoss, LabelSmoothEntropy


class GradientDiagnostic:
    """梯度诊断工具类"""
    
    def __init__(self):
        set_seed(42)
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
    
    def check_param_initialization(self, model: nn.Module) -> Dict:
        """检查参数初始化"""
        print("\n" + "="*60)
        print("1. Parameter Initialization Check")
        print("="*60)
        
        stats = {}
        layer_types = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                mean = param.data.mean().item()
                std = param.data.std().item()
                max_val = param.data.max().item()
                min_val = param.data.min().item()
                stats[name] = {
                    'mean': mean,
                    'std': std,
                    'max': max_val,
                    'min': min_val
                }
                
                if abs(mean) > 1.0 or std > 2.0 or std < 1e-6:
                    print(f"⚠️  Potential issue with {name}:")
                    print(f"   mean={mean:.4f}, std={std:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
                else:
                    print(f"✅ {name}: mean={mean:.4f}, std={std:.4f}")
        
        return stats
    
    def check_gradient_flow(self, model: nn.Module, batch: Tuple) -> Dict:
        """检查梯度流"""
        print("\n" + "="*60)
        print("2. Gradient Flow Check")
        print("="*60)
        
        model.train()
        model.zero_grad()
        
        img, label, bbox_gt = batch
        img = img.to(self.device)
        label = label.to(self.device)
        bbox_gt = bbox_gt.to(self.device)
        
        length_labels = t.ones(img.shape[0], dtype=t.long, device=self.device) * 2
        
        try:
            cls_list, bbox_list, length_logits = model(img, bbox_gt)
            
            total_loss = 0
            for h_idx, cls_out in enumerate(cls_list):
                total_loss += F.cross_entropy(cls_out, label[:, h_idx])
            total_loss = total_loss / len(cls_list)
            
            length_loss = F.cross_entropy(length_logits, length_labels)
            total_loss += length_loss * 0.1
            
            for h_idx, bbox_out in enumerate(bbox_list):
                if bbox_gt is not None:
                    bbox_target = bbox_gt[:, h_idx, :]
                    bbox_loss = F.mse_loss(bbox_out, bbox_target)
                    total_loss += bbox_loss * 0.5
            
            print(f"  Total loss: {total_loss.item():.4f}")
            print(f"  Classification loss: {(total_loss - length_loss * 0.1).item():.4f}")
            print(f"  Length loss: {length_loss.item():.4f}")
            
            total_loss.backward()
            
            grad_stats = {}
            all_grad_norms = []
            has_nan_count = 0
            has_inf_count = 0
            small_grad_count = 0
            large_grad_count = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_mean = param.grad.mean().item()
                        grad_std = param.grad.std().item()
                        has_nan = t.isnan(param.grad).any().item()
                        has_inf = t.isinf(param.grad).any().item()
                        
                        all_grad_norms.append(grad_norm)
                        if has_nan:
                            has_nan_count += 1
                        elif has_inf:
                            has_inf_count += 1
                        elif grad_norm < 1e-7:
                            small_grad_count += 1
                        elif grad_norm > 100:
                            large_grad_count += 1
                        
                        grad_stats[name] = {
                            'norm': grad_norm,
                            'mean': grad_mean,
                            'std': grad_std,
                            'has_nan': has_nan,
                            'has_inf': has_inf
                        }
                        
                        if has_nan:
                            print(f"❌ {name}: NaN detected in gradient!")
                        elif has_inf:
                            print(f"❌ {name}: Inf detected in gradient!")
                        elif grad_norm < 1e-7:
                            print(f"⚠️  {name}: Very small gradient (norm={grad_norm:.2e})")
                        elif grad_norm > 100:
                            print(f"⚠️  {name}: Very large gradient (norm={grad_norm:.2e})")
                    else:
                        print(f"❌ {name}: No gradient!")
                        grad_stats[name] = {'error': 'no_gradient'}
            
            print(f"\n  Gradient Statistics Summary:")
            if all_grad_norms:
                print(f"    Total params with grad: {len(all_grad_norms)}")
                print(f"    Min grad norm: {min(all_grad_norms):.4e}")
                print(f"    Max grad norm: {max(all_grad_norms):.4e}")
                print(f"    Mean grad norm: {sum(all_grad_norms)/len(all_grad_norms):.4e}")
                print(f"    NaN gradients: {has_nan_count}")
                print(f"    Inf gradients: {has_inf_count}")
                print(f"    Very small gradients (<1e-7): {small_grad_count}")
                print(f"    Very large gradients (>100): {large_grad_count}")
            
            return grad_stats
            
        except Exception as e:
            print(f"❌ Error during gradient flow check: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def check_module_activations(self, model: nn.Module, batch: Tuple) -> Dict:
        """检查模块激活值"""
        print("\n" + "="*60)
        print("3. Module Activation Check")
        print("="*60)
        
        model.eval()
        img, _, _ = batch
        img = img.to(self.device)
        
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        with t.no_grad():
            _ = model(img)
        
        for hook in hooks:
            hook.remove()
        
        activation_stats = {}
        dead_neurons = 0
        total_neurons = 0
        
        for name, act in activations.items():
            mean = act.mean().item()
            std = act.std().item()
            max_val = act.max().item()
            min_val = act.min().item()
            zero_count = (act == 0).sum().item()
            total_neurons += act.numel()
            dead_neurons += zero_count
            
            activation_stats[name] = {
                'mean': mean,
                'std': std,
                'max': max_val,
                'min': min_val,
                'zero_ratio': zero_count / act.numel()
            }
            
            if std < 1e-6:
                print(f"⚠️  {name}: Activation has very low std (std={std:.2e})")
            elif abs(mean) > 5:
                print(f"⚠️  {name}: Activation has large mean (mean={mean:.4f})")
            else:
                print(f"✅ {name}: mean={mean:.4f}, std={std:.4f}")
        
        print(f"\n  Activation Statistics Summary:")
        if total_neurons > 0:
            print(f"    Total neurons: {total_neurons:,}")
            print(f"    Dead neurons (zero activation): {dead_neurons:,} ({dead_neurons/total_neurons*100:.2f}%)")
        
        return activation_stats
    
    def check_attention_gradients(self, model: nn.Module, batch: Tuple):
        """检查注意力机制的梯度"""
        print("\n" + "="*60)
        print("4. Attention Gradient Check")
        print("="*60)
        
        model.train()
        model.zero_grad()
        
        img, label, bbox_gt = batch
        img = img.to(self.device)
        label = label.to(self.device)
        bbox_gt = bbox_gt.to(self.device)
        
        try:
            cls_list, bbox_list, attn_maps, head_cls_outs, length_logits = model.forward_with_attn(img, gt_bboxes=bbox_gt)
            
            for attn_map in attn_maps:
                attn_map.retain_grad()
            
            total_loss = 0
            for h_idx, cls_out in enumerate(head_cls_outs):
                total_loss += F.cross_entropy(cls_out, label[:, h_idx])
            total_loss = total_loss / len(head_cls_outs)
            
            if len(attn_maps) > 0:
                attn_sup_loss = 0
                for h_idx, attn_map in enumerate(attn_maps):
                    attn_map_flat = attn_map.flatten(2)
                    _, topk_indices = t.topk(attn_map_flat, k=5, dim=-1)
                    target = t.zeros_like(attn_map_flat)
                    batch_idx = t.arange(target.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1, target.size(1), 5)
                    channel_idx = t.arange(target.size(1)).unsqueeze(0).unsqueeze(-1).expand(target.size(0), -1, 5)
                    target[batch_idx, channel_idx, topk_indices] = 1.0
                    attn_sup_loss += F.mse_loss(attn_map_flat, target)
                if len(attn_maps) > 0:
                    attn_sup_loss = attn_sup_loss / len(attn_maps)
                    total_loss += attn_sup_loss * 100.0
                    print(f"  Added attention supervision loss: {attn_sup_loss.item():.4f}")
            
            total_loss.backward()
            
            print("\n  Attention Map Gradient Analysis:")
            for h_idx, attn_map in enumerate(attn_maps):
                if attn_map.grad is not None:
                    grad_norm = attn_map.grad.norm().item()
                    grad_mean = attn_map.grad.mean().item()
                    has_nan = t.isnan(attn_map.grad).any().item()
                    has_inf = t.isinf(attn_map.grad).any().item()
                    
                    print(f"    Head {h_idx}:")
                    print(f"      grad_norm={grad_norm:.4f}, grad_mean={grad_mean:.4f}")
                    if has_nan:
                        print("      ❌ NaN detected!")
                    elif has_inf:
                        print("      ❌ Inf detected!")
                    else:
                        print("      ✅ OK")
                else:
                    print(f"    Head {h_idx}: ❌ No gradient")
            
            model.zero_grad()
            
        except Exception as e:
            print(f"❌ Error during attention gradient check: {e}")
            import traceback
            traceback.print_exc()
    
    def check_loss_function_gradients(self, model: nn.Module, batch: Tuple):
        """检查各损失函数的梯度贡献"""
        print("\n" + "="*60)
        print("5. Loss Function Gradient Contribution")
        print("="*60)
        
        model.train()
        
        img, label, bbox_gt = batch
        img = img.to(self.device)
        label = label.to(self.device)
        bbox_gt = bbox_gt.to(self.device)
        length_labels = t.ones(img.shape[0], dtype=t.long, device=self.device) * 2
        
        loss_grad_norms = {}
        
        cls_list, bbox_list, length_logits = model(img, bbox_gt)
        
        loss_types = ['classification', 'bbox', 'length']
        for loss_type in loss_types:
            model.zero_grad()
            
            if loss_type == 'classification':
                loss = 0
                for h_idx, cls_out in enumerate(cls_list):
                    loss += F.cross_entropy(cls_out, label[:, h_idx])
                loss = loss / len(cls_list)
            elif loss_type == 'bbox':
                loss = 0
                for h_idx, bbox_out in enumerate(bbox_list):
                    if bbox_gt is not None:
                        bbox_target = bbox_gt[:, h_idx, :]
                        loss += F.mse_loss(bbox_out, bbox_target)
            elif loss_type == 'length':
                loss = F.cross_entropy(length_logits, length_labels)
            
            loss.backward(retain_graph=True)
            
            grad_norm_sum = 0
            param_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm_sum += param.grad.norm().item() ** 2
                    param_count += 1
            
            loss_grad_norms[loss_type] = (grad_norm_sum ** 0.5) / max(param_count, 1)
        
        print("\n  Average Gradient Norms per Loss Type:")
        for loss_type, avg_norm in loss_grad_norms.items():
            print(f"    {loss_type}: {avg_norm:.4e}")
        
        max_norm = max(loss_grad_norms.values())
        print("\n  Gradient Balance Analysis:")
        for loss_type, avg_norm in loss_grad_norms.items():
            ratio = avg_norm / max_norm
            if ratio < 0.1:
                print(f"    ⚠️  {loss_type}: gradient too weak (ratio={ratio:.2f})")
            elif ratio > 10:
                print(f"    ⚠️  {loss_type}: gradient too strong (ratio={ratio:.2f})")
            else:
                print(f"    ✅ {loss_type}: balanced (ratio={ratio:.2f})")
    
    def check_loss_functions(self):
        """检查损失函数"""
        print("\n" + "="*60)
        print("6. Loss Function Check")
        print("="*60)
        
        print("\nTesting FocalLoss:")
        focal_loss = FocalLoss(gamma=2.0)
        test_preds = t.randn(4, 11)
        test_targets = t.randint(0, 10, (4,))
        loss_val = focal_loss(test_preds, test_targets)
        print(f"   FocalLoss output: {loss_val.item():.4f}")
        if t.isfinite(loss_val):
            print("   ✅ FocalLoss produces finite values")
        else:
            print("   ❌ FocalLoss produces non-finite values")
        
        print("\nTesting LabelSmoothEntropy:")
        ls_loss = LabelSmoothEntropy(smooth=0.1)
        loss_val = ls_loss(test_preds, test_targets)
        print(f"   LabelSmoothEntropy output: {loss_val.item():.4f}")
        if t.isfinite(loss_val):
            print("   ✅ LabelSmoothEntropy produces finite values")
        else:
            print("   ❌ LabelSmoothEntropy produces non-finite values")
    
    def check_optimizer_setup(self, model: nn.Module):
        """检查优化器设置"""
        print("\n" + "="*60)
        print("7. Optimizer Setup Check")
        print("="*60)
        
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        print(f"   Backbone params: {len(backbone_params)}")
        print(f"   Other params: {len(other_params)}")
        print(f"   Total params: {len(list(model.parameters()))}")
        
        print(f"\n   Configured LR: {config.lr}")
        print(f"   Backbone LR factor: {config.backbone_lr_factor}")
        print(f"   Backbone effective LR: {config.lr * config.backbone_lr_factor}")
        
        if config.grad_clip_max_norm < 1.0:
            print(f"   ⚠️  Gradient clipping threshold: {config.grad_clip_max_norm} (may be too small)")
        elif config.grad_clip_max_norm > 5.0:
            print(f"   ⚠️  Gradient clipping threshold: {config.grad_clip_max_norm} (may be too large)")
        else:
            print(f"   ✅ Gradient clipping threshold: {config.grad_clip_max_norm} (appropriate)")
        
        print(f"\n   Mixed Precision Settings:")
        print(f"     use_amp: {config.use_amp}")
        print(f"     use_bf16: {config.use_bf16}")
        if config.use_bf16:
            print("     ⚠️  BF16 may cause gradient underflow for small gradients")
        else:
            print("     ✅ Using FP16 (safer for gradient stability)")
    
    def run_full_diagnostic(self):
        """运行完整诊断"""
        print("="*60)
        print("Street Character Recognition - Gradient Diagnostic")
        print("="*60)
        
        print("\n[INFO] Creating model...")
        model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads)
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Total params: {total_params:,}")
        print(f"[INFO] Trainable params: {trainable_params:,}")
        
        print("\n[INFO] Using dummy test data with batch_size=2...")
        img = t.randn(2, 3, config.input_height, config.input_width)
        label = t.randint(0, config.class_num - 1, (2, config.num_heads))
        bbox = t.rand(2, config.num_heads, 4)
        batch = (img, label, bbox)
        
        self.check_param_initialization(model)
        self.check_gradient_flow(model, batch)
        self.check_module_activations(model, batch)
        self.check_attention_gradients(model, batch)
        self.check_loss_function_gradients(model, batch)
        self.check_loss_functions()
        self.check_optimizer_setup(model)
        
        print("\n" + "="*60)
        print("Diagnostic Complete!")
        print("="*60)


def main():
    """主函数"""
    diagnostic = GradientDiagnostic()
    diagnostic.run_full_diagnostic()


if __name__ == '__main__':
    main()