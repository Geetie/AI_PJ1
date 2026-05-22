#!/usr/bin/env python3
"""快速梯度检查脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as t
from config import config, set_seed
from models.multihead import DigitsResnet101
import torch.nn.functional as F

set_seed(42)
device = t.device('cpu')

print("Creating model...")
model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads).to(device)
model.train()

print("Creating test data...")
img = t.randn(2, 3, config.input_height, config.input_width).to(device)
label = t.randint(0, config.class_num - 1, (2, config.num_heads)).to(device)
bbox = t.rand(2, config.num_heads, 4).to(device)

print("Forward pass (with attention)...")
cls_outs, bbox_outs, attn_maps, head_cls_outs, length_logits = model.forward_with_attn(img, bbox)

print("Computing loss...")
total_loss = 0
for h_idx, cls_out in enumerate(cls_outs):
    total_loss += F.cross_entropy(cls_out, label[:, h_idx])
total_loss = total_loss / len(cls_outs)

aux_loss = t.tensor(0.0, device=device)
for h_idx, cls_out in enumerate(head_cls_outs):
    aux_loss += F.cross_entropy(cls_out, label[:, h_idx])
aux_loss = aux_loss / len(head_cls_outs)
total_loss = total_loss + 0.1 * aux_loss

bbox_loss = t.tensor(0.0, device=device)
for h_idx, bbox_out in enumerate(bbox_outs):
    bbox_loss_h = F.mse_loss(bbox_out, bbox[:, h_idx, :])
    bbox_loss = bbox_loss + bbox_loss_h
total_loss = total_loss + 0.5 * bbox_loss

length_labels = t.ones(img.shape[0], dtype=t.long, device=device) * 2
length_loss = F.cross_entropy(length_logits, length_labels)
total_loss += length_loss * 0.1

print(f"Total loss: {total_loss.item():.4f}")
print(f"  cls_loss: {(total_loss - 0.1 * aux_loss - 0.5 * bbox_loss - 0.1 * length_loss).item():.4f}")
print(f"  aux_loss: {aux_loss.item():.4f}")
print(f"  bbox_loss: {bbox_loss.item():.4f}")
print(f"  length_loss: {length_loss.item():.4f}")

print("Backward pass...")
model.zero_grad()
total_loss.backward()

print("\nChecking gradients...")
issues = []
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm < 1e-7:
            issues.append(f"Very small gradient: {name} (norm={grad_norm:.2e})")
        elif grad_norm > 1000:
            issues.append(f"Very large gradient: {name} (norm={grad_norm:.2e})")
    else:
        issues.append(f"No gradient: {name}")

if issues:
    print("\nGradient issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\nAll gradients are normal!")

print("\nGradient statistics:")
grad_norms = []
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norms.append((name, param.grad.norm().item()))

grad_norms.sort(key=lambda x: x[1])
print(f"  Total params with gradients: {len(grad_norms)}")
print(f"  Min grad norm: {grad_norms[0][1]:.6f} ({grad_norms[0][0]})")
print(f"  Max grad norm: {grad_norms[-1][1]:.6f} ({grad_norms[-1][0]})")
print(f"  Avg grad norm: {sum(n for _, n in grad_norms) / len(grad_norms):.6f}")

zero_grad_count = sum(1 for _, n in grad_norms if n < 1e-7)
large_grad_count = sum(1 for _, n in grad_norms if n > 1000)
print(f"\n  Zero/Very-small gradients: {zero_grad_count} ({zero_grad_count / len(grad_norms) * 100:.1f}%)")
print(f"  Very-large gradients: {large_grad_count} ({large_grad_count / len(grad_norms) * 100:.1f}%)")
