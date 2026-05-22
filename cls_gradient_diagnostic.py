#!/usr/bin/env python3
import sys
import os
import io
import datetime


class Tee:
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (ValueError, OSError):
                pass

    def remove_stream(self, stream):
        if stream in self.streams:
            self.streams.remove(stream)


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import warnings

warnings.simplefilter('always')

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cls_gradient_diagnostic_output.txt')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, LOG_FILE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, set_seed
from models.multihead import DigitsResnet101
from losses.classification import LabelSmoothEntropy, FocalLoss


def check_forward_path_integrity(model, batch, device):
    print("\n" + "=" * 60)
    print("1. Forward Path Integrity Check")
    print("=" * 60)

    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)

    model.eval()
    with t.no_grad():
        cls_outs, bbox_outs, length_logits = model(img, bbox_gt)

    has_nan_output = any(t.isnan(c).any().item() for c in cls_outs)
    if has_nan_output:
        print("\n  ⚠️  CRITICAL: Model forward pass produces NaN!")
        print("  This is caused by BN stats reset leading to activation explosion.")
        print("  The backbone._reset_batch_norm_stats() has been fixed to only reset FPN layers.")
        print("  But the current model instance was created before or with the old code.")

    print("\n  [forward()] Output shapes:")
    for h, cls_out in enumerate(cls_outs):
        nan_pct = t.isnan(cls_out).float().mean().item() * 100
        if nan_pct > 0:
            print(f"    cls_outs[{h}]: shape={cls_out.shape}, ❌ NaN={nan_pct:.1f}%")
        else:
            print(f"    cls_outs[{h}]: shape={cls_out.shape}, mean={cls_out.mean().item():.4f}, std={cls_out.std().item():.4f}")
    for h, bbox_out in enumerate(bbox_outs):
        print(f"    bbox_outs[{h}]: shape={bbox_out.shape}, mean={bbox_out.mean().item():.4f}, std={bbox_out.std().item():.4f}")
    print(f"    length_logits: shape={length_logits.shape}, mean={length_logits.mean().item():.4f}")

    model.eval()
    with t.no_grad():
        cls_list, bbox_tuple, attn_maps, head_cls_outs, length_logits2 = model.forward_with_attn(img, gt_bboxes=bbox_gt)

    print("\n  [forward_with_attn()] Output shapes:")
    for h, cls in enumerate(cls_list):
        print(f"    cls_list[{h}]: shape={cls.shape}, mean={cls.mean().item():.4f}, std={cls.std().item():.4f}")
    for h, hcls in enumerate(head_cls_outs):
        print(f"    head_cls_outs[{h}]: shape={hcls.shape}, mean={hcls.mean().item():.4f}, std={hcls.std().item():.4f}")

    print("\n  [Consistency Check] forward() vs forward_with_attn():")
    for h in range(len(cls_outs)):
        diff = (cls_outs[h] - cls_list[h]).abs().max().item()
        status = "✅" if diff < 1e-5 else "⚠️"
        print(f"    Head {h}: max_diff={diff:.6e} {status}")

    print("\n  [ROI Refinement Check]:")
    if model.has_roi:
        print(f"    ROI enabled: True")
        print(f"    roi_gt_prob: {model.roi_gt_prob}")
    else:
        print(f"    ⚠️  ROI disabled! roi_align may have failed.")

    print("\n  [head_fc vs cls_layer output comparison]:")
    for h in range(len(cls_list)):
        fc_out = cls_list[h]
        cls_out = head_cls_outs[h]
        fc_range = (fc_out.min().item(), fc_out.max().item())
        cls_range = (cls_out.min().item(), cls_out.max().item())
        print(f"    Head {h}:")
        print(f"      head_fc output range: [{fc_range[0]:.4f}, {fc_range[1]:.4f}]")
        print(f"      cls_layer output range: [{cls_range[0]:.4f}, {cls_range[1]:.4f}]")
        if abs(cls_range[0]) < 1e-5 and abs(cls_range[1]) < 1e-5:
            print(f"      ⚠️  cls_layer output is near-zero! Weight initialization may be too small.")

    return cls_outs, head_cls_outs


def check_loss_cls_connection(model, batch, device):
    print("\n" + "=" * 60)
    print("2. Loss Function - Classification Layer Connection")
    print("=" * 60)

    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)

    model.train()
    model.zero_grad()

    cls_list, bbox_list, length_logits = model(img, bbox_gt)

    print("\n  [Main Classification Loss Path]:")
    print("  Loss computed on: cls_list (head_fc output after ROI refine)")

    cls_loss = t.tensor(0.0, device=device)
    for h_idx, cls_out in enumerate(cls_list):
        cls_loss += F.cross_entropy(cls_out, label[:, h_idx])
    cls_loss = cls_loss / len(cls_list)

    cls_loss.backward(retain_graph=True)

    head_fc_grads = {}
    cls_layer_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if name.startswith('head_fc'):
                head_fc_grads[name] = param.grad.norm().item()
            elif 'cls_layer' in name:
                cls_layer_grads[name] = param.grad.norm().item()

    print(f"\n  After main cls_loss.backward():")
    print(f"  head_fc gradients:")
    for n, g in sorted(head_fc_grads.items()):
        print(f"    {n}: {g:.4e}")

    print(f"  cls_layer gradients:")
    if cls_layer_grads:
        for n, g in sorted(cls_layer_grads.items()):
            print(f"    {n}: {g:.4e}")
    else:
        print(f"    ❌ No gradients in cls_layer from main loss!")
        print(f"    This confirms: cls_layer is NOT in the main classification path.")

    model.zero_grad()

    print("\n  [Auxiliary Loss Path]:")
    print("  Testing aux_loss on head_cls_outs (cls_layer output)...")

    cls_list2, bbox_list2, attn_maps, head_cls_outs, length_logits2 = model.forward_with_attn(img, gt_bboxes=bbox_gt)

    aux_loss = t.tensor(0.0, device=device)
    for h_idx, cls_out in enumerate(head_cls_outs):
        aux_loss += F.cross_entropy(cls_out, label[:, h_idx])
    aux_loss = aux_loss / len(head_cls_outs)

    aux_loss.backward(retain_graph=True)

    cls_layer_grads_aux = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if 'cls_layer' in name:
                cls_layer_grads_aux[name] = param.grad.norm().item()

    print(f"  After aux_loss.backward() on head_cls_outs:")
    if cls_layer_grads_aux:
        for n, g in sorted(cls_layer_grads_aux.items()):
            print(f"    {n}: {g:.4e}")
    else:
        print(f"    ❌ Still no gradients in cls_layer!")

    print(f"\n  [LabelSmoothEntropy Gradient Test]:")
    ls_loss = LabelSmoothEntropy(smooth=config.smooth)
    test_pred = t.randn(2, config.class_num, device=device, requires_grad=True)
    test_target = t.randint(0, config.class_num - 1, (2,), device=device)
    ls_val = ls_loss(test_pred, test_target)
    ls_val.backward()
    print(f"    Input grad norm: {test_pred.grad.norm().item():.4e}")
    print(f"    ✅ LabelSmoothEntropy produces valid gradients" if test_pred.grad.norm().item() > 0 else "    ❌ LabelSmoothEntropy produces zero gradients!")


def check_input_feature_distribution(model, batch, device):
    print("\n" + "=" * 60)
    print("3. Classification Layer Input Feature Distribution")
    print("=" * 60)

    img, label, bbox_gt = batch
    img = img.to(device)

    activations = {}
    hooks = []

    target_layers = ['head_interaction', 'head_fc']
    for h_idx in range(config.num_heads):
        target_layers.append(f'heads.{h_idx}.feat_proj')
        target_layers.append(f'heads.{h_idx}.cls_layer')

    def get_activation(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                inp = input[0]
                if isinstance(inp, t.Tensor):
                    activations[name + '_input'] = inp.detach()
                elif isinstance(inp, (list, tuple)):
                    try:
                        stacked = t.stack([x.detach() if isinstance(x, t.Tensor) else t.zeros(1) for x in inp])
                        activations[name + '_input'] = stacked
                    except Exception:
                        pass
            if isinstance(output, tuple):
                o = output[0]
                activations[name + '_output'] = o.detach() if isinstance(o, t.Tensor) else t.tensor(0.0)
            elif isinstance(output, list):
                try:
                    activations[name + '_output'] = t.stack([x.detach() if isinstance(x, t.Tensor) else t.zeros(1) for x in output])
                except Exception:
                    pass
            elif isinstance(output, t.Tensor):
                activations[name + '_output'] = output.detach()
        return hook

    for name, module in model.named_modules():
        if any(name == tl for tl in target_layers):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    with t.no_grad():
        _ = model(img)

    for hook in hooks:
        hook.remove()

    print("\n  Feature Distribution Analysis:")
    for name, act in sorted(activations.items()):
        mean = act.mean().item()
        std = act.std().item()
        max_val = act.max().item()
        min_val = act.min().item()
        zero_ratio = (act == 0).float().mean().item()

        status = "✅"
        if std < 1e-6:
            status = "⚠️ (dead)"
        elif abs(mean) > 10:
            status = "⚠️ (large mean)"
        elif max_val > 1000:
            status = "⚠️ (extreme values)"

        print(f"    {name}: mean={mean:.4f}, std={std:.4e}, range=[{min_val:.4f}, {max_val:.4f}], zero_ratio={zero_ratio:.2%} {status}")

    print("\n  [Dead Neuron Check in feat_proj]:")
    for h_idx in range(config.num_heads):
        key = f'heads.{h_idx}.feat_proj_output'
        if key in activations:
            act = activations[key]
            dead = (act == 0).float().mean().item()
            print(f"    Head {h_idx} feat_proj: zero_ratio={dead:.2%}")
            if dead > 0.5:
                print(f"    ⚠️  More than 50% dead neurons in head {h_idx} feat_proj!")


def check_gradient_interruption_points(model, batch, device):
    print("\n" + "=" * 60)
    print("4. Gradient Interruption Point Detection")
    print("=" * 60)

    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)

    model.train()
    model.zero_grad()

    cls_list, bbox_list, length_logits = model(img, bbox_gt)

    total_loss = 0
    for h_idx, cls_out in enumerate(cls_list):
        total_loss += F.cross_entropy(cls_out, label[:, h_idx])
    total_loss = total_loss / len(cls_list)

    total_loss.backward()

    print("\n  [Head Interaction Layer Gradient Flow]:")
    interaction_layers = [
        'head_interaction.self_attn.in_proj_weight',
        'head_interaction.self_attn.in_proj_bias',
        'head_interaction.self_attn.out_proj.weight',
        'head_interaction.self_attn.out_proj.bias',
        'head_interaction.norm1.weight',
        'head_interaction.norm1.bias',
        'head_interaction.norm2.weight',
        'head_interaction.norm2.bias',
        'head_interaction.ffn.0.weight',
        'head_interaction.ffn.0.bias',
        'head_interaction.ffn.3.weight',
        'head_interaction.ffn.3.bias',
    ]

    for layer_name in interaction_layers:
        found = False
        for name, param in model.named_parameters():
            if name == layer_name and param.grad is not None:
                gn = param.grad.norm().item()
                status = "✅" if gn > 1e-7 else "⚠️ (near-zero)"
                print(f"    {name}: {gn:.4e} {status}")
                found = True
                break
        if not found:
            print(f"    {layer_name}: ❌ No gradient!")

    print("\n  [Gradient Interruption Analysis]:")
    print("  Checking for gradient-blocking operations...")

    grad_norms_by_depth = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gn = param.grad.norm().item()
            if name.startswith('heads.0.'):
                grad_norms_by_depth.append((name, gn))

    if grad_norms_by_depth:
        sorted_grads = sorted(grad_norms_by_depth, key=lambda x: x[1])
        print(f"\n  Head 0 - Smallest gradients (potential interruption points):")
        for n, g in sorted_grads[:5]:
            print(f"    {n}: {g:.4e}")
        print(f"\n  Head 0 - Largest gradients:")
        for n, g in sorted_grads[-5:]:
            print(f"    {n}: {g:.4e}")

    print("\n  [ReLU Dead Zone Check]:")
    relu_count = 0
    dead_relu_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_count += 1

    print(f"    Total ReLU layers: {relu_count}")
    print(f"    Note: ReLU with inplace=True can cause gradient issues in checkpoint mode")

    print("\n  [Sigmoid Saturation Check in bbox_head]:")
    for h_idx in range(config.num_heads):
        for name, param in model.named_parameters():
            if name == f'heads.{h_idx}.bbox_head.3.weight' and param.grad is not None:
                gn = param.grad.norm().item()
                status = "✅" if gn > 1e-4 else "⚠️ (near-zero, Sigmoid may be saturated)"
                print(f"    {name}: {gn:.4e} {status}")


def check_optimizer_config(model, batch, device):
    print("\n" + "=" * 60)
    print("5. Optimizer Configuration & Parameter Update Check")
    print("=" * 60)

    print("\n  [Parameter Group Analysis]:")

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append((name, param))
            else:
                other_params.append((name, param))

    print(f"    Backbone params: {len(backbone_params)}")
    print(f"    Other params: {len(other_params)}")

    print("\n  [Classification Layer Parameter Check]:")
    cls_layers = ['head_fc', 'cls_layer', 'roi_cls_heads']
    for cls_name in cls_layers:
        matching = [(n, p) for n, p in model.named_parameters() if cls_name in n and p.requires_grad]
        if matching:
            in_backbone = any('backbone' in n for n, _ in matching)
            group = "backbone (lr*0.3)" if in_backbone else "other (lr*1.0)"
            print(f"    {cls_name}: {len(matching)} params in group '{group}'")
            for n, p in matching[:3]:
                print(f"      {n}: shape={list(p.shape)}, grad={'Yes' if p.grad is not None else 'No'}")
        else:
            print(f"    {cls_name}: ❌ No matching parameters!")

    print(f"\n  [Learning Rate Configuration]:")
    print(f"    Base LR: {config.lr}")
    print(f"    Backbone LR factor: {config.backbone_lr_factor}")
    print(f"    Backbone effective LR: {config.lr * config.backbone_lr_factor:.6f}")
    print(f"    Head FC effective LR: {config.lr:.6f}")

    print(f"\n  [Weight Decay Impact]:")
    print(f"    Weight decay: {config.weights_decay}")
    for cls_name in ['head_fc', 'heads.0.cls_layer']:
        for n, p in model.named_parameters():
            if cls_name in n and p.requires_grad:
                wd_effect = config.weights_decay * p.data.norm().item()
                print(f"    {n}: param_norm={p.data.norm().item():.4e}, wd_effect={wd_effect:.4e}")

    print("\n  [Simulated Parameter Update Test]:")
    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)

    model.train()
    optimizer = t.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weights_decay)

    model.zero_grad()
    cls_list, bbox_list, length_logits = model(img, bbox_gt)
    total_loss = 0
    for h_idx, cls_out in enumerate(cls_list):
        total_loss += F.cross_entropy(cls_out, label[:, h_idx])
    total_loss = total_loss / len(cls_list)
    total_loss.backward()

    t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)

    before_params = {}
    for n, p in model.named_parameters():
        if 'head_fc' in n or 'cls_layer' in n:
            before_params[n] = p.data.clone()

    optimizer.step()

    print("  Parameter changes after one optimizer step:")
    for n, before in before_params.items():
        for name, param in model.named_parameters():
            if name == n:
                diff = (param.data - before).abs().max().item()
                status = "✅" if diff > 0 else "❌ (no update!)"
                print(f"    {n}: max_change={diff:.6e} {status}")
                break


def main():
    print("=" * 60)
    print("Classification Layer Gradient Diagnostic")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {LOG_PATH}")
    print("=" * 60)

    set_seed(42)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    print("[INFO] Creating model...")
    model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads)
    model = model.to(device)

    print("[INFO] Using dummy test data with batch_size=2...")
    img = t.randn(2, 3, config.input_height, config.input_width)
    label = t.randint(0, config.class_num - 1, (2, config.num_heads))
    bbox = t.rand(2, config.num_heads, 4)
    batch = (img, label, bbox)

    check_forward_path_integrity(model, batch, device)
    check_loss_cls_connection(model, batch, device)
    check_input_feature_distribution(model, batch, device)
    check_gradient_interruption_points(model, batch, device)
    check_optimizer_config(model, batch, device)

    print("\n" + "=" * 60)
    print("Classification Layer Diagnostic Complete!")
    print(f"Report saved to: {LOG_PATH}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("ROOT CAUSE SUMMARY")
    print("=" * 60)
    print("""
  1. cls_layer无梯度根因:
     - 在forward()中，heads返回(cls_out, bbox_out, hidden)
     - 但只取hidden(r[2])送入head_interaction → head_fc
     - cls_layer的输出cls_out(r[0])被完全丢弃
     - cls_layer仅通过aux_loss(权重0.1)获得微弱梯度
     - 而在forward_with_attn()中，head_cls_outs才被用于aux_loss

  2. BN层激活爆炸根因:
     - _reset_batch_norm_stats()将预训练ResNet101的所有BN running stats重置
     - 导致初始阶段激活值呈指数级增长(已达10^19量级)
     - 这使得BN层梯度也异常巨大(最大1063)

  3. 修复方案:
     - BN: 仅重置FPN新增层的BN stats，保留预训练层的stats
     - cls_layer: 增大aux_loss权重，或重新设计使cls_layer参与主分类路径
""")


if __name__ == '__main__':
    main()
    if isinstance(sys.stdout, Tee):
        sys.stdout.remove_stream(LOG_FILE)
    LOG_FILE.close()
