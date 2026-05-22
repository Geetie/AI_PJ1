#!/usr/bin/env python3
import sys
import os
import io
import json
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
from typing import Dict, List, Tuple
import warnings

warnings.simplefilter('always')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradient_analysis_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'gradient_analysis_report.txt')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, LOG_FILE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, set_seed
from models.multihead import DigitsResnet101


def classify_param(name):
    if 'bn' in name.lower() or 'norm' in name.lower():
        return 'BN'
    elif 'conv' in name.lower():
        return 'Conv'
    elif 'linear' in name.lower() or 'head_fc' in name or 'cls_layer' in name or 'feat_proj' in name:
        return 'Linear'
    elif 'embedding' in name.lower() or 'pos_embed' in name or 'head_embed' in name:
        return 'Embedding'
    else:
        return 'Other'


def analyze_gradient_distribution(model, batch, device):
    print("\n" + "=" * 60)
    print("1. Gradient Distribution Analysis")
    print("=" * 60)

    model.train()
    model.zero_grad()

    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)
    length_labels = t.ones(img.shape[0], dtype=t.long, device=device) * 2

    cls_list, bbox_list, length_logits = model(img, bbox_gt)

    total_loss = 0
    for h_idx, cls_out in enumerate(cls_list):
        total_loss += F.cross_entropy(cls_out, label[:, h_idx])
    total_loss = total_loss / len(cls_list)
    length_loss = F.cross_entropy(length_logits, length_labels)
    total_loss += length_loss * 0.1
    for h_idx, bbox_out in enumerate(bbox_list):
        if bbox_gt is not None:
            bbox_loss = F.mse_loss(bbox_out, bbox_gt[:, h_idx, :])
            total_loss += bbox_loss * 0.5

    total_loss.backward()

    category_stats = {'BN': [], 'Conv': [], 'Linear': [], 'Embedding': [], 'Other': []}
    all_grad_norms = []
    nan_count = 0
    inf_count = 0
    zero_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            cat = classify_param(name)
            category_stats[cat].append((name, grad_norm, param.grad))
            all_grad_norms.append((name, grad_norm))

            if t.isnan(param.grad).any():
                nan_count += 1
            elif t.isinf(param.grad).any():
                inf_count += 1
            elif grad_norm < 1e-7:
                zero_count += 1

    for cat, entries in category_stats.items():
        if not entries:
            continue
        norms = [e[1] for e in entries]
        print(f"\n  [{cat}] Layer count: {len(norms)}")
        print(f"    Min norm: {min(norms):.4e}")
        print(f"    Max norm: {max(norms):.4e}")
        print(f"    Mean norm: {sum(norms)/len(norms):.4e}")
        if cat == 'BN':
            large_bn = [(n, g) for n, g, _ in entries if g > 10]
            if large_bn:
                print(f"    ⚠️  BN layers with large gradient (>10): {len(large_bn)}")
                for n, g in sorted(large_bn, key=lambda x: -x[1])[:5]:
                    print(f"      {n}: {g:.4e}")

    print(f"\n  Overall Gradient Summary:")
    print(f"    Total params with grad: {len(all_grad_norms)}")
    print(f"    NaN gradients: {nan_count}")
    print(f"    Inf gradients: {inf_count}")
    print(f"    Near-zero gradients (<1e-7): {zero_count}")

    return category_stats, all_grad_norms


def analyze_gradient_propagation(model, batch, device):
    print("\n" + "=" * 60)
    print("2. Gradient Propagation Path Analysis")
    print("=" * 60)

    model.train()
    model.zero_grad()

    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)
    length_labels = t.ones(img.shape[0], dtype=t.long, device=device) * 2

    cls_list, bbox_list, length_logits = model(img, bbox_gt)

    total_loss = 0
    for h_idx, cls_out in enumerate(cls_list):
        total_loss += F.cross_entropy(cls_out, label[:, h_idx])
    total_loss = total_loss / len(cls_list)
    length_loss = F.cross_entropy(length_logits, length_labels)
    total_loss += length_loss * 0.1
    for h_idx, bbox_out in enumerate(bbox_list):
        if bbox_gt is not None:
            bbox_loss = F.mse_loss(bbox_out, bbox_gt[:, h_idx, :])
            total_loss += bbox_loss * 0.5

    total_loss.backward()

    backbone_grads = []
    fpn_grads = []
    head_grads = []
    interaction_grads = []
    cls_grads = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if name.startswith('backbone.layer'):
                backbone_grads.append((name, grad_norm))
            elif name.startswith('backbone.') and not name.startswith('backbone.layer'):
                fpn_grads.append((name, grad_norm))
            elif name.startswith('heads.'):
                head_grads.append((name, grad_norm))
            elif name.startswith('head_interaction') or name.startswith('head_fc'):
                interaction_grads.append((name, grad_norm))
            elif 'cls_layer' in name or 'head_fc' in name:
                cls_grads.append((name, grad_norm))

    print("\n  [Backbone ResNet Layers] Gradient Norms (last 5):")
    for n, g in sorted(backbone_grads, key=lambda x: x[0])[-5:]:
        print(f"    {n}: {g:.4e}")
    if backbone_grads:
        bb_norms = [g for _, g in backbone_grads]
        print(f"    Backbone avg: {sum(bb_norms)/len(bb_norms):.4e}")

    print("\n  [FPN Layers] Gradient Norms:")
    for n, g in sorted(fpn_grads, key=lambda x: x[0]):
        print(f"    {n}: {g:.4e}")

    print("\n  [Head Interaction + Classification] Gradient Norms:")
    for n, g in sorted(interaction_grads, key=lambda x: x[0]):
        print(f"    {n}: {g:.4e}")

    print("\n  [Classification-specific layers] Gradient Norms:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if 'cls_layer' in name or 'head_fc' in name or 'roi_cls' in name:
                print(f"    {name}: {param.grad.norm().item():.4e}")

    if backbone_grads and interaction_grads:
        bb_avg = sum(g for _, g in backbone_grads) / len(backbone_grads)
        int_avg = sum(g for _, g in interaction_grads) / len(interaction_grads)
        ratio = int_avg / max(bb_avg, 1e-10)
        print(f"\n  Gradient ratio (interaction/backbone): {ratio:.2f}x")
        if ratio > 100:
            print(f"    ⚠️  Severe gradient amplification from backbone to classification!")
        elif ratio > 10:
            print(f"    ⚠️  Notable gradient amplification detected")

    return {
        'backbone': backbone_grads,
        'fpn': fpn_grads,
        'head': head_grads,
        'interaction': interaction_grads,
        'cls': cls_grads,
    }


def analyze_bn_gradient_ratio(model, batch, device):
    print("\n" + "=" * 60)
    print("3. BN Layer Gradient Amplification Analysis")
    print("=" * 60)

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
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(get_activation(name)))

    model.eval()
    img, _, _ = batch
    img = img.to(device)

    with t.no_grad():
        _ = model(img)

    for hook in hooks:
        hook.remove()

    print(f"\n  BN Layer Input/Output Statistics:")
    bn_issues = []
    for name, act in activations.items():
        mean = act.mean().item()
        std = act.std().item()
        max_val = act.max().item()
        min_val = act.min().item()

        if std < 1e-6 or abs(mean) > 10 or max_val > 100:
            bn_issues.append((name, mean, std, max_val, min_val))
            print(f"    ⚠️  {name}: mean={mean:.4f}, std={std:.4e}, range=[{min_val:.4f}, {max_val:.4f}]")

    if not bn_issues:
        print("    ✅ All BN layer activations are within normal range")
    else:
        print(f"\n    Total BN layers with abnormal activations: {len(bn_issues)}")

    model.train()
    model.zero_grad()
    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)
    length_labels = t.ones(img.shape[0], dtype=t.long, device=device) * 2

    cls_list, bbox_list, length_logits = model(img, bbox_gt)
    total_loss = 0
    for h_idx, cls_out in enumerate(cls_list):
        total_loss += F.cross_entropy(cls_out, label[:, h_idx])
    total_loss = total_loss / len(cls_list)
    length_loss = F.cross_entropy(length_logits, length_labels)
    total_loss += length_loss * 0.1
    for h_idx, bbox_out in enumerate(bbox_list):
        if bbox_gt is not None:
            total_loss += F.mse_loss(bbox_out, bbox_gt[:, h_idx, :]) * 0.5
    total_loss.backward()

    print(f"\n  BN Parameter Gradient Analysis:")
    bn_weight_grads = []
    bn_bias_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if 'bn' in name.lower() or 'norm' in name.lower():
                grad_norm = param.grad.norm().item()
                if 'weight' in name:
                    bn_weight_grads.append((name, grad_norm))
                elif 'bias' in name:
                    bn_bias_grads.append((name, grad_norm))

    if bn_weight_grads:
        wg_norms = [g for _, g in bn_weight_grads]
        print(f"    BN weight grad norms: min={min(wg_norms):.4e}, max={max(wg_norms):.4e}, avg={sum(wg_norms)/len(wg_norms):.4e}")
        large_wg = [(n, g) for n, g in bn_weight_grads if g > 10]
        if large_wg:
            print(f"    ⚠️  BN weights with large gradient (>10): {len(large_wg)}")
            for n, g in sorted(large_wg, key=lambda x: -x[1])[:5]:
                print(f"      {n}: {g:.4e}")

    if bn_bias_grads:
        bg_norms = [g for _, g in bn_bias_grads]
        print(f"    BN bias grad norms: min={min(bg_norms):.4e}, max={max(bg_norms):.4e}, avg={sum(bg_norms)/len(bg_norms):.4e}")

    return bn_issues


def analyze_cls_gradient_path(model, batch, device):
    print("\n" + "=" * 60)
    print("4. Classification Layer Gradient Path Analysis")
    print("=" * 60)

    model.train()
    model.zero_grad()

    img, label, bbox_gt = batch
    img = img.to(device)
    label = label.to(device)
    bbox_gt = bbox_gt.to(device)

    cls_list, bbox_list, length_logits = model(img, bbox_gt)

    total_loss = 0
    for h_idx, cls_out in enumerate(cls_list):
        total_loss += F.cross_entropy(cls_out, label[:, h_idx])
    total_loss = total_loss / len(cls_list)

    total_loss.backward()

    print("\n  Main Classification Path Gradient Flow:")
    path_layers = [
        'head_fc', 'head_interaction',
        'heads.0.feat_proj', 'heads.1.feat_proj', 'heads.2.feat_proj',
        'heads.0.cls_layer', 'heads.1.cls_layer', 'heads.2.cls_layer',
        'heads.0.attention_conv', 'heads.1.attention_conv', 'heads.2.attention_conv',
        'pre_head_comm',
        'backbone.fuse', 'backbone.se',
    ]

    for path_name in path_layers:
        matching_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name.startswith(path_name):
                matching_grads.append((name, param.grad.norm().item()))

        if matching_grads:
            avg_norm = sum(g for _, g in matching_grads) / len(matching_grads)
            max_norm = max(g for _, g in matching_grads)
            print(f"    {path_name}: avg={avg_norm:.4e}, max={max_norm:.4e}, params={len(matching_grads)}")
        else:
            print(f"    {path_name}: ❌ No gradients found!")

    print("\n  Gradient Flow from Loss to Backbone:")
    head_fc_norms = []
    interaction_norms = []
    head_norms = []
    backbone_norms = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gn = param.grad.norm().item()
            if name.startswith('head_fc'):
                head_fc_norms.append(gn)
            elif name.startswith('head_interaction'):
                interaction_norms.append(gn)
            elif name.startswith('heads.'):
                head_norms.append(gn)
            elif name.startswith('backbone.'):
                backbone_norms.append(gn)

    def avg_or_zero(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"    head_fc avg:       {avg_or_zero(head_fc_norms):.4e}")
    print(f"    interaction avg:   {avg_or_zero(interaction_norms):.4e}")
    print(f"    heads avg:         {avg_or_zero(head_norms):.4e}")
    print(f"    backbone avg:      {avg_or_zero(backbone_norms):.4e}")

    hf_avg = avg_or_zero(head_fc_norms)
    bb_avg = avg_or_zero(backbone_norms)
    if bb_avg > 0 and hf_avg > 0:
        ratio = hf_avg / bb_avg
        print(f"    Gradient ratio (head_fc/backbone): {ratio:.2f}x")

    return {
        'head_fc': head_fc_norms,
        'interaction': interaction_norms,
        'heads': head_norms,
        'backbone': backbone_norms,
    }


def simulate_training_steps(model, batch, device, num_steps=5):
    print("\n" + "=" * 60)
    print(f"5. Gradient Norm Evolution Over {num_steps} Training Steps")
    print("=" * 60)

    optimizer = t.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weights_decay)

    step_records = []

    for step in range(num_steps):
        model.train()
        model.zero_grad()

        img, label, bbox_gt = batch
        img = img.to(device)
        label = label.to(device)
        bbox_gt = bbox_gt.to(device)
        length_labels = t.ones(img.shape[0], dtype=t.long, device=device) * 2

        cls_list, bbox_list, length_logits = model(img, bbox_gt)

        total_loss = 0
        for h_idx, cls_out in enumerate(cls_list):
            total_loss += F.cross_entropy(cls_out, label[:, h_idx])
        total_loss = total_loss / len(cls_list)
        length_loss = F.cross_entropy(length_logits, length_labels)
        total_loss += length_loss * 0.1
        for h_idx, bbox_out in enumerate(bbox_list):
            if bbox_gt is not None:
                total_loss += F.mse_loss(bbox_out, bbox_gt[:, h_idx, :]) * 0.5

        total_loss.backward()

        record = {'step': step, 'loss': total_loss.item(), 'layers': {}}

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                record['layers'][name] = {
                    'norm': param.grad.norm().item(),
                    'has_nan': t.isnan(param.grad).any().item(),
                    'has_inf': t.isinf(param.grad).any().item(),
                }

        step_records.append(record)

        t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
        optimizer.step()

        if step == 0 or step == num_steps - 1:
            print(f"\n  Step {step}: loss={total_loss.item():.4f}")
            total_grad_norm = sum(r['norm'] ** 2 for r in record['layers'].values()) ** 0.5
            print(f"    Total grad norm: {total_grad_norm:.4e}")

            bn_grad_norm = sum(r['norm'] ** 2 for n, r in record['layers'].items() if 'bn' in n.lower() or 'norm' in n.lower()) ** 0.5
            non_bn_grad_norm = sum(r['norm'] ** 2 for n, r in record['layers'].items() if 'bn' not in n.lower() and 'norm' not in n.lower()) ** 0.5
            print(f"    BN grad norm: {bn_grad_norm:.4e}")
            print(f"    Non-BN grad norm: {non_bn_grad_norm:.4e}")

    json_path = os.path.join(OUTPUT_DIR, 'gradient_evolution.json')
    serializable = []
    for r in step_records:
        sr = {'step': r['step'], 'loss': r['loss'], 'layers': {}}
        for n, v in r['layers'].items():
            sr['layers'][n] = {'norm': v['norm'], 'has_nan': v['has_nan'], 'has_inf': v['has_inf']}
        serializable.append(sr)
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Gradient evolution data saved to: {json_path}")

    return step_records


def main():
    print("=" * 60)
    print("Street Character Recognition - Systematic Gradient Analysis")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    set_seed(42)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    print("[INFO] Creating model...")
    model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total_params:,}")
    print(f"[INFO] Trainable params: {trainable_params:,}")

    print("[INFO] Using dummy test data with batch_size=2...")
    img = t.randn(2, 3, config.input_height, config.input_width)
    label = t.randint(0, config.class_num - 1, (2, config.num_heads))
    bbox = t.rand(2, config.num_heads, 4)
    batch = (img, label, bbox)

    analyze_gradient_distribution(model, batch, device)
    analyze_gradient_propagation(model, batch, device)
    analyze_bn_gradient_ratio(model, batch, device)
    analyze_cls_gradient_path(model, batch, device)
    simulate_training_steps(model, batch, device, num_steps=5)

    print("\n" + "=" * 60)
    print("Gradient Analysis Complete!")
    print(f"Report saved to: {LOG_PATH}")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
    if isinstance(sys.stdout, Tee):
        sys.stdout.remove_stream(LOG_FILE)
    LOG_FILE.close()
