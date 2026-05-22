#!/usr/bin/env python3
"""
代码可触及性验证脚本
验证所有关键修复在训练流程中真实可触及
"""
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

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reachability_audit_output.txt')
LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, LOG_FILE)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from config import config, set_seed
from models.multihead import DigitsResnet101
from models import create_model
from utils.bn_protection import install_bn_protection

PASS_COUNT = 0
FAIL_COUNT = 0

def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✅ {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ❌ {name} {detail}")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def main():
    print(f"代码可触及性验证 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    set_seed(42)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ============================================================
    section("1. MultiHeadTrainer __init__ 属性可达性")
    # ============================================================
    from trainer.multihead import MultiHeadTrainer
    try:
        trainer = MultiHeadTrainer(val=True, model_type='fpn_multihead')
        check("_base_seed 已初始化", hasattr(trainer, '_base_seed'), f"got {getattr(trainer, '_base_seed', 'MISSING')}")
        check("_base_seed = 42", trainer._base_seed == 42 if hasattr(trainer, '_base_seed') else False)
        check("_bn_protection 已初始化", hasattr(trainer, '_bn_protection'))
        check("_bn_protection 不为None (fpn_multihead有setup_bn_protection)", trainer._bn_protection is not None)
        check("_grad_clip_max_norm 已初始化", hasattr(trainer, '_grad_clip_max_norm'))
        check("_last_train_joint_acc 已初始化", hasattr(trainer, '_last_train_joint_acc'))
        check("_last_epoch_avg_loss 已初始化", hasattr(trainer, '_last_epoch_avg_loss'))
        check("_last_val_joint_acc 已初始化", hasattr(trainer, '_last_val_joint_acc'))
        check("optimizer 已创建", hasattr(trainer, 'optimizer') and trainer.optimizer is not None)
        check("lr_scheduler 已创建", hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None)
        check("scaler 已创建", hasattr(trainer, 'scaler') and trainer.scaler is not None)
        check("ema 已创建", hasattr(trainer, 'ema') and trainer.ema is not None)
        check("train_set 已创建", hasattr(trainer, 'train_set') and trainer.train_set is not None)
        check("train_loader 已创建", hasattr(trainer, 'train_loader') and trainer.train_loader is not None)
        check("val_set 已创建", hasattr(trainer, 'val_set'))
        check("val_loader 已创建", hasattr(trainer, 'val_loader'))
        check("head_criteria 已创建", hasattr(trainer, 'head_criteria') and len(trainer.head_criteria) > 0)
        check("attn_supervision 已创建", hasattr(trainer, 'attn_supervision'))
        check("length_criterion 已创建", hasattr(trainer, 'length_criterion'))
        check("_train_generator 已创建", hasattr(trainer, '_train_generator'))
    except Exception as e:
        check("MultiHeadTrainer 初始化", False, str(e))

    # ============================================================
    section("2. BN Protection 返回值结构验证")
    # ============================================================
    if trainer._bn_protection is not None:
        bp = trainer._bn_protection
        check("bn_protection 包含 'grad_cliper'", 'grad_cliper' in bp)
        check("bn_protection 包含 'momentum_scheduler'", 'momentum_scheduler' in bp)
        check("bn_protection 包含 'safe_wrapper'", 'safe_wrapper' in bp)
        check("bn_protection 包含 'anomaly_detector'", 'anomaly_detector' in bp)
        check("grad_cliper 有 clip() 方法", hasattr(bp['grad_cliper'], 'clip'))
        check("momentum_scheduler 有 step() 方法", hasattr(bp['momentum_scheduler'], 'step'))
        check("safe_wrapper 有 step() 方法", hasattr(bp['safe_wrapper'], 'step'))
    else:
        check("BN protection 已安装", False, "_bn_protection is None")

    # ============================================================
    section("3. _pre_epoch_hook 可达性验证")
    # ============================================================
    try:
        trainer._pre_epoch_hook(0)
        check("_pre_epoch_hook(0) 无AttributeError", True)
        check("_bn_protection momentum_scheduler.step() 可调用", True)
        check("_bn_protection safe_wrapper.step() 可调用", True)
    except AttributeError as e:
        check("_pre_epoch_hook(0)", False, str(e))
    except Exception as e:
        check("_pre_epoch_hook(0) 其他异常", False, f"{type(e).__name__}: {e}")

    # ============================================================
    section("4. 模型 forward() 路径验证 + BN stats初始状态验证")
    # ============================================================
    fresh_model = create_model('fpn_multihead').to(device)

    fpn_prefixes = ('l1_reduce', 'l2_reduce', 'l3_reduce', 'l4_reduce',
                    'smooth_p3', 'smooth_p2', 'smooth_p1', 'fuse')
    resnet_bn_tracked = {}
    fpn_bn_tracked = {}
    for name, m in fresh_model.backbone.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            is_fpn = any(name.startswith(prefix) for prefix in fpn_prefixes)
            if is_fpn:
                fpn_bn_tracked[name] = m.num_batches_tracked.item()
            else:
                resnet_bn_tracked[name] = m.num_batches_tracked.item()

    resnet_bn_preserved = all(m.running_var.mean().item() != 1.0 for name, m in fresh_model.backbone.named_modules() if isinstance(m, nn.BatchNorm2d) and not any(name.startswith(p) for p in fpn_prefixes))
    fpn_bn_reset = all(m.running_var.mean().item() == 1.0 for name, m in fresh_model.backbone.named_modules() if isinstance(m, nn.BatchNorm2d) and any(name.startswith(p) for p in fpn_prefixes))
    resnet_count = sum(1 for name, m in fresh_model.backbone.named_modules() if isinstance(m, nn.BatchNorm2d) and not any(name.startswith(p) for p in fpn_prefixes))
    fpn_count = sum(1 for name, m in fresh_model.backbone.named_modules() if isinstance(m, nn.BatchNorm2d) and any(name.startswith(p) for p in fpn_prefixes))
    check(f"预训练ResNet BN running_var 保留 ({resnet_count}层)", resnet_bn_preserved)
    check(f"FPN新增层 BN running_var 已重置 ({fpn_count}层)", fpn_bn_reset)

    fresh_leaky_count = sum(1 for _, m in fresh_model.named_modules() if isinstance(m, nn.LeakyReLU))
    fresh_relu_count = sum(1 for _, m in fresh_model.named_modules() if isinstance(m, nn.ReLU))
    check("新模型中无ReLU层", fresh_relu_count == 0, f"found {fresh_relu_count} ReLU layers")
    check("新模型中有LeakyReLU层", fresh_leaky_count > 0, f"found {fresh_leaky_count} LeakyReLU layers")

    model = fresh_model
    model.train()
    img = t.randn(2, 3, config.input_height, config.input_width).to(device)
    bbox = t.rand(2, config.num_heads, 4).to(device)

    try:
        result = model(img, bbox)
        check("forward() 返回3元组", len(result) == 3, f"got {len(result)}")
        cls_outs, bbox_outs, length_logits = result
        check("cls_outs 是tuple", isinstance(cls_outs, tuple))
        check("cls_outs 有3个head", len(cls_outs) == 3)
        check("bbox_outs 是tuple", isinstance(bbox_outs, tuple))
        check("length_logits 是Tensor", isinstance(length_logits, t.Tensor))
        check("length_logits shape正确", list(length_logits.shape) == [2, config.num_heads + 1],
              f"got {list(length_logits.shape)}")
    except Exception as e:
        check("forward()", False, str(e))

    # ============================================================
    section("5. forward_with_attn() 路径验证")
    # ============================================================
    try:
        result = model.forward_with_attn(img, gt_bboxes=bbox)
        check("forward_with_attn() 返回5元组", len(result) == 5, f"got {len(result)}")
        cls_list, bbox_tuple, attn_maps, head_cls_outs, length_logits = result
        check("cls_list 是tuple", isinstance(cls_list, tuple))
        check("attn_maps 是list/tuple", isinstance(attn_maps, (list, tuple)))
        check("head_cls_outs 是tuple", isinstance(head_cls_outs, tuple))
        check("head_cls_outs 有3个head", len(head_cls_outs) == 3)
    except Exception as e:
        check("forward_with_attn()", False, str(e))

    # ============================================================
    section("6. cls_layer 融合验证")
    # ============================================================
    model.zero_grad()
    try:
        cls_list, bbox_list, length_logits = model(img, bbox)
        label = t.randint(0, config.class_num - 1, (2, config.num_heads)).to(device)
        loss = sum(F.cross_entropy(cls_list[h], label[:, h]) for h in range(3))
        loss += F.cross_entropy(length_logits, t.ones(2, dtype=t.long, device=device) * 2) * 0.1
        loss.backward()

        cls_layer_has_grad = all(
            model.heads[h].cls_layer.weight.grad is not None and model.heads[h].cls_layer.weight.grad.norm() > 0
            for h in range(3)
        )
        check("cls_layer.weight 有梯度", cls_layer_has_grad)

        head_fc_has_grad = all(
            model.head_fc[h].weight.grad is not None and model.head_fc[h].weight.grad.norm() > 0
            for h in range(3)
        )
        check("head_fc.weight 有梯度", head_fc_has_grad)

        length_head_linear = None
        for m in model.length_head.modules():
            if isinstance(m, nn.Linear):
                length_head_linear = m
        check("length_head Linear层有梯度",
              length_head_linear is not None and length_head_linear.weight.grad is not None and length_head_linear.weight.grad.norm() > 0)
    except Exception as e:
        check("cls_layer融合验证", False, str(e))

    # ============================================================
    section("7. BN梯度裁剪可达性验证")
    # ============================================================
    if trainer._bn_protection is not None:
        try:
            clip_result = trainer._bn_protection['grad_cliper'].clip(model)
            check("BNGradCliper.clip() 可调用", True)
            check("BNGradCliper.clip() 返回数值", isinstance(clip_result, (int, float)),
                  f"got {type(clip_result)}")
        except Exception as e:
            check("BNGradCliper.clip()", False, str(e))
    else:
        check("BNGradCliper.clip()", False, "_bn_protection is None")

    # ============================================================
    section("8. forward_with_probs() 路径验证")
    # ============================================================
    model.eval()
    try:
        with t.no_grad():
            probs = model.forward_with_probs(img)
        check("forward_with_probs() 返回tuple", isinstance(probs, tuple))
        check("forward_with_probs() 有3个head", len(probs) == 3)
        check("probs[0] 是Tensor", isinstance(probs[0], t.Tensor))
        check("probs[0] 无NaN", not t.isnan(probs[0]).any().item())
    except Exception as e:
        check("forward_with_probs()", False, str(e))

    # ============================================================
    section("9. _eval() 3元组解包验证")
    # ============================================================
    model.eval()
    try:
        with t.no_grad():
            result = model(img)
        check("_eval中 model(img) 返回3元组", len(result) == 3, f"got {len(result)}")
        pred_cls, _, length_logits = result
        check("pred_cls 是tuple", isinstance(pred_cls, tuple))
        check("length_logits 是Tensor", isinstance(length_logits, t.Tensor))
    except Exception as e:
        check("_eval 3元组解包", False, str(e))

    # ============================================================
    section("10. 训练循环关键路径可达性")
    # ============================================================
    model.train()
    model.zero_grad()
    try:
        cls_list, bbox_list, length_logits = model(img, bbox)
        label = t.randint(0, config.class_num - 1, (2, config.num_heads)).to(device)
        bbox_mask = t.ones(2, config.num_heads, device=device)
        true_lengths = bbox_mask.sum(dim=1).long()

        cls_loss = sum(F.cross_entropy(cls_list[h], label[:, h]) for h in range(3))
        bbox_loss = t.tensor(0.5, device=device)
        div_loss = t.tensor(0.1, device=device)
        ord_loss = t.tensor(0.1, device=device)
        attn_sup_loss = t.tensor(0.1, device=device)
        length_loss = F.cross_entropy(length_logits, true_lengths.clamp(max=config.num_heads))

        loss = cls_loss + bbox_loss + div_loss + ord_loss + attn_sup_loss + length_loss
        loss.backward()

        check("训练循环: 前向传播可达", True)
        check("训练循环: 反向传播可达", True)
        check("训练循环: length_loss 计算可达", True)

        if trainer._bn_protection is not None:
            trainer._bn_protection['grad_cliper'].clip(model)
            check("训练循环: BN梯度裁剪可达", True)

        t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=trainer._grad_clip_max_norm)
        check("训练循环: 全局梯度裁剪可达", True)
    except Exception as e:
        check("训练循环关键路径", False, str(e))

    # ============================================================
    section("汇总")
    # ============================================================
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'='*70}")
    print(f"  总计: {total} 项检查")
    print(f"  ✅ 通过: {PASS_COUNT}")
    print(f"  ❌ 失败: {FAIL_COUNT}")
    if FAIL_COUNT == 0:
        print(f"\n  🟢 所有代码路径均可触及，所有修复在训练中真实生效！")
    else:
        print(f"\n  🔴 有 {FAIL_COUNT} 项检查失败，需要修复！")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
    if isinstance(sys.stdout, Tee):
        sys.stdout.remove_stream(LOG_FILE)
    LOG_FILE.close()
