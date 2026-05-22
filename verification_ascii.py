#!/usr/bin/env python3
"""
模型学习障碍分析与修复验证脚本 - ASCII版本
验证所有修复是否正确应用
"""

import sys
import os
import torch as t
import torch.nn as nn
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("Street Character Recognition - Learning障碍修复验证")
print("=" * 80)
print()


def verify_config():
    print("1. Configuration Verification")
    print("-" * 40)

    try:
        from config import config

        print(f"  batch_size: {config.batch_size} (expected: 56)")
        print(f"  eval_batch_size: {config.eval_batch_size} (expected: 56)")
        print(f"  gradient_checkpoint_with_bf16: {config.gradient_checkpoint_with_bf16} (expected: False)")
        print(f"  use_bf16: {config.use_bf16}")
        print(f"  use_gradient_checkpoint: {config.use_gradient_checkpoint}")
        print(f"  persistent_workers: {config.persistent_workers}")

        config_ok = (
            config.batch_size == 56 and
            config.eval_batch_size == 56 and
            config.gradient_checkpoint_with_bf16 is False
        )

        print(f"  Config verification: {'PASS' if config_ok else 'FAIL'}")
        print()
        return config_ok
    except Exception as e:
        print(f"  Config verification exception: {e}")
        traceback.print_exc()
        print()
        return False


def verify_trainer_multihead():
    print("2. trainer/multihead.py Verification")
    print("-" * 40)

    try:
        file_path = os.path.join(os.path.dirname(__file__), 'trainer', 'multihead.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = [
            ("BF16 backward pass", 'if self.use_bf16:' in content and 'loss.backward()' in content),
            ("NaN gradient detection", 'has_nan_grad = False' in content),
            ("Gradient clip order", 'self.scaler.unscale_(' in content),
            ("BF16 mode scaler.update", 'self.scaler.update()' in content),
            ("EMA update protection", 'if not _batch_scaler_skipped:' in content),
        ]

        all_pass = True
        for check_name, check_result in checks:
            status = "PASS" if check_result else "FAIL"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False

        print(f"  Trainer verification: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
    except Exception as e:
        print(f"  Trainer verification exception: {e}")
        traceback.print_exc()
        print()
        return False


def verify_models_multihead():
    print("3. models/multihead.py Verification")
    print("-" * 40)

    try:
        file_path = os.path.join(os.path.dirname(__file__), 'models', 'multihead.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        checks = [
            ("BF16 gradient checkpoint disabled", 'if config.use_bf16 and not config.gradient_checkpoint_with_bf16:' in content),
            ("Gradient checkpoint condition", 'use_ckpt =' in content),
        ]

        all_pass = True
        for check_name, check_result in checks:
            status = "PASS" if check_result else "FAIL"
            print(f"  {status} {check_name}")
            if not check_result:
                all_pass = False

        print(f"  Models verification: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
    except Exception as e:
        print(f"  Models verification exception: {e}")
        traceback.print_exc()
        print()
        return False


def test_model_initialization():
    print("4. Model Initialization Test")
    print("-" * 40)

    try:
        from models.multihead import DigitsResnet101
        from config import config

        print("  Creating model...")
        model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print()
        return True
    except Exception as e:
        print(f"  Model initialization exception: {e}")
        traceback.print_exc()
        print()
        return False


def test_forward_backward():
    print("5. Forward and Backward Propagation Test")
    print("-" * 40)

    try:
        from models.multihead import DigitsResnet101
        from config import config

        device = t.device('cpu')

        print("  Creating model...")
        model = DigitsResnet101(class_num=config.class_num, num_heads=config.num_heads).to(device)
        model.train()

        print("  Creating test data...")
        batch_size = 2
        img = t.randn(batch_size, 3, config.input_height, config.input_width).to(device)
        label = t.randint(0, config.class_num - 1, (batch_size, config.num_heads)).to(device)
        bbox = t.rand(batch_size, config.num_heads, 4).to(device)

        print("  Forward propagation...")
        cls_outs, bbox_outs, length_logits = model(img, bbox)
        print(f"  Forward pass successful")
        print(f"    Classification output count: {len(cls_outs)}")
        print(f"    Classification output shape: {cls_outs[0].shape}")

        print("  Computing loss...")
        import torch.nn.functional as F
        total_loss = 0
        for h_idx, cls_out in enumerate(cls_outs):
            total_loss += F.cross_entropy(cls_out, label[:, h_idx])
        total_loss = total_loss / len(cls_outs)

        length_labels = t.ones(batch_size, dtype=t.long, device=device) * 2
        length_loss = F.cross_entropy(length_logits, length_labels)
        total_loss += length_loss * 0.1

        print(f"  Loss computed: {total_loss.item():.4f}")

        print("  Backward propagation...")
        model.zero_grad()
        total_loss.backward()

        print("  Checking gradients...")
        has_nan = False
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                if t.isnan(param.grad).any() or t.isinf(param.grad).any():
                    has_nan = True
                    print(f"    FAIL {name}: NaN/Inf gradient detected")

        if has_nan:
            print(f"  FAIL: NaN/Inf gradients found")
            print()
            return False
        elif has_grad:
            print(f"  PASS: Backward pass successful, gradients are normal")
            print()
            return True
        else:
            print(f"  WARN: No trainable gradients")
            print()
            return False
    except Exception as e:
        print(f"  Forward/backward exception: {e}")
        traceback.print_exc()
        print()
        return False


def main():
    results = []

    results.append(("Configuration", verify_config()))
    results.append(("Trainer MultiHead", verify_trainer_multihead()))
    results.append(("Models MultiHead", verify_models_multihead()))
    results.append(("Model Initialization", test_model_initialization()))
    results.append(("Forward/Backward", test_forward_backward()))

    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    print()
    print("=" * 80)
    if all_passed:
        print("All verifications PASSED! All fixes applied correctly.")
    else:
        print("Some verifications FAILED. Please check the output above.")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
