#!/usr/bin/env python3
"""
AMD ROCm / NVIDIA CUDA Environment Test Script
Verifies that the code properly detects and configures for both GPU platforms.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as t
from config import config, GPU_PLATFORM, TOTAL_VRAM_GB, NUM_PHYSICAL_CORES, print_env_info


def test_gpu_detection():
    """Test GPU platform detection"""
    print("\n=== GPU Detection Test ===")
    print(f"CUDA available: {t.cuda.is_available()}")
    print(f"GPU Platform detected: {GPU_PLATFORM.upper()}")
    
    if t.cuda.is_available():
        try:
            gpu_name = t.cuda.get_device_name(0)
            props = t.cuda.get_device_properties(0)
            vram = getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / (1024**3)
            print(f"GPU Name: {gpu_name}")
            print(f"GPU VRAM: {vram:.1f} GB")
            print(f"Detected VRAM: {TOTAL_VRAM_GB:.1f} GB")
            assert abs(vram - TOTAL_VRAM_GB) < 1.0, "VRAM detection mismatch!"
            print("✅ VRAM detection correct")
        except Exception as e:
            print(f"❌ Error getting GPU info: {e}")
            return False
    else:
        print("⚠️ No GPU detected, running on CPU")
    
    return True


def test_config_parameters():
    """Test configuration parameters"""
    print("\n=== Configuration Test ===")
    print(f"Batch Size: {config.batch_size}")
    print(f"Gradient Accumulation Steps: {config.grad_accum_steps}")
    print(f"Equivalent Batch Size: {config.batch_size * config.grad_accum_steps}")
    print(f"Data Loader Workers: {config.num_workers}")
    print(f"Use Torch Compile: {config.use_torch_compile}")
    print(f"Physical CPU Cores: {NUM_PHYSICAL_CORES}")
    
    assert config.batch_size > 0, "Batch size must be positive"
    assert config.grad_accum_steps >= 1, "Gradient accumulation steps must be >= 1"
    assert config.num_workers >= 0, "Workers must be non-negative"
    print("✅ Configuration parameters valid")
    return True


def test_device_creation():
    """Test device creation"""
    print("\n=== Device Creation Test ===")
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        test_tensor = t.randn(100, 100, device=device)
        print(f"Test tensor created on {device}: shape={test_tensor.shape}, dtype={test_tensor.dtype}")
        
        if t.cuda.is_available():
            test_tensor = test_tensor.cuda()
            print("✅ Tensor successfully moved to GPU")
    except Exception as e:
        print(f"❌ Error creating tensor: {e}")
        return False
    
    print("✅ Device creation successful")
    return True


def test_grad_scaler():
    """Test GradScaler creation"""
    print("\n=== GradScaler Test ===")
    try:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda' if t.cuda.is_available() else 'cpu', enabled=t.cuda.is_available())
        print(f"GradScaler created for device: {'cuda' if t.cuda.is_available() else 'cpu'}")
        print("✅ GradScaler creation successful")
        return True
    except Exception as e:
        print(f"❌ Error creating GradScaler: {e}")
        return False


def test_autocast():
    """Test autocast context manager"""
    print("\n=== Autocast Test ===")
    try:
        from torch.amp import autocast
        device_type = 'cuda' if t.cuda.is_available() else 'cpu'
        
        with autocast(device_type, enabled=t.cuda.is_available()):
            test_tensor = t.randn(100, 100)
            if t.cuda.is_available():
                test_tensor = test_tensor.cuda()
            result = test_tensor * 2
            print(f"Autocast test completed: result shape={result.shape}")
        
        print("✅ Autocast test successful")
        return True
    except Exception as e:
        print(f"❌ Error with autocast: {e}")
        return False


def test_model_ema():
    """Test ModelEMA creation"""
    print("\n=== ModelEMA Test ===")
    try:
        from trainer.base import ModelEMA
        
        class SimpleModel(t.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = t.nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        if t.cuda.is_available():
            model = model.cuda()
        
        ema = ModelEMA(model, decay=0.999)
        print(f"ModelEMA created with device: {ema.device}")
        
        # Test EMA update
        ema.update(model)
        print("✅ ModelEMA update successful")
        return True
    except Exception as e:
        print(f"❌ Error with ModelEMA: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("AMD ROCm / NVIDIA CUDA Environment Compatibility Test")
    print("=" * 70)
    
    print_env_info()
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Configuration", test_config_parameters),
        ("Device Creation", test_device_creation),
        ("GradScaler", test_grad_scaler),
        ("Autocast", test_autocast),
        ("ModelEMA", test_model_ema),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        print(f"\nRecommended configuration for {GPU_PLATFORM.upper()} with {TOTAL_VRAM_GB:.1f} GB VRAM:")
        print(f"  - Batch Size: {config.batch_size}")
        print(f"  - Gradient Accumulation: {config.grad_accum_steps}")
        print(f"  - Equivalent Batch: {config.batch_size * config.grad_accum_steps}")
        print(f"  - Data Loader Workers: {config.num_workers}")


if __name__ == "__main__":
    main()
