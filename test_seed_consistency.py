import os
import sys
import numpy as np
import torch as t

os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, data_dir
from data.dataset import DigitsDataset
from torch.utils.data import DataLoader


def make_epoch_generator(base_seed=42, epoch=0):
    generator = t.Generator()
    generator.manual_seed(base_seed + epoch)
    return generator


def test_dataloader_seed_consistency():
    print("=== Testing DataLoader Seed Consistency ===")
    
    config.batch_size = 32
    config.num_workers = 0
    
    train_set = DigitsDataset(mode='train', aug=True, input_size=(64, 64))
    
    results = {}
    
    for epoch in [0, 1, 2]:
        np.random.seed(42 + epoch * 1000)
        t.manual_seed(42 + epoch * 1000)
        
        generator = make_epoch_generator(42 + epoch * 999, epoch=0)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False,
                                  drop_last=True, generator=generator)
        
        first_batch_hashes = []
        for i, (img, label, bbox_target, bbox_mask) in enumerate(train_loader):
            if i == 0:
                batch_hash = t.sum(img).item()
                first_batch_hashes.append(batch_hash)
            if i == 1:
                break
        
        results[epoch] = first_batch_hashes[0]
        print(f"Epoch {epoch}: First batch hash = {results[epoch]:.6f}")
    
    print("\n=== Results ===")
    if results[0] != results[1] and results[1] != results[2]:
        print("✅ SUCCESS: Different epochs have different data order")
        return True
    else:
        print("❌ FAILURE: Some epochs have the same data order")
        print(f"  Epoch 0 hash: {results[0]}")
        print(f"  Epoch 1 hash: {results[1]}")
        print(f"  Epoch 2 hash: {results[2]}")
        return False


def test_augmentation_seed_consistency():
    print("\n=== Testing Augmentation Seed Consistency ===")
    
    config.batch_size = 1
    config.num_workers = 0
    config.aug_blur_prob = 0.5
    
    train_set = DigitsDataset(mode='train', aug=True, input_size=(64, 64))
    
    results = {}
    
    for epoch in [0, 1]:
        t.manual_seed(42 + epoch * 1000)
        np.random.seed(42 + epoch * 1000)
        
        generator = make_epoch_generator(42 + epoch * 999, epoch=0)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=False,
                                  drop_last=False, generator=generator)
        
        for i, (img, label, bbox_target, bbox_mask) in enumerate(train_loader):
            if i == 0:
                img_hash = t.sum(img).item()
                results[epoch] = img_hash
                break
    
    print(f"Epoch 0 augmentation hash: {results[0]:.6f}")
    print(f"Epoch 1 augmentation hash: {results[1]:.6f}")
    
    if results[0] != results[1]:
        print("✅ SUCCESS: Different epochs have different augmentations")
        return True
    else:
        print("❌ FAILURE: Same augmentation for different epochs")
        return False


def test_reproducibility():
    print("\n=== Testing Reproducibility ===")
    
    config.batch_size = 16
    config.num_workers = 0
    
    train_set = DigitsDataset(mode='train', aug=True, input_size=(64, 64))
    
    run1_hashes = []
    run2_hashes = []
    
    for run_idx, hashes_list in enumerate([run1_hashes, run2_hashes]):
        for epoch in [0, 1]:
            t.manual_seed(42 + epoch * 1000)
            np.random.seed(42 + epoch * 1000)
            
            generator = make_epoch_generator(42 + epoch * 999, epoch=0)
            train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=False,
                                      drop_last=True, generator=generator)
            
            for i, (img, label, bbox_target, bbox_mask) in enumerate(train_loader):
                if i == 0:
                    hashes_list.append(t.sum(img).item())
                break
    
    print(f"Run 1 - Epoch 0: {run1_hashes[0]:.6f}")
    print(f"Run 1 - Epoch 1: {run1_hashes[1]:.6f}")
    print(f"Run 2 - Epoch 0: {run2_hashes[0]:.6f}")
    print(f"Run 2 - Epoch 1: {run2_hashes[1]:.6f}")
    
    if run1_hashes[0] == run2_hashes[0] and run1_hashes[1] == run2_hashes[1]:
        print("✅ SUCCESS: Results are reproducible across runs")
        return True
    else:
        print("❌ FAILURE: Results are not reproducible")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Seed Consistency Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_dataloader_seed_consistency()
    all_passed &= test_augmentation_seed_consistency()
    all_passed &= test_reproducibility()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)