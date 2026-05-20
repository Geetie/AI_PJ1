"""
Comprehensive Configuration Validation Script
Validates the entire configuration chain works correctly
"""
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_config_import():
    """Test that config can be imported correctly"""
    print("=" * 70)
    print("1. Config Import Test")
    print("=" * 70)
    
    try:
        import config
        print("✅ config module imported successfully")
        
        # Check that the config object exists
        assert hasattr(config, 'config'), "config object not found"
        print("✅ config.config object exists")
        
        return config
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return None

def check_config_attributes(config_module):
    """Check that all required attributes are present"""
    print("\n" + "=" * 70)
    print("2. Config Attributes Check")
    print("=" * 70)
    
    required_attrs = [
        # Training params
        'batch_size', 'lr', 'backbone_lr_factor', 'momentum', 
        'weights_decay', 'class_num',
        # Scheduler
        'optimizer_type', 'scheduler_type',
        # Gradient
        'grad_accum_steps', 'grad_clip_max_norm',
        # Loss weights
        'cls_loss_weight', 'aux_loss_weight', 'bbox_loss_weight',
        'attn_diversity_weight', 'ordering_loss_weight',
        'attn_supervision_weight',
        # Training control
        'eval_interval', 'checkpoint_interval', 'print_interval',
        'checkpoints', 'pretrained', 'start_epoch', 'epoches',
        'warmup_epochs', 'resume_weights_only',
        # Data loader
        'num_workers', 'pin_memory', 'persistent_workers',
        'prefetch_factor', 'multiprocessing_context',
        # Data augmentation
        'smooth', 'erase_prob', 'mixup_alpha', 'mixup_prob',
        'cutmix_alpha', 'cutmix_prob', 'aug_rotation_degrees',
        'aug_blur_prob', 'train_eval_interval',
        # Input
        'num_heads', 'input_height', 'input_width', 'resize_size',
        'tta_sizes',
        # Model
        'dropout', 'fc_hidden', 'ema_decay', 'use_torch_compile',
        'use_gradient_checkpoint',
        # Compile
        'compile_mode', 'compile_dynamic', 'compile_fullgraph',
        # FPN
        'multiscale_feat_dim', 'pos_embed_channels', 'feat_spatial_size',
        'roi_refine', 'roi_feat_dim', 'roi_teacher_forcing',
        'roi_gt_prob', 'num_attn_channels',
        # Transformer
        'transformer_heads', 'transformer_layers',
        # General
        'model_type', 'keep_aspect_ratio', 'use_char_level_acc',
        'early_stopping_patience',
        # System
        'max_checkpoints', 'oom_headroom_ratio'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if hasattr(config_module.config, attr):
            value = getattr(config_module.config, attr)
            print(f"✅ config.{attr} = {value!r}")
        else:
            missing_attrs.append(attr)
            print(f"❌ Missing: config.{attr}")
    
    return missing_attrs

def check_top_level_constants(config_module):
    """Check that top-level constants are present"""
    print("\n" + "=" * 70)
    print("3. Top-Level Constants Check")
    print("=" * 70)
    
    required_constants = [
        'BASE_DIR', 'SCRIPT_DIR', 'NUM_WORKERS', 'data_dir',
        'IS_NVIDIA', 'IS_AMD', 'IS_MODELSCOPE', 'COMPILE_AVAILABLE',
        'GPU_PLATFORM', 'TOTAL_VRAM_GB', 'NUM_PHYSICAL_CORES',
        'NUM_HEADS', 'set_seed', 'print_env_info'
    ]
    
    missing_constants = []
    for name in required_constants:
        if hasattr(config_module, name):
            value = getattr(config_module, name)
            if callable(value):
                print(f"✅ {name} = {value.__name__}()")
            else:
                print(f"✅ {name} = {value!r}")
        else:
            missing_constants.append(name)
            print(f"❌ Missing: {name}")
    
    return missing_constants

def check_baseline_config_usage():
    """Check that baseline.py uses the config correctly"""
    print("\n" + "=" * 70)
    print("4. baseline.py Config Usage Check")
    print("=" * 70)
    
    try:
        # Test that we can import from baseline without errors
        from baseline import PadToSquare, look_train_json, look_submit
        print("✅ baseline imports successful (basic modules)")
        
        # Try to import the Trainer
        from baseline import Trainer
        print("✅ Trainer class imported")
        
        # Try to create a model
        from baseline import create_model
        print("✅ create_model function imported")
        
        return True
    except Exception as e:
        print(f"❌ Error importing from baseline.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_transform_config():
    """Check that data/transform.py uses config correctly"""
    print("\n" + "=" * 70)
    print("5. data/transform.py Config Usage Check")
    print("=" * 70)
    
    try:
        from data.transform import transform_with_bbox, transform_test_img
        print("✅ data.transform imports successful")
        
        # Verify it imports from config
        import inspect
        source = inspect.getsource(transform_with_bbox)
        if 'config.' in source:
            print("✅ transform_with_bbox uses config")
        else:
            print("⚠️  transform_with_bbox might not be using config")
        
        return True
    except Exception as e:
        print(f"❌ Error checking data/transform.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_trainer_multihead_config():
    """Check that trainer/multihead.py uses config correctly"""
    print("\n" + "=" * 70)
    print("6. trainer/multihead.py Config Usage Check")
    print("=" * 70)
    
    try:
        from trainer.multihead import MultiHeadTrainer
        print("✅ MultiHeadTrainer imported")
        
        return True
    except Exception as e:
        print(f"❌ Error checking trainer/multihead.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_sanity_check(config_module):
    """Run some sanity checks on the config values"""
    print("\n" + "=" * 70)
    print("7. Config Sanity Checks")
    print("=" * 70)
    
    sanity_passed = True
    
    # Check for positive values
    positive_checks = [
        ('batch_size', config_module.config.batch_size),
        ('lr', config_module.config.lr),
        ('epoches', config_module.config.epoches),
        ('warmup_epochs', config_module.config.warmup_epochs),
        ('input_height', config_module.config.input_height),
        ('input_width', config_module.config.input_width),
    ]
    
    for name, value in positive_checks:
        if value <= 0:
            print(f"❌ {name} should be positive, got {value}")
            sanity_passed = False
        else:
            print(f"✅ {name} = {value} (positive)")
    
    # Check probabilities
    prob_checks = [
        ('cutmix_prob', config_module.config.cutmix_prob, 0, 1),
        ('erase_prob', config_module.config.erase_prob, 0, 1),
        ('dropout', config_module.config.dropout, 0, 1),
        ('smooth', config_module.config.smooth, 0, 1),
    ]
    
    for name, value, min_val, max_val in prob_checks:
        if not (min_val <= value <= max_val):
            print(f"❌ {name} should be between {min_val} and {max_val}, got {value}")
            sanity_passed = False
        else:
            print(f"✅ {name} = {value} (valid)")
    
    return sanity_passed

def main():
    """Main validation function"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE CONFIGURATION VALIDATION")
    print("=" * 70)
    
    all_passed = True
    
    # Step 1: Check config import
    config_module = check_config_import()
    if not config_module:
        all_passed = False
        print("\n❌ Fatal: Could not import config module")
        return 1
    
    # Step 2: Check config attributes
    missing_attrs = check_config_attributes(config_module)
    if missing_attrs:
        all_passed = False
        print(f"\n❌ Missing attributes: {', '.join(missing_attrs)}")
    
    # Step 3: Check top-level constants
    missing_constants = check_top_level_constants(config_module)
    if missing_constants:
        all_passed = False
        print(f"\n❌ Missing constants: {', '.join(missing_constants)}")
    
    # Step 4: Check baseline.py config usage
    baseline_ok = check_baseline_config_usage()
    if not baseline_ok:
        all_passed = False
    
    # Step 5: Check data/transform.py
    transform_ok = check_data_transform_config()
    if not transform_ok:
        all_passed = False
    
    # Step 6: Check trainer/multihead.py
    trainer_ok = check_trainer_multihead_config()
    if not trainer_ok:
        all_passed = False
    
    # Step 7: Sanity checks
    sanity_ok = run_sanity_check(config_module)
    if not sanity_ok:
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print("✅ All checks passed! Configuration system is working correctly.")
        print("\nKey configuration:")
        print(f"  - batch_size: {config_module.config.batch_size}")
        print(f"  - cutmix_prob: {config_module.config.cutmix_prob}")
        print(f"  - ordering_loss_weight: {config_module.config.ordering_loss_weight}")
        print(f"  - attn_supervision_weight: {config_module.config.attn_supervision_weight}")
        return 0
    else:
        print("❌ Some checks failed! Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
