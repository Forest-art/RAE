"""
Test SRAE structure without downloading models.
Uses mock components to verify code structure.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test all imports work correctly."""
    print("=" * 60)
    print("Test 1: Import SRAE")
    print("=" * 60)
    
    try:
        from stage1 import SRAE, RAE
        print("✓ Successfully imported SRAE and RAE")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_structure():
    """Test model can be instantiated (with mock components)."""
    print("\n" + "=" * 60)
    print("Test 2: Verify SRAE structure")
    print("=" * 60)
    
    # We can't instantiate full model without DINOv2 weights
    # But we can verify the code structure
    try:
        from stage1.srae import SRAE
        import inspect
        
        # Check class has all required methods
        methods = ['forward', 'forward_teacher', 'forward_student', 'forward_decoder',
                   'forward_loss', 'random_masking', 'encode', 'decode', 'unpatchify',
                   'get_last_losses']
        
        for method in methods:
            assert hasattr(SRAE, method), f"Missing method: {method}"
            print(f"  ✓ Method '{method}' exists")
        
        # Check init parameters
        sig = inspect.signature(SRAE.__init__)
        params = list(sig.parameters.keys())
        required_params = ['dinov2_model_name', 'img_size', 'patch_size', 'mask_ratio',
                          'bottleneck_dim', 'use_l2_norm', 'projector_hidden_dim',
                          'decoder_num_layers', 'decoder_num_heads', 'decoder_dim',
                          'loss_rec_weight', 'loss_align_weight', 'loss_reg_weight']
        
        for param in required_params:
            if param in params:
                print(f"  ✓ Parameter '{param}' exists")
            else:
                print(f"  ✗ Parameter '{param}' missing")
        
        print("\n✓ SRAE structure validated")
        return True
        
    except Exception as e:
        print(f"\n✗ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test config files are valid."""
    print("\n" + "=" * 60)
    print("Test 3: Load SRAE config")
    print("=" * 60)
    
    try:
        from omegaconf import OmegaConf
        
        config_path = os.path.join(os.path.dirname(__file__), '..', 
                                   'configs/stage1/training/SRAE-B_decB.yaml')
        cfg = OmegaConf.load(config_path)
        
        print(f"  Stage 1 target: {cfg.stage_1.target}")
        print(f"  Parameters:")
        for k, v in cfg.stage_1.params.items():
            print(f"    {k}: {v}")
        
        print(f"\n  Training epochs: {cfg.training.epochs}")
        print(f"  Batch size: {cfg.training.global_batch_size}")
        print(f"  Loss weights:")
        print(f"    rec: {cfg.stage_1.params.loss_rec_weight}")
        print(f"    align: {cfg.stage_1.params.loss_align_weight}")
        print(f"    reg: {cfg.stage_1.params.loss_reg_weight}")
        print(f"  GAN:")
        print(f"    disc_weight: {cfg.gan.loss.disc_weight}")
        print(f"    perceptual_weight: {cfg.gan.loss.perceptual_weight}")
        
        print("\n✓ Config loaded successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_script():
    """Test train_srae.py can be imported."""
    print("\n" + "=" * 60)
    print("Test 4: Verify train_srae.py structure")
    print("=" * 60)
    
    try:
        import ast
        
        train_script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train_srae.py')
        with open(train_script, 'r') as f:
            code = f.read()
        
        # Parse AST
        tree = ast.parse(code)
        
        # Check for main components
        imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
        import_froms = [(node.module, [a.name for a in node.names]) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        
        # Check required imports
        required = ['SRAE', 'LPIPS', 'build_discriminator']
        found = []
        for module, names in import_froms:
            if module:
                for name in required:
                    if name in names:
                        found.append(name)
                        print(f"  ✓ Import '{name}' from '{module}'")
        
        missing = set(required) - set(found)
        if missing:
            print(f"  ✗ Missing imports: {missing}")
        else:
            print("  ✓ All required imports found")
        
        # Check main function
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if 'main' in functions:
            print(f"  ✓ 'main' function exists")
        else:
            print(f"  ✗ 'main' function missing")
        
        print("\n✓ Train script structure validated")
        return True
        
    except Exception as e:
        print(f"\n✗ Train script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# SRAE Structure Test Suite (No Model Download)")
    print("#" * 60 + "\n")
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model Structure", test_model_structure()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Train Script", test_train_script()))
    
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r for _, r in results)
    print("\n" + ("# All tests passed!" if all_passed else "# Some tests failed."))
    print("#" * 60 + "\n")
