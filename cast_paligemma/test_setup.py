"""
Test script to verify PaliGemma CAST setup.
Run this to check if all components work correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        import transformers
        import datasets
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import yaml
        from PIL import Image
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_action_tokenizer():
    """Test action tokenizer functionality."""
    print("\nTesting action tokenizer...")
    
    try:
        from action_tokenizer import ActionTokenizer, ActionNormalizer
        
        # Test tokenizer
        tokenizer = ActionTokenizer(vocab_size=256, num_steps=8)
        test_actions = np.random.randn(8, 2) * 0.5
        
        tokens = tokenizer.tokenize(test_actions)
        recovered_actions = tokenizer.detokenize(tokens)
        
        assert len(tokens) == 16, f"Expected 16 tokens, got {len(tokens)}"
        assert recovered_actions.shape == (8, 2), f"Wrong shape: {recovered_actions.shape}"
        
        # Test normalizer
        normalizer = ActionNormalizer()
        test_dataset = np.random.randn(100, 8, 2)
        normalizer.update_stats(test_dataset)
        
        normalized = normalizer.normalize(test_dataset)
        denormalized = normalizer.denormalize(normalized)
        
        assert np.allclose(test_dataset, denormalized, atol=1e-6), "Normalization roundtrip failed"
        
        print("✓ Action tokenizer tests passed")
        return True
    except Exception as e:
        print(f"✗ Action tokenizer test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = [
            'model', 'action', 'data', 'training', 
            'output', 'inference', 'hardware'
        ]
        
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"
        
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_loading():
    """Test model loading (without actual download)."""
    print("\nTesting model imports...")
    
    try:
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        print("✓ PaliGemma imports successful")
        return True
    except Exception as e:
        print(f"✗ Model import test failed: {e}")
        return False

def test_dataset_structure():
    """Test dataset loader structure."""
    print("\nTesting dataset structure...")
    
    try:
        from cast_dataset import CASTDataset, CASTCollator
        print("✓ Dataset classes imported successfully")
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("PaliGemma CAST Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_action_tokenizer,
        test_config,
        test_model_loading,
        test_dataset_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Run training: ./run_training.sh")
        print("2. Run inference: ./run_inference.sh ./checkpoints/best_model")
        return True
    else:
        print("✗ Some tests failed. Please check the setup.")
        print("\nTroubleshooting:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check config.yaml exists and is valid")
        print("3. Ensure CUDA is available if using GPU")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)