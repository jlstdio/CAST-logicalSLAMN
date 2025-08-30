#!/usr/bin/env python3
"""
Dataset setup utility for CAST PaliGemma training.

This script helps set up the CAST dataset for training by:
1. Attempting to download from HuggingFace
2. Providing instructions for manual setup
3. Creating dummy data for testing
"""

import os
import sys
import argparse
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_local_data_structure(data_dir: str):
    """Create a local data directory structure for CAST dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (data_path / "train").mkdir(exist_ok=True)
    (data_path / "validation").mkdir(exist_ok=True)
    (data_path / "test").mkdir(exist_ok=True)
    
    print(f"Created local data structure at: {data_path.absolute()}")
    return data_path


def create_dummy_dataset(data_dir: str, num_samples: int = 100):
    """Create a dummy dataset for testing purposes."""
    try:
        from PIL import Image
        import json
        import random
    except ImportError:
        print("PIL and json are required to create dummy dataset")
        return False
    
    data_path = Path(data_dir)
    create_local_data_structure(data_dir)
    
    for split in ["train", "validation", "test"]:
        split_path = data_path / split
        split_samples = num_samples if split == "train" else num_samples // 5
        
        for i in range(split_samples):
            # Create dummy image
            img = Image.new('RGB', (224, 224), 
                          color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            img_path = split_path / f"image_{i:04d}.jpg"
            img.save(img_path)
            
            # Create dummy annotation
            annotation = {
                'image_path': str(img_path),
                'text': f"Navigate to destination {i}",
                'actions': [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(8)]
            }
            
            ann_path = split_path / f"annotation_{i:04d}.json"
            with open(ann_path, 'w') as f:
                json.dump(annotation, f)
    
    print(f"Created dummy dataset with {num_samples} samples at: {data_path.absolute()}")
    return True


def download_huggingface_dataset(dataset_name: str, cache_dir: str):
    """Attempt to download the CAST dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print(f"Attempting to download {dataset_name}...")
        
        # Try different loading strategies
        strategies = [
            {"trust_remote_code": True, "verification_mode": "no_checks"},
            {"streaming": False, "trust_remote_code": True},
            {"data_files": None, "trust_remote_code": True}
        ]
        
        for i, kwargs in enumerate(strategies, 1):
            try:
                print(f"Strategy {i}: Loading with {kwargs}")
                dataset = load_dataset(dataset_name, cache_dir=cache_dir, **kwargs)
                print(f"Successfully downloaded dataset using strategy {i}")
                print(f"Dataset info: {dataset}")
                return True
            except Exception as e:
                print(f"Strategy {i} failed: {e}")
                continue
        
        print("All download strategies failed")
        return False
        
    except ImportError:
        print("datasets library is required to download from HuggingFace")
        return False


def update_config(config_path: str, local_data_dir: str):
    """Update config.yaml to use local data directory."""
    config = load_config(config_path)
    config['data']['local_data_dir'] = local_data_dir
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated {config_path} to use local_data_dir: {local_data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Setup CAST dataset for training")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", default="./data", help="Local data directory")
    parser.add_argument("--dummy", action="store_true", help="Create dummy dataset for testing")
    parser.add_argument("--download", action="store_true", help="Attempt to download from HuggingFace")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of dummy samples")
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
        dataset_name = config['data']['dataset_name']
        cache_dir = config['data']['cache_dir']
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    success = False
    
    # Try to download from HuggingFace first
    if args.download:
        print("=== Attempting HuggingFace Download ===")
        success = download_huggingface_dataset(dataset_name, cache_dir)
    
    # Create dummy dataset if requested or if download failed
    if args.dummy or not success:
        print("=== Creating Dummy Dataset ===")
        success = create_dummy_dataset(args.data_dir, args.num_samples)
        if success:
            update_config(args.config, os.path.abspath(args.data_dir))
    
    if success:
        print("\n=== Setup Complete ===")
        if args.dummy:
            print("WARNING: Using dummy dataset. Replace with real data for actual training.")
        print(f"You can now run training with: python train_paligemma.py")
    else:
        print("\n=== Setup Failed ===")
        print("Manual setup instructions:")
        print("1. Download CAST dataset manually")
        print("2. Extract to a local directory")
        print("3. Update config.yaml local_data_dir parameter")
        print("4. Ensure data follows the expected format")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
