"""
CAST Dataset loader for PaliGemma training.
Loads data from HuggingFace dataset 'catglossop/CAST-dataset'.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from action_tokenizer import ActionTokenizer, ActionNormalizer
import os
import tarfile
import json
import glob


class CASTDataset(Dataset):
    """Dataset class for CAST dataset with PaliGemma preprocessing."""
    
    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        action_tokenizer: Optional[ActionTokenizer] = None,
        action_normalizer: Optional[ActionNormalizer] = None,
        max_text_length: int = 128,
        cache_dir: Optional[str] = None,
        local_data_dir: Optional[str] = None,
        dataset_name: str = "catglossop/CAST-dataset"
    ):
        """
        Initialize CAST dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            image_size: Target image size (width, height)
            action_tokenizer: Tokenizer for converting actions to tokens
            action_normalizer: Normalizer for action preprocessing
            max_text_length: Maximum text sequence length
            cache_dir: Directory to cache dataset
            local_data_dir: Local directory containing dataset files
            dataset_name: Name of the HuggingFace dataset
        """
        self.split = split
        self.image_size = image_size
        self.action_tokenizer = action_tokenizer
        self.action_normalizer = action_normalizer
        self.max_text_length = max_text_length
        
        # Load dataset - try local first, then HuggingFace
        print(f"Loading CAST dataset split: {split}")
        
        if local_data_dir is not None:
            print(f"Loading from local directory: {local_data_dir}")
            try:
                self.dataset = load_local_dataset(local_data_dir, split=split)
                print(f"Successfully loaded {len(self.dataset)} samples from local directory")
            except Exception as e:
                print(f"Failed to load from local directory: {e}")
                print("Falling back to HuggingFace dataset...")
                local_data_dir = None
        
        if local_data_dir is None:
            # Load dataset from HuggingFace with multiple strategies
            try:
                self.dataset = load_dataset(
                    dataset_name, 
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
            except ValueError as e:
                if "WebDataset format" in str(e):
                    print(f"WebDataset format error: {e}")
                    print("Attempting to load with different data format...")
                    try:
                        self.dataset = load_dataset(
                            dataset_name, 
                            split=split,
                            cache_dir=cache_dir,
                            streaming=False
                        )
                    except Exception as e2:
                        print(f"Failed to load dataset: {e2}")
                        print("Trying alternative loading method...")
                        try:
                            self.dataset = load_dataset(dataset_name, cache_dir=cache_dir, verification_mode="no_checks")[split]
                        except Exception as e3:
                            print(f"All loading methods failed: {e3}")
                            print("Creating dummy dataset for testing...")
                            # Create dummy dataset as fallback
                            import random
                            dummy_data = []
                            for i in range(100):
                                dummy_data.append({
                                    'image': Image.new('RGB', image_size, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))),
                                    'text': f"Navigate to waypoint {i}",
                                    'actions': [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(8)]
                                })
                            
                            class DummyDataset:
                                def __init__(self, data):
                                    self.data = data
                                def __len__(self):
                                    return len(self.data)
                                def __getitem__(self, idx):
                                    return self.data[idx]
                                def __iter__(self):
                                    return iter(self.data)
                            
                            self.dataset = DummyDataset(dummy_data)
                            print("Using dummy dataset. Replace with real data for actual training.")
                else:
                    raise e
            
            print(f"Loaded {len(self.dataset)} samples")
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - image: Preprocessed image tensor (3, H, W)
                - text: Language instruction string
                - actions: Action sequence as tokens or raw values
                - action_tokens: Tokenized actions as string
        """
        sample = self.dataset[idx]
        
        # Process image
        if isinstance(sample['image'], str):
            # If image is base64 encoded or path
            image = Image.open(sample['image']).convert('RGB')
        else:
            # If image is PIL Image
            image = sample['image'].convert('RGB')
        
        image_tensor = self.image_transform(image)
        
        # Process text instruction
        text = sample['instruction']  # Assuming the field name
        
        # Process actions
        actions = np.array(sample['actions'])  # Shape: (num_steps, 2)
        
        # Normalize actions if normalizer is provided
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)
        
        # Tokenize actions if tokenizer is provided
        action_tokens = []
        if self.action_tokenizer is not None:
            action_tokens = self.action_tokenizer.tokenize(actions)
            action_tokens_text = self.action_tokenizer.tokens_to_text(action_tokens)
        else:
            action_tokens_text = ""
        
        return {
            'image': image_tensor,
            'text': text,
            'actions': torch.FloatTensor(actions),
            'action_tokens': action_tokens_text,
            'raw_actions': torch.FloatTensor(sample['actions'])
        }


class CASTCollator:
    """Collate function for batching CAST data."""
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of samples from CASTDataset
            
        Returns:
            Batched data ready for training
        """
        # Stack images
        images = torch.stack([sample['image'] for sample in batch])
        
        # Prepare text inputs
        texts = [sample['text'] for sample in batch]
        action_tokens = [sample['action_tokens'] for sample in batch]
        
        # Combine instruction and action tokens for PaliGemma
        # Format: "<instruction> <action_tokens>"
        combined_texts = []
        for text, actions in zip(texts, action_tokens):
            if actions:  # If we have tokenized actions
                combined_text = f"{text} {actions}"
            else:
                combined_text = text
            combined_texts.append(combined_text)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            combined_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Stack actions
        actions = torch.stack([sample['actions'] for sample in batch])
        raw_actions = torch.stack([sample['raw_actions'] for sample in batch])
        
        return {
            'pixel_values': images,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'actions': actions,
            'raw_actions': raw_actions,
            'texts': texts,
            'action_tokens': action_tokens
        }


def prepare_cast_data(
    train_split: str = "train",
    val_split: str = "validation", 
    image_size: Tuple[int, int] = (224, 224),
    vocab_size: int = 256,
    cache_dir: Optional[str] = None,
    local_data_dir: Optional[str] = None,
    dataset_name: str = "catglossop/CAST-dataset"
) -> Tuple[CASTDataset, CASTDataset, ActionTokenizer, ActionNormalizer]:
    """
    Prepare CAST datasets with proper tokenization and normalization.
    
    Args:
        train_split: Training split name
        val_split: Validation split name
        image_size: Target image size
        vocab_size: Action tokenizer vocabulary size
        cache_dir: Cache directory for dataset
        local_data_dir: Local directory containing dataset files (alternative to HuggingFace)
        dataset_name: Name of the HuggingFace dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, action_tokenizer, action_normalizer)
    """
    print("Loading dataset to compute action statistics...")
    
    # Check if local data directory is provided
    if local_data_dir is not None:
        print(f"Attempting to load from local directory: {local_data_dir}")
        try:
            # Load from local directory
            temp_dataset = load_local_dataset(local_data_dir, split=train_split)
        except Exception as e:
            print(f"Failed to load from local directory: {e}")
            print("Falling back to HuggingFace dataset...")
            local_data_dir = None
    
    # Try to load dataset from HuggingFace with different strategies
    if local_data_dir is None:
        dataset_loaded = False
        error_messages = []
        
        # Strategy 1: Try loading with different data files parameter
        try:
            print(f"Attempting to load HuggingFace dataset: {dataset_name}")
            temp_dataset = load_dataset(
                dataset_name, 
                split=train_split, 
                cache_dir=cache_dir, 
                trust_remote_code=True,
                data_files=None  # Let the dataset auto-detect files
            )
            dataset_loaded = True
            print("Successfully loaded dataset using auto-detection")
        except Exception as e:
            error_messages.append(f"Auto-detection method: {e}")
            
        # Strategy 2: Try without WebDataset assumptions
        if not dataset_loaded:
            try:
                print("Attempting to load as regular dataset...")
                temp_dataset = load_dataset(
                    dataset_name, 
                    cache_dir=cache_dir, 
                    verification_mode="no_checks",
                    trust_remote_code=True
                )[train_split]
                dataset_loaded = True
                print("Successfully loaded dataset as regular format")
            except Exception as e:
                error_messages.append(f"Regular format method: {e}")
        
        # Strategy 3: Try with streaming disabled
        if not dataset_loaded:
            try:
                print("Attempting to load with streaming disabled...")
                temp_dataset = load_dataset(
                    dataset_name, 
                    split=train_split, 
                    cache_dir=cache_dir, 
                    streaming=False,
                    trust_remote_code=True
                )
                dataset_loaded = True
                print("Successfully loaded dataset with streaming disabled")
            except Exception as e:
                error_messages.append(f"No streaming method: {e}")
                
        # Strategy 4: Try loading specific data files
        if not dataset_loaded:
            try:
                print("Attempting to load specific data files...")
                # Try to get repository info to find available files
                from datasets import get_dataset_config_names
                configs = get_dataset_config_names(dataset_name)
                print(f"Available configs: {configs}")
                
                temp_dataset = load_dataset(
                    dataset_name,
                    name=configs[0] if configs else None,
                    split=train_split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                dataset_loaded = True
                print("Successfully loaded dataset using specific config")
            except Exception as e:
                error_messages.append(f"Specific config method: {e}")
        
        if not dataset_loaded:
            print("\n=== ALL LOADING METHODS FAILED ===")
            for i, msg in enumerate(error_messages, 1):
                print(f"Method {i}: {msg}")
            print("\nPossible solutions:")
            print("1. Download the dataset manually and specify local_data_dir")
            print("2. Check your internet connection and HuggingFace access")
            print("3. Verify the dataset name is correct")
            print("4. Try using a different dataset format")
            
            # Create a minimal dummy dataset for testing
            print("\nCreating dummy dataset for testing purposes...")
            import random
            dummy_data = []
            for i in range(100):  # Create 100 dummy samples
                dummy_data.append({
                    'image': Image.new('RGB', image_size, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))),
                    'text': f"Navigate to waypoint {i}",
                    'actions': [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(8)]
                })
            
            class DummyDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
                def __iter__(self):
                    return iter(self.data)
            
            temp_dataset = DummyDataset(dummy_data)
            print("Using dummy dataset. Replace with real data for actual training.")
    
    # Extract all actions to compute statistics
    all_actions = []
    for sample in temp_dataset:
        actions = np.array(sample['actions'])
        all_actions.append(actions)
    
    all_actions = np.array(all_actions)  # Shape: (N, num_steps, 2)
    print(f"Loaded {len(all_actions)} action sequences with shape {all_actions.shape}")
    
    # Create action normalizer
    action_normalizer = ActionNormalizer()
    action_normalizer.update_stats(all_actions)
    print(f"Action statistics - Mean: {action_normalizer.mean}, Std: {action_normalizer.std}")
    
    # Normalize actions for tokenizer creation
    normalized_actions = action_normalizer.normalize(all_actions)
    
    # Create action tokenizer
    action_tokenizer = ActionTokenizer(
        vocab_size=vocab_size,
        action_bounds=(-3.0, 3.0),  # Reasonable bounds for normalized actions
        num_steps=all_actions.shape[1]
    )
    print(f"Created action tokenizer with vocab_size={vocab_size}, num_steps={all_actions.shape[1]}")
    
    # Create datasets
    train_dataset = CASTDataset(
        split=train_split,
        image_size=image_size,
        action_tokenizer=action_tokenizer,
        action_normalizer=action_normalizer,
        cache_dir=cache_dir,
        local_data_dir=local_data_dir,
        dataset_name=dataset_name
    )
    
    val_dataset = CASTDataset(
        split=val_split,
        image_size=image_size,
        action_tokenizer=action_tokenizer,
        action_normalizer=action_normalizer,
        cache_dir=cache_dir,
        local_data_dir=local_data_dir,
        dataset_name=dataset_name
    )
    
    return train_dataset, val_dataset, action_tokenizer, action_normalizer


def load_local_dataset(data_dir: str, split: str = "train"):
    """
    Load dataset from local directory structure.
    
    Expected structure:
    data_dir/
        train/
            image_0000.jpg
            annotation_0000.json
            ...
        validation/
            ...
        test/
            ...
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Find all annotation files
    annotation_files = glob.glob(os.path.join(split_dir, "annotation_*.json"))
    if not annotation_files:
        # Fallback: look for any JSON files
        annotation_files = glob.glob(os.path.join(split_dir, "*.json"))
    
    if not annotation_files:
        raise ValueError(f"No annotation files found in {split_dir}")
    
    samples = []
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            # Get image path
            if 'image_path' in annotation:
                image_path = annotation['image_path']
                if not os.path.isabs(image_path):
                    image_path = os.path.join(split_dir, image_path)
            else:
                # Infer image path from annotation filename
                base_name = os.path.splitext(os.path.basename(ann_file))[0]
                image_name = base_name.replace('annotation_', 'image_') + '.jpg'
                image_path = os.path.join(split_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            sample = {
                'image': image_path,  # Will be loaded later
                'text': annotation.get('text', 'Navigate'),
                'actions': annotation.get('actions', [[0.0, 0.0]] * 8)
            }
            samples.append(sample)
            
        except Exception as e:
            print(f"Error loading annotation {ann_file}: {e}")
            continue
    
    print(f"Loaded {len(samples)} samples from {split_dir}")
    
    class LocalDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            # Load image if it's still a path
            if isinstance(sample['image'], str):
                try:
                    sample = sample.copy()  # Don't modify original
                    sample['image'] = Image.open(sample['image']).convert('RGB')
                except Exception as e:
                    print(f"Error loading image: {e}")
                    # Create a dummy image
                    sample['image'] = Image.new('RGB', (224, 224), color=(128, 128, 128))
            return sample
        
        def __iter__(self):
            for i in range(len(self)):
                yield self[i]
    
    return LocalDataset(samples)