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


class CASTDataset(Dataset):
    """Dataset class for CAST dataset with PaliGemma preprocessing."""
    
    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        action_tokenizer: Optional[ActionTokenizer] = None,
        action_normalizer: Optional[ActionNormalizer] = None,
        max_text_length: int = 128,
        cache_dir: Optional[str] = None
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
        """
        self.split = split
        self.image_size = image_size
        self.action_tokenizer = action_tokenizer
        self.action_normalizer = action_normalizer
        self.max_text_length = max_text_length
        
        # Load dataset from HuggingFace
        print(f"Loading CAST dataset split: {split}")
        self.dataset = load_dataset(
            "catglossop/CAST-dataset", 
            split=split,
            cache_dir=cache_dir
        )
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
    cache_dir: Optional[str] = None
) -> Tuple[CASTDataset, CASTDataset, ActionTokenizer, ActionNormalizer]:
    """
    Prepare CAST datasets with proper tokenization and normalization.
    
    Args:
        train_split: Training split name
        val_split: Validation split name
        image_size: Target image size
        vocab_size: Action tokenizer vocabulary size
        cache_dir: Cache directory for dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, action_tokenizer, action_normalizer)
    """
    print("Loading dataset to compute action statistics...")
    
    # Load training data to compute statistics
    temp_dataset = load_dataset("catglossop/CAST-dataset", split=train_split, cache_dir=cache_dir)
    
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
        cache_dir=cache_dir
    )
    
    val_dataset = CASTDataset(
        split=val_split,
        image_size=image_size,
        action_tokenizer=action_tokenizer,
        action_normalizer=action_normalizer,
        cache_dir=cache_dir
    )
    
    return train_dataset, val_dataset, action_tokenizer, action_normalizer