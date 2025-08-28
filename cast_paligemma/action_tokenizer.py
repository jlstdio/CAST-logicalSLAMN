"""
Action Tokenizer for PaliGemma CAST model.
Converts xy coordinate deltas to discrete tokens and vice versa.
"""

import numpy as np
import torch
from typing import List, Tuple, Union


class ActionTokenizer:
    """Tokenizes continuous xy actions into discrete tokens for PaliGemma."""
    
    def __init__(
        self, 
        vocab_size: int = 256,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        num_steps: int = 8
    ):
        """
        Initialize action tokenizer.
        
        Args:
            vocab_size: Number of discrete tokens per dimension
            action_bounds: Min and max action values (min, max)
            num_steps: Number of action steps to predict (8 steps * 2D = 16 tokens)
        """
        self.vocab_size = vocab_size
        self.action_bounds = action_bounds
        self.num_steps = num_steps
        self.num_tokens = num_steps * 2  # x, y for each step
        
        # Create bins for quantization
        self.bins = np.linspace(action_bounds[0], action_bounds[1], vocab_size + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
    def tokenize(self, actions: Union[np.ndarray, torch.Tensor]) -> List[int]:
        """
        Convert continuous actions to discrete tokens.
        
        Args:
            actions: Array of shape (num_steps, 2) containing xy deltas
            
        Returns:
            List of token IDs (length = num_steps * 2)
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()
            
        # Ensure correct shape
        if actions.shape != (self.num_steps, 2):
            raise ValueError(f"Expected actions shape ({self.num_steps}, 2), got {actions.shape}")
            
        # Clip actions to bounds
        actions = np.clip(actions, self.action_bounds[0], self.action_bounds[1])
        
        # Quantize to bins
        tokens = []
        for step in range(self.num_steps):
            for dim in range(2):  # x, y
                value = actions[step, dim]
                # Find closest bin
                bin_idx = np.digitize(value, self.bins) - 1
                bin_idx = np.clip(bin_idx, 0, self.vocab_size - 1)
                tokens.append(int(bin_idx))
                
        return tokens
    
    def detokenize(self, tokens: List[int]) -> np.ndarray:
        """
        Convert discrete tokens back to continuous actions.
        
        Args:
            tokens: List of token IDs (length = num_steps * 2)
            
        Returns:
            Array of shape (num_steps, 2) containing xy deltas
        """
        if len(tokens) != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {len(tokens)}")
            
        actions = np.zeros((self.num_steps, 2))
        
        for i, token in enumerate(tokens):
            step = i // 2
            dim = i % 2
            
            # Clamp token to valid range
            token = max(0, min(token, self.vocab_size - 1))
            
            # Convert token back to continuous value
            actions[step, dim] = self.bin_centers[token]
            
        return actions
    
    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to space-separated text string."""
        return " ".join([str(t) for t in tokens])
    
    def text_to_tokens(self, text: str) -> List[int]:
        """Convert space-separated text string to tokens."""
        return [int(t) for t in text.split()]


class ActionNormalizer:
    """Normalizes actions using dataset statistics."""
    
    def __init__(self, action_stats: dict = None):
        """
        Initialize with action statistics.
        
        Args:
            action_stats: Dict with 'mean' and 'std' arrays
        """
        if action_stats is None:
            # Default normalization (will be computed from data)
            self.mean = np.array([0.0, 0.0])
            self.std = np.array([1.0, 1.0])
        else:
            self.mean = np.array(action_stats['mean'])
            self.std = np.array(action_stats['std'])
    
    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions using z-score normalization."""
        return (actions - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Denormalize actions back to original scale."""
        return actions * self.std + self.mean
    
    def update_stats(self, actions_dataset: np.ndarray):
        """Update normalization statistics from dataset."""
        # Flatten to (N, 2) where N is total number of action steps
        actions_flat = actions_dataset.reshape(-1, 2)
        self.mean = np.mean(actions_flat, axis=0)
        self.std = np.std(actions_flat, axis=0)


def create_action_tokenizer_from_data(actions_dataset: np.ndarray, vocab_size: int = 256) -> ActionTokenizer:
    """
    Create action tokenizer with bounds computed from dataset.
    
    Args:
        actions_dataset: Array of shape (N, num_steps, 2) 
        vocab_size: Number of discrete tokens per dimension
        
    Returns:
        ActionTokenizer instance
    """
    # Compute bounds from data
    actions_flat = actions_dataset.reshape(-1, 2)
    min_vals = np.min(actions_flat, axis=0)
    max_vals = np.max(actions_flat, axis=0)
    
    # Add small margin
    margin = 0.1 * (max_vals - min_vals)
    action_bounds = (
        float(np.min(min_vals) - margin[0]),
        float(np.max(max_vals) + margin[1])
    )
    
    num_steps = actions_dataset.shape[1]
    return ActionTokenizer(vocab_size=vocab_size, action_bounds=action_bounds, num_steps=num_steps)