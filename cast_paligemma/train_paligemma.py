"""
PaliGemma training script for CAST dataset.
Fine-tunes PaliGemma-3B for vision-language-action tasks.
"""

import os
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import transformers
from transformers import (
    PaliGemmaProcessor, 
    PaliGemmaForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import set_seed
import wandb
from tqdm import tqdm
import numpy as np

from cast_dataset import prepare_cast_data, CASTCollator
from action_tokenizer import ActionTokenizer, ActionNormalizer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaliGemmaCASTTrainer:
    """Trainer class for PaliGemma on CAST dataset."""
    
    def __init__(self, config: dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(config['output']['output_dir'], exist_ok=True)
        os.makedirs(config['output']['logging_dir'], exist_ok=True)
        
        # Initialize model and processor
        self._setup_model()
        
        # Initialize data
        self._setup_data()
        
        # Initialize training components
        self._setup_training()
        
    def _setup_model(self):
        """Initialize model and processor."""
        logger.info(f"Loading model: {self.config['model']['name']}")
        
        # Load processor and model
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.config['model']['name']
        )
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.bfloat16 if self.config['hardware']['mixed_precision'] else torch.float32,
            device_map=self.device
        )
        
        if torch.cuda.device_count() <= 1:
            self.model = self.model.to(self.device)
            
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        
    def _setup_data(self):
        """Initialize datasets and dataloaders."""
        logger.info("Setting up datasets...")
        
        # Prepare datasets with tokenization
        self.train_dataset, self.val_dataset, self.action_tokenizer, self.action_normalizer = prepare_cast_data(
            train_split=self.config['data']['train_split'],
            val_split=self.config['data']['val_split'],
            image_size=tuple(self.config['model']['image_size']),
            vocab_size=self.config['action']['vocab_size'],
            cache_dir=self.config['data']['cache_dir']
        )
        
        # Create collator
        self.collator = CASTCollator(
            tokenizer=self.processor.tokenizer,
            max_length=self.config['model']['max_text_length']
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            collate_fn=self.collator,
            pin_memory=self.config['hardware']['dataloader_pin_memory']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['evaluation']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=self.collator,
            pin_memory=self.config['hardware']['dataloader_pin_memory']
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
    def _setup_training(self):
        """Initialize optimizer, scheduler, and other training components."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=self.config['optimizer']['betas'],
            eps=self.config['optimizer']['eps'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Calculate total training steps
        self.total_steps = (
            len(self.train_loader) * self.config['training']['num_epochs']
        ) // self.config['training']['gradient_accumulation_steps']
        
        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=self.total_steps
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config['hardware']['mixed_precision'] else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Total training steps: {self.total_steps}")
        
    def compute_loss(self, batch):
        """Compute training loss."""
        # Extract inputs
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config['hardware']['mixed_precision']):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=input_ids  # For language modeling objective
            )
            loss = outputs.loss
            
        return loss
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}"
        )
        
        for step, batch in enumerate(progress_bar):
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config['training']['gradient_accumulation_steps']
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            # Update weights
            if (step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    avg_loss = total_loss * self.config['training']['gradient_accumulation_steps'] / (step + 1)
                    
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                    
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/step": self.global_step
                        })
                
                # Evaluation
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    val_loss = self.evaluate()
                    self.model.train()  # Back to training mode
                    
                # Save checkpoint
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * self.config['training']['gradient_accumulation_steps']})
        
        return total_loss / num_batches
    
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "val/loss": avg_loss,
                "val/step": self.global_step
            })
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_model")
            logger.info(f"New best model saved (loss: {avg_loss:.4f})")
        
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['output']['output_dir']) / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model and processor
        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, checkpoint_dir / "training_state.pt")
        
        # Save tokenizers
        torch.save({
            'action_tokenizer': self.action_tokenizer,
            'action_normalizer': self.action_normalizer
        }, checkpoint_dir / "tokenizers.pt")
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            epoch_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")
            
            # End-of-epoch evaluation
            val_loss = self.evaluate()
            self.model.train()
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        logger.info("Training completed!")
        

def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(42)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="paligemma-cast",
            name=config['output']['run_name'],
            config=config
        )
    
    # Create trainer and train
    trainer = PaliGemmaCASTTrainer(config)
    trainer.train()
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()