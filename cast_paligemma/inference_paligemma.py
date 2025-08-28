"""
PaliGemma inference script for CAST dataset.
Loads trained model and runs inference, saving results to CSV with images.
"""

import os
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import uuid

import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import numpy as np
from tqdm import tqdm

from cast_dataset import CASTDataset, CASTCollator
from action_tokenizer import ActionTokenizer, ActionNormalizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaliGemmaCASTInference:
    """Inference class for PaliGemma CAST model."""
    
    def __init__(self, config: dict, checkpoint_path: str):
        """Initialize inference with configuration and checkpoint."""
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(config['hardware']['device'])
        
        # Create output directories
        os.makedirs(config['inference']['output_images_dir'], exist_ok=True)
        os.makedirs(os.path.dirname(config['inference']['output_csv']), exist_ok=True)
        
        # Load model and tokenizers
        self._load_model()
        self._load_tokenizers()
        self._setup_data()
        
    def _load_model(self):
        """Load the trained model and processor."""
        logger.info(f"Loading model from: {self.checkpoint_path}")
        
        self.processor = PaliGemmaProcessor.from_pretrained(self.checkpoint_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16 if self.config['hardware']['mixed_precision'] else torch.float32,
            device_map="auto" if torch.cuda.device_count() > 1 else None
        )
        
        if torch.cuda.device_count() <= 1:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def _load_tokenizers(self):
        """Load action tokenizer and normalizer."""
        tokenizers_path = Path(self.checkpoint_path) / "tokenizers.pt"
        
        if tokenizers_path.exists():
            logger.info("Loading action tokenizers...")
            tokenizers = torch.load(tokenizers_path, map_location='cpu')
            self.action_tokenizer = tokenizers['action_tokenizer']
            self.action_normalizer = tokenizers['action_normalizer']
        else:
            logger.warning("Tokenizers not found. Creating default ones.")
            self.action_tokenizer = ActionTokenizer()
            self.action_normalizer = ActionNormalizer()
    
    def _setup_data(self):
        """Setup test dataset and dataloader."""
        logger.info("Setting up test dataset...")
        
        self.test_dataset = CASTDataset(
            split=self.config['inference']['test_split'],
            image_size=tuple(self.config['model']['image_size']),
            action_tokenizer=self.action_tokenizer,
            action_normalizer=self.action_normalizer,
            cache_dir=self.config['data']['cache_dir']
        )
        
        self.collator = CASTCollator(
            tokenizer=self.processor.tokenizer,
            max_length=self.config['model']['max_text_length']
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['inference']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=self.collator,
            pin_memory=self.config['hardware']['dataloader_pin_memory']
        )
        
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def predict_actions(self, pixel_values, input_texts):
        """
        Predict actions for given inputs.
        
        Args:
            pixel_values: Image tensor (batch_size, 3, H, W)
            input_texts: List of instruction texts
            
        Returns:
            Predicted actions as numpy arrays
        """
        with torch.no_grad():
            # Prepare inputs for generation
            inputs = self.processor(
                text=input_texts,
                images=[Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * len(input_texts),
                return_tensors="pt",
                padding=True
            )
            
            # Replace dummy images with actual pixel values
            inputs['pixel_values'] = pixel_values.to(self.device)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
            
            # Generate action tokens
            with torch.cuda.amp.autocast(enabled=self.config['hardware']['mixed_precision']):
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    pixel_values=inputs['pixel_values'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=32,  # Enough for 16 action tokens
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode generated tokens
            generated_texts = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            # Extract action tokens from generated text
            predicted_actions = []
            for text in generated_texts:
                try:
                    # Extract action part (assuming format: "instruction action_tokens")
                    parts = text.split()
                    if len(parts) >= 16:  # Look for action tokens
                        action_tokens = [int(token) for token in parts[-16:]]  # Last 16 tokens
                        actions = self.action_tokenizer.detokenize(action_tokens)
                        # Denormalize actions
                        actions = self.action_normalizer.denormalize(actions)
                        predicted_actions.append(actions)
                    else:
                        # Fallback: return zeros
                        predicted_actions.append(np.zeros((8, 2)))
                except (ValueError, IndexError):
                    # Fallback: return zeros
                    predicted_actions.append(np.zeros((8, 2)))
            
            return np.array(predicted_actions)
    
    def save_image(self, image_tensor, filename):
        """Save image tensor to file."""
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image = image_tensor * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to PIL Image
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Save image
        image_path = Path(self.config['inference']['output_images_dir']) / filename
        pil_image.save(image_path)
        
        return str(image_path)
    
    def compute_metrics(self, predictions, targets):
        """Compute evaluation metrics."""
        # Mean Squared Error
        mse = np.mean((predictions - targets) ** 2)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - targets))
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Success rate (within threshold)
        threshold = 0.1  # 10cm threshold
        distances = np.linalg.norm(predictions - targets, axis=-1)  # Distance per step
        success_rate = np.mean(distances < threshold)
        
        return {
            'mse': mse,
            'mae': mae, 
            'rmse': rmse,
            'success_rate': success_rate
        }
    
    def run_inference(self):
        """Run inference on test dataset and save results."""
        logger.info("Starting inference...")
        
        results = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Running inference")):
                batch_size = batch['pixel_values'].shape[0]
                
                # Run prediction
                predicted_actions = self.predict_actions(
                    batch['pixel_values'],
                    batch['texts']
                )
                
                # Get ground truth actions (denormalized)
                target_actions = batch['raw_actions'].cpu().numpy()
                
                # Process each sample in batch
                for i in range(batch_size):
                    sample_id = batch_idx * self.config['inference']['batch_size'] + i
                    
                    # Generate unique image filename
                    image_filename = f"sample_{sample_id:06d}_{uuid.uuid4().hex[:8]}.jpg"
                    
                    # Save image
                    image_path = self.save_image(batch['pixel_values'][i], image_filename)
                    
                    # Prepare result record
                    result = {
                        'sample_id': sample_id,
                        'image_filename': image_filename,
                        'image_path': image_path,
                        'instruction': batch['texts'][i],
                        'predicted_actions': predicted_actions[i].flatten().tolist(),
                        'target_actions': target_actions[i].flatten().tolist(),
                    }
                    
                    # Add individual action steps
                    for step in range(8):
                        result[f'pred_x_{step}'] = predicted_actions[i][step, 0]
                        result[f'pred_y_{step}'] = predicted_actions[i][step, 1] 
                        result[f'target_x_{step}'] = target_actions[i][step, 0]
                        result[f'target_y_{step}'] = target_actions[i][step, 1]
                    
                    results.append(result)
                    
                # Collect for metrics
                all_predictions.append(predicted_actions)
                all_targets.append(target_actions)
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute overall metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        
        # Add metrics as additional columns
        for metric, value in metrics.items():
            df[f'overall_{metric}'] = value
            
        # Save results to CSV
        df.to_csv(self.config['inference']['output_csv'], index=False)
        logger.info(f"Results saved to: {self.config['inference']['output_csv']}")
        logger.info(f"Images saved to: {self.config['inference']['output_images_dir']}")
        
        return df, metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint path from command line
    config['inference']['checkpoint_path'] = args.checkpoint
    
    # Create inference instance and run
    inference = PaliGemmaCASTInference(config, args.checkpoint)
    results_df, metrics = inference.run_inference()
    
    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print(f"Total samples processed: {len(results_df)}")
    print(f"Results CSV: {config['inference']['output_csv']}")
    print(f"Images directory: {config['inference']['output_images_dir']}")
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()