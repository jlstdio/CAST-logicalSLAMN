#!/bin/bash

# PaliGemma CAST Training Script
# Usage: ./run_training.sh [--wandb]

set -e

echo "Starting PaliGemma CAST training..."

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found"
    exit 1
fi

# Check if requirements are installed
python -c "import torch, transformers, datasets" 2>/dev/null || {
    echo "Installing requirements..."
    pip install -r requirements.txt
}

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p results/images

# Run training
if [ "$1" = "--wandb" ]; then
    echo "Training with Weights & Biases logging..."
    python train_paligemma.py --config config.yaml --wandb
else
    echo "Training without W&B logging..."
    python train_paligemma.py --config config.yaml
fi

echo "Training completed!"
echo "Check ./checkpoints/ for saved models"
echo "Check ./logs/ for training logs"