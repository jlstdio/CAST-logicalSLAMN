#!/bin/bash

# PaliGemma CAST Inference Script
# Usage: ./run_inference.sh <checkpoint_path>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ./checkpoints/best_model"
    exit 1
fi

CHECKPOINT_PATH=$1

echo "Starting PaliGemma CAST inference..."
echo "Using checkpoint: $CHECKPOINT_PATH"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found"
    exit 1
fi

# Create results directories
mkdir -p results/images
mkdir -p results/evaluation

# Run inference
echo "Running inference..."
python inference_paligemma.py --config config.yaml --checkpoint "$CHECKPOINT_PATH"

# Run evaluation if results exist
if [ -f "results/predictions.csv" ]; then
    echo "Running evaluation..."
    python evaluate.py --results results/predictions.csv
    
    echo ""
    echo "Inference and evaluation completed!"
    echo "Results saved to:"
    echo "  - CSV: ./results/predictions.csv"
    echo "  - Images: ./results/images/"
    echo "  - Evaluation: ./results/evaluation/"
else
    echo "Warning: Results CSV not found. Evaluation skipped."
fi