#!/bin/bash

# Quick start script for CAST PaliGemma training
# This script helps set up the environment and dataset

echo "=== CAST PaliGemma Quick Start ==="

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if they don't exist
if [ ! -f "requirements_installed.flag" ]; then
    echo "Installing requirements..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        touch requirements_installed.flag
    else
        echo "Installing basic requirements..."
        pip install torch torchvision transformers datasets pillow numpy pyyaml tqdm
        touch requirements_installed.flag
    fi
fi

# Set up dataset
echo "Setting up dataset..."
python setup_dataset.py --dummy --num-samples 200

echo ""
echo "=== Setup Complete ==="
echo "You can now run training with:"
echo "  source venv/bin/activate"
echo "  python train_paligemma.py"
echo ""
echo "Note: This setup uses dummy data for testing."
echo "For actual training, download the real CAST dataset and update config.yaml"
