# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the CAST (Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models) project, which includes:
- CAST-VLA model implementation for robotics navigation 
- Multiple baseline approaches: LeLaN, ConVoi, and PIA Planning
- Training and deployment infrastructure for vision-language-action models

## Key Architecture

### Main Components
- **cast-vla/**: Submodule containing the main CAST-VLA model implementation
- **baselines/**: Contains three baseline navigation approaches:
  - **lelan/**: Language-conditioned navigation from in-the-wild video
  - **convoi/**: GPT-based conversational navigation system  
  - **pia_planning/**: PIA planning baseline
- **deployment/**: Production deployment scripts and configuration
- **configs/**: Shared configuration files for camera and robot settings

### Baseline Systems

#### LeLaN (Learning Language-conditioned Navigation)
- Located in `baselines/lelan/`
- Training code in `train/` subdirectory with PyTorch implementation
- Deployment code for real robot navigation
- Uses conda environment `lelan` or `nomad_train`
- Key models: ViNT, NoMaD, GNM with language conditioning

#### ConVoi (Conversational Navigation)
- Located in `baselines/convoi/` 
- GPT-based system using OpenAI API
- ROS2 integration for real-time robot control
- Requires OPENAI_API_KEY and ORGANIZATION_ID environment variables

#### PIA Planning
- Located in `baselines/pia_planning/`
- Planning-based navigation approach

### Deployment Architecture
- ROS2-based robot control system
- Multi-pane tmux sessions for concurrent processes
- Camera, joystick, and lidar integration
- PD controller for velocity commands

## Environment Setup

### Training Environment (LeLaN)
```bash
# Create conda environment
conda env create -f baselines/lelan/train/train_environment.yml
conda activate nomad_train

# Install packages
pip install -e baselines/lelan/train/
```

### Deployment Environment  
- Conda environment: `hi_learn`
- ROS2 workspace: `~/create_ws/install/setup.bash`
- Hardware: NVIDIA Jetson Orin, cameras, lidar, joystick

## Common Development Commands

### Training Models
```bash
# Train LeLaN model
cd baselines/lelan/train/
python train.py -c ./config/lelan.yaml

# Train with collision avoidance
python train.py -c ./config/lelan_col_pretrain.yaml
python train.py -c ./config/lelan_col.yaml
```

### Robot Deployment
```bash
# Launch robot systems
deployment/launch/launch_robot.sh

# Navigate with VLA
deployment/navigate_vla.sh -p "target object description"

# Navigate with LeLaN  
cd baselines/lelan/deployment/src/
python navigate_lelan.py -p "prompt" --model vint --dir <topomap_dir>
```

### Baseline Systems
```bash
# ConVoi navigation
cd baselines/convoi/
./launch_convoi.sh

# PIA Planning
cd baselines/pia_planning/
./launch_pia_planning.sh
```

## Code Organization Patterns

### Configuration Management
- YAML files for model, training, and deployment configs
- Separate configs for different camera types and robot platforms
- Environment-specific settings in conda .yml files

### ROS2 Integration
- Node-based architecture for real-time processing
- Topic-based communication between navigation components
- Standard ROS message types (geometry_msgs, sensor_msgs, etc.)

### Model Architecture
- PyTorch-based implementations with modular design
- Transformer architectures (ViT) for visual processing
- Language conditioning via CLIP embeddings
- Diffusion-based action prediction

## Hardware Requirements
- NVIDIA Jetson Orin (deployment)
- Wide-angle RGB cameras (ELP fisheye, Ricoh Theta S, Intel D435i)  
- LiDAR sensor (RPLiDAR S1)
- Robot platform with ROS2 /cmd_vel control
- Joystick controller

## Key Dependencies
- PyTorch, torchvision
- ROS2 (sensor_msgs, geometry_msgs, nav_msgs)
- OpenCV, PIL
- wandb (training logging)
- diffusers (diffusion models)
- CLIP (language-vision processing)
- transformers
- OpenAI API (ConVoi baseline)