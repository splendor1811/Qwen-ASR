#!/bin/bash
# Setup script for RunPod/Vast.ai GPU instances
# Usage: bash scripts/setup_cloud.sh

set -euo pipefail

echo "=== Qwen-ASR Vietnamese Finetuning - Cloud Setup ==="
echo ""

# System packages
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget ffmpeg libsndfile1 > /dev/null 2>&1
echo "  Done."

# Python environment
echo "[2/6] Setting up Python environment..."
pip install --upgrade pip > /dev/null 2>&1

# Install project
echo "[3/6] Installing project dependencies..."
cd /workspace
if [ ! -d "Qwen-ASR" ]; then
    echo "  Cloning repo..."
    echo "  NOTE: Set your git repo URL here"
    # git clone <your-repo-url> Qwen-ASR
    echo "  Skipping clone - copy your repo to /workspace/Qwen-ASR"
fi

if [ -d "Qwen-ASR" ]; then
    cd Qwen-ASR
    pip install -e ".[train]" > /dev/null 2>&1
    echo "  Done."
else
    echo "  WARNING: Qwen-ASR directory not found"
fi

# Flash Attention
echo "[4/6] Installing Flash Attention..."
pip install flash-attn --no-build-isolation > /dev/null 2>&1 || echo "  Flash Attention install failed (may need manual install)"

# Environment variables
echo "[5/6] Setting up environment..."
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example - please edit with your tokens"
fi

# Load .env if exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "  Loaded .env"
fi

# Login to HuggingFace if token available
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    echo "  HuggingFace login complete"
fi

# Login to W&B if token available
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
    echo "  W&B login complete"
fi

# Verify GPU
echo "[6/6] Verifying GPU..."
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')" 2>/dev/null || echo "  No GPU detected"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your HF_TOKEN and WANDB_API_KEY"
echo "  2. Download data:    python scripts/download_datasets.py --datasets vivos"
echo "  3. Prepare data:     python scripts/prepare_data.py --datasets vivos --merge"
echo "  4. Start training:   python scripts/train.py --config configs/base.yaml"
echo ""
