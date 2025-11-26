#!/bin/bash
# Setup Python 3.10 environment with all dependencies for Loris's requirements
# This includes: AutoGluon, auto-sklearn, TALENT, and local LLMs

set -e

echo "============================================"
echo "Setting up Python 3.10 Environment"
echo "============================================"

# Activate conda environment
source /workspace/miniconda3/bin/activate
conda activate py310

echo "Python version: $(python --version)"
echo ""

# Navigate to project
cd /workspace/thesis/llm_feature_engineering

echo "============================================"
echo "Installing PyTorch with CUDA 12.1"
echo "============================================"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "============================================"
echo "Installing core dependencies"
echo "============================================"
pip install -r requirements.txt

echo ""
echo "============================================"
echo "Installing AutoGluon"
echo "============================================"
pip install autogluon.tabular

echo ""
echo "============================================"
echo "Installing auto-sklearn (Linux only)"
echo "============================================"
pip install auto-sklearn || echo "Warning: auto-sklearn installation failed, continuing..."

echo ""
echo "============================================"
echo "Installing TALENT library"
echo "============================================"
# Use older PyTorch-compatible version or skip
pip install git+https://github.com/LAMDA-Tabular/TALENT.git || echo "Warning: TALENT installation failed, using local CSV datasets"

echo ""
echo "============================================"
echo "Installing project package"
echo "============================================"
pip install -e .

echo ""
echo "============================================"
echo "Environment Setup Complete!"
echo "============================================"
echo ""
echo "Installed packages:"
pip list | grep -E 'torch|transformers|autogluon|auto-sklearn|featuretools|lightgbm'
echo ""
echo "To activate this environment:"
echo "  source /workspace/miniconda3/bin/activate"
echo "  conda activate py310"
echo ""
echo "To run experiments:"
echo "  cd /workspace/thesis/llm_feature_engineering"
echo "  python scripts/run_comparison_table.py --datasets electricity phoneme kc1 --trials 3 --skip-llm"
echo ""
