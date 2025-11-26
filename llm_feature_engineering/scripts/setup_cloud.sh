#!/bin/bash
# Setup script for running experiments on vast.ai or any cloud GPU (80GB+ VRAM)
# This script installs all dependencies and prepares the environment for LOCAL LLM inference
# Includes: TALENT datasets, AutoGluon, Auto-sklearn

set -e

echo "=========================================="
echo "LLM Feature Engineering - GPU Cloud Setup"
echo "=========================================="
echo "Target: Local open-weight models (no API keys needed)"
echo "Includes: TALENT, AutoGluon, Auto-sklearn, Featuretools"
echo ""

# Update system
echo "Updating system..."
apt-get update -qq

# Install system dependencies (including swig for auto-sklearn)
echo "Installing system dependencies..."
apt-get install -y -qq build-essential python3-dev python3-pip python3-venv git curl wget unzip swig libgomp1

# Check if NVIDIA driver is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "WARNING: nvidia-smi not found. Installing NVIDIA drivers and CUDA..."
    echo "This may take 10-15 minutes..."
    echo ""
    
    # Install NVIDIA drivers and CUDA toolkit
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt-get update -qq
    
    # Install CUDA toolkit (includes drivers)
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-1
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    
    echo "CUDA installation complete. You may need to reboot for drivers to take effect."
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi || echo "WARNING: No NVIDIA GPU detected! GPU may require reboot."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support - use cu121 for CUDA 12.x)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch CUDA check failed"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || true

# Install core dependencies
echo "Installing core dependencies..."
pip install -q \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    python-dotenv>=1.0.0 \
    tqdm>=4.65.0

# Install HuggingFace for local LLMs (PRIMARY)
echo "Installing HuggingFace Transformers for local inference..."
pip install -q \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece>=0.1.99 \
    tokenizers>=0.15.0

# Install downstream models
echo "Installing LightGBM..."
pip install -q lightgbm>=4.0.0

# Install Featuretools for traditional FE
echo "Installing Featuretools..."
pip install -q featuretools>=1.28.0

# Install Auto-sklearn (Linux only)
echo "Installing Auto-sklearn..."
pip install -q pyrfr>=0.9.0 liac-arff>=2.5.0
pip install -q auto-sklearn>=0.15.0 || echo "Auto-sklearn installation failed (will use fallback)"

# Install AutoGluon
echo "Installing AutoGluon..."
pip install -q autogluon.tabular>=1.1.0 || echo "AutoGluon installation failed (optional)"

# Install TALENT from GitHub
echo "Installing TALENT benchmark framework..."
pip install -q git+https://github.com/LAMDA-Tabular/TALENT.git@main || echo "TALENT installation failed (will use local copy)"

# Install gdown for Google Drive downloads
pip install -q gdown>=4.7.0

# Create HuggingFace token file (optional, for gated models)
echo "Creating .env template..."
cat > .env.template << 'EOF'
# HuggingFace Token (only needed for gated models like Llama)
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN=your-huggingface-token-here

# TALENT dataset path
TALENT_DATA_PATH=/workspace/datasets/TALENT

# Note: OpenAI/Anthropic keys are NOT required for local models
EOF

# Pre-download recommended models
echo ""
echo "Pre-downloading recommended models..."
echo "(This may take a while depending on internet speed)"

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

models_to_download = [
    'Qwen/Qwen2.5-7B-Instruct',  # 7B - good for most GPUs
]

for model_name in models_to_download:
    print(f'Downloading {model_name}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f'  ✓ Tokenizer downloaded')
        # Just download, don't load to GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cpu'
        )
        print(f'  ✓ Model downloaded')
        del model
    except Exception as e:
        print(f'  ✗ Failed: {e}')
print('Done!')
"

echo ""
echo "=========================================="
echo "TALENT DATASETS"
echo "=========================================="
echo ""
echo "TALENT datasets need to be downloaded from Google Drive:"
echo "  https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z"
echo ""
echo "Option 1: Run the download script:"
echo "  chmod +x scripts/download_talent_datasets.sh"
echo "  ./scripts/download_talent_datasets.sh"
echo ""
echo "Option 2: Manual download and extract to:"
echo "  /workspace/datasets/TALENT/"
echo ""

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Quick test (baseline + featuretools only, no LLM):"
echo "  python scripts/run_comparison_table.py --datasets electricity --trials 1 --skip-llm"
echo ""
echo "Full run with local Qwen 7B model:"
echo "  python scripts/run_comparison_table.py --datasets electricity phoneme kc1 --trials 3"
echo ""
echo "For 80GB GPU - use larger model:"
echo "  python scripts/run_comparison_table.py --llm-model Qwen/Qwen2.5-72B-Instruct --datasets electricity phoneme kc1 --trials 3"
echo ""
echo "Available models for 80GB VRAM:"
echo "  - Qwen/Qwen2.5-72B-Instruct  (best quality)"
echo "  - Qwen/Qwen2.5-32B-Instruct  (great quality, fits in fp16)"
echo "  - Qwen/Qwen2.5-7B-Instruct   (fast, ~14GB in fp16)"
echo "  - meta-llama/Llama-3.1-70B-Instruct (requires HF_TOKEN)"
echo ""
