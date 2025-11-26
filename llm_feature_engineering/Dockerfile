# LLM Feature Engineering - GPU Docker Image
# Includes: TALENT datasets, AutoGluon, Auto-sklearn, Local LLM inference
# Target: NVIDIA GPU with 24GB+ VRAM (80GB recommended for 70B models)

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    unzip \
    swig \
    libgomp1 \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0

# Install HuggingFace for local LLM inference
RUN pip install \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece>=0.1.99 \
    tokenizers>=0.15.0

# Install experiment framework
RUN pip install \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    python-dotenv>=1.0.0 \
    tqdm>=4.65.0

# Install LightGBM
RUN pip install lightgbm>=4.0.0

# Install Featuretools for traditional feature engineering
RUN pip install featuretools>=1.28.0

# Install Auto-sklearn (Linux only) - complex dependencies
RUN pip install pyrfr>=0.9.0 liac-arff>=2.5.0
RUN pip install auto-sklearn>=0.15.0 || echo "Auto-sklearn installation may have issues, continuing..."

# Install AutoGluon
RUN pip install autogluon.tabular>=1.1.0 || echo "AutoGluon installation may have issues, continuing..."

# Install TALENT from GitHub
RUN pip install git+https://github.com/LAMDA-Tabular/TALENT.git@main

# Install gdown for Google Drive downloads
RUN pip install gdown>=4.7.0

# Create directories for datasets and outputs
RUN mkdir -p /workspace/datasets /workspace/outputs /workspace/models

# Copy project files
COPY . /workspace/llm_feature_engineering/

# Set Python path
ENV PYTHONPATH="/workspace/llm_feature_engineering/src:${PYTHONPATH}"

# Environment variables for TALENT
ENV TALENT_DATA_PATH="/workspace/datasets/TALENT"

# Script to download TALENT datasets from Google Drive
COPY scripts/download_talent_datasets.sh /workspace/download_talent_datasets.sh
RUN chmod +x /workspace/download_talent_datasets.sh

# Default command
CMD ["/bin/bash"]
