# LLM Feature Engineering - GPU Docker Image
# Includes: TALENT datasets, AutoGluon, Auto-sklearn, Local LLM inference
# Target: NVIDIA GPU with 24GB+ VRAM (80GB recommended for 70B models)

# Option 1: CUDA 12.1 + Ubuntu 22.04 (recommended)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# Option 2: If you need CUDA 12.0 + Ubuntu 20.04, uncomment below:
# FROM nvidia/cuda:12.0.1-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-venv \
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

# Make python3 the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

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

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install -r /workspace/requirements.txt

# Install optional heavy dependencies (skip on failure)
RUN pip install pyrfr>=0.9.0 liac-arff>=2.5.0 || echo "Skipping pyrfr/liac-arff..."
RUN pip install auto-sklearn>=0.15.0 || echo "Skipping auto-sklearn (optional)..."
RUN pip install autogluon.tabular>=1.1.0 || echo "Skipping AutoGluon (optional)..."

# Install TALENT from GitHub (optional)
RUN pip install git+https://github.com/LAMDA-Tabular/TALENT.git@main || echo "Skipping TALENT (optional)..."

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
