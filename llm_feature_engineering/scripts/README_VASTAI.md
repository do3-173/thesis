# Running Experiments on Vast.ai (80GB GPU)

This guide explains how to run the LLM Feature Engineering experiments on [Vast.ai](https://vast.ai/) using **local open-weight models** (no API keys needed).

## SSH Connection

Connect to your Vast.ai instance using SSH with port forwarding for Jupyter/monitoring:

```bash
# Option 1: Using Vast.ai hostname
ssh -p 12329 root@ssh7.vast.ai -L 8080:localhost:8080

# Option 2: Using direct IP
ssh -p 27647 root@149.86.66.214 -L 8080:localhost:8080
```

**Note**: Replace the port number and hostname/IP with your actual instance details from Vast.ai console.

---

## Quick Start (Docker - Recommended for Easy Setup)

### 1. Create a Vast.ai Instance

1. Go to [vast.ai](https://cloud.vast.ai/)
2. Select a GPU instance:
   - **For Qwen-7B**: RTX 4090/3090 (24GB, ~$0.50/hr) ✅ Recommended
   - **For larger models**: A100-80GB (~$2.50/hr) or H100-80GB (~$3.50/hr)
3. Choose image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
4. Set disk space: **150GB minimum** (for model weights + datasets)
5. Launch the instance

### 2. SSH and Setup with Docker

```bash
# SSH into the instance
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080

# Navigate to workspace
cd /workspace

# Clone repository
git clone https://github.com/do3-173/thesis.git
cd thesis/llm_feature_engineering

# Build and start Docker container
docker-compose up -d

# Enter the container
docker exec -it llm-fe-experiments bash
```

### 3. Run Tests Inside Container

```bash
# Inside the container

# Quick test (no LLM, just baseline + featuretools)
python scripts/run_comparison_table.py --datasets electricity --trials 1 --skip-llm

# Rigorous test with Qwen-7B (optimized for 4090 24GB)
python scripts/run_comparison_table.py \
    --datasets electricity phoneme kc1 \
    --trials 3 \
    --llm-provider huggingface \
    --llm-model Qwen/Qwen2.5-7B-Instruct

# Full benchmark (5 datasets × 3 trials)
python scripts/run_comparison_table.py \
    --datasets electricity phoneme kc1 splice vehicle \
    --trials 3 \
    --llm-provider huggingface \
    --llm-model Qwen/Qwen2.5-7B-Instruct
```

---

## Alternative: Manual Setup (Without Docker)

### 1. Create a Vast.ai Instance

1. Go to [vast.ai](https://cloud.vast.ai/)
2. Select a GPU instance:
   - **For Qwen-7B**: RTX 4090/3090 (24GB, ~$0.50/hr) ✅ Recommended
   - **For larger models**: A100-80GB (~$2.50/hr) or H100-80GB (~$3.50/hr)
3. Choose image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
4. Set disk space: **150GB minimum** (for model weights + datasets)
5. Launch the instance

### 2. SSH and Setup

```bash
# SSH into the instance
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080

# Navigate to workspace
cd /workspace

# Clone repository
git clone https://github.com/do3-173/thesis.git
cd thesis/llm_feature_engineering

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install package
pip install -e .
```

### 3. Test Installation

```bash
# Quick test (no LLM, just baseline + featuretools)
python3 scripts/run_comparison_table.py --datasets electricity --trials 1 --skip-llm

# Rigorous test with Qwen-7B (optimized for 4090 24GB)
python3 scripts/run_comparison_table.py \
    --datasets electricity phoneme kc1 \
    --trials 3 \
    --llm-provider huggingface \
    --llm-model Qwen/Qwen2.5-7B-Instruct
```

---

## Alternative: Using Setup Script

### 1. Clone and Build Docker

```bash
# Clone the repository
git clone https://github.com/do3-173/thesis.git
cd thesis/llm_feature_engineering

# Build and start Docker container
docker-compose up -d

# Enter the container
docker exec -it llm-fe-experiments bash
```

### 2. Download TALENT Datasets

```bash
# Inside the container
chmod +x scripts/download_talent_datasets.sh
./scripts/download_talent_datasets.sh
```

Or manually download from [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z) and extract to `/workspace/datasets/TALENT/`.

### 3. Run Experiments

```bash
# Quick test (no LLM, just baseline + featuretools)
python scripts/run_comparison_table.py --datasets electricity --trials 1 --skip-llm

# Full run with Qwen 7B
python scripts/run_comparison_table.py \
    --datasets electricity phoneme kc1 splice vehicle \
    --trials 3 \
    --llm-provider huggingface \
    --llm-model Qwen/Qwen2.5-7B-Instruct
```

---

## Alternative: Manual Setup (Without Docker)

```bash
# Clone the repository
git clone https://github.com/do3-173/thesis.git
cd thesis/llm_feature_engineering

# Run setup script
chmod +x scripts/setup_cloud.sh
./scripts/setup_cloud.sh

# Download TALENT datasets
./scripts/download_talent_datasets.sh
```

---

## Running Experiments

### Quick Test (No LLM)

The TALENT benchmark includes 300+ tabular datasets organized in:
- **basic_benchmark**: 300 datasets (120 binary, 80 multi-class, 100 regression)
- **large_benchmark**: 22 large-scale datasets

### Dataset Structure

Each dataset folder contains:
```
dataset_name/
├── N_train.npy, N_val.npy, N_test.npy  # Numeric features
├── C_train.npy, C_val.npy, C_test.npy  # Categorical features (optional)
├── y_train.npy, y_val.npy, y_test.npy  # Labels
└── info.json  # Task type, feature counts
```

### Downloading Datasets

**Option 1: Automatic** (may fail for large folders)
```bash
./scripts/download_talent_datasets.sh
```

**Option 2: Manual**
1. Go to [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z)
2. Download `basic_benchmark.zip` and/or `large_benchmark.zip`
3. Extract to `datasets/TALENT/`

**Option 3: Clone TALENT repo example datasets**
```bash
# These are just 3 example datasets, not the full benchmark
git clone https://github.com/LAMDA-Tabular/TALENT.git /tmp/talent
cp -r /tmp/talent/example_datasets/* datasets/TALENT/
```

---

## Included Tools

### 1. Featuretools (Traditional AutoML FE)
Automatic feature engineering using Deep Feature Synthesis (DFS).

### 2. Auto-sklearn (Automated ML)
AutoML system that includes automatic feature preprocessing.
- **Requires Linux** - installed automatically in Docker
- Uses ensemble of ML models

### 3. AutoGluon (Amazon AutoML)
State-of-the-art AutoML for tabular data.
- Installed automatically in Docker
- Documentation: [auto.gluon.ai](https://auto.gluon.ai/stable/index.html)

### 4. Local LLM (HuggingFace)
Open-weight models for LLM-based feature engineering.
- No API keys needed
- Supports Qwen, Llama, Mistral, etc.

---

## Recommended Models for 80GB VRAM

| Model | Size | VRAM (fp16) | Quality | Notes |
|-------|------|-------------|---------|-------|
| `Qwen/Qwen2.5-72B-Instruct` | 72B | ~144GB* | Excellent | Auto 4-bit quant |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | ~64GB | Very Good | Fits in 80GB |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14GB | Good | Fast, reliable |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | ~140GB* | Excellent | Requires HF_TOKEN |

*Uses automatic 4-bit quantization via bitsandbytes

---

## Cost Estimation

| Instance Type | Cost/hr | Best For |
|--------------|---------|----------|
| RTX 3090/4090 (24GB) | ~$0.40-0.50 | Qwen-7B |
| A100-40GB | ~$1.50 | Qwen-32B |
| **A100-80GB** | ~$2.50 | Qwen-72B, Llama-70B |
| H100-80GB | ~$3.50 | Best performance |

**Estimated experiment time (5 datasets × 3 trials):**
- Baseline + Featuretools: ~15 minutes
- With LLM-FE (7B model): ~1-2 hours
- With LLM-FE (72B model): ~3-4 hours

---

## Docker Commands Reference

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# Enter container
docker exec -it llm-fe-experiments bash

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Remove everything (including cached models)
docker-compose down -v
```

---

## Experiment Output

Results are saved to `experiments/comparison_table/`:

```
experiments/comparison_table/
├── raw_results_YYYYMMDD_HHMMSS.csv       # All individual results
├── aggregated_results_YYYYMMDD_HHMMSS.csv # Mean±std per method
└── comparison_table_YYYYMMDD_HHMMSS.md    # Formatted table
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Use smaller model
--llm-model Qwen/Qwen2.5-7B-Instruct

# Or skip LLM experiments
--skip-llm
```

### TALENT Dataset Issues
```bash
# Set correct path
export TALENT_DATA_PATH=/workspace/datasets/TALENT

# Verify datasets exist
ls $TALENT_DATA_PATH/basic_benchmark/
```

### Auto-sklearn Issues
Auto-sklearn requires Linux and complex dependencies. It's pre-installed in Docker.
If installation fails, experiments will skip it and show a warning.

### Gated Model Access (Llama)
```bash
# Set HuggingFace token
export HF_TOKEN=hf_your_token_here

# Or add to .env file
echo "HF_TOKEN=hf_your_token_here" > .env
```

---

## Files Overview

```
llm_feature_engineering/
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker compose config
├── scripts/
│   ├── run_comparison_table.py # Main experiment script
│   ├── setup_cloud.sh          # Non-Docker setup
│   ├── download_talent_datasets.sh  # TALENT downloader
│   └── README_VASTAI.md        # This file
├── src/
│   └── llm_feature_engineering/
│       ├── evaluation.py       # LR, MLP, LGBM + MCC metric
│       ├── traditional_fe.py   # Featuretools, Auto-sklearn
│       ├── feature_selection.py # LLM4FS
│       └── llm_interface.py    # HuggingFace local inference
└── datasets/                   # Mount point for TALENT
```
