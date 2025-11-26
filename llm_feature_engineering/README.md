# LLM Feature Engineering Framework

A comprehensive Python framework for comparing feature selection methods, including Large Language Model (LLM) approaches, neural network methods, and traditional statistical techniques. This framework implements multiple feature selection paradigms and evaluates them across benchmark datasets.

## 🚀 Quick Start (Docker - Recommended)

For cloud deployment (vast.ai, runpod, etc.) with GPU:

```bash
# Clone and enter directory
git clone https://github.com/your-username/thesis.git
cd thesis/llm_feature_engineering

# Start Docker container with GPU support
docker-compose up -d

# Enter container
docker exec -it llm-fe-experiments bash

# Download TALENT datasets
./scripts/download_talent_datasets.sh

# Run experiments (no API keys needed!)
python scripts/run_comparison_table.py \
    --datasets electricity phoneme kc1 \
    --trials 3 \
    --llm-provider huggingface \
    --llm-model Qwen/Qwen2.5-7B-Instruct
```

**📖 Full cloud deployment guide: [scripts/README_VASTAI.md](scripts/README_VASTAI.md)**

## Features

- **Local LLM Inference**: Run experiments with open-weight models (Qwen, Llama, Mistral) - no API keys needed
- **Multiple FE Methods**: Baseline, Featuretools, Auto-sklearn, AutoGluon, LLM-FE
- **TALENT Benchmark**: 300+ tabular datasets for reproducible comparisons
- **Downstream Models**: Logistic Regression, MLP, LightGBM
- **Comprehensive Metrics**: Accuracy, ROC-AUC, F1-Score, MCC

## Table of Contents
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Feature Selection Methods](#feature-selection-methods)
- [Running Experiments](#running-experiments)
- [Results Analysis](#results-analysis)
- [Configuration Guide](#configuration-guide)
- [Advanced Usage](#advanced-usage)

## Project Structure

```
llm_feature_engineering/
├── config/                    # Configuration files (Hydra)
│   ├── experiment.yaml        # Main experiment settings
│   ├── benchmark/             # AutoGluon benchmark configs
│   ├── data/                  # Dataset configurations
│   ├── evaluation/            # Model evaluation settings
│   ├── llm/                   # LLM provider configurations
│   └── methods/               # Feature selection method configs
├── experiments/               # Experiment results (auto-generated)
│   └── YYYY-MM-DD_HH-MM-SS/  # Timestamped experiment runs
├── results_analysis/          # Analysis outputs (auto-generated)
├── src/llm_feature_engineering/  # Core framework code
│   ├── __init__.py
│   ├── experiment_runner.py   # Main experiment orchestrator
│   ├── dataset_manager.py     # Dataset loading and management
│   ├── feature_selection.py   # Feature selection implementations
│   ├── llm_interface.py       # LLM method interfaces
│   ├── mlp_feature_selection.py  # MLP-based methods
│   ├── evaluation.py          # Model evaluation logic
│   └── autogluon_benchmark.py # AutoGluon integration
├── examples/                  # Example usage scripts
├── get_results.py             # Results analysis script
├── run_experiments.sh         # Experiment runner script
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

## Quick Start

### 1. Installation
```bash
# Clone the repository and navigate to the project
cd llm_feature_engineering

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Dataset Setup
Download the TALENT benchmark datasets from [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z) and place them in your data directory. The framework expects:

```bash
# Expected directory structure for datasets
../datasets_csv/          # Or modify config/data/talent_datasets.yaml
├── cmc.csv
├── electricity.csv  
├── eye_movements.csv
├── kc1.csv
├── phoneme.csv
├── pol.csv
├── splice.csv
├── vehicle.csv
└── *_metadata.json      # Metadata files for each dataset
```

### 3. Environment Setup
```bash
# Required for LLM methods (text_based, llm4fs)
export ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: OpenAI API key for alternative LLM provider
export OPENAI_API_KEY=your_openai_key_here
```

### 4. Running Experiments

#### Using the Shell Script (Recommended)
```bash
# Quick test experiment (traditional methods only, 3 datasets)
./run_experiments.sh quick

# Complete experiment (all methods, all datasets, multi-trial)
./run_experiments.sh complete

# Run without LLM methods (traditional only)
./run_experiments.sh no-llm

# Custom experiment with specific configuration overrides
./run_experiments.sh talent data.datasets=[cmc,electricity] evaluation.n_trials=3
```

#### Direct Python Usage
```bash
# Run complete experiment
python -m src.llm_feature_engineering.experiment_runner \
    --config-path config --config-name experiment \
    data=talent_datasets methods=complete_methods evaluation=multi_trial

# Custom configuration
python -m src.llm_feature_engineering.experiment_runner \
    --config-path config --config-name experiment \
    data.datasets=[cmc,pol] methods.text_based.enabled=false
```

## Feature Selection Methods

This framework implements **6 feature selection methods** across different paradigms:

### **LLM-Based Methods**
- **`text_based`**: Uses LLM semantic understanding to score features individually
- **`llm4fs`**: Hybrid approach combining LLM reasoning with statistical analysis

### **Neural Network Methods** 
- **`mlp_weights`**: Analyzes MLP first-layer weight magnitudes for feature importance
- **`mlp_permutation`**: Uses permutation importance with trained MLPs

### **Traditional Statistical Methods**
- **`mutual_info`**: Mutual information between features and target
- **`random_forest`**: Tree-based feature importance from Random Forest
- **`univariate`**: Univariate statistical tests (f_classif/f_regression)

## Running Experiments

### Using run_experiments.sh (Recommended)

The shell script provides convenient presets for common experiment configurations:

```bash
# Available experiment types:
./run_experiments.sh quick       # Fast test (traditional methods, 3 datasets)
./run_experiments.sh complete    # Full experiment (all methods, all datasets, 5 trials)
./run_experiments.sh no-llm      # Without LLM methods (traditional + MLP only)
./run_experiments.sh talent      # Custom TALENT configuration
./run_experiments.sh local       # Local datasets configuration
```

#### Script Features:
- **Automatic Environment**: Activates Python environment and sets PYTHONPATH
- **Configuration Overrides**: Supports Hydra parameter overrides
- **Error Handling**: Graceful handling of interrupted experiments
- **Logging**: Comprehensive experiment logging

#### Custom Parameters:
```bash
# Override specific parameters
./run_experiments.sh complete evaluation.n_trials=3 data.datasets=[cmc,pol]

# Disable specific methods
./run_experiments.sh talent methods.text_based.enabled=false methods.llm4fs.enabled=false

# Custom dataset selection
./run_experiments.sh complete data.datasets=[electricity,phoneme,vehicle]
```

### Experiment Presets

| Preset | Methods | Datasets | Trials | Use Case |
|--------|---------|----------|--------|-----------|
| `quick` | Traditional only | 3 datasets | 1 trial | Fast testing |
| `complete` | All methods | All datasets | 5 trials | Full evaluation |
| `no-llm` | Traditional + MLP | All datasets | 3 trials | No API required |
| `talent` | All methods | TALENT datasets | Multi-trial | Standard benchmark |

## Configuration Guide

The framework uses [Hydra](https://hydra.cc/) for configuration management. All configuration files are in the `config/` directory.

### Main Configuration Structure
```
config/
├── experiment.yaml               # Main experiment settings
├── data/                        # Dataset configurations
│   ├── talent_datasets.yaml     # TALENT benchmark datasets
│   └── local_datasets.yaml      # Local/custom datasets
├── methods/                     # Feature selection method configurations
│   ├── complete_methods.yaml    # All methods enabled (default)
│   └── traditional_only.yaml    # Traditional methods only
├── llm/                         # LLM provider settings
│   ├── anthropic.yaml           # Anthropic Claude configuration
│   ├── openai.yaml             # OpenAI GPT configuration  
│   └── disabled.yaml           # Disable LLM methods
├── evaluation/                  # Model evaluation configurations
│   ├── multi_trial.yaml        # Multiple trials (5x)
│   ├── standard.yaml           # Standard evaluation
│   └── fast.yaml               # Quick evaluation
└── benchmark/                   # AutoGluon benchmark settings
    ├── autogluon_standard.yaml  # Standard AutoGluon config
    ├── autogluon_fast.yaml     # Fast AutoGluon config
    └── disabled.yaml           # Disable benchmarks
```

### 1. Method Configuration (`config/methods/complete_methods.yaml`)

This is the most important configuration file for customizing feature selection methods:

```yaml
# LLM-based methods
text_based:
  enabled: true          # Enable/disable this method
  top_k: 10             # Number of top features to select
  temperature: 0.1      # LLM temperature (0.0 = deterministic, 1.0 = creative)
  max_tokens: 300       # Maximum tokens for LLM response

llm4fs:
  enabled: true
  top_k: 10
  temperature: 0.1
  max_tokens: 3000      # Higher limit for hybrid analysis
  sample_size: 200      # Number of data samples to send to LLM

# Traditional statistical methods
traditional:
  mutual_info:
    enabled: true
    top_k: 10
  
  random_forest:
    enabled: true
    top_k: 10
    n_estimators: 100   # Number of trees in Random Forest

# Neural Network (MLP) methods
mlp_weights:
  enabled: true
  top_k: 10
  hidden_layer_sizes: [64, 32]  # MLP architecture
  max_iter: 500                 # Training iterations
  random_state: 42              # For reproducibility

mlp_permutation:
  enabled: true
  top_k: 10
  hidden_layer_sizes: [64, 32]
  max_iter: 500
  n_repeats: 5          # Permutation importance repetitions
  random_state: 42
```

#### **Method-Specific Parameters:**

**LLM Methods (`text_based`, `llm4fs`):**
- `temperature`: Controls randomness (0.0-1.0)
- `max_tokens`: Response length limit
- `sample_size`: Data samples for LLM4FS analysis

**MLP Methods (`mlp_weights`, `mlp_permutation`):**
- `hidden_layer_sizes`: Network architecture as list [layer1, layer2, ...]
- `max_iter`: Maximum training epochs
- `n_repeats`: Only for permutation - number of importance calculations

**Traditional Methods:**
- `n_estimators`: Random Forest tree count
- All methods support `top_k` for feature count

### 2. Dataset Configuration (`config/data/local_datasets.yaml`)

```yaml
source: "local"              # Data source type
dataset_dir: "../../../datasets_csv"  # Path to dataset directory
test_size: 0.2              # Train/test split ratio
datasets: ["cmc", "electricity"]  # Specific datasets or "all"

# Alternative configurations:
# datasets: "all"           # Process all datasets in directory
# datasets: ["cmc"]         # Process only CMC dataset
```

### 3. LLM Configuration (`config/llm/anthropic.yaml`)

```yaml
enabled: true
provider: "anthropic"        # LLM provider
model: "claude-3-haiku-20240307"  # Specific model
max_retries: 3              # API retry attempts
retry_delay: 1.0            # Delay between retries (seconds)
request_timeout: 30         # Request timeout (seconds)
```

### 4. Evaluation Configuration (`config/evaluation/standard.yaml`)

```yaml
n_trials: 5                 # Cross-validation repetitions
cv_folds: 5                 # K-fold cross-validation folds

# Models used to evaluate selected features
models:
  logistic_regression:
    enabled: true
    max_iter: 1000
  
  random_forest:
    enabled: true
    n_estimators: 100
  
  mlp:
    enabled: true
    max_iter: 500
    hidden_layer_sizes: [100]

primary_model: "random_forest"  # Main model for comparisons

# Metrics for different task types
metrics:
  classification: ["accuracy", "roc_auc"]
  regression: ["r2", "rmse"]
```

### 5. Experiment Configuration (`config/experiment.yaml`)

```yaml
experiment:
  name: "llm_feature_engineering_experiment"
  description: "Comprehensive evaluation of feature selection methods"
  results_dir: "results"
  random_seed: 42
  
  # Experiment phases
  run_feature_selection: true    # Run feature selection comparison
  run_autogluon_benchmark: true  # Run AutoGluon baseline comparison
  save_intermediate: false       # Save intermediate results
```

## Dataset Setup

### TALENT Benchmark Datasets

This framework is designed to work with the TALENT benchmark datasets. Download them from:

**[TALENT Datasets - Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z)**

The TALENT benchmark includes 8 carefully selected datasets for feature selection evaluation:

| Dataset | Features | Samples | Task | Domain |
|---------|----------|---------|------|--------|
| CMC | 9 | 1,473 | Classification | Medical |
| ELECTRICITY | 8 | 45,312 | Classification | Energy |
| EYE_MOVEMENTS | 27 | 10,936 | Classification | Biometric |
| KC1 | 21 | 2,109 | Regression | Software |
| PHONEME | 5 | 5,404 | Classification | Audio |
| POL | 26 | 15,000 | Classification | Synthetic |
| SPLICE | 60 | 3,190 | Classification | Bioinformatics |
| VEHICLE | 18 | 846 | Classification | Computer Vision |

### Directory Structure

Place the downloaded datasets in the expected directory structure:

```bash
# Default expected structure (modify in config/data/talent_datasets.yaml if needed)
../datasets_csv/              # Relative to llm_feature_engineering/
├── cmc.csv
├── cmc_metadata.json
├── electricity.csv
├── electricity_metadata.json  
├── eye_movements.csv
├── eye_movements_metadata.json
├── kc1.csv
├── kc1_metadata.json
├── phoneme.csv
├── phoneme_metadata.json
├── pol.csv
├── pol_metadata.json
├── splice.csv
├── splice_metadata.json
├── vehicle.csv
└── vehicle_metadata.json
```

### Supported Dataset Formats

The framework expects CSV files with:
- **Features**: All columns except the last one
- **Target**: Last column (will be automatically detected)
- **Headers**: Required for feature names
- **Missing Values**: Handled automatically during preprocessing

### Adding Custom Datasets

1. **Place CSV file** in the dataset directory
2. **Update configuration**:
   ```yaml
   # In config/data/local_datasets.yaml or talent_datasets.yaml
   datasets: ["existing_dataset", "your_custom_dataset"]
   ```
3. **Optional metadata** - Create `your_dataset_metadata.json` with task type and description
   ```json
   {
     "problem_type": "classification",
     "domain": "healthcare",
     "feature_descriptions": {
       "feature1": "Description of feature 1",
       "feature2": "Description of feature 2"
     }
   }
   ```

## Running Experiments

### Basic Execution
```bash
# Run with default configuration
python -m llm_feature_engineering.experiment_runner

# Run with specific config overrides
python -m llm_feature_engineering.experiment_runner \
  methods.mlp_weights.enabled=false \
  data.datasets=["cmc"]
```

### Method-Specific Execution

**Run only LLM methods:**
```bash
python -m llm_feature_engineering.experiment_runner \
  methods.traditional.mutual_info.enabled=false \
  methods.traditional.random_forest.enabled=false \
  methods.mlp_weights.enabled=false \
  methods.mlp_permutation.enabled=false
```

**Run only traditional methods:**
```bash
python -m llm_feature_engineering.experiment_runner \
  methods.text_based.enabled=false \
  methods.llm4fs.enabled=false \
  methods.mlp_weights.enabled=false \
  methods.mlp_permutation.enabled=false
```

**Run only MLP methods:**
```bash
python -m llm_feature_engineering.experiment_runner \
  methods.text_based.enabled=false \
  methods.llm4fs.enabled=false \
  methods.traditional.mutual_info.enabled=false \
  methods.traditional.random_forest.enabled=false
```

### Configuration Overrides

Hydra allows command-line configuration overrides:

```bash
# Change MLP architecture
python -m llm_feature_engineering.experiment_runner \
  methods.mlp_weights.hidden_layer_sizes=[128,64,32]

# Increase training iterations
python -m llm_feature_engineering.experiment_runner \
  methods.mlp_weights.max_iter=1000 \
  methods.mlp_permutation.max_iter=1000

# Change dataset split
python -m llm_feature_engineering.experiment_runner \
  data.test_size=0.3

# Modify LLM parameters
python -m llm_feature_engineering.experiment_runner \
  methods.text_based.temperature=0.2 \
  methods.llm4fs.sample_size=500
```

## Results Analysis

### Using get_results.py

After running experiments, use the comprehensive analysis script to extract and visualize results:

```bash
# Run analysis on all available experiments (automatic selection)
python get_results.py

# The script will:
# 1. Find all experiment pickle files
# 2. Score them based on completeness (datasets + methods)
# 3. Automatically select the best/most complete results
# 4. Generate comprehensive analysis and visualizations
```

### Output Structure

Each experiment creates a timestamped directory:
```
experiments/YYYY-MM-DD_HH-MM-SS/
├── .hydra/                     # Hydra configuration logs
├── experiment_runner.log       # Execution logs  
├── AutogluonModels/           # AutoGluon model artifacts
└── results/
    └── final_results_*.pkl     # Complete results (pickled data)
```

After running `get_results.py`, you'll get:
```
results_analysis/
├── comprehensive_analysis.png      # Main visualization dashboard
├── method_performance_detailed.png # Method comparison charts
├── detailed_analysis_report.txt    # Statistical analysis report
├── detailed_results.csv           # Raw experimental data
└── summary_table.csv              # Performance summary
```

### Analysis Features

**Automatic Best Results Selection:**
- Scores pickle files based on: number of datasets × 10 + (has_selection_results ? 1 : 0)
- Prioritizes files with complete feature selection results
- Handles experiments with errors gracefully

**Comprehensive Visualizations:**
- **Method Performance Comparison**: Bar charts showing mean scores ± std dev
- **Dataset Difficulty Analysis**: Performance across different datasets
- **Winner Analysis**: Which methods perform best on each dataset
- **Distribution Plots**: Score distributions by method and dataset

**Statistical Analysis:**
- Performance statistics (mean, std, min, max, range)
- Method rankings with confidence intervals
- Dataset complexity analysis
- Winning method frequency analysis

**Detailed Reporting:**
- Executive summary with key findings
- Method-by-method performance breakdown
- Dataset-specific analysis with feature counts
- Recommendations based on results

### Result Components

**Feature Selection Analysis:**
- Selected features for each method and dataset
- Method execution status (success/error)
- Performance scores across multiple trials
- Standard deviations for statistical significance

**Model Performance:**
- Cross-validation scores for each (method, classifier, dataset) combination
- Multi-trial evaluation with statistical aggregation
- Baseline classifier comparisons (logistic regression, MLP, random forest)

**Error Handling:**
- Graceful handling of failed methods
- Clear error reporting in analysis output
- Partial results inclusion when possible

## Advanced Usage

### Custom Feature Selectors

Create new feature selection methods by extending the `FeatureSelector` base class:

```python
from llm_feature_engineering.feature_selection import FeatureSelector

class CustomSelector(FeatureSelector):
    def __init__(self, **params):
        super().__init__("custom_method")
        self.params = params
    
    def select_features(self, X, y, dataset_info=None, top_k=None):
        # Your feature selection logic here
        results = []
        # Return list of feature dictionaries
        return results
```

### Batch Processing

Process multiple configuration combinations:

```python
import hydra
from hydra import compose, initialize
from llm_feature_engineering.experiment_runner import ExperimentRunner

# Initialize Hydra
with initialize(config_path="config"):
    # Different configurations
    configs = [
        {"methods.mlp_weights.hidden_layer_sizes": [32, 16]},
        {"methods.mlp_weights.hidden_layer_sizes": [128, 64]},
        {"data.test_size": 0.15},
        {"data.test_size": 0.25}
    ]
    
    for overrides in configs:
        cfg = compose(config_name="experiment", overrides=list(overrides.items()))
        runner = ExperimentRunner(cfg)
        runner.run_complete_experiment()
```

### Performance Optimization

**For large datasets:**
```yaml
# Reduce sample size for LLM methods
methods:
  llm4fs:
    sample_size: 100  # Instead of 200

# Use fewer CV folds
evaluation:
  cv_folds: 3       # Instead of 5
  n_trials: 3       # Instead of 5
```

**For faster MLP training:**
```yaml
methods:
  mlp_weights:
    max_iter: 200   # Reduce iterations
    hidden_layer_sizes: [32]  # Simpler architecture
  
  mlp_permutation:
    n_repeats: 3    # Fewer permutation repeats
```

## Troubleshooting

### Common Issues

**1. LLM API Errors:**
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test connection
python -c "from llm_feature_engineering.llm_interface import create_llm_interface; llm = create_llm_interface('anthropic', 'claude-3-haiku-20240307'); print(llm.test_connection())"
```

**2. MLP Convergence Warnings:**
```yaml
# Increase iterations or change architecture
methods:
  mlp_weights:
    max_iter: 1000
    hidden_layer_sizes: [64]  # Simpler architecture
```

**3. Memory Issues:**
```yaml
# Reduce dataset size or model complexity
data:
  test_size: 0.1  # Smaller datasets

methods:
  mlp_weights:
    hidden_layer_sizes: [32]  # Smaller networks
```

### Logging

Enable detailed logging:
```bash
export PYTHONUNBUFFERED=1
python -m llm_feature_engineering.experiment_runner --verbose
```

## Performance Tips

### Optimizing Experiment Runtime

1. **Quick Testing**: Use `./run_experiments.sh quick` for initial testing
2. **Selective Methods**: Disable expensive methods in configs:
   ```yaml
   methods:
     text_based:
       enabled: false  # Skip LLM methods for speed
   ```
3. **Reduced Trials**: Lower `evaluation.n_trials` for faster results
4. **Dataset Selection**: Test on smaller datasets first: `data.datasets=[cmc,phoneme]`

### Resource Management

- **Memory**: Monitor memory usage with large datasets (eye_movements, electricity)
- **API Limits**: LLM methods respect rate limits automatically
- **Parallel Processing**: AutoGluon uses multiple cores by default

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Run `pip install -e .` and check PYTHONPATH |
| Missing API key | Set `ANTHROPIC_API_KEY` environment variable |
| Dataset not found | Verify path in `config/data/talent_datasets.yaml` |
| Memory errors | Reduce `evaluation.n_trials` or dataset size |
| Method failures | Check logs in `experiments/*/experiment_runner.log` |

### Debugging

```bash
# Check experiment logs
tail -f experiments/*/experiment_runner.log

# Validate configuration
python -c "from hydra import compose, initialize; initialize(config_path='config'); print('Config OK')"

# Test individual components
python -c "from src.llm_feature_engineering.dataset_manager import DatasetManager; print('Dataset loading OK')"
```