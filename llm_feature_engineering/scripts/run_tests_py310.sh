#!/bin/bash
# Run comprehensive tests in Python 3.10 environment
# This runs all methods requested by Loris: Baseline, Featuretools, AutoGluon, Auto-sklearn, LLM-FE

set -e

echo "============================================"
echo "Running Comprehensive Tests (Python 3.10)"
echo "============================================"

# Activate conda environment
source /workspace/miniconda3/bin/activate
conda activate py310

echo "Python version: $(python --version)"
echo ""

# Navigate to project
cd /workspace/thesis/llm_feature_engineering

echo "Configuration:"
echo "- Datasets: electricity, phoneme, kc1"
echo "- Trials: 3 per experiment"
echo "- Methods: Baseline, Featuretools, AutoGluon, Auto-sklearn"
echo "- Models: Logistic Regression, MLP, LGBM"
echo "- Metrics: Accuracy, ROC-AUC, F1-Score, MCC"
echo ""

echo "============================================"
echo "TEST: Full Comparison (3 datasets × 3 trials)"
echo "============================================"

python scripts/run_comparison_table.py \
    --datasets electricity phoneme kc1 \
    --trials 3 \
    --skip-llm

echo ""
echo "============================================"
echo "Tests Complete!"
echo "============================================"
echo ""
echo "Results location: experiments/comparison_table/"
echo ""
echo "To view results:"
echo "  cd experiments/comparison_table"
echo "  cat comparison_table_*.md"
echo ""
