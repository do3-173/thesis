#!/bin/bash
# Full Comparison Script - Based on Loris's Requirements
# Runs complete comparison table experiments

set -e

echo "============================================"
echo "FULL COMPARISON TABLE EXPERIMENTS"
echo "Based on Loris's Nov 26 Email Requirements"
echo "============================================"
echo ""

# Configuration
DATASETS="electricity phoneme kc1"
TRIALS=3
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Configuration:"
echo "- Datasets: $DATASETS"
echo "- Trials: $TRIALS"
echo "- LLM Model: $LLM_MODEL"
echo ""

# Test 1: Quick sanity check (no LLM)
echo "============================================"
echo "TEST 1: Quick Sanity Check (No LLM)"
echo "============================================"
python3 scripts/run_comparison_table.py \
    --datasets electricity \
    --trials 1 \
    --skip-llm

echo ""
echo "Test 1 complete! ✓"
echo ""

# Test 2: Rigorous test with traditional methods (no LLM)
echo "============================================"
echo "TEST 2: Traditional Methods Only (3 trials)"
echo "============================================"
python3 scripts/run_comparison_table.py \
    --datasets $DATASETS \
    --trials $TRIALS \
    --skip-llm

echo ""
echo "Test 2 complete! ✓"
echo ""

# Test 3: Full experiment with LLM (if GPU available)
if command -v nvidia-smi &> /dev/null; then
    echo "============================================"
    echo "TEST 3: Full Experiment with LLM-FE"
    echo "============================================"
    python3 scripts/run_comparison_table.py \
        --datasets $DATASETS \
        --trials $TRIALS \
        --llm-provider huggingface \
        --llm-model "$LLM_MODEL"
    
    echo ""
    echo "Test 3 complete! ✓"
    echo ""
else
    echo "============================================"
    echo "TEST 3: SKIPPED (No GPU detected)"
    echo "============================================"
    echo "To run with LLM, ensure NVIDIA GPU is available"
    echo ""
fi

echo "============================================"
echo "ALL TESTS COMPLETE!"
echo "============================================"
echo "Results saved to: experiments/comparison_table/"
echo ""
echo "Methods tested:"
echo "  ✓ Baseline (original features)"
echo "  ✓ Featuretools (DFS)"
if [ -f "$(python3 -c 'import sys; sys.path.insert(0, "src"); from llm_feature_engineering.traditional_fe import AUTOSKLEARN_AVAILABLE; print(AUTOSKLEARN_AVAILABLE)' 2>/dev/null)" ]; then
    echo "  ✓ Auto-sklearn"
else
    echo "  ⊘ Auto-sklearn (not available)"
fi
if [ -f "$(python3 -c 'import sys; sys.path.insert(0, "src"); from llm_feature_engineering.traditional_fe import AUTOGLUON_AVAILABLE; print(AUTOGLUON_AVAILABLE)' 2>/dev/null)" ]; then
    echo "  ✓ AutoGluon"
else
    echo "  ⊘ AutoGluon (not available)"
fi
if [ "$SKIP_LLM" != "true" ] && command -v nvidia-smi &> /dev/null; then
    echo "  ✓ LLM-FE (Qwen-7B)"
else
    echo "  ⊘ LLM-FE (skipped)"
fi
echo ""
echo "Evaluation:"
echo "  - Models: Logistic Regression, MLP, LGBM"
echo "  - Metrics: Accuracy, ROC-AUC, F1-Score, MCC"
echo "  - Trials: $TRIALS repeats per experiment"
echo ""
