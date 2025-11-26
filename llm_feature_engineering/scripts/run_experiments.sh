#!/bin/bash

# LLM Feature Engineering Experiments Runner
# This script provides easy ways to run different experimental configurations

set -e

# Default configuration
CONFIG_DIR="$(dirname "$0")/../config"
SRC_DIR="$(dirname "$0")/../src"

# Add src to Python path
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

echo "LLM Feature Engineering Experiments"
echo "====================================="

# Function to run experiment with Hydra configuration
run_experiment() {
    local data_config="$1"
    local methods_config="$2"
    local evaluation_config="$3"
    local llm_config="$4"
    local benchmark_config="$5"
    shift 5
    local additional_overrides="$*"
    
    echo "Configuration:"
    echo "  Data: $data_config"
    echo "  Methods: $methods_config" 
    echo "  Evaluation: $evaluation_config"
    echo "  LLM: $llm_config"
    echo "  Benchmark: $benchmark_config"
    [[ -n "$additional_overrides" ]] && echo "  Additional: $additional_overrides"
    echo "-------------------------------------"
    
    cd "$(dirname "$0")/.."
    python -m src.llm_feature_engineering.experiment_runner \
        dataset="$data_config" \
        feature_engineering="$methods_config" \
        evaluation="$evaluation_config" \
        llm="$llm_config" \
        benchmark="$benchmark_config" \
        $additional_overrides
}

# Check command line arguments
case "${1:-help}" in
    "quick")
        echo "Running quick experiment (traditional methods only, 3 datasets)..."
        run_experiment "local_datasets" "traditional_only" "fast" "disabled" "disabled" \
            "data.datasets=[cmc,vehicle,electricity]"
        ;;
    
    "local")
        echo "Running experiment on local datasets..."
        run_experiment "local_datasets" "complete_methods" "standard" "anthropic" "disabled"
        ;;
    
    "talent")
        echo "Running experiment on TALENT datasets..."
        run_experiment "talent_datasets" "complete_methods" "standard" "anthropic" "disabled"
        ;;
    
    "complete")
        echo "Running complete experiment with all methods and multi-trial evaluation..."
        run_experiment "talent_datasets" "complete_methods" "multi_trial" "anthropic" "disabled"
        ;;
    
    "multi-trial")
        echo "Running 5-trial statistical experiment..."
        run_experiment "talent_datasets" "complete_methods" "multi_trial" "anthropic" "disabled"
        ;;
    
    "traditional")
        echo "Running experiment with traditional methods only..."
        run_experiment "talent_datasets" "traditional_only" "fast" "disabled" "disabled"
        ;;
    
    "mlp")
        echo "Running experiment with MLP methods only..."
        run_experiment "talent_datasets" "mlp_only" "fast" "disabled" "disabled"
        ;;
    
    "no-llm")
        echo "Running experiment without LLM methods (traditional + MLP)..."
        run_experiment "talent_datasets" "complete_methods" "fast" "disabled" "disabled" \
            "methods.text_based.enabled=false methods.llm4fs.enabled=false"
        ;;
    
    "anthropic")
        echo "Running experiment with Anthropic Claude LLM..."
        run_experiment "talent_datasets" "complete_methods" "fast" "anthropic" "disabled"
        ;;
    
    "openai")
        echo "Running experiment with OpenAI GPT LLM..."
        run_experiment "talent_datasets" "complete_methods" "fast" "openai" "disabled"
        ;;
    
    "benchmark-only")
        echo "Running AutoGluon benchmark only..."
        run_experiment "talent_datasets" "complete_methods" "fast" "disabled" "autogluon_standard" \
            "experiment.run_feature_selection=false"
        ;;
    
    "test-single")
        echo "Running single dataset test..."
        run_experiment "talent_datasets" "traditional_only" "fast" "disabled" "disabled" \
            "data.datasets=[cmc]"
        ;;

    "reproduction")
        echo "Running reproduction experiment (LLM4FS + CAAFE on benchmark datasets)..."
        run_experiment "talent_datasets" "reproduction" "llm4fs_evaluation" "huggingface" "disabled"
        ;;
    
    "help"|*)
        echo "Usage: $0 [COMMAND] [additional_overrides...]"
        echo ""
        echo "Commands:"
        echo "  quick         Run quick test (traditional methods, 3 datasets)"
        echo "  local         Run experiment on local CSV datasets"
        echo "  talent        Run experiment on TALENT benchmark datasets"
        echo "  complete      Run complete experiment (all methods + multi-trial)"
        echo "  multi-trial   Run 5-trial statistical experiment"
        echo "  traditional   Run traditional methods only"
        echo "  mlp           Run MLP methods only"
        echo "  no-llm        Run without LLM methods (traditional + MLP)"
        echo "  anthropic     Run with Anthropic Claude LLM"
        echo "  openai        Run with OpenAI GPT LLM"
        echo "  benchmark-only Run only AutoGluon benchmarks"
        echo "  test-single   Run single dataset test"
        echo "  reproduction  Run reproduction experiment"
        echo "  help          Show this help message"
        echo ""
        echo "Available Configurations:"
        echo "  Data: local_datasets, talent_datasets"
        echo "  Methods: traditional_only, mlp_only, complete_methods"
        echo "  Evaluation: fast, standard, multi_trial"
        echo "  LLM: disabled, anthropic, openai"
        echo "  Benchmark: disabled, autogluon_fast, autogluon_standard"
        echo ""
        echo "Examples:"
        echo "  $0 quick                    # Quick test"
        echo "  $0 traditional              # Traditional methods only"
        echo "  $0 mlp                      # MLP methods only"
        echo "  $0 no-llm                   # Traditional + MLP (no LLM)"
        echo "  $0 complete                 # All methods with multi-trial"
        echo "  $0 test-single              # Single dataset test"
        echo ""
        echo "Custom overrides can be added as additional arguments:"
        echo "  $0 traditional data.datasets=[cmc] evaluation.n_trials=3"
        ;;
esac