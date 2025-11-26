"""
LLM Feature Engineering Package

A comprehensive package for feature engineering using Large Language Models,
combining traditional machine learning techniques with modern LLM capabilities.
"""

__version__ = "0.1.0"
__author__ = "Edo"

# Main components
from .dataset_manager import DatasetManager
from .llm_interface import create_llm_interface, AnthropicInterface, OpenAIInterface
from .feature_selection import (
    create_feature_selector,
    FeatureSelector,
    TextBasedFeatureSelector,
    LLM4FSHybridSelector,
    TraditionalFeatureSelector,
    MLPWeightSelector,
    MLPPermutationSelector
)

# Traditional AutoML Feature Engineering
from .traditional_fe import (
    create_traditional_fe,
    TraditionalFEMethod,
    FeaturetoolsFE,
    BaselineFE,
    FEATURETOOLS_AVAILABLE,
    AUTOSKLEARN_AVAILABLE
)

# MLP Feature Selection (requires PyTorch)
# PyTorch implementation archived in favor of sklearn implementation
MLP_AVAILABLE = False
from .evaluation import FeatureSelectionEvaluator, LGBM_AVAILABLE
from .autogluon_benchmark import AutoGluonBenchmark
from .experiment_runner import ExperimentRunner

__all__ = [
    # Core components
    "DatasetManager",
    "create_llm_interface",
    "AnthropicInterface", 
    "OpenAIInterface",
    "FeatureSelectionEvaluator",
    "AutoGluonBenchmark",
    "ExperimentRunner",
    
    # Feature selection
    "create_feature_selector",
    "FeatureSelector",
    "TextBasedFeatureSelector",
    "LLM4FSHybridSelector", 
    "TraditionalFeatureSelector",
    "MLPWeightSelector",
    "MLPPermutationSelector",
    
    # Traditional FE
    "create_traditional_fe",
    "TraditionalFEMethod",
    "FeaturetoolsFE",
    "BaselineFE",
    "FEATURETOOLS_AVAILABLE",
    "AUTOSKLEARN_AVAILABLE",
    "LGBM_AVAILABLE",
]