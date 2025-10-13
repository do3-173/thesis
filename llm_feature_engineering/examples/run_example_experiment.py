#!/usr/bin/env python3
"""
Run Feature Selection Experiments

Example script to run feature selection experiments using the LLM Feature Engineering package.
This script demonstrates how to use the package programmatically.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_feature_engineering.dataset_manager import DatasetManager
from llm_feature_engineering.llm_interface import create_llm_interface
from llm_feature_engineering.feature_selection import create_feature_selector
from llm_feature_engineering.evaluation import FeatureSelectionEvaluator


def main():
    """Run a simple feature selection experiment."""
    print("LLM Feature Engineering - Example Experiment")
    print("=" * 50)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager("../../datasets_csv")
    
    # List available datasets
    datasets = dataset_manager.list_datasets()
    print(f"Available datasets: {datasets}")
    
    if not datasets:
        print("No datasets found. Please check the datasets_csv directory.")
        return
    
    # Use the first available dataset
    dataset_name = datasets[0]
    print(f"Using dataset: {dataset_name}")
    
    # Load dataset
    try:
        ml_data = dataset_manager.prepare_for_ml(dataset_name)
        print(f"Dataset shape: {ml_data['X_train'].shape}")
        print(f"Target: {ml_data['target_name']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize evaluator
    evaluator = FeatureSelectionEvaluator(n_trials=1)  # Single trial for demo
    
    # Create traditional feature selectors
    selectors = {
        'mutual_info': create_feature_selector('traditional', method='mutual_info'),
        'random_forest': create_feature_selector('traditional', method='random_forest')
    }
    
    # Try to create LLM-based selectors if API keys are available
    try:
        llm_interface = create_llm_interface('anthropic')
        if llm_interface.test_connection():
            selectors['text_based'] = create_feature_selector('text_based', llm_interface=llm_interface)
            print("LLM-based selector added")
    except Exception as e:
        print(f"LLM interface not available: {e}")
    
    # Run feature selection experiments
    print("\nRunning feature selection experiments...")
    
    for method_name, selector in selectors.items():
        print(f"\nTesting {method_name}...")
        
        try:
            # Run feature selection
            if method_name == 'text_based':
                dataset_info = dataset_manager.get_dataset_info(dataset_name)
                features = selector.select_features(
                    ml_data['X_train'], 
                    ml_data['y_train'],
                    dataset_info=dataset_info,
                    top_k=5
                )
            else:
                features = selector.select_features(
                    ml_data['X_train'], 
                    ml_data['y_train'],
                    top_k=5
                )
            
            if features:
                selected_features = selector.get_feature_names(features)
                print(f"Selected features: {selected_features}")
                
                # Evaluate
                result = evaluator.evaluate_method(
                    method_name,
                    selected_features,
                    ml_data['X_train'],
                    ml_data['y_train']
                )
                
                if 'error' not in result:
                    print("Evaluation successful")
                else:
                    print(f"Evaluation failed: {result['error']}")
            else:
                print("No features selected")
                
        except Exception as e:
            print(f"Error in {method_name}: {e}")
    
    # Show comparison
    comparison_df = evaluator.compare_methods()
    if not comparison_df.empty:
        print("\nMethod Comparison:")
        print(comparison_df)
    else:
        print("\nNo successful evaluations to compare")
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()