#!/usr/bin/env python3
"""
Individual Method Testing Script
Test each feature selection method individually with CMC dataset
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
from src.llm_feature_engineering.dataset_manager import DatasetManager
from src.llm_feature_engineering.feature_selection import create_feature_selector
from src.llm_feature_engineering.llm_interface import LLMInterface
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def test_method(method_name, method_config, X, y, dataset_info):
    """Test individual method"""
    print(f"\n=== TESTING: {method_name} ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Config: {method_config}")
    
    try:
        # Create selector
        if method_name in ['text_based', 'llm4fs']:
            # Initialize LLM interface for LLM methods
            from src.llm_feature_engineering.llm_interface import AnthropicInterface
            llm_interface = AnthropicInterface(provider='anthropic')
            selector = create_feature_selector(method_name, llm_interface=llm_interface, **method_config)
        elif method_name == 'traditional_rf':
            # Traditional random forest uses "traditional" selector with method parameter
            selector = create_feature_selector('traditional', **method_config)
        else:
            selector = create_feature_selector(method_name, **method_config)
        
        # Run feature selection
        results = selector.select_features(X, y, dataset_info=dataset_info, top_k=method_config.get('top_k', 5))
        
        print(f"✅ Feature selection successful!")
        print(f"   Selected {len(results)} features:")
        for i, result in enumerate(results):
            feature_name = result.get('feature', result.get('name', f'feature_{i}'))
            score = result.get('importance_score', result.get('score', 0))
            print(f"   {i+1}. {feature_name}: {score:.4f}")
        
        # Test with baseline classifiers
        selected_features = [result.get('feature', result.get('name', '')) for result in results]
        X_selected = X[selected_features]
        
        print(f"   Testing with baseline classifiers:")
        
        # Test LogisticRegression
        try:
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr_score = cross_val_score(lr, X_selected, y, cv=3, scoring='accuracy').mean()
            print(f"   - LogisticRegression: {lr_score:.4f}")
        except Exception as e:
            print(f"   - LogisticRegression: ERROR - {e}")
        
        # Test RandomForest
        try:
            rf = RandomForestClassifier(random_state=42, n_estimators=50)
            rf_score = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy').mean()
            print(f"   - RandomForest: {rf_score:.4f}")
        except Exception as e:
            print(f"   - RandomForest: ERROR - {e}")
        
        # Test MLP
        try:
            mlp = MLPClassifier(random_state=42, max_iter=300, hidden_layer_sizes=[100])
            mlp_score = cross_val_score(mlp, X_selected, y, cv=3, scoring='accuracy').mean()
            print(f"   - MLP: {mlp_score:.4f}")
        except Exception as e:
            print(f"   - MLP: ERROR - {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ {method_name} failed: {e}")
        return False

def main():
    """Main testing function"""
    print("Individual Method Testing")
    print("=" * 50)
    
    # Load CMC dataset
    try:
        dataset_manager = DatasetManager('../datasets_csv')
        cmc_data = dataset_manager.load_dataset('cmc')
        
        if not cmc_data:
            print("❌ Could not load CMC dataset")
            return
        
        print(f"Dataset keys: {list(cmc_data.keys())}")
        
        train_data = cmc_data['train_data']
        print(f"Train data keys: {list(train_data.keys())}")
        
        # Extract features and target
        feature_columns = [col for col in train_data.keys() if col != 'target']
        X = train_data[feature_columns]
        y = train_data['target']
        dataset_info = cmc_data.get('metadata', {})
        
        print(f"✅ Loaded CMC dataset: {X.shape}")
        print(f"Features: {list(X.columns)}")
        print(f"Target classes: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Test configurations
    test_configs = {
        'mlp_weights': {
            'enabled': True,
            'top_k': 5,
            'hidden_layer_sizes': [64, 32],
            'max_iter': 300,
            'random_state': 42
        },
        'mlp_permutation': {
            'enabled': True,
            'top_k': 5,
            'hidden_layer_sizes': [64, 32],
            'max_iter': 300,
            'n_repeats': 3,
            'random_state': 42
        },
        'traditional': {
            'enabled': True,
            'top_k': 5,
            'method': 'mutual_info'
        },
        'traditional_rf': {
            'enabled': True,
            'top_k': 5,
            'method': 'random_forest',
            'n_estimators': 100
        },
        'text_based': {
            'enabled': True,
            'top_k': 5,
            'temperature': 0.1,
            'max_tokens': 300
        },
        'llm4fs': {
            'enabled': True,
            'top_k': 5,
            'temperature': 0.1,
            'max_tokens': 3000,
            'sample_size': 200
        }
    }
    
    # Test each method
    results = {}
    for method_name, config in test_configs.items():
        success = test_method(method_name, config, X, y, dataset_info)
        results[method_name] = success
    
    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    for method_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{method_name:20} {status}")
    
    total_passed = sum(results.values())
    total_methods = len(results)
    print(f"\nResults: {total_passed}/{total_methods} methods passed")

if __name__ == "__main__":
    main()