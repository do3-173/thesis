#!/usr/bin/env python3
"""
Simple LLM4FS Test Script

Tests the LLM4FS hybrid feature selection method on a single dataset.
This is a simplified version for quick testing and debugging before running full experiments.

Usage:
    python run_llm4fs_simple.py --dataset bank --provider anthropic
    python run_llm4fs_simple.py --dataset credit-g --provider openai --model gpt-3.5-turbo
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_feature_engineering.llm_interface import create_llm_interface
from llm_feature_engineering.feature_selection import LLM4FSHybridSelector

# TALENT path
TALENT_PATH = Path(__file__).parent.parent / "TALENT"
if TALENT_PATH.exists():
    sys.path.insert(0, str(TALENT_PATH))
    # Set the DATA_PATH for TALENT to find datasets
    import TALENT.model.lib.data as talent_data
    talent_data.DATA_PATH = str(TALENT_PATH)
    from TALENT.model.lib.data import get_dataset
else:
    raise ImportError("TALENT library not found")


def load_talent_dataset(dataset_name: str):
    """
    Load dataset from TALENT benchmark.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        X_train, X_test, y_train, y_test, info
    """
    print(f"\nLoading dataset: {dataset_name}")
    
    # Load data - get_dataset returns (train_val_data, test_data, info)
    # where train_val_data = (N_trainval, C_trainval, y_trainval)
    # and test_data = (N_test, C_test, y_test)
    train_val_data, test_data, info = get_dataset(dataset_name, 'example_datasets')
    
    # Unpack train/val data
    N_trainval, C_trainval, y_trainval = train_val_data
    N_test, C_test, y_test = test_data
    
    # Helper function to combine numerical and categorical features
    def combine_features(N_dict, C_dict, parts):
        """Combine N and C features for given parts (train, val, test)"""
        all_features = []
        
        for part in parts:
            features = []
            
            # Add numerical features
            if N_dict is not None and part in N_dict:
                features.append(N_dict[part])
            
            # Add categorical features  
            if C_dict is not None and part in C_dict:
                features.append(C_dict[part])
            
            # Combine features for this part
            if features:
                X_part = np.hstack(features)
                all_features.append(X_part)
        
        # Concatenate all parts (e.g., train + val)
        if not all_features:
            return None
            
        X_combined = np.vstack(all_features)
        
        # Build feature names
        feature_names = []
        
        # Get a reference part to determine feature structure
        ref_part = parts[0]
        
        # Add numerical feature names
        if N_dict is not None and ref_part in N_dict:
            if 'num_feature_intro' in info:
                feature_names.extend(list(info['num_feature_intro'].keys()))
            else:
                n_num = N_dict[ref_part].shape[1]
                feature_names.extend([f'num_{i}' for i in range(n_num)])
        
        # Add categorical feature names  
        if C_dict is not None and ref_part in C_dict:
            if 'cat_feature_intro' in info:
                feature_names.extend(list(info['cat_feature_intro'].keys()))
            else:
                n_cat = C_dict[ref_part].shape[1]
                feature_names.extend([f'cat_{i}' for i in range(n_cat)])
        
        return pd.DataFrame(X_combined, columns=feature_names)
    
    # Combine train and val
    X_train = combine_features(N_trainval, C_trainval, ['train', 'val'])
    X_test = combine_features(N_test, C_test, ['test'])
    
    # Combine train and val labels
    if 'train' in y_trainval and 'val' in y_trainval:
        y_train = np.concatenate([y_trainval['train'], y_trainval['val']])
    else:
        y_train = y_trainval['train'] if 'train' in y_trainval else y_trainval['val']
    
    y_test_array = y_test['test']
    
    # Convert to pandas Series
    y_train = pd.Series(y_train, name='target')
    y_test_series = pd.Series(y_test_array, name='target')
    
    print(f"Dataset shape: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    print(f"Task: {info.get('task_type', 'unknown')}")
    
    return X_train, X_test, y_train, y_test_series, info


def evaluate_features(X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate selected features using L2 Logistic Regression with grid search.
    
    Args:
        X_train: Training features
        X_test: Test features  
        y_train: Training labels
        y_test: Test labels
        feature_names: Names of features to use
        
    Returns:
        AUROC score
    """
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    
    # Select features
    X_train_selected = X_train[feature_names].copy()
    X_test_selected = X_test[feature_names].copy()
    
    # Encode categorical features (those with string/object dtype)
    # Use OrdinalEncoder which handles unseen categories better
    categorical_cols = []
    for col in X_train_selected.columns:
        if X_train_selected[col].dtype == 'object':
            categorical_cols.append(col)
    
    if categorical_cols:
        # Use OrdinalEncoder with handle_unknown='use_encoded_value' for unseen categories
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_selected[categorical_cols] = encoder.fit_transform(X_train_selected[categorical_cols])
        X_test_selected[categorical_cols] = encoder.transform(X_test_selected[categorical_cols])
    
    # Convert all to float
    X_train_selected = X_train_selected.astype(float)
    X_test_selected = X_test_selected.astype(float)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Grid search for C parameter (as per paper)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    lr = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    grid_search = GridSearchCV(
        lr, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predict and evaluate
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Best C: {grid_search.best_params_['C']}")
    print(f"  CV AUROC: {grid_search.best_score_:.4f}")
    print(f"  Test AUROC: {auroc:.4f}")
    
    return auroc


def run_llm4fs_test(dataset_name: str, provider: str = 'anthropic', model: str = None):
    """
    Run LLM4FS feature selection test on a single dataset.
    
    Args:
        dataset_name: Name of dataset to test
        provider: LLM provider ('anthropic' or 'openai')
        model: Model name (optional, uses default if None)
    """
    print("="*80)
    print(f"LLM4FS Simple Test")
    print(f"Dataset: {dataset_name}")
    print(f"Provider: {provider}")
    print(f"Model: {model or 'default'}")
    print("="*80)
    
    # Load dataset
    X_train, X_test, y_train, y_test, info = load_talent_dataset(dataset_name)
    
    # Create LLM interface
    print(f"\nInitializing {provider} LLM interface...")
    llm = create_llm_interface(provider, model)
    
    # Test connection
    if not llm.test_connection():
        print("ERROR: LLM connection failed")
        return
    
    # Create LLM4FS selector
    selector = LLM4FSHybridSelector(llm)
    
    # Run feature selection
    print("\n" + "="*80)
    print("Running LLM4FS Hybrid Feature Selection")
    print("="*80)
    
    dataset_info = {'metadata': info}
    selected_features = selector.select_features(
        X_train, 
        y_train, 
        dataset_info=dataset_info
    )
    
    if not selected_features:
        print("ERROR: Feature selection failed")
        return
    
    # Display results
    print("\n" + "="*80)
    print("Feature Selection Results")
    print("="*80)
    print(f"\nTop {min(10, len(selected_features))} features:")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"{i:2d}. {feat['name']:20s} - Score: {feat['importance_score']:.4f}")
    
    # Evaluate at different feature proportions
    print("\n" + "="*80)
    print("Evaluation at Different Feature Proportions")
    print("="*80)
    
    n_features = len(selected_features)
    proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    for prop in proportions:
        n_select = max(1, int(n_features * prop))
        # Filter out any features that don't exist in X_train
        top_features = [f['name'] for f in selected_features[:n_select] if f['name'] in X_train.columns]
        
        if not top_features:
            print(f"\nSkipping {prop*100:.0f}% - no valid features")
            continue
        
        print(f"\nTop {prop*100:.0f}% features ({len(top_features)} features):")
        auroc = evaluate_features(X_train, X_test, y_train, y_test, top_features)
        
        results.append({
            'proportion': prop,
            'n_features': len(top_features),
            'auroc': auroc
        })
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\nDataset: {dataset_name}")
    print(f"Total features: {n_features}")
    print(f"\nAUROC by feature proportion:")
    print(f"{'Proportion':>10s} {'N Features':>10s} {'AUROC':>10s}")
    print("-" * 32)
    for r in results:
        print(f"{r['proportion']:>10.1%} {r['n_features']:>10d} {r['auroc']:>10.4f}")
    
    # Best result
    best_result = max(results, key=lambda x: x['auroc'])
    print(f"\nBest AUROC: {best_result['auroc']:.4f} at {best_result['proportion']:.1%} ({best_result['n_features']} features)")
    
    # Save results
    output_dir = Path("experiments") / "llm4fs_simple_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{dataset_name}_{provider}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'provider': provider,
            'model': model or 'default',
            'features': selected_features,
            'evaluation': results,
            'best_result': best_result
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test LLM4FS on a single dataset')
    parser.add_argument('--dataset', type=str, default='bank',
                       choices=['bank', 'credit-g', 'pima-diabetes', 'give-me-some-credit'],
                       help='Dataset to test (default: bank)')
    parser.add_argument('--provider', type=str, default='anthropic',
                       choices=['anthropic', 'openai'],
                       help='LLM provider (default: anthropic)')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model name (optional, uses provider default)')
    
    args = parser.parse_args()
    
    run_llm4fs_test(args.dataset, args.provider, args.model)


if __name__ == '__main__':
    main()
