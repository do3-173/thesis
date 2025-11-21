#!/usr/bin/env python3
"""
LLM4FS Paper Reproduction - Hydra Experiment Runner
===================================================
Reproduces results from "LLM4FS: Leveraging Large Language Models for Feature Selection"
using Hydra configuration management.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import LLM4FS components
from llm_feature_engineering.llm_interface import create_llm_interface
from llm_feature_engineering.feature_selection import LLM4FSHybridSelector

# Set up TALENT data path before importing
TALENT_PATH = Path(__file__).parent.parent / "TALENT" / "example_datasets"
import TALENT.model.lib.data as talent_data
talent_data.DATA_PATH = str(TALENT_PATH)
from TALENT.model.lib.data import get_dataset


def load_talent_dataset(dataset_name: str):
    """Load dataset from TALENT framework."""
    print(f"\nLoading {dataset_name} dataset from TALENT...")
    
    # Load train and validation data
    N_train, C_train, y_train = get_dataset(dataset_name, 'train', merge=True)
    N_test, C_test, y_test = get_dataset(dataset_name, 'test')
    
    # Combine train and validation sets (TALENT splits train into train/val)
    # We need the full training set for our experiments
    N_val, C_val, y_val = get_dataset(dataset_name, 'val')
    N_train_full = np.vstack([N_train, N_val])
    C_train_full = np.vstack([C_train, C_val]) if C_train is not None and C_val is not None else C_train
    y_train_full = np.concatenate([y_train, y_val])
    
    # Combine numerical and categorical features
    def combine_features(N, C, reference_part='train'):
        """Combine numerical and categorical features."""
        parts = []
        feature_names = []
        
        if N is not None and N.size > 0:
            parts.append(N)
            feature_names.extend([f'num_{i}' for i in range(N.shape[1])])
        
        if C is not None and C.size > 0:
            parts.append(C)
            feature_names.extend([f'cat_{i}' for i in range(C.shape[1])])
        
        if not parts:
            raise ValueError(f"No features found in {reference_part} set")
        
        combined = np.hstack(parts)
        return pd.DataFrame(combined, columns=feature_names)
    
    # Create DataFrames
    X_train = combine_features(N_train_full, C_train_full, 'train')
    X_test = combine_features(N_test, C_test, 'test')
    y_train = pd.Series(y_train_full)
    y_test = pd.Series(y_test)
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")
    
    return X_train, X_test, y_train, y_test


def evaluate_features(X_train, X_test, y_train, y_test, selected_features, C_values, cv_folds=5):
    """
    Evaluate selected features using L2-regularized Logistic Regression.
    Matches the paper's evaluation protocol.
    """
    if not selected_features:
        return 0.0
    
    # Filter to selected features
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    # Identify categorical columns (those starting with 'cat_')
    categorical_cols = []
    for col in X_train_selected.columns:
        if X_train_selected[col].dtype == 'object':
            categorical_cols.append(col)
    
    # Encode categorical features
    if categorical_cols:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X_train_selected[categorical_cols] = encoder.fit_transform(X_train_selected[categorical_cols])
        X_test_selected[categorical_cols] = encoder.transform(X_test_selected[categorical_cols])
    
    # Grid search with cross-validation
    param_grid = {'C': C_values}
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train)
    
    # Evaluate on test set
    y_pred_proba = grid_search.predict_proba(X_test_selected)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    return auroc


def run_llm4fs_experiment(cfg: DictConfig, dataset_name: str):
    """Run LLM4FS experiment on a single dataset."""
    
    print(f"\n{'='*80}")
    print(f"Running LLM4FS on {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_talent_dataset(dataset_name)
    
    # Initialize LLM interface
    llm_interface = create_llm_interface(
        provider=cfg.llm.provider,
        model=cfg.llm.model
    )
    
    # Initialize LLM4FS selector
    selector = LLM4FSHybridSelector(
        llm_interface=llm_interface,
        sample_size=cfg.methods.llm4fs.sample_size,
        temperature=cfg.methods.llm4fs.temperature,
        max_tokens=cfg.methods.llm4fs.max_tokens
    )
    
    # Run feature selection
    print(f"\nRunning LLM4FS feature selection (sample_size={cfg.methods.llm4fs.sample_size})...")
    features = selector.select_features(X_train, y_train)
    
    if not features:
        print("ERROR: No features selected!")
        return None
    
    # Filter to only features that exist in the dataset
    valid_features = [f for f in features if f['name'] in X_train.columns]
    
    print(f"\nLLM4FS extracted {len(valid_features)} features:")
    for i, feat in enumerate(valid_features[:10], 1):
        print(f"  {i}. {feat['name']}: {feat['score']:.4f}")
    if len(valid_features) > 10:
        print(f"  ... and {len(valid_features) - 10} more")
    
    # Sort features by score (descending)
    sorted_features = sorted(valid_features, key=lambda x: x['score'], reverse=True)
    
    # Evaluate at different feature proportions
    print(f"\nEvaluating features at different proportions...")
    results = {
        'dataset': dataset_name,
        'total_features': X_train.shape[1],
        'selected_features': len(sorted_features),
        'feature_scores': sorted_features,
        'proportion_results': []
    }
    
    for proportion in cfg.evaluation.feature_proportions:
        n_features = max(1, int(len(sorted_features) * proportion))
        top_features = [f['name'] for f in sorted_features[:n_features]]
        
        # Evaluate
        auroc = evaluate_features(
            X_train, X_test, y_train, y_test,
            top_features,
            cfg.methods.baseline_classifiers.logistic_regression_l2.C_values,
            cfg.methods.baseline_classifiers.logistic_regression_l2.cv_folds
        )
        
        results['proportion_results'].append({
            'proportion': proportion,
            'n_features': n_features,
            'auroc': auroc
        })
        
        print(f"  {int(proportion*100):3d}% features ({n_features:2d}): AUROC = {auroc:.4f}")
    
    # Find best result
    best_result = max(results['proportion_results'], key=lambda x: x['auroc'])
    results['best_auroc'] = best_result['auroc']
    results['best_proportion'] = best_result['proportion']
    results['best_n_features'] = best_result['n_features']
    
    print(f"\nBest result: {best_result['auroc']:.4f} AUROC using {best_result['n_features']} features ({int(best_result['proportion']*100)}%)")
    
    return results


def run_multiple_trials(cfg: DictConfig, dataset_name: str, n_trials: int):
    """Run LLM4FS multiple times for statistical significance."""
    
    print(f"\n{'='*80}")
    print(f"Running {n_trials} trials on {dataset_name.upper()}")
    print(f"{'='*80}")
    
    all_trials = []
    
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        try:
            result = run_llm4fs_experiment(cfg, dataset_name)
            if result:
                all_trials.append(result)
        except Exception as e:
            print(f"ERROR in trial {trial + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_trials:
        print(f"ERROR: No successful trials for {dataset_name}")
        return None
    
    # Aggregate results
    aggregated = {
        'dataset': dataset_name,
        'n_trials': len(all_trials),
        'trials': all_trials,
        'statistics': {}
    }
    
    # Calculate statistics for each proportion
    proportions = cfg.evaluation.feature_proportions
    for proportion in proportions:
        aurocs = []
        for trial in all_trials:
            for prop_result in trial['proportion_results']:
                if prop_result['proportion'] == proportion:
                    aurocs.append(prop_result['auroc'])
                    break
        
        if aurocs:
            aggregated['statistics'][f'proportion_{int(proportion*100)}'] = {
                'mean': np.mean(aurocs),
                'std': np.std(aurocs),
                'min': np.min(aurocs),
                'max': np.max(aurocs)
            }
    
    # Overall best statistics
    best_aurocs = [trial['best_auroc'] for trial in all_trials]
    aggregated['best_auroc_mean'] = np.mean(best_aurocs)
    aggregated['best_auroc_std'] = np.std(best_aurocs)
    aggregated['best_auroc_min'] = np.min(best_aurocs)
    aggregated['best_auroc_max'] = np.max(best_aurocs)
    
    print(f"\n{'='*80}")
    print(f"Aggregated results for {dataset_name.upper()} ({len(all_trials)} trials)")
    print(f"{'='*80}")
    print(f"Best AUROC: {aggregated['best_auroc_mean']:.4f} ± {aggregated['best_auroc_std']:.4f}")
    print(f"Range: [{aggregated['best_auroc_min']:.4f}, {aggregated['best_auroc_max']:.4f}]")
    
    return aggregated


@hydra.main(version_base=None, config_path="config/experiments", config_name="llm4fs_reproduction")
def main(cfg: DictConfig):
    """Main entry point for LLM4FS reproduction experiments."""
    
    print("="*80)
    print("LLM4FS PAPER REPRODUCTION - HYDRA EXPERIMENTS")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get datasets to process
    datasets = cfg.data.datasets
    print(f"\nDatasets to process: {datasets}")
    
    # Run experiments
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# Processing dataset: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        try:
            if cfg.evaluation.n_trials > 1:
                # Run multiple trials for statistical significance
                result = run_multiple_trials(cfg, dataset_name, cfg.evaluation.n_trials)
            else:
                # Single trial
                result = run_llm4fs_experiment(cfg, dataset_name)
            
            if result:
                all_results[dataset_name] = result
                
                # Save individual dataset results
                dataset_output_file = output_dir / f"{dataset_name}_results.json"
                with open(dataset_output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {dataset_output_file}")
        
        except Exception as e:
            print(f"\nERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all results
    if all_results:
        all_results_file = output_dir / "all_results.json"
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"All results: {all_results_file}")
        
        # Print summary
        print(f"\nSummary:")
        for dataset_name, result in all_results.items():
            if 'best_auroc_mean' in result:
                # Multiple trials
                print(f"  {dataset_name}: {result['best_auroc_mean']:.4f} ± {result['best_auroc_std']:.4f} AUROC")
            else:
                # Single trial
                print(f"  {dataset_name}: {result['best_auroc']:.4f} AUROC")
    else:
        print("\nNo successful experiments!")


if __name__ == "__main__":
    main()
