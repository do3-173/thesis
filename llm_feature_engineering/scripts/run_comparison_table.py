#!/usr/bin/env python3
"""
Comparison Table Experiment Script

Runs the experiments required for Loris's comparison table:
- Baseline (original features)
- Traditional AutoML FE (Featuretools, Auto-sklearn, AutoGluon)
- LLM-FE ("Too Many Simple Features" paper)

Evaluated on:
- 3 downstream models: Logistic Regression, MLP, LGBM
- Metrics: Accuracy (mean±std), ROC-AUC, F1-Score, MCC
- 3 repeats per experiment
"""

# Suppress warnings before any imports
import warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.pipeline import Pipeline

# Try importing optional dependencies
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: LightGBM not installed. Run: pip install lightgbm")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_feature_engineering.dataset_manager import DatasetManager
from llm_feature_engineering.traditional_fe import (
    create_traditional_fe, 
    FeaturetoolsFE, 
    BaselineFE,
    FEATURETOOLS_AVAILABLE,
    AUTOSKLEARN_AVAILABLE,
    AUTOGLUON_AVAILABLE
)

# Configuration
DEFAULT_CONFIG = {
    # Datasets to use (select 3 diverse ones)
    "datasets": ["electricity", "phoneme", "kc1"],
    
    # Number of experiment repeats
    "n_trials": 3,
    
    # Cross-validation folds
    "cv_folds": 5,
    
    # Random seed for reproducibility
    "random_seed": 42,
    
    # Output directory
    "output_dir": "experiments/comparison_table",
    
    # LLM settings (for LLM-FE) - Using local HuggingFace models
    "llm_provider": "huggingface",
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",  # Good for 80GB VRAM
    # Alternative models:
    # "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
    # "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
    # "llm_model": "Qwen/Qwen2.5-72B-Instruct",  # For 80GB VRAM
    
    # Featuretools settings
    "featuretools_max_depth": 2,
    "featuretools_max_features": 50,
    
    # AutoGluon settings
    "autogluon_degree": 2,  # Polynomial degree (2 = quadratic + interactions)
    "autogluon_max_features": 50,  # Max features to keep (by variance)
    
    # Auto-sklearn settings
    "autosklearn_time_limit": 60,
}


class ComparisonTableExperiment:
    """
    Runs comparison experiments for the thesis table.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.results = []
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset manager
        dataset_dir = Path(__file__).parent.parent.parent / "datasets_csv"
        self.dataset_manager = DatasetManager(str(dataset_dir))
        
        # Set random seed
        np.random.seed(self.config["random_seed"])
        
    def get_models(self) -> Dict[str, Any]:
        """Get the three required downstream models with proper scaling."""
        # Use Pipeline with StandardScaler for LR to fix convergence warnings
        models = {
            'LR': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=2000))
            ]),
            'MLP': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100,)))
            ]),
        }
        
        if LGBM_AVAILABLE:
            models['LGBM'] = LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
        else:
            print("WARNING: LGBM not available, using only LR and MLP")
            
        return models
    
    def get_scorers(self) -> Dict[str, Any]:
        """Get all required metrics."""
        return {
            'accuracy': 'accuracy',
            'roc_auc': 'roc_auc_ovr_weighted',
            'f1': 'f1_weighted',
            'mcc': make_scorer(matthews_corrcoef)
        }
    
    def evaluate_method(self, X: pd.DataFrame, y: pd.Series, 
                       method_name: str, trial: int) -> List[Dict]:
        """
        Evaluate a feature engineering method with all models and metrics.
        
        Args:
            X: Feature matrix (already transformed)
            y: Target variable
            method_name: Name of the FE method
            trial: Trial number
            
        Returns:
            List of result dictionaries
        """
        results = []
        models = self.get_models()
        scorers = self.get_scorers()
        
        # Encode target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y.values
        
        # CV splitter
        cv = StratifiedKFold(
            n_splits=self.config["cv_folds"], 
            shuffle=True, 
            random_state=self.config["random_seed"] + trial
        )
        
        for model_name, model in models.items():
            try:
                # No need to manually scale - Pipeline handles it for LR and MLP
                X_eval = X
                
                # Run cross-validation with all metrics
                cv_results = cross_validate(
                    model, X_eval, y_encoded,
                    cv=cv,
                    scoring=scorers,
                    return_train_score=False,
                    n_jobs=-1
                )
                
                # Store results
                result = {
                    'method': method_name,
                    'model': model_name,
                    'trial': trial,
                    'n_features': X.shape[1],
                    'accuracy_mean': cv_results['test_accuracy'].mean(),
                    'accuracy_std': cv_results['test_accuracy'].std(),
                    'roc_auc_mean': cv_results['test_roc_auc'].mean(),
                    'roc_auc_std': cv_results['test_roc_auc'].std(),
                    'f1_mean': cv_results['test_f1'].mean(),
                    'f1_std': cv_results['test_f1'].std(),
                    'mcc_mean': cv_results['test_mcc'].mean(),
                    'mcc_std': cv_results['test_mcc'].std(),
                }
                results.append(result)
                print(f"  {model_name}: Acc={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}, "
                      f"MCC={result['mcc_mean']:.4f}")
                
            except Exception as e:
                print(f"  ERROR with {model_name}: {e}")
                results.append({
                    'method': method_name,
                    'model': model_name,
                    'trial': trial,
                    'error': str(e)
                })
        
        return results
    
    def run_baseline(self, X: pd.DataFrame, y: pd.Series, 
                    dataset_name: str, trial: int) -> List[Dict]:
        """Run baseline (original features with basic preprocessing)."""
        print(f"  Running Baseline (original features)...")
        
        baseline = BaselineFE(scale=False, impute=True)
        X_baseline = baseline.fit_transform(X, y)
        
        return self.evaluate_method(X_baseline, y, "Baseline", trial)
    
    def run_featuretools(self, X: pd.DataFrame, y: pd.Series,
                        dataset_name: str, trial: int) -> List[Dict]:
        """Run Featuretools feature engineering."""
        if not FEATURETOOLS_AVAILABLE:
            print("  SKIP: Featuretools not installed")
            return [{'method': 'Featuretools', 'error': 'Not installed', 'trial': trial}]
        
        print(f"  Running Featuretools DFS...")
        try:
            ft_fe = FeaturetoolsFE(
                max_depth=self.config["featuretools_max_depth"],
                max_features=self.config["featuretools_max_features"]
            )
            X_ft = ft_fe.fit_transform(X, y)
            return self.evaluate_method(X_ft, y, "Featuretools", trial)
        except Exception as e:
            print(f"  ERROR: Featuretools failed: {e}")
            return [{'method': 'Featuretools', 'error': str(e), 'trial': trial}]
    
    def run_autosklearn(self, X: pd.DataFrame, y: pd.Series,
                       dataset_name: str, trial: int) -> List[Dict]:
        """Run Auto-sklearn feature preprocessing."""
        if not AUTOSKLEARN_AVAILABLE:
            print("  SKIP: Auto-sklearn not installed (requires Linux)")
            return [{'method': 'Auto-sklearn', 'error': 'Not installed', 'trial': trial}]
        
        print(f"  Running Auto-sklearn preprocessing...")
        try:
            from llm_feature_engineering.traditional_fe import AutoSklearnFE
            autoskl_fe = AutoSklearnFE(
                time_left_for_this_task=self.config["autosklearn_time_limit"]
            )
            X_autoskl = autoskl_fe.fit_transform(X, y)
            return self.evaluate_method(X_autoskl, y, "Auto-sklearn", trial)
        except Exception as e:
            print(f"  ERROR: Auto-sklearn failed: {e}")
            return [{'method': 'Auto-sklearn', 'error': str(e), 'trial': trial}]
    
    def run_autogluon(self, X: pd.DataFrame, y: pd.Series,
                     dataset_name: str, trial: int) -> List[Dict]:
        """Run AutoGluon feature generation."""
        if not AUTOGLUON_AVAILABLE:
            print("  SKIP: AutoGluon not installed")
            return [{'method': 'AutoGluon', 'error': 'Not installed', 'trial': trial}]
        
        print(f"  Running AutoGluon feature generation...")
        try:
            from llm_feature_engineering.traditional_fe import AutoGluonFE
            autogluon_fe = AutoGluonFE(
                degree=self.config["autogluon_degree"],
                max_features=self.config["autogluon_max_features"],
                verbosity=0
            )
            X_autogluon = autogluon_fe.fit_transform(X, y)
            return self.evaluate_method(X_autogluon, y, "AutoGluon", trial)
        except Exception as e:
            print(f"  ERROR: AutoGluon failed: {e}")
            return [{'method': 'AutoGluon', 'error': str(e), 'trial': trial}]
    
    def run_llm_fe(self, X: pd.DataFrame, y: pd.Series,
                   dataset_name: str, trial: int,
                   dataset_info: Dict = None) -> List[Dict]:
        """Run LLM-based feature engineering (LLM4FS) using local models."""
        
        # Check if LLM experiments are skipped
        if self.config.get("skip_llm", False):
            print("  SKIP: LLM-FE (--skip-llm flag set)")
            return [{'method': 'LLM-FE', 'error': 'Skipped', 'trial': trial}]
        
        print(f"  Running LLM-FE (LLM4FS) with {self.config['llm_model']}...")
        
        try:
            from llm_feature_engineering.llm_interface import create_llm_interface
            from llm_feature_engineering.feature_selection import LLM4FSHybridSelector
            
            # Create LLM interface (supports huggingface, openai, anthropic)
            llm = create_llm_interface(
                provider=self.config["llm_provider"],
                model=self.config["llm_model"]
            )
            
            # Create selector
            selector = LLM4FSHybridSelector(llm)
            
            # Run feature selection
            selected = selector.select_features(X, y, dataset_info)
            
            if not selected:
                return [{'method': 'LLM-FE', 'error': 'No features selected', 'trial': trial}]
            
            # Get top features (use all selected or top 50%)
            n_select = max(1, len(selected) // 2)
            top_features = [f['name'] for f in selected[:n_select] if f['name'] in X.columns]
            
            if not top_features:
                return [{'method': 'LLM-FE', 'error': 'Selected features not in dataset', 'trial': trial}]
            
            X_llm = X[top_features]
            return self.evaluate_method(X_llm, y, "LLM-FE", trial)
            
        except Exception as e:
            import traceback
            print(f"  ERROR: LLM-FE failed: {e}")
            traceback.print_exc()
            return [{'method': 'LLM-FE', 'error': str(e), 'trial': trial}]
    
    def run_dataset_experiments(self, dataset_name: str) -> pd.DataFrame:
        """Run all experiments for a single dataset."""
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        try:
            dataset_result = self.dataset_manager.load_dataset(
                dataset_name, 
                source="auto"  # Let it auto-detect source
            )
            
            # Extract X and y from the result dictionary
            if dataset_result.get('has_splits', False):
                # TALENT dataset with train/test splits
                train_df = dataset_result['train_data']
                # Combine train and test for CV evaluation
                if 'test_data' in dataset_result and dataset_result['test_data'] is not None:
                    df = pd.concat([train_df, dataset_result['test_data']], ignore_index=True)
                else:
                    df = train_df
            else:
                # Local dataset
                df = dataset_result['data']
            
            # Get target column from metadata
            metadata = dataset_result.get('metadata', {})
            target_col = metadata.get('target_column', metadata.get('target', df.columns[-1]))
            
            # Split into X and y
            if target_col in df.columns:
                y = df[target_col]
                X = df.drop(columns=[target_col])
            else:
                # Assume last column is target
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
            
            # Handle any non-numeric features by encoding
            for col in X.columns:
                if X[col].dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Fill NaN values
            X = X.fillna(X.median())
            
            dataset_info = {
                'name': dataset_name,
                'metadata': metadata,
                'shape': X.shape,
                'columns': list(X.columns)
            }
            
            print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            import traceback
            print(f"ERROR loading dataset: {e}")
            traceback.print_exc()
            return pd.DataFrame()
        
        all_results = []
        
        for trial in range(self.config["n_trials"]):
            print(f"\n--- Trial {trial + 1}/{self.config['n_trials']} ---")
            
            # Run all methods
            all_results.extend(self.run_baseline(X, y, dataset_name, trial))
            all_results.extend(self.run_featuretools(X, y, dataset_name, trial))
            all_results.extend(self.run_autosklearn(X, y, dataset_name, trial))
            all_results.extend(self.run_autogluon(X, y, dataset_name, trial))
            all_results.extend(self.run_llm_fe(X, y, dataset_name, trial, dataset_info))
        
        # Add dataset name to results
        for r in all_results:
            r['dataset'] = dataset_name
        
        return pd.DataFrame(all_results)
    
    def aggregate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results across trials to produce the final table.
        
        Returns table with mean±std for each metric.
        """
        if df.empty:
            return df
        
        # Filter out errors
        df_valid = df[~df['accuracy_mean'].isna()].copy()
        
        if df_valid.empty:
            return pd.DataFrame()
        
        # Aggregate by dataset, method, model
        agg_funcs = {
            'accuracy_mean': ['mean', 'std'],
            'roc_auc_mean': ['mean', 'std'],
            'f1_mean': ['mean', 'std'],
            'mcc_mean': ['mean', 'std'],
            'n_features': 'first'
        }
        
        aggregated = df_valid.groupby(['dataset', 'method', 'model']).agg(agg_funcs)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated = aggregated.reset_index()
        
        return aggregated
    
    def format_table(self, df: pd.DataFrame) -> str:
        """Format results as a nice markdown/latex table."""
        if df.empty:
            return "No results to display"
        
        lines = []
        lines.append("# Comparison Table Results")
        lines.append("")
        
        for dataset in df['dataset'].unique():
            lines.append(f"## Dataset: {dataset}")
            lines.append("")
            
            # Header
            lines.append("| Method | Model | Accuracy | ROC-AUC | F1-Score | MCC |")
            lines.append("|--------|-------|----------|---------|----------|-----|")
            
            df_dataset = df[df['dataset'] == dataset]
            
            for _, row in df_dataset.iterrows():
                acc = f"{row['accuracy_mean_mean']:.4f}±{row['accuracy_mean_std']:.4f}"
                auc = f"{row['roc_auc_mean_mean']:.4f}±{row['roc_auc_mean_std']:.4f}"
                f1 = f"{row['f1_mean_mean']:.4f}±{row['f1_mean_std']:.4f}"
                mcc = f"{row['mcc_mean_mean']:.4f}±{row['mcc_mean_std']:.4f}"
                
                lines.append(f"| {row['method']} | {row['model']} | {acc} | {auc} | {f1} | {mcc} |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def run(self):
        """Run all experiments."""
        print("="*60)
        print("COMPARISON TABLE EXPERIMENTS")
        print(f"Datasets: {self.config['datasets']}")
        print(f"Trials: {self.config['n_trials']}")
        print(f"Output: {self.output_dir}")
        print("="*60)
        
        start_time = time.time()
        all_results = []
        
        for dataset_name in self.config["datasets"]:
            df = self.run_dataset_experiments(dataset_name)
            all_results.append(df)
        
        # Combine all results
        full_results = pd.concat(all_results, ignore_index=True)
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = self.output_dir / f"raw_results_{timestamp}.csv"
        full_results.to_csv(raw_path, index=False)
        print(f"\nRaw results saved to: {raw_path}")
        
        # Aggregate results
        aggregated = self.aggregate_results(full_results)
        agg_path = self.output_dir / f"aggregated_results_{timestamp}.csv"
        aggregated.to_csv(agg_path, index=False)
        print(f"Aggregated results saved to: {agg_path}")
        
        # Format and save table
        table_str = self.format_table(aggregated)
        table_path = self.output_dir / f"comparison_table_{timestamp}.md"
        with open(table_path, 'w') as f:
            f.write(table_str)
        print(f"Formatted table saved to: {table_path}")
        
        # Print table
        print("\n" + "="*60)
        print(table_str)
        print("="*60)
        
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        
        return aggregated


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comparison table experiments")
    parser.add_argument("--datasets", nargs="+", default=None,
                       help="Datasets to use (default: electricity, phoneme, kc1)")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials per experiment")
    parser.add_argument("--output", type=str, default="experiments/comparison_table",
                       help="Output directory")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="HuggingFace model to use (e.g., Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--llm-provider", type=str, default="huggingface",
                       help="LLM provider (huggingface, openai, anthropic)")
    parser.add_argument("--skip-llm", action="store_true",
                       help="Skip LLM-FE experiments (useful for testing baseline/featuretools only)")
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    if args.datasets:
        config["datasets"] = args.datasets
    config["n_trials"] = args.trials
    config["output_dir"] = args.output
    config["llm_model"] = args.llm_model
    config["llm_provider"] = args.llm_provider
    config["skip_llm"] = args.skip_llm
    
    # Run experiments
    experiment = ComparisonTableExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
