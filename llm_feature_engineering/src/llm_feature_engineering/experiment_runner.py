"""
Experiment Runner Module

Main orchestrator for running feature engineering experiments with Hydra configuration.
Handles the complete experimental pipeline from data loading to result reporting.
"""

import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from .dataset_manager import DatasetManager
from .llm_interface import create_llm_interface
from .feature_selection import create_feature_selector
from .evaluation import FeatureSelectionEvaluator
from .autogluon_benchmark import AutoGluonBenchmark


class ExperimentRunner:
    """
    Main experiment runner for LLM-based feature engineering experiments.
    
    Orchestrates the complete experimental pipeline including data loading,
    feature selection, evaluation, and result reporting.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize experiment runner with Hydra configuration.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.results_dir = Path(config.experiment.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        

        self.dataset_manager = DatasetManager(config.dataset.dataset_dir)
        self.llm_interface = None
        self.experiment_results = {}
        

        if hasattr(config.experiment, 'random_seed'):
            np.random.seed(config.experiment.random_seed)
    
    def setup_llm_interface(self):
        """Setup LLM interface based on configuration."""
        if self.config.llm.enabled:
            try:
                self.llm_interface = create_llm_interface(
                    provider=self.config.llm.provider,
                    model=self.config.llm.model
                )
                

                if self.llm_interface.test_connection():
                    print(f"LLM interface initialized successfully: {self.config.llm.provider}")
                else:
                    print("LLM interface test failed")
                    self.llm_interface = None
                    
            except Exception as e:
                print(f"Failed to initialize LLM interface: {e}")
                self.llm_interface = None
    
    def get_datasets_to_process(self) -> List[str]:
        """
        Get list of datasets to process based on configuration.
        
        Returns:
            List of dataset names
        """
        from omegaconf import ListConfig
        
        if self.config.dataset.datasets == "all":
            return self.dataset_manager.list_datasets(self.config.dataset.source)
        elif isinstance(self.config.dataset.datasets, (list, ListConfig)):
            return list(self.config.dataset.datasets)  # Convert ListConfig to regular list
        else:
            return [self.config.dataset.datasets]
    
    def create_feature_selectors(self) -> Dict[str, Any]:
        """
        Create feature selector instances based on configuration.
        
        Returns:
            Dictionary of feature selector instances
        """
        selectors = {}
        

        if self.config.feature_engineering.text_based.enabled and self.llm_interface:
            selectors['text_based'] = create_feature_selector(
                'text_based', 
                llm_interface=self.llm_interface
            )
        
        if self.config.feature_engineering.llm4fs.enabled and self.llm_interface:
            selectors['llm4fs'] = create_feature_selector(
                'llm4fs',
                llm_interface=self.llm_interface
            )
        
        if hasattr(self.config.methods, 'caafe') and self.config.feature_engineering.caafe.enabled and self.llm_interface:
            selectors['caafe'] = create_feature_selector(
                'caafe',
                llm_interface=self.llm_interface,
                **OmegaConf.to_container(self.config.feature_engineering.caafe, resolve=True)
            )
        

        if self.config.feature_engineering.traditional.mutual_info.enabled:
            selectors['mutual_info'] = create_feature_selector(
                'traditional',
                method='mutual_info'
            )
        
        if self.config.feature_engineering.traditional.random_forest.enabled:
            selectors['random_forest'] = create_feature_selector(
                'traditional',
                method='random_forest'
            )
        

        if self.config.feature_engineering.mlp_weights.enabled:
            selectors['mlp_weights'] = create_feature_selector(
                'mlp_weights',
                **OmegaConf.to_container(self.config.feature_engineering.mlp_weights, resolve=True)
            )
        
        if self.config.feature_engineering.mlp_permutation.enabled:
            selectors['mlp_permutation'] = create_feature_selector(
                'mlp_permutation',
                **OmegaConf.to_container(self.config.feature_engineering.mlp_permutation, resolve=True)
            )
        
        return selectors
    
    def run_feature_selection_experiment(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run feature selection experiment on a single dataset.
        
        Args:
            dataset_name: Name of the dataset to process
            
        Returns:
            Dictionary with experiment results
        """
        print(f"Processing dataset: {dataset_name}")
        print("-" * 50)
        
        try:

            ml_data = self.dataset_manager.prepare_for_ml(
                dataset_name, 
                source=self.config.dataset.source,
                test_size=self.config.dataset.test_size,
                random_state=self.config.experiment.random_seed
            )
            

            dataset_info = self.dataset_manager.get_dataset_info(
                dataset_name, 
                source=self.config.dataset.source
            )
            
            print(f"Dataset shape: {ml_data['X_train'].shape}")
            print(f"Task type: {dataset_info.get('metadata', {}).get('problem_type', 'Unknown')}")
            

            selectors = self.create_feature_selectors()
            
            if not selectors:
                print("No feature selectors available")
                return {'error': 'No feature selectors available'}
            

            evaluator = FeatureSelectionEvaluator(
                random_state=self.config.experiment.random_seed,
                n_trials=self.config.evaluation.n_trials
            )
            

            selection_results = {}
            
            for method_name, selector in selectors.items():
                print(f"Running {method_name} feature selection...")
                
                try:
                    start_time = time.time()
                    

                    if method_name in ['text_based', 'llm4fs', 'caafe']:

                        features = selector.select_features(
                            ml_data['X_train'], 
                            ml_data['y_train'],
                            dataset_info=dataset_info,
                            top_k=self.config.methods[method_name].top_k
                        )
                    elif method_name in ['mlp_weights', 'mlp_permutation']:

                        features = selector.select_features(
                            ml_data['X_train'], 
                            ml_data['y_train'],
                            top_k=self.config.methods[method_name].top_k
                        )
                    else:

                        features = selector.select_features(
                            ml_data['X_train'], 
                            ml_data['y_train'],
                            top_k=self.config.feature_engineering.traditional[method_name].top_k
                        )
                    
                    selection_time = time.time() - start_time
                    
                    if features:

                        selected_features = selector.get_feature_names(features)
                        
                        print(f"Selected {len(selected_features)} features in {selection_time:.2f}s")
                        print(f"Features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
                        

                        eval_result = evaluator.evaluate_method(
                            method_name,
                            selected_features,
                            ml_data['X_train'],
                            ml_data['y_train'],
                            task_info=dataset_info.get('metadata', {})
                        )
                        
                        selection_results[method_name] = {
                            'features': features,
                            'selected_feature_names': selected_features,
                            'selection_time': selection_time,
                            'evaluation': eval_result
                        }
                        
                    else:
                        print(f"No features selected by {method_name}")
                        selection_results[method_name] = {'error': 'No features selected'}
                        
                except Exception as e:
                    print(f"Error in {method_name}: {e}")
                    selection_results[method_name] = {'error': str(e)}
            

            comparison_df = evaluator.compare_methods()
            
            return {
                'dataset_name': dataset_name,
                'dataset_info': dataset_info,
                'ml_data_shapes': {
                    'X_train': ml_data['X_train'].shape,
                    'X_test': ml_data['X_test'].shape,
                    'y_train': ml_data['y_train'].shape,
                    'y_test': ml_data['y_test'].shape
                },
                'selection_results': selection_results,
                'evaluation_comparison': comparison_df.to_dict() if not comparison_df.empty else {},
                'evaluator': evaluator
            }
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            return {'error': str(e)}
    
    def run_autogluon_benchmark(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run AutoGluon benchmark on a single dataset.
        
        Args:
            dataset_name: Name of the dataset to process
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.config.benchmark.autogluon.enabled:
            return {'skipped': 'AutoGluon benchmark disabled'}
        
        print(f"Running AutoGluon benchmark on: {dataset_name}")
        
        try:

            ml_data = self.dataset_manager.prepare_for_ml(
                dataset_name,
                source=self.config.dataset.source,
                test_size=self.config.dataset.test_size,
                random_state=self.config.experiment.random_seed
            )
            

            train_df = ml_data['X_train'].copy()
            train_df[ml_data['target_name']] = ml_data['y_train']
            
            test_df = ml_data['X_test'].copy()
            test_df[ml_data['target_name']] = ml_data['y_test']
            

            dataset_info = {
                'label_columns': [ml_data['target_name']],
                'feature_columns': ml_data['feature_names'],
                'problem_type': 'classification',  # Default assumption
                'metric': 'accuracy',
                'feature_types': ['numeric'] * len(ml_data['feature_names'])  # Simplified
            }
            

            benchmark = AutoGluonBenchmark(
                time_limit=self.config.benchmark.autogluon.time_limit,
                preset=self.config.benchmark.autogluon.preset
            )
            
            results = benchmark.run_full_benchmark(
                train_data=train_df,
                test_data=test_df,
                dataset_info=dataset_info
            )
            
            return {
                'dataset_name': dataset_name,
                'benchmark_results': results
            }
            
        except Exception as e:
            print(f"Error in AutoGluon benchmark for {dataset_name}: {e}")
            return {'error': str(e)}
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """
        Save experiment results to file.
        
        Args:
            results: Results dictionary to save
            filename: Name of the output file
        """

        pickle_path = self.results_dir / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        

        summary = self.create_results_summary(results)
        json_path = self.results_dir / f"{filename}_summary.json"
        
        import json
        from omegaconf import OmegaConf
        

        def convert_omegaconf(obj):
            if isinstance(obj, dict):
                return {str(k): convert_omegaconf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_omegaconf(item) for item in obj]
            elif hasattr(obj, '_content'):  # OmegaConf objects
                return OmegaConf.to_container(obj, resolve=True)
            else:
                return obj
        
        serializable_summary = convert_omegaconf(summary)
        with open(json_path, 'w') as f:
            json.dump(serializable_summary, f, indent=2, default=str)
        
        print(f"Results saved to: {pickle_path}")
        print(f"Summary saved to: {json_path}")
    
    def create_results_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of experiment results.
        
        Args:
            results: Complete results dictionary
            
        Returns:
            Summary dictionary
        """
        summary = {
            'experiment_config': OmegaConf.to_container(self.config),
            'datasets_processed': len(results),
            'methods_used': [],
            'dataset_summaries': {}
        }
        
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                summary['dataset_summaries'][dataset_name] = {'error': dataset_results['error']}
                continue
            

            if 'selection_results' in dataset_results:
                methods = list(dataset_results['selection_results'].keys())
                summary['methods_used'].extend(methods)
                

                comparison = dataset_results.get('evaluation_comparison', {})
                if comparison and 'Method' in comparison and 'Mean Score' in comparison:
                    best_method = None
                    best_score = -1
                    
                    # comparison structure: {'Method': {0: 'method1', 1: 'method2'}, 'Mean Score': {0: 0.8, 1: 0.7}}
                    methods = comparison['Method']
                    scores = comparison['Mean Score']
                    
                    for idx in methods.keys():
                        method = methods[idx]
                        score = scores[idx]
                        if score > best_score:
                            best_score = score
                            best_method = method
                    
                    summary['dataset_summaries'][dataset_name] = {
                        'methods_tested': methods,
                        'best_method': best_method,
                        'best_score': best_score,
                        'n_features_original': dataset_results.get('ml_data_shapes', {}).get('X_train', [0, 0])[1]
                    }
        

        summary['methods_used'] = list(set(summary['methods_used']))
        
        return summary
    
    def run_experiments(self):
        """Run all configured experiments."""
        print("Starting LLM Feature Engineering Experiments")
        print("=" * 60)
        

        self.setup_llm_interface()
        

        datasets = self.get_datasets_to_process()
        print(f"Processing {len(datasets)} datasets: {datasets}")
        

        if self.config.experiment.run_feature_selection:
            print("\nRunning Feature Selection Experiments")
            print("-" * 40)
            
            for dataset_name in datasets:
                result = self.run_feature_selection_experiment(dataset_name)
                self.experiment_results[dataset_name] = result
                

                if self.config.experiment.save_intermediate:
                    timestamp = int(time.time())
                    self.save_results(
                        {dataset_name: result}, 
                        f"intermediate_{dataset_name}_{timestamp}"
                    )
        

        if self.config.experiment.run_autogluon_benchmark:
            print("\nRunning AutoGluon Benchmarks")
            print("-" * 40)
            
            benchmark_results = {}
            for dataset_name in datasets:
                result = self.run_autogluon_benchmark(dataset_name)
                benchmark_results[dataset_name] = result
            

            for dataset_name in benchmark_results:
                if dataset_name in self.experiment_results:
                    self.experiment_results[dataset_name]['autogluon_benchmark'] = benchmark_results[dataset_name]
                else:
                    self.experiment_results[dataset_name] = {'autogluon_benchmark': benchmark_results[dataset_name]}
        

        timestamp = int(time.time())
        self.save_results(self.experiment_results, f"final_results_{timestamp}")
        
        print("\nExperiments completed!")
        print(f"Results saved in: {self.results_dir}")
        
        return self.experiment_results


@hydra.main(version_base=None, config_path="../../config", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments with Hydra.
    
    Args:
        cfg: Hydra configuration
    """
    runner = ExperimentRunner(cfg)
    results = runner.run_experiments()
    

    summary = runner.create_results_summary(results)
    print("\nExperiment Summary:")
    print("-" * 20)
    print(f"Datasets processed: {summary['datasets_processed']}")
    print(f"Methods used: {summary['methods_used']}")
    
    return results


if __name__ == "__main__":
    main()