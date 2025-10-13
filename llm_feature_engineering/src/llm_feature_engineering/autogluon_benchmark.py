"""
AutoGluon Benchmark Module

Implements tabular learning benchmarks using AutoGluon with various configurations
including stack ensembling and text feature processing.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon not available. Benchmark functionality will be limited.")

try:
    from auto_mm_bench.datasets import dataset_registry, _TEXT
    AUTO_MM_BENCH_AVAILABLE = True
except ImportError:
    AUTO_MM_BENCH_AVAILABLE = False
    print("auto_mm_bench not available. Using alternative dataset loading.")


class AutoGluonBenchmark:
    """
    Benchmark comparing AutoGluon tabular capabilities with traditional approaches.
    
    Implements multiple methods:
    - AG-Stack (no text): AutoGluon ignoring text columns
    - AG-Stack + N-Gram: AutoGluon with N-Gram text featurization  
    - Random Forest + TF-IDF: Traditional approach with manual feature engineering
    """
    
    def __init__(self, time_limit: int = 600, preset: str = 'medium_quality'):
        """
        Initialize AutoGluon benchmark.
        
        Args:
            time_limit: Time limit in seconds for AutoGluon training
            preset: AutoGluon preset ('best_quality', 'high_quality', 'medium_quality')
        """
        self.time_limit = time_limit
        self.preset = preset
        self.results = {}
        
        if not AUTOGLUON_AVAILABLE:
            print("Warning: AutoGluon not available. Some benchmarks will be skipped.")
    
    def convert_problem_type(self, problem_type: str, y_data: pd.Series) -> str:
        """
        Convert problem type to AutoGluon format.
        
        Args:
            problem_type: Original problem type
            y_data: Target data to infer type from
            
        Returns:
            AutoGluon compatible problem type
        """
        if problem_type in ['binary', 'multiclass', 'regression', 'quantile']:
            return problem_type
        
        if problem_type == 'classification':
            # Determine if binary or multiclass
            n_unique = y_data.nunique()
            return 'binary' if n_unique == 2 else 'multiclass'
        
        # Auto-detect from target data
        if y_data.dtype in ['int64', 'float64'] and y_data.nunique() > 20:
            return 'regression'
        else:
            n_unique = y_data.nunique()
            return 'binary' if n_unique == 2 else 'multiclass'

    def prepare_data_from_registry(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """
        Load and prepare train/test data from auto_mm_bench registry.
        
        Args:
            dataset_name: Name of the dataset in the registry
            
        Returns:
            Tuple of (train_data, test_data, train_dataset_info)
        """
        if not AUTO_MM_BENCH_AVAILABLE:
            raise ImportError("auto_mm_bench not available")
            
        train_dataset = dataset_registry.create(dataset_name, 'train')
        test_dataset = dataset_registry.create(dataset_name, 'test')
        
        train_data = train_dataset.data.copy()
        test_data = test_dataset.data.copy()
        
        return train_data, test_data, train_dataset
    
    def prepare_data_from_dataframe(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  target_col: str, problem_type: str = 'auto',
                                  metric: str = 'auto') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Prepare data from DataFrames with minimal dataset info.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame  
            target_col: Name of target column
            problem_type: Type of ML problem
            metric: Evaluation metric
            
        Returns:
            Tuple of (train_data, test_data, dataset_info)
        """
        dataset_info = {
            'label_columns': [target_col],
            'feature_columns': [col for col in train_df.columns if col != target_col],
            'problem_type': problem_type,
            'metric': metric,
            'feature_types': ['text' if train_df[col].dtype == 'object' and col != target_col 
                            else 'numeric' for col in train_df.columns if col != target_col]
        }
        
        return train_df.copy(), test_df.copy(), dataset_info
    
    def identify_text_columns(self, dataset_info: Dict) -> List[str]:
        """
        Identify text columns from dataset information.
        
        Args:
            dataset_info: Dataset information dictionary
            
        Returns:
            List of text column names
        """
        if hasattr(dataset_info, 'feature_types') and hasattr(dataset_info, 'feature_columns'):
            # auto_mm_bench dataset
            return [col for col, ftype in zip(dataset_info.feature_columns, dataset_info.feature_types) 
                   if ftype == _TEXT]
        elif 'feature_types' in dataset_info and 'feature_columns' in dataset_info:
            # Dictionary-based dataset info
            return [col for col, ftype in zip(dataset_info['feature_columns'], dataset_info['feature_types'])
                   if ftype == 'text']
        else:
            # Fallback: assume object columns (except target) are text
            return []
    
    def run_ag_stack_no_text(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                           dataset_info: Any) -> Dict[str, Any]:
        """
        Run AutoGluon stack ensemble ignoring text columns.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            dataset_info: Dataset information
            
        Returns:
            Dictionary with benchmark results
        """
        if not AUTOGLUON_AVAILABLE:
            return {'error': 'AutoGluon not available'}
            
        print("Running AG-Stack (no text)...")
        
        # Get dataset metadata
        if hasattr(dataset_info, 'label_columns'):
            label_col = dataset_info.label_columns[0]
            problem_type = self.convert_problem_type(dataset_info.problem_type, train_data[label_col])
            metric = dataset_info.metric
        else:
            label_col = dataset_info['label_columns'][0]
            problem_type = self.convert_problem_type(dataset_info['problem_type'], train_data[label_col])
            metric = dataset_info['metric']
        
        # Remove text columns
        text_columns = self.identify_text_columns(dataset_info)
        
        train_no_text = train_data.drop(columns=text_columns)
        test_no_text = test_data.drop(columns=text_columns)
        
        try:
            # Train AutoGluon
            predictor = TabularPredictor(
                label=label_col,
                problem_type=problem_type,
                eval_metric=metric
            )
            
            start_time = time.time()
            predictor.fit(
                train_no_text,
                time_limit=self.time_limit,
                presets=self.preset,
                ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'}
            )
            train_time = time.time() - start_time
            
            # Make predictions
            predictions = predictor.predict(test_no_text)
            
            # Calculate metric
            true_labels = test_data[label_col]
            score = self.calculate_metric(true_labels, predictions, metric, problem_type)
            
            return {
                'method': 'AG-Stack (no text)',
                'score': score,
                'metric': metric,
                'train_time': train_time,
                'n_features_used': len(train_no_text.columns) - 1
            }
            
        except Exception as e:
            return {'error': f'AG-Stack (no text) failed: {str(e)}'}
    
    def run_ag_stack_with_ngram(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                              dataset_info: Any) -> Dict[str, Any]:
        """
        Run AutoGluon stack ensemble with N-Gram text featurization.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame
            dataset_info: Dataset information
            
        Returns:
            Dictionary with benchmark results
        """
        if not AUTOGLUON_AVAILABLE:
            return {'error': 'AutoGluon not available'}
            
        print("Running AG-Stack + N-Gram...")
        
        # Get dataset metadata
        if hasattr(dataset_info, 'label_columns'):
            label_col = dataset_info.label_columns[0]
            problem_type = self.convert_problem_type(dataset_info.problem_type, train_data[label_col])
            metric = dataset_info.metric
        else:
            label_col = dataset_info['label_columns'][0]
            problem_type = self.convert_problem_type(dataset_info['problem_type'], train_data[label_col])
            metric = dataset_info['metric']
        
        try:
            # Train AutoGluon with N-Gram featurization (handled automatically)
            predictor = TabularPredictor(
                label=label_col,
                problem_type=problem_type,
                eval_metric=metric
            )
            
            start_time = time.time()
            predictor.fit(
                train_data,
                time_limit=self.time_limit,
                presets=self.preset,
                ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'}
            )
            train_time = time.time() - start_time
            
            # Make predictions
            predictions = predictor.predict(test_data)
            
            # Calculate metric
            true_labels = test_data[label_col]
            score = self.calculate_metric(true_labels, predictions, metric, problem_type)
            
            return {
                'method': 'AG-Stack + N-Gram',
                'score': score,
                'metric': metric,
                'train_time': train_time,
                'n_features_used': len(train_data.columns) - 1
            }
            
        except Exception as e:
            return {'error': f'AG-Stack + N-Gram failed: {str(e)}'}
    
    def run_random_forest_tfidf(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                              dataset_info: Any) -> Dict[str, Any]:
        """
        Run Random Forest with TF-IDF text features and basic feature engineering.
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame  
            dataset_info: Dataset information
            
        Returns:
            Dictionary with benchmark results
        """
        print("Running Random Forest + TF-IDF...")
        
        # Get dataset metadata
        if hasattr(dataset_info, 'label_columns'):
            label_col = dataset_info.label_columns[0]
            problem_type = self.convert_problem_type(dataset_info.problem_type, train_data[label_col])
            metric = dataset_info.metric
            feature_columns = dataset_info.feature_columns
        else:
            label_col = dataset_info['label_columns'][0]
            problem_type = self.convert_problem_type(dataset_info['problem_type'], train_data[label_col])
            metric = dataset_info['metric']
            feature_columns = dataset_info['feature_columns']
        
        # Identify text and non-text columns
        text_columns = self.identify_text_columns(dataset_info)
        numeric_columns = [col for col in feature_columns if col not in text_columns]
        
        try:
            start_time = time.time()
            
            # Prepare text features
            if text_columns:
                # Combine all text columns into one
                train_text = train_data[text_columns].fillna('').apply(
                    lambda x: ' '.join(x.astype(str)), axis=1)
                test_text = test_data[text_columns].fillna('').apply(
                    lambda x: ' '.join(x.astype(str)), axis=1)
                
                # TF-IDF vectorization
                tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                train_tfidf = tfidf.fit_transform(train_text).toarray()
                test_tfidf = tfidf.transform(test_text).toarray()
                
                # Create TF-IDF feature DataFrames
                tfidf_feature_names = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
                train_tfidf_df = pd.DataFrame(train_tfidf, columns=tfidf_feature_names, 
                                            index=train_data.index)
                test_tfidf_df = pd.DataFrame(test_tfidf, columns=tfidf_feature_names, 
                                           index=test_data.index)
            else:
                train_tfidf_df = pd.DataFrame(index=train_data.index)
                test_tfidf_df = pd.DataFrame(index=test_data.index)
            
            # Prepare numeric features with basic feature engineering
            if numeric_columns:
                train_numeric = train_data[numeric_columns].fillna(0)
                test_numeric = test_data[numeric_columns].fillna(0)
                
                # Add interaction features for top numeric columns
                if len(numeric_columns) > 1:
                    top_cols = numeric_columns[:min(3, len(numeric_columns))]
                    for i, col1 in enumerate(top_cols):
                        for col2 in top_cols[i+1:]:
                            if (train_numeric[col1].dtype in ['int64', 'float64'] and 
                                train_numeric[col2].dtype in ['int64', 'float64']):
                                interaction_name = f'{col1}_x_{col2}'
                                train_numeric[interaction_name] = train_numeric[col1] * train_numeric[col2]
                                test_numeric[interaction_name] = test_numeric[col1] * test_numeric[col2]
            else:
                train_numeric = pd.DataFrame(index=train_data.index)
                test_numeric = pd.DataFrame(index=test_data.index)
            
            # Combine all features
            x_train = pd.concat([train_numeric, train_tfidf_df], axis=1)
            x_test = pd.concat([test_numeric, test_tfidf_df], axis=1)
            
            # Prepare labels
            y_train = train_data[label_col]
            y_test = test_data[label_col]
            
            # Train model based on problem type
            if problem_type in ['binary', 'multiclass']:
                # Handle categorical labels
                if y_train.dtype == 'object':
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_test_encoded = le.transform(y_test)
                else:
                    y_train_encoded = y_train
                    y_test_encoded = y_test
                
                # Train Random Forest Classifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(x_train, y_train_encoded)
                
                if problem_type == 'binary' and metric.lower() in ['auc', 'roc_auc']:
                    # For AUC, use probabilities
                    predictions = rf.predict_proba(x_test)[:, 1]
                    true_labels = y_test_encoded
                else:
                    predictions = rf.predict(x_test)
                    true_labels = y_test_encoded
            else:
                # Regression
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(x_train, y_train)
                predictions = rf.predict(x_test)
                true_labels = y_test
            
            train_time = time.time() - start_time
            
            # Calculate metric
            score = self.calculate_metric(true_labels, predictions, metric, problem_type)
            
            return {
                'method': 'Random Forest + TF-IDF',
                'score': score,
                'metric': metric,
                'train_time': train_time,
                'n_features_used': x_train.shape[1]
            }
            
        except Exception as e:
            return {'error': f'Random Forest + TF-IDF failed: {str(e)}'}
    
    def calculate_metric(self, y_true, y_pred, metric_name: str, problem_type: str) -> float:
        """
        Calculate the appropriate metric based on problem type.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or probabilities
            metric_name: Name of the metric
            problem_type: Type of ML problem
            
        Returns:
            Calculated metric score
        """
        try:
            if problem_type == 'binary':
                if metric_name.lower() in ['auc', 'roc_auc']:
                    return roc_auc_score(y_true, y_pred)
                else:
                    return accuracy_score(y_true, y_pred)
            elif problem_type == 'multiclass':
                return accuracy_score(y_true, y_pred)
            elif problem_type == 'regression':
                return r2_score(y_true, y_pred)
            else:
                # Default to accuracy for unknown types
                return accuracy_score(y_true, y_pred)
        except Exception as e:
            print(f"Error calculating metric {metric_name}: {e}")
            return 0.0
    
    def run_full_benchmark(self, dataset_name: Optional[str] = None, 
                         train_data: Optional[pd.DataFrame] = None,
                         test_data: Optional[pd.DataFrame] = None,
                         dataset_info: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Run complete benchmark on a dataset.
        
        Args:
            dataset_name: Name of dataset in registry (if using registry)
            train_data: Training DataFrame (if providing data directly)  
            test_data: Test DataFrame (if providing data directly)
            dataset_info: Dataset information (if providing data directly)
            
        Returns:
            List of benchmark results
        """
        print("Running full tabular benchmark")
        print("=" * 60)
        
        # Load data
        if dataset_name:
            train_df, test_df, ds_info = self.prepare_data_from_registry(dataset_name)
            print(f"Dataset: {dataset_name}")
        elif train_data is not None and test_data is not None and dataset_info is not None:
            train_df, test_df, ds_info = train_data, test_data, dataset_info
            print("Dataset: Custom data")
        else:
            raise ValueError("Either dataset_name or (train_data, test_data, dataset_info) must be provided")
        
        results = []
        
        # Run AG-Stack without text
        try:
            result1 = self.run_ag_stack_no_text(train_df, test_df, ds_info)
            if 'error' not in result1:
                results.append(result1)
                print(f"{result1['method']}: {result1['score']:.4f} {result1['metric']}")
            else:
                print(f"AG-Stack (no text) failed: {result1['error']}")
        except Exception as e:
            print(f"AG-Stack (no text) failed: {e}")
        
        # Run AG-Stack with N-Gram
        try:
            result2 = self.run_ag_stack_with_ngram(train_df, test_df, ds_info)
            if 'error' not in result2:
                results.append(result2)
                print(f"{result2['method']}: {result2['score']:.4f} {result2['metric']}")
            else:
                print(f"AG-Stack + N-Gram failed: {result2['error']}")
        except Exception as e:
            print(f"AG-Stack + N-Gram failed: {e}")
        
        # Run Random Forest with TF-IDF
        try:
            result3 = self.run_random_forest_tfidf(train_df, test_df, ds_info)
            if 'error' not in result3:
                results.append(result3)
                print(f"{result3['method']}: {result3['score']:.4f} {result3['metric']}")
            else:
                print(f"Random Forest + TF-IDF failed: {result3['error']}")
        except Exception as e:
            print(f"Random Forest + TF-IDF failed: {e}")
        
        # Store results
        dataset_key = dataset_name or 'custom'
        self.results[dataset_key] = results
        
        return results