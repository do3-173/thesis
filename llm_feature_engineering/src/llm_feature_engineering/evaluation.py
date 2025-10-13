"""
Evaluation Module

Provides evaluation framework for comparing different feature selection methods
using various machine learning models and metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureSelectionEvaluator:
    """
    Evaluates feature selection methods by training ML models on selected features.
    
    Supports both classification and regression tasks with multiple evaluation metrics.
    """
    
    def __init__(self, random_state: int = 42, n_trials: int = 5):
        """
        Initialize evaluator.
        
        Args:
            random_state: Random state for reproducibility
            n_trials: Number of trials for statistical evaluation
        """
        self.random_state = random_state
        self.n_trials = n_trials
        self.results = {}
        
    def is_classification_task(self, y: pd.Series) -> bool:
        """
        Determine if this is a classification task.
        
        Args:
            y: Target variable
            
        Returns:
            True if classification, False if regression
        """
        # Check if dtype is object (string labels)
        if y.dtype == 'object':
            return True
        
        # Check if it's integer with small number of unique values
        if pd.api.types.is_integer_dtype(y):
            n_unique = y.nunique()
            n_samples = len(y)
            # If less than 20 unique values OR less than 5% of samples are unique
            if n_unique <= 20 or (n_unique / n_samples) < 0.05:
                return True
        
        return False
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    selected_features: List[str]) -> Tuple[pd.DataFrame, pd.Series, bool]:
        """
        Prepare data for evaluation.
        
        Args:
            X: Feature matrix
            y: Target variable
            selected_features: List of selected feature names
            
        Returns:
            Tuple of (selected_X, processed_y, is_classification)
        """
        if not selected_features:
            raise ValueError("No features selected for evaluation")
            
        # Select features
        x_selected = X[selected_features]
        
        # Determine task type
        is_classification = self.is_classification_task(y)
        
        # Process target variable for classification
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y_processed = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        else:
            y_processed = y.copy()
            
        return x_selected, y_processed, is_classification
    
    def get_models(self, is_classification: bool) -> Dict[str, Any]:
        """
        Get appropriate models for the task type.
        
        Args:
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary of model instances
        """
        if is_classification:
            return {
                'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                'mlp': MLPClassifier(random_state=self.random_state, max_iter=500, hidden_layer_sizes=(100,))
            }
        else:
            return {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=self.random_state, n_estimators=100),
                'mlp': MLPRegressor(random_state=self.random_state, max_iter=500, hidden_layer_sizes=(100,))
            }
    
    def get_cv_splitter(self, is_classification: bool, n_splits: int = 5):
        """
        Get appropriate cross-validation splitter.
        
        Args:
            is_classification: Whether this is a classification task
            n_splits: Number of CV folds
            
        Returns:
            CV splitter instance
        """
        if is_classification:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        else:
            return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
    
    def get_scoring_metric(self, is_classification: bool, task_info: Optional[Dict] = None) -> str:
        """
        Get appropriate scoring metric for the task.
        
        Args:
            is_classification: Whether this is a classification task
            task_info: Additional task information
            
        Returns:
            Scoring metric name
        """
        if is_classification:
            # Check if binary classification and AUC is preferred
            if task_info and task_info.get('metric') == 'auc':
                return 'roc_auc'
            return 'accuracy'
        else:
            return 'r2'
    
    def evaluate_single_trial(self, method_name: str, selected_features: List[str], 
                            X: pd.DataFrame, y: pd.Series, 
                            task_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a feature selection method on a single trial.
        
        Args:
            method_name: Name of the feature selection method
            selected_features: List of selected feature names
            X: Feature matrix
            y: Target variable
            task_info: Additional task information
            
        Returns:
            Dictionary with evaluation results
        """
        if not selected_features:
            return {'error': 'No features selected'}
            
        # Prepare data
        x_selected, y_processed, is_classification = self.prepare_data(X, y, selected_features)
        
        # Get models and evaluation setup
        models = self.get_models(is_classification)
        cv_splitter = self.get_cv_splitter(is_classification)
        scoring = self.get_scoring_metric(is_classification, task_info)
        
        # Evaluate each model
        model_results = {}
        for model_name, model in models.items():
            try:
                # Standardize features for neural networks
                if 'mlp' in model_name:
                    scaler = StandardScaler()
                    x_scaled = pd.DataFrame(
                        scaler.fit_transform(x_selected), 
                        columns=x_selected.columns, 
                        index=x_selected.index
                    )
                    scores = cross_val_score(model, x_scaled, y_processed, cv=cv_splitter, scoring=scoring)
                else:
                    scores = cross_val_score(model, x_selected, y_processed, cv=cv_splitter, scoring=scoring)
                
                model_results[model_name] = {
                    'scores': scores.tolist(),
                    'mean_score': scores.mean(),
                    'std_score': scores.std()
                }
            except Exception as e:
                model_results[model_name] = {'error': str(e)}
        
        return {
            'method': method_name,
            'num_features': len(selected_features),
            'selected_features': selected_features,
            'task_type': 'classification' if is_classification else 'regression',
            'metric': scoring,
            'model_results': model_results
        }
    
    def evaluate_method_multiple_trials(self, method_name: str, selected_features: List[str],
                                      X: pd.DataFrame, y: pd.Series,
                                      task_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a feature selection method over multiple trials.
        
        Args:
            method_name: Name of the feature selection method
            selected_features: List of selected feature names
            X: Feature matrix
            y: Target variable
            task_info: Additional task information
            
        Returns:
            Dictionary with aggregated results across trials
        """
        all_results = []
        
        for trial in range(self.n_trials):
            # Use different random states for each trial
            original_state = self.random_state
            self.random_state = original_state + trial
            
            result = self.evaluate_single_trial(method_name, selected_features, X, y, task_info)
            if 'error' not in result:
                all_results.append(result)
            
            # Reset random state
            self.random_state = original_state
        
        if not all_results:
            return {'error': 'All trials failed'}
        
        # Aggregate results across trials
        aggregated_results = {
            'method': method_name,
            'num_features': all_results[0]['num_features'],
            'selected_features': selected_features,
            'task_type': all_results[0]['task_type'],
            'metric': all_results[0]['metric'],
            'n_trials': len(all_results),
            'model_aggregates': {}
        }
        
        # Aggregate results for each model
        model_names = all_results[0]['model_results'].keys()
        for model_name in model_names:
            model_scores = []
            for result in all_results:
                if model_name in result['model_results'] and 'mean_score' in result['model_results'][model_name]:
                    model_scores.append(result['model_results'][model_name]['mean_score'])
            
            if model_scores:
                aggregated_results['model_aggregates'][model_name] = {
                    'scores': model_scores,
                    'mean_score': np.mean(model_scores),
                    'std_score': np.std(model_scores),
                    'min_score': np.min(model_scores),
                    'max_score': np.max(model_scores)
                }
        
        return aggregated_results
    
    def evaluate_method(self, method_name: str, selected_features: List[str],
                       X: pd.DataFrame, y: pd.Series,
                       task_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a feature selection method.
        
        Args:
            method_name: Name of the feature selection method
            selected_features: List of selected feature names
            X: Feature matrix
            y: Target variable
            task_info: Additional task information
            
        Returns:
            Dictionary with evaluation results
        """
        if self.n_trials > 1:
            result = self.evaluate_method_multiple_trials(method_name, selected_features, X, y, task_info)
        else:
            result = self.evaluate_single_trial(method_name, selected_features, X, y, task_info)
        
        self.results[method_name] = result
        return result
    
    def compare_methods(self, primary_model: str = 'random_forest') -> pd.DataFrame:
        """
        Create comparison table of all evaluated methods.
        
        Args:
            primary_model: Primary model to use for comparison
            
        Returns:
            DataFrame with method comparison
        """
        comparison_data = []
        
        for method_name, result in self.results.items():
            if 'error' not in result:
                if self.n_trials > 1 and 'model_aggregates' in result:
                    # Multi-trial results
                    if primary_model in result['model_aggregates']:
                        model_result = result['model_aggregates'][primary_model]
                        comparison_data.append({
                            'Method': method_name,
                            'Num Features': result['num_features'],
                            'Mean Score': model_result['mean_score'],
                            'Std Score': model_result['std_score'],
                            'Min Score': model_result['min_score'],
                            'Max Score': model_result['max_score'],
                            'Metric': result['metric'],
                            'Trials': result['n_trials']
                        })
                else:
                    # Single trial results
                    if primary_model in result['model_results']:
                        model_result = result['model_results'][primary_model]
                        if 'mean_score' in model_result:
                            comparison_data.append({
                                'Method': method_name,
                                'Num Features': result['num_features'],
                                'Mean Score': model_result['mean_score'],
                                'Std Score': model_result['std_score'],
                                'Metric': result['metric'],
                                'Trials': 1
                            })
                
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values('Mean Score', ascending=False)
            
        return df
    
    def plot_comparison(self, primary_model: str = 'random_forest', figsize: Tuple[int, int] = (15, 8)):
        """
        Plot comparison of feature selection methods.
        
        Args:
            primary_model: Primary model to use for comparison
            figsize: Figure size for plots
        """
        df = self.compare_methods(primary_model)
        
        if df.empty:
            print("No results to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Selection Method Comparison', fontsize=16, fontweight='bold')
        
        # Performance comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(df['Method'], df['Mean Score'], yerr=df['Std Score'], capsize=5)
        ax1.set_title(f'Performance Comparison ({df["Metric"].iloc[0]})')
        ax1.set_ylabel(f'Score ({df["Metric"].iloc[0]})')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score, std in zip(bars, df['Mean Score'], df['Std Score']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Number of features
        ax2 = axes[0, 1]  
        ax2.bar(df['Method'], df['Num Features'])
        ax2.set_title('Number of Selected Features')
        ax2.set_ylabel('Number of Features')
        ax2.tick_params(axis='x', rotation=45)
        
        # Score distribution (if multiple trials)
        ax3 = axes[1, 0]
        if 'Min Score' in df.columns:
            # Box plot style representation
            for i, (method, row) in enumerate(df.iterrows()):
                ax3.errorbar(i, row['Mean Score'], 
                           yerr=[[row['Mean Score'] - row['Min Score']], 
                                [row['Max Score'] - row['Mean Score']]],
                           fmt='o', capsize=5, capthick=2)
            ax3.set_xticks(range(len(df)))
            ax3.set_xticklabels(df['Method'], rotation=45)
            ax3.set_title('Score Range Across Trials')
            ax3.set_ylabel(f'Score ({df["Metric"].iloc[0]})')
        else:
            ax3.text(0.5, 0.5, 'Single trial results\nNo range data', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Score Range')
        
        # Model comparison for best method
        ax4 = axes[1, 1]
        if df.empty:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
        else:
            best_method = df.iloc[0]['Method']
            if best_method in self.results:
                result = self.results[best_method]
                if self.n_trials > 1 and 'model_aggregates' in result:
                    model_scores = {model: data['mean_score'] 
                                  for model, data in result['model_aggregates'].items()}
                else:
                    model_scores = {model: data['mean_score'] 
                                  for model, data in result['model_results'].items()
                                  if 'mean_score' in data}
                
                if model_scores:
                    models = list(model_scores.keys())
                    scores = list(model_scores.values())
                    ax4.bar(models, scores)
                    ax4.set_title(f'Model Comparison - {best_method}')
                    ax4.set_ylabel(f'Score ({df["Metric"].iloc[0]})')
                    ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature selection results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {'error': 'No results available'}
        
        summary = {
            'n_methods': len(self.results),
            'methods': list(self.results.keys()),
            'n_trials': self.n_trials,
        }
        
        # Get best performing method
        df = self.compare_methods()
        if not df.empty:
            best_method = df.iloc[0]
            summary['best_method'] = {
                'name': best_method['Method'],
                'score': best_method['Mean Score'],
                'std': best_method['Std Score'],
                'num_features': best_method['Num Features']
            }
        
        return summary