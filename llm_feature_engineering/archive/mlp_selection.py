"""
MLP Feature Selection Integration

Simplified MLP-based feature selection using scikit-learn MLPClassifier/MLPRegressor
This avoids PyTorch dependency while providing neural network-based feature selection.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score


class MLPFeatureSelector:
    """MLP-based feature selection using scikit-learn neural networks"""
    
    def __init__(self, method: str = 'weight_magnitude', **mlp_params):
        """
        Initialize MLP Feature Selector
        
        Args:
            method: 'weight_magnitude' or 'permutation'
            **mlp_params: Parameters for MLPClassifier/MLPRegressor
        """
        self.method = method
        self.mlp_params = {
            'hidden_layer_sizes': (64, 32),
            'max_iter': 200,
            'random_state': 42,
            **mlp_params
        }
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                       task_type: str = 'classification', top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Select features using MLP-based methods
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            task_type: 'classification' or 'regression'
            top_k: Number of top features to select (None for all)
            
        Returns:
            Dictionary with selected features and importance scores
        """
        start_time = time.time()
        
        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and train MLP
        if task_type == 'classification':
            mlp = MLPClassifier(**self.mlp_params)
        else:
            mlp = MLPRegressor(**self.mlp_params)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42,
            stratify=y if task_type == 'classification' else None
        )
        
        # Train model
        mlp.fit(X_train, y_train)
        
        # Calculate feature importance
        if self.method == 'weight_magnitude':
            importance_scores = self._weight_magnitude_importance(mlp)
        elif self.method == 'permutation':
            importance_scores = self._permutation_importance(mlp, X_test, y_test)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Normalize scores to [0, 1]
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        
        # Select top features
        if top_k is None:
            top_k = len(feature_names)
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        selected_features = [feature_names[i] for i in sorted_indices[:top_k]]
        feature_scores = [importance_scores[i] for i in sorted_indices[:top_k]]
        
        # Evaluate model performance
        y_pred = mlp.predict(X_test)
        if task_type == 'classification':
            performance_score = accuracy_score(y_test, y_pred)
            performance_metric = 'accuracy'
        else:
            performance_score = r2_score(y_test, y_pred)
            performance_metric = 'r2_score'
        
        execution_time = time.time() - start_time
        
        return {
            'selected_feature_names': selected_features,
            'selection_time': execution_time,
            'features': [
                {
                    'feature': name,
                    'importance_score': score,
                    'reasoning': f'MLP {self.method} importance: {score:.4f}'
                }
                for name, score in zip(selected_features, feature_scores)
            ],
            'evaluation': {
                'method': f'mlp_{self.method}',
                'performance_score': performance_score,
                'performance_metric': performance_metric,
                'model_params': self.mlp_params
            }
        }
    
    def _weight_magnitude_importance(self, mlp) -> np.ndarray:
        """Calculate importance based on first layer weight magnitudes"""
        # Get first layer weights (input to first hidden layer)
        first_layer_weights = mlp.coefs_[0]  # Shape: (n_features, n_hidden_units)
        
        # Calculate L2 norm for each input feature
        importance = np.linalg.norm(first_layer_weights, axis=1)
        
        return importance
    
    def _permutation_importance(self, mlp, X_test, y_test) -> np.ndarray:
        """Calculate importance using permutation importance"""
        perm_importance = permutation_importance(
            mlp, X_test, y_test, 
            n_repeats=5, 
            random_state=self.mlp_params.get('random_state', 42)
        )
        
        return perm_importance.importances_mean


def create_mlp_feature_selector(method_config: Dict[str, Any]) -> MLPFeatureSelector:
    """
    Factory function to create MLP feature selector from configuration
    
    Args:
        method_config: Configuration dictionary
        
    Returns:
        MLPFeatureSelector instance
    """
    method_type = method_config.get('method_type', 'mlp_weight_magnitude')
    method = method_type.replace('mlp_', '')
    
    # Extract MLP parameters
    mlp_params = {k: v for k, v in method_config.items() 
                  if k not in ['method_type', 'description']}
    
    return MLPFeatureSelector(method=method, **mlp_params)


# Convenience functions for integration
def mlp_weight_feature_selection(X, y, feature_names, task_type='classification', **kwargs):
    """MLP weight magnitude feature selection"""
    selector = MLPFeatureSelector(method='weight_magnitude', **kwargs)
    return selector.select_features(X, y, feature_names, task_type)


def mlp_permutation_feature_selection(X, y, feature_names, task_type='classification', **kwargs):
    """MLP permutation importance feature selection"""
    selector = MLPFeatureSelector(method='permutation', **kwargs)
    return selector.select_features(X, y, feature_names, task_type)


if __name__ == "__main__":
    # Test the MLP feature selection
    from sklearn.datasets import make_classification
    
    print("Testing MLP Feature Selection...")
    
    # Generate test data
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Test weight magnitude method
    print("\n--- Weight Magnitude Method ---")
    results = mlp_weight_feature_selection(X, y, feature_names)
    print(f"Execution time: {results['selection_time']:.3f}s")
    print("Top 5 features:")
    for feat in results['features'][:5]:
        print(f"  {feat['feature']}: {feat['importance_score']:.4f}")
    
    # Test permutation method
    print("\n--- Permutation Method ---")
    results = mlp_permutation_feature_selection(X, y, feature_names)
    print(f"Execution time: {results['selection_time']:.3f}s")
    print("Top 5 features:")
    for feat in results['features'][:5]:
        print(f"  {feat['feature']}: {feat['importance_score']:.4f}")