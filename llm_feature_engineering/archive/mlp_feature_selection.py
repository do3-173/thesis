"""
MLP-based Feature Selection Methods

This module implements neural network approaches for feature selection using MLPs.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import time
from typing import List, Tuple, Dict, Any

class MLPFeatureSelector:
    """MLP-based feature selection using different neural network approaches"""
    
    def __init__(self, hidden_size: int = 64, epochs: int = 100, lr: float = 0.001, 
                 method: str = 'weight_magnitude', dropout: float = 0.2):
        """
        Initialize MLP Feature Selector
        
        Args:
            hidden_size: Size of hidden layers
            epochs: Number of training epochs
            lr: Learning rate
            method: Feature selection method ('weight_magnitude', 'gradient_based', 'attention_weights')
            dropout: Dropout probability
        """
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.method = method
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_mlp(self, input_dim: int, output_dim: int, task_type: str = 'classification'):
        """Create MLP model based on the method"""
        
        if self.method == 'attention_weights':
            return AttentionMLP(input_dim, self.hidden_size, output_dim, self.dropout, task_type)
        else:
            return StandardMLP(input_dim, self.hidden_size, output_dim, self.dropout, task_type)
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                       task_type: str = 'classification', top_k: int = None) -> Dict[str, Any]:
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
        X_scaled = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_train).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device) if task_type == 'classification' else torch.FloatTensor(y_test).to(self.device)
        
        # Create model
        output_dim = len(np.unique(y)) if task_type == 'classification' else 1
        model = self._create_mlp(X.shape[1], output_dim, task_type).to(self.device)
        
        # Train model
        self._train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, task_type)
        
        # Extract feature importance
        feature_importance = self._extract_importance(model, X_train_tensor, feature_names)
        
        # Select top features
        if top_k is None:
            top_k = len(feature_names)
        
        # Sort features by importance
        sorted_features = sorted(
            [(name, score) for name, score in zip(feature_names, feature_importance)],
            key=lambda x: x[1], reverse=True
        )
        
        selected_features = [name for name, _ in sorted_features[:top_k]]
        feature_scores = [score for _, score in sorted_features[:top_k]]
        
        execution_time = time.time() - start_time
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'feature_ranking': sorted_features,
            'execution_time': execution_time,
            'method': f'mlp_{self.method}',
            'model_performance': self._evaluate_model(model, X_test_tensor, y_test_tensor, task_type)
        }
    
    def _train_model(self, model, X_train, y_train, X_test, y_test, task_type):
        """Train the MLP model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            
            if task_type == 'classification':
                loss = criterion(outputs, y_train)
            else:
                loss = criterion(outputs.squeeze(), y_train)
            
            loss.backward()
            optimizer.step()
            
            # Optional: Print training progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test)
                    if task_type == 'classification':
                        test_loss = criterion(test_outputs, y_test)
                        _, predicted = torch.max(test_outputs.data, 1)
                        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
                        print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}, Test Acc: {accuracy:.4f}')
                    else:
                        test_loss = criterion(test_outputs.squeeze(), y_test)
                        print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
                model.train()
    
    def _extract_importance(self, model, X, feature_names):
        """Extract feature importance based on the selected method"""
        model.eval()
        
        if self.method == 'weight_magnitude':
            return self._weight_magnitude_importance(model)
        elif self.method == 'gradient_based':
            return self._gradient_based_importance(model, X)
        elif self.method == 'attention_weights':
            return self._attention_weights_importance(model, X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _weight_magnitude_importance(self, model):
        """Calculate importance based on first layer weight magnitudes"""
        first_layer = model.layers[0] if hasattr(model, 'layers') else model.fc1
        weights = first_layer.weight.data.cpu().numpy()
        # L2 norm of weights for each input feature
        importance = np.linalg.norm(weights, axis=0)
        # Normalize to [0, 1]
        importance = importance / np.max(importance) if np.max(importance) > 0 else importance
        return importance
    
    def _gradient_based_importance(self, model, X):
        """Calculate importance based on gradients"""
        model.eval()
        X.requires_grad_(True)
        
        outputs = model(X)
        # For classification, use mean prediction; for regression, use mean output
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            target_output = outputs.mean(dim=1).sum()
        else:
            target_output = outputs.sum()
        
        target_output.backward()
        
        # Calculate gradient-based importance
        gradients = X.grad.data.cpu().numpy()
        importance = np.mean(np.abs(gradients), axis=0)
        # Normalize to [0, 1]
        importance = importance / np.max(importance) if np.max(importance) > 0 else importance
        
        X.requires_grad_(False)
        return importance
    
    def _attention_weights_importance(self, model, X):
        """Extract attention weights as feature importance"""
        if hasattr(model, 'attention'):
            model.eval()
            with torch.no_grad():
                _ = model(X)  # Forward pass to compute attention
                attention_weights = model.attention.attention_weights.cpu().numpy()
                importance = np.mean(attention_weights, axis=0)
                return importance
        else:
            # Fallback to weight magnitude
            return self._weight_magnitude_importance(model)
    
    def _evaluate_model(self, model, X_test, y_test, task_type):
        """Evaluate model performance"""
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            
            if task_type == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_test).sum().item() / y_test.size(0)
                return {'accuracy': accuracy}
            else:
                y_pred = outputs.squeeze().cpu().numpy()
                y_true = y_test.cpu().numpy()
                r2 = r2_score(y_true, y_pred)
                return {'r2_score': r2}


class StandardMLP(nn.Module):
    """Standard MLP for feature selection"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, task_type):
        super(StandardMLP, self).__init__()
        self.task_type = task_type
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layers[0](x))
        x = self.dropout(x)
        x = self.relu(self.layers[1](x))
        x = self.dropout(x)
        x = self.layers[2](x)
        
        if self.task_type == 'regression':
            return x
        else:
            return x  # Logits for classification


class AttentionMLP(nn.Module):
    """MLP with attention mechanism for feature selection"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, task_type):
        super(AttentionMLP, self).__init__()
        self.task_type = task_type
        self.input_dim = input_dim
        
        # Attention mechanism
        self.attention = FeatureAttention(input_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Apply attention to input features
        attended_x = self.attention(x)
        
        # Forward through MLP
        x = self.relu(self.fc1(attended_x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class FeatureAttention(nn.Module):
    """Feature attention mechanism"""
    
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention_layer = nn.Linear(input_dim, input_dim)
        self.attention_weights = None
        
    def forward(self, x):
        # Calculate attention weights
        attention_scores = self.attention_layer(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        self.attention_weights = attention_weights  # Store for importance extraction
        
        # Apply attention
        attended_features = x * attention_weights
        return attended_features


# Example usage and integration function
def mlp_feature_selection(X, y, feature_names, task_type='classification', method='weight_magnitude', top_k=None):
    """
    Convenient function for MLP-based feature selection
    
    Args:
        X: Feature matrix
        y: Target vector  
        feature_names: List of feature names
        task_type: 'classification' or 'regression'
        method: 'weight_magnitude', 'gradient_based', or 'attention_weights'
        top_k: Number of top features to select
        
    Returns:
        Dictionary with selection results
    """
    selector = MLPFeatureSelector(method=method, epochs=50)
    results = selector.select_features(X, y, feature_names, task_type, top_k)
    
    return {
        'selected_feature_names': results['selected_features'],
        'selection_time': results['execution_time'],
        'features': [
            {
                'feature': name,
                'importance_score': score,
                'reasoning': f'MLP {method} importance score'
            }
            for name, score in zip(results['selected_features'], results['feature_scores'])
        ],
        'evaluation': {
            'method': results['method'],
            'model_performance': results['model_performance']
        }
    }


if __name__ == "__main__":
    # Example usage
    print("MLP Feature Selection Methods Available:")
    print("1. weight_magnitude: Uses L2 norm of first layer weights")
    print("2. gradient_based: Uses gradient magnitudes w.r.t. inputs") 
    print("3. attention_weights: Uses attention mechanism weights")
    print("\nAll methods can be integrated into the existing framework!")