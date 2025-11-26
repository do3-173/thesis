"""
Feature Selection Methods Module

Implements various feature selection methods including LLM-based approaches
and traditional statistical methods for comparison.
"""

import json
import time
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from .llm_interface import LLMInterface



class FeatureSelector(ABC):
    """
    Abstract base class for feature selection methods.
    
    Provides a common interface for different feature selection approaches
    while allowing method-specific implementations.
    """
    
    def __init__(self, name: str):
        """
        Initialize feature selector.
        
        Args:
            name: Name of the feature selection method
        """
        self.name = name
        self.feature_scores = {}
    
    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       dataset_info: Optional[Dict] = None, 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select features from the dataset.
        
        Args:
            X: Feature matrix
            y: Target variable
            dataset_info: Additional dataset information
            top_k: Number of top features to select
            
        Returns:
            List of dictionaries with feature information
        """
        pass
    
    def get_feature_names(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract feature names from selection results.
        
        Args:
            results: Feature selection results
            
        Returns:
            List of selected feature names
        """
        return [result.get('name', result.get('feature', '')) for result in results]


class TextBasedFeatureSelector(FeatureSelector):
    """
    Text-based feature selection using LLM semantic understanding.
    
    Based on Li et al. 2024 - "Exploring Large Language Models for Feature Selection"
    Uses dataset descriptions and feature metadata to guide selection.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize text-based feature selector.
        
        Args:
            llm_interface: LLM interface for API calls
        """
        super().__init__("Text-based LLM")
        self.llm = llm_interface
    
    def create_dataset_description(self, dataset_info: Dict) -> str:
        """
        Create a comprehensive dataset description for the LLM.
        
        Args:
            dataset_info: Dataset metadata and information
            
        Returns:
            Formatted dataset description string
        """
        name = dataset_info.get('name', 'Unknown')
        metadata = dataset_info.get('metadata', {})
        

        if 'shape' in dataset_info:
            shape = dataset_info['shape']
            columns = dataset_info['columns']
        else:
            shape = dataset_info.get('train_shape', (0, 0))
            columns = dataset_info.get('columns', [])
        
        description = f"""
Dataset: {name}
Description: This is a tabular dataset with {shape[0]} samples and {shape[1]} features.
Task: {metadata.get('task_type', metadata.get('problem_type', 'Classification/Regression'))}
Target Variable: {columns[-1] if columns else 'Unknown'}
Domain: {metadata.get('domain', 'General')}

Features:
"""
        
        for i, col in enumerate(columns[:-1]):
            feature_desc = metadata.get('feature_descriptions', {}).get(col, 'Numerical/Categorical feature')
            description += f"- {col}: {feature_desc}\\n"
            
        return description.strip()
    
    def create_feature_selection_prompt(self, dataset_description: str, feature_name: str) -> str:
        """
        Create a prompt for feature importance scoring.
        
        Args:
            dataset_description: Comprehensive dataset description
            feature_name: Name of the feature to evaluate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are an expert data scientist specializing in feature selection for machine learning.

{dataset_description}

Your task is to evaluate the importance of the feature "{feature_name}" for predicting the target variable.

Please provide:
1. An importance score between 0.0 and 1.0 (where 1.0 is most important)
2. A brief reasoning for your score

Format your response as JSON:
{{
    "feature": "{feature_name}",
    "importance_score": 0.XX,
    "reasoning": "Brief explanation of why this feature is important/unimportant"
}}
"""
        return prompt
    
    def score_feature(self, dataset_info: Dict, feature_name: str) -> Dict[str, Any]:
        """
        Score a single feature using LLM.
        
        Args:
            dataset_info: Dataset information dictionary
            feature_name: Name of the feature to score
            
        Returns:
            Dictionary with feature score and reasoning
        """
        dataset_desc = self.create_dataset_description(dataset_info)
        prompt = self.create_feature_selection_prompt(dataset_desc, feature_name)
        
        response = self.llm.call_llm(prompt, max_tokens=300, temperature=0.1)
        
        if not response:
            return {
                "feature": feature_name, 
                "importance_score": 0.0, 
                "reasoning": "API call failed"
            }
        
        try:

            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # Fallback: extract score from text
            score_match = re.search(r'"importance_score":\\s*([0-9.]+)', response)
            score = float(score_match.group(1)) if score_match else 0.5
            
            return {
                "feature": feature_name,
                "importance_score": score,
                "reasoning": "Parsed from unstructured response"
            }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       dataset_info: Optional[Dict] = None, 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select features using text-based LLM approach.
        
        Args:
            X: Feature matrix (not used in text-based approach)
            y: Target variable (not used in text-based approach)
            dataset_info: Dataset information for description
            top_k: Number of top features to select
            
        Returns:
            List of feature scores sorted by importance
        """
        if not dataset_info:
            raise ValueError("Dataset info required for text-based feature selection")
        
        features = list(X.columns)
        results = []
        
        print(f"Scoring {len(features)} features using text-based approach...")
        
        for i, feature in enumerate(features):
            print(f"{i+1}/{len(features)}: {feature}")
            result = self.score_feature(dataset_info, feature)
            results.append(result)
            
            # Small delay to be respectful to API
            time.sleep(0.5)
        

        results.sort(key=lambda x: x['importance_score'], reverse=True)
        
        if top_k:
            results = results[:top_k]
            
        self.feature_scores = {r['feature']: r['importance_score'] for r in results}
        
        return results


class LLM4FSHybridSelector(FeatureSelector):
    """
    LLM4FS hybrid feature selection approach.
    
    Based on Li & Xiu 2025 - "LLM4FS: Leveraging Large Language Models for Feature Selection"
    Combines LLM reasoning with traditional statistical methods.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize LLM4FS hybrid selector.
        
        Args:
            llm_interface: LLM interface for API calls
        """
        super().__init__("LLM4FS Hybrid")
        self.llm = llm_interface
    
    def prepare_data_sample(self, X: pd.DataFrame, y: pd.Series, sample_size: int = 200) -> str:
        """
        Prepare a data sample for the LLM.
        
        Args:
            X: Feature matrix
            y: Target variable
            sample_size: Number of samples to include
            
        Returns:
            CSV string representation of the data sample
        """

        df = X.copy()
        df[y.name or 'target'] = y
        

        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df.copy()
            

        return sample_df.to_csv(index=False)
    
    def create_hybrid_prompt(self, data_csv: str, target_name: str, task_type: str = "classification") -> str:
        """
        Create prompt for hybrid LLM4FS approach matching the paper specification.
        
        Based on LLM4FS paper (Li & Xiu, 2025), Figure 2.
        The LLM should apply traditional data-driven methods and return feature importance scores.
        
        Args:
            data_csv: CSV representation of the data
            target_name: Name of the target/class column
            task_type: Type of ML task
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Please apply random forest, forward sequential selection, backward sequential selection, recursive feature elimination (RFE), minimum redundancy maximum relevance (MRMR), and mutual information (MI) separately to analyze the dataset samples. This is a {task_type} task, where "{target_name}" represents the classification target. Please analyze the importance scores of all features. The score range is [0.0, 1.0], and the score of each feature should be different. The output format is as follows, in JSON file format.

Format for Response:
[
    {{
        "concept-1": "feature_name_1",
        "reasoning": "The feature importance score is calculated using a random forest classifier. A higher score indicates greater importance in predicting the target variable.",
        "score": 0.95
    }},
    {{
        "concept-2": "feature_name_2", 
        "reasoning": "Combined analysis from RF, MI, RFE, MRMR, and sequential selection methods.",
        "score": 0.85
    }}
]

Dataset Samples (csv file with 200 samples):
{data_csv}

Important: 
- Apply the statistical methods (RF, MI, RFE, MRMR, sequential selection) to the data
- Provide a unique score for each feature
- Rank all features except the target column
- Base scores on actual statistical analysis, not semantic reasoning"""
        return prompt
    
    def extract_features_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract features from LLM response (handles both complete and partial JSON).
        
        Args:
            response: LLM response text
            
        Returns:
            List of feature dictionaries with scores
        """
        features = []
        

        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                for item in parsed:
                    # Handle both concept-N and name formats
                    name_key = None
                    for key in item.keys():
                        if 'concept' in key.lower() or key == 'name' or key == 'feature':
                            name_key = key
                            break
                    
                    if name_key:
                        features.append({
                            'name': item[name_key],
                            'importance_score': float(item.get('score', item.get('importance_score', 0.5))),
                            'reasoning': item.get('reasoning', 'LLM4FS hybrid score')
                        })
            return features
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract from partial/malformed JSON
        # Pattern for "concept-N": "feature_name" ... "score": 0.XX
        pattern = r'"(?:concept-\d+|name|feature)":\s*"([^"]+)"[^}]*"(?:score|importance_score)":\s*([0-9.]+)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for name, score in matches:
            try:
                features.append({
                    'name': name,
                    'importance_score': float(score),
                    'reasoning': 'LLM4FS hybrid score (extracted)'
                })
            except ValueError:
                continue
                
        return features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       dataset_info: Optional[Dict] = None, 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select features using LLM4FS hybrid approach.
        
        Implements the hybrid strategy from Li & Xiu (2025):
        - Send ~200 data samples to LLM
        - LLM applies traditional methods (RF, MI, RFE, MRMR, sequential selection)
        - Returns ranked features with importance scores
        
        Args:
            X: Feature matrix
            y: Target variable
            dataset_info: Dataset information (optional)
            top_k: Number of top features to select
            
        Returns:
            List of selected features with scores
        """
        print("Applying LLM4FS hybrid feature selection...")
        print(f"Dataset shape: {X.shape}")
        

        data_csv = self.prepare_data_sample(X, y, sample_size=200)
        

        task_type = "classification"
        if dataset_info and 'metadata' in dataset_info:
            task_type = dataset_info['metadata'].get('task_type', 
                                                   dataset_info['metadata'].get('problem_type', 'classification'))
        

        target_name = y.name if y.name else "Class"
        

        prompt = self.create_hybrid_prompt(data_csv, target_name, task_type)
        
        # Make LLM call with higher token limit (paper uses more output)
        print("Sending request to LLM...")
        response = self.llm.call_llm(prompt, max_tokens=4000, temperature=0.1)
        
        if not response:
            print("ERROR: API call failed")
            return []
        

        features = self.extract_features_from_response(response)
        
        if not features:
            print("WARNING: Could not extract features from LLM response")
            print(f"Response preview: {response[:500]}...")
            return []
        
        print(f"Successfully extracted {len(features)} features")
        
        # Sort by importance score
        features.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        

        self.feature_scores = {f['name']: f['importance_score'] for f in features}
        
        if top_k:
            features = features[:top_k]
            
        return features


class TraditionalFeatureSelector(FeatureSelector):
    """
    Traditional feature selection methods for comparison baselines.
    
    Implements mutual information and random forest importance methods.
    """
    
    def __init__(self, method: str = "mutual_info"):
        """
        Initialize traditional feature selector.
        
        Args:
            method: Selection method ('mutual_info' or 'random_forest')
        """
        super().__init__(f"Traditional {method}")
        self.method = method
    
    def is_classification_task(self, y: pd.Series) -> bool:
        """
        Determine if this is a classification task.
        
        Args:
            y: Target variable
            
        Returns:
            True if classification, False if regression
        """

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
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Mutual information based feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            top_k: Number of top features to select
            
        Returns:
            List of features with importance scores
        """

        if self.is_classification_task(y):
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            scores = mutual_info_classif(X, y_encoded, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
            
        results = []
        for i, feature in enumerate(X.columns):
            results.append({
                'name': feature,
                'importance_score': scores[i],
                'reasoning': 'Mutual information score'
            })
            
        results.sort(key=lambda x: x['importance_score'], reverse=True)
        
        if top_k:
            results = results[:top_k]
            
        return results
        
    def random_forest_selection(self, X: pd.DataFrame, y: pd.Series, 
                              top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Random Forest based feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            top_k: Number of top features to select
            
        Returns:
            List of features with importance scores
        """
        # Handle categorical target
        if self.is_classification_task(y):
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            y_encoded = y
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
        rf.fit(X, y_encoded)
        importances = rf.feature_importances_
        
        results = []
        for i, feature in enumerate(X.columns):
            results.append({
                'name': feature,
                'importance_score': importances[i],
                'reasoning': 'Random Forest feature importance'
            })
            
        results.sort(key=lambda x: x['importance_score'], reverse=True)
        
        if top_k:
            results = results[:top_k]
            
        return results
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       dataset_info: Optional[Dict] = None, 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select features using specified traditional method.
        
        Args:
            X: Feature matrix
            y: Target variable
            dataset_info: Dataset information (not used)
            top_k: Number of top features to select
            
        Returns:
            List of selected features with scores
        """
        if self.method == "mutual_info":
            return self.mutual_information_selection(X, y, top_k)
        elif self.method == "random_forest":
            return self.random_forest_selection(X, y, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def create_feature_selector(selector_type: str, **kwargs) -> FeatureSelector:
    """
    Factory function to create feature selector instances.
    
    Args:
        selector_type: Type of selector ('text_based', 'llm4fs', 'traditional')
        **kwargs: Additional arguments for selector initialization
        
    Returns:
        Feature selector instance
        
    Raises:
        ValueError: If selector type is not supported
    """
    if selector_type == "text_based":
        if 'llm_interface' not in kwargs:
            raise ValueError("LLM interface required for text-based selector")
        return TextBasedFeatureSelector(kwargs['llm_interface'])
    
    elif selector_type == "llm4fs":
        if 'llm_interface' not in kwargs:
            raise ValueError("LLM interface required for LLM4FS selector")
        return LLM4FSHybridSelector(kwargs['llm_interface'])
    
    elif selector_type == "caafe":
        if 'llm_interface' not in kwargs:
            raise ValueError("LLM interface required for CAAFE selector")
        from .caafe import CAAFESelector
        return CAAFESelector(kwargs['llm_interface'], **{k: v for k, v in kwargs.items() if k != 'llm_interface'})

    elif selector_type == "traditional":
        method = kwargs.get('method', 'mutual_info')
        return TraditionalFeatureSelector(method)
    
    elif selector_type == "mlp_weights":
        return MLPWeightSelector(**kwargs)
    
    elif selector_type == "mlp_permutation":
        return MLPPermutationSelector(**kwargs)
    
    else:
        raise ValueError(f"Unsupported selector type: {selector_type}")


class MLPWeightSelector(FeatureSelector):
    """
    MLP-based feature selection using first layer weight magnitudes.
    
    Uses the L2 norm of the first layer weights as feature importance scores.
    Fast and effective for neural network-based feature ranking.
    """
    
    def __init__(self, **config_params):
        """
        Initialize MLP Weight Selector.
        
        Args:
            **config_params: Configuration parameters (includes enabled, top_k, etc.)
        """
        super().__init__("mlp_weights")
        

        self.top_k_default = config_params.get('top_k', 10)
        
        # Filter out config-specific parameters that shouldn't go to sklearn
        sklearn_params = {}
        config_only_params = {'enabled', 'top_k'}
        
        for key, value in config_params.items():
            if key not in config_only_params:
                sklearn_params[key] = value
        

        self.mlp_params = {
            'hidden_layer_sizes': (64, 32),
            'max_iter': 200,
            'random_state': 42,
            **sklearn_params
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       dataset_info: Optional[Dict] = None, 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select features using MLP weight magnitude analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            dataset_info: Information about the dataset
            top_k: Number of top features to select
            
        Returns:
            List of selected features with importance scores
        """
        start_time = time.time()
        
        # Import here to avoid dependency issues
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # Determine task type
        task_type = self._determine_task_type(y)
        

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        

        if task_type == 'classification':
            mlp = MLPClassifier(**self.mlp_params)
        else:
            mlp = MLPRegressor(**self.mlp_params)
        

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42,
            stratify=y if task_type == 'classification' else None
        )
        

        mlp.fit(X_train, y_train)
        
        # Calculate feature importance using first layer weights
        first_layer_weights = mlp.coefs_[0]  # Shape: (n_features, n_hidden_units)
        importance_scores = np.linalg.norm(first_layer_weights, axis=1)
        

        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        

        features = []
        for i, (feature_name, score) in enumerate(zip(X.columns, importance_scores)):
            features.append({
                'feature': feature_name,
                'importance_score': float(score),
                'reasoning': f'MLP weight magnitude importance: {score:.4f}. This score represents the L2 norm of the first layer weights connected to this feature, indicating its influence on the neural network\'s initial processing.'
            })
        

        features.sort(key=lambda x: x['importance_score'], reverse=True)
        

        if top_k is not None:
            features = features[:top_k]
        elif hasattr(self, 'top_k_default'):
            features = features[:self.top_k_default]
        
        execution_time = time.time() - start_time
        print(f"MLP Weight feature selection completed in {execution_time:.3f}s")
        
        return features
    
    def _determine_task_type(self, y) -> str:
        """Determine if task is classification or regression."""
        import numpy as np
        

        if hasattr(y, 'dtype'):
            dtype = y.dtype
        else:
            dtype = np.array(y).dtype
            
        if hasattr(y, 'unique'):
            unique_values = len(y.unique())
        else:
            unique_values = len(np.unique(y))
            
        if dtype == 'object' or unique_values < 10:
            return 'classification'
        else:
            return 'regression'


class MLPPermutationSelector(FeatureSelector):
    """
    MLP-based feature selection using permutation importance.
    
    Uses permutation importance with a trained MLP to determine feature importance.
    More accurate but slower than weight-based methods.
    """
    
    def __init__(self, **config_params):
        """
        Initialize MLP Permutation Selector.
        
        Args:
            **config_params: Configuration parameters (includes enabled, top_k, n_repeats, etc.)
        """
        super().__init__("mlp_permutation")
        

        self.top_k_default = config_params.get('top_k', 10)
        self.n_repeats = config_params.get('n_repeats', 5)
        
        # Filter out config-specific parameters that shouldn't go to sklearn
        sklearn_params = {}
        config_only_params = {'enabled', 'top_k', 'n_repeats'}
        
        for key, value in config_params.items():
            if key not in config_only_params:
                sklearn_params[key] = value
        
        # Set default MLP parameters
        self.mlp_params = {
            'hidden_layer_sizes': (64, 32),
            'max_iter': 200,
            'random_state': 42,
            **sklearn_params
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       dataset_info: Optional[Dict] = None, 
                       top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Select features using MLP permutation importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            dataset_info: Information about the dataset
            top_k: Number of top features to select
            
        Returns:
            List of selected features with importance scores
        """
        start_time = time.time()
        
        # Import here to avoid dependency issues
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance
        
        # Determine task type
        task_type = self._determine_task_type(y)
        
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
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            mlp, X_test, y_test, 
            n_repeats=self.n_repeats, 
            random_state=self.mlp_params.get('random_state', 42)
        )
        
        importance_scores = perm_importance.importances_mean
        importance_std = perm_importance.importances_std
        
        # Normalize scores to [0, 1]
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
            importance_std = importance_std / np.max(perm_importance.importances_mean)
        
        # Create feature list with scores
        features = []
        for i, (feature_name, score, std) in enumerate(zip(X.columns, importance_scores, importance_std)):
            features.append({
                'feature': feature_name,
                'importance_score': float(score),
                'reasoning': f'MLP permutation importance: {score:.4f} ± {std:.4f}. This score measures the decrease in model performance when this feature is randomly shuffled, indicating its predictive value for the neural network.'
            })
        
        # Sort by importance
        features.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Select top k features if specified
        if top_k is not None:
            features = features[:top_k]
        elif hasattr(self, 'top_k_default'):
            features = features[:self.top_k_default]
        
        execution_time = time.time() - start_time
        print(f"MLP Permutation feature selection completed in {execution_time:.3f}s")
        
        return features
    
    def _determine_task_type(self, y) -> str:
        """Determine if task is classification or regression."""
        import numpy as np
        
        # Handle both pandas Series and numpy arrays
        if hasattr(y, 'dtype'):
            dtype = y.dtype
        else:
            dtype = np.array(y).dtype
            
        if hasattr(y, 'unique'):
            unique_values = len(y.unique())
        else:
            unique_values = len(np.unique(y))
            
        if dtype == 'object' or unique_values < 10:
            return 'classification'
        else:
            return 'regression'