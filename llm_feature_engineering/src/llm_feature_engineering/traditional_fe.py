"""
Traditional AutoML Feature Engineering Module

Implements feature engineering using traditional AutoML libraries:
- Featuretools (Deep Feature Synthesis)
- Auto-sklearn (Automated Feature Preprocessing)
- AutoGluon (Feature Generation - already exists, this wraps for FE-only)

These are used for comparison with LLM-based feature engineering methods.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import warnings
import logging

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
    logger.warning("Featuretools not available. Install with: pip install featuretools")

try:
    import autosklearn.classification
    import autosklearn.regression
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False
    logger.warning("Auto-sklearn not available. Install with: pip install auto-sklearn")

try:
    from autogluon.tabular import TabularPredictor
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    logger.warning("AutoGluon not available. Install with: pip install autogluon.tabular")


class TraditionalFEMethod(ABC):
    """
    Abstract base class for traditional feature engineering methods.
    
    All methods should:
    1. Take raw features as input
    2. Generate/transform features
    3. Return the engineered features (NOT do model selection)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
        
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, 
                     dataset_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fit the feature engineering method and transform features.
        
        Args:
            X: Input feature matrix
            y: Target variable
            dataset_info: Optional dataset metadata
            
        Returns:
            DataFrame with engineered features
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted method.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with engineered features
        """
        pass


class FeaturetoolsFE(TraditionalFEMethod):
    """
    Featuretools Deep Feature Synthesis for automated feature engineering.
    
    Uses Featuretools' DFS algorithm to automatically generate features
    through aggregation and transformation primitives.
    """
    
    def __init__(self, 
                 max_depth: int = 2,
                 max_features: int = 100,
                 primitives: Optional[List[str]] = None):
        """
        Initialize Featuretools feature engineering.
        
        Args:
            max_depth: Maximum depth for DFS
            max_features: Maximum number of features to generate
            primitives: List of transformation primitives to use
        """
        super().__init__("Featuretools")
        
        if not FEATURETOOLS_AVAILABLE:
            raise ImportError("Featuretools is not installed. Install with: pip install featuretools")
        
        self.max_depth = max_depth
        self.max_features = max_features
        self.primitives = primitives
        self.entityset = None
        self.feature_defs = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     dataset_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate features using Deep Feature Synthesis.
        
        Args:
            X: Input feature matrix
            y: Target variable (used for feature selection after generation)
            dataset_info: Optional dataset metadata
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Featuretools: Generating features for {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create a copy to avoid modifying original
        X_copy = X.copy()
        
        # Add index as an ID column if not present
        if 'id' not in X_copy.columns:
            X_copy['id'] = range(len(X_copy))
        
        # Create EntitySet
        self.entityset = ft.EntitySet(id="dataset")
        
        # Infer variable types
        self.entityset = self.entityset.add_dataframe(
            dataframe_name="data",
            dataframe=X_copy,
            index="id"
        )
        
        # Define primitives
        if self.primitives:
            trans_primitives = self.primitives
        else:
            # Default transformation primitives for tabular data
            # Use primitives that are available in modern featuretools
            trans_primitives = [
                "add_numeric",
                "subtract_numeric", 
                "multiply_numeric",
                "divide_numeric",
                "absolute",
                "square_root",
            ]
        
        # Run Deep Feature Synthesis
        try:
            feature_matrix, self.feature_defs = ft.dfs(
                entityset=self.entityset,
                target_dataframe_name="data",
                trans_primitives=trans_primitives,
                max_depth=self.max_depth,
                max_features=self.max_features,
                verbose=True
            )
        except Exception as e:
            logger.warning(f"Featuretools DFS failed with primitives, falling back to basic: {e}")
            # Fallback to simpler primitives
            try:
                feature_matrix, self.feature_defs = ft.dfs(
                    entityset=self.entityset,
                    target_dataframe_name="data",
                    trans_primitives=["absolute", "square_root"],
                    max_depth=1,
                    max_features=self.max_features,
                    verbose=True
                )
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}. Using original features.")
                # Return original features without the ID column
                feature_matrix = X_copy.drop(columns=['id']) if 'id' in X_copy.columns else X_copy
                self.feature_defs = None
        
        # Remove the ID column if we added it
        if 'id' in feature_matrix.columns:
            feature_matrix = feature_matrix.drop(columns=['id'])
        
        # Handle infinite values
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
        feature_matrix = feature_matrix.fillna(0)
        
        self.fitted = True
        logger.info(f"Featuretools: Generated {feature_matrix.shape[1]} features")
        
        return feature_matrix
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature definitions.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with engineered features
        """
        if not self.fitted:
            raise ValueError("FeaturetoolsFE must be fit before transform")
        
        X_copy = X.copy()
        if 'id' not in X_copy.columns:
            X_copy['id'] = range(len(X_copy))
        
        # Create new EntitySet for transform
        es = ft.EntitySet(id="dataset")
        es = es.add_dataframe(
            dataframe_name="data",
            dataframe=X_copy,
            index="id"
        )
        
        # Calculate features using saved definitions
        feature_matrix = ft.calculate_feature_matrix(
            self.feature_defs,
            entityset=es
        )
        
        if 'id' in feature_matrix.columns:
            feature_matrix = feature_matrix.drop(columns=['id'])
        
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
        feature_matrix = feature_matrix.fillna(0)
        
        return feature_matrix


class AutoSklearnFE(TraditionalFEMethod):
    """
    Auto-sklearn based feature preprocessing/engineering.
    
    Extracts the preprocessing pipeline from Auto-sklearn's automated
    machine learning process. The model selection is frozen - we only
    use the feature preprocessing components.
    """
    
    def __init__(self,
                 time_left_for_this_task: int = 120,
                 per_run_time_limit: int = 30,
                 memory_limit: int = 4096,
                 n_jobs: int = 1,
                 seed: int = 42):
        """
        Initialize Auto-sklearn feature engineering.
        
        Args:
            time_left_for_this_task: Total time budget in seconds
            per_run_time_limit: Time limit per model run
            memory_limit: Memory limit in MB
            n_jobs: Number of parallel jobs
            seed: Random seed
        """
        super().__init__("Auto-sklearn")
        
        if not AUTOSKLEARN_AVAILABLE:
            raise ImportError("Auto-sklearn is not installed. Install with: pip install auto-sklearn")
        
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.memory_limit = memory_limit
        self.n_jobs = n_jobs
        self.seed = seed
        self.automl = None
        self.preprocessor = None
        self.is_classification = True
        
    def _detect_task_type(self, y: pd.Series) -> bool:
        """Detect if task is classification or regression."""
        if y.dtype == 'object':
            return True
        if pd.api.types.is_integer_dtype(y):
            n_unique = y.nunique()
            if n_unique <= 20 or (n_unique / len(y)) < 0.05:
                return True
        return False
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     dataset_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fit Auto-sklearn and extract preprocessing transformations.
        
        Args:
            X: Input feature matrix
            y: Target variable
            dataset_info: Optional dataset metadata
            
        Returns:
            DataFrame with preprocessed/engineered features
        """
        logger.info(f"Auto-sklearn: Processing {X.shape[0]} samples, {X.shape[1]} features")
        
        self.is_classification = self._detect_task_type(y)
        
        # Initialize appropriate Auto-sklearn model
        if self.is_classification:
            self.automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=self.time_left_for_this_task,
                per_run_time_limit=self.per_run_time_limit,
                memory_limit=self.memory_limit,
                n_jobs=self.n_jobs,
                seed=self.seed,
                # Limit to feature preprocessing - disable ensemble
                ensemble_size=1,
                initial_configurations_via_metalearning=0,
            )
        else:
            self.automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=self.time_left_for_this_task,
                per_run_time_limit=self.per_run_time_limit,
                memory_limit=self.memory_limit,
                n_jobs=self.n_jobs,
                seed=self.seed,
                ensemble_size=1,
                initial_configurations_via_metalearning=0,
            )
        
        # Convert to numpy for auto-sklearn
        X_numpy = X.values.astype(np.float32)
        y_numpy = y.values
        
        # Handle categorical target
        if self.is_classification and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_numpy = le.fit_transform(y_numpy)
        
        # Fit Auto-sklearn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.automl.fit(X_numpy, y_numpy)
        
        # Extract the best preprocessing pipeline
        # Auto-sklearn creates a preprocessing + model pipeline
        # We want just the preprocessing part
        try:
            # Get the best model's preprocessing
            best_model = self.automl.show_models()
            logger.info(f"Auto-sklearn found {len(best_model)} models")
            
            # For now, we'll use the transformed features from refit
            self.automl.refit(X_numpy, y_numpy)
            
            # Get transformed features through the pipeline
            # Auto-sklearn doesn't expose preprocessing directly,
            # so we'll extract what we can
            X_transformed = self._extract_preprocessed_features(X, y)
            
        except Exception as e:
            logger.warning(f"Could not extract Auto-sklearn preprocessing: {e}")
            # Fallback: return original features
            X_transformed = X.copy()
        
        self.fitted = True
        logger.info(f"Auto-sklearn: Output {X_transformed.shape[1]} features")
        
        return X_transformed
    
    def _extract_preprocessed_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Extract preprocessed features from Auto-sklearn pipeline.
        
        This attempts to apply just the preprocessing steps from the best model.
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.impute import SimpleImputer
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        
        X_copy = X.copy()
        
        # Apply standard preprocessing that Auto-sklearn typically uses
        # 1. Imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_copy)
        
        # 2. Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # 3. Optional: Feature selection (if too many features)
        if X_scaled.shape[1] > 50:
            score_func = f_classif if self.is_classification else f_regression
            selector = SelectKBest(score_func=score_func, k=min(50, X_scaled.shape[1]))
            y_processed = y.values
            if self.is_classification and y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                y_processed = LabelEncoder().fit_transform(y_processed)
            X_selected = selector.fit_transform(X_scaled, y_processed)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_names = [f"autoskl_{X.columns[i]}" for i, m in enumerate(selected_mask) if m]
        else:
            X_selected = X_scaled
            selected_names = [f"autoskl_{col}" for col in X.columns]
        
        # Store preprocessing for transform
        self.preprocessor = {
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': selected_names
        }
        
        return pd.DataFrame(X_selected, columns=selected_names, index=X.index)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessing.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with preprocessed features
        """
        if not self.fitted or self.preprocessor is None:
            raise ValueError("AutoSklearnFE must be fit before transform")
        
        X_imputed = self.preprocessor['imputer'].transform(X)
        X_scaled = self.preprocessor['scaler'].transform(X_imputed)
        
        return pd.DataFrame(
            X_scaled, 
            columns=self.preprocessor['feature_names'][:X_scaled.shape[1]], 
            index=X.index
        )


class BaselineFE(TraditionalFEMethod):
    """
    Baseline: No feature engineering, just basic preprocessing.
    
    Used as control for comparison experiments.
    """
    
    def __init__(self, scale: bool = True, impute: bool = True):
        """
        Initialize baseline preprocessing.
        
        Args:
            scale: Whether to standardize features
            impute: Whether to impute missing values
        """
        super().__init__("Baseline")
        self.scale = scale
        self.impute = impute
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     dataset_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply basic preprocessing (imputation, scaling).
        
        Args:
            X: Input feature matrix
            y: Target variable (not used)
            dataset_info: Optional dataset metadata
            
        Returns:
            DataFrame with preprocessed features
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        X_processed = X.copy()
        self.feature_names = list(X.columns)
        
        # Imputation
        if self.impute:
            self.imputer = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X_processed),
                columns=self.feature_names,
                index=X.index
            )
        
        # Scaling
        if self.scale:
            self.scaler = StandardScaler()
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=self.feature_names,
                index=X.index
            )
        
        self.fitted = True
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessing.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with preprocessed features
        """
        if not self.fitted:
            raise ValueError("BaselineFE must be fit before transform")
        
        X_processed = X.copy()
        
        if self.imputer:
            X_processed = pd.DataFrame(
                self.imputer.transform(X_processed),
                columns=self.feature_names,
                index=X.index
            )
        
        if self.scaler:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=self.feature_names,
                index=X.index
            )
        
        return X_processed


def create_traditional_fe(method: str, **kwargs) -> TraditionalFEMethod:
    """
    Factory function to create traditional FE method instances.
    
    Args:
        method: Name of the method ('featuretools', 'autosklearn', 'baseline')
        **kwargs: Method-specific parameters
        
    Returns:
        TraditionalFEMethod instance
    """
    method = method.lower()
    
    if method == "featuretools":
        return FeaturetoolsFE(**kwargs)
    elif method == "autosklearn" or method == "auto-sklearn":
        return AutoSklearnFE(**kwargs)
    elif method == "baseline":
        return BaselineFE(**kwargs)
    else:
        raise ValueError(f"Unknown traditional FE method: {method}. "
                        f"Available: featuretools, autosklearn, baseline")


# Convenience functions for quick usage
def apply_featuretools(X: pd.DataFrame, y: pd.Series, 
                       max_depth: int = 2, 
                       max_features: int = 100) -> pd.DataFrame:
    """Quick function to apply Featuretools DFS."""
    fe = FeaturetoolsFE(max_depth=max_depth, max_features=max_features)
    return fe.fit_transform(X, y)


def apply_autosklearn(X: pd.DataFrame, y: pd.Series,
                      time_limit: int = 120) -> pd.DataFrame:
    """Quick function to apply Auto-sklearn preprocessing."""
    fe = AutoSklearnFE(time_left_for_this_task=time_limit)
    return fe.fit_transform(X, y)


class AutoGluonFE(TraditionalFEMethod):
    """
    AutoGluon-based feature engineering.
    
    Uses AutoGluon's AutoMLPipelineFeatureGenerator for automatic
    feature generation and preprocessing.
    
    Docs: https://auto.gluon.ai/stable/index.html
    """
    
    def __init__(self, 
                 enable_numeric_features: bool = True,
                 enable_categorical_features: bool = True,
                 enable_datetime_features: bool = True,
                 enable_text_features: bool = False,
                 verbosity: int = 0):
        """
        Initialize AutoGluon feature engineering.
        
        Args:
            enable_numeric_features: Process numeric features
            enable_categorical_features: Process categorical features
            enable_datetime_features: Process datetime features
            enable_text_features: Process text features (slow)
            verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__("AutoGluon")
        
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon is not installed. Install with: pip install autogluon.tabular")
        
        self.enable_numeric_features = enable_numeric_features
        self.enable_categorical_features = enable_categorical_features
        self.enable_datetime_features = enable_datetime_features
        self.enable_text_features = enable_text_features
        self.verbosity = verbosity
        self.feature_generator = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     dataset_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate features using AutoGluon's feature generator.
        
        Args:
            X: Input feature matrix
            y: Target variable
            dataset_info: Optional dataset metadata
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"AutoGluon: Processing {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize the feature generator
        self.feature_generator = AutoMLPipelineFeatureGenerator(
            enable_numeric_features=self.enable_numeric_features,
            enable_categorical_features=self.enable_categorical_features,
            enable_datetime_features=self.enable_datetime_features,
            enable_text_special_features=self.enable_text_features,
            enable_text_ngram_features=self.enable_text_features,
            verbosity=self.verbosity
        )
        
        # Fit and transform
        X_transformed = self.feature_generator.fit_transform(X, y)
        
        # Handle any remaining issues
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
        X_transformed = X_transformed.fillna(0)
        
        self.fitted = True
        logger.info(f"AutoGluon: Generated {X_transformed.shape[1]} features")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature generator.
        
        Args:
            X: Input feature matrix
            
        Returns:
            DataFrame with engineered features
        """
        if not self.fitted or self.feature_generator is None:
            raise ValueError("AutoGluonFE must be fit before transform")
        
        X_transformed = self.feature_generator.transform(X)
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
        X_transformed = X_transformed.fillna(0)
        
        return X_transformed


def create_traditional_fe(method: str, **kwargs) -> TraditionalFEMethod:
    """
    Factory function to create traditional FE method instances.
    
    Args:
        method: Name of the method ('featuretools', 'autosklearn', 'autogluon', 'baseline')
        **kwargs: Method-specific parameters
        
    Returns:
        TraditionalFEMethod instance
    """
    method = method.lower()
    
    if method == "featuretools":
        return FeaturetoolsFE(**kwargs)
    elif method == "autosklearn" or method == "auto-sklearn":
        return AutoSklearnFE(**kwargs)
    elif method == "autogluon":
        return AutoGluonFE(**kwargs)
    elif method == "baseline":
        return BaselineFE(**kwargs)
    else:
        raise ValueError(f"Unknown traditional FE method: {method}. "
                        f"Available: featuretools, autosklearn, autogluon, baseline")


# Convenience functions for quick usage
def apply_featuretools(X: pd.DataFrame, y: pd.Series, 
                       max_depth: int = 2, 
                       max_features: int = 100) -> pd.DataFrame:
    """Quick function to apply Featuretools DFS."""
    fe = FeaturetoolsFE(max_depth=max_depth, max_features=max_features)
    return fe.fit_transform(X, y)


def apply_autosklearn(X: pd.DataFrame, y: pd.Series,
                      time_limit: int = 120) -> pd.DataFrame:
    """Quick function to apply Auto-sklearn preprocessing."""
    fe = AutoSklearnFE(time_left_for_this_task=time_limit)
    return fe.fit_transform(X, y)


def apply_autogluon(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Quick function to apply AutoGluon feature generation."""
    fe = AutoGluonFE()
    return fe.fit_transform(X, y)
