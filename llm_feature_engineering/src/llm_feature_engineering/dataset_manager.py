"""
Dataset Management Module

Handles loading and processing of datasets from various sources including
TALENT benchmark library and local CSV files.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder

# Add TALENT to Python path if it exists locally
talent_path = Path(__file__).parent.parent.parent.parent / "TALENT"
if talent_path.exists():
    sys.path.insert(0, str(talent_path))

try:
    # Set DATA_PATH BEFORE importing get_dataset
    import TALENT.model.lib.data as talent_data
    # DATA_PATH should be the TALENT root, not example_datasets
    # because get_dataset will append dataset_path and dataset_name
    talent_data.DATA_PATH = str(talent_path)
    
    from TALENT.model.lib.data import get_dataset
    TALENT_AVAILABLE = True
    print(f"TALENT library loaded successfully from {talent_path}")
    print(f"TALENT DATA_PATH set to: {talent_path}")
except ImportError:
    TALENT_AVAILABLE = False
    print("TALENT library not available. Using local CSV files only.")


class DatasetManager:
    """
    Manages dataset loading from multiple sources with unified interface.
    
    Supports both TALENT benchmark datasets and local CSV files with metadata.
    """
    
    def __init__(self, dataset_dir: str = "datasets_csv"):
        """
        Initialize dataset manager.
        
        Args:
            dataset_dir: Directory containing local CSV datasets
        """
        self.dataset_dir = Path(dataset_dir)
        self.local_datasets = self._discover_local_datasets()
        self.talent_datasets = self._discover_talent_datasets() if TALENT_AVAILABLE else []
        
    def _discover_local_datasets(self) -> List[str]:
        """
        Discover available local CSV datasets.
        
        Returns:
            List of dataset names
        """
        if not self.dataset_dir.exists():
            return []
            
        csv_files = list(self.dataset_dir.glob("*.csv"))
        dataset_names = []
        
        for csv_file in csv_files:
            name = csv_file.stem
            # Skip metadata files and subsets
            if not name.endswith("_metadata") and "subset" not in name:
                dataset_names.append(name)
                
        return sorted(dataset_names)
    
    def _discover_talent_datasets(self) -> List[str]:
        """
        Discover available TALENT benchmark datasets.
        
        Returns:
            List of TALENT dataset names that are actually available locally
        """
        if not TALENT_AVAILABLE:
            return []
        
        try:
            # Check what datasets are actually available in the TALENT directory
            import os
            talent_datasets_dir = Path(__file__).parent.parent.parent.parent / "TALENT" / "example_datasets"
            if talent_datasets_dir.exists():
                # Get actual directories that exist
                actual_dirs = [d for d in os.listdir(talent_datasets_dir) 
                              if os.path.isdir(talent_datasets_dir / d) and d != '__pycache__']
                print(f"Found {len(actual_dirs)} TALENT datasets: {actual_dirs}")
                return actual_dirs
            else:
                return []
        except Exception as e:
            print(f"Error accessing TALENT datasets: {e}")
            return []
    
    def list_datasets(self, source: str = "all") -> List[str]:
        """
        List available datasets from specified source.
        
        Args:
            source: Dataset source ('local', 'talent', or 'all')
            
        Returns:
            List of dataset names
        """
        if source == "local":
            return self.local_datasets
        elif source == "talent":
            return self.talent_datasets
        elif source == "all":
            return self.local_datasets + self.talent_datasets
        else:
            raise ValueError("Source must be 'local', 'talent', or 'all'")
    
    def load_local_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load dataset from local CSV files.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (dataframe, metadata_dict)
        """
        csv_path = self.dataset_dir / f"{dataset_name}.csv"
        metadata_path = self.dataset_dir / f"{dataset_name}_metadata.json"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {csv_path}")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return df, metadata
    
    def encode_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features for sklearn compatibility.
        
        Categorical features (columns starting with 'cat_') are converted from strings
        to numeric values using LabelEncoder. This is necessary for sklearn classifiers
        that require numeric input.
        
        Args:
            train_df: Training DataFrame with potential categorical features
            test_df: Test DataFrame with potential categorical features
            
        Returns:
            Tuple of (train_encoded, test_encoded) DataFrames
        """
        train_encoded = train_df.copy()
        test_encoded = test_df.copy()
        
        # Identify categorical columns
        categorical_cols = [col for col in train_df.columns if col.startswith('cat_')]
        
        if not categorical_cols:
            return train_encoded, test_encoded
        
        print(f"Encoding {len(categorical_cols)} categorical features...")
        
        for col in categorical_cols:
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(train_df[col]):
                continue
                
            # Fit encoder on combined train+test to handle all categories
            le = LabelEncoder()
            
            # Combine all unique values from both train and test
            all_values = pd.concat([train_df[col], test_df[col]], ignore_index=True)
            le.fit(all_values)
            
            # Transform both datasets
            train_encoded[col] = le.transform(train_df[col])
            test_encoded[col] = le.transform(test_df[col])
            
            print(f"  {col}: {len(le.classes_)} unique categories encoded")
        
        return train_encoded, test_encoded
    
    def load_talent_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Load dataset from TALENT benchmark library.
        
        Args:
            dataset_name: Name of the TALENT dataset to load
            
        Returns:
            Tuple of (train_df, test_df, metadata_dict)
        """
        if not TALENT_AVAILABLE:
            raise ImportError("TALENT library not available")
            
        if dataset_name not in self.talent_datasets:
            raise ValueError(f"Dataset {dataset_name} not found in TALENT benchmark")
        
        try:
            # DATA_PATH is already set at module level
            # get_dataset returns (train_val_data, test_data, info)
            # where train_val_data = (N_trainval, C_trainval, y_trainval) with 'train' and 'val' keys
            # and test_data = (N_test, C_test, y_test) with 'test' key
            train_val_data, test_data, info = get_dataset(dataset_name, 'example_datasets')
            
            # Unpack train/val and test data
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
                
                # Build feature names from first part's shape
                feature_names = []
                if N_dict is not None and parts[0] in N_dict:
                    n_num = N_dict[parts[0]].shape[1]
                    feature_names.extend([f'num_{i}' for i in range(n_num)])
                if C_dict is not None and parts[0] in C_dict:
                    n_cat = C_dict[parts[0]].shape[1]
                    feature_names.extend([f'cat_{i}' for i in range(n_cat)])
                
                return pd.DataFrame(X_combined, columns=feature_names)
            
            # Combine train and val sets for full training data
            train_df = combine_features(N_trainval, C_trainval, ['train', 'val'])
            test_df = combine_features(N_test, C_test, ['test'])
            
            # Combine train and val targets
            y_train_full = np.concatenate([y_trainval['train'], y_trainval['val']])
            y_test_arr = y_test['test']
            
            # Add target column
            train_df['target'] = y_train_full
            test_df['target'] = y_test_arr
            
            # Encode categorical features for sklearn compatibility
            # This converts string categorical features to numeric values
            train_df, test_df = self.encode_categorical_features(train_df, test_df)
            
            # Create metadata
            feature_columns = [col for col in train_df.columns if col != 'target']
            n_num = N_trainval['train'].shape[1] if N_trainval is not None else 0
            n_cat = C_trainval['train'].shape[1] if C_trainval is not None else 0
            
            metadata = {
                'name': dataset_name,
                'problem_type': 'classification',
                'n_num_features': n_num,
                'n_cat_features': n_cat,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'n_features': len(feature_columns),
                'feature_columns': feature_columns,
                'label_columns': ['target'],
                'target_column': 'target'
            }
            
            return train_df, test_df, metadata
            
        except Exception as e:
            raise RuntimeError(f"Error loading TALENT dataset {dataset_name}: {e}")
    
    def load_dataset(self, dataset_name: str, source: str = "auto") -> Dict[str, Any]:
        """
        Load dataset with automatic source detection.
        
        Args:
            dataset_name: Name of the dataset to load
            source: Source preference ('auto', 'local', 'talent')
            
        Returns:
            Dictionary containing dataset information and data
        """
        result = {'name': dataset_name, 'source': None}
        
        if source == "auto":
            # Try TALENT first, then local
            if dataset_name in self.talent_datasets:
                source = "talent"
            elif dataset_name in self.local_datasets:
                source = "local"
            else:
                raise ValueError(f"Dataset {dataset_name} not found in any source")
        
        if source == "talent":
            train_df, test_df, metadata = self.load_talent_dataset(dataset_name)
            result.update({
                'source': 'talent',
                'train_data': train_df,
                'test_data': test_df,
                'metadata': metadata,
                'has_splits': True
            })
        elif source == "local":
            df, metadata = self.load_local_dataset(dataset_name)
            result.update({
                'source': 'local',
                'data': df,
                'metadata': metadata,
                'has_splits': False
            })
        else:
            raise ValueError("Source must be 'auto', 'local', or 'talent'")
        
        return result
    
    def get_dataset_info(self, dataset_name: str, source: str = "auto") -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset without loading full data.
        
        Args:
            dataset_name: Name of the dataset
            source: Source preference ('auto', 'local', 'talent')
            
        Returns:
            Dictionary with dataset information
        """
        dataset = self.load_dataset(dataset_name, source)
        
        info = {
            'name': dataset_name,
            'source': dataset['source'],
            'metadata': dataset['metadata']
        }
        
        if dataset['source'] == 'local':
            df = dataset['data']
            info.update({
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(3).to_dict('records')
            })
        else:  # TALENT dataset
            train_df = dataset['train_data']
            test_df = dataset['test_data']
            info.update({
                'train_shape': train_df.shape,
                'test_shape': test_df.shape,
                'columns': list(train_df.columns),
                'dtypes': train_df.dtypes.to_dict(),
                'missing_values': train_df.isnull().sum().to_dict(),
                'sample_data': train_df.head(3).to_dict('records')
            })
        
        return info
    
    def prepare_for_ml(self, dataset_name: str, source: str = "auto", 
                      test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Prepare dataset for machine learning experiments.
        
        Args:
            dataset_name: Name of the dataset
            source: Source preference
            test_size: Test split ratio (only for local datasets)
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with prepared ML data
        """
        from sklearn.model_selection import train_test_split
        
        dataset = self.load_dataset(dataset_name, source)
        
        if dataset['source'] == 'talent':
            # Use existing splits
            train_df = dataset['train_data']
            test_df = dataset['test_data']
            
            # Extract features and target
            metadata = dataset['metadata']
            feature_cols = metadata['feature_columns']
            target_col = metadata['label_columns'][0]
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
        else:
            # Create splits for local data
            df = dataset['data']
            
            # Assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        
        return {
            'dataset_name': dataset_name,
            'source': dataset['source'],
            'metadata': dataset['metadata'],
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X_train.columns),
            'target_name': y_train.name if hasattr(y_train, 'name') else 'target'
        }