#!/usr/bin/env python3
"""
Test CAAFE with Claude on a small dataset.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from llm_feature_engineering.llm_interface import create_llm_interface
from llm_feature_engineering.caafe import CAAFESelector

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_caafe():
    print("Testing CAAFE with Claude on Breast Cancer dataset...")
    
    # Load a simple dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Use only a small subset for quick testing
    X_small = X.iloc[:100, :5]  # 100 samples, 5 features
    y_small = y.iloc[:100]
    
    print(f"\nDataset shape: {X_small.shape}")
    print(f"Features: {X_small.columns.tolist()}")
    
    # Create Claude interface
    llm = create_llm_interface(provider="anthropic", model="claude-3-haiku-20240307")
    
    # Create CAAFE selector with few iterations for testing
    caafe = CAAFESelector(
        llm_interface=llm,
        dataset_description="Breast cancer diagnosis dataset with tumor measurements",
        n_iterations=5,  # More iterations to find improvements
        metric='roc_auc'
    )
    
    print("\n" + "="*60)
    print("Starting CAAFE feature engineering...")
    print("="*60)
    
    # Run CAAFE
    selected_features = caafe.select_features(X_small, y_small)
    
    print("\n" + "="*60)
    print("CAAFE Results:")
    print("="*60)
    print(f"\nTotal features after CAAFE: {len(selected_features)}")
    print(f"Code blocks applied: {len(caafe.code_history)}")
    
    print("\nTop 10 features by importance:")
    for i, feat in enumerate(selected_features[:10], 1):
        generated = "🆕" if feat.get('generated', False) else "📊"
        print(f"{i}. {generated} {feat['name']}: {feat['score']:.4f}")
    
    print("\n" + "="*60)
    print("Feature Engineering History:")
    print("="*60)
    for i, entry in enumerate(caafe.feature_history, 1):
        print(f"\nIteration {i}:")
        print(f"Score: {entry['score']:.4f}")
        print(f"Added features: {entry['added_features']}")
        print(f"Code:\n{entry['code'][:200]}...")  # Show first 200 chars
    
    print("\n✓ CAAFE test completed!")

if __name__ == "__main__":
    test_caafe()
