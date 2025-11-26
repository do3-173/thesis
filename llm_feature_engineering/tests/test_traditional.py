#!/usr/bin/env python3
"""
Test Traditional Feature Selection on a small dataset.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import logging
from sklearn.datasets import load_breast_cancer
from llm_feature_engineering.feature_selection import TraditionalFeatureSelector


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_traditional():
    print("Testing Traditional Feature Selection (Random Forest) on Breast Cancer dataset...")
    

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Use only a small subset for quick testing
    X_small = X.iloc[:100, :10]  # 100 samples, 10 features
    y_small = y.iloc[:100]
    
    print(f"\nDataset shape: {X_small.shape}")
    print(f"Features: {X_small.columns.tolist()}")
    

    selector = TraditionalFeatureSelector(method="random_forest")
    
    print("\n" + "="*60)
    print("Starting Feature Selection...")
    print("="*60)
    

    selected_features = selector.select_features(X_small, y_small, top_k=5)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    
    print("\nTop 5 features by importance:")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i}. {feat['name']}: {feat['importance_score']:.4f}")
    
    print("\n✓ Traditional test completed!")

if __name__ == "__main__":
    test_traditional()
