import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_feature_engineering.dataset_manager import DatasetManager

def verify_datasets():
    
    manager = DatasetManager()
    
    llm4fs_datasets = {
        "Bank": "bank",
        "Credit-G": "credit-g",
        "Pima Indians Diabetes": "pima-diabetes",
        "Give Me Some Credit": "give-me-some-credit"
    }
    
    paper_datasets = {
        "adult": "adult",
        "arrhythmia": "arrhythmia",
        "balance-scale": "balance-scale",
        "bank-marketing": "bank",  # Likely same as bank
        "breast-w": "breast-w",
        "blood-transfusion": "blood-transfusion",
        "car": "car-evaluation",
        "cdc-diabetes": "pima-diabetes",  # Likely related
        "cmc": "cmc",
        "communities": "communities",
        "covtype": "covtype",
        "credit-g": "credit-g",
        "diabetes": "pima-diabetes",
        "heart": "heart-disease",
        "vehicle": "vehicle",
        "myocardial": "myocardial",
        
        
        "airlines": "airlines",
        "eucalyptus": "eucalyptus",
        "jungle_chess": "jungle_chess_2pcs_raw_endgame_complete",
        "pel": "pol", 
        "tic-tac-toe": "BNG(tic-tac-toe)",
        
        "health-insurance": "health-insurance",
        "kidney-stone": "kidney-stone",
        "pharyngitis": "pharyngitis",
        "spaceship-titanic": "spaceship-titanic",
    }
    
    available_datasets = manager.list_datasets(source="all")
    print(f"Total available datasets: {len(available_datasets)}")
    print(f"Available TALENT datasets: {[d for d in available_datasets if 'kaggle' not in d.lower()]}")
    
    print("\n" + "="*70)
    print("CHECKING LLM4FS DATASETS (Required by Loris)")
    print("="*70)
    
    llm4fs_missing = []
    llm4fs_found = []
    
    for display_name, dataset_name in llm4fs_datasets.items():
        if dataset_name in available_datasets:
            llm4fs_found.append(f"{display_name} ({dataset_name})")
            try:
                info = manager.get_dataset_info(dataset_name)
                print(f"  ✅ {display_name:30} -> {dataset_name:30} Shape: {info.get('train_shape', info.get('shape', 'Unknown'))}")
            except Exception as e:
                print(f"  ⚠️  {display_name:30} -> {dataset_name:30} Found but error: {e}")
        else:
            llm4fs_missing.append(f"{display_name} ({dataset_name})")
            print(f"  ❌ {display_name:30} -> {dataset_name:30} MISSING")
    
    print("\n" + "="*70)
    print("CHECKING PAPER DATASETS (For full reproduction)")
    print("="*70)
    
    paper_missing = []
    paper_found = []
    
    for display_name, dataset_name in sorted(paper_datasets.items()):
        if dataset_name in available_datasets:
            paper_found.append(dataset_name)
            print(f"  ✅ {display_name:25} -> {dataset_name}")
        else:
            paper_missing.append(dataset_name)
            print(f"  ❌ {display_name:25} -> {dataset_name:30} MISSING")
            
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"LLM4FS datasets (priority): {len(llm4fs_found)}/{len(llm4fs_datasets)} ✅")
    print(f"Paper datasets available: {len(paper_found)}/{len(paper_datasets)}")
    print(f"Paper datasets missing: {len(paper_missing)}/{len(paper_datasets)}")
    
    if paper_missing:
        print("\nMissing datasets for full paper reproduction:")
        for m in sorted(set(paper_missing)):
            print(f"  • {m}")
            
    return len(llm4fs_missing) == 0

if __name__ == "__main__":
    success = verify_datasets()
    sys.exit(0 if success else 1)
