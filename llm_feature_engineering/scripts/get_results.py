#!/usr/bin/env python3
"""
Get Results - Comprehensive Analysis
====================================
This script extracts and analyzes all experimental results from pickle files.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def extract_detailed_results(pickle_path):
    """Extract detailed results with correct data structure"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    all_results = []
    
    for dataset_name, dataset_data in data.items():
        print(f"\nProcessing {dataset_name.upper()}...")
        

        n_features = dataset_data.get('ml_data_shapes', {}).get('X_train', [0, 0])[1]
        

        if 'selection_results' in dataset_data:
            selection_results = dataset_data['selection_results']
            
            for method_name, method_data in selection_results.items():
                print(f"  Method: {method_name}")
                
                if isinstance(method_data, dict):
                    if 'error' in method_data:
                        print(f"    Error: {method_data['error']}")
                        continue
                    

                    selected_features = method_data.get('features', [])
                    n_selected = len(selected_features) if isinstance(selected_features, list) else 0
                    

                    if 'evaluation' in method_data:
                        eval_data = method_data['evaluation']
                        
                        if isinstance(eval_data, dict) and 'model_aggregates' in eval_data:
                            model_aggregates = eval_data['model_aggregates']
                            
                            for classifier, classifier_results in model_aggregates.items():
                                if isinstance(classifier_results, dict):
                                    mean_score = classifier_results.get('mean_score', 0)
                                    std_score = classifier_results.get('std_score', 0)
                                    min_score = classifier_results.get('min_score', 0)
                                    max_score = classifier_results.get('max_score', 0)
                                    scores = classifier_results.get('scores', [])
                                    
                                    all_results.append({
                                        'Dataset': dataset_name,
                                        'Method': method_name,
                                        'Classifier': classifier,
                                        'Mean_Score': mean_score,
                                        'Std_Score': std_score,
                                        'Min_Score': min_score,
                                        'Max_Score': max_score,
                                        'Features_Selected': n_selected,
                                        'Original_Features': n_features,
                                        'Selection_Ratio': n_selected / n_features if n_features > 0 else 0,
                                        'Result_Type': 'Feature_Selection',
                                        'N_Trials': len(scores) if isinstance(scores, list) else 0
                                    })
                                    print(f"    {classifier}: Mean={mean_score:.4f} (±{std_score:.4f})")
        

        if 'autogluon_benchmark' in dataset_data:
            ag_results = dataset_data['autogluon_benchmark']
            
            for ag_method, ag_data in ag_results.items():
                if isinstance(ag_data, dict) and 'test_accuracy' in ag_data:
                    test_acc = ag_data['test_accuracy']
                    
                    all_results.append({
                        'Dataset': dataset_name,
                        'Method': f'AutoGluon_{ag_method}',
                        'Classifier': 'AutoML',
                        'Mean_Score': test_acc,
                        'Std_Score': 0,
                        'Min_Score': test_acc,
                        'Max_Score': test_acc,
                        'Features_Selected': n_features,
                        'Original_Features': n_features,
                        'Selection_Ratio': 1.0,
                        'Result_Type': 'AutoGluon',
                        'N_Trials': 1
                    })
                    print(f"  AutoGluon {ag_method}: {test_acc:.4f}")
    
    return pd.DataFrame(all_results)

def create_comprehensive_analysis(df, save_dir):
    """Create comprehensive analysis with multiple visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # Create 6-panel comprehensive analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Comprehensive Feature Selection Analysis', fontsize=16, fontweight='bold')
    

    ax1 = axes[0, 0]
    pivot_data = df.pivot_table(values='Mean_Score', index='Dataset', columns='Method', aggfunc='max', fill_value=0)
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Score'})
    ax1.set_title('Performance Heatmap (Best Mean Score per Method)')
    ax1.tick_params(axis='x', rotation=45)
    

    ax2 = axes[0, 1]
    method_stats = df.groupby('Result_Type')['Mean_Score'].agg(['mean', 'std', 'count'])
    bars = ax2.bar(method_stats.index, method_stats['mean'], yerr=method_stats['std'], 
                   capsize=5, alpha=0.7, color=['lightblue', 'lightcoral'])
    ax2.set_title('Average Performance by Method Type')
    ax2.set_ylabel('Mean Score')
    for bar, (idx, row) in zip(bars, method_stats.iterrows()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                f'{row["mean"]:.3f}\n(n={row["count"]})', ha='center', va='bottom')
    

    ax3 = axes[1, 0]
    fs_data = df[df['Result_Type'] == 'Feature_Selection']
    if not fs_data.empty:
        scatter = ax3.scatter(fs_data['Selection_Ratio'], fs_data['Mean_Score'], 
                   alpha=0.6, s=60, c=fs_data['Dataset'].astype('category').cat.codes, cmap='tab10')
        ax3.set_xlabel('Feature Selection Ratio')
        ax3.set_ylabel('Mean Score')
        ax3.set_title('Feature Selection Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for datasets
        cbar = fig.colorbar(scatter, ax=ax3)
        cbar.set_label('Dataset')
    

    ax4 = axes[1, 1]
    best_methods = df.loc[df.groupby('Dataset')['Mean_Score'].idxmax()]
    method_counts = best_methods['Method'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_counts)))
    ax4.bar(range(len(method_counts)), method_counts.values, color=colors)
    ax4.set_xticks(range(len(method_counts)))
    ax4.set_xticklabels(method_counts.index, rotation=45, ha='right')
    ax4.set_title('Best Method Frequency')
    ax4.set_ylabel('Number of Datasets')
    
    # Add count labels on bars
    for i, (method, count) in enumerate(method_counts.items()):
        ax4.text(i, count + 0.05, str(count), ha='center', va='bottom')
    

    ax5 = axes[2, 0]
    classifier_data = df[df['Result_Type'] == 'Feature_Selection']
    if not classifier_data.empty:
        unique_classifiers = classifier_data['Classifier'].unique()
        for classifier in unique_classifiers:
            data = classifier_data[classifier_data['Classifier'] == classifier]['Mean_Score']
            ax5.hist(data, alpha=0.6, label=classifier, bins=15)
        ax5.set_title('Performance Distribution by Classifier')
        ax5.set_xlabel('Mean Score')
        ax5.set_ylabel('Frequency')
        ax5.legend()
    

    ax6 = axes[2, 1]
    dataset_complexity = df.groupby('Dataset').agg({
        'Original_Features': 'first',
        'Mean_Score': 'max'
    }).reset_index()
    
    ax6.scatter(dataset_complexity['Original_Features'], dataset_complexity['Mean_Score'], 
               s=100, alpha=0.7, c=range(len(dataset_complexity)), cmap='viridis')
    ax6.set_xlabel('Number of Features')
    ax6.set_ylabel('Best Mean Score')
    ax6.set_title('Dataset Complexity vs Best Performance')
    
    # Add dataset labels
    for _, row in dataset_complexity.iterrows():
        ax6.annotate(row['Dataset'], (row['Original_Features'], row['Mean_Score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed method performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot for method performance
    ax1 = axes[0]
    methods_to_plot = df.groupby('Method')['Mean_Score'].count().sort_values(ascending=False).head(10).index
    plot_data = df[df['Method'].isin(methods_to_plot)]
    
    sns.boxplot(data=plot_data, x='Method', y='Mean_Score', ax=ax1)
    sns.swarmplot(data=plot_data, x='Method', y='Mean_Score', ax=ax1, 
                  alpha=0.7, size=4, color='red')
    
    ax1.set_title('Method Performance Distribution', fontsize=14)
    ax1.set_ylabel('Mean Score', fontsize=12)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Mean scores with error bars
    ax2 = axes[1]
    method_summary = df.groupby('Method')['Mean_Score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    
    ax2.bar(range(len(method_summary)), method_summary['mean'], 
           yerr=method_summary['std'], capsize=5, alpha=0.7)
    ax2.set_xticks(range(len(method_summary)))
    ax2.set_xticklabels(method_summary.index, rotation=45, ha='right')
    ax2.set_title('Method Performance Summary')
    ax2.set_ylabel('Mean Score')
    
    # Add value labels
    for i, (method, stats) in enumerate(method_summary.iterrows()):
        ax2.text(i, stats['mean'] + stats['std'] + 0.01, 
                f'{stats["mean"]:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'method_performance_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(df, save_dir):
    """Generate detailed statistical report"""
    save_dir = Path(save_dir)
    
    report = []
    report.append("=" * 100)
    report.append("DETAILED EXPERIMENTAL RESULTS ANALYSIS")
    report.append("=" * 100)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total experiments: {len(df)}")
    report.append("")
    

    report.append("OVERALL PERFORMANCE STATISTICS")
    report.append("-" * 60)
    report.append(f"Mean score: {df['Mean_Score'].mean():.4f} ± {df['Mean_Score'].std():.4f}")
    report.append(f"Median score: {df['Mean_Score'].median():.4f}")
    report.append(f"Best score: {df['Mean_Score'].max():.4f}")
    report.append(f"Worst score: {df['Mean_Score'].min():.4f}")
    report.append(f"Score range: {df['Mean_Score'].max() - df['Mean_Score'].min():.4f}")
    report.append("")
    

    report.append("TOP PERFORMING METHODS")
    report.append("-" * 60)
    method_performance = df.groupby('Method')['Mean_Score'].agg(['count', 'mean', 'std', 'max']).sort_values('mean', ascending=False)
    for i, (method, stats) in enumerate(method_performance.head(15).iterrows(), 1):
        report.append(f"{i:2d}. {method}")
        report.append(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        report.append(f"    Best: {stats['max']:.4f}")
        report.append("")
    

    report.append("DATASET PERFORMANCE ANALYSIS")
    report.append("-" * 60)
    dataset_stats = df.groupby('Dataset').agg({
        'Mean_Score': ['count', 'mean', 'std', 'max'],
        'Original_Features': 'first'
    })
    dataset_stats.columns = ['_'.join(col).strip() for col in dataset_stats.columns]
    dataset_stats = dataset_stats.sort_values('Mean_Score_max', ascending=False)
    
    for dataset in dataset_stats.index:
        stats = dataset_stats.loc[dataset]
        report.append(f"{dataset.upper()}")
        report.append(f"  Features: {int(stats['Original_Features_first'])}")
        report.append(f"  Methods tested: {int(stats['Mean_Score_count'])}")
        report.append(f"  Best score: {stats['Mean_Score_max']:.4f}")
        report.append(f"  Mean score: {stats['Mean_Score_mean']:.4f} ± {stats['Mean_Score_std']:.4f}")
        report.append("")
    

    report.append("METHOD COMPARISON")
    report.append("-" * 60)
    best_per_dataset = df.loc[df.groupby('Dataset')['Mean_Score'].idxmax()]
    method_wins = best_per_dataset['Method'].value_counts()
    report.append("Best method frequency:")
    for method, count in method_wins.items():
        percentage = (count / len(method_wins)) * 100
        report.append(f"  {method}: {count} wins ({percentage:.1f}%)")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open(save_dir / 'detailed_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    # Save detailed CSV
    df.to_csv(save_dir / 'detailed_results.csv', index=False)
    
    # Create summary table
    summary_table = df.groupby(['Dataset', 'Method']).agg({
        'Mean_Score': 'max',
        'Std_Score': 'mean',
        'Features_Selected': 'first',
        'Original_Features': 'first',
        'Selection_Ratio': 'first'
    }).reset_index()
    summary_table.to_csv(save_dir / 'summary_table.csv', index=False)
    
    print(report_text)
    return df

def generate_executive_summary(df):
    """Generate executive summary of results"""
    
    print()
    print("EXECUTIVE SUMMARY - LLM FEATURE ENGINEERING EXPERIMENTS")
    print("=" * 80)
    print()
    

    print("KEY METRICS")
    print("-" * 40)
    print(f"Total experiments conducted: {len(df)}")
    print(f"Datasets tested: {df['Dataset'].nunique()}")
    print(f"Feature selection methods: {df['Method'].nunique()}")
    print(f"Classification algorithms: {df['Classifier'].nunique()}")
    print(f"Overall performance range: {df['Mean_Score'].min():.3f} - {df['Mean_Score'].max():.3f}")
    print(f"Average performance: {df['Mean_Score'].mean():.3f} ± {df['Mean_Score'].std():.3f}")
    print()
    

    print("TOP PERFORMING METHODS")
    print("-" * 40)
    method_performance = df.groupby('Method')['Mean_Score'].agg(['mean', 'std', 'max']).sort_values('mean', ascending=False)
    for i, (method, stats) in enumerate(method_performance.iterrows(), 1):
        print(f"{i}. {method}: {stats['mean']:.3f} ± {stats['std']:.3f} (best: {stats['max']:.3f})")
    print()
    

    print("DATASET RESULTS")
    print("-" * 40)
    dataset_performance = df.groupby('Dataset').agg({
        'Mean_Score': 'max',
        'Original_Features': 'first'
    }).sort_values('Mean_Score', ascending=False)
    
    for dataset, stats in dataset_performance.iterrows():
        print(f"{dataset}: {stats['Mean_Score']:.3f} ({stats['Original_Features']} features)")
    print()
    

    print("WINNING METHODS PER DATASET")
    print("-" * 40)
    best_per_dataset = df.loc[df.groupby('Dataset')['Mean_Score'].idxmax()]
    for _, row in best_per_dataset.iterrows():
        print(f"{row['Dataset']}: {row['Method']} ({row['Classifier']}) - {row['Mean_Score']:.3f}")
    print()
    

    print("KEY FINDINGS")
    print("-" * 40)
    
    # Find best overall method
    best_method = method_performance.index[0]
    best_score = method_performance.loc[best_method, 'max']
    print(f"1. Best performing method: {best_method} (max score: {best_score:.3f})")
    
    # Find most consistent method
    most_consistent = method_performance.sort_values('std').index[0]
    consistency_std = method_performance.loc[most_consistent, 'std']
    print(f"2. Most consistent method: {most_consistent} (std: {consistency_std:.3f})")
    
    # Find easiest and hardest datasets
    easiest_dataset = dataset_performance.index[0]
    hardest_dataset = dataset_performance.index[-1]
    print(f"3. Easiest dataset: {easiest_dataset} ({dataset_performance.loc[easiest_dataset, 'Mean_Score']:.3f})")
    print(f"4. Hardest dataset: {hardest_dataset} ({dataset_performance.loc[hardest_dataset, 'Mean_Score']:.3f})")
    
    print()
    

    print("RECOMMENDATIONS")
    print("-" * 40)
    print(f"1. Use {best_method} for maximum performance")
    print(f"2. Use {most_consistent} for consistent results across datasets")
    
    # Find best classifier for each method
    best_classifiers = df.groupby('Classifier')['Mean_Score'].mean().sort_values(ascending=False)
    best_classifier = best_classifiers.index[0]
    print(f"3. {best_classifier} shows best overall classification performance")
    
    # Feature selection recommendation
    fs_data = df[df['Result_Type'] == 'Feature_Selection']
    if not fs_data.empty:
        avg_reduction = (1 - fs_data['Selection_Ratio'].mean()) * 100
        print(f"4. Feature selection provides ~{avg_reduction:.0f}% dimensionality reduction on average")
    
    print(f"5. Focus on {easiest_dataset} dataset type for best results")

def run_analysis():
    """Run complete analysis pipeline"""
    
    print("RUNNING COMPLETE LLM FEATURE ENGINEERING ANALYSIS")
    print("=" * 80)
    print()
    

    experiment_dirs = list(Path("experiments").glob("*/results/final_results_*.pkl"))
    if not experiment_dirs:
        print("No experiment results found!")
        print("Please run experiments first using: ./run_experiments.sh complete")
        return False
    
    print(f"Found {len(experiment_dirs)} experiment result(s)")
    

    best_pickle_path = None
    max_score = 0
    
    print("Checking pickle files for completeness...")
    for pickle_path in experiment_dirs:
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            num_datasets = len(data.keys()) if isinstance(data, dict) else 0
            
            # Check if datasets have selection_results
            has_selection_results = 0
            if isinstance(data, dict) and data:
                first_key = list(data.keys())[0]
                if 'selection_results' in data[first_key]:
                    has_selection_results = 1
            
            # Score = num_datasets * 10 + has_selection_results (prioritize selection_results)
            score = num_datasets * 10 + has_selection_results
            
            print(f"  {pickle_path}: {num_datasets} datasets, has_selection_results={bool(has_selection_results)}, score={score}")
            
            if score > max_score:
                max_score = score
                best_pickle_path = str(pickle_path)
        except Exception as e:
            print(f"  {pickle_path}: Error loading - {e}")
    
    if not best_pickle_path:
        print("No valid pickle files found!")
        return False
    
    pickle_path = best_pickle_path
    output_dir = "results_analysis"
    print(f"Using best pickle file: {pickle_path} (score={max_score})")
    
    print("Extracting detailed results from pickle file...")
    df = extract_detailed_results(pickle_path)
    
    if df.empty:
        print("No results extracted")
        return False
    
    print(f"Extracted {len(df)} detailed results")
    print(f"   - Datasets: {df['Dataset'].nunique()}")
    print(f"   - Methods: {df['Method'].nunique()}")
    print(f"   - Classifiers: {df['Classifier'].nunique()}")
    print(f"   - Result types: {df['Result_Type'].unique()}")
    

    print("\nCreating comprehensive visualizations...")
    create_comprehensive_analysis(df, output_dir)
    

    print("\nGenerating detailed report...")
    generate_detailed_report(df, output_dir)
    

    generate_executive_summary(df)
    
    print("\nAnalysis complete!")
    print(f"Files saved to: {output_dir}/")
    
    output_path = Path(output_dir)
    print("\nGenerated files:")
    for file in sorted(output_path.glob('*')):
        size = file.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
        print(f"  {file.name} ({size_str})")
    
    print()
    print("Quick Access:")
    print("-" * 40)
    print("  View visualizations: Open results_analysis/*.png files")
    print("  Read full report: results_analysis/detailed_analysis_report.txt")
    print("  Access raw data: results_analysis/detailed_results.csv")
    print("  Summary table: results_analysis/summary_table.csv")
    
    return True

def main():
    if not run_analysis():
        return
    
    print()
    print("Ready! Your LLM feature engineering analysis is complete.")
    print("Check the results_analysis/ directory for all outputs.")

if __name__ == "__main__":
    main()