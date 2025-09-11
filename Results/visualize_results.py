#!/usr/bin/env python3
"""
Visualization Script for Membership Inference Attack Results
Analyzes and visualizes experimental results from the Results directory
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsAnalyzer:
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.summary_stats = {}
        
    def parse_result_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Parse a single result file and extract experiment data"""
        results = []
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split by the delimiter pattern that appears at the end of each run
        # Pattern: ================ WhichRun: X || Data: ... ==================
        delimiter_pattern = r'=+ WhichRun: (\d+) \|\| Data: ([\w_]+) \|\| Model: (\w+) \|\| Time: ([\d.]+) \|\| rand_state: (\d+) =+'
        
        # Find all delimiters and their positions
        delimiters = list(re.finditer(delimiter_pattern, content))
        
        if not delimiters:
            # Try alternative parsing approach for files with different format
            return []
        
        # Parse each run's content
        for i, delimiter_match in enumerate(delimiters):
            run_num = int(delimiter_match.group(1))
            dataset = delimiter_match.group(2)
            model = delimiter_match.group(3) 
            time = float(delimiter_match.group(4))
            rand_state = int(delimiter_match.group(5))
            
            # Get the content BEFORE this delimiter (belongs to this run)
            if i == 0:
                # For first run, take content from start to this delimiter
                run_content = content[:delimiter_match.start()]
            else:
                # For other runs, take content from previous delimiter to this delimiter
                prev_delimiter_end = delimiters[i-1].end()
                run_content = content[prev_delimiter_end:delimiter_match.start()]
            
            if not run_content.strip():
                continue
                
            # Extract model performance
            target_match = re.search(r'TargetModel Epoch: \d+, Approx Train: ([\d.]+), Train: ([\d.]+), Test: ([\d.]+),marco: ([\d.]+),micro: ([\d.]+)', run_content)
            shadow_match = re.search(r'ShadowModel Epoch: \d+, Approx Train: ([\d.]+), Train: ([\d.]+), Test: ([\d.]+),marco: ([\d.]+),micro: ([\d.]+)', run_content)
            
            # Extract attack performance
            attack_match = re.search(r'Test accuracy with Target Train InOut: ([\d.]+)\s+AUROC: ([\d.]+) precision: ([\d.]+) recall ([\d.]+) F1 score ([\d.]+)', run_content)
            member_nonmember_match = re.search(r'Test accuracy with Target Train In: ([\d.]+)\s+\|=====\|\s+Test accuracy with Target Train Out: ([\d.]+)', run_content)
            
            if all([target_match, shadow_match, attack_match, member_nonmember_match]):
                result = {
                    'run': run_num,
                    'dataset': dataset,
                    'model': model,
                    'time': time,
                    'rand_state': rand_state,
                    # Target model performance
                    'target_approx_train': float(target_match.group(1)),
                    'target_train': float(target_match.group(2)),
                    'target_test': float(target_match.group(3)),
                    'target_macro': float(target_match.group(4)),
                    'target_micro': float(target_match.group(5)),
                    # Shadow model performance
                    'shadow_approx_train': float(shadow_match.group(1)),
                    'shadow_train': float(shadow_match.group(2)),
                    'shadow_test': float(shadow_match.group(3)),
                    'shadow_macro': float(shadow_match.group(4)),
                    'shadow_micro': float(shadow_match.group(5)),
                    # Attack performance
                    'attack_accuracy': float(attack_match.group(1)),
                    'attack_auroc': float(attack_match.group(2)),
                    'attack_precision': float(attack_match.group(3)),
                    'attack_recall': float(attack_match.group(4)),
                    'attack_f1': float(attack_match.group(5)),
                    'member_accuracy': float(member_nonmember_match.group(1)),
                    'nonmember_accuracy': float(member_nonmember_match.group(2))
                }
                results.append(result)
        
        return results
    
    def load_all_results(self):
        """Load and parse all result files"""
        print("Loading results from all files...")
        
        for filepath in self.results_dir.glob("resultfile_*.txt"):
            filename = filepath.stem
            print(f"Processing {filename}...")
            
            # Extract experiment type and model from filename
            # Pattern: resultfile_TSTS_MODEL[_TYPE].txt
            parts = filename.split('_')
            if len(parts) < 3:
                continue
                
            model = parts[2]  # Third part is the model (GCN, GAT, etc.)
            
            # Determine experiment type
            if len(parts) > 3:
                if 'baseline' in parts[3]:
                    exp_type = 'baseline'
                elif 'nosynth' in parts[3]:
                    exp_type = 'nosynth'
                else:
                    exp_type = 'synthetic'
            else:
                exp_type = 'synthetic'  # Default for files without suffix
            
            results = self.parse_result_file(filepath)
            
            if results:  # Only add if we got valid results
                key = f"{exp_type}_{model}"
                self.data[key] = results
                print(f"  -> Loaded {len(results)} runs for {key}")
            else:
                print(f"  -> No valid results found in {filename}")
            
        print(f"Loaded {len(self.data)} experiment configurations")
    
    def compute_summary_stats(self):
        """Compute summary statistics for all experiments"""
        print("Computing summary statistics...")
        
        for key, results in self.data.items():
            if not results:
                continue
                
            df = pd.DataFrame(results)
            
            # Group by dataset if multiple datasets in same experiment
            if len(df['dataset'].unique()) > 1:
                grouped = df.groupby('dataset')
                self.summary_stats[key] = {}
                for dataset, group in grouped:
                    self.summary_stats[key][dataset] = {
                        'attack_accuracy_mean': group['attack_accuracy'].mean(),
                        'attack_accuracy_std': group['attack_accuracy'].std(),
                        'attack_auroc_mean': group['attack_auroc'].mean(),
                        'attack_auroc_std': group['attack_auroc'].std(),
                        'member_accuracy_mean': group['member_accuracy'].mean(),
                        'nonmember_accuracy_mean': group['nonmember_accuracy'].mean(),
                        'target_test_mean': group['target_test'].mean(),
                        'shadow_test_mean': group['shadow_test'].mean(),
                        'count': len(group)
                    }
            else:
                dataset = df['dataset'].iloc[0]
                self.summary_stats[key] = {
                    dataset: {
                        'attack_accuracy_mean': df['attack_accuracy'].mean(),
                        'attack_accuracy_std': df['attack_accuracy'].std(),
                        'attack_auroc_mean': df['attack_auroc'].mean(),
                        'attack_auroc_std': df['attack_auroc'].std(),
                        'member_accuracy_mean': df['member_accuracy'].mean(),
                        'nonmember_accuracy_mean': df['nonmember_accuracy'].mean(),
                        'target_test_mean': df['target_test'].mean(),
                        'shadow_test_mean': df['shadow_test'].mean(),
                        'count': len(df)
                    }
                }
    
    def create_attack_accuracy_comparison(self):
        """Create bar plot comparing attack accuracy across models and datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for synthetic experiments
        models = ['GCN', 'GAT', 'SAGE', 'SGC']
        datasets = ['Custom_Twitch', 'Custom_Event']
        
        twitch_means = []
        twitch_stds = []
        event_means = []
        event_stds = []
        
        for model in models:
            key = f"synthetic_{model}"
            if key in self.summary_stats:
                if 'Custom_Twitch' in self.summary_stats[key]:
                    twitch_means.append(self.summary_stats[key]['Custom_Twitch']['attack_accuracy_mean'])
                    twitch_stds.append(self.summary_stats[key]['Custom_Twitch']['attack_accuracy_std'])
                else:
                    twitch_means.append(0)
                    twitch_stds.append(0)
                
                if 'Custom_Event' in self.summary_stats[key]:
                    event_means.append(self.summary_stats[key]['Custom_Event']['attack_accuracy_mean'])
                    event_stds.append(self.summary_stats[key]['Custom_Event']['attack_accuracy_std'])
                else:
                    event_means.append(0)
                    event_stds.append(0)
            else:
                twitch_means.append(0)
                twitch_stds.append(0)
                event_means.append(0)
                event_stds.append(0)
        
        x = np.arange(len(models))
        width = 0.35
        
        # Twitch dataset
        bars1 = ax1.bar(x - width/2, twitch_means, width, yerr=twitch_stds, 
                       label='Custom_Twitch', alpha=0.8, capsize=5)
        bars2 = ax1.bar(x + width/2, event_means, width, yerr=event_stds,
                       label='Custom_Event', alpha=0.8, capsize=5)
        
        ax1.set_xlabel('GNN Architecture')
        ax1.set_ylabel('Attack Accuracy')
        ax1.set_title('Attack Accuracy by Architecture and Dataset')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Guess')
        ax1.set_ylim(0, 0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        # AUROC comparison
        twitch_auroc = []
        event_auroc = []
        
        for model in models:
            key = f"synthetic_{model}"
            if key in self.summary_stats:
                if 'Custom_Twitch' in self.summary_stats[key]:
                    twitch_auroc.append(self.summary_stats[key]['Custom_Twitch']['attack_auroc_mean'])
                else:
                    twitch_auroc.append(0)
                
                if 'Custom_Event' in self.summary_stats[key]:
                    event_auroc.append(self.summary_stats[key]['Custom_Event']['attack_auroc_mean'])
                else:
                    event_auroc.append(0)
            else:
                twitch_auroc.append(0)
                event_auroc.append(0)
        
        bars3 = ax2.bar(x - width/2, twitch_auroc, width, 
                       label='Custom_Twitch', alpha=0.8)
        bars4 = ax2.bar(x + width/2, event_auroc, width,
                       label='Custom_Event', alpha=0.8)
        
        ax2.set_xlabel('GNN Architecture')
        ax2.set_ylabel('Attack AUROC')
        ax2.set_title('Attack AUROC by Architecture and Dataset')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Guess')
        ax2.set_ylim(0, 0.7)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('attack_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_member_nonmember_analysis(self):
        """Analyze member vs non-member classification patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = ['GCN', 'GAT', 'SAGE', 'SGC']
        datasets = ['Custom_Twitch', 'Custom_Event']
        
        for i, model in enumerate(models):
            ax = axes[i//2, i%2]
            key = f"synthetic_{model}"
            
            if key not in self.data or not self.data[key]:
                ax.set_title(f'{model} - No Data')
                continue
            
            df = pd.DataFrame(self.data[key])
            
            # Create scatter plot of member vs non-member accuracy
            for dataset in datasets:
                data_subset = df[df['dataset'] == dataset]
                if not data_subset.empty:
                    ax.scatter(data_subset['member_accuracy'], 
                             data_subset['nonmember_accuracy'],
                             label=dataset, alpha=0.7, s=60)
            
            ax.set_xlabel('Member Accuracy')
            ax.set_ylabel('Non-Member Accuracy')
            ax.set_title(f'{model} - Member vs Non-Member Classification')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Add diagonal line (balanced accuracy)
            ax.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Balanced Line')
            
            # Add quadrant labels
            ax.text(0.1, 0.9, 'High Non-Member\nLow Member', fontsize=10, alpha=0.7)
            ax.text(0.9, 0.1, 'High Member\nLow Non-Member', fontsize=10, alpha=0.7)
            ax.text(0.9, 0.9, 'High Both', fontsize=10, alpha=0.7)
            ax.text(0.1, 0.1, 'Low Both', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('member_nonmember_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_synthetic_vs_real_comparison(self):
        """Compare synthetic vs real shadow data performance"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Debug: Print available summary stats
        print("DEBUG: Available summary stats:")
        for key in self.summary_stats.keys():
            datasets_in_key = list(self.summary_stats[key].keys())
            print(f"  {key}: {datasets_in_key}")
        
        # Models that have both synthetic and nosynth data
        available_models = []
        for model in ['GCN', 'GAT', 'SAGE', 'SGC']:
            if f"synthetic_{model}" in self.summary_stats and f"nosynth_{model}" in self.summary_stats:
                available_models.append(model)
        
        print(f"DEBUG: Available models for comparison: {available_models}")
        
        if not available_models:
            print("No models with both synthetic and nosynth data found")
            return
        
        datasets = ['Custom_Twitch', 'Custom_Event']
        
        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[dataset_idx]
            
            synthetic_accs = []
            real_accs = []
            model_names = []
            
            for model in available_models:
                synth_key = f"synthetic_{model}"
                real_key = f"nosynth_{model}"
                
                print(f"DEBUG: Checking {dataset} for {model}")
                print(f"  Synthetic key {synth_key} has datasets: {list(self.summary_stats[synth_key].keys()) if synth_key in self.summary_stats else 'KEY NOT FOUND'}")
                print(f"  Real key {real_key} has datasets: {list(self.summary_stats[real_key].keys()) if real_key in self.summary_stats else 'KEY NOT FOUND'}")
                
                synth_has_dataset = synth_key in self.summary_stats and dataset in self.summary_stats[synth_key]
                real_has_dataset = real_key in self.summary_stats and dataset in self.summary_stats[real_key]
                
                print(f"  Synthetic has {dataset}: {synth_has_dataset}")
                print(f"  Real has {dataset}: {real_has_dataset}")
                
                if synth_has_dataset and real_has_dataset:
                    synthetic_accs.append(self.summary_stats[synth_key][dataset]['attack_accuracy_mean'])
                    real_accs.append(self.summary_stats[real_key][dataset]['attack_accuracy_mean'])
                    model_names.append(model)
                    print(f"  -> Added {model} to comparison")
            
            if not synthetic_accs:
                ax.set_title(f'{dataset} - No Comparison Data')
                continue
            
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, synthetic_accs, width, 
                          label='Synthetic Shadow Data', alpha=0.8)
            bars2 = ax.bar(x + width/2, real_accs, width,
                          label='Real Shadow Data', alpha=0.8)
            
            ax.set_xlabel('GNN Architecture')
            ax.set_ylabel('Attack Accuracy')
            ax.set_title(f'{dataset}: Synthetic vs Real Shadow Data')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('synthetic_vs_real_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_performance_analysis(self):
        """Analyze target and shadow model classification performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = ['GCN', 'GAT', 'SAGE', 'SGC']
        
        for i, model in enumerate(models):
            ax = axes[i//2, i%2]
            key = f"synthetic_{model}"
            
            if key not in self.data or not self.data[key]:
                ax.set_title(f'{model} - No Data')
                continue
            
            df = pd.DataFrame(self.data[key])
            
            # Plot target vs shadow model performance
            for dataset in ['Custom_Twitch', 'Custom_Event']:
                data_subset = df[df['dataset'] == dataset]
                if not data_subset.empty:
                    ax.scatter(data_subset['target_test'], 
                             data_subset['shadow_test'],
                             label=f'{dataset}', alpha=0.7, s=60)
            
            ax.set_xlabel('Target Model Test Accuracy')
            ax.set_ylabel('Shadow Model Test Accuracy')
            ax.set_title(f'{model} - Target vs Shadow Model Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add diagonal line (perfect match)
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Match')
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_table(self):
        """Create a summary table of all results"""
        print("\n" + "="*80)
        print("SUMMARY TABLE: Attack Accuracy by Architecture and Dataset")
        print("="*80)
        
        # Create DataFrame for summary
        summary_data = []
        
        for exp_type in ['synthetic', 'nosynth', 'baseline']:
            for model in ['GCN', 'GAT', 'SAGE', 'SGC']:
                key = f"{exp_type}_{model}"
                if key in self.summary_stats:
                    for dataset, stats in self.summary_stats[key].items():
                        summary_data.append({
                            'Experiment': exp_type.title(),
                            'Model': model,
                            'Dataset': dataset.replace('Custom_', ''),
                            'Attack_Accuracy': f"{stats['attack_accuracy_mean']:.3f} ± {stats['attack_accuracy_std']:.3f}" if stats['attack_accuracy_std'] > 0 else f"{stats['attack_accuracy_mean']:.3f}",
                            'AUROC': f"{stats['attack_auroc_mean']:.3f}",
                            'Runs': stats['count']
                        })
        
        df_summary = pd.DataFrame(summary_data)
        
        if not df_summary.empty:
            # Pivot table for better visualization
            pivot_accuracy = df_summary.pivot_table(
                values='Attack_Accuracy', 
                index=['Model'], 
                columns=['Experiment', 'Dataset'], 
                aggfunc='first'
            )
            
            print("\nAttack Accuracy Results:")
            print(pivot_accuracy.to_string())
            
            # Save to CSV
            df_summary.to_csv('results_summary.csv', index=False)
            print(f"\nDetailed summary saved to: results_summary.csv")
        
        # Print key findings
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        
        # Find best and worst performing combinations
        if not df_summary.empty and 'Experiment' in df_summary.columns:
            synthetic_results = df_summary[df_summary['Experiment'] == 'Synthetic']
        else:
            synthetic_results = pd.DataFrame()
            
        if not synthetic_results.empty:
            # Extract numeric accuracy values
            synthetic_results['Numeric_Accuracy'] = synthetic_results['Attack_Accuracy'].apply(
                lambda x: float(x.split(' ±')[0]) if ' ±' in x else float(x)
            )
            
            best_result = synthetic_results.loc[synthetic_results['Numeric_Accuracy'].idxmax()]
            worst_result = synthetic_results.loc[synthetic_results['Numeric_Accuracy'].idxmin()]
            
            print(f"Most Vulnerable: {best_result['Model']} on {best_result['Dataset']} - {best_result['Attack_Accuracy']}")
            print(f"Most Resistant: {worst_result['Model']} on {worst_result['Dataset']} - {worst_result['Attack_Accuracy']}")
            
            # Architecture ranking
            arch_ranking = synthetic_results.groupby('Model')['Numeric_Accuracy'].mean().sort_values(ascending=False)
            print(f"\nArchitecture Vulnerability Ranking:")
            for i, (model, avg_acc) in enumerate(arch_ranking.items(), 1):
                print(f"{i}. {model}: {avg_acc:.3f} average attack accuracy")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive results analysis...")
        
        # Load and process data
        self.load_all_results()
        self.compute_summary_stats()
        
        # Create all visualizations
        print("\nCreating visualizations...")
        self.create_attack_accuracy_comparison()
        self.create_member_nonmember_analysis()
        self.create_synthetic_vs_real_comparison()
        self.create_model_performance_analysis()
        
        # Generate summary
        self.create_summary_table()
        
        print("\n" + "="*80)
        print("Analysis complete! Generated files:")
        print("- attack_performance_comparison.png")
        print("- member_nonmember_analysis.png") 
        print("- synthetic_vs_real_comparison.png")
        print("- model_performance_analysis.png")
        print("- results_summary.csv")
        print("="*80)

def main():
    """Main function to run the analysis"""
    analyzer = ResultsAnalyzer(".")
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()