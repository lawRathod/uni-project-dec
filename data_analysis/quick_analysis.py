#!/usr/bin/env python3
"""
Quick Data Analysis Script
Analyzes existing rebMI data from TSTS to identify distribution differences
"""

import sys
import os
sys.path.append('/Users/prateek/notes/uni-project-dec/rebMIGraph')

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import degree
from rebmi_adapter import create_inductive_split_custom

def analyze_rebmi_data(dataset_name='twitch', sample_ratio=0.2):
    """Analyze rebMI data distributions"""
    
    print(f"\n{'='*60}")
    print(f"QUICK DATA ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load data
        print(f"\nLoading {dataset_name} dataset with {sample_ratio*100}% sampling...")
        data = create_inductive_split_custom(
            dataset_name, 
            sample_ratio=sample_ratio,
            normalize=False,  # Don't normalize for analysis
            simple_features=True
        )
        
        print("✅ Data loaded successfully!")
        
        # Basic statistics
        print(f"\n{'='*40}")
        print("BASIC STATISTICS")
        print(f"{'='*40}")
        
        datasets = {
            'Target Train': data.target_x,
            'Target Test': data.target_test_x,
            'Shadow Train': data.shadow_x,
            'Shadow Test': data.shadow_test_x
        }
        
        for name, tensor in datasets.items():
            if tensor is not None:
                print(f"{name:15} - Shape: {list(tensor.shape)}, Mean: {tensor.mean():.3f}, Std: {tensor.std():.3f}")
        
        # Edge statistics
        print(f"\n{'='*40}")
        print("EDGE STATISTICS")
        print(f"{'='*40}")
        
        edge_datasets = {
            'Target Train': (data.target_edge_index, data.target_x.shape[0] if data.target_x is not None else 0),
            'Target Test': (data.target_test_edge_index, data.target_test_x.shape[0] if data.target_test_x is not None else 0),
            'Shadow Train': (data.shadow_edge_index, data.shadow_x.shape[0] if data.shadow_x is not None else 0),
            'Shadow Test': (data.shadow_test_edge_index, data.shadow_test_x.shape[0] if data.shadow_test_x is not None else 0)
        }
        
        for name, (edges, num_nodes) in edge_datasets.items():
            if edges is not None and num_nodes > 0:
                degrees = degree(edges[0], num_nodes)
                print(f"{name:15} - Edges: {edges.shape[1]:6d}, Avg Degree: {degrees.mean():.2f}")
        
        # Feature comparison
        print(f"\n{'='*40}")
        print("TARGET vs SHADOW COMPARISON")
        print(f"{'='*40}")
        
        if data.target_x is not None and data.shadow_x is not None:
            # Feature scale comparison
            target_scale = data.target_x.std().item()
            shadow_scale = data.shadow_x.std().item()
            scale_ratio = target_scale / shadow_scale if shadow_scale > 0 else float('inf')
            
            print(f"\nFeature Scale Analysis:")
            print(f"  Target std:     {target_scale:.4f}")
            print(f"  Shadow std:     {shadow_scale:.4f}")
            print(f"  Scale ratio:    {scale_ratio:.4f}")
            
            if scale_ratio > 2 or scale_ratio < 0.5:
                print(f"  ⚠️  WARNING: Large scale difference detected!")
            else:
                print(f"  ✅ Feature scales are reasonably similar")
            
            # Per-feature analysis
            print(f"\nPer-Feature Analysis (first 5 features):")
            n_features = min(5, data.target_x.shape[1])
            
            for i in range(n_features):
                target_feat = data.target_x[:, i]
                shadow_feat = data.shadow_x[:, i]
                
                target_mean = target_feat.mean().item()
                shadow_mean = shadow_feat.mean().item()
                target_std = target_feat.std().item()
                shadow_std = shadow_feat.std().item()
                
                print(f"  Feature {i}: Target({target_mean:.3f}±{target_std:.3f}) vs Shadow({shadow_mean:.3f}±{shadow_std:.3f})")
        
        # Degree comparison
        if data.target_edge_index is not None and data.shadow_edge_index is not None:
            target_degrees = degree(data.target_edge_index[0], data.target_x.shape[0])
            shadow_degrees = degree(data.shadow_edge_index[0], data.shadow_x.shape[0])
            
            target_mean_deg = target_degrees.mean().item()
            shadow_mean_deg = shadow_degrees.mean().item()
            
            print(f"\nDegree Distribution Analysis:")
            print(f"  Target mean degree: {target_mean_deg:.2f}")
            print(f"  Shadow mean degree: {shadow_mean_deg:.2f}")
            print(f"  Degree difference:  {abs(target_mean_deg - shadow_mean_deg):.2f}")
            
            if abs(target_mean_deg - shadow_mean_deg) > 5:
                print(f"  ⚠️  WARNING: Large degree difference detected!")
            else:
                print(f"  ✅ Degree distributions are reasonably similar")
        
        # Label analysis
        print(f"\n{'='*40}")
        print("LABEL DISTRIBUTION ANALYSIS")
        print(f"{'='*40}")
        
        label_datasets = {
            'Target Train': data.target_y,
            'Target Test': data.target_test_y,
            'Shadow Train': data.shadow_y,
            'Shadow Test': data.shadow_test_y
        }
        
        for name, labels in label_datasets.items():
            if labels is not None:
                unique, counts = torch.unique(labels, return_counts=True)
                print(f"{name:15} - Classes: {unique.tolist()}, Counts: {counts.tolist()}")
        
        # Create visualization
        print(f"\n{'='*40}")
        print("GENERATING VISUALIZATION")
        print(f"{'='*40}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Data Distribution Analysis - {dataset_name.upper()}', fontsize=16)
        
        # Feature distributions
        ax = axes[0, 0]
        if data.target_x is not None and data.shadow_x is not None:
            target_means = data.target_x.mean(0).numpy()
            shadow_means = data.shadow_x.mean(0).numpy()
            
            ax.hist(target_means, bins=15, alpha=0.6, label='Target', color='blue')
            ax.hist(shadow_means, bins=15, alpha=0.6, label='Shadow', color='red')
            ax.set_title('Feature Mean Distributions')
            ax.set_xlabel('Mean Value')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Degree distributions
        ax = axes[0, 1]
        if data.target_edge_index is not None and data.shadow_edge_index is not None:
            target_degrees = degree(data.target_edge_index[0], data.target_x.shape[0]).numpy()
            shadow_degrees = degree(data.shadow_edge_index[0], data.shadow_x.shape[0]).numpy()
            
            ax.hist(target_degrees, bins=20, alpha=0.6, label='Target', color='blue')
            ax.hist(shadow_degrees, bins=20, alpha=0.6, label='Shadow', color='red')
            ax.set_title('Degree Distributions')
            ax.set_xlabel('Degree')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Label distributions
        ax = axes[1, 0]
        if data.target_y is not None and data.shadow_y is not None:
            target_unique, target_counts = torch.unique(data.target_y, return_counts=True)
            shadow_unique, shadow_counts = torch.unique(data.shadow_y, return_counts=True)
            
            x = np.arange(len(target_unique))
            width = 0.35
            
            ax.bar(x - width/2, target_counts.numpy(), width, label='Target', color='blue', alpha=0.7)
            ax.bar(x + width/2, shadow_counts.numpy(), width, label='Shadow', color='red', alpha=0.7)
            ax.set_title('Class Distributions')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_xticks(x)
            ax.set_xticklabels(target_unique.numpy())
            ax.legend()
        
        # Feature scatter plot (first 2 features)
        ax = axes[1, 1]
        if data.target_x is not None and data.shadow_x is not None:
            # Sample for visualization
            n_samples = min(500, data.target_x.shape[0], data.shadow_x.shape[0])
            
            target_sample = data.target_x[:n_samples, :2].numpy()
            shadow_sample = data.shadow_x[:n_samples, :2].numpy()
            
            ax.scatter(target_sample[:, 0], target_sample[:, 1], alpha=0.6, label='Target', color='blue', s=10)
            ax.scatter(shadow_sample[:, 0], shadow_sample[:, 1], alpha=0.6, label='Shadow', color='red', s=10)
            ax.set_title('Feature Space (First 2 Features)')
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Feature 1')
            ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        save_path = f'/Users/prateek/notes/uni-project-dec/data_analysis/{dataset_name}_quick_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
        plt.show()
        
        # Summary
        print(f"\n{'='*40}")
        print("SUMMARY & RECOMMENDATIONS")
        print(f"{'='*40}")
        
        issues = []
        
        if data.target_x is not None and data.shadow_x is not None:
            target_scale = data.target_x.std().item()
            shadow_scale = data.shadow_x.std().item()
            scale_ratio = target_scale / shadow_scale if shadow_scale > 0 else float('inf')
            
            if scale_ratio > 2 or scale_ratio < 0.5:
                issues.append("Large feature scale difference between target and shadow")
        
        if data.target_edge_index is not None and data.shadow_edge_index is not None:
            target_degrees = degree(data.target_edge_index[0], data.target_x.shape[0])
            shadow_degrees = degree(data.shadow_edge_index[0], data.shadow_x.shape[0])
            
            if abs(target_degrees.mean() - shadow_degrees.mean()) > 5:
                issues.append("Large degree distribution difference")
        
        if issues:
            print("\n🚨 ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n💡 RECOMMENDATIONS:")
            print("  1. Improve synthetic data generation to match real data")
            print("  2. Apply consistent preprocessing to both datasets")
            print("  3. Consider using different sampling strategies")
            print("  4. Check if normalization is needed")
        else:
            print("\n✅ NO MAJOR ISSUES FOUND")
            print("\n💡 RECOMMENDATIONS:")
            print("  1. Data distributions look reasonable")
            print("  2. Focus on model training hyperparameters")
            print("  3. Ensure sufficient overfitting for MIA signal")
            print("  4. Consider increasing training epochs")
        
        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Quick data analysis')
    parser.add_argument('--dataset', type=str, default='twitch', choices=['twitch', 'event'])
    parser.add_argument('--sample_ratio', type=float, default=0.2)
    
    args = parser.parse_args()
    analyze_rebmi_data(args.dataset, args.sample_ratio)