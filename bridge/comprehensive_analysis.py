"""
Comprehensive dataset analysis script combining all visualizations.
Generates complete analysis of datasets including MIA failure analysis,
distribution comparisons, and statistical tests.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bridge import load_dataset
from bridge.data_converter import validate_data_consistency


def get_feature_names(dataset_name):
    """Get the actual feature names for a dataset."""
    if dataset_name == 'twitch':
        return ['views', 'life_time', 'created_at', 'updated_at', 'dead_account', 'language', 'affiliate']
    elif dataset_name == 'event':
        return ['locale', 'birthyear', 'joinedAt', 'timezone']
    else:
        return [f'Feature_{i}' for i in range(10)]  # fallback


def calculate_distribution_divergence(data1, data2):
    """Calculate various distribution divergence metrics."""
    # Wasserstein distance
    wd = wasserstein_distance(data1, data2)
    
    # KS test
    ks_stat, ks_p = stats.ks_2samp(data1, data2)
    
    # Anderson-Darling test
    try:
        ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([data1, data2])
        ad_p = 1.0 if ad_stat < ad_critical[2] else 0.001  # Rough p-value estimate
    except:
        ad_stat, ad_p = 0, 1.0
    
    return {
        'wasserstein': wd,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'ad_stat': ad_stat,
        'ad_p': ad_p
    }


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_comparison_table(datasets_dict, dataset_name):
    """Create a comparison table for all dataset types."""
    
    print_section(f"COMPARISON TABLE - {dataset_name.upper()}")
    
    comparison_data = []
    
    for name, data_list in datasets_dict.items():
        if not data_list:
            continue
        
        # Graph statistics
        num_graphs = len(data_list)
        num_nodes = [d.x.size(0) for d in data_list]
        num_edges = [d.edge_index.size(1) // 2 for d in data_list]
        
        # Feature statistics
        all_features = np.vstack([d.x.numpy() for d in data_list])
        num_features = all_features.shape[1]
        
        # Label statistics
        all_labels = []
        for d in data_list:
            all_labels.extend(d.y.tolist())
        label_counts = Counter(all_labels)
        
        # Density calculation
        densities = []
        for d in data_list:
            n = d.x.size(0)
            e = d.edge_index.size(1) // 2
            if n > 1:
                density = 2 * e / (n * (n - 1))
                densities.append(density)
        
        comparison_data.append({
            'Dataset': name.upper(),
            'Graphs': num_graphs,
            'Nodes (mean±std)': f"{np.mean(num_nodes):.0f}±{np.std(num_nodes):.0f}",
            'Edges (mean±std)': f"{np.mean(num_edges):.0f}±{np.std(num_edges):.0f}",
            'Density': f"{np.mean(densities):.3f}±{np.std(densities):.3f}",
            'Features': num_features,
            'Class 0': f"{label_counts[0]} ({100*label_counts[0]/len(all_labels):.1f}%)",
            'Class 1': f"{label_counts[1]} ({100*label_counts[1]/len(all_labels):.1f}%)",
            'Balance': f"{min(label_counts.values())/max(label_counts.values()):.3f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    return df


def statistical_comparison(datasets_dict, dataset_name):
    """Perform statistical tests between datasets."""
    
    print_section(f"STATISTICAL COMPARISON - {dataset_name.upper()}")
    
    # Extract features for each dataset
    features_dict = {}
    for name, data_list in datasets_dict.items():
        if data_list:
            all_features = np.vstack([d.x.numpy() for d in data_list])
            features_dict[name] = all_features
    
    if len(features_dict) < 2:
        print("  Not enough datasets for comparison")
        return
    
    # Compare feature distributions
    print("\n  Feature Distribution Comparison (KS test p-values):")
    print("  " + "-" * 60)
    
    dataset_names = list(features_dict.keys())
    num_features = features_dict[dataset_names[0]].shape[1]
    feature_names = get_feature_names(dataset_name)
    
    for i in range(min(num_features, 7)):  # Compare first 7 features
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature_{i}'
        print(f"\n  {feature_name}:")
        
        # Compare each pair of datasets
        for j, name1 in enumerate(dataset_names):
            for name2 in dataset_names[j+1:]:
                feat1 = features_dict[name1][:, i]
                feat2 = features_dict[name2][:, i]
                
                # KS test for distribution comparison
                ks_stat, p_value = stats.ks_2samp(feat1, feat2)
                
                status = "✓ Similar" if p_value > 0.05 else "✗ Different"
                print(f"    {name1} vs {name2}: p={p_value:.4f} {status}")
    
    # Compare edge distributions
    print("\n  Edge Count Distribution Comparison:")
    print("  " + "-" * 60)
    
    edge_counts_dict = {}
    for name, data_list in datasets_dict.items():
        if data_list:
            edge_counts = [d.edge_index.size(1) // 2 for d in data_list]
            edge_counts_dict[name] = edge_counts
    
    for j, name1 in enumerate(dataset_names):
        for name2 in dataset_names[j+1:]:
            edges1 = edge_counts_dict[name1]
            edges2 = edge_counts_dict[name2]
            
            # T-test for mean comparison
            t_stat, p_value = stats.ttest_ind(edges1, edges2)
            
            status = "✓ Similar" if p_value > 0.05 else "✗ Different"
            print(f"    {name1} vs {name2}: p={p_value:.4f} {status}")
            print(f"      Mean edges: {np.mean(edges1):.1f} vs {np.mean(edges2):.1f}")


def create_basic_comparison_plots(datasets_dict, dataset_name, output_dir="bridge/plots"):
    """Create basic comparison visualizations."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot
    num_datasets = len(datasets_dict)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Dataset Comparison: {dataset_name.upper()}', fontsize=16)
    
    colors = {'train': 'blue', 'test': 'orange', 'synthetic': 'red', 'synth': 'red'}
    
    for idx, (name, data_list) in enumerate(datasets_dict.items()):
        if not data_list:
            continue
        
        color = colors.get(name, 'gray')
        
        # 1. Edge distribution
        num_edges = [d.edge_index.size(1) // 2 for d in data_list]
        axes[0, 0].hist(num_edges, bins=20, alpha=0.7, label=f'{name.capitalize()}', color=color)
        
        # 2. Density distribution
        densities = []
        for d in data_list:
            n = d.x.size(0)
            e = d.edge_index.size(1) // 2
            if n > 1:
                density = 2 * e / (n * (n - 1))
                densities.append(density)
        
        axes[0, 1].hist(densities, bins=20, alpha=0.7, label=f'{name.capitalize()}', color=color)
        
        # 3. Label distribution
        all_labels = []
        for d in data_list:
            all_labels.extend(d.y.numpy().tolist())
        
        label_counts = Counter(all_labels)
        axes[0, 2].bar([f'{name}\nClass 0', f'{name}\nClass 1'], 
                      [label_counts[0], label_counts[1]], 
                      alpha=0.7, color=color)
    
    # Set titles and labels
    axes[0, 0].set_title('Edge Count Distribution')
    axes[0, 0].set_xlabel('Number of Edges')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Graph Density Distribution') 
    axes[0, 1].set_xlabel('Density')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Class Distribution')
    axes[0, 2].set_ylabel('Count')
    
    # Feature distributions for first 3 features
    feature_names = get_feature_names(dataset_name)
    
    for feat_idx in range(min(3, len(feature_names))):
        ax = axes[1, feat_idx]
        
        for name, data_list in datasets_dict.items():
            if not data_list:
                continue
                
            color = colors.get(name, 'gray')
            all_features = np.vstack([d.x.numpy() for d in data_list])
            feature_vals = all_features[:, feat_idx]
            
            ax.hist(feature_vals, bins=30, alpha=0.6, density=True,
                   label=f'{name.capitalize()}', color=color)
        
        feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature {feat_idx}'
        ax.set_title(f'{feature_name} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_path / f"{dataset_name}_basic_comparison.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"  Saved basic comparison plot: {output_file}")
    plt.close()


def create_mia_failure_analysis(datasets_dict, dataset_name, output_dir="bridge/plots"):
    """Create comprehensive MIA failure analysis plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get data for analysis
    real_features = []
    synthetic_features = []
    
    if 'train' in datasets_dict and 'test' in datasets_dict:
        for data_list in [datasets_dict['train'], datasets_dict['test']]:
            for data in data_list:
                real_features.append(data.x.numpy())
    
    # Handle both 'synthetic' and 'synth' keys
    synth_key = 'synthetic' if 'synthetic' in datasets_dict else 'synth'
    if synth_key in datasets_dict:
        for data in datasets_dict[synth_key]:
            synthetic_features.append(data.x.numpy())
    
    if not real_features or not synthetic_features:
        print("  Insufficient data for MIA failure analysis")
        return []
    
    real_features = np.vstack(real_features)
    synthetic_features = np.vstack(synthetic_features)
    
    # Calculate divergence scores
    feature_names = get_feature_names(dataset_name)
    divergence_scores = []
    
    for i in range(real_features.shape[1]):
        div_metrics = calculate_distribution_divergence(
            real_features[:, i], synthetic_features[:, i])
        divergence_scores.append(div_metrics['wasserstein'])
    
    # Create comprehensive MIA failure analysis plot
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 0.6])
    
    fig.suptitle(f'Why MIA Attack Fails with Synthetic Data - {dataset_name.upper()}\n'
                 f'Complete Analysis of Domain Mismatch Issues', fontsize=18, y=0.98)
    
    # 1. Feature divergence radar chart
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    if divergence_scores:
        angles = np.linspace(0, 2 * np.pi, len(divergence_scores), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Normalize divergence scores for radar
        max_div = max(divergence_scores) if divergence_scores else 1
        normalized_divs = [d / max_div for d in divergence_scores] + [divergence_scores[0] / max_div]
        
        ax1.plot(angles, normalized_divs, 'o-', linewidth=2, color='red', markersize=8)
        ax1.fill(angles, normalized_divs, alpha=0.25, color='red')
        ax1.set_xticks(angles[:-1])
        
        # Use actual feature names for radar chart
        radar_labels = [name[:8] + '..' if len(name) > 8 else name for name in feature_names[:len(divergence_scores)]]
        ax1.set_xticklabels(radar_labels)
        
        ax1.set_ylim(0, 1)
        ax1.set_title('Feature Divergence\n(Wasserstein Distance)', fontsize=12, pad=20)
        ax1.grid(True)
        
        # Add interpretation
        avg_divergence = np.mean(divergence_scores)
        if avg_divergence > 0.5:
            interpretation = "SEVERE MISMATCH"
            color = 'red'
        elif avg_divergence > 0.2:
            interpretation = "MODERATE MISMATCH"  
            color = 'orange'
        else:
            interpretation = "GOOD MATCH"
            color = 'green'
        
        ax1.text(0.5, -0.15, f'Overall: {interpretation}', transform=ax1.transAxes,
                ha='center', fontweight='bold', color=color, fontsize=10)
    
    # 2. 2D PCA visualization of domain gap
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Combine data for PCA
    all_data = np.vstack([real_features, synthetic_features])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data)
    
    real_pca = pca_result[:len(real_features)]
    synth_pca = pca_result[len(real_features):]
    
    ax2.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, s=1, color='blue', label='Real Data')
    ax2.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, s=1, color='red', label='Synthetic Data')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    ax2.set_title('Domain Gap Visualization\n(Separation = Poor Transfer)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Class distribution comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    class_ratios = []
    dataset_names = []
    
    for name, data_list in datasets_dict.items():
        if not data_list:
            continue
        all_labels = []
        for data in data_list:
            all_labels.extend(data.y.tolist())
        
        label_counts = Counter(all_labels)
        ratio = min(label_counts.values()) / max(label_counts.values()) if label_counts else 0
        class_ratios.append(ratio)
        dataset_names.append(name.capitalize())
        
        # Bar plot
        colors_bar = {'Train': 'blue', 'Test': 'orange', 'Synth': 'red', 'Synthetic': 'red'}
        color = colors_bar.get(name.capitalize(), 'gray')
        ax3.bar(len(class_ratios)-1, ratio, color=color, alpha=0.7, 
               label=f'{name.capitalize()}: {ratio:.3f}')
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Class Balance Ratio')
    ax3.set_title('Class Balance Comparison\n(Imbalance affects MIA)')
    ax3.set_xticks(range(len(dataset_names)))
    ax3.set_xticklabels(dataset_names)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Balance')
    ax3.legend()
    
    # 4. Edge distribution impact
    ax4 = fig.add_subplot(gs[0, 3])
    
    edge_data = []
    edge_labels = []
    colors_edge = []
    
    for name, data_list in datasets_dict.items():
        if data_list:
            edges = [d.edge_index.size(1) // 2 for d in data_list]
            edge_data.append(edges)
            edge_labels.append(name.capitalize())
            color_map = {'train': 'blue', 'test': 'orange', 'synthetic': 'red', 'synth': 'red'}
            colors_edge.append(color_map.get(name, 'gray'))
    
    if edge_data:
        bp = ax4.boxplot(edge_data, tick_labels=edge_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_edge):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Number of Edges')
    ax4.set_title('Edge Distribution\n(Structure affects GNN learning)')
    ax4.grid(True, alpha=0.3)
    
    # 5-8. Feature distribution comparisons (most problematic features)
    if divergence_scores:
        # Get indices of most problematic features
        problem_features = np.argsort(divergence_scores)[-4:][::-1]  # Top 4 worst
        
        for i, feat_idx in enumerate(problem_features):
            row = 1 + i // 2
            col = i % 2
            
            ax = fig.add_subplot(gs[row, col])
            
            real_feat = real_features[:, feat_idx]
            synth_feat = synthetic_features[:, feat_idx]
            
            # Create histograms
            bins = np.linspace(min(real_feat.min(), synth_feat.min()), 
                             max(real_feat.max(), synth_feat.max()), 50)
            
            ax.hist(real_feat, bins=bins, alpha=0.6, density=True, color='blue', 
                   label=f'Real (μ={real_feat.mean():.3f})')
            ax.hist(synth_feat, bins=bins, alpha=0.6, density=True, color='red',
                   label=f'Synthetic (μ={synth_feat.mean():.3f})')
            
            feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
            ax.set_title(f'{feature_name} - Divergence: {divergence_scores[feat_idx]:.4f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add problem indicator
            if divergence_scores[feat_idx] > 0.5:
                ax.text(0.02, 0.95, 'CRITICAL', transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor="red", alpha=0.8),
                       color='white', fontweight='bold')
    
    # 9. MIA Attack Process Diagram (spans full width)
    ax9 = fig.add_subplot(gs[3, :4])
    ax9.axis('off')
    
    # Split the bottom area into two columns for text
    # Left side - Attack Process
    process_text = """MIA ATTACK PROCESS & FAILURE POINTS

1) SHADOW MODEL TRAINING (on Synthetic Data)
   • Learns patterns from synthetic graphs
   • Feature distributions != real data
   • Graph structure != real data
   FAILURE: Wrong patterns learned

2) ATTACK MODEL TRAINING
   • Uses synthetic-trained shadow posteriors
   • Learns to distinguish synthetic member/non-member patterns
   FAILURE: Patterns don't exist in real data

3) TARGET MODEL ATTACK (on Real Data)  
   • Attack model expects synthetic patterns
   • Real data has different distributions
   • No transferable membership signals
   RESULT: Random performance (AUROC ≈ 0.5)
"""
    
    ax9.text(0.02, 0.98, process_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # Right side - Solutions
    solution_text = """REQUIRED SOLUTIONS

IMMEDIATE FIXES:
• Feature distribution alignment (quantile matching)
• Graph structure calibration (edge density matching)
• Class balance correction

ADVANCED TECHNIQUES:
• Domain adaptation loss functions
• Adversarial training for synthetic data
• Multi-domain MIA architectures
• Progressive difficulty curriculum learning

EXPECTED IMPROVEMENTS:
• AUROC: 0.5 → 0.7+ 
• Precision: Random → Meaningful
• Attack transferability: Poor → Good

WITHOUT FIXES: MIA remains ineffective
"""
    
    ax9.text(0.52, 0.98, solution_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"{dataset_name}_mia_failure_complete_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved complete MIA failure analysis: {output_file}")
    plt.close()
    
    return divergence_scores


def create_detailed_feature_analysis(datasets_dict, dataset_name, output_dir="bridge/plots"):
    """Create detailed feature mismatch analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get real data (combine train and test)
    real_data = []
    synthetic_data = []
    
    if 'train' in datasets_dict and 'test' in datasets_dict:
        for data_list in [datasets_dict['train'], datasets_dict['test']]:
            for data in data_list:
                real_data.append(data.x.numpy())
    
    # Handle both 'synthetic' and 'synth' keys
    synth_key = 'synthetic' if 'synthetic' in datasets_dict else 'synth'
    if synth_key in datasets_dict:
        for data in datasets_dict[synth_key]:
            synthetic_data.append(data.x.numpy())
    
    if not real_data or not synthetic_data:
        print("  Insufficient data for detailed feature analysis")
        return []
    
    real_features = np.vstack(real_data)
    synthetic_features = np.vstack(synthetic_data)
    num_features = real_features.shape[1]
    
    # Get actual feature names
    feature_names = get_feature_names(dataset_name)
    
    # Create comprehensive feature mismatch plot
    fig = plt.figure(figsize=(20, 5 * ((num_features + 1) // 2)))
    gs = GridSpec((num_features + 1) // 2, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle(f'Critical Feature Distribution Mismatches - {dataset_name.upper()}\n'
                 f'Why Synthetic Shadow Models Fail in MIA Attacks', fontsize=16, y=0.98)
    
    divergence_scores = []
    
    for feat_idx in range(num_features):
        row = feat_idx // 2
        col_start = (feat_idx % 2) * 2
        
        real_feat = real_features[:, feat_idx]
        synth_feat = synthetic_features[:, feat_idx]
        
        # Calculate divergence metrics
        div_metrics = calculate_distribution_divergence(real_feat, synth_feat)
        divergence_scores.append(div_metrics['wasserstein'])
        
        # Histogram comparison
        ax1 = fig.add_subplot(gs[row, col_start])
        
        # Plot histograms
        bins = np.linspace(min(real_feat.min(), synth_feat.min()), 
                          max(real_feat.max(), synth_feat.max()), 50)
        
        ax1.hist(real_feat, bins=bins, alpha=0.7, density=True, 
                label=f'Real (μ={real_feat.mean():.3f}, σ={real_feat.std():.3f})', 
                color='blue', edgecolor='black')
        ax1.hist(synth_feat, bins=bins, alpha=0.7, density=True,
                label=f'Synthetic (μ={synth_feat.mean():.3f}, σ={synth_feat.std():.3f})',
                color='red', edgecolor='black')
        
        feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
        ax1.set_title(f'{feature_name} Distribution Mismatch\n'
                     f'Wasserstein Distance: {div_metrics["wasserstein"]:.4f}\n'
                     f'KS p-value: {div_metrics["ks_p"]:.2e}', fontsize=12)
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add significance indicator
        if div_metrics['ks_p'] < 0.001:
            ax1.text(0.02, 0.95, 'CRITICAL MISMATCH', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                    fontweight='bold', fontsize=10, color='white')
        elif div_metrics['ks_p'] < 0.05:
            ax1.text(0.02, 0.95, 'Significant Mismatch', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8),
                    fontweight='bold', fontsize=10)
        
        # QQ plot
        ax2 = fig.add_subplot(gs[row, col_start + 1])
        
        # Create QQ plot
        real_sorted = np.sort(real_feat)
        synth_sorted = np.sort(synth_feat)
        
        # Interpolate to same length
        min_len = min(len(real_sorted), len(synth_sorted))
        real_quantiles = np.interp(np.linspace(0, 1, min_len), 
                                  np.linspace(0, 1, len(real_sorted)), real_sorted)
        synth_quantiles = np.interp(np.linspace(0, 1, min_len),
                                   np.linspace(0, 1, len(synth_sorted)), synth_sorted)
        
        ax2.scatter(real_quantiles, synth_quantiles, alpha=0.5, s=1)
        
        # Add diagonal line
        min_val = min(real_quantiles.min(), synth_quantiles.min())
        max_val = max(real_quantiles.max(), synth_quantiles.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Match')
        
        ax2.set_xlabel('Real Data Quantiles')
        ax2.set_ylabel('Synthetic Data Quantiles')
        ax2.set_title(f'Q-Q Plot: {feature_name}\n'
                     f'Deviation from diagonal = Distribution shift')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / f"{dataset_name}_feature_mismatch_detailed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved detailed feature mismatch plot: {output_file}")
    plt.close()
    
    return divergence_scores


def main():
    """Main comprehensive analysis function."""
    
    print("=" * 80)
    print("  COMPREHENSIVE DATASET ANALYSIS")
    print("  • Basic comparisons and statistics")
    print("  • MIA failure analysis") 
    print("  • Detailed feature analysis")
    print("=" * 80)
    
    for dataset_name in ['twitch', 'event']:
        print(f"\n{'#' * 80}")
        print(f"  ANALYZING {dataset_name.upper()} DATASET")
        print('#' * 80)
        
        # Load datasets
        datasets = {}
        for data_type, display_name in [('train', 'training'), ('nontrain', 'test'), ('synth', 'synthetic')]:
            try:
                data = load_dataset(dataset_name, data_type)
                if data_type == 'nontrain':
                    datasets['test'] = data
                else:
                    datasets[data_type] = data
                print(f"  ✓ Loaded {len(data)} {display_name} graphs")
            except Exception as e:
                print(f"  ✗ Failed to load {display_name} data: {e}")
        
        if len(datasets) < 2:
            print("  Insufficient data for analysis")
            continue
        
        # 1. Create comparison table and statistics
        comparison_df = create_comparison_table(datasets, dataset_name)
        statistical_comparison(datasets, dataset_name)
        
        # 2. Create basic comparison plots
        print(f"\n  Creating basic comparison plots...")
        create_basic_comparison_plots(datasets, dataset_name)
        
        # 3. Create detailed feature analysis
        print(f"  Creating detailed feature analysis...")
        detailed_divergence = create_detailed_feature_analysis(datasets, dataset_name)
        
        # 4. Create MIA failure analysis
        print(f"  Creating MIA failure analysis...")
        mia_divergence = create_mia_failure_analysis(datasets, dataset_name)
        
        # 5. Validate data compatibility
        validation = validate_data_consistency(
            datasets.get('train', []) + datasets.get('synth', []) + datasets.get('test', [])
        )
        print_section(f"Data Validation - {dataset_name}")
        print(f"  Valid: {validation['valid']}")
        if not validation['valid']:
            print(f"  Reason: {validation.get('reason', 'Unknown')}")
        else:
            print(f"  Feature dimension: {validation['feature_dim']}")
            print(f"  Number of classes: {validation['num_classes']}")
            print(f"  Number of graphs: {validation['num_graphs']}")
        
        # 6. Save comparison table to CSV
        output_path = Path("bridge/plots")
        csv_file = output_path / f"{dataset_name}_comparison.csv"
        comparison_df.to_csv(csv_file, index=False)
        print(f"\n  Saved comparison table: {csv_file}")
        
        print(f"  ✅ Complete analysis created for {dataset_name}")
    
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE ANALYSIS COMPLETE")
    print("  Generated plots:")
    print("  • Basic comparison plots (*_basic_comparison.png)")
    print("  • Detailed feature analysis (*_feature_mismatch_detailed.png)")
    print("  • Complete MIA failure analysis (*_mia_failure_complete_analysis.png)")
    print("  • CSV comparison tables (*_comparison.csv)")
    print("=" * 80)


if __name__ == "__main__":
    main()