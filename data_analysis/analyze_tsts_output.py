#!/usr/bin/env python3
"""
Analyze TSTS Output Files
Analyzes the posterior probabilities and model outputs to understand MIA performance
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os

def load_posteriors(file_path):
    """Load posterior probabilities from text file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse posteriors (assuming they're space-separated numbers)
        posteriors = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Try to parse as numbers
                try:
                    values = [float(x) for x in line.split()]
                    posteriors.extend(values)
                except ValueError:
                    continue
        
        return np.array(posteriors)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_tsts_results():
    """Analyze TSTS output files"""
    
    print("="*60)
    print("TSTS OUTPUT ANALYSIS")
    print("="*60)
    
    base_path = "/Users/prateek/notes/uni-project-dec/rebMIGraph"
    
    # File paths
    files = {
        'Target Train': f'{base_path}/posteriorsTargetTrain_TSTS_Custom_Twitch_GCN.txt',
        'Target Test': f'{base_path}/posteriorsTargetOut_TSTS_Custom_Twitch_GCN.txt',
        'Shadow Train': f'{base_path}/posteriorsShadowTrain_TSTS_Custom_Twitch_GCN.txt',
        'Shadow Test': f'{base_path}/posteriorsShadowOut_TSTS_Custom_Twitch_GCN.txt'
    }
    
    # Load all posteriors
    data = {}
    for name, path in files.items():
        if os.path.exists(path):
            posteriors = load_posteriors(path)
            if posteriors is not None and len(posteriors) > 0:
                data[name] = posteriors
                print(f"✅ Loaded {name}: {len(posteriors)} samples")
            else:
                print(f"❌ Failed to load {name}")
        else:
            print(f"⚠️  File not found: {name}")
    
    if not data:
        print("❌ No data loaded. Cannot perform analysis.")
        return
    
    # Basic statistics
    print("\n" + "="*40)
    print("POSTERIOR STATISTICS")
    print("="*40)
    
    stats_data = []
    for name, posteriors in data.items():
        stats_data.append({
            'Dataset': name,
            'Count': len(posteriors),
            'Mean': np.mean(posteriors),
            'Std': np.std(posteriors),
            'Min': np.min(posteriors),
            'Max': np.max(posteriors),
            'Median': np.median(posteriors)
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False, float_format='%.4f'))
    
    # Distribution comparisons
    print("\n" + "="*40)
    print("DISTRIBUTION COMPARISONS")
    print("="*40)
    
    # Compare target vs shadow (train)
    if 'Target Train' in data and 'Shadow Train' in data:
        target_train = data['Target Train']
        shadow_train = data['Shadow Train']
        
        # Statistical test
        ks_stat, ks_p = stats.ks_2samp(target_train, shadow_train)
        
        print(f"\nTarget Train vs Shadow Train:")
        print(f"  Mean difference: {np.mean(target_train) - np.mean(shadow_train):.4f}")
        print(f"  KS test statistic: {ks_stat:.4f}")
        print(f"  KS test p-value: {ks_p:.4f}")
        
        if ks_p < 0.05:
            print(f"  ⚠️  Significant distribution difference!")
        else:
            print(f"  ✅ Distributions are similar")
    
    # Compare target vs shadow (test)
    if 'Target Test' in data and 'Shadow Test' in data:
        target_test = data['Target Test']
        shadow_test = data['Shadow Test']
        
        # Statistical test
        ks_stat, ks_p = stats.ks_2samp(target_test, shadow_test)
        
        print(f"\nTarget Test vs Shadow Test:")
        print(f"  Mean difference: {np.mean(target_test) - np.mean(shadow_test):.4f}")
        print(f"  KS test statistic: {ks_stat:.4f}")
        print(f"  KS test p-value: {ks_p:.4f}")
        
        if ks_p < 0.05:
            print(f"  ⚠️  Significant distribution difference!")
        else:
            print(f"  ✅ Distributions are similar")
    
    # MIA-relevant analysis
    print("\n" + "="*40)
    print("MIA SIGNAL ANALYSIS")
    print("="*40)
    
    # For each model, compare train vs test (this is what MIA exploits)
    if 'Target Train' in data and 'Target Test' in data:
        target_train = data['Target Train']
        target_test = data['Target Test']
        
        mean_diff = np.mean(target_train) - np.mean(target_test)
        overlap = calculate_overlap(target_train, target_test)
        
        print(f"\nTarget Model (Member vs Non-member signal):")
        print(f"  Train mean: {np.mean(target_train):.4f}")
        print(f"  Test mean:  {np.mean(target_test):.4f}")
        print(f"  Difference: {mean_diff:.4f}")
        print(f"  Overlap:    {overlap:.2%}")
        
        if abs(mean_diff) < 0.01:
            print(f"  ⚠️  Very small difference - weak MIA signal!")
        elif overlap > 0.8:
            print(f"  ⚠️  High overlap - hard to distinguish members!")
        else:
            print(f"  ✅ Reasonable MIA signal")
    
    if 'Shadow Train' in data and 'Shadow Test' in data:
        shadow_train = data['Shadow Train']
        shadow_test = data['Shadow Test']
        
        mean_diff = np.mean(shadow_train) - np.mean(shadow_test)
        overlap = calculate_overlap(shadow_train, shadow_test)
        
        print(f"\nShadow Model (Member vs Non-member signal):")
        print(f"  Train mean: {np.mean(shadow_train):.4f}")
        print(f"  Test mean:  {np.mean(shadow_test):.4f}")
        print(f"  Difference: {mean_diff:.4f}")
        print(f"  Overlap:    {overlap:.2%}")
        
        if abs(mean_diff) < 0.01:
            print(f"  ⚠️  Very small difference - weak MIA signal!")
        elif overlap > 0.8:
            print(f"  ⚠️  High overlap - hard to distinguish members!")
        else:
            print(f"  ✅ Reasonable MIA signal")
    
    # Create visualizations
    print("\n" + "="*40)
    print("GENERATING VISUALIZATIONS")
    print("="*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TSTS Posterior Analysis - Custom Twitch GCN', fontsize=16)
    
    # Plot 1: All distributions
    ax = axes[0, 0]
    colors = ['blue', 'red', 'green', 'orange']
    for i, (name, posteriors) in enumerate(data.items()):
        ax.hist(posteriors, bins=30, alpha=0.6, label=name, color=colors[i % len(colors)])
    ax.set_title('Posterior Distributions')
    ax.set_xlabel('Posterior Probability')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 2: Target model comparison
    ax = axes[0, 1]
    if 'Target Train' in data and 'Target Test' in data:
        ax.hist(data['Target Train'], bins=30, alpha=0.6, label='Target Train (Members)', color='blue')
        ax.hist(data['Target Test'], bins=30, alpha=0.6, label='Target Test (Non-members)', color='red')
        ax.set_title('Target Model: Members vs Non-members')
        ax.set_xlabel('Posterior Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Plot 3: Shadow model comparison
    ax = axes[1, 0]
    if 'Shadow Train' in data and 'Shadow Test' in data:
        ax.hist(data['Shadow Train'], bins=30, alpha=0.6, label='Shadow Train (Members)', color='green')
        ax.hist(data['Shadow Test'], bins=30, alpha=0.6, label='Shadow Test (Non-members)', color='orange')
        ax.set_title('Shadow Model: Members vs Non-members')
        ax.set_xlabel('Posterior Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Plot 4: Cross-model comparison
    ax = axes[1, 1]
    if 'Target Train' in data and 'Shadow Train' in data:
        ax.hist(data['Target Train'], bins=30, alpha=0.6, label='Target Train', color='blue')
        ax.hist(data['Shadow Train'], bins=30, alpha=0.6, label='Shadow Train', color='green')
        ax.set_title('Cross-model Comparison (Training)')
        ax.set_xlabel('Posterior Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_path = '/Users/prateek/notes/uni-project-dec/data_analysis/tsts_posterior_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.show()
    
    # Final recommendations
    print("\n" + "="*40)
    print("RECOMMENDATIONS")
    print("="*40)
    
    issues = []
    
    # Check for weak MIA signals
    if 'Target Train' in data and 'Target Test' in data:
        target_diff = abs(np.mean(data['Target Train']) - np.mean(data['Target Test']))
        if target_diff < 0.01:
            issues.append("Target model shows weak member/non-member signal")
    
    if 'Shadow Train' in data and 'Shadow Test' in data:
        shadow_diff = abs(np.mean(data['Shadow Train']) - np.mean(data['Shadow Test']))
        if shadow_diff < 0.01:
            issues.append("Shadow model shows weak member/non-member signal")
    
    # Check for distribution mismatches
    if 'Target Train' in data and 'Shadow Train' in data:
        ks_stat, ks_p = stats.ks_2samp(data['Target Train'], data['Shadow Train'])
        if ks_p < 0.05:
            issues.append("Target and shadow models have different posterior distributions")
    
    if issues:
        print("\n🚨 ISSUES IDENTIFIED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("  1. Increase model overfitting (reduce regularization)")
        print("  2. Train longer to create clearer member/non-member differences")
        print("  3. Ensure shadow model uses similar data distribution")
        print("  4. Check if models are learning meaningful patterns")
        print("  5. Consider using different model architectures")
    else:
        print("\n✅ ANALYSIS LOOKS REASONABLE")
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        print("  1. Fine-tune attack model hyperparameters")
        print("  2. Use more sophisticated attack features")
        print("  3. Ensemble multiple shadow models")
        print("  4. Increase attack model complexity")
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*60}")

def calculate_overlap(dist1, dist2):
    """Calculate overlap between two distributions"""
    # Simple overlap calculation using histogram intersection
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    bins = np.linspace(min_val, max_val, 50)
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)
    
    # Normalize to probabilities
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate overlap
    overlap = np.sum(np.minimum(hist1, hist2))
    return overlap

if __name__ == "__main__":
    analyze_tsts_results()