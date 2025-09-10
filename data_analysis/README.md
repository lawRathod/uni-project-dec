# Data Analysis for Membership Inference Attack

This directory contains analytical tools to understand why MIA attacks may fail and to compare synthetic vs real data distributions.

## Scripts

### `analyze_data_distributions.py`
Comprehensive analysis script that compares synthetic and real training data to identify distribution mismatches.

#### Features:
- **Basic Statistics**: Node counts, edge counts, feature dimensions
- **Feature Distribution Analysis**: Statistical comparisons, KS tests
- **Graph Structure Analysis**: Degree distributions, graph density
- **Class Separability Analysis**: Inter-class feature differences
- **Visualizations**: Histograms, correlations, PCA plots
- **Automated Report**: Key findings and recommendations

#### Usage:

```bash
# Analyze Twitch dataset with 20% sampling
python analyze_data_distributions.py --dataset twitch --sample_ratio 0.2

# Analyze Event dataset with full data
python analyze_data_distributions.py --dataset event --sample_ratio 1.0

# Quick analysis with minimal data
python analyze_data_distributions.py --dataset twitch --sample_ratio 0.05
```

#### Output:
- Console output with detailed statistics
- Visualization plots saved as PNG files
- Key findings and recommendations

## What This Analysis Helps Identify

### 1. **Distribution Mismatches**
- Feature scale differences between synthetic and real data
- Different statistical properties (mean, std, skewness)
- Class distribution imbalances

### 2. **Graph Structure Differences**
- Degree distribution mismatches
- Different connectivity patterns
- Graph density variations

### 3. **Feature Quality Issues**
- Poor feature separability between classes
- High correlation between synthetic features
- Missing important structural properties

### 4. **MIA Attack Failure Reasons**
- Shadow model learning different patterns than target model
- Synthetic data not representative of real data
- Insufficient distinguishable signal between members/non-members

## Interpreting Results

### Good Signs ✅
- Similar feature scales between target and shadow
- Comparable degree distributions
- Similar class separability metrics
- Overlapping PCA clusters

### Warning Signs ⚠️
- Large scale differences (>2x) between datasets
- Different number of classes
- Non-overlapping PCA clusters
- Significant KS test results (p < 0.05)

### Critical Issues ❌
- Completely different feature ranges
- Isolated nodes in one dataset but not the other
- One dataset has much higher/lower connectivity
- Clear separation in PCA space

## Recommendations Based on Analysis

### If Synthetic Data Quality is Poor:
1. Improve synthetic graph generation
2. Add more realistic noise patterns
3. Match degree distributions to real data
4. Ensure feature scales are consistent

### If Data Alignment is Good:
1. Focus on model hyperparameters
2. Increase training epochs for better overfitting
3. Adjust learning rates and regularization
4. Try different model architectures

### For Better MIA Performance:
1. Use more training data (increase sample_ratio)
2. Ensure models overfit sufficiently
3. Add more discriminative features
4. Train multiple shadow models

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn torch torch-geometric
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure you're running from the project root directory
2. **Memory Issues**: Reduce sample_ratio for large datasets
3. **Missing Data**: Check that dataset files exist and are properly formatted
4. **Visualization Errors**: Install matplotlib and seaborn properly

### Performance Tips:

- Use sample_ratio=0.1 for quick analysis
- Use sample_ratio=1.0 for comprehensive analysis
- Large datasets may require more memory and time

## Example Output

```
==============================================================
BASIC STATISTICS ANALYSIS
==============================================================

1. Node Count Statistics:
----------------------------------------
Target Train    - Nodes:   1532, Features:   5
Target Test     - Nodes:    766, Features:   5
Shadow Train    - Nodes:   1024, Features:   5
Shadow Test     - Nodes:    512, Features:   5

🔍 Key Findings:
----------------------------------------
⚠️  Large scale difference: Target std=2.345, Shadow std=0.891
⚠️  Large degree difference: Target=15.2, Shadow=8.7

📊 Recommendations:
----------------------------------------
1. Consider improving synthetic data generation to better match real data
2. Apply consistent preprocessing to both target and shadow data
```