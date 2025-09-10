# rebMIGraph Technical Documentation

Core implementation of membership inference attacks on GNNs using TSTS (Train on Subgraph, Test on Subgraph) methodology.

## Architecture

```
Pickle Data → bridge.py → rebmi_adapter.py → TSTS.py → Attack Results
             (Convert)    (Feature Eng.)     (Attack)
```

## Core Components

### TSTS.py - Attack Pipeline
**Main Configuration (Lines 72-84):**
```python
model_type = "GCN"              # GCN/GAT/SAGE/SGC
data_type = "Custom_Twitch"     # Dataset selection
CUSTOM_SUBSET_RATIO = 0.3       # Use 30% of data
CUSTOM_ADVANCED_FEATURES = True # Enhanced features
USE_IMPROVED_ATTACK = True      # Better attack model
```

**Attack Workflow:**
1. Load dataset → Create inductive splits
2. Train target model (real data) + shadow model (synthetic data)
3. Generate posteriors → Train attack classifier
4. Evaluate: Accuracy, Precision, Recall, F1, AUC-ROC

**Key Improvements:**
- Lines 1670-1747: `ImprovedAttackModel` - 5-layer network with BatchNorm/Dropout
- Lines 132-150: Optimized hyperparameters for attack performance
- Lines 805-843: Custom dataset NeighborSampler handling

### bridge.py - Data Conversion
**PickleDatasetLoader Class:**
```python
convert_to_pyg(df, graph):
  - Extract features from DataFrame
  - Convert NetworkX → PyTorch Geometric
  - Map labels: 'affiliate'/'gender' → numeric
  - Return Data(x, edge_index, y)
```

**Handles:** `(pandas.DataFrame, networkx.Graph)` → `torch_geometric.Data`

### rebmi_adapter.py - Feature Engineering

**Advanced Features (Lines 142-290):**
```python
enhance_features_advanced():
  # Structural features (15+ total)
  - Degree features: in/out/total, normalized
  - Clustering: triangles, coefficients  
  - Centrality: hub/leaf/isolated indicators
  - Ego-network: neighbor degree averages
  - Feature interactions: degree-weighted features
  
  # Memory optimization
  - Dense matrix ops only for graphs <10k nodes
  - Fallback to approximations for large graphs
```

**Data Organization:**
```python
RebMIData:
  target_x/y/edge_index      # Real train
  target_test_x/y/edge_index # Real test
  shadow_x/y/edge_index      # Synth train
  shadow_test_x/y/edge_index # Synth test
```

**Subset Selection (Lines 70-97):**
- Configurable ratio/max for performance tuning
- Random sampling with fixed seed for reproducibility

## Performance Optimizations

### Memory Efficiency
- **Large Graphs (>10k nodes)**: Automatic switch to approximation methods
- **Batch Processing**: Combined subgraphs for efficient GPU utilization
- **Smart Normalization**: Global statistics prevent data leakage

### Feature Engineering Impact
```
Basic (degree only):     ~2 features  → 70% attack accuracy
Advanced (15+ features): ~20 features → 85% attack accuracy
```

### Attack Model Improvements
```
Original: 3-layer FC → 70-75% accuracy
Improved: 5-layer + BatchNorm + Dropout → 80-85% accuracy
```

## Configuration Guide

### Quick Experiments
```python
CUSTOM_SUBSET_RATIO = 0.1  # 10% data for rapid testing
CUSTOM_MAX_SUBGRAPHS = 5   # Or fixed 5 subgraphs
```

### Best Accuracy
```python
CUSTOM_SUBSET_RATIO = 1.0        # Full data
CUSTOM_ADVANCED_FEATURES = True  # All features
USE_IMPROVED_ATTACK = True       # Enhanced model
```

### Model-Specific Settings
- **GCN**: Smaller architecture for custom datasets (32 hidden vs 256)
- **SAGE**: 2-layer with NeighborSampler, batch size 64
- **GAT**: 8 attention heads, dataset-specific configurations

## Technical Details

### Inductive Split Creation
- **Target Model**: Trained on real subgraphs
- **Shadow Model**: Trained on synthetic subgraphs (50/50 train/test split)
- **Node Offset**: Maintains graph structure when combining subgraphs

### Feature Normalization
```python
# Global normalization across all splits
combined_features = torch.cat([target, shadow, test])
mean, std = combined_features.mean(), combined_features.std()
# Apply same transform to all data
```

### Attack Training
- **Optimizer**: Adam with weight decay (1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Loss**: CrossEntropyLoss
- **Epochs**: 150 (configurable)

## Output Metrics

```
Target Model: Train/Test accuracy
Shadow Model: Train/Test accuracy  
Attack Model: 
  - Accuracy: Overall correct predictions
  - Precision: True positives / (True + False positives)
  - Recall: True positives / (True positives + False negatives)
  - F1: Harmonic mean of precision and recall
  - AUC-ROC: Area under ROC curve
```

## Debugging Tips

1. **Memory Issues**: Reduce `CUSTOM_SUBSET_RATIO` or `BATCH_SIZE_NEIGHBOR`
2. **Poor Attack Performance**: Enable `CUSTOM_ADVANCED_FEATURES`
3. **Training Instability**: Lower learning rates, check feature normalization
4. **Custom Dataset Errors**: Verify pickle format matches bridge.py expectations

## File Line References

Key configuration locations for quick modifications:

- **Model/Dataset**: TSTS.py lines 72-73
- **Performance Settings**: TSTS.py lines 76-84
- **Attack Hyperparameters**: TSTS.py lines 132-150
- **Feature Engineering**: rebmi_adapter.py lines 142-290
- **Data Loading**: bridge.py lines 28-50