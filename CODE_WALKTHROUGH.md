# Code Walkthrough: Membership Inference Attacks on Graph Neural Networks

This document provides a high-level walkthrough of the codebase architecture and execution flow for implementing membership inference attacks (MIA) on graph neural networks.

## Overview

This project implements membership inference attacks that attempt to determine whether a specific data point was used to train a machine learning model. For graph neural networks, this involves analyzing model confidence patterns to infer membership status of nodes or subgraphs.

## Execution Flow

### 1. Data Pipeline (`bridge.py` → `rebmi_adapter.py`)

**Input**: Custom datasets stored as pickle files containing (DataFrame, NetworkX Graph) tuples
**Output**: PyTorch Geometric data structures ready for GNN training

```
Pickle Files → bridge.py → rebmi_adapter.py → TSTS.py
    ↓              ↓              ↓             ↓
Raw Tuples → PyG Objects → Split Data → MIA Attack
```

#### bridge.py
- **Purpose**: Convert heterogeneous data formats to standardized PyTorch Geometric
- **Key Function**: `load_custom_dataset(dataset_name)`
- **Process**:
  1. Loads pickle files from `../datasets/{name}/{name}_{split}.pickle`
  2. Converts NetworkX graphs to PyTorch Geometric format
  3. Extracts node features from DataFrame columns
  4. Maps target labels (affiliate/gender) to tensor format
  5. Returns unified dataset dictionary with train/nontrain/synth splits

#### rebmi_adapter.py  
- **Purpose**: Create proper inductive splits for membership inference
- **Key Function**: `create_inductive_split_custom(dataset_name)`
- **Process**:
  1. Combines multiple subgraphs into single large graphs
  2. Carefully offsets edge indices to maintain connectivity
  3. Creates separate data for target vs shadow models:
     - Target: trains on real data, tests on real non-training data  
     - Shadow: trains on synthetic data, tests on synthetic data
  4. Applies feature engineering (node degrees, normalization)
  5. Returns `RebMIData` object with all splits prepared

### 2. Model Training and Attack Execution (`TSTS.py`)

**Main execution happens in TSTS.py through several phases:**

#### Phase 1: Configuration and Data Loading
```python
# Hyperparameters (lines 67-133)
model_type = "GCN"  
data_type = "Custom_Twitch"
mode = "TSTS"

# Data loading
if data_type.startswith("Custom_"):
    rebmi_data = create_inductive_split_custom(dataset_name)
```

#### Phase 2: Target Model Training  
```python
# Train target model on real training data
target_model = Net(dataset, model_type)
train_target_model(target_model, rebmi_data.target_x, rebmi_data.target_y, ...)

# Test on both training and non-training data to get posteriors
target_train_posteriors = test_model(target_model, rebmi_data.target_x, ...)
target_test_posteriors = test_model(target_model, rebmi_data.target_test_x, ...)
```

#### Phase 3: Shadow Model Training
```python  
# Train shadow model on synthetic data (mimics target model)
shadow_model = Net(dataset, model_type)  
train_shadow_model(shadow_model, rebmi_data.shadow_x, rebmi_data.shadow_y, ...)

# Test on both synthetic training and test data
shadow_train_posteriors = test_model(shadow_model, rebmi_data.shadow_x, ...)
shadow_test_posteriors = test_model(shadow_model, rebmi_data.shadow_test_x, ...)
```

#### Phase 4: Attack Model Training
```python
# Prepare attack training data from posteriors
attack_train_x = combine([target_train_posteriors, target_test_posteriors, 
                         shadow_train_posteriors, shadow_test_posteriors])
attack_train_y = [1, 0, 1, 0]  # 1=training member, 0=non-member

# Train binary classifier to detect membership
attack_model = AttackModel()
train_attack_model(attack_model, attack_train_x, attack_train_y)
```

#### Phase 5: Attack Evaluation
```python
# Test attack performance
attack_predictions = attack_model.predict(test_posteriors)
auroc = roc_auc_score(true_membership, attack_predictions)
```

## Key Classes and Functions

### Core Model Architecture (`Net` class)
- Implements multiple GNN architectures (GCN, GAT, SAGE, SGC)
- Adapts architecture based on dataset type:
  - Standard datasets: Smaller models with dropout
  - Custom datasets: Larger models (128→64→classes) without dropout for overfitting

### Attack Model (`AttackModel` class)
- Simple 3-layer MLP that takes model posteriors as input
- Binary classification: member vs non-member
- Trained on posteriors from both target and shadow models

### Data Management (`RebMIData` class)
- Container for all data splits required for inductive MIA
- Manages target/shadow train/test data separately
- Includes device management for GPU acceleration

## Critical Implementation Details

### Overfitting Strategy
The attack relies on target models overfitting to training data, creating distinguishable confidence patterns:
- **No dropout** on custom datasets to encourage overfitting
- **Optimal epoch counts** (31 epochs found optimal for current setup)
- **Larger model capacity** for custom datasets

### Feature Engineering Pipeline
1. **Original features** from dataset
2. **Node degrees** computed from graph structure  
3. **Polynomial features** (degree²) for non-linear patterns
4. **Normalization** applied consistently across all splits

### Data Separation Strategy
- **Target model**: Sees real training data vs real non-training data
- **Shadow model**: Sees synthetic training data vs synthetic test data  
- **Attack model**: Learns from posteriors of both models to detect membership patterns

## Output and Evaluation

### Generated Files
- `posteriors*_TSTS_*.txt`: Model confidence outputs for each data split
- `resultfile_TSTS_*.txt`: Attack performance metrics and statistics
- `nodesNeigbors*_TSTS_*.npy`: Structural neighbor information

### Success Metrics
- **AUROC**: Primary evaluation metric (0.5 = random, 1.0 = perfect)
- **Accuracy/Precision/Recall**: Standard classification metrics
- **Attack success**: AUROC > 0.6 indicates reasonable attack performance

## Current Performance
- **Twitch Dataset**: AUROC ~0.487-0.499 (detectable but moderate attack signal)
- **Target Class**: 'affiliate' status works better than 'mature' content classification
- **Model Configuration**: GCN with 128→64→2 architecture, no dropout, 31 epochs

This architecture successfully demonstrates membership inference vulnerabilities in graph neural networks while providing a flexible framework for testing different attack scenarios and defenses.