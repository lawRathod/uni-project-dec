# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of Membership Inference Attacks (MIA) on Graph Neural Networks, based on the paper "Membership Inference Attack on Graph Neural Network" (https://arxiv.org/abs/2101.07570). The codebase implements two attack scenarios:

- **TSTF**: Train on a subgraph and test on the full graph
- **TSTS**: Train on a subgraph, test on another subgraph (current focus)

## Core Architecture

### Main Components

1. **TSTS.py** - Primary attack implementation
   - Contains the complete MIA pipeline for graph neural networks
   - Supports multiple GNN architectures (GCN, GAT, SAGE, SGC)
   - Works with standard datasets (Cora, CiteSeer, PubMed, Reddit, Flickr) and custom datasets
   - Implements target/shadow model paradigm for membership inference

2. **bridge.py** - Custom dataset loader
   - Converts pickle datasets (Twitch, Event) to PyTorch Geometric format
   - Handles (DataFrame, NetworkX Graph) tuples from synthetic data generators
   - Maps dataset features to node attributes and target labels

3. **rebmi_adapter.py** - Inductive split adapter
   - Creates proper data splits for membership inference attacks on custom datasets
   - Combines subgraphs into single tensor representations
   - Implements feature engineering (node degrees, normalization)
   - Manages target vs shadow model data separation

### Attack Methodology

The implementation follows a standard membership inference approach:

1. **Target Model**: Trained on real training subgraphs, tested on real non-training data
2. **Shadow Model**: Trained on synthetic subgraphs to mimic target model behavior
3. **Attack Model**: Binary classifier that learns to distinguish between training vs non-training data based on model posteriors

### Key Data Flow

```
Custom Dataset (pickle) -> bridge.py -> rebmi_adapter.py -> TSTS.py
                                                              ↓
                                      Target/Shadow Models -> Attack Model -> MIA Results
```

## Common Development Tasks

### Running Membership Inference Attacks

```bash
python TSTS.py
```

The main script is configured via hyperparameters at the top of TSTS.py:
- `model_type`: GCN, GAT, SAGE, SGC  
- `data_type`: Custom_Twitch, Custom_Event, Cora, CiteSeer, etc.
- `mode`: TSTS (train/test on different subgraphs)

### Testing Dataset Loaders

```bash
# Test bridge.py dataset loading
python bridge.py

# Test rebmi_adapter.py data splitting  
python rebmi_adapter.py
```

### Key Configuration Areas

1. **Hyperparameters** (TSTS.py lines 67-133)
   - Model architectures and training parameters
   - Dataset split configurations
   - Attack model settings

2. **Custom Dataset Support** 
   - Requires datasets in `../datasets/{dataset_name}/` 
   - Files: `{dataset}_train.pickle`, `{dataset}_nontrain.pickle`, `{dataset}_synth.pickle`
   - Each pickle contains list of (DataFrame, NetworkX Graph) tuples

3. **Model Architecture Adaptations**
   - Custom datasets use larger models (128→64→classes) with no dropout for better overfitting signals
   - Standard datasets use smaller architectures with dropout

## Important Implementation Details

### Custom Dataset Integration
- The `rebmi_adapter.py` handles the complex process of combining multiple subgraphs into single tensors
- Edge indices are carefully offset to maintain graph connectivity when combining subgraphs
- Feature engineering adds node degrees and polynomial features for better discriminative power

### Attack Performance Optimization
- Target class selection is critical - 'affiliate' status performs better than 'mature' content classification for Twitch
- Model overfitting is intentionally encouraged (no dropout, appropriate epochs) to create membership signals
- Normalization and feature enhancement improve attack detectability

### Output Files
- `posteriors*_TSTS_*.txt`: Model confidence outputs used for attack training
- `resultfile_TSTS_*.txt`: Attack performance metrics (AUROC, accuracy, etc.)
- `nodesNeigbors*_TSTS_*.npy`: Structural information for advanced attacks

## Key Dependencies

- PyTorch + PyTorch Geometric for graph neural networks
- NetworkX for graph processing  
- scikit-learn for ML utilities
- pandas for data manipulation
- numpy for numerical operations

The codebase assumes CUDA availability and is configured for multi-GPU environments (CUDA_DEVICE parameter).

## Attack Evaluation

Success is measured by AUROC (Area Under ROC Curve):
- 0.5 = Random performance (no attack signal)
- >0.6 = Reasonable attack performance  
- >0.8 = Strong attack performance

Current best results achieve ~0.487-0.499 AUROC on Twitch dataset, indicating detectable but moderate membership inference capability.