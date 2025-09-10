# rebMIGraph

**Membership Inference Attacks on Graph Neural Networks** - A high-performance framework using TSTS (Train on Subgraph, Test on Subgraph) methodology.

## What It Does

Tests GNN vulnerability to membership inference attacks - determining if specific data was used in training. Uses shadow models trained on synthetic data to mimic target model behavior.

## Quick Start

```bash
# Setup environment
conda env create -f environment.yml
conda activate rebmi

# Run attack (configure in TSTS.py)
cd rebMIGraph
python TSTS.py
```

## Configuration

Edit in `TSTS.py`:

```python
# Lines 72-73: Model and Dataset
model_type = "GCN"           # Options: GCN, GAT, SAGE, SGC
data_type = "Custom_Twitch"  # Options: Custom_Twitch, Custom_Event, Cora, CiteSeer, PubMed

# Lines 79-81: Performance (custom datasets)
CUSTOM_SUBSET_RATIO = 0.3    # Use 30% of data (0.1-1.0)
CUSTOM_ADVANCED_FEATURES = True  # Enable advanced features

# Line 133: Attack Enhancement
USE_IMPROVED_ATTACK = True   # Enable improved attack model
```

## Key Features

- **Models**: GCN, GAT, SAGE, SGC
- **Datasets**: Standard (Cora, CiteSeer, PubMed, Flickr, Reddit) + Custom (Twitch, Event)
- **Performance**: Configurable data subsets, advanced graph features, optimized attack model
- **Output**: Attack accuracy, AUROC, precision, recall, F1 scores

## Dataset Format

Expects pickle files in `../datasets/{dataset_name}/`:

- `{dataset_name}_train.pickle` - Training subgraphs
- `{dataset_name}_nontrain.pickle` - Test subgraphs  
- `{dataset_name}_synth.pickle` - Synthetic subgraphs

Each pickle: list of (pandas.DataFrame, networkx.Graph) tuples

## Project Structure

```text
rebMIGraph/
├── TSTS.py           # Main attack script
├── rebmi_adapter.py  # Feature engineering & data splits
├── bridge.py         # Data format conversion
└── README.md         # Detailed implementation docs

datasets/
└── {dataset_name}/   # Dataset pickle files
```

## Performance Tips

- **Quick test**: Set `CUSTOM_SUBSET_RATIO = 0.1` (10% data)
- **Best accuracy**: Enable both `CUSTOM_ADVANCED_FEATURES` and `USE_IMPROVED_ATTACK`
- **Large graphs**: Auto-switches to memory-efficient mode (>10k nodes)

## Documentation

- [rebMIGraph/README.md](rebMIGraph/README.md) - Implementation details
- [datasets/README.md](datasets/README.md) - Dataset preparation guide

## Reference

Based on: [Membership Inference Attack on Graph Neural Networks](https://arxiv.org/abs/2101.06570)
