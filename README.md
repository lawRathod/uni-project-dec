# rebMIGraph - Membership Inference Attacks on GNNs

A framework for evaluating membership inference attacks on Graph Neural Networks using the TSTS (Train on Subgraph, Test on Subgraph) methodology.

## What This Does

Tests how vulnerable GNN models are to membership inference attacks - whether an attacker can determine if a specific data point was used in training. The attack uses shadow models trained on synthetic data to mimic the target model's behavior.

## Files

- **TSTS.py** - Main script. Trains target/shadow models and runs the attack. Configure dataset and model type here.
- **bridge.py** - Converts pickle files (DataFrame + NetworkX graph) to PyTorch Geometric format
- **rebmi_adapter.py** - Creates train/test splits and adds structural features (node degree) to improve attack

For detailed implementation information, see [rebMIGraph/README.md](rebMIGraph/README.md)

## Dataset Format

Expects pickle files in `../datasets/{dataset_name}/`:
- `{dataset_name}_train.pickle` - Training subgraphs (real data)
- `{dataset_name}_nontrain.pickle` - Test subgraphs (real data) 
- `{dataset_name}_synth.pickle` - Synthetic subgraphs for shadow model

Each pickle contains a list of (pandas.DataFrame, networkx.Graph) tuples.

## Usage

```bash
# Run attack (edit TSTS.py lines 72-73 to change dataset/model)
python TSTS.py
```

## Key Modifications

1. **Custom Dataset Support** - Added bridge.py to handle pickle format from synthetic graph generators
2. **Feature Engineering** - rebmi_adapter.py adds node degree features for better attack performance
3. **Modular Design** - Separated data loading, preprocessing, and attack logic for easier debugging

## Supported Models

- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)  
- SAGE (GraphSAGE)
- SGC (Simple Graph Convolution)

## Supported Datasets

- Standard (Existing): Cora, CiteSeer, PubMed, Flickr, Reddit
- Custom: Twitch, Event

## Requirements

PyTorch, PyTorch Geometric, NetworkX, Pandas, NumPy, Scikit-learn

## Reference

Based on: [Membership Inference Attack on Graph Neural Networks](https://arxiv.org/abs/2101.06570)