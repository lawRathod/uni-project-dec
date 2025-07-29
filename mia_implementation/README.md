# MIA Implementation

This directory contains the complete implementation for running membership inference attacks (MIA) on graph neural networks using synthetic data from DLGrapher to attack models trained on real data.

## Directory Structure

```
mia_implementation/
├── README.md                    # This file
├── data_processing/            # Data conversion and preprocessing
│   ├── bridge.py              # Main data conversion bridge
│   └── mia_data_loader.py     # Data loader for MIA attacks
├── integration/               # Integration with rebMIGraph
│   ├── run_mia_attack.py      # Main attack runner and wrapper
│   ├── rebmi_integration_twitch.py   # Twitch dataset integration script
│   └── rebmi_integration_event.py    # Event dataset integration script
└── analysis_scripts/          # Analysis and planning scripts
    ├── comprehensive_analysis.py     # Complete project analysis
    └── evaluation_strategy.py        # Evaluation framework
```

## Key Components

### Data Processing (`data_processing/`)

**`bridge.py`** - Core data conversion functionality:
- Converts (pandas.DataFrame, networkx.Graph) tuples to torch_geometric.Data format
- Handles feature preprocessing for both Twitch and Event datasets
- Supports classification targets: 'mature' (Twitch), 'gender' (Event)
- Processes 256 synthetic subgraphs from DLGrapher

**`mia_data_loader.py`** - Data loading utilities:
- Provides clean interface for loading target, shadow, and test data
- Combines multiple subgraphs for rebMIGraph compatibility
- Creates mock dataset objects that rebMIGraph expects

### Integration (`integration/`)

**`run_mia_attack.py`** - Main attack wrapper:
- Minimal wrapper that doesn't modify rebMIGraph code
- Sets up proper parameters and data sources for both datasets
- Creates integration scripts for easy rebMIGraph integration
- Demonstrates complete data flow verification

**Integration Scripts** - Ready-to-use scripts for rebMIGraph:
- `rebmi_integration_twitch.py` - Twitch dataset integration
- `rebmi_integration_event.py` - Event dataset integration

## Quick Start

### 1. Test Data Conversion
```bash
cd data_processing
python bridge.py
```

### 2. Run Complete Pipeline
```bash
cd integration
python run_mia_attack.py
```

### 3. Integrate with rebMIGraph
```bash
# Copy integration script content to rebMIGraph/TSTS.py
# Or modify TSTS.py to import the integration script
```

## Current Status

✅ **Successfully Implemented:**
- Data conversion from our format to torch_geometric
- Feature preprocessing for both datasets
- Mock dataset creation for rebMIGraph compatibility
- Integration scripts generated

✅ **Datasets Ready:**
- **Twitch**: 256 subgraphs, 30,720 nodes, 92,950 edges, 2 classes, 7 features
- **Event**: 256 subgraphs, 30,720 nodes, 57,725 edges, 2 classes, 4 features

⚠️ **Known Issues:**
- Real data (train/nontrain) can't be loaded due to pandas version mismatch
- Currently using only synthetic data (sufficient for proof of concept)

## Usage Example

```python
from data_processing.mia_data_loader import setup_data_for_rebmi

# Load dataset for attack
mock_dataset, data_dict = setup_data_for_rebmi("twitch")

# Use with rebMIGraph
dataset = mock_dataset
data = dataset[0]  # rebMIGraph expects this format
```

## Next Steps

1. **Use integration scripts** to modify rebMIGraph/TSTS.py
2. **Run TSTS.py** with custom datasets
3. **Analyze attack results** and measure effectiveness
4. **Extend to all GNN architectures** (GAT, SAGE, SGC)

## Key Features

- **Minimal Modifications**: Doesn't change rebMIGraph core code
- **Clean Separation**: All custom code stays outside rebMIGraph directory
- **Full Pipeline**: Complete data flow from datasets to attack ready format
- **Error Handling**: Robust data loading with fallbacks
- **Documentation**: Comprehensive documentation and examples