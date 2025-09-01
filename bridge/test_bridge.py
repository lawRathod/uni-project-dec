"""
Test script for the bridge module.
Tests data conversion and loading functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bridge import (
    convert_to_pyg_data,
    load_dataset,
    load_subgraphs,
    get_dataset_info,
    create_rebmi_dataset,
    get_inductive_split_from_subgraphs
)
from bridge.dataset_loader import validate_dataset_compatibility


def test_dataset_loading():
    """Test loading datasets."""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    for dataset_name in ['twitch', 'event']:
        print(f"\nTesting {dataset_name} dataset:")
        
        # Get dataset info
        try:
            info = get_dataset_info(dataset_name)
            print(f"  Dataset info: {info}")
        except Exception as e:
            print(f"  Warning: Could not get info: {e}")
        
        # Try loading different data types
        for data_type in ['train', 'nontrain', 'synth']:
            try:
                data_list = load_dataset(dataset_name, data_type)
                if data_list:
                    sample = data_list[0]
                    print(f"  {data_type}: {len(data_list)} graphs")
                    print(f"    Sample shape: x={sample.x.shape}, edges={sample.edge_index.shape}")
                    print(f"    Labels: {sample.y.unique().tolist()}")
                else:
                    print(f"  {data_type}: No data")
            except Exception as e:
                print(f"  {data_type}: Failed - {e}")


def test_rebmi_compatibility():
    """Test rebMIGraph compatibility functions."""
    print("\n" + "=" * 60)
    print("Testing rebMIGraph Compatibility")
    print("=" * 60)
    
    for dataset_name in ['twitch', 'event']:
        print(f"\nTesting {dataset_name} rebMI compatibility:")
        
        try:
            # Create rebMI dataset
            mock_dataset, data_dict = create_rebmi_dataset(dataset_name)
            
            print(f"  Mock dataset: {mock_dataset}")
            print(f"  Data splits:")
            for key, value in data_dict.items():
                if value:
                    print(f"    {key}: {len(value)} graphs")
                else:
                    print(f"    {key}: No data")
            
            # Test inductive split
            if all(data_dict.values()):
                split_data = get_inductive_split_from_subgraphs(
                    data_dict['target_train'],
                    data_dict['shadow_train'],
                    data_dict['test']
                )
                
                print(f"  Inductive split created:")
                print(f"    Target train: {split_data.target_x.shape}")
                print(f"    Shadow train: {split_data.shadow_x.shape}")
                print(f"    Target test: {split_data.target_test_x.shape}")
                print(f"    Shadow test: {split_data.shadow_test_x.shape}")
            else:
                print("  Skipping inductive split (missing data)")
                
        except Exception as e:
            print(f"  Failed: {e}")


def test_data_validation():
    """Test data validation."""
    print("\n" + "=" * 60)
    print("Testing Data Validation")
    print("=" * 60)
    
    for dataset_name in ['twitch', 'event']:
        print(f"\nValidating {dataset_name} dataset:")
        
        try:
            # Load all subgraphs
            subgraphs = load_subgraphs(dataset_name)
            
            # Validate compatibility
            results = validate_dataset_compatibility(
                subgraphs.get('train', []),
                subgraphs.get('synth', []),
                subgraphs.get('nontrain', [])
            )
            
            print(f"  Valid: {results['valid']}")
            if results.get('issues'):
                print(f"  Issues: {results['issues']}")
            if results.get('summary'):
                print(f"  Summary: {results['summary']}")
                
        except Exception as e:
            print(f"  Validation failed: {e}")


def test_single_conversion():
    """Test single data conversion."""
    print("\n" + "=" * 60)
    print("Testing Single Data Conversion")
    print("=" * 60)
    
    import pickle
    import pandas as pd
    import networkx as nx
    
    # Try to load and convert a single synthetic graph
    for dataset_name in ['twitch', 'event']:
        print(f"\nTesting {dataset_name} conversion:")
        
        try:
            file_path = Path(f"./datasets/{dataset_name}/{dataset_name}_synth.pickle")
            
            if not file_path.exists():
                print(f"  File not found: {file_path}")
                continue
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Get first subgraph
                first_item = data[0]
                
                if isinstance(first_item, tuple) and len(first_item) == 2:
                    df, graph = first_item
                    
                    print(f"  DataFrame shape: {df.shape}")
                    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                    print(f"  DataFrame columns: {df.columns.tolist()}")
                    
                    # Try conversion
                    target_col = 'mature' if dataset_name == 'twitch' else 'gender'
                    pyg_data = convert_to_pyg_data(df, graph, target_col)
                    
                    print(f"  Converted to PyG:")
                    print(f"    x shape: {pyg_data.x.shape}")
                    print(f"    edge_index shape: {pyg_data.edge_index.shape}")
                    print(f"    y shape: {pyg_data.y.shape}")
                    print(f"    Unique labels: {pyg_data.y.unique().tolist()}")
                else:
                    print(f"  Unexpected data format: {type(first_item)}")
            else:
                print(f"  Empty or invalid data")
                
        except Exception as e:
            print(f"  Conversion failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("Testing Bridge Module")
    print("=" * 60)
    
    # Run all tests
    test_single_conversion()
    test_dataset_loading()
    test_rebmi_compatibility()
    test_data_validation()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)