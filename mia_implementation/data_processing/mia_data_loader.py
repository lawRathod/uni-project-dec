"""
Data loader utilities for integrating our datasets with rebMIGraph/TSTS.py
This module provides the bridge between our data format and rebMIGraph requirements.
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import bridge
sys.path.append(str(Path(__file__).parent))

from bridge import load_dataset_for_attack, create_mock_dataset_for_rebmi
import torch
from torch_geometric.data import Data, Batch

class MIADataLoader:
    """
    Data loader for realistic MIA attack with separate data sources:
    - Target model: trains on REAL training data 
    - Shadow model: trains on SYNTHETIC data
    - Attack test: evaluates on REAL non-training data
    """
    
    def __init__(self, dataset_name):
        """
        Initialize with dataset name ('twitch' or 'event')
        """
        self.dataset_name = dataset_name
        self.target_data = None      # Real training subgraphs
        self.shadow_data = None      # Synthetic subgraphs  
        self.test_data = None        # Real non-training subgraphs
        
        # Load all data types
        self._load_all_data()
    
    def _load_all_data(self):
        """Load target (train), shadow (synth), and test (nontrain) data"""
        print(f"Loading all data for {self.dataset_name}...")
        
        # Load target training data (real data)
        try:
            self.target_data = load_dataset_for_attack(self.dataset_name, "train")
            print(f"Loaded {len(self.target_data)} target training subgraphs")
        except Exception as e:
            print(f"Warning: Could not load target training data: {e}")
            self.target_data = []
        
        # Load shadow training data (synthetic data)
        self.shadow_data = load_dataset_for_attack(self.dataset_name, "synth")
        print(f"Loaded {len(self.shadow_data)} shadow training subgraphs")
        
        # Load test data (real non-training data)
        try:
            self.test_data = load_dataset_for_attack(self.dataset_name, "nontrain")
            print(f"Loaded {len(self.test_data)} test subgraphs")
        except Exception as e:
            print(f"Warning: Could not load test data: {e}")
            self.test_data = []
    
    def get_combined_dataset(self, data_type="shadow"):
        """
        Create a combined dataset from multiple subgraphs for rebMIGraph compatibility
        
        Args:
            data_type: 'target', 'shadow', or 'test'
        """
        if data_type == "target":
            data_list = self.target_data
        elif data_type == "shadow":
            data_list = self.shadow_data
        elif data_type == "test":
            data_list = self.test_data
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        if not data_list:
            raise ValueError(f"No data available for {data_type}")
        
        # For now, combine all subgraphs into a single large graph
        # This is a simplification - ideally we'd modify rebMIGraph to handle multiple subgraphs
        combined_x = []
        combined_y = []
        combined_edges = []
        node_offset = 0
        
        for data in data_list:
            combined_x.append(data.x)
            combined_y.append(data.y)
            
            # Adjust edge indices by node offset
            if data.edge_index.shape[1] > 0:
                adjusted_edges = data.edge_index + node_offset
                combined_edges.append(adjusted_edges)
            
            node_offset += data.x.shape[0]
        
        # Combine everything
        combined_x = torch.cat(combined_x, dim=0)
        combined_y = torch.cat(combined_y, dim=0)
        
        if combined_edges:
            combined_edge_index = torch.cat(combined_edges, dim=1)
        else:
            combined_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create combined Data object
        combined_data = Data(x=combined_x, edge_index=combined_edge_index, y=combined_y)
        
        return combined_data
    
    def create_dataset_for_rebmi(self):
        """
        Create dataset objects compatible with rebMIGraph
        Returns mock dataset object and the raw data
        """
        # Use shadow data as the base (this is what rebMIGraph will primarily use)
        if not self.shadow_data:
            raise ValueError("No shadow data available")
        
        sample = self.shadow_data[0]
        num_classes = len(torch.unique(sample.y))
        num_node_features = sample.x.shape[1]
        
        # Create combined dataset
        combined_data = self.get_combined_dataset("shadow")
        
        # Create mock dataset class for rebMIGraph compatibility
        class MockDataset:
            def __init__(self, data, num_classes, num_node_features):
                self.data = data
                self.num_classes = num_classes
                self.num_node_features = num_node_features
                self.num_features = num_node_features  # Alias
            
            def __getitem__(self, idx):
                return self.data
            
            def __len__(self):
                return 1
        
        mock_dataset = MockDataset(combined_data, num_classes, num_node_features)
        
        return mock_dataset, {
            'target_data': self.target_data,
            'shadow_data': self.shadow_data,
            'test_data': self.test_data
        }

def get_realistic_attack_data(dataset_name):
    """
    Get realistic MIA attack data with separate sources for target/shadow/test
    
    Returns:
        - attack_data: Dictionary with separate data for realistic attack
    """
    loader = MIADataLoader(dataset_name)
    
    return {
        'target_train_subgraphs': loader.target_data,    # Real training data for target model
        'shadow_train_subgraphs': loader.shadow_data,    # Synthetic data for shadow model  
        'attack_test_subgraphs': loader.test_data,       # Real non-training data for testing attack
        'dataset_info': {
            'name': dataset_name,
            'num_classes': 2,  # Both datasets have binary classification
            'num_features': 7 if dataset_name == 'twitch' else 4,
            'classification_target': 'mature' if dataset_name == 'twitch' else 'gender'
        }
    }

def create_subgraph_iterator(subgraph_list, batch_size=1):
    """
    Create iterator for training on multiple subgraphs
    
    Args:
        subgraph_list: List of torch_geometric.Data objects
        batch_size: Number of subgraphs to process at once
    
    Yields:
        Batches of subgraphs for training
    """
    if not subgraph_list:
        return
        
    for i in range(0, len(subgraph_list), batch_size):
        batch = subgraph_list[i:i + batch_size]
        yield batch

def combine_subgraph_batch(subgraph_batch):
    """
    Combine a batch of subgraphs into a single graph for training using PyG Batch
    
    Args:
        subgraph_batch: List of torch_geometric.Data objects
    
    Returns:
        Combined torch_geometric.Data object
    """
    if len(subgraph_batch) == 1:
        return subgraph_batch[0]
    
    # Use PyTorch Geometric's Batch to properly combine subgraphs
    # This handles edge index adjustments automatically
    batch = Batch.from_data_list(subgraph_batch)
    
    # Convert batch back to single Data object
    # Note: This creates a disconnected graph (multiple components)
    return Data(x=batch.x, edge_index=batch.edge_index, y=batch.y, batch=batch.batch)

def setup_data_for_rebmi(dataset_name):
    """
    Setup data for rebMIGraph integration (legacy compatibility)
    
    Returns:
        - mock_dataset: Dataset object compatible with rebMIGraph
        - data_dict: Dictionary with all data types for custom training
    """
    loader = MIADataLoader(dataset_name)
    return loader.create_dataset_for_rebmi()

if __name__ == "__main__":
    # Test the data loader
    for dataset_name in ["twitch", "event"]:
        print(f"\n=== Testing MIA Data Loader for {dataset_name} ===")
        
        try:
            loader = MIADataLoader(dataset_name)
            mock_dataset, data_dict = loader.create_dataset_for_rebmi()
            
            print(f"Mock dataset created:")
            print(f"  - Num classes: {mock_dataset.num_classes}")
            print(f"  - Num features: {mock_dataset.num_node_features}")
            print(f"  - Combined data shape: {mock_dataset.data.x.shape}")
            
            print(f"Raw data available:")
            print(f"  - Target subgraphs: {len(data_dict['target_data'])}")
            print(f"  - Shadow subgraphs: {len(data_dict['shadow_data'])}")
            print(f"  - Test subgraphs: {len(data_dict['test_data'])}")
            
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")