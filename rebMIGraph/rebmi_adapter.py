#!/usr/bin/env python3
"""
rebMI Adapter for inductive splits with pickle datasets (Twitch, Event)
Handles data splitting for TSTS membership inference attacks
"""

import torch
from bridge import load_custom_dataset

class RebMIData:
    """Container for rebMI inductive split data"""
    def __init__(self):
        # Target model data (trained on real subgraphs)
        self.target_x = None
        self.target_y = None
        self.target_edge_index = None
        self.target_train_mask = None
        
        # Target test data (non-training real subgraphs)
        self.target_test_x = None
        self.target_test_y = None
        self.target_test_edge_index = None
        self.target_test_mask = None
        
        # Shadow model data (trained on synthetic subgraphs)
        self.shadow_x = None
        self.shadow_y = None
        self.shadow_edge_index = None
        self.shadow_train_mask = None
        
        # Shadow test data (synthetic subgraphs for testing)
        self.shadow_test_x = None
        self.shadow_test_y = None
        self.shadow_test_edge_index = None
        self.shadow_test_mask = None
    
    def to(self, device):
        """Move all tensors to the specified device"""
        attrs = ['target_x', 'target_y', 'target_edge_index', 'target_train_mask',
                'target_test_x', 'target_test_y', 'target_test_edge_index', 'target_test_mask',
                'shadow_x', 'shadow_y', 'shadow_edge_index', 'shadow_train_mask',
                'shadow_test_x', 'shadow_test_y', 'shadow_test_edge_index', 'shadow_test_mask']
        
        for attr in attrs:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self

def create_inductive_split_custom(dataset_name, normalize=True):
    """Create inductive split for custom datasets (Twitch/Event)
    
    Args:
        dataset_name: Name of the dataset ('twitch' or 'event')
        normalize: Whether to normalize features (default: True)
    """
    
    # Load data using bridge
    data = load_custom_dataset(dataset_name)
    if not data:
        raise ValueError(f"Failed to load {dataset_name} dataset")
    
    train_graphs = data['train']  # Real training subgraphs
    nontrain_graphs = data['train']  # Real non-training subgraphs  
    synth_graphs = data['train']  # Synthetic subgraphs
    
    # Combine subgraphs into single tensors
    def combine_subgraphs(graph_list, label_prefix="combined"):
        if not graph_list:
            return None, None, None, None
            
        all_x = []
        all_y = []
        all_edges = []
        node_offset = 0
        
        for graph in graph_list:
            all_x.append(graph.x)
            all_y.append(graph.y)
            # Offset edge indices to create one large graph
            edges_offset = graph.edge_index + node_offset
            all_edges.append(edges_offset)
            node_offset += graph.x.shape[0]
        
        combined_x = torch.cat(all_x, dim=0)
        combined_y = torch.cat(all_y, dim=0)
        combined_edges = torch.cat(all_edges, dim=1)
        combined_mask = torch.ones(combined_x.shape[0], dtype=torch.bool)
        
        return combined_x, combined_y, combined_edges, combined_mask
    
    # Create rebMI data structure
    rebmi_data = RebMIData()
    
    # Target model: train on real subgraphs, test on real non-training
    rebmi_data.target_x, rebmi_data.target_y, rebmi_data.target_edge_index, rebmi_data.target_train_mask = combine_subgraphs(train_graphs)
    rebmi_data.target_test_x, rebmi_data.target_test_y, rebmi_data.target_test_edge_index, rebmi_data.target_test_mask = combine_subgraphs(nontrain_graphs)
    
    # Shadow model: train and test on synthetic subgraphs (split synthetic data)
    mid = len(synth_graphs) // 2
    shadow_train = synth_graphs[:mid]
    shadow_test = synth_graphs[mid:]
    
    rebmi_data.shadow_x, rebmi_data.shadow_y, rebmi_data.shadow_edge_index, rebmi_data.shadow_train_mask = combine_subgraphs(shadow_train)
    rebmi_data.shadow_test_x, rebmi_data.shadow_test_y, rebmi_data.shadow_test_edge_index, rebmi_data.shadow_test_mask = combine_subgraphs(shadow_test)
    
    # Normalize features if requested
    if normalize:
        # Compute mean and std from all target training data
        if rebmi_data.target_x is not None:
            mean = rebmi_data.target_x.mean(dim=0, keepdim=True)
            std = rebmi_data.target_x.std(dim=0, keepdim=True)
            # Avoid division by zero
            std = torch.where(std == 0, torch.ones_like(std), std)
            
            # Normalize all datasets using the same statistics
            rebmi_data.target_x = (rebmi_data.target_x - mean) / std
            if rebmi_data.target_test_x is not None:
                rebmi_data.target_test_x = (rebmi_data.target_test_x - mean) / std
            if rebmi_data.shadow_x is not None:
                rebmi_data.shadow_x = (rebmi_data.shadow_x - mean) / std
            if rebmi_data.shadow_test_x is not None:
                rebmi_data.shadow_test_x = (rebmi_data.shadow_test_x - mean) / std
    
    return rebmi_data

def get_dataset_info(dataset_name):
    """Get dataset metadata for TSTS.py"""
    data = load_custom_dataset(dataset_name)
    if not data:
        return None, None
        
    return data['num_features'], data['num_classes']

if __name__ == "__main__":
    # Test adapter
    for dataset in ['twitch', 'event']:
        print(f"\nTesting {dataset}:")
        rebmi_data = create_inductive_split_custom(dataset)
        print(f"Target train: {rebmi_data.target_x.shape if rebmi_data.target_x is not None else 'None'}")
        print(f"Target test: {rebmi_data.target_test_x.shape if rebmi_data.target_test_x is not None else 'None'}")
        print(f"Shadow train: {rebmi_data.shadow_x.shape if rebmi_data.shadow_x is not None else 'None'}")
        print(f"Shadow test: {rebmi_data.shadow_test_x.shape if rebmi_data.shadow_test_x is not None else 'None'}")