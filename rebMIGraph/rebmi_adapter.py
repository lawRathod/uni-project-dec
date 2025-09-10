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

def create_inductive_split_custom(dataset_name, normalize=True, subset_ratio=1.0, max_subgraphs=None):
    """Create inductive split for custom datasets (Twitch/Event)
    
    Args:
        dataset_name: Name of the dataset ('twitch' or 'event')
        normalize: Whether to normalize features (default: True)
        subset_ratio: Fraction of data to use (0.0 to 1.0, default: 1.0 for all data)
        max_subgraphs: Maximum number of subgraphs per split (overrides subset_ratio if specified)
    """
    
    # Load data using bridge
    data = load_custom_dataset(dataset_name)
    if not data:
        raise ValueError(f"Failed to load {dataset_name} dataset")
    
    train_graphs = data['train']  # Real training subgraphs
    nontrain_graphs = data['nontrain']  # Use train data for all splits  
    synth_graphs = data['synth']  # Use train data for all splits
    
    # Apply subset selection for performance improvement
    def apply_subset(graph_list, name):
        if not graph_list:
            return graph_list
            
        original_count = len(graph_list)
        
        if max_subgraphs is not None:
            # Use explicit maximum number of subgraphs
            subset_count = min(max_subgraphs, original_count)
        else:
            # Use ratio-based subset
            subset_count = max(1, int(original_count * subset_ratio))
        
        if subset_count < original_count:
            # Randomly sample subgraphs for better generalization
            import random
            random.seed(42)  # For reproducible results
            selected_graphs = random.sample(graph_list, subset_count)
            print(f"Using subset for {name}: {subset_count}/{original_count} subgraphs ({subset_count/original_count:.1%})")
            return selected_graphs
        else:
            print(f"Using full {name} dataset: {original_count} subgraphs")
            return graph_list
    
    # Apply subset selection to improve performance
    train_graphs = apply_subset(train_graphs, "train")
    nontrain_graphs = apply_subset(nontrain_graphs, "nontrain") 
    synth_graphs = apply_subset(synth_graphs, "synth")
    
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
    
    # Feature engineering and normalization
    if rebmi_data.target_x is not None:
        # Fast lightweight feature engineering
        def enhance_features_fast(x, edge_index, y=None):
            """Add only node degree (fast) for better discriminative power"""
            from torch_geometric.utils import degree
            
            # Only add node degrees - very fast O(edges)
            degrees = degree(edge_index[0], x.shape[0]).unsqueeze(1).float()
            
            # Simple feature combination: original + degree + degree^2 (polynomial features)
            deg_squared = (degrees / degrees.max()) ** 2  # Normalized squared degree
            
            enhanced_x = torch.cat([x, degrees, deg_squared], dim=1)
            
            return enhanced_x
        
        # Enhance all datasets (fast)
        rebmi_data.target_x = enhance_features_fast(rebmi_data.target_x, rebmi_data.target_edge_index, rebmi_data.target_y)
        if rebmi_data.target_test_x is not None:
            rebmi_data.target_test_x = enhance_features_fast(rebmi_data.target_test_x, rebmi_data.target_test_edge_index, rebmi_data.target_test_y)
        if rebmi_data.shadow_x is not None:
            rebmi_data.shadow_x = enhance_features_fast(rebmi_data.shadow_x, rebmi_data.shadow_edge_index, rebmi_data.shadow_y)
        if rebmi_data.shadow_test_x is not None:
            rebmi_data.shadow_test_x = enhance_features_fast(rebmi_data.shadow_test_x, rebmi_data.shadow_test_edge_index, rebmi_data.shadow_test_y)
        
        # Use original dataset labels (no synthetic manipulation)
        print("Using original dataset labels with enhanced features...")
        
        # Now normalize the enhanced features
        if normalize:
            mean = rebmi_data.target_x.mean(dim=0, keepdim=True)
            std = rebmi_data.target_x.std(dim=0, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            
            rebmi_data.target_x = (rebmi_data.target_x - mean) / std
            if rebmi_data.target_test_x is not None:
                rebmi_data.target_test_x = (rebmi_data.target_test_x - mean) / std
            if rebmi_data.shadow_x is not None:
                rebmi_data.shadow_x = (rebmi_data.shadow_x - mean) / std
            if rebmi_data.shadow_test_x is not None:
                rebmi_data.shadow_test_x = (rebmi_data.shadow_test_x - mean) / std
    
    return rebmi_data

def get_dataset_info(dataset_name, subset_ratio=1.0, max_subgraphs=None):
    """Get dataset metadata for TSTS.py"""
    data = load_custom_dataset(dataset_name)
    if not data:
        return None, None
    
    # Create a dummy enhanced feature to get correct feature count
    dummy_data = create_inductive_split_custom(dataset_name, subset_ratio=subset_ratio, max_subgraphs=max_subgraphs)
    if dummy_data and dummy_data.target_x is not None:
        enhanced_features = dummy_data.target_x.shape[1]
    else:
        enhanced_features = data['num_features']
        
    return enhanced_features, data['num_classes']

if __name__ == "__main__":
    # Test adapter with different subset configurations
    for dataset in ['twitch', 'event']:
        print(f"\n=== Testing {dataset} ===")
        
        # Test with full dataset
        print("\n1. Full dataset:")
        rebmi_data_full = create_inductive_split_custom(dataset, subset_ratio=1.0)
        print(f"Target train: {rebmi_data_full.target_x.shape if rebmi_data_full.target_x is not None else 'None'}")
        print(f"Target test: {rebmi_data_full.target_test_x.shape if rebmi_data_full.target_test_x is not None else 'None'}")
        print(f"Shadow train: {rebmi_data_full.shadow_x.shape if rebmi_data_full.shadow_x is not None else 'None'}")
        print(f"Shadow test: {rebmi_data_full.shadow_test_x.shape if rebmi_data_full.shadow_test_x is not None else 'None'}")
        
        # Test with 30% subset
        print("\n2. 30% subset:")
        rebmi_data_subset = create_inductive_split_custom(dataset, subset_ratio=0.3)
        print(f"Target train: {rebmi_data_subset.target_x.shape if rebmi_data_subset.target_x is not None else 'None'}")
        print(f"Target test: {rebmi_data_subset.target_test_x.shape if rebmi_data_subset.target_test_x is not None else 'None'}")
        print(f"Shadow train: {rebmi_data_subset.shadow_x.shape if rebmi_data_subset.shadow_x is not None else 'None'}")
        print(f"Shadow test: {rebmi_data_subset.shadow_test_x.shape if rebmi_data_subset.shadow_test_x is not None else 'None'}")
        
        # Test with max 5 subgraphs
        print("\n3. Max 5 subgraphs:")
        rebmi_data_max = create_inductive_split_custom(dataset, max_subgraphs=5)
        print(f"Target train: {rebmi_data_max.target_x.shape if rebmi_data_max.target_x is not None else 'None'}")
        print(f"Target test: {rebmi_data_max.target_test_x.shape if rebmi_data_max.target_test_x is not None else 'None'}")
        print(f"Shadow train: {rebmi_data_max.shadow_x.shape if rebmi_data_max.shadow_x is not None else 'None'}")
        print(f"Shadow test: {rebmi_data_max.shadow_test_x.shape if rebmi_data_max.shadow_test_x is not None else 'None'}")