"""
Test suite to validate pickle data loading and get_realistic_attack_data function.
This test ensures the data pipeline correctly loads and processes pickle files
for the membership inference attack implementation.
"""

import sys
import os
from pathlib import Path
import pytest
import torch
import pandas as pd
import networkx as nx

# Add data processing module to path
sys.path.append(str(Path(__file__).parent.parent / "data_processing"))

from mia_data_loader import get_realistic_attack_data, MIADataLoader
from bridge import load_dataset_for_attack, convert_to_torch_geometric

class TestDataLoading:
    """Test suite for data loading functionality"""
    
    @pytest.fixture(scope="class")
    def datasets_dir(self):
        """Path to datasets directory"""
        return Path("/Users/prateek/notes/uni-project-dec/datasets")
    
    @pytest.fixture(scope="class", params=["twitch", "event"])
    def dataset_name(self, request):
        """Test both datasets"""
        return request.param
    
    def test_pickle_file_exists(self, datasets_dir, dataset_name):
        """Test that required pickle files exist"""
        synth_file = datasets_dir / dataset_name / f"{dataset_name}_synth.pickle"
        train_file = datasets_dir / dataset_name / f"{dataset_name}_train.pt"
        nontrain_file = datasets_dir / dataset_name / f"{dataset_name}_nontrain.pt"
        
        assert synth_file.exists(), f"Synthetic data file missing: {synth_file}"
        # Train and nontrain may not exist, but we should handle that gracefully
    
    def test_pickle_data_format(self, dataset_name):
        """Test that pickle files contain expected data format"""
        synth_data = load_dataset_for_attack(dataset_name, "synth")
        
        assert isinstance(synth_data, list), "Synthetic data should be a list"
        assert len(synth_data) > 0, "Synthetic data should not be empty"
        
        # Test first subgraph
        first_subgraph = synth_data[0]
        assert hasattr(first_subgraph, 'x'), "Data should have node features (x)"
        assert hasattr(first_subgraph, 'y'), "Data should have labels (y)"
        assert hasattr(first_subgraph, 'edge_index'), "Data should have edge indices"
        
        # Validate tensor shapes
        assert first_subgraph.x.dim() == 2, "Node features should be 2D tensor"
        assert first_subgraph.y.dim() == 1, "Labels should be 1D tensor"
        assert first_subgraph.edge_index.dim() == 2, "Edge indices should be 2D tensor"
        assert first_subgraph.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    
    def test_dataset_properties(self, dataset_name):
        """Test dataset-specific properties match expectations"""
        synth_data = load_dataset_for_attack(dataset_name, "synth")
        first_subgraph = synth_data[0]
        
        if dataset_name == "twitch":
            # Twitch should have 7 features (excluding 'mature' target)
            expected_features = 7
            expected_classes = 2  # binary classification on 'mature'
        elif dataset_name == "event":
            # Event should have 4 features (excluding 'gender' target)
            expected_features = 4
            expected_classes = 2  # binary classification on 'gender'
        
        assert first_subgraph.x.shape[1] == expected_features, \
            f"Expected {expected_features} features for {dataset_name}, got {first_subgraph.x.shape[1]}"
        
        unique_labels = torch.unique(first_subgraph.y)
        assert len(unique_labels) <= expected_classes, \
            f"Expected max {expected_classes} classes for {dataset_name}, got {len(unique_labels)}"
        
        # Test that labels are in expected range [0, num_classes-1]
        assert unique_labels.min() >= 0, "Labels should be non-negative"
        assert unique_labels.max() < expected_classes, f"Labels should be < {expected_classes}"
    
    def test_subgraph_consistency(self, dataset_name):
        """Test that all subgraphs in synthetic data have consistent properties"""
        synth_data = load_dataset_for_attack(dataset_name, "synth")
        
        if len(synth_data) < 2:
            pytest.skip("Need at least 2 subgraphs to test consistency")
        
        first_subgraph = synth_data[0]
        num_features = first_subgraph.x.shape[1]
        
        for i, subgraph in enumerate(synth_data[:10]):  # Test first 10 for efficiency
            assert subgraph.x.shape[1] == num_features, \
                f"Subgraph {i} has {subgraph.x.shape[1]} features, expected {num_features}"
            assert subgraph.x.dtype == torch.float, f"Features should be float, got {subgraph.x.dtype}"
            assert subgraph.y.dtype == torch.long, f"Labels should be long, got {subgraph.y.dtype}"
            assert subgraph.edge_index.dtype == torch.long, f"Edge indices should be long"
    
    def test_node_edge_consistency(self, dataset_name):
        """Test that edge indices are valid for the number of nodes"""
        synth_data = load_dataset_for_attack(dataset_name, "synth")
        
        for i, subgraph in enumerate(synth_data[:5]):  # Test first 5 for efficiency
            num_nodes = subgraph.x.shape[0]
            
            if subgraph.edge_index.shape[1] > 0:  # Only test if edges exist
                max_node_idx = subgraph.edge_index.max().item()
                assert max_node_idx < num_nodes, \
                    f"Subgraph {i}: max edge index {max_node_idx} >= num_nodes {num_nodes}"
                
                min_node_idx = subgraph.edge_index.min().item()
                assert min_node_idx >= 0, \
                    f"Subgraph {i}: negative node index {min_node_idx}"
    
    def test_get_realistic_attack_data_structure(self, dataset_name):
        """Test that get_realistic_attack_data returns expected structure"""
        attack_data = get_realistic_attack_data(dataset_name)
        
        # Check top-level structure
        assert isinstance(attack_data, dict), "Should return dictionary"
        required_keys = ['target_train_subgraphs', 'shadow_train_subgraphs', 
                        'attack_test_subgraphs', 'dataset_info']
        
        for key in required_keys:
            assert key in attack_data, f"Missing required key: {key}"
        
        # Check dataset_info structure
        dataset_info = attack_data['dataset_info']
        info_keys = ['name', 'num_classes', 'num_features', 'classification_target']
        
        for key in info_keys:
            assert key in dataset_info, f"Missing dataset_info key: {key}"
        
        assert dataset_info['name'] == dataset_name
        assert dataset_info['num_classes'] == 2  # Both datasets are binary
        
        if dataset_name == "twitch":
            assert dataset_info['num_features'] == 7
            assert dataset_info['classification_target'] == 'mature'
        elif dataset_name == "event":
            assert dataset_info['num_features'] == 4
            assert dataset_info['classification_target'] == 'gender'
    
    def test_get_realistic_attack_data_contents(self, dataset_name):
        """Test that get_realistic_attack_data contains valid data"""
        attack_data = get_realistic_attack_data(dataset_name)
        
        # Shadow data should always be available (synthetic)
        shadow_data = attack_data['shadow_train_subgraphs']
        assert isinstance(shadow_data, list), "Shadow data should be list"
        assert len(shadow_data) > 0, "Shadow data should not be empty"
        
        # Test that shadow data contains valid torch_geometric Data objects
        first_shadow = shadow_data[0]
        assert hasattr(first_shadow, 'x'), "Shadow data should have node features"
        assert hasattr(first_shadow, 'y'), "Shadow data should have labels"
        assert hasattr(first_shadow, 'edge_index'), "Shadow data should have edges"
    
    def test_mia_data_loader_initialization(self, dataset_name):
        """Test MIADataLoader class initialization"""
        loader = MIADataLoader(dataset_name)
        
        assert loader.dataset_name == dataset_name
        assert loader.shadow_data is not None, "Shadow data should be loaded"
        assert isinstance(loader.shadow_data, list), "Shadow data should be list"
        assert len(loader.shadow_data) > 0, "Shadow data should not be empty"
        
        # Target and test data may be None if files don't exist, which is okay
        if loader.target_data is not None:
            assert isinstance(loader.target_data, list), "Target data should be list"
        if loader.test_data is not None:
            assert isinstance(loader.test_data, list), "Test data should be list"
    
    def test_combined_dataset_creation(self, dataset_name):
        """Test combining multiple subgraphs into single dataset"""
        loader = MIADataLoader(dataset_name)
        
        # Test shadow data combination (should always work)
        combined_data = loader.get_combined_dataset("shadow")
        
        assert hasattr(combined_data, 'x'), "Combined data should have features"
        assert hasattr(combined_data, 'y'), "Combined data should have labels"
        assert hasattr(combined_data, 'edge_index'), "Combined data should have edges"
        
        # Combined data should have more nodes than individual subgraphs
        individual_nodes = loader.shadow_data[0].x.shape[0]
        combined_nodes = combined_data.x.shape[0]
        
        if len(loader.shadow_data) > 1:
            assert combined_nodes > individual_nodes, \
                "Combined data should have more nodes than individual subgraphs"
        
        # Test that node and label counts match
        assert combined_data.x.shape[0] == combined_data.y.shape[0], \
            "Number of nodes should match number of labels"
    
    def test_data_type_consistency(self, dataset_name):
        """Test that data types are consistent across the pipeline"""
        attack_data = get_realistic_attack_data(dataset_name)
        shadow_data = attack_data['shadow_train_subgraphs']
        
        for subgraph in shadow_data[:3]:  # Test first 3
            # Node features should be float
            assert subgraph.x.dtype == torch.float, \
                f"Node features should be float, got {subgraph.x.dtype}"
            
            # Labels should be long (for classification)
            assert subgraph.y.dtype == torch.long, \
                f"Labels should be long, got {subgraph.y.dtype}"
            
            # Edge indices should be long
            assert subgraph.edge_index.dtype == torch.long, \
                f"Edge indices should be long, got {subgraph.edge_index.dtype}"
    
    def test_feature_range_validity(self, dataset_name):
        """Test that features are in reasonable ranges (not NaN/inf)"""
        attack_data = get_realistic_attack_data(dataset_name)
        shadow_data = attack_data['shadow_train_subgraphs']
        
        for i, subgraph in enumerate(shadow_data[:3]):  # Test first 3
            # Check for NaN values
            assert not torch.isnan(subgraph.x).any(), \
                f"Subgraph {i} contains NaN features"
            
            # Check for infinite values
            assert not torch.isinf(subgraph.x).any(), \
                f"Subgraph {i} contains infinite features"
            
            # Labels should be valid integers
            unique_labels = torch.unique(subgraph.y)
            assert all(label.item() == int(label.item()) for label in unique_labels), \
                f"Subgraph {i} contains non-integer labels"

if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])