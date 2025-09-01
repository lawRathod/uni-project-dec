"""
Adapter module for rebMIGraph compatibility.
Creates data structures and functions needed by rebMIGraph TSTS.py
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .dataset_loader import DatasetLoader
from .data_converter import align_feature_dimensions


class RebMIAdapter:
    """Adapter for rebMIGraph compatibility."""
    
    def __init__(self, base_path: str = "./datasets"):
        """Initialize adapter with dataset base path."""
        self.loader = DatasetLoader(base_path)
    
    def create_rebmi_dataset(self, dataset_name: str) -> Tuple:
        """
        Create dataset structure compatible with rebMIGraph.
        
        Args:
            dataset_name: Name of the dataset ('twitch' or 'event')
        
        Returns:
            Tuple of (mock_dataset, data_dict)
        """
        # Load all data types
        data_dict = {
            'target_train': [],
            'shadow_train': [],
            'test': []
        }
        
        # Load target training data (real)
        try:
            data_dict['target_train'] = self.loader.load_dataset(dataset_name, 'train')
        except:
            print(f"Warning: No target training data found for {dataset_name}")
        
        # Load shadow training data (synthetic)
        try:
            data_dict['shadow_train'] = self.loader.load_dataset(dataset_name, 'synth')
        except:
            raise ValueError(f"No synthetic data found for {dataset_name}")
        
        # Load test data (real non-training)
        try:
            data_dict['test'] = self.loader.load_dataset(dataset_name, 'nontrain')
        except:
            print(f"Warning: No test data found for {dataset_name}")
        
        # Get dataset info
        info = self.loader.get_dataset_info(dataset_name)
        
        # Create mock dataset for compatibility
        mock_dataset = MockDataset(
            num_classes=info['num_classes'],
            num_features=info['num_features']
        )
        
        return mock_dataset, data_dict
    
    def get_inductive_split_from_subgraphs(
        self,
        target_subgraphs: List[Data],
        shadow_subgraphs: List[Data],
        test_subgraphs: List[Data]
    ) -> Data:
        """
        Create inductive split data structure from subgraphs.
        
        Args:
            target_subgraphs: Target training subgraphs (real)
            shadow_subgraphs: Shadow training subgraphs (synthetic)
            test_subgraphs: Test subgraphs (real non-training)
        
        Returns:
            Combined Data object with all necessary fields for TSTS.py
        """
        # Ensure all subgraphs have same feature dimensions
        all_subgraphs = target_subgraphs + shadow_subgraphs + test_subgraphs
        all_subgraphs = align_feature_dimensions(all_subgraphs)
        
        # Split back into categories
        n_target = len(target_subgraphs)
        n_shadow = len(shadow_subgraphs)
        
        target_subgraphs = all_subgraphs[:n_target]
        shadow_subgraphs = all_subgraphs[n_target:n_target+n_shadow]
        test_subgraphs = all_subgraphs[n_target+n_shadow:]
        
        # Create batched graphs for each split
        target_batch = self._create_batch(target_subgraphs) if target_subgraphs else None
        shadow_batch = self._create_batch(shadow_subgraphs) if shadow_subgraphs else None
        test_batch = self._create_batch(test_subgraphs) if test_subgraphs else None
        
        # Prepare data components
        if target_batch:
            target_x = target_batch.x
            target_y = target_batch.y
            target_edge_index = target_batch.edge_index
            target_train_mask = torch.ones(len(target_y), dtype=torch.long)
        else:
            # No target data - use shadow data dimensions
            feature_dim = shadow_batch.x.size(1) if shadow_batch else 7
            target_x = torch.empty((0, feature_dim), dtype=torch.float32)
            target_y = torch.empty((0,), dtype=torch.long)
            target_edge_index = torch.empty((2, 0), dtype=torch.long)
            target_train_mask = torch.empty((0,), dtype=torch.long)
        
        # Shadow data
        if shadow_batch:
            shadow_x = shadow_batch.x
            shadow_y = shadow_batch.y
            shadow_edge_index = shadow_batch.edge_index
            shadow_train_mask = torch.ones(len(shadow_y), dtype=torch.long)
        else:
            raise ValueError("Shadow data is required for MIA")
        
        # Test data - split into target test and shadow test
        if test_batch:
            # Split test data in half for target and shadow test sets
            n_test = test_batch.x.size(0)
            n_test_half = n_test // 2
            
            target_test_x = test_batch.x[:n_test_half]
            target_test_y = test_batch.y[:n_test_half]
            # Extract subgraph for target test
            target_test_edge_index = self._extract_subgraph_edges(
                test_batch.edge_index, n_test_half, 0
            )
            target_test_mask = torch.ones(n_test_half, dtype=torch.long)
            
            shadow_test_x = test_batch.x[n_test_half:n_test]
            shadow_test_y = test_batch.y[n_test_half:n_test]
            # Extract subgraph for shadow test
            shadow_test_edge_index = self._extract_subgraph_edges(
                test_batch.edge_index, n_test - n_test_half, n_test_half
            )
            # Adjust indices for shadow test
            shadow_test_edge_index = shadow_test_edge_index - n_test_half
            shadow_test_mask = torch.ones(n_test - n_test_half, dtype=torch.long)
        else:
            # Use portions of training data for testing if no test data
            target_test_x = target_x
            target_test_y = target_y
            target_test_edge_index = target_edge_index
            target_test_mask = target_train_mask
            
            shadow_test_x = shadow_x
            shadow_test_y = shadow_y
            shadow_test_edge_index = shadow_edge_index
            shadow_test_mask = shadow_train_mask
        
        # Create combined data for compatibility
        all_x = torch.cat([target_x, shadow_x], dim=0) if target_x.size(0) > 0 else shadow_x
        all_y = torch.cat([target_y, shadow_y], dim=0) if target_y.size(0) > 0 else shadow_y
        
        # Combine edge indices with offset for shadow
        if target_edge_index.size(1) > 0:
            shadow_offset = target_x.size(0)
            shadow_edges_offset = shadow_edge_index + shadow_offset
            all_edge_index = torch.cat([target_edge_index, shadow_edges_offset], dim=1)
        else:
            all_edge_index = shadow_edge_index
        
        # Create the combined Data object
        data = Data(
            # Target training data
            target_x=target_x,
            target_edge_index=target_edge_index,
            target_y=target_y,
            # Target test data
            target_test_x=target_test_x,
            target_test_edge_index=target_test_edge_index,
            target_test_y=target_test_y,
            # Shadow training data
            shadow_x=shadow_x,
            shadow_edge_index=shadow_edge_index,
            shadow_y=shadow_y,
            # Shadow test data
            shadow_test_x=shadow_test_x,
            shadow_test_edge_index=shadow_test_edge_index,
            shadow_test_y=shadow_test_y,
            # Masks
            target_train_mask=target_train_mask,
            shadow_train_mask=shadow_train_mask,
            target_test_mask=target_test_mask,
            shadow_test_mask=shadow_test_mask,
            # Combined data
            all_x=all_x,
            all_edge_index=all_edge_index,
            all_y=all_y
        )
        
        return data
    
    def _create_batch(self, data_list: List[Data]) -> Optional[Batch]:
        """Create a batch from list of Data objects."""
        if not data_list:
            return None
        return Batch.from_data_list(data_list)
    
    def _extract_subgraph_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        offset: int
    ) -> torch.Tensor:
        """Extract edges for a subgraph given node range."""
        mask = (edge_index[0] >= offset) & (edge_index[0] < offset + num_nodes)
        mask &= (edge_index[1] >= offset) & (edge_index[1] < offset + num_nodes)
        return edge_index[:, mask]


class MockDataset:
    """Mock dataset class for rebMIGraph compatibility."""
    
    def __init__(self, num_classes: int, num_features: int):
        """
        Initialize mock dataset.
        
        Args:
            num_classes: Number of classes
            num_features: Number of node features
        """
        self.num_classes = num_classes
        self.num_node_features = num_features
        self.num_features = num_features  # Alias for compatibility
    
    def __repr__(self):
        return f"MockDataset(classes={self.num_classes}, features={self.num_features})"


def create_rebmi_dataset(
    dataset_name: str,
    base_path: str = "./datasets"
) -> Tuple:
    """
    Create rebMIGraph-compatible dataset.
    
    Args:
        dataset_name: Name of the dataset
        base_path: Base directory for datasets
    
    Returns:
        Tuple of (mock_dataset, data_dict)
    """
    adapter = RebMIAdapter(base_path)
    return adapter.create_rebmi_dataset(dataset_name)


def get_inductive_split_from_subgraphs(
    target_subgraphs: List[Data],
    shadow_subgraphs: List[Data],
    test_subgraphs: List[Data]
) -> Data:
    """
    Create inductive split from subgraphs.
    
    Args:
        target_subgraphs: Target training subgraphs
        shadow_subgraphs: Shadow training subgraphs
        test_subgraphs: Test subgraphs
    
    Returns:
        Combined Data object for TSTS.py
    """
    adapter = RebMIAdapter()
    return adapter.get_inductive_split_from_subgraphs(
        target_subgraphs, shadow_subgraphs, test_subgraphs
    )


def create_data_masks(
    num_target_train: int,
    num_shadow_train: int,
    num_target_test: int,
    num_shadow_test: int
) -> Dict[str, torch.Tensor]:
    """
    Create data masks for train/test splits.
    
    Args:
        num_target_train: Number of target training nodes
        num_shadow_train: Number of shadow training nodes
        num_target_test: Number of target test nodes
        num_shadow_test: Number of shadow test nodes
    
    Returns:
        Dictionary of masks
    """
    return {
        'target_train_mask': torch.ones(num_target_train, dtype=torch.long),
        'shadow_train_mask': torch.ones(num_shadow_train, dtype=torch.long),
        'target_test_mask': torch.ones(num_target_test, dtype=torch.long),
        'shadow_test_mask': torch.ones(num_shadow_test, dtype=torch.long)
    }