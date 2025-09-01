"""
Dataset loading utilities for handling various dataset formats.
"""

import pickle
import torch
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from torch_geometric.data import Data

from .data_converter import convert_to_pyg_data, convert_subgraph_list


class DatasetLoader:
    """Main dataset loader class for handling different data formats."""
    
    def __init__(self, base_path: str = "./datasets"):
        """
        Initialize dataset loader.
        
        Args:
            base_path: Base directory path for datasets
        """
        self.base_path = Path(base_path)
        self._cache = {}
    
    def load_dataset(
        self,
        dataset_name: str,
        data_type: str = "train",
        use_cache: bool = True
    ) -> List[Data]:
        """
        Load dataset and convert to PyG format.
        
        Args:
            dataset_name: Name of the dataset ('twitch' or 'event')
            data_type: Type of data ('train', 'test', 'nontrain', 'synth')
            use_cache: Whether to use cached data
        
        Returns:
            List of PyG Data objects
        """
        cache_key = f"{dataset_name}_{data_type}"
        
        if use_cache and cache_key in self._cache:
            print(f"Using cached data for {cache_key}")
            return self._cache[cache_key]
        
        # Construct file path
        file_path = self._get_file_path(dataset_name, data_type)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load the data
        raw_data = self._load_raw_data(file_path)
        
        # Convert to PyG format
        data_list = self._convert_to_pyg(raw_data, dataset_name)
        
        if use_cache:
            self._cache[cache_key] = data_list
        
        return data_list
    
    def _get_file_path(self, dataset_name: str, data_type: str) -> Path:
        """Get the file path for a specific dataset."""
        dataset_dir = self.base_path / dataset_name
        
        # Map data types to file names
        file_mapping = {
            'train': f'{dataset_name}_train.pt',
            'test': f'{dataset_name}_test.pt',
            'nontrain': f'{dataset_name}_nontrain.pt',
            'synth': f'{dataset_name}_synth.pickle',
            'original': f'{dataset_name}_original.pickle'
        }
        
        if data_type not in file_mapping:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return dataset_dir / file_mapping[data_type]
    
    def _load_raw_data(self, file_path: Path) -> Any:
        """Load raw data from file."""
        if file_path.suffix == '.pt':
            # .pt files might be pickled data, not torch tensors
            try:
                # First try torch.load
                return torch.load(file_path, weights_only=False, map_location='cpu')
            except:
                # If that fails, try pickle
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        elif file_path.suffix == '.pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _convert_to_pyg(
        self,
        raw_data: Any,
        dataset_name: str
    ) -> List[Data]:
        """Convert raw data to PyG format based on data structure."""
        # Determine target column based on dataset
        target_column = 'mature' if dataset_name == 'twitch' else 'gender'
        
        # Handle different data formats
        if isinstance(raw_data, list):
            # List of subgraphs
            if all(isinstance(item, tuple) and len(item) == 2 for item in raw_data):
                # List of (DataFrame, NetworkX) tuples
                return convert_subgraph_list(raw_data, target_column)
            elif all(isinstance(item, Data) for item in raw_data):
                # Already PyG Data objects
                return raw_data
            else:
                raise ValueError("Unsupported list data format")
        
        elif isinstance(raw_data, tuple) and len(raw_data) == 2:
            # Single (DataFrame, NetworkX) tuple
            df, graph = raw_data
            if isinstance(df, pd.DataFrame) and isinstance(graph, nx.Graph):
                data = convert_to_pyg_data(df, graph, target_column)
                return [data]
            else:
                raise ValueError("Invalid tuple format")
        
        elif isinstance(raw_data, Data):
            # Single PyG Data object
            return [raw_data]
        
        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary with dataset information
        """
        # Try to load a sample to get info
        try:
            # Try synth first as it should always exist
            data_list = self.load_dataset(dataset_name, 'synth')
            
            if not data_list:
                raise ValueError("Empty dataset")
            
            sample = data_list[0]
            
            # Collect all labels to determine number of classes
            all_labels = []
            for data in data_list:
                all_labels.extend(data.y.tolist())
            
            num_classes = len(set(all_labels))
            num_features = sample.x.size(1)
            
            return {
                'name': dataset_name,
                'num_classes': num_classes,
                'num_features': num_features,
                'num_graphs': len(data_list),
                'classification_target': 'mature' if dataset_name == 'twitch' else 'gender'
            }
        except Exception as e:
            print(f"Warning: Could not get dataset info: {e}")
            # Return default values
            return {
                'name': dataset_name,
                'num_classes': 2,
                'num_features': 7 if dataset_name == 'twitch' else 4,
                'num_graphs': 0,
                'classification_target': 'mature' if dataset_name == 'twitch' else 'gender'
            }
    
    def clear_cache(self):
        """Clear the dataset cache."""
        self._cache.clear()


# Convenience functions
def load_dataset(
    dataset_name: str,
    data_type: str = "train",
    base_path: str = "./datasets"
) -> List[Data]:
    """
    Load a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_type: Type of data to load
        base_path: Base directory for datasets
    
    Returns:
        List of PyG Data objects
    """
    loader = DatasetLoader(base_path)
    return loader.load_dataset(dataset_name, data_type)


def load_subgraphs(
    dataset_name: str,
    base_path: str = "./datasets"
) -> Dict[str, List[Data]]:
    """
    Load all subgraph types for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        base_path: Base directory for datasets
    
    Returns:
        Dictionary with train, test, and synth subgraphs
    """
    loader = DatasetLoader(base_path)
    
    result = {}
    
    # Try to load each type
    for data_type in ['train', 'nontrain', 'synth']:
        try:
            result[data_type] = loader.load_dataset(dataset_name, data_type)
            print(f"Loaded {len(result[data_type])} {data_type} subgraphs")
        except Exception as e:
            print(f"Warning: Could not load {data_type} data: {e}")
            result[data_type] = []
    
    return result


def get_dataset_info(
    dataset_name: str,
    base_path: str = "./datasets"
) -> Dict:
    """
    Get dataset information.
    
    Args:
        dataset_name: Name of the dataset
        base_path: Base directory for datasets
    
    Returns:
        Dictionary with dataset information
    """
    loader = DatasetLoader(base_path)
    return loader.get_dataset_info(dataset_name)


def validate_dataset_compatibility(
    target_data: List[Data],
    shadow_data: List[Data],
    test_data: List[Data]
) -> Dict:
    """
    Validate that datasets are compatible for MIA.
    
    Args:
        target_data: Target training data
        shadow_data: Shadow training data
        test_data: Test data
    
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'issues': []
    }
    
    # Check that we have data
    if not target_data:
        results['issues'].append("No target training data")
        results['valid'] = False
    
    if not shadow_data:
        results['issues'].append("No shadow training data")
        results['valid'] = False
    
    if not test_data:
        results['issues'].append("No test data")
        results['valid'] = False
    
    if not results['valid']:
        return results
    
    # Check feature dimensions
    target_dim = target_data[0].x.size(1) if target_data else 0
    shadow_dim = shadow_data[0].x.size(1) if shadow_data else 0
    test_dim = test_data[0].x.size(1) if test_data else 0
    
    if target_dim != shadow_dim or target_dim != test_dim:
        results['issues'].append(
            f"Feature dimension mismatch: target={target_dim}, "
            f"shadow={shadow_dim}, test={test_dim}"
        )
        results['valid'] = False
    
    # Check number of classes
    def get_num_classes(data_list):
        all_labels = []
        for data in data_list:
            all_labels.extend(data.y.tolist())
        return len(set(all_labels))
    
    target_classes = get_num_classes(target_data) if target_data else 0
    shadow_classes = get_num_classes(shadow_data) if shadow_data else 0
    test_classes = get_num_classes(test_data) if test_data else 0
    
    if target_classes != shadow_classes or target_classes != test_classes:
        results['issues'].append(
            f"Number of classes mismatch: target={target_classes}, "
            f"shadow={shadow_classes}, test={test_classes}"
        )
        results['valid'] = False
    
    # Add summary info
    results['summary'] = {
        'feature_dim': target_dim,
        'num_classes': target_classes,
        'target_graphs': len(target_data),
        'shadow_graphs': len(shadow_data),
        'test_graphs': len(test_data)
    }
    
    return results