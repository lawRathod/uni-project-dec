"""
Core data conversion utilities for transforming between different graph data formats.
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler


def convert_to_pyg_data(
    df: pd.DataFrame, 
    graph: nx.Graph,
    target_column: str = 'mature',
    normalize: bool = True
) -> Data:
    """
    Convert (DataFrame, NetworkX graph) tuple to PyTorch Geometric Data object.
    
    Args:
        df: DataFrame containing node features and labels
        graph: NetworkX graph containing edge information
        target_column: Name of the target column for classification
        normalize: Whether to normalize features
    
    Returns:
        PyTorch Geometric Data object
    """
    # Handle different target column names
    if target_column not in df.columns:
        # Try alternative names
        if 'gender' in df.columns:
            target_column = 'gender'
        elif 'label' in df.columns:
            target_column = 'label'
        elif 'y' in df.columns:
            target_column = 'y'
        else:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame. "
                           f"Available columns: {df.columns.tolist()}")
    
    # Extract features (all columns except target and node_id if present)
    feature_columns = [col for col in df.columns 
                      if col not in [target_column, 'node_id', 'index']]
    
    if not feature_columns:
        raise ValueError("No feature columns found in DataFrame")
    
    # Preprocess features to handle non-numeric data
    df_features = df[feature_columns].copy()
    
    for col in df_features.columns:
        # Handle timestamp columns
        if pd.api.types.is_datetime64_any_dtype(df_features[col]):
            # Convert to numeric timestamp
            df_features[col] = df_features[col].astype(np.int64) / 10**9
        # Handle string/categorical columns
        elif df_features[col].dtype == object or pd.api.types.is_categorical_dtype(df_features[col]):
            # Use label encoding for categorical features
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
        # Handle boolean columns
        elif df_features[col].dtype == bool:
            df_features[col] = df_features[col].astype(int)
    
    # Convert features to numpy array
    features = df_features.values.astype(np.float32)
    
    # Replace any NaN values with 0
    features = np.nan_to_num(features, nan=0.0)
    
    if normalize:
        features = normalize_features(features)
    
    x = torch.tensor(features, dtype=torch.float32)
    
    # Convert labels to tensor
    y = df[target_column].values
    
    # Handle different label types
    if y.dtype == bool:
        y = y.astype(int)
    elif y.dtype == object or pd.api.types.is_categorical_dtype(df[target_column]):
        # Handle string labels (e.g., 'male', 'female')
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    elif not np.issubdtype(y.dtype, np.integer):
        # Convert float labels to int
        y = y.astype(int)
    
    y = torch.tensor(y, dtype=torch.long)
    
    # Convert edges to tensor
    # Create a mapping from node labels to indices if needed
    node_mapping = {}
    if graph.number_of_nodes() > 0:
        graph_nodes = list(graph.nodes())
        
        # Check if nodes are already 0-indexed integers
        if all(isinstance(n, (int, np.integer)) for n in graph_nodes):
            min_node = min(graph_nodes)
            if min_node != 0:
                # Nodes need reindexing
                node_mapping = {node: i for i, node in enumerate(sorted(graph_nodes))}
            else:
                # Nodes are likely already 0-indexed
                node_mapping = {node: node for node in graph_nodes}
        else:
            # Non-integer nodes, create mapping
            node_mapping = {node: i for i, node in enumerate(graph_nodes)}
    
    # Convert edges using the mapping
    edge_list = []
    for u, v in graph.edges():
        if node_mapping:
            u_idx = node_mapping.get(u, u)
            v_idx = node_mapping.get(v, v)
        else:
            u_idx, v_idx = u, v
        
        # Add both directions for undirected graph
        edge_list.append([u_idx, v_idx])
        edge_list.append([v_idx, u_idx])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # No edges - create empty edge index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Validate the data
    num_nodes = x.size(0)
    if edge_index.size(1) > 0:
        max_edge_idx = edge_index.max().item()
        if max_edge_idx >= num_nodes:
            print(f"Warning: Edge index {max_edge_idx} >= num_nodes {num_nodes}")
            # Filter out invalid edges
            valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_edges]
            data.edge_index = edge_index
    
    return data


def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize feature matrix.
    
    Args:
        features: Feature matrix to normalize
        method: Normalization method ('standard', 'minmax', or 'none')
    
    Returns:
        Normalized feature matrix
    """
    if method == 'none':
        return features
    
    if method == 'standard':
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    elif method == 'minmax':
        # Min-max normalization
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        return (features - min_vals) / range_vals
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def align_feature_dimensions(
    data_list: List[Data],
    target_dim: Optional[int] = None
) -> List[Data]:
    """
    Ensure all Data objects have the same feature dimensions.
    
    Args:
        data_list: List of PyG Data objects
        target_dim: Target feature dimension (if None, uses max dimension)
    
    Returns:
        List of Data objects with aligned feature dimensions
    """
    if not data_list:
        return data_list
    
    # Find the maximum feature dimension
    max_dim = max(data.x.size(1) for data in data_list)
    
    if target_dim is None:
        target_dim = max_dim
    elif target_dim < max_dim:
        print(f"Warning: target_dim {target_dim} < max_dim {max_dim}, using {max_dim}")
        target_dim = max_dim
    
    aligned_data = []
    for data in data_list:
        current_dim = data.x.size(1)
        
        if current_dim == target_dim:
            aligned_data.append(data)
        elif current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(data.x.size(0), target_dim - current_dim)
            new_x = torch.cat([data.x, padding], dim=1)
            
            new_data = Data(
                x=new_x,
                edge_index=data.edge_index,
                y=data.y
            )
            aligned_data.append(new_data)
        else:
            # Truncate features
            new_x = data.x[:, :target_dim]
            
            new_data = Data(
                x=new_x,
                edge_index=data.edge_index,
                y=data.y
            )
            aligned_data.append(new_data)
    
    return aligned_data


def convert_subgraph_list(
    subgraph_list: List[Tuple[pd.DataFrame, nx.Graph]],
    target_column: str = 'mature',
    normalize: bool = True,
    align_features: bool = True
) -> List[Data]:
    """
    Convert a list of (DataFrame, NetworkX) tuples to PyG Data objects.
    
    Args:
        subgraph_list: List of (DataFrame, NetworkX graph) tuples
        target_column: Target column name for labels
        normalize: Whether to normalize features
        align_features: Whether to align feature dimensions
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    if not subgraph_list:
        return []
    
    data_list = []
    for i, (df, graph) in enumerate(subgraph_list):
        try:
            data = convert_to_pyg_data(df, graph, target_column, normalize)
            data_list.append(data)
        except Exception as e:
            print(f"Warning: Failed to convert subgraph {i}: {e}")
            continue
    
    if align_features and data_list:
        data_list = align_feature_dimensions(data_list)
    
    return data_list


def validate_data_consistency(data_list: List[Data]) -> dict:
    """
    Validate consistency across a list of Data objects.
    
    Args:
        data_list: List of PyG Data objects
    
    Returns:
        Dictionary with validation results
    """
    if not data_list:
        return {"valid": False, "reason": "Empty data list"}
    
    # Check feature dimensions
    feature_dims = [data.x.size(1) for data in data_list]
    if len(set(feature_dims)) > 1:
        return {
            "valid": False,
            "reason": f"Inconsistent feature dimensions: {set(feature_dims)}"
        }
    
    # Check label values
    all_labels = []
    for data in data_list:
        all_labels.extend(data.y.tolist())
    unique_labels = set(all_labels)
    
    # Check for valid edges
    for i, data in enumerate(data_list):
        if data.edge_index.size(1) > 0:
            max_idx = data.edge_index.max().item()
            num_nodes = data.x.size(0)
            if max_idx >= num_nodes:
                return {
                    "valid": False,
                    "reason": f"Invalid edge indices in data {i}: max {max_idx} >= nodes {num_nodes}"
                }
    
    return {
        "valid": True,
        "feature_dim": feature_dims[0],
        "num_classes": len(unique_labels),
        "unique_labels": sorted(list(unique_labels)),
        "num_graphs": len(data_list)
    }