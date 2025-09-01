"""
Bridge module for converting between dataset formats.
Provides utilities to convert DLGrapher outputs to rebMIGraph-compatible formats.
"""

from .data_converter import (
    convert_to_pyg_data,
    normalize_features,
    align_feature_dimensions
)

from .dataset_loader import (
    load_dataset,
    load_subgraphs,
    get_dataset_info
)

from .rebmi_adapter import (
    create_rebmi_dataset,
    get_inductive_split_from_subgraphs,
    create_data_masks
)

__all__ = [
    'convert_to_pyg_data',
    'normalize_features', 
    'align_feature_dimensions',
    'load_dataset',
    'load_subgraphs',
    'get_dataset_info',
    'create_rebmi_dataset',
    'get_inductive_split_from_subgraphs',
    'create_data_masks'
]