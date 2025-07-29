import pickle
from pathlib import Path
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_dataset(file_path):
    """
    Loads a dataset file (either .pickle or .pt) using pickle.
    Returns:
        - tuple[pandas.DataFrame, networkx.Graph] for *_original.pickle files
        - list[tuple[pandas.DataFrame, networkx.Graph]] for others
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def preprocess_features(df, dataset_name):
    """
    Preprocess features for torch_geometric conversion
    """
    df_processed = df.copy()
    
    if dataset_name == "twitch":
        # Convert boolean columns to int
        df_processed['mature'] = df_processed['mature'].astype(int)
        df_processed['dead_account'] = df_processed['dead_account'].astype(int)
        df_processed['affiliate'] = df_processed['affiliate'].astype(int)
        
        # Convert datetime to timestamp
        df_processed['created_at'] = pd.to_datetime(df_processed['created_at']).astype(np.int64) // 10**9
        df_processed['updated_at'] = pd.to_datetime(df_processed['updated_at']).astype(np.int64) // 10**9
        
        # Encode categorical language column
        le = LabelEncoder()
        df_processed['language'] = le.fit_transform(df_processed['language'])
        
        # Normalize large features
        df_processed['views'] = np.log1p(df_processed['views'])  # Log transform views
        
    elif dataset_name == "event":
        # Encode categorical columns
        le_locale = LabelEncoder()
        le_gender = LabelEncoder()
        df_processed['locale'] = le_locale.fit_transform(df_processed['locale'])
        df_processed['gender'] = le_gender.fit_transform(df_processed['gender'])
        
        # Convert datetime to timestamp
        df_processed['joinedAt'] = pd.to_datetime(df_processed['joinedAt']).astype(np.int64) // 10**9
        
        # Normalize features
        df_processed['birthyear'] = (df_processed['birthyear'] - df_processed['birthyear'].mean()) / df_processed['birthyear'].std()
        df_processed['timezone'] = (df_processed['timezone'] - df_processed['timezone'].mean()) / df_processed['timezone'].std()
    
    return df_processed

def convert_to_torch_geometric(df, graph, dataset_name, classification_target):
    """
    Convert pandas DataFrame and NetworkX graph to torch_geometric.Data format
    """
    # Preprocess features
    df_processed = preprocess_features(df, dataset_name)
    
    # Extract node features (exclude the classification target)
    feature_cols = [col for col in df_processed.columns if col != classification_target]
    x = torch.tensor(df_processed[feature_cols].values, dtype=torch.float)
    
    # Extract labels
    y = torch.tensor(df_processed[classification_target].values, dtype=torch.long)
    
    # Convert NetworkX edges to torch format
    edge_list = list(graph.edges())
    if len(edge_list) == 0:
        # Handle empty graphs
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create torch_geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

def load_dataset_for_attack(dataset_name, data_type):
    """
    Load and convert datasets for MIA attack
    
    Args:
        dataset_name: 'twitch' or 'event'
        data_type: 'train', 'nontrain', or 'synth'
    
    Returns:
        List of torch_geometric.Data objects
    """
    datasets_dir = Path("/Users/prateek/notes/uni-project-dec/datasets")
    
    if dataset_name == "twitch":
        classification_target = "mature"
    elif dataset_name == "event":
        classification_target = "gender"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load the appropriate file
    if data_type == "synth":
        file_path = datasets_dir / dataset_name / f"{dataset_name}_synth.pickle"
    else:
        file_path = datasets_dir / dataset_name / f"{dataset_name}_{data_type}.pt"
    
    print(f"Loading {dataset_name} {data_type} dataset from {file_path}")
    
    try:
        raw_data = load_dataset(file_path)
        
        # Handle different data formats
        if isinstance(raw_data, tuple):
            # Single graph format
            df, graph = raw_data
            datasets = [(df, graph)]
        else:
            # List of graphs format
            datasets = raw_data
        
        # Convert to torch_geometric format
        torch_datasets = []
        for df, graph in datasets:
            try:
                data = convert_to_torch_geometric(df, graph, dataset_name, classification_target)
                torch_datasets.append(data)
            except Exception as e:
                print(f"Error converting subgraph: {e}")
                continue
        
        print(f"Successfully converted {len(torch_datasets)} subgraphs")
        return torch_datasets
        
    except Exception as e:
        print(f"Error loading {dataset_name} {data_type}: {e}")
        return []

# Create a mock dataset class for rebMIGraph compatibility
class MockDataset:
    def __init__(self, data_list, num_classes, num_node_features):
        self.data_list = data_list
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.num_features = num_node_features  # Alias
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def __len__(self):
        return len(self.data_list)

def create_mock_dataset_for_rebmi(dataset_name):
    """
    Create mock datasets that rebMIGraph can use
    """
    print(f"\n=== Creating mock dataset for {dataset_name} ===")
    
    # Load synthetic data (for shadow model)
    synth_data = load_dataset_for_attack(dataset_name, "synth")
    
    if not synth_data:
        print("Failed to load synthetic data")
        return None
    
    # Get dataset properties from first sample
    sample = synth_data[0]
    num_classes = len(torch.unique(sample.y))
    num_node_features = sample.x.shape[1]
    
    print(f"Dataset properties: {num_classes} classes, {num_node_features} features")
    
    # Create a combined dataset (we'll use the first subgraph as representative)
    # In practice, rebMIGraph will need to be modified to handle multiple subgraphs
    combined_data = synth_data[0]  # Use first synthetic subgraph as base
    
    # Create mock dataset object
    mock_dataset = MockDataset([combined_data], num_classes, num_node_features)
    
    return mock_dataset, synth_data

if __name__ == "__main__":
    print("=== Dataset Bridge for MIA Attack ===")
    
    # Test the conversion pipeline
    for dataset_name in ["twitch", "event"]:
        print(f"\n--- Testing {dataset_name} dataset ---")
        
        # Test synthetic data loading
        synth_data = load_dataset_for_attack(dataset_name, "synth")
        if synth_data:
            print(f"Synthetic data: {len(synth_data)} subgraphs")
            print(f"First subgraph: {synth_data[0].x.shape[0]} nodes, {synth_data[0].x.shape[1]} features")
            print(f"Classes in first subgraph: {torch.unique(synth_data[0].y)}")
        
        # Create mock dataset for rebMIGraph
        mock_dataset, _ = create_mock_dataset_for_rebmi(dataset_name)
        if mock_dataset:
            print(f"Mock dataset created successfully")
            print(f"Num classes: {mock_dataset.num_classes}")
            print(f"Num features: {mock_dataset.num_node_features}")
