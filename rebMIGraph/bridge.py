#!/usr/bin/env python3
"""
Bridge adapter for loading pickle datasets (Twitch, Event) into TSTS.py format
Handles (pandas.DataFrame, networkx.Graph) tuples from DLGrapher synthetic data
"""

import pickle
import torch
from torch_geometric.utils import from_networkx

class PickleDatasetLoader:
    """Loads and converts pickle datasets to torch_geometric format"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        self.base_path = f"../datasets/{self.dataset_name}"
        
    def load_pickle(self, file_suffix):
        """Load pickle file with error handling"""
        file_path = f"{self.base_path}/{self.dataset_name}_{file_suffix}.pickle"
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
            
    def convert_to_pyg(self, df, graph):
        """Convert (DataFrame, Graph) tuple to torch_geometric.Data"""
        # Convert networkx to torch_geometric
        data = from_networkx(graph)
        
        # Add node features from DataFrame
        if 'x' not in data:
            # Use all numeric columns except target as features
            feature_cols = [col for col in df.columns if col not in ['mature', 'gender', 'affiliate'] and df[col].dtype in ['int64', 'float64']]
            data.x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        
        # Add target labels
        if 'affiliate' in df.columns:  # Twitch dataset - use affiliate instead of mature
            data.y = torch.tensor(df['affiliate'].values, dtype=torch.long)
        elif 'gender' in df.columns:  # Event dataset
            # Convert string gender to numeric
            if df['gender'].dtype.name == 'category' or df['gender'].dtype == object:
                gender_map = {'male': 0, 'female': 1}
                data.y = torch.tensor(df['gender'].map(gender_map).values, dtype=torch.long)
            else:
                data.y = torch.tensor(df['gender'].values, dtype=torch.long)
            
        return data
        
    def load_dataset(self):
        """Load all dataset splits and return in TSTS format"""
        # Load training subgraphs 
        train_data = self.load_pickle("train")
        train_list = []
        if train_data and isinstance(train_data, list):
            for df, graph in train_data:
                train_list.append(self.convert_to_pyg(df, graph))
                
        # Load non-training subgraphs
        nontrain_data = self.load_pickle("nontrain") 
        nontrain_list = []
        if nontrain_data and isinstance(nontrain_data, list):
            for df, graph in nontrain_data:
                nontrain_list.append(self.convert_to_pyg(df, graph))
                
        # Load synthetic subgraphs
        synth_data = self.load_pickle("synth")
        synth_list = []
        if synth_data and isinstance(synth_data, list):
            for df, graph in synth_data:
                synth_list.append(self.convert_to_pyg(df, graph))
        
        # Get features and classes from first training sample
        if train_list:
            sample = train_list[0]
            num_features = sample.x.shape[1]
            num_classes = len(torch.unique(sample.y))
        else:
            num_features = num_classes = 0
        
        return {
            'train': train_list, 
            'nontrain': nontrain_list,
            'synth': synth_list,
            'num_features': num_features,
            'num_classes': num_classes
        }

def load_custom_dataset(dataset_name):
    """Main function to load Twitch/Event datasets for TSTS.py"""
    loader = PickleDatasetLoader(dataset_name)
    return loader.load_dataset()

if __name__ == "__main__":
    # Test loading
    for dataset in ['twitch', 'event']:
        print(f"\nTesting {dataset} dataset:")
        loader = PickleDatasetLoader(dataset)
        
        # Load raw data to inspect columns
        train_data = loader.load_pickle("train")
        if train_data and isinstance(train_data, list) and len(train_data) > 0:
            sample_df, sample_graph = train_data[0]
            print(f"DataFrame columns: {list(sample_df.columns)}")
            print(f"DataFrame dtypes:")
            for col, dtype in sample_df.dtypes.items():
                print(f"  {col}: {dtype}")
            print(f"Sample DataFrame shape: {sample_df.shape}")
        
        data = load_custom_dataset(dataset)
        if data:
            print(f"Train subgraphs: {len(data['train'])}")
            print(f"Nontrain subgraphs: {len(data['nontrain'])}")
            print(f"Synth subgraphs: {len(data['synth'])}")
            print(f"Features: {data['num_features']}, Classes: {data['num_classes']}")