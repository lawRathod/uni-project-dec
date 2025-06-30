import pickle
from pathlib import Path
import pandas as pd
import networkx as nx

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

print("Loading datasets...")

# twitch original
print("Loading twitch original dataset...")
dataset: tuple[pd.DataFrame, nx.Graph] = load_dataset("/home/law/project-mia/datasets/twitch/twitch_original.pickle")
dataframe, graph = dataset
print(dataframe.head())
print(graph)

# twitch train
print("Loading twitch train dataset...")
dataset: list[tuple[pd.DataFrame, nx.Graph]] = load_dataset("/home/law/project-mia/datasets/twitch/twitch_train.pt")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# twitch non train
print("Loading twitch non-train dataset...")
dataset = load_dataset("/home/law/project-mia/datasets/twitch/twitch_nontrain.pt")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# twitch synthetic
print("Loading twitch synthetic dataset...")
dataset = load_dataset("/home/law/project-mia/datasets/twitch/twitch_synth.pickle")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# event original
print("Loading event original dataset...")
dataset = load_dataset("/home/law/project-mia/datasets/event/event_original.pickle")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset
print(dataframe.head())
print(graph)

# event train
print("Loading event train dataset...")
dataset = load_dataset("/home/law/project-mia/datasets/event/event_train.pt")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# event non train
print("Loading event non-train dataset...")
dataset = load_dataset("/home/law/project-mia/datasets/event/event_nontrain.pt")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# event synthetic
print("Loading event synthetic dataset...")
dataset = load_dataset("/home/law/project-mia/datasets/event/event_synth.pickle")
print(f"Number of items in dataset: {len(dataset)}")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)
