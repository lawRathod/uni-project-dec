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

# twitch original
dataset: tuple[pd.DataFrame, nx.Graph] = load_dataset("/home/law/project-mia/datasets/twitch/twitch_original.pickle")
dataframe, graph = dataset
print(dataframe.head())
print(graph)

# twitch train
dataset: list[tuple[pd.DataFrame, nx.Graph]] = load_dataset("/home/law/project-mia/datasets/twitch/twitch_train.pt")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# twitch non train
dataset = load_dataset("/home/law/project-mia/datasets/twitch/twitch_nontrain.pt")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# twitch synthetic
dataset = load_dataset("/home/law/project-mia/datasets/twitch/twitch_synth.pickle")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# event original
dataset = load_dataset("/home/law/project-mia/datasets/event/event_original.pickle")
dataframe, graph = dataset
print(dataframe.head())
print(graph)

# event train
dataset = load_dataset("/home/law/project-mia/datasets/event/event_train.pt")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# event non train
dataset = load_dataset("/home/law/project-mia/datasets/event/event_nontrain.pt")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)

# event synthetic
dataset = load_dataset("/home/law/project-mia/datasets/event/event_synth.pickle")
dataframe, graph = dataset[0]
print(dataframe.head())
print(graph)
