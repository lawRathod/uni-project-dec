import pathlib
import pickle
from typing import Optional

import networkx as nx
import pandas as pd

from compress import Compress
from datasets.abstract_dataset import (AbstractTableGraphDataModule,
                                       AbstractTableGraphDataset)


# https://relbench.stanford.edu/datasets/rel-event/
# https://www.kaggle.com/competitions/event-recommendation-engine-challenge
class EventDataset(AbstractTableGraphDataset):
    def __init__(
            self, split, root, num_graphs: int,
            min_nodes: int, max_nodes: int, hd: bool,
            sampler: str, compress: Optional[Compress], seed=1234,
            transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            split, root, num_graphs,
            min_nodes, max_nodes, hd,
            sampler, compress,
            {'joinedAt': 'clip'},
            seed,
            transform, pre_transform, pre_filter
        )

    # def process(self):
    #     raw_data_dir = Path(__file__).parents[2] / "raw_data"
    #     g_nx: nx.Graph = nx.read_adjlist(
    #         raw_data_dir / 'user_friends.txt.bz2',
    #         comments='u', nodetype=int)
    #     hmm = list(len(x) for x in nx.connected_components(g_nx))
    #     print(len(hmm), min(hmm), max(hmm))
    #     df = load_csv_df(raw_data_dir / 'users.csv.bz2')
    #     nodes_to_remove = set(g_nx.nodes) - set(df.index)
    #     g_nx.remove_nodes_from(nodes_to_remove)
    #     hmm = list(len(x) for x in nx.connected_components(g_nx))
    #     print(len(hmm), min(hmm), max(hmm))
    #     exit()

    #     self.process_from_graph_table(g_nx, df)
    #     pass

    def download(self):
        raw_data_dir = pathlib.Path(__file__).parents[2] / "raw_data"
        g_nx: nx.Graph = nx.read_adjlist(
            raw_data_dir / 'user_friends.txt.bz2',
            comments='u', nodetype=int)

        df = pd.read_csv(raw_data_dir / 'users.csv.bz2', index_col='user_id')
        df = df.drop(columns=['location'])
        df = df.dropna()
        df = df[
            (df['birthyear'] != 'None') &
            (df['joinedAt'] != 'None') &
            (df['birthyear'] != '23-May')
        ]

        g_nx = g_nx.subgraph(df.index)

        set_largest_cc, *_ = sorted(nx.connected_components(g_nx), key=len, reverse=True)
        g_nx = g_nx.subgraph(set_largest_cc)
        df = df.loc[sorted(set_largest_cc)]

        df = df.astype(
            {
                'locale': 'category', 'birthyear': 'int', 'gender': 'category',
                'joinedAt': 'datetime64', 'timezone': 'int'
            },
            copy=False)
        df['joinedAt'] = df['joinedAt'].dt.normalize()

        with open(self.raw_paths[0], 'wb') as f:
            pickle.dump((df, g_nx.copy()), f)


class EventDataModule(AbstractTableGraphDataModule):
    def __init__(self, cfg, compress: Optional[Compress], hd: bool):
        super().__init__(cfg, EventDataset, compress, hd)
