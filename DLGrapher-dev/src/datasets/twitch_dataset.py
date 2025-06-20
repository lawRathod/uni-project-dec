import pathlib
import pickle
from typing import Optional

import networkx as nx
import pandas as pd
from torch_geometric.data import download_url, extract_zip

from compress import Compress
from datasets.abstract_dataset import (AbstractTableGraphDataModule,
                                       AbstractTableGraphDataset)


class TwitchDataset(AbstractTableGraphDataset):
    def __init__(
            self, split, root, num_graphs: int,
            min_nodes: int, max_nodes: int, hd: bool,
            sampler: str, compress: Optional[Compress], seed=1234,
            transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            split, root, num_graphs,
            min_nodes, max_nodes, hd,
            sampler, compress,
            {'views': 'log', 'updated_at': 'clip'},
            seed,
            transform, pre_transform, pre_filter
        )

    def download(self):
        zip_path = download_url('https://snap.stanford.edu/data/twitch_gamers.zip', self.raw_dir)
        extract_zip(zip_path, self.raw_dir)
        raw_dir_path = pathlib.Path(self.raw_dir)
        (raw_dir_path / 'twitch_gamers.zip').unlink()
        (raw_dir_path / 'README.txt').unlink()

        df = pd.read_csv(raw_dir_path / 'large_twitch_features.csv', index_col='numeric_id')
        df = df.astype(
        {
            'created_at': 'datetime64', 'updated_at': 'datetime64', 'language': 'category',
            'mature': 'bool', 'dead_account': 'bool', 'affiliate': 'bool'
        },
        copy=False)

        g_nx: nx.Graph = nx.read_edgelist(
            raw_dir_path / 'large_twitch_edges.csv', comments='n', delimiter=',', nodetype=int)

        with open(self.raw_paths[0], 'wb') as f:
            pickle.dump((df, g_nx), f)


class TwitchDataModule(AbstractTableGraphDataModule):
    def __init__(self, cfg, compress: Optional[Compress], hd: bool):
        super().__init__(cfg, TwitchDataset, compress, hd)
