import pathlib
import pickle
from typing import Optional
import torch

import networkx as nx
from torch_geometric.utils import to_networkx
import shutil
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset

from compress import Compress
from datasets.abstract_dataset import (AbstractTableGraphDataModule,
                                       AbstractTableGraphDataset)


class OgbnArxivDataset(AbstractTableGraphDataset):
    def __init__(
            self, split, root, num_graphs: int,
            min_nodes: int, max_nodes: int, hd: bool,
            sampler: str, compress: Optional[Compress], seed=1234,
            transform=None, pre_transform=None, pre_filter=None):
        super().__init__(
            split, root, num_graphs,
            min_nodes, max_nodes, hd,
            sampler,
            compress,
            {},
            seed,
            transform, pre_transform, pre_filter
        )

    def download(self):
        self.root: str
        y_col = "is_cv_lg_it_cl"
        data, *_ = PygNodePropPredDataset(name="ogbn-arxiv", root=self.root)
        y_col_vals = torch.isin(data.y, torch.tensor([16, 24, 28, 30], device=data.y.device))

        df = pd.DataFrame(
            torch.cat((data.x, y_col_vals), dim=-1).numpy(),
            columns=[f"c{i}" for i in range(data.num_node_features)] + [y_col])
        df = df.astype({y_col: bool}, copy=False)

        g_nx = to_networkx(data, to_undirected=True)
        set_largest_cc, *_ = sorted(nx.connected_components(g_nx), key=len, reverse=True)

        g_nx = g_nx.subgraph(set_largest_cc)
        df = df.loc[sorted(set_largest_cc)]

        shutil.rmtree(pathlib.Path(self.root) / "ogbn_arxiv")

        with open(self.raw_paths[0], 'wb') as f:
            pickle.dump((df, g_nx), f)


class OgbnArxivDataModule(AbstractTableGraphDataModule):
    def __init__(self, cfg, compress: Optional[Compress], hd: bool):
        super().__init__(cfg, OgbnArxivDataset, compress, hd)
