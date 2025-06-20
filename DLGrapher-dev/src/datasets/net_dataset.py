from typing import Optional, cast

import nets_eval_common
import networkx as nx
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData

from compress import Compress
from src.datasets.abstract_dataset import (AbstractDataModule,
                                           AbstractDatasetInfos)


class NetDataset(InMemoryDataset):
    def __init__(
            self, dataset_name, nx_graphs: list[nx.Graph],
            compress: Optional[Compress],
            transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.compress = compress
        super().__init__(None, transform, pre_transform, pre_filter)

        for nx_g in nx_graphs:
            nx.set_node_attributes(nx_g, 1., 'dummy')
            nx.set_edge_attributes(nx_g, [0., 1.], 'dummy')

        data_list = [
            torch_geometric.utils.from_networkx(
                nx_g, group_node_attrs=['dummy'], group_edge_attrs=['dummy'])
            for nx_g in nx_graphs
        ]

        for data in data_list:
            data.y = torch.zeros([1, 0], dtype=torch.float)

        if self.compress:
            data_list = [self.compress.compress(data) for data in data_list]

            for data in data_list:
                data.x[:, 0] -= data.x[:, 1]

        self.data, self.slices = self.collate(cast(list[BaseData], data_list))


class NetDataModule(AbstractDataModule):
    def __init__(self, cfg, compress: Optional[Compress]):
        self.cfg = cfg
        dataset_name = self.cfg.dataset.name

        min_nodes: int = self.cfg.general.get("min_nodes", 0)
        max_nodes: Optional[int] = self.cfg.general.get("max_nodes", None)

        if min_nodes or max_nodes:
            sampler = nets_eval_common.NeighborhoodSampler(
                min_nodes_abs=min_nodes, max_nodes_abs=max_nodes)
        else:
            sampler = None

        train_nx_graphs, eval_nx_graphs, test_nx_graphs = nets_eval_common.load_dataset(
            dataset_name, sampler)

        datasets = {
            'train': NetDataset(dataset_name, train_nx_graphs, compress),
            'val': NetDataset(dataset_name, eval_nx_graphs, compress),
            'test': NetDataset(dataset_name, test_nx_graphs, compress)
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class NetDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        # self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()   # There are no node types
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
