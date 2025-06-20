import os
import pathlib
from typing import cast

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import KarateClub

from src.datasets.abstract_dataset import (AbstractDataModule,
                                           AbstractDatasetInfos)


class KarateClubDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
            return ['processed.pt']

    def process(self):
        data_list = []

        for data in KarateClub():
            data = cast(Data, data)
            del data.train_mask
            data['x'] = torch.nn.functional.one_hot(data.y).float()
            data['y'] = torch.zeros([1, 0]).float()
            data['edge_attr'] = torch.zeros(data.edge_index.shape[-1], 2, dtype=torch.float)
            data.edge_attr[:, 1] = 1
            data.n_nodes = data.num_nodes * torch.ones(1, dtype=torch.long)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class KarateClubDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': KarateClubDataset(root=root_path),
                    'val': KarateClubDataset(root=root_path),
                    'test': KarateClubDataset(root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class KarateClubDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()    # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
