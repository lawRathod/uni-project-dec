import os
import pathlib
import shutil
from random import Random

import networkx as nx
import torch
import torch_geometric.data
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url, extract_tar

from src.datasets.abstract_dataset import AbstractDataModule

DATASET_NAME_TO_DOWNLOAD_SUFFIX = {
    'ego-twitter': 'twitter'
}
FILE_IDX = {'train': 0, 'val': 1, 'test': 2}


class EgoDataset(InMemoryDataset):
    def __init__(
            self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        self.root: str
        suffix = DATASET_NAME_TO_DOWNLOAD_SUFFIX[self.dataset_name]
        file_path = download_url(f'https://snap.stanford.edu/data/{suffix}.tar.gz', self.raw_dir)
        extract_tar(file_path, self.raw_dir)
        dir_path = pathlib.Path(self.raw_dir) / suffix
        graph_structures_list: list[torch.Tensor] = []

        for f in dir_path.glob('*.edges'):
            g: nx.Graph = nx.read_edgelist(f)
            g.add_edges_from([(f.stem, v) for v in g.nodes])
            g_relabeled = nx.convert_node_labels_to_integers(g)
            edges = torch.tensor(list(g_relabeled.edges)).T
            edges_undir = torch.cat((edges, edges.flip(0)), dim=-1)
            graph_structures_list.append(edges_undir)

        (pathlib.Path(self.raw_dir) / f'{suffix}.tar.gz').unlink()
        shutil.rmtree(dir_path)
        test_len = int(round(len(graph_structures_list) * 0.2))
        train_len = int(round((len(graph_structures_list) - test_len) * 0.8))
        val_len = len(graph_structures_list) - train_len - test_len
        Random(0).shuffle(graph_structures_list)
        torch.save(
            graph_structures_list[:train_len], self.raw_paths[FILE_IDX['train']])
        torch.save(
            graph_structures_list[train_len:train_len + val_len], self.raw_paths[FILE_IDX['val']])
        torch.save(
            graph_structures_list[-test_len:], self.raw_paths[FILE_IDX['test']])

    def process(self):
        raw_dataset: list[torch.Tensor] = torch.load(self.raw_paths[FILE_IDX[self.split]])
        data_list = []

        for edge_index in raw_dataset:
            n = int(edge_index.max()) + 1
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=n_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class EgoDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            'train': EgoDataset(
                dataset_name=self.cfg.dataset.name, split='train', root=root_path),
            'val': EgoDataset(
                dataset_name=self.cfg.dataset.name, split='val', root=root_path),
            'test': EgoDataset(
                dataset_name=self.cfg.dataset.name, split='test', root=root_path)}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
