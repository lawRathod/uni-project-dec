from typing import Iterable

import nets_eval_common
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.lightning import LightningDataset

import src.utils as utils
from src.diffusion.distributions import DistributionNodes


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, datasets):
        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],
                         batch_size=cfg.train.batch_size if 'debug' not in cfg.general.name else 2,
                         num_workers=cfg.train.num_workers,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def node_counts(self):
        max_nodes_possible = max(
            cast(int, cast(Data, d).num_nodes)
            for ds in (self.train_dataset, self.val_dataset, self.test_dataset)
            for d in cast(Dataset, ds)
        ) + 1
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}

import os
import pathlib
import pickle
import random
from typing import Optional, Union, cast

import networkx as nx
import numpy as np
import pandas as pd
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset

from compress import Compress
from graph_sampling import SRW_RWF_ISRW


class AbstractTableGraphDataset(InMemoryDataset):
    file_idx = {
        s: i for i, s in enumerate(('train', 'val', 'test', 'og_train', 'og_val', 'og_test'))}
    tti_id = len(file_idx)

    def __init__(
            self, split, root, num_graphs: int,
            min_nodes: int, max_nodes: int,
            hd: bool, sampler: str, compress: Optional[Compress],
            col_to_preproc: Optional[dict[str, utils.PREPROC_TYPES]]=None,
            seed=1234,
            transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.hd = hd
        self.sampler = sampler
        self.compress = compress
        # Each class is expected to have a static preprocessing assignment
        self.col_to_preproc = col_to_preproc or {}
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(
            self.processed_paths[self.file_idx[self.split]])

    @property
    def raw_file_names(self):
        return ["table_graph.pickle"]

    @property
    def processed_file_names(self):
        return [
            (   f"{split}-{self.num_graphs}g-{self.min_nodes}_{self.max_nodes}n"
                f"-{self.sampler}-{'y' if self.hd else 'n'}hd-{self.seed}s"
                f"-{self.compress.get_kind() if self.compress else 'no'}c.pt")
            for split in self.file_idx
        ] + [f"tti.pickle"]

    def process(self) -> None:
        df: pd.DataFrame
        g_nx: nx.Graph

        with open(self.raw_paths[0], 'rb') as f:
            df, g_nx = pickle.load(f)

        g_nx_no_attr = g_nx.copy()

        bool_cols = list(df.select_dtypes('bool'))
        cat_cols = cast(list[str], list(df.select_dtypes('category')))
        dt_cols = list(df.select_dtypes('datetime64'))
        dtypes = df.dtypes
        df_no_bool = df.astype(
            {dt: 'uint8' for dt in df.select_dtypes(include='bool').columns},
            copy=False)
        df_oh = pd.get_dummies(df_no_bool)
        utils.preproc_do(df_oh, self.col_to_preproc)
        s_min = df_oh.min()
        s_max = df_oh.max()
        learning_dtypes = df_oh.dtypes

        # Destroys dtypes
        df_norm = (df_oh - s_min) / (s_max - s_min)

        with open(self.processed_paths[self.tti_id], 'wb') as f:
            pickle.dump(utils.TabTransfInfo(
                learning_dtypes,
                s_min, s_max,
                self.col_to_preproc,
                bool_cols, dt_cols, cat_cols,
                dtypes
            ), f)

        if self.hd:
            nx.set_node_attributes(g_nx, df_norm.to_dict(orient='index'))
        else:
            nx.set_node_attributes(g_nx, 1., 'attr')

        nx.set_edge_attributes(g_nx, {(u, v): {"one": 0., "other": 1.} for u, v in g_nx.edges})
        samples_nx: list[nx.Graph]

        if self.sampler == "randomwalk":
            np_rng = np.random.default_rng(self.seed)
            random.seed(self.seed)
            obj = SRW_RWF_ISRW()
            samples_nx = [
                obj.random_walk_induced_graph_sampling(g_nx, n)
                for n in np_rng.integers(
                    low=self.min_nodes, high=cast(int, self.max_nodes),
                    size=self.num_graphs, endpoint=True)
            ]
        elif self.sampler == "neighborhood":
            samples_nx = nets_eval_common.NeighborhoodSampler(
                n_hops=2,
                min_nodes_abs=self.min_nodes,
                max_nodes_abs=self.max_nodes)._sample_single(g_nx)

            if self.num_graphs != -1:
                samples_nx = samples_nx[:self.num_graphs]
            else:
                self.num_graphs = len(samples_nx)
        else:
            raise ValueError("Unknown sampler")

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        data_list = []
        table_graphs_og: list[tuple[pd.DataFrame, nx.Graph]] = []

        for sample_nx in samples_nx:
            data = torch_geometric.utils.from_networkx(
                sample_nx,
                group_node_attrs=all if self.hd else ['attr'],
                group_edge_attrs=all)
            data.x = data.x.float()
            data.y = torch.zeros([1, 0], dtype=torch.float)

            if self.compress:
                data = self.compress.compress(data)

                if not self.hd:
                    data.x[:, 0] -= data.x[:, 1]

            data_list.append(data)
            table_graphs_og.append((
                df.loc[list(sample_nx.nodes)],
                g_nx_no_attr.edge_subgraph(sample_nx.edges).copy()
            ))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list[:train_len]),
                   self.processed_paths[self.file_idx['train']])
        torch.save(self.collate(data_list[train_len:train_len + val_len]),
                   self.processed_paths[self.file_idx['val']])
        torch.save(self.collate(data_list[train_len + val_len:]),
                   self.processed_paths[self.file_idx['test']])

        with open(self.processed_paths[self.file_idx['og_train']], 'wb') as f:
            pickle.dump(table_graphs_og[:train_len], f)

        with open(self.processed_paths[self.file_idx['og_val']], 'wb') as f:
            pickle.dump(table_graphs_og[train_len:train_len + val_len], f)

        with open(self.processed_paths[self.file_idx['og_test']], 'wb') as f:
            pickle.dump(table_graphs_og[train_len + val_len:], f)

    def load_tti(self) -> utils.TabTransfInfo:
        with open(self.processed_paths[self.tti_id], "rb") as f:
            return pickle.load(f)


class AbstractTableGraphDataModule(AbstractDataModule):
    def __init__(
            self, cfg,
            dataset_class: type[AbstractTableGraphDataset],
            compress: Optional[Compress], hd: bool):
        self.cfg = cfg
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, cfg.dataset.datadir)

        datasets = {
            'train': dataset_class(
                split='train', root=root_path,
                num_graphs=cfg.general.num_graphs,
                min_nodes=cfg.general.min_nodes,
                max_nodes=cfg.general.max_nodes,
                sampler=cfg.general.sampler,
                hd=hd,
                seed=cfg.dataset.seed,
                compress=compress
            ),
            'val': dataset_class(
                split='val', root=root_path,
                num_graphs=cfg.general.num_graphs,
                min_nodes=cfg.general.min_nodes,
                max_nodes=cfg.general.max_nodes,
                sampler=cfg.general.sampler,
                hd=hd,
                seed=cfg.dataset.seed,
                compress=compress
            ),
            'test': dataset_class(
                split='test', root=root_path,
                num_graphs=cfg.general.num_graphs,
                min_nodes=cfg.general.min_nodes,
                max_nodes=cfg.general.max_nodes,
                sampler=cfg.general.sampler,
                hd=hd,
                seed=cfg.dataset.seed,
                compress=compress
            )
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class TableGraphDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: AbstractTableGraphDataModule) -> None:
        # self.name = 'nx_graphs'
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()    # There are no node types
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
        self.tti = cast(AbstractTableGraphDataset, datamodule.inner).load_tti()

        self.bool_indices: list[int] = []
        self.cat_indices: list[list[int]] = []
        self.numeric_indices: list[int] = []
        i = 0
        last_cat_prefix = ''

        while i < len(self.tti.learning_dtypes):
            c = cast(str, self.tti.learning_dtypes.index[i])

            if c in self.tti.bool_cols:
                self.bool_indices.append(i)
            else:
                prefix, *_ = c.rsplit('_', 1)
                if prefix == last_cat_prefix:
                    self.cat_indices[-1].append(i)
                elif prefix in self.tti.cat_cols:
                    last_cat_prefix = prefix
                    self.cat_indices.append([i])
                else:
                    self.numeric_indices.append(i)

            i += 1

        super().__init__()

    def restore_table(
            self, node_data: Union[Iterable, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(node_data, pd.DataFrame):
            return node_data

        tti = self.tti
        df = pd.DataFrame(node_data, columns=tti.learning_dtypes.index)
        df = df * (tti.s_max - tti.s_min) + tti.s_min
        utils.preproc_undo(df, tti.col_to_preproc)

        for cat_col in tti.cat_cols:
            cat_val_cols = [c for c in df.columns if c.startswith(f"{cat_col}_")]
            df[cat_col] = df[cat_val_cols].astype(float, copy=False)\
                .idxmax(axis='columns').str.removeprefix(f'{cat_col}_')
            df = df.drop(columns=cat_val_cols)

        df[tti.bool_cols] = df[tti.bool_cols] > 0.5
        df = df[tti.dtypes.index]
        # df = df.round(0)
        df = df.astype(tti.dtypes, copy=False)

        for dt_col in tti.dt_cols:
            df[dt_col] = df[dt_col].dt.normalize()

        return df
