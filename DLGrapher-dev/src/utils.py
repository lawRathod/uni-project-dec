import os
from typing import Literal
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device)
    E[:, diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def unnormalize_collapse_partial(X, E, y, norm_values, norm_biases, node_mask):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]
    placeholder = PlaceHolder(X=X, E=E, y=y)
    placeholder.mask_collapse_partial(node_mask)

    return placeholder


def unnormalize_no_mask(X, E, y, norm_values, norm_biases):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device)
    E[:, diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def mask_collapse_partial(self, node_mask: torch.Tensor) -> None:
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        self.E = torch.argmax(self.E, dim=-1)
        self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1

        self.X = self.X * x_mask


def setup_wandb(cfg, mode: Literal['fit', 'test']):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': f'{cfg.general.name}-{mode}', 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

from dataclasses import dataclass
from typing import Literal, cast, Union

import networkx as nx
import numpy as np
import pandas as pd
import pandas.api.types as pd_dts

PREPROC_TYPES = Literal['log', 'clip']

def preproc_do(
        df: pd.DataFrame,
        col_to_preproc: dict[str, PREPROC_TYPES],
        ref_df: Union[pd.DataFrame, None]=None) -> None:
    ref_df_ = df if ref_df is None else ref_df
    for col, preproc in col_to_preproc.items():
        if preproc == 'log':
            # replacing -inf with -1 instead of 0 allows generating values of 0
            df[col] = cast(pd.Series, np.log2(df[col] + 1e-7)).clip(lower=-1)
        elif preproc == 'clip':
            mean, std = ref_df_[col].mean(), ref_df_[col].std()
            df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)


def preproc_undo(df: pd.DataFrame, col_to_preproc: dict[str, PREPROC_TYPES]) -> None:
    for col, preproc in col_to_preproc.items():
        if preproc == 'log':
            df[col] = np.exp2(df[col].astype(float))
        elif preproc == 'clip':
            pass


def to_edgewise_df(table: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    e_a, e_b = zip(*graph.edges)

    return pd.concat(
        (
            table.loc[list(e_a)].add_prefix("A_").reset_index(drop=True),
            table.loc[list(e_b)].add_prefix("B_").reset_index(drop=True)
        ), axis=1
    )


def get_sd_metadata(df: pd.DataFrame) -> dict:
    column_data = {}

    for col, c_dtype in df.dtypes.items():
        if pd_dts.is_bool_dtype(c_dtype):
            column_data[col] = { "sdtype": "boolean" }
        elif pd_dts.is_categorical_dtype(c_dtype):
            column_data[col] = { "sdtype": "categorical" }
        elif pd_dts.is_numeric_dtype(c_dtype) or pd_dts.is_datetime64_dtype(c_dtype):
            column_data[col] = { "sdtype": "numerical" }

    return { "columns": column_data }


@dataclass
class TabTransfInfo:
    learning_dtypes: pd.Series
    s_min: pd.Series
    s_max: pd.Series
    col_to_preproc: dict[str, PREPROC_TYPES]
    bool_cols: list
    dt_cols: list
    cat_cols: list
    dtypes: pd.Series
