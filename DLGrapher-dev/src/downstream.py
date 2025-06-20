import pickle
from typing import cast

import networkx as nx
import pandas as pd
import pandas.api.types as pd_dts
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv

import utils
from src.utils import TabTransfInfo


def _tgl_path_to_data(
        tgl_path_str: str,
        s_min: pd.Series, s_max: pd.Series,
        col_to_preproc: dict[str, utils.PREPROC_TYPES],
        df_ref: pd.DataFrame, tgt_col: str) -> list[Data]:
    data_list: list[Data] = []

    with open(tgl_path_str, "rb") as f:
         tgl: list[tuple[pd.DataFrame, nx.Graph]] = pickle.load(f)

    for t, g_nx in tgl:
        tgt_s = t[tgt_col].reindex(cast(list, g_nx.nodes))
        df = t.drop(columns=tgt_col)

        # Workaround for categoricals getting stored as objects
        if pd_dts.is_object_dtype(tgt_s):
            tgt_s = tgt_s.astype("category")

        if pd_dts.is_categorical_dtype(tgt_s):
            tgt_s = tgt_s.cat.codes

        df_no_bool = df.astype(
            {dt: 'uint8' for dt in df.select_dtypes(include='bool').columns},
            copy=False)
        df_oh = pd.get_dummies(df_no_bool)
        utils.preproc_do(df_oh, col_to_preproc, df_ref)
        df_norm = (df_oh - s_min) / (s_max - s_min)
        nx.set_node_attributes(g_nx, df_norm.to_dict(orient='index'))
        data = torch_geometric.utils.from_networkx(
                g_nx,
                group_node_attrs=all,
                group_edge_attrs=None)
        data.y = torch.tensor(tgt_s.to_numpy(dtype="float32"))
        data_list.append(data)

    return data_list


class NodeClassifier(pl.LightningModule):
    def __init__(
            self, in_channels: int, hidden_channels: int, middle_layers: int, lr=0.001) -> None:
        super().__init__()

        self.convs_non_final = ModuleList((
            SAGEConv(in_channels, hidden_channels),
            *(SAGEConv(hidden_channels, hidden_channels) for _ in range(middle_layers))
        ))

        self.conv_final = SAGEConv(hidden_channels, 1)

        self.lr = lr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        x_ = x

        for conv in self.convs_non_final:
            x_ = conv(x_, edge_index).relu()
            x_ = F.dropout(x_, p=0.5, training=self.training)

        x_ = self.conv_final(x_, edge_index)

        return x_.squeeze()

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(y_hat, batch.y)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index)

        acc = (y_hat.sigmoid().round() == batch.y).sum() / len(batch.y)
        self.log('test_acc', acc, batch_size=batch.batch.max().item(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def _run_node_classification(
        train_data: list[Data], test_data: list[Data], accelerator: str) -> None:
    model = NodeClassifier(
        train_data[0].num_node_features, train_data[0].num_node_features // 2, 1)
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=False,
        max_epochs=5_000,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    trainer.fit(model, DataLoader(train_data[:len(test_data)], batch_size=64, shuffle=True))
    trainer.test(model, DataLoader(test_data, batch_size=64, shuffle=False))


def node_classification(
        tgl_real_path_str: str, tgl_synth_path_str: str, tgl_test_path_str: str,
        tgl_ref_path_str: str, tti: TabTransfInfo, tgt_col: str, accelerator: str) -> None:

    # TODO reduce redundancy

    with open(tgl_ref_path_str, 'rb') as f:
        df, _ = pickle.load(f)

    df = df.drop(columns=tgt_col)
    df_no_bool = df.astype(
        {dt: 'uint8' for dt in df.select_dtypes(include='bool').columns},
        copy=False)
    df_oh = pd.get_dummies(df_no_bool)
    s_min = tti.s_min[df_oh.columns]
    s_max = tti.s_max[df_oh.columns]

    real_data = _tgl_path_to_data(
        tgl_real_path_str, s_min, s_max, tti.col_to_preproc, df_oh, tgt_col)
    synth_data = _tgl_path_to_data(
        tgl_synth_path_str, s_min, s_max, tti.col_to_preproc, df_oh, tgt_col)
    test_data = _tgl_path_to_data(
        tgl_test_path_str, s_min, s_max, tti.col_to_preproc, df_oh, tgt_col)

    print("Real data:")
    _run_node_classification(real_data, test_data, accelerator)
    print("Synth data:")
    _run_node_classification(synth_data, test_data, accelerator)
