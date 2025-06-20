from abc import ABC, abstractmethod
from typing import Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import (coalesce, contains_self_loops,
                                   dense_to_sparse, is_undirected,
                                   to_dense_adj)


class GNFAE(pl.LightningModule):

    class EncoderGNFAE(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            half_in_channels = in_channels // 2
            self.relu = nn.ReLU(inplace=True)

            self.conv_to_half = nng.SAGEConv(in_channels, half_in_channels)
            self.conv_to_quarter = nng.SAGEConv(half_in_channels, in_channels // 4)

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            x_conv_to_half_relu = self.relu(self.conv_to_half(x, edge_index))
            return self.conv_to_quarter(x_conv_to_half_relu, edge_index)


    class EncoderGNFVAE(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            half_in_channels = in_channels // 2
            self.relu = nn.ReLU(inplace=True)

            self.conv = nng.SAGEConv(in_channels, half_in_channels)
            self.conv_mu = nng.SAGEConv(half_in_channels, in_channels // 4)
            self.conv_logstd = nng.SAGEConv(half_in_channels, in_channels // 4)

        def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
            x_conv_relu = self.relu(self.conv(x, edge_index))
            return (    self.conv_mu(x_conv_relu, edge_index),
                        self.conv_logstd(x_conv_relu, edge_index))


    class DecoderGNFAE(nn.Module):
        def __init__(self, in_channels: int, cat_indices: list[list[int]]):
            super().__init__()
            self.in_channels = in_channels
            half_in_channels = in_channels // 2
            self.cat_start_to_inds = {inds[0]: inds for inds in cat_indices}

            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)

            self.conv_to_half = nng.SAGEConv(in_channels // 4, half_in_channels)
            self.conv_to_full = nng.SAGEConv(half_in_channels, in_channels)

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            x_conv_to_half_relu = self.relu(self.conv_to_half(x, edge_index))
            x_conv_to_full = self.conv_to_full(x_conv_to_half_relu, edge_index)

            i = 0
            outs: list[torch.Tensor] = []

            while i < self.in_channels:
                if i in self.cat_start_to_inds:
                    inds = self.cat_start_to_inds[i]
                    outs.append(self.softmax(x_conv_to_full[:, inds]))
                    i = inds[-1]
                else:
                    outs.append(self.sigmoid(x_conv_to_full[:, [i]]))
                i += 1

            return torch.cat(outs, dim=-1)


    def __init__(self, in_channels: int, cat_indices: list[list[int]], variational=False, lr=1e-3, max_logstd=2):
        super().__init__()

        if variational:
            self._encoder = self.EncoderGNFVAE(in_channels)
        else:
            self._encoder = self.EncoderGNFAE(in_channels)

        self._decoder = self.DecoderGNFAE(in_channels, cat_indices)

        self.variational = variational
        self.lr = lr
        self.max_logstd = max_logstd

        self.mu: Tensor
        self.logstd: Tensor

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.variational:
            self.mu, self.logstd = self._encoder(x, edge_index)
            self.logstd = self.logstd.clamp(max=self.max_logstd)

            return self.reparametrize(self.mu, self.logstd)

        return self._encoder(x, edge_index)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        return self._decoder(z, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.decode(self.encode(x, edge_index), edge_index)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu: Optional[Tensor]=None,
                logstd: Optional[Tensor]=None) -> Tensor:
        mu = self.mu if mu is None else mu
        logstd = self.logstd if logstd is None else logstd.clamp(max=self.max_logstd)

        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def training_step(self, batch, batch_idx):
        x_hat = self(batch.x, batch.edge_index)
        loss = F.mse_loss(x_hat, batch.x)
        loss += not self.variational or (1 / batch.num_nodes) * self.kl_loss()

        return loss

    def validation_step(self, batch, batch_idx):
        x_hat = self(batch.x, batch.edge_index)
        loss = F.mse_loss(x_hat, batch.x)
        self.log('validation_loss', loss, batch_size=batch.batch.max().item(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x_hat = self(batch.x, batch.edge_index)
        loss = F.mse_loss(x_hat, batch.x)
        self.log('test_loss', loss, batch_size=batch.batch.max().item(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class NodePairMerge:
    def encode(self, data: Data) -> Data:
        x = cast(Tensor, data.x)
        edge_index = cast(Tensor, data.edge_index)
        num_nodes = cast(int, data.num_nodes)

        if not is_undirected(edge_index) or contains_self_loops(edge_index):
            raise ValueError("can only compress undirected graph without self-loops")

        edges = edge_index
        edge_node_dists = (x[edges[0]] - x[edges[1]]).abs().sum(-1)
        edge_node_dists_argsorted = edge_node_dists.argsort()
        node_to_pair = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=x.device, requires_grad=False)
        node_is_higher = node_to_pair == 0
        working_edges = edges[:, edge_node_dists_argsorted]
        selected_edges = working_edges.new_zeros((2, num_nodes // 2 + 1), requires_grad=False)
        num_pairs = 0

        # Find the edges around which to create pairs
        while working_edges.numel():
            edge = working_edges[:, 0]
            selected_edges[:, num_pairs] = edge
            working_edges = working_edges[
                :, torch.isin(working_edges, edge, invert=True).all(dim=0)]
            num_pairs += 1

        # Discard unused entries for selected edges & sort edge endpoints based on x
        selected_edges = selected_edges[:, :num_pairs]
        pair_numbers = torch.arange(
            0, num_pairs, dtype=torch.long, device=x.device, requires_grad=False)
        selected_edges_sorted_inds = x[selected_edges].sum(dim=-1).argsort(dim=0)
        selected_edges = selected_edges[
            selected_edges_sorted_inds, pair_numbers.unsqueeze(dim=0).expand(2, -1)]
        # Fill in bookkeeping information based on selected pair edges
        node_to_pair[selected_edges[0]] = node_to_pair[selected_edges[1]] = pair_numbers
        node_is_higher[selected_edges[1]] = True
        # Gather the single nodes
        single_node_mask = node_to_pair == -1
        num_single_nodes = cast(int, single_node_mask.sum().item())
        x_single_nodes = x[single_node_mask]
        # Sort single nodes based on their x
        x_single_nodes_argsorted = x_single_nodes.sum(dim=-1).argsort()
        x_single_nodes_sorted = x_single_nodes[x_single_nodes_argsorted]
        # Add encoded node features
        x_enc = torch.cat((
            torch.cat((x[selected_edges[0]], x[selected_edges[1]]), dim=1),
            torch.cat((x_single_nodes_sorted, torch.zeros_like(x_single_nodes)), dim=1)
        ), dim=0)
        # Assign numbering to single nodes
        single_node_offsets = torch.arange(
            num_single_nodes, dtype=torch.long, device=x.device, requires_grad=False)
        node_to_pair[single_node_mask] = num_pairs + single_node_offsets
        # Map edges endpoints to their pair number
        edges_as_pairs = node_to_pair[edges]
        # Get mask for the edges between different pairs
        intra_pair_asc_edges_mask = edges_as_pairs[0] != edges_as_pairs[1]
        intra_edges = edges[:, intra_pair_asc_edges_mask]
        intra_pairs = edges_as_pairs[:, intra_pair_asc_edges_mask]
        # Fill compressed edges and metadata
        edge_attr_expanded = (
            ~(node_is_higher[intra_edges[0]] | node_is_higher[intra_edges[1]]) +
            (~node_is_higher[intra_edges[0]] & node_is_higher[intra_edges[1]]) * 2 +
            (node_is_higher[intra_edges[0]] & ~node_is_higher[intra_edges[1]]) * 4 +
            (node_is_higher[intra_edges[0]] & node_is_higher[intra_edges[1]]) * 8
        ).char()
        edge_index_enc, edge_attr = coalesce(
            edge_index=intra_pairs, edge_attr=edge_attr_expanded, reduce="add")

        # Only keep asc edges and mirror results for desc ones
        edge_mask_asc = edge_index_enc[0] < edge_index_enc[1]
        edge_index_enc = edge_index_enc[:, edge_mask_asc]
        edge_index_enc = torch.cat((edge_index_enc, edge_index_enc.flip(0)), dim=1)
        edge_attr = cast(Tensor, edge_attr)[edge_mask_asc].repeat(2)

        # Retain any extra attributes from the original data
        cd_kwargs = data.to_dict()
        edge_attr = cast(Tensor, edge_attr)
        cd_kwargs.update(
            x=x_enc, edge_index=edge_index_enc,
            edge_attr=F.one_hot(edge_attr.long(), 16).float())

        return Data(**cd_kwargs)

    def decode(self, data: Data) -> Data:
        x_enc = data.x
        edge_index_enc = data.edge_index
        edge_attr = data.edge_attr.char()
        num_nodes_enc = cast(int, data.num_nodes)
        num_edges_enc = cast(int, data.num_edges)
        num_features_one_node = data.num_node_features // 2

        # Expand pairs
        x_dec_virtual_pairs = x_enc.view(-1, num_features_one_node)
        # Only work with asc edges and mirror results for desc ones later
        asc_edges_mask = edge_index_enc[0] < edge_index_enc[1]
        num_edges_enc_asc = num_edges_enc // 2
        edge_attr_asc = edge_attr[asc_edges_mask]
        edge_index_pair_head = edge_index_enc[:, asc_edges_mask] * 2
        # Expand edges to all possible options
        edge_attr_offset_possible = torch.tensor(
            [[False, False, True, True], [False, True, False, True]],
            device=x_enc.device, requires_grad=False
        ).repeat(1, num_edges_enc_asc)
        edge_index_possible = edge_index_pair_head.repeat_interleave(4, dim=1) + \
            edge_attr_offset_possible
        # Find mask of valid edges from all possible ones
        edge_attr_possible = edge_attr_asc.repeat_interleave(4, dim=0)
        bit_position_possible = torch.tensor(
            [1, 2, 4, 8],
            dtype=torch.int8, device=x_enc.device, requires_grad=False
        ).repeat(num_edges_enc_asc)
        edge_mask_possible = edge_attr_possible.bitwise_and(bit_position_possible).bool()
        # Get intra-pair nodes and put them together with inter-pair ones
        edge_index_inter = edge_index_possible[:, edge_mask_possible]
        edge_index_intra_asc = torch.arange(
            0, 2 * num_nodes_enc,
            dtype=torch.long, device=x_enc.device, requires_grad=False).view(-1, 2).T
        edge_index_dec_virtual_pairs = torch.cat(
            [
                edge_index_inter, edge_index_intra_asc,
                edge_index_inter.flip(0), edge_index_intra_asc.flip(0)
            ], dim=1)
        # Remove dummy sibblings of single nodes
        x_dec_pair_mask = (x_dec_virtual_pairs != 0).any(dim=-1)
        x_dec = x_dec_virtual_pairs[x_dec_pair_mask]
        edge_index_dec_no_singles = edge_index_dec_virtual_pairs[
            :, x_dec_pair_mask[edge_index_dec_virtual_pairs].all(dim=0)]
        # Reindex edges to account for discarded nodes
        x_dec_shift = (~x_dec_pair_mask).cumsum(dim=-1)
        edge_index_dec = edge_index_dec_no_singles - x_dec_shift[edge_index_dec_no_singles]

        d_kwargs = data.to_dict()
        d_kwargs.update(x=x_dec, edge_index=edge_index_dec)

        return Data(**d_kwargs)


class Compress(ABC):
    @abstractmethod
    def prepare(self, dm: pl.LightningDataModule) -> None:
        ...

    @abstractmethod
    def get_kind(self) -> str:
        ...

    @abstractmethod
    def compress(self, d: Data) -> Data:
        ...

    @abstractmethod
    def decompress(self, x: Tensor, e_adj: Tensor) -> tuple[Tensor, Tensor]:
        ...


class FeatsCompress(Compress):
    def __init__(
            self,
            in_channels: int,
            cat_indices: list[list[int]],
            device: str) -> None:
        super().__init__()
        self.feats_c = GNFAE(in_channels, cat_indices, variational=True)
        self.trainer = pl.Trainer(
            accelerator=device,
            logger=False,
            callbacks=[
                EarlyStopping(monitor="validation_loss", patience=100)
            ],
            max_epochs=5000,
            enable_checkpointing=False,
            enable_progress_bar=False
        )

    def prepare(self, dm: pl.LightningDataModule) -> None:
        self.trainer.fit(model=self.feats_c, datamodule=dm)
        self.trainer.test(model=self.feats_c, datamodule=dm)

    def get_kind(self) -> str:
        return "feats"

    def compress(self, d: Data) -> Data:
        d_dict = d.to_dict()
        self.feats_c.to(d.x.device)
        self.feats_c.eval()

        with torch.no_grad():
            d_dict['x'] = self.feats_c.encode(d.x, d.edge_index)

        return Data(**d_dict)

    def decompress(self, x: Tensor, e_adj: Tensor) -> tuple[Tensor, Tensor]:
        edge_index, edge_attr = dense_to_sparse(e_adj)
        self.feats_c.to(x.device)
        self.feats_c.eval()

        with torch.no_grad():
            x_c = self.feats_c.decode(x, edge_index)

        return x_c, e_adj


class StructCompress(Compress):
    def __init__(self) -> None:
        super().__init__()
        self.struct_c = NodePairMerge()

    def prepare(self, dm: pl.LightningDataModule) -> None:
        pass

    def get_kind(self) -> str:
        return "struct"

    def compress(self, d: Data) -> Data:
        return self.struct_c.encode(d)

    def decompress(self, x: Tensor, e_adj: Tensor) -> tuple[Tensor, Tensor]:
        edge_index, edge_attr = dense_to_sparse(e_adj)
        d = self.struct_c.decode(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return d.x, to_dense_adj(d.edge_index).squeeze(dim=0)


class FullCompress(FeatsCompress, StructCompress):
    def __init__(
            self,
            in_channels: int,
            cat_indices: list[list[int]],
            device: str) -> None:
        FeatsCompress.__init__(self, in_channels, cat_indices, device)
        StructCompress.__init__(self)

    def prepare(self, dm: pl.LightningDataModule) -> None:
        FeatsCompress.prepare(self, dm)
        StructCompress.prepare(self, dm)

    def get_kind(self) -> str:
        return "full"

    def compress(self, d: Data) -> Data:
        d_ = FeatsCompress.compress(self, d)
        return StructCompress.compress(self, d_)

    def decompress(self, x: Tensor, e_adj: Tensor) -> tuple[Tensor, Tensor]:
        edge_index, edge_attr = dense_to_sparse(e_adj)
        d = self.struct_c.decode(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        self.feats_c.to(x.device)
        self.feats_c.eval()

        with torch.no_grad():
            x_ = self.feats_c.decode(d.x, d.edge_index)

        return x_, to_dense_adj(d.edge_index).squeeze(dim=0)


# class Compression:
#     def __init__(self, mode: str) -> None:
#         if mode == 'struct':
#             self.compress = self._compress_struct
#             self.struct_c = PairMerge()
#             self.feats_c = None
#             self.vae = None
#         elif mode == "feats":
#             self.compress = self._compress_feats
#             self.struct_c = None
#             self.feats_c = GNFVAE()
#         elif mode == "all":
#             self.compress = self._compress_all
#             self.struct_c = PairMerge()
#             self.feats_c = GNFVAE()
#         else:
#             self.compress = self._compress_none

#     def init(self) -> None:
#         pass

#     def _compress_struct(self, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
#         return x, e

#     def _compress_feats(self, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
#         self.struct_c
#         return x, e

#     def _compress_all(self, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
#         return x, e

#     def _compress_none(self, x: Tensor, e: Tensor) -> tuple[Tensor, Tensor]:
#         return x, e
