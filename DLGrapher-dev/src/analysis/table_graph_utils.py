import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Union, cast

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sdmetrics.reports import SingleTableQualityReport
from sdv.single_table.base import BaseSingleTableSynthesizer

import utils
import wandb
from analysis.dist_helper import compute_mmd, gaussian_emd, gaussian_tv
from analysis.spectre_utils import SpectreSamplingMetrics
from datasets.abstract_dataset import (AbstractDataModule,
                                       AbstractTableGraphDataset,
                                       TableGraphDatasetInfos)
from tab_ddpm.tab_ddpm import TabDDPM

TAB_GEN_TYPE = Union[BaseSingleTableSynthesizer, TabDDPM]
TENS_OR_DF = Union[torch.Tensor, pd.DataFrame]

class TableGraphSamplingMetrics(nn.Module):
    def __init__(
            self, datamodule: AbstractDataModule,
            dataset_infos: TableGraphDatasetInfos):
        super().__init__()

        self.spectre_sampling_metrics = SpectreSamplingMetrics(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=['degree', 'clustering', 'orbit', 'spectre'])
        self.dataset_infos = dataset_infos
        self.tgt_col = datamodule.cfg.dataset.tgt_col
        self.tgt_val = datamodule.cfg.dataset.get('tgt_val', True)

        ds = cast(AbstractTableGraphDataset, datamodule.train_dataset)

        with open(ds.processed_paths[ds.file_idx['og_val']], 'rb') as f:
            self.val_table_graphs: list[tuple[pd.DataFrame, nx.Graph]] = pickle.load(f)
        with open(ds.processed_paths[ds.file_idx['og_test']], 'rb') as f:
            self.test_table_graphs: list[tuple[pd.DataFrame, nx.Graph]] = pickle.load(f)

        self.val_tab, self.val_edge_tab = table_graphs_to_df_pair(self.val_table_graphs)
        self.test_tab, self.test_edge_tab = table_graphs_to_df_pair(self.val_table_graphs)

        self.metadata = utils.get_sd_metadata(self.val_tab)
        self.edge_metadata = utils.get_sd_metadata(self.val_edge_tab)
        self.stqr = SingleTableQualityReport()

        with ThreadPoolExecutor() as executor:
            self.val_sample_ref = [
                tgt_col_hist
                for tgt_col_hist in executor.map(self.tgt_col_worker, self.val_table_graphs)]
            self.test_sample_ref = [
                tgt_col_hist
                for tgt_col_hist in executor.map(self.tgt_col_worker, self.test_table_graphs)]

    def forward(
            self, generated_graphs: list[tuple[TENS_OR_DF, torch.Tensor]],
            name, current_epoch, val_counter, local_rank, test=False):
        self.spectre_sampling_metrics(
            generated_graphs, name, current_epoch, val_counter, local_rank, test)

        dfs: list[pd.DataFrame] = []
        # edge_dfs: list[pd.DataFrame] = []
        gs: list[nx.Graph] = []

        for x_tens, e_tens in generated_graphs:
            t = self.dataset_infos.restore_table(x_tens)
            g = nx.from_numpy_array(e_tens.numpy())
            dfs.append(t)
            # edge_dfs.append(utils.to_edgewise_df(t, g))
            gs.append(g)

        df = pd.concat(dfs, ignore_index=True)
        # edge_df = pd.concat(edge_dfs, ignore_index=True)

        if test:
            self.stqr.generate(self.test_tab, df, self.metadata, verbose=False)
            self.save_report("tabular-only_test")
        else:
            self.stqr.generate(self.val_tab, df, self.metadata, verbose=False)

        tab_only_props = self.stqr.get_properties()

        if wandb.run:
            wandb.log({
                "sampling/tabular-only-quality_column-shapes": tab_only_props["Score"][0],
                "sampling/tabular-only-quality_column-pair-trends": tab_only_props["Score"][1],
                "sampling/tabular-only-quality_average": self.stqr.get_score()
            }, commit=False)
            wandb.run.summary['tabular-only-qual_average'] = self.stqr.get_score()

        if local_rank == 0:
            print(
                "=== Tabular-only stats ===",
                tab_only_props.to_string(index=False),
                f"Overall Score (Average): {cast(float, self.stqr.get_score()) * 100:.2f}%",
                sep="\n")

        # if test:
        #     self.stqr.generate(self.test_edge_tab, edge_df, self.edge_metadata, verbose=False)
        #     self.save_report("tabular-edge_test")
        # else:
        #     self.stqr.generate(self.val_edge_tab, edge_df, self.edge_metadata, verbose=False)

        # tab_edge_props = self.stqr.get_properties()

        # if wandb.run:
        #     wandb.log({
        #         "sampling/tabular-edge-quality_column-shapes": tab_edge_props["Score"][0],
        #         "sampling/tabular-edge-quality_column-pair-trends": tab_edge_props["Score"][1],
        #         "sampling/tabular-edge-quality_average": self.stqr.get_score()
        #     }, commit=False)
        #     wandb.run.summary['tabular-edge-qual_average'] = self.stqr.get_score()

        # if local_rank == 0:
        #     print(
        #         "=== Tabular-edge stats ===",
        #         tab_edge_props.to_string(index=False),
        #         f"Overall Score (Average): {cast(float, self.stqr.get_score()) * 100:.2f}%",
        #         "=====",
        #         sep="\n")

        tgt_col_mmd = self.tgt_col_stats(
            self.test_sample_ref if test else self.val_sample_ref, list(zip(dfs, gs)))

        if wandb.run:
            wandb.log({'mmd/tgt_col': tgt_col_mmd}, commit=False)
            wandb.run.summary['tgt_col'] = tgt_col_mmd

        if local_rank == 0:
            print("Target Column MMD: ", tgt_col_mmd)

    def reset(self) -> None:
        self.spectre_sampling_metrics.reset()

    def save_report(self, prefix: str) -> None:
        for pn in ("Column Shapes", "Column Pair Trends"):
            fig = self.stqr.get_visualization(property_name=pn)
            i = 1
            filename = f"{prefix}_{pn}_{i}.png"

            while os.path.exists(filename):
                i += 1
                filename = f"{prefix}_{pn}_{i}.png"

            fig.write_image(filename)

    def tgt_col_worker(self, tg: tuple[pd.DataFrame, nx.Graph]) -> np.ndarray:
        table, graph = tg
        return np.bincount([
            (table.loc[graph.neighbors(n)][self.tgt_col] == self.tgt_val).sum()
            for n in graph.nodes])

    def tgt_col_stats(
            self,
            sample_ref: list[np.ndarray],
            tg_pred_list: list[tuple[pd.DataFrame, nx.Graph]],
            is_parallel=True, compute_emd=False) -> float:

        # in case an empty graph is generated
        tg_pred_list_remove_empty = [(t, g) for t, g in tg_pred_list if g.number_of_nodes() > 0]

        if is_parallel:
            with ThreadPoolExecutor() as executor:
                sample_pred = [
                    tgt_col_hist
                    for tgt_col_hist
                    in executor.map(self.tgt_col_worker, tg_pred_list_remove_empty)]
        else:
            sample_pred = [self.tgt_col_worker(tg_pred) for tg_pred in tg_pred_list_remove_empty]

        if compute_emd:
            mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
        else:
            mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

        return mmd_dist


def table_graph_path_to_df(path_dfs: str) -> pd.DataFrame:
    with open(path_dfs, "rb") as f:
        table_graph_list: list[tuple[pd.DataFrame, nx.Graph]] = pickle.load(f)

    return pd.concat([table for table, _ in table_graph_list], ignore_index=True)


def table_graphs_to_df_pair(tgs: list[tuple[pd.DataFrame, nx.Graph]]) -> \
    tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([t for t, _ in tgs], ignore_index=True)
    edge_df = pd.concat([utils.to_edgewise_df(t, g) for t, g in tgs], ignore_index=True)

    return df, edge_df
