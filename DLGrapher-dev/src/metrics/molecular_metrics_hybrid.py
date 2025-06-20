from metrics.molecular_metrics_discrete import BondMetricsCE
from metrics.molecular_metrics import AtomMetrics
import wandb
import torch.nn as nn


class TrainMolecularMetricsHybrid(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetrics(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred_epsX, masked_pred_E, true_epsX, true_E, log: bool):
        self.train_atom_metrics(masked_pred_epsX, true_epsX)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log['train/' + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = val.item()
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        return epoch_atom_metrics, epoch_bond_metrics
