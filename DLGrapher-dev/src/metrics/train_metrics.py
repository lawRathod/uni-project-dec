import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError
import time
from typing import Dict, Literal, overload
import wandb
from src.metrics.abstract_metrics import (CrossEntropyMetric,
                                          EdgeWeightedNodeDiff, ZeroMetric)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = MeanSquaredError()
        self.train_edge_mse = MeanSquaredError()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def calc_loss(
            self, loss_X: torch.Tensor, loss_E: torch.Tensor,
            loss_y: torch.Tensor) -> torch.Tensor:
        return loss_X + self.lambda_train[0] * loss_E + \
            self.lambda_train[1] * loss_y

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return self.calc_loss(loss_X, loss_E, loss_y)

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    @overload
    def log_epoch_metrics(self, is_train: Literal[True]=...) -> Dict: ...

    @overload
    def log_epoch_metrics(self, is_train: Literal[False]=...) -> torch.Tensor: ...

    def log_epoch_metrics(self, is_train=True):

        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        log_prefix = "train" if is_train else "val"
        to_log = {f"{log_prefix}_epoch/x_CE": epoch_node_loss,
                  f"{log_prefix}_epoch/E_CE": epoch_edge_loss,
                  f"{log_prefix}_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        if is_train:
            return to_log

        return self.calc_loss(epoch_node_loss, epoch_edge_loss, epoch_y_loss)


class TrainLossHybrid(nn.Module):
    def __init__(self, lambda_train: list[int], has_attrs=True):
        super().__init__()
        self.node_loss = MeanSquaredError() if has_attrs else ZeroMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train_E, self.lambda_train_y, *lambda_ewnd = lambda_train
        if lambda_ewnd:
            self.ewnd_loss = EdgeWeightedNodeDiff()
            self.lambda_train_ewnd = lambda_ewnd[0]
        else:
            self.ewnd_loss = ZeroMetric()
            self.lambda_train_ewnd = 0
        self.has_attrs = has_attrs

    def calc_loss(
            self, loss_X: torch.Tensor, loss_E: torch.Tensor,
            loss_y: torch.Tensor, loss_ewnd: torch.Tensor) -> torch.Tensor:
        return loss_X + self.lambda_train_E * loss_E + \
            self.lambda_train_y * loss_y + self.lambda_train_ewnd * loss_ewnd

    def forward(
            self, masked_pred_epsX: torch.Tensor, masked_pred_E: torch.Tensor, pred_y: torch.Tensor,
            pred_X: torch.Tensor,
            true_epsX: torch.Tensor, true_E: torch.Tensor, true_y: torch.Tensor,
            true_X: torch.Tensor,
            log: bool):
        """ Compute train metrics
        masked_pred_epsX : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        pred_X : tensor -- (bs, n, dx)
        true_epsX : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        log : boolean. """
        true_E_dense = true_E
        masked_pred_E_dense = masked_pred_E
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else true_epsX.sum()

        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else true_E.sum()
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else true_y.sum()
        loss_ewnd = self.ewnd_loss(pred_X, masked_pred_E_dense, true_X, true_E_dense)

        if log:
            to_log = {"train_loss/batch_loss": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_MSE": self.node_loss.compute() if true_epsX.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
                      "train_loss/EWND": self.ewnd_loss.compute()
                      }
            if wandb.run:
                wandb.log(to_log, commit=True)
        return self.calc_loss(loss_X, loss_E, loss_y, loss_ewnd)

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    @overload
    def log_epoch_metrics(self, is_train:Literal[True]=...) -> Dict: ...

    @overload
    def log_epoch_metrics(self, is_train:Literal[False]=...) -> torch.Tensor: ...

    def log_epoch_metrics(self, is_train=True):
        log_prefix = "train" if is_train else "val"

        epoch_node_loss = self.node_loss.compute() if self.node_loss.total > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1
        epoch_ewnd_loss = self.ewnd_loss.compute() if self.ewnd_loss.total_samples > 0 else -1

        to_log = {f"{log_prefix}_epoch/X_MSE": epoch_node_loss,
                  f"{log_prefix}_epoch/E_CE": epoch_edge_loss,
                  f"{log_prefix}_epoch/y_CE": epoch_y_loss,
                #   f"{log_prefix}_loss/EWND": epoch_ewnd_loss
                  }
        if wandb.run:
            wandb.log(to_log, commit=False)

        if is_train:
            return to_log

        return self.calc_loss(epoch_node_loss, epoch_edge_loss, epoch_y_loss, epoch_ewnd_loss)
