import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **kwargs):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
            """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
            diff = preds - target
            sum_squared_error = torch.sum(diff * diff)
            n_obs = preds.shape[0]
            return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


# class ProbabilityMetric(Metric):
#     def __init__(self):
#         """ This metric is used to track the marginal predicted probability of a class during training. """
#         super().__init__()
#         self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

#     def update(self, preds: Tensor) -> None:
#         self.prob += preds.sum()
#         self.total += preds.numel()

#     def compute(self):
#         return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


class EdgeWeightedNodeDiff(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred_X: Tensor, pred_E: Tensor, true_X: Tensor, true_E: Tensor) -> None:
        pred_E_weights = 1. - pred_E.softmax(-1)[:, :, :, [0]]
        true_E_weights = 1. - true_E.softmax(-1)[:, :, :, [0]]
        pred_diffs = pred_E_weights * (pred_X.unsqueeze(2) - pred_X.unsqueeze(1)).abs()
        true_diffs = true_E_weights * (true_X.unsqueeze(1) - true_X.unsqueeze(1)).abs()
        self.total_value += F.mse_loss(pred_diffs, true_diffs, reduction="sum")
        self.total_samples += pred_E_weights.numel()

    def compute(self):
        return self.total_value / self.total_samples


class ZeroMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(1.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(1.), dist_reduce_fx="sum")

    def update(self, *args, **kwargs) -> None:
        pass

    def compute(self):
        return self.total_value / self.total_samples
