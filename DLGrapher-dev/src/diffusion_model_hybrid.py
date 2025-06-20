import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from typing import Any, Optional
import wandb
import os
import pickle

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseSchedule,\
    PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossHybrid
from metrics.abstract_metrics import SumExceptBatchMSE, SumExceptBatchMetric, SumExceptBatchKL, NLL
from src.analysis.table_graph_utils import TAB_GEN_TYPE
import utils
from compress import Compress


class HybridDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, compress: Optional[Compress]=None,
                 tab_gen: Optional[TAB_GEN_TYPE]=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.norm_values = cfg.model.normalize_factors
        self.norm_biases = cfg.model.norm_biases

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.hd_src = cfg.general.hd_src

        self.train_loss = TrainLossHybrid(cfg.model.lambda_train)

        self.metric_nll = NLL()
        self.val_X_mse = SumExceptBatchMSE()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_X_mse = SumExceptBatchMSE()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        self.gamma = PredefinedNoiseSchedule(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output, device=self.device) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output, device=self.device) / self.Edim_output
            y_limit = torch.ones(self.ydim_output, device=self.device) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            if self.hd_src == 'self':
                x_marginals = torch.ones(self.Xdim_output, device=self.device) / self.Xdim_output
            else:
                node_types = self.dataset_info.node_types.float()
                x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output, device=self.device) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = 0.
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.total_train_time = 0.
        self.test_sample_time = 0.
        self.val_epoch_time = 0.
        self.compress = compress
        self.tab_gen = tab_gen
        self.keepdim = self.hd_src == 'self' or compress

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)

        noisy_data = self.apply_noise(normalized_data.X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self(noisy_data, extra_data, node_mask)
        pred_X = (noisy_data['X_t'] - noisy_data['sigma_t'] * pred.X) / noisy_data['alpha_t']
        loss = self.train_loss(masked_pred_epsX=pred.X, masked_pred_E=pred.E, pred_y=pred.y, pred_X=pred_X,
                               true_epsX=noisy_data['epsX'], true_E=E, true_y=data.y, true_X = normalized_data.X,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_epsX=pred.X, masked_pred_E=pred.E, true_epsX=noisy_data['epsX'], true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg, 'fit')

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.val_epoch_time = 0.
        self.start_epoch_time = time.perf_counter()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        epoch_time = time.perf_counter() - self.start_epoch_time - \
            self.val_epoch_time * (self.current_epoch > 0)
        self.total_train_time += epoch_time
        self.print(f"Epoch {self.current_epoch}: X_MSE: {to_log['train_epoch/X_MSE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {epoch_time:.1f}s")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")

    def on_train_end(self) -> None:
        self.print(f"Total train time {(self.total_train_time / 3600):.3f} h")
        self.print(f"Mean train epoch time {(self.total_train_time / self.current_epoch):.2f} s")

    def on_validation_epoch_start(self) -> None:
        self.val_epoch_time = time.perf_counter()
        self.metric_nll.reset()
        self.train_loss.reset()
        self.val_X_mse.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)

        noisy_data = self.apply_noise(normalized_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self(noisy_data, extra_data, node_mask)
        pred_X = (noisy_data['X_t'] - noisy_data['sigma_t'] * pred.X) / noisy_data['alpha_t']
        self.train_loss(masked_pred_epsX=pred.X, masked_pred_E=pred.E, pred_y=pred.y, pred_X=pred_X,
                        true_epsX=noisy_data['epsX'], true_E=E, true_y=data.y, true_X = normalized_data.X,
                        log=False)
        nll = self.compute_eval_loss(pred, noisy_data, normalized_data.X, dense_data.E, data.y,  node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.metric_nll.compute(), self.val_X_mse.compute(), self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_mse": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Node MSE {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)
        self.log("val/epoch_loss", self.train_loss.log_epoch_metrics(is_train=False), sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.perf_counter()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics(
                [
                    (torch.from_numpy(x) if isinstance(x, np.ndarray) else x, torch.from_numpy(e))
                    for x, e in samples],
                self.name, self.current_epoch, val_counter=-1, test=False,
                local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.perf_counter() - start:.2f} seconds\n')
            print("Validation epoch end ends...")
        self.val_epoch_time = time.perf_counter() - self.val_epoch_time

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.metric_nll.reset()
        self.test_X_mse.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg, 'test')

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        normalized_data = utils.normalize(dense_data.X, dense_data.E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self(noisy_data, extra_data, node_mask)
        nll = self.compute_eval_loss(pred, noisy_data, normalized_data.X, dense_data.E, data.y, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.metric_nll.compute(), self.test_X_mse.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_mse": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Node MSE {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        self.test_sample_time = 0.

        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        self.print(f"Test generation time {self.test_sample_time:.2f} s")
        self.print("Saving the generated graphs")
        self.save_samples(samples)
        self.print("Generated graphs Saved. Computing sampling metrics...")
        self.sampling_metrics(
            [
                (torch.from_numpy(x) if isinstance(x, np.ndarray) else x, torch.from_numpy(e))
                for x, e in samples],
            self.name, self.current_epoch, self.val_counter, test=True,
            local_rank=self.local_rank)
        self.print("Done testing.")


    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        gamma_T = self.gamma(ones)
        alpha_T = diffusion_utils.alpha(gamma_T, X.size())
        mu_T_X = alpha_T * X
        sigma_T_X = diffusion_utils.sigma(gamma_T, mu_T_X.size())
        kl_distance_X = diffusion_utils.gaussian_KL(mu_T_X, sigma_T_X)

        return kl_distance_X + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        # pred_probs_X unused
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked
        # prob_pred.X unused
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * kl_e

    def reconstruction_logp(self, t, X, E, node_mask, y, gamma_0, data_0, test, epsilon=1e-10):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        # probX0 unused
        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0.abs(), probE=probE0, node_mask=node_mask)

        # X0 unused
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1, device=self.device).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probE0 = F.softmax(pred0.E, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(
            self.Edim_output, device=self.device).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1), device=self.device).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output, device=self.device).type_as(probE0)

        X_0, E_0, y_0 = data_0.values()
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=X_0.size())
        sigma_0_X = sigma_0 * self.norm_values[0]
        unnormalized_data = utils.unnormalize(X, E, y, self.norm_values, self.norm_biases, node_mask, collapse=False)
        unnormalized_0 = utils.unnormalize(X_0, E_0, y_0, self.norm_values, self.norm_biases, node_mask, collapse=False)
        X_0, E_0, _ = unnormalized_0.X, unnormalized_0.E, unnormalized_0.y
        X_0_centered = X_0 - 1
        log_pX_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((X_0_centered + 0.5) / sigma_0_X)
            - diffusion_utils.cdf_std_gaussian((X_0_centered - 0.5) / sigma_0_X)
            + epsilon)
        norm_cst_X = torch.logsumexp(log_pX_proportional, dim=-1, keepdim=True)
        log_probabilities_X = log_pX_proportional - norm_cst_X
        probX0 = log_probabilities_X * unnormalized_data.X

        if test:
            X_logp = self.test_X_logp
            E_logp = self.test_E_logp
        else:
            X_logp = self.val_X_logp
            E_logp = self.val_E_logp

        return - X_logp(-probX0) + E_logp(E * probE0.log())

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        # probX unused
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX.abs(), probE=probE, node_mask=node_mask)

        gamma_s = diffusion_utils.inflate_batch_array(self.gamma(s_float), X.size())    # (bs, 1, 1),
        gamma_t = diffusion_utils.inflate_batch_array(self.gamma(t_float), X.size())    # (bs, 1, 1)
        # Compute alpha_t and sigma_t from gamma, with correct size for X, E and z
        alpha_t = diffusion_utils.alpha(gamma_t, X.size())                        # (bs, 1, ..., 1), same n_dims than X
        sigma_t = diffusion_utils.sigma(gamma_t, X.size())
        eps = diffusion_utils.sample_feature_noise(X.size(), E.size(), y.size(), node_mask).type_as(X)

        X_t = alpha_t * X + sigma_t * eps.X

        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask,
                      's': s_float, 'gamma_t': gamma_t, 'gamma_s': gamma_s, 'epsX': eps.X,
                      'alpha_t': alpha_t, 'sigma_t': sigma_t}
        return noisy_data

    def compute_eval_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        delta_log_px = -self.Xdim_output * N * np.log(self.norm_values[0])
        kl_prior -= delta_log_px

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        gamma_s = noisy_data['gamma_s']     # gamma_s.size() == X.size()
        gamma_t = noisy_data['gamma_t']
        SNR_weight = - (1 - diffusion_utils.SNR(gamma_s - gamma_t))
        sqrt_SNR_weight = torch.sqrt(SNR_weight)            # same n_dims than X
        # Compute the error.
        weighted_predX_diffusion = sqrt_SNR_weight * pred.X
        X_t = noisy_data['X_t']
        E_t = noisy_data['E_t']
        y_t = noisy_data['y_t']
        epsX = noisy_data['epsX']
        weighted_X_t_diffusion = sqrt_SNR_weight * epsX
        if test:
            diffusion_error = self.test_X_mse(weighted_predX_diffusion, weighted_X_t_diffusion)
        else:
            diffusion_error = self.val_X_mse(weighted_predX_diffusion, weighted_X_t_diffusion)
        loss_all_t  += 0.5 * self.T * diffusion_error

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        s = noisy_data['s']
        t_zeros = torch.zeros_like(s)
        gamma_0 = diffusion_utils.inflate_batch_array(self.gamma(t_zeros), X_t.size())      # bs, 1, 1
        # Sample z_0 given X, E, y for timestep t, from q(z_t | X, E, y)
        eps0 = diffusion_utils.sample_feature_noise(X_t.size(), E_t.size(), y_t.size(), node_mask).type_as(X_t)
        alpha_0 = diffusion_utils.alpha(gamma_0, X_t.size())
        sigma_0 = diffusion_utils.sigma(gamma_0, X_t.size())
        X_0 = alpha_0 * X_t + sigma_0 * eps0.X
        E_0 = alpha_0.unsqueeze(1) * E_t + sigma_0.unsqueeze(1) * eps0.E
        y_0 = alpha_0.squeeze(1) * y_t + sigma_0.squeeze(1) * eps0.y
        loss_term_0 = self.reconstruction_logp(t, X, E, node_mask,
                                               y=y,
                                               gamma_0=gamma_0,
                                               data_0={'X_0': X_0, 'E_0': E_0, 'y_0': y_0},
                                               test=test)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = self.metric_nll(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"eval/kl prior": kl_prior.mean(),
                       "eval/Estimator loss terms": loss_all_t.mean(),
                       "eval/log_pn": log_pN.mean(),
                       "eval/loss_term_0": loss_term_0,
                       'eval/batch_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param number_chain_steps: number of timesteps to save for each chain
        :return: sample_list
        """
        batch_start_time = time.perf_counter()

        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

        z_T_cont = diffusion_utils.sample_feature_noise(
            X_size=(batch_size, n_max, self.Xdim_output),
            E_size=(batch_size, n_max, n_max, self.Edim_output),
            y_size=(batch_size, self.ydim_output),
            node_mask=node_mask)

        X, E, y = z_T_cont.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        if self.keepdim:
            chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1), X.size(2)))
        else:
            chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size, device=self.device)
        chain_E = torch.zeros(chain_E_size, device=self.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1), device=self.device).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T

            unnormalized = utils.unnormalize_no_mask(
                X=X[:keep_chain], E=E[:keep_chain], y=y[:keep_chain],
                norm_values=self.norm_values, norm_biases=self.norm_biases)

            if self.keepdim:
                unnormalized.mask_collapse_partial(node_mask[:keep_chain])
            else:
                unnormalized.mask(node_mask[:keep_chain], collapse=True)

            chain_X[write_index] = unnormalized.X
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        if self.keepdim:
            sampled_s.mask_collapse_partial(node_mask)
        else:
            sampled_s.mask(node_mask, collapse=True)

        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        self.test_sample_time += time.perf_counter() - batch_start_time

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            if self.keepdim:
                chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1, 1)], dim=0)
            else:
                chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        sample_list = []
        sample_list_comp = []

        for i in range(batch_size):
            n = n_nodes[i]
            x, e = X[i, :n], E[i, :n, :n]

            if self.compress:
                if self.hd_src != 'self':
                    x = torch.stack((x, torch.ones_like(x)), dim=-1)

                sample_list_comp.append((x.numpy(force=True), e.numpy(force=True)))
                x, e = self.compress.decompress(x, e)

                if self.hd_src != 'self':
                    x = x.squeeze()
                    n = x.size(0)

            node_data = self.tab_gen.sample(n.item()) if self.tab_gen else x.numpy(force=True)
            sample_list.append((node_data, e.numpy(force=True)))

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_samples = chain_X.size(1)       # number of samples
            for i in range(num_samples):
                n = n_nodes[i]
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/sample_{batch_id + i}')
                # if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=True)
                if self.keepdim:
                    chain_X_i = chain_X[:, i, :n, :]
                else:
                    chain_X_i = chain_X[:, i, :n]
                chain_E_i = chain_E[:, i, :n, :n]

                visualize_chain_kwargs: dict[str, Any] = {}

                if self.compress:
                    visualize_chain_kwargs['X_chain_latent'] = chain_X_i.numpy(force=True)
                    visualize_chain_kwargs['E_chain_coarse'] = chain_E_i.numpy(force=True)
                    x_list: tuple[torch.Tensor]
                    e_list: tuple[torch.Tensor]
                    x_list, e_list = zip(
                        *(self.compress.decompress(
                            x_ if self.hd_src else torch.stack((x_, torch.ones_like(x_)), dim=-1),
                            y_)
                        for x_, y_ in zip(chain_X_i, chain_E_i))
                    )
                    max_nr_nodes = max(x_.size(0) for x_ in x_list)
                    # Pad all elements with zero to have the same size
                    x_list_padded = [
                        F.pad(x_, (0, 0, 0, max_nr_nodes - x_.size(0))) for x_ in x_list
                    ]
                    e_list_padded = [
                        F.pad(e_, (0, max_nr_nodes - e_.size(1), 0, max_nr_nodes - e_.size(0)))
                        for e_ in e_list
                    ]
                    chain_X_i = torch.stack(x_list_padded, dim=0)
                    chain_E_i = torch.stack(e_list_padded, dim=0)

                if self.tab_gen:
                    node_data, _ = sample_list[i]
                    chain_X_i = [node_data] * len(chain_X_i)
                else:
                    chain_X_i = chain_X_i.numpy(force=True)

                visualize_chain_kwargs |= {
                    'path': result_path,
                    'nodes_list': chain_X_i, 'adjacency_matrix': chain_E_i.numpy(force=True)}
                _ = self.visualization_tools.visualize_chain(**visualize_chain_kwargs)

                self.print('\r{}/{} complete'.format(i+1, num_samples), end='', flush=True)
            self.print('\nVisualizing samples...')

            # Visualize the final samples
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            if sample_list_comp:
                self.visualization_tools.visualize(
                    result_path, sample_list, sample_list_comp, save_final)
            else:
                self.visualization_tools.visualize(result_path, sample_list, save_final)
            self.print("Done.")

        return sample_list

    # def sample_discrete_graph_given_z0(self, X_0, E_0, y_0, node_mask):
    #     """ Samples X, E, y ~ p(X, E, y|z0): once the diffusion is done, we need to map the result
    #     to categorical values.
    #     """
    #     zeros = torch.zeros(size=(X_0.size(0), 1), device=X_0.device)
    #     gamma_0 = self.gamma(zeros)
    #     # Computes sqrt(sigma_0^2 / alpha_0^2)
    #     sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
    #     noisy_data = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': torch.zeros(y_0.shape[0], 1).type_as(y_0)}
    #     extra_data = self.compute_extra_data(noisy_data)
    #     eps0 = self(noisy_data, extra_data, node_mask)

    #     # Compute mu for p(zs | zt).
    #     sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
    #     alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

    #     pred_X = 1. / alpha_0 * (X_0 - sigma_0 * eps0.X)
    #     pred_E = 1. / alpha_0.unsqueeze(1) * (E_0 - sigma_0.unsqueeze(1) * eps0.E)
    #     pred_y = 1. / alpha_0.squeeze(1) * (y_0 - sigma_0.squeeze(1) * eps0.y)
    #     assert (pred_E == torch.transpose(pred_E, 1, 2)).all()

    #     sampled = diffusion_utils.sample_normal(pred_X, pred_E, pred_y, sigma, node_mask).type_as(pred_X)
    #     assert (sampled.E == torch.transpose(sampled.E, 1, 2)).all()

    #     sampled = utils.unnormalize(sampled.X, sampled.E, sampled.y, self.norm_values,
    #                                 self.norm_biases, node_mask, collapse=True)
    #     return sampled

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()


        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, X_t.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=X_t.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=X_t.size())

        mu_X = X_t / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * pred.X

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        sampled_s = diffusion_utils.sample_hybrid_features(mu_X, sigma, prob_E, node_mask=node_mask)

        X_s = sampled_s.X
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0, device=self.device))
        out_discrete = utils.PlaceHolder(
            X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0, device=self.device))

        res = out_one_hot.mask(node_mask).type_as(y_t)
        res_collapsed = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        res.X = sampled_s.X
        res_collapsed.X = sampled_s.X

        return res, res_collapsed

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_domain_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_domain_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_domain_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_domain_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def save_samples(self, samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
        i = 1
        ext = "txt" if self.hd_src == 'none' else "pickle"
        filename = f"generated_test_samples_1.{ext}"

        while os.path.exists(filename):
            i += 1
            filename = f"generated_test_samples_{i}.{ext}"

        if self.hd_src == 'none':
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    xs = item[0].tolist()
                    f.write("X: \n")
                    for x in xs:
                        f.write(f"{x} ")
                    f.write("\n")
                    f.write("E: \n")
                    for es in item[1]:
                        for e in es:
                            f.write(f"{e} ")
                        f.write("\n")
                    f.write("\n")
        else:
            list_table_graph = [(
                self.dataset_info.restore_table(x),
                nx.from_numpy_array(e))
                for x, e in samples
            ]
            with open(filename, "wb") as f:
                pickle.dump(list_table_graph, f)
