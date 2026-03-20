"""Reach-Avoid VAE Model.

Extends VAEModel with:
- Disentanglement regularizer in the ELBO (VCI-inspired)
- Terminal/intermediate reconstruction loss weighting
- Trajectory collection and reach-avoid scoring during discrete evaluation
"""

import numpy as np
import torch
import pickle
import os
from scipy.stats import spearmanr

from src.models.vae_model import VAEModel
from src.utils.helper_functions import (
    compute_kl_divergence,
    generate_perturbed_sequences,
    write_csv,
)
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed
from src.reach_avoid.losses import compute_disentanglement_loss, compute_weighted_reg_loss
from src.reach_avoid.scoring import compute_reach_avoid_score


class ReachAvoidVAEModel(VAEModel):
    """VAEModel extended with reach-avoid scoring and VCI-inspired losses."""

    def __init__(self, config, dataset_collection):
        super().__init__(config, dataset_collection)
        self.lambda_disent = getattr(self.config.exp, 'lambda_disent', 0.0)
        self.lambda_terminal = getattr(self.config.exp, 'lambda_terminal', 1.0)
        self.lambda_intermediate = getattr(self.config.exp, 'lambda_intermediate', 0.0)
        self.ra_config = getattr(self.config.exp, 'reach_avoid', None)

    def calculate_elbo(self, H_t, Y_targets, a_seq, X_targets=None,
                       num_samples=10, optimize_a=False):
        """ELBO with optional disentanglement loss and weighted reconstruction."""
        H_rep = self.auxiliary_model.build_representations(H_t, only_last=True)

        Z_s_i, loss_hy = self.inference_model.init_hidden_history(H_t)

        if self.config.dataset.treatment_mode == 'multilabel':
            predict_action_loss = self.generative_model.loss_predict_actions_bern(Z_s_i, a_seq)
        else:
            predict_action_loss = self.generative_model.loss_predict_actions_beta(Z_s_i, a_seq)

        # Reset states
        self.inference_model.reset_states()
        self.generative_model.reset_states()

        # Pre-compute action encodings
        a_seq_hiddens_1 = self.generative_model.build_action_encoding(a_seq)
        a_seq_hiddens = self.generative_model.build_reverse_action_encoding(a_seq)

        kl_losses = []
        reg_losses = []
        action_losses = []
        disent_losses = []

        Y_last = Y_targets[:, -1, :]
        reg_loss_2 = None

        hidden_states = self.generative_model.build_hidden_states(Z_s_i, a_seq)
        reversed_hidden_states = self.generative_model.build_reverse_hidden_states(hidden_states)

        for s in range(self.tau):
            a_s = a_seq[:, s, :]
            a_s_hidden = a_seq_hiddens[:, s, :]

            Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)

            if s < self.tau - 1:
                a_s_next = a_seq[:, s+1, :]
                if self.config.dataset.treatment_mode == 'multilabel':
                    action_loss = self.generative_model.bern_loss(Z_s_i, a_s_next)
                else:
                    action_loss = self.generative_model.beta_loss(Z_s_i, a_s_next)
                action_losses.append(action_loss)

            # Inference network -- with optional disentanglement
            if self.lambda_disent > 0 and not optimize_a:
                disent_loss, q_mu, q_logvar = compute_disentanglement_loss(
                    self.inference_model, Z_s_inf, a_s_hidden,
                    a_seq_hiddens, s, H_rep, Y_last
                )
                disent_losses.append(disent_loss)
            else:
                q_mu, q_logvar = self.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)

            # Generative network
            p_mu, p_logvar = self.generative_model(Z_s_i, a_s_hidden)

            # Multiple sampling
            Z_samples = self.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)

            # Compute losses
            kl_loss = compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            kl_losses.append(kl_loss)

            if s > 0 and s < self.tau - 1:
                Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                a = a_seq_hiddens_1[:, -1, :].unsqueeze(0).expand(num_samples, -1, -1)
                reg_loss = self.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a)
                if self.predict_X:
                    X_current = X_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                    reg_loss += self.generative_model.decoding_X_loss(Z_samples, X_current)
                reg_losses.append(reg_loss)

            if s == self.tau - 1:
                Y_current = Y_targets[:, s, :]
                a = a_seq_hiddens_1[:, -1, :]
                reg_losses.append(self.generative_model.decoding_Y_loss_2(
                    Z_samples.mean(dim=0), Y_current, a))

            if optimize_a:
                if s == self.tau - 1:
                    Y_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                        H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
                    Y_preds = self.generative_model.decode(Z_samples)
                    Y_after_actions = torch.tensor(
                        Y_after_actions, device=Y_preds.device, dtype=Y_preds.dtype)
                    Y_preds = Y_preds.mean(dim=0)
                    reg_loss_2 = torch.nn.functional.mse_loss(Y_after_actions, Y_preds)

                    topk1 = self.print_topk_diff(Y_after_actions.flatten(), Y_preds.flatten())
                    topk2 = self.print_topk_diff(
                        Y_after_actions.flatten(), Y_current.mean(dim=0).flatten())
                    topk3 = self.print_topk_diff(
                        Y_preds.flatten(), Y_current.mean(dim=0).flatten())

                    reg_loss_3 = torch.nn.functional.mse_loss(
                        Y_current.mean(dim=0), Y_after_actions)
                    output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                        H_t, a_seq.detach(), self.dataset_collection.train_scaling_params)
                    ture_output = Y_current.mean(dim=0).cpu().numpy()
                    loss_1 = np.sqrt(((output_after_actions - ture_output) ** 2).mean()) * self.std
                    loss_2 = np.sqrt(((Y_preds.detach().cpu().numpy() - ture_output) ** 2).mean()) * self.std
                    loss_3 = np.sqrt(((output_after_actions - Y_preds.detach().cpu().numpy()) ** 2).mean()) * self.std

            # Update state
            Z_s_i = Z_samples.mean(dim=0)

        # Aggregate losses
        kl_loss = torch.stack(kl_losses).mean()
        if optimize_a:
            reg_loss = reg_losses[-1]
        else:
            reg_loss = compute_weighted_reg_loss(
                reg_losses, self.lambda_terminal, self.lambda_intermediate)

        action_loss = torch.stack(action_losses).mean() if action_losses else 0

        # Compute final ELBO
        if optimize_a:
            elbo = reg_loss * self.config.exp.lambda_reg + \
                   kl_loss * self.config.exp.lambda_kl + \
                   action_loss * self.config.exp.lambda_step + \
                   predict_action_loss * self.config.exp.lambda_action
        else:
            elbo = reg_loss * self.config.exp.lambda_reg + \
                   kl_loss * self.config.exp.lambda_kl + \
                   action_loss * self.config.exp.lambda_step + \
                   predict_action_loss * self.config.exp.lambda_action + \
                   loss_hy * self.config.exp.lambda_hy
            # Disentanglement regularizer
            if disent_losses and self.lambda_disent > 0:
                disent_loss = torch.stack(disent_losses).mean()
                elbo += disent_loss * self.lambda_disent

        return elbo, reg_loss, reg_loss_2

    def optimize_interventions_discrete_onetime(self, k=100):
        """Discrete intervention optimization with optional reach-avoid trajectory collection."""
        set_seed(self.config['exp']['seed'])
        if not self.config.exp.test:
            data = self.dataset_collection.val_f.data
        else:
            data = self.dataset_collection.test_f.data
        dataloader = get_dataloader(CIPDataset(data, self.config),
                                    batch_size=1, shuffle=False)
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")

        self.inference_model.to(device)
        self.generative_model.to(device)
        self.auxiliary_model.to(device)

        self.inference_model.eval()
        self.generative_model.eval()
        self.auxiliary_model.eval()

        # Reach-avoid configuration
        compute_ra = self.ra_config is not None
        if compute_ra:
            ra_target_upper = self.ra_config.target_upper
            ra_safety_volume_upper = self.ra_config.safety_volume_upper
            ra_safety_chemo_upper = getattr(self.ra_config, 'safety_chemo_upper', None)
            ra_kappa = getattr(self.ra_config, 'kappa', 10.0)

        all_ranks = []
        all_correlation = []
        case_infos = []

        for (i, batch) in enumerate(dataloader):
            if i > 99:
                break
            print(f"-" * 50 + f"Individual {i}" + "-" * 50)
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)
            Y_targets = targets['outputs']
            X_targets = None if not self.predict_X else targets['vitals']
            true_actions = targets['current_treatments']

            all_sequences = generate_perturbed_sequences(
                true_actions, k, self.tau, self.a_dim, device,
                treatment_mode=self.config.dataset.treatment_mode)

            elbos = []
            true_losses = []
            traj_stats_list = []
            with torch.no_grad():
                for seq in all_sequences:
                    # Compute ELBO
                    elbo, _, _ = self.calculate_elbo(
                        H_t, Y_targets, seq, X_targets=X_targets,
                        num_samples=self.config.exp.num_samples,
                        optimize_a=True)
                    elbos.append(elbo.item())

                    if compute_ra:
                        sim_result = self.dataset_collection.val_f.simulate_output_after_actions(
                            H_t, seq, self.dataset_collection.train_scaling_params,
                            return_trajectory=True)
                        output_after_actions = sim_result['scaled_output']
                        cv_traj = sim_result['cancer_volume_trajectory']
                        cd_traj = sim_result['chemo_dosage_trajectory']
                        traj_stats_list.append({
                            'cv_terminal': float(cv_traj[0, -1]),
                            'cv_max': float(cv_traj[0].max()),
                            'cv_min': float(cv_traj[0].min()),
                            'cd_max': float(cd_traj[0].max()) if cd_traj is not None else None,
                        })
                    else:
                        output_after_actions = self.dataset_collection.val_f.simulate_output_after_actions(
                            H_t, seq, self.dataset_collection.train_scaling_params)

                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((output_after_actions - true_output) ** 2).mean())
                    true_losses.append(true_loss)

            model_losses = np.array(elbos)
            true_losses = np.array(true_losses)

            corr_model_true, _ = spearmanr(model_losses, true_losses)

            case_info = {
                'individual_id': i,
                'model_losses': model_losses,
                'true_losses': true_losses,
                'correlations': {
                    'model_true': corr_model_true,
                },
                'true_sequence': true_actions.cpu().numpy(),
                'true_sequence_rank': np.sum(model_losses < model_losses[-1]) + 1,
                'all_sequences': all_sequences.cpu().numpy(),
            }

            # Add RA trajectory features if computed
            if compute_ra:
                traj_features = {
                    'cv_terminal': np.array([s['cv_terminal'] for s in traj_stats_list]),
                    'cv_max': np.array([s['cv_max'] for s in traj_stats_list]),
                    'cd_max': np.array([s['cd_max'] for s in traj_stats_list
                                        if s['cd_max'] is not None]),
                }
                case_info['traj_features'] = traj_features
                cv_t = traj_features['cv_terminal']
                cv_m = traj_features['cv_max']
                cd_m = traj_features['cd_max']
                if len(case_infos) == 0:
                    print(f"[RA DIAG] tau individual=0: "
                          f"cv_terminal=[{cv_t.min():.2f}, {cv_t.max():.2f}] "
                          f"(median={np.median(cv_t):.2f}), "
                          f"cv_max=[{cv_m.min():.2f}, {cv_m.max():.2f}], "
                          f"cd_max=[{cd_m.min():.2f}, {cd_m.max():.2f}]")

            case_infos.append(case_info)

            elbos = np.array(elbos)
            true_losses = np.array(true_losses)

            correlation, p_value = spearmanr(elbos, true_losses)
            all_correlation.append(correlation)

            true_seq_rank = np.sum(elbos < elbos[-1]) + 1
            all_ranks.append(true_seq_rank)

            print(f"Individual {i} - True sequence rank: {true_seq_rank} out of {k}")
            if compute_ra and 'traj_features' in case_info:
                cv_t = case_info['traj_features']['cv_terminal']
                print(f"  Trajectory features: cv_terminal=[{cv_t.min():.2f}, {cv_t.max():.2f}]")
            print(f"ELBO for true sequence: {elbos[-1]}")
            print(f"True Loss for true sequence: {true_losses[-1]}")
            print(f"Rank correlation for this individual: {correlation:.3f} (p-value: {p_value:.3f})")

        all_correlation = [c for c in all_correlation if not np.isnan(c)]
        avg_rank = sum(all_ranks) / len(all_ranks)
        if len(all_correlation) > 0:
            avg_correlation = sum(all_correlation) / len(all_correlation)
        else:
            avg_correlation = 0
        print(f"Average rank of true sequences across all individuals: {avg_rank:.2f} out of {k}")
        print(f"Average rank correlation across all individuals: {avg_correlation:.3f}")

        return case_infos
