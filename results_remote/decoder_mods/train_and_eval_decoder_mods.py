#!/usr/bin/env python3
"""
Train and evaluate three decoder modifications for model-predicted RA filtering on Cancer.

Modifications:
  1. Heteroscedastic: Train with proper NLL loss (learn logvar) instead of MSE with fixed variance
  2. Wider decoder: Increase decoder hidden dims [16] → [64, 32] for more capacity
  3. MC-Dropout: Enable dropout (p=0.2) in decoder; at test time, run T forward passes for
     predictive mean/variance

The goal is to recover model-predicted RA discrimination that the vanilla decoder fails at
(model-predicted feasibility ≈ 0.1% vs oracle ≈ 70%).

Usage (on Vast.ai):
    source /root/vcip_env/bin/activate
    cd /root/VCIP
    python3 /root/train_and_eval_decoder_mods.py --variant heteroscedastic --gamma 4 --all_seeds
    python3 /root/train_and_eval_decoder_mods.py --variant wider --gamma 4 --all_seeds
    python3 /root/train_and_eval_decoder_mods.py --variant mcdropout --gamma 4 --all_seeds
    python3 /root/train_and_eval_decoder_mods.py --variant vanilla --gamma 4 --all_seeds  # baseline
"""

import os
import sys
import copy
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.stats import spearmanr

sys.path.insert(0, '/root/VCIP')
sys.path.insert(0, '/root/VCIP/src')

from src.data.cancer_sim_cont.dataset import SyntheticCancerDatasetCollectionCont
from src.models.vae_model import VAEModel
from src.models.generative_model import GenerativeModel
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed, to_float, repeat_static
from src.utils.helper_functions import generate_perturbed_sequences, compute_kl_divergence
from omegaconf import OmegaConf

# RA thresholds (moderate, from Phase 1B calibration)
TARGET_UPPER = 0.6
SAFETY_VOL_UPPER = 5.0
SAFETY_CHEMO_UPPER = 8.5

K = 100
NUM_SAMPLES = 10
NUM_PATIENTS = 100
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 4, 6, 8]


def build_config(seed, gamma):
    """Build OmegaConf config matching VCIP Cancer training."""
    config = OmegaConf.create({
        'exp': {
            'seed': seed,
            'global_seed': 22,
            'max_epochs': 100,
            'mode': 'train',
            'test': False,
            'sam': True,
            'num_samples': NUM_SAMPLES,
            'alpha': 0.001,
            'beta': 0.99,
            'batch_size': 512,
            'batch_size_val': 128,
            'dropout': 0.2,
            'entropy_reg': 1,
            'remove_aux': False,
            'beta_bound': -10,
            'lambda_reg': 1,
            'lambda_kl': 1,
            'lambda_step': 0.1,
            'lambda_action': 1,
            'lambda_hy': 1,
            'lambda_X': 0.1,
            'lambda_Y': 1,
            'tau': 6,
            'repeats': 5,
            'rank': True,
            'unscale_rmse': True,
            'percentage_rmse': True,
            'processed_data_dir': '/root/VCIP/data/processed/cancer_sim_cont',
            'log_dir': f'/root/VCIP/my_outputs/cancer_sim_cont/22/coeff_{gamma}/VCIP/train/True',
            'device': 'cuda',
            'gpus': 1,
            'weights_ema': True,
            'clear_tf': True,
            'logging': False,
            'load_data': True,
            'load_best': True,
            'val': True,
            'lr': 0.01,
            'weight_decay': 1e-5,
            'epochs': 100,
            'patience': 20,
        },
        'dataset': {
            '_target_': 'src.data.SyntheticCancerDatasetCollectionCont',
            'name': 'cancer_sim_cont',
            'gamma': gamma,
            'coeff': gamma,
            'projection_horizon': 5,
            'lag': 0,
            'window_size': 15,
            'max_seq_length': 60,
            'cf_seq_mode': 'sliding_treatment',
            'treatment_mode': 'continuous',
            'num_patients': {'train': 1000, 'val': 100, 'test': 100},
            'static_size': 1,
            'treatment_size': 2,
            'one_hot_treatment_size': 4,
            'input_size': 0,
            'output_size': 1,
            'predict_X': False,
            'autoregressive': True,
            'val_batch_size': 4096,
            'seed': seed,
        },
        'model': {
            '_target_': 'src.models.vae_model.VAEModel',
            'name': 'VCIP',
            'z_dim': 16,
            'dim_treatments': 2,
            'dim_vitals': 0,
            'dim_static_features': 1,
            'dim_outcomes': 1,
            'lr': 0.001,
            'auxiliary': {
                '_target_': 'src.models.auxiliary_model.AuxiliaryModel',
                'hidden_dim': 16, 'num_layers': 2,
                'hiddens_G_y': [16, 16], 'hiddens_G_x': [16, 16], 'lr': 0.001,
            },
            'inference': {
                '_target_': 'src.models.inference_model.InferenceModel',
                'hidden_dim': 16, 'num_layers': 2,
                'hiddens_F_mu': [16], 'hiddens_F_logvar': [16],
                'predict_y_history': [16], 'do': True,
            },
            'generative': {
                '_target_': 'src.models.generative_model.GenerativeModel',
                'entropy_lambda': 1, 'hidden_dim': 16, 'treatment_hidden_dim': 16,
                'num_layers': 2, 'hiddens_F_mu': [16], 'hiddens_F_logvar': [16],
                'hiddens_decoder': [16], 'hidden_action_decoder': [16],
                'hidden_action_decoder_step': [16],
            },
        }
    })
    return config


# ---------------------------------------------------------------------------
# Decoder modification 1: Heteroscedastic (NLL loss instead of MSE)
# ---------------------------------------------------------------------------
def patch_heteroscedastic(model):
    """Monkey-patch the generative model to use proper NLL loss with learned logvar.

    The vanilla VCIP `decoding_Y_loss_2` uses MSE and discards the learned logvar
    (replaces p_std with constant 0.1). This patch uses the actual learned variance
    from the decoder output, training it end-to-end via NLL.
    """
    gen = model.generative_model

    def decoding_Y_loss_2_nll(self, Z, y, a):
        input = torch.cat((Z, a), dim=-1)
        out = self.decoder_pa(input)
        mu, p_logvar = out[..., :self.output_dim], out[..., self.output_dim:]
        # Clamp logvar for stability
        p_logvar = torch.clamp(p_logvar, min=-6, max=4)
        p_std = torch.exp(0.5 * p_logvar) + 1e-6
        normal_dist = Normal(mu, p_std)
        nll = -normal_dist.log_prob(y).sum(dim=-1).mean()
        return nll

    import types
    gen.decoding_Y_loss_2 = types.MethodType(decoding_Y_loss_2_nll, gen)
    print("[PATCH] Heteroscedastic: decoding_Y_loss_2 → NLL with learned logvar")


# ---------------------------------------------------------------------------
# Decoder modification 2: Wider decoder (more capacity)
# ---------------------------------------------------------------------------
def patch_wider_decoder(model):
    """Replace decoder_pa with a wider network: [64, 32] hidden dims instead of [16].

    The wider decoder has more capacity to learn the cancer volume dynamics,
    potentially reducing the prediction variance that causes near-zero feasibility.
    """
    gen = model.generative_model
    device = next(gen.parameters()).device

    input_size = gen.z_dim + gen.treatment_hidden_dim
    output_size = gen.output_dim * 2

    # Build wider decoder_pa: input → 64 → ELU → 32 → ELU → output_dim*2
    new_decoder_pa = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ELU(),
        nn.Linear(64, 32),
        nn.ELU(),
        nn.Linear(32, output_size),
    ).to(device)

    gen.decoder_pa = new_decoder_pa

    # Also widen decoder_p for consistency
    input_size_p = gen.z_dim
    new_decoder_p = nn.Sequential(
        nn.Linear(input_size_p, 64),
        nn.ELU(),
        nn.Linear(64, 32),
        nn.ELU(),
        nn.Linear(32, output_size),
    ).to(device)
    gen.decoder_p = new_decoder_p

    print(f"[PATCH] Wider decoder: decoder_pa {input_size}→64→32→{output_size}, "
          f"decoder_p {input_size_p}→64→32→{output_size}")


# ---------------------------------------------------------------------------
# Decoder modification 3: MC-Dropout
# ---------------------------------------------------------------------------
def patch_mcdropout(model, p=0.2):
    """Add dropout layers to decoder_pa for MC-dropout at test time.

    During evaluation, we run T forward passes with dropout enabled,
    then use the mean prediction and empirical variance for RA filtering.
    """
    gen = model.generative_model
    device = next(gen.parameters()).device

    input_size = gen.z_dim + gen.treatment_hidden_dim
    output_size = gen.output_dim * 2

    # Rebuild decoder_pa with dropout after each hidden layer
    hiddens = gen.hiddens_decoder
    layers = []
    prev_dim = input_size
    for h in hiddens:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ELU())
        layers.append(nn.Dropout(p=p))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_size))

    gen.decoder_pa = nn.Sequential(*layers).to(device)

    # Same for decoder_p
    layers_p = []
    prev_dim = gen.z_dim
    for h in hiddens:
        layers_p.append(nn.Linear(prev_dim, h))
        layers_p.append(nn.ELU())
        layers_p.append(nn.Dropout(p=p))
        prev_dim = h
    layers_p.append(nn.Linear(prev_dim, output_size))

    gen.decoder_p = nn.Sequential(*layers_p).to(device)

    print(f"[PATCH] MC-Dropout (p={p}): dropout added to decoder_pa and decoder_p")


# ---------------------------------------------------------------------------
# Training loop (fine-tune decoder only, freeze everything else)
# ---------------------------------------------------------------------------
def fine_tune_decoder(model, dataset_collection, cfg, num_epochs=50, lr=1e-3, variant='vanilla'):
    """Fine-tune only the decoder parameters while keeping the rest frozen.

    This is much faster than full retraining and targets the specific weakness:
    the decoder's inability to produce calibrated predictions for RA filtering.
    """
    device = next(model.parameters()).device

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder parameters
    decoder_params = []
    for name, param in model.generative_model.named_parameters():
        if 'decoder_p' in name or 'decoder_pa' in name:
            param.requires_grad = True
            decoder_params.append(param)
            print(f"  Unfrozen: generative_model.{name} ({param.numel()} params)")

    if not decoder_params:
        print("WARNING: No decoder parameters to fine-tune!")
        return

    total_params = sum(p.numel() for p in decoder_params)
    print(f"  Total trainable params: {total_params}")

    optimizer = torch.optim.Adam(decoder_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Get training dataloader
    train_data = dataset_collection.train_f.data
    train_dataset = CIPDataset(train_data, cfg)
    train_loader = get_dataloader(train_dataset, batch_size=cfg.exp.batch_size, shuffle=True)

    val_data = dataset_collection.val_f.data
    val_dataset = CIPDataset(val_data, cfg)
    val_loader = get_dataloader(val_dataset, batch_size=cfg.exp.batch_size_val, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            Y_targets = targets['outputs']
            a_seq = targets['current_treatments']
            X_targets = None

            optimizer.zero_grad()
            elbo, reg_loss, _ = model.calculate_elbo(
                H_t, Y_targets, a_seq, X_targets=X_targets,
                num_samples=cfg.exp.num_samples, optimize_a=False)
            elbo.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(elbo.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                H_t, targets = batch
                for key in H_t:
                    H_t[key] = H_t[key].to(device)
                for key in targets:
                    targets[key] = targets[key].to(device)

                Y_targets = targets['outputs']
                a_seq = targets['current_treatments']

                elbo, reg_loss, _ = model.calculate_elbo(
                    H_t, Y_targets, a_seq, X_targets=None,
                    num_samples=cfg.exp.num_samples, optimize_a=False)
                val_losses.append(elbo.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                         if 'decoder_p' in k or 'decoder_pa' in k}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"best_val={best_val_loss:.6f} patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best decoder weights
    if best_state is not None:
        current_state = model.state_dict()
        current_state.update(best_state)
        model.load_state_dict(current_state)
        print(f"  Restored best decoder (val_loss={best_val_loss:.6f})")


# ---------------------------------------------------------------------------
# Model-predicted trajectory extraction (with MC-dropout support)
# ---------------------------------------------------------------------------
def extract_predicted_cv(model, H_t, Y_targets, a_seq, scaling_params,
                         num_samples=10, mc_passes=1):
    """Extract model-predicted cancer volume trajectory.

    For MC-dropout (mc_passes > 1), runs multiple forward passes with dropout
    enabled and returns mean + std of predictions.

    Returns:
        elbo: float
        cv_trajectory_unscaled: np.array (tau,) — mean predicted CV (unscaled)
        cv_trajectory_std: np.array (tau,) — std across MC passes (0 if mc_passes=1)
    """
    cv_mean = float(scaling_params[0]['cancer_volume'])
    cv_std = float(scaling_params[1]['cancer_volume'])

    all_cv_trajectories = []
    elbo_val = None

    for t in range(mc_passes):
        H_rep = model.auxiliary_model.build_representations(H_t, only_last=True)
        Z_s_i, loss_hy = model.inference_model.init_hidden_history(H_t)

        if model.config.dataset.treatment_mode == 'multilabel':
            predict_action_loss = model.generative_model.loss_predict_actions_bern(Z_s_i, a_seq)
        else:
            predict_action_loss = model.generative_model.loss_predict_actions_beta(Z_s_i, a_seq)

        model.inference_model.reset_states()
        model.generative_model.reset_states()

        a_seq_hiddens_1 = model.generative_model.build_action_encoding(a_seq)
        a_seq_hiddens = model.generative_model.build_reverse_action_encoding(a_seq)

        kl_losses = []
        reg_losses = []
        action_losses = []
        predicted_cv = []

        Y_last = Y_targets[:, -1, :]
        tau = model.tau

        hidden_states = model.generative_model.build_hidden_states(Z_s_i, a_seq)

        for s in range(tau):
            a_s_hidden = a_seq_hiddens[:, s, :]
            Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)

            if s < tau - 1:
                a_s_next = a_seq[:, s + 1, :]
                if model.config.dataset.treatment_mode == 'multilabel':
                    action_loss = model.generative_model.bern_loss(Z_s_i, a_s_next)
                else:
                    action_loss = model.generative_model.beta_loss(Z_s_i, a_s_next)
                action_losses.append(action_loss)

            q_mu, q_logvar = model.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)
            p_mu, p_logvar = model.generative_model(Z_s_i, a_s_hidden)
            Z_samples = model.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)
            kl_loss = compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            kl_losses.append(kl_loss)

            # Decode predicted outcome
            a_decode = a_seq_hiddens_1[:, -1, :]
            Z_mean = Z_samples.mean(dim=0) if Z_samples.dim() == 3 else Z_samples
            y_mu, y_logvar = model.generative_model.decode_p_a(Z_mean, a_decode)

            # Unscale: predicted_cv_unscaled = mu * cv_std + cv_mean
            cv_scaled = float(y_mu.detach().cpu().flatten()[0])
            cv_unscaled = cv_scaled * cv_std + cv_mean
            predicted_cv.append(cv_unscaled)

            if s > 0 and s < tau - 1:
                Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
                a = a_seq_hiddens_1[:, -1, :].unsqueeze(0).expand(num_samples, -1, -1)
                reg_losses.append(model.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a))

            if s == tau - 1:
                Y_current = Y_targets[:, s, :]
                a = a_seq_hiddens_1[:, -1, :]
                reg_losses.append(model.generative_model.decoding_Y_loss_2(Z_mean, Y_current, a))

            Z_s_i = Z_samples.mean(dim=0) if Z_samples.dim() == 3 else Z_samples

        all_cv_trajectories.append(np.array(predicted_cv))

        if elbo_val is None:
            kl_loss = torch.stack(kl_losses).mean()
            reg_loss = reg_losses[-1] if reg_losses else torch.tensor(0.0)
            action_loss = torch.stack(action_losses).mean() if action_losses else torch.tensor(0.0)
            elbo = (reg_loss * model.config.exp.lambda_reg +
                    kl_loss * model.config.exp.lambda_kl +
                    action_loss * model.config.exp.lambda_step +
                    predict_action_loss * model.config.exp.lambda_action +
                    loss_hy * model.config.exp.lambda_hy)
            elbo_val = elbo.item()

    all_cv = np.array(all_cv_trajectories)  # (mc_passes, tau)
    cv_mean_traj = all_cv.mean(axis=0)
    cv_std_traj = all_cv.std(axis=0) if mc_passes > 1 else np.zeros_like(cv_mean_traj)

    return elbo_val, cv_mean_traj, cv_std_traj


# ---------------------------------------------------------------------------
# RA filtering and constrained selection (same as A3)
# ---------------------------------------------------------------------------
def apply_ra_filter(cv_terminal, cv_max, cd_max):
    feasible = (cv_terminal <= TARGET_UPPER) & (cv_max <= SAFETY_VOL_UPPER)
    if cd_max is not None:
        feasible = feasible & (cd_max <= SAFETY_CHEMO_UPPER)
    return feasible


def ra_constrained_selection(elbos, feasible, true_losses):
    k = len(elbos)
    feas_rate = feasible.sum() / k

    elbo_best = np.argmin(elbos)
    oracle_best = np.argmin(true_losses)
    elbo_top1 = 1.0 if elbo_best == oracle_best else 0.0

    if feasible.sum() == 0:
        return {
            'feasibility': feas_rate, 'elbo_top1': elbo_top1,
            'cstr_top1': 0.0, 'cstr_best_idx': -1,
            'elbo_best_idx': elbo_best,
        }

    feasible_elbos = elbos.copy()
    feasible_elbos[~feasible] = np.inf
    cstr_best = np.argmin(feasible_elbos)
    cstr_top1 = 1.0 if cstr_best == oracle_best else 0.0

    return {
        'feasibility': feas_rate, 'elbo_top1': elbo_top1,
        'cstr_top1': cstr_top1, 'cstr_best_idx': cstr_best,
        'elbo_best_idx': elbo_best,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop for a single seed
# ---------------------------------------------------------------------------
def run_seed(seed, gamma, variant, device, fine_tune_epochs=50):
    print(f'\n{"="*70}')
    print(f'Seed={seed}, Gamma={gamma}, Variant={variant}')
    print(f'{"="*70}')

    cfg = build_config(seed, gamma)
    set_seed(seed)

    dataset_collection = SyntheticCancerDatasetCollectionCont(
        chemo_coeff=gamma, radio_coeff=gamma,
        num_patients={'train': 1000, 'val': 100, 'test': 100},
        seed=seed, window_size=15, max_seq_length=60,
        projection_horizon=5, lag=0,
        cf_seq_mode='sliding_treatment', treatment_mode='continuous',
    )
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    dims = len(dataset_collection.train_f.data['static_features'].shape)
    if dims == 2:
        dataset_collection = repeat_static(dataset_collection)

    # Load vanilla model
    model = VAEModel(cfg, dataset_collection)
    ckpt_path = (f'/root/VCIP/my_outputs/cancer_sim_cont/22/coeff_{gamma}/'
                 f'VCIP/train/True/models/{seed}/model.ckpt')
    print(f'Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    model.to(device)

    scaling_params = dataset_collection.train_scaling_params
    cv_mean = float(scaling_params[0]['cancer_volume'])
    cv_std_val = float(scaling_params[1]['cancer_volume'])
    print(f'CV scaling: mean={cv_mean:.4f}, std={cv_std_val:.4f}')

    # Apply modification
    if variant == 'heteroscedastic':
        patch_heteroscedastic(model)
        fine_tune_decoder(model, dataset_collection, cfg,
                         num_epochs=fine_tune_epochs, lr=1e-3, variant=variant)
    elif variant == 'wider':
        patch_wider_decoder(model)
        fine_tune_decoder(model, dataset_collection, cfg,
                         num_epochs=fine_tune_epochs, lr=1e-3, variant=variant)
    elif variant == 'mcdropout':
        patch_mcdropout(model, p=0.2)
        fine_tune_decoder(model, dataset_collection, cfg,
                         num_epochs=fine_tune_epochs, lr=1e-3, variant=variant)
    elif variant == 'vanilla':
        pass  # No modification, just evaluate
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Set MC passes for evaluation
    mc_passes = 20 if variant == 'mcdropout' else 1

    # Evaluate
    model.eval()
    if variant == 'mcdropout':
        # Keep dropout active for MC inference
        for m in model.generative_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    data = dataset_collection.val_f.data
    results = {}

    for tau_val in TAUS:
        print(f'\n--- tau={tau_val} ---')
        model.tau = tau_val
        model.config.exp.tau = tau_val
        set_seed(seed)

        dataloader = get_dataloader(CIPDataset(data, cfg), batch_size=1, shuffle=False)
        case_infos = []

        for i, batch in enumerate(dataloader):
            if i >= NUM_PATIENTS:
                break

            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            Y_targets = targets['outputs']
            true_actions = targets['current_treatments']
            true_actions_tau = true_actions[:, :tau_val, :]
            Y_targets_tau = Y_targets[:, :tau_val, :]

            all_sequences = generate_perturbed_sequences(
                true_actions_tau, K, tau_val, model.a_dim, device,
                treatment_mode=cfg.dataset.treatment_mode)

            elbos = []
            model_cv_terminals = []
            model_cv_maxes = []
            model_cv_stds = []
            oracle_cv_terminals = []
            oracle_cv_maxes = []
            oracle_cd_maxes = []
            true_losses = []

            with torch.no_grad():
                for seq in all_sequences:
                    # Model-predicted trajectory
                    elbo, cv_traj_mean, cv_traj_std = extract_predicted_cv(
                        model, H_t, Y_targets_tau, seq, scaling_params,
                        num_samples=NUM_SAMPLES, mc_passes=mc_passes)
                    elbos.append(elbo)

                    model_cv_terminals.append(cv_traj_mean[-1])
                    model_cv_maxes.append(cv_traj_mean.max())
                    model_cv_stds.append(cv_traj_std.mean())

                    # Oracle trajectory
                    sim_result = dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq, scaling_params, return_trajectory=True)

                    oracle_cv_traj = sim_result['cancer_volume_trajectory']
                    oracle_cd_traj = sim_result['chemo_dosage_trajectory']
                    oracle_scaled = sim_result['scaled_output']

                    oracle_cv_terminals.append(float(oracle_cv_traj[0, -1]))
                    oracle_cv_maxes.append(float(oracle_cv_traj[0].max()))
                    if oracle_cd_traj is not None:
                        oracle_cd_maxes.append(float(oracle_cd_traj[0].max()))

                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((oracle_scaled - true_output) ** 2).mean())
                    true_losses.append(true_loss)

            elbos = np.array(elbos)
            true_losses = np.array(true_losses)
            model_cv_terminal = np.array(model_cv_terminals)
            model_cv_max = np.array(model_cv_maxes)
            oracle_cv_terminal = np.array(oracle_cv_terminals)
            oracle_cv_max = np.array(oracle_cv_maxes)
            oracle_cd_max = np.array(oracle_cd_maxes) if oracle_cd_maxes else None

            # RA filtering
            oracle_feasible = apply_ra_filter(oracle_cv_terminal, oracle_cv_max, oracle_cd_max)
            model_feasible = apply_ra_filter(model_cv_terminal, model_cv_max, oracle_cd_max)

            oracle_metrics = ra_constrained_selection(elbos, oracle_feasible, true_losses)
            model_metrics = ra_constrained_selection(elbos, model_feasible, true_losses)

            # Check model's selection against oracle ground truth
            model_idx = model_metrics['cstr_best_idx']
            if model_idx >= 0:
                model_truly_safe = bool(oracle_feasible[model_idx])
                model_truly_intarget = bool(oracle_cv_terminal[model_idx] <= TARGET_UPPER)
            else:
                model_truly_safe = False
                model_truly_intarget = False

            case_infos.append({
                'individual_id': i,
                'oracle_metrics': oracle_metrics,
                'model_metrics': model_metrics,
                'model_truly_safe': model_truly_safe,
                'model_truly_intarget': model_truly_intarget,
                'oracle_feas': oracle_metrics['feasibility'],
                'model_feas': model_metrics['feasibility'],
                'model_cv_terminal_range': (float(model_cv_terminal.min()),
                                             float(model_cv_terminal.max())),
                'oracle_cv_terminal_range': (float(oracle_cv_terminal.min()),
                                              float(oracle_cv_terminal.max())),
                'model_cv_std_mean': float(np.mean(model_cv_stds)),
            })

            if i % 25 == 0:
                print(f'  [{i:3d}] orac_feas={oracle_metrics["feasibility"]:.2f} '
                      f'modl_feas={model_metrics["feasibility"]:.2f} '
                      f'modl_cv_t=[{model_cv_terminal.min():.2f},{model_cv_terminal.max():.2f}] '
                      f'orac_cv_t=[{oracle_cv_terminal.min():.2f},{oracle_cv_terminal.max():.2f}]')

        results[tau_val] = case_infos

        # Summary
        n = len(case_infos)
        orac_feas = np.mean([c['oracle_feas'] for c in case_infos])
        modl_feas = np.mean([c['model_feas'] for c in case_infos])
        elbo_top1 = np.mean([c['oracle_metrics']['elbo_top1'] for c in case_infos])
        orac_top1 = np.mean([c['oracle_metrics']['cstr_top1'] for c in case_infos])
        modl_top1 = np.mean([c['model_metrics']['cstr_top1'] for c in case_infos])
        modl_safe = np.mean([c['model_truly_safe'] for c in case_infos])
        modl_intgt = np.mean([c['model_truly_intarget'] for c in case_infos])

        print(f'\n  tau={tau_val} (n={n}): '
              f'feas: orac={orac_feas:.3f} modl={modl_feas:.3f} | '
              f'top1: elbo={elbo_top1:.3f} orac={orac_top1:.3f} modl={modl_top1:.3f} | '
              f'modl_truly_safe={modl_safe:.3f} modl_truly_intgt={modl_intgt:.3f}')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                       choices=['vanilla', 'heteroscedastic', 'wider', 'mcdropout'],
                       help='Decoder modification variant')
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--all_seeds', action='store_true')
    parser.add_argument('--fine_tune_epochs', type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS if args.all_seeds else ([args.seed] if args.seed else [10])

    print(f'Variant: {args.variant}')
    print(f'Gamma: {args.gamma}')
    print(f'Seeds: {seeds}')
    print(f'Fine-tune epochs: {args.fine_tune_epochs}')
    print(f'Device: {device}')

    all_results = {}
    for seed in seeds:
        results = run_seed(seed, args.gamma, args.variant, device,
                          fine_tune_epochs=args.fine_tune_epochs)
        all_results[seed] = results

    # Save
    out_dir = f'/root/results_decoder_mods'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f'{args.variant}_gamma{args.gamma}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved: {out_path}')

    # Grand summary
    print(f'\n{"="*70}')
    print(f'SUMMARY: {args.variant} (gamma={args.gamma})')
    print(f'{"="*70}')
    print(f'{"tau":>4} | {"ELBO Top1":>10} | {"Orac Feas":>10} | {"Modl Feas":>10} | '
          f'{"Orac Top1":>10} | {"Modl Top1":>10} | {"Modl Safe":>10} | {"Modl InTgt":>10}')
    print('-' * 90)

    for tau_val in TAUS:
        metrics = {'elbo_top1': [], 'orac_feas': [], 'modl_feas': [],
                   'orac_top1': [], 'modl_top1': [], 'modl_safe': [], 'modl_intgt': []}
        for seed in seeds:
            if seed not in all_results or tau_val not in all_results[seed]:
                continue
            for c in all_results[seed][tau_val]:
                metrics['elbo_top1'].append(c['oracle_metrics']['elbo_top1'])
                metrics['orac_feas'].append(c['oracle_feas'])
                metrics['modl_feas'].append(c['model_feas'])
                metrics['orac_top1'].append(c['oracle_metrics']['cstr_top1'])
                metrics['modl_top1'].append(c['model_metrics']['cstr_top1'])
                metrics['modl_safe'].append(c['model_truly_safe'])
                metrics['modl_intgt'].append(c['model_truly_intarget'])

        print(f'{tau_val:>4} | {np.mean(metrics["elbo_top1"]):>10.3f} | '
              f'{np.mean(metrics["orac_feas"]):>10.3f} | {np.mean(metrics["modl_feas"]):>10.3f} | '
              f'{np.mean(metrics["orac_top1"]):>10.3f} | {np.mean(metrics["modl_top1"]):>10.3f} | '
              f'{np.mean(metrics["modl_safe"]):>10.3f} | {np.mean(metrics["modl_intgt"]):>10.3f}')


if __name__ == '__main__':
    main()
