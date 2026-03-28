#!/usr/bin/env python3
"""
A3: Oracle vs Model-Predicted RA Filtering on Cancer.

Compares reach-avoid constrained selection using:
1. Oracle trajectories (simulator ground truth)
2. Model-predicted trajectories (VCIP decoder outputs at each step)

This is the key experiment addressing W3 (oracle dependency):
- Creates 3-level validation: oracle -> model-predicted synthetic -> model-predicted real (MIMIC)
- Quantifies the gap between oracle and model-predicted RA filtering

Usage (on Vast.ai, from VCIP root):
    source /root/vcip_env/bin/activate
    cd /root/VCIP
    python3 /root/eval_cancer_oracle_vs_model.py --gamma 4 --all_seeds
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, '/root/VCIP')
sys.path.insert(0, '/root/VCIP/src')

from src.data.cancer_sim_cont.dataset import SyntheticCancerDatasetCollectionCont
from src.models.vae_model import VAEModel
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed, to_float, repeat_static
from src.utils.helper_functions import generate_perturbed_sequences, compute_kl_divergence
from omegaconf import OmegaConf


# RA thresholds (moderate, from Phase 1B calibration)
TARGET_UPPER = 0.6       # terminal cancer volume (unscaled)
SAFETY_VOL_UPPER = 5.0   # max cancer volume at any step (unscaled)
SAFETY_CHEMO_UPPER = 8.5  # max chemo dosage

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


def extract_predicted_cv(model, H_t, Y_targets, a_seq, X_targets=None, num_samples=10):
    """Run the ELBO forward pass and extract predicted cancer volume at each step.

    Returns:
        elbo: float
        cv_trajectory_scaled: np.array of shape (tau,) -- predicted CV (scaled) at each step
    """
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

        # Decode predicted outcome at this step
        a_decode = a_seq_hiddens_1[:, -1, :]
        Z_mean = Z_samples.mean(dim=0) if Z_samples.dim() == 3 else Z_samples
        y_mu, _ = model.generative_model.decode_p_a(Z_mean, a_decode)
        predicted_cv.append(float(y_mu.detach().cpu().flatten()[0]))

        # Reg loss (mirrors calculate_elbo)
        if s > 0 and s < tau - 1:
            Y_current = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
            a = a_seq_hiddens_1[:, -1, :].unsqueeze(0).expand(num_samples, -1, -1)
            reg_losses.append(model.generative_model.decoding_Y_loss_2(Z_samples, Y_current, a))

        if s == tau - 1:
            Y_current = Y_targets[:, s, :]
            a = a_seq_hiddens_1[:, -1, :]
            reg_losses.append(model.generative_model.decoding_Y_loss_2(Z_mean, Y_current, a))

        Z_s_i = Z_samples.mean(dim=0) if Z_samples.dim() == 3 else Z_samples

    kl_loss = torch.stack(kl_losses).mean()
    reg_loss = reg_losses[-1] if reg_losses else torch.tensor(0.0, device=Z_s_i.device)
    action_loss = torch.stack(action_losses).mean() if action_losses else torch.tensor(0.0, device=Z_s_i.device)

    elbo = (reg_loss * model.config.exp.lambda_reg +
            kl_loss * model.config.exp.lambda_kl +
            action_loss * model.config.exp.lambda_step +
            predict_action_loss * model.config.exp.lambda_action +
            loss_hy * model.config.exp.lambda_hy)

    return elbo.item(), np.array(predicted_cv)


def apply_ra_filter(cv_terminal, cv_max, cd_max):
    """Apply reach-avoid feasibility check. Returns boolean array."""
    feasible = (cv_terminal <= TARGET_UPPER) & (cv_max <= SAFETY_VOL_UPPER)
    if cd_max is not None:
        feasible = feasible & (cd_max <= SAFETY_CHEMO_UPPER)
    return feasible


def ra_constrained_selection(elbos, feasible, true_losses):
    """Select best ELBO among feasible candidates. Returns metrics dict."""
    k = len(elbos)
    feas_rate = feasible.sum() / k

    # Unconstrained (ELBO-only)
    elbo_best = np.argmin(elbos)
    elbo_top1_loss = true_losses[elbo_best]
    oracle_best = np.argmin(true_losses)
    elbo_top1 = 1.0 if elbo_best == oracle_best else 0.0

    # Constrained
    if feasible.sum() == 0:
        cstr_top1 = 0.0
        cstr_top1_loss = np.nan
        cstr_best = -1
        cstr_in_target = False
        cstr_safe = False
    else:
        feasible_elbos = elbos.copy()
        feasible_elbos[~feasible] = np.inf
        cstr_best = np.argmin(feasible_elbos)
        cstr_top1_loss = true_losses[cstr_best]
        cstr_top1 = 1.0 if cstr_best == oracle_best else 0.0
        cstr_in_target = True  # by construction (feasible => in target)
        cstr_safe = True

    return {
        'feasibility': feas_rate,
        'elbo_top1': elbo_top1,
        'cstr_top1': cstr_top1,
        'elbo_best_idx': elbo_best,
        'cstr_best_idx': cstr_best,
        'elbo_top1_loss': elbo_top1_loss,
        'cstr_top1_loss': cstr_top1_loss,
    }


def run_seed(seed, gamma, device):
    """Run oracle-vs-model comparison for one seed."""
    print(f'\n{"="*70}')
    print(f'Seed={seed}, Gamma={gamma}')
    print(f'{"="*70}')

    # Build config and load data
    cfg = build_config(seed, gamma)
    set_seed(seed)

    # Instantiate dataset with explicit params (matching cancer_sim_cont.yaml)
    dataset_collection = SyntheticCancerDatasetCollectionCont(
        chemo_coeff=gamma,
        radio_coeff=gamma,
        num_patients={'train': 1000, 'val': 100, 'test': 100},
        seed=seed,
        window_size=15,
        max_seq_length=60,
        projection_horizon=5,
        lag=0,
        cf_seq_mode='sliding_treatment',
        treatment_mode='continuous',
    )
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    dims = len(dataset_collection.train_f.data['static_features'].shape)
    if dims == 2:
        dataset_collection = repeat_static(dataset_collection)

    # Instantiate model
    model = VAEModel(cfg, dataset_collection)

    # Load checkpoint
    ckpt_path = (f'/root/VCIP/my_outputs/cancer_sim_cont/22/coeff_{gamma}/'
                 f'VCIP/train/True/models/{seed}/model.ckpt')
    print(f'Loading: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Get scaling params
    scaling_params = dataset_collection.train_scaling_params
    cv_mean = float(scaling_params[0]['cancer_volume'])
    cv_std = float(scaling_params[1]['cancer_volume'])
    print(f'CV scaling: mean={cv_mean:.4f}, std={cv_std:.4f}')

    # Get validation data
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
            X_targets = None if not model.predict_X else targets['vitals']
            true_actions = targets['current_treatments']

            # Slice to tau length
            true_actions_tau = true_actions[:, :tau_val, :]
            Y_targets_tau = Y_targets[:, :tau_val, :]

            all_sequences = generate_perturbed_sequences(
                true_actions_tau, K, tau_val, model.a_dim, device,
                treatment_mode=cfg.dataset.treatment_mode)

            elbos = []
            model_cv_trajectories = []
            oracle_traj_stats = []
            true_losses = []

            with torch.no_grad():
                for seq in all_sequences:
                    # 1. Compute ELBO + extract model-predicted CV trajectory
                    elbo, cv_traj_scaled = extract_predicted_cv(
                        model, H_t, Y_targets_tau, seq, X_targets,
                        num_samples=NUM_SAMPLES)
                    elbos.append(elbo)

                    # Unscale model predictions
                    cv_traj_unscaled = cv_traj_scaled * cv_std + cv_mean
                    model_cv_trajectories.append(cv_traj_unscaled)

                    # 2. Get oracle trajectory from simulator
                    sim_result = dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq, scaling_params, return_trajectory=True)

                    oracle_cv_traj = sim_result['cancer_volume_trajectory']  # (1, tau+1)
                    oracle_cd_traj = sim_result['chemo_dosage_trajectory']   # (1, tau)
                    oracle_scaled = sim_result['scaled_output']              # (1, 1)

                    oracle_traj_stats.append({
                        'cv_terminal': float(oracle_cv_traj[0, -1]),
                        'cv_max': float(oracle_cv_traj[0].max()),
                        'cd_max': float(oracle_cd_traj[0].max()) if oracle_cd_traj is not None else None,
                    })

                    # True loss (oracle scaled output vs observed)
                    true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
                    true_loss = np.sqrt(((oracle_scaled - true_output) ** 2).mean())
                    true_losses.append(true_loss)

            elbos = np.array(elbos)
            true_losses = np.array(true_losses)
            model_cv_arr = np.array(model_cv_trajectories)  # (k, tau)

            # Build feature arrays
            oracle_cv_terminal = np.array([s['cv_terminal'] for s in oracle_traj_stats])
            oracle_cv_max = np.array([s['cv_max'] for s in oracle_traj_stats])
            oracle_cd_max = np.array([s['cd_max'] for s in oracle_traj_stats
                                      if s['cd_max'] is not None])

            model_cv_terminal = model_cv_arr[:, -1]
            model_cv_max = model_cv_arr.max(axis=1)
            # Chemo dosage is action-derived (same for both oracle and model)
            model_cd_max = oracle_cd_max.copy() if len(oracle_cd_max) > 0 else None

            # Apply RA filtering: oracle vs model
            oracle_feasible = apply_ra_filter(oracle_cv_terminal, oracle_cv_max,
                                              oracle_cd_max if len(oracle_cd_max) > 0 else None)
            model_feasible = apply_ra_filter(model_cv_terminal, model_cv_max, model_cd_max)

            oracle_metrics = ra_constrained_selection(elbos, oracle_feasible, true_losses)
            model_metrics = ra_constrained_selection(elbos, model_feasible, true_losses)

            # Check if oracle-selected plan is truly safe (always true by construction)
            # Check if model-selected plan is truly safe (oracle check on model's selection)
            model_cstr_idx = model_metrics['cstr_best_idx']
            if model_cstr_idx >= 0:
                model_selection_oracle_safe = bool(oracle_feasible[model_cstr_idx])
                model_selection_oracle_in_target = bool(oracle_cv_terminal[model_cstr_idx] <= TARGET_UPPER)
            else:
                model_selection_oracle_safe = False
                model_selection_oracle_in_target = False

            case_info = {
                'individual_id': i,
                'model_losses': elbos.astype(np.float32),
                'true_losses': true_losses.astype(np.float32),
                'oracle_traj_features': {
                    'cv_terminal': oracle_cv_terminal.astype(np.float32),
                    'cv_max': oracle_cv_max.astype(np.float32),
                    'cd_max': oracle_cd_max.astype(np.float32) if len(oracle_cd_max) > 0 else None,
                },
                'model_traj_features': {
                    'cv_terminal': model_cv_terminal.astype(np.float32),
                    'cv_max': model_cv_max.astype(np.float32),
                    'cv_trajectory': model_cv_arr.astype(np.float32),
                },
                'oracle_feasible': oracle_feasible,
                'model_feasible': model_feasible,
                'oracle_metrics': oracle_metrics,
                'model_metrics': model_metrics,
                'model_selection_oracle_safe': model_selection_oracle_safe,
                'model_selection_oracle_in_target': model_selection_oracle_in_target,
            }
            case_infos.append(case_info)

            if i % 25 == 0:
                print(f'  [{i:3d}] oracle_feas={oracle_metrics["feasibility"]:.2f} '
                      f'model_feas={model_metrics["feasibility"]:.2f} '
                      f'oracle_cv_t=[{oracle_cv_terminal.min():.3f},{oracle_cv_terminal.max():.3f}] '
                      f'model_cv_t=[{model_cv_terminal.min():.3f},{model_cv_terminal.max():.3f}]')

        results[tau_val] = case_infos

        # Print summary for this tau
        n = len(case_infos)
        print(f'\n  tau={tau_val} Summary (n={n}):')

        oracle_feas = np.mean([c['oracle_metrics']['feasibility'] for c in case_infos])
        model_feas = np.mean([c['model_metrics']['feasibility'] for c in case_infos])
        oracle_top1 = np.mean([c['oracle_metrics']['cstr_top1'] for c in case_infos])
        model_top1 = np.mean([c['model_metrics']['cstr_top1'] for c in case_infos])
        elbo_top1 = np.mean([c['oracle_metrics']['elbo_top1'] for c in case_infos])
        model_oracle_safe = np.mean([c['model_selection_oracle_safe'] for c in case_infos])
        model_oracle_intarget = np.mean([c['model_selection_oracle_in_target'] for c in case_infos])

        # In-target rates (using oracle ground truth)
        oracle_intarget_list = []
        model_intarget_list = []
        elbo_intarget_list = []
        for c in case_infos:
            oracle_idx = c['oracle_metrics']['cstr_best_idx']
            model_idx = c['model_metrics']['cstr_best_idx']
            elbo_idx = c['oracle_metrics']['elbo_best_idx']
            cv_t = c['oracle_traj_features']['cv_terminal']
            elbo_intarget_list.append(cv_t[elbo_idx] <= TARGET_UPPER)
            if oracle_idx >= 0:
                oracle_intarget_list.append(cv_t[oracle_idx] <= TARGET_UPPER)
            if model_idx >= 0:
                model_intarget_list.append(cv_t[model_idx] <= TARGET_UPPER)

        elbo_intarget = np.mean(elbo_intarget_list) if elbo_intarget_list else 0
        oracle_intarget = np.mean(oracle_intarget_list) if oracle_intarget_list else 0
        model_intarget = np.mean(model_intarget_list) if model_intarget_list else 0

        print(f'    Feasibility:    oracle={oracle_feas:.3f}  model={model_feas:.3f}')
        print(f'    ELBO Top-1:     {elbo_top1:.3f}')
        print(f'    Cstr Top-1:     oracle={oracle_top1:.3f}  model={model_top1:.3f}')
        print(f'    In-target (oracle GT): ELBO={elbo_intarget:.3f}  '
              f'oracle_RA={oracle_intarget:.3f}  model_RA={model_intarget:.3f}')
        print(f'    Model-selected truly safe: {model_oracle_safe:.3f}')
        print(f'    Model-selected truly in-target: {model_oracle_intarget:.3f}')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--all_seeds', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    print(f'Gamma: {args.gamma}')

    seeds = SEEDS if args.all_seeds else ([args.seed] if args.seed else [10])
    print(f'Seeds: {seeds}')

    all_results = {}
    for seed in seeds:
        results = run_seed(seed, args.gamma, device)
        all_results[seed] = results

    # Save
    out_dir = '/root/results_a3'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'a3_oracle_vs_model_gamma{args.gamma}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved: {out_path}')

    # Print grand summary
    print(f'\n{"="*70}')
    print(f'GRAND SUMMARY: Oracle vs Model-Predicted RA Filtering (gamma={args.gamma})')
    print(f'{"="*70}')
    print(f'{"tau":>4} | {"ELBO Top1":>10} | {"Orac Top1":>10} | {"Modl Top1":>10} | '
          f'{"ELBO InTgt":>10} | {"Orac InTgt":>10} | {"Modl InTgt":>10} | '
          f'{"Orac Feas":>10} | {"Modl Feas":>10} | {"Modl TruSafe":>11}')
    print('-' * 120)

    for tau_val in TAUS:
        elbo_top1s, orac_top1s, modl_top1s = [], [], []
        elbo_intgts, orac_intgts, modl_intgts = [], [], []
        orac_feass, modl_feass = [], []
        modl_safes = []

        for seed in seeds:
            if seed not in all_results or tau_val not in all_results[seed]:
                continue
            cases = all_results[seed][tau_val]
            for c in cases:
                elbo_top1s.append(c['oracle_metrics']['elbo_top1'])
                orac_top1s.append(c['oracle_metrics']['cstr_top1'])
                modl_top1s.append(c['model_metrics']['cstr_top1'])
                orac_feass.append(c['oracle_metrics']['feasibility'])
                modl_feass.append(c['model_metrics']['feasibility'])
                modl_safes.append(c['model_selection_oracle_safe'])

                cv_t = c['oracle_traj_features']['cv_terminal']
                elbo_idx = c['oracle_metrics']['elbo_best_idx']
                orac_idx = c['oracle_metrics']['cstr_best_idx']
                modl_idx = c['model_metrics']['cstr_best_idx']
                elbo_intgts.append(cv_t[elbo_idx] <= TARGET_UPPER)
                if orac_idx >= 0:
                    orac_intgts.append(cv_t[orac_idx] <= TARGET_UPPER)
                if modl_idx >= 0:
                    modl_intgts.append(cv_t[modl_idx] <= TARGET_UPPER)

        print(f'{tau_val:>4} | {np.mean(elbo_top1s):>10.3f} | {np.mean(orac_top1s):>10.3f} | '
              f'{np.mean(modl_top1s):>10.3f} | {np.mean(elbo_intgts):>10.3f} | '
              f'{np.mean(orac_intgts) if orac_intgts else 0:>10.3f} | '
              f'{np.mean(modl_intgts) if modl_intgts else 0:>10.3f} | '
              f'{np.mean(orac_feass):>10.3f} | {np.mean(modl_feass):>10.3f} | '
              f'{np.mean(modl_safes):>11.3f}')


if __name__ == '__main__':
    main()
