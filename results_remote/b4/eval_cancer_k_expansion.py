#!/usr/bin/env python3
"""
B4: k-Expansion Experiment for Cancer RA Filtering.

Shows that feasibility scales with candidate pool size k.
Runs gamma=4, tau=8 (worst feasibility) with k ∈ {250, 500, 1000}.

Optimization: compute simulator trajectories first (fast), then ELBO only for
feasible candidates (saves ~90% of neural network forward passes).

Usage:
    cd /root/VCIP && source /root/vcip_env/bin/activate
    python3 /root/eval_cancer_k_expansion.py --all_seeds
"""

import os, sys, pickle, argparse
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, '/root/VCIP')
sys.path.insert(0, '/root/VCIP/src')

from src.data.cancer_sim_cont.dataset import SyntheticCancerDatasetCollectionCont
from src.models.vae_model import VAEModel
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.utils.utils import set_seed, to_float, repeat_static
from src.utils.helper_functions import generate_perturbed_sequences
from omegaconf import OmegaConf

TARGET_UPPER = 0.6
SAFETY_VOL_UPPER = 5.0
SAFETY_CHEMO_UPPER = 8.5

GAMMA = 4
TAU = 8
K_VALUES = [100, 250, 500, 1000]
NUM_PATIENTS = 100
NUM_SAMPLES = 10
SEEDS = [10, 101, 1010, 10101, 101010]


def build_config(seed, gamma):
    """Build OmegaConf config matching VCIP Cancer training."""
    config = OmegaConf.create({
        'exp': {
            'seed': seed, 'global_seed': 22, 'max_epochs': 100,
            'mode': 'train', 'test': False, 'sam': True,
            'num_samples': NUM_SAMPLES, 'alpha': 0.001, 'beta': 0.99,
            'batch_size': 512, 'batch_size_val': 128, 'dropout': 0.2,
            'entropy_reg': 1, 'remove_aux': False, 'beta_bound': -10,
            'lambda_reg': 1, 'lambda_kl': 1, 'lambda_step': 0.1,
            'lambda_action': 1, 'lambda_hy': 1, 'lambda_X': 0.1, 'lambda_Y': 1,
            'tau': TAU, 'repeats': 5, 'rank': True,
            'unscale_rmse': True, 'percentage_rmse': True,
            'processed_data_dir': '/root/VCIP/data/processed/cancer_sim_cont',
            'log_dir': f'/root/VCIP/my_outputs/cancer_sim_cont/22/coeff_{gamma}/VCIP/train/True',
            'device': 'cuda', 'gpus': 1, 'weights_ema': True,
            'clear_tf': True, 'logging': False, 'load_data': True,
            'load_best': True, 'val': True, 'lr': 0.01,
            'weight_decay': 1e-5, 'epochs': 100, 'patience': 20,
        },
        'dataset': {
            '_target_': 'src.data.SyntheticCancerDatasetCollectionCont',
            'name': 'cancer_sim_cont', 'gamma': gamma, 'coeff': gamma,
            'projection_horizon': 5, 'lag': 0, 'window_size': 15,
            'max_seq_length': 60, 'cf_seq_mode': 'sliding_treatment',
            'treatment_mode': 'continuous',
            'num_patients': {'train': 1000, 'val': 100, 'test': 100},
            'static_size': 1, 'treatment_size': 2, 'one_hot_treatment_size': 4,
            'input_size': 0, 'output_size': 1, 'predict_X': False,
            'autoregressive': True, 'val_batch_size': 4096, 'seed': seed,
        },
        'model': {
            '_target_': 'src.models.vae_model.VAEModel', 'name': 'VCIP',
            'z_dim': 16, 'dim_treatments': 2, 'dim_vitals': 0,
            'dim_static_features': 1, 'dim_outcomes': 1, 'lr': 0.001,
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


def compute_elbo(model, H_t, Y_targets, a_seq, X_targets=None, num_samples=10):
    """Compute ELBO for a single candidate sequence."""
    from src.utils.helper_functions import compute_kl_divergence
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
    kl_losses, reg_losses, action_losses = [], [], []
    Y_last = Y_targets[:, -1, :]
    tau = model.tau
    hidden_states = model.generative_model.build_hidden_states(Z_s_i, a_seq)
    for s in range(tau):
        a_s_hidden = a_seq_hiddens[:, s, :]
        Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
        if s < tau - 1:
            a_s_next = a_seq[:, s + 1, :]
            if model.config.dataset.treatment_mode == 'multilabel':
                action_losses.append(model.generative_model.bern_loss(Z_s_i, a_s_next))
            else:
                action_losses.append(model.generative_model.beta_loss(Z_s_i, a_s_next))
        q_mu, q_logvar = model.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)
        p_mu, p_logvar = model.generative_model(Z_s_i, a_s_hidden)
        Z_samples = model.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)
        kl_losses.append(compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar))
        Z_mean = Z_samples.mean(dim=0) if Z_samples.dim() == 3 else Z_samples
        if s > 0 and s < tau - 1:
            Y_c = Y_targets[:, s, :].unsqueeze(0).expand(num_samples, -1, -1)
            a = a_seq_hiddens_1[:, -1, :].unsqueeze(0).expand(num_samples, -1, -1)
            reg_losses.append(model.generative_model.decoding_Y_loss_2(Z_samples, Y_c, a))
        if s == tau - 1:
            reg_losses.append(model.generative_model.decoding_Y_loss_2(
                Z_mean, Y_targets[:, s, :], a_seq_hiddens_1[:, -1, :]))
        Z_s_i = Z_mean
    kl_loss = torch.stack(kl_losses).mean()
    reg_loss = reg_losses[-1] if reg_losses else torch.tensor(0.0)
    action_loss = torch.stack(action_losses).mean() if action_losses else torch.tensor(0.0)
    elbo = (reg_loss * model.config.exp.lambda_reg +
            kl_loss * model.config.exp.lambda_kl +
            action_loss * model.config.exp.lambda_step +
            predict_action_loss * model.config.exp.lambda_action +
            loss_hy * model.config.exp.lambda_hy)
    return elbo.item()


def run_seed(seed, device):
    """Run k-expansion for one seed."""
    print(f'\n{"="*70}')
    print(f'Seed={seed}, Gamma={GAMMA}, Tau={TAU}')
    print(f'{"="*70}')

    cfg = build_config(seed, GAMMA)
    set_seed(seed)

    dataset_collection = SyntheticCancerDatasetCollectionCont(
        chemo_coeff=GAMMA, radio_coeff=GAMMA,
        num_patients={'train': 1000, 'val': 100, 'test': 100},
        seed=seed, window_size=15, max_seq_length=60,
        projection_horizon=5, lag=0, cf_seq_mode='sliding_treatment',
        treatment_mode='continuous')
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    dims = len(dataset_collection.train_f.data['static_features'].shape)
    if dims == 2:
        dataset_collection = repeat_static(dataset_collection)

    model = VAEModel(cfg, dataset_collection)
    ckpt_path = (f'/root/VCIP/my_outputs/cancer_sim_cont/22/coeff_{GAMMA}/'
                 f'VCIP/train/True/models/{seed}/model.ckpt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    model.tau = TAU
    model.config.exp.tau = TAU

    scaling_params = dataset_collection.train_scaling_params
    data = dataset_collection.val_f.data

    k_max = max(K_VALUES)
    results = {k: [] for k in K_VALUES}

    set_seed(seed)
    dataloader = get_dataloader(CIPDataset(data, cfg), batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        if i >= NUM_PATIENTS:
            break

        H_t, targets = batch
        for key in H_t:
            H_t[key] = H_t[key].to(device)
        for key in targets:
            targets[key] = targets[key].to(device)

        Y_targets = targets['outputs'][:, :TAU, :]
        X_targets = None if not model.predict_X else targets['vitals']
        true_actions = targets['current_treatments'][:, :TAU, :]

        # Generate k_max candidates
        all_sequences = generate_perturbed_sequences(
            true_actions, k_max, TAU, model.a_dim, device,
            treatment_mode=cfg.dataset.treatment_mode)

        # Phase 1: Get ALL oracle trajectories (fast, ~0.01s each)
        oracle_stats = []
        true_losses = []
        for j, seq in enumerate(all_sequences):
            sim_result = dataset_collection.val_f.simulate_output_after_actions(
                H_t, seq, scaling_params, return_trajectory=True)
            cv_traj = sim_result['cancer_volume_trajectory']
            cd_traj = sim_result['chemo_dosage_trajectory']
            oracle_stats.append({
                'cv_terminal': float(cv_traj[0, -1]),
                'cv_max': float(cv_traj[0].max()),
                'cd_max': float(cd_traj[0].max()) if cd_traj is not None else None,
            })
            true_output = targets['outputs'][:, -1, :].detach().cpu().numpy()
            true_loss = np.sqrt(((sim_result['scaled_output'] - true_output) ** 2).mean())
            true_losses.append(true_loss)

        true_losses = np.array(true_losses)
        cv_terminal = np.array([s['cv_terminal'] for s in oracle_stats])
        cv_max = np.array([s['cv_max'] for s in oracle_stats])
        cd_max = np.array([s['cd_max'] for s in oracle_stats if s['cd_max'] is not None])

        # Phase 2: For each k, compute feasibility and ELBO for feasible only
        for k in K_VALUES:
            # Use first k candidates
            k_cv_t = cv_terminal[:k]
            k_cv_m = cv_max[:k]
            k_cd_m = cd_max[:k] if len(cd_max) >= k else cd_max
            k_true = true_losses[:k]

            feasible = (k_cv_t <= TARGET_UPPER) & (k_cv_m <= SAFETY_VOL_UPPER)
            if len(k_cd_m) > 0:
                feasible = feasible & (k_cd_m <= SAFETY_CHEMO_UPPER)

            feas_rate = feasible.sum() / k
            n_feasible = feasible.sum()

            # Compute ELBO only for feasible candidates
            elbos = np.full(k, np.inf)
            if n_feasible > 0:
                feasible_indices = np.where(feasible)[0]
                with torch.no_grad():
                    for idx in feasible_indices:
                        seq = all_sequences[idx]
                        elbos[idx] = compute_elbo(
                            model, H_t, Y_targets, seq, X_targets,
                            num_samples=NUM_SAMPLES)

            # Also compute ELBO for the true sequence (last one in k=100)
            # For ranking comparison, compute ELBO for a sample of infeasible too
            # Actually, for Top-1 we need the oracle best (min true_loss)
            oracle_best = np.argmin(k_true)

            # Constrained selection
            if n_feasible > 0:
                cstr_best = np.argmin(elbos)  # best ELBO among feasible
                cstr_top1 = 1.0 if cstr_best == oracle_best else 0.0
                cstr_in_target = bool(k_cv_t[cstr_best] <= TARGET_UPPER)
                cstr_safe = bool(k_cv_m[cstr_best] <= SAFETY_VOL_UPPER)
            else:
                cstr_top1 = 0.0
                cstr_in_target = False
                cstr_safe = False

            # ELBO-only selection (need ELBO for all, or at least first 100)
            # For fair comparison, compute ELBO for first 100 (same as baseline)
            if k == K_VALUES[0]:  # Only compute for k=100
                all_elbos_100 = []
                with torch.no_grad():
                    for seq in all_sequences[:100]:
                        all_elbos_100.append(compute_elbo(
                            model, H_t, Y_targets, seq, X_targets, NUM_SAMPLES))
                all_elbos_100 = np.array(all_elbos_100)
                elbo_best_100 = np.argmin(all_elbos_100)
                elbo_top1 = 1.0 if elbo_best_100 == oracle_best else 0.0
                elbo_in_target = bool(k_cv_t[elbo_best_100] <= TARGET_UPPER)
            else:
                elbo_top1 = np.nan
                elbo_in_target = np.nan

            results[k].append({
                'individual_id': i,
                'k': k,
                'feasibility': feas_rate,
                'n_feasible': int(n_feasible),
                'cstr_top1': cstr_top1,
                'cstr_in_target': cstr_in_target,
                'cstr_safe': cstr_safe,
                'elbo_top1': elbo_top1,
                'elbo_in_target': elbo_in_target,
            })

        if i % 10 == 0:
            feas_str = ' '.join([f'k={k}:{results[k][-1]["feasibility"]:.2f}'
                                 for k in K_VALUES])
            print(f'  [{i:3d}] {feas_str}')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--all_seeds', action='store_true')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS if args.all_seeds else ([args.seed] if args.seed else [10])
    print(f'B4 k-expansion: gamma={GAMMA}, tau={TAU}, seeds={seeds}')
    print(f'k values: {K_VALUES}')

    all_results = {}
    for seed in seeds:
        results = run_seed(seed, device)
        all_results[seed] = results

    # Save
    out_dir = '/root/results_b4'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'b4_k_expansion_gamma{GAMMA}_tau{TAU}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved: {out_path}')

    # Summary
    print(f'\n{"="*70}')
    print(f'B4 SUMMARY: k-Expansion (gamma={GAMMA}, tau={TAU})')
    print(f'{"="*70}')
    print(f'{"k":>6} | {"Feasibility":>12} | {"N_feasible":>11} | {"Cstr Top-1":>11} | '
          f'{"Cstr InTgt":>11} | {"Cstr Safe":>10}')
    print('-' * 75)

    for k in K_VALUES:
        all_cases = []
        for seed in seeds:
            if seed in all_results:
                all_cases.extend(all_results[seed][k])
        if all_cases:
            feas = np.mean([c['feasibility'] for c in all_cases])
            n_feas = np.mean([c['n_feasible'] for c in all_cases])
            top1 = np.mean([c['cstr_top1'] for c in all_cases])
            intgt = np.mean([c['cstr_in_target'] for c in all_cases])
            safe = np.mean([c['cstr_safe'] for c in all_cases])
            print(f'{k:>6} | {feas:>12.3f} | {n_feas:>11.1f} | {top1:>11.3f} | '
                  f'{intgt:>11.3f} | {safe:>10.3f}')


if __name__ == '__main__':
    main()
