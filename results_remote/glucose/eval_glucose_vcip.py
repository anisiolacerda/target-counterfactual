"""
S5.2: VCIP-based RA Evaluation on Glucose-Insulin Simulator
"""

import sys
import os
sys.path.insert(0, '/root/VCIP')
os.chdir('/root/VCIP')

import pickle
import numpy as np
import torch
import argparse
from scipy.stats import spearmanr
from omegaconf import OmegaConf

TARGET_LO = 70.0
TARGET_HI = 180.0
SAFETY_LO = 50.0
SAFETY_HI = 250.0
TARGET_MID = (TARGET_LO + TARGET_HI) / 2

DATA_DIR = '/root/VCIP/glucose_data'
MODEL_BASE = '/root/VCIP/my_outputs/glucose_sim/22/coeff_4/VCIP/train/True'
GAMMA = 4
SEEDS = [10, 101, 1010, 10101, 101010]
TAUS = [2, 4, 6, 8]


def build_config(seed):
    """Build OmegaConf config matching what training used."""
    config = OmegaConf.create({
        'model': {
            'dim_treatments': 1, 'dim_vitals': 0,
            'dim_static_features': 1, 'dim_outcomes': 1,
            '_target_': 'src.models.vae_model.VAEModel',
            'name': 'VCIP', 'z_dim': 16,
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
            'lr': 0.001,
        },
        'dataset': {
            '_target_': 'src.data.glucose.GlucoseDatasetCollection',
            'name': 'glucose_simulator',
            'data_dir': DATA_DIR, 'gamma': GAMMA, 'seed': seed,
            'projection_horizon': 5, 'treatment_mode': 'continuous',
            'val_batch_size': 4096, 'static_size': 1, 'treatment_size': 1,
            'one_hot_treatment_size': 1, 'input_size': 0, 'output_size': 1,
            'predict_X': False, 'autoregressive': True,
            'max_seq_length': 48, 'coeff': GAMMA,
            'window_size': 15, 'lag': 0, 'cf_seq_mode': 'sliding_treatment',
            'num_patients': {'train': 1000, 'val': 100, 'test': 100},
        },
        'exp': {
            'alpha': 1.0, 'update_alpha': True, 'alpha_rate': 'exp',
            'bce_weight': False,
            'unscale_rmse': True, 'percentage_rmse': False,
            'processed_data_dir': f'data/processed/glucose_sim/{GAMMA}/VCIP',
            'log_dir': f'my_outputs/glucose_sim/22/coeff_{GAMMA}/VCIP/train/True',
            'device': 'cuda', 'gpus': 1,
            'weights_ema': True, 'beta': 0.99,
            'clear_tf': True, 'logging': False,
            'current_date': '2026-03-27', 'current_time': '00-00-00',
            'mode': 'train', 'gpu_resources': 0.3, 'cpu_resources': 2,
            'csv_dir': f'csvs/glucose/gamma_{GAMMA}',
            'max_epochs': 100, 'global_seed': 22,
            'sam': True, 'load_data': True, 'seed': seed,
            'tau': 6, 'repeats': 5, 'test': False, 'rank': True,
            'batch_size': 512, 'batch_size_val': 128,
            'num_samples': 10, 'dropout': 0.2,
            'epochs': 100, 'patience': 20,
            'lr': 0.01, 'weight_decay': 1e-5,
            'lambda_X': 0.1, 'lambda_Y': 1,
            'load_best': True, 'val': True,
            'lambda_step': 0.1, 'lambda_kl': 1,
            'lambda_reg': 1, 'lambda_action': 1,
            'lambda_hy': 1, 'beta_bound': -10,
            'remove_aux': False, 'entropy_reg': 1,
        },
    })
    OmegaConf.set_struct(config, False)
    return config


def load_vcip_model(seed):
    """Load a trained VCIP model for given seed."""
    from src.models.vae_model import VAEModel
    from src.data.glucose import GlucoseDatasetCollection
    from src.utils.utils import to_float, repeat_static

    config = build_config(seed)

    dataset_collection = GlucoseDatasetCollection(
        data_dir=DATA_DIR, gamma=GAMMA, seed=seed,
        projection_horizon=5, treatment_mode='continuous'
    )
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    # Expand static features from (N, 1) to (N, T, 1) manually
    # Don't use repeat_static since test_f/test_cf share same object
    for subset in [dataset_collection.train_f, dataset_collection.val_f, dataset_collection.test_f]:
        sf = subset.data['static_features']
        if sf.ndim == 2:
            T = subset.data['outputs'].shape[1]
            subset.data['static_features'] = np.repeat(sf[:, np.newaxis, :], T, axis=1)
        # Also handle data_original if it exists
        if hasattr(subset, 'data_original') and 'static_features' in subset.data_original:
            sf_orig = subset.data_original['static_features']
            if sf_orig.ndim == 2:
                T_orig = subset.data_original['outputs'].shape[1]
                subset.data_original['static_features'] = np.repeat(sf_orig[:, np.newaxis, :], T_orig, axis=1)

    model = VAEModel(config, dataset_collection)

    model_path = os.path.join(MODEL_BASE, 'models', str(seed), 'model.ckpt')
    checkpoint = torch.load(model_path, weights_only=False)
    # PL checkpoint: state_dict is nested
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"  Model loaded from {model_path}")

    return model, dataset_collection, config


def compute_elbo_for_candidates(model, dataset_collection, cf_data, tau, config):
    """Compute ELBO for each counterfactual candidate using VCIP model."""
    from src.data.cip_dataset import CIPDataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.inference_model.to(device)
    model.generative_model.to(device)
    model.auxiliary_model.to(device)
    model.inference_model.eval()
    model.generative_model.eval()
    model.auxiliary_model.eval()

    original_tau = model.tau
    model.tau = tau
    model.config.exp.tau = tau

    data = dataset_collection.test_f.data
    dataloader = DataLoader(
        CIPDataset(data, config),
        batch_size=1, shuffle=False
    )

    all_sequences = cf_data['all_sequences']  # (N, k, tau) or (N, k, tau, 1)
    if all_sequences.ndim == 3:
        all_sequences = all_sequences[..., np.newaxis]  # add a_dim=1
    N, k, _, a_dim = all_sequences.shape

    elbo_losses = np.zeros((N, k))

    for i, batch in enumerate(dataloader):
        if i >= N:
            break

        H_t, targets = batch
        for key in H_t:
            H_t[key] = H_t[key].to(device)
        for key in targets:
            targets[key] = targets[key].to(device)
        Y_targets = targets['outputs']
        X_targets = None

        with torch.no_grad():
            for j in range(k):
                seq = torch.tensor(all_sequences[i, j]).unsqueeze(0).float().to(device)

                try:
                    elbo, _, _ = model.calculate_elbo(
                        H_t, Y_targets, seq,
                        X_targets=X_targets,
                        num_samples=config.exp.num_samples,
                        optimize_a=False
                    )
                    elbo_losses[i, j] = elbo.item()
                except Exception as e:
                    if j == 0 and i == 0:
                        print(f"    ELBO error patient {i}, candidate {j}: {e}")
                    elbo_losses[i, j] = np.inf

        if (i + 1) % 25 == 0:
            print(f"    Processed {i+1}/{N} patients")

    model.tau = original_tau
    model.config.exp.tau = original_tau
    return elbo_losses


def evaluate_ra(true_traj, elbo_losses):
    """Apply RA-constrained selection and compute metrics."""
    N, k, _ = true_traj.shape

    terminal = true_traj[:, :, -1]
    true_losses = (terminal - TARGET_MID) ** 2

    results = {
        'elbo_top1': [], 'elbo_safe': [], 'elbo_intgt': [], 'elbo_tl': [],
        'ra_top1': [], 'ra_safe': [], 'ra_intgt': [], 'ra_tl': [],
        'feasibility': [], 'spearman': [],
    }

    for i in range(N):
        ml = elbo_losses[i]
        tl = true_losses[i]
        traj = true_traj[i]

        terminal_bg = traj[:, -1]
        max_bg = traj.max(axis=1)
        min_bg = traj.min(axis=1)

        in_target = (terminal_bg >= TARGET_LO) & (terminal_bg <= TARGET_HI)
        safe = (max_bg <= SAFETY_HI) & (min_bg >= SAFETY_LO)
        feasible = in_target & safe

        feas_rate = feasible.sum() / k
        results['feasibility'].append(feas_rate)

        if np.all(np.isfinite(ml)):
            corr, _ = spearmanr(ml, tl)
            results['spearman'].append(corr)
        else:
            results['spearman'].append(np.nan)

        best_true_idx = np.argmin(tl)

        elbo_idx = np.argmin(ml)
        results['elbo_top1'].append(1.0 if elbo_idx == best_true_idx else 0.0)
        results['elbo_safe'].append(1.0 if safe[elbo_idx] else 0.0)
        results['elbo_intgt'].append(1.0 if in_target[elbo_idx] else 0.0)
        results['elbo_tl'].append(tl[elbo_idx])

        if feasible.any():
            ra_ml = ml.copy()
            ra_ml[~feasible] = np.inf
            ra_idx = np.argmin(ra_ml)
            results['ra_top1'].append(1.0 if ra_idx == best_true_idx else 0.0)
            results['ra_safe'].append(1.0 if safe[ra_idx] else 0.0)
            results['ra_intgt'].append(1.0 if in_target[ra_idx] else 0.0)
            results['ra_tl'].append(tl[ra_idx])
        else:
            results['ra_top1'].append(results['elbo_top1'][-1])
            results['ra_safe'].append(results['elbo_safe'][-1])
            results['ra_intgt'].append(results['elbo_intgt'][-1])
            results['ra_tl'].append(results['elbo_tl'][-1])

    return {k: np.array(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--all_seeds', action='store_true')
    parser.add_argument('--output_dir', type=str, default='/root/glucose_experiment/results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = SEEDS if args.all_seeds else ([args.seed] if args.seed else [10])
    all_seed_results = {}

    for seed in seeds:
        print("=" * 70)
        print(f"VCIP-based RA Evaluation — seed={seed}, gamma={GAMMA}")
        print("=" * 70)

        print(f"\nLoading VCIP model (seed={seed})...")
        model, dataset_collection, config = load_vcip_model(seed)

        seed_results = {}
        for tau in TAUS:
            cf_path = os.path.join(DATA_DIR, f'gamma_{GAMMA}', f'test_cf_tau{tau}.pkl')
            if not os.path.exists(cf_path):
                print(f"\n  tau={tau}: not found, skipping")
                continue

            print(f"\n  tau={tau}:")
            sys.path.insert(0, '/root/glucose_experiment')
            import generate_glucose_data
            sys.modules['__main__'].BergmanPatient = generate_glucose_data.BergmanPatient

            with open(cf_path, 'rb') as f:
                cf_data = pickle.load(f)

            true_traj = cf_data['true_glucose_trajectories']
            N, k, T = true_traj.shape
            print(f"    N={N}, k={k}, T={T}")

            print(f"    Computing ELBO losses...")
            elbo_losses = compute_elbo_for_candidates(
                model, dataset_collection, cf_data, tau, config
            )

            print(f"    RA evaluation...")
            results = evaluate_ra(true_traj, elbo_losses)
            results['elbo_losses'] = elbo_losses
            seed_results[tau] = results

            feas = 100 * results['feasibility'].mean()
            e_safe = 100 * results['elbo_safe'].mean()
            e_intgt = 100 * results['elbo_intgt'].mean()
            r_safe = 100 * results['ra_safe'].mean()
            r_intgt = 100 * results['ra_intgt'].mean()
            rho = np.nanmean(results['spearman'])
            imp = r_intgt - e_intgt
            print(f"    Feas={feas:.1f}%  ELBO: S={e_safe:.1f}% T={e_intgt:.1f}%  "
                  f"RA: S={r_safe:.1f}% T={r_intgt:.1f}%  +{imp:+.1f}pp  rho={rho:.3f}")

        all_seed_results[seed] = seed_results

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE (mean ± std across %d seeds)" % len(seeds))
    print("=" * 70)
    print("\n%-5s | %-7s | %-20s | %-20s | %-9s | %-7s" % (
        "tau", "Feas%", "ELBO (Safe/InTgt)", "RA (Safe/InTgt)", "RA +pp", "rho"))
    print("-" * 80)

    for tau in TAUS:
        v = {'f': [], 'es': [], 'ei': [], 'rs': [], 'ri': [], 'rho': []}
        for seed in seeds:
            if seed not in all_seed_results or tau not in all_seed_results[seed]:
                continue
            r = all_seed_results[seed][tau]
            v['f'].append(100 * r['feasibility'].mean())
            v['es'].append(100 * r['elbo_safe'].mean())
            v['ei'].append(100 * r['elbo_intgt'].mean())
            v['rs'].append(100 * r['ra_safe'].mean())
            v['ri'].append(100 * r['ra_intgt'].mean())
            v['rho'].append(np.nanmean(r['spearman']))

        if not v['f']:
            continue

        print("%-5d | %5.1f%% | %5.1f±%-3.1f/%5.1f±%-3.1f | %5.1f±%-3.1f/%5.1f±%-3.1f | %+5.1f pp | %.3f" % (
            tau, np.mean(v['f']),
            np.mean(v['es']), np.std(v['es']), np.mean(v['ei']), np.std(v['ei']),
            np.mean(v['rs']), np.std(v['rs']), np.mean(v['ri']), np.std(v['ri']),
            np.mean(v['ri']) - np.mean(v['ei']), np.mean(v['rho'])))

    save_path = os.path.join(args.output_dir, f'glucose_vcip_ra_gamma{GAMMA}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_seed_results, f)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
