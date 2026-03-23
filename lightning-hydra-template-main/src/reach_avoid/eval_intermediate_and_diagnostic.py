#!/usr/bin/env python3
"""RC6 + E4: Intermediate Prediction Quality & VCI Consistency Diagnostic.

Launched via Hydra (same config system as train_vae.py).

Usage (from VCIP root):
    python src/reach_avoid/eval_intermediate_and_diagnostic.py \
        +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=4*" \
        exp.seed=10 dataset.coeff=4 exp.rank=True exp.test=False
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
import importlib

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)

from src.utils.utils import set_seed, to_float, repeat_static
from src.data.cip_dataset import CIPDataset, get_dataloader
from src.models.vae_model import compute_kl_divergence
from src.utils.helper_functions import generate_perturbed_sequences


def get_absolute_path(relative_path):
    return os.path.abspath(relative_path)


def extract_intermediate_predictions(model, H_t, Y_targets, a_seq, num_samples=10):
    """Extract model's predicted Y_s at each step s."""
    H_rep = model.auxiliary_model.build_representations(H_t, only_last=True)
    Z_s_i, _ = model.inference_model.init_hidden_history(H_t)

    model.inference_model.reset_states()
    model.generative_model.reset_states()

    a_seq_hiddens = model.generative_model.build_reverse_action_encoding(a_seq)
    a_seq_hiddens_1 = model.generative_model.build_action_encoding(a_seq)

    Y_last = Y_targets[:, -1, :]
    hidden_states = model.generative_model.build_hidden_states(Z_s_i, a_seq)

    predictions = []
    per_step_losses = []

    for s in range(model.tau):
        a_s_hidden = a_seq_hiddens[:, s, :]
        Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)

        q_mu, q_logvar = model.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)
        Z_samples = model.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)

        # Get predicted Y_s from decoder (same encoding as calculate_elbo)
        a_enc = a_seq_hiddens_1[:, -1, :]
        if s > 0:
            a_enc_exp = a_enc.unsqueeze(0).expand(num_samples, -1, -1)
            mu, _ = model.generative_model.decode_p_a(Z_samples, a_enc_exp)
            pred_Y_s = mu.mean(dim=0)
        else:
            mu, _ = model.generative_model.decode_p_a(Z_samples.mean(dim=0), a_enc)
            pred_Y_s = mu

        predictions.append(pred_Y_s.detach().cpu().numpy().squeeze())

        if s < Y_targets.shape[1]:
            Y_gt = Y_targets[:, s, :].squeeze()
            step_mse = F.mse_loss(pred_Y_s.squeeze(), Y_gt).item()
            per_step_losses.append(step_mse)

        Z_s_i = Z_samples.mean(dim=0)

    return {
        'predictions': np.array(predictions),
        'per_step_losses': np.array(per_step_losses),
    }


def compute_vci_diagnostic(model, H_t, Y_targets, a_seq_observed,
                            a_seq_alternatives, num_samples=10):
    """Compute VCI-style latent divergence: DKL[q(Z_s|a_obs) || q(Z_s|a_alt)]."""
    H_rep = model.auxiliary_model.build_representations(H_t, only_last=True)
    Z_s_i_obs, _ = model.inference_model.init_hidden_history(H_t)
    model.inference_model.reset_states()
    model.generative_model.reset_states()

    a_seq_hiddens_obs = model.generative_model.build_reverse_action_encoding(a_seq_observed)
    Y_last = Y_targets[:, -1, :]

    obs_q_mus = []
    obs_q_logvars = []
    Z_s_i = Z_s_i_obs.clone()

    for s in range(model.tau):
        a_s_hidden = a_seq_hiddens_obs[:, s, :]
        Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
        q_mu, q_logvar = model.inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)
        obs_q_mus.append(q_mu.clone())
        obs_q_logvars.append(q_logvar.clone())
        Z_samples = model.inference_model.reparameterize_multiple(q_mu, q_logvar, num_samples)
        Z_s_i = Z_samples.mean(dim=0)

    all_kls = []
    for a_seq_alt in a_seq_alternatives:
        model.inference_model.reset_states()
        model.generative_model.reset_states()

        a_seq_hiddens_alt = model.generative_model.build_reverse_action_encoding(a_seq_alt)
        Z_s_i = Z_s_i_obs.clone()
        alt_kls = []

        for s in range(model.tau):
            a_s_hidden_alt = a_seq_hiddens_alt[:, s, :]
            Z_s_inf = torch.cat([Z_s_i, Z_s_i], dim=-1)
            q_mu_alt, q_logvar_alt = model.inference_model(
                Z_s_inf, a_s_hidden_alt, H_rep, Y_last)
            kl = compute_kl_divergence(
                obs_q_mus[s], obs_q_logvars[s],
                q_mu_alt, q_logvar_alt)
            alt_kls.append(kl.item())
            Z_samples = model.inference_model.reparameterize_multiple(
                q_mu_alt, q_logvar_alt, num_samples)
            Z_s_i = Z_samples.mean(dim=0)

        all_kls.append(alt_kls)

    all_kls = np.array(all_kls)
    return {
        'mean_kl': float(all_kls.mean()),
        'per_step_kl': all_kls.mean(axis=0).tolist(),
        'per_alt_kl': all_kls.mean(axis=1).tolist(),
    }


@hydra.main(config_name='config.yaml', config_path='../../configs/')
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    seed = args.exp.seed
    gamma = args.dataset.coeff
    set_seed(seed)

    print(f'\n{"="*60}')
    print(f'  RC6+E4: gamma={gamma}, seed={seed}')
    print(f'{"="*60}')

    # Data loading
    original_cwd = get_original_cwd()
    args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args['dataset']['static_size'] > 0:
        dims = len(dataset_collection.train_f.data['static_features'].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    # Model loading
    module_path, class_name = args["model"]["_target_"].rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class(args, dataset_collection)

    model_name = args.model.name.split('/')[0]
    model_dir = os.path.join(original_cwd, 'my_outputs', 'cancer_sim_cont', '22',
                             f'coeff_{gamma}', model_name, 'train', 'True',
                             'models', str(seed))
    model_path = os.path.join(model_dir, 'model.ckpt')

    if not os.path.exists(model_path):
        print(f'ERROR: Model not found: {model_path}')
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f'Loaded: {model_path}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Eval loop
    taus = [2, 4, 6, 8]
    n_individuals = 100
    k = 100
    n_rc6_seqs = 10
    n_diagnostic_alts = 20
    results = {}

    for tau in taus:
        print(f'\n  tau={tau}:')
        model.tau = tau
        model.config.exp.tau = tau

        set_seed(seed)
        data = dataset_collection.val_f.data
        dataloader = get_dataloader(CIPDataset(data, model.config),
                                    batch_size=1, shuffle=False)

        case_results = []

        for i, batch in enumerate(dataloader):
            if i >= n_individuals:
                break

            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            Y_targets = targets['outputs']
            true_actions = targets['current_treatments']

            all_sequences = generate_perturbed_sequences(
                true_actions, k, tau, model.a_dim, device,
                treatment_mode=model.config.dataset.treatment_mode)

            with torch.no_grad():
                # ── RC6: Intermediate predictions ──
                rc6_results = []
                for seq_idx in range(min(n_rc6_seqs, k)):
                    seq = all_sequences[seq_idx]
                    preds = extract_intermediate_predictions(
                        model, H_t, Y_targets, seq, num_samples=10)

                    # Ground-truth trajectory from simulator
                    traj_dict = dataset_collection.val_f.simulate_output_after_actions(
                        H_t, seq.detach(),
                        dataset_collection.train_scaling_params,
                        return_trajectory=True)

                    gt_traj = None
                    if isinstance(traj_dict, dict) and 'cancer_volume_trajectory' in traj_dict:
                        gt_traj = traj_dict['cancer_volume_trajectory'][0, 1:]  # skip t=0

                    rc6_results.append({
                        'predictions': preds['predictions'],
                        'per_step_losses': preds['per_step_losses'],
                        'gt_trajectory': gt_traj,
                    })

                # ── E4: VCI diagnostic ──
                alt_list = [all_sequences[j] for j in range(min(n_diagnostic_alts, k))]
                e4_result = compute_vci_diagnostic(
                    model, H_t, Y_targets, true_actions,
                    alt_list, num_samples=10)

                case_results.append({
                    'individual_id': i,
                    'rc6': rc6_results,
                    'e4': {
                        'mean_kl': e4_result['mean_kl'],
                        'per_step_kl': e4_result['per_step_kl'],
                    },
                })

            if (i + 1) % 20 == 0:
                avg_kl = np.mean([c['e4']['mean_kl'] for c in case_results])
                avg_loss = np.mean([
                    np.mean([np.mean(r['per_step_losses']) for r in c['rc6']])
                    for c in case_results])
                print(f'    {i+1}/{n_individuals}: avg KL={avg_kl:.4f}, avg MSE={avg_loss:.6f}')

        results[tau] = case_results
        print(f'  tau={tau} done: {len(case_results)} individuals')

    # Save results
    save_dir = os.path.join(original_cwd, 'my_outputs', 'rc6_e4')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'rc6_e4_gamma{gamma}_seed{seed}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved: {save_path}')

    # Summary
    print(f'\n{"="*60}')
    print(f'  SUMMARY: gamma={gamma}, seed={seed}')
    print(f'{"="*60}')
    for tau in taus:
        cr = results[tau]
        avg_kl = np.mean([c['e4']['mean_kl'] for c in cr])
        per_step_kls = np.mean([c['e4']['per_step_kl'] for c in cr], axis=0)
        avg_loss = np.mean([
            np.mean([np.mean(r['per_step_losses']) for r in c['rc6']])
            for c in cr])

        # RC6: per-step correlation between pred and GT
        if cr[0]['rc6'][0]['gt_trajectory'] is not None:
            step_corrs = []
            for s in range(tau):
                preds_s = [c['rc6'][0]['predictions'][s] for c in cr
                           if len(c['rc6'][0]['predictions']) > s]
                gts_s = [c['rc6'][0]['gt_trajectory'][s] for c in cr
                         if c['rc6'][0]['gt_trajectory'] is not None
                         and len(c['rc6'][0]['gt_trajectory']) > s]
                if len(preds_s) == len(gts_s) and len(preds_s) > 2:
                    rho, _ = spearmanr(preds_s, gts_s)
                    step_corrs.append(rho if not np.isnan(rho) else 0)
                else:
                    step_corrs.append(0)
            print(f'  tau={tau}: E4 KL={avg_kl:.4f}, '
                  f'per_step_KL={[f"{x:.3f}" for x in per_step_kls]}, '
                  f'RC6 MSE={avg_loss:.6f}, '
                  f'pred-GT corr={[f"{x:.3f}" for x in step_corrs]}')
        else:
            print(f'  tau={tau}: E4 KL={avg_kl:.4f}, '
                  f'per_step_KL={[f"{x:.3f}" for x in per_step_kls]}, '
                  f'RC6 MSE={avg_loss:.6f}')


if __name__ == '__main__':
    main()
