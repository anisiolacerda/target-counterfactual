"""
MIMIC-III RA evaluation: re-run VCIP evaluation on existing models to save
predicted diastolic BP trajectories for offline RA-constrained selection.

This patches the existing optimize_interventions_discrete_onetime() to also
extract predicted outcome trajectories at each step of the ELBO loop.

Usage (on Vast.ai, from VCIP root):
    source /root/vcip_env/bin/activate
    cd /root/VCIP
    CUDA_VISIBLE_DEVICES=0 python scripts/mimic_ra/eval_mimic_traj.py \
        +dataset=mimic3_real +model=VCIP \
        exp.seed=10 exp.test=False exp.num_samples=10 \
        +exp.reach_avoid.save_traj=true

Outputs saved to: my_outputs/mimic_ra/VCIP/train/case_infos/{seed}/False/case_infos_VCIP.pkl
"""

import os
import sys
import pickle
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

# Ensure VCIP src is importable
VCIP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, VCIP_ROOT)


def extract_predicted_dbp(model, H_t, Y_targets, a_seq, X_targets=None, num_samples=10):
    """Run the ELBO forward pass and extract predicted diastolic BP at each step.

    Returns:
        elbo: float
        dbp_trajectory: np.array of shape (tau,) — predicted DBP (scaled) at each step
    """
    from src.utils.helper_functions import compute_kl_divergence

    H_rep = model.auxiliary_model.build_representations(H_t, only_last=True)
    Z_s_i, _ = model.inference_model.init_hidden_history(H_t)

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
    predicted_dbp = []

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
        predicted_dbp.append(float(y_mu.detach().cpu().flatten()[0]))

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
            predict_action_loss * model.config.exp.lambda_action)

    return elbo.item(), np.array(predicted_dbp)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate, get_original_cwd
    from src.utils.utils import set_seed, to_float, repeat_static
    from src.utils.helper_functions import generate_perturbed_sequences
    from src.data.cip_dataset import CIPDataset, get_dataloader
    import importlib

    OmegaConf.set_struct(cfg, False)
    seed = cfg.exp.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}, Seed: {seed}')

    # Load data (following train_vae.py pattern)
    original_cwd = get_original_cwd()
    cfg['exp']['processed_data_dir'] = os.path.join(original_cwd, cfg['exp']['processed_data_dir'])
    dataset_collection = instantiate(cfg.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if cfg['dataset']['static_size'] > 0:
        dims = len(dataset_collection.train_f.data['static_features'].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    # Build model with dataset_collection
    module_path, class_name = cfg["model"]["_target_"].rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class(cfg, dataset_collection)

    # Load checkpoint
    ckpt_path = os.path.join(
        original_cwd,
        f'my_outputs/mimic_real/VCIP/train/models/{seed}/model.ckpt'
    )
    print(f'Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Get scaling params (stored per-dataset, not on collection)
    scaling_params = model.dataset_collection.train_f.scaling_params
    dbp_mean = float(scaling_params['output_means'][0])
    dbp_std = float(scaling_params['output_stds'][0])
    print(f'DBP scaling: mean={dbp_mean:.2f}, std={dbp_std:.2f}')

    # Get validation data
    data = model.dataset_collection.val_f.data
    dataloader = get_dataloader(CIPDataset(data, cfg), batch_size=1, shuffle=False)

    k = 100
    results = {}

    # MIMIC has projection_horizon=4 → max tau=5
    for tau_val in [2, 3, 4, 5]:
        print(f'\n{"="*60}')
        print(f'tau={tau_val}, seed={seed}')
        print(f'{"="*60}')

        model.tau = tau_val
        model.config.exp.tau = tau_val
        set_seed(seed)

        case_infos = []
        for i, batch in enumerate(dataloader):
            if i > 99:
                break

            H_t, targets = batch
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            Y_targets = targets['outputs']
            X_targets = None if not model.predict_X else targets['vitals']
            true_actions = targets['current_treatments']

            # Slice true_actions to tau length for sequence generation
            true_actions_tau = true_actions[:, :tau_val, :]
            all_sequences = generate_perturbed_sequences(
                true_actions_tau, k, tau_val, model.a_dim, device,
                treatment_mode=cfg.dataset.treatment_mode)

            elbos = []
            dbp_trajectories = []

            # Slice Y_targets to match tau
            Y_targets_tau = Y_targets[:, :tau_val, :]

            with torch.no_grad():
                for seq in all_sequences:
                    elbo, dbp_traj = extract_predicted_dbp(
                        model, H_t, Y_targets_tau, seq, X_targets,
                        num_samples=cfg.exp.num_samples)
                    elbos.append(elbo)
                    dbp_trajectories.append(dbp_traj)

            model_losses = np.array(elbos)
            dbp_traj_arr = np.array(dbp_trajectories)  # (k, tau)

            # Unscale to mmHg
            dbp_unscaled = dbp_traj_arr * dbp_std + dbp_mean

            traj_features = {
                'dbp_terminal': dbp_unscaled[:, -1].astype(np.float32),     # (k,)
                'dbp_min': dbp_unscaled.min(axis=1).astype(np.float32),     # (k,)
                'dbp_max': dbp_unscaled.max(axis=1).astype(np.float32),     # (k,)
                'dbp_trajectory': dbp_unscaled.astype(np.float32),          # (k, tau)
            }

            # Treatment info
            seq_np = all_sequences.cpu().numpy()  # (k, 1, tau, 2)
            treatment_features = {
                'vaso_total': seq_np[:, 0, :, 0].sum(axis=1).astype(np.float32),
                'vent_total': seq_np[:, 0, :, 1].sum(axis=1).astype(np.float32),
                'sequences': seq_np[:, 0, :, :].astype(np.float32),  # (k, tau, 2)
            }

            # Observed DBP (at tau horizon)
            obs_dbp = float(Y_targets_tau[:, -1, :].cpu().numpy().flatten()[0] * dbp_std + dbp_mean)

            # True losses (constant for MIMIC - no ground truth simulator)
            true_losses = np.zeros(k, dtype=np.float32)

            case_info = {
                'individual_id': i,
                'model_losses': model_losses.astype(np.float32),
                'true_losses': true_losses,
                'true_sequence': true_actions.cpu().numpy(),
                'true_sequence_rank': int(np.sum(model_losses < model_losses[-1]) + 1),
                'traj_features': traj_features,
                'treatment_features': treatment_features,
                'observed_dbp': obs_dbp,
            }
            case_infos.append(case_info)

            if i % 25 == 0:
                dbp_t = traj_features['dbp_terminal']
                print(f'  [{i:3d}] rank={case_info["true_sequence_rank"]:3d}, '
                      f'obs_dbp={obs_dbp:.1f}, '
                      f'pred_dbp=[{dbp_t.min():.1f}, {dbp_t.max():.1f}] mmHg')

        results[tau_val] = case_infos

    # Save
    out_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        f'my_outputs/mimic_ra/VCIP/train/case_infos/{seed}/False'
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'case_infos_VCIP.pkl')

    with open(out_path, 'wb') as f:
        pickle.dump({'VCIP': results}, f)

    print(f'\nSaved: {out_path}')
    print(f'traj_features keys: {list(case_infos[0]["traj_features"].keys())}')
    print(f'DBP terminal range (last individual): '
          f'[{case_infos[-1]["traj_features"]["dbp_terminal"].min():.1f}, '
          f'{case_infos[-1]["traj_features"]["dbp_terminal"].max():.1f}] mmHg')


if __name__ == '__main__':
    main()
