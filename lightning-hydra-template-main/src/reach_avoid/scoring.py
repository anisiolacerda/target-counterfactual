"""Reach-avoid scoring functions.

Differentiable soft indicators and composite reach-avoid score J_RA for
ranking counterfactual intervention sequences.
"""

import numpy as np
import torch


def soft_indicator_interval(y, lower, upper, kappa=10.0):
    """Differentiable approximation to 1{y in [lower, upper]}.

    Uses sigmoid gates: sigma(kappa*(y - lower)) * sigma(kappa*(upper - y)).
    Works element-wise; returns same shape as y.

    Args:
        y: np.ndarray or torch.Tensor
        lower, upper: scalar bounds
        kappa: hardness (higher = sharper boundary)
    """
    if isinstance(y, torch.Tensor):
        return torch.sigmoid(kappa * (y - lower)) * torch.sigmoid(kappa * (upper - y))
    else:
        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return _sigmoid(kappa * (y - lower)) * _sigmoid(kappa * (upper - y))


def soft_indicator_upper(y, upper, kappa=10.0):
    """Differentiable approximation to 1{y <= upper}."""
    if isinstance(y, torch.Tensor):
        return torch.sigmoid(kappa * (upper - y))
    else:
        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return _sigmoid(kappa * (upper - y))


def compute_reach_avoid_score(cancer_volume_trajectory, chemo_dosage_trajectory,
                               target_upper, safety_volume_upper,
                               safety_chemo_upper=None, kappa=10.0):
    """Compute reach-avoid score for cancer simulation trajectories.

    J_RA = log[ g_T(Y_{t+tau}) * prod_{s} g_S(Y_s) ]

    where g_T is the target soft indicator (tumor volume below threshold at terminal step)
    and g_S is the safety soft indicator (volume + chemo dosage within bounds at all steps).

    Args:
        cancer_volume_trajectory: shape (num_patients, tau+1), unscaled cancer volumes
            from current_t to current_t + tau (inclusive).
        chemo_dosage_trajectory: shape (num_patients, tau), cumulative chemo dosage
            from current_t to current_t + tau - 1.
        target_upper: upper bound for target set T at terminal step (cancer volume).
        safety_volume_upper: upper bound for safety set S (cancer volume at all steps).
        safety_chemo_upper: upper bound for chemo dosage safety (optional).
        kappa: sigmoid hardness parameter.

    Returns:
        ra_scores: shape (num_patients,) -- J_RA per patient (log-scale).
        ra_details: dict with per-step soft indicator values for diagnostics.
    """
    # Target: cancer volume at terminal step below threshold
    terminal_volume = cancer_volume_trajectory[:, -1]  # shape (N,)
    g_target = soft_indicator_upper(terminal_volume, target_upper, kappa)

    # Safety: cancer volume at ALL steps (including terminal) below threshold
    g_safety_volume = soft_indicator_upper(cancer_volume_trajectory, safety_volume_upper, kappa)
    # Product across time: shape (N,)
    g_safety_vol_prod = g_safety_volume.prod(axis=1) if isinstance(g_safety_volume, np.ndarray) \
        else g_safety_volume.prod(dim=1)

    # Safety: chemo dosage below toxicity limit at all projected steps
    if safety_chemo_upper is not None and chemo_dosage_trajectory is not None:
        g_safety_chemo = soft_indicator_upper(chemo_dosage_trajectory, safety_chemo_upper, kappa)
        g_safety_chemo_prod = g_safety_chemo.prod(axis=1) if isinstance(g_safety_chemo, np.ndarray) \
            else g_safety_chemo.prod(dim=1)
    else:
        g_safety_chemo_prod = 1.0

    # Combined: J_RA = log(g_T * prod g_S)
    combined = g_target * g_safety_vol_prod * g_safety_chemo_prod
    # Clamp to avoid log(0)
    if isinstance(combined, np.ndarray):
        ra_scores = np.log(np.clip(combined, 1e-30, None))
    else:
        ra_scores = torch.log(torch.clamp(combined, min=1e-30))

    ra_details = {
        'g_target': g_target,
        'g_safety_vol_prod': g_safety_vol_prod,
        'g_safety_chemo_prod': g_safety_chemo_prod,
        'combined': combined,
    }

    return ra_scores, ra_details
