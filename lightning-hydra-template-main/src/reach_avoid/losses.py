"""Loss components for the Reach-Avoid extension.

- Disentanglement loss (VCI-inspired): DKL[q(Z|a_obs) || q(Z|a_alt)]
- Weighted reconstruction loss: lambda_terminal * L_T + lambda_intermediate * L_I
"""

import torch


def _compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar):
    """KL divergence between two diagonal Gaussians: KL(q || p)."""
    return 0.5 * torch.sum(
        p_logvar - q_logvar - 1 + (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp()
    )


def compute_disentanglement_loss(inference_model, Z_s_inf, a_s_hidden,
                                  a_seq_hiddens, step_idx, H_rep, Y_last):
    """Compute VCI-inspired disentanglement regularizer at one timestep.

    Measures DKL[q(Z_s | a_obs) || q(Z_s | a_alt)] where a_alt is a random
    permutation of the batch's actions. Low values indicate latent Z captures
    patient state rather than treatment information.

    Args:
        inference_model: The inference network (has hidden_state, cell_state).
        Z_s_inf: Concatenated latent input, shape (batch, 2*z_dim).
        a_s_hidden: Action encoding at current step, shape (batch, hidden_dim).
        a_seq_hiddens: Full action encodings, shape (batch, tau, hidden_dim).
        step_idx: Current timestep index s.
        H_rep: History representation from auxiliary model.
        Y_last: Terminal outcome target, shape (batch, output_dim).

    Returns:
        disent_loss: Scalar KL divergence.
        q_mu: Mean of q under observed action (for downstream use).
        q_logvar: Log-variance of q under observed action.
    """
    # Save LSTM state before inference call
    saved_hidden = inference_model.hidden_state.clone() if inference_model.hidden_state is not None else None
    saved_cell = inference_model.cell_state.clone() if inference_model.cell_state is not None else None

    # Compute q under observed action
    q_mu, q_logvar = inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)

    # Restore LSTM state and compute q under alternative (permuted) action
    inference_model.hidden_state = saved_hidden
    inference_model.cell_state = saved_cell
    batch_size = a_s_hidden.size(0)
    perm = torch.randperm(batch_size, device=a_s_hidden.device)
    a_s_hidden_alt = a_seq_hiddens[perm, step_idx, :]
    q_mu_alt, q_logvar_alt = inference_model(Z_s_inf, a_s_hidden_alt, H_rep, Y_last)

    # KL divergence between observed and permuted posteriors
    disent_loss = _compute_kl_divergence(q_mu, q_logvar, q_mu_alt, q_logvar_alt)

    # Restore LSTM state and re-run with original action to get correct state propagation
    inference_model.hidden_state = saved_hidden
    inference_model.cell_state = saved_cell
    q_mu, q_logvar = inference_model(Z_s_inf, a_s_hidden, H_rep, Y_last)

    return disent_loss, q_mu, q_logvar


def compute_weighted_reg_loss(reg_losses, lambda_terminal=1.0, lambda_intermediate=0.0):
    """Compute weighted reconstruction loss combining terminal and intermediate steps.

    Args:
        reg_losses: List of per-step reconstruction losses.
        lambda_terminal: Weight for the terminal (last) step loss.
        lambda_intermediate: Weight for the mean of intermediate step losses.

    Returns:
        Weighted scalar loss.
    """
    terminal_loss = reg_losses[-1]
    if len(reg_losses) > 1 and lambda_intermediate > 0:
        intermediate_loss = torch.stack(reg_losses[:-1]).mean()
        return lambda_terminal * terminal_loss + lambda_intermediate * intermediate_loss
    return lambda_terminal * terminal_loss
