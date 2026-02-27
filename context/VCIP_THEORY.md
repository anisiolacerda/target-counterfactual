# VCIP: Causal Model, ELBO, and G-Formula

## Causal Model (Paper Figure 2)

Temporal sequence: H_t -> Z_t -> Z_{t+1} -> ... -> Z_{t+tau} -> Y_{t+tau}
- Z_s: latent state at time s, encodes historical information
- a_s: intervention at time s, influences Z_{s+1}
- Y_{t+tau}: outcome decoded from final latent state

## ELBO Derivation (Paper Eq. 19)

The loss function implemented in `vae_model.py:calculate_elbo()`:

```
L_ELBO = sum_{s=t}^{t+tau} KL(q_phi(Z_s|.) || p_theta(Z_s|.))
         - E_q[log p_theta(Y_{t+tau} | Z_{t+tau})]
         + lambda * sum(E_q[log p_theta(a_s | Z_s)] - log p_theta(a_{t,tau} | H_t))
```

### Component mapping to code:

| Paper term | Code location | Implementation |
|---|---|---|
| KL(q \|\| p) | `helper_functions.py:compute_kl_divergence()` | Standard Gaussian KL between q_phi and p_theta |
| -log p(Y\|Z) | `generative_model.py:decoding_Y_loss_2()` | MSE between decoded Y and target |
| log p(a\|Z) | `generative_model.py:beta_loss()` / `bern_loss()` | Beta/Bernoulli NLL for action prediction |
| Auxiliary loss | `inference_model.py:init_hidden_history()` | History prediction from Z_t |

### Loss weights (from config):
- `lambda_reg=1`: Outcome reconstruction
- `lambda_kl=1`: KL divergence
- `lambda_step=0.1` (gamma-specific override): Step-level action loss
- `lambda_action=1`: Action prediction from initial state
- `lambda_hy=1`: Auxiliary history loss

### Ablation (Table 3):
Setting `lambda_step=0, lambda_action=0` removes the g-formula adjustment term,
equivalent to training without confounding awareness.

## G-Formula Connection (Theorem 4.1)

The g-formula bridge is implemented through:
1. **Observational likelihood**: `auxiliary_model.py` learns p_obs(Y|H,A) from data
2. **Interventional likelihood**: `generative_model.py` learns p_int(Y|Z,A) via VAE
3. **Bridge**: The lambda term in Eq. 19 aligns action distributions between observational and interventional regimes

When `lambda=1`, optimizing ELBO_2 (observational) approximates optimizing ELBO_1 (interventional) with bounded error epsilon_1 + epsilon_2.

## Intervention Optimization (Algorithm 1)

Implemented in `vae_model.py:optimize_interventions_onetime()`:

```
1. Initialize a_seq ~ Uniform(0,1) of shape (batch, tau, treatment_dim)
2. For M iterations:
   a. a_sigmoid = sigmoid(a_seq)          # Map to [0,1]
   b. ELBO = calculate_elbo(H, Y_target, a_sigmoid, optimize_a=True)
   c. loss.backward(); optimizer.step()   # Gradient descent on actions
3. Return best a_seq by ELBO
```

When `optimize_a=True`, only the final-timestep reconstruction loss is used (not intermediate steps), focusing optimization on achieving the target outcome.

## Key Implementation Details

- **Multiple sampling**: `inference_model.reparameterize_multiple(mu, logvar, num_samples=10)` creates ensemble of Z trajectories for Monte Carlo ELBO estimation
- **Bidirectional action encoding**: Both forward (`build_action_encoding`) and reverse (`build_reverse_action_encoding`) LSTM passes capture full temporal context of interventions
- **State evolution**: Autoregressive loop in `calculate_elbo()` updates Z_s at each timestep using mean of sampled posteriors
