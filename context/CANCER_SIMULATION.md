# Cancer Simulation Data Generation

Source: `src/data/cancer_sim_cont/cancer_simulation.py`
Based on Geng et al. (2017) pharmacokinetic-pharmacodynamic model for lung cancer.

## Tumor Growth Dynamics

```
V(t+1) = V(t) * (1 + rho*ln(K/V(t)) - beta_c*C(t) - (alpha*R(t) + beta*R(t)^2) + noise)
```

| Parameter | Meaning | Distribution |
|-----------|---------|-------------|
| V(t) | Tumor volume (cm^3) | Simulated |
| K | Carrying capacity | calc_volume(30) ~ 14,137 cm^3 |
| rho | Growth rate | N(7e-5, 7.23e-3^2), correlated with alpha (r=0.87) |
| alpha | Linear radio sensitivity | N(0.0398, 0.168^2) |
| beta | Quadratic radio sensitivity | alpha / 10 |
| beta_c | Chemo sensitivity | N(0.028, 0.0007^2) |
| C(t) | Chemo dosage | Exponential decay + new dose: C(t-1)*exp(-ln2/1) + 5*cs(a_chemo) |
| R(t) | Radio dosage | 2 * cs(a_radio) (no accumulation) |
| noise | Biological variability | 0.01 * N(0,1) |

Patient-specific parameters (alpha, beta_c, rho, K) are sampled per-patient.
3 patient types with different baseline treatment sensitivities.

## Confounding via Gamma

Treatment assignment depends on current tumor state:

```
P(treat | V(t)) = sigmoid(gamma/D_MAX * (D_mean - D_MAX/2))
```

Where D_mean = mean tumor diameter over last `window_size=15` days.

- **gamma=0**: Random treatment (no confounding)
- **gamma=1**: Weak confounding
- **gamma=4**: Strong confounding (larger tumors much more likely to be treated)

Treatment intensity sampled from Beta(2*P, 2-2*P):
- P=0.5 -> Beta(1,1) = Uniform
- P=0.9 -> Beta(1.8, 0.2) = concentrated near 1

## Data Splits

| Split | N patients | Purpose |
|-------|-----------|---------|
| train_f | 1000 | Training with biased treatment assignment |
| val_f | 100 | Validation |
| test_f | 100 | Test with 30% random treatment deviation |
| test_cf_one_step | 100 | One-step counterfactuals (3 alternatives per timestep) |
| test_cf_treatment_seq | 100 | Multi-step counterfactuals for tau-ahead evaluation |

## Counterfactual Generation

**One-step** (`simulate_counterfactual_1_step`):
- For each patient, timestep: apply 3 random alternative treatments, simulate 1 step

**Multi-step** (`simulate_counterfactuals_treatment_seq`):
- `cf_seq_mode='sliding_treatment'`: generates 2*projection_horizon random treatment sequences
- Each sequence: projection_horizon steps of continuous [chemo, radio] in [0,1]
- Simulates forward from factual history under counterfactual actions

**Ground truth evaluation** (`simulate_output_after_actions`):
- Takes a_seq (batch, tau, 2), applies to factual history
- Returns true outcome under intervention (oracle for evaluation)

## Projection Horizon and Tau

- `projection_horizon = 5` (from config)
- `tau = projection_horizon + 1 = 6` (default evaluation horizon)
- Paper evaluates tau in {1, 2, ..., 12} for Tables 1-2

## Preprocessing Pipeline

1. `process_data()`: Normalize with train stats, encode treatments, create time-aligned arrays
2. `process_data_encoder()`: Prepare one-step data for all splits
3. `process_data_decoder()`: Generate encoder representations, unroll to sequential examples
4. `process_sequential_test()`: Extract last projection_horizon timesteps for evaluation

Key arrays: current_covariates, prev_treatments, current_treatments, outputs, active_entries
