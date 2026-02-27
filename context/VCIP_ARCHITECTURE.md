# VCIP Model Architecture and Baselines

## VCIP Core Components

### 1. GenerativeModel (`src/models/generative_model.py`)
Models p_theta(Z_{s+1}|Z_s, a_s) and p_theta(Y|Z, a).

**Architecture:**
- `lstm_history`: LSTM(input_size -> z_dim) — encodes observational history
- `hidden_lstm`: LSTM(z_dim + treatment_size -> z_dim) — evolves Z under actions
- `action_encoder`: LSTM(treatment_size -> treatment_hidden_dim) — forward action encoding
- `reverse_action_encoder`: LSTM — backward action encoding (bidirectional)
- `fc_mu`, `fc_logvar`: MLP(hidden_dim + treatment_hidden_dim -> z_dim) — latent transition params
- `decoder_pa`: MLP(z_dim + treatment_hidden_dim -> 2*output_size) — outcome decoder
- `action_decoder_beta`: MLP(treatment_hidden_dim + z_dim -> 2*treatment_size) — Beta params for actions

**Default dims:** z_dim=16, hidden_dim=16, treatment_hidden_dim=16, num_layers=2

### 2. InferenceModel (`src/models/inference_model.py`)
Approximates q_phi(Z|H, Y, a).

**Architecture:**
- `lstm_history`: LSTM(input_size -> z_dim) — history to initial Z
- `lstm`: LSTM(2*z_dim + output_dim [+ treatment_hidden_dim] -> hidden_dim) — posterior inference
- `fc_mu`, `fc_logvar`: MLP(hidden_dim -> z_dim) — posterior params
- `predict_y_history_net`: MLP(z_dim + treatment_size -> output_size) — auxiliary task

### 3. AuxiliaryModel (`src/models/auxiliary_model.py`)
Learns observational p_obs(Y|H, A).

**Architecture:**
- `encoder`: LSTM(input_size -> hidden_dim) — history encoding
- `G_y`: MLP(hidden_dim + treatment_size -> output_size) — outcome prediction
- `G_x`: MLP(hidden_dim + treatment_size -> input_size) — optional covariate prediction
- Optional EMA for weight smoothing

### 4. DynamicParamNetwork (`src/models/dynamic_model.py`)
RBF-based parameterized dynamics for flexible state transitions.
- Multiquadric RBF basis with 5 centers
- Dynamically modulated weights based on action input

---

## Baseline Implementations

### RMSN (`src/baselines/rmsn.py`)
**Strategy:** Inverse propensity weighting (IPW)

**Components:**
- `PropensityNetworkTreatment`: LSTM -> P(A_t|A_{<t}), outputs Beta/Bernoulli density
- `PropensityNetworkHistory`: LSTM -> P(A_t|H_t), richer propensity model
- `Encoder`: LSTM, trained with stabilized IPW weights
- `Decoder`: LSTM, initialized from encoder states

**Confounding handling:** Propensity-based stabilized weights sw_tilde = prod P(A|A_{<t}) / P(A|H_t)

### CRN (`src/baselines/crn.py`)
**Strategy:** Balanced representations via gradient reversal

**Components:**
- `CRNEncoder`: VariationalLSTM + BRTreatmentOutcomeHead
  - br_size-dimensional balanced representation
  - Gradient reversal on treatment prediction branch
- `CRNDecoder`: Similar architecture, initialized from encoder

**Confounding handling:** Adversarial training makes representations treatment-invariant

### CT / Causal Transformer (`src/baselines/ct.py`, extends `src/baselines/edct.py`)
**Strategy:** Multi-input transformer with domain confusion

**Components:**
- Separate input transformations for treatments, vitals, outputs, static features
- TransformerMultiInputBlock with self-attention and cross-attention
- Relative positional encoding (max_relative_position=15)
- BRTreatmentOutcomeHead for balanced outputs

**Confounding handling:** Domain confusion balancing (alpha=0.01)

### G-Net (`src/baselines/gnet.py`)
**Strategy:** G-computation via outcome-conditional modeling

**Components:**
- `repr_net`: VariationalLSTM for representation learning
- `ROutcomeVitalsHead`: Joint outcome + vitals prediction
- Residual-based uncertainty via holdout set empirical residuals

**Confounding handling:** Models P(Y|do(A)) directly via g-computation

### ACTIN (trained via `runnables/train_actin.py`, config `baselines/actin.yaml`)
**Strategy:** GAN-based adversarial balancing with MINE

**Components:**
- TCN (Temporal Convolutional Network) with kernel_size=2-3, num_blocks=2
- Discriminator for adversarial representation balancing
- MINE-shuffle for mutual information minimization

**Confounding handling:** Distribution-based adversarial training (lambda_D=0.01)

---

## Common Patterns

| Model | Temporal backbone | Confounding strategy | Treatment modeling |
|-------|------------------|---------------------|-------------------|
| VCIP | LSTM + VAE | G-formula via lambda term | Beta/Bernoulli |
| RMSN | LSTM | IPW (propensity weighting) | Propensity networks |
| CRN | LSTM | Gradient reversal | Balanced repr. |
| CT | Transformer | Domain confusion | Balanced repr. |
| G-Net | LSTM | G-computation | Direct modeling |
| ACTIN | TCN | MINE adversarial | GAN discriminator |
