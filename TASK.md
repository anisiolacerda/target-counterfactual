# Task Tracking

## Step 1 — Understand the VCIP codebase

- [x] Map each paper result (Tables 1-3, Figures 3-6) to specific config + script combinations (see `context/EXPERIMENT_MAP.md`)
- [x] Document the causal model (Theorem 4.1, ELBO derivation, g-formula connection) (see `context/VCIP_THEORY.md`)
- [x] Document model architecture: generative model, inference model, auxiliary model, dynamic model (see `context/VCIP_ARCHITECTURE.md`)
- [x] Document baseline implementations: RMSN, CRN, CT, G-Net, ACTIN (see `context/VCIP_ARCHITECTURE.md`)
- [x] Document the cancer simulation data generation process and parameters (see `context/CANCER_SIMULATION.md`)

## Step 2 — Replicate results

- [ ] Set up `vcip` conda environment from `requirements_vcip.txt`
- [ ] Set up `baseline` conda environment from `requirements_ct.txt`
- [ ] Run smoke test: single gamma, single seed, VCIP only
- [ ] Replicate Table 1: long-range prediction, gamma=4, tau=1..12 (identical strategies)
- [ ] Replicate Table 2: long-range prediction, gamma=4, tau=1..12 (distinct strategies)
- [ ] Replicate Table 3: ablation study (with/without adjustment)
- [ ] Replicate Figure 4: GRP and RCS across models, gamma=4, tau=2,4,6,8
- [ ] Replicate Figure 6: target distances across gamma=1,2,3, tau=1..6
- [ ] Compare replicated numbers against paper (document discrepancies)

## Step 3 — Analyze weaknesses

- [ ] Create analysis notebook: load and parse experiment results
- [ ] Per-patient failure analysis: identify trajectories where VCIP underperforms
- [ ] Confounding sensitivity: characterize performance degradation across gamma values
- [ ] Horizon sensitivity: check if improvement with larger tau is uniform across subgroups
- [ ] Calibration analysis: ELBO loss vs. true target distance correlation
- [ ] Extended ablation: latent dimension, LSTM capacity, training schedule sensitivity
- [ ] Summarize identified weaknesses as candidate research directions

## Step 4 — Targeted improvement (contingent on Step 3 findings)

- [ ] Formalize the identified weakness theoretically
- [ ] Propose and implement improvement method
- [ ] Run comparative experiments
- [ ] Write NeurIPS paper

---

## Discovered During Work

_(Add new sub-tasks or TODOs discovered during development here.)_
