# Task Tracking

## Step 1 — Understand the VCIP codebase

- [x] Map each paper result (Tables 1-3, Figures 3-6) to specific config + script combinations (see `context/EXPERIMENT_MAP.md`)
- [x] Document the causal model (Theorem 4.1, ELBO derivation, g-formula connection) (see `context/VCIP_THEORY.md`)
- [x] Document model architecture: generative model, inference model, auxiliary model, dynamic model (see `context/VCIP_ARCHITECTURE.md`)
- [x] Document baseline implementations: RMSN, CRN, CT, G-Net, ACTIN (see `context/VCIP_ARCHITECTURE.md`)
- [x] Document the cancer simulation data generation process and parameters (see `context/CANCER_SIMULATION.md`)

## Step 2 — Replicate results

- [x] Set up `vcip` conda environment from `requirements_vcip.txt` (using `target-counterfactual-env`)
- [x] Set up `baseline` conda environment from `requirements_ct.txt` (using `baseline-env`)
- [x] Run smoke test: single gamma, single seed, VCIP only (2 epochs, gamma=4, seed=10, MPS device)
- [ ] Replicate Table 1: long-range prediction, gamma=4, tau=1..12 (identical strategies)
- [ ] Replicate Table 2: long-range prediction, gamma=4, tau=1..12 (distinct strategies)
- [ ] Replicate Table 3: ablation study (with/without adjustment)
- [ ] Replicate Figure 4: GRP and RCS across models, gamma=4, tau=2,4,6,8
- [ ] Replicate Figure 6: target distances across gamma=1,2,3, tau=1..6
- [ ] Compare replicated numbers against paper (document discrepancies)

## Step 2b — MIMIC-III experiments (feature/mimic-iii-experiments branch) ✓

- [x] Create feature branch and documentation (data setup, experiment map)
- [x] Fix ACTIN hparam filename mismatch (renamed to `diastolic_blood_pressure.yaml`)
- [x] Update training scripts to use 5 seeds
- [x] Create results analysis notebook (`results/mimic/analysis.ipynb`)
- [x] Obtain and preprocess MIMIC-III data (MIMIC-Extract → `all_hourly_data.h5`, 34,472 patients)
- [x] Fix `remove_aux` missing config key in `mimic3_real.yaml`
- [x] Smoke test: VCIP on MIMIC, 5 epochs, single seed (GRP rank 1.58/100)
- [x] Full VCIP run: 100 epochs, 5 seeds (~17 min/seed on Vast.ai)
- [x] Baseline runs: CRN, CT, RMSN, ACTIN (5 seeds each, ~19.5 hours total)
- [x] Download results to local machine (144 MB, 25 pickle files)
- [x] Run analysis notebook with real experiment outputs — all 3 paper claims verified
- [x] Slice discovery: GRP by treatment pattern (vaso/vent/both/neither) — VCIP advantage smallest for "Both" at tau=2 (0.160), largest for "Vent-only" at tau=8 (0.476)

### MIMIC-III Replication Results (GRP, mean +/- std across 5 seeds)

| Model | tau=2 | tau=4 | tau=6 | tau=8 |
|-------|-------|-------|-------|-------|
| VCIP  | 0.876 +/- 0.006 | 0.955 +/- 0.005 | 0.979 +/- 0.001 | 0.992 +/- 0.001 |
| ACTIN | 0.602 +/- 0.028 | 0.532 +/- 0.049 | 0.496 +/- 0.042 | 0.481 +/- 0.075 |
| CT    | 0.635 +/- 0.036 | 0.568 +/- 0.058 | 0.516 +/- 0.051 | 0.504 +/- 0.043 |
| CRN   | 0.655 +/- 0.034 | 0.595 +/- 0.045 | 0.525 +/- 0.047 | 0.508 +/- 0.048 |
| RMSN  | 0.614 +/- 0.062 | 0.536 +/- 0.060 | 0.506 +/- 0.056 | 0.510 +/- 0.065 |

**Paper claims verified:** (1) VCIP outperforms all baselines at every tau, (2) baselines degrade with larger tau, (3) VCIP improves with larger tau. RCS not applicable on real-world data (true counterfactual outcomes unobservable).

## Step 3 — Analyze weaknesses

- [x] Create analysis notebook: `results/weakness_analysis.ipynb`
- [x] Per-patient failure analysis: persistent failures, cross-seed rank stability
- [ ] Confounding sensitivity: characterize performance degradation across gamma values (requires full Cancer replication)
- [x] Horizon sensitivity: VCIP rank stability improves with tau (std: 7.0→0.8); slice discovery confirms advantage grows
- [x] Calibration analysis: Cancer ELBO-True Spearman rho=0.82-0.88; Top-1 agreement only 19% at tau=2, rising to 74% at tau=8
- [ ] Extended ablation: latent dimension, LSTM capacity, training schedule sensitivity (requires compute)
- [x] Summarize identified weaknesses as candidate research directions (6 weaknesses, 4 research directions)

### Key Weakness Findings

| ID | Weakness | Severity | Key Evidence |
|----|----------|----------|--------------|
| W1 | ELBO bound gap inconsistency | HIGH | Cancer ELBO-True rho=0.82 (not 1.0); Top-1 agreement only 19% at tau=2 |
| W2 | Rank instability across seeds | MEDIUM | Rank std=7.0 at tau=2; 1% of individuals have std>20 |
| W3 | Single target outcome | HIGH | Theoretical: clinical decisions involve multiple outcomes |
| W4 | Perturbation-dependent evaluation | MEDIUM | GRP depends on k=100 random perturbation set |
| W5 | Treatment pattern sensitivity | LOW-MED | "Both" pattern advantage smallest (0.160 vs 0.218 for "Neither") |
| W6 | No counterfactual validation on real data | HIGH | MIMIC RCS all NaN; GRP ≠ intervention quality |

### Research Directions (ranked by novelty x impact)
1. Tighter variational bounds with uncertainty quantification (W1+W2)
2. Multi-outcome counterfactual intervention planning (W3)
3. Off-policy evaluation for real-world validation (W6)
4. Treatment-interaction-aware generative models (W5)

## Step 4 — Targeted improvement (contingent on Step 3 findings)

- [ ] Formalize the identified weakness theoretically
- [ ] Propose and implement improvement method
- [ ] Run comparative experiments
- [ ] Write NeurIPS paper

---

## Discovered During Work

_(Add new sub-tasks or TODOs discovered during development here.)_
