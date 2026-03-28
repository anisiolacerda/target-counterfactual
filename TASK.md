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
- [x] Replicate Table 1: long-range prediction, gamma=4, tau=1..12 (identical strategies)
- [ ] Replicate Table 2: long-range prediction, gamma=4, tau=1..12 (distinct strategies) — requires `exp.test=True`
- [x] Replicate Table 3: ablation study (with/without adjustment) — VCIP_ablation + RMSN_ab at gamma=4
- [x] Replicate Figure 4: GRP and RCS across models, gamma=4, tau=2,4,6,8
- [x] Replicate Figure 6: target distances across gamma=1,2,3, tau=1..6
- [x] Compare replicated numbers against paper (see `results_remote/REPLICATION_REPORT.md`)
- [x] Full Cancer replication: 5 models x 5 seeds x 4 gammas (220 runs, 52h on 4x RTX 2060)
- [x] Create Cancer analysis notebook: `results/cancer/analysis.ipynb`

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

### Cancer Replication Results (GRP, gamma=4, mean +/- std across 5 seeds)

| Model | tau=2 | tau=4 | tau=6 | tau=8 |
|-------|-------|-------|-------|-------|
| VCIP  | 0.932 +/- 0.008 | 0.973 +/- 0.005 | 0.991 +/- 0.002 | 0.994 +/- 0.002 |
| ACTIN | 0.684 +/- 0.183 | 0.676 +/- 0.250 | 0.675 +/- 0.307 | 0.676 +/- 0.323 |
| CT    | 0.536 +/- 0.198 | 0.544 +/- 0.205 | 0.532 +/- 0.181 | 0.521 +/- 0.207 |
| CRN   | 0.688 +/- 0.114 | 0.690 +/- 0.173 | 0.677 +/- 0.190 | 0.663 +/- 0.207 |
| RMSN  | 0.657 +/- 0.233 | 0.578 +/- 0.260 | 0.510 +/- 0.244 | 0.461 +/- 0.240 |

**Ablation (gamma=4):** VCIP_ablation GRP=0.738 (tau=2), 0.764 (tau=4) vs VCIP 0.932, 0.973 — confounding adjustment accounts for ~20 GRP points.

**All 3 paper claims verified** on Cancer data.

## Step 3 — Analyze weaknesses

- [x] Create analysis notebook: `results/weakness_analysis.ipynb`
- [x] Per-patient failure analysis: persistent failures, cross-seed rank stability
- [x] Confounding sensitivity: VCIP advantage grows with gamma (GRP drop: -0.248 at tau=2, -0.163 at tau=8); CRN most affected (-0.362 at tau=8); CT nearly unaffected
- [x] Horizon sensitivity: VCIP rank stability improves with tau (std: 7.0→0.8); slice discovery confirms advantage grows
- [x] Calibration analysis: Cancer ELBO-True Spearman rho=0.75-0.90 (5 seeds); Top-1 agreement 19% at tau=2, rising to 76% at tau=8
- [ ] Extended ablation: latent dimension, LSTM capacity, training schedule sensitivity — **running in parallel with Step 4 Phase 0-1** (6 runs on Vast.ai, ~$3-5). Results reviewed at Phase 1 decision gate. See PLANNING.md for configurations.
- [x] Summarize identified weaknesses as candidate research directions (6 weaknesses, 4 research directions)

### Key Weakness Findings

| ID | Weakness | Severity | Key Evidence |
|----|----------|----------|--------------|
| W1 | ELBO bound gap inconsistency | HIGH | Cancer ELBO-True rho=0.75-0.90 (5 seeds); Top-1 agreement 19% at tau=2 |
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

## Step 4 — Reach-Avoid Counterfactual Intervention Planning

**Direction chosen (2026-03-15):** Focused reach-avoid formulation (set targets + safety constraints), with T2 theory proving RA ranking is more robust to variational bound-gap variation than point-target ELBO ranking. See `PLANNING.md` for full rationale and `ideas/IDEA1.md` for the broader R²-VCIP sketch (confounding robustness deferred to future work).

### Phase 0 — Infrastructure (no retraining)
- [x] 0.1 Modify `simulate_output_after_actions()` to return full trajectory + dosages
- [x] 0.2 Implement `compute_reach_avoid_score()` with sigmoid soft indicators
- [x] 0.3 Extend `optimize_interventions_discrete_onetime()` with RA scoring path
- [x] 0.4 Calibrate T and S thresholds from data distributions

### Phase 1A — RA-as-ranker experiments (Cancer) ✓ COMPLETE (2026-03-19)

Ran on Vast.ai (2× RTX 3090). 5 seeds × 4 gammas × 4 taus. Results: `results_remote/phase1_ra_v2/`

- [x] E1 Ranking comparison: RA Top-1 ~1% (chance level) vs ELBO 2.8–75.8%. RA scoring alone cannot rank.
- [x] E2 Margin analysis: RA scores near-binary (pass/fail), no discrimination among candidates.
- [x] E3 Ranking stability (Cancer): RA rank std consistently higher than ELBO.

**Decision gate (2026-03-19): RA-as-ranker FAILS.** Overall ELBO Top-1=0.276 vs RA Top-1=0.020. Pivoted to Option 2: RA as constraint/filter (not ranker). See PLANNING.md for full rationale.

### Phase 1B — RA-Constrained ELBO Selection ✓ COMPLETE (2026-03-20)

Offline analysis (no GPU). Analysis notebook: `lightning-hydra-template-main/src/reach_avoid/analysis.ipynb`

- [x] E4 Constrained selection: `ā* = argmin_{ā ∈ F_RA} ELBO(ā)` — RA as feasibility filter, ELBO as ranker
- [x] Threshold sweep (9 configs) + per-gamma×tau breakdown (moderate: target≤0.6, vol≤5.0, chemo≤8.5)
- [x] Visualization: safety–quality Pareto frontier (`figures/e4_constrained_selection_tradeoff.png`)

**Key results:** γ=1,2: negligible effect (ELBO already safe). γ=3: +18pp safety for -6pp Top-1. γ=4: +33pp safety for -18pp Top-1. Most valuable under strong confounding where ELBO alone selects unsafe plans.

### Phase 1C — Heterogeneity Analysis (E5) ✓ COMPLETE (2026-03-20)

- [x] E5 Per-individual breakdown: 82/100 individuals are losers at γ=4 (>5pp Top-1 worse). Zero winners.
- [x] E5 Concentration analysis: Top 20% of losers account for 41% of total loss — near-uniform, NOT concentrated.
- [x] E5 Loser characterization: losers have lower feasibility (68.2%) than neutral (higher feasibility).

**Finding: Top-1 loss is broadly distributed (fundamental to hard constraint approach), not concentrated in a few hard individuals. Adaptive per-patient thresholds would NOT substantially help.**

**Decision (2026-03-20):** This rules out adaptive thresholds and favors the **soft constraint variant** (Lagrangian relaxation: weighted ELBO + RA penalty) as next direction, since it avoids the hard in/out boundary.

### Phase 1D — Soft Constraint / Lagrangian Relaxation (E6) ✓ COMPLETE (2026-03-20)

- [x] E6 Lambda sweep (11 values from 0 to 50) across all gammas
- [x] E6 Detailed per-gamma×tau breakdown at λ={0.1, 0.5, 2.0} vs hard filter
- [x] E6 Pareto frontier visualization

**Finding: Hard filter Pareto-dominates soft constraint at high safety levels.** At γ=4, ~86% safety: hard filter achieves 32.0% Top-1 vs soft (λ=50) 25.9% Top-1. The soft penalty shifts rankings globally (hurting all sequences), while hard filtering is surgical (preserves ELBO ranking among feasible set).

**Decision (2026-03-20): Hard filter is the preferred method** — simpler, more interpretable, and empirically superior. This is the method for the paper: `ā* = argmin_{ā ∈ F_RA} ELBO(ā)`.

### Phase 1E — MIMIC-III RA Evaluation (E7) — COMPLETE ✓

**Status:** Eval complete. Results downloaded. Analysis executed.

- [x] Write `eval_mimic_traj.py` — extracts predicted DBP trajectories from existing VCIP models
- [x] Write `run_mimic_ra.sh` — 2-GPU parallel execution script (5 seeds, ~2-4 hours)
- [x] Create Vast.ai execution plan (`context/MIMIC_RA_VASTAI_PLAN.md`)
- [x] Add E7 analysis cells to `analysis.ipynb` (constrained selection, visualization, stability, summary)
- [x] **Run on Vast.ai** — completed on `ssh -p 11299 root@70.69.192.6` (2x RTX 3090, ~15 min total for 5 seeds)
  - Full MIMIC-Extract pipeline: PostgreSQL → concepts → MIMIC-Extract → `all_hourly_data.h5` (7.3GB, 34,472 patients)
  - Eval: 5 seeds × 4 taus (2,3,4,5) × 100 individuals × 100 candidate sequences
  - DBP scaling: mean=60.93 mmHg, std=13.96 mmHg
  - Predicted DBP range: ~[52, 63] mmHg across individuals
- [x] Download results to `results_remote/mimic_ra/` (5 pickle files, ~2.8MB each)
- [x] Download MIMIC-Extract HDF5 data to `results_remote/mimic_extract/` (7.6GB total)
- [x] Execute E7 analysis cells — see results below

**Clinical thresholds:**
- Target: diastolic BP ∈ [60, 90] mmHg
- Safety: diastolic BP ∈ [40, 120] mmHg at all intermediate steps

**E7 Results (5 seeds pooled):**

| τ | Feasibility | ELBO in-target | Constrained in-target | ELBO DBP (mean) | Constrained DBP (mean) |
|---|-------------|----------------|-----------------------|-----------------|------------------------|
| 2 | 33.4% | 73.0% | 100.0% (+27.0pp) | 60.4 mmHg | 61.3 mmHg |
| 3 | 28.6% | 79.2% | 100.0% (+20.8pp) | 60.6 mmHg | 61.6 mmHg |
| 4 | 24.7% | 80.0% | 99.8% (+19.8pp) | 60.7 mmHg | 61.8 mmHg |
| 5 | 22.5% | 81.2% | 99.2% (+18.0pp) | 60.8 mmHg | 61.8 mmHg |

**Key findings:**
1. RA constraint lifts in-target rate from ~73-81% to ~99-100% across all horizons
2. Feasibility decreases with τ (33% → 22%) — longer horizons have fewer feasible plans
3. Constrained selection strongly reduces vasopressor use (ELBO vaso=0.29-0.79, Cstr vaso=0.01-0.29)
4. Cross-seed ELBO rank std decreases with τ (7.9 → 2.7), indicating more stable ranking at longer horizons
5. Predicted DBP distribution is narrow (~52-63 mmHg), clustering near the lower target boundary (60 mmHg)
6. Figure saved: `lightning-hydra-template-main/src/reach_avoid/figures/e7_mimic_constrained_selection.png`

### Phase 1 — Remaining items
- [x] E4: VCI consistency diagnostic (DKL[q(Z|a_obs) || q(Z|a_alt)]) — KL≈10⁻⁵ everywhere; latent space is action-invariant; action-outcome coupling lives in decoder, not latent space
- [x] RC6: Intermediate prediction quality — MSE=2-4×10⁻³ (model space), Spearman ρ≈0 at all steps; vanilla VCIP not calibrated for per-step ranking; motivates Phase 2's weighted intermediate loss

### Phase 2 — RA-aware retraining (Cancer, ~$5) ✓ COMPLETE

**Result: No meaningful improvement over vanilla VCIP.** RA-constrained selection is equally effective as a pure post-hoc filter — retraining is unnecessary.

- [x] 2.1 Weighted intermediate + terminal loss (λ_terminal=1.0, λ_intermediate=0.5) — implemented in ReachAvoidVAEModel
- [x] 2.3 Retrain: 5 seeds × gamma={1,4} = 10 runs on Vast.ai (~1.5h wall time, ~$3)
- [x] 2.4 Compare RA-retrained vs vanilla VCIP — see results below
- [x] 2.5 VCI-inspired disentanglement (λ_disent=0.1) — trained jointly with 2.1
- [~] 2.2 Calibration regularizer — deferred (underspecified, and Phase 2 results show retraining doesn't help)

**Phase 2 Results (VCIP_RA vs Vanilla VCIP, gamma=4):**

| τ | Vanilla GRP | RA GRP | Feas% | Cstr-safe (Van) | Cstr-safe (RA) | Cstr-Top1 (Van) | Cstr-Top1 (RA) |
|---|-------------|--------|-------|-----------------|----------------|-----------------|----------------|
| 2 | 0.922 | 0.924 | 53.6% | 94.4% | 94.4% | 17.4% | 18.0% |
| 4 | 0.963 | 0.963 | 29.7% | 95.6% | 95.6% | 39.4% | 38.6% |
| 6 | 0.981 | 0.982 | 16.8% | 96.2% | 96.2% | 54.2% | 55.0% |
| 8 | 0.984 | 0.984 | 10.3% | 97.0% | 97.0% | 60.4% | 60.4% |

**Key insight:** Feasibility rates, safety rates, and constrained Top-1 are *identical* between vanilla and RA-retrained. This is because RA filtering operates on ground-truth simulator trajectories (Cancer) — the model only determines ELBO ranking within the feasible set. Since VCIP's ELBO ranking is already excellent (GRP>0.92), retraining cannot improve it further.

**Implication for the paper:** RA-constrained selection is a pure post-hoc method requiring no retraining. This is a *strength*: simpler, cheaper, model-agnostic, and equally effective. Phase 2 validates this claim empirically.

### Phase 3 — Gradient-based RA planning (~$10-15)
- [ ] 3.1 Implement `optimize_reach_avoid()` (gradient optimization of J_RA)
- [ ] 3.2 Compare gradient RA planning vs perturbation-based ranking

### Phase 4 — Full experimental matrix (~$50-60)
- [ ] Cancer: vanilla baseline (existing) + RA scoring + RA-retrained + gradient planning
- [ ] MIMIC: vanilla baseline (existing) + RA scoring + RA-retrained

### Phase 5 — Ablations
- [x] 5.1 Target set size sensitivity — smooth trade-off: narrow T ↑safety/↓Top-1, moderate is sweet spot (54% feas, 9-23pp safety gain)
- [x] 5.2 Sigmoid hardness κ: {1, 5, 10, 50, 100} — κ≥5 indistinguishable from hard filter (>99.9% agreement), ε_soft≈0
- [x] 5.3 Reach-only vs reach-avoid — reach (target) is primary driver (+11pp in-target), avoid adds incremental safety at long horizons
- [ ] 5.4 Intermediate loss weight ratio
- [x] 5.5 Model-agnostic: RA scoring on ACTIN, CRN, CT, RMSN — baselines benefit MORE than VCIP (rank improvement 13-15 vs +2)
- [x] 5.6 **Midpoint-baseline** (RC1): Oracle midpoint ranker has WORSE Top-1 than ELBO (58% vs 76% at τ=8, γ=4); ignores intermediate safety. RA-constrained ELBO is strictly better as a practical method.
- [x] 5.7 **ε_VI estimation** (RC2): ε_VI≈0.09-0.23 (rank MAE), Spearman ρ=0.42-0.90. T2 bound is non-vacuous. ELBO pairwise preservation 61-87%.
- [ ] 5.8 **Latent disentanglement** (VCI-inspired): with/without DKL regularizer, varying λ_disent ∈ {0.01, 0.1, 0.5}
- [ ] 5.9 **CF consistency diagnostic validation**: correlate latent divergence with ground-truth CF prediction error on Cancer

### Reviewer-Driven Experiments (from VCIP OpenReview analysis)
- [ ] RC3: Document real-data evaluation limitations explicitly in paper (MIMIC has no ground-truth counterfactuals)
- [x] RC4: Gamma sweep sensitivity — RA benefit scales with confounding: γ=1 benign no-op, γ=4 +10.8pp in-target gain. Safety gain +15-21pp avg across all gammas. RA never hurts.
- [x] RC6: Intermediate prediction quality — MSE=2-4×10⁻³ (stable across steps), Spearman ρ≈0 (no cross-individual ranking power). Decoder is action-insensitive at intermediate steps.
- [x] RC7: Figure 1 concept visualization — shows ELBO picking unsafe-path sequence vs RA picking safe-path. PNG + PDF saved.

### Paper Writing
- [x] Write full paper draft (Introduction through Discussion & Conclusion) — 9 content pages
- [x] Figure 1: Reach-avoid concept visualization (rc7_figure1_concept.pdf)
- [x] Extended Related Work (Appendix H) — comprehensive survey with 25+ new references
- [x] Main-text Related Work — concise version citing key works across all areas
- [x] NeurIPS checklist — updated to current 16-question format with justifications
- [x] Formalize T2 theorem (ranking robustness proof) ✓ COMPLETE (2026-03-27) — Fixed Part (b): C_d now correctly defined as essential supremum of squared distance (was incorrectly defined as Var^{1/2}). Parts (a) and (c) verified correct.
- [x] Final polish pass ✓ COMPLETE (2026-03-27) — no undefined refs, notation macros consistent (fixed \mathcal→\cT/\cS in glucose appendix), proof corrected, all cross-refs valid
- [x] S6.2: Failure taxonomy appendix section ✓ COMPLETE (2026-03-27) — new Appendix "Failure Taxonomy" with danger rate table, failure mode classification (A/B/C), and per-patient heterogeneity analysis
- [x] RC3: Real-data evaluation limitations documented ✓ COMPLETE (2026-03-27) — MIMIC section now notes modest DBP shift (~1 mmHg) and connects to oracle-vs-model findings

### Reviewer Defense — Pre-Submission (2026-03-26)

Addresses 12 weaknesses (W1–W12) from simulated NeurIPS reviewer analysis (Score 3, Borderline Reject). See PLANNING.md "Simulated NeurIPS Reviewer Analysis" section for full details. Strategy combines text edits with new experiments for maximum impact.

#### Priority A — Critical (text edits + key experiment)

- [x] **A1: Reframe contribution (W1 — "just threshold filtering")** [TEXT] ✓ COMPLETE
  - [x] A1.1: "Deliberate simplicity" paragraph already in Introduction (lines 98-100)
  - [x] A1.2: "First systematic safety evaluation" in contributions list (line 105)
  - [x] A1.3: RLHF rejection sampling cite added to guardrails paragraph (Bai et al. 2022)
  - [x] A1.4: "First systematic safety evaluation" language added to Abstract (2026-03-27)
- [x] **A2: Strengthen theory framing (W2 — "trivial theorem")** [TEXT] ✓ COMPLETE
  - [x] A2.1: Honesty paragraph already present before Theorem 1 (line 240): "not mathematically deep...design principle"
  - [x] A2.2: Already repositioned as design principle (line 240)
  - [x] A2.3: Coarsening insight already in line 240: "set membership is a coarsening of continuous outcomes"
- [x] **A3: Oracle-vs-model experiment (W3 — "tautological Cancer results")** [NEW EXPERIMENT] ✓ COMPLETE (2026-03-27)
  - [x] A3.1: Built `extract_predicted_cv()` for Cancer. Script: `results_remote/a3/eval_cancer_oracle_vs_model.py`.
  - [x] A3.2: Ran RA filtering on model-predicted trajectories for 5 seeds, gamma=4, tau=2,4,6,8.
  - [x] A3.3: Compared oracle vs model-predicted. **Key finding: model-predicted feasibility near zero (0.1-0.2%)** due to high cancer volume scaling (std=53.176). Small prediction errors in scaled space → large errors in unscaled space, exceeding TARGET_UPPER=0.6. Oracle feasibility ~70%. This quantifies the oracle gap as a *structural deficiency* of current trajectory prediction quality.
  - [x] A3.4: Results already integrated in paper (lines 474-477, "Oracle vs. model-predicted trajectories" paragraph in Ablations)
  - [x] A3.5: Oracle disclaimer already in Limitations (line 497): "model-predicted trajectories are too compressed..."
  - **A3 Results (gamma=4, 5 seeds pooled):**
    | tau | ELBO Top1 | Oracle Top1 | Model Top1 | Oracle Feas | Model Feas | Model TruSafe |
    |-----|-----------|-------------|------------|-------------|------------|---------------|
    | 2   | 0.166     | 0.108       | 0.028      | 0.631       | 0.002      | 0.048         |
    | 4   | 0.430     | 0.274       | 0.046      | 0.689       | 0.001      | 0.076         |
    | 6   | 0.632     | 0.346       | 0.066      | 0.716       | 0.001      | 0.100         |
    | 8   | 0.758     | 0.450       | 0.062      | 0.710       | 0.001      | 0.100         |
  - **Data:** `results_remote/a3/a3_oracle_vs_model_gamma4.pkl` (11.2MB)

#### Priority B — High (new experiments + analysis)

- [x] **B1: MIMIC calibration + correction rate (W4 — narrow DBP range)** [NEW ANALYSIS] ✓ COMPLETE (2026-03-27)
  - [x] B1.1: DBP distribution: mean=57-58 mmHg, range [48-68], 100% of patients span the target boundary
  - [x] B1.2: Calibration: Pearson r≈0.32, MAE≈10 mmHg, near-zero bias
  - [x] B1.3: Correction rate: 95-100% of out-of-target ELBO selections corrected
  - [x] B1.4: Filter genuinely discriminates (all patients' predictions span target boundary). Results already in paper (lines 399-401).
  - Scripts: `results_remote/mimic_ra/b1_mimic_calibration.py`, `b1_deep_analysis.py`
- [x] **B3: Reframe E6 as constrained RL comparison (W6)** [TEXT] ✓ COMPLETE
  - [x] B3.1: Already reframed in paper (lines 430-435): "direct analogue of constrained policy optimization methods from safe RL (e.g., CPO, RCPO)"
  - [x] B3.2: Already in Related Work (lines 124-126): distinguishes filter-based approach from constrained policy optimization
- [x] **B4: k-expansion experiment (W7 — feasibility collapse)** [NEW EXPERIMENT] ✓ COMPLETE (2026-03-27)
  - [x] B4.1: Ran Cancer γ=4, τ=8 with k ∈ {100, 250, 500, 1000}, 5 seeds × 100 patients each.
  - [x] B4.2: Results show feasibility stable at ~70% (moderate thresholds), N_feasible scales linearly with k.
  - [x] B4.3: Add results as Table~\ref{tab:k_expansion} in Appendix, main text updated to reference it. ✓ COMPLETE (2026-03-27)
  - **B4 Results (gamma=4, tau=8, 5 seeds pooled):**
    | k    | Feasibility | N_feasible | Cstr Top-1 | In-Target | Safe  |
    |------|-------------|------------|------------|-----------|-------|
    | 100  | 70.0%       | 69.8       | 32.8%      | 90.2%     | 90.2% |
    | 250  | 70.0%       | 174.9      | 27.4%      | 90.6%     | 90.6% |
    | 500  | 70.0%       | 350.0      | 26.2%      | 90.6%     | 90.6% |
    | 1000 | 71.0%       | 710.3      | 26.0%      | 90.6%     | 90.6% |
  - **Key findings:** (1) Feasibility ~70% with moderate thresholds (not 10.3% — that was stricter thresholds). (2) N_feasible scales linearly. (3) Safety/in-target stable at ~90.6%. (4) Top-1 slightly decreases with k (32.8%→26.0%): ELBO ranking noisier over larger feasible pool. (5) Per-patient heterogeneity is large (some patients 0%, others 90%+).
  - **Data:** `results_remote/b4/b4_k_expansion_gamma4_tau8.pkl` (150KB)

#### Priority C — Medium (text + appendix)

- [x] **C1: "Standard practice" argument for single simulator (W5)** [TEXT] ✓ COMPLETE
  - [x] C1.1: Glucose simulator (S5.2) eliminates W5 entirely. Paper Limitations already mentions k-expansion scaling. W5 status: LOW.
- [x] **C2: Formalize ε_VI proxy (W8)** [APPENDIX] ✓ COMPLETE (2026-03-27)
  - [x] C2.1: New Appendix section "Justification of ε_VI Proxy" (Appendix~\ref{app:eps_proxy}). Connects rank MAE → Spearman footrule → Kendall tau via Diaconis-Graham inequality. Frames as conservative bound. Added Diaconis & Graham 1977, Sugiyama et al. 2012 citations.
- [x] **C3: Fix sensitivity analysis claim (W9)** [CRITICAL TEXT FIX] ✓ COMPLETE
  - [x] C3.1-3: Line 374 already corrected to "robustness to confounding strength" + Frauen et al. 2023 cite + future work mention

#### Lower Priority (address if time permits)

- [x] **W10: Tighten "model-agnostic" language** [TEXT] ✓ COMPLETE — paper uses "requires no retraining" (line 108)
- [x] **W12: Fill related work gaps** [TEXT] ✓ COMPLETE (2026-03-27)
  - [x] Added chance-constrained optimization (Nemirovski & Shapiro 2006, Calafiore & Campi 2006) and robust MDPs (Iyengar 2005, Nilim & El Ghaoui 2005) to both main Related Work and Extended Related Work appendix

#### Phase 2: Path to Score 5 (Accept) — ~3-6 weeks additional

**S5.1: Clinical evaluation on MIMIC** [NEW EXPERIMENT + EXTERNAL COLLABORATION]

- [ ] S5.1a: Clinician plausibility assessment
  - [ ] Recruit 1-2 ICU clinicians (intensivists)
  - [ ] Prepare ~30-50 blinded patient cases (ELBO-selected vs. RA-selected treatment sequences)
  - [ ] Collect 5-point Likert ratings (1=dangerous → 5=clinically appropriate)
  - [ ] Report inter-rater agreement (if 2 clinicians) and mean plausibility scores
- [x] S5.1b: Outcome correlation analysis ✓ COMPLETE (2026-03-27)
  - [x] Identify MIMIC patients with observed good outcomes (DBP stabilized in [60,90] within 24h)
  - [x] Compare RA-selected treatment sequences against actual observed treatments
  - [x] Report step agreement and L1 distance (good-outcome: 78-92% agreement, L1=0.05-0.13; bad-outcome: 74-83%, L1=0.12-0.17)
  - [x] Written as Appendix section (app:outcome_correlation) with Table tab:outcome_correlation
- [x] S5.1c: Guideline concordance ✓ COMPLETE (2026-03-27)
  - [x] Compare RA-recommended vasopressor/fluid patterns against Surviving Sepsis Campaign 2021 guidelines
  - [x] Report concordance by patient stratum: stable patients 97.2% no-vaso (SSC-concordant); low-DBP 5.8% strict / 100% pathway-aware concordance
  - [x] Key finding: model's inverse vaso-DBP relationship drives low strict concordance for hypotensive patients
  - [x] Written as subsection within Appendix app:outcome_correlation
- [x] S5.1d: Write clinical evaluation section for paper ✓ COMPLETE (2026-03-27)
  - [x] Outcome correlation + guideline concordance written as Appendix (app:outcome_correlation)
  - [x] Main MIMIC section cross-references appendix with key numbers
  - [x] Discussion/Limitations updated to cite outcome correlation and guideline concordance findings
  - [ ] S5.1a (clinician plausibility assessment) remains deferred — requires clinical collaborator
- Infrastructure: Existing MIMIC pickle data; clinician evaluation (S5.1a) needs clinical collaborator(s) + IRB-exempt review
- Impact: Partially addresses RC-post-2 (MIMIC lacks clinical validation). Automated concordance analysis provides indirect evidence; formal clinician evaluation would complete this.

**S5.2: Second evaluation domain with ground truth** [NEW EXPERIMENT + IMPLEMENTATION] ✓ **COMPLETE (2026-03-27)**

- [x] S5.2a: Simulator selection and implementation
  - [x] Evaluate candidates → selected Bergman minimal model (simplest, well-studied, sufficient for RA demonstration)
  - [x] Implement glucose-insulin simulator with Bergman 3-ODE dynamics + confounding mechanism (sicker patients → more insulin)
  - [x] Define target range: glucose ∈ [70,180] mg/dL. Safety bounds: glucose ∈ [50,250] mg/dL.
  - [x] Generate training/test datasets: 500 patients × 48 timesteps, γ=4 confounding strength
  - [x] Create Hydra config `glucose_sim.yaml` for VCIP training pipeline
  - [x] Patch `glucose.py` data loader with `process_sequential_multi()` method
- [x] S5.2b: Model training and evaluation
  - [x] Train VCIP: 5 seeds × 100 epochs on Vast.ai (~5 min/seed). Models saved.
  - [x] Custom eval script (`eval_glucose_vcip.py`) with manual OmegaConf config, `optimize_a=False` ELBO
  - [x] Apply RA-constrained selection: +0.8 to +2.0 pp in-target improvement, Spearman ρ 0.24–0.63
  - [ ] ~~Run oracle-vs-model comparison~~ — deferred (A3 infrastructure not yet built for glucose)
  - [ ] ~~Train baselines (CRN, CT)~~ — deferred (VCIP-only sufficient for appendix demonstration)
- [x] S5.2c: Write second simulator section for paper (Appendix, `\label{app:glucose}`)
  - [x] Bergman ODEs, confounding mechanism, constraint specification
  - [x] Results table (5 seeds × 4 taus with all metrics)
  - [x] Key findings paragraph + Discussion cross-reference
  - [x] `bergman1979quantitative` BibTeX entry added
- Results: `results_remote/glucose/glucose_vcip_ra_gamma4.pkl`
- Eval script: `results_remote/glucose/eval_glucose_vcip.py`
- Impact: Partially addresses W5 (second domain with ground truth). Cross-domain generalization demonstrated.

**S5.3: Stronger theoretical contribution** [NEW THEORY]

- [x] S5.3a: Finite-sample constrained selection bound ✓ COMPLETE (2026-03-27)
  - [x] Proposition 2 (Deployment Safety): Hoeffding-based bound on population safety rate
  - [x] Verified non-vacuous on Cancer: SR ≥ 80.8-86.2% with n=100, 95% confidence
- [x] S5.3b: Connection to chance-constrained optimization ✓ COMPLETE (2026-03-27)
  - [x] Framed as single-scenario approximation to population chance-constrained program
  - [x] Connected to Nemirovski & Shapiro 2006 and Calafiore & Campi 2006 scenario approach
  - [x] Written as subsection in Appendix app:finite_sample
- [x] S5.3c: Regret bound for constrained selection ✓ COMPLETE (2026-03-27)
  - [x] Proposition 3 (Quality Bound): regret ≤ 2η via uniform proxy error
  - [x] Empirically verified: mean regret < 0.2% of loss range at γ=4
  - [x] Spearman ρ = 0.71-0.89 within F_RA confirms ranking quality preserved
- [x] S5.3d: Write strengthened theory section for paper ✓ COMPLETE (2026-03-27)
  - [x] Main text: new subsection "Finite-Sample Deployment Guarantees" after Corollary 1
  - [x] Appendix: new section "Finite-Sample Analysis" with proofs, feasibility scaling, chance-constrained interpretation, empirical verification table
- Effort: ~2-4 weeks (theory development + proof + empirical verification)
- Impact: Elevates paper from "empirical with modest theory" to "principled framework with deployment guarantees"
- **Note:** S5.3 was originally subsumed by S6.1 (conformal certificates). Since S6.1 proved too conservative for practical use, S5.3 may be worth revisiting as an independent theoretical contribution (finite-sample bounds, chance-constrained optimization connection). Lower priority than remaining Phase 1 experiments (A3, B4, S6.2).

#### Phase 3: Score 5 → 6 (Strong Accept) — ~3-4 weeks

**S6.1: Conformal safety certificates** [NEW THEORY + EXPERIMENT] — **INVESTIGATED (2026-03-27), honest negative result**

- [x] S6.1a: Conformal theory development
  - [x] Define nonconformity score for reach-avoid: s_i = max(max_s |Y_s[ā] - boundary_S|₋, |Y_{t+τ}[ā] - boundary_T|₋)
  - [x] Prove coverage theorem: P(Y_true[ā*] ∈ T ∩ S^τ) ≥ 1-α (distribution-free)
  - [ ] Address causal setting: weighted conformal with propensity reweighting (Lei & Candès 2021) — deferred
  - [ ] Write DRO interpretation remark (S6.3: TV ambiguity set connection) — deferred
- [x] S6.1b: Cancer implementation
  - [x] Split Cancer data: train/calibration/test
  - [x] Compute nonconformity scores on calibration set
  - [x] Implement conformal filter: F_conf = {ā : certified safe at level α}
  - [x] Compare conformal-RA vs threshold-RA → **Result: conformal filter too conservative**
    - Calibrated quantile thresholds are extremely high due to model trajectory prediction variance
    - Near-zero feasibility at standard α levels (0.05, 0.10) — almost no candidates certified safe
    - The distribution-free guarantee is valid but impractical with current model quality
  - [ ] ~~Ablation: coverage level α~~ — moot given conservatism finding
  - [ ] ~~Ablation: calibration set size~~ — would not resolve fundamental issue (model variance)
- [ ] ~~S6.1c: MIMIC implementation~~ — deferred (Cancer result shows impracticality)
- [x] S6.1d: Write conformal safety section for paper → **written as Appendix** (honest negative result)
  - Documents the gap between distribution-free guarantees and practical utility
  - Identifies model trajectory quality as the bottleneck
  - Suggests conformal certificates become practical with improved models or larger calibration sets
- **Revised assessment:** S6.1 as originally envisioned (primary contribution replacing RA filter) is not viable. Instead, it contributes an informative appendix finding. The core RA threshold-based filter remains the practical method. Conformal certificates may be revived in Phase 4 if S7.2 (hidden confounding) provides a framework with less conservatism.
- Impact on score trajectory: S6.1 was expected to push to Score 6. Without it as primary contribution, need to recalibrate — S5.2 (glucose) + Phase 1 text edits + remaining experiments (A3, B4, S6.2) target Score 4-5 instead.

**S6.2: Systematic failure taxonomy** [NEW ANALYSIS] ✓ **PARTIALLY COMPLETE (2026-03-27)**

- [x] S6.2a: Identify dangerous ELBO recommendations ✓ COMPLETE
  - [x] Danger rate analysis across 4 gammas × 4 taus (VCIP model, 5 seeds)
  - [x] Per-patient heterogeneity: 54-65/100 patients "sometimes unsafe," none always unsafe
- [x] S6.2b: Classify failure modes ✓ COMPLETE
  - [x] Three modes: A (aggressive overshoot), B (conservative undershoot, dominant), C (toxic path, grows with τ)
  - [x] Mode B = 100% at τ=2, Mode C grows to 44% at τ=8
  - [ ] ~~Cross-model correlation~~ — deferred (requires running analysis on all 5 models)
  - [ ] ~~CART-based patient feature prediction~~ — deferred (additional analysis)
- [x] S6.2c: RA correction analysis → included in per-patient heterogeneity section
- [x] S6.2d: Write failure taxonomy appendix ✓ COMPLETE
  - [x] Appendix "Failure Taxonomy" with danger rate table, failure mode classification, per-patient heterogeneity
  - [x] Discussion paragraph cross-references appendix
- Impact: Partially transforms narrative; full cross-model analysis would strengthen further

#### Phase 4: Score 6 → 7 (Oral / Top 1-2%) — ~4-6 weeks additional

**S7.2: Safe planning under hidden confounding** [NEW THEORY + EXPERIMENT] ★ KEY CONTRIBUTION

- [ ] S7.2a: Γ-sensitivity model for sequential treatments
  - [ ] Formalize Rosenbaum Γ-sensitivity model for multi-step treatment sequences
  - [ ] Derive identified set for Y[ā] under bounded confounding Γ
  - [ ] Address compound sensitivity over τ steps (use Markov property or bound per-step)
- [ ] S7.2b: Γ-robust conformal certificate
  - [ ] Extend S6.1 coverage theorem: P(Y_true ∈ S | confounding ≤ Γ) ≥ 1-α with Γ-adjusted quantile
  - [ ] Prove the Γ-robust coverage theorem
  - [ ] Characterize how certificate tightness degrades as Γ increases
- [ ] S7.2c: Cancer experiments with artificial hidden confounding
  - [ ] Hold out a covariate from model training (simulate hidden confounder)
  - [ ] Apply Γ-robust filter at varying Γ levels
  - [ ] Verify coverage holds: empirical coverage ≥ 1-α across Γ values
  - [ ] Compare: standard conformal (S6.1) vs. Γ-robust conformal (S7.2) under hidden confounding
- [ ] S7.2d: Cancer gamma sweep validation
  - [ ] Use gamma parameter (confounding strength) as natural robustness test
  - [ ] Show Γ-robust certificates remain valid across gamma values
- [ ] S7.2e: MIMIC application
  - [ ] Apply Γ-robust recommendations at Γ ∈ {1.0, 1.5, 2.0, 3.0}
  - [ ] Report how recommendations shift as Γ increases (conservative shift expected)
  - [ ] Analyze clinical implications of robustness parameter choice
- [ ] S7.2f: Write hidden confounding section for paper (new Section 3.3 + Theorem in Sec 4.3)
- Key references: Rosenbaum 2002, Tan 2006, Kallus & Zhou 2021, Yadlowsky et al. 2022
- Effort: ~4-6 weeks (theory: 2-3 weeks, implementation: 1-2 weeks, experiments: 1 week)
- Risk: HIGH — compound sensitivity over τ steps is technically challenging. Mitigation: present small-τ results first, frame longer horizons as future work if needed.
- Impact: First distribution-free safety guarantee for counterfactual planning under hidden confounding. Addresses W9 substantively.

**S7.1: Minimax optimality bounds (secondary, if time permits)** [NEW THEORY]

- [ ] S7.1a: Lower bound construction
  - [ ] Construct adversarial instance showing no filter achieves safety > 1-Ω(ε_VI/√k)
  - [ ] Prove information-theoretic lower bound
- [ ] S7.1b: Upper bound (from S6.1)
  - [ ] Show conformal-RA achieves safety ≥ 1-O(ε_VI/√k + 1/√n_cal)
- [ ] S7.1c: Minimax optimality corollary
  - [ ] Conformal-RA is minimax optimal as n_cal → ∞
- Effort: ~4-6 weeks of hard theory. **Pursue only if S7.2 completes ahead of schedule.**
- Impact: Tight bounds = Oral-level theory

#### Phase 5: Score 7 → 8 (Best Paper, exploratory) — Week 5 only

**S8.1: "The Safety Tax" theorem (exploratory)** [NEW THEORY]

- [ ] S8.1a: Investigate fundamental quality-safety tradeoff
  - [ ] Can we prove: safety rate ≥ 1-δ requires quality regret ≥ Ω(f(ε_VI, δ, k, τ))?
  - [ ] Does conformal-RA achieve this bound?
  - [ ] If clean result: include as theorem. If not: frame as open question in Discussion.
- Effort: ~1 week exploration (Week 5)
- Risk: VERY HIGH — this is best-paper territory and may not yield a clean result
- Impact: If successful, defines a "price of safety" analogous to "price of fairness." Career paper.

#### Weekly Milestone Checklist (NeurIPS 2025 Submission)

- [x] **Week 1 (Mar 26 – Apr 2):** **ACTUAL:** S5.2 ✓ (glucose simulator). S6.1 ✓ (conformal — honest negative result). A3 ✓ (oracle-vs-model). B4 ✓ (k-expansion). A1 ✓ (reframe contribution). A2 ✓ (theory framing). A3.4-5 ✓ (paper integration). B1 ✓ (MIMIC calibration). B3 ✓ (constrained RL reframing). C1 ✓ (standard practice). C3 ✓ (sensitivity claim fix). C4/W10 ✓ (model-agnostic language). S6.2 ✓ (failure taxonomy + paper integration). **Phase 1 COMPLETE. All Vast.ai GPU experiments done. Instance shut down.**
- [x] **Week 2 (Mar 27):** **ACTUAL (completed same day as Week 1):** C2 ✓ (ε_VI proxy formalized as appendix with Diaconis-Graham inequality). W12 ✓ (chance-constrained optimization + robust MDP citations added). RC3 ✓ (real-data evaluation limitations documented). T2 ✓ (theorem proof reviewed, Part (b) C_d definition corrected). S6.2 ✓ (full failure taxonomy appendix section). Final polish ✓ (cross-refs, notation, captions). B4.3 ✓ (k-expansion table added to appendix). **All Phase 1 text + analysis tasks COMPLETE. Paper: 28 pages, clean compile.**
- [ ] **Week 3 (Apr 9 – Apr 16):** Remaining text edits (B3, C1, C2, C4, W10, W12) + S7.2 theory development.
- [ ] **Week 4 (Apr 16 – Apr 23):** S7.2 experiments + paper integration.
- [ ] **Week 5 (Apr 23 – Apr 30):** S8.1 exploration. S5.1 (if collaborator). Near-final paper.
- [ ] **Week 6 (Apr 30 – May 6):** Final polish, compile, submit.

---

## Discovered During Work

### From VCIP OpenReview Analysis (2026-03-15)

- VCIP was accepted as ICML 2025 poster (not NeurIPS as originally assumed). Venue: `ICML.cc/2025/Conference`.
- The range-target limitation was explicitly identified by Reviewer ixW5 and acknowledged by VCIP authors in their rebuttal as future work. This is our strongest motivation — cite directly.
- Reviewer g91d (initially weak reject) was most critical of theoretical rigor. Our T2 theorem must survive similar scrutiny — the bound must be empirically non-vacuous.
- The midpoint-baseline is the most dangerous simple objection. Must be addressed as a core ablation, not an afterthought.
- VCIP's ELBO ranking inconsistency (W1) was NOT raised by any ICML reviewer — the community may not find this as compelling on its own. It works best as supporting motivation for RA scoring, not as a standalone contribution.

### From VCI Paper Analysis (Wu et al., ICLR 2025) (2026-03-15)

- VCI derives ELBO for individual-level counterfactual likelihood p(Y'|Y,X,T,T') — fundamentally different from CVAE's p(Y|X,T). Static setting, high-dim outcomes (genes, images).
- **Key transferable principle**: DKL[q(z|y,t,x) || q(z|y',t',x)] → 0 provably disentangles Z from T (Lemma 1, Propositions 1-2). Adapted as latent disentanglement regularizer for VCIP (Idea A, ablation 5.8).
- **Counterfactual consistency diagnostic**: The latent divergence can serve as a model-intrinsic metric for CF prediction reliability — crucial for MIMIC evaluation where ground-truth CFs are unavailable (Idea C, experiment E4).
- **Intermediate supervision**: VCI's log[p̂(y'|x,t')] term motivates principled intermediate prediction supervision beyond simple loss weight increase (Idea B, task 2.1).
- VCI is ICLR 2025, not a direct competitor — different setting (static vs. longitudinal). Cite in Related Work as sharing the principle of counterfactual-aware variational inference.
- Full analysis: `.claude/plans/binary-wishing-whisper.md`
