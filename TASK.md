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
- [ ] Formalize T2 theorem (ranking robustness proof) — written but needs review
- [ ] Final polish pass — check all cross-references, consistent notation, caption clarity

### Reviewer Defense — Pre-Submission (2026-03-26)

Addresses 12 weaknesses (W1–W12) from simulated NeurIPS reviewer analysis (Score 3, Borderline Reject). See PLANNING.md "Simulated NeurIPS Reviewer Analysis" section for full details. Strategy combines text edits with new experiments for maximum impact.

#### Priority A — Critical (text edits + key experiment)

- [ ] **A1: Reframe contribution (W1 — "just threshold filtering")** [TEXT]
  - [ ] A1.1: Add "deliberate simplicity" paragraph in Introduction (after Eq. 1, ~line 103)
  - [ ] A1.2: Position as first systematic safety evaluation of counterfactual planners
  - [ ] A1.3: Strengthen "guardrails" analogy in Discussion (cite RLHF rejection sampling)
  - [ ] A1.4: Add "first systematic safety evaluation" language in Abstract
- [ ] **A2: Strengthen theory framing (W2 — "trivial theorem")** [TEXT]
  - [ ] A2.1: Add honesty paragraph before Theorem 1 ("not technically deep, but makes explicit...")
  - [ ] A2.2: Reposition as design principle, not mathematical breakthrough
  - [ ] A2.3: Add remark connecting to data processing inequality / information-theoretic coarsening
- [ ] **A3: Oracle-vs-model experiment (W3 — "tautological Cancer results")** [NEW EXPERIMENT, ~3-5 days]
  - [ ] A3.1: Build `extract_predicted_cv()` for Cancer (parallel to MIMIC's `extract_predicted_dbp()` in `eval_mimic_traj.py:32-113`). Use `generative_model.py:476` (`decode_p_a`) inside the step loop at `reach_avoid/model.py:174-320`.
  - [ ] A3.2: Run RA filtering on model-predicted trajectories (decoder outputs) for 5 seeds × 4 gammas. No retraining needed — use existing checkpoints.
  - [ ] A3.3: Compare oracle-filtered vs model-predicted-filtered: safety rate, Top-1, feasibility, in-target. Report the gap.
  - [ ] A3.4: Add results to paper as new table/paragraph in Sec 6.1. Frame as 3-level validation: oracle → model-predicted synthetic → model-predicted real (MIMIC).
  - [ ] A3.5: Add oracle disclaimer text in Setup + "structural deficiency" paragraph in Discussion.

#### Priority B — High (new experiments + analysis)

- [ ] **B1: MIMIC calibration + correction rate (W4 — narrow DBP range)** [NEW ANALYSIS, ~1-2 days]
  - [ ] B1.1: Load existing pickles from `results_remote/mimic_ra/`. Report full predicted DBP distribution (mean, std, percentiles across all seeds/patients/sequences).
  - [ ] B1.2: Compare predicted DBP distribution against observed DBP in test set (calibration plot).
  - [ ] B1.3: Compute correction rate: fraction of out-of-target ELBO selections corrected by RA filtering.
  - [ ] B1.4: Analyze whether filter discriminates meaningfully or exploits boundary noise. Discuss narrow range explicitly in paper.
- [ ] **B3: Reframe E6 as constrained RL comparison (W6)** [TEXT + ANALYSIS, ~1 day]
  - [ ] B3.1: The soft-constraint Lagrangian (E6) IS the constrained RL analogue: `score = -ELBO + λ·safety_penalty` ≡ RCPO/CPO-Lagrangian. Reframe in Related Work + Sec 6.4.
  - [ ] B3.2: Add paragraph in Related Work distinguishing candidate selection from policy optimization.
- [ ] **B4: k-expansion experiment (W7 — feasibility collapse)** [NEW EXPERIMENT, ~2-4 hours]
  - [ ] B4.1: Re-run Cancer γ=4, τ=8 with k ∈ {100, 250, 500, 1000}. Change perturbation count in `optimize_interventions_discrete_onetime()`. No retraining.
  - [ ] B4.2: Report feasibility, constrained Top-1, safety rate for each k. Show feasibility scales with k.
  - [ ] B4.3: Add results as a table/figure in Appendix or Discussion.

#### Priority C — Medium (text + appendix)

- [ ] **C1: "Standard practice" argument for single simulator (W5)** [TEXT]
  - [ ] C1.1: Add paragraph in Limitations noting VCIP/CT/CRN precedent (all use only Cancer for ground truth). Cancer spans 400+ configs.
- [ ] **C2: Formalize ε_VI proxy (W8)** [APPENDIX]
  - [ ] C2.1: Write appendix section relating rank MAE → TV (DKW inequality or honest "practical diagnostic" framing)
- [ ] **C3: Fix sensitivity analysis claim (W9)** [CRITICAL TEXT FIX]
  - [ ] C3.1: Remove "partial sensitivity analysis for sequential ignorability" from Sec 6.1 (line 374) — **factually incorrect**
  - [ ] C3.2: Replace with "robustness to confounding strength" (gamma ≠ ignorability violation)
  - [ ] C3.3: Cite Kallus et al. 2019 or Frauen et al. 2023 for formal sensitivity analysis as future work

#### Lower Priority (address if time permits)

- [ ] **W10: Tighten "model-agnostic" language** [TEXT] — replace with "requires no retraining" where appropriate
- [ ] **W12: Fill related work gaps** [TEXT] — add chance-constrained optimization, robust MDP citations

#### Phase 2: Path to Score 5 (Accept) — ~3-6 weeks additional

**S5.1: Clinical evaluation on MIMIC** [NEW EXPERIMENT + EXTERNAL COLLABORATION]

- [ ] S5.1a: Clinician plausibility assessment
  - [ ] Recruit 1-2 ICU clinicians (intensivists)
  - [ ] Prepare ~30-50 blinded patient cases (ELBO-selected vs. RA-selected treatment sequences)
  - [ ] Collect 5-point Likert ratings (1=dangerous → 5=clinically appropriate)
  - [ ] Report inter-rater agreement (if 2 clinicians) and mean plausibility scores
- [ ] S5.1b: Outcome correlation analysis
  - [ ] Identify MIMIC patients with observed good outcomes (DBP stabilized in [60,90] within 24h, survived ICU)
  - [ ] Compare RA-selected treatment sequences against actual observed treatments
  - [ ] Report Jaccard similarity or treatment overlap
- [ ] S5.1c: Guideline concordance
  - [ ] Compare RA-recommended vasopressor/fluid patterns against Surviving Sepsis Campaign 2021 guidelines
  - [ ] Report concordance rate for vasopressor escalation/de-escalation
- [ ] S5.1d: Write clinical evaluation section for paper (Sec 6.2 extension or new Sec 6.3)
- Infrastructure: Existing MIMIC pickle data; need clinical collaborator(s) + IRB-exempt review
- Effort: ~2-3 weeks (mostly coordination; analysis is straightforward once ratings collected)
- Impact: Directly addresses RC-post-2 (MIMIC lacks clinical validation). Transforms MIMIC from feasibility demo to clinically validated.

**S5.2: Second evaluation domain with ground truth** [NEW EXPERIMENT + IMPLEMENTATION]

- [ ] S5.2a: Simulator selection and implementation
  - [ ] Evaluate candidates: (1) glucose-insulin (Hovorka 2004 / Bergman minimal model, `simglucose` package), (2) PK/PD 2-compartment model, (3) sepsis simulator
  - [ ] Implement chosen simulator with counterfactual outcome generation
  - [ ] Define target range (e.g., glucose ∈ [70,180] mg/dL) and safety bounds (e.g., glucose ∈ [50,250] mg/dL)
  - [ ] Generate training/test datasets with varying confounding strengths
- [ ] S5.2b: Model training and evaluation
  - [ ] Train VCIP + at least 2 baselines on generated data
  - [ ] Apply RA-constrained selection with same metrics (GRP, Top-1, safety, in-target, feasibility)
  - [ ] Run oracle-vs-model comparison (reusing A3 infrastructure)
- [ ] S5.2c: Write second simulator section for paper (Sec 6.3 or Appendix)
- Infrastructure: Simulator implementation + existing VCIP training pipeline; Vast.ai GPU time (~$10-20)
- Effort: ~2-3 weeks (1 week simulator + data, 1 week training, 1 week analysis + writing)
- Impact: Eliminates W5 entirely. Demonstrates cross-domain generalization. Strongest single improvement.

**S5.3: Stronger theoretical contribution** [NEW THEORY]

- [ ] S5.3a: Finite-sample constrained selection bound
  - [ ] Formalize: given n patients, k candidates, bound P(constrained selector picks truly-unsafe sequence)
  - [ ] Key ingredients: uniform convergence over candidate set + concentration inequality for feasibility indicator
  - [ ] Derive deployment guarantee: "with probability 1-δ, safe plan for ≥(1-α) fraction of future patients"
  - [ ] Verify bound is non-vacuous on Cancer data
- [ ] S5.3b: Connection to chance-constrained optimization
  - [ ] Frame Eq. 5 as empirical approx to: min ELBO(ā) s.t. Pr(E(ā)) ≥ 1-α
  - [ ] Derive sample complexity in k for empirical feasibility to approximate true chance constraint within ε
  - [ ] Connect to Nemirovski & Shapiro 2006 literature
- [ ] S5.3c: Regret bound for constrained selection
  - [ ] Define regret relative to oracle constrained selector
  - [ ] Bound expected regret in terms of ε_VI, k, and τ
  - [ ] Show regret vanishes as model quality improves or candidate pool grows
- [ ] S5.3d: Write strengthened theory section for paper (expand Sec 5)
- Effort: ~2-4 weeks (theory development + proof + empirical verification)
- Impact: Elevates paper from "empirical with modest theory" to "principled framework with deployment guarantees"
- **Note:** S5.3 is subsumed by S6.1 (conformal certificates). Skip as separate task; go directly to S6.1.

#### Phase 3: Score 5 → 6 (Strong Accept) — ~3-4 weeks

**S6.1: Conformal safety certificates** [NEW THEORY + EXPERIMENT] ★ PRIMARY CONTRIBUTION

- [ ] S6.1a: Conformal theory development
  - [ ] Define nonconformity score for reach-avoid: s_i = max(max_s |Y_s[ā] - boundary_S|₋, |Y_{t+τ}[ā] - boundary_T|₋)
  - [ ] Prove coverage theorem: P(Y_true[ā*] ∈ T ∩ S^τ) ≥ 1-α (distribution-free)
  - [ ] Address causal setting: weighted conformal with propensity reweighting (Lei & Candès 2021)
  - [ ] Write DRO interpretation remark (S6.3: TV ambiguity set connection)
- [ ] S6.1b: Cancer implementation
  - [ ] Split Cancer data: train/calibration/test
  - [ ] Compute nonconformity scores on calibration set
  - [ ] Implement conformal filter: F_conf = {ā : certified safe at level α}
  - [ ] Compare conformal-RA vs threshold-RA: coverage rate, selection quality, safety
  - [ ] Ablation: coverage level α ∈ {0.01, 0.05, 0.10, 0.20} vs. quality tradeoff
  - [ ] Ablation: calibration set size effect on coverage tightness
- [ ] S6.1c: MIMIC implementation
  - [ ] Implement conformal certificates using observed outcomes for calibration
  - [ ] Report coverage and selection quality
  - [ ] Compare with existing threshold-RA results
- [ ] S6.1d: Write conformal safety section for paper (new Section 3.2 + Theorem in Sec 4.2)
- Infrastructure: Existing experiment pipelines + conformal prediction library (or manual implementation ~200 lines)
- Key references: Lei & Candès 2021, Angelopoulos & Bates 2023, Vovk et al. 2005
- Effort: ~3-4 weeks (theory: 1 week, implementation: 1-2 weeks, experiments: 1 week)
- Impact: Addresses W1 (novelty), W2 (theory depth), W8 (replaces ε_VI proxy). First distribution-free safety certificates for counterfactual planning.

**S6.2: Systematic failure taxonomy** [NEW ANALYSIS]

- [ ] S6.2a: Identify dangerous ELBO recommendations
  - [ ] For all 5 models × 4 gammas × 4 horizons: flag patients where ELBO-optimal plan is unsafe
  - [ ] Compute danger rate per patient subgroup (by baseline tumor volume, treatment history, etc.)
- [ ] S6.2b: Cluster and classify failure modes
  - [ ] Use decision tree (CART) on patient features to predict danger rate
  - [ ] Classify into: "aggressive-harmful," "conservative-insufficient," "model-confused"
  - [ ] Show cross-model correlation (structural issue, not model-specific)
- [ ] S6.2c: Analyze RA correction per failure mode
  - [ ] For each mode: does RA filter correct the plan, or is feasible set empty?
  - [ ] Report correction rate by failure type
- [ ] S6.2d: Write failure taxonomy section for paper (Sec 5.4 or Appendix)
- Infrastructure: Uses existing Cancer experiment data (no new runs)
- Effort: ~1-2 weeks
- Impact: Transforms narrative from "we add safety" to "we discover systematic safety blind spot"

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

- [ ] **Week 1 (Mar 26 – Apr 2):** Phase 1 complete (A1-A3, B1, B3, B4, C1-C4). S6.1 theorem draft.
- [ ] **Week 2 (Apr 2 – Apr 9):** S6.1 working on Cancer/MIMIC. S5.2 simulator started. S6.2 failure taxonomy done.
- [ ] **Week 3 (Apr 9 – Apr 16):** S5.2 complete. S7.2 theoretical framework drafted.
- [ ] **Week 4 (Apr 16 – Apr 23):** S7.2 experiments complete. Full paper draft with all new sections.
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
