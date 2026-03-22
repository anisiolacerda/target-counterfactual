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
- [ ] E4-original: VCI-style counterfactual consistency diagnostic (DKL[q(Z|a_obs) || q(Z|a_alt)]) — addresses RC3/W6

### Phase 2 — RA-aware retraining (Cancer, ~$15-20)
- [ ] 2.1 Fix `vae_model.py:511` — weighted intermediate + terminal loss (λ_terminal=1.0, λ_intermediate=0.5)
- [ ] 2.2 Add calibration regularizer for P̂(Y_s ∈ S)
- [ ] 2.3 Retrain: 5 seeds × gamma={1,4} = 10 runs
- [ ] 2.4 Compare RA-retrained vs vanilla VCIP
- [ ] 2.5 VCI-inspired latent disentanglement regularizer: DKL[q(Z_s|a_obs) || q(Z_s|a_alt)] penalty (Wu et al., ICLR 2025)

### Phase 3 — Gradient-based RA planning (~$10-15)
- [ ] 3.1 Implement `optimize_reach_avoid()` (gradient optimization of J_RA)
- [ ] 3.2 Compare gradient RA planning vs perturbation-based ranking

### Phase 4 — Full experimental matrix (~$50-60)
- [ ] Cancer: vanilla baseline (existing) + RA scoring + RA-retrained + gradient planning
- [ ] MIMIC: vanilla baseline (existing) + RA scoring + RA-retrained

### Phase 5 — Ablations
- [ ] 5.1 Target set size sensitivity
- [ ] 5.2 Sigmoid hardness κ: {1, 5, 10, 50, 100}
- [ ] 5.3 Reach-only vs reach-avoid
- [ ] 5.4 Intermediate loss weight ratio
- [ ] 5.5 Model-agnostic: RA scoring on ACTIN, CRN, CT, RMSN
- [ ] 5.6 **Midpoint-baseline** (RC1): VCIP with Y_target = midpoint(T) vs RA scoring — must demonstrate RA is strictly better
- [ ] 5.7 **ε_VI estimation** (RC2): compute TV distance proxy on Cancer data, verify T2 bound is non-vacuous
- [ ] 5.8 **Latent disentanglement** (VCI-inspired): with/without DKL regularizer, varying λ_disent ∈ {0.01, 0.1, 0.5}
- [ ] 5.9 **CF consistency diagnostic validation**: correlate latent divergence with ground-truth CF prediction error on Cancer

### Reviewer-Driven Experiments (from VCIP OpenReview analysis)
- [ ] RC3: Document real-data evaluation limitations explicitly in paper (MIMIC has no ground-truth counterfactuals)
- [ ] RC4: Present gamma={1,2,3,4} sweep as sensitivity analysis; show RA scoring robustness across gamma levels
- [ ] RC6: Report intermediate prediction quality Y_s at each step s on Cancer (model vs simulator ground truth)
- [ ] RC7: Create Figure 1 (visual reach-avoid concept: trajectory entering T while staying in S)

### Paper Writing
- [ ] Formalize T2 theorem (ranking robustness proof)
- [ ] Write Introduction — cite VCIP Reviewer ixW5's range-target concern + author acknowledgment as direct motivation
- [ ] Write Section 3 (ELBO ranking failure motivation)
- [ ] Write Method section (J_RA, soft indicators, RA-aware training) with intuitive interpretation before formal definition
- [ ] Write Theory section (T2 theorem + corollary + "what it means" paragraph + empirical ε_VI verification)
- [ ] Write Experiments section (include midpoint-baseline, intermediate quality, sensitivity analysis)
- [ ] Write Discussion + Conclusion (explicit limitations: T/S specification, real-data evaluation, assumptions)

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
