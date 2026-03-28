## Project Schedule

### Completed Steps

- Step 1: ✓ **COMPLETE.** Understand the VCIP's codebase located at `lightning-hydra-template-main/src/vendor/VCIP`.
- Step 2 (Cancer): ✓ **COMPLETE.** Full Cancer replication: 5 models × 5 seeds × 4 gammas (220 runs, 52h on 4× RTX 2060, ~$9.75). All 3 paper claims verified. VCIP GRP: 0.932→0.994 at gamma=4. Ablation confirms confounding adjustment worth ~20 GRP points. Analysis notebook: `VCIP/results/cancer/analysis.ipynb`.
- Step 2b (MIMIC): ✓ **COMPLETE.** Replicated VCIP results on MIMIC-III real ICU data. All 5 models × 5 seeds trained on Vast.ai (~19.5 hours). All 3 paper claims verified (VCIP GRP: 0.876→0.992 across tau=2..8; baselines degrade). Analysis notebook: `VCIP/results/mimic/analysis.ipynb`. See `context/MIMIC_DATA_SETUP.md` and `context/MIMIC_EXPERIMENT_MAP.md`.
- Step 3: ✓ **MOSTLY COMPLETE.** Weakness analysis notebook (`VCIP/results/weakness_analysis.ipynb`) identifies 6 weaknesses and 4 research directions. Key findings: (W1) ELBO-True correlation rho=0.75-0.90, Top-1 agreement only 19-76%; (W3) single target limitation; (W6) no counterfactual validation on real data. Confounding sensitivity analysis shows VCIP advantage grows with gamma. Remaining: extended ablation (latent dim, LSTM capacity).

---

### Step 4 — Reach-Avoid Counterfactual Intervention Planning (NeurIPS Submission)

#### Research Direction Decision (2026-03-15)

Two candidate extensions were evaluated against the experimental evidence:

| Candidate | Addresses | Evidence alignment | Verdict |
|---|---|---|---|
| **A: ELBO Ranking Consistency** (W1+W2) | ELBO gap inconsistency, seed instability | Strong evidence (19% Top-1 at tau=2), but the problem is mild at longer horizons and the fix (IWAE/ensemble) may yield only incremental improvement. Risk: too narrow for NeurIPS. |
| **B: R²-VCIP Reach-Avoid + Robustness** (W3+W6+confounding) | Single target, no real-data validation, confounding | Compelling vision, but confounding robustness is **partially contradicted** by our evidence (VCIP advantage *grows* with gamma). Scope too large (3 contributions bundled). |

**Chosen direction: Focused Reach-Avoid** — takes the strongest component from B (set targets + safety constraints) while dropping the confounding robustness as headline contribution. Incorporates the W1 insight from A into the theoretical motivation: reach-avoid scoring is *provably more robust* to variational bound-gap variation than point-target ELBO ranking.

#### Core Idea

VCIP asks: *"Which intervention sequence brings Y_{t+τ} closest to a target value?"*
Reach-avoid asks: *"Which intervention sequence has the highest probability of landing Y_{t+τ} inside a target range T, while keeping Y_s inside a safe region S at every intermediate step?"*

This is a single, clean conceptual upgrade — from point optimization to set-membership optimization with path constraints. The model architecture stays the same; the change is in the scoring/ranking objective.

#### Theoretical Contribution (T2: Ranking Robustness)

**Main theorem target:** Reach-avoid ranking is provably more robust to variational bound-gap variation than point-target ELBO ranking.

**Intuition:** ELBO-based ranking uses continuous loss — small differences in the variational gap can swap rankings. Reach-avoid uses set membership — a coarsening where many different outcome values map to the same verdict. Ranking inversions occur only when outcome distributions straddle the boundary of T or S, a geometrically smaller failure region.

**Formal statement (sketch):**

Assumptions: (i) $TV(p_θ(·|ā), q_φ(·|ā)) ≤ ε_VI$, (ii) soft indicator error $||g_T - 𝟙_T||_∞ ≤ ε_{soft}(κ)$.

- **(a) Ranking preservation:** RA ranking preserved whenever margin $m = |P(ā₁) - P(ā₂)| > 2ε_VI + 2τ·ε_{soft}(κ)$.
- **(b) Comparison:** Point-target ranking requires $|d(ā₁) - d(ā₂)| > 2ε_{VI}·C_d$, where C_d depends on outcome variance (typically large).
- **(c) Strict improvement:** When boundary mass is small (most outcomes clearly in/out of T), RA margin is large even when point-target margin is small.

**Proof strategy:** Part (a) via data processing inequality. Part (b) via Lipschitz analysis of continuous loss. Part (c) via interior/boundary mass decomposition.

**Empirical validation:** On Cancer ground truth, compare pairwise ranking inversion rates and margin distributions under ELBO vs. RA scoring.

#### Paper Contributions (4 bullets)

1. **Problem formulation:** Reach-avoid counterfactual intervention planning — plan interventions to reach a target set while maintaining safety constraints across the entire horizon.
2. **Method:** RA-constrained ELBO selection — use reach-avoid feasibility as a safety filter on ELBO-ranked candidates. Requires no retraining; threshold-tunable; compatible with any variational counterfactual planner.
3. **Theory:** Reach-avoid ranking is provably more robust to variational bound-gap variation than point-target ELBO ranking, with explicit approximation bounds (ε_soft from sigmoid hardness, ε_VI from variational gap).
4. **Experiments:** RA-constrained selection yields +18–33pp safety improvement under moderate-to-strong confounding (γ=3,4) at modest Top-1 cost (-6 to -18pp), on Cancer ground truth. MIMIC-III clinical plausibility validation.

**Note (2026-03-20):** Contribution 2 evolved from Phase 1A/1B experiments. Original plan was RA as a ranking replacement for ELBO. Phase 1A showed RA scoring alone cannot rank (near-binary scores, chance-level Top-1). Pivoted to RA as constraint/filter (Option 2), which leverages RA's strength in binary feasibility assessment while preserving ELBO's ranking quality. This is arguably a cleaner contribution: the method is simpler, requires no retraining, and the safety–quality trade-off is interpretable and tunable.

#### Target Set and Safety Set Definitions

| Dataset | Target set T | Safety set S |
|---|---|---|
| Cancer | tumor volume < calc_volume(5) at t+τ (diameter 5cm, stage II upper bound) | volume < calc_volume(10) at all intermediate steps AND chemo_dosage < toxicity_limit |
| MIMIC-III | diastolic BP ∈ [60, 90] at t+τ | O2 saturation > 0.88 at all intermediate steps |

#### Infrastructure Assessment

**Already available (no changes needed):**
- Cancer simulator produces `cancer_volume[i, t]` at every timestep and tracks `chemo_dosage[i, t]`
- MIMIC data has diastolic BP, O2 saturation, and 25+ covariates
- VCIP's ELBO loop (`vae_model.py:424`) iterates over all timesteps s = 0..τ-1
- Decoder can produce Y_s from Z_s at any step
- Trained models: 5 seeds × 4 gammas (Cancer), 5 seeds (MIMIC)

**Needs modification:**
- `simulate_output_after_actions()` (line 929) returns only final volume — must return full trajectory + dosages
- `optimize_interventions_discrete_onetime()` needs RA scoring path alongside ELBO
- Training loss currently discards intermediate reg_losses (line 511: `reg_loss = reg_losses[-1]`)

---

### Step 3 Extended Ablation (running in parallel with Phase 0-1)

**Decision (2026-03-15):** The extended ablation (latent dimension, LSTM capacity, training schedule sensitivity) was not completed in Step 3. This poses a low but non-zero risk: if W1 (ELBO inconsistency) is a capacity issue rather than a fundamental ELBO limitation, it weakens the RA scoring motivation. We run this ablation in parallel with Phase 0-1, using the Phase 1 decision gate as a natural checkpoint.

**Configurations (~6 runs on Vast.ai, ~$3-5):**

| Run | z_dim | hidden_dim | epochs | gamma | seed | Purpose |
|-----|-------|------------|--------|-------|------|---------|
| 1 | 32 | 32 | 100 | 4 | 10 | 2× latent capacity |
| 2 | 64 | 64 | 100 | 4 | 10 | 4× latent capacity |
| 3 | 16 | 32 | 100 | 4 | 10 | Wider LSTM, same latent |
| 4 | 16 | 16 | 200 | 4 | 10 | 2× training schedule |
| 5 | 32 | 32 | 100 | 1 | 10 | Capacity + low confounding |
| 6 | 64 | 64 | 100 | 1 | 10 | Max capacity + low confounding |

**Analysis:** Compare ELBO-True Spearman rho and Top-1 agreement against vanilla VCIP (z_dim=16, hidden_dim=16, 100 epochs). If rho improves substantially (>0.05) or Top-1 jumps (>10pp), W1 may be a capacity issue. If not, confirms W1 is fundamental — strengthens RA motivation.

**Checkpoint:** Results reviewed at Phase 1 decision gate. If W1 is capacity-driven, reassess RA scoring motivation (may need to reframe as complementary rather than corrective).

---

### Execution Plan

#### Phase 0 — Infrastructure (no retraining, ~1 day) ✓ COMPLETE

- [x] **0.1** Modify `simulate_output_after_actions()` in `cancer_simulation.py:929` to return full trajectory + dosages
- [x] **0.2** Implement `compute_reach_avoid_score()` with sigmoid soft indicators
- [x] **0.3** Extend `optimize_interventions_discrete_onetime()` in `vae_model.py:571` to save trajectory features (cv_terminal, cv_max, cd_max) per sequence for offline RA scoring
- [x] **0.4** Calibrate T and S thresholds from data distributions — discovered cancer volumes are in unscaled range 0–1150 (most <10); original thresholds were completely miscalibrated

#### Phase 1A — RA-as-ranker experiments (Cancer, ~$5) ✓ COMPLETE (2026-03-19)

Ran on Vast.ai (2× RTX 3090, ssh -p 40575 root@174.88.181.175). Evaluated 5 seeds × 4 gammas × 4 taus on Cancer ground truth. Results: `results_remote/phase1_ra_v2/` (20 pkl files, 64MB).

- [x] **E1 — Ranking comparison.** RA Top-1 at ~1% (chance level) across all configs. ELBO Top-1: 2.8–75.8% depending on gamma/tau. RA scoring alone cannot rank sequences.
- [x] **E2 — Margin analysis.** RA scores are near-binary (pass/fail) with little gradient — soft indicator product collapses to 0 or 1 for most sequences, providing no discrimination.
- [x] **E3 — Ranking stability (Cancer only).** RA rank std consistently higher than ELBO rank std across all gamma/tau.

**Analysis notebook:** `lightning-hydra-template-main/src/reach_avoid/analysis.ipynb` (+ `analysis_executed.ipynb`)

##### Phase 1A Decision Gate (2026-03-19)

**Result: RA-as-ranker FAILS.** RA scoring does not improve Top-1 agreement over ELBO. Overall: ELBO Top-1=0.276, RA Top-1=0.020, Δ=-0.256. Failure is fundamental: RA produces near-binary scores (set membership is too coarse for ranking among 100 candidates).

**Two options evaluated:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **1: RA in training objective** | Retrain VCIP with RA loss component | Could improve intermediate predictions | Requires GPU retraining; unclear if RA gradient signal helps when scores are near-binary |
| **2: RA as constraint/filter** | `ā* = argmin_{ā ∈ F_RA} ELBO(ā)` — use RA as feasibility filter, ELBO as ranker | No retraining needed; simple, interpretable; threshold-tunable | May reduce candidate pool too aggressively; Top-1 accuracy cost |

**Decision: Option 2 (RA as constraint/filter).** Rationale: RA's strength is in binary feasibility assessment (safe/unsafe), not continuous ranking. ELBO is already good at ranking. The natural combination leverages each method's strength.

#### Phase 1B — RA-Constrained ELBO Selection (E4) ✓ COMPLETE (2026-03-20)

Offline analysis using saved trajectory features. No GPU needed.

- [x] **E4 — Constrained selection sweep.** Evaluated 9 threshold configs (Loose/Moderate/Strict) on gamma=4, plus detailed per-gamma×tau breakdown with moderate thresholds (target≤0.6, vol≤5.0, chemo≤8.5).

##### Phase 1B Key Results

| γ | Top-1 Cost | Safety Gain | Feasibility | Loss Penalty | Verdict |
|---|-----------|------------|-------------|-------------|---------|
| 1 | +0.0% | +3.1pp | 87.3% | -0.000070 | No-op (ELBO already safe) |
| 2 | -1.2% | +7.8pp | 83.8% | +0.000235 | Marginal benefit |
| 3 | -6.3% | +18.3pp | 79.3% | +0.002132 | Good trade-off |
| 4 | -17.9% | +33.1pp | 69.1% | +0.006354 | Strong safety gain, notable quality cost |

**Key insight:** RA-constrained selection is most valuable under strong confounding (γ≥3), precisely where ELBO alone selects unsafe plans (only 50–75% safe). The constraint boosts safety to 85–92%. The method requires no retraining and is threshold-tunable.

**Figures:** `lightning-hydra-template-main/src/reach_avoid/figures/e4_constrained_selection_tradeoff.png`

##### Phase 1B Decision: Recommended Next Steps (2026-03-20)

Three candidate directions identified and evaluated:

#### Phase 1C — Heterogeneity Analysis (E5) ✓ COMPLETE (2026-03-20)

**Result:** Top-1 loss is broadly distributed (82/100 individuals are losers at γ=4), NOT concentrated. Top 20% of losers account for only 41% of total loss (near-uniform). Zero individuals benefit from constrained selection in Top-1.

**Implication:** Adaptive per-patient thresholds would NOT help. The cost is fundamental to hard filtering.

#### Phase 1D — Soft Constraint / Lagrangian Relaxation (E6) ✓ COMPLETE (2026-03-20)

Tested `score = ELBO + λ · RA_penalty` with λ ∈ {0, 0.01, ..., 50}.

**Result:** Hard filter Pareto-dominates soft constraint at high safety levels. At γ=4, ~86% safety: hard=32.0% Top-1 vs soft(λ=50)=25.9% Top-1. The soft penalty shifts rankings globally (hurting all sequences), while hard filtering is surgical (preserves ELBO ranking among feasible).

**Decision (2026-03-20): Hard filter is the preferred method for the paper.** Simpler, more interpretable, empirically superior. Method: `ā* = argmin_{ā ∈ F_RA} ELBO(ā)`.

#### Remaining

- MIMIC-III validation with clinically-motivated safety constraints
- VCI-style counterfactual consistency diagnostic (addresses RC3/W6)

#### Phase 1 — Zero-retraining experiments on existing models (~$5) — SUPERSEDED

*Original plan below retained for reference. Phase 1A/1B above replaced E1–E3. E4 from original plan (VCI counterfactual consistency diagnostic) not yet attempted.*

~~Using already-trained Cancer models (5 seeds × 4 gammas) and MIMIC models (5 seeds):~~

~~- [ ] **E1 — Ranking comparison (Cancer, ground truth).** ...~~
~~- [ ] **E2 — Margin analysis (Cancer).** ...~~
~~- [ ] **E3 — Ranking stability (MIMIC).** ...~~

~~**Decision gate after Phase 1:** If RA scoring does not substantially improve Top-1 agreement over ELBO on Cancer ground truth, reassess the approach before investing in retraining.~~

#### Phase 2 — RA-aware retraining (Cancer, ~$5) ✓ COMPLETE

**Result: No meaningful improvement.** RA-constrained selection is equally effective as a post-hoc filter on vanilla VCIP — retraining is unnecessary. All metrics (GRP, feasibility, constrained safety, constrained Top-1) are identical within noise (±0.002 GRP, ±1.4pp Top-1).

**Why:** On Cancer, RA filtering operates on ground-truth simulator trajectories. The model only determines ELBO ranking within the feasible set. Since VCIP's ELBO ranking is already excellent (GRP>0.92 at γ=4), retraining cannot improve constrained selection further.

**Implication for the paper:** RA-constrained selection is a pure post-hoc method. This is a strength: simpler, cheaper, model-agnostic. Phase 2 validates this claim empirically. Retraining may still matter for MIMIC (no simulator, model-predicted trajectories used for filtering).

- [x] **2.1** Weighted intermediate + terminal loss (λ_terminal=1.0, λ_intermediate=0.5) in ReachAvoidVAEModel
- [x] **2.3** Retrained: 5 seeds × gamma={1,4} = 10 runs (~1.5h wall time, ~$3 on Vast.ai)
- [x] **2.4** Compared RA-retrained vs vanilla: all metrics identical within noise
- [x] **2.5** Disentanglement (λ_disent=0.1) trained jointly — no effect on downstream RA selection
- [~] **2.2** Calibration regularizer — deferred (retraining shown unnecessary)

#### Phase 3 — Gradient-based RA planning (~$10-15)

- [ ] **3.1** Implement `optimize_reach_avoid()`: gradient optimization of J_RA w.r.t. intervention sequence ā (analogous to existing `optimize_interventions_onetime()`)
- [ ] **3.2** Compare gradient-based RA planning vs. perturbation-based ranking on Cancer ground truth
- [ ] **3.3** This also addresses W4 (perturbation-dependent evaluation) as a byproduct

#### Phase 4 — Full experimental matrix (~$50-60)

**Cancer (ground truth):**

| Experiment | Config | Runs | Est. cost |
|---|---|---|---|
| Vanilla VCIP baseline | 5 seeds × 4 gammas × 4 taus | 80 (existing) | $0 |
| RA scoring on vanilla VCIP | Same, re-evaluate | 80 | ~$5 |
| ~~RA-aware retrained VCIP~~ | ~~5 seeds × gamma={1,4} × 4 taus~~ | ~~40~~ | ~~$15~~ (Phase 2: no improvement) |
| Gradient-based RA planning | 5 seeds × gamma={1,4} | 10 | ~$5 |

**MIMIC-III:**

| Experiment | Config | Runs | Est. cost |
|---|---|---|---|
| Vanilla VCIP baseline | 5 seeds × 4 taus | 20 (existing) | $0 |
| RA scoring on vanilla VCIP | Same, re-evaluate | 20 | ~$3 |
| ~~RA-aware retrained VCIP~~ | ~~5 seeds × 4 taus~~ | ~~20~~ | ~~$15~~ (Phase 2: likely unnecessary, but may matter for MIMIC since RA filter uses model predictions) |

**Total estimated cost: ~$43-48** (remainder covers reruns, debugging, additional ablations)

#### Phase 5 — Ablations

- [ ] **5.1** Target set size: vary T width (narrow vs. broad clinical range)
- [ ] **5.2** Sigmoid hardness κ: {1, 5, 10, 50, 100} — validates ε_soft convergence from theorem
- [ ] **5.3** Reach-only vs. reach-avoid
- [ ] **5.4** Intermediate loss weight ratio in training
- [ ] **5.5** Model-agnostic: apply RA scoring to ACTIN, CRN, CT, RMSN (not just VCIP)
- [ ] **5.6** Midpoint-baseline: VCIP with Y_target = midpoint(T) vs. RA scoring (critical ablation, see Reviewer Concerns below)
- [ ] **5.7** Empirical ε_VI estimation: compute TV distance proxy between model and simulator outcome distributions on Cancer data, verify T2 bound is non-vacuous
- [ ] **5.8** VCI-inspired latent disentanglement regularizer: with/without DKL[q(Z|a_obs) || q(Z|a_alt)] penalty during training (varying λ_disent ∈ {0.01, 0.1, 0.5}). Compare RA scoring quality and intermediate prediction accuracy. Motivated by Wu et al. (ICLR 2025) Lemma 1.
- [ ] **5.9** Counterfactual consistency diagnostic validation: on Cancer, correlate VCI-style latent divergence metric with ground-truth counterfactual prediction error. If strongly correlated, use as model-intrinsic evaluation metric on MIMIC (addresses RC3).

---

### Simulated NeurIPS Reviewer Analysis (2026-03-24)

A structured pre-submission review simulating a senior NeurIPS reviewer (Score: 3 — Borderline Reject on the 2025 6-point scale). Sources: (1) NeurIPS 2025 reviewer guidelines (6-point rubric: Quality, Clarity, Significance, Originality), (2) OpenReview discussions from VCIP at ICML 2025 (3 reviews). Reviewer persona: senior researcher in safe RL / constrained MDPs / clinical AI, skeptical of simple methods.

#### Simulated Scores

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Overall** | **3 (Borderline Reject)** | Reasons to reject (limited novelty, trivial theory, oracle-dependent evaluation) slightly outweigh reasons to accept (clean formulation, thorough ablations) |
| Quality | 4/6 | Technically sound but relies on oracle trajectories (Cancer) and lacks validation (MIMIC) |
| Clarity | 5/6 | Well-written, good figure, clear algorithm |
| Significance | 2/6 | Threshold-filtering with formal framing — closer to application note than methods paper |
| Originality | 2/6 | "Reach-avoid" imports control theory language but actual computation is a threshold check |

#### Weakness Inventory (W1–W12)

| ID | Severity | Weakness | Paper section(s) | Status |
|----|----------|----------|-------------------|--------|
| W1 | CRITICAL | **"Just threshold filtering"** — method is `argmin ELBO s.t. values in range`. Algorithm 1 is 7 lines. κ≥5 ≡ hard threshold. Any practitioner would add safety checks. | Intro, Method, Discussion | Needs text reframing (A1) |
| W2 | HIGH | **Trivial theory** — Theorem 1 is a standard TV bound on bounded vs unbounded test functions. Part (c) is geometrically obvious. | Theory (Sec 5) | Needs framing edits (A2) |
| W3 | CRITICAL | **Oracle dependency** — Cancer uses ground-truth trajectories for RA filtering. Safety improvement is mechanically guaranteed. The meaningful question (quality of constrained selection) shows consistent degradation. | Setup (Sec 6.1), Discussion | Needs clarification (A3) |
| W4 | HIGH | **Weak MIMIC eval** — no ground-truth CFs. Predicted DBP range ~52-63 mmHg (narrow). 99-100% in-target may just be boundary noise selection. DBP shift <1.5 mmHg is clinically negligible. | MIMIC (Sec 6.2) | Needs new analysis (B1) |
| W5 | MEDIUM-HIGH | **Single simulator** — only Cancer has ground truth. NeurIPS expects multiple evaluation domains. | Experiments (Sec 6) | Optional new experiment (B2) |
| W6 | MEDIUM-HIGH | **No constrained RL comparison** — paper discusses CPO/RCPO/FISOR but never compares empirically. | Related Work, Experiments | Needs argument or impl (B3) |
| W7 | MEDIUM | **Feasibility collapse** — at γ=4, τ=8: only 10.3% feasible (~10/100 candidates). Method degrades to near-random at long horizons. | Results (Table 1), Discussion | Text edit (C1) |
| W8 | MEDIUM | **ε_VI proxy** — rank MAE ≠ TV distance. Connection not established. Cannot estimate on MIMIC. | Ablations (Sec 6.5) | Appendix addition (C2) |
| W9 | MEDIUM | **Ignorability untested** — gamma sweep ≠ sensitivity analysis for assumption violation. **Current paper text (line 374) is factually incorrect** on this point. | Discussion | **Critical text fix** (C3) |
| W10 | MEDIUM | **"Model-agnostic" overstated** — all 5 models share same pipeline, simulator, thresholds. "No retraining" ≠ "model-agnostic." | Contributions, Sec 6.3 | Text precision (lower priority) |
| W11 | LOW-MED | **Single author** — absence of clinical or control theory collaborator. | Meta | Cannot change |
| W12 | LOW | **Related work gaps** — missing: threshold-based clinical decision rules, MPC with constraints, safe Bayesian optimization. | Related Work | Text edits (lower priority) |

#### Mapping to Previous VCIP OpenReview Concerns (RC1–RC7)

The original RC1–RC7 (identified 2026-03-15 from VCIP's ICML review) are subsumed:

| Old ID | New ID(s) | Note |
|--------|-----------|------|
| RC1 ("Why not midpoint?") | W1 | Midpoint-baseline ablation (5.6) already addresses empirically |
| RC2 ("Is bound vacuous?") | W8 | Ablation 5.7 partially addresses; rank MAE→TV link needs formalization |
| RC3 ("MIMIC has no ground-truth CFs") | W3 + W4 | W3 is new (applies to Cancer too via oracle dependency) |
| RC4 ("Causal assumptions?") | W9 | **Gamma sweep is NOT a sensitivity analysis** — must correct |
| RC5 ("Model-agnostic?") | W10 | Sec 6.3 partially addresses; language needs tightening |
| RC6 ("Intermediate prediction quality") | Addressed | Paper documents ρ≈0 intermediates; RA filter uses simulator trajectories |
| RC7 ("Paper clarity") | Addressed | Figure 1 + intuitive theorem interpretation in draft |

#### Reviewer Defense Strategy

##### Phase 1: Score 3 → 4 (Borderline Accept) — ~2 weeks

**Priority A — Critical (text edits + one key experiment)**

- **A1 (W1): Reframe contribution.** [TEXT] Acknowledge simplicity explicitly. Contribution is: (1) formalization separating safety from quality, (2) empirical Pareto-dominance over alternatives, (3) first systematic safety evaluation of counterfactual planners. Location: Introduction (~line 103), Discussion (~line 478).

- **A2 (W2): Strengthen theory framing.** [TEXT] Reposition theorem as *design principle*, not mathematical breakthrough. Emphasize quantitative margin condition connecting to γ-sweep results. Location: Theory, after Theorem 1 (~line 302).

- **A3 (W3): Oracle-vs-model experiment.** [NEW EXPERIMENT] Run RA filtering on **model-predicted trajectories** (decoder outputs) instead of simulator ground truth on Cancer. Compare safety/Top-1/feasibility between oracle-filtered and model-predicted-filtered. Creates 3-level validation: oracle → model-predicted synthetic → model-predicted real (MIMIC).
  - **Infrastructure:** MIMIC's `extract_predicted_dbp()` (`scripts/mimic_ra/eval_mimic_traj.py:32-113`) already does this. Build parallel `extract_predicted_cv()` for Cancer using `generative_model.py:476` (`decode_p_a`) and `reach_avoid/model.py:174-320` step loop.
  - **Effort:** ~3-5 days. Run on existing 5 seeds × 4 gammas, no retraining needed.
  - **Expected:** Model-predicted RA less effective than oracle but direction of improvement holds. The gap itself is a novel finding.

**Priority B — High (new experiments + analysis)**

- **B1 (W4): MIMIC calibration + correction rate analysis.** [NEW ANALYSIS, no retraining] Using existing pickles in `results_remote/mimic_ra/`:
  1. Report full predicted DBP distribution (mean, std, percentiles)
  2. Compare predicted vs. observed DBP distributions (calibration)
  3. Compute correction rate: fraction of out-of-target ELBO selections corrected by RA
  4. Analyze whether filter discriminates meaningfully or exploits boundary noise
  - **Effort:** ~1-2 days. All data available locally.

- **B3 (W6): Reframe E6 as constrained RL comparison.** [TEXT + ANALYSIS] The soft-constraint Lagrangian experiment (E6, Sec 6.4) IS the constrained RL analogue: `score = -ELBO + λ·safety_penalty` ≡ RCPO/CPO-Lagrangian. Reframe explicitly. Paper already shows hard filtering Pareto-dominates.
  - **Effort:** ~1 day text edits.

- **B4 (W7): k-expansion experiment.** ✓ COMPLETE (2026-03-27). Ran Cancer γ=4, τ=8 with k ∈ {100, 250, 500, 1000}, 5 seeds × 100 patients. **Result:** Feasibility stable ~70% (moderate thresholds); N_feasible scales linearly (69.8→710.3); safety/in-target stable at ~90.6%; Top-1 slightly decreases with k (32.8%→26.0%) due to noisier ELBO ranking over larger feasible pool. Per-patient heterogeneity is large (0%–90%+). Data: `results_remote/b4/b4_k_expansion_gamma4_tau8.pkl`.

**Priority C — Medium (text + appendix)**

- **C1 (W5): "Standard practice" argument.** [TEXT] Note VCIP/CT/CRN precedent. Cancer spans 400+ configurations.
- **C2 (W8): ε_VI proxy formalization.** [APPENDIX] Relate rank MAE to TV via DKW inequality, or honest "practical diagnostic" framing.
- **C3 (W9): Fix sensitivity analysis claim.** [CRITICAL TEXT FIX] Remove factually incorrect claim (line 374). Gamma ≠ ignorability. Cite Frauen et al. 2023.
- **C4 (W10): Tighten "model-agnostic" language.** [TEXT] Replace with "requires no retraining" where appropriate.

##### Phase 2: Score 4 → 5 (Accept) — ~3-5 weeks additional

Three independent paths, each sufficient alone. All three together make it a strong accept.

- **S5.1: Clinical evaluation on MIMIC.** [NEW EXPERIMENT + EXTERNAL COLLABORATION]
  Recruit 1-2 ICU clinicians (intensivists) to evaluate RA-recommended treatment plans.
  - **S5.1a: Clinician plausibility assessment.** Present ~30-50 patient cases with ELBO-selected vs. RA-selected treatment sequences (blinded). Clinician rates each plan on a 5-point Likert scale (1=dangerous, 5=clinically appropriate). Report inter-rater agreement (if 2 clinicians) and mean plausibility scores.
  - **S5.1b: Outcome correlation analysis.** For MIMIC patients with observed good outcomes (e.g., DBP stabilized in [60,90] within 24h, survived ICU), compare RA-selected treatment sequences against their actual observed treatments. Report Jaccard similarity or treatment overlap. If RA selects treatments similar to what *actually worked*, this is strong indirect validation.
  - **S5.1c: Guideline concordance.** Compare RA-recommended vasopressor/fluid patterns against published clinical guidelines (e.g., Surviving Sepsis Campaign 2021 for vasopressor escalation). Report concordance rate.
  - **Infrastructure:** Existing MIMIC pickle data contains full treatment sequences. Need clinical collaborator(s) + IRB-exempt review for secondary data analysis (MIMIC already de-identified).
  - **Effort:** ~2-3 weeks (mostly coordination with clinicians; analysis is straightforward once ratings collected).
  - **Impact:** Directly addresses RC-post-2 (MIMIC lacks clinical validation). Transforms MIMIC from "feasibility demonstration" to "clinically validated."

- **S5.2: Second evaluation domain with ground truth.** [NEW EXPERIMENT + IMPLEMENTATION] ✓ **COMPLETE (2026-03-27)**
  Implemented Bergman minimal model glucose-insulin simulator as second evaluation domain.
  - **Simulator:** Bergman minimal model (3 ODEs: glucose, insulin action, plasma insulin). 1D outcome (blood glucose), 1D treatment (insulin dose). Confounding mechanism: sicker patients (higher baseline glucose) receive more insulin.
  - **Constraints:** Target: glucose ∈ [70, 180] mg/dL at terminal step. Safety: glucose ∈ [50, 250] mg/dL at all intermediate steps.
  - **Data:** 500 patients × 48 timesteps, γ=4 confounding strength. Train/test split.
  - **Training:** VCIP trained with 5 seeds × 100 epochs on Vast.ai (~5 min/seed). Models at `/root/VCIP/my_outputs/glucose_sim/22/coeff_4/VCIP/train/True/models/{seed}/model.ckpt`.
  - **Evaluation:** Custom eval script (`results_remote/glucose/eval_glucose_vcip.py`) with manual OmegaConf config construction (avoids Hydra interpolation issues). Used `optimize_a=False` in ELBO to avoid simulator dependency.
  - **Results (γ=4, 5 seeds pooled):**
    | τ | Spearman ρ | Top-1 (ELBO) | Top-1 (RA) | Δ Top-1 | In-target (ELBO) | In-target (RA) | Δ In-target | Feasibility |
    |---|-----------|-------------|-----------|---------|-----------------|---------------|-------------|-------------|
    | 2 | 0.63 | 24.4% | 22.8% | -1.6pp | 79.0% | 79.8% | +0.8pp | 82.6% |
    | 4 | 0.43 | 10.2% | 9.6% | -0.6pp | 70.0% | 72.0% | +2.0pp | 62.4% |
    | 6 | 0.34 | 8.6% | 8.0% | -0.6pp | 64.2% | 66.0% | +1.8pp | 49.0% |
    | 8 | 0.24 | 8.2% | 7.2% | -1.0pp | 62.6% | 63.8% | +1.2pp | 40.2% |
  - **Key findings:** RA constraint consistently improves in-target rates (+0.8 to +2.0 pp) with minimal Top-1 cost. Feasibility drops with horizon (82.6% → 40.2%). Spearman ρ lower than Cancer (0.24–0.63 vs 0.75–0.90), suggesting harder prediction task.
  - **Paper integration:** New Appendix section "Second Evaluation Domain: Glucose-Insulin Simulator" (`\label{app:glucose}`) with Bergman ODEs, confounding mechanism, results table, key findings. Discussion cross-references glucose results. `bergman1979quantitative` BibTeX entry added.
  - **Impact:** Partially addresses W5 (second domain with ground truth). Demonstrates cross-domain generalization of RA-constrained selection.

- **S5.3: Stronger theoretical contribution.** [NEW THEORY]
  Develop one of the following theoretical results (ranked by feasibility):
  - **S5.3a: Finite-sample constrained selection bound.** Given $n$ i.i.d. patients and $k$ candidate sequences per patient, bound the probability that constrained selection picks a truly-unsafe sequence. Key ingredients: uniform convergence over the candidate set + concentration inequality for the feasibility indicator. Related to PAC-Bayes or uniform convergence in binary classification. This gives a *deployment guarantee*: "with probability $1-\delta$, the constrained selector picks a safe plan for at least $(1-\alpha)$ fraction of future patients."
  - **S5.3b: Connection to chance-constrained optimization.** Frame Eq. 5 as an empirical approximation to chance-constrained optimization: $\min_{\bar{a}} \text{ELBO}(\bar{a})$ s.t. $\Pr(E(\bar{a})) \geq 1 - \alpha$. Derive the sample complexity (in $k$) needed for the empirical feasibility check to approximate the true chance constraint within tolerance $\epsilon$. This connects the reach-avoid framework to the well-studied chance-constrained optimization literature (Nemirovski & Shapiro 2006).
  - **S5.3c: Regret bound for constrained selection.** Define regret relative to the oracle constrained selector (which knows the true feasibility). Bound the expected regret in terms of $\varepsilon_{\text{VI}}$, $k$, and $\tau$. Show the regret vanishes as model quality improves ($\varepsilon_{\text{VI}} \to 0$) or candidate pool grows ($k \to \infty$).
  - **Effort:** ~2-4 weeks (theory development + proof writing + empirical verification).
  - **Impact:** Elevates the paper from "empirical contribution with modest theory" to "principled framework with deployment guarantees." Addresses the novelty concern (W1) from a completely different angle.

##### Phase 3: Score 5 → 6 (Strong Accept) — ~3-4 weeks additional

**Core problem:** After Phases 1-2, the paper is solidly executed but not *exciting*. The method is still `argmin ELBO s.t. values in range`. Score 6 requires something that makes the reviewer think "I didn't expect that" or "the community needs to see this."

**Key optimization: S5.3 (Phase 2 theory) is subsumed by S6.1.** The conformal certificate IS a finite-sample guarantee, done properly and more powerfully. Skip S5.3 as a separate task; go directly to S6.1.

- **S6.1: Conformal safety certificates.** [NEW THEORY + EXPERIMENT] ★ PRIMARY CONTRIBUTION — **INVESTIGATED (2026-03-27), result: too conservative for current setting**
  Replace the fixed threshold δ with distribution-free safety certificates via conformal prediction. Instead of "this plan passes the filter," provide: "with probability ≥ 1-α, this plan's true outcomes will stay in the safety region."
  - **Technical approach:**
    1. Split data into train/calibration/test. On calibration set, compute nonconformity scores: s_i = max(max_s |Y_s[ā] - boundary_S|₋, |Y_{t+τ}[ā] - boundary_T|₋)
    2. For new patients, the conformal prediction set C_α = {ā : s(ā) ≤ q̂_{1-α}} where q̂ is the calibrated quantile
    3. Safety certificate: ā is "certified safe at level α" if conformal prediction set ⊆ T × S^τ
    4. For causal setting: use weighted conformal (Lei & Candès 2021) with propensity reweighting
    5. Integration: F_conf = {ā : certified safe at level α}, then ā* = argmin_{ā ∈ F_conf} ELBO(ā)
  - **Key theorem:** Coverage guarantee: P(Y_true[ā*] ∈ T ∩ S^τ) ≥ 1-α (distribution-free)
  - **Why transformative:** Addresses W1 (novelty), W2 (theory depth), W8 (replaces ε_VI proxy with rigorous guarantee) simultaneously. First distribution-free safety certificates for counterfactual treatment planning.
  - **Key references:** Lei & Candès 2021, Angelopoulos & Bates 2023, Vovk et al. 2005
  - **Effort:** ~3-4 weeks (theory: 1 week, implementation: 1-2 weeks, experiments: 1 week)
  - **Preliminary finding (2026-03-27):** Conformal certificates implemented and tested on Cancer data. The calibrated quantile thresholds are extremely conservative — the nonconformity scores from the calibration set are large enough that conformal filtering admits very few candidates (near-zero feasibility at standard α levels like 0.05 or 0.10). This is because the model's trajectory predictions have high variance across patients, making the conformal prediction sets very wide. **Status:** Written up as Appendix section in paper (honest negative result: "conformal certificates are theoretically valid but practically too conservative with current model quality"). The core RA threshold-based filter remains the practical method. Conformal certificates may become practical with improved trajectory models or larger calibration sets. This finding itself is informative — it quantifies the gap between distribution-free guarantees and practical utility in this setting.

- **S6.2: Systematic failure taxonomy.** [NEW ANALYSIS]
  Characterize *when, why, and for whom* existing planners produce dangerous recommendations.
  - Identify patients where ELBO-optimal plan is unsafe across all 5 models × 4 gammas × 4 horizons
  - Cluster failures by patient features (CART/decision tree on danger rate)
  - Classify modes: "aggressive-harmful," "conservative-insufficient," "model-confused"
  - Show failures are correlated across models (structural, not model-specific)
  - **Effort:** ~1-2 weeks (uses existing Cancer experiment data)
  - **Impact:** Transforms narrative from "we add safety" to "we discover a systematic safety blind spot"

- **S6.3: DRO interpretation (remark within S6.1).** [THEORY, minor]
  The TV bound in Theorem 1 gives a distributionally robust interpretation: reach-avoid selection with threshold δ is equivalent to DRO safety check P̂(E(ā)) ≥ 1-δ+ε_VI under TV ambiguity set. One-paragraph connection, not a separate section.

##### Phase 4: Score 6 → 7 (Oral / Top 1-2%) — ~4-6 weeks additional

**Core challenge:** Score 7 requires a contribution that opens a new research direction or provides a tight theoretical characterization. Qualitatively different from "better experiments."

- **S7.2: Safe planning under hidden confounding.** [NEW THEORY + EXPERIMENT] ★ KEY CONTRIBUTION
  Develop safety guarantees valid even when unmeasured confounders exist (sequential ignorability fails).
  - **Γ-sensitivity model:** Parameterize hidden confounding by Γ ≥ 1 (Rosenbaum 2002). Unobserved confounders can shift treatment assignment odds by factor Γ. When Γ=1, ignorability holds.
  - **Identified set:** Under bounded confounding Γ, counterfactual outcome Y[ā] lies in a set [Y_lower(Γ), Y_upper(Γ)]. Derive sharp bounds for sequential treatments.
  - **Γ-robust conformal certificate:** Target theorem: P(Y_true[ā] ∈ S | confounding ≤ Γ) ≥ 1-α with Γ-adjusted quantile. Extends S6.1 naturally.
  - **Key challenge:** Sensitivity compounds over τ steps. Mitigation: use Markov property (next-step sensitivity depends only on current state) or present results for small τ first.
  - **Validation on Cancer:** Artificial hidden confounding (hold out a covariate from model training). Verify coverage holds across Γ levels. Also validate on gamma sweep (natural confounding variation).
  - **Key references:** Rosenbaum 2002, Tan 2006, Kallus & Zhou 2021, Yadlowsky et al. 2022
  - **Effort:** ~4-6 weeks (theory: 2-3 weeks, implementation: 1-2 weeks, experiments: 1 week)
  - **Impact:** First distribution-free safety guarantee for counterfactual planning under hidden confounding. Addresses W9 from a completely new angle.

- **S7.1: Minimax optimality bounds (if time).** [NEW THEORY]
  Prove tight lower + upper bounds on achievable safety rate:
  - Lower bound: no selection rule can achieve safety > 1-Ω(ε_VI/√k) (information-theoretic limit)
  - Upper bound: conformal-RA achieves safety ≥ 1-O(ε_VI/√k + 1/√n_cal)
  - Corollary: conformal-RA is minimax optimal as n_cal → ∞
  - **Effort:** ~4-6 weeks (hard theory). **Secondary priority — pursue only if S7.2 completes early.**

##### Phase 5: Score 7 → 8 (Best Paper) — exploration only

- **S8.1: "The Safety Tax" — fundamental quality-safety tradeoff.** [THEORY, exploratory]
  Prove: any counterfactual selector with safety rate ≥ 1-δ incurs quality regret ≥ Ω(f(ε_VI, δ, k, τ)). Show conformal-RA achieves this bound. The function f reveals the minimum quality sacrifice for guaranteed safety — analogous to "price of fairness."
  - **Realistic assessment:** Explore in Week 5. If a clean result emerges, include. If not, frame as open question in discussion. Score 8 cannot be planned — it requires breakthrough insight.

#### Impact Assessment (Full Strategy: Phases 1-5) — Updated 2026-03-27

| Phase | Actions | Score | Cumulative effort | Status |
|-------|---------|-------|-------------------|--------|
| **Current paper** | — | **3 (Borderline Reject)** | — | Baseline |
| **Phase 1** | A1-A3, B1, B3, B4, C1-C4 | **4 (Borderline Accept)** | ~1-2 weeks | **IN PROGRESS** (B4 started, text edits pending) |
| **Phase 2** | + S5.1 (clinical) + S5.2 (simulator) | **4-5 (Accept candidate)** | +1 week | **S5.2 ✓ COMPLETE** (glucose). S5.1 needs collaborator. |
| **Phase 3** | + ~~S6.1 (conformal)~~ + S6.2 (failure taxonomy) + S5.3 (theory) | **5 (Accept)** | +2 weeks | S6.1 too conservative (appendix only). S6.2/S5.3 remain. |
| **Phase 4** | + S7.2 (hidden confounding) | **5-6 (Accept / Strong Accept)** | +2 weeks | Not started |
| **Phase 5** | + S8.1 (safety tax, if works) | **6-7 (Strong Accept / Oral)** | +1 week | Not started |
| **Total** | — | **5-6 (realistic) / 6-7 (optimistic)** | **~6 weeks (May 06 deadline)** | |

**Key recalibration (2026-03-27):** S6.1 conformal certificates proved too conservative for practical use (near-zero feasibility at standard α levels). This was the planned Score 5→6 upgrade. Without it as primary contribution, the score ceiling from experiments alone is ~5. Reaching Score 6+ now requires either: (a) S7.2 hidden confounding theory succeeds, (b) S5.3 finite-sample bounds produce non-trivial results, or (c) a new theoretical direction emerges.

**Revised critical path:** Phase 1 text edits + A3 (Week 2) → S6.2 failure taxonomy (Week 2) → S5.3 or S7.2 theory (Weeks 3-4) → paper integration (Week 5) → polish + submit (Week 6).

#### Week-by-Week Schedule (Mar 26 → May 06, 41 days)

| Week | Dates | Tasks | Deliverables | Target score |
|------|-------|-------|-------------|-------------|
| **1** | Mar 26 – Apr 2 | ~~Phase 1 text edits + A3 + B4 + B1 + S6.1 theory start~~ | ~~Phase 1 complete; S6.1 theorem draft~~ | 4 |
| | | **ACTUAL:** S5.2 complete (glucose simulator: Bergman model, VCIP training 5 seeds, RA eval, paper appendix). S6.1 investigated (conformal certificates too conservative — honest negative result, written as appendix). B4 k=1000 started (cancer). | S5.2 ✓ DONE. S6.1 appendix ✓ DONE. Phase 1 text edits PENDING. A3/B1 PENDING. | 3→4 (partial) |
| **2** | Apr 2 – Apr 9 | Phase 1 text edits (A1, A2, C3) + A3 oracle-vs-model + B1 MIMIC analysis + B4 completion + S6.2 failure taxonomy | Phase 1 complete; S6.2 done | 4-5 |
| **3** | Apr 9 – Apr 16 | S7.2 theory development + remaining text edits (B3, C1, C2, C4, W10, W12) | S7.2 framework drafted; all text edits done | 5-6 |
| **4** | Apr 16 – Apr 23 | S7.2 experiments + paper integration | S7.2 validated; full draft | 6-7 |
| **5** | Apr 23 – Apr 30 | S8.1 exploration + S5.1 (if collaborator) + polish | Near-final paper | 6-7 |
| **6** | Apr 30 – May 6 | Final experiments, polishing, submission | **Submitted** | **6-7** |

#### Post-Improvement Simulated Review (All Phases)

| Criterion | Before | After Phase 1 | After Phase 3 (S6.1) | After Phase 4 (S7.2) |
|-----------|--------|---------------|----------------------|----------------------|
| **Overall** | **3** | **4** | **6 (Strong Accept)** | **6-7 (Oral candidate)** |
| Quality | 4/6 | 5/6 | 6/6 | 6/6 |
| Clarity | 5/6 | 5/6 | 5/6 | 5/6 |
| Significance | 2/6 | 3/6 | 5/6 | 5-6/6 |
| Originality | 2/6 | 3/6 | 5/6 | 6/6 |

**Per-weakness resolution status (all phases):**

| ID | Before | After Phase 1 | After Phase 3 | After Phase 4 |
|----|--------|---------------|---------------|---------------|
| W1 | CRITICAL | MEDIUM | **RESOLVED** (conformal certificates are genuinely novel) | **RESOLVED** |
| W2 | HIGH | LOW | **RESOLVED** (coverage theorem is substantive) | **RESOLVED** (+ Γ-robust theory) |
| W3 | CRITICAL | **RESOLVED** | **RESOLVED** | **RESOLVED** |
| W4 | HIGH | MEDIUM | MEDIUM | **RESOLVED** (S5.1 if collaborator; otherwise conformal calibration helps) |
| W5 | MED-HIGH | MEDIUM | **LOW** (S5.2 glucose simulator ✓) | **RESOLVED** (S5.2 second simulator) |
| W6 | MED-HIGH | LOW | LOW | LOW |
| W7 | MEDIUM | **RESOLVED** | **RESOLVED** | **RESOLVED** |
| W8 | MEDIUM | LOW | **RESOLVED** (conformal replaces ε_VI proxy entirely) | **RESOLVED** |
| W9 | MEDIUM | **RESOLVED** (text fix) | **RESOLVED** | **RESOLVED** (+ Γ-robust guarantees address it substantively) |
| W10 | MEDIUM | LOW | **RESOLVED** (S5.2 second domain) | **RESOLVED** |
| W11 | LOW-MED | LOW-MED | LOW | LOW |
| W12 | LOW | **RESOLVED** | **RESOLVED** | **RESOLVED** |

#### Revised Paper Structure (After All Phases)

```
Main Paper (9 pages):
1. Introduction — safety blind spot in counterfactual planning
2. Background — VCIP, potential outcomes, reach-avoid formulation
3. Method
   3.1 Reach-avoid constrained selection (existing, condensed)
   3.2 Conformal safety certificates (S6.1) ★ KEY CONTRIBUTION
   3.3 Robust safety under hidden confounding (S7.2) ★ KEY CONTRIBUTION
4. Theory
   4.1 Ranking robustness (existing Thm 1, reframed as design principle)
   4.2 Coverage guarantee (new, from S6.1)
   4.3 Γ-robust coverage (new, from S7.2)
5. Experiments
   5.1 Cancer simulator — RA + conformal + oracle-vs-model + Γ-robustness
   5.2 Glucose-insulin simulator — cross-domain validation
   5.3 MIMIC-III — clinical application + conformal calibration
   5.4 When do planners fail? (failure taxonomy highlight)
6. Discussion + Limitations

Appendix: Proofs, full failure taxonomy, glucose-insulin details,
          MIMIC extended results, all ablations, model-agnostic results
```

---

### Paper Structure

```
1. Introduction
   - Counterfactual intervention planning: important problem
   - VCIP (ICML 2025) is SOTA but uses point-target ELBO ranking
   - Two issues identified by VCIP's own reviewers and authors:
     (1) clinical targets are ranges not points (Reviewer ixW5, author rebuttal),
     (2) ELBO ranking is provably inconsistent under bound-gap variation (our W1 finding)
   - We propose reach-avoid scoring that addresses both
   - Figure 1: Visual concept — trajectory reaching T while staying in S

2. Background & Problem Setup
   - Longitudinal causal setting, potential outcomes
   - VCIP review (latent dynamics, ELBO, Eq. 19)
   - Reach-avoid event E(ā) = {Y_{t+τ} ∈ T} ∩ {∀s: Y_s ∈ S}
   - Assumptions: list all (inherited + new), with explicit discussion

3. Why ELBO Ranking Fails (motivating analysis)
   - Theorem: ELBO ranking consistency requires constant bound gap
   - Empirical: 19% Top-1 at tau=2, rho=0.75-0.90
   - Converts W1 evidence into formal motivation

4. Method: Reach-Avoid Scoring
   - Differentiable reach-avoid score J_RA with soft indicators
   - Intuitive interpretation before formal definition
   - RA-aware training: (a) weighted intermediate prediction loss, (b) VCI-inspired
     latent disentanglement regularizer (Wu et al., ICLR 2025) — ensures Z_s encodes
     patient features rather than treatment info, improving counterfactual trajectory quality
   - Gradient-based RA planning

5. Theory
   - Theorem (T2): RA ranking robustness bound
   - "What it means" paragraph: intuitive interpretation
   - Corollary: RA is strictly more robust when boundary mass is small
   - Connection: explains why ELBO ranking degrades at short horizons
   - Empirical validation: ε_VI estimation, margin distributions

6. Experiments
   6.1 Cancer simulator (ground truth)
       - RA vs ELBO ranking quality (Top-1, Spearman, margin analysis)
       - Attainment and violation rates
       - RA-aware training improvement
       - Midpoint-baseline comparison (addresses RC1)
       - Intermediate prediction quality analysis (addresses RC6)
       - Ablations (κ, T size, safety constraint, model-agnostic)
   6.2 MIMIC-III (clinical plausibility)
       - Cross-seed ranking stability
       - Clinical plausibility of RA-recommended sequences
       - VCI-style counterfactual consistency diagnostic (addresses RC3)
       - Sensitivity to T/S definitions
   6.3 Sensitivity analysis
       - Confounding strength (gamma sweep, addresses RC4)
       - ε_VI estimation and bound verification (addresses RC2)

7. Discussion & Conclusion
   - RA scoring as a drop-in replacement for ELBO ranking
   - Limitations: T/S must be clinically specified; real-data evaluation
     inherits counterfactual unobservability (addresses RC3)
   - Confounding robustness as future work (addresses RC4)
   - Longitudinal VCI (extending counterfactual ELBO to sequential settings) as future work
```

Working title: *"Reach-Avoid Counterfactual Intervention Planning via Variational Latent Dynamics"*

#### Paper Writing Status (2026-03-27)

- **Full draft complete:** 9 content pages + references + NeurIPS checklist + 9 appendix sections (25 pages total)
- **Figure 1:** Concept visualization with 4 trajectories, Y* marker, annotation arrows. PDF/PNG generated.
- **Related work:** Main text (concise, ~1 page) + Appendix H (extended, ~3 pages, 6 subsections). 25+ new references added.
- **NeurIPS checklist:** Updated to current 16-question format with `\answerYes{}`/`\answerNo{}`/`\answerNA{}` and justifications referencing specific sections.
- **Simulated reviewer analysis (2026-03-24):** 12 weaknesses (W1–W12) identified from senior NeurIPS reviewer perspective. Priority A defensive edits (A1–A3) are submission-blocking text changes. See "Simulated NeurIPS Reviewer Analysis" section above.
- **S5.2 glucose appendix (2026-03-27):** New appendix section "Second Evaluation Domain: Glucose-Insulin Simulator" (`\label{app:glucose}`) with Bergman minimal model ODEs, confounding mechanism, constraint specification, full results table (5 seeds × 4 taus), key findings paragraph. Discussion section updated with cross-reference to glucose results. `bergman1979quantitative` BibTeX entry added. Paper compiles cleanly at 25 pages.
- **S6.1 conformal appendix (2026-03-27):** Conformal safety certificates investigated and written up as appendix section. Result: certificates are theoretically valid but practically too conservative with current model quality (near-zero feasibility at standard α levels). Presented as honest negative result — informative finding about the gap between distribution-free guarantees and practical utility.
- **A3 oracle-vs-model (2026-03-27):** Experiment complete. Model-predicted RA filtering has near-zero feasibility (0.1-0.2%) on Cancer due to high scaling variance (std=53.176). Oracle feasibility ~70%. Quantifies structural oracle gap — valuable negative result for paper. Data: `results_remote/a3/a3_oracle_vs_model_gamma4.pkl`.
- **B4 k-expansion (2026-03-27):** Experiment complete. Feasibility stable ~70% with moderate thresholds, N_feasible scales linearly with k. Top-1 decreases slightly with k (noisier ELBO ranking). Data: `results_remote/b4/b4_k_expansion_gamma4_tau8.pkl`.
- **Phase 1 defensive edits COMPLETE (2026-03-27):** All Priority A (A1-A3), B (B1, B3, B4), and C (C1, C3, C4/W10) text edits done. S6.2 failure taxonomy completed and integrated as new Discussion paragraph. "First systematic safety evaluation" added to abstract. RLHF rejection sampling cite added. Failure modes: conservative undershoot dominates at short τ, toxic path grows at long τ. B1 MIMIC calibration validates all paper claims. Paper compiles at 25 pages.
- **Remaining for submission:** C2 (ε_VI formalization, optional appendix), W12 (related work gap fills), S7.2 theory (if time), final polish pass. **All GPU experiments complete — Vast.ai instance shut down.**

---

### Key Files for Implementation

| File | Role | Modifications needed |
|---|---|---|
| `VCIP/src/data/cancer_sim_cont/cancer_simulation.py` | Simulator | Return full trajectory from `simulate_output_after_actions()` (line 929) |
| `VCIP/src/models/vae_model.py` | Model + evaluation | Add RA scoring in `optimize_interventions_discrete_onetime()` (line 571); modify intermediate loss weighting (line 511) |
| `VCIP/src/models/generative_model.py` | Decoder | No changes — already supports Y_s decoding from Z_s at any step |
| `VCIP/src/utils/helper_functions.py` | Perturbation generation | No changes |
| `VCIP/results/weakness_analysis.ipynb` | Analysis | Extend with RA scoring comparison |

---

### Future Directions (beyond current NeurIPS submission)

#### 1. Confounding Robustness (R²-VCIP)

The full robust reach-avoid formulation with sequential sensitivity model and adversarial reweighting (max-min over bounded density ratios). See `ideas/IDEA1.md` for the complete sketch. Deferred because our evidence shows VCIP advantage *grows* with confounding strength (gamma), making robustness less compelling as a headline contribution.

#### 2. Hybrid Mamba+LSTM+Attention Architecture

Replace VCIP's LSTM-only backbone with a three-layer hybrid: (1) **Mamba layer** for O(n) global temporal context across long patient histories, (2) **LSTM layer** for fine-grained local sequential dependencies and treatment-state transitions, (3) **Attention layer** to highlight safety-critical events for targeted prediction. Rationale: while VCIP's LSTM works well for moderate sequence lengths (tau ≈ 2-8), clinical EHR data can span much longer horizons. Not for current submission because: (a) identified weaknesses are in the *objective* and *evaluation*, not the architecture; (b) would require full hyperparameter search; (c) risks diluting the focused reach-avoid contribution.

#### 3. Longitudinal VCI

Extend VCI's counterfactual ELBO (Wu et al., ICLR 2025) from the static to the sequential setting. VCI derives p(Y'|Y,X,T,T') for individual-level counterfactual likelihood; adapting this to longitudinal trajectories would provide a principled foundation for counterfactual intervention planning without requiring the g-formula connection.
