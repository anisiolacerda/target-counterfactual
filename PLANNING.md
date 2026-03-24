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

### Anticipated Reviewer Concerns (from VCIP OpenReview analysis, 2026-03-15)

VCIP was accepted as ICML 2025 poster with 4 weak accepts (one upgraded from weak reject). The OpenReview discussion reveals specific criticisms that will transfer to or inform our paper. Source: `literature_review/pdfs/VCIP discussion on OpenReview.pdf`.

#### RC1 — "Why not just use the midpoint of the range?" (HIGHEST PRIORITY)

**Source:** Reviewer ixW5 noted that "in medicine usually there is no single target we want to optimise for, but rather a range of values." The VCIP authors responded: *"a straightforward approach would be taking a value from the 'range of values' as the target, such as the midpoint of the interval."*

**Risk:** This is the simplest objection to our paper. A reviewer can argue that VCIP with Y_target = midpoint(T) is a trivial baseline that might perform comparably.

**Defense (must be demonstrated experimentally in ablation 5.6):**
1. Midpoint targeting ignores safety constraints entirely — no intermediate Y_s monitoring.
2. Midpoint targeting treats barely-in-range and comfortably-in-range identically — it ranks a sequence achieving Y = T_lower_bound the same as Y = midpoint, even though the latter is clinically safer.
3. Midpoint targeting still suffers from ELBO ranking inconsistency (W1) at short horizons — RA scoring addresses this by construction (T2 theorem).
4. When multiple sequences achieve similar distance to midpoint, ELBO-based ranking is degenerate (many ties or near-ties) — RA scoring distinguishes by safety constraint satisfaction.

#### RC2 — "Is the T2 bound vacuous?" (HIGH PRIORITY)

**Source:** Reviewer g91d raised the same concern about VCIP's Theorem 4.1: *"if ELBO₁ is not maximized, ε₁ can be arbitrarily large."* This was the most critical concern in the entire review (initially scored weak reject).

**Risk:** Reviewers will ask: "What are typical values of ε_VI? Is the ranking preservation condition m > 2ε_VI + 2τ·ε_soft ever satisfied in practice?"

**Defense (must be demonstrated in ablation 5.7):**
1. Estimate ε_VI empirically on Cancer data where we have both model predictions and ground-truth outcomes (compute TV distance or related divergence).
2. Compute the margin distribution for all pairwise comparisons under RA scoring.
3. Show the fraction of pairs where m > 2ε_VI + 2τ·ε_soft, and compare to the analogous fraction under point-target ranking.
4. If the bound is tight for a substantial fraction of pairs, the theorem is actionable. If not, present it as an asymptotic guarantee and emphasize the empirical improvements.

#### RC3 — "How do you evaluate on MIMIC without ground-truth counterfactuals?"

**Source:** Reviewer whTh asked why RCS was not reported for MIMIC. VCIP authors explained that RCS requires computing true potential outcomes for each candidate sequence, which is only possible with simulated data.

**Risk:** Same limitation applies to us. We cannot compute ground-truth attainment/violation rates on MIMIC.

**Defense:**
1. Report cross-seed ranking stability (already planned in E3) — does RA produce more consistent recommendations?
2. Report clinical plausibility: do RA-recommended sequences fall within medically reasonable action patterns?
3. Acknowledge explicitly as a shared limitation of all counterfactual planning methods on observational data.
4. Note that our Cancer results (with ground truth) provide the evidential backbone; MIMIC demonstrates feasibility and plausibility.

#### RC4 — "What are the causal assumptions? Sensitivity analysis?"

**Source:** Reviewer XzCZ criticized VCIP for *"implicitly assuming consistency, positivity, and sequential ignorability without extensive empirical or theoretical exploration of sensitivity to these assumptions."* VCIP authors admitted they couldn't test sequential ignorability violations.

**Risk:** We inherit VCIP's backbone and assumptions. Reviewers will ask the same question.

**Defense:**
1. We already have a gamma={1,2,3,4} sweep (confounding strength sensitivity) — present this as a partial sensitivity analysis.
2. Show RA scoring performance across all gamma levels — if it is robust across gamma, this addresses the concern.
3. Explicitly list all assumptions: consistency, positivity, sequential ignorability (inherited from VCIP) + T and S clinically specified (new) + soft indicator approximation (controlled by κ).
4. Note that the confounding robustness extension (IDEA1/R²-VCIP) is structured future work that would directly address sequential ignorability violations.

#### RC5 — "Does RA scoring benefit only VCIP, or is it model-agnostic?"

**Source:** Reviewer g91d questioned whether VCIP's gradient optimization approach had specific advantages over applying the same optimization to other models' likelihoods.

**Risk:** If RA scoring only improves VCIP but not baselines, it may be seen as VCIP-specific rather than a general contribution.

**Defense (ablation 5.5):**
1. Apply RA scoring to ACTIN, CRN, CT, RMSN.
2. If RA scoring improves all models: strengthens paper (method-agnostic contribution).
3. If RA scoring especially benefits VCIP: explain that VCIP's latent dynamics produce better intermediate Y_s predictions because the architecture was designed for sequential trajectory modeling (KL-regularized latent transitions), while baselines use sequential prediction without the same latent structure.

#### RC6 — "Intermediate prediction quality"

**Source:** Not directly raised for VCIP, but follows logically from our approach. VCIP was designed to optimize terminal Y_{t+τ}; intermediate predictions are regularization only (line 511: `reg_loss = reg_losses[-1]`).

**Risk:** If VCIP's intermediate predictions are poor, RA scoring based on those predictions will be unreliable. Safety constraints require accurate Y_s at all steps, not just Y_{t+τ}.

**Defense:**
1. Report intermediate prediction quality on Cancer data (compare model-predicted Y_s against simulator ground truth at each step s).
2. If quality is poor at intermediate steps, this motivates Phase 2 (RA-aware retraining) and makes the contribution more substantial.
3. This is actually an advantage for our paper: we demonstrate the problem (poor intermediates) and provide the fix (RA-aware training).

#### RC7 — "Paper clarity and exposition"

**Source:** Program Chairs noted VCIP was "somewhat hard to read." Reviewer g91d: "the paper is difficult to understand."

**Action items for our paper:**
1. Include a visual figure showing reach-avoid concept (patient trajectory entering T while staying in S).
2. Provide intuitive interpretation of J_RA before the formal definition.
3. Make table captions self-contained.
4. Give the T2 theorem an intuitive "what it means" paragraph immediately after the formal statement.
5. Use the VCIP authors' own acknowledgment of the range-target limitation (from their rebuttal to Reviewer ixW5) as direct motivation in our Introduction.

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

#### Paper Writing Status (2026-03-23)

- **Full draft complete:** 9 content pages + references + checklist + 7 appendix sections (19 pages total)
- **Figure 1:** Concept visualization with 4 trajectories, Y* marker, annotation arrows. PDF/PNG generated.
- **Related work:** Main text (concise, ~1 page) + Appendix H (extended, ~3 pages, 6 subsections). 25+ new references added covering:
  - Counterfactual outcome estimation (Shalit 2017, CEVAE, GANITE, SyncTwin, TSD, Causal CPC, etc.)
  - Variational/generative causal inference (Deep SCMs, VCI, DECI, disentanglement)
  - Dynamic treatment regimes (Nie et al. 2021, Le et al. 2019)
  - Safe/constrained RL (CPO, RCPO, CUP, FISOR, Altman CMDPs)
  - Reach-avoid control (DeepReach, ISAACS)
  - Clinical AI safety (AI Clinician, Raghu et al.)
- **Remaining:** Final polish pass, check all cross-references, verify notation consistency

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
