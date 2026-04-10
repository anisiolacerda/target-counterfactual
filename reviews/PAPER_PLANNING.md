# Paper Improvement Plan
**Paper:** Reach-Avoid Constrained Selection for Safe Counterfactual Intervention Planning
**Target:** NeurIPS 2026 (main track)
**Current estimated rating:** 4.5 / 10 (borderline reject)
**Target rating:** 7+ / 10 (weak-to-clear accept)
**Date created:** 2026-04-05

---

## Executive Assessment

The paper is well-written, well-structured, and theoretically literate, but two empirical load-bearing claims are fragile: (1) the Cancer safety improvements depend on oracle simulator trajectories and collapse to a near-no-op with the current trained decoder, and (2) the MIMIC-III "in-target rate lift from 73–81% to 99–100%" is largely definitional because both filtering criterion and evaluation metric rely on the same predicted DBP. The theoretical contributions are correct but incremental — a coarsening argument + sequential Rosenbaum bound + Hoeffding/Bernstein certification, none of which is new machinery. A NeurIPS reviewer who probes either empirical claim or asks "what's technically new?" will find the paper thin. Closing the gap requires (a) re-framing the oracle vs. model-predicted regime as a first-class finding, (b) adding outcome-grounded validation on MIMIC, (c) sharpening novelty via a Pareto-optimality result for post-hoc filters, and (d) strengthening baselines and the clinician evaluation.

The overall **strategy** is:
1. **Fix the truth-in-reporting issues first** (re-label oracle results, add matched-treatment MIMIC analysis). These are cheap and eliminate the largest reviewer attack surfaces.
2. **Add one genuinely new theoretical result** (Pareto-optimality of hard filtering, or a matching lower bound). Without this, the paper is an application paper.
3. **Strengthen the empirical moat** (decoder calibration experiment, stronger baselines, expanded clinician evaluation).

---

## Gap Analysis: Current → Target

### Rating Dimension Breakdown

| Dimension | Current (est.) | Target | Gap | Priority |
|-----------|---------------|--------|-----|----------|
| Novelty | 4/10 | 7/10 | Method is a rejection rule; theorems are direct TV/coupling arguments. | P1 |
| Technical soundness | 7/10 | 8/10 | Proofs are correct but Theorem 1(c) is qualitative; sensitivity bound uses worst-case per-step TV. | P2 |
| Experimental rigor | 4/10 | 8/10 | Oracle dependence on Cancer; definitional metric on MIMIC; only one soft-constraint baseline; small clinician study with fragile pooling. | P0 |
| Clarity | 7/10 | 8/10 | Well-written. Oracle-vs-model distinction buried. | P2 |
| Significance | 5/10 | 8/10 | Clinically motivated and honest, but without the two empirical fixes the contribution looks like "one more filter." | P0 |
| Reproducibility | 6/10 | 8/10 | Code availability not stated in main body. | P2 |

### P0 — Blocking Issues (must fix or paper will be rejected)

1. **Cancer oracle dependence must be disclosed prominently and at least one deployable variant must be demonstrated.** Every reported Cancer improvement currently assumes oracle simulator trajectories. Abstract, contributions, Section 6.2, Table 1, Table 3 must all be re-labelled as "oracle safety." In addition, at least one decoder modification (e.g., variance-calibrated output head, wider decoding range, MC-dropout predictive intervals, heteroscedastic regression head) must be trained and shown to recover at least *some* discrimination under model-predicted RA filtering. Without this, a reviewer will conclude the method "does not work on Cancer in a deployable setting."

2. **MIMIC-III in-target claim must be de-tautologized with outcome-grounded evidence.** The headline lift 73–81% → 99–100% is a consequence of filtering against the same predictor used for evaluation. Fix by: (a) re-labelling clearly in Abstract and Section 6.4, (b) adding a matched-treatment subgroup analysis (patients where the selected sequence equals the observed one) reporting *observed* in-target DBP rates, (c) adding a train/test-disjoint evaluation where the RA predictor and the metric predictor differ.

### P1 — Critical Issues (strongly affect rating)

3. **Add a genuinely new theoretical contribution.** The most tractable candidate: prove a Pareto-optimality characterization for post-hoc filters (any filter achieving safety ≥ s* must lose at least X in ranking quality; hard RA filtering achieves this bound). Alternatively, a matching lower bound on ranking preservation under the coarsening coupling, or a minimax characterization against an adversary choosing ε_VI. Without at least one new theoretical artifact, novelty stays at 4–5 / 10.

4. **Strengthen the clinician evaluation or reframe its role.** Options: (a) recruit a third rater to break the 2-rater calibration tie, (b) replace the pooled Wilcoxon with a rater-random-effects model reporting effect size and paired confidence interval, (c) pre-register / justify case-selection criterion, (d) explicitly reframe as *suggestive* rather than *confirmatory*. The current headline "p < 0.0001, 83% preference" is fragile and reviewers will notice.

5. **Add stronger soft-constraint baselines.** The "Pareto-dominance" claim currently rests on a fixed-λ Lagrangian. Add (a) adaptive-λ (PID-Lagrangian) and (b) a midpoint-target-match baseline (VCIP with Y* = midpoint(T)), both in the main text. If hard filtering still dominates, the claim is earned; if not, soften to "comparable safety, better Top-1."

6. **Fill related-work gaps.** Cite high-confidence OPE (Thomas et al. 2015; Thomas & Brunskill 2016), safe policy learning (Kallus 2018), sequential Rosenbaum / distributionally robust sensitivity (Yadlowsky et al. 2022; Kallus & Zhou 2020), and conformal counterfactual prediction (Lei & Candès 2021). Replace "first systematic safety evaluation" with a sharper claim.

7. **Tighten the sensitivity bound for long horizons.** Γ_max = 2.1 at τ = 8 is right at the edge of clinical plausibility. Try (a) exploiting shared latent structure across steps to reduce the worst-case per-step TV, or (b) adding a distributionally robust alternative for long horizons.

### P2 — Important Improvements (would gain 1–2 rating points)

8. **Quantitative Theorem 1(c).** Derive an explicit inequality for the RA margin in terms of boundary mass β (e.g., m_RA ≥ 1 − 2β − 2ε_VI), replacing the informal "m_RA ≫ m_ELBO / C_d."

9. **Promote the midpoint baseline to the main text** (currently appendix). It is the most natural straw-man.

10. **Report a cost-aware / utility-weighted selection metric** so the Top-1 vs. safety tradeoff is quantified on a single axis.

11. **Move the three-regime framework (Certified / Optimistic / Oracle) from appendix into Section 3 or 4.** It is one of the paper's genuinely useful conceptual contributions and is hidden.

12. **Fix Section 3.1 S-set notation.** Currently conflates outcome-space (volume ≤ 12) and treatment-space (chemo ≤ 5) constraints; split into S_Y, S_A.

13. **Decoder calibration on MIMIC.** MAE ≈ 10 mmHg, r ≈ 0.32 is weak. Either retrain / recalibrate (heteroscedastic head, quantile regression, ensembling) or explicitly scope MIMIC as methodological.

14. **Add k-sensitivity experiment** (k ∈ {10, 50, 100, 500, 1000}) showing how the RA advantage scales.

15. **Reproducibility.** State code availability in Section 1 or the checklist. Include seeds, hyperparameters, hardware.

### P3 — Polish (shows attention to detail)

16. Clarify "15–45% of patients under moderate confounding" in Section 1 contributions (add the γ range inline).
17. Rephrase "up to 24 percentage points across all five baseline models" to remove the "up to" + "all" ambiguity.
18. Rephrase "explicitly noted in the VCIP peer review process" (Section 1) to remove meta-referee commentary.
19. One sentence in Section 4 explaining why ε_VI decreases with γ (currently only in appendix).
20. Corollary 1: add one sentence on why Cancer is effectively deterministic.
21. Remark 2 (Bernstein): replace the implicit quadratic with a closed-form approximation for readability.
22. Table 1: ensure "ELBO Safe" and "ELBO In-T" columns are not accidental duplicates; add a note if they coincide by construction.
23. Verify Figure 1 renders correctly (LaTeX path currently relative to external repo).

---

## Recommended Experiment Plan

### New experiments (ordered by priority)

| # | Experiment | Purpose | Estimated effort | Status |
|---|-----------|---------|------------------|--------|
| E1 | Cancer model-predicted RA with decoder modifications (heteroscedastic, wider, MC-dropout). | Demonstrates deployable (non-oracle) RA filtering on Cancer. | Medium. | ✗ **FAILED (2026-04-09).** All 3 variants show ~0% model feasibility. Root cause: action-invariant latent space. Activates risk mitigation: reframe oracle/model gap as diagnostic finding. |
| E2 | MIMIC matched-treatment subgroup analysis. | Outcome-grounded validation on real data. | Low. | ✓ **DONE (2026-04-09).** Observed in-target 42-50% (not 99-100%). RA-matched +2-3pp. Paper text updated (2026-04-10). |
| E3 | Adaptive-λ (PID-Lagrangian) and midpoint-target-match baselines on Cancer + MIMIC. | Credible Pareto-dominance claim. | Medium. | ✓ **DONE (2026-04-10).** PID-Lagrangian equivalent to hard filter. Midpoint promoted to main text. Fixed-λ strictly dominated. |
| E4 | k-sensitivity sweep (k ∈ {10, 50, 100, 500, 1000}) on Cancer at γ = 4. | Characterizes scaling. | Medium. | ✓ Done (2026-03-27). |
| E5 | Third clinician rater on MIMIC 27 discordant cases (+ optionally expand to ≥50 cases). | Addresses fragile pooling. | High (external). | Dropped. 2-rater evaluation with effect sizes (d=0.69, κ=0.49) reframed as suggestive. |
| E6 | Decoder variance-calibration on MIMIC (heteroscedastic / quantile head). | Reduces MAE and gives genuine RA discrimination on real data. | High. | **Deprioritized** — E1 failure shows decoder-level fixes don't address the action-invariant latent space bottleneck. |
| E7 | Rater-random-effects statistical re-analysis of existing clinician data. | Tightens the significance claim. | Low. | ✓ **DONE (2026-04-10).** Per-rater Cohen's d + bootstrap CIs + κ=0.49 in paper. |
| E8 | Scatter / histogram of per-pair m_RA vs. m_ELBO margins on Cancer. | Empirical validation of Theorem 1(c). | Low. | Droppable — quantitative Theorem 1(c) already in paper. |
| E9 | Tighter sensitivity bound for long τ. | Addresses Γ_max = 2.1 weakness. | Medium. | ✓ **DONE (2026-04-09).** Time-invariant bound: Γ_max 2.1→65.7 at τ=8. MSM assessed; layered approach adopted. |

### Experiments to drop or de-emphasize

- The "5 baseline models × oracle RA" in Table 3 can stay, but must be flagged as oracle-based. **E1 failed — cannot run with model-predicted trajectories. Flag as oracle-only throughout.**
- Glucose-insulin appendix: keep as-is (cross-domain demonstration, +0.8-2.0pp). Do not overstate.
- E6 (MIMIC decoder recalibration): deprioritized — E1 failure shows the bottleneck is upstream of the decoder.

---

## Rewriting Strategy — STATUS: ALL MAJOR ITEMS COMPLETE (2026-04-10)

### Abstract ✓
- [x] MIMIC "99-100%" replaced with matched-treatment language (+2-3pp observed)
- [x] "Under oracle trajectories" qualifier added to Cancer numbers
- [x] Time-invariant sensitivity bound highlighted (Γ_max > 30)
- [x] Model-predicted trajectory finding + deployment diagnostic added

### Section 1 (Introduction) ✓
- [x] Three-regime framework promoted to oracle-vs-model section in experiments
- [x] "First reach-avoid safety evaluation of variational counterfactual planners" (precise scope)
- [x] γ ∈ [2,4] inline for "15-45%" figure
- [x] Contribution 2: Pareto-optimality + ranking robustness
- [x] Contribution 3: time-invariant bound (Γ_max > 30)
- [x] Contribution 4: oracle/model diagnostic + matched-treatment + clinician eval

### Section 4 (Theory) ✓
- [x] Pareto-optimality theorem (Proposition 4) as new subsection
- [x] Theorem 1(c) quantitative: m_RA ≥ 1 - 2β
- [x] Time-invariant sensitivity bound (Proposition 5) — eliminates τ-dependence
- [x] ε_VI decreases with γ explanation added

### Section 6 (Experiments) ✓
- [x] Oracle-vs-model section rewritten with decoder mod findings + model feasibility diagnostic
- [x] Midpoint baseline promoted to main text
- [x] PID-Lagrangian + fixed-λ comparison in soft constraint section
- [x] MIMIC results rewritten with matched-treatment subgroup analysis
- [x] Clinician table updated with per-rater effect sizes, bootstrap CIs

### Section 7 (Discussion) ✓
- [x] Oracle-vs-model gap + conditions for deployable RA
- [x] Limitations updated with decoder mod negative result
- [x] Reproducibility statement added

---

## Timeline Estimate — ACTUAL (2026-04-10)

- **P0 fixes (items 1–2):** Completed 2026-04-09 to 2026-04-10 (~2 days). Decoder mods ran on Vast.ai ($2); text edits same day.
- **P1 fixes (items 3–7):** Completed 2026-04-10 (~1 day). Pareto theorem, PID-Lagrangian, midpoint promotion, rater re-analysis, re-scoping, references — all done.
- **P2 fixes (items 8–15):** Mostly complete 2026-04-10. Remaining: per-pair scatter (droppable), cost-aware metric (droppable), MIMIC decoder (deprioritized). All others done.
- **P3 fixes (items 16–23):** Complete 2026-04-10. Bernstein simplified, all text fixes applied.
- **Actual total:** ~2 days of focused effort for P0+P1+most P2+most P3.

---

## Re-review Checkpoints — STATUS

- **Checkpoint 1 — after P0 fixes:** ✓ PASSED (2026-04-10). Oracle relabeling, MIMIC rewrite, narrative reframe all complete.
- **Checkpoint 2 — after P1 fixes:** ✓ PASSED (2026-04-10). Pareto-optimality theorem landed. PID-Lagrangian baseline confirms Pareto-dominance. Sensitivity bound tightened. **Estimated rating: ~7.0.** Proceeding toward NeurIPS submission.
- **Checkpoint 3 — after P2+P3 polish:** ✓ MOSTLY PASSED (2026-04-10). S-set notation, code availability, table audit, figure verification, P3 text fixes all done. Paper: 41 pages, clean compile.
- **Submission-readiness gate:** **PASSED.** Item 3 (new theorem) delivered as Proposition 4 (Pareto-optimality) + Proposition 5 (time-invariant sensitivity). Continuing toward NeurIPS.

---

## Risk Register

| Risk | Probability | Mitigation | Status |
|------|-------------|------------|--------|
| E1 fails — no decoder modification recovers Cancer RA discrimination. | ~~Medium.~~ | Pre-commit to reframing the paper as "when does RA filtering help? a diagnostic-based analysis," making the oracle/model gap the contribution. | **MATERIALIZED (2026-04-09).** All 3 decoder mods fail. Root cause: action-invariant latent space (KL≈10⁻⁵). Reframing required. |
| Item 3 (new theorem) cannot be delivered in time. | ~~Medium.~~ | Fall back to strengthening empirical contributions (E3, E6) and re-target to AISTATS / CHIL. | **RESOLVED (2026-04-10).** Pareto-optimality theorem (Proposition 4) + time-invariant sensitivity bound (Proposition 5) + quantitative Theorem 1(c). Three new theoretical artifacts delivered. |
| E5 (third rater) not feasible within timeline. | Medium-high. | Rely on item 4(b), rater-random-effects reanalysis, plus explicit reframing to "suggestive." | Open. |
| Decoder retraining (E6) makes MIMIC results worse. | ~~Low-medium.~~ | ~~Keep current decoder as baseline; present recalibrated decoder as ablation.~~ | **Deprioritized** — E1 failure shows decoder-level fixes don't address the bottleneck. |
