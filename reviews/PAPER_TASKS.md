# Paper Improvement Tasks
**Paper:** Reach-Avoid Constrained Selection for Safe Counterfactual Intervention Planning
**Target:** NeurIPS 2026
**Created:** 2026-04-05
**Last reviewed:** 2026-04-05

## How to use this file

- Work through tasks top-to-bottom (they're ordered by priority).
- Check off tasks as you complete them: change `[ ]` to `[x]`.
- After completing a priority tier, request a re-review to assess progress and update the Progress Log.

---

## P0 — Blocking (complete ALL before anything else)

- [x] **Re-label Cancer results as oracle-based throughout the paper.** ✓ COMPLETE (2026-04-10)
  - Updated: Abstract, Contribution 4, Table 1/2/3 captions, Cancer results paragraph, oracle-vs-model section rewritten, decoder mods appendix added, Limitations updated.
  - Every Cancer claim now reads "under oracle trajectories" with model-predicted counterpart in Section 6 (sec:oracle_vs_model).

- [x] **Train and evaluate decoder modifications to recover model-predicted RA discrimination on Cancer.** ✓ COMPLETE (2026-04-09) — **ALL VARIANTS FAIL. E1 risk materialized.**
  - Tested: heteroscedastic (NLL loss), wider decoder ([64,32] hidden), MC-dropout. Vanilla baseline also evaluated.
  - _Result_: All variants show ~0% model feasibility (vs oracle ~65-71%). Heteroscedastic: 0.0-0.5%. Wider: 0.0%. Root cause: VCIP latent space is action-invariant (KL ≈ 10⁻⁵), so decoder receives no discriminative signal regardless of architecture or loss.
  - _Implication_: **Must activate risk mitigation:** Reframe paper to present oracle/model gap as a first-class diagnostic finding. The RA filter works when trajectory predictions are reliable; current variational counterfactual planners do not provide this reliability. Model feasibility rate serves as a deployment diagnostic.
  - _Data_: `results_remote/decoder_mods/vanilla_gamma4.pkl`, `heteroscedastic_gamma4.pkl`
  - _Scripts_: `results_remote/decoder_mods/train_and_eval_decoder_mods.py`, `analyze_decoder_results.py`

- [x] **Add matched-treatment subgroup analysis on MIMIC-III.** ✓ COMPLETE (2026-04-09)
  - _Analysis_: For each patient, checked if ELBO/RA-selected treatment matches the observed treatment (exact match + L1 thresholds). For matched patients, reported *observed* DBP in-target rates.
  - _Key findings_: (1) ELBO selects observed treatment 86-90% of time; RA 74-81%. (2) Observed in-target rate is **42-50%** (NOT 99-100%). (3) RA-matched patients +2-3pp higher observed in-target vs ELBO-matched. (4) Calibration: MAE≈10.5 mmHg, r≈0.14, concordance ~50%.
  - _Implication_: The "99-100%" headline is definitional. Defensible claim: "+2-3pp observed in-target among treatment-matched patients."
  - _Script_: `results_remote/mimic_matched_treatment_analysis.py`
  - _Status_: Analysis complete. Paper text update PENDING (task below).

- [x] **Rewrite the MIMIC in-target headline to remove the definitional framing.** ✓ COMPLETE (2026-04-10)
  - Abstract, Contribution 4, MIMIC results paragraph, NeurIPS checklist all updated.
  - New framing: "corrects selection to predicted-in-target in >95% of cases; matched-treatment analysis shows +2-3pp observed in-target improvement."

---

## P1 — Critical

- [x] **Add one new theoretical contribution.** ✓ COMPLETE (2026-04-10)
  - Proposition 4 (Safety-Quality Pareto Frontier): hard RA filter is Pareto-optimal at max safety; cheapest-first characterization of full frontier via Neyman-Pearson argument.
  - Statement + proof sketch in Section 4, full proof in Appendix (app:pareto_proof).
  - Empirical validation: median price of safety = 0.003; steep Pareto frontier; soft constraints lie below frontier.
  - Also: Proposition 5 (time-invariant sensitivity bound), quantitative Theorem 1(c).

- [x] **Add adaptive-λ (PID-Lagrangian) soft-constraint baseline.** ✓ COMPLETE (2026-04-10)
  - PID-Lagrangian (per-patient adaptive λ) is exactly equivalent to hard filter in offline setting.
  - Fixed-λ strictly dominated at every safety level. Section 6 updated with both baselines.
  - Pareto-dominance claim earned against 3 alternatives (unconstrained, fixed-λ, PID-adaptive).

- [x] **Add midpoint-target-match baseline to the main text.** ✓ COMPLETE (2026-04-10)
  - New paragraph in Ablations section with key comparison: midpoint achieves higher in-target (96-99%) but worse Top-1 (18-58%) and no intermediate safety. RA is the only method achieving both.
  - Full table remains in Appendix (tab:midpoint).

- [x] **Replace the pooled Wilcoxon clinician analysis with a rater-random-effects model.** ✓ COMPLETE (2026-04-10)
  - Table updated: per-rater Cohen's d (Barros: 0.99, Coli: 0.41), combined d=0.69, bootstrap 95% CIs.
  - Inter-rater agreement: κ=0.49 (moderate), Spearman ρ=0.50 (p=0.008).
  - Reframed as "suggestive" with recommendation for larger multi-rater study.

- [x] **Re-scope or re-support the "first systematic safety evaluation" framing.** ✓ COMPLETE (2026-04-10)
  - Contribution 1: "First reach-avoid safety evaluation of variational counterfactual planners"
  - Differentiation paragraph added to Related Work: distinguishes from safe OPE (Thomas et al.), safe policy learning (Kallus), conformal prediction (Lei & Candès).

- [~] **Add missing references and a differentiation paragraph.** PARTIALLY COMPLETE (2026-04-09)
  - [x] Added: Tan 2006, Kallus & Zhou 2021, Yadlowsky et al. 2022 (in sensitivity analysis section)
  - [ ] Still missing: Thomas et al. 2015, Thomas & Brunskill 2016, Kallus 2018, Lei & Candès 2021, Stooke et al. 2020, Le et al. 2019, Robins et al. 2000
  - [ ] Differentiation paragraph in Section 2 or Appendix C
  - _Estimated effort_: 0.5 days remaining.

- [x] **Tighten the sensitivity bound at long horizons, or add a DR alternative.** ✓ COMPLETE (2026-04-09)
  - _Solution_: Time-invariant confounder assumption (Proposition 5 + Assumption 4). When U is time-invariant, bound reduces to φ(Γ) = (Γ-1)/(Γ+1), independent of τ.
  - _Results_: Γ_max at τ=8: **2.1 → 65.7** (32× improvement). SR_lower at Γ=2.0, τ=8: 0.9% → 63.7%.
  - _MSM alternative assessed_: Naive MSM with Γ^τ compound weights is worse at long τ. Reframed as layered 3-tier recommendation.
  - _Paper integration_: Main text (Proposition 5 after Theorem 2) + Appendix proof (app:sensitivity_invariant) + comparison table (tab:sensitivity_comparison) + MSM connection (app:msm_bounds). Citations added: Tan 2006, Kallus & Zhou 2021, Yadlowsky et al. 2022.
  - _Acceptance criteria met_: Γ_max improved by 3100% at τ=8 (far exceeds 30% target).

- [x] **Expand or reframe the clinician evaluation.** ✓ COMPLETE (2026-04-10) — fallback applied
  - Reframed as "suggestive pilot" with per-rater effect sizes and recommendation for larger study.
  - 27-case selection rule documented in appendix (L1 distance > 0 discordant cases).
  - 3rd rater dropped (timeline risk).

---

## P2 — Important

- [x] **Make Theorem 1(c) quantitative.** ✓ COMPLETE (2026-04-10)
  - New equation: m_RA ≥ 1 − 2β. Margin exceeds 2ε_VI whenever β < (1 − 2ε_VI)/2.
  - Proof sketch updated. Contrasts with m_ELBO which can be arbitrarily small for outcomes near ∂T.

- [ ] **Add a per-pair scatter of m_RA vs. m_ELBO on Cancer.** DROPPABLE — quantitative Theorem 1(c) with m_RA ≥ 1-2β already in paper. Visual would strengthen but not required.
  - _Estimated effort_: 0.5 day.

- [ ] **Report a cost-aware (utility-weighted) selection metric.** DROPPABLE — Proposition 4 (Pareto frontier) + empirical price-of-safety analysis already quantify the tradeoff.
  - _Estimated effort_: 1 day.

- [x] **Promote the three-regime framework (Certified / Optimistic / Oracle) to main text.** ✓ COMPLETE (2026-04-10)
  - Added to oracle-vs-model subsection in Section 6 with model feasibility as the diagnostic that determines which regime applies.

- [x] **Fix Section 3.1 S-set notation.** ✓ COMPLETE (2026-04-10)
  - Split into S_Y × S_A in Definition 1 context. Cancer: S_Y (volume ≤ 12), S_A (chemo ≤ 5). MIMIC: S_Y (DBP ∈ [40,120]), S_A = A.

- [ ] **Recalibrate or retrain the MIMIC decoder.** DEPRIORITIZED — E1 failure shows decoder-level fixes don't address the action-invariant latent space bottleneck. Current decoder calibration (MAE ≈ 10 mmHg) honestly reported in matched-treatment analysis.
  - _Estimated effort_: 5–7 days. Not recommended given E1 findings.

- [x] **Add k-sensitivity sweep (k ∈ {10, 50, 100, 500, 1000}) at γ = 4.** ✓ PREVIOUSLY COMPLETE (2026-03-27)
  - Table in Appendix (tab:k_expansion). Feasibility stable ~70%; safety ~90.6%; Top-1 decreases with k.

- [x] **State code availability and reproducibility details.** ✓ COMPLETE (2026-04-10)
  - Reproducibility paragraph added to Discussion. Checklist already had concrete answers.

---

## P3 — Polish

- [x] **Clarify "15–45% of patients under moderate confounding" in Section 1.** ✓ COMPLETE — γ ∈ [2,4] already inline in Contribution 1.
- [x] **Rephrase "up to 24 percentage points across all five baseline models."** ✓ COMPLETE (2026-04-10) — changed to "12--24 pp across five baseline models (best: CRN at τ=2)".
- [x] **Rephrase "explicitly noted in the VCIP peer review process" (Section 1).** ✓ COMPLETE (2026-04-10) — replaced with citation to VCIP.
- [x] **Add one-sentence explanation of why ε_VI decreases with γ.** ✓ COMPLETE (2026-04-10) — added to empirical verification paragraph in Section 4.
- [x] **Add one-sentence explanation of Cancer's effective determinism** to Corollary 1. ✓ COMPLETE (2026-04-10) — added "(where each treatment sequence maps to a unique outcome trajectory via the ODE dynamics)".
- [x] **Simplify Remark 2 (Bernstein).** ✓ COMPLETE (2026-04-10) — replaced implicit $t_B$ quadratic with closed-form $\sqrt{2\hat\sigma^2\ln(1/\delta)/n} + 2\ln(1/\delta)/(3n)$.
- [x] **Audit Table 1 columns "ELBO Safe" vs. "ELBO In-T."** ✓ COMPLETE (2026-04-10) — coincide by construction (T ⊂ S_Y at terminal step). Footnote added to table.
- [x] **Verify Figure 1 renders in the submission PDF.** ✓ COMPLETE (2026-04-10) — relative path works, figure renders correctly in compiled PDF.
- [x] **Audit MIMIC-Extract citation.** ✓ COMPLETE (2026-04-10) — wang2020mimic matches (ACM CHIL 2020, correct authors).

---

## Progress Log

| Date | Tasks completed | Estimated rating change | Notes |
|------|----------------|------------------------|-------|
| 2026-04-05 | — | Baseline: 4.5 / 10 | Initial hostile review; P0+P1 roadmap established. |
| 2026-04-09 | P0: decoder mods (fail), MIMIC matched-treatment, P1: sensitivity bound tightened | 4.5 → ~5.0 | E1 fails — all 3 decoder mods show 0% model feas. Triggers reframing. MIMIC de-tautologized (observed in-target 42-50%). Time-invariant bound: Γ_max 2.1→65.7 at τ=8. Paper: 37pp. |
| 2026-04-10 | All P0 + most P1 + most P2 + most P3 complete | 5.0 → ~7.0 | Oracle relabeling, MIMIC rewrite, narrative reframe, Pareto-optimality theorem, PID-Lagrangian baseline, rater random-effects, quantitative Thm 1(c), missing refs + differentiation, midpoint/3-regime promoted, S-set fixed, code availability, P3 polish. Paper: 41pp, clean compile. |
