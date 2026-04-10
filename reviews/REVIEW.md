# Hostile Review: Reach-Avoid Constrained Selection for Safe Counterfactual Intervention Planning

**Target conference:** NeurIPS 2026
**Review date:** 2026-04-05
**Reviewer stance:** Hostile (but constructive) peer reviewer simulation

---

## Summary

The paper proposes a post-hoc safety filter for variational counterfactual planners (specifically VCIP). Given k candidate treatment sequences ranked by ELBO, the method discards sequences whose predicted terminal outcome misses a target set T, or whose predicted intermediate outcomes leave a safety set S, and then picks the best-ELBO survivor. The authors contribute (i) a coarsening-based robustness theorem arguing RA ranking is more robust than point-target ELBO ranking to the variational gap, (ii) a sequential Rosenbaum-style sensitivity bound with breakdown points Γ_max = 2.1–7.5, (iii) Hoeffding/Bernstein deployment bounds, and (iv) experiments on a cancer simulator, MIMIC-III, and a glucose-insulin simulator, plus a blinded evaluation by two ICU intensivists on 27 discordant cases.

---

## Strengths

1. **Clear, well-motivated problem.** The gap between point-target ELBO optimization and clinical range-plus-safety objectives is real and under-addressed. The framing is honest ("deliberately simple").
2. **Honest self-assessment.** The paper acknowledges several inconvenient findings: the model-predicted trajectory ablation on Cancer reveals the filter is a near-no-op with a realistic decoder; conformal certificates are vacuous; the clinician study raises construct-validity concerns.
3. **Readable writing, coherent structure.** Contributions are enumerated, theorems are stated cleanly, and the relation to VCIP is clearly demarcated.
4. **Theoretical scaffolding beyond raw empirics.** Robustness, sensitivity, finite-sample, and conformal strands together reflect serious effort to go beyond "we ran an experiment."
5. **Clinician evaluation included.** Involving two independent ICU intensivists, even in a small study, is rare for ML papers and substantively strengthens the MIMIC-III narrative.
6. **Model-agnostic evaluation across 5 baselines.** Demonstrating that the filter helps weaker baselines more than VCIP is an interesting, testable prediction.

---

## Weaknesses

Each weakness is tagged with priority: **P0** (blocking / must fix), **P1** (critical / strongly affects rating), **P2** (important / lose 1–2 points), **P3** (polish).

### [P0] 1. Headline Cancer numbers rely on oracle trajectories, not a deployable pipeline

- **What:** The main Cancer results (Tables 1, 2, and the model-agnostic Table 3) feed *simulator-oracle* outcome trajectories into the RA filter. Section 6.6 ("Oracle vs. model-predicted trajectories") admits that with model-predicted trajectories, feasibility is >98% (filter is a no-op) because VCIP's decoder output range ([-0.4, 3.9] cm³) is compressed relative to true volumes ([0, 35] cm³).
- **Why it matters:** In any realistic deployment there is no oracle simulator; the filter must use the model's own predictions. So the reported +9–11 pp safety on Cancer, and the +13–15 positions of baseline improvement in Table 3, describe an *oracle-assisted* evaluation that cannot be achieved with the current model. A sympathetic reviewer will call this a simulator-specific result; a hostile reviewer will call it misleading.
- **Where:** Section 6.1–6.3 (main results) vs. Section 6.6 (buried in ablations); Abstract and Contributions do not flag this dependence.
- **How to fix:**
  (a) Re-label the Cancer oracle numbers as *oracle safety* everywhere they are reported (Abstract, Section 1 contributions, Section 6.2, Table 1, Table 3).
  (b) Report model-predicted RA numbers side-by-side in Table 1 (even if near-no-op), making the oracle-vs-model gap a first-class finding rather than an appendix note.
  (c) Demonstrate at least one decoder modification (wider output range, variance calibration, or predictive MC dropout) that recovers useful discrimination on Cancer. Without (c), the paper's core story on Cancer evaporates once a practitioner tries to use the method.
  (d) Move the three-regime framework (Certified / Optimistic / Oracle) from the appendix into Section 4 or 5.

### [P0] 2. MIMIC in-target gains are close to definitional without stronger outcome validation

- **What:** On MIMIC-III the filter selects candidate plans whose *predicted* terminal DBP lies in [60, 90] and whose *predicted* intermediate DBPs lie in [40, 120]. "In-target rate" is then computed against predicted DBP. You filter by predicted DBP and report improvements in predicted DBP. The only outcome-grounded evidence is MAE ≈ 10 mmHg with Pearson r ≈ 0.32 — which makes the 99–100% in-target claim look tautological.
- **Why it matters:** NeurIPS reviewers will zero in on this. The correct claim is about a counterfactual outcome that is unobservable on MIMIC-III; the paper's observed "lift" is largely a consequence of the filtering criterion and the predictor's inductive bias, not a demonstrated clinical improvement.
- **Where:** Table 5, Section 6.4, Abstract ("lifts in-target diastolic blood pressure rates from 73–81% to 99–100%").
- **How to fix:**
  (a) Abstract/Section 6.4 must explicitly state: "in-target" = predicted terminal DBP, not observed, and that RA is optimizing against the same predictor used to evaluate.
  (b) Add an out-of-sample calibration check: for the subset of patients where the *observed* outcome at horizon τ is present, report observed in-target rates for ELBO-selected vs. RA-selected treatment sequences where the selected sequence matches the observed one (matched treatment analysis). Otherwise you are not measuring selection quality at all.
  (c) Consider reporting guideline-concordance metrics (appendix D.4) or treatment-appropriateness metrics in the main text as a more credible signal.

### [P1] 3. Clinician evaluation: fragile pooling, low agreement, small n

- **What:** Only 27 discordant cases, two raters, Rater 2 alone is not significant (p = 0.064). The headline p < 0.0001 pooled test treats 36 rater-case decisions as independent. Cohen's κ = 0.26–0.38 (fair) for ratings and κ = 0.49 (moderate) for preference. Rater 1 rated 81% of ELBO plans as "dangerous," which is an extreme calibration that drives most of the pooled statistic.
- **Why it matters:** The 83% preference, p < 0.0001 claim is repeated in Abstract, Section 1, Section 6.4, and discussion. Reviewer 2 will note (correctly) that pooling under clear miscalibration + fair agreement is statistically fragile, and that the one "independent" rater (#2) does not individually clear p = 0.05. The clinician feedback itself (missing arterial blood gas, comorbidities, questionable DBP as target) undermines construct validity.
- **Where:** Section 1 (contribution 4), Section 6.4, Appendix D.5.
- **How to fix:**
  (a) Use a mixed-effects / rater-random-effects model (e.g., GEE or a random-intercept logistic regression) that respects paired clustering and reports effect size rather than a pooled Wilcoxon.
  (b) Pre-register or justify the 27-case selection criterion and report sensitivity of the p-value to rater inclusion.
  (c) De-emphasize the pooled "83% / p < 0.0001" as the headline; lead with Rater-1-specific and Rater-2-specific results and discuss their differences honestly.
  (d) Consider expanding to ≥3 raters, or at minimum acknowledge that the current evaluation provides *suggestive* not *confirmatory* evidence.

### [P1] 4. Novelty framing is vulnerable to the "this is a rejection rule" critique

- **What:** The method is a hard post-hoc filter composed with ELBO ranking. The theorems are straightforward applications of TV bounds on bounded functionals (P ∈ [0,1]), plus a sequential coupling of Rosenbaum-style density ratios, plus Hoeffding/Bernstein. The paper itself calls the method "deliberately simple."
- **Why it matters:** NeurIPS reviewers who have seen shielding in safe RL, rejection sampling in RLHF, and safe off-policy evaluation literature will ask: "What is technically new here?" The paper will read to a hostile reviewer as an application / system paper rather than a methodological contribution.
- **Where:** Section 3 (Method), Section 4 (Theory), Section 1 (Contributions). The "first systematic safety evaluation" framing is thin as a novelty claim.
- **How to fix:**
  (a) Sharpen the novelty claim: re-frame the contribution as a *formalization* that ties reach-avoid control theory to variational counterfactual planning via a coarsening argument that gives *strictly tighter* ranking preservation than ELBO. Emphasize Theorem 1(c) (strict improvement) and prove a *lower bound* on improvement, not just a qualitative comparison.
  (b) Compare against an explicit shielding baseline from safe RL (Alshiekh et al. 2018) and a rejection-sampling-by-target baseline (top-K candidates whose predicted Y is nearest the target midpoint) — two credible alternatives that reduce to trivial versions of the proposed method.
  (c) Consider deriving a *Pareto frontier characterization*: what is the optimal safety/Top-1 tradeoff achievable by any post-hoc filter? The paper currently shows empirical Pareto-dominance over one soft-Lagrangian baseline; a theoretical optimality statement would substantially strengthen the methodological claim.

### [P1] 5. Soft-constraint baseline is weak

- **What:** The "hard vs. soft" comparison in Section 6.5 tests only a fixed-λ Lagrangian. Modern constrained optimization baselines — adaptive Lagrangian (PID updates, Stooke et al.), trust-region CPO (Achiam et al.), reward-constrained policy optimization (Tessler et al.), or simple logarithmic barriers — are not compared. The claim "Hard filtering Pareto-dominates soft constraints" against a single static-λ sweep (λ ∈ {0.01, …, 50}) is overclaimed.
- **Why it matters:** Pareto-dominance claims require a wider set of alternatives. This is one of the two places where the paper argues its method is not just simpler but *better*; on current evidence that claim is weakly supported.
- **How to fix:** Add ≥2 additional baselines (adaptive-λ, a top-K + target-match selector) and report the Pareto frontier with confidence bands. If hard filtering still dominates, the claim is earned; if not, the claim must be softened.

### [P1] 6. Sensitivity bound Γ_max = 2.1 at τ = 8 is exactly at the threshold of clinical realism

- **What:** Theorem 2 gives non-vacuous guarantees only for Γ < Γ_max, with Γ_max = 2.1 at τ = 8 on Cancer. Section 6 claims "non-vacuous bounds through Γ = 2.0 at all tested horizons" — this is true but a sympathetic reader would note this margin is 0.1, not a comfortable buffer.
- **Why it matters:** In clinical practice, unobserved confounding magnitudes of Γ ∈ [2, 4] are not unusual (see e.g., Ichino et al. and the sensitivity-analysis literature). At τ = 8 the theory is essentially silent at realistic confounding. Reviewers will ask whether the sensitivity result is actually useful for deployment certification.
- **How to fix:**
  (a) Explicitly state the implied "safe confounding regime" per horizon in the main text.
  (b) Derive a *tighter* bound that exploits step-dependent confounding or shared latent structure (current bound uses per-step independence of the density-ratio tilt, which is worst-case).
  (c) Add a distributionally-robust alternative (Yadlowsky et al., Kallus & Zhou) as a comparison or as the primary tool for long horizons.

### [P1] 7. Related work gaps and potentially contested "first" claims

- **What:** The paper positions itself as "the first systematic safety evaluation of counterfactual treatment planners." The related-work section discusses safe RL broadly but under-cites:
  - **High-confidence off-policy evaluation / safe policy improvement** (Thomas, Theocharous & Ghavamzadeh 2015; Thomas & Brunskill 2016).
  - **Safe policy learning with observational data** (Kallus 2018; Jiang & Li 2016).
  - **Rosenbaum sensitivity for sequential / longitudinal treatments** (Yadlowsky et al. 2022; Kallus & Zhou 2020; Dorn & Guo 2022; Marmarelis et al. 2023).
  - **Conformal counterfactual prediction** (Lei & Candès 2021; Yin et al. 2022; Chernozhukov et al. 2023).
  - **Constrained OPE / constrained contextual bandits** (Le et al. 2019; Satija et al. 2020).
- **Why it matters:** Any of these authors, if they review, will challenge the "first" claim. Even setting authorship aside, a hostile reviewer will say "prior safe-policy-improvement work already solves a version of this."
- **How to fix:** Replace "first systematic safety evaluation of counterfactual planners" with a more precise claim (e.g., "first reach-avoid-structured evaluation of variational counterfactual *planners* (a sequence-selection rather than policy-learning formulation)"). Add missing citations and a short differentiation paragraph.

### [P2] 8. Theorem 1's "strict improvement" result is qualitative, not quantitative

- **What:** Part (c) says m_RA ≫ m_ELBO / C_d "when boundary mass β is small." This is informal; there is no quantitative ratio expressed in terms of β, no lower bound on pairs strictly helped by RA, and no empirical histogram of margins to validate the mechanism.
- **How to fix:** Provide an explicit inequality of the form m_RA ≥ (1 − 2β) − (...) and contrast with the ELBO margin. Add a scatter plot of m_RA vs. m_ELBO per pair on Cancer.

### [P2] 9. Missing midpoint / nearest-neighbor-to-target baseline in main text

- **What:** A trivial alternative is: replace VCIP's point target Y* with the midpoint of T and pick the candidate whose predicted Y_{t+τ} is nearest. This removes the need for RA filtering entirely. Appendix N mentions a "midpoint baseline"; it should be in the main text since it is the most natural straw-man.
- **How to fix:** Promote the midpoint baseline to Table 1/4 with explicit safety / Top-1 comparison.

### [P2] 10. Top-1 degradation of up to 15 pp is non-trivial

- **What:** Table 1 shows Top-1 drops of 1.4–15.4 pp under RA filtering on Cancer. At τ = 8, RA loses 15.4 pp of Top-1 to gain 11.4 pp of safety. In a setting where clinicians care about *selecting the oracle-optimal plan*, this is a steep cost.
- **How to fix:** Discuss explicitly when the tradeoff is worth it. Provide a cost-aware metric (e.g., expected clinical utility with a safety weight) and show RA optimality under that metric.

### [P2] 11. MAE ≈ 10 mmHg, Pearson r ≈ 0.32 raises questions about the trained model

- **What:** The MIMIC VCIP decoder achieves Pearson r ≈ 0.32 with observed DBP. That is weak. The paper's "RA filtering is mechanically effective but clinically limited by model prediction quality" is a polite way of saying the upstream model is not good enough for the downstream selection to be meaningful.
- **How to fix:** Either retrain / recalibrate the decoder until predictions are actionable, or frame the MIMIC experiment explicitly as a *methodological demonstration* rather than a clinical result.

### [P2] 12. Deliberate simplicity vs. NeurIPS expectations

- **What:** NeurIPS 2026 guidelines emphasize "originality: does the work provide new insights, deepen understanding?" and "quality: is the submission technically sound and complete?" A deliberately simple method with mostly standard theory risks being seen as "not enough for a NeurIPS main-track paper."
- **How to fix:** Two strategies:
  (i) Lean harder into the *theoretical* contribution by adding at least one non-trivial, surprising result (e.g., an optimality characterization of post-hoc filters, a matching lower bound, or a precise characterization of when soft beats hard).
  (ii) Consider re-targeting to a venue where clinical-ML applications are weighed more heavily (e.g., CHIL, ML4H, AAAI HAI, AISTATS) if item (i) cannot be delivered.

### [P3] 13. Minor presentation issues

- Section 1: "15–45% of patients under moderate confounding" is claimed but the exact γ range and operationalization of "moderate" is not in the main text (later clarified in Section 6, but the reference in the contributions list is vague).
- Abstract: "up to 24 percentage points across all five baseline models" uses "up to" + "all" ambiguously; reads as if all baselines improve by 24 pp.
- Figure 1: Not in this repo at the referenced LaTeX path — verify it renders in the submitted PDF.
- Algorithm 1: Step 3 says "simulator or model predictions" — given Weakness 1, this line is load-bearing and should be highlighted as a decision point with both regimes benchmarked.
- "This limitation was explicitly noted in the VCIP peer review process" (Section 1) is awkward phrasing; a concrete citation or quote would be stronger.
- "$\varepsilon_{\mathrm{VI}}$ decreases with γ" is counterintuitive and deserves one sentence of explanation in Section 4 rather than just the appendix.
- Corollary 1: "For deterministic trajectories …, feasibility is binary" — worth one sentence explaining why Cancer is effectively deterministic here (tumor-growth ODE) versus MIMIC's stochastic setting.
- Remark 2 (Bernstein): The bound tightening is nice but the presentation of t_B solving a quadratic equation is clunky; closed-form approximation would improve readability.

---

## Questions for Authors

1. **Model-predicted vs. oracle gap on Cancer.** If the filter is a no-op on Cancer with the current decoder, what happens to the +13–15 position rank improvements for baseline models in Table 3 under model-predicted trajectories? Do the baseline improvements vanish as well?
2. **Counterfactual validation on MIMIC.** What is the observed-DBP distribution for patients whose RA-selected sequence matches their observed treatment, vs. those where it doesn't? Can this subgroup analysis provide any outcome-grounded validation?
3. **Rater calibration sensitivity.** If Rater 1's "dangerous" ratings are re-scored as 2 rather than 1 (i.e., truncating the harsh tail), what happens to the pooled p-value?
4. **Choice of k = 100 candidates.** How does RA's benefit scale with k? At k = 10 does the filter still help? At k = 1000 does feasibility saturate?
5. **Multi-outcome MIMIC.** The clinicians suggest MAP + SpO₂ as co-targets. Did you attempt a 2-dimensional target set? If not, why?
6. **Adaptive λ comparison.** Why was the Lagrangian baseline run with fixed λ rather than with PID-adaptive dual updates?
7. **Γ_max at τ = 8 = 2.1.** Do you believe Γ ≤ 2 is a defensible assumption for vasopressor prescribing in ICU data? If not, what is the operational safety claim at τ = 8?
8. **Spearman ρ = 0 for CT.** If Causal Transformer's ELBO ranking is uncorrelated with ground truth, is the model broken or is the training target misaligned? RA filtering improving CT by 15 rank positions on a meaningless ranking is suspicious.
9. **Retraining null result.** RA-aware retraining "produces identical results" — what loss surface feature explains this? Does it indicate the model cannot exploit the RA signal, or that the filter is orthogonal to what the ELBO captures?
10. **Glucose-insulin improvements are tiny.** If the cross-domain improvement is +0–2 pp, does this cross-domain experiment undermine or support the main claim?

---

## Missing References

Priority citations to add:

- **Safe / high-confidence off-policy evaluation:**
  - Thomas, Theocharous, Ghavamzadeh (2015). *High-confidence off-policy evaluation.*
  - Thomas & Brunskill (2016). *Data-efficient off-policy policy evaluation.*
  - Jiang & Li (2016). *Doubly robust off-policy value evaluation.*
  - Kallus (2018). *Balanced policy evaluation and learning.*
- **Sensitivity analysis for sequential / hidden confounding:**
  - Yadlowsky, Namkoong, Basu, Duchi, Tian (2022). *Bounds on the conditional and average treatment effect with unobserved confounding.*
  - Kallus & Zhou (2020). *Confounding-robust policy evaluation in infinite-horizon reinforcement learning.*
  - Dorn & Guo (2022). *Sharp sensitivity analysis for inverse propensity weighting.*
  - Marmarelis, Steeg, Galstyan (2023). *Partial identification of dose responses with hidden confounders.*
- **Conformal counterfactual prediction:**
  - Lei & Candès (2021). *Conformal inference of counterfactuals and individual treatment effects.*
  - Yin, Shi, Candès, Wainwright (2022). *Conformal sensitivity analysis for individual treatment effects.*
  - Chernozhukov, Wüthrich, Zhu (2023). *Exact and robust conformal inference.*
- **Shielding & safe RL (already partially cited; expand):**
  - Alshiekh et al. (2018). *Safe reinforcement learning via shielding.* (cited — good)
  - Stooke, Achiam, Abbeel (2020). *Responsive safety in RL by PID Lagrangian methods.*
- **Constrained OPE:**
  - Le, Voloshin, Yue (2019). *Batch policy learning under constraints.*
  - Satija, Amortila, Pineau (2020). *Constrained Markov decision processes via backward value functions.*
- **Inverse probability weighting under sequential confounding (Robins):**
  - Robins, Hernán, Brumback (2000). *Marginal structural models and causal inference in epidemiology.*

---

## Minor Issues

- Abstract character count is high; consider tightening the sensitivity-result clause.
- "reach-avoid" hyphenation: consistent throughout — good.
- Equation (3): $g_{[\ell,u]}$ lacks subscript consistency with $g_\cT$, $g_\cS$ used elsewhere.
- Section 3.1: "$\cS \subseteq \R^d$" suggests a set of *outcome* vectors, but the Cancer spec mixes outcome (volume) and *treatment* (chemo dosage ≤ 5.0) in S. This conflates outcome and action safety — clarify notation or split S into S_Y and S_A.
- Table 1: columns "ELBO Safe" and "ELBO In-T" are identical in the printed values — if they coincide by construction here, note it; otherwise this looks like a copy/paste artifact.
- "MIMIC-Extract pipeline" citation [wang2020mimic] — verify that the preprocessing version used matches the one in the citation.
- Algorithm 1 step 3 "simulator or model predictions" — this branch determines whether results are oracle or deployable; should be a first-class input to the algorithm, not a comment.

---

## Overall Assessment

- **Predicted rating (NeurIPS 1–10 scale):** **4.5 / 10** (borderline reject — top 50% of rejected papers)
- **Confidence:** 4 / 5
- **Verdict:** Thoughtful, honest, well-written paper whose *core empirical result is oracle-dependent on Cancer and nearly-definitional on MIMIC*, and whose theoretical contribution is correct but incremental; needs either a methodological sharpening or a substantially strengthened empirical story to clear the NeurIPS bar.
- **Estimated acceptance probability at NeurIPS 2026 (current form):** ~15–20%.
- **Estimated probability after P0 fixes:** ~35–45%.
- **Estimated probability after P0 + P1 fixes:** ~55–65%.

### Path to acceptance

The paper has three credible routes to a strong accept:

1. **Empirical route:** Demonstrate a decoder modification that makes Cancer model-predicted RA effective, AND provide outcome-grounded validation on MIMIC (matched-treatment subgroup analysis + stronger clinician evaluation). Leads to +1.5–2.5 rating points.
2. **Theoretical route:** Add an optimality / Pareto-frontier characterization of post-hoc filters, with a matching lower bound or minimax argument. Leads to +1.5 rating points.
3. **Applied route (fallback):** Re-target to CHIL/ML4H/AISTATS, where the current contribution is a clear accept.

Route 1 is the most tractable given the existing infrastructure. Route 2 offers the highest ceiling for NeurIPS specifically.

---

*Sources consulted:*
- [NeurIPS 2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
- [NeurIPS 2025 Reviewer Guidelines](https://neurips.cc/Conferences/2025/ReviewerGuidelines)
- [NeurIPS 2026 Area Chair Pilot Blog](https://blog.neurips.cc/2026/03/23/refining-the-review-cycle-neurips-2026-area-chair-pilot/)
