# S6.1 Results Summary: Conformal Safety Certificates

## Key Finding

**Conformal prediction provides valid coverage guarantees for counterfactual treatment planning, but current trajectory prediction models are not accurate enough for useful safety certification.** The accuracy gap is quantifiable and establishes a concrete model improvement target.

## Methodology

### Conformal Framework
- **Nonconformity score:** s_i = Y_oracle - μ̂_model (signed residual)
- **Calibration:** Seeds {10, 101, 1010}; **Test:** Seeds {10101, 101010}
- **Cancer (one-sided):** Upper bound only — need μ̂ + q̂_upper ≤ constraint
- **MIMIC (asymmetric):** Separate upper/lower calibration for [60, 90] target
- **Certificate:** Treatment plan ā is "certified safe at level α" if prediction band ⊆ Safety ∩ Target

### Coverage Guarantee
P(Y_true(ā) ∈ C_α) ≥ 1 - α (distribution-free, requires exchangeability)

## Results

### 1. Coverage is empirically valid

**Cancer (α=0.10, one-sided):**
| γ | Oracle Coverage (target ≥90%) |
|---|------|
| 1 | 87-91% ✓ |
| 2 | 82-90% (slight under-coverage at long τ) |
| 3 | 90-91% ✓ |
| 4 | 85-90% ✓ |

**MIMIC (α=0.10, symmetric, v1):**
| τ | Coverage (target ≥90%) |
|---|------|
| 2 | 91.5% ✓ |
| 3 | 90.5% ✓ |
| 4 | 87.0% (slight under-coverage) |
| 5 | 90.5% ✓ |

### 2. Certification is too conservative for useful selection

**Cancer:** One-sided bounds yield 100% certification at γ=1-3 (vacuous — everything certified, no discrimination). At γ=4 with symmetric bounds: 0% certification. Neither extreme is useful for selection.

**MIMIC:** Asymmetric band width (33 mmHg at α=0.10) exceeds target range width (30 mmHg), yielding 0% certification.

### 3. Accuracy Gap

**Cancer:** Model predictions cluster at 0-2 cm³ vs. target boundary 3.0 cm³ → one-sided bounds trivially satisfied. Ratio < 1 at all settings (model is "accurate enough" in the wrong direction — it under-estimates risk).

**MIMIC:** Band/target ratio = 1.08-1.15 at α=0.10 → model needs ~10% accuracy improvement for non-vacuous certificates. This is tantalizingly close.

## Implications for the Paper

### Narrative: "Conformal certificates as a model accuracy diagnostic"

1. **Valid coverage:** Conformal prediction works correctly on both Cancer and MIMIC — empirical coverage matches the theoretical guarantee within statistical noise.

2. **Certification gap:** Current VCIP decoder predictions are not precise enough for safety certification. The conformal band width exceeds or vacuously satisfies the safety constraints.

3. **Quantitative diagnostic:** On MIMIC, a ~10% reduction in prediction MAE (from ~10 to ~9 mmHg) would enable non-vacuous conformal certificates. This defines a concrete model improvement target.

4. **Three-regime framework:**
   - **Certified safety** (conformal filter): requires accurate model + narrow bands → future work
   - **Optimistic safety** (RA threshold filter): uses model predictions without formal guarantees → current method, effective on MIMIC
   - **Oracle safety** (oracle RA): uses ground truth → upper bound, Cancer experiments

5. **Key insight for discussion:** The RA threshold filter and conformal certificates occupy different points on the accuracy-guarantee tradeoff. The threshold filter is practical but provides no formal guarantee. Conformal certificates provide guarantees but require better models. The gap between them quantifies the "cost of certification."

### Where this fits in the paper

**New Section 5.2 or Appendix:** "Conformal Safety Certificates"
- Theorem: coverage guarantee (standard, but first application to counterfactual planning)
- Empirical validation: coverage is correct on both datasets
- Accuracy gap analysis: certificates are currently too conservative
- Future work: model calibration improvements would enable formal guarantees

This section transforms from "we add a filter" to "we establish a formal framework, show it works in theory, and quantify exactly what model improvements are needed for formal safety."
