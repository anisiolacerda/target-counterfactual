# VCIP Replication Report — Cancer Simulation

**Date:** 2026-03-02
**Hardware:** NVIDIA A100-PCIE-40GB (vast.ai)
**Seeds:** 10, 101, 1010, 10101, 101010 (5 seeds, matching paper)

## Summary

Overall, the replication is **successful**. VCIP's core claims are confirmed:
1. VCIP consistently achieves the lowest target distance across all prediction horizons (Table 1).
2. VCIP's GRP and RCS metrics significantly outperform baselines (Figure 4).
3. The g-formula adjustment provides meaningful improvement (Table 3 ablation).
4. VCIP's advantage grows with confounding strength gamma (Figure 6).

## Table 1: Long-range prediction (gamma=4, identical strategies)

Target distance (mean over 5 seeds). Percentage difference from paper in parentheses.

| Model | tau=1 | tau=2 | tau=4 | tau=6 | tau=8 | tau=10 | tau=12 |
|-------|-------|-------|-------|-------|-------|--------|--------|
| VCIP  | 0.29 (−1%) | 0.42 (+0%) | 0.60 (+0%) | 0.75 (+1%) | 0.93 (+1%) | 0.99 (−0%) | 1.09 (+0%) |
| RMSN  | 0.30 (−1%) | 0.45 (+1%) | 0.75 (−1%) | 0.98 (+0%) | 1.15 (−0%) | 1.28 (+0%) | 1.46 (−0%) |
| CRN   | 0.37 (−2%) | 0.59 (−1%) | 0.92 (+0%) | 1.21 (+1%) | 1.35 (+2%) | 1.50 (+1%) | 1.64 (+1%) |
| ACTIN | 0.41 (−2%) | 0.71 (−1%) | 1.08 (+3%) | 1.37 (+5%) | 1.61 (+9%) | 1.71 (+7%) | 1.82 (+8%) |
| CT    | 0.47 (−15%) | 0.80 (−9%) | 1.29 (−10%) | 1.55 (−8%) | 1.76 (−6%) | 1.87 (−8%) | 2.03 (−5%) |

**Verdict:** VCIP, RMSN, and CRN replicate within ~2% of paper values. ACTIN drifts slightly higher at large tau (+5-10%). CT replicates *better* than reported (−5 to −15%), possibly due to seed variance.

## Table 2: Distinct strategies

**Not replicated.** Requires running with `exp.test=True` (uses test set with different intervention strategies). Our runs used `exp.test=False` only.

## Table 3: Ablation Study

### GPR (higher is better)

| Model | tau=2 (ours) | tau=2 (paper) | tau=4 (ours) | tau=4 (paper) |
|-------|-------------|---------------|-------------|---------------|
| VCIP  | 0.923±0.007 | **0.944±0.091** | 0.963±0.005 | **0.972±0.063** |
| VCIP w/o adj | 0.730±0.026 | **0.791±0.334** | 0.756±0.016 | **0.796±0.356** |
| RMSN  | 0.651±0.231 | **0.863±0.181** | 0.573±0.257 | **0.796±0.267** |
| RMSN w/o adj | 0.687±0.153 | **0.797±0.318** | 0.574±0.262 | **0.747±0.348** |

### RCS (higher is better)

| Model | tau=2 (ours) | tau=2 (paper) | tau=4 (ours) | tau=4 (paper) |
|-------|-------------|---------------|-------------|---------------|
| VCIP  | 0.747±0.039 | **0.772±0.206** | 0.842±0.020 | **0.869±0.156** |
| VCIP w/o adj | 0.443±0.067 | **0.566±0.508** | 0.482±0.048 | **0.595±0.518** |
| RMSN  | 0.286±0.266 | **0.400±0.311** | 0.114±0.280 | **0.251±0.299** |
| RMSN w/o adj | 0.393±0.132 | **0.461±0.460** | 0.166±0.260 | **0.213±0.407** |

### Target Distance (lower is better, tau=6)

| Model | gamma=1 (ours) | gamma=1 (paper) | gamma=2 | gamma=2 (paper) | gamma=3 | gamma=3 (paper) | gamma=4 | gamma=4 (paper) |
|-------|----------------|-----------------|---------|-----------------|---------|-----------------|---------|-----------------|
| VCIP  | 0.109 | **0.101** | 0.195 | **0.192** | 0.387 | **0.382** | 0.755 | **0.746** |
| RMSN  | 0.076 | **0.078** | 0.301 | **0.301** | 0.638 | **0.599** | 0.983 | **0.985** |

**Verdict:**
- VCIP GPR/RCS: Slightly lower than paper (2-6% gap), but directional conclusions hold. The gap may be explained by seed variance — the paper's standard deviations are large (0.09-0.35).
- RMSN GPR/RCS: Larger gap (20-50% lower). This is the most notable discrepancy. RMSN's discrete ranking appears more sensitive to randomness.
- Target distance: Very close to paper values (within 1-6%).
- The ablation effect (VCIP vs VCIP w/o adjustment) is clearly confirmed: adjustment consistently improves both GPR/RCS and target distance.
- Ablation for gamma=1,2,3 was not run (scripts only run ablation for gamma=4).

## Figure 4: GRP and RCS (gamma=4, tau=2,4,6,8)

### GPR

| Model | tau=2 | tau=4 | tau=6 | tau=8 |
|-------|-------|-------|-------|-------|
| VCIP  | 0.923 | 0.963 | 0.981 | 0.984 |
| CRN   | 0.685 | 0.702 | 0.687 | 0.657 |
| ACTIN | 0.584 | 0.584 | 0.573 | 0.561 |
| CT    | 0.469 | 0.481 | 0.472 | 0.466 |
| RMSN  | 0.651 | 0.573 | 0.506 | 0.456 |

### RCS

| Model | tau=2 | tau=4 | tau=6 | tau=8 |
|-------|-------|-------|-------|-------|
| VCIP  | 0.747 | 0.842 | 0.892 | 0.904 |
| CRN   | 0.325 | 0.250 | 0.172 | 0.116 |
| ACTIN | 0.283 | 0.245 | 0.190 | 0.150 |
| CT    | −0.008 | −0.008 | 0.009 | 0.021 |
| RMSN  | 0.286 | 0.114 | −0.020 | −0.072 |

**Verdict:** The paper's key claim is strongly confirmed:
- VCIP's GRP *increases* with tau (0.923 → 0.984), while baselines decrease.
- VCIP's RCS also increases (0.747 → 0.904), while baselines deteriorate.
- Baselines (RMSN, CT) drop to near-zero or negative RCS at tau=8, confirming accumulated prediction error degrades ranking quality.

## Figure 6: Target Distance (gamma=1,2,3, tau=1..6)

Results are consistent with the paper:
- At gamma=1 (low confounding), all models perform similarly (target distance 0.02–0.12).
- At gamma=2, VCIP begins to separate from baselines, especially at larger tau.
- At gamma=3, VCIP clearly dominates: at tau=6, VCIP=0.387 vs CT=1.098, ACTIN=0.731, CRN=0.752, RMSN=0.638.
- The gap grows with both gamma (confounding strength) and tau (prediction horizon).

## Gaps and Next Steps

1. **Table 2 missing:** Need to run with `exp.test=True` for distinct-strategy evaluation.
2. **Ablation at gamma=1,2,3 missing:** Scripts only run ablation for gamma=4; paper reports all gammas.
3. **RMSN ranking discrepancy:** RMSN GPR/RCS values are notably lower than reported. Investigate whether this is a seed-sensitivity issue or code difference.
4. **10 seeds vs 5 seeds:** The paper uses 5 seeds (confirmed by script defaults matching ours), so this should not explain discrepancies.

## Analysis Script

Results were parsed using `analyze_results.py` in the project root. Raw data is in `results_remote/my_outputs/`.
