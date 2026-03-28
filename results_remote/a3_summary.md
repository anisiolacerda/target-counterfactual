# A3 Results Summary: Oracle-vs-Model Trajectory Analysis

## Key Finding

**Model-based RA filtering is ineffective on Cancer simulator.** The model's decoder predictions are too compressed and inversely correlated with oracle trajectories to identify constraint violations.

## Evidence

### 1. Model feasibility is near-100% (filter does nothing)
| Gamma | Oracle Feasibility | Model Feasibility |
|-------|-------------------|-------------------|
| 1     | 5-45%             | 100%              |
| 2     | 7-48%             | 100%              |
| 3     | 9-53%             | 100%              |
| 4     | 10-54%            | 98.6-99.8%        |

The model never predicts cancer volumes large enough to violate safety_volume_upper=12.0 cm³. Oracle values reach up to 35 cm³.

### 2. Model trajectories are inversely correlated with oracle
| Gamma | Mean Spearman r (oracle CV_terminal vs model CV_terminal) |
|-------|----------------------------------------------------------|
| 1     | -0.07 (near zero)                                        |
| 2     | -0.10                                                    |
| 3     | -0.28                                                    |
| 4     | -0.36 (inverted!)                                        |

The model INVERTS the ranking of candidates by cancer volume at high confounding.

### 3. Model prediction range is compressed
| Gamma | Oracle CV_terminal range | Model CV_terminal range |
|-------|------------------------|------------------------|
| 1     | [0, 70] cm³            | [-0.1, 0.1] cm³        |
| 4     | [0, 35] cm³            | [-0.4, 3.9] cm³        |

The model predicts a much narrower range, concentrated near the target threshold at gamma=4.

## Implications for the Paper

### Narrative change (how to frame this)
1. **Honest acknowledgment:** "Model-predicted trajectories on Cancer do not discriminate safety violations because the decoder's output distribution is compressed relative to true simulator outcomes."
2. **This is a property of the ELBO objective:** ELBO optimizes for average prediction quality, not for capturing the full distribution of outcomes — especially extreme violations.
3. **Three-level validation becomes:** (a) Oracle on Cancer = upper bound for safety filtering, (b) Model-predicted on Cancer = reveals that trajectory prediction quality is the bottleneck, (c) Model-predicted on MIMIC = where the method must be applied in practice.
4. **Key insight for discussion:** The oracle gap quantifies the *cost of imperfect trajectory prediction*. At gamma=4, safety drops from 98% (oracle-RA) to 84-91% (model-RA = ELBO) — a 7-14 percentage point gap that the model cannot close.

### Why MIMIC is different (and more favorable)
On MIMIC (B1 analysis), model-predicted feasibility is 22-33% (not near-100%). The model DOES discriminate there because:
- The target range [60, 90] mmHg is within the model's prediction range [48, 68] mmHg
- The model's predictions span the target boundary for 100% of patients
- The narrow prediction range actually HELPS on MIMIC (predictions are near the boundary where filtering matters)

### What this motivates
1. **Better trajectory prediction methods** (decoder calibration, ensemble predictions)
2. **Conformal safety certificates** (S6.1) — distribution-free coverage guarantees that don't rely on point predictions
3. **Practical recommendation:** Use model-predicted RA filtering only when the model's prediction range spans the constraint boundaries (true on MIMIC, false on Cancer at low gamma)

## Paper Text Suggestion

For the paper, add a paragraph to the Discussion or a new subsection in Experiments:

"When we apply RA filtering using model-predicted rather than oracle trajectories on Cancer (Table X), the filter becomes ineffective: model-predicted feasibility exceeds 98% at all confounding strengths because the decoder's predictions are compressed relative to true outcome diversity. This reveals that the safety improvement in Table 1 is entirely attributable to oracle trajectory access. In practice — as demonstrated on MIMIC-III — the method's value depends on whether the model's prediction range spans the constraint boundaries. On MIMIC, where predicted DBP ranges overlap the clinically meaningful threshold (60 mmHg), model-based filtering achieves near-perfect correction rates (95-100%, Section 6.2). This finding motivates future work on trajectory prediction calibration and distribution-free safety certificates."
