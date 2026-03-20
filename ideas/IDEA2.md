## Future Direction: Hybrid Mamba+LSTM+Attention Architecture (for Discussion section)

**Idea**: Replace VCIP's LSTM-only backbone with a three-layer hybrid:

1. Mamba layer — efficient state-space model for global temporal context across long patient histories
2. LSTM layer — fine-grained local sequential dependencies and treatment-state transitions
3. Attention layer — highlight specific critical events (e.g., safety-relevant episodes) for targeted prediction

**Rationale**: While the current LSTM backbone is adequate for VCIP's moderate sequence lengths (tau ≈ 2-8, history ≈ 20-50 steps), clinical EHR data can span much longer horizons. Mamba's O(n) scaling would enable modeling of full patient trajectories, attention would improve interpretability of safety-critical predictions, and the LSTM would preserve the proven local transition dynamics.

**Why not in current submission**: (1) VCIP's temporal modeling already works well (GRP 0.932-0.994); the identified weaknesses are in the objective and evaluation, not the architecture. (2) Would require full hyperparameter search and retraining, significantly increasing cost and timeline. (3) Risks diluting the paper's focused reach-avoid contribution.

**For the paper**: Mention in Discussion as: "Extending the latent dynamics backbone with state-space models (e.g., Mamba) and attention mechanisms could improve scalability to longer clinical horizons and interpretability of safety-critical predictions, which we leave as promising future work."

The hybrid architecture is now documented as a future direction in the plan. The key rationale for deferring it:

- VCIP's temporal modeling already works well (GRP 0.932-0.994) — the weaknesses are in the objective and evaluation, not architecture capacity
- It would require a full retraining/hyperparameter search, increasing cost and timeline
- It risks diluting the focused reach-avoid contribution for NeurIPS

For the paper's Discussion section, we'll include a sentence acknowledging Mamba+LSTM+Attention as a promising direction for scaling to longer clinical horizons and improving interpretability of safety-critical predictions.