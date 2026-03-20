"""Reach-Avoid Counterfactual Intervention Planning.

Extends VCIP with set-membership optimization (target sets + safety constraints)
instead of point-target ELBO ranking.
"""

from src.reach_avoid.scoring import (
    soft_indicator_upper,
    soft_indicator_interval,
    compute_reach_avoid_score,
)
from src.reach_avoid.losses import (
    compute_disentanglement_loss,
    compute_weighted_reg_loss,
)
