"""Tests for reach_avoid.scoring module."""

import numpy as np
import pytest
import torch

from src.reach_avoid.scoring import (
    soft_indicator_upper,
    soft_indicator_interval,
    compute_reach_avoid_score,
)


class TestSoftIndicatorUpper:
    """Tests for soft_indicator_upper."""

    def test_numpy_well_below_upper(self):
        """Values well below upper should yield ~1."""
        y = np.array([1.0, 2.0, 3.0])
        result = soft_indicator_upper(y, upper=10.0, kappa=10.0)
        np.testing.assert_allclose(result, np.ones(3), atol=1e-6)

    def test_numpy_well_above_upper(self):
        """Values well above upper should yield ~0."""
        y = np.array([20.0, 30.0])
        result = soft_indicator_upper(y, upper=10.0, kappa=10.0)
        np.testing.assert_allclose(result, np.zeros(2), atol=1e-6)

    def test_numpy_at_boundary(self):
        """Value at boundary should be 0.5."""
        y = np.array([5.0])
        result = soft_indicator_upper(y, upper=5.0, kappa=10.0)
        np.testing.assert_allclose(result, np.array([0.5]), atol=1e-6)

    def test_torch_well_below_upper(self):
        """Torch: values well below upper should yield ~1."""
        y = torch.tensor([1.0, 2.0, 3.0])
        result = soft_indicator_upper(y, upper=10.0, kappa=10.0)
        assert torch.allclose(result, torch.ones(3), atol=1e-6)

    def test_torch_well_above_upper(self):
        """Torch: values well above upper should yield ~0."""
        y = torch.tensor([20.0, 30.0])
        result = soft_indicator_upper(y, upper=10.0, kappa=10.0)
        assert torch.allclose(result, torch.zeros(2), atol=1e-6)

    def test_torch_at_boundary(self):
        """Torch: value at boundary should be 0.5."""
        y = torch.tensor([5.0])
        result = soft_indicator_upper(y, upper=5.0, kappa=10.0)
        assert torch.allclose(result, torch.tensor([0.5]), atol=1e-6)

    def test_high_kappa_sharpens(self):
        """Higher kappa should make the transition sharper."""
        y = np.array([4.9, 5.1])  # slightly below and above boundary=5.0
        soft_result = soft_indicator_upper(y, upper=5.0, kappa=10.0)
        sharp_result = soft_indicator_upper(y, upper=5.0, kappa=100.0)
        # With higher kappa, values should be closer to 1 and 0 respectively
        assert sharp_result[0] > soft_result[0]
        assert sharp_result[1] < soft_result[1]

    def test_2d_numpy(self):
        """Should work element-wise on 2D arrays."""
        y = np.array([[1.0, 20.0], [5.0, 3.0]])
        result = soft_indicator_upper(y, upper=5.0, kappa=10.0)
        assert result.shape == (2, 2)
        assert result[0, 0] > 0.9  # well below
        assert result[0, 1] < 0.1  # well above
        np.testing.assert_allclose(result[1, 0], 0.5, atol=1e-6)  # at boundary

    def test_torch_gradient_flows(self):
        """Gradients should flow through the soft indicator."""
        # Use value near boundary for non-vanishing gradient
        y = torch.tensor([5.0], requires_grad=True)
        result = soft_indicator_upper(y, upper=5.0, kappa=10.0)
        result.backward()
        assert y.grad is not None
        assert y.grad.item() != 0


class TestSoftIndicatorInterval:
    """Tests for soft_indicator_interval."""

    def test_numpy_inside_interval(self):
        """Value well inside [lower, upper] should yield ~1."""
        y = np.array([5.0])
        result = soft_indicator_interval(y, lower=0.0, upper=10.0, kappa=10.0)
        np.testing.assert_allclose(result, np.array([1.0]), atol=1e-3)

    def test_numpy_outside_below(self):
        """Value well below lower should yield ~0."""
        y = np.array([-5.0])
        result = soft_indicator_interval(y, lower=0.0, upper=10.0, kappa=10.0)
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-6)

    def test_numpy_outside_above(self):
        """Value well above upper should yield ~0."""
        y = np.array([20.0])
        result = soft_indicator_interval(y, lower=0.0, upper=10.0, kappa=10.0)
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-6)

    def test_torch_inside_interval(self):
        y = torch.tensor([5.0])
        result = soft_indicator_interval(y, lower=0.0, upper=10.0, kappa=10.0)
        assert result.item() > 0.99

    def test_torch_gradient_flows(self):
        y = torch.tensor([5.0], requires_grad=True)
        result = soft_indicator_interval(y, lower=0.0, upper=10.0, kappa=10.0)
        result.backward()
        assert y.grad is not None


class TestComputeReachAvoidScore:
    """Tests for compute_reach_avoid_score."""

    def test_all_safe_and_on_target_numpy(self):
        """All trajectories safe and on-target -> RA score near 0 (log(1))."""
        N, tau = 3, 4
        cv = np.ones((N, tau + 1)) * 1.0  # well below target and safety
        cd = np.ones((N, tau)) * 0.5
        scores, details = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0,
            safety_chemo_upper=3.0, kappa=10.0)
        assert scores.shape == (N,)
        # All indicators ~1, so log(~1) ~ 0
        np.testing.assert_allclose(scores, np.zeros(N), atol=0.1)

    def test_all_unsafe_numpy(self):
        """Trajectories with high cancer volume -> very negative RA score."""
        N, tau = 2, 3
        cv = np.ones((N, tau + 1)) * 50.0  # way above safety
        cd = np.ones((N, tau)) * 0.5
        scores, details = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0, kappa=10.0)
        assert scores.shape == (N,)
        assert np.all(scores < -10)  # very negative

    def test_target_violation_only(self):
        """Terminal volume high but intermediates safe -> moderately negative."""
        N, tau = 1, 3
        cv = np.ones((N, tau + 1)) * 1.0
        cv[0, -1] = 50.0  # only terminal violates target
        cd = np.ones((N, tau)) * 0.5
        scores, details = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=100.0, kappa=10.0)
        assert scores[0] < -5  # target indicator ~0

    def test_chemo_safety_violation(self):
        """High chemo dosage should reduce RA score."""
        N, tau = 1, 3
        cv = np.ones((N, tau + 1)) * 1.0
        cd_safe = np.ones((N, tau)) * 0.5
        cd_unsafe = np.ones((N, tau)) * 20.0
        scores_safe, _ = compute_reach_avoid_score(
            cv, cd_safe, target_upper=5.0, safety_volume_upper=10.0,
            safety_chemo_upper=3.0, kappa=10.0)
        scores_unsafe, _ = compute_reach_avoid_score(
            cv, cd_unsafe, target_upper=5.0, safety_volume_upper=10.0,
            safety_chemo_upper=3.0, kappa=10.0)
        assert scores_safe[0] > scores_unsafe[0]

    def test_no_chemo_safety(self):
        """Without chemo safety constraint, chemo dosage is ignored."""
        N, tau = 1, 3
        cv = np.ones((N, tau + 1)) * 1.0
        cd = np.ones((N, tau)) * 100.0  # very high
        scores, details = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0,
            safety_chemo_upper=None, kappa=10.0)
        np.testing.assert_allclose(scores, np.zeros(N), atol=0.1)

    def test_torch_backend(self):
        """RA score should work with torch tensors."""
        N, tau = 2, 4
        cv = torch.ones((N, tau + 1)) * 1.0
        cd = torch.ones((N, tau)) * 0.5
        scores, details = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0,
            safety_chemo_upper=3.0, kappa=10.0)
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (N,)
        assert torch.allclose(scores, torch.zeros(N), atol=0.1)

    def test_torch_gradient_flows(self):
        """Gradients should flow through RA score computation."""
        N, tau = 1, 3
        # Use values near boundaries for non-vanishing gradients
        cv = torch.tensor([[4.5, 4.5, 4.5, 4.5]], requires_grad=True)
        cd = torch.tensor([[2.8, 2.8, 2.8]])
        scores, _ = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0,
            safety_chemo_upper=3.0, kappa=10.0)
        scores.sum().backward()
        assert cv.grad is not None
        assert cv.grad.abs().sum().item() > 0

    def test_details_keys(self):
        """RA details dict should contain expected keys."""
        N, tau = 1, 2
        cv = np.ones((N, tau + 1)) * 2.0
        cd = np.ones((N, tau)) * 0.5
        _, details = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0, kappa=10.0)
        assert 'g_target' in details
        assert 'g_safety_vol_prod' in details
        assert 'g_safety_chemo_prod' in details
        assert 'combined' in details

    def test_ranking_order(self):
        """Safer trajectories should get higher RA scores."""
        tau = 4
        # Patient 0: low volume, patient 1: high volume
        cv = np.array([[1.0] * (tau + 1), [8.0] * (tau + 1)])
        cd = np.ones((2, tau)) * 0.5
        scores, _ = compute_reach_avoid_score(
            cv, cd, target_upper=5.0, safety_volume_upper=10.0, kappa=10.0)
        assert scores[0] > scores[1]
