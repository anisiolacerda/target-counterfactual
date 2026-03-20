"""Tests for reach_avoid.losses module."""

import pytest
import torch

from src.reach_avoid.losses import compute_weighted_reg_loss


class TestComputeWeightedRegLoss:
    """Tests for compute_weighted_reg_loss."""

    def test_terminal_only_default(self):
        """With default weights, should return terminal loss."""
        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        result = compute_weighted_reg_loss(losses)
        assert result.item() == 3.0

    def test_terminal_only_explicit(self):
        """With lambda_intermediate=0, should return lambda_terminal * terminal."""
        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(5.0)]
        result = compute_weighted_reg_loss(losses, lambda_terminal=2.0, lambda_intermediate=0.0)
        assert result.item() == 10.0

    def test_with_intermediate(self):
        """With nonzero lambda_intermediate, should combine both."""
        losses = [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(6.0)]
        # terminal = 6.0, intermediate = mean(2.0, 4.0) = 3.0
        # result = 1.0 * 6.0 + 0.5 * 3.0 = 7.5
        result = compute_weighted_reg_loss(losses, lambda_terminal=1.0, lambda_intermediate=0.5)
        assert abs(result.item() - 7.5) < 1e-6

    def test_single_loss(self):
        """With only one loss (terminal), intermediate is ignored."""
        losses = [torch.tensor(5.0)]
        result = compute_weighted_reg_loss(losses, lambda_terminal=1.0, lambda_intermediate=0.5)
        assert result.item() == 5.0

    def test_gradient_flows(self):
        """Gradients should flow through both terminal and intermediate."""
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        z = torch.tensor(4.0, requires_grad=True)
        losses = [x * 2, y * 2, z * 2]  # [4, 6, 8]
        result = compute_weighted_reg_loss(losses, lambda_terminal=1.0, lambda_intermediate=1.0)
        # result = 1.0 * 8 + 1.0 * mean(4, 6) = 8 + 5 = 13
        result.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert z.grad is not None
