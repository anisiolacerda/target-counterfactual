"""Tests for reach_avoid.model module — verify ReachAvoidVAEModel extends VAEModel correctly.

These tests verify structural properties (subclassing, method overrides, source inspection)
without instantiating models (which would require GPU/data).

NOTE: Must be run from the VCIP directory so that `src.models.*` resolves correctly:
    cd lightning-hydra-template-main/src/vendor/VCIP
    python -m pytest ../../../src/reach_avoid/tests/test_model.py -v
"""

import sys
import os
from pathlib import Path

import pytest

# The VCIP `src` package must be importable. When pytest rootdir is the
# lightning-hydra-template-main/ parent, `src` resolves to the wrong package.
# We force the VCIP directory onto sys.path as first entry, and invalidate
# caches so Python re-resolves the `src` namespace package.
_VCIP_DIR = str(Path(__file__).resolve().parents[2] / "vendor" / "VCIP")
if sys.path[0] != _VCIP_DIR:
    sys.path.insert(0, _VCIP_DIR)
    # Invalidate any cached `src` module so Python resolves from VCIP
    for mod_name in list(sys.modules):
        if mod_name == 'src' or mod_name.startswith('src.models') or mod_name.startswith('src.utils'):
            del sys.modules[mod_name]

import inspect
from src.models.vae_model import VAEModel
from src.reach_avoid.model import ReachAvoidVAEModel


class TestReachAvoidVAEModelStructure:
    """Structural tests — no GPU/data needed."""

    def test_subclass_relationship(self):
        """ReachAvoidVAEModel should be a subclass of VAEModel."""
        assert issubclass(ReachAvoidVAEModel, VAEModel)

    def test_overrides_calculate_elbo(self):
        """ReachAvoidVAEModel must override calculate_elbo."""
        assert 'calculate_elbo' in ReachAvoidVAEModel.__dict__

    def test_overrides_discrete_onetime(self):
        """ReachAvoidVAEModel must override optimize_interventions_discrete_onetime."""
        assert 'optimize_interventions_discrete_onetime' in ReachAvoidVAEModel.__dict__

    def test_vcip_does_not_import_reach_avoid(self):
        """Original VAEModel should NOT import anything from reach_avoid."""
        source = inspect.getsource(VAEModel)
        assert 'reach_avoid' not in source
        assert 'compute_reach_avoid_score' not in source

    def test_vcip_does_not_have_disent(self):
        """Original VAEModel.calculate_elbo should not contain disentanglement logic."""
        source = inspect.getsource(VAEModel.calculate_elbo)
        assert 'lambda_disent' not in source
        assert 'disent_loss' not in source

    def test_vcip_does_not_have_ra_config(self):
        """Original VAEModel should not reference reach_avoid config."""
        source = inspect.getsource(VAEModel)
        assert 'ra_config' not in source
        assert 'reach_avoid' not in source

    def test_ra_model_has_expected_attrs(self):
        """ReachAvoidVAEModel.__init__ should set RA-specific attributes."""
        source = inspect.getsource(ReachAvoidVAEModel.__init__)
        assert 'lambda_disent' in source
        assert 'lambda_terminal' in source
        assert 'lambda_intermediate' in source
        assert 'ra_config' in source

    def test_calculate_elbo_uses_weighted_reg(self):
        """ReachAvoidVAEModel.calculate_elbo should use compute_weighted_reg_loss."""
        source = inspect.getsource(ReachAvoidVAEModel.calculate_elbo)
        assert 'compute_weighted_reg_loss' in source

    def test_calculate_elbo_uses_disent(self):
        """ReachAvoidVAEModel.calculate_elbo should use compute_disentanglement_loss."""
        source = inspect.getsource(ReachAvoidVAEModel.calculate_elbo)
        assert 'compute_disentanglement_loss' in source

    def test_discrete_onetime_uses_ra_scoring(self):
        """Discrete evaluation should reference RA trajectory collection."""
        source = inspect.getsource(ReachAvoidVAEModel.optimize_interventions_discrete_onetime)
        assert 'return_trajectory' in source
        assert 'traj_features' in source

    def test_base_vcip_calculate_elbo_signature_preserved(self):
        """Base VAEModel.calculate_elbo signature should be preserved in override."""
        base_sig = inspect.signature(VAEModel.calculate_elbo)
        ra_sig = inspect.signature(ReachAvoidVAEModel.calculate_elbo)
        assert set(base_sig.parameters.keys()) == set(ra_sig.parameters.keys())
