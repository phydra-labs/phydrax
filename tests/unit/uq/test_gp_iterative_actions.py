# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import pytest

import phydrax as phx


def _state():
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.4),
        noise_scale=0.05,
    )


def test_lanczos_cg_and_gauss_seidel_fixed_capacity_evidence():
    points = jnp.linspace(0.0, 1.0, 6)[:, None]
    residual = jnp.asarray([1.0, -0.2, 0.3, 0.1, -0.4, 0.2])
    policies = (
        phx.uq.LanczosGaussianProcessActionPolicy(jnp.ones((6,)), max_actions=4),
        phx.uq.ConjugateGradientGaussianProcessActionPolicy(4),
        phx.uq.GaussSeidelGaussianProcessActionPolicy(4),
    )
    for policy in policies:
        resolved = policy.resolve(
            points,
            state=_state(),
            residual=residual if policy.requires_residual else None,
        )
        assert resolved.active_mask.shape == (4,)
        assert resolved.residual_history.shape == (5,)
        assert resolved.operator.source.size == 4


def test_residual_dependent_factor_fails_but_condition_path_resolves_actions():
    points = jnp.linspace(0.0, 1.0, 5)[:, None]
    discrepancy = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points, jnp.sin(points[:, 0])
    )
    policy = phx.uq.ConjugateGradientGaussianProcessActionPolicy(3)
    with pytest.raises(ValueError, match="residual-dependent"):
        discrepancy.factor(state=_state(), actions=policy)
    condition = discrepancy.condition(
        jnp.zeros((5,)),
        points,
        state=_state(),
        actions=policy,
    )
    assert condition.mean.shape == (5,)
