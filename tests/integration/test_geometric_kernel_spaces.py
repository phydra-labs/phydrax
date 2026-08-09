#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _assert_finite_condition(kernel, points, observations):
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.05)
    residual = model.residual(jnp.zeros_like(observations))
    factor = model.factor(state=state)
    condition = factor.condition(residual, points)

    assert jnp.isfinite(factor.log_probability(residual))
    assert jnp.all(jnp.isfinite(condition.mean))
    assert jnp.all(jnp.isfinite(condition.covariance))
    return factor


def test_compact_and_combinatorial_kernel_gp_workflows():
    sphere_points = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    sphere_kernel = phx.kernels.SphereSpectralKernel(
        2,
        8,
        phx.kernels.MaternSpectralMultiplier(0.8, 1.5),
    )
    sphere_factor = _assert_finite_condition(
        sphere_kernel,
        sphere_points,
        sphere_points[:, 0],
    )

    categorical_points = jnp.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    categorical_kernel = phx.kernels.HammingSpectralKernel(
        3,
        2,
        phx.kernels.HeatSpectralMultiplier(0.2),
    )
    categorical_factor = _assert_finite_condition(
        categorical_kernel,
        categorical_points,
        jnp.sum(categorical_points, axis=1),
    )

    assert isinstance(sphere_factor, phx.uq.ExactGaussianProcessFactor)
    assert isinstance(categorical_factor, phx.uq.ExactGaussianProcessFactor)


def test_noncompact_kernel_gp_workflows_select_exact_feature_space():
    radii = jnp.linspace(0.0, 0.8, 24)
    hyperbolic_points = jnp.stack(
        (jnp.cosh(radii), jnp.sinh(radii), jnp.zeros_like(radii)),
        axis=-1,
    )
    hyperbolic_kernel = phx.kernels.HyperbolicRandomFeatureKernel(
        phx.kernels.hyperbolic_feature_proposal(jax.random.key(41), 2, 8),
        0.8,
        1.5,
    )
    hyperbolic_factor = _assert_finite_condition(
        hyperbolic_kernel,
        hyperbolic_points,
        jnp.sin(radii),
    )

    scales = jnp.linspace(0.7, 1.4, 24)
    spd_points = jnp.stack(
        [jnp.diag(jnp.asarray([scale, 1.0 / scale])) for scale in scales]
    ).reshape((-1, 4))
    spd_kernel = phx.kernels.SPDRandomFeatureKernel(
        phx.kernels.spd_feature_proposal(jax.random.key(43), 2, 8),
        0.9,
        1.2,
    )
    spd_factor = _assert_finite_condition(
        spd_kernel,
        spd_points,
        jnp.log(scales),
    )

    assert isinstance(hyperbolic_factor, phx.uq.FiniteFeatureGaussianProcessFactor)
    assert isinstance(spd_factor, phx.uq.FiniteFeatureGaussianProcessFactor)
