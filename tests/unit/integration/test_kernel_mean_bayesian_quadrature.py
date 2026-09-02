# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def test_interval_kernel_mean_physical_and_normalized_mass_scaling():
    interval = phx.domain.Interval1d(-1.0, 2.0)
    kernel = phx.kernels.Matern32Kernel(length_scale=0.7)
    physical = phx.integration.IntervalKernelMean(interval, kernel, target_id="physical")
    normalized = phx.integration.IntervalKernelMean(
        interval, kernel, normalized=True, target_id="normalized"
    )
    points = jnp.asarray([[-0.5], [0.3], [1.5]])
    assert jnp.allclose(physical.mean(points), 3.0 * normalized.mean(points))
    assert jnp.allclose(physical.double_mean(), 9.0 * normalized.double_mean())


def test_finite_feature_bq_and_sequential_variance_are_deterministic():
    kernel = phx.kernels.FiniteFeatureKernel(
        lambda point: jnp.asarray([1.0, point[0]]),
        jnp.eye(2),
        feature_map_id="affine",
    )
    mean = phx.integration.FiniteFeatureKernelMean(
        kernel,
        jnp.asarray([1.0, 0.5]),
        target_id="unit-interval-affine",
    )
    candidates = jnp.linspace(0.0, 1.0, 8)[:, None]
    design = phx.integration.SequentialBayesianQuadratureDesign(
        candidates,
        initial_count=2,
        total_count=3,
    )
    plan = phx.integration.BayesianQuadraturePlan(
        mean,
        design,
        solve_regularization=1e-8,
    )
    prepared = phx.integration.prepare_kernel_mean_bayesian_quadrature(plan)
    value = phx.integration.reduce_kernel_mean_bayesian_quadrature(
        jnp.stack((jnp.ones((3,)), prepared.points[:, 0]), axis=1),
        prepared,
    )
    assert jnp.allclose(value, jnp.asarray([1.0, 0.5]), atol=1e-5)
    assert jnp.all(jnp.diff(prepared.posterior_variance_history) <= 1e-6)
    assert jnp.unique(prepared.source_indices).size == 3
