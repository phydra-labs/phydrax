#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _feature_map(point):
    coordinate = point[0]
    return jnp.array([1.0, coordinate, coordinate**2, jnp.sin(coordinate)])


def test_finite_feature_kernel_matches_explicit_whitened_features():
    points = jnp.linspace(-1.0, 1.0, 11)
    factor = jnp.array(
        [
            [0.7, 0.0, 0.0],
            [0.1, 0.5, 0.0],
            [0.0, 0.2, 0.3],
            [0.2, 0.1, 0.4],
        ]
    )
    kernel = phx.kernels.FiniteFeatureKernel(
        _feature_map,
        factor,
        feature_map_id="polynomial-sine",
        max_derivative_order=None,
    )
    raw = jax.vmap(_feature_map)(points[:, None])
    features = raw @ factor

    assert jnp.allclose(kernel.features(points), features)
    assert jnp.allclose(kernel.matrix(points, points), features @ features.T)
    assert jnp.allclose(kernel.diagonal(points), jnp.sum(features**2, axis=1))
    assert kernel.max_derivative_order is None


def test_exact_discrepancy_uses_weight_space_with_dense_numerical_parity():
    points = jnp.linspace(-1.0, 1.0, 30)
    observations = jnp.sin(2.0 * points)
    factor = jnp.array(
        [
            [0.7, 0.0, 0.0],
            [0.1, 0.5, 0.0],
            [0.0, 0.2, 0.3],
            [0.2, 0.1, 0.4],
        ]
    )
    kernel = phx.kernels.FiniteFeatureKernel(
        _feature_map,
        factor,
        feature_map_id="polynomial-sine",
        max_derivative_order=None,
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=jnp.linspace(0.05, 0.09, points.size),
    )
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)
    structured = model.factor(state=state)
    dense = phx.uq.ExactGaussianProcessFactor(points, state=state)
    residual = model.residual(jnp.zeros_like(observations))
    query = jnp.linspace(-0.8, 0.8, 13)
    structured_condition = structured.condition(residual, query)
    dense_condition = dense.condition(residual, query)

    assert isinstance(structured, phx.uq.FiniteFeatureGaussianProcessFactor)
    assert structured.factor_storage_elements < dense.factor_storage_elements
    assert jnp.allclose(
        structured.log_probability(residual),
        dense.log_probability(residual),
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(
        structured_condition.mean,
        dense_condition.mean,
        rtol=1e-9,
        atol=1e-9,
    )
    assert jnp.allclose(
        structured_condition.covariance,
        dense_condition.covariance,
        rtol=1e-8,
        atol=1e-8,
    )

    gradient = jax.grad(
        lambda candidate: model.log_marginal_likelihood(
            jnp.zeros_like(observations),
            state=phx.uq.GaussianProcessLikelihoodState(
                kernel=phx.kernels.FiniteFeatureKernel(
                    _feature_map,
                    candidate,
                    feature_map_id="polynomial-sine",
                    max_derivative_order=None,
                ),
                noise_scale=0.07,
            ),
        )
    )(factor)
    assert jnp.all(jnp.isfinite(gradient))


def test_structured_factor_resolves_wrapped_finite_features():
    points = jnp.linspace(-1.0, 1.0, 10)
    kernel = phx.kernels.FiniteFeatureKernel(
        _feature_map,
        jnp.eye(4),
        feature_map_id="identity-prior",
    )
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(kernel, 0.5),
        noise_scale=0.1,
    )
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, jnp.zeros_like(points))

    factor = model.factor(state=state)

    assert isinstance(factor, phx.uq.FiniteFeatureGaussianProcessFactor)
    expected = 0.5 * kernel.features(points)
    assert jnp.allclose(factor.features, expected)
    assert jnp.allclose(
        factor.log_probability(jnp.zeros_like(points)),
        phx.uq.ExactGaussianProcessFactor(points, state=state).log_probability(
            jnp.zeros_like(points)
        ),
    )


def test_automatic_feature_factor_requires_rank_below_observation_count():
    points = jnp.linspace(-1.0, 1.0, 4)
    observations = jnp.zeros_like(points)
    model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)

    equal_rank = phx.kernels.FiniteFeatureKernel(
        _feature_map,
        jnp.eye(4),
        feature_map_id="equal-rank",
    )
    larger_rank = phx.kernels.FiniteFeatureKernel(
        _feature_map,
        jnp.pad(jnp.eye(4), ((0, 0), (0, 2))),
        feature_map_id="larger-rank",
    )
    for kernel in (equal_rank, larger_rank):
        state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)
        assert isinstance(model.factor(state=state), phx.uq.ExactGaussianProcessFactor)
