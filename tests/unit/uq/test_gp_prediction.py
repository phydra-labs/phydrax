# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import phydrax as phx
from phydrax.uq._gp_prediction import (
    GaussianProcessMeanPredictor,
    prepare_gaussian_process_mean,
)


type FactorKind = Literal["exact", "finite-feature", "fitc"]
type Factor = (
    phx.uq.ExactGaussianProcessFactor
    | phx.uq.FiniteFeatureGaussianProcessFactor
    | phx.uq.SparseGaussianProcessFactor
)


def _features(point: Array) -> Array:
    return jnp.stack((jnp.ones_like(point[0]), point[0], point[0] ** 2))


def _factor(kind: FactorKind, points: Array, scale: Array) -> Factor:
    kernel: phx.kernels.AbstractPositiveDefiniteKernel
    if kind == "finite-feature":
        kernel = phx.kernels.FiniteFeatureKernel(
            _features,
            scale * jnp.eye(3, dtype=jnp.float64),
            feature_map_id="prepared-mean-polynomial",
            max_derivative_order=None,
        )
    else:
        kernel = phx.kernels.Matern52Kernel(length_scale=scale)
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=jnp.linspace(0.08, 0.13, points.shape[0]),
        jitter=1e-7,
    )
    if kind == "fitc":
        return phx.uq.SparseGaussianProcessFactor(points, points[::3], state=state)
    if kind == "finite-feature":
        return phx.uq.FiniteFeatureGaussianProcessFactor(points, state=state)
    return phx.uq.ExactGaussianProcessFactor(points, state=state)


@pytest.mark.parametrize("kind", ["exact", "finite-feature", "fitc"])
def test_prepared_mean_matches_conditioner_with_block_edges(kind: FactorKind) -> None:
    points = jnp.linspace(-1.0, 1.0, 8)
    residual = jnp.sin(2.0 * points)
    factor = _factor(kind, points, jnp.asarray(0.7))
    count = 3 if kind != "exact" else points.size
    bytes_per_row = (count + 2) * points.dtype.itemsize
    predictor = prepare_gaussian_process_mean(
        factor, residual, max_workspace_bytes=2 * bytes_per_row
    )
    for query_count in (0, 1, 2, 3, 7):
        query = jnp.linspace(-0.9, 0.8, query_count)
        actual = eqx.filter_jit(predictor)(query)
        assert actual.shape == (query_count,)
        if query_count:
            expected = factor.condition(residual, query).mean
            assert jnp.allclose(actual, expected, atol=1e-9, rtol=1e-9)
    assert jnp.all(predictor.solve_status == phx.linalg.LinearSolveStatus.SUCCESS)


@pytest.mark.parametrize("kind", ["exact", "finite-feature", "fitc"])
def test_preparation_and_queries_preserve_parameter_input_and_residual_derivatives(
    kind: FactorKind,
) -> None:
    points = jnp.linspace(-0.8, 0.9, 7)
    residual = jnp.sin(points)
    query = jnp.asarray([-0.8, -0.2, 0.9])
    scale = jnp.asarray(0.7)

    def predicted(x: Array, r: Array, q: Array, s: Array) -> Array:
        factor = _factor(kind, x, s)
        return jnp.sum(
            prepare_gaussian_process_mean(factor, r, max_workspace_bytes=256)(q)
        )

    def conditioned(x: Array, r: Array, q: Array, s: Array) -> Array:
        return jnp.sum(_factor(kind, x, s).condition(r, q).mean)

    actual = jax.grad(predicted, argnums=(0, 1, 2, 3))(points, residual, query, scale)
    expected = jax.grad(conditioned, argnums=(0, 1, 2, 3))(points, residual, query, scale)
    for derivative, reference in zip(actual, expected, strict=True):
        assert jnp.all(jnp.isfinite(derivative))
        assert jnp.allclose(derivative, reference, atol=2e-7, rtol=2e-7)
    _, tangent = jax.jvp(
        lambda q: predicted(points, residual, q, scale),
        (query,),
        (jnp.ones_like(query),),
    )
    assert jnp.allclose(tangent, jnp.sum(expected[2]), atol=2e-7, rtol=2e-7)

    def prepare(x: Array, r: Array, s: Array) -> GaussianProcessMeanPredictor:
        return prepare_gaussian_process_mean(
            _factor(kind, x, s), r, max_workspace_bytes=256
        )

    transformed = eqx.filter_jit(prepare)(points, residual, scale)
    assert jnp.allclose(
        jnp.sum(transformed(query)), predicted(points, residual, query, scale)
    )
    changed = eqx.filter_jit(prepare)(points, residual, scale * 1.4)
    assert not jnp.allclose(transformed(query), changed(query))


@pytest.mark.parametrize("kind", ["exact", "finite-feature", "fitc"])
def test_result_retention_excludes_covariance_factors_and_training_features(
    kind: FactorKind,
) -> None:
    points = jnp.linspace(-1.0, 1.0, 19)
    factor = _factor(kind, points, jnp.asarray(0.7))
    predictor = prepare_gaussian_process_mean(factor, jnp.sin(points))
    leaves = [leaf for leaf in jax.tree.leaves(predictor) if eqx.is_array(leaf)]
    expected_elements = predictor.coefficients.size + predictor.solve_status.size
    expected_bytes = (
        predictor.coefficients.size * predictor.coefficients.dtype.itemsize
        + predictor.solve_status.size * predictor.solve_status.dtype.itemsize
    )
    for leaf in jax.tree.leaves(predictor.kernel):
        if eqx.is_array(leaf):
            expected_elements += leaf.size
            expected_bytes += leaf.size * leaf.dtype.itemsize
    if predictor.reference_points is not None:
        expected_elements += predictor.reference_points.size
        expected_bytes += (
            predictor.reference_points.size * predictor.reference_points.dtype.itemsize
        )
    assert predictor.retained_storage_elements == expected_elements
    assert predictor.retained_storage_bytes == expected_bytes
    assert sum(leaf.size for leaf in leaves) == expected_elements
    if kind == "finite-feature":
        assert predictor.reference_points is None
        assert all(points.size not in leaf.shape for leaf in leaves)
    else:
        expected_count = points.size if kind == "exact" else points[::3].size
        assert predictor.coefficients.shape == (expected_count,)
        assert predictor.reference_points is not None
        assert predictor.reference_points.shape == (expected_count, 1)
    assert predictor.retained_storage_elements < factor.factor_storage_elements


def test_budget_and_invalid_preparation_fail_instead_of_predicting() -> None:
    points = jnp.linspace(-1.0, 1.0, 5)
    factor = _factor("exact", points, jnp.asarray(0.7))
    residual = jnp.sin(points)
    floor = (points.size + 1) * points.dtype.itemsize
    with pytest.raises(ValueError, match="workspace"):
        prepare_gaussian_process_mean(factor, residual, max_workspace_bytes=floor - 1)
    predictor = prepare_gaussian_process_mean(factor, residual, max_workspace_bytes=floor)
    with pytest.raises(ValueError, match="workspace"):
        predictor(points)
    with pytest.raises(ValueError, match="align"):
        prepare_gaussian_process_mean(factor, residual[:-1])
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        prepare_gaussian_process_mean(factor, residual.at[0].set(jnp.nan))
    invalid = eqx.tree_at(
        lambda candidate: candidate.correction_cholesky,
        _factor("finite-feature", points, jnp.asarray(0.7)),
        jnp.zeros((3, 3), dtype=jnp.float64),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="Cholesky"):
        prepare_gaussian_process_mean(invalid, residual)


def test_structured_kernel_query_paths_preserve_input_rank_and_variable_length() -> None:
    paths = jnp.asarray(
        [[[0.0], [0.2], [0.6]], [[0.0], [0.4], [0.7]], [[0.0], [-0.2], [0.1]]]
    )
    query = jnp.asarray([[[0.0], [0.1], [0.3], [0.5]], [[0.0], [-0.1], [-0.2], [0.1]]])
    kernel = phx.kernels.SignaturePDEKernel(
        phx.kernels.SquaredExponentialKernel(length_scale=0.8), polynomial_order=2
    )
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)
    factor = phx.uq.ExactGaussianProcessFactor(paths, state=state)
    residual = jnp.asarray([0.2, -0.1, 0.4])
    predictor = prepare_gaussian_process_mean(factor, residual, max_workspace_bytes=64)
    assert jnp.allclose(predictor(query), factor.condition(residual, query).mean)
    derivative = jax.grad(lambda q: jnp.sum(predictor(q)))(query)
    assert derivative.shape == query.shape
    assert jnp.all(jnp.isfinite(derivative))
    with pytest.raises(ValueError, match="kernel input axes"):
        predictor(query.reshape((2, 4)))
