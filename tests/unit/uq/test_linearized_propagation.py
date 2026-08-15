#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _linear_problem():
    matrix = jnp.asarray([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
    covariance = jnp.asarray([[1.5, 0.2, -0.1], [0.2, 0.8, 0.3], [-0.1, 0.3, 1.2]])
    center = jnp.asarray([0.4, -0.2, 0.7])
    return matrix, covariance, center


def test_all_covariance_representations_recover_the_same_affine_pushforward():
    matrix, covariance, center = _linear_problem()
    expected = matrix @ covariance @ matrix.T
    cholesky = jnp.linalg.cholesky(covariance)
    representations = (
        phx.uq.DenseCovariance(covariance),
        phx.uq.FactorCovariance(cholesky.T),
        phx.uq.CovarianceOperator(lambda vector: covariance @ vector),
    )

    for declared in representations:
        result = phx.uq.propagate_linearized(
            lambda value: matrix @ value + jnp.asarray([1.0, -1.0]),
            center,
            declared,
        )
        materialized = result.materialize_covariance(max_dimension=2, batch_size=1)

        assert jnp.allclose(result.mean, matrix @ center + jnp.asarray([1.0, -1.0]))
        assert jnp.allclose(materialized.matrix, expected)
        assert jnp.allclose(
            result.covariance_vector_product(jnp.asarray([0.3, -0.8])),
            expected @ jnp.asarray([0.3, -0.8]),
        )
        if not isinstance(declared, phx.uq.CovarianceOperator):
            assert jnp.allclose(result.exact_variance(batch_size=1), jnp.diag(expected))


def test_diagonal_covariance_preserves_nested_pytrees_and_coordax_dimensions():
    center = {
        "forcing": jnp.asarray([0.5, -1.0]),
        "parameter": jnp.asarray(2.0),
    }
    variance = {
        "forcing": jnp.asarray([0.2, 0.4]),
        "parameter": jnp.asarray(0.3),
    }

    def forward(value):
        data = jnp.asarray(
            [
                value["forcing"][0] + value["parameter"],
                2.0 * value["forcing"][1] - value["parameter"],
            ]
        )
        return {
            "field": cx.Field(data, dims=("component",)),
            "total": jnp.sum(value["forcing"]),
        }

    result = phx.uq.propagate_linearized(
        forward,
        center,
        phx.uq.DiagonalCovariance(variance),
    )
    exact = result.exact_variance()

    assert result.mean["field"].dims == ("component",)
    assert exact["field"].dims == ("component",)
    assert jnp.allclose(jnp.asarray(exact["field"].data), jnp.asarray([0.5, 1.9]))
    assert jnp.allclose(exact["total"], 0.6)
    tangent = jax.tree_util.tree_map(jnp.ones_like, center)
    cotangent = {
        "field": cx.Field(jnp.asarray([0.7, -0.3]), dims=("component",)),
        "total": jnp.asarray(0.2),
    }
    pushed = result.pushforward(tangent)
    pulled = result.pullback(cotangent)
    left = sum(
        jnp.vdot(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(pushed),
            jax.tree_util.tree_leaves(cotangent),
            strict=True,
        )
    )
    right = sum(
        jnp.vdot(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(tangent),
            jax.tree_util.tree_leaves(pulled),
            strict=True,
        )
    )
    assert jnp.allclose(left, right)


def test_complex_linear_propagation_uses_the_hermitian_adjoint():
    multiplier = jnp.asarray(1.0 + 2.0j)
    center = jnp.asarray([1.0 - 0.5j, -0.2 + 0.3j])
    result = phx.uq.propagate_linearized(
        lambda value: multiplier * value,
        center,
        phx.uq.DenseCovariance(jnp.eye(2)),
        complex_linear=True,
    )
    cotangent = jnp.asarray([0.4 + 0.7j, -0.1 + 0.2j])

    assert jnp.allclose(result.pullback(cotangent), jnp.conj(multiplier) * cotangent)
    assert jnp.allclose(result.exact_variance(), jnp.full((2,), 5.0))
    assert jnp.allclose(
        result.materialize_covariance().matrix,
        5.0 * jnp.eye(2),
    )

    with pytest.raises(ValueError, match="complex_linear=True"):
        phx.uq.propagate_linearized(
            lambda value: multiplier * value,
            center,
            phx.uq.DiagonalCovariance(jnp.ones(2)),
        )


def test_hutchinson_diagonal_is_keyed_and_reports_sampling_error():
    covariance = jnp.asarray(
        [
            [2.0, 0.4, -0.2, 0.1],
            [0.4, 1.5, 0.3, -0.2],
            [-0.2, 0.3, 1.2, 0.25],
            [0.1, -0.2, 0.25, 0.9],
        ]
    )
    result = phx.uq.propagate_linearized(
        lambda value: value,
        jnp.zeros(4),
        phx.uq.CovarianceOperator(lambda vector: covariance @ vector),
    )

    estimate = result.estimate_variance(jr.key(8), num_probes=2048, batch_size=127)
    replay = result.estimate_variance(jr.key(8), num_probes=2048, batch_size=127)
    repeated = result.estimate_variance(jr.key(8), num_probes=2048, batch_size=503)
    changed = result.estimate_variance(jr.key(9), num_probes=2048)

    assert estimate.approximation == "first_order_hutchinson"
    assert estimate.probe_distribution == "rademacher"
    assert estimate.num_probes == 2048
    assert not estimate.exact
    assert jnp.array_equal(estimate.variance, replay.variance)
    assert jnp.array_equal(estimate.standard_error, replay.standard_error)
    assert jnp.allclose(estimate.variance, repeated.variance, rtol=1e-14, atol=1e-14)
    assert jnp.allclose(
        estimate.standard_error,
        repeated.standard_error,
        rtol=1e-14,
        atol=1e-14,
    )
    assert not jnp.array_equal(estimate.variance, changed.variance)
    assert jnp.allclose(estimate.variance, jnp.diag(covariance), atol=0.04)
    assert jnp.all(jnp.isfinite(estimate.standard_error))
    assert jnp.all(estimate.standard_error > 0.0)


def test_large_matrix_free_operator_path_rejects_dense_shortcuts():
    dimension = 20_000
    center = jnp.linspace(-1.0, 1.0, dimension)
    result = phx.uq.propagate_linearized(
        lambda value: 3.0 * value,
        center,
        phx.uq.CovarianceOperator(lambda vector: 2.0 * vector),
    )
    probe = jnp.ones_like(center)

    assert jnp.allclose(result.covariance_vector_product(probe), 18.0 * probe)
    with pytest.raises(ValueError, match="estimate_variance"):
        result.exact_variance()
    with pytest.raises(ValueError, match="exceeds max_dimension"):
        result.materialize_covariance(max_dimension=128)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: phx.uq.DiagonalCovariance(jnp.asarray([1.0, -0.1])),
        lambda: phx.uq.DenseCovariance(jnp.asarray([[1.0, 0.5], [0.0, 1.0]])),
        lambda: phx.uq.DenseCovariance(jnp.asarray([[1.0, 2.0], [2.0, 1.0]])),
        lambda: phx.uq.FactorCovariance(jnp.ones((0, 2))),
    ],
)
def test_covariance_representations_reject_invalid_declarations(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_covariance_representations_can_be_declared_under_jit():
    diagonal, dense, factor = jax.jit(
        lambda variance, matrix, factors: (
            phx.uq.DiagonalCovariance(variance),
            phx.uq.DenseCovariance(matrix),
            phx.uq.FactorCovariance(factors),
        )
    )(jnp.asarray([0.25, 0.5]), jnp.eye(2), jnp.eye(2))

    assert jnp.array_equal(diagonal.variance, jnp.asarray([0.25, 0.5]))
    assert jnp.array_equal(dense.matrix, jnp.eye(2))
    assert jnp.array_equal(factor.factors, jnp.eye(2))


def test_linearized_propagation_rejects_shape_and_materialization_mismatches():
    with pytest.raises(ValueError, match="shape must match"):
        phx.uq.propagate_linearized(
            lambda value: value,
            jnp.ones(3),
            phx.uq.DenseCovariance(jnp.eye(2)),
        )
    result = phx.uq.propagate_linearized(
        lambda value: jnp.concatenate((value, value)),
        jnp.ones(2),
        phx.uq.DiagonalCovariance(jnp.ones(2)),
    )
    with pytest.raises(ValueError, match="max_dimension"):
        result.materialize_covariance(max_dimension=0)
    with pytest.raises(ValueError, match="at least two"):
        result.estimate_variance(jr.key(0), num_probes=1)
