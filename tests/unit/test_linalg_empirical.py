import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _dense_weighted_gram(matrix, weights, *, centered, damping):
    normalized = weights / jnp.sum(weights)
    values = matrix
    if centered:
        values = values - jnp.sum(normalized[:, None] * values, axis=0, keepdims=True)
    return jnp.conj(values.T) @ (normalized[:, None] * values) + damping * jnp.eye(
        matrix.shape[1], dtype=matrix.dtype
    )


def test_empirical_gram_matches_weighted_centered_dense_reference():
    matrix = jnp.asarray(
        [[1.0, 2.0, -1.0], [2.0, -1.0, 0.5], [4.0, 0.0, 3.0], [3.0, 1.0, 2.0]]
    )
    weights = jnp.asarray([1.0, 2.0, 0.0, 3.0])
    direction = jnp.asarray([0.3, -0.7, 0.2])
    operator = phx.linalg.EmpiricalGramLinearOperator(
        phx.linalg.DenseLinearOperator(matrix),
        weights,
        centered=True,
        damping=0.15,
    )
    expected = _dense_weighted_gram(matrix, weights, centered=True, damping=0.15)

    assert jnp.allclose(operator.mv(direction), expected @ direction)
    assert jnp.allclose(eqx.filter_jit(operator.mv)(direction), expected @ direction)
    assert operator.active_samples == 3
    assert jnp.allclose(operator.weight_ess, 36.0 / 14.0)
    assert operator.rank_upper_bound == 2
    assert operator.properties.certifies("self_adjoint")
    assert operator.properties.certifies("positive_definite")


def test_zero_weight_rows_mask_nonfinite_features_before_differentiation():
    matrix = jnp.asarray([[1.0, 2.0], [jnp.nan, jnp.nan], [3.0, -1.0]])
    operator = phx.linalg.EmpiricalGramLinearOperator(
        phx.linalg.DenseLinearOperator(matrix),
        jnp.asarray([1.0, 0.0, 1.0]),
        centered=True,
        damping=0.1,
    )
    direction = jnp.asarray([0.2, -0.3])
    finite_rows = matrix[jnp.asarray([0, 2])]
    expected = _dense_weighted_gram(
        finite_rows,
        jnp.ones((2,)),
        centered=True,
        damping=0.1,
    )

    assert jnp.allclose(operator.mv(direction), expected @ direction)
    gradient = jax.grad(lambda value: jnp.sum(operator.mv(value) ** 2))(direction)
    assert jnp.all(jnp.isfinite(gradient))


def test_empirical_gram_complex_adjoint_and_transpose_are_distinct_and_correct():
    matrix = jnp.asarray([[1.0 + 1.0j, 2.0], [0.5j, -1.0 + 2.0j], [3.0, 0.25 - 0.5j]])
    weights = jnp.asarray([1.0, 3.0, 2.0])
    direction = jnp.asarray([0.2 + 0.1j, -0.3 + 0.4j])
    operator = phx.linalg.EmpiricalGramLinearOperator(
        phx.linalg.DenseLinearOperator(matrix),
        weights,
        centered=False,
    )
    dense = _dense_weighted_gram(matrix, weights, centered=False, damping=0.0)

    assert jnp.allclose(operator.mv(direction), dense @ direction)
    assert jnp.allclose(operator.adjoint_mv(direction), jnp.conj(dense.T) @ direction)
    assert jnp.allclose(operator.transpose_mv(direction), dense.T @ direction)


def test_empirical_gram_solves_through_existing_linear_runtime():
    features = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    operator = phx.linalg.EmpiricalGramLinearOperator(
        phx.linalg.DenseLinearOperator(features),
        jnp.ones((3,)),
        centered=True,
        damping=0.2,
    )
    rhs = jnp.asarray([0.5, -0.25])
    dense = _dense_weighted_gram(features, jnp.ones((3,)), centered=True, damping=0.2)
    result = phx.linalg.solve(phx.linalg.LinearSystem(operator), rhs)

    assert result.successful
    assert jnp.allclose(result.value, jnp.linalg.solve(dense, rhs), atol=1e-10)


def test_empirical_gram_rejects_invalid_weights_and_shapes():
    features = phx.linalg.DenseLinearOperator(jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="shape"):
        phx.linalg.EmpiricalGramLinearOperator(features, jnp.ones((2,)))
    with pytest.raises(ValueError, match="finite, non-negative"):
        phx.linalg.EmpiricalGramLinearOperator(features, jnp.asarray([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="positive mass"):
        phx.linalg.EmpiricalGramLinearOperator(features, jnp.zeros((3,)))


def test_public_fisher_action_uses_uncentered_empirical_geometry():
    scores = jnp.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    vector = jnp.asarray([0.2, -0.4])
    weights = jnp.asarray([1.0, 2.0, 3.0])
    result = phx.uq.fisher_information_action(
        scores,
        vector,
        weights=weights,
        regularization=0.1,
    )
    dense = _dense_weighted_gram(scores, weights, centered=False, damping=0.1)

    assert result.valid
    assert jnp.allclose(result.action, dense @ vector)
    with pytest.raises(TypeError, match="real scores"):
        phx.uq.fisher_information_action(scores.astype(complex), vector.astype(complex))
