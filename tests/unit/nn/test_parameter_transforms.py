import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import Linear
from phydrax.nn.parameters import (
    HurwitzTransform,
    IntervalTransform,
    PositiveDefiniteTransform,
    PositiveTransform,
    SchurStableTransform,
    SimplexTransform,
    SkewSymmetricTransform,
    StiefelTransform,
    SymmetricTransform,
    TransformedParameter,
)


def test_scalar_parameter_transforms_are_strict_and_differentiable():
    raw = jnp.asarray([-3.0, 0.0, 2.0])
    positive = PositiveTransform(0.25)
    bounded = IntervalTransform(-2.0, 3.0)

    positive_value = jax.jit(positive)(raw)
    bounded_value = jax.jit(bounded)(raw)

    assert jnp.all(positive_value > 0.25)
    assert jnp.all((bounded_value > -2.0) & (bounded_value < 3.0))
    assert jnp.all(jnp.isfinite(jax.jacrev(positive)(raw)))
    assert jnp.all(jnp.isfinite(jax.jacrev(bounded)(raw)))


def test_simplex_transform_is_strict_positive_and_identifiable():
    raw = jnp.asarray([[1.0, -2.0], [0.5, 0.25]])
    transform = SimplexTransform()
    value = jax.jit(transform)(raw)

    assert value.shape == (2, 3)
    assert jnp.all(value > 0.0)
    assert jnp.allclose(jnp.sum(value, axis=-1), 1.0)
    assert jnp.all(jnp.isfinite(jax.jacrev(transform)(raw)))


def test_matrix_projection_transforms_enforce_exact_structure():
    raw = jnp.asarray([[1.0, 2.0], [-3.0, 4.0]])
    symmetric = SymmetricTransform()(raw)
    skew = SkewSymmetricTransform()(raw)

    assert jnp.array_equal(symmetric, symmetric.T)
    assert jnp.array_equal(skew, -skew.T)
    assert jnp.array_equal(jnp.diag(skew), jnp.zeros((2,)))


def test_positive_definite_transform_uses_packed_coordinates():
    raw = jnp.asarray([0.2, -0.4, 0.7, 0.3, -0.2, 0.1])
    transform = PositiveDefiniteTransform(1e-4)
    matrix = jax.jit(transform)(raw)

    assert matrix.shape == (3, 3)
    assert jnp.allclose(matrix, matrix.T)
    assert jnp.min(jnp.linalg.eigvalsh(matrix)) > 0.0
    assert jnp.all(jnp.isfinite(jax.jacrev(transform)(raw)))


def test_stability_transforms_enforce_continuous_and_discrete_stability():
    skew_raw = jnp.asarray([[0.1, 1.2], [-0.3, 0.7]])
    damping_raw = jnp.asarray([0.2, -0.1, 0.4])
    raw = (skew_raw, damping_raw)

    continuous = HurwitzTransform(1e-3)(raw)
    symmetric_part = 0.5 * (continuous + continuous.T)
    assert jnp.max(jnp.linalg.eigvalsh(symmetric_part)) < 0.0

    discrete = SchurStableTransform(minimum_damping=1e-3, step=0.25)(raw)
    assert jnp.max(jnp.abs(jnp.linalg.eigvals(discrete))) < 1.0


def test_stiefel_transform_returns_orthonormal_columns():
    raw = jnp.asarray([[1.0, 2.0], [0.5, -1.0], [2.5, 0.25], [-0.3, 0.8]])
    value = StiefelTransform()(raw)
    assert value.shape == (4, 2)
    assert jnp.allclose(value.T @ value, jnp.eye(2), atol=1e-12, rtol=1e-12)


def test_transformed_parameter_exposes_only_raw_coordinates_as_arrays():
    parameter = TransformedParameter(jnp.asarray([-1.0, 0.5]), PositiveTransform(0.1))
    assert jnp.allclose(parameter(), PositiveTransform(0.1)(parameter.raw))
    leaves = jax.tree_util.tree_leaves(parameter)
    assert len(leaves) == 1
    assert leaves[0] is parameter.raw


def test_linear_applies_shape_preserving_weight_transforms_on_demand():
    layer = Linear(
        in_size=2,
        out_size=1,
        rwf=False,
        use_bias=False,
        weight_transform=PositiveTransform(0.1),
        key=jr.key(0),
    )
    layer = eqx.tree_at(lambda node: node.weight, layer, -jnp.ones((1, 2)))

    assert jnp.all(layer.weight < 0.0)
    assert jnp.all(layer(jnp.ones(2)) > 0.0)
    assert jnp.all(jnp.isfinite(jax.jacrev(layer)(jnp.ones(2))))

    stiefel_layer = Linear(
        in_size=2,
        out_size=3,
        rwf=False,
        use_bias=False,
        weight_transform=StiefelTransform(),
        key=jr.key(3),
    )
    effective_weight = stiefel_layer.weight_transform(stiefel_layer.weight)
    assert jnp.allclose(
        effective_weight.T @ effective_weight,
        jnp.eye(2),
        atol=1e-12,
        rtol=1e-12,
    )
    assert stiefel_layer(jnp.ones(2)).shape == (3,)


def test_linear_rejects_incompatible_weight_parameterizations():
    with pytest.raises(ValueError, match="shape-preserving"):
        Linear(
            in_size=2,
            out_size=2,
            rwf=False,
            weight_transform=SimplexTransform(),
            key=jr.key(1),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        Linear(
            in_size=2,
            out_size=2,
            rwf=True,
            weight_transform=PositiveTransform(),
            key=jr.key(2),
        )


def test_transforms_reject_invalid_coordinate_shapes():
    with pytest.raises(ValueError, match="packed-triangle"):
        PositiveDefiniteTransform()(jnp.ones((4,)))
    with pytest.raises(ValueError, match="square matrix"):
        SymmetricTransform()(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="rows >= columns"):
        StiefelTransform()(jnp.ones((2, 3)))
