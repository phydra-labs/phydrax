import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import Linear
from phydrax.nn.parameters import (
    HurwitzTransform,
    IntervalTransform,
    PackedSkewSymmetricTransform,
    PositiveDefiniteTransform,
    PositiveSemidefiniteTransform,
    PositiveTransform,
    SchurStableTransform,
    SimplexTransform,
    SkewSymmetricTransform,
    StiefelTransform,
    SymmetricTransform,
    TransformedParameter,
)


def test_parameter_transforms_scenario_1() -> None:
    raw = jnp.asarray([-3.0, 0.0, 2.0])
    positive = PositiveTransform(0.25)
    bounded = IntervalTransform(-2.0, 3.0)

    positive_value = jax.jit(positive)(raw)
    bounded_value = jax.jit(bounded)(raw)

    assert jnp.all(positive_value > 0.25)
    assert jnp.all((bounded_value > -2.0) & (bounded_value < 3.0))
    assert jnp.all(jnp.isfinite(jax.jacrev(positive)(raw)))
    assert jnp.all(jnp.isfinite(jax.jacrev(bounded)(raw)))
    raw = jnp.asarray([[1.0, -2.0], [0.5, 0.25]])
    transform = SimplexTransform()
    value = jax.jit(transform)(raw)

    assert value.shape == (2, 3)
    assert jnp.all(value > 0.0)
    assert jnp.allclose(jnp.sum(value, axis=-1), 1.0)
    assert jnp.all(jnp.isfinite(jax.jacrev(transform)(raw)))
    raw_matrix = jnp.asarray([[1.0, 2.0], [-3.0, 4.0]])
    symmetric = SymmetricTransform()(raw_matrix)
    skew = SkewSymmetricTransform()(raw_matrix)
    assert jnp.array_equal(symmetric, symmetric.T)
    assert jnp.array_equal(skew, -skew.T)
    assert jnp.array_equal(jnp.diag(skew), jnp.zeros((2,)))

    positive_raw = jnp.asarray([0.2, -0.4, 0.7, 0.3, -0.2, 0.1])
    positive = PositiveDefiniteTransform(1e-4)
    positive_matrix = jax.jit(positive)(positive_raw)
    assert positive_matrix.shape == (3, 3)
    assert jnp.allclose(positive_matrix, positive_matrix.T)
    assert jnp.min(jnp.linalg.eigvalsh(positive_matrix)) > 0.0
    assert jnp.all(jnp.isfinite(jax.jacrev(positive)(positive_raw)))

    packed_raw = jnp.asarray([[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]])
    packed = PackedSkewSymmetricTransform()
    packed_matrices = jax.jit(packed)(packed_raw)
    assert packed_matrices.shape == (2, 3, 3)
    assert jnp.array_equal(
        packed_matrices,
        -jnp.swapaxes(packed_matrices, -1, -2),
    )
    assert jnp.array_equal(
        jnp.diagonal(packed_matrices, axis1=-2, axis2=-1),
        jnp.zeros((2, 3)),
    )
    assert jnp.all(jnp.isfinite(jax.jacrev(packed)(packed_raw)))

    semidefinite_raw = jnp.asarray([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])
    semidefinite = PositiveSemidefiniteTransform()
    factors = jax.jit(semidefinite.factor)(semidefinite_raw)
    semidefinite_matrices = semidefinite(semidefinite_raw)
    assert factors.shape == (2, 2, 2)
    assert jnp.array_equal(factors[0], jnp.zeros((2, 2)))
    assert jnp.array_equal(semidefinite_matrices[0], jnp.zeros((2, 2)))
    assert jnp.all(jnp.linalg.eigvalsh(semidefinite_matrices) >= -1e-12)
    assert jnp.all(jnp.isfinite(jax.jacrev(semidefinite)(semidefinite_raw)))

    stability_raw = (
        jnp.asarray([[0.1, 1.2], [-0.3, 0.7]]),
        jnp.asarray([0.2, -0.1, 0.4]),
    )
    continuous = HurwitzTransform(1e-3)(stability_raw)
    assert jnp.max(jnp.linalg.eigvalsh(0.5 * (continuous + continuous.T))) < 0.0
    discrete = SchurStableTransform(minimum_damping=1e-3, step=0.25)(stability_raw)
    assert jnp.max(jnp.abs(jnp.linalg.eigvals(discrete))) < 1.0

    stiefel = StiefelTransform()(
        jnp.asarray([[1.0, 2.0], [0.5, -1.0], [2.5, 0.25], [-0.3, 0.8]])
    )
    assert stiefel.shape == (4, 2)
    assert jnp.allclose(stiefel.T @ stiefel, jnp.eye(2), atol=1e-12, rtol=1e-12)


def test_parameter_transforms_scenario_2() -> None:
    parameter = TransformedParameter(
        jnp.asarray([-1.0, 0.5]),
        PositiveTransform(0.1),
    )
    assert jnp.allclose(parameter(), PositiveTransform(0.1)(parameter.raw))
    leaves = jax.tree_util.tree_leaves(parameter)
    assert len(leaves) == 1
    assert leaves[0] is parameter.raw

    positive_layer = Linear(
        in_size=2,
        out_size=1,
        rwf=False,
        use_bias=False,
        weight_transform=PositiveTransform(0.1),
        key=jr.key(0),
    )
    positive_layer = eqx.tree_at(
        lambda node: node.weight,
        positive_layer,
        -jnp.ones((1, 2)),
    )
    assert jnp.all(positive_layer.weight < 0.0)
    assert jnp.all(positive_layer(jnp.ones(2)) > 0.0)
    assert jnp.all(jnp.isfinite(jax.jacrev(positive_layer)(jnp.ones(2))))

    stiefel_layer = Linear(
        in_size=2,
        out_size=3,
        rwf=False,
        use_bias=False,
        weight_transform=StiefelTransform(),
        key=jr.key(3),
    )
    # ty: ignore[call-non-callable]
    effective_weight = stiefel_layer.weight_transform(stiefel_layer.weight)
    assert jnp.allclose(
        effective_weight.T @ effective_weight,
        jnp.eye(2),
        atol=1e-12,
        rtol=1e-12,
    )
    assert stiefel_layer(jnp.ones(2)).shape == (3,)

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
    with pytest.raises(ValueError, match="packed-triangle"):
        PositiveDefiniteTransform()(jnp.ones((4,)))
    with pytest.raises(ValueError, match="strict-triangle"):
        PackedSkewSymmetricTransform()(jnp.ones((2,)))
    with pytest.raises(ValueError, match="packed-triangle"):
        PositiveSemidefiniteTransform()(jnp.ones((4,)))
    with pytest.raises(ValueError, match="square matrix"):
        SymmetricTransform()(jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="rows >= columns"):
        StiefelTransform()(jnp.ones((2, 3)))
