import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import AdaptiveResidual


class _AffineBranch(eqx.Module):
    scale: jax.Array

    def __call__(self, x, context=0.0, *, key=None):
        del key
        return self.scale * x + context


class _RandomBranch(eqx.Module):
    def __call__(self, x, *, key):
        return x + jr.normal(key, x.shape)


def test_adaptive_residual_is_exact_identity_at_initialization():
    layer = AdaptiveResidual(_AffineBranch(jnp.asarray(2.0)))
    x = jnp.asarray([-1.0, 0.5, 3.0])

    assert jnp.array_equal(eqx.filter_jit(layer)(x), x)
    alpha_gradient = jax.grad(
        lambda alpha: eqx.tree_at(lambda node: node.alpha, layer, alpha)(x).sum()
    )(layer.alpha)
    assert jnp.allclose(alpha_gradient, x.sum())


def test_adaptive_residual_is_exact_branch_at_unit_gate_and_forwards_context():
    layer = AdaptiveResidual(_AffineBranch(jnp.asarray(-0.5)), initial_alpha=1.0)
    x = jnp.asarray([1.0, -2.0])

    assert jnp.array_equal(layer(x, jnp.asarray(0.25)), -0.5 * x + 0.25)


def test_adaptive_residual_channel_gates_broadcast_over_leading_axes():
    layer = AdaptiveResidual(
        lambda x: 3.0 * x,
        channel_size=3,
        initial_alpha=jnp.asarray([0.0, 0.5, 1.0]),
    )
    x = jnp.ones((2, 4, 3))

    expected = jnp.broadcast_to(jnp.asarray([1.0, 2.0, 3.0]), x.shape)
    assert jnp.array_equal(eqx.filter_jit(layer)(x), expected)


def test_adaptive_residual_propagates_explicit_random_keys():
    layer = AdaptiveResidual(_RandomBranch(), initial_alpha=1.0)
    x = jnp.zeros(5)
    key = jr.key(4)

    assert jnp.array_equal(layer(x, key=key), jr.normal(key, x.shape))


def test_adaptive_residual_rejects_shape_changing_branches():
    layer = AdaptiveResidual(lambda x: x[..., :1])
    with pytest.raises(ValueError, match="output shape"):
        layer(jnp.ones((2, 3)))
