#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import ModifiedMLP, SeparableModifiedMLP


@pytest.mark.parametrize(
    ("in_size", "out_size", "input_shape", "output_shape"),
    [
        ("scalar", "scalar", (), ()),
        (3, 2, (3,), (2,)),
        ((2, 2), (2, 1), (2, 2), (2, 1)),
    ],
)
def test_modified_mlp_value_shapes(
    in_size,
    out_size,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
):
    model = ModifiedMLP(
        in_size=in_size,
        out_size=out_size,
        width_size=8,
        depth=3,
        key=jr.key(0),
    )
    value = model(jnp.ones(input_shape))
    assert value.shape == output_shape


def test_modified_mlp_uses_persistent_encoder_gate_formula():
    model = ModifiedMLP(
        in_size=2,
        out_size="scalar",
        width_size=4,
        depth=3,
        key=jr.key(1),
    )
    x = jnp.asarray((0.2, -0.7))
    encoder_u = model.encoder_u(x)
    encoder_v = model.encoder_v(x)
    hidden = model._mix(model.layers[0](x), encoder_u, encoder_v)
    for layer in model.layers[1:-1]:
        hidden = model._mix(layer(hidden), encoder_u, encoder_v)
    expected = model.final_activation(model.layers[-1](hidden))
    assert jnp.allclose(model(x), expected)


def test_modified_mlp_scan_matches_unrolled_execution():
    model = ModifiedMLP(
        in_size=3,
        out_size=2,
        width_size=8,
        depth=4,
        scan=True,
        key=jr.key(2),
    )
    unrolled = eqx.tree_at(
        lambda value: (value.scan, value._scan_enabled, value._scan_static),
        model,
        (False, False, None),
    )
    x = jnp.asarray((0.1, 0.3, -0.2))
    assert jnp.allclose(eqx.filter_jit(model)(x), unrolled(x))


def test_modified_mlp_is_differentiable_through_input_and_parameters():
    model = ModifiedMLP(
        in_size=2,
        out_size="scalar",
        width_size=6,
        depth=2,
        key=jr.key(3),
    )
    x = jnp.asarray((0.25, -0.5))
    input_gradient = jax.grad(model)(x)
    parameter_gradient = eqx.filter_grad(lambda value: jnp.square(value(x)))(model)
    leaves = jax.tree.leaves(parameter_gradient, is_leaf=lambda value: value is None)
    assert input_gradient.shape == x.shape
    assert jnp.all(jnp.isfinite(input_gradient))
    assert any(
        leaf is not None and jnp.any(jnp.asarray(leaf) != 0.0) for leaf in leaves
    )


def test_separable_modified_mlp_supports_dense_and_coordinate_inputs():
    model = SeparableModifiedMLP(
        in_size=2,
        out_size="scalar",
        latent_size=4,
        width_size=8,
        depth=2,
        key=jr.key(4),
    )
    dense = model(jnp.asarray((0.2, -0.3)))
    grid = model((jnp.linspace(-1.0, 1.0, 5), jnp.linspace(0.0, 1.0, 7)))
    assert dense.shape == ()
    assert grid.shape == (5, 7)


def test_separable_modified_mlp_split_input_contract():
    model = SeparableModifiedMLP(
        in_size="scalar",
        out_size=2,
        split_input=3,
        latent_size=4,
        width_size=8,
        depth=2,
        key=jr.key(5),
    )
    value = model(jnp.asarray(0.25))
    assert value.shape == (2,)


def test_modified_mlp_supports_zero_bias_initialization():
    model = ModifiedMLP(
        in_size=2,
        out_size="scalar",
        width_size=4,
        depth=2,
        bias_init_lim=0.0,
        key=jr.key(6),
    )
    biases = (
        model.encoder_u.bias,
        model.encoder_v.bias,
        *(layer.bias for layer in model.layers),
    )
    assert all(bias is not None and jnp.all(bias == 0.0) for bias in biases)


def test_modified_mlp_rejects_nonpositive_width_and_depth():
    with pytest.raises(ValueError, match="width_size must be positive"):
        ModifiedMLP(
            in_size=2,
            out_size=1,
            width_size=0,
            depth=2,
            key=jr.key(6),
        )
    with pytest.raises(ValueError, match="depth must be positive"):
        ModifiedMLP(
            in_size=2,
            out_size=1,
            width_size=4,
            depth=0,
            key=jr.key(6),
        )
