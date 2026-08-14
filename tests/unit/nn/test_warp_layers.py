#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.layers._warp import (
    _normalized_lattice,
    _sample_regular_grid_linear,
)


def _configured_warp(
    *,
    spatial_ndim=1,
    channels=1,
    heads=1,
    boundary="periodic",
    displacement=None,
    fill_value=0.0,
):
    layer = phx.nn.layers.MultiheadWarp(
        spatial_ndim=spatial_ndim,
        in_channels=channels,
        out_channels=channels,
        num_heads=heads,
        boundary=boundary,
        fill_value=fill_value,
        key=jr.key(0),
    )
    if displacement is None:
        displacement = jnp.zeros((heads * spatial_ndim,))
    identity = jnp.eye(channels)
    return eqx.tree_at(
        lambda item: (
            item.value_projection.weight,
            item.value_projection.bias,
            item.displacement_output.weight,
            item.displacement_output.bias,
        ),
        layer,
        (
            identity,
            jnp.zeros((channels,)),
            jnp.zeros_like(layer.displacement_output.weight),
            jnp.asarray(displacement),
        ),
    )


@pytest.mark.parametrize("spatial_shape", ((7,), (4, 5), (3, 4, 5)))
def test_regular_grid_interpolation_zero_displacement_is_identity(spatial_shape):
    spatial_ndim = len(spatial_shape)
    boundary = ("periodic",) * spatial_ndim
    values = jr.normal(jr.key(spatial_ndim), (2,) + spatial_shape + (3,))
    coordinates = _normalized_lattice(
        spatial_shape,
        boundary,
        dtype=values.dtype,
    )
    coordinates = jnp.broadcast_to(coordinates, (2,) + coordinates.shape)

    output = _sample_regular_grid_linear(
        values,
        coordinates,
        spatial_ndim=spatial_ndim,
        boundary=boundary,
        fill_value=0.0,
    )

    assert output.shape == values.shape
    assert jnp.allclose(output, values)


def test_regular_grid_interpolation_reproduces_affine_field_and_case_batches():
    x = jnp.linspace(-1.0, 1.0, 6)
    y = jnp.linspace(-1.0, 1.0, 5)
    field = 1.3 + 2.1 * x[:, None] - 0.7 * y[None, :]
    values = jnp.stack((field, -2.0 * field), axis=-1)
    values = jnp.stack((values, values + jnp.array([4.0, -3.0])), axis=0)
    coordinates = jnp.array(
        [
            [[-0.83, -0.65], [-0.11, 0.24], [0.72, 0.91]],
            [[-0.41, 0.52], [0.37, -0.77], [0.95, -0.12]],
        ]
    )

    output = _sample_regular_grid_linear(
        values,
        coordinates,
        spatial_ndim=2,
        boundary=("clamp", "clamp"),
        fill_value=0.0,
    )
    base = 1.3 + 2.1 * coordinates[..., 0] - 0.7 * coordinates[..., 1]
    expected = jnp.stack((base, -2.0 * base), axis=-1)
    expected = expected.at[1].add(jnp.array([4.0, -3.0]))

    assert output.shape == (2, 3, 2)
    assert jnp.allclose(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("boundary", "displacement", "fill_value", "expected"),
    (
        ("periodic", 0.5, 0.0, jnp.array([1.0, 2.0, 3.0, 0.0])),
        ("reflect", 2.0, 0.0, jnp.array([3.0, 2.0, 1.0, 0.0])),
        ("clamp", 2.0, 0.0, jnp.array([3.0, 3.0, 3.0, 3.0])),
        ("constant", 2.0, -9.0, jnp.array([3.0, -9.0, -9.0, -9.0])),
    ),
)
def test_multihead_warp_boundary_modes_have_exact_semantics(
    boundary,
    displacement,
    fill_value,
    expected,
):
    layer = _configured_warp(
        boundary=boundary,
        displacement=jnp.array([displacement]),
        fill_value=fill_value,
    )

    output = layer(jnp.arange(4.0)[:, None])[:, 0]

    assert jnp.allclose(output, expected)


def test_multihead_warp_periodic_integer_cell_shift_matches_roll():
    size = 9
    layer = _configured_warp(
        channels=2,
        heads=2,
        boundary="periodic",
        displacement=jnp.full((2,), 2.0 / size),
    )
    values = jr.normal(jr.key(1), (3, size, 2))

    output = layer(values)

    assert jnp.allclose(output, jnp.roll(values, -1, axis=-2), atol=1e-6)


def test_multihead_warp_heads_and_mixed_boundaries_remain_independent():
    layer = _configured_warp(
        spatial_ndim=2,
        channels=2,
        heads=2,
        boundary=("periodic", "constant"),
        displacement=jnp.array([0.5, 0.0, 0.0, 2.0]),
        fill_value=-7.0,
    )
    first = jnp.arange(16.0).reshape(4, 4)
    second = 100.0 + first
    values = jnp.stack((first, second), axis=-1)

    output = layer(values)

    assert jnp.allclose(output[..., 0], jnp.roll(first, -1, axis=0))
    assert jnp.allclose(output[:, 0, 1], second[:, -1])
    assert jnp.all(output[:, 1:, 1] == -7.0)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_multihead_warp_eager_jit_and_value_parameter_gradients_are_finite(dtype):
    layer = phx.nn.layers.MultiheadWarp(
        spatial_ndim=2,
        in_channels=3,
        out_channels=4,
        num_heads=2,
        boundary=("reflect", "periodic"),
        key=jr.key(2),
    )
    values = jr.normal(jr.key(3), (2, 4, 5, 3), dtype=dtype)
    eager = layer(values)
    compiled = eqx.filter_jit(lambda item, field: item(field))(layer, values)
    value_gradient = jax.grad(lambda field: jnp.mean(layer(field) ** 2))(values)
    _, parameter_gradient = eqx.filter_value_and_grad(
        lambda item: jnp.mean(item(values) ** 2)
    )(layer)
    parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    assert eager.shape == (2, 4, 5, 4)
    assert eager.dtype == jnp.result_type(dtype, layer.value_projection.weight.dtype)
    assert jnp.allclose(compiled, eager, rtol=2e-5, atol=2e-6)
    assert jnp.all(jnp.isfinite(value_gradient))
    assert jnp.linalg.norm(value_gradient) > 0.0
    assert parameter_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in parameter_leaves)


def test_multihead_warp_validation_rejects_ambiguous_or_unsupported_contracts():
    with pytest.raises(ValueError, match="one, two, or three"):
        phx.nn.layers.MultiheadWarp(
            spatial_ndim=4,
            in_channels=1,
            out_channels=1,
            num_heads=1,
            boundary="periodic",
        )
    with pytest.raises(ValueError, match="divisible by num_heads"):
        phx.nn.layers.MultiheadWarp(
            spatial_ndim=1,
            in_channels=1,
            out_channels=3,
            num_heads=2,
            boundary="periodic",
        )
    with pytest.raises(ValueError, match="one mode per spatial axis"):
        phx.nn.layers.MultiheadWarp(
            spatial_ndim=2,
            in_channels=1,
            out_channels=2,
            num_heads=1,
            boundary=("periodic",),
        )
    with pytest.raises(ValueError, match="boundary modes"):
        phx.nn.layers.MultiheadWarp(
            spatial_ndim=1,
            in_channels=1,
            out_channels=1,
            num_heads=1,
            boundary="invalid",
        )

    layer = _configured_warp()
    with pytest.raises(ValueError, match="at least two nodes"):
        layer(jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="Expected 1 input channels"):
        layer(jnp.ones((4, 2)))
    with pytest.raises(TypeError, match="real-valued"):
        layer(jnp.ones((4, 1), dtype=jnp.complex64))


def test_multihead_warp_conditioning_preserves_unconditioned_path_and_case_isolation():
    settings = dict(
        spatial_ndim=1,
        in_channels=2,
        out_channels=2,
        num_heads=2,
        boundary="periodic",
        key=jr.key(4),
    )
    unconditioned = phx.nn.layers.MultiheadWarp(**settings)
    conditioned = phx.nn.layers.MultiheadWarp(conditioning_size=3, **settings)
    zero_conditioned = eqx.tree_at(
        lambda item: item.displacement_condition.weight,
        conditioned,
        jnp.zeros_like(conditioned.displacement_condition.weight),
    )
    values = jr.normal(jr.key(5), (2, 8, 2))
    conditions = jnp.array([[0.1, -0.2, 0.3], [-0.4, 0.7, 0.2]])

    reference = unconditioned(values)
    zero_context = zero_conditioned(
        values,
        condition=jnp.zeros((2, 3)),
    )
    eager = conditioned(values, condition=conditions)
    compiled = eqx.filter_jit(
        lambda item, field, context: item(field, condition=context)
    )(conditioned, values, conditions)
    separate = jnp.stack(
        tuple(
            conditioned(values[index], condition=conditions[index]) for index in range(2)
        )
    )
    condition_gradient = jax.grad(
        lambda context: jnp.mean(conditioned(values, condition=context) ** 2)
    )(conditions)

    assert jnp.allclose(zero_context, reference)
    assert jnp.allclose(compiled, eager)
    assert jnp.allclose(separate, eager)
    assert jnp.all(jnp.isfinite(condition_gradient))
    assert jnp.linalg.norm(condition_gradient) > 0.0


def test_multihead_warp_conditioning_contract_is_explicit():
    with pytest.raises(ValueError, match="conditioning_size must be non-negative"):
        phx.nn.layers.MultiheadWarp(
            spatial_ndim=1,
            in_channels=1,
            out_channels=1,
            num_heads=1,
            boundary="periodic",
            conditioning_size=-1,
        )

    values = jnp.ones((8, 1))
    plain = _configured_warp()
    conditioned = phx.nn.layers.MultiheadWarp(
        spatial_ndim=1,
        in_channels=1,
        out_channels=1,
        num_heads=1,
        boundary="periodic",
        conditioning_size=1,
        key=jr.key(6),
    )
    with pytest.raises(ValueError, match="condition must be None"):
        plain(values, condition=jnp.ones((1,)))
    with pytest.raises(ValueError, match="requires condition"):
        conditioned(values)
    with pytest.raises(ValueError, match=r"must have shape \(1,\)"):
        conditioned(values, condition=jnp.ones((2,)))
    with pytest.raises(TypeError, match="real-valued conditions"):
        conditioned(values, condition=jnp.ones((1,), dtype=jnp.complex64))
