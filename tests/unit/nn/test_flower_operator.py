#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _axis(size=8):
    return phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(size, dtype=float) / size,
        quadrature_weights=jnp.full((size,), 1.0 / size),
        periodic=True,
    )


def _conditioned_batch(values, conditions, *, query_mask=None):
    values = jnp.asarray(values)
    conditions = jnp.asarray(conditions)
    case_axes = ("case",) if values.ndim == 2 else ()
    axis = _axis(values.shape[-1])
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(values=values, axes=(axis,)),
            "shift": phx.nn.operator.FunctionSamples(values=conditions),
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
                mask=query_mask,
            )
        },
        case_axes=case_axes,
    )


def _conditioned_flower(key):
    return phx.nn.operator.architectures.Flower(
        in_channels="scalar",
        out_channels="scalar",
        spatial_ndim=1,
        boundary="periodic",
        width=4,
        levels=1,
        num_heads=2,
        groups=2,
        coordinate_embedding=False,
        source_key="state",
        conditioning_channels={"shift": "scalar"},
        key=key,
    )


def test_flower_routes_case_context_to_warp_and_preserves_case_isolation():
    model = _conditioned_flower(jr.key(0))
    nodes = _axis().nodes
    field = jnp.sin(2.0 * jnp.pi * nodes)
    values = jnp.stack((field, field))
    conditions = jnp.array([0.0, 1.0])
    batch = _conditioned_batch(values, conditions)

    output = model(batch)
    compiled = eqx.filter_jit(lambda current, data: current(data))(model, batch)
    separate = jnp.stack(
        tuple(
            model(_conditioned_batch(values[index], conditions[index]))
            for index in range(2)
        )
    )
    lifted = model.lift(values[..., None])
    zero_warp = model.bottleneck.warp(
        lifted,
        condition=jnp.zeros((2, 1)),
    )
    conditioned_warp = model.bottleneck.warp(
        lifted,
        condition=conditions[:, None],
    )

    assert model.bottleneck.warp.conditioning_size == 1
    assert model.bottleneck.warp.displacement_condition is not None
    assert output.shape == (2, 8)
    assert jnp.allclose(compiled, output)
    assert jnp.allclose(separate, output)
    assert not jnp.allclose(conditioned_warp, zero_warp)
    assert not jnp.allclose(output[0], output[1])


def test_conditioned_flower_applies_query_masks_after_context_routing():
    model = _conditioned_flower(jr.key(1))
    values = jr.normal(jr.key(2), (2, 8))
    conditions = jnp.array([-0.5, 0.75])
    query_mask = jnp.ones((2, 8), dtype=bool)
    query_mask = query_mask.at[0, -1].set(False).at[1, 0].set(False)
    batch = _conditioned_batch(
        values,
        conditions,
        query_mask=query_mask,
    )

    output = model(batch)

    assert output.shape == (2, 8)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.array_equal(output[~query_mask], jnp.zeros((2,)))


def test_flower_scalar_case_count_equal_to_grid_size_is_not_a_channel_axis():
    size = 8
    nodes = jnp.arange(size, dtype=float) / size
    values = jr.normal(jr.key(3), (size, size))
    model = phx.nn.operator.architectures.Flower(
        in_channels="scalar",
        out_channels="scalar",
        spatial_ndim=1,
        boundary="periodic",
        width=4,
        levels=1,
        num_heads=2,
        groups=2,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(4),
    )

    direct = model((values, nodes))
    separate = jnp.stack(tuple(model((values[index], nodes)) for index in range(size)))
    axis = _axis(size)
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(values=values, axes=(axis,)),
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
    )
    structured = model(batch)

    assert direct.shape == (size, size)
    assert jnp.allclose(direct, separate)
    assert jnp.allclose(structured, direct)
