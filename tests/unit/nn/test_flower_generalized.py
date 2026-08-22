#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _flower(*, spatial_ndim=1, key=jr.key(0), **kwargs):
    settings = dict(
        in_channels="scalar",
        out_channels="scalar",
        spatial_ndim=spatial_ndim,
        boundary=("clamp",) * spatial_ndim,
        width=2,
        levels=1,
        num_heads=1,
        groups=1,
        coordinate_embedding=False,
        key=key,
    )
    settings.update(kwargs)
    return phx.nn.operator.architectures.Flower(**settings)


def _axis(name, nodes, *, periodic=False, quadrature_weights=None):
    return phx.nn.operator.OperatorAxis(
        name,
        jnp.asarray(nodes),
        periodic=periodic,
        quadrature_weights=quadrature_weights,
    )


def _batch(values, axes, query, *, source_mask=None):
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(
                values=values,
                axes=axes,
                mask=source_mask,
            )
        },
        queries={"query": query},
    )


def test_flower_omitted_generalized_options_preserve_explicit_default_execution():
    settings = dict(
        in_channels="scalar",
        out_channels="scalar",
        spatial_ndim=1,
        boundary="clamp",
        width=2,
        levels=1,
        num_heads=1,
        groups=1,
        coordinate_embedding=False,
        key=jr.key(1),
    )
    implicit = phx.nn.operator.architectures.Flower(**settings)
    explicit = phx.nn.operator.architectures.Flower(
        **settings,
        fill_value=0.0,
        transition_mode="learned",
        query_mode="coincident",
        source_mask_mode="reject",
        probabilistic_routing=False,
        minimum_route_scale=1e-6,
        route_scale_factor=1e-3,
        conserve_mass=False,
    )
    nodes = jnp.linspace(-1.0, 1.0, 5)
    values = jnp.sin(jnp.pi * nodes)

    implicit_output = implicit((values, nodes))
    explicit_output = explicit((values, nodes))

    assert implicit.transition_mode == "learned"
    assert implicit.query_mode == "coincident"
    assert implicit.source_mask_mode == "reject"
    assert not implicit.probabilistic_routing
    assert not implicit.conserve_mass
    assert jnp.array_equal(implicit_output, explicit_output)


@pytest.mark.parametrize("spatial_ndim", (1, 2, 3))
def test_resolution_consistent_flower_executes_in_one_two_and_three_dimensions(
    spatial_ndim,
):
    nodes = tuple(jnp.linspace(-1.0, 1.0, 4) for _ in range(spatial_ndim))
    coordinates = jnp.meshgrid(*nodes, indexing="ij")
    values = jnp.asarray(
        sum((axis + 1.0) ** (index + 1) for index, axis in enumerate(coordinates))
    )
    model = _flower(
        spatial_ndim=spatial_ndim,
        levels=2,
        transition_mode="resolution_consistent",
        key=jr.key(10 + spatial_ndim),
    )

    output, diagnostics = model.evaluate_with_diagnostics((values,) + nodes)

    assert output.shape == values.shape
    assert jnp.all(jnp.isfinite(output))
    assert diagnostics.transition_mode == "resolution_consistent"
    assert diagnostics.level_shapes == ((4,) * spatial_ndim, (2,) * spatial_ndim)


def test_resolution_consistent_flower_supports_nonuniform_nodes_eager_and_jit():
    nodes = jnp.array([-1.0, -0.72, -0.1, 1.0])
    values = 0.5 + nodes + nodes**2
    model = _flower(
        levels=2,
        transition_mode="resolution_consistent",
        key=jr.key(20),
    )

    eager = model((values, nodes))
    compiled = eqx.filter_jit(lambda current, field, grid: current((field, grid)))(
        model, values, nodes
    )

    assert eager.shape == values.shape
    assert jnp.all(jnp.isfinite(eager))
    assert jnp.allclose(compiled, eager, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mask_mode", ("renormalize", "strict"))
def test_flower_source_holes_are_supported_and_remain_masked(mask_mode):
    nodes = jnp.linspace(-1.0, 1.0, 5)
    axis = _axis("x", nodes)
    source_mask = jnp.array([True, False, True, True, True])
    values = jnp.array([1.0, 1000.0, -2.0, 3.0, 0.5])
    batch = _batch(
        values,
        (axis,),
        phx.nn.operator.FunctionSamples(values=None, axes=(axis,)),
        source_mask=source_mask,
    )
    changed_batch = _batch(
        values.at[1].set(-1000.0),
        (axis,),
        phx.nn.operator.FunctionSamples(values=None, axes=(axis,)),
        source_mask=source_mask,
    )
    nan_values = values.at[1].set(jnp.nan)
    nan_batch = _batch(
        nan_values,
        (axis,),
        phx.nn.operator.FunctionSamples(values=None, axes=(axis,)),
        source_mask=source_mask,
    )
    model = _flower(
        source_key="state",
        source_mask_mode=mask_mode,
        transition_mode="resolution_consistent",
        key=jr.key(30),
    )

    output = model(batch)
    changed_output = model(changed_batch)
    nan_output = eqx.filter_jit(lambda current, data: current(data))(model, nan_batch)

    def squared_output(field):
        data = _batch(
            field,
            (axis,),
            phx.nn.operator.FunctionSamples(values=None, axes=(axis,)),
            source_mask=source_mask,
        )
        return jnp.sum(model(data) ** 2)

    nan_gradient = jax.grad(squared_output)(nan_values)

    assert output.shape == values.shape
    assert jnp.all(jnp.isfinite(output))
    assert jnp.array_equal(output[~source_mask], jnp.zeros((1,)))
    assert jnp.allclose(changed_output, output)
    assert jnp.all(jnp.isfinite(nan_output))
    assert jnp.allclose(nan_output, output)
    assert jnp.all(jnp.isfinite(nan_gradient))
    assert nan_gradient[1] == 0.0


@pytest.mark.parametrize("query_kind", ("tensor_grid", "points"))
def test_interpolating_flower_accepts_arbitrary_tensor_grid_and_point_queries(
    query_kind,
):
    source_x = _axis("x", jnp.array([-1.0, -0.4, 0.25, 1.0]))
    source_y = _axis("y", jnp.array([-1.0, -0.55, 0.3, 1.0]))
    x, y = jnp.meshgrid(source_x.nodes, source_y.nodes, indexing="ij")
    values = jnp.sin(x) + 0.5 * y
    if query_kind == "tensor_grid":
        query_x = _axis("x", jnp.array([-0.8, 0.0, 0.7]))
        query_y = _axis("y", jnp.array([-0.9, -0.25, 0.1, 0.55, 0.9]))
        query = phx.nn.operator.FunctionSamples(values=None, axes=(query_x, query_y))
        expected_shape = (3, 5)
    else:
        coordinates = jnp.array(
            [
                [-0.8, -0.9],
                [-0.2, 0.6],
                [0.0, 0.0],
                [0.45, -0.3],
                [0.8, 0.9],
            ]
        )
        query = phx.nn.operator.FunctionSamples(values=None, coordinates=coordinates)
        expected_shape = (5,)
    batch = _batch(values, (source_x, source_y), query)
    model = _flower(
        spatial_ndim=2,
        source_key="state",
        query_mode="interpolate",
        transition_mode="resolution_consistent",
        key=jr.key(40),
    )

    output = model(batch)

    assert output.shape == expected_shape
    assert jnp.all(jnp.isfinite(output))


def test_conservative_flower_matches_source_and_arbitrary_query_mass():
    source_weights = jnp.array([0.3, 0.65, 0.7, 0.35])
    source_axis = _axis(
        "x",
        jnp.array([-1.0, -0.4, 0.3, 1.0]),
        quadrature_weights=source_weights,
    )
    values = jnp.array([1.0, 2.0, -1.0, 4.0])
    query_weights = jnp.array([0.4, 0.9, 0.7])
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.array([[-0.8], [-0.1], [0.55]]),
        quadrature_weights=query_weights,
    )
    batch = _batch(values, (source_axis,), query)
    model = _flower(
        source_key="state",
        query_mode="interpolate",
        transition_mode="resolution_consistent",
        conserve_mass=True,
        key=jr.key(50),
    )

    output = model(batch)
    source_mass = jnp.sum(source_weights * values)
    query_mass = jnp.sum(query_weights * output)

    assert output.shape == (3,)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.allclose(query_mass, source_mass, rtol=1e-5, atol=1e-6)


def test_probabilistic_flower_is_repeatable_and_reports_every_sampled_block():
    nodes = -1.0 + 2.0 * jnp.arange(4, dtype=float) / 4.0
    values = jnp.sin(jnp.pi * nodes) + 0.2 * jnp.cos(2.0 * jnp.pi * nodes)
    model = _flower(
        boundary=("periodic",),
        levels=2,
        probabilistic_routing=True,
        route_scale_factor=0.1,
        key=jr.key(60),
    )
    data = (values, nodes)
    sample_key = jr.key(61)

    mean_first = model(data)
    mean_second = model(data)
    sampled_first = model(data, key=sample_key)
    sampled_second = model(data, key=sample_key)
    other_sample = model(data, key=jr.key(62))
    diagnosed_output, diagnostics = model.evaluate_with_diagnostics(
        data,
        key=sample_key,
    )

    assert jnp.array_equal(mean_first, mean_second)
    assert jnp.array_equal(sampled_first, sampled_second)
    assert jnp.allclose(diagnosed_output, sampled_first, rtol=1e-6, atol=1e-6)
    assert not jnp.allclose(sampled_first, other_sample, rtol=1e-6, atol=1e-7)
    assert len(diagnostics.blocks) == 2 * model.levels - 1
    assert len(diagnostics.level_shapes) == model.levels
    assert all(block.route_scale is not None for block in diagnostics.blocks)
    assert all(jnp.all(jnp.isfinite(block.route_scale)) for block in diagnostics.blocks)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"transition_mode": "invalid"}, "transition_mode"),
        ({"query_mode": "invalid"}, "query_mode"),
        ({"source_mask_mode": "invalid"}, "source_mask_mode"),
        (
            {"levels": 2, "source_mask_mode": "renormalize"},
            "resolution_consistent",
        ),
        ({"out_channels": 2, "conserve_mass": True}, "equal input and output"),
        ({"minimum_route_scale": 0.0}, "scales must be positive"),
        ({"route_scale_factor": 0.0}, "scales must be positive"),
        ({"boundary": ("clamp", "periodic")}, "one mode per spatial axis"),
    ),
)
def test_flower_rejects_invalid_generalized_configurations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _flower(**overrides)
