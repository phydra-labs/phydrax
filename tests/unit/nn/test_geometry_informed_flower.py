#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _model(*, key=jr.key(0), **overrides):
    options = {
        "in_channels": "scalar",
        "out_channels": "scalar",
        "coord_dim": 1,
        "latent_shape": (8,),
        "latent_channels": 4,
        "flower_width": 4,
        "flower_levels": 1,
        "flower_num_heads": 1,
        "flower_groups": 1,
        "encoder_neighbors": 4,
        "decoder_neighbors": 4,
        "transfer_width": 4,
        "transfer_depth": 1,
        "source_key": "u",
    }
    options.update(overrides)
    return phx.nn.GeometryInformedFlower(**options, key=key)


def _point_batch(*, cases=0, condition=None, query_weights=None, query_mask=None):
    nodes = jnp.linspace(0.0, 1.0, 8)
    query_nodes = jnp.linspace(0.05, 0.95, 5)
    values = 1.0 + jnp.sin(2.0 * jnp.pi * nodes)
    source_weights = jnp.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
    if cases:
        coordinates = jnp.broadcast_to(nodes[None, :, None], (cases, 8, 1))
        query_coordinates = jnp.broadcast_to(query_nodes[None, :, None], (cases, 5, 1))
        values = jnp.broadcast_to(values[None, :], (cases, 8))
        source_weights = jnp.broadcast_to(source_weights[None, :], (cases, 8))
        inputs = {
            "u": phx.nn.FunctionSamples(
                values=values,
                coordinates=coordinates,
                quadrature_weights=source_weights,
            )
        }
        if condition is not None:
            inputs["dt"] = phx.nn.FunctionSamples(values=condition)
        return phx.nn.OperatorBatch(
            inputs=inputs,
            queries={
                "query": phx.nn.FunctionSamples(
                    values=None,
                    coordinates=query_coordinates,
                    quadrature_weights=query_weights,
                    mask=query_mask,
                )
            },
            case_axes=("case",),
        )
    return phx.nn.OperatorBatch(
        inputs={
            "u": phx.nn.FunctionSamples(
                values=values,
                coordinates=nodes[:, None],
                quadrature_weights=source_weights,
            )
        },
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=query_nodes[:, None],
                quadrature_weights=query_weights,
                mask=query_mask,
            )
        },
    )


def test_geometry_informed_flower_runs_jit_and_has_finite_gradients():
    model = _model(key=jr.key(1))
    batch = _point_batch()

    output = eqx.filter_jit(lambda current, data: current(data))(model, batch)
    loss, gradient = eqx.filter_value_and_grad(
        lambda current: jnp.mean(current(batch) ** 2)
    )(model)
    gradient_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]

    assert output.shape == (5,)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.isfinite(loss)
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)


@pytest.mark.parametrize("support_kind", ("occupancy", "sdf"))
def test_geometry_informed_flower_projects_explicit_hard_latent_support(support_kind):
    nodes = (jnp.arange(8, dtype=float) + 0.5) / 8.0
    sdf = jnp.abs(nodes - 0.5) - 0.3
    expected_mask = sdf < 0.0
    support_values = expected_mask.astype(float) if support_kind == "occupancy" else sdf
    batch = phx.nn.OperatorBatch(
        inputs={
            "u": phx.nn.FunctionSamples(
                values=jnp.sin(2.0 * jnp.pi * nodes),
                coordinates=nodes[:, None],
                quadrature_weights=jnp.full((8,), 1.0 / 8.0),
                mask=expected_mask,
            ),
            "support": phx.nn.FunctionSamples(
                values=support_values,
                coordinates=nodes[:, None],
            ),
        },
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=nodes[expected_mask, None],
            )
        },
    )
    model = _model(
        key=jr.key(2),
        flower_levels=2,
        latent_support_key="support",
        latent_support_kind=support_kind,
        latent_support_neighbors=1,
    )

    output, diagnostics = model.evaluate_with_diagnostics(batch)

    assert output.shape == (int(jnp.sum(expected_mask)),)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.array_equal(diagnostics.latent_mask, expected_mask)
    assert jnp.allclose(diagnostics.latent_support, support_values)
    assert isinstance(diagnostics.processor, phx.nn.FlowerDiagnostics)
    assert len(diagnostics.processor.blocks) == 3


def test_latent_inverse_distance_support_reproduces_constant_fields_far_away():
    nodes = (jnp.arange(8, dtype=float) + 0.5) / 8.0
    support_coordinates = (100.0 + jnp.arange(8, dtype=float))[:, None]
    batch = phx.nn.OperatorBatch(
        inputs={
            "u": phx.nn.FunctionSamples(
                values=jnp.sin(2.0 * jnp.pi * nodes),
                coordinates=nodes[:, None],
                quadrature_weights=jnp.full((8,), 1.0 / 8.0),
            ),
            "support": phx.nn.FunctionSamples(
                values=jnp.full((8,), 0.25),
                coordinates=support_coordinates,
            ),
        },
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=nodes[:, None],
            )
        },
    )
    model = _model(
        key=jr.key(20),
        latent_support_key="support",
        latent_support_kind="occupancy",
        latent_support_neighbors=4,
        latent_support_threshold=0.1,
    )

    _output, diagnostics = model.evaluate_with_diagnostics(batch)

    assert jnp.allclose(diagnostics.latent_support, 0.25)
    assert jnp.all(diagnostics.latent_mask)


def test_geometry_informed_flower_propagates_case_conditions_and_diagnostics():
    condition = jnp.array([[0.0], [1.0]])
    batch = _point_batch(cases=2, condition=condition)
    model = _model(
        key=jr.key(3),
        conditioning_channels={"dt": 1},
    )

    output, diagnostics = eqx.filter_jit(
        lambda current, data: current.evaluate_with_diagnostics(data)
    )(model, batch)

    assert output.shape == (2, 5)
    assert jnp.max(jnp.abs(output[0] - output[1])) > 1e-8
    assert isinstance(diagnostics, phx.nn.GeometryOperatorDiagnostics)
    assert isinstance(diagnostics.processor, phx.nn.FlowerDiagnostics)
    assert diagnostics.processor.level_shapes == ((8,),)
    assert jnp.allclose(model(batch), output)


def test_geometry_informed_flower_enforces_end_to_end_conservation():
    query_weights = jnp.array([0.1, 0.2, 0.25, 0.25, 0.2])
    query_mask = jnp.array([True, True, False, True, True])
    batch = _point_batch(query_weights=query_weights, query_mask=query_mask)
    model = _model(key=jr.key(4), conserve_mass=True)

    output, diagnostics = model.evaluate_with_diagnostics(batch)
    source = batch.input("u")
    source_mass = jnp.sum(jnp.asarray(source.values) * source.weights())
    target_mass = jnp.sum(output * batch.require_single_query().weights())

    assert output[2] == 0.0
    assert jnp.allclose(target_mass, source_mass, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(diagnostics.source_mass[..., 0], source_mass)
    assert jnp.allclose(diagnostics.target_mass_after_projection[..., 0], source_mass)
    assert diagnostics.conservation_correction is not None

    missing_query_measure = _point_batch()
    with pytest.raises(ValueError, match="explicit query quadrature"):
        model(missing_query_measure)
