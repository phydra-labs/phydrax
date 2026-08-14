#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _uniform_interval_bridge(segment_count, chart):
    incidence = (
        jnp.zeros((segment_count + 1, segment_count))
        .at[jnp.arange(segment_count), jnp.arange(segment_count)]
        .set(-1.0)
        .at[jnp.arange(segment_count) + 1, jnp.arange(segment_count)]
        .set(1.0)
    )
    spacing = 1.0 / segment_count
    vertex_coordinates = jnp.linspace(0.0, 1.0, segment_count + 1)
    complex_ir = phx.graph.cochain_complex_from_incidences(
        (segment_count + 1, segment_count),
        (incidence,),
        (jnp.ones((segment_count + 1,)), jnp.ones((segment_count,))),
        coordinates=(
            vertex_coordinates[:, None],
            ((jnp.arange(segment_count) + 0.5) * spacing)[:, None],
        ),
    )
    vertices = phx.graph.OrientedCellParameterization(
        0,
        segment_count + 1,
        1,
        lambda cell, reference: vertex_coordinates[cell, None],
        lambda cell, reference: jnp.zeros((1, 0)),
        jnp.zeros((1, 0)),
        jnp.ones((1,)),
        jnp.ones((segment_count + 1,)),
    )
    edges = phx.graph.OrientedCellParameterization(
        1,
        segment_count,
        1,
        lambda cell, reference: jnp.asarray([(cell + reference[0]) * spacing]),
        lambda cell, reference: jnp.asarray([[spacing]]),
        jnp.asarray([[0.5]]),
        jnp.ones((1,)),
        jnp.ones((segment_count,)),
    )
    return phx.graph.ContinuousCochainBridge(
        complex_ir,
        chart,
        (vertices, edges),
    )


def test_continuous_cochain_projection_converges_under_uniform_refinement():
    chart = phx.metrix.CoordinateChart("refinement_line", ("x",))
    scalar = phx.metrix.DifferentialForm(
        lambda point: jnp.exp(point[0]),
        chart=chart,
        degree=0,
    )
    one_form = phx.metrix.DifferentialForm(
        lambda point: jnp.asarray([jnp.exp(point[0])]),
        chart=chart,
        degree=1,
    )
    quadrature_errors = []
    stokes_reports = []

    for segment_count in (4, 8, 16, 32):
        bridge = _uniform_interval_bridge(segment_count, chart)
        projection = phx.graph.integrate_form_to_cochain(one_form, bridge)
        edge_values = projection.values[bridge.complex.cell_entities(1)]
        quadrature_errors.append(jnp.abs(jnp.sum(edge_values) - (jnp.e - 1.0)))
        stokes_reports.append(
            phx.graph.validate_stokes_bridge(scalar, bridge, tolerance=1e-5)
        )

    for coarse, fine in zip(quadrature_errors, quadrature_errors[1:]):
        assert fine < coarse / 3.9
    for coarse, fine in zip(stokes_reports, stokes_reports[1:]):
        assert fine.maximum_residual < coarse.maximum_residual / 7.0
    assert not bool(stokes_reports[-2].valid)
    assert bool(stokes_reports[-1].valid)


def test_oriented_segment_bridge_commutes_with_exterior_derivative():
    complex = phx.graph.cochain_complex_from_incidences(
        (2, 1),
        (jnp.array([[-1.0], [1.0]]),),
        (jnp.ones((2,)), jnp.ones((1,))),
        coordinates=(jnp.array([[0.0], [1.0]]), jnp.array([[0.5]])),
    )
    chart = phx.metrix.CoordinateChart("line", ("x",))
    vertices = phx.graph.OrientedCellParameterization(
        0,
        2,
        1,
        lambda cell, reference: jnp.array([cell], dtype=float),
        lambda cell, reference: jnp.zeros((1, 0)),
        jnp.zeros((1, 0)),
        jnp.ones((1,)),
        jnp.ones((2,)),
    )
    edge = phx.graph.OrientedCellParameterization(
        1,
        1,
        1,
        lambda cell, reference: jnp.array([reference[0]]),
        lambda cell, reference: jnp.ones((1, 1)),
        jnp.array([[0.5]]),
        jnp.ones((1,)),
        jnp.ones((1,)),
    )
    bridge = phx.graph.ContinuousCochainBridge(
        complex,
        chart,
        (vertices, edge),
    )
    scalar = phx.metrix.DifferentialForm(
        lambda point: point[0] ** 2,
        chart=chart,
        degree=0,
    )
    projection = phx.graph.integrate_form_to_cochain(scalar, bridge)
    report = phx.graph.validate_stokes_bridge(scalar, bridge)

    assert projection.spec.sampling == "point_value"
    assert jnp.allclose(projection.values, jnp.array([0.0, 1.0, 0.0]))
    assert bool(report.valid)
    assert report.maximum_residual < 1e-12


def test_zero_cell_parameterization_enforces_point_value_semantics():
    with pytest.raises(ValueError, match="Zero-cell sampling"):
        phx.graph.OrientedCellParameterization(
            0,
            2,
            1,
            lambda cell, reference: jnp.array([cell], dtype=float),
            lambda cell, reference: jnp.zeros((1, 0)),
            jnp.zeros((1, 0)),
            jnp.array([2.0]),
            jnp.ones((2,)),
        )
    with pytest.raises(ValueError, match="Zero-cell sampling"):
        phx.graph.OrientedCellParameterization(
            0,
            2,
            1,
            lambda cell, reference: jnp.array([cell], dtype=float),
            lambda cell, reference: jnp.zeros((1, 0)),
            jnp.zeros((1, 0)),
            jnp.ones((1,)),
            jnp.array([1.0, -1.0]),
        )
