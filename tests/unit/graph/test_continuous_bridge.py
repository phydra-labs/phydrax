#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


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
