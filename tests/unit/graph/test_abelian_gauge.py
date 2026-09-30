#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import simplicial_cell_geometry


def _triangle_complex() -> phx.discretization.CochainDiscretization:
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32), None, 3
    )
    return phx.discretization.CochainDiscretization(
        topology,
        tuple(
            phx.discretization.DiagonalHodge(jnp.ones((count,))) for count in (3, 3, 1)
        ),
    )


def test_abelian_curvature_action_and_gauge_invariance() -> None:
    complex = _triangle_complex()
    cells, _ = simplicial_cell_geometry(complex.topology)
    edge_potentials = {(0, 1): 0.3, (0, 2): -0.2, (1, 2): 0.5}
    face_boundary = {(0, 1): 1.0, (0, 2): -1.0, (1, 2): 1.0}
    potential_values = jnp.asarray([edge_potentials[tuple(edge)] for edge in cells[1]])
    residual_values = jnp.asarray([face_boundary[tuple(edge)] for edge in cells[1]])
    parameter = phx.exterior.DiscreteForm(
        complex.realization_id, complex.form_type(0), jnp.asarray([0.2, -0.1, 0.4])
    )
    potential = phx.exterior.DiscreteForm(
        complex.realization_id, complex.form_type(1), potential_values
    )
    transformed = phx.graph.abelian_gauge_transform(complex, potential, parameter)
    curvature = phx.graph.abelian_curvature(complex, potential)
    transformed_curvature = phx.graph.abelian_curvature(complex, transformed)
    # Independent oriented boundary: A(01) + A(12) - A(02) = 1.
    assert jnp.allclose(curvature.values, jnp.asarray([1.0]))
    assert jnp.allclose(curvature.values, transformed_curvature.values)
    assert jnp.allclose(phx.graph.abelian_maxwell_action(complex, potential), 0.5)
    assert jnp.allclose(phx.graph.abelian_maxwell_action(complex, transformed), 0.5)
    residual, action = phx.graph.AbelianMaxwellOperator(complex)(potential)
    assert jnp.allclose(residual.values, residual_values)
    assert jnp.allclose(action, 0.5)
    diagnostics = phx.graph.validate_abelian_gauge_system(complex, potential, parameter)
    assert bool(diagnostics.valid)
    assert diagnostics.gauge_curvature_residual < 1e-10


def test_abelian_refuses_foreign_realization_and_dual_placement() -> None:
    complex = _triangle_complex()
    foreign = phx.exterior.DiscreteForm("foreign", complex.form_type(1), jnp.ones((3,)))
    with pytest.raises(ValueError):
        phx.graph.abelian_curvature(complex, foreign)
    dual = phx.exterior.DiscreteForm(
        complex.realization_id, complex.form_type(1).with_twist("twisted"), jnp.ones((3,))
    )
    with pytest.raises(ValueError):
        phx.graph.abelian_curvature(complex, dual)
