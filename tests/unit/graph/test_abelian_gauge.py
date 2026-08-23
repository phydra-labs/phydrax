#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _triangle_complex():
    vertex_edge = jnp.asarray([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    edge_face = jnp.ones((3, 1))
    return phx.graph.cochain_complex_from_incidences(
        (3, 3, 1),
        (vertex_edge, edge_face),
        (jnp.ones((3,)), jnp.ones((3,)), jnp.ones((1,))),
    )


def test_abelian_curvature_action_and_gauge_invariance():
    complex = _triangle_complex()
    parameter_values = jnp.zeros((7,)).at[:3].set(jnp.asarray([0.2, -0.1, 0.4]))
    potential_values = jnp.zeros((7,)).at[3:6].set(jnp.asarray([0.3, -0.2, 0.5]))
    parameter = phx.graph.CochainField(complex, parameter_values, 0, field_id="chi")
    potential = phx.graph.CochainField(complex, potential_values, 1, field_id="A")
    transformed = phx.graph.abelian_gauge_transform(potential, parameter)
    curvature = phx.graph.abelian_curvature(potential)
    transformed_curvature = phx.graph.abelian_curvature(transformed)
    assert jnp.allclose(curvature.values, transformed_curvature.values)
    assert jnp.allclose(
        phx.graph.abelian_maxwell_action(potential),
        phx.graph.abelian_maxwell_action(transformed),
    )
    diagnostics = phx.graph.validate_abelian_gauge_system(potential, parameter)
    assert bool(diagnostics.valid)
    assert diagnostics.gauge_curvature_residual < 1e-10
