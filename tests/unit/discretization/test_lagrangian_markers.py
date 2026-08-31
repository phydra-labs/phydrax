#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_lagrangian_marker_measure_compacts_active_constraint_rows():
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([10, 20, 30]),
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        jnp.asarray([0.25, 9.0, 0.75]),
        active_mask=jnp.asarray([True, False, True]),
    ).prepare()

    assert markers.capacity == 3
    assert markers.active_count == 2
    assert markers.active_velocity_space.structure().shape == (2, 2)
    assert jnp.array_equal(markers.active_indices, jnp.asarray([0, 2]))
    assert jnp.isclose(markers.material_measure.total_mass, 1.0)
    expanded = markers.expand_active(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert jnp.array_equal(expanded[1], jnp.zeros((2,)))


def test_lagrangian_marker_kinematics_masks_inactive_values():
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([0, 1]),
        jnp.zeros((2, 2)),
        jnp.asarray([1.0, 0.0]),
        active_mask=jnp.asarray([True, False]),
    ).prepare()
    state = markers.kinematics(
        jnp.asarray([[0.2, 0.3], [jnp.nan, jnp.nan]]),
        jnp.asarray([[0.1, -0.1], [jnp.nan, jnp.nan]]),
    )

    assert jnp.array_equal(state.position[1], jnp.zeros((2,)))
    assert jnp.array_equal(state.velocity[1], jnp.zeros((2,)))
    assert jnp.allclose(markers.active_values(state.position), jnp.asarray([[0.2, 0.3]]))
