from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan


def _rod(dimension: int):
    if dimension == 2:
        positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
        frames = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
        inertias = jnp.asarray((0.2, 0.3))
        stretch = jnp.broadcast_to(jnp.diag(jnp.asarray((10.0, 4.0))), (2, 2, 2))
        bend = jnp.asarray((((3.0,),),))
    else:
        positions = jnp.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)))
        frames = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
        inertias = jnp.broadcast_to(jnp.diag(jnp.asarray((0.2, 0.3, 0.4))), (2, 3, 3))
        stretch = jnp.broadcast_to(jnp.diag(jnp.asarray((10.0, 4.0, 20.0))), (2, 3, 3))
        bend = jnp.asarray((((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 5.0)),))
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            positions,
            frames,
            jnp.ones((3,), dtype=positions.dtype),
            inertias,
            stretch,
            bend,
        )
    )


@pytest.mark.parametrize(
    ("dimension", "orientation_shape", "angular_shape", "configuration_sizes"),
    (
        (2, (2,), (2,), ((3, 2), (2,))),
        (3, (2, 4), (2, 3), ((3, 3), (2, 4))),
    ),
)
def test_native_configuration_velocity_and_effort_contracts_are_exact(
    dimension, orientation_shape, angular_shape, configuration_sizes
):
    rod = _rod(dimension)
    state = rod.initialize_state()
    configuration = rod.configuration_from_state(state)
    velocity = rod.velocity_from_state(state)

    assert (
        tuple(leaf.shape for leaf in rod.configuration_schema.leaves)
        == configuration_sizes
    )
    assert configuration[0].shape == (3, dimension)
    assert configuration[1].shape == orientation_shape
    assert velocity[0].shape == (3, dimension)
    assert velocity[1].shape == angular_shape
    assert tuple(
        spec.shape for spec in jax.tree.leaves(rod.velocity_space.structure())
    ) == (
        (3, dimension),
        angular_shape,
    )
    assert tuple(
        spec.shape for spec in jax.tree.leaves(rod.effort_space.structure())
    ) == (
        (3, dimension),
        angular_shape,
    )

    rebuilt = rod.state_from_configuration(configuration)
    replaced = rod.state_with_velocity(rebuilt, velocity)
    assert jnp.array_equal(replaced.positions, state.positions)
    assert jnp.array_equal(replaced.orientations, state.orientations)
    assert jnp.array_equal(replaced.velocities, state.velocities)
    assert jnp.array_equal(replaced.angular_velocities, state.angular_velocities)


def test_spatial_point_uses_scalar_first_quaternions_and_body_angular_velocity():
    rod = _rod(3)
    state = rod.initialize_state()
    assert jnp.array_equal(
        rod.configuration_from_state(state)[1],
        jnp.asarray(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
    )
    assert rod.velocity_space.names == (
        "world_node_linear_velocity",
        "material_segment_angular_velocity",
    )


def test_effort_pairing_is_direct_force_velocity_plus_material_moment_power():
    rod = _rod(3)
    linear_velocity = jnp.asarray(((0.2, -0.1, 0.4), (-0.3, 0.5, 0.7), (0.6, -0.2, -0.8)))
    body_angular_velocity = jnp.asarray(((0.3, -0.4, 0.1), (-0.5, 0.2, 0.6)))
    forces = jnp.asarray(((1.0, 2.0, -1.0), (0.5, -0.7, 0.2), (-0.1, 0.8, 1.2)))
    material_moments = jnp.asarray(((0.4, -0.2, 0.7), (-0.3, 0.6, -0.5)))
    velocity = rod.velocity_space.validate((linear_velocity, body_angular_velocity))
    effort = rod.effort_from_load(forces, material_moments)

    paired = rod.effort_space.pair(effort, velocity)
    direct = jnp.sum(forces * linear_velocity) + jnp.sum(
        material_moments * body_angular_velocity
    )

    assert paired == pytest.approx(direct)
    recovered_forces, recovered_moments = rod.load_from_effort(effort)
    assert jnp.array_equal(recovered_forces, forces)
    assert jnp.array_equal(recovered_moments, material_moments)


def test_native_spaces_reject_quaternion_storage_as_a_spatial_moment():
    rod = _rod(3)
    with pytest.raises(ValueError, match="shape"):
        rod.effort_from_load(jnp.zeros((3, 3)), jnp.zeros((2, 4)))
    with pytest.raises(ValueError, match="intrinsic shape"):
        rod.state_from_configuration((jnp.zeros((3, 3)), jnp.zeros((2, 3))))
