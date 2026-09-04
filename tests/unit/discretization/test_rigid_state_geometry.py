#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization.particle._core import ParticleSetPlan
from phydrax.discretization.particle._rigid_body import (
    RigidBodyKinematics,
    RigidBodySetPlan,
    RigidBodyStateGeometry,
)


def _prepared_state(dimension):
    capacity = 2
    particles = ParticleSetPlan(
        jnp.arange(capacity, dtype=jnp.int64),
        jnp.ones((capacity,)),
        ambient_dimension=dimension,
    ).prepare()
    inertia = (
        jnp.ones((capacity,))
        if dimension == 2
        else jnp.broadcast_to(jnp.eye(3), (capacity, 3, 3))
    )
    bodies = RigidBodySetPlan(
        jnp.zeros((capacity,), dtype=jnp.int32),
        inertia,
    ).prepare(particles)
    position = jnp.arange(capacity * dimension, dtype=jnp.float64).reshape(
        (capacity, dimension)
    )
    orientation = (
        jnp.asarray([[0.2], [-0.4]])
        if dimension == 2
        else jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0],
            ]
        )
    )
    angular_dimension = 1 if dimension == 2 else 3
    state = bodies.kinematics(
        position,
        -0.1 * position,
        orientation,
        jnp.zeros((capacity, angular_dimension), dtype=position.dtype),
    )
    return bodies, state


def _assert_tree_allclose(left, right, *, atol=1.0e-10):
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol)


def test_rigid_quaternion_geometry_uses_physical_angular_four_spaces():
    bodies, state = _prepared_state(3)
    geometry = RigidBodyStateGeometry(bodies)
    local = geometry.local_space.unflatten(
        jnp.linspace(-0.12, 0.12, geometry.local_space.size, dtype=state.position.dtype)
    )
    direction = geometry.local_space.unflatten(
        jnp.linspace(0.15, -0.1, geometry.local_space.size, dtype=state.position.dtype)
    )
    cotangent = geometry.cotangent_space.unflatten(
        jnp.linspace(
            -0.2, 0.25, geometry.cotangent_space.size, dtype=state.position.dtype
        )
    )

    point_storage_size = sum(leaf.size for leaf in jax.tree.leaves(state))
    assert point_storage_size == 26
    assert geometry.local_space.size == 24
    assert geometry.tangent_space.size == 24
    assert geometry.local_cotangent_space.size == 24
    assert geometry.cotangent_space.size == 24
    assert local.orientation.shape == (2, 3)
    assert cotangent.orientation.shape == (2, 3)

    point = geometry.retract(state, local)
    recovered = geometry.inverse_retract(state, point)
    assert isinstance(point, RigidBodyKinematics)
    assert point.orientation.shape == (2, 4)
    assert jnp.linalg.norm(point.orientation, axis=-1) == pytest.approx(jnp.ones((2,)))
    _assert_tree_allclose(recovered, local)

    pushed = geometry.retraction_jvp(state, local, direction)
    inverse_pushed = geometry.retraction_inverse_jvp(state, point, pushed)
    pulled = geometry.retraction_vjp(state, local, cotangent)
    assert pushed.orientation.shape == (2, 3)
    _assert_tree_allclose(inverse_pushed, direction)
    assert geometry.cotangent_space.pair(cotangent, pushed) == pytest.approx(
        geometry.local_cotangent_space.pair(pulled, direction),
        abs=1.0e-10,
    )

    transported = geometry.transport_tangent(state, point, pushed)
    recovered_transport = geometry.transport_tangent(point, state, transported)
    transport_pullback = geometry.transport_cotangent_pullback(
        state,
        point,
        cotangent,
    )
    _assert_tree_allclose(recovered_transport, pushed)
    assert geometry.cotangent_space.pair(cotangent, transported) == pytest.approx(
        geometry.cotangent_space.pair(transport_pullback, pushed),
        abs=1.0e-10,
    )
    assert geometry.cut_locus_margin(state, point) > 0.0
    assert geometry.supports_exact_inverse
    assert geometry.supports_exact_differential
    assert geometry.supports_transport
    assert geometry.supports_isometric_transport


def test_planar_rigid_geometry_keeps_equal_sized_angle_roles_exact():
    bodies, state = _prepared_state(2)
    geometry = RigidBodyStateGeometry(bodies)
    local = geometry.local_space.unflatten(
        jnp.linspace(-0.1, 0.1, geometry.local_space.size, dtype=state.position.dtype)
    )
    direction = geometry.local_space.unflatten(
        jnp.linspace(0.2, -0.15, geometry.local_space.size, dtype=state.position.dtype)
    )

    assert geometry.local_space.size == geometry.tangent_space.size == 12
    point = geometry.retract(state, local)
    _assert_tree_allclose(geometry.inverse_retract(state, point), local)
    pushed = geometry.retraction_jvp(state, local, direction)
    _assert_tree_allclose(
        geometry.retraction_inverse_jvp(state, point, pushed),
        direction,
    )
