#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization.particle._rigid_contact import RigidContactGeometry
from phydrax.discretization.particle._rigid_hard_contact import (
    HardContactRoutePlan,
    project_friction_ball,
    project_isotropic_coulomb_impulse,
)
from phydrax.discretization.particle._rigid_unilateral import (
    JointLimitPlan,
    JointLimitState,
)


def _prepared_bodies(count=1, *, dimension=2, fixed_mask=None):
    ids = jnp.arange(100, 100 + count, dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        ids,
        jnp.ones((count,)),
        ambient_dimension=dimension,
    ).prepare()
    inertia = (
        jnp.ones((count,))
        if dimension == 2
        else jnp.broadcast_to(jnp.eye(3), (count, 3, 3))
    )
    fixed = (
        jnp.zeros((count,), dtype=bool)
        if fixed_mask is None
        else jnp.asarray(fixed_mask, dtype=bool)
    )
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((count,), dtype=jnp.int32),
        inertia,
        fixed_mask=fixed,
    ).prepare(particles)
    return ids, bodies


def _kinematics(bodies, velocity, *, angular_velocity=None):
    velocity = jnp.asarray(velocity, dtype=bodies.particles.safe_masses.dtype)
    position = jnp.zeros_like(velocity)
    if bodies.ambient_dimension == 2:
        orientation = jnp.zeros((bodies.capacity, 1), dtype=velocity.dtype)
        angular = (
            jnp.zeros((bodies.capacity, 1), dtype=velocity.dtype)
            if angular_velocity is None
            else jnp.asarray(angular_velocity, dtype=velocity.dtype)
        )
    else:
        orientation = jnp.broadcast_to(
            jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=velocity.dtype),
            (bodies.capacity, 4),
        )
        angular = (
            jnp.zeros((bodies.capacity, 3), dtype=velocity.dtype)
            if angular_velocity is None
            else jnp.asarray(angular_velocity, dtype=velocity.dtype)
        )
    return bodies.kinematics(position, velocity, orientation, angular)


def _geometry(
    normal,
    relative_velocity,
    *,
    gap=0.0,
    left_arm=None,
    right_arm=None,
    key=17,
    valid=True,
    successful=True,
):
    normal = jnp.asarray(normal, dtype=float).reshape((1, -1))
    relative = jnp.asarray(relative_velocity, dtype=normal.dtype).reshape((1, -1))
    dimension = normal.shape[-1]
    left_arm = (
        jnp.zeros_like(normal)
        if left_arm is None
        else jnp.asarray(left_arm, dtype=normal.dtype).reshape((1, dimension))
    )
    right_arm = (
        jnp.zeros_like(normal)
        if right_arm is None
        else jnp.asarray(right_arm, dtype=normal.dtype).reshape((1, dimension))
    )
    normal_velocity = jnp.sum(relative * normal, axis=-1)
    tangent = relative - normal_velocity[:, None] * normal
    gap_ = jnp.asarray([gap], dtype=normal.dtype)
    valid_ = jnp.asarray([valid])
    angular_dimension = 1 if dimension == 2 else 3
    angular = jnp.zeros((1, angular_dimension), dtype=normal.dtype)
    zero_i = jnp.zeros((1,), dtype=jnp.int32)
    return RigidContactGeometry(
        normal,
        gap_,
        jnp.maximum(-gap_, 0.0),
        jnp.ones((1,), dtype=normal.dtype),
        jnp.zeros_like(normal),
        left_arm,
        right_arm,
        left_arm,
        right_arm,
        relative,
        normal_velocity,
        tangent,
        angular,
        angular,
        jnp.asarray([key], dtype=jnp.int32),
        zero_i,
        zero_i,
        valid_,
        zero_i,
        jnp.asarray([abs(gap)], dtype=normal.dtype),
        jnp.asarray(successful),
        "test:rigid-contact",
    )


def _prepared_plane_contact(
    *,
    dimension=2,
    friction=0.0,
    restitution=0.0,
):
    _, bodies = _prepared_bodies(1, dimension=dimension)
    plan = HardContactRoutePlan(
        jnp.asarray([0]),
        jnp.asarray([-1]),
        jnp.asarray([17]),
        friction_coefficient=friction,
        restitution_coefficient=restitution,
        position_stabilization=0.0,
    )
    return bodies, plan.prepare(bodies)


def test_joint_limit_free_active_and_release():
    ids, bodies = _prepared_bodies(2, dimension=3, fixed_mask=[True, False])
    reference = _kinematics(bodies, jnp.zeros((2, 3)))
    hinge = phx.discretization.HingeJointSetPlan(
        jnp.asarray([501]),
        ids[:1],
        ids[1:],
        jnp.asarray([[0.5, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 1.0]]),
    )
    graph = phx.discretization.RigidJointGraphPlan(hinge=hinge).prepare(bodies, reference)
    prepared = JointLimitPlan(
        jnp.asarray([501]),
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        capacity=2,
    ).prepare(graph)

    free = prepared.evaluate(
        prepared.initial_state(jnp.asarray([0.0, 0.0])), reference, 0.1
    )
    assert free.successful
    assert not jnp.any(free.evaluation.lower_active)
    assert not jnp.any(free.evaluation.upper_active)
    assert jnp.allclose(free.evaluation.relative_speed_after, 0.0)

    closing = _kinematics(
        bodies,
        jnp.zeros((2, 3)),
        angular_velocity=jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]]),
    )
    lower = prepared.evaluate(
        prepared.initial_state(jnp.asarray([-1.0, 0.0])), closing, 0.1
    )
    assert lower.successful
    assert lower.evaluation.lower_active[0]
    assert lower.evaluation.lower_impulse[0] > 0.0
    assert lower.evaluation.relative_speed_after[0] >= 0.0
    assert lower.evaluation.certificate.velocity_primal_violation < 1.0e-8

    opening = _kinematics(
        bodies,
        jnp.zeros((2, 3)),
        angular_velocity=jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
    )
    upper = prepared.evaluate(
        prepared.initial_state(jnp.asarray([1.0, 0.0])), opening, 0.1
    )
    assert upper.successful
    assert upper.evaluation.upper_active[0]
    assert upper.evaluation.upper_impulse[0] > 0.0
    assert upper.evaluation.relative_speed_after[0] <= 0.0

    releasing_state = JointLimitState(
        jnp.asarray([-1.0, 0.0]),
        jnp.asarray([1.0, 0.0]),
        jnp.zeros((2,)),
        jnp.asarray([True, False]),
        jnp.asarray([False, False]),
        jnp.asarray(3, dtype=jnp.int32),
    )
    released = prepared.evaluate(releasing_state, opening, 0.1)
    assert released.successful
    assert released.evaluation.released_lower[0]
    assert released.evaluation.lower_impulse[0] == 0.0


@pytest.mark.parametrize("restitution", [0.0, 0.5, 1.0])
def test_sphere_plane_velocity_restitution_and_energy(restitution):
    bodies, prepared = _prepared_plane_contact(restitution=restitution)
    kinematics = _kinematics(bodies, [[0.0, -2.0]])
    geometry = _geometry(
        [0.0, 1.0],
        [0.0, -2.0],
        left_arm=[0.0, -1.0],
    )
    result = prepared.evaluate(prepared.initial_state(), kinematics, geometry, 0.01)
    assert result.successful
    assert result.evaluation.impacting[0]
    assert jnp.allclose(
        result.evaluation.normal_velocity_after[0],
        2.0 * restitution,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert result.evaluation.certificate.position_certified
    assert result.evaluation.normal_impulse[0] > 0.0
    assert result.evaluation.certificate.velocity_certified
    assert result.evaluation.energy.noncreating
    assert result.evaluation.energy.kinetic_after <= (
        result.evaluation.energy.kinetic_before + 1.0e-8
    )


def test_sphere_sphere_equal_mass_normal_impulse():
    _, bodies = _prepared_bodies(2, dimension=3)
    prepared = HardContactRoutePlan(
        jnp.asarray([0]),
        jnp.asarray([1]),
        jnp.asarray([17]),
        position_stabilization=0.0,
    ).prepare(bodies)
    kinematics = _kinematics(bodies, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    geometry = _geometry(
        [1.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        left_arm=[-1.0, 0.0, 0.0],
        right_arm=[1.0, 0.0, 0.0],
    )
    result = prepared.evaluate(prepared.initial_state(), kinematics, geometry, 0.01)
    assert result.successful
    assert jnp.allclose(result.evaluation.normal_impulse, 1.0, rtol=1.0e-6)
    assert jnp.allclose(result.accepted_kinematics.velocity, 0.0, atol=1.0e-7)


def test_resting_contact_does_not_reapply_restitution():
    bodies, prepared = _prepared_plane_contact(restitution=1.0)
    impact_kinematics = _kinematics(bodies, [[0.0, -1.0]])
    impact_geometry = _geometry([0.0, 1.0], [-0.0, -1.0], left_arm=[0.0, -1.0])
    impact = prepared.evaluate(
        prepared.initial_state(), impact_kinematics, impact_geometry, 0.01
    )
    resting_kinematics = _kinematics(bodies, [[0.0, 0.0]])
    resting_geometry = _geometry([0.0, 1.0], [0.0, 0.0], left_arm=[0.0, -1.0])
    resting = prepared.evaluate(
        impact.accepted_state, resting_kinematics, resting_geometry, 0.01
    )
    assert resting.successful
    assert resting.evaluation.restitution.resting[0]
    assert not resting.evaluation.impacting[0]
    assert resting.evaluation.restitution.target_velocity[0] == 0.0
    assert jnp.allclose(resting.evaluation.normal_velocity_after, 0.0, atol=1.0e-8)


def test_zero_friction_reduces_to_normal_contact():
    bodies, prepared = _prepared_plane_contact(friction=0.0)
    kinematics = _kinematics(bodies, [[3.0, -1.0]])
    geometry = _geometry([0.0, 1.0], [3.0, -1.0], left_arm=[0.0, -1.0])
    result = prepared.evaluate(prepared.initial_state(), kinematics, geometry, 0.01)
    assert result.successful
    assert jnp.all(result.evaluation.tangent_impulse == 0.0)
    assert jnp.allclose(result.accepted_kinematics.velocity[0, 0], 3.0)
    assert result.evaluation.certificate.cone_violation == 0.0


@pytest.mark.parametrize(
    ("friction", "expect_stick"),
    [(1.0, True), (0.1, False)],
)
def test_incline_contact_stick_and_slip(friction, expect_stick):
    bodies, prepared = _prepared_plane_contact(friction=friction)
    normal = jnp.asarray([-0.5, jnp.sqrt(0.75)])
    tangent = jnp.asarray([jnp.sqrt(0.75), 0.5])
    relative_velocity = tangent - normal
    kinematics = _kinematics(bodies, relative_velocity[None, :])
    geometry = _geometry(normal, relative_velocity, left_arm=-normal)
    result = prepared.evaluate(prepared.initial_state(), kinematics, geometry, 0.01)
    assert result.successful
    assert bool(result.evaluation.sticking[0]) is expect_stick
    assert bool(result.evaluation.sliding[0]) is (not expect_stick)
    tangent_norm = jnp.linalg.norm(result.evaluation.tangent_impulse[0])
    cone_bound = friction * result.evaluation.normal_impulse[0]
    assert tangent_norm <= cone_bound + 1.0e-8
    assert result.evaluation.energy.friction_dissipation >= -1.0e-9


def test_exact_cone_projection_and_spatial_basis_invariance():
    planar = project_isotropic_coulomb_impulse(
        jnp.asarray([1.0]), jnp.asarray([[2.0]]), jnp.asarray([0.5])
    )
    assert planar.successful[0]
    assert planar.boundary[0]
    assert jnp.allclose(jnp.abs(planar.tangent_impulse[0, 0]), 0.8)
    assert jnp.allclose(planar.normal_impulse[0], 1.6)
    assert jnp.all(jnp.isfinite(planar.generalized_jacobian))

    tangent = jnp.asarray([[0.8, -0.6]])
    rotation = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    rotated = tangent @ rotation.T
    first = project_isotropic_coulomb_impulse(0.5 * jnp.ones((1,)), tangent, 0.4)
    second = project_isotropic_coulomb_impulse(0.5 * jnp.ones((1,)), rotated, 0.4)
    assert first.successful[0] & second.successful[0]
    assert jnp.allclose(second.tangent_impulse, first.tangent_impulse @ rotation.T)
    assert jnp.allclose(
        jnp.linalg.norm(first.tangent_impulse, axis=-1), first.cone_radius
    )

    ball = project_friction_ball(
        jnp.asarray([2.0]), jnp.asarray([[3.0, 4.0]]), jnp.asarray([0.25])
    )
    assert ball.successful[0]
    assert ball.sliding[0]
    assert jnp.allclose(jnp.linalg.norm(ball.tangent_impulse, axis=-1), 0.5)
    assert jnp.all(jnp.isfinite(ball.derivative_tangent))


def test_capacity_validation_and_geometry_failure_roll_back_atomically():
    _, bodies = _prepared_bodies(1, dimension=2)
    with pytest.raises(ValueError, match="capacity"):
        HardContactRoutePlan(
            jnp.asarray([2]), jnp.asarray([-1]), jnp.asarray([17])
        ).prepare(bodies)

    prepared = HardContactRoutePlan(
        jnp.asarray([0]), jnp.asarray([-1]), jnp.asarray([17])
    ).prepare(bodies)
    state = prepared.initial_state()
    kinematics = _kinematics(bodies, [[0.0, -1.0]])
    failed_geometry = _geometry(
        [0.0, 1.0],
        [0.0, -1.0],
        left_arm=[0.0, -1.0],
        successful=False,
    )
    failed = prepared.evaluate(state, kinematics, failed_geometry, 0.01)
    assert not failed.successful
    assert jax.tree.all(
        jax.tree.map(
            lambda accepted, current: jnp.all(accepted == current),
            failed.accepted_state,
            state,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda accepted, current: jnp.all(accepted == current),
            failed.accepted_kinematics,
            kinematics,
        )
    )

    wrong_route = _geometry(
        [0.0, 1.0],
        [0.0, -1.0],
        left_arm=[0.0, -1.0],
        key=18,
    )
    route_failure = prepared.evaluate(state, kinematics, wrong_route, 0.01)
    assert not route_failure.successful
    assert not route_failure.evaluation.routes_match
    assert jax.tree.all(
        jax.tree.map(
            lambda accepted, current: jnp.all(accepted == current),
            route_failure.accepted_state,
            state,
        )
    )


def test_hard_contact_is_jittable_with_static_capacity():
    bodies, prepared = _prepared_plane_contact(friction=0.4, restitution=0.5)
    state = prepared.initial_state()
    kinematics = _kinematics(bodies, [[0.25, -1.0]])
    geometry = _geometry([0.0, 1.0], [0.25, -1.0], left_arm=[0.0, -1.0])
    result = eqx.filter_jit(prepared.evaluate)(state, kinematics, geometry, 0.01)
    assert result.successful
    assert result.accepted_state.normal_impulse.shape == (1,)
    assert result.accepted_state.tangent_impulse.shape == (1, 2)
    assert result.evaluation.certificate.velocity_certified
