#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax.applications.contact._friction import (
    LaggedCoulombFrictionPlan,
    PreparedLaggedCoulombFriction,
)
from phydrax.applications.contact._potential import (
    ContactPotentialEvaluation,
    ConvergentContactPotentialPlan,
    PreparedConvergentContactPotential,
)
from phydrax.discretization.contact._participant import (
    ContactParticipantScene,
    FunctionContactParticipant,
    LinearContactParticipant,
)
from phydrax.discretization.contact._precision import ContactPrecisionPolicy
from phydrax.discretization.contact._search import (
    ContactCandidateEpoch,
    DenseContactSearchPlan,
)
from phydrax.discretization.contact._surface import (
    CollisionSurfacePlan,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from phydrax.linalg import ArraySpace


def _curved_positions(state: Array) -> Array:
    reference = jnp.asarray(((-0.3, 0.06), (0.3, 0.06)), dtype=state.dtype)
    translation = jnp.stack((state[0], state[1] + state[2] * state[2]))
    return reference + translation[None, :]


def _case(
    dtype: DTypeLike = np.float64,
) -> tuple[
    ContactParticipantScene,
    tuple[Array, Array],
    PreparedConvergentContactPotential,
    ContactCandidateEpoch,
    PreparedLaggedCoulombFriction,
]:
    curved_space = ArraySpace((3,), dtype=dtype)
    wall_space = ArraySpace((2, 2), dtype=dtype)
    curved_plan = CollisionSurfacePlan(
        jnp.asarray((0, 1), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        edge_ids=jnp.asarray((100,), dtype=jnp.int64),
        participant_ids=0,
        body_ids=0,
    )
    wall_plan = CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        edge_ids=jnp.asarray((110,), dtype=jnp.int64),
        participant_ids=1,
        body_ids=1,
    )
    curved = FunctionContactParticipant(
        curved_plan,
        curved_space,
        _curved_positions,
        tangent_space=curved_space,
        participant_id="curved-contact-test",
    )
    wall = LinearContactParticipant(
        PreparedCollisionSurface(
            wall_plan,
            jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=dtype),
            selection_collision_operator(
                wall_space, jnp.asarray((0, 1), dtype=jnp.int32)
            ),
            precision=ContactPrecisionPolicy(geometry_dtype=dtype),
        )
    )
    scene = ContactParticipantScene((curved, wall))
    states = (
        jnp.asarray((0.03, -0.005, 0.025), dtype=dtype),
        jnp.zeros((2, 2), dtype=dtype),
    )
    rest = scene.positions((curved_space.zeros(), wall_space.zeros()))
    contact = ConvergentContactPotentialPlan(0.1, jnp.asarray(20.0, dtype=dtype)).prepare(
        scene, rest_positions=rest
    )
    epoch = DenseContactSearchPlan(
        edge_vertex_capacity=8,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    ).build(scene, scene.positions(states))
    friction = LaggedCoulombFrictionPlan(jnp.asarray(0.25, dtype=dtype), 0.01).prepare(
        scene, contact
    )
    return scene, states, contact, epoch, friction


def test_participant_reference_geometry_and_states_are_required() -> None:
    scene, states, contact, epoch, friction = _case()
    with pytest.raises(ValueError, match="explicit rest_positions"):
        ConvergentContactPotentialPlan(0.1, 20.0).prepare(scene)
    positions = scene.positions(states)
    with pytest.raises(ValueError, match="participant_states"):
        contact.evaluate(positions, epoch)
    frame = friction.build_state(positions, epoch)
    with pytest.raises(ValueError, match="participant_states"):
        friction.evaluate(jnp.zeros_like(positions), frame)


def _evaluate_contact(
    contact: PreparedConvergentContactPotential,
    scene: ContactParticipantScene,
    states: tuple[Array, Array],
    epoch: ContactCandidateEpoch,
) -> ContactPotentialEvaluation:
    return contact.evaluate(scene.positions(states), epoch, participant_states=states)


@pytest.mark.strict_jax
@pytest.mark.parametrize("dtype", (np.float32, np.float64), ids=("float32", "float64"))
def test_heterogeneous_contact_transpose_work_and_action_reaction(
    dtype: DTypeLike,
) -> None:
    scene, states, contact, epoch, _ = _case(dtype)
    evaluation = eqx.filter_jit(_evaluate_contact)(contact, scene, states, epoch)
    assert bool(evaluation.successful)
    assert isinstance(evaluation.state_force, tuple)
    force_a, force_b = evaluation.state_force
    surface = np.asarray(evaluation.surface_force)
    q = np.asarray(states[0])
    # Independent derivative of x=(X_x+q_0, X_y+q_1+q_2^2).
    jacobian = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 2.0 * q[2])), dtype=dtype)
    direction_a = np.asarray((0.2, -0.13, 0.4), dtype=dtype)
    direction_b = np.asarray(((0.07, 0.11), (-0.05, -0.08)), dtype=dtype)
    sample_direction = np.concatenate(
        (np.broadcast_to(jacobian @ direction_a, (2, 2)), direction_b), axis=0
    )
    tolerance = 2.0e-5 if np.dtype(dtype).itemsize == 4 else 2.0e-11
    np.testing.assert_allclose(
        force_a, jacobian.T @ surface[:2].sum(axis=0), rtol=tolerance
    )
    np.testing.assert_allclose(force_b, surface[2:], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        np.sum(surface * sample_direction),
        np.dot(np.asarray(force_a), direction_a)
        + np.sum(np.asarray(force_b) * direction_b),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(surface.sum(axis=0), 0.0, atol=tolerance)
    np.testing.assert_allclose(evaluation.moment_residual, 0.0, atol=tolerance)
    assert evaluation.surface_force.dtype == jnp.dtype(dtype)


def test_full_contact_jvp_contains_nonzero_geometry_curvature() -> None:
    scene, states, contact, epoch, _ = _case()
    directions = (
        jnp.asarray((0.02, -0.03, 0.4), dtype=jnp.float64),
        jnp.asarray(((0.01, 0.015), (-0.01, 0.025)), dtype=jnp.float64),
    )

    def effort(q: tuple[Array, Array]) -> tuple[Array, Array]:
        return contact.evaluate(
            scene.positions(q), epoch, participant_states=q
        ).state_force

    derivative = jax.jvp(effort, (states,), (directions,))[1]
    epsilon = 1.0e-6
    plus = (
        states[0] + epsilon * directions[0],
        states[1] + epsilon * directions[1],
    )
    minus = (
        states[0] - epsilon * directions[0],
        states[1] - epsilon * directions[1],
    )
    fd = tuple(
        (a - b) / (2.0 * epsilon)
        for a, b in zip(effort(plus), effort(minus), strict=True)
    )
    np.testing.assert_allclose(derivative[0], fd[0], rtol=2.0e-7, atol=2.0e-8)
    np.testing.assert_allclose(derivative[1], fd[1], rtol=2.0e-7, atol=2.0e-8)
    evaluation = contact.evaluate(
        scene.positions(states), epoch, participant_states=states
    )
    sample_direction = scene.velocities(states, directions)
    surface_tangent = -contact.hessian_action(
        scene.positions(states), sample_direction, epoch
    )
    frozen_transpose = scene.effort_pullback(states, surface_tangent)
    # (DJ)^T f has only the nonlinear q_2 entry for this independently known map.
    curvature = (
        2.0 * float(directions[0][2]) * np.asarray(evaluation.surface_force)[:2, 1].sum()
    )
    assert abs(curvature) > 1.0e-4
    expected = np.asarray((0.0, 0.0, curvature), dtype=np.float64)
    np.testing.assert_allclose(
        derivative[0] - frozen_transpose[0], expected, rtol=2.0e-10, atol=2.0e-10
    )


def test_friction_uses_current_transpose_and_preserves_frozen_frame() -> None:
    scene, states, _, epoch, friction = _case()
    frame = friction.build_state(scene.positions(states), epoch, state_version=3)
    rates = (
        jnp.asarray((0.2, -0.04, 0.3), dtype=jnp.float64),
        jnp.asarray(((0.01, 0.02), (0.03, -0.01)), dtype=jnp.float64),
    )
    velocity = scene.velocities(states, rates)
    evaluation = friction.evaluate(velocity, frame, participant_states=states)
    assert bool(evaluation.successful)
    assert evaluation.dissipation_rate > 0.0
    force_a, force_b = evaluation.state_force
    surface = np.asarray(evaluation.surface_force)
    jacobian = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 2.0 * float(states[0][2]))))
    np.testing.assert_allclose(
        force_a, jacobian.T @ surface[:2].sum(axis=0), atol=1.0e-12
    )
    np.testing.assert_allclose(force_b, surface[2:], atol=1.0e-12)
    work = np.dot(np.asarray(force_a), np.asarray(rates[0])) + np.sum(
        np.asarray(force_b) * np.asarray(rates[1])
    )
    np.testing.assert_allclose(work, -evaluation.dissipation_rate, atol=1.0e-12)
    np.testing.assert_allclose(surface.sum(axis=0), 0.0, atol=1.0e-12)
    refreshed = friction.build_state(scene.positions(states), epoch, state_version=4)
    np.testing.assert_array_equal(frame.route_keys, refreshed.route_keys)
    np.testing.assert_allclose(friction.lag_residual(frame, refreshed), 0.0, atol=0.0)
    assert int(frame.state_version) == 3


def test_friction_lag_detects_changed_closest_coefficients() -> None:
    scene, states, _, epoch, friction = _case()
    frame = friction.build_state(scene.positions(states), epoch)
    shifted = (states[0].at[0].add(0.02), states[1])
    refreshed = friction.build_state(scene.positions(shifted), epoch)
    np.testing.assert_allclose(frame.normal_force, refreshed.normal_force, atol=1.0e-11)
    np.testing.assert_allclose(frame.tangent_basis, refreshed.tangent_basis, atol=1.0e-11)
    assert float(friction.lag_residual(frame, refreshed)) > 0.0
