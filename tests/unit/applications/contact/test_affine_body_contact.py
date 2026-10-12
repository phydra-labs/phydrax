# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.contact._affine_body import (
    AffineBodyHistory,
    AffineBodyParticipant,
    AffineBodyPlan,
    prepare_affine_body_participant,
)
from phydrax.applications.contact._implicit_participant import (
    ContactParticipantEvaluation,
    ContactParticipantStepLimit,
    linearize_contact_kinematics,
    make_implicit_contact_participant,
    participant_residual,
)
from phydrax.discretization.contact._guarantee import (
    ContactGuaranteeEvidence,
    ContactGuaranteeLevel,
)
from phydrax.discretization.contact._surface import CollisionSurfacePlan
from phydrax.linalg import ArraySpace
from phydrax.nonlinear import (
    implicit_root_result,
    NewtonKrylov,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _body(
    *, gravity: tuple[float, float, float] = (0.0, 0.0, 0.0), modulus: float = 30.0
) -> AffineBodyParticipant:
    controls = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), np.eye(3, dtype=np.float64)), axis=0
    )[None, :, :]
    topology = CollisionSurfacePlan(
        np.arange(4, dtype=np.int64),
        ambient_dimension=3,
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        body_ids=0,
        participant_ids=0,
    )
    # Degree-two tetrahedral quadrature, exact for every product N_a N_c.
    high = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    low = (5.0 - np.sqrt(5.0)) / 20.0
    shapes = np.full((4, 4), low, dtype=np.float64)
    np.fill_diagonal(shapes, high)
    plan = AffineBodyPlan(
        topology,
        controls,
        controls[0],
        np.zeros(4, dtype=np.int32),
        shapes[None, :, :],
        np.full((1, 4), 1.0 / 24.0, dtype=np.float64),
        np.array([2.0], dtype=np.float64),
        np.array([modulus], dtype=np.float64),
        gravity=gravity,
    )
    return prepare_affine_body_participant(plan)


def _history() -> AffineBodyHistory:
    return AffineBodyHistory(
        jnp.zeros((1, 4, 3), dtype=jnp.float64), jnp.zeros((1, 4, 3), dtype=jnp.float64)
    )


@pytest.mark.strict_jax
def test_twelve_controls_consistent_mass_and_surface_transpose() -> None:
    participant = _body()
    assert participant.dof_space.shape == (1, 4, 3)
    # Analytic simplex monomial moments: integral N_a²=V/10,
    # integral N_a N_c=V/20, a!=c; mass=rho*V=1/3.
    expected = (np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64)) / 60.0
    np.testing.assert_allclose(participant.plan.control_mass[0], expected, atol=1e-15)
    zero = jnp.zeros((1, 4, 3), dtype=jnp.float64)
    direction = jnp.arange(12, dtype=jnp.float64).reshape((1, 4, 3)) / 12.0
    effort = jnp.arange(12, dtype=jnp.float64).reshape((4, 3)) / 7.0
    kinematics = linearize_contact_kinematics(participant.geometry, zero)
    np.testing.assert_allclose(kinematics.tangent.mv(direction), direction[0], atol=1e-15)
    evidence = participant.geometry.duality_evidence(zero, direction, effort)
    assert bool(evidence.valid)


@pytest.mark.parametrize(
    "angle", [0.0, 0.6, np.pi], ids=["identity", "proper-rotation", "half-turn"]
)
def test_proper_rotation_zero_rigidity_and_gradient(angle: float) -> None:
    participant = _body()
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    reference = np.asarray(participant.plan.reference_controls)
    state = jnp.asarray(reference @ rotation.T - reference, dtype=jnp.float64)
    evaluation = participant.evaluate(
        state,
        _history(),
        jnp.zeros_like(state),
        jnp.zeros((), dtype=state.dtype),
        None,
        None,
    )
    assert bool(evaluation.admissible)
    if evaluation.potential_energy is None:
        raise ValueError(
            "Affine rigidity evaluation must provide its declared potential."
        )
    np.testing.assert_allclose(evaluation.potential_energy, 0.0, atol=1e-28)
    np.testing.assert_allclose(evaluation.residual, 0.0, atol=1e-13)


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([1.1, 1.0, 1.0]),
        np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
    ids=["stretch", "shear"],
)
def test_rigidity_matches_independent_polynomial_energy(matrix: np.ndarray) -> None:
    participant = _body()
    reference = np.asarray(participant.plan.reference_controls)
    state = jnp.asarray(reference @ matrix.T - reference, dtype=jnp.float64)
    strain = matrix.T @ matrix - np.eye(3, dtype=np.float64)
    reference_energy = (1.0 / 6.0) * 30.0 * np.sum(strain**2) / 8.0
    evaluation = participant.evaluate(
        state,
        _history(),
        jnp.zeros_like(state),
        jnp.zeros((), dtype=state.dtype),
        None,
        None,
    )
    if evaluation.potential_energy is None:
        raise ValueError(
            "Affine rigidity evaluation must provide its declared potential."
        )
    np.testing.assert_allclose(evaluation.potential_energy, reference_energy, rtol=1e-13)
    assert reference_energy > 0.0


@pytest.mark.strict_jax
def test_polynomial_hvp_and_jvp_finite_at_identity() -> None:
    participant = _body()
    state = jnp.zeros((1, 4, 3), dtype=jnp.float64)
    direction = jnp.arange(12, dtype=jnp.float64).reshape(state.shape) / 15.0
    time = jnp.zeros((), dtype=jnp.float64)
    exact = participant.residual_jvp(
        state, direction, _history(), state, time, None, None
    )
    epsilon = jnp.asarray(1e-6, dtype=state.dtype)
    plus = participant.evaluate(
        state + epsilon * direction, _history(), state, time, None, None
    ).residual
    minus = participant.evaluate(
        state - epsilon * direction, _history(), state, time, None, None
    ).residual
    np.testing.assert_allclose(
        exact, (plus - minus) / (2.0 * epsilon), atol=1e-9, rtol=1e-8
    )
    assert bool(jnp.all(jnp.isfinite(exact)))
    np.testing.assert_allclose(
        participant.potential_hvp(state, direction, _history(), state, time, None, None),
        exact,
        atol=1e-14,
    )


def test_reflection_and_interior_orientation_crossing_refuse() -> None:
    participant = _body()
    reference = participant.plan.reference_controls
    reflection = jnp.asarray(np.diag([-1.0, 1.0, 1.0]), dtype=jnp.float64)
    reflected = reference @ reflection.T - reference
    zero = jnp.zeros_like(reference)
    evaluation = participant.evaluate(
        reflected, _history(), zero, jnp.zeros((), dtype=zero.dtype), None, None
    )
    assert not bool(evaluation.admissible)
    # Both endpoints proper, but the linear control path collapses at alpha=1/2.
    rotation = jnp.asarray(np.diag([-1.0, -1.0, 1.0]), dtype=jnp.float64)
    direction = reference @ rotation.T - reference
    limit = eqx.filter_jit(participant.step_limit)(
        zero, direction, zero, jnp.zeros((), dtype=zero.dtype), None, None
    )
    assert bool(limit.successful)
    assert bool(limit.conservative)
    assert 0.0 < float(limit.maximum_step) < 0.5
    refused = participant.step_limit(
        reflected, zero, zero, jnp.zeros((), dtype=zero.dtype), None, None
    )
    assert not bool(refused.successful)


def test_newmark_gravity_trial_history_and_static_omission() -> None:
    participant = _body(gravity=(0.0, 0.0, -2.0))
    zero = jnp.zeros((1, 4, 3), dtype=jnp.float64)
    velocity = jnp.broadcast_to(
        jnp.asarray([0.3, -0.2, 0.1], dtype=zero.dtype), zero.shape
    )
    acceleration = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, -2.0], dtype=zero.dtype), zero.shape
    )
    history = AffineBodyHistory(velocity, acceleration)
    dt = jnp.asarray(0.1, dtype=zero.dtype)
    candidate = dt * velocity + 0.5 * dt**2 * acceleration
    time = jnp.zeros((), dtype=zero.dtype)
    evaluation = participant.evaluate(candidate, history, zero, time, dt, None)
    np.testing.assert_allclose(evaluation.residual, 0.0, atol=1e-14)
    trial = evaluation.trial_history
    assert isinstance(trial, AffineBodyHistory)
    np.testing.assert_allclose(trial.velocity, velocity + dt * acceleration, atol=1e-14)
    np.testing.assert_allclose(trial.acceleration, acceleration, atol=1e-14)
    np.testing.assert_array_equal(history.velocity, velocity)
    static = participant.evaluate(candidate, history, zero, time, None, None)
    mass_rows = np.sum(np.asarray(participant.plan.control_mass), axis=-1)
    expected = (
        -mass_rows[:, :, None] * np.asarray(participant.plan.gravity)[None, None, :]
    )
    np.testing.assert_allclose(static.residual, expected, atol=1e-14)
    assert static.trial_history is history


def test_reflected_reference_and_rank_deficient_mass_refuse() -> None:
    participant = _body()
    plan = participant.plan
    reference = np.asarray(plan.reference_controls).copy()
    reference[:, [1, 2], :] = reference[:, [2, 1], :]
    with pytest.raises(ValueError, match="orientation"):
        AffineBodyPlan(
            plan.topology,
            reference,
            plan.material_coordinates,
            plan.vertex_body_ids,
            plan.quadrature_shape_values,
            plan.quadrature_weights,
            plan.density,
            plan.rigidity_modulus,
        )
    with pytest.raises(ValueError, match="mass"):
        AffineBodyPlan(
            plan.topology,
            plan.reference_controls,
            plan.material_coordinates,
            plan.vertex_body_ids,
            np.full((1, 1, 4), 0.25, dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
            plan.density,
            plan.rigidity_modulus,
        )


def _spin_error(dt: float) -> tuple[float, float]:
    participant = _body(modulus=3000.0)
    controls = np.asarray(participant.plan.reference_controls)
    center = np.full((1, 1, 3), 0.25, dtype=np.float64)
    omega = np.full(3, 0.2 / np.sqrt(3.0), dtype=np.float64)
    velocity = np.cross(np.broadcast_to(omega, controls.shape), controls - center)
    history = AffineBodyHistory(velocity, np.zeros_like(velocity))
    accepted = jnp.zeros_like(participant.plan.reference_controls)
    step = jnp.asarray(dt, dtype=jnp.float64)
    clock = jnp.zeros((), dtype=jnp.float64)

    def residual(state: Array, args: object) -> Array:
        return participant_residual(
            participant, state, history, accepted, clock, step, args
        )

    for _ in range(round(0.02 / dt)):
        predictor = (
            accepted
            + step * history.velocity
            + step**2 * (0.5 - participant.newmark.beta) * history.acceleration
        )
        root = implicit_root_result(
            NonlinearSystemProblem(residual, state_space=participant.dof_space),
            predictor,
            method=NewtonKrylov(),
            termination=NonlinearTermination(
                maximum_steps=30, absolute_residual=1e-11, relative_residual=1e-11
            ),
        )
        assert bool(root.successful)
        evaluation = participant.evaluate(
            root.state, history, accepted, clock, step, None
        )
        candidate_history = evaluation.trial_history
        assert isinstance(candidate_history, AffineBodyHistory)
        accepted = root.state
        history = candidate_history
        clock = clock + step
    shapes = np.asarray(participant.plan.quadrature_shape_values[0])
    masses = np.full(4, 1.0 / 12.0, dtype=np.float64)
    before_x = shapes @ controls[0]
    before_v = shapes @ velocity[0]
    after_x = shapes @ (controls[0] + np.asarray(accepted)[0])
    after_v = shapes @ np.asarray(history.velocity)[0]
    momentum_before = np.sum(masses[:, None] * np.cross(before_x, before_v), axis=0)
    momentum_after = np.sum(masses[:, None] * np.cross(after_x, after_v), axis=0)
    kinetic_before = 0.5 * np.sum(masses[:, None] * before_v**2)
    # Uniform tetrahedron monomial moments give principal inertia 1/30
    # about its (1,1,1) symmetry axis through the centroid (mass=1/3).
    np.testing.assert_allclose(momentum_before, omega / 30.0, atol=1e-15)
    np.testing.assert_allclose(kinetic_before, 0.5 * (1.0 / 30.0) * 0.2**2, atol=1e-15)
    kinetic_after = 0.5 * np.sum(masses[:, None] * after_v**2)
    frame = (controls[0] + np.asarray(accepted)[0])[1:] - (
        controls[0] + np.asarray(accepted)[0]
    )[:1]
    strain = frame @ frame.T - np.eye(3, dtype=np.float64)
    energy_after = kinetic_after + (3000.0 / 48.0) * np.sum(strain**2)
    return float(np.linalg.norm(momentum_after - momentum_before)), float(
        abs(energy_after - kinetic_before)
    )


def test_free_spin_momentum_and_energy_refine_against_quadrature_balance() -> None:
    coarse_momentum, coarse_energy = _spin_error(0.002)
    fine_momentum, fine_energy = _spin_error(0.001)
    assert coarse_momentum < 1e-5
    assert fine_energy < coarse_energy
    assert fine_momentum < 1e-5
    assert fine_energy < 1e-5


def test_contact_velocity_is_newmark_rate_not_secant() -> None:
    participant = _body()
    zero = jnp.zeros((1, 4, 3), dtype=jnp.float64)
    history = AffineBodyHistory(jnp.ones_like(zero), jnp.zeros_like(zero))
    step = jnp.asarray(0.1, dtype=zero.dtype)
    position = jnp.full_like(zero, 0.15)
    velocity = participant.contact_velocity(
        position, history, zero, jnp.zeros((), dtype=zero.dtype), step, None
    )
    np.testing.assert_allclose(velocity, 2.0, atol=1e-14)
    np.testing.assert_allclose(participant.contact_velocity_scale(step), 20.0, atol=1e-14)


def test_functional_participant_uses_mandatory_jvp_and_differentiates_arguments() -> None:
    def evaluate(
        state: Array,
        history: tuple[()],
        accepted: Array,
        time: Array,
        step: Array | None,
        args: Array,
        /,
    ) -> ContactParticipantEvaluation:
        del accepted, time, step
        # A provider may have its own differentiable owner; its declared state
        # action, not incidental tracing through evaluate, is authoritative.
        residual = jax.lax.stop_gradient(state) ** 2 + args
        ok = jnp.asarray(True, dtype=jnp.bool_)
        return ContactParticipantEvaluation(
            residual,
            history,
            None,
            ok,
            ok,
            ok,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(jnp.inf, dtype=state.dtype),
            jnp.int32(0),
            {},
        )

    def action(
        state: Array,
        direction: Array,
        history: tuple[()],
        accepted: Array,
        time: Array,
        step: Array | None,
        args: Array,
        /,
    ) -> Array:
        del history, accepted, time, step, args
        return 2.0 * state * direction

    def limit(
        state: Array,
        direction: Array,
        accepted: Array,
        time: Array,
        step: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantStepLimit:
        del direction, accepted, time, step, args
        ok = jnp.asarray(True, dtype=jnp.bool_)
        guarantee = ContactGuaranteeEvidence(
            ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE, backend_id="test-functional"
        )
        return ContactParticipantStepLimit(
            jnp.ones((), dtype=state.dtype),
            ok,
            ok,
            guarantee,
            jnp.int32(0),
            jnp.int32(-1),
            "test-functional",
        )

    participant = make_implicit_contact_participant(
        ArraySpace((2,), dtype=jnp.float64),
        evaluate,
        action,
        limit,
        participant_id="test-functional",
    )
    state = jnp.asarray([0.3, 0.5], dtype=jnp.float64)
    direction = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    arguments = jnp.asarray([0.1, 0.4], dtype=jnp.float64)
    argument_direction = jnp.asarray([0.7, 0.6], dtype=jnp.float64)
    clock = jnp.zeros((), dtype=jnp.float64)

    def residual(q: Array, args: Array, /) -> Array:
        return participant_residual(
            participant, q, (), jnp.zeros_like(q), clock, None, args
        )

    _, derivative = eqx.filter_jvp(
        residual, (state, arguments), (direction, argument_direction)
    )
    np.testing.assert_allclose(
        derivative, 2.0 * state * direction + argument_direction, atol=1e-14
    )


def test_monolithic_affine_wall_collision_preserves_certified_clearance_and_history() -> (
    None
):
    from phydrax.applications.contact._potential import ConvergentContactPotentialPlan
    from phydrax.applications.contact._solver import (
        ContactSolvePolicy,
        initialize_monolithic_contact_state,
        MonolithicContactPlan,
        prepare_monolithic_contact_step,
        solve_monolithic_contact_step,
    )
    from phydrax.discretization.contact._ccd import InclusionCCDPlan
    from phydrax.discretization.contact._participant import (
        ContactParticipantScene,
        LinearContactParticipant,
    )
    from phydrax.discretization.contact._search import DenseContactSearchPlan
    from phydrax.discretization.contact._surface import (
        PreparedCollisionSurface,
        static_collision_operator,
    )

    original = _body().plan
    controls = (
        0.1 * original.reference_controls
        + jnp.asarray([0.0, 0.0, 0.015], dtype=jnp.float64)[None, None, :]
    )
    topology = CollisionSurfacePlan(
        np.arange(4, dtype=np.int64),
        ambient_dimension=3,
        faces=np.asarray(original.topology.faces),
        body_ids=0,
        solver_clearance=0.001,
    )
    body = AffineBodyPlan(
        topology,
        controls,
        original.material_coordinates,
        original.vertex_body_ids,
        original.quadrature_shape_values,
        0.001 * original.quadrature_weights,
        original.density,
        jnp.asarray([3000.0], dtype=jnp.float64),
        gravity=(0.0, 0.0, 0.0),
    )
    participant = prepare_affine_body_participant(body)
    wall_topology = CollisionSurfacePlan(
        np.arange(100, 104, dtype=np.int64),
        ambient_dimension=3,
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        body_ids=1,
        static_mask=True,
    )
    wall_positions = jnp.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    wall_space = ArraySpace((4, 3), dtype=jnp.float64)
    wall = LinearContactParticipant(
        PreparedCollisionSurface(
            wall_topology,
            wall_positions,
            static_collision_operator(wall_space, 4, 3, dtype=jnp.float64),
        )
    )
    scene = ContactParticipantScene((participant.geometry, wall))
    zero = jnp.zeros_like(controls)
    fixed = jnp.zeros_like(wall_positions)
    contact = ConvergentContactPotentialPlan(
        0.03, jnp.asarray(100.0, dtype=jnp.float64)
    ).prepare(scene, rest_positions=scene.positions((zero, fixed)))
    search = DenseContactSearchPlan(
        edge_vertex_capacity=scene.edge_count * scene.vertex_count,
        edge_edge_capacity=scene.edge_count**2,
        face_vertex_capacity=scene.face_count * scene.vertex_count,
        activation_distance=0.03,
        envelope_radius=0.1,
    )
    plan = MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        search,
        InclusionCCDPlan(maximum_iterations=256),
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, fixed),
        policy=ContactSolvePolicy(),
    )
    velocity = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, -0.5], dtype=jnp.float64), zero.shape
    )
    history = AffineBodyHistory(velocity, jnp.zeros_like(zero))
    initial = initialize_monolithic_contact_state(plan, (zero,), (history,))
    result = solve_monolithic_contact_step(
        prepare_monolithic_contact_step(
            plan, initial, jnp.asarray(0.02, dtype=jnp.float64)
        )
    )
    assert bool(result.accepted)
    assert bool(result.safety.successful)
    assert bool(result.safety.conservative)
    committed = result.commit()
    positions = np.asarray(participant.geometry.positions(committed.coordinates[0]))
    # Plane z=0 is exact inside this finite wall: linear control interpolation
    # remains above clearance iff both endpoint signed heights are above it.
    assert np.min(positions[:, 2]) > 0.001
    assert (
        min(float(np.min(np.asarray(controls)[0, :, 2])), float(np.min(positions[:, 2])))
        > 0.001
    )
    frame = positions[1:] - positions[:1]
    assert np.linalg.det(frame) > 0.0
    trial = committed.histories[0]
    assert isinstance(trial, AffineBodyHistory)
    mass_rows = np.sum(np.asarray(body.control_mass), axis=-1)
    vertical_momentum = np.sum(mass_rows * np.asarray(trial.velocity)[:, :, 2])
    assert vertical_momentum > 0.0
    np.testing.assert_array_equal(history.velocity, velocity)
    np.testing.assert_array_equal(initial.coordinates[0], zero)


def test_two_affine_boxes_stack_keeps_clearance_and_proper_frames() -> None:
    from itertools import product

    from phydrax.applications.contact._potential import ConvergentContactPotentialPlan
    from phydrax.applications.contact._solver import (
        initialize_monolithic_contact_state,
        MonolithicContactPlan,
        prepare_monolithic_contact_step,
        solve_monolithic_contact_step,
    )
    from phydrax.discretization.contact._ccd import InclusionCCDPlan
    from phydrax.discretization.contact._participant import (
        ContactParticipantScene,
        LinearContactParticipant,
    )
    from phydrax.discretization.contact._search import DenseContactSearchPlan
    from phydrax.discretization.contact._surface import (
        PreparedCollisionSurface,
        static_collision_operator,
    )

    vertices = np.asarray(list(product((0.0, 1.0), repeat=3)), dtype=np.float64)
    faces = np.asarray(
        [
            [0, 3, 1],
            [0, 2, 3],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    frame = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), np.eye(3, dtype=np.float64)), axis=0
    )
    controls = np.stack(
        (
            0.1 * frame + np.array([0.0, 0.0, 0.02], dtype=np.float64),
            0.1 * frame + np.array([0.0, 0.0, 0.14], dtype=np.float64),
        )
    )
    topology = CollisionSurfacePlan(
        np.arange(16, dtype=np.int64),
        ambient_dimension=3,
        faces=np.concatenate((faces, faces + 8)),
        body_ids=np.repeat(np.arange(2, dtype=np.int32), 8),
        solver_clearance=0.001,
    )
    gauss = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
    quadrature = np.asarray(list(product(gauss, repeat=3)), dtype=np.float64)
    shapes = np.concatenate(
        (1.0 - quadrature.sum(axis=-1, keepdims=True), quadrature), axis=-1
    )
    body = AffineBodyPlan(
        topology,
        controls,
        np.tile(vertices, (2, 1)),
        np.repeat(np.arange(2, dtype=np.int32), 8),
        np.broadcast_to(shapes, (2, 8, 4)),
        np.full((2, 8), 0.001 / 8.0, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.full(2, 1e5, dtype=np.float64),
    )
    participant = prepare_affine_body_participant(body)
    wall_topology = CollisionSurfacePlan(
        np.arange(100, 104, dtype=np.int64),
        ambient_dimension=3,
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        body_ids=2,
        static_mask=True,
    )
    wall_positions = jnp.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    wall = LinearContactParticipant(
        PreparedCollisionSurface(
            wall_topology,
            wall_positions,
            static_collision_operator(
                ArraySpace((4, 3), dtype=jnp.float64), 4, 3, dtype=jnp.float64
            ),
        )
    )
    scene = ContactParticipantScene((participant.geometry, wall))
    zero = jnp.zeros_like(body.reference_controls)
    fixed = jnp.zeros_like(wall_positions)
    contact = ConvergentContactPotentialPlan(
        0.03, jnp.asarray(0.01, dtype=jnp.float64)
    ).prepare(scene, rest_positions=scene.positions((zero, fixed)))
    search = DenseContactSearchPlan(
        edge_vertex_capacity=scene.edge_count * scene.vertex_count,
        edge_edge_capacity=scene.edge_count**2,
        face_vertex_capacity=scene.face_count * scene.vertex_count,
        activation_distance=0.03,
        envelope_radius=0.05,
    )
    plan = MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        search,
        InclusionCCDPlan(maximum_iterations=256),
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, fixed),
    )
    state = initialize_monolithic_contact_state(
        plan, (zero,), (AffineBodyHistory(zero, zero),)
    )
    for _ in range(3):
        result = solve_monolithic_contact_step(
            prepare_monolithic_contact_step(
                plan, state, jnp.asarray(0.001, dtype=jnp.float64)
            )
        )
        assert bool(result.accepted)
        assert bool(result.safety.conservative)
        assert bool(result.safety.successful)
        state = result.commit()
    samples = np.asarray(participant.geometry.positions(state.coordinates[0])).reshape(
        (2, 8, 3)
    )
    assert samples[0, :, 2].min() > 0.001
    assert samples[1, :, 2].min() - samples[0, :, 2].max() > 0.002
    current = np.asarray(body.reference_controls) + np.asarray(state.coordinates[0])
    assert np.all(np.linalg.det(current[:, 1:] - current[:, :1]) > 0.0)
    # The bounded duration is 0.003s; no body may move by its own side length.
    assert (
        np.max(
            np.abs(
                samples
                - np.stack(
                    (0.1 * vertices + [0.0, 0.0, 0.02], 0.1 * vertices + [0.0, 0.0, 0.14])
                )
            )
        )
        < 0.1
    )


def test_participant_scalar_evidence_is_canonical_dynamic_array_data() -> None:
    residual = jnp.ones((2,), dtype=jnp.float64)
    history = (jnp.zeros((1,), dtype=jnp.float64),)
    evaluation = ContactParticipantEvaluation(
        residual, history, 2.0, True, True, False, False, 0.0, 0.5, 3, {"work": 2}
    )
    assert evaluation.residual is residual
    assert evaluation.trial_history is history
    assert evaluation.successful.dtype == jnp.bool_
    assert evaluation.tangent_successful.dtype == jnp.bool_
    assert evaluation.potential_consistency_checked.dtype == jnp.bool_
    assert evaluation.status_code.dtype == jnp.int32
    assert evaluation.suggested_step.dtype == residual.dtype
    assert evaluation.potential_consistency_defect.dtype == residual.dtype
    if evaluation.potential_energy is None:
        raise ValueError("The supplied potential cannot be omitted.")
    assert evaluation.potential_energy.dtype == residual.dtype
    assert all(isinstance(value, Array) for value in jax.tree.leaves(evaluation))


def test_affine_local_control_rows_bind_distinct_global_body_identity() -> None:
    original = _body().plan
    global_topology = CollisionSurfacePlan(
        np.arange(4, dtype=np.int64),
        ambient_dimension=3,
        faces=np.asarray(original.topology.faces),
        body_ids=37,
    )
    admitted = AffineBodyPlan(
        global_topology,
        original.reference_controls,
        original.material_coordinates,
        original.vertex_body_ids,
        original.quadrature_shape_values,
        original.quadrature_weights,
        original.density,
        original.rigidity_modulus,
    )
    np.testing.assert_array_equal(admitted.vertex_body_ids, np.zeros(4, dtype=np.int32))
    mixed_topology = CollisionSurfacePlan(
        np.arange(4, dtype=np.int64),
        ambient_dimension=3,
        body_ids=np.asarray([37, 37, 37, 38], dtype=np.int32),
        codimensional_mask=np.ones(4, dtype=np.bool_),
        allow_isolated_vertices=True,
    )
    with pytest.raises(ValueError, match="homogeneous"):
        AffineBodyPlan(
            mixed_topology,
            original.reference_controls,
            original.material_coordinates,
            original.vertex_body_ids,
            original.quadrature_shape_values,
            original.quadrature_weights,
            original.density,
            original.rigidity_modulus,
        )


def test_distinct_affine_control_bodies_cannot_share_global_body_identity() -> None:
    original = _body().plan
    faces = np.asarray(original.topology.faces)
    topology = CollisionSurfacePlan(
        np.arange(8, dtype=np.int64),
        ambient_dimension=3,
        faces=np.concatenate((faces, faces + 4)),
        body_ids=37,
    )
    controls = np.concatenate(
        (
            np.asarray(original.reference_controls),
            np.asarray(original.reference_controls)
            + np.asarray([2.0, 0.0, 0.0], dtype=np.float64)[None, None, :],
        ),
        axis=0,
    )
    with pytest.raises(ValueError, match="unique"):
        AffineBodyPlan(
            topology,
            controls,
            np.tile(np.asarray(original.material_coordinates), (2, 1)),
            np.repeat(np.arange(2, dtype=np.int32), 4),
            np.repeat(np.asarray(original.quadrature_shape_values), 2, axis=0),
            np.repeat(np.asarray(original.quadrature_weights), 2, axis=0),
            np.repeat(np.asarray(original.density), 2),
            np.repeat(np.asarray(original.rigidity_modulus), 2),
        )


def test_fem_orphan_free_nodal_dof_refuses_at_native_preparation() -> None:
    from phydrax.discretization._cell_mesh import CellMesh
    from phydrax.discretization.fem._generic import (
        FiniteElementFieldSpec,
        FiniteElementPlan,
    )
    from phydrax.discretization.fem._reference import lagrange_element

    coordinates = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=jnp.float64
    )
    cells = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    with pytest.raises(ValueError):
        mesh = CellMesh.from_triangles(coordinates, cells)
        FiniteElementPlan(
            mesh,
            FiniteElementFieldSpec(
                "u", lagrange_element("triangle", 1), component_shape=(2,)
            ),
        )
