# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax.applications.contact import ContactMeritMode
from phydrax.applications.solid_mechanics import (
    DruckerPragerMPMConstitutivePlan,
    DruckerPragerParameters,
)


def _evaluation(
    q: Array,
    history: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Any:
    del accepted, time, dt
    residual = 40.0 * q - args["force"]
    potential = 20.0 * jnp.sum(q * q) - jnp.sum(args["force"] * q)
    valid = jnp.all(jnp.isfinite(q)) & args["valid"]
    return phx.applications.contact.ContactParticipantEvaluation(
        residual,
        history + jnp.sum(q * q),
        potential,
        valid,
        valid,
        valid,
        jnp.asarray(True),
        jnp.asarray(0.0, dtype=q.dtype),
        jnp.asarray(1.0, dtype=q.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        {},
    )


def _jvp(
    q: Array,
    direction: Array,
    history: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Array:
    del q, history, accepted, time, dt, args
    return 40.0 * direction


def _potential(
    q: Array,
    history: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Array:
    return _evaluation(q, history, accepted, time, dt, args).potential_energy


def _limit(
    q: Array,
    direction: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Any:
    del direction, accepted, time, dt, args
    return phx.applications.contact.ContactParticipantStepLimit(
        jnp.asarray(1.0, dtype=q.dtype),
        jnp.asarray(True),
        jnp.asarray(True),
        phx.discretization.ContactGuaranteeEvidence(
            phx.discretization.ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE,
            backend_id="unrestricted-linear-mechanics",
        ),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int64),
        "spring-domain",
    )


def mixed_case(*, capacity: int = 24, merit: ContactMeritMode = "automatic") -> Any:
    space = phx.linalg.ArraySpace((2, 2), dtype=jnp.float64)
    participants = tuple(
        phx.applications.contact.make_implicit_contact_participant(
            space,
            _evaluation,
            _jvp,
            _limit,
            participant_id=f"elastic-block-{index}",
            potential_action=_potential,
            potential_hvp_action=_jvp,
        )
        for index in range(2)
    )
    rest = (
        jnp.asarray(((-0.4, 0.08), (0.4, 0.08)), dtype=jnp.float64),
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
    )
    geometry = tuple(
        phx.discretization.LinearContactParticipant(
            phx.discretization.PreparedCollisionSurface(
                phx.discretization.CollisionSurfacePlan(
                    jnp.asarray((10 * index, 10 * index + 1), dtype=jnp.int64),
                    ambient_dimension=2,
                    edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
                    body_ids=index,
                ),
                rest[index],
                phx.discretization.selection_collision_operator(
                    space, jnp.asarray((0, 1), dtype=jnp.int32)
                ),
            )
        )
        for index in range(2)
    )
    scene = phx.discretization.ContactParticipantScene(geometry)
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=jnp.concatenate(rest)
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        participants,
        scene,
        contact,
        phx.discretization.SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=capacity,
            edge_edge_capacity=0,
            face_vertex_capacity=0,
            activation_distance=0.1,
            envelope_radius=0.5,
        ),
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-8),
        geometry_owner_indices=(0, 1),
        fixed_geometry_states=(None, None),
        policy=phx.applications.contact.ContactSolvePolicy(merit=merit),
    )
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan,
        (jnp.zeros((2, 2), dtype=jnp.float64),) * 2,
        (jnp.asarray(2.0, dtype=jnp.float64), jnp.asarray(3.0, dtype=jnp.float64)),
    )
    args = {
        "force": jnp.asarray(((0.01, -0.02), (0.01, -0.02)), dtype=jnp.float64),
        "valid": jnp.asarray(True),
    }
    return plan, state, args


@pytest.mark.strict_jax
def test_full_mixed_jvp_and_both_cross_blocks() -> None:
    plan, state, args = mixed_case()
    prepared = phx.applications.contact.prepare_monolithic_contact_step(
        plan, state, 0.02, args=args
    )
    q = state.coordinates
    direction = (
        jnp.asarray(((0.02, 0.1), (-0.03, 0.05)), dtype=jnp.float64),
        jnp.asarray(((0.01, -0.04), (0.02, -0.06)), dtype=jnp.float64),
    )
    action = prepared.residual_jvp(q, direction, args)
    epsilon = 1.0e-6
    plus = tuple(x + epsilon * d for x, d in zip(q, direction, strict=True))
    minus = tuple(x - epsilon * d for x, d in zip(q, direction, strict=True))
    rp = prepared.problem.residual(plus, args)
    rm = prepared.problem.residual(minus, args)
    for actual, left, right in zip(action, rp, rm, strict=True):
        np.testing.assert_allclose(
            actual, (left - right) / (2.0 * epsilon), rtol=2.0e-6, atol=1.0e-8
        )
    for source in range(2):
        seed = tuple(
            direction[index] if index == source else jnp.zeros_like(q[index])
            for index in range(2)
        )
        cross = prepared.residual_jvp(q, seed, args)[1 - source]
        assert float(jnp.linalg.norm(cross)) > 1.0e-4
    force = jnp.asarray(((0.3, -0.4), (0.2, 0.1)), dtype=jnp.float64)
    for geometry, coordinate, seed in zip(
        plan.contact_scene.participants, q, direction, strict=True
    ):
        evidence = geometry.duality_evidence(coordinate, seed, force)
        assert bool(evidence.valid)
        np.testing.assert_allclose(evidence.residual, 0.0, atol=1.0e-12)


@pytest.mark.parametrize("failure", ("capacity", "participant"))
def test_atomic_rollback_preserves_every_block_history_and_clock(failure: str) -> None:
    plan, state, args = mixed_case(capacity=1 if failure == "capacity" else 24)
    if failure == "participant":
        args = {**args, "valid": jnp.asarray(False)}
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            plan, state, 0.02, args=args
        )
    )
    assert not bool(result.accepted)
    assert bool(result.rollback_applied)
    assert eqx.tree_equal(result.rollback(), state)
    assert eqx.tree_equal(result.commit(), state)
    assert eqx.tree_equal(result.accepted_state, state)
    if failure == "capacity":
        assert result.diagnostics.overflow_count > 0
        assert not bool(result.candidate_epoch.successful)


@pytest.mark.parametrize("merit", ("automatic", "potential", "residual"))
def test_merit_selection_and_equilibrium_transaction(merit: ContactMeritMode) -> None:
    plan, state, args = mixed_case(merit=merit)
    result = phx.applications.contact.solve_monolithic_contact_equilibrium(
        phx.applications.contact.prepare_monolithic_contact_equilibrium(
            plan, state, args=args
        )
    )
    assert bool(result.accepted)
    assert bool(result.diagnostics.potential_merit_used) == (merit != "residual")
    assert result.commit().time == state.time
    assert result.commit().step == state.step
    assert result.commit().state_version == state.state_version + 1
    assert eqx.tree_equal(result.rollback(), state)


def test_sensitivity_dynamic_arguments_matches_resolve_and_adjoint() -> None:
    plan, state, args = mixed_case()
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            plan, state, 0.02, args=args
        )
    )
    tangent = {
        "force": jnp.ones_like(args["force"]),
        "valid": jnp.zeros((), dtype=jax.dtypes.float0),
    }
    derivative = phx.applications.contact.monolithic_contact_solution_jvp(
        plan, result, args, tangent
    )
    assert bool(derivative.successful)
    epsilon = 1.0e-5
    roots = []
    for sign in (-1.0, 1.0):
        perturbed = {**args, "force": args["force"] + sign * epsilon}
        root = phx.applications.contact.solve_monolithic_contact_step(
            phx.applications.contact.prepare_monolithic_contact_step(
                plan, state, 0.02, args=perturbed
            )
        )
        assert bool(root.accepted)
        roots.append(root.commit().coordinates)
    for actual, lower, upper in zip(derivative.value, roots[0], roots[1], strict=True):
        np.testing.assert_allclose(
            actual, (upper - lower) / (2.0 * epsilon), rtol=2.0e-5, atol=1.0e-8
        )
    cotangent = tuple(jnp.ones_like(q) for q in state.coordinates)
    adjoint = phx.applications.contact.monolithic_contact_solution_vjp(
        plan, result, args, cotangent
    )
    assert bool(adjoint.successful)
    np.testing.assert_allclose(
        sum(jnp.sum(q) for q in derivative.value),
        jnp.sum(adjoint.value["force"]),
        rtol=1.0e-8,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("theta", (0.35, 0.55))
def test_incline_coulomb_force_and_acceleration_outside_regularization(
    theta: float,
) -> None:
    plan, state, _ = mixed_case()
    rotation = jnp.asarray(
        ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))),
        dtype=jnp.float64,
    )
    positions = plan.contact_scene.positions(state.coordinates) @ rotation.T
    epoch = plan.search.build(plan.contact_scene, positions)
    friction = phx.applications.contact.LaggedCoulombFrictionPlan(0.2, 1.0e-3).prepare(
        plan.contact_scene, plan.contact
    )
    frame = friction.build_state(positions, epoch)
    mass, gravity = 2.0, 9.81
    normal = mass * gravity * np.cos(theta)
    count = jnp.sum(frame.valid)
    assert count > 0
    frame = eqx.tree_at(
        lambda item: item.normal_force, frame, jnp.where(frame.valid, normal / count, 0.0)
    )
    tangent = rotation[:, 0]
    velocities = jnp.concatenate(
        (jnp.broadcast_to(-tangent, (2, 2)), jnp.zeros((2, 2), dtype=jnp.float64))
    )
    evaluation = friction.evaluate(
        velocities, frame, participant_states=state.coordinates
    )
    assert bool(evaluation.successful)
    tangential_force = jnp.sum(evaluation.surface_force[:2] @ tangent)
    np.testing.assert_allclose(tangential_force, 0.2 * normal, rtol=1.0e-10, atol=1.0e-10)
    acceleration = gravity * np.sin(theta) - tangential_force / mass
    np.testing.assert_allclose(
        acceleration,
        gravity * np.sin(theta) - 0.2 * gravity * np.cos(theta),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(evaluation.dissipation_rate, 0.2 * normal, rtol=1.0e-10)


def test_candidate_routes_remain_frozen_across_repeated_linearizations() -> None:
    plan, state, args = mixed_case()
    prepared = phx.applications.contact.prepare_monolithic_contact_step(
        plan, state, 0.02, args=args
    )
    epoch = prepared.candidate_epoch
    for scale in (0.0, 0.001, -0.001):
        q = tuple(value + scale for value in state.coordinates)
        prepared.residual_jvp(q, tuple(jnp.ones_like(value) for value in q), args)
        assert eqx.tree_equal(prepared.candidate_epoch, epoch)
        assert eqx.tree_equal(prepared.accepted, state)


def test_ccd_crossing_is_limited_before_exact_segment_plane_impact() -> None:
    plan, state, _ = mixed_case()
    start = plan.contact_scene.positions(state.coordinates)
    end = start.at[:2, 1].add(-0.2)
    epoch = plan.search.build(plan.contact_scene, start, end_positions=end)
    safety = phx.discretization.collision_free_step_limit(
        plan.ccd, plan.contact_scene, epoch, start, end
    )
    # Both upper vertices move on vertical lines inside the lower segment.
    # Thus the first intersection occurs exactly at t = 0.08 / 0.2.
    exact_impact = 0.08 / 0.2
    assert bool(safety.successful)
    assert 0.0 < float(safety.step_size) < exact_impact
    accepted_path_gap = 0.08 - 0.2 * safety.step_size
    assert accepted_path_gap > 0.0


@final
class _DPModel(StrictModule):
    material: DruckerPragerMPMConstitutivePlan
    parameters: DruckerPragerParameters

    def response(self, q: Array, history: Array, time: Array, dt: Array | None) -> Any:
        deformation = (
            jnp.asarray(
                ((1.0, 0.16, 0.0), (0.0, 0.92, 0.0), (0.0, 0.0, 1.08)), dtype=q.dtype
            )
            .at[:2, :2]
            .add(q)
        )
        # Rate-independent static increments have no physical rate dependence.
        interval = jnp.asarray(1.0, dtype=q.dtype) if dt is None else dt
        return self.material.evaluate_linearized(
            deformation, history, 1.0, self.parameters, time, interval
        )


@final
class _DPResidualAction(StrictModule):
    model: _DPModel

    def __call__(
        self,
        q: Array,
        history: Array,
        accepted: Array,
        time: Array,
        dt: Array | None,
        args: dict[str, Array],
    ) -> Any:
        del accepted
        response = self.model.response(q, history, time, dt)
        residual = response.response.first_piola[:2, :2] + 100.0 * q - args["force"]
        return phx.applications.contact.ContactParticipantEvaluation(
            residual,
            response.response.trial_state,
            None,
            response.response.successful,
            response.response.successful,
            response.tangent_successful,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=q.dtype),
            jnp.asarray(1.0, dtype=q.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            {},
        )


@final
class _DPJVPAction(StrictModule):
    model: _DPModel

    def __call__(
        self,
        q: Array,
        direction: Array,
        history: Array,
        accepted: Array,
        time: Array,
        dt: Array | None,
        args: dict[str, Array],
    ) -> Array:
        del accepted, args
        response = self.model.response(q, history, time, dt)
        return (
            jnp.einsum(
                "ijkl,kl->ij", response.algorithmic_tangent[:2, :2, :2, :2], direction
            )
            + 100.0 * direction
        )


@pytest.mark.strict_jax
def test_nonassociated_dp_nonsymmetric_jacobian_uses_residual_merit() -> None:
    original, state, args = mixed_case()
    model = _DPModel(
        DruckerPragerMPMConstitutivePlan(),
        DruckerPragerParameters(10.0, 30.0, 0.05, 0.5, 0.2, 1.0),
    )
    participant = phx.applications.contact.make_implicit_contact_participant(
        original.participants[0].dof_space,
        _DPResidualAction(model),
        _DPJVPAction(model),
        _limit,
        participant_id="nonassociated-drucker-prager-patch",
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        (participant, original.participants[1]),
        original.contact_scene,
        original.contact,
        original.search,
        original.ccd,
        geometry_owner_indices=(0, 1),
        fixed_geometry_states=(None, None),
        policy=phx.applications.contact.ContactSolvePolicy(merit="automatic"),
    )
    history = model.material.initialize_state((), jnp.float64)
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan, state.coordinates, (history, state.histories[1])
    )
    response = model.response(
        state.coordinates[0], history, state.time, jnp.asarray(0.02)
    )
    args = {**args, "force": response.response.first_piola[:2, :2]}
    prepared = phx.applications.contact.prepare_monolithic_contact_step(
        plan, state, 0.02, args=args
    )
    columns = []
    for index in range(4):
        direction = (
            jnp.eye(4, dtype=jnp.float64)[index].reshape(2, 2),
            jnp.zeros_like(state.coordinates[1]),
        )
        columns.append(
            prepared.residual_jvp(state.coordinates, direction, args)[0].reshape(-1)
        )
    jacobian = jnp.stack(columns, axis=1)
    assert float(jnp.linalg.norm(jacobian - jacobian.T)) > 1.0e-3
    result = phx.applications.contact.solve_monolithic_contact_step(prepared)
    assert bool(result.accepted)
    assert not bool(result.diagnostics.potential_merit_used)
    assert result.diagnostics.linear_iterations > 0
    assert result.diagnostics.final_residual_norm < 1.0e-6
    assert bool(result.participant_evidence[0].tangent_successful)
    with pytest.raises(ValueError, match="potential"):
        potential_plan = phx.applications.contact.MonolithicContactPlan(
            plan.participants,
            plan.contact_scene,
            plan.contact,
            plan.search,
            plan.ccd,
            geometry_owner_indices=(0, 1),
            fixed_geometry_states=(None, None),
            policy=phx.applications.contact.ContactSolvePolicy(merit="potential"),
        )
        potential_state = phx.applications.contact.initialize_monolithic_contact_state(
            potential_plan, state.coordinates, state.histories
        )
        phx.applications.contact.prepare_monolithic_contact_step(
            potential_plan, potential_state, 0.02, args=args
        )


def test_lag_exhaustion_rolls_back_every_participant() -> None:
    original, state, args = mixed_case()
    friction = phx.applications.contact.LaggedCoulombFrictionPlan(
        0.3, 1.0e-3, maximum_lag_iterations=1, lag_tolerance=1.0e-30
    ).prepare(original.contact_scene, original.contact)
    plan = phx.applications.contact.MonolithicContactPlan(
        original.participants,
        original.contact_scene,
        original.contact,
        original.search,
        original.ccd,
        geometry_owner_indices=(0, 1),
        fixed_geometry_states=(None, None),
        friction=friction,
        policy=phx.applications.contact.ContactSolvePolicy(
            maximum_lag_iterations=1, lag_tolerance=1.0e-30
        ),
    )
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan, state.coordinates, state.histories
    )
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            plan, state, 0.02, args=args
        )
    )
    assert not bool(result.accepted)
    assert result.diagnostics.lag_residual > 0.0
    assert eqx.tree_equal(result.commit(), state)
    assert eqx.tree_equal(result.rollback(), state)
    with pytest.raises(ValueError, match="friction"):
        phx.applications.contact.prepare_monolithic_contact_equilibrium(
            plan, state, args=args
        )


def _incline_potential(
    q: Array,
    history: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Array:
    del time
    if dt is None:
        raise ValueError("Incline reference is incremental.")
    prediction = accepted + dt * history
    return 0.5 * args["node_mass"] * jnp.sum((q - prediction) ** 2) / dt**2 - args[
        "node_mass"
    ] * jnp.sum(q * args["gravity"])


def _incline_evaluation(
    q: Array,
    history: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Any:
    if dt is None:
        raise ValueError("Incline reference is incremental.")
    residual = args["node_mass"] * (
        (q - accepted - dt * history) / dt**2 - args["gravity"]
    )
    valid = jnp.all(jnp.isfinite(residual))
    return phx.applications.contact.ContactParticipantEvaluation(
        residual,
        (q - accepted) / dt,
        _incline_potential(q, history, accepted, time, dt, args),
        valid,
        valid,
        valid,
        jnp.asarray(True),
        jnp.asarray(0.0, dtype=q.dtype),
        dt,
        jnp.asarray(0, dtype=jnp.int32),
        {},
    )


def _incline_jvp(
    q: Array,
    direction: Array,
    history: Array,
    accepted: Array,
    time: Array,
    dt: Array | None,
    args: dict[str, Array],
) -> Array:
    del q, history, accepted, time
    if dt is None:
        raise ValueError("Incline reference is incremental.")
    return args["node_mass"] * direction / dt**2


def test_monolithic_lagged_incline_matches_analytic_sliding_acceleration() -> None:
    original, initial, _ = mixed_case()
    theta, gravity, coefficient = 0.45, 9.81, 0.2
    rotation = jnp.asarray(
        ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))),
        dtype=jnp.float64,
    )
    tangent, normal = rotation[:, 0], rotation[:, 1]
    # Rotate reference surfaces at the explicit host preparation boundary.
    geometries = tuple(
        phx.discretization.LinearContactParticipant(
            phx.discretization.PreparedCollisionSurface(
                geometry.surface.plan,
                geometry.surface.rest_positions @ rotation.T,
                geometry.surface.displacement_operator,
            )
        )
        for geometry in original.contact_scene.participants
    )
    scene = phx.discretization.ContactParticipantScene(geometries)
    zeros = initial.coordinates[0]
    positions = scene.positions((zeros, zeros))
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=positions
    )
    epoch = original.search.build(scene, positions)
    initial_force = contact.evaluate(
        positions, epoch, participant_states=(zeros, zeros)
    ).surface_force
    normal_load = jnp.sum(initial_force[:2] @ normal)
    assert normal_load > 0.0
    node_mass = normal_load / (2.0 * gravity * np.cos(theta))
    participant = phx.applications.contact.make_implicit_contact_participant(
        original.participants[0].dof_space,
        _incline_evaluation,
        _incline_jvp,
        _limit,
        participant_id="incline-backward-euler",
        potential_action=_incline_potential,
        potential_hvp_action=_incline_jvp,
    )
    friction = phx.applications.contact.LaggedCoulombFrictionPlan(
        coefficient, 1.0e-3, maximum_lag_iterations=8, lag_tolerance=1.0e-8
    ).prepare(scene, contact)
    plan = phx.applications.contact.MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        original.search,
        original.ccd,
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, zeros),
        friction=friction,
        policy=phx.applications.contact.ContactSolvePolicy(
            merit="residual",
            maximum_lag_iterations=8,
            lag_tolerance=1.0e-8,
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1.0e-8, relative_residual=1.0e-10
            ),
        ),
    )
    velocity = jnp.broadcast_to(-tangent, zeros.shape)
    accepted = phx.applications.contact.initialize_monolithic_contact_state(
        plan, (zeros,), (velocity,)
    )
    args = {
        "node_mass": node_mass,
        "gravity": jnp.asarray((0.0, -gravity), dtype=jnp.float64),
    }
    dt = 1.0e-3
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            plan, accepted, dt, args=args
        )
    )
    assert bool(result.accepted)
    assert result.friction is not None and bool(result.friction.successful)
    assert result.diagnostics.lag_iterations >= 1
    assert result.diagnostics.lag_residual <= plan.policy.lag_tolerance
    acceleration = jnp.mean((result.commit().histories[0] - velocity) @ (-tangent)) / dt
    reference = gravity * np.sin(theta) - coefficient * gravity * np.cos(theta)
    np.testing.assert_allclose(acceleration, reference, rtol=2.0e-3, atol=1.0e-5)
    assert (
        float(jnp.min(result.commit().histories[0] @ (-tangent)))
        > friction.plan.velocity_threshold
    )


def _quadratic_positions(q: Array) -> Array:
    rest = jnp.asarray(((-0.4, 0.08), (0.4, 0.08)), dtype=q.dtype)
    return (rest + q).at[:, 1].add(q[:, 0] ** 2)


def _quadratic_velocity(q: Array, direction: Array) -> Array:
    return direction.at[:, 1].add(2.0 * q[:, 0] * direction[:, 0])


def _quadratic_pullback(q: Array, force: Array) -> Array:
    return force.at[:, 0].add(2.0 * q[:, 0] * force[:, 1])


def _quadratic_bounds(start: Array, end: Array) -> tuple[Array, Array]:
    lower, upper = jnp.minimum(start, end), jnp.maximum(start, end)
    square_lower = jnp.where(
        (lower[:, 0] <= 0.0) & (upper[:, 0] >= 0.0),
        0.0,
        jnp.minimum(lower[:, 0] ** 2, upper[:, 0] ** 2),
    )
    square_upper = jnp.maximum(lower[:, 0] ** 2, upper[:, 0] ** 2)
    rest = jnp.asarray(((-0.4, 0.08), (0.4, 0.08)), dtype=start.dtype)
    return (
        (rest + lower).at[:, 1].add(square_lower),
        (rest + upper).at[:, 1].add(square_upper),
    )


def quadratic_case(*, certified: bool = True) -> Any:
    original, state, args = mixed_case()
    geometry = phx.discretization.FunctionContactParticipant(
        original.contact_scene.participants[0].surface_plan,
        original.participants[0].dof_space,
        _quadratic_positions,
        tangent_space=original.participants[0].dof_space,
        velocity_action=_quadratic_velocity,
        effort_pullback_action=_quadratic_pullback,
        bounds_action=_quadratic_bounds if certified else None,
        participant_id="quadratic-contact-map",
    )
    scene = phx.discretization.ContactParticipantScene(
        (geometry, original.contact_scene.participants[1])
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=scene.positions(state.coordinates)
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        original.participants,
        scene,
        contact,
        original.search,
        original.ccd,
        geometry_owner_indices=(0, 1),
        fixed_geometry_states=(None, None),
    )
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan, state.coordinates, state.histories
    )
    return plan, state, args


@pytest.mark.strict_jax
def test_nonlinear_geometry_full_jvp_retains_pullback_curvature() -> None:
    plan, state, args = quadratic_case()
    prepared = phx.applications.contact.prepare_monolithic_contact_step(
        plan, state, 0.02, args=args
    )
    direction = (
        jnp.asarray(((0.2, 0.01), (-0.1, 0.02)), dtype=jnp.float64),
        jnp.zeros_like(state.coordinates[1]),
    )
    full = prepared.residual_jvp(state.coordinates, direction, args)
    epsilon = 1.0e-6
    upper = tuple(
        q + epsilon * d for q, d in zip(state.coordinates, direction, strict=True)
    )
    lower = tuple(
        q - epsilon * d for q, d in zip(state.coordinates, direction, strict=True)
    )
    rp, rm = (
        prepared.problem.residual(upper, args),
        prepared.problem.residual(lower, args),
    )
    for actual, plus, minus in zip(full, rp, rm, strict=True):
        np.testing.assert_allclose(
            actual, (plus - minus) / (2.0 * epsilon), rtol=2.0e-6, atol=1.0e-8
        )

    def frozen_pullback_force(q: tuple[Array, Array]) -> Any:
        return plan.contact.evaluate(
            plan.contact_scene.positions(q),
            prepared.candidate_epoch,
            participant_states=state.coordinates,
        ).state_force

    force, frozen_action = jax.jvp(
        frozen_pullback_force, (state.coordinates,), (direction,)
    )
    curvature = full[0] - (40.0 * direction[0] - frozen_action[0])
    expected = (
        jnp.zeros_like(direction[0])
        .at[:, 0]
        .set(-2.0 * direction[0][:, 0] * force[0][:, 1])
    )
    assert float(jnp.linalg.norm(expected)) > 1.0e-4
    np.testing.assert_allclose(curvature, expected, rtol=1.0e-10, atol=1.0e-10)
    evidence = plan.contact_scene.participants[0].duality_evidence(
        upper[0], direction[0], jnp.ones_like(direction[0])
    )
    assert bool(evidence.valid)


def test_nonlinear_geometry_without_full_path_provider_refuses() -> None:
    with pytest.raises(ValueError, match="trajectory|bound|conservative"):
        plan, state, args = quadratic_case(certified=False)
        phx.applications.contact.prepare_monolithic_contact_step(
            plan, state, 0.02, args=args
        )
