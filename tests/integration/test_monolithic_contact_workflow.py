# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root

import phydrax as phx


def affine_body(
    *,
    offset: float = 0.0,
    body_id: int = 1,
    gravity: float = 0.0,
    backward_euler: bool = False,
) -> Any:
    controls = jnp.asarray(
        (
            (
                (-1.0, -1.0, offset),
                (1.0, -1.0, offset),
                (-1.0, 1.0, offset),
                (-1.0, -1.0, offset + 1.0),
            ),
        ),
        dtype=jnp.float64,
    )
    material = jnp.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ),
        dtype=jnp.float64,
    )
    topology = phx.discretization.CollisionSurfacePlan(
        jnp.arange(100 * body_id, 100 * body_id + 8, dtype=jnp.int64),
        ambient_dimension=3,
        faces=jnp.asarray(
            (
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            ),
            dtype=jnp.int32,
        ),
        body_ids=body_id,
    )
    # Tensor Gauss quadrature exactly integrates the affine cube's consistent mass.
    nodes = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
    quadrature = np.asarray(
        [(x, y, z) for x in nodes for y in nodes for z in nodes], dtype=np.float64
    )
    shapes = np.column_stack((1.0 - quadrature.sum(axis=1), quadrature))[None]
    plan = phx.applications.contact.AffineBodyPlan(
        topology,
        controls,
        material,
        jnp.zeros((8,), dtype=jnp.int32),
        shapes,
        np.full((1, 8), 0.5, dtype=np.float64),
        jnp.asarray((1.0,), dtype=jnp.float64),
        jnp.asarray((200.0,), dtype=jnp.float64),
        gravity=(0.0, 0.0, gravity),
    )
    method = (
        phx.applications.solid_mechanics.ImplicitNewmarkMethod(1.0, 1.0)
        if backward_euler
        else None
    )
    return phx.applications.contact.prepare_affine_body_participant(plan, method=method)


def fem_abd_case(*, capacity: int = 128) -> Any:
    coordinates = jnp.asarray(
        ((-0.4, -0.4, 1.08), (0.1, -0.4, 1.08), (-0.4, 0.1, 1.08), (-0.4, -0.4, 1.58)),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_tetrahedra(coordinates, cells)
    fem = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()

    def energy(values: Any, gradients: Any, points: Any, context: Any) -> Any:
        del values, points, context
        return 10.0 * jnp.sum(gradients * gradients, axis=(-1, -2))

    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "mixed-p1",
            "u",
            (phx.equations.CellEnergyAction("u", energy, action_id="mixed-p1-energy"),),
        ),
        fem,
    )
    participant = phx.applications.contact.prepare_finite_element_contact_participant(
        compiled,
        inversion=phx.discretization.SimplexInversionStepPlan(cells, coordinates),
    )
    abd = affine_body()
    scene = phx.discretization.ContactParticipantScene(
        (participant.geometry, abd.geometry)
    )
    q = (jnp.zeros_like(coordinates), jnp.zeros_like(abd.plan.reference_controls))
    rest = scene.positions(q)
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=rest
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        (participant, abd),
        scene,
        contact,
        phx.discretization.SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=0,
            edge_edge_capacity=capacity,
            face_vertex_capacity=capacity,
            activation_distance=0.1,
            envelope_radius=0.5,
        ),
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-8),
        geometry_owner_indices=(0, 1),
        fixed_geometry_states=(None, None),
    )
    history = (
        phx.applications.solid_mechanics.FiniteElementDynamicsState(
            q[0],
            jnp.broadcast_to(
                jnp.asarray((0.0, 0.0, -0.1), dtype=jnp.float64), q[0].shape
            ),
            jnp.zeros_like(q[0]),
        ),
        phx.applications.contact.AffineBodyHistory(
            jnp.zeros_like(q[1]), jnp.zeros_like(q[1])
        ),
    )
    state = phx.applications.contact.initialize_monolithic_contact_state(plan, q, history)
    return plan, state, compiled, coordinates


@pytest.mark.strict_jax
def test_fem_abd_shared_root_matches_p1_oracle_and_mixed_derivative() -> None:
    plan, state, compiled, coordinates = fem_abd_case()
    # Independent P1 shape gradients from the affine Vandermonde system.
    vandermonde = np.column_stack((np.ones(4), np.asarray(coordinates)))
    gradients = np.linalg.solve(vandermonde, np.eye(4))[1:, :].T
    volume = (
        abs(np.linalg.det(np.asarray(coordinates)[1:] - np.asarray(coordinates)[0])) / 6.0
    )
    stiffness = 20.0 * volume * gradients @ gradients.T
    probe = np.arange(12, dtype=np.float64).reshape(4, 3) * 0.001
    np.testing.assert_allclose(
        compiled.residual(jnp.asarray(probe)),
        stiffness @ probe,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    prepared = phx.applications.contact.prepare_monolithic_contact_step(plan, state, 0.01)
    direction = (
        jnp.zeros_like(state.coordinates[0]).at[:, 2].set(1.0),
        jnp.zeros_like(state.coordinates[1]),
    )
    action = prepared.residual_jvp(state.coordinates, direction)
    assert float(jnp.linalg.norm(action[1])) > 1.0e-6
    epsilon = 1.0e-6
    upper = tuple(
        q + epsilon * d for q, d in zip(state.coordinates, direction, strict=True)
    )
    lower = tuple(
        q - epsilon * d for q, d in zip(state.coordinates, direction, strict=True)
    )
    rp = prepared.problem.residual(upper, None)
    rm = prepared.problem.residual(lower, None)
    for value, plus, minus in zip(action, rp, rm, strict=True):
        np.testing.assert_allclose(
            value, (plus - minus) / (2.0 * epsilon), rtol=1.0e-5, atol=1.0e-7
        )
    result = phx.applications.contact.solve_monolithic_contact_step(prepared)
    assert bool(result.accepted)
    assert bool(result.safety.successful)
    assert result.contact.minimum_gap > 0.0
    assert result.diagnostics.final_residual_norm < 1.0e-6
    np.testing.assert_allclose(result.contact.action_reaction_residual, 0.0, atol=1.0e-9)
    np.testing.assert_allclose(result.contact.moment_residual, 0.0, atol=1.0e-9)
    fem_mass = volume / 20.0 * (np.ones((4, 4)) + np.eye(4))
    abd_mass = 4.0 * np.asarray(
        (
            (0.5, -1 / 3, -1 / 3, -1 / 3),
            (-1 / 3, 1 / 3, 0.25, 0.25),
            (-1 / 3, 0.25, 1 / 3, 0.25),
            (-1 / 3, 0.25, 0.25, 1 / 3),
        ),
        dtype=np.float64,
    )
    reference_controls = np.asarray(plan.participants[1].plan.reference_controls[0])
    reference_frame = (reference_controls[1:] - reference_controls[0]).T
    fem_velocity = np.asarray(state.histories[0].velocity)

    def independent_root_residual(packed: np.ndarray) -> np.ndarray:
        uf = packed[:12].reshape(4, 3)
        ua = packed[12:].reshape(4, 3)
        current_controls = reference_controls + ua
        frame = (current_controls[1:] - current_controls[0]).T
        deformation = np.linalg.solve(reference_frame.T, frame.T).T
        gradient_f = (
            4.0 * 200.0 / 2.0 * deformation @ (deformation.T @ deformation - np.eye(3))
        )
        gradient_frame = np.linalg.solve(reference_frame, gradient_f.T).T
        rigidity = np.vstack((-np.sum(gradient_frame, axis=1), gradient_frame.T))
        rf = stiffness @ uf + fem_mass @ (4.0 * (uf - 0.01 * fem_velocity) / 0.01**2)
        ra = rigidity + abd_mass @ (4.0 * ua / 0.01**2)
        blocks = (jnp.asarray(uf), jnp.asarray(ua[None]))
        contact = plan.contact.evaluate(
            plan.contact_scene.positions(blocks),
            prepared.candidate_epoch,
            participant_states=blocks,
        )
        return np.concatenate(
            (
                (rf - np.asarray(contact.state_force[0])).reshape(-1),
                (ra - np.asarray(contact.state_force[1][0])).reshape(-1),
            )
        )

    reference_root = root(
        independent_root_residual, np.zeros(24, dtype=np.float64), tol=1.0e-10
    )
    assert reference_root.success
    np.testing.assert_allclose(
        independent_root_residual(reference_root.x), 0.0, atol=1.0e-8
    )
    actual = np.concatenate(
        tuple(np.asarray(q).reshape(-1) for q in result.commit().coordinates)
    )
    np.testing.assert_allclose(actual, reference_root.x, rtol=1.0e-6, atol=1.0e-9)


def test_mixed_capacity_failure_is_atomic() -> None:
    plan, state, _, _ = fem_abd_case(capacity=1)
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(plan, state, 0.01)
    )
    assert not bool(result.accepted)
    assert result.diagnostics.overflow_count > 0
    assert eqx.tree_equal(result.rollback(), state)
    assert eqx.tree_equal(result.commit(), state)


def test_affine_stack_shared_transaction_preserves_total_momentum() -> None:
    bodies = tuple(
        affine_body(offset=1.05 * index, body_id=index + 1) for index in range(3)
    )
    scene = phx.discretization.ContactParticipantScene(
        tuple(body.geometry for body in bodies)
    )
    coordinates = tuple(jnp.zeros_like(body.plan.reference_controls) for body in bodies)
    histories = tuple(
        phx.applications.contact.AffineBodyHistory(jnp.zeros_like(q), jnp.zeros_like(q))
        for q in coordinates
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=scene.positions(coordinates)
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        bodies,
        scene,
        contact,
        phx.discretization.SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=0,
            # Three unordered body pairs; 18 edges and 12 faces/8 vertices per box.
            edge_edge_capacity=3 * 18 * 18,
            face_vertex_capacity=3 * 2 * 12 * 8,
            activation_distance=0.1,
            envelope_radius=0.5,
        ),
        phx.discretization.InclusionCCDPlan(),
        geometry_owner_indices=(0, 1, 2),
        fixed_geometry_states=(None, None, None),
        # Momentum balance resolves much smaller errors than the default relative root tolerance.
        policy=phx.applications.contact.ContactSolvePolicy(
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1.0e-11, relative_residual=1.0e-11
            )
        ),
    )
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan, coordinates, histories
    )
    for _ in range(3):
        result = phx.applications.contact.solve_monolithic_contact_step(
            phx.applications.contact.prepare_monolithic_contact_step(plan, state, 0.002)
        )
        assert bool(result.accepted)
        assert bool(result.safety.successful)
        assert result.contact.minimum_gap > 0.0
        state = result.commit()
        momentum = sum(
            jnp.sum(body.plan.control_mass @ history.velocity, axis=(0, 1))
            for body, history in zip(bodies, state.histories, strict=True)
        )
        np.testing.assert_allclose(momentum, 0.0, atol=1.0e-7)


def mpm_abd_case() -> Any:
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    bounds = jnp.asarray(((-2.0, -2.0, -2.0), (3.0, 3.0, 3.0)), dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(bounds)
    positions = jnp.asarray(
        ((-0.3, -0.3, 1.08), (-0.2, -0.35, 1.08), (-0.35, -0.2, 1.08)), dtype=jnp.float64
    )
    volume = jnp.full((3,), 0.01, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3, dtype=jnp.int64), volume, ambient_dimension=3
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("monolithic-landing", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True, True), support_margin=0.0
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = jnp.broadcast_to(
        jnp.asarray((0.0, 0.0, -0.1), dtype=jnp.float64), positions.shape
    )
    initial = compiled.initialize_state(positions, velocity, volume, arguments)
    prepared = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).prepare_residual(
        initial, 1.0e-4, arguments
    )
    mechanics = phx.applications.contact.prepare_implicit_mpm_contact_participant(
        prepared
    )
    topology = phx.discretization.CollisionSurfacePlan(
        jnp.arange(3, dtype=jnp.int64),
        ambient_dimension=3,
        body_ids=0,
        faces=jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
    )
    geometry = phx.applications.contact.prepare_mpm_contact_participant(
        topology, positions, mechanics.particle_displacement_operator
    )
    abd = affine_body(backward_euler=True)
    scene = phx.discretization.ContactParticipantScene((geometry, abd.geometry))
    coordinates = (
        jnp.zeros_like(prepared.initial_velocity),
        jnp.zeros_like(abd.plan.reference_controls),
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=scene.positions(coordinates)
    )
    plan = phx.applications.contact.MonolithicContactPlan(
        (mechanics, abd),
        scene,
        contact,
        phx.discretization.SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=0,
            edge_edge_capacity=64,
            face_vertex_capacity=64,
            activation_distance=0.1,
            envelope_radius=0.5,
        ),
        phx.discretization.InclusionCCDPlan(),
        geometry_owner_indices=(0, 1),
        fixed_geometry_states=(None, None),
        # Match the native standalone MPM linear accuracy and restart capacity;
        # the grid/control mass scales otherwise leave a linear-error plateau.
        policy=phx.applications.contact.ContactSolvePolicy(
            method=phx.solver.ImplicitMPMMethodPlan().nonlinear_method
        ),
    )
    state = phx.applications.contact.initialize_monolithic_contact_state(
        plan,
        coordinates,
        (
            initial,
            phx.applications.contact.AffineBodyHistory(
                jnp.zeros_like(coordinates[1]), jnp.zeros_like(coordinates[1])
            ),
        ),
    )
    return plan, state, prepared


@pytest.mark.strict_jax
def test_mpm_abd_landing_matches_frozen_residual_and_particle_grid_reference() -> None:
    plan, state, mpm = mpm_abd_case()
    step = phx.applications.contact.prepare_monolithic_contact_step(
        plan, state, mpm.step_size
    )
    probe = mpm.step_size * mpm.initial_velocity
    mechanics = plan.participants[0]
    evaluation = mechanics.evaluate(
        probe, state.histories[0], state.coordinates[0], state.time, step.step_size, None
    )
    reference = mpm.evaluate_detailed(probe / step.step_size)
    np.testing.assert_allclose(
        evaluation.residual, reference.residual / step.step_size, atol=1.0e-12
    )
    assert eqx.tree_equal(evaluation.trial_history, reference.candidate_state)
    assert eqx.tree_equal(mpm.state, state.histories[0])
    routes = mpm.execution_routes.assignment_state
    indices = np.asarray(routes.indices).reshape((3, -1))
    weights = np.asarray(routes.weights).reshape((3, -1))
    valid = np.asarray(routes.valid).reshape((3, -1))
    mass = np.asarray(mpm.mass)
    particle_velocity = np.asarray(state.histories[0].particles.velocity)
    grid_mass = np.zeros(mpm.grid_mass.size, dtype=np.float64)
    grid_momentum = np.zeros((mpm.grid_mass.size, 3), dtype=np.float64)
    # Independent scalar scatter, without using the implicit participant or gather.
    for particle in range(3):
        for route in range(indices.shape[1]):
            if valid[particle, route]:
                node = indices[particle, route]
                grid_mass[node] += mass[particle] * weights[particle, route]
                grid_momentum[node] += (
                    mass[particle]
                    * weights[particle, route]
                    * particle_velocity[particle]
                )
    np.testing.assert_allclose(
        grid_mass, np.asarray(mpm.grid_mass).reshape(-1), atol=1.0e-12
    )
    np.testing.assert_allclose(
        grid_momentum, np.asarray(mpm.grid_momentum).reshape(-1, 3), atol=1.0e-12
    )
    result = phx.applications.contact.solve_monolithic_contact_step(step)
    assert bool(result.accepted)
    assert result.contact.minimum_gap > 0.0
    accepted = result.commit()
    q = accepted.coordinates[0]
    candidate = mpm.evaluate_detailed(q / step.step_size).candidate_state
    np.testing.assert_allclose(
        accepted.histories[0].particles.position,
        candidate.particles.position,
        atol=1.0e-11,
    )
    grid_velocity = np.asarray(q / step.step_size).reshape(-1, 3)
    gathered = np.zeros((3, 3), dtype=np.float64)
    for particle in range(3):
        for route in range(indices.shape[1]):
            if valid[particle, route]:
                gathered[particle] += (
                    weights[particle, route] * grid_velocity[indices[particle, route]]
                )
    np.testing.assert_allclose(
        accepted.histories[0].particles.position,
        state.histories[0].particles.position + mpm.step_size * gathered,
        atol=1.0e-11,
    )
    # Internal/contact forces exchange momentum, never manufacture an external impulse.
    initial_momentum = np.sum(mass[:, None] * particle_velocity, axis=0)
    final_momentum = np.sum(mass[:, None] * gathered, axis=0)
    positions = plan.contact_scene.positions(state.coordinates)
    contact = plan.contact.evaluate(
        positions, step.candidate_epoch, participant_states=state.coordinates
    )
    initial_contact_force = np.asarray(contact.surface_force[:3])
    assert np.linalg.norm(initial_contact_force) > 0.0
    # Explicit Euler momentum balance at the accepted boundary; mechanics and
    # particle-grid transfers are independent of the implicit contact participant.
    # The canonical barrier is intentionally shared, not a second contact law.
    grid_force = np.zeros_like(grid_momentum)
    for particle in range(3):
        for route in range(indices.shape[1]):
            if valid[particle, route]:
                grid_force[indices[particle, route]] += (
                    weights[particle, route] * initial_contact_force[particle]
                )
    reference_grid_momentum = grid_momentum + float(mpm.step_size) * grid_force
    reference_grid_velocity = np.zeros_like(grid_momentum)
    for node in range(grid_mass.size):
        if grid_mass[node] > 0.0:
            reference_grid_velocity[node] = (
                reference_grid_momentum[node] / grid_mass[node]
            )
    reference_velocity = np.zeros_like(particle_velocity)
    for particle in range(3):
        for route in range(indices.shape[1]):
            if valid[particle, route]:
                reference_velocity[particle] += (
                    weights[particle, route]
                    * reference_grid_velocity[indices[particle, route]]
                )
    reference_impulse = float(mpm.step_size) * np.sum(initial_contact_force, axis=0)
    actual_impulse = final_momentum - initial_momentum
    np.testing.assert_allclose(
        actual_impulse, reference_impulse, rtol=2.0e-2, atol=1.0e-8
    )
    np.testing.assert_allclose(
        accepted.histories[0].particles.position,
        state.histories[0].particles.position + mpm.step_size * reference_velocity,
        rtol=0.0,
        atol=2.0e-7,
    )
    abd = plan.participants[1]
    final_momentum += np.asarray(
        jnp.sum(abd.plan.control_mass @ accepted.histories[1].velocity, axis=(0, 1))
    )
    np.testing.assert_allclose(final_momentum, initial_momentum, rtol=1.0e-6, atol=1.0e-8)
    assert eqx.tree_equal(
        mpm.execution_routes, plan.participants[0].prepared.execution_routes
    )


@pytest.mark.parametrize("translation", (0.0, 1.0e-3))
def test_mpm_safe_rigid_translation_admits_entire_step(translation: float) -> None:
    plan, state, mpm = mpm_abd_case()
    mechanics = plan.participants[0]
    q = state.coordinates[0]
    direction = jnp.full_like(q, translation)
    # Constant grid displacement reproduces a rigid particle translation: F is
    # unchanged and these particles remain strictly inside [-2,3]^3 throughout.
    limit = mechanics.step_limit(q, direction, q, state.time, mpm.step_size, None)
    assert bool(limit.successful)
    assert bool(limit.conservative)
    np.testing.assert_array_equal(limit.maximum_step, 1.0)
