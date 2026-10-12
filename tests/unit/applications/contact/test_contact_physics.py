#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _finite_element_contact_case(*, friction: Any = False) -> Any:
    coordinates = jnp.asarray(
        ((-0.25, 0.08), (0.25, 0.08), (0.0, 0.48)), dtype=jnp.float64
    )
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(coordinates, cells)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("triangle", 1),
            component_shape=(2,),
        ),
    ).prepare()

    def density(fields: Any, geometry: Any, context: Any) -> Any:
        del geometry, context
        gradient = fields["u"].gradient
        return 0.5 * 20.0 * jnp.sum(gradient * gradient, axis=(-1, -2))

    functional = phx.variational.Functional(
        "contact-test",
        (
            phx.variational.LocalIntegralTerm(
                "elasticity",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", gradient=True),),
                density=density,
                density_id="contact-test-elasticity",
            ),
        ),
        variable_fields=("u",),
    )
    compiled = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u"},
        regions={"body": None},
    )
    participant = phx.applications.contact.prepare_finite_element_contact_participant(
        compiled,
        inversion=phx.discretization.SimplexInversionStepPlan(cells, coordinates),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11), dtype=jnp.int64),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),), dtype=jnp.int32),
        body_ids=1,
        static_mask=True,
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=jnp.float64),
        phx.discretization.static_collision_operator(
            compiled.state_space, 2, 2, dtype=np.float64
        ),
    )
    scene = phx.discretization.ContactParticipantScene(
        (participant.geometry, phx.discretization.LinearContactParticipant(static))
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=24,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=0.5,
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene, rest_positions=jnp.concatenate((coordinates, static.rest_positions))
    )
    friction_plan = (
        phx.applications.contact.LaggedCoulombFrictionPlan(
            0.25, 1.0e-2, maximum_lag_iterations=3, lag_tolerance=1.0e-4
        ).prepare(scene, contact)
        if friction
        else None
    )
    mechanics = phx.applications.solid_mechanics.FiniteElementDynamicsState(
        jnp.zeros_like(coordinates),
        jnp.broadcast_to(jnp.asarray((0.15, -0.05)), coordinates.shape),
        jnp.zeros_like(coordinates),
    )
    dynamics = phx.applications.contact.MonolithicContactPlan(
        (participant,),
        scene,
        contact,
        search,
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-7),
        geometry_owner_indices=(0, None),
        fixed_geometry_states=(None, jnp.zeros_like(coordinates)),
        friction=friction_plan,
        policy=phx.applications.contact.ContactSolvePolicy(
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1.0e-5 if friction else 1.0e-7,
                relative_residual=1.0e-5 if friction else 1.0e-7,
                maximum_steps=40,
            ),
            maximum_lag_iterations=3,
            lag_tolerance=1.0e-4,
        ),
    )
    accepted = phx.applications.contact.initialize_monolithic_contact_state(
        dynamics, (mechanics.displacement,), (mechanics,)
    )
    return coordinates, cells, compiled, scene, search, contact, accepted, dynamics


def test_contact_physics_scenario_1() -> None:
    _, _, compiled, _, _, _, accepted, _ = _finite_element_contact_case()
    displacement = accepted.coordinates[0].at[2, 0].set(0.1)
    gradient = compiled.residual(displacement)

    assert compiled.potential_compatible
    np.testing.assert_allclose(
        gradient,
        jax.grad(compiled.potential)(displacement),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    _, _, _, scene, search, contact, accepted, _ = _finite_element_contact_case()
    positions = scene.positions((accepted.coordinates[0], accepted.coordinates[0]))
    epoch = search.build(scene, positions)
    evaluation = contact.evaluate(
        positions,
        epoch,
        participant_states=(accepted.coordinates[0], accepted.coordinates[0]),
    )

    assert bool(epoch.successful)
    assert bool(evaluation.successful)
    assert evaluation.energy > 0.0
    assert evaluation.minimum_gap > 0.0
    np.testing.assert_allclose(evaluation.action_reaction_residual, 0.0, atol=2.0e-10)
    np.testing.assert_allclose(evaluation.moment_residual, 0.0, atol=2.0e-10)
    _, _, _, _, _, _, accepted, dynamics = _finite_element_contact_case()
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(dynamics, accepted, 0.02)
    )

    assert bool(result.accepted)
    assert bool(result.safety.successful)
    assert all(bool(limit.successful) for limit in result.participant_step_limits)
    assert bool(result.contact.successful)
    assert result.contact.minimum_gap > 0.0
    assert result.accepted_state.state_version == accepted.state_version + 1
    _, _, _, scene, _, contact, accepted, original = _finite_element_contact_case()
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
        envelope_radius=0.5,
    )
    dynamics = phx.applications.contact.MonolithicContactPlan(
        original.participants,
        scene,
        contact,
        search,
        phx.discretization.InclusionCCDPlan(),
        geometry_owner_indices=original.geometry_owner_indices,
        fixed_geometry_states=original.fixed_geometry_states,
    )
    accepted = phx.applications.contact.initialize_monolithic_contact_state(
        dynamics, accepted.coordinates, accepted.histories
    )
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(dynamics, accepted, 0.02)
    )

    assert not bool(result.accepted)
    assert bool(result.rollback_applied)
    np.testing.assert_array_equal(
        result.accepted_state.coordinates[0],
        accepted.coordinates[0],
    )
    assert int(result.rejection_reasons) & int(
        phx.applications.contact.ContactRejectionReason.SEARCH
    )
    _, _, _, scene, search, contact, accepted, dynamics = _finite_element_contact_case(
        friction=True
    )
    positions = scene.positions((accepted.coordinates[0], accepted.coordinates[0]))
    epoch = search.build(scene, positions)
    friction = dynamics.friction
    assert friction is not None
    state = friction.build_state(positions, epoch)
    velocity = jnp.concatenate(
        (
            scene.participants[0].velocities(
                accepted.coordinates[0], accepted.histories[0].velocity
            ),
            jnp.zeros((2, 2), dtype=jnp.float64),
        )
    )
    evaluation = friction.evaluate(
        velocity,
        state,
        participant_states=(accepted.coordinates[0], accepted.coordinates[0]),
    )

    assert bool(evaluation.successful)
    assert evaluation.active_contacts > 0
    assert evaluation.dissipation_rate >= 0.0
    _, _, _, _, _, _, accepted, dynamics = _finite_element_contact_case(friction=True)
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(dynamics, accepted, 0.02)
    )

    assert bool(result.accepted)
    assert result.friction is not None and bool(result.friction.successful)
    assert result.accepted_state.friction_state is not None
    assert result.diagnostics.lag_iterations >= 1
    assert result.diagnostics.lag_residual <= dynamics.policy.lag_tolerance


def test_fixed_route_contact_sensitivity_is_qualified() -> None:
    _, _, _, _, _, _, accepted, dynamics = _finite_element_contact_case()
    arguments = {"load_scale": jnp.asarray(1.0, dtype=jnp.float64)}
    tangent = {"load_scale": jnp.asarray(1.0, dtype=jnp.float64)}
    result = phx.applications.contact.solve_monolithic_contact_step(
        phx.applications.contact.prepare_monolithic_contact_step(
            dynamics, accepted, 0.02, args=arguments
        )
    )
    derivative = phx.applications.contact.monolithic_contact_solution_jvp(
        dynamics, result, arguments, tangent
    )
    assert bool(derivative.successful)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(derivative.value))
    adjoint = phx.applications.contact.monolithic_contact_solution_vjp(
        dynamics, result, arguments, (jnp.ones_like(accepted.coordinates[0]),)
    )
    assert bool(adjoint.successful)
    np.testing.assert_allclose(
        sum(jnp.sum(leaf) for leaf in jax.tree.leaves(derivative.value)),
        adjoint.value["load_scale"],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
