#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _finite_element_contact_case(*, friction=False):
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

    def density(fields, geometry, context):
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
    moving = phx.discretization.prepare_cell_mesh_collision_surface(
        mesh, compiled.state_space, body_id=0
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
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=24,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    contact = phx.applications.contact.ConvergentContactPotentialPlan(0.1, 20.0).prepare(
        scene
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
    accepted = phx.applications.contact.ContactDynamicsState(mechanics)
    dynamics = phx.applications.contact.prepare_finite_element_contact_dynamics(
        compiled,
        accepted,
        scene,
        contact,
        search,
        phx.discretization.InclusionCCDPlan(time_tolerance=1.0e-7),
        inversion=phx.discretization.SimplexInversionStepPlan(cells, coordinates),
        friction=friction_plan,
        solve_policy=phx.applications.contact.ContactSolvePolicy(
            absolute_gradient=1.0e-5 if friction else 1.0e-7,
            relative_gradient=1.0e-5 if friction else 1.0e-7,
            maximum_iterations=40,
        ),
    )
    return coordinates, cells, compiled, scene, search, contact, accepted, dynamics


def test_declared_finite_element_potential_generates_existing_residual():
    _, _, compiled, _, _, _, accepted, _ = _finite_element_contact_case()
    displacement = accepted.mechanics.displacement.at[2, 0].set(0.1)
    gradient = compiled.residual(displacement)

    assert compiled.potential_compatible
    np.testing.assert_allclose(
        gradient,
        jax.grad(compiled.potential)(displacement),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_contact_potential_is_finite_balanced_and_positive():
    _, _, _, scene, search, contact, accepted, _ = _finite_element_contact_case()
    positions = scene.positions(accepted.mechanics.displacement)
    epoch = search.build(scene, positions)
    evaluation = contact.evaluate(positions, epoch)

    assert bool(epoch.successful)
    assert bool(evaluation.successful)
    assert evaluation.energy > 0.0
    assert evaluation.minimum_gap > 0.0
    np.testing.assert_allclose(evaluation.action_reaction_residual, 0.0, atol=2.0e-10)
    np.testing.assert_allclose(evaluation.moment_residual, 0.0, atol=2.0e-10)


def test_lagged_friction_is_finite_and_dissipative():
    _, _, _, scene, search, contact, accepted, dynamics = _finite_element_contact_case(
        friction=True
    )
    positions = scene.positions(accepted.mechanics.displacement)
    epoch = search.build(scene, positions)
    friction = dynamics.friction
    assert friction is not None
    state = friction.build_state(positions, epoch)
    velocity = scene.map_values(accepted.mechanics.velocity)
    evaluation = friction.evaluate(velocity, state)

    assert bool(evaluation.successful)
    assert evaluation.active_contacts > 0
    assert evaluation.dissipation_rate >= 0.0


def test_lagged_friction_contact_step_converges_without_fallback():
    _, _, _, _, _, _, accepted, dynamics = _finite_element_contact_case(friction=True)
    result = phx.applications.contact.solve_finite_element_contact_step(
        phx.applications.contact.prepare_finite_element_contact_step(
            dynamics, accepted, 0.02
        )
    )

    assert bool(result.accepted)
    assert result.friction is not None and bool(result.friction.successful)
    assert result.accepted_state.friction_state is not None
    assert result.diagnostics.lag_iterations >= 1
    assert result.diagnostics.lag_residual <= dynamics.friction.plan.lag_tolerance


def test_contact_newmark_step_is_transactional_and_safe():
    _, _, _, _, _, _, accepted, dynamics = _finite_element_contact_case()
    result = phx.applications.contact.solve_finite_element_contact_step(
        phx.applications.contact.prepare_finite_element_contact_step(
            dynamics, accepted, 0.02
        )
    )

    assert bool(result.accepted)
    assert bool(result.safety.successful)
    assert result.inversion is not None and bool(result.inversion.successful)
    assert bool(result.contact.successful)
    assert result.contact.minimum_gap > 0.0
    assert result.accepted_state.state_version == accepted.state_version + 1


def test_contact_capacity_failure_rolls_back_complete_dynamic_state():
    coordinates, cells, compiled, scene, _, contact, accepted, _ = (
        _finite_element_contact_case()
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=1,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    dynamics = phx.applications.contact.prepare_finite_element_contact_dynamics(
        compiled,
        accepted,
        scene,
        contact,
        search,
        phx.discretization.InclusionCCDPlan(),
        inversion=phx.discretization.SimplexInversionStepPlan(cells, coordinates),
    )
    result = phx.applications.contact.solve_finite_element_contact_step(
        phx.applications.contact.prepare_finite_element_contact_step(
            dynamics, accepted, 0.02
        )
    )

    assert not bool(result.accepted)
    assert bool(result.rollback_applied)
    np.testing.assert_array_equal(
        result.accepted_state.mechanics.displacement,
        accepted.mechanics.displacement,
    )
    assert int(result.rejection_reasons) & int(
        phx.applications.contact.ContactRejectionReason.SEARCH
    )


def test_fixed_route_contact_sensitivity_is_qualified():
    _, _, _, scene, _, contact, accepted, dynamics = _finite_element_contact_case()
    result = phx.applications.contact.solve_finite_element_contact_step(
        phx.applications.contact.prepare_finite_element_contact_step(
            dynamics, accepted, 0.02
        )
    )
    rest = jnp.concatenate(
        tuple(surface.rest_positions for surface in scene.surfaces), axis=0
    )
    arguments = phx.applications.contact.ContactDynamicsSensitivityArguments(
        rest,
        contact.plan.stiffness,
        accepted.mechanics.displacement,
        accepted.mechanics.velocity,
        accepted.mechanics.acceleration,
        0.02,
    )
    tangent = phx.applications.contact.ContactDynamicsSensitivityArguments(
        jnp.zeros_like(rest),
        jnp.asarray(1.0),
        jnp.zeros_like(accepted.mechanics.displacement),
        jnp.zeros_like(accepted.mechanics.velocity),
        jnp.zeros_like(accepted.mechanics.acceleration),
        0.0,
    )
    derivative = phx.applications.contact.contact_dynamics_solution_jvp(
        dynamics,
        result,
        arguments,
        tangent,
    )

    assert bool(derivative.successful)
    assert jnp.all(jnp.isfinite(derivative.value))
