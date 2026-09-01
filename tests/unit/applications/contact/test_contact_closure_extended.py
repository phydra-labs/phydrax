#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _closure_case():
    source = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray(((-0.5, 0.05), (0.5, 0.05))),
        phx.discretization.selection_collision_operator(source, jnp.asarray((0, 1))),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
        pair_policy=phx.discretization.ContactPairPolicy(
            2,
            body_ids=jnp.ones((2,), dtype=jnp.int64),
            material_ids=jnp.zeros((2,), dtype=jnp.int64),
            static_mask=jnp.ones((2,), dtype=bool),
        ),
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (1.0, 0.0))),
        phx.discretization.static_collision_operator(source, 2, 2),
    )
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    positions = scene.positions(source.zeros())
    velocities = scene.map_values(
        jnp.broadcast_to(jnp.asarray((0.2, -0.05)), source.shape)
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    epoch = search.build(scene, positions)
    kinematics = phx.discretization.evaluate_contact_kinematics(
        scene,
        epoch,
        positions,
        velocities,
        0.01,
        activation_distance=0.1,
    )
    materials = phx.applications.contact.ContactMaterialPairTable.uniform(
        normal_stiffness=50.0,
        static_friction=0.5,
        dynamic_friction=0.4,
        restitution=0.2,
        adhesion_energy=0.01,
        thermal_conductance=2.0,
        electrical_conductance=3.0,
        wear_coefficient=1.0e-4,
        hardness=10.0,
        roughness=1.0e-3,
    )
    closure = phx.applications.contact.ContactClosurePlan(
        phx.applications.contact.AdhesiveBarrierNormalLaw(0.1, 0.08),
        materials,
        tangential=phx.applications.contact.RegularizedCoulombContactLaw(1.0e-3),
        evolution=phx.applications.contact.FrictionWearEvolutionLaw(
            critical_slip_distance=0.1,
            damage_onset=0.06,
            damage_completion=0.2,
        ),
        transport=phx.applications.contact.CoupledContactTransportLaw(gap_decay=0.1),
    )
    state = phx.applications.contact.ContactRouteState.empty(0, 1, closure.closure_id)
    transition = phx.applications.contact.remap_contact_route_state(state, kinematics)
    return scene, positions, kinematics, closure, transition.candidate


def test_composed_closure_assembles_balanced_stateful_multiphysics_response():
    scene, positions, kinematics, closure, state = _closure_case()
    capacity = state.capacity
    evaluation = phx.applications.contact.evaluate_contact_closure(
        closure,
        kinematics,
        state,
        driving_jump=jnp.broadcast_to(jnp.asarray((10.0, 2.0, 0.5)), (capacity, 3)),
    )
    assembly = phx.applications.contact.assemble_smooth_contact(
        kinematics, evaluation, positions
    )
    transport = evaluation.batches[0].transport
    flux = phx.applications.contact.assemble_contact_fluxes(
        transport, kinematics.batches[0].quadrature_weight
    )

    assert bool(evaluation.evidence.successful)
    assert bool(assembly.successful)
    assert bool(flux.successful)
    np.testing.assert_allclose(assembly.action_reaction_residual, 0.0, atol=1.0e-10)
    assert jnp.all(evaluation.candidate_state.wear_depth >= state.wear_depth)
    assert jnp.all(evaluation.candidate_state.adhesion_damage >= state.adhesion_damage)


def test_cross_discretization_participants_share_one_closure_without_state_aliasing():
    scene, _, _, closure, _ = _closure_case()
    participants = phx.discretization.ContactParticipantScene(
        tuple(
            phx.discretization.LinearContactParticipant(surface)
            for surface in scene.surfaces
        )
    )
    states = tuple(
        participant.source_space.zeros() for participant in participants.participants
    )
    rates = (
        jnp.broadcast_to(jnp.asarray((0.2, -0.05)), states[0].shape),
        jnp.zeros_like(states[1]),
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    route_state = phx.applications.contact.ContactRouteState.empty(
        0, 1, closure.closure_id
    )
    rest = jnp.concatenate(
        tuple(surface.rest_positions for surface in scene.surfaces),
        axis=0,
    )
    result = phx.applications.contact.evaluate_cross_discretization_contact(
        participants,
        states,
        rates,
        search,
        closure,
        route_state,
        0.01,
        rest,
        activation_distance=0.1,
    )

    assert bool(result.successful)
    assert len(result.generalized_forces) == 2
    np.testing.assert_allclose(
        result.assembly.action_reaction_residual, 0.0, atol=1.0e-10
    )


def test_contact_cone_impact_and_rolling_resistance_are_dissipative():
    _, _, kinematics, _, _ = _closure_case()
    local_dimension = 2
    route_count = sum(batch.capacity for batch in kinematics.batches)
    blocks = jnp.broadcast_to(jnp.eye(local_dimension), (route_count, 2, 2))
    materials = phx.applications.contact.ContactMaterialPairTable.uniform(
        normal_stiffness=1.0,
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.2,
    )
    program = phx.applications.contact.build_contact_cone_program(
        kinematics, materials, blocks, compliance=1.0e-8
    )
    result = phx.applications.contact.solve_contact_cone(program)
    rolling = phx.applications.contact.evaluate_rolling_spinning_resistance(
        phx.applications.contact.RollingSpinningResistancePlan(
            rolling_coefficient=0.05,
            spinning_coefficient=0.02,
        ),
        result.impulse[:, 0],
        jnp.broadcast_to(jnp.asarray((0.2, 0.1, 0.3)), (route_count, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (route_count, 3)),
        jnp.ones((route_count,)),
    )
    differentiable_program = phx.applications.contact.ContactConeProgram(
        jnp.asarray(((-1.0, 0.0),)),
        jnp.eye(2),
        jnp.zeros((2,)),
        jnp.asarray((0.5,)),
        jnp.asarray((1,), dtype=jnp.int64),
        jnp.asarray((True,)),
        1,
        "interior-cone-derivative",
    )
    cone_derivative = phx.applications.contact.contact_cone_solution_jvp(
        differentiable_program,
        jnp.asarray(((0.1, 0.0),)),
        jnp.zeros((2, 2)),
    )

    assert bool(result.evidence.successful)
    assert bool(rolling.successful)
    assert result.evidence.cone_defect < 1.0e-8
    assert jnp.all(rolling.dissipated_work >= 0.0)
    assert bool(cone_derivative.evidence.successful)
    assert jnp.all(jnp.isfinite(cone_derivative.impulse_tangent))


def _interface_case(gap=-0.05):
    interface = phx.discretization.ContactInterfacePlan(
        jnp.asarray(((0, 1),)),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((0, 1),)),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((0.0, 1.0),)),
        jnp.asarray((1.0,)),
        plus_node_count=2,
        minus_node_count=2,
    )
    plus = jnp.asarray(((-0.5, gap), (0.5, gap)))
    minus = jnp.asarray(((-1.0, 0.0), (1.0, 0.0)))
    kinematics = phx.discretization.evaluate_contact_interface(interface, plus, minus)
    return interface, kinematics


def test_mortar_nitsche_and_mesh_tie_share_balanced_interface_assembly():
    interface, kinematics = _interface_case()
    mortar_plan = phx.applications.contact.MortarContactPlan(penalty=100.0, friction=0.3)
    state = phx.applications.contact.MortarContactState.initialize(interface, mortar_plan)
    mortar = phx.applications.contact.evaluate_mortar_contact(
        mortar_plan, interface, kinematics, state
    )
    nitsche = phx.applications.contact.evaluate_nitsche_contact(
        phx.applications.contact.UnbiasedNitscheContactPlan(100.0),
        interface,
        kinematics,
        jnp.asarray((0.0,)),
        minus_normal_stress=jnp.asarray((0.0,)),
    )
    tie = phx.applications.contact.evaluate_mesh_tie(
        phx.applications.contact.MeshTiePlan(100.0),
        interface,
        kinematics,
    )
    mortar_derivative = phx.applications.contact.mortar_gap_jvp(
        mortar_plan,
        interface,
        kinematics,
        state,
        jnp.ones_like(kinematics.gap),
    )

    assert bool(mortar.evidence.successful)
    assert bool(nitsche.successful)
    assert bool(tie.successful)
    assert mortar.candidate_state.normal_multiplier[0] > 0.0
    np.testing.assert_allclose(mortar.residual.action_reaction_residual, 0.0)
    np.testing.assert_allclose(nitsche.residual.action_reaction_residual, 0.0)
    assert bool(mortar_derivative.evidence.successful)


def test_hydroelastic_rough_and_lubricated_patch_closures_are_physical():
    interface, kinematics = _interface_case()
    velocity = jnp.asarray(((0.2, -0.1),))
    hydro = phx.applications.contact.evaluate_hydroelastic_contact(
        interface,
        kinematics,
        phx.applications.contact.HydroelasticMaterialPlan(
            modulus=1.0e5,
            slab_thickness=0.1,
            dissipation=0.2,
            friction=0.3,
        ),
        None,
        velocity,
    )
    hertz = phx.applications.contact.hertz_sphere_half_space(1.0, 1000.0, 10.0)
    homogenized = phx.applications.contact.evaluate_homogenized_rough_contact(
        phx.applications.contact.HomogenizedRoughContactPlan(
            pressure_scale=10.0,
            separation_scale=0.1,
            rms_roughness=0.02,
        ),
        jnp.asarray((0.05,)),
    )
    rough_plan = phx.applications.contact.PeriodicRoughContactPlan(
        jnp.ones((4, 4)), maximum_iterations=100, tolerance=1.0e-8
    )
    rough = phx.applications.contact.solve_periodic_rough_contact(
        rough_plan, -0.1 * jnp.ones((4, 4))
    )
    lubrication = phx.applications.contact.evaluate_lubrication_contact(
        phx.applications.contact.LubricationContactPlan(
            viscosity=0.1,
            minimum_film_thickness=1.0e-4,
            asperity_transition=0.01,
        ),
        jnp.asarray((0.005,)),
        jnp.asarray((-0.1,)),
        jnp.asarray(((0.2,),)),
        jnp.asarray((1.0,)),
        asperity_pressure=jnp.asarray((100.0,)),
    )

    assert bool(hydro.evidence.successful)
    assert hydro.pressure[0] > 0.0
    assert hertz.contact_radius > 0.0 and hertz.maximum_pressure > 0.0
    assert bool(homogenized.successful)
    assert bool(rough.evidence.successful)
    assert bool(lubrication.successful)
    assert lubrication.dissipated_power[0] >= 0.0


def test_state_transfer_preconditioner_and_fixed_branch_derivatives_are_qualified():
    _, _, kinematics, closure, state = _closure_case()
    capacity = state.capacity
    evaluation = phx.applications.contact.evaluate_contact_closure(
        closure, kinematics, state
    )
    derivative = phx.applications.contact.contact_closure_gap_jvp(
        closure,
        kinematics,
        state,
        tuple(jnp.ones_like(batch.gap) for batch in kinematics.batches),
    )
    graph = phx.applications.contact.ContactGraphPlan.from_kinematics(kinematics)
    blocks = jnp.broadcast_to(jnp.eye(2), (capacity, 2, 2))
    preconditioned = phx.applications.contact.apply_contact_block_preconditioner(
        phx.applications.contact.ContactBlockPreconditionerPlan(2),
        blocks,
        jnp.ones((capacity, 2)),
        graph=graph,
    )
    parent_slots = jnp.arange(capacity, dtype=jnp.int32)[:, None]
    transfer = phx.applications.contact.transfer_contact_route_state(
        phx.applications.contact.ContactStateTransferPlan(
            state.route_keys + 1000,
            parent_slots,
            jnp.ones_like(parent_slots, dtype=jnp.float64),
        ),
        evaluation.candidate_state,
    )

    assert bool(derivative.evidence.successful)
    assert bool(preconditioned.successful)
    assert bool(transfer.evidence.successful)
    assert jnp.all(transfer.state.wear_depth >= evaluation.candidate_state.wear_depth)


def test_advanced_cone_solver_families_recover_interior_contact_solution():
    program = phx.applications.contact.ContactConeProgram(
        jnp.asarray(((-1.0, 0.0),)),
        jnp.eye(2),
        jnp.zeros((2,)),
        jnp.asarray((0.5,)),
        jnp.asarray((1,), dtype=jnp.int64),
        jnp.asarray((True,)),
        1,
        "advanced-cone-program",
    )
    sap = phx.applications.contact.solve_contact_sap(
        program,
        solver=phx.applications.contact.SAPContactSolverPlan(
            maximum_iterations=100, tolerance=1.0e-8
        ),
    )
    semismooth = phx.applications.contact.solve_contact_semismooth(
        program,
        solver=phx.applications.contact.SemismoothContactSolverPlan(tolerance=1.0e-8),
    )
    primal_dual = phx.applications.contact.solve_contact_primal_dual(
        program,
        solver=phx.applications.contact.PrimalDualContactSolverPlan(tolerance=1.0e-6),
    )

    assert bool(sap.evidence.successful)
    assert bool(semismooth.evidence.successful)
    assert bool(primal_dual.evidence.successful)
    np.testing.assert_allclose(sap.impulse, ((1.0, 0.0),), atol=1.0e-6)
    np.testing.assert_allclose(semismooth.impulse, ((1.0, 0.0),), atol=1.0e-6)
    np.testing.assert_allclose(primal_dual.impulse, ((1.0, 0.0),), atol=1.0e-4)


def test_rigid_participant_and_geometric_filter_preserve_explicit_kinematics():
    surface_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
    )
    rigid = phx.applications.contact.RigidContactParticipant(
        surface_plan,
        jnp.asarray(((-0.5, 0.0), (0.5, 0.0))),
        jnp.asarray((0, 0)),
        body_count=1,
    )
    state = (jnp.zeros((1, 2)), jnp.zeros((1, 1)))
    direction = (
        jnp.asarray(((0.2, -0.1),)),
        jnp.asarray(((0.3,),)),
    )
    force = jnp.asarray(((1.0, 2.0), (-0.5, 0.25)))
    duality = rigid.duality_evidence(state, direction, force)
    scene, positions, kinematics, _, _ = _closure_case()
    filtered = phx.discretization.filter_geometric_contacts(
        phx.discretization.GeometricContactFilterPlan(
            require_closed_surface=False,
            normal_alignment=-1.0,
        ),
        scene,
        kinematics,
        positions,
    )
    point_space = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    point_participant = phx.applications.contact.prepare_point_contact_participant(
        point_space,
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
    )

    assert bool(duality.valid)
    assert bool(filtered.evidence.successful)
    assert point_participant.surface_plan.intrinsic_dimension == 0
    np.testing.assert_allclose(
        point_participant.positions(point_space.zeros()),
        ((0.0, 0.0), (1.0, 0.0)),
    )
    assert filtered.evidence.output_contacts > 0
