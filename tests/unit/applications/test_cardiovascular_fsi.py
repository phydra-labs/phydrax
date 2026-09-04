#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cardiovascular.hemodynamics._ale import (
    ALEMinimumGapRoute,
    ALETransitionStatus,
    CardiovascularALEPlan,
    CardiovascularALEState,
)
from phydrax.applications.cardiovascular.hemodynamics._immersed_fsi import (
    build_immersed_fsi_participants,
    build_immersed_lbm_participant,
    ImmersedDirectForcingPlan,
    ImmersedFEMAdvanceResult,
    ImmersedLBMAdvanceResult,
    SparseMarkerTransferPlan,
)
from phydrax.applications.cardiovascular.hemodynamics._leaflets import (
    CutCellLeafletRoute,
    LeafletContactWorkflowPlan,
    LeafletStructuralAdvanceResult,
    LeafletTransitionStatus,
)
from phydrax.solver._partitioned_coupling_runtime import advance_coupling_window
from phydrax.solver._partitioned_coupling_types import CouplingWindow


def _grid(count=8, *, periodic=True):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def _lattice(count=8):
    return phx.discretization.LatticeBoltzmannPlan(
        _grid(count, periodic=True), phx.discretization.D2Q9()
    ).prepare()


def _finite_volume(count=4):
    return phx.discretization.FiniteVolumePlan(_grid(count, periodic=False)).prepare()


def test_sparse_lbm_marker_transfer_preserves_transpose_force_torque_and_power():
    lattice = _lattice()
    position = jnp.asarray([[0.31, 0.37], [0.68, 0.59], [0.2, 0.8]])
    transfer_plan = SparseMarkerTransferPlan(
        lattice,
        jnp.asarray([101, 202, 303]),
        stencil_width=4,
        minimum_coverage=1.0,
    )
    transfer = transfer_plan.prepare(position, active=jnp.asarray([True, True, False]))
    relation = transfer.relation(position)
    coordinates = lattice.grid.points.reshape(lattice.grid.shape + (2,))
    grid_velocity = jnp.stack(
        (
            0.2 + coordinates[..., 0] - 0.3 * coordinates[..., 1],
            -0.1 + 0.4 * coordinates[..., 0] + coordinates[..., 1],
        ),
        axis=-1,
    )
    marker_force = jnp.asarray([[0.4, -0.3], [-0.2, 0.5], [9.0, 9.0]])
    diagnostics = transfer.diagnostics(
        relation,
        grid_velocity,
        marker_force,
        marker_velocity=transfer.interpolate(relation, grid_velocity),
    )

    assert transfer_plan.route_width == 16
    assert relation.cell_indices.shape == (3, 16)
    assert relation.evidence.successful
    assert diagnostics.successful
    assert jnp.max(relation.evidence.partition_residual) < 1.0e-10
    assert jnp.max(relation.evidence.first_moment_residual) < 1.0e-10
    assert jnp.max(jnp.abs(diagnostics.force_residual)) < 1.0e-10
    assert jnp.max(jnp.abs(diagnostics.torque_residual)) < 1.0e-10
    assert jnp.abs(diagnostics.transpose_power_residual) < 1.0e-10
    assert jnp.abs(diagnostics.interface_power_residual) < 1.0e-10


def test_fixed_marker_routes_fail_closed_after_losing_prepared_coverage():
    lattice = _lattice()
    initial = jnp.asarray([[0.45, 0.45]])
    transfer = SparseMarkerTransferPlan(
        lattice, jnp.asarray([7]), minimum_coverage=1.0
    ).prepare(initial)

    relation = transfer.relation(jnp.asarray([[0.9, 0.9]]))

    assert not relation.evidence.successful
    assert not bool(relation.evidence.covered[0])
    assert relation.evidence.coverage_fraction[0] < transfer.plan.minimum_coverage


def test_sparse_direct_forcing_couples_a_compliant_marker_without_dense_action():
    lattice = _lattice()
    position = jnp.asarray([[0.44, 0.53]])
    transfer = SparseMarkerTransferPlan(
        lattice, jnp.asarray([19]), minimum_coverage=1.0
    ).prepare(position)
    relation = transfer.relation(position)
    cell_measure = lattice.cell_size**2
    exact_measure = cell_measure / jnp.sum(relation.weights[0] ** 2)
    forcing = ImmersedDirectForcingPlan(
        transfer, iteration_count=2, convergence_tolerance=1.0e-10
    )
    velocity = jnp.zeros(lattice.grid.shape + (2,))
    target = jnp.asarray([[0.08, -0.03]])
    result = forcing.apply(
        velocity,
        jnp.ones(lattice.grid.shape),
        position,
        target,
        jnp.asarray([exact_measure]),
        0.01,
    )

    assert result.evidence.successful
    assert result.evidence.maximum_velocity_residual < 1.0e-10
    assert jnp.max(jnp.abs(result.evidence.transpose.force_residual)) < 1.0e-10
    assert jnp.max(jnp.abs(result.evidence.transpose.torque_residual)) < 1.0e-10
    assert result.force_density.shape == lattice.grid.shape + (2,)


def test_lbm_participant_gates_actual_post_advance_no_slip():
    lattice = _lattice()
    position = jnp.asarray([[0.44, 0.53]])
    transfer = SparseMarkerTransferPlan(
        lattice, jnp.asarray([29]), minimum_coverage=1.0
    ).prepare(position)
    relation = transfer.relation(position)
    marker_measure = lattice.cell_size**2 / jnp.sum(relation.weights[0] ** 2)
    forcing = ImmersedDirectForcingPlan(
        transfer, iteration_count=2, convergence_tolerance=1.0e-9
    )
    velocity = jnp.zeros(lattice.grid.shape + (2,))
    density = jnp.ones(lattice.grid.shape)

    def ignore_fluid_force(_window, state, _force_density, _args):
        candidate_velocity, candidate_density = state
        return ImmersedLBMAdvanceResult(
            state,
            candidate_velocity,
            candidate_density,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    participant = build_immersed_lbm_participant(
        forcing,
        lambda state, _args: state,
        ignore_fluid_force,
        jnp.asarray([marker_measure]),
        coupling_id="post-lbm-gate",
        position_reference=1.0,
        velocity_reference=0.1,
        force_reference=1.0,
    )
    result = participant.advance_window(
        CouplingWindow(0, 0.0, 0.01),
        (velocity, density),
        (position, jnp.asarray([[0.08, -0.03]])),
        None,
    )
    direct = result.auxiliary[1]
    post_lbm = result.auxiliary[2]

    assert direct.evidence.successful
    assert not result.successful
    assert not post_lbm.successful
    assert post_lbm.maximum_velocity_residual > forcing.convergence_tolerance


def test_partitioned_lbm_fem_builder_runs_added_mass_iteration():
    lattice = _lattice(count=6)
    position = jnp.asarray([[0.5, 0.5]])
    transfer = SparseMarkerTransferPlan(
        lattice, jnp.asarray([11]), minimum_coverage=1.0
    ).prepare(position)
    relation = transfer.relation(position)
    marker_measure = lattice.cell_size**2 / jnp.sum(relation.weights[0] ** 2)
    forcing = ImmersedDirectForcingPlan(
        transfer, iteration_count=2, convergence_tolerance=1.0e-9
    )
    grid_velocity = jnp.zeros(lattice.grid.shape + (2,))
    density = jnp.ones(lattice.grid.shape)

    def fluid_fields(state, _args):
        return state

    def advance_fluid(window, state, force_density, _args):
        velocity, rho = state
        candidate_velocity = velocity + window.size * force_density / rho[..., None]
        return ImmersedLBMAdvanceResult(
            (candidate_velocity, rho),
            candidate_velocity,
            rho,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    def advance_solid(_window, _state, body_load, _args):
        candidate_velocity = 0.02 * body_load
        candidate = (position, candidate_velocity)
        return ImmersedFEMAdvanceResult(
            candidate,
            position,
            candidate_velocity,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.sqrt(jnp.sum(candidate_velocity**2)),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    bundle = build_immersed_fsi_participants(
        forcing,
        fluid_fields,
        advance_fluid,
        advance_solid,
        jnp.asarray([marker_measure]),
        position_reference=10.0,
        velocity_reference=0.2,
        force_reference=5.0,
        position_tolerance=1.0e-8,
        velocity_tolerance=1.0e-7,
        force_tolerance=1.0e-7,
        damping=0.5,
        maximum_iterations=30,
    )
    initial_marker_velocity = jnp.asarray([[0.1, 0.0]])
    prepared = bundle.prepare(
        (grid_velocity, density),
        (position, initial_marker_velocity),
        position,
        initial_marker_velocity,
    )
    result = advance_coupling_window(prepared, prepared.reference_state, 0.01)

    fluid = bundle.graph.subsystems[0]
    assert len(fluid.input_ports) == 2
    assert fluid.input_ports[0].space.space_id != fluid.input_ports[1].space.space_id
    assert fluid.input_ports[0].reference_scale == 10.0
    assert fluid.input_ports[1].reference_scale == 0.2
    assert fluid.output_ports[0].reference_scale == 5.0
    assert len(bundle.policy.tolerances) == 3
    assert result.successful
    assert result.converged
    assert result.diagnostics.coupling_iterations > 1
    assert jnp.max(result.diagnostics.exchange_residual_norms) <= 1.0e-7
    assert result.accepted_state.window_index == 1


def test_conforming_ale_qualifies_gcl_and_rolls_back_on_minimum_gap():
    finite_volume = _finite_volume()
    motion = phx.solver.MACALEGeometryPlan(
        finite_volume,
        lambda _time, points, _args: points,
        lambda _time, points, _args: jnp.zeros_like(points),
        mapping_id="cardio-identity-ale",
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    state = CardiovascularALEState(velocity, jnp.zeros(finite_volume.cell_shape))
    accepted_plan = CardiovascularALEPlan(
        motion,
        ALEMinimumGapRoute(
            lambda _geometry, gap: gap[0],
            lambda _start, _end, _start_time, _end_time, gap: gap[1],
            minimum_gap=0.05,
            route_id="accepted-clearance",
        ),
    ).prepare()
    accepted = accepted_plan.advance(
        state,
        0.0,
        0.01,
        jnp.asarray([[0.08, 0.09], [0.07, 0.08]]),
    )

    assert accepted.successful
    assert accepted.evidence.gcl_certified
    assert accepted.evidence.admissible
    assert accepted.evidence.maximum_gcl_residual < 1.0e-10
    assert accepted.evidence.gap.swept_certified

    rejected_plan = CardiovascularALEPlan(
        motion,
        ALEMinimumGapRoute(
            lambda _geometry, gap: gap[0],
            lambda _start, _end, _start_time, _end_time, gap: gap[1],
            minimum_gap=0.05,
            route_id="rejected-clearance",
        ),
    ).prepare()
    rejected = rejected_plan.advance(
        state,
        0.0,
        0.01,
        jnp.asarray([[0.08], [0.01]]),
    )

    assert not rejected.successful
    assert rejected.evidence.gap.swept_certified
    assert rejected.evidence.gap.minimum_gap == 0.01
    assert rejected.status == int(ALETransitionStatus.MINIMUM_GAP_FAILURE)
    assert all(
        jnp.array_equal(new, old)
        for new, old in zip(rejected.accepted_state.velocity, state.velocity, strict=True)
    )
    assert jnp.array_equal(rejected.accepted_state.pressure, state.pressure)


def _contact_residual():
    contact = phx.applications.contact
    collision = phx.discretization.contact
    query_space = phx.linalg.ArraySpace((1, 2), dtype=np.float64)
    surface_space = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    query = contact.prepare_point_contact_participant(
        query_space,
        jnp.asarray([[0.0, 0.1]]),
        vertex_ids=jnp.asarray([0]),
        body_ids=jnp.asarray([0]),
        physical_radius=jnp.asarray([0.1]),
    )
    surface_plan = collision.CollisionSurfacePlan(
        jnp.asarray([1, 2]),
        ambient_dimension=2,
        edges=jnp.asarray([[0, 1]], dtype=jnp.int32),
        body_ids=1,
        material_ids=0,
        static_mask=True,
        physical_radius=0.1,
    )
    surface = collision.LinearContactParticipant(
        collision.PreparedCollisionSurface(
            surface_plan,
            jnp.asarray([[-1.0, 0.0], [1.0, 0.0]]),
            collision.selection_collision_operator(surface_space, jnp.asarray([0, 1])),
        )
    )
    scene = collision.ContactParticipantScene((query, surface))
    search = collision.DenseContactSearchPlan(
        edge_vertex_capacity=4,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.3,
    )
    materials = contact.ContactMaterialPairTable.uniform(
        normal_stiffness=20.0,
        static_friction=0.0,
        dynamic_friction=0.0,
        restitution=0.0,
        adhesion_energy=0.0,
        thermal_conductance=0.0,
        electrical_conductance=0.0,
        wear_coefficient=0.0,
        hardness=1.0,
        roughness=0.0,
    )
    closure = contact.ContactClosurePlan(contact.CompliantNormalContactLaw(), materials)
    route_state = contact.ContactRouteState.empty(0, 1, closure.closure_id)
    rest = scene.positions((query_space.zeros(), surface_space.zeros()))
    return phx.solver.DeformableContactResidualPlan(
        scene,
        search,
        closure,
        route_state,
        rest,
        lambda q, v, _args: (q, v),
        lambda _q, _v, _args: (
            surface_space.zeros(),
            surface_space.zeros(),
        ),
        lambda query_force, _surface_force, _args: query_force,
        kinematics_id="leaflet-node",
        assembly_id="leaflet-contact-residual",
        activation_distance=0.3,
    )


def _cut_cell_route(maximum_leakage_proxy):
    finite_volume = _finite_volume()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    geometry = phx.discretization.MACDiffuseSDFGeometryPlan(
        operators,
        lambda points, _time, configuration: points[..., 0] - configuration[0, 0],
        lambda points, _time, _configuration: jnp.zeros_like(points),
        field_id=f"leaflet-sdf-{maximum_leakage_proxy}",
        interface_width=0.2,
    )
    masks = []
    for axis, layout in enumerate(finite_volume.face_layouts):
        mask = jnp.zeros(layout.shape, dtype=bool)
        if axis == 0:
            mask = mask.at[2, :].set(True)
        masks.append(mask)
    return CutCellLeafletRoute(
        geometry,
        lambda configuration, _velocity, _args: configuration,
        tuple(masks),
        maximum_leakage_proxy=maximum_leakage_proxy,
        maximum_gcl_residual=1.0e-10,
        maximum_small_cell_fraction=1.0,
    )


def _stationary_leaflet_step(
    _start, _step, configuration, velocity, _contact_residual, _args
):
    return LeafletStructuralAdvanceResult(
        configuration,
        velocity,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        jnp.asarray(1, dtype=jnp.int32),
    )


def test_cut_cell_leaflet_contact_reports_leakage_without_sealing_claim():
    workflow = LeafletContactWorkflowPlan(
        _contact_residual(),
        _cut_cell_route(1.0),
        _stationary_leaflet_step,
        maximum_penetration=0.02,
    ).prepare()
    configuration = jnp.asarray([[0.4, -0.01]])
    state = workflow.initialize(configuration, jnp.zeros_like(configuration))
    result = workflow.advance(state, 0.0, 0.01)

    assert result.successful
    assert result.evidence.contact_before.native_successful
    assert result.evidence.contact_candidate.native_successful
    assert result.evidence.contact_candidate.active_contact_count == 1
    assert result.evidence.contact_candidate.maximum_penetration == 0.01
    assert result.evidence.contact_candidate.force_balance_residual < 1.0e-10
    assert 0.0 < result.evidence.fluid.leakage_proxy <= 1.0
    assert not result.evidence.fluid.exact_sealing_certified
    assert not result.evidence.fluid.refinement_required


def test_leaflet_leakage_failure_rolls_back_structure_and_cut_cell_state_atomically():
    workflow = LeafletContactWorkflowPlan(
        _contact_residual(),
        _cut_cell_route(0.5),
        _stationary_leaflet_step,
        maximum_penetration=0.02,
    ).prepare()
    configuration = jnp.asarray([[0.4, -0.01]])
    velocity = jnp.zeros_like(configuration)
    state = workflow.initialize(configuration, velocity)
    result = workflow.advance(state, 0.0, 0.01)

    assert not result.successful
    assert result.status == int(LeafletTransitionStatus.LEAKAGE_FAILURE)
    assert jnp.array_equal(result.accepted_state.configuration, state.configuration)
    assert jnp.array_equal(result.accepted_state.velocity, state.velocity)
    assert jnp.array_equal(
        result.accepted_state.fluid_state.geometry.cell_fluid_fraction,
        state.fluid_state.geometry.cell_fluid_fraction,
    )


def test_leaflet_rejects_invalid_native_start_contact_even_if_candidate_is_valid():
    def move_into_contact_band(
        _start, _step, configuration, velocity, _contact_residual, _args
    ):
        candidate = configuration.at[0, 1].set(0.1)
        return LeafletStructuralAdvanceResult(
            candidate,
            velocity,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            jnp.asarray(1, dtype=jnp.int32),
        )

    workflow = LeafletContactWorkflowPlan(
        _contact_residual(),
        _cut_cell_route(1.0),
        move_into_contact_band,
        maximum_penetration=0.02,
    ).prepare()
    configuration = jnp.asarray([[0.4, 0.3]])
    state = workflow.initialize(configuration, jnp.zeros_like(configuration))
    result = workflow.advance(state, 0.0, 0.01)

    assert not result.evidence.contact_before.native_successful
    assert result.evidence.contact_candidate.native_successful
    assert not result.successful
    assert result.status == int(LeafletTransitionStatus.CONTACT_FAILURE)
    assert jnp.array_equal(result.accepted_state.configuration, configuration)
