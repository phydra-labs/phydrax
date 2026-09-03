#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._sharp_measures import exact_sharp_geometry
from phydrax.applications.incompressible_flow._immersed_admission import (
    ImmersedRuntimeAdmissionFailure,
    ImmersedRuntimeAdmissionPlan,
    ImmersedRuntimeEvidence,
    ImmersedRuntimePreflightEvidence,
)
from phydrax.applications.incompressible_flow._immersed_profile import (
    ImmersedDNSQualificationProfile,
)
from phydrax.applications.incompressible_flow._immersed_qualification import (
    ImmersedReferenceCampaignPlan,
    ImmersedReferenceCaseEvidence,
)
from phydrax.applications.incompressible_flow._immersed_support import (
    ImmersedBodyRegimePlan,
    ImmersedNearGapRegime,
)
from phydrax.discretization.finite_volume._distributed_marker_transfer import (
    DistributedMACMarkerTransfer,
    DistributedMarkerOwnershipPlan,
)
from phydrax.discretization.particle._resolved_lubrication import (
    ResolvedLubricationCorrectionPlan,
)
from phydrax.solver._mac_sharp_interface import MACSharpInterfaceProjectionPlan


def _flow(count=6):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    pressure = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=boundaries, solve_method="transform"
    )
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01), momentum, pressure
    )
    return finite_volume, operators, boundaries, dynamics


def _marker_owner(operators, boundaries):
    positions = jnp.asarray(((0.35, 0.35), (0.65, 0.35), (0.65, 0.65), (0.35, 0.65)))
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), positions, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    owner = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, transfer, boundaries=boundaries, tolerance=1.0e-8
    )
    return markers, transfer, owner


def _rigid_owner(finite_volume, operators, dynamics):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((0,)), jnp.asarray((1.0,)), ambient_dimension=2
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray((0,)), jnp.asarray((0.1,))
    ).prepare(particles)
    offsets = jnp.asarray(((-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)))
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), offsets, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    rigid_map = phx.discretization.RigidMarkerMapPlan(
        markers, bodies, jnp.zeros((4,), dtype=jnp.int32)
    ).prepare()
    projection = phx.solver.MACRigidImmersedProjectionPlan(
        dynamics,
        rigid_map,
        transfer,
        constraint_length=1.0 / finite_volume.cell_shape[0],
        tolerance=1.0e-8,
    )
    base = phx.solver.MACRigidImmersedEulerMethod(dynamics, projection, 1.0e-3)
    backward = phx.solver.MACRigidImmersedBackwardEulerMethod(
        base, maximum_iterations=1, tolerance=1.0e-8
    )
    hard_contact = phx.discretization.HardContactRoutePlan(
        jnp.asarray((0,)),
        jnp.asarray((-1,)),
        jnp.asarray((17,)),
        position_stabilization=0.0,
    ).prepare(bodies)
    contact = phx.solver.MACRigidImmersedContactMethod(
        backward, hard_contact, maximum_iterations=1, tolerance=1.0e-8
    )
    return markers, projection, contact


def _deformable_owner(operators, boundaries, dynamics):
    positions = jnp.asarray(((0.35, 0.5), (0.65, 0.5)))
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(2), positions, jnp.asarray((0.5, 0.5))
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, transfer, boundaries=boundaries, tolerance=1.0e-8
    )
    configuration_space = phx.linalg.ArraySpace((4,))
    marker_map = phx.discretization.FiniteElementImmersedMarkerMapPlan(
        markers, configuration_space, jnp.eye(4)
    ).prepare()
    structure = phx.dynamics.SecondOrderDifferentialSystem(
        lambda _time, _q, _v, acceleration, _args: acceleration,
        state_shape=(4,),
        system_id="immersed-qualification-zero-mass",
    )
    method = phx.solver.MACDeformableImmersedBackwardEulerMethod(
        dynamics,
        projection,
        marker_map,
        structure,
        lambda q, _args: 0.5 * jnp.sum(q * 0.0),
        1.0e-3,
        energy_id="immersed-qualification-zero-energy",
        structural_contact_residual=lambda q, _v, _args: jnp.zeros_like(q),
    )
    return markers, method


def _sharp_owner(finite_volume, operators, boundaries):
    pairing_id = canonical_fingerprint(
        {
            "pressure": operators.pressure_space.space_id,
            "velocity": operators.velocity_space.space_id,
        }
    )
    geometry = exact_sharp_geometry(
        finite_volume.cell_volumes,
        finite_volume.cell_volumes,
        finite_volume.face_measures,
        finite_volume.face_measures,
        measure_evidence_id="immersed-qualification-exact-measures",
        source_id="immersed-qualification-sharp-source",
        source_fidelity="exact-polytope",
        support_id=finite_volume.support.support_id,
        cell_field_id=finite_volume.cell_space.field_space_id,
        face_field_ids=tuple(space.field_space_id for space in finite_volume.face_spaces),
        operator_id=operators.prepared_id,
        pairing_id=pairing_id,
    )
    return geometry, MACSharpInterfaceProjectionPlan(operators, boundaries, geometry)


@pytest.fixture(scope="module")
def regimes():
    finite_volume, operators, boundaries, dynamics = _flow()
    markers, transfer, marker_owner = _marker_owner(operators, boundaries)
    marker = ImmersedBodyRegimePlan(
        marker_owner,
        marker_set_id=markers.prepared_id,
        geometry_id="manufactured-marker-geometry",
        route_id="manufactured-marker-route",
        topology_epoch_id="marker-topology-0",
        geometry_epoch=0,
        moving=False,
    )
    rigid_markers, rigid_projection, rigid_contact = _rigid_owner(
        finite_volume, operators, dynamics
    )
    rigid = ImmersedBodyRegimePlan(
        rigid_projection,
        marker_set_id=rigid_markers.prepared_id,
        geometry_id="rigid-sphere-geometry",
        route_id="rigid-sphere-route",
        topology_epoch_id="rigid-topology-0",
        motion_epoch_id="rigid-motion-4",
        geometry_epoch=4,
        moving=True,
    )
    contact = ImmersedBodyRegimePlan(
        rigid_contact,
        marker_set_id=rigid_markers.prepared_id,
        geometry_id="rigid-contact-geometry",
        route_id="rigid-contact-route",
        topology_epoch_id="contact-topology-0",
        motion_epoch_id="contact-motion-3",
        geometry_epoch=3,
        moving=True,
        lubrication=ResolvedLubricationCorrectionPlan(1.0, 0.1, 0.01),
    )
    deformable_markers, deformable_owner = _deformable_owner(
        operators, boundaries, dynamics
    )
    deformable = ImmersedBodyRegimePlan(
        deformable_owner,
        marker_set_id=deformable_markers.prepared_id,
        geometry_id="flexible-marker-geometry",
        route_id="flexible-marker-route",
        topology_epoch_id="flexible-topology-0",
        motion_epoch_id="flexible-motion-1",
        geometry_epoch=1,
        moving=True,
    )
    sharp_geometry, sharp_owner = _sharp_owner(finite_volume, operators, boundaries)
    sharp = ImmersedBodyRegimePlan(
        sharp_owner,
        marker_set_id="not-applicable-sharp-interface",
        geometry_id=sharp_geometry.realization_id,
        route_id="sharp-fixed-route",
        topology_epoch_id="sharp-topology-0",
        geometry_epoch=0,
        moving=False,
    )
    ownership = DistributedMarkerOwnershipPlan(
        markers.plan.marker_ids,
        jnp.zeros((markers.capacity,), dtype=jnp.int32),
        jnp.zeros((markers.capacity, 1), dtype=jnp.int32),
        jnp.ones((markers.capacity, 1), dtype=bool),
        rank_count=1,
    )
    distributed_transfer = DistributedMACMarkerTransfer(transfer, ownership, 0)
    distributed = ImmersedBodyRegimePlan(
        marker_owner,
        marker_set_id=markers.prepared_id,
        geometry_id="distributed-marker-geometry",
        route_id="distributed-marker-route",
        topology_epoch_id="distributed-topology-0",
        geometry_epoch=0,
        moving=False,
        distributed_transfer=distributed_transfer,
    )
    return {
        "marker": marker,
        "rigid": rigid,
        "contact": contact,
        "deformable": deformable,
        "sharp": sharp,
        "distributed": distributed,
    }


def _preflight(regime, **updates):
    values = {
        "owner_plan_id": regime.owner_plan_id,
        "support_tuple_id": regime.support_tuple.support_tuple_id,
        "marker_numerical_rank": regime.marker_constraint_count,
        "marker_condition": 2.0,
        "observed_resource_bytes": regime.estimated_marker_resource_bytes,
        "rank_certified": True,
        "campaign_qualified": True,
    }
    values.update(updates)
    return ImmersedRuntimePreflightEvidence(
        values["owner_plan_id"],
        values["support_tuple_id"],
        values["marker_numerical_rank"],
        values["marker_condition"],
        values["observed_resource_bytes"],
        values["rank_certified"],
        values["campaign_qualified"],
        evidence_ids=("campaign-report", "rank-resource-probe"),
    )


def _runtime(regime, **updates):
    values = {
        "owner_plan_id": regime.owner_plan_id,
        "support_tuple_id": regime.support_tuple.support_tuple_id,
        "marker_set_id": regime.marker_set_id,
        "geometry_id": regime.geometry_id,
        "route_id": regime.route_id,
        "topology_epoch_id": regime.topology_epoch_id,
        "motion_epoch_id": regime.motion_epoch_id,
        "geometry_epoch": regime.geometry_epoch,
        "support_truncated": False,
        "topology_changed": False,
        "geometry_refresh_required": False,
        "sharp_certificate_valid": True,
        "differentiation_routes_frozen": True,
        "gap": None,
        "distributed": None,
        "load_record": None,
    }
    values.update(updates)
    return ImmersedRuntimeEvidence(
        **values,
        evidence_ids=("accepted-owner-step",),
    )


def test_profile_declares_separate_unsigned_candidate_support_and_campaigns():
    profile = ImmersedDNSQualificationProfile()
    regimes = {dict(value.attributes)["regime"] for value in profile.support_tuples}

    assert regimes == {
        "prescribed-marker",
        "free-rigid-marker",
        "fixed-topology-sharp",
        "deformable-contact",
        "lbm-body",
        "resolved-cfd-dem",
    }
    assert set(profile.required_reference_cases) == {
        "manufactured-loads",
        "fixed-cylinder",
        "moving-cylinder",
        "fixed-sphere",
        "moving-sphere",
        "added-mass",
        "free-settling",
        "flexible-contact-state",
        "sharp-certificate",
    }
    assert not profile.released
    assert not profile.capability_profile.released
    assert profile.capability_profile.release_evidence == ()
    assert len(profile.qualification_matrix.predicates) == 9


def test_load_provenance_distinguishes_unavailable_from_available_zero(regimes):
    regime = regimes["marker"]
    plan = regime.load_plan(jnp.asarray((41,)), 2, reference_point_id="body-41-centre")
    velocity = jnp.asarray(((0.25, 0.0),))
    angular = jnp.zeros((1, 1))
    available_zero = plan.record(
        0.0,
        0.5,
        velocity,
        angular,
        interval_id="load-window-7",
        pressure_force=jnp.zeros((1, 2)),
        pressure_torque=jnp.zeros((1, 1)),
        marker_force=jnp.asarray(((2.0, 0.0),)),
        marker_torque=jnp.zeros((1, 1)),
    )
    unavailable = plan.record(
        0.0,
        0.5,
        velocity,
        angular,
        interval_id="load-window-8",
        marker_force=jnp.asarray(((2.0, 0.0),)),
        marker_torque=jnp.zeros((1, 1)),
    )

    assert available_zero.pressure_available[0]
    assert not unavailable.pressure_available[0]
    assert not available_zero.viscous_available[0]
    assert available_zero.marker_available[0]
    assert available_zero.marker_set_id == regime.marker_set_id
    assert available_zero.geometry_id == regime.geometry_id
    assert available_zero.route_id == regime.route_id
    assert available_zero.topology_epoch_id == regime.topology_epoch_id
    assert available_zero.reference_point_id == "body-41-centre"
    assert available_zero.interval_id == "load-window-7"
    assert available_zero.successful
    assert jnp.allclose(available_zero.force, jnp.asarray(((2.0, 0.0),)))
    assert jnp.allclose(available_zero.work, jnp.asarray((0.25,)))


def test_reference_campaign_covers_complete_body_portfolio(regimes):
    profile = ImmersedDNSQualificationProfile()
    bound = (
        regimes["marker"],
        regimes["rigid"],
        regimes["deformable"],
        regimes["sharp"],
    )
    campaign = ImmersedReferenceCampaignPlan(profile, bound)
    by_case = {
        "manufactured-loads": regimes["marker"],
        "fixed-cylinder": regimes["marker"],
        "moving-cylinder": regimes["marker"],
        "fixed-sphere": regimes["rigid"],
        "moving-sphere": regimes["rigid"],
        "added-mass": regimes["rigid"],
        "free-settling": regimes["rigid"],
        "flexible-contact-state": regimes["deformable"],
        "sharp-certificate": regimes["sharp"],
    }
    manufactured_record = (
        regimes["marker"]
        .load_plan(
            jnp.asarray((0,)),
            2,
            reference_point_id="manufactured-body-origin",
        )
        .record(
            0.0,
            0.1,
            jnp.zeros((1, 2)),
            jnp.zeros((1, 1)),
            interval_id="manufactured-load-window",
            marker_force=jnp.asarray(((1.0, -0.5),)),
            marker_torque=jnp.asarray(((0.25,),)),
        )
    )
    evidence = tuple(
        ImmersedReferenceCaseEvidence.manufactured_loads(
            regime,
            manufactured_record,
            jnp.asarray(((1.0, -0.5),)),
            jnp.asarray(((0.25,),)),
            raw_artifact_ids=(f"artifact-{case}",),
        )
        if case == "manufactured-loads"
        else ImmersedReferenceCaseEvidence.sharp_certificate(
            regime,
            regime.owner.geometry,
            raw_artifact_ids=(f"artifact-{case}",),
        )
        if case == "sharp-certificate"
        else ImmersedReferenceCaseEvidence(
            case,
            regime.plan_id,
            regime.support_tuple.support_tuple_id,
            0.0,
            True,
            subject_ids=(regime.owner_plan_id,),
            raw_artifact_ids=(f"artifact-{case}",),
        )
        for case, regime in by_case.items()
    )
    result = campaign.evaluate(evidence)

    assert result.case_ids == profile.required_reference_cases
    assert result.successful
    assert jnp.all(result.passed)


def test_marker_rank_condition_resource_and_runtime_epoch_refusal(regimes):
    profile = ImmersedDNSQualificationProfile()
    regime = regimes["marker"]
    plan = ImmersedRuntimeAdmissionPlan(
        profile,
        regime,
        maximum_resource_bytes=regime.estimated_marker_resource_bytes + 1,
        derivative_mode="jvp",
    )
    accepted = plan.admit(_preflight(regime), _runtime(regime))
    rank_failed = plan.prepare(
        _preflight(regime, marker_numerical_rank=regime.marker_constraint_count - 1)
    )
    condition_failed = plan.prepare(_preflight(regime, marker_condition=1.0e30))
    resource_failed = ImmersedRuntimeAdmissionPlan(
        profile, regime, maximum_resource_bytes=1
    ).prepare(_preflight(regime))
    truncated = plan.admit(_preflight(regime), _runtime(regime, support_truncated=True))
    wrong_epoch = plan.admit(
        _preflight(regime), _runtime(regime, topology_epoch_id="marker-topology-1")
    )

    assert accepted.admitted
    assert not rank_failed.prepared
    assert int(rank_failed.status) & int(
        ImmersedRuntimeAdmissionFailure.MARKER_RANK_FAILED
    )
    assert not condition_failed.prepared
    assert int(condition_failed.status) & int(
        ImmersedRuntimeAdmissionFailure.MARKER_CONDITION_FAILED
    )
    assert not resource_failed.prepared
    assert int(resource_failed.status) & int(
        ImmersedRuntimeAdmissionFailure.RESOURCE_BUDGET_EXCEEDED
    )
    assert not truncated.admitted
    assert int(truncated.status) & int(ImmersedRuntimeAdmissionFailure.SUPPORT_TRUNCATED)
    assert not wrong_epoch.admitted
    assert int(wrong_epoch.status) & int(
        ImmersedRuntimeAdmissionFailure.TOPOLOGY_EPOCH_MISMATCH
    )


def test_moving_body_epoch_and_near_gap_crossover_are_fail_closed(regimes):
    profile = ImmersedDNSQualificationProfile()
    rigid = regimes["rigid"]
    rigid_plan = ImmersedRuntimeAdmissionPlan(
        profile, rigid, maximum_resource_bytes=10**9
    )
    stale_motion = rigid_plan.admit(
        _preflight(rigid), _runtime(rigid, motion_epoch_id="rigid-motion-3")
    )
    contact = regimes["contact"]
    crossover = contact.classify_gap(jnp.asarray((0.2, 0.05, 0.005)))
    contact_plan = ImmersedRuntimeAdmissionPlan(
        profile, contact, maximum_resource_bytes=10**9
    )
    admitted = contact_plan.admit(
        _preflight(contact), _runtime(contact, gap=jnp.asarray((0.2, 0.05, 0.005)))
    )

    assert not stale_motion.admitted
    assert int(stale_motion.status) & int(
        ImmersedRuntimeAdmissionFailure.MOTION_EPOCH_MISMATCH
    )
    assert jnp.array_equal(
        crossover.regime,
        jnp.asarray(
            (
                int(ImmersedNearGapRegime.RESOLVED_GRID),
                int(ImmersedNearGapRegime.LUBRICATION),
                int(ImmersedNearGapRegime.CONTACT),
            ),
            dtype=jnp.int32,
        ),
    )
    assert crossover.admissible
    assert admitted.admitted


def test_distributed_owner_work_force_reduction_and_sharp_topology_scope(regimes):
    profile = ImmersedDNSQualificationProfile()
    distributed = regimes["distributed"]
    transfer = distributed.distributed_transfer
    assert transfer is not None
    relation = transfer.local.relation(transfer.local.markers.reference_position)
    velocity = tuple(
        jnp.zeros(layout.shape)
        for layout in transfer.local.operators.discretization.face_layouts
    )
    diagnostics = transfer.diagnostics(
        relation,
        velocity,
        jnp.zeros_like(transfer.local.markers.reference_position),
        lambda _kind, value: value,
    )
    distributed_plan = ImmersedRuntimeAdmissionPlan(
        profile, distributed, maximum_resource_bytes=10**9
    )
    distributed_result = distributed_plan.admit(
        _preflight(distributed),
        _runtime(
            distributed,
            support_truncated=relation.support_truncated,
            distributed=diagnostics,
        ),
    )

    sharp = regimes["sharp"]
    sharp_plan = ImmersedRuntimeAdmissionPlan(
        profile,
        sharp,
        maximum_resource_bytes=10**9,
        derivative_mode="vjp",
    )
    accepted = sharp_plan.admit(_preflight(sharp), _runtime(sharp))
    topology_changed = sharp_plan.admit(
        _preflight(sharp), _runtime(sharp, topology_changed=True)
    )
    epoch_changed = sharp_plan.admit(_preflight(sharp), _runtime(sharp, geometry_epoch=1))

    assert diagnostics.successful
    assert jnp.allclose(diagnostics.global_force_residual, 0.0)
    assert jnp.allclose(diagnostics.global_work_residual, 0.0)
    assert distributed_result.admitted
    assert accepted.admitted
    assert not topology_changed.admitted
    assert int(topology_changed.status) & int(
        ImmersedRuntimeAdmissionFailure.TOPOLOGY_CHANGE
    )
    assert not epoch_changed.admitted
    assert int(epoch_changed.status) & int(
        ImmersedRuntimeAdmissionFailure.GEOMETRY_EPOCH_MISMATCH
    )
