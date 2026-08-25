#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._unstructured_overset import (
    PeriodicSlidingInterfacePlan,
)
from phydrax.solver._finite_volume_topology_events import (
    FiniteVolumeTopologyArtifactEvidence,
    FiniteVolumeTopologyEpoch,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventScheduler,
    TopologyEventKind,
    TopologyEventStatus,
)


def _plan(**kwargs):
    return PeriodicSlidingInterfacePlan(
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.0, 0.25, 0.75, 1.0]),
        1.0,
        interface_id="seam",
        **kwargs,
    )


def _epoch(name, parent=None):
    return FiniteVolumeTopologyEpoch(
        f"prepared-{name}",
        f"topology-{name}",
        f"geometry-{name}",
        parent_epoch_id=parent,
        topology_artifact_id=f"topology-artifact-{name}",
        metrics_artifact_id=f"metrics-artifact-{name}",
        operators_artifact_id=f"operators-artifact-{name}",
    )


def test_stationary_sliding_parity_and_coverage_evidence():
    plan = _plan()
    coupling = plan.coupling(0.0)
    values = jnp.asarray([[1.0], [3.0]])
    assert np.allclose(coupling.interpolate_left_to_right(values), [[1.0], [2.0], [3.0]])
    assert bool(coupling.coverage_passed)
    assert float(coupling.coverage_error) <= plan.coverage_tolerance
    assert coupling.shift_precision == plan.shift_precision
    assert coupling.evidence_id


def test_moving_shift_changes_routes_only_at_boundary():
    plan = _plan()
    stationary = plan.coupling(0.0)
    moved = plan.coupling(0.25)
    assert moved.normalized_shift == pytest.approx(0.25)
    assert moved.coupling_id != stationary.coupling_id
    assert moved.evidence_id != stationary.evidence_id
    assert np.array_equal(
        np.asarray(stationary.left_measures), np.asarray(moved.left_measures)
    )


def test_deterministic_shift_equivalence_and_precision_identity():
    plan = _plan(shift_precision=12)
    first = plan.coupling(0.25)
    equivalent = plan.coupling(1.25)
    assert first.coupling_id == equivalent.coupling_id
    assert first.evidence_id == equivalent.evidence_id
    assert plan.plan_id != _plan(shift_precision=11).plan_id


def test_equal_opposite_integrated_seam_flux_is_conservative():
    coupling = _plan().coupling(0.125)
    density = jnp.asarray([[2.0], [4.0]])
    left, right = coupling.integrated_seam_flux(density, 0.2)
    assert np.allclose(np.sum(left, axis=0) + np.sum(right, axis=0), 0.0)
    assert np.allclose(coupling.flux_conservation_defect(density * 0.2, right), 0.0)


def test_fixed_stage_path_is_jittable_and_map_is_frozen():
    coupling = _plan().coupling(0.125)
    values = jnp.asarray([[2.0], [4.0]])

    @jax.jit
    def apply_map(state):
        return coupling.interpolate_left_to_right(state)

    first = apply_map(values)
    second = apply_map(values)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert coupling.normalized_shift == pytest.approx(0.125)


def test_accepted_step_scheduler_creates_one_successor_event():
    initial = _epoch("initial")
    successor = _epoch("successor", parent=initial.epoch_id)
    scheduler = FiniteVolumeTopologyEventScheduler(
        FiniteVolumeTopologyEventJournal.allocate(initial, capacity=3)
    )
    request = FiniteVolumeTopologyEventRequest(
        TopologyEventKind.OVERSET_DONOR_REBUILD,
        initial.epoch_id,
        "sliding-plan",
        payload_id="coupling-evidence",
    )
    scheduler.submit(request, 1, 0.5, accepted=True)
    evidence = FiniteVolumeTopologyArtifactEvidence(
        passed=True,
        status=TopologyEventStatus.SUCCESS,
        coverage_error=0.0,
        conservation_defect=0.0,
        evidence_id="sliding-evidence",
    )
    result = scheduler.transact(
        accepted=True,
        source_content=None,
        artifact=evidence,
        candidate_epoch=successor,
        remap=evidence,
        metrics=evidence,
        evidence=evidence,
        status=TopologyEventStatus.SUCCESS,
    )
    assert result.committed
    assert len(result.events) == 1
    assert result.events[0].accepted_step == 1
    assert result.events[0].payload_id == "coupling-evidence"
    assert result.journal.current_epoch_id == successor.epoch_id


def test_failed_coverage_transaction_rolls_back_without_successor():
    initial = _epoch("initial")
    successor = _epoch("successor", parent=initial.epoch_id)
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyEventJournal,
    )

    scheduler = FiniteVolumeTopologyEventScheduler(
        FiniteVolumeTopologyEventJournal.allocate(initial, capacity=3)
    )
    request = FiniteVolumeTopologyEventRequest(
        TopologyEventKind.OVERSET_DONOR_REBUILD,
        initial.epoch_id,
        "sliding-plan",
        payload_id="bad-evidence",
    )
    scheduler.submit(request, 1, 0.5)
    evidence = FiniteVolumeTopologyArtifactEvidence(
        passed=True,
        status=TopologyEventStatus.SUCCESS,
        coverage_error=1.0,
        conservation_defect=0.0,
        evidence_id="bad-evidence",
    )
    result = scheduler.transact(
        accepted=True,
        source_content=None,
        artifact=evidence,
        candidate_epoch=successor,
        remap=evidence,
        metrics=evidence,
        evidence=evidence,
        status=TopologyEventStatus.SUCCESS,
    )
    assert not result.committed
    assert result.result_epoch is None
    assert result.journal.current_epoch_id == initial.epoch_id
    assert result.statuses == (TopologyEventStatus.FAILED_COVERAGE,)


def test_restart_replay_preserves_shift_and_event_identity():
    plan = _plan(shift_precision=13)
    coupling = plan.coupling(-0.375)
    replay = plan.coupling(coupling.normalized_shift)
    assert coupling.coupling_id == replay.coupling_id
    assert coupling.evidence_id == replay.evidence_id
    assert coupling.shift_precision == 13


def test_rejected_step_cannot_enqueue_sliding_event():
    initial = _epoch("initial")
    scheduler = FiniteVolumeTopologyEventScheduler(
        FiniteVolumeTopologyEventJournal.allocate(initial, capacity=2)
    )
    request = FiniteVolumeTopologyEventRequest(
        TopologyEventKind.OVERSET_DONOR_REBUILD,
        initial.epoch_id,
        "sliding-plan",
    )
    with pytest.raises(ValueError, match="accepted-step"):
        scheduler.submit(request, 0, 0.0, accepted=False)
    assert scheduler.pending_requests == ()


def test_stale_sliding_request_is_rejected_before_artifact_use():
    initial = _epoch("initial")
    stale = _epoch("stale")
    scheduler = FiniteVolumeTopologyEventScheduler(
        FiniteVolumeTopologyEventJournal.allocate(initial, capacity=2)
    )
    request = FiniteVolumeTopologyEventRequest(
        TopologyEventKind.OVERSET_DONOR_REBUILD,
        stale.epoch_id,
        "sliding-plan",
    )
    with pytest.raises(ValueError, match="stale"):
        scheduler.submit(request, 1, 0.5)


def _sliding_grid_plan(system):
    vertices = np.asarray([(i / 2.0, j / 2.0) for j in range(3) for i in range(3)])
    cells = np.asarray(
        (
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (3, 4, 7, 6),
            (4, 5, 8, 7),
        ),
        dtype=np.int32,
    )
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=cells,
        vertex_global_ids=np.arange(100, 109),
        cell_global_ids=np.arange(500, 504),
        component_names=system.component_names,
    )


def _certified_face_values(discretization, face_ids, face_cells):
    ids = np.asarray(face_ids, dtype=np.int32)
    cells = np.asarray(face_cells, dtype=np.int32)
    owners = np.asarray(discretization.owner_cells)[ids]
    neighbours = np.asarray(discretization.neighbour_cells)[ids]
    assert np.all((owners == cells) | (neighbours == cells))
    orientation = np.where(owners == cells, 1.0, -1.0)
    unit = (
        np.asarray(discretization.area_vectors)[ids]
        / np.asarray(discretization.face_measures)[ids, None]
    )
    quadrature_points = np.asarray(discretization.face_quadrature_points)[ids]
    normals = np.broadcast_to(
        orientation[:, None, None] * unit[:, None, :],
        quadrature_points.shape,
    )
    return {
        "receptor_face_ids": ids,
        "receptor_face_points": quadrature_points,
        "receptor_face_normals": normals,
        "receptor_face_measures": np.asarray(discretization.face_quadrature_weights)[ids],
        "receptor_face_cells": cells,
    }


def _moving_sliding_runtime(
    *,
    motion=None,
    consistency_policy=None,
    step_policy=None,
):
    system = phx.equations.EulerSystem(2)
    mesh_plan = _sliding_grid_plan(system)
    discretization = mesh_plan.prepare()
    face_artifact = _certified_face_values(
        discretization,
        np.asarray((9, 7, 10), dtype=np.int32),
        np.asarray((2, 2, 3), dtype=np.int32),
    )
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((2, 3), dtype=np.int32),
        np.asarray((0, 1, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((0.25, 0.25)),
        epoch_id="moving-sliding-epoch",
        **face_artifact,
    )

    def translation(time, vertices, args):
        del args
        return vertices.at[:, 0].add(0.2 * time)

    motion_plan = phx.discretization.FixedConnectivityMotionPlan(
        mesh_plan,
        translation if motion is None else motion,
        mapping_id="moving-sliding-routes",
        consistency_policy=consistency_policy,
    )
    coupling_plan = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        motion=motion_plan,
        overset=overset,
        sliding=_plan(),
        topology_event_capacity=8,
        topology_event_policy="accepted_step",
    )
    wall_speed = 0.2 if motion is None else 0.0

    def wall_velocity(time, points, normals, args):
        del time, points, normals, args
        return jnp.asarray((wall_speed, 0.0))

    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.MovingSlipWallBoundary(
                wall_velocity,
                wall_velocity_provider_id=f"moving-sliding-routes:{name}",
            )
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "moving-sliding-routes",
        "state",
        system,
        boundaries,
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        coupling=coupling_plan,
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        step_policy,
    )
    return system, discretization, overset, coupling_plan, runtime


def _nonuniform_state(system, discretization):
    primitive = jnp.asarray(
        (
            (1.0, 0.25, 0.0, 1.0),
            (2.0, -0.15, 0.0, 1.2),
            (1.4, 0.0, 0.0, 1.1),
            (0.8, 0.0, 0.0, 0.9),
        )
    )
    assert primitive.shape == discretization.state_shape
    return system.primitive_to_conserved(primitive)


def _overset_budget(block, cell_count):
    scattered = np.zeros(
        (cell_count, np.asarray(block.flux_integral).shape[-1]),
        dtype=np.asarray(block.flux_integral).dtype,
    )
    np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(block.flux_integral))
    np.add.at(
        scattered,
        np.asarray(block.neighbour_cells),
        np.asarray(block.flux_integral),
    )
    return scattered


def _overset_rate_budget(block, cell_count):
    scattered = np.zeros(
        (cell_count, np.asarray(block.flux_rate).shape[-1]),
        dtype=np.asarray(block.flux_rate).dtype,
    )
    np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(block.flux_rate))
    np.add.at(
        scattered,
        np.asarray(block.neighbour_cells),
        np.asarray(block.flux_rate),
    )
    return scattered


def test_prepared_coupling_accepts_only_current_plan_sliding_map():
    _, discretization, _, coupling_plan, _ = _moving_sliding_runtime()
    assert coupling_plan.sliding is not None
    supplied = coupling_plan.sliding.coupling(0.2)
    prepared = coupling_plan.prepare(
        discretization,
        sliding_coupling=supplied,
    )
    assert prepared.sliding_coupling is supplied
    assert prepared.sliding_coupling.normalized_shift == pytest.approx(0.2)

    foreign = _plan(shift_precision=13).coupling(0.2)
    with pytest.raises(ValueError, match="stale|belong"):
        coupling_plan.prepare(discretization, sliding_coupling=foreign)


def test_moved_overset_correction_uses_stage_faces_and_grid_velocity():
    system, discretization, overset, coupling_plan, runtime = _moving_sliding_runtime()
    initial = runtime.initialize_state(
        _nonuniform_state(system, discretization),
        0.0,
        1.0e-3,
    )
    result = runtime.advance(initial, {"sliding_shift": 0.2})

    assert bool(np.asarray(result.accepted))
    assert result.ale is not None
    moved = result.ale.geometry.stage_2.face_blocks[0]
    ids = np.asarray(overset.receptor_face_ids)
    assert not np.allclose(
        np.asarray(moved.quadrature_points)[ids],
        np.asarray(overset.receptor_face_points),
    )
    assert np.any(np.abs(np.asarray(moved.quadrature_grid_normal_velocity)[ids]) > 0.0)
    assert coupling_plan.sliding is not None
    shifted_dynamics = runtime.dynamics.with_sliding_coupling(
        coupling_plan.sliding.coupling(0.2)
    )
    moved_block, _, _ = runtime.dynamics._overset_correction(
        initial.cell_average(),
        result.ale.geometry.stage_2,
        None,
    )
    zero_grid_metrics = eqx.tree_at(
        lambda metrics: metrics.face_blocks[0].quadrature_grid_normal_velocity,
        result.ale.geometry.stage_2,
        jnp.zeros_like(moved.quadrature_grid_normal_velocity),
    )
    zero_grid_block, _, _ = runtime.dynamics._overset_correction(
        initial.cell_average(),
        zero_grid_metrics,
        None,
    )
    assert moved_block is not None and zero_grid_block is not None
    assert not np.allclose(
        np.asarray(moved_block.flux_rate),
        np.asarray(zero_grid_block.flux_rate),
    )
    shifted_block, shifted_speed, shifted_measures = shifted_dynamics._overset_correction(
        initial.cell_average(),
        result.ale.geometry.stage_2,
        None,
    )
    assert shifted_block is not None
    np.testing.assert_array_equal(
        np.unique(np.asarray(shifted_block.owner_cells)),
        np.asarray((0, 1)),
    )
    np.testing.assert_array_equal(
        np.unique(np.asarray(shifted_block.neighbour_cells)),
        np.asarray((2, 3)),
    )
    moved_budget = _overset_rate_budget(moved_block, discretization.cell_count)
    shifted_budget = _overset_rate_budget(shifted_block, discretization.cell_count)
    assert not np.allclose(moved_budget, shifted_budget)
    np.testing.assert_allclose(moved_budget.sum(axis=0), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(shifted_budget.sum(axis=0), 0.0, atol=1.0e-12)
    assert np.all(np.asarray(shifted_speed) >= 0.0)
    assert np.all(np.asarray(shifted_measures) > 0.0)

    def shifted_mass_objective(value):
        block, _, _ = shifted_dynamics._overset_correction(
            value,
            result.ale.geometry.stage_2,
            None,
        )
        assert block is not None
        return jnp.sum(block.flux_rate[:, 0] ** 2)

    average = initial.cell_average()
    objective = eqx.filter_jit(shifted_mass_objective)(average)
    gradient = jax.grad(shifted_mass_objective)(average)
    assert float(np.asarray(objective)) > 0.0
    assert bool(np.asarray(jnp.all(jnp.isfinite(gradient))))
    assert bool(np.asarray(jnp.any(jnp.abs(gradient) > 0.0)))
    correction = result.ale.stage_rate_ledgers[1].blocks[-1]
    assert correction.block_kind == "overset-correction"


def test_accepted_shift_changes_overset_ledger_and_successor_runtime():
    system, discretization, _, _, runtime = _moving_sliding_runtime()
    initial = runtime.initialize_state(
        _nonuniform_state(system, discretization),
        0.0,
        2.0e-4,
    )
    first = runtime.advance(initial, {"sliding_shift": 0.2})

    assert bool(np.asarray(first.accepted))
    assert first.successor_runtime is not None
    successor = first.successor_runtime
    assert successor.sliding_initial_coupling is not None
    assert successor.sliding_initial_coupling.normalized_shift == pytest.approx(0.2)
    assert (
        successor.sliding_initial_coupling.coupling_id
        == first.runtime_state.sliding_coupling_id
    )
    assert (
        first.runtime_state.content_state.topology_epoch_id
        == first.runtime_state.topology_journal.current_epoch_id
        == successor.topology_epoch_id
    )
    with pytest.raises(ValueError, match="successor runtime|journal"):
        runtime.advance(first.runtime_state, {"sliding_shift": 0.2})

    second = successor.advance(first.runtime_state, {"sliding_shift": 0.2})
    assert bool(np.asarray(second.accepted))
    first_block = first.accepted_flux_integrals.blocks[-1]
    second_block = second.accepted_flux_integrals.blocks[-1]
    first_budget = _overset_budget(first_block, discretization.cell_count)
    second_budget = _overset_budget(second_block, discretization.cell_count)
    assert not np.allclose(first_budget, second_budget)
    np.testing.assert_allclose(first_budget.sum(axis=0), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(second_budget.sum(axis=0), 0.0, atol=1.0e-12)


def test_sliding_map_is_frozen_across_ale_retries():
    consistency = phx.discretization.finite_volume.ALEGeometryConsistencyPolicy(
        absolute_tolerance=3.0e-3,
        relative_tolerance=0.0,
        reduction_safety_factor=0.8,
        minimum_reduction_factor=0.1,
    )

    def nonlinear_deformation(time, vertices, args):
        del args
        return vertices.at[4, 0].add(0.8 * time**2)

    policy = phx.solver.FiniteVolumeStepPolicy(
        maximum_retries=3,
        reduction_factor=0.5,
    )
    system, discretization, _, _, runtime = _moving_sliding_runtime(
        motion=nonlinear_deformation,
        consistency_policy=consistency,
        step_policy=policy,
    )
    initial = runtime.initialize_state(
        _nonuniform_state(system, discretization),
        0.0,
        0.2,
    )
    shift_calls = []

    def accepted_shift(time, args):
        del args
        shift_calls.append(float(np.asarray(time)))
        return 0.2

    result = runtime.advance(initial, {"sliding_shift": accepted_shift})
    assert bool(np.asarray(result.accepted))
    assert int(np.asarray(result.retries)) > 0
    assert len(shift_calls) == 1
    frozen_block_id = runtime.dynamics.overset_rate_block_template.block_id
    assert result.ale is not None
    for ledger in result.ale.stage_rate_ledgers:
        assert ledger.blocks[-1].block_id == frozen_block_id
    assert result.successor_runtime is not None
    assert (
        result.successor_runtime.dynamics.overset_rate_block_template.block_id
        != frozen_block_id
    )


def test_receptor_face_ids_are_validated_and_missing_artifacts_fail_closed():
    system = phx.equations.EulerSystem(2)
    mesh_plan = _sliding_grid_plan(system)
    discretization = mesh_plan.prepare()
    artifact = _certified_face_values(
        discretization,
        np.asarray((9, 7, 10), dtype=np.int32),
        np.asarray((2, 2, 3), dtype=np.int32),
    )
    common = (
        discretization,
        discretization,
        np.asarray((2, 3), dtype=np.int32),
        np.asarray((0, 1, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((0.25, 0.25)),
    )
    stale_artifact = dict(artifact)
    stale_artifact["receptor_face_ids"] = np.asarray((8, 7, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="stale|incident"):
        phx.discretization.UnstructuredOversetPlan(*common, **stale_artifact)

    missing_ids = dict(artifact)
    del missing_ids["receptor_face_ids"]
    with pytest.raises(ValueError, match="require IDs"):
        phx.discretization.UnstructuredOversetPlan(*common, **missing_ids)

    uncertified = phx.discretization.UnstructuredOversetPlan(*common)

    def translation(time, vertices, args):
        del args
        return vertices.at[:, 0].add(0.2 * time)

    motion = phx.discretization.FixedConnectivityMotionPlan(
        mesh_plan,
        translation,
        mapping_id="uncertified-sliding",
    )
    plan = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        motion=motion,
        overset=uncertified,
        sliding=_plan(),
        topology_event_capacity=2,
        topology_event_policy="accepted_step",
    )
    with pytest.raises(ValueError, match="fully certified"):
        plan.prepare(discretization)
