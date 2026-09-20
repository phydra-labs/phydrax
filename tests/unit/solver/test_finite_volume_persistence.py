#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy
import json
import zipfile
from importlib.util import find_spec
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint


def _prepared_runtime(cells=16):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "persistent-euler",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics, phx.discretization.FluxPositivityPlan()
    )
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.1, 1.0]), (cells, 3))
    state = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.2,
        0.001,
        accepted_step=7,
        controller_state=jnp.asarray([0.3, 0.4]),
        integrator_state=jnp.asarray([1.2]),
        output_cursor=4,
    )
    return runtime, discretization, state


def _prepared_block_runtime():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2),
            phx.discretization.BlockLevelPlan(1, (2,), 8),
        ),
    )
    prepared = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    initial = prepared.initial_topology()
    tags = jnp.zeros((2, 4), dtype="bool").at[0, 1].set(True)
    compiled = prepared.compile_topology(initial, (tags,))
    assert compiled.status.successful
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="persistent-block-advection",
    )
    boundary = phx.discretization.ExtrapolationBoundary()
    boundaries = phx.discretization.FiniteVolumeBoundarySet(
        ("x",),
        (phx.discretization.FiniteVolumeBoundaryPair(boundary, boundary),),
    )
    finite_volume = phx.discretization.BlockAMRFiniteVolumePlan(
        prepared,
        system,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        boundaries,
    )
    runtime = phx.solver.BlockAMRRuntimePlan(finite_volume).prepare(compiled.topology)
    levels = []
    for level, (level_plan, metadata) in enumerate(
        zip(compiled.topology.plan.levels, compiled.topology.levels, strict=True)
    ):
        values = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape, 1),
            dtype=jnp.float64,
        )
        active = metadata.active.reshape(
            (level_plan.maximum_blocks,) + (1,) * (values.ndim - 1)
        )
        values = jnp.where(active, jnp.asarray(level + 1.0), values)
        levels.append(phx.discretization.BlockLevelState(level_plan, metadata, values))
    hierarchy_state = phx.discretization.BlockHierarchyState(
        compiled.topology, tuple(levels)
    )
    state = runtime.initial_state(
        hierarchy_state,
        time=0.25,
        accepted_step=3,
        level_accepted_steps=jnp.asarray((3, 6), dtype=jnp.int32),
    )
    return prepared, compiled, runtime, state


def _assert_runtime_state_exact(actual, expected):
    np.testing.assert_array_equal(
        actual.content_state.conservative_content,
        expected.content_state.conservative_content,
    )
    np.testing.assert_array_equal(
        actual.content_state.effective_cell_volumes,
        expected.content_state.effective_cell_volumes,
    )
    np.testing.assert_array_equal(
        actual.content_state.active_cell_mask,
        expected.content_state.active_cell_mask,
    )
    np.testing.assert_array_equal(actual.cell_average(), expected.cell_average())
    np.testing.assert_array_equal(actual.time, expected.time)
    np.testing.assert_array_equal(actual.accepted_step, expected.accepted_step)
    np.testing.assert_array_equal(actual.step_size, expected.step_size)
    np.testing.assert_array_equal(actual.last_status, expected.last_status)
    np.testing.assert_array_equal(actual.controller_state, expected.controller_state)
    np.testing.assert_array_equal(actual.integrator_state, expected.integrator_state)
    np.testing.assert_array_equal(actual.output_cursor, expected.output_cursor)
    assert (
        actual.content_state.topology_epoch_id == expected.content_state.topology_epoch_id
    )
    assert (
        actual.content_state.geometry_layout_id
        == expected.content_state.geometry_layout_id
    )
    assert (
        actual.content_state.evidence_policy_id
        == expected.content_state.evidence_policy_id
    )
    np.testing.assert_array_equal(
        actual.content_state.geometry_version,
        expected.content_state.geometry_version,
    )
    np.testing.assert_array_equal(
        actual.content_state.evidence_version,
        expected.content_state.evidence_version,
    )
    assert (
        actual.topology_journal.to_archive_record()
        == expected.topology_journal.to_archive_record()
    )
    assert actual.topology_journal.journal_id == expected.topology_journal.journal_id
    for name, actual_array in actual.topology_journal.archive_arrays().items():
        np.testing.assert_array_equal(
            actual_array,
            expected.topology_journal.archive_arrays()[name],
        )
    assert actual.sliding_coupling_id == expected.sliding_coupling_id
    assert actual.sliding_event_id == expected.sliding_event_id
    np.testing.assert_array_equal(actual.sliding_shift, expected.sliding_shift)


def _replace_journal(state, journal, *, content_state=None):
    return phx.solver.FiniteVolumeRuntimeState(
        state.content_state if content_state is None else content_state,
        journal,
        state.step_size,
        accepted_step=state.accepted_step,
        last_status=state.last_status,
        controller_state=state.controller_state,
        integrator_state=state.integrator_state,
        output_cursor=state.output_cursor,
        sliding_coupling=state.sliding_coupling,
        sliding_shift=state.sliding_shift,
        sliding_event_id=state.sliding_event_id,
    )


def test_case_schema_is_content_addressed_and_strict():
    runtime, _, _ = _prepared_runtime()
    execution = phx.solver.FiniteVolumeExecutionSpec(1.0, 1000)
    case = phx.solver.FiniteVolumeCaseSpec(
        "portable-euler",
        runtime,
        execution,
        precision=phx.solver.FiniteVolumePrecisionPolicy("float64"),
    )
    payload = case.to_dict()

    phx.solver.FiniteVolumeCaseSpec.validate_dict(payload)
    assert "schema_version" not in payload
    assert payload["case_id"] == case.case_id
    restored = phx.solver.FiniteVolumeCaseSpec.from_dict(payload, runtime, execution)
    assert restored.case_id == case.case_id
    with pytest.raises(ValueError, match="unknown"):
        phx.solver.FiniteVolumeCaseSpec.validate_dict(
            {**payload, "misspelled_flux": "HLLC"}
        )


def test_checkpoint_roundtrip_preserves_exact_runtime_state(tmp_path):
    runtime, _, state = _prepared_runtime()
    initial = state.topology_journal.epoch_table[0]
    request = phx.solver.FiniteVolumeTopologyEventRequest(
        phx.solver.TopologyEventKind.REMESH,
        initial.epoch_id,
        "roundtrip-requested-topology",
        reason="checkpoint-roundtrip",
    )
    result_epoch = phx.discretization.TopologyEpoch(
        initial.index + 1,
        "roundtrip-geometry",
        "roundtrip-topology",
        initial.partition_id,
    )
    result_artifacts = phx.solver.FiniteVolumeTopologyArtifacts(
        result_epoch,
        "roundtrip-prepared-topology",
    )
    journal = state.topology_journal.append_requested(request, 7, state.time).commit(
        0, result_epoch, result_artifacts
    )
    original_content = state.content_state
    content = phx.solver.FiniteVolumeConservativeContentState(
        original_content.conservative_content,
        original_content.effective_cell_volumes,
        original_content.active_cell_mask,
        original_content.time,
        topology_epoch_id=result_epoch.epoch_id,
        geometry_family_id=result_epoch.geometry_id,
        geometry_layout_id="roundtrip-geometry-layout",
        geometry_version=original_content.geometry_version + 1,
        evidence_policy_id="roundtrip-evidence-policy",
        evidence_version=original_content.evidence_version + 1,
        precision=original_content.precision,
    )
    state = _replace_journal(state, journal, content_state=content)
    case = phx.solver.FiniteVolumeCaseSpec(
        "checkpoint-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "restart.fvckpt"
    written = phx.solver.write_finite_volume_checkpoint(path, plan, state)
    loaded = phx.solver.read_finite_volume_checkpoint(path, plan)

    assert written.payload_id == loaded.payload_id
    _assert_runtime_state_exact(loaded.runtime_state, state)
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert "schema_version" not in manifest
    assert "runtime_state_schema_version" not in manifest
    assert "schema_version" not in manifest["content"]
    assert (
        manifest["content"]["geometry_family_id"]
        == state.content_state.geometry_family_id
    )
    assert (
        manifest["content"]["topology_epoch_id"] == state.content_state.topology_epoch_id
    )
    assert manifest["topology_journal"]["journal_id"] == state.topology_journal.journal_id
    assert "conservative_state" not in manifest["arrays"]
    assert "content/conservative_content" in manifest["arrays"]
    assert "content/effective_cell_volumes" in manifest["arrays"]
    assert "content/active_cell_mask" in manifest["arrays"]
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_block_checkpoint_roundtrip_preserves_canonical_hierarchy_and_routes(tmp_path):
    prepared, compiled, runtime, state = _prepared_block_runtime()
    partition = phx.discretization.BlockAMRPartitionPlan(
        compiled.topology.plan, 1
    ).prepare(compiled, prepared)
    plan = phx.solver.FiniteVolumeCheckpointPlan(runtime, partition=partition)
    path = tmp_path / "block.fvckpt"

    written = phx.solver.write_finite_volume_checkpoint(path, plan, state)
    loaded = phx.solver.read_finite_volume_checkpoint(path, plan)

    assert written.payload_id == loaded.payload_id
    restored = loaded.runtime_state
    assert isinstance(restored, phx.solver.BlockAMRRuntimeState)
    assert (
        restored.hierarchy_state.topology.epoch.epoch_id
        == state.hierarchy_state.topology.epoch.epoch_id
    )
    assert (
        restored.topology_journal.to_archive_record()
        == state.topology_journal.to_archive_record()
    )
    np.testing.assert_array_equal(restored.time, state.time)
    np.testing.assert_array_equal(restored.accepted_step, state.accepted_step)
    np.testing.assert_array_equal(
        restored.level_accepted_steps, state.level_accepted_steps
    )
    for actual, expected in zip(
        restored.hierarchy_state.levels,
        state.hierarchy_state.levels,
        strict=True,
    ):
        assert actual.metadata.metadata_id == expected.metadata.metadata_id
        np.testing.assert_array_equal(actual.safe_values(), expected.safe_values())
        active_ids = np.asarray(actual.metadata.block_ids)[
            np.asarray(actual.metadata.active, dtype="bool")
        ]
        np.testing.assert_array_equal(active_ids, np.sort(active_ids))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["archive_kind"] == "finite-volume-checkpoint"
    assert "schema_version" not in manifest
    assert "runtime_state_schema_version" not in manifest
    assert (
        manifest["topology"]["epoch"]["epoch_id"]
        == state.hierarchy_state.topology.epoch.epoch_id
    )
    assert manifest["topology"]["metadata_ids"]
    assert manifest["topology"]["covered_cell_ids"]
    assert manifest["topology"]["face_route_ids"]
    assert manifest["topology"]["edge_route_ids"]
    assert (
        manifest["topology"]["distributed_partition"]
        == partition.manifest_compatibility_data()
    )


def test_block_checkpoint_rejects_a_different_topology_runtime(tmp_path):
    prepared, _, runtime, state = _prepared_block_runtime()
    path = tmp_path / "block.fvckpt"
    phx.solver.write_finite_volume_checkpoint(
        path,
        phx.solver.FiniteVolumeCheckpointPlan(runtime),
        state,
    )
    incompatible = phx.solver.BlockAMRRuntimePlan(runtime.plan.finite_volume).prepare(
        prepared.initial_topology()
    )

    with pytest.raises(ValueError, match="incompatible"):
        phx.solver.read_finite_volume_checkpoint(
            path,
            phx.solver.FiniteVolumeCheckpointPlan(incompatible),
        )


def test_block_checkpoint_rejects_stale_prepared_route_artifacts(tmp_path):
    _, _, runtime, state = _prepared_block_runtime()
    topology = state.hierarchy_state.topology
    stale_artifacts = phx.solver.FiniteVolumeTopologyArtifacts(
        topology.epoch,
        "stale-block-runtime",
        topology_artifact_id=topology.topology_id,
    )
    stale_journal = state.topology_journal._new(
        artifact_table=(stale_artifacts,),
    )
    stale_state = phx.solver.BlockAMRRuntimeState(
        state.hierarchy_state,
        stale_journal,
        state.time,
        accepted_step=state.accepted_step,
        level_accepted_steps=state.level_accepted_steps,
        last_status=state.last_status,
    )

    with pytest.raises(ValueError, match="prepared routes"):
        phx.solver.write_finite_volume_checkpoint(
            tmp_path / "stale-block.fvckpt",
            phx.solver.FiniteVolumeCheckpointPlan(runtime),
            stale_state,
        )


def test_block_output_records_epoch_metadata_coverage_routes_and_precision(tmp_path):
    prepared, compiled, runtime, state = _prepared_block_runtime()
    partition = phx.discretization.BlockAMRPartitionPlan(
        compiled.topology.plan, 1
    ).prepare(compiled, prepared)
    plan = phx.solver.FiniteVolumeOutputPlan(
        tmp_path / "block-output.h5",
        runtime,
        partition=partition,
    )
    if find_spec("h5py") is None:
        with pytest.raises(ImportError, match="h5py"):
            plan.write_snapshot(runtime, state)
        return

    assert plan.write_snapshot(runtime, state) == 0
    h5py = pytest.importorskip("h5py")
    with h5py.File(plan.hdf5_path) as handle:
        assert handle.attrs["geometry_kind"] == "block_amr"
        assert (
            handle["block_hierarchy"].attrs["topology_epoch_id"]
            == state.hierarchy_state.topology.epoch.epoch_id
        )
        assert (
            json.loads(handle["block_hierarchy"].attrs["distributed_partition_json"])
            == partition.manifest_compatibility_data()
        )
        routes = json.loads(handle["block_hierarchy"].attrs["route_ids_json"])
        assert routes["fill_patch_plan_ids"]
        assert routes["face_route_ids"]
        assert routes["edge_route_ids"]
        for level, expected in enumerate(state.hierarchy_state.levels):
            topology_group = handle[f"block_hierarchy/levels/{level:04d}"]
            step_group = handle[f"steps/00000000/levels/{level:04d}"]
            assert topology_group.attrs["metadata_id"] == expected.metadata.metadata_id
            assert (
                topology_group.attrs["covered_cells_id"]
                == array_tree_fingerprint(
                    state.hierarchy_state.topology.covered_cells[level]
                )["sha256"]
            )
            np.testing.assert_array_equal(
                topology_group["covered_cells"],
                state.hierarchy_state.topology.covered_cells[level],
            )
            np.testing.assert_array_equal(
                step_group["cell_average"],
                np.asarray(
                    expected.safe_values(),
                    dtype=runtime.dynamics.plan.precision.numpy_dtype("output"),
                ),
            )
    assert Path(plan.xdmf_path).exists()


def test_checkpoint_rejects_manifest_corruption(tmp_path):
    runtime, _, state = _prepared_runtime()
    case = phx.solver.FiniteVolumeCaseSpec(
        "corrupt-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "restart.fvckpt"
    phx.solver.write_finite_volume_checkpoint(path, plan, state)
    with zipfile.ZipFile(path, "r") as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["case"]["method_id"] = "changed"
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)

    with pytest.raises(ValueError, match="case identity|corrupt"):
        phx.solver.read_finite_volume_checkpoint(path, plan)


def test_topology_archive_reconstruction_rejects_malformed_epoch_and_event():
    _, _, state = _prepared_runtime()
    initial = state.topology_journal.epoch_table[0]
    malformed_epoch = initial.to_archive_record()
    malformed_epoch["geometry_id"] = "changed-geometry"
    with pytest.raises(ValueError, match="epoch archive identity"):
        phx.discretization.TopologyEpoch.from_archive_record(malformed_epoch)
    malformed_artifacts = state.topology_journal.artifact_table[0].to_archive_record()
    malformed_artifacts["prepared_id"] = "changed-preparation"
    with pytest.raises(ValueError, match="artifacts archive identity"):
        phx.solver.FiniteVolumeTopologyArtifacts.from_archive_record(
            malformed_artifacts,
            initial,
        )

    request = phx.solver.FiniteVolumeTopologyEventRequest(
        phx.solver.TopologyEventKind.REMESH,
        initial.epoch_id,
        "requested-topology",
        reason="archive-validation",
    )
    journal = state.topology_journal.append_requested(request, 7, state.time).fail(
        0,
        result_id="rejected-topology",
    )
    malformed_journal = copy.deepcopy(journal.to_archive_record())
    malformed_journal["events"][0]["state"] = int(phx.solver.TopologyEventState.COMMITTED)
    with pytest.raises(ValueError, match="committed event"):
        phx.solver.FiniteVolumeTopologyEventJournal.from_archive_record(
            malformed_journal,
            journal.archive_arrays(),
        )


def test_output_plan_is_explicitly_optional_when_h5py_is_unavailable(tmp_path):
    _, discretization, state = _prepared_runtime()
    plan = phx.solver.FiniteVolumeOutputPlan(tmp_path / "solution.h5", discretization)
    if find_spec("h5py") is None:
        with pytest.raises(ImportError, match="h5py"):
            plan.write_snapshot(discretization, state)
    else:
        index = plan.write_snapshot(discretization, state)
        assert index == 0
        assert Path(plan.hdf5_path).exists()
        assert Path(plan.xdmf_path).exists()


def test_allowlisted_case_loader_builds_portable_runtime():
    payload = {
        "name": "loaded-euler",
        "grid": {
            "cells": 16,
            "lower": 0.0,
            "upper": 1.0,
            "periodic": True,
        },
        "equation": {
            "type": "ideal_gas_euler",
            "gamma": 1.4,
            "gas_constant": 1.0,
        },
        "method": {
            "reconstruction": "muscl",
            "flux": "hllc",
        },
        "boundary": {"type": "periodic"},
        "execution": {"end_time": 0.1, "maximum_steps": 100},
        "precision": {"dtype": "float64"},
    }
    prepared = phx.solver.load_finite_volume_case(payload)
    assert isinstance(
        prepared.discretization,
        phx.discretization.FiniteVolumeDiscretization,
    )

    assert prepared.discretization.cell_shape == (16,)
    assert prepared.runtime.dynamics.system.component_names == (
        "density",
        "momentum_0",
        "total_energy",
    )
    with pytest.raises(ValueError, match="unknown"):
        phx.solver.load_finite_volume_case({**payload, "misspelled_method": "hllc"})


def test_interrupted_checkpoint_trajectory_matches_uninterrupted(tmp_path):
    runtime, _, initial = _prepared_runtime()

    def advance_many(state, count):
        current = state
        for _ in range(count):
            current = runtime.advance(current).runtime_state
        return current

    uninterrupted = advance_many(initial, 3)
    interrupted = advance_many(initial, 1)
    case = phx.solver.FiniteVolumeCaseSpec(
        "segmented-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case)
    path = tmp_path / "segmented.fvckpt"
    phx.solver.write_finite_volume_checkpoint(path, plan, interrupted)
    restored = phx.solver.read_finite_volume_checkpoint(path, plan).runtime_state
    resumed = advance_many(restored, 2)

    _assert_runtime_state_exact(resumed, uninterrupted)
