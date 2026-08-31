#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy
import hashlib
import io
import json
import zipfile
from importlib.util import find_spec
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import write_array_archive
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint


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


_LEGACY_ARRAY_NAMES = (
    "conservative_state",
    "time",
    "accepted_step",
    "step_size",
    "last_status",
    "controller_state",
    "integrator_state",
    "forcing_state",
    "random_state",
    "output_cursor",
)


def _legacy_arrays(case, state):
    checkpoint_dtype = case.precision.numpy_dtype("checkpoint")
    return {
        "conservative_state": np.asarray(
            state.cell_average().reshape(case.state_shape),
            dtype=checkpoint_dtype,
        ),
        "time": np.asarray(state.time),
        "accepted_step": np.asarray(state.accepted_step, dtype=np.int64),
        "step_size": np.asarray(state.step_size),
        "last_status": np.asarray(state.last_status, dtype=np.int32),
        "controller_state": np.asarray(state.controller_state),
        "integrator_state": np.asarray(state.integrator_state),
        "forcing_state": np.asarray(()),
        "random_state": np.asarray((), dtype=np.uint32),
        "output_cursor": np.asarray(state.output_cursor, dtype=np.int32),
    }


def _legacy_case_v1(case):
    current = case.to_dict()
    legacy = {
        name: current[name]
        for name in (
            "name",
            "runtime_id",
            "system_id",
            "discretization_id",
            "method_id",
            "boundary_id",
            "precision",
            "execution",
        )
    }
    legacy["schema_version"] = 1
    legacy["case_id"] = canonical_fingerprint(legacy)
    return legacy


def _legacy_checkpoint_id(version, case_id, case):
    payload = {
        "kind": "finite-volume-checkpoint-plan",
        "schema_version": version,
        "case": case_id,
        "precision_policy_id": case.precision.policy_id,
        "precision_evidence_id": case.precision.evidence().evidence_id,
        "checkpoint_dtype": case.precision.checkpoint_dtype,
    }
    if version == 3:
        payload.update(
            {
                "runtime_state_schema_version": 2,
                "topology": case.mesh_topology_id,
                "geometry": case.mesh_geometry_id,
            }
        )
    return canonical_fingerprint(payload)


def _legacy_payload_id(manifest, arrays):
    metadata = {
        name: value
        for name, value in manifest.items()
        if name not in ("arrays", "payload_id")
    }
    return canonical_fingerprint(
        {
            "manifest": metadata,
            "arrays": {
                name: array_tree_fingerprint(value)
                for name, value in sorted(arrays.items())
            },
        }
    )


def _write_legacy_checkpoint(path, version, case, state):
    arrays = _legacy_arrays(case, state)
    if version == 2:
        case_record = _legacy_case_v1(case)
        payloads = {}
        for name, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            payloads[name] = stream.getvalue()
        manifest = {
            "schema_version": 2,
            "checkpoint_id": _legacy_checkpoint_id(2, case_record["case_id"], case),
            "case": case_record,
            "precision_evidence": case.precision.evidence().to_dict(),
            "arrays": {
                name: {
                    "file": f"arrays/{name}.npy",
                    "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                    "shape": list(arrays[name].shape),
                    "dtype": str(arrays[name].dtype),
                }
                for name in _LEGACY_ARRAY_NAMES
            },
        }
        unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["payload_id"] = hashlib.sha256(
            unsigned + b"".join(payloads.values())
        ).hexdigest()
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            for name in _LEGACY_ARRAY_NAMES:
                archive.writestr(f"arrays/{name}.npy", payloads[name])
        return
    manifest = {
        "archive_kind": "finite-volume-checkpoint",
        "schema_version": 3,
        "runtime_state_schema_version": 2,
        "checkpoint_id": _legacy_checkpoint_id(3, case.case_id, case),
        "case": case.to_dict(),
        "precision_evidence": case.precision.evidence().to_dict(),
        "mesh": {
            "kind": case.mesh_kind,
            "topology_id": case.mesh_topology_id,
            "geometry_id": case.mesh_geometry_id,
        },
    }
    manifest["payload_id"] = _legacy_payload_id(manifest, arrays)
    write_array_archive(path, manifest=manifest, arrays=arrays)


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


def test_case_schema_is_versioned_content_addressed_and_strict():
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
    assert payload["schema_version"] == 2
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
    result_epoch = phx.solver.FiniteVolumeTopologyEpoch(
        "roundtrip-prepared-topology",
        "roundtrip-topology",
        "roundtrip-geometry",
        parent_epoch_id=initial.epoch_id,
    )
    journal = state.topology_journal.append_requested(request, 7, state.time).commit(
        0, result_epoch
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
    assert manifest["schema_version"] == 5
    assert manifest["runtime_state_schema_version"] == 4
    assert manifest["content"]["schema_version"] == 2
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
        phx.solver.FiniteVolumeTopologyEpoch.from_archive_record(malformed_epoch)

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
        "schema_version": 1,
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


@pytest.mark.parametrize("schema_version", (2, 3))
def test_public_legacy_checkpoints_migrate_and_continue(tmp_path, schema_version):
    runtime, _, state = _prepared_runtime()
    case = phx.solver.FiniteVolumeCaseSpec(
        f"legacy-schema-{schema_version}",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    plan = phx.solver.FiniteVolumeCheckpointPlan(case, runtime=runtime)
    path = tmp_path / f"legacy-{schema_version}.fvckpt"
    _write_legacy_checkpoint(path, schema_version, case, state)

    restored = phx.solver.read_finite_volume_checkpoint(path, plan).runtime_state

    np.testing.assert_array_equal(restored.cell_average(), state.cell_average())
    np.testing.assert_allclose(
        restored.content_state.conservative_content,
        state.content_state.conservative_content,
        rtol=0.0,
        atol=np.finfo(np.asarray(restored.cell_average()).dtype).eps,
    )
    assert restored.content_state.precision.policy_id == runtime.precision.policy_id
    assert restored.content_state.topology_epoch_id == runtime.topology_epoch_id
    assert restored.topology_journal.current_epoch_id == runtime.topology_epoch_id
    assert int(np.asarray(restored.topology_journal.count)) == 0
    assert int(np.asarray(restored.content_state.geometry_version)) == 0
    assert int(np.asarray(restored.content_state.evidence_version)) == 0

    continued = runtime.advance(restored)
    assert bool(np.asarray(continued.accepted))
    assert int(np.asarray(continued.runtime_state.accepted_step)) == (
        int(np.asarray(restored.accepted_step)) + 1
    )


def test_checkpoint_manifest_version_matches_unstructured_guide(tmp_path):
    runtime, _, state = _prepared_runtime()
    case = phx.solver.FiniteVolumeCaseSpec(
        "documented-checkpoint-schema",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    path = tmp_path / "documented.fvckpt"
    phx.solver.write_finite_volume_checkpoint(
        path,
        phx.solver.FiniteVolumeCheckpointPlan(case),
        state,
    )
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    guide = (
        Path(__file__).parents[3] / "docs" / "guides_unstructured_finite_volume.md"
    ).read_text()
    assert f"schema version {manifest['schema_version']}" in guide
    assert (
        f"runtime-state schema version {manifest['runtime_state_schema_version']}"
        in guide
    )
