#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._finite_volume_case import FiniteVolumeCaseSpec
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_runtime import (
    FiniteVolumeRunStatus,
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)
from ._finite_volume_topology_events import FiniteVolumeTopologyEventJournal


_RUNTIME_ARRAY_NAMES = (
    "content/conservative_content",
    "content/effective_cell_volumes",
    "content/active_cell_mask",
    "content/time",
    "content/geometry_version",
    "content/evidence_version",
    "accepted_step",
    "step_size",
    "last_status",
    "controller_state",
    "integrator_state",
    "forcing_state",
    "random_state",
    "output_cursor",
    "topology_journal/kinds",
    "topology_journal/states",
    "topology_journal/statuses",
    "topology_journal/accepted_steps",
    "topology_journal/times",
    "topology_journal/next_sequence",
    "topology_journal/count",
    "topology_journal/overflowed",
)
_SLIDING_ARRAY_NAMES = (
    "sliding/shift",
    "sliding/left_routes",
    "sliding/right_routes",
    "sliding/overlap_measures",
    "sliding/left_measures",
    "sliding/right_measures",
)
_JOURNAL_ARRAY_PREFIX = "topology_journal/"
_CONTENT_RECORD_FIELDS = frozenset(
    (
        "schema_version",
        "content_policy_id",
        "content_layout_id",
        "precision_policy_id",
        "topology_epoch_id",
        "geometry_family_id",
        "geometry_layout_id",
        "evidence_policy_id",
    )
)


def _payload_id(manifest, arrays, /) -> str:
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


def _content_policy_id(
    precision_policy_id: str,
    evidence_policy_id: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "finite-volume-conservative-content-policy",
            "schema_version": 1,
            "precision_policy_id": precision_policy_id,
            "evidence_policy_id": evidence_policy_id,
        }
    )


def _content_layout_id(
    content_state: FiniteVolumeConservativeContentState,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "finite-volume-conservative-content-layout",
            "schema_version": 2,
            "topology_epoch_id": content_state.topology_epoch_id,
            "geometry_family_id": content_state.geometry_family_id,
            "geometry_layout_id": content_state.geometry_layout_id,
            "content_shape": list(content_state.conservative_content.shape),
            "effective_volume_shape": list(content_state.effective_cell_volumes.shape),
            "active_mask_shape": list(content_state.active_cell_mask.shape),
        }
    )


def _content_record(
    content_state: FiniteVolumeConservativeContentState,
    /,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "content_policy_id": _content_policy_id(
            content_state.precision.policy_id,
            content_state.evidence_policy_id,
        ),
        "content_layout_id": _content_layout_id(content_state),
        "precision_policy_id": content_state.precision.policy_id,
        "topology_epoch_id": content_state.topology_epoch_id,
        "geometry_family_id": content_state.geometry_family_id,
        "geometry_layout_id": content_state.geometry_layout_id,
        "evidence_policy_id": content_state.evidence_policy_id,
    }


def _require_identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _validate_array_inventory(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    expected_names: set[str],
    /,
) -> None:
    inventory = manifest.get("arrays")
    if not isinstance(inventory, dict) or set(inventory) != expected_names:
        raise ValueError("Finite-volume checkpoint array inventory changed.")
    if set(arrays) != expected_names:
        raise ValueError("Finite-volume checkpoint array payload changed.")
    for index, name in enumerate(sorted(expected_names)):
        record = inventory[name]
        array = np.asarray(arrays[name])
        expected_record_fields = {"member", "shape", "dtype", "sha256"}
        if not isinstance(record, dict) or set(record) != expected_record_fields:
            raise ValueError(
                f"Finite-volume checkpoint inventory record {name!r} changed."
            )
        checksum = record["sha256"]
        if (
            record["member"] != f"arrays/{index:06d}.npy"
            or record["shape"] != list(array.shape)
            or record["dtype"] != array.dtype.str
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError(
                f"Finite-volume checkpoint inventory identity {name!r} changed."
            )


def _expected_content_shape(plan: FiniteVolumeCheckpointPlan, /) -> tuple[int, ...]:
    return (
        int(np.prod(plan.case.state_shape[:-1])),
        plan.case.state_shape[-1],
    )


def _validate_initial_epoch(
    plan: FiniteVolumeCheckpointPlan,
    journal: FiniteVolumeTopologyEventJournal,
    /,
) -> None:
    initial = journal.epoch_table[0]
    if (
        initial.prepared_id != plan.case.discretization_id
        or initial.topology_id != plan.case.mesh_topology_id
        or initial.geometry_id != plan.case.mesh_geometry_id
    ):
        raise ValueError(
            "Finite-volume checkpoint initial topology epoch is incompatible."
        )


class FiniteVolumeCheckpointPlan(StrictModule, NonTrainableState):
    case: FiniteVolumeCaseSpec
    runtime: PreparedFiniteVolumeRuntime | None
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        case: FiniteVolumeCaseSpec,
        /,
        *,
        runtime: PreparedFiniteVolumeRuntime | None = None,
    ):
        if not isinstance(case, FiniteVolumeCaseSpec):
            raise TypeError("case must be a FiniteVolumeCaseSpec.")
        if runtime is not None:
            if not isinstance(runtime, PreparedFiniteVolumeRuntime):
                raise TypeError("runtime must be PreparedFiniteVolumeRuntime or None.")
            dynamics = runtime.dynamics
            discretization = dynamics.discretization
            if (
                runtime.runtime_id != case.runtime_id
                or runtime.precision.policy_id != case.precision.policy_id
                or dynamics.system.system_id != case.system_id
                or discretization.prepared_id != case.discretization_id
                or dynamics.method.method_id != case.method_id
                or dynamics.boundaries.boundary_set_id != case.boundary_id
            ):
                raise ValueError(
                    "Checkpoint prepared runtime is incompatible with the case."
                )
        self.case = case
        self.runtime = runtime
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "finite-volume-checkpoint-plan",
                "schema_version": 5,
                "runtime_state_schema_version": 4,
                "content_state_schema_version": 2,
                "topology_journal_schema_version": 1,
                "case": case.case_id,
                "topology": case.mesh_topology_id,
                "geometry": case.mesh_geometry_id,
                "precision_policy_id": case.precision.policy_id,
                "precision_evidence_id": case.precision.evidence().evidence_id,
                "checkpoint_dtype": case.precision.checkpoint_dtype,
            }
        )


class FiniteVolumeCheckpoint(StrictModule):
    runtime_state: FiniteVolumeRuntimeState
    checkpoint_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope


_SLIDING_RECORD_FIELDS = frozenset(
    (
        "schema_version",
        "plan_id",
        "normalized_shift_hex",
        "shift_precision",
        "coupling_id",
        "evidence_id",
        "event_id",
    )
)


def _sliding_record(
    plan: FiniteVolumeCheckpointPlan,
    runtime_state: FiniteVolumeRuntimeState,
    /,
) -> dict[str, Any] | None:
    coupling = runtime_state.sliding_coupling
    sliding_plan = None if plan.runtime is None else plan.runtime.sliding_plan
    if coupling is None:
        if runtime_state.sliding_event_id is not None:
            raise ValueError("Checkpoint sliding event has no certified coupling.")
        if sliding_plan is not None:
            raise ValueError("Checkpoint sliding runtime has no certified coupling.")
        return None
    if sliding_plan is None:
        raise ValueError(
            "Checkpoint sliding state requires the originating prepared runtime."
        )
    runtime_shift = np.asarray(runtime_state.sliding_shift)
    expected_runtime_shift = np.asarray(
        coupling.normalized_shift,
        dtype=plan.case.precision.numpy_dtype("reduction"),
    )
    if not np.array_equal(runtime_shift, expected_runtime_shift):
        raise ValueError("Checkpoint sliding shift and coupling identity are stale.")
    rebuilt = sliding_plan.coupling(coupling.normalized_shift)
    scalar_names = (
        "coverage_error",
        "coverage_passed",
        "passed",
        "status",
        "conservation_defect",
    )
    array_names = (
        "left_routes",
        "right_routes",
        "overlap_measures",
        "left_measures",
        "right_measures",
    )
    if (
        rebuilt.coupling_id != coupling.coupling_id
        or rebuilt.evidence_id != coupling.evidence_id
        or rebuilt.shift_precision != coupling.shift_precision
        or any(
            not np.array_equal(
                np.asarray(getattr(rebuilt, name)),
                np.asarray(getattr(coupling, name)),
            )
            for name in (*array_names, *scalar_names)
        )
    ):
        raise ValueError("Checkpoint sliding coupling cannot be rebuilt exactly.")

    event_id = runtime_state.sliding_event_id
    if event_id is not None:
        journal_record = runtime_state.topology_journal.to_archive_record()
        events = [
            event for event in journal_record["events"] if event["event_id"] == event_id
        ]
        current_epoch = runtime_state.topology_journal.epoch_table[-1]
        if (
            len(events) != 1
            or events[0]["payload_id"] != coupling.evidence_id
            or events[0]["result_id"] != current_epoch.epoch_id
            or current_epoch.topology_artifact_id != coupling.coupling_id
            or current_epoch.metrics_artifact_id != coupling.evidence_id
            or current_epoch.operators_artifact_id != sliding_plan.plan_id
        ):
            raise ValueError(
                "Checkpoint sliding event, coupling, and successor epoch are stale."
            )
    return {
        "schema_version": 1,
        "plan_id": sliding_plan.plan_id,
        "normalized_shift_hex": float(coupling.normalized_shift).hex(),
        "shift_precision": coupling.shift_precision,
        "coupling_id": coupling.coupling_id,
        "evidence_id": coupling.evidence_id,
        "event_id": event_id,
    }


def _expected_array_names(
    plan: FiniteVolumeCheckpointPlan,
    sliding_record: dict[str, Any] | None,
    /,
) -> set[str]:
    names = set(_RUNTIME_ARRAY_NAMES)
    if plan.case.mesh_kind == "unstructured":
        names.update(
            {
                "mesh/vertices",
                "mesh/vertex_global_ids",
                "mesh/cell_global_ids",
            }
        )
    if sliding_record is not None:
        names.update(_SLIDING_ARRAY_NAMES)
    return names


def _runtime_arrays(
    plan: FiniteVolumeCheckpointPlan,
    runtime_state: FiniteVolumeRuntimeState,
    /,
) -> dict[str, np.ndarray]:
    checkpoint_dtype = plan.case.precision.numpy_dtype("checkpoint")
    content = runtime_state.content_state
    arrays = {
        "content/conservative_content": np.asarray(
            content.conservative_content, dtype=checkpoint_dtype
        ),
        "content/effective_cell_volumes": np.asarray(
            content.effective_cell_volumes, dtype=checkpoint_dtype
        ),
        "content/active_cell_mask": np.asarray(content.active_cell_mask, dtype=np.bool_),
        "content/time": np.asarray(content.time, dtype=checkpoint_dtype),
        "content/geometry_version": np.asarray(content.geometry_version, dtype=np.int32),
        "content/evidence_version": np.asarray(content.evidence_version, dtype=np.int32),
        "accepted_step": np.asarray(runtime_state.accepted_step, dtype=np.int32),
        "step_size": np.asarray(runtime_state.step_size, dtype=checkpoint_dtype),
        "last_status": np.asarray(runtime_state.last_status, dtype=np.int32),
        "controller_state": np.asarray(runtime_state.controller_state),
        "integrator_state": np.asarray(runtime_state.integrator_state),
        "forcing_state": np.asarray(runtime_state.forcing_state),
        "random_state": np.asarray(runtime_state.random_state, dtype=np.uint32),
        "output_cursor": np.asarray(runtime_state.output_cursor, dtype=np.int32),
    }
    arrays.update(
        {
            f"{_JOURNAL_ARRAY_PREFIX}{name}": value
            for name, value in runtime_state.topology_journal.archive_arrays().items()
        }
    )
    if plan.case.mesh_kind == "unstructured":
        arrays.update(
            {
                "mesh/vertices": np.asarray(plan.case.mesh_vertices),
                "mesh/vertex_global_ids": np.asarray(
                    plan.case.vertex_global_ids, dtype=np.int64
                ),
                "mesh/cell_global_ids": np.asarray(
                    plan.case.cell_global_ids, dtype=np.int64
                ),
            }
        )
    if runtime_state.sliding_coupling is not None:
        coupling = runtime_state.sliding_coupling
        arrays.update(
            {
                "sliding/shift": np.asarray(
                    runtime_state.sliding_shift,
                    dtype=plan.case.precision.numpy_dtype("reduction"),
                ),
                "sliding/left_routes": np.asarray(coupling.left_routes),
                "sliding/right_routes": np.asarray(coupling.right_routes),
                "sliding/overlap_measures": np.asarray(coupling.overlap_measures),
                "sliding/left_measures": np.asarray(coupling.left_measures),
                "sliding/right_measures": np.asarray(coupling.right_measures),
            }
        )
    return arrays


def _validate_runtime_arrays(
    plan: FiniteVolumeCheckpointPlan,
    arrays: dict[str, np.ndarray],
    /,
) -> None:
    checkpoint_dtype = plan.case.precision.numpy_dtype("checkpoint")
    content_shape = _expected_content_shape(plan)
    cell_count = content_shape[0]
    exact_shapes_and_dtypes = {
        "content/conservative_content": (content_shape, checkpoint_dtype),
        "content/effective_cell_volumes": ((cell_count,), checkpoint_dtype),
        "content/active_cell_mask": ((cell_count,), np.dtype(np.bool_)),
        "content/time": ((), checkpoint_dtype),
        "content/geometry_version": ((), np.dtype(np.int32)),
        "content/evidence_version": ((), np.dtype(np.int32)),
        "accepted_step": ((), np.dtype(np.int32)),
        "step_size": ((), checkpoint_dtype),
        "last_status": ((), np.dtype(np.int32)),
        "output_cursor": ((), np.dtype(np.int32)),
    }
    for name, (shape, dtype) in exact_shapes_and_dtypes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Finite-volume checkpoint runtime array {name!r} changed.")
    if np.asarray(arrays["random_state"]).dtype != np.dtype(np.uint32):
        raise ValueError("Finite-volume checkpoint random state dtype changed.")
    content = np.asarray(arrays["content/conservative_content"])
    volumes = np.asarray(arrays["content/effective_cell_volumes"])
    active = np.asarray(arrays["content/active_cell_mask"])
    if not np.all(np.isfinite(content)) or not np.all(np.isfinite(volumes)):
        raise ValueError("Finite-volume checkpoint content is nonfinite.")
    if (
        np.any(active & (volumes <= 0.0))
        or np.any((~active) & (volumes != 0.0))
        or np.any(content[~active] != 0.0)
    ):
        raise ValueError("Finite-volume checkpoint active-cell ownership changed.")
    scalar_values = (
        arrays["content/time"],
        arrays["step_size"],
    )
    if not all(np.isfinite(np.asarray(value)).item() for value in scalar_values):
        raise ValueError("Finite-volume checkpoint runtime timing is nonfinite.")
    if float(np.asarray(arrays["step_size"])) <= 0.0:
        raise ValueError("Finite-volume checkpoint step size must be positive.")
    if (
        int(np.asarray(arrays["accepted_step"])) < 0
        or int(np.asarray(arrays["output_cursor"])) < 0
        or int(np.asarray(arrays["content/geometry_version"])) < 0
        or int(np.asarray(arrays["content/evidence_version"])) < 0
    ):
        raise ValueError("Finite-volume checkpoint runtime counters are invalid.")
    if int(np.asarray(arrays["last_status"])) not in {
        int(status) for status in FiniteVolumeRunStatus
    }:
        raise ValueError("Finite-volume checkpoint runtime status is invalid.")


def _journal_arrays(
    arrays: dict[str, np.ndarray],
    /,
) -> dict[str, np.ndarray]:
    return {
        name.removeprefix(_JOURNAL_ARRAY_PREFIX): value
        for name, value in arrays.items()
        if name.startswith(_JOURNAL_ARRAY_PREFIX)
    }


def _restore_sliding(
    plan: FiniteVolumeCheckpointPlan,
    record: Any,
    arrays: dict[str, np.ndarray],
    /,
):
    sliding_plan = None if plan.runtime is None else plan.runtime.sliding_plan
    if record is None:
        if sliding_plan is not None:
            raise ValueError("Finite-volume checkpoint omitted sliding runtime state.")
        return (
            None,
            np.asarray(0.0, dtype=plan.case.precision.numpy_dtype("reduction")),
            None,
        )
    if not isinstance(record, dict) or set(record) != _SLIDING_RECORD_FIELDS:
        raise ValueError("Finite-volume checkpoint sliding record fields changed.")
    if isinstance(record["schema_version"], bool) or record["schema_version"] != 1:
        raise ValueError("Unsupported finite-volume checkpoint sliding schema.")
    if sliding_plan is None:
        raise ValueError(
            "Reading a sliding checkpoint requires the originating prepared runtime."
        )
    if record["plan_id"] != sliding_plan.plan_id:
        raise ValueError("Finite-volume checkpoint sliding plan identity changed.")
    for name in ("coupling_id", "evidence_id"):
        _require_identifier(record[name], f"sliding {name}")
    event_id = record["event_id"]
    if event_id is not None:
        _require_identifier(event_id, "sliding event_id")
    shift_hex = record["normalized_shift_hex"]
    if not isinstance(shift_hex, str):
        raise ValueError("Finite-volume checkpoint sliding shift encoding changed.")
    try:
        normalized_shift = float.fromhex(shift_hex)
    except ValueError as error:
        raise ValueError(
            "Finite-volume checkpoint sliding shift encoding changed."
        ) from error
    if not np.isfinite(normalized_shift) or normalized_shift.hex() != shift_hex:
        raise ValueError("Finite-volume checkpoint sliding shift is not canonical.")
    if (
        isinstance(record["shift_precision"], bool)
        or record["shift_precision"] != sliding_plan.shift_precision
    ):
        raise ValueError("Finite-volume checkpoint sliding precision changed.")
    coupling = sliding_plan.coupling(normalized_shift)
    if (
        coupling.coupling_id != record["coupling_id"]
        or coupling.evidence_id != record["evidence_id"]
    ):
        raise ValueError("Finite-volume checkpoint sliding identity changed.")
    shift = np.asarray(arrays["sliding/shift"])
    expected_shift = np.asarray(
        normalized_shift,
        dtype=plan.case.precision.numpy_dtype("reduction"),
    )
    if (
        shift.shape != ()
        or shift.dtype != expected_shift.dtype
        or not np.isfinite(shift).item()
        or not np.array_equal(shift, expected_shift)
    ):
        raise ValueError("Finite-volume checkpoint sliding runtime shift changed.")
    for archive_name, attribute in (
        ("sliding/left_routes", "left_routes"),
        ("sliding/right_routes", "right_routes"),
        ("sliding/overlap_measures", "overlap_measures"),
        ("sliding/left_measures", "left_measures"),
        ("sliding/right_measures", "right_measures"),
    ):
        archived = np.asarray(arrays[archive_name])
        rebuilt = np.asarray(getattr(coupling, attribute))
        if (
            archived.shape != rebuilt.shape
            or archived.dtype != rebuilt.dtype
            or not np.array_equal(archived, rebuilt)
        ):
            raise ValueError(
                f"Finite-volume checkpoint sliding array {archive_name!r} changed."
            )
    return coupling, shift, event_id


def write_finite_volume_checkpoint(
    path: str | Path,
    plan: FiniteVolumeCheckpointPlan,
    runtime_state: FiniteVolumeRuntimeState,
    /,
) -> FiniteVolumeCheckpoint:
    """Write a canonical content-authoritative schema-5 FV restart archive."""

    if not isinstance(plan, FiniteVolumeCheckpointPlan):
        raise TypeError("plan must be a FiniteVolumeCheckpointPlan.")
    if not isinstance(runtime_state, FiniteVolumeRuntimeState):
        raise TypeError("runtime_state must be a FiniteVolumeRuntimeState.")
    content = runtime_state.content_state
    if content.precision.policy_id != plan.case.precision.policy_id:
        raise ValueError("Checkpoint content precision policy changed.")
    if (
        plan.runtime is not None
        and content.geometry_family_id != plan.runtime.geometry_family_id
    ):
        raise ValueError(
            "Checkpoint content geometry family does not match the prepared runtime."
        )
    plan.case.precision.validate_state(content.conservative_content)
    if content.conservative_content.shape != _expected_content_shape(plan):
        raise ValueError("Checkpoint conservative content shape changed.")
    _validate_initial_epoch(plan, runtime_state.topology_journal)
    sliding = _sliding_record(plan, runtime_state)
    arrays = _runtime_arrays(plan, runtime_state)
    _validate_runtime_arrays(plan, arrays)
    manifest = {
        "archive_kind": "finite-volume-checkpoint",
        "schema_version": 5,
        "runtime_state_schema_version": 4,
        "content_state_schema_version": 2,
        "checkpoint_id": plan.checkpoint_id,
        "case": plan.case.to_dict(),
        "precision_evidence": plan.case.precision.evidence().to_dict(),
        "mesh": {
            "kind": plan.case.mesh_kind,
            "topology_id": plan.case.mesh_topology_id,
            "geometry_id": plan.case.mesh_geometry_id,
        },
        "content": _content_record(content),
        "topology_journal": runtime_state.topology_journal.to_archive_record(),
        "sliding": sliding,
    }
    payload_id = _payload_id(manifest, arrays)
    manifest["payload_id"] = payload_id
    write_array_archive(path, manifest=manifest, arrays=arrays)
    return FiniteVolumeCheckpoint(
        runtime_state,
        plan.checkpoint_id,
        payload_id,
        plan.case.precision.evidence(),
    )


def _read_schema5_checkpoint(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> FiniteVolumeCheckpoint:
    """Strictly reconstruct one content-authoritative schema-5 archive."""

    required_manifest = {
        "archive_kind",
        "schema_version",
        "runtime_state_schema_version",
        "content_state_schema_version",
        "checkpoint_id",
        "case",
        "precision_evidence",
        "mesh",
        "content",
        "topology_journal",
        "payload_id",
        "arrays",
        "sliding",
    }
    if set(manifest) != required_manifest:
        raise ValueError("Finite-volume checkpoint manifest fields changed.")
    if (
        manifest["archive_kind"] != "finite-volume-checkpoint"
        or manifest["schema_version"] != 5
        or manifest["runtime_state_schema_version"] != 4
        or manifest["content_state_schema_version"] != 2
    ):
        raise ValueError("Unsupported finite-volume checkpoint schema.")
    if manifest["checkpoint_id"] != plan.checkpoint_id:
        raise ValueError("Finite-volume checkpoint is incompatible with this plan.")
    FiniteVolumeCaseSpec.validate_dict(manifest["case"])
    if manifest["case"] != plan.case.to_dict():
        raise ValueError("Finite-volume checkpoint case identity changed.")
    expected_mesh = {
        "kind": plan.case.mesh_kind,
        "topology_id": plan.case.mesh_topology_id,
        "geometry_id": plan.case.mesh_geometry_id,
    }
    if manifest["mesh"] != expected_mesh:
        raise ValueError("Finite-volume checkpoint mesh identity changed.")
    precision_evidence = PrecisionEvidenceEnvelope.from_dict(
        manifest["precision_evidence"]
    )
    expected_precision = plan.case.precision.evidence()
    if precision_evidence.evidence_id != expected_precision.evidence_id:
        raise ValueError("Finite-volume checkpoint precision evidence changed.")

    expected_arrays = _expected_array_names(plan, manifest["sliding"])
    _validate_array_inventory(manifest, arrays, expected_arrays)
    if _payload_id(manifest, arrays) != manifest["payload_id"]:
        raise ValueError("Finite-volume checkpoint payload identity changed.")
    _validate_runtime_arrays(plan, arrays)
    if plan.case.mesh_kind == "unstructured":
        if (
            not np.array_equal(arrays["mesh/vertices"], plan.case.mesh_vertices)
            or not np.array_equal(
                arrays["mesh/vertex_global_ids"], plan.case.vertex_global_ids
            )
            or not np.array_equal(
                arrays["mesh/cell_global_ids"], plan.case.cell_global_ids
            )
        ):
            raise ValueError("Finite-volume checkpoint mesh payload changed.")

    content_record = manifest["content"]
    if (
        not isinstance(content_record, dict)
        or set(content_record) != _CONTENT_RECORD_FIELDS
    ):
        raise ValueError("Finite-volume checkpoint content record fields changed.")
    if (
        isinstance(content_record["schema_version"], bool)
        or content_record["schema_version"] != 2
    ):
        raise ValueError("Unsupported finite-volume checkpoint content schema.")
    for name in (
        "content_policy_id",
        "content_layout_id",
        "precision_policy_id",
        "topology_epoch_id",
        "geometry_family_id",
        "geometry_layout_id",
        "evidence_policy_id",
    ):
        _require_identifier(content_record[name], name)
    if content_record["precision_policy_id"] != plan.case.precision.policy_id:
        raise ValueError("Finite-volume checkpoint content precision identity changed.")

    journal = FiniteVolumeTopologyEventJournal.from_archive_record(
        manifest["topology_journal"],
        _journal_arrays(arrays),
    )
    _validate_initial_epoch(plan, journal)
    sliding_coupling, sliding_shift, sliding_event_id = _restore_sliding(
        plan,
        manifest["sliding"],
        arrays,
    )
    if journal.current_epoch_id != content_record["topology_epoch_id"]:
        raise ValueError(
            "Finite-volume checkpoint content and journal epoch identities changed."
        )
    content = FiniteVolumeConservativeContentState(
        plan.case.precision.storage(arrays["content/conservative_content"]),
        plan.case.precision.reduction(arrays["content/effective_cell_volumes"]),
        arrays["content/active_cell_mask"],
        plan.case.precision.decision(arrays["content/time"]),
        topology_epoch_id=content_record["topology_epoch_id"],
        geometry_family_id=content_record["geometry_family_id"],
        geometry_layout_id=content_record["geometry_layout_id"],
        geometry_version=arrays["content/geometry_version"],
        evidence_policy_id=content_record["evidence_policy_id"],
        evidence_version=arrays["content/evidence_version"],
        precision=plan.case.precision,
    )
    if _content_record(content) != content_record:
        raise ValueError("Finite-volume checkpoint content identity changed.")
    if (
        plan.runtime is not None
        and content.geometry_family_id != plan.runtime.geometry_family_id
    ):
        raise ValueError("Finite-volume checkpoint content geometry family is stale.")
    runtime = FiniteVolumeRuntimeState(
        content,
        journal,
        plan.case.precision.decision(arrays["step_size"]),
        accepted_step=arrays["accepted_step"],
        last_status=arrays["last_status"],
        controller_state=arrays["controller_state"],
        integrator_state=arrays["integrator_state"],
        forcing_state=arrays["forcing_state"],
        random_state=arrays["random_state"],
        output_cursor=arrays["output_cursor"],
        sliding_coupling=sliding_coupling,
        sliding_shift=sliding_shift,
        sliding_event_id=sliding_event_id,
    )
    if _sliding_record(plan, runtime) != manifest["sliding"]:
        raise ValueError("Finite-volume checkpoint sliding identity changed.")
    return FiniteVolumeCheckpoint(
        runtime,
        plan.checkpoint_id,
        manifest["payload_id"],
        precision_evidence,
    )


_LEGACY_RUNTIME_ARRAY_NAMES = (
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


def _checkpoint_manifest_version(path: str | Path, /) -> int:
    try:
        with zipfile.ZipFile(Path(path), "r") as archive:
            if archive.namelist().count("manifest.json") != 1:
                raise ValueError("Finite-volume checkpoint must contain one manifest.")
            manifest = json.loads(archive.read("manifest.json"))
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError("Finite-volume checkpoint manifest is corrupt.") from error
    if not isinstance(manifest, dict):
        raise ValueError("Finite-volume checkpoint manifest must be an object.")
    version = manifest.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Finite-volume checkpoint schema version is invalid.")
    return version


def _validate_legacy_case(payload: Any, plan: FiniteVolumeCheckpointPlan, /) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Finite-volume checkpoint case record changed.")
    version = payload.get("schema_version")
    if version == 2:
        FiniteVolumeCaseSpec.validate_dict(payload)
        if payload != plan.case.to_dict():
            raise ValueError("Finite-volume checkpoint case identity changed.")
        return payload["case_id"]
    required = {
        "schema_version",
        "name",
        "runtime_id",
        "system_id",
        "discretization_id",
        "method_id",
        "boundary_id",
        "precision",
        "execution",
        "case_id",
    }
    if version != 1 or set(payload) != required:
        raise ValueError("Unsupported legacy finite-volume case schema.")
    identity_payload = dict(payload)
    case_id = identity_payload.pop("case_id")
    if not isinstance(case_id, str) or canonical_fingerprint(identity_payload) != case_id:
        raise ValueError("Legacy finite-volume checkpoint case identity changed.")
    expected = plan.case.to_dict()
    for name in (
        "name",
        "runtime_id",
        "system_id",
        "discretization_id",
        "method_id",
        "boundary_id",
        "precision",
        "execution",
    ):
        if payload[name] != expected[name]:
            raise ValueError("Legacy finite-volume checkpoint case identity changed.")
    return case_id


def _legacy_checkpoint_id(
    version: int,
    case_id: str,
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> str:
    payload = {
        "kind": "finite-volume-checkpoint-plan",
        "schema_version": version,
        "case": case_id,
        "precision_policy_id": plan.case.precision.policy_id,
        "precision_evidence_id": plan.case.precision.evidence().evidence_id,
        "checkpoint_dtype": plan.case.precision.checkpoint_dtype,
    }
    if version == 3:
        payload.update(
            {
                "runtime_state_schema_version": 2,
                "topology": plan.case.mesh_topology_id,
                "geometry": plan.case.mesh_geometry_id,
            }
        )
    return canonical_fingerprint(payload)


def _validate_legacy_precision(
    manifest: dict[str, Any],
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> PrecisionEvidenceEnvelope:
    precision = PrecisionEvidenceEnvelope.from_dict(manifest["precision_evidence"])
    current = plan.case.precision.evidence()
    if precision.evidence_id != current.evidence_id:
        raise ValueError("Finite-volume checkpoint precision evidence changed.")
    return current


def _validate_legacy_runtime_arrays(
    plan: FiniteVolumeCheckpointPlan,
    arrays: dict[str, np.ndarray],
    /,
) -> None:
    state = np.asarray(arrays["conservative_state"])
    if (
        state.shape != plan.case.state_shape
        or state.dtype != plan.case.precision.numpy_dtype("checkpoint")
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("Legacy finite-volume checkpoint state changed.")
    for name in ("time", "accepted_step", "step_size", "last_status", "output_cursor"):
        if np.asarray(arrays[name]).shape != ():
            raise ValueError(f"Legacy finite-volume checkpoint scalar {name!r} changed.")
    if (
        not np.isfinite(np.asarray(arrays["time"])).item()
        or not np.isfinite(np.asarray(arrays["step_size"])).item()
        or float(np.asarray(arrays["step_size"])) <= 0.0
    ):
        raise ValueError("Legacy finite-volume checkpoint timing changed.")
    accepted_step = int(np.asarray(arrays["accepted_step"]))
    output_cursor = int(np.asarray(arrays["output_cursor"]))
    if (
        accepted_step < 0
        or accepted_step > np.iinfo(np.int32).max
        or output_cursor < 0
        or output_cursor > np.iinfo(np.int32).max
        or int(np.asarray(arrays["last_status"]))
        not in {int(status) for status in FiniteVolumeRunStatus}
        or np.asarray(arrays["random_state"]).dtype != np.dtype(np.uint32)
    ):
        raise ValueError("Legacy finite-volume checkpoint runtime state changed.")


def _migrate_legacy_runtime(
    plan: FiniteVolumeCheckpointPlan,
    arrays: dict[str, np.ndarray],
    /,
) -> FiniteVolumeRuntimeState:
    runtime = plan.runtime
    if runtime is None:
        raise ValueError(
            "Reading checkpoint schema 2 or 3 requires the prepared runtime."
        )
    coupling = getattr(runtime.dynamics, "coupling", None)
    if runtime.sliding_plan is not None or (
        coupling is not None and getattr(coupling, "motion", None) is not None
    ):
        raise ValueError(
            "Checkpoint schemas 2 and 3 cannot restore ALE or sliding state."
        )
    _validate_legacy_runtime_arrays(plan, arrays)
    migrated = runtime.initialize_state(
        plan.case.precision.storage(arrays["conservative_state"]),
        plan.case.precision.decision(arrays["time"]),
        plan.case.precision.decision(arrays["step_size"]),
        accepted_step=arrays["accepted_step"],
        last_status=arrays["last_status"],
        controller_state=arrays["controller_state"],
        integrator_state=arrays["integrator_state"],
        forcing_state=arrays["forcing_state"],
        random_state=arrays["random_state"],
        output_cursor=arrays["output_cursor"],
    )
    if migrated.content_state.geometry_family_id != runtime.geometry_family_id:
        raise ValueError("Legacy checkpoint migration produced a stale geometry family.")
    return migrated


def _read_schema3_checkpoint(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> FiniteVolumeCheckpoint:
    required_manifest = {
        "archive_kind",
        "schema_version",
        "runtime_state_schema_version",
        "checkpoint_id",
        "case",
        "precision_evidence",
        "mesh",
        "payload_id",
        "arrays",
    }
    if set(manifest) != required_manifest:
        raise ValueError("Finite-volume checkpoint schema-3 manifest fields changed.")
    if (
        manifest["archive_kind"] != "finite-volume-checkpoint"
        or manifest["schema_version"] != 3
        or manifest["runtime_state_schema_version"] != 2
    ):
        raise ValueError("Unsupported finite-volume checkpoint schema.")
    case_id = _validate_legacy_case(manifest["case"], plan)
    if manifest["checkpoint_id"] != _legacy_checkpoint_id(3, case_id, plan):
        raise ValueError("Finite-volume checkpoint is incompatible with this plan.")
    expected_mesh = {
        "kind": plan.case.mesh_kind,
        "topology_id": plan.case.mesh_topology_id,
        "geometry_id": plan.case.mesh_geometry_id,
    }
    if manifest["mesh"] != expected_mesh:
        raise ValueError("Finite-volume checkpoint mesh identity changed.")
    precision = _validate_legacy_precision(manifest, plan)
    expected_arrays = set(_LEGACY_RUNTIME_ARRAY_NAMES)
    if plan.case.mesh_kind == "unstructured":
        expected_arrays.update(
            {
                "mesh/vertices",
                "mesh/vertex_global_ids",
                "mesh/cell_global_ids",
            }
        )
    _validate_array_inventory(manifest, arrays, expected_arrays)
    if _payload_id(manifest, arrays) != manifest["payload_id"]:
        raise ValueError("Finite-volume checkpoint payload identity changed.")
    if plan.case.mesh_kind == "unstructured" and (
        not np.array_equal(arrays["mesh/vertices"], plan.case.mesh_vertices)
        or not np.array_equal(
            arrays["mesh/vertex_global_ids"], plan.case.vertex_global_ids
        )
        or not np.array_equal(arrays["mesh/cell_global_ids"], plan.case.cell_global_ids)
    ):
        raise ValueError("Finite-volume checkpoint mesh payload changed.")
    runtime = _migrate_legacy_runtime(plan, arrays)
    return FiniteVolumeCheckpoint(
        runtime,
        plan.checkpoint_id,
        manifest["payload_id"],
        precision,
    )


def _read_schema2_checkpoint(
    path: str | Path,
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> FiniteVolumeCheckpoint:
    try:
        with zipfile.ZipFile(Path(path), "r") as archive:
            members = archive.namelist()
            manifest = json.loads(archive.read("manifest.json"))
            required_manifest = {
                "schema_version",
                "checkpoint_id",
                "case",
                "precision_evidence",
                "arrays",
                "payload_id",
            }
            if not isinstance(manifest, dict) or set(manifest) != required_manifest:
                raise ValueError(
                    "Finite-volume checkpoint schema-2 manifest fields changed."
                )
            if manifest["schema_version"] != 2:
                raise ValueError("Unsupported finite-volume checkpoint schema.")
            case_id = _validate_legacy_case(manifest["case"], plan)
            if manifest["checkpoint_id"] != _legacy_checkpoint_id(2, case_id, plan):
                raise ValueError(
                    "Finite-volume checkpoint is incompatible with this plan."
                )
            precision = _validate_legacy_precision(manifest, plan)
            inventory = manifest["arrays"]
            if not isinstance(inventory, dict) or set(inventory) != set(
                _LEGACY_RUNTIME_ARRAY_NAMES
            ):
                raise ValueError(
                    "Finite-volume checkpoint schema-2 array inventory changed."
                )
            expected_members = {"manifest.json"}
            arrays: dict[str, np.ndarray] = {}
            payloads: list[bytes] = []
            for name in _LEGACY_RUNTIME_ARRAY_NAMES:
                record = inventory[name]
                expected_fields = {"file", "sha256", "shape", "dtype"}
                expected_file = f"arrays/{name}.npy"
                if (
                    not isinstance(record, dict)
                    or set(record) != expected_fields
                    or record["file"] != expected_file
                ):
                    raise ValueError(
                        f"Finite-volume checkpoint schema-2 record {name!r} changed."
                    )
                expected_members.add(expected_file)
                payload = archive.read(expected_file)
                checksum = hashlib.sha256(payload).hexdigest()
                if checksum != record["sha256"]:
                    raise ValueError(
                        f"Finite-volume checkpoint array {name!r} is corrupt."
                    )
                value = np.load(io.BytesIO(payload), allow_pickle=False)
                if (
                    list(value.shape) != record["shape"]
                    or str(value.dtype) != record["dtype"]
                ):
                    raise ValueError(
                        f"Finite-volume checkpoint array {name!r} metadata changed."
                    )
                arrays[name] = value
                payloads.append(payload)
            if len(members) != len(set(members)) or set(members) != expected_members:
                raise ValueError("Finite-volume checkpoint schema-2 members changed.")
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError(
            "Finite-volume checkpoint schema-2 archive is corrupt."
        ) from error
    if plan.case.mesh_kind != "structured":
        raise ValueError(
            "Checkpoint schema 2 supports only its public structured runtime."
        )
    unsigned = dict(manifest)
    payload_id = unsigned.pop("payload_id")
    manifest_payload = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":")
    ).encode()
    actual_payload_id = hashlib.sha256(manifest_payload + b"".join(payloads)).hexdigest()
    if actual_payload_id != payload_id:
        raise ValueError("Finite-volume checkpoint manifest or payload is corrupt.")
    runtime = _migrate_legacy_runtime(plan, arrays)
    return FiniteVolumeCheckpoint(
        runtime,
        plan.checkpoint_id,
        actual_payload_id,
        precision,
    )


def read_finite_volume_checkpoint(
    path: str | Path,
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> FiniteVolumeCheckpoint:
    """Read schemas 2, 3, or 5 through explicit, identity-preserving migrations."""

    if not isinstance(plan, FiniteVolumeCheckpointPlan):
        raise TypeError("plan must be a FiniteVolumeCheckpointPlan.")
    version = _checkpoint_manifest_version(path)
    if version == 2:
        return _read_schema2_checkpoint(path, plan)
    if version in (3, 5):
        manifest, arrays = read_array_archive(path)
        if version == 3:
            return _read_schema3_checkpoint(manifest, arrays, plan)
        return _read_schema5_checkpoint(manifest, arrays, plan)
    raise ValueError("Unsupported finite-volume checkpoint schema.")


__all__ = [
    "FiniteVolumeCheckpoint",
    "FiniteVolumeCheckpointPlan",
    "read_finite_volume_checkpoint",
    "write_finite_volume_checkpoint",
]
