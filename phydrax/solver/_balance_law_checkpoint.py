#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._balance_law import (
    BalanceLawProcessState,
    BalanceLawRuntimeState,
    PreparedBalanceLawRuntime,
)


class BalanceLawCheckpointPlan(StrictModule, NonTrainableState):
    runtime: PreparedBalanceLawRuntime
    temporal_mesh_id: str = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedBalanceLawRuntime,
        temporal_mesh_id: str,
        /,
        *,
        realization_id: str | None = None,
    ):
        if not isinstance(runtime, PreparedBalanceLawRuntime):
            raise TypeError("runtime must be PreparedBalanceLawRuntime.")
        if not runtime.transport.checkpoint_supported:
            raise TypeError(
                "Balance-law transport does not support portable checkpoints."
            )
        mesh_id = str(temporal_mesh_id)
        realization = None if realization_id is None else str(realization_id)
        if not mesh_id or (realization is not None and not realization):
            raise ValueError("Checkpoint mesh/realization IDs must be non-empty.")
        self.runtime = runtime
        self.temporal_mesh_id = mesh_id
        self.realization_id = realization
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "balance-law-checkpoint",
                "runtime": runtime.runtime_id,
                "temporal_mesh": mesh_id,
                "realization": realization,
            }
        )


class BalanceLawCheckpoint(StrictModule):
    runtime_state: BalanceLawRuntimeState
    checkpoint_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


def _process_manifest(state: BalanceLawRuntimeState, /) -> list[dict[str, Any]]:
    return [
        {
            "process_id": process_state.process_id,
            "fields": list(process_state.field_names),
        }
        for process_state in state.process_states
    ]


def _arrays(
    state: BalanceLawRuntimeState,
    runtime: PreparedBalanceLawRuntime,
    /,
) -> dict[str, np.ndarray]:
    arrays = runtime.transport.checkpoint_arrays(state.transport_state)
    for process_state in state.process_states:
        for name, value in zip(
            process_state.field_names, process_state.values, strict=True
        ):
            arrays[f"process/{process_state.process_id}/{name}"] = np.asarray(value)
    return arrays


def _payload_id(manifest: dict[str, Any], arrays: dict[str, np.ndarray], /) -> str:
    return canonical_fingerprint(
        {
            "manifest": manifest,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "bytes": canonical_fingerprint(value.tobytes().hex()),
                }
                for name, value in sorted(arrays.items())
            },
        }
    )


def write_balance_law_checkpoint(
    path: str | Path,
    plan: BalanceLawCheckpointPlan,
    state: BalanceLawRuntimeState,
    /,
) -> BalanceLawCheckpoint:
    if not isinstance(plan, BalanceLawCheckpointPlan):
        raise TypeError("plan must be BalanceLawCheckpointPlan.")
    if not isinstance(state, BalanceLawRuntimeState):
        raise TypeError("state must be BalanceLawRuntimeState.")
    if state.process_ids != plan.runtime.process_ids:
        raise ValueError("Checkpoint process order does not match its runtime.")
    manifest = {
        "kind": "balance-law-checkpoint",
        "checkpoint_id": plan.checkpoint_id,
        "runtime_id": plan.runtime.runtime_id,
        "transport_kind": plan.runtime.transport.transport_kind,
        "temporal_mesh_id": plan.temporal_mesh_id,
        "realization_id": plan.realization_id,
        "processes": _process_manifest(state),
    }
    arrays = _arrays(state, plan.runtime)
    payload_id = _payload_id(manifest, arrays)
    write_array_archive(
        path,
        manifest={**manifest, "payload_id": payload_id},
        arrays=arrays,
    )
    return BalanceLawCheckpoint(state, plan.checkpoint_id, payload_id)


def read_balance_law_checkpoint(
    path: str | Path,
    plan: BalanceLawCheckpointPlan,
    /,
) -> BalanceLawCheckpoint:
    if not isinstance(plan, BalanceLawCheckpointPlan):
        raise TypeError("plan must be BalanceLawCheckpointPlan.")
    manifest, arrays = read_array_archive(path)
    required = {
        "kind",
        "checkpoint_id",
        "runtime_id",
        "transport_kind",
        "temporal_mesh_id",
        "realization_id",
        "processes",
        "payload_id",
        "arrays",
    }
    if set(manifest) != required:
        raise ValueError("Balance-law checkpoint manifest fields changed.")
    if (
        manifest["kind"] != "balance-law-checkpoint"
        or manifest["checkpoint_id"] != plan.checkpoint_id
        or manifest["runtime_id"] != plan.runtime.runtime_id
        or manifest["transport_kind"] != plan.runtime.transport.transport_kind
        or manifest["temporal_mesh_id"] != plan.temporal_mesh_id
        or manifest["realization_id"] != plan.realization_id
    ):
        raise ValueError("Balance-law checkpoint identity does not match its plan.")
    process_records = manifest["processes"]
    if not isinstance(process_records, list) or any(
        not isinstance(record, dict) for record in process_records
    ):
        raise ValueError("Balance-law checkpoint process metadata is invalid.")
    if tuple(record.get("process_id") for record in process_records) != (
        plan.runtime.process_ids
    ):
        raise ValueError("Balance-law checkpoint process order changed.")
    expected_names = set(plan.runtime.transport.checkpoint_array_names())
    for record in process_records:
        if not isinstance(record, dict) or set(record) != {"process_id", "fields"}:
            raise ValueError("Balance-law checkpoint process metadata changed.")
        process_id = str(record["process_id"])
        fields = record["fields"]
        if not isinstance(fields, list) or any(
            not isinstance(name, str) or not name for name in fields
        ):
            raise ValueError("Balance-law checkpoint process fields are invalid.")
        expected_names.update(f"process/{process_id}/{name}" for name in fields)
    if set(arrays) != expected_names:
        raise ValueError("Balance-law checkpoint array inventory changed.")
    payload_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in ("arrays", "payload_id")
    }
    payload_id = _payload_id(payload_manifest, arrays)
    if payload_id != manifest["payload_id"]:
        raise ValueError("Balance-law checkpoint payload identity failed.")
    transport_state = plan.runtime.transport.restore_checkpoint(arrays)
    process_states = tuple(
        BalanceLawProcessState(
            str(record["process_id"]),
            tuple(record["fields"]),
            tuple(
                jnp.asarray(arrays[f"process/{record['process_id']}/{name}"])
                for name in record["fields"]
            ),
        )
        for record in process_records
    )
    state = BalanceLawRuntimeState(transport_state, process_states)
    return BalanceLawCheckpoint(state, plan.checkpoint_id, payload_id)


__all__ = [
    "BalanceLawCheckpoint",
    "BalanceLawCheckpointPlan",
    "read_balance_law_checkpoint",
    "write_balance_law_checkpoint",
]
