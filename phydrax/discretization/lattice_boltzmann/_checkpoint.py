#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._program import KineticProgramManifest


_KINETIC_CHECKPOINT_FORMAT = "phydrax-kinetic-checkpoint"


class KineticCheckpointPlan(StrictModule, NonTrainableState):
    """Exact compatibility identity for one prepared kinetic runtime."""

    program_manifest: KineticProgramManifest
    runtime_id: str = eqx.field(static=True)
    geometry_epoch_id: str | None = eqx.field(static=True)
    topology_id: str | None = eqx.field(static=True)
    execution_id: str | None = eqx.field(static=True)
    replay_policy_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_id: str,
        program_manifest: KineticProgramManifest,
        /,
        *,
        geometry_epoch_id: str | None = None,
        topology_id: str | None = None,
        execution_id: str | None = None,
        replay_policy_id: str | None = None,
    ):
        runtime = str(runtime_id)
        if not runtime:
            raise ValueError("runtime_id must be nonempty.")
        if not isinstance(program_manifest, KineticProgramManifest):
            raise TypeError("program_manifest must be KineticProgramManifest.")
        optional = tuple(
            None if value is None else str(value)
            for value in (
                geometry_epoch_id,
                topology_id,
                execution_id,
                replay_policy_id,
            )
        )
        if any(value == "" for value in optional):
            raise ValueError("Optional kinetic checkpoint identities must be nonempty.")
        self.runtime_id = runtime
        self.program_manifest = program_manifest
        (
            self.geometry_epoch_id,
            self.topology_id,
            self.execution_id,
            self.replay_policy_id,
        ) = optional
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-checkpoint-plan",
                "runtime": runtime,
                "program_manifest": program_manifest.manifest_id,
                "checkpoint_fields": program_manifest.checkpoint_fields,
                "geometry_epoch": optional[0],
                "topology": optional[1],
                "execution": optional[2],
                "replay": optional[3],
            }
        )


class KineticCheckpoint(StrictModule):
    time: Any
    step_index: Any
    state: Any
    args: Any
    payload_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _validate_clock(time: Any, step_index: Any, /) -> tuple[Any, Any]:
    time_ = jnp.asarray(time)
    step_ = jnp.asarray(step_index)
    if (
        time_.shape != ()
        or not jnp.issubdtype(time_.dtype, jnp.inexact)
        or not bool(jnp.isfinite(time_))
    ):
        raise ValueError("Kinetic checkpoint time must be one finite inexact scalar.")
    if step_.shape != () or step_.dtype.kind not in "iu" or bool(step_ < 0):
        raise ValueError(
            "Kinetic checkpoint step index must be one nonnegative integer scalar."
        )
    return time_, step_


def _compatibility(plan: KineticCheckpointPlan, /) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "runtime_id": plan.runtime_id,
        "program_manifest_id": plan.program_manifest.manifest_id,
        "lattice_id": plan.program_manifest.lattice_id,
        "precision_policy_id": plan.program_manifest.precision_policy_id,
        "geometry_epoch_id": plan.geometry_epoch_id,
        "topology_id": plan.topology_id,
        "execution_id": plan.execution_id,
        "replay_policy_id": plan.replay_policy_id,
        "checkpoint_fields": list(plan.program_manifest.checkpoint_fields),
    }


def write_kinetic_checkpoint(
    path: str | Path,
    plan: KineticCheckpointPlan,
    time: Any,
    step_index: Any,
    state: Any,
    /,
    *,
    args: Any = None,
) -> KineticCheckpoint:
    if not isinstance(plan, KineticCheckpointPlan):
        raise TypeError("plan must be KineticCheckpointPlan.")
    time_, step_ = _validate_clock(time, step_index)
    arrays: dict[str, object] = {
        "time": time_,
        "step_index": step_,
    }
    state_specification = pack_array_tree("state", state, arrays)
    args_specification = None if args is None else pack_array_tree("args", args, arrays)
    compatibility = _compatibility(plan)
    payload_id = canonical_fingerprint(
        {
            "kind": "kinetic-checkpoint-payload",
            "compatibility": compatibility,
            "state": state_specification,
            "args": args_specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    write_array_archive(
        path,
        manifest={
            "format": _KINETIC_CHECKPOINT_FORMAT,
            "kind": "kinetic-runtime",
            **compatibility,
            "state": state_specification,
            "args": args_specification,
            "payload_id": payload_id,
        },
        arrays=arrays,
    )
    return KineticCheckpoint(time_, step_, state, args, payload_id, plan.plan_id)


def read_kinetic_checkpoint(
    path: str | Path,
    plan: KineticCheckpointPlan,
    state_template: Any,
    /,
    *,
    args_template: Any = None,
) -> KineticCheckpoint:
    if not isinstance(plan, KineticCheckpointPlan):
        raise TypeError("plan must be KineticCheckpointPlan.")
    manifest, arrays = read_array_archive(path)
    expected_keys = {
        "format",
        "kind",
        "plan_id",
        "runtime_id",
        "program_manifest_id",
        "lattice_id",
        "precision_policy_id",
        "geometry_epoch_id",
        "topology_id",
        "execution_id",
        "replay_policy_id",
        "checkpoint_fields",
        "state",
        "args",
        "payload_id",
        "arrays",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            "Kinetic checkpoint manifest is not the canonical current format."
        )
    if (
        manifest["format"] != _KINETIC_CHECKPOINT_FORMAT
        or manifest["kind"] != "kinetic-runtime"
    ):
        raise ValueError("File is not a kinetic runtime checkpoint.")
    compatibility = _compatibility(plan)
    for name, expected in compatibility.items():
        if manifest[name] != expected:
            raise ValueError(f"Kinetic checkpoint {name} does not match the runtime.")
    state = unpack_array_tree(manifest["state"], arrays, state_template)
    args_specification = manifest["args"]
    if (args_specification is None) != (args_template is None):
        raise ValueError("Kinetic checkpoint runtime-argument presence changed.")
    args = (
        None
        if args_specification is None
        else unpack_array_tree(args_specification, arrays, args_template)
    )
    time, step_index = _validate_clock(arrays["time"], arrays["step_index"])
    payload_id = str(manifest["payload_id"])
    expected_payload = canonical_fingerprint(
        {
            "kind": "kinetic-checkpoint-payload",
            "compatibility": compatibility,
            "state": manifest["state"],
            "args": args_specification,
            "arrays": array_tree_fingerprint(
                {name: value for name, value in arrays.items() if name != "__unused__"}
            ),
        }
    )
    if not payload_id or payload_id != expected_payload:
        raise ValueError("Kinetic checkpoint payload identity is invalid.")
    return KineticCheckpoint(
        time,
        step_index,
        state,
        args,
        payload_id,
        plan.plan_id,
    )


__all__ = [
    "KineticCheckpoint",
    "KineticCheckpointPlan",
    "read_kinetic_checkpoint",
    "write_kinetic_checkpoint",
]
