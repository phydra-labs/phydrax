#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class CoupledFieldCheckpointPlan(StrictModule, NonTrainableState):
    runtime_id: str = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    geometry_id: str | None = eqx.field(static=True)
    topology_id: str | None = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_id: str,
        program_id: str,
        field_names,
        /,
        *,
        geometry_id: str | None = None,
        topology_id: str | None = None,
    ):
        runtime = str(runtime_id)
        program = str(program_id)
        fields = tuple(str(value) for value in field_names)
        if (
            not runtime
            or not program
            or not fields
            or any(not value for value in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError("Checkpoint runtime/program/field identities are invalid.")
        self.runtime_id = runtime
        self.program_id = program
        self.geometry_id = None if geometry_id is None else str(geometry_id)
        self.topology_id = None if topology_id is None else str(topology_id)
        self.field_names = fields
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-field-checkpoint",
                "runtime": runtime,
                "program": program,
                "geometry": self.geometry_id,
                "topology": self.topology_id,
                "fields": list(fields),
            }
        )


class CoupledFieldCheckpoint(StrictModule):
    time: Any
    step_index: Any
    state: Any
    runtime_args: Any
    plan_id: str = eqx.field(static=True)


def write_coupled_field_checkpoint(
    path: str | Path,
    plan: CoupledFieldCheckpointPlan,
    time: Any,
    step_index: Any,
    state: Any,
    /,
    *,
    runtime_args: Any = None,
) -> None:
    if not isinstance(plan, CoupledFieldCheckpointPlan):
        raise TypeError("plan must be CoupledFieldCheckpointPlan.")
    time_ = jnp.asarray(time)
    step_ = jnp.asarray(step_index)
    if (
        time_.shape != ()
        or not jnp.issubdtype(time_.dtype, jnp.inexact)
        or not bool(jnp.isfinite(time_))
    ):
        raise ValueError("Checkpoint time must be one finite inexact scalar.")
    if step_.shape != () or step_.dtype.kind not in "iu" or bool(step_ < 0):
        raise ValueError("Checkpoint step index must be one nonnegative integer.")
    arrays: dict[str, object] = {"time": time_, "step_index": step_}
    state_spec = pack_array_tree("state", state, arrays)
    args_spec = pack_array_tree("runtime_args", runtime_args, arrays)
    payload_id = array_tree_fingerprint(arrays)
    manifest = {
        "kind": "coupled-field-checkpoint",
        "plan_id": plan.plan_id,
        "runtime_id": plan.runtime_id,
        "program_id": plan.program_id,
        "geometry_id": plan.geometry_id,
        "topology_id": plan.topology_id,
        "field_names": list(plan.field_names),
        "state": state_spec,
        "runtime_args": args_spec,
        "payload_id": payload_id,
    }
    write_array_archive(path, manifest=manifest, arrays=arrays)


def read_coupled_field_checkpoint(
    path: str | Path,
    plan: CoupledFieldCheckpointPlan,
    state_template: Any,
    /,
    *,
    runtime_args_template: Any = None,
) -> CoupledFieldCheckpoint:
    if not isinstance(plan, CoupledFieldCheckpointPlan):
        raise TypeError("plan must be CoupledFieldCheckpointPlan.")
    manifest, arrays = read_array_archive(path)
    expected = {
        "kind": "coupled-field-checkpoint",
        "plan_id": plan.plan_id,
        "runtime_id": plan.runtime_id,
        "program_id": plan.program_id,
        "geometry_id": plan.geometry_id,
        "topology_id": plan.topology_id,
        "field_names": list(plan.field_names),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Coupled-field checkpoint {key} is incompatible.")
    if manifest.get("payload_id") != array_tree_fingerprint(arrays):
        raise ValueError("Coupled-field checkpoint payload fingerprint is invalid.")
    time = jnp.asarray(arrays["time"])
    step = jnp.asarray(arrays["step_index"])
    state = unpack_array_tree(manifest["state"], arrays, state_template)
    runtime_args = unpack_array_tree(
        manifest["runtime_args"], arrays, runtime_args_template
    )
    return CoupledFieldCheckpoint(
        time,
        step,
        state,
        runtime_args,
        plan.plan_id,
    )


__all__ = [
    "CoupledFieldCheckpoint",
    "CoupledFieldCheckpointPlan",
    "read_coupled_field_checkpoint",
    "write_coupled_field_checkpoint",
]
