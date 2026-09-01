#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx

from .._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics


_CHECKPOINT_FORMAT = "phydrax-atomistic-dynamics-checkpoint"


class AtomisticCheckpointPlan(StrictModule, NonTrainableState):
    dynamics: PreparedAtomisticDynamics
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedAtomisticDynamics, /):
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        self.dynamics = dynamics
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "atomistic-checkpoint-plan",
                "dynamics": dynamics.prepared_id,
                "system": dynamics.system.prepared_id,
                "potential": dynamics.potential.prepared_id,
                "integrator": dynamics.integrator.plan_id,
            }
        )


class AtomisticCheckpoint(StrictModule):
    state: AtomisticDynamicsState
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


def write_atomistic_checkpoint(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    state: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    if not isinstance(plan, AtomisticCheckpointPlan):
        raise TypeError("plan must be AtomisticCheckpointPlan.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("state must be AtomisticDynamicsState.")
    if state.prepared_dynamics_id != plan.dynamics.prepared_id:
        raise ValueError("Checkpoint state belongs to another dynamics runtime.")
    arrays: dict[str, object] = {}
    specification = pack_array_tree("runtime", state, arrays)
    manifest = {
        "format": _CHECKPOINT_FORMAT,
        "kind": "atomistic-dynamics-runtime",
        "checkpoint_id": plan.checkpoint_id,
        "prepared_dynamics_id": plan.dynamics.prepared_id,
        "system_id": plan.dynamics.system.prepared_id,
        "potential_id": plan.dynamics.potential.prepared_id,
        "integrator_id": plan.dynamics.integrator.plan_id,
        "state": specification,
    }
    payload_id = canonical_fingerprint(
        {
            "kind": "atomistic-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "time": float(state.time),
            "step": int(state.step_index),
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    write_array_archive(
        path,
        manifest={**manifest, "payload_id": payload_id},
        arrays=arrays,
    )
    return AtomisticCheckpoint(state, payload_id, plan.checkpoint_id)


def read_atomistic_checkpoint(
    path: str | Path,
    plan: AtomisticCheckpointPlan,
    template: AtomisticDynamicsState,
    /,
) -> AtomisticCheckpoint:
    if not isinstance(plan, AtomisticCheckpointPlan):
        raise TypeError("plan must be AtomisticCheckpointPlan.")
    if not isinstance(template, AtomisticDynamicsState):
        raise TypeError("template must be AtomisticDynamicsState.")
    if template.prepared_dynamics_id != plan.dynamics.prepared_id:
        raise ValueError("Checkpoint template belongs to another dynamics runtime.")
    manifest, arrays = read_array_archive(path)
    expected = {
        "format",
        "kind",
        "checkpoint_id",
        "prepared_dynamics_id",
        "system_id",
        "potential_id",
        "integrator_id",
        "state",
        "payload_id",
        "arrays",
    }
    if set(manifest) != expected:
        raise ValueError(
            "Atomistic checkpoint manifest is not the canonical current format."
        )
    if (
        manifest["format"] != _CHECKPOINT_FORMAT
        or manifest["kind"] != "atomistic-dynamics-runtime"
    ):
        raise ValueError("File is not an atomistic dynamics checkpoint.")
    identities = {
        "checkpoint_id": plan.checkpoint_id,
        "prepared_dynamics_id": plan.dynamics.prepared_id,
        "system_id": plan.dynamics.system.prepared_id,
        "potential_id": plan.dynamics.potential.prepared_id,
        "integrator_id": plan.dynamics.integrator.plan_id,
    }
    for name, expected_value in identities.items():
        if manifest[name] != expected_value:
            raise ValueError(f"Atomistic checkpoint {name} does not match the runtime.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("Checkpoint did not reconstruct AtomisticDynamicsState.")
    payload_id = str(manifest["payload_id"])
    if not payload_id:
        raise ValueError("Checkpoint payload_id is empty.")
    expected_payload_id = canonical_fingerprint(
        {
            "kind": "atomistic-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "time": float(state.time),
            "step": int(state.step_index),
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if payload_id != expected_payload_id:
        raise ValueError("Atomistic checkpoint payload identity is corrupt.")
    return AtomisticCheckpoint(state, payload_id, plan.checkpoint_id)


__all__ = [
    "AtomisticCheckpoint",
    "AtomisticCheckpointPlan",
    "read_atomistic_checkpoint",
    "write_atomistic_checkpoint",
]
