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
from ._spin_dynamics import ClassicalSpinDynamicsState, PreparedLandauLifshitzGilbert
from ._units import AtomisticUnitSystem


_CHECKPOINT_FORMAT = "phydrax-atomistic-spin-checkpoint"


class AtomisticSpinCheckpointPlan(StrictModule, NonTrainableState):
    dynamics: PreparedLandauLifshitzGilbert
    scope_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedLandauLifshitzGilbert,
        /,
        *,
        scope_id: str | None = None,
    ):
        if not isinstance(dynamics, PreparedLandauLifshitzGilbert):
            raise TypeError("dynamics must be PreparedLandauLifshitzGilbert.")
        if scope_id is not None and (
            not isinstance(scope_id, str) or not scope_id or scope_id != scope_id.strip()
        ):
            raise ValueError("scope_id must be a canonical nonempty string or None.")
        self.dynamics = dynamics
        self.scope_id = scope_id
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "atomistic-spin-checkpoint-plan",
                "dynamics": dynamics.prepared_id,
                "hamiltonian": dynamics.plan.hamiltonian.prepared_id,
                **({} if scope_id is None else {"scope_id": scope_id}),
            }
        )


class AtomisticSpinCheckpoint(StrictModule):
    state: ClassicalSpinDynamicsState
    payload_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)


def write_atomistic_spin_checkpoint(
    path: str | Path,
    plan: AtomisticSpinCheckpointPlan,
    state: ClassicalSpinDynamicsState,
    /,
) -> AtomisticSpinCheckpoint:
    if not isinstance(plan, AtomisticSpinCheckpointPlan):
        raise TypeError("plan must be AtomisticSpinCheckpointPlan.")
    if not isinstance(state, ClassicalSpinDynamicsState):
        raise TypeError("state must be ClassicalSpinDynamicsState.")
    if state.prepared_dynamics_id != plan.dynamics.prepared_id:
        raise ValueError("Spin checkpoint state belongs to another dynamics runtime.")
    arrays: dict[str, object] = {}
    specification = pack_array_tree("spin-runtime", state, arrays)
    manifest = {
        "format": _CHECKPOINT_FORMAT,
        "kind": "atomistic-spin-runtime",
        "checkpoint_id": plan.checkpoint_id,
        "prepared_dynamics_id": plan.dynamics.prepared_id,
        "hamiltonian_id": plan.dynamics.plan.hamiltonian.prepared_id,
        "unit_system": plan.dynamics.plan.hamiltonian.plan.system.plan.units.to_dict(),
        "state": specification,
        **({} if plan.scope_id is None else {"scope_id": plan.scope_id}),
    }
    payload_id = canonical_fingerprint(
        {
            "kind": "atomistic-spin-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "time": float(state.time),
            "step": int(state.step_index),
            "wiener": state.wiener_realization_id,
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    write_array_archive(
        path, manifest={**manifest, "payload_id": payload_id}, arrays=arrays
    )
    return AtomisticSpinCheckpoint(state, payload_id, plan.checkpoint_id)


def read_atomistic_spin_checkpoint(
    path: str | Path,
    plan: AtomisticSpinCheckpointPlan,
    template: ClassicalSpinDynamicsState,
    /,
) -> AtomisticSpinCheckpoint:
    if not isinstance(plan, AtomisticSpinCheckpointPlan):
        raise TypeError("plan must be AtomisticSpinCheckpointPlan.")
    if not isinstance(template, ClassicalSpinDynamicsState):
        raise TypeError("template must be ClassicalSpinDynamicsState.")
    if template.prepared_dynamics_id != plan.dynamics.prepared_id:
        raise ValueError("Spin checkpoint template belongs to another dynamics runtime.")
    manifest, arrays = read_array_archive(path)
    expected = {
        "format",
        "kind",
        "checkpoint_id",
        "prepared_dynamics_id",
        "hamiltonian_id",
        "unit_system",
        "state",
        "payload_id",
        "arrays",
    }
    if plan.scope_id is not None:
        expected.add("scope_id")
    if set(manifest) != expected:
        raise ValueError("Spin checkpoint manifest is not the canonical current format.")
    identities = {
        "format": _CHECKPOINT_FORMAT,
        "kind": "atomistic-spin-runtime",
        "checkpoint_id": plan.checkpoint_id,
        "prepared_dynamics_id": plan.dynamics.prepared_id,
        "hamiltonian_id": plan.dynamics.plan.hamiltonian.prepared_id,
    }
    if plan.scope_id is not None:
        identities["scope_id"] = plan.scope_id
    for name, expected_value in identities.items():
        if manifest[name] != expected_value:
            raise ValueError(f"Spin checkpoint {name} does not match the runtime.")
    units = AtomisticUnitSystem.from_dict(manifest["unit_system"])
    expected_units = plan.dynamics.plan.hamiltonian.plan.system.plan.units
    if units.unit_system_id != expected_units.unit_system_id:
        raise ValueError("Spin checkpoint unit system does not match the runtime.")
    state = unpack_array_tree(manifest["state"], arrays, template)
    if not isinstance(state, ClassicalSpinDynamicsState):
        raise TypeError("Checkpoint did not reconstruct ClassicalSpinDynamicsState.")
    payload_id = str(manifest["payload_id"])
    expected_payload = canonical_fingerprint(
        {
            "kind": "atomistic-spin-checkpoint-payload",
            "checkpoint": plan.checkpoint_id,
            "time": float(state.time),
            "step": int(state.step_index),
            "wiener": state.wiener_realization_id,
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if payload_id != expected_payload:
        raise ValueError("Spin checkpoint payload identity is corrupt.")
    return AtomisticSpinCheckpoint(state, payload_id, plan.checkpoint_id)


__all__ = [
    "AtomisticSpinCheckpoint",
    "AtomisticSpinCheckpointPlan",
    "read_atomistic_spin_checkpoint",
    "write_atomistic_spin_checkpoint",
]
