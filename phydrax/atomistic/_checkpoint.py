#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._array_archive import read_array_archive, write_array_archive
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics


_CHECKPOINT_FORMAT = "phydrax-atomistic-dynamics-checkpoint"


def _pack_array_tree(
    prefix: str, tree: Any, arrays: dict[str, object], /
) -> dict[str, object]:
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    paths: list[str] = []
    names: list[str] = []
    for index, (path, leaf) in enumerate(path_leaves):
        value = np.asarray(leaf)
        if value.dtype.hasobject:
            raise TypeError("Atomistic checkpoint state cannot contain object arrays.")
        name = f"{prefix}/{index:06d}"
        paths.append(jax.tree_util.keystr(path) or "<root>")
        names.append(name)
        arrays[name] = value
    return {"paths": paths, "arrays": names, "num_leaves": len(names)}


def _unpack_array_tree(
    specification: Mapping[str, Any],
    arrays: Mapping[str, Any],
    template: Any,
    /,
) -> Any:
    template_path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    expected_paths = [
        jax.tree_util.keystr(path) or "<root>" for path, _ in template_path_leaves
    ]
    names = specification.get("arrays")
    if specification.get("paths") != expected_paths or not isinstance(names, list):
        raise ValueError("Checkpoint state tree does not match the runtime template.")
    leaves = []
    for name, (_, template_leaf) in zip(names, template_path_leaves, strict=True):
        if not isinstance(name, str) or name not in arrays:
            raise ValueError("Checkpoint state array is missing.")
        value = jnp.asarray(arrays[name])
        expected = jnp.asarray(template_leaf)
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError("Checkpoint state array shape or dtype changed.")
        leaves.append(value)
    return jax.tree_util.tree_unflatten(treedef, leaves)


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
    specification = _pack_array_tree("runtime", state, arrays)
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
    state = _unpack_array_tree(manifest["state"], arrays, template)
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
