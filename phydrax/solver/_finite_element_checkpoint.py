#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations import MaterialSiteId, MaterialState, MaterialTransaction


_CHECKPOINT_VERSION = 1


class FiniteElementCheckpoint(StrictModule, NonTrainableState):
    """Portable accepted FE field/material state bound to compiled identities."""

    prepared_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    time: Array
    step: int = eqx.field(static=True)
    field_state: tuple[Array, ...]
    materials: MaterialTransaction | None
    material_payload_id: str | None = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_id: str,
        compilation_id: str,
        time: ArrayLike,
        step: int,
        field_state: tuple[ArrayLike, ...],
        /,
        *,
        materials: MaterialTransaction | None = None,
    ):
        prepared = str(prepared_id)
        compiled = str(compilation_id)
        time_ = jnp.asarray(time)
        step_ = int(step)
        fields = tuple(jnp.asarray(value) for value in field_state)
        material_state = materials
        if not prepared or not compiled or time_.shape != () or step_ < 0:
            raise ValueError("FE checkpoint identity, time, or step is invalid.")
        if not fields or not all(
            jnp.issubdtype(value.dtype, jnp.inexact) for value in fields
        ):
            raise ValueError("FE checkpoint requires one or more inexact field arrays.")
        if material_state is not None and not isinstance(
            material_state, MaterialTransaction
        ):
            raise TypeError("materials must be a MaterialTransaction or None.")
        self.prepared_id = prepared
        self.compilation_id = compiled
        self.time = time_
        self.step = step_
        self.field_state = fields
        self.materials = material_state
        material_payload = (
            None if material_state is None else material_state.checkpoint_payload()
        )
        self.material_payload_id = (
            None if material_payload is None else material_payload.payload_id
        )
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "finite-element-checkpoint",
                "version": _CHECKPOINT_VERSION,
                "prepared_id": prepared,
                "compilation_id": compiled,
                "time": array_tree_fingerprint(np.asarray(time_)),
                "step": step_,
                "fields": [array_tree_fingerprint(np.asarray(value)) for value in fields],
                "materials": self.material_payload_id,
            }
        )


def write_finite_element_checkpoint(
    path: str | Path,
    checkpoint: FiniteElementCheckpoint,
    /,
) -> None:
    if not isinstance(checkpoint, FiniteElementCheckpoint):
        raise TypeError("checkpoint must be FiniteElementCheckpoint.")
    metadata = {
        "version": _CHECKPOINT_VERSION,
        "prepared_id": checkpoint.prepared_id,
        "compilation_id": checkpoint.compilation_id,
        "time": float(checkpoint.time),
        "step": checkpoint.step,
        "checkpoint_id": checkpoint.checkpoint_id,
        "field_count": len(checkpoint.field_state),
        "material_layout_id": (
            None if checkpoint.materials is None else checkpoint.materials.layout_id
        ),
        "material_payload_id": checkpoint.material_payload_id,
        "materials": (
            []
            if checkpoint.materials is None
            else [
                {
                    "site_key": state.site_id.key,
                    "model_id": state.model_id,
                    "state_version": state.state_version,
                }
                for state in checkpoint.materials.states
            ]
        ),
    }
    arrays: dict[str, Any] = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update(
        {
            f"field_{index}": np.asarray(value)
            for index, value in enumerate(checkpoint.field_state)
        }
    )
    if checkpoint.materials is not None:
        for index, state in enumerate(checkpoint.materials.states):
            arrays[f"material_{index}"] = np.asarray(state.committed)
    np.savez(Path(path), **arrays)


def read_finite_element_checkpoint(
    path: str | Path,
    /,
    *,
    prepared_id: str,
    compilation_id: str,
) -> FiniteElementCheckpoint:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["version"] != _CHECKPOINT_VERSION:
            raise ValueError("Unsupported FE checkpoint version.")
        if (
            metadata["prepared_id"] != prepared_id
            or metadata["compilation_id"] != compilation_id
        ):
            raise ValueError(
                "FE checkpoint does not match the requested compiled problem."
            )
        fields = tuple(
            archive[f"field_{index}"] for index in range(int(metadata["field_count"]))
        )
        material_states = tuple(
            MaterialState(
                MaterialSiteId(item["site_key"]),
                item["model_id"],
                archive[f"material_{index}"],
                state_version=int(item["state_version"]),
            )
            for index, item in enumerate(metadata["materials"])
        )
        materials = None if not material_states else MaterialTransaction(material_states)
        if (
            materials is not None
            and materials.layout_id != metadata["material_layout_id"]
        ):
            raise ValueError("FE checkpoint material layout identity mismatch.")
        if (
            materials is not None
            and materials.checkpoint_payload().payload_id
            != metadata["material_payload_id"]
        ):
            raise ValueError("FE checkpoint material payload identity mismatch.")
    checkpoint = FiniteElementCheckpoint(
        prepared_id,
        compilation_id,
        metadata["time"],
        int(metadata["step"]),
        fields,
        materials=materials,
    )
    if checkpoint.checkpoint_id != metadata["checkpoint_id"]:
        raise ValueError("FE checkpoint content identity mismatch.")
    return checkpoint


def write_partitioned_finite_element_checkpoint(
    directory: str | Path,
    partition_id: int,
    partition_count: int,
    checkpoint: FiniteElementCheckpoint,
    /,
) -> Path:
    """Write one deterministic partition shard and return its path."""

    root = Path(directory)
    partition = int(partition_id)
    count = int(partition_count)
    if partition < 0 or count <= 0 or partition >= count:
        raise ValueError("Partition checkpoint IDs are invalid.")
    root.mkdir(parents=True, exist_ok=True)
    shard = root / f"part-{partition:06d}-of-{count:06d}.npz"
    write_finite_element_checkpoint(shard, checkpoint)
    return shard


__all__ = [
    "FiniteElementCheckpoint",
    "read_finite_element_checkpoint",
    "write_finite_element_checkpoint",
    "write_partitioned_finite_element_checkpoint",
]
