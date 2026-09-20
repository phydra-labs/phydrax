#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._array_archive import (
    array_collection_digest,
    array_payload_byte_count,
    array_payload_digest,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations import MaterialSiteId, MaterialState, MaterialTransaction
from ..lifecycle import (
    CheckpointManifest,
    CheckpointShard,
    create as create_lifecycle_archive,
    open as open_lifecycle_archive,
)


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
    material_states = () if checkpoint.materials is None else checkpoint.materials.states
    metadata = {
        "kind": "finite-element-checkpoint",
        "field_count": len(checkpoint.field_state),
        "material_layout_id": (
            None if checkpoint.materials is None else checkpoint.materials.layout_id
        ),
        "material_payload_id": checkpoint.material_payload_id,
        "materials": [
            {
                "site_key": state.site_id.key,
                "model_id": state.model_id,
                "state_version": state.state_version,
            }
            for state in material_states
        ],
        "step": checkpoint.step,
    }
    arrays: dict[str, object] = {
        "time": np.asarray(checkpoint.time),
        **{
            f"field_{index}": np.asarray(value)
            for index, value in enumerate(checkpoint.field_state)
        },
        **{
            f"material_{index}": np.asarray(state.committed)
            for index, state in enumerate(material_states)
        },
    }
    metadata_text = json.dumps(metadata, allow_nan=False, sort_keys=True)
    shards = tuple(
        CheckpointShard(
            name,
            array_payload_digest(value),
            array_payload_byte_count(value),
            (checkpoint.prepared_id, checkpoint.compilation_id),
            metadata={"finite_element": metadata_text} if index == 0 else {},
        )
        for index, (name, value) in enumerate(sorted(arrays.items()))
    )
    manifest = CheckpointManifest(
        checkpoint.checkpoint_id,
        checkpoint.prepared_id,
        array_collection_digest(arrays),
        checkpoint.compilation_id,
        shards,
        complete=True,
    )
    create_lifecycle_archive(path, manifest=manifest, arrays=arrays)


def read_finite_element_checkpoint(
    path: str | Path,
    /,
    *,
    prepared_id: str,
    compilation_id: str,
) -> FiniteElementCheckpoint:
    archive = open_lifecycle_archive(path)
    manifest = archive.manifest
    if not isinstance(manifest, CheckpointManifest):
        raise ValueError("FE checkpoint is not a lifecycle checkpoint.")
    if (
        manifest.analysis_plan_id != prepared_id
        or manifest.execution_plan_id != compilation_id
    ):
        raise ValueError("FE checkpoint does not match the requested compiled problem.")
    metadata_text = next(
        (
            dict(shard.metadata)["finite_element"]
            for shard in manifest.shards
            if "finite_element" in dict(shard.metadata)
        ),
        None,
    )
    if metadata_text is None:
        raise ValueError("FE checkpoint metadata is missing.")
    metadata = json.loads(metadata_text)
    fields = tuple(
        archive.arrays[f"field_{index}"] for index in range(int(metadata["field_count"]))
    )
    material_states = tuple(
        MaterialState(
            MaterialSiteId(item["site_key"]),
            item["model_id"],
            archive.arrays[f"material_{index}"],
            state_version=int(item["state_version"]),
        )
        for index, item in enumerate(metadata["materials"])
    )
    materials = None if not material_states else MaterialTransaction(material_states)
    if materials is not None and materials.layout_id != metadata["material_layout_id"]:
        raise ValueError("FE checkpoint material layout identity mismatch.")
    if (
        materials is not None
        and materials.checkpoint_payload().payload_id != metadata["material_payload_id"]
    ):
        raise ValueError("FE checkpoint material payload identity mismatch.")
    checkpoint = FiniteElementCheckpoint(
        prepared_id,
        compilation_id,
        archive.arrays["time"],
        int(metadata["step"]),
        fields,
        materials=materials,
    )
    if checkpoint.checkpoint_id != manifest.checkpoint_id:
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
    shard = root / f"part-{partition:06d}-of-{count:06d}.phx"
    write_finite_element_checkpoint(shard, checkpoint)
    return shard


__all__ = [
    "FiniteElementCheckpoint",
    "read_finite_element_checkpoint",
    "write_finite_element_checkpoint",
    "write_partitioned_finite_element_checkpoint",
]
