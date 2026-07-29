#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from ..models.core._operator import FunctionSamples, OperatorBatch, OperatorTargetBatch
from ..models.core._operator_topology import operator_topology_fingerprint
from ._dtype import OperatorDTypePolicy
from ._normalization import OperatorNormalizationPolicy


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _prune_superseded_states(directory: Path, current_state_name: str, /) -> None:
    for stale_state in directory.glob("state-*.eqx"):
        if stale_state.name != current_state_name:
            stale_state.unlink(missing_ok=True)
    (directory / "state.tmp.eqx").unlink(missing_ok=True)


def _samples_schema(samples: FunctionSamples, /) -> dict[str, Any]:
    values = None
    if samples.values is not None:
        array = jnp.asarray(samples.values)
        values = {"shape": list(array.shape), "dtype": str(array.dtype)}
    coordinate_dim = None
    if samples.coordinates is not None:
        coordinate_dim = int(samples.coordinates.shape[-1])
    elif samples.axes:
        coordinate_dim = len(samples.axes)
    topology = None
    if samples.topology is not None:
        topology = {
            "kind": samples.topology.kind,
            "site": samples.topology.site,
            "entity": samples.topology.entity,
            "case_shape": list(samples.topology.case_shape),
            "sample_shape": list(samples.topology.sample_shape),
            "num_graphs": samples.topology.graph.num_graphs,
            "entity_count": samples.topology.entity_count,
            "edge_count": int(samples.topology.graph.senders.shape[0])
            if samples.topology.graph.senders is not None
            else 0,
            "fingerprint": operator_topology_fingerprint(samples.topology),
        }
    return {
        "values": values,
        "sample_shape": list(samples.sample_shape),
        "coordinate_dim": coordinate_dim,
        "geometry_case_shape": list(samples.geometry_case_shape),
        "axes": [
            {
                "name": axis.name,
                "size": axis.size,
                "basis": axis.basis,
                "periodic": axis.periodic,
            }
            for axis in samples.axes
        ],
        "has_quadrature": samples.quadrature_weights is not None
        or any(axis.quadrature_weights is not None for axis in samples.axes),
        "has_mask": samples.mask is not None,
        "topology": topology,
    }


def operator_batch_schema(
    batch: OperatorBatch,
    /,
    *,
    target: OperatorTargetBatch | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe compatibility contract for an operator batch."""
    schema = {
        "case_axes": list(batch.case_axes),
        "case_shape": list(batch.case_shape),
        "inputs": {
            name: _samples_schema(samples) for name, samples in batch.inputs.items()
        },
        "queries": {
            name: _samples_schema(samples) for name, samples in batch.queries.items()
        },
    }
    if target is not None:
        target.validate(batch)
        schema["targets"] = {
            name: {
                "shape": list(field.values.shape),
                "dtype": str(field.values.dtype),
                "query_name": field.query_name,
                "channels": field.spec.channels,
                "component_names": list(field.spec.component_names),
            }
            for name, field in target.fields.items()
        }
    return schema


@dataclass(frozen=True)
class OperatorTrainingCheckpoint:
    """Fully restored state needed for bitwise-equivalent training continuation."""

    model: Any
    optimizer_state: Any
    step: int
    key: Key[Array, ""]
    normalization: OperatorNormalizationPolicy | None
    dtype_policy: OperatorDTypePolicy | None
    schema: Mapping[str, Any] | None
    metadata: Mapping[str, Any]


def save_operator_training_checkpoint(
    path: str | Path,
    model: Any,
    optimizer_state: Any,
    /,
    *,
    step: int,
    key: Key[Array, ""],
    normalization: OperatorNormalizationPolicy | None = None,
    dtype_policy: OperatorDTypePolicy | None = None,
    schema: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish an exact model/optimizer/RNG training checkpoint."""
    if int(step) < 0:
        raise ValueError("step must be non-negative.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_state = destination / "state.tmp.eqx"
    eqx.tree_serialise_leaves(temporary_state, (model, optimizer_state))
    checksum = _sha256(temporary_state)
    state_name = f"state-{checksum[:16]}.eqx"
    state_path = destination / state_name
    os.replace(temporary_state, state_path)
    manifest = {
        "state_file": state_name,
        "state_sha256": checksum,
        "step": int(step),
        "key_data": np.asarray(jr.key_data(key)).tolist(),
        "key_impl": str(jr.key_impl(key)),
        "normalization": (None if normalization is None else normalization.to_dict()),
        "dtype_policy": None if dtype_policy is None else dtype_policy.to_dict(),
        "schema": None if schema is None else dict(schema),
        "metadata": {} if metadata is None else dict(metadata),
    }
    temporary_manifest = destination / "manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, destination / "manifest.json")
    _prune_superseded_states(destination, state_name)
    return destination


def load_operator_training_checkpoint(
    path: str | Path,
    model_like: Any,
    optimizer_state_like: Any,
    /,
    *,
    expected_schema: Mapping[str, Any] | None = None,
) -> OperatorTrainingCheckpoint:
    """Verify and restore a checkpoint against explicit PyTree templates."""
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "state_file",
        "state_sha256",
        "step",
        "key_data",
        "key_impl",
        "normalization",
        "dtype_policy",
        "schema",
        "metadata",
    }
    if not isinstance(manifest, dict):
        raise ValueError("Operator training checkpoint manifest must be an object.")
    missing = expected - set(manifest)
    unknown = set(manifest) - expected
    if missing or unknown:
        raise ValueError(
            "Operator training checkpoint must use the current canonical fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    state_path = source / manifest["state_file"]
    actual_checksum = _sha256(state_path)
    if actual_checksum != manifest["state_sha256"]:
        raise ValueError("Operator training checkpoint state checksum mismatch.")
    if expected_schema is not None and manifest["schema"] != dict(expected_schema):
        raise ValueError("Operator training checkpoint schema mismatch.")
    model, optimizer_state = eqx.tree_deserialise_leaves(
        state_path,
        (model_like, optimizer_state_like),
    )
    _prune_superseded_states(source, state_path.name)
    key = jr.wrap_key_data(
        jnp.asarray(manifest["key_data"], dtype=jnp.uint32),
        impl=manifest["key_impl"],
    )
    normalization = manifest["normalization"]
    dtype_policy = manifest["dtype_policy"]
    return OperatorTrainingCheckpoint(
        model=model,
        optimizer_state=optimizer_state,
        step=int(manifest["step"]),
        key=key,
        normalization=(
            None
            if normalization is None
            else OperatorNormalizationPolicy.from_dict(normalization)
        ),
        dtype_policy=(
            None if dtype_policy is None else OperatorDTypePolicy.from_dict(dtype_policy)
        ),
        schema=manifest["schema"],
        metadata=manifest["metadata"],
    )


__all__ = [
    "OperatorTrainingCheckpoint",
    "load_operator_training_checkpoint",
    "operator_batch_schema",
    "save_operator_training_checkpoint",
]
