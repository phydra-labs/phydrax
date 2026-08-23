#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
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

from ._dtype import OperatorDTypePolicy
from ._fingerprint import operator_batch_schema
from ._normalization import OperatorNormalizationPolicy


_OPERATOR_TRAINING_CHECKPOINT_FORMAT = "phydrax-operator-training-checkpoint"
_OPERATOR_TRAINING_CHECKPOINT_VERSION = 3


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
        "format": _OPERATOR_TRAINING_CHECKPOINT_FORMAT,
        "version": _OPERATOR_TRAINING_CHECKPOINT_VERSION,
        "state_file": state_name,
        "state_sha256": checksum,
        "step": int(step),
        "key_data": np.asarray(jr.key_data(key)).tolist(),
        "key_impl": str(jr.key_impl(key)),
        "normalization": None if normalization is None else normalization.to_dict(),
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


def _read_operator_training_manifest(
    path: str | Path,
    /,
) -> tuple[dict[str, Any], Path]:
    """Validate one current checkpoint manifest and its state checksum."""
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "format",
        "version",
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
    if manifest["format"] != _OPERATOR_TRAINING_CHECKPOINT_FORMAT:
        raise ValueError("File is not a PhydraX operator training checkpoint.")
    if manifest["version"] != _OPERATOR_TRAINING_CHECKPOINT_VERSION:
        raise ValueError(
            "Operator training checkpoint version does not match the current runtime."
        )
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("Operator training checkpoint metadata must be an object.")
    state_name = manifest["state_file"]
    if not isinstance(state_name, str) or not state_name:
        raise ValueError("Operator training checkpoint state_file must be non-empty.")
    state_path = source / state_name
    actual_checksum = _sha256(state_path)
    if actual_checksum != manifest["state_sha256"]:
        raise ValueError("Operator training checkpoint state checksum mismatch.")
    return manifest, state_path


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
    manifest, state_path = _read_operator_training_manifest(source)
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
