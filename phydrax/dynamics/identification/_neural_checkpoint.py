#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from ..._training_checkpoint import (
    _deserialize_root_key,
    _prune_state_files,
    _publish_manifest,
    _publish_state,
    _read_manifest,
    _serialize_root_key,
    _verify_state,
)


_NEURAL_CHECKPOINT_FORMAT = "phydrax-discrete-model-training-checkpoint"


@dataclass(frozen=True, slots=True)
class _NeuralTrainingCheckpoint:
    model: Any
    optimizer_state: Any
    step: int
    key: Key[Array, ""]
    metadata: dict[str, Any]


def _save_neural_training_checkpoint(
    path: Path,
    model: Any,
    optimizer_state: Any,
    /,
    *,
    step: int,
    key: Key[Array, ""],
    metadata: dict[str, Any],
) -> None:
    if int(step) < 0:
        raise ValueError("step must be nonnegative.")
    state_path, checksum = _publish_state(
        path,
        lambda target: eqx.tree_serialise_leaves(
            target,
            (model, optimizer_state),
        ),
    )
    manifest = {
        "format": _NEURAL_CHECKPOINT_FORMAT,
        "state_file": state_path.name,
        "state_sha256": checksum,
        "step": int(step),
        **_serialize_root_key(key),
        "metadata": dict(metadata),
    }
    _publish_manifest(path / "manifest.json", manifest)
    _prune_state_files(path, state_path.name)


def _read_neural_training_manifest(path: Path, /) -> tuple[dict[str, Any], Path]:
    manifest = _read_manifest(path / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("Discrete-model checkpoint manifest must be an object.")
    expected = {
        "format",
        "state_file",
        "state_sha256",
        "step",
        "key_data",
        "key_impl",
        "metadata",
    }
    missing = expected - set(manifest)
    unknown = set(manifest) - expected
    if missing or unknown:
        raise ValueError(
            "Discrete-model checkpoint must use the canonical fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    if manifest["format"] != _NEURAL_CHECKPOINT_FORMAT:
        raise ValueError("File is not a PhydraX discrete-model training checkpoint.")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("Discrete-model checkpoint metadata must be an object.")
    state_name = manifest["state_file"]
    if not isinstance(state_name, str) or not state_name:
        raise ValueError("Discrete-model checkpoint state_file must be nonempty.")
    state_path = path / state_name
    _verify_state(state_path, manifest["state_sha256"])
    return manifest, state_path


def _load_neural_training_checkpoint(
    path: Path,
    model_like: Any,
    optimizer_state_like: Any,
    /,
) -> _NeuralTrainingCheckpoint:
    manifest, state_path = _read_neural_training_manifest(path)
    model, optimizer_state = eqx.tree_deserialise_leaves(
        state_path,
        (model_like, optimizer_state_like),
    )
    _prune_state_files(path, state_path.name)
    return _NeuralTrainingCheckpoint(
        model=model,
        optimizer_state=optimizer_state,
        step=int(manifest["step"]),
        key=_deserialize_root_key(manifest["key_data"], manifest["key_impl"]),
        metadata=manifest["metadata"],
    )


__all__: list[str] = []
