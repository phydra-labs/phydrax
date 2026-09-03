#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from ..._training import TrainingProgress
from ..._training_checkpoint import (
    _deserialize_root_key,
    _prune_state_files,
    _publish_manifest,
    _publish_state,
    _read_manifest,
    _serialize_root_key,
    _verify_state,
)


_VARIATIONAL_CHECKPOINT_FORMAT = "phydrax-variational-kinetic-training-checkpoint"


@dataclass(frozen=True, slots=True)
class _VariationalTrainingCheckpoint:
    model: Any
    optimizer_state: Any
    best_model: Any
    progress: TrainingProgress
    step: int
    key: Key[Array, ""]
    metadata: dict[str, Any]
    history: dict[str, list[Any]]


def _save_variational_training_checkpoint(
    path: Path,
    model: Any,
    optimizer_state: Any,
    best_model: Any,
    progress: TrainingProgress,
    /,
    *,
    step: int,
    key: Key[Array, ""],
    metadata: dict[str, Any],
    history: dict[str, list[Any]],
) -> None:
    if int(step) < 0:
        raise ValueError("step must be nonnegative.")
    state_path, checksum = _publish_state(
        path,
        lambda target: eqx.tree_serialise_leaves(
            target, (model, optimizer_state, best_model, progress)
        ),
    )
    manifest = {
        "format": _VARIATIONAL_CHECKPOINT_FORMAT,
        "state_file": state_path.name,
        "state_sha256": checksum,
        "step": int(step),
        **_serialize_root_key(key),
        "metadata": dict(metadata),
        "history": dict(history),
    }
    _publish_manifest(path / "manifest.json", manifest)
    _prune_state_files(path, state_path.name)


def _load_variational_training_checkpoint(
    path: Path,
    model_like: Any,
    optimizer_state_like: Any,
    best_model_like: Any,
    progress_like: TrainingProgress,
    /,
) -> _VariationalTrainingCheckpoint:
    manifest = _read_manifest(path / "manifest.json")
    expected = {
        "format",
        "state_file",
        "state_sha256",
        "step",
        "key_data",
        "key_impl",
        "metadata",
        "history",
    }
    if not isinstance(manifest, dict):
        raise TypeError("Variational kinetic checkpoint manifest must be an object.")
    if set(manifest) != expected:
        raise ValueError("Variational kinetic checkpoint fields are not canonical.")
    if manifest["format"] != _VARIATIONAL_CHECKPOINT_FORMAT:
        raise ValueError("File is not a variational kinetic training checkpoint.")
    if not isinstance(manifest["metadata"], dict):
        raise TypeError("Variational checkpoint metadata must be an object.")
    if not isinstance(manifest["history"], dict):
        raise TypeError("Variational checkpoint history must be an object.")
    state_name = manifest["state_file"]
    if not isinstance(state_name, str) or not state_name:
        raise ValueError("Variational checkpoint state_file must be nonempty.")
    state_path = path / state_name
    _verify_state(state_path, manifest["state_sha256"])
    model, optimizer_state, best_model, progress = eqx.tree_deserialise_leaves(
        state_path,
        (model_like, optimizer_state_like, best_model_like, progress_like),
    )
    _prune_state_files(path, state_path.name)
    return _VariationalTrainingCheckpoint(
        model=model,
        optimizer_state=optimizer_state,
        best_model=best_model,
        progress=progress,
        step=int(manifest["step"]),
        key=_deserialize_root_key(manifest["key_data"], manifest["key_impl"]),
        metadata=manifest["metadata"],
        history=manifest["history"],
    )


__all__: list[str] = []
