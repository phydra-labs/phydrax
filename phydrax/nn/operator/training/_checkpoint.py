#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from ...._model import deserialize_model_leaf, serialize_model_leaf
from ...._model._structure import preflight_model_tree_serialization
from ...._training_checkpoint import (
    _deserialize_root_key,
    _open_verified_state,
    _prune_state_files,
    _publish_manifest,
    _publish_state,
    _read_manifest,
    _serialize_root_key,
)
from ._dtype import OperatorDTypePolicy
from ._fingerprint import operator_batch_schema
from ._normalization import OperatorNormalizationPolicy


_OPERATOR_TRAINING_CHECKPOINT_FORMAT = "phydrax-operator-training-checkpoint"


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
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer.")
    destination = Path(path)
    state_path, checksum = _publish_state(
        destination,
        lambda target: eqx.tree_serialise_leaves(
            target,
            (model, optimizer_state),
            filter_spec=serialize_model_leaf,
        ),
    )
    state_name = state_path.name
    manifest = {
        "format": _OPERATOR_TRAINING_CHECKPOINT_FORMAT,
        "state_file": state_name,
        "state_sha256": checksum,
        "step": int(step),
        **_serialize_root_key(key),
        "normalization": None if normalization is None else normalization.to_dict(),
        "dtype_policy": None if dtype_policy is None else dtype_policy.to_dict(),
        "schema": None if schema is None else dict(schema),
        "metadata": {} if metadata is None else dict(metadata),
    }
    _publish_manifest(destination / "manifest.json", manifest)
    _prune_state_files(destination, state_name)
    return destination


def _read_operator_training_manifest(
    path: str | Path,
    /,
) -> tuple[dict[str, Any], Path]:
    """Validate one current checkpoint manifest and its state checksum."""
    source = Path(path)
    manifest = _read_manifest(source / "manifest.json")
    expected = {
        "format",
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
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("Operator training checkpoint metadata must be an object.")
    if type(manifest["step"]) is not int or manifest["step"] < 0:
        raise ValueError("Operator training checkpoint step is invalid.")
    if any(
        value is not None and not isinstance(value, dict)
        for value in (
            manifest["normalization"],
            manifest["dtype_policy"],
            manifest["schema"],
        )
    ):
        raise ValueError("Operator training checkpoint policies are invalid.")
    key_impl = manifest["key_impl"]
    key_data = manifest["key_data"]
    key_words = {"threefry2x32": 2, "rbg": 4, "unsafe_rbg": 4}
    if (
        not isinstance(key_impl, str)
        or key_impl not in key_words
        or not isinstance(key_data, list)
        or len(key_data) != key_words[key_impl]
        or any(type(word) is not int or not 0 <= word <= 0xFFFFFFFF for word in key_data)
    ):
        raise ValueError("Operator training checkpoint root key is invalid.")
    state_name = manifest["state_file"]
    checksum = manifest["state_sha256"]
    if (
        not isinstance(state_name, str)
        or not state_name
        or "\\" in state_name
        or Path(state_name).name != state_name
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("Operator training checkpoint state identity is invalid.")
    return manifest, Path(state_name)


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
    manifest, state_name = _read_operator_training_manifest(source)
    if expected_schema is not None and manifest["schema"] != dict(expected_schema):
        raise ValueError("Operator training checkpoint schema mismatch.")
    with _open_verified_state(
        source,
        state_name.as_posix(),
        manifest["state_sha256"],
    ) as stream:
        try:
            preflight_model_tree_serialization(
                stream,
                (model_like, optimizer_state_like),
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "Operator training checkpoint leaf inventory is invalid."
            ) from error
        model, optimizer_state = eqx.tree_deserialise_leaves(
            stream,
            (model_like, optimizer_state_like),
            filter_spec=deserialize_model_leaf,
        )
        if stream.read(1):
            raise ValueError("Operator training checkpoint state has trailing payload.")
    key = _deserialize_root_key(manifest["key_data"], manifest["key_impl"])
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
