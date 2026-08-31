#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_mapping,
)
from ..._model import (
    artifact_value_id,
    model_from_structure_recipe,
    model_structure_recipe,
)


_NATIVE_FORMAT = "phydrax-native-velocimetry"
_NATIVE_MANIFEST_FIELDS = {
    "format",
    "archive_id",
    "value_kind",
    "value_type",
    "structure",
    "leaves",
    "provenance",
    "arrays",
}


@dataclass(frozen=True, slots=True)
class VelocimetryArchive:
    """Restored native value and checksum-verified archive metadata."""

    value: Any
    archive_id: str
    value_kind: str
    value_type: str
    provenance: Mapping[str, Any]


def write_velocimetry_archive(
    path: str | Path,
    value: Any,
    /,
    *,
    value_kind: str,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a registered native value as exact structure plus array leaves."""
    kind = str(value_kind).strip()
    if not kind:
        raise ValueError("value_kind must be non-empty.")
    value_type = artifact_value_id(type(value))
    structure = model_structure_recipe(value, path="value")
    template = model_from_structure_recipe(structure)
    leaves, tree = jax.tree_util.tree_flatten_with_path(value)
    template_leaves, template_tree = jax.tree_util.tree_flatten_with_path(template)
    if tree != template_tree or len(leaves) != len(template_leaves):
        raise TypeError("Native value structure cannot be reconstructed canonically.")

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for index, ((path_key, leaf), (template_path, template_leaf)) in enumerate(
        zip(leaves, template_leaves, strict=True)
    ):
        if path_key != template_path:
            raise TypeError("Native value leaf paths are not structurally stable.")
        if not _is_array(leaf):
            continue
        name = f"leaves/{len(records):06d}"
        payload, backend = _array_payload(leaf)
        arrays[name] = payload
        records.append(
            {
                "name": name,
                "tree_index": index,
                "path": jax.tree_util.keystr(path_key) or "<root>",
                "shape": list(payload.shape),
                "dtype": payload.dtype.str,
                "backend": backend,
            }
        )
        if not _is_array(template_leaf):
            raise TypeError("Native value array leaves do not match their recipe.")

    provenance_ = canonical_mapping(dict(provenance or {}))
    identity_payload = {
        "format": _NATIVE_FORMAT,
        "value_kind": kind,
        "value_type": value_type,
        "structure": structure,
        "leaves": records,
        "provenance": provenance_,
        "content": array_tree_fingerprint(arrays),
    }
    archive_id = canonical_fingerprint(identity_payload)
    return write_array_archive(
        path,
        manifest={
            "format": _NATIVE_FORMAT,
            "archive_id": archive_id,
            "value_kind": kind,
            "value_type": value_type,
            "structure": structure,
            "leaves": records,
            "provenance": provenance_,
        },
        arrays=arrays,
    )


def read_velocimetry_archive(
    path: str | Path,
    /,
    *,
    expected_kind: str | None = None,
    expected_type: type | None = None,
) -> VelocimetryArchive:
    """Read an exact native value, rejecting manifest or leaf inconsistencies."""
    manifest, arrays = read_array_archive(path)
    if set(manifest) != _NATIVE_MANIFEST_FIELDS:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive manifest fields are invalid."
        )
    if manifest["format"] != _NATIVE_FORMAT:
        raise ArrayArchiveCorruptionError("Archive is not native Phydrax velocimetry.")
    kind = manifest["value_kind"]
    value_type = manifest["value_type"]
    structure = manifest["structure"]
    records = manifest["leaves"]
    provenance = manifest["provenance"]
    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(value_type, str)
        or not value_type
        or not isinstance(structure, dict)
        or not isinstance(records, list)
        or not isinstance(provenance, dict)
    ):
        raise ArrayArchiveCorruptionError("Velocimetry archive metadata is invalid.")
    if expected_kind is not None and kind != str(expected_kind):
        raise ArrayArchiveCorruptionError(
            f"Expected velocimetry value kind {expected_kind!r}; found {kind!r}."
        )

    expected_names: set[str] = set()
    previous_index = -1
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "name",
            "tree_index",
            "path",
            "shape",
            "dtype",
            "backend",
        }:
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive leaf inventory is invalid."
            )
        name = record["name"]
        index = record["tree_index"]
        if (
            name != f"leaves/{ordinal:06d}"
            or not isinstance(index, int)
            or index <= previous_index
            or record["backend"] not in ("jax", "numpy", "prng_key")
        ):
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive leaf ordering is invalid."
            )
        previous_index = index
        expected_names.add(name)
    if set(arrays) != expected_names:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive array payloads do not match its leaf inventory."
        )

    identity_payload = {
        "format": _NATIVE_FORMAT,
        "value_kind": kind,
        "value_type": value_type,
        "structure": structure,
        "leaves": records,
        "provenance": provenance,
        "content": array_tree_fingerprint(arrays),
    }
    archive_id = canonical_fingerprint(identity_payload)
    if manifest["archive_id"] != archive_id:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive semantic fingerprint is inconsistent."
        )

    template = model_from_structure_recipe(structure)
    if artifact_value_id(type(template)) != value_type:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive value type is inconsistent with its structure."
        )
    if expected_type is not None and type(template) is not expected_type:
        raise ArrayArchiveCorruptionError(
            f"Expected archived type {expected_type.__name__}; found {type(template).__name__}."
        )
    path_leaves, tree = jax.tree_util.tree_flatten_with_path(template)
    restored = [leaf for _, leaf in path_leaves]
    for record in records:
        index = record["tree_index"]
        if index < 0 or index >= len(path_leaves):
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive leaf index is outside its structure."
            )
        path_key, template_leaf = path_leaves[index]
        payload = arrays[record["name"]]
        if (
            (jax.tree_util.keystr(path_key) or "<root>") != record["path"]
            or list(payload.shape) != record["shape"]
            or payload.dtype.str != record["dtype"]
            or not _is_array(template_leaf)
        ):
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive leaf metadata is inconsistent."
            )
        restored[index] = _restore_array(payload, template_leaf, record["backend"])
    value = jax.tree_util.tree_unflatten(tree, restored)
    if artifact_value_id(type(value)) != value_type:
        raise ArrayArchiveCorruptionError("Restored velocimetry value type changed.")
    return VelocimetryArchive(
        value=value,
        archive_id=archive_id,
        value_kind=kind,
        value_type=value_type,
        provenance=MappingProxyType(provenance),
    )


def _is_array(value: Any, /) -> bool:
    return isinstance(value, (jax.Array, np.ndarray))


def _array_payload(value: Any, /) -> tuple[np.ndarray, str]:
    if isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    ):
        return np.asarray(jr.key_data(value)), "prng_key"
    if isinstance(value, jax.Array):
        return np.asarray(value), "jax"
    return np.asarray(value), "numpy"


def _restore_array(payload: np.ndarray, template: Any, backend: str, /) -> Any:
    if backend == "prng_key":
        if not (
            isinstance(template, jax.Array)
            and jax.dtypes.issubdtype(template.dtype, jax.dtypes.prng_key)
        ):
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive PRNG leaf does not match its structure."
            )
        return jr.wrap_key_data(
            jnp.asarray(payload, dtype=jnp.uint32), impl=str(jr.key_impl(template))
        )
    if backend == "jax" and isinstance(template, jax.Array):
        if payload.shape != template.shape or payload.dtype != np.dtype(template.dtype):
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive JAX leaf shape or dtype changed."
            )
        return jnp.asarray(payload)
    if backend == "numpy" and isinstance(template, np.ndarray):
        if payload.shape != template.shape or payload.dtype != template.dtype:
            raise ArrayArchiveCorruptionError(
                "Velocimetry archive NumPy leaf shape or dtype changed."
            )
        return np.array(payload, copy=True)
    raise ArrayArchiveCorruptionError(
        "Velocimetry archive leaf backend does not match its structure."
    )


__all__ = [
    "VelocimetryArchive",
    "read_velocimetry_archive",
    "write_velocimetry_archive",
]
