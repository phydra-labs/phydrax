#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
from ..._model import artifact_value_id, model_structure_recipe
from ..._model._structure import (
    model_from_array_recipe,
    model_recipe_array_inventory,
    model_recipe_template,
    pack_model_array_tree,
)


_NATIVE_FORMAT = "phydrax-native-velocimetry"
_NATIVE_MANIFEST_FIELDS = {
    "format",
    "archive_id",
    "value_kind",
    "value_type",
    "structure",
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
    arrays = pack_model_array_tree(value, structure, prefix="leaves")
    provenance_ = canonical_mapping(dict(provenance or {}))
    identity_payload = {
        "format": _NATIVE_FORMAT,
        "value_kind": kind,
        "value_type": value_type,
        "structure": structure,
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
    if expected_type is not None and not isinstance(expected_type, type):
        raise TypeError("expected_type must be a type or None.")
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
    provenance = manifest["provenance"]
    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(value_type, str)
        or not value_type
        or not isinstance(structure, dict)
        or not isinstance(provenance, dict)
    ):
        raise ArrayArchiveCorruptionError("Velocimetry archive metadata is invalid.")
    if expected_kind is not None and kind != str(expected_kind):
        raise ArrayArchiveCorruptionError(
            f"Expected velocimetry value kind {expected_kind!r}; found {kind!r}."
        )

    try:
        template = model_recipe_template(structure)
        inventory = model_recipe_array_inventory(structure, prefix="leaves")
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive structure recipe is invalid."
        ) from error
    if artifact_value_id(type(template)) != value_type:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive value type is inconsistent with its structure."
        )
    if expected_type is not None and type(template) is not expected_type:
        raise ArrayArchiveCorruptionError(
            f"Expected archived type {expected_type.__name__}; found {type(template).__name__}."
        )
    if set(arrays) != {entry.name for entry in inventory}:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive array payloads do not match its structure."
        )

    identity_payload = {
        "format": _NATIVE_FORMAT,
        "value_kind": kind,
        "value_type": value_type,
        "structure": structure,
        "provenance": provenance,
        "content": array_tree_fingerprint(arrays),
    }
    archive_id = canonical_fingerprint(identity_payload)
    if manifest["archive_id"] != archive_id:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive semantic fingerprint is inconsistent."
        )

    try:
        value = model_from_array_recipe(structure, arrays, prefix="leaves")
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive arrays are incompatible with its structure."
        ) from error
    if artifact_value_id(type(value)) != value_type:
        raise ArrayArchiveCorruptionError(
            "Velocimetry archive value type is inconsistent with its structure."
        )
    if expected_type is not None and type(value) is not expected_type:
        raise ArrayArchiveCorruptionError(
            f"Expected archived type {expected_type.__name__}; found {type(value).__name__}."
        )
    if model_structure_recipe(value, path="value") != structure:
        raise ArrayArchiveCorruptionError("Restored velocimetry value structure changed.")
    return VelocimetryArchive(
        value=value,
        archive_id=archive_id,
        value_kind=kind,
        value_type=value_type,
        provenance=MappingProxyType(provenance),
    )


__all__ = [
    "VelocimetryArchive",
    "read_velocimetry_archive",
    "write_velocimetry_archive",
]
