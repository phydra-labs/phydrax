#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable typed-result archives restored against caller-owned structure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._array_archive import (
    array_collection_digest,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(sorted(_identifier(value, name) for value in values))
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique.")
    return result


def _structure_ids(values: Mapping[str, str], /) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError("structure_ids must be a non-empty mapping.")
    result = tuple(
        sorted(
            (
                _identifier(name, "structure coordinate"),
                _identifier(value, f"structure coordinate {name!r}"),
            )
            for name, value in values.items()
        )
    )
    if len({name for name, _ in result}) != len(result):
        raise ValueError("Structure coordinate names must be unique.")
    return result


def _type_name(value: object, /) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


@dataclass(frozen=True, slots=True)
class ArrayArtifactProvenance:
    """Content-addressed source, profile, unit, and producer identity."""

    producer_id: str
    source_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    provenance_id: str

    def __init__(
        self,
        producer_id: str,
        source_ids: Sequence[str],
        profile_ids: Sequence[str],
        unit_ids: Sequence[str],
        /,
    ):
        producer = _identifier(producer_id, "producer_id")
        sources = _identifiers(source_ids, "source_ids")
        profiles = _identifiers(profile_ids, "profile_ids")
        units = _identifiers(unit_ids, "unit_ids")
        record = {
            "kind": "array-artifact-provenance",
            "producer_id": producer,
            "source_ids": list(sources),
            "profile_ids": list(profiles),
            "unit_ids": list(units),
        }
        object.__setattr__(self, "producer_id", producer)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "profile_ids", profiles)
        object.__setattr__(self, "unit_ids", units)
        object.__setattr__(self, "provenance_id", canonical_fingerprint(record))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "array-artifact-provenance",
            "producer_id": self.producer_id,
            "source_ids": list(self.source_ids),
            "profile_ids": list(self.profile_ids),
            "unit_ids": list(self.unit_ids),
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> "ArrayArtifactProvenance":
        if not isinstance(record, Mapping) or set(record) != {
            "kind",
            "producer_id",
            "source_ids",
            "profile_ids",
            "unit_ids",
            "provenance_id",
        }:
            raise ValueError("Array-artifact provenance record is malformed.")
        if record["kind"] != "array-artifact-provenance":
            raise ValueError("Array-artifact provenance kind is invalid.")
        sources = record["source_ids"]
        profiles = record["profile_ids"]
        units = record["unit_ids"]
        if any(
            not isinstance(values, Sequence) or isinstance(values, str)
            for values in (sources, profiles, units)
        ):
            raise TypeError("Serialized provenance identifiers must be sequences.")
        value = cls(
            str(record["producer_id"]),
            tuple(str(item) for item in sources),
            tuple(str(item) for item in profiles),
            tuple(str(item) for item in units),
        )
        if value.provenance_id != record["provenance_id"]:
            raise ValueError("Array-artifact provenance content address is invalid.")
        return value


@dataclass(frozen=True, slots=True)
class ArrayArtifactReceipt:
    """Immutable identity returned after writing or restoring one artifact."""

    path: Path
    artifact_kind: str
    artifact_type: str
    artifact_id: str
    provenance_id: str
    array_digest: str


def write_typed_array_artifact(
    path: str | Path,
    artifact: Any,
    /,
    *,
    artifact_kind: str,
    provenance: ArrayArtifactProvenance,
    structure_ids: Mapping[str, str],
) -> ArrayArtifactReceipt:
    """Archive numeric leaves only; executable/static objects remain caller-owned."""

    kind = _identifier(artifact_kind, "artifact_kind")
    if not isinstance(provenance, ArrayArtifactProvenance):
        raise TypeError("provenance must be ArrayArtifactProvenance.")
    structure = _structure_ids(structure_ids)
    arrays: dict[str, object] = {}
    tree = pack_array_tree("artifact", artifact, arrays)
    if not arrays:
        raise ValueError("Typed array artifacts must contain numeric leaves.")
    array_digest = array_collection_digest(arrays)
    content = {
        "kind": "typed-array-artifact",
        "artifact_kind": kind,
        "artifact_type": _type_name(artifact),
        "provenance": provenance.to_record(),
        "structure_ids": dict(structure),
        "tree": tree,
        "array_digest": array_digest,
    }
    artifact_id = canonical_fingerprint(content)
    destination = write_array_archive(
        path,
        manifest={**content, "artifact_id": artifact_id},
        arrays=arrays,
    )
    return ArrayArtifactReceipt(
        destination,
        kind,
        content["artifact_type"],
        artifact_id,
        provenance.provenance_id,
        array_digest,
    )


def read_typed_array_artifact(
    path: str | Path,
    template: Any,
    /,
    *,
    artifact_kind: str,
    provenance: ArrayArtifactProvenance,
    structure_ids: Mapping[str, str],
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[Any, ArrayArtifactReceipt]:
    """Restore arrays only when the caller supplies exactly matching structure."""

    kind = _identifier(artifact_kind, "artifact_kind")
    if not isinstance(provenance, ArrayArtifactProvenance):
        raise TypeError("provenance must be ArrayArtifactProvenance.")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be ArrayArchiveLimits.")
    structure = dict(_structure_ids(structure_ids))
    manifest, arrays = read_array_archive(path, limits=limits)
    if set(manifest) != {
        "kind",
        "artifact_kind",
        "artifact_type",
        "provenance",
        "structure_ids",
        "tree",
        "array_digest",
        "artifact_id",
        "arrays",
    }:
        raise ValueError("Typed array artifact manifest is malformed.")
    if (
        manifest["kind"] != "typed-array-artifact"
        or manifest["artifact_kind"] != kind
        or manifest["artifact_type"] != _type_name(template)
        or manifest["structure_ids"] != structure
    ):
        raise ValueError("Typed array artifact does not match caller-prepared structure.")
    archived_provenance = ArrayArtifactProvenance.from_record(manifest["provenance"])
    if archived_provenance.provenance_id != provenance.provenance_id:
        raise ValueError("Typed array artifact provenance does not match the caller.")
    array_digest = array_collection_digest(arrays)
    if manifest["array_digest"] != array_digest:
        raise ValueError("Typed array artifact collection digest is invalid.")
    content = {
        name: manifest[name]
        for name in (
            "kind",
            "artifact_kind",
            "artifact_type",
            "provenance",
            "structure_ids",
            "tree",
            "array_digest",
        )
    }
    artifact_id = canonical_fingerprint(content)
    if manifest["artifact_id"] != artifact_id:
        raise ValueError("Typed array artifact content address is invalid.")
    restored = unpack_array_tree(manifest["tree"], arrays, template)
    return restored, ArrayArtifactReceipt(
        Path(path),
        kind,
        str(manifest["artifact_type"]),
        artifact_id,
        provenance.provenance_id,
        array_digest,
    )


__all__ = [
    "ArrayArtifactProvenance",
    "ArrayArtifactReceipt",
    "read_typed_array_artifact",
    "write_typed_array_artifact",
]
