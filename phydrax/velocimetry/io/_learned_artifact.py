#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..._array_archive import ArrayArchiveCorruptionError
from ..._fingerprint import canonical_mapping
from ..._model import artifact_value_id, register_artifact_value
from ._archive import read_velocimetry_archive, write_velocimetry_archive


_LEARNED_KIND = "learned-piv-model"
_COORDINATE_CONVENTION = "row-down-column-right"
_PROVENANCE_FIELDS = {
    "architecture_id",
    "coordinate_convention",
    "normalization",
    "training_data_id",
    "qualification",
    "provenance",
}


@dataclass(frozen=True, slots=True)
class LearnedPIVArtifactManifest:
    """Validated scientific metadata for a native learned PIV model."""

    archive_id: str
    architecture_id: str
    coordinate_convention: str
    normalization: Mapping[str, Any]
    training_data_id: str
    qualification: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LearnedPIVArtifact:
    """Restored learned estimator and its verified scientific provenance."""

    model: Any
    manifest: LearnedPIVArtifactManifest


def register_learned_piv_model(
    architecture_id: str,
    model_type: type,
    /,
) -> type:
    """Register a path-independent native learned-PIV model identity."""
    if not isinstance(model_type, type):
        raise TypeError("model_type must be a type.")
    return register_artifact_value(str(architecture_id), model_type)


def save_learned_piv_artifact(
    path: str | Path,
    model: Any,
    /,
    *,
    normalization: Mapping[str, Any],
    training_data_id: str,
    qualification: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    architecture_id: str | None = None,
) -> Path:
    """Persist a native learned PIV model without executable or pickle payloads."""
    registered_id = artifact_value_id(type(model))
    architecture = (
        registered_id if architecture_id is None else str(architecture_id).strip()
    )
    if architecture != registered_id:
        raise ValueError(
            "architecture_id must be the registered identity of the model's exact type."
        )
    training = str(training_data_id).strip()
    if not training:
        raise ValueError("training_data_id must be non-empty.")
    metadata = {
        "architecture_id": architecture,
        "coordinate_convention": _COORDINATE_CONVENTION,
        "normalization": canonical_mapping(normalization),
        "training_data_id": training,
        "qualification": canonical_mapping(qualification),
        "provenance": canonical_mapping(dict(provenance or {})),
    }
    return write_velocimetry_archive(
        path,
        model,
        value_kind=_LEARNED_KIND,
        provenance=metadata,
    )


def read_learned_piv_artifact(path: str | Path, /) -> LearnedPIVArtifact:
    """Restore a registered learned model after exact metadata and checksum checks."""
    archive = read_velocimetry_archive(path, expected_kind=_LEARNED_KIND)
    metadata = dict(archive.provenance)
    if set(metadata) != _PROVENANCE_FIELDS:
        raise ArrayArchiveCorruptionError(
            "Learned PIV artifact provenance fields are invalid."
        )
    architecture = metadata["architecture_id"]
    convention = metadata["coordinate_convention"]
    training = metadata["training_data_id"]
    normalization = metadata["normalization"]
    qualification = metadata["qualification"]
    provenance = metadata["provenance"]
    if (
        architecture != artifact_value_id(type(archive.value))
        or convention != _COORDINATE_CONVENTION
        or not isinstance(training, str)
        or not training
        or not isinstance(normalization, dict)
        or not isinstance(qualification, dict)
        or not isinstance(provenance, dict)
    ):
        raise ArrayArchiveCorruptionError(
            "Learned PIV artifact scientific metadata is inconsistent."
        )
    manifest = LearnedPIVArtifactManifest(
        archive_id=archive.archive_id,
        architecture_id=architecture,
        coordinate_convention=convention,
        normalization=MappingProxyType(normalization),
        training_data_id=training,
        qualification=MappingProxyType(qualification),
        provenance=MappingProxyType(provenance),
    )
    return LearnedPIVArtifact(model=archive.value, manifest=manifest)


def load_learned_piv_model(path: str | Path, /) -> Any:
    """Restore only the verified learned PIV estimator."""
    return read_learned_piv_artifact(path).model


__all__ = [
    "LearnedPIVArtifact",
    "LearnedPIVArtifactManifest",
    "load_learned_piv_model",
    "read_learned_piv_artifact",
    "register_learned_piv_model",
    "save_learned_piv_artifact",
]
