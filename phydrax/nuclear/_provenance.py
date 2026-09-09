#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evaluated and processed nuclear-data provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .._fingerprint import canonical_fingerprint, canonical_mapping
from ..qualification import ReferenceArtifactManifest


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return normalized


@dataclass(frozen=True, slots=True)
class NuclearDataProvenance:
    """Authority, evaluation, processing, rights, and lineage of nuclear data."""

    reference: ReferenceArtifactManifest
    source_uri: str
    library_name: str
    release_id: str
    evaluation_id: str
    processing_tool: str | None = None
    processing_release: str | None = None
    processing_parameters: Mapping[str, Any] | None = None
    parent_data_ids: Sequence[str] = ()
    provenance_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ReferenceArtifactManifest):
            raise TypeError("reference must be ReferenceArtifactManifest.")
        source = _text(self.source_uri, "source_uri")
        library = _text(self.library_name, "library_name")
        release = _text(self.release_id, "release_id")
        evaluation = _text(self.evaluation_id, "evaluation_id")
        tool = (
            None
            if self.processing_tool is None
            else _text(self.processing_tool, "processing_tool")
        )
        processing_release = (
            None
            if self.processing_release is None
            else _text(self.processing_release, "processing_release")
        )
        if (tool is None) != (processing_release is None):
            raise ValueError(
                "processing_tool and processing_release must be provided together."
            )
        parameters = canonical_mapping(
            {} if self.processing_parameters is None else self.processing_parameters
        )
        if parameters and tool is None:
            raise ValueError("Processing parameters require a processing tool identity.")
        parents = tuple(_text(value, "parent_data_id") for value in self.parent_data_ids)
        if len(parents) != len(set(parents)):
            raise ValueError("parent_data_ids must be unique.")
        object.__setattr__(self, "source_uri", source)
        object.__setattr__(self, "library_name", library)
        object.__setattr__(self, "release_id", release)
        object.__setattr__(self, "evaluation_id", evaluation)
        object.__setattr__(self, "processing_tool", tool)
        object.__setattr__(self, "processing_release", processing_release)
        object.__setattr__(self, "processing_parameters", MappingProxyType(parameters))
        object.__setattr__(self, "parent_data_ids", parents)
        object.__setattr__(
            self,
            "provenance_id",
            canonical_fingerprint(
                {
                    "kind": "nuclear-data-provenance",
                    "reference": self.reference.manifest_id,
                    "source_uri": source,
                    "library": library,
                    "release": release,
                    "evaluation": evaluation,
                    "processing_tool": tool,
                    "processing_release": processing_release,
                    "processing_parameters": parameters,
                    "parents": list(parents),
                }
            ),
        )

    @property
    def is_processed(self) -> bool:
        return self.processing_tool is not None


__all__ = ["NuclearDataProvenance"]
