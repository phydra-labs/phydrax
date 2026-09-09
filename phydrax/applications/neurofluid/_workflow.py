#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Declarative neurofluid stage lineage; execution remains externally orchestrated."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..._fingerprint import canonical_fingerprint


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty identifier.")
    return value


@dataclass(frozen=True, slots=True)
class NeurofluidPipelineStage:
    stage_id: str
    kind: str
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    dependency_stage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _identifier(self.stage_id, "stage_id"))
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        collections = (
            ("input_artifact_ids", self.input_artifact_ids),
            ("output_artifact_ids", self.output_artifact_ids),
            ("dependency_stage_ids", self.dependency_stage_ids),
        )
        for name, raw in collections:
            values = tuple(_identifier(value, name) for value in raw)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique identifiers.")
            object.__setattr__(self, name, values)
        if not self.output_artifact_ids:
            raise ValueError("Pipeline stages require at least one output artifact.")


@dataclass(frozen=True, slots=True)
class NeurofluidPipelineManifest:
    case_revision: str
    stages: tuple[NeurofluidPipelineStage, ...]
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        revision = _identifier(self.case_revision, "case_revision")
        if not self.stages or any(
            not isinstance(value, NeurofluidPipelineStage) for value in self.stages
        ):
            raise ValueError("stages must contain NeurofluidPipelineStage values.")
        identifiers = tuple(value.stage_id for value in self.stages)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Pipeline stage IDs must be unique.")
        known: set[str] = set()
        produced: set[str] = set()
        for stage in self.stages:
            if any(dependency not in known for dependency in stage.dependency_stage_ids):
                raise ValueError("Pipeline stages must be topologically ordered.")
            if produced.intersection(stage.output_artifact_ids):
                raise ValueError(
                    "An artifact may be produced by only one pipeline stage."
                )
            known.add(stage.stage_id)
            produced.update(stage.output_artifact_ids)
        object.__setattr__(self, "case_revision", revision)
        object.__setattr__(
            self,
            "manifest_id",
            canonical_fingerprint(
                {
                    "kind": "neurofluid-pipeline-manifest",
                    "case": revision,
                    "stages": [
                        {
                            "id": value.stage_id,
                            "kind": value.kind,
                            "inputs": list(value.input_artifact_ids),
                            "outputs": list(value.output_artifact_ids),
                            "dependencies": list(value.dependency_stage_ids),
                        }
                        for value in self.stages
                    ],
                }
            ),
        )


__all__ = ["NeurofluidPipelineManifest", "NeurofluidPipelineStage"]
