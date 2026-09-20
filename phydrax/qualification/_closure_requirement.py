#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fine-grained capability closure requirements and resolutions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .._fingerprint import canonical_fingerprint
from ._closure_taxonomy import (
    CapabilityDepth,
    CarrierRepresentation,
    ClosureDisposition,
    CouplingLocation,
    ExecutionRegime,
    PhysicsField,
    TopologyRegime,
    WorkflowClass,
)
from ._registry import _capability_name, _identifier


def _strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    return tuple(sorted(_identifier(str(value), label) for value in values))


def _enums(kind, values):
    return tuple(sorted((kind(value) for value in values), key=lambda item: item.value))


@dataclass(frozen=True, slots=True)
class CapabilityClosureRequirement:
    requirement: str
    minimum_depth: CapabilityDepth
    physical_fields: tuple[PhysicsField, ...]
    carriers: tuple[CarrierRepresentation, ...]
    coupling_locations: tuple[CouplingLocation, ...]
    execution_regimes: tuple[ExecutionRegime, ...]
    topology_regimes: tuple[TopologyRegime, ...]
    workflow_classes: tuple[WorkflowClass, ...]
    required_benchmarks: tuple[str, ...]
    required_evidence_dimensions: tuple[str, ...]
    required_providers: tuple[str, ...]
    required_public_symbols: tuple[str, ...]
    required_documents: tuple[str, ...]
    source_ids: tuple[str, ...]
    rationale: str

    @classmethod
    def create(
        cls,
        requirement,
        /,
        *,
        minimum_depth,
        physical_fields,
        carriers,
        coupling_locations,
        execution_regimes,
        topology_regimes,
        workflow_classes,
        required_benchmarks=(),
        required_evidence_dimensions=(),
        required_providers=(),
        required_public_symbols=(),
        required_documents=(),
        source_ids=(),
        rationale,
    ):
        return cls(
            _identifier(requirement, "requirement"),
            CapabilityDepth(minimum_depth),
            _enums(PhysicsField, physical_fields),
            _enums(CarrierRepresentation, carriers),
            _enums(CouplingLocation, coupling_locations),
            _enums(ExecutionRegime, execution_regimes),
            _enums(TopologyRegime, topology_regimes),
            _enums(WorkflowClass, workflow_classes),
            _strings(required_benchmarks, "benchmark"),
            _strings(required_evidence_dimensions, "evidence dimension"),
            _strings(required_providers, "provider"),
            _strings(required_public_symbols, "public symbol"),
            _strings(required_documents, "document"),
            _strings(source_ids, "source ID"),
            _identifier(rationale, "rationale"),
        )

    @property
    def requirement_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "capability-closure-requirement",
            "requirement": self.requirement,
            "minimum_depth": self.minimum_depth.name.lower().replace("_", "-"),
            "physical_fields": [v.value for v in self.physical_fields],
            "carriers": [v.value for v in self.carriers],
            "coupling_locations": [v.value for v in self.coupling_locations],
            "execution_regimes": [v.value for v in self.execution_regimes],
            "topology_regimes": [v.value for v in self.topology_regimes],
            "workflow_classes": [v.value for v in self.workflow_classes],
            "required_benchmarks": list(self.required_benchmarks),
            "required_evidence_dimensions": list(self.required_evidence_dimensions),
            "required_providers": list(self.required_providers),
            "required_public_symbols": list(self.required_public_symbols),
            "required_documents": list(self.required_documents),
            "source_ids": list(self.source_ids),
            "rationale": self.rationale,
        }
        return {**record, "requirement_id": self.requirement_id} if include_id else record


@dataclass(frozen=True, slots=True)
class CapabilityGapResolution:
    requirement_id: str
    disposition: ClosureDisposition
    actual_depth: CapabilityDepth
    capability_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    rationale: str
    release_authorized: bool

    @classmethod
    def create(
        cls,
        requirement_id,
        disposition,
        actual_depth=CapabilityDepth.SEMANTIC,
        /,
        *,
        capability_ids=(),
        evidence_ids=(),
        provider_ids=(),
        rationale,
        release_authorized=False,
    ):
        return cls(
            _identifier(requirement_id, "requirement ID"),
            ClosureDisposition(disposition),
            CapabilityDepth(actual_depth),
            tuple(
                sorted(_capability_name(value, "capability") for value in capability_ids)
            ),
            _strings(evidence_ids, "evidence ID"),
            _strings(provider_ids, "provider ID"),
            _identifier(rationale, "rationale"),
            bool(release_authorized),
        )

    @property
    def resolution_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "capability-gap-resolution",
            "requirement_id": self.requirement_id,
            "disposition": self.disposition.value,
            "actual_depth": self.actual_depth.name.lower().replace("_", "-"),
            "capability_ids": list(self.capability_ids),
            "evidence_ids": list(self.evidence_ids),
            "provider_ids": list(self.provider_ids),
            "rationale": self.rationale,
            "release_authorized": self.release_authorized,
        }
        return {**record, "resolution_id": self.resolution_id} if include_id else record


__all__ = ["CapabilityClosureRequirement", "CapabilityGapResolution"]
