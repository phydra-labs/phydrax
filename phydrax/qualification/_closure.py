#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Orthogonal capability-closure taxonomy, source ledger, and gap matrices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .._fingerprint import canonical_fingerprint
from ._registry import _capability_name, _identifier


class PhysicsField(StrEnum):
    SOLID_MECHANICS = "solid-mechanics"
    FLUID_MECHANICS = "fluid-mechanics"
    THERMAL = "thermal"
    CHEMICAL_SPECIES = "chemical-species"
    ELECTRIC = "electric"
    MAGNETIC = "magnetic"
    ACOUSTIC = "acoustic"
    OPTICAL = "optical"
    IONIZING_RADIATION = "ionizing-radiation"
    ELECTRONIC_QUANTUM = "electronic-quantum"
    GRAVITATIONAL = "gravitational"
    BIOLOGICAL_ACTIVE = "biological-active"


class CarrierRepresentation(StrEnum):
    CONTINUUM_VOLUME = "continuum-volume"
    INTERFACE_SURFACE = "interface-surface"
    PARTICLE = "particle"
    NETWORK = "network"
    KINETIC_DISTRIBUTION = "kinetic-distribution"
    ATOMISTIC = "atomistic"
    REDUCED_SYSTEM = "reduced-system"


class CouplingLocation(StrEnum):
    BULK = "bulk"
    BOUNDARY = "boundary"
    MOVING_INTERFACE = "moving-interface"
    CONTACT = "contact"
    PARTICLE_FIELD = "particle-field"
    NETWORK_PORT = "network-port"
    CROSS_SCALE_TRANSFER = "cross-scale-transfer"


class ExecutionRegime(StrEnum):
    STATIC = "static"
    TRANSIENT = "transient"
    FREQUENCY_DOMAIN = "frequency-domain"
    EIGENVALUE_STABILITY = "eigenvalue-stability"
    PERIODIC_STEADY_STATE = "periodic-steady-state"
    STOCHASTIC = "stochastic"
    MULTIRATE = "multirate"
    MULTISCALE = "multiscale"


class TopologyRegime(StrEnum):
    FIXED = "fixed"
    MOVING_MESH = "moving-mesh"
    FREE_SURFACE = "free-surface"
    PHASE_CHANGE = "phase-change"
    FRACTURE = "fracture"
    CONTACT = "contact"
    ACTIVATION_REMOVAL = "activation-removal"
    REMESHING = "remeshing"
    TOPOLOGY_TRANSITION = "topology-transition"


class WorkflowClass(StrEnum):
    FORWARD = "forward"
    INVERSE = "inverse"
    CALIBRATION = "calibration"
    OPTIMIZATION = "optimization"
    CONTROL = "control"
    UNCERTAINTY_QUANTIFICATION = "uncertainty-quantification"
    EXPERIMENT_CORRELATION = "experiment-correlation"
    RESTART_REPLAY = "restart-replay"
    QUALIFICATION = "qualification"
    DEPLOYMENT = "deployment"


class ImplementationOwnership(StrEnum):
    NATIVE = "native"
    PROVIDER = "provider"
    RESEARCH = "research"
    REJECTED = "rejected"


class SourceReuseClass(StrEnum):
    PERMISSIVE = "permissive"
    WEAK_COPYLEFT = "weak-copyleft"
    STRONG_COPYLEFT = "strong-copyleft"
    SOURCE_AVAILABLE = "source-available"
    PROPRIETARY = "proprietary"
    PUBLIC_DOMAIN = "public-domain"
    UNKNOWN = "unknown"


class ClosureDisposition(StrEnum):
    IMPLEMENTED = "implemented"
    CANDIDATE = "candidate"
    PROVIDER = "provider"
    RESEARCH = "research"
    REJECTED = "rejected"
    MISSING = "missing"


def _canonical_strings(values: Sequence[str], label: str, /) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(str(value), label) for value in values))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} values must be unique.")
    return result


def _canonical_enums(enum_type, values, label: str, /):
    result = tuple(
        sorted((enum_type(value) for value in values), key=lambda item: item.value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label} values must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class SourceReference:
    source_id: str
    repository_url: str
    revision: str
    licence: str
    reuse_class: SourceReuseClass
    concepts: tuple[str, ...]
    relevant_paths: tuple[str, ...] = ()
    relevant_documents: tuple[str, ...] = ()
    code_inspected: bool = False
    copying_permitted: bool = False
    provider_only: bool = False
    notice_required: bool = True
    data_rights: str = "not-applicable"
    reviewer: str = "unreviewed"

    @classmethod
    def create(
        cls,
        source_id: str,
        repository_url: str,
        revision: str,
        licence: str,
        reuse_class: SourceReuseClass | str,
        /,
        *,
        concepts: Sequence[str],
        relevant_paths: Sequence[str] = (),
        relevant_documents: Sequence[str] = (),
        code_inspected: bool = False,
        copying_permitted: bool = False,
        provider_only: bool = False,
        notice_required: bool = True,
        data_rights: str = "not-applicable",
        reviewer: str = "unreviewed",
    ) -> SourceReference:
        return cls(
            _identifier(source_id, "source ID"),
            str(repository_url).strip(),
            _identifier(revision, "source revision"),
            _identifier(licence, "source licence"),
            SourceReuseClass(reuse_class),
            _canonical_strings(concepts, "source concept"),
            _canonical_strings(relevant_paths, "source path"),
            _canonical_strings(relevant_documents, "source document"),
            bool(code_inspected),
            bool(copying_permitted),
            bool(provider_only),
            bool(notice_required),
            _identifier(data_rights, "data-rights disposition"),
            _identifier(reviewer, "source reviewer"),
        )

    def __post_init__(self) -> None:
        if not self.repository_url.startswith(("https://", "http://")):
            raise ValueError("Source repository URL must be absolute HTTP(S).")
        if not self.concepts:
            raise ValueError("Source references require studied concepts.")
        if (
            self.reuse_class
            in (SourceReuseClass.STRONG_COPYLEFT, SourceReuseClass.PROPRIETARY)
            and self.copying_permitted
        ):
            raise ValueError("Strong-copyleft/proprietary sources cannot permit copying.")

    @property
    def source_reference_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record = {
            "kind": "source-reference",
            "source_id": self.source_id,
            "repository_url": self.repository_url,
            "revision": self.revision,
            "licence": self.licence,
            "reuse_class": self.reuse_class.value,
            "concepts": list(self.concepts),
            "relevant_paths": list(self.relevant_paths),
            "relevant_documents": list(self.relevant_documents),
            "code_inspected": self.code_inspected,
            "copying_permitted": self.copying_permitted,
            "provider_only": self.provider_only,
            "notice_required": self.notice_required,
            "data_rights": self.data_rights,
            "reviewer": self.reviewer,
        }
        return (
            {**record, "source_reference_id": self.source_reference_id}
            if include_id
            else record
        )


@dataclass(frozen=True, slots=True)
class SourceAbsorptionLedger:
    sources: tuple[SourceReference, ...]

    @classmethod
    def create(cls, sources: Sequence[SourceReference], /) -> SourceAbsorptionLedger:
        return cls(tuple(sorted(sources, key=lambda item: item.source_id)))

    def __post_init__(self) -> None:
        if not self.sources or any(
            not isinstance(value, SourceReference) for value in self.sources
        ):
            raise TypeError("Source ledger requires SourceReference values.")
        identities = tuple(value.source_id for value in self.sources)
        if len(set(identities)) != len(identities):
            raise ValueError("Source IDs must be unique.")

    @property
    def ledger_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record = {
            "kind": "source-absorption-ledger",
            "sources": [value.to_record() for value in self.sources],
        }
        return {**record, "ledger_id": self.ledger_id} if include_id else record


@dataclass(frozen=True, slots=True)
class CapabilityClosureRequirement:
    requirement: str
    physical_fields: tuple[PhysicsField, ...]
    carriers: tuple[CarrierRepresentation, ...]
    coupling_locations: tuple[CouplingLocation, ...]
    execution_regimes: tuple[ExecutionRegime, ...]
    topology_regimes: tuple[TopologyRegime, ...]
    workflow_classes: tuple[WorkflowClass, ...]
    rationale: str
    source_ids: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        requirement: str,
        /,
        *,
        physical_fields,
        carriers,
        coupling_locations,
        execution_regimes,
        topology_regimes,
        workflow_classes,
        rationale: str,
        source_ids: Sequence[str] = (),
    ) -> CapabilityClosureRequirement:
        return cls(
            _identifier(requirement, "closure requirement"),
            _canonical_enums(PhysicsField, physical_fields, "physics field"),
            _canonical_enums(CarrierRepresentation, carriers, "carrier"),
            _canonical_enums(CouplingLocation, coupling_locations, "coupling location"),
            _canonical_enums(ExecutionRegime, execution_regimes, "execution regime"),
            _canonical_enums(TopologyRegime, topology_regimes, "topology regime"),
            _canonical_enums(WorkflowClass, workflow_classes, "workflow class"),
            _identifier(rationale, "closure rationale"),
            _canonical_strings(source_ids, "source ID"),
        )

    def __post_init__(self) -> None:
        if (
            not self.physical_fields
            or not self.carriers
            or not self.execution_regimes
            or not self.workflow_classes
        ):
            raise ValueError(
                "Closure requirements need fields, carriers, execution, and workflows."
            )

    @property
    def requirement_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record = {
            "kind": "capability-closure-requirement",
            "requirement": self.requirement,
            "physical_fields": [value.value for value in self.physical_fields],
            "carriers": [value.value for value in self.carriers],
            "coupling_locations": [value.value for value in self.coupling_locations],
            "execution_regimes": [value.value for value in self.execution_regimes],
            "topology_regimes": [value.value for value in self.topology_regimes],
            "workflow_classes": [value.value for value in self.workflow_classes],
            "rationale": self.rationale,
            "source_ids": list(self.source_ids),
        }
        return {**record, "requirement_id": self.requirement_id} if include_id else record


@dataclass(frozen=True, slots=True)
class CapabilityGapResolution:
    requirement_id: str
    disposition: ClosureDisposition
    capability_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rationale: str

    @classmethod
    def create(
        cls,
        requirement_id: str,
        disposition: ClosureDisposition | str,
        /,
        *,
        capability_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        rationale: str,
    ) -> CapabilityGapResolution:
        return cls(
            _identifier(requirement_id, "requirement ID"),
            ClosureDisposition(disposition),
            tuple(
                sorted(_capability_name(value, "capability") for value in capability_ids)
            ),
            _canonical_strings(evidence_ids, "evidence ID"),
            _identifier(rationale, "gap-resolution rationale"),
        )

    def __post_init__(self) -> None:
        if (
            self.disposition
            in (ClosureDisposition.IMPLEMENTED, ClosureDisposition.CANDIDATE)
            and not self.capability_ids
        ):
            raise ValueError("Implemented/candidate resolutions require capabilities.")
        if self.disposition is ClosureDisposition.IMPLEMENTED and not self.evidence_ids:
            raise ValueError("Implemented resolutions require evidence.")

    @property
    def resolution_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record = {
            "kind": "capability-gap-resolution",
            "requirement_id": self.requirement_id,
            "disposition": self.disposition.value,
            "capability_ids": list(self.capability_ids),
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }
        return {**record, "resolution_id": self.resolution_id} if include_id else record


@dataclass(frozen=True, slots=True)
class CapabilityClosureMatrix:
    family: str
    requirements: tuple[CapabilityClosureRequirement, ...]
    resolutions: tuple[CapabilityGapResolution, ...]

    @classmethod
    def create(cls, family: str, requirements, resolutions, /) -> CapabilityClosureMatrix:
        return cls(
            _capability_name(family, "capability family"),
            tuple(sorted(requirements, key=lambda item: item.requirement_id)),
            tuple(sorted(resolutions, key=lambda item: item.requirement_id)),
        )

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("Closure matrices require requirements.")
        requirement_ids = tuple(value.requirement_id for value in self.requirements)
        resolution_ids = tuple(value.requirement_id for value in self.resolutions)
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("Closure requirement IDs must be unique.")
        if len(set(resolution_ids)) != len(resolution_ids):
            raise ValueError("Closure requirements cannot have multiple resolutions.")
        if set(resolution_ids).difference(requirement_ids):
            raise ValueError("Closure resolutions reference unknown requirements.")

    @property
    def unclassified_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {value.requirement_id for value in self.requirements}.difference(
                    value.requirement_id for value in self.resolutions
                )
            )
        )

    @property
    def closed(self) -> bool:
        return not self.unclassified_requirement_ids and all(
            value.disposition is not ClosureDisposition.MISSING
            for value in self.resolutions
        )

    @property
    def matrix_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record = {
            "kind": "capability-closure-matrix",
            "family": self.family,
            "requirements": [value.to_record() for value in self.requirements],
            "resolutions": [value.to_record() for value in self.resolutions],
            "unclassified_requirement_ids": list(self.unclassified_requirement_ids),
            "closed": self.closed,
        }
        return {**record, "matrix_id": self.matrix_id} if include_id else record


__all__ = [
    "CapabilityClosureMatrix",
    "CapabilityClosureRequirement",
    "CapabilityGapResolution",
    "CarrierRepresentation",
    "ClosureDisposition",
    "CouplingLocation",
    "ExecutionRegime",
    "ImplementationOwnership",
    "PhysicsField",
    "SourceAbsorptionLedger",
    "SourceReference",
    "SourceReuseClass",
    "TopologyRegime",
    "WorkflowClass",
]
