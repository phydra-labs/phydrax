#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical capability disposition, evidence, and ownership declarations.

A capability declaration is intentionally not a release decision.  Release remains
owned by :class:`CapabilityProfile`, :class:`ReleaseIndex`, and the configured trust
policy.  This catalog gives every public or intentionally internal capability one
owner and one honest global disposition while retaining domain-specific maturity
labels as metadata.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .._fingerprint import canonical_fingerprint
from ._registry import _capability_name, _identifier, CapabilityProfile


class CapabilityDisposition(StrEnum):
    """Repository-wide visibility and release disposition."""

    INTERNAL = "internal"
    RESEARCH = "research"
    CANDIDATE = "candidate"
    RELEASED = "released"
    RETIRED = "retired"


class EvidenceDimension(StrEnum):
    """Orthogonal evidence axes; passing one never implies another."""

    IMPLEMENTATION = "implementation"
    NUMERICAL = "numerical"
    DERIVATIVE = "derivative"
    PERFORMANCE = "performance"
    HARDWARE_PROVIDER = "hardware-provider"
    SCIENTIFIC = "scientific"
    OPERATIONS = "operations"
    RIGHTS_SECURITY = "rights-security"
    RELEASE = "release"


class EvidenceState(StrEnum):
    """Current state of one evidence dimension."""

    UNASSESSED = "unassessed"
    BLOCKED = "blocked"
    FAILED = "failed"
    PASSED = "passed"


_ALL_EVIDENCE_DIMENSIONS = tuple(EvidenceDimension)


def _canonical_items(values: Iterable[str], label: str, /) -> tuple[str, ...]:
    normalized = tuple(sorted(_identifier(value, label) for value in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} values must be unique.")
    return normalized


def _repository_path(value: str, label: str, /) -> str:
    normalized = _identifier(value, label)
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise ValueError(f"{label} must be a repository-relative path.")
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Evidence state for one orthogonal qualification dimension."""

    dimension: EvidenceDimension
    state: EvidenceState
    evidence_ids: tuple[str, ...]
    reason: str

    def __init__(
        self,
        dimension: EvidenceDimension | str,
        state: EvidenceState | str,
        /,
        *,
        evidence_ids: Sequence[str] = (),
        reason: str = "not-assessed",
    ):
        dimension_ = EvidenceDimension(dimension)
        state_ = EvidenceState(state)
        evidence_ = _canonical_items(evidence_ids, "evidence ID")
        reason_ = _identifier(reason, "evidence reason")
        if state_ is EvidenceState.PASSED and not evidence_:
            raise ValueError("Passed evidence dimensions must cite retained evidence.")
        if state_ is not EvidenceState.PASSED and evidence_:
            raise ValueError(
                "Only passed evidence dimensions may cite accepted evidence IDs."
            )
        object.__setattr__(self, "dimension", dimension_)
        object.__setattr__(self, "state", state_)
        object.__setattr__(self, "evidence_ids", evidence_)
        object.__setattr__(self, "reason", reason_)

    def to_record(self) -> dict[str, object]:
        return {
            "dimension": self.dimension.value,
            "state": self.state.value,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> EvidenceAssessment:
        evidence_ids = record["evidence_ids"]
        if not isinstance(evidence_ids, Sequence) or isinstance(evidence_ids, str):
            raise TypeError("Serialized evidence IDs must be a sequence.")
        return cls(
            str(record["dimension"]),
            str(record["state"]),
            evidence_ids=tuple(str(value) for value in evidence_ids),
            reason=str(record["reason"]),
        )


@dataclass(frozen=True, slots=True)
class CapabilityDeclaration:
    """One authoritative capability owner and its current honest disposition."""

    capability: str
    owner: str
    disposition: CapabilityDisposition
    domain_maturity: str
    public_symbols: tuple[str, ...]
    profiles: tuple[CapabilityProfile, ...]
    dependencies: tuple[str, ...]
    evidence: tuple[EvidenceAssessment, ...]
    documentation: tuple[str, ...]
    examples: tuple[str, ...]
    intended_uses: tuple[str, ...]
    nonclaims: tuple[str, ...]
    declaration_id: str

    def __init__(
        self,
        capability: str,
        owner: str,
        disposition: CapabilityDisposition | str,
        /,
        *,
        domain_maturity: str = "unspecified",
        public_symbols: Sequence[str] = (),
        profiles: Sequence[CapabilityProfile] = (),
        dependencies: Sequence[str] = (),
        evidence: Sequence[EvidenceAssessment] = (),
        documentation: Sequence[str] = (),
        examples: Sequence[str] = (),
        intended_uses: Sequence[str] = (),
        nonclaims: Sequence[str] = (),
    ):
        capability_ = _capability_name(capability, "capability")
        owner_ = _identifier(owner, "capability owner")
        disposition_ = CapabilityDisposition(disposition)
        maturity_ = _identifier(domain_maturity, "domain maturity")
        symbols_ = _canonical_items(public_symbols, "public symbol")
        if any(not symbol.startswith("phydrax.") for symbol in symbols_):
            raise ValueError("Public symbols must use the canonical phydrax namespace.")
        profiles_ = tuple(sorted(profiles, key=lambda item: item.profile_id))
        if any(not isinstance(item, CapabilityProfile) for item in profiles_):
            raise TypeError("profiles must contain CapabilityProfile values.")
        if any(item.capability != capability_ for item in profiles_):
            raise ValueError("Every profile must describe the declared capability.")
        profile_ids = tuple(item.profile_id for item in profiles_)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("Capability declarations cannot contain duplicate profiles.")
        dependencies_ = tuple(
            sorted(
                _capability_name(value, "capability dependency") for value in dependencies
            )
        )
        if len(set(dependencies_)) != len(dependencies_):
            raise ValueError("Capability dependencies must be unique.")
        if capability_ in dependencies_:
            raise ValueError("A capability cannot depend on itself.")
        evidence_ = tuple(sorted(evidence, key=lambda item: item.dimension.value))
        if any(not isinstance(item, EvidenceAssessment) for item in evidence_):
            raise TypeError("evidence must contain EvidenceAssessment values.")
        dimensions = tuple(item.dimension for item in evidence_)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("A capability has duplicate evidence dimensions.")
        documents_ = tuple(
            sorted(
                _repository_path(value, "documentation path") for value in documentation
            )
        )
        examples_ = tuple(
            sorted(_repository_path(value, "example path") for value in examples)
        )
        uses_ = _canonical_items(intended_uses, "intended use")
        nonclaims_ = _canonical_items(nonclaims, "nonclaim")
        released_profiles = tuple(profile for profile in profiles_ if profile.released)
        if disposition_ is CapabilityDisposition.RELEASED:
            if not profiles_ or len(released_profiles) != len(profiles_):
                raise ValueError(
                    "Released declarations require only explicitly released profiles."
                )
            by_dimension = {item.dimension: item for item in evidence_}
            missing = tuple(
                dimension.value
                for dimension in _ALL_EVIDENCE_DIMENSIONS
                if dimension not in by_dimension
                or by_dimension[dimension].state is not EvidenceState.PASSED
            )
            if missing:
                raise ValueError(
                    "Released declarations require passed evidence in every dimension: "
                    + ", ".join(missing)
                )
            if not symbols_ or not documents_ or not examples_:
                raise ValueError(
                    "Released declarations require public symbols, docs, and examples."
                )
        elif released_profiles:
            raise ValueError(
                "A non-released declaration cannot contain a released profile."
            )
        if disposition_ is CapabilityDisposition.CANDIDATE and not profiles_:
            raise ValueError("Candidate declarations require exact candidate profiles.")
        if (
            disposition_
            in (
                CapabilityDisposition.RESEARCH,
                CapabilityDisposition.CANDIDATE,
            )
            and not nonclaims_
        ):
            raise ValueError("Research and candidate declarations require nonclaims.")
        if (
            disposition_
            in (
                CapabilityDisposition.INTERNAL,
                CapabilityDisposition.RETIRED,
            )
            and symbols_
        ):
            raise ValueError(
                "Internal and retired capabilities cannot expose public symbols."
            )
        object.__setattr__(self, "capability", capability_)
        object.__setattr__(self, "owner", owner_)
        object.__setattr__(self, "disposition", disposition_)
        object.__setattr__(self, "domain_maturity", maturity_)
        object.__setattr__(self, "public_symbols", symbols_)
        object.__setattr__(self, "profiles", profiles_)
        object.__setattr__(self, "dependencies", dependencies_)
        object.__setattr__(self, "evidence", evidence_)
        object.__setattr__(self, "documentation", documents_)
        object.__setattr__(self, "examples", examples_)
        object.__setattr__(self, "intended_uses", uses_)
        object.__setattr__(self, "nonclaims", nonclaims_)
        object.__setattr__(
            self,
            "declaration_id",
            canonical_fingerprint(self._content_record()),
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "capability-declaration",
            "capability": self.capability,
            "owner": self.owner,
            "disposition": self.disposition.value,
            "domain_maturity": self.domain_maturity,
            "public_symbols": list(self.public_symbols),
            "profiles": [profile.to_record() for profile in self.profiles],
            "dependencies": list(self.dependencies),
            "evidence": [item.to_record() for item in self.evidence],
            "documentation": list(self.documentation),
            "examples": list(self.examples),
            "intended_uses": list(self.intended_uses),
            "nonclaims": list(self.nonclaims),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "declaration_id": self.declaration_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CapabilityDeclaration:
        def strings(name: str) -> tuple[str, ...]:
            values = record[name]
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TypeError(f"Serialized {name} must be a sequence.")
            return tuple(str(value) for value in values)

        profile_records = record["profiles"]
        evidence_records = record["evidence"]
        if not isinstance(profile_records, Sequence) or isinstance(profile_records, str):
            raise TypeError("Serialized profiles must be a sequence.")
        if not isinstance(evidence_records, Sequence) or isinstance(
            evidence_records, str
        ):
            raise TypeError("Serialized evidence must be a sequence.")
        value = cls(
            str(record["capability"]),
            str(record["owner"]),
            str(record["disposition"]),
            domain_maturity=str(record["domain_maturity"]),
            public_symbols=strings("public_symbols"),
            profiles=tuple(
                CapabilityProfile.from_record(item)
                for item in profile_records
                if isinstance(item, Mapping)
            ),
            dependencies=strings("dependencies"),
            evidence=tuple(
                EvidenceAssessment.from_record(item)
                for item in evidence_records
                if isinstance(item, Mapping)
            ),
            documentation=strings("documentation"),
            examples=strings("examples"),
            intended_uses=strings("intended_uses"),
            nonclaims=strings("nonclaims"),
        )
        if value.declaration_id != record.get("declaration_id"):
            raise ValueError("Capability declaration content address is invalid.")
        return value


@dataclass(frozen=True, slots=True)
class CapabilityCatalog:
    """Validated, deterministic repository-wide capability graph."""

    declarations: tuple[CapabilityDeclaration, ...]
    catalog_id: str

    def __init__(self, declarations: Sequence[CapabilityDeclaration], /):
        declarations_ = tuple(sorted(declarations, key=lambda item: item.capability))
        if not declarations_ or any(
            not isinstance(item, CapabilityDeclaration) for item in declarations_
        ):
            raise TypeError(
                "declarations must contain at least one CapabilityDeclaration."
            )
        capabilities = tuple(item.capability for item in declarations_)
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("A capability catalog cannot contain duplicate owners.")
        known = set(capabilities)
        for declaration in declarations_:
            missing = tuple(
                dependency
                for dependency in declaration.dependencies
                if dependency not in known
            )
            if missing:
                raise ValueError(
                    f"Capability {declaration.capability} has unknown dependencies: "
                    + ", ".join(missing)
                )
        self._require_acyclic(declarations_)
        symbol_owner: dict[str, str] = {}
        profile_owner: dict[str, str] = {}
        for declaration in declarations_:
            for symbol in declaration.public_symbols:
                if symbol in symbol_owner:
                    raise ValueError(
                        f"Public symbol {symbol} is owned by both {symbol_owner[symbol]} and {declaration.capability}."
                    )
                symbol_owner[symbol] = declaration.capability
            for profile in declaration.profiles:
                if profile.profile_id in profile_owner:
                    raise ValueError(
                        f"Profile {profile.profile_id} is owned by both "
                        f"{profile_owner[profile.profile_id]} and {declaration.capability}."
                    )
                profile_owner[profile.profile_id] = declaration.capability
        object.__setattr__(self, "declarations", declarations_)
        object.__setattr__(
            self,
            "catalog_id",
            canonical_fingerprint(self._content_record()),
        )

    @staticmethod
    def _require_acyclic(declarations: Sequence[CapabilityDeclaration]) -> None:
        graph = {item.capability: item.dependencies for item in declarations}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(capability: str) -> None:
            if capability in visiting:
                raise ValueError(f"Capability dependency cycle includes {capability}.")
            if capability in visited:
                return
            visiting.add(capability)
            for dependency in graph[capability]:
                visit(dependency)
            visiting.remove(capability)
            visited.add(capability)

        for capability in graph:
            visit(capability)

    def declaration(self, capability: str, /) -> CapabilityDeclaration:
        capability_ = _capability_name(capability, "capability")
        for declaration in self.declarations:
            if declaration.capability == capability_:
                return declaration
        raise KeyError(f"Unknown capability {capability_!r}.")

    def by_disposition(
        self, disposition: CapabilityDisposition | str, /
    ) -> tuple[CapabilityDeclaration, ...]:
        disposition_ = CapabilityDisposition(disposition)
        return tuple(
            item for item in self.declarations if item.disposition is disposition_
        )

    def symbol_owner(self, symbol: str, /) -> CapabilityDeclaration:
        symbol_ = _identifier(symbol, "public symbol")
        owners = tuple(
            declaration
            for declaration in self.declarations
            if symbol_ in declaration.public_symbols
        )
        if len(owners) != 1:
            raise KeyError(f"Public symbol {symbol_!r} has no unique capability owner.")
        return owners[0]

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "capability-catalog",
            "declarations": [item.to_record() for item in self.declarations],
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "catalog_id": self.catalog_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CapabilityCatalog:
        values = record["declarations"]
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise TypeError("Serialized declarations must be a sequence.")
        catalog = cls(
            tuple(
                CapabilityDeclaration.from_record(item)
                for item in values
                if isinstance(item, Mapping)
            )
        )
        if catalog.catalog_id != record.get("catalog_id"):
            raise ValueError("Capability catalog content address is invalid.")
        return catalog


def declarations_from_profiles(
    profiles: Iterable[CapabilityProfile],
    /,
    *,
    owners: Mapping[str, str],
    documentation: Mapping[str, Sequence[str]] | None = None,
    examples: Mapping[str, Sequence[str]] | None = None,
    public_symbols: Mapping[str, Sequence[str]] | None = None,
    domain_maturity: Mapping[str, str] | None = None,
    nonclaim: str = "unreleased-candidate",
) -> tuple[CapabilityDeclaration, ...]:
    """Group exact evidence-free profiles into canonical candidate declarations."""

    grouped: dict[str, list[CapabilityProfile]] = defaultdict(list)
    seen_profiles: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, CapabilityProfile):
            raise TypeError("profiles must contain CapabilityProfile values.")
        if profile.profile_id in seen_profiles:
            continue
        if profile.released:
            raise ValueError("Candidate discovery cannot absorb released profiles.")
        seen_profiles.add(profile.profile_id)
        grouped[profile.capability].append(profile)
    documentation_ = {} if documentation is None else documentation
    examples_ = {} if examples is None else examples
    symbols_ = {} if public_symbols is None else public_symbols
    maturity_ = {} if domain_maturity is None else domain_maturity
    declarations = []
    for capability, grouped_profiles in grouped.items():
        if capability not in owners:
            raise KeyError(f"No canonical owner declared for {capability!r}.")
        declarations.append(
            CapabilityDeclaration(
                capability,
                owners[capability],
                CapabilityDisposition.CANDIDATE,
                domain_maturity=maturity_.get(capability, "candidate"),
                public_symbols=symbols_.get(capability, ()),
                profiles=grouped_profiles,
                documentation=documentation_.get(capability, ()),
                examples=examples_.get(capability, ()),
                intended_uses=("bounded-engineering-evaluation",),
                nonclaims=(nonclaim,),
            )
        )
    return tuple(sorted(declarations, key=lambda item: item.capability))


__all__ = [
    "CapabilityCatalog",
    "CapabilityDeclaration",
    "CapabilityDisposition",
    "EvidenceAssessment",
    "EvidenceDimension",
    "EvidenceState",
    "declarations_from_profiles",
]
