#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .._fingerprint import canonical_fingerprint


_ALLOWED_OCCURRENCE_KINDS = frozenset(
    {
        "assembly",
        "part",
        "solid",
        "shell",
        "face",
        "wire",
        "edge",
        "coedge",
        "vertex",
    }
)


def _identifier(name: str, value: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class AssociationStatus(str, Enum):
    """Proof status of one exact, revision-to-revision correspondence query."""

    UNIQUE = "unique"
    NO_PREIMAGE = "no-preimage"
    MULTIPLE = "multiple"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class CADOccurrence:
    """One revision-scoped CAD occurrence, distinct from its shared entity."""

    revision_id: str
    occurrence_id: str
    entity_id: str
    kind: str
    path: tuple[str, ...]
    parent_occurrence_id: str | None = None
    orientation: int = 1

    def __post_init__(self) -> None:
        revision_id = _identifier("revision_id", self.revision_id)
        occurrence_id = _identifier("occurrence_id", self.occurrence_id)
        entity_id = _identifier("entity_id", self.entity_id)
        kind = str(self.kind)
        path = tuple(str(value) for value in self.path)
        parent = (
            None
            if self.parent_occurrence_id is None
            else _identifier("parent_occurrence_id", self.parent_occurrence_id)
        )
        orientation = int(self.orientation)
        if kind not in _ALLOWED_OCCURRENCE_KINDS:
            raise ValueError(f"Unsupported CAD occurrence kind {kind!r}.")
        if not path or any(not value for value in path) or path[-1] != occurrence_id:
            raise ValueError(
                "CAD occurrence path must be non-empty and end at occurrence_id."
            )
        if (parent is None) != (len(path) == 1):
            raise ValueError(
                "Root CAD occurrences have one path component; all others need a parent."
            )
        if orientation not in (-1, 1):
            raise ValueError("CAD occurrence orientation must be +1 or -1.")
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "occurrence_id", occurrence_id)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "parent_occurrence_id", parent)
        object.__setattr__(self, "orientation", orientation)


@dataclass(frozen=True, slots=True)
class CADSelector:
    """Exact occurrence selector; it is never interpreted geometrically."""

    revision_id: str
    occurrence_id: str
    kind: str
    path: tuple[str, ...]
    entity_id: str

    @classmethod
    def from_occurrence(cls, occurrence: CADOccurrence, /) -> CADSelector:
        if not isinstance(occurrence, CADOccurrence):
            raise TypeError("occurrence must be a CADOccurrence.")
        return cls(
            occurrence.revision_id,
            occurrence.occurrence_id,
            occurrence.kind,
            occurrence.path,
            occurrence.entity_id,
        )

    def __post_init__(self) -> None:
        revision_id = _identifier("revision_id", self.revision_id)
        occurrence_id = _identifier("occurrence_id", self.occurrence_id)
        entity_id = _identifier("entity_id", self.entity_id)
        kind = str(self.kind)
        path = tuple(str(value) for value in self.path)
        if kind not in _ALLOWED_OCCURRENCE_KINDS:
            raise ValueError(f"Unsupported CAD selector kind {kind!r}.")
        if not path or path[-1] != occurrence_id or any(not value for value in path):
            raise ValueError("CAD selector path must end at occurrence_id.")
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "occurrence_id", occurrence_id)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class CADRevision:
    """Immutable inventory against which exact occurrence selectors are composed."""

    revision_id: str
    source_id: str
    occurrences: tuple[CADOccurrence, ...]
    provenance_id: str

    def __post_init__(self) -> None:
        revision_id = _identifier("revision_id", self.revision_id)
        source_id = _identifier("source_id", self.source_id)
        provenance_id = _identifier("provenance_id", self.provenance_id)
        occurrences = tuple(self.occurrences)
        if not occurrences:
            raise ValueError("A CAD revision requires at least one occurrence.")
        if any(not isinstance(value, CADOccurrence) for value in occurrences):
            raise TypeError("occurrences must contain only CADOccurrence values.")
        if any(value.revision_id != revision_id for value in occurrences):
            raise ValueError("Every CAD occurrence must belong to this revision.")
        by_id = {value.occurrence_id: value for value in occurrences}
        if len(by_id) != len(occurrences):
            raise ValueError("CAD occurrence IDs must be unique within a revision.")
        if len({value.path for value in occurrences}) != len(occurrences):
            raise ValueError("CAD occurrence paths must be unique within a revision.")
        for occurrence in occurrences:
            parent_id = occurrence.parent_occurrence_id
            if parent_id is None:
                continue
            if parent_id not in by_id:
                raise ValueError(
                    f"CAD occurrence {occurrence.occurrence_id!r} has an unknown parent."
                )
            parent = by_id[parent_id]
            if occurrence.path != parent.path + (occurrence.occurrence_id,):
                raise ValueError(
                    "CAD occurrence paths must be exact parent-path compositions."
                )
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "provenance_id", provenance_id)

    def occurrence(self, occurrence_id: str, /) -> CADOccurrence:
        identifier = _identifier("occurrence_id", occurrence_id)
        for occurrence in self.occurrences:
            if occurrence.occurrence_id == identifier:
                return occurrence
        raise KeyError(f"Unknown occurrence {identifier!r} in CAD revision.")

    def select(self, occurrence_id: str, /) -> CADSelector:
        return CADSelector.from_occurrence(self.occurrence(occurrence_id))

    def compose(
        self,
        parent: CADSelector,
        child_occurrence_id: str,
        /,
        *,
        kind: str | None = None,
    ) -> CADSelector:
        """Compose a face/edge/coedge path by exact stored incidence only."""
        if not isinstance(parent, CADSelector):
            raise TypeError("parent must be a CADSelector.")
        if parent.revision_id != self.revision_id:
            raise ValueError("Cannot compose selectors from another CAD revision.")
        stored_parent = self.occurrence(parent.occurrence_id)
        if CADSelector.from_occurrence(stored_parent) != parent:
            raise ValueError("Parent selector does not match the revision inventory.")
        child = self.occurrence(child_occurrence_id)
        if child.parent_occurrence_id != parent.occurrence_id:
            raise ValueError("Requested child is not exactly incident to the parent.")
        if kind is not None and child.kind != str(kind):
            raise ValueError(
                f"Requested {kind!r} child resolves to {child.kind!r}, not an alias."
            )
        return CADSelector.from_occurrence(child)

    def children(
        self, parent: CADSelector, /, *, kind: str | None = None
    ) -> tuple[CADSelector, ...]:
        if parent.revision_id != self.revision_id:
            raise ValueError("Cannot enumerate children from another CAD revision.")
        self.occurrence(parent.occurrence_id)
        expected = None if kind is None else str(kind)
        return tuple(
            CADSelector.from_occurrence(value)
            for value in self.occurrences
            if value.parent_occurrence_id == parent.occurrence_id
            and (expected is None or value.kind == expected)
        )


@dataclass(frozen=True, slots=True)
class OccurrenceCorrespondence:
    """One provider-certified exact edge in an occurrence association graph."""

    source_occurrence_id: str
    target_occurrence_id: str
    evidence_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_occurrence_id",
            _identifier("source_occurrence_id", self.source_occurrence_id),
        )
        object.__setattr__(
            self,
            "target_occurrence_id",
            _identifier("target_occurrence_id", self.target_occurrence_id),
        )
        object.__setattr__(
            self, "evidence_id", _identifier("evidence_id", self.evidence_id)
        )


@dataclass(frozen=True, slots=True)
class AssociationCoverageEvidence:
    """Provider evidence specifying which absence decisions are exhaustive."""

    source_exhaustive: bool
    target_exhaustive: bool
    certificate_id: str
    provider: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_exhaustive", bool(self.source_exhaustive))
        object.__setattr__(self, "target_exhaustive", bool(self.target_exhaustive))
        object.__setattr__(
            self, "certificate_id", _identifier("certificate_id", self.certificate_id)
        )
        object.__setattr__(self, "provider", _identifier("provider", self.provider))


@dataclass(frozen=True, slots=True)
class OccurrenceCorrespondenceTransaction:
    """Closed immutable correspondence transaction between complete CAD inventories."""

    transaction_id: str
    source_revision_id: str
    target_revision_id: str
    correspondences: tuple[OccurrenceCorrespondence, ...]
    unresolved_source_ids: frozenset[str]
    unresolved_target_ids: frozenset[str]
    coverage: AssociationCoverageEvidence

    def __post_init__(self) -> None:
        transaction_id = _identifier("transaction_id", self.transaction_id)
        source_revision_id = _identifier("source_revision_id", self.source_revision_id)
        target_revision_id = _identifier("target_revision_id", self.target_revision_id)
        correspondences = tuple(self.correspondences)
        if source_revision_id == target_revision_id:
            raise ValueError("A correspondence transaction must cross two revisions.")
        if any(
            not isinstance(value, OccurrenceCorrespondence) for value in correspondences
        ):
            raise TypeError(
                "correspondences must contain only OccurrenceCorrespondence values."
            )
        pairs = {
            (value.source_occurrence_id, value.target_occurrence_id)
            for value in correspondences
        }
        if len(pairs) != len(correspondences):
            raise ValueError("A correspondence transaction cannot repeat an edge.")
        unresolved_source = frozenset(
            _identifier("unresolved_source_id", value)
            for value in self.unresolved_source_ids
        )
        unresolved_target = frozenset(
            _identifier("unresolved_target_id", value)
            for value in self.unresolved_target_ids
        )
        if not isinstance(self.coverage, AssociationCoverageEvidence):
            raise TypeError("coverage must be AssociationCoverageEvidence.")
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "source_revision_id", source_revision_id)
        object.__setattr__(self, "target_revision_id", target_revision_id)
        object.__setattr__(self, "correspondences", correspondences)
        object.__setattr__(self, "unresolved_source_ids", unresolved_source)
        object.__setattr__(self, "unresolved_target_ids", unresolved_target)


@dataclass(frozen=True, slots=True)
class AssociationResolution:
    """Exact candidate set plus the proof status governing selector validity."""

    status: AssociationStatus
    direction: str
    queried_selector: CADSelector
    candidate_selectors: tuple[CADSelector, ...]
    relation: str
    transaction_id: str
    coverage_certificate_id: str

    @property
    def valid(self) -> bool:
        return self.status is AssociationStatus.UNIQUE

    @property
    def selector(self) -> CADSelector | None:
        if not self.valid:
            return None
        return self.candidate_selectors[0]

    def require_unique(self) -> CADSelector:
        if self.status is not AssociationStatus.UNIQUE:
            raise ValueError(
                f"CAD selector is invalid after correspondence: {self.status.value}."
            )
        return self.candidate_selectors[0]


@dataclass(frozen=True, slots=True)
class AssociationGraph:
    """Exact bipartite occurrence graph; geometry proximity is never consulted."""

    source_revision: CADRevision
    target_revision: CADRevision
    transaction: OccurrenceCorrespondenceTransaction
    graph_id: str

    def __init__(
        self,
        source_revision: CADRevision,
        target_revision: CADRevision,
        transaction: OccurrenceCorrespondenceTransaction,
        /,
    ):
        if not isinstance(source_revision, CADRevision) or not isinstance(
            target_revision, CADRevision
        ):
            raise TypeError("AssociationGraph requires two CADRevision inventories.")
        if not isinstance(transaction, OccurrenceCorrespondenceTransaction):
            raise TypeError("transaction must be an OccurrenceCorrespondenceTransaction.")
        if (
            transaction.source_revision_id != source_revision.revision_id
            or transaction.target_revision_id != target_revision.revision_id
        ):
            raise ValueError("Correspondence transaction revision IDs do not match.")
        source_ids = {value.occurrence_id for value in source_revision.occurrences}
        target_ids = {value.occurrence_id for value in target_revision.occurrences}
        for edge in transaction.correspondences:
            if edge.source_occurrence_id not in source_ids:
                raise ValueError(
                    f"Correspondence references unknown source {edge.source_occurrence_id!r}."
                )
            if edge.target_occurrence_id not in target_ids:
                raise ValueError(
                    f"Correspondence references unknown target {edge.target_occurrence_id!r}."
                )
        if not transaction.unresolved_source_ids <= source_ids:
            raise ValueError("Transaction has unknown unresolved source occurrences.")
        if not transaction.unresolved_target_ids <= target_ids:
            raise ValueError("Transaction has unknown unresolved target occurrences.")
        edge_sources = {
            value.source_occurrence_id for value in transaction.correspondences
        }
        edge_targets = {
            value.target_occurrence_id for value in transaction.correspondences
        }
        if transaction.unresolved_source_ids & edge_sources:
            raise ValueError(
                "A source occurrence cannot be both resolved and unresolved."
            )
        if transaction.unresolved_target_ids & edge_targets:
            raise ValueError(
                "A target occurrence cannot be both resolved and unresolved."
            )
        graph_id = canonical_fingerprint(
            {
                "kind": "cad-association-graph",
                "source_revision": source_revision.revision_id,
                "target_revision": target_revision.revision_id,
                "transaction": transaction.transaction_id,
                "edges": sorted(
                    (
                        value.source_occurrence_id,
                        value.target_occurrence_id,
                        value.evidence_id,
                    )
                    for value in transaction.correspondences
                ),
                "unresolved_source": sorted(transaction.unresolved_source_ids),
                "unresolved_target": sorted(transaction.unresolved_target_ids),
                "coverage": transaction.coverage.certificate_id,
            }
        )
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "target_revision", target_revision)
        object.__setattr__(self, "transaction", transaction)
        object.__setattr__(self, "graph_id", graph_id)

    def _forward_ids(self, source_id: str, /) -> tuple[str, ...]:
        return tuple(
            edge.target_occurrence_id
            for edge in self.transaction.correspondences
            if edge.source_occurrence_id == source_id
        )

    def _reverse_ids(self, target_id: str, /) -> tuple[str, ...]:
        return tuple(
            edge.source_occurrence_id
            for edge in self.transaction.correspondences
            if edge.target_occurrence_id == target_id
        )

    def resolve_target(self, selector: CADSelector, /) -> AssociationResolution:
        if not isinstance(selector, CADSelector):
            raise TypeError("selector must be a CADSelector.")
        if selector.revision_id != self.source_revision.revision_id:
            raise ValueError("Source selector belongs to another CAD revision.")
        if self.source_revision.select(selector.occurrence_id) != selector:
            raise ValueError("Source selector does not match its revision inventory.")
        target_ids = self._forward_ids(selector.occurrence_id)
        targets = tuple(self.target_revision.select(value) for value in target_ids)
        unresolved = selector.occurrence_id in self.transaction.unresolved_source_ids
        inverse_counts = tuple(len(self._reverse_ids(value)) for value in target_ids)
        if unresolved:
            status = AssociationStatus.UNRESOLVED
            relation = "unresolved"
        elif not target_ids:
            if self.transaction.coverage.source_exhaustive:
                status = AssociationStatus.NO_PREIMAGE
                relation = "deleted"
            else:
                status = AssociationStatus.UNRESOLVED
                relation = "unresolved"
        elif len(target_ids) > 1 and any(value > 1 for value in inverse_counts):
            status = AssociationStatus.MULTIPLE
            relation = "split-merge"
        elif len(target_ids) > 1:
            status = AssociationStatus.MULTIPLE
            relation = "split"
        elif inverse_counts[0] > 1:
            status = AssociationStatus.MULTIPLE
            relation = "merge"
        else:
            status = AssociationStatus.UNIQUE
            relation = "preserved"
        return AssociationResolution(
            status,
            "source-to-target",
            selector,
            targets,
            relation,
            self.transaction.transaction_id,
            self.transaction.coverage.certificate_id,
        )

    def resolve_preimage(self, selector: CADSelector, /) -> AssociationResolution:
        if not isinstance(selector, CADSelector):
            raise TypeError("selector must be a CADSelector.")
        if selector.revision_id != self.target_revision.revision_id:
            raise ValueError("Target selector belongs to another CAD revision.")
        if self.target_revision.select(selector.occurrence_id) != selector:
            raise ValueError("Target selector does not match its revision inventory.")
        source_ids = self._reverse_ids(selector.occurrence_id)
        sources = tuple(self.source_revision.select(value) for value in source_ids)
        unresolved = selector.occurrence_id in self.transaction.unresolved_target_ids
        forward_counts = tuple(len(self._forward_ids(value)) for value in source_ids)
        if unresolved:
            status = AssociationStatus.UNRESOLVED
            relation = "unresolved"
        elif not source_ids:
            if self.transaction.coverage.target_exhaustive:
                status = AssociationStatus.NO_PREIMAGE
                relation = "created"
            else:
                status = AssociationStatus.UNRESOLVED
                relation = "unresolved"
        elif len(source_ids) > 1 and any(value > 1 for value in forward_counts):
            status = AssociationStatus.MULTIPLE
            relation = "split-merge"
        elif len(source_ids) > 1:
            status = AssociationStatus.MULTIPLE
            relation = "merge"
        elif forward_counts[0] > 1:
            status = AssociationStatus.MULTIPLE
            relation = "split"
        else:
            status = AssociationStatus.UNIQUE
            relation = "preserved"
        return AssociationResolution(
            status,
            "target-to-source",
            selector,
            sources,
            relation,
            self.transaction.transaction_id,
            self.transaction.coverage.certificate_id,
        )


__all__ = [
    "AssociationCoverageEvidence",
    "AssociationGraph",
    "AssociationResolution",
    "AssociationStatus",
    "CADOccurrence",
    "CADRevision",
    "CADSelector",
    "OccurrenceCorrespondence",
    "OccurrenceCorrespondenceTransaction",
]
