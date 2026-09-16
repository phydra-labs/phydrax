#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Durable, content-addressed event graphs for finite dark-sector epochs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .._fingerprint import canonical_fingerprint, canonical_json
from ._chunk_repository import (
    ArtifactManifest,
    ArtifactRepository,
    LeaseRecord,
    RepositoryConflictError,
    RepositoryCorruptionError,
)
from ._repository import ObjectNotFoundError


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
ConservationStatus = Literal["conserved", "violated", "incomplete"]
FailureInjector = Callable[[str], None]


def _identifier(value: str, role: str, /) -> str:
    result = str(value).strip()
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{role} must be a non-empty portable identifier.")
    return result


def _digest(value: str, role: str, /) -> str:
    result = str(value)
    if _DIGEST.fullmatch(result) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256 digest.")
    return result


def _optional_digest(value: str | None, role: str, /) -> str | None:
    return None if value is None else _digest(value, role)


def _digests(values: Sequence[str], role: str, /, *, ordered: bool) -> tuple[str, ...]:
    result = tuple(_digest(value, role) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{role} values must be unique.")
    return result if ordered else tuple(sorted(result))


def _identifiers(values: Sequence[str], role: str, /) -> tuple[str, ...]:
    result = tuple(sorted(_identifier(value, role) for value in values))
    if len(set(result)) != len(result):
        raise ValueError(f"{role} values must be unique.")
    return result


def _nonnegative(value: int, role: str, /) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{role} must be a non-negative integer.")
    return value


def checkpoint_content_id(payload: bytes | bytearray | memoryview, /) -> str:
    """Return the immutable identity of exact checkpoint bytes."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("checkpoint payload must be bytes-like.")
    return hashlib.sha256(bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class GlobalEntity:
    """One immutable globally identified physical or bookkeeping entity."""

    entity_kind: str
    species_id: str
    state_digest: str
    frame_id: str
    frame_realization_id: str
    unit_contract_id: str
    rights_id: str
    provenance_ids: tuple[str, ...]
    parent_event_ids: tuple[str, ...]
    entity_id: str

    def __init__(
        self,
        entity_kind: str,
        species_id: str,
        state_digest: str,
        /,
        *,
        frame_id: str,
        frame_realization_id: str,
        unit_contract_id: str,
        rights_id: str,
        provenance_ids: Sequence[str] = (),
        parent_event_ids: Sequence[str] = (),
    ):
        kind = _identifier(entity_kind, "entity_kind")
        species = _identifier(species_id, "species_id")
        state = _digest(state_digest, "state_digest")
        frame = _digest(frame_id, "frame_id")
        frame_realization = _digest(frame_realization_id, "frame_realization_id")
        units = _digest(unit_contract_id, "unit_contract_id")
        rights = _identifier(rights_id, "rights_id")
        provenance = _identifiers(provenance_ids, "provenance ID")
        parents = _digests(parent_event_ids, "parent event ID", ordered=False)
        content: dict[str, object] = {
            "kind": "dark-sector-global-entity",
            "entity_kind": kind,
            "species_id": species,
            "state_digest": state,
            "frame_id": frame,
            "frame_realization_id": frame_realization,
            "unit_contract_id": units,
            "rights_id": rights,
            "provenance_ids": list(provenance),
            "parent_event_ids": list(parents),
        }
        object.__setattr__(self, "entity_kind", kind)
        object.__setattr__(self, "species_id", species)
        object.__setattr__(self, "state_digest", state)
        object.__setattr__(self, "frame_id", frame)
        object.__setattr__(self, "frame_realization_id", frame_realization)
        object.__setattr__(self, "unit_contract_id", units)
        object.__setattr__(self, "rights_id", rights)
        object.__setattr__(self, "provenance_ids", provenance)
        object.__setattr__(self, "parent_event_ids", parents)
        object.__setattr__(self, "entity_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-global-entity",
            "entity_kind": self.entity_kind,
            "species_id": self.species_id,
            "state_digest": self.state_digest,
            "frame_id": self.frame_id,
            "frame_realization_id": self.frame_realization_id,
            "unit_contract_id": self.unit_contract_id,
            "rights_id": self.rights_id,
            "provenance_ids": list(self.provenance_ids),
            "parent_event_ids": list(self.parent_event_ids),
            "entity_id": self.entity_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> GlobalEntity:
        _require_kind(record, "dark-sector-global-entity")
        result = cls(
            _string(record, "entity_kind"),
            _string(record, "species_id"),
            _string(record, "state_digest"),
            frame_id=_string(record, "frame_id"),
            frame_realization_id=_string(record, "frame_realization_id"),
            unit_contract_id=_string(record, "unit_contract_id"),
            rights_id=_string(record, "rights_id"),
            provenance_ids=_string_sequence(record, "provenance_ids"),
            parent_event_ids=_string_sequence(record, "parent_event_ids"),
        )
        _require_identity(record, "entity_id", result.entity_id)
        return result


@dataclass(frozen=True, slots=True)
class GlobalEvent:
    """One immutable interaction vertex with explicit lineage and evidence."""

    event_kind: str
    model_revision_id: str
    input_entity_ids: tuple[str, ...]
    output_entity_ids: tuple[str, ...]
    epoch_sequence: int
    parent_event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    event_id: str

    def __init__(
        self,
        event_kind: str,
        model_revision_id: str,
        input_entity_ids: Sequence[str],
        output_entity_ids: Sequence[str],
        /,
        *,
        epoch_sequence: int,
        parent_event_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
    ):
        kind = _identifier(event_kind, "event_kind")
        revision = _digest(model_revision_id, "model_revision_id")
        inputs = _digests(input_entity_ids, "input entity ID", ordered=True)
        outputs = _digests(output_entity_ids, "output entity ID", ordered=True)
        if not inputs and not outputs:
            raise ValueError(
                "A global event must consume or produce at least one entity."
            )
        if set(inputs) & set(outputs):
            raise ValueError(
                "An event cannot consume and produce the same immutable entity."
            )
        epoch = _nonnegative(epoch_sequence, "epoch_sequence")
        parents = _digests(parent_event_ids, "parent event ID", ordered=False)
        evidence = _identifiers(evidence_ids, "event evidence ID")
        content: dict[str, object] = {
            "kind": "dark-sector-global-event",
            "event_kind": kind,
            "model_revision_id": revision,
            "input_entity_ids": list(inputs),
            "output_entity_ids": list(outputs),
            "epoch_sequence": epoch,
            "parent_event_ids": list(parents),
            "evidence_ids": list(evidence),
        }
        object.__setattr__(self, "event_kind", kind)
        object.__setattr__(self, "model_revision_id", revision)
        object.__setattr__(self, "input_entity_ids", inputs)
        object.__setattr__(self, "output_entity_ids", outputs)
        object.__setattr__(self, "epoch_sequence", epoch)
        object.__setattr__(self, "parent_event_ids", parents)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "event_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-global-event",
            "event_kind": self.event_kind,
            "model_revision_id": self.model_revision_id,
            "input_entity_ids": list(self.input_entity_ids),
            "output_entity_ids": list(self.output_entity_ids),
            "epoch_sequence": self.epoch_sequence,
            "parent_event_ids": list(self.parent_event_ids),
            "evidence_ids": list(self.evidence_ids),
            "event_id": self.event_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> GlobalEvent:
        _require_kind(record, "dark-sector-global-event")
        result = cls(
            _string(record, "event_kind"),
            _string(record, "model_revision_id"),
            _string_sequence(record, "input_entity_ids"),
            _string_sequence(record, "output_entity_ids"),
            epoch_sequence=_integer(record, "epoch_sequence"),
            parent_event_ids=_string_sequence(record, "parent_event_ids"),
            evidence_ids=_string_sequence(record, "evidence_ids"),
        )
        _require_identity(record, "event_id", result.event_id)
        return result


@dataclass(frozen=True, slots=True)
class GlobalEventEdge:
    """Content-addressed entity flow from one event vertex to another."""

    source_event_id: str
    target_event_id: str
    entity_id: str
    relation: str
    edge_id: str

    def __init__(
        self,
        source_event_id: str,
        target_event_id: str,
        entity_id: str,
        relation: str,
        /,
    ):
        source = _digest(source_event_id, "source_event_id")
        target = _digest(target_event_id, "target_event_id")
        if source == target:
            raise ValueError("Event graph edges cannot be self loops.")
        entity = _digest(entity_id, "entity_id")
        relation_ = _identifier(relation, "edge relation")
        content = {
            "kind": "dark-sector-global-event-edge",
            "source_event_id": source,
            "target_event_id": target,
            "entity_id": entity,
            "relation": relation_,
        }
        object.__setattr__(self, "source_event_id", source)
        object.__setattr__(self, "target_event_id", target)
        object.__setattr__(self, "entity_id", entity)
        object.__setattr__(self, "relation", relation_)
        object.__setattr__(self, "edge_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-global-event-edge",
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "entity_id": self.entity_id,
            "relation": self.relation,
            "edge_id": self.edge_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> GlobalEventEdge:
        _require_kind(record, "dark-sector-global-event-edge")
        result = cls(
            _string(record, "source_event_id"),
            _string(record, "target_event_id"),
            _string(record, "entity_id"),
            _string(record, "relation"),
        )
        _require_identity(record, "edge_id", result.edge_id)
        return result


@dataclass(frozen=True, slots=True)
class GlobalWorkItem:
    """Immutable semantic work identity, independent of lane and worker placement."""

    operation: str
    input_entity_ids: tuple[str, ...]
    model_revision_id: str
    partition_key: str
    priority: int
    parent_work_id: str | None
    work_id: str

    def __init__(
        self,
        operation: str,
        input_entity_ids: Sequence[str],
        model_revision_id: str,
        /,
        *,
        partition_key: str,
        priority: int = 0,
        parent_work_id: str | None = None,
    ):
        operation_ = _identifier(operation, "work operation")
        inputs = _digests(input_entity_ids, "work input entity ID", ordered=True)
        revision = _digest(model_revision_id, "work model_revision_id")
        partition = _identifier(partition_key, "partition_key")
        if type(priority) is not int:
            raise TypeError("priority must be an integer.")
        parent = _optional_digest(parent_work_id, "parent_work_id")
        content: dict[str, object] = {
            "kind": "dark-sector-global-work-item",
            "operation": operation_,
            "input_entity_ids": list(inputs),
            "model_revision_id": revision,
            "partition_key": partition,
            "priority": priority,
            "parent_work_id": parent,
        }
        object.__setattr__(self, "operation", operation_)
        object.__setattr__(self, "input_entity_ids", inputs)
        object.__setattr__(self, "model_revision_id", revision)
        object.__setattr__(self, "partition_key", partition)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "parent_work_id", parent)
        object.__setattr__(self, "work_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-global-work-item",
            "operation": self.operation,
            "input_entity_ids": list(self.input_entity_ids),
            "model_revision_id": self.model_revision_id,
            "partition_key": self.partition_key,
            "priority": self.priority,
            "parent_work_id": self.parent_work_id,
            "work_id": self.work_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> GlobalWorkItem:
        _require_kind(record, "dark-sector-global-work-item")
        result = cls(
            _string(record, "operation"),
            _string_sequence(record, "input_entity_ids"),
            _string(record, "model_revision_id"),
            partition_key=_string(record, "partition_key"),
            priority=_integer(record, "priority"),
            parent_work_id=_optional_string(record.get("parent_work_id")),
        )
        _require_identity(record, "work_id", result.work_id)
        return result


@dataclass(frozen=True, slots=True)
class EventGraphEpochManifest:
    """Exactly-once global manifest for one finite resident epoch."""

    run_id: str
    epoch_sequence: int
    parent_manifest_id: str | None
    plan_id: str
    compile_signature_id: str
    capacity_revision_id: str
    species_revision_id: str
    topology_revision_id: str
    entity_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    work_ids: tuple[str, ...]
    deferred_work_ids: tuple[str, ...]
    matrix_element_revision_id: str
    checkpoint_id: str
    commit_owner_id: str
    conservation_status: ConservationStatus
    evidence_ids: tuple[str, ...]
    epoch_manifest_id: str

    def __init__(
        self,
        run_id: str,
        epoch_sequence: int,
        parent_manifest_id: str | None,
        plan_id: str,
        compile_signature_id: str,
        capacity_revision_id: str,
        species_revision_id: str,
        topology_revision_id: str,
        /,
        *,
        entity_ids: Sequence[str] = (),
        event_ids: Sequence[str] = (),
        edge_ids: Sequence[str] = (),
        work_ids: Sequence[str] = (),
        deferred_work_ids: Sequence[str] = (),
        matrix_element_revision_id: str,
        checkpoint_id: str,
        commit_owner_id: str,
        conservation_status: ConservationStatus,
        evidence_ids: Sequence[str] = (),
    ):
        run = _identifier(run_id, "run_id")
        epoch = _nonnegative(epoch_sequence, "epoch_sequence")
        parent = _optional_digest(parent_manifest_id, "parent_manifest_id")
        if (epoch == 0) != (parent is None):
            raise ValueError("Only epoch zero may omit a parent manifest.")
        plans = tuple(
            _digest(value, role)
            for value, role in (
                (plan_id, "plan_id"),
                (compile_signature_id, "compile_signature_id"),
                (capacity_revision_id, "capacity_revision_id"),
                (species_revision_id, "species_revision_id"),
                (topology_revision_id, "topology_revision_id"),
            )
        )
        entities = _digests(entity_ids, "manifest entity ID", ordered=False)
        events = _digests(event_ids, "manifest event ID", ordered=False)
        edges = _digests(edge_ids, "manifest edge ID", ordered=False)
        work = _digests(work_ids, "manifest work ID", ordered=False)
        deferred = _digests(deferred_work_ids, "manifest deferred work ID", ordered=True)
        if not set(deferred).issubset(set(work)):
            raise ValueError("Every deferred work ID must be present in work_ids.")
        matrix = _digest(matrix_element_revision_id, "matrix_element_revision_id")
        checkpoint = _digest(checkpoint_id, "checkpoint_id")
        owner = _identifier(commit_owner_id, "commit_owner_id")
        if conservation_status not in ("conserved", "violated", "incomplete"):
            raise ValueError("conservation_status is not recognized.")
        evidence = _identifiers(evidence_ids, "epoch evidence ID")
        content: dict[str, object] = {
            "kind": "dark-sector-event-graph-epoch",
            "run_id": run,
            "epoch_sequence": epoch,
            "parent_manifest_id": parent,
            "plan_id": plans[0],
            "compile_signature_id": plans[1],
            "capacity_revision_id": plans[2],
            "species_revision_id": plans[3],
            "topology_revision_id": plans[4],
            "entity_ids": list(entities),
            "event_ids": list(events),
            "edge_ids": list(edges),
            "work_ids": list(work),
            "deferred_work_ids": list(deferred),
            "matrix_element_revision_id": matrix,
            "checkpoint_id": checkpoint,
            "commit_owner_id": owner,
            "conservation_status": conservation_status,
            "evidence_ids": list(evidence),
        }
        object.__setattr__(self, "run_id", run)
        object.__setattr__(self, "epoch_sequence", epoch)
        object.__setattr__(self, "parent_manifest_id", parent)
        object.__setattr__(self, "plan_id", plans[0])
        object.__setattr__(self, "compile_signature_id", plans[1])
        object.__setattr__(self, "capacity_revision_id", plans[2])
        object.__setattr__(self, "species_revision_id", plans[3])
        object.__setattr__(self, "topology_revision_id", plans[4])
        object.__setattr__(self, "entity_ids", entities)
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "edge_ids", edges)
        object.__setattr__(self, "work_ids", work)
        object.__setattr__(self, "deferred_work_ids", deferred)
        object.__setattr__(self, "matrix_element_revision_id", matrix)
        object.__setattr__(self, "checkpoint_id", checkpoint)
        object.__setattr__(self, "commit_owner_id", owner)
        object.__setattr__(self, "conservation_status", conservation_status)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "epoch_manifest_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-event-graph-epoch",
            "run_id": self.run_id,
            "epoch_sequence": self.epoch_sequence,
            "parent_manifest_id": self.parent_manifest_id,
            "plan_id": self.plan_id,
            "compile_signature_id": self.compile_signature_id,
            "capacity_revision_id": self.capacity_revision_id,
            "species_revision_id": self.species_revision_id,
            "topology_revision_id": self.topology_revision_id,
            "entity_ids": list(self.entity_ids),
            "event_ids": list(self.event_ids),
            "edge_ids": list(self.edge_ids),
            "work_ids": list(self.work_ids),
            "deferred_work_ids": list(self.deferred_work_ids),
            "matrix_element_revision_id": self.matrix_element_revision_id,
            "checkpoint_id": self.checkpoint_id,
            "commit_owner_id": self.commit_owner_id,
            "conservation_status": self.conservation_status,
            "evidence_ids": list(self.evidence_ids),
            "epoch_manifest_id": self.epoch_manifest_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> EventGraphEpochManifest:
        _require_kind(record, "dark-sector-event-graph-epoch")
        status = _string(record, "conservation_status")
        if status not in ("conserved", "violated", "incomplete"):
            raise ValueError("Serialized conservation status is not recognized.")
        result = cls(
            _string(record, "run_id"),
            _integer(record, "epoch_sequence"),
            _optional_string(record.get("parent_manifest_id")),
            _string(record, "plan_id"),
            _string(record, "compile_signature_id"),
            _string(record, "capacity_revision_id"),
            _string(record, "species_revision_id"),
            _string(record, "topology_revision_id"),
            entity_ids=_string_sequence(record, "entity_ids"),
            event_ids=_string_sequence(record, "event_ids"),
            edge_ids=_string_sequence(record, "edge_ids"),
            work_ids=_string_sequence(record, "work_ids"),
            deferred_work_ids=_string_sequence(record, "deferred_work_ids"),
            matrix_element_revision_id=_string(record, "matrix_element_revision_id"),
            checkpoint_id=_string(record, "checkpoint_id"),
            commit_owner_id=_string(record, "commit_owner_id"),
            conservation_status=status,
            evidence_ids=_string_sequence(record, "evidence_ids"),
        )
        _require_identity(record, "epoch_manifest_id", result.epoch_manifest_id)
        return result


@dataclass(frozen=True, slots=True)
class RunTip:
    run_id: str
    epoch_sequence: int
    epoch_manifest_id: str
    previous_tip_id: str | None
    tip_id: str

    def __init__(
        self,
        run_id: str,
        epoch_sequence: int,
        epoch_manifest_id: str,
        previous_tip_id: str | None,
        /,
    ):
        run = _identifier(run_id, "run_id")
        epoch = _nonnegative(epoch_sequence, "epoch_sequence")
        manifest = _digest(epoch_manifest_id, "epoch_manifest_id")
        previous = _optional_digest(previous_tip_id, "previous_tip_id")
        content = {
            "kind": "dark-sector-run-tip",
            "run_id": run,
            "epoch_sequence": epoch,
            "epoch_manifest_id": manifest,
            "previous_tip_id": previous,
        }
        object.__setattr__(self, "run_id", run)
        object.__setattr__(self, "epoch_sequence", epoch)
        object.__setattr__(self, "epoch_manifest_id", manifest)
        object.__setattr__(self, "previous_tip_id", previous)
        object.__setattr__(self, "tip_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-run-tip",
            "run_id": self.run_id,
            "epoch_sequence": self.epoch_sequence,
            "epoch_manifest_id": self.epoch_manifest_id,
            "previous_tip_id": self.previous_tip_id,
            "tip_id": self.tip_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> RunTip:
        _require_kind(record, "dark-sector-run-tip")
        result = cls(
            _string(record, "run_id"),
            _integer(record, "epoch_sequence"),
            _string(record, "epoch_manifest_id"),
            _optional_string(record.get("previous_tip_id")),
        )
        _require_identity(record, "tip_id", result.tip_id)
        return result


@dataclass(frozen=True, slots=True)
class WorkLease:
    work_id: str
    worker_id: str
    eligible_worker_ids: tuple[str, ...]
    generation: int
    issued_at: int
    expires_at: int
    previous_lease_id: str | None
    lease_id: str

    def __init__(
        self,
        work_id: str,
        worker_id: str,
        eligible_worker_ids: Sequence[str],
        generation: int,
        issued_at: int,
        expires_at: int,
        /,
        *,
        previous_lease_id: str | None = None,
    ):
        work = _digest(work_id, "work_id")
        worker = _identifier(worker_id, "worker_id")
        eligible = _identifiers(eligible_worker_ids, "eligible worker ID")
        if worker not in eligible:
            raise ValueError("The lease worker must be eligible.")
        generation_ = _nonnegative(generation, "lease generation")
        issued = _nonnegative(issued_at, "lease issued_at")
        expires = _nonnegative(expires_at, "lease expires_at")
        if expires <= issued:
            raise ValueError("Work lease expiry must be later than issuance.")
        previous = _optional_digest(previous_lease_id, "previous_lease_id")
        content = {
            "kind": "dark-sector-work-lease",
            "work_id": work,
            "worker_id": worker,
            "eligible_worker_ids": list(eligible),
            "generation": generation_,
            "issued_at": issued,
            "expires_at": expires,
            "previous_lease_id": previous,
        }
        object.__setattr__(self, "work_id", work)
        object.__setattr__(self, "worker_id", worker)
        object.__setattr__(self, "eligible_worker_ids", eligible)
        object.__setattr__(self, "generation", generation_)
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "previous_lease_id", previous)
        object.__setattr__(self, "lease_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "dark-sector-work-lease",
            "work_id": self.work_id,
            "worker_id": self.worker_id,
            "eligible_worker_ids": list(self.eligible_worker_ids),
            "generation": self.generation,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "previous_lease_id": self.previous_lease_id,
            "lease_id": self.lease_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> WorkLease:
        _require_kind(record, "dark-sector-work-lease")
        result = cls(
            _string(record, "work_id"),
            _string(record, "worker_id"),
            _string_sequence(record, "eligible_worker_ids"),
            _integer(record, "generation"),
            _integer(record, "issued_at"),
            _integer(record, "expires_at"),
            previous_lease_id=_optional_string(record.get("previous_lease_id")),
        )
        _require_identity(record, "lease_id", result.lease_id)
        return result


@dataclass(frozen=True, slots=True)
class PersistedEventGraphEpoch:
    manifest: EventGraphEpochManifest
    checkpoint_payload: bytes


@dataclass(frozen=True, slots=True)
class EpochCommitReceipt:
    manifest: EventGraphEpochManifest
    tip: RunTip
    repository_manifest_id: str
    receipt_id: str

    def __init__(
        self,
        manifest: EventGraphEpochManifest,
        tip: RunTip,
        repository_manifest_id: str,
        /,
    ):
        repository_manifest = _digest(repository_manifest_id, "repository_manifest_id")
        if tip.epoch_manifest_id != manifest.epoch_manifest_id:
            raise ValueError("Commit tip does not name the epoch manifest.")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "tip", tip)
        object.__setattr__(self, "repository_manifest_id", repository_manifest)
        object.__setattr__(
            self,
            "receipt_id",
            canonical_fingerprint(
                {
                    "kind": "dark-sector-epoch-commit-receipt",
                    "epoch_manifest_id": manifest.epoch_manifest_id,
                    "tip_id": tip.tip_id,
                    "repository_manifest_id": repository_manifest,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class EventGraphGarbageCollectionReport:
    root_run_ids: tuple[str, ...]
    reachable_epoch_manifest_ids: tuple[str, ...]
    tombstoned_artifact_ids: tuple[str, ...]
    report_id: str

    def __init__(
        self,
        root_run_ids: Sequence[str],
        reachable_epoch_manifest_ids: Sequence[str],
        tombstoned_artifact_ids: Sequence[str],
        /,
    ):
        roots = _identifiers(root_run_ids, "root run ID")
        reachable = _digests(
            reachable_epoch_manifest_ids, "reachable epoch manifest ID", ordered=False
        )
        tombstoned = tuple(
            sorted(_identifier(value, "artifact ID") for value in tombstoned_artifact_ids)
        )
        object.__setattr__(self, "root_run_ids", roots)
        object.__setattr__(self, "reachable_epoch_manifest_ids", reachable)
        object.__setattr__(self, "tombstoned_artifact_ids", tombstoned)
        object.__setattr__(
            self,
            "report_id",
            canonical_fingerprint(
                {
                    "kind": "dark-sector-event-graph-gc-report",
                    "root_run_ids": list(roots),
                    "reachable_epoch_manifest_ids": list(reachable),
                    "tombstoned_artifact_ids": list(tombstoned),
                }
            ),
        )


def deterministic_commit_owner(
    semantic_id: str, eligible_worker_ids: Sequence[str], /
) -> str:
    """Choose one owner by deterministic rendezvous hashing."""

    semantic = _digest(semantic_id, "semantic_id")
    workers = _identifiers(eligible_worker_ids, "eligible worker ID")
    if not workers:
        raise ValueError("At least one eligible worker is required.")
    return min(
        workers,
        key=lambda worker: canonical_fingerprint(
            {
                "kind": "dark-sector-commit-owner",
                "semantic_id": semantic,
                "worker": worker,
            }
        ),
    )


class EventGraphRepository:
    """Append-only logical graph over transactional chunk repository primitives."""

    def __init__(
        self,
        repository: ArtifactRepository,
        /,
        *,
        namespace: str = "dark-sector",
        maximum_record_bytes: int = 64 * 1024 * 1024,
        maximum_lineage_records: int = 1_000_000,
        failure_injector: FailureInjector | None = None,
    ):
        namespace_ = _identifier(namespace, "event graph namespace")
        if len(namespace_) > 64:
            raise ValueError("Event graph namespace cannot exceed 64 characters.")
        if maximum_record_bytes < 1 or maximum_lineage_records < 1:
            raise ValueError("Event graph resource bounds must be positive.")
        self.repository = repository
        self.namespace = namespace_
        self.maximum_record_bytes = int(maximum_record_bytes)
        self.maximum_lineage_records = int(maximum_lineage_records)
        self.failure_injector = failure_injector

    def put_entity(
        self,
        entity: GlobalEntity,
        writer_id: str,
        /,
        *,
        committed_at: int | None = None,
    ) -> ArtifactManifest:
        if not isinstance(entity, GlobalEntity):
            raise TypeError("entity must be GlobalEntity.")
        return self._put_record(
            "entity", entity.entity_id, entity.to_record(), writer_id, committed_at
        )

    def put_event(
        self, event: GlobalEvent, writer_id: str, /, *, committed_at: int | None = None
    ) -> ArtifactManifest:
        if not isinstance(event, GlobalEvent):
            raise TypeError("event must be GlobalEvent.")
        for entity_id in (*event.input_entity_ids, *event.output_entity_ids):
            self.get_entity(entity_id)
        for parent_id in event.parent_event_ids:
            self.get_event(parent_id)
        return self._put_record(
            "event", event.event_id, event.to_record(), writer_id, committed_at
        )

    def put_edge(
        self,
        edge: GlobalEventEdge,
        writer_id: str,
        /,
        *,
        committed_at: int | None = None,
    ) -> ArtifactManifest:
        if not isinstance(edge, GlobalEventEdge):
            raise TypeError("edge must be GlobalEventEdge.")
        source = self.get_event(edge.source_event_id)
        target = self.get_event(edge.target_event_id)
        self.get_entity(edge.entity_id)
        if (
            edge.entity_id not in source.output_entity_ids
            or edge.entity_id not in target.input_entity_ids
        ):
            raise ValueError("Event edge entity flow contradicts its endpoint events.")
        if source.event_id not in target.parent_event_ids:
            raise ValueError("Target event lineage does not include the edge source.")
        return self._put_record(
            "edge", edge.edge_id, edge.to_record(), writer_id, committed_at
        )

    def put_work(
        self,
        work: GlobalWorkItem,
        writer_id: str,
        /,
        *,
        committed_at: int | None = None,
    ) -> ArtifactManifest:
        if not isinstance(work, GlobalWorkItem):
            raise TypeError("work must be GlobalWorkItem.")
        for entity_id in work.input_entity_ids:
            self.get_entity(entity_id)
        if work.parent_work_id is not None:
            self.get_work(work.parent_work_id)
        return self._put_record(
            "work", work.work_id, work.to_record(), writer_id, committed_at
        )

    def get_entity(self, entity_id: str, /) -> GlobalEntity:
        return GlobalEntity.from_record(
            self._read_record("entity", _digest(entity_id, "entity_id"))
        )

    def get_event(self, event_id: str, /) -> GlobalEvent:
        return GlobalEvent.from_record(
            self._read_record("event", _digest(event_id, "event_id"))
        )

    def get_edge(self, edge_id: str, /) -> GlobalEventEdge:
        return GlobalEventEdge.from_record(
            self._read_record("edge", _digest(edge_id, "edge_id"))
        )

    def get_work(self, work_id: str, /) -> GlobalWorkItem:
        return GlobalWorkItem.from_record(
            self._read_record("work", _digest(work_id, "work_id"))
        )

    def run_tip(self, run_id: str, /) -> RunTip | None:
        run = _identifier(run_id, "run_id")
        record = self._read_record_optional(self._tip_artifact_id(run))
        return None if record is None else RunTip.from_record(record)

    def epoch_slot(
        self, run_id: str, epoch_sequence: int, /
    ) -> EventGraphEpochManifest | None:
        """Return an immutable staged/committed epoch occupying a global slot."""

        run = _identifier(run_id, "run_id")
        sequence = _nonnegative(epoch_sequence, "epoch_sequence")
        record = self._read_record_optional(self._slot_artifact_id(run, sequence))
        if record is None:
            return None
        _require_kind(record, "dark-sector-epoch-slot")
        if (
            _string(record, "run_id") != run
            or _integer(record, "epoch_sequence") != sequence
        ):
            raise RepositoryCorruptionError(
                "Epoch slot lookup identity does not match its content."
            )
        return self.load_epoch(_string(record, "epoch_manifest_id")).manifest

    def append_epoch(
        self,
        manifest: EventGraphEpochManifest,
        checkpoint_payload: bytes | bytearray | memoryview,
        /,
        *,
        writer_id: str,
        committed_at: int | None = None,
    ) -> EpochCommitReceipt:
        """Commit an immutable epoch then advance the run tip by compare-and-swap."""

        if not isinstance(manifest, EventGraphEpochManifest):
            raise TypeError("manifest must be EventGraphEpochManifest.")
        payload = bytes(checkpoint_payload)
        if len(payload) > self.maximum_record_bytes:
            raise ValueError("Checkpoint payload exceeds maximum_record_bytes.")
        if checkpoint_content_id(payload) != manifest.checkpoint_id:
            raise ValueError("Checkpoint bytes do not match manifest checkpoint_id.")
        if manifest.conservation_status != "conserved":
            raise ValueError("Only conservation-complete epochs may be committed.")
        self._validate_manifest_graph(manifest)

        epoch_record = {
            "kind": "dark-sector-persisted-epoch",
            "manifest": manifest.to_record(),
            "checkpoint_hex": payload.hex(),
        }
        epoch_artifact_id = self._epoch_artifact_id(manifest.epoch_manifest_id)
        epoch_repository_manifest = self._put_json_artifact(
            epoch_artifact_id, epoch_record, writer_id, committed_at
        )
        slot_record = {
            "kind": "dark-sector-epoch-slot",
            "run_id": manifest.run_id,
            "epoch_sequence": manifest.epoch_sequence,
            "epoch_manifest_id": manifest.epoch_manifest_id,
        }
        self._put_json_artifact(
            self._slot_artifact_id(manifest.run_id, manifest.epoch_sequence),
            slot_record,
            writer_id,
            committed_at,
        )
        self._fail("after_epoch_manifest")

        current = self.run_tip(manifest.run_id)
        if (
            current is not None
            and current.epoch_manifest_id == manifest.epoch_manifest_id
        ):
            return EpochCommitReceipt(
                manifest, current, epoch_repository_manifest.manifest_id
            )
        expected_sequence = 0 if current is None else current.epoch_sequence + 1
        expected_parent = None if current is None else current.epoch_manifest_id
        if (
            manifest.epoch_sequence != expected_sequence
            or manifest.parent_manifest_id != expected_parent
        ):
            raise RepositoryConflictError(
                "Run tip does not match the epoch manifest parent relation."
            )
        tip = RunTip(
            manifest.run_id,
            manifest.epoch_sequence,
            manifest.epoch_manifest_id,
            None if current is None else current.tip_id,
        )
        self._fail("before_tip")
        try:
            self._replace_json_artifact(
                self._tip_artifact_id(manifest.run_id),
                tip.to_record(),
                writer_id,
                committed_at,
            )
        except RepositoryConflictError:
            winner = self.run_tip(manifest.run_id)
            if winner is None or winner.epoch_manifest_id != manifest.epoch_manifest_id:
                raise
            tip = winner
        self._fail("after_tip")
        return EpochCommitReceipt(manifest, tip, epoch_repository_manifest.manifest_id)

    def load_epoch(self, epoch_manifest_id: str, /) -> PersistedEventGraphEpoch:
        identity = _digest(epoch_manifest_id, "epoch_manifest_id")
        record = self._read_record(self._epoch_artifact_id(identity))
        _require_kind(record, "dark-sector-persisted-epoch")
        raw_manifest = record.get("manifest")
        if not isinstance(raw_manifest, Mapping):
            raise RepositoryCorruptionError("Persisted epoch manifest is malformed.")
        manifest = EventGraphEpochManifest.from_record(raw_manifest)
        if manifest.epoch_manifest_id != identity:
            raise RepositoryCorruptionError(
                "Epoch lookup identity does not match content."
            )
        checkpoint_hex = _string(record, "checkpoint_hex")
        try:
            checkpoint = bytes.fromhex(checkpoint_hex)
        except ValueError as error:
            raise RepositoryCorruptionError(
                "Persisted checkpoint encoding is invalid."
            ) from error
        if checkpoint_content_id(checkpoint) != manifest.checkpoint_id:
            raise RepositoryCorruptionError("Persisted checkpoint identity is invalid.")
        return PersistedEventGraphEpoch(manifest, checkpoint)

    def acquire_work_lease(
        self,
        work: GlobalWorkItem,
        worker_id: str,
        /,
        *,
        eligible_worker_ids: Sequence[str],
        issued_at: int,
        expires_at: int,
    ) -> WorkLease:
        """Claim work, allowing deterministic stealing only after lease expiry."""

        if not isinstance(work, GlobalWorkItem):
            raise TypeError("work must be GlobalWorkItem.")
        persisted = self.get_work(work.work_id)
        if persisted != work:
            raise RepositoryCorruptionError(
                "Persisted work content does not match claim."
            )
        eligible = _identifiers(eligible_worker_ids, "eligible worker ID")
        owner = deterministic_commit_owner(work.work_id, eligible)
        worker = _identifier(worker_id, "worker_id")
        if worker != owner:
            raise RepositoryConflictError(
                f"Worker {worker!r} is not deterministic commit owner {owner!r}."
            )
        issued = _nonnegative(issued_at, "issued_at")
        expires = _nonnegative(expires_at, "expires_at")
        artifact_id = self._work_lease_artifact_id(work.work_id)
        current_record = self._read_record_optional(artifact_id)
        current = (
            None if current_record is None else WorkLease.from_record(current_record)
        )
        if current is not None and current.expires_at > issued:
            if current.worker_id == worker and current.eligible_worker_ids == eligible:
                return current
            raise RepositoryConflictError("Work is protected by an unexpired lease.")
        lease = WorkLease(
            work.work_id,
            worker,
            eligible,
            0 if current is None else current.generation + 1,
            issued,
            expires,
            previous_lease_id=None if current is None else current.lease_id,
        )
        try:
            self._replace_json_artifact(artifact_id, lease.to_record(), worker, issued)
        except RepositoryConflictError:
            winner_record = self._read_record_optional(artifact_id)
            winner = (
                None if winner_record is None else WorkLease.from_record(winner_record)
            )
            if winner is None or winner.lease_id != lease.lease_id:
                raise
            return winner
        return lease

    def lease_epoch(
        self,
        epoch_manifest_id: str,
        holder_id: str,
        /,
        *,
        expires_at: int,
        lease_id: str | None = None,
        issued_at: int | None = None,
    ) -> LeaseRecord:
        identity = _digest(epoch_manifest_id, "epoch_manifest_id")
        self.load_epoch(identity)
        return self.repository.acquire_lease(
            self._epoch_artifact_id(identity),
            holder_id,
            expires_at=expires_at,
            lease_id=lease_id,
            issued_at=issued_at,
        )

    def release_epoch_lease(self, lease: LeaseRecord, /) -> None:
        self.repository.release_lease(lease)

    def collect_unreachable(
        self,
        root_run_ids: Sequence[str],
        /,
        *,
        epoch_manifest_ids: Sequence[str] = (),
        entity_ids: Sequence[str] = (),
        event_ids: Sequence[str] = (),
        edge_ids: Sequence[str] = (),
        work_ids: Sequence[str] = (),
        reason: str,
        now: int,
    ) -> EventGraphGarbageCollectionReport:
        """Tombstone only candidates proven unreachable from every supplied run tip."""

        roots = _identifiers(root_run_ids, "root run ID")
        if not roots:
            raise ValueError("Reachability collection requires at least one root run.")
        reachable_epochs: set[str] = set()
        reachable_entities: set[str] = set()
        reachable_events: set[str] = set()
        reachable_edges: set[str] = set()
        reachable_work: set[str] = set()
        for run_id in roots:
            tip = self.run_tip(run_id)
            current = None if tip is None else tip.epoch_manifest_id
            while current is not None:
                if current in reachable_epochs:
                    break
                reachable_epochs.add(current)
                epoch = self.load_epoch(current).manifest
                reachable_entities.update(epoch.entity_ids)
                reachable_events.update(epoch.event_ids)
                reachable_edges.update(epoch.edge_ids)
                reachable_work.update(epoch.work_ids)
                current = epoch.parent_manifest_id
                if len(reachable_epochs) > self.maximum_lineage_records:
                    raise RepositoryCorruptionError(
                        "Reachable epoch lineage exceeds maximum_lineage_records."
                    )
        candidates = (
            (
                "epoch",
                _digests(epoch_manifest_ids, "epoch manifest ID", ordered=False),
                reachable_epochs,
            ),
            (
                "entity",
                _digests(entity_ids, "entity ID", ordered=False),
                reachable_entities,
            ),
            (
                "event",
                _digests(event_ids, "event ID", ordered=False),
                reachable_events,
            ),
            (
                "edge",
                _digests(edge_ids, "edge ID", ordered=False),
                reachable_edges,
            ),
            (
                "work",
                _digests(work_ids, "work ID", ordered=False),
                reachable_work,
            ),
        )
        tombstoned: list[str] = []
        for kind, identities, reachable in candidates:
            for identity in identities:
                if identity in reachable:
                    continue
                if kind == "epoch":
                    epoch = self.load_epoch(identity).manifest
                    artifact_ids = (
                        self._epoch_artifact_id(identity),
                        self._slot_artifact_id(epoch.run_id, epoch.epoch_sequence),
                    )
                else:
                    artifact_ids = (self._record_artifact_id(kind, identity),)
                for artifact_id in artifact_ids:
                    self.repository.tombstone(
                        artifact_id, reason, created_at=now, eligible_at=now
                    )
                    tombstoned.append(artifact_id)
        return EventGraphGarbageCollectionReport(
            roots, tuple(reachable_epochs), tombstoned
        )

    def _validate_manifest_graph(self, manifest: EventGraphEpochManifest, /) -> None:
        for identity in manifest.entity_ids:
            self.get_entity(identity)
        events = {identity: self.get_event(identity) for identity in manifest.event_ids}
        for identity in manifest.edge_ids:
            edge = self.get_edge(identity)
            if edge.source_event_id not in events:
                self.get_event(edge.source_event_id)
            if edge.target_event_id not in events:
                self.get_event(edge.target_event_id)
        for identity in manifest.work_ids:
            self.get_work(identity)
        self._validate_event_lineage(tuple(events))
        if any(
            event.epoch_sequence > manifest.epoch_sequence for event in events.values()
        ):
            raise ValueError("Epoch manifest cannot contain events from a future epoch.")

    def _validate_event_lineage(self, event_ids: Sequence[str], /) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()
        count = 0

        def visit(event_id: str) -> None:
            nonlocal count
            if event_id in visited:
                return
            if event_id in visiting:
                raise ValueError("Event lineage must be acyclic.")
            count += 1
            if count > self.maximum_lineage_records:
                raise RepositoryCorruptionError(
                    "Event lineage exceeds maximum_lineage_records."
                )
            visiting.add(event_id)
            event = self.get_event(event_id)
            for parent_id in event.parent_event_ids:
                visit(parent_id)
            visiting.remove(event_id)
            visited.add(event_id)

        for event_id in event_ids:
            visit(event_id)

    def _put_record(
        self,
        kind: str,
        identity: str,
        record: Mapping[str, object],
        writer_id: str,
        committed_at: int | None,
    ) -> ArtifactManifest:
        return self._put_json_artifact(
            self._record_artifact_id(kind, identity), record, writer_id, committed_at
        )

    def _read_record(
        self, kind_or_artifact: str, identity: str | None = None
    ) -> dict[str, object]:
        artifact_id = (
            kind_or_artifact
            if identity is None
            else self._record_artifact_id(kind_or_artifact, identity)
        )
        payload = self._read_payload(artifact_id)
        return _json_record(payload)

    def _read_record_optional(self, artifact_id: str, /) -> dict[str, object] | None:
        payload = self._read_payload_optional(artifact_id)
        return None if payload is None else _json_record(payload)

    def _put_json_artifact(
        self,
        artifact_id: str,
        record: Mapping[str, object],
        writer_id: str,
        committed_at: int | None,
    ) -> ArtifactManifest:
        payload = (canonical_json(record) + "\n").encode("utf-8")
        if len(payload) > self.maximum_record_bytes:
            raise ValueError("Event graph record exceeds maximum_record_bytes.")
        existing = self._read_payload_optional(artifact_id)
        if existing is not None:
            if existing != payload:
                raise RepositoryConflictError(
                    f"Immutable event graph artifact {artifact_id!r} conflicts."
                )
            return self.repository.get_manifest(artifact_id)
        transaction = self.repository.begin(
            artifact_id,
            _identifier(writer_id, "writer_id"),
            started_at=committed_at,
        )
        chunks = []
        offset = 0
        index = 0
        while offset < len(payload):
            end = min(offset + self.repository.maximum_chunk_bytes, len(payload))
            chunks.append(
                self.repository.write_chunk(
                    transaction,
                    "record",
                    index,
                    offset,
                    payload[offset:end],
                    encoding="identity",
                )
            )
            offset = end
            index += 1
        try:
            return self.repository.commit(
                transaction,
                chunks,
                metadata={"logical-kind": "dark-sector-event-graph"},
                committed_at=committed_at,
            )
        except RepositoryConflictError:
            winner = self._read_payload_optional(artifact_id)
            if winner != payload:
                raise
            return self.repository.get_manifest(artifact_id)

    def _replace_json_artifact(
        self,
        artifact_id: str,
        record: Mapping[str, object],
        writer_id: str,
        committed_at: int | None,
    ) -> ArtifactManifest:
        payload = (canonical_json(record) + "\n").encode("utf-8")
        if len(payload) > self.maximum_record_bytes:
            raise ValueError("Event graph record exceeds maximum_record_bytes.")
        transaction = self.repository.begin(
            artifact_id, _identifier(writer_id, "writer_id"), started_at=committed_at
        )
        chunks = []
        offset = 0
        index = 0
        while offset < len(payload):
            end = min(offset + self.repository.maximum_chunk_bytes, len(payload))
            chunks.append(
                self.repository.write_chunk(
                    transaction, "record", index, offset, payload[offset:end]
                )
            )
            offset = end
            index += 1
        return self.repository.commit(
            transaction,
            chunks,
            metadata={"logical-kind": "dark-sector-event-graph-cas"},
            committed_at=committed_at,
        )

    def _read_payload(self, artifact_id: str, /) -> bytes:
        manifest = self.repository.get_manifest(artifact_id)
        chunks = tuple(
            chunk for chunk in manifest.chunks if chunk.logical_name == "record"
        )
        total = sum(chunk.plaintext_size for chunk in chunks)
        if not chunks or total > self.maximum_record_bytes:
            raise RepositoryCorruptionError("Event graph record violates its byte bound.")
        return b"".join(
            self.repository.read_chunk(
                manifest, chunk, maximum_plaintext_bytes=self.maximum_record_bytes
            )
            for chunk in chunks
        )

    def _read_payload_optional(self, artifact_id: str, /) -> bytes | None:
        try:
            return self._read_payload(artifact_id)
        except ObjectNotFoundError:
            return None

    def _record_artifact_id(self, kind: str, identity: str, /) -> str:
        return f"{self.namespace}.{kind}.{identity}"

    def _epoch_artifact_id(self, identity: str, /) -> str:
        return self._record_artifact_id("epoch", identity)

    def _slot_artifact_id(self, run_id: str, sequence: int, /) -> str:
        run_key = canonical_fingerprint({"kind": "dark-sector-run-key", "run_id": run_id})
        return f"{self.namespace}.slot.{run_key}.{sequence:x}"

    def _tip_artifact_id(self, run_id: str, /) -> str:
        run_key = canonical_fingerprint({"kind": "dark-sector-run-key", "run_id": run_id})
        return f"{self.namespace}.tip.{run_key}"

    def _work_lease_artifact_id(self, work_id: str, /) -> str:
        return self._record_artifact_id("work-lease", work_id)

    def _fail(self, point: str, /) -> None:
        if self.failure_injector is not None:
            self.failure_injector(point)


def _json_record(payload: bytes, /) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RepositoryCorruptionError(
            "Event graph record is not canonical JSON."
        ) from error
    if not isinstance(value, dict):
        raise RepositoryCorruptionError("Event graph record must be a JSON object.")
    if (canonical_json(value) + "\n").encode("utf-8") != payload:
        raise RepositoryCorruptionError("Event graph record is not canonically encoded.")
    return value


def _require_kind(record: Mapping[str, object], expected: str, /) -> None:
    if not isinstance(record, Mapping) or record.get("kind") != expected:
        raise ValueError(f"Expected serialized record kind {expected!r}.")


def _require_identity(record: Mapping[str, object], field: str, expected: str, /) -> None:
    if record.get(field) != expected:
        raise ValueError(f"Serialized {field} does not match record content.")


def _string(record: Mapping[str, object], field: str, /) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"Serialized {field} must be a string.")
    return value


def _optional_string(value: object, /) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError("Serialized optional identity must be a string or null.")
    return value


def _integer(record: Mapping[str, object], field: str, /) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise TypeError(f"Serialized {field} must be an integer.")
    return value


def _string_sequence(record: Mapping[str, object], field: str, /) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Serialized {field} must be a string sequence.")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"Serialized {field} must contain only strings.")
    return tuple(value)


__all__ = [
    "ConservationStatus",
    "EpochCommitReceipt",
    "EventGraphEpochManifest",
    "EventGraphGarbageCollectionReport",
    "EventGraphRepository",
    "GlobalEntity",
    "GlobalEvent",
    "GlobalEventEdge",
    "GlobalWorkItem",
    "PersistedEventGraphEpoch",
    "RunTip",
    "WorkLease",
    "checkpoint_content_id",
    "deterministic_commit_owner",
]
