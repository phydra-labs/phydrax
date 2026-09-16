#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable matrix-element identity and between-epoch adaptation provenance."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _identity(value: str, role: str, /) -> str:
    result = str(value).strip()
    if not result or len(result) > 512:
        raise ValueError(f"{role} must be a non-empty bounded identity.")
    return result


def _digest(value: str, role: str, /) -> str:
    result = str(value)
    if _DIGEST.fullmatch(result) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256 digest.")
    return result


def _optional_digest(value: str | None, role: str, /) -> str | None:
    return None if value is None else _digest(value, role)


@dataclass(frozen=True, slots=True)
class MatrixElementRevision:
    """Complete immutable scientific, provider, rights, and optimizer identity."""

    process_id: str
    model_id: str
    provider_id: str
    rights_id: str
    normalization_id: str
    support_id: str
    proposal_id: str
    adaptation_id: str
    training_data_id: str
    optimizer_id: str
    error_model_id: str
    differentiation_id: str
    parameter_digest: str
    parent_revision_id: str | None
    valid_from_epoch_manifest_id: str | None
    held_out_evidence_id: str | None
    revision_id: str

    def __init__(
        self,
        process_id: str,
        model_id: str,
        provider_id: str,
        rights_id: str,
        normalization_id: str,
        support_id: str,
        proposal_id: str,
        adaptation_id: str,
        training_data_id: str,
        optimizer_id: str,
        error_model_id: str,
        parameter_digest: str,
        /,
        *,
        differentiation_id: str,
        parent_revision_id: str | None = None,
        valid_from_epoch_manifest_id: str | None = None,
        held_out_evidence_id: str | None = None,
    ):
        values = tuple(
            _identity(value, role)
            for value, role in (
                (process_id, "process_id"),
                (model_id, "model_id"),
                (provider_id, "provider_id"),
                (rights_id, "rights_id"),
                (normalization_id, "normalization_id"),
                (support_id, "support_id"),
                (proposal_id, "proposal_id"),
                (adaptation_id, "adaptation_id"),
                (training_data_id, "training_data_id"),
                (optimizer_id, "optimizer_id"),
                (error_model_id, "error_model_id"),
                (differentiation_id, "differentiation_id"),
            )
        )
        parameters = _digest(parameter_digest, "parameter_digest")
        parent = _optional_digest(parent_revision_id, "parent_revision_id")
        boundary = _optional_digest(
            valid_from_epoch_manifest_id, "valid_from_epoch_manifest_id"
        )
        evidence = _optional_digest(held_out_evidence_id, "held_out_evidence_id")
        if len({parent is None, boundary is None, evidence is None}) != 1:
            raise ValueError(
                "Adapted revisions require parent, committed boundary, and "
                "held-out evidence together."
            )
        content: dict[str, object] = {
            "kind": "matrix-element-revision",
            "process_id": values[0],
            "model_id": values[1],
            "provider_id": values[2],
            "rights_id": values[3],
            "normalization_id": values[4],
            "support_id": values[5],
            "proposal_id": values[6],
            "adaptation_id": values[7],
            "training_data_id": values[8],
            "optimizer_id": values[9],
            "error_model_id": values[10],
            "differentiation_id": values[11],
            "parameter_digest": parameters,
            "parent_revision_id": parent,
            "valid_from_epoch_manifest_id": boundary,
            "held_out_evidence_id": evidence,
        }
        for field, value in zip(
            (
                "process_id",
                "model_id",
                "provider_id",
                "rights_id",
                "normalization_id",
                "support_id",
                "proposal_id",
                "adaptation_id",
                "training_data_id",
                "optimizer_id",
                "error_model_id",
                "differentiation_id",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, field, value)
        object.__setattr__(self, "parameter_digest", parameters)
        object.__setattr__(self, "parent_revision_id", parent)
        object.__setattr__(self, "valid_from_epoch_manifest_id", boundary)
        object.__setattr__(self, "held_out_evidence_id", evidence)
        object.__setattr__(self, "revision_id", canonical_fingerprint(content))

    @property
    def is_adapted(self) -> bool:
        return self.parent_revision_id is not None

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "matrix-element-revision",
            "process_id": self.process_id,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "rights_id": self.rights_id,
            "normalization_id": self.normalization_id,
            "support_id": self.support_id,
            "proposal_id": self.proposal_id,
            "adaptation_id": self.adaptation_id,
            "training_data_id": self.training_data_id,
            "optimizer_id": self.optimizer_id,
            "error_model_id": self.error_model_id,
            "differentiation_id": self.differentiation_id,
            "parameter_digest": self.parameter_digest,
            "parent_revision_id": self.parent_revision_id,
            "valid_from_epoch_manifest_id": self.valid_from_epoch_manifest_id,
            "held_out_evidence_id": self.held_out_evidence_id,
            "revision_id": self.revision_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> MatrixElementRevision:
        _require_kind(record, "matrix-element-revision")
        result = cls(
            _string(record, "process_id"),
            _string(record, "model_id"),
            _string(record, "provider_id"),
            _string(record, "rights_id"),
            _string(record, "normalization_id"),
            _string(record, "support_id"),
            _string(record, "proposal_id"),
            _string(record, "adaptation_id"),
            _string(record, "training_data_id"),
            _string(record, "optimizer_id"),
            _string(record, "error_model_id"),
            _string(record, "parameter_digest"),
            differentiation_id=_string(record, "differentiation_id"),
            parent_revision_id=_optional_string(record.get("parent_revision_id")),
            valid_from_epoch_manifest_id=_optional_string(
                record.get("valid_from_epoch_manifest_id")
            ),
            held_out_evidence_id=_optional_string(record.get("held_out_evidence_id")),
        )
        _require_id(record, "revision_id", result.revision_id)
        return result


@dataclass(frozen=True, slots=True)
class MatrixElementAdaptationProposal:
    """Candidate draft proposed strictly after one committed epoch boundary."""

    base_revision_id: str
    candidate_revision_id: str
    training_data_id: str
    optimizer_id: str
    adaptation_id: str
    proposed_after_epoch_manifest_id: str
    proposal_record_id: str

    def __init__(
        self,
        base_revision_id: str,
        candidate_revision_id: str,
        training_data_id: str,
        optimizer_id: str,
        adaptation_id: str,
        proposed_after_epoch_manifest_id: str,
        /,
    ):
        base = _digest(base_revision_id, "base_revision_id")
        candidate = _digest(candidate_revision_id, "candidate_revision_id")
        if base == candidate:
            raise ValueError(
                "An adaptation candidate must differ from its base revision."
            )
        training = _identity(training_data_id, "training_data_id")
        optimizer = _identity(optimizer_id, "optimizer_id")
        adaptation = _identity(adaptation_id, "adaptation_id")
        boundary = _digest(
            proposed_after_epoch_manifest_id, "proposed_after_epoch_manifest_id"
        )
        content = {
            "kind": "matrix-element-adaptation-proposal",
            "base_revision_id": base,
            "candidate_revision_id": candidate,
            "training_data_id": training,
            "optimizer_id": optimizer,
            "adaptation_id": adaptation,
            "proposed_after_epoch_manifest_id": boundary,
        }
        object.__setattr__(self, "base_revision_id", base)
        object.__setattr__(self, "candidate_revision_id", candidate)
        object.__setattr__(self, "training_data_id", training)
        object.__setattr__(self, "optimizer_id", optimizer)
        object.__setattr__(self, "adaptation_id", adaptation)
        object.__setattr__(self, "proposed_after_epoch_manifest_id", boundary)
        object.__setattr__(self, "proposal_record_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "matrix-element-adaptation-proposal",
            "base_revision_id": self.base_revision_id,
            "candidate_revision_id": self.candidate_revision_id,
            "training_data_id": self.training_data_id,
            "optimizer_id": self.optimizer_id,
            "adaptation_id": self.adaptation_id,
            "proposed_after_epoch_manifest_id": self.proposed_after_epoch_manifest_id,
            "proposal_record_id": self.proposal_record_id,
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> MatrixElementAdaptationProposal:
        _require_kind(record, "matrix-element-adaptation-proposal")
        result = cls(
            _string(record, "base_revision_id"),
            _string(record, "candidate_revision_id"),
            _string(record, "training_data_id"),
            _string(record, "optimizer_id"),
            _string(record, "adaptation_id"),
            _string(record, "proposed_after_epoch_manifest_id"),
        )
        _require_id(record, "proposal_record_id", result.proposal_record_id)
        return result


@dataclass(frozen=True, slots=True)
class MatrixElementAdaptationEvidence:
    """Locked held-out comparison; acceptance is derived, never caller asserted."""

    proposal_record_id: str
    held_out_data_id: str
    metric_id: str
    baseline_error: float
    candidate_error: float
    minimum_improvement: float
    evaluated_after_epoch_manifest_id: str
    accepted: bool
    evidence_id: str

    def __init__(
        self,
        proposal_record_id: str,
        held_out_data_id: str,
        metric_id: str,
        baseline_error: float,
        candidate_error: float,
        minimum_improvement: float,
        evaluated_after_epoch_manifest_id: str,
        /,
    ):
        proposal = _digest(proposal_record_id, "proposal_record_id")
        held_out = _identity(held_out_data_id, "held_out_data_id")
        metric = _identity(metric_id, "metric_id")
        baseline = float(baseline_error)
        candidate = float(candidate_error)
        improvement = float(minimum_improvement)
        if not all(math.isfinite(value) for value in (baseline, candidate, improvement)):
            raise ValueError("Adaptation errors and threshold must be finite.")
        if baseline < 0.0 or candidate < 0.0 or improvement < 0.0:
            raise ValueError(
                "Adaptation errors and minimum improvement must be non-negative."
            )
        boundary = _digest(
            evaluated_after_epoch_manifest_id, "evaluated_after_epoch_manifest_id"
        )
        accepted = candidate <= baseline - improvement
        content = {
            "kind": "matrix-element-adaptation-evidence",
            "proposal_record_id": proposal,
            "held_out_data_id": held_out,
            "metric_id": metric,
            "baseline_error": baseline,
            "candidate_error": candidate,
            "minimum_improvement": improvement,
            "evaluated_after_epoch_manifest_id": boundary,
            "accepted": accepted,
        }
        object.__setattr__(self, "proposal_record_id", proposal)
        object.__setattr__(self, "held_out_data_id", held_out)
        object.__setattr__(self, "metric_id", metric)
        object.__setattr__(self, "baseline_error", baseline)
        object.__setattr__(self, "candidate_error", candidate)
        object.__setattr__(self, "minimum_improvement", improvement)
        object.__setattr__(self, "evaluated_after_epoch_manifest_id", boundary)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "evidence_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "matrix-element-adaptation-evidence",
            "proposal_record_id": self.proposal_record_id,
            "held_out_data_id": self.held_out_data_id,
            "metric_id": self.metric_id,
            "baseline_error": self.baseline_error,
            "candidate_error": self.candidate_error,
            "minimum_improvement": self.minimum_improvement,
            "evaluated_after_epoch_manifest_id": self.evaluated_after_epoch_manifest_id,
            "accepted": self.accepted,
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> MatrixElementAdaptationEvidence:
        _require_kind(record, "matrix-element-adaptation-evidence")
        result = cls(
            _string(record, "proposal_record_id"),
            _string(record, "held_out_data_id"),
            _string(record, "metric_id"),
            _number(record, "baseline_error"),
            _number(record, "candidate_error"),
            _number(record, "minimum_improvement"),
            _string(record, "evaluated_after_epoch_manifest_id"),
        )
        if _boolean(record, "accepted") != result.accepted:
            raise ValueError(
                "Serialized adaptation acceptance contradicts held-out metrics."
            )
        _require_id(record, "evidence_id", result.evidence_id)
        return result


@dataclass(frozen=True, slots=True)
class MatrixElementAdaptationDecision:
    """Replayable decision selecting base or one evidence-bound accepted revision."""

    base_revision_id: str
    candidate_draft_revision_id: str
    active_revision: MatrixElementRevision
    proposal_record_id: str
    evidence_id: str
    after_epoch_manifest_id: str
    accepted: bool
    decision_id: str

    def __init__(
        self,
        base_revision_id: str,
        candidate_draft_revision_id: str,
        active_revision: MatrixElementRevision,
        proposal_record_id: str,
        evidence_id: str,
        after_epoch_manifest_id: str,
        accepted: bool,
        /,
    ):
        if not isinstance(active_revision, MatrixElementRevision):
            raise TypeError("active_revision must be MatrixElementRevision.")
        base = _digest(base_revision_id, "base_revision_id")
        candidate = _digest(candidate_draft_revision_id, "candidate_draft_revision_id")
        proposal = _digest(proposal_record_id, "proposal_record_id")
        evidence = _digest(evidence_id, "evidence_id")
        boundary = _digest(after_epoch_manifest_id, "after_epoch_manifest_id")
        accepted_ = bool(accepted)
        expected_active = candidate if accepted_ else base
        if accepted_:
            if (
                active_revision.parent_revision_id != base
                or active_revision.valid_from_epoch_manifest_id != boundary
                or active_revision.held_out_evidence_id != evidence
            ):
                raise ValueError("Accepted revision lacks exact adaptation provenance.")
        elif active_revision.revision_id != expected_active:
            raise ValueError("Rejected adaptation must preserve the base revision.")
        content = {
            "kind": "matrix-element-adaptation-decision",
            "base_revision_id": base,
            "candidate_draft_revision_id": candidate,
            "active_revision_id": active_revision.revision_id,
            "proposal_record_id": proposal,
            "evidence_id": evidence,
            "after_epoch_manifest_id": boundary,
            "accepted": accepted_,
        }
        object.__setattr__(self, "base_revision_id", base)
        object.__setattr__(self, "candidate_draft_revision_id", candidate)
        object.__setattr__(self, "active_revision", active_revision)
        object.__setattr__(self, "proposal_record_id", proposal)
        object.__setattr__(self, "evidence_id", evidence)
        object.__setattr__(self, "after_epoch_manifest_id", boundary)
        object.__setattr__(self, "accepted", accepted_)
        object.__setattr__(self, "decision_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "matrix-element-adaptation-decision",
            "base_revision_id": self.base_revision_id,
            "candidate_draft_revision_id": self.candidate_draft_revision_id,
            "active_revision": self.active_revision.to_record(),
            "proposal_record_id": self.proposal_record_id,
            "evidence_id": self.evidence_id,
            "after_epoch_manifest_id": self.after_epoch_manifest_id,
            "accepted": self.accepted,
            "decision_id": self.decision_id,
        }

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], /
    ) -> MatrixElementAdaptationDecision:
        _require_kind(record, "matrix-element-adaptation-decision")
        raw_revision = record.get("active_revision")
        if not isinstance(raw_revision, Mapping):
            raise TypeError("Serialized adaptation decision requires an active revision.")
        result = cls(
            _string(record, "base_revision_id"),
            _string(record, "candidate_draft_revision_id"),
            MatrixElementRevision.from_record(raw_revision),
            _string(record, "proposal_record_id"),
            _string(record, "evidence_id"),
            _string(record, "after_epoch_manifest_id"),
            _boolean(record, "accepted"),
        )
        _require_id(record, "decision_id", result.decision_id)
        return result


@dataclass(frozen=True, slots=True)
class MatrixElementWeightSnapshot:
    """Immutable event weights bound to the revision that generated them."""

    event_ids: tuple[str, ...]
    weights: tuple[float, ...]
    matrix_element_revision_id: str
    epoch_manifest_id: str
    weight_snapshot_id: str

    def __init__(
        self,
        event_ids: Sequence[str],
        weights: Sequence[float],
        matrix_element_revision_id: str,
        epoch_manifest_id: str,
        /,
    ):
        events = tuple(_digest(value, "event_id") for value in event_ids)
        if len(set(events)) != len(events):
            raise ValueError("Weight snapshot event IDs must be unique.")
        values = tuple(float(value) for value in weights)
        if len(values) != len(events) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("Finite weights must align exactly with event IDs.")
        revision = _digest(matrix_element_revision_id, "matrix_element_revision_id")
        epoch = _digest(epoch_manifest_id, "epoch_manifest_id")
        content = {
            "kind": "matrix-element-weight-snapshot",
            "event_ids": list(events),
            "weights": list(values),
            "matrix_element_revision_id": revision,
            "epoch_manifest_id": epoch,
        }
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "weights", values)
        object.__setattr__(self, "matrix_element_revision_id", revision)
        object.__setattr__(self, "epoch_manifest_id", epoch)
        object.__setattr__(self, "weight_snapshot_id", canonical_fingerprint(content))

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "matrix-element-weight-snapshot",
            "event_ids": list(self.event_ids),
            "weights": list(self.weights),
            "matrix_element_revision_id": self.matrix_element_revision_id,
            "epoch_manifest_id": self.epoch_manifest_id,
            "weight_snapshot_id": self.weight_snapshot_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> MatrixElementWeightSnapshot:
        _require_kind(record, "matrix-element-weight-snapshot")
        result = cls(
            _string_sequence(record, "event_ids"),
            _number_sequence(record, "weights"),
            _string(record, "matrix_element_revision_id"),
            _string(record, "epoch_manifest_id"),
        )
        _require_id(record, "weight_snapshot_id", result.weight_snapshot_id)
        return result


def resolve_matrix_element_adaptation(
    base: MatrixElementRevision,
    candidate_draft: MatrixElementRevision,
    proposal: MatrixElementAdaptationProposal,
    evidence: MatrixElementAdaptationEvidence,
    /,
    *,
    after_epoch_manifest_id: str,
) -> MatrixElementAdaptationDecision:
    """Resolve adaptation without mutating the base or any prior event weights."""

    if not isinstance(base, MatrixElementRevision) or not isinstance(
        candidate_draft, MatrixElementRevision
    ):
        raise TypeError("base and candidate_draft must be MatrixElementRevision.")
    if not isinstance(proposal, MatrixElementAdaptationProposal):
        raise TypeError("proposal must be MatrixElementAdaptationProposal.")
    if not isinstance(evidence, MatrixElementAdaptationEvidence):
        raise TypeError("evidence must be MatrixElementAdaptationEvidence.")
    boundary = _digest(after_epoch_manifest_id, "after_epoch_manifest_id")
    if base.is_adapted and base.valid_from_epoch_manifest_id == boundary:
        raise ValueError("At most one adaptation may activate at one committed boundary.")
    if candidate_draft.is_adapted:
        raise ValueError("candidate_draft must not already carry activation provenance.")
    if proposal.base_revision_id != base.revision_id:
        raise ValueError("Adaptation proposal does not name the base revision.")
    if proposal.candidate_revision_id != candidate_draft.revision_id:
        raise ValueError("Adaptation proposal does not name the candidate draft.")
    if proposal.proposed_after_epoch_manifest_id != boundary:
        raise ValueError("Adaptation proposal is bound to a different epoch boundary.")
    if evidence.proposal_record_id != proposal.proposal_record_id:
        raise ValueError("Held-out evidence does not name the adaptation proposal.")
    if evidence.evaluated_after_epoch_manifest_id != boundary:
        raise ValueError("Held-out evidence was evaluated at a different boundary.")
    if evidence.held_out_data_id == proposal.training_data_id:
        raise ValueError("Held-out adaptation data must be disjoint from training data.")
    if (
        candidate_draft.training_data_id != proposal.training_data_id
        or candidate_draft.optimizer_id != proposal.optimizer_id
        or candidate_draft.adaptation_id != proposal.adaptation_id
    ):
        raise ValueError("Candidate draft does not bind proposal training provenance.")
    if evidence.accepted:
        active = MatrixElementRevision(
            candidate_draft.process_id,
            candidate_draft.model_id,
            candidate_draft.provider_id,
            candidate_draft.rights_id,
            candidate_draft.normalization_id,
            candidate_draft.support_id,
            candidate_draft.proposal_id,
            candidate_draft.adaptation_id,
            candidate_draft.training_data_id,
            candidate_draft.optimizer_id,
            candidate_draft.error_model_id,
            candidate_draft.parameter_digest,
            differentiation_id=candidate_draft.differentiation_id,
            parent_revision_id=base.revision_id,
            valid_from_epoch_manifest_id=boundary,
            held_out_evidence_id=evidence.evidence_id,
        )
    else:
        active = base
    return MatrixElementAdaptationDecision(
        base.revision_id,
        candidate_draft.revision_id,
        active,
        proposal.proposal_record_id,
        evidence.evidence_id,
        boundary,
        evidence.accepted,
    )


def _require_kind(record: Mapping[str, object], expected: str, /) -> None:
    if record.get("kind") != expected:
        raise ValueError(f"Expected serialized record kind {expected!r}.")


def _require_id(record: Mapping[str, object], field: str, expected: str, /) -> None:
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


def _number(record: Mapping[str, object], field: str, /) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Serialized {field} must be numeric.")
    return float(value)


def _boolean(record: Mapping[str, object], field: str, /) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise TypeError(f"Serialized {field} must be boolean.")
    return value


def _string_sequence(record: Mapping[str, object], field: str, /) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Serialized {field} must be a string sequence.")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"Serialized {field} must contain only strings.")
    return tuple(value)


def _number_sequence(record: Mapping[str, object], field: str, /) -> tuple[float, ...]:
    value = record.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Serialized {field} must be a numeric sequence.")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise TypeError(f"Serialized {field} must contain only numbers.")
    return tuple(float(item) for item in value)


__all__ = [
    "MatrixElementAdaptationDecision",
    "MatrixElementAdaptationEvidence",
    "MatrixElementAdaptationProposal",
    "MatrixElementRevision",
    "MatrixElementWeightSnapshot",
    "resolve_matrix_element_adaptation",
]
