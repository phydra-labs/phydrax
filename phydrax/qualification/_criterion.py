#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._evidence import QualificationEvidence


_MAX_TIMESTAMP = 2**63 - 1


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _timestamp(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer timestamp.")
    if value < 0 or value > _MAX_TIMESTAMP:
        raise ValueError(f"{name} must be a non-negative signed 64-bit timestamp.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(normalized))


class QualificationCriterion(StrictModule, NonTrainableState):
    """One approved quantitative criterion for one exact support tuple."""

    support_tuple_id: str = eqx.field(static=True)
    metric: str = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    comparison: str = eqx.field(static=True)
    target: float = eqx.field(static=True)
    aggregation: str = eqx.field(static=True)
    uncertainty: str = eqx.field(static=True)
    applicability: str = eqx.field(static=True)
    approval_id: str = eqx.field(static=True)
    issued_at: int = eqx.field(static=True)
    valid_until: int | None = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        support_tuple_id: str,
        metric: str,
        unit: str,
        comparison: str,
        target: int | float,
        aggregation: str,
        uncertainty: str,
        applicability: str,
        approval_id: str,
        issued_at: int,
        valid_until: int | None = None,
    ):
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise TypeError("target must be a real number.")
        target_ = float(target)
        if not math.isfinite(target_):
            raise ValueError("target must be finite.")
        if target_ == 0.0:
            target_ = 0.0
        issued = _timestamp(issued_at, "issued_at")
        if valid_until is None:
            deadline = None
        else:
            deadline = _timestamp(valid_until, "valid_until")
            if deadline <= issued:
                raise ValueError("A criterion validity deadline must follow issuance.")
        comparison_ = _identifier(comparison, "criterion comparison")
        if comparison_ not in (
            "equal",
            "less-than-or-equal",
            "greater-than-or-equal",
        ):
            raise ValueError("Unsupported quantitative qualification comparison.")
        self.support_tuple_id = _identifier(
            support_tuple_id, "criterion support-tuple ID"
        )
        self.metric = _identifier(metric, "criterion metric")
        self.unit = _identifier(unit, "criterion unit")
        self.comparison = comparison_
        self.target = target_
        self.aggregation = _identifier(aggregation, "criterion aggregation")
        self.uncertainty = _identifier(uncertainty, "criterion uncertainty")
        self.applicability = _identifier(applicability, "criterion applicability")
        self.approval_id = _identifier(approval_id, "criterion approval ID")
        self.issued_at = issued
        self.valid_until = deadline
        self.criterion_id = canonical_fingerprint(self._content_record())

    def is_valid(self, at_time: int, /) -> bool:
        """Return whether the issued criterion is valid at ``at_time``."""
        timestamp = _timestamp(at_time, "at_time")
        return self.issued_at <= timestamp and (
            self.valid_until is None or timestamp <= self.valid_until
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "qualification-criterion",
            "support_tuple_id": self.support_tuple_id,
            "metric": self.metric,
            "unit": self.unit,
            "comparison": self.comparison,
            "target": self.target,
            "aggregation": self.aggregation,
            "uncertainty": self.uncertainty,
            "applicability": self.applicability,
            "approval_id": self.approval_id,
            "issued_at": self.issued_at,
            "valid_until": self.valid_until,
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready criterion record."""
        return {**self._content_record(), "criterion_id": self.criterion_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> QualificationCriterion:
        """Reconstruct and content-verify a serialized criterion."""
        if not isinstance(record, Mapping):
            raise TypeError("Qualification-criterion record must be a mapping.")
        if record.get("kind") != "qualification-criterion":
            raise ValueError("Serialized qualification criterion has an invalid kind.")
        value = cls(
            support_tuple_id=record["support_tuple_id"],
            metric=record["metric"],
            unit=record["unit"],
            comparison=record["comparison"],
            target=record["target"],
            aggregation=record["aggregation"],
            uncertainty=record["uncertainty"],
            applicability=record["applicability"],
            approval_id=record["approval_id"],
            issued_at=record["issued_at"],
            valid_until=record["valid_until"],
        )
        recorded_id = record.get("criterion_id")
        if recorded_id is not None and recorded_id != value.criterion_id:
            raise ValueError(
                "Serialized qualification criterion has an invalid content address."
            )
        return value


class CampaignStartRecord(StrictModule, NonTrainableState):
    """Immutable start boundary for one criterion and resolved campaign run."""

    campaign_spec_id: str = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)
    resolved_run_spec_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    started_at: int = eqx.field(static=True)
    start_record_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        campaign_spec_id: str,
        criterion_id: str,
        resolved_run_spec_id: str,
        support_tuple_id: str,
        started_at: int,
    ):
        self.campaign_spec_id = _identifier(campaign_spec_id, "campaign specification ID")
        self.criterion_id = _identifier(criterion_id, "campaign criterion ID")
        self.resolved_run_spec_id = _identifier(
            resolved_run_spec_id, "resolved run-specification ID"
        )
        self.support_tuple_id = _identifier(support_tuple_id, "campaign support-tuple ID")
        self.started_at = _timestamp(started_at, "started_at")
        self.start_record_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "campaign-start-record",
            "campaign_spec_id": self.campaign_spec_id,
            "criterion_id": self.criterion_id,
            "resolved_run_spec_id": self.resolved_run_spec_id,
            "support_tuple_id": self.support_tuple_id,
            "started_at": self.started_at,
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready campaign-start record."""
        return {**self._content_record(), "start_record_id": self.start_record_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CampaignStartRecord:
        """Reconstruct and content-verify a serialized campaign start."""
        if not isinstance(record, Mapping):
            raise TypeError("Campaign-start record must be a mapping.")
        if record.get("kind") != "campaign-start-record":
            raise ValueError("Serialized campaign start has an invalid kind.")
        value = cls(
            campaign_spec_id=record["campaign_spec_id"],
            criterion_id=record["criterion_id"],
            resolved_run_spec_id=record["resolved_run_spec_id"],
            support_tuple_id=record["support_tuple_id"],
            started_at=record["started_at"],
        )
        recorded_id = record.get("start_record_id")
        if recorded_id is not None and recorded_id != value.start_record_id:
            raise ValueError("Serialized campaign start has an invalid content address.")
        return value


class CampaignObservationRecord(StrictModule, NonTrainableState):
    """Immutable observation boundary for one exact campaign start."""

    start_record_id: str = eqx.field(static=True)
    campaign_spec_id: str = eqx.field(static=True)
    criterion_id: str = eqx.field(static=True)
    resolved_run_spec_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    raw_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    observed_at: int = eqx.field(static=True)
    observation_record_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        start_record_id: str,
        campaign_spec_id: str,
        criterion_id: str,
        resolved_run_spec_id: str,
        support_tuple_id: str,
        raw_artifact_ids: Sequence[str],
        observed_at: int,
    ):
        self.start_record_id = _identifier(start_record_id, "campaign-start record ID")
        self.campaign_spec_id = _identifier(campaign_spec_id, "campaign specification ID")
        self.criterion_id = _identifier(criterion_id, "campaign criterion ID")
        self.resolved_run_spec_id = _identifier(
            resolved_run_spec_id, "resolved run-specification ID"
        )
        self.support_tuple_id = _identifier(support_tuple_id, "campaign support-tuple ID")
        self.raw_artifact_ids = _identifiers(
            raw_artifact_ids, "campaign raw-artifact IDs"
        )
        self.observed_at = _timestamp(observed_at, "observed_at")
        self.observation_record_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "campaign-observation-record",
            "start_record_id": self.start_record_id,
            "campaign_spec_id": self.campaign_spec_id,
            "criterion_id": self.criterion_id,
            "resolved_run_spec_id": self.resolved_run_spec_id,
            "support_tuple_id": self.support_tuple_id,
            "raw_artifact_ids": list(self.raw_artifact_ids),
            "observed_at": self.observed_at,
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready campaign-observation record."""
        return {
            **self._content_record(),
            "observation_record_id": self.observation_record_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CampaignObservationRecord:
        """Reconstruct and content-verify a serialized campaign observation."""
        if not isinstance(record, Mapping):
            raise TypeError("Campaign-observation record must be a mapping.")
        if record.get("kind") != "campaign-observation-record":
            raise ValueError("Serialized campaign observation has an invalid kind.")
        raw_artifact_ids = record["raw_artifact_ids"]
        if not isinstance(raw_artifact_ids, Sequence) or isinstance(
            raw_artifact_ids, str
        ):
            raise TypeError("Serialized campaign raw-artifact IDs must be a sequence.")
        value = cls(
            start_record_id=record["start_record_id"],
            campaign_spec_id=record["campaign_spec_id"],
            criterion_id=record["criterion_id"],
            resolved_run_spec_id=record["resolved_run_spec_id"],
            support_tuple_id=record["support_tuple_id"],
            raw_artifact_ids=raw_artifact_ids,
            observed_at=record["observed_at"],
        )
        recorded_id = record.get("observation_record_id")
        if recorded_id is not None and recorded_id != value.observation_record_id:
            raise ValueError(
                "Serialized campaign observation has an invalid content address."
            )
        return value


def validate_qualification_causality(
    criterion: QualificationCriterion,
    start: CampaignStartRecord,
    observation: CampaignObservationRecord,
    evidence: QualificationEvidence,
    /,
) -> str:
    """Return the evidence ID after exact linkage and causality validation."""
    if not isinstance(criterion, QualificationCriterion):
        raise TypeError("criterion must be a QualificationCriterion.")
    if not isinstance(start, CampaignStartRecord):
        raise TypeError("start must be a CampaignStartRecord.")
    if not isinstance(observation, CampaignObservationRecord):
        raise TypeError("observation must be a CampaignObservationRecord.")
    if not isinstance(evidence, QualificationEvidence):
        raise TypeError("evidence must be QualificationEvidence.")
    QualificationCriterion.from_record(criterion.to_record())
    CampaignStartRecord.from_record(start.to_record())
    CampaignObservationRecord.from_record(observation.to_record())
    QualificationEvidence.from_record(evidence.to_record())

    if start.criterion_id != criterion.criterion_id:
        raise ValueError("Campaign start does not bind the exact criterion.")
    if start.support_tuple_id != criterion.support_tuple_id:
        raise ValueError("Campaign start does not bind the criterion support tuple.")
    if observation.start_record_id != start.start_record_id:
        raise ValueError("Campaign observation was borrowed from another start record.")
    if observation.campaign_spec_id != start.campaign_spec_id:
        raise ValueError("Campaign observation has a mismatched campaign_spec_id.")
    if observation.criterion_id != start.criterion_id:
        raise ValueError("Campaign observation has a mismatched criterion_id.")
    if observation.resolved_run_spec_id != start.resolved_run_spec_id:
        raise ValueError("Campaign observation has a mismatched resolved_run_spec_id.")
    if observation.support_tuple_id != start.support_tuple_id:
        raise ValueError("Campaign observation has a mismatched support_tuple_id.")
    if not criterion.issued_at < start.started_at:
        raise ValueError("Qualification criterion must be issued before campaign start.")
    if not criterion.is_valid(start.started_at):
        raise ValueError("Qualification criterion expired before campaign start.")
    if not start.started_at <= observation.observed_at <= evidence.issued_at:
        raise ValueError(
            "Qualification causality requires started_at <= observed_at "
            "<= evidence issued_at."
        )
    if criterion.criterion_id not in evidence.criteria_ids:
        raise ValueError("Qualification evidence does not bind the exact criterion.")
    if start.start_record_id not in evidence.campaign_start_record_ids:
        raise ValueError("Qualification evidence does not bind the campaign start.")
    if observation.observation_record_id not in evidence.campaign_observation_record_ids:
        raise ValueError("Qualification evidence does not bind the campaign observation.")
    if evidence.raw_artifact_ids != observation.raw_artifact_ids:
        raise ValueError(
            "Qualification evidence does not bind the exact campaign raw artifacts."
        )
    return evidence.evidence_id


__all__ = [
    "CampaignObservationRecord",
    "CampaignStartRecord",
    "QualificationCriterion",
    "validate_qualification_causality",
]
