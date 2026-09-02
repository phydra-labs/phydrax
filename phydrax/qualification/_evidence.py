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


_MAX_TIMESTAMP = 2**63 - 1
_EVIDENCE_KINDS = frozenset(
    (
        "unit",
        "smoke",
        "performance",
        "scientific",
        "reference",
        "operational",
        "security",
    )
)
_EVIDENCE_OUTCOMES = frozenset(("passed", "failed", "inconclusive"))
_PREDICATE_FIELDS = frozenset(
    (
        "evidence_id",
        "evidence_kind",
        "subject_id",
        "build_id",
        "environment_id",
        "backend",
        "topology",
        "precision",
        "reduction",
        "replay_id",
        "criterion_id",
        "raw_artifact_id",
        "reviewer_id",
        "requalification_trigger",
        "observed_resource_record_id",
        "forecast_resource_record_id",
    )
)


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


def _identifiers(
    values: Sequence[str],
    name: str,
    /,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(normalized))


def _measurements(
    values: Mapping[str, int | float], name: str, /
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError(f"{name} must be a non-empty measurement mapping.")
    normalized: list[tuple[str, float]] = []
    for metric, value in values.items():
        metric_ = _identifier(metric, f"{name} metric")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} values must be real numbers.")
        value_ = float(value)
        if not math.isfinite(value_) or value_ < 0.0:
            raise ValueError(f"{name} values must be finite and non-negative.")
        value_ = abs(value_)
        normalized.append((metric_, value_))
    normalized.sort()
    if len({metric for metric, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} metric names must be unique.")
    return tuple(normalized)


class SupportDependency(StrictModule, NonTrainableState):
    """One dependency on one exact tuple of another capability profile."""

    profile_id: str = eqx.field(static=True)
    support_tuple_id: str = eqx.field(static=True)
    dependency_id: str = eqx.field(static=True)

    def __init__(self, profile_id: str, support_tuple_id: str, /):
        self.profile_id = _identifier(profile_id, "dependency profile ID")
        self.support_tuple_id = _identifier(
            support_tuple_id, "dependency support-tuple ID"
        )
        self.dependency_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "exact-support-dependency",
            "profile_id": self.profile_id,
            "support_tuple_id": self.support_tuple_id,
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready exact-dependency record."""
        return {**self._content_record(), "dependency_id": self.dependency_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> SupportDependency:
        """Reconstruct and content-verify an exact support dependency."""
        if not isinstance(record, Mapping):
            raise TypeError("Support-dependency record must be a mapping.")
        value = cls(record["profile_id"], record["support_tuple_id"])
        recorded_id = record.get("dependency_id")
        if recorded_id is not None and str(recorded_id) != value.dependency_id:
            raise ValueError(
                "Serialized support dependency has an invalid content address."
            )
        return value


class ObservedResourceRecord(StrictModule, NonTrainableState):
    """Measured resource use for one exact build and execution environment."""

    subject_id: str = eqx.field(static=True)
    build_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    topology: str = eqx.field(static=True)
    measurements: tuple[tuple[str, float], ...] = eqx.field(static=True)
    observed_at: int = eqx.field(static=True)
    raw_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        subject_id: str,
        build_id: str,
        environment_id: str,
        /,
        *,
        backend: str,
        topology: str,
        measurements: Mapping[str, int | float],
        observed_at: int,
        raw_artifact_ids: Sequence[str],
    ):
        self.subject_id = _identifier(subject_id, "resource subject ID")
        self.build_id = _identifier(build_id, "resource build ID")
        self.environment_id = _identifier(environment_id, "resource environment ID")
        self.backend = _identifier(backend, "resource backend")
        self.topology = _identifier(topology, "resource topology")
        self.measurements = _measurements(measurements, "resource measurements")
        self.observed_at = _timestamp(observed_at, "observed_at")
        self.raw_artifact_ids = _identifiers(
            raw_artifact_ids, "resource raw-artifact IDs"
        )
        self.record_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "observed-resource-record",
            "subject_id": self.subject_id,
            "build_id": self.build_id,
            "environment_id": self.environment_id,
            "backend": self.backend,
            "topology": self.topology,
            "measurements": dict(self.measurements),
            "observed_at": self.observed_at,
            "raw_artifact_ids": list(self.raw_artifact_ids),
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready observed-resource record."""
        return {**self._content_record(), "record_id": self.record_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ObservedResourceRecord:
        """Reconstruct and content-verify observed resource use."""
        if not isinstance(record, Mapping):
            raise TypeError("Observed-resource record must be a mapping.")
        measurements = record["measurements"]
        raw_artifact_ids = record["raw_artifact_ids"]
        if not isinstance(measurements, Mapping):
            raise TypeError("Serialized resource measurements must be a mapping.")
        if not isinstance(raw_artifact_ids, Sequence) or isinstance(
            raw_artifact_ids, str
        ):
            raise TypeError("Serialized resource artifact IDs must be a sequence.")
        value = cls(
            str(record["subject_id"]),
            str(record["build_id"]),
            str(record["environment_id"]),
            backend=str(record["backend"]),
            topology=str(record["topology"]),
            measurements=measurements,
            observed_at=int(record["observed_at"]),
            raw_artifact_ids=tuple(str(item) for item in raw_artifact_ids),
        )
        recorded_id = record.get("record_id")
        if recorded_id is not None and str(recorded_id) != value.record_id:
            raise ValueError(
                "Serialized observed-resource record has an invalid content address."
            )
        return value


class ForecastResourceRecord(StrictModule, NonTrainableState):
    """Bounded resource forecast derived from named observations and a model."""

    subject_id: str = eqx.field(static=True)
    build_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    topology: str = eqx.field(static=True)
    estimates: tuple[tuple[str, float], ...] = eqx.field(static=True)
    uncertainty_bounds: tuple[tuple[str, tuple[float, float]], ...] = eqx.field(
        static=True
    )
    forecast_model_id: str = eqx.field(static=True)
    source_record_ids: tuple[str, ...] = eqx.field(static=True)
    issued_at: int = eqx.field(static=True)
    expires_at: int = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        subject_id: str,
        build_id: str,
        environment_id: str,
        /,
        *,
        backend: str,
        topology: str,
        estimates: Mapping[str, int | float],
        uncertainty_bounds: Mapping[str, Sequence[int | float]],
        forecast_model_id: str,
        source_record_ids: Sequence[str],
        issued_at: int,
        expires_at: int,
    ):
        estimates_ = _measurements(estimates, "resource estimates")
        if not isinstance(uncertainty_bounds, Mapping):
            raise TypeError("uncertainty_bounds must be a mapping.")
        if set(uncertainty_bounds) != {name for name, _ in estimates_}:
            raise ValueError(
                "Forecast uncertainty bounds must cover every estimate exactly."
            )
        bounds: list[tuple[str, tuple[float, float]]] = []
        estimates_by_name = dict(estimates_)
        for metric, values in uncertainty_bounds.items():
            metric_ = _identifier(metric, "forecast uncertainty metric")
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TypeError("Each forecast uncertainty bound must be a pair.")
            pair = tuple(values)
            if len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in pair
            ):
                raise TypeError("Each forecast uncertainty bound must be a real pair.")
            lower, upper = float(pair[0]), float(pair[1])
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower < 0.0
                or lower > estimates_by_name[metric_]
                or estimates_by_name[metric_] > upper
            ):
                raise ValueError(
                    "Forecast bounds must be finite, non-negative, and contain the estimate."
                )
            lower, upper = abs(lower), abs(upper)
            bounds.append((metric_, (lower, upper)))
        bounds.sort()
        issued = _timestamp(issued_at, "issued_at")
        expires = _timestamp(expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("A resource forecast must expire after it is issued.")
        self.subject_id = _identifier(subject_id, "forecast subject ID")
        self.build_id = _identifier(build_id, "forecast build ID")
        self.environment_id = _identifier(environment_id, "forecast environment ID")
        self.backend = _identifier(backend, "forecast backend")
        self.topology = _identifier(topology, "forecast topology")
        self.estimates = estimates_
        self.uncertainty_bounds = tuple(bounds)
        self.forecast_model_id = _identifier(forecast_model_id, "forecast model ID")
        self.source_record_ids = _identifiers(
            source_record_ids, "forecast source-record IDs"
        )
        self.issued_at = issued
        self.expires_at = expires
        self.record_id = canonical_fingerprint(self._content_record())

    def is_current(self, at_time: int, /) -> bool:
        timestamp = _timestamp(at_time, "at_time")
        return self.issued_at <= timestamp <= self.expires_at

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "forecast-resource-record",
            "subject_id": self.subject_id,
            "build_id": self.build_id,
            "environment_id": self.environment_id,
            "backend": self.backend,
            "topology": self.topology,
            "estimates": dict(self.estimates),
            "uncertainty_bounds": {
                name: list(bounds) for name, bounds in self.uncertainty_bounds
            },
            "forecast_model_id": self.forecast_model_id,
            "source_record_ids": list(self.source_record_ids),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready forecast-resource record."""
        return {**self._content_record(), "record_id": self.record_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ForecastResourceRecord:
        """Reconstruct and content-verify a resource forecast."""
        if not isinstance(record, Mapping):
            raise TypeError("Forecast-resource record must be a mapping.")
        estimates = record["estimates"]
        uncertainty_bounds = record["uncertainty_bounds"]
        source_record_ids = record["source_record_ids"]
        if not isinstance(estimates, Mapping) or not isinstance(
            uncertainty_bounds, Mapping
        ):
            raise TypeError("Serialized forecast estimates and bounds must be mappings.")
        if not isinstance(source_record_ids, Sequence) or isinstance(
            source_record_ids, str
        ):
            raise TypeError("Serialized forecast source IDs must be a sequence.")
        value = cls(
            str(record["subject_id"]),
            str(record["build_id"]),
            str(record["environment_id"]),
            backend=str(record["backend"]),
            topology=str(record["topology"]),
            estimates=estimates,
            uncertainty_bounds=uncertainty_bounds,
            forecast_model_id=str(record["forecast_model_id"]),
            source_record_ids=tuple(str(item) for item in source_record_ids),
            issued_at=int(record["issued_at"]),
            expires_at=int(record["expires_at"]),
        )
        recorded_id = record.get("record_id")
        if recorded_id is not None and str(recorded_id) != value.record_id:
            raise ValueError(
                "Serialized forecast-resource record has an invalid content address."
            )
        return value


class QualificationEvidence(StrictModule, NonTrainableState):
    """One reviewed outcome bound to an exact execution and evidence scope."""

    evidence_kind: str = eqx.field(static=True)
    outcome: str = eqx.field(static=True)
    subject_ids: tuple[str, ...] = eqx.field(static=True)
    build_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    topology: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    reduction: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    criteria_ids: tuple[str, ...] = eqx.field(static=True)
    raw_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    observed_resource_record_ids: tuple[str, ...] = eqx.field(static=True)
    forecast_resource_record_ids: tuple[str, ...] = eqx.field(static=True)
    reviewer_id: str = eqx.field(static=True)
    issued_at: int = eqx.field(static=True)
    expires_at: int = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    supersedes_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    requalification_triggers: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        evidence_kind: str,
        outcome: str,
        subject_ids: Sequence[str],
        /,
        *,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        criteria_ids: Sequence[str],
        raw_artifact_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
        reason: str,
        supersedes_evidence_ids: Sequence[str] = (),
        requalification_triggers: Sequence[str] = (),
        observed_resource_record_ids: Sequence[str] = (),
        forecast_resource_record_ids: Sequence[str] = (),
    ):
        kind = _identifier(evidence_kind, "evidence kind")
        if kind not in _EVIDENCE_KINDS:
            raise ValueError(
                "Evidence kind must be one of unit, smoke, performance, scientific, "
                "reference, operational, or security."
            )
        outcome_ = _identifier(outcome, "evidence outcome")
        if outcome_ not in _EVIDENCE_OUTCOMES:
            raise ValueError("Evidence outcome must be passed, failed, or inconclusive.")
        issued = _timestamp(issued_at, "issued_at")
        expires = _timestamp(expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("Qualification evidence must expire after it is issued.")
        self.evidence_kind = kind
        self.outcome = outcome_
        self.subject_ids = _identifiers(subject_ids, "evidence subject IDs")
        self.build_id = _identifier(build_id, "evidence build ID")
        self.environment_id = _identifier(environment_id, "evidence environment ID")
        self.backend = _identifier(backend, "evidence backend")
        self.topology = _identifier(topology, "evidence topology")
        self.precision = _identifier(precision, "evidence precision")
        self.reduction = _identifier(reduction, "evidence reduction")
        self.replay_id = _identifier(replay_id, "evidence replay ID")
        self.criteria_ids = _identifiers(criteria_ids, "evidence criteria IDs")
        self.raw_artifact_ids = _identifiers(
            raw_artifact_ids, "evidence raw-artifact IDs"
        )
        self.observed_resource_record_ids = _identifiers(
            observed_resource_record_ids,
            "observed resource-record IDs",
            allow_empty=True,
        )
        self.forecast_resource_record_ids = _identifiers(
            forecast_resource_record_ids,
            "forecast resource-record IDs",
            allow_empty=True,
        )
        self.reviewer_id = _identifier(reviewer_id, "evidence reviewer ID")
        self.issued_at = issued
        self.expires_at = expires
        self.reason = _identifier(reason, "evidence outcome reason")
        self.supersedes_evidence_ids = _identifiers(
            supersedes_evidence_ids,
            "superseded evidence IDs",
            allow_empty=True,
        )
        self.requalification_triggers = _identifiers(
            requalification_triggers,
            "evidence requalification triggers",
            allow_empty=True,
        )
        self.evidence_id = canonical_fingerprint(self._content_record())
        if self.evidence_id in self.supersedes_evidence_ids:
            raise ValueError("Qualification evidence cannot supersede itself.")

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def failed(self) -> bool:
        return self.outcome == "failed"

    @property
    def inconclusive(self) -> bool:
        return self.outcome == "inconclusive"

    def is_current(self, at_time: int, /) -> bool:
        timestamp = _timestamp(at_time, "at_time")
        return self.issued_at <= timestamp <= self.expires_at

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "qualification-evidence",
            "evidence_kind": self.evidence_kind,
            "outcome": self.outcome,
            "subject_ids": list(self.subject_ids),
            "build_id": self.build_id,
            "environment_id": self.environment_id,
            "backend": self.backend,
            "topology": self.topology,
            "precision": self.precision,
            "reduction": self.reduction,
            "replay_id": self.replay_id,
            "criteria_ids": list(self.criteria_ids),
            "raw_artifact_ids": list(self.raw_artifact_ids),
            "observed_resource_record_ids": list(self.observed_resource_record_ids),
            "forecast_resource_record_ids": list(self.forecast_resource_record_ids),
            "reviewer_id": self.reviewer_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "reason": self.reason,
            "supersedes_evidence_ids": list(self.supersedes_evidence_ids),
            "requalification_triggers": list(self.requalification_triggers),
        }

    def to_record(self) -> dict[str, object]:
        """Return a complete deterministic JSON-ready evidence record."""
        return {**self._content_record(), "evidence_id": self.evidence_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> QualificationEvidence:
        """Reconstruct and content-verify serialized qualification evidence."""
        if not isinstance(record, Mapping):
            raise TypeError("Qualification-evidence record must be a mapping.")
        sequence_fields = (
            "subject_ids",
            "criteria_ids",
            "raw_artifact_ids",
            "observed_resource_record_ids",
            "forecast_resource_record_ids",
            "supersedes_evidence_ids",
            "requalification_triggers",
        )
        for name in sequence_fields:
            values = record[name]
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TypeError(f"Serialized {name} must be a sequence.")
        value = cls(
            str(record["evidence_kind"]),
            str(record["outcome"]),
            tuple(str(item) for item in record["subject_ids"]),
            build_id=str(record["build_id"]),
            environment_id=str(record["environment_id"]),
            backend=str(record["backend"]),
            topology=str(record["topology"]),
            precision=str(record["precision"]),
            reduction=str(record["reduction"]),
            replay_id=str(record["replay_id"]),
            criteria_ids=tuple(str(item) for item in record["criteria_ids"]),
            raw_artifact_ids=tuple(str(item) for item in record["raw_artifact_ids"]),
            observed_resource_record_ids=tuple(
                str(item) for item in record["observed_resource_record_ids"]
            ),
            forecast_resource_record_ids=tuple(
                str(item) for item in record["forecast_resource_record_ids"]
            ),
            reviewer_id=str(record["reviewer_id"]),
            issued_at=int(record["issued_at"]),
            expires_at=int(record["expires_at"]),
            reason=str(record["reason"]),
            supersedes_evidence_ids=tuple(
                str(item) for item in record["supersedes_evidence_ids"]
            ),
            requalification_triggers=tuple(
                str(item) for item in record["requalification_triggers"]
            ),
        )
        recorded_id = record.get("evidence_id")
        if recorded_id is not None and str(recorded_id) != value.evidence_id:
            raise ValueError(
                "Serialized qualification evidence has an invalid content address."
            )
        return value


def _predicate_matches(
    evidence: QualificationEvidence,
    predicate: tuple[tuple[str, str], ...],
    /,
) -> bool:
    for name, value in predicate:
        if name == "evidence_id" and evidence.evidence_id != value:
            return False
        if name == "evidence_kind" and evidence.evidence_kind != value:
            return False
        if name == "subject_id" and value not in evidence.subject_ids:
            return False
        if name == "build_id" and evidence.build_id != value:
            return False
        if name == "environment_id" and evidence.environment_id != value:
            return False
        if name == "backend" and evidence.backend != value:
            return False
        if name == "topology" and evidence.topology != value:
            return False
        if name == "precision" and evidence.precision != value:
            return False
        if name == "reduction" and evidence.reduction != value:
            return False
        if name == "replay_id" and evidence.replay_id != value:
            return False
        if name == "criterion_id" and value not in evidence.criteria_ids:
            return False
        if name == "raw_artifact_id" and value not in evidence.raw_artifact_ids:
            return False
        if name == "reviewer_id" and evidence.reviewer_id != value:
            return False
        if (
            name == "requalification_trigger"
            and value not in evidence.requalification_triggers
        ):
            return False
        if (
            name == "observed_resource_record_id"
            and value not in evidence.observed_resource_record_ids
        ):
            return False
        if (
            name == "forecast_resource_record_id"
            and value not in evidence.forecast_resource_record_ids
        ):
            return False
    return True


class QualificationCoverageReport(StrictModule, NonTrainableState):
    """Deterministic pass, failure, and gap accounting for one matrix evaluation."""

    matrix_id: str = eqx.field(static=True)
    evaluated_at: int = eqx.field(static=True)
    outcome: str = eqx.field(static=True)
    passed_predicate_ids: tuple[str, ...] = eqx.field(static=True)
    failed_predicate_ids: tuple[str, ...] = eqx.field(static=True)
    inconclusive_predicate_ids: tuple[str, ...] = eqx.field(static=True)
    gaps: tuple[tuple[str, str, tuple[str, ...]], ...] = eqx.field(static=True)
    matched_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix_id: str,
        /,
        *,
        evaluated_at: int,
        passed_predicate_ids: Sequence[str],
        failed_predicate_ids: Sequence[str],
        inconclusive_predicate_ids: Sequence[str],
        gaps: Sequence[tuple[str, str, Sequence[str]]],
        matched_evidence_ids: Sequence[str],
    ):
        passed = _identifiers(
            passed_predicate_ids, "passed predicate IDs", allow_empty=True
        )
        failed = _identifiers(
            failed_predicate_ids, "failed predicate IDs", allow_empty=True
        )
        inconclusive = _identifiers(
            inconclusive_predicate_ids,
            "inconclusive predicate IDs",
            allow_empty=True,
        )
        all_ids = passed + failed + inconclusive
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("A matrix predicate must have exactly one reported outcome.")
        normalized_gaps: list[tuple[str, str, tuple[str, ...]]] = []
        for predicate_id, outcome, reasons in gaps:
            predicate_id_ = _identifier(predicate_id, "gap predicate ID")
            outcome_ = _identifier(outcome, "gap outcome")
            if outcome_ not in ("failed", "inconclusive"):
                raise ValueError("A matrix gap must be failed or inconclusive.")
            reasons_ = _identifiers(reasons, "matrix gap reasons")
            normalized_gaps.append((predicate_id_, outcome_, reasons_))
        normalized_gaps.sort()
        if tuple(item[0] for item in normalized_gaps) != tuple(
            sorted(failed + inconclusive)
        ):
            raise ValueError("Coverage gaps must describe every non-passing predicate.")
        if any(
            (predicate_id in failed) != (outcome == "failed")
            for predicate_id, outcome, _ in normalized_gaps
        ):
            raise ValueError("Coverage gap outcomes must match predicate outcomes.")
        outcome_ = "failed" if failed else "inconclusive" if inconclusive else "passed"
        self.matrix_id = _identifier(matrix_id, "qualification matrix ID")
        self.evaluated_at = _timestamp(evaluated_at, "evaluated_at")
        self.outcome = outcome_
        self.passed_predicate_ids = passed
        self.failed_predicate_ids = failed
        self.inconclusive_predicate_ids = inconclusive
        self.gaps = tuple(normalized_gaps)
        self.matched_evidence_ids = _identifiers(
            matched_evidence_ids, "matched evidence IDs", allow_empty=True
        )
        self.report_id = canonical_fingerprint(self._content_record())

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def complete(self) -> bool:
        return not self.inconclusive_predicate_ids

    @property
    def coverage_fraction(self) -> float:
        count = (
            len(self.passed_predicate_ids)
            + len(self.failed_predicate_ids)
            + len(self.inconclusive_predicate_ids)
        )
        if count == 0:
            return 1.0
        return (len(self.passed_predicate_ids) + len(self.failed_predicate_ids)) / count

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "qualification-coverage-report",
            "matrix_id": self.matrix_id,
            "evaluated_at": self.evaluated_at,
            "outcome": self.outcome,
            "passed_predicate_ids": list(self.passed_predicate_ids),
            "failed_predicate_ids": list(self.failed_predicate_ids),
            "inconclusive_predicate_ids": list(self.inconclusive_predicate_ids),
            "gaps": [
                {
                    "predicate_id": predicate_id,
                    "outcome": outcome,
                    "reasons": list(reasons),
                }
                for predicate_id, outcome, reasons in self.gaps
            ],
            "matched_evidence_ids": list(self.matched_evidence_ids),
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready coverage report."""
        return {**self._content_record(), "report_id": self.report_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> QualificationCoverageReport:
        """Reconstruct and content-verify a qualification coverage report."""
        if not isinstance(record, Mapping):
            raise TypeError("Qualification-coverage report must be a mapping.")
        sequence_fields = (
            "passed_predicate_ids",
            "failed_predicate_ids",
            "inconclusive_predicate_ids",
            "gaps",
            "matched_evidence_ids",
        )
        for name in sequence_fields:
            values = record[name]
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TypeError(f"Serialized {name} must be a sequence.")
        gaps: list[tuple[str, str, tuple[str, ...]]] = []
        for item in record["gaps"]:
            if not isinstance(item, Mapping):
                raise TypeError("Serialized coverage gaps must be mappings.")
            reasons = item["reasons"]
            if not isinstance(reasons, Sequence) or isinstance(reasons, str):
                raise TypeError("Serialized coverage gap reasons must be a sequence.")
            gaps.append(
                (
                    str(item["predicate_id"]),
                    str(item["outcome"]),
                    tuple(str(reason) for reason in reasons),
                )
            )
        value = cls(
            str(record["matrix_id"]),
            evaluated_at=int(record["evaluated_at"]),
            passed_predicate_ids=tuple(
                str(item) for item in record["passed_predicate_ids"]
            ),
            failed_predicate_ids=tuple(
                str(item) for item in record["failed_predicate_ids"]
            ),
            inconclusive_predicate_ids=tuple(
                str(item) for item in record["inconclusive_predicate_ids"]
            ),
            gaps=gaps,
            matched_evidence_ids=tuple(
                str(item) for item in record["matched_evidence_ids"]
            ),
        )
        if str(record["outcome"]) != value.outcome:
            raise ValueError("Serialized coverage report has an invalid outcome.")
        recorded_id = record.get("report_id")
        if recorded_id is not None and str(recorded_id) != value.report_id:
            raise ValueError(
                "Serialized qualification coverage has an invalid content address."
            )
        return value


class QualificationMatrix(StrictModule, NonTrainableState):
    """Named exact evidence predicates evaluated as a fail-closed conjunction."""

    predicates: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = eqx.field(
        static=True
    )
    matrix_id: str = eqx.field(static=True)

    def __init__(
        self,
        predicates: Mapping[str, Mapping[str, str]],
        /,
    ):
        if not isinstance(predicates, Mapping) or not predicates:
            raise TypeError("Qualification predicates must be a non-empty mapping.")
        normalized: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for predicate_id, requirements in predicates.items():
            predicate_id_ = _identifier(predicate_id, "qualification predicate ID")
            if not isinstance(requirements, Mapping) or not requirements:
                raise TypeError(
                    "Each qualification predicate must be a non-empty mapping."
                )
            unknown = set(requirements) - _PREDICATE_FIELDS
            if unknown:
                raise ValueError(
                    "Unknown qualification predicate fields: "
                    + ", ".join(sorted(str(name) for name in unknown))
                )
            if "evidence_kind" not in requirements:
                raise ValueError(
                    "Every qualification predicate must name an exact evidence_kind."
                )
            predicate = tuple(
                sorted(
                    (
                        _identifier(name, "qualification predicate field"),
                        _identifier(value, f"qualification predicate {name}"),
                    )
                    for name, value in requirements.items()
                )
            )
            evidence_kind = dict(predicate)["evidence_kind"]
            if evidence_kind not in _EVIDENCE_KINDS:
                raise ValueError(f"Unknown evidence kind {evidence_kind!r}.")
            normalized.append((predicate_id_, predicate))
        normalized.sort()
        if len({predicate_id for predicate_id, _ in normalized}) != len(normalized):
            raise ValueError("Qualification predicate IDs must be unique.")
        self.predicates = tuple(normalized)
        self.matrix_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "qualification-matrix",
            "predicates": {
                predicate_id: dict(predicate)
                for predicate_id, predicate in self.predicates
            },
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready qualification matrix."""
        return {**self._content_record(), "matrix_id": self.matrix_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> QualificationMatrix:
        """Reconstruct and content-verify a qualification matrix."""
        if not isinstance(record, Mapping):
            raise TypeError("Qualification-matrix record must be a mapping.")
        predicates = record["predicates"]
        if not isinstance(predicates, Mapping) or any(
            not isinstance(value, Mapping) for value in predicates.values()
        ):
            raise TypeError("Serialized qualification predicates must be mappings.")
        value = cls(
            {
                str(predicate_id): {
                    str(name): str(expected) for name, expected in requirements.items()
                }
                for predicate_id, requirements in predicates.items()
            }
        )
        recorded_id = record.get("matrix_id")
        if recorded_id is not None and str(recorded_id) != value.matrix_id:
            raise ValueError(
                "Serialized qualification matrix has an invalid content address."
            )
        return value

    def evaluate(
        self,
        evidence: Sequence[QualificationEvidence],
        /,
        *,
        at_time: int,
    ) -> QualificationCoverageReport:
        """Evaluate every predicate without conflating failures with missing proof."""
        if not isinstance(evidence, Sequence) or isinstance(evidence, str):
            raise TypeError("evidence must be a sequence of QualificationEvidence.")
        if any(not isinstance(item, QualificationEvidence) for item in evidence):
            raise TypeError("evidence must contain QualificationEvidence values.")
        timestamp = _timestamp(at_time, "at_time")
        by_id = {item.evidence_id: item for item in evidence}
        if len(by_id) != len(evidence):
            raise ValueError("Qualification evidence IDs must be unique.")
        records = tuple(sorted(by_id.values(), key=lambda item: item.evidence_id))
        superseded_ids = frozenset(
            superseded
            for item in records
            if item.issued_at <= timestamp
            for superseded in item.supersedes_evidence_ids
        )
        passed: list[str] = []
        failed: list[str] = []
        inconclusive: list[str] = []
        gaps: list[tuple[str, str, tuple[str, ...]]] = []
        matched_ids: set[str] = set()
        for predicate_id, predicate in self.predicates:
            matching = tuple(
                item for item in records if _predicate_matches(item, predicate)
            )
            active = tuple(
                item
                for item in matching
                if item.is_current(timestamp) and item.evidence_id not in superseded_ids
            )
            matched_ids.update(item.evidence_id for item in active)
            failed_records = tuple(item for item in active if item.failed)
            inconclusive_records = tuple(item for item in active if item.inconclusive)
            if failed_records:
                failed.append(predicate_id)
                gaps.append(
                    (
                        predicate_id,
                        "failed",
                        tuple(
                            sorted(
                                f"failed:{item.evidence_id}:{item.reason}"
                                for item in failed_records
                            )
                        ),
                    )
                )
            elif inconclusive_records:
                inconclusive.append(predicate_id)
                gaps.append(
                    (
                        predicate_id,
                        "inconclusive",
                        tuple(
                            sorted(
                                f"inconclusive:{item.evidence_id}:{item.reason}"
                                for item in inconclusive_records
                            )
                        ),
                    )
                )
            elif active:
                passed.append(predicate_id)
            else:
                reasons: list[str] = []
                for item in matching:
                    if item.evidence_id in superseded_ids:
                        reasons.append(f"superseded-evidence:{item.evidence_id}")
                    elif item.issued_at > timestamp:
                        reasons.append(f"future-evidence:{item.evidence_id}")
                    else:
                        reasons.append(f"expired-evidence:{item.evidence_id}")
                if not reasons:
                    reasons.append("missing-evidence")
                inconclusive.append(predicate_id)
                gaps.append((predicate_id, "inconclusive", tuple(sorted(set(reasons)))))
        return QualificationCoverageReport(
            self.matrix_id,
            evaluated_at=timestamp,
            passed_predicate_ids=passed,
            failed_predicate_ids=failed,
            inconclusive_predicate_ids=inconclusive,
            gaps=gaps,
            matched_evidence_ids=tuple(sorted(matched_ids)),
        )


__all__ = [
    "ForecastResourceRecord",
    "ObservedResourceRecord",
    "QualificationCoverageReport",
    "QualificationEvidence",
    "QualificationMatrix",
    "SupportDependency",
]
