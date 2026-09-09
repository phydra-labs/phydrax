#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._evidence import QualificationEvidence, QualificationMatrix
from ._registry import SupportTuple


MetricDirection = Literal["at_most", "at_least", "between"]
MetricAggregation = Literal["pooled", "independent_unit_macro", "worst_stratum"]
_STAGE_IDS = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "numerical-validity",
        "chemical-validity",
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
        "external-transfer",
        "prospective-intervention",
    )
)
_DIRECTIONS = frozenset(("at_most", "at_least", "between"))
_AGGREGATIONS = frozenset(("pooled", "independent_unit_macro", "worst_stratum"))


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
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


def _bound(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number or None.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite when present.")
    return normalized


@dataclass(frozen=True, slots=True)
class ScientificMetricCriterion:
    """One unit- and aggregation-exact scientific metric threshold."""

    metric_id: str
    direction: MetricDirection
    lower: float | None
    upper: float | None
    unit_id: str
    aggregation: MetricAggregation
    criterion_id: str = field(init=False)

    def __post_init__(self) -> None:
        metric_id = _identifier(self.metric_id, "metric_id")
        direction = _identifier(self.direction, "metric direction")
        unit_id = _identifier(self.unit_id, "unit_id")
        aggregation = _identifier(self.aggregation, "metric aggregation")
        if direction not in _DIRECTIONS:
            raise ValueError("Metric direction must be at_most, at_least, or between.")
        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                "Metric aggregation must be pooled, independent_unit_macro, "
                "or worst_stratum."
            )
        lower = _bound(self.lower, "metric lower bound")
        upper = _bound(self.upper, "metric upper bound")
        if direction == "at_most" and (lower is not None or upper is None):
            raise ValueError("An at_most criterion requires only an upper bound.")
        if direction == "at_least" and (lower is None or upper is not None):
            raise ValueError("An at_least criterion requires only a lower bound.")
        if direction == "between" and (lower is None or upper is None or lower > upper):
            raise ValueError(
                "A between criterion requires ordered lower and upper bounds."
            )
        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "aggregation", aggregation)
        object.__setattr__(
            self,
            "criterion_id",
            canonical_fingerprint(self._content_record()),
        )

    def passes(self, value: float, /) -> bool:
        """Return whether one finite value satisfies this criterion."""
        if self.direction == "at_most":
            return value <= self.upper
        if self.direction == "at_least":
            return value >= self.lower
        return self.lower <= value <= self.upper

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "scientific-metric-criterion",
            "metric_id": self.metric_id,
            "direction": self.direction,
            "lower": self.lower,
            "upper": self.upper,
            "unit_id": self.unit_id,
            "aggregation": self.aggregation,
        }

    def to_record(self) -> dict[str, object]:
        """Return the deterministic JSON-ready criterion record."""
        return {**self._content_record(), "criterion_id": self.criterion_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ScientificMetricCriterion:
        """Reconstruct a scientific metric criterion."""
        if not isinstance(record, Mapping):
            raise TypeError("Scientific-metric criterion record must be a mapping.")
        lower = record["lower"]
        upper = record["upper"]
        value = cls(
            str(record["metric_id"]),
            str(record["direction"]),
            None if lower is None else float(lower),
            None if upper is None else float(upper),
            str(record["unit_id"]),
            str(record["aggregation"]),
        )
        recorded_id = record.get("criterion_id")
        if recorded_id is not None and str(recorded_id) != value.criterion_id:
            raise ValueError(
                "Serialized scientific metric criterion has an invalid content address."
            )
        return value


class ScientificClaimProfile(StrictModule, NonTrainableState):
    """An exact scientific claim evaluated against existing evidence records."""

    capability_name: str = eqx.field(static=True)
    support: SupportTuple = eqx.field(static=True)
    observable_ids: tuple[str, ...] = eqx.field(static=True)
    condition_domain_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    required_stage_ids: tuple[str, ...] = eqx.field(static=True)
    criteria: tuple[ScientificMetricCriterion, ...] = eqx.field(static=True)
    frozen_criteria_ids: tuple[str, ...] = eqx.field(static=True)
    abstention_policy_id: str = eqx.field(static=True)
    invalidation_triggers: tuple[str, ...] = eqx.field(static=True)
    claim_id: str = eqx.field(static=True)

    def __init__(
        self,
        capability_name: str,
        support: SupportTuple,
        observable_ids: Sequence[str],
        condition_domain_ids: Sequence[str],
        campaign_id: str,
        required_stage_ids: Sequence[str],
        criteria: Sequence[ScientificMetricCriterion],
        abstention_policy_id: str,
        invalidation_triggers: Sequence[str],
        /,
        *,
        frozen_criteria_ids: Sequence[str],
    ):
        capability = _identifier(capability_name, "capability_name")
        if not isinstance(support, SupportTuple):
            raise TypeError("support must be a SupportTuple.")
        if support.capability != capability:
            raise ValueError("Claim capability_name must match support.capability.")
        observables = _identifiers(observable_ids, "observable_ids")
        conditions = _identifiers(condition_domain_ids, "condition_domain_ids")
        campaign = _identifier(campaign_id, "campaign_id")
        stages = _identifiers(required_stage_ids, "required_stage_ids")
        unknown_stages = set(stages) - _STAGE_IDS
        if unknown_stages:
            raise ValueError(
                "Unknown scientific qualification stage IDs: "
                + ", ".join(sorted(unknown_stages))
            )
        if (
            not isinstance(criteria, Sequence)
            or isinstance(criteria, str)
            or not criteria
        ):
            raise TypeError(
                "criteria must be a non-empty sequence of ScientificMetricCriterion."
            )
        if any(not isinstance(item, ScientificMetricCriterion) for item in criteria):
            raise TypeError("criteria must contain ScientificMetricCriterion values.")
        criteria_ = tuple(sorted(criteria, key=lambda item: item.metric_id))
        metric_ids = tuple(item.metric_id for item in criteria_)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("Scientific metric IDs must be unique within a claim.")
        frozen_criteria = _identifiers(
            frozen_criteria_ids,
            "frozen_criteria_ids",
        )
        unfrozen_criteria = sorted(
            {item.criterion_id for item in criteria_} - set(frozen_criteria)
        )
        if unfrozen_criteria:
            raise ValueError(
                "Scientific claim criteria must be frozen in campaign criteria_ids: "
                + ", ".join(unfrozen_criteria)
            )
        abstention = _identifier(abstention_policy_id, "abstention_policy_id")
        triggers = _identifiers(invalidation_triggers, "invalidation_triggers")

        self.capability_name = capability
        self.support = support
        self.observable_ids = observables
        self.condition_domain_ids = conditions
        self.campaign_id = campaign
        self.required_stage_ids = stages
        self.criteria = criteria_
        self.frozen_criteria_ids = frozen_criteria
        self.abstention_policy_id = abstention
        self.invalidation_triggers = triggers
        self.claim_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "scientific-claim-profile",
            "capability_name": self.capability_name,
            "support": self.support.to_record(),
            "observable_ids": list(self.observable_ids),
            "condition_domain_ids": list(self.condition_domain_ids),
            "campaign_id": self.campaign_id,
            "required_stage_ids": list(self.required_stage_ids),
            "criteria": [criterion.to_record() for criterion in self.criteria],
            "frozen_criteria_ids": list(self.frozen_criteria_ids),
            "abstention_policy_id": self.abstention_policy_id,
            "invalidation_triggers": list(self.invalidation_triggers),
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready claim record with its content address."""
        return {**self._content_record(), "claim_id": self.claim_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ScientificClaimProfile:
        """Reconstruct and content-verify a serialized scientific claim."""
        if not isinstance(record, Mapping):
            raise TypeError("Scientific-claim profile record must be a mapping.")
        support = record["support"]
        observables = record["observable_ids"]
        conditions = record["condition_domain_ids"]
        stages = record["required_stage_ids"]
        criteria = record["criteria"]
        frozen_criteria = record["frozen_criteria_ids"]
        triggers = record["invalidation_triggers"]
        if not isinstance(support, Mapping):
            raise TypeError("Serialized claim support must be a mapping.")
        sequence_fields = {
            "observable_ids": observables,
            "condition_domain_ids": conditions,
            "required_stage_ids": stages,
            "criteria": criteria,
            "frozen_criteria_ids": frozen_criteria,
            "invalidation_triggers": triggers,
        }
        if any(
            not isinstance(values, Sequence) or isinstance(values, str)
            for values in sequence_fields.values()
        ):
            raise TypeError("Serialized claim collections must be sequences.")
        value = cls(
            str(record["capability_name"]),
            SupportTuple.from_record(support),
            tuple(str(item) for item in observables),
            tuple(str(item) for item in conditions),
            str(record["campaign_id"]),
            tuple(str(item) for item in stages),
            tuple(ScientificMetricCriterion.from_record(item) for item in criteria),
            str(record["abstention_policy_id"]),
            tuple(str(item) for item in triggers),
            frozen_criteria_ids=tuple(str(item) for item in frozen_criteria),
        )
        recorded_id = record.get("claim_id")
        if recorded_id is not None and str(recorded_id) != value.claim_id:
            raise ValueError(
                "Serialized scientific claim has an invalid content address."
            )
        return value

    def evaluate(
        self,
        metric_values: Mapping[str, float],
        stage_evidence: Sequence[QualificationEvidence],
        /,
        *,
        metric_units: Mapping[str, str],
        metric_aggregations: Mapping[str, str],
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        raw_artifact_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
    ) -> QualificationEvidence:
        """Evaluate exact stage and metric evidence without conflating gaps with failures."""
        mappings = {
            "metric_values": metric_values,
            "metric_units": metric_units,
            "metric_aggregations": metric_aggregations,
        }
        if any(not isinstance(values, Mapping) for values in mappings.values()):
            raise TypeError("Metric values, units, and aggregations must be mappings.")
        if not isinstance(stage_evidence, Sequence) or isinstance(stage_evidence, str):
            raise TypeError("stage_evidence must be a sequence of QualificationEvidence.")
        if any(not isinstance(item, QualificationEvidence) for item in stage_evidence):
            raise TypeError("stage_evidence must contain QualificationEvidence values.")

        stage_matrix = QualificationMatrix(
            {
                stage_id: {
                    "evidence_kind": "scientific",
                    "subject_id": self.campaign_id,
                    "criterion_id": stage_id,
                }
                for stage_id in self.required_stage_ids
            }
        )
        stage_report = stage_matrix.evaluate(stage_evidence, at_time=issued_at)
        matched_evidence_ids = frozenset(stage_report.matched_evidence_ids)
        matched_evidence = tuple(
            item for item in stage_evidence if item.evidence_id in matched_evidence_ids
        )
        effective_expires_at = expires_at
        if (
            matched_evidence
            and isinstance(expires_at, int)
            and not isinstance(expires_at, bool)
        ):
            effective_expires_at = min(
                expires_at,
                *(item.expires_at for item in matched_evidence),
            )
            if effective_expires_at <= issued_at:
                raise ValueError(
                    "Matched prerequisite evidence must remain current after "
                    "derived issuance."
                )

        failed_metrics: list[str] = []
        inconclusive_metrics: list[str] = []
        metric_evaluation_records: list[dict[str, object]] = []
        for criterion in self.criteria:
            metric_id = criterion.metric_id
            value_present = metric_id in metric_values
            unit_present = metric_id in metric_units
            aggregation_present = metric_id in metric_aggregations
            raw_value = metric_values[metric_id] if value_present else None
            unit = metric_units[metric_id] if unit_present else None
            aggregation = metric_aggregations[metric_id] if aggregation_present else None
            value: float | None = None
            if not isinstance(raw_value, bool) and isinstance(raw_value, Real):
                candidate = float(raw_value)
                if math.isfinite(candidate):
                    value = candidate
            metric_evaluation_records.append(
                {
                    "metric_id": metric_id,
                    "value_present": value_present,
                    "value": value,
                    "unit_present": unit_present,
                    "unit": unit if isinstance(unit, str) else None,
                    "aggregation_present": aggregation_present,
                    "aggregation": (
                        aggregation if isinstance(aggregation, str) else None
                    ),
                }
            )
            if not value_present or not unit_present or not aggregation_present:
                inconclusive_metrics.append(metric_id)
                continue
            if not isinstance(unit, str) or not isinstance(aggregation, str):
                inconclusive_metrics.append(metric_id)
                continue
            if unit != criterion.unit_id or aggregation != criterion.aggregation:
                inconclusive_metrics.append(metric_id)
                continue
            if value is None:
                inconclusive_metrics.append(metric_id)
                continue
            if not criterion.passes(value):
                failed_metrics.append(metric_id)
        metric_evaluation_id = canonical_fingerprint(
            {
                "kind": "scientific-claim-metric-evaluation",
                "claim_id": self.claim_id,
                "metrics": metric_evaluation_records,
            }
        )

        failed_ids = tuple(sorted((*stage_report.failed_predicate_ids, *failed_metrics)))
        inconclusive_ids = tuple(
            sorted((*stage_report.inconclusive_predicate_ids, *inconclusive_metrics))
        )
        if failed_ids:
            outcome = "failed"
            reason = "failed-required-scientific-evidence:" + ",".join(failed_ids)
        elif inconclusive_ids:
            outcome = "inconclusive"
            reason = "inconclusive-required-scientific-evidence:" + ",".join(
                inconclusive_ids
            )
        else:
            outcome = "passed"
            reason = "required-scientific-stages-and-metrics-passed"

        criterion_ids = tuple(
            sorted(
                set(self.required_stage_ids)
                | {criterion.metric_id for criterion in self.criteria}
            )
        )
        return QualificationEvidence(
            "scientific",
            outcome,
            (
                self.campaign_id,
                self.claim_id,
                self.support.support_tuple_id,
                metric_evaluation_id,
                stage_report.report_id,
                *stage_report.matched_evidence_ids,
            ),
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            criteria_ids=criterion_ids,
            raw_artifact_ids=raw_artifact_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=effective_expires_at,
            reason=reason,
            requalification_triggers=self.invalidation_triggers,
        )


__all__ = [
    "MetricAggregation",
    "MetricDirection",
    "ScientificClaimProfile",
    "ScientificMetricCriterion",
]
