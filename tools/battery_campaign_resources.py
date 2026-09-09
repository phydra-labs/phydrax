#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite, ex-ante absolute workload criteria independent of relative regressions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.battery._release_contracts import RESOURCE_METRICS
from phydrax.qualification import QualificationCriterion


@dataclass(frozen=True, slots=True)
class BatteryResourcePlan:
    case_id: str
    workload_id: str
    sample_count: int
    criteria: tuple[QualificationCriterion, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.workload_id:
            raise ValueError("Resource plan requires exact case and workload IDs.")
        if type(self.sample_count) is not int or not 5 <= self.sample_count <= 256:
            raise ValueError("Resource plans require 5..256 synchronized warm samples.")
        schema = {
            name: (unit, aggregation) for name, unit, aggregation, _ in RESOURCE_METRICS
        }
        if len(self.criteria) != len(schema) or {
            criterion.metric for criterion in self.criteria
        } != set(schema):
            raise ValueError(
                "Resource plan must precommit the complete absolute resource matrix."
            )
        supports = {criterion.support_tuple_id for criterion in self.criteria}
        if len(supports) != 1:
            raise ValueError("Resource criteria must share exact model support.")
        for criterion in self.criteria:
            QualificationCriterion.from_record(criterion.to_record())
            unit, aggregation = schema[criterion.metric]
            if (
                criterion.unit != unit
                or criterion.aggregation != aggregation
                or criterion.uncertainty != "observed-single-environment"
                or criterion.applicability != self.case_id
                or criterion.comparison != "less-than-or-equal"
                or not math.isfinite(criterion.target)
                or criterion.target < 0.0
            ):
                raise ValueError(
                    "Resource criterion differs from its finite observable contract."
                )
            if (
                criterion.metric in ("global-dense-arrays", "unsuccessful-executions")
                and criterion.target != 0.0
            ):
                raise ValueError(
                    "No-dense and execution-success criteria must precommit exactly zero violations."
                )

    @property
    def plan_id(self) -> str:
        return canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "battery-absolute-resource-plan",
            "case_id": self.case_id,
            "workload_id": self.workload_id,
            "sample_count": self.sample_count,
            "criteria": [
                criterion.to_record()
                for criterion in sorted(self.criteria, key=lambda item: item.metric)
            ],
            "formulas": {name: formula for name, _, _, formula in RESOURCE_METRICS},
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "plan_id": self.plan_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> BatteryResourcePlan:
        if not isinstance(record, Mapping) or set(record) != {
            "kind",
            "case_id",
            "workload_id",
            "sample_count",
            "criteria",
            "formulas",
            "plan_id",
        }:
            raise ValueError("Resource plan must contain exactly the registered fields.")
        result = cls(
            record["case_id"],
            record["workload_id"],
            record["sample_count"],
            tuple(
                QualificationCriterion.from_record(item) for item in record["criteria"]
            ),
        )
        if record != result.to_record():
            raise ValueError("Resource plan schema or content address is invalid.")
        return result


def classify_resource(
    criterion: QualificationCriterion, value: float | int | None, /
) -> tuple[str, str]:
    if value is None:
        return "inconclusive", "required-resource-measurement-unavailable"
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        return "inconclusive", "resource-measurement-invalid"
    if value > criterion.target:
        return "failed", "absolute-resource-target-exceeded"
    return "passed", "absolute-resource-target-satisfied"
