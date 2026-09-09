#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact executable campaign contracts, not a catalogue of promised support."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.battery._release_contracts import (
    battery_release_contract,
    CampaignCase,
    CampaignMetric,
)
from phydrax.applications.battery._results import BatterySelectedOutputs
from phydrax.qualification import CapabilityProfile, QualificationCriterion, SupportTuple


def metric_key(case_id: str, metric: str, /) -> str:
    return f"{case_id}/{metric}"


def trajectory_observation(
    case_id: str, run_id: str, outputs: BatterySelectedOutputs, /
) -> dict[str, object]:
    """Encode measured samples and absent suffixes as strict portable JSON."""
    times = np.asarray(outputs.times_s)
    values = np.asarray(outputs.values)
    valid = np.asarray(outputs.valid)
    return {
        "case_id": case_id,
        "run_id": run_id,
        "times_s": [
            float(time) if active and np.isfinite(time) else None
            for time, active in zip(times, valid, strict=True)
        ],
        "outputs": [
            [float(value) if active and np.isfinite(value) else None for value in row]
            for row, active in zip(values, valid, strict=True)
        ],
        "valid": valid.tolist(),
        "output_names": list(outputs.names),
    }


@dataclass(frozen=True, slots=True)
class PreparedCampaign:
    campaign_kind: str
    discretization_id: str
    parameter_id: str
    model_selection_id: str
    execute: Callable[[], object]
    operation: Callable[[], object]
    resources: Callable[[object], dict[str, int | float | None]] | None = None

    def prepared_configuration_id(self, schedule_id: str, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "battery-builtin-prepared-configuration",
                "campaign_kind": self.campaign_kind,
                "discretization_id": self.discretization_id,
                "parameter_id": self.parameter_id,
                "model_selection_id": self.model_selection_id,
                "schedule_id": schedule_id,
                "registry_entry_id": get_campaign_entry(self.campaign_kind).entry_id,
            }
        )


@dataclass(frozen=True, slots=True)
class CampaignEntry:
    campaign_kind: str
    candidate_profile: CapabilityProfile
    candidate_support: SupportTuple
    cases: tuple[CampaignCase, ...]
    reference_count: int
    required_rights: tuple[str, ...]
    prepare: Callable[[Sequence[float]], PreparedCampaign]
    raw_output: Callable[[object, object, Path], dict[str, object]]
    raw_fields: tuple[str, ...] = ("metrics",)

    def __post_init__(self) -> None:
        if not self.campaign_kind or not self.cases:
            raise ValueError("Campaign entries require a kind and finite cases.")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Campaign case IDs must be unique.")
        if type(self.reference_count) is not int or self.reference_count < 0:
            raise ValueError("Campaign reference count must be a nonnegative integer.")
        if not set(self.required_rights) <= {
            "commercial_use",
            "redistribution",
            "training_use",
            "export",
        }:
            raise ValueError("Unknown reference use right.")
        if "metrics" not in self.raw_fields or len(set(self.raw_fields)) != len(
            self.raw_fields
        ):
            raise ValueError("Raw schema must declare unique fields including metrics.")
        if not callable(self.prepare) or not callable(self.raw_output):
            raise TypeError(
                "Registry entries require real preparation and observation evaluators."
            )
        contract = battery_release_contract(
            dict(self.candidate_support.attributes)["model_id"]
        )
        expected = dict(contract.scientific_cases)
        if any(
            case.case_id not in expected or case.metrics != expected[case.case_id]
            for case in self.cases
        ):
            raise ValueError(
                "Executable campaign declarations must consume the exact native model metric schema."
            )

    @property
    def entry_id(self) -> str:
        return canonical_fingerprint(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "battery-campaign-registry-entry",
            "campaign_kind": self.campaign_kind,
            "candidate_profile_id": self.candidate_profile.profile_id,
            "candidate_support_tuple_id": self.candidate_support.support_tuple_id,
            "cases": [case.to_record() for case in self.cases],
            "reference_count": self.reference_count,
            "required_rights": list(self.required_rights),
            "raw_fields": list(self.raw_fields),
            "raw_metric_fields": ["value", "unavailable_reason"],
            "outcome_policy": "observed-violation-failed;unavailable-inconclusive",
            "prepare": f"{self.prepare.__module__}.{self.prepare.__qualname__}",
            "evaluator": f"{self.raw_output.__module__}.{self.raw_output.__qualname__}",
        }

    def metrics(self) -> dict[str, CampaignMetric]:
        return {
            metric_key(case.case_id, metric.name): metric
            for case in self.cases
            for metric in case.metrics
        }

    def evidence_kind(self, criterion: QualificationCriterion, /) -> str:
        contract = battery_release_contract(
            dict(self.candidate_support.attributes)["model_id"]
        )
        return (
            "reference"
            if criterion.applicability in contract.reference_cases
            else "scientific"
        )

    def validate_criteria(self, criteria: Sequence[QualificationCriterion], /) -> None:
        expected = self.metrics()
        actual: set[str] = set()
        for criterion in criteria:
            if criterion.support_tuple_id != self.candidate_support.support_tuple_id:
                raise ValueError(
                    "Criterion support differs from the registered campaign."
                )
            key = metric_key(criterion.applicability, criterion.metric)
            if key not in expected or key in actual:
                raise ValueError(
                    "Criteria must bind each registered case/metric exactly once."
                )
            expected[key].validate(criterion)
            actual.add(key)
        if actual != set(expected):
            raise ValueError("Criteria do not cover the complete finite campaign matrix.")

    def validate_raw(self, raw: Mapping[str, object], /) -> None:
        if set(raw) != set(self.raw_fields):
            raise ValueError("Campaign output differs from its exact raw schema.")
        metrics = raw["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(self.metrics()):
            raise ValueError("Raw output must account for every registered case/metric.")
        for measurement in metrics.values():
            if not isinstance(measurement, Mapping) or set(measurement) != {
                "value",
                "unavailable_reason",
            }:
                raise ValueError("Raw measurements require value and unavailable_reason.")
            value, reason = measurement["value"], measurement["unavailable_reason"]
            if value is None:
                if not isinstance(reason, str) or not reason:
                    raise ValueError(
                        "Unavailable measurements require an explicit reason."
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or reason is not None
            ):
                raise ValueError(
                    "Observed measurements require finite numbers and no unavailable reason."
                )
        canonical_fingerprint(raw)

    def classify(
        self, criterion: QualificationCriterion, measurement: Mapping[str, object], /
    ) -> tuple[str, str]:
        value = measurement["value"]
        if value is None:
            return "inconclusive", str(measurement["unavailable_reason"])
        if criterion.comparison == "equal":
            passed = value == criterion.target
        elif criterion.comparison == "less-than-or-equal":
            passed = value <= criterion.target
        else:
            passed = value >= criterion.target
        return (
            ("passed", "registered-threshold-satisfied")
            if passed
            else ("failed", "observed-threshold-violation")
        )


def get_campaign_entry(campaign_kind: str, /) -> CampaignEntry:
    if campaign_kind == "ecm-analytic":
        from tools._battery_ecm_campaign import campaign_entry
    elif campaign_kind == "spme-marquis2019-scientific":
        from tools._battery_spme_campaign import campaign_entry
    elif campaign_kind == "circuit-ecm-analytic":
        from tools._battery_circuit_ecm_campaign import campaign_entry
    elif campaign_kind == "circuit-ecm-lifecycle":
        from tools._battery_circuit_ecm_lifecycle_campaign import campaign_entry
    else:
        raise ValueError(
            f"Unsupported battery campaign kind {campaign_kind!r}; no executable registry entry."
        )
    entry = campaign_entry()
    if entry.campaign_kind != campaign_kind:
        raise ValueError("Campaign registry factory returned a mismatched kind.")
    return entry


def prepare_builtin_campaign(
    campaign_kind: str, sample_times_s: Sequence[float], /
) -> PreparedCampaign:
    times = tuple(sample_times_s)
    if not 2 <= len(times) <= 4096 or any(
        isinstance(value, bool) or not math.isfinite(value) for value in times
    ):
        raise ValueError("Campaign schedules require 2..4096 finite sampling times.")
    if times[0] != 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(
            "Campaign sampling times must start at zero and strictly increase."
        )
    return get_campaign_entry(campaign_kind).prepare(times)
