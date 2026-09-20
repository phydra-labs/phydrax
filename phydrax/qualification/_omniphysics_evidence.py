#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Content-addressed omniphysics controls, refinements, providers, and validations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .._fingerprint import canonical_fingerprint
from ._registry import _identifier


@dataclass(frozen=True, slots=True)
class NumericalControlEvidence:
    family: str
    control: str
    observed: float
    reference: float
    absolute_tolerance: float
    relative_tolerance: float
    independent_reference: str

    def __post_init__(self):
        values = (
            self.observed,
            self.reference,
            self.absolute_tolerance,
            self.relative_tolerance,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Numerical-control values must be finite.")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("Numerical-control tolerances must be non-negative.")
        _identifier(self.family, "control family")
        _identifier(self.control, "control name")
        _identifier(self.independent_reference, "independent reference")

    @property
    def absolute_error(self) -> float:
        return abs(self.observed - self.reference)

    @property
    def passed(self) -> bool:
        threshold = self.absolute_tolerance + self.relative_tolerance * abs(
            self.reference
        )
        return self.absolute_error <= threshold

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "omniphysics-numerical-control",
            "family": self.family,
            "control": self.control,
            "observed": self.observed,
            "reference": self.reference,
            "absolute_error": self.absolute_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "independent_reference": self.independent_reference,
            "passed": self.passed,
        }
        return {**record, "evidence_id": self.evidence_id} if include_id else record


@dataclass(frozen=True, slots=True)
class RefinementCampaignEvidence:
    campaign: str
    family_ids: tuple[str, ...]
    resolutions: tuple[float, ...]
    errors: tuple[float, ...]
    minimum_order: float
    reference: str

    def __post_init__(self):
        _identifier(self.campaign, "refinement campaign")
        _identifier(self.reference, "refinement reference")
        if not self.family_ids or len(set(self.family_ids)) != len(self.family_ids):
            raise ValueError("Refinement campaign requires unique family IDs.")
        if len(self.resolutions) != len(self.errors) or len(self.errors) < 3:
            raise ValueError(
                "Refinement campaign requires at least three aligned levels."
            )
        if any(value <= 0 or not np.isfinite(value) for value in self.resolutions):
            raise ValueError("Refinement resolutions must be finite and positive.")
        if any(value <= 0 or not np.isfinite(value) for value in self.errors):
            raise ValueError("Refinement errors must be finite and positive.")
        if any(fine >= coarse for coarse, fine in pairwise(self.resolutions)):
            raise ValueError("Refinement resolution values must strictly decrease.")
        if self.minimum_order < 0:
            raise ValueError("Minimum refinement order must be non-negative.")

    @property
    def observed_orders(self) -> tuple[float, ...]:
        orders = []
        for (coarse_h, fine_h), (coarse_error, fine_error) in zip(
            pairwise(self.resolutions),
            pairwise(self.errors),
            strict=True,
        ):
            orders.append(
                float(np.log(coarse_error / fine_error) / np.log(coarse_h / fine_h))
            )
        return tuple(orders)

    @property
    def passed(self) -> bool:
        decreasing = all(fine < coarse for coarse, fine in pairwise(self.errors))
        return decreasing and all(
            order >= self.minimum_order for order in self.observed_orders
        )

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "omniphysics-refinement-campaign",
            "campaign": self.campaign,
            "family_ids": list(self.family_ids),
            "resolutions": list(self.resolutions),
            "errors": list(self.errors),
            "observed_orders": list(self.observed_orders),
            "minimum_order": self.minimum_order,
            "reference": self.reference,
            "passed": self.passed,
        }
        return {**record, "evidence_id": self.evidence_id} if include_id else record


@dataclass(frozen=True, slots=True)
class HardwareProviderEvidence:
    provider_id: str
    platform: str
    process_count: int
    host_count: int
    device_count: int
    device_kinds: tuple[str, ...]
    precision: str
    executed: bool

    def __post_init__(self):
        for value, label in (
            (self.provider_id, "provider ID"),
            (self.platform, "provider platform"),
            (self.precision, "provider precision"),
        ):
            _identifier(value, label)
        if min(self.process_count, self.host_count, self.device_count) <= 0:
            raise ValueError(
                "Provider process, host, and device counts must be positive."
            )
        if not self.device_kinds:
            raise ValueError("Provider evidence requires device kinds.")

    @property
    def distributed(self) -> bool:
        return self.process_count > 1 and self.host_count > 1

    @property
    def qualifies_distributed(self) -> bool:
        return self.executed and self.distributed

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "omniphysics-hardware-provider",
            "provider_id": self.provider_id,
            "platform": self.platform,
            "process_count": self.process_count,
            "host_count": self.host_count,
            "device_count": self.device_count,
            "device_kinds": list(self.device_kinds),
            "precision": self.precision,
            "executed": self.executed,
            "distributed": self.distributed,
            "qualifies_distributed": self.qualifies_distributed,
        }
        return {**record, "evidence_id": self.evidence_id} if include_id else record


@dataclass(frozen=True, slots=True)
class ApplicationValidationEvidence:
    application: str
    reference_case: str
    metrics: tuple[tuple[str, float, float, float], ...]
    independent_source: str

    def __post_init__(self):
        _identifier(self.application, "validated application")
        _identifier(self.reference_case, "validation reference case")
        _identifier(self.independent_source, "validation source")
        if not self.metrics or len({name for name, *_ in self.metrics}) != len(
            self.metrics
        ):
            raise ValueError("Application validation requires unique metrics.")
        if any(
            not np.isfinite(observed)
            or not np.isfinite(reference)
            or not np.isfinite(tolerance)
            or tolerance < 0
            for _, observed, reference, tolerance in self.metrics
        ):
            raise ValueError("Application validation metrics are invalid.")

    @property
    def passed(self) -> bool:
        return all(
            abs(observed - reference) <= tolerance
            for _, observed, reference, tolerance in self.metrics
        )

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "omniphysics-application-validation",
            "application": self.application,
            "reference_case": self.reference_case,
            "metrics": [
                {
                    "name": name,
                    "observed": observed,
                    "reference": reference,
                    "absolute_tolerance": tolerance,
                    "passed": abs(observed - reference) <= tolerance,
                }
                for name, observed, reference, tolerance in self.metrics
            ],
            "independent_source": self.independent_source,
            "passed": self.passed,
        }
        return {**record, "evidence_id": self.evidence_id} if include_id else record


@dataclass(frozen=True, slots=True)
class OmniphysicsQualificationEvidence:
    controls: tuple[NumericalControlEvidence, ...]
    refinements: tuple[RefinementCampaignEvidence, ...]
    providers: tuple[HardwareProviderEvidence, ...]
    applications: tuple[ApplicationValidationEvidence, ...]

    @property
    def evidence_id(self) -> str:
        return canonical_fingerprint(self.to_record(include_id=False))

    def to_record(self, *, include_id=True):
        record = {
            "kind": "omniphysics-qualification-evidence",
            "controls": [value.to_record() for value in self.controls],
            "refinements": [value.to_record() for value in self.refinements],
            "providers": [value.to_record() for value in self.providers],
            "applications": [value.to_record() for value in self.applications],
            "passed": all(value.passed for value in self.controls)
            and all(value.passed for value in self.refinements)
            and all(value.passed for value in self.applications),
            "distributed_qualified": any(
                value.qualifies_distributed for value in self.providers
            ),
        }
        return {**record, "evidence_id": self.evidence_id} if include_id else record


__all__ = [
    "ApplicationValidationEvidence",
    "HardwareProviderEvidence",
    "NumericalControlEvidence",
    "OmniphysicsQualificationEvidence",
    "RefinementCampaignEvidence",
]
