#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..._fingerprint import canonical_fingerprint
from ...artifacts import DifferentiationContract


class DerivativeTier(str, Enum):
    """Strict design-derivative evidence tiers.

    ``Q0`` certifies values only. ``Q1`` is local fixed-trace automatic
    differentiation, ``Q2`` a converged smooth fixed-branch implicit derivative,
    ``Q3`` a named generalized or semismooth derivative, ``Q4`` a discrete shape
    derivative, and ``Q5`` agreement between continuous and discrete shape
    derivatives. ``Q6`` records a discrete plan transition and is deliberately not
    a derivative tier.
    """

    Q0 = "Q0"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    Q5 = "Q5"
    Q6 = "Q6"

    @property
    def is_derivative(self) -> bool:
        return self not in (DerivativeTier.Q0, DerivativeTier.Q6)


@dataclass(frozen=True, slots=True)
class DesignQualificationEvidence:
    """Content-addressed evidence for one design derivative qualification tier."""

    tier: DerivativeTier
    analysis_plan_id: str
    numeric_revision_id: str
    execution_plan_id: str
    primal_residual: float | None = None
    adjoint_residual: float | None = None
    condition_estimate: float | None = None
    event_margins: tuple[float, ...] = ()
    event_ids: tuple[str, ...] = ()
    gradient_error: float | None = None
    taylor_slope: float | None = None
    shape_error: float | None = None
    transfer_plan_id: str | None = None
    transition_event_id: str | None = None
    valid: bool = False
    diagnostic_ids: tuple[str, ...] = ()
    evidence_id: str = field(init=False)

    def __post_init__(self):
        tier = (
            self.tier
            if isinstance(self.tier, DerivativeTier)
            else DerivativeTier(self.tier)
        )
        object.__setattr__(self, "tier", tier)
        identifiers = (
            self.analysis_plan_id,
            self.numeric_revision_id,
            self.execution_plan_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError(
                "Analysis-plan, numeric-revision, and execution-plan IDs "
                "must be non-empty strings."
            )

        primal_residual = _nonnegative(self.primal_residual, "primal_residual")
        adjoint_residual = _nonnegative(self.adjoint_residual, "adjoint_residual")
        condition_estimate = _nonnegative(self.condition_estimate, "condition_estimate")
        gradient_error = _nonnegative(self.gradient_error, "gradient_error")
        shape_error = _nonnegative(self.shape_error, "shape_error")
        object.__setattr__(self, "primal_residual", primal_residual)
        object.__setattr__(self, "adjoint_residual", adjoint_residual)
        object.__setattr__(self, "condition_estimate", condition_estimate)
        object.__setattr__(self, "gradient_error", gradient_error)
        object.__setattr__(self, "shape_error", shape_error)

        if self.taylor_slope is not None:
            slope = float(self.taylor_slope)
            if not math.isfinite(slope):
                raise ValueError("taylor_slope must be finite.")
            object.__setattr__(self, "taylor_slope", slope)

        margins = tuple(float(value) for value in self.event_margins)
        if any(not math.isfinite(value) for value in margins):
            raise ValueError("event_margins must be finite.")
        event_ids = _identifiers(self.event_ids, "event_ids")
        diagnostics = _identifiers(self.diagnostic_ids, "diagnostic_ids")
        if margins and len(margins) != len(event_ids):
            raise ValueError("event_margins and event_ids must have equal lengths.")
        object.__setattr__(self, "event_margins", margins)
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "diagnostic_ids", diagnostics)

        transfer_id = _optional_identifier(self.transfer_plan_id, "transfer_plan_id")
        transition_id = _optional_identifier(
            self.transition_event_id, "transition_event_id"
        )
        object.__setattr__(self, "transfer_plan_id", transfer_id)
        object.__setattr__(self, "transition_event_id", transition_id)
        object.__setattr__(self, "valid", bool(self.valid))

        if tier is DerivativeTier.Q0 and any(
            value is not None
            for value in (
                self.adjoint_residual,
                self.gradient_error,
                self.taylor_slope,
                self.shape_error,
                transfer_id,
                transition_id,
            )
        ):
            raise ValueError("Q0 is value-only evidence.")
        if tier is DerivativeTier.Q6:
            if transfer_id is None or transition_id is None:
                raise ValueError("Q6 requires transfer_plan_id and transition_event_id.")
            if any(
                value is not None
                for value in (
                    self.adjoint_residual,
                    self.gradient_error,
                    self.taylor_slope,
                    self.shape_error,
                )
            ):
                raise ValueError("Q6 records a transition, not derivative evidence.")
        elif transfer_id is not None or transition_id is not None:
            raise ValueError("Transition identifiers are reserved for Q6 evidence.")

        payload = {
            "kind": "design-qualification-evidence",
            "tier": tier.value,
            "analysis_plan_id": self.analysis_plan_id,
            "numeric_revision_id": self.numeric_revision_id,
            "execution_plan_id": self.execution_plan_id,
            "primal_residual": primal_residual,
            "adjoint_residual": adjoint_residual,
            "condition_estimate": condition_estimate,
            "event_margins": list(margins),
            "event_ids": list(event_ids),
            "gradient_error": gradient_error,
            "taylor_slope": self.taylor_slope,
            "shape_error": shape_error,
            "transfer_plan_id": transfer_id,
            "transition_event_id": transition_id,
            "valid": self.valid,
            "diagnostic_ids": list(diagnostics),
        }
        object.__setattr__(self, "evidence_id", canonical_fingerprint(payload))

    def coarse_contract_allows(self, contract: DifferentiationContract, /) -> bool:
        """Check only the necessary coarse capability; this never replaces evidence."""
        if not isinstance(contract, DifferentiationContract):
            raise TypeError("contract must be a DifferentiationContract.")
        if not self.tier.is_derivative:
            return True
        return bool(
            contract.upstream_physical_parameters
            or contract.stored_values
            or contract.query_coordinates
            or contract.local_parameters
            or contract.stochastic_realization
        )


def _nonnegative(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return numeric


def _identifiers(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    identifiers = tuple(values)
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must not contain duplicates.")
    return identifiers


def _optional_identifier(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string or None.")
    return value


__all__ = ["DerivativeTier", "DesignQualificationEvidence"]
