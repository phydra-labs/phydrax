#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared result and evidence records for explicit valuation routes."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..core._currency import Currency
from ..core._evidence import FinanceEvidenceBinding
from ..core._laws import PricingLaw


class ValuationStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    ARBITRAGE_BOUND_VIOLATION = 2
    INVALID_MOMENT = 3
    NONFINITE_RESULT = 4
    NOT_CONVERGED = 5
    SINGULAR_SYSTEM = 6
    UNSUPPORTED_ROUTE = 7
    INVALID_PATHS = 8
    REPLAY_MISMATCH = 9


class ValuationEvidence(StrictModule):
    """Separated admissibility, numerical and use-evidence for one route."""

    finite: Array
    inputs_admissible: Array
    converged: Array
    exercise_consistent: Array
    error_estimate: Array | None
    residual_norm: Array | None
    binding: FinanceEvidenceBinding | None
    route: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    status: ValuationStatus = eqx.field(static=True)

    def __init__(
        self,
        *,
        route: str,
        finite: ArrayLike,
        inputs_admissible: ArrayLike = True,
        converged: ArrayLike = True,
        exercise_consistent: ArrayLike = True,
        error_estimate: ArrayLike | None = None,
        residual_norm: ArrayLike | None = None,
        binding: FinanceEvidenceBinding | None = None,
        pricing_law: PricingLaw | None = None,
        status: ValuationStatus = ValuationStatus.SUCCESS,
    ):
        if not isinstance(route, str) or not route:
            raise ValueError("route must be a non-empty string.")
        if binding is not None and not isinstance(binding, FinanceEvidenceBinding):
            raise TypeError("binding must be FinanceEvidenceBinding or None.")
        if pricing_law is not None and not isinstance(pricing_law, PricingLaw):
            raise TypeError("pricing_law must be PricingLaw or None.")
        if not isinstance(status, ValuationStatus):
            raise TypeError("status must be a ValuationStatus.")
        self.finite = jnp.asarray(finite, dtype=bool)
        self.inputs_admissible = jnp.asarray(inputs_admissible, dtype=bool)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.exercise_consistent = jnp.asarray(exercise_consistent, dtype=bool)
        self.error_estimate = (
            None if error_estimate is None else jnp.asarray(error_estimate)
        )
        self.residual_norm = None if residual_norm is None else jnp.asarray(residual_norm)
        self.binding = binding
        self.route = route
        self.law_id = "" if pricing_law is None else pricing_law.law_id
        self.status = status

    @property
    def successful(self) -> Array:
        return (
            (self.status is ValuationStatus.SUCCESS)
            & self.finite
            & self.inputs_admissible
            & self.converged
            & self.exercise_consistent
        )


class ValuationResult(StrictModule):
    """Monetary major-unit value and route-specific numerical evidence."""

    value: Array
    lower_bound: Array | None
    upper_bound: Array | None
    standard_error: Array | None
    currency: Currency | None = eqx.field(static=True)
    evidence: ValuationEvidence
    diagnostics: Any

    def __init__(
        self,
        value: ArrayLike,
        evidence: ValuationEvidence,
        /,
        *,
        lower_bound: ArrayLike | None = None,
        upper_bound: ArrayLike | None = None,
        standard_error: ArrayLike | None = None,
        currency: Currency | None = None,
        diagnostics: Any = None,
    ):
        if not isinstance(evidence, ValuationEvidence):
            raise TypeError("evidence must be ValuationEvidence.")
        if currency is not None and not isinstance(currency, Currency):
            raise TypeError("currency must be Currency or None.")
        value_ = jnp.asarray(value)
        if jnp.iscomplexobj(value_):
            raise TypeError("valuation values must be real.")
        lower = (
            None if lower_bound is None else jnp.asarray(lower_bound, dtype=value_.dtype)
        )
        upper = (
            None if upper_bound is None else jnp.asarray(upper_bound, dtype=value_.dtype)
        )
        if (lower is None) != (upper is None):
            raise ValueError("lower_bound and upper_bound must be supplied together.")
        if lower is not None:
            value_ = eqx.error_if(
                value_,
                jnp.any(lower > upper)
                | jnp.any(value_ < lower - 1.0e-7)
                | jnp.any(value_ > upper + 1.0e-7),
                "valuation lies outside its declared no-arbitrage bounds.",
            )
        self.value = value_
        self.lower_bound = lower
        self.upper_bound = upper
        self.standard_error = (
            None
            if standard_error is None
            else jnp.asarray(standard_error, dtype=value_.dtype)
        )
        self.currency = currency
        self.evidence = evidence
        self.diagnostics = diagnostics

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class ImpliedVolatilityResult(StrictModule):
    volatility: Array
    residual: Array
    lower_bracket: Array
    upper_bracket: Array
    iterations: Array
    converged: Array
    model: str = eqx.field(static=True)


class ValuationReplayEvidence(StrictModule):
    expected_value: Array
    replayed_value: Array
    absolute_difference: Array
    tolerance: Array
    matches: Array
    route: str = eqx.field(static=True)

    def __init__(
        self,
        expected_value: ArrayLike,
        replayed_value: ArrayLike,
        tolerance: ArrayLike,
        /,
        *,
        route: str,
    ):
        expected = jnp.asarray(expected_value)
        replayed = jnp.asarray(replayed_value, dtype=expected.dtype)
        tolerance_ = jnp.asarray(tolerance, dtype=expected.dtype)
        if expected.shape != replayed.shape or tolerance_.shape != ():
            raise ValueError("replay values must align and tolerance must be scalar.")
        tolerance_ = eqx.error_if(
            tolerance_,
            ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
            "replay tolerance must be finite and non-negative.",
        )
        difference = jnp.abs(replayed - expected)
        self.expected_value = expected
        self.replayed_value = replayed
        self.absolute_difference = difference
        self.tolerance = tolerance_
        self.matches = jnp.all(jnp.isfinite(difference) & (difference <= tolerance_))
        self.route = route


__all__ = [
    "ImpliedVolatilityResult",
    "ValuationEvidence",
    "ValuationReplayEvidence",
    "ValuationResult",
    "ValuationStatus",
]
