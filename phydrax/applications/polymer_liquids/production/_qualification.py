#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._support import decide_polymer_production_regime, PolymerProductionRegime


class PolymerQualificationCase(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    reference: Array
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    required: bool = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reference: ArrayLike,
        /,
        *,
        absolute_tolerance: float,
        relative_tolerance: float = 0.0,
        required: bool = True,
    ):
        identifier = str(name)
        value = np.asarray(reference)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if not identifier or identifier != identifier.strip():
            raise ValueError("Qualification case name must be canonical and nonempty.")
        if value.size == 0 or np.any(~np.isfinite(value)):
            raise ValueError("Qualification reference must be finite and nonempty.")
        if (
            not math.isfinite(absolute)
            or absolute < 0.0
            or not math.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Qualification tolerances must be finite and nonnegative.")
        self.name = identifier
        self.reference = jnp.asarray(value)
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.required = bool(required)
        self.case_id = canonical_fingerprint(
            {
                "kind": "polymer-qualification-case",
                "name": identifier,
                "reference": array_tree_fingerprint(value),
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
                "required": self.required,
            }
        )


class PolymerQualificationCampaignPlan(StrictModule, NonTrainableState):
    regime: PolymerProductionRegime
    cases: tuple[PolymerQualificationCase, ...]
    campaign_id: str = eqx.field(static=True)

    def __init__(
        self,
        regime: PolymerProductionRegime,
        cases: Sequence[PolymerQualificationCase],
        /,
    ):
        decision = decide_polymer_production_regime(regime)
        decision.require_supported()
        values = tuple(cases)
        if not values or any(
            not isinstance(case, PolymerQualificationCase) for case in values
        ):
            raise TypeError("Qualification campaign requires typed nonempty cases.")
        names = tuple(case.name for case in values)
        if len(set(names)) != len(names):
            raise ValueError("Qualification case names must be unique.")
        if not any(case.required for case in values):
            raise ValueError("Qualification campaign must contain a required gate.")
        self.regime = regime
        self.cases = values
        self.campaign_id = canonical_fingerprint(
            {
                "kind": "polymer-qualification-campaign",
                "regime": regime.regime_id,
                "cases": [case.case_id for case in values],
            }
        )


class PolymerQualificationResult(StrictModule, NonTrainableState):
    maximum_absolute_residuals: Array
    tolerances: Array
    passed: Array
    required: Array
    finite: Array
    qualified: Array
    case_names: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)


def evaluate_polymer_qualification(
    plan: PolymerQualificationCampaignPlan,
    observations: Mapping[str, ArrayLike],
    /,
) -> PolymerQualificationResult:
    if not isinstance(plan, PolymerQualificationCampaignPlan):
        raise TypeError("plan must be PolymerQualificationCampaignPlan.")
    if set(observations) != {case.name for case in plan.cases}:
        raise ValueError("Qualification observations must match campaign cases exactly.")
    residuals = []
    tolerances = []
    passed = []
    finite_values = []
    observed_host: dict[str, np.ndarray] = {}
    for case in plan.cases:
        observed = jnp.asarray(observations[case.name], dtype=case.reference.dtype)
        if observed.shape != case.reference.shape:
            raise ValueError(
                f"Qualification observation {case.name!r} has the wrong shape."
            )
        residual = jnp.max(jnp.abs(observed - case.reference))
        scale = jnp.max(jnp.abs(case.reference))
        tolerance = case.absolute_tolerance + case.relative_tolerance * scale
        finite = jnp.all(jnp.isfinite(observed)) & jnp.isfinite(residual)
        residuals.append(residual)
        tolerances.append(tolerance)
        passed.append(finite & (residual <= tolerance))
        finite_values.append(finite)
        observed_host[case.name] = np.asarray(observed)
    residual_array = jnp.stack(residuals)
    tolerance_array = jnp.asarray(tolerances, dtype=residual_array.dtype)
    passed_array = jnp.stack(passed)
    required = jnp.asarray([case.required for case in plan.cases])
    finite_array = jnp.stack(finite_values)
    qualified = jnp.all(jnp.where(required, passed_array, True))
    evidence_id = canonical_fingerprint(
        {
            "kind": "polymer-qualification-evidence",
            "campaign": plan.campaign_id,
            "observations": array_tree_fingerprint(observed_host),
            "passed": np.asarray(passed_array),
        }
    )
    return PolymerQualificationResult(
        residual_array,
        tolerance_array,
        passed_array,
        required,
        finite_array,
        qualified,
        tuple(case.name for case in plan.cases),
        evidence_id,
        plan.campaign_id,
    )


def default_polymer_qualification_cases() -> tuple[PolymerQualificationCase, ...]:
    return (
        PolymerQualificationCase(
            "rpy-symmetry-residual", 0.0, absolute_tolerance=1.0e-10
        ),
        PolymerQualificationCase(
            "fdt-relative-covariance-error", 0.0, absolute_tolerance=0.1
        ),
        PolymerQualificationCase(
            "ppa-oracle-relative-ne-error", 0.0, absolute_tolerance=0.15
        ),
        PolymerQualificationCase("reptation-time-exponent", 3.0, absolute_tolerance=0.25),
        PolymerQualificationCase(
            "zero-flow-invariance-residual", 0.0, absolute_tolerance=1.0e-10
        ),
        PolymerQualificationCase(
            "driven-work-balance-residual", 0.0, absolute_tolerance=1.0e-6
        ),
        PolymerQualificationCase(
            "kr-recurrence-residual", 0.0, absolute_tolerance=1.0e-10
        ),
    )


__all__ = [
    "PolymerQualificationCampaignPlan",
    "PolymerQualificationCase",
    "PolymerQualificationResult",
    "default_polymer_qualification_cases",
    "evaluate_polymer_qualification",
]
