#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    determinant_small_linear,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ._state import make_z4c_state, Z4cState


_SMALL_3X3 = SmallLinearSolvePlan(3)


def _trailing_matrix(value: Array, /) -> Array:
    return jnp.moveaxis(value, (0, 1), (-2, -1))


def _component_matrix(value: Array, /) -> Array:
    return jnp.moveaxis(value, (-2, -1), (0, 1))


class Z4cEnforcementEvidence(StrictModule):
    determinant_defect_before: Array
    determinant_defect_after: Array
    trace_defect_before: Array
    trace_defect_after: Array
    correction_norm: Array
    applied: Array
    finite: Array
    successful: Array


class Z4cEnforcementResult(StrictModule):
    state: Z4cState
    evidence: Z4cEnforcementEvidence


class Z4cAlgebraicEnforcement(StrictModule, NonTrainableState):
    """Accepted-step projection to det(gamma-tilde)=1 and tr(A-tilde)=0."""

    tolerance: float = eqx.field(static=True)
    maximum_correction: float = eqx.field(static=True)
    enforcement_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_correction: float = float("inf"),
    ):
        tolerance_ = float(tolerance)
        maximum = float(maximum_correction)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        if not (isfinite(maximum) or maximum == float("inf")) or maximum <= 0.0:
            raise ValueError("maximum_correction must be positive and not NaN.")
        self.tolerance = tolerance_
        self.maximum_correction = maximum
        self.enforcement_id = canonical_fingerprint(
            {
                "kind": "z4c-accepted-step-algebraic-enforcement",
                "tolerance": tolerance_,
                "maximum_correction": maximum if isfinite(maximum) else None,
            }
        )

    def apply(self, state: Z4cState, /) -> Z4cEnforcementResult:
        if not isinstance(state, Z4cState):
            raise TypeError("state must be a Z4cState.")
        metric_trailing = _trailing_matrix(state.conformal_metric)
        extrinsic = state.conformal_extrinsic_curvature
        determinant = determinant_small_linear(_SMALL_3X3, metric_trailing)
        safe_determinant = jnp.where(determinant > 0.0, determinant, 1.0)
        normalized_metric = (
            state.conformal_metric / jnp.cbrt(safe_determinant)[None, None, ...]
        )
        inverse_result = inverse_small_linear(
            _SMALL_3X3, _trailing_matrix(normalized_metric)
        )
        inverse = _component_matrix(inverse_result.value)
        trace_before = ein.contract("ij...,ij...->...", inverse, extrinsic, backend="jax")
        normalized_extrinsic = (
            extrinsic - normalized_metric * trace_before[None, None, ...] / 3.0
        )
        trace_after = ein.contract(
            "ij...,ij...->...", inverse, normalized_extrinsic, backend="jax"
        )
        determinant_after = determinant_small_linear(
            _SMALL_3X3, _trailing_matrix(normalized_metric)
        )
        metric_correction = jnp.max(jnp.abs(normalized_metric - state.conformal_metric))
        extrinsic_correction = jnp.max(jnp.abs(normalized_extrinsic - extrinsic))
        correction = jnp.maximum(metric_correction, extrinsic_correction)
        determinant_defect_before = jnp.max(jnp.abs(determinant - 1.0))
        determinant_defect_after = jnp.max(jnp.abs(determinant_after - 1.0))
        trace_defect_before = jnp.max(jnp.abs(trace_before))
        trace_defect_after = jnp.max(jnp.abs(trace_after))
        finite = (
            jnp.all(jnp.isfinite(normalized_metric))
            & jnp.all(jnp.isfinite(normalized_extrinsic))
            & jnp.isfinite(correction)
        )
        successful = (
            finite
            & jnp.all(determinant > 0.0)
            & jnp.all(inverse_result.successful)
            & (correction <= self.maximum_correction)
        )
        applied = successful & (
            (determinant_defect_before > self.tolerance)
            | (trace_defect_before > self.tolerance)
        )
        proposed = make_z4c_state(
            state.chi,
            normalized_metric,
            state.k_hat,
            normalized_extrinsic,
            state.theta,
            state.conformal_connection,
            state.lapse,
            state.shift,
            state.shift_driver,
            grid_id=state.grid_id,
        )
        accepted_values = jnp.where(successful, proposed.values, state.values)
        evidence = Z4cEnforcementEvidence(
            determinant_defect_before,
            determinant_defect_after,
            trace_defect_before,
            trace_defect_after,
            correction,
            applied,
            finite,
            successful,
        )
        return Z4cEnforcementResult(state.with_values(accepted_values), evidence)


__all__ = [
    "Z4cAlgebraicEnforcement",
    "Z4cEnforcementEvidence",
    "Z4cEnforcementResult",
]
