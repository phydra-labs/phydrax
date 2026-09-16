#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class TheoryVariationMode(StrEnum):
    REPLICA = "replica"
    HESSIAN = "hessian"
    ENVELOPE = "envelope"


class TheoryPrediction(StrictModule, NonTrainableState):
    values: Array
    covariance: Array
    valid: Array
    observable_names: tuple[str, ...] = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    process_plan_id: str = eqx.field(static=True)
    prediction_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        observable_names: Sequence[str],
        unit_id: str,
        process_plan_id: str,
    ):
        values_ = np.asarray(values, dtype=float)
        covariance_ = np.asarray(covariance, dtype=float)
        names = tuple(str(value).strip() for value in observable_names)
        if (
            values_.ndim != 1
            or values_.size < 1
            or covariance_.shape != (values_.size, values_.size)
            or len(names) != values_.size
        ):
            raise ValueError(
                "Theory prediction values, covariance, and names do not align."
            )
        if (
            any(not value for value in names)
            or len(set(names)) != len(names)
            or np.any(~np.isfinite(values_))
            or np.any(~np.isfinite(covariance_))
            or not np.allclose(covariance_, covariance_.T)
        ):
            raise ValueError("Theory prediction values/covariance/names are invalid.")
        valid_ = (
            np.ones(values_.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != values_.shape:
            raise ValueError("valid must align with prediction values.")
        unit = str(unit_id).strip()
        plan = str(process_plan_id).strip()
        if not unit or not plan:
            raise ValueError("Prediction unit and process plan identity are required.")
        self.values = jnp.asarray(values_)
        self.covariance = jnp.asarray(covariance_)
        self.valid = jnp.asarray(valid_)
        self.observable_names = names
        self.unit_id = unit
        self.process_plan_id = plan
        self.prediction_id = canonical_fingerprint(
            {
                "kind": "collider-theory-prediction",
                "arrays": array_tree_fingerprint((values_, covariance_, valid_)),
                "observables": list(names),
                "unit": unit,
                "process": plan,
            }
        )


def covariance_from_variations(
    nominal: ArrayLike,
    variations: ArrayLike,
    mode: TheoryVariationMode,
    /,
) -> Array:
    """Construct an explicitly declared theory covariance from provider variations."""
    nominal_ = jnp.asarray(nominal)
    variations_ = jnp.asarray(variations, dtype=nominal_.dtype)
    if (
        nominal_.ndim != 1
        or variations_.ndim != 2
        or variations_.shape[1] != nominal_.shape[0]
    ):
        raise ValueError("Theory variations must have shape (variation, observable).")
    if not isinstance(mode, TheoryVariationMode):
        raise TypeError("mode must be TheoryVariationMode.")
    delta = variations_ - nominal_[None, :]
    if mode is TheoryVariationMode.REPLICA:
        centered = variations_ - jnp.mean(variations_, axis=0, keepdims=True)
        denominator = jnp.maximum(variations_.shape[0] - 1, 1)
        return ein.contract("vi,vj->ij", centered, centered) / denominator
    if mode is TheoryVariationMode.HESSIAN:
        if variations_.shape[0] % 2 != 0:
            raise ValueError("Hessian variations require plus/minus pairs.")
        pairs = delta.reshape((-1, 2, nominal_.shape[0]))
        difference = 0.5 * (pairs[:, 0] - pairs[:, 1])
        return ein.contract("vi,vj->ij", difference, difference)
    envelope = jnp.max(jnp.abs(delta), axis=0)
    return envelope[:, None] * envelope[None, :]


class EFTMorphingPlan(StrictModule, NonTrainableState):
    base: Array
    linear: Array
    quadratic: Array
    coefficient_names: tuple[str, ...] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    support_lower: Array
    support_upper: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: ArrayLike,
        linear: ArrayLike,
        quadratic: ArrayLike,
        support_lower: ArrayLike,
        support_upper: ArrayLike,
        /,
        *,
        coefficient_names: Sequence[str],
        observable_names: Sequence[str],
        source_prediction_id: str,
    ):
        base_ = np.asarray(base, dtype=float)
        linear_ = np.asarray(linear, dtype=float)
        quadratic_ = np.asarray(quadratic, dtype=float)
        lower = np.asarray(support_lower, dtype=float)
        upper = np.asarray(support_upper, dtype=float)
        coefficients = tuple(str(value).strip() for value in coefficient_names)
        observables = tuple(str(value).strip() for value in observable_names)
        coefficient_count = len(coefficients)
        if (
            base_.ndim != 1
            or linear_.shape != (coefficient_count, base_.size)
            or quadratic_.shape != (coefficient_count, coefficient_count, base_.size)
            or lower.shape != (coefficient_count,)
            or upper.shape != lower.shape
        ):
            raise ValueError(
                "EFT morphing arrays do not align with coefficients/observables."
            )
        if (
            len(observables) != base_.size
            or any(not value for value in coefficients + observables)
            or len(set(coefficients)) != coefficient_count
            or np.any(~np.isfinite(base_))
            or np.any(~np.isfinite(linear_))
            or np.any(~np.isfinite(quadratic_))
            or np.any(lower >= upper)
        ):
            raise ValueError("EFT morphing content, names, or support is invalid.")
        source = str(source_prediction_id).strip()
        if not source:
            raise ValueError("source_prediction_id is required.")
        self.base = jnp.asarray(base_)
        self.linear = jnp.asarray(linear_)
        self.quadratic = jnp.asarray(quadratic_)
        self.coefficient_names = coefficients
        self.observable_names = observables
        self.support_lower = jnp.asarray(lower)
        self.support_upper = jnp.asarray(upper)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "eft-quadratic-morphing-plan",
                "arrays": array_tree_fingerprint(
                    (base_, linear_, quadratic_, lower, upper)
                ),
                "coefficients": list(coefficients),
                "observables": list(observables),
                "source": source,
            }
        )


class EFTMorphingResult(StrictModule, NonTrainableState):
    values: Array
    in_support: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def evaluate_eft_morphing(
    plan: EFTMorphingPlan, coefficients: ArrayLike, /
) -> EFTMorphingResult:
    if not isinstance(plan, EFTMorphingPlan):
        raise TypeError("plan must be EFTMorphingPlan.")
    values = jnp.asarray(coefficients, dtype=plan.base.dtype)
    if values.shape != plan.support_lower.shape:
        raise ValueError("coefficients must align with EFT support.")
    prediction = (
        plan.base
        + ein.contract("i,io->o", values, plan.linear)
        + ein.contract("i,j,ijo->o", values, values, plan.quadratic)
    )
    in_support = jnp.all((values >= plan.support_lower) & (values <= plan.support_upper))
    finite = jnp.all(jnp.isfinite(prediction))
    return EFTMorphingResult(
        jnp.where(in_support, prediction, jnp.nan),
        in_support,
        finite,
        in_support & finite,
        plan.plan_id,
    )


__all__ = [
    "EFTMorphingPlan",
    "EFTMorphingResult",
    "TheoryPrediction",
    "TheoryVariationMode",
    "covariance_from_variations",
    "evaluate_eft_morphing",
]
