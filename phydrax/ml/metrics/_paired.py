#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule
from ._base import (
    _nan_where_invalid,
    _prepare_pair,
    _status,
    METRIC_UNDEFINED,
)


class PairedLossComparisonPlan(StrictModule):
    """Predeclared percentile-bootstrap recipe for paired additive losses."""

    confidence: float = eqx.field(static=True)
    resamples: int = eqx.field(static=True)
    noninferiority_margin: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        confidence: float = 0.95,
        resamples: int = 2_000,
        noninferiority_margin: float = 0.0,
    ):
        confidence_ = float(confidence)
        resamples_ = int(resamples)
        margin = float(noninferiority_margin)
        if not isfinite(confidence_) or not 0.0 < confidence_ < 1.0:
            raise ValueError("confidence must lie strictly between zero and one.")
        if resamples_ < 2:
            raise ValueError("resamples must be at least two.")
        if not isfinite(margin) or margin < 0.0:
            raise ValueError("noninferiority_margin must be finite and nonnegative.")
        self.confidence = confidence_
        self.resamples = resamples_
        self.noninferiority_margin = margin


class PairedLossComparisonResult(StrictModule):
    """Paired candidate-minus-reference effect and bootstrap uncertainty."""

    loss_difference: Array
    valid_mask: Array
    effect: Array
    bootstrap_effects: Array
    interval_lower: Array
    interval_upper: Array
    noninferiority_upper_bound: Array
    noninferior: Array
    effective_weight: Array
    independent_unit_count: Array
    valid: Array
    status: Array
    plan: PairedLossComparisonPlan
    grouped: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        loss_difference: ArrayLike,
        valid_mask: ArrayLike,
        effect: ArrayLike,
        bootstrap_effects: ArrayLike,
        interval_lower: ArrayLike,
        interval_upper: ArrayLike,
        noninferiority_upper_bound: ArrayLike,
        noninferior: ArrayLike,
        effective_weight: ArrayLike,
        independent_unit_count: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        plan: PairedLossComparisonPlan,
        grouped: bool,
    ):
        self.loss_difference = jnp.asarray(loss_difference)
        self.valid_mask = jnp.asarray(valid_mask, dtype=bool)
        self.effect = jnp.asarray(effect)
        self.bootstrap_effects = jnp.asarray(bootstrap_effects)
        self.interval_lower = jnp.asarray(interval_lower)
        self.interval_upper = jnp.asarray(interval_upper)
        self.noninferiority_upper_bound = jnp.asarray(noninferiority_upper_bound)
        self.noninferior = jnp.asarray(noninferior, dtype=bool)
        self.effective_weight = jnp.asarray(effective_weight)
        self.independent_unit_count = jnp.asarray(independent_unit_count, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.plan = plan
        self.grouped = bool(grouped)


def _case_units(
    difference: Array, weight: Array, support: Array, /
) -> tuple[Array, Array, Array]:
    order = jnp.argsort(jnp.where(support, 0, 1), stable=True)
    unit_weight = weight[order]
    unit_numerator = unit_weight * difference[order]
    unit_count = jnp.sum(support).astype(jnp.int32)
    return unit_numerator, unit_weight, unit_count


def _group_units(
    difference: Array,
    weight: Array,
    support: Array,
    groups: Array,
    /,
) -> tuple[Array, Array, Array]:
    capacity = difference.shape[0]
    if capacity == 0:
        empty = jnp.empty((0,), dtype=weight.dtype)
        return empty, empty, jnp.asarray(0, dtype=jnp.int32)
    support_key = jnp.where(support, 0, 1)
    order = jnp.lexsort((groups, support_key))
    sorted_support = support[order]
    sorted_group = groups[order]
    sorted_weight = jnp.where(sorted_support, weight[order], 0.0)
    sorted_difference = difference[order]
    group_start = jnp.concatenate(
        (
            jnp.ones((1,), dtype=bool),
            (support_key[order][1:] != support_key[order][:-1])
            | (sorted_group[1:] != sorted_group[:-1]),
        )
    )
    group_index = jnp.cumsum(group_start.astype(jnp.int32)) - 1
    group_numerator = jax.ops.segment_sum(
        sorted_weight * sorted_difference,
        group_index,
        num_segments=capacity,
        indices_are_sorted=True,
    )
    group_weight = jax.ops.segment_sum(
        sorted_weight,
        group_index,
        num_segments=capacity,
        indices_are_sorted=True,
    )
    group_count = jnp.sum(group_weight > 0.0).astype(jnp.int32)
    return group_numerator, group_weight, group_count


def _bootstrap_effects(
    unit_numerator: Array,
    unit_weight: Array,
    unit_count: Array,
    key: Key[Array, ""],
    resamples: int,
    /,
) -> Array:
    capacity = unit_weight.shape[0]
    draw_position = jnp.arange(capacity)
    safe_unit_count = jnp.maximum(unit_count, 1)

    def one_resample(_, replicate):
        replicate_key = jr.fold_in(key, replicate)
        sampled = jr.randint(
            replicate_key,
            (capacity,),
            minval=0,
            maxval=safe_unit_count,
            dtype=jnp.int32,
        )
        drawn = draw_position < unit_count
        numerator = jnp.sum(jnp.where(drawn, unit_numerator[sampled], 0.0))
        denominator = jnp.sum(jnp.where(drawn, unit_weight[sampled], 0.0))
        effect = numerator / jnp.where(denominator > 0.0, denominator, 1.0)
        return None, effect

    _, effects = jax.lax.scan(
        one_resample,
        None,
        jnp.arange(resamples, dtype=jnp.uint32),
    )
    return effects


def compare_paired_losses(
    reference_loss: ArrayLike,
    candidate_loss: ArrayLike,
    /,
    *,
    key: Key[Array, ""],
    plan: PairedLossComparisonPlan,
    sample_weight: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    groups: ArrayLike | None = None,
) -> PairedLossComparisonResult:
    """Compare aligned additive losses from two frozen methods.

    The effect is the weighted candidate loss minus reference loss, so lower is
    better. Bootstrap draws are paired by construction. Supplying ``groups``
    forces whole-group resampling; group weighted numerators and denominators are
    preaggregated before drawing. The returned central percentile interval is
    descriptive, while noninferiority uses its distinct one-sided upper bound.
    """
    if not isinstance(plan, PairedLossComparisonPlan):
        raise TypeError("plan must be a PairedLossComparisonPlan.")
    jr.key_data(key)
    reference = jnp.asarray(reference_loss)
    candidate = jnp.asarray(candidate_loss)
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("paired losses must be aligned one-dimensional arrays.")
    reference_, candidate_, weights, active, invalid, _ = _prepare_pair(
        reference,
        candidate,
        sample_weight=sample_weight,
        mask=mask,
        sample_axis=-1,
        metric="compare_paired_losses",
        allow_complex=False,
    )
    invalid = invalid | jnp.any(active & ((reference_ < 0.0) | (candidate_ < 0.0)))
    difference = candidate_ - reference_
    weights = jnp.where(active, weights, 0.0)
    support = active & (weights > 0.0)
    effective_weight = jnp.sum(weights)
    observed_effect = jnp.sum(weights * difference) / jnp.where(
        effective_weight > 0.0, effective_weight, 1.0
    )

    grouped = groups is not None
    if grouped:
        groups_ = jnp.asarray(groups)
        if groups_.shape != reference.shape:
            raise ValueError("groups must have the same shape as the loss arrays.")
        if not jnp.issubdtype(groups_.dtype, jnp.integer):
            raise TypeError("groups must use an integer dtype.")
        unit_numerator, unit_weight, unit_count = _group_units(
            difference,
            weights,
            support,
            groups_,
        )
    else:
        unit_numerator, unit_weight, unit_count = _case_units(
            difference, weights, support
        )

    bootstrap_effects = _bootstrap_effects(
        unit_numerator,
        unit_weight,
        unit_count,
        key,
        plan.resamples,
    )
    tail = 0.5 * (1.0 - plan.confidence)
    interval_lower = jnp.quantile(bootstrap_effects, tail)
    interval_upper = jnp.quantile(bootstrap_effects, 1.0 - tail)
    noninferiority_upper_bound = jnp.quantile(bootstrap_effects, plan.confidence)

    valid, status = _status(
        invalid=invalid,
        empty=effective_weight <= 0.0,
        undefined=unit_count < 2,
        undefined_status=METRIC_UNDEFINED,
    )
    observed_effect = _nan_where_invalid(observed_effect, valid)
    bootstrap_effects = jnp.where(valid, bootstrap_effects, jnp.nan)
    interval_lower = _nan_where_invalid(interval_lower, valid)
    interval_upper = _nan_where_invalid(interval_upper, valid)
    noninferiority_upper_bound = _nan_where_invalid(noninferiority_upper_bound, valid)
    noninferior = valid & (noninferiority_upper_bound <= plan.noninferiority_margin)
    return PairedLossComparisonResult(
        loss_difference=jnp.where(support, difference, 0.0),
        valid_mask=support,
        effect=observed_effect,
        bootstrap_effects=bootstrap_effects,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        noninferiority_upper_bound=noninferiority_upper_bound,
        noninferior=noninferior,
        effective_weight=effective_weight,
        independent_unit_count=unit_count,
        valid=valid,
        status=status,
        plan=plan,
        grouped=grouped,
    )


__all__ = [
    "PairedLossComparisonPlan",
    "PairedLossComparisonResult",
    "compare_paired_losses",
]
