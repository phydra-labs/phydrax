#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


MultipleTestingMethod: TypeAlias = Literal["bonferroni", "holm", "benjamini-hochberg"]
MultipleTestingStatus: TypeAlias = Literal[0, 1]

MULTIPLE_TESTING_SUCCESS = 0
MULTIPLE_TESTING_EMPTY_FAMILY = 1


class MultipleTestingPlan(StrictModule):
    """A finite, predeclared hypothesis-family correction plan."""

    alpha: float = eqx.field(static=True)
    method: MultipleTestingMethod = eqx.field(static=True)

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        method: MultipleTestingMethod = "holm",
    ):
        level = float(alpha)
        if not math.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("alpha must be finite and strictly between zero and one.")
        if method not in ("bonferroni", "holm", "benjamini-hochberg"):
            raise ValueError(
                "method must be 'bonferroni', 'holm', or 'benjamini-hochberg'."
            )
        self.alpha = level
        self.method = method


class MultipleTestingResult(StrictModule):
    """Adjusted probabilities and selections with the complete family retained."""

    raw_p_values: Array
    adjusted_p_values: Array
    family_mask: Array
    rejected: Array
    ordered_indices: Array
    ordered_raw_p_values: Array
    ordered_adjusted_p_values: Array
    critical_values: Array
    family_size: Array
    status: Array
    alpha: float = eqx.field(static=True)
    method: MultipleTestingMethod = eqx.field(static=True)
    axis: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == MULTIPLE_TESTING_SUCCESS


def _inputs(
    p_values: ArrayLike,
    mask: ArrayLike | None,
    axis: int,
) -> tuple[Array, Array, int]:
    values = jnp.asarray(p_values)
    if values.ndim < 1:
        raise ValueError("p_values must have at least one family axis.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += values.ndim
    if not 0 <= resolved_axis < values.ndim:
        raise ValueError("axis is out of bounds for p_values.")
    family_mask = (
        jnp.ones(values.shape, dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if family_mask.shape != values.shape:
        raise ValueError("mask must have the same shape as p_values.")
    invalid = family_mask & (~jnp.isfinite(values) | (values < 0.0) | (values > 1.0))
    values = eqx.error_if(
        values,
        jnp.any(invalid),
        "active p-values must be finite and lie in [0, 1].",
    )
    return (
        jnp.moveaxis(values, resolved_axis, -1),
        jnp.moveaxis(family_mask, resolved_axis, -1),
        resolved_axis,
    )


def _ordered(values: Array, mask: Array) -> tuple[Array, Array, Array, Array]:
    safe = jnp.where(mask, values, jnp.inf)
    order = jnp.argsort(safe, axis=-1, stable=True)
    ordered_values = jnp.take_along_axis(safe, order, axis=-1)
    ordered_mask = jnp.take_along_axis(mask, order, axis=-1)
    family_size = jnp.sum(mask, axis=-1).astype(jnp.int32)
    return order, ordered_values, ordered_mask, family_size


def _adjust_ordered(
    ordered: Array,
    ordered_mask: Array,
    family_size: Array,
    method: MultipleTestingMethod,
) -> tuple[Array, Array]:
    width = ordered.shape[-1]
    index = jnp.arange(width, dtype=ordered.dtype)
    count = family_size[..., None].astype(ordered.dtype)
    if method == "bonferroni":
        raw_adjusted = ordered * count
        adjusted = raw_adjusted
        critical = jnp.broadcast_to(1.0 / jnp.maximum(count, 1.0), ordered.shape)
    elif method == "holm":
        multiplier = jnp.maximum(count - index, 1.0)
        raw_adjusted = ordered * multiplier
        adjusted = jnp.maximum.accumulate(raw_adjusted, axis=-1)
        critical = 1.0 / multiplier
    else:
        rank = index + 1.0
        raw_adjusted = ordered * count / rank
        reversed_minimum = jnp.minimum.accumulate(raw_adjusted[..., ::-1], axis=-1)
        adjusted = reversed_minimum[..., ::-1]
        critical = rank / jnp.maximum(count, 1.0)
    adjusted = jnp.where(ordered_mask, jnp.clip(adjusted, 0.0, 1.0), 1.0)
    critical = jnp.where(ordered_mask, critical, 0.0)
    return adjusted, critical


def adjust_p_values(
    p_values: ArrayLike,
    /,
    *,
    method: MultipleTestingMethod = "holm",
    alpha: float = 0.05,
    mask: ArrayLike | None = None,
    axis: int = -1,
) -> MultipleTestingResult:
    """Correct one fixed hypothesis family along ``axis``.

    Masked hypotheses remain present with adjusted value one and cannot be selected.
    Ties use their stable original order, while adjusted values obey the method's
    monotonicity constraint in sorted order.
    """

    plan = MultipleTestingPlan(alpha=alpha, method=method)
    values, family_mask, resolved_axis = _inputs(p_values, mask, axis)
    order, ordered, ordered_mask, family_size = _ordered(values, family_mask)
    adjusted_ordered, unit_critical = _adjust_ordered(
        ordered, ordered_mask, family_size, plan.method
    )
    inverse = jnp.argsort(order, axis=-1, stable=True)
    adjusted = jnp.take_along_axis(adjusted_ordered, inverse, axis=-1)
    rejected = family_mask & (adjusted <= plan.alpha)
    critical = unit_critical * plan.alpha
    status = jnp.where(
        family_size > 0,
        MULTIPLE_TESTING_SUCCESS,
        MULTIPLE_TESTING_EMPTY_FAMILY,
    ).astype(jnp.int32)
    return MultipleTestingResult(
        raw_p_values=jnp.moveaxis(values, -1, resolved_axis),
        adjusted_p_values=jnp.moveaxis(adjusted, -1, resolved_axis),
        family_mask=jnp.moveaxis(family_mask, -1, resolved_axis),
        rejected=jnp.moveaxis(rejected, -1, resolved_axis),
        ordered_indices=order,
        ordered_raw_p_values=jnp.where(ordered_mask, ordered, 1.0),
        ordered_adjusted_p_values=adjusted_ordered,
        critical_values=critical,
        family_size=family_size,
        status=status,
        alpha=plan.alpha,
        method=plan.method,
        axis=resolved_axis,
    )


def holm_adjust(
    p_values: ArrayLike,
    /,
    *,
    alpha: float = 0.05,
    mask: ArrayLike | None = None,
    axis: int = -1,
) -> MultipleTestingResult:
    """Apply Holm's step-down family-wise error correction."""

    return adjust_p_values(p_values, method="holm", alpha=alpha, mask=mask, axis=axis)


def benjamini_hochberg(
    p_values: ArrayLike,
    /,
    *,
    alpha: float = 0.05,
    mask: ArrayLike | None = None,
    axis: int = -1,
) -> MultipleTestingResult:
    """Apply Benjamini--Hochberg false-discovery-rate correction."""

    return adjust_p_values(
        p_values,
        method="benjamini-hochberg",
        alpha=alpha,
        mask=mask,
        axis=axis,
    )


__all__ = [
    "MULTIPLE_TESTING_EMPTY_FAMILY",
    "MULTIPLE_TESTING_SUCCESS",
    "MultipleTestingMethod",
    "MultipleTestingPlan",
    "MultipleTestingResult",
    "adjust_p_values",
    "benjamini_hochberg",
    "holm_adjust",
]
