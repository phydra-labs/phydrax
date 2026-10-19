#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared interval products and Krawczyk inclusion for BRep root owners."""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    inverse_small_linear,
    LinearSolvePolicy,
    LinearSystem,
    SmallLinearSolvePlan,
    solve,
)
from .._interval_enclosure import interval_midpoint_radius
from ._interval_source import (
    evaluate_interval,
    IntervalEvaluator,
    ScopedIntervalEvaluator,
)
from ._request_preparations import TrimRequestPreparations


_UNIT = np.finfo(np.float64).eps


@eqx.filter_jit
def _inverses(matrices: Array, plan: SmallLinearSolvePlan) -> Array:
    return inverse_small_linear(plan, matrices).value


@eqx.filter_jit
def _dense_inverses(matrices: Array) -> Array:
    def one(matrix: Array) -> Array:
        result = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            jnp.eye(matrix.shape[0], dtype=matrix.dtype),
            policy=LinearSolvePolicy(DenseLU()),
        )
        return jnp.where(
            jnp.all(result.successful), result.value, jnp.zeros_like(result.value)
        )

    return jax.vmap(one)(matrices)


def round_interval(
    lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    # Indeterminate midpoint products arise for valid unbounded enclosures
    # (for example, a coarse rational denominator interval straddling zero).
    # They mean no information, NEVER an exclusion certificate.
    lower = np.where(np.isnan(lower), -np.inf, lower)
    upper = np.where(np.isnan(upper), np.inf, upper)
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def point_times_interval(
    matrix: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """``matrix @ [lower, upper]`` for batched point matrices ``(B, m, k)``."""
    middle, radius = (
        np.asarray(value, dtype=np.float64)
        for value in interval_midpoint_radius((lower, upper))
    )
    absolute = np.abs(matrix)
    center = matrix @ middle
    width = absolute @ radius
    length = matrix.shape[-1] + 1
    gamma = length * _UNIT / (1.0 - length * _UNIT)
    width = np.nextafter(width + gamma * (absolute @ np.abs(middle) + width), np.inf)
    return round_interval(center - width, center + width)


def interval_times_interval(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    first_mid, first_rad = (
        np.asarray(value, dtype=np.float64)
        for value in interval_midpoint_radius((first_lower, first_upper))
    )
    second_mid, second_rad = (
        np.asarray(value, dtype=np.float64)
        for value in interval_midpoint_radius((second_lower, second_upper))
    )
    center = first_mid @ second_mid
    width = (
        np.abs(first_mid) @ second_rad
        + first_rad @ np.abs(second_mid)
        + first_rad @ second_rad
    )
    length = first_mid.shape[-1] + 1
    gamma = length * _UNIT / (1.0 - length * _UNIT)
    width = np.nextafter(
        width + gamma * (np.abs(first_mid) @ np.abs(second_mid) + width), np.inf
    )
    return round_interval(center - width, center + width)


def interval_magnitude(lower: np.ndarray, upper: np.ndarray, /) -> np.ndarray:
    return np.maximum(np.abs(lower), np.abs(upper))


class KrawczykEvaluator(Protocol):
    @property
    def value(self) -> IntervalEvaluator | ScopedIntervalEvaluator: ...

    @property
    def jacobian(self) -> IntervalEvaluator | ScopedIntervalEvaluator: ...

    def point_jacobians(
        self,
        points: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray: ...


def approximate_inverses(matrices: np.ndarray, /) -> np.ndarray:
    """Floating preconditioners for Krawczyk; any matrix keeps the test sound."""
    dimension = matrices.shape[-1]
    count = matrices.shape[0]
    batch = 64
    parts = []
    for start in range(0, count, batch):
        chunk = matrices[start : start + batch]
        padding = batch - chunk.shape[0]
        padded = np.concatenate(
            (chunk, np.repeat(np.eye(dimension, dtype=np.float64)[None], padding, axis=0))
        )
        if dimension <= 4:
            inverse = _inverses(jnp.asarray(padded), SmallLinearSolvePlan(dimension))
        else:
            inverse = _dense_inverses(jnp.asarray(padded))
        parts.append(np.asarray(inverse)[: chunk.shape[0]])
    inverses = np.concatenate(parts) if parts else np.zeros_like(matrices)
    return np.where(np.isfinite(inverses), inverses, 0.0)


class KrawczykResult(StrictModule):
    certified: np.ndarray
    excluded: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    preconditioner: np.ndarray
    contraction: np.ndarray


def krawczyk(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    parameter_axis: int | None = None,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> KrawczykResult:
    """(Parametric) Krawczyk inclusion over batched boxes.

    Without a parameter axis the system is square; with one, the chart axis is
    an interval parameter and inclusion certifies a unique graph point for every
    parameter value.
    """
    dimension = lower.shape[1]
    free = np.asarray(
        [axis for axis in range(dimension) if axis != parameter_axis], dtype=np.int64
    )
    center = 0.5 * (lower + upper)
    center_lower = center.copy()
    center_upper = center.copy()
    if parameter_axis is not None:
        center_lower[:, parameter_axis] = lower[:, parameter_axis]
        center_upper[:, parameter_axis] = upper[:, parameter_axis]
    value_lower, value_upper = evaluate_interval(
        prepared.value,
        center_lower,
        center_upper,
        budget=budget,
        preparations=preparations,
    )
    jacobian_lower, jacobian_upper = evaluate_interval(
        prepared.jacobian, lower, upper, budget=budget, preparations=preparations
    )
    jacobian_lower = jacobian_lower[:, :, free]
    jacobian_upper = jacobian_upper[:, :, free]
    point_jacobian = prepared.point_jacobians(
        center, budget=budget, preparations=preparations
    )[:, :, free]
    preconditioner = approximate_inverses(point_jacobian)
    shift_lower, shift_upper = point_times_interval(
        preconditioner, value_lower[..., None], value_upper[..., None]
    )
    product_lower, product_upper = point_times_interval(
        preconditioner, jacobian_lower, jacobian_upper
    )
    identity = np.eye(free.shape[0], dtype=np.float64)[None]
    residual_lower, residual_upper = round_interval(
        identity - product_upper, identity - product_lower
    )
    offset_lower = (lower - center)[:, free, None]
    offset_upper = (upper - center)[:, free, None]
    spread_lower, spread_upper = interval_times_interval(
        residual_lower, residual_upper, offset_lower, offset_upper
    )
    base = center[:, free]
    result_lower, result_upper = round_interval(
        base - shift_upper[..., 0] + spread_lower[..., 0],
        base - shift_lower[..., 0] + spread_upper[..., 0],
    )
    finite = np.all(np.isfinite(result_lower) & np.isfinite(result_upper), axis=1)
    box_lower = lower[:, free]
    box_upper = upper[:, free]
    certified = (
        finite
        & np.all(result_lower > box_lower, axis=1)
        & np.all(result_upper < box_upper, axis=1)
    )
    excluded = finite & np.any(
        (result_upper < box_lower) | (result_lower > box_upper), axis=1
    )
    contraction = np.max(
        np.nextafter(
            np.sum(interval_magnitude(residual_lower, residual_upper), axis=2)
            * (1.0 + 8.0 * _UNIT),
            np.inf,
        ),
        axis=1,
    )
    certified &= contraction < 1.0
    return KrawczykResult(
        certified=certified,
        excluded=excluded,
        lower=np.where(finite[:, None], result_lower, box_lower),
        upper=np.where(finite[:, None], result_upper, box_upper),
        preconditioner=preconditioner,
        contraction=contraction,
    )
