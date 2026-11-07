#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Outward interval products and Krawczyk inclusion for native B-Rep roots."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    inverse_small_linear,
    LinearSolvePolicy,
    LinearSystem,
    SmallLinearSolvePlan,
    solve,
)


_UNIT = np.finfo(np.float64).eps
_PRECONDITIONER_ROWS = 64


class IntervalEvaluator(Protocol):
    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]: ...


class KrawczykEvaluator(Protocol):
    @property
    def value(self) -> IntervalEvaluator: ...

    @property
    def jacobian(self) -> IntervalEvaluator: ...

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray: ...


class KrawczykResult(StrictModule):
    certified: np.ndarray
    excluded: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    preconditioner: np.ndarray
    contraction: np.ndarray

    def __init__(
        self,
        *,
        certified: np.ndarray,
        excluded: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        preconditioner: np.ndarray,
        contraction: np.ndarray,
    ) -> None:
        self.certified = certified
        self.excluded = excluded
        self.lower = lower
        self.upper = upper
        self.preconditioner = preconditioner
        self.contraction = contraction


@jax.jit
def _small_preconditioners(matrices: Array) -> Array:
    plan = SmallLinearSolvePlan(matrices.shape[-1])
    return inverse_small_linear(plan, matrices).value


@jax.jit
def _dense_preconditioners(matrices: Array) -> Array:
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


def _midpoint_radius(
    lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite center and enclosing radius, including unbounded inputs."""
    with np.errstate(invalid="ignore", over="ignore"):
        midpoint = 0.5 * lower + 0.5 * upper
    midpoint = np.where(np.isfinite(midpoint), midpoint, 0.0)
    with np.errstate(invalid="ignore", over="ignore"):
        radius = np.maximum(midpoint - lower, upper - midpoint)
    radius = np.where(np.isfinite(radius), radius, np.inf)
    return midpoint, np.nextafter(radius, np.inf)


def round_interval(
    lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Round endpoints outward; indeterminate arithmetic conveys no exclusion."""
    lower = np.where(np.isnan(lower), -np.inf, lower)
    upper = np.where(np.isnan(upper), np.inf, upper)
    return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


def point_times_interval(
    matrix: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Enclose batched point-matrix products with interval vectors/matrices."""
    middle, radius = _midpoint_radius(lower, upper)
    absolute = np.abs(matrix)
    with np.errstate(invalid="ignore", over="ignore"):
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
    """Enclose batched interval-matrix products using midpoint/radius bounds."""
    first_mid, first_rad = _midpoint_radius(first_lower, first_upper)
    second_mid, second_rad = _midpoint_radius(second_lower, second_upper)
    with np.errstate(invalid="ignore", over="ignore"):
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


def approximate_inverses(matrices: np.ndarray, /) -> np.ndarray:
    """Compute floating Krawczyk preconditioners in stable compiled row batches."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Preconditioner matrices must have shape (count, n, n).")
    count, dimension = matrices.shape[0], matrices.shape[-1]
    if count == 0:
        return np.zeros_like(matrices)
    kernel = _small_preconditioners if dimension <= 4 else _dense_preconditioners
    identity = np.eye(dimension, dtype=np.float64)
    inverses = np.empty_like(matrices)
    for start in range(0, count, _PRECONDITIONER_ROWS):
        chunk = matrices[start : start + _PRECONDITIONER_ROWS]
        active = chunk.shape[0]
        padded = np.empty((_PRECONDITIONER_ROWS, dimension, dimension), dtype=np.float64)
        padded[:active] = chunk
        padded[active:] = identity
        inverses[start : start + active] = np.asarray(kernel(padded))[:active]
    return np.where(np.isfinite(inverses), inverses, 0.0)


def krawczyk(
    prepared: KrawczykEvaluator,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
    *,
    parameter_axis: int | None = None,
) -> KrawczykResult:
    """Certify regular roots or graphs over a parameter axis in interval boxes."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.ndim != 2 or upper.shape != lower.shape:
        raise ValueError("Krawczyk bounds must have aligned shape (count, dimension).")
    dimension = lower.shape[1]
    if (
        dimension == 0
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise ValueError(
            "Krawczyk bounds must be ordered, non-NaN, and nonempty in dimension."
        )
    if parameter_axis is not None and (
        isinstance(parameter_axis, bool)
        or not isinstance(parameter_axis, int)
        or not 0 <= parameter_axis < dimension
    ):
        raise ValueError("parameter_axis must select one system coordinate.")
    free = np.asarray(
        [axis for axis in range(dimension) if axis != parameter_axis], dtype=np.int64
    )
    if free.size == 0:
        raise ValueError("Krawczyk inclusion requires at least one solved coordinate.")
    center = 0.5 * lower + 0.5 * upper
    center_lower = center.copy()
    center_upper = center.copy()
    if parameter_axis is not None:
        center_lower[:, parameter_axis] = lower[:, parameter_axis]
        center_upper[:, parameter_axis] = upper[:, parameter_axis]
    value_lower, value_upper = prepared.value.evaluate(center_lower, center_upper)
    jacobian_lower, jacobian_upper = prepared.jacobian.evaluate(lower, upper)
    jacobian_lower = jacobian_lower[:, :, free]
    jacobian_upper = jacobian_upper[:, :, free]
    point_jacobian = np.asarray(prepared.point_jacobians(center), dtype=np.float64)[
        :, :, free
    ]
    preconditioner = approximate_inverses(point_jacobian)
    shift_lower, shift_upper = point_times_interval(
        preconditioner, value_lower[..., None], value_upper[..., None]
    )
    product_lower, product_upper = point_times_interval(
        preconditioner, jacobian_lower, jacobian_upper
    )
    identity = np.eye(free.size, dtype=np.float64)[None]
    residual_lower, residual_upper = round_interval(
        identity - product_upper, identity - product_lower
    )
    # Every binary64 operation below rounds to nearest once, so each result is
    # widened by one outward step before it feeds the next operation.
    offset_lower, offset_upper = round_interval(lower - center, upper - center)
    spread_lower, spread_upper = interval_times_interval(
        residual_lower,
        residual_upper,
        offset_lower[:, free, None],
        offset_upper[:, free, None],
    )
    base = center[:, free]
    step_lower, step_upper = round_interval(
        base - shift_upper[..., 0], base - shift_lower[..., 0]
    )
    result_lower, result_upper = round_interval(
        step_lower + spread_lower[..., 0], step_upper + spread_upper[..., 0]
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


__all__ = [
    "KrawczykEvaluator",
    "KrawczykResult",
    "approximate_inverses",
    "interval_magnitude",
    "interval_times_interval",
    "krawczyk",
    "point_times_interval",
    "round_interval",
]
