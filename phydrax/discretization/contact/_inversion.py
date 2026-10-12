#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from itertools import combinations
from math import ceil, comb, isfinite, log2

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg._small_batched import determinant_small_linear, SmallLinearSolvePlan
from ...typing import checked


_TRIANGLE_DETERMINANT = SmallLinearSolvePlan(2)
_TETRAHEDRON_DETERMINANT = SmallLinearSolvePlan(3)


class InversionStatus(IntEnum):
    SUCCESS = 0
    INITIAL_INVALID = 1
    WORK_LIMIT = 2
    NONFINITE_INPUT = 3


class SimplexInversionStepPlan(StrictModule, NonTrainableState):
    """Conservative Bernstein enclosure for affine triangle/tetrahedron orientation."""

    cells: Array
    reference_positions: Array
    reference_measure: Array
    minimum_ratio: float = eqx.field(static=True)
    time_tolerance: float = eqx.field(static=True)
    numerical_error: float = eqx.field(static=True)
    conservative_rescaling: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: ArrayLike,
        reference_positions: ArrayLike,
        /,
        *,
        minimum_ratio: float = 1.0e-8,
        time_tolerance: float = 1.0e-10,
        numerical_error: float = 1.0e-13,
        conservative_rescaling: float = 0.8,
        maximum_iterations: int = 1_000_000,
    ) -> None:
        topology = np.asarray(cells)
        reference = np.asarray(reference_positions, dtype=np.float64)
        if (
            reference.ndim != 2
            or reference.shape[1] not in (2, 3)
            or np.any(~np.isfinite(reference))
        ):
            raise ValueError(
                "reference_positions must be finite with dimension two or three."
            )
        dimension = reference.shape[1]
        expected_arity = dimension + 1
        if (
            topology.ndim != 2
            or topology.shape[1:] != (expected_arity,)
            or not np.issubdtype(topology.dtype, np.integer)
        ):
            raise TypeError("Simplex cells must have arity dimension + one.")
        topology = topology.astype(np.int32, copy=False)
        if (
            topology.shape[0] == 0
            or np.any(topology < 0)
            or np.any(topology >= reference.shape[0])
        ):
            raise ValueError(
                "Simplex cells must be nonempty and index reference positions."
            )
        if np.any(np.diff(np.sort(topology, axis=1), axis=1) == 0):
            raise ValueError("Simplex cells cannot repeat a vertex.")
        ratios = (
            float(minimum_ratio),
            float(time_tolerance),
            float(numerical_error),
            float(conservative_rescaling),
        )
        if not isfinite(ratios[0]) or not 0.0 <= ratios[0] < 1.0:
            raise ValueError("minimum_ratio must lie in [0, 1).")
        if not isfinite(ratios[1]) or ratios[1] <= 0.0:
            raise ValueError("time_tolerance must be finite and positive.")
        if not isfinite(ratios[2]) or ratios[2] < 0.0:
            raise ValueError("numerical_error must be finite and nonnegative.")
        if not isfinite(ratios[3]) or not 0.0 < ratios[3] < 1.0:
            raise ValueError(
                "conservative_rescaling must lie strictly between zero and one."
            )
        iterations = int(maximum_iterations)
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        reference_measure = _simplex_determinants(reference[topology])
        if np.any(reference_measure == 0.0) or np.any(~np.isfinite(reference_measure)):
            raise ValueError("Reference simplex cells must be nondegenerate.")
        self.cells = jnp.asarray(topology, dtype=jnp.int32)
        self.reference_positions = jnp.asarray(reference, dtype=jnp.float64)
        self.reference_measure = jnp.asarray(reference_measure, dtype=jnp.float64)
        (
            self.minimum_ratio,
            self.time_tolerance,
            self.numerical_error,
            self.conservative_rescaling,
        ) = ratios
        self.maximum_iterations = iterations
        self.dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "simplex-inversion-step-plan",
                "cells": array_tree_fingerprint(topology),
                "reference": array_tree_fingerprint(reference),
                "minimum_ratio": ratios[0].hex(),
                "time_tolerance": ratios[1].hex(),
                "numerical_error": ratios[2].hex(),
                "conservative_rescaling": ratios[3].hex(),
                "maximum_iterations": iterations,
            }
        )


class InversionStepEvidence(StrictModule):
    step_size: Array
    minimum_time: Array
    limiting_cell: Array
    interval_count: Array
    minimum_initial_ratio: Array
    status: Array
    finite: Array
    conservative: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(InversionStatus.SUCCESS)


def _simplex_determinants(cells: np.ndarray, /) -> np.ndarray:
    edges = cells[:, 1:] - cells[:, :1]
    if cells.shape[-1] == 2:
        return edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
    return np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=-1)


def _determinant_power_coefficients(
    start: Array, end: Array, reference_measure: Array, /
) -> Array:
    dimension = start.shape[1]
    edges = start[1:] - start[:1]
    delta = end[1:] - end[:1] - edges
    determinant_plan = (
        _TRIANGLE_DETERMINANT if dimension == 2 else _TETRAHEDRON_DETERMINANT
    )
    # Multilinearity produces the exact degree-d determinant polynomial.
    coefficients = []
    for order in range(dimension + 1):
        coefficient = jnp.zeros((), dtype=start.dtype)
        for columns in combinations(range(dimension), order):
            matrix = jnp.stack(
                tuple(
                    delta[row] if row in columns else edges[row]
                    for row in range(dimension)
                )
            )
            coefficient += determinant_small_linear(determinant_plan, matrix)
        coefficients.append(coefficient)
    return jnp.stack(coefficients) * jnp.sign(reference_measure)


def _power_to_bernstein(power: Array, /) -> Array:
    degree = power.shape[0] - 1
    return jnp.stack(
        tuple(
            sum(
                (comb(index, order) / comb(degree, order)) * power[order]
                for order in range(index + 1)
            )
            for index in range(degree + 1)
        )
    )


def _split_bernstein(values: Array, /) -> tuple[Array, Array]:
    degree = values.shape[0] - 1
    levels = [values]
    for _ in range(degree):
        levels.append(0.5 * (levels[-1][:-1] + levels[-1][1:]))
    left = jnp.stack(tuple(levels[index][0] for index in range(degree + 1)))
    right = jnp.stack(tuple(levels[degree - index][index] for index in range(degree + 1)))
    return left, right


type _CellCarry = tuple[Array, Array, Array, Array, Array, Array, Array]


def _cell_limit(
    plan: SimplexInversionStepPlan, start: Array, end: Array, reference_measure: Array, /
) -> tuple[Array, Array, Array, Array]:
    power = _determinant_power_coefficients(start, end, reference_measure)
    power = power.at[0].add(-plan.minimum_ratio * jnp.abs(reference_measure))
    bernstein = _power_to_bernstein(power) - plan.numerical_error
    invalid = bernstein[0] <= 0.0
    capacity = max(2, ceil(-log2(plan.time_tolerance)) + 3)
    lower = jnp.zeros((capacity,), dtype=start.dtype)
    upper = jnp.zeros((capacity,), dtype=start.dtype).at[0].set(1.0)
    stack = (
        jnp.zeros((capacity, plan.dimension + 1), dtype=start.dtype).at[0].set(bernstein)
    )
    infinity = jnp.full((), jnp.inf, dtype=start.dtype)
    initial = (
        lower,
        upper,
        stack,
        jnp.where(invalid, 0, 1).astype(jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(invalid, jnp.zeros_like(infinity), infinity),
        jnp.asarray(False, dtype=jnp.bool_),
    )

    def condition(carry: _CellCarry) -> Array:
        return (carry[3] > 0) & (carry[4] < plan.maximum_iterations) & ~carry[6]

    def body(carry: _CellCarry) -> _CellCarry:
        lowers, uppers, coefficients, size, iterations, toi, exhausted = carry
        index = size - 1
        low, high, values = lowers[index], uppers[index], coefficients[index]
        free = jnp.all(values > 0.0)
        terminal = high - low <= plan.time_tolerance
        midpoint = 0.5 * (low + high)
        split = ~free & ~terminal
        room = (index + 1 < capacity) & (midpoint > low) & (midpoint < high)
        unresolved = ~free & terminal
        left, right = _split_bernstein(values)
        next_index = jnp.minimum(index + 1, capacity - 1)
        lowers = lowers.at[index].set(midpoint).at[next_index].set(low)
        uppers = uppers.at[index].set(high).at[next_index].set(midpoint)
        coefficients = coefficients.at[index].set(right).at[next_index].set(left)
        size = jnp.where(unresolved, 0, jnp.where(split & room, index + 2, index))
        return (
            lowers,
            uppers,
            coefficients,
            size,
            iterations + 1,
            jnp.where(unresolved, low, toi),
            exhausted | (split & ~room),
        )

    result = jax.lax.while_loop(condition, body, initial)
    exhausted = result[6] | (result[3] > 0)
    return (
        jnp.where(exhausted, jnp.zeros_like(infinity), result[5]),
        jnp.where(invalid, 1, result[4]).astype(jnp.int32),
        invalid,
        exhausted,
    )


type _InversionCarry = tuple[Array, Array, Array, Array, Array]


@checked
def simplex_inversion_step_limit(
    plan: SimplexInversionStepPlan,
    start_positions: ArrayLike,
    end_positions: ArrayLike,
    /,
) -> InversionStepEvidence:
    """Bound the entire linear simplex path, including interior orientation loss."""
    start = jnp.asarray(start_positions, dtype=plan.reference_positions.dtype)
    end = jnp.asarray(end_positions, dtype=start.dtype)
    expected = plan.reference_positions.shape
    if start.shape != expected or end.shape != expected:
        raise ValueError("Inversion positions must match reference_positions.")
    finite = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
    determinant_plan = (
        _TRIANGLE_DETERMINANT if plan.dimension == 2 else _TETRAHEDRON_DETERMINANT
    )
    reference_measure = plan.reference_measure
    cells_start = start[plan.cells]
    start_measure = determinant_small_linear(
        determinant_plan, cells_start[:, 1:] - cells_start[:, :1]
    )
    ratio = start_measure / reference_measure
    initial = (
        jnp.full((), jnp.inf, dtype=start.dtype),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False, dtype=jnp.bool_),
        jnp.asarray(False, dtype=jnp.bool_),
    )

    def body(cell: Array, carry: _InversionCarry) -> _InversionCarry:
        earliest, limiting, intervals, invalid, exhausted = carry
        limit, count, initial_invalid, work_limit = _cell_limit(
            plan, start[plan.cells[cell]], end[plan.cells[cell]], reference_measure[cell]
        )
        return (
            jnp.minimum(earliest, limit),
            jnp.where(limit < earliest, cell, limiting),
            intervals + count,
            invalid | initial_invalid,
            exhausted | work_limit,
        )

    earliest, limiting, intervals, invalid, exhausted = jax.lax.cond(
        finite,
        lambda _: jax.lax.fori_loop(0, plan.cells.shape[0], body, initial),
        lambda _: initial,
        None,
    )
    status = jnp.where(
        ~finite,
        int(InversionStatus.NONFINITE_INPUT),
        jnp.where(
            invalid,
            int(InversionStatus.INITIAL_INVALID),
            jnp.where(
                exhausted, int(InversionStatus.WORK_LIMIT), int(InversionStatus.SUCCESS)
            ),
        ),
    ).astype(jnp.int32)
    reported = jnp.where(jnp.isfinite(earliest), earliest, jnp.ones_like(earliest))
    step = jnp.where(
        status == int(InversionStatus.SUCCESS),
        jnp.where(
            jnp.isfinite(earliest),
            plan.conservative_rescaling * earliest,
            jnp.ones_like(earliest),
        ),
        jnp.zeros_like(earliest),
    )
    return InversionStepEvidence(
        step,
        reported,
        limiting,
        intervals,
        jnp.where(finite, jnp.min(ratio), jnp.full((), jnp.nan, dtype=start.dtype)),
        status,
        finite,
        jnp.asarray(True, dtype=jnp.bool_),
        plan.plan_id,
    )


__all__ = [
    "InversionStatus",
    "InversionStepEvidence",
    "SimplexInversionStepPlan",
    "simplex_inversion_step_limit",
]
