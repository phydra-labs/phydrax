#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import comb, isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class InversionStatus(IntEnum):
    SUCCESS = 0
    INITIAL_INVALID = 1
    WORK_LIMIT = 2
    NONFINITE_INPUT = 3


class SimplexInversionStepPlan(StrictModule, NonTrainableState):
    """Conservative Bernstein enclosure for affine triangle/tetrahedron orientation."""

    cells: Array
    reference_positions: Array
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
    ):
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
        dimension = int(reference.shape[1])
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
        self.reference_positions = jnp.asarray(reference)
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


def _det2(first, second, /):
    return first[0] * second[1] - first[1] * second[0]


def _det3(first, second, third, /):
    return float(np.dot(first, np.cross(second, third)))


def _simplex_determinants(cells: np.ndarray, /) -> np.ndarray:
    edges = cells[:, 1:] - cells[:, :1]
    if cells.shape[-1] == 2:
        return edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
    return np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=-1)


def _determinant_power_coefficients(
    start: np.ndarray,
    end: np.ndarray,
    orientation: float,
    threshold: float,
    /,
) -> np.ndarray:
    start_edges = start[1:] - start[0]
    end_edges = end[1:] - end[0]
    delta = end_edges - start_edges
    if start.shape[1] == 2:
        coefficients = np.asarray(
            (
                _det2(start_edges[0], start_edges[1]),
                _det2(delta[0], start_edges[1]) + _det2(start_edges[0], delta[1]),
                _det2(delta[0], delta[1]),
            ),
            dtype=np.float64,
        )
    else:
        coefficients = np.asarray(
            (
                _det3(start_edges[0], start_edges[1], start_edges[2]),
                _det3(delta[0], start_edges[1], start_edges[2])
                + _det3(start_edges[0], delta[1], start_edges[2])
                + _det3(start_edges[0], start_edges[1], delta[2]),
                _det3(delta[0], delta[1], start_edges[2])
                + _det3(delta[0], start_edges[1], delta[2])
                + _det3(start_edges[0], delta[1], delta[2]),
                _det3(delta[0], delta[1], delta[2]),
            ),
            dtype=np.float64,
        )
    coefficients *= orientation
    coefficients[0] -= threshold
    return coefficients


def _power_to_bernstein(power: np.ndarray, /) -> np.ndarray:
    degree = power.size - 1
    bernstein = np.zeros_like(power)
    for index in range(degree + 1):
        bernstein[index] = sum(
            comb(index, order) / comb(degree, order) * power[order]
            for order in range(index + 1)
        )
    return bernstein


def _split_bernstein(values: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    degree = values.size - 1
    levels = [np.asarray(values, dtype=np.float64)]
    for _ in range(degree):
        levels.append(0.5 * (levels[-1][:-1] + levels[-1][1:]))
    left = np.asarray([levels[index][0] for index in range(degree + 1)])
    right = np.asarray([levels[degree - index][index] for index in range(degree + 1)])
    return left, right


def _cell_limit(
    plan: SimplexInversionStepPlan,
    start: np.ndarray,
    end: np.ndarray,
    reference_measure: float,
    /,
) -> tuple[float, int, bool, bool]:
    orientation = 1.0 if reference_measure > 0.0 else -1.0
    threshold = plan.minimum_ratio * abs(reference_measure)
    power = _determinant_power_coefficients(start, end, orientation, threshold)
    bernstein = _power_to_bernstein(power) - plan.numerical_error
    if bernstein[0] <= 0.0:
        return 0.0, 1, True, False
    stack: list[tuple[float, float, np.ndarray]] = [(0.0, 1.0, bernstein)]
    iterations = 0
    while stack:
        if iterations >= plan.maximum_iterations:
            return 0.0, iterations, False, True
        lower, upper, coefficients = stack.pop()
        iterations += 1
        if np.all(coefficients > 0.0):
            continue
        if upper - lower <= plan.time_tolerance:
            return lower, iterations, False, False
        left, right = _split_bernstein(coefficients)
        midpoint = 0.5 * (lower + upper)
        stack.append((midpoint, upper, right))
        stack.append((lower, midpoint, left))
    return np.inf, iterations, False, False


def simplex_inversion_step_limit(
    plan: SimplexInversionStepPlan,
    start_positions: ArrayLike,
    end_positions: ArrayLike,
    /,
) -> InversionStepEvidence:
    if not isinstance(plan, SimplexInversionStepPlan):
        raise TypeError("plan must be SimplexInversionStepPlan.")
    start = np.asarray(start_positions, dtype=np.float64)
    end = np.asarray(end_positions, dtype=np.float64)
    expected = np.asarray(plan.reference_positions).shape
    finite = (
        start.shape == expected
        and end.shape == expected
        and np.all(np.isfinite(start))
        and np.all(np.isfinite(end))
    )
    if not finite:
        return InversionStepEvidence(
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.nan),
            jnp.asarray(int(InversionStatus.NONFINITE_INPUT), dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(True),
            plan.plan_id,
        )
    topology = np.asarray(plan.cells, dtype=np.int32)
    reference_measure = _simplex_determinants(
        np.asarray(plan.reference_positions)[topology]
    )
    start_measure = _simplex_determinants(start[topology])
    ratio = np.sign(reference_measure) * start_measure / np.abs(reference_measure)
    minimum_ratio = float(np.min(ratio))
    earliest = np.inf
    limiting_cell = -1
    intervals = 0
    initial_invalid = False
    exhausted = False
    for cell in range(topology.shape[0]):
        limit, count, initial, work_exhausted = _cell_limit(
            plan,
            start[topology[cell]],
            end[topology[cell]],
            float(reference_measure[cell]),
        )
        intervals += count
        initial_invalid = initial_invalid or initial
        exhausted = exhausted or work_exhausted
        if limit < earliest:
            earliest = limit
            limiting_cell = cell
    if initial_invalid:
        status = InversionStatus.INITIAL_INVALID
        step = 0.0
    elif exhausted:
        status = InversionStatus.WORK_LIMIT
        step = 0.0
    else:
        status = InversionStatus.SUCCESS
        step = (
            1.0
            if not np.isfinite(earliest)
            else min(1.0, plan.conservative_rescaling * earliest)
        )
    reported = 1.0 if not np.isfinite(earliest) else earliest
    return InversionStepEvidence(
        jnp.asarray(step),
        jnp.asarray(reported),
        jnp.asarray(limiting_cell, dtype=jnp.int32),
        jnp.asarray(intervals, dtype=jnp.int32),
        jnp.asarray(minimum_ratio),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(True),
        plan.plan_id,
    )


__all__ = [
    "InversionStatus",
    "InversionStepEvidence",
    "SimplexInversionStepPlan",
    "simplex_inversion_step_limit",
]
