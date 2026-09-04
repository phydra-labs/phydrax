#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded one-dimensional discrete HJB reference calculations."""

from __future__ import annotations

import operator
from collections.abc import Callable
from enum import IntEnum
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import TimeGrid


_REFERENCE_METHOD = "bounded-uniform-1d-explicit-upwind-central"


class DiscreteHJBStatus(IntEnum):
    """Stable outcomes for the bounded one-dimensional reference calculation."""

    SUCCESS_DISCRETE_REFERENCE = 0
    NONFINITE_DISCRETE_OUTPUT = 1
    BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE = 2
    OPERATOR_RESIDUAL_TOO_LARGE = 3
    ACTION_MINIMUM_RESIDUAL_TOO_LARGE = 4
    REFINEMENT_GATE_FAILED = 5


class BoundedUniformGrid1D(StrictModule, NonTrainableState):
    """A closed bounded interval represented by at least three uniform points."""

    points: Array
    lower_bound: float = eqx.field(static=True)
    upper_bound: float = eqx.field(static=True)
    spacing: float = eqx.field(static=True)

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        num_points: int,
        /,
    ):
        lower = float(lower_bound)
        upper = float(upper_bound)
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError("Grid bounds must be finite and strictly increasing.")
        if isinstance(num_points, (bool, np.bool_)):
            raise TypeError("num_points must be an integer.")
        count = operator.index(num_points)
        if count < 3:
            raise ValueError(
                "A bounded 1D reference grid requires at least three points."
            )
        spacing = (upper - lower) / (count - 1)
        self.points = jnp.asarray(np.linspace(lower, upper, count, dtype=float))
        self.lower_bound = lower
        self.upper_bound = upper
        self.spacing = spacing

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def interior_points(self) -> Array:
        return self.points[1:-1]


class DiscreteHJBProblem(StrictModule, NonTrainableState):
    """A scalar-state, bounded-grid, reference-only finite-action HJB problem.

    ``drift``, ``diffusion``, and ``running_cost`` receive scalar
    ``(time, state, action, args)`` arguments and must return a finite real scalar.
    ``diffusion`` is the scalar diffusion amplitude, so the discrete generator uses
    one half of its square. Boundary columns are ordered lower then upper.
    """

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    actions: Array
    terminal_values: Array
    boundary_values: Array
    drift: Callable[[Array, Array, Array, Any], ArrayLike]
    diffusion: Callable[[Array, Array, Array, Any], ArrayLike]
    running_cost: Callable[[Array, Array, Array, Any], ArrayLike]
    args: Any
    corner_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_grid: BoundedUniformGrid1D,
        time_grid: TimeGrid,
        actions: ArrayLike,
        terminal_values: ArrayLike,
        boundary_values: ArrayLike,
        drift: Callable[[Array, Array, Array, Any], ArrayLike],
        diffusion: Callable[[Array, Array, Array, Any], ArrayLike],
        running_cost: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        args: Any = None,
        corner_tolerance: float = 0.0,
        problem_id: str,
    ):
        if not isinstance(spatial_grid, BoundedUniformGrid1D):
            raise TypeError("spatial_grid must be a BoundedUniformGrid1D.")
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        for name, callback in (
            ("drift", drift),
            ("diffusion", diffusion),
            ("running_cost", running_cost),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        tolerance = float(corner_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("corner_tolerance must be finite and nonnegative.")

        action_array = _finite_real_array(actions, "actions")
        if action_array.ndim != 1 or action_array.size == 0:
            raise ValueError("actions must be a nonempty rank-one scalar action grid.")
        terminal = _finite_real_array(terminal_values, "terminal_values")
        expected_terminal = (spatial_grid.num_points,)
        if terminal.shape != expected_terminal:
            raise ValueError(
                "terminal_values must have shape "
                f"{expected_terminal}; got {terminal.shape}."
            )
        boundary = _finite_real_array(boundary_values, "boundary_values")
        expected_boundary = (time_grid.num_times, 2)
        if boundary.shape != expected_boundary:
            raise ValueError(
                "boundary_values must have shape "
                f"{expected_boundary}; got {boundary.shape}."
            )
        corner_residual = float(
            np.max(np.abs(boundary[-1] - terminal[[0, terminal.size - 1]]))
        )
        if corner_residual > tolerance:
            raise ValueError(
                "Terminal values and final-time boundary data are incompatible at "
                "the interval corners."
            )

        dtype = jnp.result_type(action_array, terminal, boundary, float)
        self.spatial_grid = spatial_grid
        self.time_grid = time_grid
        self.actions = jnp.asarray(action_array, dtype=dtype)
        self.terminal_values = jnp.asarray(terminal, dtype=dtype)
        self.boundary_values = jnp.asarray(boundary, dtype=dtype)
        self.drift = drift
        self.diffusion = diffusion
        self.running_cost = running_cost
        self.args = args
        self.corner_tolerance = tolerance
        self.problem_id = identifier


class DiscreteHJBEvidence(StrictModule, NonTrainableState):
    """Residual and nested-grid evidence for one declared discrete scheme."""

    maximum_boundary_residual: Array
    maximum_terminal_residual: Array
    maximum_operator_residual: Array
    maximum_action_minimum_residual: Array
    maximum_refinement_difference: Array
    refinement_threshold: Array
    maximum_courant_number: Array
    minimum_monotonicity_margin: Array
    finite: Array
    boundary_passed: Array
    terminal_passed: Array
    operator_passed: Array
    action_minimum_passed: Array
    refinement_passed: Array
    method: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


class DiscreteHJBResult(StrictModule, NonTrainableState):
    """Finite-grid value table, minimizing selectors, and discrete evidence."""

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    actions: Array
    values: Array
    action_selectors: Array
    selected_actions: Array
    evidence: DiscreteHJBEvidence
    successful: Array
    status: Array
    problem_id: str = eqx.field(static=True)
    status_label: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


class DiscreteHJBRefinementResult(StrictModule, NonTrainableState):
    """Primary result plus the once-refined table used by its refinement gate."""

    result: DiscreteHJBResult
    refined_spatial_grid: BoundedUniformGrid1D
    refined_time_grid: TimeGrid
    refined_values: Array
    common_grid_difference: Array
    passed: Array
    status: Array
    status_label: str = eqx.field(static=True)


class _RawHJBSolution(NamedTuple):
    """Internal host result without a second refinement solve."""

    values: np.ndarray
    selectors: np.ndarray
    boundary_residual: float
    terminal_residual: float
    operator_residual: float
    action_residual: float
    maximum_courant: float
    minimum_margin: float
    finite: bool


def _finite_real_array(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _nonnegative_tolerance(value: float, name: str, /) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return tolerance


def _callback_scalar(
    callback: Callable,
    time: float,
    state: float,
    action: float,
    args: Any,
    name: str,
    /,
) -> float:
    value = np.asarray(
        callback(
            jnp.asarray(time),
            jnp.asarray(state),
            jnp.asarray(action),
            args,
        )
    )
    if value.shape != ():
        raise ValueError(f"{name} must return a scalar for scalar inputs.")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{name} must return a real scalar.")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must return a finite scalar on the declared grids.")
    return scalar


def _hjb_coefficients(
    problem: DiscreteHJBProblem,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    times = np.asarray(problem.time_grid.times, dtype=float)
    points = np.asarray(problem.spatial_grid.points, dtype=float)
    actions = np.asarray(problem.actions, dtype=float)
    shape = (times.size - 1, points.size - 2, actions.size)
    drift = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)
    cost = np.empty(shape, dtype=float)
    maximum_courant = 0.0
    minimum_margin = 1.0
    dx = problem.spatial_grid.spacing
    for step, (time, duration) in enumerate(zip(times[:-1], np.diff(times), strict=True)):
        for point_index, state in enumerate(points[1:-1]):
            for action_index, action in enumerate(actions):
                drift_value = _callback_scalar(
                    problem.drift,
                    time,
                    state,
                    action,
                    problem.args,
                    "drift",
                )
                diffusion_value = _callback_scalar(
                    problem.diffusion,
                    time,
                    state,
                    action,
                    problem.args,
                    "diffusion",
                )
                variance_value = diffusion_value * diffusion_value
                if not np.isfinite(variance_value):
                    raise ValueError(
                        "Squared diffusion must be finite on the declared grids."
                    )
                drift[step, point_index, action_index] = drift_value
                variance[step, point_index, action_index] = variance_value
                cost[step, point_index, action_index] = _callback_scalar(
                    problem.running_cost,
                    time,
                    state,
                    action,
                    problem.args,
                    "running_cost",
                )
                courant = duration * (abs(drift_value) / dx + variance_value / (dx * dx))
                maximum_courant = max(maximum_courant, courant)
                minimum_margin = min(minimum_margin, 1.0 - courant)
    if minimum_margin < -32.0 * np.finfo(float).eps:
        raise ValueError(
            "The declared time and spatial grids violate the explicit monotone "
            "upwind-diffusion step condition."
        )
    return drift, variance, cost, maximum_courant, minimum_margin


def _hjb_hamiltonian(
    values: np.ndarray,
    drift: np.ndarray,
    variance: np.ndarray,
    cost: np.ndarray,
    spacing: float,
    /,
) -> np.ndarray:
    backward = (values[1:-1] - values[:-2]) / spacing
    forward = (values[2:] - values[1:-1]) / spacing
    second = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (spacing * spacing)
    return (
        cost
        + np.maximum(drift, 0.0) * forward[:, None]
        + np.minimum(drift, 0.0) * backward[:, None]
        + 0.5 * variance * second[:, None]
    )


def _solve_hjb_raw(problem: DiscreteHJBProblem, /) -> _RawHJBSolution:
    drift, variance, cost, maximum_courant, minimum_margin = _hjb_coefficients(problem)
    times = np.asarray(problem.time_grid.times, dtype=float)
    boundary = np.asarray(problem.boundary_values, dtype=float)
    terminal = np.asarray(problem.terminal_values, dtype=float)
    values = np.empty((times.size, problem.spatial_grid.num_points), dtype=float)
    selectors = np.empty((times.size - 1, values.shape[1] - 2), dtype=np.int32)
    values[-1] = terminal
    for step in range(times.size - 2, -1, -1):
        hamiltonian = _hjb_hamiltonian(
            values[step + 1],
            drift[step],
            variance[step],
            cost[step],
            problem.spatial_grid.spacing,
        )
        selector = np.argmin(hamiltonian, axis=-1)
        selected = np.take_along_axis(hamiltonian, selector[:, None], axis=-1)[:, 0]
        values[step, 1:-1] = (
            values[step + 1, 1:-1] + (times[step + 1] - times[step]) * selected
        )
        values[step, 0] = boundary[step, 0]
        values[step, -1] = boundary[step, 1]
        selectors[step] = selector

    operator_residual = 0.0
    action_residual = 0.0
    for step, duration in enumerate(np.diff(times)):
        hamiltonian = _hjb_hamiltonian(
            values[step + 1],
            drift[step],
            variance[step],
            cost[step],
            problem.spatial_grid.spacing,
        )
        selected = np.take_along_axis(hamiltonian, selectors[step, :, None], axis=-1)[
            :, 0
        ]
        minimum = np.min(hamiltonian, axis=-1)
        operator_residual = max(
            operator_residual,
            float(
                np.max(
                    np.abs(
                        (values[step, 1:-1] - values[step + 1, 1:-1]) / duration
                        - selected
                    )
                )
            ),
        )
        action_residual = max(action_residual, float(np.max(np.abs(selected - minimum))))
    boundary_residual = float(
        np.max(np.abs(values[:, (0, values.shape[1] - 1)] - boundary))
    )
    terminal_residual = float(np.max(np.abs(values[-1] - terminal)))
    finite = bool(np.all(np.isfinite(values)))
    return _RawHJBSolution(
        values,
        selectors,
        boundary_residual,
        terminal_residual,
        operator_residual,
        action_residual,
        maximum_courant,
        minimum_margin,
        finite,
    )


def _refined_hjb_problem(problem: DiscreteHJBProblem, /) -> DiscreteHJBProblem:
    coarse_times = np.asarray(problem.time_grid.times, dtype=float)
    fractions = np.arange(4, dtype=float) / 4.0
    refined_times = np.concatenate(
        tuple(
            coarse_times[index]
            + fractions * (coarse_times[index + 1] - coarse_times[index])
            for index in range(coarse_times.size - 1)
        )
        + (coarse_times[-1:],)
    )
    refined_grid = BoundedUniformGrid1D(
        problem.spatial_grid.lower_bound,
        problem.spatial_grid.upper_bound,
        2 * (problem.spatial_grid.num_points - 1) + 1,
    )
    coarse_points = np.asarray(problem.spatial_grid.points, dtype=float)
    refined_points = np.asarray(refined_grid.points, dtype=float)
    terminal = np.interp(
        refined_points, coarse_points, np.asarray(problem.terminal_values, dtype=float)
    )
    boundary = np.column_stack(
        tuple(
            np.interp(
                refined_times,
                coarse_times,
                np.asarray(problem.boundary_values, dtype=float)[:, side],
            )
            for side in range(2)
        )
    )
    time_grid = TimeGrid(
        refined_times,
        time_id=f"{problem.time_grid.time_id}/space-2-time-4",
    )
    return DiscreteHJBProblem(
        refined_grid,
        time_grid,
        problem.actions,
        terminal,
        boundary,
        problem.drift,
        problem.diffusion,
        problem.running_cost,
        args=problem.args,
        corner_tolerance=problem.corner_tolerance,
        problem_id=f"{problem.problem_id}/space-2-time-4",
    )


def _hjb_status(
    *,
    finite: bool,
    boundary_passed: bool,
    terminal_passed: bool,
    operator_passed: bool,
    action_passed: bool,
    refinement_passed: bool,
) -> DiscreteHJBStatus:
    if not finite:
        return DiscreteHJBStatus.NONFINITE_DISCRETE_OUTPUT
    if not boundary_passed or not terminal_passed:
        return DiscreteHJBStatus.BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE
    if not operator_passed:
        return DiscreteHJBStatus.OPERATOR_RESIDUAL_TOO_LARGE
    if not action_passed:
        return DiscreteHJBStatus.ACTION_MINIMUM_RESIDUAL_TOO_LARGE
    if not refinement_passed:
        return DiscreteHJBStatus.REFINEMENT_GATE_FAILED
    return DiscreteHJBStatus.SUCCESS_DISCRETE_REFERENCE


def _solve_hjb_with_refinement(
    problem: DiscreteHJBProblem,
    /,
    *,
    residual_tolerance: float,
    refinement_absolute_tolerance: float,
    refinement_relative_tolerance: float,
) -> tuple[
    DiscreteHJBResult,
    BoundedUniformGrid1D,
    TimeGrid,
    _RawHJBSolution,
    np.ndarray,
]:
    if not isinstance(problem, DiscreteHJBProblem):
        raise TypeError("problem must be a DiscreteHJBProblem.")
    residual_tolerance = _nonnegative_tolerance(residual_tolerance, "residual_tolerance")
    absolute_tolerance = _nonnegative_tolerance(
        refinement_absolute_tolerance, "refinement_absolute_tolerance"
    )
    relative_tolerance = _nonnegative_tolerance(
        refinement_relative_tolerance, "refinement_relative_tolerance"
    )
    raw = _solve_hjb_raw(problem)
    refined_problem = _refined_hjb_problem(problem)
    refined = _solve_hjb_raw(refined_problem)
    refined_common = refined.values[::4, ::2]
    common_difference = np.abs(raw.values - refined_common)
    maximum_difference = float(np.max(common_difference))
    refinement_scale = float(np.max(np.abs(refined_common)))
    refinement_threshold = absolute_tolerance + relative_tolerance * refinement_scale

    maximum_boundary_residual = max(raw.boundary_residual, refined.boundary_residual)
    maximum_terminal_residual = max(raw.terminal_residual, refined.terminal_residual)
    maximum_operator_residual = max(raw.operator_residual, refined.operator_residual)
    maximum_action_residual = max(raw.action_residual, refined.action_residual)
    maximum_courant = max(raw.maximum_courant, refined.maximum_courant)
    minimum_margin = min(raw.minimum_margin, refined.minimum_margin)
    finite = raw.finite and refined.finite and bool(np.isfinite(maximum_difference))
    boundary_passed = maximum_boundary_residual <= residual_tolerance
    terminal_passed = maximum_terminal_residual <= residual_tolerance
    operator_passed = maximum_operator_residual <= residual_tolerance
    action_passed = maximum_action_residual <= residual_tolerance
    refinement_passed = maximum_difference <= refinement_threshold
    status = _hjb_status(
        finite=finite,
        boundary_passed=boundary_passed,
        terminal_passed=terminal_passed,
        operator_passed=operator_passed,
        action_passed=action_passed,
        refinement_passed=refinement_passed,
    )
    evidence = DiscreteHJBEvidence(
        maximum_boundary_residual=jnp.asarray(maximum_boundary_residual),
        maximum_terminal_residual=jnp.asarray(maximum_terminal_residual),
        maximum_operator_residual=jnp.asarray(maximum_operator_residual),
        maximum_action_minimum_residual=jnp.asarray(maximum_action_residual),
        maximum_refinement_difference=jnp.asarray(maximum_difference),
        refinement_threshold=jnp.asarray(refinement_threshold),
        maximum_courant_number=jnp.asarray(maximum_courant),
        minimum_monotonicity_margin=jnp.asarray(minimum_margin),
        finite=jnp.asarray(finite),
        boundary_passed=jnp.asarray(boundary_passed),
        terminal_passed=jnp.asarray(terminal_passed),
        operator_passed=jnp.asarray(operator_passed),
        action_minimum_passed=jnp.asarray(action_passed),
        refinement_passed=jnp.asarray(refinement_passed),
        method=_REFERENCE_METHOD,
        scope="declared-bounded-grid-discrete-residuals-only",
    )
    selectors = jnp.asarray(raw.selectors)
    result = DiscreteHJBResult(
        spatial_grid=problem.spatial_grid,
        time_grid=problem.time_grid,
        actions=problem.actions,
        values=jnp.asarray(raw.values),
        action_selectors=selectors,
        selected_actions=problem.actions[selectors],
        evidence=evidence,
        successful=jnp.asarray(status == DiscreteHJBStatus.SUCCESS_DISCRETE_REFERENCE),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        problem_id=problem.problem_id,
        status_label=status.name,
        method=_REFERENCE_METHOD,
    )
    return (
        result,
        refined_problem.spatial_grid,
        refined_problem.time_grid,
        refined,
        common_difference,
    )


def solve_discrete_hjb_reference(
    problem: DiscreteHJBProblem,
    /,
    *,
    residual_tolerance: float = 1.0e-8,
    refinement_absolute_tolerance: float = 2.0e-2,
    refinement_relative_tolerance: float = 5.0e-2,
) -> DiscreteHJBResult:
    """Solve and gate a finite-action HJB table against one nested refinement.

    The result concerns only the declared bounded grid and discrete operator. It
    makes no claim outside that finite calculation.
    """

    result, _, _, _, _ = _solve_hjb_with_refinement(
        problem,
        residual_tolerance=residual_tolerance,
        refinement_absolute_tolerance=refinement_absolute_tolerance,
        refinement_relative_tolerance=refinement_relative_tolerance,
    )
    return result


def refine_discrete_hjb_reference(
    problem: DiscreteHJBProblem,
    /,
    *,
    residual_tolerance: float = 1.0e-8,
    refinement_absolute_tolerance: float = 2.0e-2,
    refinement_relative_tolerance: float = 5.0e-2,
) -> DiscreteHJBRefinementResult:
    """Return the nested table and pointwise common-grid comparison explicitly."""

    result, spatial_grid, time_grid, refined, difference = _solve_hjb_with_refinement(
        problem,
        residual_tolerance=residual_tolerance,
        refinement_absolute_tolerance=refinement_absolute_tolerance,
        refinement_relative_tolerance=refinement_relative_tolerance,
    )
    return DiscreteHJBRefinementResult(
        result=result,
        refined_spatial_grid=spatial_grid,
        refined_time_grid=time_grid,
        refined_values=jnp.asarray(refined.values),
        common_grid_difference=jnp.asarray(difference),
        passed=result.evidence.refinement_passed,
        status=result.status,
        status_label=result.status_label,
    )


__all__ = [
    "BoundedUniformGrid1D",
    "DiscreteHJBEvidence",
    "DiscreteHJBProblem",
    "DiscreteHJBRefinementResult",
    "DiscreteHJBResult",
    "DiscreteHJBStatus",
    "refine_discrete_hjb_reference",
    "solve_discrete_hjb_reference",
]
