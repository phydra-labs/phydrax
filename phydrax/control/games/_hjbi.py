#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded one-dimensional zero-sum HJBI reference calculations."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import TimeGrid
from ..stochastic._hjb import (
    _finite_real_array,
    _nonnegative_tolerance,
    BoundedUniformGrid1D,
)


HJBIActionOrder: TypeAlias = Literal["max_min", "min_max"]
_REFERENCE_METHOD = "bounded-uniform-1d-explicit-upwind-central-nested-actions"


class DiscreteHJBIStatus(IntEnum):
    """Stable outcomes for the bounded one-dimensional zero-sum reference."""

    SUCCESS_DISCRETE_SADDLE_REFERENCE = 0
    NONFINITE_DISCRETE_OUTPUT = 1
    BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE = 2
    OPERATOR_RESIDUAL_TOO_LARGE = 3
    ACTION_ORDER_RESIDUAL_TOO_LARGE = 4
    REFINEMENT_GATE_FAILED = 5
    ISAACS_GAP_EXCEEDS_TOLERANCE = 6
    NONCANONICAL_SADDLE_ACTION_ORDERS = 7


class ScalarLQHJBIStatus(IntEnum):
    """Stable outcome for the closed-form scalar LQ HJBI calculation."""

    SUCCESS_ANALYTIC_SCALAR_LQ_HJBI = 0


class DiscreteZeroSumHJBIProblem(StrictModule, NonTrainableState):
    """A scalar-state, bounded-grid, reference-only finite-action HJBI problem.

    Coefficient callbacks receive scalar
    ``(time, state, minimizer_action, maximizer_action, args)`` arguments.
    ``diffusion`` returns an amplitude. ``lower_order`` and ``upper_order`` are
    separately declared and evaluated; neither is silently rewritten. Only lower
    ``max_min`` together with upper ``min_max`` is eligible for the saddle success
    label. Boundary columns are ordered lower then upper.
    """

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    minimizer_actions: Array
    maximizer_actions: Array
    terminal_values: Array
    boundary_values: Array
    drift: Callable[[Array, Array, Array, Array, Any], ArrayLike]
    diffusion: Callable[[Array, Array, Array, Array, Any], ArrayLike]
    running_cost: Callable[[Array, Array, Array, Array, Any], ArrayLike]
    args: Any
    lower_order: HJBIActionOrder = eqx.field(static=True)
    upper_order: HJBIActionOrder = eqx.field(static=True)
    corner_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_grid: BoundedUniformGrid1D,
        time_grid: TimeGrid,
        minimizer_actions: ArrayLike,
        maximizer_actions: ArrayLike,
        terminal_values: ArrayLike,
        boundary_values: ArrayLike,
        drift: Callable[[Array, Array, Array, Array, Any], ArrayLike],
        diffusion: Callable[[Array, Array, Array, Array, Any], ArrayLike],
        running_cost: Callable[[Array, Array, Array, Array, Any], ArrayLike],
        /,
        *,
        lower_order: HJBIActionOrder,
        upper_order: HJBIActionOrder,
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
        lower = _action_order(lower_order, "lower_order")
        upper = _action_order(upper_order, "upper_order")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        tolerance = _nonnegative_tolerance(corner_tolerance, "corner_tolerance")

        minimizer = _finite_real_array(minimizer_actions, "minimizer_actions")
        maximizer = _finite_real_array(maximizer_actions, "maximizer_actions")
        if minimizer.ndim != 1 or minimizer.size == 0:
            raise ValueError(
                "minimizer_actions must be a nonempty rank-one scalar action grid."
            )
        if maximizer.ndim != 1 or maximizer.size == 0:
            raise ValueError(
                "maximizer_actions must be a nonempty rank-one scalar action grid."
            )
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

        dtype = jnp.result_type(minimizer, maximizer, terminal, boundary, float)
        self.spatial_grid = spatial_grid
        self.time_grid = time_grid
        self.minimizer_actions = jnp.asarray(minimizer, dtype=dtype)
        self.maximizer_actions = jnp.asarray(maximizer, dtype=dtype)
        self.terminal_values = jnp.asarray(terminal, dtype=dtype)
        self.boundary_values = jnp.asarray(boundary, dtype=dtype)
        self.drift = drift
        self.diffusion = diffusion
        self.running_cost = running_cost
        self.lower_order = lower
        self.upper_order = upper
        self.args = args
        self.corner_tolerance = tolerance
        self.problem_id = identifier


class DiscreteHJBIEvidence(StrictModule, NonTrainableState):
    """Finite-grid evidence for both declared action orders and their gap."""

    maximum_boundary_residual: Array
    maximum_terminal_residual: Array
    maximum_lower_operator_residual: Array
    maximum_upper_operator_residual: Array
    maximum_lower_action_order_residual: Array
    maximum_upper_action_order_residual: Array
    maximum_lower_refinement_difference: Array
    maximum_upper_refinement_difference: Array
    refinement_threshold: Array
    maximum_isaacs_gap: Array
    isaacs_threshold: Array
    maximum_courant_number: Array
    minimum_monotonicity_margin: Array
    finite: Array
    boundary_passed: Array
    terminal_passed: Array
    operator_passed: Array
    action_orders_passed: Array
    canonical_saddle_action_orders: Array
    refinement_passed: Array
    isaacs_gap_passed: Array
    method: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


class DiscreteZeroSumHJBIResult(StrictModule, NonTrainableState):
    """Independent lower/upper tables, action selectors, gap, and gate status."""

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    minimizer_actions: Array
    maximizer_actions: Array
    lower_values: Array
    upper_values: Array
    lower_minimizer_selectors: Array
    lower_maximizer_selectors: Array
    upper_minimizer_selectors: Array
    upper_maximizer_selectors: Array
    lower_selected_minimizer_actions: Array
    lower_selected_maximizer_actions: Array
    upper_selected_minimizer_actions: Array
    upper_selected_maximizer_actions: Array
    isaacs_gap: Array
    evidence: DiscreteHJBIEvidence
    saddle: Array
    status: Array
    lower_order: HJBIActionOrder = eqx.field(static=True)
    upper_order: HJBIActionOrder = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    status_label: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


class ScalarLQHJBISolution(StrictModule, NonTrainableState):
    """Closed-form scalar terminal-cost LQ zero-sum HJBI solution.

    The represented game has dynamics ``x_dot = u + v``, running cost
    ``(u**2 - gamma**2 * v**2) / 2``, and terminal cost
    ``terminal_weight * x**2 / 2``.
    """

    time_grid: TimeGrid
    denominator: Array
    value_coefficient: Array
    minimizer_feedback_gain: Array
    maximizer_feedback_gain: Array
    gamma: float = eqx.field(static=True)
    terminal_weight: float = eqx.field(static=True)
    well_posed: Array
    status: Array
    status_label: str = eqx.field(static=True)

    def value(self, state: ArrayLike, /) -> Array:
        """Evaluate one half of the time-indexed quadratic value coefficient."""

        state_array = jnp.asarray(state)
        return 0.5 * self.value_coefficient * state_array * state_array


class _RawHJBI(NamedTuple):
    lower_values: np.ndarray
    upper_values: np.ndarray
    lower_minimizer_selectors: np.ndarray
    lower_maximizer_selectors: np.ndarray
    upper_minimizer_selectors: np.ndarray
    upper_maximizer_selectors: np.ndarray
    boundary_residual: float
    terminal_residual: float
    lower_operator_residual: float
    upper_operator_residual: float
    lower_action_residual: float
    upper_action_residual: float
    maximum_courant: float
    minimum_margin: float
    finite: bool


def _action_order(value: str, name: str, /) -> HJBIActionOrder:
    order = str(value)
    if order not in ("max_min", "min_max"):
        raise ValueError(f"{name} must be 'max_min' or 'min_max'.")
    return order


def _callback_scalar(
    callback: Callable,
    time: float,
    state: float,
    minimizer_action: float,
    maximizer_action: float,
    args: Any,
    name: str,
    /,
) -> float:
    value = np.asarray(
        callback(
            jnp.asarray(time),
            jnp.asarray(state),
            jnp.asarray(minimizer_action),
            jnp.asarray(maximizer_action),
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


def _hjbi_coefficients(
    problem: DiscreteZeroSumHJBIProblem,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    times = np.asarray(problem.time_grid.times, dtype=float)
    points = np.asarray(problem.spatial_grid.points, dtype=float)
    minimizer = np.asarray(problem.minimizer_actions, dtype=float)
    maximizer = np.asarray(problem.maximizer_actions, dtype=float)
    shape = (
        times.size - 1,
        points.size - 2,
        minimizer.size,
        maximizer.size,
    )
    drift = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)
    cost = np.empty(shape, dtype=float)
    maximum_courant = 0.0
    minimum_margin = 1.0
    dx = problem.spatial_grid.spacing
    for step, (time, duration) in enumerate(zip(times[:-1], np.diff(times), strict=True)):
        for point_index, state in enumerate(points[1:-1]):
            for minimizer_index, minimizer_action in enumerate(minimizer):
                for maximizer_index, maximizer_action in enumerate(maximizer):
                    drift_value = _callback_scalar(
                        problem.drift,
                        time,
                        state,
                        minimizer_action,
                        maximizer_action,
                        problem.args,
                        "drift",
                    )
                    diffusion_value = _callback_scalar(
                        problem.diffusion,
                        time,
                        state,
                        minimizer_action,
                        maximizer_action,
                        problem.args,
                        "diffusion",
                    )
                    variance_value = diffusion_value * diffusion_value
                    if not np.isfinite(variance_value):
                        raise ValueError(
                            "Squared diffusion must be finite on the declared grids."
                        )
                    index = (step, point_index, minimizer_index, maximizer_index)
                    drift[index] = drift_value
                    variance[index] = variance_value
                    cost[index] = _callback_scalar(
                        problem.running_cost,
                        time,
                        state,
                        minimizer_action,
                        maximizer_action,
                        problem.args,
                        "running_cost",
                    )
                    courant = duration * (
                        abs(drift_value) / dx + variance_value / (dx * dx)
                    )
                    maximum_courant = max(maximum_courant, courant)
                    minimum_margin = min(minimum_margin, 1.0 - courant)
    if minimum_margin < -32.0 * np.finfo(float).eps:
        raise ValueError(
            "The declared time and spatial grids violate the explicit monotone "
            "upwind-diffusion step condition."
        )
    return drift, variance, cost, maximum_courant, minimum_margin


def _hjbi_hamiltonian(
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
        + np.maximum(drift, 0.0) * forward[:, None, None]
        + np.minimum(drift, 0.0) * backward[:, None, None]
        + 0.5 * variance * second[:, None, None]
    )


def _ordered_action_value(
    hamiltonian: np.ndarray,
    order: HJBIActionOrder,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(hamiltonian.shape[0])
    if order == "min_max":
        maximized = np.max(hamiltonian, axis=2)
        minimizer_selector = np.argmin(maximized, axis=1)
        maximizer_selector = np.argmax(hamiltonian[rows, minimizer_selector, :], axis=1)
    else:
        minimized = np.min(hamiltonian, axis=1)
        maximizer_selector = np.argmax(minimized, axis=1)
        minimizer_selector = np.argmin(hamiltonian[rows, :, maximizer_selector], axis=1)
    value = hamiltonian[rows, minimizer_selector, maximizer_selector]
    return value, minimizer_selector, maximizer_selector


def _solve_hjbi_raw(problem: DiscreteZeroSumHJBIProblem, /) -> _RawHJBI:
    drift, variance, cost, maximum_courant, minimum_margin = _hjbi_coefficients(problem)
    times = np.asarray(problem.time_grid.times, dtype=float)
    terminal = np.asarray(problem.terminal_values, dtype=float)
    boundary = np.asarray(problem.boundary_values, dtype=float)
    value_shape = (times.size, problem.spatial_grid.num_points)
    selector_shape = (times.size - 1, value_shape[1] - 2)
    lower_values = np.empty(value_shape, dtype=float)
    upper_values = np.empty(value_shape, dtype=float)
    lower_minimizer = np.empty(selector_shape, dtype=np.int32)
    lower_maximizer = np.empty(selector_shape, dtype=np.int32)
    upper_minimizer = np.empty(selector_shape, dtype=np.int32)
    upper_maximizer = np.empty(selector_shape, dtype=np.int32)
    lower_values[-1] = terminal
    upper_values[-1] = terminal

    for step in range(times.size - 2, -1, -1):
        duration = times[step + 1] - times[step]
        lower_hamiltonian = _hjbi_hamiltonian(
            lower_values[step + 1],
            drift[step],
            variance[step],
            cost[step],
            problem.spatial_grid.spacing,
        )
        upper_hamiltonian = _hjbi_hamiltonian(
            upper_values[step + 1],
            drift[step],
            variance[step],
            cost[step],
            problem.spatial_grid.spacing,
        )
        lower_value, lower_u, lower_v = _ordered_action_value(
            lower_hamiltonian, problem.lower_order
        )
        upper_value, upper_u, upper_v = _ordered_action_value(
            upper_hamiltonian, problem.upper_order
        )
        lower_values[step, 1:-1] = lower_values[step + 1, 1:-1] + duration * lower_value
        upper_values[step, 1:-1] = upper_values[step + 1, 1:-1] + duration * upper_value
        lower_values[step, (0, value_shape[1] - 1)] = boundary[step]
        upper_values[step, (0, value_shape[1] - 1)] = boundary[step]
        lower_minimizer[step] = lower_u
        lower_maximizer[step] = lower_v
        upper_minimizer[step] = upper_u
        upper_maximizer[step] = upper_v

    lower_operator_residual = 0.0
    upper_operator_residual = 0.0
    lower_action_residual = 0.0
    upper_action_residual = 0.0
    for step, duration in enumerate(np.diff(times)):
        lower_hamiltonian = _hjbi_hamiltonian(
            lower_values[step + 1],
            drift[step],
            variance[step],
            cost[step],
            problem.spatial_grid.spacing,
        )
        upper_hamiltonian = _hjbi_hamiltonian(
            upper_values[step + 1],
            drift[step],
            variance[step],
            cost[step],
            problem.spatial_grid.spacing,
        )
        lower_expected, expected_lower_u, expected_lower_v = _ordered_action_value(
            lower_hamiltonian, problem.lower_order
        )
        upper_expected, expected_upper_u, expected_upper_v = _ordered_action_value(
            upper_hamiltonian, problem.upper_order
        )
        rows = np.arange(value_shape[1] - 2)
        lower_selected = lower_hamiltonian[
            rows, lower_minimizer[step], lower_maximizer[step]
        ]
        upper_selected = upper_hamiltonian[
            rows, upper_minimizer[step], upper_maximizer[step]
        ]
        lower_operator_residual = max(
            lower_operator_residual,
            float(
                np.max(
                    np.abs(
                        (lower_values[step, 1:-1] - lower_values[step + 1, 1:-1])
                        / duration
                        - lower_selected
                    )
                )
            ),
        )
        upper_operator_residual = max(
            upper_operator_residual,
            float(
                np.max(
                    np.abs(
                        (upper_values[step, 1:-1] - upper_values[step + 1, 1:-1])
                        / duration
                        - upper_selected
                    )
                )
            ),
        )
        lower_action_residual = max(
            lower_action_residual,
            float(np.max(np.abs(lower_selected - lower_expected))),
            float(np.max(np.abs(lower_minimizer[step] - expected_lower_u))),
            float(np.max(np.abs(lower_maximizer[step] - expected_lower_v))),
        )
        upper_action_residual = max(
            upper_action_residual,
            float(np.max(np.abs(upper_selected - upper_expected))),
            float(np.max(np.abs(upper_minimizer[step] - expected_upper_u))),
            float(np.max(np.abs(upper_maximizer[step] - expected_upper_v))),
        )

    lower_boundary = np.max(np.abs(lower_values[:, (0, value_shape[1] - 1)] - boundary))
    upper_boundary = np.max(np.abs(upper_values[:, (0, value_shape[1] - 1)] - boundary))
    boundary_residual = float(max(lower_boundary, upper_boundary))
    terminal_residual = float(
        max(
            np.max(np.abs(lower_values[-1] - terminal)),
            np.max(np.abs(upper_values[-1] - terminal)),
        )
    )
    finite = bool(np.all(np.isfinite(lower_values)) and np.all(np.isfinite(upper_values)))
    return _RawHJBI(
        lower_values,
        upper_values,
        lower_minimizer,
        lower_maximizer,
        upper_minimizer,
        upper_maximizer,
        boundary_residual,
        terminal_residual,
        lower_operator_residual,
        upper_operator_residual,
        lower_action_residual,
        upper_action_residual,
        maximum_courant,
        minimum_margin,
        finite,
    )


def _refined_hjbi_problem(
    problem: DiscreteZeroSumHJBIProblem,
    /,
) -> DiscreteZeroSumHJBIProblem:
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
    refined_time_grid = TimeGrid(
        refined_times,
        time_id=f"{problem.time_grid.time_id}/space-2-time-4",
    )
    return DiscreteZeroSumHJBIProblem(
        refined_grid,
        refined_time_grid,
        problem.minimizer_actions,
        problem.maximizer_actions,
        terminal,
        boundary,
        problem.drift,
        problem.diffusion,
        problem.running_cost,
        lower_order=problem.lower_order,
        upper_order=problem.upper_order,
        args=problem.args,
        corner_tolerance=problem.corner_tolerance,
        problem_id=f"{problem.problem_id}/space-2-time-4",
    )


def _hjbi_status(
    *,
    finite: bool,
    boundary_passed: bool,
    terminal_passed: bool,
    operator_passed: bool,
    action_passed: bool,
    canonical_saddle_orders: bool,
    refinement_passed: bool,
    isaacs_passed: bool,
) -> DiscreteHJBIStatus:
    if not finite:
        return DiscreteHJBIStatus.NONFINITE_DISCRETE_OUTPUT
    if not boundary_passed or not terminal_passed:
        return DiscreteHJBIStatus.BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE
    if not operator_passed:
        return DiscreteHJBIStatus.OPERATOR_RESIDUAL_TOO_LARGE
    if not action_passed:
        return DiscreteHJBIStatus.ACTION_ORDER_RESIDUAL_TOO_LARGE
    if not canonical_saddle_orders:
        return DiscreteHJBIStatus.NONCANONICAL_SADDLE_ACTION_ORDERS
    if not refinement_passed:
        return DiscreteHJBIStatus.REFINEMENT_GATE_FAILED
    if not isaacs_passed:
        return DiscreteHJBIStatus.ISAACS_GAP_EXCEEDS_TOLERANCE
    return DiscreteHJBIStatus.SUCCESS_DISCRETE_SADDLE_REFERENCE


def solve_discrete_hjbi_reference(
    problem: DiscreteZeroSumHJBIProblem,
    /,
    *,
    residual_tolerance: float = 1.0e-8,
    refinement_absolute_tolerance: float = 2.0e-2,
    refinement_relative_tolerance: float = 5.0e-2,
    isaacs_absolute_tolerance: float = 1.0e-8,
    isaacs_relative_tolerance: float = 1.0e-8,
) -> DiscreteZeroSumHJBIResult:
    """Compute both declared action orders and gate a discrete saddle result.

    The success label is earned only by the canonical lower ``max_min`` and upper
    ``min_max`` pair, finite discrete output, boundary and terminal reproduction,
    both operator/action-order residuals, nested-grid agreement, and the declared
    Isaacs-gap tolerances. No continuum or global statement is produced.
    """

    if not isinstance(problem, DiscreteZeroSumHJBIProblem):
        raise TypeError("problem must be a DiscreteZeroSumHJBIProblem.")
    residual_tolerance = _nonnegative_tolerance(residual_tolerance, "residual_tolerance")
    refinement_absolute_tolerance = _nonnegative_tolerance(
        refinement_absolute_tolerance, "refinement_absolute_tolerance"
    )
    refinement_relative_tolerance = _nonnegative_tolerance(
        refinement_relative_tolerance, "refinement_relative_tolerance"
    )
    isaacs_absolute_tolerance = _nonnegative_tolerance(
        isaacs_absolute_tolerance, "isaacs_absolute_tolerance"
    )
    isaacs_relative_tolerance = _nonnegative_tolerance(
        isaacs_relative_tolerance, "isaacs_relative_tolerance"
    )

    raw = _solve_hjbi_raw(problem)
    refined = _solve_hjbi_raw(_refined_hjbi_problem(problem))
    refined_lower_common = refined.lower_values[::4, ::2]
    refined_upper_common = refined.upper_values[::4, ::2]
    maximum_lower_difference = float(
        np.max(np.abs(raw.lower_values - refined_lower_common))
    )
    maximum_upper_difference = float(
        np.max(np.abs(raw.upper_values - refined_upper_common))
    )
    refinement_scale = float(
        max(
            np.max(np.abs(refined_lower_common)),
            np.max(np.abs(refined_upper_common)),
        )
    )
    refinement_threshold = (
        refinement_absolute_tolerance + refinement_relative_tolerance * refinement_scale
    )
    isaacs_gap = np.abs(raw.upper_values - raw.lower_values)
    maximum_isaacs_gap = float(np.max(isaacs_gap))
    isaacs_scale = float(
        max(np.max(np.abs(raw.lower_values)), np.max(np.abs(raw.upper_values)))
    )
    isaacs_threshold = (
        isaacs_absolute_tolerance + isaacs_relative_tolerance * isaacs_scale
    )

    maximum_boundary_residual = max(raw.boundary_residual, refined.boundary_residual)
    maximum_terminal_residual = max(raw.terminal_residual, refined.terminal_residual)
    maximum_lower_operator_residual = max(
        raw.lower_operator_residual, refined.lower_operator_residual
    )
    maximum_upper_operator_residual = max(
        raw.upper_operator_residual, refined.upper_operator_residual
    )
    maximum_lower_action_residual = max(
        raw.lower_action_residual, refined.lower_action_residual
    )
    maximum_upper_action_residual = max(
        raw.upper_action_residual, refined.upper_action_residual
    )
    maximum_courant = max(raw.maximum_courant, refined.maximum_courant)
    minimum_margin = min(raw.minimum_margin, refined.minimum_margin)
    finite = (
        raw.finite
        and refined.finite
        and bool(np.isfinite(maximum_lower_difference))
        and bool(np.isfinite(maximum_upper_difference))
        and bool(np.isfinite(maximum_isaacs_gap))
    )
    boundary_passed = maximum_boundary_residual <= residual_tolerance
    terminal_passed = maximum_terminal_residual <= residual_tolerance
    operator_passed = (
        max(maximum_lower_operator_residual, maximum_upper_operator_residual)
        <= residual_tolerance
    )
    action_passed = (
        max(maximum_lower_action_residual, maximum_upper_action_residual)
        <= residual_tolerance
    )
    refinement_passed = (
        max(maximum_lower_difference, maximum_upper_difference) <= refinement_threshold
    )
    isaacs_passed = maximum_isaacs_gap <= isaacs_threshold
    canonical_saddle_orders = (
        problem.lower_order == "max_min" and problem.upper_order == "min_max"
    )
    status = _hjbi_status(
        finite=finite,
        boundary_passed=boundary_passed,
        terminal_passed=terminal_passed,
        operator_passed=operator_passed,
        action_passed=action_passed,
        canonical_saddle_orders=canonical_saddle_orders,
        refinement_passed=refinement_passed,
        isaacs_passed=isaacs_passed,
    )
    evidence = DiscreteHJBIEvidence(
        maximum_boundary_residual=jnp.asarray(maximum_boundary_residual),
        maximum_terminal_residual=jnp.asarray(maximum_terminal_residual),
        maximum_lower_operator_residual=jnp.asarray(maximum_lower_operator_residual),
        maximum_upper_operator_residual=jnp.asarray(maximum_upper_operator_residual),
        maximum_lower_action_order_residual=jnp.asarray(maximum_lower_action_residual),
        maximum_upper_action_order_residual=jnp.asarray(maximum_upper_action_residual),
        maximum_lower_refinement_difference=jnp.asarray(maximum_lower_difference),
        maximum_upper_refinement_difference=jnp.asarray(maximum_upper_difference),
        refinement_threshold=jnp.asarray(refinement_threshold),
        maximum_isaacs_gap=jnp.asarray(maximum_isaacs_gap),
        isaacs_threshold=jnp.asarray(isaacs_threshold),
        maximum_courant_number=jnp.asarray(maximum_courant),
        minimum_monotonicity_margin=jnp.asarray(minimum_margin),
        finite=jnp.asarray(finite),
        boundary_passed=jnp.asarray(boundary_passed),
        terminal_passed=jnp.asarray(terminal_passed),
        operator_passed=jnp.asarray(operator_passed),
        action_orders_passed=jnp.asarray(action_passed),
        canonical_saddle_action_orders=jnp.asarray(canonical_saddle_orders),
        refinement_passed=jnp.asarray(refinement_passed),
        isaacs_gap_passed=jnp.asarray(isaacs_passed),
        method=_REFERENCE_METHOD,
        scope="declared-bounded-grid-discrete-residuals-only",
    )
    lower_u = jnp.asarray(raw.lower_minimizer_selectors)
    lower_v = jnp.asarray(raw.lower_maximizer_selectors)
    upper_u = jnp.asarray(raw.upper_minimizer_selectors)
    upper_v = jnp.asarray(raw.upper_maximizer_selectors)
    saddle = status == DiscreteHJBIStatus.SUCCESS_DISCRETE_SADDLE_REFERENCE
    return DiscreteZeroSumHJBIResult(
        spatial_grid=problem.spatial_grid,
        time_grid=problem.time_grid,
        minimizer_actions=problem.minimizer_actions,
        maximizer_actions=problem.maximizer_actions,
        lower_values=jnp.asarray(raw.lower_values),
        upper_values=jnp.asarray(raw.upper_values),
        lower_minimizer_selectors=lower_u,
        lower_maximizer_selectors=lower_v,
        upper_minimizer_selectors=upper_u,
        upper_maximizer_selectors=upper_v,
        lower_selected_minimizer_actions=problem.minimizer_actions[lower_u],
        lower_selected_maximizer_actions=problem.maximizer_actions[lower_v],
        upper_selected_minimizer_actions=problem.minimizer_actions[upper_u],
        upper_selected_maximizer_actions=problem.maximizer_actions[upper_v],
        isaacs_gap=jnp.asarray(isaacs_gap),
        evidence=evidence,
        saddle=jnp.asarray(saddle),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        lower_order=problem.lower_order,
        upper_order=problem.upper_order,
        problem_id=problem.problem_id,
        status_label=status.name,
        method=_REFERENCE_METHOD,
    )


def scalar_lq_hjbi_solution(
    time_grid: TimeGrid,
    /,
    *,
    terminal_weight: float,
    gamma: float,
) -> ScalarLQHJBISolution:
    """Return the corrected closed-form scalar terminal-cost LQ HJBI solution.

    For terminal time ``T`` and terminal weight ``q_f``, the denominator is
    ``D(t) = 1 + q_f * (1 - gamma**-2) * (T - t)``. The declared horizon is
    rejected exactly when ``gamma`` is nonpositive or this denominator loses
    strict positivity. In particular, well-posed horizons with ``gamma <= 1``
    remain supported.
    """

    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    gamma_value = float(gamma)
    terminal = float(terminal_weight)
    if not np.isfinite(gamma_value) or gamma_value <= 0.0:
        raise ValueError("gamma must be finite and strictly positive.")
    if not np.isfinite(terminal):
        raise ValueError("terminal_weight must be finite.")
    times = np.asarray(time_grid.times, dtype=float)
    denominator = 1.0 + terminal * (1.0 - gamma_value**-2) * (times[-1] - times)
    denominator_floor = (
        32.0
        * np.finfo(float).eps
        * np.maximum(
            1.0,
            np.abs(terminal * (1.0 - gamma_value**-2) * (times[-1] - times)),
        )
    )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= denominator_floor):
        raise ValueError(
            "The scalar LQ HJBI Riccati denominator must remain strictly positive "
            "over the declared horizon."
        )
    coefficient = terminal / denominator
    status = ScalarLQHJBIStatus.SUCCESS_ANALYTIC_SCALAR_LQ_HJBI
    return ScalarLQHJBISolution(
        time_grid=time_grid,
        denominator=jnp.asarray(denominator),
        value_coefficient=jnp.asarray(coefficient),
        minimizer_feedback_gain=jnp.asarray(-coefficient),
        maximizer_feedback_gain=jnp.asarray(coefficient / (gamma_value * gamma_value)),
        gamma=gamma_value,
        terminal_weight=terminal,
        well_posed=jnp.asarray(True),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        status_label=status.name,
    )


__all__ = [
    "DiscreteHJBIEvidence",
    "DiscreteHJBIStatus",
    "DiscreteZeroSumHJBIProblem",
    "DiscreteZeroSumHJBIResult",
    "HJBIActionOrder",
    "ScalarLQHJBISolution",
    "ScalarLQHJBIStatus",
    "scalar_lq_hjbi_solution",
    "solve_discrete_hjbi_reference",
]
