#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded one-dimensional finite-action impulse-QVI reference calculations."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics._grid import TimeGrid
from ._hjb import BoundedUniformGrid1D


_REFERENCE_METHOD = "bounded-uniform-1d-explicit-upwind-central-single-impulse"


class ImpulseQVIStatus(IntEnum):
    """Stable outcomes for the declared finite-grid QVI calculation."""

    SUCCESS_DISCRETE_QVI_REFERENCE = 0
    NONFINITE_DISCRETE_OUTPUT = 1
    BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE = 2
    COMPLEMENTARITY_RESIDUAL_TOO_LARGE = 3
    ACTION_SELECTION_RESIDUAL_TOO_LARGE = 4
    REFINEMENT_GATE_FAILED = 5


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _finite_real_array(value: ArrayLike, owner: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{owner} must be finite.")
    return array


def _nonnegative(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return resolved


class BoundedImpulseQVIProblem(StrictModule, NonTrainableState):
    """Scalar-state finite-action minimization QVI on a bounded uniform grid.

    A grid decision first computes the continuation Bellman value. The intervention
    operator then permits at most one declared impulse at that decision time, with
    post-intervention value interpolated from that continuation table. This is a
    finite-grid reference and does not claim a continuous-time viscosity solution.
    """

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    continuation_actions: Array
    impulse_actions: Array
    terminal_values: Array
    boundary_values: Array
    drift: Callable[[Array, Array, Array, Any], ArrayLike]
    diffusion: Callable[[Array, Array, Array, Any], ArrayLike]
    running_cost: Callable[[Array, Array, Array, Any], ArrayLike]
    intervention_state: Callable[[Array, Array, Array, Any], ArrayLike]
    intervention_cost: Callable[[Array, Array, Array, Any], ArrayLike]
    args: Any
    allow_negative_intervention_cost: bool = eqx.field(static=True)
    corner_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_grid: BoundedUniformGrid1D,
        time_grid: TimeGrid,
        continuation_actions: ArrayLike,
        impulse_actions: ArrayLike,
        terminal_values: ArrayLike,
        boundary_values: ArrayLike,
        drift: Callable[[Array, Array, Array, Any], ArrayLike],
        diffusion: Callable[[Array, Array, Array, Any], ArrayLike],
        running_cost: Callable[[Array, Array, Array, Any], ArrayLike],
        intervention_state: Callable[[Array, Array, Array, Any], ArrayLike],
        intervention_cost: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        args: Any = None,
        allow_negative_intervention_cost: bool = False,
        corner_tolerance: float = 0.0,
        problem_id: str,
    ):
        if not isinstance(spatial_grid, BoundedUniformGrid1D):
            raise TypeError("spatial_grid must be a BoundedUniformGrid1D.")
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        for owner, callback in (
            ("drift", drift),
            ("diffusion", diffusion),
            ("running_cost", running_cost),
            ("intervention_state", intervention_state),
            ("intervention_cost", intervention_cost),
        ):
            if not callable(callback):
                raise TypeError(f"{owner} must be callable.")
        continuation = _finite_real_array(continuation_actions, "continuation_actions")
        impulses = _finite_real_array(impulse_actions, "impulse_actions")
        if continuation.ndim != 1 or continuation.size == 0:
            raise ValueError("continuation_actions must be a nonempty rank-one grid.")
        if impulses.ndim != 1 or impulses.size == 0:
            raise ValueError("impulse_actions must be a nonempty rank-one grid.")
        terminal = _finite_real_array(terminal_values, "terminal_values")
        boundary = _finite_real_array(boundary_values, "boundary_values")
        if terminal.shape != (spatial_grid.num_points,):
            raise ValueError(
                "terminal_values must have shape "
                f"({spatial_grid.num_points},); got {terminal.shape}."
            )
        if boundary.shape != (time_grid.num_times, 2):
            raise ValueError(
                "boundary_values must have shape "
                f"({time_grid.num_times}, 2); got {boundary.shape}."
            )
        tolerance = _nonnegative(corner_tolerance, "corner_tolerance")
        if float(np.max(np.abs(boundary[-1] - terminal[[0, -1]]))) > tolerance:
            raise ValueError(
                "Terminal values and final-time boundary data are incompatible at "
                "the interval corners."
            )
        dtype = jnp.result_type(continuation, impulses, terminal, boundary, float)
        self.spatial_grid = spatial_grid
        self.time_grid = time_grid
        self.continuation_actions = jnp.asarray(continuation, dtype=dtype)
        self.impulse_actions = jnp.asarray(impulses, dtype=dtype)
        self.terminal_values = jnp.asarray(terminal, dtype=dtype)
        self.boundary_values = jnp.asarray(boundary, dtype=dtype)
        self.drift = drift
        self.diffusion = diffusion
        self.running_cost = running_cost
        self.intervention_state = intervention_state
        self.intervention_cost = intervention_cost
        self.args = args
        self.allow_negative_intervention_cost = bool(allow_negative_intervention_cost)
        self.corner_tolerance = tolerance
        self.problem_id = _identifier(problem_id, "problem_id")


class ImpulseQVIPlan(StrictModule, NonTrainableState):
    """Residual and nested-grid thresholds for a bounded QVI reference."""

    residual_tolerance: float = eqx.field(static=True)
    refinement_absolute_tolerance: float = eqx.field(static=True)
    refinement_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_tolerance: float = 1.0e-8,
        refinement_absolute_tolerance: float = 2.0e-2,
        refinement_relative_tolerance: float = 5.0e-2,
        plan_id: str,
    ):
        self.residual_tolerance = _nonnegative(residual_tolerance, "residual_tolerance")
        self.refinement_absolute_tolerance = _nonnegative(
            refinement_absolute_tolerance, "refinement_absolute_tolerance"
        )
        self.refinement_relative_tolerance = _nonnegative(
            refinement_relative_tolerance, "refinement_relative_tolerance"
        )
        self.plan_id = _identifier(plan_id, "plan_id")


class ImpulseQVIEvidence(StrictModule, NonTrainableState):
    """Complementarity, selector, monotonicity, and refinement evidence."""

    maximum_boundary_residual: Array
    maximum_terminal_residual: Array
    maximum_complementarity_residual: Array
    maximum_action_selection_residual: Array
    maximum_refinement_difference: Array
    refinement_threshold: Array
    maximum_courant_number: Array
    minimum_monotonicity_margin: Array
    finite: Array
    boundary_passed: Array
    terminal_passed: Array
    complementarity_passed: Array
    action_selection_passed: Array
    refinement_passed: Array
    method: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


class ImpulseQVIResult(StrictModule, NonTrainableState):
    """Discrete values, continuation/intervention tables, selectors, and evidence."""

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    values: Array
    continuation_values: Array
    intervention_values: Array
    continuation_selectors: Array
    intervention_selectors: Array
    selected_continuation_actions: Array
    selected_impulse_actions: Array
    intervention_region: Array
    continuation_gap: Array
    intervention_gap: Array
    evidence: ImpulseQVIEvidence
    successful: Array
    status: Array
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    status_label: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


class ImpulseQVIRefinementResult(StrictModule, NonTrainableState):
    """Primary result and its independently computed nested-grid comparison."""

    result: ImpulseQVIResult
    refined_spatial_grid: BoundedUniformGrid1D
    refined_time_grid: TimeGrid
    refined_values: Array
    common_grid_difference: Array
    passed: Array
    status: Array
    status_label: str = eqx.field(static=True)


class _RawQVI(NamedTuple):
    values: np.ndarray
    continuation_values: np.ndarray
    intervention_values: np.ndarray
    continuation_selectors: np.ndarray
    intervention_selectors: np.ndarray
    intervention_region: np.ndarray
    continuation_gap: np.ndarray
    intervention_gap: np.ndarray
    boundary_residual: float
    terminal_residual: float
    complementarity_residual: float
    action_selection_residual: float
    maximum_courant: float
    minimum_margin: float
    finite: bool


def _callback_scalar(
    callback: Callable,
    time: float,
    state: float,
    action: float,
    args: Any,
    owner: str,
    /,
) -> float:
    value = np.asarray(
        callback(jnp.asarray(time), jnp.asarray(state), jnp.asarray(action), args)
    )
    if value.shape != ():
        raise ValueError(f"{owner} must return a scalar for scalar inputs.")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{owner} must return a real scalar.")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{owner} must return finite values on the declared grids.")
    return scalar


def _coefficients(
    problem: BoundedImpulseQVIProblem,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    times = np.asarray(problem.time_grid.times, dtype=float)
    points = np.asarray(problem.spatial_grid.points, dtype=float)
    continuation_actions = np.asarray(problem.continuation_actions, dtype=float)
    impulse_actions = np.asarray(problem.impulse_actions, dtype=float)
    continuation_shape = (
        times.size - 1,
        points.size - 2,
        continuation_actions.size,
    )
    impulse_shape = (times.size - 1, points.size - 2, impulse_actions.size)
    drift = np.empty(continuation_shape, dtype=float)
    variance = np.empty(continuation_shape, dtype=float)
    running_cost = np.empty(continuation_shape, dtype=float)
    post_state = np.empty(impulse_shape, dtype=float)
    impulse_cost = np.empty(impulse_shape, dtype=float)
    maximum_courant = 0.0
    minimum_margin = 1.0
    dx = problem.spatial_grid.spacing
    for step, (time, duration) in enumerate(zip(times[:-1], np.diff(times), strict=True)):
        for point_index, state in enumerate(points[1:-1]):
            for action_index, action in enumerate(continuation_actions):
                drift_value = _callback_scalar(
                    problem.drift, time, state, action, problem.args, "drift"
                )
                diffusion_value = _callback_scalar(
                    problem.diffusion, time, state, action, problem.args, "diffusion"
                )
                variance_value = diffusion_value * diffusion_value
                if not np.isfinite(variance_value):
                    raise ValueError("Squared diffusion must be finite.")
                drift[step, point_index, action_index] = drift_value
                variance[step, point_index, action_index] = variance_value
                running_cost[step, point_index, action_index] = _callback_scalar(
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
            for action_index, action in enumerate(impulse_actions):
                mapped = _callback_scalar(
                    problem.intervention_state,
                    time,
                    state,
                    action,
                    problem.args,
                    "intervention_state",
                )
                if mapped < points[0] or mapped > points[-1]:
                    raise ValueError(
                        "intervention_state must remain inside the bounded spatial grid."
                    )
                cost = _callback_scalar(
                    problem.intervention_cost,
                    time,
                    state,
                    action,
                    problem.args,
                    "intervention_cost",
                )
                if not problem.allow_negative_intervention_cost and cost < 0.0:
                    raise ValueError(
                        "intervention_cost must be nonnegative unless the explicit "
                        "reward convention is enabled."
                    )
                post_state[step, point_index, action_index] = mapped
                impulse_cost[step, point_index, action_index] = cost
    if minimum_margin < -32.0 * np.finfo(float).eps:
        raise ValueError(
            "The declared time and spatial grids violate the explicit monotone "
            "upwind-diffusion step condition."
        )
    return (
        drift,
        variance,
        running_cost,
        post_state,
        impulse_cost,
        maximum_courant,
        minimum_margin,
    )


def _hamiltonian(
    next_values: np.ndarray,
    drift: np.ndarray,
    variance: np.ndarray,
    running_cost: np.ndarray,
    spacing: float,
    /,
) -> np.ndarray:
    backward = (next_values[1:-1] - next_values[:-2]) / spacing
    forward = (next_values[2:] - next_values[1:-1]) / spacing
    second = (next_values[2:] - 2.0 * next_values[1:-1] + next_values[:-2]) / (
        spacing * spacing
    )
    return (
        running_cost
        + np.maximum(drift, 0.0) * forward[:, None]
        + np.minimum(drift, 0.0) * backward[:, None]
        + 0.5 * variance * second[:, None]
    )


def _solve_raw(problem: BoundedImpulseQVIProblem, /) -> _RawQVI:
    (
        drift,
        variance,
        running_cost,
        post_state,
        impulse_cost,
        maximum_courant,
        minimum_margin,
    ) = _coefficients(problem)
    times = np.asarray(problem.time_grid.times, dtype=float)
    points = np.asarray(problem.spatial_grid.points, dtype=float)
    boundary = np.asarray(problem.boundary_values, dtype=float)
    terminal = np.asarray(problem.terminal_values, dtype=float)
    shape = (times.size, points.size)
    decision_shape = (times.size - 1, points.size)
    values = np.empty(shape, dtype=float)
    continuation_values = np.empty(decision_shape, dtype=float)
    intervention_values = np.empty(decision_shape, dtype=float)
    continuation_selectors = np.full(decision_shape, -1, dtype=np.int32)
    intervention_selectors = np.full(decision_shape, -1, dtype=np.int32)
    intervention_region = np.zeros(decision_shape, dtype=bool)
    values[-1] = terminal
    for step in range(times.size - 2, -1, -1):
        duration = times[step + 1] - times[step]
        hamiltonian = _hamiltonian(
            values[step + 1],
            drift[step],
            variance[step],
            running_cost[step],
            problem.spatial_grid.spacing,
        )
        continuation_selector = np.argmin(hamiltonian, axis=-1)
        selected_hamiltonian = np.take_along_axis(
            hamiltonian, continuation_selector[:, None], axis=-1
        )[:, 0]
        continuation = np.empty((points.size,), dtype=float)
        continuation[1:-1] = values[step + 1, 1:-1] + duration * selected_hamiltonian
        continuation[0] = boundary[step, 0]
        continuation[-1] = boundary[step, 1]
        candidates = impulse_cost[step] + np.stack(
            [
                np.interp(post_state[step, :, action], points, continuation)
                for action in range(problem.impulse_actions.size)
            ],
            axis=-1,
        )
        intervention_selector = np.argmin(candidates, axis=-1)
        intervention = np.empty((points.size,), dtype=float)
        intervention[1:-1] = np.take_along_axis(
            candidates, intervention_selector[:, None], axis=-1
        )[:, 0]
        intervention[0] = continuation[0]
        intervention[-1] = continuation[-1]
        values[step] = np.minimum(continuation, intervention)
        region = intervention < continuation
        continuation_values[step] = continuation
        intervention_values[step] = intervention
        continuation_selectors[step, 1:-1] = continuation_selector
        intervention_selectors[step, 1:-1] = intervention_selector
        intervention_region[step] = region

    continuation_gap = continuation_values - values[:-1]
    intervention_gap = intervention_values - values[:-1]
    negative_gap = max(
        float(np.max(np.maximum(-continuation_gap, 0.0))),
        float(np.max(np.maximum(-intervention_gap, 0.0))),
    )
    complementarity = float(np.max(np.abs(continuation_gap * intervention_gap)))
    complementarity_residual = max(negative_gap, complementarity)
    action_residual = 0.0
    for step in range(times.size - 1):
        duration = times[step + 1] - times[step]
        hamiltonian = _hamiltonian(
            values[step + 1],
            drift[step],
            variance[step],
            running_cost[step],
            problem.spatial_grid.spacing,
        )
        selected_continuation = np.take_along_axis(
            hamiltonian,
            continuation_selectors[step, 1:-1, None],
            axis=-1,
        )[:, 0]
        expected_continuation = values[step + 1, 1:-1] + duration * np.min(
            hamiltonian, axis=-1
        )
        continuation_residual = float(
            np.max(np.abs(continuation_values[step, 1:-1] - expected_continuation))
        )
        selector_residual = float(
            np.max(np.abs(selected_continuation - np.min(hamiltonian, axis=-1)))
        )
        candidates = impulse_cost[step] + np.stack(
            [
                np.interp(
                    post_state[step, :, action],
                    points,
                    continuation_values[step],
                )
                for action in range(problem.impulse_actions.size)
            ],
            axis=-1,
        )
        selected_intervention = np.take_along_axis(
            candidates,
            intervention_selectors[step, 1:-1, None],
            axis=-1,
        )[:, 0]
        intervention_residual = float(
            np.max(np.abs(selected_intervention - np.min(candidates, axis=-1)))
        )
        action_residual = max(
            action_residual,
            continuation_residual,
            selector_residual,
            intervention_residual,
        )
    boundary_residual = float(
        np.max(np.abs(values[:, (0, values.shape[1] - 1)] - boundary))
    )
    terminal_residual = float(np.max(np.abs(values[-1] - terminal)))
    finite = bool(
        np.all(np.isfinite(values))
        and np.all(np.isfinite(continuation_values))
        and np.all(np.isfinite(intervention_values))
    )
    return _RawQVI(
        values,
        continuation_values,
        intervention_values,
        continuation_selectors,
        intervention_selectors,
        intervention_region,
        continuation_gap,
        intervention_gap,
        boundary_residual,
        terminal_residual,
        complementarity_residual,
        action_residual,
        maximum_courant,
        minimum_margin,
        finite,
    )


def _refined_problem(problem: BoundedImpulseQVIProblem, /) -> BoundedImpulseQVIProblem:
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
    return BoundedImpulseQVIProblem(
        refined_grid,
        TimeGrid(
            refined_times,
            time_id=f"{problem.time_grid.time_id}/space-2-time-4",
        ),
        problem.continuation_actions,
        problem.impulse_actions,
        terminal,
        boundary,
        problem.drift,
        problem.diffusion,
        problem.running_cost,
        problem.intervention_state,
        problem.intervention_cost,
        args=problem.args,
        allow_negative_intervention_cost=problem.allow_negative_intervention_cost,
        corner_tolerance=problem.corner_tolerance,
        problem_id=f"{problem.problem_id}/space-2-time-4",
    )


def _status(
    *,
    finite: bool,
    boundary_passed: bool,
    terminal_passed: bool,
    complementarity_passed: bool,
    action_selection_passed: bool,
    refinement_passed: bool,
) -> ImpulseQVIStatus:
    if not finite:
        return ImpulseQVIStatus.NONFINITE_DISCRETE_OUTPUT
    if not boundary_passed or not terminal_passed:
        return ImpulseQVIStatus.BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE
    if not complementarity_passed:
        return ImpulseQVIStatus.COMPLEMENTARITY_RESIDUAL_TOO_LARGE
    if not action_selection_passed:
        return ImpulseQVIStatus.ACTION_SELECTION_RESIDUAL_TOO_LARGE
    if not refinement_passed:
        return ImpulseQVIStatus.REFINEMENT_GATE_FAILED
    return ImpulseQVIStatus.SUCCESS_DISCRETE_QVI_REFERENCE


def _solve_with_refinement(
    problem: BoundedImpulseQVIProblem,
    plan: ImpulseQVIPlan,
    /,
) -> tuple[ImpulseQVIResult, BoundedImpulseQVIProblem, _RawQVI, np.ndarray]:
    if not isinstance(problem, BoundedImpulseQVIProblem):
        raise TypeError("problem must be a BoundedImpulseQVIProblem.")
    if not isinstance(plan, ImpulseQVIPlan):
        raise TypeError("plan must be an ImpulseQVIPlan.")
    raw = _solve_raw(problem)
    refined_problem = _refined_problem(problem)
    refined = _solve_raw(refined_problem)
    refined_common = refined.values[::4, ::2]
    common_difference = np.abs(raw.values - refined_common)
    maximum_difference = float(np.max(common_difference))
    refinement_scale = float(np.max(np.abs(refined_common)))
    refinement_threshold = (
        plan.refinement_absolute_tolerance
        + plan.refinement_relative_tolerance * refinement_scale
    )
    boundary_residual = max(raw.boundary_residual, refined.boundary_residual)
    terminal_residual = max(raw.terminal_residual, refined.terminal_residual)
    complementarity_residual = max(
        raw.complementarity_residual, refined.complementarity_residual
    )
    action_residual = max(
        raw.action_selection_residual, refined.action_selection_residual
    )
    maximum_courant = max(raw.maximum_courant, refined.maximum_courant)
    minimum_margin = min(raw.minimum_margin, refined.minimum_margin)
    finite = raw.finite and refined.finite and np.isfinite(maximum_difference)
    boundary_passed = boundary_residual <= plan.residual_tolerance
    terminal_passed = terminal_residual <= plan.residual_tolerance
    complementarity_passed = complementarity_residual <= plan.residual_tolerance
    action_selection_passed = action_residual <= plan.residual_tolerance
    refinement_passed = maximum_difference <= refinement_threshold
    status = _status(
        finite=bool(finite),
        boundary_passed=boundary_passed,
        terminal_passed=terminal_passed,
        complementarity_passed=complementarity_passed,
        action_selection_passed=action_selection_passed,
        refinement_passed=refinement_passed,
    )
    evidence = ImpulseQVIEvidence(
        maximum_boundary_residual=jnp.asarray(boundary_residual),
        maximum_terminal_residual=jnp.asarray(terminal_residual),
        maximum_complementarity_residual=jnp.asarray(complementarity_residual),
        maximum_action_selection_residual=jnp.asarray(action_residual),
        maximum_refinement_difference=jnp.asarray(maximum_difference),
        refinement_threshold=jnp.asarray(refinement_threshold),
        maximum_courant_number=jnp.asarray(maximum_courant),
        minimum_monotonicity_margin=jnp.asarray(minimum_margin),
        finite=jnp.asarray(finite),
        boundary_passed=jnp.asarray(boundary_passed),
        terminal_passed=jnp.asarray(terminal_passed),
        complementarity_passed=jnp.asarray(complementarity_passed),
        action_selection_passed=jnp.asarray(action_selection_passed),
        refinement_passed=jnp.asarray(refinement_passed),
        method=_REFERENCE_METHOD,
        scope="declared-bounded-grid-single-impulse-discrete-residuals-only",
    )
    continuation_selectors = jnp.asarray(raw.continuation_selectors)
    intervention_selectors = jnp.asarray(raw.intervention_selectors)
    safe_continuation = jnp.maximum(continuation_selectors, 0)
    safe_intervention = jnp.maximum(intervention_selectors, 0)
    result = ImpulseQVIResult(
        spatial_grid=problem.spatial_grid,
        time_grid=problem.time_grid,
        values=jnp.asarray(raw.values),
        continuation_values=jnp.asarray(raw.continuation_values),
        intervention_values=jnp.asarray(raw.intervention_values),
        continuation_selectors=continuation_selectors,
        intervention_selectors=intervention_selectors,
        selected_continuation_actions=jnp.where(
            continuation_selectors >= 0,
            problem.continuation_actions[safe_continuation],
            0.0,
        ),
        selected_impulse_actions=jnp.where(
            intervention_selectors >= 0,
            problem.impulse_actions[safe_intervention],
            0.0,
        ),
        intervention_region=jnp.asarray(raw.intervention_region),
        continuation_gap=jnp.asarray(raw.continuation_gap),
        intervention_gap=jnp.asarray(raw.intervention_gap),
        evidence=evidence,
        successful=jnp.asarray(status == ImpulseQVIStatus.SUCCESS_DISCRETE_QVI_REFERENCE),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        status_label=status.name,
        method=_REFERENCE_METHOD,
    )
    return result, refined_problem, refined, common_difference


def solve_impulse_qvi_reference(
    problem: BoundedImpulseQVIProblem,
    plan: ImpulseQVIPlan,
    /,
) -> ImpulseQVIResult:
    """Solve and gate one bounded finite-action discrete QVI reference."""

    return _solve_with_refinement(problem, plan)[0]


def refine_impulse_qvi_reference(
    problem: BoundedImpulseQVIProblem,
    plan: ImpulseQVIPlan,
    /,
) -> ImpulseQVIRefinementResult:
    """Expose the nested table and pointwise common-grid comparison."""

    result, refined_problem, refined, difference = _solve_with_refinement(problem, plan)
    return ImpulseQVIRefinementResult(
        result=result,
        refined_spatial_grid=refined_problem.spatial_grid,
        refined_time_grid=refined_problem.time_grid,
        refined_values=jnp.asarray(refined.values),
        common_grid_difference=jnp.asarray(difference),
        passed=result.evidence.refinement_passed,
        status=result.status,
        status_label=result.status_label,
    )


__all__ = [
    "BoundedImpulseQVIProblem",
    "ImpulseQVIEvidence",
    "ImpulseQVIPlan",
    "ImpulseQVIRefinementResult",
    "ImpulseQVIResult",
    "ImpulseQVIStatus",
    "refine_impulse_qvi_reference",
    "solve_impulse_qvi_reference",
]
