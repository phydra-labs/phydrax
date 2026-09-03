#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded one-dimensional nonzero-sum coupled-HJB reference calculations."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
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


CoupledHJBUpdate: TypeAlias = Literal["jacobi", "gauss_seidel"]
LOCAL_FEEDBACK_FIXED_POINT = "LOCAL_FEEDBACK_FIXED_POINT"
_REFERENCE_METHOD = "bounded-uniform-1d-explicit-upwind-central-coupled-policy-iteration"
_SELECTOR_ID = "own-hamiltonian-argmin"
_TIE_BREAK_ID = "lowest-declared-action-index"
_BRANCH_SELECTION_ID = "first-success-then-smallest-fixed-point-residual"


class DiscreteCoupledHJBStatus(IntEnum):
    """Stable outcomes for the finite coupled-HJB reference calculation."""

    SUCCESS_LOCAL_FEEDBACK_FIXED_POINT = 0
    MAXIMUM_POLICY_ITERATIONS = 1
    NONFINITE_DISCRETE_OUTPUT = 2
    BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE = 3
    POLICY_EVALUATION_RESIDUAL_TOO_LARGE = 4
    OWN_ACTION_HAMILTONIAN_GAP_TOO_LARGE = 5
    REFINEMENT_GATE_FAILED = 6


class DiscreteCoupledHJBProblem(StrictModule, NonTrainableState):
    """A bounded scalar-state finite-action coupled-HJB problem.

    Each coefficient callback receives
    ``(player, time, state, joint_action, args)`` and returns a finite real
    scalar. ``joint_action`` contains one scalar physical action per player,
    simultaneously. The diffusion callback returns an amplitude; its square is
    used in the player's linear policy-evaluation generator. Boundary columns
    are ordered lower then upper.
    """

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    player_actions: tuple[Array, ...]
    terminal_values: Array
    boundary_values: Array
    drift: Callable[[int, Array, Array, Array, Any], ArrayLike]
    diffusion: Callable[[int, Array, Array, Array, Any], ArrayLike]
    running_cost: Callable[[int, Array, Array, Array, Any], ArrayLike]
    args: Any
    corner_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_grid: BoundedUniformGrid1D,
        time_grid: TimeGrid,
        player_actions: Sequence[ArrayLike],
        terminal_values: ArrayLike,
        boundary_values: ArrayLike,
        drift: Callable[[int, Array, Array, Array, Any], ArrayLike],
        diffusion: Callable[[int, Array, Array, Array, Any], ArrayLike],
        running_cost: Callable[[int, Array, Array, Array, Any], ArrayLike],
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
        tolerance = _nonnegative_tolerance(corner_tolerance, "corner_tolerance")

        action_inputs = tuple(player_actions)
        if not action_inputs:
            raise ValueError("player_actions must contain at least one player.")
        actions: list[np.ndarray] = []
        for player, action_input in enumerate(action_inputs):
            action_array = _finite_real_array(action_input, f"player_actions[{player}]")
            if action_array.ndim != 1 or action_array.size == 0:
                raise ValueError("Each player action grid must be nonempty and rank one.")
            actions.append(action_array)

        players = len(actions)
        terminal = _finite_real_array(terminal_values, "terminal_values")
        expected_terminal = (players, spatial_grid.num_points)
        if terminal.shape != expected_terminal:
            raise ValueError(
                "terminal_values must have shape "
                f"{expected_terminal}; got {terminal.shape}."
            )
        boundary = _finite_real_array(boundary_values, "boundary_values")
        expected_boundary = (players, time_grid.num_times, 2)
        if boundary.shape != expected_boundary:
            raise ValueError(
                "boundary_values must have shape "
                f"{expected_boundary}; got {boundary.shape}."
            )
        corner_residual = float(
            np.max(np.abs(boundary[:, -1] - terminal[:, (0, terminal.shape[1] - 1)]))
        )
        if corner_residual > tolerance:
            raise ValueError(
                "Terminal values and final-time boundary data are incompatible at "
                "the interval corners."
            )

        dtype = jnp.result_type(*actions, terminal, boundary, float)
        self.spatial_grid = spatial_grid
        self.time_grid = time_grid
        self.player_actions = tuple(
            jnp.asarray(action, dtype=dtype) for action in actions
        )
        self.terminal_values = jnp.asarray(terminal, dtype=dtype)
        self.boundary_values = jnp.asarray(boundary, dtype=dtype)
        self.drift = drift
        self.diffusion = diffusion
        self.running_cost = running_cost
        self.args = args
        self.corner_tolerance = tolerance
        self.problem_id = identifier

    @property
    def num_players(self) -> int:
        return len(self.player_actions)


class CoupledHJBPolicyIterationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity policy-iteration choices and local stopping tolerance.

    Damping convexly relaxes each categorical finite-action feedback policy
    toward its deterministic best response; it never interpolates the declared
    physical action values.
    """

    maximum_iterations: int = eqx.field(static=True)
    fixed_point_tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    update: CoupledHJBUpdate = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int,
        fixed_point_tolerance: float,
        damping: float = 1.0,
        update: CoupledHJBUpdate = "jacobi",
        plan_id: str,
    ):
        if isinstance(maximum_iterations, (bool, np.bool_)):
            raise TypeError("maximum_iterations must be an integer.")
        capacity = operator.index(maximum_iterations)
        if capacity <= 0:
            raise ValueError("maximum_iterations must be positive.")
        tolerance = _nonnegative_tolerance(fixed_point_tolerance, "fixed_point_tolerance")
        damping_value = float(damping)
        if not np.isfinite(damping_value) or not 0.0 < damping_value <= 1.0:
            raise ValueError("damping must be finite and in (0, 1].")
        update_name = str(update)
        if update_name not in ("jacobi", "gauss_seidel"):
            raise ValueError("update must be 'jacobi' or 'gauss_seidel'.")
        identifier = str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.maximum_iterations = capacity
        self.fixed_point_tolerance = tolerance
        self.damping = damping_value
        self.update = update_name
        self.plan_id = identifier


class CoupledHJBBranchEvidence(StrictModule, NonTrainableState):
    """Fixed-capacity iteration records and terminal evidence for every start."""

    initial_action_selectors: Array
    final_action_selectors: Array
    final_best_response_selectors: Array
    action_selector_history: Array
    policy_probability_history: tuple[Array, ...]
    fixed_point_residual_history: Array
    update_residual_history: Array
    own_action_hamiltonian_gap_history: Array
    selector_change_history: Array
    iteration_validity_history: Array
    iterations: Array
    converged: Array
    refined_converged: Array
    statuses: Array
    final_fixed_point_residuals: Array
    maximum_fixed_point_residuals: Array
    maximum_boundary_residuals: Array
    maximum_terminal_residuals: Array
    maximum_policy_evaluation_residuals: Array
    maximum_own_action_hamiltonian_gaps: Array
    maximum_refinement_differences: Array
    refinement_thresholds: Array
    maximum_tie_counts: Array
    finite: Array
    maximum_between_branch_value_difference: Array
    branch_dependence_detected: Array
    branch_ids: tuple[str, ...] = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    branch_selection_id: str = eqx.field(static=True)


class DiscreteCoupledHJBEvidence(StrictModule, NonTrainableState):
    """Local fixed-point, PDE, Hamiltonian, boundary, and refinement evidence."""

    maximum_boundary_residual: Array
    maximum_terminal_residual: Array
    maximum_policy_evaluation_residual: Array
    maximum_own_action_hamiltonian_gap: Array
    maximum_fixed_point_residual: Array
    maximum_refinement_difference: Array
    refinement_threshold: Array
    maximum_courant_number: Array
    minimum_monotonicity_margin: Array
    maximum_tie_count: Array
    finite: Array
    boundary_passed: Array
    terminal_passed: Array
    policy_evaluation_passed: Array
    own_action_hamiltonian_gap_passed: Array
    fixed_point_passed: Array
    refinement_passed: Array
    branch: CoupledHJBBranchEvidence
    method: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


class DiscreteCoupledHJBResult(StrictModule, NonTrainableState):
    """One selected local branch plus complete evidence for every supplied start.

    ``policy_probabilities`` is the authoritative damped categorical policy.
    ``action_selectors`` and ``selected_actions`` are its deterministic
    lowest-index modes, which coincide with the physical feedback policy at an
    exact successful fixed point.
    """

    spatial_grid: BoundedUniformGrid1D
    time_grid: TimeGrid
    refined_spatial_grid: BoundedUniformGrid1D
    refined_time_grid: TimeGrid
    player_actions: tuple[Array, ...]
    joint_action_profiles: Array
    values: Array
    refined_values: Array
    common_grid_difference: Array
    action_selectors: Array
    selected_actions: Array
    policy_probabilities: tuple[Array, ...]
    best_response_selectors: Array
    player_policy_evaluation_residuals: Array
    own_action_hamiltonian_gaps: Array
    branch_values: Array
    branch_action_selectors: Array
    branch_policy_probabilities: tuple[Array, ...]
    branch_player_policy_evaluation_residuals: Array
    branch_own_action_hamiltonian_gaps: Array
    evidence: DiscreteCoupledHJBEvidence
    selected_branch: Array
    successful: Array
    local_feedback_fixed_point: Array
    status: Array
    update: CoupledHJBUpdate = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    fixed_point_tolerance: float = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    selected_branch_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    status_label: str = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)
    tie_break_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    candidate_evaluation_only: bool = eqx.field(static=True)
    viscosity_solution_claimed: bool = eqx.field(static=True)
    unique_solution_claimed: bool = eqx.field(static=True)
    global_nash_equilibrium_claimed: bool = eqx.field(static=True)


class _CoefficientTable(NamedTuple):
    profile_indices: np.ndarray
    profiles: np.ndarray
    drift: np.ndarray
    variance: np.ndarray
    cost: np.ndarray
    maximum_courant: float
    minimum_margin: float


class _RawBranchSet(NamedTuple):
    values: np.ndarray
    probabilities: tuple[np.ndarray, ...]
    selectors: np.ndarray
    best_response_selectors: np.ndarray
    tie_counts: np.ndarray
    fixed_point_residuals: np.ndarray
    boundary_residuals: np.ndarray
    terminal_residuals: np.ndarray
    policy_evaluation_residuals: np.ndarray
    player_policy_evaluation_residuals: np.ndarray
    own_action_gaps: np.ndarray
    own_action_gap_tables: np.ndarray
    finite: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    selector_history: np.ndarray
    probability_history: tuple[np.ndarray, ...]
    fixed_point_residual_history: np.ndarray
    update_residual_history: np.ndarray
    gap_history: np.ndarray
    selector_change_history: np.ndarray
    iteration_validity_history: np.ndarray


def _callback_scalar(
    callback: Callable,
    player: int,
    time: float,
    state: float,
    joint_action: np.ndarray,
    args: Any,
    name: str,
    /,
) -> float:
    value = np.asarray(
        callback(
            player,
            jnp.asarray(time),
            jnp.asarray(state),
            jnp.asarray(joint_action),
            args,
        )
    )
    if value.shape != ():
        raise ValueError(f"{name} must return a scalar for scalar grid inputs.")
    if np.issubdtype(value.dtype, np.complexfloating):
        raise TypeError(f"{name} must return a real scalar.")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite on the declared grids and profiles.")
    return scalar


def _coefficient_table(problem: DiscreteCoupledHJBProblem, /) -> _CoefficientTable:
    actions = tuple(np.asarray(action, dtype=float) for action in problem.player_actions)
    action_counts = tuple(action.size for action in actions)
    profile_indices = (
        np.indices(action_counts, dtype=np.int32).reshape(problem.num_players, -1).T
    )
    profiles = np.column_stack(
        tuple(
            actions[player][profile_indices[:, player]]
            for player in range(problem.num_players)
        )
    )
    times = np.asarray(problem.time_grid.times, dtype=float)
    points = np.asarray(problem.spatial_grid.points, dtype=float)
    shape = (
        problem.num_players,
        times.size - 1,
        points.size - 2,
        profile_indices.shape[0],
    )
    drift = np.empty(shape, dtype=float)
    variance = np.empty(shape, dtype=float)
    cost = np.empty(shape, dtype=float)
    maximum_courant = 0.0
    minimum_margin = 1.0
    spacing = problem.spatial_grid.spacing
    for player in range(problem.num_players):
        for step, (time, duration) in enumerate(
            zip(times[:-1], np.diff(times), strict=True)
        ):
            for point_index, state in enumerate(points[1:-1]):
                for profile_index, profile in enumerate(profiles):
                    drift_value = _callback_scalar(
                        problem.drift,
                        player,
                        time,
                        state,
                        profile,
                        problem.args,
                        "drift",
                    )
                    diffusion_value = _callback_scalar(
                        problem.diffusion,
                        player,
                        time,
                        state,
                        profile,
                        problem.args,
                        "diffusion",
                    )
                    variance_value = diffusion_value * diffusion_value
                    if not np.isfinite(variance_value):
                        raise ValueError(
                            "Squared diffusion must be finite on the declared grids "
                            "and profiles."
                        )
                    drift[player, step, point_index, profile_index] = drift_value
                    variance[player, step, point_index, profile_index] = variance_value
                    cost[player, step, point_index, profile_index] = _callback_scalar(
                        problem.running_cost,
                        player,
                        time,
                        state,
                        profile,
                        problem.args,
                        "running_cost",
                    )
                    courant = duration * (
                        abs(drift_value) / spacing + variance_value / (spacing * spacing)
                    )
                    maximum_courant = max(maximum_courant, courant)
                    minimum_margin = min(minimum_margin, 1.0 - courant)
    if minimum_margin < -32.0 * np.finfo(float).eps:
        raise ValueError(
            "The declared time and spatial grids violate the explicit monotone "
            "upwind-diffusion step condition for at least one player and joint "
            "action profile."
        )
    return _CoefficientTable(
        profile_indices,
        profiles,
        drift,
        variance,
        cost,
        maximum_courant,
        minimum_margin,
    )


def _joint_profile_weights(
    probabilities: tuple[np.ndarray, ...],
    profile_indices: np.ndarray,
    step: int,
    /,
) -> np.ndarray:
    points = probabilities[0].shape[1]
    weights = np.ones((points, profile_indices.shape[0]), dtype=float)
    for player, probability in enumerate(probabilities):
        weights *= probability[step][:, profile_indices[:, player]]
    return weights


def _profile_hamiltonian(
    next_values: np.ndarray,
    drift: np.ndarray,
    variance: np.ndarray,
    cost: np.ndarray,
    spacing: float,
    /,
) -> np.ndarray:
    backward = (next_values[1:-1] - next_values[:-2]) / spacing
    forward = (next_values[2:] - next_values[1:-1]) / spacing
    second = (next_values[2:] - 2.0 * next_values[1:-1] + next_values[:-2]) / (
        spacing * spacing
    )
    return (
        cost
        + np.maximum(drift, 0.0) * forward[:, None]
        + np.minimum(drift, 0.0) * backward[:, None]
        + 0.5 * variance * second[:, None]
    )


def _evaluate_player(
    problem: DiscreteCoupledHJBProblem,
    coefficients: _CoefficientTable,
    probabilities: tuple[np.ndarray, ...],
    player: int,
    /,
) -> np.ndarray:
    times = np.asarray(problem.time_grid.times, dtype=float)
    boundary = np.asarray(problem.boundary_values[player], dtype=float)
    values = np.empty((times.size, problem.spatial_grid.num_points), dtype=float)
    values[-1] = np.asarray(problem.terminal_values[player], dtype=float)
    for step in range(times.size - 2, -1, -1):
        profile_hamiltonian = _profile_hamiltonian(
            values[step + 1],
            coefficients.drift[player, step],
            coefficients.variance[player, step],
            coefficients.cost[player, step],
            problem.spatial_grid.spacing,
        )
        weights = _joint_profile_weights(
            probabilities, coefficients.profile_indices, step
        )
        expected_hamiltonian = np.sum(weights * profile_hamiltonian, axis=-1)
        values[step, 1:-1] = (
            values[step + 1, 1:-1]
            + (times[step + 1] - times[step]) * expected_hamiltonian
        )
        values[step, 0] = boundary[step, 0]
        values[step, -1] = boundary[step, 1]
    return values


def _evaluate_policy(
    problem: DiscreteCoupledHJBProblem,
    coefficients: _CoefficientTable,
    probabilities: tuple[np.ndarray, ...],
    /,
) -> np.ndarray:
    return np.stack(
        tuple(
            _evaluate_player(problem, coefficients, probabilities, player)
            for player in range(problem.num_players)
        )
    )


def _own_hamiltonians(
    problem: DiscreteCoupledHJBProblem,
    coefficients: _CoefficientTable,
    probabilities: tuple[np.ndarray, ...],
    player: int,
    values: np.ndarray,
    /,
) -> np.ndarray:
    steps = problem.time_grid.num_times - 1
    interior = problem.spatial_grid.num_points - 2
    action_count = int(problem.player_actions[player].shape[0])
    own = np.empty((steps, interior, action_count), dtype=float)
    for step in range(steps):
        profile_hamiltonian = _profile_hamiltonian(
            values[step + 1],
            coefficients.drift[player, step],
            coefficients.variance[player, step],
            coefficients.cost[player, step],
            problem.spatial_grid.spacing,
        )
        opponents_weight = np.ones_like(profile_hamiltonian)
        for opponent, probability in enumerate(probabilities):
            if opponent != player:
                opponents_weight *= probability[step][
                    :, coefficients.profile_indices[:, opponent]
                ]
        for action in range(action_count):
            mask = coefficients.profile_indices[:, player] == action
            own[step, :, action] = np.sum(
                opponents_weight[:, mask] * profile_hamiltonian[:, mask], axis=-1
            )
    return own


def _best_response(
    own_hamiltonians: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.min(own_hamiltonians, axis=-1, keepdims=True)
    ties = own_hamiltonians == minimum
    selectors = np.argmax(ties, axis=-1).astype(np.int32)
    tie_counts = np.sum(ties, axis=-1, dtype=np.int32)
    return selectors, tie_counts


def _one_hot(selectors: np.ndarray, action_count: int, /) -> np.ndarray:
    return np.eye(action_count, dtype=float)[selectors]


def _modal_selectors(probabilities: tuple[np.ndarray, ...], /) -> np.ndarray:
    return np.stack(
        tuple(
            np.argmax(probability, axis=-1).astype(np.int32)
            for probability in probabilities
        )
    )


def _initial_selectors(
    problem: DiscreteCoupledHJBProblem,
    value: ArrayLike | None,
    /,
) -> np.ndarray:
    expected = (
        problem.num_players,
        problem.time_grid.num_times - 1,
        problem.spatial_grid.num_points - 2,
    )
    if value is None:
        return np.zeros((1, *expected), dtype=np.int32)
    selectors = np.asarray(value)
    if np.issubdtype(selectors.dtype, np.bool_) or not np.issubdtype(
        selectors.dtype, np.integer
    ):
        raise TypeError("initial_policy_selectors must contain integer indices.")
    selectors = np.asarray(selectors, dtype=np.int64)
    if selectors.shape == expected:
        selectors = selectors[None]
    elif selectors.ndim != 4 or selectors.shape[1:] != expected:
        raise ValueError(
            "initial_policy_selectors must have shape "
            f"{expected} or (num_branches, {expected[0]}, {expected[1]}, "
            f"{expected[2]}); got {selectors.shape}."
        )
    if selectors.shape[0] == 0:
        raise ValueError("initial_policy_selectors must contain at least one branch.")
    for player, actions in enumerate(problem.player_actions):
        player_selectors = selectors[:, player]
        if np.any(player_selectors < 0) or np.any(
            player_selectors >= int(actions.shape[0])
        ):
            raise ValueError(
                f"Initial selectors for player {player} are outside its action grid."
            )
    return selectors.astype(np.int32)


def _branch_ids(value: Sequence[str] | None, branches: int, /) -> tuple[str, ...]:
    if value is None:
        return tuple(f"start-{index}" for index in range(branches))
    identifiers = tuple(str(identifier) for identifier in value)
    if len(identifiers) != branches:
        raise ValueError("branch_ids must have one entry per initial policy branch.")
    if any(not identifier for identifier in identifiers):
        raise ValueError("branch_ids entries must be non-empty.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("branch_ids entries must be unique.")
    return identifiers


def _run_branches(
    problem: DiscreteCoupledHJBProblem,
    plan: CoupledHJBPolicyIterationPlan,
    coefficients: _CoefficientTable,
    initial_selectors: np.ndarray,
    /,
) -> _RawBranchSet:
    branches = initial_selectors.shape[0]
    players = problem.num_players
    steps = problem.time_grid.num_times - 1
    interior = problem.spatial_grid.num_points - 2
    capacity = plan.maximum_iterations
    values = np.full(
        (branches, players, problem.time_grid.num_times, problem.spatial_grid.num_points),
        np.nan,
        dtype=float,
    )
    final_probabilities = tuple(
        np.full((branches, steps, interior, int(actions.shape[0])), np.nan, dtype=float)
        for actions in problem.player_actions
    )
    selectors = np.full((branches, players, steps, interior), -1, dtype=np.int32)
    best_response_selectors = np.full_like(selectors, -1)
    tie_counts = np.zeros_like(selectors)
    fixed_point_residuals = np.full((branches,), np.inf, dtype=float)
    boundary_residuals = np.full((branches,), np.inf, dtype=float)
    terminal_residuals = np.full((branches,), np.inf, dtype=float)
    policy_evaluation_residuals = np.full((branches,), np.inf, dtype=float)
    player_policy_evaluation_residuals = np.full((branches, players), np.inf, dtype=float)
    own_action_gaps = np.full((branches,), np.inf, dtype=float)
    own_action_gap_tables = np.full(
        (branches, players, steps, interior), np.inf, dtype=float
    )
    finite = np.zeros((branches,), dtype=bool)
    converged = np.zeros((branches,), dtype=bool)
    iterations = np.zeros((branches,), dtype=np.int32)
    selector_history = np.full(
        (branches, capacity, players, steps, interior), -1, dtype=np.int32
    )
    probability_history = tuple(
        np.full(
            (branches, capacity, steps, interior, int(actions.shape[0])),
            np.nan,
            dtype=float,
        )
        for actions in problem.player_actions
    )
    fixed_point_history = np.full((branches, capacity), np.nan, dtype=float)
    update_history = np.full((branches, capacity), np.nan, dtype=float)
    gap_history = np.full((branches, capacity), np.nan, dtype=float)
    selector_change_history = np.full((branches, capacity), -1, dtype=np.int32)
    iteration_validity = np.zeros((branches, capacity), dtype=bool)

    for branch in range(branches):
        probabilities = tuple(
            _one_hot(initial_selectors[branch, player], int(actions.shape[0]))
            for player, actions in enumerate(problem.player_actions)
        )
        for iteration in range(capacity):
            iterations[branch] = iteration + 1
            selector_history[branch, iteration] = _modal_selectors(probabilities)
            for player in range(players):
                probability_history[player][branch, iteration] = probabilities[player]

            old_probabilities = tuple(probability.copy() for probability in probabilities)
            proposal_selectors = np.empty((players, steps, interior), dtype=np.int32)
            iteration_gap = 0.0
            if plan.update == "jacobi":
                evaluated = _evaluate_policy(problem, coefficients, probabilities)
                proposals: list[np.ndarray] = []
                for player, actions in enumerate(problem.player_actions):
                    own = _own_hamiltonians(
                        problem,
                        coefficients,
                        probabilities,
                        player,
                        evaluated[player],
                    )
                    proposal, _ = _best_response(own)
                    proposal_selectors[player] = proposal
                    minimum = np.min(own, axis=-1)
                    current = np.sum(probabilities[player] * own, axis=-1)
                    iteration_gap = max(
                        iteration_gap,
                        float(np.max(np.maximum(current - minimum, 0.0))),
                    )
                    pure = _one_hot(proposal, int(actions.shape[0]))
                    proposals.append(
                        (1.0 - plan.damping) * probabilities[player] + plan.damping * pure
                    )
                probabilities = tuple(proposals)
            else:
                working = list(probabilities)
                for player, actions in enumerate(problem.player_actions):
                    working_tuple = tuple(working)
                    player_values = _evaluate_player(
                        problem, coefficients, working_tuple, player
                    )
                    own = _own_hamiltonians(
                        problem,
                        coefficients,
                        working_tuple,
                        player,
                        player_values,
                    )
                    proposal, _ = _best_response(own)
                    proposal_selectors[player] = proposal
                    minimum = np.min(own, axis=-1)
                    current = np.sum(working[player] * own, axis=-1)
                    iteration_gap = max(
                        iteration_gap,
                        float(np.max(np.maximum(current - minimum, 0.0))),
                    )
                    pure = _one_hot(proposal, int(actions.shape[0]))
                    working[player] = (1.0 - plan.damping) * working[
                        player
                    ] + plan.damping * pure
                probabilities = tuple(working)

            map_residual = max(
                float(
                    np.max(
                        np.abs(
                            old_probabilities[player]
                            - _one_hot(
                                proposal_selectors[player],
                                int(problem.player_actions[player].shape[0]),
                            )
                        )
                    )
                )
                for player in range(players)
            )
            update_residual = max(
                float(np.max(np.abs(probabilities[player] - old_probabilities[player])))
                for player in range(players)
            )
            selector_changes = int(
                np.count_nonzero(
                    _modal_selectors(old_probabilities) != _modal_selectors(probabilities)
                )
            )
            valid_iteration = all(
                bool(np.all(np.isfinite(probability))) for probability in probabilities
            )
            fixed_point_history[branch, iteration] = map_residual
            update_history[branch, iteration] = update_residual
            gap_history[branch, iteration] = iteration_gap
            selector_change_history[branch, iteration] = selector_changes
            iteration_validity[branch, iteration] = valid_iteration
            if not valid_iteration:
                break
            if map_residual <= plan.fixed_point_tolerance:
                break

        final_values = _evaluate_policy(problem, coefficients, probabilities)
        final_best_responses = np.empty((players, steps, interior), dtype=np.int32)
        final_ties = np.empty_like(final_best_responses)
        final_gap = 0.0
        final_gap_table = np.empty((players, steps, interior), dtype=float)
        final_fixed_point = 0.0
        for player, actions in enumerate(problem.player_actions):
            own = _own_hamiltonians(
                problem,
                coefficients,
                probabilities,
                player,
                final_values[player],
            )
            response, player_ties = _best_response(own)
            final_best_responses[player] = response
            final_ties[player] = player_ties
            minimum = np.min(own, axis=-1)
            current = np.sum(probabilities[player] * own, axis=-1)
            player_gap = np.maximum(current - minimum, 0.0)
            final_gap_table[player] = player_gap
            final_gap = max(final_gap, float(np.max(player_gap)))
            final_fixed_point = max(
                final_fixed_point,
                float(
                    np.max(
                        np.abs(
                            probabilities[player]
                            - _one_hot(response, int(actions.shape[0]))
                        )
                    )
                ),
            )

        times = np.asarray(problem.time_grid.times, dtype=float)
        policy_residual = 0.0
        player_policy_residual = np.zeros((players,), dtype=float)
        for player in range(players):
            for step, duration in enumerate(np.diff(times)):
                profile_hamiltonian = _profile_hamiltonian(
                    final_values[player, step + 1],
                    coefficients.drift[player, step],
                    coefficients.variance[player, step],
                    coefficients.cost[player, step],
                    problem.spatial_grid.spacing,
                )
                weights = _joint_profile_weights(
                    probabilities, coefficients.profile_indices, step
                )
                expected = np.sum(weights * profile_hamiltonian, axis=-1)
                step_residual = float(
                    np.max(
                        np.abs(
                            (
                                final_values[player, step, 1:-1]
                                - final_values[player, step + 1, 1:-1]
                            )
                            / duration
                            - expected
                        )
                    )
                )
                player_policy_residual[player] = max(
                    player_policy_residual[player], step_residual
                )
                policy_residual = max(policy_residual, step_residual)
        boundary = np.asarray(problem.boundary_values, dtype=float)
        terminal = np.asarray(problem.terminal_values, dtype=float)
        boundary_residual = float(
            np.max(np.abs(final_values[:, :, (0, final_values.shape[-1] - 1)] - boundary))
        )
        terminal_residual = float(np.max(np.abs(final_values[:, -1] - terminal)))
        branch_finite = bool(
            np.all(np.isfinite(final_values))
            and all(np.all(np.isfinite(probability)) for probability in probabilities)
            and np.isfinite(final_fixed_point)
            and np.isfinite(final_gap)
            and np.isfinite(policy_residual)
        )

        values[branch] = final_values
        for player in range(players):
            final_probabilities[player][branch] = probabilities[player]
        selectors[branch] = _modal_selectors(probabilities)
        best_response_selectors[branch] = final_best_responses
        tie_counts[branch] = final_ties
        fixed_point_residuals[branch] = final_fixed_point
        boundary_residuals[branch] = boundary_residual
        terminal_residuals[branch] = terminal_residual
        policy_evaluation_residuals[branch] = policy_residual
        player_policy_evaluation_residuals[branch] = player_policy_residual
        own_action_gaps[branch] = final_gap
        own_action_gap_tables[branch] = final_gap_table
        finite[branch] = branch_finite
        converged[branch] = branch_finite and (
            final_fixed_point <= plan.fixed_point_tolerance
        )

    return _RawBranchSet(
        values,
        final_probabilities,
        selectors,
        best_response_selectors,
        tie_counts,
        fixed_point_residuals,
        boundary_residuals,
        terminal_residuals,
        policy_evaluation_residuals,
        player_policy_evaluation_residuals,
        own_action_gaps,
        own_action_gap_tables,
        finite,
        converged,
        iterations,
        selector_history,
        probability_history,
        fixed_point_history,
        update_history,
        gap_history,
        selector_change_history,
        iteration_validity,
    )


def _refined_problem(
    problem: DiscreteCoupledHJBProblem,
    /,
) -> DiscreteCoupledHJBProblem:
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
    terminal = np.stack(
        tuple(
            np.interp(
                refined_points,
                coarse_points,
                np.asarray(problem.terminal_values[player], dtype=float),
            )
            for player in range(problem.num_players)
        )
    )
    boundary = np.empty((problem.num_players, refined_times.size, 2), dtype=float)
    coarse_boundary = np.asarray(problem.boundary_values, dtype=float)
    for player in range(problem.num_players):
        for side in range(2):
            boundary[player, :, side] = np.interp(
                refined_times, coarse_times, coarse_boundary[player, :, side]
            )
    time_grid = TimeGrid(
        refined_times,
        time_id=f"{problem.time_grid.time_id}/space-2-time-4",
    )
    return DiscreteCoupledHJBProblem(
        refined_grid,
        time_grid,
        problem.player_actions,
        terminal,
        boundary,
        problem.drift,
        problem.diffusion,
        problem.running_cost,
        args=problem.args,
        corner_tolerance=problem.corner_tolerance,
        problem_id=f"{problem.problem_id}/space-2-time-4",
    )


def _refined_initial_selectors(
    problem: DiscreteCoupledHJBProblem,
    selectors: np.ndarray,
    /,
) -> np.ndarray:
    fine_steps = 4 * (problem.time_grid.num_times - 1)
    fine_interior = 2 * (problem.spatial_grid.num_points - 1) - 1
    time_indices = np.arange(fine_steps, dtype=np.int32) // 4
    fine_full_indices = np.arange(1, fine_interior + 1, dtype=np.int32)
    coarse_full_indices = np.clip(
        fine_full_indices // 2, 1, problem.spatial_grid.num_points - 2
    )
    coarse_interior_indices = coarse_full_indices - 1
    return selectors[:, :, time_indices][:, :, :, coarse_interior_indices]


def _status(
    *,
    finite: bool,
    converged: bool,
    boundary_passed: bool,
    terminal_passed: bool,
    policy_evaluation_passed: bool,
    own_action_gap_passed: bool,
    refinement_passed: bool,
) -> DiscreteCoupledHJBStatus:
    if not finite:
        return DiscreteCoupledHJBStatus.NONFINITE_DISCRETE_OUTPUT
    if not converged:
        return DiscreteCoupledHJBStatus.MAXIMUM_POLICY_ITERATIONS
    if not boundary_passed or not terminal_passed:
        return DiscreteCoupledHJBStatus.BOUNDARY_OR_TERMINAL_RESIDUAL_TOO_LARGE
    if not policy_evaluation_passed:
        return DiscreteCoupledHJBStatus.POLICY_EVALUATION_RESIDUAL_TOO_LARGE
    if not own_action_gap_passed:
        return DiscreteCoupledHJBStatus.OWN_ACTION_HAMILTONIAN_GAP_TOO_LARGE
    if not refinement_passed:
        return DiscreteCoupledHJBStatus.REFINEMENT_GATE_FAILED
    return DiscreteCoupledHJBStatus.SUCCESS_LOCAL_FEEDBACK_FIXED_POINT


def _maximum_between_branches(values: np.ndarray, /) -> float:
    maximum = 0.0
    for first in range(values.shape[0]):
        for second in range(first + 1, values.shape[0]):
            maximum = max(maximum, float(np.max(np.abs(values[first] - values[second]))))
    return maximum


def _selected_actions(
    player_actions: tuple[Array, ...], selectors: np.ndarray, /
) -> np.ndarray:
    dtype = np.result_type(
        *(np.asarray(action).dtype for action in player_actions), float
    )
    selected = np.empty(selectors.shape, dtype=dtype)
    for player, actions in enumerate(player_actions):
        selected[player] = np.asarray(actions)[selectors[player]]
    return selected


def solve_coupled_hjb_reference(
    problem: DiscreteCoupledHJBProblem,
    plan: CoupledHJBPolicyIterationPlan,
    /,
    *,
    initial_policy_selectors: ArrayLike | None = None,
    branch_ids: Sequence[str] | None = None,
    residual_tolerance: float = 1.0e-8,
    refinement_absolute_tolerance: float = 2.0e-2,
    refinement_relative_tolerance: float = 5.0e-2,
) -> DiscreteCoupledHJBResult:
    """Evaluate local feedback fixed-point branches on a bounded discrete grid.

    The result is only finite-grid policy-iteration evidence for the supplied
    starts, update order, damping, selector, and nested refinement. It does not
    establish a viscosity solution, uniqueness, or a global game claim.
    """

    if not isinstance(problem, DiscreteCoupledHJBProblem):
        raise TypeError("problem must be a DiscreteCoupledHJBProblem.")
    if not isinstance(plan, CoupledHJBPolicyIterationPlan):
        raise TypeError("plan must be a CoupledHJBPolicyIterationPlan.")
    residual_tolerance = _nonnegative_tolerance(residual_tolerance, "residual_tolerance")
    absolute_tolerance = _nonnegative_tolerance(
        refinement_absolute_tolerance, "refinement_absolute_tolerance"
    )
    relative_tolerance = _nonnegative_tolerance(
        refinement_relative_tolerance, "refinement_relative_tolerance"
    )
    starts = _initial_selectors(problem, initial_policy_selectors)
    identifiers = _branch_ids(branch_ids, starts.shape[0])

    coefficients = _coefficient_table(problem)
    raw = _run_branches(problem, plan, coefficients, starts)
    refined_problem = _refined_problem(problem)
    refined_starts = _refined_initial_selectors(problem, starts)
    refined_coefficients = _coefficient_table(refined_problem)
    refined = _run_branches(refined_problem, plan, refined_coefficients, refined_starts)
    refined_common = refined.values[:, :, ::4, ::2]
    common_difference = np.abs(raw.values - refined_common)
    maximum_difference = np.max(common_difference, axis=(1, 2, 3))
    refinement_scale = np.max(np.abs(refined_common), axis=(1, 2, 3))
    refinement_thresholds = absolute_tolerance + relative_tolerance * refinement_scale

    branch_statuses = np.empty((starts.shape[0],), dtype=np.int32)
    for branch in range(starts.shape[0]):
        finite = bool(
            raw.finite[branch]
            and refined.finite[branch]
            and np.isfinite(maximum_difference[branch])
        )
        converged = bool(raw.converged[branch] and refined.converged[branch])
        branch_statuses[branch] = int(
            _status(
                finite=finite,
                converged=converged,
                boundary_passed=max(
                    raw.boundary_residuals[branch],
                    refined.boundary_residuals[branch],
                )
                <= residual_tolerance,
                terminal_passed=max(
                    raw.terminal_residuals[branch],
                    refined.terminal_residuals[branch],
                )
                <= residual_tolerance,
                policy_evaluation_passed=max(
                    raw.policy_evaluation_residuals[branch],
                    refined.policy_evaluation_residuals[branch],
                )
                <= residual_tolerance,
                own_action_gap_passed=max(
                    raw.own_action_gaps[branch], refined.own_action_gaps[branch]
                )
                <= residual_tolerance,
                refinement_passed=maximum_difference[branch]
                <= refinement_thresholds[branch],
            )
        )

    successful_branches = np.flatnonzero(
        branch_statuses
        == int(DiscreteCoupledHJBStatus.SUCCESS_LOCAL_FEEDBACK_FIXED_POINT)
    )
    if successful_branches.size:
        selected_branch = int(successful_branches[0])
    else:
        finite_residuals = np.where(
            np.isfinite(raw.fixed_point_residuals),
            raw.fixed_point_residuals,
            np.inf,
        )
        selected_branch = int(np.argmin(finite_residuals))
    status = DiscreteCoupledHJBStatus(int(branch_statuses[selected_branch]))
    successful = status == DiscreteCoupledHJBStatus.SUCCESS_LOCAL_FEEDBACK_FIXED_POINT

    maximum_boundary_residual = max(
        raw.boundary_residuals[selected_branch],
        refined.boundary_residuals[selected_branch],
    )
    maximum_terminal_residual = max(
        raw.terminal_residuals[selected_branch],
        refined.terminal_residuals[selected_branch],
    )
    maximum_policy_residual = max(
        raw.policy_evaluation_residuals[selected_branch],
        refined.policy_evaluation_residuals[selected_branch],
    )
    maximum_gap = max(
        raw.own_action_gaps[selected_branch],
        refined.own_action_gaps[selected_branch],
    )
    maximum_fixed_point_residual = max(
        raw.fixed_point_residuals[selected_branch],
        refined.fixed_point_residuals[selected_branch],
    )
    finite = bool(
        raw.finite[selected_branch]
        and refined.finite[selected_branch]
        and np.isfinite(maximum_difference[selected_branch])
    )
    boundary_passed = maximum_boundary_residual <= residual_tolerance
    terminal_passed = maximum_terminal_residual <= residual_tolerance
    policy_passed = maximum_policy_residual <= residual_tolerance
    gap_passed = maximum_gap <= residual_tolerance
    fixed_point_passed = bool(
        raw.converged[selected_branch] and refined.converged[selected_branch]
    )
    refinement_passed = bool(
        maximum_difference[selected_branch] <= refinement_thresholds[selected_branch]
    )
    between_branch_difference = _maximum_between_branches(raw.values)
    selector_dependence = any(
        not np.array_equal(raw.selectors[0], raw.selectors[branch])
        for branch in range(1, starts.shape[0])
    )
    probability_dependence = any(
        any(
            not np.allclose(
                raw.probabilities[player][0],
                raw.probabilities[player][branch],
                rtol=0.0,
                atol=plan.fixed_point_tolerance,
            )
            for player in range(problem.num_players)
        )
        for branch in range(1, starts.shape[0])
    )
    branch_dependence = (
        selector_dependence
        or probability_dependence
        or between_branch_difference > plan.fixed_point_tolerance
    )

    branch_evidence = CoupledHJBBranchEvidence(
        initial_action_selectors=jnp.asarray(starts),
        final_action_selectors=jnp.asarray(raw.selectors),
        final_best_response_selectors=jnp.asarray(raw.best_response_selectors),
        action_selector_history=jnp.asarray(raw.selector_history),
        policy_probability_history=tuple(
            jnp.asarray(history) for history in raw.probability_history
        ),
        fixed_point_residual_history=jnp.asarray(raw.fixed_point_residual_history),
        update_residual_history=jnp.asarray(raw.update_residual_history),
        own_action_hamiltonian_gap_history=jnp.asarray(raw.gap_history),
        selector_change_history=jnp.asarray(raw.selector_change_history),
        iteration_validity_history=jnp.asarray(raw.iteration_validity_history),
        iterations=jnp.asarray(raw.iterations),
        converged=jnp.asarray(raw.converged),
        refined_converged=jnp.asarray(refined.converged),
        statuses=jnp.asarray(branch_statuses, dtype=jnp.int32),
        final_fixed_point_residuals=jnp.asarray(raw.fixed_point_residuals),
        maximum_fixed_point_residuals=jnp.asarray(
            np.maximum(raw.fixed_point_residuals, refined.fixed_point_residuals)
        ),
        maximum_boundary_residuals=jnp.asarray(
            np.maximum(raw.boundary_residuals, refined.boundary_residuals)
        ),
        maximum_terminal_residuals=jnp.asarray(
            np.maximum(raw.terminal_residuals, refined.terminal_residuals)
        ),
        maximum_policy_evaluation_residuals=jnp.asarray(
            np.maximum(
                raw.policy_evaluation_residuals,
                refined.policy_evaluation_residuals,
            )
        ),
        maximum_own_action_hamiltonian_gaps=jnp.asarray(
            np.maximum(raw.own_action_gaps, refined.own_action_gaps)
        ),
        maximum_refinement_differences=jnp.asarray(maximum_difference),
        refinement_thresholds=jnp.asarray(refinement_thresholds),
        maximum_tie_counts=jnp.asarray(
            np.maximum(
                np.max(raw.tie_counts, axis=(1, 2, 3)),
                np.max(refined.tie_counts, axis=(1, 2, 3)),
            ),
            dtype=jnp.int32,
        ),
        finite=jnp.asarray(raw.finite & refined.finite),
        maximum_between_branch_value_difference=jnp.asarray(between_branch_difference),
        branch_dependence_detected=jnp.asarray(branch_dependence),
        branch_ids=identifiers,
        history_capacity=plan.maximum_iterations,
        branch_selection_id=_BRANCH_SELECTION_ID,
    )
    method = f"{_REFERENCE_METHOD}-{plan.update}"
    evidence = DiscreteCoupledHJBEvidence(
        maximum_boundary_residual=jnp.asarray(maximum_boundary_residual),
        maximum_terminal_residual=jnp.asarray(maximum_terminal_residual),
        maximum_policy_evaluation_residual=jnp.asarray(maximum_policy_residual),
        maximum_own_action_hamiltonian_gap=jnp.asarray(maximum_gap),
        maximum_fixed_point_residual=jnp.asarray(maximum_fixed_point_residual),
        maximum_refinement_difference=jnp.asarray(maximum_difference[selected_branch]),
        refinement_threshold=jnp.asarray(refinement_thresholds[selected_branch]),
        maximum_courant_number=jnp.asarray(
            max(coefficients.maximum_courant, refined_coefficients.maximum_courant)
        ),
        minimum_monotonicity_margin=jnp.asarray(
            min(coefficients.minimum_margin, refined_coefficients.minimum_margin)
        ),
        maximum_tie_count=jnp.asarray(
            max(
                int(np.max(raw.tie_counts[selected_branch])),
                int(np.max(refined.tie_counts[selected_branch])),
            ),
            dtype=jnp.int32,
        ),
        finite=jnp.asarray(finite),
        boundary_passed=jnp.asarray(boundary_passed),
        terminal_passed=jnp.asarray(terminal_passed),
        policy_evaluation_passed=jnp.asarray(policy_passed),
        own_action_hamiltonian_gap_passed=jnp.asarray(gap_passed),
        fixed_point_passed=jnp.asarray(fixed_point_passed),
        refinement_passed=jnp.asarray(refinement_passed),
        branch=branch_evidence,
        method=method,
        scope="declared-bounded-grid-local-feedback-fixed-point-evidence-only",
    )
    selected_probabilities = tuple(
        jnp.asarray(probability[selected_branch]) for probability in raw.probabilities
    )
    return DiscreteCoupledHJBResult(
        spatial_grid=problem.spatial_grid,
        time_grid=problem.time_grid,
        refined_spatial_grid=refined_problem.spatial_grid,
        refined_time_grid=refined_problem.time_grid,
        player_actions=problem.player_actions,
        joint_action_profiles=jnp.asarray(coefficients.profiles),
        values=jnp.asarray(raw.values[selected_branch]),
        refined_values=jnp.asarray(refined.values[selected_branch]),
        common_grid_difference=jnp.asarray(common_difference[selected_branch]),
        action_selectors=jnp.asarray(raw.selectors[selected_branch]),
        selected_actions=jnp.asarray(
            _selected_actions(problem.player_actions, raw.selectors[selected_branch])
        ),
        policy_probabilities=selected_probabilities,
        best_response_selectors=jnp.asarray(raw.best_response_selectors[selected_branch]),
        player_policy_evaluation_residuals=jnp.asarray(
            raw.player_policy_evaluation_residuals[selected_branch]
        ),
        own_action_hamiltonian_gaps=jnp.asarray(
            raw.own_action_gap_tables[selected_branch]
        ),
        branch_values=jnp.asarray(raw.values),
        branch_action_selectors=jnp.asarray(raw.selectors),
        branch_policy_probabilities=tuple(
            jnp.asarray(probability) for probability in raw.probabilities
        ),
        branch_player_policy_evaluation_residuals=jnp.asarray(
            raw.player_policy_evaluation_residuals
        ),
        branch_own_action_hamiltonian_gaps=jnp.asarray(raw.own_action_gap_tables),
        evidence=evidence,
        selected_branch=jnp.asarray(selected_branch, dtype=jnp.int32),
        successful=jnp.asarray(successful),
        local_feedback_fixed_point=jnp.asarray(successful),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        update=plan.update,
        damping=plan.damping,
        fixed_point_tolerance=plan.fixed_point_tolerance,
        history_capacity=plan.maximum_iterations,
        selected_branch_id=identifiers[selected_branch],
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        status_label=(LOCAL_FEEDBACK_FIXED_POINT if successful else status.name),
        certificate_label=LOCAL_FEEDBACK_FIXED_POINT,
        selector_id=_SELECTOR_ID,
        tie_break_id=_TIE_BREAK_ID,
        method=method,
        candidate_evaluation_only=True,
        viscosity_solution_claimed=False,
        unique_solution_claimed=False,
        global_nash_equilibrium_claimed=False,
    )


__all__ = [
    "CoupledHJBBranchEvidence",
    "CoupledHJBPolicyIterationPlan",
    "CoupledHJBUpdate",
    "DiscreteCoupledHJBEvidence",
    "DiscreteCoupledHJBProblem",
    "DiscreteCoupledHJBResult",
    "DiscreteCoupledHJBStatus",
    "LOCAL_FEEDBACK_FIXED_POINT",
    "solve_coupled_hjb_reference",
]
