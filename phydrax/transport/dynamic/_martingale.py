#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...linalg import pseudoinverse
from ._problem import SchrodingerBridgeProblem
from ._solver import _reference_log_transitions


class MartingaleBridgeStatus(IntEnum):
    """Terminal status of a finite martingale Schrödinger bridge."""

    CONVERGED = 0
    INFEASIBLE_SUPPORT = 1
    MAXIMUM_ITERATIONS_REACHED = 2
    NONFINITE_ITERATE = 3


class MartingaleBridgeDiagnostics(StrictModule):
    """Endpoint, conditional-mean, entropy and primal-dual evidence."""

    status: Array
    num_iterations: Array
    first_converged_iteration: Array
    endpoint_residual: Array
    martingale_defect: Array
    primal_relative_entropy: Array
    dual_objective: Array
    primal_dual_gap: Array
    entropy: Array
    reference_row_residual: Array
    controlled_row_residual: Array
    endpoint_residual_history: Array
    martingale_defect_history: Array
    primal_dual_gap_history: Array
    feasible: Array
    finite: Array


class MartingaleBridgeRefinementEvidence(StrictModule):
    """Independent comparison between coarse and refined bridge calculations."""

    coarse_relative_entropy: Array
    refined_relative_entropy: Array
    objective_change: Array
    coarse_endpoint_residual: Array
    refined_endpoint_residual: Array
    coarse_martingale_defect: Array
    refined_martingale_defect: Array
    accepted: Array
    objective_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)


class MartingaleSchrodingerBridgeProblem(StrictModule):
    """Finite bridge with an explicit martingale coordinate on every transition.

    The wrapped bridge supplies a normalized Markov reference and common finite
    state support. ``martingale_coordinates`` states which coordinates, rather
    than which arbitrary state features, obey the conditional-mean constraint.
    """

    bridge: SchrodingerBridgeProblem
    martingale_coordinates: Array
    constraint_tolerance: float = eqx.field(static=True)
    constraint_kind: str = eqx.field(static=True, default="martingale")

    def __init__(
        self,
        bridge: SchrodingerBridgeProblem,
        /,
        *,
        martingale_coordinates: ArrayLike | None = None,
        constraint_tolerance: float = 1e-7,
    ):
        if not isinstance(bridge, SchrodingerBridgeProblem):
            raise TypeError("bridge must be a SchrodingerBridgeProblem.")
        tolerance = float(constraint_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("constraint_tolerance must be finite and nonnegative.")
        cases = bridge.num_cases
        states = bridge.num_states
        nodes = bridge.num_steps + 1
        if martingale_coordinates is None:
            shared = bridge.state_support.reshape((cases, states, -1))
            coordinates = jnp.broadcast_to(
                shared[:, None, :, :],
                (cases, nodes, states, shared.shape[-1]),
            )
        else:
            supplied = jnp.asarray(martingale_coordinates, dtype=float)
            if supplied.ndim == 1 and supplied.shape == (states,):
                supplied = supplied[:, None]
            if supplied.ndim == 2 and supplied.shape[0] == states:
                coordinates = jnp.broadcast_to(
                    supplied[None, None, :, :],
                    (cases, nodes, states, supplied.shape[-1]),
                )
            elif supplied.ndim == 3 and supplied.shape[:2] == (nodes, states):
                coordinates = jnp.broadcast_to(
                    supplied[None, :, :, :],
                    (cases,) + tuple(supplied.shape),
                )
            elif supplied.ndim >= 4 and supplied.shape[:3] == (cases, nodes, states):
                coordinates = supplied.reshape((cases, nodes, states, -1))
            elif supplied.shape[: len(bridge.case_shape) + 2] == bridge.case_shape + (
                nodes,
                states,
            ):
                coordinates = supplied.reshape((cases, nodes, states, -1))
            else:
                raise ValueError(
                    "martingale_coordinates must have shared (state, coordinate), "
                    "(time, state, coordinate), or case + time/state/coordinate shape."
                )
        if coordinates.shape[3] < 1 or bool(jnp.any(~jnp.isfinite(coordinates))):
            raise ValueError("Martingale bridge coordinates must be finite and nonempty.")
        self.bridge = bridge
        self.martingale_coordinates = coordinates.reshape(
            bridge.case_shape + (nodes, states, coordinates.shape[3])
        )
        self.constraint_tolerance = tolerance
        self.constraint_kind = "martingale"

    @property
    def coordinate_size(self) -> int:
        return int(self.martingale_coordinates.shape[-1])


class MartingaleSchrodingerBridgeResult(StrictModule):
    """Enumerated finite path law and its controlled martingale Markov projection."""

    problem: MartingaleSchrodingerBridgeProblem
    path_state_indices: Array
    path_probabilities: Array
    marginal_probabilities: Array
    pair_probabilities: Array
    endpoint_coupling: Array
    controlled_transition_probabilities: Array
    initial_potential: Array
    terminal_potential: Array
    martingale_potential: Array
    diagnostics: MartingaleBridgeDiagnostics
    method: str = eqx.field(static=True, default="finite-path-cyclic-entropic-projection")

    @property
    def converged(self) -> Array:
        return self.diagnostics.status == int(MartingaleBridgeStatus.CONVERGED)

    @property
    def successful(self) -> Array:
        return self.converged

    def marginal_weights(self) -> Array:
        return self.problem.bridge.mass[..., None, None] * self.marginal_probabilities

    def initial_marginal(self) -> Array:
        return (
            self.problem.bridge.mass[..., None] * self.marginal_probabilities[..., 0, :]
        )

    def terminal_marginal(self) -> Array:
        return (
            self.problem.bridge.mass[..., None] * self.marginal_probabilities[..., -1, :]
        )

    def physical_endpoint_coupling(self) -> Array:
        return self.problem.bridge.mass[..., None, None] * self.endpoint_coupling


class MartingaleSchrodingerBridgeSolver(StrictModule):
    """Bounded cyclic KL projections for an enumerated finite path law."""

    max_iterations: int = eqx.field(static=True)
    moment_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_path_entries: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_iterations: int = 500,
        moment_iterations: int = 64,
        tolerance: float = 1e-8,
        max_path_entries: int = 1_000_000,
    ):
        iterations = int(max_iterations)
        moments = int(moment_iterations)
        tolerance_ = float(tolerance)
        budget = int(max_path_entries)
        if iterations < 1 or moments < 1 or budget < 1:
            raise ValueError("Bridge iteration and path budgets must be positive.")
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        self.max_iterations = iterations
        self.moment_iterations = moments
        self.tolerance = tolerance_
        self.max_path_entries = budget

    def solve(
        self, problem: MartingaleSchrodingerBridgeProblem, /
    ) -> MartingaleSchrodingerBridgeResult:
        if not isinstance(problem, MartingaleSchrodingerBridgeProblem):
            raise TypeError(
                "MartingaleSchrodingerBridgeSolver requires a "
                "MartingaleSchrodingerBridgeProblem; a classical bridge does not "
                "carry conditional-mean constraints."
            )
        return _solve_martingale_bridge(
            problem,
            max_iterations=self.max_iterations,
            moment_iterations=self.moment_iterations,
            tolerance=min(self.tolerance, problem.constraint_tolerance),
            max_path_entries=self.max_path_entries,
        )

    def __call__(
        self, problem: MartingaleSchrodingerBridgeProblem, /
    ) -> MartingaleSchrodingerBridgeResult:
        return self.solve(problem)


def _enumerate_paths(states: int, nodes: int, /, *, max_path_entries: int) -> Array:
    count = states**nodes
    if count > max_path_entries:
        raise ValueError(
            f"Martingale bridge needs {count} enumerated paths, exceeding explicit "
            f"budget {max_path_entries}."
        )
    flat = jnp.arange(count, dtype=jnp.int32)
    columns = []
    stride = count
    for _ in range(nodes):
        stride //= states
        columns.append((flat // stride) % states)
    return jnp.stack(tuple(columns), axis=-1)


def _reference_path_probabilities(
    initial: Array, log_transitions: Array, paths: Array, /
) -> Array:
    initial_values = initial[paths[:, 0]]
    log_probability = jnp.where(initial_values > 0.0, jnp.log(initial_values), -jnp.inf)
    for step in range(log_transitions.shape[0]):
        log_probability = (
            log_probability + log_transitions[step, paths[:, step], paths[:, step + 1]]
        )
    probability = jnp.exp(log_probability)
    return probability / jnp.sum(probability)


def _group_projection(
    probability: Array,
    groups: Array,
    target: Array,
    potential: Array,
    /,
) -> tuple[Array, Array, Array]:
    group_count = int(target.shape[0])
    feasible = jnp.asarray(True)
    result = probability
    updated = potential
    for group in range(group_count):
        mask = groups == group
        current = jnp.sum(jnp.where(mask, result, 0.0))
        desired = target[group]
        group_feasible = (desired <= 0.0) | (current > 0.0)
        feasible = feasible & group_feasible
        safe_scale = jnp.where(
            desired <= 0.0,
            0.0,
            desired / jnp.where(current > 0.0, current, 1.0),
        )
        result = jnp.where(mask, result * safe_scale, result)
        log_scale = jnp.where(
            desired <= 0.0,
            -jnp.inf,
            jnp.log(jnp.where(safe_scale > 0.0, safe_scale, 1.0)),
        )
        updated = updated.at[group].add(log_scale)
    return result, updated, feasible


def _scalar_moment_projection(
    probability: Array,
    group: Array,
    delta: Array,
    /,
    *,
    iterations: int,
    tolerance: float,
) -> tuple[Array, Array, Array]:
    active = group & (probability > 0.0)
    active_delta = jnp.where(active, delta, 0.0)
    minimum = jnp.min(jnp.where(active, delta, jnp.inf))
    maximum = jnp.max(jnp.where(active, delta, -jnp.inf))
    mass = jnp.sum(jnp.where(active, probability, 0.0))
    feasible = (mass <= 0.0) | ((minimum <= tolerance) & (maximum >= -tolerance))

    def mean(theta):
        score = jnp.where(active, theta * delta, -jnp.inf)
        maximum_score = jnp.max(score)
        weights = jnp.where(
            active,
            probability
            * jnp.exp(score - jnp.where(jnp.isfinite(maximum_score), maximum_score, 0.0)),
            0.0,
        )
        denominator = jnp.sum(weights)
        return jnp.sum(weights * active_delta) / jnp.where(
            denominator > 0.0, denominator, 1.0
        )

    lower = jnp.asarray(-1.0, dtype=probability.dtype)
    upper = jnp.asarray(1.0, dtype=probability.dtype)
    for _ in range(iterations):
        lower = jnp.where(mean(lower) > 0.0, 2.0 * lower, lower)
        upper = jnp.where(mean(upper) < 0.0, 2.0 * upper, upper)
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        middle_mean = mean(middle)
        lower = jnp.where(middle_mean < 0.0, middle, lower)
        upper = jnp.where(middle_mean >= 0.0, middle, upper)
    theta = jnp.where(mass > 0.0, 0.5 * (lower + upper), 0.0)
    exponent = theta * delta
    factor = jnp.where(group, jnp.exp(exponent), 1.0)
    projected = jnp.where(feasible, probability * factor, probability)
    return projected, jnp.where(feasible, theta, 0.0), feasible


def _vector_moment_projection(
    probability: Array,
    group: Array,
    delta: Array,
    /,
    *,
    iterations: int,
    tolerance: float,
) -> tuple[Array, Array, Array]:
    dimension = int(delta.shape[1])
    theta = jnp.zeros((dimension,), dtype=probability.dtype)
    active = group & (probability > 0.0)
    for _ in range(iterations):
        score = delta @ theta
        score = jnp.where(active, score, -jnp.inf)
        maximum = jnp.max(score)
        weights = jnp.where(
            active,
            probability * jnp.exp(score - jnp.where(jnp.isfinite(maximum), maximum, 0.0)),
            0.0,
        )
        total = jnp.sum(weights)
        normalized = weights / jnp.where(total > 0.0, total, 1.0)
        mean = ein.contract("p,pd->d", normalized, delta)
        second = ein.contract("p,pd,pe->de", normalized, delta, delta)
        covariance = second - mean[:, None] * mean[None, :]
        scale = jnp.maximum(jnp.max(jnp.abs(covariance)), 1.0)
        regularized = covariance + (
            32.0 * jnp.finfo(probability.dtype).eps * scale
        ) * jnp.eye(dimension, dtype=probability.dtype)
        step = pseudoinverse(regularized).value @ mean
        theta = theta - step
    exponent = delta @ theta
    projected = jnp.where(group, probability * jnp.exp(exponent), probability)
    numerator = ein.contract("p,pd->d", jnp.where(group, projected, 0.0), delta)
    feasible = (jnp.sum(jnp.where(group, probability, 0.0)) <= 0.0) | (
        jnp.max(jnp.abs(numerator)) <= tolerance
    )
    return (
        jnp.where(feasible, projected, probability),
        jnp.where(feasible, theta, jnp.zeros_like(theta)),
        feasible,
    )


def _moment_projection(
    probability: Array,
    paths: Array,
    coordinates: Array,
    potential: Array,
    /,
    *,
    moment_iterations: int,
    tolerance: float,
) -> tuple[Array, Array, Array]:
    result = probability
    updated = potential
    feasible = jnp.asarray(True)
    steps = paths.shape[1] - 1
    states = coordinates.shape[1]
    dimension = coordinates.shape[2]
    for step in range(steps):
        current = paths[:, step]
        following = paths[:, step + 1]
        delta = coordinates[step + 1, following] - coordinates[step, current]
        for state in range(states):
            group = current == state
            if dimension == 1:
                result, theta, valid = _scalar_moment_projection(
                    result,
                    group,
                    delta[:, 0],
                    iterations=moment_iterations,
                    tolerance=tolerance,
                )
                updated = updated.at[step, state, 0].add(theta)
            else:
                result, theta, valid = _vector_moment_projection(
                    result,
                    group,
                    delta,
                    iterations=moment_iterations,
                    tolerance=tolerance,
                )
                updated = updated.at[step, state].add(theta)
            feasible = feasible & valid
    return result, updated, feasible


def _path_statistics(
    probability: Array,
    reference: Array,
    paths: Array,
    coordinates: Array,
    log_transitions: Array,
    initial_target: Array,
    terminal_target: Array,
    alpha: Array,
    beta: Array,
    theta: Array,
    /,
) -> tuple[Array, ...]:
    nodes = paths.shape[1]
    steps = nodes - 1
    states = coordinates.shape[1]
    marginals = []
    for node in range(nodes):
        marginals.append(jnp.bincount(paths[:, node], weights=probability, length=states))
    marginal = jnp.stack(tuple(marginals), axis=0)
    pairs = []
    martingale_defect = jnp.asarray(0.0, dtype=probability.dtype)
    controlled_rows = []
    row_residual = jnp.asarray(0.0, dtype=probability.dtype)
    for step in range(steps):
        flat_pair = paths[:, step] * states + paths[:, step + 1]
        pair = jnp.bincount(
            flat_pair, weights=probability, length=states * states
        ).reshape((states, states))
        pairs.append(pair)
        row_mass = jnp.sum(pair, axis=1)
        safe_mass = jnp.where(row_mass > 0.0, row_mass, 1.0)
        controlled = pair / safe_mass[:, None]
        reference_transition = jnp.exp(log_transitions[step])
        controlled = jnp.where(row_mass[:, None] > 0.0, controlled, reference_transition)
        controlled_rows.append(controlled)
        row_residual = jnp.maximum(
            row_residual, jnp.max(jnp.abs(jnp.sum(controlled, axis=1) - 1.0))
        )
        conditional = controlled @ coordinates[step + 1]
        defect = jnp.where(
            row_mass[:, None] > 0.0,
            jnp.abs(conditional - coordinates[step]),
            0.0,
        )
        martingale_defect = jnp.maximum(martingale_defect, jnp.max(defect))
    pair_probabilities = jnp.stack(tuple(pairs), axis=0)
    controlled_probabilities = jnp.stack(tuple(controlled_rows), axis=0)
    endpoint_index = paths[:, 0] * states + paths[:, -1]
    endpoint = jnp.bincount(
        endpoint_index, weights=probability, length=states * states
    ).reshape((states, states))
    endpoint_residual = jnp.maximum(
        jnp.sum(jnp.abs(marginal[0] - initial_target)),
        jnp.sum(jnp.abs(marginal[-1] - terminal_target)),
    )
    positive = probability > 0.0
    log_ratio = jnp.where(
        positive,
        jnp.log(jnp.where(positive, probability, 1.0))
        - jnp.log(jnp.where(reference > 0.0, reference, 1.0)),
        0.0,
    )
    primal = jnp.sum(
        jnp.where(positive, probability * log_ratio, 0.0) - probability + reference
    )
    alpha_term = jnp.sum(jnp.where(initial_target > 0.0, alpha * initial_target, 0.0))
    beta_term = jnp.sum(jnp.where(terminal_target > 0.0, beta * terminal_target, 0.0))
    dual = alpha_term + beta_term - jnp.sum(probability) + jnp.sum(reference)
    gap = jnp.abs(primal - dual)
    entropy = -jnp.sum(
        jnp.where(
            positive, probability * jnp.log(jnp.where(positive, probability, 1.0)), 0.0
        )
    )
    finite = (
        jnp.all(jnp.isfinite(probability))
        & jnp.isfinite(primal)
        & jnp.isfinite(dual)
        & jnp.isfinite(gap)
        & jnp.isfinite(entropy)
    )
    return (
        marginal,
        pair_probabilities,
        endpoint,
        controlled_probabilities,
        endpoint_residual,
        martingale_defect,
        primal,
        dual,
        gap,
        entropy,
        row_residual,
        finite,
    )


def _solve_case(
    reference: Array,
    paths: Array,
    coordinates: Array,
    log_transitions: Array,
    initial: Array,
    terminal: Array,
    /,
    *,
    max_iterations: int,
    moment_iterations: int,
    tolerance: float,
) -> tuple[Array, ...]:
    states = coordinates.shape[1]
    steps = paths.shape[1] - 1
    probability = reference
    alpha = jnp.zeros((states,), dtype=reference.dtype)
    beta = jnp.zeros((states,), dtype=reference.dtype)
    theta = jnp.zeros((steps, states, coordinates.shape[2]), dtype=reference.dtype)
    histories_endpoint = []
    histories_martingale = []
    histories_gap = []
    feasible = jnp.asarray(True)
    initial_stats = _path_statistics(
        probability,
        reference,
        paths,
        coordinates,
        log_transitions,
        initial,
        terminal,
        alpha,
        beta,
        theta,
    )
    histories_endpoint.append(initial_stats[4])
    histories_martingale.append(initial_stats[5])
    histories_gap.append(initial_stats[8])
    for _ in range(max_iterations):
        probability, alpha, initial_feasible = _group_projection(
            probability, paths[:, 0], initial, alpha
        )
        probability, beta, terminal_feasible = _group_projection(
            probability, paths[:, -1], terminal, beta
        )
        probability, theta, martingale_feasible = _moment_projection(
            probability,
            paths,
            coordinates,
            theta,
            moment_iterations=moment_iterations,
            tolerance=tolerance,
        )
        feasible = feasible & initial_feasible & terminal_feasible & martingale_feasible
        stats = _path_statistics(
            probability,
            reference,
            paths,
            coordinates,
            log_transitions,
            initial,
            terminal,
            alpha,
            beta,
            theta,
        )
        histories_endpoint.append(stats[4])
        histories_martingale.append(stats[5])
        histories_gap.append(stats[8])
    endpoint_history = jnp.stack(tuple(histories_endpoint))
    martingale_history = jnp.stack(tuple(histories_martingale))
    gap_history = jnp.stack(tuple(histories_gap))
    stats = _path_statistics(
        probability,
        reference,
        paths,
        coordinates,
        log_transitions,
        initial,
        terminal,
        alpha,
        beta,
        theta,
    )
    convergence = (
        (endpoint_history <= tolerance)
        & (martingale_history <= tolerance)
        & (gap_history <= 10.0 * tolerance)
        & feasible
    )
    any_converged = jnp.any(convergence)
    first = jnp.where(any_converged, jnp.argmax(convergence), -1).astype(jnp.int32)
    iterations = jnp.where(any_converged, first, max_iterations).astype(jnp.int32)
    status = jnp.where(
        ~feasible,
        int(MartingaleBridgeStatus.INFEASIBLE_SUPPORT),
        jnp.where(
            ~stats[11],
            int(MartingaleBridgeStatus.NONFINITE_ITERATE),
            jnp.where(
                any_converged,
                int(MartingaleBridgeStatus.CONVERGED),
                int(MartingaleBridgeStatus.MAXIMUM_ITERATIONS_REACHED),
            ),
        ),
    ).astype(jnp.int32)
    return (
        probability,
        *stats[:11],
        stats[11],
        alpha,
        beta,
        theta,
        endpoint_history,
        martingale_history,
        gap_history,
        feasible,
        status,
        iterations,
        first,
    )


def _solve_martingale_bridge(
    problem: MartingaleSchrodingerBridgeProblem,
    /,
    *,
    max_iterations: int,
    moment_iterations: int,
    tolerance: float,
    max_path_entries: int,
) -> MartingaleSchrodingerBridgeResult:
    bridge = problem.bridge
    paths = _enumerate_paths(
        bridge.num_states,
        bridge.num_steps + 1,
        max_path_entries=max_path_entries,
    )
    log_transitions, reference_row_residual = _reference_log_transitions(bridge)
    cases = bridge.num_cases
    initial = bridge.initial_probabilities.reshape((cases, bridge.num_states))
    terminal = bridge.terminal_probabilities.reshape((cases, bridge.num_states))
    coordinates = problem.martingale_coordinates.reshape(
        (
            cases,
            bridge.num_steps + 1,
            bridge.num_states,
            problem.coordinate_size,
        )
    )
    outputs = []
    for case in range(cases):
        reference = _reference_path_probabilities(
            initial[case], log_transitions[case], paths
        )
        outputs.append(
            _solve_case(
                reference,
                paths,
                coordinates[case],
                log_transitions[case],
                initial[case],
                terminal[case],
                max_iterations=max_iterations,
                moment_iterations=moment_iterations,
                tolerance=tolerance,
            )
        )
    stacked = tuple(
        jnp.stack(tuple(output[index] for output in outputs))
        for index in range(len(outputs[0]))
    )
    (
        path_probabilities,
        marginal_probabilities,
        pair_probabilities,
        endpoint_coupling,
        controlled,
        endpoint_residual,
        martingale_defect,
        primal,
        dual,
        gap,
        entropy,
        controlled_row_residual,
        finite,
        alpha,
        beta,
        theta,
        endpoint_history,
        martingale_history,
        gap_history,
        feasible,
        status,
        iterations,
        first,
    ) = stacked
    case_shape = bridge.case_shape
    reshape = lambda value, trailing: value.reshape(case_shape + trailing)
    diagnostics = MartingaleBridgeDiagnostics(
        reshape(status, ()),
        reshape(iterations, ()),
        reshape(first, ()),
        reshape(endpoint_residual, ()),
        reshape(martingale_defect, ()),
        reshape(primal, ()),
        reshape(dual, ()),
        reshape(gap, ()),
        reshape(entropy, ()),
        reference_row_residual.reshape(case_shape),
        reshape(controlled_row_residual, ()),
        reshape(endpoint_history, (max_iterations + 1,)),
        reshape(martingale_history, (max_iterations + 1,)),
        reshape(gap_history, (max_iterations + 1,)),
        reshape(feasible, ()),
        reshape(finite, ()),
    )
    return MartingaleSchrodingerBridgeResult(
        problem,
        paths,
        reshape(path_probabilities, (paths.shape[0],)),
        reshape(marginal_probabilities, (bridge.num_steps + 1, bridge.num_states)),
        reshape(
            pair_probabilities, (bridge.num_steps, bridge.num_states, bridge.num_states)
        ),
        reshape(endpoint_coupling, (bridge.num_states, bridge.num_states)),
        reshape(controlled, (bridge.num_steps, bridge.num_states, bridge.num_states)),
        reshape(alpha, (bridge.num_states,)),
        reshape(beta, (bridge.num_states,)),
        reshape(theta, (bridge.num_steps, bridge.num_states, problem.coordinate_size)),
        diagnostics,
        "finite-path-cyclic-entropic-projection",
    )


def martingale_bridge_refinement_evidence(
    coarse: MartingaleSchrodingerBridgeResult,
    refined: MartingaleSchrodingerBridgeResult,
    /,
    *,
    refinement_id: str,
    objective_tolerance: float,
    constraint_tolerance: float,
) -> MartingaleBridgeRefinementEvidence:
    """Compare bridge objectives and constraints without relabeling a plain bridge."""
    if not isinstance(coarse, MartingaleSchrodingerBridgeResult) or not isinstance(
        refined, MartingaleSchrodingerBridgeResult
    ):
        raise TypeError(
            "Refinement requires two MartingaleSchrodingerBridgeResult objects."
        )
    identifier = str(refinement_id)
    objective_tolerance_ = float(objective_tolerance)
    constraint_tolerance_ = float(constraint_tolerance)
    if not identifier:
        raise ValueError("refinement_id must be nonempty.")
    if (
        not isfinite(objective_tolerance_)
        or objective_tolerance_ < 0.0
        or not isfinite(constraint_tolerance_)
        or constraint_tolerance_ < 0.0
    ):
        raise ValueError("Refinement tolerances must be finite and nonnegative.")
    if coarse.problem.coordinate_size != refined.problem.coordinate_size:
        raise ValueError("Refinement bridges must use the same martingale dimension.")
    change = jnp.abs(
        refined.diagnostics.primal_relative_entropy
        - coarse.diagnostics.primal_relative_entropy
    )
    accepted = (
        coarse.successful
        & refined.successful
        & (change <= objective_tolerance_)
        & (refined.diagnostics.endpoint_residual <= constraint_tolerance_)
        & (refined.diagnostics.martingale_defect <= constraint_tolerance_)
    )
    return MartingaleBridgeRefinementEvidence(
        coarse.diagnostics.primal_relative_entropy,
        refined.diagnostics.primal_relative_entropy,
        change,
        coarse.diagnostics.endpoint_residual,
        refined.diagnostics.endpoint_residual,
        coarse.diagnostics.martingale_defect,
        refined.diagnostics.martingale_defect,
        accepted,
        objective_tolerance_,
        constraint_tolerance_,
        identifier,
    )


__all__ = [
    "MartingaleBridgeDiagnostics",
    "MartingaleBridgeRefinementEvidence",
    "MartingaleBridgeStatus",
    "MartingaleSchrodingerBridgeProblem",
    "MartingaleSchrodingerBridgeResult",
    "MartingaleSchrodingerBridgeSolver",
    "martingale_bridge_refinement_evidence",
]
