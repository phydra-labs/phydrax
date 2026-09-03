#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics import DiscreteStepContext, TimeGrid
from ...optim import (
    AbstractRiskMeasure,
    CVaRRisk,
    EntropicRisk,
    ExpectationRisk,
    MeanVarianceRisk,
)
from ...stochastic import (
    is_stochastic_realization,
    realization_independence_labels,
    realization_path_labels,
    StochasticRealization,
)


CoverageMethod: TypeAlias = Literal["none", "asymptotic-normal", "hoeffding"]
SampleRole: TypeAlias = Literal["training", "holdout"]
FeedbackPolicy: TypeAlias = Callable[[DiscreteStepContext, Array, Any], ArrayLike]
ControlledTransition: TypeAlias = Callable[
    [DiscreteStepContext, Array, Array, Array, Any], ArrayLike
]
StageCost: TypeAlias = Callable[[DiscreteStepContext, Array, Array, Any], ArrayLike]
TerminalCost: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


class FeedbackPolicyEvaluationStatus(IntEnum):
    """Stable per-path and aggregate evaluation status codes."""

    SUCCESS = 0
    INVALID_NOISE_PATH = 1
    NONFINITE_ACTION = 2
    NONFINITE_TRANSITION = 3
    NONFINITE_STAGE_COST = 4
    NONFINITE_TERMINAL_COST = 5
    NONFINITE_RETURN = 6
    PARTIAL_PATH_FAILURE = 7
    NO_VALID_PATHS = 8


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    resolved = tuple(int(size) for size in value)
    if any(size <= 0 for size in resolved):
        raise ValueError(f"{owner} dimensions must be positive.")
    return resolved


def _real_inexact(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.number)
        and not jnp.issubdtype(array.dtype, jnp.complexfloating)
    ):
        raise TypeError(f"{owner} must be a real numeric array.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _event_finite(value: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return jnp.isfinite(value)
    axes = tuple(range(value.ndim - len(event_shape), value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes)


def _scalar_values(value: ArrayLike, count: int, owner: str, /) -> Array:
    array = _real_inexact(value, owner)
    expected = (count,)
    if tuple(array.shape) != expected:
        raise ValueError(f"{owner} must return shape {expected}; got {array.shape}.")
    return array


def _confidence(value: float, /) -> float:
    level = float(value)
    if not isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError(
            "confidence must be finite and lie strictly between zero and one."
        )
    return level


def _coverage_method(value: str, /) -> CoverageMethod:
    if value == "none":
        return "none"
    if value == "asymptotic-normal":
        return "asymptotic-normal"
    if value == "hoeffding":
        return "hoeffding"
    raise ValueError("method must be 'none', 'asymptotic-normal', or 'hoeffding'.")


def _sample_role(value: str, /) -> SampleRole:
    if value == "training":
        return "training"
    if value == "holdout":
        return "holdout"
    raise ValueError("sample_role must be 'training' or 'holdout'.")


def _bounds(value: tuple[float, float] | None, /) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("return_bounds must contain exactly two entries.")
    lower, upper = (float(bound) for bound in value)
    if not isfinite(lower) or not isfinite(upper) or not lower < upper:
        raise ValueError("return_bounds must be finite and strictly increasing.")
    return lower, upper


class ControlledTransitionProblem(StrictModule):
    """Finite-grid controlled transition with exogenous, fully supplied noise."""

    transition: ControlledTransition
    time_grid: TimeGrid
    initial_state: Array
    stage_cost: StageCost
    terminal_cost: TerminalCost
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: ControlledTransition,
        time_grid: TimeGrid,
        initial_state: ArrayLike,
        /,
        *,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        noise_shape: Sequence[int],
        stage_cost: StageCost,
        terminal_cost: TerminalCost,
        args: Any = None,
        problem_id: str,
    ):
        if not callable(transition):
            raise TypeError("transition must be callable.")
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if not callable(stage_cost):
            raise TypeError("stage_cost must be callable.")
        if not callable(terminal_cost):
            raise TypeError("terminal_cost must be callable.")
        states = _shape(state_shape, "state_shape")
        actions = _shape(action_shape, "action_shape")
        noises = _shape(noise_shape, "noise_shape")
        initial = _real_inexact(initial_state, "initial_state")
        if tuple(initial.shape) != states:
            raise ValueError(
                f"initial_state must have state_shape {states}; got {initial.shape}."
            )
        if not bool(jnp.all(jnp.isfinite(initial))):
            raise ValueError("initial_state must be finite.")
        self.transition = transition
        self.time_grid = time_grid
        self.initial_state = initial
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.args = args
        self.state_shape = states
        self.action_shape = actions
        self.noise_shape = noises
        self.problem_id = _identifier(problem_id, "problem_id")


class PreparedControlledNoise(StrictModule):
    """Complete replayable noise paths with explicit coupling and independence."""

    increments: Array
    valid: Array
    independence_labels: Array
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)

    def __init__(
        self,
        increments: ArrayLike,
        /,
        *,
        valid: ArrayLike,
        realization_ids: Sequence[str],
        coupling_id: str,
        independence_labels: ArrayLike,
        noise_shape: Sequence[int],
    ):
        noises = _shape(noise_shape, "noise_shape")
        values = _real_inexact(increments, "increments")
        expected_rank = 2 + len(noises)
        if values.ndim != expected_rank:
            raise ValueError(
                "increments must have one path axis, one step axis, and the declared "
                f"noise axes; expected rank {expected_rank}, got {values.ndim}."
            )
        if noises and tuple(values.shape[-len(noises) :]) != noises:
            raise ValueError(
                f"increments must end in noise_shape {noises}; got {values.shape}."
            )
        path_count = int(values.shape[0])
        step_count = int(values.shape[1])
        if path_count < 1 or step_count < 1:
            raise ValueError("increments must contain at least one path and one step.")
        validity = jnp.asarray(valid, dtype=bool)
        if tuple(validity.shape) != (path_count,):
            raise ValueError(f"valid must have shape ({path_count},).")
        raw_labels = jnp.asarray(independence_labels)
        if not jnp.issubdtype(raw_labels.dtype, jnp.integer):
            raise TypeError("independence_labels must have an integer dtype.")
        if tuple(raw_labels.shape) != (path_count,):
            raise ValueError(f"independence_labels must have shape ({path_count},).")
        maximum_label = np.iinfo(np.int32).max
        if bool(jnp.any(raw_labels < 0)) or bool(jnp.any(raw_labels > maximum_label)):
            raise ValueError(
                "independence_labels must lie in the non-negative int32 range."
            )
        labels = raw_labels.astype(jnp.int32)
        identifiers = tuple(realization_ids)
        if len(identifiers) != path_count:
            raise ValueError(
                f"realization_ids must contain one ID for each of {path_count} paths."
            )
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in identifiers
        ):
            raise ValueError("realization_ids must contain non-empty strings.")
        if len(set(identifiers)) != path_count:
            raise ValueError("realization_ids must uniquely identify every path.")
        self.increments = values
        self.valid = validity
        self.independence_labels = labels
        self.realization_ids = identifiers
        self.coupling_id = _identifier(coupling_id, "coupling_id")
        self.noise_shape = noises
        self.num_paths = path_count
        self.num_steps = step_count

    @classmethod
    def from_realization(
        cls,
        increments: ArrayLike,
        realization: StochasticRealization,
        /,
        *,
        valid: ArrayLike,
        noise_shape: Sequence[int],
    ) -> PreparedControlledNoise:
        """Attach stochastic-realization provenance to evaluated increments."""
        if not is_stochastic_realization(realization):
            raise TypeError("realization must be a StochasticRealization.")
        path_ids = realization_path_labels(
            "controlled-noise", realization, realization.sample_shape
        )
        cluster_ids = realization_independence_labels(
            realization, realization.sample_shape
        )
        integer_by_label: dict[str, int] = {}
        integer_labels: list[int] = []
        for label in cluster_ids:
            if label is None:
                raise ValueError(
                    "realization must declare independence labels for every path."
                )
            if label not in integer_by_label:
                integer_by_label[label] = len(integer_by_label)
            integer_labels.append(integer_by_label[label])
        return cls(
            increments,
            valid=valid,
            realization_ids=path_ids,
            coupling_id=realization.coupling_id,
            independence_labels=jnp.asarray(integer_labels, dtype=jnp.int32),
            noise_shape=noise_shape,
        )


class ControlledPathBatch(StrictModule):
    """Full-state feedback rollouts over one explicit prepared-noise batch."""

    time_grid: TimeGrid
    states: Array
    actions: Array
    noise_paths: Array
    noise_valid: Array
    valid: Array
    status: Array
    stage_costs: Array
    terminal_costs: Array
    returns: Array
    independence_labels: Array
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: ControlledTransitionProblem,
        prepared_noise: PreparedControlledNoise,
        states: ArrayLike,
        actions: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        stage_costs: ArrayLike,
        terminal_costs: ArrayLike,
        returns: ArrayLike,
        policy_id: str,
    ):
        if not isinstance(problem, ControlledTransitionProblem):
            raise TypeError("problem must be a ControlledTransitionProblem.")
        if not isinstance(prepared_noise, PreparedControlledNoise):
            raise TypeError("prepared_noise must be PreparedControlledNoise.")
        if prepared_noise.noise_shape != problem.noise_shape:
            raise ValueError("prepared_noise noise_shape does not match the problem.")
        if prepared_noise.num_steps != problem.time_grid.num_steps:
            raise ValueError(
                "prepared_noise step count does not match the problem time grid."
            )
        count = prepared_noise.num_paths
        steps = problem.time_grid.num_steps
        expected_states = (count, steps + 1) + problem.state_shape
        expected_actions = (count, steps) + problem.action_shape
        states_ = _real_inexact(states, "states")
        actions_ = _real_inexact(actions, "actions")
        stage = _real_inexact(stage_costs, "stage_costs")
        terminal = _real_inexact(terminal_costs, "terminal_costs")
        returns_ = _real_inexact(returns, "returns")
        validity = jnp.asarray(valid, dtype=bool)
        statuses = jnp.asarray(status, dtype=jnp.int32)
        expected_path = (count,)
        if tuple(states_.shape) != expected_states:
            raise ValueError(f"states must have shape {expected_states}.")
        if tuple(actions_.shape) != expected_actions:
            raise ValueError(f"actions must have shape {expected_actions}.")
        if tuple(stage.shape) != (count, steps):
            raise ValueError(f"stage_costs must have shape ({count}, {steps}).")
        for name, value in (
            ("terminal_costs", terminal),
            ("returns", returns_),
            ("valid", validity),
            ("status", statuses),
        ):
            if tuple(value.shape) != expected_path:
                raise ValueError(f"{name} must have shape {expected_path}.")
        self.time_grid = problem.time_grid
        self.states = states_
        self.actions = actions_
        self.noise_paths = prepared_noise.increments
        self.noise_valid = prepared_noise.valid
        self.valid = validity
        self.status = statuses
        self.stage_costs = stage
        self.terminal_costs = terminal
        self.returns = returns_
        self.independence_labels = prepared_noise.independence_labels
        self.realization_ids = prepared_noise.realization_ids
        self.coupling_id = prepared_noise.coupling_id
        self.policy_id = _identifier(policy_id, "policy_id")
        self.problem_id = problem.problem_id
        self.state_shape = problem.state_shape
        self.action_shape = problem.action_shape
        self.noise_shape = problem.noise_shape

    @property
    def path_count(self) -> int:
        return int(self.returns.shape[0])

    @property
    def path_ids(self) -> tuple[str, ...]:
        return self.realization_ids

    @property
    def costs(self) -> Array:
        return self.stage_costs

    @property
    def terminal_cost(self) -> Array:
        return self.terminal_costs

    @property
    def successful(self) -> Array:
        success = int(FeedbackPolicyEvaluationStatus.SUCCESS)
        return self.valid & (self.status == success)


class MonteCarloEvidence(StrictModule):
    """Cluster-aware finite-sample record without a population-optimality claim."""

    valid_path_count: Array
    independent_cluster_count: Array
    estimate: Array
    standard_error: Array
    lower: Array
    upper: Array
    confidence: float = eqx.field(static=True)
    coverage: CoverageMethod = eqx.field(static=True)
    coverage_assumptions: tuple[str, ...] = eqx.field(static=True)
    risk_kind: str = eqx.field(static=True)
    sample_role: SampleRole = eqx.field(static=True)
    return_bounds: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid_path_count: ArrayLike,
        independent_cluster_count: ArrayLike,
        estimate: ArrayLike,
        standard_error: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        confidence: float,
        coverage: CoverageMethod,
        coverage_assumptions: Sequence[str],
        risk_kind: str,
        sample_role: SampleRole,
        return_bounds: tuple[float, float] | None,
    ):
        scalar_fields = {}
        for name, value in (
            ("valid_path_count", valid_path_count),
            ("independent_cluster_count", independent_cluster_count),
            ("estimate", estimate),
            ("standard_error", standard_error),
            ("lower", lower),
            ("upper", upper),
        ):
            scalar = jnp.asarray(value)
            if scalar.shape != ():
                raise ValueError(f"{name} must be scalar.")
            scalar_fields[name] = scalar
        assumptions = tuple(coverage_assumptions)
        if any(not isinstance(item, str) or not item for item in assumptions):
            raise ValueError("coverage_assumptions must contain non-empty strings.")
        self.valid_path_count = scalar_fields["valid_path_count"].astype(jnp.int32)
        self.independent_cluster_count = scalar_fields[
            "independent_cluster_count"
        ].astype(jnp.int32)
        self.estimate = scalar_fields["estimate"]
        self.standard_error = scalar_fields["standard_error"]
        self.lower = scalar_fields["lower"]
        self.upper = scalar_fields["upper"]
        self.confidence = _confidence(confidence)
        self.coverage = _coverage_method(coverage)
        self.coverage_assumptions = assumptions
        self.risk_kind = _identifier(risk_kind, "risk_kind")
        self.sample_role = _sample_role(sample_role)
        self.return_bounds = _bounds(return_bounds)

    @property
    def interval(self) -> tuple[Array, Array]:
        return self.lower, self.upper

    @property
    def interval_lower(self) -> Array:
        return self.lower

    @property
    def interval_upper(self) -> Array:
        return self.upper

    @property
    def independent_clusters(self) -> Array:
        return self.independent_cluster_count

    @property
    def has_coverage_claim(self) -> bool:
        return self.coverage != "none"


class FeedbackPolicyEvaluation(StrictModule):
    """Raw paths, empirical risk, and separately qualified Monte Carlo evidence."""

    paths: ControlledPathBatch
    empirical_risk: Array
    evidence: MonteCarloEvidence
    status: Array
    risk_kind: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.status == int(FeedbackPolicyEvaluationStatus.SUCCESS)


class PairedPolicyComparison(StrictModule):
    """Common-random-number difference, defined as right return minus left return."""

    left_paths: ControlledPathBatch
    right_paths: ControlledPathBatch
    paired_differences: Array
    evidence: MonteCarloEvidence
    left_policy_id: str = eqx.field(static=True)
    right_policy_id: str = eqx.field(static=True)
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)

    @property
    def mean_difference(self) -> Array:
        return self.evidence.estimate

    @property
    def standard_error(self) -> Array:
        return self.evidence.standard_error

    @property
    def interval(self) -> tuple[Array, Array]:
        return self.evidence.interval


def _set_first_status(
    status: Array,
    failed: Array,
    code: FeedbackPolicyEvaluationStatus,
) -> Array:
    return jnp.where(
        (status == int(FeedbackPolicyEvaluationStatus.SUCCESS)) & failed,
        int(code),
        status,
    ).astype(jnp.int32)


def rollout_feedback(
    problem: ControlledTransitionProblem,
    policy: FeedbackPolicy,
    prepared_noise: PreparedControlledNoise,
    /,
    *,
    policy_id: str,
) -> ControlledPathBatch:
    """Replay feedback chosen before the current noise is exposed."""
    if not isinstance(problem, ControlledTransitionProblem):
        raise TypeError("problem must be a ControlledTransitionProblem.")
    if not callable(policy):
        raise TypeError("policy must be callable.")
    if not isinstance(prepared_noise, PreparedControlledNoise):
        raise TypeError("prepared_noise must be PreparedControlledNoise.")
    _identifier(policy_id, "policy_id")
    if prepared_noise.noise_shape != problem.noise_shape:
        raise ValueError("prepared noise_shape does not match the problem.")
    if prepared_noise.num_steps != problem.time_grid.num_steps:
        raise ValueError(
            "prepared noise step count does not match the problem time grid."
        )

    count = prepared_noise.num_paths
    state = jnp.broadcast_to(problem.initial_state, (count,) + problem.state_shape)
    states = [state]
    actions = []
    stage_costs = []
    finite_noise = jnp.all(
        _event_finite(prepared_noise.increments, problem.noise_shape), axis=1
    )
    status = jnp.where(
        prepared_noise.valid & finite_noise,
        int(FeedbackPolicyEvaluationStatus.SUCCESS),
        int(FeedbackPolicyEvaluationStatus.INVALID_NOISE_PATH),
    ).astype(jnp.int32)

    for step in range(problem.time_grid.num_steps):
        context = DiscreteStepContext(
            problem.time_grid.times[step],
            problem.time_grid.times[step + 1],
            jnp.asarray(step, dtype=jnp.int32),
        )
        action = jax.vmap(
            lambda current_state: jnp.asarray(
                policy(context, current_state, problem.args)
            )
        )(state)
        action = _real_inexact(action, "policy output")
        expected_action = (count,) + problem.action_shape
        if tuple(action.shape) != expected_action:
            raise ValueError(
                f"policy must return action_shape {problem.action_shape} per path; "
                f"got batched shape {action.shape}."
            )
        stage = jax.vmap(
            lambda current_state, current_action: jnp.asarray(
                problem.stage_cost(context, current_state, current_action, problem.args)
            )
        )(state, action)
        stage = _scalar_values(stage, count, "stage_cost")
        noise = prepared_noise.increments[:, step]
        next_state = jax.vmap(
            lambda current_state, current_action, current_noise: jnp.asarray(
                problem.transition(
                    context,
                    current_state,
                    current_action,
                    current_noise,
                    problem.args,
                )
            )
        )(state, action, noise)
        next_state = _real_inexact(next_state, "transition output")
        expected_state = (count,) + problem.state_shape
        if tuple(next_state.shape) != expected_state:
            raise ValueError(
                f"transition must return state_shape {problem.state_shape} per path; "
                f"got batched shape {next_state.shape}."
            )
        status = _set_first_status(
            status,
            ~_event_finite(action, problem.action_shape),
            FeedbackPolicyEvaluationStatus.NONFINITE_ACTION,
        )
        status = _set_first_status(
            status,
            ~jnp.isfinite(stage),
            FeedbackPolicyEvaluationStatus.NONFINITE_STAGE_COST,
        )
        status = _set_first_status(
            status,
            ~_event_finite(next_state, problem.state_shape),
            FeedbackPolicyEvaluationStatus.NONFINITE_TRANSITION,
        )
        actions.append(action)
        stage_costs.append(stage)
        states.append(next_state)
        state = next_state

    terminal = jax.vmap(
        lambda final_state: jnp.asarray(
            problem.terminal_cost(problem.time_grid.times[-1], final_state, problem.args)
        )
    )(state)
    terminal = _scalar_values(terminal, count, "terminal_cost")
    status = _set_first_status(
        status,
        ~jnp.isfinite(terminal),
        FeedbackPolicyEvaluationStatus.NONFINITE_TERMINAL_COST,
    )
    state_values = jnp.stack(states, axis=1)
    action_values = jnp.stack(actions, axis=1)
    stage_values = jnp.stack(stage_costs, axis=1)
    returns = jnp.sum(stage_values, axis=1) + terminal
    status = _set_first_status(
        status,
        ~jnp.isfinite(returns),
        FeedbackPolicyEvaluationStatus.NONFINITE_RETURN,
    )
    valid = status == int(FeedbackPolicyEvaluationStatus.SUCCESS)
    return ControlledPathBatch(
        problem=problem,
        prepared_noise=prepared_noise,
        states=state_values,
        actions=action_values,
        valid=valid,
        status=status,
        stage_costs=stage_values,
        terminal_costs=terminal,
        returns=returns,
        policy_id=policy_id,
    )


def _cluster_summary(
    values: Array,
    valid: Array,
    independence_labels: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    labels = np.asarray(independence_labels, dtype=np.int64)
    unique_labels = tuple(int(label) for label in np.unique(labels))
    summaries = []
    complete = []
    for label in unique_labels:
        indices = np.nonzero(labels == label)[0]
        selected = jnp.asarray(indices, dtype=jnp.int32)
        summaries.append(jnp.mean(values[selected]))
        complete.append(jnp.all(valid[selected]))
    raw_summaries = jnp.stack(summaries)
    complete_values = jnp.stack(complete)
    summary_values = jnp.where(complete_values, raw_summaries, 0.0)
    cluster_count = jnp.sum(complete_values, dtype=jnp.int32)
    safe_count = jnp.maximum(cluster_count, 1)
    estimate = jnp.sum(summary_values) / safe_count
    sample_variance = jnp.where(
        cluster_count >= 2,
        jnp.sum(
            jnp.where(
                complete_values,
                jnp.square(raw_summaries - estimate),
                0.0,
            )
        )
        / (cluster_count - 1),
        jnp.asarray(jnp.nan, dtype=values.dtype),
    )
    standard_error = jnp.sqrt(sample_variance / safe_count)
    weight_square_sum = jnp.asarray(1.0, dtype=values.dtype) / safe_count
    return estimate, standard_error, cluster_count, weight_square_sum


def _evidence(
    values: Array,
    valid: Array,
    independence_labels: Array,
    /,
    *,
    estimate: Array,
    confidence: float,
    method: CoverageMethod,
    return_bounds: tuple[float, float] | None,
    risk_kind: str,
    sample_role: SampleRole,
    expectation_functional: bool,
) -> MonteCarloEvidence:
    level = _confidence(confidence)
    requested = _coverage_method(method)
    bounds = _bounds(return_bounds)
    role = _sample_role(sample_role)
    if requested == "hoeffding" and bounds is None:
        raise ValueError("Hoeffding evidence requires declared finite return_bounds.")
    if bounds is not None:
        lower_bound, upper_bound = bounds
        within_bounds = (~valid) | ((values >= lower_bound) & (values <= upper_bound))
        if not bool(jnp.all(within_bounds)):
            raise ValueError("Observed valid returns violate declared return_bounds.")

    (
        cluster_estimate,
        standard_error,
        cluster_count,
        weight_square_sum,
    ) = _cluster_summary(values, valid, independence_labels)
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    all_valid = bool(jnp.all(valid))
    enough_clusters = int(cluster_count) >= 2
    coverage: CoverageMethod = requested
    assumptions: tuple[str, ...]
    lower = jnp.asarray(jnp.nan, dtype=values.dtype)
    upper = jnp.asarray(jnp.nan, dtype=values.dtype)
    resolved_estimate = estimate

    if not expectation_functional:
        coverage = "none"
        standard_error = jnp.asarray(jnp.nan, dtype=values.dtype)
        assumptions = ("risk-functional sampling coverage is not implemented",)
    elif role == "training" and requested != "none":
        coverage = "none"
        resolved_estimate = cluster_estimate
        assumptions = ("training-sample reuse precludes a holdout coverage claim",)
    elif not all_valid and requested != "none":
        coverage = "none"
        resolved_estimate = cluster_estimate
        assumptions = ("invalid-path selection precludes a coverage claim",)
    elif requested == "none":
        assumptions = ("no population coverage claim",)
    elif not enough_clusters:
        coverage = "none"
        resolved_estimate = cluster_estimate
        assumptions = ("fewer than two independent clusters",)
    elif requested == "asymptotic-normal":
        resolved_estimate = cluster_estimate
        critical = jsp.special.ndtri(jnp.asarray(0.5 + 0.5 * level))
        lower = resolved_estimate - critical * standard_error
        upper = resolved_estimate + critical * standard_error
        assumptions = (
            "independent clusters",
            "finite second moment",
            "asymptotic normal approximation",
            "policy fixed before holdout evaluation",
        )
    else:
        resolved_estimate = cluster_estimate
        if bounds is None:
            raise ValueError("Hoeffding evidence requires declared return_bounds.")
        lower_bound, upper_bound = bounds
        radius = (upper_bound - lower_bound) * jnp.sqrt(
            0.5
            * weight_square_sum
            * jnp.log(jnp.asarray(2.0 / (1.0 - level), dtype=values.dtype))
        )
        lower = resolved_estimate - radius
        upper = resolved_estimate + radius
        standard_error = jnp.asarray(jnp.nan, dtype=values.dtype)
        assumptions = (
            "independent clusters",
            "almost-sure declared return bounds",
            "policy fixed before holdout evaluation",
        )

    return MonteCarloEvidence(
        valid_path_count=valid_count,
        independent_cluster_count=cluster_count,
        estimate=resolved_estimate,
        standard_error=standard_error,
        lower=lower,
        upper=upper,
        confidence=level,
        coverage=coverage,
        coverage_assumptions=assumptions,
        risk_kind=risk_kind,
        sample_role=role,
        return_bounds=bounds,
    )


def evaluate_feedback_policy(
    problem: ControlledTransitionProblem,
    policy: FeedbackPolicy,
    prepared_noise: PreparedControlledNoise,
    /,
    *,
    policy_id: str,
    risk: AbstractRiskMeasure | None = None,
    confidence: float = 0.95,
    method: CoverageMethod = "asymptotic-normal",
    return_bounds: tuple[float, float] | None = None,
    sample_role: SampleRole = "holdout",
) -> FeedbackPolicyEvaluation:
    """Evaluate an empirical risk and separately state its available MC evidence."""
    risk_measure = ExpectationRisk() if risk is None else risk
    if not isinstance(
        risk_measure, (ExpectationRisk, MeanVarianceRisk, CVaRRisk, EntropicRisk)
    ):
        raise TypeError(
            "risk must be ExpectationRisk, MeanVarianceRisk, CVaRRisk, EntropicRisk, "
            "or None."
        )
    paths = rollout_feedback(problem, policy, prepared_noise, policy_id=policy_id)
    valid_count = jnp.sum(paths.valid)
    weights = paths.valid.astype(paths.returns.dtype) / jnp.maximum(valid_count, 1)
    safe_returns = jnp.where(paths.valid, paths.returns, 0.0)
    empirical_risk = jnp.where(
        valid_count > 0,
        risk_measure.evaluate(safe_returns, weights),
        jnp.asarray(jnp.nan, dtype=paths.returns.dtype),
    )
    if bool(jnp.all(paths.valid)):
        aggregate_status = FeedbackPolicyEvaluationStatus.SUCCESS
    elif int(valid_count) == 0:
        aggregate_status = FeedbackPolicyEvaluationStatus.NO_VALID_PATHS
    else:
        aggregate_status = FeedbackPolicyEvaluationStatus.PARTIAL_PATH_FAILURE
    evidence = _evidence(
        paths.returns,
        paths.valid,
        paths.independence_labels,
        estimate=empirical_risk,
        confidence=confidence,
        method=method,
        return_bounds=return_bounds,
        risk_kind=risk_measure.risk_id,
        sample_role=sample_role,
        expectation_functional=isinstance(risk_measure, ExpectationRisk),
    )
    return FeedbackPolicyEvaluation(
        paths=paths,
        empirical_risk=empirical_risk,
        evidence=evidence,
        status=jnp.asarray(int(aggregate_status), dtype=jnp.int32),
        risk_kind=risk_measure.risk_id,
    )


def compare_feedback_policies(
    left_paths: ControlledPathBatch,
    right_paths: ControlledPathBatch,
    /,
    *,
    confidence: float = 0.95,
    method: CoverageMethod = "asymptotic-normal",
    return_bounds: tuple[float, float] | None = None,
) -> PairedPolicyComparison:
    """Compare right-minus-left returns under verified common random numbers."""
    if not isinstance(left_paths, ControlledPathBatch) or not isinstance(
        right_paths, ControlledPathBatch
    ):
        raise TypeError("left_paths and right_paths must be ControlledPathBatch values.")
    if left_paths.problem_id != right_paths.problem_id:
        raise ValueError("paired paths must have identical problem identity.")
    if left_paths.path_count != right_paths.path_count:
        raise ValueError("paired paths must have identical path counts.")
    if left_paths.realization_ids != right_paths.realization_ids:
        raise ValueError("paired paths must have identical realization IDs.")
    if left_paths.coupling_id != right_paths.coupling_id:
        raise ValueError("paired paths must have identical coupling IDs.")
    if not np.array_equal(
        np.asarray(left_paths.independence_labels),
        np.asarray(right_paths.independence_labels),
    ):
        raise ValueError("paired paths must have identical independence labels.")
    if not np.array_equal(
        np.asarray(left_paths.noise_valid),
        np.asarray(right_paths.noise_valid),
    ):
        raise ValueError(
            "paired paths must have identical supplied-noise validity support."
        )
    if not np.array_equal(
        np.asarray(left_paths.valid),
        np.asarray(right_paths.valid),
    ):
        raise ValueError("paired paths must have identical valid path support.")
    if not np.array_equal(
        np.asarray(left_paths.noise_paths),
        np.asarray(right_paths.noise_paths),
        equal_nan=True,
    ):
        raise ValueError("paired paths must contain identical supplied noise paths.")

    bounds = _bounds(return_bounds)
    if bounds is not None:
        lower_bound, upper_bound = bounds
        left_within = (~left_paths.valid) | (
            (left_paths.returns >= lower_bound) & (left_paths.returns <= upper_bound)
        )
        right_within = (~right_paths.valid) | (
            (right_paths.returns >= lower_bound) & (right_paths.returns <= upper_bound)
        )
        if not bool(jnp.all(left_within & right_within)):
            raise ValueError(
                "Observed valid policy returns violate declared return_bounds."
            )
    difference_bounds = (
        None if bounds is None else (bounds[0] - bounds[1], bounds[1] - bounds[0])
    )
    differences = right_paths.returns - left_paths.returns
    empirical_mean = jnp.mean(differences)
    evidence = _evidence(
        differences,
        left_paths.valid,
        left_paths.independence_labels,
        estimate=empirical_mean,
        confidence=confidence,
        method=method,
        return_bounds=difference_bounds,
        risk_kind="paired-expectation-difference",
        sample_role="holdout",
        expectation_functional=True,
    )
    return PairedPolicyComparison(
        left_paths=left_paths,
        right_paths=right_paths,
        paired_differences=differences,
        evidence=evidence,
        left_policy_id=left_paths.policy_id,
        right_policy_id=right_paths.policy_id,
        realization_ids=left_paths.realization_ids,
        coupling_id=left_paths.coupling_id,
    )


__all__ = [
    "ControlledPathBatch",
    "ControlledTransitionProblem",
    "FeedbackPolicyEvaluation",
    "FeedbackPolicyEvaluationStatus",
    "MonteCarloEvidence",
    "PairedPolicyComparison",
    "PreparedControlledNoise",
    "compare_feedback_policies",
    "evaluate_feedback_policy",
    "rollout_feedback",
]
