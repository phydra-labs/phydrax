#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen-policy finite-horizon fitted Bellman evaluation."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...dynamics import DiscreteStepContext
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FactorizationPolicy,
    factorize,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    RankPolicy,
    solve,
    TolerancePolicy,
)
from ...stochastic import (
    BSDEEvaluation,
    BSDEPathBatch,
    BSDEProblem,
    evaluate_bsde,
    FeynmanKacLabelBatch,
    FeynmanKacSamplingPlan,
    trajectory_node_feynman_kac_labels,
)
from ._evaluation import ControlledPathBatch, ControlledTransitionProblem


FROZEN_POLICY_FITTED_BELLMAN = "FROZEN_POLICY_FITTED_BELLMAN"

FeatureMap: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
FeedbackPolicy: TypeAlias = Callable[[DiscreteStepContext, Array, Any], ArrayLike]
ControlledDrift: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
ControlledDiffusion: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
BSDEControlPredictor: TypeAlias = Callable[[Array, Array], ArrayLike]


class FittedBellmanStatus(IntEnum):
    """Stable outcomes for a frozen-policy fitted Bellman evaluation."""

    SUCCESS = 0
    NO_VALID_TRAINING_PATHS = 1
    NO_VALID_HOLDOUT_PATHS = 2
    INSUFFICIENT_TRAINING_PATHS = 3
    RANK_DEFICIENT = 4
    CONDITION_LIMIT_REACHED = 5
    LINEAR_SOLVE_FAILED = 6
    NONFINITE_OUTPUT = 7
    DEPENDENCY_FAILED = 8


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _nonnegative(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return resolved


def _positive(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{owner} must be finite and positive.")
    return resolved


def _path_weights(
    value: ArrayLike | None,
    paths: ControlledPathBatch,
    owner: str,
    /,
) -> Array:
    if value is None:
        return jnp.ones((paths.path_count,), dtype=paths.states.dtype)
    weights = jnp.asarray(value)
    if weights.shape != (paths.path_count,):
        raise ValueError(f"{owner} must have shape ({paths.path_count},).")
    if not jnp.issubdtype(weights.dtype, jnp.number) or jnp.issubdtype(
        weights.dtype, jnp.complexfloating
    ):
        raise TypeError(f"{owner} must be a real numeric array.")
    weights = weights.astype(jnp.result_type(weights, paths.states, float))
    if bool(jnp.any(~jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return weights


def _compatible_batches(
    training: ControlledPathBatch, holdout: ControlledPathBatch, /
) -> None:
    if training is holdout:
        raise ValueError("training_paths and holdout_paths must be separate batches.")
    if training.problem_id != holdout.problem_id:
        raise ValueError("Training and holdout problem IDs must match.")
    if training.policy_id != holdout.policy_id:
        raise ValueError("Training and holdout frozen-policy IDs must match.")
    if training.time_grid.time_id != holdout.time_grid.time_id or not bool(
        jnp.array_equal(training.time_grid.times, holdout.time_grid.times)
    ):
        raise ValueError("Training and holdout time grids must be identical.")
    if (
        training.state_shape != holdout.state_shape
        or training.action_shape != holdout.action_shape
        or training.noise_shape != holdout.noise_shape
    ):
        raise ValueError("Training and holdout event shapes must match.")
    if training.coupling_id == holdout.coupling_id:
        raise ValueError("Training and holdout coupling IDs must be distinct.")
    overlap = set(training.realization_ids).intersection(holdout.realization_ids)
    if overlap:
        raise ValueError("Training and holdout realization IDs must be disjoint.")


class FittedBellmanProblem(StrictModule):
    """Fixed features and statistically separate paths for one frozen policy.

    The feature callback has signature ``feature_map(time, state, args)`` and
    returns exactly ``(num_features,)``.  Neither it nor the physical policy
    identified by the two path batches is changed by this evaluator.
    """

    training_paths: ControlledPathBatch
    holdout_paths: ControlledPathBatch
    training_weights: Array
    holdout_weights: Array
    args: Any
    feature_map: FeatureMap = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        training_paths: ControlledPathBatch,
        holdout_paths: ControlledPathBatch,
        feature_map: FeatureMap,
        /,
        *,
        num_features: int,
        feature_id: str,
        problem_id: str | None = None,
        training_weights: ArrayLike | None = None,
        holdout_weights: ArrayLike | None = None,
        args: Any = None,
    ):
        if not isinstance(training_paths, ControlledPathBatch):
            raise TypeError("training_paths must be a ControlledPathBatch.")
        if not isinstance(holdout_paths, ControlledPathBatch):
            raise TypeError("holdout_paths must be a ControlledPathBatch.")
        if not callable(feature_map):
            raise TypeError("feature_map must be callable.")
        _compatible_batches(training_paths, holdout_paths)
        count = int(num_features)
        if count < 1:
            raise ValueError("num_features must be positive.")
        probe = jnp.asarray(
            feature_map(
                training_paths.time_grid.times[0],
                jnp.zeros(training_paths.state_shape, dtype=training_paths.states.dtype),
                args,
            )
        )
        if probe.shape != (count,):
            raise ValueError(
                f"feature_map must return shape {(count,)}; got {probe.shape}."
            )
        if not jnp.issubdtype(probe.dtype, jnp.number) or jnp.issubdtype(
            probe.dtype, jnp.complexfloating
        ):
            raise TypeError("feature_map must return real numeric features.")
        self.training_paths = training_paths
        self.holdout_paths = holdout_paths
        self.training_weights = _path_weights(
            training_weights, training_paths, "training_weights"
        )
        self.holdout_weights = _path_weights(
            holdout_weights, holdout_paths, "holdout_weights"
        )
        self.args = args
        self.feature_map = feature_map
        self.num_features = count
        self.feature_id = _identifier(feature_id, "feature_id")
        self.problem_id = _identifier(
            training_paths.problem_id if problem_id is None else problem_id,
            "problem_id",
        )
        self.policy_id = training_paths.policy_id


class FittedBellmanPlan(StrictModule):
    """Immutable weighted-regression and acceptance policy."""

    ridge: float = eqx.field(static=True)
    rank_relative_tolerance: float = eqx.field(static=True)
    rank_absolute_tolerance: float = eqx.field(static=True)
    solve_tolerance: float = eqx.field(static=True)
    maximum_condition: float | None = eqx.field(static=True)
    minimum_training_paths: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ridge: float,
        plan_id: str,
        rank_relative_tolerance: float = 1e-10,
        rank_absolute_tolerance: float = 0.0,
        solve_tolerance: float = 1e-8,
        maximum_condition: float | None = None,
        minimum_training_paths: int = 1,
    ):
        minimum = int(minimum_training_paths)
        if minimum < 1:
            raise ValueError("minimum_training_paths must be positive.")
        condition = None if maximum_condition is None else float(maximum_condition)
        if condition is not None and (not isfinite(condition) or condition <= 1.0):
            raise ValueError("maximum_condition must exceed one or be None.")
        self.ridge = _nonnegative(ridge, "ridge")
        self.rank_relative_tolerance = _nonnegative(
            rank_relative_tolerance, "rank_relative_tolerance"
        )
        self.rank_absolute_tolerance = _nonnegative(
            rank_absolute_tolerance, "rank_absolute_tolerance"
        )
        self.solve_tolerance = _positive(solve_tolerance, "solve_tolerance")
        self.maximum_condition = condition
        self.minimum_training_paths = minimum
        self.plan_id = _identifier(plan_id, "plan_id")


class FittedBellmanPrepared(StrictModule):
    """Shape-checked feature tables and masks ready for backward evaluation."""

    problem: FittedBellmanProblem
    plan: FittedBellmanPlan
    training_features: Array
    holdout_features: Array
    training_path_valid: Array
    holdout_path_valid: Array
    training_regression_mask: Array
    holdout_evaluation_mask: Array

    @property
    def num_steps(self) -> int:
        return self.problem.training_paths.time_grid.num_steps


def _feature_table(
    problem: FittedBellmanProblem,
    paths: ControlledPathBatch,
    /,
) -> Array:
    count = paths.path_count
    nodes = paths.time_grid.num_times
    flat_states = paths.states.reshape((-1,) + paths.state_shape)
    flat_times = jnp.broadcast_to(paths.time_grid.times, (count, nodes)).reshape((-1,))

    def evaluate(time, state):
        return jnp.asarray(problem.feature_map(time, state, problem.args))

    features = jax.vmap(evaluate)(flat_times, flat_states)
    expected = (count * nodes, problem.num_features)
    if features.shape != expected:
        raise ValueError(
            f"feature_map returned batched shape {features.shape}; expected {expected}."
        )
    if not jnp.issubdtype(features.dtype, jnp.number) or jnp.issubdtype(
        features.dtype, jnp.complexfloating
    ):
        raise TypeError("feature_map must return real numeric features.")
    return features.reshape((count, nodes, problem.num_features))


def _finite_paths(paths: ControlledPathBatch, features: Array, /) -> Array:
    state_axes = tuple(range(2, paths.states.ndim))
    action_axes = tuple(range(2, paths.actions.ndim))
    state_finite = jnp.all(jnp.isfinite(paths.states), axis=state_axes)
    action_finite = jnp.all(jnp.isfinite(paths.actions), axis=action_axes)
    return (
        paths.successful
        & jnp.all(state_finite, axis=1)
        & jnp.all(action_finite, axis=1)
        & jnp.all(jnp.isfinite(features), axis=(1, 2))
        & jnp.all(jnp.isfinite(paths.stage_costs), axis=1)
        & jnp.isfinite(paths.terminal_costs)
    )


def prepare_fitted_bellman(
    problem: FittedBellmanProblem,
    plan: FittedBellmanPlan,
    /,
) -> FittedBellmanPrepared:
    """Evaluate the fixed feature map without fitting any coefficients."""
    if not isinstance(problem, FittedBellmanProblem):
        raise TypeError("problem must be a FittedBellmanProblem.")
    if not isinstance(plan, FittedBellmanPlan):
        raise TypeError("plan must be a FittedBellmanPlan.")
    training_features = _feature_table(problem, problem.training_paths)
    holdout_features = _feature_table(problem, problem.holdout_paths)
    training_valid = _finite_paths(problem.training_paths, training_features)
    holdout_valid = _finite_paths(problem.holdout_paths, holdout_features)
    return FittedBellmanPrepared(
        problem=problem,
        plan=plan,
        training_features=training_features,
        holdout_features=holdout_features,
        training_path_valid=training_valid,
        holdout_path_valid=holdout_valid,
        training_regression_mask=training_valid & (problem.training_weights > 0.0),
        holdout_evaluation_mask=holdout_valid & (problem.holdout_weights > 0.0),
    )


class FittedBellmanResult(StrictModule):
    """Backward value fit plus separate training and holdout evidence."""

    problem: FittedBellmanProblem
    plan: FittedBellmanPlan
    prepared: FittedBellmanPrepared
    coefficients: Array
    training_value_predictions: Array
    holdout_value_predictions: Array
    training_targets: Array
    holdout_targets: Array
    training_regression_residuals: Array
    holdout_bellman_residuals: Array
    training_weighted_rmse: Array
    holdout_weighted_rmse: Array
    sample_counts: Array
    design_ranks: Array
    design_condition_numbers: Array
    system_condition_numbers: Array
    linear_status: Array
    original_normal_equation_residuals: Array
    ridge_normal_equation_residuals: Array
    stage_status: Array
    valid_stages: Array
    valid: Array
    status: Array
    training_path_valid: Array
    holdout_path_valid: Array
    training_realization_ids: tuple[str, ...] = eqx.field(static=True)
    holdout_realization_ids: tuple[str, ...] = eqx.field(static=True)
    training_coupling_id: str = eqx.field(static=True)
    holdout_coupling_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    result_label: str = eqx.field(static=True)
    frozen_policy_evaluation: bool = eqx.field(static=True)
    policy_improvement_performed: bool = eqx.field(static=True)
    optimality_claimed: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def value_coefficients(self) -> Array:
        return self.coefficients

    @property
    def training_values(self) -> Array:
        return self.training_value_predictions

    @property
    def holdout_values(self) -> Array:
        return self.holdout_value_predictions

    @property
    def training_residuals(self) -> Array:
        return self.training_regression_residuals

    @property
    def holdout_residuals(self) -> Array:
        return self.holdout_bellman_residuals

    @property
    def ranks(self) -> Array:
        return self.design_ranks

    @property
    def condition_numbers(self) -> Array:
        return self.design_condition_numbers

    def predict(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        """Evaluate the piecewise-grid fitted scalar value at one state."""
        time_value = jnp.asarray(time)
        state_value = jnp.asarray(state)
        if state_value.shape != self.problem.training_paths.state_shape:
            raise ValueError("state must have exactly the fitted state_shape.")
        times = self.problem.training_paths.time_grid.times
        index = jnp.searchsorted(times, time_value, side="right") - 1
        index = jnp.where(time_value >= times[-1], times.shape[0] - 1, index)
        index = jnp.clip(index, 0, times.shape[0] - 1)
        features = jnp.asarray(
            self.problem.feature_map(time_value, state_value, self.problem.args)
        )
        if features.shape != (self.problem.num_features,):
            raise ValueError("feature_map returned an incompatible feature shape.")
        return jnp.dot(features, self.coefficients[index])


def _weighted_rmse(residual: Array, mask: Array, weights: Array, /) -> Array:
    safe_weight = jnp.where(mask, weights, 0.0)
    total = jnp.sum(safe_weight)
    safe_residual = jnp.where(mask, residual, 0.0)
    return jnp.where(
        total > 0.0,
        jnp.sqrt(jnp.sum(safe_weight * jnp.square(safe_residual)) / total),
        jnp.asarray(jnp.nan, dtype=residual.dtype),
    )


def _regression_step(
    design: Array,
    target: Array,
    mask: Array,
    weights: Array,
    plan: FittedBellmanPlan,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    dtype = jnp.result_type(design, target, weights, float)
    safe_weight = jnp.where(mask, weights, 0.0).astype(dtype)
    total_weight = jnp.sum(safe_weight)
    normalized_weight = safe_weight / jnp.where(total_weight > 0.0, total_weight, 1.0)
    safe_design = jnp.where(mask[:, None], design, 0.0).astype(dtype)
    safe_target = jnp.where(mask, target, 0.0).astype(dtype)
    weighted_design = jnp.sqrt(normalized_weight)[:, None] * safe_design
    weighted_target = jnp.sqrt(normalized_weight) * safe_target

    rank_policy = RankPolicy(
        relative_cutoff=plan.rank_relative_tolerance,
        absolute_cutoff=plan.rank_absolute_tolerance,
        require_full_rank=False,
    )
    svd = factorize(
        DenseLinearOperator(weighted_design, operator_id="fitted-bellman:design"),
        FactorizationPolicy(
            "svd",
            rank=rank_policy,
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
        ),
    )
    singular_values = svd.singular_values()
    maximum_singular = jnp.max(singular_values, initial=0.0)
    cutoff = (
        plan.rank_absolute_tolerance + plan.rank_relative_tolerance * maximum_singular
    )
    rank = jnp.sum(singular_values > cutoff, dtype=jnp.int32)
    full_column_rank = rank == design.shape[1]
    retained_minimum = jnp.min(
        jnp.where(singular_values > cutoff, singular_values, jnp.inf),
        initial=jnp.inf,
    )
    design_condition = jnp.where(
        full_column_rank & (retained_minimum > 0.0),
        maximum_singular / retained_minimum,
        jnp.asarray(jnp.inf, dtype=dtype),
    )
    smallest_square = jnp.where(
        full_column_rank, jnp.square(retained_minimum), jnp.asarray(0.0, dtype=dtype)
    )
    system_condition = (jnp.square(maximum_singular) + plan.ridge) / jnp.maximum(
        smallest_square + plan.ridge,
        jnp.finfo(dtype).tiny,
    )

    gram = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_target
    system = gram + plan.ridge * jnp.eye(design.shape[1], dtype=dtype)
    linear = solve(
        LinearSystem(
            DenseLinearOperator(system, operator_id="fitted-bellman:normal-equation"),
            problem_id="fitted-bellman:weighted-ridge",
        ),
        rhs,
        policy=LinearSolvePolicy(
            DenseLU(),
            tolerance=TolerancePolicy(
                relative=plan.solve_tolerance, absolute=plan.solve_tolerance
            ),
            rank=RankPolicy(require_full_rank=False),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
        ),
    )
    coefficients_raw = linear.value
    prediction_raw = design @ coefficients_raw
    residual_raw = target - prediction_raw
    original_moment = safe_design.T @ (
        normalized_weight * jnp.where(mask, residual_raw, 0.0)
    )
    ridge_moment = original_moment - plan.ridge * coefficients_raw
    count = jnp.sum(mask, dtype=jnp.int32)
    enough = count >= plan.minimum_training_paths
    rank_ok = full_column_rank | (plan.ridge > 0.0)
    condition_ok = (
        jnp.asarray(True)
        if plan.maximum_condition is None
        else system_condition <= plan.maximum_condition
    )
    linear_ok = linear.status == int(LinearSolveStatus.SUCCESS)
    finite = (
        jnp.all(jnp.isfinite(coefficients_raw))
        & jnp.all(jnp.isfinite(jnp.where(mask, residual_raw, 0.0)))
        & jnp.isfinite(system_condition)
    )
    valid = (count > 0) & enough & rank_ok & condition_ok & linear_ok & finite
    coefficients = jnp.where(valid, coefficients_raw, jnp.nan)
    prediction = jnp.where(valid, prediction_raw, jnp.nan)
    residual = jnp.where(mask & valid, residual_raw, jnp.nan)
    original_error = jnp.where(
        valid, jnp.max(jnp.abs(original_moment), initial=0.0), jnp.nan
    )
    ridge_error = jnp.where(valid, jnp.max(jnp.abs(ridge_moment), initial=0.0), jnp.nan)
    status = jnp.where(
        count == 0,
        int(FittedBellmanStatus.NO_VALID_TRAINING_PATHS),
        jnp.where(
            ~enough,
            int(FittedBellmanStatus.INSUFFICIENT_TRAINING_PATHS),
            jnp.where(
                ~rank_ok,
                int(FittedBellmanStatus.RANK_DEFICIENT),
                jnp.where(
                    ~condition_ok,
                    int(FittedBellmanStatus.CONDITION_LIMIT_REACHED),
                    jnp.where(
                        ~linear_ok,
                        int(FittedBellmanStatus.LINEAR_SOLVE_FAILED),
                        jnp.where(
                            ~finite,
                            int(FittedBellmanStatus.NONFINITE_OUTPUT),
                            int(FittedBellmanStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return (
        coefficients,
        prediction,
        residual,
        count,
        rank,
        design_condition,
        system_condition,
        jnp.asarray(linear.status, dtype=jnp.int32),
        original_error,
        ridge_error,
        jnp.asarray(valid),
        status,
    )


def evaluate_fitted_bellman(prepared: FittedBellmanPrepared, /) -> FittedBellmanResult:
    """Fit backward conditional values; never improve or replace the policy."""
    if not isinstance(prepared, FittedBellmanPrepared):
        raise TypeError("prepared must be a FittedBellmanPrepared.")
    problem = prepared.problem
    plan = prepared.plan
    training = problem.training_paths
    holdout = problem.holdout_paths
    nodes = training.time_grid.num_times
    train_count = training.path_count
    holdout_count = holdout.path_count
    feature_count = problem.num_features
    dtype = jnp.result_type(
        prepared.training_features,
        prepared.holdout_features,
        training.stage_costs,
        holdout.stage_costs,
        float,
    )

    coefficients = jnp.full((nodes, feature_count), jnp.nan, dtype=dtype)
    training_predictions = jnp.full((train_count, nodes), jnp.nan, dtype=dtype)
    holdout_predictions = jnp.full((holdout_count, nodes), jnp.nan, dtype=dtype)
    training_targets = jnp.full((train_count, nodes), jnp.nan, dtype=dtype)
    holdout_targets = jnp.full((holdout_count, nodes), jnp.nan, dtype=dtype)
    training_residuals = jnp.full((train_count, nodes), jnp.nan, dtype=dtype)
    holdout_residuals = jnp.full((holdout_count, nodes), jnp.nan, dtype=dtype)
    training_rmse = jnp.full((nodes,), jnp.nan, dtype=dtype)
    holdout_rmse = jnp.full((nodes,), jnp.nan, dtype=dtype)
    sample_counts = jnp.zeros((nodes,), dtype=jnp.int32)
    ranks = jnp.zeros((nodes,), dtype=jnp.int32)
    design_conditions = jnp.full((nodes,), jnp.nan, dtype=dtype)
    system_conditions = jnp.full((nodes,), jnp.nan, dtype=dtype)
    linear_statuses = jnp.full(
        (nodes,), int(LinearSolveStatus.NONFINITE_INPUT), dtype=jnp.int32
    )
    original_errors = jnp.full((nodes,), jnp.nan, dtype=dtype)
    ridge_errors = jnp.full((nodes,), jnp.nan, dtype=dtype)
    stage_status = jnp.full(
        (nodes,), int(FittedBellmanStatus.DEPENDENCY_FAILED), dtype=jnp.int32
    )
    valid_stages = jnp.zeros((nodes,), dtype=bool)
    continuation_valid = jnp.asarray(True)

    for node in range(nodes - 1, -1, -1):
        if node == nodes - 1:
            train_target = training.terminal_costs
            holdout_target = holdout.terminal_costs
        else:
            next_train = prepared.training_features[:, node + 1] @ coefficients[node + 1]
            next_holdout = prepared.holdout_features[:, node + 1] @ coefficients[node + 1]
            train_target = training.stage_costs[:, node] + next_train
            holdout_target = holdout.stage_costs[:, node] + next_holdout

        target_finite = jnp.isfinite(train_target)
        train_mask = prepared.training_regression_mask & target_finite
        regression = _regression_step(
            prepared.training_features[:, node],
            train_target,
            train_mask,
            problem.training_weights,
            plan,
        )
        (
            coefficient,
            train_prediction,
            train_residual,
            count,
            rank,
            design_condition,
            system_condition,
            linear_status,
            original_error,
            ridge_error,
            regression_valid,
            regression_status,
        ) = regression
        node_valid = continuation_valid & regression_valid
        node_status = jnp.where(
            continuation_valid,
            regression_status,
            int(FittedBellmanStatus.DEPENDENCY_FAILED),
        ).astype(jnp.int32)
        coefficient = jnp.where(node_valid, coefficient, jnp.nan)
        train_prediction = prepared.training_features[:, node] @ coefficient
        holdout_prediction = prepared.holdout_features[:, node] @ coefficient
        train_residual = train_target - train_prediction
        holdout_residual = holdout_target - holdout_prediction
        train_residual_mask = train_mask & node_valid
        holdout_mask = (
            prepared.holdout_evaluation_mask & jnp.isfinite(holdout_target) & node_valid
        )

        coefficients = coefficients.at[node].set(coefficient)
        training_predictions = training_predictions.at[:, node].set(
            jnp.where(
                prepared.training_path_valid & node_valid, train_prediction, jnp.nan
            )
        )
        holdout_predictions = holdout_predictions.at[:, node].set(
            jnp.where(
                prepared.holdout_path_valid & node_valid, holdout_prediction, jnp.nan
            )
        )
        training_targets = training_targets.at[:, node].set(
            jnp.where(prepared.training_path_valid & node_valid, train_target, jnp.nan)
        )
        holdout_targets = holdout_targets.at[:, node].set(
            jnp.where(prepared.holdout_path_valid & node_valid, holdout_target, jnp.nan)
        )
        training_residuals = training_residuals.at[:, node].set(
            jnp.where(train_residual_mask, train_residual, jnp.nan)
        )
        holdout_residuals = holdout_residuals.at[:, node].set(
            jnp.where(holdout_mask, holdout_residual, jnp.nan)
        )
        training_rmse = training_rmse.at[node].set(
            _weighted_rmse(train_residual, train_residual_mask, problem.training_weights)
        )
        holdout_rmse = holdout_rmse.at[node].set(
            _weighted_rmse(holdout_residual, holdout_mask, problem.holdout_weights)
        )
        sample_counts = sample_counts.at[node].set(count)
        ranks = ranks.at[node].set(rank)
        design_conditions = design_conditions.at[node].set(design_condition)
        system_conditions = system_conditions.at[node].set(system_condition)
        linear_statuses = linear_statuses.at[node].set(linear_status)
        original_errors = original_errors.at[node].set(original_error)
        ridge_errors = ridge_errors.at[node].set(ridge_error)
        stage_status = stage_status.at[node].set(node_status)
        valid_stages = valid_stages.at[node].set(node_valid)
        continuation_valid = node_valid

    has_holdout = jnp.any(prepared.holdout_evaluation_mask)
    all_stages = jnp.all(valid_stages)
    valid = all_stages & has_holdout
    direct_failure = (stage_status != int(FittedBellmanStatus.SUCCESS)) & (
        stage_status != int(FittedBellmanStatus.DEPENDENCY_FAILED)
    )
    causal_failure = jnp.argmax(direct_failure)
    regression_status = jnp.where(
        jnp.any(direct_failure),
        stage_status[causal_failure],
        int(FittedBellmanStatus.DEPENDENCY_FAILED),
    )
    status = jnp.where(
        ~all_stages,
        regression_status,
        jnp.where(
            ~has_holdout,
            int(FittedBellmanStatus.NO_VALID_HOLDOUT_PATHS),
            int(FittedBellmanStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return FittedBellmanResult(
        problem=problem,
        plan=plan,
        prepared=prepared,
        coefficients=coefficients,
        training_value_predictions=training_predictions,
        holdout_value_predictions=holdout_predictions,
        training_targets=training_targets,
        holdout_targets=holdout_targets,
        training_regression_residuals=training_residuals,
        holdout_bellman_residuals=holdout_residuals,
        training_weighted_rmse=training_rmse,
        holdout_weighted_rmse=holdout_rmse,
        sample_counts=sample_counts,
        design_ranks=ranks,
        design_condition_numbers=design_conditions,
        system_condition_numbers=system_conditions,
        linear_status=linear_statuses,
        original_normal_equation_residuals=original_errors,
        ridge_normal_equation_residuals=ridge_errors,
        stage_status=stage_status,
        valid_stages=valid_stages,
        valid=valid,
        status=status,
        training_path_valid=prepared.training_path_valid,
        holdout_path_valid=prepared.holdout_path_valid,
        training_realization_ids=training.realization_ids,
        holdout_realization_ids=holdout.realization_ids,
        training_coupling_id=training.coupling_id,
        holdout_coupling_id=holdout.coupling_id,
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        policy_id=problem.policy_id,
        feature_id=problem.feature_id,
        time_id=training.time_grid.time_id,
        result_label=FROZEN_POLICY_FITTED_BELLMAN,
        frozen_policy_evaluation=True,
        policy_improvement_performed=False,
        optimality_claimed=False,
    )


def fit_frozen_policy_bellman(
    problem: FittedBellmanProblem,
    plan: FittedBellmanPlan,
    /,
) -> FittedBellmanResult:
    """Prepare and evaluate one frozen policy on independent path batches."""
    return evaluate_fitted_bellman(prepare_fitted_bellman(problem, plan))


class FittedBellmanBSDEBridge(StrictModule):
    """Current-path BSDE/Feynman--Kac view with physical action kept separate."""

    fitted_result: FittedBellmanResult
    controlled_problem: ControlledTransitionProblem
    controlled_paths: ControlledPathBatch
    bsde_problem: BSDEProblem
    bsde_paths: BSDEPathBatch
    evaluation: BSDEEvaluation
    feynman_kac_labels: FeynmanKacLabelBatch
    physical_actions: Array
    martingale_integrands: Array
    independence_labels: Array
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    sample_role: str = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    z_shape: tuple[int, ...] = eqx.field(static=True)
    action_is_martingale_integrand: bool = eqx.field(static=True)
    policy_improvement_performed: bool = eqx.field(static=True)

    @property
    def z(self) -> Array:
        return self.martingale_integrands


def _policy_context(paths: ControlledPathBatch, time: Array, /) -> DiscreteStepContext:
    times = paths.time_grid.times
    index = jnp.searchsorted(times, time, side="right") - 1
    index = jnp.clip(index, 0, paths.time_grid.num_steps - 1)
    return DiscreteStepContext(times[index], times[index + 1], index.astype(jnp.int32))


def bridge_fitted_bellman_to_bsde(
    result: FittedBellmanResult,
    controlled_problem: ControlledTransitionProblem,
    frozen_policy: FeedbackPolicy,
    controlled_drift: ControlledDrift,
    controlled_diffusion: ControlledDiffusion,
    /,
    *,
    paths: ControlledPathBatch | None = None,
    z_predictor: BSDEControlPredictor | None = None,
    action_tolerance: float = 1e-8,
    path_role: str = "holdout",
) -> FittedBellmanBSDEBridge:
    """Evaluate the fitted value as a current frozen-policy BSDE.

    ``frozen_policy`` produces physical actions.  ``z_predictor`` produces the
    BSDE martingale integrand with shape ``(1,) + noise_shape``; when omitted,
    that integrand is obtained as ``grad(value) @ controlled_diffusion``.  These
    quantities are never identified with one another.
    """
    if not isinstance(result, FittedBellmanResult):
        raise TypeError("result must be a FittedBellmanResult.")
    if not bool(result.valid):
        raise ValueError("result must be a valid fitted Bellman evaluation.")
    if not isinstance(controlled_problem, ControlledTransitionProblem):
        raise TypeError("controlled_problem must be a ControlledTransitionProblem.")
    for owner, callback in (
        ("frozen_policy", frozen_policy),
        ("controlled_drift", controlled_drift),
        ("controlled_diffusion", controlled_diffusion),
    ):
        if not callable(callback):
            raise TypeError(f"{owner} must be callable.")
    if z_predictor is not None and not callable(z_predictor):
        raise TypeError("z_predictor must be callable or None.")
    tolerance = _nonnegative(action_tolerance, "action_tolerance")
    if path_role not in ("training", "holdout"):
        raise ValueError("path_role must be 'training' or 'holdout'.")
    source = (
        result.problem.training_paths
        if path_role == "training"
        else result.problem.holdout_paths
    )
    selected = source if paths is None else paths
    if not isinstance(selected, ControlledPathBatch):
        raise TypeError("paths must be a ControlledPathBatch or None.")
    if not bool(jnp.any(selected.successful)):
        raise ValueError("paths must contain at least one successful path.")
    if (
        selected.problem_id != controlled_problem.problem_id
        or selected.problem_id != result.problem.training_paths.problem_id
    ):
        raise ValueError("Controlled, fitted, and selected path problem IDs must match.")
    if selected.policy_id != result.policy_id:
        raise ValueError("Selected paths do not use the fitted frozen-policy ID.")
    if (
        selected.state_shape != controlled_problem.state_shape
        or selected.action_shape != controlled_problem.action_shape
        or selected.noise_shape != controlled_problem.noise_shape
    ):
        raise ValueError(
            "Selected path event shapes do not match the controlled problem."
        )
    if (
        selected.time_grid.time_id != result.time_id
        or selected.time_grid.time_id != controlled_problem.time_grid.time_id
        or not bool(
            jnp.array_equal(selected.time_grid.times, controlled_problem.time_grid.times)
        )
    ):
        raise ValueError("Selected, fitted, and controlled time grids must match.")

    def action_at(time, state):
        context = _policy_context(selected, time)
        value = jnp.asarray(frozen_policy(context, state, controlled_problem.args))
        if value.shape != controlled_problem.action_shape:
            raise ValueError("frozen_policy returned an incompatible action shape.")
        return value

    count = selected.path_count
    left_states = selected.states[:, :-1]
    flat_states = left_states.reshape((-1,) + selected.state_shape)
    flat_times = jnp.broadcast_to(
        selected.time_grid.times[:-1], (count, selected.time_grid.num_steps)
    ).reshape((-1,))
    policy_actions = jax.vmap(action_at)(flat_times, flat_states).reshape(
        selected.actions.shape
    )
    comparable = selected.successful.reshape(
        (count,) + (1,) * (selected.actions.ndim - 1)
    )
    action_error = jnp.max(
        jnp.where(comparable, jnp.abs(policy_actions - selected.actions), 0.0),
        initial=0.0,
    )
    if not bool(jnp.isfinite(action_error)) or float(action_error) > tolerance:
        raise ValueError(
            "frozen_policy does not reproduce the selected physical actions."
        )

    state_axes = tuple(range(2, selected.states.ndim))
    state_finite = jnp.all(jnp.isfinite(selected.states), axis=state_axes)
    node_valid = selected.successful[:, None] & state_finite
    batch_path_id = selected.coupling_id
    process_id = controlled_problem.problem_id
    bsde_paths = BSDEPathBatch(
        selected.time_grid.times,
        selected.states,
        selected.noise_paths,
        sample_shape=(selected.path_count,),
        state_shape=selected.state_shape,
        noise_shape=selected.noise_shape,
        path_id=batch_path_id,
        process_id=process_id,
        valid=node_valid,
        metadata={
            "time_id": selected.time_grid.time_id,
            "realization_ids": selected.realization_ids,
            "coupling_id": selected.coupling_id,
            "policy_id": selected.policy_id,
            "independence_labels": selected.independence_labels,
            "sample_role": path_role,
            "physical_actions": selected.actions,
            "physical_action_shape": selected.action_shape,
        },
    )

    def closed_drift(time, state, args):
        del args
        action = action_at(time, state)
        value = jnp.asarray(
            controlled_drift(time, state, action, controlled_problem.args)
        )
        if value.shape != controlled_problem.state_shape:
            raise ValueError("controlled_drift returned an incompatible state shape.")
        return value

    def closed_diffusion(time, state, args):
        del args
        action = action_at(time, state)
        value = jnp.asarray(
            controlled_diffusion(time, state, action, controlled_problem.args)
        )
        expected = controlled_problem.state_shape + controlled_problem.noise_shape
        if value.shape != expected:
            raise ValueError(
                "controlled_diffusion must return state_shape + noise_shape."
            )
        return value

    probe_time = selected.time_grid.times[0]
    probe_state = selected.states[0, 0]
    closed_drift(probe_time, probe_state, None)
    closed_diffusion(probe_time, probe_state, None)

    def generator(time, state, value, z, args):
        del value, z, args
        context = _policy_context(selected, time)
        action = action_at(time, state)
        cost = jnp.asarray(
            controlled_problem.stage_cost(context, state, action, controlled_problem.args)
        )
        if cost.shape != ():
            raise ValueError("controlled stage_cost must return a scalar.")
        return (cost / (context.target - context.source))[None]

    def terminal(state, args):
        del args
        value = jnp.asarray(
            controlled_problem.terminal_cost(
                selected.time_grid.times[-1], state, controlled_problem.args
            )
        )
        if value.shape != ():
            raise ValueError("controlled terminal_cost must return a scalar.")
        return value[None]

    bsde_problem = BSDEProblem(
        lambda key: bsde_paths,
        closed_drift,
        closed_diffusion,
        generator,
        terminal,
        state_shape=selected.state_shape,
        noise_shape=selected.noise_shape,
        output_shape=(1,),
        problem_id=f"{result.problem_id}:frozen-policy-bsde",
        process_id=process_id,
        args=None,
        time_label="t",
        state_label="x",
    )

    def value_predictor(time, state):
        return result.predict(time, state)[None]

    control_mode = "autodiff" if z_predictor is None else "explicit"
    evaluation = evaluate_bsde(
        bsde_problem,
        bsde_paths,
        value_predictor,
        control_predictor=z_predictor,
        control_mode=control_mode,
        quadrature="left",
    )
    sampling_plan = FeynmanKacSamplingPlan(
        initial_time=float(selected.time_grid.times[0]),
        terminal_time=float(selected.time_grid.times[-1]),
        sampling_mode="trajectory_nodes",
        num_paths_per_query=1,
        num_time_steps=selected.time_grid.num_steps,
        quadrature="left",
        control_target_mode="none",
        antithetic=False,
        time_weighting="uniform",
        refresh_mode="fixed",
        plan_id=f"{result.plan_id}:current:{path_role}",
    )
    raw_labels = trajectory_node_feynman_kac_labels(
        bsde_problem,
        bsde_paths,
        sampling_plan,
        source_value=value_predictor,
        source_control=z_predictor,
    )
    label_metadata = dict(raw_labels.metadata)
    label_metadata.update(
        {
            "time_id": selected.time_grid.time_id,
            "realization_ids": selected.realization_ids,
            "coupling_id": selected.coupling_id,
            "policy_id": selected.policy_id,
            "sample_role": path_role,
        }
    )
    labels = FeynmanKacLabelBatch(
        raw_labels.query_times,
        raw_labels.query_states,
        raw_labels.value_targets,
        state_shape=raw_labels.state_shape,
        noise_shape=raw_labels.noise_shape,
        output_shape=raw_labels.output_shape,
        problem_id=raw_labels.problem_id,
        process_id=raw_labels.process_id,
        plan_id=raw_labels.plan_id,
        value_standard_errors=raw_labels.value_standard_errors,
        control_targets=raw_labels.control_targets,
        control_standard_errors=raw_labels.control_standard_errors,
        valid=raw_labels.valid,
        control_valid=raw_labels.control_valid,
        sample_weights=raw_labels.sample_weights,
        cluster_ids=jnp.repeat(
            selected.independence_labels, selected.time_grid.num_times
        ),
        source_path_count=selected.path_count,
        metadata=label_metadata,
    )
    return FittedBellmanBSDEBridge(
        fitted_result=result,
        controlled_problem=controlled_problem,
        controlled_paths=selected,
        bsde_problem=bsde_problem,
        bsde_paths=bsde_paths,
        evaluation=evaluation,
        feynman_kac_labels=labels,
        physical_actions=selected.actions,
        martingale_integrands=evaluation.controls,
        independence_labels=selected.independence_labels,
        realization_ids=selected.realization_ids,
        coupling_id=selected.coupling_id,
        path_id=batch_path_id,
        process_id=process_id,
        time_id=selected.time_grid.time_id,
        policy_id=selected.policy_id,
        sample_role=path_role,
        action_shape=selected.action_shape,
        z_shape=(1,) + selected.noise_shape,
        action_is_martingale_integrand=False,
        policy_improvement_performed=False,
    )


__all__ = [
    "bridge_fitted_bellman_to_bsde",
    "evaluate_fitted_bellman",
    "fit_frozen_policy_bellman",
    "FittedBellmanBSDEBridge",
    "FittedBellmanPlan",
    "FittedBellmanPrepared",
    "FittedBellmanProblem",
    "FittedBellmanResult",
    "FittedBellmanStatus",
    "FROZEN_POLICY_FITTED_BELLMAN",
    "prepare_fitted_bellman",
]
