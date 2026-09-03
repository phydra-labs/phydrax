#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Residual-globalized iterative local-quadratic feedback games."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import DiscreteStepContext
from .._lqr import AffineFeedbackPolicy
from .._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_SUCCESS,
    ControlTrajectory,
)
from ._linear_quadratic import finite_horizon_lq_feedback_nash
from ._local_lq import (
    LocalAffineGamePolicy,
    LocalAffineGameSuggestion,
    suggest_local_affine_game_policy,
)
from ._nonlinear import (
    _first_false,
    _stage_cost_vector,
    _terminal_cost_vector,
    DeterministicFeedbackGameProblem,
    GamePolicyEvaluation,
    GamePolicyEvaluationStatus,
    ILQGameScaling,
    nominal_nash_residual,
    NominalNashResidual,
)


_CERTIFICATE = "LOCAL_NOMINAL_NASH_STATIONARY"
_METHOD = "residual-globalized-iterative-local-quadratic-feedback-game"
_ACCEPTANCE_METHOD = "original-unregularized-dimensionless-residual-armijo"
_DIFFERENTIATION_METHOD = "fixed-capacity-unrolled-no-implicit-differentiation"


class ILQFeedbackGameStatus(IntEnum):
    """Stable case-local termination codes for the iLQ game solve."""

    SUCCESS = 0
    MAX_ITERATIONS = 1
    INITIAL_POLICY_EVALUATION_FAILED = 2
    NOMINAL_RESIDUAL_FAILED = 3
    LOCAL_LQ_FAILED = 4
    LINE_SEARCH_FAILED = 5
    FINAL_UNREGULARIZED_LOCAL_LQ_FAILED = 6

    # Explicit compatibility spellings; aliases do not create new outcomes.
    INITIAL_EVALUATION_FAILED = INITIAL_POLICY_EVALUATION_FAILED
    RESIDUAL_EVALUATION_FAILED = NOMINAL_RESIDUAL_FAILED
    LOCAL_SUGGESTION_FAILED = LOCAL_LQ_FAILED
    UNREGULARIZED_LOCAL_LQ_FAILED = FINAL_UNREGULARIZED_LOCAL_LQ_FAILED
    INITIAL_ROLLOUT_FAILED = INITIAL_POLICY_EVALUATION_FAILED
    RESIDUAL_FAILED = NOMINAL_RESIDUAL_FAILED
    LOCAL_QUADRATIC_SUGGESTION_FAILED = LOCAL_LQ_FAILED
    UNREGULARIZED_SUGGESTION_FAILED = FINAL_UNREGULARIZED_LOCAL_LQ_FAILED
    REGULARIZED_ONLY = FINAL_UNREGULARIZED_LOCAL_LQ_FAILED


class ILQFeedbackGameTrialReason(IntEnum):
    """Acceptance-critical outcome for one declared geometric alpha trial."""

    NOT_EVALUATED = 0
    ACCEPTED = 1
    NONFINITE_OR_INVALID_ROLLOUT = 2
    NONFINITE_OR_INVALID_RESIDUAL = 3
    NONFINITE_SCALED_STEP = 4
    SCALED_STATE_STEP_GUARD_EXCEEDED = 5
    SCALED_CONTROL_STEP_GUARD_EXCEEDED = 6
    ORIGINAL_RESIDUAL_ARMIJO_FAILED = 7

    INVALID_ROLLOUT = NONFINITE_OR_INVALID_ROLLOUT
    INVALID_RESIDUAL = NONFINITE_OR_INVALID_RESIDUAL
    NONFINITE_STEP = NONFINITE_SCALED_STEP
    STATE_STEP_LIMIT_EXCEEDED = SCALED_STATE_STEP_GUARD_EXCEEDED
    CONTROL_STEP_LIMIT_EXCEEDED = SCALED_CONTROL_STEP_GUARD_EXCEEDED
    ARMIJO_FAILED = ORIGINAL_RESIDUAL_ARMIJO_FAILED
    ROLLOUT_FAILED = NONFINITE_OR_INVALID_ROLLOUT
    RESIDUAL_FAILED = NONFINITE_OR_INVALID_RESIDUAL
    STATE_STEP_GUARD_EXCEEDED = SCALED_STATE_STEP_GUARD_EXCEEDED
    CONTROL_STEP_GUARD_EXCEEDED = SCALED_CONTROL_STEP_GUARD_EXCEEDED
    RESIDUAL_ARMIJO_FAILED = ORIGINAL_RESIDUAL_ARMIJO_FAILED


class ILQFeedbackGamePlan(StrictModule, NonTrainableState):
    """Static capacities, physical scaling, and acceptance policy."""

    scaling: ILQGameScaling
    maximum_iterations: int = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    step_tolerance: float = eqx.field(static=True)
    dynamics_tolerance: float = eqx.field(static=True)
    maximum_scaled_state_step: float = eqx.field(static=True)
    maximum_scaled_control_step: float = eqx.field(static=True)
    initial_alpha: float = eqx.field(static=True)
    alpha_contraction: float = eqx.field(static=True)
    armijo: float = eqx.field(static=True)
    initial_proximal_regularization: float = eqx.field(static=True)
    proximal_regularization_growth: float = eqx.field(static=True)
    maximum_proximal_regularization: float = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    curvature_tolerance: float = eqx.field(static=True)
    rank_relative_tolerance: float | None = eqx.field(static=True)
    rank_absolute_tolerance: float | None = eqx.field(static=True)
    maximum_condition: float | None = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    scaling_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def max_iterations(self) -> int:
        return self.maximum_iterations

    @property
    def line_search_steps(self) -> int:
        return self.maximum_line_search_steps

    @property
    def initial_step_size(self) -> float:
        return self.initial_alpha

    @property
    def line_search_decay(self) -> float:
        return self.alpha_contraction

    @property
    def maximum_state_step(self) -> float:
        return self.maximum_scaled_state_step

    @property
    def maximum_control_step(self) -> float:
        return self.maximum_scaled_control_step


class PreparedILQFeedbackGame(StrictModule, NonTrainableState):
    """One planned topology bound to refreshable problem and policy arrays."""

    plan: ILQFeedbackGamePlan
    problem: DeterministicFeedbackGameProblem
    initial_policy: LocalAffineGamePolicy | AffineFeedbackPolicy
    materialization_version: Array
    initial_policy_id: str = eqx.field(static=True)
    initial_policy_kind: str = eqx.field(static=True)
    materialization_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def numeric_version(self) -> Array:
        return self.materialization_version


class LocalNominalNashDiagnostics(StrictModule):
    """Fixed-capacity evidence for a local nominal-stationarity result.

    Iteration histories have shape ``case_shape + (maximum_iterations, ...)``;
    trial histories append ``maximum_line_search_steps`` before their payload.
    Entries whose corresponding validity mask is false are padding and never
    participate in acceptance. Player objectives are evidence only.
    """

    status: Array
    converged: Array
    valid: Array
    iterations: Array
    accepted_iterations: Array
    history_valid: Array
    accepted_history: Array
    evaluation_valid_history: Array
    residual_valid_history: Array
    unregularized_local_valid_history: Array
    unregularized_local_status_history: Array
    direction_valid_history: Array
    direction_status_history: Array
    proximal_regularization_history: Array
    residual_merit_history: Array
    stationarity_rms_history: Array
    stationarity_infinity_history: Array
    dynamics_rms_history: Array
    dynamics_infinity_history: Array
    player_cost_history: Array
    accepted_alpha_history: Array
    accepted_state_step_rms_history: Array
    accepted_state_step_infinity_history: Array
    accepted_control_step_rms_history: Array
    accepted_control_step_infinity_history: Array
    line_search_evaluations_history: Array
    trial_history_valid: Array
    trial_reason_history: Array
    trial_alpha_history: Array
    trial_evaluation_valid_history: Array
    trial_residual_valid_history: Array
    trial_residual_merit_history: Array
    trial_armijo_bound_history: Array
    trial_state_step_rms_history: Array
    trial_state_step_infinity_history: Array
    trial_control_step_rms_history: Array
    trial_control_step_infinity_history: Array
    trial_player_cost_history: Array
    final_residual_merit: Array
    final_stationarity_rms: Array
    final_stationarity_infinity: Array
    final_dynamics_rms: Array
    final_dynamics_infinity: Array
    final_state_step_rms: Array
    final_state_step_infinity: Array
    final_control_step_rms: Array
    final_control_step_infinity: Array
    final_unregularized_local_valid: Array
    final_unregularized_local_status: Array
    certificate_valid: Array
    certificate: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    acceptance_method: str = eqx.field(static=True)
    differentiation_method: str = eqx.field(static=True)
    player_costs_used_for_acceptance: bool = eqx.field(static=True)
    feedback_nash_claimed: bool = eqx.field(static=True)
    global_convergence_claimed: bool = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.certificate_valid

    @property
    def iteration_count(self) -> Array:
        return self.iterations

    @property
    def accepted_count(self) -> Array:
        return self.accepted_iterations

    @property
    def regularization_history(self) -> Array:
        return self.proximal_regularization_history

    @property
    def step_size_history(self) -> Array:
        return self.accepted_alpha_history

    @property
    def trial_status_history(self) -> Array:
        return self.trial_reason_history

    @property
    def player_costs_history(self) -> Array:
        return self.player_cost_history

    @property
    def residual_norm_history(self) -> Array:
        return self.stationarity_infinity_history


class LocalNominalNashResult(StrictModule):
    """Accepted policy profile and exact local nominal-stationarity evidence."""

    plan: ILQFeedbackGamePlan
    policy: LocalAffineGamePolicy
    evaluation: GamePolicyEvaluation
    residual: NominalNashResidual
    local_suggestion: LocalAffineGameSuggestion
    diagnostics: LocalNominalNashDiagnostics
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    certificate: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    materialization_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Return cases carrying the exact local certificate."""

        return (
            self.evaluation.successful
            & self.residual.successful
            & self.diagnostics.certificate_valid
            & (self.diagnostics.status == int(ILQFeedbackGameStatus.SUCCESS))
        )

    @property
    def valid(self) -> Array:
        return self.successful

    @property
    def status(self) -> Array:
        return self.diagnostics.status

    @property
    def converged(self) -> Array:
        return self.diagnostics.converged

    @property
    def trajectory(self) -> ControlTrajectory:
        return self.evaluation.trajectory

    @property
    def player_costs(self) -> Array:
        return self.evaluation.total_costs

    @property
    def total_costs(self) -> Array:
        return self.evaluation.total_costs

    @property
    def final_unregularized_suggestion(self) -> LocalAffineGameSuggestion:
        return self.local_suggestion

    @property
    def certificate_label(self) -> str:
        return self.certificate

    @property
    def scaling(self) -> ILQGameScaling:
        return self.plan.scaling

    @property
    def final_evaluation(self) -> GamePolicyEvaluation:
        return self.evaluation

    @property
    def final_residual(self) -> NominalNashResidual:
        return self.residual


def _positive_integer(value: int, name: str, /) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _finite_nonnegative(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _finite_positive(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _optional_finite_positive(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    return _finite_positive(value, name)


def _optional_finite_nonnegative(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, name)


def _policy_identity(
    policy: LocalAffineGamePolicy | AffineFeedbackPolicy, /
) -> tuple[str, str]:
    if isinstance(policy, LocalAffineGamePolicy):
        return policy.policy_id, "local-affine-game-policy"
    if isinstance(policy, AffineFeedbackPolicy):
        return policy.parameterization_id, "affine-feedback-policy"
    raise TypeError(
        "initial_policy must be LocalAffineGamePolicy or AffineFeedbackPolicy."
    )


def _validate_scaling(
    problem: DeterministicFeedbackGameProblem, scaling: ILQGameScaling, /
) -> None:
    if not isinstance(scaling, ILQGameScaling):
        raise TypeError("scaling must be an ILQGameScaling.")
    if scaling.state_shape != problem.state_shape:
        raise ValueError("scaling state_shape must match the problem state_shape.")
    if scaling.control_shape != problem.control_shape:
        raise ValueError("scaling control_shape must match the problem control_shape.")
    if scaling.num_players != problem.num_players:
        raise ValueError("scaling must contain exactly one cost scale per player.")


def _validate_initial_policy(
    problem: DeterministicFeedbackGameProblem,
    policy: LocalAffineGamePolicy | AffineFeedbackPolicy,
    /,
) -> None:
    _policy_identity(policy)
    system_layout = problem.dynamics.system.input_layout
    assert system_layout is not None
    if policy.state_shape != problem.state_shape:
        raise ValueError("initial_policy state_shape must match the problem.")
    if policy.control_shape != problem.control_shape:
        raise ValueError("initial_policy control_shape must match the problem.")
    if policy.case_shape not in ((), problem.case_shape):
        raise ValueError(
            "initial_policy case_shape must be scalar or exactly problem.case_shape."
        )
    if isinstance(policy, LocalAffineGamePolicy):
        if policy.time_grid.time_id != problem.time_grid.time_id:
            raise ValueError(
                "initial_policy time-grid identity must exactly match the problem."
            )
        if policy.input_layout.layout_id != system_layout.layout_id:
            raise ValueError(
                "initial_policy input-layout identity must exactly match the dynamics."
            )
        if policy.partition.partition_id != problem.partition.partition_id:
            raise ValueError(
                "initial_policy partition identity must exactly match the problem."
            )
    else:
        if not policy.finite_horizon or policy.time_grid is None:
            raise ValueError("initial AffineFeedbackPolicy must be finite-horizon.")
        if policy.time_grid.time_id != problem.time_grid.time_id:
            raise ValueError(
                "initial_policy time-grid identity must exactly match the problem."
            )


def _topology_record(
    problem: DeterministicFeedbackGameProblem,
    scaling: ILQGameScaling,
    /,
) -> dict[str, Any]:
    return {
        "problem": problem.problem_id,
        "dynamics": problem.dynamics.dynamics_id,
        "time": problem.time_grid.time_id,
        "partition": problem.partition.partition_id,
        "scaling": scaling.scaling_id,
        "case_shape": list(problem.case_shape),
        "horizon": problem.time_grid.num_steps,
        "state_shape": list(problem.state_shape),
        "control_shape": list(problem.control_shape),
        "num_players": problem.num_players,
        "stage_cost_semantics": problem.stage_cost_semantics,
    }


def plan_ilq_feedback_game(
    problem: DeterministicFeedbackGameProblem,
    scaling: ILQGameScaling,
    /,
    *,
    maximum_iterations: int = 50,
    maximum_line_search_steps: int = 12,
    residual_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-8,
    dynamics_tolerance: float = 1.0e-10,
    maximum_scaled_state_step: float = 1.0e3,
    maximum_scaled_control_step: float = 1.0e3,
    initial_alpha: float = 1.0,
    alpha_contraction: float = 0.5,
    armijo: float = 1.0e-4,
    initial_proximal_regularization: float = 0.0,
    proximal_regularization_growth: float = 10.0,
    maximum_proximal_regularization: float = 1.0e8,
    symmetry_tolerance: float = 1.0e-10,
    curvature_tolerance: float = 1.0e-10,
    rank_relative_tolerance: float | None = None,
    rank_absolute_tolerance: float | None = None,
    maximum_condition: float | None = None,
    max_iterations: int | None = None,
    line_search_steps: int | None = None,
    initial_step_size: float | None = None,
    line_search_decay: float | None = None,
    maximum_state_step: float | None = None,
    maximum_control_step: float | None = None,
    proximal_regularization: float | None = None,
    regularization_growth: float | None = None,
    maximum_regularization: float | None = None,
    armijo_coefficient: float | None = None,
) -> ILQFeedbackGamePlan:
    """Plan one homogeneous fixed-capacity residual-globalized solve.

    The final keyword arguments are spelling-compatible aliases for the
    established control lifecycle names. Supplying an alias replaces its
    corresponding canonical value.
    """

    if not isinstance(problem, DeterministicFeedbackGameProblem):
        raise TypeError("problem must be a DeterministicFeedbackGameProblem.")
    _validate_scaling(problem, scaling)
    if max_iterations is not None:
        maximum_iterations = max_iterations
    if line_search_steps is not None:
        maximum_line_search_steps = line_search_steps
    if initial_step_size is not None:
        initial_alpha = initial_step_size
    if line_search_decay is not None:
        alpha_contraction = line_search_decay
    if maximum_state_step is not None:
        maximum_scaled_state_step = maximum_state_step
    if maximum_control_step is not None:
        maximum_scaled_control_step = maximum_control_step
    if proximal_regularization is not None:
        initial_proximal_regularization = proximal_regularization
    if regularization_growth is not None:
        proximal_regularization_growth = regularization_growth
    if maximum_regularization is not None:
        maximum_proximal_regularization = maximum_regularization
    if armijo_coefficient is not None:
        armijo = armijo_coefficient

    iterations = _positive_integer(maximum_iterations, "maximum_iterations")
    searches = _positive_integer(maximum_line_search_steps, "maximum_line_search_steps")
    residual = _finite_nonnegative(residual_tolerance, "residual_tolerance")
    step = _finite_nonnegative(step_tolerance, "step_tolerance")
    dynamics = _finite_nonnegative(dynamics_tolerance, "dynamics_tolerance")
    state_guard = _finite_positive(maximum_scaled_state_step, "maximum_scaled_state_step")
    control_guard = _finite_positive(
        maximum_scaled_control_step, "maximum_scaled_control_step"
    )
    alpha = _finite_positive(initial_alpha, "initial_alpha")
    if alpha > 1.0:
        raise ValueError("initial_alpha must not exceed one.")
    contraction = _finite_positive(alpha_contraction, "alpha_contraction")
    if contraction >= 1.0:
        raise ValueError("alpha_contraction must be strictly less than one.")
    armijo_value = _finite_positive(armijo, "armijo")
    if armijo_value >= 1.0:
        raise ValueError("armijo must be strictly less than one.")
    initial_regularization = _finite_nonnegative(
        initial_proximal_regularization, "initial_proximal_regularization"
    )
    growth = _finite_positive(
        proximal_regularization_growth, "proximal_regularization_growth"
    )
    if growth <= 1.0:
        raise ValueError("proximal_regularization_growth must exceed one.")
    maximum_regularization = _finite_nonnegative(
        maximum_proximal_regularization, "maximum_proximal_regularization"
    )
    if maximum_regularization < initial_regularization:
        raise ValueError(
            "maximum_proximal_regularization must not be smaller than the initial value."
        )
    symmetry = _finite_nonnegative(symmetry_tolerance, "symmetry_tolerance")
    curvature = _finite_nonnegative(curvature_tolerance, "curvature_tolerance")
    relative_rank = _optional_finite_nonnegative(
        rank_relative_tolerance, "rank_relative_tolerance"
    )
    absolute_rank = _optional_finite_nonnegative(
        rank_absolute_tolerance, "rank_absolute_tolerance"
    )
    condition = _optional_finite_positive(maximum_condition, "maximum_condition")
    if condition is not None and condition <= 1.0:
        raise ValueError("maximum_condition must exceed one or be None.")

    topology = _topology_record(problem, scaling)
    policy_record = {
        "maximum_iterations": iterations,
        "maximum_line_search_steps": searches,
        "residual_tolerance": residual,
        "step_tolerance": step,
        "dynamics_tolerance": dynamics,
        "maximum_scaled_state_step": state_guard,
        "maximum_scaled_control_step": control_guard,
        "initial_alpha": alpha,
        "alpha_contraction": contraction,
        "armijo": armijo_value,
        "initial_proximal_regularization": initial_regularization,
        "proximal_regularization_growth": growth,
        "maximum_proximal_regularization": maximum_regularization,
        "symmetry_tolerance": symmetry,
        "curvature_tolerance": curvature,
        "rank_relative_tolerance": relative_rank,
        "rank_absolute_tolerance": absolute_rank,
        "maximum_condition": condition,
    }
    plan_id = "ilq-feedback-game-plan:" + canonical_fingerprint(
        {"kind": "ilq-feedback-game-plan", "topology": topology, **policy_record}
    )
    return ILQFeedbackGamePlan(
        scaling,
        iterations,
        searches,
        residual,
        step,
        dynamics,
        state_guard,
        control_guard,
        alpha,
        contraction,
        armijo_value,
        initial_regularization,
        growth,
        maximum_regularization,
        symmetry,
        curvature,
        relative_rank,
        absolute_rank,
        condition,
        problem.case_shape,
        problem.time_grid.num_steps,
        problem.state_size,
        problem.control_size,
        problem.num_players,
        problem.problem_id,
        problem.dynamics.dynamics_id,
        problem.time_grid.time_id,
        problem.partition.partition_id,
        scaling.scaling_id,
        plan_id,
    )


def _validate_topology(
    plan: ILQFeedbackGamePlan,
    problem: DeterministicFeedbackGameProblem,
    /,
) -> None:
    if not isinstance(plan, ILQFeedbackGamePlan):
        raise TypeError("plan must be an ILQFeedbackGamePlan.")
    if not isinstance(problem, DeterministicFeedbackGameProblem):
        raise TypeError("problem must be a DeterministicFeedbackGameProblem.")
    expected = (
        plan.problem_id,
        plan.dynamics_id,
        plan.time_id,
        plan.partition_id,
        plan.case_shape,
        plan.horizon,
        plan.state_size,
        plan.control_size,
        plan.num_players,
    )
    actual = (
        problem.problem_id,
        problem.dynamics.dynamics_id,
        problem.time_grid.time_id,
        problem.partition.partition_id,
        problem.case_shape,
        problem.time_grid.num_steps,
        problem.state_size,
        problem.control_size,
        problem.num_players,
    )
    if actual != expected:
        raise ValueError("problem does not match the planned static game topology.")
    _validate_scaling(problem, plan.scaling)


def _validate_refresh_structure(
    current: DeterministicFeedbackGameProblem,
    replacement: DeterministicFeedbackGameProblem,
    /,
) -> None:
    if current.dynamics.method_id != replacement.dynamics.method_id:
        raise ValueError("refresh cannot change the discrete dynamics method.")
    if current.dynamics.system.transition is not replacement.dynamics.system.transition:
        raise ValueError("refresh cannot change the discrete transition callback.")
    if len(current.stage_costs) != len(replacement.stage_costs) or any(
        old is not new
        for old, new in zip(current.stage_costs, replacement.stage_costs, strict=True)
    ):
        raise ValueError("refresh cannot change stage-cost callbacks.")
    if len(current.terminal_costs) != len(replacement.terminal_costs) or any(
        old is not new
        for old, new in zip(
            current.terminal_costs, replacement.terminal_costs, strict=True
        )
    ):
        raise ValueError("refresh cannot change terminal-cost callbacks.")
    if jax.tree_util.tree_structure(current.args) != jax.tree_util.tree_structure(
        replacement.args
    ):
        raise ValueError("refresh cannot change the static args PyTree structure.")


def _materialization_identity(
    plan: ILQFeedbackGamePlan,
    problem: DeterministicFeedbackGameProblem,
    policy: LocalAffineGamePolicy | AffineFeedbackPolicy,
    version: int,
    /,
) -> tuple[str, str]:
    policy_id, policy_kind = _policy_identity(policy)
    materialization = "ilq-feedback-game-materialization:" + canonical_fingerprint(
        {
            "plan": plan.plan_id,
            "version": version,
            "initial_policy_id": policy_id,
            "initial_policy_kind": policy_kind,
            "problem_arrays": array_tree_fingerprint(problem),
            "policy_arrays": array_tree_fingerprint(policy),
        }
    )
    prepared = "prepared-ilq-feedback-game:" + canonical_fingerprint(
        {"plan": plan.plan_id, "materialization": materialization}
    )
    return materialization, prepared


def prepare_ilq_feedback_game(
    plan: ILQFeedbackGamePlan,
    problem: DeterministicFeedbackGameProblem,
    initial_policy: LocalAffineGamePolicy | AffineFeedbackPolicy,
    /,
) -> PreparedILQFeedbackGame:
    """Bind one compatible physical policy to a planned static topology."""

    _validate_topology(plan, problem)
    _validate_initial_policy(problem, initial_policy)
    policy_id, policy_kind = _policy_identity(initial_policy)
    materialization_id, prepared_id = _materialization_identity(
        plan, problem, initial_policy, 0
    )
    return PreparedILQFeedbackGame(
        plan,
        problem,
        initial_policy,
        jnp.asarray(0, dtype=jnp.int32),
        policy_id,
        policy_kind,
        materialization_id,
        prepared_id,
    )


def refresh_ilq_feedback_game(
    prepared: PreparedILQFeedbackGame,
    *,
    problem: DeterministicFeedbackGameProblem | None = None,
    initial_policy: LocalAffineGamePolicy | AffineFeedbackPolicy | None = None,
) -> PreparedILQFeedbackGame:
    """Refresh dynamic data without changing any planned static provenance."""

    if not isinstance(prepared, PreparedILQFeedbackGame):
        raise TypeError("prepared must be a PreparedILQFeedbackGame.")
    next_problem = prepared.problem if problem is None else problem
    next_policy = prepared.initial_policy if initial_policy is None else initial_policy
    _validate_topology(prepared.plan, next_problem)
    if problem is not None:
        _validate_refresh_structure(prepared.problem, next_problem)
    _validate_initial_policy(next_problem, next_policy)
    policy_id, policy_kind = _policy_identity(next_policy)
    if policy_kind != prepared.initial_policy_kind:
        raise ValueError("refresh cannot change the initial policy representation kind.")
    version = int(np.asarray(prepared.materialization_version)) + 1
    materialization_id, prepared_id = _materialization_identity(
        prepared.plan, next_problem, next_policy, version
    )
    return PreparedILQFeedbackGame(
        prepared.plan,
        next_problem,
        next_policy,
        jnp.asarray(version, dtype=jnp.int32),
        policy_id,
        policy_kind,
        materialization_id,
        prepared_id,
    )


def _broadcast_policy_array(
    value: Array, policy_cases: tuple[int, ...], problem_cases: tuple[int, ...], /
) -> Array:
    if policy_cases == problem_cases:
        return value
    return jnp.broadcast_to(value, problem_cases + value.shape)


def _initial_policy_arrays(
    problem: DeterministicFeedbackGameProblem,
    policy: LocalAffineGamePolicy | AffineFeedbackPolicy,
    /,
) -> tuple[str, Array, Array, Array, Array, Array]:
    cases = problem.case_shape
    if isinstance(policy, LocalAffineGamePolicy):
        states = _broadcast_policy_array(policy.nominal_states, policy.case_shape, cases)
        controls = _broadcast_policy_array(
            policy.nominal_controls, policy.case_shape, cases
        )
        gain = _broadcast_policy_array(policy.feedback_gain, policy.case_shape, cases)
        feedforward = _broadcast_policy_array(
            policy.feedforward, policy.case_shape, cases
        )
        return (
            "local",
            states,
            controls,
            gain,
            feedforward,
            policy.feedforward_scale,
        )
    gain = _broadcast_policy_array(policy.feedback_gain, policy.case_shape, cases)
    feedforward = _broadcast_policy_array(policy.feedforward, policy.case_shape, cases)
    empty_states = jnp.zeros(
        cases + (problem.time_grid.num_times, problem.state_size), dtype=gain.dtype
    )
    empty_controls = jnp.zeros(
        cases + (problem.time_grid.num_steps, problem.control_size), dtype=gain.dtype
    )
    return (
        "absolute",
        empty_states,
        empty_controls,
        gain,
        feedforward,
        jnp.asarray(1.0, dtype=gain.dtype),
    )


def _evaluate_affine_profile(
    problem: DeterministicFeedbackGameProblem,
    *,
    law_kind: str,
    nominal_states: Array,
    nominal_controls: Array,
    feedback_gain: Array,
    feedforward: Array,
    feedforward_scale: Array,
    policy_id: str,
) -> GamePolicyEvaluation:
    cases = problem.case_shape
    count = prod(cases) if cases else 1
    horizon = problem.time_grid.num_steps
    players = problem.num_players
    state_size = problem.state_size
    control_size = problem.control_size
    initial = problem.initial_state.reshape((count, state_size))
    nominal_states_flat = nominal_states.reshape((count, horizon + 1, state_size))
    nominal_controls_flat = nominal_controls.reshape((count, horizon, control_size))
    gain_flat = feedback_gain.reshape((count, horizon, control_size, state_size))
    feedforward_flat = feedforward.reshape((count, horizon, control_size))
    initial_valid = jnp.all(jnp.isfinite(initial), axis=-1)
    initial_status = jnp.where(
        initial_valid,
        int(GamePolicyEvaluationStatus.SUCCESS),
        int(GamePolicyEvaluationStatus.NONFINITE_INITIAL_STATE),
    ).astype(jnp.int32)
    initial_failed_step = jnp.where(initial_valid, -1, 0).astype(jnp.int32)
    initial_failed_player = jnp.full((count,), -1, dtype=jnp.int32)

    def scan_step(carry, step_index):
        state, trajectory_active, status, failed_step, failed_player = carry
        context = DiscreteStepContext(
            problem.time_grid.times[step_index],
            problem.time_grid.times[step_index + 1],
            step_index,
        )
        gain = gain_flat[:, step_index]
        if law_kind == "local":
            control = (
                nominal_controls_flat[:, step_index]
                + ein.contract(
                    "cmn,cn->cm",
                    gain,
                    state - nominal_states_flat[:, step_index],
                )
                + feedforward_scale * feedforward_flat[:, step_index]
            )
        else:
            control = (
                ein.contract("cmn,cn->cm", gain, state) + feedforward_flat[:, step_index]
            )
        control_finite = jnp.all(jnp.isfinite(control), axis=-1)
        stage = jax.vmap(
            lambda case_state, case_control: _stage_cost_vector(
                problem, context, case_state, case_control
            )
        )(state, control)
        stage_finite = jnp.isfinite(stage)
        candidate = jax.vmap(
            lambda case_state, case_control: problem.dynamics.system.evaluate(
                context,
                case_state,
                problem.args,
                inputs=case_control,
            )
        )(state, control)
        transition_finite = jnp.all(jnp.isfinite(candidate), axis=-1)
        stage_all_finite = jnp.all(stage_finite, axis=-1)
        stage_status = jnp.where(
            ~control_finite,
            int(GamePolicyEvaluationStatus.NONFINITE_POLICY_CONTROL),
            jnp.where(
                ~transition_finite,
                int(GamePolicyEvaluationStatus.NONFINITE_DYNAMICS_STATE),
                jnp.where(
                    ~stage_all_finite,
                    int(GamePolicyEvaluationStatus.NONFINITE_STAGE_COST),
                    int(GamePolicyEvaluationStatus.SUCCESS),
                ),
            ),
        ).astype(jnp.int32)
        first_here = (status == int(GamePolicyEvaluationStatus.SUCCESS)) & (
            stage_status != int(GamePolicyEvaluationStatus.SUCCESS)
        )
        cost_failure = first_here & (
            stage_status == int(GamePolicyEvaluationStatus.NONFINITE_STAGE_COST)
        )
        status = jnp.where(first_here, stage_status, status).astype(jnp.int32)
        failed_step = jnp.where(first_here, step_index, failed_step).astype(jnp.int32)
        failed_player = jnp.where(
            cost_failure, _first_false(stage_finite), failed_player
        ).astype(jnp.int32)
        next_active = trajectory_active & control_finite & transition_finite
        return (
            candidate,
            next_active,
            status,
            failed_step,
            failed_player,
        ), (
            candidate,
            control,
            next_active,
            control_finite,
            transition_finite,
            stage,
            stage_finite,
        )

    (_, _, status, failed_step, failed_player), output = jax.lax.scan(
        scan_step,
        (
            initial,
            initial_valid,
            initial_status,
            initial_failed_step,
            initial_failed_player,
        ),
        jnp.arange(horizon, dtype=jnp.int32),
    )
    (
        state_tail,
        controls_time,
        trajectory_valid_tail,
        control_valid_time,
        transition_valid_time,
        stage_time,
        stage_valid_time,
    ) = output
    final_state = state_tail[-1]
    terminal = jax.vmap(lambda state: _terminal_cost_vector(problem, state))(final_state)
    terminal_valid_flat = jnp.isfinite(terminal)
    terminal_all_finite = jnp.all(terminal_valid_flat, axis=-1)
    terminal_failure = (
        status == int(GamePolicyEvaluationStatus.SUCCESS)
    ) & ~terminal_all_finite
    status = jnp.where(
        terminal_failure,
        int(GamePolicyEvaluationStatus.NONFINITE_TERMINAL_COST),
        status,
    ).astype(jnp.int32)
    failed_step = jnp.where(terminal_failure, horizon, failed_step).astype(jnp.int32)
    failed_player = jnp.where(
        terminal_failure, _first_false(terminal_valid_flat), failed_player
    ).astype(jnp.int32)
    valid_flat = status == int(GamePolicyEvaluationStatus.SUCCESS)

    states_flat = jnp.concatenate((initial[None], state_tail), axis=0)
    trajectory_valid_flat = jnp.concatenate(
        (initial_valid[None], trajectory_valid_tail), axis=0
    )
    states = jnp.moveaxis(states_flat, 0, 1).reshape(cases + (horizon + 1, state_size))
    controls = jnp.moveaxis(controls_time, 0, 1).reshape(cases + (horizon, control_size))
    trajectory_valid = jnp.moveaxis(trajectory_valid_flat, 0, 1).reshape(
        cases + (horizon + 1,)
    )
    trajectory_status_flat = jnp.where(
        jnp.all(trajectory_valid_flat, axis=0),
        CONTROL_SUCCESS,
        CONTROL_DYNAMICS_FAILED,
    ).astype(jnp.int32)
    trajectory_status = trajectory_status_flat.reshape(cases)
    trajectory = ControlTrajectory(
        time_grid=problem.time_grid,
        states=states,
        controls=controls,
        valid=trajectory_valid,
        status=trajectory_status,
        backend_status=trajectory_status,
        case_shape=cases,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        problem_id=problem.problem_id,
        dynamics_id=problem.dynamics.dynamics_id,
        control_id=policy_id,
        backend_id="backend:jax:lax-scan",
        method_id=_METHOD,
        discretization_id=problem.time_grid.time_id,
        approximation_id="control:local-affine-game-policy",
    )
    stage = jnp.swapaxes(jnp.moveaxis(stage_time, 0, 1), -1, -2).reshape(
        cases + (players, horizon)
    )
    stage_valid = jnp.swapaxes(jnp.moveaxis(stage_valid_time, 0, 1), -1, -2).reshape(
        cases + (players, horizon)
    )
    terminal_costs = terminal.reshape(cases + (players,))
    terminal_valid = terminal_valid_flat.reshape(cases + (players,))
    total_unchecked = jnp.sum(stage, axis=-1) + terminal_costs
    valid = valid_flat.reshape(cases)
    total_costs = jnp.where(valid[..., None], total_unchecked, jnp.nan)
    return GamePolicyEvaluation(
        partition=problem.partition,
        trajectory=trajectory,
        stage_costs=stage,
        terminal_costs=terminal_costs,
        total_costs=total_costs,
        control_valid=jnp.moveaxis(control_valid_time, 0, 1).reshape(cases + (horizon,)),
        transition_valid=jnp.moveaxis(transition_valid_time, 0, 1).reshape(
            cases + (horizon,)
        ),
        stage_cost_valid=stage_valid,
        terminal_cost_valid=terminal_valid,
        first_failed_step=failed_step.reshape(cases),
        first_failed_player=failed_player.reshape(cases),
        valid=valid,
        status=status.reshape(cases),
        case_shape=cases,
        num_players=players,
        stage_cost_semantics=problem.stage_cost_semantics,
        policy_id=policy_id,
        evaluation_id=f"game-policy-evaluation:{problem.problem_id}:{policy_id}",
        method_id=_METHOD,
    )


def _case_where(mask: Array, on_true: Array, on_false: Array, /) -> Array:
    return jnp.where(
        mask.reshape(mask.shape + (1,) * (on_true.ndim - mask.ndim)),
        on_true,
        on_false,
    )


def _select_case_tree(
    mask: Array, on_true: Any, on_false: Any, cases: tuple[int, ...], /
) -> Any:
    def select(new, old):
        if not eqx.is_array(new):
            return new
        if not cases or tuple(new.shape[: len(cases)]) == cases:
            return _case_where(mask, new, old)
        return new

    return jax.tree_util.tree_map(select, on_true, on_false)


def _residual_merit(residual: NominalNashResidual, /) -> Array:
    return 0.5 * jnp.square(residual.rms_norm)


def _scaled_step_metrics(
    current: GamePolicyEvaluation,
    trial: GamePolicyEvaluation,
    scaling: ILQGameScaling,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    state_step = (
        trial.trajectory.states - current.trajectory.states
    ) / scaling.state_scales
    control_step = (
        trial.trajectory.controls - current.trajectory.controls
    ) / scaling.control_scales
    state_square = jnp.square(state_step)
    control_square = jnp.square(control_step)
    state_rms = jnp.sqrt(jnp.mean(state_square, axis=(-2, -1)))
    state_infinity = jnp.max(jnp.abs(state_step), axis=(-2, -1))
    control_rms = jnp.sqrt(jnp.mean(control_square, axis=(-2, -1)))
    control_infinity = jnp.max(jnp.abs(control_step), axis=(-2, -1))
    finite = (
        jnp.all(jnp.isfinite(state_step), axis=(-2, -1))
        & jnp.all(jnp.isfinite(control_step), axis=(-2, -1))
        & jnp.isfinite(state_rms)
        & jnp.isfinite(state_infinity)
        & jnp.isfinite(control_rms)
        & jnp.isfinite(control_infinity)
    )
    return state_rms, state_infinity, control_rms, control_infinity, finite


def _regularized_direction(
    plan: ILQFeedbackGamePlan,
    problem: DeterministicFeedbackGameProblem,
    suggestion: LocalAffineGameSuggestion,
    regularization: Array,
    policy_id: str,
    /,
):
    model = suggestion.model
    owner = jnp.asarray(problem.partition.control_owner, dtype=jnp.int32)
    ownership = jax.nn.one_hot(owner, problem.num_players, dtype=model.R.dtype).T
    diagonal_entries = (
        plan.scaling.cost_scales.astype(model.R.dtype)[:, None]
        * ownership
        / jnp.square(plan.scaling.control_scales.astype(model.R.dtype))[None, :]
    )
    diagonal = jax.vmap(jnp.diag)(diagonal_entries)
    cases = problem.case_shape
    shift = regularization.reshape(cases + (1, 1, 1, 1)) * diagonal.reshape(
        (1,) * len(cases)
        + (problem.num_players, 1, problem.control_size, problem.control_size)
    )
    return finite_horizon_lq_feedback_nash(
        model.A,
        model.B,
        model.Q,
        model.R + shift,
        model.terminal_Q,
        model.partition,
        dynamics_bias=model.dynamics_bias,
        state_control_cross=model.N,
        state_linear=model.q,
        control_linear=model.r,
        stage_constants=model.stage_constants,
        terminal_linear=model.terminal_q,
        terminal_constants=model.terminal_constants,
        time_grid=model.time_grid,
        policy_id=policy_id,
        symmetry_tolerance=plan.symmetry_tolerance,
        curvature_tolerance=plan.curvature_tolerance,
        rank_relative_tolerance=plan.rank_relative_tolerance,
        rank_absolute_tolerance=plan.rank_absolute_tolerance,
        maximum_condition=plan.maximum_condition,
    )


def _suggest(
    plan: ILQFeedbackGamePlan,
    problem: DeterministicFeedbackGameProblem,
    evaluation: GamePolicyEvaluation,
    suggestion_id: str,
    /,
) -> LocalAffineGameSuggestion:
    return suggest_local_affine_game_policy(
        problem,
        evaluation,
        plan.scaling,
        symmetry_tolerance=plan.symmetry_tolerance,
        curvature_tolerance=plan.curvature_tolerance,
        rank_relative_tolerance=plan.rank_relative_tolerance,
        rank_absolute_tolerance=plan.rank_absolute_tolerance,
        maximum_condition=plan.maximum_condition,
        suggestion_id=suggestion_id,
    )


def _evaluate_local_direction(
    problem: DeterministicFeedbackGameProblem,
    current: GamePolicyEvaluation,
    feedback_gain: Array,
    feedforward: Array,
    alpha: Array,
    policy_id: str,
    /,
) -> GamePolicyEvaluation:
    return _evaluate_affine_profile(
        problem,
        law_kind="local",
        nominal_states=current.trajectory.states,
        nominal_controls=current.trajectory.controls,
        feedback_gain=feedback_gain,
        feedforward=feedforward,
        feedforward_scale=alpha,
        policy_id=policy_id,
    )


def _stationary(
    plan: ILQFeedbackGamePlan,
    evaluation: GamePolicyEvaluation,
    residual: NominalNashResidual,
    suggestion: LocalAffineGameSuggestion,
    step_metrics: tuple[Array, Array, Array, Array, Array],
    /,
) -> Array:
    state_rms, state_infinity, control_rms, control_infinity, step_finite = step_metrics
    return (
        evaluation.successful
        & residual.successful
        & suggestion.successful
        & step_finite
        & (residual.stationarity_rms_norm <= plan.residual_tolerance)
        & (residual.stationarity_infinity_norm <= plan.residual_tolerance)
        & (residual.dynamics_defect_rms_norm <= plan.dynamics_tolerance)
        & (residual.dynamics_defect_infinity_norm <= plan.dynamics_tolerance)
        & (state_rms <= plan.step_tolerance)
        & (state_infinity <= plan.step_tolerance)
        & (control_rms <= plan.step_tolerance)
        & (control_infinity <= plan.step_tolerance)
    )


def _history_arrays(
    problem: DeterministicFeedbackGameProblem,
    plan: ILQFeedbackGamePlan,
    dtype,
    /,
) -> tuple[Array, ...]:
    iteration_shape = problem.case_shape + (plan.maximum_iterations,)
    trial_shape = iteration_shape + (plan.maximum_line_search_steps,)
    nan_iteration = jnp.full(iteration_shape, jnp.nan, dtype=dtype)
    nan_trial = jnp.full(trial_shape, jnp.nan, dtype=dtype)
    false_iteration = jnp.zeros(iteration_shape, dtype=bool)
    false_trial = jnp.zeros(trial_shape, dtype=bool)
    return (
        false_iteration,
        false_iteration,
        false_iteration,
        false_iteration,
        false_iteration,
        jnp.full(iteration_shape, -1, dtype=jnp.int32),
        false_iteration,
        jnp.full(iteration_shape, -1, dtype=jnp.int32),
        nan_iteration,
        nan_iteration,
        nan_iteration,
        nan_iteration,
        nan_iteration,
        nan_iteration,
        jnp.full(iteration_shape + (problem.num_players,), jnp.nan, dtype=dtype),
        nan_iteration,
        nan_iteration,
        nan_iteration,
        nan_iteration,
        nan_iteration,
        jnp.zeros(iteration_shape, dtype=jnp.int32),
        false_trial,
        jnp.full(
            trial_shape,
            int(ILQFeedbackGameTrialReason.NOT_EVALUATED),
            dtype=jnp.int32,
        ),
        nan_trial,
        false_trial,
        false_trial,
        nan_trial,
        nan_trial,
        nan_trial,
        nan_trial,
        nan_trial,
        nan_trial,
        jnp.full(trial_shape + (problem.num_players,), jnp.nan, dtype=dtype),
    )


def solve_prepared_ilq_feedback_game(
    prepared: PreparedILQFeedbackGame,
    /,
    *,
    policy_id: str | None = None,
    result_id: str | None = None,
) -> LocalNominalNashResult:
    """Run the fixed-capacity case-local iLQ game kernel.

    Every trial is judged only by the original, unregularized, dimensionless
    nominal residual. Proximal terms can change a direction but never the merit,
    acceptance guards, final residual, or final local-model requirement.
    """

    if not isinstance(prepared, PreparedILQFeedbackGame):
        raise TypeError("prepared must be a PreparedILQFeedbackGame.")
    plan = prepared.plan
    problem = prepared.problem
    runtime_policy_id = (
        f"ilq-feedback-game-policy:{prepared.materialization_id}"
        if policy_id is None
        else policy_id
    )
    if not isinstance(runtime_policy_id, str) or not runtime_policy_id:
        raise ValueError("policy_id must be a non-empty string or None.")
    runtime_result_id = (
        f"local-nominal-nash-result:{prepared.materialization_id}"
        if result_id is None
        else result_id
    )
    if not isinstance(runtime_result_id, str) or not runtime_result_id:
        raise ValueError("result_id must be a non-empty string or None.")
    suggestion_id = f"{runtime_policy_id}:unregularized-local-suggestion"
    regularized_policy_id = f"{runtime_policy_id}:proximal-direction-lq"

    kind, anchor_states, anchor_controls, gain, bias, bias_scale = _initial_policy_arrays(
        problem, prepared.initial_policy
    )
    initial_evaluation = _evaluate_affine_profile(
        problem,
        law_kind=kind,
        nominal_states=anchor_states,
        nominal_controls=anchor_controls,
        feedback_gain=gain,
        feedforward=bias,
        feedforward_scale=bias_scale,
        policy_id=runtime_policy_id,
    )
    incumbent_states = initial_evaluation.trajectory.states
    incumbent_controls = initial_evaluation.trajectory.controls
    incumbent_gain = gain
    zero_bias = jnp.zeros_like(incumbent_controls)
    dtype = jnp.result_type(
        incumbent_states,
        incumbent_controls,
        plan.scaling.state_scales,
        plan.scaling.control_scales,
        float,
    )
    cases = problem.case_shape
    initial_active = initial_evaluation.successful
    initial_status = jnp.where(
        initial_active,
        int(ILQFeedbackGameStatus.MAX_ITERATIONS),
        int(ILQFeedbackGameStatus.INITIAL_POLICY_EVALUATION_FAILED),
    ).astype(jnp.int32)
    histories = _history_arrays(problem, plan, dtype)
    carry = (
        incumbent_states,
        incumbent_controls,
        incumbent_gain,
        initial_active,
        initial_status,
        jnp.zeros(cases, dtype=dtype),
        jnp.zeros(cases, dtype=jnp.int32),
        jnp.zeros(cases, dtype=jnp.int32),
        *histories,
    )

    def iteration(iteration_index, loop):
        (
            current_states,
            current_controls,
            current_gain,
            active,
            status,
            regularization,
            iterations,
            accepted_iterations,
            history_valid,
            accepted_history,
            evaluation_valid_history,
            residual_valid_history,
            unregularized_local_valid_history,
            unregularized_local_status_history,
            direction_valid_history,
            direction_status_history,
            proximal_regularization_history,
            residual_merit_history,
            stationarity_rms_history,
            stationarity_infinity_history,
            dynamics_rms_history,
            dynamics_infinity_history,
            player_cost_history,
            accepted_alpha_history,
            accepted_state_step_rms_history,
            accepted_state_step_infinity_history,
            accepted_control_step_rms_history,
            accepted_control_step_infinity_history,
            line_search_evaluations_history,
            trial_history_valid,
            trial_reason_history,
            trial_alpha_history,
            trial_evaluation_valid_history,
            trial_residual_valid_history,
            trial_residual_merit_history,
            trial_armijo_bound_history,
            trial_state_step_rms_history,
            trial_state_step_infinity_history,
            trial_control_step_rms_history,
            trial_control_step_infinity_history,
            trial_player_cost_history,
        ) = loop

        current = _evaluate_affine_profile(
            problem,
            law_kind="local",
            nominal_states=current_states,
            nominal_controls=current_controls,
            feedback_gain=current_gain,
            feedforward=zero_bias,
            feedforward_scale=jnp.asarray(0.0, dtype=dtype),
            policy_id=runtime_policy_id,
        )
        residual = nominal_nash_residual(problem, current, plan.scaling)
        unregularized = _suggest(plan, problem, current, suggestion_id)
        full_unregularized_trial = _evaluate_local_direction(
            problem,
            current,
            unregularized.feedback_gain,
            unregularized.feedforward,
            jnp.asarray(1.0, dtype=dtype),
            runtime_policy_id,
        )
        unregularized_step = _scaled_step_metrics(
            current, full_unregularized_trial, plan.scaling
        )
        converged_here = active & _stationary(
            plan, current, residual, unregularized, unregularized_step
        )
        residual_failed = active & ~current.successful
        residual_failed = residual_failed | (
            active & current.successful & ~residual.successful
        )

        regularized = _regularized_direction(
            plan,
            problem,
            unregularized,
            regularization,
            regularized_policy_id,
        )
        use_regularized = regularization > 0.0
        direction_gain = _case_where(
            use_regularized,
            regularized.feedback_gain,
            unregularized.feedback_gain,
        )
        direction_bias = _case_where(
            use_regularized,
            regularized.feedforward,
            unregularized.feedforward,
        )
        direction_valid = jnp.where(
            use_regularized, regularized.valid, unregularized.successful
        )
        direction_status = jnp.where(
            use_regularized, regularized.status, unregularized.status
        ).astype(jnp.int32)
        search_active = (
            active
            & current.successful
            & residual.successful
            & ~converged_here
            & direction_valid
        )
        current_merit = _residual_merit(residual)

        search_initial = (
            jnp.zeros(cases, dtype=bool),
            current,
            residual,
            current_gain,
            jnp.zeros(cases, dtype=dtype),
            jnp.full(cases, jnp.nan, dtype=dtype),
            jnp.full(cases, jnp.nan, dtype=dtype),
            jnp.full(cases, jnp.nan, dtype=dtype),
            jnp.full(cases, jnp.nan, dtype=dtype),
            jnp.zeros(cases, dtype=jnp.int32),
            jnp.zeros(cases + (plan.maximum_line_search_steps,), dtype=bool),
            jnp.full(
                cases + (plan.maximum_line_search_steps,),
                int(ILQFeedbackGameTrialReason.NOT_EVALUATED),
                dtype=jnp.int32,
            ),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.zeros(cases + (plan.maximum_line_search_steps,), dtype=bool),
            jnp.zeros(cases + (plan.maximum_line_search_steps,), dtype=bool),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.full(cases + (plan.maximum_line_search_steps,), jnp.nan, dtype=dtype),
            jnp.full(
                cases + (plan.maximum_line_search_steps, problem.num_players),
                jnp.nan,
                dtype=dtype,
            ),
        )

        def search(search_index, search_loop):
            (
                found,
                best_evaluation,
                best_residual,
                best_gain,
                best_alpha,
                best_state_rms,
                best_state_infinity,
                best_control_rms,
                best_control_infinity,
                evaluations,
                row_valid,
                row_reason,
                row_alpha,
                row_evaluation_valid,
                row_residual_valid,
                row_merit,
                row_bound,
                row_state_rms,
                row_state_infinity,
                row_control_rms,
                row_control_infinity,
                row_costs,
            ) = search_loop
            alpha = jnp.asarray(
                plan.initial_alpha * plan.alpha_contraction**search_index,
                dtype=dtype,
            )
            trial = _evaluate_local_direction(
                problem,
                current,
                direction_gain,
                direction_bias,
                alpha,
                runtime_policy_id,
            )
            trial_residual = nominal_nash_residual(problem, trial, plan.scaling)
            trial_merit = _residual_merit(trial_residual)
            state_rms, state_inf, control_rms, control_inf, step_finite = (
                _scaled_step_metrics(current, trial, plan.scaling)
            )
            armijo_bound = (1.0 - plan.armijo * alpha) * current_merit
            attempted = search_active & ~found
            acceptable = (
                attempted
                & trial.successful
                & trial_residual.successful
                & step_finite
                & (state_inf <= plan.maximum_scaled_state_step)
                & (control_inf <= plan.maximum_scaled_control_step)
                & jnp.isfinite(trial_merit)
                & jnp.isfinite(armijo_bound)
                & (trial_merit <= armijo_bound)
            )
            reason = jnp.where(
                ~trial.successful,
                int(ILQFeedbackGameTrialReason.NONFINITE_OR_INVALID_ROLLOUT),
                jnp.where(
                    ~trial_residual.successful,
                    int(ILQFeedbackGameTrialReason.NONFINITE_OR_INVALID_RESIDUAL),
                    jnp.where(
                        ~step_finite,
                        int(ILQFeedbackGameTrialReason.NONFINITE_SCALED_STEP),
                        jnp.where(
                            state_inf > plan.maximum_scaled_state_step,
                            int(
                                ILQFeedbackGameTrialReason.SCALED_STATE_STEP_GUARD_EXCEEDED
                            ),
                            jnp.where(
                                control_inf > plan.maximum_scaled_control_step,
                                int(
                                    ILQFeedbackGameTrialReason.SCALED_CONTROL_STEP_GUARD_EXCEEDED
                                ),
                                jnp.where(
                                    acceptable,
                                    int(ILQFeedbackGameTrialReason.ACCEPTED),
                                    int(
                                        ILQFeedbackGameTrialReason.ORIGINAL_RESIDUAL_ARMIJO_FAILED
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            reason = jnp.where(
                attempted,
                reason,
                int(ILQFeedbackGameTrialReason.NOT_EVALUATED),
            ).astype(jnp.int32)
            next_evaluation = _select_case_tree(acceptable, trial, best_evaluation, cases)
            next_residual = _select_case_tree(
                acceptable, trial_residual, best_residual, cases
            )
            next_gain = _case_where(acceptable, direction_gain, best_gain)
            next_found = found | acceptable
            return (
                next_found,
                next_evaluation,
                next_residual,
                next_gain,
                jnp.where(acceptable, alpha, best_alpha),
                jnp.where(acceptable, state_rms, best_state_rms),
                jnp.where(acceptable, state_inf, best_state_infinity),
                jnp.where(acceptable, control_rms, best_control_rms),
                jnp.where(acceptable, control_inf, best_control_infinity),
                evaluations + attempted.astype(jnp.int32),
                row_valid.at[..., search_index].set(attempted),
                row_reason.at[..., search_index].set(reason),
                row_alpha.at[..., search_index].set(jnp.where(attempted, alpha, jnp.nan)),
                row_evaluation_valid.at[..., search_index].set(
                    attempted & trial.successful
                ),
                row_residual_valid.at[..., search_index].set(
                    attempted & trial_residual.successful
                ),
                row_merit.at[..., search_index].set(
                    jnp.where(attempted, trial_merit, jnp.nan)
                ),
                row_bound.at[..., search_index].set(
                    jnp.where(attempted, armijo_bound, jnp.nan)
                ),
                row_state_rms.at[..., search_index].set(
                    jnp.where(attempted, state_rms, jnp.nan)
                ),
                row_state_infinity.at[..., search_index].set(
                    jnp.where(attempted, state_inf, jnp.nan)
                ),
                row_control_rms.at[..., search_index].set(
                    jnp.where(attempted, control_rms, jnp.nan)
                ),
                row_control_infinity.at[..., search_index].set(
                    jnp.where(attempted, control_inf, jnp.nan)
                ),
                row_costs.at[..., search_index, :].set(
                    jnp.where(attempted[..., None], trial.total_costs, jnp.nan)
                ),
            )

        searched = jax.lax.fori_loop(
            0, plan.maximum_line_search_steps, search, search_initial
        )
        (
            accepted,
            accepted_evaluation,
            accepted_residual,
            accepted_gain,
            accepted_alpha,
            accepted_state_rms,
            accepted_state_infinity,
            accepted_control_rms,
            accepted_control_infinity,
            search_evaluations,
            row_valid,
            row_reason,
            row_alpha,
            row_evaluation_valid,
            row_residual_valid,
            row_merit,
            row_bound,
            row_state_rms,
            row_state_infinity,
            row_control_rms,
            row_control_infinity,
            row_costs,
        ) = searched
        del accepted_residual

        configured_regularization = plan.initial_proximal_regularization > 0.0
        below_maximum = regularization < plan.maximum_proximal_regularization
        retry = (
            active
            & current.successful
            & residual.successful
            & ~converged_here
            & ~accepted
            & configured_regularization
            & below_maximum
        )
        first_regularization = jnp.asarray(
            plan.initial_proximal_regularization, dtype=dtype
        )
        grown_regularization = jnp.minimum(
            jnp.where(
                regularization > 0.0,
                regularization * plan.proximal_regularization_growth,
                first_regularization,
            ),
            plan.maximum_proximal_regularization,
        )
        next_regularization = jnp.where(
            accepted,
            0.0,
            jnp.where(retry, grown_regularization, regularization),
        )
        local_failed = (
            active
            & current.successful
            & residual.successful
            & ~converged_here
            & ~accepted
            & ~direction_valid
            & ~retry
        )
        line_search_failed = (
            active
            & current.successful
            & residual.successful
            & ~converged_here
            & ~accepted
            & direction_valid
            & ~retry
        )
        next_status = jnp.where(
            converged_here,
            int(ILQFeedbackGameStatus.SUCCESS),
            jnp.where(
                residual_failed,
                int(ILQFeedbackGameStatus.NOMINAL_RESIDUAL_FAILED),
                jnp.where(
                    local_failed,
                    int(ILQFeedbackGameStatus.LOCAL_LQ_FAILED),
                    jnp.where(
                        line_search_failed,
                        int(ILQFeedbackGameStatus.LINE_SEARCH_FAILED),
                        status,
                    ),
                ),
            ),
        ).astype(jnp.int32)
        next_active = accepted | retry
        next_states = _case_where(
            accepted, accepted_evaluation.trajectory.states, current_states
        )
        next_controls = _case_where(
            accepted, accepted_evaluation.trajectory.controls, current_controls
        )
        next_gain = _case_where(accepted, accepted_gain, current_gain)

        return (
            next_states,
            next_controls,
            next_gain,
            next_active,
            next_status,
            next_regularization,
            iterations + active.astype(jnp.int32),
            accepted_iterations + accepted.astype(jnp.int32),
            history_valid.at[..., iteration_index].set(active),
            accepted_history.at[..., iteration_index].set(accepted),
            evaluation_valid_history.at[..., iteration_index].set(
                active & current.successful
            ),
            residual_valid_history.at[..., iteration_index].set(
                active & residual.successful
            ),
            unregularized_local_valid_history.at[..., iteration_index].set(
                active & unregularized.successful
            ),
            unregularized_local_status_history.at[..., iteration_index].set(
                jnp.where(active, unregularized.status, -1)
            ),
            direction_valid_history.at[..., iteration_index].set(
                active & direction_valid
            ),
            direction_status_history.at[..., iteration_index].set(
                jnp.where(active, direction_status, -1)
            ),
            proximal_regularization_history.at[..., iteration_index].set(
                jnp.where(active, regularization, jnp.nan)
            ),
            residual_merit_history.at[..., iteration_index].set(
                jnp.where(active, current_merit, jnp.nan)
            ),
            stationarity_rms_history.at[..., iteration_index].set(
                jnp.where(active, residual.stationarity_rms_norm, jnp.nan)
            ),
            stationarity_infinity_history.at[..., iteration_index].set(
                jnp.where(active, residual.stationarity_infinity_norm, jnp.nan)
            ),
            dynamics_rms_history.at[..., iteration_index].set(
                jnp.where(active, residual.dynamics_defect_rms_norm, jnp.nan)
            ),
            dynamics_infinity_history.at[..., iteration_index].set(
                jnp.where(active, residual.dynamics_defect_infinity_norm, jnp.nan)
            ),
            player_cost_history.at[..., iteration_index, :].set(
                jnp.where(active[..., None], current.total_costs, jnp.nan)
            ),
            accepted_alpha_history.at[..., iteration_index].set(
                jnp.where(accepted, accepted_alpha, jnp.nan)
            ),
            accepted_state_step_rms_history.at[..., iteration_index].set(
                jnp.where(accepted, accepted_state_rms, jnp.nan)
            ),
            accepted_state_step_infinity_history.at[..., iteration_index].set(
                jnp.where(accepted, accepted_state_infinity, jnp.nan)
            ),
            accepted_control_step_rms_history.at[..., iteration_index].set(
                jnp.where(accepted, accepted_control_rms, jnp.nan)
            ),
            accepted_control_step_infinity_history.at[..., iteration_index].set(
                jnp.where(accepted, accepted_control_infinity, jnp.nan)
            ),
            line_search_evaluations_history.at[..., iteration_index].set(
                search_evaluations
            ),
            trial_history_valid.at[..., iteration_index, :].set(row_valid),
            trial_reason_history.at[..., iteration_index, :].set(row_reason),
            trial_alpha_history.at[..., iteration_index, :].set(row_alpha),
            trial_evaluation_valid_history.at[..., iteration_index, :].set(
                row_evaluation_valid
            ),
            trial_residual_valid_history.at[..., iteration_index, :].set(
                row_residual_valid
            ),
            trial_residual_merit_history.at[..., iteration_index, :].set(row_merit),
            trial_armijo_bound_history.at[..., iteration_index, :].set(row_bound),
            trial_state_step_rms_history.at[..., iteration_index, :].set(row_state_rms),
            trial_state_step_infinity_history.at[..., iteration_index, :].set(
                row_state_infinity
            ),
            trial_control_step_rms_history.at[..., iteration_index, :].set(
                row_control_rms
            ),
            trial_control_step_infinity_history.at[..., iteration_index, :].set(
                row_control_infinity
            ),
            trial_player_cost_history.at[..., iteration_index, :, :].set(row_costs),
        )

    final_loop = jax.lax.fori_loop(0, plan.maximum_iterations, iteration, carry)
    (
        final_states,
        final_controls,
        final_gain,
        _,
        loop_status,
        _,
        iterations,
        accepted_iterations,
        *final_histories,
    ) = final_loop
    final_evaluation = _evaluate_affine_profile(
        problem,
        law_kind="local",
        nominal_states=final_states,
        nominal_controls=final_controls,
        feedback_gain=final_gain,
        feedforward=zero_bias,
        feedforward_scale=jnp.asarray(0.0, dtype=dtype),
        policy_id=runtime_policy_id,
    )
    final_residual = nominal_nash_residual(problem, final_evaluation, plan.scaling)
    final_suggestion = _suggest(plan, problem, final_evaluation, suggestion_id)
    final_unregularized_trial = _evaluate_local_direction(
        problem,
        final_evaluation,
        final_suggestion.feedback_gain,
        final_suggestion.feedforward,
        jnp.asarray(1.0, dtype=dtype),
        runtime_policy_id,
    )
    final_step = _scaled_step_metrics(
        final_evaluation, final_unregularized_trial, plan.scaling
    )
    stationary = _stationary(
        plan, final_evaluation, final_residual, final_suggestion, final_step
    )
    tolerance_without_model = (
        final_evaluation.successful
        & final_residual.successful
        & final_step[4]
        & (final_residual.stationarity_rms_norm <= plan.residual_tolerance)
        & (final_residual.stationarity_infinity_norm <= plan.residual_tolerance)
        & (final_residual.dynamics_defect_rms_norm <= plan.dynamics_tolerance)
        & (final_residual.dynamics_defect_infinity_norm <= plan.dynamics_tolerance)
        & (final_step[0] <= plan.step_tolerance)
        & (final_step[1] <= plan.step_tolerance)
        & (final_step[2] <= plan.step_tolerance)
        & (final_step[3] <= plan.step_tolerance)
    )
    final_status = jnp.where(
        stationary,
        int(ILQFeedbackGameStatus.SUCCESS),
        jnp.where(
            ~final_evaluation.successful,
            int(ILQFeedbackGameStatus.INITIAL_POLICY_EVALUATION_FAILED),
            jnp.where(
                ~final_residual.successful,
                int(ILQFeedbackGameStatus.NOMINAL_RESIDUAL_FAILED),
                jnp.where(
                    tolerance_without_model & ~final_suggestion.successful,
                    int(ILQFeedbackGameStatus.FINAL_UNREGULARIZED_LOCAL_LQ_FAILED),
                    loop_status,
                ),
            ),
        ),
    ).astype(jnp.int32)
    converged = final_status == int(ILQFeedbackGameStatus.SUCCESS)
    final_policy_gain = _case_where(converged, final_suggestion.feedback_gain, final_gain)
    system_input_layout = problem.dynamics.system.input_layout
    assert system_input_layout is not None
    final_policy = LocalAffineGamePolicy(
        final_evaluation.trajectory.states,
        final_evaluation.trajectory.controls,
        final_policy_gain,
        jnp.zeros_like(final_evaluation.trajectory.controls),
        feedforward_scale=jnp.asarray(0.0, dtype=dtype),
        time_grid=problem.time_grid,
        input_layout=system_input_layout,
        partition=problem.partition,
        case_shape=problem.case_shape,
        policy_id=runtime_policy_id,
    )
    diagnostics = LocalNominalNashDiagnostics(
        final_status,
        converged,
        converged,
        iterations,
        accepted_iterations,
        *final_histories,
        _residual_merit(final_residual),
        final_residual.stationarity_rms_norm,
        final_residual.stationarity_infinity_norm,
        final_residual.dynamics_defect_rms_norm,
        final_residual.dynamics_defect_infinity_norm,
        final_step[0],
        final_step[1],
        final_step[2],
        final_step[3],
        final_suggestion.successful,
        final_suggestion.status,
        converged,
        _CERTIFICATE,
        _METHOD,
        _ACCEPTANCE_METHOD,
        _DIFFERENTIATION_METHOD,
        False,
        False,
        False,
        False,
    )
    return LocalNominalNashResult(
        plan,
        final_policy,
        final_evaluation,
        final_residual,
        final_suggestion,
        diagnostics,
        runtime_result_id,
        _METHOD,
        _CERTIFICATE,
        plan.plan_id,
        prepared.prepared_id,
        prepared.materialization_id,
    )


def solve_ilq_feedback_game(
    problem: DeterministicFeedbackGameProblem,
    scaling: ILQGameScaling | LocalAffineGamePolicy | AffineFeedbackPolicy,
    initial_policy: LocalAffineGamePolicy | AffineFeedbackPolicy | ILQGameScaling,
    /,
    *,
    policy_id: str | None = None,
    result_id: str | None = None,
    **plan_options,
) -> LocalNominalNashResult:
    """Plan, prepare, and solve one residual-globalized nonlinear game."""

    if isinstance(scaling, ILQGameScaling):
        game_scaling = scaling
        game_policy = initial_policy
    elif isinstance(initial_policy, ILQGameScaling):
        game_scaling = initial_policy
        game_policy = scaling
    else:
        raise TypeError("direct solve requires one ILQGameScaling and one policy.")
    plan = plan_ilq_feedback_game(problem, game_scaling, **plan_options)
    prepared = prepare_ilq_feedback_game(plan, problem, game_policy)
    return solve_prepared_ilq_feedback_game(
        prepared, policy_id=policy_id, result_id=result_id
    )


__all__ = [
    "ILQFeedbackGamePlan",
    "ILQFeedbackGameStatus",
    "ILQFeedbackGameTrialReason",
    "LocalNominalNashDiagnostics",
    "LocalNominalNashResult",
    "PreparedILQFeedbackGame",
    "plan_ilq_feedback_game",
    "prepare_ilq_feedback_game",
    "refresh_ilq_feedback_game",
    "solve_ilq_feedback_game",
    "solve_prepared_ilq_feedback_game",
]
