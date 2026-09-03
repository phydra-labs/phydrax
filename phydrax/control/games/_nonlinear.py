#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic nonlinear full-state simultaneous game evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import prod
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import AbstractInputPolicy, DiscreteStepContext, TimeGrid
from .._dynamics import DiscreteControlDynamics
from .._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_SUCCESS,
    ControlTrajectory,
)
from ._layout import PlayerControlPartition


_STAGE_COST_SEMANTICS = "unweighted-discrete-stage-sum"
_EVALUATION_METHOD = "deterministic-full-state-simultaneous-game-evaluation"
_RESIDUAL_METHOD = "nominal-owned-row-discrete-adjoint"
_CERTIFICATE = "LOCAL_NOMINAL_NASH_STATIONARY"


class GameStageCost(Protocol):
    """One player's scalar discrete-stage cost.

    The callback signature is ``cost(context, state, joint_control, args)``.
    ``context`` is the exact :class:`DiscreteStepContext` used for the
    transition. The returned value is a scalar stage cost, not a rate; it is
    summed once and is never multiplied by the interval duration.
    """

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        joint_control: Array,
        args: Any,
        /,
    ) -> ArrayLike: ...


class GameTerminalCost(Protocol):
    """One player's scalar terminal cost.

    The callback signature is ``cost(terminal_time, terminal_state, args)``.
    """

    def __call__(
        self,
        terminal_time: Array,
        terminal_state: Array,
        args: Any,
        /,
    ) -> ArrayLike: ...


class GamePolicyEvaluationStatus(IntEnum):
    """Stable case-local status codes for deterministic policy evaluation."""

    SUCCESS = 0
    NONFINITE_INITIAL_STATE = 1
    NONFINITE_POLICY_CONTROL = 2
    NONFINITE_DYNAMICS_STATE = 3
    NONFINITE_STAGE_COST = 4
    NONFINITE_TERMINAL_COST = 5
    NONFINITE_DERIVATIVE = 6


class DeterministicFeedbackGameProblem(StrictModule):
    """Finite-horizon deterministic simultaneous full-state game.

    ``initial_state`` has shape ``case_shape + (state_size,)``. The ordered
    ``stage_costs`` and ``terminal_costs`` sequences contain exactly one
    callback per ``partition.player_ids`` entry. Stage callbacks have signature
    ``(DiscreteStepContext, state, joint_control, args) -> scalar``; terminal
    callbacks have signature ``(terminal_time, terminal_state, args) -> scalar``.

    Every player's objective is the unweighted discrete sum
    ``sum_k stage_costs[player](...) + terminal_costs[player](...)``. In
    particular, no implicit interval-duration multiplier is applied. The joint
    policy is evaluated once at each state before the shared transition, so all
    players act simultaneously.
    """

    dynamics: DiscreteControlDynamics
    time_grid: TimeGrid
    initial_state: Array
    partition: PlayerControlPartition
    stage_costs: tuple[GameStageCost, ...] = eqx.field(static=True)
    terminal_costs: tuple[GameTerminalCost, ...] = eqx.field(static=True)
    args: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    stage_cost_semantics: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: DiscreteControlDynamics,
        time_grid: TimeGrid,
        initial_state: ArrayLike,
        partition: PlayerControlPartition,
        /,
        *,
        stage_costs: Sequence[GameStageCost],
        terminal_costs: Sequence[GameTerminalCost],
        args: Any = None,
        problem_id: str,
    ):
        if not isinstance(dynamics, DiscreteControlDynamics):
            raise TypeError(
                "DeterministicFeedbackGameProblem dynamics must be "
                "DiscreteControlDynamics."
            )
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        if len(dynamics.state_shape) != 1:
            raise ValueError("Deterministic games require a rank-one state_shape.")
        if len(dynamics.control_shape) != 1:
            raise ValueError("Deterministic games require a rank-one control_shape.")
        if dynamics.control_shape != (partition.joint_control_size,):
            raise ValueError(
                "partition joint control size must match dynamics control_shape."
            )
        state = jnp.asarray(initial_state)
        if state.ndim < 1 or tuple(state.shape[-1:]) != dynamics.state_shape:
            raise ValueError(
                "initial_state must have shape case_shape + "
                f"{dynamics.state_shape}; got {state.shape}."
            )
        if jnp.issubdtype(state.dtype, jnp.complexfloating):
            raise TypeError("Deterministic game states must be real-valued.")
        if not jnp.issubdtype(state.dtype, jnp.inexact):
            state = state.astype(float)
        cases = tuple(int(size) for size in state.shape[:-1])
        if any(size <= 0 for size in cases):
            raise ValueError("Deterministic game case dimensions must be positive.")

        stage = tuple(stage_costs)
        terminal = tuple(terminal_costs)
        if len(stage) != partition.num_players:
            raise ValueError("stage_costs must provide exactly one callback per player.")
        if len(terminal) != partition.num_players:
            raise ValueError(
                "terminal_costs must provide exactly one callback per player."
            )
        if any(not callable(callback) for callback in stage):
            raise TypeError("Every stage_costs entry must be callable.")
        if any(not callable(callback) for callback in terminal):
            raise TypeError("Every terminal_costs entry must be callable.")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string.")

        self.dynamics = dynamics
        self.time_grid = time_grid
        self.initial_state = state
        self.partition = partition
        self.stage_costs = stage
        self.terminal_costs = terminal
        self.args = args
        self.case_shape = cases
        self.state_shape = dynamics.state_shape
        self.control_shape = dynamics.control_shape
        self.state_size = dynamics.state_shape[0]
        self.control_size = dynamics.control_shape[0]
        self.num_players = partition.num_players
        self.problem_id = problem_id
        self.stage_cost_semantics = _STAGE_COST_SEMANTICS


class ILQGameScaling(StrictModule):
    """Fixed positive physical scales for dimensionless game residuals.

    ``state_scales`` has shape ``(state_size,)``, ``control_scales`` has shape
    ``(joint_control_size,)``, and ``cost_scales`` has shape ``(num_players,)``.
    ``state_shape``, ``control_shape``, and ``num_players`` record that static
    topology; ``scaling_id`` is the supplied identity or a deterministic
    content identity.

    An owned stationarity row is nondimensionalized by multiplying by its
    control scale and dividing by its owning player's cost scale. A dynamics
    defect is divided by its state scale.
    """

    state_scales: Array
    control_scales: Array
    cost_scales: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_scales: ArrayLike,
        control_scales: ArrayLike,
        cost_scales: ArrayLike,
        /,
        *,
        scaling_id: str | None = None,
    ):
        state = _positive_real_vector(state_scales, "state_scales")
        control = _positive_real_vector(control_scales, "control_scales")
        cost = _positive_real_vector(cost_scales, "cost_scales")
        dtype = jnp.result_type(state, control, cost, float)
        state = state.astype(dtype)
        control = control.astype(dtype)
        cost = cost.astype(dtype)
        if scaling_id is None:
            scaling_id = "ilq-game-scaling:" + canonical_fingerprint(
                {
                    "state_scales": np.asarray(state).tolist(),
                    "control_scales": np.asarray(control).tolist(),
                    "cost_scales": np.asarray(cost).tolist(),
                }
            )
        if not isinstance(scaling_id, str) or not scaling_id:
            raise ValueError("scaling_id must be a non-empty string or None.")
        self.state_scales = state
        self.control_scales = control
        self.cost_scales = cost
        self.state_shape = tuple(state.shape)
        self.control_shape = tuple(control.shape)
        self.num_players = int(cost.shape[0])
        self.scaling_id = scaling_id


class GamePolicyEvaluation(StrictModule):
    """Physical rollout, ordered player costs, and causal failure evidence.

    ``partition`` fixes player order and control ownership. ``trajectory`` is
    the physical :class:`ControlTrajectory`, with ``T + 1`` state samples and
    ``T`` applied joint controls. ``stage_costs`` and ``stage_cost_valid`` have
    shape ``case_shape + (num_players, T)``. ``terminal_costs``,
    ``terminal_cost_valid``, and ``total_costs`` have shape
    ``case_shape + (num_players,)``; invalid cases receive a nonfinite total.

    ``control_valid`` and ``transition_valid`` have shape
    ``case_shape + (T,)`` and retain stage-local numerical evidence.
    ``first_failed_step`` is ``-1`` on success, ``0`` for an invalid initial
    state, a stage index in ``[0, T - 1]`` for an in-horizon failure, and ``T``
    for a terminal-cost failure. ``first_failed_player`` is the first ordered
    player with an invalid cost, or ``-1`` for non-cost failures. ``valid`` and
    ``status`` have ``case_shape`` and use :class:`GamePolicyEvaluationStatus`.
    The remaining static fields record topology, cost semantics, policy,
    evaluation, and method provenance.
    """

    partition: PlayerControlPartition
    trajectory: ControlTrajectory
    stage_costs: Array
    terminal_costs: Array
    total_costs: Array
    control_valid: Array
    transition_valid: Array
    stage_cost_valid: Array
    terminal_cost_valid: Array
    first_failed_step: Array
    first_failed_player: Array
    valid: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    stage_cost_semantics: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Return the case-local successful-evaluation mask."""

        return self.valid & (self.status == int(GamePolicyEvaluationStatus.SUCCESS))


class NominalNashResidual(StrictModule):
    """Owned-row first-order residual along one evaluated nominal trajectory.

    ``scaling`` records the fixed physical scales. ``player_costates`` has
    shape ``case_shape + (num_players, T + 1, state_size)``.
    ``raw_owned_stationarity`` and ``dimensionless_owned_stationarity`` have
    shape ``case_shape + (T, joint_control_size)``; every row is selected from
    the complete joint-control derivative of its owning player's full
    objective. ``dynamics_defect`` and ``dimensionless_dynamics_defect`` have
    shape ``case_shape + (T, state_size)``.

    The stationarity and defect RMS/infinity fields reduce their corresponding
    dimensionless arrays. ``rms_norm`` pools every dimensionless stationarity
    and defect entry; ``infinity_norm`` is their joint maximum magnitude. All
    norms, ``valid``, and ``status`` have ``case_shape``. ``certificate`` is
    the exact local first-order contract label; it does not override
    ``valid``. ``residual_id`` and ``method_id`` record provenance.
    """

    scaling: ILQGameScaling
    player_costates: Array
    raw_owned_stationarity: Array
    dimensionless_owned_stationarity: Array
    dynamics_defect: Array
    dimensionless_dynamics_defect: Array
    stationarity_rms_norm: Array
    stationarity_infinity_norm: Array
    dynamics_defect_rms_norm: Array
    dynamics_defect_infinity_norm: Array
    rms_norm: Array
    infinity_norm: Array
    valid: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    certificate: str = eqx.field(static=True)
    residual_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Return the case-local usable-residual mask."""

        return self.valid & (self.status == int(GamePolicyEvaluationStatus.SUCCESS))


def _positive_real_vector(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1 or int(array.shape[0]) < 1:
        raise ValueError(f"{name} must be a nonempty rank-one array.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    host = np.asarray(array)
    if not np.all(np.isfinite(host) & (host > 0.0)):
        raise ValueError(f"{name} must be finite and strictly positive.")
    return array


def _real_scalar(value: ArrayLike, owner: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{owner} must return one scalar.")
    if jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must return a real scalar.")
    if not jnp.issubdtype(scalar.dtype, jnp.inexact):
        scalar = scalar.astype(float)
    return scalar


def _stage_cost_vector(
    problem: DeterministicFeedbackGameProblem,
    context: DiscreteStepContext,
    state: Array,
    control: Array,
    /,
) -> Array:
    return jnp.stack(
        tuple(
            _real_scalar(
                callback(context, state, control, problem.args),
                f"stage_costs[{player}]",
            )
            for player, callback in enumerate(problem.stage_costs)
        )
    )


def _terminal_cost_vector(
    problem: DeterministicFeedbackGameProblem,
    state: Array,
    /,
) -> Array:
    terminal_time = problem.time_grid.times[-1]
    return jnp.stack(
        tuple(
            _real_scalar(
                callback(terminal_time, state, problem.args),
                f"terminal_costs[{player}]",
            )
            for player, callback in enumerate(problem.terminal_costs)
        )
    )


def _first_false(values: Array, /) -> Array:
    return jnp.argmax(~values, axis=-1).astype(jnp.int32)


def _validate_policy(
    problem: DeterministicFeedbackGameProblem,
    policy: AbstractInputPolicy,
    /,
) -> None:
    if not isinstance(problem, DeterministicFeedbackGameProblem):
        raise TypeError("problem must be a DeterministicFeedbackGameProblem.")
    if not isinstance(policy, AbstractInputPolicy):
        raise TypeError("policy must implement AbstractInputPolicy.")
    system_layout = problem.dynamics.system.input_layout
    assert system_layout is not None
    if policy.input_layout.layout_id != system_layout.layout_id:
        raise ValueError(
            "policy input_layout must exactly match the dynamics input_layout."
        )


def evaluate_game_policy(
    problem: DeterministicFeedbackGameProblem,
    policy: AbstractInputPolicy,
    /,
) -> GamePolicyEvaluation:
    """Evaluate one joint full-state policy and all ordered player objectives.

    The policy callback is ``policy.evaluate_step(context, state, problem.args)``
    and must return the complete joint control vector. Numerical failures are
    recorded independently for every declared case. If several failures first
    occur at the same stage, policy output takes precedence over transition
    output, which takes precedence over stage cost in the stable status code.
    """

    _validate_policy(problem, policy)
    cases = problem.case_shape
    count = prod(cases) if cases else 1
    horizon = problem.time_grid.num_steps
    players = problem.num_players
    state_size = problem.state_size
    control_size = problem.control_size
    initial = problem.initial_state.reshape((count, state_size))
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
        control = jax.vmap(
            lambda case_state: policy.evaluate_step(context, case_state, problem.args)
        )(state)
        if control.shape != (count, control_size):
            raise ValueError(
                "policy.evaluate_step must return the complete joint control shape."
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
        if candidate.shape != (count, state_size):
            raise ValueError("dynamics returned the wrong case/state shape.")
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
            cost_failure,
            _first_false(stage_finite),
            failed_player,
        ).astype(jnp.int32)
        next_trajectory_active = trajectory_active & control_finite & transition_finite
        return (
            candidate,
            next_trajectory_active,
            status,
            failed_step,
            failed_player,
        ), (
            candidate,
            control,
            next_trajectory_active,
            control_finite,
            transition_finite,
            stage,
            stage_finite,
        )

    (_, _, status, failed_step, failed_player), scan_output = jax.lax.scan(
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
        controls_time_major,
        trajectory_valid_tail,
        control_valid_time_major,
        transition_valid_time_major,
        stage_time_major,
        stage_valid_time_major,
    ) = scan_output
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
        terminal_failure,
        _first_false(terminal_valid_flat),
        failed_player,
    ).astype(jnp.int32)
    valid_flat = status == int(GamePolicyEvaluationStatus.SUCCESS)

    states_flat = jnp.concatenate((initial[None, ...], state_tail), axis=0)
    trajectory_valid_flat = jnp.concatenate(
        (initial_valid[None, ...], trajectory_valid_tail), axis=0
    )
    states = jnp.moveaxis(states_flat, 0, 1).reshape(cases + (horizon + 1, state_size))
    controls = jnp.moveaxis(controls_time_major, 0, 1).reshape(
        cases + (horizon, control_size)
    )
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
        control_id=policy.policy_id,
        backend_id="backend:jax:lax-scan",
        method_id=_EVALUATION_METHOD,
        discretization_id=problem.time_grid.time_id,
        approximation_id="approximation:none",
    )

    stage = jnp.swapaxes(jnp.moveaxis(stage_time_major, 0, 1), -1, -2).reshape(
        cases + (players, horizon)
    )
    stage_valid = jnp.swapaxes(
        jnp.moveaxis(stage_valid_time_major, 0, 1), -1, -2
    ).reshape(cases + (players, horizon))
    terminal_costs = terminal.reshape(cases + (players,))
    terminal_valid = terminal_valid_flat.reshape(cases + (players,))
    total_unchecked = jnp.sum(stage, axis=-1) + terminal_costs
    valid = valid_flat.reshape(cases)
    total_costs = jnp.where(valid[..., None], total_unchecked, jnp.nan)
    evaluation_id = f"game-policy-evaluation:{problem.problem_id}:{policy.policy_id}"
    return GamePolicyEvaluation(
        partition=problem.partition,
        trajectory=trajectory,
        stage_costs=stage,
        terminal_costs=terminal_costs,
        total_costs=total_costs,
        control_valid=jnp.moveaxis(control_valid_time_major, 0, 1).reshape(
            cases + (horizon,)
        ),
        transition_valid=jnp.moveaxis(transition_valid_time_major, 0, 1).reshape(
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
        stage_cost_semantics=_STAGE_COST_SEMANTICS,
        policy_id=policy.policy_id,
        evaluation_id=evaluation_id,
        method_id=_EVALUATION_METHOD,
    )


def _validate_residual_inputs(
    problem: DeterministicFeedbackGameProblem,
    evaluation: GamePolicyEvaluation,
    scaling: ILQGameScaling,
    /,
) -> None:
    if not isinstance(problem, DeterministicFeedbackGameProblem):
        raise TypeError("problem must be a DeterministicFeedbackGameProblem.")
    if not isinstance(evaluation, GamePolicyEvaluation):
        raise TypeError("evaluation must be a GamePolicyEvaluation.")
    if not isinstance(scaling, ILQGameScaling):
        raise TypeError("scaling must be an ILQGameScaling.")
    if evaluation.trajectory.problem_id != problem.problem_id:
        raise ValueError("evaluation problem identity does not match problem.")
    if evaluation.partition.partition_id != problem.partition.partition_id:
        raise ValueError("evaluation player partition does not match problem.")
    if evaluation.case_shape != problem.case_shape:
        raise ValueError("evaluation case_shape does not match problem.")
    if evaluation.trajectory.discretization_id != problem.time_grid.time_id:
        raise ValueError("evaluation time-grid identity does not match problem.")
    if evaluation.trajectory.dynamics_id != problem.dynamics.dynamics_id:
        raise ValueError("evaluation dynamics identity does not match problem.")
    if evaluation.trajectory.case_shape != problem.case_shape:
        raise ValueError("evaluation trajectory case_shape does not match problem.")
    if evaluation.trajectory.state_shape != problem.state_shape:
        raise ValueError("evaluation trajectory state_shape does not match problem.")
    if evaluation.trajectory.control_shape != problem.control_shape:
        raise ValueError("evaluation trajectory control_shape does not match problem.")
    if evaluation.num_players != problem.num_players:
        raise ValueError("evaluation player count does not match problem.")
    if evaluation.stage_cost_semantics != _STAGE_COST_SEMANTICS:
        raise ValueError("evaluation stage-cost semantics do not match problem.")
    horizon = problem.time_grid.num_steps
    expected_player_stages = problem.case_shape + (problem.num_players, horizon)
    expected_player_totals = problem.case_shape + (problem.num_players,)
    expected_stages = problem.case_shape + (horizon,)
    expected_cases = problem.case_shape
    shaped_values = (
        ("stage_costs", evaluation.stage_costs, expected_player_stages),
        ("stage_cost_valid", evaluation.stage_cost_valid, expected_player_stages),
        ("terminal_costs", evaluation.terminal_costs, expected_player_totals),
        ("terminal_cost_valid", evaluation.terminal_cost_valid, expected_player_totals),
        ("total_costs", evaluation.total_costs, expected_player_totals),
        ("control_valid", evaluation.control_valid, expected_stages),
        ("transition_valid", evaluation.transition_valid, expected_stages),
        ("first_failed_step", evaluation.first_failed_step, expected_cases),
        ("first_failed_player", evaluation.first_failed_player, expected_cases),
        ("valid", evaluation.valid, expected_cases),
        ("status", evaluation.status, expected_cases),
    )
    for name, value, expected in shaped_values:
        if tuple(value.shape) != expected:
            raise ValueError(
                f"evaluation {name} must have shape {expected}; got {value.shape}."
            )
    if scaling.state_shape != problem.state_shape:
        raise ValueError("state_scales must exactly match problem.state_shape.")
    if scaling.control_shape != problem.control_shape:
        raise ValueError("control_scales must exactly match problem.control_shape.")
    if scaling.num_players != problem.num_players:
        raise ValueError("cost_scales must have exactly one entry per player.")


def nominal_nash_residual(
    problem: DeterministicFeedbackGameProblem,
    evaluation: GamePolicyEvaluation,
    scaling: ILQGameScaling,
    /,
) -> NominalNashResidual:
    """Compute exact discrete adjoints and owned stationarity at a nominal path.

    JAX differentiates every player's complete scalar stage and terminal costs
    with respect to the complete state and joint-control vectors. Only after
    forming the full joint-control stationarity for each player are rows
    selected according to ``problem.partition.control_owner``. The calculation
    is local to the supplied physical trajectory and makes no strategy-solution
    claim beyond the explicit ``LOCAL_NOMINAL_NASH_STATIONARY`` contract label.
    """

    _validate_residual_inputs(problem, evaluation, scaling)
    cases = problem.case_shape
    count = prod(cases) if cases else 1
    horizon = problem.time_grid.num_steps
    players = problem.num_players
    state_size = problem.state_size
    control_size = problem.control_size
    states = evaluation.trajectory.states.reshape((count, horizon + 1, state_size))
    controls = evaluation.trajectory.controls.reshape((count, horizon, control_size))
    step_indices = jnp.arange(horizon, dtype=jnp.int32)

    def derivatives_at_step(step_index, state, control, next_state):
        context = DiscreteStepContext(
            problem.time_grid.times[step_index],
            problem.time_grid.times[step_index + 1],
            step_index,
        )

        def transition(current_state, joint_control):
            return problem.dynamics.system.evaluate(
                context,
                current_state,
                problem.args,
                inputs=joint_control,
            )

        def player_costs(current_state, joint_control):
            return _stage_cost_vector(
                problem,
                context,
                current_state,
                joint_control,
            )

        dynamics_state, dynamics_control = jax.jacrev(transition, argnums=(0, 1))(
            state, control
        )
        cost_state, cost_control = jax.jacrev(player_costs, argnums=(0, 1))(
            state, control
        )
        defect = next_state - transition(state, control)
        return dynamics_state, dynamics_control, cost_state, cost_control, defect

    def derivatives_for_case(case_states, case_controls):
        return jax.vmap(derivatives_at_step)(
            step_indices,
            case_states[:-1],
            case_controls,
            case_states[1:],
        )

    dynamics_state, dynamics_control, cost_state, cost_control, defect = jax.vmap(
        derivatives_for_case
    )(states, controls)
    terminal_costate = jax.vmap(
        lambda state: jax.jacrev(
            lambda terminal_state: _terminal_cost_vector(problem, terminal_state)
        )(state)
    )(states[:, -1])

    def adjoint_step(next_costate, inputs):
        (
            dynamics_state_step,
            dynamics_control_step,
            cost_state_step,
            cost_control_step,
        ) = inputs
        full_stationarity = cost_control_step + ein.contract(
            "cab,cpa->cpb",
            dynamics_control_step,
            next_costate,
        )
        current_costate = cost_state_step + ein.contract(
            "cab,cpa->cpb",
            dynamics_state_step,
            next_costate,
        )
        return current_costate, (current_costate, full_stationarity)

    _, (costates_reverse, stationarity_reverse) = jax.lax.scan(
        adjoint_step,
        terminal_costate,
        (
            jnp.moveaxis(dynamics_state, 1, 0)[::-1],
            jnp.moveaxis(dynamics_control, 1, 0)[::-1],
            jnp.moveaxis(cost_state, 1, 0)[::-1],
            jnp.moveaxis(cost_control, 1, 0)[::-1],
        ),
    )
    costates_stage = jnp.moveaxis(costates_reverse[::-1], 0, 2)
    player_costates_flat = jnp.concatenate(
        (costates_stage, terminal_costate[:, :, None, :]), axis=2
    )
    player_stationarity = jnp.moveaxis(stationarity_reverse[::-1], 0, 1)
    ownership = jax.nn.one_hot(
        jnp.asarray(problem.partition.control_owner, dtype=jnp.int32),
        players,
        dtype=player_stationarity.dtype,
    )
    raw_owned_flat = ein.contract(
        "ctpm,mp->ctm",
        player_stationarity,
        ownership,
    )

    owner = jnp.asarray(problem.partition.control_owner, dtype=jnp.int32)
    owner_cost_scales = jnp.take(scaling.cost_scales, owner)
    dimensionless_owned_flat = (
        raw_owned_flat
        * scaling.control_scales.astype(raw_owned_flat.dtype)[None, None, :]
        / owner_cost_scales.astype(raw_owned_flat.dtype)[None, None, :]
    )
    dimensionless_defect_flat = (
        defect / scaling.state_scales.astype(defect.dtype)[None, None, :]
    )

    stationarity_square = jnp.square(dimensionless_owned_flat)
    defect_square = jnp.square(dimensionless_defect_flat)
    stationarity_rms = jnp.sqrt(jnp.mean(stationarity_square, axis=(-2, -1)))
    stationarity_infinity = jnp.max(jnp.abs(dimensionless_owned_flat), axis=(-2, -1))
    defect_rms = jnp.sqrt(jnp.mean(defect_square, axis=(-2, -1)))
    defect_infinity = jnp.max(jnp.abs(dimensionless_defect_flat), axis=(-2, -1))
    total_entries = horizon * (control_size + state_size)
    rms = jnp.sqrt(
        (
            jnp.sum(stationarity_square, axis=(-2, -1))
            + jnp.sum(defect_square, axis=(-2, -1))
        )
        / total_entries
    )
    infinity = jnp.maximum(stationarity_infinity, defect_infinity)
    residual_finite = (
        jnp.all(jnp.isfinite(player_costates_flat), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(raw_owned_flat), axis=(-2, -1))
        & jnp.all(jnp.isfinite(dimensionless_owned_flat), axis=(-2, -1))
        & jnp.all(jnp.isfinite(defect), axis=(-2, -1))
        & jnp.all(jnp.isfinite(dimensionless_defect_flat), axis=(-2, -1))
    )
    evaluation_valid_flat = evaluation.valid.reshape((count,))
    valid_flat = evaluation_valid_flat & residual_finite
    status_flat = jnp.where(
        evaluation_valid_flat,
        jnp.where(
            residual_finite,
            int(GamePolicyEvaluationStatus.SUCCESS),
            int(GamePolicyEvaluationStatus.NONFINITE_DERIVATIVE),
        ),
        evaluation.status.reshape((count,)),
    ).astype(jnp.int32)

    nan = jnp.asarray(jnp.nan, dtype=rms.dtype)
    stationarity_rms = jnp.where(valid_flat, stationarity_rms, nan)
    stationarity_infinity = jnp.where(valid_flat, stationarity_infinity, nan)
    defect_rms = jnp.where(valid_flat, defect_rms, nan)
    defect_infinity = jnp.where(valid_flat, defect_infinity, nan)
    rms = jnp.where(valid_flat, rms, nan)
    infinity = jnp.where(valid_flat, infinity, nan)
    residual_id = f"nominal-nash-residual:{evaluation.evaluation_id}:{scaling.scaling_id}"
    return NominalNashResidual(
        scaling=scaling,
        player_costates=player_costates_flat.reshape(
            cases + (players, horizon + 1, state_size)
        ),
        raw_owned_stationarity=raw_owned_flat.reshape(cases + (horizon, control_size)),
        dimensionless_owned_stationarity=dimensionless_owned_flat.reshape(
            cases + (horizon, control_size)
        ),
        dynamics_defect=defect.reshape(cases + (horizon, state_size)),
        dimensionless_dynamics_defect=dimensionless_defect_flat.reshape(
            cases + (horizon, state_size)
        ),
        stationarity_rms_norm=stationarity_rms.reshape(cases),
        stationarity_infinity_norm=stationarity_infinity.reshape(cases),
        dynamics_defect_rms_norm=defect_rms.reshape(cases),
        dynamics_defect_infinity_norm=defect_infinity.reshape(cases),
        rms_norm=rms.reshape(cases),
        infinity_norm=infinity.reshape(cases),
        valid=valid_flat.reshape(cases),
        status=status_flat.reshape(cases),
        case_shape=cases,
        num_players=players,
        certificate=_CERTIFICATE,
        residual_id=residual_id,
        method_id=_RESIDUAL_METHOD,
    )


__all__ = [
    "DeterministicFeedbackGameProblem",
    "GamePolicyEvaluation",
    "GamePolicyEvaluationStatus",
    "GameStageCost",
    "GameTerminalCost",
    "ILQGameScaling",
    "NominalNashResidual",
    "evaluate_game_policy",
    "nominal_nash_residual",
]
