#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-step local quadratic-game policy suggestions."""

from __future__ import annotations

from enum import IntEnum
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import (
    AbstractInputPolicy,
    DiscreteStepContext,
    InputLayout,
    TimeGrid,
)
from ._layout import PlayerControlPartition
from ._linear_quadratic import (
    finite_horizon_lq_feedback_nash,
    FiniteHorizonLQFeedbackNashDiagnostics,
    FiniteHorizonLQFeedbackNashResult,
    LQFeedbackNashStatus,
)
from ._nonlinear import (
    _stage_cost_vector,
    _terminal_cost_vector,
    _validate_residual_inputs,
    DeterministicFeedbackGameProblem,
    evaluate_game_policy,
    GamePolicyEvaluation,
    ILQGameScaling,
)


_MODEL_METHOD = "exact-cost-hessian-first-order-discrete-dynamics"
_SUGGESTION_METHOD = "one-step-local-quadratic-game-suggestion"
_SUGGESTION_SCOPE = "LOCAL_QUADRATIC_SUGGESTION"


class LocalAffineGameSuggestionStatus(IntEnum):
    """Stable outcomes for a one-step local quadratic-game suggestion.

    Codes zero through nine intentionally equal the corresponding exact LQ
    kernel codes. A usable nominal therefore propagates the nested LQ outcome
    without translation.
    """

    SUCCESS = int(LQFeedbackNashStatus.SUCCESS)
    NONFINITE_INPUT = int(LQFeedbackNashStatus.NONFINITE_INPUT)
    NONSYMMETRIC_COST = int(LQFeedbackNashStatus.NONSYMMETRIC_COST)
    OWN_CURVATURE_NOT_POSITIVE_DEFINITE = int(
        LQFeedbackNashStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE
    )
    COUPLED_SYSTEM_RANK_DEFICIENT = int(
        LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT
    )
    CONDITION_LIMIT_REACHED = int(LQFeedbackNashStatus.CONDITION_LIMIT_REACHED)
    LINEAR_SOLVE_FAILED = int(LQFeedbackNashStatus.LINEAR_SOLVE_FAILED)
    NONFINITE_OUTPUT = int(LQFeedbackNashStatus.NONFINITE_OUTPUT)
    RESIDUAL_TOO_LARGE = int(LQFeedbackNashStatus.RESIDUAL_TOO_LARGE)
    DEPENDENCY_FAILED = int(LQFeedbackNashStatus.DEPENDENCY_FAILED)
    NOMINAL_EVALUATION_FAILED = 10


class _LocalQuadraticGame(StrictModule):
    """Exact local derivatives expressed in physical deviation coordinates.

    At stage ``k``, ``dx = x - nominal_states[k]`` and
    ``du = u - nominal_controls[k]``. The cost convention is

    ``constant + q @ dx + r @ du + dx @ N @ du
    + 0.5 * dx @ Q @ dx + 0.5 * du @ R @ du``.

    ``dynamics_defects`` uses the residual convention
    ``nominal_states[k + 1] - f(nominal_states[k], nominal_controls[k])``.
    Consequently the affine bias in the deviation dynamics is its negative.
    No Hessian is symmetrized or otherwise modified.
    """

    partition: PlayerControlPartition
    time_grid: TimeGrid
    nominal_states: Array
    nominal_controls: Array
    nominal_dynamics: Array
    dynamics_defects: Array
    dynamics_bias: Array
    A: Array
    B: Array
    q: Array
    r: Array
    Q: Array
    R: Array
    N: Array
    stage_constants: Array
    terminal_q: Array
    terminal_Q: Array
    terminal_constants: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    nominal_policy_id: str = eqx.field(static=True)
    scaling_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class LocalAffineGamePolicy(AbstractInputPolicy):
    """Physical local policy around one nominal trajectory.

    The stored ``feedback_gain`` and ``feedforward`` act on physical
    deviations. At stage ``k`` the returned joint control is

    ``nominal_controls[k] + feedback_gain[k] @ (x - nominal_states[k])
    + feedforward_scale * feedforward[k]``.

    Thus ``feedforward_scale=0`` applies only the local feedback correction and
    ``feedforward_scale=1`` applies the complete one-step suggestion. The policy
    implements :class:`AbstractInputPolicy` and can be passed directly to
    :func:`evaluate_game_policy` for a scalar-case problem. For a declared case
    shape, :meth:`evaluate` and :meth:`evaluate_step` operate on the complete
    case-shaped physical state.
    """

    nominal_states: Array
    nominal_controls: Array
    feedback_gain: Array
    feedforward: Array
    feedforward_scale: Array
    time_grid: TimeGrid
    input_layout: InputLayout
    partition: PlayerControlPartition
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        nominal_states: ArrayLike,
        nominal_controls: ArrayLike,
        feedback_gain: ArrayLike,
        feedforward: ArrayLike,
        /,
        *,
        feedforward_scale: ArrayLike,
        time_grid: TimeGrid,
        input_layout: InputLayout,
        partition: PlayerControlPartition,
        case_shape: tuple[int, ...] = (),
        policy_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout.")
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string.")
        cases = tuple(int(size) for size in case_shape)
        if any(size <= 0 for size in cases):
            raise ValueError("Local policy case dimensions must be positive.")
        if len(input_layout.shape) != 1:
            raise ValueError("Local game policies require a rank-one input layout.")

        states = jnp.asarray(nominal_states)
        controls = jnp.asarray(nominal_controls)
        gain = jnp.asarray(feedback_gain)
        bias = jnp.asarray(feedforward)
        scale = jnp.asarray(feedforward_scale)
        horizon = time_grid.num_steps
        control_size = input_layout.shape[0]
        if states.ndim < 2 or tuple(states.shape[:-2]) != cases:
            raise ValueError(
                "nominal_states must have shape case_shape + (T + 1, state_size)."
            )
        state_size = int(states.shape[-1])
        expected_states = cases + (horizon + 1, state_size)
        expected_controls = cases + (horizon, control_size)
        expected_gain = cases + (horizon, control_size, state_size)
        if tuple(states.shape) != expected_states:
            raise ValueError(
                f"nominal_states must have shape {expected_states}; got {states.shape}."
            )
        if tuple(controls.shape) != expected_controls:
            raise ValueError(
                "nominal_controls must have shape "
                f"{expected_controls}; got {controls.shape}."
            )
        if tuple(gain.shape) != expected_gain:
            raise ValueError(
                f"feedback_gain must have shape {expected_gain}; got {gain.shape}."
            )
        if tuple(bias.shape) != expected_controls:
            raise ValueError(
                f"feedforward must have shape {expected_controls}; got {bias.shape}."
            )
        if scale.shape != ():
            raise ValueError("feedforward_scale must be a real scalar.")
        values = (states, controls, gain, bias, scale)
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in values):
            raise TypeError("Local game policies require real-valued arrays.")
        dtype = jnp.result_type(*values, float)
        scale = scale.astype(dtype)
        scale = eqx.error_if(
            scale,
            ~jnp.isfinite(scale),
            "feedforward_scale must be finite.",
        )

        self.nominal_states = states.astype(dtype)
        self.nominal_controls = controls.astype(dtype)
        self.feedback_gain = gain.astype(dtype)
        self.feedforward = bias.astype(dtype)
        self.feedforward_scale = scale
        self.time_grid = time_grid
        self.input_layout = input_layout
        self.partition = partition
        self.case_shape = cases
        self.state_shape = (state_size,)
        self.control_shape = (control_size,)
        self.policy_id = policy_id

    @property
    def absolute_feedforward(self) -> Array:
        """Return ``ubar - K xbar + alpha k`` for every physical stage."""

        return (
            self.nominal_controls
            - ein.contract(
                "...tmn,...tn->...tm",
                self.feedback_gain,
                self.nominal_states[..., :-1, :],
            )
            + self.feedforward_scale * self.feedforward
        )

    def with_feedforward_scale(
        self,
        feedforward_scale: ArrayLike,
        /,
        *,
        policy_id: str,
    ) -> LocalAffineGamePolicy:
        """Return the same local law with an explicitly identified scale."""

        return LocalAffineGamePolicy(
            self.nominal_states,
            self.nominal_controls,
            self.feedback_gain,
            self.feedforward,
            feedforward_scale=feedforward_scale,
            time_grid=self.time_grid,
            input_layout=self.input_layout,
            partition=self.partition,
            case_shape=self.case_shape,
            policy_id=policy_id,
        )

    def _evaluate_index(self, index: Array, state: ArrayLike, /) -> Array:
        state_array = jnp.asarray(state)
        expected_state = self.case_shape + self.state_shape
        if tuple(state_array.shape) != expected_state:
            raise ValueError(
                f"Physical state must have shape {expected_state}; got {state_array.shape}."
            )
        if jnp.issubdtype(state_array.dtype, jnp.complexfloating):
            raise TypeError("Physical state must be real-valued.")
        if not jnp.issubdtype(state_array.dtype, jnp.inexact):
            state_array = state_array.astype(float)
        axis = len(self.case_shape)
        nominal_state = jnp.take(self.nominal_states, index, axis=axis)
        nominal_control = jnp.take(self.nominal_controls, index, axis=axis)
        gain = jnp.take(self.feedback_gain, index, axis=axis)
        feedforward = jnp.take(self.feedforward, index, axis=axis)
        deviation = state_array - nominal_state
        return (
            nominal_control
            + ein.contract("...mn,...n->...m", gain, deviation)
            + self.feedforward_scale * feedforward
        )

    def evaluate(
        self,
        coordinate: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate the physical law at a scalar coordinate."""

        del args
        time = jnp.asarray(coordinate, dtype=self.time_grid.times.dtype)
        if time.shape != ():
            raise ValueError("LocalAffineGamePolicy coordinate must be scalar.")
        time = eqx.error_if(
            time,
            ~jnp.isfinite(time)
            | (time < self.time_grid.times[0])
            | (time > self.time_grid.times[-1]),
            "Local game policy coordinate lies outside its time grid.",
        )
        index = jnp.searchsorted(self.time_grid.times, time, side="right") - 1
        index = jnp.clip(index, 0, self.time_grid.num_steps - 1)
        return self._evaluate_index(index, state)

    def evaluate_step(
        self,
        context: DiscreteStepContext,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate the physical law at the context's exact stage index."""

        del args
        if not isinstance(context, DiscreteStepContext):
            raise TypeError("evaluate_step requires DiscreteStepContext.")
        index = eqx.error_if(
            context.step_index,
            (context.step_index < 0) | (context.step_index >= self.time_grid.num_steps),
            "Local game policy step lies outside its time grid.",
        )
        return self._evaluate_index(index, state)

    def rollout(
        self,
        problem: DeterministicFeedbackGameProblem,
        /,
    ) -> GamePolicyEvaluation:
        """Apply this physical policy to a scalar-case nonlinear game."""

        if self.case_shape:
            raise ValueError(
                "rollout requires a scalar-case policy; evaluate case-shaped "
                "policies on complete case-shaped states."
            )
        return evaluate_game_policy(problem, self)


class LocalAffineGameSuggestion(StrictModule):
    """A one-step local quadratic model and its affine physical suggestion.

    This result is local diagnostic evidence only. ``lq_result`` preserves the
    complete exact LQ kernel evidence, while ``status`` propagates that status
    for every successfully evaluated nominal case. It is not an iterative
    solver or a certificate for the nonlinear game.
    """

    scaling: ILQGameScaling
    model: _LocalQuadraticGame
    policy: LocalAffineGamePolicy
    lq_result: FiniteHorizonLQFeedbackNashResult
    dimensionless_dynamics_defects: Array
    dynamics_defect_rms_norm: Array
    dynamics_defect_infinity_norm: Array
    derivative_finite: Array
    model_finite: Array
    evaluation_valid: Array
    evaluation_status: Array
    valid: Array
    status: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    suggestion_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Return cases with a usable local quadratic suggestion."""

        return self.valid & (self.status == int(LocalAffineGameSuggestionStatus.SUCCESS))

    @property
    def feedback_gain(self) -> Array:
        """Return the physical deviation-feedback gain."""

        return self.policy.feedback_gain

    @property
    def feedforward(self) -> Array:
        """Return the unscaled physical deviation feedforward direction."""

        return self.policy.feedforward

    @property
    def lq_diagnostics(self) -> FiniteHorizonLQFeedbackNashDiagnostics:
        """Return the unchanged nested LQ diagnostic evidence."""

        return self.lq_result.diagnostics


def _all_finite_case(value: Array, payload_rank: int, /) -> Array:
    axes = tuple(range(value.ndim - payload_rank, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes)


def suggest_local_affine_game_policy(
    problem: DeterministicFeedbackGameProblem,
    evaluation: GamePolicyEvaluation,
    scaling: ILQGameScaling,
    /,
    *,
    symmetry_tolerance: float = 1e-10,
    curvature_tolerance: float = 1e-10,
    rank_relative_tolerance: float | None = None,
    rank_absolute_tolerance: float | None = None,
    maximum_condition: float | None = None,
    suggestion_id: str,
) -> LocalAffineGameSuggestion:
    """Build and solve one exact-cost, first-order-dynamics local game.

    All derivatives are taken by JAX at the supplied physical nominal states and
    controls. Costs retain every player's complete joint-control derivatives.
    The exact LQ kernel solves the resulting model in physical deviation
    coordinates without regularization, clipping, fallback, or pseudoinverse.
    Its ``du = K dx + k`` strategy is exposed as the physical policy
    ``u = ubar + K (x - xbar) + alpha k``.
    """

    _validate_residual_inputs(problem, evaluation, scaling)
    if not isinstance(suggestion_id, str) or not suggestion_id:
        raise ValueError("suggestion_id must be a non-empty string.")

    cases = problem.case_shape
    count = prod(cases) if cases else 1
    horizon = problem.time_grid.num_steps
    players = problem.num_players
    state_size = problem.state_size
    control_size = problem.control_size
    states_flat = evaluation.trajectory.states.reshape((count, horizon + 1, state_size))
    controls_flat = evaluation.trajectory.controls.reshape((count, horizon, control_size))
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

        nominal_dynamics = transition(state, control)
        A, B = jax.jacrev(transition, argnums=(0, 1))(state, control)
        q, r = jax.jacrev(player_costs, argnums=(0, 1))(state, control)
        hessian = jax.jacfwd(
            jax.jacrev(player_costs, argnums=(0, 1)),
            argnums=(0, 1),
        )(state, control)
        Q = hessian[0][0]
        N = hessian[0][1]
        R = hessian[1][1]
        constants = player_costs(state, control)
        defect = next_state - nominal_dynamics
        return nominal_dynamics, defect, A, B, q, r, Q, R, N, constants

    def derivatives_for_case(case_states, case_controls):
        return jax.vmap(derivatives_at_step)(
            step_indices,
            case_states[:-1],
            case_controls,
            case_states[1:],
        )

    (
        nominal_dynamics_flat,
        defects_flat,
        A_flat,
        B_flat,
        q_time_flat,
        r_time_flat,
        Q_time_flat,
        R_time_flat,
        N_time_flat,
        stage_constants_time_flat,
    ) = jax.vmap(derivatives_for_case)(states_flat, controls_flat)

    def terminal_derivatives(state):
        def player_costs(terminal_state):
            return _terminal_cost_vector(problem, terminal_state)

        terminal_q = jax.jacrev(player_costs)(state)
        terminal_Q = jax.jacfwd(jax.jacrev(player_costs))(state)
        terminal_constants = player_costs(state)
        return terminal_q, terminal_Q, terminal_constants

    terminal_q_flat, terminal_Q_flat, terminal_constants_flat = jax.vmap(
        terminal_derivatives
    )(states_flat[:, -1])

    def player_stage_axis(value: Array) -> Array:
        return jnp.swapaxes(value, 1, 2)

    A = A_flat.reshape(cases + (horizon, state_size, state_size))
    B = B_flat.reshape(cases + (horizon, state_size, control_size))
    q = player_stage_axis(q_time_flat).reshape(cases + (players, horizon, state_size))
    r = player_stage_axis(r_time_flat).reshape(cases + (players, horizon, control_size))
    Q = player_stage_axis(Q_time_flat).reshape(
        cases + (players, horizon, state_size, state_size)
    )
    R = player_stage_axis(R_time_flat).reshape(
        cases + (players, horizon, control_size, control_size)
    )
    N = player_stage_axis(N_time_flat).reshape(
        cases + (players, horizon, state_size, control_size)
    )
    stage_constants = player_stage_axis(stage_constants_time_flat).reshape(
        cases + (players, horizon)
    )
    terminal_q = terminal_q_flat.reshape(cases + (players, state_size))
    terminal_Q = terminal_Q_flat.reshape(cases + (players, state_size, state_size))
    terminal_constants = terminal_constants_flat.reshape(cases + (players,))
    nominal_states = states_flat.reshape(cases + (horizon + 1, state_size))
    nominal_controls = controls_flat.reshape(cases + (horizon, control_size))
    nominal_dynamics = nominal_dynamics_flat.reshape(cases + (horizon, state_size))
    dynamics_defects = defects_flat.reshape(cases + (horizon, state_size))
    dynamics_bias = -dynamics_defects

    derivative_finite_flat = (
        _all_finite_case(A_flat, 3)
        & _all_finite_case(B_flat, 3)
        & _all_finite_case(q_time_flat, 3)
        & _all_finite_case(r_time_flat, 3)
        & _all_finite_case(Q_time_flat, 4)
        & _all_finite_case(R_time_flat, 4)
        & _all_finite_case(N_time_flat, 4)
        & _all_finite_case(terminal_q_flat, 2)
        & _all_finite_case(terminal_Q_flat, 3)
    )
    model_finite_flat = (
        derivative_finite_flat
        & _all_finite_case(states_flat, 2)
        & _all_finite_case(controls_flat, 2)
        & _all_finite_case(nominal_dynamics_flat, 2)
        & _all_finite_case(defects_flat, 2)
        & _all_finite_case(stage_constants_time_flat, 2)
        & _all_finite_case(terminal_constants_flat, 1)
    )
    derivative_finite = derivative_finite_flat.reshape(cases)
    model_finite = model_finite_flat.reshape(cases)

    model_id = f"local-quadratic-game:{evaluation.evaluation_id}:{scaling.scaling_id}"
    model = _LocalQuadraticGame(
        partition=problem.partition,
        time_grid=problem.time_grid,
        nominal_states=nominal_states,
        nominal_controls=nominal_controls,
        nominal_dynamics=nominal_dynamics,
        dynamics_defects=dynamics_defects,
        dynamics_bias=dynamics_bias,
        A=A,
        B=B,
        q=q,
        r=r,
        Q=Q,
        R=R,
        N=N,
        stage_constants=stage_constants,
        terminal_q=terminal_q,
        terminal_Q=terminal_Q,
        terminal_constants=terminal_constants,
        case_shape=cases,
        state_size=state_size,
        control_size=control_size,
        num_players=players,
        problem_id=problem.problem_id,
        evaluation_id=evaluation.evaluation_id,
        nominal_policy_id=evaluation.policy_id,
        scaling_id=scaling.scaling_id,
        model_id=model_id,
        method_id=_MODEL_METHOD,
    )

    lq_result = finite_horizon_lq_feedback_nash(
        model.A,
        model.B,
        model.Q,
        model.R,
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
        policy_id=f"{suggestion_id}:deviation-lq",
        symmetry_tolerance=symmetry_tolerance,
        curvature_tolerance=curvature_tolerance,
        rank_relative_tolerance=rank_relative_tolerance,
        rank_absolute_tolerance=rank_absolute_tolerance,
        maximum_condition=maximum_condition,
    )
    system_input_layout = problem.dynamics.system.input_layout
    assert system_input_layout is not None
    policy = LocalAffineGamePolicy(
        model.nominal_states,
        model.nominal_controls,
        lq_result.feedback_gain,
        lq_result.feedforward,
        feedforward_scale=jnp.asarray(1.0, dtype=lq_result.feedback_gain.dtype),
        time_grid=problem.time_grid,
        input_layout=system_input_layout,
        partition=problem.partition,
        case_shape=cases,
        policy_id=f"{suggestion_id}:physical-local-affine-policy",
    )

    dimensionless_defects = dynamics_defects / scaling.state_scales.astype(
        dynamics_defects.dtype
    )
    defect_rms = jnp.sqrt(jnp.mean(jnp.square(dimensionless_defects), axis=(-2, -1)))
    defect_infinity = jnp.max(jnp.abs(dimensionless_defects), axis=(-2, -1))
    evaluation_valid = evaluation.successful
    evidence_valid = evaluation_valid & model_finite
    nan = jnp.asarray(jnp.nan, dtype=defect_rms.dtype)
    defect_rms = jnp.where(evidence_valid, defect_rms, nan)
    defect_infinity = jnp.where(evidence_valid, defect_infinity, nan)
    valid = evidence_valid & lq_result.valid
    status = jnp.where(
        evaluation_valid,
        lq_result.status,
        int(LocalAffineGameSuggestionStatus.NOMINAL_EVALUATION_FAILED),
    ).astype(jnp.int32)

    return LocalAffineGameSuggestion(
        scaling=scaling,
        model=model,
        policy=policy,
        lq_result=lq_result,
        dimensionless_dynamics_defects=dimensionless_defects,
        dynamics_defect_rms_norm=defect_rms,
        dynamics_defect_infinity_norm=defect_infinity,
        derivative_finite=derivative_finite,
        model_finite=model_finite,
        evaluation_valid=evaluation.valid,
        evaluation_status=evaluation.status,
        valid=valid,
        status=status,
        case_shape=cases,
        suggestion_id=suggestion_id,
        method_id=_SUGGESTION_METHOD,
        scope=_SUGGESTION_SCOPE,
    )


__all__ = [
    "LocalAffineGamePolicy",
    "LocalAffineGameSuggestion",
    "LocalAffineGameSuggestionStatus",
    "suggest_local_affine_game_policy",
]
