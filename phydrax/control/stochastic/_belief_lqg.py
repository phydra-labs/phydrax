#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-horizon centralized LQG on an explicit Gaussian belief."""

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import DiscreteStepContext, TimeGrid
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    RHSLayout,
    solve,
)
from .._lqr import (
    finite_horizon_lqr,
    FiniteHorizonLQRResult,
    QuadraticValueFunction,
)
from .._riccati import _error_if, _require_shape
from ..games._information import GaussianBelief
from ._lqg import _all_finite, _covariance_evidence


_RESULT_LABEL = "CENTRALIZED_GAUSSIAN_BELIEF_LQG"
_METHOD_ID = "finite-horizon-centralized-gaussian-belief-lqg-v1"
_PRE_ACTION_TIMING = "pre-action"


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _covariance_tolerance(value: float, /) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("covariance_tolerance must be finite and non-negative.")
    return tolerance


def _zeros(shape: tuple[int, ...], dtype, /) -> Array:
    return jnp.zeros(shape, dtype=dtype)


def _to_time_major(value: Array, payload_rank: int, /) -> Array:
    return jnp.moveaxis(value, -(payload_rank + 1), 0)


def _from_time_major(value: Array, payload_rank: int, /) -> Array:
    return jnp.moveaxis(value, 0, -(payload_rank + 1))


def _matrix_evidence(matrix: Array, /) -> tuple[Array, Array, Array]:
    symmetry = jnp.max(
        jnp.abs(matrix - jnp.swapaxes(matrix, -1, -2)),
        axis=(-2, -1),
    )
    minimum = jnp.min(jnp.linalg.eigvalsh(matrix), axis=-1)
    finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
    return symmetry, minimum, finite


class CentralizedLQGStatus(IntEnum):
    """Stable validity codes for centralized finite-horizon belief LQG."""

    SUCCESS = 0
    DETERMINISTIC_SOLVE_FAILED = 1
    INNOVATION_COVARIANCE_NOT_POSITIVE_DEFINITE = 2
    INNOVATION_SOLVE_FAILED = 3
    COVARIANCE_NOT_SYMMETRIC = 4
    COVARIANCE_NOT_POSITIVE_SEMIDEFINITE = 5
    NONFINITE_OUTPUT = 6


class CentralizedLQGProblem(StrictModule):
    """A centralized, observation-before-action linear Gaussian control problem.

    The stage convention is

    ``y[k] = C[k] x[k] + d[k] + v[k]``, then ``u[k]``, then
    ``x[k+1] = A[k] x[k] + B[k] u[k] + c[k] + G[k] w[k]``.

    Driving noises are zero mean, mutually independent across time, and satisfy
    ``Cov(w[k]) = Omega[k]`` and ``Cov(v[k]) = V[k]``. Cross-correlated noise
    and action-dependent observation models are deliberately not represented by
    this exact classical interface. Every coefficient has explicit case and
    time axes; the one supplied Gaussian prior is shared across cases.
    """

    dynamics_matrices: Array
    control_matrices: Array
    dynamics_bias: Array
    process_noise_factors: Array
    process_noise_covariances: Array
    process_covariances: Array
    observation_matrices: Array
    observation_bias: Array
    measurement_covariances: Array
    state_costs: Array
    control_costs: Array
    terminal_state_cost: Array
    state_control_cross: Array
    state_linear: Array
    control_linear: Array
    stage_constants: Array
    terminal_linear: Array
    terminal_constant: Array
    initial_belief: GaussianBelief
    time_grid: TimeGrid
    process_covariance_symmetry_residuals: Array
    process_covariance_minimum_eigenvalues: Array
    measurement_covariance_symmetry_residuals: Array
    measurement_covariance_minimum_eigenvalues: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    process_noise_size: int = eqx.field(static=True)
    observation_size: int = eqx.field(static=True)
    observation_timing: str = eqx.field(static=True)
    information_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    covariance_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        dynamics_matrices: ArrayLike,
        control_matrices: ArrayLike,
        process_noise_factors: ArrayLike,
        process_noise_covariances: ArrayLike,
        observation_matrices: ArrayLike,
        measurement_covariances: ArrayLike,
        initial_belief: GaussianBelief,
        state_costs: ArrayLike,
        control_costs: ArrayLike,
        terminal_state_cost: ArrayLike,
        /,
        *,
        dynamics_bias: ArrayLike | None = None,
        observation_bias: ArrayLike | None = None,
        state_control_cross: ArrayLike | None = None,
        state_linear: ArrayLike | None = None,
        control_linear: ArrayLike | None = None,
        stage_constants: ArrayLike | None = None,
        terminal_linear: ArrayLike | None = None,
        terminal_constant: ArrayLike = 0.0,
        time_grid: TimeGrid | None = None,
        observation_timing: str = _PRE_ACTION_TIMING,
        information_id: str = "centralized-gaussian-observation",
        problem_id: str = "centralized-belief-lqg",
        covariance_tolerance: float = 1.0e-7,
        observation_control_matrices: ArrayLike | None = None,
        process_measurement_cross_covariances: ArrayLike | None = None,
    ):
        tolerance = _covariance_tolerance(covariance_tolerance)
        resolved_information_id = _identifier(information_id, "information_id")
        resolved_problem_id = _identifier(problem_id, "problem_id")
        if observation_timing != _PRE_ACTION_TIMING:
            raise ValueError(
                "CentralizedLQGProblem requires pre-action observation timing."
            )
        if observation_control_matrices is not None:
            raise ValueError(
                "CentralizedLQGProblem requires action-independent sensing; "
                "observation_control_matrices are unsupported."
            )
        if process_measurement_cross_covariances is not None:
            raise ValueError(
                "CentralizedLQGProblem requires independent process and measurement "
                "noise; cross covariances are unsupported."
            )
        if not isinstance(initial_belief, GaussianBelief):
            raise TypeError("initial_belief must be a GaussianBelief.")

        dynamics = jnp.asarray(dynamics_matrices)
        if dynamics.ndim < 3 or dynamics.shape[-2] != dynamics.shape[-1]:
            raise ValueError(
                "dynamics_matrices must have shape case_shape + (horizon, n, n)."
            )
        cases = tuple(dynamics.shape[:-3])
        horizon = int(dynamics.shape[-3])
        state_size = int(dynamics.shape[-1])
        if horizon < 1:
            raise ValueError("CentralizedLQGProblem requires at least one stage.")
        dynamics = _require_shape(
            dynamics,
            cases + (horizon, state_size, state_size),
            "dynamics_matrices",
        )

        controls = jnp.asarray(control_matrices)
        if (
            controls.ndim != len(cases) + 3
            or tuple(controls.shape[: len(cases)]) != cases
            or controls.shape[-3] != horizon
            or controls.shape[-2] != state_size
        ):
            raise ValueError(
                "control_matrices must have shape case_shape + (horizon, n, m); "
                f"got {controls.shape}."
            )
        control_size = int(controls.shape[-1])
        if control_size < 1:
            raise ValueError("control_matrices must have a positive control size.")
        controls = _require_shape(
            controls,
            cases + (horizon, state_size, control_size),
            "control_matrices",
        )

        factors = jnp.asarray(process_noise_factors)
        if (
            factors.ndim != len(cases) + 3
            or tuple(factors.shape[: len(cases)]) != cases
            or factors.shape[-3] != horizon
            or factors.shape[-2] != state_size
        ):
            raise ValueError(
                "process_noise_factors must have shape case_shape + "
                f"(horizon, n, process_noise_size); got {factors.shape}."
            )
        process_noise_size = int(factors.shape[-1])
        if process_noise_size < 1:
            raise ValueError("process_noise_factors must have a positive noise size.")
        factors = _require_shape(
            factors,
            cases + (horizon, state_size, process_noise_size),
            "process_noise_factors",
        )
        driving_covariances = _require_shape(
            process_noise_covariances,
            cases + (horizon, process_noise_size, process_noise_size),
            "process_noise_covariances",
        )
        (
            driving_covariances,
            process_symmetry,
            process_minimum,
        ) = _covariance_evidence(
            driving_covariances,
            "process_noise_covariances",
            tolerance,
        )
        process_covariances = (
            factors @ driving_covariances @ jnp.swapaxes(factors, -1, -2)
        )
        process_covariances = _error_if(
            process_covariances,
            jnp.any(~jnp.isfinite(process_covariances)),
            "The implied process_covariances must contain only finite values.",
        )

        observations = jnp.asarray(observation_matrices)
        if (
            observations.ndim != len(cases) + 3
            or tuple(observations.shape[: len(cases)]) != cases
            or observations.shape[-3] != horizon
            or observations.shape[-1] != state_size
        ):
            raise ValueError(
                "observation_matrices must have shape case_shape + "
                f"(horizon, observation_size, n); got {observations.shape}."
            )
        observation_size = int(observations.shape[-2])
        if observation_size < 1:
            raise ValueError(
                "observation_matrices must have a positive observation size."
            )
        observations = _require_shape(
            observations,
            cases + (horizon, observation_size, state_size),
            "observation_matrices",
        )
        measurement = _require_shape(
            measurement_covariances,
            cases + (horizon, observation_size, observation_size),
            "measurement_covariances",
        )
        measurement, measurement_symmetry, measurement_minimum = _covariance_evidence(
            measurement,
            "measurement_covariances",
            tolerance,
        )

        dtype_inputs = (
            dynamics,
            controls,
            factors,
            driving_covariances,
            observations,
            measurement,
            initial_belief.mean,
            initial_belief.covariance,
            state_costs,
            control_costs,
            terminal_state_cost,
            terminal_constant,
        ) + tuple(
            value
            for value in (
                dynamics_bias,
                observation_bias,
                state_control_cross,
                state_linear,
                control_linear,
                stage_constants,
                terminal_linear,
            )
            if value is not None
        )
        dtype = jnp.result_type(*dtype_inputs, float)
        dynamics_bias_ = (
            _zeros(cases + (horizon, state_size), dtype)
            if dynamics_bias is None
            else _require_shape(
                dynamics_bias,
                cases + (horizon, state_size),
                "dynamics_bias",
            )
        )
        observation_bias_ = (
            _zeros(cases + (horizon, observation_size), dtype)
            if observation_bias is None
            else _require_shape(
                observation_bias,
                cases + (horizon, observation_size),
                "observation_bias",
            )
        )
        state_costs_ = _require_shape(
            state_costs,
            cases + (horizon, state_size, state_size),
            "state_costs",
        )
        control_costs_ = _require_shape(
            control_costs,
            cases + (horizon, control_size, control_size),
            "control_costs",
        )
        terminal_state_cost_ = _require_shape(
            terminal_state_cost,
            cases + (state_size, state_size),
            "terminal_state_cost",
        )
        cross = (
            _zeros(cases + (horizon, state_size, control_size), dtype)
            if state_control_cross is None
            else _require_shape(
                state_control_cross,
                cases + (horizon, state_size, control_size),
                "state_control_cross",
            )
        )
        state_linear_ = (
            _zeros(cases + (horizon, state_size), dtype)
            if state_linear is None
            else _require_shape(
                state_linear,
                cases + (horizon, state_size),
                "state_linear",
            )
        )
        control_linear_ = (
            _zeros(cases + (horizon, control_size), dtype)
            if control_linear is None
            else _require_shape(
                control_linear,
                cases + (horizon, control_size),
                "control_linear",
            )
        )
        stage_constants_ = (
            _zeros(cases + (horizon,), dtype)
            if stage_constants is None
            else _require_shape(
                stage_constants,
                cases + (horizon,),
                "stage_constants",
            )
        )
        terminal_linear_ = (
            _zeros(cases + (state_size,), dtype)
            if terminal_linear is None
            else _require_shape(
                terminal_linear,
                cases + (state_size,),
                "terminal_linear",
            )
        )
        terminal_constant_ = jnp.asarray(terminal_constant)
        if terminal_constant_.shape == () and cases:
            terminal_constant_ = jnp.broadcast_to(terminal_constant_, cases)
        terminal_constant_ = _require_shape(
            terminal_constant_, cases, "terminal_constant"
        )
        if initial_belief.dimension != state_size:
            raise ValueError(
                "initial_belief dimension must match the dynamics state size."
            )
        if time_grid is None:
            resolved_time_grid = TimeGrid(
                jnp.arange(horizon + 1, dtype=dtype),
                time_id=f"{resolved_problem_id}:time",
            )
        else:
            if not isinstance(time_grid, TimeGrid):
                raise TypeError("time_grid must be a TimeGrid or None.")
            if time_grid.num_steps != horizon:
                raise ValueError(
                    f"time_grid must contain {horizon + 1} times for this horizon."
                )
            resolved_time_grid = time_grid

        self.dynamics_matrices = dynamics.astype(dtype)
        self.control_matrices = controls.astype(dtype)
        self.dynamics_bias = dynamics_bias_.astype(dtype)
        self.process_noise_factors = factors.astype(dtype)
        self.process_noise_covariances = driving_covariances.astype(dtype)
        self.process_covariances = process_covariances.astype(dtype)
        self.observation_matrices = observations.astype(dtype)
        self.observation_bias = observation_bias_.astype(dtype)
        self.measurement_covariances = measurement.astype(dtype)
        self.state_costs = state_costs_.astype(dtype)
        self.control_costs = control_costs_.astype(dtype)
        self.terminal_state_cost = terminal_state_cost_.astype(dtype)
        self.state_control_cross = cross.astype(dtype)
        self.state_linear = state_linear_.astype(dtype)
        self.control_linear = control_linear_.astype(dtype)
        self.stage_constants = stage_constants_.astype(dtype)
        self.terminal_linear = terminal_linear_.astype(dtype)
        self.terminal_constant = terminal_constant_.astype(dtype)
        self.initial_belief = initial_belief
        self.time_grid = resolved_time_grid
        self.process_covariance_symmetry_residuals = process_symmetry
        self.process_covariance_minimum_eigenvalues = process_minimum
        self.measurement_covariance_symmetry_residuals = measurement_symmetry
        self.measurement_covariance_minimum_eigenvalues = measurement_minimum
        self.case_shape = cases
        self.horizon = horizon
        self.state_size = state_size
        self.control_size = control_size
        self.process_noise_size = process_noise_size
        self.observation_size = observation_size
        self.observation_timing = observation_timing
        self.information_id = resolved_information_id
        self.problem_id = resolved_problem_id
        self.covariance_tolerance = tolerance


class BeliefFeedbackPolicy(StrictModule):
    """Affine certainty-equivalent control exposed only on a Gaussian belief."""

    feedback_gain: Array
    feedforward: Array
    time_grid: TimeGrid
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    observation_timing: str = eqx.field(static=True)
    information_id: str = eqx.field(static=True)
    belief_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        feedback_gain: ArrayLike,
        feedforward: ArrayLike,
        /,
        *,
        time_grid: TimeGrid,
        case_shape: tuple[int, ...],
        state_size: int,
        observation_timing: str,
        information_id: str,
        belief_id: str,
        policy_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if observation_timing != _PRE_ACTION_TIMING:
            raise ValueError("BeliefFeedbackPolicy requires pre-action beliefs.")
        gain = jnp.asarray(feedback_gain)
        bias = jnp.asarray(feedforward)
        expected_prefix = case_shape + (time_grid.num_steps,)
        if tuple(gain.shape[:-2]) != expected_prefix or gain.shape[-1] != state_size:
            raise ValueError(
                "feedback_gain must have shape case_shape + "
                "(horizon, control_size, state_size)."
            )
        control_size = int(gain.shape[-2])
        if tuple(bias.shape) != expected_prefix + (control_size,):
            raise ValueError(
                "feedforward must have shape case_shape + (horizon, control_size)."
            )
        self.feedback_gain = gain
        self.feedforward = bias
        self.time_grid = time_grid
        self.case_shape = case_shape
        self.state_size = state_size
        self.control_size = control_size
        self.observation_timing = observation_timing
        self.information_id = _identifier(information_id, "information_id")
        self.belief_id = _identifier(belief_id, "belief_id")
        self.policy_id = _identifier(policy_id, "policy_id")

    def action(
        self,
        context: DiscreteStepContext,
        belief: GaussianBelief,
        args=None,
        /,
    ) -> Array:
        """Act on the posterior mean; no latent state can enter this interface."""
        del args
        if not isinstance(context, DiscreteStepContext):
            raise TypeError("context must be a DiscreteStepContext.")
        if not isinstance(belief, GaussianBelief):
            raise TypeError("belief must be a GaussianBelief, not a latent state.")
        if belief.dimension != self.state_size:
            raise ValueError("belief dimension must match the policy state size.")
        if belief.belief_id != self.belief_id:
            raise ValueError(
                f"belief_id must match the policy belief_id {self.belief_id!r}."
            )
        raw_index = context.step_index
        invalid_index = (raw_index < 0) | (raw_index >= self.time_grid.num_steps)
        checked_index = _error_if(
            raw_index,
            invalid_index,
            "Belief-feedback step_index lies outside the policy horizon.",
        )
        index = jnp.clip(checked_index, 0, self.time_grid.num_steps - 1)
        expected_source = jnp.take(self.time_grid.times, index)
        expected_target = jnp.take(self.time_grid.times, index + 1)
        index = _error_if(
            index,
            (context.source != expected_source) | (context.target != expected_target),
            "Belief-feedback context does not match the policy time-grid interval.",
        )
        axis = len(self.case_shape)
        gain = jnp.take(self.feedback_gain, index, axis=axis)
        bias = jnp.take(self.feedforward, index, axis=axis)
        return ein.contract("...ij,j->...i", gain, belief.mean) + bias


class CentralizedLQGResult(StrictModule):
    """Centralized belief feedback, Kalman schedule, and exact trace evidence.

    ``predicted_means`` and ``posterior_means`` are the unconditional means of
    their respective random conditional means, so they agree at observation
    updates. A realized posterior mean enters control only through
    :meth:`BeliefFeedbackPolicy.action`. ``initial_covariance_cost`` is the
    deterministic Riccati trace against the full prior covariance, while
    ``initial_observation_trace_cost`` is the part added when averaging the
    posterior-mean value over the first observation.
    """

    problem: CentralizedLQGProblem
    deterministic_result: FiniteHorizonLQRResult
    policy: BeliefFeedbackPolicy
    value: QuadraticValueFunction
    kalman_gains: Array
    innovation_covariances: Array
    process_covariances: Array
    process_covariance_symmetry_residuals: Array
    process_covariance_minimum_eigenvalues: Array
    measurement_covariance_symmetry_residuals: Array
    measurement_covariance_minimum_eigenvalues: Array
    predicted_means: Array
    posterior_means: Array
    terminal_mean: Array
    expected_actions: Array
    predicted_covariances: Array
    posterior_covariances: Array
    terminal_covariance: Array
    posterior_mean_innovation_covariances: Array
    state_covariance_costs: Array
    posterior_mean_innovation_costs: Array
    future_mean_trace_increments: Array
    value_constant_corrections: Array
    initial_covariance_cost: Array
    initial_observation_trace_cost: Array
    initial_expected_cost: Array
    innovation_solve_residuals: Array
    innovation_solve_statuses: Array
    innovation_solve_relative_residuals: Array
    innovation_solve_successful: Array
    innovation_symmetry_residuals: Array
    innovation_minimum_eigenvalues: Array
    innovation_inactive: Array
    predicted_covariance_symmetry_residuals: Array
    predicted_covariance_minimum_eigenvalues: Array
    posterior_covariance_symmetry_residuals: Array
    posterior_covariance_minimum_eigenvalues: Array
    posterior_mean_innovation_covariance_symmetry_residuals: Array
    posterior_mean_innovation_covariance_minimum_eigenvalues: Array
    terminal_covariance_symmetry_residual: Array
    terminal_covariance_minimum_eigenvalue: Array
    innovation_positive_definite: Array
    innovation_well_posed: Array
    covariance_symmetric: Array
    covariance_positive_semidefinite: Array
    covariance_finite: Array
    output_finite: Array
    valid: Array
    status: Array
    result_label: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    information_id: str = eqx.field(static=True)
    observation_timing: str = eqx.field(static=True)

    @property
    def feedback_gain(self) -> Array:
        return self.policy.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.policy.feedforward


def finite_horizon_centralized_lqg(
    problem: CentralizedLQGProblem,
    /,
    *,
    policy_id: str = "lqg:finite-horizon-centralized-belief",
    tolerance: float = 1.0e-9,
    cost_tolerance: float = 1.0e-10,
) -> CentralizedLQGResult:
    """Solve the checked classical centralized Gaussian-belief LQG problem.

    The Riccati recursion is deterministic and the controller acts only on the
    posterior mean. The forward Kalman schedule uses native dense solves and a
    Joseph covariance update. An exactly zero innovation and zero state-
    observation cross covariance is the exact inactive-observation branch with
    zero gain; every other singular innovation is invalid. No inverse,
    pseudoinverse, covariance projection, or symmetry repair is performed.
    """
    if not isinstance(problem, CentralizedLQGProblem):
        raise TypeError("problem must be a CentralizedLQGProblem.")
    resolved_policy_id = _identifier(policy_id, "policy_id")
    deterministic = finite_horizon_lqr(
        problem.dynamics_matrices,
        problem.control_matrices,
        problem.state_costs,
        problem.control_costs,
        problem.terminal_state_cost,
        dynamics_bias=problem.dynamics_bias,
        state_control_cross=problem.state_control_cross,
        state_linear=problem.state_linear,
        control_linear=problem.control_linear,
        stage_constants=problem.stage_constants,
        terminal_linear=problem.terminal_linear,
        terminal_constant=problem.terminal_constant,
        time_grid=problem.time_grid,
        policy_id=resolved_policy_id,
        tolerance=tolerance,
        cost_tolerance=cost_tolerance,
    )

    cases = problem.case_shape
    horizon = problem.horizon
    state_size = problem.state_size
    dtype = jnp.result_type(
        problem.dynamics_matrices,
        problem.observation_matrices,
        problem.measurement_covariances,
        problem.initial_belief.mean,
        problem.initial_belief.covariance,
        float,
    )
    initial_mean = jnp.broadcast_to(
        problem.initial_belief.mean.astype(dtype), cases + (state_size,)
    )
    initial_covariance = jnp.broadcast_to(
        problem.initial_belief.covariance.astype(dtype),
        cases + (state_size, state_size),
    )
    identity = jnp.broadcast_to(
        jnp.eye(state_size, dtype=dtype), cases + (state_size, state_size)
    )
    observation_identity = jnp.broadcast_to(
        jnp.eye(problem.observation_size, dtype=dtype),
        cases + (problem.observation_size, problem.observation_size),
    )
    innovation_policy = LinearSolvePolicy(
        DenseLU(),
        failure=FailurePolicy("status"),
    )
    innovation_rhs_layout = RHSLayout(
        (problem.state_size,),
        names=("state",),
    )

    filter_inputs = (
        _to_time_major(problem.observation_matrices.astype(dtype), 2),
        _to_time_major(problem.measurement_covariances.astype(dtype), 2),
        _to_time_major(problem.dynamics_matrices.astype(dtype), 2),
        _to_time_major(problem.process_covariances.astype(dtype), 2),
    )

    def filter_step(prior_covariance, stage):
        observation, measurement, dynamics, process_covariance = stage
        innovation = (
            observation @ prior_covariance @ jnp.swapaxes(observation, -1, -2)
            + measurement
        )
        cross = observation @ prior_covariance
        inactive = jnp.all(innovation == 0.0, axis=(-2, -1)) & jnp.all(
            cross == 0.0, axis=(-2, -1)
        )
        safe_innovation = jnp.where(
            inactive[..., None, None], observation_identity, innovation
        )
        safe_cross = jnp.where(inactive[..., None, None], 0.0, cross)
        gain_solve = solve(
            LinearSystem(
                DenseLinearOperator(
                    safe_innovation,
                    operator_id=f"{problem.problem_id}:innovation",
                ),
                problem_id=f"{problem.problem_id}:innovation-gain",
            ),
            safe_cross,
            policy=innovation_policy,
            rhs_layout=innovation_rhs_layout,
        )
        gain_transpose = gain_solve.value
        gain = jnp.swapaxes(gain_transpose, -1, -2)
        gain_solve_status = jnp.where(
            inactive[..., None],
            jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32),
            gain_solve.status.astype(jnp.int32),
        )
        gain_solve_relative_residual = jnp.where(
            inactive[..., None],
            jnp.zeros_like(gain_solve.diagnostics.relative_residual),
            gain_solve.diagnostics.relative_residual,
        )
        complement = identity - gain @ observation
        posterior_covariance = complement @ prior_covariance @ jnp.swapaxes(
            complement, -1, -2
        ) + gain @ measurement @ jnp.swapaxes(gain, -1, -2)
        next_prior = (
            dynamics @ posterior_covariance @ jnp.swapaxes(dynamics, -1, -2)
            + process_covariance
        )
        solve_residual = jnp.max(
            jnp.abs(innovation @ gain_transpose - cross), axis=(-2, -1)
        )
        output = (
            prior_covariance,
            innovation,
            gain,
            posterior_covariance,
            solve_residual,
            inactive,
            gain_solve_status,
            gain_solve_relative_residual,
        )
        return next_prior, output

    terminal_covariance, filter_outputs = jax.lax.scan(
        filter_step,
        initial_covariance,
        filter_inputs,
    )
    (
        predicted_covariances_tm,
        innovation_covariances_tm,
        kalman_gains_tm,
        posterior_covariances_tm,
        innovation_solve_residuals_tm,
        innovation_inactive_tm,
        innovation_solve_statuses_tm,
        innovation_solve_relative_residuals_tm,
    ) = filter_outputs
    predicted_covariances = _from_time_major(predicted_covariances_tm, 2)
    innovation_covariances = _from_time_major(innovation_covariances_tm, 2)
    kalman_gains = _from_time_major(kalman_gains_tm, 2)
    posterior_covariances = _from_time_major(posterior_covariances_tm, 2)
    innovation_solve_residuals = _from_time_major(innovation_solve_residuals_tm, 0)
    innovation_solve_statuses = _from_time_major(innovation_solve_statuses_tm, 1)
    innovation_solve_relative_residuals = _from_time_major(
        innovation_solve_relative_residuals_tm, 1
    )
    innovation_inactive = _from_time_major(innovation_inactive_tm, 0)

    mean_inputs = (
        _to_time_major(problem.dynamics_matrices.astype(dtype), 2),
        _to_time_major(problem.control_matrices.astype(dtype), 2),
        _to_time_major(problem.dynamics_bias.astype(dtype), 1),
        _to_time_major(deterministic.feedback_gain.astype(dtype), 2),
        _to_time_major(deterministic.feedforward.astype(dtype), 1),
    )

    def mean_step(predicted_mean, stage):
        dynamics, controls, dynamics_bias, feedback, feedforward = stage
        posterior_mean = predicted_mean
        action = ein.contract("...ij,...j->...i", feedback, posterior_mean) + feedforward
        next_predicted_mean = (
            ein.contract("...ij,...j->...i", dynamics, posterior_mean)
            + ein.contract("...ij,...j->...i", controls, action)
            + dynamics_bias
        )
        return next_predicted_mean, (predicted_mean, posterior_mean, action)

    terminal_mean, mean_outputs = jax.lax.scan(
        mean_step,
        initial_mean,
        mean_inputs,
    )
    predicted_means_tm, posterior_means_tm, expected_actions_tm = mean_outputs
    predicted_means = _from_time_major(predicted_means_tm, 1)
    posterior_means = _from_time_major(posterior_means_tm, 1)
    expected_actions = _from_time_major(expected_actions_tm, 1)

    posterior_mean_innovation_covariances = (
        kalman_gains @ innovation_covariances @ jnp.swapaxes(kalman_gains, -1, -2)
    )
    stage_covariance_costs = 0.5 * ein.contract(
        "...tij,...tji->...t",
        problem.state_costs,
        posterior_covariances,
    )
    terminal_covariance_cost = 0.5 * ein.contract(
        "...ij,...ji->...",
        problem.terminal_state_cost,
        terminal_covariance,
    )
    state_covariance_costs = jnp.concatenate(
        (stage_covariance_costs, terminal_covariance_cost[..., None]), axis=-1
    )
    posterior_mean_innovation_costs = 0.5 * ein.contract(
        "...tij,...tji->...t",
        deterministic.value.matrices[..., :-1, :, :],
        posterior_mean_innovation_covariances,
    )
    future_mean_trace_increments = jnp.concatenate(
        (
            posterior_mean_innovation_costs[..., 1:],
            jnp.zeros(cases + (1,), dtype=dtype),
        ),
        axis=-1,
    )
    stage_corrections = stage_covariance_costs + future_mean_trace_increments
    reversed_corrections = (
        jnp.flip(jnp.cumsum(jnp.flip(stage_corrections, axis=-1), axis=-1), axis=-1)
        + terminal_covariance_cost[..., None]
    )
    value_constant_corrections = jnp.concatenate(
        (reversed_corrections, terminal_covariance_cost[..., None]), axis=-1
    )
    value = QuadraticValueFunction(
        deterministic.value.matrices,
        deterministic.value.linear,
        deterministic.value.constants + value_constant_corrections,
        time_grid=deterministic.value.time_grid,
        case_shape=cases,
    )
    initial_observation_trace_cost = posterior_mean_innovation_costs[..., 0]
    initial_covariance_cost = 0.5 * ein.contract(
        "...ij,...ji->...",
        deterministic.value.matrices[..., 0, :, :],
        initial_covariance,
    )
    initial_matrix = value.matrices[..., 0, :, :]
    initial_linear = value.linear[..., 0, :]
    initial_expected_cost = (
        0.5
        * ein.contract(
            "...i,...ij,...j->...",
            initial_mean,
            initial_matrix,
            initial_mean,
        )
        + ein.contract("...i,...i->...", initial_linear, initial_mean)
        + value.constants[..., 0]
        + initial_observation_trace_cost
    )

    innovation_symmetry, innovation_minimum, innovation_finite = _matrix_evidence(
        innovation_covariances
    )
    predicted_symmetry, predicted_minimum, predicted_finite = _matrix_evidence(
        predicted_covariances
    )
    posterior_symmetry, posterior_minimum, posterior_finite = _matrix_evidence(
        posterior_covariances
    )
    (
        posterior_mean_innovation_symmetry,
        posterior_mean_innovation_minimum,
        posterior_mean_innovation_finite,
    ) = _matrix_evidence(posterior_mean_innovation_covariances)
    terminal_symmetry, terminal_minimum, terminal_finite = _matrix_evidence(
        terminal_covariance
    )
    tolerance_array = jnp.asarray(problem.covariance_tolerance, dtype=dtype)
    innovation_positive_definite = jnp.all(
        innovation_finite
        & (innovation_symmetry <= tolerance_array)
        & (innovation_minimum > tolerance_array),
        axis=-1,
    )
    innovation_well_posed = jnp.all(
        (
            innovation_finite
            & (innovation_symmetry <= tolerance_array)
            & (innovation_minimum > tolerance_array)
        )
        | innovation_inactive,
        axis=-1,
    )
    innovation_solve_successful = jnp.all(
        innovation_solve_statuses == int(LinearSolveStatus.SUCCESS),
        axis=-1,
    )
    innovation_solve_valid = jnp.all(innovation_solve_successful, axis=-1)
    solve_finite = (
        jnp.all(jnp.isfinite(kalman_gains), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(innovation_solve_residuals), axis=-1)
        & jnp.all(jnp.isfinite(innovation_solve_relative_residuals), axis=(-2, -1))
    )
    covariance_symmetric = (
        jnp.all(predicted_symmetry <= tolerance_array, axis=-1)
        & jnp.all(posterior_symmetry <= tolerance_array, axis=-1)
        & jnp.all(
            posterior_mean_innovation_symmetry <= tolerance_array,
            axis=-1,
        )
        & (terminal_symmetry <= tolerance_array)
    )
    covariance_positive_semidefinite = (
        jnp.all(predicted_minimum >= -tolerance_array, axis=-1)
        & jnp.all(posterior_minimum >= -tolerance_array, axis=-1)
        & jnp.all(
            posterior_mean_innovation_minimum >= -tolerance_array,
            axis=-1,
        )
        & (terminal_minimum >= -tolerance_array)
    )
    covariance_finite = (
        jnp.all(predicted_finite, axis=-1)
        & jnp.all(posterior_finite, axis=-1)
        & jnp.all(posterior_mean_innovation_finite, axis=-1)
        & terminal_finite
    )
    output_finite = (
        innovation_finite.all(axis=-1)
        & covariance_finite
        & solve_finite
        & _all_finite(predicted_means, 2)
        & _all_finite(posterior_means, 2)
        & _all_finite(expected_actions, 2)
        & _all_finite(terminal_mean, 1)
        & _all_finite(state_covariance_costs, 1)
        & _all_finite(posterior_mean_innovation_costs, 1)
        & _all_finite(value_constant_corrections, 1)
        & _all_finite(value.constants, 1)
        & jnp.isfinite(initial_covariance_cost)
        & jnp.isfinite(initial_observation_trace_cost)
        & jnp.isfinite(initial_expected_cost)
    )
    valid = (
        deterministic.valid
        & innovation_solve_valid
        & innovation_well_posed
        & covariance_symmetric
        & covariance_positive_semidefinite
        & output_finite
    )
    status = jnp.where(
        ~deterministic.valid,
        int(CentralizedLQGStatus.DETERMINISTIC_SOLVE_FAILED),
        jnp.where(
            ~innovation_solve_valid,
            int(CentralizedLQGStatus.INNOVATION_SOLVE_FAILED),
            jnp.where(
                ~innovation_well_posed,
                int(CentralizedLQGStatus.INNOVATION_COVARIANCE_NOT_POSITIVE_DEFINITE),
                jnp.where(
                    ~covariance_symmetric,
                    int(CentralizedLQGStatus.COVARIANCE_NOT_SYMMETRIC),
                    jnp.where(
                        ~covariance_positive_semidefinite,
                        int(CentralizedLQGStatus.COVARIANCE_NOT_POSITIVE_SEMIDEFINITE),
                        jnp.where(
                            output_finite,
                            int(CentralizedLQGStatus.SUCCESS),
                            int(CentralizedLQGStatus.NONFINITE_OUTPUT),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    assert deterministic.value.time_grid is not None
    policy = BeliefFeedbackPolicy(
        deterministic.feedback_gain,
        deterministic.feedforward,
        time_grid=deterministic.value.time_grid,
        case_shape=cases,
        state_size=state_size,
        observation_timing=problem.observation_timing,
        information_id=problem.information_id,
        belief_id=problem.initial_belief.belief_id,
        policy_id=resolved_policy_id,
    )
    return CentralizedLQGResult(
        problem=problem,
        deterministic_result=deterministic,
        policy=policy,
        value=value,
        kalman_gains=kalman_gains,
        innovation_covariances=innovation_covariances,
        process_covariances=problem.process_covariances,
        process_covariance_symmetry_residuals=(
            problem.process_covariance_symmetry_residuals
        ),
        process_covariance_minimum_eigenvalues=(
            problem.process_covariance_minimum_eigenvalues
        ),
        measurement_covariance_symmetry_residuals=(
            problem.measurement_covariance_symmetry_residuals
        ),
        measurement_covariance_minimum_eigenvalues=(
            problem.measurement_covariance_minimum_eigenvalues
        ),
        predicted_means=predicted_means,
        posterior_means=posterior_means,
        terminal_mean=terminal_mean,
        expected_actions=expected_actions,
        predicted_covariances=predicted_covariances,
        posterior_covariances=posterior_covariances,
        terminal_covariance=terminal_covariance,
        posterior_mean_innovation_covariances=posterior_mean_innovation_covariances,
        state_covariance_costs=state_covariance_costs,
        posterior_mean_innovation_costs=posterior_mean_innovation_costs,
        future_mean_trace_increments=future_mean_trace_increments,
        value_constant_corrections=value_constant_corrections,
        initial_covariance_cost=initial_covariance_cost,
        initial_observation_trace_cost=initial_observation_trace_cost,
        initial_expected_cost=initial_expected_cost,
        innovation_solve_residuals=innovation_solve_residuals,
        innovation_solve_statuses=innovation_solve_statuses,
        innovation_solve_relative_residuals=innovation_solve_relative_residuals,
        innovation_solve_successful=innovation_solve_successful,
        innovation_symmetry_residuals=innovation_symmetry,
        innovation_minimum_eigenvalues=innovation_minimum,
        predicted_covariance_symmetry_residuals=predicted_symmetry,
        predicted_covariance_minimum_eigenvalues=predicted_minimum,
        posterior_covariance_symmetry_residuals=posterior_symmetry,
        posterior_covariance_minimum_eigenvalues=posterior_minimum,
        innovation_inactive=innovation_inactive,
        posterior_mean_innovation_covariance_symmetry_residuals=(
            posterior_mean_innovation_symmetry
        ),
        posterior_mean_innovation_covariance_minimum_eigenvalues=(
            posterior_mean_innovation_minimum
        ),
        terminal_covariance_symmetry_residual=terminal_symmetry,
        terminal_covariance_minimum_eigenvalue=terminal_minimum,
        innovation_positive_definite=innovation_positive_definite,
        covariance_symmetric=covariance_symmetric,
        covariance_positive_semidefinite=covariance_positive_semidefinite,
        covariance_finite=covariance_finite,
        output_finite=output_finite,
        valid=valid,
        status=status,
        innovation_well_posed=innovation_well_posed,
        result_label=_RESULT_LABEL,
        method_id=_METHOD_ID,
        problem_id=problem.problem_id,
        policy_id=resolved_policy_id,
        information_id=problem.information_id,
        observation_timing=problem.observation_timing,
    )


__all__ = [
    "BeliefFeedbackPolicy",
    "CentralizedLQGProblem",
    "CentralizedLQGResult",
    "CentralizedLQGStatus",
    "finite_horizon_centralized_lqg",
]
