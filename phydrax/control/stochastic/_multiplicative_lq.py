#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-horizon multiplicative-noise linear-quadratic control."""

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import TimeGrid
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
    RHSLayout,
    solve,
    TolerancePolicy,
)
from .._lqr import AffineFeedbackPolicy, QuadraticValueFunction


class MultiplicativeLQStateFeedbackStatus(IntEnum):
    """Stable validity codes for multiplicative-noise LQ solves."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONSYMMETRIC_COST = 2
    NOISE_COVARIANCE_NONSYMMETRIC = 3
    NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE = 4
    CONTROL_CURVATURE_NOT_POSITIVE_DEFINITE = 5
    CONTROL_SYSTEM_RANK_DEFICIENT = 6
    CONDITION_LIMIT_REACHED = 7
    LINEAR_SOLVE_FAILED = 8
    NONFINITE_OUTPUT = 9
    RESIDUAL_TOO_LARGE = 10
    DEPENDENCY_FAILED = 11


class FiniteHorizonMultiplicativeLQStateFeedbackDiagnostics(StrictModule):
    """Per-stage covariance, curvature, solve, and Bellman evidence."""

    stage_status: Array
    terminal_status: Array
    linear_status: Array
    diagnostic_available: Array
    state_cost_symmetry_residuals: Array
    control_cost_symmetry_residuals: Array
    terminal_cost_symmetry_residuals: Array
    covariance_symmetry_residuals: Array
    covariance_minimum_eigenvalues: Array
    control_symmetry_residuals: Array
    control_minimum_eigenvalues: Array
    control_ranks: Array
    rank_cutoffs: Array
    minimum_singular_values: Array
    maximum_singular_values: Array
    control_condition_numbers: Array
    linear_relative_residuals: Array
    stationarity_residuals: Array
    bellman_residuals: Array
    value_symmetry_residuals: Array
    maximum_stationarity_residual: Array
    maximum_bellman_residual: Array
    maximum_value_symmetry_residual: Array
    minimum_control_eigenvalue: Array
    maximum_control_condition_number: Array
    first_failed_stage: Array
    finite: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)
    linear_backend: str = eqx.field(static=True)
    linear_method: str = eqx.field(static=True)


class FiniteHorizonMultiplicativeLQStateFeedbackResult(StrictModule):
    """Affine policy, expected value, and multiplicative-noise evidence."""

    policy: AffineFeedbackPolicy
    value: QuadraticValueFunction
    noise_covariances: Array
    trace_increments: Array
    diagnostics: FiniteHorizonMultiplicativeLQStateFeedbackDiagnostics
    valid: Array
    status: Array

    @property
    def feedback_gain(self) -> Array:
        return self.policy.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.policy.feedforward


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    return array


def _require_shape(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> Array:
    array = jnp.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array


def _symmetric(matrix: Array, /) -> Array:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _normalized_symmetry_residual(matrix: Array, /) -> Array:
    difference = matrix - jnp.swapaxes(matrix, -1, -2)
    numerator = jnp.linalg.norm(difference, axis=(-2, -1))
    scale = jnp.linalg.norm(matrix, axis=(-2, -1))
    return numerator / jnp.maximum(jnp.asarray(1.0, dtype=matrix.dtype), scale)


def _case_where(mask: Array, on_true: Array, on_false: Array, /) -> Array:
    extra = on_true.ndim - mask.ndim
    shaped = jnp.reshape(mask, mask.shape + (1,) * extra)
    return jnp.where(shaped, on_true, on_false)


def _all_finite(value: Array, payload_rank: int, /) -> Array:
    axes = tuple(range(value.ndim - payload_rank, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes) if axes else jnp.isfinite(value)


def _normalized_combined_residual(
    residuals: tuple[Array, ...],
    references: tuple[Array, ...],
    payload_ranks: tuple[int, ...],
    /,
) -> Array:
    residual_square = 0.0
    reference_square = 0.0
    for residual, reference, rank in zip(
        residuals, references, payload_ranks, strict=True
    ):
        axes = tuple(range(residual.ndim - rank, residual.ndim))
        residual_square = residual_square + jnp.sum(jnp.square(residual), axis=axes)
        reference_square = reference_square + jnp.sum(jnp.square(reference), axis=axes)
    return jnp.sqrt(residual_square) / (1.0 + jnp.sqrt(reference_square))


def _nanmax(value: Array, axis, /) -> Array:
    available = ~jnp.isnan(value)
    replaced = jnp.where(available, value, -jnp.inf)
    maximum = jnp.max(replaced, axis=axis)
    return jnp.where(jnp.any(available, axis=axis), maximum, jnp.nan)


def _nanmin(value: Array, axis, /) -> Array:
    available = ~jnp.isnan(value)
    replaced = jnp.where(available, value, jnp.inf)
    minimum = jnp.min(replaced, axis=axis)
    return jnp.where(jnp.any(available, axis=axis), minimum, jnp.nan)


def _finite_inputs(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_cost: ArrayLike,
    state_noise_matrices: ArrayLike,
    control_noise_matrices: ArrayLike,
    noise_covariances: ArrayLike,
    dynamics_bias: ArrayLike | None,
    noise_bias: ArrayLike | None,
    state_control_cross: ArrayLike | None,
    state_linear: ArrayLike | None,
    control_linear: ArrayLike | None,
    stage_constants: ArrayLike | None,
    terminal_linear: ArrayLike | None,
    terminal_constant: ArrayLike,
    /,
):
    a = _real_array(dynamics_matrices, "dynamics_matrices")
    if a.ndim < 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            "dynamics_matrices must have shape case_shape + (horizon, n, n)."
        )
    case_shape = tuple(a.shape[:-3])
    horizon = int(a.shape[-3])
    n = int(a.shape[-1])
    if horizon < 1:
        raise ValueError("Finite-horizon multiplicative LQ requires at least one stage.")

    b = _real_array(control_matrices, "control_matrices")
    if b.ndim < 3 or tuple(b.shape[:-3]) != case_shape or b.shape[-3:-1] != (horizon, n):
        raise ValueError(
            "control_matrices must have shape case_shape + (horizon, n, m); "
            f"got {b.shape}."
        )
    m = int(b.shape[-1])
    state_noise = _real_array(state_noise_matrices, "state_noise_matrices")
    expected_rank = len(case_shape) + 4
    if (
        state_noise.ndim != expected_rank
        or tuple(state_noise.shape[: len(case_shape)]) != case_shape
        or state_noise.shape[-4] != horizon
        or state_noise.shape[-2:] != (n, n)
    ):
        raise ValueError(
            "state_noise_matrices must have shape case_shape + "
            f"(horizon, noise_size, n, n); got {state_noise.shape}."
        )
    noise_size = int(state_noise.shape[-3])
    if noise_size < 1:
        raise ValueError("state_noise_matrices must have a positive noise_size.")

    values = [
        _require_shape(a, case_shape + (horizon, n, n), "dynamics_matrices"),
        _require_shape(b, case_shape + (horizon, n, m), "control_matrices"),
        _require_shape(
            _real_array(state_costs, "state_costs"),
            case_shape + (horizon, n, n),
            "state_costs",
        ),
        _require_shape(
            _real_array(control_costs, "control_costs"),
            case_shape + (horizon, m, m),
            "control_costs",
        ),
        _require_shape(
            _real_array(terminal_state_cost, "terminal_state_cost"),
            case_shape + (n, n),
            "terminal_state_cost",
        ),
        _require_shape(
            state_noise,
            case_shape + (horizon, noise_size, n, n),
            "state_noise_matrices",
        ),
        _require_shape(
            _real_array(control_noise_matrices, "control_noise_matrices"),
            case_shape + (horizon, noise_size, n, m),
            "control_noise_matrices",
        ),
        _require_shape(
            _real_array(noise_covariances, "noise_covariances"),
            case_shape + (horizon, noise_size, noise_size),
            "noise_covariances",
        ),
    ]
    dtype = jnp.result_type(*values, float)
    zeros = lambda shape: jnp.zeros(shape, dtype=dtype)
    c = (
        zeros(case_shape + (horizon, n))
        if dynamics_bias is None
        else _require_shape(
            _real_array(dynamics_bias, "dynamics_bias"),
            case_shape + (horizon, n),
            "dynamics_bias",
        )
    )
    d = (
        zeros(case_shape + (horizon, noise_size, n))
        if noise_bias is None
        else _require_shape(
            _real_array(noise_bias, "noise_bias"),
            case_shape + (horizon, noise_size, n),
            "noise_bias",
        )
    )
    cross = (
        zeros(case_shape + (horizon, n, m))
        if state_control_cross is None
        else _require_shape(
            _real_array(state_control_cross, "state_control_cross"),
            case_shape + (horizon, n, m),
            "state_control_cross",
        )
    )
    q_linear = (
        zeros(case_shape + (horizon, n))
        if state_linear is None
        else _require_shape(
            _real_array(state_linear, "state_linear"),
            case_shape + (horizon, n),
            "state_linear",
        )
    )
    r_linear = (
        zeros(case_shape + (horizon, m))
        if control_linear is None
        else _require_shape(
            _real_array(control_linear, "control_linear"),
            case_shape + (horizon, m),
            "control_linear",
        )
    )
    constants = (
        zeros(case_shape + (horizon,))
        if stage_constants is None
        else _require_shape(
            _real_array(stage_constants, "stage_constants"),
            case_shape + (horizon,),
            "stage_constants",
        )
    )
    terminal_vector = (
        zeros(case_shape + (n,))
        if terminal_linear is None
        else _require_shape(
            _real_array(terminal_linear, "terminal_linear"),
            case_shape + (n,),
            "terminal_linear",
        )
    )
    terminal_scalar = _real_array(terminal_constant, "terminal_constant")
    if terminal_scalar.shape == () and case_shape:
        terminal_scalar = jnp.broadcast_to(terminal_scalar, case_shape)
    terminal_scalar = _require_shape(terminal_scalar, case_shape, "terminal_constant")
    values.extend(
        (c, d, cross, q_linear, r_linear, constants, terminal_vector, terminal_scalar)
    )
    return (
        tuple(value.astype(dtype) for value in values),
        case_shape,
        horizon,
        n,
        m,
        noise_size,
    )


def finite_horizon_multiplicative_lq_state_feedback(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_cost: ArrayLike,
    /,
    *,
    state_noise_matrices: ArrayLike,
    control_noise_matrices: ArrayLike,
    noise_covariances: ArrayLike,
    noise_bias: ArrayLike | None = None,
    dynamics_bias: ArrayLike | None = None,
    state_control_cross: ArrayLike | None = None,
    state_linear: ArrayLike | None = None,
    control_linear: ArrayLike | None = None,
    stage_constants: ArrayLike | None = None,
    terminal_linear: ArrayLike | None = None,
    terminal_constant: ArrayLike = 0.0,
    time_grid: TimeGrid | None = None,
    policy_id: str = "multiplicative-lq:finite-horizon-state-feedback",
    tolerance: float = 1e-9,
    symmetry_tolerance: float = 1e-10,
    covariance_tolerance: float = 0.0,
    curvature_tolerance: float = 1e-10,
    rank_relative_tolerance: float | None = None,
    rank_absolute_tolerance: float | None = None,
    maximum_condition: float | None = None,
) -> FiniteHorizonMultiplicativeLQStateFeedbackResult:
    """Solve an exact discrete multiplicative-noise LQ control problem.

    Dynamics are ``x[k+1] = A[k]x[k] + B[k]u[k] + c[k] +
    sum_r (C[k,r]x[k] + D[k,r]u[k] + d[k,r]) xi[k,r]``. The noise is
    zero mean with the explicitly supplied channel covariance
    ``E[xi[k,r] xi[k,s]] = Gamma[k,r,s]``. Every stage and noise channel
    axis is explicit; no time or noise broadcasting occurs.
    """
    scalar_parameters = (
        ("tolerance", tolerance, False),
        ("symmetry_tolerance", symmetry_tolerance, True),
        ("covariance_tolerance", covariance_tolerance, True),
        ("curvature_tolerance", curvature_tolerance, True),
    )
    for name, value, allow_zero in scalar_parameters:
        resolved = float(value)
        invalid = (
            not math.isfinite(resolved)
            or resolved < 0.0
            or (not allow_zero and resolved == 0.0)
        )
        if invalid:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {relation}.")
    relative_rank = (
        None if rank_relative_tolerance is None else float(rank_relative_tolerance)
    )
    absolute_rank = (
        None if rank_absolute_tolerance is None else float(rank_absolute_tolerance)
    )
    if relative_rank is not None and (
        not math.isfinite(relative_rank) or relative_rank < 0.0
    ):
        raise ValueError("rank_relative_tolerance must be finite and non-negative.")
    if absolute_rank is not None and (
        not math.isfinite(absolute_rank) or absolute_rank < 0.0
    ):
        raise ValueError("rank_absolute_tolerance must be finite and non-negative.")
    condition_limit = None if maximum_condition is None else float(maximum_condition)
    if condition_limit is not None and (
        not math.isfinite(condition_limit) or condition_limit <= 1.0
    ):
        raise ValueError("maximum_condition must be finite and exceed one or be None.")

    values, case_shape, horizon, n, m, _ = _finite_inputs(
        dynamics_matrices,
        control_matrices,
        state_costs,
        control_costs,
        terminal_state_cost,
        state_noise_matrices,
        control_noise_matrices,
        noise_covariances,
        dynamics_bias,
        noise_bias,
        state_control_cross,
        state_linear,
        control_linear,
        stage_constants,
        terminal_linear,
        terminal_constant,
    )
    (
        a,
        b,
        q_raw,
        r_raw,
        q_terminal_raw,
        state_noise,
        control_noise,
        gamma_raw,
        c,
        noise_offset,
        cross,
        q_linear,
        r_linear,
        constants,
        q_terminal_linear,
        terminal_constant_,
    ) = values
    if time_grid is None:
        time_grid = TimeGrid(
            jnp.arange(horizon + 1, dtype=a.dtype), time_id=f"{policy_id}:time"
        )
    elif not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid or None.")
    if time_grid.num_steps != horizon:
        raise ValueError(
            f"time_grid must contain {horizon + 1} times for this problem horizon."
        )

    q_symmetry = _normalized_symmetry_residual(q_raw)
    r_symmetry = _normalized_symmetry_residual(r_raw)
    terminal_symmetry = _normalized_symmetry_residual(q_terminal_raw)
    covariance_symmetry = jnp.max(
        jnp.abs(gamma_raw - jnp.swapaxes(gamma_raw, -1, -2)),
        axis=(-2, -1),
    )
    q = _symmetric(q_raw)
    r = _symmetric(r_raw)
    q_terminal = _symmetric(q_terminal_raw)
    gamma = _symmetric(gamma_raw)
    covariance_minimum = jnp.min(
        jnp.linalg.eigvalsh(jax.lax.stop_gradient(gamma)), axis=-1
    )

    terminal_finite = (
        _all_finite(q_terminal_raw, 2)
        & _all_finite(q_terminal_linear, 1)
        & jnp.isfinite(terminal_constant_)
    )
    terminal_status = jnp.where(
        ~terminal_finite,
        int(MultiplicativeLQStateFeedbackStatus.NONFINITE_INPUT),
        jnp.where(
            terminal_symmetry > symmetry_tolerance,
            int(MultiplicativeLQStateFeedbackStatus.NONSYMMETRIC_COST),
            int(MultiplicativeLQStateFeedbackStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    terminal_valid = terminal_status == int(MultiplicativeLQStateFeedbackStatus.SUCCESS)
    first_failed = jnp.where(
        terminal_valid,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(horizon, dtype=jnp.int32),
    )
    carry_terminal = _case_where(
        terminal_valid, q_terminal, jnp.full_like(q_terminal, jnp.nan)
    )
    carry_linear = _case_where(
        terminal_valid,
        q_terminal_linear,
        jnp.full_like(q_terminal_linear, jnp.nan),
    )
    carry_constant = jnp.where(
        terminal_valid, terminal_constant_, jnp.full_like(terminal_constant_, jnp.nan)
    )

    rank_relative = (
        float(m * jnp.finfo(a.dtype).eps) if relative_rank is None else relative_rank
    )
    rank_absolute = 0.0 if absolute_rank is None else absolute_rank
    linear_policy = LinearSolvePolicy(
        DenseLU(),
        tolerance=TolerancePolicy(relative=tolerance, absolute=tolerance),
        rank=RankPolicy(require_full_rank=False),
        differentiation=DifferentiationPolicy("mathematical"),
        failure=FailurePolicy("status"),
    )
    svd_policy = FactorizationPolicy(
        "svd",
        rank=RankPolicy(
            relative_cutoff=rank_relative,
            absolute_cutoff=rank_absolute,
            require_full_rank=False,
        ),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    rhs_layout = RHSLayout((n + 1,), names=("feedback-and-affine",))

    def to_time_major(value: Array, payload_rank: int) -> Array:
        return jnp.moveaxis(value, -(payload_rank + 1), 0)

    inputs = (
        jnp.arange(horizon, dtype=jnp.int32),
        to_time_major(a, 2),
        to_time_major(b, 2),
        to_time_major(q, 2),
        to_time_major(r, 2),
        to_time_major(state_noise, 3),
        to_time_major(control_noise, 3),
        to_time_major(gamma, 2),
        to_time_major(c, 1),
        to_time_major(noise_offset, 2),
        to_time_major(cross, 2),
        to_time_major(q_linear, 1),
        to_time_major(r_linear, 1),
        to_time_major(constants, 0),
        to_time_major(q_symmetry, 0),
        to_time_major(r_symmetry, 0),
        to_time_major(covariance_symmetry, 0),
        to_time_major(covariance_minimum, 0),
    )

    def step(carry, stage):
        (
            p_next,
            linear_next,
            constant_next,
            continuation_valid,
            causal_stage,
            causal_status,
        ) = carry
        (
            stage_index,
            a_t,
            b_t,
            q_t,
            r_t,
            state_noise_t,
            control_noise_t,
            gamma_t,
            c_t,
            noise_offset_t,
            cross_t,
            q_t_linear,
            r_t_linear,
            d_t,
            q_symmetry_t,
            r_symmetry_t,
            covariance_symmetry_t,
            covariance_minimum_t,
        ) = stage

        control_hessian = (
            r_t
            + jnp.swapaxes(b_t, -1, -2) @ p_next @ b_t
            + ein.contract(
                "...rai,...ab,...sbj,...rs->...ij",
                control_noise_t,
                p_next,
                control_noise_t,
                gamma_t,
            )
        )
        state_control = (
            jnp.swapaxes(b_t, -1, -2) @ p_next @ a_t
            + jnp.swapaxes(cross_t, -1, -2)
            + ein.contract(
                "...rai,...ab,...sbj,...rs->...ij",
                control_noise_t,
                p_next,
                state_noise_t,
                gamma_t,
            )
        )
        affine_next = ein.contract("...ij,...j->...i", p_next, c_t) + linear_next
        control_affine = (
            r_t_linear
            + ein.contract("...ji,...j->...i", b_t, affine_next)
            + ein.contract(
                "...rai,...ab,...sb,...rs->...i",
                control_noise_t,
                p_next,
                noise_offset_t,
                gamma_t,
            )
        )
        state_hessian = (
            q_t
            + jnp.swapaxes(a_t, -1, -2) @ p_next @ a_t
            + ein.contract(
                "...rai,...ab,...sbj,...rs->...ij",
                state_noise_t,
                p_next,
                state_noise_t,
                gamma_t,
            )
        )
        state_affine = (
            q_t_linear
            + ein.contract("...ji,...j->...i", a_t, affine_next)
            + ein.contract(
                "...rai,...ab,...sb,...rs->...i",
                state_noise_t,
                p_next,
                noise_offset_t,
                gamma_t,
            )
        )
        delta = (
            d_t
            + constant_next
            + 0.5 * ein.contract("...i,...ij,...j->...", c_t, p_next, c_t)
            + ein.contract("...i,...i->...", linear_next, c_t)
            + 0.5
            * ein.contract(
                "...ra,...ab,...sb,...rs->...",
                noise_offset_t,
                p_next,
                noise_offset_t,
                gamma_t,
            )
        )
        rhs = jnp.concatenate((state_control, control_affine[..., None]), axis=-1)
        solve_result = solve(
            LinearSystem(
                DenseLinearOperator(
                    control_hessian,
                    operator_id="control:multiplicative-lq:lu",
                ),
                problem_id="control:multiplicative-lq:stage",
            ),
            rhs,
            policy=linear_policy,
            rhs_layout=rhs_layout,
        )
        solved = solve_result.value
        feedback = -solved[..., :n]
        feedforward = -solved[..., n]

        diagnostic_hessian = jax.lax.stop_gradient(control_hessian)
        svd_factorization = factorize(
            DenseLinearOperator(
                diagnostic_hessian,
                operator_id="control:multiplicative-lq:svd",
            ),
            svd_policy,
        )
        singular_values = svd_factorization.singular_values()
        maximum_singular = jnp.max(singular_values, axis=-1)
        minimum_singular = jnp.min(singular_values, axis=-1)
        rank_cutoff = rank_absolute + rank_relative * maximum_singular
        rank = jnp.sum(singular_values > rank_cutoff[..., None], axis=-1).astype(
            jnp.int32
        )
        condition = jnp.where(
            minimum_singular > 0.0,
            maximum_singular / minimum_singular,
            jnp.asarray(jnp.inf, dtype=a.dtype),
        )
        control_symmetry = _normalized_symmetry_residual(control_hessian)
        control_minimum = jnp.min(
            jnp.linalg.eigvalsh(jax.lax.stop_gradient(_symmetric(control_hessian))),
            axis=-1,
        )

        p_raw = state_hessian + jnp.swapaxes(state_control, -1, -2) @ feedback
        p_current = _symmetric(p_raw)
        linear_current = state_affine + ein.contract(
            "...ji,...j->...i", state_control, feedforward
        )
        constant_current = delta + 0.5 * ein.contract(
            "...i,...i->...", control_affine, feedforward
        )

        closed_loop = a_t + b_t @ feedback
        closed_bias = c_t + ein.contract("...ij,...j->...i", b_t, feedforward)
        closed_noise = state_noise_t + ein.contract(
            "...rij,...jk->...rik", control_noise_t, feedback
        )
        closed_noise_bias = noise_offset_t + ein.contract(
            "...rij,...j->...ri", control_noise_t, feedforward
        )
        trace_increment = 0.5 * ein.contract(
            "...ra,...ab,...sb,...rs->...",
            closed_noise_bias,
            p_next,
            closed_noise_bias,
            gamma_t,
        )
        bellman_p = (
            q_t
            + cross_t @ feedback
            + jnp.swapaxes(feedback, -1, -2) @ jnp.swapaxes(cross_t, -1, -2)
            + jnp.swapaxes(feedback, -1, -2) @ r_t @ feedback
            + jnp.swapaxes(closed_loop, -1, -2) @ p_next @ closed_loop
            + ein.contract(
                "...rai,...ab,...sbj,...rs->...ij",
                closed_noise,
                p_next,
                closed_noise,
                gamma_t,
            )
        )
        bellman_linear = (
            q_t_linear
            + ein.contract("...ij,...j->...i", cross_t, feedforward)
            + ein.contract(
                "...ji,...j->...i",
                feedback,
                ein.contract("...ij,...j->...i", r_t, feedforward) + r_t_linear,
            )
            + ein.contract(
                "...ji,...j->...i",
                closed_loop,
                ein.contract("...ij,...j->...i", p_next, closed_bias) + linear_next,
            )
            + ein.contract(
                "...rai,...ab,...sb,...rs->...i",
                closed_noise,
                p_next,
                closed_noise_bias,
                gamma_t,
            )
        )
        bellman_constant = (
            d_t
            + constant_next
            + 0.5 * ein.contract("...i,...ij,...j->...", feedforward, r_t, feedforward)
            + ein.contract("...i,...i->...", r_t_linear, feedforward)
            + 0.5 * ein.contract("...i,...ij,...j->...", closed_bias, p_next, closed_bias)
            + ein.contract("...i,...i->...", linear_next, closed_bias)
            + trace_increment
        )
        stationarity_matrix = control_hessian @ feedback + state_control
        stationarity_vector = (
            ein.contract("...ij,...j->...i", control_hessian, feedforward)
            + control_affine
        )
        stationarity_residual = _normalized_combined_residual(
            (stationarity_matrix, stationarity_vector),
            (state_control, control_affine),
            (2, 1),
        )
        bellman_residual = _normalized_combined_residual(
            (
                p_current - bellman_p,
                linear_current - bellman_linear,
                constant_current - bellman_constant,
            ),
            (bellman_p, bellman_linear, bellman_constant),
            (2, 1, 0),
        )
        value_symmetry = _normalized_symmetry_residual(p_raw)

        input_finite = (
            _all_finite(a_t, 2)
            & _all_finite(b_t, 2)
            & _all_finite(q_t, 2)
            & _all_finite(r_t, 2)
            & _all_finite(state_noise_t, 3)
            & _all_finite(control_noise_t, 3)
            & _all_finite(gamma_t, 2)
            & _all_finite(c_t, 1)
            & _all_finite(noise_offset_t, 2)
            & _all_finite(cross_t, 2)
            & _all_finite(q_t_linear, 1)
            & _all_finite(r_t_linear, 1)
            & jnp.isfinite(d_t)
        )
        costs_symmetric = (q_symmetry_t <= symmetry_tolerance) & (
            r_symmetry_t <= symmetry_tolerance
        )
        covariance_symmetric = covariance_symmetry_t <= covariance_tolerance
        covariance_psd = covariance_minimum_t >= -covariance_tolerance
        curvature_finite = jnp.isfinite(control_minimum) & jnp.isfinite(control_symmetry)
        curvature_valid = control_minimum > curvature_tolerance
        svd_finite = jnp.all(jnp.isfinite(singular_values), axis=-1)
        diagnostic_available = continuation_valid & input_finite & svd_finite
        rank_reported = jnp.where(diagnostic_available, rank, -1)
        cutoff_reported = jnp.where(diagnostic_available, rank_cutoff, jnp.nan)
        minimum_singular_reported = jnp.where(
            diagnostic_available, minimum_singular, jnp.nan
        )
        maximum_singular_reported = jnp.where(
            diagnostic_available, maximum_singular, jnp.nan
        )
        condition_reported = jnp.where(diagnostic_available, condition, jnp.nan)
        rank_valid = rank == m
        condition_valid = (
            jnp.ones_like(condition, dtype=bool)
            if condition_limit is None
            else condition <= condition_limit
        )
        linear_status = solve_result.status.astype(jnp.int32)
        linear_valid = jnp.all(linear_status == int(LinearSolveStatus.SUCCESS), axis=-1)
        linear_relative = jnp.max(solve_result.diagnostics.relative_residual, axis=-1)
        output_finite = (
            _all_finite(feedback, 2)
            & _all_finite(feedforward, 1)
            & _all_finite(p_current, 2)
            & _all_finite(linear_current, 1)
            & jnp.isfinite(constant_current)
            & jnp.isfinite(trace_increment)
            & jnp.isfinite(stationarity_residual)
            & jnp.isfinite(bellman_residual)
            & jnp.isfinite(value_symmetry)
        )
        residual_valid = (
            (linear_relative <= tolerance)
            & (stationarity_residual <= tolerance)
            & (bellman_residual <= tolerance)
            & (value_symmetry <= symmetry_tolerance)
            & (control_symmetry <= symmetry_tolerance)
        )
        direct_status = jnp.where(
            ~input_finite,
            int(MultiplicativeLQStateFeedbackStatus.NONFINITE_INPUT),
            jnp.where(
                ~costs_symmetric,
                int(MultiplicativeLQStateFeedbackStatus.NONSYMMETRIC_COST),
                jnp.where(
                    ~covariance_symmetric,
                    int(
                        MultiplicativeLQStateFeedbackStatus.NOISE_COVARIANCE_NONSYMMETRIC
                    ),
                    jnp.where(
                        ~covariance_psd,
                        int(
                            MultiplicativeLQStateFeedbackStatus.NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE
                        ),
                        jnp.where(
                            ~curvature_finite | ~svd_finite,
                            int(MultiplicativeLQStateFeedbackStatus.NONFINITE_OUTPUT),
                            jnp.where(
                                ~curvature_valid,
                                int(
                                    MultiplicativeLQStateFeedbackStatus.CONTROL_CURVATURE_NOT_POSITIVE_DEFINITE
                                ),
                                jnp.where(
                                    ~rank_valid,
                                    int(
                                        MultiplicativeLQStateFeedbackStatus.CONTROL_SYSTEM_RANK_DEFICIENT
                                    ),
                                    jnp.where(
                                        ~condition_valid,
                                        int(
                                            MultiplicativeLQStateFeedbackStatus.CONDITION_LIMIT_REACHED
                                        ),
                                        jnp.where(
                                            ~linear_valid,
                                            int(
                                                MultiplicativeLQStateFeedbackStatus.LINEAR_SOLVE_FAILED
                                            ),
                                            jnp.where(
                                                ~output_finite,
                                                int(
                                                    MultiplicativeLQStateFeedbackStatus.NONFINITE_OUTPUT
                                                ),
                                                jnp.where(
                                                    ~residual_valid,
                                                    int(
                                                        MultiplicativeLQStateFeedbackStatus.RESIDUAL_TOO_LARGE
                                                    ),
                                                    int(
                                                        MultiplicativeLQStateFeedbackStatus.SUCCESS
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        stage_status = jnp.where(
            continuation_valid,
            direct_status,
            int(MultiplicativeLQStateFeedbackStatus.DEPENDENCY_FAILED),
        ).astype(jnp.int32)
        local_valid = direct_status == int(MultiplicativeLQStateFeedbackStatus.SUCCESS)
        stage_valid = continuation_valid & local_valid
        direct_failure = continuation_valid & ~local_valid
        next_failed_stage = jnp.where(direct_failure, stage_index, causal_stage)
        next_causal_status = jnp.where(direct_failure, direct_status, causal_status)
        next_p = _case_where(stage_valid, p_current, jnp.full_like(p_current, jnp.nan))
        next_linear = _case_where(
            stage_valid,
            linear_current,
            jnp.full_like(linear_current, jnp.nan),
        )
        next_constant = jnp.where(
            stage_valid,
            constant_current,
            jnp.full_like(constant_current, jnp.nan),
        )
        output = (
            p_current,
            linear_current,
            constant_current,
            feedback,
            feedforward,
            trace_increment,
            stage_status,
            linear_status,
            diagnostic_available,
            control_symmetry,
            control_minimum,
            rank_reported,
            cutoff_reported,
            minimum_singular_reported,
            maximum_singular_reported,
            condition_reported,
            linear_relative,
            stationarity_residual,
            bellman_residual,
            value_symmetry,
        )
        return (
            next_p,
            next_linear,
            next_constant,
            stage_valid,
            next_failed_stage,
            next_causal_status,
        ), output

    initial_carry = (
        carry_terminal,
        carry_linear,
        carry_constant,
        terminal_valid,
        first_failed,
        terminal_status,
    )
    final_carry, outputs = jax.lax.scan(step, initial_carry, inputs, reverse=True)
    _, _, _, _, first_failed_stage, status = final_carry
    (
        p_stages,
        linear_stages,
        constant_stages,
        feedback,
        feedforward,
        trace_increments,
        stage_status,
        linear_status,
        diagnostic_available,
        control_symmetry,
        control_minimum,
        ranks,
        rank_cutoffs,
        minimum_singular,
        maximum_singular,
        conditions,
        linear_relative,
        stationarity,
        bellman,
        value_symmetry,
    ) = outputs

    p_stages = jnp.moveaxis(p_stages, 0, -3)
    linear_stages = jnp.moveaxis(linear_stages, 0, -2)
    constant_stages = jnp.moveaxis(constant_stages, 0, -1)
    p_all = jnp.concatenate((p_stages, q_terminal[..., None, :, :]), axis=-3)
    linear_all = jnp.concatenate(
        (linear_stages, q_terminal_linear[..., None, :]), axis=-2
    )
    constant_all = jnp.concatenate(
        (constant_stages, terminal_constant_[..., None]), axis=-1
    )
    feedback = jnp.moveaxis(feedback, 0, -3)
    feedforward = jnp.moveaxis(feedforward, 0, -2)
    trace_increments = jnp.moveaxis(trace_increments, 0, -1)
    stage_status = jnp.moveaxis(stage_status, 0, -1)
    linear_status = jnp.moveaxis(linear_status, 0, -2)
    diagnostic_available = jnp.moveaxis(diagnostic_available, 0, -1)
    control_symmetry = jnp.moveaxis(control_symmetry, 0, -1)
    control_minimum = jnp.moveaxis(control_minimum, 0, -1)
    ranks = jnp.moveaxis(ranks, 0, -1)
    rank_cutoffs = jnp.moveaxis(rank_cutoffs, 0, -1)
    minimum_singular = jnp.moveaxis(minimum_singular, 0, -1)
    maximum_singular = jnp.moveaxis(maximum_singular, 0, -1)
    conditions = jnp.moveaxis(conditions, 0, -1)
    linear_relative = jnp.moveaxis(linear_relative, 0, -1)
    stationarity = jnp.moveaxis(stationarity, 0, -1)
    bellman = jnp.moveaxis(bellman, 0, -1)
    value_symmetry = jnp.moveaxis(value_symmetry, 0, -1)

    maximum_stationarity = _nanmax(stationarity, -1)
    maximum_bellman = _nanmax(bellman, -1)
    maximum_value_symmetry = _nanmax(value_symmetry, -1)
    minimum_control = _nanmin(control_minimum, -1)
    maximum_condition_number = _nanmax(conditions, -1)
    result_finite = (
        _all_finite(feedback, 3)
        & _all_finite(feedforward, 2)
        & _all_finite(p_all, 3)
        & _all_finite(linear_all, 2)
        & _all_finite(constant_all, 1)
        & _all_finite(trace_increments, 1)
    )
    valid = status == int(MultiplicativeLQStateFeedbackStatus.SUCCESS)
    policy = AffineFeedbackPolicy(
        feedback,
        feedforward,
        time_grid=time_grid,
        state_size=n,
        case_shape=case_shape,
        policy_id=policy_id,
        _allow_nonfinite=True,
    )
    value = QuadraticValueFunction(
        p_all,
        linear_all,
        constant_all,
        time_grid=time_grid,
        case_shape=case_shape,
    )
    diagnostics = FiniteHorizonMultiplicativeLQStateFeedbackDiagnostics(
        stage_status=stage_status,
        terminal_status=terminal_status,
        linear_status=linear_status,
        diagnostic_available=diagnostic_available,
        state_cost_symmetry_residuals=q_symmetry,
        control_cost_symmetry_residuals=r_symmetry,
        terminal_cost_symmetry_residuals=terminal_symmetry,
        covariance_symmetry_residuals=covariance_symmetry,
        covariance_minimum_eigenvalues=covariance_minimum,
        control_symmetry_residuals=control_symmetry,
        control_minimum_eigenvalues=control_minimum,
        control_ranks=ranks,
        rank_cutoffs=rank_cutoffs,
        minimum_singular_values=minimum_singular,
        maximum_singular_values=maximum_singular,
        control_condition_numbers=conditions,
        linear_relative_residuals=linear_relative,
        stationarity_residuals=stationarity,
        bellman_residuals=bellman,
        value_symmetry_residuals=value_symmetry,
        maximum_stationarity_residual=maximum_stationarity,
        maximum_bellman_residual=maximum_bellman,
        maximum_value_symmetry_residual=maximum_value_symmetry,
        minimum_control_eigenvalue=minimum_control,
        maximum_control_condition_number=maximum_condition_number,
        first_failed_stage=first_failed_stage,
        finite=result_finite,
        valid=valid,
        status=status,
        method="backward-multiplicative-noise-lq",
        linear_backend="jax-dense",
        linear_method=DenseLU().name,
    )
    return FiniteHorizonMultiplicativeLQStateFeedbackResult(
        policy=policy,
        value=value,
        noise_covariances=gamma,
        trace_increments=trace_increments,
        diagnostics=diagnostics,
        valid=valid,
        status=status,
    )
