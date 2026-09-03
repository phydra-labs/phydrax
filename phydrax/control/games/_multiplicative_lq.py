#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-horizon multiplicative-noise LQ feedback Nash games."""

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
from ..stochastic._multiplicative_lq import (
    _all_finite,
    _case_where,
    _nanmax,
    _nanmin,
    _normalized_combined_residual,
    _normalized_symmetry_residual,
    _real_array,
    _require_shape,
    _symmetric,
)
from ._layout import PlayerControlPartition
from ._linear_quadratic import _game_inputs


class MultiplicativeLQFeedbackNashStatus(IntEnum):
    """Stable validity codes for multiplicative-noise feedback Nash solves."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONSYMMETRIC_COST = 2
    NOISE_COVARIANCE_NONSYMMETRIC = 3
    NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE = 4
    OWN_CURVATURE_NOT_POSITIVE_DEFINITE = 5
    COUPLED_SYSTEM_RANK_DEFICIENT = 6
    CONDITION_LIMIT_REACHED = 7
    LINEAR_SOLVE_FAILED = 8
    NONFINITE_OUTPUT = 9
    RESIDUAL_TOO_LARGE = 10
    DEPENDENCY_FAILED = 11


class FiniteHorizonMultiplicativeLQFeedbackNashDiagnostics(StrictModule):
    """Per-stage covariance, equilibrium, curvature, and value evidence."""

    stage_status: Array
    terminal_status: Array
    linear_status: Array
    diagnostic_available: Array
    state_cost_symmetry_residuals: Array
    control_cost_symmetry_residuals: Array
    terminal_cost_symmetry_residuals: Array
    covariance_symmetry_residuals: Array
    covariance_minimum_eigenvalues: Array
    own_control_symmetry_residuals: Array
    own_control_minimum_eigenvalues: Array
    coupled_ranks: Array
    rank_cutoffs: Array
    minimum_singular_values: Array
    maximum_singular_values: Array
    coupled_condition_numbers: Array
    linear_relative_residuals: Array
    stationarity_residuals: Array
    bellman_residuals: Array
    value_symmetry_residuals: Array
    maximum_stationarity_residual: Array
    maximum_bellman_residual: Array
    maximum_value_symmetry_residual: Array
    minimum_own_control_eigenvalue: Array
    maximum_coupled_condition_number: Array
    first_failed_stage: Array
    finite: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)
    linear_backend: str = eqx.field(static=True)
    linear_method: str = eqx.field(static=True)


class FiniteHorizonMultiplicativeLQFeedbackNashResult(StrictModule):
    """Joint affine strategy and each player's exact expected value."""

    partition: PlayerControlPartition
    policy: AffineFeedbackPolicy
    values: tuple[QuadraticValueFunction, ...]
    noise_covariances: Array
    trace_increments: Array
    diagnostics: FiniteHorizonMultiplicativeLQFeedbackNashDiagnostics
    valid: Array
    status: Array

    @property
    def feedback_gain(self) -> Array:
        return self.policy.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.policy.feedforward


def _noise_inputs(
    state_noise_matrices: ArrayLike,
    control_noise_matrices: ArrayLike,
    noise_covariances: ArrayLike,
    noise_bias: ArrayLike | None,
    case_shape: tuple[int, ...],
    horizon: int,
    n: int,
    m: int,
    /,
) -> tuple[Array, Array, Array, Array]:
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
    control_noise = _require_shape(
        _real_array(control_noise_matrices, "control_noise_matrices"),
        case_shape + (horizon, noise_size, n, m),
        "control_noise_matrices",
    )
    covariance = _require_shape(
        _real_array(noise_covariances, "noise_covariances"),
        case_shape + (horizon, noise_size, noise_size),
        "noise_covariances",
    )
    offset = (
        jnp.zeros(case_shape + (horizon, noise_size, n), dtype=state_noise.dtype)
        if noise_bias is None
        else _require_shape(
            _real_array(noise_bias, "noise_bias"),
            case_shape + (horizon, noise_size, n),
            "noise_bias",
        )
    )
    return state_noise, control_noise, covariance, offset


def finite_horizon_multiplicative_lq_feedback_nash(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_costs: ArrayLike,
    partition: PlayerControlPartition,
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
    terminal_constants: ArrayLike = 0.0,
    time_grid: TimeGrid | None = None,
    policy_id: str = "game:multiplicative-lq-feedback-nash",
    tolerance: float = 1e-9,
    symmetry_tolerance: float = 1e-10,
    covariance_tolerance: float = 0.0,
    curvature_tolerance: float = 1e-10,
    rank_relative_tolerance: float | None = None,
    rank_absolute_tolerance: float | None = None,
    maximum_condition: float | None = None,
) -> FiniteHorizonMultiplicativeLQFeedbackNashResult:
    """Solve an exact all-minimizer multiplicative-noise feedback Nash game.

    The common dynamics are ``x[k+1] = A[k]x[k] + B[k]u[k] + c[k] +
    sum_r (C[k,r]x[k] + D[k,r]u[k] + d[k,r]) xi[k,r]`` with zero-mean
    channels satisfying ``E[xi[k,r] xi[k,s]] = Gamma[k,r,s]``. Costs carry
    an explicit player axis, while dynamics and noise are common to all players.
    Every player minimizes its full-state expected quadratic-affine cost.
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

    deterministic_values, case_shape, horizon, n, m, players = _game_inputs(
        dynamics_matrices,
        control_matrices,
        state_costs,
        control_costs,
        terminal_state_costs,
        partition,
        dynamics_bias,
        state_control_cross,
        state_linear,
        control_linear,
        stage_constants,
        terminal_linear,
        terminal_constants,
    )
    state_noise, control_noise, gamma_raw, noise_offset = _noise_inputs(
        state_noise_matrices,
        control_noise_matrices,
        noise_covariances,
        noise_bias,
        case_shape,
        horizon,
        n,
        m,
    )
    dtype = jnp.result_type(
        *deterministic_values,
        state_noise,
        control_noise,
        gamma_raw,
        noise_offset,
        float,
    )
    (
        a,
        b,
        q_raw,
        r_raw,
        q_terminal_raw,
        c,
        cross,
        q_linear,
        r_linear,
        constants,
        q_terminal_linear,
        terminal_constant,
    ) = tuple(value.astype(dtype) for value in deterministic_values)
    state_noise = state_noise.astype(dtype)
    control_noise = control_noise.astype(dtype)
    gamma_raw = gamma_raw.astype(dtype)
    noise_offset = noise_offset.astype(dtype)
    if time_grid is None:
        time_grid = TimeGrid(
            jnp.arange(horizon + 1, dtype=dtype), time_id=f"{policy_id}:time"
        )
    elif not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid or None.")
    if time_grid.num_steps != horizon:
        raise ValueError(
            f"time_grid must contain {horizon + 1} times for this game horizon."
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
        _all_finite(q_terminal_raw, 3)
        & _all_finite(q_terminal_linear, 2)
        & _all_finite(terminal_constant, 1)
    )
    terminal_symmetric = jnp.all(terminal_symmetry <= symmetry_tolerance, axis=-1)
    terminal_status = jnp.where(
        ~terminal_finite,
        int(MultiplicativeLQFeedbackNashStatus.NONFINITE_INPUT),
        jnp.where(
            ~terminal_symmetric,
            int(MultiplicativeLQFeedbackNashStatus.NONSYMMETRIC_COST),
            int(MultiplicativeLQFeedbackNashStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    terminal_valid = terminal_status == int(MultiplicativeLQFeedbackNashStatus.SUCCESS)
    failed_stage = jnp.where(
        terminal_valid,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(horizon, dtype=jnp.int32),
    )
    carry_q_terminal = _case_where(
        terminal_valid, q_terminal, jnp.full_like(q_terminal, jnp.nan)
    )
    carry_terminal_linear = _case_where(
        terminal_valid,
        q_terminal_linear,
        jnp.full_like(q_terminal_linear, jnp.nan),
    )
    carry_terminal_constant = _case_where(
        terminal_valid,
        terminal_constant,
        jnp.full_like(terminal_constant, jnp.nan),
    )

    rank_relative = (
        float(m * jnp.finfo(dtype).eps) if relative_rank is None else relative_rank
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
    owner = jnp.asarray(partition.control_owner, dtype=jnp.int32)
    rows = jnp.arange(m, dtype=jnp.int32)

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
            z_next,
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

        b_player = b_t[..., None, :, :]
        b_transpose = jnp.swapaxes(b_t, -1, -2)[..., None, :, :]
        h = (
            r_t
            + b_transpose @ z_next @ b_player
            + ein.contract(
                "...rai,...pab,...sbj,...rs->...pij",
                control_noise_t,
                z_next,
                control_noise_t,
                gamma_t,
            )
        )
        w = (
            b_transpose @ z_next @ a_t[..., None, :, :]
            + jnp.swapaxes(cross_t, -1, -2)
            + ein.contract(
                "...rai,...pab,...sbj,...rs->...pij",
                control_noise_t,
                z_next,
                state_noise_t,
                gamma_t,
            )
        )
        affine_next = ein.contract("...pij,...j->...pi", z_next, c_t) + linear_next
        g = (
            r_t_linear
            + ein.contract("...ji,...pj->...pi", b_t, affine_next)
            + ein.contract(
                "...rai,...pab,...sb,...rs->...pi",
                control_noise_t,
                z_next,
                noise_offset_t,
                gamma_t,
            )
        )
        state_hessian = (
            q_t
            + jnp.swapaxes(a_t, -1, -2)[..., None, :, :] @ z_next @ a_t[..., None, :, :]
            + ein.contract(
                "...rai,...pab,...sbj,...rs->...pij",
                state_noise_t,
                z_next,
                state_noise_t,
                gamma_t,
            )
        )
        state_affine = (
            q_t_linear
            + ein.contract("...ji,...pj->...pi", a_t, affine_next)
            + ein.contract(
                "...rai,...pab,...sb,...rs->...pi",
                state_noise_t,
                z_next,
                noise_offset_t,
                gamma_t,
            )
        )
        delta = (
            d_t
            + constant_next
            + 0.5 * ein.contract("...i,...pij,...j->...p", c_t, z_next, c_t)
            + ein.contract("...pi,...i->...p", linear_next, c_t)
            + 0.5
            * ein.contract(
                "...ra,...pab,...sb,...rs->...p",
                noise_offset_t,
                z_next,
                noise_offset_t,
                gamma_t,
            )
        )

        coupled = h[..., owner, rows, :]
        feedback_rhs = w[..., owner, rows, :]
        affine_rhs = g[..., owner, rows]
        rhs = jnp.concatenate((feedback_rhs, affine_rhs[..., None]), axis=-1)
        own_minimum_eigenvalues = []
        own_symmetry_residuals = []
        for player, (start, stop) in enumerate(partition.control_slices):
            own = h[..., player, start:stop, start:stop]
            own_symmetry_residuals.append(_normalized_symmetry_residual(own))
            eigenvalues = jnp.linalg.eigvalsh(jax.lax.stop_gradient(_symmetric(own)))
            own_minimum_eigenvalues.append(jnp.min(eigenvalues, axis=-1))
        own_minimum = jnp.stack(own_minimum_eigenvalues, axis=-1)
        own_symmetry = jnp.stack(own_symmetry_residuals, axis=-1)

        solve_result = solve(
            LinearSystem(
                DenseLinearOperator(
                    coupled,
                    operator_id="control-games:multiplicative-lq-nash:lu",
                ),
                problem_id="control-games:multiplicative-lq-nash:stage",
            ),
            rhs,
            policy=linear_policy,
            rhs_layout=rhs_layout,
        )
        solved = solve_result.value
        feedback = -solved[..., :n]
        feedforward = -solved[..., n]

        svd_factorization = factorize(
            DenseLinearOperator(
                jax.lax.stop_gradient(coupled),
                operator_id="control-games:multiplicative-lq-nash:svd",
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
            jnp.asarray(jnp.inf, dtype=dtype),
        )

        feedback_player = feedback[..., None, :, :]
        feedback_transpose = jnp.swapaxes(feedback, -1, -2)[..., None, :, :]
        closed_loop = a_t + b_t @ feedback
        closed_bias = c_t + ein.contract("...ij,...j->...i", b_t, feedforward)
        closed_noise = state_noise_t + ein.contract(
            "...rij,...jk->...rik", control_noise_t, feedback
        )
        closed_noise_bias = noise_offset_t + ein.contract(
            "...rij,...j->...ri", control_noise_t, feedforward
        )
        trace_increment = 0.5 * ein.contract(
            "...ra,...pab,...sb,...rs->...p",
            closed_noise_bias,
            z_next,
            closed_noise_bias,
            gamma_t,
        )
        z_raw = (
            q_t
            + cross_t @ feedback_player
            + feedback_transpose @ jnp.swapaxes(cross_t, -1, -2)
            + feedback_transpose @ r_t @ feedback_player
            + jnp.swapaxes(closed_loop, -1, -2)[..., None, :, :]
            @ z_next
            @ closed_loop[..., None, :, :]
            + ein.contract(
                "...rai,...pab,...sbj,...rs->...pij",
                closed_noise,
                z_next,
                closed_noise,
                gamma_t,
            )
        )
        z_current = _symmetric(z_raw)
        linear_current = (
            q_t_linear
            + ein.contract("...pij,...j->...pi", cross_t, feedforward)
            + ein.contract(
                "...ji,...pj->...pi",
                feedback,
                ein.contract("...pij,...j->...pi", r_t, feedforward) + r_t_linear,
            )
            + ein.contract(
                "...ji,...pj->...pi",
                closed_loop,
                ein.contract("...pij,...j->...pi", z_next, closed_bias) + linear_next,
            )
            + ein.contract(
                "...rai,...pab,...sb,...rs->...pi",
                closed_noise,
                z_next,
                closed_noise_bias,
                gamma_t,
            )
        )
        constant_current = (
            d_t
            + constant_next
            + 0.5 * ein.contract("...i,...pij,...j->...p", feedforward, r_t, feedforward)
            + ein.contract("...pi,...i->...p", r_t_linear, feedforward)
            + 0.5
            * ein.contract("...i,...pij,...j->...p", closed_bias, z_next, closed_bias)
            + ein.contract("...pi,...i->...p", linear_next, closed_bias)
            + trace_increment
        )
        bellman_z = (
            state_hessian
            + jnp.swapaxes(w, -1, -2) @ feedback_player
            + feedback_transpose @ w
            + feedback_transpose @ h @ feedback_player
        )
        bellman_linear = (
            state_affine
            + ein.contract("...pji,...j->...pi", w, feedforward)
            + ein.contract(
                "...ji,...pj->...pi",
                feedback,
                ein.contract("...pij,...j->...pi", h, feedforward) + g,
            )
        )
        bellman_constant = (
            delta
            + 0.5 * ein.contract("...i,...pij,...j->...p", feedforward, h, feedforward)
            + ein.contract("...pi,...i->...p", g, feedforward)
        )

        stationarity_matrices = []
        stationarity_vectors = []
        stationarity_matrix_references = []
        stationarity_vector_references = []
        for player, (start, stop) in enumerate(partition.control_slices):
            h_owned = h[..., player, start:stop, :]
            w_owned = w[..., player, start:stop, :]
            g_owned = g[..., player, start:stop]
            h_feedback = h_owned @ feedback
            h_feedforward = ein.contract("...ij,...j->...i", h_owned, feedforward)
            stationarity_matrices.append(h_feedback + w_owned)
            stationarity_vectors.append(h_feedforward + g_owned)
            stationarity_matrix_references.extend((h_feedback, w_owned))
            stationarity_vector_references.extend((h_feedforward, g_owned))
        stationarity_matrix = jnp.concatenate(stationarity_matrices, axis=-2)
        stationarity_vector = jnp.concatenate(stationarity_vectors, axis=-1)
        stationarity_reference_matrix = jnp.concatenate(
            stationarity_matrix_references, axis=-2
        )
        stationarity_reference_vector = jnp.concatenate(
            stationarity_vector_references, axis=-1
        )
        stationarity_residual = _normalized_combined_residual(
            (stationarity_matrix, stationarity_vector),
            (stationarity_reference_matrix, stationarity_reference_vector),
            (2, 1),
        )
        bellman_residual = _normalized_combined_residual(
            (
                z_current - bellman_z,
                linear_current - bellman_linear,
                constant_current - bellman_constant,
            ),
            (bellman_z, bellman_linear, bellman_constant),
            (3, 2, 1),
        )
        value_symmetry = _normalized_symmetry_residual(z_raw)

        input_finite = (
            _all_finite(a_t, 2)
            & _all_finite(b_t, 2)
            & _all_finite(c_t, 1)
            & _all_finite(q_t, 3)
            & _all_finite(r_t, 3)
            & _all_finite(state_noise_t, 3)
            & _all_finite(control_noise_t, 3)
            & _all_finite(gamma_t, 2)
            & _all_finite(noise_offset_t, 2)
            & _all_finite(cross_t, 3)
            & _all_finite(q_t_linear, 2)
            & _all_finite(r_t_linear, 2)
            & _all_finite(d_t, 1)
        )
        input_symmetric = jnp.all(
            (q_symmetry_t <= symmetry_tolerance) & (r_symmetry_t <= symmetry_tolerance),
            axis=-1,
        )
        covariance_symmetric = covariance_symmetry_t <= covariance_tolerance
        covariance_psd = covariance_minimum_t >= -covariance_tolerance
        curvature_finite = jnp.all(jnp.isfinite(own_minimum), axis=-1) & jnp.all(
            jnp.isfinite(own_symmetry), axis=-1
        )
        curvature_valid = jnp.all(own_minimum > curvature_tolerance, axis=-1)
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
            & _all_finite(z_current, 3)
            & _all_finite(linear_current, 2)
            & _all_finite(constant_current, 1)
            & _all_finite(trace_increment, 1)
            & jnp.isfinite(stationarity_residual)
            & jnp.isfinite(bellman_residual)
            & jnp.all(jnp.isfinite(value_symmetry), axis=-1)
        )
        residual_valid = (
            (linear_relative <= tolerance)
            & (stationarity_residual <= tolerance)
            & (bellman_residual <= tolerance)
            & jnp.all(value_symmetry <= symmetry_tolerance, axis=-1)
            & jnp.all(own_symmetry <= symmetry_tolerance, axis=-1)
        )
        direct_status = jnp.where(
            ~input_finite,
            int(MultiplicativeLQFeedbackNashStatus.NONFINITE_INPUT),
            jnp.where(
                ~input_symmetric,
                int(MultiplicativeLQFeedbackNashStatus.NONSYMMETRIC_COST),
                jnp.where(
                    ~covariance_symmetric,
                    int(MultiplicativeLQFeedbackNashStatus.NOISE_COVARIANCE_NONSYMMETRIC),
                    jnp.where(
                        ~covariance_psd,
                        int(
                            MultiplicativeLQFeedbackNashStatus.NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE
                        ),
                        jnp.where(
                            ~curvature_finite | ~svd_finite,
                            int(MultiplicativeLQFeedbackNashStatus.NONFINITE_OUTPUT),
                            jnp.where(
                                ~curvature_valid,
                                int(
                                    MultiplicativeLQFeedbackNashStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE
                                ),
                                jnp.where(
                                    ~rank_valid,
                                    int(
                                        MultiplicativeLQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT
                                    ),
                                    jnp.where(
                                        ~condition_valid,
                                        int(
                                            MultiplicativeLQFeedbackNashStatus.CONDITION_LIMIT_REACHED
                                        ),
                                        jnp.where(
                                            ~linear_valid,
                                            int(
                                                MultiplicativeLQFeedbackNashStatus.LINEAR_SOLVE_FAILED
                                            ),
                                            jnp.where(
                                                ~output_finite,
                                                int(
                                                    MultiplicativeLQFeedbackNashStatus.NONFINITE_OUTPUT
                                                ),
                                                jnp.where(
                                                    ~residual_valid,
                                                    int(
                                                        MultiplicativeLQFeedbackNashStatus.RESIDUAL_TOO_LARGE
                                                    ),
                                                    int(
                                                        MultiplicativeLQFeedbackNashStatus.SUCCESS
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
            int(MultiplicativeLQFeedbackNashStatus.DEPENDENCY_FAILED),
        ).astype(jnp.int32)
        local_valid = direct_status == int(MultiplicativeLQFeedbackNashStatus.SUCCESS)
        stage_valid = continuation_valid & local_valid
        direct_failure = continuation_valid & ~local_valid
        next_failed_stage = jnp.where(direct_failure, stage_index, causal_stage)
        next_causal_status = jnp.where(direct_failure, direct_status, causal_status)
        next_z = _case_where(stage_valid, z_current, jnp.full_like(z_current, jnp.nan))
        next_linear = _case_where(
            stage_valid,
            linear_current,
            jnp.full_like(linear_current, jnp.nan),
        )
        next_constant = _case_where(
            stage_valid,
            constant_current,
            jnp.full_like(constant_current, jnp.nan),
        )
        output = (
            z_current,
            linear_current,
            constant_current,
            feedback,
            feedforward,
            trace_increment,
            stage_status,
            linear_status,
            diagnostic_available,
            own_symmetry,
            own_minimum,
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
            next_z,
            next_linear,
            next_constant,
            stage_valid,
            next_failed_stage,
            next_causal_status,
        ), output

    initial_carry = (
        carry_q_terminal,
        carry_terminal_linear,
        carry_terminal_constant,
        terminal_valid,
        failed_stage,
        terminal_status,
    )
    final_carry, outputs = jax.lax.scan(step, initial_carry, inputs, reverse=True)
    _, _, _, _, first_failed_stage, status = final_carry
    (
        z_stages,
        linear_stages,
        constant_stages,
        feedback,
        feedforward,
        trace_increments,
        stage_status,
        linear_status,
        diagnostic_available,
        own_symmetry,
        own_minimum,
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

    z_stages = jnp.moveaxis(z_stages, 0, -3)
    linear_stages = jnp.moveaxis(linear_stages, 0, -2)
    constant_stages = jnp.moveaxis(constant_stages, 0, -1)
    z_all = jnp.concatenate((z_stages, q_terminal[..., None, :, :]), axis=-3)
    linear_all = jnp.concatenate(
        (linear_stages, q_terminal_linear[..., None, :]), axis=-2
    )
    constant_all = jnp.concatenate(
        (constant_stages, terminal_constant[..., None]), axis=-1
    )
    feedback = jnp.moveaxis(feedback, 0, -3)
    feedforward = jnp.moveaxis(feedforward, 0, -2)
    trace_increments = jnp.moveaxis(trace_increments, 0, -1)
    stage_status = jnp.moveaxis(stage_status, 0, -1)
    linear_status = jnp.moveaxis(linear_status, 0, -2)
    diagnostic_available = jnp.moveaxis(diagnostic_available, 0, -1)
    own_symmetry = jnp.moveaxis(own_symmetry, 0, -1)
    own_minimum = jnp.moveaxis(own_minimum, 0, -1)
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
    maximum_value_symmetry = _nanmax(value_symmetry, (-2, -1))
    minimum_own = _nanmin(own_minimum, (-2, -1))
    maximum_condition_number = _nanmax(conditions, -1)
    result_finite = (
        _all_finite(feedback, 3)
        & _all_finite(feedforward, 2)
        & _all_finite(z_all, 4)
        & _all_finite(linear_all, 3)
        & _all_finite(constant_all, 2)
        & _all_finite(trace_increments, 2)
    )
    valid = status == int(MultiplicativeLQFeedbackNashStatus.SUCCESS)
    policy = AffineFeedbackPolicy(
        feedback,
        feedforward,
        time_grid=time_grid,
        state_size=n,
        case_shape=case_shape,
        policy_id=policy_id,
        _allow_nonfinite=True,
    )
    player_axis = len(case_shape)
    player_values = tuple(
        QuadraticValueFunction(
            jnp.take(z_all, player, axis=player_axis),
            jnp.take(linear_all, player, axis=player_axis),
            jnp.take(constant_all, player, axis=player_axis),
            time_grid=time_grid,
            case_shape=case_shape,
        )
        for player in range(players)
    )
    diagnostics = FiniteHorizonMultiplicativeLQFeedbackNashDiagnostics(
        stage_status=stage_status,
        terminal_status=terminal_status,
        linear_status=linear_status,
        diagnostic_available=diagnostic_available,
        state_cost_symmetry_residuals=q_symmetry,
        control_cost_symmetry_residuals=r_symmetry,
        terminal_cost_symmetry_residuals=terminal_symmetry,
        covariance_symmetry_residuals=covariance_symmetry,
        covariance_minimum_eigenvalues=covariance_minimum,
        own_control_symmetry_residuals=own_symmetry,
        own_control_minimum_eigenvalues=own_minimum,
        coupled_ranks=ranks,
        rank_cutoffs=rank_cutoffs,
        minimum_singular_values=minimum_singular,
        maximum_singular_values=maximum_singular,
        coupled_condition_numbers=conditions,
        linear_relative_residuals=linear_relative,
        stationarity_residuals=stationarity,
        bellman_residuals=bellman,
        value_symmetry_residuals=value_symmetry,
        maximum_stationarity_residual=maximum_stationarity,
        maximum_bellman_residual=maximum_bellman,
        maximum_value_symmetry_residual=maximum_value_symmetry,
        minimum_own_control_eigenvalue=minimum_own,
        maximum_coupled_condition_number=maximum_condition_number,
        first_failed_stage=first_failed_stage,
        finite=result_finite,
        valid=valid,
        status=status,
        method="backward-multiplicative-noise-feedback-nash",
        linear_backend="jax-dense",
        linear_method=DenseLU().name,
    )
    return FiniteHorizonMultiplicativeLQFeedbackNashResult(
        partition=partition,
        policy=policy,
        values=player_values,
        noise_covariances=gamma,
        trace_increments=trace_increments,
        diagnostics=diagnostics,
        valid=valid,
        status=status,
    )
