#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-horizon affine linear-quadratic feedback Nash games."""

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
from ._layout import PlayerControlPartition


class LQFeedbackNashStatus(IntEnum):
    """Stable validity codes for finite-horizon LQ feedback Nash solves."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    NONSYMMETRIC_COST = 2
    OWN_CURVATURE_NOT_POSITIVE_DEFINITE = 3
    COUPLED_SYSTEM_RANK_DEFICIENT = 4
    CONDITION_LIMIT_REACHED = 5
    LINEAR_SOLVE_FAILED = 6
    NONFINITE_OUTPUT = 7
    RESIDUAL_TOO_LARGE = 8
    DEPENDENCY_FAILED = 9


class FiniteHorizonLQFeedbackNashDiagnostics(StrictModule):
    """Per-stage equilibrium, curvature, rank, and failure evidence."""

    stage_status: Array
    terminal_status: Array
    linear_status: Array
    diagnostic_available: Array
    state_cost_symmetry_residuals: Array
    control_cost_symmetry_residuals: Array
    terminal_cost_symmetry_residuals: Array
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


class FiniteHorizonLQFeedbackNashResult(StrictModule):
    """Joint affine strategy, player values, and feedback-Nash evidence."""

    partition: PlayerControlPartition
    policy: AffineFeedbackPolicy
    values: tuple[QuadraticValueFunction, ...]
    diagnostics: FiniteHorizonLQFeedbackNashDiagnostics
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


def _normalized_symmetry_residual(matrix: Array, /) -> Array:
    difference = matrix - jnp.swapaxes(matrix, -1, -2)
    numerator = jnp.sqrt(jnp.sum(jnp.square(difference), axis=(-2, -1)))
    scale = jnp.sqrt(jnp.sum(jnp.square(matrix), axis=(-2, -1)))
    return numerator / jnp.maximum(jnp.asarray(1.0, dtype=matrix.dtype), scale)


def _symmetric(matrix: Array, /) -> Array:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


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
        residuals,
        references,
        payload_ranks,
        strict=True,
    ):
        axes = tuple(range(residual.ndim - rank, residual.ndim))
        residual_square = residual_square + (
            jnp.sum(jnp.square(residual), axis=axes) if axes else jnp.square(residual)
        )
        reference_square = reference_square + (
            jnp.sum(jnp.square(reference), axis=axes) if axes else jnp.square(reference)
        )
    return jnp.sqrt(residual_square) / (1.0 + jnp.sqrt(reference_square))


def _nanmax(value: Array, axis, /) -> Array:
    available = ~jnp.isnan(value)
    maximum = jnp.max(jnp.where(available, value, -jnp.inf), axis=axis)
    any_available = jnp.any(available, axis=axis)
    return jnp.where(any_available, maximum, jnp.nan)


def _nanmin(value: Array, axis, /) -> Array:
    available = ~jnp.isnan(value)
    minimum = jnp.min(jnp.where(available, value, jnp.inf), axis=axis)
    any_available = jnp.any(available, axis=axis)
    return jnp.where(any_available, minimum, jnp.nan)


def _game_inputs(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_costs: ArrayLike,
    partition: PlayerControlPartition,
    dynamics_bias: ArrayLike | None,
    state_control_cross: ArrayLike | None,
    state_linear: ArrayLike | None,
    control_linear: ArrayLike | None,
    stage_constants: ArrayLike | None,
    terminal_linear: ArrayLike | None,
    terminal_constants: ArrayLike,
    /,
):
    if not isinstance(partition, PlayerControlPartition):
        raise TypeError("partition must be a PlayerControlPartition.")
    a = _real_array(dynamics_matrices, "dynamics_matrices")
    if a.ndim < 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            "dynamics_matrices must have shape case_shape + (horizon, n, n)."
        )
    case_shape = tuple(a.shape[:-3])
    horizon = int(a.shape[-3])
    n = int(a.shape[-1])
    if horizon < 1:
        raise ValueError("Finite-horizon games require at least one stage.")

    b = _real_array(control_matrices, "control_matrices")
    if b.ndim < 3 or tuple(b.shape[:-3]) != case_shape or b.shape[-3:-1] != (horizon, n):
        raise ValueError(
            "control_matrices must have shape case_shape + (horizon, n, m); "
            f"got {b.shape}."
        )
    m = int(b.shape[-1])
    if partition.joint_control_size != m:
        raise ValueError(
            "partition joint control size must match control_matrices; "
            f"got {partition.joint_control_size} and {m}."
        )
    players = partition.num_players

    a = _require_shape(a, case_shape + (horizon, n, n), "dynamics_matrices")
    b = _require_shape(b, case_shape + (horizon, n, m), "control_matrices")
    q = _require_shape(
        _real_array(state_costs, "state_costs"),
        case_shape + (players, horizon, n, n),
        "state_costs",
    )
    r = _require_shape(
        _real_array(control_costs, "control_costs"),
        case_shape + (players, horizon, m, m),
        "control_costs",
    )
    q_terminal = _require_shape(
        _real_array(terminal_state_costs, "terminal_state_costs"),
        case_shape + (players, n, n),
        "terminal_state_costs",
    )

    c = (
        jnp.zeros(case_shape + (horizon, n), dtype=a.dtype)
        if dynamics_bias is None
        else _require_shape(
            _real_array(dynamics_bias, "dynamics_bias"),
            case_shape + (horizon, n),
            "dynamics_bias",
        )
    )
    cross = (
        jnp.zeros(case_shape + (players, horizon, n, m), dtype=q.dtype)
        if state_control_cross is None
        else _require_shape(
            _real_array(state_control_cross, "state_control_cross"),
            case_shape + (players, horizon, n, m),
            "state_control_cross",
        )
    )
    q_linear = (
        jnp.zeros(case_shape + (players, horizon, n), dtype=q.dtype)
        if state_linear is None
        else _require_shape(
            _real_array(state_linear, "state_linear"),
            case_shape + (players, horizon, n),
            "state_linear",
        )
    )
    r_linear = (
        jnp.zeros(case_shape + (players, horizon, m), dtype=r.dtype)
        if control_linear is None
        else _require_shape(
            _real_array(control_linear, "control_linear"),
            case_shape + (players, horizon, m),
            "control_linear",
        )
    )
    constants = (
        jnp.zeros(case_shape + (players, horizon), dtype=q.dtype)
        if stage_constants is None
        else _require_shape(
            _real_array(stage_constants, "stage_constants"),
            case_shape + (players, horizon),
            "stage_constants",
        )
    )
    q_terminal_linear = (
        jnp.zeros(case_shape + (players, n), dtype=q_terminal.dtype)
        if terminal_linear is None
        else _require_shape(
            _real_array(terminal_linear, "terminal_linear"),
            case_shape + (players, n),
            "terminal_linear",
        )
    )
    terminal_constant_value = _real_array(terminal_constants, "terminal_constants")
    if terminal_constant_value.shape == ():
        terminal_constant_value = jnp.broadcast_to(
            terminal_constant_value,
            case_shape + (players,),
        )
    terminal_constant = _require_shape(
        terminal_constant_value,
        case_shape + (players,),
        "terminal_constants",
    )
    values = (
        a,
        b,
        q,
        r,
        q_terminal,
        c,
        cross,
        q_linear,
        r_linear,
        constants,
        q_terminal_linear,
        terminal_constant,
    )
    dtype = jnp.result_type(*values, float)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise TypeError("Finite-horizon games require real-valued arrays.")
    return (
        tuple(value.astype(dtype) for value in values),
        case_shape,
        horizon,
        n,
        m,
        players,
    )


def finite_horizon_lq_feedback_nash(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_costs: ArrayLike,
    partition: PlayerControlPartition,
    /,
    *,
    dynamics_bias: ArrayLike | None = None,
    state_control_cross: ArrayLike | None = None,
    state_linear: ArrayLike | None = None,
    control_linear: ArrayLike | None = None,
    stage_constants: ArrayLike | None = None,
    terminal_linear: ArrayLike | None = None,
    terminal_constants: ArrayLike = 0.0,
    time_grid: TimeGrid | None = None,
    policy_id: str = "game:lq-feedback-nash",
    tolerance: float = 1e-9,
    symmetry_tolerance: float = 1e-10,
    curvature_tolerance: float = 1e-10,
    rank_relative_tolerance: float | None = None,
    rank_absolute_tolerance: float | None = None,
    maximum_condition: float | None = None,
) -> FiniteHorizonLQFeedbackNashResult:
    """Solve a finite-horizon affine LQ full-state feedback Nash game.

    Every player minimizes a quadratic-affine cost over the same affine dynamics.
    Player costs may depend on the full joint control. All stage arrays carry an
    explicit time axis and all cost arrays carry a player axis immediately before
    that time axis. No time or player broadcasting occurs.
    """
    scalar_parameters = (
        ("tolerance", tolerance, False),
        ("symmetry_tolerance", symmetry_tolerance, True),
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

    values, case_shape, horizon, n, m, players = _game_inputs(
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
    ) = values
    if time_grid is None:
        time_grid = TimeGrid(
            jnp.arange(horizon + 1, dtype=a.dtype),
            time_id=f"{policy_id}:time",
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
    q = _symmetric(q_raw)
    r = _symmetric(r_raw)
    q_terminal = _symmetric(q_terminal_raw)

    terminal_finite = (
        _all_finite(q_terminal_raw, 3)
        & _all_finite(q_terminal_linear, 2)
        & _all_finite(terminal_constant, 1)
    )
    terminal_symmetric = jnp.all(
        terminal_symmetry <= symmetry_tolerance,
        axis=-1,
    )
    terminal_status = jnp.where(
        ~terminal_finite,
        int(LQFeedbackNashStatus.NONFINITE_INPUT),
        jnp.where(
            ~terminal_symmetric,
            int(LQFeedbackNashStatus.NONSYMMETRIC_COST),
            int(LQFeedbackNashStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    terminal_valid = terminal_status == int(LQFeedbackNashStatus.SUCCESS)
    failed_stage = jnp.where(
        terminal_valid,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(horizon, dtype=jnp.int32),
    )
    nan_q_terminal = jnp.full_like(q_terminal, jnp.nan)
    nan_terminal_linear = jnp.full_like(q_terminal_linear, jnp.nan)
    nan_terminal_constant = jnp.full_like(terminal_constant, jnp.nan)
    carry_q_terminal = _case_where(terminal_valid, q_terminal, nan_q_terminal)
    carry_terminal_linear = _case_where(
        terminal_valid,
        q_terminal_linear,
        nan_terminal_linear,
    )
    carry_terminal_constant = _case_where(
        terminal_valid,
        terminal_constant,
        nan_terminal_constant,
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
        to_time_major(c, 1),
        to_time_major(cross, 2),
        to_time_major(q_linear, 1),
        to_time_major(r_linear, 1),
        to_time_major(constants, 0),
        to_time_major(q_symmetry, 0),
        to_time_major(r_symmetry, 0),
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
            c_t,
            cross_t,
            q_t_linear,
            r_t_linear,
            d_t,
            q_symmetry_t,
            r_symmetry_t,
        ) = stage

        b_player = b_t[..., None, :, :]
        z_b = z_next @ b_player
        b_transpose = jnp.swapaxes(b_t, -1, -2)[..., None, :, :]
        h = r_t + b_transpose @ z_b
        w = b_transpose @ z_next @ a_t[..., None, :, :] + jnp.swapaxes(
            cross_t,
            -1,
            -2,
        )
        affine_next = ein.contract("...pij,...j->...pi", z_next, c_t) + linear_next
        g = r_t_linear + ein.contract("...ji,...pj->...pi", b_t, affine_next)

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
                DenseLinearOperator(coupled, operator_id="control-games:lq-nash:lu"),
                problem_id="control-games:lq-nash:stage",
            ),
            rhs,
            policy=linear_policy,
            rhs_layout=rhs_layout,
        )
        solved = solve_result.value
        p = solved[..., :n]
        alpha = solved[..., n]
        feedback = -p
        feedforward = -alpha

        diagnostic_coupled = jax.lax.stop_gradient(coupled)
        svd_factorization = factorize(
            DenseLinearOperator(
                diagnostic_coupled,
                operator_id="control-games:lq-nash:svd",
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

        closed_loop = a_t - b_t @ p
        closed_bias = c_t - ein.contract("...ij,...j->...i", b_t, alpha)
        z_raw = (
            q_t
            - cross_t @ p[..., None, :, :]
            - jnp.swapaxes(p, -1, -2)[..., None, :, :]
            @ jnp.swapaxes(
                cross_t,
                -1,
                -2,
            )
            + jnp.swapaxes(p, -1, -2)[..., None, :, :] @ r_t @ p[..., None, :, :]
            + jnp.swapaxes(closed_loop, -1, -2)[..., None, :, :]
            @ z_next
            @ closed_loop[..., None, :, :]
        )
        z_current = _symmetric(z_raw)
        linear_current = (
            q_t_linear
            - ein.contract("...pij,...j->...pi", cross_t, alpha)
            + ein.contract(
                "...ji,...pj->...pi",
                p,
                ein.contract("...pij,...j->...pi", r_t, alpha) - r_t_linear,
            )
            + ein.contract(
                "...ji,...pj->...pi",
                closed_loop,
                ein.contract("...pij,...j->...pi", z_next, closed_bias) + linear_next,
            )
        )
        constant_current = (
            d_t
            + constant_next
            + 0.5
            * ein.contract("...i,...pij,...j->...p", closed_bias, z_next, closed_bias)
            + ein.contract("...pi,...i->...p", linear_next, closed_bias)
            + 0.5 * ein.contract("...i,...pij,...j->...p", alpha, r_t, alpha)
            - ein.contract("...pi,...i->...p", r_t_linear, alpha)
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
            stationarity_matrix_references,
            axis=-2,
        )
        stationarity_reference_vector = jnp.concatenate(
            stationarity_vector_references,
            axis=-1,
        )
        stationarity_residual = _normalized_combined_residual(
            (stationarity_matrix, stationarity_vector),
            (stationarity_reference_matrix, stationarity_reference_vector),
            (2, 1),
        )

        g_state = (
            q_t
            + jnp.swapaxes(a_t, -1, -2)[..., None, :, :] @ z_next @ a_t[..., None, :, :]
        )
        a_state = q_t_linear + ein.contract(
            "...ji,...pj->...pi",
            a_t,
            affine_next,
        )
        delta = (
            d_t
            + constant_next
            + 0.5 * ein.contract("...i,...pij,...j->...p", c_t, z_next, c_t)
            + ein.contract("...pi,...i->...p", linear_next, c_t)
        )
        feedback_player = feedback[..., None, :, :]
        bellman_z = (
            g_state
            + jnp.swapaxes(w, -1, -2) @ feedback_player
            + jnp.swapaxes(feedback, -1, -2)[..., None, :, :] @ w
            + jnp.swapaxes(feedback, -1, -2)[..., None, :, :] @ h @ feedback_player
        )
        bellman_linear = (
            a_state
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
            & _all_finite(cross_t, 3)
            & _all_finite(q_t_linear, 2)
            & _all_finite(r_t_linear, 2)
            & _all_finite(d_t, 1)
        )
        input_symmetric = jnp.all(
            (q_symmetry_t <= symmetry_tolerance) & (r_symmetry_t <= symmetry_tolerance),
            axis=-1,
        )
        curvature_finite = jnp.all(jnp.isfinite(own_minimum), axis=-1) & jnp.all(
            jnp.isfinite(own_symmetry),
            axis=-1,
        )
        curvature_valid = jnp.all(own_minimum > curvature_tolerance, axis=-1)
        svd_finite = jnp.all(jnp.isfinite(singular_values), axis=-1)
        diagnostic_available = continuation_valid & input_finite & svd_finite
        rank_reported = jnp.where(diagnostic_available, rank, -1)
        cutoff_reported = jnp.where(diagnostic_available, rank_cutoff, jnp.nan)
        minimum_singular_reported = jnp.where(
            diagnostic_available,
            minimum_singular,
            jnp.nan,
        )
        maximum_singular_reported = jnp.where(
            diagnostic_available,
            maximum_singular,
            jnp.nan,
        )
        condition_reported = jnp.where(diagnostic_available, condition, jnp.nan)
        rank_valid = rank == m
        condition_valid = (
            jnp.ones_like(condition, dtype=bool)
            if condition_limit is None
            else condition <= condition_limit
        )
        linear_status = solve_result.status.astype(jnp.int32)
        linear_valid = jnp.all(
            linear_status == int(LinearSolveStatus.SUCCESS),
            axis=-1,
        )
        linear_relative = jnp.max(solve_result.diagnostics.relative_residual, axis=-1)
        output_finite = (
            _all_finite(feedback, 2)
            & _all_finite(feedforward, 1)
            & _all_finite(z_current, 3)
            & _all_finite(linear_current, 2)
            & _all_finite(constant_current, 1)
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
            int(LQFeedbackNashStatus.NONFINITE_INPUT),
            jnp.where(
                ~input_symmetric,
                int(LQFeedbackNashStatus.NONSYMMETRIC_COST),
                jnp.where(
                    ~curvature_finite | ~svd_finite,
                    int(LQFeedbackNashStatus.NONFINITE_OUTPUT),
                    jnp.where(
                        ~curvature_valid,
                        int(LQFeedbackNashStatus.OWN_CURVATURE_NOT_POSITIVE_DEFINITE),
                        jnp.where(
                            ~rank_valid,
                            int(LQFeedbackNashStatus.COUPLED_SYSTEM_RANK_DEFICIENT),
                            jnp.where(
                                ~condition_valid,
                                int(LQFeedbackNashStatus.CONDITION_LIMIT_REACHED),
                                jnp.where(
                                    ~linear_valid,
                                    int(LQFeedbackNashStatus.LINEAR_SOLVE_FAILED),
                                    jnp.where(
                                        ~output_finite,
                                        int(LQFeedbackNashStatus.NONFINITE_OUTPUT),
                                        jnp.where(
                                            ~residual_valid,
                                            int(LQFeedbackNashStatus.RESIDUAL_TOO_LARGE),
                                            int(LQFeedbackNashStatus.SUCCESS),
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
            int(LQFeedbackNashStatus.DEPENDENCY_FAILED),
        ).astype(jnp.int32)
        local_valid = direct_status == int(LQFeedbackNashStatus.SUCCESS)
        stage_valid = continuation_valid & local_valid
        direct_failure = continuation_valid & ~local_valid
        next_failed_stage = jnp.where(direct_failure, stage_index, causal_stage)
        next_causal_status = jnp.where(direct_failure, direct_status, causal_status)

        nan_z = jnp.full_like(z_current, jnp.nan)
        nan_linear = jnp.full_like(linear_current, jnp.nan)
        nan_constant = jnp.full_like(constant_current, jnp.nan)
        next_z = _case_where(stage_valid, z_current, nan_z)
        next_linear = _case_where(stage_valid, linear_current, nan_linear)
        next_constant = _case_where(stage_valid, constant_current, nan_constant)
        output = (
            z_current,
            linear_current,
            constant_current,
            feedback,
            feedforward,
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
        (linear_stages, q_terminal_linear[..., None, :]),
        axis=-2,
    )
    constant_all = jnp.concatenate(
        (constant_stages, terminal_constant[..., None]),
        axis=-1,
    )
    feedback = jnp.moveaxis(feedback, 0, -3)
    feedforward = jnp.moveaxis(feedforward, 0, -2)
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
    )
    valid = status == int(LQFeedbackNashStatus.SUCCESS)

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
    diagnostics = FiniteHorizonLQFeedbackNashDiagnostics(
        stage_status=stage_status,
        terminal_status=terminal_status,
        linear_status=linear_status,
        diagnostic_available=diagnostic_available,
        state_cost_symmetry_residuals=q_symmetry,
        control_cost_symmetry_residuals=r_symmetry,
        terminal_cost_symmetry_residuals=terminal_symmetry,
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
        method="backward-feedback-nash",
        linear_backend="jax-dense",
        linear_method=DenseLU().name,
    )
    return FiniteHorizonLQFeedbackNashResult(
        partition=partition,
        policy=policy,
        values=player_values,
        diagnostics=diagnostics,
        valid=valid,
        status=status,
    )
