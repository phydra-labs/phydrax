#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-horizon additive-noise LQG feedback Nash games."""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import TimeGrid
from .._lqr import AffineFeedbackPolicy, QuadraticValueFunction
from ..stochastic._lqg import (
    _all_finite,
    _initial_distribution,
    _noise_inputs,
    _reverse_cumulative,
)
from ._layout import PlayerControlPartition
from ._linear_quadratic import (
    finite_horizon_lq_feedback_nash,
    FiniteHorizonLQFeedbackNashResult,
)


class LQGFeedbackNashStatus(IntEnum):
    """Stable validity codes for additive-noise LQG feedback Nash solves."""

    SUCCESS = 0
    DETERMINISTIC_NASH_SOLVE_FAILED = 1
    NONFINITE_OUTPUT = 2


class FiniteHorizonLQGFeedbackNashResult(StrictModule):
    """Certainty-equivalent Nash strategy and each player's corrected value.

    The stochastic process is common to all players. Players remain an explicit
    game axis in trace and cost evidence and are never folded into case axes.
    """

    deterministic_result: FiniteHorizonLQFeedbackNashResult
    values: tuple[QuadraticValueFunction, ...]
    trace_increments: Array
    process_covariances: Array
    value_constant_corrections: Array
    initial_mean: Array
    initial_covariance: Array
    initial_covariance_cost: Array
    initial_expected_cost: Array
    covariance_symmetry_residuals: Array
    covariance_minimum_eigenvalues: Array
    initial_covariance_symmetry_residual: Array
    initial_covariance_minimum_eigenvalue: Array
    covariance_finite: Array
    initial_covariance_finite: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)

    @property
    def partition(self) -> PlayerControlPartition:
        return self.deterministic_result.partition

    @property
    def policy(self) -> AffineFeedbackPolicy:
        return self.deterministic_result.policy

    @property
    def feedback_gain(self) -> Array:
        return self.deterministic_result.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.deterministic_result.feedforward


def finite_horizon_lqg_feedback_nash(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_costs: ArrayLike,
    partition: PlayerControlPartition,
    /,
    *,
    process_noise_factors: ArrayLike,
    process_noise_covariances: ArrayLike,
    initial_mean: ArrayLike | None = None,
    initial_covariance: ArrayLike | None = None,
    dynamics_bias: ArrayLike | None = None,
    state_control_cross: ArrayLike | None = None,
    state_linear: ArrayLike | None = None,
    control_linear: ArrayLike | None = None,
    stage_constants: ArrayLike | None = None,
    terminal_linear: ArrayLike | None = None,
    terminal_constants: ArrayLike = 0.0,
    time_grid: TimeGrid | None = None,
    policy_id: str = "game:lqg-feedback-nash",
    tolerance: float = 1e-9,
    symmetry_tolerance: float = 1e-10,
    curvature_tolerance: float = 1e-10,
    rank_relative_tolerance: float | None = None,
    rank_absolute_tolerance: float | None = None,
    maximum_condition: float | None = None,
    covariance_tolerance: float = 0.0,
) -> FiniteHorizonLQGFeedbackNashResult:
    """Solve an exact full-state LQG feedback Nash game with additive noise.

    Dynamics are ``x[k+1] = A[k]x[k] + B[k]u[k] + c[k] + G[k]w[k]`` with
    zero-mean, action-independent ``w[k]`` of covariance ``Ω[k]``. The common
    additive noise changes each player's value constants but not the coupled
    deterministic feedback-Nash equations.
    """
    (
        factors,
        driving_covariances,
        process_covariances,
        covariance_symmetry,
        covariance_minimum,
        covariance_finite,
        case_shape,
        state_size,
    ) = _noise_inputs(
        dynamics_matrices,
        process_noise_factors,
        process_noise_covariances,
        covariance_tolerance,
    )
    dtype = jnp.result_type(
        jnp.asarray(dynamics_matrices),
        factors,
        driving_covariances,
        float,
    )
    (
        resolved_initial_mean,
        resolved_initial_covariance,
        initial_covariance_symmetry,
        initial_covariance_minimum,
        initial_covariance_finite,
    ) = _initial_distribution(
        initial_mean,
        initial_covariance,
        case_shape,
        state_size,
        dtype,
        covariance_tolerance,
    )

    deterministic_result = finite_horizon_lq_feedback_nash(
        dynamics_matrices,
        control_matrices,
        state_costs,
        control_costs,
        terminal_state_costs,
        partition,
        dynamics_bias=dynamics_bias,
        state_control_cross=state_control_cross,
        state_linear=state_linear,
        control_linear=control_linear,
        stage_constants=stage_constants,
        terminal_linear=terminal_linear,
        terminal_constants=terminal_constants,
        time_grid=time_grid,
        policy_id=policy_id,
        tolerance=tolerance,
        symmetry_tolerance=symmetry_tolerance,
        curvature_tolerance=curvature_tolerance,
        rank_relative_tolerance=rank_relative_tolerance,
        rank_absolute_tolerance=rank_absolute_tolerance,
        maximum_condition=maximum_condition,
    )
    player_axis = len(case_shape)
    players = len(deterministic_result.values)
    matrices = jnp.stack(
        tuple(value.matrices for value in deterministic_result.values),
        axis=player_axis,
    )
    linear = jnp.stack(
        tuple(value.linear for value in deterministic_result.values),
        axis=player_axis,
    )
    constants = jnp.stack(
        tuple(value.constants for value in deterministic_result.values),
        axis=player_axis,
    )
    trace_increments = 0.5 * ein.contract(
        "...ptij,...tji->...pt",
        matrices[..., 1:, :, :],
        process_covariances,
    )
    cumulative = _reverse_cumulative(trace_increments)
    value_constant_corrections = jnp.concatenate(
        (
            cumulative,
            jnp.zeros(case_shape + (players, 1), dtype=cumulative.dtype),
        ),
        axis=-1,
    )
    corrected_constants = constants + value_constant_corrections
    values = tuple(
        QuadraticValueFunction(
            deterministic_value.matrices,
            deterministic_value.linear,
            jnp.take(corrected_constants, player, axis=player_axis),
            time_grid=deterministic_value.time_grid,
            case_shape=case_shape,
        )
        for player, deterministic_value in enumerate(deterministic_result.values)
    )

    initial_matrices = matrices[..., 0, :, :]
    initial_linear = linear[..., 0, :]
    initial_covariance_cost = 0.5 * ein.contract(
        "...pij,...ji->...p",
        initial_matrices,
        resolved_initial_covariance,
    )
    matrix_times_mean = ein.contract(
        "...pij,...j->...pi",
        initial_matrices,
        resolved_initial_mean,
    )
    initial_expected_cost = (
        0.5
        * ein.contract(
            "...pi,...i->...p",
            matrix_times_mean,
            resolved_initial_mean,
        )
        + ein.contract(
            "...pi,...i->...p",
            initial_linear,
            resolved_initial_mean,
        )
        + corrected_constants[..., 0]
        + initial_covariance_cost
    )
    output_finite = (
        covariance_finite
        & initial_covariance_finite
        & _all_finite(trace_increments, 2)
        & _all_finite(value_constant_corrections, 2)
        & _all_finite(corrected_constants, 2)
        & _all_finite(initial_covariance_cost, 1)
        & _all_finite(initial_expected_cost, 1)
    )
    valid = deterministic_result.valid & output_finite
    status = jnp.where(
        ~deterministic_result.valid,
        int(LQGFeedbackNashStatus.DETERMINISTIC_NASH_SOLVE_FAILED),
        jnp.where(
            output_finite,
            int(LQGFeedbackNashStatus.SUCCESS),
            int(LQGFeedbackNashStatus.NONFINITE_OUTPUT),
        ),
    ).astype(jnp.int32)
    return FiniteHorizonLQGFeedbackNashResult(
        deterministic_result=deterministic_result,
        values=values,
        trace_increments=trace_increments,
        process_covariances=process_covariances,
        value_constant_corrections=value_constant_corrections,
        initial_mean=resolved_initial_mean,
        initial_covariance=resolved_initial_covariance,
        initial_covariance_cost=initial_covariance_cost,
        initial_expected_cost=initial_expected_cost,
        covariance_symmetry_residuals=covariance_symmetry,
        covariance_minimum_eigenvalues=covariance_minimum,
        initial_covariance_symmetry_residual=initial_covariance_symmetry,
        initial_covariance_minimum_eigenvalue=initial_covariance_minimum,
        covariance_finite=covariance_finite,
        initial_covariance_finite=initial_covariance_finite,
        valid=valid,
        status=status,
        method="certainty-equivalent-additive-noise-feedback-nash",
    )
