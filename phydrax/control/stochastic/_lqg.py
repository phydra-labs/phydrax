#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-horizon additive-noise LQG state feedback."""

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import TimeGrid
from .._lqr import (
    AffineFeedbackPolicy,
    finite_horizon_lqr,
    FiniteHorizonLQRResult,
    QuadraticValueFunction,
)
from .._riccati import _error_if, _require_shape


class LQGStateFeedbackStatus(IntEnum):
    """Stable validity codes for finite-horizon additive-noise LQG solves."""

    SUCCESS = 0
    DETERMINISTIC_SOLVE_FAILED = 1
    NONFINITE_OUTPUT = 2


class FiniteHorizonLQGStateFeedbackResult(StrictModule):
    """Certainty-equivalent feedback and exact additive-noise value evidence.

    ``process_covariances`` contains ``G[k] Ω[k] G[k]ᵀ``. The covariance
    spectral evidence refers to the supplied driving covariances ``Ω[k]``.
    ``initial_expected_cost`` includes both future process noise and the
    optional Gaussian initial covariance.
    """

    deterministic_result: FiniteHorizonLQRResult
    value: QuadraticValueFunction
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
    def policy(self) -> AffineFeedbackPolicy:
        return self.deterministic_result.policy

    @property
    def feedback_gain(self) -> Array:
        return self.deterministic_result.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.deterministic_result.feedforward


def _all_finite(value: Array, payload_rank: int, /) -> Array:
    axes = tuple(range(value.ndim - payload_rank, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes) if axes else jnp.isfinite(value)


def _covariance_evidence(
    covariance: Array,
    name: str,
    tolerance: float,
    /,
) -> tuple[Array, Array, Array]:
    symmetry = jnp.max(
        jnp.abs(covariance - jnp.swapaxes(covariance, -1, -2)),
        axis=(-2, -1),
    )
    covariance = _error_if(
        covariance,
        jnp.any(symmetry > tolerance),
        f"{name} must be symmetric within covariance_tolerance.",
    )
    minimum = jnp.min(jnp.linalg.eigvalsh(covariance), axis=-1)
    covariance = _error_if(
        covariance,
        jnp.any(minimum < -tolerance),
        f"{name} must be positive semidefinite.",
    )
    return covariance, symmetry, minimum


def _noise_inputs(
    dynamics_matrices: ArrayLike,
    process_noise_factors: ArrayLike,
    process_noise_covariances: ArrayLike,
    covariance_tolerance: float,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    tuple[int, ...],
    int,
]:
    tolerance = float(covariance_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("covariance_tolerance must be finite and non-negative.")

    dynamics = jnp.asarray(dynamics_matrices)
    if dynamics.ndim < 3 or dynamics.shape[-1] != dynamics.shape[-2]:
        raise ValueError(
            "dynamics_matrices must have shape case_shape + (horizon, n, n)."
        )
    case_shape = tuple(dynamics.shape[:-3])
    horizon = int(dynamics.shape[-3])
    state_size = int(dynamics.shape[-1])

    factors = jnp.asarray(process_noise_factors)
    expected_rank = len(case_shape) + 3
    if (
        factors.ndim != expected_rank
        or tuple(factors.shape[: len(case_shape)]) != case_shape
        or factors.shape[-3] != horizon
        or factors.shape[-2] != state_size
    ):
        raise ValueError(
            "process_noise_factors must have shape case_shape + "
            f"(horizon, n, noise_size); got {factors.shape}."
        )
    noise_size = int(factors.shape[-1])
    if noise_size < 1:
        raise ValueError("process_noise_factors must have a positive noise_size.")
    factors = _require_shape(
        factors,
        case_shape + (horizon, state_size, noise_size),
        "process_noise_factors",
    )
    driving_covariances = _require_shape(
        process_noise_covariances,
        case_shape + (horizon, noise_size, noise_size),
        "process_noise_covariances",
    )
    (
        driving_covariances,
        symmetry_residuals,
        minimum_eigenvalues,
    ) = _covariance_evidence(
        driving_covariances,
        "process_noise_covariances",
        tolerance,
    )
    process_covariances = factors @ driving_covariances @ jnp.swapaxes(factors, -1, -2)
    process_covariances = _error_if(
        process_covariances,
        jnp.any(~jnp.isfinite(process_covariances)),
        "The implied process_covariances must contain only finite values.",
    )
    covariance_finite = (
        _all_finite(driving_covariances, 3)
        & _all_finite(factors, 3)
        & _all_finite(process_covariances, 3)
    )
    return (
        factors,
        driving_covariances,
        process_covariances,
        symmetry_residuals,
        minimum_eigenvalues,
        covariance_finite,
        case_shape,
        state_size,
    )


def _initial_distribution(
    initial_mean: ArrayLike | None,
    initial_covariance: ArrayLike | None,
    case_shape: tuple[int, ...],
    state_size: int,
    dtype,
    covariance_tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    mean = (
        jnp.zeros(case_shape + (state_size,), dtype=dtype)
        if initial_mean is None
        else _require_shape(
            initial_mean,
            case_shape + (state_size,),
            "initial_mean",
        ).astype(dtype)
    )
    covariance = (
        jnp.zeros(case_shape + (state_size, state_size), dtype=dtype)
        if initial_covariance is None
        else _require_shape(
            initial_covariance,
            case_shape + (state_size, state_size),
            "initial_covariance",
        ).astype(dtype)
    )
    covariance, symmetry, minimum = _covariance_evidence(
        covariance,
        "initial_covariance",
        float(covariance_tolerance),
    )
    return mean, covariance, symmetry, minimum, _all_finite(covariance, 2)


def _reverse_cumulative(increments: Array, /) -> Array:
    return jnp.flip(
        jnp.cumsum(jnp.flip(increments, axis=-1), axis=-1),
        axis=-1,
    )


def _initial_cost_evidence(
    matrices: Array,
    linear: Array,
    constants: Array,
    initial_mean: Array,
    initial_covariance: Array,
    /,
) -> tuple[Array, Array]:
    matrix = matrices[..., 0, :, :]
    vector = linear[..., 0, :]
    constant = constants[..., 0]
    covariance_cost = 0.5 * ein.contract(
        "...ij,...ji->...",
        matrix,
        initial_covariance,
    )
    expected_cost = (
        0.5
        * ein.contract(
            "...i,...ij,...j->...",
            initial_mean,
            matrix,
            initial_mean,
        )
        + ein.contract("...i,...i->...", vector, initial_mean)
        + constant
        + covariance_cost
    )
    return covariance_cost, expected_cost


def finite_horizon_lqg_state_feedback(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_cost: ArrayLike,
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
    terminal_constant: ArrayLike = 0.0,
    time_grid: TimeGrid | None = None,
    policy_id: str = "lqg:finite-horizon-state-feedback",
    tolerance: float = 1e-9,
    cost_tolerance: float = 1e-10,
    covariance_tolerance: float = 0.0,
) -> FiniteHorizonLQGStateFeedbackResult:
    """Solve an exact discrete LQG problem with additive zero-mean noise.

    The dynamics are ``x[k+1] = A[k]x[k] + B[k]u[k] + c[k] + G[k]w[k]``,
    where ``E[w[k]] = 0`` and ``Cov(w[k]) = Ω[k]``. All noise arrays have
    explicit case and time axes. State- or action-dependent noise is outside
    this exact certainty-equivalent interface.
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

    deterministic_result = finite_horizon_lqr(
        dynamics_matrices,
        control_matrices,
        state_costs,
        control_costs,
        terminal_state_cost,
        dynamics_bias=dynamics_bias,
        state_control_cross=state_control_cross,
        state_linear=state_linear,
        control_linear=control_linear,
        stage_constants=stage_constants,
        terminal_linear=terminal_linear,
        terminal_constant=terminal_constant,
        time_grid=time_grid,
        policy_id=policy_id,
        tolerance=tolerance,
        cost_tolerance=cost_tolerance,
    )
    future_matrices = deterministic_result.value.matrices[..., 1:, :, :]
    trace_increments = 0.5 * ein.contract(
        "...tij,...tji->...t",
        future_matrices,
        process_covariances,
    )
    cumulative = _reverse_cumulative(trace_increments)
    value_constant_corrections = jnp.concatenate(
        (cumulative, jnp.zeros(case_shape + (1,), dtype=cumulative.dtype)),
        axis=-1,
    )
    corrected_constants = (
        deterministic_result.value.constants + value_constant_corrections
    )
    value = QuadraticValueFunction(
        deterministic_result.value.matrices,
        deterministic_result.value.linear,
        corrected_constants,
        time_grid=deterministic_result.value.time_grid,
        case_shape=case_shape,
    )
    initial_covariance_cost, initial_expected_cost = _initial_cost_evidence(
        value.matrices,
        value.linear,
        value.constants,
        resolved_initial_mean,
        resolved_initial_covariance,
    )
    output_finite = (
        covariance_finite
        & initial_covariance_finite
        & _all_finite(trace_increments, 1)
        & _all_finite(value_constant_corrections, 1)
        & _all_finite(value.constants, 1)
        & jnp.isfinite(initial_covariance_cost)
        & jnp.isfinite(initial_expected_cost)
    )
    valid = deterministic_result.valid & output_finite
    status = jnp.where(
        ~deterministic_result.valid,
        int(LQGStateFeedbackStatus.DETERMINISTIC_SOLVE_FAILED),
        jnp.where(
            output_finite,
            int(LQGStateFeedbackStatus.SUCCESS),
            int(LQGStateFeedbackStatus.NONFINITE_OUTPUT),
        ),
    ).astype(jnp.int32)
    return FiniteHorizonLQGStateFeedbackResult(
        deterministic_result=deterministic_result,
        value=value,
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
        method="certainty-equivalent-additive-noise",
    )
