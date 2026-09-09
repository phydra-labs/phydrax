#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Auditable SVI/eSSVI nonlinear least-squares calibration lifecycle."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ...optim import least_squares, LevenbergMarquardt, OptimizationTermination
from ..core._evidence import FinanceEvidenceBinding
from ._surface import (
    ESSVISurface,
    evaluate_surface_arbitrage,
    SurfaceArbitrageEvidence,
    SVIParameters,
    SVISlice,
    SVISurface,
    VolatilityObservationSet,
)


CalibrationFamily = Literal["svi", "essvi"]


class CalibrationStatus(IntEnum):
    SUCCESS = 0
    OPTIMIZER_FAILED = 1
    SINGULAR_JACOBIAN = 2
    STATIC_ARBITRAGE = 3
    NONFINITE = 4
    REPLAY_MISMATCH = 5


class CalibrationPlan(StrictModule):
    family: CalibrationFamily = eqx.field(static=True)
    slice_expiries: Array
    maximum_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    initial_damping: float = eqx.field(static=True)
    fail_on_arbitrage: bool = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: CalibrationFamily,
        slice_expiries: ArrayLike,
        /,
        *,
        maximum_steps: int = 200,
        tolerance: float = 1.0e-9,
        initial_damping: float = 1.0e-3,
        fail_on_arbitrage: bool = True,
    ):
        if family not in ("svi", "essvi"):
            raise ValueError("calibration family must be svi or essvi.")
        expiries = jnp.asarray(slice_expiries, dtype=float)
        if expiries.ndim != 1 or expiries.size < (1 if family == "svi" else 2):
            raise ValueError("slice_expiries has insufficient slices for the family.")
        expiries = eqx.error_if(
            expiries,
            jnp.any(~jnp.isfinite(expiries))
            | jnp.any(expiries <= 0.0)
            | jnp.any(jnp.diff(expiries) <= 0.0),
            "slice_expiries must be positive and strictly increasing.",
        )
        if (
            isinstance(maximum_steps, bool)
            or not isinstance(maximum_steps, int)
            or maximum_steps < 1
        ):
            raise ValueError("maximum_steps must be a positive integer.")
        tolerance_, damping = float(tolerance), float(initial_damping)
        if (
            not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not isfinite(damping)
            or damping <= 0.0
        ):
            raise ValueError(
                "calibration tolerance and damping must be finite and positive."
            )
        if not isinstance(fail_on_arbitrage, bool):
            raise TypeError("fail_on_arbitrage must be boolean.")
        count = int(expiries.size)
        self.family, self.slice_expiries = family, expiries
        self.maximum_steps, self.tolerance, self.initial_damping = (
            maximum_steps,
            tolerance_,
            damping,
        )
        self.fail_on_arbitrage = fail_on_arbitrage
        self.parameter_count = 5 * count if family == "svi" else 2 * count + 2
        self.plan_id = canonical_fingerprint(
            {
                "kind": "volatility-surface-calibration-plan",
                "family": family,
                "slice_expiries": np.asarray(expiries).tolist(),
                "maximum_steps": maximum_steps,
                "tolerance": tolerance_,
                "initial_damping": damping,
                "fail_on_arbitrage": fail_on_arbitrage,
            }
        )


class PreparedCalibration(StrictModule):
    plan: CalibrationPlan
    observations: VolatilityObservationSet
    initial_parameters: Array
    observation_slice_index: Array
    evidence_binding: FinanceEvidenceBinding | None
    prepared_id: str = eqx.field(static=True)


class CalibrationEvidence(StrictModule):
    finite: Array
    optimizer_converged: Array
    jacobian_nonsingular: Array
    arbitrage_free: Array
    binding: FinanceEvidenceBinding | None
    status: CalibrationStatus = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            (self.status is CalibrationStatus.SUCCESS)
            & self.finite
            & self.optimizer_converged
            & self.jacobian_nonsingular
            & self.arbitrage_free
        )


class CalibrationResult(StrictModule):
    surface: SVISurface | ESSVISurface
    raw_parameters: Array
    fitted_implied_volatilities: Array
    residuals: Array
    jacobian_singular_values: Array
    objective: Array
    iterations: Array
    optimizer_status: Array
    arbitrage: SurfaceArbitrageEvidence
    evidence: CalibrationEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class CalibrationReplay(StrictModule):
    expected_fitted_volatilities: Array
    replayed_fitted_volatilities: Array
    maximum_absolute_difference: Array
    tolerance: Array
    matches: Array
    binding: FinanceEvidenceBinding | None
    plan_id: str = eqx.field(static=True)


def compile_calibration(
    observations: VolatilityObservationSet,
    /,
    *,
    family: CalibrationFamily = "svi",
    maximum_steps: int = 200,
    tolerance: float = 1.0e-9,
    initial_damping: float = 1.0e-3,
    fail_on_arbitrage: bool = True,
) -> CalibrationPlan:
    """Resolve expiry buckets on the host before fixed-shape optimization."""

    if not isinstance(observations, VolatilityObservationSet):
        raise TypeError("observations must be VolatilityObservationSet.")
    host_expiries = np.asarray(observations.expiries)
    host_valid = np.asarray(observations.valid)
    unique = np.unique(host_expiries[host_valid])
    if unique.size < (1 if family == "svi" else 2):
        raise ValueError("observations contain insufficient active expiry slices.")
    minimum_per_slice = 5 if family == "svi" else 3
    if any(
        np.count_nonzero(host_valid & (host_expiries == expiry)) < minimum_per_slice
        for expiry in unique
    ):
        raise ValueError(
            "each expiry slice is undersampled for the requested surface family."
        )
    parameter_count = 5 * unique.size if family == "svi" else 2 * unique.size + 2
    if np.count_nonzero(host_valid) < parameter_count:
        raise ValueError("active observations are fewer than calibrated parameters.")
    return CalibrationPlan(
        family,
        unique,
        maximum_steps=maximum_steps,
        tolerance=tolerance,
        initial_damping=initial_damping,
        fail_on_arbitrage=fail_on_arbitrage,
    )


def _inverse_softplus(value: Array) -> Array:
    value = jnp.maximum(value, 1.0e-8)
    return jnp.log(jnp.expm1(value))


def _default_initial(
    plan: CalibrationPlan, observations: VolatilityObservationSet, indices: Array
) -> Array:
    count = int(plan.slice_expiries.size)
    total_variance = observations.total_variances
    theta = jnp.stack(
        tuple(
            jnp.sum(
                jnp.where(
                    observations.valid & (indices == index),
                    observations.weights * total_variance,
                    0.0,
                )
            )
            / jnp.sum(
                jnp.where(
                    observations.valid & (indices == index), observations.weights, 0.0
                )
            )
            for index in range(count)
        )
    )
    if plan.family == "svi":
        raw = jnp.zeros((count, 5), dtype=total_variance.dtype)
        raw = raw.at[:, 0].set(_inverse_softplus(0.5 * theta))
        raw = raw.at[:, 1].set(_inverse_softplus(0.1 * theta + 1.0e-4))
        raw = raw.at[:, 4].set(_inverse_softplus(jnp.full_like(theta, 0.2)))
        return raw
    increments = jnp.concatenate((theta[:1], jnp.maximum(jnp.diff(theta), 1.0e-5)))
    return jnp.concatenate(
        (_inverse_softplus(increments), jnp.zeros((count + 2,), dtype=theta.dtype))
    )


def prepare_calibration(
    plan: CalibrationPlan,
    observations: VolatilityObservationSet,
    /,
    *,
    initial_parameters: ArrayLike | None = None,
    evidence_binding: FinanceEvidenceBinding | None = None,
) -> PreparedCalibration:
    if not isinstance(plan, CalibrationPlan) or not isinstance(
        observations, VolatilityObservationSet
    ):
        raise TypeError("plan and observations must be calibration records.")
    if evidence_binding is not None and not isinstance(
        evidence_binding, FinanceEvidenceBinding
    ):
        raise TypeError("evidence_binding must be FinanceEvidenceBinding or None.")
    differences = jnp.abs(observations.expiries[:, None] - plan.slice_expiries[None, :])
    indices = jnp.argmin(differences, axis=1).astype(jnp.int32)
    matched = jnp.take(plan.slice_expiries, indices)
    indices = eqx.error_if(
        indices,
        jnp.any(observations.valid & (matched != observations.expiries)),
        "observation expiry is absent from the compiled calibration plan.",
    )
    expected_shape = (
        (int(plan.slice_expiries.size), 5)
        if plan.family == "svi"
        else (plan.parameter_count,)
    )
    initial = (
        _default_initial(plan, observations, indices)
        if initial_parameters is None
        else jnp.asarray(initial_parameters, dtype=float)
    )
    if initial.shape != expected_shape:
        raise ValueError(f"initial_parameters must have shape {expected_shape}.")
    initial = eqx.error_if(
        initial, jnp.any(~jnp.isfinite(initial)), "initial_parameters must be finite."
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-volatility-calibration",
            "plan": plan.plan_id,
            "observation_capacity": observations.observation_count,
        }
    )
    return PreparedCalibration(
        plan, observations, initial, indices, evidence_binding, prepared_id
    )


def _svi_transformed(raw: Array) -> tuple[Array, Array, Array, Array, Array]:
    return (
        jax.nn.softplus(raw[:, 0]),
        jax.nn.softplus(raw[:, 1]) + 1.0e-10,
        0.999 * jnp.tanh(raw[:, 2]),
        raw[:, 3],
        jax.nn.softplus(raw[:, 4]) + 1.0e-10,
    )


def _essvi_transformed(plan: CalibrationPlan, raw: Array):
    count = int(plan.slice_expiries.size)
    theta = jnp.cumsum(jax.nn.softplus(raw[:count]) + 1.0e-10)
    rho = 0.999 * jnp.tanh(raw[count : 2 * count])
    gamma = jax.nn.sigmoid(raw[-1])
    base = theta**gamma * (1.0 + theta) ** (1.0 - gamma)
    first_bound = 4.0 * base / (theta * (1.0 + jnp.abs(rho)))
    second_bound = 2.0 * base / jnp.sqrt(theta * (1.0 + jnp.abs(rho)))
    eta_limit = 0.99 * jnp.min(jnp.minimum(first_bound, second_bound))
    eta = eta_limit * jax.nn.sigmoid(raw[-2])
    return theta, rho, eta, gamma


def _predicted_total_variance(prepared: PreparedCalibration, raw: Array) -> Array:
    observations, indices = prepared.observations, prepared.observation_slice_index
    if prepared.plan.family == "svi":
        level, slope, rho, center, width = _svi_transformed(raw)
        shifted = observations.log_moneyness - center[indices]
        return level[indices] + slope[indices] * (
            rho[indices] * shifted + jnp.sqrt(shifted**2 + width[indices] ** 2)
        )
    theta, rho, eta, gamma = _essvi_transformed(prepared.plan, raw)
    theta_i, rho_i = theta[indices], rho[indices]
    phi = eta / (theta_i**gamma * (1.0 + theta_i) ** (1.0 - gamma))
    k = observations.log_moneyness
    return (
        0.5
        * theta_i
        * (1.0 + rho_i * phi * k + jnp.sqrt((phi * k + rho_i) ** 2 + 1.0 - rho_i**2))
    )


def _residual(prepared: PreparedCalibration, raw: Array) -> Array:
    prediction = _predicted_total_variance(prepared, raw)
    return jnp.where(
        prepared.observations.valid,
        jnp.sqrt(prepared.observations.weights)
        * (prediction - prepared.observations.total_variances),
        0.0,
    )


def _surface(prepared: PreparedCalibration, raw: Array) -> SVISurface | ESSVISurface:
    if prepared.plan.family == "svi":
        transformed = _svi_transformed(raw)
        slices = tuple(
            SVISlice(
                prepared.plan.slice_expiries[index],
                SVIParameters(*(value[index] for value in transformed)),
            )
            for index in range(int(prepared.plan.slice_expiries.size))
        )
        return SVISurface(slices)
    theta, rho, eta, gamma = _essvi_transformed(prepared.plan, raw)
    return ESSVISurface(prepared.plan.slice_expiries, theta, rho, eta, gamma)


def evaluate_calibration(prepared: PreparedCalibration, /) -> CalibrationResult:
    if not isinstance(prepared, PreparedCalibration):
        raise TypeError("evaluate_calibration requires PreparedCalibration.")
    optimizer = least_squares(
        lambda parameters, _: _residual(prepared, parameters),
        prepared.initial_parameters,
        method=LevenbergMarquardt(initial_damping=prepared.plan.initial_damping),
        termination=OptimizationTermination(
            absolute_optimality=prepared.plan.tolerance,
            relative_optimality=prepared.plan.tolerance,
            absolute_step=prepared.plan.tolerance,
            relative_step=prepared.plan.tolerance,
            maximum_steps=prepared.plan.maximum_steps,
        ),
    )
    raw = optimizer.parameters
    jacobian = jax.jacrev(lambda parameters: _residual(prepared, parameters))(raw)
    jacobian_matrix = jacobian.reshape(
        (prepared.observations.observation_count, prepared.plan.parameter_count)
    )
    decomposition = factorize(
        DenseLinearOperator(jacobian_matrix), FactorizationPolicy("svd")
    )
    rank = decomposition.rank()
    singular_values = decomposition.singular_values()
    nonsingular = rank == prepared.plan.parameter_count
    raw = eqx.error_if(raw, ~nonsingular, "calibration Jacobian is rank deficient.")
    surface = _surface(prepared, raw)
    arbitrage = evaluate_surface_arbitrage(surface)
    if prepared.plan.fail_on_arbitrage:
        raw = eqx.error_if(
            raw,
            ~arbitrage.valid,
            "calibrated surface violates static-arbitrage conditions.",
        )
    predicted_total = _predicted_total_variance(prepared, raw)
    fitted_volatility = jnp.where(
        prepared.observations.valid,
        jnp.sqrt(predicted_total / prepared.observations.expiries),
        0.0,
    )
    residuals = _residual(prepared, raw)
    finite = (
        jnp.all(jnp.isfinite(raw))
        & jnp.all(jnp.isfinite(fitted_volatility))
        & jnp.all(jnp.isfinite(residuals))
    )
    optimizer_converged = optimizer.successful
    raw = eqx.error_if(
        raw,
        ~optimizer_converged | ~finite,
        "surface calibration optimizer did not converge to finite parameters.",
    )
    evidence = CalibrationEvidence(
        finite,
        optimizer_converged,
        nonsingular,
        arbitrage.valid,
        prepared.evidence_binding,
        CalibrationStatus.SUCCESS,
    )
    return CalibrationResult(
        surface,
        raw,
        fitted_volatility,
        residuals,
        singular_values,
        optimizer.objective,
        optimizer.diagnostics.iterations,
        optimizer.status,
        arbitrage,
        evidence,
        prepared.plan.plan_id,
    )


def replay_calibration(
    prepared: PreparedCalibration,
    expected: CalibrationResult,
    /,
    *,
    tolerance: ArrayLike = 1.0e-10,
    evidence_binding: FinanceEvidenceBinding | None = None,
) -> CalibrationReplay:
    if not isinstance(prepared, PreparedCalibration) or not isinstance(
        expected, CalibrationResult
    ):
        raise TypeError("prepared and expected must be calibration lifecycle records.")
    if expected.plan_id != prepared.plan.plan_id:
        raise ValueError("expected result was produced by a different calibration plan.")
    tolerance_ = jnp.asarray(tolerance, dtype=float)
    if tolerance_.shape != ():
        raise ValueError("tolerance must be scalar.")
    tolerance_ = eqx.error_if(
        tolerance_,
        ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
        "replay tolerance must be finite and non-negative.",
    )
    replay_total = _predicted_total_variance(prepared, expected.raw_parameters)
    replay = jnp.where(
        prepared.observations.valid,
        jnp.sqrt(replay_total / prepared.observations.expiries),
        0.0,
    )
    difference = jnp.max(jnp.abs(replay - expected.fitted_implied_volatilities))
    matches = jnp.isfinite(difference) & (difference <= tolerance_)
    return CalibrationReplay(
        expected.fitted_implied_volatilities,
        replay,
        difference,
        tolerance_,
        matches,
        evidence_binding,
        prepared.plan.plan_id,
    )


__all__ = [
    "CalibrationEvidence",
    "CalibrationPlan",
    "CalibrationReplay",
    "CalibrationResult",
    "CalibrationStatus",
    "PreparedCalibration",
    "compile_calibration",
    "evaluate_calibration",
    "prepare_calibration",
    "replay_calibration",
]
