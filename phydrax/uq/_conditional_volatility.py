#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._numerics import solve_weighted_least_squares
from .._strict import StrictModule
from ..optim import minimize, NewtonTrustRegion, OptimizationTermination


ConditionalVolatilityKind: TypeAlias = Literal["garch", "gjr-garch", "egarch"]
ConditionalVolatilityStatus: TypeAlias = Literal[0, 1, 2, 3]

CONDITIONAL_VOLATILITY_SUCCESS = 0
CONDITIONAL_VOLATILITY_INSUFFICIENT = 1
CONDITIONAL_VOLATILITY_UNSTABLE = 2
CONDITIONAL_VOLATILITY_NONFINITE = 3


def _univariate(values: ArrayLike, mask: ArrayLike | None) -> tuple[Array, Array]:
    data = jnp.asarray(values)
    if data.ndim != 1 or data.shape[0] < 3:
        raise ValueError("values must be a vector with at least three entries.")
    if not jnp.issubdtype(data.dtype, jnp.inexact):
        data = data.astype(float)
    valid = (
        jnp.ones(data.shape, dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if valid.shape != data.shape:
        raise ValueError("mask must have the same shape as values.")
    data = eqx.error_if(
        data,
        jnp.any(valid & ~jnp.isfinite(data)),
        "active values must be finite.",
    )
    return jnp.where(valid, data, 0.0), valid


class GARCHModel(StrictModule):
    """A diagnosed GARCH, GJR-GARCH, or EGARCH(1,1) recursion."""

    omega: Array
    alpha: Array
    beta: Array
    gamma: Array
    persistence: Array
    stable: Array
    kind: ConditionalVolatilityKind = eqx.field(static=True)

    def __init__(
        self,
        omega: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        /,
        *,
        gamma: ArrayLike = 0.0,
        kind: ConditionalVolatilityKind = "garch",
    ):
        if kind not in ("garch", "gjr-garch", "egarch"):
            raise ValueError("kind must be 'garch', 'gjr-garch', or 'egarch'.")
        dtype = jnp.result_type(omega, alpha, beta, gamma, float)
        omega_ = jnp.asarray(omega, dtype=dtype)
        alpha_ = jnp.asarray(alpha, dtype=dtype)
        beta_ = jnp.asarray(beta, dtype=dtype)
        gamma_ = jnp.asarray(gamma, dtype=dtype)
        if any(value.ndim != 0 for value in (omega_, alpha_, beta_, gamma_)):
            raise ValueError("conditional-volatility parameters must be scalars.")
        finite = (
            jnp.isfinite(omega_)
            & jnp.isfinite(alpha_)
            & jnp.isfinite(beta_)
            & jnp.isfinite(gamma_)
        )
        if kind == "egarch":
            admissible = (alpha_ >= 0.0) & (jnp.abs(beta_) < 1.0)
            persistence = jnp.abs(beta_)
        elif kind == "gjr-garch":
            admissible = (
                (omega_ > 0.0) & (alpha_ >= 0.0) & (beta_ >= 0.0) & (gamma_ >= 0.0)
            )
            persistence = alpha_ + beta_ + 0.5 * gamma_
        else:
            admissible = (
                (omega_ > 0.0) & (alpha_ >= 0.0) & (beta_ >= 0.0) & (gamma_ == 0.0)
            )
            persistence = alpha_ + beta_
        omega_ = eqx.error_if(
            omega_,
            ~finite | ~admissible,
            "conditional-volatility parameters violate their model constraints.",
        )
        self.omega = omega_
        self.alpha = alpha_
        self.beta = beta_
        self.gamma = gamma_
        self.persistence = persistence
        self.stable = persistence < 1.0
        self.kind = kind

    def conditional_variance(
        self,
        residuals: ArrayLike,
        /,
        *,
        mask: ArrayLike | None = None,
        initial_variance: ArrayLike | None = None,
    ) -> Array:
        values, valid = _univariate(residuals, mask)
        observed_square = jnp.where(valid, jnp.square(values), 0.0)
        count = jnp.maximum(jnp.sum(valid), 1)
        empirical = jnp.sum(observed_square) / count
        initial = (
            jnp.maximum(empirical, jnp.finfo(values.dtype).tiny)
            if initial_variance is None
            else jnp.asarray(initial_variance, dtype=values.dtype)
        )
        if initial.ndim != 0:
            raise ValueError("initial_variance must be scalar or None.")
        initial = eqx.error_if(
            initial,
            ~jnp.isfinite(initial) | (initial <= 0.0),
            "initial_variance must be finite and positive.",
        )

        def step(previous_variance, inputs):
            previous_residual, previous_valid = inputs
            if self.kind == "egarch":
                scale = jnp.sqrt(
                    jnp.maximum(previous_variance, jnp.finfo(values.dtype).tiny)
                )
                standardized = previous_residual / scale
                candidate_log = (
                    self.omega
                    + self.beta * jnp.log(previous_variance)
                    + self.alpha * (jnp.abs(standardized) - jnp.sqrt(2.0 / jnp.pi))
                    + self.gamma * standardized
                )
                candidate = jnp.exp(candidate_log)
            else:
                leverage = (
                    self.gamma
                    * (previous_residual < 0.0).astype(values.dtype)
                    * jnp.square(previous_residual)
                )
                candidate = (
                    self.omega
                    + self.alpha * jnp.square(previous_residual)
                    + leverage
                    + self.beta * previous_variance
                )
            next_variance = jnp.where(previous_valid, candidate, previous_variance)
            next_variance = jnp.maximum(next_variance, jnp.finfo(values.dtype).tiny)
            return next_variance, next_variance

        _, tail = jax.lax.scan(step, initial, (values[:-1], valid[:-1]))
        return jnp.concatenate((initial[None], tail))


class GARCHFit(StrictModule):
    """QMLE fit retaining likelihood, stationarity, and optimizer evidence."""

    model: GARCHModel
    conditional_variance: Array
    standardized_residuals: Array
    valid_mask: Array
    log_likelihood: Array
    effective_sample_count: Array
    gradient_norm: Array
    iterations: Array
    optimizer_status: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == CONDITIONAL_VOLATILITY_SUCCESS


def _softplus_inverse(value: Array) -> Array:
    return jnp.log(jnp.expm1(jnp.maximum(value, jnp.finfo(value.dtype).eps)))


def _decode_garch(raw: Array, kind: ConditionalVolatilityKind, dtype) -> GARCHModel:
    epsilon = 64.0 * jnp.finfo(dtype).eps
    if kind == "egarch":
        return GARCHModel(
            raw[0],
            jax.nn.softplus(raw[1]),
            (1.0 - epsilon) * jnp.tanh(raw[2]),
            gamma=raw[3],
            kind=kind,
        )
    count = 3 if kind == "gjr-garch" else 2
    weights = jax.nn.softmax(
        jnp.concatenate((raw[1 : 1 + count], jnp.zeros((1,), dtype=dtype)))
    )
    weights = weights[:-1] * (1.0 - epsilon)
    if kind == "gjr-garch":
        alpha, beta, half_gamma = weights
        gamma = 2.0 * half_gamma
    else:
        alpha, beta = weights
        gamma = jnp.asarray(0.0, dtype=dtype)
    return GARCHModel(
        jax.nn.softplus(raw[0]) + jnp.finfo(dtype).tiny,
        alpha,
        beta,
        gamma=gamma,
        kind=kind,
    )


def fit_garch(
    residuals: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    kind: ConditionalVolatilityKind = "garch",
    maximum_steps: int = 128,
) -> GARCHFit:
    """Fit a constrained conditional-volatility recursion by Gaussian QMLE."""

    values, valid = _univariate(residuals, mask)
    steps = int(maximum_steps)
    if steps < 1:
        raise ValueError("maximum_steps must be positive.")
    count = jnp.sum(valid).astype(jnp.int32)
    empirical = jnp.sum(jnp.where(valid, jnp.square(values), 0.0)) / jnp.maximum(count, 1)
    empirical = jnp.maximum(empirical, jnp.finfo(values.dtype).eps)
    if kind == "egarch":
        initial = jnp.asarray(
            [jnp.log(empirical) * 0.05, -2.0, 1.5, 0.0], dtype=values.dtype
        )
    else:
        persistence_weights = (
            jnp.asarray([0.05, 0.85, 0.03], dtype=values.dtype)
            if kind == "gjr-garch"
            else jnp.asarray([0.08, 0.85], dtype=values.dtype)
        )
        remainder = 1.0 - jnp.sum(persistence_weights)
        logits = jnp.log(persistence_weights / remainder)
        omega_initial = empirical * jnp.maximum(1.0 - jnp.sum(persistence_weights), 0.01)
        initial = jnp.concatenate((_softplus_inverse(omega_initial)[None], logits))

    def objective(raw, arguments):
        observations, observation_mask, initial_variance = arguments
        model = _decode_garch(raw, kind, observations.dtype)
        variance = model.conditional_variance(
            observations,
            mask=observation_mask,
            initial_variance=initial_variance,
        )
        terms = 0.5 * (
            jnp.log(2.0 * jnp.pi * variance) + jnp.square(observations) / variance
        )
        return jnp.sum(jnp.where(observation_mask, terms, 0.0))

    optimization = minimize(
        objective,
        initial,
        method=NewtonTrustRegion(),
        termination=OptimizationTermination(maximum_steps=steps),
        args=(values, valid, empirical),
    )
    model = _decode_garch(optimization.parameters, kind, values.dtype)
    variance = model.conditional_variance(values, mask=valid, initial_variance=empirical)
    standardized = jnp.where(valid, values / jnp.sqrt(variance), 0.0)
    log_likelihood = -objective(optimization.parameters, (values, valid, empirical))
    finite = (
        jnp.isfinite(log_likelihood)
        & jnp.all(jnp.isfinite(variance))
        & jnp.all(variance > 0.0)
    )
    status = jnp.where(
        ~finite,
        CONDITIONAL_VOLATILITY_NONFINITE,
        jnp.where(
            count < 8,
            CONDITIONAL_VOLATILITY_INSUFFICIENT,
            jnp.where(
                ~model.stable,
                CONDITIONAL_VOLATILITY_UNSTABLE,
                CONDITIONAL_VOLATILITY_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return GARCHFit(
        model=model,
        conditional_variance=variance,
        standardized_residuals=standardized,
        valid_mask=valid,
        log_likelihood=log_likelihood,
        effective_sample_count=count,
        gradient_norm=optimization.diagnostics.final_optimality_norm,
        iterations=optimization.diagnostics.iterations,
        optimizer_status=optimization.status,
        status=status,
    )


class HARModel(StrictModule):
    """Heterogeneous autoregressive model over declared trailing windows."""

    intercept: Array
    coefficients: Array
    innovation_variance: Array
    windows: tuple[int, ...] = eqx.field(static=True)

    def forecast(self, history: ArrayLike, /) -> Array:
        values = jnp.asarray(history, dtype=self.coefficients.dtype)
        if values.ndim != 1 or values.shape[0] < max(self.windows):
            raise ValueError("history is shorter than the largest HAR window.")
        regressors = jnp.stack(
            tuple(jnp.mean(values[-window:]) for window in self.windows)
        )
        return self.intercept + jnp.sum(self.coefficients * regressors)


class HARFit(StrictModule):
    """HAR fit with complete-window masks and linear-system diagnostics."""

    model: HARModel
    fitted_values: Array
    residuals: Array
    valid_mask: Array
    log_likelihood: Array
    effective_sample_count: Array
    rank: Array
    condition_number: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == CONDITIONAL_VOLATILITY_SUCCESS


def fit_har(
    realized_measure: ArrayLike,
    /,
    *,
    windows: tuple[int, ...] = (1, 5, 22),
    mask: ArrayLike | None = None,
    include_intercept: bool = True,
    ridge: float = 0.0,
    rcond: float | None = None,
) -> HARFit:
    """Fit HAR using only complete, strictly historical trailing windows."""

    values, valid = _univariate(realized_measure, mask)
    windows_ = tuple(int(window) for window in windows)
    if not windows_ or any(window < 1 for window in windows_):
        raise ValueError("windows must be a nonempty tuple of positive integers.")
    if len(set(windows_)) != len(windows_):
        raise ValueError("HAR windows must be unique.")
    maximum = max(windows_)
    feature_count = int(include_intercept) + len(windows_)
    if values.shape[0] - maximum <= feature_count:
        raise ValueError("series is too short for the requested HAR windows.")
    columns = []
    window_masks = []
    if include_intercept:
        columns.append(jnp.ones((values.shape[0] - maximum,), dtype=values.dtype))
    for window in windows_:
        means = []
        complete = []
        for time in range(maximum, values.shape[0]):
            interval = values[time - window : time]
            interval_valid = valid[time - window : time]
            means.append(jnp.sum(jnp.where(interval_valid, interval, 0.0)) / window)
            complete.append(jnp.all(interval_valid))
        columns.append(jnp.stack(means))
        window_masks.append(jnp.stack(complete))
    design = jnp.stack(columns, axis=-1)
    target = values[maximum:]
    fit_mask = valid[maximum:] & jnp.all(jnp.stack(window_masks, axis=-1), axis=-1)
    least_squares = solve_weighted_least_squares(
        design,
        target,
        mask=fit_mask,
        ridge=ridge,
        rcond=rcond,
        min_samples=feature_count + 1,
        max_features=feature_count,
    )
    coefficients = least_squares.raw_coefficients
    offset = 1 if include_intercept else 0
    intercept = coefficients[0] if include_intercept else jnp.asarray(0.0, values.dtype)
    slopes = coefficients[offset:]
    residual = jnp.where(fit_mask, least_squares.residual, 0.0)
    count = jnp.sum(fit_mask).astype(jnp.int32)
    variance = jnp.sum(jnp.square(residual)) / jnp.maximum(count, 1)
    variance = jnp.maximum(variance, jnp.finfo(values.dtype).tiny)
    model = HARModel(
        intercept=intercept,
        coefficients=slopes,
        innovation_variance=variance,
        windows=windows_,
    )
    terms = -0.5 * (jnp.log(2.0 * jnp.pi * variance) + jnp.square(residual) / variance)
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(variance)
    status = jnp.where(
        ~finite,
        CONDITIONAL_VOLATILITY_NONFINITE,
        jnp.where(
            count <= feature_count,
            CONDITIONAL_VOLATILITY_INSUFFICIENT,
            jnp.where(
                (float(ridge) == 0.0) & (least_squares.rank < feature_count),
                CONDITIONAL_VOLATILITY_NONFINITE,
                CONDITIONAL_VOLATILITY_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return HARFit(
        model=model,
        fitted_values=least_squares.prediction,
        residuals=residual,
        valid_mask=fit_mask,
        log_likelihood=jnp.sum(jnp.where(fit_mask, terms, 0.0)),
        effective_sample_count=count,
        rank=least_squares.rank,
        condition_number=least_squares.condition_number,
        status=status,
    )


__all__ = [
    "CONDITIONAL_VOLATILITY_INSUFFICIENT",
    "CONDITIONAL_VOLATILITY_NONFINITE",
    "CONDITIONAL_VOLATILITY_SUCCESS",
    "CONDITIONAL_VOLATILITY_UNSTABLE",
    "ConditionalVolatilityKind",
    "ConditionalVolatilityStatus",
    "GARCHFit",
    "GARCHModel",
    "HARFit",
    "HARModel",
    "fit_garch",
    "fit_har",
]
