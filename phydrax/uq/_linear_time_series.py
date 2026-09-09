#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._numerics import solve_weighted_least_squares
from .._strict import StrictModule
from ..linalg._dense_inverse import dense_inverse


TimeSeriesFitStatus: TypeAlias = Literal[0, 1, 2, 3]
TIME_SERIES_SUCCESS = 0
TIME_SERIES_INSUFFICIENT = 1
TIME_SERIES_RANK_DEFICIENT = 2
TIME_SERIES_NONFINITE = 3


def _series(values: ArrayLike, mask: ArrayLike | None) -> tuple[Array, Array]:
    data = jnp.asarray(values)
    if data.ndim not in (1, 2):
        raise ValueError("values must have shape (time,) or (time, variables).")
    if data.shape[0] < 2:
        raise ValueError("values must contain at least two time points.")
    if not jnp.issubdtype(data.dtype, jnp.inexact):
        data = data.astype(float)
    valid = (
        jnp.ones(data.shape, dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if valid.shape == (data.shape[0],) and data.ndim == 2:
        valid = jnp.broadcast_to(valid[:, None], data.shape)
    if valid.shape != data.shape:
        raise ValueError("mask must match values or be a time mask for a matrix series.")
    invalid = valid & ~jnp.isfinite(data)
    data = eqx.error_if(
        data, jnp.any(invalid), "active time-series values must be finite."
    )
    return jnp.where(valid, data, 0.0), valid


def _difference(values: Array, valid: Array, order: int) -> tuple[Array, Array]:
    result = values
    result_valid = valid
    for _ in range(order):
        result = result[1:] - result[:-1]
        result_valid = result_valid[1:] & result_valid[:-1]
    return result, result_valid


def _companion_radius(coefficients: Array) -> Array:
    order = int(coefficients.shape[0])
    if order == 0:
        return jnp.asarray(0.0, dtype=coefficients.dtype)
    companion = jnp.zeros((order, order), dtype=coefficients.dtype)
    companion = companion.at[0].set(coefficients)
    if order > 1:
        companion = companion.at[1:, :-1].set(
            jnp.eye(order - 1, dtype=coefficients.dtype)
        )
    return jnp.max(jnp.abs(jnp.linalg.eigvals(companion))).real


def _gaussian_log_likelihood(residual: Array, valid: Array, variance: Array) -> Array:
    safe_variance = jnp.maximum(variance, jnp.finfo(residual.dtype).tiny)
    terms = -0.5 * (
        jnp.log(2.0 * jnp.pi * safe_variance) + jnp.square(residual) / safe_variance
    )
    return jnp.sum(jnp.where(valid, terms, 0.0))


class ARIMAModel(StrictModule):
    """Fitted scalar ARIMA law in differenced coordinates."""

    intercept: Array
    autoregressive: Array
    moving_average: Array
    innovation_variance: Array
    autoregressive_radius: Array
    moving_average_radius: Array
    stable: Array
    invertible: Array
    p: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    q: int = eqx.field(static=True)

    def __init__(
        self,
        intercept: ArrayLike,
        autoregressive: ArrayLike,
        moving_average: ArrayLike,
        innovation_variance: ArrayLike,
        /,
        *,
        differencing: int,
    ):
        ar = jnp.asarray(autoregressive)
        ma = jnp.asarray(moving_average)
        if ar.ndim != 1 or ma.ndim != 1:
            raise ValueError("autoregressive and moving_average must be vectors.")
        dtype = jnp.result_type(ar, ma, intercept, innovation_variance, float)
        ar = ar.astype(dtype)
        ma = ma.astype(dtype)
        intercept_ = jnp.asarray(intercept, dtype=dtype)
        variance = jnp.asarray(innovation_variance, dtype=dtype)
        if intercept_.ndim != 0 or variance.ndim != 0:
            raise ValueError("intercept and innovation_variance must be scalars.")
        ar = eqx.error_if(
            ar,
            jnp.any(~jnp.isfinite(ar))
            | jnp.any(~jnp.isfinite(ma))
            | ~jnp.isfinite(intercept_)
            | ~jnp.isfinite(variance)
            | (variance <= 0.0),
            "ARIMA parameters must be finite and innovation variance positive.",
        )
        differencing_ = int(differencing)
        if differencing_ < 0:
            raise ValueError("differencing must be nonnegative.")
        ar_radius = _companion_radius(ar)
        ma_radius = _companion_radius(-ma)
        self.intercept = intercept_
        self.autoregressive = ar
        self.moving_average = ma
        self.innovation_variance = variance
        self.autoregressive_radius = ar_radius
        self.moving_average_radius = ma_radius
        self.stable = ar_radius < 1.0
        self.invertible = ma_radius < 1.0
        self.p = int(ar.shape[0])
        self.d = differencing_
        self.q = int(ma.shape[0])

    def forecast(
        self,
        history: ArrayLike,
        steps: int,
        /,
        *,
        innovations: ArrayLike | None = None,
    ) -> Array:
        """Forecast levels, using zero future innovations by default."""

        values = jnp.asarray(history, dtype=self.autoregressive.dtype)
        if values.ndim != 1 or values.shape[0] <= self.d:
            raise ValueError("history must be a vector longer than differencing order.")
        count = int(steps)
        if count < 1:
            raise ValueError("steps must be positive.")
        levels = [values]
        for _ in range(self.d):
            levels.append(levels[-1][1:] - levels[-1][:-1])
        transformed = levels[-1]
        residual_history = (
            jnp.zeros_like(transformed)
            if innovations is None
            else jnp.asarray(innovations, dtype=values.dtype)
        )
        if (
            residual_history.ndim != 1
            or residual_history.shape[0] != transformed.shape[0]
        ):
            raise ValueError("innovations must match the differenced history length.")
        last_levels = [level[-1] for level in levels[:-1]]
        output = []
        for _ in range(count):
            ar_term = jnp.asarray(0.0, dtype=values.dtype)
            for lag in range(self.p):
                ar_term = ar_term + self.autoregressive[lag] * transformed[-lag - 1]
            ma_term = jnp.asarray(0.0, dtype=values.dtype)
            for lag in range(self.q):
                ma_term = ma_term + self.moving_average[lag] * residual_history[-lag - 1]
            highest = self.intercept + ar_term + ma_term
            transformed = jnp.concatenate((transformed, highest[None]))
            residual_history = jnp.concatenate(
                (residual_history, jnp.zeros((1,), dtype=values.dtype))
            )
            level_value = highest
            for level in range(self.d - 1, -1, -1):
                level_value = last_levels[level] + level_value
                last_levels[level] = level_value
            output.append(level_value)
        return jnp.stack(output)


class ARIMAFit(StrictModule):
    """Conditional-likelihood ARIMA fit with complete numerical diagnostics."""

    model: ARIMAModel
    fitted_values: Array
    residuals: Array
    residual_mask: Array
    log_likelihood: Array
    effective_sample_count: Array
    rank: Array
    condition_number: Array
    normal_equation_error: Array
    status: Array
    iterations: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == TIME_SERIES_SUCCESS


def fit_arima(
    values: ArrayLike,
    /,
    *,
    p: int,
    d: int = 0,
    q: int = 0,
    mask: ArrayLike | None = None,
    include_intercept: bool = True,
    iterations: int = 4,
    ridge: float = 0.0,
    rcond: float | None = None,
) -> ARIMAFit:
    """Fit a scalar ARIMA model by diagnosed iterated conditional least squares."""

    p_, d_, q_ = int(p), int(d), int(q)
    iterations_ = int(iterations)
    if min(p_, d_, q_) < 0 or p_ + q_ < 1:
        raise ValueError(
            "orders must be nonnegative and at least one of p or q positive."
        )
    if iterations_ < 1:
        raise ValueError("iterations must be positive.")
    ridge_ = float(ridge)
    if not math.isfinite(ridge_) or ridge_ < 0.0:
        raise ValueError("ridge must be finite and nonnegative.")
    data, valid = _series(values, mask)
    if data.ndim != 1:
        raise ValueError("ARIMA values must be one-dimensional.")
    transformed, transformed_valid = _difference(data, valid, d_)
    lag = max(p_, q_)
    features = int(include_intercept) + p_ + q_
    if transformed.shape[0] - lag <= features:
        raise ValueError("time series is too short for the requested ARIMA orders.")
    residual = jnp.zeros_like(transformed)
    residual_valid = transformed_valid
    last_result = None
    target = transformed[lag:]
    row_mask = transformed_valid[lag:]
    for _ in range(iterations_):
        columns = []
        column_masks = []
        if include_intercept:
            columns.append(jnp.ones_like(target))
            column_masks.append(jnp.ones_like(row_mask))
        for order in range(1, p_ + 1):
            columns.append(transformed[lag - order : transformed.shape[0] - order])
            column_masks.append(
                transformed_valid[lag - order : transformed.shape[0] - order]
            )
        for order in range(1, q_ + 1):
            columns.append(residual[lag - order : transformed.shape[0] - order])
            column_masks.append(
                residual_valid[lag - order : transformed.shape[0] - order]
            )
        design = jnp.stack(columns, axis=-1)
        row_mask = transformed_valid[lag:] & jnp.all(
            jnp.stack(column_masks, axis=-1), axis=-1
        )
        last_result = solve_weighted_least_squares(
            design,
            target,
            mask=row_mask,
            ridge=ridge_,
            rcond=rcond,
            min_samples=features + 1,
            max_features=features,
        )
        fitted_tail = last_result.prediction
        residual = residual.at[lag:].set(jnp.where(row_mask, target - fitted_tail, 0.0))
        residual_valid = residual_valid.at[:lag].set(False)
        residual_valid = residual_valid.at[lag:].set(row_mask)
    coefficients = last_result.raw_coefficients
    offset = 1 if include_intercept else 0
    intercept = coefficients[0] if include_intercept else jnp.asarray(0.0, data.dtype)
    ar = coefficients[offset : offset + p_]
    ma = coefficients[offset + p_ :]
    count = jnp.sum(residual_valid).astype(jnp.int32)
    variance = jnp.sum(
        jnp.where(residual_valid, jnp.square(residual), 0.0)
    ) / jnp.maximum(count, 1)
    variance = jnp.maximum(variance, jnp.finfo(data.dtype).tiny)
    model = ARIMAModel(intercept, ar, ma, variance, differencing=d_)
    fitted = jnp.where(residual_valid, transformed - residual, 0.0)
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(variance)
    status = jnp.where(
        ~finite,
        TIME_SERIES_NONFINITE,
        jnp.where(
            count <= features,
            TIME_SERIES_INSUFFICIENT,
            jnp.where(
                (ridge_ == 0.0) & (last_result.rank < features),
                TIME_SERIES_RANK_DEFICIENT,
                TIME_SERIES_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return ARIMAFit(
        model=model,
        fitted_values=fitted,
        residuals=residual,
        residual_mask=residual_valid,
        log_likelihood=_gaussian_log_likelihood(residual, residual_valid, variance),
        effective_sample_count=count,
        rank=last_result.rank,
        condition_number=last_result.condition_number,
        normal_equation_error=last_result.normal_equation_error,
        status=status,
        iterations=iterations_,
    )


class UnitRootResult(StrictModule):
    """Augmented Dickey--Fuller coefficient and test evidence."""

    level_coefficient: Array
    standard_error: Array
    statistic: Array
    critical_value: Array
    stationary: Array
    residual_variance: Array
    effective_sample_count: Array
    rank: Array
    condition_number: Array
    valid: Array
    lag_differences: int = eqx.field(static=True)
    include_intercept: bool = eqx.field(static=True)


def augmented_dickey_fuller(
    values: ArrayLike,
    /,
    *,
    lag_differences: int = 0,
    mask: ArrayLike | None = None,
    include_intercept: bool = True,
    critical_value: float = -2.86,
    ridge: float = 0.0,
) -> UnitRootResult:
    """Test the lagged-level coefficient in an explicitly declared ADF regression."""

    data, valid = _series(values, mask)
    if data.ndim != 1:
        raise ValueError("ADF values must be one-dimensional.")
    lags = int(lag_differences)
    critical = float(critical_value)
    ridge_ = float(ridge)
    if lags < 0:
        raise ValueError("lag_differences must be nonnegative.")
    if not math.isfinite(critical) or not math.isfinite(ridge_) or ridge_ < 0.0:
        raise ValueError("critical_value must be finite and ridge finite/nonnegative.")
    differences = data[1:] - data[:-1]
    difference_valid = valid[1:] & valid[:-1]
    start = lags + 1
    if data.shape[0] - start <= lags + int(include_intercept) + 2:
        raise ValueError("series is too short for the requested ADF regression.")
    target = differences[lags:]
    columns = []
    masks = []
    if include_intercept:
        columns.append(jnp.ones_like(target))
        masks.append(jnp.ones_like(target, dtype=bool))
    columns.append(data[start - 1 : -1])
    masks.append(valid[start - 1 : -1])
    for lag in range(1, lags + 1):
        columns.append(differences[lags - lag : -lag])
        masks.append(difference_valid[lags - lag : -lag])
    design = jnp.stack(columns, axis=-1)
    row_mask = difference_valid[lags:] & jnp.all(jnp.stack(masks, axis=-1), axis=-1)
    fit = solve_weighted_least_squares(
        design,
        target,
        mask=row_mask,
        ridge=ridge_,
        min_samples=design.shape[-1] + 2,
    )
    level_index = int(include_intercept)
    coefficient = fit.raw_coefficients[level_index]
    count = jnp.sum(row_mask).astype(jnp.int32)
    degrees = jnp.maximum(count - design.shape[-1], 1)
    residual_variance = (
        jnp.sum(jnp.where(row_mask, jnp.square(fit.residual), 0.0)) / degrees
    )
    masked_design = jnp.where(row_mask[:, None], design, 0.0)
    gram = masked_design.T @ masked_design
    inverse_gram = dense_inverse(
        gram
        + jnp.maximum(ridge_, jnp.finfo(data.dtype).eps)
        * jnp.eye(design.shape[-1], dtype=data.dtype),
        positive_definite=True,
    )
    standard_error = jnp.sqrt(
        jnp.maximum(
            residual_variance * inverse_gram[level_index, level_index],
            jnp.finfo(data.dtype).tiny,
        )
    )
    statistic = coefficient / standard_error
    valid_result = (
        fit.valid
        & jnp.isfinite(statistic)
        & jnp.isfinite(residual_variance)
        & (standard_error > 0.0)
    )
    return UnitRootResult(
        level_coefficient=coefficient,
        standard_error=standard_error,
        statistic=statistic,
        critical_value=jnp.asarray(critical, dtype=data.dtype),
        stationary=valid_result & (statistic < critical),
        residual_variance=residual_variance,
        effective_sample_count=count,
        rank=fit.rank,
        condition_number=fit.condition_number,
        valid=valid_result,
        lag_differences=lags,
        include_intercept=bool(include_intercept),
    )


class VARModel(StrictModule):
    """Stable-vector-autoregression parameterization and forecast recurrence."""

    intercept: Array
    lag_matrices: Array
    innovation_covariance: Array
    companion_eigenvalues: Array
    spectral_radius: Array
    stable: Array
    order: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        intercept: ArrayLike,
        lag_matrices: ArrayLike,
        innovation_covariance: ArrayLike,
        /,
    ):
        intercept_ = jnp.asarray(intercept)
        matrices = jnp.asarray(lag_matrices)
        covariance = jnp.asarray(innovation_covariance)
        if intercept_.ndim != 1:
            raise ValueError("intercept must have shape (variables,).")
        dimension = int(intercept_.shape[0])
        if matrices.ndim != 3 or matrices.shape[1:] != (dimension, dimension):
            raise ValueError("lag_matrices must have shape (lags, variables, variables).")
        if matrices.shape[0] < 1:
            raise ValueError("VAR requires at least one lag.")
        if covariance.shape != (dimension, dimension):
            raise ValueError("innovation_covariance must be square over variables.")
        dtype = jnp.result_type(intercept_, matrices, covariance, float)
        intercept_ = intercept_.astype(dtype)
        matrices = matrices.astype(dtype)
        covariance = covariance.astype(dtype)
        invalid = (
            jnp.any(~jnp.isfinite(intercept_))
            | jnp.any(~jnp.isfinite(matrices))
            | jnp.any(~jnp.isfinite(covariance))
        )
        intercept_ = eqx.error_if(intercept_, invalid, "VAR parameters must be finite.")
        order = int(matrices.shape[0])
        companion = jnp.zeros((dimension * order, dimension * order), dtype=dtype)
        companion = companion.at[:dimension].set(
            jnp.concatenate(tuple(matrices[index] for index in range(order)), axis=-1)
        )
        if order > 1:
            companion = companion.at[dimension:, :-dimension].set(
                jnp.eye(dimension * (order - 1), dtype=dtype)
            )
        eigenvalues = jnp.linalg.eigvals(companion)
        radius = jnp.max(jnp.abs(eigenvalues))
        self.intercept = intercept_
        self.lag_matrices = matrices
        self.innovation_covariance = 0.5 * (covariance + covariance.T)
        self.companion_eigenvalues = eigenvalues
        self.spectral_radius = radius
        self.stable = radius < 1.0
        self.order = order
        self.dimension = dimension

    def forecast(self, history: ArrayLike, steps: int, /) -> Array:
        values = jnp.asarray(history, dtype=self.intercept.dtype)
        if (
            values.ndim != 2
            or values.shape[1] != self.dimension
            or values.shape[0] < self.order
        ):
            raise ValueError("history must have shape (time >= order, variables).")
        count = int(steps)
        if count < 1:
            raise ValueError("steps must be positive.")
        output = []
        extended = values
        for _ in range(count):
            next_value = self.intercept
            for lag in range(self.order):
                next_value = next_value + self.lag_matrices[lag] @ extended[-lag - 1]
            output.append(next_value)
            extended = jnp.concatenate((extended, next_value[None, :]), axis=0)
        return jnp.stack(output, axis=0)


class VARFit(StrictModule):
    """VAR fit with residual likelihood, rank, conditioning, and stability."""

    model: VARModel
    fitted_values: Array
    residuals: Array
    residual_mask: Array
    log_likelihood: Array
    effective_sample_count: Array
    rank: Array
    condition_number: Array
    normal_equation_error: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == TIME_SERIES_SUCCESS


def _multivariate_log_likelihood(
    residual: Array, valid: Array, covariance: Array
) -> Array:
    dimension = residual.shape[-1]
    values, vectors = jnp.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = jnp.finfo(values.dtype).eps * jnp.maximum(values[-1], 1.0)
    clipped = jnp.maximum(values, floor)
    precision = (vectors / clipped[None, :]) @ vectors.T
    log_determinant = jnp.sum(jnp.log(clipped))
    quadratic = ein.contract("ti,ij,tj->t", residual, precision, residual)
    terms = -0.5 * (dimension * jnp.log(2.0 * jnp.pi) + log_determinant + quadratic)
    return jnp.sum(jnp.where(valid, terms, 0.0))


def fit_var(
    values: ArrayLike,
    /,
    *,
    order: int,
    mask: ArrayLike | None = None,
    include_intercept: bool = True,
    ridge: float = 0.0,
    rcond: float | None = None,
) -> VARFit:
    """Fit a complete-row VAR with explicit irregular-mask propagation."""

    data, component_valid = _series(values, mask)
    if data.ndim != 2:
        raise ValueError("VAR values must have shape (time, variables).")
    order_ = int(order)
    if order_ < 1:
        raise ValueError("order must be positive.")
    time_count, dimension = data.shape
    feature_count = int(include_intercept) + order_ * dimension
    if time_count - order_ <= feature_count:
        raise ValueError("series is too short for the requested VAR order.")
    row_valid = jnp.all(component_valid, axis=-1)
    columns = []
    masks = []
    if include_intercept:
        columns.append(jnp.ones((time_count - order_, 1), dtype=data.dtype))
        masks.append(jnp.ones((time_count - order_,), dtype=bool))
    for lag in range(1, order_ + 1):
        columns.append(data[order_ - lag : time_count - lag])
        masks.append(row_valid[order_ - lag : time_count - lag])
    design = jnp.concatenate(columns, axis=-1)
    target = data[order_:]
    fit_mask = row_valid[order:] & jnp.all(jnp.stack(masks, axis=-1), axis=-1)
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
    intercept = (
        coefficients[0] if include_intercept else jnp.zeros((dimension,), data.dtype)
    )
    lag_matrices = jnp.stack(
        tuple(
            coefficients[offset + lag * dimension : offset + (lag + 1) * dimension].T
            for lag in range(order_)
        ),
        axis=0,
    )
    residual = jnp.where(fit_mask[:, None], least_squares.residual, 0.0)
    count = jnp.sum(fit_mask).astype(jnp.int32)
    covariance = residual.T @ residual / jnp.maximum(count, 1)
    covariance = 0.5 * (covariance + covariance.T)
    model = VARModel(intercept, lag_matrices, covariance)
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.all(jnp.isfinite(covariance))
    status = jnp.where(
        ~finite,
        TIME_SERIES_NONFINITE,
        jnp.where(
            count <= feature_count,
            TIME_SERIES_INSUFFICIENT,
            jnp.where(
                (float(ridge) == 0.0) & (least_squares.rank < feature_count),
                TIME_SERIES_RANK_DEFICIENT,
                TIME_SERIES_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return VARFit(
        model=model,
        fitted_values=least_squares.prediction,
        residuals=residual,
        residual_mask=fit_mask,
        log_likelihood=_multivariate_log_likelihood(residual, fit_mask, covariance),
        effective_sample_count=count,
        rank=least_squares.rank,
        condition_number=least_squares.condition_number,
        normal_equation_error=least_squares.normal_equation_error,
        status=status,
    )


class CointegrationResult(StrictModule):
    """Johansen eigenstructure and rank-selection evidence."""

    eigenvalues: Array
    cointegration_vectors: Array
    trace_statistics: Array
    critical_values: Array
    rejected_ranks: Array
    selected_rank: Array
    effective_sample_count: Array
    covariance_condition_number: Array
    valid: Array
    lag_differences: int = eqx.field(static=True)
    deterministic: str = eqx.field(static=True)


def test_cointegration(
    values: ArrayLike,
    /,
    *,
    lag_differences: int = 0,
    mask: ArrayLike | None = None,
    critical_values: ArrayLike | None = None,
    deterministic: Literal["constant", "none"] = "constant",
    ridge: float = 1e-10,
) -> CointegrationResult:
    """Compute Johansen trace statistics after residualizing short-run dynamics."""

    data, component_valid = _series(values, mask)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("cointegration values must have shape (time, variables >= 2).")
    lags = int(lag_differences)
    if lags < 0:
        raise ValueError("lag_differences must be nonnegative.")
    if deterministic not in ("constant", "none"):
        raise ValueError("deterministic must be 'constant' or 'none'.")
    time_count, dimension = data.shape
    start = lags + 1
    control_count = int(deterministic == "constant") + lags * dimension
    if time_count - start <= control_count + dimension:
        raise ValueError("series is too short for Johansen residualization.")
    row_valid = jnp.all(component_valid, axis=-1)
    difference = data[1:] - data[:-1]
    difference_valid = row_valid[1:] & row_valid[:-1]
    response_difference = difference[lags:]
    lagged_level = data[start - 1 : time_count - 1]
    target_valid = difference_valid[lags:] & row_valid[start - 1 : time_count - 1]
    controls = []
    control_masks = []
    if deterministic == "constant":
        controls.append(jnp.ones((time_count - start, 1), dtype=data.dtype))
        control_masks.append(jnp.ones((time_count - start,), dtype=bool))
    for lag in range(1, lags + 1):
        controls.append(difference[lags - lag : time_count - 1 - lag])
        control_masks.append(difference_valid[lags - lag : time_count - 1 - lag])
    if controls:
        control = jnp.concatenate(controls, axis=-1)
        fit_mask = target_valid & jnp.all(jnp.stack(control_masks, axis=-1), axis=-1)
        short_fit = solve_weighted_least_squares(
            control,
            response_difference,
            mask=fit_mask,
            ridge=ridge,
            min_samples=control.shape[-1] + dimension,
        )
        level_fit = solve_weighted_least_squares(
            control,
            lagged_level,
            mask=fit_mask,
            ridge=ridge,
            min_samples=control.shape[-1] + dimension,
        )
        r0 = jnp.where(fit_mask[:, None], short_fit.residual, 0.0)
        r1 = jnp.where(fit_mask[:, None], level_fit.residual, 0.0)
    else:
        control = jnp.zeros((time_count - start, 0), dtype=data.dtype)
        fit_mask = target_valid
        r0 = jnp.where(fit_mask[:, None], response_difference, 0.0)
        r1 = jnp.where(fit_mask[:, None], lagged_level, 0.0)
    count = jnp.sum(fit_mask).astype(jnp.int32)
    denominator = jnp.maximum(count, 1)
    s00 = r0.T @ r0 / denominator
    s11 = r1.T @ r1 / denominator
    s01 = r0.T @ r1 / denominator
    s10 = s01.T
    s00_inverse = dense_inverse(
        s00 + float(ridge) * jnp.eye(dimension, dtype=data.dtype),
        positive_definite=True,
    )
    s11_values, s11_vectors = jnp.linalg.eigh(
        0.5 * (s11 + s11.T) + float(ridge) * jnp.eye(dimension, dtype=data.dtype)
    )
    s11_floor = jnp.maximum(s11_values, jnp.finfo(data.dtype).eps)
    inverse_root = (s11_vectors / jnp.sqrt(s11_floor)[None, :]) @ s11_vectors.T
    eigen_problem = inverse_root @ s10 @ s00_inverse @ s01 @ inverse_root
    eigen_problem = 0.5 * (eigen_problem + eigen_problem.T)
    eigenvalues, whitened_vectors = jnp.linalg.eigh(eigen_problem)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.clip(eigenvalues[order], 0.0, 1.0 - jnp.finfo(data.dtype).eps)
    beta = inverse_root @ whitened_vectors[:, order]
    norms = jnp.linalg.norm(beta, axis=0)
    beta = beta / jnp.maximum(norms, jnp.finfo(data.dtype).tiny)
    pivot = jnp.argmax(jnp.abs(beta), axis=0)
    signs = jnp.sign(beta[pivot, jnp.arange(dimension)])
    beta = beta * jnp.where(signs == 0.0, 1.0, signs)[None, :]
    log_terms = -count.astype(data.dtype) * jnp.log1p(-eigenvalues)
    trace_statistics = jnp.cumsum(log_terms[::-1])[::-1]
    if critical_values is None:
        critical = jnp.full((dimension,), jnp.nan, dtype=data.dtype)
        threshold = 1.0 / jnp.sqrt(jnp.maximum(count, 1).astype(data.dtype))
        rejected = eigenvalues > threshold
    else:
        critical = jnp.asarray(critical_values, dtype=data.dtype)
        if critical.shape != (dimension,):
            raise ValueError("critical_values must have one value per candidate rank.")
        critical = eqx.error_if(
            critical,
            jnp.any(~jnp.isfinite(critical) | (critical <= 0.0)),
            "critical_values must be finite and positive.",
        )
        rejected = trace_statistics > critical
    sequential = jnp.cumprod(rejected.astype(jnp.int32)).astype(bool)
    selected = jnp.sum(sequential).astype(jnp.int32)
    condition = s11_floor[-1] / s11_floor[0]
    valid_result = (
        (count > control.shape[-1] + dimension)
        & jnp.all(jnp.isfinite(eigenvalues))
        & jnp.isfinite(condition)
    )
    return CointegrationResult(
        eigenvalues=eigenvalues,
        cointegration_vectors=beta,
        trace_statistics=trace_statistics,
        critical_values=critical,
        rejected_ranks=rejected,
        selected_rank=selected,
        effective_sample_count=count,
        covariance_condition_number=condition,
        valid=valid_result,
        lag_differences=lags,
        deterministic=deterministic,
    )


class VECMModel(StrictModule):
    """Vector error-correction model with identified cointegration vectors."""

    intercept: Array
    adjustment: Array
    cointegration_vectors: Array
    short_run_matrices: Array
    innovation_covariance: Array
    level_companion_eigenvalues: Array
    unit_root_count: Array
    stable_root_count: Array
    rank: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    lag_differences: int = eqx.field(static=True)

    def forecast(self, history: ArrayLike, steps: int, /) -> Array:
        values = jnp.asarray(history, dtype=self.intercept.dtype)
        required = self.lag_differences + 1
        if (
            values.ndim != 2
            or values.shape[1] != self.dimension
            or values.shape[0] < required
        ):
            raise ValueError(
                "history must have shape (time >= lag_differences + 1, variables)."
            )
        count = int(steps)
        if count < 1:
            raise ValueError("steps must be positive.")
        extended = values
        output = []
        for _ in range(count):
            correction = self.adjustment @ (self.cointegration_vectors.T @ extended[-1])
            change = self.intercept + correction
            for lag in range(self.lag_differences):
                previous_change = extended[-lag - 1] - extended[-lag - 2]
                change = change + self.short_run_matrices[lag] @ previous_change
            next_level = extended[-1] + change
            extended = jnp.concatenate((extended, next_level[None, :]), axis=0)
            output.append(next_level)
        return jnp.stack(output)


class VECMFit(StrictModule):
    """VECM fit retaining the complete rank and likelihood evidence."""

    model: VECMModel
    cointegration: CointegrationResult
    fitted_differences: Array
    residuals: Array
    residual_mask: Array
    log_likelihood: Array
    effective_sample_count: Array
    design_rank: Array
    condition_number: Array
    status: Array

    @property
    def successful(self) -> Array:
        return self.status == TIME_SERIES_SUCCESS


def fit_vecm(
    values: ArrayLike,
    /,
    *,
    rank: int,
    lag_differences: int = 0,
    mask: ArrayLike | None = None,
    include_intercept: bool = True,
    ridge: float = 1e-10,
    rcond: float | None = None,
) -> VECMFit:
    """Fit a VECM conditional on an explicit, previously interpretable rank."""

    data, component_valid = _series(values, mask)
    if data.ndim != 2:
        raise ValueError("VECM values must have shape (time, variables).")
    dimension = int(data.shape[1])
    rank_ = int(rank)
    lags = int(lag_differences)
    if not 1 <= rank_ < dimension:
        raise ValueError("rank must lie between one and variables - one.")
    if lags < 0:
        raise ValueError("lag_differences must be nonnegative.")
    cointegration = test_cointegration(
        data,
        lag_differences=lags,
        mask=component_valid,
        deterministic="constant" if include_intercept else "none",
        ridge=ridge,
    )
    beta = cointegration.cointegration_vectors[:, :rank_]
    time_count = data.shape[0]
    start = lags + 1
    difference = data[1:] - data[:-1]
    row_valid = jnp.all(component_valid, axis=-1)
    difference_valid = row_valid[1:] & row_valid[:-1]
    target = difference[lags:]
    lagged_level = data[start - 1 : time_count - 1]
    columns = []
    masks = []
    if include_intercept:
        columns.append(jnp.ones((time_count - start, 1), dtype=data.dtype))
        masks.append(jnp.ones((time_count - start,), dtype=bool))
    columns.append(lagged_level @ beta)
    masks.append(row_valid[start - 1 : time_count - 1])
    for lag in range(1, lags + 1):
        columns.append(difference[lags - lag : time_count - 1 - lag])
        masks.append(difference_valid[lags - lag : time_count - 1 - lag])
    design = jnp.concatenate(columns, axis=-1)
    fit_mask = difference_valid[lags:] & jnp.all(jnp.stack(masks, axis=-1), axis=-1)
    least_squares = solve_weighted_least_squares(
        design,
        target,
        mask=fit_mask,
        ridge=ridge,
        rcond=rcond,
        min_samples=design.shape[-1] + dimension,
    )
    coefficients = least_squares.raw_coefficients
    offset = 1 if include_intercept else 0
    intercept = (
        coefficients[0] if include_intercept else jnp.zeros((dimension,), data.dtype)
    )
    adjustment = coefficients[offset : offset + rank_].T
    short_run = (
        jnp.stack(
            tuple(
                coefficients[
                    offset + rank_ + lag * dimension : offset
                    + rank_
                    + (lag + 1) * dimension
                ].T
                for lag in range(lags)
            ),
            axis=0,
        )
        if lags
        else jnp.zeros((0, dimension, dimension), dtype=data.dtype)
    )
    residual = jnp.where(fit_mask[:, None], least_squares.residual, 0.0)
    count = jnp.sum(fit_mask).astype(jnp.int32)
    covariance = residual.T @ residual / jnp.maximum(count, 1)
    covariance = 0.5 * (covariance + covariance.T)

    level_order = lags + 1
    level_matrices = []
    correction = adjustment @ beta.T
    if lags == 0:
        level_matrices.append(jnp.eye(dimension, dtype=data.dtype) + correction)
    else:
        level_matrices.append(
            jnp.eye(dimension, dtype=data.dtype) + correction + short_run[0]
        )
        for lag in range(1, lags):
            level_matrices.append(short_run[lag] - short_run[lag - 1])
        level_matrices.append(-short_run[-1])
    companion = jnp.zeros(
        (dimension * level_order, dimension * level_order), dtype=data.dtype
    )
    companion = companion.at[:dimension].set(
        jnp.concatenate(tuple(level_matrices), axis=-1)
    )
    if level_order > 1:
        companion = companion.at[dimension:, :-dimension].set(
            jnp.eye(dimension * (level_order - 1), dtype=data.dtype)
        )
    roots = jnp.linalg.eigvals(companion)
    modulus = jnp.abs(roots)
    tolerance = 128.0 * jnp.sqrt(jnp.finfo(data.dtype).eps)
    unit_roots = jnp.sum(jnp.abs(modulus - 1.0) <= tolerance).astype(jnp.int32)
    stable_roots = jnp.sum(modulus < 1.0 - tolerance).astype(jnp.int32)
    model = VECMModel(
        intercept=intercept,
        adjustment=adjustment,
        cointegration_vectors=beta,
        short_run_matrices=short_run,
        innovation_covariance=covariance,
        level_companion_eigenvalues=roots,
        unit_root_count=unit_roots,
        stable_root_count=stable_roots,
        rank=rank_,
        dimension=dimension,
        lag_differences=lags,
    )
    finite = jnp.all(jnp.isfinite(coefficients)) & jnp.all(jnp.isfinite(covariance))
    status = jnp.where(
        ~finite,
        TIME_SERIES_NONFINITE,
        jnp.where(
            count <= design.shape[-1],
            TIME_SERIES_INSUFFICIENT,
            jnp.where(
                (float(ridge) == 0.0) & (least_squares.rank < design.shape[-1]),
                TIME_SERIES_RANK_DEFICIENT,
                TIME_SERIES_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    return VECMFit(
        model=model,
        cointegration=cointegration,
        fitted_differences=least_squares.prediction,
        residuals=residual,
        residual_mask=fit_mask,
        log_likelihood=_multivariate_log_likelihood(residual, fit_mask, covariance),
        effective_sample_count=count,
        design_rank=least_squares.rank,
        condition_number=least_squares.condition_number,
        status=status,
    )


__all__ = [
    "TIME_SERIES_INSUFFICIENT",
    "TIME_SERIES_NONFINITE",
    "TIME_SERIES_RANK_DEFICIENT",
    "TIME_SERIES_SUCCESS",
    "ARIMAFit",
    "ARIMAModel",
    "CointegrationResult",
    "TimeSeriesFitStatus",
    "UnitRootResult",
    "VARFit",
    "VARModel",
    "VECMFit",
    "VECMModel",
    "augmented_dickey_fuller",
    "fit_arima",
    "fit_var",
    "fit_vecm",
    "test_cointegration",
]
