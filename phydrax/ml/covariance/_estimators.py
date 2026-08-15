#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._model import AbstractArrayModel
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_RANK_DEFICIENT,
    ML_SUCCESS,
)


def _real_dtype(dtype: jnp.dtype) -> jnp.dtype:
    return jnp.empty((), dtype=dtype).real.dtype


def _hermitian(matrix: Array) -> Array:
    return (matrix + jnp.conj(jnp.swapaxes(matrix, -1, -2))) * 0.5


@jax.custom_jvp
def _stable_eigvalsh(matrix: Array) -> Array:
    return jnp.linalg.eigvalsh(_hermitian(matrix))


@_stable_eigvalsh.defjvp
def _stable_eigvalsh_jvp(primals, tangents):
    (matrix,), (matrix_tangent,) = primals, tangents
    values, vectors = jnp.linalg.eigh(_hermitian(matrix))
    tangent_basis = oe.contract(
        "...ki,...kl,...lj->...ij",
        jnp.conj(vectors),
        _hermitian(matrix_tangent),
        vectors,
    )
    return values, jnp.real(jnp.diagonal(tangent_basis, axis1=-2, axis2=-1))


@jax.custom_jvp
def _spectral_floor(matrix: Array, floor: Array) -> tuple[Array, Array]:
    values, vectors = jnp.linalg.eigh(_hermitian(matrix))
    floor_ = jnp.asarray(floor, dtype=values.dtype)
    clipped = jnp.maximum(values, floor_[..., None])
    projected = oe.contract(
        "...ik,...k,...jk->...ij",
        vectors,
        clipped,
        jnp.conj(vectors),
    )
    return projected, values


@_spectral_floor.defjvp
def _spectral_floor_jvp(primals, tangents):
    (matrix, floor), (matrix_tangent, floor_tangent) = primals, tangents
    matrix = _hermitian(matrix)
    matrix_tangent = _hermitian(matrix_tangent)
    values, vectors = jnp.linalg.eigh(matrix)
    floor = jnp.asarray(floor, dtype=values.dtype)
    floor_tangent = jnp.asarray(floor_tangent, dtype=values.dtype)
    clipped = jnp.maximum(values, floor[..., None])
    active = values > floor[..., None]
    active_i = active[..., :, None]
    active_j = active[..., None, :]
    crossed = active_i != active_j
    value_difference = values[..., :, None] - values[..., None, :]
    clipped_difference = clipped[..., :, None] - clipped[..., None, :]
    safe_difference = jnp.where(crossed, value_difference, 1.0)
    coefficient = jnp.where(
        active_i & active_j,
        1.0,
        jnp.where(crossed, clipped_difference / safe_difference, 0.0),
    )
    tangent_basis = oe.contract(
        "...ki,...kl,...lj->...ij",
        jnp.conj(vectors),
        matrix_tangent,
        vectors,
    )
    eigenvalue_tangent = jnp.real(jnp.diagonal(tangent_basis, axis1=-2, axis2=-1))
    floor_diagonal = (~active).astype(values.dtype) * floor_tangent[..., None]
    tangent_basis = coefficient * tangent_basis
    tangent_basis = tangent_basis + floor_diagonal[..., None, :] * jnp.eye(
        matrix.shape[-1], dtype=matrix.dtype
    )
    projected = oe.contract(
        "...ik,...k,...jk->...ij",
        vectors,
        clipped,
        jnp.conj(vectors),
    )
    projected_tangent = oe.contract(
        "...ik,...kl,...jl->...ij",
        vectors,
        tangent_basis,
        jnp.conj(vectors),
    )
    return (projected, values), (projected_tangent, eigenvalue_tangent)


@jax.custom_jvp
def _frobenius_norm(matrix: Array) -> Array:
    return jnp.sqrt(jnp.sum(jnp.real(jnp.conj(matrix) * matrix), axis=(-2, -1)))


@_frobenius_norm.defjvp
def _frobenius_norm_jvp(primals, tangents):
    (matrix,), (matrix_tangent,) = primals, tangents
    norm = _frobenius_norm(matrix)
    inner = jnp.sum(jnp.real(jnp.conj(matrix) * matrix_tangent), axis=(-2, -1))
    positive = norm > 0.0
    safe_norm = jnp.where(positive, norm, 1.0)
    return norm, jnp.where(positive, inner / safe_norm, 0.0)


def _validated_scalar(value: Any, name: str, /, *, allow_zero: bool) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    relation = "nonnegative" if allow_zero else "positive"
    invalid = ~jnp.isfinite(scalar) | (scalar < 0.0 if allow_zero else scalar <= 0.0)
    return eqx.error_if(scalar, invalid, f"{name} must be finite and {relation}.")


def _nonnegative_scalar(value: Any, name: str, /) -> Array:
    return _validated_scalar(value, name, allow_zero=True)


def _positive_scalar(value: Any, name: str, /) -> Array:
    return _validated_scalar(value, name, allow_zero=False)


def _active_data(batch: MLBatch, policy: WeightPolicy) -> tuple[Array, Array, Array]:
    x = batch.dense_features()
    complete = jnp.all(batch.feature_mask, axis=-1)
    raw = batch.effective_weight(policy)
    weights_ok = jnp.isfinite(raw) & (raw >= 0.0)
    finite_x = jnp.all(jnp.isfinite(x), axis=-1)
    active = complete & finite_x & weights_ok
    w = jnp.where(active, raw, 0.0).astype(_real_dtype(x.dtype))
    x = jnp.where(active[..., None], x, jnp.zeros((), dtype=x.dtype))
    invalid = jnp.any(batch.sample_mask & (~weights_ok | (complete & ~finite_x)), axis=-1)
    return x, w, invalid


def _moments(
    x: Array, w: Array, correction: Array | float
) -> tuple[Array, Array, Array, Array, Array]:
    real_dtype = _real_dtype(x.dtype)
    tiny = jnp.finfo(real_dtype).tiny
    mass = jnp.sum(w, axis=-1)
    square_mass = jnp.sum(jnp.square(w), axis=-1)
    safe_mass = jnp.maximum(mass, tiny)
    mean = oe.contract("...n,...nf->...f", w, x) / safe_mass[..., None]
    centered = jnp.where(w[..., None] > 0.0, x - mean[..., None, :], 0)
    scatter = oe.contract("...ni,...n,...nj->...ij", jnp.conj(centered), w, centered)
    denominator = mass - correction * square_mass / safe_mass
    safe_denominator = jnp.maximum(denominator, tiny)
    covariance = scatter / safe_denominator[..., None, None]
    valid = (mass > 0.0) & (denominator > 0.0)
    return mean, covariance, mass, square_mass, valid


def _regularize(
    covariance: Array, regularization: Array
) -> tuple[Array, Array, Array, Array]:
    feature_count = covariance.shape[-1]
    real_dtype = _real_dtype(covariance.dtype)
    eye = jnp.eye(feature_count, dtype=covariance.dtype)
    hermitian = _hermitian(covariance)
    scale = jnp.maximum(
        jnp.real(jnp.trace(hermitian, axis1=-2, axis2=-1)) / feature_count,
        1.0,
    )
    shift = jnp.asarray(regularization, dtype=real_dtype) * scale
    floor = jnp.maximum(shift, jnp.finfo(real_dtype).eps * scale)
    shifted = hermitian + shift[..., None, None] * eye
    covariance_, eigenvalues = _spectral_floor(shifted, floor)
    precision = jnp.linalg.solve(covariance_, eye)
    log_det = jnp.linalg.slogdet(covariance_)[1]
    clipped = jnp.maximum(eigenvalues, floor[..., None])
    rank = jnp.sum(eigenvalues > floor[..., None], axis=-1).astype(jnp.int32)
    condition = clipped[..., -1] / clipped[..., 0]
    return covariance_, precision, log_det, jnp.stack((rank, condition), axis=-1)


def _input_sample_ndim(values: Array, case_shape: tuple[int, ...], in_size: int) -> int:
    minimum_rank = len(case_shape) + 1
    if values.ndim < minimum_rank:
        raise ValueError("case axes and the final feature axis must be distinct.")
    if values.shape[: len(case_shape)] != case_shape or values.shape[-1] != in_size:
        raise ValueError("input must have shape case + sample_shape + (feature,).")
    return values.ndim - minimum_rank


class CovarianceModel(AbstractArrayModel):
    """Immutable Gaussian covariance geometry; calls return squared Mahalanobis distance."""

    mean: Array
    covariance: Array
    precision: Array
    log_determinant: Array
    factor_loadings: Array
    diagonal: Array
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)

    def __init__(
        self,
        mean: Array,
        covariance: Array,
        precision: Array,
        log_determinant: Array,
        /,
        *,
        factor_loadings: Array | None = None,
        diagonal: Array | None = None,
        method: str,
    ):
        self.mean = jnp.asarray(mean)
        self.covariance = jnp.asarray(covariance)
        self.precision = jnp.asarray(precision)
        self.log_determinant = jnp.asarray(log_determinant)
        feature_count = self.mean.shape[-1]
        case_shape = self.mean.shape[:-1]
        self.factor_loadings = (
            jnp.zeros(case_shape + (feature_count, 0), dtype=self.mean.dtype)
            if factor_loadings is None
            else jnp.asarray(factor_loadings)
        )
        self.diagonal = (
            jnp.real(jnp.diagonal(self.covariance, axis1=-2, axis2=-1))
            if diagonal is None
            else jnp.asarray(diagonal)
        )
        self.in_size = feature_count
        self.out_size = "scalar"
        self.case_shape = case_shape
        self.method = str(method)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        values = jnp.asarray(x)
        sample_ndim = _input_sample_ndim(values, self.case_shape, self.in_size)
        mean = self.mean.reshape(self.case_shape + (1,) * sample_ndim + (self.in_size,))
        precision = self.precision.reshape(
            self.case_shape + (1,) * sample_ndim + (self.in_size, self.in_size)
        )
        centered = values - mean
        return jnp.real(
            oe.contract("...i,...ij,...j->...", jnp.conj(centered), precision, centered)
        )

    def log_density(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        sample_ndim = _input_sample_ndim(values, self.case_shape, self.in_size)
        distance = self(values)
        log_determinant = self.log_determinant.reshape(
            self.case_shape + (1,) * sample_ndim
        )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            return -(distance + self.in_size * jnp.log(jnp.pi) + log_determinant)
        return -0.5 * (distance + self.in_size * jnp.log(2.0 * jnp.pi) + log_determinant)

    def whiten(self, x: Any, /) -> Array:
        values = jnp.asarray(x)
        sample_ndim = _input_sample_ndim(values, self.case_shape, self.in_size)
        root = jnp.linalg.cholesky(self.precision)
        root = root.reshape(
            self.case_shape + (1,) * sample_ndim + (self.in_size, self.in_size)
        )
        mean = self.mean.reshape(self.case_shape + (1,) * sample_ndim + (self.in_size,))
        return oe.contract("...i,...ij->...j", values - mean, jnp.conj(root))


def _result(
    batch: MLBatch,
    *,
    mean: Array,
    covariance: Array,
    mass: Array,
    squared_mass: Array,
    valid_moments: Array,
    invalid_input: Array,
    regularization: Array,
    method: str,
    iterations: Any = 1,
    converged: Array | None = None,
    factor_loadings: Array | None = None,
    diagonal: Array | None = None,
    precision_override: Array | None = None,
    fit_mode: Literal["direct", "unrolled"] = "direct",
) -> FitResult:
    if precision_override is None:
        covariance_, precision, log_det, rank_condition = _regularize(
            covariance, regularization
        )
    else:
        precision = _hermitian(jnp.asarray(precision_override))
        eye = jnp.eye(precision.shape[-1], dtype=precision.dtype)
        covariance_ = jnp.linalg.solve(precision, eye)
        precision_values = _stable_eigvalsh(precision)
        floor = jnp.finfo(_real_dtype(precision.dtype)).eps * jnp.maximum(
            precision_values[..., -1], 1.0
        )
        rank_override = jnp.sum(precision_values > floor[..., None], axis=-1).astype(
            jnp.int32
        )
        condition_override = precision_values[..., -1] / jnp.maximum(
            precision_values[..., 0], floor
        )
        log_det = -jnp.linalg.slogdet(precision)[1]
        rank_condition = jnp.stack((rank_override, condition_override), axis=-1)
    rank = rank_condition[..., 0].astype(jnp.int32)
    condition = rank_condition[..., 1]
    finite = (
        jnp.all(jnp.isfinite(covariance_), axis=(-2, -1))
        & jnp.all(jnp.isfinite(precision), axis=(-2, -1))
        & jnp.all(jnp.isfinite(mean), axis=-1)
        & jnp.isfinite(log_det)
    )
    converged_ = jnp.ones_like(valid_moments) if converged is None else converged
    valid = valid_moments & finite & converged_ & ~invalid_input
    status = jnp.where(
        invalid_input | ~finite,
        ML_NONFINITE,
        jnp.where(
            ~valid_moments,
            ML_INSUFFICIENT_DATA,
            jnp.where(
                ~converged_,
                ML_NONCONVERGED,
                jnp.where(rank < batch.feature_count, ML_RANK_DEFICIENT, ML_SUCCESS),
            ),
        ),
    )
    objective = log_det + jnp.real(jnp.trace(covariance_ @ precision, axis1=-2, axis2=-1))
    positive_squared_mass = squared_mass > 0.0
    safe_squared_mass = jnp.where(positive_squared_mass, squared_mass, 1.0)
    diagnostics = FitDiagnostics(
        valid=valid,
        status=status,
        objective=objective,
        iterations=iterations,
        effective_samples=jnp.where(
            positive_squared_mass, mass * mass / safe_squared_mass, 0.0
        ),
        rank=rank,
        condition=condition,
        method=method,
    )
    model = CovarianceModel(
        mean,
        covariance_,
        precision,
        log_det,
        factor_loadings=factor_loadings,
        diagonal=diagonal,
        method=method,
    )
    gradient = GradientContract(
        prediction_inputs="smooth",
        prediction_parameters="smooth",
        fit_features="conditional",
        fit_weights="conditional",
        fit_mode=fit_mode,
        conditions=("fixed active mask", "positive regularized covariance"),
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=gradient,
    )


class EmpiricalCovariance(AbstractRecipe):
    correction: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        correction: float = 0.0,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.correction = _nonnegative_scalar(correction, "correction")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, w, invalid = _active_data(batch, self.weight_policy)
        mean, covariance, mass, squared_mass, valid = _moments(x, w, self.correction)
        return _result(
            batch,
            mean=mean,
            covariance=covariance,
            mass=mass,
            squared_mass=squared_mass,
            valid_moments=valid,
            invalid_input=invalid,
            regularization=self.regularization,
            method="empirical-covariance",
        )


class WeightedCovariance(AbstractRecipe):
    """Hermitian covariance using explicit nonnegative batch weights."""

    correction: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        correction: float = 1.0,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        if weight_policy == "none":
            raise ValueError("WeightedCovariance requires a weighted batch policy.")
        self.correction = _nonnegative_scalar(correction, "correction")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, w, invalid = _active_data(batch, self.weight_policy)
        mean, covariance, mass, squared_mass, valid = _moments(x, w, self.correction)
        return _result(
            batch,
            mean=mean,
            covariance=covariance,
            mass=mass,
            squared_mass=squared_mass,
            valid_moments=valid,
            invalid_input=invalid,
            regularization=self.regularization,
            method="weighted-covariance",
        )


class DiagonalCovariance(AbstractRecipe):
    correction: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        correction: float = 0.0,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.correction = _nonnegative_scalar(correction, "correction")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, w, invalid = _active_data(batch, self.weight_policy)
        mean, covariance, mass, squared_mass, valid = _moments(x, w, self.correction)
        diagonal = jnp.real(jnp.diagonal(covariance, axis1=-2, axis2=-1))
        covariance = (
            jnp.eye(batch.feature_count, dtype=covariance.dtype) * diagonal[..., None, :]
        )
        return _result(
            batch,
            mean=mean,
            covariance=covariance,
            mass=mass,
            squared_mass=squared_mass,
            valid_moments=valid,
            invalid_input=invalid,
            regularization=self.regularization,
            diagonal=diagonal,
            method="diagonal-covariance",
        )


class FactorCovariance(AbstractRecipe):
    rank: int = eqx.field(static=True)
    correction: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        rank: int,
        /,
        *,
        correction: float = 0.0,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.rank = int(rank)
        self.correction = _nonnegative_scalar(correction, "correction")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.rank > batch.feature_count:
            raise ValueError("rank cannot exceed feature count.")
        x, w, invalid = _active_data(batch, self.weight_policy)
        mean, covariance, mass, squared_mass, valid = _moments(x, w, self.correction)
        values, vectors = jnp.linalg.eigh(
            (covariance + jnp.conj(jnp.swapaxes(covariance, -1, -2))) * 0.5
        )
        values = values[..., -self.rank :]
        vectors = vectors[..., :, -self.rank :]
        loadings = vectors * jnp.sqrt(jnp.maximum(values, 0.0))[..., None, :]
        low_rank = loadings @ jnp.conj(jnp.swapaxes(loadings, -1, -2))
        diagonal = jnp.maximum(
            jnp.real(jnp.diagonal(covariance - low_rank, axis1=-2, axis2=-1)), 0.0
        )
        reconstructed = (
            low_rank
            + jnp.eye(batch.feature_count, dtype=covariance.dtype)
            * diagonal[..., None, :]
        )
        return _result(
            batch,
            mean=mean,
            covariance=reconstructed,
            mass=mass,
            squared_mass=squared_mass,
            valid_moments=valid,
            invalid_input=invalid,
            regularization=self.regularization,
            factor_loadings=loadings,
            diagonal=diagonal,
            method="factor-covariance",
        )


def _shrinkage_fit(
    batch: MLBatch,
    policy: WeightPolicy,
    correction: Array,
    regularization: Array,
    method: str,
) -> FitResult:
    x, w, invalid = _active_data(batch, policy)
    mean, covariance, mass, square_mass, valid = _moments(x, w, correction)
    p = batch.feature_count
    mu = jnp.real(jnp.trace(covariance, axis1=-2, axis2=-1)) / p
    target = mu[..., None, None] * jnp.eye(p, dtype=covariance.dtype)
    positive_square_mass = square_mass > 0.0
    safe_square_mass = jnp.where(positive_square_mass, square_mass, 1.0)
    n_eff = jnp.where(positive_square_mass, mass * mass / safe_square_mass, 0.0)
    if method == "ledoit-wolf":
        centered = jnp.where(w[..., None] > 0.0, x - mean[..., None, :], 0)
        outer = oe.contract("...ni,...nj->...nij", jnp.conj(centered), centered)
        residual = outer - covariance[..., None, :, :]
        beta = oe.contract(
            "...n,...nij,...nij->...",
            w * w,
            jnp.conj(residual),
            residual,
        ).real
        mass_squared = mass * mass
        beta = beta / jnp.where(mass_squared > 0.0, mass_squared, 1.0)
        delta = jnp.sum(
            jnp.real(jnp.conj(covariance - target) * (covariance - target)),
            axis=(-2, -1),
        )
        separated = delta > jnp.finfo(_real_dtype(x.dtype)).tiny
        ratio = jnp.where(separated, beta / jnp.where(separated, delta, 1.0), 0.0)
        shrinkage = jnp.clip(ratio, 0.0, 1.0)
    else:
        trace_s2 = jnp.real(jnp.sum(jnp.conj(covariance) * covariance, axis=(-2, -1)))
        trace_s = jnp.real(jnp.trace(covariance, axis1=-2, axis2=-1))
        numerator = (1.0 - 2.0 / p) * trace_s2 + trace_s * trace_s
        denominator = (n_eff + 1.0 - 2.0 / p) * (trace_s2 - trace_s * trace_s / p)
        separated = denominator > jnp.finfo(_real_dtype(x.dtype)).tiny
        ratio = jnp.where(
            separated,
            numerator / jnp.where(separated, denominator, 1.0),
            1.0,
        )
        shrinkage = jnp.clip(ratio, 0.0, 1.0)
    shrunk = (1.0 - shrinkage[..., None, None]) * covariance + shrinkage[
        ..., None, None
    ] * target
    return _result(
        batch,
        mean=mean,
        covariance=shrunk,
        mass=mass,
        squared_mass=square_mass,
        valid_moments=valid,
        invalid_input=invalid,
        regularization=regularization,
        method=method,
    )


class LedoitWolfCovariance(AbstractRecipe):
    correction: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        correction: float = 0.0,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.correction = _nonnegative_scalar(correction, "correction")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _shrinkage_fit(
            batch, self.weight_policy, self.correction, self.regularization, "ledoit-wolf"
        )


class OASCovariance(AbstractRecipe):
    correction: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        correction: float = 0.0,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.correction = _nonnegative_scalar(correction, "correction")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _shrinkage_fit(
            batch, self.weight_policy, self.correction, self.regularization, "oas"
        )


class RobustCovariance(AbstractRecipe):
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    huber_delta: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_iterations: int = 32,
        tolerance: float = 1e-5,
        huber_delta: float = 2.5,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        self.max_iterations = int(max_iterations)
        self.tolerance = _nonnegative_scalar(tolerance, "tolerance")
        self.huber_delta = _positive_scalar(huber_delta, "huber_delta")
        self.regularization = _nonnegative_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, base_w, invalid = _active_data(batch, self.weight_policy)
        mean0, covariance0, mass, squared_mass, valid0 = _moments(x, base_w, 0.0)

        def step(_, state):
            mean, covariance, delta, iteration = state
            _, precision, _, _ = _regularize(covariance, self.regularization)
            centered = x - mean[..., None, :]
            squared_distance = jnp.real(
                oe.contract(
                    "...ni,...ij,...nj->...n",
                    jnp.conj(centered),
                    precision,
                    centered,
                )
            )
            outlier = squared_distance > jnp.square(self.huber_delta)
            safe_squared_distance = jnp.where(outlier, squared_distance, 1.0)
            robust = jnp.where(
                outlier,
                self.huber_delta / jnp.sqrt(safe_squared_distance),
                1.0,
            )
            weights = base_w * robust
            next_mean, next_covariance, _, _, _ = _moments(x, weights, 0.0)
            numerator = _frobenius_norm(next_covariance - covariance)
            denominator = jnp.maximum(
                _frobenius_norm(covariance),
                jnp.finfo(_real_dtype(x.dtype)).eps,
            )
            relative = numerator / denominator
            active = delta >= self.tolerance
            mean = jnp.where(active[..., None], next_mean, mean)
            covariance = jnp.where(active[..., None, None], next_covariance, covariance)
            delta = jnp.where(active, relative, delta)
            iteration = iteration + active.astype(jnp.int32)
            return mean, covariance, delta, iteration

        case_shape = x.shape[:-2]
        state = (
            mean0,
            covariance0,
            jnp.full(case_shape, jnp.inf, dtype=_real_dtype(x.dtype)),
            jnp.zeros(case_shape, dtype=jnp.int32),
        )
        mean, covariance, delta, iterations = jax.lax.fori_loop(
            0, self.max_iterations, step, state
        )
        converged = delta < self.tolerance
        return _result(
            batch,
            mean=mean,
            covariance=covariance,
            mass=mass,
            squared_mass=squared_mass,
            valid_moments=valid0,
            invalid_input=invalid,
            regularization=self.regularization,
            iterations=iterations,
            converged=converged,
            method="robust-covariance",
            fit_mode="unrolled",
        )


class GraphicalLasso(AbstractRecipe):
    penalty: Array
    max_iterations: int = eqx.field(static=True)
    tolerance: Array
    step_size: Array
    regularization: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        penalty: float = 0.01,
        max_iterations: int = 64,
        tolerance: float = 1e-5,
        step_size: float = 0.1,
        regularization: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        self.penalty = _nonnegative_scalar(penalty, "penalty")
        self.max_iterations = int(max_iterations)
        self.tolerance = _nonnegative_scalar(tolerance, "tolerance")
        self.step_size = _positive_scalar(step_size, "step_size")
        self.regularization = _positive_scalar(regularization, "regularization")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x, w, invalid = _active_data(batch, self.weight_policy)
        mean, sample_covariance, mass, squared_mass, valid0 = _moments(x, w, 0.0)
        covariance0, precision0, _, _ = _regularize(
            sample_covariance, self.regularization
        )
        p = batch.feature_count
        eye = jnp.eye(p, dtype=precision0.dtype)

        def step(_, state):
            precision, delta, iteration = state
            covariance = jnp.linalg.solve(precision, eye)
            proposal = precision - self.step_size * (sample_covariance - covariance)
            diagonal = jnp.diagonal(proposal, axis1=-2, axis2=-1)
            off = proposal - diagonal[..., :, None] * eye
            magnitude = jnp.abs(off)
            threshold = self.step_size * self.penalty
            selected = magnitude > threshold
            safe_magnitude = jnp.where(selected, magnitude, 1.0)
            shrunk = jnp.where(
                selected,
                off * (1.0 - threshold / safe_magnitude),
                jnp.zeros_like(off),
            )
            proposal = shrunk + diagonal[..., :, None] * eye
            floor = jnp.asarray(self.regularization, dtype=_real_dtype(proposal.dtype))
            proposal, _ = _spectral_floor(proposal, floor)
            numerator = _frobenius_norm(proposal - precision)
            denominator = jnp.maximum(
                _frobenius_norm(precision),
                jnp.finfo(_real_dtype(precision.dtype)).eps,
            )
            relative = numerator / denominator
            active = delta >= self.tolerance
            precision = jnp.where(active[..., None, None], proposal, precision)
            delta = jnp.where(active, relative, delta)
            iteration = iteration + active.astype(jnp.int32)
            return precision, delta, iteration

        case_shape = x.shape[:-2]
        state = (
            precision0,
            jnp.full(case_shape, jnp.inf, dtype=_real_dtype(x.dtype)),
            jnp.zeros(case_shape, dtype=jnp.int32),
        )
        precision, delta, iterations = jax.lax.fori_loop(
            0, self.max_iterations, step, state
        )
        covariance = jnp.linalg.solve(precision, eye)
        converged = delta < self.tolerance
        return _result(
            batch,
            mean=mean,
            covariance=covariance,
            mass=mass,
            squared_mass=squared_mass,
            valid_moments=valid0,
            invalid_input=invalid,
            regularization=self.regularization,
            iterations=iterations,
            converged=converged,
            precision_override=precision,
            method="graphical-lasso",
            fit_mode="unrolled",
        )


__all__ = [
    "CovarianceModel",
    "DiagonalCovariance",
    "EmpiricalCovariance",
    "FactorCovariance",
    "GraphicalLasso",
    "LedoitWolfCovariance",
    "OASCovariance",
    "RobustCovariance",
    "WeightedCovariance",
]
