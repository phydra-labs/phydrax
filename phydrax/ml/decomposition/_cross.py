#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size, solve_weighted_least_squares
from .._numerics._spectral import _canonicalize_rows


class CrossDecompositionDiagnostics(StrictModule):
    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    numerical_rank: Array
    weighted_orthogonality_error: Array
    minimum_eigengap: Array
    projector_gradient_supported: Array
    basis_gradient_supported: Array
    repeated_spectrum: Array
    canonicalization_valid: Array
    valid: Array
    status: Array
    effective_samples: Array
    sign_phase_convention: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


def _shape_product(shape: tuple[int, ...], /) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _flatten_targets(batch: MLBatch, /) -> tuple[Array, Array]:
    targets = batch.require_targets()
    sample_shape = batch.case_shape + (batch.sample_count,)
    width = _shape_product(
        tuple(int(size) for size in targets.shape[len(sample_shape) :])
    )
    if width <= 0:
        width = 1
    values = targets.reshape(sample_shape + (width,))
    assert batch.target_mask is not None
    mask = batch.target_mask.reshape(sample_shape + (width,))
    return values, mask


def _weighted_center(
    values: Array,
    mask: Array,
    weights: Array,
    /,
) -> tuple[Array, Array]:
    observed_weight = weights[..., :, None] * mask.astype(weights.dtype)
    mass = jnp.sum(observed_weight, axis=-2)
    safe = jnp.where(mask, values, 0)
    mean = jnp.where(
        mass > 0.0,
        jnp.sum(observed_weight * safe, axis=-2)
        / jnp.maximum(mass, jnp.finfo(weights.dtype).tiny),
        0,
    )
    return mean, jnp.where(mask, safe - mean[..., None, :], 0)


def _hermitian_inverse_root(
    matrix: Array, regularization: float, /
) -> tuple[Array, Array]:
    values, vectors = jnp.linalg.eigh(matrix)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True, initial=0.0)
    retained = values > (scale * max(matrix.shape[-2:]) * jnp.finfo(values.dtype).eps)
    inverse = jax_reciprocal_sqrt(
        jnp.maximum(values, 0.0) + jnp.asarray(float(regularization), dtype=values.dtype)
    )
    inverse = jnp.where(retained | (float(regularization) > 0.0), inverse, 0.0)
    root = (vectors * inverse[..., None, :]) @ jnp.swapaxes(jnp.conj(vectors), -1, -2)
    return root, jnp.sum(retained, axis=-1, dtype=jnp.int32)


def jax_reciprocal_sqrt(value: Array, /) -> Array:
    return 1.0 / jnp.sqrt(jnp.maximum(value, jnp.finfo(value.dtype).tiny))


def _canonicalize_columns(columns: Array, /) -> Array:
    return jnp.swapaxes(_canonicalize_rows(jnp.swapaxes(columns, -1, -2)), -1, -2)


def _center_value(value: Array, mean: Array, case_shape: tuple[int, ...], /) -> Array:
    leading = tuple(int(size) for size in value.shape[:-1])
    if case_shape and leading[: len(case_shape)] != case_shape:
        raise ValueError(f"Input must begin with fitted case shape {case_shape}.")
    sample_ndim = len(leading) - len(case_shape)
    return value - mean.reshape(case_shape + (1,) * sample_ndim + (mean.shape[-1],))


def _apply_matrix(
    value: Array,
    matrix: Array,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    leading = tuple(int(size) for size in value.shape[:-1])
    cases = _shape_product(case_shape)
    flat = value.reshape((cases, -1, value.shape[-1]))
    matrix_flat = matrix.reshape((cases, matrix.shape[-2], matrix.shape[-1]))
    result = oe.contract("cni,cio->cno", flat, matrix_flat)
    return result.reshape(leading + (matrix.shape[-1],))


class CCAModel(AbstractArrayModel):
    """Canonical correlation encoder with paired target coordinates."""

    x_mean: Array
    y_mean: Array
    x_rotations: Array
    y_rotations: Array
    canonical_correlations: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(
        self,
        x_mean: ArrayLike,
        y_mean: ArrayLike,
        x_rotations: ArrayLike,
        y_rotations: ArrayLike,
        canonical_correlations: ArrayLike,
        /,
    ):
        self.x_mean = jnp.asarray(x_mean)
        self.y_mean = jnp.asarray(y_mean)
        self.x_rotations = jnp.asarray(x_rotations)
        self.y_rotations = jnp.asarray(y_rotations)
        self.canonical_correlations = jnp.asarray(canonical_correlations)
        self.in_size = int(self.x_mean.shape[-1])
        self.target_size = int(self.y_mean.shape[-1])
        self.out_size = int(self.x_rotations.shape[-1])
        self.case_shape = tuple(int(size) for size in self.x_mean.shape[:-1])

    def _apply(self, value: Array, mean: Array, rotations: Array, width: int, /) -> Array:
        if value.shape[-1:] != (width,):
            raise ValueError(f"Expected final feature axis {width}; got {value.shape}.")
        centered = _center_value(value, mean, self.case_shape)
        return _apply_matrix(centered, rotations, self.case_shape)

    def transform(self, x: ArrayLike, /) -> Array:
        return self._apply(jnp.asarray(x), self.x_mean, self.x_rotations, self.in_size)

    def transform_targets(self, y: ArrayLike, /) -> Array:
        return self._apply(
            jnp.asarray(y), self.y_mean, self.y_rotations, self.target_size
        )

    def inverse_transform(self, scores: ArrayLike, /) -> Array:
        value = jnp.asarray(scores)
        inverse = jnp.linalg.pinv(self.x_rotations)
        reconstruction = _apply_matrix(value, inverse, self.case_shape)
        return reconstruction + self.x_mean.reshape(
            self.case_shape
            + (1,) * (reconstruction.ndim - len(self.case_shape) - 1)
            + (self.in_size,)
        )

    def predict_targets(self, x: ArrayLike, /) -> Array:
        raw_scores = self.transform(x)
        correlations = self.canonical_correlations.reshape(
            self.case_shape
            + (1,) * (raw_scores.ndim - len(self.case_shape) - 1)
            + (self.out_size,)
        )
        scores = raw_scores * correlations
        inverse = jnp.linalg.pinv(self.y_rotations)
        reconstruction = _apply_matrix(scores, inverse, self.case_shape)
        return reconstruction + self.y_mean.reshape(
            self.case_shape
            + (1,) * (reconstruction.ndim - len(self.case_shape) - 1)
            + (self.target_size,)
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


class CCA(AbstractRecipe):
    """Regularized weighted canonical correlation analysis."""

    n_components: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        regularization: float = 1e-8,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.n_components = int(n_components)
        self.regularization = float(regularization)
        self.weight_policy = weight_policy
        if self.n_components <= 0 or self.regularization < 0.0:
            raise ValueError(
                "CCA requires positive n_components and nonnegative regularization."
            )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x = batch.dense_features()
        y, y_mask = _flatten_targets(batch)
        weights = batch.effective_weight(self.weight_policy)
        x_mask = batch.feature_mask & batch.sample_mask[..., :, None]
        y_mask = y_mask & batch.sample_mask[..., :, None]
        x_mean, xc = _weighted_center(x, x_mask, weights)
        y_mean, yc = _weighted_center(y, y_mask, weights)
        total = jnp.sum(weights, axis=-1)
        denominator = jnp.maximum(total, jnp.finfo(weights.dtype).tiny)
        cxx = oe.contract("...ni,...n,...nj->...ij", jnp.conj(xc), weights, xc)
        cyy = oe.contract("...ni,...n,...nj->...ij", jnp.conj(yc), weights, yc)
        cxy = oe.contract("...ni,...n,...nj->...ij", jnp.conj(xc), weights, yc)
        cxx = cxx / denominator[..., None, None]
        cyy = cyy / denominator[..., None, None]
        cxy = cxy / denominator[..., None, None]
        x_whitener, x_rank = _hermitian_inverse_root(cxx, self.regularization)
        y_whitener, y_rank = _hermitian_inverse_root(cyy, self.regularization)
        whitened = x_whitener @ cxy @ y_whitener
        u, correlations, vh = jnp.linalg.svd(whitened, full_matrices=False)
        available = min(int(x.shape[-1]), int(y.shape[-1]))
        if self.n_components > available:
            raise ValueError(f"n_components cannot exceed {available}.")
        rank = self.n_components
        x_rotations = _canonicalize_columns(x_whitener @ u[..., :rank])
        y_rotations = y_whitener @ jnp.swapaxes(jnp.conj(vh[..., :rank, :]), -1, -2)
        phase_alignment = jnp.sum(
            jnp.conj(x_rotations) * (x_whitener @ u[..., :rank]), axis=-2
        )
        phase = jnp.where(
            jnp.abs(phase_alignment) > 0.0, phase_alignment / jnp.abs(phase_alignment), 1
        )
        y_rotations = y_rotations * jnp.conj(phase)[..., None, :]
        retained_values = correlations[..., :rank]
        energy = correlations * correlations
        total_energy = jnp.sum(energy, axis=-1)
        explained = jnp.where(
            total_energy[..., None] > 0.0, energy / total_energy[..., None], 0
        )
        residual = jnp.maximum(total_energy - jnp.sum(energy[..., :rank], axis=-1), 0.0)
        numerical_rank = jnp.minimum(x_rank, y_rank)
        finite = (
            jnp.all(jnp.isfinite(x_rotations), axis=(-2, -1))
            & jnp.all(jnp.isfinite(y_rotations), axis=(-2, -1))
            & jnp.all(jnp.isfinite(correlations), axis=-1)
        )
        enough = (total > 0.0) & (numerical_rank >= rank)
        valid = finite & enough & batch.weights_valid(self.weight_policy)
        status = jnp.where(
            ~batch.weights_valid(self.weight_policy),
            ML_INFEASIBLE,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
            ),
        ).astype(jnp.int32)
        if rank < correlations.shape[-1]:
            gaps = correlations[..., :rank] - correlations[..., 1 : rank + 1]
        elif rank > 1:
            gaps = correlations[..., : rank - 1] - correlations[..., 1:rank]
        else:
            gaps = jnp.full(correlations.shape[:-1] + (1,), jnp.inf)
        minimum_gap = jnp.min(gaps, axis=-1)
        largest = jnp.max(correlations, axis=-1, initial=0.0)
        repeated = minimum_gap <= 64.0 * jnp.finfo(correlations.dtype).eps * jnp.maximum(
            largest, 1.0
        )
        x_metric = cxx + self.regularization * jnp.eye(cxx.shape[-1], dtype=cxx.dtype)
        x_gram = jnp.swapaxes(jnp.conj(x_rotations), -1, -2) @ x_metric @ x_rotations
        orthogonality = jnp.max(
            jnp.abs(x_gram - jnp.eye(rank, dtype=x_gram.dtype)), axis=(-2, -1)
        )
        model = CCAModel(x_mean, y_mean, x_rotations, y_rotations, retained_values)
        diagnostics = CrossDecompositionDiagnostics(
            singular_values=retained_values,
            explained_energy=explained[..., :rank],
            retained_energy=jnp.sum(explained[..., :rank], axis=-1),
            residual_energy=residual,
            numerical_rank=numerical_rank,
            weighted_orthogonality_error=orthogonality,
            minimum_eigengap=minimum_gap,
            projector_gradient_supported=valid & ~repeated,
            basis_gradient_supported=valid & ~repeated,
            repeated_spectrum=repeated,
            canonicalization_valid=valid & ~repeated,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="x largest-magnitude-entry positive-real; y phase paired",
            centering_provenance="masked-weighted-feature-and-target-means",
            weighting_provenance=f"sample:{self.weight_policy}",
            method="regularized-cca-svd",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="regularized-cca-svd",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_mode="spectral",
                conditions=(
                    "canonical subspace gradients require separated singular subspaces",
                    "basis gradients additionally require non-repeated canonical correlations",
                ),
            ),
        )


class PLSModel(AbstractArrayModel):
    """Two-block PLS latent encoder, decoder, and target predictor."""

    x_mean: Array
    y_mean: Array
    x_weights: Array
    x_decoder: Array
    y_loadings: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)

    case_shape: tuple[int, ...] = eqx.field(static=True)
    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(self, x_mean, y_mean, x_weights, x_decoder, y_loadings):
        self.x_mean = jnp.asarray(x_mean)
        self.y_mean = jnp.asarray(y_mean)
        self.x_weights = jnp.asarray(x_weights)
        self.x_decoder = jnp.asarray(x_decoder)
        self.y_loadings = jnp.asarray(y_loadings)
        self.in_size = int(self.x_mean.shape[-1])
        self.out_size = int(self.x_weights.shape[-1])
        self.target_size = int(self.y_mean.shape[-1])

        self.case_shape = tuple(int(size) for size in self.x_mean.shape[:-1])

    def transform(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        centered = _center_value(value, self.x_mean, self.case_shape)
        return _apply_matrix(centered, self.x_weights, self.case_shape)

    def inverse_transform(self, scores: ArrayLike, /) -> Array:
        value = jnp.asarray(scores)
        reconstruction = _apply_matrix(value, self.x_decoder, self.case_shape)
        return reconstruction + self.x_mean.reshape(
            self.case_shape
            + (1,) * (reconstruction.ndim - len(self.case_shape) - 1)
            + (self.in_size,)
        )

    def predict(self, x: ArrayLike, /) -> Array:
        scores = self.transform(x)
        prediction = _apply_matrix(scores, self.y_loadings, self.case_shape)
        return prediction + self.y_mean.reshape(
            self.case_shape
            + (1,) * (prediction.ndim - len(self.case_shape) - 1)
            + (self.target_size,)
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


class PLS(AbstractRecipe):
    """Weighted two-block PLS-SVD with affine reconstruction and prediction."""

    n_components: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.n_components = int(n_components)
        self.weight_policy = weight_policy
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x = batch.dense_features()
        y, y_mask = _flatten_targets(batch)
        weights = batch.effective_weight(self.weight_policy)
        x_mean, xc = _weighted_center(
            x, batch.feature_mask & batch.sample_mask[..., :, None], weights
        )
        y_mean, yc = _weighted_center(
            y, y_mask & batch.sample_mask[..., :, None], weights
        )
        cross = oe.contract("...ni,...n,...nj->...ij", jnp.conj(xc), weights, yc)
        u, singular, _vh = jnp.linalg.svd(cross, full_matrices=False)
        available = min(int(x.shape[-1]), int(y.shape[-1]))
        if self.n_components > available:
            raise ValueError(f"n_components cannot exceed {available}.")
        rank = self.n_components
        x_weights = _canonicalize_columns(u[..., :rank])
        scores = xc @ x_weights
        ridge = jnp.finfo(scores.real.dtype).eps * jnp.maximum(
            oe.contract(
                "...nr,...n,...nr->...",
                jnp.conj(scores),
                weights,
                scores,
            ).real,
            1.0,
        )
        x_decoder = solve_weighted_least_squares(
            scores,
            xc,
            weights,
            ridge=ridge,
            fit_intercept=False,
        ).coefficients
        y_decoder = solve_weighted_least_squares(
            scores,
            yc,
            weights,
            ridge=ridge,
            fit_intercept=False,
        ).coefficients
        retained = singular[..., :rank]
        energy = singular * singular
        total_energy = jnp.sum(energy, axis=-1)
        explained = jnp.where(
            total_energy[..., None] > 0.0, energy / total_energy[..., None], 0
        )
        residual = jnp.maximum(total_energy - jnp.sum(energy[..., :rank], axis=-1), 0.0)
        largest = jnp.max(singular, axis=-1, initial=0.0)
        cutoff = (
            largest
            * max(x.shape[-2], x.shape[-1], y.shape[-1])
            * jnp.finfo(singular.dtype).eps
        )
        numerical_rank = jnp.sum(singular > cutoff[..., None], axis=-1, dtype=jnp.int32)
        if rank < singular.shape[-1]:
            gaps = singular[..., :rank] - singular[..., 1 : rank + 1]
        elif rank > 1:
            gaps = singular[..., : rank - 1] - singular[..., 1:rank]
        else:
            gaps = jnp.full(singular.shape[:-1] + (1,), jnp.inf)
        minimum_gap = jnp.min(gaps, axis=-1)
        repeated = minimum_gap <= 64.0 * jnp.finfo(singular.dtype).eps * jnp.maximum(
            largest, 1.0
        )
        finite = jnp.all(jnp.isfinite(x_weights), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(singular), axis=-1
        )
        enough = numerical_rank >= rank
        valid = finite & enough & batch.weights_valid(self.weight_policy)
        status = jnp.where(
            ~batch.weights_valid(self.weight_policy),
            ML_INFEASIBLE,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(enough, ML_SUCCESS, ML_INSUFFICIENT_DATA),
            ),
        ).astype(jnp.int32)
        gram = jnp.swapaxes(jnp.conj(x_weights), -1, -2) @ x_weights
        orthogonality = jnp.max(
            jnp.abs(gram - jnp.eye(rank, dtype=gram.dtype)), axis=(-2, -1)
        )
        model = PLSModel(x_mean, y_mean, x_weights, x_decoder, y_decoder)
        diagnostics = CrossDecompositionDiagnostics(
            singular_values=retained,
            explained_energy=explained[..., :rank],
            retained_energy=jnp.sum(explained[..., :rank], axis=-1),
            residual_energy=residual,
            numerical_rank=numerical_rank,
            weighted_orthogonality_error=orthogonality,
            minimum_eigengap=minimum_gap,
            projector_gradient_supported=valid & ~repeated,
            basis_gradient_supported=valid & ~repeated,
            repeated_spectrum=repeated,
            canonicalization_valid=valid & ~repeated,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="largest-magnitude-entry-positive-real",
            centering_provenance="masked-weighted-feature-and-target-means",
            weighting_provenance=f"sample:{self.weight_policy}",
            method="two-block-pls-svd",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="two-block-pls-svd",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_mode="spectral",
                conditions=(
                    "PLS weight gradients require separated cross-covariance spectrum",
                ),
            ),
        )


__all__ = [
    "CCA",
    "CCAModel",
    "CrossDecompositionDiagnostics",
    "PLS",
    "PLSModel",
]
