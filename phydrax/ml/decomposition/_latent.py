#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
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
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size, fit_weighted_subspace
from .._numerics._spectral import _canonicalize_rows


class LatentDecompositionDiagnostics(StrictModule):
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
    objective: Array
    iterations: Array
    converged: Array
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


def _masked_center(batch: MLBatch, policy: WeightPolicy, /) -> tuple[Array, Array, Array]:
    x = batch.dense_features()
    weights = batch.effective_weight(policy)
    mask = batch.feature_mask & batch.sample_mask[..., :, None]
    observed_weight = weights[..., :, None] * mask.astype(weights.dtype)
    mass = jnp.sum(observed_weight, axis=-2)
    mean = jnp.where(
        mass > 0.0,
        jnp.sum(observed_weight * jnp.where(mask, x, 0), axis=-2)
        / jnp.maximum(mass, jnp.finfo(weights.dtype).tiny),
        0,
    )
    centered = jnp.where(mask, x - mean[..., None, :], 0)
    return mean, centered, weights


def _canonicalize_columns(columns: Array, /) -> Array:
    return jnp.swapaxes(_canonicalize_rows(jnp.swapaxes(columns, -1, -2)), -1, -2)


def _center_value(value: Array, mean: Array, case_shape: tuple[int, ...], /) -> Array:
    leading = tuple(int(size) for size in value.shape[:-1])
    if case_shape and leading[: len(case_shape)] != case_shape:
        raise ValueError(f"Input must begin with fitted case shape {case_shape}.")
    return value - mean.reshape(
        case_shape + (1,) * (len(leading) - len(case_shape)) + (mean.shape[-1],)
    )


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


class FactorAnalysisModel(AbstractArrayModel):
    """Diagonal-noise latent Gaussian factor encoder and affine decoder."""

    mean: Array
    loadings: Array
    noise_variance: Array
    posterior_matrix: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(self, mean, loadings, noise_variance):
        self.mean = jnp.asarray(mean)
        self.loadings = jnp.asarray(loadings)
        self.noise_variance = jnp.asarray(noise_variance)
        inverse_noise = 1.0 / jnp.maximum(
            self.noise_variance, jnp.finfo(self.noise_variance.dtype).tiny
        )
        weighted = (
            jnp.swapaxes(jnp.conj(self.loadings), -1, -2) * inverse_noise[..., None, :]
        )
        precision = weighted @ self.loadings + jnp.eye(
            self.loadings.shape[-1], dtype=self.loadings.dtype
        )
        self.posterior_matrix = jnp.linalg.solve(precision, weighted)
        self.in_size = int(self.mean.shape[-1])
        self.out_size = int(self.loadings.shape[-1])
        self.case_shape = tuple(int(size) for size in self.mean.shape[:-1])

    def transform(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        centered = _center_value(value, self.mean, self.case_shape)
        matrix = jnp.swapaxes(self.posterior_matrix, -1, -2)
        return _apply_matrix(centered, matrix, self.case_shape)

    def inverse_transform(self, scores: ArrayLike, /) -> Array:
        value = jnp.asarray(scores)
        matrix = jnp.swapaxes(self.loadings, -1, -2)
        reconstruction = _apply_matrix(value, matrix, self.case_shape)
        return reconstruction + self.mean.reshape(
            self.case_shape
            + (1,) * (reconstruction.ndim - len(self.case_shape) - 1)
            + (self.in_size,)
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


class FactorAnalysis(AbstractRecipe):
    """Weighted principal-axis factor analysis with diagonal uniqueness updates."""

    n_components: int = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    min_noise: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        max_iterations: int = 64,
        tolerance: float = 1e-6,
        min_noise: float = 1e-8,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.n_components = int(n_components)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.min_noise = float(min_noise)
        self.weight_policy = weight_policy
        if self.n_components <= 0 or self.max_iterations <= 0:
            raise ValueError("FactorAnalysis sizes and iteration count must be positive.")
        if self.tolerance < 0.0 or self.min_noise <= 0.0:
            raise ValueError(
                "FactorAnalysis tolerances must be nonnegative and noise positive."
            )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        mean, centered, weights = _masked_center(batch, self.weight_policy)
        width = int(centered.shape[-1])
        if self.n_components > width:
            raise ValueError(f"n_components cannot exceed feature count {width}.")
        total = jnp.sum(weights, axis=-1)
        covariance = oe.contract(
            "...ni,...n,...nj->...ij", jnp.conj(centered), weights, centered
        ) / jnp.maximum(total[..., None, None], jnp.finfo(weights.dtype).tiny)
        diagonal = jnp.real(jnp.diagonal(covariance, axis1=-2, axis2=-1))
        noise0 = jnp.maximum(0.5 * diagonal, self.min_noise)

        def step(carry, _):
            noise, _previous, done, used = carry
            reduced = (
                covariance - jnp.eye(width, dtype=covariance.dtype) * noise[..., None, :]
            )
            eigenvalues, eigenvectors = jnp.linalg.eigh(reduced)
            values = jnp.maximum(eigenvalues[..., -self.n_components :], 0.0)
            vectors = eigenvectors[..., :, -self.n_components :]
            loadings = vectors * jnp.sqrt(values)[..., None, :]
            communalities = jnp.sum(jnp.real(loadings * jnp.conj(loadings)), axis=-1)
            candidate = jnp.maximum(diagonal - communalities, self.min_noise)
            residual = jnp.max(jnp.abs(candidate - noise), axis=-1)
            newly_done = residual <= self.tolerance
            next_noise = jnp.where(done[..., None], noise, candidate)
            next_used = jnp.where(done, used, used + 1)
            return (next_noise, residual, done | newly_done, next_used), residual

        (noise, residual, converged, iterations), _history = jax.lax.scan(
            step,
            (
                noise0,
                jnp.full(total.shape, jnp.inf),
                jnp.zeros(total.shape, dtype=bool),
                jnp.zeros(total.shape, dtype=jnp.int32),
            ),
            xs=None,
            length=self.max_iterations,
        )
        reduced = (
            covariance - jnp.eye(width, dtype=covariance.dtype) * noise[..., None, :]
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(reduced)
        retained_eigenvalues = jnp.maximum(
            eigenvalues[..., -self.n_components :][..., ::-1], 0.0
        )
        retained_vectors = eigenvectors[..., :, -self.n_components :][..., ::-1]
        loadings = _canonicalize_columns(
            retained_vectors * jnp.sqrt(retained_eigenvalues)[..., None, :]
        )
        model = FactorAnalysisModel(mean, loadings, noise)
        singular = jnp.sqrt(retained_eigenvalues)
        total_energy = jnp.sum(jnp.maximum(eigenvalues, 0.0), axis=-1)
        retained_energy_raw = jnp.sum(retained_eigenvalues, axis=-1)
        retained_ratio = jnp.where(
            total_energy > 0.0, retained_energy_raw / total_energy, 0.0
        )
        explained = jnp.where(
            total_energy[..., None] > 0.0,
            retained_eigenvalues / total_energy[..., None],
            0,
        )
        numerical_rank = jnp.sum(eigenvalues > self.min_noise, axis=-1, dtype=jnp.int32)
        if self.n_components < width:
            boundary_gap = (
                eigenvalues[..., -self.n_components]
                - eigenvalues[..., -self.n_components - 1]
            )
        else:
            boundary_gap = jnp.full(total.shape, jnp.inf)
        largest = jnp.max(jnp.abs(eigenvalues), axis=-1, initial=0.0)
        repeated = boundary_gap <= 64.0 * jnp.finfo(eigenvalues.dtype).eps * jnp.maximum(
            largest, 1.0
        )
        finite = jnp.all(jnp.isfinite(loadings), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(noise), axis=-1
        )
        enough = numerical_rank >= self.n_components
        weights_valid = batch.weights_valid(self.weight_policy)
        valid = finite & enough & converged & weights_valid
        status = jnp.where(
            ~weights_valid,
            ML_INFEASIBLE,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(
                    ~enough,
                    ML_INSUFFICIENT_DATA,
                    jnp.where(converged, ML_SUCCESS, ML_NONCONVERGED),
                ),
            ),
        ).astype(jnp.int32)
        gram = jnp.swapaxes(jnp.conj(retained_vectors), -1, -2) @ retained_vectors
        orthogonality = jnp.max(
            jnp.abs(gram - jnp.eye(self.n_components, dtype=gram.dtype)), axis=(-2, -1)
        )
        diagnostics = LatentDecompositionDiagnostics(
            singular_values=singular,
            explained_energy=explained,
            retained_energy=retained_ratio,
            residual_energy=jnp.maximum(total_energy - retained_energy_raw, 0.0),
            numerical_rank=numerical_rank,
            weighted_orthogonality_error=orthogonality,
            minimum_eigengap=boundary_gap,
            projector_gradient_supported=valid & ~repeated,
            basis_gradient_supported=valid & ~repeated,
            repeated_spectrum=repeated,
            canonicalization_valid=valid & ~repeated,
            objective=residual,
            iterations=iterations,
            converged=converged,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="largest-magnitude-entry-positive-real",
            centering_provenance="masked-weighted-feature-mean",
            weighting_provenance=f"sample:{self.weight_policy};noise:diagonal",
            method="principal-axis-factor-analysis",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="principal-axis-factor-analysis",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_weights="conditional",
                fit_mode="unrolled",
                conditions=(
                    "factor loading gradients require separated retained eigenspaces",
                    "convergence-masked uniqueness iterations are piecewise smooth",
                ),
            ),
        )


class ICAModel(AbstractArrayModel):
    """Fixed independent-component encoder and affine mixing decoder."""

    mean: Array
    unmixing: Array
    mixing: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(self, mean, unmixing):
        self.mean = jnp.asarray(mean)
        self.unmixing = jnp.asarray(unmixing)
        self.mixing = jnp.linalg.pinv(self.unmixing)
        self.in_size = int(self.mean.shape[-1])
        self.out_size = int(self.unmixing.shape[-2])
        self.case_shape = tuple(int(size) for size in self.mean.shape[:-1])

    def transform(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        centered = _center_value(value, self.mean, self.case_shape)
        matrix = jnp.swapaxes(self.unmixing, -1, -2)
        return _apply_matrix(centered, matrix, self.case_shape)

    def inverse_transform(self, scores: ArrayLike, /) -> Array:
        value = jnp.asarray(scores)
        matrix = jnp.swapaxes(self.mixing, -1, -2)
        reconstruction = _apply_matrix(value, matrix, self.case_shape)
        return reconstruction + self.mean.reshape(
            self.case_shape
            + (1,) * (reconstruction.ndim - len(self.case_shape) - 1)
            + (self.in_size,)
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


def _sym_decorrelate(matrix: Array, /) -> Array:
    gram = matrix @ jnp.swapaxes(matrix, -1, -2)
    values, vectors = jnp.linalg.eigh(gram)
    inverse_root = 1.0 / jnp.sqrt(jnp.maximum(values, jnp.finfo(values.dtype).tiny))
    whitening = (vectors * inverse_root[..., None, :]) @ jnp.swapaxes(vectors, -1, -2)
    return whitening @ matrix


class ICA(AbstractRecipe):
    """Weighted symmetric FastICA with explicit random initialization."""

    n_components: int = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.n_components = int(n_components)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy
        if self.n_components <= 0 or self.max_iterations <= 0 or self.tolerance < 0.0:
            raise ValueError("ICA dimensions and iteration settings are invalid.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("ICA fitting requires an explicit JAX key.")
        x = batch.dense_features()
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError(
                "This real FastICA recipe does not define complex ICA semantics."
            )
        mean, centered, weights = _masked_center(batch, self.weight_policy)
        spectral = fit_weighted_subspace(
            centered,
            weights,
            rank=self.n_components,
            centered=False,
        )
        scale = jnp.maximum(spectral.singular_values, jnp.finfo(centered.dtype).tiny)
        whitening = spectral.components / scale[..., :, None]
        whitened = centered @ jnp.swapaxes(whitening, -1, -2)
        case_shape = tuple(int(size) for size in batch.case_shape)
        random = jax.random.normal(
            key,
            case_shape + (self.n_components, self.n_components),
            dtype=centered.dtype,
        )
        initial = _sym_decorrelate(random)
        normalized_weights = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), jnp.finfo(weights.dtype).tiny
        )

        def step(carry, _):
            matrix, done, used, residual = carry
            projections = whitened @ jnp.swapaxes(matrix, -1, -2)
            activation = jnp.tanh(projections)
            derivative = 1.0 - activation * activation
            candidate = (
                oe.contract(
                    "...n,...nr,...nf->...rf", normalized_weights, activation, whitened
                )
                - jnp.sum(normalized_weights[..., :, None] * derivative, axis=-2)[
                    ..., :, None
                ]
                * matrix
            )
            candidate = _sym_decorrelate(candidate)
            alignment = jnp.abs(jnp.sum(candidate * matrix, axis=-1))
            next_residual = jnp.max(jnp.abs(alignment - 1.0), axis=-1)
            newly_done = next_residual <= self.tolerance
            candidate = jnp.where(done[..., None, None], matrix, candidate)
            used = jnp.where(done, used, used + 1)
            return (candidate, done | newly_done, used, next_residual), next_residual

        (rotation, converged, iterations, residual), _history = jax.lax.scan(
            step,
            (
                initial,
                jnp.zeros(case_shape, dtype=bool),
                jnp.zeros(case_shape, dtype=jnp.int32),
                jnp.full(case_shape, jnp.inf),
            ),
            xs=None,
            length=self.max_iterations,
        )
        unmixing = _canonicalize_rows(rotation @ whitening)
        model = ICAModel(mean, unmixing)
        finite = jnp.all(jnp.isfinite(unmixing), axis=(-2, -1))
        valid = (
            finite & spectral.valid & converged & batch.weights_valid(self.weight_policy)
        )
        weights_valid = batch.weights_valid(self.weight_policy)
        status = jnp.where(
            ~weights_valid,
            ML_INFEASIBLE,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(~converged, ML_NONCONVERGED, spectral.status),
            ),
        ).astype(jnp.int32)
        largest = jnp.max(spectral.singular_values, axis=-1, initial=0.0)
        repeated = spectral.minimum_retained_gap <= 64.0 * jnp.finfo(
            centered.dtype
        ).eps * jnp.maximum(largest, 1.0)
        diagnostics = LatentDecompositionDiagnostics(
            singular_values=spectral.singular_values,
            explained_energy=spectral.explained_energy,
            retained_energy=spectral.retained_energy,
            residual_energy=spectral.residual_energy,
            numerical_rank=spectral.numerical_rank,
            weighted_orthogonality_error=spectral.orthogonality_error,
            minimum_eigengap=spectral.minimum_retained_gap,
            projector_gradient_supported=valid & ~repeated,
            basis_gradient_supported=valid & ~repeated,
            repeated_spectrum=repeated,
            canonicalization_valid=valid & ~repeated,
            objective=residual,
            iterations=iterations,
            converged=converged,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="largest-magnitude-entry-positive",
            centering_provenance="masked-weighted-feature-mean",
            weighting_provenance=f"sample:{self.weight_policy}",
            method="symmetric-fastica",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="symmetric-fastica",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_weights="conditional",
                fit_mode="unrolled",
                conditions=(
                    "ICA initialization key is explicit and fixed during differentiation",
                    "whitening gradients require separated retained and discarded spectra",
                    "FastICA fixed-point iterations must converge without component collisions",
                ),
            ),
        )


__all__ = [
    "FactorAnalysis",
    "FactorAnalysisModel",
    "ICA",
    "ICAModel",
    "LatentDecompositionDiagnostics",
]
