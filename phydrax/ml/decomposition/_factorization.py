#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._numerics import effective_sample_size, soft_threshold


class FactorizationDiagnostics(StrictModule):
    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    weighted_orthogonality_error: Array
    minimum_eigengap: Array
    repeated_spectrum: Array
    canonicalization_valid: Array
    objective: Array
    reconstruction_error: Array
    regularization: Array
    iterations: Array
    converged: Array
    numerical_rank: Array
    atom_norm_error: Array
    valid: Array
    status: Array
    effective_samples: Array
    sign_phase_convention: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    mask_provenance: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


def _prepare(batch: MLBatch, policy: WeightPolicy, /) -> tuple[Array, Array, Array]:
    mask = batch.feature_mask & batch.sample_mask[..., :, None]
    values = jnp.where(mask, batch.dense_features(), 0)
    weights = batch.effective_weight(policy)
    return values, weights, mask


def _normalize_atoms(dictionary: Array, /) -> Array:
    norms = jnp.linalg.norm(dictionary, axis=-1, keepdims=True)
    return dictionary / jnp.maximum(norms, jnp.finfo(dictionary.real.dtype).tiny)


def _dictionary_spectrum(
    dictionary: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    singular = jnp.linalg.svd(dictionary, compute_uv=False)
    largest = jnp.max(singular, axis=-1, initial=0.0)
    numerical_rank = jnp.sum(
        singular
        > largest[..., None] * jnp.finfo(singular.dtype).eps * max(dictionary.shape[-2:]),
        axis=-1,
        dtype=jnp.int32,
    )
    energy = singular * singular
    total = jnp.sum(energy, axis=-1)
    explained = jnp.where(total[..., None] > 0.0, energy / total[..., None], 0.0)
    if singular.shape[-1] > 1:
        minimum_gap = jnp.min(singular[..., :-1] - singular[..., 1:], axis=-1)
    else:
        minimum_gap = jnp.full(singular.shape[:-1], jnp.inf)
    repeated = minimum_gap <= (
        64.0 * jnp.finfo(singular.dtype).eps * jnp.maximum(largest, 1.0)
    )
    gram = dictionary @ jnp.swapaxes(jnp.conj(dictionary), -1, -2)
    orthogonality = jnp.max(
        jnp.abs(gram - jnp.eye(dictionary.shape[-2], dtype=gram.dtype)),
        axis=(-2, -1),
    )
    return singular, explained, numerical_rank, orthogonality, minimum_gap, repeated


def _ista_codes(
    values: Array,
    dictionary: Array,
    regularization: float,
    iterations: int,
    /,
    *,
    mask: Array | None = None,
) -> Array:
    rank = int(dictionary.shape[-2])
    codes = jnp.zeros(values.shape[:-1] + (rank,), dtype=values.dtype)
    lipschitz = jnp.sum(jnp.real(dictionary * jnp.conj(dictionary)), axis=(-2, -1))
    step = 1.0 / jnp.maximum(lipschitz, jnp.finfo(values.real.dtype).tiny)

    def body(_, current):
        residual = current @ dictionary - values
        if mask is not None:
            residual = jnp.where(mask, residual, 0)
        gradient = residual @ jnp.swapaxes(jnp.conj(dictionary), -1, -2)
        return soft_threshold(
            current - step[..., None, None] * gradient,
            jnp.asarray(regularization, dtype=values.real.dtype) * step[..., None, None],
        )

    return jax.lax.fori_loop(0, int(iterations), body, codes)


class SparseCodingModel(AbstractArrayModel):
    """Fixed dictionary with a deterministic unrolled proximal encoder."""

    dictionary: Array
    regularization: float = eqx.field(static=True)
    transform_iterations: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(
        self,
        dictionary: ArrayLike,
        /,
        *,
        regularization: float,
        transform_iterations: int,
    ):
        dictionary_ = jnp.asarray(dictionary)
        if not jnp.issubdtype(dictionary_.dtype, jnp.inexact):
            dictionary_ = dictionary_.astype(jnp.float32)
        if dictionary_.ndim < 2:
            raise ValueError("dictionary must end in (atom, feature).")
        self.dictionary = dictionary_
        self.regularization = float(regularization)
        self.transform_iterations = int(transform_iterations)
        self.in_size = int(dictionary_.shape[-1])
        self.out_size = int(dictionary_.shape[-2])
        self.case_shape = tuple(int(size) for size in dictionary_.shape[:-2])
        if self.regularization < 0.0 or self.transform_iterations <= 0:
            raise ValueError("Sparse coding regularization and iterations are invalid.")

    def transform(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(self.dictionary.dtype)
        if value.shape[-1:] != (self.in_size,):
            raise ValueError(
                f"Expected final feature axis {self.in_size}; got {value.shape}."
            )
        if self.case_shape and value.shape[: len(self.case_shape)] != self.case_shape:
            raise ValueError(
                f"Input must begin with fitted case shape {self.case_shape}."
            )
        added_sample = value.ndim == len(self.case_shape) + 1
        working = value[..., None, :] if added_sample else value
        codes = _ista_codes(
            working,
            self.dictionary,
            self.regularization,
            self.transform_iterations,
        )
        return codes[..., 0, :] if added_sample else codes

    def inverse_transform(self, codes: ArrayLike, /) -> Array:
        value = jnp.asarray(codes)
        if value.shape[-1:] != (self.out_size,):
            raise ValueError(
                f"Expected final code axis {self.out_size}; got {value.shape}."
            )
        return value @ self.dictionary

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


class NMFModel(AbstractArrayModel):
    """Nonnegative basis with multiplicative nonnegative encoding."""

    components: Array
    transform_iterations: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(self, components, *, transform_iterations: int, epsilon: float):
        components_ = jnp.asarray(components)
        if not jnp.issubdtype(components_.dtype, jnp.floating):
            raise TypeError("NMF components must use a real floating dtype.")
        self.components = components_
        self.transform_iterations = int(transform_iterations)
        self.epsilon = float(epsilon)
        self.in_size = int(self.components.shape[-1])
        self.out_size = int(self.components.shape[-2])
        self.case_shape = tuple(int(size) for size in self.components.shape[:-2])

    def transform(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("NMF is defined only for real nonnegative inputs.")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            value = value.astype(self.components.dtype)
        value = eqx.error_if(
            value, jnp.any(value < 0.0), "NMF inputs must be nonnegative."
        )
        added_sample = value.ndim == len(self.case_shape) + 1
        working = value[..., None, :] if added_sample else value
        codes = jnp.ones(
            working.shape[:-1] + (self.out_size,),
            dtype=working.dtype,
        )
        gram = self.components @ jnp.swapaxes(self.components, -1, -2)

        def body(_, current):
            numerator = working @ jnp.swapaxes(self.components, -1, -2)
            denominator = current @ gram
            return current * numerator / jnp.maximum(denominator, self.epsilon)

        codes = jax.lax.fori_loop(0, self.transform_iterations, body, codes)
        return codes[..., 0, :] if added_sample else codes

    def inverse_transform(self, scores: ArrayLike, /) -> Array:
        return jnp.asarray(scores) @ self.components

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


class NMF(AbstractRecipe):
    """Weighted masked nonnegative matrix factorization with explicit initialization key."""

    n_components: int = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    transform_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        max_iterations: int = 200,
        transform_iterations: int = 64,
        tolerance: float = 1e-6,
        epsilon: float = 1e-8,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.n_components = int(n_components)
        self.max_iterations = int(max_iterations)
        self.transform_iterations = int(transform_iterations)
        self.tolerance = float(tolerance)
        self.epsilon = float(epsilon)
        self.weight_policy = weight_policy
        if min(self.n_components, self.max_iterations, self.transform_iterations) <= 0:
            raise ValueError("NMF dimensions and iteration counts must be positive.")
        if self.tolerance < 0.0 or self.epsilon <= 0.0:
            raise ValueError("NMF tolerances are invalid.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("NMF fitting requires an explicit JAX key.")
        values, weights, feature_mask = _prepare(batch, self.weight_policy)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("NMF is defined only for real nonnegative arrays.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            values = values.astype(jnp.float32)
        values = eqx.error_if(
            values, jnp.any(values < 0.0), "NMF inputs must be nonnegative."
        )
        key_codes, key_components = jax.random.split(key)
        case_shape = tuple(int(size) for size in batch.case_shape)
        codes = jax.random.uniform(
            key_codes,
            values.shape[:-1] + (self.n_components,),
            dtype=values.dtype,
            minval=self.epsilon,
            maxval=1.0,
        )
        components = jax.random.uniform(
            key_components,
            case_shape + (self.n_components, batch.feature_count),
            dtype=values.dtype,
            minval=self.epsilon,
            maxval=1.0,
        )

        def body(_, state):
            current_codes, current_components = state
            prediction = current_codes @ current_components
            numerator_codes = values @ jnp.swapaxes(current_components, -1, -2)
            denominator_codes = jnp.where(feature_mask, prediction, 0) @ jnp.swapaxes(
                current_components, -1, -2
            )
            next_codes = (
                current_codes
                * numerator_codes
                / jnp.maximum(denominator_codes, self.epsilon)
            )
            weighted_values = weights[..., :, None] * values
            numerator_components = jnp.swapaxes(next_codes, -1, -2) @ weighted_values
            next_prediction = next_codes @ current_components
            weighted_prediction = (
                weights[..., :, None]
                * feature_mask.astype(weights.dtype)
                * next_prediction
            )
            denominator_components = (
                jnp.swapaxes(next_codes, -1, -2) @ weighted_prediction
            )
            next_components = (
                current_components
                * numerator_components
                / jnp.maximum(denominator_components, self.epsilon)
            )
            norms = jnp.maximum(jnp.linalg.norm(next_components, axis=-1), self.epsilon)
            next_components = next_components / norms[..., :, None]
            next_codes = next_codes * norms[..., None, :]
            return next_codes, next_components

        codes, components = jax.lax.fori_loop(
            0, self.max_iterations, body, (codes, components)
        )
        prediction = codes @ components
        residual = jnp.where(feature_mask, values - prediction, 0)
        reconstruction = jnp.sum(
            weights[..., :, None] * residual * residual, axis=(-2, -1)
        )
        objective = 0.5 * reconstruction
        numerator_codes = values @ jnp.swapaxes(components, -1, -2)
        denominator_codes = jnp.where(feature_mask, prediction, 0) @ jnp.swapaxes(
            components, -1, -2
        )
        stationarity = jnp.max(
            jnp.abs(codes * (denominator_codes - numerator_codes)),
            axis=(-2, -1),
        )
        converged = stationarity <= self.tolerance
        finite = jnp.all(jnp.isfinite(codes), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(components), axis=(-2, -1)
        )
        weights_valid = batch.weights_valid(self.weight_policy)
        valid = finite & converged & weights_valid
        status = jnp.where(
            ~weights_valid,
            ML_INFEASIBLE,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(converged, ML_SUCCESS, ML_NONCONVERGED),
            ),
        ).astype(jnp.int32)
        (
            singular,
            explained,
            numerical_rank,
            orthogonality,
            minimum_gap,
            repeated,
        ) = _dictionary_spectrum(components)
        atom_error = jnp.max(jnp.abs(jnp.linalg.norm(components, axis=-1) - 1.0), axis=-1)
        model = NMFModel(
            components,
            transform_iterations=self.transform_iterations,
            epsilon=self.epsilon,
        )
        data_energy = jnp.sum(weights[..., :, None] * values * values, axis=(-2, -1))
        retained_energy = jnp.where(
            data_energy > 0.0,
            jnp.maximum(1.0 - reconstruction / data_energy, 0.0),
            0.0,
        )
        diagnostics = FactorizationDiagnostics(
            singular_values=singular,
            explained_energy=explained,
            retained_energy=retained_energy,
            residual_energy=reconstruction,
            weighted_orthogonality_error=orthogonality,
            minimum_eigengap=minimum_gap,
            repeated_spectrum=repeated,
            canonicalization_valid=jnp.zeros_like(valid),
            objective=objective,
            reconstruction_error=reconstruction,
            regularization=jnp.zeros_like(objective),
            iterations=jnp.full(objective.shape, self.max_iterations, dtype=jnp.int32),
            converged=converged,
            numerical_rank=numerical_rank,
            atom_norm_error=atom_error,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="nonnegative atoms; permutation symmetry remains",
            centering_provenance="origin",
            weighting_provenance=f"sample:{self.weight_policy}",
            mask_provenance="zero-extension",
            method="multiplicative-nmf",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="multiplicative-nmf",
            gradient_contract=GradientContract(
                prediction_inputs="almost-everywhere",
                fit_features="almost-everywhere",
                fit_weights="conditional",
                fit_mode="unrolled",
                conditions=("multiplicative iterates must stay strictly positive",),
            ),
        )


class SparseCoding(AbstractRecipe):
    """Immutable fixed-dictionary sparse coding recipe."""

    dictionary: Array
    regularization: float = eqx.field(static=True)
    transform_iterations: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        dictionary: ArrayLike,
        /,
        *,
        regularization: float = 1e-2,
        transform_iterations: int = 64,
        weight_policy: WeightPolicy = "statistical",
    ):
        dictionary_ = jnp.asarray(dictionary)
        if not jnp.issubdtype(dictionary_.dtype, jnp.inexact):
            dictionary_ = dictionary_.astype(jnp.float32)
        if dictionary_.ndim < 2:
            raise ValueError("dictionary must end in (atom, feature).")
        norms = jnp.linalg.norm(dictionary_, axis=-1)
        dictionary_ = eqx.error_if(
            dictionary_,
            jnp.any(~jnp.isfinite(norms) | (norms <= 0.0)),
            "SparseCoding dictionary atoms must be finite and nonzero.",
        )
        self.dictionary = _normalize_atoms(dictionary_)
        self.regularization = float(regularization)
        self.transform_iterations = int(transform_iterations)
        self.weight_policy = weight_policy
        if self.regularization < 0.0 or self.transform_iterations <= 0:
            raise ValueError("SparseCoding solver settings are invalid.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        values, weights, feature_mask = _prepare(batch, self.weight_policy)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(self.dictionary.dtype)
        dictionary = jnp.broadcast_to(
            self.dictionary,
            batch.case_shape + self.dictionary.shape[-2:],
        )
        if dictionary.shape[-1] != batch.feature_count:
            raise ValueError("Dictionary feature width must match the batch.")
        codes = _ista_codes(
            values,
            dictionary,
            self.regularization,
            self.transform_iterations,
            mask=feature_mask,
        )
        residual = jnp.where(feature_mask, values - codes @ dictionary, 0)
        reconstruction = jnp.sum(
            weights[..., :, None] * jnp.real(residual * jnp.conj(residual)), axis=(-2, -1)
        )
        penalty = self.regularization * jnp.sum(jnp.abs(codes), axis=(-2, -1))
        objective = 0.5 * reconstruction + penalty
        finite = jnp.all(jnp.isfinite(codes), axis=(-2, -1))
        weights_valid = batch.weights_valid(self.weight_policy)
        valid = finite & weights_valid
        status = jnp.where(
            ~weights_valid,
            ML_INFEASIBLE,
            jnp.where(finite, ML_SUCCESS, ML_NONFINITE),
        ).astype(jnp.int32)
        (
            singular,
            explained,
            numerical_rank,
            orthogonality,
            minimum_gap,
            repeated,
        ) = _dictionary_spectrum(dictionary)
        atom_error = jnp.max(jnp.abs(jnp.linalg.norm(dictionary, axis=-1) - 1.0), axis=-1)
        model = SparseCodingModel(
            dictionary,
            regularization=self.regularization,
            transform_iterations=self.transform_iterations,
        )
        data_energy = jnp.sum(
            weights[..., :, None] * jnp.real(values * jnp.conj(values)),
            axis=(-2, -1),
        )
        retained_energy = jnp.where(
            data_energy > 0.0,
            jnp.maximum(1.0 - reconstruction / data_energy, 0.0),
            0.0,
        )
        diagnostics = FactorizationDiagnostics(
            singular_values=singular,
            explained_energy=explained,
            retained_energy=retained_energy,
            residual_energy=reconstruction,
            weighted_orthogonality_error=orthogonality,
            minimum_eigengap=minimum_gap,
            repeated_spectrum=repeated,
            canonicalization_valid=jnp.zeros_like(valid),
            objective=objective,
            reconstruction_error=reconstruction,
            regularization=penalty,
            iterations=jnp.full(
                objective.shape, self.transform_iterations, dtype=jnp.int32
            ),
            converged=jnp.ones(objective.shape, dtype=bool),
            numerical_rank=numerical_rank,
            atom_norm_error=atom_error,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="uncanonicalized signed/phase atoms; permutation symmetry remains",
            centering_provenance="origin",
            weighting_provenance=f"sample:{self.weight_policy}",
            mask_provenance="zero-extension",
            method="fixed-dictionary-ista",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="fixed-dictionary-ista",
            gradient_contract=GradientContract(
                prediction_inputs="almost-everywhere",
                prediction_parameters="almost-everywhere",
                fit_features="almost-everywhere",
                fit_mode="unrolled",
                nondifferentiable_outputs=("active_set",),
                conditions=("ISTA gradients exclude soft-threshold knots",),
            ),
        )


class DictionaryLearning(AbstractRecipe):
    """Alternating proximal dictionary learning with an explicit initialization key."""

    n_components: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    code_iterations: int = eqx.field(static=True)
    transform_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        regularization: float = 1e-2,
        max_iterations: int = 100,
        code_iterations: int = 32,
        transform_iterations: int = 64,
        tolerance: float = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.n_components = int(n_components)
        self.regularization = float(regularization)
        self.max_iterations = int(max_iterations)
        self.code_iterations = int(code_iterations)
        self.transform_iterations = int(transform_iterations)
        self.tolerance = float(tolerance)
        self.weight_policy = weight_policy
        if (
            min(
                self.n_components,
                self.max_iterations,
                self.code_iterations,
                self.transform_iterations,
            )
            <= 0
        ):
            raise ValueError(
                "DictionaryLearning dimensions and iterations must be positive."
            )
        if self.regularization < 0.0 or self.tolerance < 0.0:
            raise ValueError(
                "DictionaryLearning regularization and tolerance are invalid."
            )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("DictionaryLearning fitting requires an explicit JAX key.")
        values, weights, feature_mask = _prepare(batch, self.weight_policy)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(jnp.float32)
        weights = weights.astype(values.real.dtype)
        case_shape = tuple(int(size) for size in batch.case_shape)
        dictionary = jax.random.normal(
            key,
            case_shape + (self.n_components, batch.feature_count),
            dtype=values.dtype,
        )
        dictionary = _normalize_atoms(dictionary)

        def body(_, current):
            codes = _ista_codes(
                values,
                current,
                self.regularization,
                self.code_iterations,
                mask=feature_mask,
            )
            residual = jnp.where(feature_mask, codes @ current - values, 0)
            gradient = jnp.swapaxes(jnp.conj(codes), -1, -2) @ (
                weights[..., :, None] * residual
            )
            code_scale = jnp.sum(
                weights[..., :, None] * jnp.real(codes * jnp.conj(codes)), axis=(-2, -1)
            )
            step = 1.0 / jnp.maximum(code_scale, jnp.finfo(values.real.dtype).tiny)
            return _normalize_atoms(current - step[..., None, None] * gradient)

        dictionary = jax.lax.fori_loop(0, self.max_iterations, body, dictionary)
        codes = _ista_codes(
            values,
            dictionary,
            self.regularization,
            self.transform_iterations,
            mask=feature_mask,
        )
        residual = jnp.where(feature_mask, values - codes @ dictionary, 0)
        reconstruction = jnp.sum(
            weights[..., :, None] * jnp.real(residual * jnp.conj(residual)), axis=(-2, -1)
        )
        penalty = self.regularization * jnp.sum(jnp.abs(codes), axis=(-2, -1))
        objective = 0.5 * reconstruction + penalty
        gradient = jnp.swapaxes(jnp.conj(codes), -1, -2) @ (
            weights[..., :, None]
            * jnp.where(feature_mask, codes @ dictionary - values, 0)
        )
        stationarity = jnp.max(jnp.abs(gradient), axis=(-2, -1))
        converged = stationarity <= self.tolerance
        finite = jnp.all(jnp.isfinite(dictionary), axis=(-2, -1))
        weights_valid = batch.weights_valid(self.weight_policy)
        valid = finite & converged & weights_valid
        status = jnp.where(
            ~weights_valid,
            ML_INFEASIBLE,
            jnp.where(
                ~finite,
                ML_NONFINITE,
                jnp.where(converged, ML_SUCCESS, ML_NONCONVERGED),
            ),
        ).astype(jnp.int32)
        (
            singular,
            explained,
            numerical_rank,
            orthogonality,
            minimum_gap,
            repeated,
        ) = _dictionary_spectrum(dictionary)
        atom_error = jnp.max(jnp.abs(jnp.linalg.norm(dictionary, axis=-1) - 1.0), axis=-1)
        model = SparseCodingModel(
            dictionary,
            regularization=self.regularization,
            transform_iterations=self.transform_iterations,
        )
        data_energy = jnp.sum(
            weights[..., :, None] * jnp.real(values * jnp.conj(values)),
            axis=(-2, -1),
        )
        retained_energy = jnp.where(
            data_energy > 0.0,
            jnp.maximum(1.0 - reconstruction / data_energy, 0.0),
            0.0,
        )
        diagnostics = FactorizationDiagnostics(
            singular_values=singular,
            explained_energy=explained,
            retained_energy=retained_energy,
            residual_energy=reconstruction,
            weighted_orthogonality_error=orthogonality,
            minimum_eigengap=minimum_gap,
            repeated_spectrum=repeated,
            canonicalization_valid=jnp.zeros_like(valid),
            objective=objective,
            reconstruction_error=reconstruction,
            regularization=penalty,
            iterations=jnp.full(objective.shape, self.max_iterations, dtype=jnp.int32),
            converged=converged,
            numerical_rank=numerical_rank,
            atom_norm_error=atom_error,
            valid=valid,
            status=status,
            effective_samples=effective_sample_size(weights),
            sign_phase_convention="uncanonicalized signed/phase atoms; permutation symmetry remains",
            centering_provenance="origin",
            weighting_provenance=f"sample:{self.weight_policy}",
            mask_provenance="zero-extension",
            method="alternating-ista-dictionary-learning",
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="alternating-ista-dictionary-learning",
            gradient_contract=GradientContract(
                prediction_inputs="almost-everywhere",
                prediction_parameters="almost-everywhere",
                fit_features="almost-everywhere",
                fit_weights="conditional",
                fit_mode="unrolled",
                nondifferentiable_outputs=("active_set",),
                conditions=(
                    "unrolled sparse-code gradients exclude soft-threshold knots",
                    "atom normalization requires nonzero atoms",
                ),
            ),
        )


__all__ = [
    "DictionaryLearning",
    "FactorizationDiagnostics",
    "NMF",
    "NMFModel",
    "SparseCoding",
    "SparseCodingModel",
]
