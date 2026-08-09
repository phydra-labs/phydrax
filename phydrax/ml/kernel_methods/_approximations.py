#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel
from ..._model._binding import ModelBinding
from ...kernels import FiniteFeatureKernel, SquaredExponentialKernel
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitDiagnostics,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._sparse_features import SparseFeatures
from ._utils import (
    case_kernel_matrix,
    finite_array,
    query_kernel_matrix,
    validate_kernel,
    validated_weights,
)


def _size(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _case_matmul(matrix: Array, factor: Array, case_shape: tuple[int, ...]) -> Array:
    if not case_shape:
        return matrix @ factor
    cases = _size(case_shape)
    query_shape = matrix.shape[len(case_shape) : -1]
    q = _size(tuple(int(s) for s in query_shape)) if query_shape else 1
    output = jax.vmap(jnp.matmul)(
        matrix.reshape((cases, q, matrix.shape[-1])),
        factor.reshape((cases, factor.shape[-2], factor.shape[-1])),
    )
    return output.reshape(case_shape + query_shape + (factor.shape[-1],))


class KernelPCAModel(AbstractArrayModel):
    support: Array
    support_mask: Array
    normalized_weight: Array
    support_column_mean: Array
    total_mean: Array
    components: Array
    kernel: Any
    feature_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        support: Array,
        support_mask: Array,
        normalized_weight: Array,
        support_column_mean: Array,
        total_mean: Array,
        components: Array,
        kernel: Any,
        feature_count: int,
        component_count: int,
        case_shape: tuple[int, ...],
    ):
        self.support = support
        self.support_mask = support_mask
        self.normalized_weight = normalized_weight
        self.support_column_mean = support_column_mean
        self.total_mean = total_mean
        self.components = components
        self.kernel = kernel
        self.feature_count = int(feature_count)
        self.component_count = int(component_count)
        self.case_shape = tuple(int(size) for size in case_shape)
        self.in_size = self.feature_count
        self.out_size = self.component_count

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        cross, query_shape = query_kernel_matrix(
            self.kernel, jnp.asarray(x), self.support, self.case_shape
        )
        support_shape = (
            self.case_shape + (1,) * len(query_shape) + (self.support.shape[-2],)
        )
        support_mask = self.support_mask.reshape(support_shape)
        normalized_weight = self.normalized_weight.reshape(support_shape)
        cross = cross * support_mask
        row_mean = jnp.sum(cross * normalized_weight, axis=-1, keepdims=True)
        case_rank = len(self.case_shape)
        query_ndim = cross.ndim - case_rank - 1
        column_mean = self.support_column_mean.reshape(
            self.case_shape + (1,) * query_ndim + (self.support.shape[-2],)
        )
        total = self.total_mean.reshape(self.case_shape + (1,) * query_ndim + (1,))
        centered = cross - row_mean - column_mean + total
        return _case_matmul(centered, self.components, self.case_shape)


class KernelPCARecipe(AbstractRecipe):
    kernel: Any
    n_components: int = eqx.field(static=True)
    eigenvalue_floor: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        n_components: int,
        eigenvalue_floor: ArrayLike = 1e-10,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.kernel = validate_kernel(kernel)
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        self.n_components = int(n_components)
        self.eigenvalue_floor = jnp.asarray(eigenvalue_floor, dtype=float)
        if self.eigenvalue_floor.ndim != 0 or bool(self.eigenvalue_floor < 0):
            raise ValueError("eigenvalue_floor must be a nonnegative scalar.")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        if self.n_components > batch.sample_count:
            raise ValueError("n_components cannot exceed the fixed support capacity.")
        x = batch.dense_features()
        feature_valid = jnp.all(finite_array(x), axis=-1)
        x = jnp.where(finite_array(x), x, 0.0)
        weights = (
            validated_weights(batch.effective_weight(self.weight_policy)) * feature_valid
        )
        normalized = weights / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), jnp.finfo(weights.dtype).tiny
        )
        gram = case_kernel_matrix(self.kernel, x, x, batch.case_shape)
        column_mean = jnp.sum(normalized[..., :, None] * gram, axis=-2)
        row_mean = jnp.sum(gram * normalized[..., None, :], axis=-1)
        total_mean = jnp.sum(normalized * row_mean, axis=-1)
        centered = (
            gram
            - row_mean[..., :, None]
            - column_mean[..., None, :]
            + total_mean[..., None, None]
        )
        active = weights > 0
        centered = jnp.where(active[..., :, None] & active[..., None, :], centered, 0.0)
        eigenvalues, eigenvectors = jnp.linalg.eigh(centered)
        values = eigenvalues[..., -self.n_components :][..., ::-1]
        vectors = eigenvectors[..., :, -self.n_components :][..., ::-1]
        scale = jnp.sqrt(jnp.maximum(values, self.eigenvalue_floor))
        components = vectors / jnp.where(scale > 0, scale, 1.0)[..., None, :]
        effective = jnp.sum(active, axis=-1)
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        valid = finite & (effective > self.n_components)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective > self.n_components, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = KernelPCAModel(
            support=x,
            support_mask=active,
            normalized_weight=normalized,
            support_column_mean=column_mean,
            total_mean=total_mean,
            components=components,
            kernel=self.kernel,
            feature_count=batch.feature_count,
            component_count=self.n_components,
            case_shape=batch.case_shape,
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            objective=jnp.sum(jnp.maximum(values, 0.0), axis=-1),
            effective_samples=effective,
            rank=jnp.sum(values > self.eigenvalue_floor, axis=-1),
            method="weighted-kernel-eigh",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="spectral",
            conditions=("Selected eigenspace is separated and support mask is fixed.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="kernel-pca",
            gradient_contract=contract,
        )


class NystromModel(AbstractArrayModel):
    landmarks: Array
    whitening: Array
    kernel: Any
    feature_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        landmarks: Array,
        whitening: Array,
        kernel: Any,
        feature_count: int,
        component_count: int,
        case_shape: tuple[int, ...],
    ):
        self.landmarks = landmarks
        self.whitening = whitening
        self.kernel = kernel
        self.feature_count = int(feature_count)
        self.component_count = int(component_count)
        self.case_shape = tuple(int(size) for size in case_shape)
        self.in_size = self.feature_count
        self.out_size = self.component_count

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        cross, _ = query_kernel_matrix(
            self.kernel, jnp.asarray(x), self.landmarks, self.case_shape
        )
        return _case_matmul(cross, self.whitening, self.case_shape)

    def approximate_matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_features = self(left)
        right_features = self(right)
        return left_features @ jnp.swapaxes(jnp.conj(right_features), -1, -2)

    def as_kernel(self) -> FiniteFeatureKernel:
        if self.case_shape:
            raise ValueError(
                "A case-batched Nystrom map does not define one global kernel."
            )
        if jnp.iscomplexobj(self.whitening):
            raise TypeError("FiniteFeatureKernel currently represents real feature maps.")
        return FiniteFeatureKernel(
            self,
            jnp.eye(self.component_count, dtype=self.whitening.dtype),
            feature_map_id="nystrom",
            max_derivative_order=0,
        )


class NystromRecipe(AbstractRecipe):
    kernel: Any
    n_components: int = eqx.field(static=True)
    selection: Literal["even", "random"] = eqx.field(static=True)
    eigenvalue_floor: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        /,
        *,
        n_components: int,
        selection: Literal["even", "random"] = "even",
        eigenvalue_floor: ArrayLike = 1e-10,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.kernel = validate_kernel(kernel)
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        if selection not in ("even", "random"):
            raise ValueError("selection must be 'even' or 'random'.")
        self.n_components = int(n_components)
        self.selection = selection
        self.eigenvalue_floor = jnp.asarray(eigenvalue_floor, dtype=float)
        if self.eigenvalue_floor.ndim != 0 or bool(self.eigenvalue_floor <= 0):
            raise ValueError("eigenvalue_floor must be positive.")
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if self.n_components > batch.sample_count:
            raise ValueError("n_components cannot exceed sample capacity.")
        x = batch.dense_features()
        feature_valid = jnp.all(finite_array(x), axis=-1)
        x = jnp.where(finite_array(x), x, 0.0)
        weight = validated_weights(batch.effective_weight(self.weight_policy))
        active = batch.sample_mask & feature_valid & (weight > 0)
        base_score = jnp.broadcast_to(
            jnp.arange(batch.sample_count, dtype=float) / batch.sample_count,
            batch.case_shape + (batch.sample_count,),
        )
        if self.selection == "random":
            if key is None:
                raise ValueError(
                    "Random Nystrom landmark selection requires an explicit JAX key."
                )
            base_score = jr.uniform(key, batch.case_shape + (batch.sample_count,))
        selection_score = jnp.where(active, base_score, 2.0)
        indices = jnp.argsort(selection_score, axis=-1)[..., : self.n_components]
        gather_indices = jnp.broadcast_to(
            indices[..., :, None],
            batch.case_shape + (self.n_components, batch.feature_count),
        )
        landmarks = jnp.take_along_axis(x, gather_indices, axis=-2)
        gram = case_kernel_matrix(self.kernel, landmarks, landmarks, batch.case_shape)
        values, vectors = jnp.linalg.eigh(gram)
        retained = values > self.eigenvalue_floor
        safe_values = jnp.where(retained, values, 1.0)
        inverse_root = jnp.where(retained, jax.lax.rsqrt(safe_values), 0.0)
        whitening = (vectors * inverse_root[..., None, :]) @ jnp.swapaxes(
            jnp.conj(vectors), -1, -2
        )
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        rank = jnp.sum(values > self.eigenvalue_floor, axis=-1)
        effective = jnp.sum(active, axis=-1)
        valid = finite & (rank > 0) & (effective >= self.n_components)
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective >= self.n_components, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = NystromModel(
            landmarks=landmarks,
            whitening=whitening,
            kernel=self.kernel,
            feature_count=batch.feature_count,
            component_count=self.n_components,
            case_shape=batch.case_shape,
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=effective,
            rank=rank,
            condition=jnp.max(values, axis=-1)
            / jnp.maximum(
                jnp.min(
                    jnp.where(values > self.eigenvalue_floor, values, jnp.inf), axis=-1
                ),
                self.eigenvalue_floor,
            ),
            method=f"nystrom-{self.selection}",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional" if self.selection == "even" else "none",
            fit_hyperparameters="conditional",
            fit_mode="spectral",
            nondifferentiable_outputs=("landmark_indices",),
            conditions=("Landmark selection and eigenspace rank are fixed.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="nystrom",
            gradient_contract=contract,
        )


class RandomFourierFeatureModel(AbstractArrayModel):
    frequencies: Array
    phases: Array
    scale: Array
    feature_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        frequencies: Array,
        phases: Array,
        scale: Array,
        feature_count: int,
        component_count: int,
    ):
        self.frequencies = frequencies
        self.phases = phases
        self.scale = scale
        self.feature_count = int(feature_count)
        self.component_count = int(component_count)
        self.in_size = self.feature_count
        self.out_size = self.component_count

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        points = jnp.asarray(x)
        if points.shape[-1] != self.feature_count:
            raise ValueError("Query feature size does not match random frequencies.")
        return self.scale * jnp.cos(points @ self.frequencies.T + self.phases)

    def approximate_matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_features = self(left)
        right_features = self(right)
        return left_features @ jnp.swapaxes(right_features, -1, -2)

    def as_kernel(self) -> FiniteFeatureKernel:
        return FiniteFeatureKernel(
            self,
            jnp.eye(self.component_count, dtype=self.frequencies.dtype),
            feature_map_id="random-fourier-squared-exponential",
            max_derivative_order=None,
        )


class RandomFourierFeaturesRecipe(AbstractRecipe):
    kernel: SquaredExponentialKernel
    n_components: int = eqx.field(static=True)

    def __init__(self, kernel: SquaredExponentialKernel, /, *, n_components: int):
        if not isinstance(kernel, SquaredExponentialKernel):
            raise TypeError(
                "Exact random Fourier sampling is supported only for SquaredExponentialKernel."
            )
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")
        self.kernel = kernel
        self.n_components = int(n_components)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if isinstance(batch.features, SparseFeatures):
            raise TypeError(
                "RandomFourierFeaturesRecipe requires dense features; call "
                "SparseFeatures.to_dense() explicitly before fitting."
            )
        if key is None:
            raise ValueError(
                "Random Fourier feature fitting requires an explicit JAX key."
            )
        frequency_key, phase_key = jr.split(key)
        length_scale = jnp.broadcast_to(self.kernel.length_scale, (batch.feature_count,))
        frequencies = (
            jr.normal(
                frequency_key,
                (self.n_components, batch.feature_count),
                dtype=length_scale.dtype,
            )
            / length_scale
        )
        phases = jr.uniform(
            phase_key,
            (self.n_components,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=length_scale.dtype,
        )
        scale = jnp.sqrt(jnp.asarray(2.0 / self.n_components, dtype=length_scale.dtype))
        model = RandomFourierFeatureModel(
            frequencies=frequencies,
            phases=phases,
            scale=scale,
            feature_count=batch.feature_count,
            component_count=self.n_components,
        )
        valid = jnp.ones(batch.case_shape or (), dtype=bool)
        status = jnp.zeros(batch.case_shape or (), dtype=jnp.int32)
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            effective_samples=batch.sample_count,
            rank=self.n_components,
            method="squared-exponential-spectral-sampling",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_hyperparameters="conditional",
            fit_mode="stopped",
            nondifferentiable_outputs=("sampled_frequencies",),
            conditions=("Random draw is fixed by the explicit key.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="random-fourier-features",
            gradient_contract=contract,
        )


__all__ = [
    "KernelPCAModel",
    "KernelPCARecipe",
    "NystromModel",
    "NystromRecipe",
    "RandomFourierFeatureModel",
    "RandomFourierFeaturesRecipe",
]
