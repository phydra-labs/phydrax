#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel
from ..._model._binding import ModelBinding
from ...uq._gp_classification import (
    BernoulliGaussianProcessPosterior,
    CategoricalGaussianProcessPosterior,
    condition_bernoulli_gaussian_process,
    condition_categorical_gaussian_process,
)
from ...uq._gp_likelihood import GaussianProcessLikelihoodState
from ...uq._gp_scalar import (
    ExactGaussianProcessFactor,
    FiniteFeatureGaussianProcessFactor,
)
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
from ._utils import finite_array, validated_weights


def _size(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


class GaussianProcessClassifierModel(AbstractArrayModel):
    """Smooth class probabilities from UQ Laplace-conditioned GP factors."""

    posteriors: tuple[
        BernoulliGaussianProcessPosterior | CategoricalGaussianProcessPosterior, ...
    ]
    feature_count: int = eqx.field(static=True)
    class_count: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        posteriors: tuple[
            BernoulliGaussianProcessPosterior | CategoricalGaussianProcessPosterior, ...
        ],
        feature_count: int,
        class_count: int,
        case_shape: tuple[int, ...],
    ):
        self.posteriors = tuple(posteriors)
        self.feature_count = int(feature_count)
        self.class_count = int(class_count)
        self.case_shape = tuple(int(size) for size in case_shape)
        self.in_size = self.feature_count
        self.out_size = self.class_count

    _input_binding: ClassVar[ModelBinding] = ModelBinding.blockwise(input_mode="flat")

    def __call__(self, x: ArrayLike, /, *, key: Any = None) -> Array:
        del key
        points = jnp.asarray(x)
        if points.shape[-1] != self.feature_count:
            raise ValueError("Query feature size does not match GP observations.")
        if self.case_shape:
            if points.shape[: len(self.case_shape)] != self.case_shape:
                raise ValueError(
                    f"Query must begin with fitted case shape {self.case_shape}."
                )
            query_shape = points.shape[len(self.case_shape) : -1]
            q = _size(tuple(int(s) for s in query_shape)) if query_shape else 1
            cases = _size(self.case_shape)
            flat = points.reshape((cases, q, self.feature_count))
            outputs = []
            for case_index, posterior in enumerate(self.posteriors):
                probability = posterior.probabilities(flat[case_index])
                if self.class_count == 2 and probability.ndim == 1:
                    probability = jnp.stack((1.0 - probability, probability), axis=-1)
                outputs.append(probability)
            return jnp.stack(outputs).reshape(
                self.case_shape + query_shape + (self.class_count,)
            )
        probability = self.posteriors[0].probabilities(
            points.reshape((-1, self.feature_count))
        )
        if self.class_count == 2 and probability.ndim == 1:
            probability = jnp.stack((1.0 - probability, probability), axis=-1)
        return probability.reshape(points.shape[:-1] + (self.class_count,))

    def probabilities(self, x: ArrayLike, /) -> Array:
        return self(x)

    def predict(self, x: ArrayLike, /) -> Array:
        return jnp.argmax(self(x), axis=-1).astype(jnp.int32)


class GaussianProcessClassifierRecipe(AbstractRecipe):
    state: GaussianProcessLikelihoodState
    class_count: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    curvature_floor: Array
    weight_policy: WeightPolicy = eqx.field(static=True)

    def __init__(
        self,
        state: GaussianProcessLikelihoodState,
        /,
        *,
        class_count: int = 2,
        iterations: int = 12,
        curvature_floor: ArrayLike = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        if not isinstance(state, GaussianProcessLikelihoodState):
            raise TypeError("state must be a GaussianProcessLikelihoodState.")
        if int(class_count) < 2 or int(iterations) <= 0:
            raise ValueError("class_count must be at least two and iterations positive.")
        floor = jnp.asarray(curvature_floor, dtype=float)
        if floor.ndim != 0:
            raise ValueError("curvature_floor must be a scalar.")
        floor = eqx.error_if(
            floor,
            ~jnp.isfinite(floor) | (floor <= 0.0),
            "curvature_floor must be finite and positive.",
        )
        self.state = state
        self.class_count = int(class_count)
        self.iterations = int(iterations)
        self.curvature_floor = floor
        self.weight_policy = weight_policy

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        labels = batch.require_targets()
        if labels.shape != batch.case_shape + (batch.sample_count,):
            raise ValueError("GP classification requires one integer label per sample.")
        if not jnp.issubdtype(labels.dtype, jnp.integer) and not jnp.issubdtype(
            labels.dtype, jnp.bool_
        ):
            raise TypeError(
                "GP classification labels must use an integer or boolean dtype."
            )
        points = batch.dense_features()
        if jnp.iscomplexobj(points):
            raise TypeError(
                "Shared UQ Gaussian-process kernels require real coordinates."
            )
        feature_valid = jnp.all(finite_array(points), axis=-1)
        points = jnp.where(finite_array(points), points, 0.0)
        weights = (
            validated_weights(batch.effective_weight(self.weight_policy)) * feature_valid
        )
        if batch.target_mask is not None:
            weights = weights * batch.target_mask
        labels = jnp.where(weights > 0, labels, 0)
        labels = eqx.error_if(
            labels,
            jnp.any((weights > 0) & ((labels < 0) | (labels >= self.class_count))),
            "Active GP classification labels must lie in [0, class_count).",
        )
        cases = _size(batch.case_shape)
        point_cases = points.reshape((cases, batch.sample_count, batch.feature_count))
        label_cases = labels.reshape((cases, batch.sample_count))
        weight_cases = weights.reshape((cases, batch.sample_count))
        posteriors = []
        for case_index in range(cases):
            if self.class_count == 2:
                posterior = condition_bernoulli_gaussian_process(
                    point_cases[case_index],
                    label_cases[case_index],
                    state=self.state,
                    observation_weight=weight_cases[case_index],
                    iterations=self.iterations,
                    curvature_floor=self.curvature_floor,
                )
            else:
                posterior = condition_categorical_gaussian_process(
                    point_cases[case_index],
                    label_cases[case_index],
                    state=self.state,
                    observation_weight=weight_cases[case_index],
                    class_count=self.class_count,
                    iterations=self.iterations,
                    curvature_floor=self.curvature_floor,
                )
            posteriors.append(posterior)
        finite_cases = []
        for posterior in posteriors:
            factors = (
                (posterior,)
                if isinstance(posterior, BernoulliGaussianProcessPosterior)
                else posterior.factors
            )
            factor_finite = []
            for factor in factors:
                storage = factor.factor
                if isinstance(storage, ExactGaussianProcessFactor):
                    geometry_finite = jnp.all(jnp.isfinite(storage.cholesky))
                elif isinstance(storage, FiniteFeatureGaussianProcessFactor):
                    geometry_finite = (
                        jnp.all(jnp.isfinite(storage.features))
                        & jnp.all(jnp.isfinite(storage.diagonal))
                        & jnp.all(jnp.isfinite(storage.correction_cholesky))
                    )
                else:
                    raise TypeError("GP classifier received an unsupported factor type.")
                factor_finite.append(
                    jnp.all(jnp.isfinite(factor.pseudo_observations)) & geometry_finite
                )
            finite_cases.append(jnp.all(jnp.stack(tuple(factor_finite))))
        finite = jnp.stack(tuple(finite_cases)).reshape(batch.case_shape or ())
        effective = jnp.sum(weights > 0, axis=-1)
        valid = (effective > 0) & finite
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(effective > 0, ML_SUCCESS, ML_INSUFFICIENT_DATA),
        )
        model = GaussianProcessClassifierModel(
            posteriors=tuple(posteriors),
            feature_count=batch.feature_count,
            class_count=self.class_count,
            case_shape=batch.case_shape,
        )
        diagnostics = FitDiagnostics(
            valid=valid,
            status=status,
            iterations=self.iterations,
            effective_samples=effective,
            method="uq-gp-laplace-classification",
        )
        contract = GradientContract(
            prediction_inputs="smooth",
            prediction_parameters="smooth",
            fit_features="conditional",
            fit_targets="none",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="unrolled",
            nondifferentiable_outputs=("predict",),
            conditions=("Class labels and Newton iteration count are fixed.",),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="gp-classification",
            gradient_contract=contract,
        )


class BernoulliGaussianProcessClassifierRecipe(AbstractRecipe):
    recipe: GaussianProcessClassifierRecipe

    def __init__(
        self,
        state: GaussianProcessLikelihoodState,
        /,
        *,
        iterations: int = 12,
        curvature_floor: ArrayLike = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        self.recipe = GaussianProcessClassifierRecipe(
            state,
            class_count=2,
            iterations=iterations,
            curvature_floor=curvature_floor,
            weight_policy=weight_policy,
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return self.recipe.fit_batch(batch, key=key)


class CategoricalGaussianProcessClassifierRecipe(AbstractRecipe):
    recipe: GaussianProcessClassifierRecipe

    def __init__(
        self,
        state: GaussianProcessLikelihoodState,
        /,
        *,
        class_count: int,
        iterations: int = 12,
        curvature_floor: ArrayLike = 1e-6,
        weight_policy: WeightPolicy = "statistical",
    ):
        if int(class_count) <= 2:
            raise ValueError(
                "Categorical GP classification requires class_count greater than two."
            )
        self.recipe = GaussianProcessClassifierRecipe(
            state,
            class_count=class_count,
            iterations=iterations,
            curvature_floor=curvature_floor,
            weight_policy=weight_policy,
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return self.recipe.fit_batch(batch, key=key)


__all__ = [
    "BernoulliGaussianProcessClassifierRecipe",
    "CategoricalGaussianProcessClassifierRecipe",
    "GaussianProcessClassifierModel",
    "GaussianProcessClassifierRecipe",
]
