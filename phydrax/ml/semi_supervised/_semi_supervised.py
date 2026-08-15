#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from ...kernels import AbstractPositiveDefiniteKernel, SquaredExponentialKernel
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
)
from .._sparse_features import SparseFeatures


class GraphFitDiagnostics(StrictModule):
    residual: Array
    labelled_samples: Array
    iterations: Array
    valid: Array
    status: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        residual: Any,
        labelled_samples: Any,
        /,
        *,
        iterations: int,
        valid: Any,
        status: Any,
        method: str,
    ):
        self.residual = jnp.asarray(residual)
        self.labelled_samples = jnp.asarray(labelled_samples)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.method = str(method)


class SelfTrainingDiagnostics(StrictModule):
    confidence: Array
    labelled_samples: Array
    child_status: Array
    valid: Array
    status: Array
    iterations: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        confidence: Any,
        labelled_samples: Any,
        child_status: Any,
        /,
        *,
        valid: Any,
        status: Any,
        iterations: int,
        method: str,
    ):
        self.confidence = jnp.asarray(confidence)
        self.labelled_samples = jnp.asarray(labelled_samples)
        self.child_status = jnp.asarray(child_status, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.method = str(method)


def _validated_sample_weight(batch: MLBatch) -> Array:
    weights = jnp.asarray(batch.sample_weight)
    return eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
        "Sample weights must be finite and nonnegative.",
    )


def _kernel_matrix_cases(
    kernel: AbstractPositiveDefiniteKernel, left: Array, right: Array
) -> Array:
    case_shape = left.shape[:-2]
    if right.shape[:-2] != case_shape or left.shape[-1] != right.shape[-1]:
        raise ValueError("Kernel designs must share case and feature axes.")
    cases = 1
    for size in case_shape:
        cases *= int(size)
    left_cases = left.reshape((cases, left.shape[-2], left.shape[-1]))
    right_cases = right.reshape((cases, right.shape[-2], right.shape[-1]))
    matrix = jax.vmap(kernel.matrix)(left_cases, right_cases)
    return matrix.reshape(case_shape + (left.shape[-2], right.shape[-2]))


def _normalise_probabilities(value: Array) -> Array:
    nonnegative = jnp.maximum(jnp.real(value), 0.0)
    total = jnp.sum(nonnegative, axis=-1, keepdims=True)
    classes = value.shape[-1]
    uniform = jnp.full_like(nonnegative, 1.0 / classes)
    return jnp.where(
        total > 0.0,
        nonnegative / jnp.maximum(total, jnp.finfo(nonnegative.dtype).tiny),
        uniform,
    )


def _target_distributions(
    batch: MLBatch, num_classes: int | None
) -> tuple[Array, Array, int, Array]:
    targets = batch.require_targets()
    sample_ndim = len(batch.case_shape) + 1
    if jnp.issubdtype(targets.dtype, jnp.complexfloating):
        raise TypeError("Class distributions cannot be complex.")
    schema_labels = batch.target_schema.class_labels
    if targets.ndim == sample_ndim:
        classes = len(schema_labels) if num_classes is None else int(num_classes)
        if classes < 2:
            raise ValueError(
                "num_classes or a target-schema class vocabulary is required "
                "for hard labels."
            )
        if schema_labels and len(schema_labels) != classes:
            raise ValueError(
                "num_classes does not match the target-schema class vocabulary."
            )
        class_labels = (
            jnp.asarray(schema_labels)
            if schema_labels
            else jnp.arange(classes, dtype=jnp.int32)
        )
        if class_labels.ndim != 1 or jnp.issubdtype(
            class_labels.dtype, jnp.complexfloating
        ):
            raise TypeError("Class labels must be a real one-dimensional vocabulary.")
        target_mask = batch.target_mask
        labelled = batch.sample_mask & (
            jnp.ones_like(targets, dtype=bool) if target_mask is None else target_mask
        )
        matches = targets[..., None] == class_labels
        targets = eqx.error_if(
            targets,
            jnp.any(labelled & (~jnp.isfinite(targets) | ~jnp.any(matches, axis=-1))),
            "Every labelled hard target must occur in the class vocabulary.",
        )
        distribution_dtype = jnp.result_type(targets.dtype, jnp.float32)
        distributions = matches.astype(distribution_dtype)
    elif targets.ndim == sample_ndim + 1:
        classes = int(targets.shape[-1])
        if num_classes is not None and int(num_classes) != classes:
            raise ValueError("num_classes does not match the soft-target axis.")
        if schema_labels and len(schema_labels) != classes:
            raise ValueError(
                "The target-schema class vocabulary does not match the soft-target axis."
            )
        class_labels = (
            jnp.asarray(schema_labels)
            if schema_labels
            else jnp.arange(classes, dtype=jnp.int32)
        )
        if class_labels.ndim != 1 or jnp.issubdtype(
            class_labels.dtype, jnp.complexfloating
        ):
            raise TypeError("Class labels must be a real one-dimensional vocabulary.")
        target_mask = batch.target_mask
        if target_mask is not None:
            partially_labelled = jnp.any(target_mask, axis=-1) & ~jnp.all(
                target_mask, axis=-1
            )
            targets = eqx.error_if(
                targets,
                jnp.any(batch.sample_mask & partially_labelled),
                "Soft label masks must select either every class or no class.",
            )
        labelled = batch.sample_mask & (
            jnp.ones(targets.shape[:-1], dtype=bool)
            if target_mask is None
            else jnp.all(target_mask, axis=-1)
        )
        targets = eqx.error_if(
            targets,
            jnp.any(
                labelled
                & (
                    ~jnp.all(jnp.isfinite(targets), axis=-1)
                    | jnp.any(targets < 0.0, axis=-1)
                    | (jnp.sum(targets, axis=-1) <= 0.0)
                )
            ),
            "Labelled soft targets must be finite nonnegative distributions.",
        )
        distributions = _normalise_probabilities(targets)
    else:
        raise ValueError(
            "Graph propagation requires hard labels or one class-probability "
            "vector per sample."
        )
    distributions = jnp.where(labelled[..., None], distributions, 0.0)
    return distributions, labelled, classes, class_labels


class LabelPropagationModel(AbstractArrayModel):
    """Blockwise kernel interpolation of propagated class distributions."""

    training_features: Array
    distributions: Array
    training_weight: Array
    prior: Array
    class_labels: Array
    kernel: AbstractPositiveDefiniteKernel
    case_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding = ModelBinding.blockwise("structured")

    def __init__(
        self,
        training_features: Any,
        distributions: Any,
        training_weight: Any,
        kernel: AbstractPositiveDefiniteKernel,
        /,
        *,
        class_labels: Any = None,
    ):
        x = jnp.asarray(training_features)
        probabilities = jnp.asarray(distributions)
        weights = jnp.asarray(training_weight)
        labels = (
            jnp.arange(probabilities.shape[-1], dtype=jnp.int32)
            if class_labels is None
            else jnp.asarray(class_labels)
        )
        if labels.shape != (probabilities.shape[-1],):
            raise ValueError("class_labels must align with the class axis.")
        if jnp.issubdtype(labels.dtype, jnp.complexfloating):
            raise TypeError("class_labels must be real-valued.")
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel.")
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("Graph propagation kernels require real-valued features.")
        if jnp.issubdtype(probabilities.dtype, jnp.complexfloating):
            raise TypeError("Class distributions must be real-valued.")
        weights = eqx.error_if(
            weights,
            jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
            "Training weights must be finite and nonnegative.",
        )
        if (
            x.ndim < 2
            or probabilities.shape[:-1] != x.shape[:-1]
            or weights.shape != x.shape[:-1]
        ):
            raise ValueError("Training features, distributions, and weights must align.")
        weighted = weights[..., None] * probabilities
        prior = jnp.sum(weighted, axis=-2) / jnp.maximum(
            jnp.sum(weights, axis=-1, keepdims=True), jnp.finfo(weights.dtype).tiny
        )
        self.training_features = x
        self.distributions = probabilities
        self.training_weight = weights
        self.prior = _normalise_probabilities(prior)
        self.kernel = kernel
        self.class_labels = labels
        self.case_shape = tuple(int(size) for size in x.shape[:-2])
        self.in_size = int(x.shape[-1])
        self.out_size = int(probabilities.shape[-1])

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        points = jnp.asarray(x)
        if jnp.issubdtype(points.dtype, jnp.complexfloating):
            raise TypeError("Graph propagation kernels require real-valued features.")
        if points.ndim == len(self.case_shape) + 1:
            points = points[..., None, :]
            squeeze = True
        else:
            squeeze = False
        if points.shape[:-2] != self.case_shape or points.shape[-1] != self.in_size:
            raise ValueError(
                "LabelPropagationModel expects case_shape + (point, feature)."
            )
        cross = _kernel_matrix_cases(self.kernel, points, self.training_features)
        weighted = cross * self.training_weight[..., None, :]
        numerator = oe.contract("...mn,...nc->...mc", weighted, self.distributions)
        denominator = jnp.sum(weighted, axis=-1, keepdims=True)
        prior = self.prior[..., None, :]
        probabilities = jnp.where(
            denominator > 0.0,
            numerator / jnp.maximum(denominator, jnp.finfo(weighted.dtype).tiny),
            prior,
        )
        probabilities = _normalise_probabilities(probabilities)
        return probabilities[..., 0, :] if squeeze else probabilities


class HardLabelPropagationModel(AbstractArrayModel):
    """Exact class reporting counterpart to LabelPropagationModel."""

    soft_model: LabelPropagationModel
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.blockwise("structured")

    def __init__(self, soft_model: LabelPropagationModel, /):
        self.soft_model = soft_model
        self.in_size = soft_model.in_size
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        indices = jnp.argmax(self.soft_model(x, key=key), axis=-1)
        return jax.lax.stop_gradient(self.soft_model.class_labels[indices])


class LabelPropagationRecipe(AbstractRecipe):
    kernel: AbstractPositiveDefiniteKernel
    iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    num_classes: int | None = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel | None = None,
        /,
        *,
        iterations: int = 100,
        tolerance: float = 1e-6,
        num_classes: int | None = None,
    ):
        if int(iterations) <= 0 or float(tolerance) < 0.0:
            raise ValueError("iterations must be positive and tolerance nonnegative.")
        if kernel is not None and not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel.")
        self.kernel = SquaredExponentialKernel() if kernel is None else kernel
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.num_classes = None if num_classes is None else int(num_classes)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x = batch.dense_features()
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("Graph propagation kernels require real-valued features.")
        targets, labelled, _, class_labels = _target_distributions(
            batch, self.num_classes
        )
        weights = _validated_sample_weight(batch)
        valid_weight = jnp.where(batch.sample_mask, weights, 0.0)
        graph = _kernel_matrix_cases(self.kernel, x, x)
        identity = jnp.eye(batch.sample_count, dtype=bool)
        graph = jnp.where(identity, 0.0, graph)
        graph = graph * valid_weight[..., :, None] * valid_weight[..., None, :]
        propagation_dtype = jnp.result_type(graph.dtype, targets.dtype)
        graph = graph.astype(propagation_dtype)
        targets = targets.astype(propagation_dtype)
        transition = graph / jnp.maximum(
            jnp.sum(graph, axis=-1, keepdims=True), jnp.finfo(graph.dtype).tiny
        )

        def update(_, current):
            propagated = oe.contract("...nm,...mc->...nc", transition, current)
            return jnp.where(labelled[..., None], targets, propagated)

        distributions = jax.lax.fori_loop(0, self.iterations, update, targets)
        distributions = _normalise_probabilities(distributions)
        residual = jnp.max(
            jnp.abs(update(0, distributions) - distributions), axis=(-2, -1)
        )
        labelled_count = jnp.sum(labelled, axis=-1)
        enough = labelled_count > 0
        finite = jnp.isfinite(residual)
        converged = residual <= self.tolerance
        valid = enough & finite & converged
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough,
                ML_INSUFFICIENT_DATA,
                jnp.where(converged, ML_SUCCESS, ML_NONCONVERGED),
            ),
        )
        diagnostics = GraphFitDiagnostics(
            residual,
            labelled_count,
            iterations=self.iterations,
            valid=valid,
            status=status,
            method="label-propagation",
        )
        model = LabelPropagationModel(
            x,
            distributions,
            valid_weight,
            self.kernel,
            class_labels=class_labels,
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method="label-propagation",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_hyperparameters="conditional",
                fit_mode="unrolled",
                conditions=("The class vocabulary and labelled mask are fixed.",),
            ),
        )


class LabelSpreadingRecipe(AbstractRecipe):
    kernel: AbstractPositiveDefiniteKernel
    alpha: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    num_classes: int | None = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractPositiveDefiniteKernel | None = None,
        /,
        *,
        alpha: float = 0.8,
        iterations: int = 100,
        tolerance: float = 1e-6,
        num_classes: int | None = None,
    ):
        if not 0.0 < float(alpha) < 1.0 or int(iterations) <= 0 or float(tolerance) < 0.0:
            raise ValueError(
                "alpha must be in (0, 1), iterations positive, and tolerance nonnegative."
            )
        if kernel is not None and not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be an AbstractPositiveDefiniteKernel.")
        self.kernel = SquaredExponentialKernel() if kernel is None else kernel
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.num_classes = None if num_classes is None else int(num_classes)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        x = batch.dense_features()
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError("Graph spreading kernels require real-valued features.")
        targets, labelled, _, class_labels = _target_distributions(
            batch, self.num_classes
        )
        weights = _validated_sample_weight(batch)
        valid_weight = jnp.where(batch.sample_mask, weights, 0.0)
        graph = _kernel_matrix_cases(self.kernel, x, x)
        identity = jnp.eye(batch.sample_count, dtype=bool)
        graph = jnp.where(identity, 0.0, graph)
        graph = graph * valid_weight[..., :, None] * valid_weight[..., None, :]
        propagation_dtype = jnp.result_type(graph.dtype, targets.dtype)
        graph = graph.astype(propagation_dtype)
        targets = targets.astype(propagation_dtype)
        degree = jnp.sum(graph, axis=-1)
        inverse_sqrt = jax.lax.rsqrt(jnp.maximum(degree, jnp.finfo(graph.dtype).tiny))
        transition = inverse_sqrt[..., :, None] * graph * inverse_sqrt[..., None, :]
        alpha = jnp.asarray(self.alpha, dtype=transition.dtype)

        def update(_, current):
            return (
                alpha * oe.contract("...nm,...mc->...nc", transition, current)
                + (jnp.ones((), dtype=transition.dtype) - alpha) * targets
            )

        distributions = jax.lax.fori_loop(0, self.iterations, update, targets)
        distributions = _normalise_probabilities(distributions)
        residual = jnp.max(
            jnp.abs(update(0, distributions) - distributions), axis=(-2, -1)
        )
        labelled_count = jnp.sum(labelled, axis=-1)
        enough = labelled_count > 0
        finite = jnp.isfinite(residual)
        converged = residual <= self.tolerance
        valid = enough & finite & converged
        status = jnp.where(
            ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough,
                ML_INSUFFICIENT_DATA,
                jnp.where(converged, ML_SUCCESS, ML_NONCONVERGED),
            ),
        )
        diagnostics = GraphFitDiagnostics(
            residual,
            labelled_count,
            iterations=self.iterations,
            valid=valid,
            status=status,
            method="label-spreading",
        )
        return FitResult(
            LabelPropagationModel(
                x,
                distributions,
                valid_weight,
                self.kernel,
                class_labels=class_labels,
            ),
            diagnostics,
            valid=valid,
            status=status,
            method="label-spreading",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_hyperparameters="conditional",
                fit_mode="unrolled",
                conditions=("The class vocabulary and labelled mask are fixed.",),
            ),
        )


class HardLabelPropagationRecipe(AbstractRecipe):
    soft_recipe: LabelPropagationRecipe | LabelSpreadingRecipe

    def __init__(self, soft_recipe: LabelPropagationRecipe | LabelSpreadingRecipe, /):
        if not isinstance(soft_recipe, (LabelPropagationRecipe, LabelSpreadingRecipe)):
            raise TypeError("soft_recipe must be label propagation or label spreading.")
        self.soft_recipe = soft_recipe

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        result = self.soft_recipe.fit_batch(batch, key=key)
        model = result.as_trainable()
        if not isinstance(model, LabelPropagationModel):
            raise TypeError("soft_recipe returned an incompatible model.")
        return FitResult(
            HardLabelPropagationModel(model),
            result.diagnostics,
            valid=result.valid,
            status=result.status,
            method="hard-label-propagation",
            gradient_contract=GradientContract(
                prediction_inputs="none",
                prediction_parameters="none",
                fit_mode="stopped",
                nondifferentiable_outputs=("class_index",),
            ),
        )


def _replace_targets(
    batch: MLBatch,
    targets: Any,
    /,
    *,
    target_mask: Any,
    sample_weight: Any,
) -> MLBatch:
    return MLBatch(
        batch.features,
        targets,
        feature_mask=None
        if isinstance(batch.features, SparseFeatures)
        else batch.feature_mask,
        target_mask=target_mask,
        sample_mask=batch.sample_mask,
        sample_weight=sample_weight,
        measure_weight=batch.measure_weight,
        groups=batch.groups,
        feature_schema=batch.feature_schema,
        target_schema=batch.target_schema,
    )


def _self_training_targets(batch: MLBatch) -> tuple[Array, Array]:
    targets = batch.require_targets()
    sample_ndim = len(batch.case_shape) + 1
    if targets.ndim != sample_ndim + 1 or targets.shape[-1] < 2:
        raise ValueError("Self-training requires one soft class vector per sample.")
    if jnp.issubdtype(targets.dtype, jnp.complexfloating):
        raise TypeError("Class probabilities cannot be complex.")
    if batch.target_mask is not None:
        partially_labelled = jnp.any(batch.target_mask, axis=-1) & ~jnp.all(
            batch.target_mask, axis=-1
        )
        targets = eqx.error_if(
            targets,
            jnp.any(batch.sample_mask & partially_labelled),
            "Self-training masks must select either every class or no class.",
        )
    labelled = batch.sample_mask & (
        jnp.ones(targets.shape[:-1], dtype=bool)
        if batch.target_mask is None
        else jnp.all(batch.target_mask, axis=-1)
    )
    targets = eqx.error_if(
        targets,
        jnp.any(
            labelled
            & (
                ~jnp.all(jnp.isfinite(targets), axis=-1)
                | jnp.any(targets < 0.0, axis=-1)
                | (jnp.sum(targets, axis=-1) <= 0.0)
            )
        ),
        "Labelled self-training targets must be finite nonnegative distributions.",
    )
    known = _normalise_probabilities(targets)
    uniform = jnp.full_like(known, 1.0 / targets.shape[-1])
    return jnp.where(labelled[..., None], known, uniform), labelled


class SoftSelfTrainingModel(AbstractArrayModel):
    model: AbstractArrayModel
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(self, model: AbstractArrayModel, /):
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return self.model(x, key=key)


class HardSelfTrainingModel(AbstractArrayModel):
    model: AbstractArrayModel
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(self, model: AbstractArrayModel, /):
        self.model = model
        self.in_size = model.in_size
        self.out_size = model.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        return self.model(x, key=key)


class SoftSelfTrainingRecipe(AbstractRecipe):
    recipe: AbstractRecipe
    iterations: int = eqx.field(static=True)
    blend: float = eqx.field(static=True)

    def __init__(
        self, recipe: AbstractRecipe, /, *, iterations: int = 5, blend: float = 1.0
    ):
        if not isinstance(recipe, AbstractRecipe):
            raise TypeError("recipe must be an AbstractRecipe.")
        if int(iterations) <= 0 or not 0.0 < float(blend) <= 1.0:
            raise ValueError("iterations must be positive and blend in (0, 1].")
        self.recipe = recipe
        self.iterations = int(iterations)
        self.blend = float(blend)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("SoftSelfTrainingRecipe requires an explicit JAX key.")
        current, labelled = _self_training_targets(batch)
        base_weight = _validated_sample_weight(batch)
        known = current
        confidence = labelled.astype(base_weight.dtype)
        statuses = []
        result = None
        for step in range(self.iterations):
            weights = base_weight * jnp.where(labelled, 1.0, confidence)
            training = _replace_targets(
                batch,
                current,
                target_mask=jnp.ones_like(current, dtype=bool),
                sample_weight=weights,
            )
            result = self.recipe.fit_batch(training, key=jr.fold_in(key, step))
            statuses.append(result.status)
            probabilities = _normalise_probabilities(
                jnp.asarray(
                    result.model(batch.dense_features(), key=jr.fold_in(key, 1000 + step))
                )
            )
            if probabilities.shape != current.shape:
                raise ValueError(
                    "The child model must return one class vector per sample."
                )
            updated = (1.0 - self.blend) * current + self.blend * probabilities
            current = jnp.where(
                labelled[..., None], known, _normalise_probabilities(updated)
            )
            confidence = jnp.where(labelled, 1.0, jnp.sum(current * current, axis=-1))
        final_training = _replace_targets(
            batch,
            current,
            target_mask=jnp.ones_like(current, dtype=bool),
            sample_weight=base_weight * jnp.where(labelled, 1.0, confidence),
        )
        result = self.recipe.fit_batch(
            final_training, key=jr.fold_in(key, self.iterations)
        )
        statuses.append(result.status)
        valid = result.valid
        status = result.status
        diagnostics = SelfTrainingDiagnostics(
            confidence,
            jnp.sum(labelled, axis=-1),
            jnp.stack(statuses),
            valid=valid,
            status=status,
            iterations=self.iterations,
            method="soft-self-training",
        )
        return FitResult(
            SoftSelfTrainingModel(result.as_trainable()),
            diagnostics,
            valid=valid,
            status=status,
            method="soft-self-training",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_hyperparameters="conditional",
                fit_mode="unrolled",
                conditions=("The labelled mask and class axis are fixed.",),
            ),
        )


class HardSelfTrainingRecipe(AbstractRecipe):
    recipe: AbstractRecipe
    iterations: int = eqx.field(static=True)
    confidence_threshold: float = eqx.field(static=True)

    def __init__(
        self,
        recipe: AbstractRecipe,
        /,
        *,
        iterations: int = 5,
        confidence_threshold: float = 0.0,
    ):
        if not isinstance(recipe, AbstractRecipe):
            raise TypeError("recipe must be an AbstractRecipe.")
        if int(iterations) <= 0 or not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError(
                "iterations must be positive and confidence_threshold in [0, 1]."
            )
        self.recipe = recipe
        self.iterations = int(iterations)
        self.confidence_threshold = float(confidence_threshold)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("HardSelfTrainingRecipe requires an explicit JAX key.")
        current, labelled = _self_training_targets(batch)
        base_weight = _validated_sample_weight(batch)
        known = current
        confidence = labelled.astype(base_weight.dtype)
        statuses = []
        result = None
        for step in range(self.iterations):
            accepted = (
                labelled
                if step == 0
                else labelled | (confidence >= self.confidence_threshold)
            )
            training = _replace_targets(
                batch,
                current,
                target_mask=jnp.broadcast_to(accepted[..., None], current.shape),
                sample_weight=base_weight * accepted,
            )
            result = self.recipe.fit_batch(training, key=jr.fold_in(key, step))
            statuses.append(result.status)
            probabilities = _normalise_probabilities(
                jnp.asarray(
                    result.model(batch.dense_features(), key=jr.fold_in(key, 1000 + step))
                )
            )
            if probabilities.shape != current.shape:
                raise ValueError(
                    "The child model must return one class vector per sample."
                )
            indices = jax.lax.stop_gradient(jnp.argmax(probabilities, axis=-1))
            pseudo = jax.lax.stop_gradient(
                jax.nn.one_hot(indices, current.shape[-1], dtype=current.dtype)
            )
            current = jnp.where(labelled[..., None], known, pseudo)
            confidence = jax.lax.stop_gradient(jnp.max(probabilities, axis=-1))
        accepted = labelled | (confidence >= self.confidence_threshold)
        final_training = _replace_targets(
            batch,
            current,
            target_mask=jnp.broadcast_to(accepted[..., None], current.shape),
            sample_weight=base_weight * accepted,
        )
        result = self.recipe.fit_batch(
            final_training, key=jr.fold_in(key, self.iterations)
        )
        statuses.append(result.status)
        valid = result.valid
        status = result.status
        diagnostics = SelfTrainingDiagnostics(
            confidence,
            jnp.sum(labelled, axis=-1),
            jnp.stack(statuses),
            valid=valid,
            status=status,
            iterations=self.iterations,
            method="hard-self-training",
        )
        return FitResult(
            HardSelfTrainingModel(result.as_trainable()),
            diagnostics,
            valid=valid,
            status=status,
            method="hard-self-training",
            gradient_contract=GradientContract(
                prediction_inputs="conditional",
                prediction_parameters="conditional",
                fit_mode="stopped",
                nondifferentiable_outputs=("pseudo_label", "pseudo_label_acceptance"),
                conditions=(
                    "Inference gradients are those of the final child model only.",
                ),
            ),
        )


def _score_vector(score: Array, leading_shape: tuple[int, ...]) -> Array:
    if score.shape == leading_shape + (1,):
        return score[..., 0]
    if score.shape != leading_shape:
        raise ValueError("One-class detector must return one scalar score per sample.")
    return score


class SoftOneClassCompositionModel(AbstractArrayModel):
    detector: AbstractArrayModel
    predictor: AbstractArrayModel
    threshold: Array
    temperature: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self,
        detector: AbstractArrayModel,
        predictor: AbstractArrayModel,
        /,
        *,
        threshold: Any,
        temperature: Any,
    ):
        if detector.in_size != predictor.in_size:
            raise ValueError("detector and predictor must share an input size.")
        self.detector = detector
        self.predictor = predictor
        self.threshold = jnp.asarray(threshold)
        self.temperature = eqx.error_if(
            jnp.asarray(temperature),
            jnp.any(jnp.asarray(temperature) <= 0.0),
            "temperature must be positive.",
        )
        self.in_size = predictor.in_size
        self.out_size = predictor.out_size

    def acceptance(self, x: Any, /, *, key: Any = None) -> Array:
        points = jnp.asarray(x)
        score = _score_vector(
            jnp.asarray(self.detector(points, key=key)), points.shape[:-1]
        )
        return jax.nn.sigmoid((score - self.threshold) / self.temperature)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        prediction = jnp.asarray(self.predictor(x, key=key))
        gate = self.acceptance(x, key=key)
        return prediction * gate.reshape(
            gate.shape + (1,) * (prediction.ndim - gate.ndim)
        )


class HardOneClassCompositionModel(AbstractArrayModel):
    detector: AbstractArrayModel
    predictor: AbstractArrayModel
    threshold: Array
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    _input_binding = ModelBinding.pointwise()

    def __init__(
        self,
        detector: AbstractArrayModel,
        predictor: AbstractArrayModel,
        /,
        *,
        threshold: Any,
    ):
        if detector.in_size != predictor.in_size:
            raise ValueError("detector and predictor must share an input size.")
        self.detector = detector
        self.predictor = predictor
        self.threshold = jnp.asarray(threshold)
        self.in_size = predictor.in_size
        self.out_size = predictor.out_size

    def acceptance(self, x: Any, /, *, key: Any = None) -> Array:
        points = jnp.asarray(x)
        score = _score_vector(
            jnp.asarray(self.detector(points, key=key)), points.shape[:-1]
        )
        return jax.lax.stop_gradient(score >= self.threshold)

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        prediction = jnp.asarray(self.predictor(x, key=key))
        accepted = self.acceptance(x, key=key)
        return jnp.where(
            accepted.reshape(accepted.shape + (1,) * (prediction.ndim - accepted.ndim)),
            prediction,
            0,
        )


def _unsupervised_batch(batch: MLBatch) -> MLBatch:
    return MLBatch(
        batch.features,
        feature_mask=None
        if isinstance(batch.features, SparseFeatures)
        else batch.feature_mask,
        sample_mask=batch.sample_mask,
        sample_weight=batch.sample_weight,
        measure_weight=batch.measure_weight,
        groups=batch.groups,
        feature_schema=batch.feature_schema,
    )


class SoftOneClassCompositionRecipe(AbstractRecipe):
    detector_recipe: AbstractRecipe
    predictor_recipe: AbstractRecipe
    threshold: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)

    def __init__(
        self,
        detector_recipe: AbstractRecipe,
        predictor_recipe: AbstractRecipe,
        /,
        *,
        threshold: float = 0.0,
        temperature: float = 0.1,
    ):
        if not isinstance(detector_recipe, AbstractRecipe) or not isinstance(
            predictor_recipe, AbstractRecipe
        ):
            raise TypeError("detector_recipe and predictor_recipe must be recipes.")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive.")
        self.detector_recipe = detector_recipe
        self.predictor_recipe = predictor_recipe
        self.threshold = float(threshold)
        self.temperature = float(temperature)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError(
                "SoftOneClassCompositionRecipe requires an explicit JAX key."
            )
        base_weight = _validated_sample_weight(batch)
        detector_result = self.detector_recipe.fit_batch(
            _unsupervised_batch(batch), key=jr.fold_in(key, 1)
        )
        score = _score_vector(
            jnp.asarray(
                detector_result.model(batch.dense_features(), key=jr.fold_in(key, 2))
            ),
            batch.sample_mask.shape,
        )
        gate = jax.nn.sigmoid((score - self.threshold) / self.temperature)
        predictor_batch = _replace_targets(
            batch,
            batch.require_targets(),
            target_mask=batch.target_mask,
            sample_weight=base_weight * gate,
        )
        predictor_result = self.predictor_recipe.fit_batch(
            predictor_batch, key=jr.fold_in(key, 3)
        )
        valid = detector_result.valid & predictor_result.valid
        status = jnp.where(
            valid,
            ML_SUCCESS,
            jnp.maximum(detector_result.status, predictor_result.status),
        )
        diagnostics = SelfTrainingDiagnostics(
            gate,
            jnp.sum(batch.sample_mask, axis=-1),
            jnp.stack((detector_result.status, predictor_result.status)),
            valid=valid,
            status=status,
            iterations=1,
            method="soft-one-class-composition",
        )
        return FitResult(
            SoftOneClassCompositionModel(
                detector_result.as_trainable(),
                predictor_result.as_trainable(),
                threshold=self.threshold,
                temperature=self.temperature,
            ),
            diagnostics,
            valid=valid,
            status=status,
            method="soft-one-class-composition",
            gradient_contract=GradientContract(
                fit_features="conditional",
                fit_targets="conditional",
                fit_weights="conditional",
                fit_hyperparameters="conditional",
                fit_mode="unrolled",
                conditions=(
                    "Detector and predictor recipes must expose compatible gradients.",
                ),
            ),
        )


class HardOneClassCompositionRecipe(AbstractRecipe):
    detector_recipe: AbstractRecipe
    predictor_recipe: AbstractRecipe
    threshold: float = eqx.field(static=True)

    def __init__(
        self,
        detector_recipe: AbstractRecipe,
        predictor_recipe: AbstractRecipe,
        /,
        *,
        threshold: float = 0.0,
    ):
        if not isinstance(detector_recipe, AbstractRecipe) or not isinstance(
            predictor_recipe, AbstractRecipe
        ):
            raise TypeError("detector_recipe and predictor_recipe must be recipes.")
        self.detector_recipe = detector_recipe
        self.predictor_recipe = predictor_recipe
        self.threshold = float(threshold)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError(
                "HardOneClassCompositionRecipe requires an explicit JAX key."
            )
        base_weight = _validated_sample_weight(batch)
        detector_result = self.detector_recipe.fit_batch(
            _unsupervised_batch(batch), key=jr.fold_in(key, 1)
        )
        score = _score_vector(
            jnp.asarray(
                detector_result.model(batch.dense_features(), key=jr.fold_in(key, 2))
            ),
            batch.sample_mask.shape,
        )
        accepted = jax.lax.stop_gradient(score >= self.threshold)
        predictor_batch = _replace_targets(
            batch,
            batch.require_targets(),
            target_mask=batch.target_mask,
            sample_weight=base_weight * accepted,
        )
        predictor_result = self.predictor_recipe.fit_batch(
            predictor_batch, key=jr.fold_in(key, 3)
        )
        valid = detector_result.valid & predictor_result.valid
        status = jnp.where(
            valid,
            ML_SUCCESS,
            jnp.maximum(detector_result.status, predictor_result.status),
        )
        diagnostics = SelfTrainingDiagnostics(
            accepted,
            jnp.sum(batch.sample_mask, axis=-1),
            jnp.stack((detector_result.status, predictor_result.status)),
            valid=valid,
            status=status,
            iterations=1,
            method="hard-one-class-composition",
        )
        return FitResult(
            HardOneClassCompositionModel(
                detector_result.as_trainable(),
                predictor_result.as_trainable(),
                threshold=self.threshold,
            ),
            diagnostics,
            valid=valid,
            status=status,
            method="hard-one-class-composition",
            gradient_contract=GradientContract(
                prediction_inputs="none",
                prediction_parameters="conditional",
                fit_mode="stopped",
                nondifferentiable_outputs=("one_class_acceptance",),
            ),
        )


__all__ = [
    "GraphFitDiagnostics",
    "HardLabelPropagationModel",
    "HardLabelPropagationRecipe",
    "HardOneClassCompositionModel",
    "HardOneClassCompositionRecipe",
    "HardSelfTrainingModel",
    "HardSelfTrainingRecipe",
    "LabelPropagationModel",
    "LabelPropagationRecipe",
    "LabelSpreadingRecipe",
    "SelfTrainingDiagnostics",
    "SoftOneClassCompositionModel",
    "SoftOneClassCompositionRecipe",
    "SoftSelfTrainingModel",
    "SoftSelfTrainingRecipe",
]
