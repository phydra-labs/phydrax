#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._hard import TreeFitDiagnostics
from ._representation import (
    _tree_output_shape,
    TreeEnsemble,
    TreeStructureDiagnostics,
)
from ._soft import SoftDecisionTree, SoftGradientBoostedTrees, SoftRandomForest


SoftTreeModel = SoftDecisionTree | SoftRandomForest | SoftGradientBoostedTrees


class FeatureImportance(StrictModule):
    """Per-feature gain, cover, and split-frequency inspection values."""

    gain: Array
    cover: Array
    frequency: Array
    normalized_gain: Array
    normalized_cover: Array
    normalized_frequency: Array

    def __init__(
        self,
        *,
        gain: Any,
        cover: Any,
        frequency: Any,
    ):
        gain_ = jnp.asarray(gain)
        cover_ = jnp.asarray(cover)
        frequency_ = jnp.asarray(frequency)
        self.gain = gain_
        self.cover = cover_
        self.frequency = frequency_
        gain_sum = jnp.sum(gain_, axis=-1, keepdims=True)
        cover_sum = jnp.sum(cover_, axis=-1, keepdims=True)
        frequency_sum = jnp.sum(frequency_, axis=-1, keepdims=True)
        self.normalized_gain = gain_ / jnp.where(gain_sum > 0.0, gain_sum, 1.0)
        self.normalized_cover = cover_ / jnp.where(cover_sum > 0.0, cover_sum, 1.0)
        self.normalized_frequency = frequency_ / jnp.where(
            frequency_sum > 0.0, frequency_sum, 1.0
        )


class TreeConvergenceDiagnostics(StrictModule):
    """Portable convergence view extracted from a tree fit diagnostic."""

    valid: Array
    status: Array
    objective: Array
    iterations: Array
    converged: Array
    capacity_exhausted: Array

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any,
        iterations: Any,
        converged: Any,
        capacity_exhausted: Any,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.capacity_exhausted = jnp.asarray(capacity_exhausted, dtype=bool)


class PartialDependenceResult(StrictModule):
    """Feature grid and weighted marginal predictions."""

    grid: Array
    average: Array
    individual: Array | None
    feature_index: int

    def __init__(
        self,
        *,
        grid: Any,
        average: Any,
        individual: Any | None,
        feature_index: int,
    ):
        self.grid = jnp.asarray(grid)
        self.average = jnp.asarray(average)
        self.individual = None if individual is None else jnp.asarray(individual)
        self.feature_index = int(feature_index)


class TreeSHAPExplanation(StrictModule):
    """Exact interventional Shapley decomposition for a bounded feature set."""

    values: Array
    base_values: Array
    predictions: Array

    def __init__(self, *, values: Any, base_values: Any, predictions: Any):
        self.values = jnp.asarray(values)
        self.base_values = jnp.asarray(base_values)
        self.predictions = jnp.asarray(predictions)


class GradientAttribution(StrictModule):
    """Soft-tree input gradients and optional gradient-times-displacement values."""

    gradients: Array
    attributions: Array
    baseline: Array | None

    def __init__(self, *, gradients: Any, attributions: Any, baseline: Any | None):
        self.gradients = jnp.asarray(gradients)
        self.attributions = jnp.asarray(attributions)
        self.baseline = None if baseline is None else jnp.asarray(baseline)


class TreeExport(StrictModule):
    """One selected tree's immutable arrays for external inspection."""

    feature_index: Array
    threshold: Array
    left_child: Array
    right_child: Array
    default_left: Array
    split_kind: Array
    category_values: Array
    category_mask: Array
    leaf_value: Array
    node_mask: Array
    leaf_mask: Array
    gain: Array
    cover: Array

    def __init__(self, model: TreeEnsemble, tree_index: int, /, *, case_index=()):
        tree = int(tree_index)
        if tree < 0 or tree >= model.tree_capacity:
            raise IndexError("tree_index is outside the fixed tree capacity.")
        case = tuple(case_index) if isinstance(case_index, tuple) else (int(case_index),)
        if model.case_shape:
            if len(case) != len(model.case_shape):
                raise IndexError("case_index must identify every case axis.")
            if any(
                index < 0 or index >= size
                for index, size in zip(case, model.case_shape, strict=True)
            ):
                raise IndexError("case_index is out of range.")
            index = case + (tree,)
        else:
            if case not in {(), (0,)}:
                raise IndexError("A case index is invalid for a case-independent model.")
            index = (tree,)
        self.feature_index = model.feature_index[index]
        self.threshold = model.threshold[index]
        self.left_child = model.left_child[index]
        self.right_child = model.right_child[index]
        self.default_left = model.default_left[index]
        self.split_kind = model.split_kind[index]
        self.category_values = model.category_values[index]
        self.category_mask = model.category_mask[index]
        self.leaf_value = model.leaf_value[index]
        self.node_mask = model.node_mask[index]
        self.leaf_mask = model.leaf_mask[index]
        self.gain = model.node_gain[index]
        self.cover = model.node_cover[index]


def feature_importance(model: TreeEnsemble, /) -> FeatureImportance:
    """Aggregate split gain, parent cover, and frequency onto the feature axis."""
    if not isinstance(model, TreeEnsemble):
        raise TypeError("feature_importance requires a hard TreeEnsemble.")
    split = model.node_mask & ~model.leaf_mask & model.tree_mask[..., :, None]
    safe_feature = jnp.clip(model.feature_index, 0, model.in_size - 1)
    assignment = jax.nn.one_hot(safe_feature, model.in_size) * split[..., None]
    gain = jnp.sum(assignment * model.node_gain[..., None], axis=(-3, -2))
    cover = jnp.sum(assignment * model.node_cover[..., None], axis=(-3, -2))
    frequency = jnp.sum(assignment, axis=(-3, -2))
    return FeatureImportance(gain=gain, cover=cover, frequency=frequency)


def capacity_diagnostics(model: TreeEnsemble, /) -> TreeStructureDiagnostics:
    """Return fixed-capacity utilization and represented-structure validity."""
    if not isinstance(model, TreeEnsemble):
        raise TypeError("capacity_diagnostics requires a hard TreeEnsemble.")
    return model.structure_diagnostics()


def export_tree(
    model: TreeEnsemble,
    tree_index: int,
    /,
    *,
    case_index: int | tuple[int, ...] = (),
) -> TreeExport:
    """Export one tree without converting its JAX arrays to host containers."""
    return TreeExport(model, tree_index, case_index=case_index)


def partial_dependence(
    model: TreeEnsemble | SoftTreeModel,
    x: ArrayLike,
    feature_index: int,
    grid: ArrayLike,
    /,
    *,
    sample_weight: ArrayLike | None = None,
    return_individual: bool = False,
) -> PartialDependenceResult:
    """Evaluate weighted one-way partial dependence while preserving case/output axes."""
    values = jnp.asarray(x)
    feature = int(feature_index)
    if values.ndim < 2 or values.shape[-1] != model.in_size:
        raise ValueError("x must end in (sample, feature) axes for partial dependence.")
    if feature < 0 or feature >= model.in_size:
        raise IndexError("feature_index is out of range.")
    if (
        model.case_shape
        and tuple(values.shape[: len(model.case_shape)]) != model.case_shape
    ):
        raise ValueError("x does not begin with the model case_shape.")
    sample_axis = len(model.case_shape)
    if values.ndim != sample_axis + 2:
        raise ValueError("partial_dependence accepts exactly one sample axis.")
    grid_ = jnp.asarray(grid, dtype=values.dtype)
    if grid_.ndim != 1 or grid_.size == 0:
        raise ValueError("grid must be a non-empty one-dimensional array.")
    sample_shape = model.case_shape + (values.shape[-2],)
    weights = (
        jnp.ones(sample_shape, dtype=values.real.dtype)
        if sample_weight is None
        else jnp.broadcast_to(
            jnp.asarray(sample_weight, dtype=values.real.dtype), sample_shape
        )
    )
    if bool(jax.device_get(jnp.any(~jnp.isfinite(weights) | (weights < 0.0)))):
        raise ValueError("sample_weight must be finite and nonnegative.")
    weight_sum = jnp.sum(weights, axis=sample_axis)
    if bool(jax.device_get(jnp.any(weight_sum <= 0.0))):
        raise ValueError("sample_weight must have positive mass in every case.")
    denominator = jnp.where(weight_sum > 0.0, weight_sum, 1.0)

    def one_grid(value):
        prediction = model(values.at[..., feature].set(value))
        output_ndim = prediction.ndim - weights.ndim
        weighted = prediction * weights.reshape(weights.shape + (1,) * output_ndim)
        average = jnp.sum(weighted, axis=sample_axis) / denominator.reshape(
            denominator.shape + (1,) * output_ndim
        )
        return average, prediction

    average, individual = jax.vmap(one_grid)(grid_)
    if model.case_shape:
        # vmap introduces the grid before case axes; expose case axes first.
        average = jnp.moveaxis(average, 0, len(model.case_shape))
        individual = jnp.moveaxis(individual, 0, len(model.case_shape))
    return PartialDependenceResult(
        grid=grid_,
        average=average,
        individual=individual if return_individual else None,
        feature_index=feature,
    )


def tree_shap(
    model: TreeEnsemble,
    x: ArrayLike,
    baseline: ArrayLike,
    /,
    *,
    max_features: int = 12,
) -> TreeSHAPExplanation:
    """Compute exact interventional TreeSHAP by bounded coalition enumeration.

    This exact implementation supports case-independent fixed structures with at most
    ``max_features`` features. The explicit bound prevents accidental exponential
    compilation. Categorical and missing paths remain exact because coalitions are
    evaluated by the represented tree traversal itself.
    """
    if not isinstance(model, TreeEnsemble):
        raise TypeError("tree_shap requires a hard TreeEnsemble.")
    if model.case_shape:
        raise ValueError("tree_shap currently supports case-independent structures only.")
    if model.in_size > max_features:
        raise ValueError(
            f"Exact TreeSHAP is bounded to {max_features} features; got {model.in_size}."
        )
    values = jnp.asarray(x)
    if values.shape[-1:] != (model.in_size,):
        raise ValueError("x has the wrong feature axis.")
    baseline_ = jnp.broadcast_to(jnp.asarray(baseline, dtype=values.dtype), values.shape)
    flat_x = values.reshape((-1, model.in_size))
    flat_baseline = baseline_.reshape(flat_x.shape)
    feature_count = model.in_size
    subset_count = 1 << feature_count
    coalition_values = []
    for subset in range(subset_count):
        included = jnp.asarray(
            [(subset >> feature) & 1 for feature in range(feature_count)], dtype=bool
        )
        coalition = jnp.where(included, flat_x, flat_baseline)
        prediction = model(coalition)
        if model.out_size == "scalar":
            prediction = prediction[..., None]
        else:
            prediction = prediction.reshape((flat_x.shape[0], -1))
        coalition_values.append(prediction)
    coalition_values_ = jnp.stack(coalition_values)
    shap_values = []
    factorial = math.factorial
    normalization = factorial(feature_count)
    for feature in range(feature_count):
        contribution = jnp.zeros_like(coalition_values_[0])
        bit = 1 << feature
        for subset in range(subset_count):
            if subset & bit:
                continue
            size = subset.bit_count()
            coefficient = (
                factorial(size) * factorial(feature_count - size - 1) / normalization
            )
            contribution = contribution + coefficient * (
                coalition_values_[subset | bit] - coalition_values_[subset]
            )
        shap_values.append(contribution)
    shap = jnp.stack(shap_values, axis=1)
    output_shape = (
        ()
        if model.out_size == "scalar"
        else tuple(
            model.out_size if isinstance(model.out_size, tuple) else (model.out_size,)
        )
    )
    lead_shape = values.shape[:-1]
    if model.out_size == "scalar":
        shap = shap[..., 0].reshape(lead_shape + (feature_count,))
        base_value = coalition_values_[0, ..., 0].reshape(lead_shape)
        prediction = coalition_values_[-1, ..., 0].reshape(lead_shape)
    else:
        shap = shap.reshape(lead_shape + (feature_count,) + output_shape)
        base_value = coalition_values_[0].reshape(lead_shape + output_shape)
        prediction = coalition_values_[-1].reshape(lead_shape + output_shape)
    return TreeSHAPExplanation(
        values=shap, base_values=base_value, predictions=prediction
    )


def soft_tree_gradient_attribution(
    model: SoftTreeModel,
    x: ArrayLike,
    /,
    *,
    baseline: ArrayLike | None = None,
) -> GradientAttribution:
    """Return exact input Jacobians for soft trees and gradient-times-displacement."""
    if not isinstance(
        model, (SoftDecisionTree, SoftRandomForest, SoftGradientBoostedTrees)
    ):
        raise TypeError("soft_tree_gradient_attribution requires a soft-tree model.")
    if model.case_shape:
        raise ValueError(
            "soft_tree_gradient_attribution currently supports case-independent models only."
        )
    values = jnp.asarray(x)
    if values.shape[-1:] != (model.in_size,):
        raise ValueError("x has the wrong feature axis.")
    flat = values.reshape((-1, model.in_size))
    if jnp.issubdtype(model.leaf_value.dtype, jnp.complexfloating):
        real_jacobian = jax.vmap(jax.jacrev(lambda point: jnp.real(model(point))))(flat)
        imaginary_jacobian = jax.vmap(jax.jacrev(lambda point: jnp.imag(model(point))))(
            flat
        )
        gradients = real_jacobian + 1j * imaginary_jacobian
    else:
        gradients = jax.vmap(jax.jacrev(model))(flat)
    output_shape = _tree_output_shape(model.out_size)
    if model.out_size == "scalar":
        gradients = gradients.reshape(values.shape)
    else:
        gradients = gradients.reshape(values.shape[:-1] + output_shape + (model.in_size,))
    if baseline is None:
        attribution = gradients
        baseline_ = None
    else:
        baseline_ = jnp.broadcast_to(
            jnp.asarray(baseline, dtype=values.dtype), values.shape
        )
        displacement = values - baseline_
        if model.out_size == "scalar":
            attribution = gradients * displacement
        else:
            attribution = gradients * displacement.reshape(
                values.shape[:-1] + (1,) * len(output_shape) + (model.in_size,)
            )
    return GradientAttribution(
        gradients=gradients, attributions=attribution, baseline=baseline_
    )


def convergence_diagnostics(diagnostics: Any, /) -> TreeConvergenceDiagnostics:
    """Extract convergence/capacity fields from hard or soft tree fit diagnostics."""
    if not isinstance(diagnostics, TreeFitDiagnostics):
        raise TypeError("diagnostics must be TreeFitDiagnostics.")
    return TreeConvergenceDiagnostics(
        valid=diagnostics.valid,
        status=diagnostics.status,
        objective=diagnostics.objective,
        iterations=diagnostics.iterations,
        converged=diagnostics.converged,
        capacity_exhausted=diagnostics.capacity_exhausted,
    )


__all__ = [
    "FeatureImportance",
    "GradientAttribution",
    "PartialDependenceResult",
    "TreeExport",
    "TreeSHAPExplanation",
    "TreeConvergenceDiagnostics",
    "capacity_diagnostics",
    "convergence_diagnostics",
    "export_tree",
    "feature_importance",
    "partial_dependence",
    "soft_tree_gradient_attribution",
    "tree_shap",
]
