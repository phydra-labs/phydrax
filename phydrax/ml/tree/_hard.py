#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from .._batch import MLBatch
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_CAPACITY_EXHAUSTED,
    ML_INSUFFICIENT_DATA,
    ML_SUCCESS,
)
from .._schema import TargetSchema
from ._representation import _traverse_one_tree, _weighted_median_case, TreeEnsemble


SplitSearch: TypeAlias = Literal["exact", "histogram", "random"]
XGBObjective: TypeAlias = Literal[
    "auto",
    "squared_error",
    "logistic",
    "softmax",
    "poisson",
    "pairwise_ranking",
]

_HARD_CONTRACT = GradientContract(
    prediction_inputs="none",
    prediction_parameters="almost-everywhere",
    fit_features="none",
    fit_targets="none",
    fit_weights="none",
    fit_hyperparameters="none",
    fit_mode="stopped",
    nondifferentiable_outputs=(
        "split structure",
        "leaf indices",
        "decision paths",
        "class labels",
    ),
    conditions=(
        "Finite values away from represented split thresholds are locally constant.",
    ),
)


def _as_bool(value: Any) -> bool:
    return bool(jax.device_get(jnp.asarray(value)))


def _as_float(value: Any) -> float:
    return float(jax.device_get(jnp.asarray(value)))


def _weight_sum(weight: Array, /) -> Array:
    return jnp.maximum(jnp.sum(weight), jnp.finfo(weight.dtype).tiny)


class TreeFitDiagnostics(StrictModule):
    """Immutable fit audit for fixed-capacity hard tree recipes."""

    valid: Array
    status: Array
    objective: Array
    iterations: Array
    effective_samples: Array
    trees_built: Array
    nodes_used: Array
    leaves_used: Array
    capacity_exhausted: Array
    converged: Array
    method: str = eqx.field(static=True)
    split_search: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any,
        iterations: Any,
        effective_samples: Any,
        trees_built: Any,
        nodes_used: Any,
        leaves_used: Any,
        capacity_exhausted: Any,
        converged: Any,
        method: str,
        split_search: str,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.trees_built = jnp.asarray(trees_built, dtype=jnp.int32)
        self.nodes_used = jnp.asarray(nodes_used, dtype=jnp.int32)
        self.leaves_used = jnp.asarray(leaves_used, dtype=jnp.int32)
        self.capacity_exhausted = jnp.asarray(capacity_exhausted, dtype=bool)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.method = str(method)
        self.split_search = str(split_search)


def _validate_common(
    *,
    max_depth: int,
    max_nodes: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    min_weight_leaf: float,
    min_gain: float,
    max_features: int | float | Literal["sqrt", "log2"] | None,
    max_bins: int,
    max_categories: int,
    monotonic_constraints: tuple[int, ...],
    interaction_constraints: tuple[tuple[int, ...], ...],
) -> tuple[int, int]:
    depth = int(max_depth)
    if depth < 0:
        raise ValueError("max_depth must be nonnegative.")
    full_capacity = 2 ** (depth + 1) - 1
    capacity = full_capacity if max_nodes is None else int(max_nodes)
    if capacity <= 0:
        raise ValueError("max_nodes must be positive.")
    if min_samples_split < 2 or min_samples_leaf < 1:
        raise ValueError("min_samples_split >= 2 and min_samples_leaf >= 1 are required.")
    if min_weight_leaf < 0.0 or min_gain < 0.0:
        raise ValueError("Minimum weights and gains must be nonnegative.")
    if isinstance(max_features, float) and not (0.0 < max_features <= 1.0):
        raise ValueError("Floating max_features must lie in (0, 1].")
    if isinstance(max_features, int) and max_features <= 0:
        raise ValueError("Integer max_features must be positive.")
    if max_features not in {None, "sqrt", "log2"} and not isinstance(
        max_features, (int, float)
    ):
        raise ValueError("Unsupported max_features specification.")
    if max_bins < 2 or max_categories < 1:
        raise ValueError("max_bins >= 2 and max_categories >= 1 are required.")
    if any(value not in {-1, 0, 1} for value in monotonic_constraints):
        raise ValueError("Monotonic constraints must be -1, 0, or 1.")
    if any(
        len(group) == 0 or len(set(group)) != len(group)
        for group in interaction_constraints
    ):
        raise ValueError("Interaction constraint groups must be non-empty sets.")
    return depth, capacity


def _feature_count(specification, feature_count: int) -> int:
    if specification is None:
        return feature_count
    if specification == "sqrt":
        return max(1, int(math.sqrt(feature_count)))
    if specification == "log2":
        return max(1, int(math.log2(feature_count)))
    if isinstance(specification, float):
        return max(1, min(feature_count, int(math.ceil(specification * feature_count))))
    return min(feature_count, int(specification))


def _empty_tree(
    node_capacity: int,
    output_count: int,
    category_capacity: int,
    *,
    value_dtype,
    threshold_dtype,
) -> dict[str, Array | bool]:
    return {
        "feature_index": jnp.full((node_capacity,), -1, dtype=jnp.int32),
        "threshold": jnp.zeros((node_capacity,), dtype=threshold_dtype),
        "left_child": jnp.full((node_capacity,), -1, dtype=jnp.int32),
        "right_child": jnp.full((node_capacity,), -1, dtype=jnp.int32),
        "default_left": jnp.zeros((node_capacity,), dtype=bool),
        "split_kind": jnp.zeros((node_capacity,), dtype=jnp.int8),
        "category_values": jnp.zeros(
            (node_capacity, category_capacity), dtype=threshold_dtype
        ),
        "category_mask": jnp.zeros((node_capacity, category_capacity), dtype=bool),
        "leaf_value": jnp.zeros((node_capacity, output_count), dtype=value_dtype),
        "node_mask": jnp.zeros((node_capacity,), dtype=bool),
        "leaf_mask": jnp.zeros((node_capacity,), dtype=bool),
        "node_gain": jnp.zeros((node_capacity,), dtype=jnp.asarray(0.0).dtype),
        "node_cover": jnp.zeros((node_capacity,), dtype=jnp.asarray(0.0).dtype),
        "capacity_exhausted": False,
    }


def _cart_stats(
    y: Array, weight: Array, mask: Array
) -> tuple[Array, Array, Array, Array]:
    selected_weight = jnp.where(mask, weight, 0.0)
    cover = jnp.sum(selected_weight)
    safe_cover = jnp.maximum(cover, jnp.finfo(selected_weight.dtype).tiny)
    value = jnp.sum(selected_weight[:, None] * y, axis=0) / safe_cover
    residual = y - value
    loss = jnp.sum(selected_weight[:, None] * jnp.real(residual * jnp.conj(residual)))
    count = jnp.sum(mask & (weight > 0.0))
    return value, loss, cover, count


def _newton_stats(
    gradient: Array,
    hessian: Array,
    mask: Array,
    *,
    l2_regularization: float,
    l1_regularization: float,
    max_delta_step: float | None,
) -> tuple[Array, Array, Array, Array]:
    selected = mask[:, None]
    grad = jnp.sum(jnp.where(selected, gradient, 0.0), axis=0)
    hess = jnp.sum(jnp.where(selected, hessian, 0.0), axis=0)
    magnitude = jnp.abs(grad)
    direction = jnp.where(magnitude > 0.0, grad / jnp.maximum(magnitude, 1e-30), 0.0)
    soft_grad = direction * jnp.maximum(magnitude - l1_regularization, 0.0)
    denominator = hess + l2_regularization
    safe = jnp.maximum(denominator, jnp.finfo(hessian.dtype).tiny)
    value = -soft_grad / safe
    if max_delta_step is not None:
        value_magnitude = jnp.abs(value)
        value = value * jnp.minimum(
            1.0, max_delta_step / jnp.maximum(value_magnitude, 1e-30)
        )
    score = 0.5 * jnp.sum(jnp.real(soft_grad * jnp.conj(soft_grad)) / safe)
    cover = jnp.sum(hess)
    count = jnp.sum(mask & (jnp.sum(hessian, axis=-1) > 0.0))
    return value, score, cover, count


def _interaction_allowed(
    feature: int,
    used_features: tuple[int, ...],
    constraints: tuple[tuple[int, ...], ...],
) -> bool:
    if not constraints:
        return True
    requested = set(used_features) | {feature}
    return any(requested.issubset(set(group)) for group in constraints)


def _candidate_indices(
    sample_count: int, split_search: SplitSearch, max_bins: int
) -> tuple[int, ...]:
    if sample_count <= 1:
        return ()
    if split_search != "histogram" or sample_count - 1 <= max_bins:
        return tuple(range(sample_count - 1))
    return tuple(
        sorted(
            {
                min(sample_count - 2, max(0, (bin_index * sample_count) // max_bins - 1))
                for bin_index in range(1, max_bins + 1)
            }
        )
    )


def _build_tree(
    x: Array,
    weight: Array,
    root_mask: Array,
    *,
    node_capacity: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    min_weight_leaf: float,
    min_gain: float,
    max_leaf_nodes: int | None,
    max_features: int | float | Literal["sqrt", "log2"] | None,
    split_search: SplitSearch,
    max_bins: int,
    feature_kinds: tuple[str, ...],
    max_categories: int,
    monotonic_constraints: tuple[int, ...],
    interaction_constraints: tuple[tuple[int, ...], ...],
    key: Any,
    y: Array | None = None,
    gradient: Array | None = None,
    hessian: Array | None = None,
    l2_regularization: float = 0.0,
    l1_regularization: float = 0.0,
    gamma: float = 0.0,
    max_delta_step: float | None = None,
) -> dict[str, Array | bool]:
    newton = gradient is not None
    if newton == (hessian is None) or newton == (y is not None):
        raise ValueError("Supply exactly y or both gradient and hessian.")
    target = gradient if newton else y
    assert target is not None
    tree = _empty_tree(
        node_capacity,
        int(target.shape[-1]),
        max_categories,
        value_dtype=target.dtype,
        threshold_dtype=x.dtype,
    )
    feature_count = int(x.shape[-1])
    selected_feature_count = _feature_count(max_features, feature_count)
    if monotonic_constraints and len(monotonic_constraints) != feature_count:
        raise ValueError("monotonic_constraints must align with the feature axis.")
    monotonic_enabled = any(value != 0 for value in monotonic_constraints)
    if monotonic_enabled and target.shape[-1] != 1:
        raise ValueError("Monotonic constraints require a scalar tree output.")
    if monotonic_enabled and jnp.issubdtype(target.dtype, jnp.complexfloating):
        raise TypeError("Monotonic ordering is undefined for complex tree outputs.")
    if monotonic_enabled and any(
        constraint != 0 and kind in {"categorical", "boolean"}
        for constraint, kind in zip(monotonic_constraints, feature_kinds, strict=True)
    ):
        raise ValueError("Monotonic constraints require ordered feature semantics.")
    if any(
        index < 0 or index >= feature_count
        for group in interaction_constraints
        for index in group
    ):
        raise ValueError("interaction_constraints contain an out-of-range feature.")

    value_dtype = target.real.dtype
    queue: list[tuple[int, Array, int, tuple[int, ...], Array, Array]] = [
        (
            0,
            root_mask,
            0,
            (),
            jnp.asarray(-jnp.inf, dtype=value_dtype),
            jnp.asarray(jnp.inf, dtype=value_dtype),
        )
    ]
    next_node = 1
    leaf_count = 1
    node_key = key
    while queue:
        node, node_samples, depth, used_features, lower_bound, upper_bound = queue.pop(0)
        if newton:
            assert gradient is not None and hessian is not None
            parent_value, parent_metric, parent_cover, parent_count = _newton_stats(
                gradient,
                hessian,
                node_samples,
                l2_regularization=l2_regularization,
                l1_regularization=l1_regularization,
                max_delta_step=max_delta_step,
            )
        else:
            assert y is not None
            parent_value, parent_metric, parent_cover, parent_count = _cart_stats(
                y, weight, node_samples
            )
        if monotonic_enabled:
            parent_value = jnp.clip(parent_value, lower_bound, upper_bound)
        tree["node_mask"] = tree["node_mask"].at[node].set(True)  # type: ignore[union-attr]
        tree["leaf_mask"] = tree["leaf_mask"].at[node].set(True)  # type: ignore[union-attr]
        tree["leaf_value"] = (
            tree["leaf_value"]
            .at[node]
            .set(  # type: ignore[union-attr]
                parent_value.astype(tree["leaf_value"].dtype)  # type: ignore[index]
            )
        )
        tree["node_cover"] = tree["node_cover"].at[node].set(parent_cover)  # type: ignore[union-attr]
        if (
            depth >= max_depth
            or _as_float(parent_count) < min_samples_split
            or _as_float(parent_cover) <= 0.0
            or (max_leaf_nodes is not None and leaf_count >= max_leaf_nodes)
        ):
            continue

        feature_ids = jnp.arange(feature_count, dtype=jnp.int32)
        if selected_feature_count < feature_count:
            if node_key is None:
                raise ValueError(
                    "Random feature subsampling requires an explicit JAX key."
                )
            node_key, feature_key = jax.random.split(node_key)
            feature_ids = jax.random.permutation(feature_key, feature_ids)[
                :selected_feature_count
            ]

        best_gain = jnp.asarray(-jnp.inf, dtype=jnp.asarray(parent_metric).real.dtype)
        best_feature = -1
        best_threshold = jnp.asarray(0.0, dtype=x.dtype)
        best_default_left = False
        best_kind = 0
        best_categories = jnp.zeros((max_categories,), dtype=x.dtype)
        best_category_mask = jnp.zeros((max_categories,), dtype=bool)
        best_left_mask = node_samples
        best_right_mask = node_samples
        best_left_value = parent_value
        best_right_value = parent_value

        for feature_position in range(selected_feature_count):
            feature = int(jax.device_get(feature_ids[feature_position]))
            if not _interaction_allowed(feature, used_features, interaction_constraints):
                continue
            values = x[:, feature]
            present = node_samples & jnp.isfinite(values)
            present_count = int(jax.device_get(jnp.sum(present)))
            categorical = feature_kinds[feature] in {"categorical", "boolean"}
            candidates: list[tuple[Array, int, Array, Array]] = []
            if categorical:
                sorted_values = jnp.sort(jnp.where(present, values, jnp.inf))
                for candidate_index in range(present_count):
                    category = sorted_values[candidate_index]
                    category_valid = jnp.isfinite(category)
                    categories = (
                        jnp.zeros((max_categories,), dtype=x.dtype)
                        .at[0]
                        .set(jnp.where(category_valid, category, 0.0))
                    )
                    category_mask = (
                        jnp.zeros((max_categories,), dtype=bool).at[0].set(category_valid)
                    )
                    candidates.append((category, 1, categories, category_mask))
            elif split_search == "random":
                if node_key is None:
                    raise ValueError("Random thresholds require an explicit JAX key.")
                node_key, threshold_key = jax.random.split(node_key)
                minimum = jnp.min(jnp.where(present, values, jnp.inf))
                maximum = jnp.max(jnp.where(present, values, -jnp.inf))
                threshold = jax.random.uniform(
                    threshold_key,
                    (),
                    dtype=x.dtype,
                    minval=jnp.where(jnp.isfinite(minimum), minimum, 0.0),
                    maxval=jnp.where(maximum > minimum, maximum, minimum + 1.0),
                )
                candidates.append(
                    (
                        threshold,
                        0,
                        jnp.zeros((max_categories,), dtype=x.dtype),
                        jnp.zeros((max_categories,), dtype=bool),
                    )
                )
            else:
                sorted_values = jnp.sort(jnp.where(present, values, jnp.inf))
                for candidate_index in _candidate_indices(
                    present_count, split_search, max_bins
                ):
                    lower = sorted_values[candidate_index]
                    upper = sorted_values[candidate_index + 1]
                    threshold = lower + (upper - lower) * 0.5
                    candidates.append(
                        (
                            threshold,
                            0,
                            jnp.zeros((max_categories,), dtype=x.dtype),
                            jnp.zeros((max_categories,), dtype=bool),
                        )
                    )

            for candidate, kind, categories, category_mask in candidates:
                if kind == 1:
                    ordinary_left = present & jnp.any(
                        category_mask[None, :] & (values[:, None] == categories[None, :]),
                        axis=-1,
                    )
                    candidate_valid = jnp.any(category_mask)
                else:
                    ordinary_left = present & (values <= candidate)
                    candidate_valid = jnp.isfinite(candidate)
                missing = node_samples & ~present
                for default_left in (False, True):
                    left_mask = node_samples & (ordinary_left | (missing & default_left))
                    right_mask = node_samples & ~(
                        ordinary_left | (missing & default_left)
                    )
                    if newton:
                        left_value, left_metric, left_cover, left_count = _newton_stats(
                            gradient,
                            hessian,
                            left_mask,
                            l2_regularization=l2_regularization,
                            l1_regularization=l1_regularization,
                            max_delta_step=max_delta_step,
                        )
                        right_value, right_metric, right_cover, right_count = (
                            _newton_stats(
                                gradient,
                                hessian,
                                right_mask,
                                l2_regularization=l2_regularization,
                                l1_regularization=l1_regularization,
                                max_delta_step=max_delta_step,
                            )
                        )
                        gain = left_metric + right_metric - parent_metric - gamma
                    else:
                        left_value, left_metric, left_cover, left_count = _cart_stats(
                            y, weight, left_mask
                        )
                        right_value, right_metric, right_cover, right_count = _cart_stats(
                            y, weight, right_mask
                        )
                        gain = parent_metric - left_metric - right_metric - gamma
                    if monotonic_enabled:
                        left_value = jnp.clip(left_value, lower_bound, upper_bound)
                        right_value = jnp.clip(right_value, lower_bound, upper_bound)
                    valid = (
                        candidate_valid
                        & (left_count >= min_samples_leaf)
                        & (right_count >= min_samples_leaf)
                        & (left_cover >= min_weight_leaf)
                        & (right_cover >= min_weight_leaf)
                        & jnp.isfinite(gain)
                    )
                    monotonic = (
                        0 if not monotonic_constraints else monotonic_constraints[feature]
                    )
                    if monotonic != 0:
                        ordering = left_value[0] <= right_value[0]
                        valid = valid & (ordering if monotonic > 0 else ~ordering)
                    improve = valid & (gain > best_gain)
                    if _as_bool(improve):
                        best_gain = gain
                        best_feature = feature
                        best_threshold = candidate
                        best_default_left = default_left
                        best_kind = kind
                        best_categories = categories
                        best_category_mask = category_mask
                        best_left_mask = left_mask
                        best_right_mask = right_mask
                        best_left_value = left_value
                        best_right_value = right_value

        if best_feature < 0 or _as_float(best_gain) <= min_gain:
            continue
        if next_node + 1 >= node_capacity:
            tree["capacity_exhausted"] = True
            continue
        left_node, right_node = next_node, next_node + 1
        next_node += 2
        leaf_count += 1
        tree["leaf_mask"] = tree["leaf_mask"].at[node].set(False)  # type: ignore[union-attr]
        tree["feature_index"] = tree["feature_index"].at[node].set(best_feature)  # type: ignore[union-attr]
        tree["threshold"] = tree["threshold"].at[node].set(best_threshold)  # type: ignore[union-attr]
        tree["left_child"] = tree["left_child"].at[node].set(left_node)  # type: ignore[union-attr]
        tree["right_child"] = tree["right_child"].at[node].set(right_node)  # type: ignore[union-attr]
        tree["default_left"] = tree["default_left"].at[node].set(best_default_left)  # type: ignore[union-attr]
        tree["split_kind"] = tree["split_kind"].at[node].set(best_kind)  # type: ignore[union-attr]
        tree["category_values"] = tree["category_values"].at[node].set(best_categories)  # type: ignore[union-attr]
        tree["category_mask"] = tree["category_mask"].at[node].set(best_category_mask)  # type: ignore[union-attr]
        tree["node_gain"] = tree["node_gain"].at[node].set(best_gain)  # type: ignore[union-attr]
        next_used = tuple(sorted(set(used_features) | {best_feature}))
        left_lower, left_upper = lower_bound, upper_bound
        right_lower, right_upper = lower_bound, upper_bound
        direction = (
            0 if not monotonic_constraints else monotonic_constraints[best_feature]
        )
        if direction != 0:
            midpoint = 0.5 * (best_left_value[0] + best_right_value[0])
            if direction > 0:
                left_upper = jnp.minimum(left_upper, midpoint)
                right_lower = jnp.maximum(right_lower, midpoint)
            else:
                left_lower = jnp.maximum(left_lower, midpoint)
                right_upper = jnp.minimum(right_upper, midpoint)
        queue.append(
            (
                left_node,
                best_left_mask,
                depth + 1,
                next_used,
                left_lower,
                left_upper,
            )
        )
        queue.append(
            (
                right_node,
                best_right_mask,
                depth + 1,
                next_used,
                right_lower,
                right_upper,
            )
        )
    return tree


def _tree_predict(tree: dict[str, Array | bool], x: Array) -> Array:
    return jax.vmap(
        lambda point: _traverse_one_tree(
            point,
            tree["feature_index"],
            tree["threshold"],
            tree["left_child"],
            tree["right_child"],
            tree["default_left"],
            tree["split_kind"],
            tree["category_values"],
            tree["category_mask"],
            tree["leaf_value"],
            tree["node_mask"],
            tree["leaf_mask"],
            int(tree["feature_index"].shape[0]),  # type: ignore[union-attr]
        )[0]
    )(x)  # type: ignore[arg-type]


def _prepare_batch(
    batch: MLBatch,
    *,
    classification: bool,
    num_classes: int | None,
) -> tuple[Array, Array, Array, Array, int, tuple[int, ...], TargetSchema]:
    targets = batch.require_targets()
    if batch.target_shape is None:
        raise ValueError("Tree fitting requires targets.")
    if classification and batch.target_shape != ():
        raise ValueError("Classification tree targets must be scalar class indices.")
    features = batch.dense_features(fill_value=jnp.nan)
    if jnp.issubdtype(features.dtype, jnp.complexfloating):
        raise TypeError("Hard tree ordering is undefined for complex features.")
    if not jnp.issubdtype(features.dtype, jnp.inexact):
        features = features.astype(jnp.float32)
    features = jnp.where(batch.feature_mask, features, jnp.nan)
    case_count = math.prod(batch.case_shape) if batch.case_shape else 1
    x = features.reshape((case_count, batch.sample_count, batch.feature_count))
    raw_y = targets.reshape((case_count, batch.sample_count, -1))
    weight = batch.effective_weight("statistical").reshape(
        (case_count, batch.sample_count)
    )
    sample_mask = batch.sample_mask.reshape((case_count, batch.sample_count))
    invalid_weight = sample_mask & (~jnp.isfinite(weight) | (weight < 0.0))
    invalid_weight_predicate = jnp.any(invalid_weight)
    if isinstance(invalid_weight_predicate, jax.core.Tracer):
        weight = eqx.error_if(
            weight,
            invalid_weight_predicate,
            "Tree fitting requires finite nonnegative sample weights.",
        )
    elif bool(invalid_weight_predicate):
        raise ValueError("Tree fitting requires finite nonnegative sample weights.")
    target_mask = (
        jnp.ones_like(raw_y, dtype=bool)
        if batch.target_mask is None
        else batch.target_mask.reshape(raw_y.shape)
    )
    target_valid = jnp.all(target_mask & jnp.isfinite(raw_y), axis=-1)
    sample_mask = sample_mask & target_valid
    weight = jnp.where(sample_mask & jnp.isfinite(weight) & (weight >= 0.0), weight, 0.0)

    if classification:
        labels = raw_y[..., 0]
        integral = labels == jnp.floor(labels)
        sample_mask = sample_mask & integral & (labels >= 0)
        if num_classes is None:
            if batch.target_schema.num_classes:
                num_classes = batch.target_schema.num_classes
            else:
                maximum = jnp.max(jnp.where(sample_mask, labels, -1.0))
                num_classes = int(jax.device_get(maximum)) + 1
        if num_classes < 2:
            raise ValueError("Classification requires at least two classes.")
        sample_mask = sample_mask & (labels < num_classes)
        weight = jnp.where(sample_mask, weight, 0.0)
        y = jax.nn.one_hot(labels.astype(jnp.int32), num_classes, dtype=features.dtype)
        target_schema = (
            batch.target_schema
            if batch.target_schema.num_classes == num_classes
            else TargetSchema(
                "binary" if num_classes == 2 else "multiclass",
                class_labels=tuple(range(num_classes)),
            )
        )
        return x, y, weight, sample_mask, num_classes, (num_classes,), target_schema

    output_count = int(raw_y.shape[-1])
    out_shape = batch.target_shape
    return x, raw_y, weight, sample_mask, output_count, out_shape, batch.target_schema


def _stack_model(
    case_forests: list[list[dict[str, Array | bool]]],
    tree_weights: list[Array],
    base_scores: list[Array],
    *,
    batch: MLBatch,
    tree_capacity: int,
    node_capacity: int,
    category_capacity: int,
    output_count: int,
    out_size,
    target_schema: TargetSchema,
    objective_transform,
    capacity_exhausted: list[bool],
    aggregation="sum",
) -> TreeEnsemble:
    keys = (
        "feature_index",
        "threshold",
        "left_child",
        "right_child",
        "default_left",
        "split_kind",
        "category_values",
        "category_mask",
        "leaf_value",
        "node_mask",
        "leaf_mask",
        "node_gain",
        "node_cover",
    )
    stacked: dict[str, Array] = {}
    exemplar = case_forests[0][0]
    for name in keys:
        padded_cases = []
        for forest in case_forests:
            values = [tree[name] for tree in forest]
            while len(values) < tree_capacity:
                values.append(jnp.zeros_like(exemplar[name]))  # type: ignore[arg-type]
            padded_cases.append(jnp.stack(values[:tree_capacity]))
        value = jnp.stack(padded_cases)
        trailing = value.shape[1:]
        stacked[name] = value.reshape(batch.case_shape + trailing)
    tree_mask = jnp.stack(
        [jnp.arange(tree_capacity) < len(forest) for forest in case_forests]
    ).reshape(batch.case_shape + (tree_capacity,))
    weights = jnp.stack(tree_weights).reshape(batch.case_shape + (tree_capacity,))
    base = jnp.stack(base_scores).reshape(batch.case_shape + (output_count,))
    exhausted = jnp.asarray(capacity_exhausted).reshape(batch.case_shape)
    return TreeEnsemble(
        feature_index=stacked["feature_index"],
        threshold=stacked["threshold"],
        left_child=stacked["left_child"],
        right_child=stacked["right_child"],
        default_left=stacked["default_left"],
        split_kind=stacked["split_kind"],
        category_values=stacked["category_values"],
        category_mask=stacked["category_mask"],
        leaf_value=stacked["leaf_value"],
        node_mask=stacked["node_mask"],
        leaf_mask=stacked["leaf_mask"],
        tree_mask=tree_mask,
        tree_weight=weights,
        node_gain=stacked["node_gain"],
        node_cover=stacked["node_cover"],
        base_score=base,
        feature_schema=batch.feature_schema,
        target_schema=target_schema,
        objective_transform=objective_transform,
        aggregation=aggregation,
        case_shape=batch.case_shape,
        out_size=out_size,
        max_steps=node_capacity,
        capacity_exhausted=exhausted,
    )


def _finish_result(
    model: TreeEnsemble,
    *,
    batch: MLBatch,
    case_forests: list[list[dict[str, Array | bool]]],
    effective_samples: list[Array],
    objectives: list[Array],
    valid: list[bool],
    capacity: list[bool],
    method: str,
    split_search: str,
) -> FitResult:
    trees = [len(forest) for forest in case_forests]
    nodes = [
        sum(int(_as_float(jnp.sum(tree["node_mask"]))) for tree in forest)
        for forest in case_forests
    ]
    leaves = [
        sum(
            int(_as_float(jnp.sum(tree["node_mask"] & tree["leaf_mask"])))
            for tree in forest
        )
        for forest in case_forests
    ]  # type: ignore[operator]
    status = [
        ML_CAPACITY_EXHAUSTED
        if exhausted
        else (ML_SUCCESS if okay else ML_INSUFFICIENT_DATA)
        for okay, exhausted in zip(valid, capacity, strict=True)
    ]
    complete = [
        okay and not exhausted for okay, exhausted in zip(valid, capacity, strict=True)
    ]
    shape = batch.case_shape
    diagnostics = TreeFitDiagnostics(
        valid=jnp.asarray(complete).reshape(shape),
        status=jnp.asarray(status).reshape(shape),
        objective=jnp.asarray(objectives).reshape(shape),
        iterations=jnp.asarray(trees).reshape(shape),
        effective_samples=jnp.asarray(effective_samples).reshape(shape),
        trees_built=jnp.asarray(trees).reshape(shape),
        nodes_used=jnp.asarray(nodes).reshape(shape),
        leaves_used=jnp.asarray(leaves).reshape(shape),
        capacity_exhausted=jnp.asarray(capacity).reshape(shape),
        converged=jnp.asarray(complete).reshape(shape),
        method=method,
        split_search=split_search,
    )
    valid_array = jnp.asarray(complete).reshape(shape)
    status_array = jnp.asarray(status).reshape(shape)
    return FitResult(
        model,
        diagnostics,
        valid=valid_array,
        status=status_array,
        method=method,
        gradient_contract=_HARD_CONTRACT,
    )


def _fit_bagged(
    recipe,
    batch: MLBatch,
    *,
    key: Any,
    classification: bool,
    tree_count: int,
    bootstrap: bool,
    random_thresholds: bool,
    method: str,
) -> FitResult:
    stochastic = (
        bootstrap
        or random_thresholds
        or _feature_count(recipe.max_features, batch.feature_count) < batch.feature_count
    )
    if stochastic and key is None:
        raise ValueError(f"{method} requires an explicit JAX key.")
    x, y, weight, sample_mask, output_count, output_shape, target_schema = _prepare_batch(
        batch,
        classification=classification,
        num_classes=recipe.num_classes if classification else None,
    )
    case_count = int(x.shape[0])
    if key is None:
        all_keys = [[None] * tree_count for _ in range(case_count)]
    else:
        flat_keys = list(jax.random.split(key, case_count * tree_count))
        all_keys = [
            flat_keys[case * tree_count : (case + 1) * tree_count]
            for case in range(case_count)
        ]
    forests: list[list[dict[str, Array | bool]]] = []
    weights_out: list[Array] = []
    bases: list[Array] = []
    effective: list[Array] = []
    objectives: list[Array] = []
    valid: list[bool] = []
    capacity: list[bool] = []
    for case in range(case_count):
        forest = []
        for tree_index in range(tree_count):
            tree_key = all_keys[case][tree_index]
            tree_weight = weight[case]
            if bootstrap:
                tree_key, sample_key = jax.random.split(tree_key)
                probabilities = sample_mask[case].astype(weight.dtype)
                probability_sum = jnp.sum(probabilities)
                probabilities = jnp.where(
                    probability_sum > 0.0,
                    probabilities
                    / jnp.maximum(probability_sum, jnp.finfo(probabilities.dtype).tiny),
                    jnp.full_like(probabilities, 1.0 / batch.sample_count),
                )
                indices = jax.random.choice(
                    sample_key,
                    batch.sample_count,
                    shape=(batch.sample_count,),
                    replace=True,
                    p=probabilities,
                )
                counts = jnp.bincount(indices, length=batch.sample_count)
                tree_weight = weight[case] * counts
            tree = _build_tree(
                x[case],
                tree_weight,
                sample_mask[case] & (tree_weight > 0.0),
                node_capacity=recipe.max_nodes,
                max_depth=recipe.max_depth,
                min_samples_split=recipe.min_samples_split,
                min_samples_leaf=recipe.min_samples_leaf,
                min_weight_leaf=recipe.min_weight_leaf,
                min_gain=recipe.min_gain + recipe.ccp_alpha,
                max_leaf_nodes=recipe.max_leaf_nodes,
                max_features=recipe.max_features,
                split_search="random" if random_thresholds else recipe.split_search,
                max_bins=recipe.max_bins,
                feature_kinds=batch.feature_schema.kinds,
                max_categories=recipe.max_categories,
                monotonic_constraints=recipe.monotonic_constraints,
                interaction_constraints=recipe.interaction_constraints,
                key=tree_key,
                y=y[case],
            )
            forest.append(tree)
        forests.append(forest)
        weights_out.append(jnp.full((tree_count,), 1.0 / tree_count, dtype=y.real.dtype))
        bases.append(jnp.zeros((output_count,), dtype=y.dtype))
        effective.append(jnp.sum(sample_mask[case] & (weight[case] > 0.0)))
        okay = _as_bool(jnp.sum(weight[case]) > 0.0)
        valid.append(okay)
        exhausted = any(bool(tree["capacity_exhausted"]) for tree in forest)
        capacity.append(exhausted)
        prediction = sum(_tree_predict(tree, x[case]) for tree in forest) / tree_count
        objectives.append(
            jnp.sum(weight[case, :, None] * jnp.real((prediction - y[case]) ** 2))
            / _weight_sum(weight[case])
        )
    out_size = (
        output_shape if output_shape else (output_count if classification else "scalar")
    )
    model = _stack_model(
        forests,
        weights_out,
        bases,
        batch=batch,
        tree_capacity=tree_count,
        node_capacity=recipe.max_nodes,
        category_capacity=recipe.max_categories,
        output_count=output_count,
        out_size=out_size,
        target_schema=target_schema,
        objective_transform="identity",
        capacity_exhausted=capacity,
    )
    return _finish_result(
        model,
        batch=batch,
        case_forests=forests,
        effective_samples=effective,
        objectives=objectives,
        valid=valid,
        capacity=capacity,
        method=method,
        split_search="random" if random_thresholds else recipe.split_search,
    )


class _AbstractCARTRecipe(AbstractRecipe):
    max_depth: int = eqx.field(static=True)
    max_nodes: int = eqx.field(static=True)
    max_leaf_nodes: int | None = eqx.field(static=True)
    min_samples_split: int = eqx.field(static=True)
    min_samples_leaf: int = eqx.field(static=True)
    min_weight_leaf: float = eqx.field(static=True)
    min_gain: float = eqx.field(static=True)
    ccp_alpha: float = eqx.field(static=True)
    max_features: int | float | Literal["sqrt", "log2"] | None = eqx.field(static=True)
    split_search: SplitSearch = eqx.field(static=True)
    max_bins: int = eqx.field(static=True)
    max_categories: int = eqx.field(static=True)
    monotonic_constraints: tuple[int, ...] = eqx.field(static=True)
    interaction_constraints: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    num_classes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_depth: int = 6,
        max_nodes: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_leaf: float = 0.0,
        min_gain: float = 0.0,
        ccp_alpha: float = 0.0,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        split_search: SplitSearch = "exact",
        max_bins: int = 256,
        max_categories: int = 32,
        monotonic_constraints: tuple[int, ...] = (),
        interaction_constraints: tuple[tuple[int, ...], ...] = (),
        num_classes: int | None = None,
    ):
        depth, capacity = _validate_common(
            max_depth=max_depth,
            max_nodes=max_nodes,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_leaf=min_weight_leaf,
            min_gain=min_gain,
            max_features=max_features,
            max_bins=max_bins,
            max_categories=max_categories,
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
        )
        if max_leaf_nodes is not None and max_leaf_nodes < 1:
            raise ValueError("max_leaf_nodes must be positive.")
        if ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be nonnegative.")
        if split_search not in {"exact", "histogram", "random"}:
            raise ValueError("Unsupported split search.")
        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.max_depth = depth
        self.max_nodes = capacity
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_leaf = float(min_weight_leaf)
        self.min_gain = float(min_gain)
        self.ccp_alpha = float(ccp_alpha)
        self.max_features = max_features
        self.split_search = split_search
        self.max_bins = int(max_bins)
        self.max_categories = int(max_categories)
        self.monotonic_constraints = tuple(monotonic_constraints)
        self.interaction_constraints = tuple(
            tuple(group) for group in interaction_constraints
        )
        self.num_classes = num_classes


class DecisionTreeRegressor(_AbstractCARTRecipe):
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=False,
            tree_count=1,
            bootstrap=False,
            random_thresholds=self.split_search == "random",
            method="decision_tree_regressor",
        )


class DecisionTreeClassifier(_AbstractCARTRecipe):
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=True,
            tree_count=1,
            bootstrap=False,
            random_thresholds=self.split_search == "random",
            method="decision_tree_classifier",
        )


class RandomTreeRegressor(_AbstractCARTRecipe):
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=False,
            tree_count=1,
            bootstrap=False,
            random_thresholds=True,
            method="random_tree_regressor",
        )


class RandomTreeClassifier(_AbstractCARTRecipe):
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=True,
            tree_count=1,
            bootstrap=False,
            random_thresholds=True,
            method="random_tree_classifier",
        )


class ExtraTreeRegressor(_AbstractCARTRecipe):
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=False,
            tree_count=1,
            bootstrap=False,
            random_thresholds=True,
            method="extra_tree_regressor",
        )


class ExtraTreeClassifier(_AbstractCARTRecipe):
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=True,
            tree_count=1,
            bootstrap=False,
            random_thresholds=True,
            method="extra_tree_classifier",
        )


def _init_forest(instance, *, n_estimators: int, bootstrap: bool, common: dict[str, Any]):
    _AbstractCARTRecipe.__init__(instance, **common)
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    instance.n_estimators = int(n_estimators)
    instance.bootstrap = bool(bootstrap)


class RandomForestRegressor(_AbstractCARTRecipe):
    n_estimators: int = eqx.field(static=True)
    bootstrap: bool = eqx.field(static=True)

    def __init__(self, *, n_estimators: int = 100, bootstrap: bool = True, **kwargs):
        _init_forest(self, n_estimators=n_estimators, bootstrap=bootstrap, common=kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=False,
            tree_count=self.n_estimators,
            bootstrap=self.bootstrap,
            random_thresholds=False,
            method="random_forest_regressor",
        )


class RandomForestClassifier(_AbstractCARTRecipe):
    n_estimators: int = eqx.field(static=True)
    bootstrap: bool = eqx.field(static=True)

    def __init__(self, *, n_estimators: int = 100, bootstrap: bool = True, **kwargs):
        _init_forest(self, n_estimators=n_estimators, bootstrap=bootstrap, common=kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=True,
            tree_count=self.n_estimators,
            bootstrap=self.bootstrap,
            random_thresholds=False,
            method="random_forest_classifier",
        )


class ExtraTreesRegressor(_AbstractCARTRecipe):
    n_estimators: int = eqx.field(static=True)
    bootstrap: bool = eqx.field(static=True)

    def __init__(self, *, n_estimators: int = 100, bootstrap: bool = False, **kwargs):
        _init_forest(self, n_estimators=n_estimators, bootstrap=bootstrap, common=kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=False,
            tree_count=self.n_estimators,
            bootstrap=self.bootstrap,
            random_thresholds=True,
            method="extra_trees_regressor",
        )


class ExtraTreesClassifier(_AbstractCARTRecipe):
    n_estimators: int = eqx.field(static=True)
    bootstrap: bool = eqx.field(static=True)

    def __init__(self, *, n_estimators: int = 100, bootstrap: bool = False, **kwargs):
        _init_forest(self, n_estimators=n_estimators, bootstrap=bootstrap, common=kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_bagged(
            self,
            batch,
            key=key,
            classification=True,
            tree_count=self.n_estimators,
            bootstrap=self.bootstrap,
            random_thresholds=True,
            method="extra_trees_classifier",
        )


class AdaBoostClassifier(_AbstractCARTRecipe):
    n_estimators: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        **kwargs,
    ):
        super().__init__(max_depth=max_depth, **kwargs)
        if n_estimators <= 0 or learning_rate <= 0.0:
            raise ValueError("AdaBoost requires positive estimators and learning rate.")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None and (
            self.split_search == "random"
            or _feature_count(self.max_features, batch.feature_count)
            < batch.feature_count
        ):
            raise ValueError("Stochastic AdaBoost requires an explicit JAX key.")
        x, y, weight, sample_mask, classes, _, target_schema = _prepare_batch(
            batch, classification=True, num_classes=self.num_classes
        )
        case_count = int(x.shape[0])
        keys = (
            [None] * (case_count * self.n_estimators)
            if key is None
            else list(jax.random.split(key, case_count * self.n_estimators))
        )
        forests = []
        tree_weights = []
        bases = []
        effective = []
        objectives = []
        valid = []
        capacity = []
        for case in range(case_count):
            current_weight = weight[case] / _weight_sum(weight[case])
            forest = []
            alphas = []
            for index in range(self.n_estimators):
                tree = _build_tree(
                    x[case],
                    current_weight,
                    sample_mask[case] & (current_weight > 0.0),
                    node_capacity=self.max_nodes,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_leaf=self.min_weight_leaf,
                    min_gain=self.min_gain + self.ccp_alpha,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_features=self.max_features,
                    split_search=self.split_search,
                    max_bins=self.max_bins,
                    feature_kinds=batch.feature_schema.kinds,
                    max_categories=self.max_categories,
                    monotonic_constraints=(),
                    interaction_constraints=self.interaction_constraints,
                    key=keys[case * self.n_estimators + index],
                    y=y[case],
                )
                predicted = jnp.argmax(_tree_predict(tree, x[case]), axis=-1)
                truth = jnp.argmax(y[case], axis=-1)
                incorrect = predicted != truth
                error = jnp.sum(
                    jnp.where(sample_mask[case], current_weight * incorrect, 0.0)
                )
                error = jnp.clip(
                    error,
                    jnp.finfo(current_weight.dtype).eps,
                    1.0 - jnp.finfo(current_weight.dtype).eps,
                )
                alpha = self.learning_rate * (
                    jnp.log((1.0 - error) / error) + jnp.log(max(classes - 1, 1))
                )
                alpha = jnp.maximum(alpha, 0.0)
                hard_leaves = jax.nn.one_hot(
                    jnp.argmax(tree["leaf_value"], axis=-1), classes, dtype=y.dtype
                )
                tree["leaf_value"] = hard_leaves
                forest.append(tree)
                alphas.append(alpha)
                current_weight = current_weight * jnp.exp(alpha * incorrect)
                current_weight = current_weight / _weight_sum(current_weight)
            forests.append(forest)
            tree_weights.append(jnp.stack(alphas))
            bases.append(jnp.zeros((classes,), dtype=y.dtype))
            effective.append(jnp.sum(sample_mask[case] & (weight[case] > 0.0)))
            okay = _as_bool(jnp.sum(weight[case]) > 0.0)
            valid.append(okay)
            exhausted = any(bool(tree["capacity_exhausted"]) for tree in forest)
            capacity.append(exhausted)
            raw = sum(
                alpha * _tree_predict(tree, x[case])
                for alpha, tree in zip(alphas, forest, strict=True)
            )
            probabilities = jax.nn.softmax(raw, axis=-1)
            objectives.append(
                -jnp.sum(
                    weight[case, :, None]
                    * y[case]
                    * jnp.log(jnp.maximum(probabilities, 1e-12))
                )
                / _weight_sum(weight[case])
            )
        model = _stack_model(
            forests,
            tree_weights,
            bases,
            batch=batch,
            tree_capacity=self.n_estimators,
            node_capacity=self.max_nodes,
            category_capacity=self.max_categories,
            output_count=classes,
            out_size=classes,
            target_schema=target_schema,
            objective_transform="softmax",
            capacity_exhausted=capacity,
        )
        return _finish_result(
            model,
            batch=batch,
            case_forests=forests,
            effective_samples=effective,
            objectives=objectives,
            valid=valid,
            capacity=capacity,
            method="adaboost_classifier",
            split_search=self.split_search,
        )


class AdaBoostRegressor(_AbstractCARTRecipe):
    """Native AdaBoost.R2 with its defining weighted-median prediction rule."""

    n_estimators: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    loss: Literal["linear", "square", "exponential"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: Literal["linear", "square", "exponential"] = "linear",
        max_depth: int = 3,
        **kwargs,
    ):
        super().__init__(max_depth=max_depth, **kwargs)
        if n_estimators <= 0 or learning_rate <= 0.0:
            raise ValueError(
                "AdaBoost.R2 requires positive estimators and learning rate."
            )
        if loss not in {"linear", "square", "exponential"}:
            raise ValueError("Unsupported AdaBoost.R2 loss.")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None and (
            self.split_search == "random"
            or _feature_count(self.max_features, batch.feature_count)
            < batch.feature_count
        ):
            raise ValueError("Stochastic AdaBoost.R2 requires an explicit JAX key.")
        x, y, weight, sample_mask, outputs, output_shape, target_schema = _prepare_batch(
            batch, classification=False, num_classes=None
        )
        if jnp.issubdtype(y.dtype, jnp.complexfloating):
            raise TypeError("AdaBoost.R2 weighted medians require real targets.")
        case_count = int(x.shape[0])
        keys = (
            [None] * (case_count * self.n_estimators)
            if key is None
            else list(jax.random.split(key, case_count * self.n_estimators))
        )
        forests = []
        tree_weights = []
        bases = []
        effective = []
        objectives = []
        valid = []
        capacity = []
        for case in range(case_count):
            current_weight = weight[case] / _weight_sum(weight[case])
            forest = []
            alphas = []
            predictions = []
            for index in range(self.n_estimators):
                tree = _build_tree(
                    x[case],
                    current_weight,
                    sample_mask[case] & (current_weight > 0.0),
                    node_capacity=self.max_nodes,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_leaf=self.min_weight_leaf,
                    min_gain=self.min_gain + self.ccp_alpha,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_features=self.max_features,
                    split_search=self.split_search,
                    max_bins=self.max_bins,
                    feature_kinds=batch.feature_schema.kinds,
                    max_categories=self.max_categories,
                    monotonic_constraints=self.monotonic_constraints,
                    interaction_constraints=self.interaction_constraints,
                    key=keys[case * self.n_estimators + index],
                    y=y[case],
                )
                prediction = _tree_predict(tree, x[case])
                sample_error = jnp.mean(jnp.abs(prediction - y[case]), axis=-1)
                maximum = jnp.max(jnp.where(sample_mask[case], sample_error, 0.0))
                normalized = sample_error / jnp.maximum(
                    maximum, jnp.finfo(current_weight.dtype).eps
                )
                if self.loss == "square":
                    normalized = normalized**2
                elif self.loss == "exponential":
                    normalized = 1.0 - jnp.exp(-normalized)
                estimator_error = jnp.sum(
                    jnp.where(sample_mask[case], current_weight * normalized, 0.0)
                )
                estimator_error = jnp.clip(
                    estimator_error,
                    jnp.finfo(current_weight.dtype).eps,
                    0.5 - jnp.finfo(current_weight.dtype).eps,
                )
                beta = estimator_error / (1.0 - estimator_error)
                alpha = self.learning_rate * jnp.log(1.0 / beta)
                current_weight = current_weight * beta ** (
                    self.learning_rate * (1.0 - normalized)
                )
                current_weight = current_weight / _weight_sum(current_weight)
                forest.append(tree)
                alphas.append(alpha)
                predictions.append(prediction)
            alpha_array = jnp.stack(alphas)
            tree_predictions = jnp.stack(predictions, axis=1)
            median = _weighted_median_case(
                tree_predictions,
                alpha_array,
                jnp.ones((self.n_estimators,), dtype=bool),
                jnp.zeros((outputs,), dtype=y.dtype),
            )
            forests.append(forest)
            tree_weights.append(alpha_array)
            bases.append(jnp.zeros((outputs,), dtype=y.dtype))
            effective.append(jnp.sum(sample_mask[case] & (weight[case] > 0.0)))
            okay = _as_bool(jnp.sum(weight[case]) > 0.0)
            valid.append(okay)
            exhausted = any(bool(tree["capacity_exhausted"]) for tree in forest)
            capacity.append(exhausted)
            objectives.append(
                jnp.sum(
                    weight[case, :, None]
                    * jnp.real((median - y[case]) * jnp.conj(median - y[case]))
                )
                / _weight_sum(weight[case])
            )
        out_size = output_shape if output_shape else "scalar"
        model = _stack_model(
            forests,
            tree_weights,
            bases,
            batch=batch,
            tree_capacity=self.n_estimators,
            node_capacity=self.max_nodes,
            category_capacity=self.max_categories,
            output_count=outputs,
            out_size=out_size,
            target_schema=target_schema,
            objective_transform="identity",
            aggregation="weighted_median",
            capacity_exhausted=capacity,
        )
        return _finish_result(
            model,
            batch=batch,
            case_forests=forests,
            effective_samples=effective,
            objectives=objectives,
            valid=valid,
            capacity=capacity,
            method="adaboost_regressor",
            split_search=self.split_search,
        )


def _boosting_gradients(
    objective: XGBObjective,
    raw: Array,
    y: Array,
    weight: Array,
    sample_mask: Array,
    groups: Array | None,
) -> tuple[Array, Array, Array]:
    weighted = jnp.where(sample_mask, weight, 0.0)[:, None]
    if objective == "squared_error":
        residual = raw - y
        return (
            residual * weighted,
            jnp.ones_like(raw.real) * weighted,
            0.5 * jnp.sum(weighted * jnp.real(residual * jnp.conj(residual))),
        )
    if objective == "logistic":
        probability = jax.nn.sigmoid(raw)
        gradient = (probability - y) * weighted
        hessian = jnp.maximum(probability * (1.0 - probability), 1e-7) * weighted
        loss = -jnp.sum(
            weighted
            * (y * jax.nn.log_sigmoid(raw) + (1.0 - y) * jax.nn.log_sigmoid(-raw))
        )
        return gradient, hessian, loss
    if objective == "softmax":
        probability = jax.nn.softmax(raw, axis=-1)
        gradient = (probability - y) * weighted
        hessian = jnp.maximum(2.0 * probability * (1.0 - probability), 1e-7) * weighted
        loss = -jnp.sum(weighted * y * jnp.log(jnp.maximum(probability, 1e-12)))
        return gradient, hessian, loss
    if objective == "poisson":
        mean = jnp.exp(raw)
        return (
            (mean - y) * weighted,
            mean * weighted,
            jnp.sum(weighted * (mean - y * raw)),
        )
    if objective == "pairwise_ranking":
        if groups is None:
            raise ValueError("pairwise_ranking requires integer group identifiers.")
        scores = raw[:, 0]
        labels = y[:, 0]
        preference = (labels[:, None] > labels[None, :]) & (
            groups[:, None] == groups[None, :]
        )
        preference = preference & sample_mask[:, None] & sample_mask[None, :]
        difference = scores[:, None] - scores[None, :]
        probability = jax.nn.sigmoid(-difference)
        pair_weight = jnp.sqrt(weight[:, None] * weight[None, :])
        pair = preference * pair_weight
        first = -jnp.sum(pair * probability, axis=1)
        second = jnp.sum(pair * probability, axis=0)
        gradient = first + second
        curvature = pair * probability * (1.0 - probability)
        hessian = jnp.sum(curvature, axis=1) + jnp.sum(curvature, axis=0)
        loss = jnp.sum(pair * jax.nn.softplus(-difference))
        return gradient[:, None], jnp.maximum(hessian[:, None], 1e-7), loss
    raise ValueError(f"Unsupported boosting objective {objective!r}.")


def _initial_score(objective: XGBObjective, y: Array, weight: Array) -> Array:
    denominator = jnp.maximum(jnp.sum(weight), jnp.finfo(weight.dtype).tiny)
    mean = jnp.sum(weight[:, None] * y, axis=0) / denominator
    if objective == "logistic":
        probability = jnp.clip(mean, 1e-6, 1.0 - 1e-6)
        return jnp.log(probability) - jnp.log1p(-probability)
    if objective == "softmax":
        probability = jnp.clip(mean, 1e-6, 1.0)
        return jnp.log(probability)
    if objective == "poisson":
        return jnp.log(jnp.maximum(mean, 1e-6))
    if objective == "pairwise_ranking":
        return jnp.zeros_like(mean)
    return mean


def _fit_boosted(
    recipe, batch: MLBatch, *, key: Any, classical: bool, method: str
) -> FitResult:
    stochastic = (
        recipe.subsample < 1.0
        or recipe.colsample < 1.0
        or recipe.split_search == "random"
    )
    if stochastic and key is None:
        raise ValueError(f"{method} stochastic fitting requires an explicit JAX key.")
    objective_name = recipe.objective
    classification = objective_name in {"auto", "logistic", "softmax"}
    x, y, weight, sample_mask, output_count, output_shape, target_schema = _prepare_batch(
        batch,
        classification=classification,
        num_classes=recipe.num_classes if classification else None,
    )
    if objective_name == "auto":
        objective_name = "logistic" if output_count == 2 else "softmax"
    if objective_name == "logistic":
        if output_count != 2:
            raise ValueError("logistic boosting requires exactly two classes.")
        y = y[..., 1:]
        output_count = 1
        output_shape = ()
        target_schema = TargetSchema("binary", class_labels=target_schema.class_labels)
    elif objective_name == "pairwise_ranking":
        if batch.target_shape != ():
            raise ValueError("Ranking labels must be scalar.")
        output_count = 1
        output_shape = ()
        target_schema = TargetSchema("ranking", names=batch.target_schema.names)
    elif objective_name == "poisson" and _as_bool(jnp.any(y < 0.0)):
        raise ValueError("Poisson boosting requires nonnegative targets.")
    if jnp.issubdtype(y.dtype, jnp.complexfloating) and objective_name != "squared_error":
        raise TypeError("Only squared-error boosting supports complex targets.")

    case_count = int(x.shape[0])
    all_keys = (
        [None] * (case_count * recipe.n_estimators)
        if key is None
        else list(jax.random.split(key, case_count * recipe.n_estimators))
    )
    forests = []
    tree_weights = []
    bases = []
    effective = []
    objectives = []
    valid = []
    capacity = []
    groups = (
        None
        if batch.groups is None
        else batch.groups.reshape((case_count, batch.sample_count))
    )
    for case in range(case_count):
        base = _initial_score(objective_name, y[case], weight[case])
        raw = jnp.broadcast_to(base, y[case].shape)
        forest = []
        for index in range(recipe.n_estimators):
            tree_key = all_keys[case * recipe.n_estimators + index]
            row_mask = sample_mask[case]
            if recipe.subsample < 1.0:
                tree_key, row_key = jax.random.split(tree_key)
                row_mask = row_mask & jax.random.bernoulli(
                    row_key, recipe.subsample, (batch.sample_count,)
                )
            if classical:
                if objective_name == "squared_error":
                    pseudo_target = y[case] - raw
                elif objective_name == "logistic":
                    pseudo_target = y[case] - jax.nn.sigmoid(raw)
                elif objective_name == "softmax":
                    pseudo_target = y[case] - jax.nn.softmax(raw, axis=-1)
                else:
                    raise ValueError(
                        "Classical gradient boosting supports squared, logistic, and softmax objectives."
                    )
                tree = _build_tree(
                    x[case],
                    weight[case],
                    row_mask,
                    node_capacity=recipe.max_nodes,
                    max_depth=recipe.max_depth,
                    min_samples_split=recipe.min_samples_split,
                    min_samples_leaf=recipe.min_samples_leaf,
                    min_weight_leaf=recipe.min_weight_leaf,
                    min_gain=recipe.min_gain + recipe.ccp_alpha,
                    max_leaf_nodes=recipe.max_leaf_nodes,
                    max_features=recipe.colsample,
                    split_search=recipe.split_search,
                    max_bins=recipe.max_bins,
                    feature_kinds=batch.feature_schema.kinds,
                    max_categories=recipe.max_categories,
                    monotonic_constraints=recipe.monotonic_constraints,
                    interaction_constraints=recipe.interaction_constraints,
                    key=tree_key,
                    y=pseudo_target,
                )
            else:
                gradient, hessian, _ = _boosting_gradients(
                    objective_name,
                    raw,
                    y[case],
                    weight[case],
                    sample_mask[case],
                    None if groups is None else groups[case],
                )
                tree = _build_tree(
                    x[case],
                    weight[case],
                    row_mask,
                    node_capacity=recipe.max_nodes,
                    max_depth=recipe.max_depth,
                    min_samples_split=recipe.min_samples_split,
                    min_samples_leaf=recipe.min_samples_leaf,
                    min_weight_leaf=recipe.min_child_weight,
                    min_gain=recipe.min_gain + recipe.ccp_alpha,
                    max_leaf_nodes=recipe.max_leaf_nodes,
                    max_features=recipe.colsample,
                    split_search=recipe.split_search,
                    max_bins=recipe.max_bins,
                    feature_kinds=batch.feature_schema.kinds,
                    max_categories=recipe.max_categories,
                    monotonic_constraints=recipe.monotonic_constraints,
                    interaction_constraints=recipe.interaction_constraints,
                    key=tree_key,
                    gradient=gradient,
                    hessian=hessian,
                    l2_regularization=recipe.l2_regularization,
                    l1_regularization=recipe.l1_regularization,
                    gamma=recipe.gamma,
                    max_delta_step=recipe.max_delta_step,
                )
            update = _tree_predict(tree, x[case])
            raw = raw + recipe.learning_rate * update
            forest.append(tree)
        _, _, final_objective = _boosting_gradients(
            objective_name,
            raw,
            y[case],
            weight[case],
            sample_mask[case],
            None if groups is None else groups[case],
        )
        forests.append(forest)
        tree_weights.append(
            jnp.full((recipe.n_estimators,), recipe.learning_rate, dtype=y.real.dtype)
        )
        bases.append(base)
        effective.append(jnp.sum(sample_mask[case] & (weight[case] > 0.0)))
        okay = _as_bool(jnp.sum(weight[case]) > 0.0)
        valid.append(okay)
        exhausted = any(bool(tree["capacity_exhausted"]) for tree in forest)
        capacity.append(exhausted)
        objectives.append(final_objective / _weight_sum(weight[case]))
    transform = (
        "sigmoid"
        if objective_name == "logistic"
        else (
            "softmax"
            if objective_name == "softmax"
            else ("exponential" if objective_name == "poisson" else "identity")
        )
    )
    out_size = (
        output_shape if output_shape else (output_count if output_count > 1 else "scalar")
    )
    model = _stack_model(
        forests,
        tree_weights,
        bases,
        batch=batch,
        tree_capacity=recipe.n_estimators,
        node_capacity=recipe.max_nodes,
        category_capacity=recipe.max_categories,
        output_count=output_count,
        out_size=out_size,
        target_schema=target_schema,
        objective_transform=transform,
        capacity_exhausted=capacity,
    )
    return _finish_result(
        model,
        batch=batch,
        case_forests=forests,
        effective_samples=effective,
        objectives=objectives,
        valid=valid,
        capacity=capacity,
        method=method,
        split_search=recipe.split_search,
    )


class _AbstractBoostingRecipe(_AbstractCARTRecipe):
    n_estimators: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    objective: XGBObjective = eqx.field(static=True)
    subsample: float = eqx.field(static=True)
    colsample: float = eqx.field(static=True)
    l2_regularization: float = eqx.field(static=True)
    l1_regularization: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    min_child_weight: float = eqx.field(static=True)
    max_delta_step: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        objective: XGBObjective = "squared_error",
        subsample: float = 1.0,
        colsample: float = 1.0,
        l2_regularization: float = 1.0,
        l1_regularization: float = 0.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        max_delta_step: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_estimators <= 0 or learning_rate <= 0.0:
            raise ValueError("Boosting requires positive estimators and learning rate.")
        if objective not in {
            "auto",
            "squared_error",
            "logistic",
            "softmax",
            "poisson",
            "pairwise_ranking",
        }:
            raise ValueError("Unsupported boosting objective.")
        if not (0.0 < subsample <= 1.0 and 0.0 < colsample <= 1.0):
            raise ValueError("Row and column subsampling fractions must lie in (0, 1].")
        if min(l2_regularization, l1_regularization, gamma, min_child_weight) < 0.0:
            raise ValueError("Boosting regularization parameters must be nonnegative.")
        if max_delta_step is not None and max_delta_step <= 0.0:
            raise ValueError("max_delta_step must be positive when provided.")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.objective = objective
        self.subsample = float(subsample)
        self.colsample = float(colsample)
        self.l2_regularization = float(l2_regularization)
        self.l1_regularization = float(l1_regularization)
        self.gamma = float(gamma)
        self.min_child_weight = float(min_child_weight)
        self.max_delta_step = max_delta_step


class GradientBoostingRegressor(_AbstractBoostingRecipe):
    def __init__(
        self, *, objective: Literal["squared_error"] = "squared_error", **kwargs
    ):
        super().__init__(objective=objective, **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_boosted(
            self, batch, key=key, classical=True, method="gradient_boosting_regressor"
        )


class GradientBoostingClassifier(_AbstractBoostingRecipe):
    def __init__(
        self, *, objective: Literal["auto", "logistic", "softmax"] = "auto", **kwargs
    ):
        super().__init__(objective=objective, **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_boosted(
            self, batch, key=key, classical=True, method="gradient_boosting_classifier"
        )


class HistGradientBoostingRegressor(_AbstractBoostingRecipe):
    def __init__(self, **kwargs):
        kwargs["split_search"] = "histogram"
        super().__init__(objective="squared_error", **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_boosted(
            self,
            batch,
            key=key,
            classical=True,
            method="hist_gradient_boosting_regressor",
        )


class HistGradientBoostingClassifier(_AbstractBoostingRecipe):
    def __init__(
        self,
        *,
        objective: Literal["auto", "logistic", "softmax"] = "auto",
        **kwargs,
    ):
        kwargs["split_search"] = "histogram"
        super().__init__(objective=objective, **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_boosted(
            self,
            batch,
            key=key,
            classical=True,
            method="hist_gradient_boosting_classifier",
        )


class XGBoostRegressor(_AbstractBoostingRecipe):
    def __init__(
        self,
        *,
        objective: Literal["squared_error", "poisson"] = "squared_error",
        **kwargs,
    ):
        super().__init__(objective=objective, **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_boosted(
            self, batch, key=key, classical=False, method="xgboost_regressor"
        )


class XGBoostClassifier(_AbstractBoostingRecipe):
    def __init__(
        self, *, objective: Literal["auto", "logistic", "softmax"] = "auto", **kwargs
    ):
        super().__init__(objective=objective, **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        return _fit_boosted(
            self, batch, key=key, classical=False, method="xgboost_classifier"
        )


class XGBoostRanker(_AbstractBoostingRecipe):
    def __init__(self, **kwargs):
        super().__init__(objective="pairwise_ranking", **kwargs)

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if batch.groups is None:
            raise ValueError("XGBoostRanker requires batch groups.")
        return _fit_boosted(
            self, batch, key=key, classical=False, method="xgboost_ranker"
        )


__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomTreeClassifier",
    "RandomTreeRegressor",
    "SplitSearch",
    "TreeFitDiagnostics",
    "XGBObjective",
    "XGBoostClassifier",
    "XGBoostRanker",
    "XGBoostRegressor",
]
