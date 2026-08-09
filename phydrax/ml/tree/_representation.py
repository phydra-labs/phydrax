#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from .._schema import FeatureSchema, TargetSchema


ObjectiveTransform: TypeAlias = Literal[
    "identity", "sigmoid", "softmax", "exponential", "positive"
]

EnsembleAggregation: TypeAlias = Literal["sum", "weighted_median"]
TreeInputDType: TypeAlias = Literal["preserve", "float32", "float64"]


def apply_objective(raw: Array, transform: ObjectiveTransform, /) -> Array:
    """Apply a tree ensemble's declared link inverse without changing axes."""
    if transform == "identity":
        return raw
    if transform == "sigmoid":
        return jax.nn.sigmoid(raw)
    if transform == "softmax":
        return jax.nn.softmax(raw, axis=-1)
    if transform in {"exponential", "positive"}:
        return jnp.exp(raw)
    raise ValueError(f"Unsupported tree objective transform {transform!r}.")


def _tree_output_shape(
    out_size: int | tuple[int, ...] | Literal["scalar"],
) -> tuple[int, ...]:
    if out_size == "scalar":
        return ()
    if isinstance(out_size, int):
        return (out_size,)
    return out_size


def _traverse_one_tree(
    x: Array,
    feature_index: Array,
    threshold: Array,
    left_child: Array,
    right_child: Array,
    default_left: Array,
    split_kind: Array,
    category_values: Array,
    category_mask: Array,
    leaf_value: Array,
    node_mask: Array,
    leaf_mask: Array,
    max_steps: int,
) -> tuple[Array, Array, Array]:
    """Traverse one tree for one point using a statically bounded JAX loop."""
    node_capacity = feature_index.shape[0]
    path0 = jnp.zeros((node_capacity,), dtype=bool).at[0].set(node_mask[0])

    def step(_, state):
        node, done, path = state
        safe_node = jnp.clip(node, 0, node_capacity - 1)
        valid = node_mask[safe_node]
        is_leaf = leaf_mask[safe_node] | ~valid
        feature = jnp.clip(feature_index[safe_node], 0, x.shape[-1] - 1)
        value = x[feature]
        missing = ~jnp.isfinite(value)
        numeric_left = jnp.where(
            split_kind[safe_node] == 2,
            value < threshold[safe_node],
            value <= threshold[safe_node],
        )
        categories = category_values[safe_node]
        category_valid = category_mask[safe_node]
        categorical_left = jnp.any(category_valid & (value == categories))
        go_left = jnp.where(
            missing,
            default_left[safe_node],
            jnp.where(split_kind[safe_node] == 1, categorical_left, numeric_left),
        )
        child = jnp.where(go_left, left_child[safe_node], right_child[safe_node])
        child = jnp.clip(child, 0, node_capacity - 1)
        move = ~done & valid & ~is_leaf
        next_node = jnp.where(move, child, safe_node)
        next_path = path.at[next_node].set(path[next_node] | move)
        return next_node, done | is_leaf | ~valid, next_path

    node, done, path = jax.lax.fori_loop(
        0, max_steps, step, (jnp.asarray(0, dtype=jnp.int32), False, path0)
    )
    safe_node = jnp.clip(node, 0, node_capacity - 1)
    value = eqx.error_if(
        leaf_value[safe_node],
        ~done | ~node_mask[safe_node] | ~leaf_mask[safe_node],
        "Tree traversal exhausted its bound before reaching a valid leaf.",
    )
    return value, safe_node, path


def _predict_case(
    points: Array,
    feature_index: Array,
    threshold: Array,
    left_child: Array,
    right_child: Array,
    default_left: Array,
    split_kind: Array,
    category_values: Array,
    category_mask: Array,
    leaf_value: Array,
    node_mask: Array,
    leaf_mask: Array,
    tree_mask: Array,
    tree_weight: Array,
    base_score: Array,
    max_steps: int,
) -> tuple[Array, Array, Array, Array]:
    def point_prediction(point):
        values, leaves, paths = jax.vmap(
            lambda fi, th, lc, rc, dl, sk, cv, cm, lv, nm, lm: _traverse_one_tree(
                point, fi, th, lc, rc, dl, sk, cv, cm, lv, nm, lm, max_steps
            )
        )(
            feature_index,
            threshold,
            left_child,
            right_child,
            default_left,
            split_kind,
            category_values,
            category_mask,
            leaf_value,
            node_mask,
            leaf_mask,
        )
        effective_weight = jnp.where(tree_mask, tree_weight, 0.0)
        raw = base_score + jnp.sum(values * effective_weight[:, None], axis=0)
        return raw, values, leaves, paths

    return jax.vmap(point_prediction)(points)


def _weighted_median_case(
    values: Array,
    weights: Array,
    tree_mask: Array,
    base_score: Array,
) -> Array:
    effective = jnp.where(tree_mask, jnp.maximum(weights, 0.0), 0.0)
    order = jnp.argsort(values, axis=1)
    ordered_values = jnp.take_along_axis(values, order, axis=1)
    broadcast_weight = jnp.broadcast_to(effective[None, :, None], values.shape)
    ordered_weight = jnp.take_along_axis(broadcast_weight, order, axis=1)

    def accumulate(total, item):
        updated = total + item
        return updated, updated

    _, cumulative_tree_first = jax.lax.scan(
        accumulate,
        jnp.zeros_like(ordered_weight[:, 0, :]),
        jnp.swapaxes(ordered_weight, 0, 1),
    )
    cumulative = jnp.swapaxes(cumulative_tree_first, 0, 1)
    cutoff = 0.5 * cumulative[:, -1, :]
    median_index = jnp.argmax(cumulative >= cutoff[:, None, :], axis=1)
    median = jnp.take_along_axis(ordered_values, median_index[:, None, :], axis=1)[
        :, 0, :
    ]
    return base_score + median


class TreeStructureDiagnostics(StrictModule):
    """Pure-JAX validity and fixed-capacity utilization for a represented ensemble."""

    valid: Array
    used_trees: Array
    tree_capacity: Array
    used_nodes: Array
    node_capacity: Array
    used_leaves: Array
    maximum_depth_bound: Array
    capacity_exhausted: Array

    def __init__(
        self,
        *,
        valid: Any,
        used_trees: Any,
        tree_capacity: Any,
        used_nodes: Any,
        node_capacity: Any,
        used_leaves: Any,
        maximum_depth_bound: Any,
        capacity_exhausted: Any,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.used_trees = jnp.asarray(used_trees, dtype=jnp.int32)
        self.tree_capacity = jnp.asarray(tree_capacity, dtype=jnp.int32)
        self.used_nodes = jnp.asarray(used_nodes, dtype=jnp.int32)
        self.node_capacity = jnp.asarray(node_capacity, dtype=jnp.int32)
        self.used_leaves = jnp.asarray(used_leaves, dtype=jnp.int32)
        self.maximum_depth_bound = jnp.asarray(maximum_depth_bound, dtype=jnp.int32)
        self.capacity_exhausted = jnp.asarray(capacity_exhausted, dtype=bool)


class TreeEnsemble(AbstractArrayModel):
    """Frozen, fixed-capacity collection of array-native decision trees.

    Child indices are local to a tree. Split kind zero is numeric ``<=``, one is
    categorical membership, and two is numeric ``<``. Categorical membership is
    represented by the fixed final axis of ``category_values`` and ``category_mask``.
    Missing values always follow the explicitly stored ``default_left`` direction.
    All traversal loops are bounded by ``max_steps`` and therefore remain compilable
    under JAX transformations.
    """

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
    tree_mask: Array
    tree_weight: Array
    node_gain: Array
    node_cover: Array
    base_score: Array
    feature_schema: FeatureSchema = eqx.field(static=True)
    target_schema: TargetSchema = eqx.field(static=True)
    objective_transform: ObjectiveTransform = eqx.field(static=True)
    aggregation: EnsembleAggregation = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    input_dtype: TreeInputDType = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    capacity_exhausted: Array

    def __init__(
        self,
        *,
        feature_index: ArrayLike,
        threshold: ArrayLike,
        left_child: ArrayLike,
        right_child: ArrayLike,
        default_left: ArrayLike,
        leaf_value: ArrayLike,
        node_mask: ArrayLike,
        leaf_mask: ArrayLike,
        tree_mask: ArrayLike,
        base_score: ArrayLike,
        feature_schema: FeatureSchema,
        target_schema: TargetSchema | None = None,
        objective_transform: ObjectiveTransform = "identity",
        aggregation: EnsembleAggregation = "sum",
        input_dtype: TreeInputDType = "preserve",
        tree_weight: ArrayLike | None = None,
        split_kind: ArrayLike | None = None,
        category_values: ArrayLike | None = None,
        category_mask: ArrayLike | None = None,
        node_gain: ArrayLike | None = None,
        node_cover: ArrayLike | None = None,
        case_shape: tuple[int, ...] = (),
        out_size: int | tuple[int, ...] | Literal["scalar"] | None = None,
        max_steps: int | None = None,
        capacity_exhausted: ArrayLike = False,
    ):
        case_shape_ = tuple(int(size) for size in case_shape)
        prefix = case_shape_
        feature_index_ = jnp.asarray(feature_index, dtype=jnp.int32)
        if feature_index_.ndim != len(prefix) + 2:
            raise ValueError("feature_index must have shape case_shape + (tree, node).")
        if tuple(feature_index_.shape[: len(prefix)]) != prefix:
            raise ValueError("feature_index does not begin with case_shape.")
        tree_capacity, node_capacity = map(int, feature_index_.shape[-2:])
        if tree_capacity <= 0 or node_capacity <= 0:
            raise ValueError("Tree and node capacities must be positive.")
        node_shape = prefix + (tree_capacity, node_capacity)

        threshold_ = jnp.asarray(threshold)
        if not jnp.issubdtype(threshold_.dtype, jnp.inexact):
            threshold_ = threshold_.astype(jnp.float32)
        left_child_ = jnp.asarray(left_child, dtype=jnp.int32)
        right_child_ = jnp.asarray(right_child, dtype=jnp.int32)
        default_left_ = jnp.asarray(default_left, dtype=bool)
        node_mask_ = jnp.asarray(node_mask, dtype=bool)
        leaf_mask_ = jnp.asarray(leaf_mask, dtype=bool)
        for name, value in (
            ("threshold", threshold_),
            ("left_child", left_child_),
            ("right_child", right_child_),
            ("default_left", default_left_),
            ("node_mask", node_mask_),
            ("leaf_mask", leaf_mask_),
        ):
            if value.shape != node_shape:
                raise ValueError(
                    f"{name} must have shape {node_shape}; got {value.shape}."
                )

        leaf_value_ = jnp.asarray(leaf_value)
        if not jnp.issubdtype(leaf_value_.dtype, jnp.inexact):
            leaf_value_ = leaf_value_.astype(jnp.float32)
        if leaf_value_.ndim != len(prefix) + 3 or leaf_value_.shape[:-1] != node_shape:
            raise ValueError(
                "leaf_value must have shape case_shape + (tree, node, output)."
            )
        output_count = int(leaf_value_.shape[-1])
        if output_count <= 0:
            raise ValueError("Tree outputs must be non-empty.")
        inferred_out: int | tuple[int, ...] | Literal["scalar"] = (
            "scalar" if output_count == 1 else output_count
        )
        out_size_ = inferred_out if out_size is None else out_size
        expected_outputs = math.prod(_tree_output_shape(out_size_))
        if (1 if out_size_ == "scalar" else expected_outputs) != output_count:
            raise ValueError("out_size does not match the leaf value output axis.")

        tree_shape = prefix + (tree_capacity,)
        tree_mask_ = jnp.asarray(tree_mask, dtype=bool)
        if tree_mask_.shape != tree_shape:
            raise ValueError(f"tree_mask must have shape {tree_shape}.")
        tree_weight_ = (
            jnp.ones(tree_shape, dtype=leaf_value_.real.dtype)
            if tree_weight is None
            else jnp.asarray(tree_weight, dtype=leaf_value_.real.dtype)
        )
        if tree_weight_.shape != tree_shape:
            raise ValueError(f"tree_weight must have shape {tree_shape}.")

        split_kind_ = (
            jnp.zeros(node_shape, dtype=jnp.int8)
            if split_kind is None
            else jnp.asarray(split_kind, dtype=jnp.int8)
        )
        if split_kind_.shape != node_shape:
            raise ValueError(f"split_kind must have shape {node_shape}.")
        if category_values is None:
            category_values_ = jnp.zeros(node_shape + (1,), dtype=threshold_.dtype)
            category_mask_ = jnp.zeros(node_shape + (1,), dtype=bool)
        else:
            category_values_ = jnp.asarray(category_values, dtype=threshold_.dtype)
            if (
                category_values_.shape[:-1] != node_shape
                or category_values_.shape[-1] <= 0
            ):
                raise ValueError(
                    "category_values must have shape case_shape + (tree, node, category)."
                )
            category_mask_ = (
                jnp.ones(category_values_.shape, dtype=bool)
                if category_mask is None
                else jnp.asarray(category_mask, dtype=bool)
            )
            if category_mask_.shape != category_values_.shape:
                raise ValueError("category_mask must match category_values.")

        node_gain_ = (
            jnp.zeros(node_shape, dtype=leaf_value_.real.dtype)
            if node_gain is None
            else jnp.asarray(node_gain, dtype=leaf_value_.real.dtype)
        )
        node_cover_ = (
            jnp.zeros(node_shape, dtype=leaf_value_.real.dtype)
            if node_cover is None
            else jnp.asarray(node_cover, dtype=leaf_value_.real.dtype)
        )
        if node_gain_.shape != node_shape or node_cover_.shape != node_shape:
            raise ValueError("node_gain and node_cover must match the node array shape.")

        base_score_ = jnp.asarray(base_score, dtype=leaf_value_.dtype)
        if base_score_.shape != prefix + (output_count,):
            raise ValueError("base_score must have shape case_shape + (output,).")
        if len(feature_schema.names) <= 0:
            raise ValueError("feature_schema must be non-empty.")
        if aggregation not in {"sum", "weighted_median"}:
            raise ValueError(f"Unsupported ensemble aggregation {aggregation!r}.")
        if input_dtype not in {"preserve", "float32", "float64"}:
            raise ValueError(f"Unsupported tree input dtype policy {input_dtype!r}.")
        if aggregation == "weighted_median" and jnp.issubdtype(
            leaf_value_.dtype, jnp.complexfloating
        ):
            raise TypeError("Weighted-median tree aggregation requires real leaf values.")
        if objective_transform not in {
            "identity",
            "sigmoid",
            "softmax",
            "exponential",
            "positive",
        }:
            raise ValueError(f"Unsupported objective transform {objective_transform!r}.")
        if objective_transform == "sigmoid" and output_count != 1:
            raise ValueError("A sigmoid tree objective requires one raw output.")
        if objective_transform == "softmax" and output_count < 2:
            raise ValueError(
                "A softmax tree objective requires at least two raw outputs."
            )
        max_steps_ = node_capacity if max_steps is None else int(max_steps)
        if max_steps_ <= 0:
            raise ValueError("max_steps must be positive.")

        self.feature_index = feature_index_
        self.threshold = threshold_
        self.left_child = left_child_
        self.right_child = right_child_
        self.default_left = default_left_
        self.split_kind = split_kind_
        self.category_values = category_values_
        self.category_mask = category_mask_
        self.leaf_value = leaf_value_
        self.node_mask = node_mask_
        self.input_dtype = input_dtype
        self.leaf_mask = leaf_mask_
        self.tree_mask = tree_mask_
        self.tree_weight = tree_weight_
        self.node_gain = node_gain_
        self.node_cover = node_cover_
        self.base_score = base_score_
        self.feature_schema = feature_schema
        self.target_schema = TargetSchema() if target_schema is None else target_schema
        self.objective_transform = objective_transform
        self.aggregation = aggregation
        self.case_shape = case_shape_
        self.in_size = len(feature_schema.names)
        self.out_size = out_size_
        self.max_steps = max_steps_
        self.capacity_exhausted = jnp.broadcast_to(
            jnp.asarray(capacity_exhausted, dtype=bool), case_shape_
        )

    @property
    def class_schema(self) -> TargetSchema:
        """Return the class/target vocabulary and semantics carried by the model."""
        return self.target_schema

    @property
    def tree_capacity(self) -> int:
        return int(self.feature_index.shape[-2])

    @property
    def node_capacity(self) -> int:
        return int(self.feature_index.shape[-1])

    @property
    def output_count(self) -> int:
        return int(self.leaf_value.shape[-1])

    def _flat_case_arrays(self):
        count = math.prod(self.case_shape) if self.case_shape else 1
        return (
            self.feature_index.reshape((count,) + self.feature_index.shape[-2:]),
            self.threshold.reshape((count,) + self.threshold.shape[-2:]),
            self.left_child.reshape((count,) + self.left_child.shape[-2:]),
            self.right_child.reshape((count,) + self.right_child.shape[-2:]),
            self.default_left.reshape((count,) + self.default_left.shape[-2:]),
            self.split_kind.reshape((count,) + self.split_kind.shape[-2:]),
            self.category_values.reshape((count,) + self.category_values.shape[-3:]),
            self.category_mask.reshape((count,) + self.category_mask.shape[-3:]),
            self.leaf_value.reshape((count,) + self.leaf_value.shape[-3:]),
            self.node_mask.reshape((count,) + self.node_mask.shape[-2:]),
            self.leaf_mask.reshape((count,) + self.leaf_mask.shape[-2:]),
            self.tree_mask.reshape((count,) + self.tree_mask.shape[-1:]),
            self.tree_weight.reshape((count,) + self.tree_weight.shape[-1:]),
            self.base_score.reshape((count, self.output_count)),
        )

    def _evaluate(self, x: Any, /) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
        values = jnp.asarray(x)
        if values.shape[-1:] != (self.in_size,):
            raise ValueError(f"Expected final feature axis of size {self.in_size}.")
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Hard tree ordering is undefined for complex features.")
        if self.input_dtype != "preserve":
            values = values.astype(self.input_dtype)
        values = eqx.error_if(
            values,
            jnp.any(~self.structure_diagnostics().valid),
            "TreeEnsemble contains invalid active children, features, or roots.",
        )
        arrays = self._flat_case_arrays()
        if self.case_shape:
            if tuple(values.shape[: len(self.case_shape)]) != self.case_shape:
                raise ValueError(
                    "Case-dependent tree parameters require inputs beginning with case_shape."
                )
            point_shape = tuple(values.shape[len(self.case_shape) : -1])
            points = values.reshape((math.prod(self.case_shape), -1, self.in_size))
            raw, tree_values, leaves, paths = jax.vmap(
                lambda pts, *case_arrays: _predict_case(pts, *case_arrays, self.max_steps)
            )(points, *arrays)
            if self.aggregation == "weighted_median":
                raw = jax.vmap(_weighted_median_case)(
                    tree_values, arrays[12], arrays[11], arrays[13]
                )
            lead_shape = self.case_shape + point_shape
        else:
            point_shape = tuple(values.shape[:-1])
            points = values.reshape((-1, self.in_size))
            raw, tree_values, leaves, paths = _predict_case(
                points, *(a[0] for a in arrays), self.max_steps
            )
            if self.aggregation == "weighted_median":
                raw = _weighted_median_case(
                    tree_values, arrays[12][0], arrays[11][0], arrays[13][0]
                )
            lead_shape = point_shape
        return raw, tree_values, leaves, paths, lead_shape

    def predict_raw(self, x: Any, /) -> Array:
        raw, _, _, _, lead_shape = self._evaluate(x)
        output_shape = _tree_output_shape(self.out_size)
        if self.out_size == "scalar":
            return raw.reshape(lead_shape + (1,))[..., 0]
        return raw.reshape(lead_shape + output_shape)

    def predict_trees(self, x: Any, /) -> Array:
        """Return reached leaf values before tree weights or objective transform."""
        _, tree_values, _, _, lead_shape = self._evaluate(x)
        output_shape = _tree_output_shape(self.out_size)
        return tree_values.reshape(lead_shape + (self.tree_capacity,) + output_shape)

    def predict_leaf(self, x: Any, /) -> Array:
        """Return hard reached-node indices; this output is nondifferentiable."""
        _, _, leaves, _, lead_shape = self._evaluate(x)
        return leaves.reshape(lead_shape + (self.tree_capacity,))

    def decision_path(self, x: Any, /) -> Array:
        """Return hard visited-node masks; this output is nondifferentiable."""
        _, _, _, paths, lead_shape = self._evaluate(x)
        return paths.reshape(lead_shape + (self.tree_capacity, self.node_capacity))

    def predict_labels(self, x: Any, /, *, threshold: float = 0.5) -> Array:
        """Return nondifferentiable integer class indices."""
        prediction = self(x)
        if self.objective_transform == "sigmoid":
            return (prediction >= threshold).astype(jnp.int32)
        if self.objective_transform == "softmax" or self.target_schema.kind in {
            "binary",
            "multiclass",
        }:
            return jnp.argmax(prediction, axis=-1).astype(jnp.int32)
        raise ValueError("Class labels require a classification target schema.")

    def structure_diagnostics(self, /) -> TreeStructureDiagnostics:
        children_valid = jnp.where(
            self.node_mask & ~self.leaf_mask,
            (self.left_child >= 0)
            & (self.left_child < self.node_capacity)
            & (self.right_child >= 0)
            & (self.right_child < self.node_capacity),
            True,
        )
        feature_valid = jnp.where(
            self.node_mask & ~self.leaf_mask,
            (self.feature_index >= 0) & (self.feature_index < self.in_size),
            True,
        )
        split = self.node_mask & ~self.leaf_mask
        split_kind_valid = jnp.where(
            split,
            (self.split_kind == 0) | (self.split_kind == 1) | (self.split_kind == 2),
            True,
        )
        numeric_threshold_valid = jnp.where(
            split & (self.split_kind != 1), ~jnp.isnan(self.threshold), True
        )
        categorical_metadata_valid = jnp.where(
            split & (self.split_kind == 1),
            jnp.any(self.category_mask, axis=-1)
            & jnp.all(~self.category_mask | jnp.isfinite(self.category_values), axis=-1),
            True,
        )
        leaf_value_valid = jnp.where(
            self.node_mask & self.leaf_mask,
            jnp.all(jnp.isfinite(self.leaf_value), axis=-1),
            True,
        )
        score_valid = jnp.all(jnp.isfinite(self.base_score), axis=-1) & jnp.all(
            ~self.tree_mask | jnp.isfinite(self.tree_weight), axis=-1
        )
        roots_valid = jnp.all(~self.tree_mask | self.node_mask[..., 0], axis=-1)
        valid = (
            jnp.all(children_valid, axis=(-2, -1))
            & jnp.all(feature_valid, axis=(-2, -1))
            & jnp.all(split_kind_valid, axis=(-2, -1))
            & jnp.all(numeric_threshold_valid, axis=(-2, -1))
            & jnp.all(categorical_metadata_valid, axis=(-2, -1))
            & jnp.all(leaf_value_valid, axis=(-2, -1))
            & roots_valid
            & score_valid
        )
        return TreeStructureDiagnostics(
            valid=valid,
            used_trees=jnp.sum(self.tree_mask, axis=-1),
            tree_capacity=self.tree_capacity,
            used_nodes=jnp.sum(self.node_mask, axis=(-2, -1)),
            node_capacity=self.tree_capacity * self.node_capacity,
            used_leaves=jnp.sum(self.node_mask & self.leaf_mask, axis=(-2, -1)),
            maximum_depth_bound=self.max_steps,
            capacity_exhausted=self.capacity_exhausted,
        )

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return apply_objective(self.predict_raw(x), self.objective_transform)


__all__ = [
    "EnsembleAggregation",
    "ObjectiveTransform",
    "TreeEnsemble",
    "TreeStructureDiagnostics",
    "apply_objective",
]
