#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil, floor, log

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..tree._representation import TreeEnsemble
from ._types import (
    PredictorConstraintCompilation,
    PredictorInputBinding,
    PredictorOutputConstraint,
)


class _Leaf:
    __slots__ = ("lower", "node", "upper")

    def __init__(self, node, lower, upper):
        self.node = int(node)
        self.lower = lower
        self.upper = upper


def _model_id(model: TreeEnsemble, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "tree-ensemble",
            "feature_layout": model.feature_schema.layout_id,
            "target": {
                "kind": model.target_schema.kind,
                "names": list(model.target_schema.names),
                "class_labels": list(model.target_schema.class_labels),
            },
            "transform": model.objective_transform,
            "aggregation": model.aggregation,
            "arrays": array_tree_fingerprint(
                (
                    model.feature_index,
                    model.threshold,
                    model.left_child,
                    model.right_child,
                    model.default_left,
                    model.split_kind,
                    model.leaf_value,
                    model.node_mask,
                    model.leaf_mask,
                    model.tree_mask,
                    model.tree_weight,
                    model.base_score,
                )
            ),
        }
    )


def _raw_threshold(
    model: TreeEnsemble,
    constraint: PredictorOutputConstraint,
    /,
) -> float:
    threshold = constraint.threshold
    if constraint.semantic == "raw" or model.objective_transform == "identity":
        return threshold
    if model.objective_transform == "sigmoid":
        if not 0.0 < threshold < 1.0:
            raise ValueError("Sigmoid prediction thresholds must lie in (0, 1).")
        return log(threshold / (1.0 - threshold))
    if model.objective_transform in ("exponential", "positive"):
        if threshold <= 0.0:
            raise ValueError("Exponential prediction thresholds must be positive.")
        return log(threshold)
    raise ValueError("Softmax tree prediction constraints are not supported.")


def _integer_split_bounds(
    coefficient: float,
    offset: float,
    threshold: float,
    split_kind: int,
    /,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ratio = (threshold - offset) / coefficient
    if split_kind == 0:
        if coefficient > 0.0:
            return (-np.inf, floor(ratio)), (floor(ratio) + 1, np.inf)
        return (ceil(ratio), np.inf), (-np.inf, ceil(ratio) - 1)
    if split_kind == 2:
        if coefficient > 0.0:
            return (-np.inf, ceil(ratio) - 1), (ceil(ratio), np.inf)
        return (floor(ratio) + 1, np.inf), (-np.inf, floor(ratio))
    raise ValueError("Only numeric tree split kinds are supported.")


def _tree_leaves(
    model: TreeEnsemble,
    binding: PredictorInputBinding,
    tree: int,
    /,
) -> list[_Leaf]:
    feature_index = np.asarray(model.feature_index[tree])
    threshold = np.asarray(model.threshold[tree])
    left_child = np.asarray(model.left_child[tree])
    right_child = np.asarray(model.right_child[tree])
    split_kind = np.asarray(model.split_kind[tree])
    node_mask = np.asarray(model.node_mask[tree])
    leaf_mask = np.asarray(model.leaf_mask[tree])
    matrix = np.asarray(binding.matrix)
    offset = np.asarray(binding.offset)
    discrete = np.asarray(binding.discrete_decisions)
    root_lower = np.asarray(binding.decision_lower)
    root_upper = np.asarray(binding.decision_upper)
    leaves: list[_Leaf] = []

    def visit(node, lower, upper, depth, path):
        if depth > model.max_steps:
            raise ValueError("Tree path exceeds the declared max_steps bound.")
        if node < 0 or node >= len(node_mask) or not node_mask[node]:
            raise ValueError("Tree child points outside the active node topology.")
        if node in path:
            raise ValueError("Tree topology contains a reachable cycle.")
        if leaf_mask[node]:
            leaves.append(_Leaf(node, lower, upper))
            return
        kind = int(split_kind[node])
        if kind == 1:
            raise ValueError("Categorical decision-tree splits are not yet supported.")
        feature = int(feature_index[node])
        if feature < 0 or feature >= binding.feature_count:
            raise ValueError("Tree split feature index exceeds the input binding.")
        row = matrix[feature]
        nonzero = np.flatnonzero(row)
        if len(nonzero) == 0:
            go_left = (
                offset[feature] < threshold[node]
                if kind == 2
                else offset[feature] <= threshold[node]
            )
            child = int(left_child[node] if go_left else right_child[node])
            visit(child, lower, upper, depth + 1, path | {node})
            return
        if len(nonzero) != 1:
            raise ValueError(
                "Exact tree compilation requires each split feature to bind one canonical decision coordinate."
            )
        variable = int(nonzero[0])
        if not discrete[variable]:
            raise ValueError(
                "Strict numeric tree splits require a discrete decision coordinate."
            )
        if root_lower[variable] != np.rint(root_lower[variable]) or root_upper[
            variable
        ] != np.rint(root_upper[variable]):
            raise ValueError("Tree-bound discrete decisions require integral bounds.")
        left_interval, right_interval = _integer_split_bounds(
            float(row[variable]),
            float(offset[feature]),
            float(threshold[node]),
            kind,
        )
        for child, interval in (
            (int(left_child[node]), left_interval),
            (int(right_child[node]), right_interval),
        ):
            child_lower = lower.copy()
            child_upper = upper.copy()
            child_lower[variable] = max(child_lower[variable], interval[0])
            child_upper[variable] = min(child_upper[variable], interval[1])
            if child_lower[variable] <= child_upper[variable]:
                visit(
                    child,
                    child_lower,
                    child_upper,
                    depth + 1,
                    path | {node},
                )

    visit(0, root_lower.copy(), root_upper.copy(), 0, set())
    if not leaves:
        raise ValueError("An active tree has no reachable leaf under decision bounds.")
    return leaves


def compile_tree_predictor_constraint(
    model: TreeEnsemble,
    binding: PredictorInputBinding,
    constraint: PredictorOutputConstraint,
    /,
) -> PredictorConstraintCompilation:
    """Compile a bounded discrete-input numeric tree ensemble."""
    if not isinstance(model, TreeEnsemble):
        raise TypeError("model must be a TreeEnsemble.")
    if not isinstance(binding, PredictorInputBinding):
        raise TypeError("binding must be a PredictorInputBinding.")
    if not isinstance(constraint, PredictorOutputConstraint):
        raise TypeError("constraint must be a PredictorOutputConstraint.")
    if model.case_shape:
        raise ValueError("Tree compilation requires an unbatched fitted ensemble.")
    if model.aggregation != "sum":
        raise ValueError("Tree compilation supports sum aggregation only.")
    if model.input_dtype != "preserve":
        raise ValueError("Tree compilation requires input_dtype='preserve'.")
    if bool(np.asarray(model.capacity_exhausted)):
        raise ValueError("Capacity-exhausted tree artifacts cannot be compiled.")
    if model.in_size != binding.feature_count:
        raise ValueError("Binding feature count does not match TreeEnsemble.in_size.")
    if model.feature_schema.layout_id != binding.feature_layout_id:
        raise ValueError("Binding feature layout does not match the TreeEnsemble schema.")
    leaf_value = np.asarray(model.leaf_value)
    output_count = leaf_value.shape[-1]
    if constraint.output_index >= output_count:
        raise ValueError("output_index exceeds the tree output dimension.")
    active_trees = np.flatnonzero(np.asarray(model.tree_mask))
    leaves_by_tree = [_tree_leaves(model, binding, int(tree)) for tree in active_trees]
    base = binding.base_variable_count
    leaf_count = sum(len(leaves) for leaves in leaves_by_tree)
    auxiliary = 1 + leaf_count
    total = base + auxiliary
    output_variable = base
    leaf_variables: list[list[int]] = []
    cursor = base + 1
    for leaves in leaves_by_tree:
        variables = list(range(cursor, cursor + len(leaves)))
        leaf_variables.append(variables)
        cursor += len(leaves)

    equality_rows = []
    equality_rhs = []
    output_row = (
        jnp.zeros((total,), dtype=model.leaf_value.dtype).at[output_variable].set(1.0)
    )
    base_score = np.asarray(model.base_score).reshape((-1,))[constraint.output_index]
    tree_weight = np.asarray(model.tree_weight)
    for tree_position, (tree, leaves, variables) in enumerate(
        zip(active_trees, leaves_by_tree, leaf_variables, strict=True)
    ):
        del tree_position
        for leaf, variable in zip(leaves, variables, strict=True):
            coefficient = (
                tree_weight[tree] * leaf_value[tree, leaf.node, constraint.output_index]
            )
            output_row = output_row.at[variable].set(-coefficient)
        selector = jnp.zeros((total,), dtype=model.leaf_value.dtype)
        selector = selector.at[jnp.asarray(variables, dtype=jnp.int32)].set(1.0)
        equality_rows.append(selector)
        equality_rhs.append(jnp.asarray(1.0, dtype=model.leaf_value.dtype))
    equality_rows.insert(0, output_row)
    equality_rhs.insert(0, jnp.asarray(base_score, dtype=model.leaf_value.dtype))

    inequality_rows = []
    inequality_rhs = []
    root_lower = np.asarray(binding.decision_lower)
    root_upper = np.asarray(binding.decision_upper)
    for leaves, variables in zip(leaves_by_tree, leaf_variables, strict=True):
        for leaf, variable in zip(leaves, variables, strict=True):
            for coordinate in range(base):
                if leaf.upper[coordinate] < root_upper[coordinate]:
                    big_m = root_upper[coordinate] - leaf.upper[coordinate]
                    row = jnp.zeros((total,), dtype=model.leaf_value.dtype)
                    row = row.at[coordinate].set(1.0)
                    row = row.at[variable].set(big_m)
                    inequality_rows.append(row)
                    inequality_rhs.append(
                        jnp.asarray(
                            leaf.upper[coordinate] + big_m,
                            dtype=model.leaf_value.dtype,
                        )
                    )
                if leaf.lower[coordinate] > root_lower[coordinate]:
                    big_m = leaf.lower[coordinate] - root_lower[coordinate]
                    row = jnp.zeros((total,), dtype=model.leaf_value.dtype)
                    row = row.at[coordinate].set(-1.0)
                    row = row.at[variable].set(big_m)
                    inequality_rows.append(row)
                    inequality_rhs.append(
                        jnp.asarray(
                            -leaf.lower[coordinate] + big_m,
                            dtype=model.leaf_value.dtype,
                        )
                    )

    threshold = _raw_threshold(model, constraint)
    threshold_row = (
        jnp.zeros((total,), dtype=model.leaf_value.dtype).at[output_variable].set(1.0)
    )
    if constraint.sense == "upper":
        inequality_rows.append(threshold_row)
        inequality_rhs.append(jnp.asarray(threshold, dtype=model.leaf_value.dtype))
    elif constraint.sense == "lower":
        inequality_rows.append(-threshold_row)
        inequality_rhs.append(jnp.asarray(-threshold, dtype=model.leaf_value.dtype))
    else:
        equality_rows.append(threshold_row)
        equality_rhs.append(jnp.asarray(threshold, dtype=model.leaf_value.dtype))

    weighted_values = []
    for tree, leaves in zip(active_trees, leaves_by_tree, strict=True):
        values = np.asarray(
            [
                tree_weight[tree] * leaf_value[tree, leaf.node, constraint.output_index]
                for leaf in leaves
            ]
        )
        weighted_values.append(values)
    output_lower = float(base_score) + sum(
        float(np.min(values)) for values in weighted_values
    )
    output_upper = float(base_score) + sum(
        float(np.max(values)) for values in weighted_values
    )
    auxiliary_lower = jnp.concatenate(
        (
            jnp.asarray([output_lower], dtype=model.leaf_value.dtype),
            jnp.zeros((leaf_count,), dtype=model.leaf_value.dtype),
        )
    )
    auxiliary_upper = jnp.concatenate(
        (
            jnp.asarray([output_upper], dtype=model.leaf_value.dtype),
            jnp.ones((leaf_count,), dtype=model.leaf_value.dtype),
        )
    )
    return PredictorConstraintCompilation(
        auxiliary_lower,
        auxiliary_upper,
        jnp.stack(equality_rows),
        jnp.stack(equality_rhs),
        (
            jnp.stack(inequality_rows)
            if inequality_rows
            else jnp.empty((0, total), dtype=model.leaf_value.dtype)
        ),
        (
            jnp.stack(inequality_rhs)
            if inequality_rhs
            else jnp.empty((0,), dtype=model.leaf_value.dtype)
        ),
        binary_indices=tuple(range(base + 1, total)),
        output_variable=output_variable,
        base_variable_count=base,
        model_id=_model_id(model),
        binding_id=binding.binding_id,
        constraint_id=constraint.constraint_id,
        guarantee="exact",
    )


__all__ = ["compile_tree_predictor_constraint"]
