#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import FeatureSchema, TargetSchema
from phydrax.ml.tree import apply_objective, TreeEnsemble


def _stump(
    *,
    split_kind=0,
    threshold=0.0,
    categories=(0.0,),
    default_left=True,
    left_value=-1.0,
    right_value=1.0,
    tree_weight=1.0,
    base_score=0.0,
    objective_transform="identity",
):
    category_values = (
        jnp.zeros((1, 3, len(categories))).at[0, 0].set(jnp.asarray(categories))
    )
    category_mask = jnp.zeros_like(category_values, dtype=bool).at[0, 0].set(True)
    return TreeEnsemble(
        feature_index=jnp.array([[0, -1, -1]]),
        threshold=jnp.array([[threshold, 0.0, 0.0]]),
        left_child=jnp.array([[1, -1, -1]]),
        right_child=jnp.array([[2, -1, -1]]),
        default_left=jnp.array([[default_left, False, False]]),
        split_kind=jnp.array([[split_kind, 0, 0]]),
        category_values=category_values,
        category_mask=category_mask,
        leaf_value=jnp.array([[[0.0], [left_value], [right_value]]]),
        node_mask=jnp.ones((1, 3), dtype=bool),
        leaf_mask=jnp.array([[False, True, True]]),
        tree_mask=jnp.array([True]),
        tree_weight=jnp.array([tree_weight]),
        base_score=jnp.array([base_score]),
        feature_schema=FeatureSchema(("x",)),
        target_schema=TargetSchema("continuous", names=("y",)),
        objective_transform=objective_transform,
        out_size="scalar",
        max_steps=2,
    )


def test_numeric_traversal_has_stable_threshold_tie_paths_and_tree_outputs():
    model = _stump(threshold=2.0, left_value=10.0, right_value=20.0, base_score=3.0)
    points = jnp.array([[1.0], [2.0], [3.0]])

    assert jnp.array_equal(model(points), jnp.array([13.0, 13.0, 23.0]))
    assert jnp.array_equal(model.predict_raw(points), model(points))
    assert jnp.array_equal(
        model.predict_trees(points)[..., 0], jnp.array([10.0, 10.0, 20.0])
    )
    assert jnp.array_equal(model.predict_leaf(points), jnp.array([[1], [1], [2]]))
    assert jnp.array_equal(
        model.decision_path(points),
        jnp.array(
            [
                [[True, True, False]],
                [[True, True, False]],
                [[True, False, True]],
            ]
        ),
    )


def test_categorical_membership_and_missing_default_direction_are_distinct():
    left_default = _stump(
        split_kind=1,
        categories=(2.0, 4.0),
        default_left=True,
        left_value=7.0,
        right_value=-3.0,
    )
    right_default = _stump(
        split_kind=1,
        categories=(2.0, 4.0),
        default_left=False,
        left_value=7.0,
        right_value=-3.0,
    )
    points = jnp.array([[2.0], [3.0], [4.0], [jnp.nan], [jnp.inf]])

    assert jnp.array_equal(left_default(points), jnp.array([7.0, -3.0, 7.0, 7.0, 7.0]))
    assert jnp.array_equal(right_default(points), jnp.array([7.0, -3.0, 7.0, -3.0, -3.0]))


def test_tree_objectives_labels_weighted_aggregation_and_inactive_trees():
    binary = _stump(
        threshold=0.0,
        left_value=-2.0,
        right_value=2.0,
        objective_transform="sigmoid",
    )
    points = jnp.array([[-1.0], [1.0]])
    assert jnp.allclose(binary(points), jax.nn.sigmoid(jnp.array([-2.0, 2.0])))
    assert jnp.array_equal(binary.predict_labels(points), jnp.array([0, 1]))
    assert jnp.allclose(apply_objective(jnp.array([[-1.0, 1.0]]), "softmax").sum(-1), 1.0)
    assert jnp.all(apply_objective(jnp.array([-2.0, 2.0]), "positive") > 0.0)
    assert jnp.allclose(
        apply_objective(jnp.array([0.0, 1.0]), "exponential"),
        jnp.exp(jnp.array([0.0, 1.0])),
    )
    with pytest.raises(ValueError, match="Unsupported tree objective"):
        apply_objective(jnp.array([0.0]), "unknown")

    median = TreeEnsemble(
        feature_index=jnp.full((3, 1), -1),
        threshold=jnp.zeros((3, 1)),
        left_child=jnp.full((3, 1), -1),
        right_child=jnp.full((3, 1), -1),
        default_left=jnp.zeros((3, 1), dtype=bool),
        leaf_value=jnp.array([[[1.0]], [[5.0]], [[100.0]]]),
        node_mask=jnp.ones((3, 1), dtype=bool),
        leaf_mask=jnp.ones((3, 1), dtype=bool),
        tree_mask=jnp.array([True, True, False]),
        tree_weight=jnp.array([1.0, 3.0, 1000.0]),
        base_score=jnp.array([2.0]),
        feature_schema=FeatureSchema(("x",)),
        aggregation="weighted_median",
        max_steps=1,
    )
    assert median(jnp.array([[0.0], [9.0]])).tolist() == [7.0, 7.0]


def test_case_dependent_trees_preserve_case_and_point_axes_under_jit():
    common = dict(
        feature_index=jnp.full((2, 1, 1), -1),
        threshold=jnp.zeros((2, 1, 1)),
        left_child=jnp.full((2, 1, 1), -1),
        right_child=jnp.full((2, 1, 1), -1),
        default_left=jnp.zeros((2, 1, 1), dtype=bool),
        leaf_value=jnp.array([[[[1.0]]], [[[4.0]]]]),
        node_mask=jnp.ones((2, 1, 1), dtype=bool),
        leaf_mask=jnp.ones((2, 1, 1), dtype=bool),
        tree_mask=jnp.ones((2, 1), dtype=bool),
        base_score=jnp.array([[0.0], [10.0]]),
        feature_schema=FeatureSchema(("x",)),
        case_shape=(2,),
        max_steps=1,
    )
    model = TreeEnsemble(**common)
    points = jnp.zeros((2, 3, 1))

    expected = jnp.array([[1.0, 1.0, 1.0], [14.0, 14.0, 14.0]])
    assert jnp.array_equal(jax.jit(model)(points), expected)
    with pytest.raises(ValueError, match="beginning with case_shape"):
        model(jnp.zeros((3, 1)))


def test_case_independent_tree_is_vmappable_and_has_declared_hard_gradients():
    model = _stump(threshold=0.0, left_value=-2.0, right_value=3.0)
    points = jnp.array([[-2.0], [2.0]])
    assert jnp.array_equal(jax.vmap(model)(points), jnp.array([-2.0, 3.0]))
    assert jnp.array_equal(
        jax.grad(lambda values: jnp.sum(model(values)))(points), jnp.zeros_like(points)
    )

    def prediction_from_leaves(leaves):
        parameterized = _stump(left_value=leaves[0], right_value=leaves[1])
        return jnp.sum(parameterized(points))

    leaf_gradient = jax.grad(prediction_from_leaves)(jnp.array([-2.0, 3.0]))
    assert jnp.array_equal(leaf_gradient, jnp.ones((2,)))


def test_invalid_structure_complex_inputs_and_bounded_nonconvergence_fail_closed():
    with pytest.raises(TypeError, match="complex"):
        _stump()(jnp.array([[1.0 + 2.0j]]))
    with pytest.raises(ValueError, match="final feature axis"):
        _stump()(jnp.zeros((3, 2)))

    invalid_child = _stump()
    invalid_child = TreeEnsemble(
        feature_index=invalid_child.feature_index,
        threshold=invalid_child.threshold,
        left_child=jnp.array([[9, -1, -1]]),
        right_child=invalid_child.right_child,
        default_left=invalid_child.default_left,
        leaf_value=invalid_child.leaf_value,
        node_mask=invalid_child.node_mask,
        leaf_mask=invalid_child.leaf_mask,
        tree_mask=invalid_child.tree_mask,
        base_score=invalid_child.base_score,
        feature_schema=invalid_child.feature_schema,
        max_steps=2,
    )
    assert not invalid_child.structure_diagnostics().valid
    with pytest.raises(Exception, match="invalid active children"):
        invalid_child(jnp.array([[0.0]]))

    cycle = TreeEnsemble(
        feature_index=jnp.array([[0]]),
        threshold=jnp.array([[0.0]]),
        left_child=jnp.array([[0]]),
        right_child=jnp.array([[0]]),
        default_left=jnp.array([[True]]),
        leaf_value=jnp.zeros((1, 1, 1)),
        node_mask=jnp.ones((1, 1), dtype=bool),
        leaf_mask=jnp.zeros((1, 1), dtype=bool),
        tree_mask=jnp.ones((1,), dtype=bool),
        base_score=jnp.zeros((1,)),
        feature_schema=FeatureSchema(("x",)),
        max_steps=1,
    )
    with pytest.raises(Exception, match="exhausted its bound"):
        cycle(jnp.array([[0.0]]))
