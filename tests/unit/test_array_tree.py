#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_tree import ArrayPyTreeSchema


def _mixed_tree():
    return {
        "flags": jnp.asarray([True, False, True], dtype=bool),
        "indices": jnp.asarray([[1, 2], [3, 4], [5, 6]], dtype=jnp.int32),
        "values": jnp.asarray([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "zero": jnp.zeros((3, 0, 2), dtype=jnp.float16),
    }


def test_mixed_array_tree_schema_round_trips_and_records_intrinsic_metadata():
    tree = _mixed_tree()
    schema = ArrayPyTreeSchema.from_tree(tree, case_ndim=1)

    assert schema.validate(tree) == (3,)
    assert schema.leaf_paths == (
        "['flags']",
        "['indices']",
        "['values']",
        "['zero']",
    )
    assert tuple(leaf.shape for leaf in schema.leaves) == ((), (2,), (2,), (0, 2))
    assert tuple(leaf.dtype for leaf in schema.leaves) == tuple(
        np.dtype(value.dtype) for value in jax.tree_util.tree_leaves(tree)
    )

    rebuilt = schema.unflatten(schema.flatten(tree))
    assert jax.tree_util.tree_structure(rebuilt) == jax.tree_util.tree_structure(tree)
    for rebuilt_leaf, original_leaf in zip(
        jax.tree_util.tree_leaves(rebuilt),
        jax.tree_util.tree_leaves(tree),
        strict=True,
    ):
        np.testing.assert_array_equal(rebuilt_leaf, original_leaf)

    duplicate = ArrayPyTreeSchema.from_tree(tree, case_ndim=1, schema_id="mixed")
    assert duplicate.schema_id == "mixed"
    assert duplicate.content_id == schema.content_id
    assert schema.storage_bytes((3,)) == sum(
        np.asarray(value).nbytes for value in jax.tree_util.tree_leaves(tree)
    )


def test_finite_mask_and_case_selection_preserve_leaf_dtypes_and_shapes():
    tree = _mixed_tree()
    schema = ArrayPyTreeSchema.from_tree(tree, case_ndim=1)
    zeros = schema.zeros((3,))

    np.testing.assert_array_equal(
        schema.finite_mask(tree),
        jnp.asarray([True, False, True]),
    )
    selected = schema.select_cases(
        jnp.asarray([True, False, True]),
        tree,
        zeros,
    )

    for selected_leaf, original_leaf in zip(
        jax.tree_util.tree_leaves(selected),
        jax.tree_util.tree_leaves(tree),
        strict=True,
    ):
        assert selected_leaf.shape == original_leaf.shape
        assert selected_leaf.dtype == original_leaf.dtype
    np.testing.assert_array_equal(selected["flags"], jnp.asarray([True, False, True]))
    np.testing.assert_array_equal(
        selected["indices"],
        jnp.asarray([[1, 2], [0, 0], [5, 6]], dtype=jnp.int32),
    )
    np.testing.assert_allclose(
        selected["values"],
        jnp.asarray([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]),
    )
    assert selected["zero"].shape == (3, 0, 2)


def test_zero_sized_case_axes_remain_fixed_shape():
    tree = {
        "flags": jnp.zeros((0,), dtype=bool),
        "values": jnp.zeros((0, 2), dtype=jnp.float32),
    }
    schema = ArrayPyTreeSchema.from_tree(tree, case_ndim=1)

    assert schema.validate(tree) == (0,)
    assert schema.finite_mask(tree).shape == (0,)
    assert schema.zeros((0,))["values"].shape == (0, 2)
    assert schema.storage_bytes((0,)) == 0


def test_array_tree_schema_rejects_structure_shape_dtype_and_case_mismatch():
    tree = _mixed_tree()
    schema = ArrayPyTreeSchema.from_tree(tree, case_ndim=1)

    with pytest.raises(ValueError, match="treedef"):
        schema.validate(tuple(tree.values()))

    wrong_shape = dict(tree)
    wrong_shape["indices"] = jnp.zeros((3, 3), dtype=jnp.int32)
    with pytest.raises(ValueError, match="intrinsic shape"):
        schema.validate(wrong_shape)

    wrong_dtype = dict(tree)
    wrong_dtype["indices"] = tree["indices"].astype(jnp.int16)
    with pytest.raises(TypeError, match="dtype"):
        schema.validate(wrong_dtype)

    wrong_cases = dict(tree)
    wrong_cases["flags"] = jnp.zeros((2,), dtype=bool)
    with pytest.raises(ValueError, match="share the case shape"):
        schema.validate(wrong_cases)
    with pytest.raises(ValueError, match="share the case shape"):
        ArrayPyTreeSchema.from_tree(wrong_cases, case_ndim=1)

    with pytest.raises(TypeError, match="boolean dtype"):
        schema.select_cases(jnp.ones((3,), dtype=jnp.int32), tree, tree)
    with pytest.raises(ValueError, match="selector shape"):
        schema.select_cases(jnp.ones((1,), dtype=bool), tree, tree)
