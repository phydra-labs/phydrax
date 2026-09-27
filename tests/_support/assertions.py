from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import numpy as np


def _leaf_differences(
    left: object,
    right: object,
    *,
    rtol: float,
    atol: float,
) -> Sequence[str]:
    left_with_paths, left_tree = jax.tree_util.tree_flatten_with_path(left)
    right_with_paths, right_tree = jax.tree_util.tree_flatten_with_path(right)
    if left_tree != right_tree:
        return (f"PyTree structures differ: {left_tree!r} != {right_tree!r}",)

    differences: list[str] = []
    for (left_path, left_leaf), (right_path, right_leaf) in zip(
        left_with_paths,
        right_with_paths,
        strict=True,
    ):
        path = jax.tree_util.keystr(left_path)
        if left_path != right_path:
            differences.append(
                f"leaf paths differ: {path!r} != {jax.tree_util.keystr(right_path)!r}"
            )
            continue
        if type(left_leaf) is not type(right_leaf):
            differences.append(
                f"{path}: leaf types differ: {type(left_leaf)!r} != {type(right_leaf)!r}"
            )
            continue
        if eqx.is_array(left_leaf) and eqx.is_array(right_leaf):
            if left_leaf.shape != right_leaf.shape:
                differences.append(
                    f"{path}: shapes differ: {left_leaf.shape!r} != {right_leaf.shape!r}"
                )
                continue
            if left_leaf.dtype != right_leaf.dtype:
                differences.append(
                    f"{path}: dtypes differ: {left_leaf.dtype!r} != {right_leaf.dtype!r}"
                )
                continue
            left_host = np.asarray(left_leaf)
            right_host = np.asarray(right_leaf)
            if not np.allclose(
                left_host,
                right_host,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                absolute = np.abs(left_host - right_host)
                differences.append(
                    f"{path}: values differ; maximum absolute error "
                    f"{float(np.max(absolute, initial=0.0))!r}"
                )
        elif left_leaf != right_leaf:
            differences.append(f"{path}: values differ: {left_leaf!r} != {right_leaf!r}")
    return differences


def assert_tree_equal(left: object, right: object) -> None:
    """Assert exact PyTree value, structure, leaf-type, shape, and dtype equality."""
    result = eqx.tree_equal(left, right, typematch=True)
    if bool(result):
        return
    differences = _leaf_differences(left, right, rtol=0.0, atol=0.0)
    raise AssertionError("PyTrees differ:\n" + "\n".join(differences))


def assert_tree_close(
    left: object,
    right: object,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert close PyTree values with exact structure, leaf types, shapes, and dtypes."""
    result = eqx.tree_equal(left, right, typematch=True, rtol=rtol, atol=atol)
    if bool(result):
        return
    differences = _leaf_differences(left, right, rtol=rtol, atol=atol)
    raise AssertionError("PyTrees differ:\n" + "\n".join(differences))
