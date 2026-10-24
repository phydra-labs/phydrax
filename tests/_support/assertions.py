from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import numpy as np
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, jaxprs_in_params


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


def assert_tree_values_equal(left: object, right: object) -> None:
    """Assert equal PyTree structure and exactly equal array leaves.

    Host and device arrays compare by value, shape, and dtype, and NaN equals
    NaN at the same position, so restored host checkpoints compare with live
    device states including unavailable (NaN) diagnostics.
    """
    left_leaves, left_tree = jax.tree_util.tree_flatten_with_path(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if left_tree != right_tree:
        raise AssertionError(f"PyTree structures differ: {left_tree!r} != {right_tree!r}")
    for (path, left_leaf), right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
            strict=True,
            err_msg=jax.tree_util.keystr(path),
        )


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


def float64_program_values(program: Jaxpr | ClosedJaxpr, /) -> list[str]:
    """Every float64/complex128 value in a traced program, including sub-programs."""
    found: list[str] = []

    def check(where: str, dtype: object) -> None:
        if str(dtype) in ("float64", "complex128"):
            found.append(f"{where}: {dtype}")

    def visit(inner: Jaxpr | ClosedJaxpr) -> None:
        for const in inner.consts:
            check("const", getattr(const, "dtype", None))
        for var in (*inner.all_invars, *inner.outvars):
            if isinstance(var.aval, ShapedArray):
                check("jaxpr variable", var.aval.dtype)
        for equation in inner.eqns:
            name = equation.primitive.name
            for atom in (*equation.invars, *equation.outvars):
                if isinstance(atom.aval, ShapedArray):
                    check(name, atom.aval.dtype)
            for nested in jaxprs_in_params(equation.params):
                visit(nested)

    visit(program)
    return found


def assert_no_float64_values(program: Jaxpr | ClosedJaxpr, /) -> None:
    """Assert a traced program holds no float64/complex128 value anywhere."""
    assert float64_program_values(program) == []
