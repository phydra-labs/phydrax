#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, PyTree

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule


_ARRAY_TYPES = (jax.Array, jax_core.Tracer, np.ndarray, np.generic)
_NUMERIC_KINDS = frozenset("biufc")


def _nonnegative_integer(value: Any, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value_ = int(value)
    if value_ < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value_


def _array_metadata(value: Any, path: str, /) -> tuple[tuple[int, ...], np.dtype]:
    if not isinstance(value, _ARRAY_TYPES):
        raise TypeError(f"Array PyTree leaf {path} must be an array.")
    shape = tuple(int(size) for size in value.shape)
    dtype = np.dtype(value.dtype)
    if dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"Array PyTree leaf {path} has unsupported dtype {dtype}; "
            "numeric and boolean dtypes are required."
        )
    return shape, dtype


def _case_shape(value: Sequence[int], case_ndim: int, /) -> tuple[int, ...]:
    shape = tuple(_nonnegative_integer(size, "case_shape dimension") for size in value)
    if len(shape) != case_ndim:
        raise ValueError(f"case_shape must contain exactly {case_ndim} dimensions.")
    return shape


def _leaf_path(path: tuple[Any, ...], /) -> str:
    return jax.tree_util.keystr(path) or "<root>"


class ArrayLeafSchema(StrictModule):
    """Stable path and intrinsic array metadata for one PyTree leaf."""

    path: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(self, path: str, shape: Sequence[int], dtype: Any, /):
        path_ = str(path)
        shape_ = tuple(
            _nonnegative_integer(size, "Array leaf dimension") for size in shape
        )
        dtype_ = np.dtype(dtype)
        if not path_:
            raise ValueError("Array leaf paths must be non-empty.")
        if dtype_.kind not in _NUMERIC_KINDS:
            raise TypeError("Array leaf dtypes must be numeric or boolean.")
        storage = prod(shape_) * dtype_.itemsize
        self.path = path_
        self.shape = shape_
        self.dtype = dtype_
        self.storage_bytes = storage
        self.content_id = canonical_fingerprint(
            {
                "kind": "array-leaf-schema",
                "path": path_,
                "shape": list(shape_),
                "dtype": dtype_.str,
            }
        )


class ArrayPyTreeSchema(StrictModule):
    """Exact PyTree structure and per-leaf intrinsic array contracts.

    ``case_ndim`` leading axes are case axes shared by every leaf. The remaining
    axes are intrinsic to each leaf and are recorded in :attr:`leaves`.
    """

    treedef: jax.tree_util.PyTreeDef = eqx.field(static=True)
    leaves: tuple[ArrayLeafSchema, ...] = eqx.field(static=True)
    case_ndim: int = eqx.field(static=True)
    intrinsic_storage_bytes: int = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        treedef: jax.tree_util.PyTreeDef,
        leaves: Sequence[ArrayLeafSchema],
        case_ndim: int,
        /,
        *,
        schema_id: str | None = None,
    ):
        if not isinstance(treedef, jax.tree_util.PyTreeDef):
            raise TypeError("treedef must be a JAX PyTreeDef.")
        leaves_ = tuple(leaves)
        if any(not isinstance(leaf, ArrayLeafSchema) for leaf in leaves_):
            raise TypeError("leaves must contain only ArrayLeafSchema values.")
        if treedef.num_leaves != len(leaves_):
            raise ValueError("treedef and leaves must have the same leaf count.")
        paths = tuple(leaf.path for leaf in leaves_)
        if len(set(paths)) != len(paths):
            raise ValueError("Array leaf paths must be unique.")
        case_ndim_ = _nonnegative_integer(case_ndim, "case_ndim")
        content_id = canonical_fingerprint(
            {
                "kind": "array-pytree-schema",
                "treedef": str(treedef),
                "case_ndim": case_ndim_,
                "leaves": [leaf.content_id for leaf in leaves_],
            }
        )
        if schema_id is None:
            schema_id_ = content_id
        else:
            if not isinstance(schema_id, str) or not schema_id.strip():
                raise ValueError("schema_id must be a non-empty string or None.")
            schema_id_ = schema_id.strip()
        self.treedef = treedef
        self.leaves = leaves_
        self.case_ndim = case_ndim_
        self.intrinsic_storage_bytes = sum(leaf.storage_bytes for leaf in leaves_)
        self.content_id = content_id
        self.schema_id = schema_id_

    @classmethod
    def from_tree(
        cls,
        tree: PyTree[Any],
        /,
        *,
        case_ndim: int,
        schema_id: str | None = None,
    ) -> ArrayPyTreeSchema:
        """Infer intrinsic metadata after validating common leading case axes."""
        case_ndim_ = _nonnegative_integer(case_ndim, "case_ndim")
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
        case_shape: tuple[int, ...] | None = None
        schemas: list[ArrayLeafSchema] = []
        for path, value in path_leaves:
            path_ = _leaf_path(path)
            shape, dtype = _array_metadata(value, path_)
            if len(shape) < case_ndim_:
                raise ValueError(
                    f"Array PyTree leaf {path_} has fewer axes than case_ndim."
                )
            leading = shape[:case_ndim_]
            if case_shape is None:
                case_shape = leading
            elif leading != case_shape:
                raise ValueError("All Array PyTree leaves must share the case shape.")
            schemas.append(ArrayLeafSchema(path_, shape[case_ndim_:], dtype))
        if not schemas and case_ndim_:
            raise ValueError("A nonzero case_ndim requires at least one array leaf.")
        return cls(treedef, schemas, case_ndim_, schema_id=schema_id)

    @property
    def leaf_paths(self) -> tuple[str, ...]:
        return tuple(leaf.path for leaf in self.leaves)

    def validate(self, tree: PyTree[Any], /) -> tuple[int, ...]:
        """Validate exact structure, paths, intrinsic shapes, and dtypes."""
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
        if treedef != self.treedef:
            raise ValueError("Array PyTree treedef does not match the schema.")
        observed_paths = tuple(_leaf_path(path) for path, _ in path_leaves)
        if observed_paths != self.leaf_paths:
            raise ValueError("Array PyTree leaf paths do not match the schema.")
        case_shape: tuple[int, ...] | None = None
        for (_, value), leaf in zip(path_leaves, self.leaves, strict=True):
            shape, dtype = _array_metadata(value, leaf.path)
            if len(shape) != self.case_ndim + len(leaf.shape):
                raise ValueError(
                    f"Array PyTree leaf {leaf.path} rank does not match the schema."
                )
            leading = shape[: self.case_ndim]
            intrinsic = shape[self.case_ndim :]
            if intrinsic != leaf.shape:
                raise ValueError(
                    f"Array PyTree leaf {leaf.path} intrinsic shape does not match."
                )
            if dtype != leaf.dtype:
                raise TypeError(
                    f"Array PyTree leaf {leaf.path} dtype {dtype} does not match "
                    f"schema dtype {leaf.dtype}."
                )
            if case_shape is None:
                case_shape = leading
            elif leading != case_shape:
                raise ValueError("All Array PyTree leaves must share the case shape.")
        return () if case_shape is None else case_shape

    def flatten(self, tree: PyTree[Any], /) -> tuple[Any, ...]:
        """Return validated leaves in stable schema-path order."""
        self.validate(tree)
        return tuple(jax.tree_util.tree_leaves(tree))

    def unflatten(self, leaves: Sequence[Any], /) -> PyTree[Any]:
        """Reconstruct and validate a tree from leaves in schema-path order."""
        leaves_ = tuple(leaves)
        if len(leaves_) != len(self.leaves):
            raise ValueError("Unflattened leaf count does not match the schema.")
        tree = self.treedef.unflatten(leaves_)
        self.validate(tree)
        return tree

    def finite_mask(self, tree: PyTree[Any], /) -> Array:
        """Return one finite flag per case without coercing numeric leaf dtypes."""
        case_shape = self.validate(tree)
        finite = jnp.ones(case_shape, dtype=bool)
        for value, leaf in zip(jax.tree_util.tree_leaves(tree), self.leaves, strict=True):
            axes = tuple(range(self.case_ndim, self.case_ndim + len(leaf.shape)))
            finite = finite & jnp.all(jnp.isfinite(value), axis=axes)
        return finite

    def select_cases(
        self,
        selector: Any,
        if_true: PyTree[Any],
        if_false: PyTree[Any],
        /,
    ) -> PyTree[Any]:
        """Select complete cases between two trees while preserving every dtype."""
        true_case_shape = self.validate(if_true)
        false_case_shape = self.validate(if_false)
        if true_case_shape != false_case_shape:
            raise ValueError("Selected Array PyTrees must have the same case shape.")
        selector_ = jnp.asarray(selector)
        if np.dtype(selector_.dtype) != np.dtype(bool):
            raise TypeError("Case selectors must have boolean dtype.")
        if selector_.shape != true_case_shape:
            raise ValueError("Case selector shape must exactly match the case shape.")
        true_leaves = jax.tree_util.tree_leaves(if_true)
        false_leaves = jax.tree_util.tree_leaves(if_false)
        selected = []
        for true_value, false_value, leaf in zip(
            true_leaves, false_leaves, self.leaves, strict=True
        ):
            expanded = jnp.reshape(selector_, selector_.shape + (1,) * len(leaf.shape))
            selected.append(jnp.where(expanded, true_value, false_value))
        return self.treedef.unflatten(selected)

    def zeros(self, case_shape: Sequence[int] = (), /) -> PyTree[Array]:
        """Construct a schema-exact zero tree for the requested fixed case shape."""
        case_shape_ = _case_shape(case_shape, self.case_ndim)
        leaves = [
            jnp.zeros(case_shape_ + leaf.shape, dtype=leaf.dtype) for leaf in self.leaves
        ]
        return self.treedef.unflatten(leaves)

    def storage_bytes(self, case_shape: Sequence[int] = (), /) -> int:
        """Return exact dense storage for one schema realization."""
        case_shape_ = _case_shape(case_shape, self.case_ndim)
        case_count = prod(case_shape_) if case_shape_ else 1
        return case_count * self.intrinsic_storage_bytes


__all__ = ["ArrayLeafSchema", "ArrayPyTreeSchema"]
