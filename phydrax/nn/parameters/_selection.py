#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule


class ParameterSubspace(StrictModule):
    """Explicit selected array leaves and frozen complement of a model PyTree."""

    initial: PyTree[Any]
    frozen: PyTree[Any]
    leaf_paths: tuple[str, ...] = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    leaf_dtypes: tuple[str, ...] = eqx.field(static=True)
    total_dimension: int = eqx.field(static=True)

    def __init__(self, tree: PyTree[Any], filter_spec: PyTree[bool] | Callable):
        selected, frozen = eqx.partition(tree, filter_spec)
        path_leaves = jax.tree_util.tree_flatten_with_path(selected)[0]
        selected_paths: list[str] = []
        selected_shapes: list[tuple[int, ...]] = []
        for path, leaf in path_leaves:
            if not eqx.is_inexact_array(leaf):
                raise TypeError(
                    "ParameterSubspace may select only inexact JAX array leaves."
                )
            selected_paths.append(jax.tree_util.keystr(path))
            selected_shapes.append(tuple(int(size) for size in leaf.shape))
        if not selected_paths:
            raise ValueError("ParameterSubspace must select at least one array leaf.")
        self.initial = selected
        self.frozen = frozen
        self.leaf_paths = tuple(selected_paths)
        self.leaf_dtypes = tuple(
            jnp.dtype(leaf.dtype).str for leaf in jax.tree_util.tree_leaves(selected)
        )
        self.leaf_shapes = tuple(selected_shapes)
        self.total_dimension = sum(
            int(jnp.size(leaf)) for leaf in jax.tree_util.tree_leaves(selected)
        )

    @staticmethod
    def array_leaf_paths(tree: PyTree[Any], /) -> tuple[str, ...]:
        """List selectable inexact-array paths in deterministic PyTree order."""
        return tuple(
            jax.tree_util.keystr(path)
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
            if eqx.is_inexact_array(leaf)
        )

    @classmethod
    def from_leaf_paths(
        cls,
        tree: PyTree[Any],
        leaf_paths: Sequence[str],
        /,
    ) -> ParameterSubspace:
        """Select exact named array leaves, rejecting missing or duplicate paths."""
        requested = tuple(str(path) for path in leaf_paths)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("leaf_paths must contain distinct named paths.")
        available = cls.array_leaf_paths(tree)
        missing = tuple(path for path in requested if path not in available)
        if missing:
            raise ValueError(f"Unknown parameter leaf paths: {missing!r}.")
        selected = frozenset(requested)
        filter_spec = jax.tree_util.tree_map_with_path(
            lambda path, _: jax.tree_util.keystr(path) in selected,
            tree,
        )
        return cls(tree, filter_spec)

    @classmethod
    def from_subtree_paths(
        cls,
        tree: PyTree[Any],
        subtree_paths: Sequence[str],
        /,
    ) -> ParameterSubspace:
        """Select all inexact-array leaves below explicit disjoint PyTree paths."""
        requested = tuple(str(path) for path in subtree_paths)
        if (
            not requested
            or any(not path for path in requested)
            or len(set(requested)) != len(requested)
        ):
            raise ValueError(
                "subtree_paths must contain distinct, non-empty named paths."
            )
        available = cls.array_leaf_paths(tree)

        def below(path: str, subtree_path: str, /) -> bool:
            return (
                path == subtree_path
                or path.startswith(f"{subtree_path}.")
                or path.startswith(f"{subtree_path}[")
            )

        matches = tuple(
            tuple(path for path in available if below(path, subtree_path))
            for subtree_path in requested
        )
        missing = tuple(
            subtree_path
            for subtree_path, matched in zip(requested, matches, strict=True)
            if not matched
        )
        if missing:
            raise ValueError(f"Unknown parameter subtree paths: {missing!r}.")
        selected_count = sum(len(matched) for matched in matches)
        selected = frozenset(path for matched in matches for path in matched)
        if len(selected) != selected_count:
            raise ValueError("subtree_paths must select disjoint parameter subtrees.")
        return cls.from_leaf_paths(
            tree,
            [path for path in available if path in selected],
        )

    def pack(self, selected: PyTree[Any] | None = None, /) -> Array:
        """Flatten a selected parameter position in deterministic leaf order."""
        position = self.initial if selected is None else selected
        if jax.tree_util.tree_structure(position) != jax.tree_util.tree_structure(
            self.initial
        ):
            raise ValueError("Selected position has incompatible PyTree structure.")
        leaves = jax.tree_util.tree_leaves(position)
        if len(leaves) != len(self.leaf_shapes):
            raise ValueError("Selected position has an incompatible array-leaf count.")
        for leaf, shape, dtype in zip(
            leaves, self.leaf_shapes, self.leaf_dtypes, strict=True
        ):
            if (
                not eqx.is_inexact_array(leaf)
                or tuple(leaf.shape) != shape
                or jnp.dtype(leaf.dtype).str != dtype
            ):
                raise ValueError(
                    "Selected parameter leaves must preserve shape and exact dtype."
                )
        dtype = jnp.result_type(*(leaf.dtype for leaf in leaves))
        return jnp.concatenate(
            tuple(jnp.asarray(leaf, dtype=dtype).reshape(-1) for leaf in leaves),
            axis=0,
        )

    def unpack(self, vector: Array, /) -> PyTree[Any]:
        """Restore a flat vector to the selected parameter PyTree."""
        values = jnp.asarray(vector)
        if values.ndim != 1 or int(values.shape[0]) != self.total_dimension:
            raise ValueError(
                f"vector must have shape ({self.total_dimension},); got {values.shape}."
            )
        initial_leaves, structure = jax.tree_util.tree_flatten(self.initial)
        restored = []
        start = 0
        for leaf, shape in zip(initial_leaves, self.leaf_shapes, strict=True):
            size = int(jnp.size(leaf))
            restored.append(
                values[start : start + size].astype(leaf.dtype).reshape(shape)
            )
            start += size
        return jax.tree_util.tree_unflatten(structure, restored)

    def reconstruct_vector(self, vector: Array, /) -> PyTree[Any]:
        """Reconstruct a complete model directly from packed selected parameters."""
        return self.reconstruct(self.unpack(vector))

    def reconstruct(self, selected: PyTree[Any], /) -> PyTree[Any]:
        """Recombine a selected position with the frozen complement."""
        if jax.tree_util.tree_structure(selected) != jax.tree_util.tree_structure(
            self.initial
        ):
            raise ValueError("Selected position has incompatible PyTree structure.")
        return eqx.combine(selected, self.frozen)


__all__ = ["ParameterSubspace"]
