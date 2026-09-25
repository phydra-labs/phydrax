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
from ..._trainable import _lane_tree, resolve_array_roles


def _inexact_paths(tree: PyTree[Any], /) -> frozenset[str]:
    return frozenset(
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
        if eqx.is_inexact_array(leaf)
    )


def _unselectable_error(paths: Sequence[str], /) -> ValueError:
    return ValueError(
        "ParameterSubspace cannot select FIXED or MODEL_STATE leaves (below a "
        "Domain, NonTrainableState or ExplicitFreeze node, or declared with "
        f"fixed_field or model_state_field): {tuple(paths)!r}."
    )


def _selection_mask(
    tree: PyTree[Any], filter_spec: PyTree[bool] | Callable, /
) -> list[bool]:
    """Per-leaf selection of an explicit filter spec, validated against roles.

    Boolean spec leaves are explicit declarations: every covered leaf they select
    must be a selectable inexact array. Callable spec leaves filter only the
    selectable leaves they cover.
    """
    resolution = resolve_array_roles(tree)
    selectable = frozenset(resolution.selectable_paths)
    spec_leaves, spec_treedef = jax.tree_util.tree_flatten(filter_spec)
    subtrees = spec_treedef.flatten_up_to(tree)
    mask: list[bool] = []
    unselectable: list[str] = []
    paths = iter(resolution.paths)
    for spec, subtree in zip(spec_leaves, subtrees, strict=True):
        for leaf in jax.tree_util.tree_leaves(subtree):
            path = next(paths)
            if callable(spec):
                mask.append(path in selectable and bool(spec(leaf)))
                continue
            if not isinstance(spec, bool):
                raise TypeError("filter_spec leaves must be bool or callable.")
            if spec and not eqx.is_inexact_array(leaf):
                raise TypeError(
                    "ParameterSubspace may select only inexact JAX array leaves."
                )
            if spec and path not in selectable:
                unselectable.append(path)
            mask.append(spec)
    if unselectable:
        raise _unselectable_error(unselectable)
    return mask


class ParameterSubspace(StrictModule):
    """Explicit selected array leaves and frozen complement of a model PyTree."""

    initial: PyTree[Any]
    frozen: PyTree[Any]
    leaf_paths: tuple[str, ...] = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    leaf_dtypes: tuple[str, ...] = eqx.field(static=True)
    total_dimension: int = eqx.field(static=True)
    alias_groups: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        tree: PyTree[Any],
        filter_spec: PyTree[bool] | Callable,
        *,
        alias_groups: Sequence[Sequence[str]] = (),
    ):
        mask = _selection_mask(tree, filter_spec)
        # Terminal nodes are wholly frozen: `None` in the selection and whole in
        # the complement, matching the role lanes of `partition_parameters`.
        selected = _lane_tree(tree, mask, terminals=False)
        frozen = _lane_tree(tree, (not keep for keep in mask), terminals=True)
        path_leaves = jax.tree_util.tree_flatten_with_path(selected)[0]
        selected_paths = [jax.tree_util.keystr(path) for path, _ in path_leaves]
        selected_shapes = [tuple(leaf.shape) for _, leaf in path_leaves]
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
            jnp.size(leaf) for leaf in jax.tree_util.tree_leaves(selected)
        )
        groups = tuple(tuple(str(path) for path in group) for group in alias_groups)
        available = self.array_leaf_paths(tree)
        used: set[str] = set()
        for group in groups:
            if (
                len(group) < 2
                or len(set(group)) != len(group)
                or group[0] not in self.leaf_paths
                or any(path not in available for path in group)
                or any(path in used for path in group)
            ):
                raise ValueError("Parameter alias groups are invalid or overlapping.")
            canonical = dict(
                (jax.tree_util.keystr(path), leaf)
                for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
            )[group[0]]
            for alias in group[1:]:
                value = dict(
                    (jax.tree_util.keystr(path), leaf)
                    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
                )[alias]
                if value.shape != canonical.shape or value.dtype != canonical.dtype:
                    raise ValueError(
                        "Aliased parameter leaves must match shape and dtype."
                    )
            used.update(group)
        self.alias_groups = groups

    @staticmethod
    def array_leaf_paths(tree: PyTree[Any], /) -> tuple[str, ...]:
        """List selectable inexact-array paths in deterministic PyTree order.

        Leaves below terminal nodes and leaves declared with `fixed_field` or
        `model_state_field` are not selectable.
        """
        return resolve_array_roles(tree).selectable_paths

    @classmethod
    def from_leaf_paths(
        cls,
        tree: PyTree[Any],
        leaf_paths: Sequence[str],
        /,
        alias_groups: Sequence[Sequence[str]] = (),
    ) -> ParameterSubspace:
        """Select exact named array leaves, rejecting missing or duplicate paths."""
        requested = tuple(str(path) for path in leaf_paths)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("leaf_paths must contain distinct named paths.")
        groups = tuple(tuple(str(path) for path in group) for group in alias_groups)
        alias_paths = frozenset(path for group in groups for path in group[1:])
        canonical_requested = tuple(path for path in requested if path not in alias_paths)
        resolution = resolve_array_roles(tree)
        available = frozenset(resolution.selectable_paths)
        inexact = _inexact_paths(tree)
        named = requested + tuple(path for group in groups for path in group)
        missing = tuple(path for path in named if path not in inexact)
        if missing:
            raise ValueError(f"Unknown parameter leaf paths: {missing!r}.")
        unselectable = tuple(path for path in named if path not in available)
        if unselectable:
            raise _unselectable_error(unselectable)
        selected = frozenset(canonical_requested)
        filter_spec = jax.tree_util.tree_unflatten(
            resolution.treedef, [path in selected for path in resolution.paths]
        )
        return cls(tree, filter_spec, alias_groups=groups)

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
            inexact = _inexact_paths(tree)
            unselectable = tuple(
                subtree_path
                for subtree_path in missing
                if any(below(path, subtree_path) for path in inexact)
            )
            if unselectable:
                raise _unselectable_error(unselectable)
            raise ValueError(f"Unknown parameter subtree paths: {missing!r}.")
        selected_count = sum(len(matched) for matched in matches)
        selected = frozenset(path for matched in matches for path in matched)
        if len(selected) != selected_count:
            raise ValueError("subtree_paths must select disjoint parameter subtrees.")
        return cls.from_leaf_paths(
            tree,
            [path for path in available if path in selected],
        )

    def validate_root(self, tree: PyTree[Any], /) -> None:
        """Require the exact root tree used to construct this parameter subspace."""
        equal = eqx.tree_equal(self.reconstruct(self.initial), tree)
        matched = equal if isinstance(equal, bool) else bool(jax.device_get(equal))
        if not matched:
            raise ValueError(
                "ParameterSubspace does not describe the supplied root PyTree."
            )

    def rebase(
        self,
        tree: PyTree[Any],
        /,
        *,
        exact_dtype: bool = True,
    ) -> ParameterSubspace:
        """Apply the same exact leaf paths to a compatible updated root tree."""
        rebased = type(self).from_leaf_paths(
            tree,
            self.leaf_paths,
            alias_groups=self.alias_groups,
        )
        if rebased.leaf_shapes != self.leaf_shapes:
            raise ValueError("Rebased parameter leaves changed shape.")
        if exact_dtype and rebased.leaf_dtypes != self.leaf_dtypes:
            raise ValueError("Rebased parameter leaves changed dtype.")
        return rebased

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
        if values.ndim != 1 or values.shape[0] != self.total_dimension:
            raise ValueError(
                f"vector must have shape ({self.total_dimension},); got {values.shape}."
            )
        initial_leaves, structure = jax.tree_util.tree_flatten(self.initial)
        restored = []
        start = 0
        for leaf, shape in zip(initial_leaves, self.leaf_shapes, strict=True):
            size = jnp.size(leaf)
            restored.append(
                values[start : start + size].astype(leaf.dtype).reshape(shape)
            )
            start += size
        return jax.tree_util.tree_unflatten(structure, restored)

    def reconstruct_vector(self, vector: Array, /) -> PyTree[Any]:
        """Reconstruct a complete model directly from packed selected parameters."""
        return self.reconstruct(self.unpack(vector))

    def reconstruct(self, selected: PyTree[Any], /) -> PyTree[Any]:
        """Recombine selected parameters and copy canonical values to aliases."""
        if jax.tree_util.tree_structure(selected) != jax.tree_util.tree_structure(
            self.initial
        ):
            raise ValueError("Selected position has incompatible PyTree structure.")
        combined = eqx.combine(selected, self.frozen)
        if not self.alias_groups:
            return combined
        leaves = {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(combined)[0]
        }
        alias_to_canonical = {
            alias: group[0] for group in self.alias_groups for alias in group[1:]
        }
        return jax.tree_util.tree_map_with_path(
            lambda path, value: (
                leaves[alias_to_canonical[jax.tree_util.keystr(path)]]
                if jax.tree_util.keystr(path) in alias_to_canonical
                else value
            ),
            combined,
        )


__all__ = ["ParameterSubspace"]
