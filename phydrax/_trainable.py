#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
from jaxtyping import PyTree


class NonTrainableState:
    """Marker for numeric state that must remain fixed during solver training."""


def is_non_trainable_leaf(node: Any, /) -> bool:
    """Return whether a PyTree node should be kept wholly non-trainable."""
    from phydrax.domain import Domain

    return isinstance(node, (NonTrainableState, Domain))


def is_trainable_leaf(node: Any, /) -> bool:
    """Return whether a PyTree leaf should be optimized by FunctionalSolver."""
    if is_non_trainable_leaf(node):
        return False
    return bool(eqx.is_inexact_array(node))


def partition_trainable(tree: PyTree[Any], /) -> tuple[PyTree[Any], PyTree[Any]]:
    """Split a PyTree into trainable and non-trainable parts."""
    return eqx.partition(
        tree,
        is_trainable_leaf,
        is_leaf=is_non_trainable_leaf,
    )


def combine_trainable(
    trainable: PyTree[Any],
    non_trainable: PyTree[Any],
    /,
) -> PyTree[Any]:
    """Recombine trees produced by `partition_trainable`."""
    return eqx.combine(
        trainable,
        non_trainable,
        is_leaf=is_non_trainable_leaf,
    )


def place_array_leaves(tree: PyTree[Any], placement: Any, /) -> PyTree[Any]:
    """Place array leaves while retaining domains and fixed state as whole leaves."""

    arrays, fixed = eqx.partition(
        tree,
        eqx.is_array,
        is_leaf=is_non_trainable_leaf,
    )
    placed = jax.tree.map(
        lambda leaf: jax.device_put(leaf, placement) if eqx.is_array(leaf) else leaf,
        arrays,
        is_leaf=is_non_trainable_leaf,
    )
    return eqx.combine(
        placed,
        fixed,
        is_leaf=is_non_trainable_leaf,
    )


__all__ = [
    "NonTrainableState",
    "combine_trainable",
    "is_non_trainable_leaf",
    "is_trainable_leaf",
    "place_array_leaves",
    "partition_trainable",
]
