#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree


ChainMethod = Literal["sequential", "vectorized"]


def _validate_chain_method(value: ChainMethod, /) -> ChainMethod:
    if value not in ("sequential", "vectorized"):
        raise ValueError("chain_method must be 'sequential' or 'vectorized'.")
    return value


def _prepare_chain_positions(
    reference: PyTree[Any],
    /,
    *,
    num_chains: int,
    initial_position: PyTree[Any] | None = None,
    initial_positions: PyTree[Any] | None = None,
) -> tuple[PyTree[Any], PyTree[Array]]:
    """Validate and construct one unbatched template plus chain-stacked positions."""
    chains = int(num_chains)
    if chains <= 0:
        raise ValueError("num_chains must be positive.")
    if initial_position is not None and initial_positions is not None:
        raise ValueError(
            "initial_position and initial_positions cannot both be supplied."
        )
    reference_structure = jax.tree_util.tree_structure(reference)
    reference_leaves = jax.tree_util.tree_leaves(reference)
    if not reference_leaves:
        raise ValueError("Initial positions must contain at least one array leaf.")

    if initial_positions is not None:
        if jax.tree_util.tree_structure(initial_positions) != reference_structure:
            raise ValueError(
                "initial_positions must have the reference initial PyTree structure."
            )
        position_leaves = jax.tree_util.tree_leaves(initial_positions)
        for position_leaf, reference_leaf in zip(
            position_leaves,
            reference_leaves,
            strict=True,
        ):
            if not eqx.is_inexact_array(position_leaf):
                raise TypeError("Every initial_positions leaf must be an inexact array.")
            expected_shape = (chains, *reference_leaf.shape)
            if position_leaf.shape != expected_shape:
                raise ValueError(
                    "Every initial_positions leaf must have shape "
                    f"{expected_shape}; received {position_leaf.shape}."
                )
        return reference, jax.tree_util.tree_map(jnp.asarray, initial_positions)

    position = reference if initial_position is None else initial_position
    if jax.tree_util.tree_structure(position) != reference_structure:
        raise ValueError("initial_position must have the reference initial PyTree structure.")
    position_leaves = jax.tree_util.tree_leaves(position)
    for position_leaf, reference_leaf in zip(
        position_leaves,
        reference_leaves,
        strict=True,
    ):
        if not eqx.is_inexact_array(position_leaf):
            raise TypeError("Every initial_position leaf must be an inexact array.")
        if position_leaf.shape != reference_leaf.shape:
            raise ValueError(
                "Every initial_position leaf must have shape "
                f"{reference_leaf.shape}; received {position_leaf.shape}."
            )
    arrays = jax.tree_util.tree_map(jnp.asarray, position)
    return arrays, jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (chains, *leaf.shape)),
        arrays,
    )


def _split_chain_keys(key: Array, num_chains: int, /) -> tuple[Array, Array]:
    root_key = jnp.asarray(key)
    return root_key, jr.split(root_key, int(num_chains))


def _stack_trees(values):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *values)


def _unstack_tree(tree, count: int):
    return tuple(
        jax.tree_util.tree_map(lambda value: value[index], tree)
        for index in range(int(count))
    )


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree))


__all__ = ["ChainMethod"]
