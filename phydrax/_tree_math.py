#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any
import equinox as eqx

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


def validate_inexact_tree(
    tree: PyTree[Any],
    /,
    *,
    name: str,
    real: bool = False,
) -> PyTree[Array]:
    """Validate and canonicalize a nonempty PyTree of inexact JAX arrays."""
    leaves, treedef = jax.tree.flatten(tree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one array leaf.")
    arrays = []
    for leaf in leaves:
        if not eqx.is_inexact_array(leaf):
            raise TypeError(f"Every {name} leaf must be an inexact JAX array.")
        array = jnp.asarray(leaf)
        if real and not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"Every {name} leaf must be real floating-point.")
        arrays.append(array)
    return jax.tree.unflatten(treedef, arrays)


def validate_real_inexact_tree(
    tree: PyTree[Any],
    /,
    *,
    name: str,
) -> PyTree[Array]:
    """Validate a nonempty PyTree of real floating-point JAX arrays."""
    return validate_inexact_tree(tree, name=name, real=True)


def tree_inner(left: PyTree[Any], right: PyTree[Any], /) -> Array:
    """Return the real Euclidean pairing of two congruent array PyTrees."""
    products = jax.tree.leaves(
        jax.tree.map(lambda x, y: jnp.real(jnp.vdot(x, y)), left, right)
    )
    if not products:
        raise ValueError("A numerical vector must contain array leaves.")
    total = products[0]
    for product in products[1:]:
        total = total + product
    return total


def tree_norm(vector: PyTree[Any], /) -> Array:
    """Return the Euclidean norm of an array PyTree."""
    return jnp.sqrt(jnp.maximum(tree_inner(vector, vector), 0.0))


def tree_allfinite(vector: PyTree[Any], /) -> Array:
    """Return whether every array element in a nonempty PyTree is finite."""
    values = tuple(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(vector))
    if not values:
        return jnp.asarray(False)
    finite = values[0]
    for value in values[1:]:
        finite = finite & value
    return finite


def tree_all(tree: PyTree[Any], /) -> Array:
    """Return the conjunction of all Boolean-like elements in a PyTree."""
    values = tuple(
        jnp.all(jnp.asarray(leaf, dtype=bool)) for leaf in jax.tree.leaves(tree)
    )
    if not values:
        return jnp.asarray(False)
    result = values[0]
    for value in values[1:]:
        result = result & value
    return result


def tree_negative(vector: PyTree[Any], /) -> PyTree[Array]:
    """Negate every array leaf."""
    return jax.tree.map(lambda leaf: -leaf, vector)


def tree_scale(scale: Any, vector: PyTree[Any], /) -> PyTree[Array]:
    """Scale every array leaf."""
    return jax.tree.map(lambda leaf: scale * leaf, vector)


def tree_add_scaled(
    parameters: PyTree[Any],
    direction: PyTree[Any],
    scale: Any,
    /,
) -> PyTree[Array]:
    """Return ``parameters + scale * direction`` leafwise."""
    return jax.tree.map(
        lambda parameter, tangent: parameter + scale * tangent,
        parameters,
        direction,
    )


def tree_where(
    condition: Any,
    proposed: PyTree[Any],
    current: PyTree[Any],
    /,
) -> PyTree[Array]:
    """Select between congruent PyTrees using one broadcastable condition."""
    return jax.tree.map(
        lambda new, old: jnp.where(condition, new, old),
        proposed,
        current,
    )


__all__ = [
    "validate_inexact_tree",
    "validate_real_inexact_tree",
    "tree_add_scaled",
    "tree_all",
    "tree_allfinite",
    "tree_inner",
    "tree_negative",
    "tree_norm",
    "tree_scale",
    "tree_where",
]
