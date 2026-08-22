#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def _ratio_basis(x: Array, nodes: Array, weights: Array, /) -> Array:
    differences = x - nodes
    raw = weights / differences
    return raw / jnp.sum(raw)


def _product_basis(x: Array, nodes: Array, weights: Array, /) -> Array:
    differences = x - nodes
    count = int(nodes.shape[0])
    factors = jnp.broadcast_to(differences, (count, count))
    factors = jnp.where(jnp.eye(count, dtype=bool), 1.0, factors)
    raw = weights * jnp.prod(factors, axis=1)
    return raw / jnp.sum(raw)


def barycentric_basis(
    x: Array,
    nodes: Array,
    weights: Array,
    /,
) -> Array:
    """Evaluate a stable one-dimensional Lagrange basis."""
    x_ = jnp.asarray(x).reshape(())
    nodes_ = jnp.asarray(nodes)
    weights_ = jnp.asarray(weights)
    if nodes_.ndim != 1 or weights_.shape != nodes_.shape:
        raise ValueError(
            "Barycentric nodes and weights must be matching rank-one arrays."
        )
    if int(nodes_.shape[0]) == 1:
        return jnp.ones((1,), dtype=jnp.result_type(x_, nodes_, weights_))
    distance = jnp.min(jnp.abs(x_ - nodes_))
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(nodes_)))
    tolerance = jnp.sqrt(jnp.finfo(nodes_.dtype).eps) * scale
    return jax.lax.cond(
        distance <= tolerance,
        lambda _: _product_basis(x_, nodes_, weights_),
        lambda _: _ratio_basis(x_, nodes_, weights_),
        operand=None,
    )


def barycentric_differentiation_matrix(
    nodes: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
) -> Array:
    """Return the first-derivative matrix of a global nodal interpolant."""
    nodes_ = jnp.asarray(nodes)
    if nodes_.ndim != 1 or not int(nodes_.size):
        raise ValueError("Barycentric differentiation nodes must be a nonempty vector.")
    dtype = jnp.result_type(nodes_, float)
    nodes_ = nodes_.astype(dtype)
    count = int(nodes_.shape[0])
    if count == 1:
        return jnp.zeros((1, 1), dtype=dtype)
    differences = nodes_[:, None] - nodes_[None, :]
    safe_differences = differences + jnp.eye(count, dtype=dtype)
    if weights is None:
        weights_ = jnp.reciprocal(jnp.prod(safe_differences, axis=1))
    else:
        weights_ = jnp.asarray(weights, dtype=dtype)
        if weights_.shape != nodes_.shape:
            raise ValueError("Barycentric differentiation weights must match nodes.")
    matrix = (weights_[None, :] / weights_[:, None]) / safe_differences
    matrix = matrix - jnp.diag(jnp.diag(matrix))
    return matrix.at[jnp.arange(count), jnp.arange(count)].set(-jnp.sum(matrix, axis=1))


def barycentric_interpolate(
    x: Array,
    nodes: Array,
    weights: Array,
    values: Array,
    /,
) -> Array:
    """Interpolate values whose leading axis corresponds to one node sequence."""
    basis = barycentric_basis(x, nodes, weights)
    values_ = jnp.asarray(values)
    if values_.ndim < 1 or int(values_.shape[0]) != int(nodes.shape[0]):
        raise ValueError("Barycentric values must have one leading entry per node.")
    return jnp.tensordot(basis, values_, axes=((0,), (0,)))


__all__ = [
    "barycentric_basis",
    "barycentric_differentiation_matrix",
    "barycentric_interpolate",
]
