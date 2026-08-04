#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array


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


__all__ = ["barycentric_basis", "barycentric_interpolate"]
