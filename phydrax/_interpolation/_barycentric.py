#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def _nodes(nodes: ArrayLike, name: str, /) -> Array:
    raw = jnp.asarray(nodes)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if raw.ndim != 1 or raw.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty rank-one array.")
    values = raw.astype(jnp.result_type(raw.dtype, jnp.float64))
    duplicate = jnp.any(jnp.diff(jnp.sort(values)) == 0.0)
    return eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | duplicate,
        f"{name} must contain finite distinct nodes.",
    )


def _weights(weights: ArrayLike, nodes: Array, name: str, /) -> Array:
    raw = jnp.asarray(weights)
    if jnp.issubdtype(raw.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if raw.shape != nodes.shape:
        raise ValueError(f"{name} must match the nodes.")
    values = raw.astype(jnp.result_type(raw.dtype, nodes.dtype))
    return eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | jnp.any(values == 0.0),
        f"{name} must contain finite nonzero values.",
    )


def _ratio_basis(x: Array, nodes: Array, weights: Array, /) -> Array:
    differences = x - nodes
    raw = weights / differences
    return raw / jnp.sum(raw)


def _product_basis(x: Array, nodes: Array, weights: Array, /) -> Array:
    differences = x - nodes
    count = nodes.shape[0]
    factors = jnp.broadcast_to(differences, (count, count))
    factors = jnp.where(jnp.eye(count, dtype=jnp.bool_), 1.0, factors)
    raw = weights * jnp.prod(factors, axis=1)
    return raw / jnp.sum(raw)


def barycentric_basis(
    x: Array,
    nodes: Array,
    weights: Array,
    /,
) -> Array:
    """Evaluate a stable one-dimensional Lagrange basis."""
    x_raw = jnp.asarray(x)
    if x_raw.ndim != 0:
        raise ValueError("Barycentric evaluation points must be scalar.")
    if jnp.issubdtype(x_raw.dtype, jnp.complexfloating):
        raise TypeError("Barycentric evaluation points must be real-valued.")
    nodes_ = _nodes(nodes, "Barycentric nodes")
    weights_ = _weights(weights, nodes_, "Barycentric weights")
    x_ = x_raw.astype(jnp.result_type(x_raw.dtype, nodes_.dtype))
    if nodes_.shape[0] == 1:
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
    nodes_ = _nodes(nodes, "Barycentric differentiation nodes")
    dtype = nodes_.dtype
    count = nodes_.shape[0]
    if count == 1:
        if weights is not None:
            _weights(weights, nodes_, "Barycentric differentiation weights")
        return jnp.zeros((1, 1), dtype=dtype)
    differences = nodes_[:, None] - nodes_[None, :]
    safe_differences = differences + jnp.eye(count, dtype=dtype)
    if weights is None:
        weights_ = jnp.reciprocal(jnp.prod(safe_differences, axis=1))
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_)) | jnp.any(weights_ == 0.0),
            "Barycentric differentiation weights are not finite and nonzero.",
        )
    else:
        weights_ = _weights(weights, nodes_, "Barycentric differentiation weights")
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
    nodes_ = jnp.asarray(nodes)
    basis = barycentric_basis(x, nodes_, weights)
    values_ = jnp.asarray(values)
    if values_.ndim < 1 or values_.shape[0] != nodes_.shape[0]:
        raise ValueError("Barycentric values must have one leading entry per node.")
    return jnp.tensordot(basis, values_, axes=((0,), (0,)))


__all__ = [
    "barycentric_basis",
    "barycentric_differentiation_matrix",
    "barycentric_interpolate",
]
