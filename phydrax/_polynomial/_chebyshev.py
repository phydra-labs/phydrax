#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._quadrature_rules import clenshaw_curtis_data
from .._strict import StrictModule
from .._trainable import NonTrainableState


_DEFAULT_CONSTRUCTION_BYTES = 512 * 1024**2


class ChebyshevLobattoData(StrictModule, NonTrainableState):
    """Prepared ascending Chebyshev--Lobatto interpolation and calculus data."""

    nodes: Array
    barycentric_weights: Array
    quadrature_weights: Array
    differentiation_matrices: tuple[Array, ...]
    num_nodes: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    construction_bytes: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def differentiation_matrix(self, order: int, /) -> Array:
        """Return the prepared reference-coordinate derivative matrix."""
        if isinstance(order, bool) or not isinstance(order, Integral):
            raise TypeError("Chebyshev derivative order must be an integer.")
        order_ = int(order)
        if order_ < 1 or order_ > self.maximum_derivative_order:
            raise ValueError(
                "Chebyshev derivative order must lie within the prepared range."
            )
        return self.differentiation_matrices[order_ - 1]


def chebyshev_lobatto_data(
    num_nodes: int,
    /,
    *,
    maximum_derivative_order: int = 1,
    dtype=float,
    maximum_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
) -> ChebyshevLobattoData:
    """Prepare reference nodes, interpolation weights, quadrature, and derivatives."""
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
        raise TypeError("Chebyshev node count must be an integer.")
    count = int(num_nodes)
    if count < 2:
        raise ValueError("Chebyshev--Lobatto data requires at least two nodes.")
    if isinstance(maximum_derivative_order, bool) or not isinstance(
        maximum_derivative_order, Integral
    ):
        raise TypeError("maximum_derivative_order must be an integer.")
    derivative_order = int(maximum_derivative_order)
    if derivative_order < 0:
        raise ValueError("maximum_derivative_order must be nonnegative.")
    if (
        isinstance(maximum_construction_bytes, bool)
        or not isinstance(maximum_construction_bytes, Integral)
        or int(maximum_construction_bytes) <= 0
    ):
        raise ValueError("maximum_construction_bytes must be a positive integer.")

    dtype_ = jnp.dtype(dtype)
    itemsize = np.dtype(dtype_).itemsize
    construction_bytes = (3 * count + derivative_order * count * count) * itemsize
    if construction_bytes > int(maximum_construction_bytes):
        raise ValueError(
            "Chebyshev--Lobatto construction exceeds maximum_construction_bytes."
        )

    quadrature = clenshaw_curtis_data(count)
    nodes_host = np.asarray(quadrature.nodes, dtype=float)
    quadrature_host = np.asarray(quadrature.weights, dtype=float)
    barycentric_host = (-1.0) ** np.arange(count, dtype=float)
    barycentric_host[[0, -1]] *= 0.5

    differences = nodes_host[:, None] - nodes_host[None, :]
    safe_differences = differences + np.eye(count)
    first = (barycentric_host[None, :] / barycentric_host[:, None]) / safe_differences
    first = first - np.diag(np.diag(first))
    first[np.diag_indices(count)] = -np.sum(first, axis=1)

    matrices_host: list[np.ndarray] = []
    current = np.eye(count)
    for _ in range(derivative_order):
        current = first @ current
        matrices_host.append(np.asarray(current))
    if any(np.any(~np.isfinite(matrix)) for matrix in matrices_host):
        raise ValueError("Chebyshev differentiation matrices must be finite.")

    nodes = jnp.asarray(nodes_host, dtype=dtype_)
    barycentric_weights = jnp.asarray(barycentric_host, dtype=dtype_)
    quadrature_weights = jnp.asarray(quadrature_host, dtype=dtype_)
    differentiation_matrices = tuple(
        jnp.asarray(matrix, dtype=dtype_) for matrix in matrices_host
    )
    data_id = canonical_fingerprint(
        {
            "kind": "chebyshev-lobatto-data-v1",
            "num_nodes": count,
            "degree": count - 1,
            "maximum_derivative_order": derivative_order,
            "construction_bytes": construction_bytes,
            "data": array_tree_fingerprint(
                (
                    nodes_host,
                    barycentric_host,
                    quadrature_host,
                    tuple(matrices_host),
                )
            ),
        }
    )
    return ChebyshevLobattoData(
        nodes=nodes,
        barycentric_weights=barycentric_weights,
        quadrature_weights=quadrature_weights,
        differentiation_matrices=differentiation_matrices,
        num_nodes=count,
        degree=count - 1,
        maximum_derivative_order=derivative_order,
        construction_bytes=construction_bytes,
        data_id=data_id,
    )


__all__ = ["ChebyshevLobattoData", "chebyshev_lobatto_data"]
