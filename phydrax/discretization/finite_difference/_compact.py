#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
)
from .._tensor_support import PreparedTensorGrid


class CompactFirstDerivative(AbstractLinearOperator):
    """Fourth-order periodic compact derivative with a prepared cyclic line solve."""

    source: ArraySpace
    target: ArraySpace
    left_matrix: Array
    right_matrix: Array
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        maximum_dimension: int = 512,
    ):
        if not isinstance(grid, PreparedTensorGrid) or len(grid.shape) != 1:
            raise TypeError("Compact derivative requires a one-dimensional tensor grid.")
        nodes = np.asarray(grid.axes[0].nodes, dtype=float)
        count = int(nodes.size)
        if not grid.axes[0].periodic or count < 5:
            raise ValueError("Compact derivative requires at least five periodic points.")
        if count > int(maximum_dimension):
            raise ValueError("Compact line solve exceeds maximum_dimension budget.")
        spacing = np.diff(nodes)
        if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Compact derivative requires uniform spacing.")
        delta = float(spacing[0])
        alpha = 0.25
        left = np.eye(count)
        for index in range(count):
            left[index, (index - 1) % count] = alpha
            left[index, (index + 1) % count] = alpha
        right = np.zeros((count, count))
        coefficient = 0.75 / delta
        for index in range(count):
            right[index, (index + 1) % count] = coefficient
            right[index, (index - 1) % count] = -coefficient
        space = grid.field_space("compact_state").vector_space
        if not isinstance(space, ArraySpace):
            raise TypeError("Compact derivative requires ArraySpace.")
        self.source = space
        self.target = space
        self.properties = OperatorProperties(evidence={})
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "compact-first-derivative",
                "grid": grid.prepared_id,
                "count": count,
            }
        )
        self.left_matrix = jnp.asarray(left)
        self.right_matrix = jnp.asarray(right)
        self.grid_id = grid.prepared_id

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        return jnp.linalg.solve(self.left_matrix, self.right_matrix @ value)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        return self.right_matrix.T @ jnp.linalg.solve(self.left_matrix.T, value)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(value)))

    def _materialize(self, /) -> Array:
        return jnp.linalg.solve(self.left_matrix, self.right_matrix)


__all__ = ["CompactFirstDerivative"]
