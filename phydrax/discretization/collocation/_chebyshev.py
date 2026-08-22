#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def chebyshev_lobatto_matrices(
    count: int,
    lower: float = -1.0,
    upper: float = 1.0,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ascending Chebyshev–Gauss–Lobatto nodes and first/second derivatives."""
    size = int(count)
    lower_ = float(lower)
    upper_ = float(upper)
    if size < 3 or not np.isfinite(lower_) or not np.isfinite(upper_) or upper_ <= lower_:
        raise ValueError(
            "Chebyshev grid requires count >= 3 and increasing finite bounds."
        )
    index = np.arange(size)
    descending = np.cos(np.pi * index / (size - 1))
    c = np.ones((size,))
    c[[0, -1]] = 2.0
    c = c * (-1.0) ** index
    difference = descending[:, None] - descending[None, :]
    derivative = (c[:, None] / c[None, :]) / (difference + np.eye(size))
    derivative = derivative - np.diag(np.sum(derivative, axis=1))
    permutation = np.arange(size - 1, -1, -1)
    derivative = derivative[np.ix_(permutation, permutation)]
    reference_nodes = descending[permutation]
    scale = 2.0 / (upper_ - lower_)
    nodes = lower_ + 0.5 * (reference_nodes + 1.0) * (upper_ - lower_)
    first = scale * derivative
    second = first @ first
    return nodes, first, second


class ChebyshevCollocation(StrictModule, NonTrainableState):
    """Budgeted dense one-dimensional polynomial collocation and Dirichlet solve."""

    nodes: Array
    first_derivative: Array
    second_derivative: Array
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)

    def __init__(
        self,
        count: int,
        /,
        *,
        lower: float = -1.0,
        upper: float = 1.0,
        maximum_dimension: int = 256,
    ):
        maximum = int(maximum_dimension)
        if int(count) > maximum or maximum < 3:
            raise ValueError("Chebyshev collocation exceeds maximum_dimension budget.")
        nodes, first, second = chebyshev_lobatto_matrices(count, lower, upper)
        self.nodes = jnp.asarray(nodes)
        self.first_derivative = jnp.asarray(first)
        self.second_derivative = jnp.asarray(second)
        self.lower = float(lower)
        self.upper = float(upper)
        self.maximum_dimension = maximum
        self.discretization_id = canonical_fingerprint(
            {
                "kind": "chebyshev-collocation",
                "nodes": array_tree_fingerprint(nodes),
                "bounds": [float(lower), float(upper)],
            }
        )

    @property
    def count(self) -> int:
        return int(self.nodes.size)

    def derivative(self, values: ArrayLike, order: int = 1, /) -> Array:
        array = jnp.asarray(values)
        if array.shape != (self.count,):
            raise ValueError("Chebyshev values must match node count.")
        if int(order) == 1:
            return self.first_derivative @ array
        if int(order) == 2:
            return self.second_derivative @ array
        raise ValueError("Initial Chebyshev derivative supports order one or two.")

    def solve_helmholtz_dirichlet(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        diagonal_shift: ArrayLike = 0.0,
        lower_value: ArrayLike = 0.0,
        upper_value: ArrayLike = 0.0,
    ) -> Array:
        rhs = jnp.asarray(right_hand_side)
        shift = jnp.asarray(diagonal_shift)
        if rhs.shape != (self.count,) or shift.shape != ():
            raise ValueError("Helmholtz RHS/count or scalar shift is invalid.")
        operator = self.second_derivative - shift * jnp.eye(
            self.count,
            dtype=self.second_derivative.dtype,
        )
        operator = operator.at[0].set(jnp.eye(self.count)[0])
        operator = operator.at[-1].set(jnp.eye(self.count)[-1])
        constrained_rhs = (
            rhs.at[0].set(jnp.asarray(lower_value)).at[-1].set(jnp.asarray(upper_value))
        )
        return jnp.linalg.solve(operator, constrained_rhs)


__all__ = ["ChebyshevCollocation", "chebyshev_lobatto_matrices"]
