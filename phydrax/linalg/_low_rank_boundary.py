#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class LowRankBoundaryCorrectionPlan(StrictModule, NonTrainableState):
    """Prepared constrained solve using a base inverse and boundary Schur complement."""

    operator: Array
    boundary_indices: Array
    green_columns: Array
    capacitance: Array
    maximum_construction_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: ArrayLike,
        boundary_indices: ArrayLike,
        /,
        *,
        maximum_construction_bytes: int = 512 * 1024**2,
    ):
        matrix = np.asarray(operator)
        indices = np.asarray(boundary_indices, dtype=np.int32).reshape((-1,))
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("Boundary correction operator must be non-empty and square.")
        if (
            indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= matrix.shape[0])
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("boundary_indices must be unique and in bounds.")
        estimate = (
            matrix.nbytes + 2 * matrix.shape[0] * indices.size * matrix.dtype.itemsize
        )
        budget = int(maximum_construction_bytes)
        if budget <= 0 or estimate > budget:
            raise ValueError(
                f"Low-rank boundary preparation requires {estimate} bytes, budget is {budget}."
            )
        selector_transpose = np.zeros((matrix.shape[0], indices.size), dtype=matrix.dtype)
        selector_transpose[indices, np.arange(indices.size)] = 1.0
        green = np.linalg.solve(matrix, selector_transpose)
        capacitance = green[indices]
        if not np.all(np.isfinite(green)) or not np.all(np.isfinite(capacitance)):
            raise ValueError(
                "Boundary Green/capacitance preparation produced nonfinite values."
            )
        self.operator = jnp.asarray(matrix)
        self.boundary_indices = jnp.asarray(indices)
        self.green_columns = jnp.asarray(green)
        self.capacitance = jnp.asarray(capacitance)
        self.maximum_construction_bytes = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "low-rank-boundary-correction",
                "operator": array_tree_fingerprint(matrix),
                "boundary_indices": array_tree_fingerprint(indices),
                "budget": budget,
            }
        )

    def solve(
        self,
        right_hand_side: ArrayLike,
        boundary_values: ArrayLike,
        /,
    ) -> Array:
        rhs = jnp.asarray(right_hand_side)
        boundary = jnp.asarray(boundary_values)
        dimension = int(self.operator.shape[0])
        if rhs.shape != (dimension,) or boundary.shape != self.boundary_indices.shape:
            raise ValueError(
                "Boundary correction RHS or boundary values have wrong shape."
            )
        base = jnp.linalg.solve(self.operator, rhs)
        mismatch = boundary - base[self.boundary_indices]
        multipliers = jnp.linalg.solve(self.capacitance, mismatch)
        return base + self.green_columns @ multipliers


__all__ = ["LowRankBoundaryCorrectionPlan"]
