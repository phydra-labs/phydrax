#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import prepare_local_block_factorization, solve_local_blocks


class LocalEliminationResult(StrictModule):
    schur: Array
    right_hand_side: Array
    interior_solution_operator: Array
    interior_load: Array
    failed: Array


class FiniteElementLocalEliminationPlan(StrictModule, NonTrainableState):
    """Static partition of cell-local coordinates into retained and private DOFs."""

    retained_dofs: Array
    eliminated_dofs: Array
    local_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_size: int,
        retained_dofs: ArrayLike,
        /,
    ):
        size = int(local_size)
        retained = np.asarray(retained_dofs, dtype=np.int32)
        if size <= 1 or retained.ndim != 1 or retained.size == 0:
            raise ValueError("Local elimination requires retained and eliminated DOFs.")
        if np.any(retained < 0) or np.any(retained >= size):
            raise ValueError("Retained local DOF indices are out of bounds.")
        if np.unique(retained).size != retained.size:
            raise ValueError("Retained local DOF indices must be unique.")
        eliminated = np.setdiff1d(
            np.arange(size, dtype=np.int32),
            retained,
            assume_unique=True,
        )
        if eliminated.size == 0:
            raise ValueError("Local elimination requires at least one private DOF.")
        self.retained_dofs = jnp.asarray(retained)
        self.eliminated_dofs = jnp.asarray(eliminated)
        self.local_size = size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-local-elimination",
                "local_size": size,
                "retained": retained.tolist(),
                "eliminated": eliminated.tolist(),
            }
        )

    def condense(
        self,
        local_matrix: ArrayLike,
        local_rhs: ArrayLike,
        /,
    ) -> LocalEliminationResult:
        matrix = jnp.asarray(local_matrix)
        rhs = jnp.asarray(local_rhs)
        if matrix.ndim != 3 or matrix.shape[1:] != (self.local_size, self.local_size):
            raise ValueError("local_matrix must have shape (cells, local, local).")
        if rhs.shape[:2] != (matrix.shape[0], self.local_size):
            raise ValueError("local_rhs must begin with shape (cells, local).")
        retained = self.retained_dofs
        eliminated = self.eliminated_dofs
        a_rr = matrix[:, retained[:, None], retained[None, :]]
        a_ri = matrix[:, retained[:, None], eliminated[None, :]]
        a_ir = matrix[:, eliminated[:, None], retained[None, :]]
        a_ii = matrix[:, eliminated[:, None], eliminated[None, :]]
        f_r = rhs[:, retained]
        f_i = rhs[:, eliminated]
        factorization = prepare_local_block_factorization(a_ii)
        interior_operator, failed_operator = solve_local_blocks(factorization, a_ir)
        interior_load, failed_load = solve_local_blocks(factorization, f_i)
        schur = a_rr - jnp.matmul(a_ri, interior_operator)
        reduced_rhs = f_r - jnp.matmul(a_ri, interior_load[..., None])[..., 0]
        return LocalEliminationResult(
            schur=schur,
            right_hand_side=reduced_rhs,
            interior_solution_operator=interior_operator,
            interior_load=interior_load,
            failed=failed_operator | failed_load,
        )

    def reconstruct(
        self,
        retained_solution: ArrayLike,
        result: LocalEliminationResult,
        /,
    ) -> Array:
        if not isinstance(result, LocalEliminationResult):
            raise TypeError("result must be LocalEliminationResult.")
        retained = jnp.asarray(retained_solution)
        if retained.shape[:2] != (
            result.interior_solution_operator.shape[0],
            self.retained_dofs.size,
        ):
            raise ValueError("retained_solution shape is incompatible with the plan.")
        interior = (
            result.interior_load
            - jnp.matmul(
                result.interior_solution_operator,
                retained[..., None],
            )[..., 0]
        )
        full = jnp.zeros(
            (retained.shape[0], self.local_size),
            dtype=retained.dtype,
        )
        full = full.at[:, self.retained_dofs].set(retained)
        return full.at[:, self.eliminated_dofs].set(interior)


__all__ = [
    "FiniteElementLocalEliminationPlan",
    "LocalEliminationResult",
]
