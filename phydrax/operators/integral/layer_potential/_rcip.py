#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


def _dense_solve(matrix: Array, right: Array, problem_id: str, /) -> Array:
    return solve(
        LinearSystem(
            DenseLinearOperator(matrix),
            problem_id=problem_id,
        ),
        right,
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        ),
    ).value


class RCIPPreconditioner2D(StrictModule, NonTrainableState):
    """Nested fine-grid inverse compression for declared corner operators.

    The caller supplies the Nyström operator at every refinement level together
    with restriction/prolongation maps. The prepared action is the recursively
    compressed inverse on the coarsest unknowns; no same-grid block inverse is
    silently relabeled as RCIP.
    """

    coarse_matrix: Array
    fine_matrices: tuple[Array, ...]
    restrictions: tuple[Array, ...]
    prolongations: tuple[Array, ...]
    compressed_inverses: tuple[Array, ...]
    topology_id: str = eqx.field(static=True)
    levels: int = eqx.field(static=True)
    preconditioner_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_matrix: ArrayLike,
        fine_matrices: Sequence[ArrayLike],
        restrictions: Sequence[ArrayLike],
        prolongations: Sequence[ArrayLike],
        /,
        *,
        topology_id: str,
    ):
        coarse = jnp.asarray(coarse_matrix)
        if coarse.ndim != 2 or coarse.shape[0] != coarse.shape[1]:
            raise ValueError("RCIP coarse_matrix must be square.")
        fine = tuple(jnp.asarray(matrix) for matrix in fine_matrices)
        restrict = tuple(jnp.asarray(matrix) for matrix in restrictions)
        prolong = tuple(jnp.asarray(matrix) for matrix in prolongations)
        if len(fine) != len(restrict) or len(fine) != len(prolong):
            raise ValueError(
                "RCIP levels require matching fine, restriction, and prolongation tuples."
            )
        previous_size = int(coarse.shape[0])
        for matrix, restriction, prolongation in zip(
            fine, restrict, prolong, strict=True
        ):
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Every RCIP fine matrix must be square.")
            fine_size = int(matrix.shape[0])
            if restriction.shape != (previous_size, fine_size):
                raise ValueError("RCIP restriction shape does not match adjacent levels.")
            if prolongation.shape != (fine_size, previous_size):
                raise ValueError(
                    "RCIP prolongation shape does not match adjacent levels."
                )
            identity = jnp.eye(
                previous_size, dtype=jnp.result_type(restriction, prolongation)
            )
            if not bool(
                jnp.allclose(
                    restriction @ prolongation,
                    identity,
                    rtol=1e-8,
                    atol=1e-10,
                )
            ):
                raise ValueError(
                    "RCIP restriction/prolongation must reproduce coarse data."
                )
            previous_size = fine_size
        if not topology_id:
            raise ValueError("RCIP topology_id must be nonempty.")
        base = fine[-1] if fine else coarse
        current = _dense_solve(
            base,
            jnp.eye(base.shape[0], dtype=base.dtype),
            f"rcip:{topology_id}:finest-inverse",
        )
        compressed_reversed = [current]
        for level in range(len(fine) - 2, -1, -1):
            compressed = restrictions[level + 1] @ current @ prolongations[level + 1]
            identity = jnp.eye(compressed.shape[0], dtype=compressed.dtype)
            local_operator = fine[level]
            current = _dense_solve(
                identity + compressed @ (local_operator - identity),
                compressed,
                f"rcip:{topology_id}:level-{level}-compression",
            )
            compressed_reversed.append(current)
        if fine:
            compressed = restrictions[0] @ current @ prolongations[0]
            identity = jnp.eye(compressed.shape[0], dtype=compressed.dtype)
            current = _dense_solve(
                identity + compressed @ (coarse - identity),
                compressed,
                f"rcip:{topology_id}:coarse-compression",
            )
            compressed_reversed.append(current)
        compressed = tuple(reversed(compressed_reversed))
        self.coarse_matrix = coarse
        self.fine_matrices = fine
        self.restrictions = restrict
        self.prolongations = prolong
        self.compressed_inverses = compressed
        self.topology_id = str(topology_id)
        self.levels = len(fine)
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "rcip-nested-compression-2d-v1",
                "topology_id": self.topology_id,
                "coarse_matrix": array_tree_fingerprint(coarse),
                "fine_matrices": array_tree_fingerprint(fine),
                "restrictions": array_tree_fingerprint(restrict),
                "prolongations": array_tree_fingerprint(prolong),
            }
        )

    def apply(self, vector: ArrayLike, /) -> Array:
        """Apply the compressed coarse inverse to coarse unknowns."""
        values = jnp.asarray(vector)
        if values.shape != (self.coarse_matrix.shape[0],):
            raise ValueError("RCIP vector must match the coarse matrix dimension.")
        return self.compressed_inverses[0] @ values


__all__ = ["RCIPPreconditioner2D"]
