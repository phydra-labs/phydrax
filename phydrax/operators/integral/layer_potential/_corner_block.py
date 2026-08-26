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


def _inverse(matrix: Array, problem_id: str, /) -> Array:
    return solve(
        LinearSystem(DenseLinearOperator(matrix), problem_id=problem_id),
        jnp.eye(matrix.shape[0], dtype=matrix.dtype),
        policy=LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("error"),
        ),
    ).value


def _recursive_inverse(
    matrix: Array,
    levels: int,
    /,
    *,
    problem_id: str,
) -> Array:
    size = int(matrix.shape[0])
    if size <= 1 or levels <= 1:
        return _inverse(matrix, f"{problem_id}:leaf")
    coarse_size = max(size // 2, 1)
    coarse = matrix[:coarse_size, :coarse_size]
    coarse_fine = matrix[:coarse_size, coarse_size:]
    fine_coarse = matrix[coarse_size:, :coarse_size]
    fine = matrix[coarse_size:, coarse_size:]
    fine_inverse = _recursive_inverse(
        fine,
        levels - 1,
        problem_id=f"{problem_id}:fine",
    )
    schur = coarse - coarse_fine @ fine_inverse @ fine_coarse
    schur_inverse = _inverse(schur, f"{problem_id}:schur")
    upper_right = -schur_inverse @ coarse_fine @ fine_inverse
    lower_left = -fine_inverse @ fine_coarse @ schur_inverse
    lower_right = (
        fine_inverse
        + fine_inverse @ fine_coarse @ schur_inverse @ coarse_fine @ fine_inverse
    )
    return jnp.block(((schur_inverse, upper_right), (lower_left, lower_right)))


class CornerBlockInversePreconditioner2D(StrictModule, NonTrainableState):
    """Recursive local block inverse; not an RCIP claim."""

    matrix: Array
    corner_blocks: tuple[Array, ...]
    local_inverses: tuple[Array, ...]
    levels: int = eqx.field(static=True)
    preconditioner_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        corner_blocks: Sequence[ArrayLike],
        /,
        *,
        levels: int = 3,
    ):
        matrix_ = jnp.asarray(matrix)
        if matrix_.ndim != 2 or matrix_.shape[0] != matrix_.shape[1]:
            raise ValueError("Corner block matrix must be square.")
        depth = int(levels)
        if depth < 1:
            raise ValueError("Corner block recursion levels must be positive.")
        blocks = tuple(
            jnp.asarray(block, dtype=jnp.int32).reshape((-1,)) for block in corner_blocks
        )
        all_indices = (
            jnp.concatenate(blocks) if blocks else jnp.asarray((), dtype=jnp.int32)
        )
        if blocks and jnp.unique(all_indices).size != all_indices.size:
            raise ValueError("Corner blocks must be disjoint.")
        if blocks and bool(
            jnp.any((all_indices < 0) | (all_indices >= matrix_.shape[0]))
        ):
            raise ValueError("Corner block index is outside the matrix.")
        inverses = tuple(
            _recursive_inverse(
                matrix_[jnp.ix_(block, block)],
                depth,
                problem_id=f"corner-block-{index}",
            )
            for index, block in enumerate(blocks)
        )
        self.matrix = matrix_
        self.corner_blocks = blocks
        self.local_inverses = inverses
        self.levels = depth
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "corner-block-recursive-inverse-2d-v1",
                "matrix_shape": matrix_.shape,
                "corner_blocks": array_tree_fingerprint(blocks),
                "levels": depth,
            }
        )

    def apply(self, vector: ArrayLike, /) -> Array:
        values = jnp.asarray(vector)
        if values.shape[0] != self.matrix.shape[0]:
            raise ValueError("Corner block vector must match matrix dimension.")
        output = values
        for block, inverse in zip(self.corner_blocks, self.local_inverses, strict=True):
            local = values[block]
            output = output.at[block].set(output[block] + inverse @ local - local)
        return output


__all__ = ["CornerBlockInversePreconditioner2D"]
