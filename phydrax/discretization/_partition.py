#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_mesh import CellMesh


class CellPartition(StrictModule, NonTrainableState):
    """Solver-neutral exactly-once cell ownership in canonical local cell order."""

    cell_owner: Array
    part_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(self, cell_owner: ArrayLike, part_count: int, /):
        owner = np.asarray(cell_owner)
        count = operator.index(part_count)
        if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer):
            raise TypeError("Cell ownership must be an integer vector.")
        if (
            isinstance(part_count, bool)
            or count <= 0
            or np.any(owner < 0)
            or np.any(owner >= count)
        ):
            raise ValueError("Cell ownership or part_count is invalid.")
        if np.unique(owner).size != count:
            raise ValueError("Every partition must own at least one cell.")
        owner = owner.astype(np.int32, copy=False)
        self.cell_owner = jnp.asarray(owner)
        self.part_count = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "cell-partition",
                "cell_owner": array_tree_fingerprint(owner),
                "part_count": count,
            }
        )


def partition_cells_contiguous(mesh: CellMesh, part_count: int, /) -> CellPartition:
    """Partition the canonical concatenated block ordering without splitting cells."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    count = sum(block.cell_count for block in mesh.blocks)
    parts = operator.index(part_count)
    if isinstance(part_count, bool) or parts <= 0 or parts > count:
        raise ValueError("part_count must lie between one and the cell count.")
    owner = np.arange(count, dtype=np.int64) * parts // count
    return CellPartition(owner, parts)


__all__ = ["CellPartition", "partition_cells_contiguous"]
