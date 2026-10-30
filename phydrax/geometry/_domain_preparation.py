#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare checked dense region assignments and exact cell coordinate maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax.typing import ArrayLike

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import Expression
from ._domain_numerics import cell_coordinate_expressions


@dataclass(frozen=True, slots=True)
class CellCoverageAssignment:
    """One target cell row bound to its stable ID and dense source region index."""

    row: int
    block_index: int
    block_row: int
    global_id: int
    region: int


@dataclass(frozen=True, slots=True)
class DomainCoveragePreparation:
    """Immutable request-local cell maps and dense assignments."""

    cell_regions: np.ndarray
    assignments: tuple[CellCoverageAssignment, ...]
    coordinate_maps: tuple[tuple[Expression, ...] | None, ...]


def prepare_domain_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: ArrayLike,
    region_count: int,
    /,
) -> DomainCoveragePreparation:
    """Validate dense region-table indices and prepare exact maps in cell order."""
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Domain coverage requires CellMesh and CellGeometrySpec inputs.")
    if (
        isinstance(region_count, bool)
        or not isinstance(region_count, int)
        or region_count < 1
    ):
        raise ValueError("region_count must be a positive region-table size.")
    raw = np.asarray(cell_regions)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise TypeError(
            "cell_regions must be a one-dimensional integer region-index array."
        )
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("cell_regions contains values outside the int64 range.")
    regions = np.array(raw, dtype=np.int64, copy=True)
    cell_count = sum(block.cell_count for block in mesh.blocks)
    if regions.shape != (cell_count,):
        raise ValueError("cell_regions must assign every target cell exactly once.")
    if np.any(regions < 0) or np.any(regions >= region_count):
        raise ValueError("cell_regions must index the dense domain.region_ids table.")
    maps = cell_coordinate_expressions(mesh, geometry)
    if len(maps) != cell_count:
        raise ValueError("Exact coordinate maps must align with target cell order.")
    assignments: list[CellCoverageAssignment] = []
    cursor = 0
    for block_index, block in enumerate(mesh.blocks):
        for block_row, global_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64)
        ):
            assignments.append(
                CellCoverageAssignment(
                    cursor,
                    block_index,
                    block_row,
                    int(global_id),
                    int(regions[cursor]),
                )
            )
            cursor += 1
    regions.setflags(write=False)
    return DomainCoveragePreparation(regions, tuple(assignments), maps)


__all__ = [
    "CellCoverageAssignment",
    "DomainCoveragePreparation",
    "prepare_domain_coverage",
]
