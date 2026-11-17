#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare checked dense region assignments and exact cell coordinate maps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from jax.typing import ArrayLike

from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    CoordinateSourceBank,
    Expression,
    PreparedLinearCombinationBasis,
    RestrictedCoordinateMapPreparation,
)
from ._domain_numerics import _coordinate_route_matrix


@dataclass(frozen=True, slots=True)
class CellCoverageAssignment:
    """One target cell row bound to its stable ID and dense source region index."""

    row: int
    block_index: int
    block_row: int
    global_id: int
    region: int


type _LazyMapRow = tuple[int, ScalarCoordinateElement, np.ndarray, int] | None


class _LazyCoordinateMaps(Sequence[tuple[Expression, ...] | None]):
    """Materialize the same exact maps as domain numerics, only when consumed."""

    def __init__(self, mesh: CellMesh, geometry: CellGeometrySpec, count: int, /) -> None:
        self._mesh = mesh
        self._geometry = geometry
        self._count = count
        self._rows: list[_LazyMapRow] | None = None
        self._coefficients: CoordinateSourceBank | None = None
        self._restricted: dict[int, RestrictedCoordinateMapPreparation] = {}
        self._bases: dict[int, PreparedLinearCombinationBasis] = {}
        self._maps: dict[int, tuple[Expression, ...] | None] = {}
        self._prepare()

    def __len__(self) -> int:
        return self._count

    def _prepare(self) -> None:
        if self._rows is not None:
            return
        elements, routes, _ = self._geometry.resolve(self._mesh)
        coefficients = self._geometry.source_coordinates()
        rows: list[_LazyMapRow] = []
        for block_index, (block, element, route_bank) in enumerate(
            zip(self._mesh.blocks, elements, routes, strict=True)
        ):
            if isinstance(element, CellVertexGeometryElement):
                rows.extend((None,) * block.cell_count)
                continue
            if not isinstance(element, ScalarCoordinateElement):
                raise TypeError(
                    "Exact domain maps require a supported scalar coordinate element."
                )
            routes_host = _coordinate_route_matrix(route_bank)
            rows.extend(
                (block_index, element, routes_host, row)
                for row in range(routes_host.shape[0])
            )
        if len(rows) != self._count:
            raise ValueError("Exact coordinate maps must align with target cell order.")
        self._coefficients = coefficients
        self._rows = rows

    @overload
    def __getitem__(self, index: int, /) -> tuple[Expression, ...] | None: ...

    @overload
    def __getitem__(
        self, index: slice, /
    ) -> tuple[tuple[Expression, ...] | None, ...]: ...

    def __getitem__(
        self, index: int | slice, /
    ) -> tuple[Expression, ...] | None | tuple[tuple[Expression, ...] | None, ...]:
        if isinstance(index, slice):
            return tuple(self[row] for row in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("coordinate map index out of range")
        if index in self._maps:
            return self._maps[index]
        self._prepare()
        rows = self._rows
        coefficients = self._coefficients
        if rows is None or coefficients is None:
            raise RuntimeError("Coordinate map preparation is incomplete.")
        row = rows[index]
        if row is None:
            result = None
        else:
            block_index, element, routes, row_index = row
            local = tuple(coefficients[int(value)] for value in routes[row_index])
            if isinstance(element, PerCellRestrictedCellGeometryElement):
                if block_index not in self._restricted:
                    self._restricted[block_index] = RestrictedCoordinateMapPreparation(
                        element, coefficients, routes, mode="source"
                    )
                expressions = self._restricted[block_index].expressions(row_index)
            else:
                expressions = None
            if expressions is None:
                basis = (
                    None
                    if block_index in self._bases
                    and not isinstance(element, PerCellRestrictedCellGeometryElement)
                    else coordinate_effective_basis(element, cell_row=row_index)
                )
                if basis is not None and len(basis) != len(local):
                    raise ValueError(
                        "Exact coordinate basis and local source bank widths differ."
                    )
                if basis is not None:
                    prepared_basis = PreparedLinearCombinationBasis(basis)
                    if not isinstance(element, PerCellRestrictedCellGeometryElement):
                        self._bases[block_index] = prepared_basis
                else:
                    prepared_basis = self._bases[block_index]
                components = tuple(
                    tuple(local[dof][axis] for dof in range(len(local)))
                    for axis in range(len(local[0]))
                )
                expressions = prepared_basis.combine(components)
            result = expressions
        self._maps[index] = result
        return result

    def __iter__(self) -> Iterator[tuple[Expression, ...] | None]:
        return (self[index] for index in range(self._count))


@dataclass(frozen=True, slots=True)
class DomainCoveragePreparation:
    """Request-local exact maps and immutable dense assignments."""

    cell_regions: np.ndarray
    assignments: tuple[CellCoverageAssignment, ...]
    coordinate_maps: Sequence[tuple[Expression, ...] | None]


def prepare_domain_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: ArrayLike,
    region_count: int,
    /,
) -> DomainCoveragePreparation:
    """Validate dense region-table indices and defer exact maps until consumed."""
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
    return DomainCoveragePreparation(
        regions, tuple(assignments), _LazyCoordinateMaps(mesh, geometry, cell_count)
    )


__all__ = [
    "CellCoverageAssignment",
    "DomainCoveragePreparation",
    "prepare_domain_coverage",
]
