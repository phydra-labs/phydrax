"""Native declared-contact proofs against the exact rational contact pass.

The native proof states only that every tetrahedron contact of a cell pair
lies in one convex piece of its declared shared stratum; each such statement
must be the rational verdict of the exact contact vertices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction

import numpy as np

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.geometry._embedding_cells import EmbeddingCell, prepare_embedding_cells
from phydrax.geometry._exact_polyhedral_geometry import tetrahedron_contact_points
from phydrax.geometry._global_embedding import (
    _cell_simplices,
    _PairStratum,
    _point3,
    _VolumePieceFilter,
)


type CornerMap = Callable[[ExactScalar], ExactScalar]


def _rational_declared(
    first: EmbeddingCell,
    second: EmbeddingCell,
    first_pieces: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
    second_pieces: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
    /,
) -> bool:
    stratum = _PairStratum(first, second)
    return all(
        stratum.declared(
            tetrahedron_contact_points(
                tuple(_point3(point) for point in left),
                tuple(_point3(point) for point in right),
            )
        )
        is True
        for left in first_pieces
        for right in second_pieces
    )


def declared_contact_differential(
    mesh: CellMesh, corner_map: CornerMap | None = None, /
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """``(native, rational)`` declared cell pairs of an affine polyhedral mesh.

    ``corner_map`` replaces every corner coordinate exactly (an affine map
    keeps incidence, planarity and convexity). Every native pair must be
    rational; the rational set is computed for every pair of cells.
    """
    cells = prepare_embedding_cells(mesh, CellGeometrySpec.affine(mesh))
    if corner_map is not None:
        cells = tuple(
            replace(
                cell,
                corner_images=tuple(
                    tuple(corner_map(value) for value in point)
                    for point in cell.corner_images
                ),
            )
            for cell in cells
        )
    pieces = [_cell_simplices(cell) for cell in cells]
    piece_filter = _VolumePieceFilter.prepare(mesh, cells, [*pieces])
    if piece_filter is None:
        raise ValueError("The differential requires straight cells in R3.")
    first, second = np.triu_indices(len(cells), 1)
    _, _, native = piece_filter.certify(first, second)
    rational = {
        (left, right)
        for left, right in zip(first.tolist(), second.tolist(), strict=True)
        if _rational_declared(cells[left], cells[right], pieces[left], pieces[right])
    }
    return native, rational


def non_dyadic(value: ExactScalar, /) -> ExactScalar:
    """An affine image whose coordinates are off binary64."""
    if not isinstance(value, Fraction):
        raise TypeError("Only rational coordinates have a rational image.")
    return value / 3 + Fraction(1, 7)
