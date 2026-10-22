# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit source-lattice construction; not Delaunay or PLC recovery.

``certified-unit-cell-complex`` uses the immutable 15-site/32-tetrahedron
sphere-intersection template below. Alternating integer-cell reflections give
identical shared face triangulations. The template was selected offline; no
optimization, coordinate recognition, or replacement source triangles occur here.

The authored source coefficients are binary64 lattice-frame values. Rational
affine actions retain these coefficients before rounding world representatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import frexp, inf, nextafter, sqrt
from typing import Literal

import numpy as np

from .._meshcore import exact_orient3d, PlcSourceConstraints
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._exact_rational_geometry import (
    _affine_image_geometry_from_frame,
    affine_image_geometry_from_routes,
)
from ..typing import checked
from ._plc_source import PiecewiseLinearComplex


type LatticeKey = tuple[int, int, int]
type UnitNodeIdentity = tuple[Literal["corner", "face", "body"], *tuple[int, ...]]
UNIT_CELL_CONSTRUCTION = "certified-unit-cell-complex"
_LOCAL_CELLS = (
    (1, 0, 8, 14),
    (0, 1, 10, 14),
    (0, 2, 8, 14),
    (2, 0, 12, 14),
    (4, 0, 10, 14),
    (0, 4, 12, 14),
    (3, 1, 8, 14),
    (1, 3, 13, 14),
    (1, 5, 10, 13),
    (10, 1, 13, 14),
    (2, 3, 8, 11),
    (6, 2, 8, 11),
    (2, 6, 8, 14),
    (6, 2, 12, 14),
    (7, 3, 11, 13),
    (3, 8, 11, 14),
    (3, 11, 13, 14),
    (4, 5, 9, 13),
    (5, 4, 10, 13),
    (6, 4, 9, 12),
    (4, 9, 12, 13),
    (4, 10, 13, 14),
    (12, 4, 13, 14),
    (5, 7, 9, 13),
    (7, 6, 9, 11),
    (6, 8, 9, 11),
    (8, 6, 9, 14),
    (9, 6, 12, 14),
    (9, 7, 11, 13),
    (8, 9, 11, 14),
    (11, 9, 13, 14),
    (9, 12, 13, 14),
)
_CORNERS = tuple(product((-1, 1), repeat=3))
_FACE_OFFSET = nextafter(sqrt(0.75), inf) - 0.5
_LOCAL_POINTS = tuple(tuple(0.5 * v for v in p) for p in _CORNERS) + (
    (-0.5, _FACE_OFFSET, 0.0),
    (0.5, _FACE_OFFSET, 0.0),
    (0.0, -0.5, _FACE_OFFSET),
    (0.0, 0.5, _FACE_OFFSET),
    (_FACE_OFFSET, 0.0, -0.5),
    (_FACE_OFFSET, 0.0, 0.5),
    (0.0, 0.0, sqrt(0.5) - 0.5),
)


@dataclass(frozen=True, slots=True)
class CertifiedSourceLattice:
    """Compiler-authored integer identities bound to the unchanged PLC bank.

    Corner keys are doubled source-index coordinates, never recovered from
    world points. ``affine`` is a signed permutation times one power of
    two, followed by the authored translation. Cells retain their region rows.
    """

    complex_id: str
    cells: tuple[tuple[int, int, int, int], ...]
    vertex_keys: tuple[LatticeKey, ...]
    polygon_loops: tuple[tuple[LatticeKey, ...], ...]
    polygon_facets: tuple[int, ...]
    region_ids: tuple[str, ...]
    affine: tuple[tuple[float, ...], ...]
    physical_cell_length: float


def signed_dyadic_cell_length(matrix: np.ndarray, /) -> float | None:
    """Admit only an authored signed permutation with one power-of-two scale."""
    linear = np.asarray(matrix, dtype=np.float64)[:3, :3]
    nonzero = linear != 0.0
    if not (np.all(nonzero.sum(axis=0) == 1) and np.all(nonzero.sum(axis=1) == 1)):
        return None
    magnitudes = np.abs(linear[nonzero])
    scale = float(magnitudes[0])
    if not np.all(magnitudes == scale) or frexp(scale)[0] != 0.5:
        return None
    return scale


@dataclass(frozen=True, slots=True)
class UnitCellArrays:
    points: np.ndarray
    cells: np.ndarray
    regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    local_coordinates: tuple[tuple[Fraction, ...], ...]
    affine_frame: np.ndarray


def _source_face_ids(
    lattice: CertifiedSourceLattice,
    complex_: PiecewiseLinearComplex,
) -> dict[tuple[int, int, int, int], int]:
    """Resolve authored polygon rectangles into integer cell-face incidences."""
    result: dict[tuple[int, int, int, int], int] = {}
    for loop, facet in zip(lattice.polygon_loops, lattice.polygon_facets, strict=True):
        axes = [axis for axis in range(3) if len({p[axis] for p in loop}) == 1]
        if len(axes) != 1:
            raise ValueError(
                f"CertifiedSourceLattice.polygons must be lattice face loops; got constant_axes={axes}"
            )
        axis = axes[0]
        transverse = [a for a in range(3) if a != axis]
        lower = [min(p[a] for p in loop) for a in transverse]
        upper = [max(p[a] for p in loop) for a in transverse]
        # Compiler loops are conforming axis-aligned rectangles, possibly with
        # retained collinear corners; integer rectangles own every unit face.
        for first in range((lower[0] + 1) // 2, (upper[0] + 1) // 2):
            for second in range((lower[1] + 1) // 2, (upper[1] + 1) // 2):
                key = (axis, loop[0][axis], first, second)
                if key in result:
                    raise ValueError(
                        f"CertifiedSourceLattice.polygons must not overlap; got repeated_face={key}"
                    )
                result[key] = facet
    occupied = {cell[:3]: cell[3] for cell in lattice.cells}
    if len(occupied) != len(lattice.cells):
        raise ValueError(
            "CertifiedSourceLattice.cells must not overlap; got repeated cell indices"
        )
    expected = {}
    for cell, region in occupied.items():
        for axis in range(3):
            transverse = [a for a in range(3) if a != axis]
            for direction in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += direction
                other = occupied.get(tuple(neighbor), -1)
                if other == region:
                    continue
                key = (axis, 2 * cell[axis] + direction, *(cell[a] for a in transverse))
                expected[key] = {region, other}
    if result.keys() != expected.keys():
        raise ValueError(
            "CertifiedSourceLattice.facets must cover exactly the occupied cell boundary; got different face incidences"
        )
    for key, facet in result.items():
        if set(complex_.facet_regions[facet].tolist()) != expected[key]:
            raise ValueError(
                f"CertifiedSourceLattice.facets must retain material incidence; got facet={facet}, cell_regions={sorted(expected[key])}"
            )
    return result


@checked
def construct_unit_cells(
    lattice: CertifiedSourceLattice,
    complex_: PiecewiseLinearComplex,
    constraints: PlcSourceConstraints,
) -> UnitCellArrays:
    """Join by integer corner/face/body identities, then certify orientation.

    The caller reserves and charges the bounded host phase before
    entering. Native source validation independently owns source geometry,
    coverage, intersections, protected strata, and all publication acceptance.
    """
    if (
        lattice.complex_id != complex_.complex_id
        or lattice.region_ids != complex_.region_ids
    ):
        raise ValueError(
            "CertifiedSourceLattice.source must match the declared PLC; got different source or region identities"
        )
    source_faces = _source_face_ids(lattice, complex_)
    matrix = np.asarray(lattice.affine, dtype=np.float64)
    identities: dict[UnitNodeIdentity, int] = {
        ("corner", *key): row for row, key in enumerate(lattice.vertex_keys)
    }
    points = list(np.asarray(complex_.vertices, dtype=np.float64))
    local_coordinates = [
        tuple(Fraction(value, 2) for value in key) for key in lattice.vertex_keys
    ]
    cells = []
    regions = []
    constrained = {}
    # Template boundary is tetrahedral incidence, with explicit local support.
    local_faces = {}
    supports = tuple(frozenset(enumerate(corner)) for corner in _CORNERS) + tuple(
        frozenset(((axis, direction),)) for axis in range(3) for direction in (-1, 1)
    )
    for tet_index, tet in enumerate(_LOCAL_CELLS):
        for face in (
            (tet[1], tet[2], tet[3]),
            (tet[0], tet[3], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[1]),
        ):
            local_faces.setdefault(tuple(sorted(face)), []).append((face, tet_index))
    boundary = []
    for incidences in local_faces.values():
        if len(incidences) != 1:
            continue
        face, tet_index = incidences[0]
        support = frozenset.intersection(*(supports[v] for v in face))
        if len(support) != 1:
            raise ValueError(
                f"UnitCellTemplate.faces must have one explicit cell-face support; got support={sorted(support)}"
            )
        boundary.append((face, *next(iter(support)), tet_index))
    for entry in lattice.cells:
        cell, region = entry[:3], entry[3]
        signs = tuple(1 if index % 2 == 0 else -1 for index in cell)
        rows = []
        for local, point in enumerate(_LOCAL_POINTS):
            if local < 8:
                key = tuple(2 * cell[a] + signs[a] * _CORNERS[local][a] for a in range(3))
                identity = ("corner", *key)
            elif local < 14:
                axis, side = divmod(local - 8, 2)
                transverse = [a for a in range(3) if a != axis]
                identity = (
                    "face",
                    axis,
                    2 * cell[axis] + signs[axis] * (2 * side - 1),
                    *(cell[a] for a in transverse),
                )
            else:
                identity = ("body", *cell)
            row = identities.get(identity)
            if row is None:
                row = len(points)
                identities[identity] = row
                q = np.asarray(
                    [cell[a] + signs[a] * point[a] for a in range(3)], dtype=np.float64
                )
                authored = tuple(Fraction(float(value)) for value in q)
                local_coordinates.append(authored)
                points.append(
                    np.asarray(
                        [
                            float(
                                Fraction(float(matrix[a, 3]))
                                + sum(
                                    (
                                        Fraction(float(matrix[a, b])) * authored[b]
                                        for b in range(3)
                                    ),
                                    Fraction(0),
                                )
                            )
                            for a in range(3)
                        ],
                        dtype=np.float64,
                    )
                )
            rows.append(row)
        cell_start = len(cells)
        cells.extend(tuple(rows[v] for v in tet) for tet in _LOCAL_CELLS)
        regions.extend([region] * len(_LOCAL_CELLS))
        for face, axis, direction, tet_index in boundary:
            transverse = [a for a in range(3) if a != axis]
            identity = (
                axis,
                2 * cell[axis] + signs[axis] * direction,
                *(cell[a] for a in transverse),
            )
            facet = source_faces.get(identity)
            if facet is not None:
                mapped = tuple(rows[v] for v in face)
                constrained.setdefault(
                    tuple(sorted(mapped)),
                    (mapped, facet, cell_start + tet_index, region),
                )
    point_array = np.asarray(points, dtype=np.float64)
    cell_array = np.asarray(cells, dtype=np.int32)
    orientations = exact_orient3d(
        *(point_array[cell_array[:, slot]] for slot in range(4))
    )
    if np.any(orientations == 0):
        raise ValueError(
            "UnitCellTemplate.cells must retain nonzero exact orientation; got a degenerate cell"
        )
    negative = np.flatnonzero(orientations < 0)
    cell_array[negative, :2] = cell_array[negative, 1::-1]
    faces = []
    face_sources = []
    for face, facet, tet_index, region in constrained.values():
        # Incidence is outward from a positive tetrahedron. Publish the
        # authored facet orientation (facet_regions[:, 1] side).
        reverse = (orientations[tet_index] < 0) ^ (
            region != complex_.facet_regions[facet, 1]
        )
        faces.append(face[::-1] if reverse else face)
        face_sources.append(facet)
    # Every PLC edge is an explicit lattice-axis interval. Split at
    # integer corners without changing its scientific source row identity.
    segments = []
    segment_sources = []
    for source in range(constraints.plc_edges.shape[0]):
        start = lattice.vertex_keys[int(constraints.plc_edges[source, 0])]
        end = lattice.vertex_keys[int(constraints.plc_edges[source, 1])]
        varying = [a for a in range(3) if start[a] != end[a]]
        if len(varying) != 1:
            raise ValueError(
                f"CertifiedSourceLattice.edges must have lattice-axis support; got source_edge={source}, varying_axes={varying}"
            )
        axis = varying[0]
        low, high = sorted((start[axis], end[axis]))
        keys = []
        for value in range(low, high + 1, 2):
            key = list(start)
            key[axis] = value
            keys.append(identities[("corner", *key)])
        segments.extend(zip(keys[:-1], keys[1:]))
        segment_sources.extend([source] * (len(keys) - 1))
    affine_frame = np.concatenate(
        (matrix[:3, 3][None], matrix[:3, :3].T, np.zeros((1, 3), dtype=np.float64))
    )
    affine_frame.setflags(write=False)
    return UnitCellArrays(
        point_array,
        cell_array,
        np.asarray(regions, dtype=np.int32),
        np.asarray(faces, dtype=np.int32).reshape(-1, 3),
        np.asarray(face_sources, dtype=np.int64),
        np.asarray(segments, dtype=np.int32).reshape(-1, 2),
        np.asarray(segment_sources, dtype=np.int64),
        tuple(local_coordinates),
        affine_frame,
    )


@checked
def unit_cell_geometry(
    arrays: UnitCellArrays,
    mesh: CellMesh,
    /,
    *,
    vertex_rows: np.ndarray | None = None,
    budget: CoordinateEnclosureBudget,
) -> CellGeometrySpec:
    """Bind lattice coefficients to the authored affine and its execution view.

    The parent bank stores the translation, three linear columns, and zero
    separately; no rounded sum of frame coefficients becomes source authority.
    Only unchanged source rows or explicit compaction are supported.
    """
    rows = (
        range(len(arrays.local_coordinates))
        if vertex_rows is None
        else vertex_rows.tolist()
    )
    geometry = _affine_image_geometry_from_frame(
        arrays.affine_frame,
        tuple(arrays.local_coordinates[index] for index in rows),
        {
            block.name: coordinate_lagrange_element(block.cell_kind, 1)
            for block in mesh.blocks
        },
        {block.name: block.vertices for block in mesh.blocks},
        budget=budget,
    )
    geometry.resolve(mesh, budget=budget)
    return geometry


@checked
def affine_image_geometry(
    mesh: CellMesh,
    affine: np.ndarray,
    local_coordinates: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellGeometrySpec:
    """Apply an authored affine exactly, retaining correctly rounded coordinates."""
    if len(local_coordinates) != mesh.coordinates.shape[0]:
        raise ValueError(
            "affine_image_geometry.local_coordinates must name every mesh vertex; got a different coefficient count"
        )
    return affine_image_geometry_from_routes(
        affine,
        local_coordinates,
        {
            block.name: coordinate_lagrange_element(block.cell_kind, 1)
            for block in mesh.blocks
        },
        {block.name: block.vertices for block in mesh.blocks},
        budget=budget,
    )


