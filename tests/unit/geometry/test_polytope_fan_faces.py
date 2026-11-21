"""Polyhedron faces: planar polygons, non-planar loops as vertex-average fans."""

from __future__ import annotations

from fractions import Fraction

import pytest

from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._exact_polyhedral_geometry import determinant3, Point3
from phydrax.geometry._global_embedding import _volume_tetrahedra, UnsupportedEmbeddingMap
from phydrax.geometry._planar_chain_embedding import (
    polytope_face_triangles,
    triangulate_planar_polygon,
)


type _Point = tuple[Fraction, Fraction, Fraction]


def _p(x: int | Fraction, y: int | Fraction, z: int | Fraction, /) -> _Point:
    return Fraction(x), Fraction(y), Fraction(z)


def _polyhedron(
    points: tuple[_Point, ...], faces: tuple[tuple[int, ...], ...]
) -> EmbeddingCell:
    return EmbeddingCell(
        global_id=0,
        block_name="cells",
        local_index=0,
        kind="polyhedron",
        dimension=3,
        domain="polyhedron",
        vertex_indices=tuple(range(len(points))),
        vertex_ids=tuple(range(len(points))),
        reference_vertices=(),
        coordinate_expressions=(),
        corner_images=points,
        affine_offset=None,
        affine_matrix=None,
        image_lower=tuple(float(min(p[axis] for p in points)) for axis in range(3)),
        image_upper=tuple(float(max(p[axis] for p in points)) for axis in range(3)),
        facet_routes=faces,
    )


_HEXAGON = ((2, 0), (1, 2), (-1, 2), (-2, 0), (-1, -2), (1, -2))


def _prism(
    top_heights: tuple[Fraction, ...], ring: tuple[tuple[int, int], ...] = _HEXAGON
) -> EmbeddingCell:
    """A prism over a polygon at z = 0, its top loop at the given heights."""
    count = len(ring)
    points = tuple(_p(x, y, 0) for x, y in ring) + tuple(
        _p(x, y, z) for (x, y), z in zip(ring, top_heights, strict=True)
    )
    faces = (
        tuple(range(count - 1, -1, -1)),
        tuple(range(count, 2 * count)),
        *((k, (k + 1) % count, count + (k + 1) % count, count + k) for k in range(count)),
    )
    return _polyhedron(points, faces)


def _fan_volume(cell: EmbeddingCell) -> ExactScalar:
    origin = _p(0, 0, 0)
    images: tuple[Point3, ...] = tuple(
        (point[0], point[1], point[2]) for point in cell.corner_images
    )
    return (
        sum(
            (
                determinant3(origin, *triangle)
                for face in cell.facet_routes
                for triangle in polytope_face_triangles(
                    tuple(images[index] for index in face)
                )
            ),
            Fraction(0),
        )
        / 6
    )


def test_planar_face_keeps_its_exact_polygon() -> None:
    square = (_p(0, 0, 0), _p(1, 0, 0), _p(1, 1, 0), _p(0, 1, 0))

    triangles = polytope_face_triangles(square)

    assert triangles == tuple(
        (square[a], square[b], square[c])
        for a, b, c in triangulate_planar_polygon(square)
    )


def test_twisted_quad_is_the_fan_from_its_exact_vertex_average() -> None:
    quad = (_p(0, 0, 0), _p(1, 0, Fraction(1, 3)), _p(1, 1, 0), _p(0, 1, Fraction(1, 7)))

    triangles = polytope_face_triangles(quad)

    centre = _p(Fraction(1, 2), Fraction(1, 2), Fraction(10, 84))
    assert triangles == tuple((centre, quad[k], quad[(k + 1) % 4]) for k in range(4))


def test_non_planar_hexagon_cell_decomposes_into_its_fan_volume() -> None:
    heights = tuple(
        Fraction(value)
        for value in (1, Fraction(6, 5), 1, Fraction(6, 5), 1, Fraction(6, 5))
    )
    cell = _prism(heights)

    tetrahedra = _volume_tetrahedra(cell)

    volume = sum((determinant3(*tet) for tet in tetrahedra), Fraction(0)) / 6
    assert volume > 0
    assert volume == _fan_volume(cell)
    # The non-planar top contributes one fan triangle
    # per edge from its exact vertex average.
    assert sum(1 for tet in tetrahedra if _p(0, 0, Fraction(11, 10)) in tet) == 6


def test_inverted_fan_refuses() -> None:
    # An L-shaped prism is star-shaped from its corner square, but the vertex
    # average of its L lies in the notch: once the top loop leaves its plane,
    # the top fan folds over the notch and no exact star point sees it.
    ring = ((0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4))
    flat = _prism((Fraction(1),) * 6, ring)
    assert (
        sum((determinant3(*tet) for tet in _volume_tetrahedra(flat)), Fraction(0))
        == 6 * 7
    )

    with pytest.raises(UnsupportedEmbeddingMap, match="star point"):
        _volume_tetrahedra(_prism((Fraction(1), Fraction(11, 10)) * 3, ring))
