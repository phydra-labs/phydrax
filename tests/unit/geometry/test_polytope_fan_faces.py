"""Polyhedron faces: planar polygons, non-planar loops as vertex-average fans."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.geometry._embedding_cells import EmbeddingCell
from phydrax.geometry._exact_polyhedral_geometry import determinant3, Point3
from phydrax.geometry._global_embedding import (
    _volume_tetrahedra,
    certify_global_embedding,
)
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


def _hexahedron_faces(v: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    b0, b1, b2, b3, t0, t1, t2, t3 = v
    return (
        (b0, b3, b2, b1),
        (t0, t1, t2, t3),
        (b0, b1, t1, t0),
        (b1, b2, t2, t1),
        (b2, b3, t3, t2),
        (b3, b0, t0, t3),
    )


def _embedding_status(
    coordinates: list[tuple[float, float, float]], cells: list[tuple[int, ...]]
) -> str:
    mesh = CellMesh.from_polyhedra(
        np.asarray(coordinates, dtype=np.float64),
        [
            [np.asarray(face, dtype=np.int32) for face in _hexahedron_faces(cell)]
            for cell in cells
        ],
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    return certify_global_embedding(mesh, geometry, validity).status


_SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


@pytest.mark.parametrize("top", [2.0, 2.25])
def test_stacked_cells_embed_across_a_non_planar_shared_face(top: float) -> None:
    # The middle face (shared) is twisted; the top face is planar or twisted
    # (a non-planar exterior face in the degree routes).
    coordinates = (
        [(x, y, 0.0) for x, y in _SQUARE]
        + [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.25), (0.0, 1.0, 1.0)]
        + [(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, top), (0.0, 1.0, 2.0)]
    )

    status = _embedding_status(coordinates, [tuple(range(8)), tuple(range(4, 12))])

    assert status == "certified"


_FOLDED_CUBE = [
    (0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0),
    (4.0, 4.0, 0.0),
    (3.0, 4.0, 2.0),
    (0.0, 0.0, 4.0),
    (4.0, 0.0, 4.0),
    (4.0, 4.0, 4.0),
    (0.0, 3.0, 4.0),
]


def test_folded_fan_refuses() -> None:
    # The side loop (2, 3, 7, 6) of this cube with two moved corners has a
    # fan triangle against its area vector: its fan folds.
    loop = (_p(4, 4, 0), _p(3, 4, 2), _p(0, 3, 4), _p(4, 4, 4))
    with pytest.raises(ValueError, match="folds"):
        polytope_face_triangles(loop)

    assert _embedding_status(_FOLDED_CUBE, [tuple(range(8))]) != "certified"


@pytest.mark.parametrize(
    "loop",
    [
        ((0, 0), (1, 1), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    ],
    ids=["bowtie", "repeated-vertex", "collinear"],
)
def test_zero_area_loops_refuse(loop: tuple[tuple[int, int], ...]) -> None:
    with pytest.raises(ValueError, match="zero exact area"):
        polytope_face_triangles(tuple(_p(x, y, 0) for x, y in loop))


def test_fan_loop_winding_twice_refuses() -> None:
    # A non-planar pentagram: every fan triangle turns forward about the
    # area vector, but the loop winds twice about its vertex average.
    ring = ((1000, 0), (309, 951), (-809, 588), (-809, -588), (309, -951))
    loop = tuple(
        _p(*ring[index], Fraction(height, 10))
        for index, height in zip((0, 2, 4, 1, 3), (0, 1, 0, 1, 0), strict=True)
    )
    with pytest.raises(ValueError, match="winds more than once"):
        polytope_face_triangles(loop)


def _degree_witness_values(coordinates: list[tuple[float, float, float]]) -> str:
    mesh = CellMesh.from_polyhedra(
        np.asarray(coordinates, dtype=np.float64),
        [
            [
                np.asarray(face, dtype=np.int32)
                for face in _hexahedron_faces(tuple(range(8)))
            ]
        ],
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = certify_global_embedding(mesh, geometry, validity)
    assert certificate.status == "certified"
    return repr(
        [
            (witness.premise_ids, witness.exact_values)
            for witness in certificate.witnesses
            if witness.theorem == "global_degree_one_injectivity"
        ]
    )


def test_degree_witness_names_planar_and_fan_exterior_boundaries() -> None:
    cube = [(x, y, 0.0) for x, y in _SQUARE] + [(x, y, 1.0) for x, y in _SQUARE]
    planar = _degree_witness_values(cube)
    assert "embedded_planar_exterior_boundary" in planar
    assert "exterior_planar_embedded_facets:" in planar
    assert "fan" not in planar

    cube[6] = (1.0, 1.0, 1.25)
    fan = _degree_witness_values(cube)
    assert "embedded_fan_exterior_boundary" in fan
    assert "exterior_fan_embedded_facets:" in fan
    assert "embedded_planar_exterior_boundary" not in fan
    assert "exterior_planar_embedded_facets:" not in fan
