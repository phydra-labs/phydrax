"""Native polyhedral star proofs equal the rational proofs they replace."""

from __future__ import annotations

import random
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
from scipy.spatial import ConvexHull

import phydrax._meshcore as meshcore
from phydrax.discretization import CellMesh, CellValidityPolicy, PolyhedralConnectivity
from phydrax.discretization._cell_geometry_source_validity import (
    _polyhedron_proof,
    _rational_polyhedron_proof,
)
from phydrax.discretization._cell_geometry_validity import polyhedral_star_tables
from phydrax.geometry._boundary_degree_embedding import _native_face_routes
from phydrax.geometry._embedding_cells import EmbeddingCell, UnsupportedEmbeddingMap
from phydrax.geometry._global_embedding import (
    _native_volume_tetrahedra,
    _volume_tetrahedra,
)
from phydrax.geometry._planar_chain_embedding import triangulate_planar_polygon


pytestmark = pytest.mark.skipif(
    not meshcore.meshcore_available(), reason="the native meshcore library is absent"
)

type Polyhedron = tuple[np.ndarray, tuple[tuple[int, ...], ...]]

_CUBE = np.asarray(
    [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
    dtype=np.float64,
)
_CUBE_FACES = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def _cube() -> Polyhedron:
    return _CUBE.copy(), _CUBE_FACES


def _sheared() -> Polyhedron:
    shear = np.asarray(
        [[1.0, 0.375, -0.25], [0.125, 0.75, 0.5], [-0.5, 0.0625, 1.25]], dtype=np.float64
    )
    return _CUBE @ shear.T + np.asarray([0.3, -2.0, 1e-3]), _CUBE_FACES


def _hanging() -> Polyhedron:
    # Hanging nodes: an edge midpoint and a face-edge midpoint, collinear with
    # their face corners.
    points = np.vstack((_CUBE, [[0.5, 0.0, 0.0], [1.0, 0.5, 1.0]]))
    faces = (
        (0, 2, 3, 1, 8),
        (4, 5, 9, 7, 6),
        (0, 8, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 9, 5),
    )
    return points, faces


def _inward() -> Polyhedron:
    return _CUBE.copy(), tuple(face[::-1] for face in _CUBE_FACES)


def _mixed_orientation() -> Polyhedron:
    return _CUBE.copy(), tuple(
        face[::-1] if index % 2 else face for index, face in enumerate(_CUBE_FACES)
    )


def _bent() -> Polyhedron:
    points = _CUBE.copy()
    points[7, 2] = 1.25
    return points, _CUBE_FACES


def _two_components() -> Polyhedron:
    points = np.vstack((_CUBE, _CUBE + np.asarray([3.0, 0.0, 0.0])))
    faces = _CUBE_FACES + tuple(
        tuple(index + 8 for index in face) for face in _CUBE_FACES
    )
    return points, faces


def _self_intersecting() -> Polyhedron:
    # A pyramid over a self-intersecting (bowtie) base.
    points = np.asarray(
        [[0, 0, 0], [3, 2, 0], [3, 0, 0], [0, 1, 0], [1.5, 1.0, 1.0]], dtype=np.float64
    )
    faces = ((0, 1, 2, 3), (0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0))
    return points, faces


def _tiny() -> Polyhedron:
    points = _CUBE.copy()
    points[1, 1] = 2.0**-200
    return points, _CUBE_FACES


def _hull(seed: int) -> Callable[[], Polyhedron]:
    def build() -> Polyhedron:
        generator = np.random.default_rng(seed)
        # Dyadic coordinates keep every vertex an exact binary64 point.
        points = np.round(generator.normal(size=(24, 3)) * 64.0) / 64.0
        hull = ConvexHull(points)
        used = np.unique(hull.simplices)
        index = np.full((points.shape[0],), -1, dtype=np.int64)
        index[used] = np.arange(used.size)
        simplices = []
        for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
            a, b, c = points[simplex]
            outward = float(np.dot(np.cross(b - a, c - a), equation[:3])) > 0.0
            simplices.append(simplex if outward else simplex[::-1])
        faces = tuple(
            tuple(int(index[vertex]) for vertex in simplex) for simplex in simplices
        )
        return points[used], faces

    return build


_FIXTURES: dict[str, Callable[[], Polyhedron]] = {
    "cube": _cube,
    "sheared": _sheared,
    "hanging-nodes": _hanging,
    "inward": _inward,
    "mixed-orientation": _mixed_orientation,
    "bent": _bent,
    "two-components": _two_components,
    "self-intersecting": _self_intersecting,
    "tiny-coordinate": _tiny,
    **{f"hull-{seed}": _hull(seed) for seed in range(4)},
}


def _exact(value: float, rational: bool, /) -> Fraction:
    # Non-dyadic images keep every statement exact but off binary64.
    exact = Fraction(float(value))
    return exact / 3 + Fraction(1, 7) if rational else exact


def _embedding_cell(
    points: np.ndarray, faces: tuple[tuple[int, ...], ...], rational: bool
) -> EmbeddingCell:
    images = tuple(tuple(_exact(value, rational) for value in row) for row in points)
    return EmbeddingCell(
        global_id=0,
        block_name="cells",
        local_index=0,
        kind="polyhedron",
        dimension=3,
        domain="polyhedron",
        vertex_indices=tuple(range(len(images))),
        vertex_ids=tuple(range(len(images))),
        reference_vertices=(),
        coordinate_expressions=(),
        corner_images=images,
        affine_offset=None,
        affine_matrix=None,
        image_lower=tuple(float(np.min(points[:, axis])) for axis in range(3)),
        image_upper=tuple(float(np.max(points[:, axis])) for axis in range(3)),
        facet_routes=faces,
    )


@pytest.mark.parametrize("rational", (False, True), ids=("binary64", "rational"))
@pytest.mark.parametrize("name", tuple(_FIXTURES))
def test_native_star_decomposition_equals_the_rational_one(
    name: str, rational: bool
) -> None:
    cell = _embedding_cell(*_FIXTURES[name](), rational)
    try:
        expected: object = _volume_tetrahedra(cell)
    except UnsupportedEmbeddingMap as error:
        expected = str(error)
    native = _native_volume_tetrahedra((cell,))[0]

    if name == "two-components":
        # A star point off the vertex average: the rational proof owns it.
        assert native is None
        assert isinstance(expected, str)
    else:
        assert native == expected
    if name == "bent":
        # Its top face is non-planar: the fan from its exact vertex average.
        assert isinstance(native, tuple) and len(native) == 4 + 5 * 2


def _validity(
    proof: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    polyhedra: tuple[Polyhedron, ...],
    floor: float,
    rational: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.cumsum([0, *(points.shape[0] for points, _ in polyhedra)])
    points = np.vstack(tuple(points for points, _ in polyhedra))
    mesh = CellMesh.from_polyhedra(
        points,
        tuple(
            tuple(tuple(int(offset + index) for index in face) for face in faces)
            for offset, (_, faces) in zip(offsets, polyhedra, strict=False)
        ),
    )
    block = mesh.blocks[0]
    exact = {
        vertex: tuple(_exact(value, rational) for value in points[vertex])
        for vertex in range(points.shape[0])
    }
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    return proof(
        mesh,
        0,
        block,
        np.arange(block.cell_count, dtype=np.int64),
        exact,
        CellValidityPolicy(relative_determinant_floor=floor),
        star_tables=polyhedral_star_tables(connectivity),
    )


@pytest.mark.parametrize("rational", (False, True), ids=("binary64", "rational"))
@pytest.mark.parametrize(
    "floor", (0.0, 1e-6, 0.5), ids=("no-floor", "floor", "high-floor")
)
def test_native_star_validity_equals_the_rational_one(
    floor: float, rational: bool
) -> None:
    polyhedra = tuple(
        build()
        for name, build in _FIXTURES.items()
        if name not in ("self-intersecting", "mixed-orientation")
    )
    native = _validity(_polyhedron_proof, polyhedra, floor, rational)
    expected = _validity(_rational_polyhedron_proof, polyhedra, floor, rational)

    for left, right in zip(native, expected, strict=True):
        np.testing.assert_array_equal(left, right)
    assert native[0].dtype == expected[0].dtype
    # Certified and invalid (inward) cells both occur.
    assert set(native[0].tolist()) >= {0, 1}


def _star_polygon(generator: random.Random) -> tuple[tuple[Fraction, ...], ...]:
    # A star-shaped (often nonconvex) loop with a collinear hanging vertex,
    # embedded in a random rational plane.
    count = generator.randint(3, 9)
    angles = sorted(generator.sample(range(0, 360, 15), count))
    radii = [generator.choice((1, 2, 3)) for _ in angles]
    lattice = [
        (
            Fraction(round(radius * 8 * np.cos(np.radians(angle)))),
            Fraction(round(radius * 8 * np.sin(np.radians(angle)))),
        )
        for radius, angle in zip(radii, angles, strict=True)
    ]
    if generator.random() < 0.5:
        first, second = lattice[0], lattice[1]
        lattice.insert(1, ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2))
    if generator.random() < 0.5:
        lattice.reverse()
    u = tuple(
        Fraction(generator.randint(-3, 3), generator.randint(1, 4)) for _ in range(3)
    )
    v = tuple(
        Fraction(generator.randint(-3, 3), generator.randint(1, 4)) for _ in range(3)
    )
    origin = tuple(Fraction(generator.randint(-5, 5), 3) for _ in range(3))
    return tuple(
        tuple(origin[axis] + x * u[axis] + y * v[axis] for axis in range(3))
        for x, y in lattice
    )


def test_native_face_triangulation_equals_the_rational_ear_clipping() -> None:
    generator = random.Random(11)
    loops = [_star_polygon(generator) for _ in range(300)]
    # Refusals: a repeated vertex, a collinear loop, a nonplanar loop.
    square = _star_polygon(random.Random(3))
    loops.append((*square, square[0]))
    loops.append(tuple((Fraction(t), Fraction(2 * t), Fraction(0)) for t in range(4)))
    loops.append(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(1)),
            (Fraction(0), Fraction(1), Fraction(0)),
        )
    )
    routes = _native_face_routes(loops)
    refused = 0
    for loop, native in zip(loops, routes, strict=True):
        try:
            expected = triangulate_planar_polygon(loop)
        except ValueError:
            expected = None
        if native is None:
            refused += 1
        else:
            assert native == expected
    assert refused >= 2
    assert routes[-3] is None and routes[-2] is None
