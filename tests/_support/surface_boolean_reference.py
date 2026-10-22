#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Surface-solid builders and independent host reference geometry.

Unit-cube tables preserve oriented source-face order. Sphere refinement shares
one normalized midpoint per undirected edge. Rational volume is independent
of the Boolean implementation; cone intersection uses a different rational
polytope-clipping algorithm rather than fragment classification.
"""

import math
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike

from phydrax import SpatialCoordinateContract
from phydrax._meshcore import tetrahedron_intersection_moments
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel
from tests._support.surface_arrangements import CUBE_FACES, CUBE_VERTICES


def rotation_matrix(axis: ArrayLike, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    cross = np.asarray(
        (
            (0.0, -unit[2], unit[1]),
            (unit[2], 0.0, -unit[0]),
            (-unit[1], unit[0], 0.0),
        )
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross


def surface_model(vertices: np.ndarray, faces: np.ndarray, name: str) -> SurfaceModel:
    return SurfaceModel.from_triangles(
        vertices,
        faces,
        SurfaceMetadata(
            source_id=name,
            source_revision="0",
            coordinate_contract=SpatialCoordinateContract.si(),
            provenance=("unit-test",),
        ),
    )


def cube_surface(
    offset: ArrayLike = (0.0, 0.0, 0.0),
    size: float = 1.0,
    rotation: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    centered = (CUBE_VERTICES - 0.5) * size
    if rotation is not None:
        centered = centered @ np.asarray(rotation).T
    return centered + 0.5 * size + np.asarray(offset, dtype=np.float64), CUBE_FACES


def icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    points = [
        np.asarray(point, dtype=np.float64)
        for point in (
            (-1, golden, 0),
            (1, golden, 0),
            (-1, -golden, 0),
            (1, -golden, 0),
            (0, -1, golden),
            (0, 1, golden),
            (0, -1, -golden),
            (0, 1, -golden),
            (golden, 0, -1),
            (golden, 0, 1),
            (-golden, 0, -1),
            (-golden, 0, 1),
        )
    ]
    points = [point / np.linalg.norm(point) for point in points]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    for _ in range(level):
        middles: dict[tuple[int, int], int] = {}

        def middle(first: int, second: int) -> int:
            key = (min(first, second), max(first, second))
            if key not in middles:
                point = points[first] + points[second]
                points.append(point / np.linalg.norm(point))
                middles[key] = len(points) - 1
            return middles[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    return np.asarray(points), np.asarray(faces, dtype=np.int64)


def signed_volume_reference(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Return oriented volume from rational determinants of binary64 coordinates.

    The divergence identity is evaluated in Fraction arithmetic, independently
    of the product's floating cross-product/reduction implementation.
    """
    volume = Fraction(0)
    for face in faces:
        a, b, c = (
            tuple(Fraction.from_float(float(value)) for value in vertices[index])
            for index in face
        )
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        volume += determinant / 6
    return float(volume)


def surface_area_reference(vertices: np.ndarray, faces: np.ndarray) -> float:
    corners = vertices[faces]
    return float(
        np.sum(
            np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
        )
        / 2.0
    )


def cone_tetrahedra(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Tetrahedra coning a closed star-shaped surface from its vertex centroid."""

    apex = np.mean(vertices, axis=0)
    return np.concatenate(
        (np.repeat(apex[None, None], faces.shape[0], axis=0), vertices[faces]), axis=1
    )


def clipped_volume_reference(first: np.ndarray, second: np.ndarray) -> float:
    """Independent reference: rational pairwise tetrahedron clipping of two cones."""

    rows, columns = np.meshgrid(
        np.arange(first.shape[0]), np.arange(second.shape[0]), indexing="ij"
    )
    volume, _, status = tetrahedron_intersection_moments(
        first[rows.reshape(-1)], second[columns.reshape(-1)]
    )
    assert np.all(status == 0)
    return float(np.sum(volume))
