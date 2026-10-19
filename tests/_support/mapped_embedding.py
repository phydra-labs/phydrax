# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Independent polynomial and oriented-mesh fixtures for embedding certificates."""

from collections.abc import Callable
from fractions import Fraction

import numpy as np

from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.meshing._curving import straight_geometry


type HostCoordinateTransform = Callable[[np.ndarray], np.ndarray]


def p2_geometry(mesh: CellMesh, transform: HostCoordinateTransform) -> CellGeometrySpec:
    """Map shared P2 nodes through a declared host coordinate transformation."""
    straight = straight_geometry(mesh, 2)
    return CellGeometrySpec(
        dict(zip(straight.block_names, straight.elements, strict=True)),
        dict(zip(straight.block_names, straight.geometry_dofs, strict=True)),
        transform(np.asarray(straight.coordinates, dtype=np.float64)),
    )


def certify_mapped_fixture(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> GlobalEmbeddingCertificate:
    """Certify an embedding after requiring all fixture cell maps to be valid."""
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    return certify_global_embedding(mesh, geometry, validity)


def shear(points: np.ndarray) -> np.ndarray:
    """Apply the unit-Jacobian polynomial shear x += y(1-y)/4."""
    result = points.copy()
    result[:, 0] += 0.25 * points[:, 1] * (1.0 - points[:, 1])
    return result


def square_with_hole() -> CellMesh:
    grid = np.linspace(0.0, 3.0, 4, dtype=np.float64)
    points = np.stack(np.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    index = np.arange(16, dtype=np.int64).reshape(4, 4)
    triangles = []
    for i, j in np.ndindex(3, 3):
        if (i, j) == (1, 1):
            continue
        a, b = int(index[i, j]), int(index[i + 1, j])
        c, d = int(index[i + 1, j + 1]), int(index[i, j + 1])
        triangles.extend(((a, b, c), (a, c, d)))
    return CellMesh.from_triangles(points, np.asarray(triangles, dtype=np.int64))


def complex_cube_map(x_offset: int) -> tuple[Polynomial, Polynomial]:
    """Expand (x + iy)^3 for x = x_offset + u, y = 4v - 2 independently."""
    x: Polynomial = {(0, 0): Fraction(x_offset), (1, 0): Fraction(1)}
    y: Polynomial = {(0, 0): Fraction(-2), (0, 1): Fraction(4)}

    def product(left: Polynomial, right: Polynomial) -> Polynomial:
        result: Polynomial = {}
        for a, value_a in left.items():
            for b, value_b in right.items():
                index = (a[0] + b[0], a[1] + b[1])
                result[index] = result.get(index, Fraction(0)) + value_a * value_b
        return {index: value for index, value in result.items() if value}

    def difference(left: Polynomial, right: Polynomial, factor: int) -> Polynomial:
        result = dict(left)
        for index, value in right.items():
            result[index] = result.get(index, Fraction(0)) - factor * value
        return {index: value for index, value in result.items() if value}

    real = difference(product(product(x, x), x), product(x, product(y, y)), 3)
    imaginary = difference(
        {index: 3 * value for index, value in product(product(x, x), y).items()},
        product(product(y, y), y),
        1,
    )
    return real, imaginary
