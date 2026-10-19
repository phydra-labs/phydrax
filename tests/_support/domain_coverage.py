"""Small two-region coverage fixtures with dyadic coordinates and half-unit areas."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    certify_cell_geometry_validity,
    coordinate_lagrange_element,
)
from phydrax.geometry import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
    PiecewiseLinearDomain,
)


SPLIT_SQUARE_POINTS = np.asarray(
    ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0), (0.0, 1.0)),
    dtype=np.float64,
)


def split_square_domain() -> PiecewiseLinearDomain:
    """Declare two half-unit regions and their oriented vertical interface."""
    return PiecewiseLinearDomain(
        SPLIT_SQUARE_POINTS,
        np.asarray(
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)), dtype=np.int64
        ),
        np.asarray(
            ((0, -1), (1, -1), (1, -1), (1, -1), (0, -1), (0, -1), (0, 1)), dtype=np.int64
        ),
        ("left", "right"),
        source_id="split-square",
    )


def mapped_split_square(
    cells: int = 2,
    right_bulge: float = 0.0,
) -> tuple[CellMesh, CellGeometrySpec, GlobalEmbeddingCertificate]:
    """Prepare Q2 square cells with an optional outward right-edge midpoint."""
    loops = (
        np.asarray((0, 1, 4, 5), dtype=np.int32),
        np.asarray((1, 2, 3, 4), dtype=np.int32),
    )[:cells]
    mesh = CellMesh.from_polygons(SPLIT_SQUARE_POINTS, loops)
    element = coordinate_lagrange_element("quadrilateral", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    local = np.stack(
        tuple(
            reference * np.asarray((0.5, 1.0), dtype=np.float64)
            + np.asarray((0.5 * index, 0.0), dtype=np.float64)
            for index in range(cells)
        )
    )
    # The dyadic interior bubble vanishes on every face and preserves signed area.
    local[:, :, 0] += 0.125 * np.prod(reference * (1.0 - reference), axis=1)
    local[-1, np.all(reference == (1.0, 0.5), axis=1), 0] += right_bulge
    name = mesh.blocks[0].name
    geometry = CellGeometrySpec(
        {name: element},
        {
            name: np.arange(local.shape[0] * local.shape[1], dtype=np.int64).reshape(
                local.shape[:2]
            )
        },
        local.reshape(-1, 2),
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    return mesh, geometry, embedding


def split_square_coverage_carrier(
    *,
    mapped: bool,
) -> tuple[CellMesh, CellGeometrySpec, GlobalEmbeddingCertificate, NDArray[np.int64]]:
    """Prepare affine triangles or Q2 cells with the same boundary and region areas."""
    if mapped:
        mesh, geometry, embedding = mapped_split_square()
        return mesh, geometry, embedding, np.asarray((0, 1), dtype=np.int64)
    mesh = CellMesh.from_triangles(
        SPLIT_SQUARE_POINTS,
        np.asarray(((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4)), dtype=np.int32),
    )
    geometry = CellGeometrySpec.affine(mesh)
    regions = np.asarray((0, 0, 1, 1), dtype=np.int64)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    return mesh, geometry, embedding, regions
