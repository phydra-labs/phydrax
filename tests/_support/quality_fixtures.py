from __future__ import annotations

import numpy as np

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry import coordinate_lagrange_element


def reference_quality_mesh(cell_kind: str, count: int = 1) -> CellMesh:
    """Build disjoint translated reference cells with deterministic row identities."""
    reference = np.asarray(reference_cell_topology(cell_kind).vertices, dtype=np.float64)
    shift = 3.0 * np.arange(count, dtype=np.float64)
    points = reference[None] + shift[:, None, None]
    rows = np.arange(count * reference.shape[0], dtype=np.int32).reshape(count, -1)
    return CellMesh(
        points.reshape(-1, reference.shape[1]), (CellBlock("cells", cell_kind, rows),)
    )


def curved_quality_tetrahedron(sign: float = 1.0) -> tuple[CellMesh, CellGeometrySpec]:
    """Build the quadratic map with constant signed determinant sign / 10."""
    element = coordinate_lagrange_element("tetrahedron", 2)

    def physical(reference: np.ndarray) -> np.ndarray:
        x = reference[:, 0] + 0.25 * reference[:, 2]
        y = reference[:, 1] + 0.25 * reference[:, 2]
        return np.column_stack((x, y, sign * (0.1 * reference[:, 2] + y * y)))

    corners = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    mesh = CellMesh(
        physical(corners),
        (
            CellBlock(
                "curved",
                "tetrahedron",
                np.arange(4, dtype=np.int32)[None],
                global_ids=np.asarray([701], dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.arange(element.local_dof_count, dtype=np.int32)[None]},
        physical(np.asarray(element.reference_nodes, dtype=np.float64)),
    )
    return mesh, geometry


def sheared_quality_polyhedron() -> CellMesh:
    """Build an affine cube whose volume is the transform determinant."""
    topology = reference_cell_topology("hexahedron")
    transform = np.asarray(
        ((1.0, 0.13, 0.07), (0.03, 0.9, 0.11), (0.01, 0.05, 1.2)), dtype=np.float64
    )
    points = np.asarray(topology.vertices, dtype=np.float64) @ transform
    return CellMesh.from_polyhedra(
        points, [[np.asarray(face, dtype=np.int32) for face in topology.entities[2]]]
    )
