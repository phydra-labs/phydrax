"""Quadratic shear sources with stable mixed-block identities for overset queries."""

from collections.abc import Sequence

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing import certify_cell_mesh
from phydrax.meshing._mesh_part import MeshPart


def mapped_source(kinds: Sequence[str]) -> MeshPart:
    """Return separated cells with x += 0.1*y*z and their quadratic source bank."""
    blocks, corners, elements, dofs, coordinates = [], [], {}, {}, []
    for index, kind in enumerate(kinds):
        name = f"block{index}"
        element = coordinate_lagrange_element(kind, 2)
        reference = np.asarray(element.reference_nodes)
        physical = reference.copy()
        physical[:, 0] += 0.1 * reference[:, 1] * reference[:, 2] + 2 * index
        vertex_points = np.asarray(
            reference_cell_topology(kind).vertices, dtype=np.float64
        )
        vertex_points[:, 0] += 0.1 * vertex_points[:, 1] * vertex_points[:, 2] + 2 * index
        blocks.append(
            CellBlock(
                name,
                kind,
                np.arange(len(vertex_points))[None] + sum(len(p) for p in corners),
                global_ids=np.asarray([90 - 17 * index]),
            )
        )
        corners.append(vertex_points)
        elements[name] = element
        dofs[name] = np.arange(len(reference))[None] + sum(len(p) for p in coordinates)
        coordinates.append(physical)
    mesh = (
        CellMesh.from_mixed_3d(np.concatenate(corners), tuple(blocks), polyhedra={})
        if "prism" in kinds
        else CellMesh(np.concatenate(corners), tuple(blocks))
    )
    geometry = CellGeometrySpec(elements, dofs, np.concatenate(coordinates))
    return MeshPart(
        "source",
        certify_cell_mesh(mesh, SpatialCoordinateContract.si(), geometry=geometry),
    )
