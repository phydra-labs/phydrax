from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    CellGeometryTransitionEvidence,
)
from phydrax.meshing._lineage import VertexInterpolationStencil
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._topology_edit import assemble_topology_edit


class NestedMaterialGeometry(NamedTuple):
    transition: CellGeometryTransition
    vertex_stencil: VertexInterpolationStencil
    source_regions: NDArray[np.int64]
    target_regions: NDArray[np.int64]


def nested_material_geometry() -> NestedMaterialGeometry:
    """Native two-to-twelve-cell restriction with a nonrepresentable face midpoint.

    Coordinates retain the archived-layer-core prism6674 and one opposite-side
    point. This is an explicitly authored smaller model, not inferred ancestry
    of the complete qualification case.
    """
    hexadecimal = (
        ("0x1.7000000000000p-1", "0x1.680485e62e1e5p-1", "0x1.1c7615a783ce8p-4"),
        ("0x1.8000000000000p-1", "0x1.7004def7e2e59p-1", "0x1.0faf379d8ebbbp-4"),
        ("0x1.7000000000000p-1", "0x1.7004def7e2e59p-1", "0x1.0faf379d8ebbbp-4"),
        ("0x1.7000000000000p-1", "0x1.68090bcc5c3cap-1", "0x1.45b8f81bd469dp-4"),
        ("0x1.8000000000000p-1", "0x1.7009bdefc5cb2p-1", "0x1.38f808d4b7110p-4"),
        ("0x1.7000000000000p-1", "0x1.7009bdefc5cb2p-1", "0x1.38f808d4b7110p-4"),
        ("0x1.7800000000000p-1", "0x1.7009bdefc5cb2p-1", "0x1.7668acabc14e7p-4"),
    )
    coordinates = np.asarray(
        [[float.fromhex(value) for value in point] for point in hexadecimal],
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "prisms",
                "prism",
                np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int32),
                global_ids=np.asarray([101], dtype=np.int64),
            ),
            CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray([[6, 3, 5, 4]], dtype=np.int32),
                global_ids=np.asarray([303], dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    refinement = adapt_mixed_mesh(
        mesh,
        refine_cell_ids=np.asarray([101], dtype=np.int64),
        source_geometry=geometry,
        cell_classes=np.asarray([0, 1], dtype=np.int64),
    )
    target, _, stencil = assemble_topology_edit(
        mesh, refinement.edit, numeric_version="nested-material"
    )
    if refinement.target_geometry is None:
        raise ValueError("Native refinement lost its coordinate geometry.")
    transition = CellGeometryTransition(
        mesh,
        geometry,
        target,
        refinement.target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction",
            reference_map_count=sum(
                len(group.target_cell_ids) for group in refinement.reference_maps
            ),
        ),
        reference_map=refinement.reference_maps,
    )
    classes = {
        cell.cell_id: cell.cell_class for cell in refinement.hierarchy.active_cells
    }
    target_regions = np.asarray(
        [
            classes[int(identifier)]
            for block in target.blocks
            for identifier in np.asarray(block.global_ids)
        ],
        dtype=np.int64,
    )
    return NestedMaterialGeometry(
        transition, stencil, np.asarray([0, 1], dtype=np.int64), target_regions
    )
