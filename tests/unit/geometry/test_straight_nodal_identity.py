from __future__ import annotations

import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization.fem import (
    canonical_nodal_entity_identity,
    lagrange_element,
    lagrange_reference_node_data,
)
from phydrax.meshing._curving import _straight_geometry


def _expected_source_map_id(
    mesh: CellMesh, degree: int, geometry: CellGeometrySpec
) -> str:
    """Derive each occurrence's token independently, without identity reuse."""
    linear = CellGeometrySpec(
        {block.name: lagrange_element(block.cell_kind, 1) for block in mesh.blocks},
        {block.name: np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks},
        np.asarray(mesh.coordinates, dtype=np.float64),
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    tokens = []
    for block in mesh.blocks:
        nodes = lagrange_reference_node_data(lagrange_element(block.cell_kind, degree))
        for vertices, cell_id in zip(
            np.asarray(block.vertices, dtype=np.int64).tolist(),
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            strict=True,
        ):
            ids = tuple(int(vertex_ids[index]) for index in vertices)
            for local_dof, node in enumerate(nodes):
                identity, orientation = canonical_nodal_entity_identity(
                    node, ids, cell_global_id=int(cell_id)
                )
                tokens.append(
                    canonical_fingerprint(
                        {
                            "block": block.name,
                            "cell_id": int(cell_id),
                            "local_dof": local_dof,
                            "identity": identity,
                            "orientation": orientation,
                        }
                    )
                )
    return canonical_fingerprint(
        {
            "kind": "exact-coordinate-bank-map",
            "mesh_id": mesh.mesh_id,
            "topology_id": mesh.topology_id,
            "numeric_version": mesh.numeric_version,
            "source_id": mesh.mesh_id,
            "source_revision": mesh.numeric_version,
            "identity_context": (
                "straight-p1-mesh-map",
                str(degree),
                cell_geometry_id(linear),
                *tokens,
            ),
            "exact_coordinate_bank": geometry.source_coordinates().coefficient_payloads(),
        }
    )


@pytest.mark.parametrize(
    "kind",
    ("triangle", "quadrilateral", "tetrahedron", "prism", "pyramid", "hexahedron"),
)
@pytest.mark.parametrize("degree", (2, 3))
def test_straight_nodal_source_identity(kind: str, degree: int) -> None:
    points = np.asarray(
        reference_cell_topology(kind).vertices, dtype=np.float64
    ) * np.float64(1.25)
    block = CellBlock(
        'a"é\\z',
        kind,
        np.arange(len(points), dtype=np.int64)[None, :],
        global_ids=np.asarray((43,), dtype=np.int64),
    )
    mesh = CellMesh(points, (block,))
    geometry = _straight_geometry(mesh, degree)
    assert geometry.source_map_id == _expected_source_map_id(mesh, degree, geometry)


@pytest.mark.parametrize("degree", (2, 3))
@pytest.mark.parametrize(
    ("kind", "points", "cells", "vertex_ids"),
    (
        pytest.param(
            "triangle",
            ((0, 0), (1, 0), (0, 1), (0, 0), (1, 0), (0, 1)),
            ((0, 1, 2), (3, 4, 5)),
            (10, 11, 12, 20, 21, 22),
            id="coincident-distinct",
        ),
        pytest.param(
            "triangle",
            ((0, 0), (1, 0), (0, 1), (1, -1)),
            ((0, 1, 2), (1, 0, 3)),
            (10, 11, 12, 20),
            id="reversed-shared-edge",
        ),
        pytest.param(
            "tetrahedron",
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
            ((0, 1, 2, 3), (2, 1, 0, 4)),
            (10, 11, 12, 13, 20),
            id="reversed-shared-face",
        ),
    ),
)
def test_multicell_nodal_source_identity(
    kind: str,
    points: tuple[tuple[int, ...], ...],
    cells: tuple[tuple[int, ...], ...],
    vertex_ids: tuple[int, ...],
    degree: int,
) -> None:
    block = CellBlock(
        "cells",
        kind,
        np.asarray(cells, dtype=np.int64),
        global_ids=np.asarray((80, 81), dtype=np.int64),
    )
    mesh = CellMesh(
        np.asarray(points, dtype=np.float64),
        (block,),
        vertex_global_ids=np.asarray(vertex_ids, dtype=np.int64),
    )
    geometry = _straight_geometry(mesh, degree)
    assert geometry.source_map_id == _expected_source_map_id(mesh, degree, geometry)
