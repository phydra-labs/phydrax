#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Parity of volume cell incidence with canonical vertex identities."""

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing.providers._native_plc_geometry import volume_cell_entity_rows


def _reference_rows(mesh: CellMesh, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        levels = (np.asarray(connectivity.entities[1]), np.asarray(connectivity.entities[2]))
    else:
        assert isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity))
        levels = (np.asarray(connectivity.edges), np.asarray(connectivity.faces))
    maps = tuple(
        {tuple(sorted(row)): index for index, row in enumerate(level.tolist())}
        for level in levels
    )
    cells = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True):
            cells[int(identifier)] = tuple(
                tuple(
                    mapping[tuple(sorted(int(vertices[index]) for index in entity))]
                    for entity in topology.entities[dimension]
                )
                for dimension, mapping in enumerate(maps, start=1)
            )
    return (
        np.asarray([cells[int(identifier)][0] for identifier in requested], dtype=np.int64),
        np.asarray([cells[int(identifier)][1] for identifier in requested], dtype=np.int64),
    )


def _shared_mesh() -> CellMesh:
    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.float64,
    )
    return CellMesh(
        points,
        (
            CellBlock("a", "simplex:3", [[0, 1, 2, 3]], global_ids=[41]),
            CellBlock("b", "tetrahedron", [[0, 2, 1, 4]], global_ids=[17]),
        ),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_simplex_blocks_shared_face_and_requested_order(reverse: bool) -> None:
    mesh = _shared_mesh()
    requested = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    if reverse:
        requested = requested[::-1]
    actual = volume_cell_entity_rows(mesh, requested)
    expected = _reference_rows(mesh, requested)
    for result, reference in zip(actual, expected, strict=True):
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, reference)
    assert np.intersect1d(actual[1][0], actual[1][1]).size == 1
    if not reverse:
        for result, default in zip(actual, volume_cell_entity_rows(mesh), strict=True):
            np.testing.assert_array_equal(result, default)


@pytest.mark.parametrize("kind", ["tetrahedron", "hexahedron"])
def test_single_volume_block_matches_vertex_reference(kind: str) -> None:
    topology = reference_cell_topology(kind)
    points = np.asarray(topology.vertices, dtype=np.float64)
    mesh = CellMesh(points, (CellBlock("cells", kind, [list(range(len(points)))], global_ids=[9]),))
    requested = np.asarray([9], dtype=np.int64)
    for result, reference in zip(
        volume_cell_entity_rows(mesh, requested), _reference_rows(mesh, requested), strict=True
    ):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize(
    ("ids", "error", "message"),
    [
        (np.asarray([[17, 41]]), TypeError, "integer vector"),
        (np.asarray([17.0, 41.0]), TypeError, "integer vector"),
        (np.asarray([17, 2**63], dtype=np.uint64), ValueError, "fit int64"),
        (np.asarray([17]), ValueError, "cover every cell exactly once"),
        (np.asarray([17, 17]), ValueError, "cover every cell exactly once"),
        (np.asarray([17, 99]), ValueError, "absent from connectivity"),
    ],
)
def test_invalid_requested_cell_ids(ids: np.ndarray, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        volume_cell_entity_rows(_shared_mesh(), ids)
