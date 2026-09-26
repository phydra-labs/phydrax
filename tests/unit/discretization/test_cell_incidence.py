import numpy as np
import pytest

from phydrax.discretization import (
    CellBlock,
    CellMesh,
    EntitySet,
    OrientedIncidence,
    polygonal_cell_complex,
    polygonal_connectivity,
    polyhedral_cell_complex,
    polyhedral_connectivity,
    tetrahedral_cell_complex,
    tetrahedral_connectivity,
)
from phydrax.discretization._hexahedral import hexahedral_connectivity
from phydrax.sparse import EdgeRelation


def _assert_boundary_of_boundary_vanishes(topology):
    boundaries = [incidence.scipy_boundary() for incidence in topology.incidences]
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        assert not np.any((lower @ upper).toarray())


@pytest.mark.parametrize("scale", [1, 2**40])
def test_oriented_incidence_is_canonical_and_rejects_repeated_entries(scale):
    vertices = EntitySet("vertices", 0, np.asarray([7, 3, 5]) * scale)
    edges = EntitySet("edges", 1, np.asarray([4, 1]) * scale)
    sources, targets, signs = [0, 1, 1, 2], [0, 0, 1, 1], [-1.0, 1.0, -1.0, 1.0]
    order = [3, 0, 2, 1]

    def incidence(route_order):
        relation = EdgeRelation(
            np.asarray(sources)[route_order],
            np.asarray(targets)[route_order],
            source_size=3,
            target_size=2,
        )
        return OrientedIncidence(
            1, vertices, edges, relation, np.asarray(signs)[route_order]
        )

    assert incidence(order).incidence_id == incidence(np.arange(4)).incidence_id
    repeated = EdgeRelation([0, 1, 1, 0], [0, 0, 1, 0], source_size=3, target_size=2)
    with pytest.raises(ValueError, match="incidence pairs must be unique"):
        OrientedIncidence(1, vertices, edges, repeated, [-1.0, 1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="Active entity IDs must be unique"):
        EntitySet("vertices", 0, np.asarray([7, 3, 7]) * scale)


def test_polygonal_incidence_orients_mixed_blocks_and_rejects_defects():
    triangles = np.asarray([[0, 1, 3], [1, 4, 3]])
    quadrilaterals = np.asarray([[1, 2, 5, 4]])
    pentagons = np.asarray([[3, 4, 5, 6, 7]])
    connectivity = polygonal_connectivity(
        triangles, quadrilaterals, 8, polygons=(pentagons,)
    )
    topology = polygonal_cell_complex(triangles, quadrilaterals, 8, polygons=(pentagons,))
    counts = np.asarray(connectivity.edge_cell_counts)
    assert np.count_nonzero(counts == 2) == 4
    _assert_boundary_of_boundary_vanishes(topology)
    boundary = topology.incidences[1].scipy_boundary() @ np.ones(4)
    np.testing.assert_array_equal(boundary != 0.0, counts == 1)

    with pytest.raises(ValueError, match="edge-manifold"):
        polygonal_connectivity(np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]]), None, 5)
    with pytest.raises(ValueError, match="opposite orientation"):
        polygonal_connectivity(np.asarray([[0, 1, 2]]), np.asarray([[2, 3, 4, 1]]), 5)
    with pytest.raises(ValueError, match="duplicate cells"):
        polygonal_connectivity(
            np.asarray([[0, 1, 2]]), None, 3, polygons=(np.asarray([[2, 0, 1]]),)
        )


def test_tetrahedral_incidence_rejects_nonmanifold_faces_and_flipped_cells():
    with pytest.raises(ValueError, match="face-manifold"):
        tetrahedral_connectivity(
            np.asarray([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]]), 6
        )
    with pytest.raises(ValueError, match="opposite orientation"):
        tetrahedral_connectivity(np.asarray([[0, 1, 2, 3], [0, 1, 2, 4]]), 5)


def test_tetrahedral_entity_ids_ignore_cell_order_and_even_vertex_permutations():
    tetrahedra = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 1, 5], [0, 3, 2, 6]])
    vertex_ids = np.asarray([50, 30, 90, 10, 70, 20, 40])
    cell_ids = np.asarray([11, 22, 33, 44])
    order = np.asarray([2, 0, 3, 1])
    even = np.asarray([[1, 2, 0, 3], [0, 1, 2, 3], [2, 0, 1, 3], [1, 0, 3, 2]])
    shuffled = np.take_along_axis(tetrahedra[order], even, axis=1)

    def oriented_entities(cells, identifiers):
        connectivity = tetrahedral_connectivity(cells, 7)
        topology = tetrahedral_cell_complex(
            cells, 7, vertex_global_ids=vertex_ids, cell_global_ids=identifiers
        )
        _assert_boundary_of_boundary_vanishes(topology)
        faces = vertex_ids[np.asarray(connectivity.faces)]
        face_ids = np.asarray(topology.entities(2).entity_ids)
        edge_ids = np.asarray(topology.entities(1).entity_ids)
        edges = vertex_ids[np.asarray(connectivity.edges)]
        face_signs = {
            (int(identifiers[cell]), tuple(sorted(faces[face]))): float(sign)
            for cell, (row, signs) in enumerate(
                zip(
                    np.asarray(connectivity.cell_faces),
                    np.asarray(connectivity.cell_face_signs),
                    strict=True,
                )
            )
            for face, sign in zip(row, signs, strict=True)
        }
        return (
            {tuple(sorted(key)): int(value) for key, value in zip(faces, face_ids)},
            {tuple(sorted(key)): int(value) for key, value in zip(edges, edge_ids)},
            face_signs,
        )

    assert oriented_entities(tetrahedra, cell_ids) == oriented_entities(
        shuffled, cell_ids[order]
    )


def test_hexahedral_incidence_rejects_incompatible_nonmanifold_and_mirrored_cells():
    lower = np.arange(8)
    upper = np.arange(8) + 4
    stacked = hexahedral_connectivity(np.stack((lower, upper)), 12)
    counts = np.asarray(stacked.face_cell_counts)
    assert np.count_nonzero(counts == 2) == 1
    shared = int(np.flatnonzero(counts == 2)[0])
    signs = np.asarray(stacked.cell_face_signs)[np.asarray(stacked.cell_faces) == shared]
    np.testing.assert_array_equal(np.sort(signs), (-1.0, 1.0))
    np.testing.assert_array_equal(np.asarray(stacked.boundary_faces), counts == 1)

    with pytest.raises(ValueError, match="cycles are incompatible"):
        hexahedral_connectivity(np.stack((lower, upper[[1, 0, 2, 3, 4, 5, 6, 7]])), 12)
    with pytest.raises(ValueError, match="Non-manifold hexahedral face"):
        hexahedral_connectivity(
            np.stack((lower, upper, np.concatenate((upper[:4], upper[4:] + 4)))), 16
        )
    with pytest.raises(ValueError, match="opposite orientation"):
        hexahedral_connectivity(np.stack((lower, upper[[1, 0, 3, 2, 5, 4, 7, 6]])), 12)


def test_polyhedral_shared_faces_must_be_reversed_and_first_cell_defect_wins():
    first = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    # The shared face (1, 2, 3) appears reversed and rotated as (3, 2, 1).
    second = ((3, 2, 1), (3, 1, 4), (2, 3, 4), (1, 2, 4))
    connectivity = polyhedral_connectivity((first, second), 5)
    counts = np.asarray(connectivity.face_cell_counts)
    assert np.count_nonzero(counts == 2) == 1
    shared = int(np.flatnonzero(counts == 2)[0])
    values = np.asarray(connectivity.cell_face_values)
    np.testing.assert_array_equal(
        np.sort(np.asarray(connectivity.cell_face_sign_values)[values == shared]),
        (-1.0, 1.0),
    )
    topology = polyhedral_cell_complex(connectivity)
    _assert_boundary_of_boundary_vanishes(topology)
    boundary = topology.incidences[2].scipy_boundary() @ np.ones(2)
    np.testing.assert_array_equal(boundary != 0.0, counts == 1)

    inward = tuple(face[::-1] for face in second)
    with pytest.raises(ValueError, match="Shared polyhedral faces must have opposite"):
        polyhedral_connectivity((first, inward), 5)
    with pytest.raises(ValueError, match="closed oriented two-manifold"):
        polyhedral_connectivity((first, ((1, 2, 3),) + second[1:]), 5)

    three_faces = first[:3]
    non_simple = first[:3] + ((2, 0, 3, 0),)
    with pytest.raises(ValueError, match="at least four faces"):
        polyhedral_connectivity((three_faces, non_simple), 5)
    with pytest.raises(ValueError, match="must be simple"):
        polyhedral_connectivity((non_simple, three_faces), 5)


def test_mixed_standard_blocks_share_one_oriented_complex():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 2.0],
            [0.5, -0.5, 1.3],
            [2.0, 0.5, 1.0],
            [2.0, 0.5, 0.0],
        ]
    )
    blocks = (
        CellBlock("tetrahedra", "tetrahedron", [[4, 5, 8, 9]], global_ids=[3]),
        CellBlock("pyramids", "pyramid", [[4, 5, 6, 7, 8]], global_ids=[2]),
        CellBlock("prisms", "prism", [[5, 6, 10, 1, 2, 11]], global_ids=[1]),
        CellBlock("hexahedra", "hexahedron", [list(range(8))], global_ids=[0]),
    )
    mesh = CellMesh(points, blocks)
    counts = np.asarray(mesh.connectivity.face_cell_counts)
    assert mesh.connectivity.face_count == 6 + 5 + 5 + 4 - 3
    assert np.count_nonzero(counts == 2) == 3
    _assert_boundary_of_boundary_vanishes(mesh.topology)
    boundary = mesh.topology.incidences[2].scipy_boundary() @ np.ones(4)
    np.testing.assert_array_equal(boundary != 0.0, counts == 1)
    np.testing.assert_array_equal(
        np.asarray(mesh.connectivity.boundary_faces), counts == 1
    )

    moved = mesh.with_coordinates(points + 0.5, numeric_version="moved")
    assert moved.topology_id == mesh.topology_id
    assert moved.geometry_id != mesh.geometry_id
