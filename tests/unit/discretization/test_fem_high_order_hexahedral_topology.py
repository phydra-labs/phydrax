#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._hexahedral import (
    _EDGES,
    _FACES,
    _quadrilateral_tensor_permutation,
    hexahedral_connectivity,
)


def _two_hex_mesh(*, cells=None, global_ids=(17, 41)):
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 1.0),
        )
    )
    if cells is None:
        cells = (
            (0, 1, 2, 3, 4, 5, 6, 7),
            (1, 8, 9, 2, 5, 10, 11, 6),
        )
    block = phx.discretization.CellBlock(
        "hexes",
        "hexahedron",
        jnp.asarray(cells, dtype=jnp.int32),
        global_ids=jnp.asarray(global_ids, dtype=jnp.int64),
    )
    return phx.discretization.CellMesh(coordinates, (block,))


def _element(order):
    return phx.discretization.fem.ReferenceNodalFamily(
        "hexahedron",
        order,
    ).finite_element()


def _dof_map(mesh, order):
    return phx.discretization.FiniteElementDofMap(mesh, (_element(order),))


def _local_face_trace_dofs(element, local_face):
    face_vertices = _FACES[local_face]
    edge_lookup = {frozenset(edge): index for index, edge in enumerate(_EDGES)}
    face_edges = tuple(
        edge_lookup[
            frozenset((face_vertices[position], face_vertices[(position + 1) % 4]))
        ]
        for position in range(4)
    )
    return (
        tuple(dof for vertex in face_vertices for dof in element.entity_dofs[0][vertex])
        + tuple(dof for edge in face_edges for dof in element.entity_dofs[1][edge])
        + element.entity_dofs[2][local_face]
    )


def test_q1_hex_routing_remains_vertex_compatible():
    mesh = _two_hex_mesh()
    dof_map = phx.discretization.FiniteElementDofMap(
        mesh,
        (phx.discretization.lagrange_element("hexahedron", 1),),
    )

    assert dof_map.association == "vertex"
    assert dof_map.global_dof_count == 12
    assert np.array_equal(dof_map.cell_dofs[0], mesh.blocks[0].vertices)
    assert np.array_equal(dof_map.dof_coordinates, mesh.coordinates)


@pytest.mark.parametrize(
    ("degree", "global_count", "entity_counts", "boundary_count"),
    (
        (2, 45, (12, 20, 11, 2), 42),
        (3, 112, (12, 40, 44, 16), 92),
    ),
)
def test_high_order_hex_global_counts_layout_and_boundary_mask(
    degree,
    global_count,
    entity_counts,
    boundary_count,
):
    mesh = _two_hex_mesh()
    element = _element(degree)
    dof_map = phx.discretization.FiniteElementDofMap(mesh, (element,))
    field = phx.discretization.FiniteElementFieldSpec("u", element)
    prepared = phx.discretization.FiniteElementPlan(mesh, field).prepare()

    assert dof_map.association == "entity"
    assert dof_map.global_dof_count == global_count
    assert dof_map.entity_dof_counts == entity_counts
    assert np.count_nonzero(dof_map.boundary_dof_mask) == boundary_count
    assert prepared.field_spaces[0].layout.names == (
        "vertices",
        "edges",
        "faces",
        "cells",
    )


def test_shared_hex_face_trace_is_single_valued_at_p3():
    mesh = _two_hex_mesh()
    element = _element(3)
    dof_map = phx.discretization.FiniteElementDofMap(mesh, (element,))
    cell_faces = np.asarray(mesh.connectivity.cell_faces)
    shared_face = int(
        np.flatnonzero(np.asarray(mesh.connectivity.face_cell_counts) == 2)[0]
    )
    local_faces = tuple(
        int(np.flatnonzero(cell_faces[cell] == shared_face)[0]) for cell in range(2)
    )
    trace_routes = tuple(
        np.asarray(dof_map.cell_dofs[0])[cell][
            np.asarray(_local_face_trace_dofs(element, local_face))
        ]
        for cell, local_face in enumerate(local_faces)
    )

    assert len(trace_routes[0]) == 16
    assert set(trace_routes[0].tolist()) == set(trace_routes[1].tolist())


def test_all_quadrilateral_face_orientations_have_explicit_tensor_permutations():
    expected = {
        (0, 1, 2, 3): (0, 1, 2, 3, 4, 5),
        (1, 2, 3, 0): (4, 2, 0, 5, 3, 1),
        (2, 3, 0, 1): (5, 4, 3, 2, 1, 0),
        (3, 0, 1, 2): (1, 3, 5, 0, 2, 4),
        (0, 3, 2, 1): (0, 2, 4, 1, 3, 5),
        (3, 2, 1, 0): (2, 1, 0, 5, 4, 3),
        (2, 1, 0, 3): (5, 3, 1, 4, 2, 0),
        (1, 0, 3, 2): (3, 4, 5, 0, 1, 2),
    }
    for cycle, tensor_permutation in expected.items():
        cell = np.empty((8,), dtype=np.int32)
        for local_vertex, vertex in zip(_FACES[0], cycle, strict=True):
            cell[local_vertex] = vertex
            cell[local_vertex + 4] = vertex + 4
        connectivity = hexahedral_connectivity(cell[None, :], 8)

        assert tuple(connectivity.cell_face_vertex_permutations[0, 0]) == cycle
        assert tuple(connectivity.cell_face_permutations(2, 3)[0, 0]) == (
            tensor_permutation
        )
        assert tuple(_quadrilateral_tensor_permutation(cycle, 2, 3)) == tensor_permutation


def test_high_order_hex_routing_and_content_identity_ignore_cell_row_order():
    mesh = _two_hex_mesh()
    cells = np.asarray(mesh.blocks[0].vertices)
    reordered = _two_hex_mesh(cells=cells[::-1], global_ids=(41, 17))
    element = _element(3)
    original_map = phx.discretization.FiniteElementDofMap(mesh, (element,))
    reordered_map = phx.discretization.FiniteElementDofMap(reordered, (element,))

    assert mesh.topology_id == reordered.topology_id
    assert original_map.dof_map_id == reordered_map.dof_map_id
    assert np.array_equal(original_map.cell_dofs[0], reordered_map.cell_dofs[0][::-1])
    assert np.array_equal(
        original_map.boundary_dof_mask,
        reordered_map.boundary_dof_mask,
    )


def test_high_order_hex_dof_coordinates_reproduce_each_affine_cell():
    mesh = _two_hex_mesh()
    element = _element(3)
    dof_map = phx.discretization.FiniteElementDofMap(mesh, (element,))
    gathered = np.asarray(dof_map.dof_coordinates)[np.asarray(dof_map.cell_dofs[0])]
    expected = np.stack(
        (
            np.asarray(element.reference_nodes),
            np.asarray(element.reference_nodes) + np.asarray((1.0, 0.0, 0.0)),
        )
    )

    assert np.allclose(gathered, expected, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        dof_map.evaluate_coordinates(mesh, mesh.coordinates),
        dof_map.dof_coordinates,
    )


def test_compatible_anisotropic_hex_traces_route_globally():
    mesh = _two_hex_mesh()
    dof_map = _dof_map(mesh, (2, 3, 4))

    assert dof_map.global_dof_count == 100
    assert dof_map.entity_dof_counts == (12, 38, 38, 12)
    assert np.count_nonzero(dof_map.boundary_dof_mask) == 82


def test_incompatible_anisotropic_hex_trace_requires_a_mortar():
    mesh = _two_hex_mesh(
        cells=(
            (0, 1, 2, 3, 4, 5, 6, 7),
            (8, 1, 5, 10, 9, 2, 6, 11),
        )
    )

    with pytest.raises(ValueError, match="mortar is required"):
        _dof_map(mesh, (2, 3, 4))
