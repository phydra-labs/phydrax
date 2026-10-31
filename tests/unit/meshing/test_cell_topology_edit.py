from fractions import Fraction

import numpy as np

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._nested_reference_types import exact_reference_cell_vertices
from phydrax.discretization._reference_cell import (
    facet_orientation_between,
    reference_cell_topology,
)
from phydrax.meshing._lineage import EntityLineageKind
from phydrax.meshing._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    EntityRelations,
    reference_cell_maps_from_edit,
    SharedFaceWitnesses,
    TopologyEditBlock,
)


def _empty_relations(dimension: int, key_width: int) -> EntityRelations:
    return EntityRelations(
        dimension,
        np.zeros((0, key_width), dtype=np.int64),
        np.zeros((0, key_width), dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
    )


def _facet_index(kind: str, vertices: tuple[int, ...], wanted: set[int]) -> int:
    topology = reference_cell_topology(kind)
    for index, face in enumerate(topology.entities[topology.dimension - 1]):
        if {vertices[position] for position in face} == wanted:
            return index
    raise ValueError("The requested face is not a reference-cell facet.")


def test_mixed_quad_triangle_edit_retains_ids_and_oriented_shared_face() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ),
        dtype=np.float64,
    )
    vertex_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    source = CellMesh(
        coordinates,
        (
            CellBlock(
                "triangles",
                "triangle",
                np.asarray(
                    ((0, 1, 3), (0, 3, 2), (1, 4, 3), (4, 5, 3)),
                    dtype=np.int32,
                ),
                global_ids=np.asarray((1, 2, 3, 4), dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
    )
    quad = np.asarray(((0, 1, 3, 2),), dtype=np.int32)
    triangles = np.asarray(((3, 1, 4), (4, 5, 3)), dtype=np.int32)
    left = tuple(int(value) for value in quad[0])
    right = tuple(int(value) for value in triangles[0])
    left_facet = _facet_index("quadrilateral", left, {1, 3})
    right_facet = _facet_index("triangle", right, {1, 3})
    left_order = reference_cell_topology("quadrilateral").entities[1][left_facet]
    right_order = reference_cell_topology("triangle").entities[1][right_facet]
    action = facet_orientation_between(
        tuple(left[index] for index in left_order),
        tuple(right[index] for index in right_order),
    )
    assert action.permutation == (1, 0)
    empty_vertices = _empty_relations(0, 1)
    empty_edges = _empty_relations(1, 2)
    cell_lineage = EntityRelations(
        2,
        np.asarray(((1,), (2,)), dtype=np.int64),
        np.asarray(((100,), (100,)), dtype=np.int64),
        np.full((2,), int(EntityLineageKind.COARSENED_INTO), dtype=np.int32),
    )
    edit = CellTopologyEdit(
        coordinates,
        vertex_ids,
        (
            TopologyEditBlock(
                "quads",
                "quadrilateral",
                None,
                quad,
                np.asarray((100,), dtype=np.int64),
            ),
            TopologyEditBlock(
                "triangles",
                "triangle",
                "triangle",
                triangles,
                np.asarray((3, 4), dtype=np.int64),
            ),
        ),
        np.broadcast_to(vertex_ids[:, None], (vertex_ids.size, 1)).copy(),
        np.ones((vertex_ids.size, 1), dtype=np.float64),
        np.ones((vertex_ids.size, 1), dtype=np.bool_),
        (empty_vertices, empty_edges, cell_lineage),
        shared_faces=SharedFaceWitnesses(
            np.asarray((100,), dtype=np.int64),
            np.asarray((left_facet,), dtype=np.int32),
            np.asarray((3,), dtype=np.int64),
            np.asarray((right_facet,), dtype=np.int32),
            np.asarray((action.permutation,), dtype=np.int32),
        ),
    )

    target, lineage, stencil = assemble_topology_edit(
        source, edit, numeric_version="mixed-face-edit"
    )

    assert tuple(block.cell_kind for block in target.blocks) == (
        "triangle",
        "quadrilateral",
    )
    np.testing.assert_array_equal(target.vertex_global_ids, vertex_ids)
    np.testing.assert_array_equal(stencil.target_global_ids, vertex_ids)
    cells = lineage.entities[target.topological_dimension]
    records = {
        (int(source_id), int(target_id), EntityLineageKind(int(kind)))
        for source_id, target_id, kind in zip(
            np.asarray(cells.source_global_ids),
            np.asarray(cells.target_global_ids),
            np.asarray(cells.relation_kinds),
            strict=True,
        )
    }
    assert (1, 100, EntityLineageKind.COARSENED_INTO) in records
    assert (2, 100, EntityLineageKind.COARSENED_INTO) in records
    assert (3, 3, EntityLineageKind.PRESERVED) in records
    assert (4, 4, EntityLineageKind.PRESERVED) in records


def test_reference_cell_maps_from_edit_preserves_exact_vertex_stencils() -> None:
    source = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "cells",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((42,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((11, 13, 17), dtype=np.int64),
    )
    target_ids = np.asarray((11, 13, 17, 99), dtype=np.int64)
    blocks = (
        TopologyEditBlock(
            "cells",
            "triangle",
            "triangle",
            np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32),
            np.asarray((100, 101), dtype=np.int64),
        ),
    )
    cell_relations = EntityRelations(
        2,
        np.asarray(((42,), (42,)), dtype=np.int64),
        np.asarray(((100,), (101,)), dtype=np.int64),
        np.full((2,), int(EntityLineageKind.REFINED_FROM), dtype=np.int32),
    )
    edit = CellTopologyEdit(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0)),
            dtype=np.float64,
        ),
        target_ids,
        blocks,
        np.asarray(((11, 11), (13, 13), (17, 17), (11, 13)), dtype=np.int64),
        np.asarray(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 0.5))),
        np.asarray(
            ((True, False), (True, False), (True, False), (True, True)),
            dtype=np.bool_,
        ),
        (_empty_relations(0, 1), _empty_relations(1, 2), cell_relations),
    )
    target, _, _ = assemble_topology_edit(
        source, edit, numeric_version="exact-reference-refinement"
    )

    (witness,) = reference_cell_maps_from_edit(source, target, edit)
    exact = exact_reference_cell_vertices(witness)
    assert exact == (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        (
            (Fraction(1, 2), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
