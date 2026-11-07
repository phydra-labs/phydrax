from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np
import pytest

from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    constant,
    evaluate_expression,
    Expression,
    scale,
)
from phydrax.discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._reference_chain_embedding import (
    certify_reference_chain_embedding,
)


def _two_adjacent_hexes() -> tuple[
    CellMesh, CellGeometrySpec, GlobalEmbeddingCertificate
]:
    reference_vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.int64
    )
    coordinates = np.asarray(
        tuple(
            (float(x), float(y), float(z))
            for z in (0, 1)
            for y in (0, 1)
            for x in range(3)
        ),
        dtype=np.float64,
    )
    row_by_point = {
        tuple(int(value) for value in point): row for row, point in enumerate(coordinates)
    }
    cells = np.asarray(
        tuple(
            tuple(
                row_by_point[(origin + int(vertex[0]), int(vertex[1]), int(vertex[2]))]
                for vertex in reference_vertices
            )
            for origin in (0, 1)
        ),
        dtype=np.int64,
    )
    block = CellBlock(
        "hexes",
        "hexahedron",
        cells,
        global_ids=np.asarray((901, 1703), dtype=np.int64),
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64) + 101,
    )
    element = lagrange_element("hexahedron", 1)
    geometry = CellGeometrySpec(
        {"hexes": element},
        {"hexes": cells},
        coordinates,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert validity.all_certified
    assert embedding.status == "certified", embedding.findings
    return mesh, geometry, embedding


def _target_partition(
    parent_mesh: CellMesh,
    parent_geometry: CellGeometrySpec,
    duplicate_local_vertices: tuple[int, ...] = (),
) -> tuple[CellMesh, CellGeometrySpec, dict[str, ReferenceCellMapWitnesses]]:
    parent_block = parent_mesh.blocks[0]
    target_cells = np.asarray(parent_block.vertices, dtype=np.int64).copy()
    target_coordinates = np.asarray(parent_mesh.coordinates, dtype=np.float64).copy()
    target_vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64).copy()
    parent_vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(parent_block.vertices, dtype=np.int64)
    ]
    for local_vertex in duplicate_local_vertices:
        duplicate_row = target_coordinates.shape[0]
        target_coordinates = np.concatenate(
            (
                target_coordinates,
                target_coordinates[target_cells[1, local_vertex]][None, :],
            ),
            axis=0,
        )
        target_vertex_ids = np.concatenate(
            (target_vertex_ids, np.asarray((10_000 + duplicate_row,), dtype=np.int64))
        )
        target_cells[1, local_vertex] = duplicate_row
    target_block = CellBlock(
        "hexes",
        "hexahedron",
        target_cells,
        global_ids=np.asarray(parent_block.global_ids, dtype=np.int64),
    )
    target_mesh = CellMesh(
        target_coordinates,
        (target_block,),
        vertex_global_ids=target_vertex_ids,
    )
    exact_reference_vertices = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    exact_vertex_bank = (exact_reference_vertices, exact_reference_vertices)
    witness = ReferenceCellMapWitnesses(
        np.arange(target_block.cell_count, dtype=np.int64),
        np.asarray(parent_block.global_ids, dtype=np.int64),
        "hexahedron",
        "hexahedron",
        np.asarray(exact_vertex_bank, dtype=np.float64),
        exact_vertex_bank,
        freeze_reference_chart_arguments((axes(3), axes(3))),
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(parent_geometry),
        parent_mesh.topology_id,
        {"hexes": np.asarray(parent_block.global_ids, dtype=np.int64)},
        {"hexes": parent_vertex_ids},
        block_parent_reference_maps={"hexes": witness},
    )
    target_geometry = CellGeometrySpec(
        {"hexes": lagrange_element("hexahedron", 1)},
        {"hexes": target_cells},
        target_coordinates,
        restriction_source=restriction,
    )
    return target_mesh, target_geometry, {"hexes": witness}


def _mixed_hex_prism_partition(
    parent_mesh: CellMesh,
    parent_geometry: CellGeometrySpec,
) -> tuple[CellMesh, CellGeometrySpec, dict[str, ReferenceCellMapWitnesses]]:
    parent_block = parent_mesh.blocks[0]
    parent_cells = np.asarray(parent_block.vertices, dtype=np.int64)
    parent_cell_ids = np.asarray(parent_block.global_ids, dtype=np.int64)
    parent_vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64)[
        parent_cells
    ]
    coordinates = np.asarray(parent_mesh.coordinates, dtype=np.float64)
    row_by_coordinate = {
        (
            int(coordinates[row, 0]),
            int(coordinates[row, 1]),
            int(coordinates[row, 2]),
        ): row
        for row in range(coordinates.shape[0])
    }
    hex_block = CellBlock(
        "hexes",
        "hexahedron",
        parent_cells[:1],
        global_ids=np.asarray((301,), dtype=np.int64),
    )
    u, v, w = axes(3)
    one = constant(Fraction(1), 3)
    prism_charts = (
        (u, v, w),
        (
            add(one, scale(v, Fraction(-1))),
            add(u, v),
            w,
        ),
    )
    prism_reference_vertices = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("prism").vertices
    )

    def exact_chart_point(
        chart: tuple[Expression, ...],
        point: tuple[Fraction, ...],
    ) -> tuple[Fraction, Fraction, Fraction]:
        values = tuple(evaluate_expression(expression, point) for expression in chart)
        if not all(isinstance(value, Fraction) for value in values):
            raise ValueError("Mixed prism source corners must remain rational.")
        return cast(tuple[Fraction, Fraction, Fraction], values)

    prism_cells = []
    for chart in prism_charts:
        local_rows = []
        for target_point in prism_reference_vertices:
            source_point = exact_chart_point(chart, target_point)
            physical_point = (
                Fraction(1) + source_point[0],
                source_point[1],
                source_point[2],
            )
            local_rows.append(
                row_by_coordinate[
                    (
                        int(physical_point[0]),
                        int(physical_point[1]),
                        int(physical_point[2]),
                    )
                ]
            )
        prism_cells.append(local_rows)
    prism_block = CellBlock(
        "prisms",
        "prism",
        np.asarray(prism_cells, dtype=np.int64),
        global_ids=np.asarray((401, 402), dtype=np.int64),
    )
    target_mesh = CellMesh(
        coordinates,
        (hex_block, prism_block),
        vertex_global_ids=np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64),
    )

    hex_exact_vertices = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    hex_witness = ReferenceCellMapWitnesses(
        np.asarray((0,), dtype=np.int64),
        parent_cell_ids[:1],
        "hexahedron",
        "hexahedron",
        np.asarray((hex_exact_vertices,), dtype=np.float64),
        (hex_exact_vertices,),
        freeze_reference_chart_arguments((axes(3),)),
    )
    prism_exact_vertices = tuple(
        tuple(exact_chart_point(chart, point) for point in prism_reference_vertices)
        for chart in prism_charts
    )
    prism_witness = ReferenceCellMapWitnesses(
        np.arange(2, dtype=np.int64),
        np.repeat(parent_cell_ids[1:2], 2),
        "prism",
        "hexahedron",
        np.asarray(prism_exact_vertices, dtype=np.float64),
        prism_exact_vertices,
        freeze_reference_chart_arguments(prism_charts),
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(parent_geometry),
        parent_mesh.topology_id,
        {
            "hexes": parent_cell_ids[:1],
            "prisms": np.repeat(parent_cell_ids[1:2], 2),
        },
        {
            "hexes": parent_vertex_ids[:1],
            "prisms": np.repeat(parent_vertex_ids[1:2], 2, axis=0),
        },
        block_parent_reference_maps={
            "hexes": hex_witness,
            "prisms": prism_witness,
        },
    )
    target_geometry = CellGeometrySpec(
        {
            "hexes": lagrange_element("hexahedron", 1),
            "prisms": lagrange_element("prism", 1),
        },
        {
            "hexes": np.asarray(hex_block.vertices, dtype=np.int64),
            "prisms": np.asarray(prism_block.vertices, dtype=np.int64),
        },
        coordinates,
        restriction_source=restriction,
    )
    return (
        target_mesh,
        target_geometry,
        {"hexes": hex_witness, "prisms": prism_witness},
    )


def _shared_source_face_local_vertices(parent_mesh: CellMesh) -> tuple[int, ...]:
    block = parent_mesh.blocks[0]
    cells = np.asarray(block.vertices, dtype=np.int64)
    parent_vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64)
    shared_ids = set(parent_vertex_ids[cells[0]]) & set(parent_vertex_ids[cells[1]])
    return tuple(
        local
        for local, vertex_row in enumerate(cells[1])
        if int(parent_vertex_ids[int(vertex_row)]) in shared_ids
    )


def test_exact_two_parent_reference_chain_records_shared_strata() -> None:
    parent_mesh, parent_geometry, parent_embedding = _two_adjacent_hexes()
    target_mesh, target_geometry, maps = _target_partition(parent_mesh, parent_geometry)

    certificate = certify_reference_chain_embedding(
        parent_mesh,
        parent_geometry,
        target_mesh,
        target_geometry,
        maps,
        parent_embedding,
    )

    assert certificate.status == "certified", certificate.findings
    assert certificate.cross_parent_coverage_complete
    premises = certificate.cross_parent_premises
    assert any(
        premise.source_dimension == premise.target_entity_dimension == 2
        for premise in premises
    )
    assert any(
        premise.source_dimension == premise.target_entity_dimension == 1
        for premise in premises
    )
    assert any(
        premise.source_dimension == premise.target_entity_dimension == 0
        for premise in premises
    )
    assert all(
        len(premise.root_trace_ids_by_parent) == 2
        and len(premise.physical_trace_id) == 64
        for premise in premises
    )
    assert all(
        premise.orientation_signs
        for premise in premises
        if premise.target_entity_dimension > 0
    )
    assert (
        certificate.revalidate(
            parent_mesh,
            parent_geometry,
            target_mesh,
            target_geometry,
            maps,
            parent_embedding,
        ).certificate_id
        == certificate.certificate_id
    )


def test_mixed_hexahedron_prism_target_chain_is_certified() -> None:
    parent_mesh, parent_geometry, parent_embedding = _two_adjacent_hexes()
    target_mesh, target_geometry, maps = _mixed_hex_prism_partition(
        parent_mesh, parent_geometry
    )

    certificate = certify_reference_chain_embedding(
        parent_mesh,
        parent_geometry,
        target_mesh,
        target_geometry,
        maps,
        parent_embedding,
    )

    assert certificate.status == "certified", certificate.findings
    assert certificate.cross_parent_coverage_complete
    assert len(certificate.premises) == 2
    assert {
        chart.target_cell_kind
        for premise in certificate.premises
        for chart in premise.charts
    } == {"hexahedron", "prism"}
    assert any(
        premise.source_dimension == premise.target_entity_dimension == 2
        for premise in certificate.cross_parent_premises
    )


@pytest.mark.parametrize(
    ("duplicate_mode", "expected_dimension"),
    (("face", 2), ("edge", 1), ("vertex", 0)),
)
def test_duplicated_target_strata_cannot_certify_shared_parent_strata(
    duplicate_mode: str,
    expected_dimension: int,
) -> None:
    parent_mesh, parent_geometry, parent_embedding = _two_adjacent_hexes()
    shared_face_vertices = _shared_source_face_local_vertices(parent_mesh)
    if duplicate_mode == "face":
        duplicate_vertices = shared_face_vertices
    elif duplicate_mode == "edge":
        topology = reference_cell_topology("hexahedron")
        block = parent_mesh.blocks[0]
        cells = np.asarray(block.vertices, dtype=np.int64)
        parent_vertex_ids = np.asarray(parent_mesh.vertex_global_ids, dtype=np.int64)
        shared_ids = set(parent_vertex_ids[cells[0]]) & set(parent_vertex_ids[cells[1]])
        edge = next(
            tuple(int(local) for local in edge_vertices)
            for edge_vertices in topology.entities[1]
            if all(
                int(parent_vertex_ids[int(cells[1, int(local)])]) in shared_ids
                for local in edge_vertices
            )
        )
        duplicate_vertices = edge
    else:
        duplicate_vertices = (shared_face_vertices[0],)
    target_mesh, target_geometry, maps = _target_partition(
        parent_mesh,
        parent_geometry,
        duplicate_vertices,
    )

    certificate = certify_reference_chain_embedding(
        parent_mesh,
        parent_geometry,
        target_mesh,
        target_geometry,
        maps,
        parent_embedding,
    )

    assert certificate.status != "certified"
    assert not certificate.cross_parent_coverage_complete
    assert any(
        finding.check == "cross_parent_target_strata"
        and f"target dimension-{expected_dimension}" in finding.detail
        for finding in certificate.findings
    ), certificate.findings


def test_duck_typed_parent_certificate_is_rejected() -> None:
    parent_mesh, parent_geometry, _ = _two_adjacent_hexes()
    target_mesh, target_geometry, maps = _target_partition(parent_mesh, parent_geometry)

    class FakeParentCertificate:
        status = "certified"
        certificate_id = "forged-certificate-id"

        def require_current(self, *_operands: object) -> None:
            return None

    certificate = certify_reference_chain_embedding(
        parent_mesh,
        parent_geometry,
        target_mesh,
        target_geometry,
        maps,
        cast(GlobalEmbeddingCertificate, FakeParentCertificate()),
    )

    assert certificate.status == "violated"
    assert any(
        finding.check == "parent_global_embedding_type"
        for finding in certificate.findings
    )


def test_wrong_concrete_parent_certificate_is_rejected() -> None:
    parent_mesh, parent_geometry, _ = _two_adjacent_hexes()
    parent_block = parent_mesh.blocks[0]
    wrong_geometry = CellGeometrySpec(
        {"hexes": parent_geometry.elements[0]},
        {"hexes": np.asarray(parent_block.vertices, dtype=np.int64)},
        np.asarray(parent_geometry.coordinates, dtype=np.float64)
        + np.asarray((3.0, 0.0, 0.0), dtype=np.float64),
    )
    wrong_validity = certify_cell_geometry_validity(wrong_geometry, mesh=parent_mesh)
    wrong_embedding = certify_global_embedding(
        parent_mesh, wrong_geometry, wrong_validity
    )
    assert wrong_validity.all_certified
    assert wrong_embedding.status == "certified", wrong_embedding.findings

    target_mesh, target_geometry, maps = _target_partition(parent_mesh, parent_geometry)
    certificate = certify_reference_chain_embedding(
        parent_mesh,
        parent_geometry,
        target_mesh,
        target_geometry,
        maps,
        wrong_embedding,
    )

    assert certificate.status == "violated"
    assert any(
        finding.check == "parent_global_embedding_currentness"
        for finding in certificate.findings
    )


def test_reference_chain_reuses_only_actual_validated_parent_embedding() -> None:
    from copy import copy
    import pickle

    from phydrax.geometry._global_embedding import ValidatedGlobalEmbedding

    parent_mesh, parent_geometry, _ = _two_adjacent_hexes()
    target_mesh, target_geometry, maps = _target_partition(parent_mesh, parent_geometry)
    validity = certify_cell_geometry_validity(parent_geometry, mesh=parent_mesh)
    computed = ValidatedGlobalEmbedding(parent_mesh, parent_geometry, validity)
    independent = certify_reference_chain_embedding(
        parent_mesh, parent_geometry, target_mesh, target_geometry, maps, computed.certificate
    )
    reused = certify_reference_chain_embedding(
        parent_mesh,
        parent_geometry,
        target_mesh,
        target_geometry,
        maps,
        computed.certificate,
        _validated_parent_embedding=computed,
    )
    assert reused == independent
    assert reused.status == "certified", reused.findings
    with pytest.raises(ValueError, match="actual validated embedding"):
        certify_reference_chain_embedding(
            parent_mesh,
            parent_geometry,
            target_mesh,
            target_geometry,
            maps,
            copy(computed.certificate),
            _validated_parent_embedding=computed,
        )
    with pytest.raises(ValueError, match="different exact mesh or geometry"):
        certify_reference_chain_embedding(
            copy(parent_mesh),
            parent_geometry,
            target_mesh,
            target_geometry,
            maps,
            computed.certificate,
            _validated_parent_embedding=computed,
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(computed)
