#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._cad_revision import AssociationStatus, CADSelectionSet
from phydrax.geometry.brep import (
    BRepPartitionPolicy,
    BRepPartitionRole,
    partition_planar,
    PlanarEmbedding,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
)
from phydrax.geometry.brep._patches import ExactLineCurve
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)
_EMBEDDING = PlanarEmbedding(
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _rectangle(
    x_min: float,
    x_max: float,
    y_min: float = 0.0,
    y_max: float = 1.0,
    *,
    feature_id: str,
) -> PlanarMeshRegion:
    return PlanarMeshRegion(
        np.asarray(
            (
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            )
        ),
        ((0, 1, 2, 3),),
        feature_id=feature_id,
    )


def _operand(
    operand_id: str,
    source: Any,
    role: BRepPartitionRole = BRepPartitionRole.REGION,
    *,
    targets: tuple[str, ...] = (),
) -> PlanarPartitionOperand:
    return PlanarPartitionOperand(
        operand_id,
        source,
        role,
        target_region_ids=targets,
    )


def _plan(
    operands: Any,
    precedence: tuple[str, ...],
    *,
    embedding: PlanarEmbedding = _EMBEDDING,
) -> PlanarPartitionPlan:
    return PlanarPartitionPlan(
        _COORDINATES,
        embedding,
        tuple(operands),
        BRepPartitionPolicy(precedence),
    )


def _area(result: Any, name: str) -> float:
    # Exact shoelace area over coedge start points. A p-curve follows its
    # edge's parameter, so a reversed coedge starts at the p-curve's end.
    geometry = result.model.geometry
    total = Fraction(0)
    for entity in result.region(name).entity_ids:
        for loop in geometry.face_loops[entity.index]:
            points = tuple(
                geometry.pcurves[coedge].evaluate_exact(
                    Fraction(0 if geometry.coedge_senses[coedge] > 0 else 1)
                )
                for coedge in loop
            )
            total += (
                sum(
                    (first[0] * second[1] - first[1] * second[0])
                    for first, second in zip(
                        points, (*points[1:], points[0]), strict=True
                    )
                )
                / 2
            )
    return float(total)


def test_polygonal_hole_is_preserved_as_an_exact_curved_trim_loop() -> None:
    source = PlanarMeshRegion(
        np.asarray(
            (
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 4.0),
                (0.0, 4.0),
                (1.0, 1.0),
                (1.0, 3.0),
                (3.0, 3.0),
                (3.0, 1.0),
            )
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="holed-region",
    )
    vertices = np.asarray(source.vertices).copy()
    result = partition_planar(_plan((_operand("region", source),), ("region",)))

    assert _area(result, "region") == pytest.approx(12.0)
    assert len(result.region("region").entity_ids) == 1
    assert len(result.model.geometry.face_loops[0]) == 2
    assert result.model.topology.num_edges == 8
    assert result.report.created_target_occurrences == 0
    np.testing.assert_array_equal(source.vertices, vertices)


def test_triangle_partition_is_executable_and_preserves_source() -> None:
    source = PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 1.0))),
        ((0, 1, 2),),
        feature_id="triangle",
    )
    vertices = np.asarray(source.vertices).copy()
    result = partition_planar(_plan((_operand("region", source),), ("region",)))

    assert _area(result, "region") == pytest.approx(1.0)
    assert result.model.topology.num_faces == 1
    assert result.model.geometry.face_loops[0]
    assert result.model.tessellation is None
    np.testing.assert_array_equal(source.vertices, vertices)


def test_crossing_triangle_sources_partition_with_exact_lineage() -> None:
    first = PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (4.0, 0.0), (0.0, 4.0))),
        ((0, 1, 2),),
        feature_id="first-triangle",
    )
    second = PlanarMeshRegion(
        np.asarray(((1.0, -1.0), (5.0, 3.0), (1.0, 3.0))),
        ((0, 1, 2),),
        feature_id="second-triangle",
    )
    first_vertices = np.asarray(first.vertices).copy()
    second_vertices = np.asarray(second.vertices).copy()
    result = partition_planar(
        _plan(
            (_operand("first", first), _operand("second", second)),
            ("first", "second"),
        )
    )

    # Exact oracle: "first" wins the 7/2 overlap, so it keeps its whole area 8
    # and "second" keeps 8 - 7/2.
    assert _area(result, "first") == pytest.approx(8.0)
    assert _area(result, "second") == pytest.approx(4.5)
    assert result.model.topology.num_faces > 2
    # Only the exact edge crossings (1, 0), (2, 0) and (3, 1) lack a source
    # vertex; (1, 3) is a vertex of the second source.
    assert result.report.created_target_occurrences == 3
    assert result.association_graph.transaction.coverage.source_exhaustive
    assert result.association_graph.transaction.coverage.target_exhaustive
    np.testing.assert_array_equal(first.vertices, first_vertices)
    np.testing.assert_array_equal(second.vertices, second_vertices)


def test_embedding_roundtrip_and_signed_plane_residual() -> None:
    embedding = PlanarEmbedding(
        (3.0, -2.0, 5.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    )
    planar = np.asarray(((0.0, 0.0), (2.0, -4.0), (-1.5, 0.25)))
    world = embedding.to_world(planar)

    np.testing.assert_array_equal(embedding.to_planar(world), planar)
    np.testing.assert_array_equal(embedding.plane_residual(world), np.zeros(3))
    displaced = world + 0.75 * np.asarray(embedding.normal)
    np.testing.assert_allclose(embedding.plane_residual(displaced), 0.75)
    with pytest.raises(ValueError, match="right-handed"):
        PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
        )


def test_oblique_partition_retains_exact_vertices_beside_rounded_coordinates() -> None:
    root = float(np.sqrt(0.5))
    embedding = PlanarEmbedding(
        (0.1, -0.2, 0.3),
        (root, root, 0.0),
        (-root, root, 0.0),
        (0.0, 0.0, 1.0),
    )
    result = partition_planar(
        _plan(
            (_operand("region", _rectangle(0.1, 0.3, 0.2, 0.4, feature_id="oblique")),),
            ("region",),
            embedding=embedding,
        )
    )
    exact_points = result.model.geometry.vertex_exact_points
    rounded_points = np.asarray(result.model.geometry.vertex_points)
    assert all(point is not None for point in exact_points)
    assert any(
        value != Fraction.from_float(float(value))
        for point in exact_points
        if point is not None
        for value in point
    )
    np.testing.assert_array_equal(
        rounded_points,
        np.asarray(
            [
                [float(value) for value in point]
                for point in exact_points
                if point is not None
            ]
        ),
    )


def test_adjacent_regions_publish_shared_edges_with_opposite_incidence() -> None:
    result = partition_planar(
        _plan(
            (
                _operand("left", _rectangle(0.0, 1.0, feature_id="left")),
                _operand("right", _rectangle(1.0, 2.0, feature_id="right")),
            ),
            ("left", "right"),
        )
    )

    interfaces = [
        patch
        for patch in result.patches
        if patch.adjacent_region_ids == ("left", "right")
    ]
    assert result.topological_dimension == 2
    assert result.model.topology.num_solids == 0
    assert result.model.tessellation is None
    assert result.model.source_id.startswith("native-planar-partition:")
    geometry = result.model.geometry
    assert all(isinstance(curve, ExactLineCurve) for curve in geometry.curves)
    assert all(point is not None for point in geometry.vertex_exact_points)
    for edge, curve_index in enumerate(geometry.edge_curves):
        curve = geometry.curves[curve_index]
        assert isinstance(curve, ExactLineCurve)
        first_vertex, last_vertex = geometry.edge_vertices[edge]
        assert (
            curve.evaluate_exact(Fraction(0))
            == geometry.vertex_exact_points[first_vertex]
        )
        assert (
            curve.evaluate_exact(Fraction(1)) == geometry.vertex_exact_points[last_vertex]
        )
    assert len(interfaces) == 1
    assert len(interfaces[0].entity_ids) == 1
    edge_index = interfaces[0].entity_ids[0].index
    face_indices = result.model.topology.edge_faces[edge_index]
    assert len(face_indices) == 2
    signs = [
        next(
            result.model.geometry.coedge_senses[coedge]
            for loop in result.model.geometry.face_loops[face_index]
            for coedge in loop
            if result.model.geometry.coedge_edges[coedge] == edge_index
        )
        for face_index in face_indices
    ]
    assert signs[0] == -signs[1]
    assert result.named_region_entity_ids == result.named_face_entity_ids
    assert result.named_patch_entity_ids == result.named_edge_entity_ids
    assert result.named_solid_entity_ids == ()
    assert {entity.kind for region in result.regions for entity in region.entity_ids} == {
        "face"
    }
    assert {entity.kind for patch in result.patches for entity in patch.entity_ids} == {
        "edge"
    }
    assert result.report.target_regions == result.model.topology.num_faces
    assert result.report.target_patches == result.model.topology.num_edges
    assert _area(result, "left") == pytest.approx(1.0)
    assert _area(result, "right") == pytest.approx(1.0)


def test_precedence_and_voids_partition_rectangles_in_memory() -> None:
    high = _rectangle(0.0, 2.0, feature_id="high")
    low = _rectangle(1.0, 3.0, feature_id="low")
    forward = partition_planar(
        _plan(
            (_operand("low", low), _operand("high", high)),
            ("high", "low"),
        )
    )
    reverse = partition_planar(
        _plan(
            (_operand("high", high), _operand("low", low)),
            ("high", "low"),
        )
    )

    assert forward.model.source_revision == reverse.model.source_revision
    assert forward.association_graph.graph_id == reverse.association_graph.graph_id
    assert _area(forward, "high") == pytest.approx(2.0)
    assert _area(forward, "low") == pytest.approx(1.0)

    carved = partition_planar(
        _plan(
            (
                _operand("right", _rectangle(3.0, 5.0, feature_id="right")),
                _operand(
                    "cut",
                    _rectangle(1.0, 4.0, feature_id="cut"),
                    BRepPartitionRole.VOID,
                    targets=("left",),
                ),
                _operand("left", _rectangle(0.0, 2.0, feature_id="left")),
            ),
            ("left", "right"),
        )
    )
    assert _area(carved, "left") == pytest.approx(1.0)
    assert _area(carved, "right") == pytest.approx(2.0)


def test_exact_face_and_edge_histories_cover_split_and_deleted_sources() -> None:
    result = partition_planar(
        _plan(
            (
                _operand("region", _rectangle(0.0, 3.0, feature_id="region")),
                _operand(
                    "void",
                    _rectangle(1.0, 2.0, feature_id="void"),
                    BRepPartitionRole.VOID,
                    targets=("region",),
                ),
            ),
            ("region",),
        )
    )
    graph = result.association_graph
    region_selector = graph.source_revision.select("operand:region:face:0")
    split = graph.resolve_target(region_selector)
    mapped = graph.resolve_target_selection(
        CADSelectionSet(graph.source_revision.revision_id, (region_selector,))
    )
    void_selector = graph.source_revision.select("operand:void:face:0")
    deleted = graph.resolve_target(void_selector)

    assert split.status is AssociationStatus.MULTIPLE
    assert split.relation == "split"
    assert len(mapped.selectors) == 2
    assert {value.kind for value in mapped.selectors} == {"face"}
    assert deleted.status is AssociationStatus.NO_PREIMAGE
    assert deleted.relation == "deleted"
    assert result.report.created_target_occurrences == 0
    assert result.report.deleted_source_occurrences > 0
    assert {
        occurrence.occurrence_id for occurrence in graph.target_revision.occurrences
    } == {
        correspondence.target_occurrence_id
        for correspondence in graph.transaction.correspondences
    }
    region_edge_resolutions = tuple(
        graph.resolve_target(graph.source_revision.select(occurrence.occurrence_id))
        for occurrence in graph.source_revision.occurrences
        if occurrence.kind == "edge"
        and occurrence.parent_occurrence_id == "operand:region:face:0"
    )
    assert any(
        value.status is AssociationStatus.MULTIPLE
        and {candidate.kind for candidate in value.candidate_selectors} == {"edge"}
        for value in region_edge_resolutions
    )


def test_exact_planar_brep_face_is_a_source_operand() -> None:
    source_mesh = _rectangle(0.0, 1.0, feature_id="source-mesh")
    source_model = partition_planar(
        _plan((_operand("source", source_mesh),), ("source",))
    ).model
    from phydrax.geometry.brep._partition import cad_revision_from_brep_model

    original_revision = cad_revision_from_brep_model(source_model)
    original_face = next(
        occurrence
        for occurrence in original_revision.occurrences
        if occurrence.kind == "face"
    )
    original_vertices = np.asarray(source_model.geometry.vertex_points).copy()
    result = partition_planar(_plan((_operand("native", source_model),), ("native",)))

    selected_face = next(
        occurrence
        for occurrence in result.association_graph.source_revision.occurrences
        if occurrence.kind == "face"
    )
    lineage = result.association_graph.resolve_target(
        result.association_graph.source_revision.select(selected_face.occurrence_id)
    )
    assert selected_face.entity_id == original_face.entity_id
    assert lineage.status is AssociationStatus.UNIQUE
    assert _area(result, "native") == pytest.approx(1.0)
    assert result.model.tessellation is None
    np.testing.assert_array_equal(source_model.geometry.vertex_points, original_vertices)


def test_planar_plan_requires_total_region_precedence() -> None:
    with pytest.raises(ValueError, match="region_precedence"):
        _plan(
            (_operand("region", _rectangle(0.0, 1.0, feature_id="region")),),
            ("other",),
        )
