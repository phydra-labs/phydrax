#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import PlanarEmbedding
from phydrax.geometry._cad_revision import AssociationStatus, CADSelectionSet
from phydrax.geometry.brep import (
    BRepPartitionPolicy,
    BRepPartitionResult,
    BRepPartitionRole,
    partition_planar,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
    prepare_brep_query,
)
from phydrax.geometry.brep._model import BRepEntityId, BRepModel
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.units import METER, MILLIMETER


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
    source: PlanarMeshRegion | BRepModel,
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
    operands: Sequence[PlanarPartitionOperand],
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


def _execute(plan: PlanarPartitionPlan, destination: Path) -> BRepPartitionResult:
    return partition_planar(
        plan,
        destination=destination,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )


def _face_areas(result: BRepPartitionResult) -> dict[BRepEntityId, float]:
    measures = prepare_brep_query(result.model).measures
    return {
        entity_id: float(measures.face_areas[index])
        for index, entity_id in enumerate(result.model.face_ids)
    }


def _region_area(result: BRepPartitionResult, name: str) -> float:
    areas = _face_areas(result)
    return sum(areas[value] for value in result.region(name).entity_ids)


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


def test_adjacent_regions_publish_shared_edges_with_opposite_incidence(
    tmp_path: Any,
) -> None:
    result = _execute(
        _plan(
            (
                _operand("left", _rectangle(0.0, 1.0, feature_id="left")),
                _operand("right", _rectangle(1.0, 2.0, feature_id="right")),
            ),
            ("left", "right"),
        ),
        tmp_path / "adjacent.brep",
    )

    interfaces = [
        patch
        for patch in result.patches
        if patch.adjacent_region_ids == ("left", "right")
    ]
    assert result.topological_dimension == 2
    assert result.model.topology.num_solids == 0
    assert len(interfaces) == 1
    assert len(interfaces[0].entity_ids) == 1
    edge_index = interfaces[0].entity_ids[0].index
    face_indices = result.model.topology.edge_faces[edge_index]
    assert len(face_indices) == 2
    signs = [
        next(
            1 if signed_edge > 0 else -1
            for wire in result.model.topology.face_wires[face_index]
            for signed_edge in wire
            if abs(signed_edge) - 1 == edge_index
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
    assert result.region_entity_kind == "face"
    assert result.patch_entity_kind == "edge"
    assert {region.entity_kind for region in result.regions} == {"face"}
    assert {patch.entity_kind for patch in result.patches} == {"edge"}
    assert result.report.topological_dimension == 2
    assert {region.topological_dimension for region in result.regions} == {2}
    assert {patch.topological_dimension for patch in result.patches} == {2}


def test_oriented_hole_is_preserved_as_exact_face_wires(tmp_path: Any) -> None:
    region = PlanarMeshRegion(
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
    result = _execute(
        _plan((_operand("region", region),), ("region",)),
        tmp_path / "hole.brep",
    )

    assert _region_area(result, "region") == pytest.approx(12.0)
    assert len(result.region("region").entity_ids) == 1
    face_index = result.region("region").entity_ids[0].index
    assert len(result.model.topology.face_wires[face_index]) == 2
    assert set(result.model.topology.edge_faces) == {(face_index,)}


def test_precedence_is_independent_of_planar_operand_order(tmp_path: Any) -> None:
    high = _rectangle(0.0, 2.0, feature_id="high")
    low = _rectangle(1.0, 3.0, feature_id="low")
    forward = _execute(
        _plan(
            (_operand("low", low), _operand("high", high)),
            ("high", "low"),
        ),
        tmp_path / "precedence-forward.brep",
    )
    reverse = _execute(
        _plan(
            (_operand("high", high), _operand("low", low)),
            ("high", "low"),
        ),
        tmp_path / "precedence-reverse.brep",
    )

    assert forward.model.source_revision == reverse.model.source_revision
    assert forward.association_graph.graph_id == reverse.association_graph.graph_id
    assert _region_area(forward, "high") == pytest.approx(2.0)
    assert _region_area(forward, "low") == pytest.approx(1.0)


def test_void_subtracts_only_its_declared_planar_target(tmp_path: Any) -> None:
    result = _execute(
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
        ),
        tmp_path / "targeted-void.brep",
    )

    assert _region_area(result, "left") == pytest.approx(1.0)
    assert _region_area(result, "right") == pytest.approx(2.0)


def test_void_empties_winning_planar_region_without_lower_refill(tmp_path: Path) -> None:
    result = _execute(
        _plan(
            (
                _operand("left", _rectangle(0.0, 2.0, feature_id="left")),
                _operand("right", _rectangle(1.0, 3.0, feature_id="right")),
                _operand(
                    "cut",
                    _rectangle(1.0, 2.0, feature_id="cut"),
                    BRepPartitionRole.VOID,
                    targets=("left",),
                ),
            ),
            ("left", "right"),
        ),
        tmp_path / "owned-hole.brep",
    )
    assert _region_area(result, "left") == pytest.approx(1.0)
    assert _region_area(result, "right") == pytest.approx(1.0)
    assert all(len(patch.adjacent_region_ids) == 1 for patch in result.patches)
    coverage = result.association_graph.transaction.coverage
    assert coverage.source_exhaustive and coverage.target_exhaustive


def test_exact_face_and_edge_histories_cover_split_and_deleted_sources(
    tmp_path: Any,
) -> None:
    result = _execute(
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
        ),
        tmp_path / "history.brep",
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


def test_persisted_root_faces_reenter_the_same_partition_contract(tmp_path: Any) -> None:
    first = _execute(
        _plan(
            (_operand("source", _rectangle(0.0, 2.0, feature_id="source")),),
            ("source",),
        ),
        tmp_path / "first.brep",
    )
    second = _execute(
        _plan((_operand("copy", first.model),), ("copy",)),
        tmp_path / "second.brep",
    )

    assert second.model.topology.num_solids == 0
    assert _region_area(second, "copy") == pytest.approx(2.0)
    roots = tuple(
        value
        for value in second.association_graph.source_revision.occurrences
        if value.parent_occurrence_id is None
    )
    assert roots
    assert {value.kind for value in roots} == {"face"}
    assert {
        value.kind
        for value in second.association_graph.source_revision.occurrences
        if value.parent_occurrence_id is not None
    } == {"edge"}
    with pytest.raises(ValueError, match="coordinate contract"):
        PlanarPartitionPlan(
            SpatialCoordinateContract(METER),
            _EMBEDDING,
            (_operand("copy", first.model),),
            BRepPartitionPolicy(("copy",)),
        )


def test_planar_sources_must_share_the_declared_embedding_plane(tmp_path: Any) -> None:
    first = _execute(
        _plan(
            (_operand("source", _rectangle(0.0, 1.0, feature_id="source")),),
            ("source",),
        ),
        tmp_path / "plane-source.brep",
    )
    displaced = PlanarEmbedding(
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="not coplanar"):
        _execute(
            _plan((_operand("source", first.model),), ("source",), embedding=displaced),
            tmp_path / "wrong-plane.brep",
        )
    assert not (tmp_path / "wrong-plane.brep").exists()


def test_holes_must_be_strictly_inside_the_outer_loop(tmp_path: Any) -> None:
    invalid = PlanarMeshRegion(
        np.asarray(
            (
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (3.0, 0.0),
                (3.0, 1.0),
                (4.0, 1.0),
                (4.0, 0.0),
            )
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="outside-hole",
    )

    with pytest.raises(ValueError, match="strictly inside"):
        _execute(
            _plan((_operand("region", invalid),), ("region",)),
            tmp_path / "invalid-hole.brep",
        )
    assert not (tmp_path / "invalid-hole.brep").exists()
