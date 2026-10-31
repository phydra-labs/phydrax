#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._cad_revision import (
    AssociationStatus,
    CADOccurrence,
    CADRevision,
    CADSelectionSet,
    CADSelector,
)
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cylinder,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._model import BRepEntityId
from phydrax.geometry.brep._partition import (
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_brep,
)
from phydrax.geometry.brep._patches import ExactLineCurve, RevolutionSurface
from phydrax.meshing import MeshingScope
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)


def _box(x: float, length: float = 1.0) -> Any:
    return brep_box(
        (x, 0.0, 0.0),
        (x + length, 1.0, 1.0),
        coordinate_contract=_COORDINATES,
    )


def _operand(
    operand_id: str,
    model: Any,
    role: BRepPartitionRole = BRepPartitionRole.REGION,
    *,
    targets: tuple[str, ...] = (),
) -> Any:
    revision = cad_revision_from_brep_model(model)
    selection = CADSelectionSet.from_revision(
        revision,
        tuple(
            occurrence.occurrence_id
            for occurrence in revision.occurrences
            if occurrence.kind == "solid"
        ),
    )
    return BRepPartitionOperand(operand_id, model, role, selection, targets)


def _plan(operands: Any, precedence: tuple[str, ...]) -> Any:
    return BRepPartitionPlan(
        _COORDINATES,
        tuple(operands),
        BRepPartitionPolicy(precedence),
    )


def _execute(plan: Any) -> Any:
    return partition_brep(plan)


def _solid_volumes(result: Any) -> dict[BRepEntityId, float]:
    geometry = result.model.geometry
    vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
    volumes: dict[BRepEntityId, float] = {}
    for entity_id, faces in zip(
        result.model.solid_ids,
        result.model.topology.solid_faces,
        strict=True,
    ):
        vertex_ids = {
            vertex
            for face in faces
            for loop in geometry.face_loops[face]
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        }
        extents = np.ptp(vertices[np.asarray(sorted(vertex_ids), dtype=np.intp)], axis=0)
        volumes[entity_id] = float(np.prod(extents))
    return volumes


def test_touching_regions_publish_exact_interface_and_meshing_entity_sets() -> None:
    left = _box(0.0)
    right = _box(1.0)
    result = _execute(
        _plan(
            (
                _operand("left", left),
                _operand("right", right),
            ),
            ("left", "right"),
        )
    )

    interfaces = [
        patch for patch in result.patches if len(patch.adjacent_region_ids) == 2
    ]
    assert result.model.topology.num_solids == 2
    assert result.topological_dimension == 3
    assert len(interfaces) == 1
    assert interfaces[0].adjacent_region_ids == ("left", "right")
    assert len(interfaces[0].entity_ids) == 1
    transaction = result.association_graph.transaction
    assert transaction.coverage.source_exhaustive
    assert transaction.coverage.target_exhaustive
    assert transaction.unresolved_source_ids == frozenset()
    assert transaction.unresolved_target_ids == frozenset()
    assert {
        value.occurrence_id
        for value in result.association_graph.target_revision.occurrences
    } == {value.target_occurrence_id for value in transaction.correspondences}
    source_assemblies = tuple(
        occurrence
        for occurrence in result.association_graph.source_revision.occurrences
        if occurrence.kind == "assembly"
    )
    target_assemblies = {
        occurrence.occurrence_id
        for occurrence in result.association_graph.target_revision.occurrences
        if occurrence.kind == "assembly"
    }
    assert source_assemblies
    assert target_assemblies
    for occurrence in source_assemblies:
        resolution = result.association_graph.resolve_target(
            result.association_graph.source_revision.select(occurrence.occurrence_id)
        )
        assert resolution.status in (
            AssociationStatus.UNIQUE,
            AssociationStatus.MULTIPLE,
        )
        assert all(
            selector.occurrence_id in target_assemblies
            for selector in resolution.candidate_selectors
        )
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
    assert set(
        entity for _, values in result.named_solid_entity_ids for entity in values
    ) == set(result.model.solid_ids)
    assert set(
        entity for _, values in result.named_face_entity_ids for entity in values
    ) == set(result.model.face_ids)
    assert result.named_region_entity_ids == result.named_solid_entity_ids
    assert result.named_patch_entity_ids == result.named_face_entity_ids
    assert result.named_edge_entity_ids == ()
    assert (
        result.report.source_region_occurrences == result.report.source_solid_occurrences
    )
    assert result.report.source_patch_occurrences == result.report.source_face_occurrences
    assert result.report.target_regions == result.model.topology.num_solids
    assert result.report.target_patches == result.model.topology.num_faces
    assert {entity.kind for region in result.regions for entity in region.entity_ids} == {
        "solid"
    }
    assert {entity.kind for patch in result.patches for entity in patch.entity_ids} == {
        "face"
    }
    assert result.region_entity_kind == "solid"
    assert result.patch_entity_kind == "face"
    assert {region.entity_kind for region in result.regions} == {"solid"}
    assert {patch.entity_kind for patch in result.patches} == {"face"}
    assert result.report.topological_dimension == 3
    assert {region.topological_dimension for region in result.regions} == {3}
    assert {patch.topological_dimension for patch in result.patches} == {3}
    left_scope = MeshingScope.from_geometry(
        result.model, result.region("left").entity_ids
    )
    interface_scope = MeshingScope.from_geometry(result.model, interfaces[0].entity_ids)
    assert left_scope.entity_dimension == 3
    assert interface_scope.entity_dimension == 2


def test_disjoint_regions_have_only_one_sided_boundary_patches() -> None:
    first = _box(0.0)
    second = _box(2.0)
    result = _execute(
        _plan(
            (_operand("first", first), _operand("second", second)),
            ("first", "second"),
        )
    )

    assert result.model.topology.num_solids == 2
    assert {patch.adjacent_region_ids for patch in result.patches} == {
        ("first",),
        ("second",),
    }
    assert all(len(indices) == 1 for indices in result.model.topology.face_solids)


def test_explicit_precedence_is_independent_of_operand_order() -> None:
    high = _box(0.0, 2.0)
    low = _box(1.0, 2.0)
    forward = _execute(
        _plan(
            (_operand("low", low), _operand("high", high)),
            ("high", "low"),
        )
    )
    reverse = _execute(
        _plan(
            (_operand("high", high), _operand("low", low)),
            ("high", "low"),
        )
    )

    assert forward.model.source_revision == reverse.model.source_revision
    assert forward.association_graph.graph_id == reverse.association_graph.graph_id
    volumes = _solid_volumes(forward)
    assert sum(
        volumes[value] for value in forward.region("high").entity_ids
    ) == pytest.approx(2.0)
    assert sum(
        volumes[value] for value in forward.region("low").entity_ids
    ) == pytest.approx(1.0)


def test_void_subtracts_only_its_declared_target_region() -> None:
    left = _box(0.0, 2.0)
    right = _box(3.0, 2.0)
    cutting = _box(1.0, 3.0)
    result = _execute(
        _plan(
            (
                _operand("right", right),
                _operand(
                    "cut",
                    cutting,
                    BRepPartitionRole.VOID,
                    targets=("left",),
                ),
                _operand("left", left),
            ),
            ("left", "right"),
        )
    )

    volumes = _solid_volumes(result)
    assert sum(
        volumes[value] for value in result.region("left").entity_ids
    ) == pytest.approx(1.0)
    assert sum(
        volumes[value] for value in result.region("right").entity_ids
    ) == pytest.approx(2.0)


def test_split_and_deleted_histories_map_exact_selection_sets() -> None:
    region = _box(0.0, 3.0)
    cutting = _box(1.0, 1.0)
    result = _execute(
        _plan(
            (
                _operand("region", region),
                _operand(
                    "void",
                    cutting,
                    BRepPartitionRole.VOID,
                    targets=("region",),
                ),
            ),
            ("region",),
        )
    )

    graph = result.association_graph
    region_selector = graph.source_revision.select("operand:region:solid:0")
    split = graph.resolve_target(region_selector)
    mapped = graph.resolve_target_selection(
        CADSelectionSet(graph.source_revision.revision_id, (region_selector,))
    )
    void_selector = graph.source_revision.select("operand:void:solid:0")
    deleted = graph.resolve_target(void_selector)

    assert split.status is AssociationStatus.MULTIPLE
    assert split.relation == "split"
    assert len(mapped.selectors) == 2
    assert {value.kind for value in mapped.selectors} == {"solid"}
    assert deleted.status is AssociationStatus.NO_PREIMAGE
    assert deleted.relation == "deleted"
    assert (
        graph.resolve_target_selection(
            CADSelectionSet(graph.source_revision.revision_id, (void_selector,))
        ).selectors
        == ()
    )
    assert result.report.deleted_source_occurrences > 0


def _l_shaped_partition(tessellation: Any = None) -> Any:
    first = brep_box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), coordinate_contract=_COORDINATES)
    second = brep_box((1.0, 0.0, 0.0), (3.0, 2.0, 1.0), coordinate_contract=_COORDINATES)
    return partition_brep(
        _plan((_operand("a", first), _operand("b", second)), ("a", "b")),
        tessellation=tessellation,
    )


def test_region_solids_are_exact_unions_of_grid_atoms_with_maximal_faces() -> None:
    result = _l_shaped_partition()
    model = result.model
    geometry = model.geometry

    assert model.topology.num_solids == 2
    assert [len(region.entity_ids) for region in result.regions] == [1, 1]
    # Box ``a`` keeps 5 own faces, the L-shaped remainder of ``b`` 6, and the two
    # interface faces are shared once by both region solids.
    assert model.topology.num_faces == 13
    interfaces = [
        patch for patch in result.patches if patch.adjacent_region_ids == ("a", "b")
    ]
    assert len(interfaces) == 1
    assert len(interfaces[0].entity_ids) == 2
    assert sum(len(solids) == 2 for solids in model.topology.face_solids) == 2
    assert all(len(loops) == 1 for loops in geometry.face_loops)
    assert sorted(len(loops[0]) for loops in geometry.face_loops).count(6) == 2
    edges = {
        tuple(sorted(geometry.vertex_exact_points[vertex] for vertex in vertices))
        for vertices in geometry.edge_vertices
    }
    zero, two = Fraction(0), Fraction(2)
    assert ((zero, zero, zero), (two, zero, zero)) in edges

    graph = result.association_graph
    first = graph.resolve_target(graph.source_revision.select("operand:a:solid:0"))
    second = graph.resolve_target(graph.source_revision.select("operand:b:solid:0"))
    # Region ``a`` absorbs the atom of ``b`` it outranks: ``a`` merges into one
    # solid while ``b`` splits across both region solids.
    assert first.relation == "merge"
    assert len(first.candidate_selectors) == 1
    assert second.relation == "split-merge"
    assert len(second.candidate_selectors) == 2
    transaction = graph.transaction
    assert {value.occurrence_id for value in graph.target_revision.occurrences} == {
        value.target_occurrence_id for value in transaction.correspondences
    }


def test_region_interface_is_tessellated_once_with_both_region_sides() -> None:
    result = _l_shaped_partition(
        BRepTessellationPolicy(
            linear_deflection=0.5, angular_deflection=0.2, trim_samples_per_edge=3
        )
    )
    tessellation = result.model.tessellation
    topology = result.model.tessellation_multiregion_surface()
    assert tessellation is not None and topology is not None

    active = np.asarray(topology.face_active, dtype=bool)
    labels = np.asarray(topology.face_labels, dtype=np.int64)[active]
    rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(tessellation.triangle_global_ids, dtype=np.int64).tolist()
        )
    }
    triangles = np.asarray(tessellation.triangles, dtype=np.int64)[
        [
            rows[int(identifier)]
            for identifier in np.asarray(topology.face_global_ids)[active].tolist()
        ]
    ]
    corners = np.asarray(tessellation.vertices, dtype=np.float64)[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    exterior = len(topology.region_ids) - 1
    interface = np.all(labels != exterior, axis=1)

    assert len({tuple(sorted(row)) for row in triangles.tolist()}) == len(triangles)
    assert {tuple(sorted(row)) for row in labels[interface].tolist()} == {(0, 1)}
    assert float(np.sum(areas[interface])) == pytest.approx(2.0)
    assert float(np.sum(areas[~interface])) == pytest.approx(20.0)


def test_curved_cylinder_partition_publishes_exact_notch_and_preserves_sources() -> None:
    cylinder = brep_cylinder(
        1.0,
        2.0,
        coordinate_contract=_COORDINATES,
        source_id="partition-cylinder",
    )
    notch = brep_box(
        (0.75, -0.25, 0.5),
        (1.25, 0.25, 1.5),
        coordinate_contract=_COORDINATES,
        source_id="partition-notch",
    )
    cylinder_revision = cylinder.source_revision
    cylinder_geometry = cylinder.geometry.geometry_id
    cylinder_operand = _operand("cylinder", cylinder)
    result = partition_brep(
        _plan(
            (
                cylinder_operand,
                _operand(
                    "notch",
                    notch,
                    BRepPartitionRole.VOID,
                    targets=("cylinder",),
                ),
            ),
            ("cylinder",),
        )
    )

    source_selector = cylinder_operand.selection.selectors[0]
    source_occurrence = next(
        occurrence
        for occurrence in result.association_graph.source_revision.occurrences
        if occurrence.kind == "solid"
        and occurrence.entity_id == source_selector.entity_id
    )
    mapped = result.association_graph.resolve_target(
        result.association_graph.source_revision.select(source_occurrence.occurrence_id)
    )
    assert mapped.status is AssociationStatus.UNIQUE
    assert mapped.selector is not None and mapped.selector.kind == "solid"
    assert result.model.topology.num_solids == 1
    assert any(isinstance(patch, RevolutionSurface) for patch in result.model.patches)
    assert any(
        isinstance(curve, IntersectionCurve) for curve in result.model.geometry.curves
    )
    assert result.report.deleted_source_occurrences > 0
    assert cylinder.source_revision == cylinder_revision
    assert cylinder.geometry.geometry_id == cylinder_geometry


def test_revision_children_rejects_a_forged_parent_selector() -> None:
    parent = CADOccurrence(
        "revision",
        "root",
        "assembly-entity",
        "assembly",
        ("root",),
    )
    child = CADOccurrence(
        "revision",
        "face",
        "face-entity",
        "face",
        ("root", "face"),
        parent_occurrence_id="root",
    )
    revision = CADRevision("revision", "source", (parent, child), "provenance")
    forged = CADSelector(
        "revision",
        "root",
        "part",
        ("root",),
        "different-entity",
    )

    with pytest.raises(ValueError, match="does not match the revision inventory"):
        revision.children(forged)
