#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    AssociationStatus,
    CADOccurrence,
    CADRevision,
    CADSelectionSet,
    CADSelector,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
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
from phydrax.geometry.design import RecipeEntityName, RecipeEntityNames, RecipeNamingError
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
    with pytest.raises(ValueError, match="deleted"):
        graph.resolve_target_selection(
            CADSelectionSet(graph.source_revision.revision_id, (void_selector,))
        )
    assert result.report.deleted_source_occurrences > 0


def test_changed_partition_branch_refuses_old_selection() -> None:
    region = _box(0.0, 3.0)
    first = _execute(
        _plan(
            (
                _operand("region", region),
                _operand("void", _box(1.0), BRepPartitionRole.VOID, targets=("region",)),
            ),
            ("region",),
        )
    )
    changed = _execute(
        _plan(
            (
                _operand("region", region),
                _operand(
                    "void", _box(0.0, 1.0), BRepPartitionRole.VOID, targets=("region",)
                ),
            ),
            ("region",),
        )
    )
    old = CADSelectionSet.from_revision(
        first.association_graph.source_revision, ("operand:region:solid:0",)
    )
    with pytest.raises(ValueError, match="another source CAD revision"):
        changed.association_graph.resolve_target_selection(old)


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
    selection_a = CADSelectionSet.from_revision(
        graph.source_revision, ("operand:a:solid:0",)
    )
    with pytest.raises(ValueError, match="partial merge"):
        graph.resolve_target_selection(selection_a)
    both = CADSelectionSet.from_revision(
        graph.source_revision, ("operand:a:solid:0", "operand:b:solid:0")
    )
    descendants = graph.resolve_target_selection(both)
    assert len(descendants.selectors) == 2
    assert len(MeshingScope.from_geometry(model, descendants).entity_ids) == 2
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


def test_exact_selection_composes_split_and_merge_without_partial_reattachment() -> None:
    def revision(identifier: str, entities: tuple[str, ...]) -> CADRevision:
        return CADRevision(
            identifier,
            "source",
            tuple(
                CADOccurrence(identifier, entity, entity, "face", (entity,))
                for entity in entities
            ),
            "provenance",
        )

    def graph(
        source: CADRevision, target: CADRevision, pairs: tuple[tuple[str, str], ...]
    ) -> AssociationGraph:
        evidence = AssociationCoverageEvidence(True, True, "coverage", "test")
        transaction = OccurrenceCorrespondenceTransaction(
            f"{source.revision_id}-{target.revision_id}",
            source.revision_id,
            target.revision_id,
            tuple(OccurrenceCorrespondence(a, b, "exact") for a, b in pairs),
            frozenset(),
            frozenset(),
            evidence,
        )
        return AssociationGraph(source, target, transaction)

    first = revision("first", ("wall",))
    second = revision("second", ("left", "right"))
    third = revision("third", ("merged",))
    split = graph(first, second, (("wall", "left"), ("wall", "right")))
    merge = graph(second, third, (("left", "merged"), ("right", "merged")))
    wall = first.select("wall")
    assert split.resolve_target(wall).status is AssociationStatus.MULTIPLE
    with pytest.raises(ValueError, match="multiple"):
        split.resolve_target(wall).require_unique()
    descendants = split.resolve_target_selection(CADSelectionSet("first", (wall,)))
    assert {value.occurrence_id for value in descendants.selectors} == {"left", "right"}
    with pytest.raises(ValueError, match="partial merge"):
        merge.resolve_target_selection(
            CADSelectionSet("second", (second.select("left"),))
        )
    merged = merge.resolve_target_selection(descendants)
    assert merged.selectors == (third.select("merged"),)


def test_selection_orientation_requires_certified_rebase_in_both_directions() -> None:
    source = CADRevision(
        "orientation-source",
        "source",
        (
            CADOccurrence(
                "orientation-source",
                "face",
                "entity",
                "face",
                ("face",),
                orientation=-1,
            ),
        ),
        "provenance",
    )
    target = CADRevision(
        "orientation-target",
        "target",
        (
            CADOccurrence(
                "orientation-target",
                "face",
                "entity",
                "face",
                ("face",),
                orientation=1,
            ),
        ),
        "provenance",
    )

    def graph(transform: int | None) -> AssociationGraph:
        return AssociationGraph(
            source,
            target,
            OccurrenceCorrespondenceTransaction(
                "orientation-transaction",
                source.revision_id,
                target.revision_id,
                (OccurrenceCorrespondence("face", "face", "incidence-proof", transform),),
                frozenset(),
                frozenset(),
                AssociationCoverageEvidence(True, True, "coverage", "test"),
            ),
        )

    source_selection = CADSelectionSet.from_revision(source, ("face",))
    target_selection = CADSelectionSet.from_revision(target, ("face",))
    for transform in (None, 0, 1):
        with pytest.raises(ValueError, match="orientation"):
            graph(transform).resolve_target_selection(source_selection)
        with pytest.raises(ValueError, match="orientation"):
            graph(transform).resolve_preimage_selection(target_selection)
        with pytest.raises(ValueError, match="orientation"):
            graph(transform).resolve_target(source.select("face")).require_unique()
        with pytest.raises(ValueError, match="orientation"):
            graph(transform).resolve_preimage(target.select("face")).require_unique()
    certified = graph(-1)
    assert certified.graph_id != graph(None).graph_id
    assert certified.resolve_target_selection(source_selection) == target_selection
    assert certified.resolve_preimage_selection(target_selection) == source_selection
    assert certified.resolve_target(
        source.select("face")
    ).require_unique() == target.select("face")
    assert certified.resolve_preimage(
        target.select("face")
    ).require_unique() == source.select("face")
    with pytest.raises(ValueError, match="orientation_transform"):
        OccurrenceCorrespondence("face", "face", "proof", 2)


def test_unresolved_orientation_refuses_even_equal_incidences() -> None:
    source = CADRevision(
        "uncertain-source",
        "source",
        (CADOccurrence("uncertain-source", "face", "face", "face", ("face",)),),
        "provenance",
    )
    target = CADRevision(
        "uncertain-target",
        "target",
        (CADOccurrence("uncertain-target", "face", "face", "face", ("face",)),),
        "provenance",
    )
    graph = AssociationGraph(
        source,
        target,
        OccurrenceCorrespondenceTransaction(
            "uncertain",
            source.revision_id,
            target.revision_id,
            (OccurrenceCorrespondence("face", "face", "ancestry-only", 0),),
            frozenset(),
            frozenset(),
            AssociationCoverageEvidence(True, True, "coverage", "test"),
        ),
    )
    with pytest.raises(ValueError, match="orientation"):
        graph.resolve_target(source.select("face")).require_unique()
    with pytest.raises(ValueError, match="orientation"):
        graph.resolve_preimage(target.select("face")).require_unique()
    with pytest.raises(ValueError, match="orientation"):
        graph.resolve_target_selection(CADSelectionSet.from_revision(source, ("face",)))
    with pytest.raises(ValueError, match="orientation"):
        graph.resolve_preimage_selection(CADSelectionSet.from_revision(target, ("face",)))


def test_selection_orientation_rejects_merge_with_opposite_target() -> None:
    source = CADRevision(
        "merge-source",
        "source",
        tuple(
            CADOccurrence("merge-source", name, name, "face", (name,), orientation=-1)
            for name in ("first", "second")
        ),
        "provenance",
    )
    target = CADRevision(
        "merge-target",
        "target",
        (CADOccurrence("merge-target", "merged", "merged", "face", ("merged",)),),
        "provenance",
    )
    selection = CADSelectionSet.from_revision(source, ("first", "second"))

    def graph(second_transform: int | None) -> AssociationGraph:
        return AssociationGraph(
            source,
            target,
            OccurrenceCorrespondenceTransaction(
                "merge-transaction",
                source.revision_id,
                target.revision_id,
                (
                    OccurrenceCorrespondence("first", "merged", "proof-first", -1),
                    OccurrenceCorrespondence(
                        "second", "merged", "proof-second", second_transform
                    ),
                ),
                frozenset(),
                frozenset(),
                AssociationCoverageEvidence(True, True, "coverage", "test"),
            ),
        )

    with pytest.raises(ValueError, match="orientation"):
        graph(None).resolve_target_selection(selection)
    assert graph(-1).resolve_target_selection(selection) == CADSelectionSet.from_revision(
        target, ("merged",)
    )


def test_authored_name_binds_only_its_exact_qualified_source_occurrence() -> None:
    region = _box(0.0, 3.0)
    void = _box(1.0)
    result = _execute(
        _plan(
            (
                _operand("region", region),
                _operand("void", void, BRepPartitionRole.VOID, targets=("region",)),
            ),
            ("region",),
        )
    )
    names = tuple(
        RecipeEntityName("face", "region", f"wall-{entity.index}", ("authored",))
        for entity in region.face_ids
    )
    bindings = RecipeEntityNames(region, names, None)
    qualified = next(
        record.member
        for record in region.geometry.qualified_entity_incidence(
            region.source_id, region.source_revision
        )
        if record.member.kind == "face" and record.member.index == 0
    )
    from phydrax.geometry.brep._partition import _entity_text

    graph = result.association_graph
    occurrence = next(
        value
        for value in graph.source_revision.occurrences
        if value.kind == "face" and value.entity_id == _entity_text(qualified)
    )
    selection = bindings.select_occurrence(
        names[0], graph.source_revision, occurrence.occurrence_id, qualified
    )
    assert selection.selectors == (
        graph.source_revision.select(occurrence.occurrence_id),
    )
    with pytest.raises(RecipeNamingError, match="definition"):
        bindings.select_occurrence(
            names[1], graph.source_revision, occurrence.occurrence_id, qualified
        )


def test_partition_selection_cold_archive_and_tampered_history_refusal(
    tmp_path: Any,
) -> None:
    import dataclasses
    import subprocess
    import sys

    from phydrax._array_archive import ArrayArchiveCorruptionError
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    result = _execute(
        _plan(
            (
                _operand("region", _box(0.0, 3.0)),
                _operand("void", _box(1.0), BRepPartitionRole.VOID, targets=("region",)),
            ),
            ("region",),
        )
    )
    graph = result.association_graph
    selection = graph.resolve_target_selection(
        CADSelectionSet.from_revision(graph.source_revision, ("operand:region:solid:0",))
    )

    def composition(history: AssociationGraph, selected: CADSelectionSet) -> Composition:
        return Composition(
            (
                CompositionEntry(
                    result,
                    entry_id="partition",
                    role="model-state",
                    owner_id="cad",
                    structure_id=result.model.model_id,
                    revision_id=result.revision.revision_id,
                    semantics_id="exact-partition",
                ),
                CompositionEntry(
                    history,
                    entry_id="history",
                    role="model-state",
                    owner_id="cad",
                    structure_id=history.graph_id,
                    revision_id=history.graph_id,
                    semantics_id="exact-history",
                ),
                CompositionEntry(
                    selected,
                    entry_id="selection",
                    role="model-state",
                    owner_id="cad",
                    structure_id=selected.selection_id,
                    revision_id=selected.revision_id,
                    semantics_id="descendants",
                    dependencies=(),
                ),
            ),
            boundary_id="cad-selection",
        )

    path = write_model_artifact(
        tmp_path / "selection.phx",
        composition(graph, selection),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    restored = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.value("history").graph_id == graph.graph_id
    assert restored.value("selection") == selection
    assert restored.value("partition").operand_association_graph("region").graph_id == (
        result.operand_association_graph("region").graph_id
    )
    inner = composition(graph, selection)
    outer = Composition(
        (
            CompositionEntry(
                inner,
                entry_id="nested",
                role="model-state",
                owner_id="cad",
                structure_id=inner.composition_id,
                revision_id=inner.composition_id,
                semantics_id="nested-cad",
            ),
        ),
        boundary_id="outer",
    )
    nested_path = write_model_artifact(
        tmp_path / "nested.phx", outer, limits=MODEL_ARCHIVE_LIMITS
    )
    nested = read_model_artifact(
        nested_path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert nested.value("nested").value("history").graph_id == graph.graph_id
    assert nested.value("nested").value("selection") == selection

    renamed_regions = tuple(
        dataclasses.replace(region, name="forged-region") for region in result.regions
    )
    renamed_adjacency = tuple(
        dataclasses.replace(
            patch,
            adjacent_region_ids=tuple("forged-region" for _ in patch.adjacent_region_ids),
        )
        for patch in result.patches
    )
    forged_results = (
        dataclasses.replace(result, regions=renamed_regions, patches=renamed_adjacency),
        dataclasses.replace(
            result,
            patches=(
                dataclasses.replace(result.patches[0], name="forged-patch"),
                *result.patches[1:],
            ),
        ),
    )
    for index, forged_result in enumerate(forged_results):
        forged_owner = Composition(
            (
                CompositionEntry(
                    forged_result,
                    entry_id="partition",
                    role="model-state",
                    owner_id="cad",
                    structure_id=result.model.model_id,
                    revision_id=result.revision.revision_id,
                    semantics_id="exact-partition",
                ),
            ),
            boundary_id="cad-owner-forgery",
        )
        forged_owner_path = write_model_artifact(
            tmp_path / f"forged-owner-{index}.phx",
            forged_owner,
            limits=MODEL_ARCHIVE_LIMITS,
        )
        with pytest.raises((ArrayArchiveCorruptionError, ValueError)):
            read_model_artifact(
                forged_owner_path,
                expected_type=Composition,
                restore=restore_model_archive_value,
                limits=MODEL_ARCHIVE_LIMITS,
            )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from phydrax.lifecycle import Composition; "
                "from phydrax.meshing._model_codec import "
                "MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value; "
                "c=read_model_artifact(sys.argv[1], expected_type=Composition, "
                "restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS); "
                "assert c.value('history').graph_id == sys.argv[2]; "
                "assert c.value('selection').selection_id == sys.argv[3]; "
                "assert c.value('partition').operand_association_graph('region').graph_id == sys.argv[4]"
            ),
            str(path),
            graph.graph_id,
            selection.selection_id,
            result.operand_association_graph("region").graph_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    # A graph's own recomputed fingerprint authenticates bytes, not CAD history.
    # Keep inventories and coverage current while forging a correspondence.
    pairs = {
        (item.source_occurrence_id, item.target_occurrence_id)
        for item in graph.transaction.correspondences
    }
    original, unrelated = next(
        (item, target)
        for item in graph.transaction.correspondences
        for target in graph.target_revision.occurrences
        if target.kind == graph.source_revision.occurrence(item.source_occurrence_id).kind
        and (item.source_occurrence_id, target.occurrence_id) not in pairs
    )
    edge = dataclasses.replace(
        original,
        target_occurrence_id=unrelated.occurrence_id,
        evidence_id="forged-evidence",
    )
    transaction = dataclasses.replace(
        graph.transaction,
        correspondences=tuple(
            edge if item == original else item
            for item in graph.transaction.correspondences
        ),
    )
    rehashed_graph = AssociationGraph(
        graph.source_revision, graph.target_revision, transaction
    )
    forged_result = dataclasses.replace(result, association_graph=rehashed_graph)
    forged_composition = Composition(
        tuple(
            CompositionEntry(
                forged_result
                if entry.entry_id == "partition"
                else rehashed_graph
                if entry.entry_id == "history"
                else entry.value,
                entry_id=entry.entry_id,
                role=entry.role,
                owner_id=entry.owner_id,
                structure_id=rehashed_graph.graph_id
                if entry.entry_id == "history"
                else entry.structure_id,
                revision_id=rehashed_graph.graph_id
                if entry.entry_id == "history"
                else entry.revision_id,
                semantics_id=entry.semantics_id,
                dependencies=entry.dependencies,
            )
            for entry in composition(graph, selection).entries
        ),
        boundary_id="cad-selection",
    )
    forged_path = write_model_artifact(
        tmp_path / "rehashed-history.phx", forged_composition, limits=MODEL_ARCHIVE_LIMITS
    )
    with pytest.raises((ArrayArchiveCorruptionError, ValueError)):
        read_model_artifact(
            forged_path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )

    altered = dataclasses.replace(
        graph.source_revision.occurrences[0],
        orientation=-graph.source_revision.occurrences[0].orientation,
    )
    revision = dataclasses.replace(
        graph.source_revision,
        occurrences=(altered, *graph.source_revision.occurrences[1:]),
    )
    forged = AssociationGraph(revision, graph.target_revision, graph.transaction)
    object.__setattr__(forged, "graph_id", graph.graph_id)
    path = write_model_artifact(
        tmp_path / "forged.phx",
        composition(forged, selection),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(
        (ArrayArchiveCorruptionError, ValueError), match="identity|incompatible"
    ):
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_planar_overlay_cold_partition_archive(tmp_path: Any) -> None:
    import subprocess
    import sys

    from phydrax.geometry._planar_embedding import PlanarEmbedding
    from phydrax.geometry.brep._planar import (
        partition_planar,
        PlanarPartitionOperand,
        PlanarPartitionPlan,
    )
    from phydrax.geometry.simplicial import PlanarMeshRegion
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, write_model_artifact

    def rectangle(x: float, name: str) -> PlanarMeshRegion:
        return PlanarMeshRegion(
            np.asarray(((x, 0), (x + 2, 0), (x + 2, 1), (x, 1))),
            ((0, 1, 2, 3),),
            feature_id=name,
        )

    embedding = PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    plan = PlanarPartitionPlan(
        _COORDINATES,
        embedding,
        (
            PlanarPartitionOperand(
                "left", rectangle(0, "left"), BRepPartitionRole.REGION
            ),
            PlanarPartitionOperand(
                "right", rectangle(1, "right"), BRepPartitionRole.REGION
            ),
        ),
        BRepPartitionPolicy(("left", "right")),
    )
    result = partition_planar(plan)
    composition = Composition(
        (
            CompositionEntry(
                result,
                entry_id="partition",
                role="model-state",
                owner_id="cad",
                structure_id=result.model.model_id,
                revision_id=result.revision.revision_id,
                semantics_id="exact-partition",
            ),
        ),
        boundary_id="planar-partition",
    )
    path = write_model_artifact(
        tmp_path / "planar.phx", composition, limits=MODEL_ARCHIVE_LIMITS
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from phydrax.lifecycle import Composition; "
            "from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, "
            "read_model_artifact, restore_model_archive_value; "
            "c=read_model_artifact(sys.argv[1], expected_type=Composition, "
            "restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS); "
            "p=c.value('partition'); "
            "assert p.plan.plan_id == sys.argv[2]; "
            "assert p.model.model_id == sys.argv[3]; "
            "assert p.association_graph.graph_id == sys.argv[4]; "
            "assert p.plan.operands[0].source.feature_id == 'left'",
            str(path),
            plan.plan_id,
            result.model.model_id,
            result.association_graph.graph_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr


def test_partition_operand_graph_composes_real_operations() -> None:
    original = _box(0.0, 3.0)
    first = _execute(_plan((_operand("body", original),), ("body",)))
    local = cad_revision_from_brep_model(original)
    selected = CADSelectionSet.from_revision(
        local,
        tuple(item.occurrence_id for item in local.occurrences if item.kind == "solid"),
    )
    selected = first.operand_association_graph("body").resolve_target_selection(selected)
    selected = first.association_graph.resolve_target_selection(selected)

    second = _execute(
        _plan(
            (
                _operand("body", first.model),
                _operand("void", _box(1.0), BRepPartitionRole.VOID, targets=("body",)),
            ),
            ("body",),
        )
    )
    selected = second.operand_association_graph("body").resolve_target_selection(selected)
    resolution = second.association_graph.resolve_target(selected.selectors[0])
    assert resolution.status is AssociationStatus.MULTIPLE
    with pytest.raises(ValueError, match="multiple"):
        resolution.require_unique()
    selected = second.association_graph.resolve_target_selection(selected)
    assert len(selected.selectors) == 2
    assert len(MeshingScope.from_geometry(second.model, selected).entity_ids) == 2


def test_box_operand_face_incidence_has_certified_orientation() -> None:
    model = _box(0.0)
    result = _execute(_plan((_operand("body", model),), ("body",)))
    graph = result.operand_association_graph("body")
    edge = next(
        edge
        for edge in graph.transaction.correspondences
        if edge.orientation_transform == 1
        and graph.source_revision.occurrence(edge.source_occurrence_id).kind == "face"
    )
    source = CADSelectionSet.from_revision(
        graph.source_revision, (edge.source_occurrence_id,)
    )
    transported = graph.resolve_target_selection(source)
    assert transported.selectors == (
        graph.target_revision.select(edge.target_occurrence_id),
    )
    assert graph.source_revision.occurrence(edge.source_occurrence_id).orientation == 1
    assert graph.target_revision.occurrence(edge.target_occurrence_id).orientation == 1


def test_partition_rejects_missing_operand_history() -> None:
    import dataclasses

    result = _l_shaped_partition()
    with pytest.raises(ValueError, match="exactly cover"):
        dataclasses.replace(result, operand_association_graphs=())
    with pytest.raises(ValueError, match="exactly cover"):
        dataclasses.replace(
            result, operand_association_graphs=result.operand_association_graphs[:1]
        )


def test_void_face_lineage_carries_certified_cavity_orientation() -> None:
    result = _execute(
        _plan(
            (
                _operand("region", _box(0.0, 3.0)),
                _operand("void", _box(1.0), BRepPartitionRole.VOID, targets=("region",)),
            ),
            ("region",),
        )
    )
    graph = result.association_graph
    edge = next(
        item
        for item in graph.transaction.correspondences
        if item.source_occurrence_id.startswith("operand:void:")
        and graph.source_revision.occurrence(item.source_occurrence_id).kind == "face"
        and graph.target_revision.occurrence(item.target_occurrence_id).kind == "face"
        and item.orientation_transform in (-1, 1)
        and len(graph._reverse_ids(item.target_occurrence_id)) == 1
    )
    source = CADSelectionSet.from_revision(
        graph.source_revision, (edge.source_occurrence_id,)
    )
    target = graph.resolve_target_selection(source)
    assert graph.target_revision.select(edge.target_occurrence_id) in target.selectors
    assert MeshingScope.from_geometry(result.model, target).entity_dimension == 2
