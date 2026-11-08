#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compute CAD Boolean and material-owner arrangements from source geometry.

Rectilinear solids use source-coordinate cells. Other surfaces
use certified coupled intersection curves and source-rooted trims. No
tessellation participates in classification, construction, or ancestry, and
unresolved source evidence never becomes a fitted or triangle-derived B-Rep.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import product

import phydrax.geometry.brep._boolean_coincidence as boolean_coincidence_owner
import phydrax.geometry.brep._boolean_full_overlay as boolean_full_overlay_owner
import phydrax.geometry.brep._placement as placement_owner

from ..._fingerprint import canonical_fingerprint
from ...typing import checked, parse
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._assembly import BRepAffinePlacement
from ._boolean_core import (
    assemble,
    BooleanSourceLineage,
    boundary,
    BoundaryFace,
    BRepBooleanFailure,
    BRepBooleanOperation,
    BRepBooleanPolicy,
    BRepBooleanResult,
    Cell,
    components,
    entity_occurrence_index,
    membership,
    rectangles,
    rectilinear_family,
    selected,
    source_assembly_occurrence_id,
    source_lineage,
    target_assembly_occurrence_id,
)
from ._model import BRepEntityId, BRepModel
from ._partition import (
    _assembly_namespace_occurrence_id,
    cad_revision_from_brep_model,
)
from ._patches import LineCurve


def _line_data(
    model: BRepModel, edge: int, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction, Fraction] | None:
    geometry = model.geometry
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        return None
    curve = geometry.curves[curve_index]
    if not isinstance(curve, LineCurve):
        return None
    origin = tuple(Fraction(float(value)) for value in curve.origin)
    direction = tuple(Fraction(float(value)) for value in curve.direction)
    first, last = (Fraction(float(value)) for value in geometry.edge_ranges[edge])
    if first >= last or not any(direction):
        return None
    return origin, direction, first, last


def _line_parameter(
    point: tuple[Fraction, ...],
    data: tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction, Fraction],
    /,
) -> Fraction | None:
    origin, direction, first, last = data
    axes = tuple(axis for axis, value in enumerate(direction) if value)
    if len(axes) != 1:
        return None
    axis = axes[0]
    parameter = (point[axis] - origin[axis]) / direction[axis]
    if (
        any(
            origin[coordinate] + parameter * direction[coordinate] != point[coordinate]
            for coordinate in range(3)
        )
        or not first <= parameter <= last
    ):
        return None
    return parameter


def _rectilinear_entity_pairs(
    models: tuple[BRepModel, BRepModel],
    source: BooleanSourceLineage,
    target_model: BRepModel,
    target_index: dict[tuple[str, int], tuple[tuple[BRepEntityId, str], ...]],
    faces: tuple[BoundaryFace, ...],
    /,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    target_geometry = target_model.geometry
    for face_index, boundary_face in enumerate(faces):
        target_face_occurrences = target_index.get(("face", face_index), ())
        if not target_face_occurrences:
            raise BRepBooleanFailure(
                "BRepBoolean.lineage.target_face must exist in the published CAD revision"
            )
        target_loops = target_geometry.face_loops[face_index]
        for operand, parent_solid, parent_face in boundary_face.sources:
            source_model = models[operand]
            source_geometry = source_model.geometry
            source_face_entities = source.entities_for_solid(
                operand, "face", parent_face, parent_solid
            )
            if not source_face_entities:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.source_face must have an exact occurrence under its source solid"
                )
            for source_face in source_face_entities:
                source_face_occurrence = source.occurrence_id(operand, source_face)
                for _, target_face_occurrence in target_face_occurrences:
                    pairs.add((source_face_occurrence, target_face_occurrence))

            source_edges: dict[
                int,
                tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction, Fraction],
            ] = {}
            source_vertex_points: dict[int, tuple[Fraction, ...]] = {}
            for source_loop in source_geometry.face_loops[parent_face]:
                for coedge in source_loop:
                    source_edge = source_geometry.coedge_edges[coedge]
                    data = _line_data(source_model, source_edge)
                    if data is None:
                        continue
                    source_edges[source_edge] = data
                    origin, direction, first, last = data
                    endpoints = (
                        tuple(
                            origin[axis] + first * direction[axis] for axis in range(3)
                        ),
                        tuple(origin[axis] + last * direction[axis] for axis in range(3)),
                    )
                    for vertex, point in zip(
                        source_geometry.edge_vertices[source_edge], endpoints, strict=True
                    ):
                        exact_point = source_geometry.vertex_exact_points[vertex]
                        if exact_point is not None and exact_point != point:
                            raise BRepBooleanFailure(
                                "BRepBoolean.lineage.source_vertex must match its exact source edge endpoint"
                            )
                        previous = source_vertex_points.get(vertex)
                        if previous is not None and previous != point:
                            raise BRepBooleanFailure(
                                "BRepBoolean.lineage.source_vertex has inconsistent exact edge endpoint definitions"
                            )
                        source_vertex_points[vertex] = point

            for target_loop in target_loops:
                for coedge in target_loop:
                    target_edge = target_geometry.coedge_edges[coedge]
                    target_edge_occurrences = target_index.get(("edge", target_edge), ())
                    first_vertex, last_vertex = target_geometry.edge_vertices[target_edge]
                    first_point = target_geometry.vertex_exact_points[first_vertex]
                    last_point = target_geometry.vertex_exact_points[last_vertex]
                    if first_point is not None and last_point is not None:
                        for source_edge, data in source_edges.items():
                            first_parameter = _line_parameter(first_point, data)
                            last_parameter = _line_parameter(last_point, data)
                            if (
                                first_parameter is None
                                or last_parameter is None
                                or first_parameter == last_parameter
                            ):
                                continue
                            source_edge_entities = source.entities_for_solid(
                                operand, "edge", source_edge, parent_solid
                            )
                            for source_edge_entity in source_edge_entities:
                                source_edge_occurrence = source.occurrence_id(
                                    operand, source_edge_entity
                                )
                                for _, target_edge_occurrence in target_edge_occurrences:
                                    pairs.add(
                                        (source_edge_occurrence, target_edge_occurrence)
                                    )
                    for target_vertex in (first_vertex, last_vertex):
                        target_point = target_geometry.vertex_exact_points[target_vertex]
                        if target_point is None:
                            continue
                        for source_vertex, source_point in source_vertex_points.items():
                            if source_point != target_point:
                                continue
                            source_vertex_entities = source.entities_for_solid(
                                operand, "vertex", source_vertex, parent_solid
                            )
                            for source_vertex_entity in source_vertex_entities:
                                source_vertex_occurrence = source.occurrence_id(
                                    operand, source_vertex_entity
                                )
                                for _, target_vertex_occurrence in target_index.get(
                                    ("vertex", target_vertex), ()
                                ):
                                    pairs.add(
                                        (
                                            source_vertex_occurrence,
                                            target_vertex_occurrence,
                                        )
                                    )
    return pairs


def _graph(
    source: BooleanSourceLineage,
    models: tuple[BRepModel, BRepModel],
    model: BRepModel,
    faces: tuple[BoundaryFace, ...],
    components: tuple[tuple[Cell, ...], ...],
    membership: dict[Cell, tuple[tuple[int, int], ...]],
    certificate: str,
) -> AssociationGraph:
    target = cad_revision_from_brep_model(model)
    target_index = entity_occurrence_index(model, target)
    pairs: set[tuple[str, str]] = set()
    for solid, component in enumerate(components):
        ancestors = {ancestor for cell in component for ancestor in membership[cell]}
        target_occurrences = target_index.get(("solid", solid), ())
        for operand, parent_solid in ancestors:
            source_entities = source.entities_for_index(operand, "solid", parent_solid)
            for source_entity in source_entities:
                source_occurrence = source.occurrence_id(operand, source_entity)
                source_assembly = source_assembly_occurrence_id(
                    source, operand, models[operand], source_entity.occurrence_path
                )
                for target_entity, target_occurrence in target_occurrences:
                    pairs.add((source_occurrence, target_occurrence))
                    target_assembly = target_assembly_occurrence_id(
                        target, model, target_entity.occurrence_path
                    )
                    pairs.add((source_assembly, target_assembly))
    for face_index, face in enumerate(faces):
        target_occurrences = target_index.get(("face", face_index), ())
        for operand, parent_solid, parent_face in face.sources:
            for source_entity in source.entities_for_solid(
                operand, "face", parent_face, parent_solid
            ):
                source_occurrence = source.occurrence_id(operand, source_entity)
                pairs.update(
                    (source_occurrence, target_occurrence)
                    for _, target_occurrence in target_occurrences
                )
    pairs.update(_rectilinear_entity_pairs(models, source, model, target_index, faces))
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision.revision_id,
        target.revision_id,
        tuple(
            OccurrenceCorrespondence(first, second, certificate)
            for first, second in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-rectilinear-cad-arrangement"
        ),
    )
    return AssociationGraph(source.revision, target, transaction)


def _namespace_paths(model: BRepModel, /) -> tuple[tuple[str, ...], ...]:
    geometry = model.geometry
    entity_paths = {
        record.member.occurrence_path
        for record in geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        )
        if record.container == record.member and record.member.occurrence_path
    }
    return tuple(
        sorted(
            entity_paths
            | {item.path for item in geometry.iter_assembly_containers()}
            | {item.path for item in geometry.iter_occurrences()}
        )
    )


def _coincident_result(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
) -> BRepBooleanResult:
    first = models[0]
    certificate = canonical_fingerprint(
        {
            "kind": "native-identical-cad-boolean",
            "source": first.model_id,
            "operation": operation,
        }
    )
    source = source_lineage(models, certificate)
    if operation == "difference":
        model = assemble(models, ((), (), ()), (), policy, certificate)
        pairs: tuple[OccurrenceCorrespondence, ...] = ()
    else:
        model = BRepModel(
            geometry=first.geometry,
            patches=first.patches,
            parameter_bounds=first.parameter_bounds,
            orientation=first.orientation,
            physical_tags=first.physical_tags,
            coordinate_contract=first.coordinate_contract,
            report=replace(
                first.report,
                source_digest=certificate,
                source_id=f"native-boolean:{certificate}",
            ),
            trim_domains=first.trim_domains,
            tessellation=first.tessellation,
        )
        target = cad_revision_from_brep_model(model)
        target_index = entity_occurrence_index(model, target)
        pair_values: set[tuple[str, str]] = set()
        for (kind, index), target_entities in target_index.items():
            for target_entity, target_occurrence in target_entities:
                for operand in range(len(models)):
                    for source_entity in source.entities_for_index(operand, kind, index):
                        if source_entity.occurrence_path == target_entity.occurrence_path:
                            pair_values.add(
                                (
                                    source.occurrence_id(operand, source_entity),
                                    target_occurrence,
                                )
                            )
        target_assembly = {
            occurrence.entity_id: occurrence.occurrence_id
            for occurrence in target.occurrences
            if occurrence.kind == "assembly"
        }
        for path in _namespace_paths(model):
            target_namespace = _assembly_namespace_occurrence_id(
                model.source_id, model.source_revision, path
            )
            target_occurrence = target_assembly.get(target_namespace)
            if target_occurrence is None:
                raise BRepBooleanFailure(
                    "BRepBoolean.lineage.target_assembly must exist in the canonical CAD revision",
                    (target_namespace,),
                )
            for operand, source_model in enumerate(models):
                source_namespace = _assembly_namespace_occurrence_id(
                    source_model.source_id, source_model.source_revision, path
                )
                source_occurrence = source.local_occurrence_ids.get(
                    (operand, source_namespace)
                )
                if source_occurrence is None:
                    raise BRepBooleanFailure(
                        "BRepBoolean.lineage.source_assembly must exist in the canonical CAD revision",
                        (source_namespace,),
                    )
                pair_values.add((source_occurrence, target_occurrence))
        pairs = tuple(
            OccurrenceCorrespondence(source_id, target_id, certificate)
            for source_id, target_id in sorted(pair_values)
        )
    target = cad_revision_from_brep_model(model)
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision.revision_id,
        target.revision_id,
        pairs,
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-identical-cad-boolean"
        ),
    )
    graph = AssociationGraph(source.revision, target, transaction)
    mapped = {edge.source_occurrence_id for edge in pairs}
    deleted = tuple(
        sorted(
            occurrence.occurrence_id
            for occurrence in source.revision.occurrences
            if occurrence.occurrence_id not in mapped
        )
    )
    return BRepBooleanResult(model, graph, operation, certificate, deleted)


def _boolean_world(
    first: BRepModel,
    second: BRepModel,
    operation: BRepBooleanOperation,
    /,
    *,
    policy: BRepBooleanPolicy | None = None,
) -> BRepBooleanResult:
    """Regularized native CAD union, intersection, or difference.

    Tangent boundary-only intersections are empty regularized solids. Coincident
    faces are classified once by arrangement occupancy. Disconnected components
    and enclosed void shells retain separate oriented topology.
    Rectilinear solids use source-coordinate cell arrangements. Other
    surfaces use exhaustive coupled intersection curves and source-rooted trims.
    Unresolved discovery, root identity, classification, or sewing never returns
    an approximate fitted or triangle-derived B-Rep.
    """
    if first.coordinate_contract.spatial_id != second.coordinate_contract.spatial_id:
        raise ValueError(
            "boolean_brep.operands.coordinate_contract must use the same spatial identity; got different spatial IDs"
        )
    operation_ = parse(operation, BRepBooleanOperation, "operation")
    policy_ = BRepBooleanPolicy() if policy is None else policy
    models = first, second
    for model in models:
        if model.geometry is None or (model.patches and not model.topology.num_solids):
            raise BRepBooleanFailure(
                "boolean_brep.operands must contain closed-solid geometry; got missing geometry or solids",
                (model.source_revision,),
            )
    if first.model_id == second.model_id and first.geometry is not None:
        return _coincident_result(models, operation_, policy_)
    if not rectilinear_family(models):
        coincident = boolean_coincidence_owner.coincident_boolean_brep(
            models, operation_, policy_
        )
        if coincident is not None:
            return coincident

        return boolean_full_overlay_owner.full_overlay_boolean_brep(
            models, operation_, policy_
        )
    rectangles_bank = rectangles(models)
    coordinates = tuple(
        tuple(
            sorted(
                {rectangle.lower[d] for rectangle in rectangles_bank}
                | {rectangle.upper[d] for rectangle in rectangles_bank}
            )
        )
        for d in range(3)
    )
    membership_bank: dict[Cell, tuple[tuple[int, int], ...]] = {}
    selected_bank: set[Cell] = set()
    for i, j, k in product(*(range(len(values) - 1) for values in coordinates)):
        cell = i, j, k
        point = (
            (Fraction(coordinates[0][i]) + Fraction(coordinates[0][i + 1])) / 2,
            (Fraction(coordinates[1][j]) + Fraction(coordinates[1][j + 1])) / 2,
            (Fraction(coordinates[2][k]) + Fraction(coordinates[2][k + 1])) / 2,
        )
        members = membership(point, rectangles_bank, models)
        membership_bank[cell] = members
        if selected(operation_, members):
            selected_bank.add(cell)
    certificate = canonical_fingerprint(
        {
            "kind": "native-rectilinear-cad-boolean",
            "sources": [model.model_id for model in models],
            "operation": operation_,
            "selected_cells": sorted(selected_bank),
            "coordinates": coordinates,
        }
    )
    source = source_lineage(models, certificate)
    components_bank = components(selected_bank)
    faces = boundary(components_bank, coordinates, rectangles_bank)
    model = assemble(models, coordinates, faces, policy_, certificate)
    graph = _graph(
        source, models, model, faces, components_bank, membership_bank, certificate
    )
    mapped = {edge.source_occurrence_id for edge in graph.transaction.correspondences}
    deleted = tuple(
        occurrence.occurrence_id
        for occurrence in source.revision.occurrences
        if occurrence.occurrence_id not in mapped
    )
    return BRepBooleanResult(model, graph, operation_, certificate, deleted)


@checked
def boolean_brep(
    first: BRepModel,
    second: BRepModel,
    operation: BRepBooleanOperation,
    /,
    *,
    policy: BRepBooleanPolicy | None = None,
) -> BRepBooleanResult:
    """Compute a Boolean of authored physical instances with source lineage."""
    policy_ = BRepBooleanPolicy() if policy is None else policy
    originals = (first, second)
    physical_instances = False
    identity = BRepAffinePlacement.identity()
    for model in originals:
        geometry = model.geometry
        occurrence_count = 0
        used_solids: set[int] = set()
        for occurrence in geometry.iter_occurrences():
            occurrence_count += 1
            if occurrence.placement != identity or occurrence.solid_index in used_solids:
                physical_instances = True
                break
            used_solids.add(occurrence.solid_index)
        if occurrence_count != model.topology.num_solids:
            physical_instances = True
    if not physical_instances:
        return _boolean_world(first, second, operation, policy=policy_)
    lifts = tuple(
        placement_owner.materialize_brep_occurrences(
            model, tessellation=policy_.tessellation
        )
        for model in originals
    )
    result = _boolean_world(lifts[0].model, lifts[1].model, operation, policy=policy_)
    certificate = canonical_fingerprint(
        {
            "kind": "exact-placed-cad-boolean-lineage",
            "sources": [model.model_id for model in originals],
            "placements": [lift.certificate_id for lift in lifts],
            "boolean": result.certificate_id,
        }
    )
    source = source_lineage(originals, certificate)
    intermediate = source_lineage((lifts[0].model, lifts[1].model), result.certificate_id)
    replacement: dict[str, set[str]] = {}
    unresolved_source: set[str] = set()
    for operand, lift in enumerate(lifts):
        for edge in lift.association_graph.transaction.correspondences:
            source_id = source.local_occurrence_ids.get(
                (operand, edge.source_occurrence_id)
            )
            intermediate_id = intermediate.local_occurrence_ids.get(
                (operand, edge.target_occurrence_id)
            )
            if source_id is None or intermediate_id is None:
                raise BRepBooleanFailure(
                    "placed Boolean lineage must bind every materialized source occurrence"
                )
            replacement.setdefault(intermediate_id, set()).add(source_id)
        for occurrence_id in lift.association_graph.transaction.unresolved_source_ids:
            source_id = source.local_occurrence_ids.get((operand, occurrence_id))
            if source_id is not None:
                unresolved_source.add(source_id)
    pairs = {
        (original, edge.target_occurrence_id)
        for edge in result.association_graph.transaction.correspondences
        for original in replacement.get(edge.source_occurrence_id, ())
    }
    for intermediate_id in result.association_graph.transaction.unresolved_source_ids:
        unresolved_source.update(replacement.get(intermediate_id, ()))
    unresolved_target = set(result.association_graph.transaction.unresolved_target_ids)
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision.revision_id,
        result.association_graph.target_revision.revision_id,
        tuple(
            OccurrenceCorrespondence(source_id, target_id, certificate)
            for source_id, target_id in sorted(pairs)
        ),
        frozenset(unresolved_source),
        frozenset(unresolved_target),
        AssociationCoverageEvidence(
            all(
                lift.association_graph.transaction.coverage.source_exhaustive
                for lift in lifts
            )
            and result.association_graph.transaction.coverage.source_exhaustive,
            all(
                lift.association_graph.transaction.coverage.target_exhaustive
                for lift in lifts
            )
            and result.association_graph.transaction.coverage.target_exhaustive,
            certificate,
            "native-exact-placed-cad-boolean",
        ),
    )
    graph = AssociationGraph(
        source.revision, result.association_graph.target_revision, transaction
    )
    mapped = {source_id for source_id, _ in pairs}
    deleted = tuple(
        sorted(
            occurrence.occurrence_id
            for occurrence in source.revision.occurrences
            if occurrence.occurrence_id not in mapped
            and occurrence.occurrence_id not in unresolved_source
        )
    )
    return BRepBooleanResult(result.model, graph, result.operation, certificate, deleted)


__all__ = ["boolean_brep"]
