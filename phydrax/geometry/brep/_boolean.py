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
from math import prod

import phydrax.geometry.brep._boolean_coincidence as boolean_coincidence_owner
import phydrax.geometry.brep._boolean_full_overlay as boolean_full_overlay_owner
import phydrax.geometry.brep._placement as placement_owner

from ..._fingerprint import canonical_fingerprint
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._boolean_core import (
    assemble,
    boundary,
    BoundaryFace,
    BRepBooleanFailure,
    BRepBooleanOperation,
    BRepBooleanPolicy,
    BRepBooleanResult,
    Cell,
    components,
    membership,
    rectangles,
    rectilinear_family,
    selected,
    source_revision,
)
from ._coordinate_request import coordinate_source_request
from ._model import BRepModel
from ._request_preparations import TrimRequestPreparations
from ._revision import cad_revision_from_brep_model


def _graph(
    source: CADRevision,
    model: BRepModel,
    faces: tuple[BoundaryFace, ...],
    components: tuple[tuple[Cell, ...], ...],
    membership: dict[Cell, tuple[tuple[int, int], ...]],
    certificate: str,
) -> AssociationGraph:

    target = cad_revision_from_brep_model(model)
    pairs = set()
    for solid, component in enumerate(components):
        ancestors = {ancestor for cell in component for ancestor in membership[cell]}
        for operand, parent_solid in ancestors:
            pairs.add((f"operand:{operand}/solid:{parent_solid}", f"solid:{solid}"))
    for face_index, face in enumerate(faces):
        for operand, parent_solid, parent_face in face.sources:
            pairs.add(
                (
                    f"operand:{operand}/solid:{parent_solid}/face:{parent_face}",
                    f"solid:{face.solid}/face:{face_index}",
                )
            )
    correspondences = tuple(
        OccurrenceCorrespondence(first, second, certificate)
        for first, second in sorted(pairs)
    )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        correspondences,
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-rectilinear-cad-arrangement"
        ),
    )
    return AssociationGraph(source, target, transaction)


def _coincident_result(
    models: tuple[BRepModel, BRepModel],
    operation: BRepBooleanOperation,
    policy: BRepBooleanPolicy,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> BRepBooleanResult:

    first = models[0]
    certificate = canonical_fingerprint(
        {
            "kind": "native-identical-cad-boolean",
            "source": first.model_id,
            "operation": operation,
        }
    )
    source = source_revision(models, certificate)
    if operation == "difference":
        model = assemble(
            models,
            ((), (), ()),
            (),
            policy,
            certificate,
            preparations=preparations,
            budget=budget,
        )
        pairs = ()
    else:
        if first.topology.num_faces > policy.maximum_faces:
            raise BRepBooleanFailure(
                "boolean_brep.faces must fit maximum_faces; got face allocation budget exhausted"
            )
        model = first.evolve(
            report=replace(
                first.report,
                source_digest=certificate,
                source_id=f"native-boolean:{certificate}",
            ),
            budget=budget,
            preparations=preparations,
        )
        pairs = tuple(
            OccurrenceCorrespondence(
                occurrence.occurrence_id,
                occurrence.occurrence_id.split("/", 1)[1],
                certificate,
            )
            for occurrence in source.occurrences
        )
    target = cad_revision_from_brep_model(model)
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source.revision_id,
        target.revision_id,
        pairs,
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-identical-cad-boolean"
        ),
    )
    graph = AssociationGraph(source, target, transaction)
    deleted = (
        tuple(occurrence.occurrence_id for occurrence in source.occurrences)
        if operation == "difference"
        else ()
    )
    return BRepBooleanResult(model, graph, operation, certificate, deleted)


def _boolean_world(
    first: BRepModel,
    second: BRepModel,
    operation: BRepBooleanOperation,
    /,
    *,
    policy: BRepBooleanPolicy | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
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
        return _coincident_result(
            models, operation_, policy_, preparations=preparations, budget=budget
        )
    if not rectilinear_family(models):
        coincident = boolean_coincidence_owner.coincident_boolean_brep(
            models, operation_, policy_, preparations=preparations, budget=budget
        )
        if coincident is not None:
            return coincident

        return boolean_full_overlay_owner.full_overlay_boolean_brep(
            models, operation_, policy_, preparations=preparations, budget=budget
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
    count = prod(max(0, len(values) - 1) for values in coordinates)
    if count > policy_.maximum_cells:
        raise BRepBooleanFailure(
            "boolean_brep.arrangement.cells must fit maximum_cells; got arrangement cell budget exhausted",
            tuple(model.source_revision for model in models),
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
    source = source_revision(models, certificate)
    components_bank = components(selected_bank)
    faces = boundary(components_bank, coordinates, rectangles_bank, policy_)
    model = assemble(
        models,
        coordinates,
        faces,
        policy_,
        certificate,
        preparations=preparations,
        budget=budget,
    )
    graph = _graph(source, model, faces, components_bank, membership_bank, certificate)
    mapped = {edge.source_occurrence_id for edge in graph.transaction.correspondences}
    deleted = tuple(
        occurrence.occurrence_id
        for occurrence in source.occurrences
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
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepBooleanResult:
    """Compute a Boolean of authored physical instances with source lineage."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                first,
                second,
                operation,
            )
        ) as budget:
            return boolean_brep(
                first,
                second,
                operation,
                policy=policy,
                preparations=preparations,
                budget=budget,
            )

    policy_ = BRepBooleanPolicy() if policy is None else policy
    originals = (first, second)
    physical_instances = False
    for model in originals:
        geometry = model.geometry
        if geometry is None:
            raise BRepBooleanFailure(
                "boolean_brep.operands.geometry must be defined; got None"
            )
        for occurrence in geometry.occurrences:
            evidence = occurrence.placement_evidence
            if evidence is not None and evidence.point_deviation_bound > 0.0:
                raise BRepBooleanFailure(
                    "boolean_brep.occurrences must have supported displacement evidence; "
                    f"got placement-deviation at path={occurrence.path!r}, "
                    f"bound={evidence.point_deviation_bound}",
                    ("/".join(occurrence.path),),
                )
        physical_instances |= (
            any(
                occurrence.rotation != ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                or occurrence.translation != (0.0, 0.0, 0.0)
                for occurrence in geometry.occurrences
            )
            or len(geometry.occurrences) != model.topology.num_solids
        )
    if not physical_instances:
        return _boolean_world(
            first,
            second,
            operation,
            policy=policy_,
            preparations=preparations,
            budget=budget,
        )
    lifts = tuple(
        placement_owner.materialize_brep_occurrences(
            model,
            tessellation=policy_.tessellation,
            preparations=preparations,
            budget=budget,
        )
        for model in originals
    )
    result = _boolean_world(
        lifts[0].model,
        lifts[1].model,
        operation,
        policy=policy_,
        preparations=preparations,
        budget=budget,
    )
    certificate = canonical_fingerprint(
        {
            "kind": "exact-placed-cad-boolean-lineage",
            "sources": [model.model_id for model in originals],
            "placements": [lift.certificate_id for lift in lifts],
            "boolean": result.certificate_id,
        }
    )
    occurrences: list[CADOccurrence] = []
    replacement: dict[str, tuple[str, ...]] = {}
    for operand, lift in enumerate(lifts):
        reverse: dict[tuple[str, int], list[str]] = {}
        source_inventory = {
            occurrence.occurrence_id: occurrence
            for occurrence in lift.association_graph.source_revision.occurrences
        }
        for source, target in zip(
            lift.source_entities, lift.target_entities, strict=True
        ):
            if source.kind not in ("solid", "face"):
                continue
            identifier = f"operand:{operand}/{placement_owner.entity_occurrence(source)}"
            source_occurrence = source_inventory[
                placement_owner.entity_occurrence(source)
            ]
            occurrences.append(
                CADOccurrence(
                    certificate,
                    identifier,
                    placement_owner.entity_text(source),
                    source.kind,
                    (identifier,),
                    orientation=source_occurrence.orientation,
                )
            )
            reverse.setdefault((target.kind, target.index), []).append(identifier)
        for source_occurrence in result.association_graph.source_revision.occurrences:
            prefix = f"operand:{operand}/"
            if not source_occurrence.occurrence_id.startswith(prefix):
                continue
            fields = source_occurrence.occurrence_id[len(prefix) :].split("/")
            solid = int(fields[0].removeprefix("solid:"))
            if source_occurrence.kind == "solid":
                replacement[source_occurrence.occurrence_id] = tuple(
                    reverse[("solid", solid)]
                )
            elif source_occurrence.kind == "face":
                face = int(fields[1].removeprefix("face:"))
                replacement[source_occurrence.occurrence_id] = tuple(
                    reverse[("face", face)]
                )
    source_revision = CADRevision(
        certificate,
        f"native-placed-boolean-sources:{certificate}",
        tuple(occurrences),
        certificate,
    )
    pairs = {
        (original, edge.target_occurrence_id)
        for edge in result.association_graph.transaction.correspondences
        for original in replacement[edge.source_occurrence_id]
    }
    transaction = OccurrenceCorrespondenceTransaction(
        certificate,
        source_revision.revision_id,
        result.association_graph.target_revision.revision_id,
        tuple(
            OccurrenceCorrespondence(source, target, certificate)
            for source, target in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate, "native-exact-placed-cad-boolean"
        ),
    )
    graph = AssociationGraph(
        source_revision, result.association_graph.target_revision, transaction
    )
    mapped = {source for source, _ in pairs}
    deleted = tuple(
        sorted(
            occurrence.occurrence_id
            for occurrence in occurrences
            if occurrence.occurrence_id not in mapped
        )
    )
    return BRepBooleanResult(result.model, graph, result.operation, certificate, deleted)


__all__ = ["boolean_brep"]
