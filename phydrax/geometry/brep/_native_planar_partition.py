#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact in-memory arrangements for polygonal planar mesh regions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization._exact_algebraic_geometry import (
    coefficient_payload,
    ExactScalar,
)
from .._atlas import TrimDomain
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from .._planar_embedding import PlanarEmbedding
from ..simplicial import PlanarMeshRegion
from ._constructors import assemble_brep_model, BRepTessellationPolicy
from ._intersection_curve import CurveTrimSegment
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._patches import ExactLineCurve, LineCurve, PlanePatch
from ._planar_arrangement import (
    arrange,
    Cell,
    Loops,
    on_segment,
    Point,
    rational_loops,
    Segment,
)


if TYPE_CHECKING:
    from ._partition import BRepPartitionResult
    from ._planar import PlanarPartitionOperand, PlanarPartitionPlan


@dataclass(frozen=True, slots=True)
class _PlanarSourceFace:
    operand_id: str
    face_index: int
    loops: Loops
    edge_segments: tuple[tuple[int, Segment], ...]
    vertex_points: tuple[tuple[int, Point], ...]
    source_label: str
    source_face_occurrence: CADOccurrence | None = None
    source_edge_occurrences: tuple[tuple[int, CADOccurrence], ...] = ()
    source_vertex_occurrences: tuple[tuple[int, CADOccurrence], ...] = ()


def _fraction_point(point: np.ndarray, /) -> Point:
    return Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1]))


def _rational_point3(
    point: tuple[ExactScalar, ...], /
) -> tuple[Fraction, Fraction, Fraction]:
    if len(point) != 3:
        raise ValueError("A planar B-rep source endpoint must be three-dimensional.")
    first, second, third = point
    if not (
        isinstance(first, Fraction)
        and isinstance(second, Fraction)
        and isinstance(third, Fraction)
    ):
        raise ValueError("Planar B-rep source edges require rational exact endpoints.")
    return first, second, third


def _world_exact(
    embedding: PlanarEmbedding, point: Point, /
) -> tuple[Fraction, Fraction, Fraction]:
    origin = tuple(Fraction.from_float(float(value)) for value in embedding.origin)
    first_axis = tuple(Fraction.from_float(float(value)) for value in embedding.x_axis)
    second_axis = tuple(Fraction.from_float(float(value)) for value in embedding.y_axis)
    return (
        origin[0] + point[0] * first_axis[0] + point[1] * second_axis[0],
        origin[1] + point[0] * first_axis[1] + point[1] * second_axis[1],
        origin[2] + point[0] * first_axis[2] + point[1] * second_axis[2],
    )


def _mesh_source(
    source: PlanarMeshRegion, /
) -> tuple[Loops, tuple[Segment, ...], tuple[tuple[int, Point], ...]]:
    vertices = np.asarray(source.vertices, dtype=np.float64)
    edges = np.asarray(source.edges, dtype=np.int64)
    offsets = np.asarray(source.loop_offsets, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 2
        or not np.all(np.isfinite(vertices))
        or edges.ndim != 2
        or edges.shape[1] != 2
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != len(edges)
        or np.any(np.diff(offsets) < 3)
        or np.any(edges < 0)
        or np.any(edges >= len(vertices))
    ):
        raise ValueError("Planar region source topology is malformed.")
    loops: list[tuple[Point, ...]] = []
    segments: list[Segment] = []
    vertex_points: dict[int, Point] = {}
    for loop_index in range(len(offsets) - 1):
        loop_edges = edges[offsets[loop_index] : offsets[loop_index + 1]]
        if np.any(loop_edges[:, 1] != np.roll(loop_edges[:, 0], -1)):
            raise ValueError("Planar region edges must form connected oriented loops.")
        loop_segments = tuple(
            (_fraction_point(vertices[first]), _fraction_point(vertices[last]))
            for first, last in loop_edges
        )
        if any(first == last for first, last in loop_segments):
            raise ValueError("Planar region loops cannot contain zero-length edges.")
        loops.append(tuple(first for first, _ in loop_segments))
        segments.extend(loop_segments)
        # Mesh vertex rows are the source vertex identities.
        vertex_points.update(
            (int(row), segment[0])
            for row, segment in zip(loop_edges[:, 0], loop_segments, strict=True)
        )
    return (
        rational_loops(tuple(loops)),
        tuple(segments),
        tuple(sorted(vertex_points.items())),
    )


def _exact_planar_point(
    point: tuple[Fraction, Fraction, Fraction], embedding: PlanarEmbedding, /
) -> Point:
    origin = tuple(Fraction.from_float(float(value)) for value in embedding.origin)
    first_axis = tuple(Fraction.from_float(float(value)) for value in embedding.x_axis)
    second_axis = tuple(Fraction.from_float(float(value)) for value in embedding.y_axis)
    delta = tuple(value - origin[index] for index, value in enumerate(point))

    def dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...], /) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    first_norm = dot(first_axis, first_axis)
    mixed = dot(first_axis, second_axis)
    second_norm = dot(second_axis, second_axis)
    determinant = first_norm * second_norm - mixed * mixed
    if determinant == 0:
        raise ValueError(
            "The planar source embedding has a singular exact coordinate frame."
        )
    first_projection = dot(delta, first_axis)
    second_projection = dot(delta, second_axis)
    planar = (
        (second_norm * first_projection - mixed * second_projection) / determinant,
        (first_norm * second_projection - mixed * first_projection) / determinant,
    )
    reconstructed = tuple(
        origin[index] + planar[0] * first_axis[index] + planar[1] * second_axis[index]
        for index in range(3)
    )
    if reconstructed != point:
        raise ValueError(
            "A planar B-rep source edge is not exactly in the operation embedding plane."
        )
    return planar


def _vector3(value: object, /) -> tuple[Fraction, Fraction, Fraction]:
    array = np.asarray(value, dtype=np.float64).reshape((3,))
    return (
        Fraction.from_float(float(array[0])),
        Fraction.from_float(float(array[1])),
        Fraction.from_float(float(array[2])),
    )


def _cross3(
    first: tuple[Fraction, Fraction, Fraction],
    second: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[Fraction, Fraction, Fraction]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot3(
    first: tuple[Fraction, Fraction, Fraction],
    second: tuple[Fraction, Fraction, Fraction],
    /,
) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _brep_source_faces(
    operand: PlanarPartitionOperand, embedding: PlanarEmbedding, /
) -> tuple[_PlanarSourceFace, ...]:
    from ._partition import _entity_text, cad_revision_from_brep_model

    source = operand.source
    if not isinstance(source, BRepModel):
        raise TypeError("Planar B-rep source faces require a BRepModel operand.")
    model = source
    geometry = model.geometry
    revision = cad_revision_from_brep_model(model)
    face_indices = {_entity_text(entity): entity.index for entity in model.face_ids}
    occurrences = {
        (occurrence.kind, occurrence.entity_id): occurrence
        for occurrence in revision.occurrences
        if occurrence.kind in ("edge", "vertex")
    }
    edge_texts = tuple(_entity_text(entity) for entity in model.edge_ids)
    vertex_texts = tuple(_entity_text(entity) for entity in model.vertex_ids)
    planar_normal = _cross3(_vector3(embedding.x_axis), _vector3(embedding.y_axis))
    selection = operand.selection
    if selection is None:
        raise RuntimeError("A persisted planar B-rep operand lost its face selection.")
    result: list[_PlanarSourceFace] = []
    for selector in sorted(
        selection.selectors,
        key=lambda value: face_indices[value.entity_id],
    ):
        face = face_indices.get(selector.entity_id)
        if face is None:
            raise ValueError(
                "A planar B-rep selection does not identify a native face row."
            )
        patch = model.patches[face]
        if not isinstance(patch, PlanePatch):
            raise ValueError(
                "Exact straight-loop planar partition admits only PlanePatch faces."
            )
        patch_normal = _cross3(_vector3(patch.first_axis), _vector3(patch.second_axis))
        if (
            patch_normal == (Fraction(0), Fraction(0), Fraction(0))
            or _cross3(patch_normal, planar_normal)
            != (Fraction(0), Fraction(0), Fraction(0))
            or _dot3(patch_normal, planar_normal)
            * int(np.asarray(model.orientation)[face])
            <= 0
        ):
            raise ValueError(
                "A planar B-rep face orientation must agree with the operation embedding."
            )
        _exact_planar_point(_vector3(patch.origin), embedding)
        face_occurrence = revision.occurrence(selector.occurrence_id)
        edge_segments: dict[int, Segment] = {}
        vertex_points: dict[int, Point] = {}
        for loop in geometry.face_loops[face]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if edge in edge_segments:
                    continue
                curve_index = geometry.edge_curves[edge]
                if curve_index < 0:
                    raise ValueError(
                        "Exact planar B-rep partition requires regular source edge curves."
                    )
                first_parameter, last_parameter = (
                    Fraction.from_float(float(value))
                    for value in np.asarray(geometry.edge_ranges)[edge]
                )
                curve = geometry.curves[curve_index]
                if not isinstance(curve, (LineCurve, ExactLineCurve)):
                    raise ValueError(
                        "Exact planar B-rep partition requires straight source edge curves."
                    )
                first_exact = _rational_point3(curve.evaluate_exact(first_parameter))
                last_exact = _rational_point3(curve.evaluate_exact(last_parameter))
                for point_index, point_exact in zip(
                    geometry.edge_vertices[edge], (first_exact, last_exact), strict=True
                ):
                    vertex_exact = geometry.vertex_exact_points[point_index]
                    if vertex_exact is None or tuple(vertex_exact) != point_exact:
                        raise ValueError(
                            "A planar B-rep line endpoint is not bound to its exact source vertex."
                        )
                edge_segments[edge] = (
                    _exact_planar_point(first_exact, embedding),
                    _exact_planar_point(last_exact, embedding),
                )
                vertex_points.update(
                    zip(geometry.edge_vertices[edge], edge_segments[edge], strict=True)
                )
        loops: list[tuple[Point, ...]] = []
        for loop in geometry.face_loops[face]:
            loop_points = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                segment = edge_segments[edge]
                loop_points.append(
                    segment[0 if geometry.coedge_senses[coedge] > 0 else 1]
                )
            loops.append(tuple(loop_points))
        source_loops = rational_loops(tuple(loops))
        source_edge_occurrences = []
        for edge in sorted(edge_segments):
            occurrence = occurrences.get(("edge", edge_texts[edge]))
            if occurrence is None:
                raise ValueError(
                    "A selected planar B-rep edge has no source occurrence identity."
                )
            source_edge_occurrences.append((edge, occurrence))
        source_vertex_occurrences = []
        for vertex in sorted(vertex_points):
            occurrence = occurrences.get(("vertex", vertex_texts[vertex]))
            if occurrence is None:
                raise ValueError(
                    "A selected planar B-rep vertex has no source occurrence identity."
                )
            source_vertex_occurrences.append((vertex, occurrence))
        result.append(
            _PlanarSourceFace(
                operand.operand_id,
                face,
                source_loops,
                tuple(sorted(edge_segments.items())),
                tuple(sorted(vertex_points.items())),
                model.source_id,
                face_occurrence,
                tuple(source_edge_occurrences),
                tuple(source_vertex_occurrences),
            )
        )
    return tuple(result)


def _planar_source_faces(plan: PlanarPartitionPlan, /) -> tuple[_PlanarSourceFace, ...]:
    sources: list[_PlanarSourceFace] = []
    for operand in sorted(plan.operands, key=lambda value: value.operand_id):
        source = operand.source
        if isinstance(source, PlanarMeshRegion):
            loops, segments, vertex_points = _mesh_source(source)
            sources.append(
                _PlanarSourceFace(
                    operand.operand_id,
                    0,
                    loops,
                    tuple(enumerate(segments)),
                    vertex_points,
                    source.feature_id,
                )
            )
        elif isinstance(source, BRepModel):
            sources.extend(_brep_source_faces(operand, plan.embedding))
        else:
            raise TypeError(
                "Planar partition sources must be PlanarMeshRegion or BRepModel values."
            )
    if not sources:
        raise ValueError("A planar partition requires at least one selected source face.")
    return tuple(sources)


def _contains_segment(inner: Segment, outer: Segment, /) -> bool:
    return on_segment(inner[0], outer) and on_segment(inner[1], outer)


type _ExactNested = Fraction | int | str | None | tuple[_ExactNested, ...]


def _exact_payload(value: _ExactNested, /) -> object:
    """Encode nested exact rational planar data as canonical identity payload."""
    if isinstance(value, Fraction):
        return coefficient_payload(value)
    if isinstance(value, tuple):
        return tuple(_exact_payload(item) for item in value)
    return value


def _source_revision(
    plan: PlanarPartitionPlan,
    sources: tuple[_PlanarSourceFace, ...],
    /,
) -> tuple[CADRevision, dict[tuple[str, str], str]]:
    revision_id = canonical_fingerprint(
        {
            "kind": "native-planar-partition-source-revision",
            "plan": plan.plan_id,
            "sources": tuple(
                (
                    source.operand_id,
                    source.face_index,
                    source.source_label,
                    _exact_payload(source.loops),
                    _exact_payload(source.edge_segments),
                    _exact_payload(source.vertex_points),
                    None
                    if source.source_face_occurrence is None
                    else source.source_face_occurrence.entity_id,
                    tuple(
                        (edge, occurrence.entity_id)
                        for edge, occurrence in source.source_edge_occurrences
                    ),
                    tuple(
                        (vertex, occurrence.entity_id)
                        for vertex, occurrence in source.source_vertex_occurrences
                    ),
                )
                for source in sources
            ),
        }
    )
    occurrences: list[CADOccurrence] = []
    identifiers: dict[tuple[str, str], str] = {}
    seen_source_edges: set[str] = set()
    seen_source_vertices: set[str] = set()
    for source in sources:
        original_face = source.source_face_occurrence
        face_id = (
            f"operand:{source.operand_id}:face:{source.face_index}"
            if original_face is None
            else f"operand:{source.operand_id}:source:{original_face.occurrence_id}"
        )
        face_entity_id = (
            f"{source.source_label}:face:{source.face_index}"
            if original_face is None
            else original_face.entity_id
        )
        occurrences.append(
            CADOccurrence(revision_id, face_id, face_entity_id, "face", (face_id,))
        )
        identifiers[(source.operand_id, f"face:{source.face_index}")] = face_id
        original_edges = dict(source.source_edge_occurrences)
        for edge_index, segment in source.edge_segments:
            original_edge = original_edges.get(edge_index)
            if original_edge is None:
                edge_id = f"{face_id}/edge:{edge_index}"
                orientation = 1 if segment[0] < segment[1] else -1
                if edge_id not in seen_source_edges:
                    occurrences.append(
                        CADOccurrence(
                            revision_id,
                            edge_id,
                            f"{source.source_label}:edge:{edge_index}",
                            "edge",
                            (face_id, edge_id),
                            face_id,
                            orientation,
                        )
                    )
                    seen_source_edges.add(edge_id)
            else:
                edge_id = (
                    f"operand:{source.operand_id}:source:{original_edge.occurrence_id}"
                )
                if edge_id not in seen_source_edges:
                    occurrences.append(
                        CADOccurrence(
                            revision_id,
                            edge_id,
                            original_edge.entity_id,
                            "edge",
                            (edge_id,),
                            orientation=original_edge.orientation,
                        )
                    )
                    seen_source_edges.add(edge_id)
            identifiers[(source.operand_id, f"edge:{edge_index}")] = edge_id
        original_vertices = dict(source.source_vertex_occurrences)
        for vertex_index, _point in source.vertex_points:
            original_vertex = original_vertices.get(vertex_index)
            if original_vertex is None:
                vertex_id = f"{face_id}/vertex:{vertex_index}"
                occurrence = CADOccurrence(
                    revision_id,
                    vertex_id,
                    f"{source.source_label}:vertex:{vertex_index}",
                    "vertex",
                    (face_id, vertex_id),
                    face_id,
                )
            else:
                vertex_id = (
                    f"operand:{source.operand_id}:source:{original_vertex.occurrence_id}"
                )
                occurrence = CADOccurrence(
                    revision_id,
                    vertex_id,
                    original_vertex.entity_id,
                    "vertex",
                    (vertex_id,),
                )
            if vertex_id not in seen_source_vertices:
                occurrences.append(occurrence)
                seen_source_vertices.add(vertex_id)
            identifiers[(source.operand_id, f"vertex:{vertex_index}")] = vertex_id
    return (
        CADRevision(
            revision_id,
            f"native-planar-partition:{plan.plan_id}",
            tuple(occurrences),
            plan.plan_id,
        ),
        identifiers,
    )


def _cells(
    plan: PlanarPartitionPlan, sources: tuple[_PlanarSourceFace, ...], /
) -> tuple[tuple[Cell, str], ...]:
    regions = tuple(source.loops for source in sources)
    voids = tuple(value for value in plan.operands if value.role.value == "void")
    owners: list[tuple[Cell, str]] = []
    for cell in arrange(regions):
        memberships = {sources[index].operand_id for index in cell.members}
        owner = next(
            (name for name in plan.policy.region_precedence if name in memberships),
            None,
        )
        if owner is None or any(
            owner in void.target_region_ids and void.operand_id in memberships
            for void in voids
        ):
            continue
        owners.append((cell, owner))
    if not owners:
        raise ValueError("Native planar partition removes every region cell.")
    if {owner for _, owner in owners} != set(plan.policy.region_precedence):
        raise ValueError("Every planar region must retain a positive-area cell.")
    rank = {name: index for index, name in enumerate(plan.policy.region_precedence)}
    owners.sort(key=lambda value: (value[0].loops, rank[value[1]]))
    return tuple(owners)


def _build_model(
    plan: PlanarPartitionPlan,
    cells: tuple[tuple[Cell, str], ...],
    tessellation: BRepTessellationPolicy | None,
    /,
) -> tuple[BRepModel, tuple[Segment, ...], tuple[Point, ...]]:
    embedding = plan.embedding
    vertices: list[Point] = []
    vertex_index: dict[Point, int] = {}
    curves: list[ExactLineCurve] = []
    edge_ranges: list[tuple[float, float]] = []
    edge_vertices: list[tuple[int, int]] = []
    edge_index: dict[Segment, int] = {}
    edge_keys: list[Segment] = []
    pcurves: list[ExactLineCurve] = []
    coedge_edges: list[int] = []
    coedge_senses: list[int] = []
    face_loops: list[tuple[tuple[int, ...], ...]] = []
    trims: list[TrimDomain] = []
    patches: list[PlanePatch] = []
    bounds: list[np.ndarray] = []
    tags: list[str] = []

    def vertex(point: Point, /) -> int:
        index = vertex_index.get(point)
        if index is None:
            index = len(vertices)
            vertex_index[point] = index
            vertices.append(point)
        return index

    for cell, owner in cells:
        face_coedge_loops: list[tuple[int, ...]] = []
        trim_loops: list[tuple[CurveTrimSegment, ...]] = []
        for points in cell.loops:
            loop: list[int] = []
            trim_loop: list[CurveTrimSegment] = []
            for index, start in enumerate(points):
                end = points[(index + 1) % len(points)]
                key = (start, end) if start < end else (end, start)
                edge = edge_index.get(key)
                if edge is None:
                    edge = len(curves)
                    edge_index[key] = edge
                    edge_keys.append(key)
                    curves.append(
                        ExactLineCurve(
                            _world_exact(embedding, key[0]),
                            _world_exact(embedding, key[1]),
                        )
                    )
                    edge_ranges.append((0.0, 1.0))
                    edge_vertices.append((vertex(key[0]), vertex(key[1])))
                coedge = len(coedge_edges)
                coedge_edges.append(edge)
                sense = 1 if start == key[0] else -1
                coedge_senses.append(sense)
                # The p-curve shares the edge's parameter; the coedge sense
                # carries the loop traversal.
                pcurve = ExactLineCurve(key[0], key[1])
                pcurves.append(pcurve)
                loop.append(coedge)
                trim_loop.append(CurveTrimSegment(pcurve, 0.0, 1.0, reversed=sense < 0))
            face_coedge_loops.append(tuple(loop))
            trim_loops.append(tuple(trim_loop))
        face_loops.append(tuple(face_coedge_loops))
        trims.append(TrimDomain.from_curve_loops(tuple(trim_loops)))
        patches.append(PlanePatch(embedding.origin, embedding.x_axis, embedding.y_axis))
        face_points = tuple(point for loop in cell.loops for point in loop)
        lower = tuple(min(point[axis] for point in face_points) for axis in range(2))
        upper = tuple(max(point[axis] for point in face_points) for axis in range(2))
        bounds.append(
            np.asarray(
                (
                    tuple(float(value) for value in lower),
                    tuple(float(value) for value in upper),
                ),
                dtype=np.float64,
            )
        )
        tags.append(owner)

    exact_world_vertices = tuple(_world_exact(embedding, point) for point in vertices)
    geometry = BRepGeometry(
        vertex_points=np.asarray(
            [tuple(float(value) for value in point) for point in exact_world_vertices],
            dtype=np.float64,
        ).reshape((-1, 3)),
        vertex_exact_points=exact_world_vertices,
        curves=tuple(curves),
        edge_curves=tuple(range(len(curves))),
        edge_ranges=np.asarray(edge_ranges, dtype=np.float64).reshape((-1, 2)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(coedge_senses),
        face_loops=tuple(face_loops),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    source_digest = canonical_fingerprint(
        {
            "kind": "native-planar-partition",
            "plan": plan.plan_id,
            "cells": tuple((_exact_payload(cell.loops), owner) for cell, owner in cells),
        }
    )
    model = assemble_brep_model(
        geometry,
        tuple(patches),
        np.asarray(bounds, dtype=np.float64),
        np.ones(len(patches), dtype=np.float64),
        tuple(tags),
        coordinate_contract=plan.coordinate_contract,
        source_id=f"native-planar-partition:{plan.plan_id}",
        source_digest=source_digest,
        tessellation=tessellation,
        trim_domains=tuple(trims),
    )
    return model, tuple(edge_keys), tuple(vertices)


def partition_native_planar_arrangement(
    plan: PlanarPartitionPlan,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> BRepPartitionResult:
    """Partition exact straight-loop sources by rational arrangement in memory."""
    from ._partition import (
        _entity_text,
        _operand_composite_graph,
        BRepPartitionPatch,
        BRepPartitionRegion,
        BRepPartitionReport,
        BRepPartitionResult,
        cad_revision_from_brep_model,
    )

    source_faces = _planar_source_faces(plan)
    cells = _cells(plan, source_faces)
    model, target_edge_keys, target_vertex_points = _build_model(
        plan, cells, tessellation
    )
    source_revision, source_ids = _source_revision(plan, source_faces)
    operand_graphs = []
    for operand in plan.operands:
        if not isinstance(operand.source, BRepModel):
            continue
        mapping: dict[str, str] = {}
        for source in source_faces:
            if source.operand_id != operand.operand_id:
                continue
            if source.source_face_occurrence is not None:
                mapping[source.source_face_occurrence.occurrence_id] = source_ids[
                    (operand.operand_id, f"face:{source.face_index}")
                ]
            for index, occurrence in source.source_edge_occurrences:
                mapping[occurrence.occurrence_id] = source_ids[
                    (operand.operand_id, f"edge:{index}")
                ]
            for index, occurrence in source.source_vertex_occurrences:
                mapping[occurrence.occurrence_id] = source_ids[
                    (operand.operand_id, f"vertex:{index}")
                ]
        operand_graphs.append(
            (
                operand.operand_id,
                _operand_composite_graph(
                    cad_revision_from_brep_model(operand.source),
                    source_revision,
                    mapping,
                    operand.operand_id,
                ),
            )
        )
    target_revision = cad_revision_from_brep_model(model)
    role_by_id = {operand.operand_id: operand.role.value for operand in plan.operands}
    rank = {name: index for index, name in enumerate(plan.policy.region_precedence)}
    face_owner = tuple(owner for _, owner in cells)
    source_edge_pairs = {
        (
            source_ids[(source.operand_id, f"edge:{edge_index}")],
            segment,
        )
        for source in source_faces
        for edge_index, segment in source.edge_segments
    }
    source_edges = tuple(sorted(source_edge_pairs))
    target_entities = {
        _entity_text(record.member): record.member
        for record in model.geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        )
    }
    target_occurrences: dict[tuple[str, int, tuple[str, ...]], str] = {}
    for occurrence in target_revision.occurrences:
        entity = target_entities.get(occurrence.entity_id)
        if entity is not None:
            target_occurrences[(entity.kind, entity.index, entity.occurrence_path)] = (
                occurrence.occurrence_id
            )
    target_edges: dict[int, list[str]] = {}
    target_vertices: dict[int, list[str]] = {}
    for (kind, index, _path), occurrence_id in target_occurrences.items():
        if kind == "edge":
            target_edges.setdefault(index, []).append(occurrence_id)
        elif kind == "vertex":
            target_vertices.setdefault(index, []).append(occurrence_id)
    correspondence_pairs: set[tuple[str, str]] = set()
    for face_index, (cell, _) in enumerate(cells):
        target_face_id = target_occurrences[("face", face_index, ())]
        for member in cell.members:
            source = source_faces[member]
            if role_by_id[source.operand_id] == "region":
                correspondence_pairs.add(
                    (
                        source_ids[(source.operand_id, f"face:{source.face_index}")],
                        target_face_id,
                    )
                )
    for edge_index, segment in enumerate(target_edge_keys):
        for source_id, source_segment in source_edges:
            if _contains_segment(segment, source_segment):
                correspondence_pairs.update(
                    (source_id, target_id)
                    for target_id in target_edges.get(edge_index, ())
                )
    # A target vertex descends from every source vertex at its exact point;
    # arrangement intersection vertices have no source vertex and are created.
    source_vertices: dict[Point, set[str]] = {}
    for source in source_faces:
        for vertex_index, point in source.vertex_points:
            source_vertices.setdefault(point, set()).add(
                source_ids[(source.operand_id, f"vertex:{vertex_index}")]
            )
    for vertex_index, point in enumerate(target_vertex_points):
        for source_id in source_vertices.get(point, ()):
            correspondence_pairs.update(
                (source_id, target_id)
                for target_id in target_vertices.get(vertex_index, ())
            )
    certificate_id = canonical_fingerprint(
        {
            "kind": "native-planar-arrangement-history",
            "plan": plan.plan_id,
            "source_revision": source_revision.revision_id,
            "target_revision": target_revision.revision_id,
            "correspondences": tuple(sorted(correspondence_pairs)),
        }
    )
    correspondences = tuple(
        OccurrenceCorrespondence(source_id, target_id, certificate_id)
        for source_id, target_id in sorted(correspondence_pairs)
    )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate_id,
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(correspondences),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate_id, "native-planar-arrangement"
        ),
    )
    graph = AssociationGraph(source_revision, target_revision, transaction)
    regions = tuple(
        BRepPartitionRegion(
            name,
            tuple(
                model.face_ids[index]
                for index, owner in enumerate(face_owner)
                if owner == name
            ),
            2,
        )
        for name in plan.policy.region_precedence
    )
    grouped_edges: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for edge_index, faces in enumerate(model.topology.edge_faces):
        adjacent = tuple(
            sorted((face_owner[index] for index in faces), key=rank.__getitem__)
        )
        if len(adjacent) not in (1, 2):
            raise RuntimeError("Native planar arrangement produced a nonmanifold edge.")
        grouped_edges.setdefault(adjacent, []).append(model.edge_ids[edge_index])
    patches = tuple(
        BRepPartitionPatch(
            "boundary:" + ":".join(adjacent),
            tuple(edge_ids),
            adjacent,
            2,
        )
        for adjacent, edge_ids in sorted(
            grouped_edges.items(),
            key=lambda item: tuple(rank[value] for value in item[0]),
        )
    )
    source_linked = {value.source_occurrence_id for value in correspondences}
    target_linked = {value.target_occurrence_id for value in correspondences}
    report = BRepPartitionReport(
        plan.plan_id,
        model.model_id,
        source_revision.revision_id,
        target_revision.revision_id,
        certificate_id,
        0,
        sum(value.kind == "face" for value in source_revision.occurrences),
        0,
        model.topology.num_faces,
        sum(
            value.occurrence_id not in source_linked
            for value in source_revision.occurrences
        ),
        sum(
            value.occurrence_id not in target_linked
            for value in target_revision.occurrences
        ),
        sum(value.kind == "edge" for value in source_revision.occurrences),
        model.topology.num_edges,
        2,
    )
    return BRepPartitionResult(
        model,
        target_revision,
        graph,
        regions,
        patches,
        report,
        2,
        operand_association_graphs=tuple(operand_graphs),
    )


__all__ = ["partition_native_planar_arrangement"]
