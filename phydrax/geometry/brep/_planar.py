#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Partition planar operands through source, arrangement, and publication phases.

Operand construction validates CAD selections once; subsequent phases consume
the narrowed selection and publish a model through its validated evolve path.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import equinox as eqx
import numpy as np

from ..._external_resource import ResourceLimits
from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._validation import normalized_identifier
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    CADSelectionSet,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from .._planar_embedding import PlanarEmbedding
from ..simplicial import PlanarMeshRegion
from ._boolean_core import precedence_owner
from ._constructors import BRepTessellationPolicy, Builder, finish
from ._coordinate_request import coordinate_source_request
from ._model import BRepEntityId, BRepModel
from ._partition import (
    all_brep_faces,
    BRepPartitionHistoryError,
    BRepPartitionPatch,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionReport,
    BRepPartitionResult,
    BRepPartitionRole,
    publish_staged,
)
from ._patches import LineCurve, PlanePatch
from ._planar_arrangement import (
    area,
    arrange,
    Cell,
    curve_segment,
    face_loops,
    on_segment,
    point,
    rational_loops,
    require_coplanar_face,
    Segment,
)
from ._request_preparations import TrimRequestPreparations
from ._revision import cad_revision_from_brep_model, edge_occurrence_id


@dataclass(frozen=True, slots=True)
class PlanarPartitionOperand:
    """Select a planar mesh region or B-Rep root faces for a 2D partition."""

    operand_id: str
    source: PlanarMeshRegion | BRepModel
    role: BRepPartitionRole
    selection: CADSelectionSet | None = None
    target_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operand_id = normalized_identifier(str(self.operand_id), "operand_id")
        role = BRepPartitionRole(self.role)
        source = self.source
        selection = self.selection
        match source:
            case PlanarMeshRegion():
                if selection is not None:
                    raise ValueError(
                        "PlanarPartitionOperand.selection must be None for mesh operands; got a CAD selection"
                    )
            case BRepModel():
                if source.topology.num_solids:
                    raise ValueError(
                        "PlanarPartitionOperand.source.topology must contain root faces and zero solids; got solid topology"
                    )
                revision = cad_revision_from_brep_model(source)
                if selection is None:
                    selection = all_brep_faces(source)
                if not isinstance(selection, CADSelectionSet):
                    raise TypeError(
                        "PlanarPartitionOperand.selection must be a CADSelectionSet or None; got an invalid selection"
                    )
                selection.require_kind("face")
                if not selection.selectors:
                    raise ValueError(
                        "PlanarPartitionOperand.selection must select at least one face; got an empty selection"
                    )
                if selection.revision_id != revision.revision_id:
                    raise ValueError(
                        "PlanarPartitionOperand.selection must use the source revision; got another B-Rep revision"
                    )
                for selector in selection.selectors:
                    occurrence = revision.occurrence(selector.occurrence_id)
                    if occurrence.parent_occurrence_id is not None:
                        raise ValueError(
                            "PlanarPartitionOperand.selection must contain only root faces; got a nested occurrence"
                        )
                    if revision.select(selector.occurrence_id) != selector:
                        raise ValueError(
                            "PlanarPartitionOperand.selection must match the B-Rep occurrence inventory; got an incompatible selector"
                        )
            case _:
                raise TypeError(
                    "PlanarPartitionOperand.source must be a PlanarMeshRegion or BRepModel; got an unsupported source"
                )
        target_region_ids = tuple(
            normalized_identifier(str(value), "target_region_id")
            for value in self.target_region_ids
        )
        if len(set(target_region_ids)) != len(target_region_ids):
            raise ValueError(
                "PlanarPartitionOperand.target_region_ids must be unique; got a repeated target region"
            )
        if role is BRepPartitionRole.REGION and target_region_ids:
            raise ValueError(
                "PlanarPartitionOperand.target_region_ids must be empty for regions; got void targets"
            )
        if role is BRepPartitionRole.VOID and not target_region_ids:
            raise ValueError(
                "PlanarPartitionOperand.target_region_ids must be nonempty for voids; got no target region"
            )
        object.__setattr__(self, "operand_id", operand_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "target_region_ids", target_region_ids)

    @property
    def face_selection(self) -> CADSelectionSet:
        """Return the face selection validated during operand construction."""
        selection = self.selection
        if selection is None:
            raise ValueError(
                "PlanarPartitionOperand.selection must select B-Rep faces; got None"
            )
        return selection


@dataclass(frozen=True, slots=True)
class PlanarPartitionPlan:
    """Describe a 2D CAD partition in one physical world embedding."""

    coordinate_contract: SpatialCoordinateContract
    embedding: PlanarEmbedding
    operands: tuple[PlanarPartitionOperand, ...]
    policy: BRepPartitionPolicy
    plan_id: str = field(init=False)
    topological_dimension: int = field(init=False, default=2)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError(
                "PlanarPartitionPlan.coordinate_contract must be a SpatialCoordinateContract; got an invalid contract"
            )
        if self.coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                "PlanarPartitionPlan.coordinate_contract must use Cartesian coordinates; got a non-Cartesian coordinate system"
            )
        if not isinstance(self.embedding, PlanarEmbedding):
            raise TypeError(
                "PlanarPartitionPlan.embedding must be a PlanarEmbedding; got an invalid embedding"
            )
        operands = tuple(self.operands)
        if not operands or any(
            not isinstance(value, PlanarPartitionOperand) for value in operands
        ):
            raise TypeError(
                "PlanarPartitionPlan.operands must contain at least one PlanarPartitionOperand; got invalid operands"
            )
        if not isinstance(self.policy, BRepPartitionPolicy):
            raise TypeError(
                "PlanarPartitionPlan.policy must be a BRepPartitionPolicy; got an invalid policy"
            )
        operand_ids = tuple(value.operand_id for value in operands)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError(
                "PlanarPartitionPlan.operands must have unique operand IDs; got repeated IDs"
            )
        region_ids = {
            value.operand_id
            for value in operands
            if value.role is BRepPartitionRole.REGION
        }
        if set(self.policy.region_precedence) != region_ids:
            raise ValueError(
                "PlanarPartitionPlan.policy.region_precedence must name every planar region once; got a region inventory mismatch"
            )
        for operand in operands:
            if (
                isinstance(operand.source, BRepModel)
                and operand.source.coordinate_contract.spatial_id
                != self.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "PlanarPartitionOperand.source must use the plan coordinate contract; got a coordinate contract mismatch"
                )
            if (
                operand.role is BRepPartitionRole.VOID
                and not set(operand.target_region_ids) <= region_ids
            ):
                raise ValueError(
                    "PlanarPartitionOperand.target_region_ids must name declared regions; got an unknown partition region"
                )
        plan_id = canonical_fingerprint(
            {
                "kind": "planar-brep-partition-plan",
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "embedding": self.embedding.embedding_id,
                "topological_dimension": self.topological_dimension,
                "operands": sorted(_operand_descriptor(value) for value in operands),
                "region_precedence": self.policy.region_precedence,
                "overwrite": self.policy.overwrite,
                "run_parallel": self.policy.run_parallel,
                "maximum_cells": self.policy.maximum_cells,
                "maximum_faces": self.policy.maximum_faces,
                "sewing_tolerance": self.policy.sewing.tolerance,
                "maximum_coedges": self.policy.sewing.maximum_coedges,
            }
        )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "plan_id", plan_id)


def _operand_descriptor(operand: PlanarPartitionOperand) -> tuple[object, ...]:
    source = operand.source
    if isinstance(source, BRepModel):
        selection = operand.face_selection
        source_descriptor: object = (
            "brep",
            source.model_id,
            selection.selection_id,
        )
    else:
        source_descriptor = (
            "planar-mesh-region",
            source.feature_id,
            np.asarray(source.vertices),
            np.asarray(source.edges),
            np.asarray(source.loop_offsets),
        )
    return (
        operand.operand_id,
        source_descriptor,
        operand.role.value,
        tuple(sorted(operand.target_region_ids)),
    )


@checked
def mesh_loops(region: PlanarMeshRegion, /) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    edges = np.asarray(region.edges, dtype=np.int64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError(
            "PlanarMeshRegion.vertices must be finite with shape (N, 2); got invalid vertices"
        )
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != edges.shape[0]
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ValueError(
            "PlanarMeshRegion.loop_offsets must delimit valid indexed loops; got malformed loop topology"
        )
    loops: list[np.ndarray] = []
    for loop_index, (start, stop) in enumerate(pairwise(offsets)):
        loop_edges = edges[int(start) : int(stop)]
        vertex_ids = loop_edges[:, 0]
        if (
            vertex_ids.size < 3
            or np.any(loop_edges[:, 1] != np.roll(vertex_ids, -1))
            or np.any(vertex_ids < 0)
            or np.any(vertex_ids >= vertices.shape[0])
            or len({int(value) for value in vertex_ids}) != vertex_ids.size
        ):
            raise ValueError(
                "PlanarMeshRegion.edges must form simple closed edge cycles; got an invalid loop"
            )
        points = vertices[vertex_ids]
        if len({tuple(value) for value in points}) != points.shape[0]:
            raise ValueError(
                "PlanarMeshRegion.vertices must be distinct within each loop; got a repeated point"
            )
        area = _signed_area(points)
        if (loop_index == 0 and area <= 0.0) or (loop_index > 0 and area >= 0.0):
            raise ValueError(
                "PlanarMeshRegion.loops must have a counter-clockwise outer loop and clockwise holes; got an invalid orientation"
            )
        loops.append(points)
    _validate_loop_arrangement(tuple(loops))
    return tuple(loops)


def _signed_area(points: np.ndarray, /) -> float:
    return float(area(tuple(point(point_bank) for point_bank in points)))


def _validate_loop_arrangement(loops: tuple[np.ndarray, ...], /) -> None:
    rational_loops(loops)


@dataclass(frozen=True, slots=True)
class _NativeSourceFace:
    operand_id: str
    occurrence: CADOccurrence
    loops: tuple[np.ndarray, ...]
    edges: tuple[tuple[CADOccurrence, Segment], ...]


class _NativeOperand(StrictModule):
    identity: str = eqx.field(static=True)
    items: tuple[tuple[int, tuple[np.ndarray, ...], BRepModel | None], ...]


class _PlanarArrangement(StrictModule):
    source_revision: CADRevision
    sources: tuple[_NativeSourceFace, ...]
    cells: tuple[Cell, ...] = eqx.field(static=True)
    owners: tuple[str, ...] = eqx.field(static=True)


def _prepare_native_operand(
    plan: PlanarPartitionPlan, operand: PlanarPartitionOperand, /
) -> _NativeOperand:
    source = operand.source
    if isinstance(source, PlanarMeshRegion):
        identity = canonical_fingerprint(
            {
                "kind": "embedded-planar-mesh-region",
                "feature_id": source.feature_id,
                "vertices": np.asarray(source.vertices),
                "edges": np.asarray(source.edges),
                "loop_offsets": np.asarray(source.loop_offsets),
                "coordinate_contract": plan.coordinate_contract.spatial_id,
                "embedding": plan.embedding.embedding_id,
            }
        )
        items = ((0, mesh_loops(source), None),)
    else:
        identity = source.source_revision
        selection = operand.face_selection
        selected_ids = {selector.occurrence_id for selector in selection.selectors}
        items = tuple(
            (index, face_loops(source, index, plan.embedding), source)
            for index in range(source.topology.num_faces)
            if f"face:{index}" in selected_ids
        )
    return _NativeOperand(identity, items)


def _source_segments(
    model: BRepModel | None,
    face_index: int,
    loops: tuple[np.ndarray, ...],
    embedding: PlanarEmbedding,
    /,
) -> tuple[tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction], int], ...]:
    if model is None:
        segments = tuple(
            (index, point(first), point(second), 1)
            for index, (first, second) in enumerate(
                (
                    pair
                    for loop in loops
                    for pair in zip(loop, np.roll(loop, -1, axis=0), strict=True)
                )
            )
        )
    else:
        geometry = model.require_geometry()
        segments = tuple(
            (
                edge,
                point(segment[0]),
                point(segment[1]),
                next(
                    geometry.coedge_senses[coedge]
                    for loop in geometry.face_loops[face_index]
                    for coedge in loop
                    if geometry.coedge_edges[coedge] == edge
                ),
            )
            for edge in model.topology.face_edges[face_index]
            for segment in (curve_segment(model, edge, embedding),)
        )
    return segments


def _native_sources(
    plan: PlanarPartitionPlan, /
) -> tuple[CADRevision, tuple[_NativeSourceFace, ...]]:
    revision_id = canonical_fingerprint(
        {"kind": "planar-brep-partition-source-revision", "plan_id": plan.plan_id}
    )
    occurrences: list[CADOccurrence] = []
    faces: list[_NativeSourceFace] = []
    for operand in sorted(plan.operands, key=lambda value: value.operand_id):
        prepared = _prepare_native_operand(plan, operand)
        identity, items = prepared.identity, prepared.items
        for face_index, loops, model in items:
            face_id = f"operand:{operand.operand_id}:face:{face_index}"
            occurrence = CADOccurrence(
                revision_id, face_id, f"{identity}:face:{face_index}", "face", (face_id,)
            )
            occurrences.append(occurrence)
            edge_rows = []
            segments = _source_segments(model, face_index, loops, plan.embedding)
            for edge_index, first, second, sense in segments:
                edge_id = f"{face_id}/edge:{edge_index}"
                edge_occurrence = CADOccurrence(
                    revision_id,
                    edge_id,
                    f"{identity}:edge:{edge_index}",
                    "edge",
                    (face_id, edge_id),
                    face_id,
                    sense,
                )
                occurrences.append(edge_occurrence)
                edge_rows.append((edge_occurrence, (first, second)))
            faces.append(
                _NativeSourceFace(operand.operand_id, occurrence, loops, tuple(edge_rows))
            )
    return CADRevision(
        revision_id,
        f"planar-brep-partition-input:{plan.plan_id}",
        tuple(occurrences),
        plan.plan_id,
    ), tuple(faces)


def _cell_owner(
    plan: PlanarPartitionPlan, sources: tuple[_NativeSourceFace, ...], cell: Cell, /
) -> str | None:
    members = frozenset(sources[index].operand_id for index in cell.members)
    voids = tuple(
        (operand.operand_id, operand.target_region_ids)
        for operand in plan.operands
        if operand.role is BRepPartitionRole.VOID
    )
    return precedence_owner(plan.policy.region_precedence, voids, members)


def _stage_cells(
    cells: tuple[Cell, ...],
    owners: tuple[str, ...],
    embedding: PlanarEmbedding,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[Builder, tuple[Segment, ...]]:
    builder = Builder()
    # Rational arrangement atoms determine intersection incidence before their
    # coordinates are rounded to the native carrier representation.
    points = sorted({point for cell in cells for loop in cell.loops for point in loop})
    emitted = tuple(
        tuple(
            float(value)
            for value in embedding.to_world(np.asarray(point, dtype=np.float64))
        )
        for point in points
    )
    if len(set(emitted)) != len(points):
        raise BRepPartitionHistoryError(
            "partition_planar.intersections must remain distinct at carrier precision; got collapsed intersections"
        )
    vertices = {
        point: builder.vertex(embedding.to_world(np.asarray(point, dtype=np.float64)))
        for point in points
    }
    edge_ids: dict[Segment, int] = {}
    for cell, owner in zip(cells, owners, strict=True):
        loops = []
        for loop in cell.loops:
            coedges = []
            for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
                start, end = sorted((first, second))
                atom = (start, end)
                if atom not in edge_ids:
                    world_start, world_end = embedding.to_world(
                        np.asarray(atom, dtype=np.float64)
                    )
                    edge_ids[atom] = builder.edge(
                        LineCurve(world_start, world_end - world_start),
                        0.0,
                        1.0,
                        vertices[start],
                        vertices[end],
                    )
                coedges.append(
                    builder.coedge(
                        edge_ids[atom],
                        1 if first == start else -1,
                        LineCurve(
                            np.asarray(start, dtype=np.float64),
                            np.asarray(end, dtype=np.float64)
                            - np.asarray(start, dtype=np.float64),
                        ),
                        preparations=preparations,
                        budget=budget,
                    )
                )
            loops.append(coedges)
        bounds = np.asarray(cell.loops[0], dtype=np.float64)
        builder.face(
            PlanePatch(embedding.origin, embedding.x_axis, embedding.y_axis),
            np.stack((np.min(bounds, axis=0), np.max(bounds, axis=0))),
            loops,
            owner,
            None,
        )
    return builder, tuple(edge_ids)


def _native_history(
    plan: PlanarPartitionPlan,
    source_revision: CADRevision,
    sources: tuple[_NativeSourceFace, ...],
    model: BRepModel,
    cells: tuple[Cell, ...],
    atoms: tuple[Segment, ...],
    /,
) -> AssociationGraph:
    target_revision = cad_revision_from_brep_model(model)
    pairs: set[tuple[str, str]] = set()
    for face, cell in enumerate(cells):
        for member in cell.members:
            pairs.add((sources[member].occurrence.occurrence_id, f"face:{face}"))
    for edge, atom in enumerate(atoms):
        for source in sources:
            for occurrence, segment in source.edges:
                if all(on_segment(point, segment) for point in atom):
                    for face in model.topology.edge_faces[edge]:
                        pairs.add(
                            (occurrence.occurrence_id, edge_occurrence_id(face, edge))
                        )
    source_ids = {value.occurrence_id for value in source_revision.occurrences}
    target_ids = {value.occurrence_id for value in target_revision.occurrences}
    if {target for _, target in pairs} != target_ids:
        raise BRepPartitionHistoryError(
            "partition_planar.provenance must exhaust the final face-edge inventory; got missing incidence"
        )
    certificate_id = canonical_fingerprint(
        {
            "kind": "native-rational-planar-arrangement-history",
            "plan_id": plan.plan_id,
            "target": model.model_id,
            "pairs": sorted(pairs),
            "deleted": sorted(source_ids - {source for source, _ in pairs}),
        }
    )
    transaction = OccurrenceCorrespondenceTransaction(
        canonical_fingerprint(
            {"kind": "planar-partition-transaction", "certificate": certificate_id}
        ),
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(
            OccurrenceCorrespondence(
                source,
                target,
                canonical_fingerprint(
                    {
                        "kind": "rational-arrangement-incidence",
                        "source": source,
                        "target": target,
                        "certificate": certificate_id,
                    }
                ),
            )
            for source, target in sorted(pairs)
        ),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True, True, certificate_id, "phydrax-native-rational-planar-arrangement"
        ),
    )
    return AssociationGraph(source_revision, target_revision, transaction)


def _regions_and_patches(
    plan: PlanarPartitionPlan, model: BRepModel, owners: tuple[str, ...], /
) -> tuple[tuple[BRepPartitionRegion, ...], tuple[BRepPartitionPatch, ...]]:
    regions = tuple(
        BRepPartitionRegion(
            name,
            tuple(
                model.face_ids[index]
                for index, owner in enumerate(owners)
                if owner == name
            ),
            2,
        )
        for name in plan.policy.region_precedence
    )
    rank = {name: index for index, name in enumerate(plan.policy.region_precedence)}
    groups: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for edge, faces in enumerate(model.topology.edge_faces):
        if len(faces) not in (1, 2):
            raise BRepPartitionHistoryError(
                "partition_planar.edges must be one- or two-sided; got an invalid face incidence count"
            )
        signs = tuple(
            1 if signed > 0 else -1
            for face in faces
            for wire in model.topology.face_wires[face]
            for signed in wire
            if abs(signed) - 1 == edge
        )
        if len(signs) != len(faces) or (len(signs) == 2 and signs[0] == signs[1]):
            raise BRepPartitionHistoryError(
                "partition_planar.shared_edges must have opposite, unique face incidence; got an invalid shared edge"
            )
        adjacent = tuple(sorted((owners[face] for face in faces), key=rank.__getitem__))
        groups.setdefault(adjacent, []).append(model.edge_ids[edge])
    patches = []
    for adjacent, edges in sorted(groups.items()):
        if len(adjacent) == 1:
            name = f"boundary:{adjacent[0]}"
        elif adjacent[0] == adjacent[1]:
            name = f"internal:{adjacent[0]}"
        else:
            name = f"interface:{adjacent[0]}:{adjacent[1]}"
        patches.append(BRepPartitionPatch(name, tuple(edges), adjacent, 2))
    return regions, tuple(patches)


type _PointSignature = tuple[float, float, float]
type _EdgeSignature = tuple[_PointSignature, ...]
type _FaceSignature = tuple[tuple[_EdgeSignature, ...], ...]


def _roundtrip_order(
    original: BRepModel, restored: BRepModel, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Bind serialized records by their vertex coordinates and loop incidence.

    This maps a known publication round-trip, not a general geometry matching
    query. Ambiguous or missing incidence is a failed publication.
    """
    first_geometry, second_geometry = original.geometry, restored.geometry
    if first_geometry is None or second_geometry is None:
        raise BRepPartitionHistoryError(
            "partition_planar.geometry must retain native topology; got None"
        )
    if restored.topology.num_solids:
        raise BRepPartitionHistoryError(
            "partition_planar.topology must contain zero solids; got solid topology"
        )

    def signatures(
        model: BRepModel, /
    ) -> tuple[tuple[_FaceSignature, ...], tuple[_EdgeSignature, ...]]:
        geometry = model.geometry
        if geometry is None:
            raise BRepPartitionHistoryError(
                "partition_planar.restored.geometry must provide native B-Rep geometry; got None"
            )
        points = tuple(
            (float(point[0]), float(point[1]), float(point[2]))
            for point in np.asarray(geometry.vertex_points)
        )
        edges = tuple(
            tuple(sorted((points[start], points[end])))
            for start, end in geometry.edge_vertices
        )
        faces = tuple(
            tuple(
                sorted(
                    tuple(sorted(edges[geometry.coedge_edges[coedge]] for coedge in loop))
                    for loop in loops
                )
            )
            for loops in geometry.face_loops
        )
        return faces, edges

    first_faces, first_edges = signatures(original)
    second_faces, second_edges = signatures(restored)

    def order[T: Hashable](
        first: tuple[T, ...], second: tuple[T, ...], kind: str, /
    ) -> tuple[int, ...]:
        if (
            len(first) != len(second)
            or len(set(first)) != len(first)
            or len(set(second)) != len(second)
            or set(first) != set(second)
        ):
            raise BRepPartitionHistoryError(
                f"partition_planar.roundtrip_incidence must preserve unique {kind} incidence; got changed or ambiguous incidence"
            )
        return tuple(first.index(value) for value in second)

    return order(first_faces, second_faces, "face"), order(
        first_edges, second_edges, "edge"
    )


def _prepare_planar_arrangement(plan: PlanarPartitionPlan, /) -> _PlanarArrangement:
    source_revision, sources = _native_sources(plan)
    if (
        sum(sum(loop.shape[0] for loop in source.loops) for source in sources)
        > plan.policy.sewing.maximum_coedges
    ):
        raise BRepPartitionHistoryError(
            "partition_planar.source_coedges must fit maximum_coedges; got a source budget overflow"
        )
    arranged = arrange(tuple(source.loops for source in sources))
    if len(arranged) > plan.policy.maximum_cells:
        raise BRepPartitionHistoryError(
            "partition_planar.cells must fit maximum_cells; got an arrangement budget overflow"
        )
    retained = tuple(
        (cell, owner)
        for cell in arranged
        if (owner := _cell_owner(plan, sources, cell)) is not None
    )
    if not retained:
        raise ValueError(
            "partition_planar.cells must retain material; got no retained material region"
        )
    cells = tuple(cell for cell, _ in retained)
    owners = tuple(owner for _, owner in retained)
    if set(owners) != set(plan.policy.region_precedence):
        raise ValueError(
            "partition_planar.owners must retain every declared region; got a region without a face"
        )
    if (
        len(cells) > plan.policy.maximum_faces
        or sum(len(loop) for cell in cells for loop in cell.loops)
        > plan.policy.sewing.maximum_coedges
    ):
        raise BRepPartitionHistoryError(
            "partition_planar.output must fit its face/coedge budget; got a publication budget overflow"
        )
    return _PlanarArrangement(source_revision, sources, cells, owners)


class _PlanarRestoration(StrictModule):
    model: BRepModel
    cells: tuple[Cell, ...] = eqx.field(static=True)
    owners: tuple[str, ...] = eqx.field(static=True)
    atoms: tuple[Segment, ...] = eqx.field(static=True)


def _restore_planar_publication(
    plan: PlanarPartitionPlan,
    model: BRepModel,
    arranged: _PlanarArrangement,
    atoms: tuple[Segment, ...],
    staging: Path,
    target: Path,
    policy: BRepTessellationPolicy,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> _PlanarRestoration:
    cells, owners = arranged.cells, arranged.owners
    if target.suffix.lower() == ".phx":
        # Lazy: brep._planar -> interchange._cad_archive -> brep import cycle.
        import phydrax.interchange._cad_archive as cad_archive_owner

        cad_archive_owner.save_brep_archive(model, staging)
        restored = cad_archive_owner.load_brep_archive(staging, preparations=preparations)
    else:
        # Lazy: brep._planar -> interchange._cad_brep_text -> _cad -> brep import cycle.
        import phydrax.interchange._cad as cad_owner
        import phydrax.interchange._cad_brep_text as cad_brep_text_owner

        cad_brep_text_owner.write_brep_text(model, staging)
        restored = cad_brep_text_owner.read_brep_text(
            staging,
            cad_owner.CadImportPolicy(
                plan.coordinate_contract,
                ResourceLimits(64 * 1024 * 1024, 128, 1_000_000, 10_000_000, 0),
                tessellation=policy,
            ),
            trusted_root=target.parent,
            source_length_unit=plan.coordinate_contract.length_unit,
        ).model
        restored = restored.evolve(
            report=replace(restored.report, source_id=str(target)),
            budget=budget,
            preparations=preparations,
        )
    for face in range(restored.topology.num_faces):
        require_coplanar_face(restored, face, plan.embedding)
    face_order, edge_order = _roundtrip_order(model, restored)
    cells = tuple(cells[index] for index in face_order)
    owners = tuple(owners[index] for index in face_order)
    atoms = tuple(atoms[index] for index in edge_order)
    return _PlanarRestoration(restored, cells, owners, atoms)


def _assemble_planar_result(
    plan: PlanarPartitionPlan,
    arranged: _PlanarArrangement,
    publication: _PlanarRestoration,
    /,
) -> BRepPartitionResult:
    source_revision, sources = arranged.source_revision, arranged.sources
    restored, cells, owners, atoms = (
        publication.model,
        publication.cells,
        publication.owners,
        publication.atoms,
    )
    graph = _native_history(plan, source_revision, sources, restored, cells, atoms)
    regions, patches = _regions_and_patches(plan, restored, owners)
    source_ids = {value.occurrence_id for value in source_revision.occurrences}
    mapped_sources = {
        value.source_occurrence_id for value in graph.transaction.correspondences
    }
    report = BRepPartitionReport(
        plan.plan_id,
        restored.model_id,
        source_revision.revision_id,
        restored.source_revision,
        graph.transaction.coverage.certificate_id,
        0,
        len(sources),
        0,
        restored.topology.num_faces,
        len(source_ids - mapped_sources),
        0,
        sum(len(source.edges) for source in sources),
        restored.topology.num_edges,
        2,
    )
    result = BRepPartitionResult(
        restored, graph.target_revision, graph, regions, patches, report, 2
    )
    return result


@checked
def partition_planar(
    plan: PlanarPartitionPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepPartitionResult:
    """Partition straight-edge coplanar regions using rational incidence.

    Nonconvex polygons, disconnected components, holes, voids and precedence
    are resolved in one arrangement. Curved trims require certified curve
    intersection and subdivision rather than tessellation.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((plan,)) as budget:
            return partition_planar(
                plan,
                destination=destination,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                trim_samples_per_edge=trim_samples_per_edge,
                preparations=preparations,
                budget=budget,
            )

    if plan.policy.run_parallel:
        raise ValueError(
            "PlanarPartitionPlan.policy.run_parallel must be False; got True"
        )
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in (".brep", ".brp", ".phx"):
        raise ValueError(
            "partition_planar.destination must end in .brep, .brp, or .phx; got an unsupported suffix"
        )
    if target.exists() and not plan.policy.overwrite:
        raise FileExistsError(target)
    policy = BRepTessellationPolicy(
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    arranged = _prepare_planar_arrangement(plan)
    cells, owners = arranged.cells, arranged.owners
    builder, atoms = _stage_cells(
        cells, owners, plan.embedding, preparations=preparations, budget=budget
    )
    model = finish(
        builder,
        solid=False,
        coordinate_contract=plan.coordinate_contract,
        source_id=f"native-planar-partition:{plan.plan_id}",
        tessellation=policy,
        preparations=preparations,
        budget=budget,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.partition-", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    staging = Path(name)
    staging.unlink()
    try:
        publication = _restore_planar_publication(
            plan,
            model,
            arranged,
            atoms,
            staging,
            target,
            policy,
            preparations=preparations,
            budget=budget,
        )
        result = _assemble_planar_result(plan, arranged, publication)
        publish_staged(staging, target, plan.policy.overwrite)
        return result
    finally:
        staging.unlink(missing_ok=True)


__all__ = ["PlanarPartitionOperand", "PlanarPartitionPlan", "partition_planar"]
