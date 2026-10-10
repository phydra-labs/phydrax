#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import tempfile
from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
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
from ._boolean import _precedence_owner
from ._constructors import _Builder, _finish, BRepTessellationPolicy
from ._model import BRepEntityId, BRepModel
from ._partition import (
    _edge_occurrence_id,
    _publish_staged,
    _text,
    all_brep_faces,
    BRepPartitionHistoryError,
    BRepPartitionPatch,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionReport,
    BRepPartitionResult,
    BRepPartitionRole,
    cad_revision_from_brep_model,
)
from ._patches import LineCurve, PlanePatch
from ._planar_arrangement import (
    _area,
    _arrange,
    _Cell,
    _curve_segment,
    _face_loops,
    _on_segment,
    _point,
    _rational_loops,
    _require_coplanar_face,
    Segment,
)


@dataclass(frozen=True, slots=True)
class PlanarPartitionOperand:
    """One planar mesh region or exact root-face selection in a 2D partition."""

    operand_id: str
    source: PlanarMeshRegion | BRepModel
    role: BRepPartitionRole
    selection: CADSelectionSet | None = None
    target_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operand_id = _text(self.operand_id, "operand_id")
        role = BRepPartitionRole(self.role)
        source = self.source
        selection = self.selection
        if isinstance(source, PlanarMeshRegion):
            if selection is not None:
                raise ValueError("Planar mesh operands do not accept CAD selections.")
        elif isinstance(source, BRepModel):
            if source.topology.num_solids:
                raise ValueError(
                    "Persisted planar operands must contain root faces and zero solids."
                )
            revision = cad_revision_from_brep_model(source)
            if selection is None:
                selection = all_brep_faces(source)
            if not isinstance(selection, CADSelectionSet):
                raise TypeError("selection must be a CADSelectionSet or None.")
            selection.require_kind("face")
            if not selection.selectors:
                raise ValueError("A planar B-Rep operand must select at least one face.")
            if selection.revision_id != revision.revision_id:
                raise ValueError("Planar selection belongs to another B-Rep revision.")
            for selector in selection.selectors:
                occurrence = revision.occurrence(selector.occurrence_id)
                if occurrence.parent_occurrence_id is not None:
                    raise ValueError("Planar selections must contain only root faces.")
                if revision.select(selector.occurrence_id) != selector:
                    raise ValueError(
                        "Planar selection does not match the B-Rep occurrence inventory."
                    )
        else:
            raise TypeError("source must be a PlanarMeshRegion or persisted BRepModel.")
        target_region_ids = tuple(
            _text(value, "target_region_id") for value in self.target_region_ids
        )
        if len(set(target_region_ids)) != len(target_region_ids):
            raise ValueError("A planar void cannot repeat a target region.")
        if role is BRepPartitionRole.REGION and target_region_ids:
            raise ValueError("Planar region operands cannot declare void targets.")
        if role is BRepPartitionRole.VOID and not target_region_ids:
            raise ValueError("Planar void operands require at least one target region.")
        object.__setattr__(self, "operand_id", operand_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "target_region_ids", target_region_ids)


@dataclass(frozen=True, slots=True)
class PlanarPartitionPlan:
    """Closed exact 2D CAD partition recipe in one physical world embedding."""

    coordinate_contract: SpatialCoordinateContract
    embedding: PlanarEmbedding
    operands: tuple[PlanarPartitionOperand, ...]
    policy: BRepPartitionPolicy
    plan_id: str = field(init=False)
    topological_dimension: int = field(init=False, default=2)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if self.coordinate_contract.coordinate_system != "cartesian":
            raise ValueError("Planar embeddings require a Cartesian coordinate contract.")
        if not isinstance(self.embedding, PlanarEmbedding):
            raise TypeError("embedding must be a PlanarEmbedding.")
        operands = tuple(self.operands)
        if not operands or any(
            not isinstance(value, PlanarPartitionOperand) for value in operands
        ):
            raise TypeError("operands must contain at least one PlanarPartitionOperand.")
        if not isinstance(self.policy, BRepPartitionPolicy):
            raise TypeError("policy must be a BRepPartitionPolicy.")
        operand_ids = tuple(value.operand_id for value in operands)
        if len(set(operand_ids)) != len(operand_ids):
            raise ValueError("Planar partition operand IDs must be unique.")
        region_ids = {
            value.operand_id
            for value in operands
            if value.role is BRepPartitionRole.REGION
        }
        if set(self.policy.region_precedence) != region_ids:
            raise ValueError(
                "region_precedence must name every planar region exactly once."
            )
        for operand in operands:
            if (
                isinstance(operand.source, BRepModel)
                and operand.source.coordinate_contract.spatial_id
                != self.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Every persisted planar operand must use the plan coordinate contract."
                )
            if (
                operand.role is BRepPartitionRole.VOID
                and not set(operand.target_region_ids) <= region_ids
            ):
                raise ValueError("A planar void targets an unknown partition region.")
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
        selection = operand.selection
        if selection is None:
            raise RuntimeError("A planar B-Rep operand lost its exact face selection.")
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


def _mesh_loops(region: PlanarMeshRegion, /) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    edges = np.asarray(region.edges, dtype=np.int64)
    offsets = np.asarray(region.loop_offsets, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)):
        raise ValueError("Planar region vertices must be finite with shape (N, 2).")
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != edges.shape[0]
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise ValueError("Planar region loop topology is malformed.")
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
            raise ValueError("Planar region loops must be simple closed edge cycles.")
        points = vertices[vertex_ids]
        if len({tuple(value) for value in points}) != points.shape[0]:
            raise ValueError("Planar region loops cannot repeat a point.")
        area = _signed_area(points)
        if (loop_index == 0 and area <= 0.0) or (loop_index > 0 and area >= 0.0):
            raise ValueError(
                "The planar outer loop must be counter-clockwise and holes clockwise."
            )
        loops.append(points)
    _validate_loop_arrangement(tuple(loops))
    return tuple(loops)


def _signed_area(points: np.ndarray, /) -> float:
    return float(_area(tuple(_point(point) for point in points)))


def _validate_loop_arrangement(loops: tuple[np.ndarray, ...], /) -> None:
    _rational_loops(loops)


@dataclass(frozen=True, slots=True)
class _NativeSourceFace:
    operand_id: str
    occurrence: CADOccurrence
    loops: tuple[np.ndarray, ...]
    edges: tuple[tuple[CADOccurrence, Segment], ...]


def _native_sources(
    plan: PlanarPartitionPlan, /
) -> tuple[CADRevision, tuple[_NativeSourceFace, ...]]:
    revision_id = canonical_fingerprint(
        {"kind": "planar-brep-partition-source-revision", "plan_id": plan.plan_id}
    )
    occurrences: list[CADOccurrence] = []
    faces: list[_NativeSourceFace] = []
    for operand in sorted(plan.operands, key=lambda value: value.operand_id):
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
            items = ((0, _mesh_loops(source), None),)
        else:
            identity = source.source_revision
            selection = operand.selection
            if selection is None:
                raise RuntimeError("A native planar operand lost its face selection.")
            selected_ids = {selector.occurrence_id for selector in selection.selectors}
            items = tuple(
                (index, _face_loops(source, index, plan.embedding), source)
                for index in range(source.topology.num_faces)
                if f"face:{index}" in selected_ids
            )
        for face_index, loops, model in items:
            face_id = f"operand:{operand.operand_id}:face:{face_index}"
            occurrence = CADOccurrence(
                revision_id, face_id, f"{identity}:face:{face_index}", "face", (face_id,)
            )
            occurrences.append(occurrence)
            edge_rows = []
            if model is None:
                segments = tuple(
                    (index, _point(first), _point(second), 1)
                    for index, (first, second) in enumerate(
                        (
                            pair
                            for loop in loops
                            for pair in zip(loop, np.roll(loop, -1, axis=0), strict=True)
                        )
                    )
                )
            else:
                geometry = model.geometry
                if geometry is None:
                    raise RuntimeError("A native planar source lost its exact geometry.")
                segments = tuple(
                    (
                        edge,
                        _point(segment[0]),
                        _point(segment[1]),
                        next(
                            geometry.coedge_senses[coedge]
                            for loop in geometry.face_loops[face_index]
                            for coedge in loop
                            if geometry.coedge_edges[coedge] == edge
                        ),
                    )
                    for edge in model.topology.face_edges[face_index]
                    for segment in (_curve_segment(model, edge, plan.embedding),)
                )
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
    plan: PlanarPartitionPlan, sources: tuple[_NativeSourceFace, ...], cell: _Cell, /
) -> str | None:
    members = frozenset(sources[index].operand_id for index in cell.members)
    voids = tuple(
        (operand.operand_id, operand.target_region_ids)
        for operand in plan.operands
        if operand.role is BRepPartitionRole.VOID
    )
    return _precedence_owner(plan.policy.region_precedence, voids, members)


def _stage_cells(
    cells: tuple[_Cell, ...], owners: tuple[str, ...], embedding: PlanarEmbedding, /
) -> tuple[_Builder, tuple[Segment, ...]]:
    builder = _Builder()
    # Arrangement atoms own exact intersection incidence; this is not
    # approximate coordinate welding or inferred scientific CAD identity.
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
            "Distinct exact arrangement intersections collapse at native carrier precision."
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
    cells: tuple[_Cell, ...],
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
                if all(_on_segment(point, segment) for point in atom):
                    for face in model.topology.edge_faces[edge]:
                        pairs.add(
                            (occurrence.occurrence_id, _edge_occurrence_id(face, edge))
                        )
    source_ids = {value.occurrence_id for value in source_revision.occurrences}
    target_ids = {value.occurrence_id for value in target_revision.occurrences}
    if {target for _, target in pairs} != target_ids:
        raise BRepPartitionHistoryError(
            "Native arrangement provenance does not exhaust the final face-edge inventory."
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
                "Native planar edges must be exactly one- or two-sided."
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
                "Native planar shared edges require opposite, unique face incidence."
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
    """Bind controlled serialization records by exact vertex/loop incidence.

    This maps a known publication round-trip, not a general geometry matching
    query. Ambiguous or missing incidence is a failed publication.
    """
    first_geometry, second_geometry = original.geometry, restored.geometry
    if first_geometry is None or second_geometry is None:
        raise BRepPartitionHistoryError("Native planar publication lost exact topology.")
    if restored.topology.num_solids:
        raise BRepPartitionHistoryError("Planar publication acquired solid topology.")

    def signatures(
        model: BRepModel, /
    ) -> tuple[tuple[_FaceSignature, ...], tuple[_EdgeSignature, ...]]:
        geometry = model.geometry
        if geometry is None:
            raise BRepPartitionHistoryError(
                "Native planar round-trip has no exact geometry."
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
                f"Native planar round-trip changed or ambiguously identified {kind} incidence."
            )
        return tuple(first.index(value) for value in second)

    return order(first_faces, second_faces, "face"), order(
        first_edges, second_edges, "edge"
    )


def partition_planar(
    plan: PlanarPartitionPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepPartitionResult:
    """Partition straight-edge coplanar regions with exact rational incidence.

    Nonconvex polygons, disconnected components, holes, voids and precedence
    are resolved in one arrangement. Curved trims require native exact curve
    intersection and subdivision; they are never replaced by tessellation.
    """
    from ..._external_resource import ResourceLimits
    from ...interchange._cad import CadImportPolicy
    from ...interchange._cad_archive import load_brep_archive, save_brep_archive
    from ...interchange._cad_brep_text import read_brep_text, write_brep_text

    if not isinstance(plan, PlanarPartitionPlan):
        raise TypeError("plan must be a PlanarPartitionPlan.")
    if plan.policy.run_parallel:
        raise ValueError("Native planar partition does not yet admit parallel execution.")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in (".brep", ".brp", ".phx"):
        raise ValueError("Native planar publication requires .brep, .brp, or .phx.")
    if target.exists() and not plan.policy.overwrite:
        raise FileExistsError(target)
    policy = BRepTessellationPolicy(
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    source_revision, sources = _native_sources(plan)
    if (
        sum(sum(loop.shape[0] for loop in source.loops) for source in sources)
        > plan.policy.sewing.maximum_coedges
    ):
        raise BRepPartitionHistoryError("Native planar source exceeds maximum_coedges.")
    arranged = _arrange(tuple(source.loops for source in sources))
    if len(arranged) > plan.policy.maximum_cells:
        raise BRepPartitionHistoryError(
            "Native planar arrangement exceeds maximum_cells."
        )
    retained = tuple(
        (cell, owner)
        for cell in arranged
        if (owner := _cell_owner(plan, sources, cell)) is not None
    )
    if not retained:
        raise ValueError("Native planar partition has no retained material region.")
    cells = tuple(cell for cell, _ in retained)
    owners = tuple(owner for _, owner in retained)
    if set(owners) != set(plan.policy.region_precedence):
        raise ValueError("Every declared planar region must retain a face.")
    if (
        len(cells) > plan.policy.maximum_faces
        or sum(len(loop) for cell in cells for loop in cell.loops)
        > plan.policy.sewing.maximum_coedges
    ):
        raise BRepPartitionHistoryError(
            "Native planar output exceeds its face/coedge budget."
        )
    builder, atoms = _stage_cells(cells, owners, plan.embedding)
    model = _finish(
        builder,
        solid=False,
        coordinate_contract=plan.coordinate_contract,
        source_id=f"native-planar-partition:{plan.plan_id}",
        tessellation=policy,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.partition-", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    staging = Path(name)
    staging.unlink()
    try:
        if target.suffix.lower() == ".phx":
            save_brep_archive(model, staging)
            restored = load_brep_archive(staging)
        else:
            write_brep_text(model, staging)
            restored = read_brep_text(
                staging,
                CadImportPolicy(
                    plan.coordinate_contract,
                    ResourceLimits(64 * 1024 * 1024, 128, 1_000_000, 10_000_000, 0),
                    tessellation=policy,
                ),
                trusted_root=target.parent,
                source_length_unit=plan.coordinate_contract.length_unit,
            ).model
            restored = BRepModel(
                patches=restored.patches,
                parameter_bounds=restored.parameter_bounds,
                orientation=restored.orientation,
                trim_domains=restored.trim_domains,
                topology=restored.topology,
                coordinate_contract=restored.coordinate_contract,
                mesh_vertices=restored.mesh_vertices,
                mesh_faces=restored.mesh_faces,
                triangle_face_ids=restored.triangle_face_ids,
                triangle_parameters=restored.triangle_parameters,
                tessellation_deviation_bounds=restored.tessellation_deviation_bounds,
                tessellation_normal_bounds=restored.tessellation_normal_bounds,
                mesh_vertex_source_dimensions=restored.mesh_vertex_source_dimensions,
                mesh_vertex_source_indices=restored.mesh_vertex_source_indices,
                mesh_vertex_parameters=restored.mesh_vertex_parameters,
                mesh_chart_restriction_vertices=restored.mesh_chart_restriction_vertices,
                mesh_chart_restriction_edges=restored.mesh_chart_restriction_edges,
                mesh_chart_restriction_endpoint_parameters=(
                    restored.mesh_chart_restriction_endpoint_parameters
                ),
                mesh_chart_restriction_parameters=(
                    restored.mesh_chart_restriction_parameters
                ),
                coedge_deviation_bounds=restored.coedge_deviation_bounds,
                triangle_occurrence_ids=restored.triangle_occurrence_ids,
                vertex_occurrence_ids=restored.vertex_occurrence_ids,
                physical_tags=restored.physical_tags,
                report=replace(restored.report, source_id=str(target)),
                geometry=restored.geometry,
            )
        for face in range(restored.topology.num_faces):
            _require_coplanar_face(restored, face, plan.embedding)
        face_order, edge_order = _roundtrip_order(model, restored)
        cells = tuple(cells[index] for index in face_order)
        owners = tuple(owners[index] for index in face_order)
        atoms = tuple(atoms[index] for index in edge_order)
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
        _publish_staged(staging, target, plan.policy.overwrite)
        return result
    finally:
        staging.unlink(missing_ok=True)


__all__ = ["PlanarPartitionOperand", "PlanarPartitionPlan", "partition_planar"]
