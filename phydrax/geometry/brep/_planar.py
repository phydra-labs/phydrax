#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from OCP.BOPAlgo import BOPAlgo_CellsBuilder  # ty: ignore[unresolved-import]
from OCP.BRep import BRep_Tool  # ty: ignore[unresolved-import]
from OCP.BRepAdaptor import BRepAdaptor_Surface  # ty: ignore[unresolved-import]
from OCP.BRepBuilderAPI import (  # ty: ignore[unresolved-import]
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
)
from OCP.BRepCheck import BRepCheck_Analyzer  # ty: ignore[unresolved-import]
from OCP.BRepTools import BRepTools_WireExplorer  # ty: ignore[unresolved-import]
from OCP.GeomAbs import GeomAbs_Plane  # ty: ignore[unresolved-import]
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt  # ty: ignore[unresolved-import]
from OCP.TopAbs import (  # ty: ignore[unresolved-import]
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp_Explorer  # ty: ignore[unresolved-import]
from OCP.TopoDS import TopoDS, TopoDS_Shape  # ty: ignore[unresolved-import]

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
from ..simplicial import PlanarMeshRegion
from ._model import BRepEntityId, BRepModel
from ._occt import import_brep, persist_occt_shape, read_occt_shape
from ._partition import (
    _capture_history,
    _edge_occurrence_id,
    _exact_roundtrip_map,
    _explore_unique,
    _load_model,
    _publish_staged,
    _shape_index,
    _shape_list,
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


_FRAME_TOLERANCE = 128.0 * np.finfo(float).eps


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite three-vector.")
    return tuple(float(component) for component in array)


@dataclass(frozen=True, slots=True, init=False)
class PlanarEmbedding:
    """Exact affine embedding of planar coordinates in one world frame."""

    origin: tuple[float, float, float]
    x_axis: tuple[float, float, float]
    y_axis: tuple[float, float, float]
    normal: tuple[float, float, float]
    embedding_id: str

    def __init__(
        self,
        origin: Sequence[float],
        x_axis: Sequence[float],
        y_axis: Sequence[float],
        normal: Sequence[float],
        /,
    ):
        origin_ = _vector3(origin, "origin")
        x_axis_ = _vector3(x_axis, "x_axis")
        y_axis_ = _vector3(y_axis, "y_axis")
        normal_ = _vector3(normal, "normal")
        frame = np.stack((x_axis_, y_axis_, normal_))
        if not np.allclose(
            frame @ frame.T,
            np.eye(3),
            rtol=0.0,
            atol=_FRAME_TOLERANCE,
        ):
            raise ValueError("Planar embedding axes must be orthonormal unit vectors.")
        if not np.allclose(
            np.cross(frame[0], frame[1]),
            frame[2],
            rtol=0.0,
            atol=_FRAME_TOLERANCE,
        ):
            raise ValueError("Planar embedding axes must form a right-handed frame.")
        embedding_id = canonical_fingerprint(
            {
                "kind": "planar-embedding",
                "origin": origin_,
                "x_axis": x_axis_,
                "y_axis": y_axis_,
                "normal": normal_,
            }
        )
        object.__setattr__(self, "origin", origin_)
        object.__setattr__(self, "x_axis", x_axis_)
        object.__setattr__(self, "y_axis", y_axis_)
        object.__setattr__(self, "normal", normal_)
        object.__setattr__(self, "embedding_id", embedding_id)

    def to_world(self, coordinates: Any, /) -> np.ndarray:
        """Map coordinates with trailing dimension two into the world frame."""

        planar = np.asarray(coordinates, dtype=float)
        if planar.ndim == 0 or planar.shape[-1] != 2:
            raise ValueError("Planar coordinates must have trailing dimension two.")
        if not np.all(np.isfinite(planar)):
            raise ValueError("Planar coordinates must be finite.")
        basis = np.stack((self.x_axis, self.y_axis), axis=0)
        return np.asarray(self.origin) + planar @ basis

    def to_planar(self, points: Any, /) -> np.ndarray:
        """Return the exact frame coordinates of world points on the plane."""

        world = np.asarray(points, dtype=float)
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError("World points must have trailing dimension three.")
        if not np.all(np.isfinite(world)):
            raise ValueError("World points must be finite.")
        basis = np.stack((self.x_axis, self.y_axis), axis=1)
        return (world - np.asarray(self.origin)) @ basis

    def plane_residual(self, points: Any, /) -> np.ndarray:
        """Return the signed world-frame residual along the plane normal."""

        world = np.asarray(points, dtype=float)
        if world.ndim == 0 or world.shape[-1] != 3:
            raise ValueError("World points must have trailing dimension three.")
        if not np.all(np.isfinite(world)):
            raise ValueError("World points must be finite.")
        return (world - np.asarray(self.origin)) @ np.asarray(self.normal)


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
            }
        )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "plan_id", plan_id)


@dataclass(frozen=True, slots=True)
class _SourceEdge:
    occurrence: CADOccurrence
    shape: Any


@dataclass(frozen=True, slots=True)
class _SourceFace:
    operand_id: str
    occurrence: CADOccurrence
    shape: Any
    edges: tuple[_SourceEdge, ...]


@dataclass(frozen=True, slots=True)
class _PlanarRoundtrip:
    faces: tuple[Any, ...]
    edges: tuple[Any, ...]
    face_map: tuple[int, ...]
    edge_map: tuple[int, ...]


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
    vertices = np.asarray(region.vertices, dtype=float)
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
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    first_edge = second - first
    second_edge = third - first
    return float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])


def _on_segment(first: np.ndarray, point: np.ndarray, second: np.ndarray) -> bool:
    return bool(
        _orientation(first, second, point) == 0.0
        and min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
        and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if (
        orientations[0] * orientations[1] < 0.0
        and orientations[2] * orientations[3] < 0.0
    ):
        return True
    return bool(
        (orientations[0] == 0.0 and _on_segment(first_start, second_start, first_end))
        or (orientations[1] == 0.0 and _on_segment(first_start, second_end, first_end))
        or (orientations[2] == 0.0 and _on_segment(second_start, first_start, second_end))
        or (orientations[3] == 0.0 and _on_segment(second_start, first_end, second_end))
    )


def _point_in_loop(point: np.ndarray, loop: np.ndarray) -> bool:
    inside = False
    for first, second in zip(loop, np.roll(loop, -1, axis=0), strict=True):
        if _on_segment(first, point, second):
            return False
        if (first[1] > point[1]) != (second[1] > point[1]):
            crossing = first[0] + (point[1] - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if crossing > point[0]:
                inside = not inside
    return inside


def _validate_loop_arrangement(loops: tuple[np.ndarray, ...], /) -> None:
    segments = tuple(
        tuple(zip(loop, np.roll(loop, -1, axis=0), strict=True)) for loop in loops
    )
    for loop_segments in segments:
        count = len(loop_segments)
        for first_index, first in enumerate(loop_segments):
            for second_index in range(first_index + 1, count):
                if second_index == first_index + 1 or (
                    first_index == 0 and second_index == count - 1
                ):
                    continue
                if _segments_intersect(*first, *loop_segments[second_index]):
                    raise ValueError("A planar region loop cannot self-intersect.")
    for first_index, first_segments in enumerate(segments):
        for second_segments in segments[first_index + 1 :]:
            if any(
                _segments_intersect(*first, *second)
                for first in first_segments
                for second in second_segments
            ):
                raise ValueError("Planar outer and hole loops cannot touch or intersect.")
    outer = loops[0]
    for hole in loops[1:]:
        if not _point_in_loop(hole[0], outer):
            raise ValueError("Every planar hole must lie strictly inside the outer loop.")
    for first_index, first in enumerate(loops[1:]):
        for second in loops[first_index + 2 :]:
            if _point_in_loop(first[0], second) or _point_in_loop(second[0], first):
                raise ValueError("Planar holes cannot overlap or contain one another.")


def _wire(points: np.ndarray, /) -> Any:
    builder = BRepBuilderAPI_MakePolygon()
    for point in points:
        builder.Add(gp_Pnt(*(float(value) for value in point)))
    builder.Close()
    if not builder.IsDone():
        raise ValueError("OCCT could not construct an exact planar loop.")
    return builder.Wire()


def _roundoff_tolerance(embedding: PlanarEmbedding, points: np.ndarray) -> float:
    scale = max(
        1.0,
        float(np.max(np.abs(np.asarray(embedding.origin)))),
        float(np.max(np.abs(points))) if points.size else 0.0,
    )
    return 512.0 * np.finfo(float).eps * scale


def _face_vertices(face: Any, /) -> np.ndarray:
    points: list[tuple[float, float, float]] = []
    for vertex in _explore_unique(face, TopAbs_VERTEX, TopoDS.Vertex_s):
        point = BRep_Tool.Pnt_s(vertex)
        points.append((point.X(), point.Y(), point.Z()))
    return np.asarray(points, dtype=float)


def _require_coplanar_face(face: Any, embedding: PlanarEmbedding, /) -> None:
    if face.IsNull() or not BRepCheck_Analyzer(face).IsValid():
        raise ValueError("A planar partition source face must be a valid OCCT face.")
    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_Plane:
        raise ValueError("Persisted planar partition faces must be exact planes.")
    plane = adaptor.Plane()
    direction = plane.Position().Direction()
    normal = np.asarray((direction.X(), direction.Y(), direction.Z()), dtype=float)
    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal
    vertices = _face_vertices(face)
    tolerance = _roundoff_tolerance(embedding, vertices)
    if not np.allclose(
        normal,
        np.asarray(embedding.normal),
        rtol=0.0,
        atol=512.0 * np.finfo(float).eps,
    ):
        raise ValueError(
            "Every planar source face must have the embedding's positive orientation."
        )
    origin = gp_Pnt(*(float(value) for value in embedding.origin))
    if float(plane.Distance(origin)) > tolerance:
        raise ValueError("A planar source face lies on another plane.")
    if vertices.size and np.max(np.abs(embedding.plane_residual(vertices))) > tolerance:
        raise ValueError("A planar source face is not coplanar with its embedding.")


def _face_from_mesh(
    region: PlanarMeshRegion,
    embedding: PlanarEmbedding,
    /,
) -> Any:
    loops = _mesh_loops(region)
    world_loops = tuple(embedding.to_world(loop) for loop in loops)
    axis = gp_Ax3(
        gp_Pnt(*(float(value) for value in embedding.origin)),
        gp_Dir(*(float(value) for value in embedding.normal)),
        gp_Dir(*(float(value) for value in embedding.x_axis)),
    )
    face_builder = BRepBuilderAPI_MakeFace(
        gp_Pln(axis),
        _wire(world_loops[0]),
        True,
    )
    for hole in world_loops[1:]:
        face_builder.Add(_wire(hole))
    if not face_builder.IsDone():
        raise ValueError("OCCT could not construct the exact planar region face.")
    face = face_builder.Face()
    _require_coplanar_face(face, embedding)
    return face


def _face_edge_occurrences(
    face: Any,
    global_edges: tuple[Any, ...],
    /,
) -> tuple[tuple[tuple[int, ...], ...], dict[int, Any]]:
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    wires: list[tuple[int, ...]] = []
    oriented_edges: dict[int, Any] = {}
    while wire_explorer.More():
        wire = TopoDS.Wire_s(wire_explorer.Current())
        edge_explorer = BRepTools_WireExplorer(wire, face)
        signed_indices: list[int] = []
        while edge_explorer.More():
            edge = TopoDS.Edge_s(edge_explorer.Current())
            edge_index = _shape_index(global_edges, edge)
            if edge_index in oriented_edges:
                raise ValueError(
                    "A planar face cannot repeat an edge in its boundary loops."
                )
            oriented_edges[edge_index] = edge
            sign = -1 if edge.Orientation() == TopAbs_REVERSED else 1
            signed_indices.append(sign * (edge_index + 1))
            edge_explorer.Next()
        if len(signed_indices) < 1:
            raise ValueError("A planar face cannot contain an empty boundary wire.")
        wires.append(tuple(signed_indices))
        wire_explorer.Next()
    if not wires:
        raise ValueError("A planar face must contain at least one boundary wire.")
    return tuple(wires), oriented_edges


def _mesh_identity(
    region: PlanarMeshRegion,
    plan: PlanarPartitionPlan,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "embedded-planar-mesh-region",
            "feature_id": region.feature_id,
            "vertices": np.asarray(region.vertices),
            "edges": np.asarray(region.edges),
            "loop_offsets": np.asarray(region.loop_offsets),
            "coordinate_contract": plan.coordinate_contract.spatial_id,
            "embedding": plan.embedding.embedding_id,
        }
    )


def _composite_sources(
    plan: PlanarPartitionPlan,
    loaded: dict[tuple[str, str], Any],
    /,
) -> tuple[CADRevision, tuple[_SourceFace, ...], dict[str, tuple[Any, ...]]]:
    revision_id = canonical_fingerprint(
        {
            "kind": "planar-brep-partition-source-revision",
            "plan_id": plan.plan_id,
        }
    )
    occurrences: list[CADOccurrence] = []
    source_faces: list[_SourceFace] = []
    operand_shapes: dict[str, tuple[Any, ...]] = {}
    for operand in sorted(plan.operands, key=lambda value: value.operand_id):
        selected: list[Any] = []
        source = operand.source
        if isinstance(source, PlanarMeshRegion):
            source_identity: str | None = _mesh_identity(source, plan)
            face_items = ((0, _face_from_mesh(source, plan.embedding), None),)
        else:
            source_identity = None
            item = loaded[(source.source_id, source.source_digest)]
            selection = operand.selection
            if selection is None:
                raise RuntimeError("A planar B-Rep operand lost its face selection.")
            face_items_list: list[tuple[int, Any, Any]] = []
            for selector in selection.selectors:
                source_face_index = next(
                    index
                    for index in range(len(item.faces))
                    if item.revision.select(f"face:{index}") == selector
                )
                face_items_list.append(
                    (source_face_index, item.faces[source_face_index], item)
                )
            face_items = tuple(face_items_list)
        for source_face_index, face, item in face_items:
            _require_coplanar_face(face, plan.embedding)
            global_edges = (
                _explore_unique(face, TopAbs_EDGE, TopoDS.Edge_s)
                if item is None
                else item.edges
            )
            wires, oriented_edges = _face_edge_occurrences(face, global_edges)
            face_occurrence_id = f"operand:{operand.operand_id}:face:{source_face_index}"
            if item is None:
                if source_identity is None:
                    raise RuntimeError("A planar mesh source lost its exact identity.")
                face_entity_id = f"{source_identity}:face:0"
            else:
                original_face = item.revision.select(f"face:{source_face_index}")
                face_entity_id = original_face.entity_id
            face_occurrence = CADOccurrence(
                revision_id,
                face_occurrence_id,
                face_entity_id,
                "face",
                (face_occurrence_id,),
            )
            occurrences.append(face_occurrence)
            source_edges: list[_SourceEdge] = []
            expected_edge_indices = (
                tuple(range(len(global_edges)))
                if item is None
                else item.model.topology.face_edges[source_face_index]
            )
            if set(expected_edge_indices) != set(oriented_edges):
                raise ValueError(
                    "Persisted planar face-edge incidence disagrees with its model."
                )
            for edge_index in expected_edge_indices:
                edge_occurrence_id = f"{face_occurrence_id}/edge:{edge_index}"
                if item is None:
                    entity_id = f"{source_identity}:edge:{edge_index}"
                    orientation = next(
                        1 if value > 0 else -1
                        for wire in wires
                        for value in wire
                        if abs(value) - 1 == edge_index
                    )
                else:
                    original_edge = item.revision.occurrence(
                        _edge_occurrence_id(source_face_index, edge_index)
                    )
                    entity_id = original_edge.entity_id
                    orientation = original_edge.orientation
                edge_occurrence = CADOccurrence(
                    revision_id,
                    edge_occurrence_id,
                    entity_id,
                    "edge",
                    (face_occurrence_id, edge_occurrence_id),
                    face_occurrence_id,
                    orientation,
                )
                occurrences.append(edge_occurrence)
                source_edges.append(
                    _SourceEdge(edge_occurrence, oriented_edges[edge_index])
                )
            selected.append(face)
            source_faces.append(
                _SourceFace(
                    operand.operand_id,
                    face_occurrence,
                    face,
                    tuple(source_edges),
                )
            )
        operand_shapes[operand.operand_id] = tuple(selected)
    if len({value.occurrence_id for value in occurrences}) != len(occurrences):
        raise ValueError("Planar operand IDs produce ambiguous CAD occurrences.")
    revision = CADRevision(
        revision_id,
        f"planar-brep-partition-input:{plan.plan_id}",
        tuple(occurrences),
        plan.plan_id,
    )
    return revision, tuple(source_faces), operand_shapes


def _members(
    builder: BOPAlgo_CellsBuilder,
    shape: Any,
    atoms: tuple[Any, ...],
    /,
) -> frozenset[int]:
    builder.RemoveAllFromResult()
    builder.AddToResult(_shape_list((shape,)), _shape_list(()))
    selected = _explore_unique(builder.Shape(), TopAbs_FACE, TopoDS.Face_s)
    return frozenset(_shape_index(atoms, value) for value in selected)


def _classify_atoms(
    plan: PlanarPartitionPlan,
    operand_members: dict[str, frozenset[int]],
    atom_count: int,
    /,
) -> tuple[str | None, ...]:
    voids = tuple(
        value for value in plan.operands if value.role is BRepPartitionRole.VOID
    )
    owners: list[str | None] = []
    for atom_index in range(atom_count):
        owner = next(
            (
                region_id
                for region_id in plan.policy.region_precedence
                if atom_index in operand_members[region_id]
            ),
            None,
        )
        if owner is not None and any(
            owner in void.target_region_ids
            and atom_index in operand_members[void.operand_id]
            for void in voids
        ):
            owner = None
        owners.append(owner)
    retained = {value for value in owners if value is not None}
    if not retained:
        raise ValueError("Planar partition policy removes every result face.")
    if retained != set(plan.policy.region_precedence):
        raise ValueError("Every declared planar region must retain a face.")
    return tuple(owners)


def _build_final_shape(
    plan: PlanarPartitionPlan,
    builder: BOPAlgo_CellsBuilder,
    operand_shapes: dict[str, tuple[Any, ...]],
    /,
) -> TopoDS_Shape:
    voids = tuple(
        value for value in plan.operands if value.role is BRepPartitionRole.VOID
    )
    builder.RemoveAllFromResult()
    for rank, region_id in enumerate(plan.policy.region_precedence):
        higher = plan.policy.region_precedence[:rank]
        avoid = tuple(
            shape for higher_id in higher for shape in operand_shapes[higher_id]
        ) + tuple(
            shape
            for void in voids
            if region_id in void.target_region_ids
            for shape in operand_shapes[void.operand_id]
        )
        for shape in operand_shapes[region_id]:
            builder.AddToResult(
                _shape_list((shape,)),
                _shape_list(avoid),
                rank + 1,
            )
    result = builder.Shape()
    if result.IsNull():
        raise ValueError("OCCT produced an empty planar partition result.")
    return result


def _final_face_owners(
    final_faces: tuple[Any, ...],
    atoms: tuple[Any, ...],
    atom_owners: tuple[str | None, ...],
    /,
) -> tuple[str, ...]:
    retained = {index for index, owner in enumerate(atom_owners) if owner is not None}
    used_atoms: set[int] = set()
    owners: list[str] = []
    for face in final_faces:
        candidates = tuple(index for index, atom in enumerate(atoms) if face.IsSame(atom))
        if len(candidates) != 1 or atom_owners[candidates[0]] is None:
            raise BRepPartitionHistoryError(
                "Final planar ownership lacks unique exact Boolean-cell incidence."
            )
        atom_index = candidates[0]
        owner = atom_owners[atom_index]
        if owner is None:
            raise BRepPartitionHistoryError(
                "A discarded planar Boolean cell entered the final shape."
            )
        if atom_index in used_atoms:
            raise BRepPartitionHistoryError(
                "A retained planar Boolean cell occurs more than once."
            )
        used_atoms.add(atom_index)
        owners.append(owner)
    if used_atoms != retained:
        raise BRepPartitionHistoryError(
            "Final planar ownership does not exhaust retained exact cells."
        )
    return tuple(owners)


def _verify_roundtrip(
    original_shape: Any,
    reopened_shape: Any,
    model: BRepModel,
    embedding: PlanarEmbedding,
    /,
) -> _PlanarRoundtrip:
    if _explore_unique(original_shape, TopAbs_SOLID, TopoDS.Solid_s) or _explore_unique(
        reopened_shape, TopAbs_SOLID, TopoDS.Solid_s
    ):
        raise BRepPartitionHistoryError(
            "A persisted planar partition cannot acquire solid topology."
        )
    original_faces = _explore_unique(original_shape, TopAbs_FACE, TopoDS.Face_s)
    original_edges = _explore_unique(original_shape, TopAbs_EDGE, TopoDS.Edge_s)
    reopened_faces = _explore_unique(reopened_shape, TopAbs_FACE, TopoDS.Face_s)
    reopened_edges = _explore_unique(reopened_shape, TopAbs_EDGE, TopoDS.Edge_s)
    face_map = _exact_roundtrip_map(original_faces, reopened_faces, "face")
    edge_map = _exact_roundtrip_map(original_edges, reopened_edges, "edge")
    if (
        model.topology.num_solids != 0
        or len(reopened_faces) != model.topology.num_faces
        or len(reopened_edges) != model.topology.num_edges
    ):
        raise BRepPartitionHistoryError(
            "Reopened planar model does not match its native face-edge inventory."
        )
    for original_face_index, face in enumerate(original_faces):
        wires, _ = _face_edge_occurrences(face, original_edges)
        mapped_wires = tuple(
            tuple(
                (1 if signed_index > 0 else -1) * (edge_map[abs(signed_index) - 1] + 1)
                for signed_index in wire
            )
            for wire in wires
        )
        if mapped_wires != model.topology.face_wires[face_map[original_face_index]]:
            raise BRepPartitionHistoryError(
                "Reopened planar partition changed exact face-edge incidence."
            )
    for face in reopened_faces:
        _require_coplanar_face(face, embedding)
    _certify_planar_topology(model)
    return _PlanarRoundtrip(original_faces, original_edges, face_map, edge_map)


def _certify_planar_topology(model: BRepModel, /) -> None:
    if model.topology.num_solids:
        raise BRepPartitionHistoryError(
            "A planar partition model must contain zero solids."
        )
    if not model.topology.num_faces or not model.topology.num_edges:
        raise BRepPartitionHistoryError(
            "A planar partition requires nonempty face and edge inventories."
        )
    for edge_index, face_indices in enumerate(model.topology.edge_faces):
        if len(face_indices) not in (1, 2):
            raise BRepPartitionHistoryError(
                "A planar edge is nonmanifold or absent from every face."
            )
        signs: list[int] = []
        for face_index in face_indices:
            matches = tuple(
                signed_edge
                for wire in model.topology.face_wires[face_index]
                for signed_edge in wire
                if abs(signed_edge) - 1 == edge_index
            )
            if len(matches) != 1:
                raise BRepPartitionHistoryError(
                    "Planar face-edge incidence is repeated or unresolved."
                )
            signs.append(1 if matches[0] > 0 else -1)
        if len(signs) == 2 and signs[0] == signs[1]:
            raise BRepPartitionHistoryError(
                "Shared planar edges require opposite face incidence."
            )


def _identity_association_graph(
    plan: PlanarPartitionPlan,
    source_revision: CADRevision,
    source_faces: tuple[_SourceFace, ...],
    target_model: BRepModel,
    roundtrip: _PlanarRoundtrip,
    /,
) -> tuple[AssociationGraph, str]:
    """Certify a one-face no-op partition through exact persisted topology."""

    if len(source_faces) != 1 or len(roundtrip.faces) != 1:
        raise BRepPartitionHistoryError(
            "Identity planar partition requires exactly one source and target face."
        )
    target_revision = cad_revision_from_brep_model(target_model)
    source_face = source_faces[0]
    target_face_index = roundtrip.face_map[0]
    target_face_id = f"face:{target_face_index}"
    correspondences = [
        OccurrenceCorrespondence(
            source_face.occurrence.occurrence_id,
            target_face_id,
            canonical_fingerprint(
                {
                    "kind": "exact-planar-identity-face",
                    "plan_id": plan.plan_id,
                    "source": source_face.occurrence.occurrence_id,
                    "target": target_face_id,
                }
            ),
        )
    ]
    for source_edge in source_face.edges:
        original_edge_index = _shape_index(roundtrip.edges, source_edge.shape)
        target_edge_index = roundtrip.edge_map[original_edge_index]
        target_faces = target_model.topology.edge_faces[target_edge_index]
        if target_faces != (target_face_index,):
            raise BRepPartitionHistoryError(
                "Identity planar partition changed exact face-edge incidence."
            )
        target_edge_id = _edge_occurrence_id(target_face_index, target_edge_index)
        correspondences.append(
            OccurrenceCorrespondence(
                source_edge.occurrence.occurrence_id,
                target_edge_id,
                canonical_fingerprint(
                    {
                        "kind": "exact-planar-identity-edge",
                        "plan_id": plan.plan_id,
                        "source": source_edge.occurrence.occurrence_id,
                        "target": target_edge_id,
                    }
                ),
            )
        )
    target_ids = {occurrence.occurrence_id for occurrence in target_revision.occurrences}
    correspondence_targets = {
        correspondence.target_occurrence_id for correspondence in correspondences
    }
    if correspondence_targets != target_ids:
        raise BRepPartitionHistoryError(
            "Identity planar partition does not exhaust its target topology."
        )
    certificate_id = canonical_fingerprint(
        {
            "kind": "exact-planar-identity-persistence",
            "plan_id": plan.plan_id,
            "source_revision": source_revision.revision_id,
            "target_revision": target_revision.revision_id,
            "correspondences": sorted(
                (
                    correspondence.source_occurrence_id,
                    correspondence.target_occurrence_id,
                    correspondence.evidence_id,
                )
                for correspondence in correspondences
            ),
        }
    )
    transaction = OccurrenceCorrespondenceTransaction(
        canonical_fingerprint(
            {
                "kind": "planar-identity-correspondence-transaction",
                "plan_id": plan.plan_id,
                "certificate": certificate_id,
            }
        ),
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(correspondences),
        frozenset(),
        frozenset(),
        AssociationCoverageEvidence(
            True,
            True,
            certificate_id,
            "OCP.exact-persistence-identity",
        ),
    )
    return AssociationGraph(source_revision, target_revision, transaction), certificate_id


def _association_graph(
    plan: PlanarPartitionPlan,
    source_revision: CADRevision,
    source_faces: tuple[_SourceFace, ...],
    target_model: BRepModel,
    roundtrip: _PlanarRoundtrip,
    builder: BOPAlgo_CellsBuilder,
    /,
) -> tuple[AssociationGraph, str]:
    if not builder.HasHistory():
        raise BRepPartitionHistoryError(
            "OCCT did not provide the required live Boolean history."
        )
    target_revision = cad_revision_from_brep_model(target_model)
    correspondences: list[OccurrenceCorrespondence] = []
    evidence_rows: list[tuple[object, ...]] = []
    for source_face in source_faces:
        face_history = _capture_history(
            builder,
            source_face.shape,
            roundtrip.faces,
            TopAbs_FACE,
            TopoDS.Face_s,
        )
        mapped_faces = tuple(
            roundtrip.face_map[index] for index in face_history.target_indices
        )
        evidence_rows.append(
            (
                source_face.occurrence.occurrence_id,
                mapped_faces,
                face_history.modified_count,
                face_history.generated_count,
                face_history.deleted,
            )
        )
        for target_face_index in mapped_faces:
            target_id = f"face:{target_face_index}"
            correspondences.append(
                OccurrenceCorrespondence(
                    source_face.occurrence.occurrence_id,
                    target_id,
                    canonical_fingerprint(
                        {
                            "kind": "occt-exact-planar-face-history",
                            "plan_id": plan.plan_id,
                            "source": source_face.occurrence.occurrence_id,
                            "target": target_id,
                        }
                    ),
                )
            )
        for source_edge in source_face.edges:
            edge_history = _capture_history(
                builder,
                source_edge.shape,
                roundtrip.edges,
                TopAbs_EDGE,
                TopoDS.Edge_s,
            )
            mapped_edges = tuple(
                roundtrip.edge_map[index] for index in edge_history.target_indices
            )
            evidence_rows.append(
                (
                    source_edge.occurrence.occurrence_id,
                    mapped_edges,
                    edge_history.modified_count,
                    edge_history.generated_count,
                    edge_history.deleted,
                )
            )
            for target_edge_index in mapped_edges:
                for target_face_index in target_model.topology.edge_faces[
                    target_edge_index
                ]:
                    target_id = _edge_occurrence_id(target_face_index, target_edge_index)
                    correspondences.append(
                        OccurrenceCorrespondence(
                            source_edge.occurrence.occurrence_id,
                            target_id,
                            canonical_fingerprint(
                                {
                                    "kind": "occt-exact-planar-edge-history",
                                    "plan_id": plan.plan_id,
                                    "source": source_edge.occurrence.occurrence_id,
                                    "target": target_id,
                                }
                            ),
                        )
                    )
    pairs = {
        (value.source_occurrence_id, value.target_occurrence_id): value
        for value in correspondences
    }
    correspondences = sorted(
        pairs.values(),
        key=lambda value: (
            value.source_occurrence_id,
            value.target_occurrence_id,
        ),
    )
    source_ids = {value.occurrence_id for value in source_revision.occurrences}
    target_ids = {value.occurrence_id for value in target_revision.occurrences}
    edge_sources = {value.source_occurrence_id for value in correspondences}
    edge_targets = {value.target_occurrence_id for value in correspondences}
    if edge_targets != target_ids:
        raise BRepPartitionHistoryError(
            "OCCT planar history does not exhaust the final face-edge inventory."
        )
    certificate_id = canonical_fingerprint(
        {
            "kind": "occt-cells-builder-exhaustive-planar-history",
            "plan_id": plan.plan_id,
            "source_revision": source_revision.revision_id,
            "target_revision": target_revision.revision_id,
            "history": sorted(evidence_rows),
            "edges": sorted(pairs),
            "deleted": sorted(source_ids - edge_sources),
        }
    )
    coverage = AssociationCoverageEvidence(
        True,
        True,
        certificate_id,
        "OCP.BOPAlgo_CellsBuilder",
    )
    transaction = OccurrenceCorrespondenceTransaction(
        canonical_fingerprint(
            {
                "kind": "planar-brep-partition-correspondence-transaction",
                "plan_id": plan.plan_id,
                "certificate": certificate_id,
            }
        ),
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(correspondences),
        frozenset(),
        frozenset(),
        coverage,
    )
    return AssociationGraph(source_revision, target_revision, transaction), certificate_id


def _regions_and_patches(
    plan: PlanarPartitionPlan,
    model: BRepModel,
    original_owners: tuple[str, ...],
    face_map: tuple[int, ...],
    /,
) -> tuple[tuple[BRepPartitionRegion, ...], tuple[BRepPartitionPatch, ...]]:
    model_owners = [""] * len(original_owners)
    for original_index, model_index in enumerate(face_map):
        model_owners[model_index] = original_owners[original_index]
    regions = tuple(
        BRepPartitionRegion(
            region_id,
            tuple(
                model.face_ids[index]
                for index, owner in enumerate(model_owners)
                if owner == region_id
            ),
            2,
        )
        for region_id in plan.policy.region_precedence
    )
    rank = {
        region_id: index for index, region_id in enumerate(plan.policy.region_precedence)
    }
    groups: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for edge_index, face_indices in enumerate(model.topology.edge_faces):
        if len(face_indices) not in (1, 2):
            raise BRepPartitionHistoryError(
                "Final planar edge is not exactly one- or two-sided."
            )
        adjacent = tuple(
            sorted(
                (model_owners[index] for index in face_indices),
                key=rank.__getitem__,
            )
        )
        groups.setdefault(adjacent, []).append(model.edge_ids[edge_index])
    patches: list[BRepPartitionPatch] = []
    for adjacent in sorted(
        groups,
        key=lambda value: tuple(rank[item] for item in value),
    ):
        if len(adjacent) == 1:
            name = f"boundary:{adjacent[0]}"
        elif adjacent[0] == adjacent[1]:
            name = f"internal:{adjacent[0]}"
        else:
            name = f"interface:{adjacent[0]}:{adjacent[1]}"
        patches.append(BRepPartitionPatch(name, tuple(groups[adjacent]), adjacent, 2))
    return regions, tuple(patches)


def partition_planar(
    plan: PlanarPartitionPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepPartitionResult:
    """Partition coplanar faces, certify exact history, and publish one BREP."""

    if not isinstance(plan, PlanarPartitionPlan):
        raise TypeError("plan must be a PlanarPartitionPlan.")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in {".brep", ".brp"}:
        raise ValueError(
            "A planar partition destination requires a .brep or .brp suffix."
        )
    if target.exists() and not plan.policy.overwrite:
        raise FileExistsError(target)
    loaded: dict[tuple[str, str], Any] = {}
    for operand in plan.operands:
        if isinstance(operand.source, BRepModel):
            key = (operand.source.source_id, operand.source.source_digest)
            if key not in loaded:
                loaded[key] = _load_model(operand.source)
    source_revision, source_faces, operand_shapes = _composite_sources(plan, loaded)

    arguments: list[Any] = []
    for source_face in source_faces:
        if not any(value.IsSame(source_face.shape) for value in arguments):
            arguments.append(source_face.shape)
    identity_partition = len(arguments) == 1 and len(source_faces) == 1
    if identity_partition:
        builder = None
        atoms = (source_faces[0].shape,)
        operand_members = {
            operand_id: frozenset(
                0 for shape in shapes if shape.IsSame(source_faces[0].shape)
            )
            for operand_id, shapes in operand_shapes.items()
        }
        final_shape = source_faces[0].shape
    else:
        builder = BOPAlgo_CellsBuilder()
        builder.SetRunParallel(plan.policy.run_parallel)
        builder.SetNonDestructive(True)
        builder.SetToFillHistory(True)
        for argument in arguments:
            builder.AddArgument(argument)
        builder.Perform()
        if builder.HasErrors():
            raise RuntimeError("OCCT failed to construct the exact planar Boolean cells.")
        atoms = _explore_unique(builder.GetAllParts(), TopAbs_FACE, TopoDS.Face_s)
        if not atoms:
            raise RuntimeError("OCCT produced no planar partition cells.")
        operand_members = {
            operand_id: frozenset(
                atom_index
                for shape in shapes
                for atom_index in _members(builder, shape, atoms)
            )
            for operand_id, shapes in operand_shapes.items()
        }
        final_shape = _build_final_shape(plan, builder, operand_shapes)
    atom_owners = _classify_atoms(plan, operand_members, len(atoms))
    if not BRepCheck_Analyzer(final_shape).IsValid():
        raise RuntimeError("OCCT produced an invalid planar partition topology.")
    if _explore_unique(final_shape, TopAbs_SOLID, TopoDS.Solid_s):
        raise RuntimeError("OCCT produced solids for a planar partition.")
    final_faces = _explore_unique(final_shape, TopAbs_FACE, TopoDS.Face_s)
    final_edges = _explore_unique(final_shape, TopAbs_EDGE, TopoDS.Edge_s)
    if not final_faces or not final_edges:
        raise RuntimeError("OCCT produced an incomplete planar partition result.")
    for face in final_faces:
        _require_coplanar_face(face, plan.embedding)
    original_owners = _final_face_owners(final_faces, atoms, atom_owners)

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.partition-",
        suffix=".brep",
        dir=target.parent,
    )
    os.close(descriptor)
    staging = Path(staging_name)
    staging.unlink()
    try:
        staged_model = persist_occt_shape(
            final_shape,
            staging,
            coordinate_contract=plan.coordinate_contract,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
        reopened_shape, source_format, source_digest = read_occt_shape(staging)
        if source_format != "brep" or source_digest != staged_model.source_digest:
            raise BRepPartitionHistoryError(
                "Staged planar partition failed exact identity verification."
            )
        roundtrip = _verify_roundtrip(
            final_shape,
            reopened_shape,
            staged_model,
            plan.embedding,
        )
        if identity_partition:
            association_graph, history_certificate_id = _identity_association_graph(
                plan,
                source_revision,
                source_faces,
                staged_model,
                roundtrip,
            )
        else:
            association_graph, history_certificate_id = _association_graph(
                plan,
                source_revision,
                source_faces,
                staged_model,
                roundtrip,
                builder,
            )
        regions, patches = _regions_and_patches(
            plan,
            staged_model,
            original_owners,
            roundtrip.face_map,
        )
        correspondence_sources = {
            value.source_occurrence_id
            for value in association_graph.transaction.correspondences
        }
        correspondence_targets = {
            value.target_occurrence_id
            for value in association_graph.transaction.correspondences
        }
        report = BRepPartitionReport(
            plan_id=plan.plan_id,
            model_id=staged_model.model_id,
            source_revision_id=source_revision.revision_id,
            target_revision_id=staged_model.source_revision,
            history_certificate_id=history_certificate_id,
            source_solid_occurrences=0,
            source_face_occurrences=sum(
                value.kind == "face" for value in source_revision.occurrences
            ),
            target_solids=0,
            target_faces=staged_model.topology.num_faces,
            deleted_source_occurrences=sum(
                value.occurrence_id not in correspondence_sources
                for value in source_revision.occurrences
            ),
            created_target_occurrences=sum(
                value.occurrence_id not in correspondence_targets
                for value in association_graph.target_revision.occurrences
            ),
            source_edge_occurrences=sum(
                value.kind == "edge" for value in source_revision.occurrences
            ),
            target_edges=staged_model.topology.num_edges,
            topological_dimension=2,
        )
        BRepPartitionResult(
            staged_model,
            association_graph.target_revision,
            association_graph,
            regions,
            patches,
            report,
            2,
        )
        if target.exists() and not plan.policy.overwrite:
            raise FileExistsError(target)
        _publish_staged(staging, target, plan.policy.overwrite)
        final_model = import_brep(
            target,
            coordinate_contract=plan.coordinate_contract,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
        if (
            final_model.source_revision != staged_model.source_revision
            or final_model.model_id != staged_model.model_id
        ):
            raise RuntimeError(
                "Published planar partition differs from its verified staged artifact."
            )
        final_revision = cad_revision_from_brep_model(final_model)
        final_graph = AssociationGraph(
            association_graph.source_revision,
            final_revision,
            association_graph.transaction,
        )
        return BRepPartitionResult(
            final_model,
            final_revision,
            final_graph,
            regions,
            patches,
            report,
            2,
        )
    finally:
        staging.unlink(missing_ok=True)


__all__ = [
    "PlanarEmbedding",
    "PlanarPartitionOperand",
    "PlanarPartitionPlan",
    "partition_planar",
]
