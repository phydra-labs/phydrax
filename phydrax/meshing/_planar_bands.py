#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .._fingerprint import canonical_fingerprint
from ..geometry._cad_revision import CADSelectionSet
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry.brep import (
    BRepEntityId,
    BRepImportReport,
    BRepModel,
    BRepPartitionPatch,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionResult,
    partition_planar,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
)
from ..geometry.brep._constructors import BRepTessellationPolicy
from ..geometry.brep._partition import BRepPartitionRole
from ..geometry.brep._patches import LineCurve, PlanePatch
from ..geometry.simplicial import PlanarMeshRegion
from ._controls import LayerSchedule


def _text(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True, slots=True, init=False)
class PlanarBandControl:
    """Exact normal layers on each named side of one straight partition patch."""

    patch_name: str
    side_schedules: tuple[tuple[str, LayerSchedule], ...]
    tangential_target: float
    control_id: str

    def __init__(
        self,
        patch_name: str,
        side_schedules: Mapping[str, LayerSchedule] | Sequence[tuple[str, LayerSchedule]],
        tangential_target: float,
        /,
    ) -> None:
        patch = _text(patch_name, "patch_name")
        if isinstance(side_schedules, Mapping):
            entries = tuple(side_schedules.items())
        else:
            entries = tuple(side_schedules)
        if not entries:
            raise ValueError("A planar band requires at least one controlled side.")
        normalized = tuple(
            sorted(
                (
                    (_text(region, "side region name"), schedule)
                    for region, schedule in entries
                ),
                key=lambda value: value[0],
            )
        )
        if len({region for region, _ in normalized}) != len(normalized):
            raise ValueError("A planar band cannot repeat a side region.")
        if any(not isinstance(schedule, LayerSchedule) for _, schedule in normalized):
            raise TypeError(
                "side_schedules must map region names to LayerSchedule values."
            )
        target = float(tangential_target)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("tangential_target must be positive and finite.")
        object.__setattr__(self, "patch_name", patch)
        object.__setattr__(self, "side_schedules", normalized)
        object.__setattr__(self, "tangential_target", target)
        object.__setattr__(
            self,
            "control_id",
            canonical_fingerprint(
                {
                    "kind": "planar-band-control",
                    "patch": patch,
                    "side_schedules": tuple(
                        (region, schedule.schedule_id) for region, schedule in normalized
                    ),
                    "tangential_target": target,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanarBandPlan:
    """One exact CAD partition operation for nonintersecting straight bands."""

    partition: BRepPartitionResult
    embedding: PlanarEmbedding
    controls: tuple[PlanarBandControl, ...]
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.partition, BRepPartitionResult):
            raise TypeError("partition must be a BRepPartitionResult.")
        if (
            self.partition.topological_dimension != 2
            or self.partition.model.topology.num_solids
        ):
            raise ValueError("Planar bands require a zero-solid 2D BRep partition.")
        if not isinstance(self.embedding, PlanarEmbedding):
            raise TypeError("embedding must be a PlanarEmbedding.")
        controls = tuple(self.controls)
        if not controls or any(
            not isinstance(control, PlanarBandControl) for control in controls
        ):
            raise TypeError("controls must contain at least one PlanarBandControl.")
        patch_names = tuple(control.patch_name for control in controls)
        if len(set(patch_names)) != len(patch_names):
            raise ValueError("A partition patch can carry at most one planar band.")
        patches = {patch.name: patch for patch in self.partition.patches}
        unknown = set(patch_names) - set(patches)
        if unknown:
            raise ValueError(f"Unknown planar band patches {sorted(unknown)}.")
        for control in controls:
            patch = patches[control.patch_name]
            requested_sides = tuple(region for region, _ in control.side_schedules)
            if tuple(sorted(requested_sides)) != tuple(sorted(patch.adjacent_region_ids)):
                raise ValueError(
                    "Planar band side schedules must name every patch-adjacent region exactly once."
                )
            if len(patch.entity_ids) != 1:
                raise ValueError(
                    "Planar bands require one exact straight edge, not a corner or junction patch."
                )
        object.__setattr__(self, "controls", controls)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "planar-band-plan",
                    "partition": self.partition.report.target_revision_id,
                    "embedding": self.embedding.embedding_id,
                    "controls": tuple(control.control_id for control in controls),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _PlanarBandLayer:
    control_id: str
    source_patch_name: str
    region_name: str
    layer_index: int
    thickness: float
    cumulative_distance: float
    face_entity_ids: tuple[BRepEntityId, ...]
    front_patch_name: str
    tangent: tuple[float, float]
    inward_normal: tuple[float, float]
    source_origin: tuple[float, float]
    source_length: float
    tangential_target: float


@dataclass(frozen=True, slots=True)
class PlanarBandResult:
    """In-memory exact band partition and identities needed for meshing."""

    partition: BRepPartitionResult
    embedding: PlanarEmbedding
    controls: tuple[PlanarBandControl, ...]
    layer_partitions: tuple[_PlanarBandLayer, ...]
    plan_id: str
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.partition, BRepPartitionResult):
            raise TypeError("partition must be a BRepPartitionResult.")
        if self.partition.topological_dimension != 2:
            raise ValueError("A planar band result must contain a 2D partition.")
        if not isinstance(self.embedding, PlanarEmbedding):
            raise TypeError("embedding must be a PlanarEmbedding.")
        controls = tuple(self.controls)
        layers = tuple(self.layer_partitions)
        if not controls or not layers:
            raise ValueError("A planar band result requires controls and exact layers.")
        control_ids = {control.control_id for control in controls}
        if any(layer.control_id not in control_ids for layer in layers):
            raise ValueError("A planar band layer references an unknown control.")
        for layer in layers:
            if not layer.face_entity_ids:
                raise ValueError(
                    "Every planar band layer requires exact partition faces."
                )
            patch = self.partition.patch(layer.front_patch_name)
            if patch.adjacent_region_ids != (
                layer.region_name,
                layer.region_name,
            ):
                raise ValueError("A planar band front has incorrect region ownership.")
        plan = _text(self.plan_id, "plan_id")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "layer_partitions", layers)
        object.__setattr__(self, "plan_id", plan)
        object.__setattr__(
            self,
            "result_id",
            canonical_fingerprint(
                {
                    "kind": "planar-band-result",
                    "plan": plan,
                    "partition": self.partition.report.target_revision_id,
                    "layers": tuple(
                        (
                            layer.control_id,
                            layer.region_name,
                            layer.layer_index,
                            tuple(str(value) for value in layer.face_entity_ids),
                            layer.front_patch_name,
                        )
                        for layer in layers
                    ),
                }
            ),
        )

    @property
    def model(self) -> BRepModel:
        return self.partition.model

    @property
    def report(self) -> BRepImportReport:
        return self.partition.model.report


def _curve_segment(
    model: BRepModel, edge_index: int, embedding: PlanarEmbedding, /
) -> np.ndarray:
    geometry = model.geometry
    curve_index = geometry.edge_curves[edge_index]
    if curve_index < 0:
        raise ValueError("Planar bands support exact straight patch edges only.")
    curve = geometry.curves[curve_index]
    if not isinstance(curve, LineCurve):
        raise ValueError("Planar bands support exact straight patch edges only.")
    edge_range = np.asarray(geometry.edge_ranges[edge_index], dtype=np.float64)
    curve_origin = np.asarray(curve.origin, dtype=np.float64)
    curve_direction = np.asarray(curve.direction, dtype=np.float64)
    world = np.stack(
        (
            curve_origin + edge_range[0] * curve_direction,
            curve_origin + edge_range[1] * curve_direction,
        )
    )
    vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
    endpoint_vertices = np.asarray(geometry.edge_vertices[edge_index], dtype=np.int64)
    endpoint_points = vertices[endpoint_vertices]
    tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(world), initial=0.0)))
    )
    direct_error = float(np.max(np.abs(world - endpoint_points), initial=0.0))
    reverse_error = float(np.max(np.abs(world - endpoint_points[::-1]), initial=0.0))
    if min(direct_error, reverse_error) > tolerance:
        raise ValueError("A planar band edge disagrees with its exact endpoint vertices.")
    if np.max(np.abs(embedding.plane_residual(world)), initial=0.0) > tolerance:
        raise ValueError("A planar band patch lies outside the declared embedding.")
    planar = embedding.to_planar(world)
    if np.linalg.norm(planar[1] - planar[0]) <= tolerance:
        raise ValueError("A planar band patch has zero length.")
    return planar


def _inward_normal(
    model: BRepModel,
    face_index: int,
    edge_index: int,
    embedding: PlanarEmbedding,
    segments: tuple[np.ndarray, ...],
    /,
) -> np.ndarray:
    geometry = model.geometry
    oriented_loops: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]] = []
    for coedge_loop in geometry.face_loops[face_index]:
        points: list[np.ndarray] = []
        edge_senses: list[tuple[int, int]] = []
        for coedge_index in coedge_loop:
            edge = geometry.coedge_edges[coedge_index]
            sense = geometry.coedge_senses[coedge_index]
            if sense not in (-1, 1):
                raise ValueError("A planar face has an invalid coedge orientation.")
            segment = segments[edge]
            directed = segment if sense == 1 else segment[::-1]
            points.append(directed[0])
            edge_senses.append((edge, sense))
        loop_points = np.asarray(points, dtype=np.float64)
        if loop_points.shape[0] < 3:
            raise ValueError("A planar face boundary loop requires at least three edges.")
        loop_area = _signed_area(loop_points)
        if loop_area == 0.0:
            raise ValueError("A planar face boundary loop must enclose positive area.")
        oriented_loops.append((loop_points, tuple(edge_senses)))
    if not oriented_loops:
        raise ValueError("A planar face requires an oriented boundary loop.")
    loop_areas = tuple(_signed_area(points) for points, _ in oriented_loops)
    outer = max(range(len(loop_areas)), key=lambda index: abs(loop_areas[index]))
    if outer != 0:
        raise ValueError("A planar face must declare its outer loop first.")
    outer_orientation = 1 if loop_areas[0] > 0.0 else -1
    if any(
        (1 if area > 0.0 else -1)
        != (outer_orientation if index == 0 else -outer_orientation)
        for index, area in enumerate(loop_areas)
    ):
        raise ValueError(
            "Planar face outer and hole loops have inconsistent orientation."
        )
    matching = tuple(
        sense
        for _, edges in oriented_loops
        for edge, sense in edges
        if edge == edge_index
    )
    if len(matching) != 1:
        raise ValueError(
            "A planar-band edge must have one oriented occurrence in its adjacent face."
        )
    tangent = segments[edge_index][1] - segments[edge_index][0]
    tangent *= matching[0]
    tangent /= np.linalg.norm(tangent)
    return outer_orientation * np.asarray((-tangent[1], tangent[0]))


def _rectangle(
    segment: np.ndarray,
    inward: np.ndarray,
    lower: float,
    upper: float,
    feature_id: str,
    /,
) -> PlanarMeshRegion:
    points = np.asarray(
        (
            segment[0] + lower * inward,
            segment[1] + lower * inward,
            segment[1] + upper * inward,
            segment[0] + upper * inward,
        ),
        dtype=np.float64,
    )
    signed_area = 0.5 * np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    if signed_area < 0.0:
        points = points[[0, 3, 2, 1]]
    return PlanarMeshRegion(points, ((0, 1, 2, 3),), feature_id=feature_id)


def _signed_area(points: np.ndarray, /) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _selection(
    partition: BRepPartitionResult, face_indices: tuple[int, ...], /
) -> CADSelectionSet:
    return CADSelectionSet.from_revision(
        partition.revision, tuple(f"face:{index}" for index in face_indices)
    )


def _front_edge_index(
    region: PlanarMeshRegion,
    segment: np.ndarray,
    tolerance: float,
    /,
) -> int:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    edges = np.asarray(region.edges, dtype=np.int64).reshape((-1, 2))
    matches = tuple(
        edge_index
        for edge_index, (first, second) in enumerate(edges)
        if min(
            float(np.max(np.abs(vertices[[first, second]] - segment), initial=0.0)),
            float(np.max(np.abs(vertices[[first, second]] - segment[::-1]), initial=0.0)),
        )
        <= tolerance
    )
    if len(matches) != 1:
        raise ValueError("A planar-band rectangle must retain one exact front edge.")
    return matches[0]


def _rectangles_overlap(
    first: np.ndarray, second: np.ndarray, tolerance: float, /
) -> bool:
    axes = []
    for polygon in (first, second):
        for index in range(2):
            edge = polygon[index + 1] - polygon[index]
            normal = np.asarray((-edge[1], edge[0]), dtype=np.float64)
            normal /= np.linalg.norm(normal)
            axes.append(normal)
    for axis in axes:
        first_interval = first @ axis
        second_interval = second @ axis
        overlap = min(
            float(np.max(first_interval)), float(np.max(second_interval))
        ) - max(float(np.min(first_interval)), float(np.min(second_interval)))
        if overlap <= tolerance:
            return False
    return True


def _entity_text(entity: BRepEntityId, /) -> str:
    return f"{entity.source_revision}:{entity.kind}:{entity.index}"


def _organized_partition(
    raw: BRepPartitionResult,
    source: BRepPartitionResult,
    temporary_owner: Mapping[str, str],
    front_source_edges: Mapping[str, tuple[str, int]],
    /,
) -> BRepPartitionResult:
    entities_by_region: dict[str, list[BRepEntityId]] = {
        region.name: [] for region in source.regions
    }
    for region in raw.regions:
        owner = temporary_owner.get(region.name, region.name)
        if owner not in entities_by_region:
            raise ValueError("A native band partition returned an undeclared region.")
        entities_by_region[owner].extend(region.entity_ids)
    regions = tuple(
        BRepPartitionRegion(name, tuple(entities_by_region[name]), 2)
        for name in entities_by_region
    )

    source_patch_by_entity = {
        _entity_text(entity): patch.name
        for patch in source.patches
        for entity in patch.entity_ids
    }
    target_edge_by_entity = {
        _entity_text(entity): entity.index for entity in raw.model.edge_ids
    }
    front_occurrence_names: dict[str, str] = {}
    for occurrence in raw.association_graph.source_revision.occurrences:
        if occurrence.kind != "edge" or occurrence.parent_occurrence_id is None:
            continue
        for temporary, (front_name, edge_index) in front_source_edges.items():
            if occurrence.parent_occurrence_id.startswith(
                f"operand:{temporary}:face:"
            ) and occurrence.occurrence_id.endswith(f"/edge:{edge_index}"):
                front_occurrence_names[occurrence.occurrence_id] = front_name
    target_names: dict[int, set[str]] = {}
    graph = raw.association_graph
    for correspondence in graph.transaction.correspondences:
        source_occurrence = graph.source_revision.occurrence(
            correspondence.source_occurrence_id
        )
        target_occurrence = graph.target_revision.occurrence(
            correspondence.target_occurrence_id
        )
        if source_occurrence.kind != "edge" or target_occurrence.kind != "edge":
            continue
        target_index = target_edge_by_entity.get(target_occurrence.entity_id)
        if target_index is None:
            continue
        names = target_names.setdefault(target_index, set())
        source_patch = source_patch_by_entity.get(source_occurrence.entity_id)
        if source_patch is not None:
            names.add(source_patch)
        front_name = front_occurrence_names.get(source_occurrence.occurrence_id)
        if front_name is not None:
            names.add(front_name)
    expected_front_names = {name for name, _ in front_source_edges.values()}
    mapped_front_names = {
        name
        for names in target_names.values()
        for name in names
        if name in expected_front_names
    }
    if mapped_front_names != expected_front_names:
        raise ValueError("Native partition lineage did not preserve every band front.")

    region_rank = {region.name: index for index, region in enumerate(source.regions)}
    adjacent_by_edge = {
        entity.index: patch.adjacent_region_ids
        for patch in raw.patches
        for entity in patch.entity_ids
    }
    groups: dict[tuple[str, tuple[str, ...]], list[BRepEntityId]] = {}
    for edge_index, entity in enumerate(raw.model.edge_ids):
        adjacent = tuple(
            sorted(
                (
                    temporary_owner.get(name, name)
                    for name in adjacent_by_edge[edge_index]
                ),
                key=region_rank.__getitem__,
            )
        )
        matches = target_names.get(edge_index, set())
        if len(matches) > 1:
            raise ValueError("Native edge lineage has ambiguous patch ownership.")
        name = next(iter(matches)) if matches else f"band-seam:{':'.join(adjacent)}"
        groups.setdefault((name, adjacent), []).append(entity)
    duplicate_names: dict[str, int] = {}
    patches = []
    for (name, adjacent), entity_ids in sorted(groups.items()):
        count = duplicate_names.get(name, 0)
        duplicate_names[name] = count + 1
        unique_name = name if count == 0 else f"{name}:{count}"
        patches.append(BRepPartitionPatch(unique_name, tuple(entity_ids), adjacent, 2))
    return BRepPartitionResult(
        raw.model,
        raw.revision,
        raw.association_graph,
        regions,
        tuple(patches),
        raw.report,
        2,
    )


def _require_coplanar_face(
    model: BRepModel,
    face_index: int,
    embedding: PlanarEmbedding,
    tolerance: float,
    /,
) -> None:
    patch = model.patches[face_index]
    if not isinstance(patch, PlanePatch):
        raise ValueError("Planar bands require exact planar source faces.")
    first = np.asarray(patch.first_axis, dtype=np.float64)
    second = np.asarray(patch.second_axis, dtype=np.float64)
    cross = np.cross(first, second)
    norm = float(np.linalg.norm(cross))
    origin = np.asarray(patch.origin, dtype=np.float64)
    if not np.isfinite(norm) or norm == 0.0 or not np.all(np.isfinite(origin)):
        raise ValueError("A planar source face has an invalid plane frame.")
    normal = cross / norm
    embedding_normal = np.asarray(embedding.normal, dtype=np.float64)
    alignment = float(normal @ embedding_normal)
    if not np.isfinite(alignment) or abs(abs(alignment) - 1.0) > 1.0e-12:
        raise ValueError("A planar source face is not parallel to its embedding.")
    if abs(float(embedding.plane_residual(origin))) > tolerance:
        raise ValueError("A planar source face lies outside its declared embedding.")


def prepare_planar_bands(
    plan: PlanarBandPlan,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> PlanarBandResult:
    """Partition all requested straight layer fronts in memory."""

    if not isinstance(plan, PlanarBandPlan):
        raise TypeError("plan must be a PlanarBandPlan.")
    if tessellation is not None and not isinstance(tessellation, BRepTessellationPolicy):
        raise TypeError("tessellation must be a BRepTessellationPolicy or None.")
    source = plan.partition
    vertices = np.asarray(source.model.geometry.vertex_points, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(vertices), initial=0.0)))
    tolerance = float(4096.0 * np.finfo(np.float64).eps * scale)
    for face_index in range(source.model.topology.num_faces):
        _require_coplanar_face(source.model, face_index, plan.embedding, tolerance)
    original_segments = tuple(
        _curve_segment(source.model, edge_index, plan.embedding)
        for edge_index in range(source.model.topology.num_edges)
    )
    region_by_face = {
        entity.index: region.name
        for region in source.regions
        for entity in region.entity_ids
    }
    operands: list[PlanarPartitionOperand] = []
    for region in source.regions:
        face_indices = tuple(entity.index for entity in region.entity_ids)
        operands.append(
            PlanarPartitionOperand(
                region.name,
                source.model,
                BRepPartitionRole.REGION,
                _selection(source, face_indices),
            )
        )

    temporary_owner: dict[str, str] = {}
    temporary_layers: list[
        tuple[
            PlanarBandControl,
            str,
            int,
            float,
            float,
            str,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            str,
            float,
        ]
    ] = []
    occupied_rectangles: list[tuple[str, np.ndarray]] = []
    front_source_edges: dict[str, tuple[str, int]] = {}
    for control_index, control in enumerate(plan.controls):
        patch = source.patch(control.patch_name)
        edge_index = patch.entity_ids[0].index
        segment = original_segments[edge_index]
        tangent = segment[1] - segment[0]
        length = float(np.linalg.norm(tangent))
        tangent /= length
        adjacent_faces = source.model.topology.edge_faces[edge_index]
        for region_name, schedule in control.side_schedules:
            face_candidates = tuple(
                index for index in adjacent_faces if region_by_face[index] == region_name
            )
            if len(face_candidates) != 1:
                raise ValueError(
                    "A planar band side must meet exactly one face of its named region."
                )
            face_index = face_candidates[0]
            inward = _inward_normal(
                source.model,
                face_index,
                edge_index,
                plan.embedding,
                original_segments,
            )
            cumulative = np.cumsum(np.asarray(schedule.thicknesses, dtype=np.float64))
            lower = 0.0
            for layer_index, upper_value in enumerate(cumulative):
                upper = float(upper_value)
                temporary = f"__planar_band__:{control_index}:{region_name}:{layer_index}"
                rectangle = _rectangle(
                    segment,
                    inward,
                    lower,
                    upper,
                    temporary,
                )
                rectangle_points = np.asarray(rectangle.vertices, dtype=np.float64)
                for other_name, other_points in occupied_rectangles:
                    if _rectangles_overlap(rectangle_points, other_points, tolerance):
                        raise ValueError(
                            f"Planar bands {control.patch_name!r} and {other_name!r} collide."
                        )
                occupied_rectangles.append((control.patch_name, rectangle_points))
                operands.append(
                    PlanarPartitionOperand(
                        temporary,
                        rectangle,
                        BRepPartitionRole.REGION,
                    )
                )
                temporary_owner[temporary] = region_name
                front_name = (
                    f"{control.patch_name}:band:{region_name}:front:{layer_index + 1}"
                )
                front = segment + upper * inward
                front_source_edges[temporary] = (
                    front_name,
                    _front_edge_index(rectangle, front, tolerance),
                )
                temporary_layers.append(
                    (
                        control,
                        region_name,
                        layer_index,
                        float(schedule.thicknesses[layer_index]),
                        upper,
                        temporary,
                        tangent.copy(),
                        inward.copy(),
                        segment[0].copy(),
                        front_name,
                        float(np.linalg.norm(front[1] - front[0])),
                    )
                )
                lower = upper

    precedence = (
        *(layer[5] for layer in temporary_layers),
        *(region.name for region in source.regions),
    )
    raw = partition_planar(
        PlanarPartitionPlan(
            source.model.coordinate_contract,
            plan.embedding,
            tuple(operands),
            BRepPartitionPolicy(tuple(precedence)),
        ),
        tessellation=tessellation,
    )
    expected_faces = source.model.topology.num_faces + len(temporary_layers)
    if raw.model.topology.num_faces != expected_faces:
        raise ValueError(
            "Planar-band partition changed topology beyond the requested straight strips."
        )
    organized = _organized_partition(
        raw,
        source,
        temporary_owner,
        front_source_edges,
    )
    layers = []
    for (
        control,
        region_name,
        layer_index,
        thickness,
        cumulative,
        temporary,
        tangent,
        inward,
        origin,
        front_name,
        front_length,
    ) in temporary_layers:
        face_ids = raw.region(temporary).entity_ids
        if len(face_ids) != 1:
            raise ValueError(
                "An exact planar-band layer did not remain one straight strip face."
            )
        front_patches = tuple(
            patch
            for patch in organized.patches
            if patch.name == front_name or patch.name.startswith(f"{front_name}:")
        )
        if not front_patches or any(
            patch.adjacent_region_ids != (region_name, region_name)
            for patch in front_patches
        ):
            raise ValueError(
                "A planar-band front lacks exact clearance inside its named region."
            )
        layers.append(
            _PlanarBandLayer(
                control.control_id,
                control.patch_name,
                region_name,
                layer_index,
                thickness,
                cumulative,
                face_ids,
                front_name,
                (float(tangent[0]), float(tangent[1])),
                (float(inward[0]), float(inward[1])),
                (float(origin[0]), float(origin[1])),
                front_length,
                control.tangential_target,
            )
        )
    return PlanarBandResult(
        organized,
        plan.embedding,
        plan.controls,
        tuple(layers),
        plan.plan_id,
    )


__all__ = [
    "PlanarBandControl",
    "PlanarBandPlan",
    "PlanarBandResult",
    "prepare_planar_bands",
]
