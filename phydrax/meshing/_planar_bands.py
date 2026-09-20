#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve  # ty: ignore[unresolved-import]
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common  # ty: ignore[unresolved-import]
from OCP.BRepGProp import BRepGProp  # ty: ignore[unresolved-import]
from OCP.GeomAbs import GeomAbs_Line  # ty: ignore[unresolved-import]
from OCP.GProp import GProp_GProps  # ty: ignore[unresolved-import]
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE  # ty: ignore[unresolved-import]
from OCP.TopoDS import TopoDS  # ty: ignore[unresolved-import]

from .._fingerprint import canonical_fingerprint
from ..geometry._cad_revision import CADSelectionSet
from ..geometry.brep import (
    BRepEntityId,
    BRepPartitionPatch,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionResult,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_planar,
    PlanarEmbedding,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
    read_occt_shape,
)
from ..geometry.brep._planar import (
    _explore_unique,
    _face_from_mesh,
    _require_coplanar_face,
)
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
    ):
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
    """Published exact band partition and the identities needed for meshing."""

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
    def model(self):
        return self.partition.model

    @property
    def report(self):
        return self.partition.model.report


def _curve_segment(edge: Any, embedding: PlanarEmbedding, /) -> np.ndarray:
    curve = BRepAdaptor_Curve(edge)
    if curve.GetType() != GeomAbs_Line:
        raise ValueError("Planar bands support only exact straight patch edges.")
    points = tuple(
        curve.Value(float(parameter))
        for parameter in (curve.FirstParameter(), curve.LastParameter())
    )
    world = np.asarray(
        tuple((point.X(), point.Y(), point.Z()) for point in points), dtype=np.float64
    )
    tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(world), initial=0.0)))
    )
    if np.max(np.abs(embedding.plane_residual(world)), initial=0.0) > tolerance:
        raise ValueError("A planar band patch lies outside the declared embedding.")
    planar = embedding.to_planar(world)
    if np.linalg.norm(planar[1] - planar[0]) <= tolerance:
        raise ValueError("A planar band patch has zero length.")
    return planar


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


def _surface_area(shape: Any, /) -> float:
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, properties)
    return abs(float(properties.Mass()))


def _common_area(first: Any, second: Any, /) -> float:
    operation = BRepAlgoAPI_Common(first, second)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError("OCCT failed to evaluate exact planar-band clearance.")
    return _surface_area(operation.Shape())


def _selection(partition: BRepPartitionResult, face_indices: tuple[int, ...], /):
    revision = cad_revision_from_brep_model(partition.model)
    return CADSelectionSet.from_revision(
        revision, tuple(f"face:{index}" for index in face_indices)
    )


def _segment_subset(
    candidate: np.ndarray, reference: np.ndarray, tolerance: float, /
) -> bool:
    direction = reference[1] - reference[0]
    length = float(np.linalg.norm(direction))
    unit = direction / length
    offsets = candidate - reference[0]
    normal = np.asarray((-unit[1], unit[0]))
    parameters = offsets @ unit
    return bool(
        np.max(np.abs(offsets @ normal), initial=0.0) <= tolerance
        and np.min(parameters, initial=0.0) >= -tolerance
        and np.max(parameters, initial=0.0) <= length + tolerance
    )


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


def _edge_segments(shape: Any, embedding: PlanarEmbedding, /) -> tuple[np.ndarray, ...]:
    return tuple(
        _curve_segment(edge, embedding)
        for edge in _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)
    )


def _organized_partition(
    raw: BRepPartitionResult,
    source: BRepPartitionResult,
    temporary_owner: Mapping[str, str],
    original_segments: tuple[np.ndarray, ...],
    front_segments: Mapping[str, np.ndarray],
    embedding: PlanarEmbedding,
    tolerance: float,
    /,
) -> BRepPartitionResult:
    owner_by_face: dict[int, str] = {}
    for region in raw.regions:
        owner = temporary_owner.get(region.name, region.name)
        for entity in region.entity_ids:
            owner_by_face[entity.index] = owner
    regions = tuple(
        BRepPartitionRegion(
            region.name,
            tuple(
                raw.model.face_ids[index]
                for index in range(raw.model.topology.num_faces)
                if owner_by_face[index] == region.name
            ),
            2,
        )
        for region in source.regions
    )
    source_patch_by_edge = {
        entity.index: patch.name
        for patch in source.patches
        for entity in patch.entity_ids
    }
    target_shape, _, _ = read_occt_shape(raw.model.source_id)
    target_segments = _edge_segments(target_shape, embedding)
    region_rank = {region.name: index for index, region in enumerate(source.regions)}
    groups: dict[tuple[str, tuple[str, ...]], list[Any]] = {}
    for edge_index, (segment, face_indices) in enumerate(
        zip(target_segments, raw.model.topology.edge_faces, strict=True)
    ):
        adjacent = tuple(
            sorted(
                (owner_by_face[index] for index in face_indices),
                key=region_rank.__getitem__,
            )
        )
        matches = tuple(
            source_patch_by_edge[index]
            for index, source_segment in enumerate(original_segments)
            if _segment_subset(segment, source_segment, tolerance)
        )
        if len(set(matches)) > 1:
            raise RuntimeError("A partition edge has ambiguous source-patch ownership.")
        if matches:
            name = matches[0]
        else:
            front_matches = tuple(
                name
                for name, front in front_segments.items()
                if _segment_subset(segment, front, tolerance)
            )
            if len(front_matches) > 1:
                raise RuntimeError("A planar-band front has ambiguous exact ownership.")
            name = (
                front_matches[0] if front_matches else f"band-seam:{':'.join(adjacent)}"
            )
        groups.setdefault((name, adjacent), []).append(raw.model.edge_ids[edge_index])
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


def prepare_planar_bands(
    plan: PlanarBandPlan,
    /,
    *,
    destination: str | Path,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
    overwrite: bool = False,
) -> PlanarBandResult:
    """Partition all requested straight layer fronts in one certified CAD operation."""

    if not isinstance(plan, PlanarBandPlan):
        raise TypeError("plan must be a PlanarBandPlan.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool.")
    source = plan.partition
    shape, source_format, source_digest = read_occt_shape(source.model.source_id)
    if (
        source_format != source.model.report.source_format
        or source_digest != source.model.report.source_digest
    ):
        raise ValueError("The planar-band source bytes changed after partitioning.")
    faces = _explore_unique(shape, TopAbs_FACE, TopoDS.Face_s)
    edges = _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)
    if (
        len(faces) != source.model.topology.num_faces
        or len(edges) != source.model.topology.num_edges
    ):
        raise ValueError("The planar-band source topology changed after import.")
    original_segments = tuple(_curve_segment(edge, plan.embedding) for edge in edges)
    scale = max(
        1.0,
        float(np.max(np.abs(np.asarray(source.model.mesh_vertices)), initial=0.0)),
    )
    tolerance = 4096.0 * np.finfo(np.float64).eps * scale
    region_by_face = {
        entity.index: region.name
        for region in source.regions
        for entity in region.entity_ids
    }
    for face in faces:
        _require_coplanar_face(face, plan.embedding)
    operands = []
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
        ]
    ] = []
    occupied_rectangles: list[tuple[str, np.ndarray, Any]] = []
    front_segments: dict[str, np.ndarray] = {}
    for control_index, control in enumerate(plan.controls):
        patch = source.patch(control.patch_name)
        edge_index = patch.entity_ids[0].index
        segment = original_segments[edge_index]
        tangent = segment[1] - segment[0]
        length = float(np.linalg.norm(tangent))
        tangent /= length
        candidate_normals = (
            np.asarray((-tangent[1], tangent[0])),
            np.asarray((tangent[1], -tangent[0])),
        )
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
            total = schedule.total_thickness
            expected_area = length * total
            matching_normals = []
            for normal_index, normal in enumerate(candidate_normals):
                candidate = _rectangle(
                    segment,
                    normal,
                    0.0,
                    total,
                    f"band-probe:{control.control_id}:{region_name}:{normal_index}",
                )
                candidate_shape = _face_from_mesh(candidate, plan.embedding)
                area = _common_area(candidate_shape, faces[face_index])
                if abs(area - expected_area) <= max(
                    tolerance * max(length, total), 1e-11 * expected_area
                ):
                    matching_normals.append(normal)
            if len(matching_normals) != 1:
                raise ValueError(
                    "A planar band lacks unique collision-free clearance inside its adjacent region."
                )
            inward = matching_normals[0]
            cumulative = np.cumsum(np.asarray(schedule.thicknesses, dtype=np.float64))
            lower = 0.0
            for layer_index, upper in enumerate(cumulative):
                temporary = f"__planar_band__:{control_index}:{region_name}:{layer_index}"
                rectangle = _rectangle(
                    segment,
                    inward,
                    lower,
                    float(upper),
                    temporary,
                )
                rectangle_shape = _face_from_mesh(rectangle, plan.embedding)
                rectangle_points = np.asarray(rectangle.vertices, dtype=np.float64)
                for other_name, other_points, other_shape in occupied_rectangles:
                    if (
                        _rectangles_overlap(rectangle_points, other_points, tolerance)
                        and _common_area(rectangle_shape, other_shape) > tolerance**2
                    ):
                        raise ValueError(
                            f"Planar bands {control.patch_name!r} and {other_name!r} collide."
                        )
                occupied_rectangles.append(
                    (control.patch_name, rectangle_points, rectangle_shape)
                )
                operands.append(
                    PlanarPartitionOperand(temporary, rectangle, BRepPartitionRole.REGION)
                )
                temporary_owner[temporary] = region_name
                front_name = (
                    f"{control.patch_name}:band:{region_name}:front:{layer_index + 1}"
                )
                front = segment + float(upper) * inward
                front_segments[front_name] = front
                temporary_layers.append(
                    (
                        control,
                        region_name,
                        layer_index,
                        float(schedule.thicknesses[layer_index]),
                        float(upper),
                        temporary,
                        tangent.copy(),
                        inward.copy(),
                        segment[0].copy(),
                        front_name,
                    )
                )
                lower = float(upper)

    precedence = (
        *(layer[5] for layer in temporary_layers),
        *(region.name for region in source.regions),
    )
    raw = partition_planar(
        PlanarPartitionPlan(
            source.model.coordinate_contract,
            plan.embedding,
            tuple(operands),
            BRepPartitionPolicy(tuple(precedence), overwrite=overwrite),
        ),
        destination=destination,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
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
        original_segments,
        front_segments,
        plan.embedding,
        tolerance,
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
    ) in temporary_layers:
        face_ids = raw.region(temporary).entity_ids
        if len(face_ids) != 1:
            raise ValueError(
                "An exact planar-band layer did not remain one straight strip face."
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
                tuple(float(value) for value in tangent),
                tuple(float(value) for value in inward),
                tuple(float(value) for value in origin),
                float(
                    np.linalg.norm(
                        front_segments[front_name][1] - front_segments[front_name][0]
                    )
                ),
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
