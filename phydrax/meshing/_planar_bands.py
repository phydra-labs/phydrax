#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import normalized_identifier, positive_finite_float
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
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_planar,
    PlanarPartitionOperand,
    PlanarPartitionPlan,
)
from ..geometry.brep._planar import mesh_loops
from ..geometry.brep._planar_arrangement import (
    common_area,
    curve_segment,
    face_loops,
)
from ..geometry.simplicial import PlanarMeshRegion
from ..typing import checked
from ._controls import LayerSchedule


@dataclass(frozen=True, slots=True, init=False)
class PlanarBandControl:
    """Describe layer schedules on named sides of a straight partition patch."""

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
        patch = normalized_identifier(str(patch_name), "PlanarBandControl.patch_name")
        if isinstance(side_schedules, Mapping):
            entries = tuple(side_schedules.items())
        else:
            entries = tuple(side_schedules)
        if not entries:
            raise ValueError(
                f"PlanarBandControl.side_schedules must contain a controlled side; got {len(entries)} entries."
            )
        normalized = tuple(
            sorted(
                (
                    (
                        normalized_identifier(
                            str(region), "PlanarBandControl.side_schedules.region"
                        ),
                        schedule,
                    )
                    for region, schedule in entries
                ),
                key=lambda value: value[0],
            )
        )
        if len({region for region, _ in normalized}) != len(normalized):
            raise ValueError(
                f"PlanarBandControl.side_schedules must name unique regions; got {tuple(region for region, _ in normalized)}."
            )
        if any(not isinstance(schedule, LayerSchedule) for _, schedule in normalized):
            raise TypeError(
                f"PlanarBandControl.side_schedules must map regions to LayerSchedule values; got {tuple(type(schedule).__name__ for _, schedule in normalized)}."
            )
        target = positive_finite_float(
            tangential_target, "PlanarBandControl.tangential_target"
        )
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


@dataclass(frozen=True, slots=True, init=False)
class PlanarBandPlan:
    """Request one CAD partition operation for nonintersecting straight bands."""

    partition: BRepPartitionResult
    embedding: PlanarEmbedding
    controls: tuple[PlanarBandControl, ...]
    plan_id: str = field(init=False)

    @checked
    def __init__(
        self,
        partition: BRepPartitionResult,
        embedding: PlanarEmbedding,
        controls: tuple[PlanarBandControl, ...],
    ) -> None:
        if partition.topological_dimension != 2 or partition.model.topology.num_solids:
            raise ValueError(
                f"PlanarBandPlan.partition must be a zero-solid 2D partition; got dimension={partition.topological_dimension}, solids={partition.model.topology.num_solids}."
            )
        controls = tuple(controls)
        if not controls or any(
            not isinstance(control, PlanarBandControl) for control in controls
        ):
            raise TypeError(
                f"PlanarBandPlan.controls must contain at least one PlanarBandControl; got {len(controls)} entries of kinds {tuple(type(control).__name__ for control in controls)}."
            )
        patch_names = tuple(control.patch_name for control in controls)
        if len(set(patch_names)) != len(patch_names):
            raise ValueError(
                f"PlanarBandPlan.controls must name distinct partition patches; got {patch_names}."
            )
        patches = {patch.name: patch for patch in partition.patches}
        unknown = set(patch_names) - set(patches)
        if unknown:
            raise ValueError(
                f"PlanarBandPlan.controls.patch_name must identify partition patches; got unknown names {sorted(unknown)}."
            )
        for control in controls:
            patch = patches[control.patch_name]
            requested_sides = tuple(region for region, _ in control.side_schedules)
            if tuple(sorted(requested_sides)) != tuple(sorted(patch.adjacent_region_ids)):
                raise ValueError(
                    f"PlanarBandPlan.side_schedules must name every patch-adjacent region once; got {requested_sides}, expected {patch.adjacent_region_ids}."
                )
            if len(patch.entity_ids) != 1:
                raise ValueError(
                    f"PlanarBandPlan.patch.entity_ids must contain one straight edge; got {len(patch.entity_ids)} entities for {patch.name!r}."
                )
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "planar-band-plan",
                    "partition": partition.report.target_revision_id,
                    "embedding": embedding.embedding_id,
                    "controls": tuple(control.control_id for control in controls),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanarBandLayer:
    """Describe a partitioned straight strip and its named layer front."""

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


@dataclass(frozen=True, slots=True, init=False)
class PlanarBandResult:
    """Publish the band partition and the identities needed for meshing."""

    partition: BRepPartitionResult
    embedding: PlanarEmbedding
    controls: tuple[PlanarBandControl, ...]
    layer_partitions: tuple[PlanarBandLayer, ...]
    plan_id: str
    result_id: str = field(init=False)

    @checked
    def __init__(
        self,
        partition: BRepPartitionResult,
        embedding: PlanarEmbedding,
        controls: tuple[PlanarBandControl, ...],
        layer_partitions: tuple[PlanarBandLayer, ...],
        plan_id: str,
    ) -> None:
        if partition.topological_dimension != 2:
            raise ValueError(
                f"PlanarBandResult.partition.dimension must be 2; got {partition.topological_dimension}."
            )
        controls = tuple(controls)
        layers = tuple(layer_partitions)
        if not controls or not layers:
            raise ValueError(
                f"PlanarBandResult.layers must contain controls and partitioned layers; got controls={len(controls)}, layers={len(layers)}."
            )
        control_ids = {control.control_id for control in controls}
        if any(layer.control_id not in control_ids for layer in layers):
            raise ValueError(
                f"PlanarBandResult.layers.control_id must reference a declared control; got {tuple(layer.control_id for layer in layers)}."
            )
        for layer in layers:
            if not layer.face_entity_ids:
                raise ValueError(
                    f"PlanarBandResult.layers.face_entity_ids must contain partition faces; got {layer.face_entity_ids}."
                )
            patch = partition.patch(layer.front_patch_name)
            if patch.adjacent_region_ids != (
                layer.region_name,
                layer.region_name,
            ):
                raise ValueError(
                    f"PlanarBandResult.front.adjacent_region_ids must retain layer region ownership; got {patch.adjacent_region_ids}, expected {(layer.region_name, layer.region_name)}."
                )
        plan = normalized_identifier(str(plan_id), "PlanarBandResult.plan_id")
        object.__setattr__(self, "partition", partition)
        object.__setattr__(self, "embedding", embedding)
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
                    "partition": partition.report.target_revision_id,
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


def _selection(
    partition: BRepPartitionResult, face_indices: tuple[int, ...], /
) -> CADSelectionSet:
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


def _edge_segments(
    model: BRepModel, embedding: PlanarEmbedding, /
) -> tuple[np.ndarray, ...]:
    return tuple(
        curve_segment(model, edge, embedding) for edge in range(model.topology.num_edges)
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
    target_segments = _edge_segments(raw.model, embedding)
    region_rank = {region.name: index for index, region in enumerate(source.regions)}
    groups: dict[tuple[str, tuple[str, ...]], list[BRepEntityId]] = {}
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
            raise RuntimeError(
                f"prepare_planar_bands.edge.source_patch must identify one source patch; got {matches}."
            )
        if matches:
            name = matches[0]
        else:
            front_matches = tuple(
                name
                for name, front in front_segments.items()
                if _segment_subset(segment, front, tolerance)
            )
            if len(front_matches) > 1:
                raise RuntimeError(
                    f"prepare_planar_bands.front.identity must identify one named front; got {front_matches}."
                )
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


@final
class _BandSource(StrictModule):
    """Prepare planar source loops, segment identities, and region operands."""

    faces: tuple[tuple[np.ndarray, ...], ...]
    segments: tuple[np.ndarray, ...]
    tolerance: float = eqx.field(static=True)
    region_by_face: dict[int, str] = eqx.field(static=True)
    operands: tuple[PlanarPartitionOperand, ...]


@final
class _BandSide(StrictModule):
    """Carry one straight segment and its uniquely contained inward strip."""

    segment: np.ndarray
    tangent: np.ndarray
    inward: np.ndarray


@final
class _BandLayerDraft(StrictModule):
    control: PlanarBandControl = eqx.field(static=True)
    region_name: str = eqx.field(static=True)
    layer_index: int = eqx.field(static=True)
    thickness: float = eqx.field(static=True)
    cumulative: float = eqx.field(static=True)
    temporary: str = eqx.field(static=True)
    tangent: np.ndarray
    inward: np.ndarray
    origin: np.ndarray
    front_name: str = eqx.field(static=True)


@final
class _BandOperands(StrictModule):
    operands: tuple[PlanarPartitionOperand, ...]
    temporary_owner: dict[str, str] = eqx.field(static=True)
    layers: tuple[_BandLayerDraft, ...]
    front_segments: dict[str, np.ndarray]


@final
class _BandLayers(StrictModule):
    layers: tuple[PlanarBandLayer, ...] = eqx.field(static=True)


def _prepare_band_source(plan: PlanarBandPlan, /) -> _BandSource:
    """Prepare host loops and the scale-dependent geometric tolerance."""
    source = plan.partition
    faces = tuple(
        face_loops(source.model, face, plan.embedding)
        for face in range(source.model.topology.num_faces)
    )
    original_segments = _edge_segments(source.model, plan.embedding)
    scale = max(
        1.0,
        float(np.max(np.abs(np.asarray(source.model.mesh_vertices)), initial=0.0)),
    )
    tolerance = float(4096.0 * np.finfo(np.float64).eps * scale)
    region_by_face = {
        entity.index: region.name
        for region in source.regions
        for entity in region.entity_ids
    }
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
    return _BandSource(
        faces, original_segments, tolerance, region_by_face, tuple(operands)
    )


def _prepare_band_side(
    plan: PlanarBandPlan,
    prepared: _BandSource,
    control: PlanarBandControl,
    region_name: str,
    schedule: LayerSchedule,
    /,
) -> _BandSide:
    """Find the single normal whose full scheduled strip lies inside the region."""
    source = plan.partition
    patch = source.patch(control.patch_name)
    edge_index = patch.entity_ids[0].index
    segment = prepared.segments[edge_index]
    tangent = segment[1] - segment[0]
    length = float(np.linalg.norm(tangent))
    tangent /= length
    candidate_normals = (
        np.asarray((-tangent[1], tangent[0])),
        np.asarray((tangent[1], -tangent[0])),
    )
    adjacent_faces = source.model.topology.edge_faces[edge_index]
    face_candidates = tuple(
        index for index in adjacent_faces if prepared.region_by_face[index] == region_name
    )
    if len(face_candidates) != 1:
        raise ValueError(
            f"prepare_planar_bands.side.region must meet one named-region face; got {region_name!r} with {len(face_candidates)} faces."
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
        candidate_shape = mesh_loops(candidate)
        area = common_area(candidate_shape, prepared.faces[face_index])
        if abs(area - expected_area) <= max(
            prepared.tolerance * max(length, total), 1e-11 * expected_area
        ):
            matching_normals.append(normal)
    if len(matching_normals) != 1:
        raise ValueError(
            f"prepare_planar_bands.side.clearance must contain one collision-free inward strip; got {len(matching_normals)} candidate normals for {control.patch_name!r}/{region_name!r}."
        )
    inward = matching_normals[0]
    return _BandSide(segment, tangent, inward)


def _prepare_band_operands(
    plan: PlanarBandPlan, prepared: _BandSource, /
) -> _BandOperands:
    """Build nonoverlapping layer operands and named front segments in request order."""
    operands = list(prepared.operands)
    temporary_owner: dict[str, str] = {}
    temporary_layers: list[_BandLayerDraft] = []
    occupied_rectangles: list[tuple[str, np.ndarray, tuple[np.ndarray, ...]]] = []
    front_segments: dict[str, np.ndarray] = {}
    for control_index, control in enumerate(plan.controls):
        for region_name, schedule in control.side_schedules:
            side = _prepare_band_side(plan, prepared, control, region_name, schedule)
            segment, tangent, inward = side.segment, side.tangent, side.inward
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
                rectangle_shape = mesh_loops(rectangle)
                rectangle_points = np.asarray(rectangle.vertices, dtype=np.float64)
                for other_name, other_points, other_shape in occupied_rectangles:
                    if (
                        _rectangles_overlap(
                            rectangle_points, other_points, prepared.tolerance
                        )
                        and common_area(rectangle_shape, other_shape)
                        > prepared.tolerance**2
                    ):
                        raise ValueError(
                            f"prepare_planar_bands.layers must not collide; got overlapping strips {control.patch_name!r} and {other_name!r}."
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
                    _BandLayerDraft(
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
    return _BandOperands(
        tuple(operands), temporary_owner, tuple(temporary_layers), front_segments
    )


def _publish_band_layers(
    raw: BRepPartitionResult, prepared: _BandOperands, /
) -> _BandLayers:
    """Bind each requested layer to its single partition face and front identity."""
    layers = []
    for draft in prepared.layers:
        control = draft.control
        region_name = draft.region_name
        layer_index = draft.layer_index
        thickness = draft.thickness
        cumulative = draft.cumulative
        temporary = draft.temporary
        tangent, inward, origin = draft.tangent, draft.inward, draft.origin
        front_name = draft.front_name
        face_ids = raw.region(temporary).entity_ids
        if len(face_ids) != 1:
            raise ValueError(
                f"prepare_planar_bands.layer.face_entity_ids must retain one straight strip face; got {len(face_ids)} for {temporary!r}."
            )
        layers.append(
            PlanarBandLayer(
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
                float(
                    np.linalg.norm(
                        prepared.front_segments[front_name][1]
                        - prepared.front_segments[front_name][0]
                    )
                ),
                control.tangential_target,
            )
        )
    return _BandLayers(tuple(layers))


@checked
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
    """Partition requested straight layer fronts in one CAD operation."""
    if not isinstance(overwrite, bool):
        raise TypeError(
            f"prepare_planar_bands.overwrite must be a bool; got {type(overwrite).__name__}."
        )
    source = plan.partition
    prepared_source = _prepare_band_source(plan)
    prepared = _prepare_band_operands(plan, prepared_source)
    precedence = (
        *(layer.temporary for layer in prepared.layers),
        *(region.name for region in source.regions),
    )
    raw = partition_planar(
        PlanarPartitionPlan(
            source.model.coordinate_contract,
            plan.embedding,
            prepared.operands,
            BRepPartitionPolicy(tuple(precedence), overwrite=overwrite),
        ),
        destination=destination,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )
    expected_faces = source.model.topology.num_faces + len(prepared.layers)
    if raw.model.topology.num_faces != expected_faces:
        raise ValueError(
            "prepare_planar_bands.partition.faces must retain the requested strip topology; "
            f"got {raw.model.topology.num_faces}, expected {expected_faces}."
        )
    organized = _organized_partition(
        raw,
        source,
        prepared.temporary_owner,
        prepared_source.segments,
        prepared.front_segments,
        plan.embedding,
        prepared_source.tolerance,
    )
    published = _publish_band_layers(raw, prepared)
    return PlanarBandResult(
        organized,
        plan.embedding,
        plan.controls,
        published.layers,
        plan.plan_id,
    )


__all__ = [
    "PlanarBandControl",
    "PlanarBandLayer",
    "PlanarBandPlan",
    "PlanarBandResult",
    "prepare_planar_bands",
]
