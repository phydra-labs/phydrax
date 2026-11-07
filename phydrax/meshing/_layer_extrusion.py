#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact CAD extrusion of planar walls into sweepable boundary-layer slabs.

An EXACT_SWEEP control realizes a pre-partitioned straight slab cap to cap.
CAD_EXTRUSION first extrudes planar walls along their inward normals and splits
the source in memory. The partition retains a named core region when material
remains; a slab that covers its entire source solid has an exterior cap and no
core region. Every accepted slab has exact source coverage, positive volume,
and lateral faces on the original source boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._cad_revision import AssociationGraph, CADRevision, CADSelectionSet
from ..geometry.brep import (
    BRepModel,
    BRepPartitionOperand,
    BRepPartitionPatch,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRegion,
    BRepPartitionReport,
    BRepPartitionResult,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_brep,
)
from ..geometry.brep._constructors import (
    brep_extrusion,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from ..geometry.brep._patches import ExactLineCurve, LineCurve, PlanePatch
from ..geometry.brep._projection_contracts import brep_entity_id
from ..typing import checked
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._controls import BoundaryLayerControl, BoundaryLayerRoute
from ._scope import MeshingEntityKind, MeshingScope
from ._trace import MeshingStageKind


_STAGE = MeshingStageKind.LAYER_GENERATION.value


def _unsupported(message: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.UNSUPPORTED_COMBINATION, message, stage=_STAGE
    )


def _plane(model: BRepModel, face_index: int, /) -> tuple[np.ndarray, float] | None:
    patch = model.patches[face_index]
    if not isinstance(patch, PlanePatch):
        return None
    origin = np.asarray(patch.origin, dtype=np.float64)
    first = np.asarray(patch.first_axis, dtype=np.float64)
    second = np.asarray(patch.second_axis, dtype=np.float64)
    first_norm = float(np.linalg.norm(first))
    cross = np.cross(first, second)
    cross_norm = float(np.linalg.norm(cross))
    if (
        not np.all(np.isfinite(origin))
        or not np.isfinite(first_norm)
        or not np.isfinite(cross_norm)
        or first_norm == 0.0
        or cross_norm == 0.0
    ):
        return None
    normal = cross / cross_norm
    if float(np.asarray(model.orientation)[face_index]) < 0.0:
        normal = -normal
    return normal, float(normal @ origin)


def _loop_area(points: np.ndarray, /) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _profile_from_face(
    model: BRepModel, face_index: int, /
) -> tuple[PlanarProfile, np.ndarray, float, float]:
    plane = _plane(model, face_index)
    if plane is None:
        raise _unsupported("CAD_EXTRUSION requires planar wall faces.")
    normal, offset = plane
    patch = model.patches[face_index]
    if not isinstance(patch, PlanePatch):
        raise RuntimeError("A validated planar face lost its PlanePatch.")
    origin = np.asarray(patch.origin, dtype=np.float64)
    first = np.asarray(patch.first_axis, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    geometry = model.geometry
    vertex_points = np.asarray(geometry.vertex_points, dtype=np.float64)
    loops: list[np.ndarray] = []
    tolerance = (
        512.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(vertex_points), initial=0.0)))
    )
    for coedge_loop in geometry.face_loops[face_index]:
        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        for coedge_index in coedge_loop:
            edge_index = geometry.coedge_edges[coedge_index]
            curve_index = geometry.edge_curves[edge_index]
            if curve_index < 0:
                raise _unsupported(
                    "CAD_EXTRUSION requires line-bounded planar wall faces."
                )
            curve = geometry.curves[curve_index]
            if not isinstance(curve, LineCurve):
                raise _unsupported(
                    "CAD_EXTRUSION requires line-bounded planar wall faces."
                )
            edge_range = np.asarray(geometry.edge_ranges[edge_index], dtype=np.float64)
            edge_vertices = tuple(
                int(value) for value in geometry.edge_vertices[edge_index]
            )
            if len(edge_vertices) != 2:
                raise _unsupported("A wall boundary edge must have two endpoints.")
            curve_origin = np.asarray(curve.origin, dtype=np.float64)
            curve_direction = np.asarray(curve.direction, dtype=np.float64)
            curve_endpoints = np.stack(
                (
                    curve_origin + edge_range[0] * curve_direction,
                    curve_origin + edge_range[1] * curve_direction,
                )
            )
            vertex_endpoints = vertex_points[np.asarray(edge_vertices)]
            direct_error = float(
                np.max(np.abs(curve_endpoints - vertex_endpoints), initial=0.0)
            )
            reverse_error = float(
                np.max(np.abs(curve_endpoints - vertex_endpoints[::-1]), initial=0.0)
            )
            if min(direct_error, reverse_error) > tolerance:
                raise _unsupported(
                    "A wall boundary line does not match its declared endpoint vertices."
                )
            increasing_vertices = (
                edge_vertices if direct_error <= reverse_error else edge_vertices[::-1]
            )
            sense = geometry.coedge_senses[coedge_index]
            if sense not in (-1, 1):
                raise _unsupported("A wall coedge has an invalid orientation.")
            start_vertex, end_vertex = (
                increasing_vertices if sense == 1 else increasing_vertices[::-1]
            )
            starts.append(vertex_points[start_vertex])
            ends.append(vertex_points[end_vertex])
        if len(starts) < 3 or any(
            np.max(
                np.abs(ends[index] - starts[(index + 1) % len(starts)]),
                initial=0.0,
            )
            > tolerance
            for index in range(len(starts))
        ):
            raise _unsupported("A wall face boundary loop is not a closed line cycle.")
        world = np.asarray(starts, dtype=np.float64)
        if np.max(np.abs((world - origin) @ normal), initial=0.0) > tolerance:
            raise _unsupported("A wall face boundary is not in its declared plane.")
        loops.append(
            np.column_stack(((world - origin) @ first, (world - origin) @ second))
        )
    if not loops:
        raise _unsupported("A planar wall face requires at least one boundary loop.")
    areas = tuple(_loop_area(points) for points in loops)
    area = abs(areas[0]) - sum(abs(value) for value in areas[1:])
    if not np.isfinite(area) or area <= 0.0:
        raise _unsupported("A planar wall face must have positive bounded area.")
    profile = PlanarProfile(
        ProfilePlane(
            (float(origin[0]), float(origin[1]), float(origin[2])),
            (float(first[0]), float(first[1]), float(first[2])),
            (float(second[0]), float(second[1]), float(second[2])),
        ),
        ProfileLoop.polygon(loops[0]),
        holes=tuple(ProfileLoop.polygon(points) for points in loops[1:]),
    )
    return profile, normal, offset, area


def _scope(
    model: BRepModel, dimension: int, identifiers: Sequence[int], /
) -> MeshingScope:
    revision = model.source_revision
    return MeshingScope(
        model.report.source_id,
        revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        f"{revision}:brep:{dimension}",
        np.asarray(sorted(identifiers), dtype=np.int64),
    )


class BoundaryLayerExtrusion(StrictModule, NonTrainableState):
    """In-memory slab partition and the EXACT_SWEEP control that meshes it."""

    source: BRepModel
    revision: CADRevision = eqx.field(static=True)
    association_graph: AssociationGraph = eqx.field(static=True)
    regions: tuple[BRepPartitionRegion, ...] = eqx.field(static=True)
    patches: tuple[BRepPartitionPatch, ...] = eqx.field(static=True)
    report: BRepPartitionReport = eqx.field(static=True)
    control: BoundaryLayerControl
    layer_solid_ids: tuple[int, ...] = eqx.field(static=True)
    core_solid_ids: tuple[int, ...] = eqx.field(static=True)
    slab_volumes: Array
    maximum_relative_volume_residual: float = eqx.field(static=True)
    extrusion_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        partition: BRepPartitionResult,
        control: BoundaryLayerControl,
        layer_solid_ids: tuple[int, ...],
        core_solid_ids: tuple[int, ...],
        slab_volumes: ArrayLike,
        maximum_relative_volume_residual: float,
        /,
    ) -> None:
        if not isinstance(control, BoundaryLayerControl) or (
            control.route is not BoundaryLayerRoute.EXACT_SWEEP
        ):
            raise TypeError("control must be an EXACT_SWEEP BoundaryLayerControl.")
        if partition.topological_dimension != 3:
            raise ValueError("Boundary-layer extrusion requires a 3D partition.")
        volumes = np.asarray(slab_volumes, dtype=np.float64)
        layer_ids = tuple(int(value) for value in layer_solid_ids)
        core_ids = tuple(int(value) for value in core_solid_ids)
        residual = float(maximum_relative_volume_residual)
        if volumes.shape != (len(layer_ids),) or not np.all(np.isfinite(volumes)):
            raise ValueError(
                "slab_volumes must contain one finite value per layer solid."
            )
        if np.any(volumes <= 0.0):
            raise ValueError("Every layer solid must have positive volume.")
        if (
            not layer_ids
            or tuple(sorted(layer_ids)) != layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or any(value < 0 for value in layer_ids)
        ):
            raise ValueError("layer_solid_ids must be unique, non-negative and sorted.")
        if (
            tuple(sorted(core_ids)) != core_ids
            or len(set(core_ids)) != len(core_ids)
            or any(value < 0 for value in core_ids)
        ):
            raise ValueError("core_solid_ids must be unique, non-negative and sorted.")
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError(
                "maximum_relative_volume_residual must be finite and non-negative."
            )
        model = partition.model
        wall_scope = control.wall_scope
        volume_scope = control.volume_scope
        cap_scope = control.cap_scope
        if volume_scope is None or cap_scope is None:
            raise ValueError(
                "The EXACT_SWEEP control requires wall, cap and volume scopes."
            )
        for scope, dimension in (
            (wall_scope, 2),
            (volume_scope, 3),
            (cap_scope, 2),
        ):
            if (
                scope.source_id != model.source_id
                or scope.source_revision != model.source_revision
                or scope.entity_set_id != f"{model.source_revision}:brep:{dimension}"
            ):
                raise ValueError("EXACT_SWEEP scopes must bind the partition model.")
        wall_ids = tuple(int(value) for value in np.asarray(wall_scope.entity_ids))
        cap_ids = tuple(int(value) for value in np.asarray(cap_scope.entity_ids))
        if len(wall_ids) != len(layer_ids) or len(cap_ids) != len(layer_ids):
            raise ValueError("Wall, cap and layer solid scopes must align one-to-one.")
        if (
            tuple(int(value) for value in np.asarray(volume_scope.entity_ids))
            != layer_ids
        ):
            raise ValueError("The EXACT_SWEEP volume scope must match layer_solid_ids.")
        if any(value >= model.topology.num_solids for value in (*layer_ids, *core_ids)):
            raise ValueError(
                "Layer and core solid IDs must belong to the partition model."
            )
        if any(value >= model.topology.num_faces for value in (*wall_ids, *cap_ids)):
            raise ValueError("Wall and cap face IDs must belong to the partition model.")
        self.source = model
        self.revision = partition.revision
        self.association_graph = partition.association_graph
        self.regions = partition.regions
        self.patches = partition.patches
        self.report = partition.report
        self.control = control
        self.layer_solid_ids = layer_ids
        self.core_solid_ids = core_ids
        self.slab_volumes = jnp.asarray(volumes)
        self.maximum_relative_volume_residual = residual
        self.extrusion_id = canonical_fingerprint(
            {
                "kind": "boundary-layer-extrusion",
                "partition_model": model.model_id,
                "partition_revision": partition.revision.revision_id,
                "association_graph": partition.association_graph.graph_id,
                "control": control.control_id,
                "layer_solids": self.layer_solid_ids,
                "core_solids": self.core_solid_ids,
                "volumes": array_tree_fingerprint(volumes),
            }
        )


def _validated_request(
    model: BRepModel, control: BoundaryLayerControl, /
) -> tuple[tuple[int, ...], tuple[int, ...], MeshingScope]:
    if not isinstance(model, BRepModel):
        raise TypeError("source must be BRepModel.")
    if not isinstance(control, BoundaryLayerControl):
        raise TypeError("control must be BoundaryLayerControl.")
    if control.route is not BoundaryLayerRoute.CAD_EXTRUSION:
        raise ValueError(
            "prepare_boundary_layer_extrusion realizes CAD_EXTRUSION controls."
        )
    volume_scope = control.volume_scope
    if volume_scope is None:
        raise ValueError(
            "CAD_EXTRUSION requires a wall face scope, a volume scope, and no cap."
        )
    revision = model.source_revision
    for scope, dimension in ((control.wall_scope, 2), (volume_scope, 3)):
        if (
            scope.source_id != model.report.source_id
            or scope.source_revision != revision
            or scope.entity_set_id != f"{revision}:brep:{dimension}"
        ):
            raise ValueError("The control scopes must bind this BRep source revision.")
    walls = tuple(int(value) for value in np.asarray(control.wall_scope.entity_ids))
    solids = {int(value) for value in np.asarray(volume_scope.entity_ids)}
    if not walls:
        raise ValueError("CAD_EXTRUSION requires at least one wall face.")
    if not solids:
        raise ValueError("CAD_EXTRUSION requires at least one controlled solid.")
    if any(wall < 0 or wall >= model.topology.num_faces for wall in walls):
        raise ValueError("The wall scope selects an absent B-Rep face.")
    if any(solid < 0 or solid >= model.topology.num_solids for solid in solids):
        raise ValueError("The volume scope selects an absent B-Rep solid.")
    owners: list[int] = []
    for wall in walls:
        incident = set(model.topology.face_solids[wall]) & solids
        if len(incident) != 1 or len(model.topology.face_solids[wall]) != 1:
            raise _unsupported(
                "Every extruded wall must be a boundary face of one controlled solid."
            )
        owners.append(incident.pop())
    return walls, tuple(owners), volume_scope


def _solid_volume(model: BRepModel, solid_index: int, /) -> float:
    geometry = model.geometry
    vertices = np.asarray(geometry.vertex_points, dtype=np.float64)
    face_indices = model.topology.solid_faces[solid_index]
    face_orientations = model.topology.solid_face_orientations[solid_index]
    vertex_indices = {
        int(vertex)
        for face_index in face_indices
        for loop in geometry.face_loops[face_index]
        for coedge in loop
        for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
    }
    reference = np.mean(
        vertices[np.asarray(sorted(vertex_indices), dtype=np.int64)], axis=0
    )
    signed_volume = 0.0
    for face_index, shell_orientation in zip(
        face_indices, face_orientations, strict=True
    ):
        patch = model.patches[face_index]
        if not isinstance(patch, PlanePatch):
            raise _unsupported(
                "Exact slab volume identity requires planar extrusion faces."
            )
        polygons: list[np.ndarray] = []
        for loop in geometry.face_loops[face_index]:
            points = []
            for coedge in loop:
                edge_index = geometry.coedge_edges[coedge]
                curve_index = geometry.edge_curves[edge_index]
                if curve_index < 0 or not isinstance(
                    geometry.curves[curve_index], (LineCurve, ExactLineCurve)
                ):
                    raise _unsupported(
                        "Exact slab volume identity requires line-bounded faces."
                    )
                first_vertex, second_vertex = geometry.edge_vertices[edge_index]
                if geometry.coedge_senses[coedge] < 0:
                    first_vertex, second_vertex = second_vertex, first_vertex
                points.append(vertices[first_vertex])
            if len(points) < 3:
                raise _unsupported("A slab boundary face has a degenerate loop.")
            polygons.append(np.asarray(points, dtype=np.float64))
        first_axis = np.asarray(patch.first_axis, dtype=np.float64)
        second_axis = np.asarray(patch.second_axis, dtype=np.float64)
        surface_normal = np.cross(first_axis, second_axis)
        normal_length = float(np.linalg.norm(surface_normal))
        if normal_length == 0.0:
            raise _unsupported("A slab boundary face has a degenerate plane frame.")
        surface_normal /= normal_length
        outer = polygons[0] - reference
        area_vector = np.sum(np.cross(outer, np.roll(outer, -1, axis=0)), axis=0)
        alignment = (
            float(area_vector @ surface_normal)
            * float(np.asarray(model.orientation)[face_index])
            * int(shell_orientation)
        )
        if alignment == 0.0:
            raise _unsupported("A slab boundary loop has unresolved orientation.")
        face_sign = 1 if alignment > 0.0 else -1
        for polygon in polygons:
            origin = polygon[0] - reference
            for index in range(1, polygon.shape[0] - 1):
                signed_volume += (
                    face_sign
                    * float(
                        origin
                        @ np.cross(
                            polygon[index] - reference,
                            polygon[index + 1] - reference,
                        )
                    )
                    / 6.0
                )
    volume = abs(signed_volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise _unsupported("A published layer slab must have positive volume.")
    return volume


def _solid_selection(model: BRepModel, solid_index: int, /) -> CADSelectionSet:
    """Select the solid's unique source occurrence by its exact entity identity."""
    revision = cad_revision_from_brep_model(model)
    entity_ids = {
        brep_entity_id(
            model.source_id,
            model.source_revision,
            3,
            solid_index,
            occurrence_path=occurrence.path,
        )
        for occurrence in model.geometry.iter_occurrences()
        if occurrence.solid_index == solid_index
    }
    matches = tuple(
        occurrence.occurrence_id
        for occurrence in revision.occurrences
        if occurrence.kind == "solid" and occurrence.entity_id in entity_ids
    )
    if len(entity_ids) != 1 or len(matches) != 1:
        raise _unsupported(
            "A controlled solid must resolve to exactly one CAD source occurrence."
        )
    return CADSelectionSet.from_revision(revision, matches)


def prepare_boundary_layer_extrusion(
    source: BRepModel,
    control: BoundaryLayerControl,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> BoundaryLayerExtrusion:
    """Build exact planar wall slabs and partition their source in memory.

    The returned source is the native split BRep; mesh it with
    ``result.control`` and region controls naming ``layer_solid_ids`` and
    ``core_solid_ids``.
    """
    walls, owners, volume_scope = _validated_request(source, control)
    if tessellation is None:
        source_tessellation = source.tessellation
        tessellation = (
            BRepTessellationPolicy()
            if source_tessellation is None
            else BRepTessellationPolicy(
                linear_deflection=source_tessellation.linear_deflection,
                angular_deflection=source_tessellation.angular_deflection,
            )
        )
    elif not isinstance(tessellation, BRepTessellationPolicy):
        raise TypeError("tessellation must be a BRepTessellationPolicy or None.")

    total = control.schedule.total_thickness
    layer_ids: list[str] = []
    layer_models: list[BRepModel] = []
    layer_planes: list[tuple[np.ndarray, float, float]] = []
    for wall in walls:
        profile, normal, offset, area = _profile_from_face(source, wall)
        inward = -normal * total
        layer_id = f"layer:{wall}"
        layer_model = brep_extrusion(
            profile,
            tuple(float(value) for value in inward),
            coordinate_contract=source.coordinate_contract,
            tessellation=tessellation,
            source_id=(
                f"boundary-layer:{source.model_id}:{control.control_id}:wall:{wall}"
            ),
        )
        if layer_model.topology.num_solids != 1:
            raise _unsupported("A wall extrusion must construct exactly one solid.")
        layer_ids.append(layer_id)
        layer_models.append(layer_model)
        layer_planes.append((normal, offset, area))

    # A solid whose wall slabs exactly fill it keeps no core region: an empty
    # declared region is not a partition region. Coverage is re-proved below.
    slab_volumes_by_owner: dict[int, float] = {}
    for owner, (_, _, area) in zip(owners, layer_planes, strict=True):
        slab_volumes_by_owner[owner] = (
            slab_volumes_by_owner.get(owner, 0.0) + area * total
        )
    covered = set()
    for owner, slab_volume in slab_volumes_by_owner.items():
        source_volume = _solid_volume(source, owner)
        if abs(source_volume - slab_volume) <= 1.0e-9 * source_volume:
            covered.add(owner)
    core_ids = tuple(
        f"core:{solid_index}"
        for solid_index in range(source.topology.num_solids)
        if solid_index not in covered
    )
    partition_operands = [
        BRepPartitionOperand(
            layer_id,
            model,
            BRepPartitionRole.REGION,
        )
        for layer_id, model in zip(layer_ids, layer_models, strict=True)
    ]
    for core_id in core_ids:
        partition_operands.append(
            BRepPartitionOperand(
                core_id,
                source,
                BRepPartitionRole.REGION,
                selection=_solid_selection(source, int(core_id.split(":")[1])),
            )
        )
    partition = partition_brep(
        BRepPartitionPlan(
            source.coordinate_contract,
            tuple(partition_operands),
            BRepPartitionPolicy((*layer_ids, *core_ids)),
        ),
        tessellation=tessellation,
    )

    if partition.topological_dimension != 3:
        raise _unsupported("A wall extrusion must produce a 3D partition.")
    partition_regions = {region.name: region for region in partition.regions}
    region_by_solid = {
        entity.index: region.name
        for region in partition.regions
        for entity in region.entity_ids
    }
    scale = max(
        1.0,
        float(
            np.max(
                np.abs(np.asarray(source.geometry.vertex_points, dtype=np.float64)),
                initial=0.0,
            )
        ),
    )
    plane_tolerance = 1024.0 * np.finfo(np.float64).eps * scale
    slabs: list[int] = []
    bottoms: list[int] = []
    tops: list[int] = []
    volumes: list[float] = []
    residuals: list[float] = []
    slab_owners: list[int] = []
    for wall, owner, layer_id, (normal, offset, area) in zip(
        walls, owners, layer_ids, layer_planes, strict=True
    ):
        layer_region = partition_regions.get(layer_id)
        if layer_region is None:
            raise _unsupported("The partition lost a declared wall-layer region.")
        if len(layer_region.entity_ids) != 1:
            raise _unsupported(
                "Each extruded wall must remain one exact layer solid after partition."
            )
        slab = layer_region.entity_ids[0].index
        slab_faces = partition.model.topology.solid_faces[slab]
        bottom_faces: list[int] = []
        cap_faces: list[int] = []
        for face_index in slab_faces:
            face_plane = _plane(partition.model, face_index)
            if face_plane is None:
                continue
            face_normal, face_offset = face_plane
            alignment = float(normal @ face_normal)
            if abs(abs(alignment) - 1.0) > 1.0e-12:
                continue
            level = alignment * face_offset
            if abs(level - offset) <= plane_tolerance:
                bottom_faces.append(face_index)
            elif abs(level - (offset - total)) <= plane_tolerance:
                cap_faces.append(face_index)
        if len(bottom_faces) != 1 or len(cap_faces) != 1:
            raise _unsupported(
                "A wall slab must retain one exact wall face and one exact cap face."
            )
        bottom, cap = bottom_faces[0], cap_faces[0]
        bottom_regions = tuple(
            region_by_solid[index]
            for index in partition.model.topology.face_solids[bottom]
        )
        cap_regions = tuple(
            region_by_solid[index] for index in partition.model.topology.face_solids[cap]
        )
        if bottom_regions != (layer_id,):
            raise _unsupported("The extruded wall face lost its exact layer identity.")
        expected_cap_regions = {layer_id}
        if f"core:{owner}" in partition_regions:
            expected_cap_regions.add(f"core:{owner}")
        if set(cap_regions) != expected_cap_regions:
            raise _unsupported(
                "The extruded cap must be shared only by its layer and any remaining source core."
            )
        lateral_faces = set(slab_faces) - {bottom, cap}
        if any(
            len(partition.model.topology.face_solids[face]) != 1 for face in lateral_faces
        ):
            raise _unsupported(
                "Extruded slab lateral faces must lie on the source boundary."
            )
        expected_volume = area * total
        volume = _solid_volume(partition.model, slab)
        residual = abs(volume - expected_volume) / expected_volume
        if residual > 1.0e-9:
            raise _unsupported("The partition changed the declared wall-slab volume.")
        slabs.append(slab)
        bottoms.append(bottom)
        tops.append(cap)
        volumes.append(volume)
        slab_owners.append(owner)
        residuals.append(residual)

    controlled_cores = {
        solid_id for solid_id in np.asarray(volume_scope.entity_ids, dtype=np.int64)
    }
    controlled_core_names = {f"core:{solid_id}" for solid_id in controlled_cores}
    for owner in controlled_cores:
        core_name = f"core:{owner}"
        if core_name in partition_regions:
            continue
        if owner not in slab_owners:
            raise _unsupported(
                "The wall extrusion leaves no core in a controlled solid without an authored slab."
            )
        source_volume = _solid_volume(source, owner)
        owner_slab_volume = sum(
            volume
            for slab_owner, volume in zip(slab_owners, volumes, strict=True)
            if slab_owner == owner
        )
        if abs(source_volume - owner_slab_volume) / source_volume > 1.0e-9:
            raise _unsupported(
                "A controlled solid without a remaining core must be completely covered by its wall slabs."
            )
    core = tuple(
        entity.index
        for region in partition.regions
        if region.name in controlled_core_names
        for entity in region.entity_ids
    )
    sweep = BoundaryLayerControl(
        _scope(partition.model, 2, bottoms),
        control.schedule,
        route=BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=_scope(partition.model, 3, slabs),
        cap_scope=_scope(partition.model, 2, tops),
        corner=control.corner,
        feature_angle=control.feature_angle,
        minimum_thickness_fraction=control.minimum_thickness_fraction,
        growth_rate_bounds=control.growth_rate_bounds,
        maximum_corner_stretch=control.maximum_corner_stretch,
        smoothing_iterations=control.smoothing_iterations,
        core_maximum_size=control.core_maximum_size,
    )
    slab_volume_pairs = tuple(sorted(zip(slabs, volumes, strict=True)))
    ordered_slab_ids = tuple(slab for slab, _ in slab_volume_pairs)
    ordered_volumes = np.asarray(
        tuple(volume for _, volume in slab_volume_pairs), dtype=np.float64
    )
    return BoundaryLayerExtrusion(
        partition,
        sweep,
        ordered_slab_ids,
        tuple(sorted(core)),
        ordered_volumes,
        max(residuals),
    )
