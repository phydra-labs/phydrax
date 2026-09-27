#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh layered generation: planar bands, straight sweeps, and boundary-layer fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...discretization import CellMesh
from ...discretization._cell_complex import PolyhedralConnectivity
from ...geometry.brep import BRepModel, PlanarEmbedding
from .._boundary_layer import _nearest_distances
from .._contracts import MeshingFailure, MeshingFailureCategory, SurfaceMeshingSpec
from .._controls import (
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerRoute,
)
from .._organization import MeshAttribute, MeshAttributeRole
from .._planar_bands import PlanarBandResult
from .._quality import evaluate_swept_layer_quality
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import MeshingStageKind
from ._gmsh_elements import _ElementRows, _local_connectivity
from ._gmsh_evidence import _connectivity_face_incidents, _connectivity_face_rows
from ._gmsh_import import (
    _brep_model,
    _CadEntityMap,
    _entity_scope,
    _resolve_entities,
    _source_scale,
)


@dataclass(frozen=True, slots=True)
class _PlanarBandFront:
    layer: object
    curves: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PlanarBandGeneration:
    embedding: PlanarEmbedding
    fronts: tuple[_PlanarBandFront, ...]


def _straight_surface_curves(
    gmsh: Any,
    surface: int,
    embedding: PlanarEmbedding,
    /,
) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    boundary = gmsh.model.getBoundary(
        [(2, surface)], combined=False, oriented=False, recursive=False
    )
    curves = tuple(abs(int(tag)) for dimension, tag in boundary if dimension == 1)
    if len(curves) != 4 or len(curves) != len(boundary):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Pure quadrilateral planar bands require every closure face to have four curves.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    directions: dict[int, np.ndarray] = {}
    for curve in curves:
        if str(gmsh.model.getType(1, curve)).lower() != "line":
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Pure quadrilateral planar bands require straight closure curves.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        vertices = gmsh.model.getBoundary(
            [(1, curve)], combined=False, oriented=False, recursive=False
        )
        if len(vertices) != 2 or any(dimension != 0 for dimension, _ in vertices):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band boundary curve is not one straight segment.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        coordinates = np.asarray(
            [gmsh.model.getValue(0, abs(int(tag)), []) for _, tag in vertices],
            dtype=np.float64,
        )
        planar = embedding.to_planar(coordinates)
        direction = planar[1] - planar[0]
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band closure curve has zero length.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        directions[curve] = direction / length
    return curves, directions


def _configure_full_quad_band_closure(
    gmsh: Any,
    embedding: PlanarEmbedding,
    constrained_curves: dict[int, int],
    target_size: float | None,
    /,
) -> None:
    """Propagate exact band node counts across rectangular Q4 closure faces."""

    surfaces = tuple(
        int(tag) for dimension, tag in gmsh.model.getEntities(2) if dimension == 2
    )
    boundaries = {
        surface: _straight_surface_curves(gmsh, surface, embedding)
        for surface in surfaces
    }
    parallel_pairs: dict[int, tuple[tuple[int, int], ...]] = {}
    for surface, (curves, directions) in boundaries.items():
        remaining = set(curves)
        pairs: list[tuple[int, int]] = []
        while remaining:
            first = min(remaining)
            matches = tuple(
                second
                for second in remaining - {first}
                if abs(float(np.dot(directions[first], directions[second])))
                >= 1.0 - 1.0e-10
            )
            if len(matches) != 1:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Pure quadrilateral planar bands require rectangular closure faces.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            second = matches[0]
            remaining.remove(first)
            remaining.remove(second)
            pairs.append((first, second))
        parallel_pairs[surface] = tuple(pairs)
    neighbors: dict[int, set[int]] = {
        curve: set() for curves, _ in boundaries.values() for curve in curves
    }
    # ty: ignore[invalid-assignment]
    for pairs in parallel_pairs.values():
        for first, second in pairs:
            neighbors[first].add(second)
            neighbors[second].add(first)
    unresolved = set(neighbors)
    while unresolved:
        root = min(unresolved)
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for neighbor in neighbors[current] - component:
                component.add(neighbor)
                frontier.append(neighbor)
        unresolved -= component
        prescribed = {
            constrained_curves[curve]
            for curve in component
            if curve in constrained_curves
        }
        if len(prescribed) > 1:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Pure quadrilateral planar band closure has incompatible opposite curve counts.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        if prescribed:
            count = next(iter(prescribed))
        elif target_size is None:
            count = 2
        else:
            lengths = []
            for curve in component:
                vertices = gmsh.model.getBoundary(
                    [(1, curve)], combined=False, oriented=False, recursive=False
                )
                coordinates = np.asarray(
                    [gmsh.model.getValue(0, abs(int(tag)), []) for _, tag in vertices],
                    dtype=np.float64,
                )
                planar = embedding.to_planar(coordinates)
                lengths.append(float(np.linalg.norm(planar[1] - planar[0])))
            count = max(2, int(np.ceil(max(lengths) / target_size)) + 1)
        for curve in component:
            constrained_curves[curve] = count
    for surface, (curves, _) in boundaries.items():
        for curve in curves:
            gmsh.model.mesh.setTransfiniteCurve(curve, constrained_curves[curve])
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(2, surface)


def _apply_planar_band_constraints(
    gmsh: Any,
    bands: PlanarBandResult | None,
    cad_entities: _CadEntityMap | None,
    requested_kinds: set[str],
    target_size: float | None,
    /,
) -> _PlanarBandGeneration | None:
    if bands is None:
        return None
    if cad_entities is None or not cad_entities.edge_to_curve:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Planar bands require resolved source face and edge identities.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    constrained_curves: dict[int, int] = {}
    fronts = []
    for layer in bands.layer_partitions:
        if len(layer.face_entity_ids) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Each exact planar band layer must be one strip face.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        surface = cad_entities.face_to_surface[layer.face_entity_ids[0].index]
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        curves = tuple(abs(int(tag)) for dimension, tag in boundary if dimension == 1)
        if len(curves) != 4 or len(curves) != len(boundary):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band layer is not an exact four-curve straight strip.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        tangential_nodes = max(
            2, int(np.ceil(layer.source_length / layer.tangential_target)) + 1
        )
        tangent = np.asarray(layer.tangent, dtype=np.float64)
        tangential_count = 0
        normal_count = 0
        for curve in curves:
            points = gmsh.model.getBoundary(
                [(1, curve)], combined=False, oriented=False, recursive=False
            )
            if len(points) != 2 or any(dimension != 0 for dimension, _ in points):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "A planar band boundary curve is not one straight segment.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            coordinates = np.asarray(
                [gmsh.model.getValue(0, abs(int(tag)), []) for _, tag in points],
                dtype=np.float64,
            )
            planar = bands.embedding.to_planar(coordinates)
            direction = planar[1] - planar[0]
            direction /= np.linalg.norm(direction)
            is_tangential = abs(float(np.dot(direction, tangent))) >= 1.0 - 1.0e-10
            node_count = tangential_nodes if is_tangential else 2
            previous = constrained_curves.setdefault(curve, node_count)
            if previous != node_count:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Intersecting planar band constraints require incompatible curve nodes.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            gmsh.model.mesh.setTransfiniteCurve(curve, node_count)
            tangential_count += int(is_tangential)
            normal_count += int(not is_tangential)
        if tangential_count != 2 or normal_count != 2:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "A planar band strip lacks two tangential and two normal curves.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        gmsh.model.mesh.setTransfiniteSurface(surface)
        if "quadrilateral" in requested_kinds:
            gmsh.model.mesh.setRecombine(2, surface)
        front_patch = bands.partition.patch(layer.front_patch_name)
        front_curves = tuple(
            cad_entities.edge_to_curve[entity.index] for entity in front_patch.entity_ids
        )
        if not front_curves:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An exact planar band front resolved to no source curves.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        fronts.append(_PlanarBandFront(layer, front_curves))
    if requested_kinds == {"quadrilateral"}:
        _configure_full_quad_band_closure(
            gmsh,
            bands.embedding,
            constrained_curves,
            target_size,
        )
    return _PlanarBandGeneration(bands.embedding, tuple(fronts))


def _audit_planar_band_fronts(
    gmsh: Any, generation: _PlanarBandGeneration | None, /
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    if generation is None:
        return (), ()
    requested = []
    achieved = []
    for record in generation.fronts:
        layer = record.layer
        coordinates = []
        for curve in record.curves:
            _, values, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
            coordinates.append(np.asarray(values, dtype=np.float64).reshape((-1, 3)))
        points = np.concatenate(coordinates)
        if not points.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "An exact planar band front contains no mesh nodes.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        planar = generation.embedding.to_planar(points)
        # ty: ignore[unresolved-attribute]
        origin = np.asarray(layer.source_origin, dtype=np.float64)
        # ty: ignore[unresolved-attribute]
        inward = np.asarray(layer.inward_normal, dtype=np.float64)
        distances = (planar - origin) @ inward
        residual = float(
            # ty: ignore[unresolved-attribute]
            np.max(np.abs(distances - layer.cumulative_distance), initial=0.0)
        )
        # ty: ignore[unresolved-attribute]
        tangent = np.asarray(layer.tangent, dtype=np.float64)
        tangential_positions = np.unique((planar - origin) @ tangent)
        if tangential_positions.size < 2:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "An exact planar band front has fewer than two tangential nodes.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        maximum_tangential_spacing = float(np.max(np.diff(tangential_positions)))
        scale = max(
            1.0,
            # ty: ignore[unresolved-attribute]
            layer.source_length,
            # ty: ignore[unresolved-attribute]
            layer.cumulative_distance,
            float(np.max(np.abs(planar), initial=0.0)),
        )
        tolerance = 8192.0 * np.finfo(np.float64).eps * scale
        if residual > tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Generated planar band nodes do not lie on the exact requested front.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        # ty: ignore[unresolved-attribute]
        if maximum_tangential_spacing > layer.tangential_target + tolerance:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Generated planar band tangential spacing exceeds its target.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        # ty: ignore[unresolved-attribute]
        key = f"planar_band:{layer.control_id}:{layer.region_name}:front:{layer.layer_index + 1}"
        requested.extend(
            (
                # ty: ignore[unresolved-attribute]
                (f"{key}:distance", layer.cumulative_distance),
                # ty: ignore[unresolved-attribute]
                (f"{key}:tangential_target", layer.tangential_target),
            )
        )
        achieved.extend(
            (
                (f"{key}:distance", float(np.mean(distances))),
                (f"{key}:maximum_residual", residual),
                (
                    f"{key}:maximum_tangential_spacing",
                    maximum_tangential_spacing,
                ),
            )
        )
    return tuple(requested), tuple(achieved)


@dataclass(frozen=True, slots=True)
class _SweptVolume:
    control_index: int
    control: BoundaryLayerControl
    solid_index: int
    volume_tag: int
    source_face_index: int
    target_face_index: int
    source_surface: int
    target_surface: int
    lateral_face_indices: tuple[int, ...]
    lateral_surfaces: tuple[int, ...]
    origin: np.ndarray
    direction: np.ndarray
    unit: np.ndarray
    levels: np.ndarray
    relative_cad_difference: float


@dataclass(frozen=True, slots=True)
class _SweepGeneration:
    volumes: tuple[_SweptVolume, ...]


@dataclass(frozen=True, slots=True)
class _LayerAudit:
    requested: tuple[tuple[str, float], ...]
    achieved: tuple[tuple[str, float], ...]
    control_by_element_tag: dict[int, int]
    layer_by_element_tag: dict[int, int]


def _cad_symmetric_difference(
    gmsh: Any, dimension: int, left: Any, right: Any, /
) -> float:
    baseline = set(gmsh.model.getEntities())
    measure = 0.0
    for first, second in ((left, right), (right, left)):
        first_copy = gmsh.model.occ.copy([first])
        second_copy = gmsh.model.occ.copy([second])
        difference, _ = gmsh.model.occ.cut(first_copy, second_copy)
        measure += sum(
            gmsh.model.occ.getMass(dim, tag)
            for dim, tag in difference
            if dim == dimension
        )
    additions = sorted(set(gmsh.model.getEntities()) - baseline, reverse=True)
    if additions:
        gmsh.model.occ.remove(additions, recursive=True)
        gmsh.model.occ.synchronize()
    return float(measure)


def _certify_swept_volume(
    gmsh: Any,
    volume: tuple[int, int],
    source_surface: int,
    target_surface: int,
    direction: np.ndarray,
    /,
) -> tuple[float, float]:
    baseline = set(gmsh.model.getEntities())
    translated = gmsh.model.occ.copy([(2, source_surface)])
    gmsh.model.occ.translate(translated, *direction.tolist())
    extruded = gmsh.model.occ.extrude(
        gmsh.model.occ.copy([(2, source_surface)]),
        *direction.tolist(),
    )
    gmsh.model.occ.synchronize()
    generated_volumes = tuple(entity for entity in extruded if entity[0] == 3)
    if len(translated) != 1 or len(generated_volumes) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Swept-layer CAD certification did not produce one translated cap and volume.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    face_difference = _cad_symmetric_difference(
        gmsh, 2, translated[0], (2, target_surface)
    )
    volume_difference = _cad_symmetric_difference(gmsh, 3, generated_volumes[0], volume)
    additions = sorted(set(gmsh.model.getEntities()) - baseline, reverse=True)
    if additions:
        gmsh.model.occ.remove(additions, recursive=True)
        gmsh.model.occ.synchronize()
    return face_difference, volume_difference


def _prepare_swept_geometry(
    gmsh: Any, plan: Any, shape: Any, cad_entities: Any, /
) -> Any:
    controls = plan.specification.layer_controls
    if not controls or any(
        control.route is not BoundaryLayerRoute.EXACT_SWEEP for control in controls
    ):
        return None
    source = _brep_model(plan.source)
    scale = _source_scale(source)
    if cad_entities is None:
        volume_entities = tuple(gmsh.model.getEntities(3))
        if source.topology.num_solids != 1 or len(volume_entities) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Non-semantic swept meshing requires one source solid.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        face_to_surface = tuple(
            _resolve_entities(gmsh, source, shape, _entity_scope(source, 2, (face,)))[0]
            for face in range(source.report.num_faces)
        )
        solid_to_volume = (int(volume_entities[0][1]),)
    else:
        face_to_surface = cad_entities.face_to_surface
        solid_to_volume = cad_entities.solid_to_volume
    prepared = []
    for control_index, control in enumerate(controls):
        solid_ids = np.asarray(control.volume_scope.entity_ids, dtype=np.int64)
        source_faces = {int(value) for value in np.asarray(control.wall_scope.entity_ids)}
        target_faces = {int(value) for value in np.asarray(control.cap_scope.entity_ids)}
        thicknesses = np.asarray(control.schedule.thicknesses, dtype=np.float64)
        levels = np.concatenate(([0.0], np.cumsum(thicknesses)))
        for solid_value in solid_ids:
            solid = int(solid_value)
            source_candidates = source_faces & set(source.topology.solid_faces[solid])
            target_candidates = target_faces & set(source.topology.solid_faces[solid])
            if len(source_candidates) != 1 or len(target_candidates) != 1:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SPECIFICATION,
                    "Every controlled solid requires one exact source and target cap.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            source_face = source_candidates.pop()
            target_face = target_candidates.pop()
            source_surface = int(face_to_surface[source_face])
            target_surface = int(face_to_surface[target_face])
            if (
                source_surface == target_surface
                or gmsh.model.getType(2, source_surface) != "Plane"
                or gmsh.model.getType(2, target_surface) != "Plane"
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Swept layers require distinct planar source and target faces.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            origin = np.asarray(
                gmsh.model.occ.getCenterOfMass(2, source_surface), dtype=np.float64
            )
            target_center = np.asarray(
                gmsh.model.occ.getCenterOfMass(2, target_surface), dtype=np.float64
            )
            direction = target_center - origin
            length = float(np.linalg.norm(direction))
            tolerance = 1.0e-9 * max(scale, control.schedule.total_thickness)
            if (
                not np.isfinite(length)
                or length <= 0.0
                or abs(length - control.schedule.total_thickness) > tolerance
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SPECIFICATION,
                    "Swept-layer schedule total thickness does not equal the source-to-target translation.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            unit = direction / length
            _, parameters = gmsh.model.getClosestPoint(2, source_surface, origin.tolist())
            normal = np.asarray(
                gmsh.model.getNormal(source_surface, parameters), dtype=np.float64
            ).reshape(3)
            normal /= np.linalg.norm(normal)
            if abs(float(np.dot(normal, unit))) < 1.0 - 1.0e-10:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Swept-layer translation must be normal to its planar source face.",
                    stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                )
            volume_tag = int(solid_to_volume[solid])
            face_difference, volume_difference = _certify_swept_volume(
                gmsh,
                (3, volume_tag),
                source_surface,
                target_surface,
                direction,
            )
            face_area = float(gmsh.model.occ.getMass(2, source_surface))
            volume_measure = float(gmsh.model.occ.getMass(3, volume_tag))
            if face_difference > 1.0e-9 * max(
                face_area, scale**2
            ) or volume_difference > 1.0e-9 * max(volume_measure, scale**3):
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "Controlled volume is not the exact straight extrusion from source cap to target cap.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                )
            lateral_faces = tuple(
                int(face)
                for face in source.topology.solid_faces[solid]
                if face not in (source_face, target_face)
            )
            prepared.append(
                _SweptVolume(
                    control_index,
                    control,
                    solid,
                    volume_tag,
                    source_face,
                    target_face,
                    source_surface,
                    target_surface,
                    lateral_faces,
                    tuple(int(face_to_surface[face]) for face in lateral_faces),
                    origin,
                    direction,
                    unit,
                    levels,
                    # ty: ignore[invalid-argument-type]
                    volume_difference / max(volume_measure, np.finfo(np.float64).tiny),
                )
            )
    by_solid = {value.solid_index: value for value in prepared}
    for value in prepared:
        for face in value.lateral_face_indices:
            adjacent = tuple(
                by_solid[owner]
                for owner in source.topology.face_solids[face]
                if owner != value.solid_index and owner in by_solid
            )
            for other in adjacent:
                if not np.allclose(
                    value.direction,
                    other.direction,
                    rtol=0.0,
                    atol=1.0e-9 * scale,
                ):
                    raise MeshingFailure(
                        MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                        "A swept cohort has inconsistent lateral translation vectors.",
                        stage=MeshingStageKind.CONTROL_RESOLUTION.value,
                    )
    for value in prepared:
        transform = np.eye(4)
        transform[:3, 3] = value.direction
        gmsh.model.mesh.setPeriodic(
            2,
            [value.target_surface],
            [value.source_surface],
            transform.reshape(-1).tolist(),
        )
    return _SweepGeneration(tuple(prepared))


def _entity_linear_triangles(gmsh: Any, surface: int, /) -> np.ndarray:
    blocks = []
    element_types, _, node_blocks = gmsh.model.mesh.getElements(2, surface)
    for element_type, node_values in zip(element_types, node_blocks, strict=True):
        name, _, order, count, _, corners = gmsh.model.mesh.getElementProperties(
            int(element_type)
        )
        if name.split()[0] != "Triangle" or int(order) != 1 or int(corners) != 3:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Swept source caps require complete linear triangle meshes.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        blocks.append(np.asarray(node_values, dtype=np.int64).reshape((-1, int(count))))
    if not blocks:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Swept source cap generated no triangles.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    return np.concatenate(blocks)


def _matching_lateral_surface(
    gmsh: Any, candidates: tuple[int, ...], point: np.ndarray, tolerance: float, /
) -> int:
    matches = []
    for surface in candidates:
        closest, _ = gmsh.model.getClosestPoint(2, surface, point.tolist())
        closest_point = np.asarray(closest, dtype=np.float64).reshape((-1, 3))
        if (
            closest_point.shape == (1, 3)
            and np.linalg.norm(closest_point[0] - point) <= tolerance
            and gmsh.model.isInside(2, surface, point.tolist()) == 1
        ):
            matches.append(surface)
    if len(matches) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "An extruded source edge does not map uniquely to one lateral CAD surface.",
            stage=MeshingStageKind.LAYER_GENERATION.value,
        )
    return matches[0]


def _install_swept_cells(gmsh: Any, sweep: _SweepGeneration, /) -> None:
    for value in sweep.volumes:
        gmsh.model.mesh.removeElements(3, value.volume_tag)
    for surface in sorted(
        {surface for value in sweep.volumes for surface in value.lateral_surfaces}
    ):
        gmsh.model.mesh.removeElements(2, surface)
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coordinates = np.asarray(node_coordinates, dtype=np.float64).reshape((-1, 3))
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    node_coordinates = node_coordinates[order]
    coordinates = {
        int(tag): point for tag, point in zip(node_tags, node_coordinates, strict=True)
    }
    scale = max(float(np.ptp(node_coordinates, axis=0).max()), 1.0)
    tolerance = 1.0e-9 * scale
    coordinate_nodes = {
        tuple(np.rint(point / tolerance).astype(np.int64)): int(tag)
        for tag, point in zip(node_tags, node_coordinates, strict=True)
    }
    next_node_tag = int(np.max(node_tags, initial=0)) + 1
    new_nodes: dict[int, list[tuple[int, np.ndarray]]] = {}
    prism_blocks: dict[int, list[np.ndarray]] = {}
    quad_blocks: dict[int, dict[tuple[int, ...], np.ndarray]] = {}
    for value in sweep.volumes:
        triangles = _entity_linear_triangles(gmsh, value.source_surface)
        actual_master, slave_nodes, master_nodes, affine = (
            gmsh.model.mesh.getPeriodicNodes(
                2, value.target_surface, includeHighOrderNodes=True
            )
        )
        transform = np.eye(4)
        transform[:3, 3] = value.direction
        if int(actual_master) != value.source_surface or not np.allclose(
            np.asarray(affine, dtype=np.float64).reshape((4, 4)),
            transform,
            rtol=0.0,
            atol=tolerance,
        ):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Gmsh did not preserve the exact source-to-target cap translation.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        target_by_source = {
            int(master): int(slave)
            for slave, master in zip(slave_nodes, master_nodes, strict=True)
        }
        source_nodes = np.unique(triangles)
        if any(int(node) not in target_by_source for node in source_nodes):
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Target cap is not a complete translated copy of the source triangle mesh.",
                stage=MeshingStageKind.LAYER_GENERATION.value,
            )
        level_nodes: dict[int, tuple[int, ...]] = {}
        for source_node_value in source_nodes:
            source_node = int(source_node_value)
            route = [source_node]
            for distance in value.levels[1:-1]:
                point = coordinates[source_node] + value.unit * float(distance)
                key = tuple(np.rint(point / tolerance).astype(np.int64))
                node = coordinate_nodes.get(key)
                if node is None:
                    node = next_node_tag
                    next_node_tag += 1
                    coordinate_nodes[key] = node
                    coordinates[node] = point
                    new_nodes.setdefault(value.volume_tag, []).append((node, point))
                route.append(node)
            route.append(target_by_source[source_node])
            level_nodes[source_node] = tuple(route)
        edges = np.concatenate(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            )
        )
        edge_keys, edge_counts = np.unique(
            np.sort(edges, axis=1), axis=0, return_counts=True
        )
        boundary_edges = edge_keys[edge_counts == 1]
        for triangle in triangles:
            oriented = np.asarray(triangle, dtype=np.int64).copy()
            first, second, third = (coordinates[int(node)] for node in oriented)
            if np.dot(np.cross(second - first, third - first), value.direction) < 0.0:
                oriented[1], oriented[2] = oriented[2], oriented[1]
            for layer in range(value.control.schedule.layer_count):
                bottom = np.asarray(
                    [level_nodes[int(node)][layer] for node in oriented],
                    dtype=np.int64,
                )
                top = np.asarray(
                    [level_nodes[int(node)][layer + 1] for node in oriented],
                    dtype=np.int64,
                )
                prism_blocks.setdefault(value.volume_tag, []).append(
                    np.concatenate((bottom, top))
                )
        for edge in boundary_edges:
            first, second = (int(node) for node in edge)
            for layer in range(value.control.schedule.layer_count):
                quad = np.asarray(
                    (
                        level_nodes[first][layer],
                        level_nodes[second][layer],
                        level_nodes[second][layer + 1],
                        level_nodes[first][layer + 1],
                    ),
                    dtype=np.int64,
                )
                center = np.mean(
                    np.asarray([coordinates[int(node)] for node in quad]), axis=0
                )
                surface = _matching_lateral_surface(
                    gmsh, value.lateral_surfaces, center, tolerance
                )
                key = tuple(sorted(int(node) for node in quad))
                existing = quad_blocks.setdefault(surface, {}).get(key)
                if existing is not None and set(existing) != set(quad):
                    raise MeshingFailure(
                        MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                        "A swept cohort produced inconsistent shared quad incidence.",
                        stage=MeshingStageKind.LAYER_GENERATION.value,
                    )
                quad_blocks[surface][key] = quad
    for volume_tag, entries in new_nodes.items():
        tags = np.asarray([tag for tag, _ in entries], dtype=np.int64)
        values = np.asarray([point for _, point in entries], dtype=np.float64)
        gmsh.model.mesh.addNodes(3, volume_tag, tags, values.reshape(-1))
    prism_type = gmsh.model.mesh.getElementType("Prism", 1)
    for volume_tag, prisms in prism_blocks.items():
        gmsh.model.mesh.addElementsByType(
            volume_tag,
            prism_type,
            [],
            np.asarray(prisms, dtype=np.int64).reshape(-1),
        )
    quadrilateral_type = gmsh.model.mesh.getElementType("Quadrangle", 1)
    for surface, quads in quad_blocks.items():
        gmsh.model.mesh.addElementsByType(
            surface,
            quadrilateral_type,
            [],
            np.asarray(tuple(quads.values()), dtype=np.int64).reshape(-1),
        )


def _audit_layers(sweep: Any, rows: Any, node_tags: Any, points: Any, /) -> _LayerAudit:
    if sweep is None:
        return _LayerAudit((), (), {}, {})
    volume_map = {value.volume_tag: value for value in sweep.volumes}
    prism_rows = tuple(row for row in rows if row.cell_kind == "prism")
    control_by_tag = {}
    layer_by_tag = {}
    evaluations: dict[int, list] = {}
    for block in rows:
        controlled = np.asarray(
            [int(tag) in volume_map for tag in block.entity_tags], dtype=np.bool_
        )
        if np.any(controlled) and block.cell_kind != "prism":
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A controlled swept volume contains a non-prism cell.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        if block.cell_kind == "prism" and np.any(~controlled):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A prism cell was generated outside the controlled swept volumes.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
    for value in sweep.volumes:
        selected_vertices = []
        selected_tags = []
        for block in prism_rows:
            selected = block.entity_tags == value.volume_tag
            if np.any(selected):
                selected_vertices.append(block.vertices[selected, : block.corner_count])
                selected_tags.append(block.tags[selected])
        if not selected_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A controlled swept volume contains no generated prisms.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        tags = np.concatenate(selected_tags)
        vertices = _local_connectivity(node_tags, np.concatenate(selected_vertices))
        corners = points[vertices]
        lower = np.mean((corners[:, :3] - value.origin) @ value.unit, axis=1)
        intervals = np.argmin(np.abs(lower[:, None] - value.levels[None, :-1]), axis=1)
        evaluation = evaluate_swept_layer_quality(
            points,
            vertices,
            intervals,
            value.origin,
            value.unit,
            value.control.schedule.thicknesses,
        )
        tolerance = 1.0e-9 * max(float(value.levels[-1]), 1.0)
        if (
            not evaluation.valid
            or evaluation.maximum_thickness_residual > tolerance
            or evaluation.maximum_alignment_residual > 1.0e-10
            or evaluation.maximum_interface_residual > tolerance
            or np.unique(intervals).size != value.control.schedule.layer_count
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Generated prisms do not realize the exact requested thickness, alignment, and interface levels.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        evaluations.setdefault(value.control_index, []).append(evaluation)
        for tag, layer in zip(tags, intervals, strict=True):
            control_by_tag[int(tag)] = value.control_index
            layer_by_tag[int(tag)] = int(layer)
    requested = []
    achieved = []
    control_map = {value.control_index: value.control for value in sweep.volumes}
    controls = tuple(control_map[index] for index in sorted(control_map))
    for control_index, control in zip(sorted(control_map), controls, strict=True):
        key = f"layer:{control.control_id}"
        requested.extend(
            (
                (f"{key}:layer_count", float(control.schedule.layer_count)),
                (f"{key}:total_thickness", control.schedule.total_thickness),
                *(
                    (f"{key}:thickness:{layer}", thickness)
                    for layer, thickness in enumerate(control.schedule.thicknesses)
                ),
            )
        )
        local = evaluations[control_index]
        measured = np.mean(
            np.asarray(
                [np.asarray(value.measured_thicknesses) for value in local],
                dtype=np.float64,
            ),
            axis=0,
        )
        growth = (
            measured[1:] / measured[:-1]
            if measured.size > 1
            else np.empty((0,), dtype=np.float64)
        )
        achieved.extend(
            (
                (f"{key}:layer_count", float(measured.size)),
                (f"{key}:total_thickness", float(np.sum(measured))),
                *(
                    (f"{key}:thickness:{layer}", float(thickness))
                    for layer, thickness in enumerate(measured)
                ),
                *(
                    (f"{key}:growth:{layer}", float(ratio))
                    for layer, ratio in enumerate(growth, start=1)
                ),
                (
                    f"{key}:maximum_thickness_residual",
                    max(value.maximum_thickness_residual for value in local),
                ),
                (
                    f"{key}:maximum_alignment_residual",
                    max(value.maximum_alignment_residual for value in local),
                ),
                (
                    f"{key}:maximum_interface_residual",
                    max(value.maximum_interface_residual for value in local),
                ),
                (
                    f"{key}:relative_cad_symmetric_difference",
                    max(
                        value.relative_cad_difference
                        for value in sweep.volumes
                        if value.control_index == control_index
                    ),
                ),
            )
        )
    if len(controls) == 1:
        control = controls[0]
        measured = np.asarray(evaluations[0][0].measured_thicknesses, dtype=np.float64)
        achieved.extend(
            (
                ("layer_count", float(measured.size)),
                ("first_layer_thickness", float(measured[0])),
                (
                    "layer_growth_rate",
                    float(np.max(measured[1:] / measured[:-1]))
                    if measured.size > 1
                    else 1.0,
                ),
                (
                    "layer_interface_maximum_residual",
                    max(value.maximum_interface_residual for value in evaluations[0]),
                ),
            )
        )
        requested.extend(
            (
                ("layer_count", float(control.schedule.layer_count)),
                ("total_layer_thickness", control.schedule.total_thickness),
            )
        )
    return _LayerAudit(
        tuple(requested),
        tuple(achieved),
        control_by_tag,
        layer_by_tag,
    )


def _layer_attributes(
    mesh: CellMesh,
    rows: tuple[_ElementRows, ...],
    row_orders: dict[str, np.ndarray],
    audit: _LayerAudit,
    /,
) -> tuple[MeshAttribute, ...]:
    if not audit.layer_by_element_tag:
        return ()
    cell_ids = []
    control_indices = []
    layer_indices = []
    rows_by_name = {row.block_name: row for row in rows}
    for block in mesh.blocks:
        if block.cell_kind != "prism":
            continue
        source_rows = rows_by_name[block.name]
        tags = source_rows.tags[row_orders[block.name]]
        cell_ids.extend(int(value) for value in np.asarray(block.global_ids))
        control_indices.extend(audit.control_by_element_tag[int(tag)] for tag in tags)
        layer_indices.extend(audit.layer_by_element_tag[int(tag)] for tag in tags)
    entity_set = mesh.entity_set(3)
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        3,
        entity_set.entity_set_id,
        np.asarray(cell_ids, dtype=np.int64),
    )
    return (
        MeshAttribute(
            "layer_index",
            MeshAttributeRole.MARKER,
            scope,
            np.asarray(layer_indices, dtype=np.int32),
        ),
        MeshAttribute(
            "layer_control_index",
            MeshAttributeRole.MARKER,
            scope,
            np.asarray(control_indices, dtype=np.int32),
        ),
    )


def _audit_swept_interfaces(
    mesh: CellMesh,
    source: BRepModel,
    cell_solid_ids: np.ndarray,
    mesh_face_source: np.ndarray,
    sweep: _SweepGeneration | None,
    /,
) -> tuple[tuple[str, float], ...]:
    if sweep is None or {block.cell_kind for block in mesh.blocks} != {
        "prism",
        "tetrahedron",
    }:
        return ()
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "Mixed swept output requires canonical PolyhedralConnectivity.",
            stage=MeshingStageKind.CANONICALIZATION.value,
        )
    faces = _connectivity_face_rows(connectivity)
    incidents = _connectivity_face_incidents(connectivity)
    cell_kinds = np.concatenate(
        tuple(
            np.full((block.cell_count,), block.cell_kind, dtype=object)
            for block in mesh.blocks
        )
    )
    controlled = {value.solid_index for value in sweep.volumes}
    expected_caps = {
        face
        for value in sweep.volumes
        for face in (value.source_face_index, value.target_face_index)
        if any(owner not in controlled for owner in source.topology.face_solids[face])
    }
    observed_caps = set()
    for face_index, adjacent in enumerate(incidents):
        kinds = {str(cell_kinds[cell]) for cell in adjacent}
        source_face = int(mesh_face_source[face_index])
        if kinds == {"prism", "tetrahedron"}:
            if len(faces[face_index]) != 3 or source_face not in expected_caps:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    "Prism/tetrahedron cells may meet only on a triangular swept cap.",
                    stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
                )
            observed_caps.add(source_face)
        if len(faces[face_index]) == 4 and "tetrahedron" in kinds:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A swept quad curtain adjoins a tetrahedral cell.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
    if observed_caps != expected_caps:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Swept cap triangles do not form the complete conforming prism/tetrahedron interface.",
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    return (("layer_interface_compliance", 1.0),)


# Relative agreement between the measured and requested planar layers.
_FIELD_TOLERANCE = 1.0e-2


@dataclass(frozen=True, slots=True)
class _BoundaryLayerField:
    control: BoundaryLayerControl
    curves: tuple[int, ...]
    fan_points: tuple[int, ...]


def _endpoint_tangent(gmsh: Any, curve: int, point: np.ndarray, /) -> np.ndarray:
    """Unit tangent of ``curve`` at its endpoint ``point``, directed into the curve."""
    lower, upper = gmsh.model.getParametrizationBounds(1, curve)
    ends = np.asarray(
        [gmsh.model.getValue(1, curve, [value]) for value in (lower[0], upper[0])],
        dtype=np.float64,
    )
    at_upper = np.linalg.norm(ends[1] - point) < np.linalg.norm(ends[0] - point)
    parameter = upper[0] if at_upper else lower[0]
    tangent = np.asarray(
        gmsh.model.getDerivative(1, curve, [parameter]), dtype=np.float64
    )
    tangent = -tangent if at_upper else tangent
    return tangent / np.linalg.norm(tangent)


def _wall_fan_points(
    gmsh: Any, control: BoundaryLayerControl, curves: Any, /
) -> tuple[int, ...]:
    """Classify wall-curve junctions; FAN convex corners, REJECT any feature corner."""
    incident: dict[int, list[int]] = {}
    for curve in curves:
        for _, point in gmsh.model.getBoundary(
            [(1, curve)], combined=False, oriented=False, recursive=False
        ):
            incident.setdefault(abs(int(point)), []).append(curve)
    fans = []
    for point, members in sorted(incident.items()):
        if len(members) != 2:
            continue
        location = np.asarray(gmsh.model.getValue(0, point, []), dtype=np.float64)
        first, second = (_endpoint_tangent(gmsh, curve, location) for curve in members)
        deviation = np.pi - np.arccos(np.clip(first @ second, -1.0, 1.0))
        if deviation <= control.feature_angle:
            continue
        surfaces = gmsh.model.getAdjacencies(1, members[0])[0]
        probe = location + 1.0e-6 * (first + second) / np.linalg.norm(first + second)
        # The tangent bisector leaves the domain exactly at convex (fanned) corners.
        convex = not any(
            gmsh.model.isInside(2, int(surface), probe.tolist()) for surface in surfaces
        )
        match control.corner:
            case BoundaryLayerCornerPolicy.REJECT:
                raise MeshingFailure(
                    MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                    "The REJECT corner policy refuses wall corners beyond the feature angle.",
                    stage=MeshingStageKind.LAYER_GENERATION.value,
                    entity_ids=(point,),
                    locations=(tuple(float(value) for value in location),),
                )
            case BoundaryLayerCornerPolicy.FAN:
                if convex:
                    fans.append(point)
            case BoundaryLayerCornerPolicy.SMOOTH:
                continue
            case _:
                raise TypeError("corner must be BoundaryLayerCornerPolicy.")
    return tuple(fans)


def _apply_boundary_layer_field(
    gmsh: Any,
    specification: Any,
    cad_entities: _CadEntityMap | None,
    requested_kinds: set[str],
    /,
) -> _BoundaryLayerField | None:
    """Lower one planar PROVIDER control to a Gmsh BoundaryLayer field."""
    if (
        not isinstance(specification, SurfaceMeshingSpec)
        or not specification.layer_controls
    ):
        return None
    control = specification.layer_controls[0]
    if cad_entities is None or not cad_entities.edge_to_curve:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Planar boundary layers require resolved source edge identities.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    curves = tuple(
        sorted(
            {
                int(cad_entities.edge_to_curve[int(edge)])
                for edge in np.asarray(control.wall_scope.entity_ids)
            }
        )
    )
    thicknesses = np.asarray(control.schedule.thicknesses, dtype=np.float64)
    rates = thicknesses[1:] / thicknesses[:-1]
    if rates.size and np.ptp(rates) > 1.0e-12 * np.max(rates):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Gmsh boundary-layer fields realize geometric schedules only.",
            stage=MeshingStageKind.CONTROL_RESOLUTION.value,
        )
    fans = _wall_fan_points(gmsh, control, curves)
    field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(field, "CurvesList", list(curves))
    gmsh.model.mesh.field.setNumber(field, "Size", float(thicknesses[0]))
    gmsh.model.mesh.field.setNumber(
        field, "Ratio", float(rates[0]) if rates.size else 1.0
    )
    gmsh.model.mesh.field.setNumber(field, "NbLayers", thicknesses.size)
    # A marginally larger cap lets exactly NbLayers geometric layers fit.
    gmsh.model.mesh.field.setNumber(
        field, "Thickness", float(np.sum(thicknesses)) * (1.0 + 1.0e-9)
    )
    gmsh.model.mesh.field.setNumber(
        field, "Quads", 1 if "quadrilateral" in requested_kinds else 0
    )
    if fans:
        gmsh.model.mesh.field.setNumbers(field, "FanPointsList", list(fans))
    gmsh.model.mesh.field.setAsBoundaryLayer(field)
    return _BoundaryLayerField(control, curves, fans)


def _field_columns(
    points: np.ndarray,
    distance: np.ndarray,
    neighbors: np.ndarray,
    start: np.ndarray,
    steps: int,
    /,
) -> np.ndarray:
    """Distances along layer columns: nearest first node, then best-aligned rising nodes."""
    current = start
    previous = start
    levels = []
    for step in range(steps):
        candidate = neighbors[current]
        valid = candidate >= 0
        safe = np.maximum(candidate, 0)
        rising = valid & (distance[safe] > distance[current][:, None])
        if step == 0:
            score = np.where(rising, -distance[safe], -np.inf)
        else:
            axis = points[current] - points[previous]
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            offset = points[safe] - points[current][:, None, :]
            offset /= np.maximum(np.linalg.norm(offset, axis=2, keepdims=True), 1.0e-300)
            score = np.where(rising, np.sum(offset * axis[:, None, :], axis=2), -np.inf)
        choice = np.argmax(score, axis=1)
        following = safe[np.arange(current.size), choice]
        stalled = ~np.isfinite(score[np.arange(current.size), choice])
        following = np.where(stalled, current, following)
        levels.append(distance[following])
        previous, current = current, following
    return np.asarray(levels)


def _audit_boundary_layer_field(
    gmsh: Any,
    field: _BoundaryLayerField | None,
    rows: tuple[_ElementRows, ...],
    node_tags: np.ndarray,
    points: np.ndarray,
    /,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], tuple[str, ...]]:
    """Measure first-layer thickness, growth, and layer count along wall columns."""
    if field is None:
        return (), (), ()
    control = field.control
    requested_values = np.asarray(control.schedule.thicknesses, dtype=np.float64)
    layers = requested_values.size
    edges = []
    for block in rows:
        local = _local_connectivity(node_tags, block.vertices[:, : block.corner_count])
        edges.extend(
            local[:, (corner, (corner + 1) % block.corner_count)]
            for corner in range(block.corner_count)
        )
    edges = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
    ends = np.concatenate((edges, edges[:, ::-1]))
    order = np.argsort(ends[:, 0], kind="stable")
    ends = ends[order]
    degree = np.bincount(ends[:, 0], minlength=points.shape[0])
    starts = np.concatenate(([0], np.cumsum(degree)[:-1]))
    neighbors = np.full((points.shape[0], int(degree.max())), -1, dtype=np.int64)
    neighbors[ends[:, 0], np.arange(ends.shape[0]) - starts[ends[:, 0]]] = ends[:, 1]
    segments = []
    corner_nodes = []
    for curve in field.curves:
        _, _, node_blocks = gmsh.model.mesh.getElements(1, curve)
        segments.append(
            _local_connectivity(
                node_tags, np.asarray(node_blocks[0], dtype=np.int64).reshape(-1, 2)
            )
        )
        for _, point in gmsh.model.getBoundary(
            [(1, curve)], combined=False, oriented=False, recursive=False
        ):
            tags, _, _ = gmsh.model.mesh.getNodes(0, abs(int(point)))
            corner_nodes.append(
                _local_connectivity(node_tags, np.asarray(tags, dtype=np.int64))
            )
    segments = np.concatenate(segments)
    distance = _nearest_distances(points[segments][:, (0, 1, 1)], points)
    start = np.setdiff1d(np.unique(segments), np.concatenate(corner_nodes))
    levels = np.median(
        _field_columns(points, distance, neighbors, start, layers + 1), axis=1
    )
    measured = np.diff(np.concatenate(([0.0], levels)))
    matching = (
        np.abs(measured[:layers] - requested_values)
        <= _FIELD_TOLERANCE * requested_values
    )
    count = layers if np.all(matching) else int(np.argmin(matching))
    key = f"layer:{control.control_id}"
    requested = (
        (f"{key}:layer_count", float(layers)),
        (f"{key}:first_layer_thickness", float(requested_values[0])),
        *(
            (f"{key}:thickness:{index}", float(value))
            for index, value in enumerate(requested_values)
        ),
    )
    achieved = (
        (f"{key}:layer_count", float(count)),
        (f"{key}:first_layer_thickness", float(measured[0])),
        *(
            (f"{key}:thickness:{index}", float(value))
            for index, value in enumerate(measured[:layers])
        ),
        *(
            (f"{key}:growth:{index}", float(value))
            for index, value in enumerate(
                measured[1:layers] / measured[: layers - 1], start=1
            )
        ),
        (f"{key}:fan_point_count", float(len(field.fan_points))),
        (f"{key}:first_core_spacing", float(measured[layers])),
    )
    issues = () if count == layers else (f"layer_thickness:{control.control_id}",)
    return requested, achieved, issues
