#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellMesh


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _nonadjacent_segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    first = first_end - first_start
    second = second_end - second_start
    offset = second_start - first_start
    denominator = _cross_2d(first, second)
    if abs(denominator) <= tolerance:
        if abs(_cross_2d(offset, first)) > tolerance:
            return False
        scale = float(np.dot(first, first))
        parameters = sorted(
            (
                float(np.dot(second_start - first_start, first) / scale),
                float(np.dot(second_end - first_start, first) / scale),
            )
        )
        return min(1.0, parameters[1]) >= max(0.0, parameters[0]) - tolerance
    first_parameter = _cross_2d(offset, second) / denominator
    second_parameter = _cross_2d(offset, first) / denominator
    return (
        -tolerance <= first_parameter <= 1.0 + tolerance
        and -tolerance <= second_parameter <= 1.0 + tolerance
    )


class CrackProjection(StrictModule, NonTrainableState):
    """Closest finite-segment projection with the selected local chart."""

    points: Array
    segment_indices: Array
    parameters: Array
    distances: Array
    signed_normal_coordinates: Array


class CrackFrontGeometry(StrictModule, NonTrainableState):
    """An oriented, finite, non-self-intersecting two-dimensional crack front."""

    vertices: Array
    segments: Array
    segment_ids: Array
    tip_vertex_ids: Array
    tip_ids: Array
    orientation: int = eqx.field(static=True)
    crack_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        segments: ArrayLike,
        /,
        *,
        segment_ids: ArrayLike | None = None,
        tip_vertex_ids: ArrayLike | None = None,
        tip_ids: ArrayLike | None = None,
        orientation: int = 1,
        crack_id: str = "crack",
        intersection_tolerance: float = 1.0e-12,
    ):
        points = np.asarray(vertices, dtype=float)
        connectivity = np.asarray(segments, dtype=np.int32)
        orientation_ = int(orientation)
        identifier = str(crack_id)
        tolerance = float(intersection_tolerance)
        if (
            points.ndim != 2
            or points.shape[1:] != (2,)
            or points.shape[0] < 2
            or np.any(~np.isfinite(points))
        ):
            raise ValueError("Crack-front vertices require finite shape (n >= 2, 2).")
        if (
            connectivity.ndim != 2
            or connectivity.shape[1:] != (2,)
            or connectivity.shape[0] == 0
            or np.any(connectivity < 0)
            or np.any(connectivity >= points.shape[0])
            or np.any(connectivity[:, 0] == connectivity[:, 1])
        ):
            raise ValueError(
                "Crack-front segments require valid pairs of distinct vertices."
            )
        if (
            orientation_ not in (-1, 1)
            or not identifier
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Crack orientation, identifier, and tolerance are invalid.")
        undirected = np.sort(connectivity, axis=1)
        if np.unique(undirected, axis=0).shape[0] != connectivity.shape[0]:
            raise ValueError("Crack-front segments must be unique.")
        vectors = points[connectivity[:, 1]] - points[connectivity[:, 0]]
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= tolerance):
            raise ValueError("Crack-front segments must have positive finite length.")

        degrees = np.bincount(connectivity.reshape((-1,)), minlength=points.shape[0])
        if np.any((degrees < 1) | (degrees > 2)):
            raise ValueError("A crack-front vertex must belong to one or two segments.")
        endpoints = np.flatnonzero(degrees == 1).astype(np.int32)
        declared_tip_vertices = (
            endpoints
            if tip_vertex_ids is None
            else np.asarray(tip_vertex_ids, dtype=np.int32)
        )
        if (
            declared_tip_vertices.ndim != 1
            or np.any(declared_tip_vertices < 0)
            or np.any(declared_tip_vertices >= points.shape[0])
            or np.unique(declared_tip_vertices).size != declared_tip_vertices.size
            or not np.array_equal(np.sort(declared_tip_vertices), endpoints)
        ):
            raise ValueError(
                "Live crack tips must identify every degree-one front vertex exactly once."
            )
        declared_tip_ids = (
            declared_tip_vertices.astype(np.int64)
            if tip_ids is None
            else np.asarray(tip_ids, dtype=np.int64)
        )
        declared_segment_ids = (
            np.arange(connectivity.shape[0], dtype=np.int64)
            if segment_ids is None
            else np.asarray(segment_ids, dtype=np.int64)
        )
        if (
            declared_tip_ids.shape != declared_tip_vertices.shape
            or np.any(declared_tip_ids < 0)
            or np.unique(declared_tip_ids).size != declared_tip_ids.size
            or declared_segment_ids.shape != (connectivity.shape[0],)
            or np.any(declared_segment_ids < 0)
            or np.unique(declared_segment_ids).size != declared_segment_ids.size
        ):
            raise ValueError(
                "Crack tip and segment IDs must be stable unique nonnegative integers."
            )

        segment_sets = tuple(frozenset(row.tolist()) for row in connectivity)
        for first_index in range(connectivity.shape[0]):
            first = connectivity[first_index]
            for second_index in range(first_index + 1, connectivity.shape[0]):
                shared_vertices = segment_sets[first_index] & segment_sets[second_index]
                if shared_vertices:
                    shared_vertex = next(iter(shared_vertices))
                    first_other = (
                        int(first[1]) if int(first[0]) == shared_vertex else int(first[0])
                    )
                    second = connectivity[second_index]
                    second_other = (
                        int(second[1])
                        if int(second[0]) == shared_vertex
                        else int(second[0])
                    )
                    first_ray = points[first_other] - points[shared_vertex]
                    second_ray = points[second_other] - points[shared_vertex]
                    if (
                        abs(_cross_2d(first_ray, second_ray)) <= tolerance
                        and float(np.dot(first_ray, second_ray)) > 0.0
                    ):
                        raise ValueError(
                            "Adjacent crack-front segments may meet only at their common endpoint."
                        )
                    continue
                second = connectivity[second_index]
                if _nonadjacent_segments_intersect(
                    points[first[0]],
                    points[first[1]],
                    points[second[0]],
                    points[second[1]],
                    tolerance,
                ):
                    raise ValueError(
                        "Nonadjacent crack-front segments must not intersect."
                    )

        self.vertices = jnp.asarray(points)
        self.segments = jnp.asarray(connectivity)
        self.segment_ids = jnp.asarray(declared_segment_ids)
        self.tip_vertex_ids = jnp.asarray(declared_tip_vertices)
        self.tip_ids = jnp.asarray(declared_tip_ids)
        self.orientation = orientation_
        self.crack_id = identifier
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "finite-crack-front-geometry",
                "crack": identifier,
                "vertices": points.tolist(),
                "segments": connectivity.tolist(),
                "segment_ids": declared_segment_ids.tolist(),
                "tip_vertices": declared_tip_vertices.tolist(),
                "tip_ids": declared_tip_ids.tolist(),
                "orientation": orientation_,
            }
        )

    @property
    def length(self) -> Array:
        vectors = self.vertices[self.segments[:, 1]] - self.vertices[self.segments[:, 0]]
        return jnp.sum(jnp.linalg.norm(vectors, axis=1))

    def segment_tangents(self) -> Array:
        vectors = self.vertices[self.segments[:, 1]] - self.vertices[self.segments[:, 0]]
        return vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    def segment_normals(self) -> Array:
        tangents = self.segment_tangents()
        return self.orientation * jnp.stack((-tangents[:, 1], tangents[:, 0]), axis=-1)

    def project(self, points: ArrayLike, /) -> CrackProjection:
        query = jnp.asarray(points)
        if query.ndim < 1 or query.shape[-1] != 2:
            raise ValueError("Crack projection points must end in two coordinates.")
        starts = self.vertices[self.segments[:, 0]]
        vectors = self.vertices[self.segments[:, 1]] - starts
        relative = query[..., None, :] - starts
        parameters = jnp.clip(
            jnp.sum(relative * vectors, axis=-1) / jnp.sum(vectors * vectors, axis=-1),
            0.0,
            1.0,
        )
        projections = starts + parameters[..., None] * vectors
        differences = query[..., None, :] - projections
        squared_distances = jnp.sum(differences * differences, axis=-1)
        indices = jnp.argmin(squared_distances, axis=-1)
        point_indices = jnp.broadcast_to(
            indices[..., None, None],
            projections.shape[:-2] + (1, projections.shape[-1]),
        )
        selected_points = jnp.take_along_axis(
            projections,
            point_indices,
            axis=-2,
        )[..., 0, :]
        selected_parameters = jnp.take_along_axis(
            parameters,
            indices[..., None],
            axis=-1,
        )[..., 0]
        selected_distances = jnp.sqrt(
            jnp.take_along_axis(squared_distances, indices[..., None], axis=-1)[..., 0]
        )
        normals = self.segment_normals()[indices]
        signed = jnp.sum((query - selected_points) * normals, axis=-1)
        return CrackProjection(
            selected_points,
            indices,
            selected_parameters,
            selected_distances,
            signed,
        )

    def signed_distance(self, points: ArrayLike, /) -> Array:
        return self.project(points).signed_normal_coordinates

    def heaviside(self, points: ArrayLike, /) -> Array:
        return jnp.where(self.signed_distance(points) >= 0.0, 1.0, -1.0)

    def tip_frame(self, tip_id: int, /) -> tuple[Array, Array, Array]:
        identifier = int(tip_id)
        tip_ids = np.asarray(self.tip_ids)
        positions = np.flatnonzero(tip_ids == identifier)
        if positions.size != 1:
            raise ValueError("tip_id does not identify one live crack tip.")
        vertex_index = int(np.asarray(self.tip_vertex_ids)[positions[0]])
        segments = np.asarray(self.segments)
        adjacent = np.flatnonzero(np.any(segments == vertex_index, axis=1))
        if adjacent.size != 1:
            raise ValueError("A live crack tip must have exactly one adjacent segment.")
        segment_index = int(adjacent[0])
        segment = segments[segment_index]
        tangent = self.segment_tangents()[segment_index]
        outward = tangent if int(segment[1]) == vertex_index else -tangent
        normal = self.orientation * jnp.asarray((-outward[1], outward[0]))
        return self.vertices[vertex_index], outward, normal

    def tip_local_coordinates(self, points: ArrayLike, tip_id: int, /) -> Array:
        query = jnp.asarray(points)
        if query.ndim < 1 or query.shape[-1] != 2:
            raise ValueError("Tip-coordinate points must end in two coordinates.")
        origin, tangent, normal = self.tip_frame(tip_id)
        relative = query - origin
        axial = jnp.sum(relative * tangent, axis=-1)
        transverse = jnp.sum(relative * normal, axis=-1)
        radius = jnp.sqrt(axial * axial + transverse * transverse)
        angle = jnp.arctan2(transverse, axial)
        return jnp.stack((radius, angle), axis=-1)

    def with_tip_extension(
        self,
        tip_id: int,
        endpoint: ArrayLike,
        /,
    ) -> CrackFrontGeometry:
        identifier = int(tip_id)
        point = np.asarray(endpoint, dtype=float)
        if point.shape != (2,) or np.any(~np.isfinite(point)):
            raise ValueError(
                "A crack-growth endpoint must be one finite two-dimensional point."
            )
        tip_ids = np.asarray(self.tip_ids)
        positions = np.flatnonzero(tip_ids == identifier)
        if positions.size != 1:
            raise ValueError("tip_id does not identify one live crack tip.")
        tip_position = int(positions[0])
        tip_vertex = int(np.asarray(self.tip_vertex_ids)[tip_position])
        segments = np.asarray(self.segments)
        adjacent = np.flatnonzero(np.any(segments == tip_vertex, axis=1))
        segment = segments[int(adjacent[0])]
        vertices = np.concatenate((np.asarray(self.vertices), point[None, :]), axis=0)
        new_vertex = vertices.shape[0] - 1
        new_segment = (
            np.asarray((tip_vertex, new_vertex), dtype=np.int32)
            if int(segment[1]) == tip_vertex
            else np.asarray((new_vertex, tip_vertex), dtype=np.int32)
        )
        updated_segments = np.concatenate((segments, new_segment[None, :]), axis=0)
        updated_tip_vertices = np.asarray(self.tip_vertex_ids).copy()
        updated_tip_vertices[tip_position] = new_vertex
        next_segment_id = int(np.max(np.asarray(self.segment_ids))) + 1
        return CrackFrontGeometry(
            vertices,
            updated_segments,
            segment_ids=np.concatenate(
                (
                    np.asarray(self.segment_ids),
                    np.asarray((next_segment_id,), dtype=np.int64),
                )
            ),
            tip_vertex_ids=updated_tip_vertices,
            tip_ids=self.tip_ids,
            orientation=self.orientation,
            crack_id=self.crack_id,
        )


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, tolerance: float) -> bool:
    matrix = np.stack((triangle[1] - triangle[0], triangle[2] - triangle[0]), axis=1)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= tolerance:
        raise ValueError("Sharp-crack classification requires nondegenerate triangles.")
    coordinates = np.linalg.solve(matrix, point - triangle[0])
    return bool(
        coordinates[0] >= -tolerance
        and coordinates[1] >= -tolerance
        and coordinates.sum() <= 1.0 + tolerance
    )


def _segment_triangle_interval(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    tolerance: float,
) -> tuple[float, float] | None:
    direction = end - start
    parameters: list[float] = []
    if _point_in_triangle(start, triangle, tolerance):
        parameters.append(0.0)
    if _point_in_triangle(end, triangle, tolerance):
        parameters.append(1.0)
    for index in range(3):
        edge_start = triangle[index]
        edge_end = triangle[(index + 1) % 3]
        edge = edge_end - edge_start
        offset = edge_start - start
        denominator = _cross_2d(direction, edge)
        if abs(denominator) <= tolerance:
            if abs(_cross_2d(offset, direction)) <= tolerance:
                projection = float(
                    np.dot(edge_start - start, direction) / np.dot(direction, direction)
                )
                projection_end = float(
                    np.dot(edge_end - start, direction) / np.dot(direction, direction)
                )
                overlap = min(1.0, max(projection, projection_end)) - max(
                    0.0, min(projection, projection_end)
                )
                if overlap > tolerance:
                    raise ValueError("A crack segment may not overlap a cell edge.")
            continue
        segment_parameter = _cross_2d(offset, edge) / denominator
        edge_parameter = _cross_2d(offset, direction) / denominator
        if (
            -tolerance <= segment_parameter <= 1.0 + tolerance
            and -tolerance <= edge_parameter <= 1.0 + tolerance
        ):
            parameters.append(float(np.clip(segment_parameter, 0.0, 1.0)))
    if len(parameters) < 2:
        return None
    unique = (
        np.unique(np.round(np.asarray(parameters) / tolerance).astype(np.int64))
        * tolerance
    )
    lower = float(np.min(unique))
    upper = float(np.max(unique))
    if upper - lower <= tolerance:
        return None
    return lower, upper


class SharpCrackTopology(StrictModule, NonTrainableState):
    """Frozen finite-segment decisions for one mesh and crack geometry."""

    geometry: CrackFrontGeometry
    cut_cell_ids: Array
    split_cell_ids: Array
    tip_cell_ids: Array
    cell_segment_ids: Array
    heaviside_vertex_ids: Array
    branch_vertex_ids: Array
    classification_margin: Array
    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_version: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CrackFrontGeometry,
        cut_cell_ids: ArrayLike,
        split_cell_ids: ArrayLike,
        tip_cell_ids: ArrayLike,
        cell_segment_ids: ArrayLike,
        heaviside_vertex_ids: ArrayLike,
        branch_vertex_ids: ArrayLike,
        /,
        *,
        mesh_id: str,
        classification_margin: ArrayLike,
        topology_version: int = 0,
    ):
        if not isinstance(geometry, CrackFrontGeometry):
            raise TypeError("geometry must be CrackFrontGeometry.")
        cut = np.asarray(cut_cell_ids, dtype=np.int64)
        split = np.asarray(split_cell_ids, dtype=np.int64)
        tip = np.asarray(tip_cell_ids, dtype=np.int64)
        segment_ids = np.asarray(cell_segment_ids, dtype=np.int64)
        heaviside = np.asarray(heaviside_vertex_ids, dtype=np.int64)
        branch = np.asarray(branch_vertex_ids, dtype=np.int64)
        margin = np.asarray(classification_margin)
        identifier = str(mesh_id)
        version = int(topology_version)
        arrays = (cut, split, tip, heaviside, branch)
        if (
            any(value.ndim != 1 or np.any(value < 0) for value in arrays)
            or any(np.unique(value).size != value.size for value in arrays)
            or segment_ids.shape != cut.shape
            or not set(split.tolist()).issubset(set(cut.tolist()))
            or not set(tip.tolist()).issubset(set(cut.tolist()))
            or margin.shape != ()
            or not np.isfinite(margin)
            or margin < 0.0
            or not identifier
            or version < 0
        ):
            raise ValueError("Sharp-crack topology decisions are inconsistent.")
        if not set(segment_ids.tolist()).issubset(
            set(np.asarray(geometry.segment_ids).tolist())
        ):
            raise ValueError("Topology cell segments must belong to its crack geometry.")
        self.geometry = geometry
        self.cut_cell_ids = jnp.asarray(cut)
        self.split_cell_ids = jnp.asarray(split)
        self.tip_cell_ids = jnp.asarray(tip)
        self.cell_segment_ids = jnp.asarray(segment_ids)
        self.heaviside_vertex_ids = jnp.asarray(heaviside)
        self.branch_vertex_ids = jnp.asarray(branch)
        self.classification_margin = jnp.asarray(margin)
        self.mesh_id = identifier
        self.geometry_id = geometry.geometry_id
        self.topology_version = version
        self.topology_id = canonical_fingerprint(
            {
                "kind": "sharp-crack-topology",
                "mesh": identifier,
                "geometry": geometry.geometry_id,
                "cut_cells": cut.tolist(),
                "split_cells": split.tolist(),
                "tip_cells": tip.tolist(),
                "cell_segments": segment_ids.tolist(),
                "heaviside_vertices": heaviside.tolist(),
                "branch_vertices": branch.tolist(),
                "margin": float(margin),
                "version": version,
            }
        )


def build_sharp_crack_topology(
    mesh: CellMesh,
    geometry: CrackFrontGeometry,
    /,
    *,
    tolerance: float = 1.0e-10,
    topology_version: int = 0,
) -> SharpCrackTopology:
    """Classify only cells intersected by finite crack segments, never their extensions."""

    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CrackFrontGeometry):
        raise TypeError("Sharp-crack topology requires CellMesh and CrackFrontGeometry.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError(
            "Sharp-crack classification tolerance must be positive and finite."
        )
    if (
        mesh.ambient_dimension != 2
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind != "triangle"
    ):
        raise ValueError(
            "Sharp-crack topology currently requires one two-dimensional T3 block."
        )
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    crack_vertices = np.asarray(geometry.vertices)
    crack_segments = np.asarray(geometry.segments)
    crack_segment_ids = np.asarray(geometry.segment_ids)
    tip_vertices = set(np.asarray(geometry.tip_vertex_ids).tolist())
    for tip_vertex in tip_vertices:
        tip_point = crack_vertices[tip_vertex]
        if not any(
            _point_in_triangle(tip_point, coordinates[cell], tolerance_) for cell in cells
        ):
            raise ValueError("Every live crack tip must lie in the classified body mesh.")

    cut_cells: list[int] = []
    split_cells: list[int] = []
    tip_cells: list[int] = []
    cell_segments: list[int] = []
    heaviside_vertices: set[int] = set()
    branch_vertices: set[int] = set()
    margins: list[float] = []
    vertex_global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for cell, cell_id in zip(cells, cell_ids, strict=True):
        triangle = coordinates[cell]
        hits: list[tuple[int, float, float]] = []
        for segment_index, segment in enumerate(crack_segments):
            interval = _segment_triangle_interval(
                crack_vertices[segment[0]],
                crack_vertices[segment[1]],
                triangle,
                tolerance_,
            )
            if interval is not None:
                hits.append((segment_index, interval[0], interval[1]))
        if not hits:
            continue
        if len(hits) > 1:
            hit_vectors = np.asarray(
                [
                    crack_vertices[crack_segments[index, 1]]
                    - crack_vertices[crack_segments[index, 0]]
                    for index, _, _ in hits
                ]
            )
            hit_tangents = hit_vectors / np.linalg.norm(
                hit_vectors, axis=1, keepdims=True
            )
            if np.any(
                np.abs(
                    hit_tangents[1:, 0] * hit_tangents[0, 1]
                    - hit_tangents[1:, 1] * hit_tangents[0, 0]
                )
                > tolerance_
            ):
                raise ValueError(
                    "A kink spanning multiple front segments in one T3 cell "
                    "requires mesh refinement."
                )
        segment_index, lower, upper = max(
            hits,
            key=lambda value: value[2] - value[1],
        )
        segment = crack_segments[segment_index]
        live_tip_in_cell = any(
            vertex in tip_vertices
            and _point_in_triangle(crack_vertices[vertex], triangle, tolerance_)
            for hit_index, _, _ in hits
            for vertex in crack_segments[hit_index]
        )
        cut_cells.append(int(cell_id))
        cell_segments.append(int(crack_segment_ids[segment_index]))
        heaviside_vertices.update(vertex_global_ids[cell].tolist())
        segment_length = float(
            np.linalg.norm(crack_vertices[segment[1]] - crack_vertices[segment[0]])
        )
        margins.append((upper - lower) * segment_length)
        if live_tip_in_cell:
            tip_cells.append(int(cell_id))
            branch_vertices.update(vertex_global_ids[cell].tolist())
        else:
            split_cells.append(int(cell_id))

    return SharpCrackTopology(
        geometry,
        np.asarray(cut_cells, dtype=np.int64),
        np.asarray(split_cells, dtype=np.int64),
        np.asarray(tip_cells, dtype=np.int64),
        np.asarray(cell_segments, dtype=np.int64),
        np.asarray(sorted(heaviside_vertices), dtype=np.int64),
        np.asarray(sorted(branch_vertices), dtype=np.int64),
        mesh_id=mesh.mesh_id,
        classification_margin=np.asarray(min(margins) if margins else 0.0),
        topology_version=topology_version,
    )


__all__ = [
    "CrackFrontGeometry",
    "CrackProjection",
    "SharpCrackTopology",
    "build_sharp_crack_topology",
]
