#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import assert_never, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._polygon import (
    orientation as _orientation,
    signed_area2 as _signed_area2,
    validate_simple_polygon as _validate_simple_polygon,
)
from ._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh


class PolyhedralFaceTriangulation(StrictModule, NonTrainableState):
    """Shared native boundary-preserving triangles of planar polyhedral faces."""

    triangle_offsets: np.ndarray
    triangle_vertices: np.ndarray
    face_area_vectors: np.ndarray
    face_measures: np.ndarray
    face_centroids: np.ndarray
    native_work_evidence: np.ndarray
    native_memory_evidence: np.ndarray
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    triangulation_id: str = eqx.field(static=True)


def prepare_polyhedral_face_triangulation(
    mesh: CellMesh,
    /,
    *,
    planarity_tolerance: float = 1e-10,
    maximum_entries: int = 100_000_000,
    cell_geometry: CellGeometrySpec | None = None,
) -> PolyhedralFaceTriangulation:
    """Triangulate every face exactly in projection, retaining all boundary nodes.

    No fan center is assumed to lie inside a nonconvex face. The native CDT
    uses its closed constrained boundary to classify the domain. Triangle
    incidence and projected area are independently checked here; zero-area
    triangles, discarded boundary points and unresolved native status refuse.
    """
    from fractions import Fraction

    from .._meshcore import exact_orient2d
    from ..geometry._triangulation import ConstrainedDelaunayTriangulation
    from ._cell_complex import PolyhedralConnectivity
    from ._cell_mesh import CellMesh

    if not isinstance(mesh, CellMesh) or not isinstance(
        mesh.connectivity, PolyhedralConnectivity
    ):
        raise TypeError(
            "Polyhedral face preparation requires canonical polyhedral connectivity."
        )
    tolerance = float(planarity_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("planarity_tolerance must be finite and positive.")
    if (
        isinstance(maximum_entries, bool)
        or not isinstance(maximum_entries, int)
        or maximum_entries < 1
    ):
        raise ValueError("maximum_entries must be a positive integer.")
    if cell_geometry is not None:
        from ._exact_power_consumers import exact_power_face_triangulation

        match cell_geometry.exact_source:
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
            ):
                return exact_power_face_triangulation(
                    mesh, cell_geometry, maximum_entries=maximum_entries
                )
            case None | ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
                raise ValueError(
                    "Polyhedral source face preparation requires exact power geometry."
                )
            case invalid:
                assert_never(invalid)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    c = mesh.connectivity
    offsets, values = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    counts = np.diff(offsets) - 2
    if np.any(counts < 1) or 3 * int(np.sum(counts)) > maximum_entries:
        raise ValueError("Polyhedral face triangulation capacity exceeded.")
    triangle_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    triangles = np.empty((int(triangle_offsets[-1]), 3), dtype=np.int32)
    vectors = np.empty((c.face_count, 3), dtype=np.float64)
    measures = np.empty(c.face_count, dtype=np.float64)
    centers = np.empty((c.face_count, 3), dtype=np.float64)
    work, memory, evidence = [], [], []
    for face, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        loop = values[start:stop]
        polygon = points[loop]
        shifted = polygon - polygon[0]
        vector = np.sum(np.cross(shifted, np.roll(shifted, -1, axis=0)), axis=0) / 2
        magnitude = float(np.linalg.norm(vector))
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError(f"Polyhedral face {face} has unresolved/degenerate area.")
        normal = vector / magnitude
        if np.max(np.abs(shifted @ normal)) > tolerance * max(
            1.0, float(np.max(np.linalg.norm(shifted, axis=1)))
        ):
            raise ValueError(f"Polyhedral face {face} fails planarity admission.")
        axis = int(np.argmax(np.abs(vector)))
        axes = ((axis + 1) % 3, (axis + 2) % 3)
        projected = polygon[:, axes]
        area2 = sum(
            Fraction(float(a[0])) * Fraction(float(b[1]))
            - Fraction(float(a[1])) * Fraction(float(b[0]))
            for a, b in zip(projected, np.roll(projected, -1, axis=0), strict=True)
        )
        if area2 == 0:
            raise ValueError(f"Polyhedral face {face} has zero projected area.")
        boundary = np.column_stack(
            (np.arange(loop.size), np.roll(np.arange(loop.size), -1))
        )
        prepared = ConstrainedDelaunayTriangulation(
            projected,
            boundary,
            max_steiner=0,
            max_triangles=int(2 * loop.size + 4),
            max_cavity_cells=max(1, int(2 * loop.size + 4)),
        )
        if (
            prepared.evidence.status != "ok"
            or prepared.points.shape != projected.shape
            or not np.array_equal(prepared.points, projected)
        ):
            raise ValueError(
                f"Polyhedral face {face} lacks a resolved boundary-preserving native triangulation."
            )
        local = np.asarray(prepared.triangles, dtype=np.int32).copy()
        if local.shape != (loop.size - 2, 3) or not np.array_equal(
            np.unique(local), np.arange(loop.size)
        ):
            raise ValueError(
                f"Polyhedral face {face} triangulation loses its boundary topology."
            )
        if area2 < 0:
            local[:, [1, 2]] = local[:, [2, 1]]
        signs = exact_orient2d(
            projected[local[:, 0]], projected[local[:, 1]], projected[local[:, 2]]
        )
        if np.any(signs != (1 if area2 > 0 else -1)):
            raise ValueError(f"Polyhedral face {face} contains nonpositive triangles.")
        directed = {}
        partition = Fraction(0)
        for triangle in local:
            corners = projected[triangle]
            partition += (
                Fraction(float(corners[1, 0])) - Fraction(float(corners[0, 0]))
            ) * (Fraction(float(corners[2, 1])) - Fraction(float(corners[0, 1]))) - (
                Fraction(float(corners[1, 1])) - Fraction(float(corners[0, 1]))
            ) * (Fraction(float(corners[2, 0])) - Fraction(float(corners[0, 0])))
            for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
                a, b = int(a), int(b)
                edge = (min(a, b), max(a, b))
                directed[edge] = directed.get(edge, 0) + (1 if a < b else -1)
        expected = {
            (min(int(a), int(b)), max(int(a), int(b))): (1 if a < b else -1)
            for a, b in boundary
        }
        actual = {edge: sign for edge, sign in directed.items() if sign}
        if actual != expected or partition != area2:
            raise ValueError(
                f"Polyhedral face {face} triangulation fails reciprocal boundary/area closure."
            )
        global_triangles = loop[local]
        triangle_points = points[global_triangles]
        triangle_vectors = (
            np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            / 2
        )
        weights = triangle_vectors @ normal
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(
                f"Polyhedral face {face} has unresolved physical triangle measures."
            )
        triangles[triangle_offsets[face] : triangle_offsets[face + 1]] = global_triangles
        vectors[face] = np.sum(triangle_vectors, axis=0)
        measures[face] = np.sum(weights)
        centers[face] = (
            np.sum(weights[:, None] * np.mean(triangle_points, axis=1), axis=0)
            / measures[face]
        )
        work.append(prepared.work_evidence)
        memory.append(prepared.memory_evidence)
        evidence.append(prepared.evidence.evidence_id)
    arrays = (
        triangle_offsets,
        triangles,
        vectors,
        measures,
        centers,
        np.asarray(work),
        np.asarray(memory),
    )
    for array in arrays:
        array.setflags(write=False)
    return PolyhedralFaceTriangulation(
        *arrays,
        tuple(evidence),
        canonical_fingerprint(
            {
                "kind": "polyhedral-face-triangulation",
                "mesh": mesh.mesh_id,
                "triangles": array_tree_fingerprint(triangles),
                "planarity_tolerance": tolerance,
                "native": evidence,
            }
        ),
    )


def _cross_2d(left: Array, right: Array) -> Array:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _remove_collinear(points: np.ndarray, indices: list[int], /) -> list[int]:
    changed = True
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
    result = list(indices)
    while changed and len(result) > 3:
        changed = False
        for position in range(len(result)):
            left = result[position - 1]
            center = result[position]
            right = result[(position + 1) % len(result)]
            if (
                abs(_orientation(points[left], points[center], points[right]))
                <= tolerance
            ):
                result.pop(position)
                changed = True
                break
    return result


def _inside_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float | np.floating,
    /,
) -> bool | np.bool_:
    return (
        _orientation(a, b, point) >= -tolerance
        and _orientation(b, c, point) >= -tolerance
        and _orientation(c, a, point) >= -tolerance
    )


def _ear_clip(points: np.ndarray, /) -> tuple[tuple[int, int, int], ...]:
    remaining = _remove_collinear(points, list(range(points.shape[0])))
    triangles: list[tuple[int, int, int]] = []
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale * scale
    while len(remaining) > 3:
        clipped = False
        for position in range(len(remaining)):
            left = remaining[position - 1]
            center = remaining[position]
            right = remaining[(position + 1) % len(remaining)]
            if _orientation(points[left], points[center], points[right]) <= tolerance:
                continue
            if any(
                candidate not in (left, center, right)
                and _inside_triangle(
                    points[candidate],
                    points[left],
                    points[center],
                    points[right],
                    tolerance,
                )
                for candidate in remaining
            ):
                continue
            triangles.append((left, center, right))
            remaining.pop(position)
            clipped = True
            break
        if not clipped:
            raise ValueError("Polygon triangulation could not identify a valid ear.")
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    if not triangles:
        raise ValueError("Polygon triangulation produced no positive triangles.")
    return tuple(triangles)


def _clip_half_plane(
    polygon: list[np.ndarray], start: np.ndarray, stop: np.ndarray, /
) -> list[np.ndarray]:
    if not polygon:
        return []
    edge = stop - start

    def signed(point: np.ndarray) -> float:
        return _orientation(start, stop, point)

    result: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = signed(previous)
    for current in polygon:
        current_value = signed(current)
        previous_inside = previous_value >= 0.0
        current_inside = current_value >= 0.0
        if previous_inside != current_inside:
            direction = current - previous
            denominator = edge[0] * direction[1] - edge[1] * direction[0]
            if denominator != 0.0:
                numerator = edge[0] * (start[1] - previous[1]) - edge[1] * (
                    start[0] - previous[0]
                )
                result.append(previous + (numerator / denominator) * direction)
        if current_inside:
            result.append(current)
        previous = current
        previous_value = current_value
    return result


def _kernel_witness(points: np.ndarray, /) -> tuple[np.ndarray, float]:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    extent = max(float(np.max(upper - lower)), 1.0)
    lower = lower - extent
    upper = upper + extent
    kernel = [
        np.asarray((lower[0], lower[1])),
        np.asarray((upper[0], lower[1])),
        np.asarray((upper[0], upper[1])),
        np.asarray((lower[0], upper[1])),
    ]
    for index in range(points.shape[0]):
        kernel = _clip_half_plane(
            kernel, points[index], points[(index + 1) % points.shape[0]]
        )
        if not kernel:
            raise ValueError("Polygon is not star-shaped.")
    witness = np.mean(np.stack(kernel), axis=0)
    edge = np.roll(points, -1, axis=0) - points
    lengths = np.sqrt(np.sum(edge * edge, axis=1))
    margins = (
        np.asarray(
            [
                _orientation(points[i], points[(i + 1) % points.shape[0]], witness)
                for i in range(points.shape[0])
            ]
        )
        / lengths
    )
    area = 0.5 * _signed_area2(points)
    return witness, float(np.min(margins) / math.sqrt(area))


class PolygonAdmissibilityPolicy(StrictModule, NonTrainableState):
    minimum_star_margin: float = eqx.field(static=True)
    minimum_edge_ratio: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_star_margin: float = 1.0e-8,
        minimum_edge_ratio: float = 1.0e-10,
    ) -> None:
        star = float(minimum_star_margin)
        edge = float(minimum_edge_ratio)
        if star < 0.0 or edge < 0.0:
            raise ValueError("Polygon admissibility thresholds must be nonnegative.")
        self.minimum_star_margin = star
        self.minimum_edge_ratio = edge
        self.policy_id = canonical_fingerprint(
            {"kind": "polygon-admissibility", "star": star, "edge": edge}
        )


class PolygonTriangulation(StrictModule, NonTrainableState):
    local_triangles: Array
    triangle_valid: Array
    witness_weights: Array
    star_margin: Array
    triangulation_id: str = eqx.field(static=True)


class PolygonGeometryEvidence(StrictModule):
    valid: Array
    minimum_edge_ratio: Array
    star_margin: Array
    area_partition_error: Array
    minimum_triangle_measure: Array

    @property
    def passed(self) -> Array:
        return jnp.all(self.valid)


class PolygonGeometry(StrictModule):
    vertices: Array
    edge_vectors: Array
    edge_lengths: Array
    outward_normals: Array
    areas: Array
    centroids: Array
    characteristic_lengths: Array
    diameters: Array
    triangle_points: Array
    triangle_measures: Array
    evidence: PolygonGeometryEvidence
    geometry_id: str = eqx.field(static=True)


class PolygonCubature(StrictModule):
    points: Array
    weights: Array
    degree: int = eqx.field(static=True)
    cubature_id: str = eqx.field(static=True)


def prepare_polygon_triangulation(
    coordinates: ArrayLike,
    cells: ArrayLike,
    /,
    *,
    policy: PolygonAdmissibilityPolicy | None = None,
) -> PolygonTriangulation:
    points = np.asarray(coordinates, dtype=np.float64)
    cells_ = np.asarray(cells, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Polygon coordinates must have shape (vertices, 2).")
    if cells_.ndim != 2 or cells_.shape[1] < 3:
        raise ValueError("Polygon cells must have shape (cells, arity >= 3).")
    selected = points[cells_]
    capacity = cells_.shape[1] - 2
    triangles = np.zeros((cells_.shape[0], capacity, 3), dtype=np.int32)
    valid = np.zeros((cells_.shape[0], capacity), dtype=np.bool_)
    witness_weights = np.zeros((cells_.shape[0], cells_.shape[1]), dtype=np.float64)
    star_margins = np.zeros((cells_.shape[0],), dtype=np.float64)
    policy_ = PolygonAdmissibilityPolicy() if policy is None else policy
    for cell in range(selected.shape[0]):
        polygon = selected[cell]
        _validate_simple_polygon(polygon)
        local_triangles = _ear_clip(polygon)
        for index, triangle in enumerate(local_triangles):
            triangles[cell, index] = triangle
            valid[cell, index] = True
        witness, star_margin = _kernel_witness(polygon)
        if star_margin < policy_.minimum_star_margin:
            raise ValueError("Polygon star-kernel margin is below policy.")
        augmented = np.concatenate((polygon.T, np.ones((1, polygon.shape[0]))), axis=0)
        target = np.concatenate((witness, np.ones((1,))))
        weights = np.linalg.lstsq(augmented, target, rcond=None)[0]
        witness_weights[cell] = weights
        star_margins[cell] = star_margin
    return PolygonTriangulation(
        local_triangles=jnp.asarray(triangles),
        triangle_valid=jnp.asarray(valid),
        witness_weights=jnp.asarray(witness_weights),
        star_margin=jnp.asarray(star_margins),
        triangulation_id=canonical_fingerprint(
            {
                "kind": "polygon-triangulation",
                "cells": array_tree_fingerprint(cells_),
                "triangles": array_tree_fingerprint(triangles),
                "valid": array_tree_fingerprint(valid),
                "witness_weights": array_tree_fingerprint(witness_weights),
                "policy": policy_.policy_id,
            }
        ),
    )


def evaluate_polygon_geometry(
    coordinates: ArrayLike,
    cells: ArrayLike,
    triangulation: PolygonTriangulation,
    /,
    *,
    policy: PolygonAdmissibilityPolicy | None = None,
    geometry_id: str = "polygon-geometry",
) -> PolygonGeometry:
    points = jnp.asarray(coordinates)
    cells_ = jnp.asarray(cells, dtype=jnp.int32)
    vertices = points[cells_]
    following = jnp.roll(vertices, -1, axis=1)
    edge_vectors = following - vertices
    edge_lengths = jnp.sqrt(jnp.sum(edge_vectors * edge_vectors, axis=-1))
    area2 = jnp.sum(_cross_2d(vertices, following), axis=1)
    areas = 0.5 * area2
    safe_area2 = jnp.where(area2 != 0.0, area2, 1.0)
    centroid_factor = _cross_2d(vertices, following)[..., None]
    centroids = jnp.sum((vertices + following) * centroid_factor, axis=1) / (
        3.0 * safe_area2[:, None]
    )
    characteristic = jnp.sqrt(jnp.maximum(areas, jnp.finfo(points.dtype).tiny))
    safe_lengths = jnp.maximum(edge_lengths, jnp.finfo(points.dtype).tiny)
    outward_normals = (
        jnp.stack((edge_vectors[..., 1], -edge_vectors[..., 0]), axis=-1)
        / safe_lengths[..., None]
    )
    pair_difference = vertices[:, :, None, :] - vertices[:, None, :, :]
    diameters = jnp.sqrt(
        jnp.max(jnp.sum(pair_difference * pair_difference, axis=-1), axis=(1, 2))
    )
    local_triangles = triangulation.local_triangles
    safe_triangles = jnp.where(
        triangulation.triangle_valid[..., None], local_triangles, 0
    )
    cell_indices = jnp.arange(vertices.shape[0])[:, None, None]
    triangle_points = vertices[cell_indices, safe_triangles]
    first = triangle_points[:, :, 1] - triangle_points[:, :, 0]
    second = triangle_points[:, :, 2] - triangle_points[:, :, 0]
    triangle_measures = 0.5 * _cross_2d(first, second)
    triangle_measures = jnp.where(triangulation.triangle_valid, triangle_measures, 0.0)
    witness = ein.contract("cv,cvd->cd", triangulation.witness_weights, vertices)
    witness_delta = witness[:, None, :] - vertices
    inward = _cross_2d(edge_vectors, witness_delta) / safe_lengths
    runtime_star_margin = jnp.min(inward, axis=1) / characteristic
    edge_ratio = jnp.min(edge_lengths, axis=1) / jnp.maximum(
        diameters, jnp.finfo(points.dtype).tiny
    )
    area_error = jnp.abs(jnp.sum(triangle_measures, axis=1) - areas)
    active_triangle_measure = jnp.where(
        triangulation.triangle_valid,
        triangle_measures,
        jnp.asarray(jnp.inf, dtype=points.dtype),
    )
    minimum_triangle = jnp.min(active_triangle_measure, axis=1)
    policy_ = PolygonAdmissibilityPolicy() if policy is None else policy
    tolerance = 512.0 * jnp.finfo(points.dtype).eps * jnp.maximum(areas, 1.0)
    valid = (
        (areas > 0.0)
        & jnp.all(edge_lengths > 0.0, axis=1)
        & (minimum_triangle > 0.0)
        & (area_error <= tolerance)
        & (runtime_star_margin >= policy_.minimum_star_margin)
        & (edge_ratio >= policy_.minimum_edge_ratio)
    )
    evidence = PolygonGeometryEvidence(
        valid=valid,
        minimum_edge_ratio=edge_ratio,
        star_margin=runtime_star_margin,
        area_partition_error=area_error,
        minimum_triangle_measure=minimum_triangle,
    )
    return PolygonGeometry(
        vertices=vertices,
        edge_vectors=edge_vectors,
        edge_lengths=edge_lengths,
        outward_normals=outward_normals,
        areas=areas,
        centroids=centroids,
        characteristic_lengths=characteristic,
        diameters=diameters,
        triangle_points=triangle_points,
        triangle_measures=triangle_measures,
        evidence=evidence,
        geometry_id=canonical_fingerprint(
            {
                "kind": str(geometry_id),
                "triangulation": triangulation.triangulation_id,
                "coordinate_shape": list(points.shape),
                "coordinate_dtype": str(points.dtype),
            }
        ),
    )


def polygon_cubature(
    geometry: PolygonGeometry,
    triangulation: PolygonTriangulation,
    degree: int,
    /,
) -> PolygonCubature:
    from ..integration import (
        GaussLegendreRule,
        reference_rule_data,
        ReferenceTriangleRule,
    )

    degree_ = int(degree)
    if degree_ < 0:
        raise ValueError("Polygon cubature degree must be nonnegative.")
    order = max(2, (degree_ + 3) // 2)
    data = reference_rule_data(ReferenceTriangleRule(GaussLegendreRule(order)))
    reference = jnp.asarray(data.points)
    reference_weights = jnp.asarray(data.weights)
    triangles = geometry.triangle_points
    first = triangles[:, :, 0]
    axis_one = triangles[:, :, 1] - first
    axis_two = triangles[:, :, 2] - first
    points = (
        first[:, :, None, :]
        + reference[None, None, :, 0, None] * axis_one[:, :, None, :]
        + reference[None, None, :, 1, None] * axis_two[:, :, None, :]
    )
    jacobian = 2.0 * geometry.triangle_measures
    weights = jacobian[:, :, None] * reference_weights[None, None, :]
    weights = jnp.where(triangulation.triangle_valid[..., None], weights, 0.0)
    return PolygonCubature(
        points=points.reshape((points.shape[0], -1, 2)),
        weights=weights.reshape((weights.shape[0], -1)),
        degree=degree_,
        cubature_id=canonical_fingerprint(
            {
                "kind": "polygon-cubature",
                "geometry": geometry.geometry_id,
                "degree": degree_,
                "rule": type(data).__name__,
                "point_count": points.shape[1] * points.shape[2],
            }
        ),
    )


__all__ = [
    "PolygonAdmissibilityPolicy",
    "PolygonCubature",
    "PolygonGeometry",
    "PolygonGeometryEvidence",
    "PolygonTriangulation",
    "PolyhedralFaceTriangulation",
    "evaluate_polygon_geometry",
    "polygon_cubature",
    "prepare_polygon_triangulation",
    "prepare_polyhedral_face_triangulation",
]
