#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _cross_2d(left, right):
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _signed_area2(points: np.ndarray, /) -> float:
    return float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a, b, c, d, tolerance: float, /) -> bool:
    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    return (
        (first > tolerance and second < -tolerance)
        or (first < -tolerance and second > tolerance)
    ) and (
        (third > tolerance and fourth < -tolerance)
        or (third < -tolerance and fourth > tolerance)
    )


def _validate_simple_polygon(points: np.ndarray, /) -> None:
    count = points.shape[0]
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale * scale
    if _signed_area2(points) <= tolerance:
        raise ValueError("Polygon cells must be counter-clockwise with positive area.")
    edges = np.roll(points, -1, axis=0) - points
    if np.any(np.sum(edges * edges, axis=1) <= tolerance):
        raise ValueError("Polygon cells cannot contain zero-length edges.")
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first in (second, second_next) or first_next in (second, second_next):
                continue
            if _segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
                tolerance,
            ):
                raise ValueError(
                    "Polygon cells must be simple and non-self-intersecting."
                )


def _remove_collinear(points: np.ndarray, indices: list[int], /) -> list[int]:
    changed = True
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale * scale
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


def _inside_triangle(point, a, b, c, tolerance: float, /) -> bool:
    return (
        _orientation(a, b, point) >= -tolerance
        and _orientation(b, c, point) >= -tolerance
        and _orientation(c, a, point) >= -tolerance
    )


def _ear_clip(points: np.ndarray, /) -> tuple[tuple[int, int, int], ...]:
    remaining = _remove_collinear(points, list(range(points.shape[0])))
    triangles: list[tuple[int, int, int]] = []
    scale = max(float(np.max(np.abs(points))), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale * scale
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
        triangles.append(tuple(remaining))
    if not triangles:
        raise ValueError("Polygon triangulation produced no positive triangles.")
    return tuple(triangles)


def _clip_half_plane(
    polygon: list[np.ndarray], start: np.ndarray, stop: np.ndarray, /
) -> list[np.ndarray]:
    if not polygon:
        return []
    edge = stop - start

    def signed(point):
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
    ):
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
    points = np.asarray(coordinates, dtype=float)
    cells_ = np.asarray(cells, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Polygon coordinates must have shape (vertices, 2).")
    if cells_.ndim != 2 or cells_.shape[1] < 3:
        raise ValueError("Polygon cells must have shape (cells, arity >= 3).")
    selected = points[cells_]
    capacity = cells_.shape[1] - 2
    triangles = np.zeros((cells_.shape[0], capacity, 3), dtype=np.int32)
    valid = np.zeros((cells_.shape[0], capacity), dtype=bool)
    witness_weights = np.zeros((cells_.shape[0], cells_.shape[1]), dtype=float)
    star_margins = np.zeros((cells_.shape[0],), dtype=float)
    policy_ = PolygonAdmissibilityPolicy() if policy is None else policy
    for cell, polygon in enumerate(selected):
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
    witness = oe.contract("cv,cvd->cd", triangulation.witness_weights, vertices)
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
                "point_count": int(points.shape[1] * points.shape[2]),
            }
        ),
    )


__all__ = [
    "PolygonAdmissibilityPolicy",
    "PolygonCubature",
    "PolygonGeometry",
    "PolygonGeometryEvidence",
    "PolygonTriangulation",
    "evaluate_polygon_geometry",
    "polygon_cubature",
    "prepare_polygon_triangulation",
]
