#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ._contracts import ClosestPointResult


def _point_tolerance(points: Array, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(points), axis=-1), 1.0)
    return 64.0 * jnp.finfo(points.dtype).eps * scale


def radial_closest_point(
    points: Array,
    center: Array,
    radius: Array,
    /,
    *,
    represented_geometry_id: str,
    physical_geometry_id: str | None = None,
) -> ClosestPointResult:
    """Return the exact closest-point map for a circle or sphere boundary."""
    points_ = jnp.asarray(points, dtype=float)
    center_ = jnp.asarray(center, dtype=points_.dtype)
    radius_ = jnp.asarray(radius, dtype=points_.dtype).reshape(())
    if points_.ndim == 0 or center_.shape != points_.shape[-1:]:
        raise ValueError("Radial closest-point inputs have incompatible dimensions.")
    relative = points_ - center_
    distance_from_center = jnp.linalg.norm(relative, axis=-1)
    tolerance = _point_tolerance(points_)
    unique = distance_from_center > tolerance
    safe_distance = jnp.where(unique, distance_from_center, 1.0)
    normal = relative / safe_distance[..., None]
    fallback = jnp.zeros_like(relative).at[..., 0].set(1.0)
    normal = jnp.where(unique[..., None], normal, fallback)
    closest = center_ + radius_ * normal
    coordinate = distance_from_center - radius_
    inward_margin = radius_ + jnp.minimum(coordinate, 0.0)
    margin = jnp.where(unique, jnp.maximum(inward_margin, 0.0), 0.0)
    leading = points_.shape[:-1]
    return ClosestPointResult(
        closest_point=closest,
        normal_coordinate=coordinate,
        oriented_normal=normal,
        source_entity_id=jnp.zeros(leading, dtype=jnp.int32),
        unique=unique,
        regular=unique,
        margin=margin,
        normal_coordinate_valid=unique,
        represented_geometry_id=represented_geometry_id,
        physical_geometry_id=(
            represented_geometry_id
            if physical_geometry_id is None
            else physical_geometry_id
        ),
        exact_to_physical=True,
    )


def box_closest_point(
    points: Array,
    center: Array,
    size: Array,
    /,
    *,
    represented_geometry_id: str,
    physical_geometry_id: str | None = None,
) -> ClosestPointResult:
    """Return the exact closest point on an axis-aligned box boundary.

    Projection is reported as regular only on the relative interior of one face.
    Edges, corners, and interior medial-axis ties remain valid closest points but
    are deliberately excluded from classical normal-collar certificates.
    """
    points_ = jnp.asarray(points, dtype=float)
    center_ = jnp.asarray(center, dtype=points_.dtype)
    size_ = jnp.asarray(size, dtype=points_.dtype)
    if points_.ndim == 0 or center_.shape != points_.shape[-1:]:
        raise ValueError("Box closest-point inputs have incompatible dimensions.")
    if size_.shape != center_.shape:
        raise ValueError("Box size must match the center dimension.")
    half = 0.5 * size_
    relative = points_ - center_
    absolute = jnp.abs(relative)
    outside_axes = absolute > half
    outside = jnp.any(outside_axes, axis=-1)
    clipped = jnp.clip(relative, -half, half)

    gaps = half - absolute
    inside_axis = jnp.argmin(gaps, axis=-1)
    selected_gap = jnp.take_along_axis(gaps, inside_axis[..., None], axis=-1)[..., 0]
    selected_coordinate = jnp.take_along_axis(relative, inside_axis[..., None], axis=-1)[
        ..., 0
    ]
    selected_sign = jnp.where(selected_coordinate < 0.0, -1.0, 1.0)
    axis_selector = jnp.arange(points_.shape[-1]) == inside_axis[..., None]
    inside_face_coordinate = selected_sign * jnp.take(half, inside_axis)
    inside_closest = jnp.where(
        axis_selector,
        inside_face_coordinate[..., None],
        relative,
    )
    closest_relative = jnp.where(outside[..., None], clipped, inside_closest)
    closest = center_ + closest_relative

    difference = relative - closest_relative
    unsigned = jnp.linalg.norm(difference, axis=-1)
    coordinate = jnp.where(outside, unsigned, -selected_gap)
    active_faces = jnp.isclose(
        jnp.abs(closest_relative),
        half,
        rtol=64.0 * jnp.finfo(points_.dtype).eps,
        atol=_point_tolerance(points_)[..., None],
    )
    active_count = jnp.sum(active_faces, axis=-1)
    normal = jnp.sign(closest_relative) * active_faces.astype(points_.dtype)
    normal_norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / jnp.maximum(normal_norm, jnp.finfo(points_.dtype).eps)

    sorted_gaps = jnp.sort(gaps, axis=-1)
    interior_tie_margin = (
        sorted_gaps[..., 1] - sorted_gaps[..., 0]
        if points_.shape[-1] > 1
        else jnp.full(points_.shape[:-1], jnp.inf, dtype=points_.dtype)
    )
    tolerance = _point_tolerance(points_)
    unique = outside | (interior_tie_margin > tolerance)
    regular = unique & (active_count == 1)
    tangential_margin = jnp.min(
        jnp.where(active_faces, jnp.inf, half - jnp.abs(closest_relative)), axis=-1
    )
    margin = jnp.where(
        regular,
        jnp.maximum(
            jnp.minimum(
                tangential_margin,
                jnp.where(outside, tangential_margin, interior_tie_margin),
            ),
            0.0,
        ),
        0.0,
    )
    face_axis = jnp.argmax(active_faces, axis=-1).astype(jnp.int32)
    face_sign = jnp.take_along_axis(closest_relative, face_axis[..., None], axis=-1)[
        ..., 0
    ]
    entity = 2 * face_axis + (face_sign >= 0.0).astype(jnp.int32)
    physical = (
        represented_geometry_id if physical_geometry_id is None else physical_geometry_id
    )
    return ClosestPointResult(
        closest_point=closest,
        normal_coordinate=coordinate,
        oriented_normal=normal,
        source_entity_id=entity,
        unique=unique,
        regular=regular,
        margin=margin,
        normal_coordinate_valid=regular,
        represented_geometry_id=represented_geometry_id,
        physical_geometry_id=physical,
        exact_to_physical=True,
    )


def segment_query_evidence(
    points: Array,
    closest_by_segment: Array,
    segment_coordinates: Array,
    selected_segment: Array,
    segment_lengths: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Certify one regular closest segment without inferring loop topology."""
    points_ = jnp.asarray(points)
    distance_sq = jnp.sum((points_[..., None, :] - closest_by_segment) ** 2, axis=-1)
    sorted_distance = jnp.sort(distance_sq, axis=-1)
    second = (
        sorted_distance[..., 1]
        if distance_sq.shape[-1] > 1
        else jnp.full(distance_sq.shape[:-1], jnp.inf, dtype=distance_sq.dtype)
    )
    first = sorted_distance[..., 0]
    selected_coordinate = jnp.take_along_axis(
        segment_coordinates, selected_segment[..., None], axis=-1
    )[..., 0]
    selected_length = jnp.take(segment_lengths, selected_segment)
    feature_margin = (
        jnp.minimum(selected_coordinate, 1.0 - selected_coordinate) * selected_length
    )
    tolerance = _point_tolerance(points_)
    distance_gap = jnp.sqrt(second) - jnp.sqrt(first)
    unique = (distance_gap > tolerance) & (feature_margin > tolerance)
    regular = unique
    margin = jnp.where(
        regular, jnp.maximum(jnp.minimum(distance_gap, feature_margin), 0.0), 0.0
    )
    return unique, regular, margin


def triangle_query_evidence(
    points: Array,
    triangles: Array,
    closest_by_face: Array,
    selected_face: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Certify a unique regular closest point in one triangle interior."""
    points_ = jnp.asarray(points)
    distance_sq = jnp.sum((points_[..., None, :] - closest_by_face) ** 2, axis=-1)
    sorted_distance = jnp.sort(distance_sq, axis=-1)
    second = (
        sorted_distance[..., 1]
        if distance_sq.shape[-1] > 1
        else jnp.full(distance_sq.shape[:-1], jnp.inf, dtype=distance_sq.dtype)
    )
    first = sorted_distance[..., 0]
    triangle = triangles[selected_face]
    closest = jnp.take_along_axis(
        closest_by_face, selected_face[..., None, None], axis=-2
    )[..., 0, :]
    first_edge = triangle[..., 1, :] - triangle[..., 0, :]
    second_edge = triangle[..., 2, :] - triangle[..., 0, :]
    relative = closest - triangle[..., 0, :]
    d00 = jnp.sum(first_edge * first_edge, axis=-1)
    d01 = jnp.sum(first_edge * second_edge, axis=-1)
    d11 = jnp.sum(second_edge * second_edge, axis=-1)
    d20 = jnp.sum(relative * first_edge, axis=-1)
    d21 = jnp.sum(relative * second_edge, axis=-1)
    denominator = d00 * d11 - d01 * d01
    second_bary = (d11 * d20 - d01 * d21) / denominator
    third_bary = (d00 * d21 - d01 * d20) / denominator
    barycentric = jnp.stack(
        (1.0 - second_bary - third_bary, second_bary, third_bary), axis=-1
    )
    edge_lengths = jnp.stack(
        (
            jnp.linalg.norm(first_edge, axis=-1),
            jnp.linalg.norm(second_edge - first_edge, axis=-1),
            jnp.linalg.norm(second_edge, axis=-1),
        ),
        axis=-1,
    )
    feature_margin = jnp.min(barycentric, axis=-1) * jnp.min(edge_lengths, axis=-1)
    distance_gap = jnp.sqrt(second) - jnp.sqrt(first)
    tolerance = _point_tolerance(points_)
    unique = (distance_gap > tolerance) & (feature_margin > tolerance)
    regular = unique
    margin = jnp.where(
        regular, jnp.maximum(jnp.minimum(distance_gap, feature_margin), 0.0), 0.0
    )
    return unique, regular, margin


def represented_mesh_closest_point(
    points: Array,
    /,
    *,
    closest_point: Array,
    distance: Array,
    normal: Array,
    source_entity_id: Array,
    inside: Array,
    unique: Array,
    regular: Array,
    margin: Array,
    represented_geometry_id: str,
    physical_geometry_id: str | None = None,
    exact_to_physical: bool = False,
) -> ClosestPointResult:
    """Package a mesh query without promoting proximity to topology evidence."""
    points_ = jnp.asarray(points, dtype=float)
    closest = jnp.asarray(closest_point, dtype=points_.dtype)
    distance_ = jnp.asarray(distance, dtype=points_.dtype)
    difference = points_ - closest
    on_boundary = distance_ <= _point_tolerance(points_)
    signed_distance = jnp.where(jnp.asarray(inside, dtype=bool), -distance_, distance_)
    boundary_coordinate = jnp.sum(difference * jnp.asarray(normal), axis=-1)
    coordinate = jnp.where(on_boundary, boundary_coordinate, signed_distance)
    return ClosestPointResult(
        closest_point=closest,
        normal_coordinate=coordinate,
        oriented_normal=normal,
        source_entity_id=source_entity_id,
        unique=unique,
        regular=regular,
        margin=margin,
        normal_coordinate_valid=jnp.asarray(regular, dtype=bool),
        represented_geometry_id=represented_geometry_id,
        physical_geometry_id=physical_geometry_id,
        exact_to_physical=exact_to_physical,
    )


__all__ = [
    "box_closest_point",
    "radial_closest_point",
    "represented_mesh_closest_point",
    "segment_query_evidence",
    "triangle_query_evidence",
]
