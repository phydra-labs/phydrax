#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class PointEdgeFeature(IntEnum):
    POINT_FIRST = 0
    POINT_SECOND = 1
    POINT_EDGE = 2


class PointTriangleFeature(IntEnum):
    POINT_FIRST = 0
    POINT_SECOND = 1
    POINT_THIRD = 2
    POINT_EDGE_FIRST_SECOND = 3
    POINT_EDGE_SECOND_THIRD = 4
    POINT_EDGE_THIRD_FIRST = 5
    POINT_FACE = 6


class EdgeEdgeFeature(IntEnum):
    FIRST_FIRST = 0
    FIRST_SECOND = 1
    SECOND_FIRST = 2
    SECOND_SECOND = 3
    EDGE_FIRST = 4
    EDGE_SECOND = 5
    FIRST_EDGE = 6
    SECOND_EDGE = 7
    EDGE_EDGE = 8


class ContactDistanceEvaluation(StrictModule):
    squared_distance: Array
    distance_vector: Array
    coefficients: Array
    left_witness: Array
    right_witness: Array
    normal: Array
    feature: Array
    feature_margin: Array
    nondegenerate: Array
    finite: Array


def _squared_norm(value: Array, /) -> Array:
    return jnp.sum(value * value, axis=-1)


def _safe_normal(displacement: Array, tolerance: float, /) -> tuple[Array, Array]:
    squared = _squared_norm(displacement)
    positive = squared > tolerance * tolerance
    distance = jnp.sqrt(jnp.maximum(squared, 0.0))
    normal = displacement / jnp.where(positive, distance, 1.0)[..., None]
    return jnp.where(positive[..., None], normal, 0.0), positive


def _selection_margin(values: Array, /) -> Array:
    ordered = jnp.sort(values, axis=-1)
    return jnp.maximum(ordered[..., 1] - ordered[..., 0], 0.0)


def point_point_distance(
    first: ArrayLike,
    second: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-12,
) -> ContactDistanceEvaluation:
    first_ = jnp.asarray(first)
    second_ = jnp.asarray(second, dtype=first_.dtype)
    if first_.shape != second_.shape or first_.ndim < 1 or first_.shape[-1] not in (2, 3):
        raise ValueError(
            "Point-point inputs must have matching trailing dimension two or three."
        )
    displacement = first_ - second_
    squared = _squared_norm(displacement)
    normal, positive = _safe_normal(displacement, tolerance)
    coefficients = jnp.zeros(first_.shape[:-1] + (4,), dtype=first_.dtype)
    coefficients = coefficients.at[..., 0].set(1.0)
    coefficients = coefficients.at[..., 1].set(-1.0)
    finite = (
        jnp.isfinite(squared)
        & jnp.all(jnp.isfinite(first_), axis=-1)
        & jnp.all(jnp.isfinite(second_), axis=-1)
    )
    return ContactDistanceEvaluation(
        squared,
        displacement,
        coefficients,
        first_,
        second_,
        normal,
        jnp.zeros(first_.shape[:-1], dtype=jnp.int32),
        jnp.where(positive, jnp.sqrt(squared), 0.0),
        positive,
        finite,
    )


def _point_segment_data(
    point: Array,
    first: Array,
    second: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    edge = second - first
    length_squared = _squared_norm(edge)
    nondegenerate = length_squared > tolerance * tolerance
    safe_length = jnp.where(nondegenerate, length_squared, 1.0)
    raw = jnp.sum((point - first) * edge, axis=-1) / safe_length
    coordinate = jnp.clip(raw, 0.0, 1.0)
    weights = jnp.stack((1.0 - coordinate, coordinate), axis=-1)
    witness = first + coordinate[..., None] * edge
    displacement = point - witness
    squared = _squared_norm(displacement)
    margin = jnp.minimum(jnp.abs(raw), jnp.abs(1.0 - raw))
    return witness, weights, squared, nondegenerate, margin, raw


def point_edge_distance(
    point: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-12,
) -> ContactDistanceEvaluation:
    point_ = jnp.asarray(point)
    first_ = jnp.asarray(first, dtype=point_.dtype)
    second_ = jnp.asarray(second, dtype=point_.dtype)
    if point_.shape != first_.shape or point_.shape != second_.shape:
        raise ValueError("Point-edge inputs must have matching shapes.")
    if point_.ndim < 1 or point_.shape[-1] not in (2, 3):
        raise ValueError("Point-edge inputs require trailing dimension two or three.")
    witness, weights, squared, nondegenerate, margin, raw = _point_segment_data(
        point_, first_, second_, tolerance
    )
    displacement = point_ - witness
    normal, positive = _safe_normal(displacement, tolerance)
    feature = jnp.where(
        raw <= 0.0,
        int(PointEdgeFeature.POINT_FIRST),
        jnp.where(
            raw >= 1.0,
            int(PointEdgeFeature.POINT_SECOND),
            int(PointEdgeFeature.POINT_EDGE),
        ),
    ).astype(jnp.int32)
    feature = jax.lax.stop_gradient(feature)
    coefficients = jnp.zeros(point_.shape[:-1] + (4,), dtype=point_.dtype)
    coefficients = coefficients.at[..., 0].set(1.0)
    coefficients = coefficients.at[..., 1].set(-weights[..., 0])
    coefficients = coefficients.at[..., 2].set(-weights[..., 1])
    finite = (
        jnp.isfinite(squared)
        & jnp.all(jnp.isfinite(point_), axis=-1)
        & jnp.all(jnp.isfinite(first_), axis=-1)
        & jnp.all(jnp.isfinite(second_), axis=-1)
    )
    return ContactDistanceEvaluation(
        squared,
        displacement,
        coefficients,
        point_,
        witness,
        normal,
        feature,
        jnp.maximum(jnp.minimum(margin, jnp.sqrt(jnp.maximum(squared, 0.0))), 0.0),
        nondegenerate & positive,
        finite,
    )


def point_triangle_distance(
    point: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    third: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-12,
) -> ContactDistanceEvaluation:
    point_ = jnp.asarray(point)
    first_ = jnp.asarray(first, dtype=point_.dtype)
    second_ = jnp.asarray(second, dtype=point_.dtype)
    third_ = jnp.asarray(third, dtype=point_.dtype)
    if (
        point_.shape != first_.shape
        or point_.shape != second_.shape
        or point_.shape != third_.shape
    ):
        raise ValueError("Point-triangle inputs must have matching shapes.")
    if point_.ndim < 1 or point_.shape[-1] != 3:
        raise ValueError("Point-triangle inputs require trailing dimension three.")

    edge_first = second_ - first_
    edge_second = third_ - first_
    face_cross = jnp.cross(edge_first, edge_second)
    area_squared = _squared_norm(face_cross)
    face_nondegenerate = area_squared > tolerance * tolerance
    safe_area_squared = jnp.where(face_nondegenerate, area_squared, 1.0)
    plane_parameter = jnp.sum((point_ - first_) * face_cross, axis=-1) / safe_area_squared
    face_witness = point_ - plane_parameter[..., None] * face_cross

    dot00 = _squared_norm(edge_first)
    dot01 = jnp.sum(edge_first * edge_second, axis=-1)
    dot11 = _squared_norm(edge_second)
    rhs0 = jnp.sum((face_witness - first_) * edge_first, axis=-1)
    rhs1 = jnp.sum((face_witness - first_) * edge_second, axis=-1)
    determinant = dot00 * dot11 - dot01 * dot01
    safe_determinant = jnp.where(face_nondegenerate, determinant, 1.0)
    second_weight = (dot11 * rhs0 - dot01 * rhs1) / safe_determinant
    third_weight = (dot00 * rhs1 - dot01 * rhs0) / safe_determinant
    first_weight = 1.0 - second_weight - third_weight
    face_weights = jnp.stack((first_weight, second_weight, third_weight), axis=-1)
    face_valid = face_nondegenerate & jnp.all(face_weights >= 0.0, axis=-1)

    witness_ab, weights_ab, squared_ab, valid_ab, margin_ab, raw_ab = _point_segment_data(
        point_, first_, second_, tolerance
    )
    witness_bc, weights_bc, squared_bc, valid_bc, margin_bc, raw_bc = _point_segment_data(
        point_, second_, third_, tolerance
    )
    witness_ca, weights_ca, squared_ca, valid_ca, margin_ca, raw_ca = _point_segment_data(
        point_, third_, first_, tolerance
    )
    zero = jnp.zeros_like(first_weight)
    weights_ab3 = jnp.stack((weights_ab[..., 0], weights_ab[..., 1], zero), axis=-1)
    weights_bc3 = jnp.stack((zero, weights_bc[..., 0], weights_bc[..., 1]), axis=-1)
    weights_ca3 = jnp.stack((weights_ca[..., 1], zero, weights_ca[..., 0]), axis=-1)
    face_squared = _squared_norm(point_ - face_witness)
    maximum = jnp.asarray(jnp.finfo(point_.dtype).max, dtype=point_.dtype)
    candidate_squared = jnp.stack(
        (
            jnp.where(face_valid, face_squared, maximum),
            jnp.where(valid_ab, squared_ab, maximum),
            jnp.where(valid_bc, squared_bc, maximum),
            jnp.where(valid_ca, squared_ca, maximum),
        ),
        axis=-1,
    )
    witnesses = jnp.stack((face_witness, witness_ab, witness_bc, witness_ca), axis=-2)
    weights = jnp.stack((face_weights, weights_ab3, weights_bc3, weights_ca3), axis=-2)
    selected = jax.lax.stop_gradient(jnp.argmin(candidate_squared, axis=-1))
    selector = jax.nn.one_hot(selected, 4, dtype=point_.dtype)
    witness = jnp.sum(selector[..., :, None] * witnesses, axis=-2)
    interpolation = jnp.sum(selector[..., :, None] * weights, axis=-2)
    squared = jnp.sum(selector * candidate_squared, axis=-1)

    edge_feature = jnp.stack(
        (
            jnp.full(raw_ab.shape, int(PointTriangleFeature.POINT_FACE), dtype=jnp.int32),
            jnp.where(
                raw_ab <= 0.0,
                int(PointTriangleFeature.POINT_FIRST),
                jnp.where(
                    raw_ab >= 1.0,
                    int(PointTriangleFeature.POINT_SECOND),
                    int(PointTriangleFeature.POINT_EDGE_FIRST_SECOND),
                ),
            ),
            jnp.where(
                raw_bc <= 0.0,
                int(PointTriangleFeature.POINT_SECOND),
                jnp.where(
                    raw_bc >= 1.0,
                    int(PointTriangleFeature.POINT_THIRD),
                    int(PointTriangleFeature.POINT_EDGE_SECOND_THIRD),
                ),
            ),
            jnp.where(
                raw_ca <= 0.0,
                int(PointTriangleFeature.POINT_THIRD),
                jnp.where(
                    raw_ca >= 1.0,
                    int(PointTriangleFeature.POINT_FIRST),
                    int(PointTriangleFeature.POINT_EDGE_THIRD_FIRST),
                ),
            ),
        ),
        axis=-1,
    )
    feature = jnp.take_along_axis(edge_feature, selected[..., None], axis=-1)[..., 0]
    feature = jax.lax.stop_gradient(feature.astype(jnp.int32))
    coordinate_margins = jnp.stack(
        (
            jnp.min(jnp.maximum(face_weights, 0.0), axis=-1),
            margin_ab,
            margin_bc,
            margin_ca,
        ),
        axis=-1,
    )
    coordinate_margin = jnp.sum(selector * coordinate_margins, axis=-1)
    margin = jnp.minimum(coordinate_margin, _selection_margin(candidate_squared))
    displacement = point_ - witness
    normal, positive = _safe_normal(displacement, tolerance)
    coefficients = jnp.concatenate(
        (jnp.ones(interpolation.shape[:-1] + (1,), dtype=point_.dtype), -interpolation),
        axis=-1,
    )
    finite = (
        jnp.isfinite(squared)
        & jnp.all(jnp.isfinite(point_), axis=-1)
        & jnp.all(jnp.isfinite(first_), axis=-1)
        & jnp.all(jnp.isfinite(second_), axis=-1)
        & jnp.all(jnp.isfinite(third_), axis=-1)
    )
    return ContactDistanceEvaluation(
        squared,
        displacement,
        coefficients,
        point_,
        witness,
        normal,
        feature,
        jnp.maximum(margin, 0.0),
        face_nondegenerate & positive,
        finite,
    )


def edge_edge_distance(
    first_a: ArrayLike,
    second_a: ArrayLike,
    first_b: ArrayLike,
    second_b: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-12,
) -> ContactDistanceEvaluation:
    a0 = jnp.asarray(first_a)
    a1 = jnp.asarray(second_a, dtype=a0.dtype)
    b0 = jnp.asarray(first_b, dtype=a0.dtype)
    b1 = jnp.asarray(second_b, dtype=a0.dtype)
    if a0.shape != a1.shape or a0.shape != b0.shape or a0.shape != b1.shape:
        raise ValueError("Edge-edge inputs must have matching shapes.")
    if a0.ndim < 1 or a0.shape[-1] != 3:
        raise ValueError("Edge-edge inputs require trailing dimension three.")
    ua = a1 - a0
    ub = b1 - b0
    aa = _squared_norm(ua)
    bb = jnp.sum(ua * ub, axis=-1)
    cc = _squared_norm(ub)
    w = a0 - b0
    dd = jnp.sum(ua * w, axis=-1)
    ee = jnp.sum(ub * w, axis=-1)
    determinant = aa * cc - bb * bb
    edges_valid = (aa > tolerance * tolerance) & (cc > tolerance * tolerance)
    interior_valid = edges_valid & (determinant > tolerance * tolerance * aa * cc)
    safe_determinant = jnp.where(interior_valid, determinant, 1.0)
    s = (bb * ee - cc * dd) / safe_determinant
    t = (aa * ee - bb * dd) / safe_determinant
    interior_valid = interior_valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    left_interior = a0 + s[..., None] * ua
    right_interior = b0 + t[..., None] * ub
    squared_interior = _squared_norm(left_interior - right_interior)

    a0_b = _point_segment_data(a0, b0, b1, tolerance)
    a1_b = _point_segment_data(a1, b0, b1, tolerance)
    b0_a = _point_segment_data(b0, a0, a1, tolerance)
    b1_a = _point_segment_data(b1, a0, a1, tolerance)
    maximum = jnp.asarray(jnp.finfo(a0.dtype).max, dtype=a0.dtype)
    candidate_squared = jnp.stack(
        (
            jnp.where(interior_valid, squared_interior, maximum),
            jnp.where(a0_b[3], a0_b[2], maximum),
            jnp.where(a1_b[3], a1_b[2], maximum),
            jnp.where(b0_a[3], b0_a[2], maximum),
            jnp.where(b1_a[3], b1_a[2], maximum),
        ),
        axis=-1,
    )
    left_candidates = jnp.stack((left_interior, a0, a1, b0_a[0], b1_a[0]), axis=-2)
    right_candidates = jnp.stack((right_interior, a0_b[0], a1_b[0], b0, b1), axis=-2)
    selected = jax.lax.stop_gradient(jnp.argmin(candidate_squared, axis=-1))
    selector = jax.nn.one_hot(selected, 5, dtype=a0.dtype)
    left_witness = jnp.sum(selector[..., :, None] * left_candidates, axis=-2)
    right_witness = jnp.sum(selector[..., :, None] * right_candidates, axis=-2)
    squared = jnp.sum(selector * candidate_squared, axis=-1)

    zero = jnp.zeros_like(s)
    one = jnp.ones_like(s)
    interior_coefficients = jnp.stack((1.0 - s, s, -(1.0 - t), -t), axis=-1)
    a0_coefficients = jnp.stack((one, zero, -a0_b[1][..., 0], -a0_b[1][..., 1]), axis=-1)
    a1_coefficients = jnp.stack((zero, one, -a1_b[1][..., 0], -a1_b[1][..., 1]), axis=-1)
    b0_coefficients = jnp.stack((-b0_a[1][..., 0], -b0_a[1][..., 1], one, zero), axis=-1)
    b1_coefficients = jnp.stack((-b1_a[1][..., 0], -b1_a[1][..., 1], zero, one), axis=-1)
    candidate_coefficients = jnp.stack(
        (
            interior_coefficients,
            a0_coefficients,
            a1_coefficients,
            b0_coefficients,
            b1_coefficients,
        ),
        axis=-2,
    )
    coefficients = jnp.sum(selector[..., :, None] * candidate_coefficients, axis=-2)

    def endpoint_feature(raw, left_endpoint, right_first, right_second):
        return jnp.where(
            raw <= 0.0,
            right_first,
            jnp.where(raw >= 1.0, right_second, left_endpoint),
        )

    features = jnp.stack(
        (
            jnp.full(s.shape, int(EdgeEdgeFeature.EDGE_EDGE), dtype=jnp.int32),
            endpoint_feature(
                a0_b[5],
                int(EdgeEdgeFeature.FIRST_EDGE),
                int(EdgeEdgeFeature.FIRST_FIRST),
                int(EdgeEdgeFeature.FIRST_SECOND),
            ),
            endpoint_feature(
                a1_b[5],
                int(EdgeEdgeFeature.SECOND_EDGE),
                int(EdgeEdgeFeature.SECOND_FIRST),
                int(EdgeEdgeFeature.SECOND_SECOND),
            ),
            endpoint_feature(
                b0_a[5],
                int(EdgeEdgeFeature.EDGE_FIRST),
                int(EdgeEdgeFeature.FIRST_FIRST),
                int(EdgeEdgeFeature.SECOND_FIRST),
            ),
            endpoint_feature(
                b1_a[5],
                int(EdgeEdgeFeature.EDGE_SECOND),
                int(EdgeEdgeFeature.FIRST_SECOND),
                int(EdgeEdgeFeature.SECOND_SECOND),
            ),
        ),
        axis=-1,
    )
    feature = jnp.take_along_axis(features, selected[..., None], axis=-1)[..., 0]
    feature = jax.lax.stop_gradient(feature.astype(jnp.int32))
    coordinate_margin = jnp.stack(
        (
            jnp.minimum(
                jnp.minimum(jnp.abs(s), jnp.abs(1.0 - s)),
                jnp.minimum(jnp.abs(t), jnp.abs(1.0 - t)),
            ),
            a0_b[4],
            a1_b[4],
            b0_a[4],
            b1_a[4],
        ),
        axis=-1,
    )
    margin = jnp.minimum(
        jnp.sum(selector * coordinate_margin, axis=-1),
        _selection_margin(candidate_squared),
    )
    displacement = left_witness - right_witness
    normal, positive = _safe_normal(displacement, tolerance)
    finite = (
        jnp.isfinite(squared)
        & jnp.all(jnp.isfinite(a0), axis=-1)
        & jnp.all(jnp.isfinite(a1), axis=-1)
        & jnp.all(jnp.isfinite(b0), axis=-1)
        & jnp.all(jnp.isfinite(b1), axis=-1)
    )
    return ContactDistanceEvaluation(
        squared,
        displacement,
        coefficients,
        left_witness,
        right_witness,
        normal,
        feature,
        jnp.maximum(margin, 0.0),
        edges_valid & positive,
        finite,
    )


def edge_edge_mollifier_threshold(
    rest_first_a: ArrayLike,
    rest_second_a: ArrayLike,
    rest_first_b: ArrayLike,
    rest_second_b: ArrayLike,
    /,
    *,
    relative_threshold: float = 1.0e-3,
) -> Array:
    a0 = jnp.asarray(rest_first_a)
    a1 = jnp.asarray(rest_second_a, dtype=a0.dtype)
    b0 = jnp.asarray(rest_first_b, dtype=a0.dtype)
    b1 = jnp.asarray(rest_second_b, dtype=a0.dtype)
    relative = float(relative_threshold)
    if not 0.0 < relative < 1.0:
        raise ValueError("relative_threshold must lie strictly between zero and one.")
    return relative * _squared_norm(a1 - a0) * _squared_norm(b1 - b0)


def edge_edge_mollifier(
    first_a: ArrayLike,
    second_a: ArrayLike,
    first_b: ArrayLike,
    second_b: ArrayLike,
    threshold: ArrayLike,
    /,
) -> tuple[Array, Array]:
    a0 = jnp.asarray(first_a)
    a1 = jnp.asarray(second_a, dtype=a0.dtype)
    b0 = jnp.asarray(first_b, dtype=a0.dtype)
    b1 = jnp.asarray(second_b, dtype=a0.dtype)
    threshold_ = jnp.asarray(threshold, dtype=a0.dtype)
    cross_squared = _squared_norm(jnp.cross(a1 - a0, b1 - b0))
    safe_threshold = jnp.where(threshold_ > 0.0, threshold_, 1.0)
    ratio = cross_squared / safe_threshold
    value = jnp.where(cross_squared < threshold_, ratio * (2.0 - ratio), 1.0)
    margin = jnp.abs(cross_squared - threshold_)
    return value, margin


def contact_tangent_basis(normal: ArrayLike, /) -> Array:
    normal_ = jnp.asarray(normal)
    if normal_.ndim < 1 or normal_.shape[-1] not in (2, 3):
        raise ValueError("Contact normals require trailing dimension two or three.")
    if normal_.shape[-1] == 2:
        tangent = jnp.stack((-normal_[..., 1], normal_[..., 0]), axis=-1)
        return tangent[..., :, None]
    axis_index = jax.lax.stop_gradient(jnp.argmin(jnp.abs(normal_), axis=-1))
    axis = jax.nn.one_hot(axis_index, 3, dtype=normal_.dtype)
    first = jnp.cross(normal_, axis)
    first_norm = jnp.sqrt(
        jnp.maximum(_squared_norm(first), jnp.finfo(normal_.dtype).tiny)
    )
    first = first / first_norm[..., None]
    second = jnp.cross(normal_, first)
    return jnp.stack((first, second), axis=-1)


__all__ = [
    "ContactDistanceEvaluation",
    "EdgeEdgeFeature",
    "PointEdgeFeature",
    "PointTriangleFeature",
    "contact_tangent_basis",
    "edge_edge_distance",
    "edge_edge_mollifier",
    "edge_edge_mollifier_threshold",
    "point_edge_distance",
    "point_point_distance",
    "point_triangle_distance",
]
