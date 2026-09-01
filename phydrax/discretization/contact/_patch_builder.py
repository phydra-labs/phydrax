#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._interface import ContactInterfacePlan


class ContactPatchBuildEvidence(StrictModule):
    overlapping_face_pairs: Array
    quadrature_count: Array
    overflow_count: Array
    total_measure: Array
    finite: Array
    complete: Array
    successful: Array


class ContactPatchBuildResult(StrictModule):
    interface: ContactInterfacePlan
    evidence: ContactPatchBuildEvidence


def _signed_area(polygon):
    if len(polygon) < 3:
        return 0.0
    value = 0.0
    for index, point in enumerate(polygon):
        other = polygon[(index + 1) % len(polygon)]
        value += point[0] * other[1] - point[1] * other[0]
    return 0.5 * value


def _inside(point, first, second, orientation, tolerance):
    cross = (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )
    return orientation * cross >= -tolerance


def _line_intersection(subject_first, subject_second, clip_first, clip_second):
    subject_direction = subject_second - subject_first
    clip_direction = clip_second - clip_first
    denominator = (
        subject_direction[0] * clip_direction[1]
        - subject_direction[1] * clip_direction[0]
    )
    if abs(denominator) <= np.finfo(float).eps:
        return 0.5 * (subject_first + subject_second)
    relative = clip_first - subject_first
    parameter = (
        relative[0] * clip_direction[1] - relative[1] * clip_direction[0]
    ) / denominator
    return subject_first + parameter * subject_direction


def _clip_polygon(subject, clip, tolerance):
    output = [np.asarray(point, dtype=float) for point in subject]
    orientation = 1.0 if _signed_area(clip) >= 0.0 else -1.0
    for index, clip_first in enumerate(clip):
        clip_second = clip[(index + 1) % len(clip)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        previous_inside = _inside(
            previous, clip_first, clip_second, orientation, tolerance
        )
        for current in input_polygon:
            current_inside = _inside(
                current, clip_first, clip_second, orientation, tolerance
            )
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, clip_first, clip_second)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, clip_first, clip_second)
                )
            previous = current
            previous_inside = current_inside
    return output


def _barycentric(point, triangle, tolerance):
    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    relative = point - triangle[0]
    denominator = first[0] * second[1] - first[1] * second[0]
    if abs(denominator) <= tolerance:
        return None
    weight_second = (relative[0] * second[1] - relative[1] * second[0]) / denominator
    weight_third = (first[0] * relative[1] - first[1] * relative[0]) / denominator
    return np.asarray((1.0 - weight_second - weight_third, weight_second, weight_third))


def build_triangle_mortar_interface(
    plus_positions: ArrayLike,
    plus_faces: ArrayLike,
    minus_positions: ArrayLike,
    minus_faces: ArrayLike,
    /,
    *,
    capacity: int,
    tolerance: float = 1.0e-10,
) -> ContactPatchBuildResult:
    """Build coplanar projected triangle-overlap quadrature.

    Each clipped polygon is fan-triangulated and integrated by one centroid rule
    per fan triangle. This builder is deterministic and fail-closed on capacity
    overflow.
    """
    plus = np.asarray(plus_positions, dtype=float)
    minus = np.asarray(minus_positions, dtype=float)
    plus_topology = np.asarray(plus_faces)
    minus_topology = np.asarray(minus_faces)
    count = int(capacity)
    tolerance_ = float(tolerance)
    if plus.ndim != 2 or plus.shape[1] != 3 or minus.ndim != 2 or minus.shape[1] != 3:
        raise ValueError("Triangle mortar builder requires three-dimensional positions.")
    if (
        plus_topology.ndim != 2
        or plus_topology.shape[1:] != (3,)
        or minus_topology.ndim != 2
        or minus_topology.shape[1:] != (3,)
    ):
        raise ValueError("Triangle mortar builder requires triangle connectivity.")
    if count <= 0 or tolerance_ <= 0.0:
        raise ValueError("Triangle mortar capacity/tolerance is invalid.")
    records = []
    overlapping_pairs = 0
    for plus_face_index, plus_face in enumerate(plus_topology):
        plus_triangle_3d = plus[plus_face]
        normal = np.cross(
            plus_triangle_3d[1] - plus_triangle_3d[0],
            plus_triangle_3d[2] - plus_triangle_3d[0],
        )
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= tolerance_:
            continue
        normal = normal / normal_norm
        drop_axis = int(np.argmax(np.abs(normal)))
        keep = [axis for axis in range(3) if axis != drop_axis]
        plus_triangle = plus_triangle_3d[:, keep]
        for minus_face_index, minus_face in enumerate(minus_topology):
            minus_triangle_3d = minus[minus_face]
            minus_triangle = minus_triangle_3d[:, keep]
            polygon = _clip_polygon(list(plus_triangle), list(minus_triangle), tolerance_)
            if len(polygon) < 3 or abs(_signed_area(polygon)) <= tolerance_:
                continue
            overlapping_pairs += 1
            for local_index in range(1, len(polygon) - 1):
                triangle = np.asarray(
                    (polygon[0], polygon[local_index], polygon[local_index + 1])
                )
                area = abs(_signed_area(list(triangle)))
                if area <= tolerance_:
                    continue
                centroid = triangle.mean(axis=0)
                plus_weights = _barycentric(centroid, plus_triangle, tolerance_)
                minus_weights = _barycentric(centroid, minus_triangle, tolerance_)
                if plus_weights is None or minus_weights is None:
                    continue
                key = (
                    plus_face_index * max(1, minus_topology.shape[0]) + minus_face_index
                ) * 16 + local_index
                records.append(
                    (
                        plus_face.copy(),
                        plus_weights,
                        minus_face.copy(),
                        minus_weights,
                        normal.copy(),
                        area,
                        key,
                    )
                )
    actual = len(records)
    overflow = max(actual - count, 0)
    plus_indices = np.zeros((count, 3), dtype=np.int32)
    plus_weights = np.full((count, 3), 1.0 / 3.0)
    minus_indices = np.zeros((count, 3), dtype=np.int32)
    minus_weights = np.full((count, 3), 1.0 / 3.0)
    normals = np.zeros((count, 3), dtype=float)
    normals[:, 2] = 1.0
    measures = np.zeros((count,), dtype=float)
    keys = np.arange(count, dtype=np.int64)
    valid = np.zeros((count,), dtype=bool)
    for slot, record in enumerate(records[:count]):
        (
            plus_indices[slot],
            plus_weights[slot],
            minus_indices[slot],
            minus_weights[slot],
            normals[slot],
            measures[slot],
            keys[slot],
        ) = record
    if overflow == 0:
        valid[:actual] = True
    interface = ContactInterfacePlan(
        plus_indices,
        plus_weights,
        minus_indices,
        minus_weights,
        normals,
        measures,
        plus_node_count=plus.shape[0],
        minus_node_count=minus.shape[0],
        route_keys=keys,
        valid=valid,
    )
    finite = (
        np.all(np.isfinite(plus))
        and np.all(np.isfinite(minus))
        and np.all(np.isfinite(measures))
    )
    complete = finite and overflow == 0
    evidence = ContactPatchBuildEvidence(
        jnp.asarray(overlapping_pairs, dtype=jnp.int32),
        jnp.asarray(actual, dtype=jnp.int32),
        jnp.asarray(overflow, dtype=jnp.int32),
        jnp.asarray(measures.sum()),
        jnp.asarray(finite),
        jnp.asarray(complete),
        jnp.asarray(complete),
    )
    return ContactPatchBuildResult(interface, evidence)


__all__ = [
    "ContactPatchBuildEvidence",
    "ContactPatchBuildResult",
    "build_triangle_mortar_interface",
]
