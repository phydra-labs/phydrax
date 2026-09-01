#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....integration import GaussLegendreRule, reference_rule_data, ReferenceTriangleRule


_PAIR_NAMES = ("coincident", "shared-edge", "shared-vertex", "near")


class _SurfacePairData3D(StrictModule, NonTrainableState):
    """Sparse non-regular pair corrections and regular face quadrature."""

    exception_keys: Array
    targets: Array
    sources: Array
    classes: Array
    values: Array
    regular_points: Array
    regular_weights: Array
    counts: tuple[int, int, int, int, int] = eqx.field(static=True)
    maximum_errors: Array
    evaluations: Array
    preparation_workspace_bytes: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)


def _gauss01(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _duffy_rule(order: int, adjacency: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standard-triangle product rules adapted from Bempp-cl.

    See ``LICENSES/BEMPP-CL-MIT.txt``. The transformations implement the
    Sauter--Schwab/Duffy coincident, edge-adjacent, and vertex-adjacent regions.
    """
    nodes, weights_1d = _gauss01(order)
    tensor_points = np.asarray([(x, y) for y in nodes for x in nodes], dtype=float)
    tensor_weights = np.asarray(
        [wy * wx for wy in weights_1d for wx in weights_1d], dtype=float
    )
    test_points: list[tuple[float, float]] = []
    trial_points: list[tuple[float, float]] = []
    weights: list[float] = []
    for test_index, (xsi, eta1) in enumerate(tensor_points):
        for trial_index, (eta2, eta3) in enumerate(tensor_points):
            base = tensor_weights[test_index] * tensor_weights[trial_index]
            eta12 = eta1 * eta2
            eta123 = eta12 * eta3
            if adjacency == "coincident":
                weight = base * xsi**3 * eta1**2 * eta2
                regions = (
                    (
                        (xsi, xsi * (1.0 - eta1 + eta12)),
                        (xsi * (1.0 - eta123), xsi * (1.0 - eta1)),
                        weight,
                    ),
                    (
                        (xsi * (1.0 - eta123), xsi * (1.0 - eta1)),
                        (xsi, xsi * (1.0 - eta1 + eta12)),
                        weight,
                    ),
                    (
                        (xsi, xsi * (eta1 - eta12 + eta123)),
                        (xsi * (1.0 - eta12), xsi * (eta1 - eta12)),
                        weight,
                    ),
                    (
                        (xsi * (1.0 - eta12), xsi * (eta1 - eta12)),
                        (xsi, xsi * (eta1 - eta12 + eta123)),
                        weight,
                    ),
                    (
                        (xsi * (1.0 - eta123), xsi * (eta1 - eta123)),
                        (xsi, xsi * (eta1 - eta12)),
                        weight,
                    ),
                    (
                        (xsi, xsi * (eta1 - eta12)),
                        (xsi * (1.0 - eta123), xsi * (eta1 - eta123)),
                        weight,
                    ),
                )
            elif adjacency == "shared-edge":
                weight = base * xsi**3 * eta1**2
                regions = (
                    (
                        (xsi, xsi * eta1 * eta3),
                        (xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2)),
                        weight,
                    ),
                    (
                        (xsi, xsi * eta1),
                        (xsi * (1.0 - eta123), xsi * eta12 * (1.0 - eta3)),
                        weight * eta2,
                    ),
                    (
                        (xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2)),
                        (xsi, xsi * eta123),
                        weight * eta2,
                    ),
                    (
                        (xsi * (1.0 - eta123), xsi * eta12 * (1.0 - eta3)),
                        (xsi, xsi * eta1),
                        weight * eta2,
                    ),
                    (
                        (xsi * (1.0 - eta123), xsi * eta1 * (1.0 - eta2 * eta3)),
                        (xsi, xsi * eta12),
                        weight * eta2,
                    ),
                )
            elif adjacency == "shared-vertex":
                weight = base * xsi**3 * eta2
                regions = (
                    ((xsi, xsi * eta1), (xsi * eta2, xsi * eta2 * eta3), weight),
                    ((xsi * eta2, xsi * eta2 * eta3), (xsi, xsi * eta1), weight),
                )
            else:
                raise ValueError("Unknown singular pair adjacency.")
            for test, trial, region_weight in regions:
                test_points.append((test[0] - test[1], test[1]))
                trial_points.append((trial[0] - trial[1], trial[1]))
                weights.append(region_weight)
    return (
        np.asarray(test_points, dtype=float),
        np.asarray(trial_points, dtype=float),
        np.asarray(weights, dtype=float),
    )


def _remap_vertex(points: np.ndarray, vertex: int) -> np.ndarray:
    if vertex == 0:
        return points
    result = np.empty_like(points)
    if vertex == 1:
        result[:, 0] = 1.0 - points[:, 0] - points[:, 1]
        result[:, 1] = points[:, 1]
        return result
    if vertex == 2:
        result[:, 0] = points[:, 0]
        result[:, 1] = 1.0 - points[:, 0] - points[:, 1]
        return result
    raise ValueError("Triangle vertex must be 0, 1, or 2.")


def _remap_edge(points: np.ndarray, first: int, second: int) -> np.ndarray:
    if first == second or first not in (0, 1, 2) or second not in (0, 1, 2):
        raise ValueError("Shared edge requires two distinct local vertices.")
    reference_vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    third = 3 - first - second
    vertices = reference_vertices[[first, second, third]]
    affine = np.stack((vertices[1] - vertices[0], vertices[2] - vertices[0]), axis=1)
    return points @ affine.T + vertices[0]


def _map_triangle(triangle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (
        triangle[0]
        + reference[:, :1] * (triangle[1] - triangle[0])
        + reference[:, 1:] * (triangle[2] - triangle[0])
    )


def _surface_jacobian(triangle: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    )


def _kernel_sum(
    test_triangle: np.ndarray,
    trial_triangle: np.ndarray,
    test_reference: np.ndarray,
    trial_reference: np.ndarray,
    reference_weights: np.ndarray,
) -> float:
    test_points = _map_triangle(test_triangle, test_reference)
    trial_points = _map_triangle(trial_triangle, trial_reference)
    distance = np.linalg.norm(test_points - trial_points, axis=1)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("Transformed Galerkin quadrature produced a singular point.")
    return float(
        np.sum(reference_weights / (4.0 * np.pi * distance))
        * _surface_jacobian(test_triangle)
        * _surface_jacobian(trial_triangle)
    )


def _singular_pair_value(
    triangles: np.ndarray,
    faces: np.ndarray,
    target: int,
    source: int,
    adjacency: str,
    order: int,
) -> float:
    test_reference, trial_reference, weights = _duffy_rule(order, adjacency)
    if adjacency == "shared-edge":
        shared = sorted(set(map(int, faces[target])) & set(map(int, faces[source])))
        test_local = tuple(
            int(np.flatnonzero(faces[target] == value)[0]) for value in shared
        )
        source_local = tuple(
            int(np.flatnonzero(faces[source] == value)[0]) for value in shared
        )
        test_reference = _remap_edge(test_reference, *test_local)
        trial_reference = _remap_edge(trial_reference, *source_local)
    elif adjacency == "shared-vertex":
        shared = tuple(set(map(int, faces[target])) & set(map(int, faces[source])))
        if len(shared) != 1:
            raise ValueError("Vertex-adjacent faces must share one vertex.")
        test_local = int(np.flatnonzero(faces[target] == shared[0])[0])
        source_local = int(np.flatnonzero(faces[source] == shared[0])[0])
        test_reference = _remap_vertex(test_reference, test_local)
        trial_reference = _remap_vertex(trial_reference, source_local)
    return _kernel_sum(
        triangles[target],
        triangles[source],
        test_reference,
        trial_reference,
        weights,
    )


def _regular_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    data = reference_rule_data(ReferenceTriangleRule(GaussLegendreRule(int(order))))
    return np.asarray(data.points, dtype=float), np.asarray(data.weights, dtype=float)


def _regular_pair_value(
    test_triangle: np.ndarray,
    trial_triangle: np.ndarray,
    order: int,
) -> float:
    points, weights = _regular_rule(order)
    test_points = _map_triangle(test_triangle, points)
    trial_points = _map_triangle(trial_triangle, points)
    difference = test_points[:, None, :] - trial_points[None, :, :]
    distance = np.linalg.norm(difference, axis=-1)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("Regular Galerkin quadrature encountered a singular point.")
    return float(
        np.sum(weights[:, None] * weights[None, :] / (4.0 * np.pi * distance))
        * _surface_jacobian(test_triangle)
        * _surface_jacobian(trial_triangle)
    )


def _subdivide_triangle(triangle: np.ndarray) -> tuple[np.ndarray, ...]:
    ab = 0.5 * (triangle[0] + triangle[1])
    bc = 0.5 * (triangle[1] + triangle[2])
    ca = 0.5 * (triangle[2] + triangle[0])
    return (
        np.asarray((triangle[0], ab, ca)),
        np.asarray((ab, triangle[1], bc)),
        np.asarray((ca, bc, triangle[2])),
        np.asarray((ab, bc, ca)),
    )


def _diameter(triangle: np.ndarray) -> float:
    return max(
        float(np.linalg.norm(triangle[1] - triangle[0])),
        float(np.linalg.norm(triangle[2] - triangle[1])),
        float(np.linalg.norm(triangle[0] - triangle[2])),
    )


def _near_pair_value(
    test_triangle: np.ndarray,
    trial_triangle: np.ndarray,
    *,
    low_order: int,
    high_order: int,
    max_depth: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    depth: int = 0,
) -> tuple[float, float, int]:
    low = _regular_pair_value(test_triangle, trial_triangle, low_order)
    high = _regular_pair_value(test_triangle, trial_triangle, high_order)
    error = abs(high - low)
    evaluations = low_order**4 + high_order**4
    threshold = absolute_tolerance + relative_tolerance * abs(high)
    if error <= threshold:
        return high, error, evaluations
    if depth >= max_depth:
        raise ValueError("Near-pair quadrature exhausted its subdivision capacity.")
    if _diameter(test_triangle) >= _diameter(trial_triangle):
        children = tuple(
            (child, trial_triangle) for child in _subdivide_triangle(test_triangle)
        )
    else:
        children = tuple(
            (test_triangle, child) for child in _subdivide_triangle(trial_triangle)
        )
    values = [
        _near_pair_value(
            left,
            right,
            low_order=low_order,
            high_order=high_order,
            max_depth=max_depth,
            absolute_tolerance=absolute_tolerance / 4.0,
            relative_tolerance=relative_tolerance,
            depth=depth + 1,
        )
        for left, right in children
    ]
    return (
        sum(value[0] for value in values),
        sum(value[1] for value in values),
        evaluations + sum(value[2] for value in values),
    )


def _regular_face_quadrature(
    triangles: np.ndarray,
    order: int,
) -> tuple[Array, Array]:
    points, weights = _regular_rule(order)
    mapped = np.stack([_map_triangle(triangle, points) for triangle in triangles])
    jacobians = np.asarray([_surface_jacobian(triangle) for triangle in triangles])
    physical_weights = jacobians[:, None] * weights[None, :]
    return jnp.asarray(mapped), jnp.asarray(physical_weights)


def _prepare_surface_pairs_3d(
    vertices: Array,
    faces: Array,
    /,
    *,
    regular_order: int,
    singular_order: int,
    near_order: int,
    near_ratio: float,
    near_max_depth: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_exception_pairs: int,
    max_preparation_workspace_bytes: int,
    max_resident_bytes: int,
) -> _SurfacePairData3D:
    vertices_host = np.asarray(vertices, dtype=float)
    faces_host = np.asarray(faces, dtype=np.int32)
    triangles = vertices_host[faces_host]
    face_count = faces_host.shape[0]
    bounds_min = np.min(triangles, axis=1)
    bounds_max = np.max(triangles, axis=1)
    diameters = np.asarray([_diameter(triangle) for triangle in triangles])

    records: list[tuple[int, int, int]] = []
    regular_count = 0
    regular_error = 0.0
    regular_evaluations = 0
    high_regular = regular_order + 2
    for target in range(face_count):
        target_vertices = set(map(int, faces_host[target]))
        for source in range(face_count):
            if target == source:
                pair_class = 0
            else:
                shared_count = len(target_vertices & set(map(int, faces_host[source])))
                if shared_count == 2:
                    pair_class = 1
                elif shared_count == 1:
                    pair_class = 2
                else:
                    gap = np.maximum(
                        np.maximum(
                            bounds_min[target] - bounds_max[source],
                            bounds_min[source] - bounds_max[target],
                        ),
                        0.0,
                    )
                    scale = max(diameters[target], diameters[source])
                    pair_class = (
                        3 if float(np.linalg.norm(gap)) / scale < near_ratio else -1
                    )
            if pair_class < 0:
                low = _regular_pair_value(
                    triangles[target], triangles[source], regular_order
                )
                high = _regular_pair_value(
                    triangles[target], triangles[source], high_regular
                )
                error = abs(high - low)
                threshold = absolute_tolerance + relative_tolerance * abs(high)
                if error > threshold:
                    raise ValueError("Regular-pair quadrature exceeds its tolerance.")
                regular_error = max(regular_error, error)
                regular_evaluations += regular_order**4 + high_regular**4
                regular_count += 1
                continue
            if len(records) >= int(max_exception_pairs):
                raise ValueError("Surface pair exceptions exceed max_exception_pairs.")
            records.append((target, source, pair_class))

    estimated_workspace = max(face_count, 1) * 2 * 3 * np.dtype(float).itemsize
    estimated_workspace += max(len(records), 1) * 5 * np.dtype(float).itemsize
    singular_rule_workspace = 6 * (singular_order + 2) ** 4 * 5 * np.dtype(float).itemsize
    near_rule_workspace = (near_order + 2) ** 4 * 4 * np.dtype(float).itemsize
    estimated_workspace += max(singular_rule_workspace, near_rule_workspace)
    if estimated_workspace > int(max_preparation_workspace_bytes):
        raise ValueError("Surface pair preparation exceeds its workspace budget.")

    values = np.empty((len(records),), dtype=float)
    classes = np.empty((len(records),), dtype=np.int32)
    errors = np.zeros((5,), dtype=float)
    evaluations = np.zeros((5,), dtype=np.int64)
    counts = np.zeros((4,), dtype=np.int64)
    high_singular = singular_order + 2
    high_near = near_order + 2
    errors[4] = regular_error
    evaluations[4] = regular_evaluations
    for index, (target, source, pair_class) in enumerate(records):
        if pair_class == 0:
            adjacency = "coincident"
        elif pair_class == 1:
            adjacency = "shared-edge"
        elif pair_class == 2:
            adjacency = "shared-vertex"
        else:
            adjacency = "near"
        if pair_class < 3:
            low = _singular_pair_value(
                triangles, faces_host, target, source, adjacency, singular_order
            )
            high = _singular_pair_value(
                triangles, faces_host, target, source, adjacency, high_singular
            )
            error = abs(high - low)
            count = (singular_order**4 + high_singular**4) * (6, 5, 2)[pair_class]
            threshold = absolute_tolerance + relative_tolerance * abs(high)
            if error > threshold:
                raise ValueError(
                    f"{_PAIR_NAMES[pair_class]} quadrature exceeds its tolerance."
                )
            value = high
        else:
            value, error, count = _near_pair_value(
                triangles[target],
                triangles[source],
                low_order=near_order,
                high_order=high_near,
                max_depth=near_max_depth,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        values[index] = value
        classes[index] = pair_class
        errors[pair_class] = max(errors[pair_class], error)
        evaluations[pair_class] += count
        counts[pair_class] += 1

    if records:
        targets = np.asarray([record[0] for record in records], dtype=np.int32)
        sources = np.asarray([record[1] for record in records], dtype=np.int32)
        keys = targets.astype(np.int64) * face_count + sources.astype(np.int64)
        order = np.argsort(keys, kind="stable")
        keys, targets, sources = keys[order], targets[order], sources[order]
        classes, values = classes[order], values[order]
    else:
        keys = np.zeros((0,), dtype=np.int64)
        targets = np.zeros((0,), dtype=np.int32)
        sources = np.zeros((0,), dtype=np.int32)

    regular_points, regular_weights = _regular_face_quadrature(triangles, regular_order)
    resident_bytes = sum(
        array.nbytes
        for array in (
            keys,
            targets,
            sources,
            classes,
            values,
            np.asarray(regular_points),
            np.asarray(regular_weights),
        )
    )
    if resident_bytes > int(max_resident_bytes):
        raise ValueError("Surface pair state exceeds its resident-byte budget.")
    return _SurfacePairData3D(
        exception_keys=jnp.asarray(keys, dtype=jnp.int64),
        targets=jnp.asarray(targets, dtype=jnp.int32),
        sources=jnp.asarray(sources, dtype=jnp.int32),
        classes=jnp.asarray(classes, dtype=jnp.int32),
        values=jnp.asarray(values),
        regular_points=regular_points,
        regular_weights=regular_weights,
        counts=(
            int(counts[0]),
            int(counts[1]),
            int(counts[2]),
            int(counts[3]),
            int(regular_count),
        ),
        maximum_errors=jnp.asarray(errors),
        evaluations=jnp.asarray(evaluations, dtype=jnp.int64),
        preparation_workspace_bytes=int(estimated_workspace),
        resident_bytes=int(resident_bytes),
    )


__all__: list[str] = []
