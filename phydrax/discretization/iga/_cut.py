#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections import deque
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_INSIDE = 1
_BOUNDARY = 0
_OUTSIDE = -1


def _finite_array(value: ArrayLike, shape_tail: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim < len(shape_tail) or array.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must end in shape {shape_tail}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.longdouble:
    ax, ay = np.longdouble(a[0]), np.longdouble(a[1])
    bx, by = np.longdouble(b[0]), np.longdouble(b[1])
    cx, cy = np.longdouble(c[0]), np.longdouble(c[1])
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _point_on_segment(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, tolerance: float
) -> bool:
    scale = max(1.0, float(np.linalg.norm(b - a)))
    if abs(float(_orientation(a, b, point))) > tolerance * scale:
        return False
    return bool(
        np.all(point >= np.minimum(a, b) - tolerance)
        and np.all(point <= np.maximum(a, b) + tolerance)
    )


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tolerance: float,
) -> bool:
    o1 = float(_orientation(a, b, c))
    o2 = float(_orientation(a, b, d))
    o3 = float(_orientation(c, d, a))
    o4 = float(_orientation(c, d, b))
    if (
        (o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)
    ) and ((o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)):
        return True
    return any(
        (abs(value) <= tolerance and _point_on_segment(point, first, second, tolerance))
        for value, point, first, second in (
            (o1, c, a, b),
            (o2, d, a, b),
            (o3, a, c, d),
            (o4, b, c, d),
        )
    )


def _classify_polygon(
    vertices: np.ndarray, points: np.ndarray, tolerance: float
) -> np.ndarray:
    result = np.full(points.shape[0], _OUTSIDE, dtype=np.int8)
    for point_index, point in enumerate(points):
        winding = 0
        on_boundary = False
        for edge_index, a in enumerate(vertices):
            b = vertices[(edge_index + 1) % vertices.shape[0]]
            if _point_on_segment(point, a, b, tolerance):
                on_boundary = True
                break
            if a[1] <= point[1] < b[1] and _orientation(a, b, point) > 0:
                winding += 1
            elif b[1] <= point[1] < a[1] and _orientation(a, b, point) < 0:
                winding -= 1
        if on_boundary:
            result[point_index] = _BOUNDARY
        elif winding != 0:
            result[point_index] = _INSIDE
    return result


class UVTrimLoop(StrictModule, NonTrainableState):
    """A simple, implicitly closed, piecewise-linear loop in exact UV coordinates."""

    vertices: Array
    signed_area: float = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    loop_id: str = eqx.field(static=True)

    def __init__(self, vertices: ArrayLike, /):
        vertices_ = _finite_array(vertices, (2,), "UV loop vertices")
        if vertices_.ndim != 2 or vertices_.shape[0] < 3:
            raise ValueError("A UV trim loop requires at least three rank-2 vertices.")
        if np.array_equal(vertices_[0], vertices_[-1]):
            vertices_ = vertices_[:-1]
        if vertices_.shape[0] < 3:
            raise ValueError("A UV trim loop requires three distinct vertices.")
        edges = np.roll(vertices_, -1, axis=0) - vertices_
        scale = max(1.0, float(np.max(np.ptp(vertices_, axis=0))))
        tolerance = 64.0 * np.finfo(float).eps * scale
        if np.any(np.linalg.norm(edges, axis=1) <= tolerance):
            raise ValueError("UV trim loops cannot contain zero-length edges.")
        count = vertices_.shape[0]
        for first in range(count):
            for second in range(first + 1, count):
                if (
                    second in (first, (first + 1) % count)
                    or first == (second + 1) % count
                ):
                    continue
                if _segments_intersect(
                    vertices_[first],
                    vertices_[(first + 1) % count],
                    vertices_[second],
                    vertices_[(second + 1) % count],
                    tolerance,
                ):
                    raise ValueError("UV trim loops must be simple.")
        twice_area = float(
            np.sum(
                vertices_[:, 0] * np.roll(vertices_[:, 1], -1)
                - vertices_[:, 1] * np.roll(vertices_[:, 0], -1)
            )
        )
        if abs(twice_area) <= tolerance * scale:
            raise ValueError("UV trim loops must enclose nonzero area.")
        self.vertices = jnp.asarray(vertices_)
        self.signed_area = 0.5 * twice_area
        self.orientation = 1 if twice_area > 0.0 else -1
        self.loop_id = canonical_fingerprint(
            {
                "kind": "iga-uv-trim-loop",
                "vertices": array_tree_fingerprint(vertices_),
            }
        )


class UVTrimCertificate(StrictModule, NonTrainableState):
    """Topology and exact polygonal measure evidence for a trimmed UV domain."""

    parametric_area: float = eqx.field(static=True)
    boundary_length: float = eqx.field(static=True)
    loop_count: int = eqx.field(static=True)
    predicate_tolerance: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        parametric_area: float,
        boundary_length: float,
        loop_ids: tuple[str, ...],
        predicate_tolerance: float,
        /,
    ):
        area = float(parametric_area)
        length = float(boundary_length)
        tolerance = float(predicate_tolerance)
        if not isfinite(area) or area <= 0.0 or not isfinite(length) or length <= 0.0:
            raise ValueError(
                "Trim certificates require positive finite area and boundary length."
            )
        if not loop_ids or not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "Trim certificates require loops and a positive predicate tolerance."
            )
        self.parametric_area = area
        self.boundary_length = length
        self.loop_count = len(loop_ids)
        self.predicate_tolerance = tolerance
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "iga-uv-trim-certificate",
                "area": area,
                "boundary_length": length,
                "loops": list(loop_ids),
                "predicate_tolerance": tolerance,
            }
        )


class UVTrimmedSurface(StrictModule, NonTrainableState):
    """One counter-clockwise outer UV loop minus disjoint clockwise hole loops."""

    outer: UVTrimLoop
    holes: tuple[UVTrimLoop, ...]
    certificate: UVTrimCertificate
    surface_id: str = eqx.field(static=True)

    def __init__(
        self,
        outer: UVTrimLoop,
        holes: tuple[UVTrimLoop, ...] = (),
        /,
        *,
        predicate_tolerance: float | None = None,
    ):
        if not isinstance(outer, UVTrimLoop) or any(
            not isinstance(loop, UVTrimLoop) for loop in holes
        ):
            raise TypeError("Trimmed surfaces require UVTrimLoop objects.")
        if outer.orientation != 1 or any(loop.orientation != -1 for loop in holes):
            raise ValueError(
                "The outer trim loop must be counter-clockwise and holes clockwise."
            )
        all_vertices = np.concatenate(
            [np.asarray(outer.vertices)] + [np.asarray(loop.vertices) for loop in holes],
            axis=0,
        )
        scale = max(1.0, float(np.max(np.ptp(all_vertices, axis=0))))
        tolerance = (
            64.0 * np.finfo(float).eps * scale
            if predicate_tolerance is None
            else float(predicate_tolerance)
        )
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("predicate_tolerance must be positive and finite.")
        outer_vertices = np.asarray(outer.vertices)
        for hole_index, hole in enumerate(holes):
            hole_vertices = np.asarray(hole.vertices)
            if np.any(
                _classify_polygon(outer_vertices, hole_vertices, tolerance) != _INSIDE
            ):
                raise ValueError("Every hole must lie strictly inside the outer loop.")
            for other in holes[:hole_index]:
                other_vertices = np.asarray(other.vertices)
                for edge_index, a in enumerate(hole_vertices):
                    b = hole_vertices[(edge_index + 1) % hole_vertices.shape[0]]
                    for other_index, c in enumerate(other_vertices):
                        d = other_vertices[(other_index + 1) % other_vertices.shape[0]]
                        if _segments_intersect(a, b, c, d, tolerance):
                            raise ValueError("Trim holes must be pairwise disjoint.")
                if (
                    _classify_polygon(other_vertices, hole_vertices[:1], tolerance)[0]
                    != _OUTSIDE
                ):
                    raise ValueError(
                        "Nested trim holes are not part of the declared subset."
                    )
        area = outer.signed_area + sum(loop.signed_area for loop in holes)
        boundary_length = sum(
            float(
                np.sum(
                    np.linalg.norm(
                        np.roll(np.asarray(loop.vertices), -1, axis=0)
                        - np.asarray(loop.vertices),
                        axis=1,
                    )
                )
            )
            for loop in (outer,) + holes
        )
        certificate = UVTrimCertificate(
            area,
            boundary_length,
            tuple(loop.loop_id for loop in (outer,) + holes),
            tolerance,
        )
        self.outer = outer
        self.holes = holes
        self.certificate = certificate
        self.surface_id = canonical_fingerprint(
            {
                "kind": "iga-uv-trimmed-surface",
                "outer": outer.loop_id,
                "holes": [loop.loop_id for loop in holes],
                "certificate": certificate.certificate_id,
            }
        )

    def classify(self, uv_points: ArrayLike, /) -> Array:
        """Return +1 inside, 0 on a loop, and -1 outside without sampling masks."""

        points = _finite_array(uv_points, (2,), "UV classification points")
        original_shape = points.shape[:-1]
        flat = points.reshape((-1, 2))
        result = _classify_polygon(
            np.asarray(self.outer.vertices), flat, self.certificate.predicate_tolerance
        )
        for hole in self.holes:
            hole_state = _classify_polygon(
                np.asarray(hole.vertices), flat, self.certificate.predicate_tolerance
            )
            result = np.where(hole_state == _BOUNDARY, _BOUNDARY, result)
            result = np.where(hole_state == _INSIDE, _OUTSIDE, result)
        return jnp.asarray(result.reshape(original_shape))


class AbstractImmersedBRepClassifier(StrictModule, NonTrainableState):
    """Fail-closed interface for an oriented immersed boundary representation."""

    @abc.abstractmethod
    def classify(self, points: ArrayLike, /) -> Array:
        """Return +1 inside, 0 on the boundary, and -1 outside."""


class ConvexBRepCertificate(StrictModule, NonTrainableState):
    """Closed-manifold, orientation, convexity, and volume evidence."""

    signed_volume: float = eqx.field(static=True)
    minimum_face_area: float = eqx.field(static=True)
    maximum_halfspace_residual: float = eqx.field(static=True)
    predicate_tolerance: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        signed_volume: float,
        minimum_face_area: float,
        maximum_halfspace_residual: float,
        predicate_tolerance: float,
        topology_fingerprint: dict[str, object],
        /,
    ):
        volume = float(signed_volume)
        area = float(minimum_face_area)
        residual = float(maximum_halfspace_residual)
        tolerance = float(predicate_tolerance)
        if volume <= 0.0 or area <= 0.0 or residual > tolerance:
            raise ValueError("The triangular BRep is not outward-oriented and convex.")
        self.signed_volume = volume
        self.minimum_face_area = area
        self.maximum_halfspace_residual = residual
        self.predicate_tolerance = tolerance
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "iga-convex-brep-certificate",
                "volume": volume,
                "minimum_face_area": area,
                "maximum_halfspace_residual": residual,
                "predicate_tolerance": tolerance,
                "topology": topology_fingerprint,
            }
        )


class ConvexTriangleBRepClassifier(AbstractImmersedBRepClassifier):
    """Exact half-space classifier for a closed outward convex triangular BRep."""

    vertices: Array
    triangles: Array
    unit_normals: Array
    plane_offsets: Array
    certificate: ConvexBRepCertificate
    classifier_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        predicate_tolerance: float | None = None,
    ):
        vertices_ = _finite_array(vertices, (3,), "BRep vertices")
        triangles_ = np.asarray(triangles, dtype=np.int64)
        if vertices_.ndim != 2 or vertices_.shape[0] < 4:
            raise ValueError("A convex BRep requires at least four 3D vertices.")
        if triangles_.ndim != 2 or triangles_.shape[1] != 3 or triangles_.shape[0] < 4:
            raise ValueError("BRep triangles must have shape (n, 3).")
        if np.any(triangles_ < 0) or np.any(triangles_ >= vertices_.shape[0]):
            raise ValueError("BRep triangle indices are out of bounds.")
        scale = max(1.0, float(np.max(np.ptp(vertices_, axis=0))))
        tolerance = (
            128.0 * np.finfo(float).eps * scale
            if predicate_tolerance is None
            else float(predicate_tolerance)
        )
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("predicate_tolerance must be positive and finite.")
        directed_edges: dict[tuple[int, int], int] = {}
        undirected_counts: dict[tuple[int, int], int] = {}
        for triangle in triangles_:
            if len(set(int(value) for value in triangle)) != 3:
                raise ValueError("BRep triangles cannot repeat vertices.")
            for start, end in zip(triangle, np.roll(triangle, -1), strict=True):
                edge = (min(int(start), int(end)), max(int(start), int(end)))
                undirected_counts[edge] = undirected_counts.get(edge, 0) + 1
                direction = 1 if int(start) < int(end) else -1
                directed_edges[edge] = directed_edges.get(edge, 0) + direction
        if any(count != 2 for count in undirected_counts.values()) or any(
            direction != 0 for direction in directed_edges.values()
        ):
            raise ValueError(
                "The triangular BRep must be closed and consistently oriented."
            )
        a = vertices_[triangles_[:, 0]]
        b = vertices_[triangles_[:, 1]]
        c = vertices_[triangles_[:, 2]]
        normals = np.cross(b - a, c - a)
        twice_areas = np.linalg.norm(normals, axis=1)
        if np.any(twice_areas <= tolerance * scale):
            raise ValueError("BRep triangles must be nondegenerate.")
        unit_normals = normals / twice_areas[:, None]
        offsets = np.sum(unit_normals * a, axis=1)
        halfspace = vertices_ @ unit_normals.T - offsets[None, :]
        maximum_residual = float(np.max(halfspace))
        signed_volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
        topology = array_tree_fingerprint((vertices_, triangles_))
        certificate = ConvexBRepCertificate(
            signed_volume,
            0.5 * float(np.min(twice_areas)),
            maximum_residual,
            tolerance,
            topology,
        )
        self.vertices = jnp.asarray(vertices_)
        self.triangles = jnp.asarray(triangles_, dtype=jnp.int32)
        self.unit_normals = jnp.asarray(unit_normals)
        self.plane_offsets = jnp.asarray(offsets)
        self.certificate = certificate
        self.classifier_id = canonical_fingerprint(
            {
                "kind": "iga-convex-triangle-brep-classifier",
                "certificate": certificate.certificate_id,
            }
        )

    def classify(self, points: ArrayLike, /) -> Array:
        points_ = _finite_array(points, (3,), "BRep classification points")
        original_shape = points_.shape[:-1]
        distances = (
            points_.reshape((-1, 3)) @ np.asarray(self.unit_normals).T
            - np.asarray(self.plane_offsets)[None, :]
        )
        maximum = np.max(distances, axis=1)
        tolerance = self.certificate.predicate_tolerance
        labels = np.where(
            maximum > tolerance,
            _OUTSIDE,
            np.where(maximum >= -tolerance, _BOUNDARY, _INSIDE),
        )
        return jnp.asarray(labels.reshape(original_shape))


def _clip_polygon_axis(
    polygon: np.ndarray, axis: int, bound: float, keep_greater: bool
) -> np.ndarray:
    if polygon.shape[0] == 0:
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
    for current in polygon:
        current_inside = (
            current[axis] >= bound if keep_greater else current[axis] <= bound
        )
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if delta == 0.0:
                raise ValueError("Degenerate polygon clipping transition.")
            ratio = (bound - previous[axis]) / delta
            output.append(previous + ratio * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=float).reshape((-1, 2))


def _clip_polygon_box(polygon: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    result = polygon
    result = _clip_polygon_axis(result, 0, bounds[0, 0], True)
    result = _clip_polygon_axis(result, 0, bounds[0, 1], False)
    result = _clip_polygon_axis(result, 1, bounds[1, 0], True)
    result = _clip_polygon_axis(result, 1, bounds[1, 1], False)
    return result


def _clip_segment_box(
    start: np.ndarray, end: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    direction = end - start
    lower = 0.0
    upper = 1.0
    for axis in range(2):
        for numerator, denominator in (
            (start[axis] - bounds[axis, 0], -direction[axis]),
            (bounds[axis, 1] - start[axis], direction[axis]),
        ):
            if denominator == 0.0:
                if numerator < 0.0:
                    return None
                continue
            ratio = numerator / denominator
            if denominator > 0.0:
                upper = min(upper, ratio)
            else:
                lower = max(lower, ratio)
            if lower >= upper:
                return None
    return start + lower * direction, start + upper * direction


def _legendre_unit(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (points + 1.0), 0.5 * weights


def _surface_measure(jacobian: np.ndarray) -> np.ndarray:
    gram = np.einsum("nai,naj->nij", jacobian, jacobian)
    determinant = np.linalg.det(gram)
    if np.any(determinant <= 0.0) or np.any(~np.isfinite(determinant)):
        raise ValueError("The geometry Jacobian must have full surface rank.")
    return np.sqrt(determinant)


def _triangle_quadrature(
    triangle: np.ndarray,
    order: int,
    geometry_jacobian: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    points_1d, weights_1d = _legendre_unit(order)
    rr, ss = np.meshgrid(points_1d, points_1d, indexing="ij")
    wr, ws = np.meshgrid(weights_1d, weights_1d, indexing="ij")
    r = rr.reshape((-1,))
    s = ss.reshape((-1,))
    a, b, c = triangle
    uv = a + r[:, None] * (b - a) + ((1.0 - r) * s)[:, None] * (c - a)
    determinant = float(np.linalg.det(np.stack((b - a, c - a), axis=1)))
    jacobian = np.asarray(geometry_jacobian(uv), dtype=float)
    if jacobian.ndim != 3 or jacobian.shape[0] != uv.shape[0] or jacobian.shape[2] != 2:
        raise ValueError("geometry_jacobian must return shape (points, ambient, 2).")
    weights = (
        (wr * ws).reshape((-1,)) * (1.0 - r) * determinant * _surface_measure(jacobian)
    )
    return uv, weights


def _split_triangle(triangle: np.ndarray) -> tuple[np.ndarray, ...]:
    a, b, c = triangle
    ab = 0.5 * (a + b)
    bc = 0.5 * (b + c)
    ca = 0.5 * (c + a)
    return (
        np.stack((a, ab, ca)),
        np.stack((ab, b, bc)),
        np.stack((ca, bc, c)),
        np.stack((ab, bc, ca)),
    )


def _adaptive_triangle(
    triangle: np.ndarray,
    order: int,
    geometry_jacobian: Callable[[np.ndarray], np.ndarray],
    absolute_tolerance: float,
    relative_tolerance: float,
    depth: int,
    maximum_depth: int,
) -> tuple[list[np.ndarray], list[np.ndarray], float, bool, int]:
    _, coarse_weights = _triangle_quadrature(triangle, order, geometry_jacobian)
    fine_points, fine_weights = _triangle_quadrature(
        triangle, order + 2, geometry_jacobian
    )
    error = abs(float(np.sum(fine_weights) - np.sum(coarse_weights)))
    threshold = absolute_tolerance + relative_tolerance * abs(float(np.sum(fine_weights)))
    if error <= threshold or depth == maximum_depth:
        return [fine_points], [fine_weights], error, error <= threshold, 1
    point_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    total_error = 0.0
    satisfied = True
    leaves = 0
    for child in _split_triangle(triangle):
        points, weights, child_error, child_satisfied, child_leaves = _adaptive_triangle(
            child,
            order,
            geometry_jacobian,
            0.25 * absolute_tolerance,
            relative_tolerance,
            depth + 1,
            maximum_depth,
        )
        point_blocks.extend(points)
        weight_blocks.extend(weights)
        total_error += child_error
        satisfied = satisfied and child_satisfied
        leaves += child_leaves
    return point_blocks, weight_blocks, total_error, satisfied, leaves


def _segment_quadrature(
    segment: np.ndarray,
    order: int,
    geometry_jacobian: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    points_1d, weights_1d = _legendre_unit(order)
    direction = segment[1] - segment[0]
    uv = segment[0] + points_1d[:, None] * direction
    jacobian = np.asarray(geometry_jacobian(uv), dtype=float)
    if jacobian.ndim != 3 or jacobian.shape[0] != uv.shape[0] or jacobian.shape[2] != 2:
        raise ValueError("geometry_jacobian must return shape (points, ambient, 2).")
    tangents = np.einsum("nai,i->na", jacobian, direction)
    physical_speed = np.linalg.norm(tangents, axis=1)
    if np.any(physical_speed <= 0.0) or np.any(~np.isfinite(physical_speed)):
        raise ValueError("Trim boundary images must have nonzero finite tangents.")
    return uv, weights_1d * physical_speed


def _adaptive_segment(
    segment: np.ndarray,
    order: int,
    geometry_jacobian: Callable[[np.ndarray], np.ndarray],
    absolute_tolerance: float,
    relative_tolerance: float,
    depth: int,
    maximum_depth: int,
) -> tuple[list[np.ndarray], list[np.ndarray], float, bool, int]:
    _, coarse_weights = _segment_quadrature(segment, order, geometry_jacobian)
    fine_points, fine_weights = _segment_quadrature(segment, order + 2, geometry_jacobian)
    error = abs(float(np.sum(fine_weights) - np.sum(coarse_weights)))
    threshold = absolute_tolerance + relative_tolerance * float(np.sum(fine_weights))
    if error <= threshold or depth == maximum_depth:
        return [fine_points], [fine_weights], error, error <= threshold, 1
    midpoint = 0.5 * (segment[0] + segment[1])
    point_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    total_error = 0.0
    satisfied = True
    leaves = 0
    for child in (np.stack((segment[0], midpoint)), np.stack((midpoint, segment[1]))):
        points, weights, child_error, child_satisfied, child_leaves = _adaptive_segment(
            child,
            order,
            geometry_jacobian,
            0.5 * absolute_tolerance,
            relative_tolerance,
            depth + 1,
            maximum_depth,
        )
        point_blocks.extend(points)
        weight_blocks.extend(weights)
        total_error += child_error
        satisfied = satisfied and child_satisfied
        leaves += child_leaves
    return point_blocks, weight_blocks, total_error, satisfied, leaves


class CutQuadratureCertificate(StrictModule, NonTrainableState):
    """Error and signed-chain evidence for adaptive physical cut quadrature."""

    parametric_measure: float = eqx.field(static=True)
    physical_measure: float = eqx.field(static=True)
    physical_boundary_measure: float = eqx.field(static=True)
    volume_error_bound: float = eqx.field(static=True)
    boundary_error_bound: float = eqx.field(static=True)
    triangle_leaf_count: int = eqx.field(static=True)
    segment_leaf_count: int = eqx.field(static=True)
    tolerance_satisfied: bool = eqx.field(static=True)
    exact_polygon_clipping: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        parametric_measure: float,
        physical_measure: float,
        physical_boundary_measure: float,
        volume_error_bound: float,
        boundary_error_bound: float,
        triangle_leaf_count: int,
        segment_leaf_count: int,
        tolerance_satisfied: bool,
        source_id: str,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                parametric_measure,
                physical_measure,
                physical_boundary_measure,
                volume_error_bound,
                boundary_error_bound,
            )
        )
        if (
            any(not isfinite(value) for value in values)
            or values[0] <= 0.0
            or values[1] <= 0.0
        ):
            raise ValueError("Cut quadrature measures and error bounds must be finite.")
        if values[2] <= 0.0 or values[3] < 0.0 or values[4] < 0.0:
            raise ValueError("Cut boundary measure and error bounds are invalid.")
        if triangle_leaf_count <= 0 or segment_leaf_count <= 0:
            raise ValueError("Cut quadrature must contain physical and boundary leaves.")
        self.parametric_measure = values[0]
        self.physical_measure = values[1]
        self.physical_boundary_measure = values[2]
        self.volume_error_bound = values[3]
        self.boundary_error_bound = values[4]
        self.triangle_leaf_count = int(triangle_leaf_count)
        self.segment_leaf_count = int(segment_leaf_count)
        self.tolerance_satisfied = bool(tolerance_satisfied)
        self.exact_polygon_clipping = True
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "iga-cut-quadrature-certificate",
                "source": source_id,
                "measures": list(values[:3]),
                "errors": list(values[3:]),
                "leaves": [int(triangle_leaf_count), int(segment_leaf_count)],
                "tolerance_satisfied": bool(tolerance_satisfied),
                "exact_polygon_clipping": True,
            }
        )


class CutQuadratureRule(StrictModule, NonTrainableState):
    """Parametric and physical points with signed physical cell weights."""

    volume_uv_points: Array
    volume_points: Array
    volume_weights: Array
    volume_cell_ids: Array
    boundary_uv_points: Array
    boundary_points: Array
    boundary_weights: Array
    boundary_cell_ids: Array
    boundary_normals_uv: Array
    boundary_conormals: Array
    boundary_loop_ids: Array
    certificate: CutQuadratureCertificate
    quadrature_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_uv_points: np.ndarray,
        volume_points: np.ndarray,
        volume_weights: np.ndarray,
        volume_cell_ids: np.ndarray,
        boundary_uv_points: np.ndarray,
        boundary_points: np.ndarray,
        boundary_weights: np.ndarray,
        boundary_cell_ids: np.ndarray,
        boundary_normals_uv: np.ndarray,
        boundary_conormals: np.ndarray,
        boundary_loop_ids: np.ndarray,
        certificate: CutQuadratureCertificate,
        /,
    ):
        if volume_uv_points.shape != (volume_weights.size, 2):
            raise ValueError("Volume UV points and weights are incompatible.")
        if (
            volume_points.shape[0] != volume_weights.size
            or volume_cell_ids.shape != volume_weights.shape
        ):
            raise ValueError("Physical volume quadrature arrays are incompatible.")
        if boundary_uv_points.shape != (boundary_weights.size, 2):
            raise ValueError("Boundary UV points and weights are incompatible.")
        if (
            boundary_points.shape[0] != boundary_weights.size
            or boundary_cell_ids.shape != boundary_weights.shape
            or boundary_normals_uv.shape != boundary_uv_points.shape
            or boundary_conormals.shape != boundary_points.shape
            or boundary_loop_ids.shape != boundary_weights.shape
        ):
            raise ValueError("Physical boundary quadrature arrays are incompatible.")
        if np.any(~np.isfinite(volume_weights)) or np.any(boundary_weights <= 0.0):
            raise ValueError("Cut weights must be finite and boundary weights positive.")
        self.volume_uv_points = jnp.asarray(volume_uv_points)
        self.volume_points = jnp.asarray(volume_points)
        self.volume_weights = jnp.asarray(volume_weights)
        self.volume_cell_ids = jnp.asarray(volume_cell_ids, dtype=jnp.int32)
        self.boundary_uv_points = jnp.asarray(boundary_uv_points)
        self.boundary_points = jnp.asarray(boundary_points)
        self.boundary_weights = jnp.asarray(boundary_weights)
        self.boundary_cell_ids = jnp.asarray(boundary_cell_ids, dtype=jnp.int32)
        self.boundary_normals_uv = jnp.asarray(boundary_normals_uv)
        self.boundary_conormals = jnp.asarray(boundary_conormals)
        self.boundary_loop_ids = jnp.asarray(boundary_loop_ids, dtype=jnp.int32)
        self.certificate = certificate
        self.quadrature_id = canonical_fingerprint(
            {
                "kind": "iga-cut-quadrature-rule",
                "certificate": certificate.certificate_id,
                "arrays": array_tree_fingerprint(
                    (
                        volume_uv_points,
                        volume_weights,
                        volume_cell_ids,
                        boundary_uv_points,
                        boundary_weights,
                        boundary_cell_ids,
                    )
                ),
            }
        )


def build_trimmed_surface_quadrature(
    surface: UVTrimmedSurface,
    cell_bounds: ArrayLike,
    /,
    *,
    geometry_map: Callable[[np.ndarray], np.ndarray] | None = None,
    geometry_jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
    order: int = 3,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-8,
    maximum_depth: int = 8,
    require_tolerance: bool = True,
) -> CutQuadratureRule:
    """Clip loops against cells and adapt quadrature to the physical geometry map."""

    if not isinstance(surface, UVTrimmedSurface):
        raise TypeError("surface must be a UVTrimmedSurface.")
    bounds = _finite_array(cell_bounds, (2, 2), "cut cell bounds")
    if bounds.ndim != 3 or np.any(bounds[:, :, 1] <= bounds[:, :, 0]):
        raise ValueError("cell_bounds must have shape (cells, 2, 2) with positive spans.")
    for first in range(bounds.shape[0]):
        for second in range(first + 1, bounds.shape[0]):
            overlap = np.minimum(bounds[first, :, 1], bounds[second, :, 1]) - np.maximum(
                bounds[first, :, 0], bounds[second, :, 0]
            )
            if np.all(overlap > 0.0):
                raise ValueError("Cut quadrature cells must have disjoint interiors.")
    if order < 1 or maximum_depth < 0:
        raise ValueError("Quadrature order and maximum depth are invalid.")
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    if (
        not isfinite(absolute)
        or not isfinite(relative)
        or absolute <= 0.0
        or relative < 0.0
    ):
        raise ValueError("Quadrature tolerances are invalid.")

    def identity_map(points: np.ndarray) -> np.ndarray:
        return points

    def identity_jacobian(points: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.eye(2), (points.shape[0], 2, 2))

    map_ = identity_map if geometry_map is None else geometry_map
    jacobian_ = identity_jacobian if geometry_jacobian is None else geometry_jacobian
    volume_points_blocks: list[np.ndarray] = []
    volume_weight_blocks: list[np.ndarray] = []
    volume_cell_blocks: list[np.ndarray] = []
    volume_error = 0.0
    volume_satisfied = True
    triangle_leaves = 0
    parametric_measure = 0.0
    loops = (surface.outer,) + surface.holes
    for cell_id, cell in enumerate(bounds):
        for loop in loops:
            polygon = _clip_polygon_box(np.asarray(loop.vertices), cell)
            if polygon.shape[0] < 3:
                continue
            polygon_area = 0.5 * float(
                np.sum(
                    polygon[:, 0] * np.roll(polygon[:, 1], -1)
                    - polygon[:, 1] * np.roll(polygon[:, 0], -1)
                )
            )
            if abs(polygon_area) <= surface.certificate.predicate_tolerance:
                continue
            parametric_measure += polygon_area
            anchor = polygon[0]
            for index in range(1, polygon.shape[0] - 1):
                triangle = np.stack((anchor, polygon[index], polygon[index + 1]))
                if (
                    abs(
                        float(
                            np.linalg.det(
                                np.stack(
                                    (triangle[1] - anchor, triangle[2] - anchor), axis=1
                                )
                            )
                        )
                    )
                    <= surface.certificate.predicate_tolerance
                ):
                    continue
                points, weights, error, satisfied, leaves = _adaptive_triangle(
                    triangle,
                    int(order),
                    jacobian_,
                    absolute / max(1, bounds.shape[0] * len(loops)),
                    relative,
                    0,
                    int(maximum_depth),
                )
                for point_block, weight_block in zip(points, weights, strict=True):
                    volume_points_blocks.append(point_block)
                    volume_weight_blocks.append(weight_block)
                    volume_cell_blocks.append(
                        np.full(weight_block.shape, cell_id, dtype=np.int64)
                    )
                volume_error += error
                volume_satisfied = volume_satisfied and satisfied
                triangle_leaves += leaves
    if not volume_weight_blocks:
        raise ValueError("The supplied cells do not intersect the trimmed domain.")

    clipped_segments: dict[
        tuple[float, ...], tuple[int, int, np.ndarray, np.ndarray]
    ] = {}
    for loop_id, loop in enumerate(loops):
        vertices = np.asarray(loop.vertices)
        for edge_id, start in enumerate(vertices):
            end = vertices[(edge_id + 1) % vertices.shape[0]]
            direction = end - start
            normal = np.asarray((direction[1], -direction[0]), dtype=float)
            normal /= np.linalg.norm(normal)
            for cell_id, cell in enumerate(bounds):
                clipped = _clip_segment_box(start, end, cell)
                if clipped is None:
                    continue
                segment = np.stack(clipped)
                key = tuple(np.round(segment.reshape((-1,)), 14))
                previous = clipped_segments.get(key)
                candidate = (cell_id, loop_id, segment, normal)
                if previous is None or cell_id < previous[0]:
                    clipped_segments[key] = candidate
    boundary_point_blocks: list[np.ndarray] = []
    boundary_weight_blocks: list[np.ndarray] = []
    boundary_cell_blocks: list[np.ndarray] = []
    boundary_normal_blocks: list[np.ndarray] = []
    boundary_loop_blocks: list[np.ndarray] = []
    boundary_error = 0.0
    boundary_satisfied = True
    segment_leaves = 0
    for cell_id, loop_id, segment, normal in clipped_segments.values():
        points, weights, error, satisfied, leaves = _adaptive_segment(
            segment,
            int(order),
            jacobian_,
            absolute / max(1, len(clipped_segments)),
            relative,
            0,
            int(maximum_depth),
        )
        for point_block, weight_block in zip(points, weights, strict=True):
            boundary_point_blocks.append(point_block)
            boundary_weight_blocks.append(weight_block)
            boundary_cell_blocks.append(
                np.full(weight_block.shape, cell_id, dtype=np.int64)
            )
            boundary_normal_blocks.append(np.broadcast_to(normal, point_block.shape))
            boundary_loop_blocks.append(
                np.full(weight_block.shape, loop_id, dtype=np.int64)
            )
        boundary_error += error
        boundary_satisfied = boundary_satisfied and satisfied
        segment_leaves += leaves
    if not boundary_weight_blocks:
        raise ValueError("The supplied cells do not intersect the trim boundary.")

    volume_uv = np.concatenate(volume_points_blocks, axis=0)
    volume_weights = np.concatenate(volume_weight_blocks)
    volume_cells = np.concatenate(volume_cell_blocks)
    boundary_uv = np.concatenate(boundary_point_blocks, axis=0)
    boundary_weights = np.concatenate(boundary_weight_blocks)
    boundary_cells = np.concatenate(boundary_cell_blocks)
    boundary_normals = np.concatenate(boundary_normal_blocks, axis=0)
    boundary_loop_ids = np.concatenate(boundary_loop_blocks)
    physical_volume = np.asarray(map_(volume_uv), dtype=float)
    physical_boundary = np.asarray(map_(boundary_uv), dtype=float)
    if (
        physical_volume.ndim != 2
        or physical_volume.shape[0] != volume_uv.shape[0]
        or physical_boundary.shape != (boundary_uv.shape[0], physical_volume.shape[1])
        or np.any(~np.isfinite(physical_volume))
        or np.any(~np.isfinite(physical_boundary))
    ):
        raise ValueError(
            "geometry_map must return finite shape (points, ambient) arrays."
        )
    boundary_jacobian = np.asarray(jacobian_(boundary_uv), dtype=float)
    gram = np.einsum("nai,naj->nij", boundary_jacobian, boundary_jacobian)
    metric_normal = np.linalg.solve(gram, boundary_normals[..., None])[..., 0]
    conormals = np.einsum("nai,ni->na", boundary_jacobian, metric_normal)
    conormal_norm = np.linalg.norm(conormals, axis=1)
    if np.any(conormal_norm <= 0.0):
        raise ValueError("Physical trim conormals are degenerate.")
    conormals /= conormal_norm[:, None]
    tolerance_satisfied = volume_satisfied and boundary_satisfied
    if require_tolerance and not tolerance_satisfied:
        raise ValueError(
            "Adaptive cut quadrature did not meet tolerance before maximum_depth."
        )
    certificate = CutQuadratureCertificate(
        parametric_measure,
        float(np.sum(volume_weights)),
        float(np.sum(boundary_weights)),
        volume_error,
        boundary_error,
        triangle_leaves,
        segment_leaves,
        tolerance_satisfied,
        surface.surface_id,
    )
    return CutQuadratureRule(
        volume_uv,
        physical_volume,
        volume_weights,
        volume_cells,
        boundary_uv,
        physical_boundary,
        boundary_weights,
        boundary_cells,
        boundary_normals,
        conormals,
        boundary_loop_ids,
        certificate,
    )


class CutStabilizationPlan(StrictModule, NonTrainableState):
    """Deterministic cell aggregation and cut-neighbour ghost-penalty routes."""

    volume_fractions: Array
    aggregate_root_cells: Array
    support_root_cells: Array
    ghost_owner_cells: Array
    ghost_neighbour_cells: Array
    ghost_penalty_weights: Array
    support_threshold: float = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume_fractions: np.ndarray,
        aggregate_root_cells: np.ndarray,
        support_root_cells: np.ndarray,
        ghost_owner_cells: np.ndarray,
        ghost_neighbour_cells: np.ndarray,
        ghost_penalty_weights: np.ndarray,
        support_threshold: float,
        polynomial_degree: int,
        /,
    ):
        if (
            volume_fractions.ndim != 1
            or aggregate_root_cells.shape != volume_fractions.shape
        ):
            raise ValueError("Cell fractions and aggregation roots are incompatible.")
        if support_root_cells.ndim != 1:
            raise ValueError("support_root_cells must be rank-1.")
        if not (
            ghost_owner_cells.shape
            == ghost_neighbour_cells.shape
            == ghost_penalty_weights.shape
        ):
            raise ValueError("Ghost-penalty facet routes are incompatible.")
        self.volume_fractions = jnp.asarray(volume_fractions)
        self.aggregate_root_cells = jnp.asarray(aggregate_root_cells, dtype=jnp.int32)
        self.support_root_cells = jnp.asarray(support_root_cells, dtype=jnp.int32)
        self.ghost_owner_cells = jnp.asarray(ghost_owner_cells, dtype=jnp.int32)
        self.ghost_neighbour_cells = jnp.asarray(ghost_neighbour_cells, dtype=jnp.int32)
        self.ghost_penalty_weights = jnp.asarray(ghost_penalty_weights)
        self.support_threshold = float(support_threshold)
        self.polynomial_degree = int(polynomial_degree)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-cut-stabilization-plan",
                "threshold": float(support_threshold),
                "degree": int(polynomial_degree),
                "routes": array_tree_fingerprint(
                    (
                        volume_fractions,
                        aggregate_root_cells,
                        support_root_cells,
                        ghost_owner_cells,
                        ghost_neighbour_cells,
                        ghost_penalty_weights,
                    )
                ),
            }
        )


def plan_cut_stabilization(
    volume_fractions: ArrayLike,
    cell_adjacency: ArrayLike,
    basis_cell_support: ArrayLike,
    cell_diameters: ArrayLike,
    /,
    *,
    support_threshold: float = 0.1,
    polynomial_degree: int = 2,
    ghost_penalty: float = 1.0,
) -> CutStabilizationPlan:
    """Aggregate every small active cell into its nearest well-supported component cell."""

    fractions = np.asarray(volume_fractions, dtype=float)
    adjacency = np.asarray(cell_adjacency, dtype=np.int64)
    support = np.asarray(basis_cell_support, dtype=bool)
    diameters = np.asarray(cell_diameters, dtype=float)
    threshold = float(support_threshold)
    penalty = float(ghost_penalty)
    if (
        fractions.ndim != 1
        or np.any(~np.isfinite(fractions))
        or np.any((fractions < 0.0) | (fractions > 1.0))
        or adjacency.ndim != 2
        or adjacency.shape[1] != 2
        or np.any(adjacency < 0)
        or np.any(adjacency >= fractions.size)
        or support.ndim != 2
        or support.shape[1] != fractions.size
        or diameters.shape != fractions.shape
        or np.any(~np.isfinite(diameters))
        or np.any(diameters <= 0.0)
    ):
        raise ValueError("Cut support, adjacency, or diameter data are invalid.")
    if (
        not 0.0 < threshold <= 1.0
        or polynomial_degree < 1
        or not isfinite(penalty)
        or penalty <= 0.0
    ):
        raise ValueError("Cut stabilization parameters are invalid.")
    active = fractions > 0.0
    good = fractions >= threshold
    neighbours: list[list[int]] = [[] for _ in range(fractions.size)]
    for owner, neighbour in adjacency:
        if owner == neighbour:
            raise ValueError("Cell adjacency cannot contain self-facets.")
        neighbours[int(owner)].append(int(neighbour))
        neighbours[int(neighbour)].append(int(owner))
    roots = np.full(fractions.shape, -1, dtype=np.int64)
    roots[good] = np.flatnonzero(good)
    queue: deque[int] = deque(int(value) for value in np.flatnonzero(good))
    distance = np.full(fractions.shape, np.iinfo(np.int64).max, dtype=np.int64)
    distance[good] = 0
    while queue:
        cell = queue.popleft()
        for neighbour in sorted(neighbours[cell]):
            if not active[neighbour]:
                continue
            candidate_distance = distance[cell] + 1
            if candidate_distance < distance[neighbour] or (
                candidate_distance == distance[neighbour]
                and roots[cell] < roots[neighbour]
            ):
                distance[neighbour] = candidate_distance
                roots[neighbour] = roots[cell]
                queue.append(neighbour)
    if np.any(active & (roots < 0)):
        raise ValueError(
            "Every active cut-cell component must contain a cell above support_threshold."
        )
    support_roots = np.full((support.shape[0],), -1, dtype=np.int64)
    for basis_id, cells in enumerate(support):
        active_cells = np.flatnonzero(cells & active)
        if active_cells.size:
            candidate_roots, counts = np.unique(roots[active_cells], return_counts=True)
            maximum = np.max(counts)
            support_roots[basis_id] = int(np.min(candidate_roots[counts == maximum]))
    ghost_mask = (
        active[adjacency[:, 0]]
        & active[adjacency[:, 1]]
        & (~good[adjacency[:, 0]] | ~good[adjacency[:, 1]])
    )
    ghost = adjacency[ghost_mask]
    weights = (
        penalty
        * float((int(polynomial_degree) + 1) ** 2)
        / np.minimum(diameters[ghost[:, 0]], diameters[ghost[:, 1]])
        if ghost.size
        else np.empty((0,), dtype=float)
    )
    return CutStabilizationPlan(
        fractions,
        roots,
        support_roots,
        ghost[:, 0] if ghost.size else np.empty((0,), dtype=np.int64),
        ghost[:, 1] if ghost.size else np.empty((0,), dtype=np.int64),
        weights,
        threshold,
        int(polynomial_degree),
    )


class CutConditionEvidence(StrictModule, NonTrainableState):
    """Spectral evidence for a concrete support stabilization matrix."""

    unstabilized_minimum_eigenvalue: float = eqx.field(static=True)
    stabilized_minimum_eigenvalue: float = eqx.field(static=True)
    unstabilized_condition: float = eqx.field(static=True)
    stabilized_condition: float = eqx.field(static=True)
    improvement_factor: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        unstabilized_minimum_eigenvalue: float,
        stabilized_minimum_eigenvalue: float,
        unstabilized_condition: float,
        stabilized_condition: float,
        improvement_factor: float,
        passed: bool,
        plan_id: str,
        matrix_fingerprint: dict[str, object],
        /,
    ):
        self.unstabilized_minimum_eigenvalue = float(unstabilized_minimum_eigenvalue)
        self.stabilized_minimum_eigenvalue = float(stabilized_minimum_eigenvalue)
        self.unstabilized_condition = float(unstabilized_condition)
        self.stabilized_condition = float(stabilized_condition)
        self.improvement_factor = float(improvement_factor)
        self.passed = bool(passed)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "iga-cut-condition-evidence",
                "plan": plan_id,
                "matrix": matrix_fingerprint,
                "minimum_eigenvalues": [
                    float(unstabilized_minimum_eigenvalue),
                    float(stabilized_minimum_eigenvalue),
                ],
                "conditions": [
                    float(unstabilized_condition),
                    float(stabilized_condition),
                ],
                "improvement_factor": float(improvement_factor),
                "passed": bool(passed),
            }
        )


def certify_cut_condition(
    unstabilized_matrix: ArrayLike,
    stabilized_matrix: ArrayLike,
    plan: CutStabilizationPlan,
    /,
    *,
    eigenvalue_tolerance: float = 1.0e-12,
) -> CutConditionEvidence:
    """Compare symmetric spectra and retain failed evidence instead of claiming success."""

    if not isinstance(plan, CutStabilizationPlan):
        raise TypeError("plan must be a CutStabilizationPlan.")
    unstabilized = np.asarray(unstabilized_matrix, dtype=float)
    stabilized = np.asarray(stabilized_matrix, dtype=float)
    tolerance = float(eigenvalue_tolerance)
    if (
        unstabilized.ndim != 2
        or unstabilized.shape[0] != unstabilized.shape[1]
        or stabilized.shape != unstabilized.shape
        or np.any(~np.isfinite(unstabilized))
        or np.any(~np.isfinite(stabilized))
        or not isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError(
            "Condition evidence requires finite square matrices and tolerance."
        )
    if not np.allclose(
        unstabilized, unstabilized.T, rtol=0.0, atol=tolerance
    ) or not np.allclose(stabilized, stabilized.T, rtol=0.0, atol=tolerance):
        raise ValueError("Condition evidence is restricted to symmetric matrices.")
    unstable_values = np.linalg.eigvalsh(0.5 * (unstabilized + unstabilized.T))
    stable_values = np.linalg.eigvalsh(0.5 * (stabilized + stabilized.T))
    unstable_min = float(unstable_values[0])
    stable_min = float(stable_values[0])
    unstable_condition = (
        float(unstable_values[-1] / unstable_min)
        if unstable_min > tolerance
        else float("inf")
    )
    stable_condition = (
        float(stable_values[-1] / stable_min) if stable_min > tolerance else float("inf")
    )
    improvement = (
        float("inf")
        if np.isinf(unstable_condition) and np.isfinite(stable_condition)
        else unstable_condition / stable_condition
    )
    passed = stable_min > tolerance and stable_condition < unstable_condition
    return CutConditionEvidence(
        unstable_min,
        stable_min,
        unstable_condition,
        stable_condition,
        improvement,
        passed,
        plan.plan_id,
        array_tree_fingerprint((unstabilized, stabilized)),
    )


__all__ = [
    "AbstractImmersedBRepClassifier",
    "ConvexBRepCertificate",
    "ConvexTriangleBRepClassifier",
    "CutConditionEvidence",
    "CutQuadratureCertificate",
    "CutQuadratureRule",
    "CutStabilizationPlan",
    "UVTrimCertificate",
    "UVTrimLoop",
    "UVTrimmedSurface",
    "build_trimmed_surface_quadrature",
    "certify_cut_condition",
    "plan_cut_stabilization",
]
