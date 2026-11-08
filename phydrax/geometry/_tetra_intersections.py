#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative host-side intersections of affine tetrahedra.

The implementation deliberately does not contain JAX primitives.  A tetrahedron
intersection is a small, topology-changing geometry operation and is therefore
prepared on the host before a remap is committed.  The status and volume are
decided exactly: the source is clipped by the closed target half-spaces in
rational arithmetic on the binary64 inputs.  The reported intersection polytope
is formed by enumerating vertices of the eight half-spaces (four from each input
cell).  Every numerical decision is conservative: malformed cells, uncertain
predicates, and unresolved constructions return a typed non-success result,
never a plausible but unverifiable volume. Orientation and plane-triple singularity
decisions are certified by the geometric predicates of the precision policy
(exact with meshcore; otherwise filtered, with unresolved signs reported as
``UNCERTAIN_PREDICATE``).  Half-space admission and face membership are exact
residual decisions (a fixed-order filter with a derived rounding bound, then
rational arithmetic), and plane-triple vertices are exact rational points
reported correctly rounded, so no decision depends on the BLAS accumulation
order of a dot product.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import combinations
from typing import Any, cast

import numpy as np

from .._geometry_precision import GeometryPrecisionPolicy
from .._geometry_predicates import orient3d, PredicateMode, resolve_host_predicate_mode
from ._rational import (
    binary64_norm_lower,
    binary64_norm_upper,
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)


class TetraIntersectionStatus(str, Enum):
    """Terminal status of a host-side tetrahedron intersection."""

    SUCCESS = "success"
    DISJOINT = "disjoint"
    ZERO_MEASURE_CONTACT = "zero_measure_contact"
    NONFINITE_INPUT = "nonfinite_input"
    INVALID_TETRAHEDRON = "invalid_tetrahedron"
    INVERTED_TETRAHEDRON = "inverted_tetrahedron"
    DEGENERATE_TETRAHEDRON = "degenerate_tetrahedron"
    UNCERTAIN_PREDICATE = "uncertain_predicate"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class TetraIntersectionTolerance:
    """Scale-aware host predicate tolerances.

    ``absolute`` is an absolute determinant/coordinate tolerance and
    ``relative`` scales coordinate and volume decisions by the largest input
    edge.  The defaults are intentionally conservative; they are not used to
    enlarge a remap row or to renormalize any resulting weights.
    """

    absolute: float = 1.0e-13
    relative: float = 1.0e-11

    def __post_init__(self) -> None:
        absolute = float(self.absolute)
        relative = float(self.relative)
        if not (math.isfinite(absolute) and math.isfinite(relative)):
            raise ValueError("Tetrahedron tolerances must be finite.")
        if absolute < 0.0 or relative < 0.0:
            raise ValueError("Tetrahedron tolerances must be nonnegative.")
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)


@dataclass(frozen=True, slots=True)
class TetraIntersectionEvidence:
    """Evidence produced while certifying an intersection result."""

    vertex_count: int = 0
    face_count: int = 0
    predicate_uncertain: bool = False
    volume_error: float = 0.0
    source_volume: float = 0.0
    target_volume: float = 0.0
    volume_only: bool = False
    predicate_mode: PredicateMode | None = None


@dataclass(frozen=True, slots=True)
class TetraIntersectionResult:
    """Typed result for an affine tetrahedron intersection.

    ``vertices`` is a read-only ``(N, 3)`` array in lexicographic order and
    ``faces`` contains outward-oriented polygon index tuples in canonical
    order.  For ``volume_only=True`` the topology is not constructed: the
    arrays are empty and the vertex and face counts are zero while the exact
    status, certified volume, and remaining evidence stay available.
    """

    status: TetraIntersectionStatus
    volume: float
    vertices: np.ndarray = field(repr=False)
    faces: tuple[tuple[int, ...], ...] = ()
    source_id: Any = 0
    target_id: Any = 0
    pair_id: str = "0:0"
    evidence: TetraIntersectionEvidence = field(default_factory=TetraIntersectionEvidence)

    def __post_init__(self) -> None:
        array = np.asarray(self.vertices, dtype=np.float64)
        if array.size == 0:
            array = np.empty((0, 3), dtype=np.float64)
        elif array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("Intersection vertices must have shape (N, 3).")
        array = np.array(array, copy=True)
        array.setflags(write=False)
        object.__setattr__(self, "vertices", array)
        object.__setattr__(self, "volume", float(self.volume))
        object.__setattr__(self, "faces", tuple(tuple(face) for face in self.faces))
        object.__setattr__(self, "pair_id", str(self.pair_id))

    @property
    def successful(self) -> bool:
        """Whether a positive-volume intersection was certified."""

        return self.status is TetraIntersectionStatus.SUCCESS

    @property
    def valid(self) -> bool:
        """Whether the result is geometrically resolved (including contact)."""

        return self.status in {
            TetraIntersectionStatus.SUCCESS,
            TetraIntersectionStatus.DISJOINT,
            TetraIntersectionStatus.ZERO_MEASURE_CONTACT,
        }

    @property
    def volume_only(self) -> bool:
        """Whether topology was intentionally omitted from this result."""

        return bool(self.evidence.volume_only)

    @property
    def canonical_vertices(self) -> np.ndarray:
        """Read-only canonical vertices."""

        return self.vertices

    @property
    def canonical_faces(self) -> tuple[tuple[int, ...], ...]:
        """Canonical outward-oriented faces."""

        return self.faces


@dataclass(frozen=True, slots=True)
class _Plane:
    """Closed half-space ``normal . (x - origin) <= 0`` of one tetrahedron face.

    ``exact_normal`` lifts ``normal`` and ``exact_offset`` is the rational
    ``normal . origin``.
    """

    normal: np.ndarray
    origin: np.ndarray
    exact_normal: _ExactPoint
    exact_offset: Fraction


def stable_tetra_pair_id(source_id: Any = 0, target_id: Any = 0) -> str:
    """Return a deterministic, process-independent ordered pair identifier."""

    return f"{_stable_id_text(source_id)}:{_stable_id_text(target_id)}"


def _stable_id_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, bool)):
        return str(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return format(float(value), ".17g")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ",".join(_stable_id_text(item) for item in value) + "]"
    return repr(value)


def _resolve_tolerance(tolerance: Any) -> TetraIntersectionTolerance:
    if tolerance is None:
        return TetraIntersectionTolerance()
    if isinstance(tolerance, TetraIntersectionTolerance):
        return tolerance
    if isinstance(tolerance, Mapping):
        absolute = tolerance.get(
            "absolute",
            tolerance.get("absolute_tolerance", tolerance.get("abs", 1.0e-13)),
        )
        relative = tolerance.get(
            "relative",
            tolerance.get("relative_tolerance", tolerance.get("rel", 1.0e-11)),
        )
        return TetraIntersectionTolerance(float(absolute), float(relative))
    if np.isscalar(tolerance):
        value = float(cast(float, tolerance))
        return TetraIntersectionTolerance(value, value)
    raise TypeError("tolerance must be a scalar, mapping, or TetraIntersectionTolerance.")


def _empty_result(
    status: TetraIntersectionStatus,
    source_id: Any,
    target_id: Any,
    pair_id: str,
    *,
    source_volume: float = 0.0,
    target_volume: float = 0.0,
    predicate_uncertain: bool = False,
    volume_only: bool = False,
    predicate_mode: PredicateMode | None = None,
) -> TetraIntersectionResult:
    return TetraIntersectionResult(
        status=status,
        volume=0.0,
        vertices=np.empty((0, 3), dtype=np.float64),
        faces=(),
        source_id=source_id,
        target_id=target_id,
        pair_id=pair_id,
        evidence=TetraIntersectionEvidence(
            predicate_uncertain=predicate_uncertain,
            source_volume=source_volume,
            target_volume=target_volume,
            volume_only=volume_only,
            predicate_mode=predicate_mode,
        ),
    )


def _coerce_tetrahedron(
    value: Any,
) -> tuple[np.ndarray | None, TetraIntersectionStatus | None]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, TetraIntersectionStatus.UNSUPPORTED
    if array.shape != (4, 3):
        return None, TetraIntersectionStatus.UNSUPPORTED
    if not bool(np.all(np.isfinite(array))):
        return None, TetraIntersectionStatus.NONFINITE_INPUT
    return np.array(array, copy=True), None


def _dot3(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    """Three-term dot products along the last axis, summed as ``(x0 + x1) + x2``.

    Elementwise IEEE operations replace ``np.dot``, whose BLAS kernel chooses the
    accumulation order per platform and CPU, so every host decides alike.
    """
    leading = first[..., 0] * second[..., 0] + first[..., 1] * second[..., 1]
    return leading + first[..., 2] * second[..., 2]


def _norm3(vector: np.ndarray, /) -> np.ndarray:
    return np.sqrt(_dot3(vector, vector))


def _det3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(_dot3(a, np.cross(b, c)))


def _edge_scale(vertices: np.ndarray) -> float:
    scale = 0.0
    for i, j in combinations(range(4), 2):
        scale = max(scale, float(_norm3(vertices[i] - vertices[j])))
    return scale


def _tetra_volume(vertices: np.ndarray) -> float:
    return (
        abs(
            _det3(
                vertices[1] - vertices[0],
                vertices[2] - vertices[0],
                vertices[3] - vertices[0],
            )
        )
        / 6.0
    )


def _validate_tetrahedron(
    vertices: np.ndarray,
    tolerance: TetraIntersectionTolerance,
    mode: PredicateMode,
) -> tuple[TetraIntersectionStatus | None, float, float, bool]:
    scale = _edge_scale(vertices)
    if not math.isfinite(scale) or scale == 0.0:
        return TetraIntersectionStatus.DEGENERATE_TETRAHEDRON, 0.0, scale, False
    determinant = _det3(
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
        vertices[3] - vertices[0],
    )
    if not math.isfinite(determinant):
        return TetraIntersectionStatus.NONFINITE_INPUT, 0.0, scale, False
    determinant_tolerance = max(
        tolerance.absolute, tolerance.relative * max(scale**3, 1.0e-300)
    )
    # The tolerance is a volume admissibility threshold; the orientation sign
    # itself comes from the certified predicate.
    if abs(determinant) <= determinant_tolerance:
        return TetraIntersectionStatus.DEGENERATE_TETRAHEDRON, 0.0, scale, False
    orientation = orient3d(vertices[0], vertices[1], vertices[2], vertices[3], mode=mode)
    if not bool(orientation.certain):
        return TetraIntersectionStatus.UNCERTAIN_PREDICATE, 0.0, scale, True
    sign = int(orientation.signs)
    if sign < 0:
        return TetraIntersectionStatus.INVERTED_TETRAHEDRON, 0.0, scale, False
    if sign == 0:
        return TetraIntersectionStatus.DEGENERATE_TETRAHEDRON, 0.0, scale, False
    return None, abs(determinant) / 6.0, scale, False


_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0
# Gradual underflow: each rounded product may lose up to 2**-1075 absolutely.
_UNDERFLOW_SLACK = 2.0**-1070
# Rationals at or beyond this magnitude round to an infinite binary64.
_BINARY64_OVERFLOW = Fraction(2**1024 - 2**970)


def _exact_point(point: np.ndarray, /) -> _ExactPoint:
    return rational_vector((float(point[0]), float(point[1]), float(point[2])))


def _residuals_exceed(
    normals: np.ndarray,
    origins: np.ndarray,
    points: np.ndarray,
    limit: float,
    exact_points: Sequence[_ExactPoint] | None = None,
    /,
) -> np.ndarray:
    """Decide ``normal . (point - origin) > limit * |normal|`` exactly, row by row.

    ``normals``, ``origins``, and ``points`` are ``(k, 3)`` binary64 rows and
    ``limit`` is a nonnegative binary64.  When ``exact_points`` is given, row
    ``k`` of ``points`` is its correctly rounded value and the decision is about
    the rational point.

    A fixed-order filter decides most rows.  The residual is evaluated as
    ``r~ = fl((n0 d0 + n1 d1) + n2 d2)`` with ``d = fl(point - origin)``.  Each
    exact term ``n_i (p_i - o_i)`` meets at most four roundings: the
    difference, the product, and two additions.  Any summation order has the
    same count, so ``|r - r~| <= gamma_4 sum |n_i (p_i - o_i)|``.  The computed
    ``S~ = fl(sum |n_i| |d_i|)`` exceeds ``(1 - u)**4 sum |n_i (p_i - o_i)|``,
    and ``8 u S~`` dominates ``(1 + u) gamma_4 / (1 - u)**4 S~``.  A rounded
    point adds ``|n . (x - x~)| <= sum |n_i| (u |x~_i| + 2**-1075)``, which
    ``2 u fl(sum |n_i| |x~_i|) + 2**-1073 sum |n_i|`` covers.  Underflowing
    products add ``_UNDERFLOW_SLACK``.  ``|normal|`` lies between the directed
    norm bounds, and one outward step encloses its product with ``limit``.  A
    row is decided when ``fl(r~ - limit_upper)`` exceeds the bound or
    ``fl(limit_lower - r~)`` reaches it.  The comparison's own rounding is
    covered by the factor ``1 + u``.  The remaining rows, including every
    non-finite one, are decided in rational arithmetic on the binary64 inputs.
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        differences = points - origins
        residual = _dot3(normals, differences)
        magnitude = np.abs(normals)
        bound = 8.0 * _UNIT_ROUNDOFF * _dot3(magnitude, np.abs(differences))
        if exact_points is not None:
            bound = (
                bound
                + 2.0 * _UNIT_ROUNDOFF * _dot3(magnitude, np.abs(points))
                + 2.0**-1073 * ((magnitude[:, 0] + magnitude[:, 1]) + magnitude[:, 2])
            )
        bound = bound + _UNDERFLOW_SLACK
        limit_upper = np.nextafter(limit * binary64_norm_upper(normals), np.inf)
        limit_lower = np.maximum(
            np.nextafter(limit * binary64_norm_lower(normals), -np.inf), 0.0
        )
        finite = np.isfinite(residual) & np.isfinite(bound)
        exceeds = finite & ((residual - limit_upper) > bound)
        within = finite & ((limit_lower - residual) >= bound)
    exact_limit = Fraction(limit)
    for row in np.flatnonzero(~(exceeds | within)).tolist():
        normal = _exact_point(normals[row])
        point = _exact_point(points[row]) if exact_points is None else exact_points[row]
        exact_residual = rational_dot(
            normal, _exact_difference(point, _exact_point(origins[row]))
        )
        exceeds[row] = (
            exact_residual > 0
            and exact_residual * exact_residual
            > exact_limit * exact_limit * rational_dot(normal, normal)
        )
    return exceeds


def _tetra_planes(vertices: np.ndarray) -> tuple[_Plane, ...]:
    faces = [[index for index in range(4) if index != opposite] for opposite in range(4)]
    origins = np.asarray([vertices[face[0]] for face in faces], dtype=np.float64)
    normals = np.asarray(
        [
            np.cross(
                vertices[face[1]] - vertices[face[0]],
                vertices[face[2]] - vertices[face[0]],
            )
            for face in faces
        ],
        dtype=np.float64,
    )
    # Orient each normal away from the opposite vertex, by the exact sign.
    outward = ~_residuals_exceed(normals, origins, vertices, 0.0)
    planes: list[_Plane] = []
    for normal, origin, keep in zip(normals, origins, outward.tolist(), strict=True):
        oriented = np.array(normal if keep else -normal, dtype=np.float64)
        oriented.setflags(write=False)
        anchor = np.array(origin, dtype=np.float64)
        anchor.setflags(write=False)
        exact_normal = _exact_point(oriented)
        planes.append(
            _Plane(
                oriented,
                anchor,
                exact_normal,
                rational_dot(exact_normal, _exact_point(anchor)),
            )
        )
    return tuple(planes)


def _inside(
    points: np.ndarray,
    planes: tuple[_Plane, ...],
    limit: float,
    exact_points: Sequence[_ExactPoint] | None = None,
    /,
) -> np.ndarray:
    """Rows of ``points`` within ``limit * |normal|`` of every closed half-space.

    The admission is exact on the binary64 inputs (or on ``exact_points``), so
    it does not depend on how a platform accumulates dot products.
    """
    count = len(points)
    normals = np.asarray([plane.normal for plane in planes], dtype=np.float64)
    origins = np.asarray([plane.origin for plane in planes], dtype=np.float64)
    exceeds = _residuals_exceed(
        np.tile(normals, (count, 1)),
        np.tile(origins, (count, 1)),
        np.repeat(np.asarray(points, dtype=np.float64), len(planes), axis=0),
        limit,
        None
        if exact_points is None
        else tuple(point for point in exact_points for _ in planes),
    )
    return ~np.any(exceeds.reshape(count, len(planes)), axis=1)


def _triple_intersection(
    planes: tuple[_Plane, ...],
    indices: tuple[int, int, int],
    orientation: int,
    certain: bool,
) -> tuple[_ExactPoint | None, bool]:
    """Exact common point of three face planes, or ``None`` with uncertainty."""
    matrix = np.stack([planes[index].normal for index in indices], axis=0)
    # Coincident source/target faces are represented by independently built
    # normals.  Their cross product can carry a few ulps even though the
    # geometric planes are identical; this is a rank-zero combination, not an
    # uncertain predicate.
    for first, second in combinations(range(3), 2):
        normal_scale = max(
            1.0,
            float(_norm3(matrix[first])) * float(_norm3(matrix[second])),
        )
        if float(_norm3(np.cross(matrix[first], matrix[second]))) <= (
            np.finfo(np.float64).eps * 128.0 * normal_scale
        ):
            return None, False
    if not certain:
        return None, True
    if orientation == 0:
        return None, False
    # The certified nonzero orientation is the sign of det(matrix), so Cramer's
    # rule over the exact planes normal . x = normal . origin is well defined.
    first_normal, second_normal, third_normal = (
        planes[index].exact_normal for index in indices
    )
    first_offset, second_offset, third_offset = (
        planes[index].exact_offset for index in indices
    )
    second_third = rational_cross(second_normal, third_normal)
    third_first = rational_cross(third_normal, first_normal)
    first_second = rational_cross(first_normal, second_normal)
    determinant = rational_dot(first_normal, second_third)
    first_coordinate, second_coordinate, third_coordinate = (
        (
            first_offset * second_third[axis]
            + second_offset * third_first[axis]
            + third_offset * first_second[axis]
        )
        / determinant
        for axis in range(3)
    )
    return (first_coordinate, second_coordinate, third_coordinate), False


def _rounded_point(point: _ExactPoint, /) -> np.ndarray:
    """Correctly rounded binary64 coordinates; overflow becomes a signed infinity."""
    return np.asarray(
        [
            float(value)
            if abs(value) < _BINARY64_OVERFLOW
            else (math.inf if value > 0 else -math.inf)
            for value in point
        ],
        dtype=np.float64,
    )


def _append_unique(
    candidates: list[tuple[np.ndarray, _ExactPoint]],
    point: np.ndarray,
    exact_point: _ExactPoint,
    point_tolerance: float,
    /,
) -> None:
    for index, (existing, _) in enumerate(candidates):
        if float(_norm3(point - existing)) <= point_tolerance:
            if tuple(float(item) for item in point) < tuple(
                float(item) for item in existing
            ):
                candidates[index] = (point, exact_point)
            return
    candidates.append((point, exact_point))


def _face_order(
    indices: list[int],
    vertices: np.ndarray,
    normal: np.ndarray,
) -> tuple[int, ...] | None:
    if len(indices) < 3:
        return None
    centroid = np.mean(vertices[indices], axis=0)
    unit_normal = normal / _norm3(normal)
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(unit_normal)))] = 1.0
    first = np.cross(unit_normal, axis)
    first_norm = float(_norm3(first))
    if first_norm == 0.0 or not math.isfinite(first_norm):
        return None
    first /= first_norm
    second = np.cross(unit_normal, first)
    angles: list[tuple[float, int]] = []
    for index in indices:
        delta = vertices[index] - centroid
        angles.append(
            (math.atan2(float(_dot3(delta, second)), float(_dot3(delta, first))), index)
        )
    ordered = [index for _, index in sorted(angles)]
    area_vector = np.zeros(3, dtype=np.float64)
    for index, next_index in zip(ordered, ordered[1:] + ordered[:1]):
        area_vector += np.cross(
            vertices[index] - centroid, vertices[next_index] - centroid
        )
    if float(_dot3(area_vector, unit_normal)) < 0.0:
        ordered.reverse()
    pivot = min(range(len(ordered)), key=lambda index: ordered[index])
    ordered = ordered[pivot:] + ordered[:pivot]
    return tuple(ordered)


type _ExactPoint = tuple[Fraction, Fraction, Fraction]
type _ExactTetrahedron = tuple[_ExactPoint, _ExactPoint, _ExactPoint, _ExactPoint]


def _exact_difference(first: _ExactPoint, second: _ExactPoint, /) -> _ExactPoint:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _exact_halfspaces(
    tetrahedron: _ExactTetrahedron, /
) -> tuple[tuple[_ExactPoint, Fraction], ...]:
    """Exact closed half-spaces ``normal . x + offset >= 0`` bounding a tetrahedron."""
    halfspaces: list[tuple[_ExactPoint, Fraction]] = []
    for opposite in range(4):
        first, second, third = (
            tetrahedron[index] for index in range(4) if index != opposite
        )
        normal = rational_cross(
            _exact_difference(second, first), _exact_difference(third, first)
        )
        offset = -rational_dot(normal, first)
        if rational_dot(normal, tetrahedron[opposite]) + offset < 0:
            normal = (-normal[0], -normal[1], -normal[2])
            offset = -offset
        halfspaces.append((normal, offset))
    return tuple(halfspaces)


def _prism_tetrahedra(
    bottom: tuple[_ExactPoint, _ExactPoint, _ExactPoint],
    top: tuple[_ExactPoint, _ExactPoint, _ExactPoint],
    /,
) -> tuple[_ExactTetrahedron, _ExactTetrahedron, _ExactTetrahedron]:
    """Staircase triangulation of a convex prism with lateral edges ``bottom[k]-top[k]``."""
    return (
        (bottom[0], bottom[1], bottom[2], top[2]),
        (bottom[0], bottom[1], top[1], top[2]),
        (bottom[0], top[0], top[1], top[2]),
    )


def _clip_exact_tetrahedron(
    tetrahedron: _ExactTetrahedron, normal: _ExactPoint, offset: Fraction, /
) -> tuple[_ExactTetrahedron, ...]:
    """Tetrahedra tiling ``tetrahedron`` intersected with ``normal . x + offset >= 0``."""
    values = tuple(rational_dot(normal, point) + offset for point in tetrahedron)
    inside = tuple(index for index in range(4) if values[index] >= 0)
    outside = tuple(index for index in range(4) if values[index] < 0)

    def cut(kept: int, dropped: int, /) -> _ExactPoint:
        weight = values[kept] / (values[kept] - values[dropped])
        start, stop = tetrahedron[kept], tetrahedron[dropped]
        return (
            start[0] + weight * (stop[0] - start[0]),
            start[1] + weight * (stop[1] - start[1]),
            start[2] + weight * (stop[2] - start[2]),
        )

    match len(inside):
        case 4:
            return (tetrahedron,)
        case 0:
            return ()
        case 1:
            kept = inside[0]
            return (
                (
                    tetrahedron[kept],
                    cut(kept, outside[0]),
                    cut(kept, outside[1]),
                    cut(kept, outside[2]),
                ),
            )
        case 2:
            first, second = inside
            return _prism_tetrahedra(
                (
                    tetrahedron[first],
                    cut(first, outside[0]),
                    cut(first, outside[1]),
                ),
                (
                    tetrahedron[second],
                    cut(second, outside[0]),
                    cut(second, outside[1]),
                ),
            )
        case 3:
            dropped = outside[0]
            return _prism_tetrahedra(
                (tetrahedron[inside[0]], tetrahedron[inside[1]], tetrahedron[inside[2]]),
                (
                    cut(inside[0], dropped),
                    cut(inside[1], dropped),
                    cut(inside[2], dropped),
                ),
            )
        case _:
            raise RuntimeError("A tetrahedron has exactly four vertices.")


def _exact_intersection_pieces(
    source: np.ndarray, target: np.ndarray, /
) -> tuple[_ExactTetrahedron, ...]:
    """Tetrahedra whose union is exactly the closed intersection of the inputs.

    ``source`` is clipped by the four closed half-spaces of ``target`` in exact
    rational arithmetic on the binary64 vertices.  Each clip replaces a
    (possibly degenerate) tetrahedron by tetrahedra covering exactly its kept
    part: nothing when every vertex is strictly outside, itself, a corner
    tetrahedron, or a convex prism with planar lateral faces split by the
    staircase triangulation.  The pieces are interior-disjoint, so the result is
    empty exactly when the closed intersection is empty and the absolute
    determinants sum to its exact volume.
    """

    def exact_tetrahedron(vertices: np.ndarray, /) -> _ExactTetrahedron:
        first, second, third, fourth = (
            rational_vector((float(row[0]), float(row[1]), float(row[2])))
            for row in vertices
        )
        return first, second, third, fourth

    pieces: tuple[_ExactTetrahedron, ...] = (exact_tetrahedron(source),)
    for normal, offset in _exact_halfspaces(exact_tetrahedron(target)):
        pieces = tuple(
            clipped
            for piece in pieces
            for clipped in _clip_exact_tetrahedron(piece, normal, offset)
        )
    return pieces


def _exact_pieces_volume(pieces: tuple[_ExactTetrahedron, ...], /) -> Fraction:
    """Exact volume of a union of interior-disjoint tetrahedra."""
    return (
        sum(
            (
                abs(
                    rational_dot(
                        _exact_difference(piece[1], piece[0]),
                        rational_cross(
                            _exact_difference(piece[2], piece[0]),
                            _exact_difference(piece[3], piece[0]),
                        ),
                    )
                )
                for piece in pieces
            ),
            Fraction(0),
        )
        / 6
    )


def _build_faces(
    vertices: np.ndarray,
    exact_vertices: tuple[_ExactPoint, ...],
    planes: tuple[_Plane, ...],
    tolerance: TetraIntersectionTolerance,
    scale: float,
    /,
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    area_tolerance = max(tolerance.absolute, tolerance.relative * max(scale**2, 1.0e-300))
    boundary_limit = tolerance.absolute + tolerance.relative * max(scale, 1.0e-300)
    for plane in planes:
        norm = float(_norm3(plane.normal))
        if norm == 0.0 or not math.isfinite(norm):
            return (), True
    # A vertex bounds a face when its exact residual satisfies
    # |normal . (vertex - origin)| <= boundary_limit * |normal|.
    count = len(vertices)
    normals = np.repeat(
        np.asarray([plane.normal for plane in planes], dtype=np.float64), count, axis=0
    )
    origins = np.repeat(
        np.asarray([plane.origin for plane in planes], dtype=np.float64), count, axis=0
    )
    points = np.tile(vertices, (len(planes), 1))
    exact_points = exact_vertices * len(planes)
    off_plane = (
        _residuals_exceed(normals, origins, points, boundary_limit, exact_points)
        | _residuals_exceed(-normals, origins, points, boundary_limit, exact_points)
    ).reshape(len(planes), count)
    faces_by_key: dict[tuple[int, ...], tuple[int, ...]] = {}
    for plane, outside in zip(planes, off_plane, strict=True):
        boundary = np.flatnonzero(~outside).tolist()
        if len(boundary) < 3:
            continue
        ordered = _face_order(boundary, vertices, plane.normal)
        if ordered is None:
            return (), True
        area_vector = np.zeros(3, dtype=np.float64)
        center = np.mean(vertices[list(ordered)], axis=0)
        for index, next_index in zip(ordered, ordered[1:] + ordered[:1]):
            area_vector += (
                np.cross(vertices[index] - center, vertices[next_index] - center) * 0.5
            )
        if float(_norm3(area_vector)) <= area_tolerance:
            continue
        key = tuple(sorted(ordered))
        faces_by_key.setdefault(key, ordered)
    faces = tuple(faces_by_key[key] for key in sorted(faces_by_key))
    return faces, False


def _closed_surface(vertex_count: int, faces: tuple[tuple[int, ...], ...], /) -> bool:
    """Whether ``faces`` close a polytope surface over all ``vertex_count`` vertices.

    Every edge borders exactly two faces, every vertex lies on a face, and the
    Euler characteristic ``V - E + F`` is 2.  A construction that lost a vertex
    of a solid intersection leaves an edge on a single face.
    """
    edges: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in zip(face, face[1:] + face[:1]):
            key = (min(first, second), max(first, second))
            edges[key] = edges.get(key, 0) + 1
    return (
        len(faces) >= 4
        and {index for face in faces for index in face} == set(range(vertex_count))
        and all(count == 2 for count in edges.values())
        and vertex_count - len(edges) + len(faces) == 2
    )


def _polytope_topology(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: TetraIntersectionTolerance,
    scale: float,
    mode: PredicateMode,
    solid: bool,
    /,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]] | None:
    """Constructed vertices and faces of a nonempty intersection, or ``None``.

    Candidates are the input vertices and the exact common points of plane
    triples; each is admitted by its exact residuals and reported correctly
    rounded.  ``None`` reports an unresolved construction: an uncertain
    plane-triple orientation, an admitted point beyond the binary64 range, no
    constructed vertex, an unorderable face, or, for a ``solid`` intersection,
    faces that do not close a polytope surface.
    """
    planes = _tetra_planes(source) + _tetra_planes(target)
    candidates: list[tuple[np.ndarray, _ExactPoint]] = []
    point_tolerance = tolerance.absolute + tolerance.relative * max(scale, 1.0e-300)
    inputs = np.concatenate((source, target), axis=0)
    for point, admitted in zip(
        inputs, _inside(inputs, planes, point_tolerance).tolist(), strict=True
    ):
        if admitted:
            _append_unique(candidates, point, _exact_point(point), point_tolerance)
    triples = tuple(combinations(range(len(planes)), 3))
    triple_normals = np.asarray(
        [[planes[index].normal for index in indices] for indices in triples],
        dtype=np.float64,
    )
    triple_orientation = orient3d(
        np.zeros(3, dtype=np.float64),
        triple_normals[:, 0],
        triple_normals[:, 1],
        triple_normals[:, 2],
        mode=mode,
    )
    triple_points: list[_ExactPoint] = []
    for triple_index, indices in enumerate(triples):
        point, point_uncertain = _triple_intersection(
            planes,
            indices,
            int(triple_orientation.signs[triple_index]),
            bool(triple_orientation.certain[triple_index]),
        )
        if point_uncertain:
            return None
        if point is not None:
            triple_points.append(point)
    if triple_points:
        rounded = [_rounded_point(point) for point in triple_points]
        admitted = _inside(
            np.asarray(rounded, dtype=np.float64), planes, point_tolerance, triple_points
        )
        for point, exact, keep in zip(
            rounded, triple_points, admitted.tolist(), strict=True
        ):
            if not keep:
                continue
            if not bool(np.all(np.isfinite(point))):
                return None
            _append_unique(candidates, point, exact, point_tolerance)
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: tuple(float(value) for value in candidate[0]))
    vertices = np.asarray([point for point, _ in candidates], dtype=np.float64)
    vertices.setflags(write=False)
    faces, face_failure = _build_faces(
        vertices, tuple(exact for _, exact in candidates), planes, tolerance, scale
    )
    if face_failure or (solid and not _closed_surface(len(vertices), faces)):
        return None
    return vertices, faces


def intersect_tetrahedra(
    source_vertices: Any,
    target_vertices: Any,
    *,
    source_id: Any = 0,
    target_id: Any = 0,
    tolerance: Any = None,
    volume_only: bool = False,
    precision: GeometryPrecisionPolicy | None = None,
) -> TetraIntersectionResult:
    """Intersect two affine tetrahedra with complete finite candidate enumeration.

    The status is decided exactly on the binary64 inputs: the source is clipped
    by the closed target half-spaces in rational arithmetic, so ``DISJOINT``
    means the closed tetrahedra share no point, and a nonempty intersection is
    ``SUCCESS`` when its exact volume exceeds the volume tolerance and
    ``ZERO_MEASURE_CONTACT`` otherwise.  ``volume`` and ``volume_error`` enclose
    that exact volume.  The reported vertices and faces come from the
    tolerance-based enumeration: input vertices and exact plane-triple points
    are admitted by exact residual predicates and reported correctly rounded,
    so no decision depends on the platform's dot-product accumulation order.
    Filtered orientation uncertainty, an unorderable constructed face, or a
    construction that loses a nonempty intersection (for ``SUCCESS``, faces
    that do not close a polytope surface) returns ``UNCERTAIN_PREDICATE``.
    ``volume_only`` skips that construction, so its status never depends on
    it.  No candidate-count or topology cap truncates work.
    """

    policy = GeometryPrecisionPolicy() if precision is None else precision
    if not isinstance(policy, GeometryPrecisionPolicy):
        raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
    mode = resolve_host_predicate_mode(policy.predicate_mode)
    pair_id = stable_tetra_pair_id(source_id, target_id)
    try:
        tolerance_ = _resolve_tolerance(tolerance)
    except (TypeError, ValueError, OverflowError):
        return _empty_result(
            TetraIntersectionStatus.UNSUPPORTED,
            source_id,
            target_id,
            pair_id,
            predicate_mode=mode,
        )
    source, source_status = _coerce_tetrahedron(source_vertices)
    target, target_status = _coerce_tetrahedron(target_vertices)
    if source_status is not None:
        return _empty_result(
            source_status, source_id, target_id, pair_id, predicate_mode=mode
        )
    if target_status is not None:
        return _empty_result(
            target_status, source_id, target_id, pair_id, predicate_mode=mode
        )
    if source is None or target is None:
        raise RuntimeError("tetrahedron coercion lost a validated input")
    source_failure, source_volume, source_scale, source_uncertain = _validate_tetrahedron(
        source, tolerance_, mode
    )
    if source_failure is not None:
        return _empty_result(
            source_failure,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            predicate_uncertain=source_uncertain,
            predicate_mode=mode,
        )
    target_failure, target_volume, target_scale, target_uncertain = _validate_tetrahedron(
        target, tolerance_, mode
    )
    if target_failure is not None:
        return _empty_result(
            target_failure,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            target_volume=target_volume,
            predicate_uncertain=target_uncertain,
            predicate_mode=mode,
        )
    scale = max(source_scale, target_scale)
    pieces = _exact_intersection_pieces(source, target)
    if not pieces:
        return _empty_result(
            TetraIntersectionStatus.DISJOINT,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            target_volume=target_volume,
            predicate_mode=mode,
        )
    exact_volume = _exact_pieces_volume(pieces)
    volume = fraction_down(exact_volume)
    volume_error = fraction_up(exact_volume - Fraction(volume))
    volume_tolerance = max(
        tolerance_.absolute, tolerance_.relative * max(scale**3, 1.0e-300)
    )
    status = (
        TetraIntersectionStatus.SUCCESS
        if exact_volume > Fraction(volume_tolerance)
        else TetraIntersectionStatus.ZERO_MEASURE_CONTACT
    )
    vertices = np.empty((0, 3), dtype=np.float64)
    faces: tuple[tuple[int, ...], ...] = ()
    if not volume_only:
        topology = _polytope_topology(
            source,
            target,
            tolerance_,
            scale,
            mode,
            status is TetraIntersectionStatus.SUCCESS,
        )
        if topology is None:
            return _empty_result(
                TetraIntersectionStatus.UNCERTAIN_PREDICATE,
                source_id,
                target_id,
                pair_id,
                source_volume=source_volume,
                target_volume=target_volume,
                predicate_uncertain=True,
                predicate_mode=mode,
            )
        vertices, faces = topology
    return TetraIntersectionResult(
        status=status,
        volume=volume,
        vertices=vertices,
        faces=faces,
        source_id=source_id,
        target_id=target_id,
        pair_id=pair_id,
        evidence=TetraIntersectionEvidence(
            vertex_count=len(vertices),
            face_count=len(faces),
            predicate_uncertain=False,
            volume_error=volume_error,
            source_volume=source_volume,
            target_volume=target_volume,
            volume_only=bool(volume_only),
            predicate_mode=mode,
        ),
    )


__all__ = [
    "TetraIntersectionEvidence",
    "TetraIntersectionResult",
    "TetraIntersectionStatus",
    "TetraIntersectionTolerance",
    "intersect_tetrahedra",
    "stable_tetra_pair_id",
]
