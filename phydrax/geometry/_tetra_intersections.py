#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative host-side intersections of affine tetrahedra.

The implementation deliberately does not contain JAX primitives.  A tetrahedron
intersection is a small, topology-changing geometry operation and is therefore
prepared on the host before a remap is committed.  The intersection polytope is
formed by enumerating vertices of the eight half-spaces (four from each input
cell).  Every numerical decision is conservative: malformed cells, uncertain
predicates, and exhausted limits return a typed non-success result rather than a
plausible but unverifiable volume.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, cast

import numpy as np


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
    CANDIDATE_LIMIT = "candidate_limit"


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

    @property
    def absolute_tolerance(self) -> float:
        """Compatibility spelling used by geometry policies."""

        return self.absolute

    @property
    def relative_tolerance(self) -> float:
        """Compatibility spelling used by geometry policies."""

        return self.relative


@dataclass(frozen=True, slots=True)
class TetraIntersectionLimits:
    """Hard bounds on temporary and returned intersection topology."""

    max_candidates: int = 64
    max_vertices: int = 64
    max_faces: int = 16

    def __post_init__(self) -> None:
        values = (self.max_candidates, self.max_vertices, self.max_faces)
        if any(isinstance(value, bool) for value in values):
            raise ValueError("Intersection limits must be integer counts.")
        if any(int(value) != value or int(value) <= 0 for value in values):
            raise ValueError("Intersection limits must be positive integers.")
        object.__setattr__(self, "max_candidates", int(self.max_candidates))
        object.__setattr__(self, "max_vertices", int(self.max_vertices))
        object.__setattr__(self, "max_faces", int(self.max_faces))

    @property
    def max_candidate_vertices(self) -> int:
        """Explicit spelling for the candidate-vertex allocation bound."""

        return self.max_candidates


@dataclass(frozen=True, slots=True)
class TetraIntersectionEvidence:
    """Evidence produced while certifying an intersection result."""

    candidate_count: int = 0
    vertex_count: int = 0
    face_count: int = 0
    predicate_uncertain: bool = False
    volume_error: float = 0.0
    source_volume: float = 0.0
    target_volume: float = 0.0
    volume_only: bool = False


@dataclass(frozen=True, slots=True)
class TetraIntersectionResult:
    """Typed result for an affine tetrahedron intersection.

    ``vertices`` is a read-only ``(N, 3)`` array in lexicographic order and
    ``faces`` contains outward-oriented polygon index tuples in canonical
    order.  For ``volume_only=True`` the arrays are intentionally empty while
    the certified volume and evidence remain available.
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
        object.__setattr__(
            self, "faces", tuple(tuple(int(i) for i in face) for face in self.faces)
        )
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
    normal: np.ndarray
    offset: float


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


def _resolve_limits(limits: Any) -> TetraIntersectionLimits:
    if limits is None:
        return TetraIntersectionLimits()
    if isinstance(limits, TetraIntersectionLimits):
        return limits
    if isinstance(limits, Mapping):
        candidate = limits.get(
            "max_candidates",
            limits.get(
                "max_candidate_vertices",
                limits.get("max_candidate_points", limits.get("candidate_limit", 64)),
            ),
        )
        vertices = limits.get("max_vertices", limits.get("max_output_vertices", 64))
        faces = limits.get("max_faces", limits.get("max_output_faces", 16))
        return TetraIntersectionLimits(int(candidate), int(vertices), int(faces))
    if np.isscalar(limits) and not isinstance(limits, bool):
        value = int(cast(int, limits))
        return TetraIntersectionLimits(value, value, 16)
    raise TypeError("limits must be an integer, mapping, or TetraIntersectionLimits.")


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
    candidate_count: int = 0,
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
            candidate_count=candidate_count,
            predicate_uncertain=predicate_uncertain,
            source_volume=source_volume,
            target_volume=target_volume,
            volume_only=volume_only,
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


def _det3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.dot(a, np.cross(b, c)))


def _det3_longdouble(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.longdouble:
    aa = np.asarray(a, dtype=np.longdouble)
    bb = np.asarray(b, dtype=np.longdouble)
    cc = np.asarray(c, dtype=np.longdouble)
    return (
        aa[0] * (bb[1] * cc[2] - bb[2] * cc[1])
        - aa[1] * (bb[0] * cc[2] - bb[2] * cc[0])
        + aa[2] * (bb[0] * cc[1] - bb[1] * cc[0])
    )


def _edge_scale(vertices: np.ndarray) -> float:
    scale = 0.0
    for i, j in combinations(range(4), 2):
        scale = max(scale, float(np.linalg.norm(vertices[i] - vertices[j])))
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
) -> tuple[TetraIntersectionStatus | None, float, float, bool]:
    scale = _edge_scale(vertices)
    if not math.isfinite(scale) or scale == 0.0:
        return TetraIntersectionStatus.DEGENERATE_TETRAHEDRON, 0.0, scale, False
    a = vertices[1] - vertices[0]
    b = vertices[2] - vertices[0]
    c = vertices[3] - vertices[0]
    determinant = _det3(a, b, c)
    determinant_ld = _det3_longdouble(a, b, c)
    if not math.isfinite(determinant) or not np.isfinite(determinant_ld):
        return TetraIntersectionStatus.NONFINITE_INPUT, 0.0, scale, False
    max_term = max(
        float(abs(a[i] * b[j] * c[k]))
        for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    )
    roundoff = np.finfo(np.float64).eps * max(1.0, max_term) * 32.0
    if (
        abs(determinant) <= roundoff
        and determinant != 0.0
        and np.sign(determinant) != np.sign(float(determinant_ld))
    ):
        return TetraIntersectionStatus.UNCERTAIN_PREDICATE, 0.0, scale, True
    determinant_tolerance = max(
        tolerance.absolute, tolerance.relative * max(scale**3, 1.0e-300)
    )
    if determinant_ld < -determinant_tolerance:
        return TetraIntersectionStatus.INVERTED_TETRAHEDRON, 0.0, scale, False
    if abs(determinant_ld) <= determinant_tolerance:
        return TetraIntersectionStatus.DEGENERATE_TETRAHEDRON, 0.0, scale, False
    return None, float(abs(determinant_ld) / 6.0), scale, False


def _tetra_planes(vertices: np.ndarray) -> tuple[_Plane, ...]:
    planes: list[_Plane] = []
    for opposite in range(4):
        face = [index for index in range(4) if index != opposite]
        origin, first, second = (vertices[index] for index in face)
        normal = np.cross(first - origin, second - origin)
        if float(np.dot(normal, vertices[opposite] - origin)) > 0.0:
            normal = -normal
        offset = float(np.dot(normal, origin))
        normal = np.asarray(normal, dtype=np.float64)
        normal.setflags(write=False)
        planes.append(_Plane(normal, offset))
    return tuple(planes)


def _inside(
    point: np.ndarray,
    planes: tuple[_Plane, ...],
    tolerance: TetraIntersectionTolerance,
    scale: float,
) -> bool:
    for plane in planes:
        normal = plane.normal
        normal_norm = float(np.linalg.norm(normal))
        residual = float(np.dot(normal, point) - plane.offset)
        limit = (
            tolerance.absolute + tolerance.relative * max(scale, 1.0e-300)
        ) * normal_norm
        if residual > limit:
            return False
    return True


def _triple_intersection(
    planes: tuple[_Plane, ...],
    indices: tuple[int, int, int],
    scale: float,
) -> tuple[np.ndarray | None, bool]:
    matrix = np.stack([planes[index].normal for index in indices], axis=0)
    rhs = np.asarray([planes[index].offset for index in indices], dtype=np.float64)
    # Coincident source/target faces are represented by independently built
    # normals.  Their cross product can carry a few ulps even though the
    # geometric planes are identical; this is a rank-zero combination, not an
    # uncertain predicate.
    for first, second in combinations(range(3), 2):
        normal_scale = max(
            1.0,
            float(np.linalg.norm(matrix[first])) * float(np.linalg.norm(matrix[second])),
        )
        if float(np.linalg.norm(np.cross(matrix[first], matrix[second]))) <= (
            np.finfo(np.float64).eps * 128.0 * normal_scale
        ):
            return None, False
    determinant = _det3(matrix[0], matrix[1], matrix[2])
    determinant_ld = _det3_longdouble(matrix[0], matrix[1], matrix[2])
    max_term = max(
        float(abs(matrix[0, i] * matrix[1, j] * matrix[2, k]))
        for i, j, k in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    )
    roundoff = np.finfo(np.float64).eps * max(max_term, 1.0e-300) * 64.0
    if determinant_ld == 0:
        return None, False
    if not np.isfinite(determinant_ld) or abs(determinant) <= roundoff:
        return None, True
    try:
        point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None, True
    if not bool(np.all(np.isfinite(point))):
        return None, True
    return point, False


def _append_unique(
    candidates: list[np.ndarray],
    point: np.ndarray,
    point_tolerance: float,
    maximum: int,
) -> bool:
    for index, existing in enumerate(candidates):
        if float(np.linalg.norm(point - existing)) <= point_tolerance:
            if tuple(float(item) for item in point) < tuple(
                float(item) for item in existing
            ):
                candidates[index] = point
            return True
    if len(candidates) >= maximum:
        return False
    candidates.append(point)
    return True


def _face_order(
    indices: list[int],
    vertices: np.ndarray,
    normal: np.ndarray,
) -> tuple[int, ...] | None:
    if len(indices) < 3:
        return None
    centroid = np.mean(vertices[indices], axis=0)
    unit_normal = normal / np.linalg.norm(normal)
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(unit_normal)))] = 1.0
    first = np.cross(unit_normal, axis)
    first_norm = float(np.linalg.norm(first))
    if first_norm == 0.0 or not math.isfinite(first_norm):
        return None
    first /= first_norm
    second = np.cross(unit_normal, first)
    angles: list[tuple[float, int]] = []
    for index in indices:
        delta = vertices[index] - centroid
        angles.append(
            (math.atan2(float(np.dot(delta, second)), float(np.dot(delta, first))), index)
        )
    ordered = [index for _, index in sorted(angles)]
    area_vector = np.zeros(3, dtype=np.float64)
    for index, next_index in zip(ordered, ordered[1:] + ordered[:1]):
        area_vector += np.cross(
            vertices[index] - centroid, vertices[next_index] - centroid
        )
    if float(np.dot(area_vector, unit_normal)) < 0.0:
        ordered.reverse()
    pivot = min(range(len(ordered)), key=lambda index: ordered[index])
    ordered = ordered[pivot:] + ordered[:pivot]
    return tuple(ordered)


def _compensated_volume(
    vertices: np.ndarray, faces: tuple[tuple[int, ...], ...]
) -> tuple[float, float]:
    if not faces:
        return 0.0, 0.0
    origin = np.mean(vertices, axis=0)
    total = 0.0
    correction = 0.0
    scale = max(float(np.max(np.abs(vertices - origin))), 1.0)
    terms: list[float] = []
    for face in faces:
        anchor = vertices[face[0]] - origin
        for index in range(1, len(face) - 1):
            first = vertices[face[index]] - origin
            second = vertices[face[index + 1]] - origin
            term = float(np.dot(anchor, np.cross(first, second))) / 6.0
            terms.append(term)
            updated = total + term
            if abs(total) >= abs(term):
                correction += (total - updated) + term
            else:
                correction += (term - updated) + total
            total = updated
    signed = total + correction
    volume = abs(float(signed))
    # The accumulated correction is an evidence bound, not a volume adjustment.
    error = abs(float(correction)) + np.finfo(np.float64).eps * max(1.0, scale**3) * max(
        1, len(terms)
    )
    return volume, error


def _build_faces(
    vertices: np.ndarray,
    planes: tuple[_Plane, ...],
    tolerance: TetraIntersectionTolerance,
    scale: float,
    maximum: int,
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    area_tolerance = max(tolerance.absolute, tolerance.relative * max(scale**2, 1.0e-300))
    faces_by_key: dict[tuple[int, ...], tuple[int, ...]] = {}
    for plane in planes:
        norm = float(np.linalg.norm(plane.normal))
        if norm == 0.0 or not math.isfinite(norm):
            return (), True
        boundary: list[int] = []
        for index, point in enumerate(vertices):
            distance = abs(float(np.dot(plane.normal, point) - plane.offset)) / norm
            boundary_limit = tolerance.absolute + tolerance.relative * max(
                scale, 1.0e-300
            )
            if distance <= boundary_limit:
                boundary.append(index)
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
        if float(np.linalg.norm(area_vector)) <= area_tolerance:
            continue
        key = tuple(sorted(ordered))
        faces_by_key.setdefault(key, ordered)
        if len(faces_by_key) > maximum:
            return (), True
    faces = tuple(faces_by_key[key] for key in sorted(faces_by_key))
    return faces, False


def intersect_tetrahedra(
    source_vertices: Any,
    target_vertices: Any,
    *,
    source_id: Any = 0,
    target_id: Any = 0,
    tolerance: Any = None,
    limits: Any = None,
    volume_only: bool = False,
) -> TetraIntersectionResult:
    """Intersect two positively oriented affine tetrahedra on the host.

    The operation is deterministic under vertex permutation.  It returns
    ``DISJOINT`` for separated cells and ``ZERO_MEASURE_CONTACT`` for a face,
    edge, or vertex contact.  No tolerance-based volume or row normalization is
    performed.
    """

    pair_id = stable_tetra_pair_id(source_id, target_id)
    try:
        tolerance_ = _resolve_tolerance(tolerance)
        limits_ = _resolve_limits(limits)
    except (TypeError, ValueError, OverflowError):
        return _empty_result(
            TetraIntersectionStatus.UNSUPPORTED, source_id, target_id, pair_id
        )
    source, source_status = _coerce_tetrahedron(source_vertices)
    target, target_status = _coerce_tetrahedron(target_vertices)
    if source_status is not None:
        return _empty_result(source_status, source_id, target_id, pair_id)
    if target_status is not None:
        return _empty_result(target_status, source_id, target_id, pair_id)
    assert source is not None and target is not None
    source_failure, source_volume, source_scale, source_uncertain = _validate_tetrahedron(
        source, tolerance_
    )
    if source_failure is not None:
        return _empty_result(
            source_failure,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            predicate_uncertain=source_uncertain,
        )
    target_failure, target_volume, target_scale, target_uncertain = _validate_tetrahedron(
        target, tolerance_
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
        )
    scale = max(source_scale, target_scale)
    source_planes = _tetra_planes(source)
    target_planes = _tetra_planes(target)
    planes = source_planes + target_planes
    candidates: list[np.ndarray] = []
    candidate_attempts = 0
    point_tolerance = tolerance_.absolute + tolerance_.relative * max(scale, 1.0e-300)
    for point in tuple(source) + tuple(target):
        if _inside(point, planes, tolerance_, scale):
            candidate_attempts += 1
            if not _append_unique(
                candidates, point, point_tolerance, limits_.max_candidates
            ):
                return _empty_result(
                    TetraIntersectionStatus.CANDIDATE_LIMIT,
                    source_id,
                    target_id,
                    pair_id,
                    source_volume=source_volume,
                    target_volume=target_volume,
                    candidate_count=candidate_attempts,
                )
    uncertain = False
    for indices in combinations(range(len(planes)), 3):
        point, point_uncertain = _triple_intersection(planes, indices, scale)
        uncertain = uncertain or point_uncertain
        if point is None:
            continue
        if _inside(point, planes, tolerance_, scale):
            candidate_attempts += 1
            if not _append_unique(
                candidates, point, point_tolerance, limits_.max_candidates
            ):
                return _empty_result(
                    TetraIntersectionStatus.CANDIDATE_LIMIT,
                    source_id,
                    target_id,
                    pair_id,
                    source_volume=source_volume,
                    target_volume=target_volume,
                    predicate_uncertain=uncertain,
                    candidate_count=candidate_attempts,
                )
    if uncertain:
        return _empty_result(
            TetraIntersectionStatus.UNCERTAIN_PREDICATE,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            target_volume=target_volume,
            predicate_uncertain=True,
            candidate_count=candidate_attempts,
        )
    if not candidates:
        return _empty_result(
            TetraIntersectionStatus.DISJOINT,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            target_volume=target_volume,
            candidate_count=candidate_attempts,
        )
    candidates.sort(key=lambda point: tuple(float(value) for value in point))
    if len(candidates) > limits_.max_vertices:
        return _empty_result(
            TetraIntersectionStatus.CANDIDATE_LIMIT,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            target_volume=target_volume,
            candidate_count=candidate_attempts,
        )
    vertices = np.asarray(candidates, dtype=np.float64)
    vertices.setflags(write=False)
    faces, face_failure = _build_faces(
        vertices, planes, tolerance_, scale, limits_.max_faces
    )
    if face_failure:
        return _empty_result(
            TetraIntersectionStatus.CANDIDATE_LIMIT,
            source_id,
            target_id,
            pair_id,
            source_volume=source_volume,
            target_volume=target_volume,
            predicate_uncertain=False,
            candidate_count=candidate_attempts,
        )
    volume, volume_error = _compensated_volume(vertices, faces)
    volume_tolerance = max(
        tolerance_.absolute, tolerance_.relative * max(scale**3, 1.0e-300)
    )
    status = (
        TetraIntersectionStatus.SUCCESS
        if volume > volume_tolerance
        else TetraIntersectionStatus.ZERO_MEASURE_CONTACT
    )
    output_vertices = np.empty((0, 3), dtype=np.float64) if volume_only else vertices
    output_faces: tuple[tuple[int, ...], ...] = () if volume_only else faces
    return TetraIntersectionResult(
        status=status,
        volume=volume,
        vertices=output_vertices,
        faces=output_faces,
        source_id=source_id,
        target_id=target_id,
        pair_id=pair_id,
        evidence=TetraIntersectionEvidence(
            candidate_count=candidate_attempts,
            vertex_count=len(vertices),
            face_count=len(faces),
            predicate_uncertain=False,
            volume_error=volume_error,
            source_volume=source_volume,
            target_volume=target_volume,
            volume_only=bool(volume_only),
        ),
    )


__all__ = [
    "TetraIntersectionEvidence",
    "TetraIntersectionLimits",
    "TetraIntersectionResult",
    "TetraIntersectionStatus",
    "TetraIntersectionTolerance",
    "intersect_tetrahedra",
    "stable_tetra_pair_id",
]
