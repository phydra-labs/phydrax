#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded source-root discovery for constrained closest-point multiplicity."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .._atlas import CurveTrimLoop, TrimDomain
from .._interval_enclosure import interval_add, interval_multiply, interval_subtract
from ._intersection import _krawczyk, RootEndpoint
from ._intersection_curve import IntersectionCurve, IntersectionPCurve
from ._model import BRepCurve, BRepModel
from ._patches import CircleCurve, PlanePatch, SpherePatch
from ._root_bindings import surface_chart_regular


if TYPE_CHECKING:
    from ._query import _FaceCertificate


def _dot(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], /
) -> tuple[np.ndarray, np.ndarray]:
    products = interval_multiply(first, second)
    lower = np.asarray(0.0, dtype=np.float64)
    upper = np.asarray(0.0, dtype=np.float64)
    for low, high in zip(products[0], products[1], strict=True):
        lower, upper = interval_add((lower, upper), (low, high))
    return lower, upper


def source_distance_bounds(box: np.ndarray, point: np.ndarray, /) -> tuple[float, float]:
    gap = interval_subtract((box[0], box[1]), (point, point))
    low = np.maximum(np.maximum(gap[0], -gap[1]), 0.0)
    high = np.maximum(np.abs(gap[0]), np.abs(gap[1]))
    first = _dot((low, low), (low, low))[0]
    second = _dot((high, high), (high, high))[1]
    return max(0.0, float(np.nextafter(np.sqrt(max(0.0, float(first))), -np.inf))), float(
        np.nextafter(np.sqrt(max(0.0, float(second))), np.inf)
    )


def _trim_position(domain: TrimDomain, box: np.ndarray, depth: int, /) -> int:
    """Whole-box interior/exterior, or undecided when a source boundary meets it."""
    for loop in domain.loops:
        if not isinstance(loop, CurveTrimLoop):
            raise ValueError(
                "Stationary discovery requires authoritative native trim curves."
            )
        lower = np.concatenate((loop.arc_lower, loop.junction_lower))
        upper = np.concatenate((loop.arc_upper, loop.junction_upper))
        if np.any(np.all((upper >= box[0]) & (lower <= box[1]), axis=1)):
            return 0
    result = domain.classify((0.5 * (box[0] + box[1]))[None], maximum_depth=depth)
    if not result.resolved[0]:
        return 0
    return 1 if result.inside[0] else -1


@dataclass(frozen=True, slots=True)
class StationaryRoot:
    face: int
    dimension: int
    entity: int
    parameter_box: np.ndarray
    point_box: np.ndarray
    distance_lower: float
    distance_upper: float
    continuum: bool = False


@dataclass(frozen=True, slots=True)
class ClosestStationaryIsolation:
    """Discovered roots plus the distance floor of everything left undecided.

    ``unresolved_distance_lower`` is a certified lower bound on the distance
    over every unresolved box, span or stratum (``inf`` when none remain).
    Boxes certified outside their face trim never contribute to it.
    """

    roots: tuple[StationaryRoot, ...]
    complete: bool
    boxes_processed: int
    unresolved_boxes: int
    unresolved_distance_lower: float


@dataclass(frozen=True, slots=True)
class _FaceIntervals:
    certificate: _FaceCertificate
    point: np.ndarray
    jacobian: bool

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.broadcast_to(self.point, (lower.shape[0], 3))
        program = self.certificate.hessian if self.jacobian else self.certificate.gradient
        return program.evaluate(
            np.concatenate((lower, points), axis=1),
            np.concatenate((upper, points), axis=1),
        )


@dataclass(frozen=True, slots=True)
class _FaceEquation:
    certificate: _FaceCertificate
    point: np.ndarray

    @property
    def value(self) -> _FaceIntervals:
        return _FaceIntervals(self.certificate, self.point, False)

    @property
    def jacobian(self) -> _FaceIntervals:
        return _FaceIntervals(self.certificate, self.point, True)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        lower, upper = self.jacobian.evaluate(points, points)
        return 0.5 * (lower + upper)


@dataclass(frozen=True, slots=True)
class _EdgeIntervals:
    curve: BRepCurve
    point: np.ndarray
    jacobian: bool
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None]

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for first, last in zip(lower[:, 0], upper[:, 0], strict=True):
            if isinstance(self.curve, IntersectionCurve):
                box = self.curve.bounding_box(
                    float(first), float(last), endpoint_roots=self.endpoint_roots
                )
                tangent = self.curve.derivative_bounds(
                    float(first),
                    float(last),
                    order=1,
                    endpoint_roots=self.endpoint_roots,
                )
            else:
                box = self.curve.bounding_box(float(first), float(last))
                tangent = self.curve.derivative_bounds(float(first), float(last), order=1)
            gap = interval_subtract((box[0], box[1]), (self.point, self.point))
            if self.jacobian:
                acceleration = (
                    self.curve.derivative_bounds(
                        float(first),
                        float(last),
                        order=2,
                        endpoint_roots=self.endpoint_roots,
                    )
                    if isinstance(self.curve, IntersectionCurve)
                    else self.curve.derivative_bounds(float(first), float(last), order=2)
                )
                result = interval_add(_dot(tangent, tangent), _dot(gap, acceleration))
                lows.append([[result[0]]])
                highs.append([[result[1]]])
            else:
                result = _dot(gap, tangent)
                lows.append([result[0]])
                highs.append([result[1]])
        return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _EdgeEquation:
    curve: BRepCurve
    point: np.ndarray
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None]

    @property
    def value(self) -> _EdgeIntervals:
        return _EdgeIntervals(self.curve, self.point, False, self.endpoint_roots)

    @property
    def jacobian(self) -> _EdgeIntervals:
        return _EdgeIntervals(self.curve, self.point, True, self.endpoint_roots)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        lower, upper = self.jacobian.evaluate(points, points)
        return 0.5 * (lower + upper)


def _circle_continuum(curve: CircleCurve, point: np.ndarray, /) -> bool:
    first = tuple(Fraction(float(value)) for value in np.asarray(curve.first_axis))
    second = tuple(Fraction(float(value)) for value in np.asarray(curve.second_axis))
    delta = tuple(
        Fraction(float(value)) - Fraction(float(center))
        for value, center in zip(point, np.asarray(curve.center), strict=True)
    )
    aa = sum((value * value for value in first), Fraction(0))
    bb = sum((value * value for value in second), Fraction(0))
    ab = sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
    da = sum((a * b for a, b in zip(delta, first, strict=True)), Fraction(0))
    db = sum((a * b for a, b in zip(delta, second, strict=True)), Fraction(0))
    return aa > 0 and aa == bb and ab == 0 and da == 0 and db == 0


def _sphere_squared_radius(patch: SpherePatch, /) -> Fraction | None:
    axes = tuple(
        tuple(Fraction(float(value)) for value in np.asarray(axis))
        for axis in (patch.first_axis, patch.second_axis, patch.axis)
    )
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
            for second in axes
        )
        for first in axes
    )
    if (
        gram[0][0] <= 0
        or any(gram[index][index] != gram[0][0] for index in range(3))
        or any(gram[i][j] != 0 for i in range(3) for j in range(3) if i != j)
    ):
        return None
    return Fraction(float(patch.radius)) ** 2 * gram[0][0]


def _branch_continuum(curve: IntersectionCurve, point: np.ndarray, /) -> bool:
    """Exact constant squared distance implied by the actual generating equations."""
    first, second = curve.first.patch, curve.second.patch
    if not isinstance(first, SpherePatch):
        first, second = second, first
    if not isinstance(first, SpherePatch):
        return False
    radius = _sphere_squared_radius(first)
    if radius is None:
        return False
    center = tuple(Fraction(float(value)) for value in np.asarray(first.center))
    delta = tuple(
        Fraction(float(value)) - origin
        for value, origin in zip(point, center, strict=True)
    )
    if isinstance(second, SpherePatch):
        if _sphere_squared_radius(second) is None:
            return False
        normal = tuple(
            Fraction(float(value)) - origin
            for value, origin in zip(np.asarray(second.center), center, strict=True)
        )
    elif isinstance(second, PlanePatch):
        a = tuple(Fraction(float(value)) for value in np.asarray(second.first_axis))
        b = tuple(Fraction(float(value)) for value in np.asarray(second.second_axis))
        normal = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    else:
        return False
    nonzero = next((index for index, value in enumerate(normal) if value != 0), None)
    if nonzero is None:
        return False
    multiplier = delta[nonzero] / normal[nonzero]
    return all(
        value == multiplier * direction
        for value, direction in zip(delta, normal, strict=True)
    )


def _tighten_root(
    equation: _FaceEquation | _EdgeEquation,
    box: np.ndarray,
    tolerance: float,
    maximum: int,
    /,
) -> tuple[np.ndarray, int]:
    used = 0
    while used < maximum and np.max(box[1] - box[0]) > tolerance:
        result = _krawczyk(equation, box[0][None], box[1][None])
        used += 1
        following = np.stack(
            (np.maximum(box[0], result.lower[0]), np.minimum(box[1], result.upper[0]))
        )
        if np.any(following[0] > following[1]):
            raise RuntimeError(
                "A certified stationary root lost its enclosing source box."
            )
        if np.all(following == box):
            break
        box = following
    return box, used


def isolate_closest_stationary(
    model: BRepModel,
    point: np.ndarray,
    /,
    *,
    certificates: tuple[_FaceCertificate, ...],
    trim_domains: tuple[TrimDomain, ...],
    distance_upper: float,
    tolerance: float,
    parameter_tolerance: float,
    max_boxes: int,
    max_depth: int,
    faces: tuple[int, ...] | None = None,
) -> ClosestStationaryIsolation:
    """Discover every possibly minimizing source root, or expose incomplete work.

    Interior faces, all authored trim edges and explicit source vertices are
    searched separately. Spatial proximity never merges root identities.
    Degenerate stationary sets and boxes meeting an undecided trim or knot wall
    remain unresolved, rather than borrowing numerical seed multiplicity.
    """
    if not isinstance(model, BRepModel) or model.geometry is None:
        raise ValueError("Stationary discovery requires an exact native BRepModel.")
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("A source closest-point query requires one finite 3-vector.")
    if (
        not np.isfinite(distance_upper)
        or distance_upper < 0.0
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError(
            "A source isolation requires finite distance and positive tolerance bounds."
        )
    if not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0.0:
        raise ValueError("A source isolation requires a positive parameter tolerance.")
    if len(trim_domains) != len(model.patches):
        raise ValueError(
            "Stationary trim domains must align with the exact source faces."
        )
    if (
        type(max_boxes) is not int
        or type(max_depth) is not int
        or max_boxes <= 0
        or max_depth <= 0
    ):
        raise ValueError("Stationary isolation requires positive integer work limits.")
    selected = tuple(range(len(model.patches))) if faces is None else faces
    if not selected or any(
        type(face) is not int or not 0 <= face < len(model.patches) for face in selected
    ):
        raise ValueError("Stationary isolation faces must index the exact source model.")
    geometry = model.geometry
    roots: list[StationaryRoot] = []
    root_families: list[tuple[int, int, int] | None] = []
    root_domains: list[np.ndarray | None] = []
    processed = unresolved = 0
    floor = np.inf

    def undecided(count: int, lower: float) -> None:
        nonlocal unresolved, floor
        unresolved += count
        floor = min(floor, max(0.0, lower))

    def retain(
        face: int,
        dimension: int,
        entity: int,
        parameter_box: np.ndarray,
        point_box: np.ndarray,
        *,
        continuum: bool = False,
        equation: _FaceEquation | _EdgeEquation | None = None,
        family: tuple[int, int, int] | None = None,
        proof_domain: np.ndarray | None = None,
    ) -> None:
        nonlocal processed
        low, high = source_distance_bounds(point_box, point)
        if low > distance_upper + tolerance:
            return
        if equation is not None and family is not None:
            for index, previous in enumerate(roots):
                if root_families[index] != family:
                    continue
                if np.any(previous.parameter_box[1] < parameter_box[0]) or np.any(
                    parameter_box[1] < previous.parameter_box[0]
                ):
                    continue
                previous_domain = root_domains[index]
                contained = (
                    previous_domain is not None
                    and np.all(parameter_box[0] >= previous_domain[0])
                    and np.all(parameter_box[1] <= previous_domain[1])
                ) or (
                    proof_domain is not None
                    and np.all(previous.parameter_box[0] >= proof_domain[0])
                    and np.all(previous.parameter_box[1] <= proof_domain[1])
                )
                if not contained:
                    if processed >= max_boxes:
                        undecided(1, low)
                        break
                    union = np.stack(
                        (
                            np.minimum(previous.parameter_box[0], parameter_box[0]),
                            np.maximum(previous.parameter_box[1], parameter_box[1]),
                        )
                    )
                    proof = _krawczyk(equation, union[0][None], union[1][None])
                    processed += 1
                    if not proof.certified[0]:
                        undecided(1, low)
                        break
                parameters = np.stack(
                    (
                        np.maximum(previous.parameter_box[0], parameter_box[0]),
                        np.minimum(previous.parameter_box[1], parameter_box[1]),
                    )
                )
                points = np.stack(
                    (
                        np.maximum(previous.point_box[0], point_box[0]),
                        np.minimum(previous.point_box[1], point_box[1]),
                    )
                )
                if np.any(parameters[0] > parameters[1]) or np.any(points[0] > points[1]):
                    raise RuntimeError(
                        "Source-unique stationary roots have contradictory enclosures."
                    )
                low, high = source_distance_bounds(points, point)
                roots[index] = StationaryRoot(
                    face, dimension, entity, parameters, points, low, high, continuum
                )
                return
        roots.append(
            StationaryRoot(
                face, dimension, entity, parameter_box, point_box, low, high, continuum
            )
        )
        root_families.append(family)
        root_domains.append(proof_domain)

    for certificate in certificates:
        source_face = certificate.face
        if source_face not in selected:
            continue
        patch = model.patches[source_face]
        domain = trim_domains[source_face]
        initial = np.asarray((certificate.lower, certificate.upper), dtype=np.float64)
        if certificate.bound(initial, point) > distance_upper + tolerance:
            continue
        if not patch.is_c1_on(np.asarray(model.parameter_bounds)[source_face, :, :]):
            # A nonsmooth source span wall is itself a constrained stratum.
            # It is not an ordinary smooth stationary equation.
            undecided(1, certificate.bound(initial, point))
            continue
        equation = _FaceEquation(certificate, point)
        queue = [(initial, 0)]
        while queue:
            box, depth = queue.pop()
            if processed >= max_boxes:
                undecided(
                    len(queue) + 1,
                    min(
                        certificate.bound(item, point)
                        for item, _ in ((box, depth), *queue)
                    ),
                )
                break
            processed += 1
            if (
                certificate.bound(box, point) > distance_upper + tolerance
                or _trim_position(domain, box, max_depth) < 0
            ):
                continue
            low, high = equation.value.evaluate(box[0][None], box[1][None])
            if np.any((low[0] > 0.0) | (high[0] < 0.0)):
                continue
            result = _krawczyk(equation, box[0][None], box[1][None])
            if result.excluded[0]:
                continue
            if result.certified[0]:
                root_box = np.stack(
                    (
                        np.maximum(box[0], result.lower[0]),
                        np.minimum(box[1], result.upper[0]),
                    )
                )
                root_box, used = _tighten_root(
                    equation,
                    root_box,
                    parameter_tolerance,
                    min(max_depth, max_boxes - processed),
                )
                processed += used
                position = _trim_position(domain, root_box, max_depth)
                if position > 0:
                    retain(
                        source_face,
                        2,
                        source_face,
                        root_box,
                        patch.bounding_box(root_box),
                        equation=equation,
                        family=(2, source_face, certificate.piece_index),
                        proof_domain=box,
                    )
                    continue
                if position < 0:
                    continue
            if depth >= max_depth:
                undecided(1, certificate.bound(box, point))
                continue
            widths = (box[1] - box[0]) / (initial[1] - initial[0])
            axis = int(np.argmax(widths))
            middle = 0.5 * (box[0, axis] + box[1, axis])
            if not box[0, axis] < middle < box[1, axis]:
                undecided(1, certificate.bound(box, point))
                continue
            a, b = box.copy(), box.copy()
            overlap = (box[1, axis] - box[0, axis]) / 16.0
            a[1, axis] = min(box[1, axis], float(np.nextafter(middle + overlap, np.inf)))
            b[0, axis] = max(box[0, axis], float(np.nextafter(middle - overlap, -np.inf)))
            queue.extend(((b, depth + 1), (a, depth + 1)))

    owners = {
        edge: min(face for face in selected if edge in model.topology.face_edges[face])
        for edge in sorted(
            {edge for face in selected for edge in model.topology.face_edges[face]}
        )
    }
    vertices = sorted(
        {vertex for edge in owners for vertex in geometry.edge_vertices[edge]}
    )
    for vertex in vertices:
        root = geometry.vertex_roots[vertex]
        box = (
            np.stack(
                (
                    np.asarray(geometry.vertex_points)[vertex],
                    np.asarray(geometry.vertex_points)[vertex],
                )
            )
            if root is None
            else root.point_enclosure()
        )
        face = min(
            owners[edge] for edge in owners if vertex in geometry.edge_vertices[edge]
        )
        retain(face, 0, vertex, np.empty((2, 0), dtype=np.float64), box)
    for edge, face in owners.items():
        carrier_index = geometry.edge_curves[edge]
        if carrier_index < 0:
            continue
        curve = geometry.curves[carrier_index]
        if isinstance(curve, IntersectionCurve) and not curve.fully_certified:
            undecided(
                1,
                source_distance_bounds(
                    curve.bounding_box(0.0, float(curve.num_charts)), point
                )[0],
            )
            continue
        endpoints = geometry.edge_parameter_enclosure(edge)
        first, last = float(endpoints[0, 0]), float(endpoints[1, 1])

        def source_box(start: float, stop: float, /) -> np.ndarray:
            if isinstance(curve, IntersectionCurve):
                return curve.bounding_box(
                    start, stop, endpoint_roots=geometry.edge_endpoint_roots[edge]
                )
            return curve.bounding_box(start, stop)

        if (
            isinstance(curve, CircleCurve)
            and _circle_continuum(curve, point)
            or isinstance(curve, IntersectionCurve)
            and _branch_continuum(curve, point)
        ):
            lower, upper = float(endpoints[0, 1]), float(endpoints[1, 0])
            if lower >= upper:
                undecided(1, source_distance_bounds(source_box(first, last), point)[0])
                continue
            parameter = 0.5 * (lower + upper)
            retain(
                face,
                1,
                edge,
                np.asarray([[parameter], [parameter]], dtype=np.float64),
                source_box(parameter, parameter),
                continuum=True,
            )
            continue
        equation = _EdgeEquation(curve, point, geometry.edge_endpoint_roots[edge])
        if isinstance(curve, IntersectionCurve):
            # The native atlas parameter changes chart scale at each integer.
            # A single smooth stationarity equation cannot cross that change.
            breaks = sorted(
                {
                    first,
                    last,
                    *(
                        float(chart)
                        for chart in range(curve.num_charts + 1)
                        if first < chart < last
                    ),
                }
            )
            queue = [
                (np.asarray([[start], [stop]], dtype=np.float64), 0)
                for start, stop in zip(breaks[:-1], breaks[1:], strict=True)
            ]
        else:
            if not curve.is_c1_on(first, last):
                bound = source_distance_bounds(source_box(first, last), point)[0]
                if bound <= distance_upper + tolerance:
                    undecided(1, bound)
                continue
            queue = [(np.asarray([[first], [last]], dtype=np.float64), 0)]
        while queue:
            box, depth = queue.pop()
            if processed >= max_boxes:
                undecided(
                    len(queue) + 1,
                    min(
                        source_distance_bounds(
                            source_box(float(item[0, 0]), float(item[1, 0])), point
                        )[0]
                        for item, _ in ((box, depth), *queue)
                    ),
                )
                break
            processed += 1
            if (
                source_distance_bounds(
                    source_box(float(box[0, 0]), float(box[1, 0])), point
                )[0]
                > distance_upper + tolerance
            ):
                continue
            low, high = equation.value.evaluate(box[0][None], box[1][None])
            if low[0, 0] > 0.0 or high[0, 0] < 0.0:
                continue
            result = _krawczyk(equation, box[0][None], box[1][None])
            if result.excluded[0]:
                continue
            if result.certified[0]:
                root_box = np.stack(
                    (
                        np.maximum(box[0], result.lower[0]),
                        np.minimum(box[1], result.upper[0]),
                    )
                )
                root_box, used = _tighten_root(
                    equation,
                    root_box,
                    parameter_tolerance,
                    min(max_depth, max_boxes - processed),
                )
                processed += used
                if root_box[0, 0] > endpoints[0, 1] and root_box[1, 0] < endpoints[1, 0]:
                    phase = (
                        int(np.floor(root_box[0, 0]))
                        if isinstance(curve, IntersectionCurve)
                        else -1
                    )
                    retain(
                        face,
                        1,
                        edge,
                        root_box,
                        source_box(float(root_box[0, 0]), float(root_box[1, 0])),
                        equation=equation,
                        family=(1, edge, phase),
                        proof_domain=box,
                    )
                    continue
            if box[1, 0] - box[0, 0] <= parameter_tolerance and (
                box[0, 0] <= endpoints[0, 1] or box[1, 0] >= endpoints[1, 0]
            ):
                # Within query parameter resolution of an authored endpoint,
                # the stratum is that endpoint's vertex, retained separately.
                continue
            if depth >= max_depth:
                undecided(
                    1,
                    source_distance_bounds(
                        source_box(float(box[0, 0]), float(box[1, 0])), point
                    )[0],
                )
                continue
            middle = 0.5 * (box[0, 0] + box[1, 0])
            if not box[0, 0] < middle < box[1, 0]:
                undecided(
                    1,
                    source_distance_bounds(
                        source_box(float(box[0, 0]), float(box[1, 0])), point
                    )[0],
                )
                continue
            overlap = (box[1, 0] - box[0, 0]) / 16.0
            lower = max(box[0, 0], float(np.nextafter(middle - overlap, -np.inf)))
            upper = min(box[1, 0], float(np.nextafter(middle + overlap, np.inf)))
            queue.extend(
                (
                    (np.asarray([[lower], [box[1, 0]]], dtype=np.float64), depth + 1),
                    (np.asarray([[box[0, 0]], [upper]], dtype=np.float64), depth + 1),
                )
            )
    return ClosestStationaryIsolation(
        tuple(roots), unresolved == 0, processed, unresolved, floor
    )


def source_closure_upper(
    model: BRepModel, point: np.ndarray, faces: tuple[int, ...], /
) -> float:
    """A feasible distance upper bound from authored, strictly interior edge phases."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("A closure distance requires exact native geometry.")
    upper = np.inf
    for face in faces:
        for loop in geometry.face_loops[face]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                endpoints = geometry.edge_parameter_enclosure(edge)
                first, last = float(endpoints[0, 1]), float(endpoints[1, 0])
                if first >= last:
                    continue
                parameter = 0.5 * (first + last)
                pcurve = geometry.pcurves[coedge]
                # A coupled branch p-curve encloses single parameters directly;
                # its range validator admits only nondegenerate subranges.
                uv = (
                    pcurve.enclosure(parameter, parameter)
                    if isinstance(pcurve, IntersectionPCurve)
                    else pcurve.bounding_box(parameter, parameter)
                )
                box = model.patches[face].bounding_box(uv)
                upper = min(upper, source_distance_bounds(box, point)[1])
    return float(upper)


def stationary_root_representative(
    model: BRepModel,
    root: StationaryRoot,
    /,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float, bool]:
    """Realize a source root while retaining its UV and physical error enclosure."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("A stationary representative requires exact native geometry.")
    edge = -1
    parameter = np.nan
    correspondence = 0.0
    if root.dimension == 2:
        uv_box = root.parameter_box
        uv = 0.5 * (uv_box[0] + uv_box[1])
        point = np.asarray(
            model.patches[root.face].evaluate(jnp.asarray(uv, dtype=jnp.float64))
        )
    else:
        coedges = tuple(
            coedge for loop in geometry.face_loops[root.face] for coedge in loop
        )
        if root.dimension == 1:
            edge = root.entity
            coedge = next(
                coedge for coedge in coedges if geometry.coedge_edges[coedge] == edge
            )
            parameter = float(0.5 * (root.parameter_box[0, 0] + root.parameter_box[1, 0]))
            source = geometry.curves[geometry.edge_curves[edge]]
            if isinstance(source, IntersectionCurve):
                realization = source.evaluate(jnp.asarray(parameter, dtype=jnp.float64))
                point = np.asarray(realization.point)
            else:
                point = np.asarray(
                    source.evaluate(jnp.asarray(parameter, dtype=jnp.float64))
                )
            parameter_lower, parameter_upper = (
                float(root.parameter_box[0, 0]),
                float(root.parameter_box[1, 0]),
            )
        elif root.dimension == 0:
            coedge = next(
                coedge
                for coedge in coedges
                if root.entity in geometry.edge_vertices[geometry.coedge_edges[coedge]]
            )
            edge = geometry.coedge_edges[coedge]
            endpoint = 0 if geometry.edge_vertices[edge][0] == root.entity else 1
            binding = geometry.coedge_endpoint_roots[coedge][endpoint]
            if binding is None:
                parameter_lower = parameter_upper = float(
                    geometry.edge_ranges[edge, endpoint]
                )
            else:
                parameter_lower, parameter_upper = binding.parameter_enclosure()
            parameter = 0.5 * (parameter_lower + parameter_upper)
            binding_root = geometry.vertex_roots[root.entity]
            if binding_root is None:
                point = np.asarray(geometry.vertex_points[root.entity])
            else:
                point, _, certified = binding_root.evaluate()
                if not certified:
                    raise ValueError(
                        "A source vertex realization lost its root certificate."
                    )
        else:
            raise ValueError(
                "A stationary root dimension must identify a source vertex, edge or face."
            )
        pcurve = geometry.pcurves[coedge]
        uv_box = (
            pcurve.bounding_box(
                parameter_lower,
                parameter_upper,
                endpoint_roots=geometry.coedge_endpoint_roots[coedge],
            )
            if isinstance(pcurve, IntersectionPCurve)
            else pcurve.bounding_box(parameter_lower, parameter_upper)
        )
        uv = np.asarray(pcurve.evaluate(jnp.asarray(parameter, dtype=jnp.float64)))
        correspondence = float(model.coedge_deviation_bounds[coedge])
    uv_error = float(np.max(np.maximum(np.abs(uv_box[0] - uv), np.abs(uv_box[1] - uv))))
    point_error = float(
        np.nextafter(
            np.linalg.norm(
                np.maximum(
                    np.abs(root.point_box[0] - point), np.abs(root.point_box[1] - point)
                )
            )
            + correspondence,
            np.inf,
        )
    )
    return (
        point,
        uv,
        uv_error,
        point_error,
        edge,
        parameter,
        surface_chart_regular(model.patches[root.face], uv_box),
    )


def stationary_root_fidelity(model: BRepModel, root: StationaryRoot, /) -> float:
    """Bound only the curve/vertex support discrepancy of this actual stratum."""
    if root.dimension == 2:
        return 0.0
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Stationary source fidelity requires exact native geometry.")
    error = 0.0
    for loop in geometry.face_loops[root.face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            if root.dimension == 1:
                if edge == root.entity:
                    error = max(error, float(model.coedge_deviation_bounds[coedge]))
                continue
            if root.entity not in geometry.edge_vertices[edge]:
                continue
            if geometry.vertex_roots[root.entity] is not None:
                error = max(error, float(model.coedge_deviation_bounds[coedge]))
                continue
            endpoint = 0 if geometry.edge_vertices[edge][0] == root.entity else 1
            parameter = float(geometry.edge_ranges[edge, endpoint])
            pcurve = geometry.pcurves[coedge]
            uv = (
                pcurve.bounding_box(
                    parameter,
                    parameter,
                    endpoint_roots=geometry.coedge_endpoint_roots[coedge],
                )
                if isinstance(pcurve, IntersectionPCurve)
                else pcurve.bounding_box(parameter, parameter)
            )
            point = np.asarray(geometry.vertex_points[root.entity])
            support = model.patches[root.face].bounding_box(uv)
            discrepancy = interval_subtract((support[0], support[1]), (point, point))
            error = max(
                error,
                source_distance_bounds(
                    np.stack(discrepancy), np.zeros((3,), dtype=np.float64)
                )[1],
            )
    return error
