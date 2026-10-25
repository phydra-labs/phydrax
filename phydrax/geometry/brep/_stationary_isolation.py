#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded source-root discovery for constrained closest-point multiplicity."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import final

import jax.numpy as jnp
import numpy as np

from ..._strict import StrictModule
from ..._validation import positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._atlas import CurveTrimLoop, TrimDomain
from .._interval_enclosure import interval_add, interval_dot, interval_subtract
from .._rational import rational_cross, rational_dot, rational_vector
from ._coordinate_request import coordinate_source_request
from ._intersection import RootEndpoint
from ._intersection_curve import IntersectionCurve, IntersectionPCurve
from ._intersection_kernel import krawczyk
from ._interval_source import evaluate_interval
from ._model import BRepCurve, BRepGeometry, BRepModel, BRepPCurve
from ._patches import CircleCurve, PlanePatch, SpherePatch
from ._query_types import FaceCertificate
from ._request_preparations import TrimRequestPreparations
from ._root_bindings import surface_chart_regular


def source_distance_bounds(box: np.ndarray, point: np.ndarray, /) -> tuple[float, float]:
    gap = interval_subtract((box[0], box[1]), (point, point))
    gap_lower = np.asarray(gap[0], dtype=np.float64)
    gap_upper = np.asarray(gap[1], dtype=np.float64)
    low = np.maximum(np.maximum(gap_lower, -gap_upper), 0.0)
    high = np.maximum(np.abs(gap_lower), np.abs(gap_upper))
    first = interval_dot((low, low), (low, low))[0]
    second = interval_dot((high, high), (high, high))[1]
    return max(0.0, float(np.nextafter(np.sqrt(max(0.0, float(first))), -np.inf))), float(
        np.nextafter(np.sqrt(max(0.0, float(second))), np.inf)
    )


def _trim_position(
    domain: TrimDomain,
    box: np.ndarray,
    depth: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> int:
    """Classify a whole box, leaving boundary intersections undecided."""
    for loop in domain.loops:
        match loop:
            case CurveTrimLoop():
                pass
            case _:
                raise ValueError(
                    f"isolation.trim_domain.loops must contain native curve loops; got {type(loop).__name__}"
                )
    result = domain.classify_boxes(
        box[None],
        maximum_depth=depth,
        preparations=preparations,
        budget=budget,
    )
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
    certificate: FaceCertificate
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
    certificate: FaceCertificate
    point: np.ndarray

    @property
    def value(self) -> _FaceIntervals:
        return _FaceIntervals(self.certificate, self.point, False)

    @property
    def jacobian(self) -> _FaceIntervals:
        return _FaceIntervals(self.certificate, self.point, True)

    def point_jacobians(
        self,
        points: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        lower, upper = evaluate_interval(
            self.jacobian, points, points, budget=budget, preparations=preparations
        )
        return 0.5 * (lower + upper)


@dataclass(frozen=True, slots=True)
class _EdgeIntervals:
    curve: BRepCurve
    point: np.ndarray
    jacobian: bool
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None]

    def evaluate_scope(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for first, last in zip(lower[:, 0], upper[:, 0], strict=True):
            box = _edge_box(
                self.curve,
                float(first),
                float(last),
                self.endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
            tangent = _edge_derivative_bounds(
                self.curve,
                float(first),
                float(last),
                self.endpoint_roots,
                order=1,
                budget=budget,
                preparations=preparations,
            )
            gap = interval_subtract((box[0], box[1]), (self.point, self.point))
            if self.jacobian:
                acceleration = _edge_derivative_bounds(
                    self.curve,
                    float(first),
                    float(last),
                    self.endpoint_roots,
                    order=2,
                    budget=budget,
                    preparations=preparations,
                )
                result = interval_add(
                    interval_dot(tangent, tangent), interval_dot(gap, acceleration)
                )
                lows.append([[result[0]]])
                highs.append([[result[1]]])
            else:
                result = interval_dot(gap, tangent)
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

    def point_jacobians(
        self,
        points: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        lower, upper = evaluate_interval(
            self.jacobian, points, points, budget=budget, preparations=preparations
        )
        return 0.5 * (lower + upper)


def _circle_continuum(curve: CircleCurve, point: np.ndarray, /) -> bool:
    first = rational_vector(np.asarray(curve.first_axis))
    second = rational_vector(np.asarray(curve.second_axis))
    delta = tuple(
        value - center
        for value, center in zip(
            rational_vector(point), rational_vector(np.asarray(curve.center)), strict=True
        )
    )
    aa = rational_dot(first, first)
    bb = rational_dot(second, second)
    ab = rational_dot(first, second)
    da = rational_dot(delta, first)
    db = rational_dot(delta, second)
    return aa > 0 and aa == bb and ab == 0 and da == 0 and db == 0


def _sphere_squared_radius(patch: SpherePatch, /) -> Fraction | None:
    axes = tuple(
        rational_vector(np.asarray(axis))
        for axis in (patch.first_axis, patch.second_axis, patch.axis)
    )
    gram = tuple(tuple(rational_dot(first, second) for second in axes) for first in axes)
    if (
        gram[0][0] <= 0
        or any(gram[index][index] != gram[0][0] for index in range(3))
        or any(gram[i][j] != 0 for i in range(3) for j in range(3) if i != j)
    ):
        return None
    return Fraction(float(patch.radius)) ** 2 * gram[0][0]


def _branch_continuum(curve: IntersectionCurve, point: np.ndarray, /) -> bool:
    """Test constant squared distance from the branch's generating equations."""
    match (curve.first.patch, curve.second.patch):
        case (SpherePatch() as first, second) | (second, SpherePatch() as first):
            radius = _sphere_squared_radius(first)
        case _:
            return False
    if radius is None:
        return False
    center = rational_vector(np.asarray(first.center))
    delta = tuple(
        value - origin
        for value, origin in zip(rational_vector(point), center, strict=True)
    )
    match second:
        case SpherePatch():
            if _sphere_squared_radius(second) is None:
                return False
            normal = tuple(
                value - origin
                for value, origin in zip(
                    rational_vector(np.asarray(second.center)), center, strict=True
                )
            )
        case PlanePatch():
            normal = rational_cross(
                rational_vector(np.asarray(second.first_axis)),
                rational_vector(np.asarray(second.second_axis)),
            )
        case _:
            return False
    nonzero = next((index for index, value in enumerate(normal) if value != 0), None)
    if nonzero is None:
        return False
    multiplier = delta[nonzero] / normal[nonzero]
    return all(
        value == multiplier * direction
        for value, direction in zip(delta, normal, strict=True)
    )


class StationaryEnclosureError(RuntimeError):
    """A certified contraction or source-unique root has contradictory enclosures."""


def _tighten_root(
    equation: _FaceEquation | _EdgeEquation,
    box: np.ndarray,
    tolerance: float,
    maximum: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
) -> tuple[np.ndarray, int]:
    used = 0
    while used < maximum and np.max(box[1] - box[0]) > tolerance:
        result = krawczyk(
            equation, box[0][None], box[1][None], budget=budget, preparations=preparations
        )
        used += 1
        following = np.stack(
            (np.maximum(box[0], result.lower[0]), np.minimum(box[1], result.upper[0]))
        )
        if np.any(following[0] > following[1]):
            raise StationaryEnclosureError(
                "stationary_root.parameter_box must enclose the certified root; got an empty contraction"
            )
        if np.all(following == box):
            break
        box = following
    return box, used


@final
class _IsolationContext(StrictModule):
    model: BRepModel
    geometry: BRepGeometry
    point: np.ndarray
    trim_domains: tuple[TrimDomain, ...]
    distance_upper: float
    tolerance: float
    parameter_tolerance: float
    max_boxes: int
    max_depth: int
    preparations: TrimRequestPreparations
    budget: CoordinateEnclosureBudget


@final
class _RetainedRoot(StrictModule):
    root: StationaryRoot
    family: tuple[int, int, int] | None
    proof_domain: np.ndarray | None


@dataclass
class _IsolationWork:
    context: _IsolationContext
    retained: list[_RetainedRoot] = field(default_factory=list)
    processed: int = 0
    unresolved: int = 0
    floor: float = np.inf

    def undecided(self, count: int, lower: float) -> None:
        self.unresolved += count
        self.floor = min(self.floor, max(0.0, lower))

    def retain(
        self,
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
        context = self.context
        low, high = source_distance_bounds(point_box, context.point)
        if low > context.distance_upper + context.tolerance:
            return
        if equation is not None and family is not None:
            for index, retained in enumerate(self.retained):
                previous = retained.root
                if retained.family != family:
                    continue
                if np.any(previous.parameter_box[1] < parameter_box[0]) or np.any(
                    parameter_box[1] < previous.parameter_box[0]
                ):
                    continue
                previous_domain = retained.proof_domain
                if not _certify_root_union(
                    retained, parameter_box, proof_domain, equation, low, self
                ):
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
                    raise StationaryEnclosureError(
                        "stationary_root.enclosures must agree for a source-unique root; got disjoint enclosures"
                    )
                low, high = source_distance_bounds(points, context.point)
                root = StationaryRoot(
                    face, dimension, entity, parameters, points, low, high, continuum
                )
                self.retained[index] = _RetainedRoot(root, family, previous_domain)
                return
        root = StationaryRoot(
            face, dimension, entity, parameter_box, point_box, low, high, continuum
        )
        self.retained.append(_RetainedRoot(root, family, proof_domain))


def _certify_root_union(
    retained: _RetainedRoot,
    parameter_box: np.ndarray,
    proof_domain: np.ndarray | None,
    equation: _FaceEquation | _EdgeEquation,
    lower_distance: float,
    work: _IsolationWork,
    /,
) -> bool:
    previous = retained.root
    previous_domain = retained.proof_domain
    contained = (
        previous_domain is not None
        and np.all(parameter_box[0] >= previous_domain[0])
        and np.all(parameter_box[1] <= previous_domain[1])
    ) or (
        proof_domain is not None
        and np.all(previous.parameter_box[0] >= proof_domain[0])
        and np.all(previous.parameter_box[1] <= proof_domain[1])
    )
    if contained:
        return True
    context = work.context
    if work.processed >= context.max_boxes:
        work.undecided(1, lower_distance)
        return False
    union = np.stack(
        (
            np.minimum(previous.parameter_box[0], parameter_box[0]),
            np.maximum(previous.parameter_box[1], parameter_box[1]),
        )
    )
    proof = krawczyk(
        equation,
        union[0][None],
        union[1][None],
        budget=context.budget,
        preparations=context.preparations,
    )
    work.processed += 1
    if not proof.certified[0]:
        work.undecided(1, lower_distance)
        return False
    return True


@final
class _FaceStratum(StrictModule):
    certificate: FaceCertificate
    equation: _FaceEquation
    initial: np.ndarray


@final
class _EdgeStratum(StrictModule):
    face: int
    edge: int
    equation: _EdgeEquation
    endpoints: np.ndarray
    initial: np.ndarray


def _edge_box(
    curve: BRepCurve,
    first: float,
    last: float,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None],
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    match curve:
        case IntersectionCurve():
            return curve.bounding_box(
                first,
                last,
                endpoint_roots=endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
        case _:
            return curve.bounding_box(
                first, last, preparations=preparations, budget=budget
            )


def _stratum_box(
    stratum: _FaceStratum | _EdgeStratum,
    box: np.ndarray,
    context: _IsolationContext,
    /,
) -> np.ndarray:
    match stratum:
        case _FaceStratum():
            return context.model.patches[stratum.certificate.face].bounding_box(
                box, preparations=context.preparations, budget=context.budget
            )
        case _EdgeStratum():
            return _edge_box(
                stratum.equation.curve,
                float(box[0, 0]),
                float(box[1, 0]),
                stratum.equation.endpoint_roots,
                preparations=context.preparations,
                budget=context.budget,
            )


def _stratum_bound(
    stratum: _FaceStratum | _EdgeStratum,
    box: np.ndarray,
    context: _IsolationContext,
    /,
) -> float:
    match stratum:
        case _FaceStratum():
            return stratum.certificate.bound(
                box,
                context.point,
                budget=context.budget,
                preparations=context.preparations,
            )
        case _EdgeStratum():
            return source_distance_bounds(
                _stratum_box(stratum, box, context), context.point
            )[0]


def _stratum_position(
    stratum: _FaceStratum | _EdgeStratum,
    box: np.ndarray,
    context: _IsolationContext,
    /,
) -> int:
    match stratum:
        case _FaceStratum():
            return _trim_position(
                context.trim_domains[stratum.certificate.face],
                box,
                context.max_depth,
                preparations=context.preparations,
                budget=context.budget,
            )
        case _EdgeStratum():
            return int(
                box[0, 0] > stratum.endpoints[0, 1]
                and box[1, 0] < stratum.endpoints[1, 0]
            )


def _retain_stratum_root(
    stratum: _FaceStratum | _EdgeStratum,
    root_box: np.ndarray,
    proof_domain: np.ndarray,
    work: _IsolationWork,
    /,
) -> None:
    match stratum:
        case _FaceStratum():
            face = stratum.certificate.face
            dimension, entity = 2, face
            family = (2, face, stratum.certificate.piece_index)
        case _EdgeStratum():
            face, dimension, entity = stratum.face, 1, stratum.edge
            phase = (
                int(np.floor(root_box[0, 0]))
                if isinstance(stratum.equation.curve, IntersectionCurve)
                else -1
            )
            family = (1, entity, phase)
    work.retain(
        face,
        dimension,
        entity,
        root_box,
        _stratum_box(stratum, root_box, work.context),
        equation=stratum.equation,
        family=family,
        proof_domain=proof_domain,
    )


def _edge_derivative_bounds(
    curve: BRepCurve,
    first: float,
    last: float,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None],
    /,
    *,
    order: int,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
) -> tuple[np.ndarray, np.ndarray]:
    match curve:
        case IntersectionCurve():
            return curve.derivative_bounds(
                first,
                last,
                order=order,
                endpoint_roots=endpoint_roots,
                budget=budget,
                preparations=preparations,
            )
        case _:
            return curve.derivative_bounds(
                first, last, order=order, budget=budget, preparations=preparations
            )


class _BoxSubdivision(StrictModule):
    first: np.ndarray
    last: np.ndarray


def _split_stationary_box(
    box: np.ndarray, initial: np.ndarray, /
) -> _BoxSubdivision | None:
    axis = (
        0
        if box.shape[1] == 1
        else int(np.argmax((box[1] - box[0]) / (initial[1] - initial[0])))
    )
    middle = 0.5 * (box[0, axis] + box[1, axis])
    if not box[0, axis] < middle < box[1, axis]:
        return None
    first, last = box.copy(), box.copy()
    overlap = (box[1, axis] - box[0, axis]) / 16.0
    first[1, axis] = min(box[1, axis], float(np.nextafter(middle + overlap, np.inf)))
    last[0, axis] = max(box[0, axis], float(np.nextafter(middle - overlap, -np.inf)))
    return _BoxSubdivision(first, last)


def _search_stratum(
    stratum: _FaceStratum | _EdgeStratum,
    queue: list[tuple[np.ndarray, int]],
    work: _IsolationWork,
    /,
) -> None:
    """Subdivide one smooth stationary equation without merging source identities."""
    context = work.context
    equation = stratum.equation
    while queue:
        box, depth = queue.pop()
        if work.processed >= context.max_boxes:
            work.undecided(
                len(queue) + 1,
                min(
                    _stratum_bound(stratum, item, context)
                    for item, _ in ((box, depth), *queue)
                ),
            )
            break
        work.processed += 1
        if (
            _stratum_bound(stratum, box, context)
            > context.distance_upper + context.tolerance
        ):
            continue
        # Equation exclusion and Krawczyk exclusion skip a box whatever its trim
        # position, and neither consumes the box allowance. Deciding them before
        # the exact trim classification therefore retains every decision while
        # refining the trim only where the box can still hold a stationary root.
        low, high = evaluate_interval(
            equation.value,
            box[0][None],
            box[1][None],
            budget=context.budget,
            preparations=context.preparations,
        )
        if np.any((low[0] > 0.0) | (high[0] < 0.0)):
            continue
        result = krawczyk(
            equation,
            box[0][None],
            box[1][None],
            budget=context.budget,
            preparations=context.preparations,
        )
        if result.excluded[0]:
            continue
        match stratum:
            case _FaceStratum() if _stratum_position(stratum, box, context) < 0:
                continue
            case _:
                pass
        if result.certified[0]:
            root_box = np.stack(
                (np.maximum(box[0], result.lower[0]), np.minimum(box[1], result.upper[0]))
            )
            root_box, used = _tighten_root(
                equation,
                root_box,
                context.parameter_tolerance,
                min(context.max_depth, context.max_boxes - work.processed),
                budget=context.budget,
                preparations=context.preparations,
            )
            work.processed += used
            position = _stratum_position(stratum, root_box, context)
            if position > 0:
                _retain_stratum_root(stratum, root_box, box, work)
                continue
            if position < 0:
                continue
        match stratum:
            case _EdgeStratum() if box[1, 0] - box[
                0, 0
            ] <= context.parameter_tolerance and (
                box[0, 0] <= stratum.endpoints[0, 1]
                or box[1, 0] >= stratum.endpoints[1, 0]
            ):
                # At query resolution the endpoint belongs to its separately retained vertex.
                continue
            case _:
                pass
        if depth >= context.max_depth:
            work.undecided(1, _stratum_bound(stratum, box, context))
            continue
        split = _split_stationary_box(box, stratum.initial)
        if split is None:
            work.undecided(1, _stratum_bound(stratum, box, context))
            continue
        queue.extend(((split.last, depth + 1), (split.first, depth + 1)))


def _search_faces(
    certificates: tuple[FaceCertificate, ...],
    selected: tuple[int, ...],
    work: _IsolationWork,
    /,
) -> None:
    context = work.context
    for certificate in certificates:
        face = certificate.face
        if face not in selected:
            continue
        initial = np.asarray((certificate.lower, certificate.upper), dtype=np.float64)
        if (
            certificate.bound(
                initial,
                context.point,
                budget=context.budget,
                preparations=context.preparations,
            )
            > context.distance_upper + context.tolerance
        ):
            continue
        patch = context.model.patches[face]
        if not patch.is_c1_on(
            np.asarray(context.model.parameter_bounds)[face, :, :],
            preparations=context.preparations,
            budget=context.budget,
        ):
            # Nonsmooth span walls are constrained strata, not smooth stationary equations.
            work.undecided(
                1,
                certificate.bound(
                    initial,
                    context.point,
                    budget=context.budget,
                    preparations=context.preparations,
                ),
            )
            continue
        stratum = _FaceStratum(
            certificate, _FaceEquation(certificate, context.point), initial
        )
        _search_stratum(stratum, [(initial, 0)], work)


def _retain_vertices(owners: dict[int, int], work: _IsolationWork, /) -> None:
    context = work.context
    geometry = context.geometry
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
            else root.point_enclosure(
                preparations=context.preparations, budget=context.budget
            )
        )
        face = min(
            owners[edge] for edge in owners if vertex in geometry.edge_vertices[edge]
        )
        work.retain(face, 0, vertex, np.empty((2, 0), dtype=np.float64), box)


def _prepare_edge(edge: int, face: int, work: _IsolationWork, /) -> _EdgeStratum | None:
    context = work.context
    geometry = context.geometry
    carrier_index = geometry.edge_curves[edge]
    if carrier_index < 0:
        return None
    curve = geometry.curves[carrier_index]
    if isinstance(curve, IntersectionCurve) and not curve.fully_certified:
        work.undecided(
            1,
            source_distance_bounds(
                curve.bounding_box(
                    0.0,
                    float(curve.num_charts),
                    preparations=context.preparations,
                    budget=context.budget,
                ),
                context.point,
            )[0],
        )
        return None
    endpoints = geometry.edge_parameter_enclosure(
        edge, preparations=context.preparations, budget=context.budget
    )
    first, last = float(endpoints[0, 0]), float(endpoints[1, 1])
    initial = np.asarray([[first], [last]], dtype=np.float64)
    equation = _EdgeEquation(curve, context.point, geometry.edge_endpoint_roots[edge])
    stratum = _EdgeStratum(face, edge, equation, endpoints, initial)
    if (
        isinstance(curve, CircleCurve)
        and _circle_continuum(curve, context.point)
        or isinstance(curve, IntersectionCurve)
        and _branch_continuum(curve, context.point)
    ):
        lower, upper = float(endpoints[0, 1]), float(endpoints[1, 0])
        if lower >= upper:
            work.undecided(1, _stratum_bound(stratum, initial, context))
            return None
        parameter = 0.5 * (lower + upper)
        box = np.asarray([[parameter], [parameter]], dtype=np.float64)
        work.retain(
            face, 1, edge, box, _stratum_box(stratum, box, context), continuum=True
        )
        return None
    if not isinstance(curve, IntersectionCurve) and not curve.is_c1_on(
        first, last, preparations=context.preparations, budget=context.budget
    ):
        bound = _stratum_bound(stratum, initial, context)
        if bound <= context.distance_upper + context.tolerance:
            work.undecided(1, bound)
        return None
    return stratum


def _edge_queue(stratum: _EdgeStratum, /) -> list[tuple[np.ndarray, int]]:
    match stratum.equation.curve:
        case IntersectionCurve() as curve:
            first, last = float(stratum.initial[0, 0]), float(stratum.initial[1, 0])
            # Integer boundaries change chart scale and cannot share a smooth equation.
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
            return [
                (np.asarray([[start], [stop]], dtype=np.float64), 0)
                for start, stop in zip(breaks[:-1], breaks[1:], strict=True)
            ]
        case _:
            return [(stratum.initial, 0)]


@checked
def isolate_closest_stationary(
    model: BRepModel,
    point: np.ndarray,
    /,
    *,
    certificates: tuple[FaceCertificate, ...],
    trim_domains: tuple[TrimDomain, ...],
    distance_upper: float,
    tolerance: float,
    parameter_tolerance: float,
    max_boxes: int,
    max_depth: int,
    faces: tuple[int, ...] | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> ClosestStationaryIsolation:
    """Discover minimizing source roots and bound all unresolved strata.

    Face interiors, trim edges, and vertices retain separate identities.
    Degenerate sets and undecided trim or knot walls contribute to the
    unresolved distance floor.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request((model, point)) as budget:
            return isolate_closest_stationary(
                model,
                point,
                certificates=certificates,
                trim_domains=trim_domains,
                distance_upper=distance_upper,
                tolerance=tolerance,
                parameter_tolerance=parameter_tolerance,
                max_boxes=max_boxes,
                max_depth=max_depth,
                faces=faces,
                preparations=preparations,
                budget=budget,
            )
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "isolation.model.geometry must contain native geometry; got None"
        )
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(
            f"isolation.point must be a finite 3-vector; got shape {point.shape}"
        )
    if (
        not np.isfinite(distance_upper)
        or distance_upper < 0.0
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError(
            "isolation.bounds must be finite with nonnegative distance_upper and positive tolerance; "
            f"got {distance_upper!r}/{tolerance!r}"
        )
    if not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0.0:
        raise ValueError(
            f"isolation.parameter_tolerance must be finite and positive; got {parameter_tolerance!r}"
        )
    if len(trim_domains) != len(model.patches):
        raise ValueError(
            f"isolation.trim_domains must align with source faces; got {len(trim_domains)} domains for {len(model.patches)} faces"
        )
    # Builtin work limits exclude booleans and NumPy scalar metadata.
    if type(max_boxes) is not int or type(max_depth) is not int:
        raise ValueError(
            f"isolation.work_limits must be positive Python integers for max_boxes and max_depth; got {max_boxes!r}/{max_depth!r}"
        )
    max_boxes = positive_integer(max_boxes, "isolation.max_boxes")
    max_depth = positive_integer(max_depth, "isolation.max_depth")
    selected = tuple(range(len(model.patches))) if faces is None else faces
    if not selected or any(
        type(face) is not int or not 0 <= face < len(model.patches) for face in selected
    ):
        raise ValueError(f"isolation.faces must index source faces; got {selected!r}")
    context = _IsolationContext(
        model,
        geometry,
        point,
        trim_domains,
        distance_upper,
        tolerance,
        parameter_tolerance,
        max_boxes,
        max_depth,
        preparations,
        budget,
    )
    work = _IsolationWork(context)
    _search_faces(certificates, selected, work)
    owners = {
        edge: min(face for face in selected if edge in model.topology.face_edges[face])
        for edge in sorted(
            {edge for face in selected for edge in model.topology.face_edges[face]}
        )
    }
    _retain_vertices(owners, work)
    for edge, face in owners.items():
        stratum = _prepare_edge(edge, face, work)
        if stratum is not None:
            _search_stratum(stratum, _edge_queue(stratum), work)
    return ClosestStationaryIsolation(
        tuple(retained.root for retained in work.retained),
        work.unresolved == 0,
        work.processed,
        work.unresolved,
        work.floor,
    )


@checked
def source_closure_upper(
    model: BRepModel,
    point: np.ndarray,
    faces: tuple[int, ...],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> float:
    """Return a feasible distance upper bound from strictly interior edge phases."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                model,
                point,
                faces,
            )
        ) as budget:
            return source_closure_upper(
                model, point, faces, preparations=preparations, budget=budget
            )
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "source_closure.model.geometry must contain native geometry; got None"
        )
    upper = np.inf
    for face in faces:
        for loop in geometry.face_loops[face]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                endpoints = geometry.edge_parameter_enclosure(
                    edge, preparations=preparations, budget=budget
                )
                first, last = float(endpoints[0, 1]), float(endpoints[1, 0])
                if first >= last:
                    continue
                parameter = 0.5 * (first + last)
                pcurve = geometry.pcurves[coedge]
                # A coupled branch p-curve encloses single parameters directly;
                # its range validator admits only nondegenerate subranges.
                uv = (
                    pcurve.enclosure(
                        parameter, parameter, preparations=preparations, budget=budget
                    )
                    if isinstance(pcurve, IntersectionPCurve)
                    else pcurve.bounding_box(
                        parameter, parameter, preparations=preparations, budget=budget
                    )
                )
                box = model.patches[face].bounding_box(
                    uv, preparations=preparations, budget=budget
                )
                upper = min(upper, source_distance_bounds(box, point)[1])
    return float(upper)


@final
class StationaryRootRepresentative(StrictModule):
    """Realized point, chart parameters, and retained enclosure errors of one root."""

    point: np.ndarray
    uv: np.ndarray
    uv_error: float
    point_error: float
    edge: int
    parameter: float
    regular: bool


def _pcurve_box(
    pcurve: BRepPCurve,
    first: float,
    last: float,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None],
    /,
    *,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    match pcurve:
        case IntersectionPCurve():
            return pcurve.bounding_box(
                first,
                last,
                endpoint_roots=endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
        case _:
            return pcurve.bounding_box(
                first, last, preparations=preparations, budget=budget
            )


@checked
def stationary_root_representative(
    model: BRepModel,
    root: StationaryRoot,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> StationaryRootRepresentative:
    """Realize a source root while retaining its UV and physical error enclosure."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                model,
                root,
            )
        ) as budget:
            return stationary_root_representative(
                model, root, preparations=preparations, budget=budget
            )
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "stationary_representative.model.geometry must contain native geometry; got None"
        )
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
            match source:
                case IntersectionCurve():
                    realization = source.evaluate(
                        jnp.asarray(parameter, dtype=jnp.float64)
                    )
                    point = np.asarray(realization.point)
                case _:
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
                parameter_lower, parameter_upper = binding.parameter_enclosure(
                    preparations=preparations, budget=budget
                )
            parameter = 0.5 * (parameter_lower + parameter_upper)
            binding_root = geometry.vertex_roots[root.entity]
            if binding_root is None:
                point = np.asarray(geometry.vertex_points[root.entity])
            else:
                point, _, certified = binding_root.evaluate()
                if not certified:
                    raise ValueError(
                        "stationary_representative.vertex must retain its root certificate; got an uncertified realization"
                    )
        else:
            raise ValueError(
                f"stationary_root.dimension must identify a vertex, edge, or face; got {root.dimension!r}"
            )
        pcurve = geometry.pcurves[coedge]
        uv_box = _pcurve_box(
            pcurve,
            parameter_lower,
            parameter_upper,
            geometry.coedge_endpoint_roots[coedge],
            preparations=preparations,
            budget=budget,
        )
        match pcurve:
            case IntersectionPCurve():
                uv = np.asarray(
                    pcurve.evaluate(
                        jnp.asarray(parameter, dtype=jnp.float64),
                        preparations=preparations,
                        budget=budget,
                    )
                )
            case _:
                uv = np.asarray(
                    pcurve.evaluate(jnp.asarray(parameter, dtype=jnp.float64))
                )
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
    return StationaryRootRepresentative(
        point,
        uv,
        uv_error,
        point_error,
        edge,
        parameter,
        surface_chart_regular(
            model.patches[root.face], uv_box, preparations=preparations, budget=budget
        ),
    )


@checked
def stationary_root_fidelity(
    model: BRepModel,
    root: StationaryRoot,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> float:
    """Bound the curve or vertex support discrepancy of the root stratum."""
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                model,
                root,
            )
        ) as budget:
            return stationary_root_fidelity(
                model, root, preparations=preparations, budget=budget
            )
    if root.dimension == 2:
        return 0.0
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "stationary_fidelity.model.geometry must contain native geometry; got None"
        )
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
            uv = _pcurve_box(
                pcurve,
                parameter,
                parameter,
                geometry.coedge_endpoint_roots[coedge],
                preparations=preparations,
                budget=budget,
            )
            point = np.asarray(geometry.vertex_points[root.entity])
            support = model.patches[root.face].bounding_box(
                uv, preparations=preparations, budget=budget
            )
            discrepancy = interval_subtract((support[0], support[1]), (point, point))
            error = max(
                error,
                source_distance_bounds(
                    np.stack(discrepancy), np.zeros((3,), dtype=np.float64)
                )[1],
            )
    return error
