#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native trim coverage proofs and bounded host solid-membership strategies."""

from __future__ import annotations

from fractions import Fraction
from math import pi
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry.brep._containment as containment_owner

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ._affine_query import (
    AffineFace,
    certify_affine_parity,
    prepare_affine_faces,
    QueryBroadphase,
)
from ._exact import square_root_interval
from ._intersection import NativePeriodEndpoint
from ._intersection_curve import (
    affine_curve_coefficients,
    pcurve_periodic_source,
    PeriodicPCurve,
)
from ._model import BRepGeometry, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    EllipseCurve,
    LineCurve,
    SurfaceIsoparametricCurve,
)
from ._projection_contracts import BRepProjectionStatus
from ._query_types import BRepQueryBudget, FaceData
from ._request_preparations import TrimRequestPreparations
from ._revolution_membership import (
    classify_full_revolution,
    prepare_full_revolution,
    RevolutionSource,
)
from ._sphere_membership import classify_full_sphere, prepare_full_sphere, SphereSource
from ._trim_helpers import period_offset


if TYPE_CHECKING:
    from ._query import PreparedBRepQuery

_UNIQUE = int(BRepProjectionStatus.UNIQUE)
_AMBIGUOUS = int(BRepProjectionStatus.AMBIGUOUS)
_FAILED = int(BRepProjectionStatus.FAILED)


@final
class BRepContainmentResult(StrictModule):
    """Closed-solid membership with its decision route.

    ``status`` is UNIQUE for a certified inside/outside decision and AMBIGUOUS
    for points on the boundary (within the classifier tolerance) or with an
    unresolved source proof. ``winding`` is NaN when no diagnostic is requested.
    ``distances`` and ``distance_upper_bounds`` bound boundary distance from
    above; the lower bound may be a conservative source-carrier separation
    when membership does not require a closest-point solve.
    """

    inside: Array
    status: Array
    winding: Array
    distances: Array
    unresolved: Array
    distance_lower_bounds: Array
    distance_upper_bounds: Array
    query_operations: Array
    resource_exhausted: Array


@final
class _HostContainmentBatch(StrictModule):
    """Host membership decisions and directed boundary-distance evidence."""

    inside: np.ndarray
    unresolved: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    query_operations: np.ndarray
    exhausted: np.ndarray

    @classmethod
    def empty(cls, count: int, /) -> _HostContainmentBatch:
        return cls(
            inside=np.zeros((count,), dtype=np.bool_),
            unresolved=np.ones((count,), dtype=np.bool_),
            lower=np.zeros((count,), dtype=np.float64),
            upper=np.full((count,), np.inf, dtype=np.float64),
            query_operations=np.zeros((count,), dtype=np.int64),
            exhausted=np.zeros((count,), dtype=np.bool_),
        )

    def result(self) -> BRepContainmentResult:
        return BRepContainmentResult(
            inside=jnp.asarray(self.inside),
            status=jnp.asarray(
                np.where(
                    self.exhausted,
                    _FAILED,
                    np.where(self.unresolved, _AMBIGUOUS, _UNIQUE),
                ),
                dtype=jnp.int8,
            ),
            winding=jnp.full(self.inside.shape, jnp.nan, dtype=jnp.float64),
            distances=jnp.asarray(self.upper),
            unresolved=jnp.asarray(self.unresolved),
            distance_lower_bounds=jnp.asarray(self.lower),
            distance_upper_bounds=jnp.asarray(self.upper),
            query_operations=jnp.asarray(self.query_operations),
            resource_exhausted=jnp.asarray(self.exhausted),
        )


def _containment_box_bounds(point: np.ndarray, boxes: np.ndarray, /) -> np.ndarray:
    """Enclose gap distance below/above and farthest-corner distance above.

    Every subtraction, square, addition and square root is rounded outward.
    The gap lower endpoint also bounds the old rational square-root endpoint:
    its initial rounded square root can be decremented by at most one ulp.
    """
    with np.errstate(over="ignore", under="ignore"):
        first = boxes[:, 0] - point
        last = point - boxes[:, 1]
        gap = np.maximum(np.maximum(first, last), 0.0)
        gap_lower = np.maximum(np.nextafter(gap, -np.inf), 0.0)
        gap_upper = np.where(gap == 0.0, 0.0, np.nextafter(gap, np.inf))
        far = np.maximum(np.abs(first), np.abs(last))
        far_upper = np.where(far == 0.0, 0.0, np.nextafter(far, np.inf))
        lower_squares = np.maximum(np.nextafter(gap_lower * gap_lower, -np.inf), 0.0)
        upper_components = np.stack((gap_upper, far_upper), axis=1)
        upper_squares = np.where(
            upper_components == 0.0,
            0.0,
            np.nextafter(upper_components * upper_components, np.inf),
        )
        lower_sum = lower_squares[:, 0]
        upper_sum = upper_squares[:, :, 0]
        for axis in (1, 2):
            lower_sum = np.maximum(
                np.nextafter(lower_sum + lower_squares[:, axis], -np.inf), 0.0
            )
            addition = upper_sum + upper_squares[:, :, axis]
            upper_sum = np.where(addition == 0.0, 0.0, np.nextafter(addition, np.inf))
        lower = np.maximum(np.nextafter(np.sqrt(lower_sum), -np.inf), 0.0)
        upper = np.where(upper_sum == 0.0, 0.0, np.nextafter(np.sqrt(upper_sum), np.inf))
    return np.column_stack((lower, upper))


def _containment_near_mask(
    point: np.ndarray,
    boxes: np.ndarray,
    bounds: np.ndarray,
    classifier: float,
    /,
) -> np.ndarray:
    """Retain the rational lower-square-root predicate at straddled thresholds."""
    near = bounds[:, 0] <= classifier
    uncertain = np.flatnonzero(near & (bounds[:, 1] > classifier))
    if uncertain.size:
        delta = tuple(Fraction(float(value)) for value in point)
        for index in uncertain:
            first = tuple(Fraction(float(value)) for value in boxes[index, 0])
            last = tuple(Fraction(float(value)) for value in boxes[index, 1])
            gap = tuple(
                max(Fraction(0), a - q, q - b)
                for a, b, q in zip(first, last, delta, strict=True)
            )
            exact_lower = square_root_interval(
                sum((value * value for value in gap), Fraction(0))
            )[0]
            bounds[index, 0] = exact_lower
            near[index] = exact_lower <= classifier
    return near


def _containment_face_filter(
    point: np.ndarray,
    boxes: np.ndarray,
    faces: tuple[int, ...],
    face_indices: dict[int, int],
    broadphase: QueryBroadphase,
    classifier: float,
    /,
) -> tuple[tuple[int, ...], float, float, float]:
    """Prune separated BVH nodes, retaining conservative evidence for their cover."""
    frontier = np.asarray((0,), dtype=np.int64)
    candidates = []
    separated_lower = np.inf
    upper = np.inf
    while frontier.size:
        node_boxes = np.stack(
            (broadphase.lower[frontier], broadphase.upper[frontier]), axis=1
        )
        node_bounds = _containment_box_bounds(point, node_boxes)
        if frontier[0] == 0:
            # The root covers every selected face. A child may cover only faces
            # of another solid, so its farthest corner is not an upper bound.
            upper = node_bounds[0, 2]
        separated = node_bounds[:, 0] > classifier
        separated_lower = min(
            separated_lower, np.min(node_bounds[separated, 0], initial=np.inf)
        )
        active = frontier[~separated]
        leaves = broadphase.leaves[active]
        for leaf in leaves[leaves >= 0]:
            candidates.extend(
                face_indices[int(face)]
                for face in broadphase.items[leaf]
                if int(face) in face_indices
            )
        internal = active[leaves < 0]
        frontier = np.column_stack(
            (broadphase.left[internal], broadphase.right[internal])
        ).reshape(-1)
    if not candidates:
        return (), float(separated_lower), float(upper), float(separated_lower)
    indices = np.asarray(sorted(candidates), dtype=np.int64)
    candidate_boxes = boxes[indices]
    bounds = _containment_box_bounds(point, candidate_boxes)
    near_mask = _containment_near_mask(point, candidate_boxes, bounds, classifier)
    near = tuple(faces[index] for index in indices[near_mask])
    lower = min(separated_lower, np.min(bounds[:, 0], initial=np.inf))
    upper = min(upper, np.min(bounds[:, 2], initial=np.inf))
    far_lower = min(separated_lower, np.min(bounds[~near_mask, 0], initial=np.inf))
    return near, float(lower), float(upper), float(far_lower)


@final
class _PeriodicWall(StrictModule):
    """One oriented rectangle wall established from a native chart gauge."""

    axis: int = eqx.field(static=True)
    side: int = eqx.field(static=True)
    first: tuple[int, int] = eqx.field(static=True)
    last: tuple[int, int] = eqx.field(static=True)


def _periodic_wall_extent(
    geometry: BRepGeometry,
    coedge: int,
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    bounds: tuple[tuple[Fraction, ...], ...],
    origin: tuple[Fraction, ...],
    direction: tuple[Fraction, ...],
    shifts: tuple[int, ...],
    axis: int,
    /,
) -> int | None:
    varying = 1 - axis
    roots = geometry.coedge_endpoint_roots[coedge]
    edge = geometry.coedge_edges[coedge]
    first, last = (float(value) for value in geometry.edge_ranges[edge])
    period = patch.periods[varying]
    if period is None:
        values = tuple(
            origin[varying] + direction[varying] * Fraction(value)
            for value in (first, last)
        )
        if (
            set(values) != {bounds[0][varying], bounds[1][varying]}
            or shifts[varying] != 0
        ):
            return None
        start = 0 if values[0] == bounds[0][varying] else 1
    else:
        if direction[varying] != 1 or shifts[varying] != 0:
            return None
        if (
            origin[varying] + Fraction(first) != bounds[0][varying]
            or last != first + period
        ):
            return None
        # Authored native-period endpoints are exactly the start and one
        # full turn later on this wall, never floating representatives.
        expected = (
            period_offset(Fraction(first), Fraction(0)),
            period_offset(Fraction(first), Fraction(1)),
        )
        if any(
            root is not None
            and (
                not isinstance(root, NativePeriodEndpoint)
                or root.exact_parameter != value
            )
            for root, value in zip(roots, expected, strict=True)
        ):
            return None
        carrier_index = geometry.edge_curves[edge]
        if carrier_index >= 0:
            carrier = geometry.curves[carrier_index]
            if not isinstance(
                carrier, (CircleCurve, EllipseCurve, SurfaceIsoparametricCurve)
            ):
                return None
            if carrier.period != period:
                return None
        else:
            if (axis, float(origin[axis])) not in patch.degenerate_isolines(box):
                return None
        start = 0
    return start


def _periodic_coedge_wall(
    geometry: BRepGeometry,
    coedge: int,
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    bounds: tuple[tuple[Fraction, ...], ...],
    /,
) -> _PeriodicWall | None:
    roots = geometry.coedge_endpoint_roots[coedge]
    if any(
        root is not None and not isinstance(root, NativePeriodEndpoint) for root in roots
    ):
        return None
    extracted = pcurve_periodic_source(geometry.pcurves[coedge], patch)
    if extracted is None:
        return None
    base, shifts = extracted
    if not isinstance(base, AbstractCurve):
        return None
    coefficients = affine_curve_coefficients(base, np.zeros((2,), dtype=np.float64))
    if coefficients is None:
        return None
    origin, direction = coefficients
    fixed = [axis for axis in range(2) if direction[axis] == 0]
    if len(fixed) != 1:
        return None
    axis = fixed[0]
    if any(root is not None for root in roots) and patch.periods[1 - axis] is None:
        return None
    if patch.periods[axis] is not None:
        if origin[axis] != bounds[0][axis] or shifts[axis] not in (0, 1):
            return None
        side = shifts[axis]
    else:
        if shifts[axis] != 0 or origin[axis] not in (
            bounds[0][axis],
            bounds[1][axis],
        ):
            return None
        side = 0 if origin[axis] == bounds[0][axis] else 1
    start = _periodic_wall_extent(
        geometry, coedge, patch, box, bounds, origin, direction, shifts, axis
    )
    if start is None:
        return None
    end = 1 - start
    first_corner = (side, start) if axis == 0 else (start, side)
    last_corner = (side, end) if axis == 0 else (end, side)
    if geometry.coedge_senses[coedge] < 0:
        first_corner, last_corner = last_corner, first_corner
    return _PeriodicWall(axis=axis, side=side, first=first_corner, last=last_corner)


def _periodic_rectangle(
    geometry: BRepGeometry, face: int, patch: AbstractSurfacePatch, box: np.ndarray, /
) -> bool:
    """Prove four chart walls from native integer gauges and authored edge phases."""

    loops = geometry.face_loops[face]
    if len(loops) != 1 or len(loops[0]) != 4:
        return False
    if not any(
        isinstance(geometry.pcurves[coedge], PeriodicPCurve) for coedge in loops[0]
    ):
        return False
    bounds = tuple(tuple(Fraction(float(value)) for value in row) for row in box)
    corners: list[tuple[tuple[int, int], tuple[int, int]]] = []
    walls: set[tuple[int, int]] = set()
    for coedge in loops[0]:
        wall = _periodic_coedge_wall(geometry, coedge, patch, box, bounds)
        if wall is None:
            return False
        corners.append((wall.first, wall.last))
        walls.add((wall.axis, wall.side))
    return walls == {(0, 0), (0, 1), (1, 0), (1, 1)} and all(
        last == corners[(index + 1) % len(corners)][0]
        for index, (_, last) in enumerate(corners)
    )


def full_rectangle(
    geometry: BRepGeometry, face: int, patch: AbstractSurfacePatch, box: np.ndarray, /
) -> bool:
    if _periodic_rectangle(geometry, face, patch, box):
        return True
    loops = geometry.face_loops[face]
    if len(loops) != 1:
        return False
    bounds = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in box
    )
    ranges = np.asarray(geometry.edge_ranges)
    segments: list[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]] = []
    for coedge in loops[0]:
        curve = geometry.pcurves[coedge]
        edge = geometry.coedge_edges[coedge]
        if not isinstance(curve, LineCurve) or any(
            root is not None for root in geometry.coedge_endpoint_roots[coedge]
        ):
            return False
        origin, direction = np.asarray(curve.origin), np.asarray(curve.direction)
        ends = tuple(
            tuple(
                Fraction.from_float(float(origin[axis]))
                + Fraction.from_float(float(parameter))
                * Fraction.from_float(float(direction[axis]))
                for axis in range(2)
            )
            for parameter in ranges[edge]
        )
        first = (ends[0][0], ends[0][1])
        last = (ends[1][0], ends[1][1])
        if not all(
            bounds[0][axis] <= point[axis] <= bounds[1][axis]
            for point in (first, last)
            for axis in range(2)
        ):
            return False
        if not any(
            first[axis] == last[axis]
            and first[axis] in (bounds[0][axis], bounds[1][axis])
            for axis in range(2)
        ):
            return False
        segments.append(
            (first, last) if geometry.coedge_senses[coedge] > 0 else (last, first)
        )
    if any(
        last != segments[(index + 1) % len(segments)][0]
        for index, (_, last) in enumerate(segments)
    ):
        return False
    area = sum(
        (first[0] * last[1] - first[1] * last[0] for first, last in segments), Fraction(0)
    )
    return abs(area) == 2 * (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])


def _whole_turn_chart(model: BRepModel, face_index: int, face: FaceData, /) -> bool:
    """Prove the trim spans the chart walls across one native turn."""
    if face.full_rectangle:
        return True
    geometry = model.require_geometry()
    bounds = np.asarray((face.lower, face.upper), dtype=np.float64)
    # A root-enclosure box may exceed one turn. Prove the source's native-gauge
    # walls rather than treating that box as a trim.
    native_box = np.asarray(
        ((0.0, bounds[0, 1]), (face.periods[0], bounds[1, 1])), dtype=np.float64
    )
    return bool(
        bounds[0, 0] <= 0.0
        and bounds[1, 0] >= face.periods[0]
        and _periodic_rectangle(geometry, face_index, face.patch, native_box)
    )


def _solid_sphere_source(
    model: BRepModel,
    faces: tuple[int, ...],
    face: FaceData,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SphereSource | None:
    """Prove angular coverage before using sphere radial membership."""
    if not (
        len(faces) == 1
        and face.periods[0] > 0.0
        and face.lower[1] == -0.5 * pi
        and face.upper[1] == 0.5 * pi
    ):
        return None
    source = prepare_full_sphere(face.patch, budget=budget)
    if source is None or _whole_turn_chart(model, faces[0], face):
        return source
    return None


def _solid_revolution_source(
    model: BRepModel,
    faces: tuple[int, ...],
    face: FaceData,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> RevolutionSource | None:
    """Prove one native turn of a closed meridian before half-plane parity."""
    if len(faces) != 1 or not face.periods[0] > 0.0:
        return None
    bounds = np.asarray((face.lower, face.upper), dtype=np.float64)
    source = prepare_full_revolution(
        face.patch, float(bounds[0, 1]), float(bounds[1, 1]), budget=budget
    )
    if source is None or not _whole_turn_chart(model, faces[0], face):
        return None
    return source


def _contains_affine_host(
    query: PreparedBRepQuery,
    points: np.ndarray,
    faces: tuple[int, ...],
    state: _HostContainmentBatch,
    prepared: tuple[AffineFace, ...],
    broadphase: QueryBroadphase,
    source_boxes: np.ndarray | None,
    /,
    *,
    budget: BRepQueryBudget | None,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _HostContainmentBatch:
    inside, unresolved = state.inside, state.unresolved
    lower, upper = state.lower, state.upper
    query_operations, exhausted = state.query_operations, state.exhausted
    closest = query.affine_closest(
        points,
        faces,
        budget=budget,
        prepared=prepared,
        broadphase=broadphase,
        source_boxes=source_boxes,
        coordinate_budget=coordinate_budget,
    )
    if closest is None:
        raise RuntimeError(
            "PreparedBRepQuery.affine_source must retain its admitted constrained-query capability; got unsupported source"
        )
    lower, upper = (
        np.asarray(closest.distance_lower_bounds),
        np.asarray(closest.distances),
    )
    query_operations[:] = np.asarray(closest.query_operations)
    for row, point in enumerate(points):
        if budget is not None:
            budget.consume(1)
        if upper[row] <= query.tolerances.classifier:
            inside[row] = True
            continue
        decision = certify_affine_parity(point, prepared, faces)
        complete, operations = decision.complete, decision.operations
        if budget is not None:
            budget.consume(operations)
        query_operations[row] += operations
        exhausted[row] = decision.resource_exhausted
        inside[row], unresolved[row] = (
            decision.inside,
            not complete or lower[row] <= query.tolerances.classifier,
        )
    return _HostContainmentBatch(
        inside=inside,
        unresolved=unresolved,
        lower=lower,
        upper=upper,
        query_operations=query_operations,
        exhausted=exhausted,
    )


def _contains_sphere_host(
    query: PreparedBRepQuery,
    points: np.ndarray,
    faces: tuple[int, ...],
    state: _HostContainmentBatch,
    sphere_source: SphereSource,
    /,
    *,
    budget: BRepQueryBudget | None,
) -> _HostContainmentBatch:
    inside, unresolved = state.inside, state.unresolved
    lower, upper = state.lower, state.upper
    query_operations, exhausted = state.query_operations, state.exhausted
    for row, point in enumerate(points):
        query_operations[row] = 1 + len(faces)
        if budget is not None:
            budget.consume(1 + len(faces))
        decision = classify_full_sphere(sphere_source, point)
        lower[row], upper[row] = (
            decision.distance_lower,
            decision.distance_upper,
        )
        inside[row] = decision.inside or upper[row] <= query.tolerances.classifier
        unresolved[row] = lower[row] <= query.tolerances.classifier
    return _HostContainmentBatch(
        inside=inside,
        unresolved=unresolved,
        lower=lower,
        upper=upper,
        query_operations=query_operations,
        exhausted=exhausted,
    )


def _contains_revolution_host(
    query: PreparedBRepQuery,
    points: np.ndarray,
    faces: tuple[int, ...],
    state: _HostContainmentBatch,
    revolution_source: RevolutionSource,
    /,
    *,
    budget: BRepQueryBudget | None,
    preparations: TrimRequestPreparations | None,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _HostContainmentBatch:
    inside, unresolved = state.inside, state.unresolved
    lower, upper = state.lower, state.upper
    query_operations, exhausted = state.query_operations, state.exhausted
    for row, point in enumerate(points):
        query_operations[row] = 1 + len(faces)
        if budget is not None:
            budget.consume(1 + len(faces))
        remaining = (
            query.policy.maximum_subdivisions
            if budget is None
            else min(
                query.policy.maximum_subdivisions,
                budget.maximum_operations - budget.operations,
            )
        )
        if remaining < 1:
            exhausted[row] = True
            continue
        membership = classify_full_revolution(
            revolution_source,
            point,
            tolerance=query.tolerances.classifier,
            maximum_rows=remaining,
            preparations=preparations,
            budget=coordinate_budget,
        )
        if budget is not None:
            budget.consume(membership.operations, subdivisions=membership.operations)
        query_operations[row] += membership.operations
        lower[row], upper[row] = (
            membership.distance_lower,
            membership.distance_upper,
        )
        inside[row] = membership.inside or upper[row] <= query.tolerances.classifier
        unresolved[row] = (
            not membership.complete or lower[row] <= query.tolerances.classifier
        )
        exhausted[row] = membership.exhausted
    return _HostContainmentBatch(
        inside=inside,
        unresolved=unresolved,
        lower=lower,
        upper=upper,
        query_operations=query_operations,
        exhausted=exhausted,
    )


def _contains_general_host(
    query: PreparedBRepQuery,
    points: np.ndarray,
    solid: int,
    faces: tuple[int, ...],
    state: _HostContainmentBatch,
    broadphase: QueryBroadphase,
    /,
    *,
    budget: BRepQueryBudget | None,
    preparations: TrimRequestPreparations | None,
    coordinate_budget: CoordinateEnclosureBudget,
) -> _HostContainmentBatch:
    inside, unresolved = state.inside, state.unresolved
    lower, upper = state.lower, state.upper
    query_operations, exhausted = state.query_operations, state.exhausted
    boxes = np.asarray(query.face_boxes)[list(faces)]
    face_indices = {face: index for index, face in enumerate(faces)}
    for row, point in enumerate(points):
        query_operations[row] = 1 + len(faces)
        if budget is not None:
            budget.consume(1 + len(faces))
        near, lower[row], upper[row], far_lower = _containment_face_filter(
            point, boxes, faces, face_indices, broadphase, query.tolerances.classifier
        )
        if near:
            remaining = (
                query.policy.maximum_subdivisions
                if budget is None
                else min(
                    query.policy.maximum_subdivisions,
                    budget.maximum_operations - budget.operations,
                )
            )
            if remaining < 1:
                exhausted[row] = True
                continue
            closest = query.closest_definition(
                jnp.asarray(point[None]),
                near,
                maximum_boxes=remaining,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
            if budget is not None:
                refinements = int(np.sum(np.asarray(closest.refinements)))
                budget.consume(int(closest.query_operations[0]), subdivisions=refinements)
            query_operations[row] += int(closest.query_operations[0])
            lower[row], upper[row] = (
                min(float(closest.distance_lower_bounds[0]), far_lower),
                float(closest.distances[0]),
            )
            if upper[row] <= query.tolerances.classifier:
                inside[row] = True
                continue
            if lower[row] <= query.tolerances.classifier:
                exhausted[row] = (
                    query.policy.maximum_depth == 0
                    or query.policy.maximum_subdivisions == 0
                    or int(closest.refinements[0]) >= remaining
                )
                continue
        remaining = (
            query.policy.maximum_subdivisions
            if budget is None
            else min(
                query.policy.maximum_subdivisions,
                budget.maximum_operations - budget.operations,
            )
        )
        if remaining < 1:
            exhausted[row] = True
            continue
        certificate = containment_owner.certify_solid_containment(
            query.model,
            point,
            solid,
            maximum_boxes=remaining,
            maximum_depth=query.policy.maximum_depth,
            maximum_operations=None
            if budget is None
            else budget.maximum_operations - budget.operations,
            source_boxes=boxes,
            trim_domains=tuple(query.faces[face].trim_domain for face in faces),
            broadphase=broadphase,
            preparations=preparations,
            budget=coordinate_budget,
        )
        if budget is not None:
            budget.consume(
                certificate.owner_operations,
                subdivisions=certificate.boxes_processed,
            )
        query_operations[row] += certificate.owner_operations
        exhausted[row] = certificate.resource_exhausted
        inside[row], unresolved[row] = (
            certificate.inside,
            not certificate.complete,
        )
    return _HostContainmentBatch(
        inside=inside,
        unresolved=unresolved,
        lower=lower,
        upper=upper,
        query_operations=query_operations,
        exhausted=exhausted,
    )


def classify_host_containment(
    query: PreparedBRepQuery,
    points: np.ndarray,
    solid: int,
    /,
    *,
    budget: BRepQueryBudget | None,
    affine_preparation: tuple[tuple[AffineFace, ...], QueryBroadphase, np.ndarray]
    | None = None,
    preparations: TrimRequestPreparations | None = None,
    coordinate_budget: CoordinateEnclosureBudget,
) -> BRepContainmentResult:
    faces = query.solid_faces[solid]
    prepared = (
        prepare_affine_faces(query.model, faces, budget=coordinate_budget)
        if affine_preparation is None
        else affine_preparation[0]
    )
    broadphase = (
        QueryBroadphase.prepare(query.face_bvh)
        if affine_preparation is None
        else affine_preparation[1]
    )
    if budget is not None:
        budget.consume(0, points=len(points))
    state = _HostContainmentBatch.empty(len(points))
    if prepared is not None:
        state = _contains_affine_host(
            query,
            points,
            faces,
            state,
            prepared,
            broadphase,
            None if affine_preparation is None else affine_preparation[2],
            budget=budget,
            coordinate_budget=coordinate_budget,
        )
    else:
        face = query.faces[faces[0]]
        sphere = _solid_sphere_source(query.model, faces, face, budget=coordinate_budget)
        revolution = (
            _solid_revolution_source(query.model, faces, face, budget=coordinate_budget)
            if sphere is None
            else None
        )
        match (sphere, revolution):
            case (SphereSource(), _):
                state = _contains_sphere_host(
                    query, points, faces, state, sphere, budget=budget
                )
            case (None, RevolutionSource()):
                state = _contains_revolution_host(
                    query,
                    points,
                    faces,
                    state,
                    revolution,
                    budget=budget,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
            case (None, None):
                state = _contains_general_host(
                    query,
                    points,
                    solid,
                    faces,
                    state,
                    broadphase,
                    budget=budget,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
            case _:
                raise ValueError(
                    "PreparedBRepQuery.membership_source must select an admitted strategy; got incompatible source proofs"
                )
    return state.result()


__all__ = ["BRepContainmentResult", "classify_host_containment", "full_rectangle"]
