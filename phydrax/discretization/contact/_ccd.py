#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._guarantee import (
    ContactGuaranteeEvidence,
    ContactGuaranteeLevel,
)
from ._search import ContactCandidateEpoch
from ._stencils import ContactStencilBatch, ContactStencilKind
from ._surface import PreparedCollisionScene


class CCDStatus(IntEnum):
    SUCCESS = 0
    INITIAL_DISTANCE_VIOLATION = 1
    SEARCH_INCOMPLETE = 2
    WORK_LIMIT = 3
    NONFINITE_INPUT = 4


class InclusionCCDPlan(StrictModule, NonTrainableState):
    """Conservative linear-trajectory CCD using Lipschitz interval inclusion."""

    time_tolerance: float = eqx.field(static=True)
    numerical_error: float = eqx.field(static=True)
    conservative_rescaling: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    required_guarantee: ContactGuaranteeLevel = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        time_tolerance: float = 1.0e-8,
        numerical_error: float = 1.0e-12,
        conservative_rescaling: float = 0.8,
        maximum_iterations: int = 1_000_000,
        required_guarantee: ContactGuaranteeLevel = (
            ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
        ),
    ):
        tolerance = float(time_tolerance)
        error = float(numerical_error)
        rescaling = float(conservative_rescaling)
        iterations = int(maximum_iterations)
        required = ContactGuaranteeLevel(required_guarantee)
        if required > ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE:
            raise ValueError(
                "InclusionCCDPlan cannot satisfy a guarantee stronger than "
                "ENCLOSURE_CONSERVATIVE."
            )
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("time_tolerance must be finite and positive.")
        if not isfinite(error) or error < 0.0:
            raise ValueError("numerical_error must be finite and nonnegative.")
        if not isfinite(rescaling) or not 0.0 < rescaling < 1.0:
            raise ValueError(
                "conservative_rescaling must lie strictly between zero and one."
            )
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        self.time_tolerance = tolerance
        self.numerical_error = error
        self.conservative_rescaling = rescaling
        self.maximum_iterations = iterations
        self.required_guarantee = required
        self.plan_id = canonical_fingerprint(
            {
                "kind": "inclusion-ccd-plan",
                "time_tolerance": tolerance.hex(),
                "numerical_error": error.hex(),
                "conservative_rescaling": rescaling.hex(),
                "maximum_iterations": iterations,
                "required_guarantee": int(required),
            }
        )


class CertifiedAABBCCDPlan(StrictModule, NonTrainableState):
    """Roundoff-directed swept-AABB interval CCD.

    This backend is deliberately conservative: an interval is certified free
    only when outward-rounded primitive AABBs are separated by more than the
    requested minimum distance. Unresolved intervals report contact at their
    lower endpoint.
    """

    time_tolerance: float = eqx.field(static=True)
    conservative_rescaling: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    required_guarantee: ContactGuaranteeLevel = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        time_tolerance: float = 1.0e-10,
        conservative_rescaling: float = 0.8,
        maximum_iterations: int = 1_000_000,
        required_guarantee: ContactGuaranteeLevel = (
            ContactGuaranteeLevel.ROUNDING_CERTIFIED
        ),
    ):
        tolerance = float(time_tolerance)
        rescaling = float(conservative_rescaling)
        iterations = int(maximum_iterations)
        required = ContactGuaranteeLevel(required_guarantee)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("time_tolerance must be finite and positive.")
        if not isfinite(rescaling) or not 0.0 < rescaling < 1.0:
            raise ValueError(
                "conservative_rescaling must lie strictly between zero and one."
            )
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if required > ContactGuaranteeLevel.ROUNDING_CERTIFIED:
            raise ValueError("Unsupported CCD guarantee requirement.")
        self.time_tolerance = tolerance
        self.conservative_rescaling = rescaling
        self.maximum_iterations = iterations
        self.required_guarantee = required
        self.plan_id = canonical_fingerprint(
            {
                "kind": "certified-aabb-ccd-plan",
                "time_tolerance": tolerance.hex(),
                "conservative_rescaling": rescaling.hex(),
                "maximum_iterations": iterations,
                "required_guarantee": int(required),
            }
        )


CCDPlan = InclusionCCDPlan | CertifiedAABBCCDPlan


class ContactSafetyEvidence(StrictModule):
    step_size: Array
    minimum_time_of_impact: Array
    limiting_route_key: Array
    query_count: Array
    interval_count: Array
    initial_violations: Array
    status: Array
    finite: Array
    conservative: Array
    guarantee: ContactGuaranteeEvidence
    epoch_id: str = eqx.field(static=True)
    ccd_plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (self.status == int(CCDStatus.SUCCESS)) & self.guarantee.successful


def _point_segment_distance(point, first, second, /) -> float:
    edge = second - first
    denominator = float(np.dot(edge, edge))
    if denominator <= 0.0:
        return float(np.linalg.norm(point - first))
    coordinate = float(np.clip(np.dot(point - first, edge) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + coordinate * edge)))


def _point_triangle_distance(point, first, second, third, /) -> float:
    edge_first = second - first
    edge_second = third - first
    normal = np.cross(edge_first, edge_second)
    area_squared = float(np.dot(normal, normal))
    candidates = [
        _point_segment_distance(point, first, second),
        _point_segment_distance(point, second, third),
        _point_segment_distance(point, third, first),
    ]
    if area_squared > 0.0:
        witness = point - np.dot(point - first, normal) / area_squared * normal
        dot00 = float(np.dot(edge_first, edge_first))
        dot01 = float(np.dot(edge_first, edge_second))
        dot11 = float(np.dot(edge_second, edge_second))
        rhs0 = float(np.dot(witness - first, edge_first))
        rhs1 = float(np.dot(witness - first, edge_second))
        determinant = dot00 * dot11 - dot01 * dot01
        if determinant > 0.0:
            second_weight = (dot11 * rhs0 - dot01 * rhs1) / determinant
            third_weight = (dot00 * rhs1 - dot01 * rhs0) / determinant
            first_weight = 1.0 - second_weight - third_weight
            if min(first_weight, second_weight, third_weight) >= 0.0:
                candidates.append(float(np.linalg.norm(point - witness)))
    return min(candidates)


def _segment_segment_distance(a0, a1, b0, b1, /) -> float:
    ua = a1 - a0
    ub = b1 - b0
    aa = float(np.dot(ua, ua))
    bb = float(np.dot(ua, ub))
    cc = float(np.dot(ub, ub))
    w = a0 - b0
    dd = float(np.dot(ua, w))
    ee = float(np.dot(ub, w))
    determinant = aa * cc - bb * bb
    candidates = [
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    ]
    if determinant > 0.0:
        s = (bb * ee - cc * dd) / determinant
        t = (aa * ee - bb * dd) / determinant
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            candidates.append(float(np.linalg.norm((a0 + s * ua) - (b0 + t * ub))))
    return min(candidates)


def _stencil_distance(kind: ContactStencilKind, vertices: np.ndarray, /) -> float:
    if kind == ContactStencilKind.VERTEX_VERTEX:
        return float(np.linalg.norm(vertices[0] - vertices[1]))
    if kind == ContactStencilKind.EDGE_VERTEX:
        return _point_segment_distance(vertices[0], vertices[1], vertices[2])
    if kind == ContactStencilKind.FACE_VERTEX:
        return _point_triangle_distance(
            vertices[0], vertices[1], vertices[2], vertices[3]
        )
    if kind == ContactStencilKind.EDGE_EDGE:
        return _segment_segment_distance(
            vertices[0], vertices[1], vertices[2], vertices[3]
        )
    raise TypeError("Unsupported CCD stencil kind.")


def _relative_speed_bound(kind: ContactStencilKind, velocities: np.ndarray, /) -> float:
    speed = np.linalg.norm(velocities, axis=1)
    if kind == ContactStencilKind.VERTEX_VERTEX:
        return float(speed[0] + speed[1])
    if kind == ContactStencilKind.EDGE_VERTEX:
        return float(speed[0] + max(speed[1], speed[2]))
    if kind == ContactStencilKind.FACE_VERTEX:
        return float(speed[0] + max(speed[1], speed[2], speed[3]))
    if kind == ContactStencilKind.EDGE_EDGE:
        return float(max(speed[0], speed[1]) + max(speed[2], speed[3]))
    raise TypeError("Unsupported CCD stencil kind.")


def _candidate_toi(
    plan: InclusionCCDPlan,
    kind: ContactStencilKind,
    start: np.ndarray,
    end: np.ndarray,
    minimum_separation: float,
    /,
) -> tuple[float, int, bool, bool]:
    initial = _stencil_distance(kind, start)
    if initial <= minimum_separation + plan.numerical_error:
        return 0.0, 1, True, False
    velocity = end - start
    speed_bound = _relative_speed_bound(kind, velocity)
    if speed_bound == 0.0:
        return np.inf, 1, False, False
    stack: list[tuple[float, float]] = [(0.0, 1.0)]
    iterations = 0
    while stack:
        if iterations >= plan.maximum_iterations:
            return 0.0, iterations, False, True
        lower_time, upper_time = stack.pop()
        iterations += 1
        midpoint = 0.5 * (lower_time + upper_time)
        midpoint_vertices = start + midpoint * velocity
        midpoint_distance = _stencil_distance(kind, midpoint_vertices)
        half_width = 0.5 * (upper_time - lower_time)
        lower_distance = (
            midpoint_distance - half_width * speed_bound - plan.numerical_error
        )
        if lower_distance > minimum_separation:
            continue
        if upper_time - lower_time <= plan.time_tolerance:
            return lower_time, iterations, False, False
        split = midpoint
        stack.append((split, upper_time))
        stack.append((lower_time, split))
    return np.inf, iterations, False, False


def _primitive_partition(
    kind: ContactStencilKind, vertices: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    if kind == ContactStencilKind.VERTEX_VERTEX:
        return vertices[:1], vertices[1:2]
    if kind in (
        ContactStencilKind.EDGE_VERTEX,
        ContactStencilKind.FACE_VERTEX,
    ):
        return vertices[:1], vertices[1:]
    if kind == ContactStencilKind.EDGE_EDGE:
        return vertices[:2], vertices[2:]
    raise TypeError("Unsupported CCD stencil kind.")


def _outward_bounds(
    start: np.ndarray,
    velocity: np.ndarray,
    lower_time: float,
    upper_time: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lower_positions = start + lower_time * velocity
    upper_positions = start + upper_time * velocity
    lower = np.minimum(lower_positions, upper_positions).min(axis=0)
    upper = np.maximum(lower_positions, upper_positions).max(axis=0)
    for _ in range(4):
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
    return lower, upper


def _rounded_aabb_distance_squared(
    first_min: np.ndarray,
    first_max: np.ndarray,
    second_min: np.ndarray,
    second_max: np.ndarray,
    /,
) -> float:
    separation = np.maximum(
        0.0,
        np.maximum(first_min - second_max, second_min - first_max),
    )
    for _ in range(4):
        separation = np.maximum(0.0, np.nextafter(separation, -np.inf))
    squared = np.nextafter(separation * separation, -np.inf)
    result = np.asarray(0.0, dtype=np.float64)
    for value in squared:
        result = np.nextafter(result + value, -np.inf)
    return float(max(result, 0.0))


def _candidate_toi_certified(
    plan: CertifiedAABBCCDPlan,
    kind: ContactStencilKind,
    start: np.ndarray,
    end: np.ndarray,
    minimum_separation: float,
    /,
) -> tuple[float, int, bool, bool]:
    initial = np.nextafter(_stencil_distance(kind, start), -np.inf)
    threshold = np.nextafter(minimum_separation, np.inf)
    if initial <= threshold:
        return 0.0, 1, True, False
    velocity = end - start
    if np.all(velocity == 0.0):
        return np.inf, 1, False, False
    first_start, second_start = _primitive_partition(kind, start)
    first_velocity, second_velocity = _primitive_partition(kind, velocity)
    threshold_squared = np.nextafter(minimum_separation * minimum_separation, np.inf)
    stack: list[tuple[float, float]] = [(0.0, 1.0)]
    iterations = 0
    while stack:
        if iterations >= plan.maximum_iterations:
            return 0.0, iterations, False, True
        lower_time, upper_time = stack.pop()
        iterations += 1
        first_min, first_max = _outward_bounds(
            first_start,
            first_velocity,
            lower_time,
            upper_time,
        )
        second_min, second_max = _outward_bounds(
            second_start,
            second_velocity,
            lower_time,
            upper_time,
        )
        lower_squared = _rounded_aabb_distance_squared(
            first_min, first_max, second_min, second_max
        )
        if lower_squared > threshold_squared:
            continue
        if upper_time - lower_time <= plan.time_tolerance:
            return lower_time, iterations, False, False
        midpoint = 0.5 * (lower_time + upper_time)
        stack.append((midpoint, upper_time))
        stack.append((lower_time, midpoint))
    return np.inf, iterations, False, False


def _batch_step_limit(
    plan: CCDPlan,
    batch: ContactStencilBatch,
    start: np.ndarray,
    end: np.ndarray,
    /,
) -> tuple[float, int, int, int, int, bool]:
    minimum_toi = np.inf
    limiting_key = 0
    query_count = 0
    interval_count = 0
    initial_violations = 0
    exhausted = False
    for slot in np.flatnonzero(np.asarray(batch.valid, dtype=bool)).tolist():
        indices = np.asarray(batch.vertex_indices[slot], dtype=np.int32)
        arity = (
            2
            if batch.kind == ContactStencilKind.VERTEX_VERTEX
            else 3
            if batch.kind == ContactStencilKind.EDGE_VERTEX
            else 4
        )
        vertices_start = start[indices[:arity]]
        vertices_end = end[indices[:arity]]
        candidate_query = (
            _candidate_toi_certified
            if isinstance(plan, CertifiedAABBCCDPlan)
            else _candidate_toi
        )
        toi, intervals, initial, work_exhausted = candidate_query(
            plan,
            batch.kind,
            vertices_start,
            vertices_end,
            float(batch.minimum_separation[slot]),
        )
        query_count += 1
        interval_count += intervals
        initial_violations += int(initial)
        exhausted = exhausted or work_exhausted
        if toi < minimum_toi:
            minimum_toi = toi
            limiting_key = int(batch.route_keys[slot])
    return (
        minimum_toi,
        limiting_key,
        query_count,
        interval_count,
        initial_violations,
        exhausted,
    )


def _make_ccd_guarantee(
    plan: CCDPlan,
    /,
    *,
    finite: bool,
    work_complete: bool,
    failure_code: int,
    margin: float,
) -> ContactGuaranteeEvidence:
    level = (
        ContactGuaranteeLevel.ROUNDING_CERTIFIED
        if isinstance(plan, CertifiedAABBCCDPlan)
        else ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
    )
    return ContactGuaranteeEvidence(
        level,
        required_level=plan.required_guarantee,
        finite=finite,
        work_complete=work_complete,
        failure_code=failure_code,
        margin=margin,
        backend_id=plan.plan_id,
    )


def collision_free_step_limit(
    plan: CCDPlan,
    scene: PreparedCollisionScene,
    epoch: ContactCandidateEpoch,
    start_positions: ArrayLike,
    end_positions: ArrayLike,
    /,
) -> ContactSafetyEvidence:
    if not isinstance(plan, (InclusionCCDPlan, CertifiedAABBCCDPlan)):
        raise TypeError("plan must be a concrete CCD plan.")
    if not isinstance(scene, PreparedCollisionScene):
        raise TypeError("scene must be PreparedCollisionScene.")
    if not isinstance(epoch, ContactCandidateEpoch):
        raise TypeError("epoch must be ContactCandidateEpoch.")
    start = np.asarray(start_positions, dtype=np.float64)
    end = np.asarray(end_positions, dtype=np.float64)
    expected = (scene.vertex_count, scene.ambient_dimension)
    finite = (
        start.shape == expected
        and end.shape == expected
        and np.all(np.isfinite(start))
        and np.all(np.isfinite(end))
    )
    if not finite:
        return ContactSafetyEvidence(
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(CCDStatus.NONFINITE_INPUT), dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(True),
            _make_ccd_guarantee(
                plan,
                finite=False,
                work_complete=False,
                failure_code=int(CCDStatus.NONFINITE_INPUT),
                margin=0.0,
            ),
            epoch.epoch_id,
            plan.plan_id,
        )
    if not bool(epoch.successful):
        status = CCDStatus.SEARCH_INCOMPLETE
        return ContactSafetyEvidence(
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(status), dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(True),
            _make_ccd_guarantee(
                plan,
                finite=True,
                work_complete=False,
                failure_code=int(CCDStatus.SEARCH_INCOMPLETE),
                margin=0.0,
            ),
            epoch.epoch_id,
            plan.plan_id,
        )
    minimum_toi = np.inf
    limiting_key = 0
    queries = 0
    intervals = 0
    initial_violations = 0
    exhausted = False
    for batch in epoch.active_batches:
        result = _batch_step_limit(plan, batch, start, end)
        toi, key, query_count, interval_count, initial_count, work_exhausted = result
        if toi < minimum_toi:
            minimum_toi = toi
            limiting_key = key
        queries += query_count
        intervals += interval_count
        initial_violations += initial_count
        exhausted = exhausted or work_exhausted
    if initial_violations:
        status = CCDStatus.INITIAL_DISTANCE_VIOLATION
        step = 0.0
    elif exhausted:
        status = CCDStatus.WORK_LIMIT
        step = 0.0
    else:
        status = CCDStatus.SUCCESS
        step = (
            1.0
            if not np.isfinite(minimum_toi)
            else min(1.0, plan.conservative_rescaling * minimum_toi)
        )
    reported_toi = 1.0 if not np.isfinite(minimum_toi) else minimum_toi
    return ContactSafetyEvidence(
        jnp.asarray(step),
        jnp.asarray(reported_toi),
        jnp.asarray(limiting_key, dtype=jnp.int64),
        jnp.asarray(queries, dtype=jnp.int32),
        jnp.asarray(intervals, dtype=jnp.int32),
        jnp.asarray(initial_violations, dtype=jnp.int32),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(True),
        _make_ccd_guarantee(
            plan,
            finite=True,
            work_complete=not exhausted,
            failure_code=(0 if status == CCDStatus.SUCCESS else int(status)),
            margin=float(step),
        ),
        epoch.epoch_id,
        plan.plan_id,
    )


__all__ = [
    "CCDStatus",
    "ContactSafetyEvidence",
    "InclusionCCDPlan",
    "CertifiedAABBCCDPlan",
    "ContactGuaranteeEvidence",
    "ContactGuaranteeLevel",
    "collision_free_step_limit",
]
