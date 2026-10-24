#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import ceil, isfinite, log2

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked
from ._distance import (
    ContactWitnessSlot,
    edge_edge_distance,
    point_edge_distance,
    point_point_distance,
    point_triangle_distance,
)
from ._guarantee import (
    ContactGuaranteeEvidence,
    ContactGuaranteeLevel,
)
from ._participant import ContactParticipantScene
from ._search import ContactCandidateEpoch, scene_route_geometry
from ._stencils import ContactStencilBatch, ContactStencilKind
from ._surface import PreparedCollisionScene
from ._trajectory import AbstractContactTrajectory


class CCDStatus(IntEnum):
    SUCCESS = 0
    INITIAL_DISTANCE_VIOLATION = 1
    SEARCH_INCOMPLETE = 2
    WORK_LIMIT = 3
    NONFINITE_INPUT = 4
    TRAJECTORY_BOUND_FAILURE = 5
    INVALID_GEOMETRY = 6


class InclusionCCDPlan(StrictModule, NonTrainableState):
    """Conservative CCD using Lipschitz or supplied full-path interval enclosures.

    Unsupported degenerate primitive distances refuse a collision-free certificate.
    """

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
    ) -> None:
        tolerance = float(time_tolerance)
        error = float(numerical_error)
        rescaling = float(conservative_rescaling)
        iterations = int(maximum_iterations)
        required = ContactGuaranteeLevel(required_guarantee)
        if required > ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE:
            raise ValueError(
                "InclusionCCDPlan cannot satisfy a guarantee stronger than ENCLOSURE_CONSERVATIVE."
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
    ) -> None:
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


_PLANE_KINDS = (ContactStencilKind.VERTEX_PLANE, ContactStencilKind.EDGE_PLANE)


def _partition_slots(
    kind: ContactStencilKind, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Static left/right vertex-column partition of one route kind."""
    match kind:
        case ContactStencilKind.VERTEX_VERTEX:
            return (0,), (1,)
        case ContactStencilKind.EDGE_VERTEX:
            return (0,), (1, 2)
        case ContactStencilKind.FACE_VERTEX:
            return (0,), (1, 2, 3)
        case ContactStencilKind.EDGE_EDGE:
            return (0, 1), (2, 3)
        case ContactStencilKind.VERTEX_PLANE:
            return (0,), ()
        case ContactStencilKind.EDGE_PLANE:
            return (0, 1), ()
        case _:
            raise TypeError("Unsupported CCD stencil kind.")


def _arity(kind: ContactStencilKind, /) -> int:
    left, right = _partition_slots(kind)
    return len(left) + len(right)


def _unit_normal(plane_normal: Array, /) -> Array:
    """Whether a plane normal is finite and unit within the shared tolerance."""
    squared = jnp.sum(plane_normal * plane_normal)
    tolerance = 64.0 * jnp.finfo(plane_normal.dtype).eps
    return jnp.all(jnp.isfinite(plane_normal)) & (jnp.abs(squared - 1.0) <= tolerance)


def _directed_dot(points: Array, direction: Array, /) -> tuple[Array, Array]:
    """Directed lower/upper bounds of ``points @ direction`` in floating point.

    The rounding error of a length-three dot product is below ``1.5 eps`` times
    ``|points| @ |direction|``; ``4 eps`` plus one outward ulp per operation is
    used.
    """
    value = points @ direction
    error = 4.0 * jnp.finfo(points.dtype).eps * (jnp.abs(points) @ jnp.abs(direction))
    error = jnp.nextafter(error, jnp.full_like(error, jnp.inf))
    return (
        jnp.nextafter(value - error, jnp.full_like(value, -jnp.inf)),
        jnp.nextafter(value + error, jnp.full_like(value, jnp.inf)),
    )


def _plane_heights_lower(
    vertices: Array, plane_normal: Array, plane_offset: Array, /
) -> Array:
    """Directed lower bounds of ``n·x - d`` for each vertex."""
    lower, _ = _directed_dot(vertices, plane_normal)
    return jnp.nextafter(lower - plane_offset, jnp.full_like(lower, -jnp.inf))


def _separating_axis_lower(
    kind: ContactStencilKind, vertices: Array, direction: Array, /
) -> Array:
    """Rigorous lower bound of the core distance along a trial axis.

    For any nonzero ``d`` and convex features ``L`` and ``R``,
    ``dist(L, R) >= (min_L d·x - max_R d·x) / |d|``, with extrema attained at
    vertices. The bound holds for any axis, so an ill-conditioned closest-point
    solve (for example determinant cancellation of nearly parallel crossing
    segments) can only weaken it, never make it unsound. Distances are
    nonnegative, so nonpositive projections give the trivial bound zero.
    """
    left, right = _partition_slots(kind)
    left_lower, _ = _directed_dot(vertices[jnp.asarray(left, dtype=jnp.int32)], direction)
    _, right_upper = _directed_dot(
        vertices[jnp.asarray(right, dtype=jnp.int32)], direction
    )
    separation = jnp.nextafter(
        jnp.min(left_lower) - jnp.max(right_upper),
        jnp.full((), -jnp.inf, dtype=vertices.dtype),
    )
    norm = jnp.sqrt(jnp.sum(direction * direction)) * (
        1.0 + 4.0 * jnp.finfo(vertices.dtype).eps
    )
    norm = jnp.nextafter(norm, jnp.full_like(norm, jnp.inf))
    usable = jnp.isfinite(separation) & (separation > 0.0) & (norm > 0.0)
    quotient = jnp.nextafter(
        separation / jnp.where(usable, norm, 1.0),
        jnp.full((), -jnp.inf, dtype=vertices.dtype),
    )
    return jnp.where(usable, quotient, 0.0)


def _core_distance(
    kind: ContactStencilKind,
    vertices: Array,
    plane_normal: Array,
    plane_offset: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array, Array]:
    """Return ``(lower, witness, valid)`` core distances of one route.

    ``lower`` is a rigorous floating-point lower bound (signed for planes);
    ``witness`` is the distance between the kernel's witness points, an upper
    estimate used only to prove violations. ``valid`` requires nondegenerate
    primitives and, for planes, a finite unit normal. Touching cores are valid.
    """
    squared_tolerance = tolerance * tolerance

    def length_squared(first: Array, second: Array) -> Array:
        difference = second - first
        return jnp.sum(difference * difference, axis=-1)

    match kind:
        case ContactStencilKind.VERTEX_VERTEX:
            result = point_point_distance(vertices[0], vertices[1], tolerance=tolerance)
            valid = jnp.asarray(True)
        case ContactStencilKind.EDGE_VERTEX:
            result = point_edge_distance(
                vertices[0], vertices[1], vertices[2], tolerance=tolerance
            )
            valid = length_squared(vertices[1], vertices[2]) > squared_tolerance
        case ContactStencilKind.FACE_VERTEX:
            result = point_triangle_distance(
                vertices[0], vertices[1], vertices[2], vertices[3], tolerance=tolerance
            )
            cross = jnp.cross(vertices[2] - vertices[1], vertices[3] - vertices[1])
            valid = jnp.sum(cross * cross) > squared_tolerance
        case ContactStencilKind.EDGE_EDGE:
            result = edge_edge_distance(
                vertices[0], vertices[1], vertices[2], vertices[3], tolerance=tolerance
            )
            valid = (length_squared(vertices[0], vertices[1]) > squared_tolerance) & (
                length_squared(vertices[2], vertices[3]) > squared_tolerance
            )
        case ContactStencilKind.VERTEX_PLANE | ContactStencilKind.EDGE_PLANE:
            left, _ = _partition_slots(kind)
            features = vertices[jnp.asarray(left, dtype=jnp.int32)]
            lower = jnp.min(_plane_heights_lower(features, plane_normal, plane_offset))
            witness = jnp.min(features @ plane_normal) - plane_offset
            valid = _unit_normal(plane_normal) & jnp.isfinite(plane_offset)
            if kind == ContactStencilKind.EDGE_PLANE:
                valid = valid & (
                    length_squared(vertices[0], vertices[1]) > squared_tolerance
                )
            return lower, witness, valid & jnp.isfinite(lower)
        case _:
            raise TypeError("Unsupported CCD stencil kind.")
    witness = jnp.sqrt(jnp.maximum(result.squared_distance, 0.0))
    lower = _separating_axis_lower(kind, vertices, result.distance_vector)
    return lower, witness, valid & result.finite & jnp.isfinite(lower)


def _stencil_distance(
    kind: ContactStencilKind,
    vertices: Array,
    plane_normal: Array,
    plane_offset: Array,
    /,
) -> Array:
    lower, _, valid = _core_distance(kind, vertices, plane_normal, plane_offset, 0.0)
    # Degenerate or invalid primitives are not a separation certificate.
    return jnp.where(valid, lower, jnp.minimum(lower, 0.0))


def _outward_bounds(
    start: Array, velocity: Array, lower_time: Array, upper_time: Array, /
) -> tuple[Array, Array]:
    negative = jnp.full_like(start, -jnp.inf)
    positive = jnp.full_like(start, jnp.inf)
    velocity_lower = jnp.nextafter(velocity, negative)
    velocity_upper = jnp.nextafter(velocity, positive)

    def endpoint(time: Array) -> tuple[Array, Array]:
        product_lower = jnp.nextafter(time * velocity_lower, negative)
        product_upper = jnp.nextafter(time * velocity_upper, positive)
        return (
            jnp.nextafter(start + product_lower, negative),
            jnp.nextafter(start + product_upper, positive),
        )

    first_lower, first_upper = endpoint(lower_time)
    second_lower, second_upper = endpoint(upper_time)
    lower = jnp.minimum(first_lower, second_lower).min(axis=0)
    upper = jnp.maximum(first_upper, second_upper).max(axis=0)
    return lower, upper


def _rounded_aabb_distance_squared(
    first_min: Array, first_max: Array, second_min: Array, second_max: Array, /
) -> Array:
    separation = jnp.maximum(
        jnp.zeros_like(first_min),
        jnp.maximum(first_min - second_max, second_min - first_max),
    )
    for _ in range(4):
        separation = jnp.maximum(
            jnp.zeros_like(separation),
            jnp.nextafter(separation, jnp.full_like(separation, -jnp.inf)),
        )
    squared = jnp.nextafter(separation * separation, jnp.full_like(separation, -jnp.inf))

    def add(result: Array, value: Array) -> tuple[Array, None]:
        total = result + value
        return jnp.nextafter(total, jnp.full_like(total, -jnp.inf)), None

    total, _ = jax.lax.scan(add, jnp.zeros((), dtype=separation.dtype), squared)
    return jnp.maximum(total, jnp.zeros_like(total))


def _rounded_plane_lower(
    lower: Array, upper: Array, plane_normal: Array, plane_offset: Array, /
) -> Array:
    """Directed lower bound of ``n·x - d`` over the union of vertex boxes."""
    negative = jnp.full((), -jnp.inf, dtype=lower.dtype)
    corner = jnp.where(plane_normal >= 0.0, lower, upper)
    products = jnp.nextafter(corner * plane_normal, negative)

    def add(result: Array, value: Array) -> tuple[Array, None]:
        return jnp.nextafter(result + value, negative), None

    support, _ = jax.lax.scan(
        add, jnp.zeros(lower.shape[:-1], dtype=lower.dtype), jnp.moveaxis(products, -1, 0)
    )
    return jnp.nextafter(jnp.min(support) - plane_offset, negative)


def _box_lower_distance(
    kind: ContactStencilKind,
    lower: Array,
    upper: Array,
    plane_normal: Array,
    plane_offset: Array,
    /,
) -> Array:
    """Lower bound of the core distance from per-vertex boxes."""
    left, right = _partition_slots(kind)
    left_index = jnp.asarray(left, dtype=jnp.int32)
    if kind in _PLANE_KINDS:
        return _rounded_plane_lower(
            lower[left_index], upper[left_index], plane_normal, plane_offset
        )
    right_index = jnp.asarray(right, dtype=jnp.int32)
    squared = _rounded_aabb_distance_squared(
        lower[left_index].min(axis=0),
        upper[left_index].max(axis=0),
        lower[right_index].min(axis=0),
        upper[right_index].max(axis=0),
    )
    return jnp.nextafter(jnp.sqrt(squared), jnp.full((), -jnp.inf, dtype=lower.dtype))


def _ball_lower_distance(
    kind: ContactStencilKind,
    centers: Array,
    radii: Array,
    plane_normal: Array,
    plane_offset: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array]:
    """Midpoint distance lower bound minus Hausdorff displacement bounds.

    Each feature is the convex hull of its vertices; if every vertex stays
    within ``radii[i]`` of ``centers[i]`` then the feature moves by at most the
    maximum vertex radius. Plane gaps are bounded vertexwise. All operations
    round toward the lower bound.
    """
    negative = jnp.full((), -jnp.inf, dtype=centers.dtype)
    left, right = _partition_slots(kind)
    left_index = jnp.asarray(left, dtype=jnp.int32)
    distance, _, valid = _core_distance(
        kind, centers, plane_normal, plane_offset, tolerance
    )
    if kind in _PLANE_KINDS:
        heights = _plane_heights_lower(centers[left_index], plane_normal, plane_offset)
        return jnp.min(jnp.nextafter(heights - radii[left_index], negative)), valid
    right_index = jnp.asarray(right, dtype=jnp.int32)
    bound = jnp.nextafter(distance - jnp.max(radii[left_index]), negative)
    bound = jnp.nextafter(bound - jnp.max(radii[right_index]), negative)
    return bound, valid


def _interval_free(
    plan: CCDPlan,
    kind: ContactStencilKind,
    start: Array,
    velocity: Array,
    separation: Array,
    lower: Array,
    upper: Array,
    plane_normal: Array,
    plane_offset: Array,
    /,
) -> Array:
    if isinstance(plan, CertifiedAABBCCDPlan):
        left, right = _partition_slots(kind)
        if kind in _PLANE_KINDS:
            boxes = tuple(
                _outward_bounds(
                    start[index : index + 1], velocity[index : index + 1], lower, upper
                )
                for index in left
            )
            height = _rounded_plane_lower(
                jnp.stack(tuple(box[0] for box in boxes)),
                jnp.stack(tuple(box[1] for box in boxes)),
                plane_normal,
                plane_offset,
            )
            return _unit_normal(plane_normal) & (
                height > jnp.nextafter(separation, jnp.full_like(separation, jnp.inf))
            )
        left_index = jnp.asarray(left, dtype=jnp.int32)
        right_index = jnp.asarray(right, dtype=jnp.int32)
        first_min, first_max = _outward_bounds(
            start[left_index], velocity[left_index], lower, upper
        )
        second_min, second_max = _outward_bounds(
            start[right_index], velocity[right_index], lower, upper
        )
        distance = _rounded_aabb_distance_squared(
            first_min, first_max, second_min, second_max
        )
        threshold = separation * separation
        threshold = jnp.nextafter(threshold, jnp.full_like(threshold, jnp.inf))
        return distance > threshold
    left, right = _partition_slots(kind)
    speeds = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
    left_speed = speeds[jnp.asarray(left, dtype=jnp.int32)].max()
    right_speed = (
        speeds[jnp.asarray(right, dtype=jnp.int32)].max()
        if right
        else jnp.zeros((), dtype=speeds.dtype)
    )
    midpoint = (lower + upper) * 0.5
    distance = _stencil_distance(
        kind, start + midpoint * velocity, plane_normal, plane_offset
    )
    return (
        distance
        - (upper - lower) * 0.5 * (left_speed + right_speed)
        - plan.numerical_error
        > separation
    )


type ContactTrajectoryBoundsAction = Callable[[Array, Array], tuple[Array, Array, Array]]


def _trajectory_enclosure(
    action: ContactTrajectoryBoundsAction,
    lower_time: Array,
    upper_time: Array,
    expected: tuple[int, ...],
    /,
) -> tuple[Array, Array, Array]:
    lower, upper, successful = action(lower_time, upper_time)
    lower = jnp.asarray(lower, dtype=lower_time.dtype)
    upper = jnp.asarray(upper, dtype=lower_time.dtype)
    successful = jnp.asarray(successful)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError("trajectory_bounds must return the complete scene vertex shape.")
    if successful.shape != () or successful.dtype != jnp.bool_:
        raise ValueError("trajectory_bounds must return one Boolean success scalar.")
    valid = (
        successful
        & jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(upper))
        & jnp.all(lower <= upper)
    )
    return lower, upper, valid


def _enclosure_interval_free(
    plan: CCDPlan,
    kind: ContactStencilKind,
    action: ContactTrajectoryBoundsAction,
    separation: Array,
    lower_time: Array,
    upper_time: Array,
    plane_normal: Array,
    plane_offset: Array,
    /,
) -> tuple[Array, Array]:
    """Certify an interval from box enclosures.

    The lower distance bound is the larger of the box separation and the
    box-center distance minus the box half-diagonals: the latter certifies
    oblique separated features whose axis-aligned boxes overlap.
    """
    lower, upper, valid = action(lower_time, upper_time)
    box_bound = _box_lower_distance(kind, lower, upper, plane_normal, plane_offset)
    centers = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    radii = jnp.sqrt(jnp.sum(half * half, axis=-1))
    radii = radii * (1.0 + 8.0 * jnp.finfo(radii.dtype).eps)
    ball_bound, ball_valid = _ball_lower_distance(
        kind, centers, radii, plane_normal, plane_offset, 0.0
    )
    ball_bound = ball_bound - 64.0 * jnp.finfo(radii.dtype).eps * (
        jnp.max(jnp.abs(centers)) + jnp.max(radii)
    )
    bound = jnp.maximum(box_bound, jnp.where(ball_valid, ball_bound, -jnp.inf))
    if kind in _PLANE_KINDS:
        bound = jnp.where(_unit_normal(plane_normal), bound, -jnp.inf)
    threshold = separation
    if isinstance(plan, InclusionCCDPlan):
        threshold = threshold + plan.numerical_error
    threshold = jnp.nextafter(threshold, jnp.full_like(threshold, jnp.inf))
    return valid & (bound > threshold), valid


type _IntervalCarry = tuple[Array, Array, Array, Array, Array, Array, Array]


def _candidate_toi(
    plan: CCDPlan,
    kind: ContactStencilKind,
    start: Array,
    end: Array,
    minimum_separation: Array,
    plane_normal: Array,
    plane_offset: Array,
    /,
    *,
    trajectory_bounds: ContactTrajectoryBoundsAction | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    dtype = start.dtype
    zero = jnp.zeros((), dtype=dtype)
    infinity = jnp.full((), jnp.inf, dtype=dtype)
    initial_distance = _stencil_distance(kind, start, plane_normal, plane_offset)
    if isinstance(plan, CertifiedAABBCCDPlan):
        initial_distance = jnp.nextafter(initial_distance, -infinity)
        threshold = jnp.nextafter(minimum_separation, infinity)
    else:
        threshold = minimum_separation + plan.numerical_error
    initial_invalid = initial_distance <= threshold
    velocity = end - start
    capacity = max(2, ceil(-log2(plan.time_tolerance)) + 3)
    lower_stack = jnp.zeros((capacity,), dtype=dtype)
    upper_stack = jnp.zeros((capacity,), dtype=dtype).at[0].set(1.0)
    carry = (
        lower_stack,
        upper_stack,
        jnp.where(initial_invalid, 0, 1).astype(jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(initial_invalid, zero, infinity),
        jnp.asarray(False, dtype=jnp.bool_),
        jnp.asarray(False, dtype=jnp.bool_),
    )

    def condition(values: _IntervalCarry) -> Array:
        return (
            (values[2] > 0)
            & (values[3] < plan.maximum_iterations)
            & ~values[5]
            & ~values[6]
        )

    def body(values: _IntervalCarry) -> _IntervalCarry:
        lowers, uppers, size, iterations, toi, exhausted, bound_failed = values
        index = size - 1
        lower, upper = lowers[index], uppers[index]
        bounds_valid = jnp.asarray(True, dtype=jnp.bool_)
        if trajectory_bounds is None:
            free = _interval_free(
                plan,
                kind,
                start,
                velocity,
                minimum_separation,
                lower,
                upper,
                plane_normal,
                plane_offset,
            )
        else:
            free, bounds_valid = _enclosure_interval_free(
                plan,
                kind,
                trajectory_bounds,
                minimum_separation,
                lower,
                upper,
                plane_normal,
                plane_offset,
            )
        midpoint = (lower + upper) * 0.5
        terminal = upper - lower <= plan.time_tolerance
        split = ~free & ~terminal
        room = (index + 1 < capacity) & (midpoint > lower) & (midpoint < upper)
        unresolved = ~free & terminal
        lowers = (
            lowers.at[index]
            .set(midpoint)
            .at[jnp.minimum(index + 1, capacity - 1)]
            .set(lower)
        )
        uppers = (
            uppers.at[index]
            .set(upper)
            .at[jnp.minimum(index + 1, capacity - 1)]
            .set(midpoint)
        )
        next_size = jnp.where(split & room, index + 2, index)
        return (
            lowers,
            uppers,
            jnp.where(unresolved | ~bounds_valid, 0, next_size),
            iterations + 1,
            jnp.where(~bounds_valid, zero, jnp.where(unresolved, lower, toi)),
            exhausted | (split & ~room),
            bound_failed | ~bounds_valid,
        )

    result = jax.lax.while_loop(condition, body, carry)
    exhausted = result[5] | (result[2] > 0)
    toi = jnp.where(exhausted, zero, result[4])
    count = jnp.where(initial_invalid, 1, result[3]).astype(jnp.int32)
    return toi, count, initial_invalid, exhausted, result[6]


type _BatchCarry = tuple[Array, Array, Array, Array, Array, Array, Array]


def _scene_planes(
    scene: PreparedCollisionScene | ContactParticipantScene, dtype: jnp.dtype, /
) -> tuple[tuple[Array, Array, int], Array]:
    """Return ``((normals, offsets, first plane feature), geometry valid)``.

    Validity is the search owner's scene-wide route-geometry gate: finite
    nonnegative extents, finite offsets and finite unit normals. Empty plane
    sets are padded with one unused row so static non-plane routes can index.
    """
    _, normals, offsets, valid = scene_route_geometry(scene, dtype)
    if normals.shape[0] == 0:
        normals = jnp.zeros((1, scene.ambient_dimension), dtype=dtype)
        offsets = jnp.zeros((1,), dtype=dtype)
    return (normals, offsets, scene.feature_count - scene.plane_count), valid


def _row_plane(
    batch: ContactStencilBatch,
    slot: Array,
    normals: Array,
    offsets: Array,
    first_plane: int,
    /,
) -> tuple[Array, Array]:
    if batch.kind not in _PLANE_KINDS:
        return normals[0], offsets[0]
    rank = jnp.clip(
        batch.right_feature_indices[slot] - first_plane, 0, offsets.shape[0] - 1
    )
    return normals[rank], offsets[rank]


def _row_indices(batch: ContactStencilBatch, slot: Array, /) -> Array:
    return jnp.maximum(batch.vertex_indices[slot, : _arity(batch.kind)], 0)


def _batch_step_limit(
    plan: CCDPlan,
    batch: ContactStencilBatch,
    start: Array,
    end: Array,
    planes: tuple[Array, Array, int],
    /,
    *,
    trajectory_bounds: ContactTrajectoryBoundsAction | None = None,
) -> _BatchCarry:
    initial = (
        jnp.full((), jnp.inf, dtype=start.dtype),
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False, dtype=jnp.bool_),
        jnp.asarray(False, dtype=jnp.bool_),
    )
    normals, offsets, first_plane = planes

    def body(slot: Array, carry: _BatchCarry) -> _BatchCarry:
        def query(values: _BatchCarry) -> _BatchCarry:
            toi, key, queries, intervals, violations, exhausted, bound_failed = values
            indices = _row_indices(batch, slot)
            plane_normal, plane_offset = _row_plane(
                batch, slot, normals, offsets, first_plane
            )

            def route_bounds(
                lower_time: Array, upper_time: Array
            ) -> tuple[Array, Array, Array]:
                if trajectory_bounds is None:
                    raise ValueError("A route enclosure requires trajectory_bounds.")
                lower, upper, valid = _trajectory_enclosure(
                    trajectory_bounds, lower_time, upper_time, start.shape
                )
                return lower[indices], upper[indices], valid

            limit, count, invalid, work_limit, bounds_invalid = _candidate_toi(
                plan,
                batch.kind,
                start[indices],
                end[indices],
                batch.minimum_separation[slot].astype(start.dtype),
                plane_normal,
                plane_offset,
                trajectory_bounds=None if trajectory_bounds is None else route_bounds,
            )
            return (
                jnp.minimum(toi, limit),
                jnp.where(limit < toi, batch.route_keys[slot], key),
                queries + 1,
                intervals + count,
                violations + invalid.astype(jnp.int32),
                exhausted | work_limit,
                bound_failed | bounds_invalid,
            )

        # One certificate per primitive pair: secondary support witnesses share
        # the pair's minimum core distance.
        primary = batch.witness_slots[slot] == int(ContactWitnessSlot.FIRST)
        return jax.lax.cond(
            batch.valid[slot] & primary, query, lambda values: values, carry
        )

    return jax.lax.fori_loop(0, batch.valid.shape[0], body, initial)


def _bounds_action(
    trajectory_bounds: ContactTrajectoryBoundsAction | AbstractContactTrajectory | None,
    /,
) -> ContactTrajectoryBoundsAction | None:
    if not isinstance(trajectory_bounds, AbstractContactTrajectory):
        return trajectory_bounds
    trajectory = trajectory_bounds

    def action(lower_time: Array, upper_time: Array, /) -> tuple[Array, Array, Array]:
        enclosure = trajectory.enclosure(lower_time, upper_time)
        return enclosure.lower, enclosure.upper, enclosure.successful

    return action


@checked
def collision_free_step_limit(
    plan: CCDPlan,
    scene: PreparedCollisionScene | ContactParticipantScene,
    epoch: ContactCandidateEpoch,
    start_positions: ArrayLike,
    end_positions: ArrayLike,
    /,
    *,
    trajectory_bounds: (
        ContactTrajectoryBoundsAction | AbstractContactTrajectory | None
    ) = None,
) -> ContactSafetyEvidence:
    """Certify strict separation of a complete trajectory with bounded JAX work.

    The default certifies linear sample trajectories. Nonlinear maps require an
    explicit ``trajectory_bounds`` provider (a callback or a declared
    trajectory) enclosing the whole scene over each queried interval; endpoints
    alone are never a nonlinear path certificate. Interval predicates use the
    larger of box separation and box-center distance minus box half-diagonals.
    Initially touching states are refused: this is the strict-separation
    contract of barrier consumers. Hard-contact resting paths use
    :func:`certify_admissible_path`. External providers establish
    ENCLOSURE_CONSERVATIVE, not ROUNDING_CERTIFIED.
    """
    dtype = jnp.dtype(jnp.float64)
    start = jnp.asarray(start_positions, dtype=dtype)
    end = jnp.asarray(end_positions, dtype=dtype)
    expected = (scene.vertex_count, scene.ambient_dimension)
    if start.shape != expected or end.shape != expected:
        raise ValueError("CCD positions must match the scene vertex shape.")
    action = _bounds_action(trajectory_bounds)
    planes, geometry_valid = _scene_planes(scene, dtype)
    # Foreign catalogue/topology/routing raises; changed extents or planes do
    # not have this epoch's coverage.
    searched = epoch.successful & epoch.binds(scene)
    finite = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
    minimum_toi = jnp.full((), jnp.inf, dtype=start.dtype)
    limiting_key = jnp.asarray(0, dtype=jnp.int64)
    queries = jnp.asarray(0, dtype=jnp.int32)
    intervals = jnp.asarray(0, dtype=jnp.int32)
    violations = jnp.asarray(0, dtype=jnp.int32)
    exhausted = jnp.asarray(False, dtype=jnp.bool_)
    bounds_failed = jnp.asarray(False, dtype=jnp.bool_)
    if action is not None:
        lower_enclosure, upper_enclosure, bounds_valid = _trajectory_enclosure(
            action,
            jnp.zeros((), dtype=start.dtype),
            jnp.ones((), dtype=start.dtype),
            expected,
        )
        bounds_valid = bounds_valid & (
            jnp.all(start >= lower_enclosure)
            & jnp.all(start <= upper_enclosure)
            & jnp.all(end >= lower_enclosure)
            & jnp.all(end <= upper_enclosure)
        )
        bounds_failed = ~bounds_valid | (
            plan.required_guarantee > ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
        )
    # Heterogeneous stencil kinds are static; slots use bounded lax work.
    for batch in epoch.active_batches:
        empty = (
            minimum_toi,
            limiting_key,
            jnp.zeros_like(queries),
            jnp.zeros_like(intervals),
            jnp.zeros_like(violations),
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(False, dtype=jnp.bool_),
        )
        toi, key, count, work, invalid, work_limit, bounds_invalid = jax.lax.cond(
            finite & geometry_valid & searched & ~bounds_failed,
            lambda _, batch=batch: _batch_step_limit(
                plan, batch, start, end, planes, trajectory_bounds=action
            ),
            lambda _: empty,
            None,
        )
        limiting_key = jnp.where(toi < minimum_toi, key, limiting_key)
        minimum_toi = jnp.minimum(minimum_toi, toi)
        queries += count
        intervals += work
        violations += invalid
        exhausted |= work_limit
        bounds_failed |= bounds_invalid
    status = jnp.where(
        ~finite,
        int(CCDStatus.NONFINITE_INPUT),
        jnp.where(
            ~geometry_valid,
            int(CCDStatus.INVALID_GEOMETRY),
            jnp.where(
                ~searched,
                int(CCDStatus.SEARCH_INCOMPLETE),
                jnp.where(
                    bounds_failed,
                    int(CCDStatus.TRAJECTORY_BOUND_FAILURE),
                    jnp.where(
                        violations > 0,
                        int(CCDStatus.INITIAL_DISTANCE_VIOLATION),
                        jnp.where(
                            exhausted,
                            int(CCDStatus.WORK_LIMIT),
                            int(CCDStatus.SUCCESS),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    reported = jnp.where(
        jnp.isfinite(minimum_toi), minimum_toi, jnp.ones_like(minimum_toi)
    )
    step = jnp.where(
        status == int(CCDStatus.SUCCESS),
        jnp.where(
            jnp.isfinite(minimum_toi),
            plan.conservative_rescaling * minimum_toi,
            jnp.ones_like(minimum_toi),
        ),
        jnp.zeros_like(minimum_toi),
    )
    level = (
        ContactGuaranteeLevel.ROUNDING_CERTIFIED
        if isinstance(plan, CertifiedAABBCCDPlan) and action is None
        else ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
    )
    guarantee = ContactGuaranteeEvidence(
        level,
        required_level=plan.required_guarantee,
        finite=finite,
        work_complete=finite & geometry_valid & searched & ~exhausted & ~bounds_failed,
        failure_code=status,
        margin=step,
        backend_id=plan.plan_id,
    )
    return ContactSafetyEvidence(
        step,
        reported,
        limiting_key,
        queries,
        intervals,
        violations,
        status,
        finite,
        jnp.asarray(True, dtype=jnp.bool_),
        guarantee,
        epoch.epoch_id,
        plan.plan_id,
    )


class AdmissiblePathStatus(IntEnum):
    """Outcome of hard-contact path admissibility certification."""

    ADMISSIBLE = 0
    INITIALLY_INADMISSIBLE = 1
    CERTIFIED_PREFIX = 2
    UNRESOLVED_INTERVAL = 3
    INVALID_GEOMETRY = 4
    WORK_LIMIT = 5
    SEARCH_INCOMPLETE = 6
    COVERAGE_MISMATCH = 7
    NONFINITE_INPUT = 8


_RUNNING = -1


class AdmissiblePathPlan(StrictModule, NonTrainableState):
    """Declared gap floor and bounded work of hard-contact path certification.

    ``gap_floor`` is the geometric floor ``epsilon_gap`` (length units): the
    certificate proves ``g(s) >= -gap_floor`` for every candidate pair and every
    ``s`` in ``[0, 1]``. It is a fixed geometric tolerance, not an accumulated
    displacement allowance. ``maximum_intervals`` bounds the shared interval
    subdivision work; ``time_tolerance`` is the smallest interval width.
    """

    gap_floor: float = eqx.field(static=True)
    time_tolerance: float = eqx.field(static=True)
    maximum_intervals: int = eqx.field(static=True)
    distance_tolerance: float = eqx.field(static=True)
    required_guarantee: ContactGuaranteeLevel = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gap_floor: float,
        /,
        *,
        time_tolerance: float = 1.0e-8,
        maximum_intervals: int = 4096,
        distance_tolerance: float = 1.0e-12,
        required_guarantee: ContactGuaranteeLevel = (
            ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
        ),
    ) -> None:
        floor = float(gap_floor)
        tolerance = float(time_tolerance)
        intervals = int(maximum_intervals)
        distance = float(distance_tolerance)
        required = ContactGuaranteeLevel(required_guarantee)
        if not isfinite(floor) or floor < 0.0:
            raise ValueError("gap_floor must be finite and nonnegative.")
        if not isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("time_tolerance must lie strictly between zero and one.")
        if intervals <= 0:
            raise ValueError("maximum_intervals must be positive.")
        if not isfinite(distance) or distance < 0.0:
            raise ValueError("distance_tolerance must be finite and nonnegative.")
        if required > ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE:
            raise ValueError(
                "Admissible-path certification cannot exceed ENCLOSURE_CONSERVATIVE."
            )
        self.gap_floor = floor
        self.time_tolerance = tolerance
        self.maximum_intervals = intervals
        self.distance_tolerance = distance
        self.required_guarantee = required
        self.plan_id = canonical_fingerprint(
            {
                "kind": "admissible-path-plan",
                "gap_floor": floor.hex(),
                "time_tolerance": tolerance.hex(),
                "maximum_intervals": intervals,
                "distance_tolerance": distance.hex(),
                "required_guarantee": int(required),
            }
        )


class AdmissiblePathEvidence(StrictModule):
    """Hard-contact path admissibility of one declared trajectory.

    ``certified_prefix`` is the normalized time ``t`` such that ``[0, t]`` is
    certified; a prefix is never a successful full step. ``minimum_lower_gap`` is
    the smallest certified interval lower gap bound, ``minimum_initial_gap`` the
    smallest candidate gap at ``s = 0``.
    """

    status: Array
    certified_prefix: Array
    minimum_initial_gap: Array
    minimum_lower_gap: Array
    limiting_route_key: Array
    interval_count: Array
    finite: Array
    guarantee: ContactGuaranteeEvidence
    trajectory_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            self.status == int(AdmissiblePathStatus.ADMISSIBLE)
        ) & self.guarantee.successful


type _RowReduction = tuple[Array, Array, Array, Array]


def _row_gaps(
    plan: AdmissiblePathPlan,
    batch: ContactStencilBatch,
    lower: Array,
    upper: Array,
    positions: Array,
    displacement: Array,
    radius: Array,
    planes: tuple[Array, Array, int],
    /,
) -> _RowReduction:
    """Return ``(lower_gap, witness_gap, primitive_valid, key)`` per row.

    ``lower_gap`` is a directed lower bound over the interval; ``witness_gap``
    is the midpoint gap between actual witness points (an upper estimate used
    only to prove a violation).
    """
    normals, offsets, first_plane = planes
    arity = _arity(batch.kind)
    indices = jnp.maximum(batch.vertex_indices[:, :arity], 0)
    if batch.kind in _PLANE_KINDS:
        rank = jnp.clip(
            batch.right_feature_indices - first_plane, 0, offsets.shape[0] - 1
        )
        row_normals = normals[rank]
        row_offsets = offsets[rank]
    else:
        row_normals = jnp.broadcast_to(normals[0], (indices.shape[0], normals.shape[1]))
        row_offsets = jnp.broadcast_to(offsets[0], (indices.shape[0],))
    left_radius = radius[jnp.maximum(batch.left_feature_indices, 0)]
    right_radius = radius[jnp.maximum(batch.right_feature_indices, 0)]
    negative = jnp.full((), -jnp.inf, dtype=positions.dtype)

    def row(
        vertex_rows: Array, normal: Array, offset: Array
    ) -> tuple[Array, Array, Array]:
        _, witness, valid = _core_distance(
            batch.kind, positions[vertex_rows], normal, offset, plan.distance_tolerance
        )
        ball, _ = _ball_lower_distance(
            batch.kind,
            positions[vertex_rows],
            displacement[vertex_rows],
            normal,
            offset,
            plan.distance_tolerance,
        )
        box = _box_lower_distance(
            batch.kind, lower[vertex_rows], upper[vertex_rows], normal, offset
        )
        return jnp.maximum(ball, box), witness, valid

    core_lower, core_witness, valid = jax.vmap(row)(indices, row_normals, row_offsets)
    primary = batch.valid & (batch.witness_slots == int(ContactWitnessSlot.FIRST))
    lower_gap = jnp.nextafter(
        jnp.nextafter(core_lower - left_radius, negative) - right_radius, negative
    )
    lower_gap = jnp.where(primary, lower_gap, jnp.inf)
    witness_gap = jnp.where(primary, core_witness - left_radius - right_radius, jnp.inf)
    return lower_gap, witness_gap, ~primary | valid, batch.route_keys


@checked
def certify_admissible_path(
    plan: AdmissiblePathPlan,
    scene: ContactParticipantScene,
    epoch: ContactCandidateEpoch,
    trajectory: AbstractContactTrajectory,
    /,
) -> AdmissiblePathEvidence:
    """Prove the hard-contact gap floor over a declared trajectory.

    Every candidate primitive pair of ``epoch`` must satisfy
    ``g(s) >= -gap_floor`` for all ``s`` in ``[0, 1]``, including legal initial
    resting or sliding contact. Each interval lower bound is the larger of the
    directed box separation (or plane support) of the trajectory enclosure and
    a rigorous separating-axis lower bound of the midpoint core distance minus
    both features' Hausdorff displacement bounds, minus physical radii; an
    ill-conditioned closest-point solve only weakens it. Violations are proved
    with actual witness points. Intervals are processed in time order, so an
    unresolved or violated interval leaves a certified prefix up to its start.
    Pairs outside ``epoch`` are covered only when the epoch binds this scene's
    catalogue, topology, routing policy, contact extents and planes, and its
    searched enclosure belongs to this trajectory, revision and interval and
    covers the trajectory's own full-path enclosure. Foreign catalogue,
    topology or routing raises; other mismatches refuse as coverage mismatch.
    Invalid scene route geometry (for example a non-unit plane normal) refuses
    as invalid geometry.
    """
    if trajectory.guarantee_level < plan.required_guarantee:
        raise ValueError("Trajectory guarantee is weaker than the plan requires.")
    dtype = jnp.dtype(jnp.float64)
    planes, geometry_valid = _scene_planes(scene, dtype)
    radius = jnp.asarray(scene.feature_physical_radius, dtype=dtype)
    floor = jnp.asarray(plan.gap_floor, dtype=dtype)
    batches = epoch.active_batches
    full = trajectory.enclosure(0.0, 1.0)
    expected = (scene.vertex_count, scene.ambient_dimension)
    if full.lower.shape != expected:
        raise ValueError("Trajectory vertices must match the contact scene.")
    same_trajectory = epoch.trajectory_id == trajectory.trajectory_id
    coverage = (
        jnp.asarray(same_trajectory)
        & epoch.binds(scene)
        & (epoch.revision == trajectory.revision)
        & (epoch.lower_time <= 0.0)
        & (epoch.upper_time >= 1.0)
        & full.successful
        & epoch.covers(full.lower, full.upper)
    )
    start = jnp.asarray(trajectory.evaluate(0.0), dtype=dtype)
    zeros = jnp.zeros(scene.vertex_count, dtype=dtype)
    finite = jnp.all(jnp.isfinite(start)) & full.finite
    initial_lower = jnp.full((), jnp.inf, dtype=dtype)
    initial_gap = jnp.full((), jnp.inf, dtype=dtype)
    initial_valid = jnp.asarray(True)
    for batch in batches:
        lower_gaps, gaps, valid, _ = _row_gaps(
            plan, batch, start, start, start, zeros, radius, planes
        )
        initial_lower = jnp.minimum(initial_lower, jnp.min(lower_gaps, initial=jnp.inf))
        initial_gap = jnp.minimum(initial_gap, jnp.min(gaps, initial=jnp.inf))
        initial_valid = initial_valid & jnp.all(valid)
    preliminary = jnp.where(
        ~finite,
        int(AdmissiblePathStatus.NONFINITE_INPUT),
        jnp.where(
            ~geometry_valid,
            int(AdmissiblePathStatus.INVALID_GEOMETRY),
            jnp.where(
                ~epoch.successful,
                int(AdmissiblePathStatus.SEARCH_INCOMPLETE),
                jnp.where(
                    ~coverage,
                    int(AdmissiblePathStatus.COVERAGE_MISMATCH),
                    jnp.where(
                        ~initial_valid,
                        int(AdmissiblePathStatus.INVALID_GEOMETRY),
                        jnp.where(
                            initial_gap < -floor,
                            int(AdmissiblePathStatus.INITIALLY_INADMISSIBLE),
                            jnp.where(
                                initial_lower < -floor,
                                int(AdmissiblePathStatus.UNRESOLVED_INTERVAL),
                                _RUNNING,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)

    capacity = max(2, ceil(-log2(plan.time_tolerance)) + 3)
    lowers = jnp.zeros((capacity,), dtype=dtype)
    uppers = jnp.zeros((capacity,), dtype=dtype).at[0].set(1.0)
    carry = (
        lowers,
        uppers,
        jnp.where(preliminary == _RUNNING, 1, 0).astype(jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        preliminary,
        jnp.zeros((), dtype=dtype),
        jnp.full((), jnp.inf, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int64),
    )
    type Carry = tuple[Array, Array, Array, Array, Array, Array, Array, Array]

    def condition(values: Carry) -> Array:
        return (
            (values[2] > 0)
            & (values[4] == _RUNNING)
            & (values[3] < plan.maximum_intervals)
        )

    def body(values: Carry) -> Carry:
        lowers, uppers, size, count, status, prefix, minimum, key = values
        index = size - 1
        lower_time, upper_time = lowers[index], uppers[index]
        enclosure, midpoint = trajectory.interval_bounds(lower_time, upper_time)
        lower_gap = jnp.full((), jnp.inf, dtype=dtype)
        midpoint_gap = jnp.full((), jnp.inf, dtype=dtype)
        primitive_valid = jnp.asarray(True)
        limiting = key
        for batch in batches:
            gaps, centers, valid, keys = _row_gaps(
                plan,
                batch,
                jnp.asarray(enclosure.lower, dtype=dtype),
                jnp.asarray(enclosure.upper, dtype=dtype),
                jnp.asarray(midpoint.positions, dtype=dtype),
                jnp.asarray(midpoint.displacement, dtype=dtype),
                radius,
                planes,
            )
            batch_minimum = jnp.min(gaps, initial=jnp.inf)
            if batch.capacity:
                limiting = jnp.where(
                    batch_minimum < lower_gap, keys[jnp.argmin(gaps)], limiting
                )
            lower_gap = jnp.minimum(lower_gap, batch_minimum)
            midpoint_gap = jnp.minimum(midpoint_gap, jnp.min(centers, initial=jnp.inf))
            primitive_valid = primitive_valid & jnp.all(valid)
        bounds_valid = enclosure.successful & midpoint.successful
        invalid = ~bounds_valid | ~primitive_valid
        certified = ~invalid & (lower_gap >= -floor)
        violated = ~invalid & (midpoint_gap < -floor)
        terminal = upper_time - lower_time <= plan.time_tolerance
        middle = 0.5 * (lower_time + upper_time)
        split = ~certified & ~violated & ~invalid & ~terminal
        room = (index + 1 < capacity) & (middle > lower_time) & (middle < upper_time)
        next_status = jnp.where(
            invalid,
            int(AdmissiblePathStatus.INVALID_GEOMETRY),
            jnp.where(
                violated,
                int(AdmissiblePathStatus.CERTIFIED_PREFIX),
                jnp.where(
                    ~certified & terminal,
                    int(AdmissiblePathStatus.UNRESOLVED_INTERVAL),
                    jnp.where(
                        split & ~room, int(AdmissiblePathStatus.WORK_LIMIT), _RUNNING
                    ),
                ),
            ),
        ).astype(jnp.int32)
        lowers = (
            lowers.at[index]
            .set(middle)
            .at[jnp.minimum(index + 1, capacity - 1)]
            .set(lower_time)
        )
        uppers = (
            uppers.at[index]
            .set(upper_time)
            .at[jnp.minimum(index + 1, capacity - 1)]
            .set(middle)
        )
        return (
            lowers,
            uppers,
            jnp.where(split & room, index + 2, index),
            count + 1,
            next_status,
            jnp.where(certified, upper_time, prefix),
            jnp.where(certified, jnp.minimum(minimum, lower_gap), minimum),
            jnp.where(certified, key, limiting),
        )

    (_, _, size, count, status, prefix, minimum, key) = jax.lax.while_loop(
        condition, body, carry
    )
    status = jnp.where(
        status == _RUNNING,
        jnp.where(
            size > 0,
            int(AdmissiblePathStatus.WORK_LIMIT),
            int(AdmissiblePathStatus.ADMISSIBLE),
        ),
        status,
    ).astype(jnp.int32)
    admissible = status == int(AdmissiblePathStatus.ADMISSIBLE)
    prefix = jnp.where(admissible, jnp.ones_like(prefix), prefix)
    guarantee = ContactGuaranteeEvidence(
        trajectory.guarantee_level,
        required_level=plan.required_guarantee,
        finite=finite,
        work_complete=finite
        & geometry_valid
        & epoch.successful
        & coverage
        & (status != int(AdmissiblePathStatus.WORK_LIMIT)),
        failure_code=status,
        margin=jnp.where(admissible, minimum + floor, jnp.zeros_like(minimum)),
        backend_id=plan.plan_id,
    )
    return AdmissiblePathEvidence(
        status,
        prefix,
        initial_gap,
        minimum,
        key,
        count,
        finite,
        guarantee,
        trajectory.trajectory_id,
        epoch.epoch_id,
        plan.plan_id,
    )


__all__ = [
    "AdmissiblePathEvidence",
    "AdmissiblePathPlan",
    "AdmissiblePathStatus",
    "CCDStatus",
    "ContactSafetyEvidence",
    "InclusionCCDPlan",
    "CertifiedAABBCCDPlan",
    "ContactGuaranteeEvidence",
    "ContactGuaranteeLevel",
    "certify_admissible_path",
    "collision_free_step_limit",
]
