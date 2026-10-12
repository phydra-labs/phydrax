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
from ._search import ContactCandidateEpoch
from ._stencils import ContactStencilBatch, ContactStencilKind
from ._surface import PreparedCollisionScene


class CCDStatus(IntEnum):
    SUCCESS = 0
    INITIAL_DISTANCE_VIOLATION = 1
    SEARCH_INCOMPLETE = 2
    WORK_LIMIT = 3
    NONFINITE_INPUT = 4
    TRAJECTORY_BOUND_FAILURE = 5


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


def _stencil_distance(kind: ContactStencilKind, vertices: Array, /) -> Array:
    match kind:
        case ContactStencilKind.VERTEX_VERTEX:
            result = point_point_distance(vertices[0], vertices[1], tolerance=0.0)
        case ContactStencilKind.EDGE_VERTEX:
            result = point_edge_distance(
                vertices[0], vertices[1], vertices[2], tolerance=0.0
            )
        case ContactStencilKind.FACE_VERTEX:
            result = point_triangle_distance(
                vertices[0], vertices[1], vertices[2], vertices[3], tolerance=0.0
            )
        case ContactStencilKind.EDGE_EDGE:
            result = edge_edge_distance(
                vertices[0], vertices[1], vertices[2], vertices[3], tolerance=0.0
            )
        case _:
            raise TypeError("Unsupported CCD stencil kind.")
    # Degenerate primitive kernels may mask unsupported features with a large
    # finite sentinel; that value is not a separation certificate.
    squared = jnp.where(result.nondegenerate, result.squared_distance, 0.0)
    return jnp.sqrt(squared)


def _primitive_partition(
    kind: ContactStencilKind, vertices: Array, /
) -> tuple[Array, Array]:
    match kind:
        case ContactStencilKind.VERTEX_VERTEX:
            return vertices[:1], vertices[1:2]
        case ContactStencilKind.EDGE_VERTEX | ContactStencilKind.FACE_VERTEX:
            return vertices[:1], vertices[1:]
        case ContactStencilKind.EDGE_EDGE:
            return vertices[:2], vertices[2:]
        case _:
            raise TypeError("Unsupported CCD stencil kind.")


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


def _interval_free(
    plan: CCDPlan,
    kind: ContactStencilKind,
    start: Array,
    velocity: Array,
    separation: Array,
    lower: Array,
    upper: Array,
    /,
) -> Array:
    if isinstance(plan, CertifiedAABBCCDPlan):
        first, second = _primitive_partition(kind, start)
        first_velocity, second_velocity = _primitive_partition(kind, velocity)
        first_min, first_max = _outward_bounds(first, first_velocity, lower, upper)
        second_min, second_max = _outward_bounds(second, second_velocity, lower, upper)
        distance = _rounded_aabb_distance_squared(
            first_min, first_max, second_min, second_max
        )
        threshold = separation * separation
        threshold = jnp.nextafter(threshold, jnp.full_like(threshold, jnp.inf))
        return distance > threshold
    first_velocity, second_velocity = _primitive_partition(kind, velocity)
    speed = (
        jnp.sqrt(jnp.sum(first_velocity * first_velocity, axis=-1)).max()
        + jnp.sqrt(jnp.sum(second_velocity * second_velocity, axis=-1)).max()
    )
    midpoint = (lower + upper) * 0.5
    distance = _stencil_distance(kind, start + midpoint * velocity)
    return distance - (upper - lower) * 0.5 * speed - plan.numerical_error > separation


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
    /,
) -> tuple[Array, Array]:
    lower, upper, valid = action(lower_time, upper_time)
    first_lower, second_lower = _primitive_partition(kind, lower)
    first_upper, second_upper = _primitive_partition(kind, upper)
    distance = _rounded_aabb_distance_squared(
        first_lower.min(axis=0),
        first_upper.max(axis=0),
        second_lower.min(axis=0),
        second_upper.max(axis=0),
    )
    threshold = separation
    if isinstance(plan, InclusionCCDPlan):
        threshold = threshold + plan.numerical_error
    squared_threshold = threshold * threshold
    squared_threshold = jnp.nextafter(
        squared_threshold, jnp.full_like(squared_threshold, jnp.inf)
    )
    return valid & (distance > squared_threshold), valid


type _IntervalCarry = tuple[Array, Array, Array, Array, Array, Array, Array]


def _candidate_toi(
    plan: CCDPlan,
    kind: ContactStencilKind,
    start: Array,
    end: Array,
    minimum_separation: Array,
    /,
    *,
    trajectory_bounds: ContactTrajectoryBoundsAction | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    dtype = start.dtype
    zero = jnp.zeros((), dtype=dtype)
    infinity = jnp.full((), jnp.inf, dtype=dtype)
    initial_distance = _stencil_distance(kind, start)
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
                plan, kind, start, velocity, minimum_separation, lower, upper
            )
        else:
            free, bounds_valid = _enclosure_interval_free(
                plan, kind, trajectory_bounds, minimum_separation, lower, upper
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


def _batch_step_limit(
    plan: CCDPlan,
    batch: ContactStencilBatch,
    start: Array,
    end: Array,
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
    arity = (
        2
        if batch.kind == ContactStencilKind.VERTEX_VERTEX
        else (3 if batch.kind == ContactStencilKind.EDGE_VERTEX else 4)
    )

    def body(slot: Array, carry: _BatchCarry) -> _BatchCarry:
        def query(values: _BatchCarry) -> _BatchCarry:
            toi, key, queries, intervals, violations, exhausted, bound_failed = values
            indices = batch.vertex_indices[slot, :arity]

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

        return jax.lax.cond(batch.valid[slot], query, lambda values: values, carry)

    return jax.lax.fori_loop(0, batch.valid.shape[0], body, initial)


@checked
def collision_free_step_limit(
    plan: CCDPlan,
    scene: PreparedCollisionScene | ContactParticipantScene,
    epoch: ContactCandidateEpoch,
    start_positions: ArrayLike,
    end_positions: ArrayLike,
    /,
    *,
    trajectory_bounds: ContactTrajectoryBoundsAction | None = None,
) -> ContactSafetyEvidence:
    """Certify a complete trajectory with bounded, fixed-capacity JAX work.

    The default certifies linear sample trajectories. Nonlinear maps require an
    explicit ``trajectory_bounds`` provider enclosing the whole scene over each
    queried interval; endpoints alone are never a nonlinear path certificate.
    External providers establish ENCLOSURE_CONSERVATIVE, not ROUNDING_CERTIFIED.
    Search and scene construction remain explicit host preparation operations.
    """
    dtype = jnp.float64
    start = jnp.asarray(start_positions, dtype=dtype)
    end = jnp.asarray(end_positions, dtype=dtype)
    expected = (scene.vertex_count, scene.ambient_dimension)
    if start.shape != expected or end.shape != expected:
        raise ValueError("CCD positions must match the scene vertex shape.")
    finite = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
    minimum_toi = jnp.full((), jnp.inf, dtype=start.dtype)
    limiting_key = jnp.asarray(0, dtype=jnp.int64)
    queries = jnp.asarray(0, dtype=jnp.int32)
    intervals = jnp.asarray(0, dtype=jnp.int32)
    violations = jnp.asarray(0, dtype=jnp.int32)
    exhausted = jnp.asarray(False, dtype=jnp.bool_)
    bounds_failed = jnp.asarray(False, dtype=jnp.bool_)
    if trajectory_bounds is not None:
        lower_enclosure, upper_enclosure, bounds_valid = _trajectory_enclosure(
            trajectory_bounds,
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
            finite & epoch.successful & ~bounds_failed,
            lambda _: _batch_step_limit(
                plan, batch, start, end, trajectory_bounds=trajectory_bounds
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
            ~epoch.successful,
            int(CCDStatus.SEARCH_INCOMPLETE),
            jnp.where(
                bounds_failed,
                int(CCDStatus.TRAJECTORY_BOUND_FAILURE),
                jnp.where(
                    violations > 0,
                    int(CCDStatus.INITIAL_DISTANCE_VIOLATION),
                    jnp.where(
                        exhausted, int(CCDStatus.WORK_LIMIT), int(CCDStatus.SUCCESS)
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
        if isinstance(plan, CertifiedAABBCCDPlan) and trajectory_bounds is None
        else ContactGuaranteeLevel.ENCLOSURE_CONSERVATIVE
    )
    guarantee = ContactGuaranteeEvidence(
        level,
        required_level=plan.required_guarantee,
        finite=finite,
        work_complete=finite & epoch.successful & ~exhausted & ~bounds_failed,
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


__all__ = [
    "CCDStatus",
    "ContactSafetyEvidence",
    "InclusionCCDPlan",
    "CertifiedAABBCCDPlan",
    "ContactGuaranteeEvidence",
    "ContactGuaranteeLevel",
    "collision_free_step_limit",
]
