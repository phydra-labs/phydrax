#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Near-zone Liénard–Wiechert fields of sampled point-charge trajectories.

For an observer event ``(t, x)`` and one lane the retarded time ``t_r`` solves

    g(t_r) = t − t_r − |x − r(t_r)| / c = 0,

which is strictly decreasing for subluminal motion (``g' = −κ``), so the root
is unique. With ``R = |x − r(t_r)|``, ``n = (x − r(t_r)) / R`` and
``κ = 1 − n·β`` the field is

    E = q / (4 π ε₀) [ (n − β)(1 − β²) / (κ³ R²) + n × ((n − β) × β̇) / (c κ³ R) ],
    B = n × E / c,

the first term being the velocity field and the second the acceleration
(radiation) field. ``κ`` is evaluated as ``(1/γ² + |n × β|²) / (1 + n·β)``,
which has no cancellation at large ``γ``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._custom_root import custom_root
from .._fingerprint import canonical_fingerprint
from .._interpolation import cubic_hermite_segment, linear_segment
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_finite_float, positive_integer
from ..ein import contract
from ..nonlinear import (
    NonlinearStatus,
    NonlinearTermination,
    scalar_root,
    ScalarRootProblem,
    TOMS748,
)
from ..typing import Bool, checked, Dim, Float64, Identifier, Int32, parse, Scalar, Scope
from ._trajectory_radiation import _float64_array, _hermite_basis, ChargedTrajectory


LienardWiechertHistory: TypeAlias = Literal["refuse", "inertial-extrapolation"]
LienardWiechertInterpolation: TypeAlias = Literal["hermite-cubic", "hermite-quintic"]

# The retarded residual is solved in units of the segment step to its
# rounding floor: the residual terms are O(1), so 16 ε bounds the accepted
# residual, and a bracket narrower than 4 ε in the segment coordinate is the
# root at machine resolution. The observer-time error of the field is then
# limited by the rounding of the inputs themselves, ε·max(t, R/c).
_ROOT_TOLERANCE = 16.0 * float(np.finfo(np.float64).eps)
_ROOT_RESOLUTION = 4.0 * float(np.finfo(np.float64).eps)
# A position interpolant whose speed disagrees with the proper-velocity
# interpolant by more than 1% of 1/γ² does not resolve the kinematics.
_LORENTZ_MISMATCH_LIMIT = 1.0e-2
_REAL_ITEMSIZE = 8
_STATUS_ITEMSIZE = 4
# Live float64 channels of one observer–particle pair: two Hermite end states,
# the bisection bounds, the bracketed root state, and the retarded kinematics.
_PAIR_CHANNELS = 96
# Per-sample lane channels: time, position, velocity, acceleration, proper
# velocity and its rate, 1/γ², and activity.
_LANE_CHANNELS = 18
# Per-observer accumulators: three field sums and seven evidence reductions.
_OBSERVER_CHANNELS = 16
# Per-observer outputs: four field vectors, the event, and nine evidence leaves.
_OBSERVER_OUTPUT_CHANNELS = 25


class LienardWiechertStatus(IntFlag):
    """Per observer–particle status bits; observer status is their union.

    ``RETARDED_BEFORE_WINDOW`` marks a retarded time before the first sample:
    unsupported under ``history="refuse"``, inertially extrapolated otherwise.
    ``RETARDED_AFTER_WINDOW``, ``ROOT_FAILED``, ``NONMONOTONE_TIME``,
    ``SUPERLUMINAL_SAMPLES``, and ``SUPERLUMINAL_INTERPOLANT`` are unsupported.
    ``EXCLUDED_CHARGE`` and ``INACTIVE_RETARDED`` pairs contribute no field and
    are accounted in ``excluded_charge`` and ``absent_charge``.
    """

    SUCCESS = 0
    NONFINITE = 1
    RETARDED_BEFORE_WINDOW = 2
    RETARDED_AFTER_WINDOW = 4
    EXCLUDED_CHARGE = 8
    INACTIVE_RETARDED = 16
    ROOT_FAILED = 32
    NONMONOTONE_TIME = 64
    SUPERLUMINAL_SAMPLES = 128
    SUPERLUMINAL_INTERPOLANT = 256
    UNRESOLVED_KINEMATICS = 512


class LienardWiechertResourceError(ValueError):
    """A Liénard–Wiechert evaluation exceeds its declared resource policy."""


class _ObserverDim(Dim, minimum=1):
    """Observer events."""


class _ParticleDim(Dim, minimum=1):
    """Particle lanes."""


class LienardWiechertResources(StrictModule, NonTrainableState):
    """Bounded execution policy for near-zone fields.

    Observer events execute in ``lax.map`` blocks of ``observer_chunk``; inside
    each block particle lanes execute in ``lax.scan`` chunks of
    ``particle_chunk``, so one step holds ``observer_chunk × particle_chunk``
    pairs. The retarded root takes at most ``maximum_root_steps`` bracketed
    iterations. Working set and outputs are estimated before execution and
    refused above ``maximum_working_bytes`` / ``maximum_output_bytes``.
    """

    __strict_contract__ = True

    observer_chunk: int = eqx.field(static=True)
    particle_chunk: int = eqx.field(static=True)
    maximum_root_steps: int = eqx.field(static=True)
    maximum_working_bytes: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    resources_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        observer_chunk: int = 256,
        particle_chunk: int = 32,
        maximum_root_steps: int = 64,
        maximum_working_bytes: int = 2**31,
        maximum_output_bytes: int = 2**31,
    ) -> None:
        self.observer_chunk = positive_integer(observer_chunk, "observer_chunk")
        self.particle_chunk = positive_integer(particle_chunk, "particle_chunk")
        self.maximum_root_steps = positive_integer(
            maximum_root_steps, "maximum_root_steps"
        )
        self.maximum_working_bytes = positive_integer(
            maximum_working_bytes, "maximum_working_bytes"
        )
        self.maximum_output_bytes = positive_integer(
            maximum_output_bytes, "maximum_output_bytes"
        )
        self.resources_id = canonical_fingerprint(
            {
                "kind": "lienard-wiechert-resources",
                "observer_chunk": self.observer_chunk,
                "particle_chunk": self.particle_chunk,
                "maximum_root_steps": self.maximum_root_steps,
                "maximum_working_bytes": self.maximum_working_bytes,
                "maximum_output_bytes": self.maximum_output_bytes,
            }
        )


class LienardWiechertResourceEstimate(StrictModule):
    """Static byte estimate of one evaluation.

    ``working_bytes`` covers one ``observer_chunk × particle_chunk`` step: pair
    state, the particle chunk's lane samples, and the observer accumulators.
    ``output_bytes`` holds retarded times and pair status for every pair plus
    the per-observer fields and evidence.
    """

    working_bytes: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    maximum_working_bytes: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    observer_count: int = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    observer_chunk: int = eqx.field(static=True)
    particle_chunk: int = eqx.field(static=True)
    bisection_steps: int = eqx.field(static=True)


class LienardWiechertFieldPlan(StrictModule, NonTrainableState):
    """Static near-zone field request for point charges in vacuum.

    ``interpolation`` reconstructs each sample segment: ``"hermite-cubic"``
    uses a cubic Hermite position from sampled positions and velocities with
    ``1/γ²`` from the proper-velocity chord; ``"hermite-quintic"`` also uses
    ``proper_accelerations``, a quintic Hermite position, and a cubic Hermite
    proper velocity. ``history`` decides retarded times before the first
    sample: ``"refuse"`` reports them unsupported, ``"inertial-extrapolation"``
    continues the first sample's uniform motion backwards. Pairs whose
    retarded distance is below ``exclusion_radius`` are excluded from the
    field and their charge is reported.
    """

    __strict_contract__ = True

    scale: ElectromagneticScaleContract
    history: LienardWiechertHistory = eqx.field(static=True)
    exclusion_radius: float = eqx.field(static=True)
    interpolation: LienardWiechertInterpolation = eqx.field(static=True)
    resources: LienardWiechertResources
    plan_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        /,
        *,
        history: LienardWiechertHistory,
        exclusion_radius: float,
        interpolation: LienardWiechertInterpolation,
        resources: LienardWiechertResources | None = None,
    ) -> None:
        history_ = parse(history, LienardWiechertHistory, "history")
        interpolation_ = parse(
            interpolation, LienardWiechertInterpolation, "interpolation"
        )
        radius = positive_finite_float(exclusion_radius, "exclusion_radius")
        policy = LienardWiechertResources() if resources is None else resources
        if not isinstance(policy, LienardWiechertResources):
            raise TypeError("resources must be a LienardWiechertResources.")
        self.scale = scale
        self.history = history_
        self.exclusion_radius = radius
        self.interpolation = interpolation_
        self.resources = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lienard-wiechert-field-plan",
                "scale": scale.scale_id,
                "history": history_,
                "exclusion_radius": radius,
                "interpolation": interpolation_,
                "resources": policy.resources_id,
            }
        )

    def prepare(self) -> PreparedLienardWiechertField:
        return PreparedLienardWiechertField(self)


class LienardWiechertEvidence(StrictModule):
    """Support, accounting, and accuracy evidence of one field evaluation.

    ``supported`` is false when any lane has an unsupported retarded point;
    the fields of such observers are NaN, never zero-filled. ``resolved`` also
    requires finite fields and resolved kinematics. ``derivative_valid`` is
    false where unsupported, nonfinite, or where an exclusion or activity
    boundary (a discrete event) decided a contribution. ``excluded_charge`` and
    ``absent_charge`` are the multiplicity-weighted charges left out by the
    exclusion radius and by inactive retarded samples. ``maximum_root_residual``
    is the observer-time residual of the retarded solve, and
    ``maximum_lorentz_mismatch`` the relative disagreement of ``1 − β²`` from
    the position interpolant with ``1/γ²`` from the proper velocity.
    """

    __strict_contract__ = True

    status: Int32[_ObserverDim]
    pair_status: Int32[_ObserverDim, _ParticleDim]
    supported: Bool[_ObserverDim]
    finite: Bool[_ObserverDim]
    resolved: Bool[_ObserverDim]
    derivative_valid: Bool[_ObserverDim]
    excluded_charge: Float64[_ObserverDim]
    excluded_count: Int32[_ObserverDim]
    absent_charge: Float64[_ObserverDim]
    extrapolated_count: Int32[_ObserverDim]
    minimum_retardation_factor: Float64[_ObserverDim]
    maximum_root_residual: Float64[_ObserverDim]
    maximum_lorentz_mismatch: Float64[_ObserverDim]
    resource_estimate: LienardWiechertResourceEstimate = eqx.field(static=True)


class LienardWiechertFieldResult(StrictModule):
    """Fields at observer events ``(t, x, y, z)``.

    ``electric_field = velocity_field + acceleration_field``; ``magnetic_field``
    sums ``n × E / c`` per lane. ``retarded_times[O, P]`` are NaN for
    unsupported pairs.
    """

    __strict_contract__ = True

    electric_field: Float64[_ObserverDim, Literal[3]]
    magnetic_field: Float64[_ObserverDim, Literal[3]]
    velocity_field: Float64[_ObserverDim, Literal[3]]
    acceleration_field: Float64[_ObserverDim, Literal[3]]
    retarded_times: Float64[_ObserverDim, _ParticleDim]
    observer_events: Float64[_ObserverDim, Literal[4]]
    evidence: LienardWiechertEvidence
    history: LienardWiechertHistory = eqx.field(static=True)
    interpolation: LienardWiechertInterpolation = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


class _Lanes(NamedTuple):
    """Particle-major lane samples; rate channels exist only for quintic."""

    times: Array
    positions: Array
    velocities: Array
    accelerations: Array | None
    proper_velocities: Array
    proper_rates: Array | None
    inverse_gamma_sq: Array
    active: Array
    flags: Array
    weights: Array
    valid: Array


class _Segment(NamedTuple):
    step: Array
    chord: Array
    velocity0: Array
    velocity1: Array
    acceleration0: Array | None
    acceleration1: Array | None
    proper0: Array
    proper1: Array
    proper_rate0: Array | None
    proper_rate1: Array | None


class _Pair(NamedTuple):
    velocity_field: Array
    acceleration_field: Array
    magnetic_field: Array
    retarded_time: Array
    status: Array
    contributes: Array
    unsupported: Array
    excluded: Array
    inactive: Array
    extrapolated: Array
    kappa: Array
    residual: Array
    mismatch: Array


class _Retarded(NamedTuple):
    """Retarded kinematics relative to a reference sample ``(t₀, r₀)``.

    ``separation = x − r₀``, ``offset = r(t_r) − r₀``, ``delay = t_r − t₀``;
    ``position_speed_sq`` is ``|ṙ/c|²`` of the position reconstruction and
    ``residual`` the observer-time residual of the retarded condition.
    """

    separation: Array
    offset: Array
    delay: Array
    beta: Array
    beta_rate: Array
    inverse_gamma_sq: Array
    position_speed_sq: Array
    residual: Array
    solved: Array


class _Config(NamedTuple):
    interpolation: LienardWiechertInterpolation
    history: LienardWiechertHistory
    exclusion_radius: float
    speed_of_light: float
    bisection_steps: int
    termination: NonlinearTermination


class _Accumulator(NamedTuple):
    velocity_field: Array
    acceleration_field: Array
    magnetic_field: Array
    unsupported: Array
    status: Array
    excluded_charge: Array
    excluded_count: Array
    absent_charge: Array
    extrapolated_count: Array
    minimum_kappa: Array
    maximum_residual: Array
    maximum_mismatch: Array


def _safe_norm(vector: Array, /) -> Array:
    square = jnp.sum(vector * vector, axis=-1)
    positive = square > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, square, 1.0)), 0.0)


def _distance_change(separation: Array, offset: Array, distance: Array, /) -> Array:
    """``|d − Δr| − |d|`` without the cancellation of two large distances."""
    total = _safe_norm(separation - offset) + distance
    numerator = -2.0 * jnp.dot(separation, offset) + jnp.dot(offset, offset)
    return numerator / jnp.where(total > 0.0, total, 1.0)


def _inverse_gamma_sq(proper: Array, speed_of_light: float, /) -> Array:
    return 1.0 / (1.0 + jnp.sum(proper * proper, axis=-1) / speed_of_light**2)


def _quintic_position(segment: _Segment, s: Array, derivative: int, /) -> Array:
    if segment.acceleration0 is None or segment.acceleration1 is None:
        raise ValueError("hermite-quintic requires proper_accelerations.")
    h = segment.step
    basis = _hermite_basis(s)[derivative]
    # The first basis function multiplies r₀, which the offset form removes.
    data = (
        h * segment.velocity0,
        h * h * segment.acceleration0,
        segment.chord,
        h * segment.velocity1,
        h * h * segment.acceleration1,
    )
    total = sum(
        (weight * value for weight, value in zip(basis[1:], data, strict=True)),
        start=jnp.zeros((3,), dtype=jnp.float64),
    )
    return total / h**derivative


def _position_offset(
    interpolation: LienardWiechertInterpolation,
    segment: _Segment,
    s: Array,
    derivative: int,
    /,
) -> Array:
    """``r(s) − r₀`` or its ``derivative``-th time derivative on one segment."""
    match interpolation:
        case "hermite-cubic":
            return cubic_hermite_segment(
                jnp.zeros((3,), dtype=jnp.float64),
                segment.chord,
                segment.velocity0,
                segment.velocity1,
                s,
                segment.step,
                derivative_order=derivative,
            )
        case "hermite-quintic":
            return _quintic_position(segment, s, derivative)
        case _:
            assert_never(interpolation)


def _segment_kinematics(
    interpolation: LienardWiechertInterpolation,
    segment: _Segment,
    s: Array,
    speed_of_light: float,
    /,
) -> tuple[Array, Array, Array]:
    """``(β, β̇, 1/γ²)`` at ``s`` on one segment.

    The cubic route has no sampled rates: ``β`` and ``β̇`` differentiate the
    cubic position and ``1/γ²`` follows the proper-velocity chord. The quintic
    route takes all three from the cubic Hermite proper velocity, so
    ``1 − β² = 1/γ²`` holds exactly and ``β̇`` never differentiates rounded
    positions twice.
    """
    c = speed_of_light
    match interpolation:
        case "hermite-cubic":
            proper = linear_segment(segment.proper0, segment.proper1, s, segment.step)
            return (
                _position_offset(interpolation, segment, s, 1) / c,
                _position_offset(interpolation, segment, s, 2) / c,
                _inverse_gamma_sq(proper, c),
            )
        case "hermite-quintic":
            if segment.proper_rate0 is None or segment.proper_rate1 is None:
                raise ValueError("hermite-quintic requires proper_accelerations.")
            ends = (
                segment.proper0,
                segment.proper1,
                segment.proper_rate0,
                segment.proper_rate1,
                s,
                segment.step,
            )
            proper = cubic_hermite_segment(*ends)
            rate = cubic_hermite_segment(*ends, derivative_order=1)
            inverse_gamma_sq = _inverse_gamma_sq(proper, c)
            inverse_gamma = jnp.sqrt(inverse_gamma_sq)
            # d/dt (u / γ) = u̇ / γ − u (u·u̇) / (c² γ³)
            acceleration = (
                rate * inverse_gamma
                - proper * jnp.dot(proper, rate) * inverse_gamma**3 / c**2
            )
            return proper * inverse_gamma / c, acceleration / c, inverse_gamma_sq
        case _:
            assert_never(interpolation)


def _segment_retarded(
    config: _Config,
    segment: _Segment,
    separation: Array,
    distance: Array,
    lag: Array,
    /,
) -> _Retarded:
    """Bracketed retarded root on one Hermite segment, ``s ∈ [0, 1]``.

    The residual is ``g(t₀ + h s) / h`` with ``g(t₀) = lag``; the implicit
    function theorem differentiates the accepted root through a custom JVP
    root rule, never through the bracketing iterations.
    """
    c = config.speed_of_light
    h = segment.step
    base = lag / h
    zero = jnp.zeros((), dtype=jnp.float64)
    one = jnp.ones((), dtype=jnp.float64)

    def residual(s: Array) -> Array:
        offset = _position_offset(config.interpolation, segment, s, 0)
        return base - s - _distance_change(separation, offset, distance) / (c * h)

    def solve(function: Callable[[Array], Array], guess: Array) -> tuple[Array, Array]:
        del guess
        result = scalar_root(
            ScalarRootProblem(
                lambda state, args: function(state),
                bracket=(zero, one),
                problem_id="lienard-wiechert-retarded-time",
            ),
            method=TOMS748(),
            termination=config.termination,
        )
        # The node bisection guarantees g(t₁) ≤ 0; a nonnegative residual at
        # s = 1 is the rounding of a root on the node itself.
        upper = function(one)
        at_upper = upper >= 0.0
        return (
            jnp.where(at_upper, one, result.root),
            jnp.where(
                at_upper,
                upper <= _ROOT_TOLERANCE,
                result.successful
                | (result.status == int(NonlinearStatus.RESIDUAL_STAGNATION)),
            ),
        )

    s, solved = custom_root(
        residual,
        0.5 * one,
        solve,
        lambda linearized, rhs: rhs / linearized(jnp.ones_like(rhs)),
        has_aux=True,
    )
    beta, beta_rate, inverse_gamma_sq = _segment_kinematics(
        config.interpolation, segment, s, c
    )
    position_beta = _position_offset(config.interpolation, segment, s, 1) / c
    return _Retarded(
        separation,
        _position_offset(config.interpolation, segment, s, 0),
        h * s,
        beta,
        beta_rate,
        inverse_gamma_sq,
        jnp.dot(position_beta, position_beta),
        h * residual(s),
        solved,
    )


def _extrapolated_retarded(
    tau: Array,
    separation: Array,
    distance: Array,
    lag: Array,
    velocity: Array,
    inverse_gamma_sq: Array,
    before: Array,
    speed_of_light: float,
    /,
) -> _Retarded:
    """Closed-form retarded time on the backward inertial continuation.

    With ``s = t_r − t₀`` the retarded condition squares to
    ``(c² − v²) s² − 2 (c² τ − d·v) s + (c τ − |d|)(c τ + |d|) = 0``; the
    retarded root is the smaller one, taken in the cancellation-free form, and
    ``c² − v² = c²/γ²`` comes from the proper velocity.
    """
    c = speed_of_light
    linear = c * c * tau - jnp.dot(separation, velocity)
    constant = (c * lag) * (c * tau + distance)
    quadratic = c * c * inverse_gamma_sq
    discriminant = jnp.where(
        before, jnp.maximum(linear * linear - quadratic * constant, 0.0), 1.0
    )
    root = jnp.sqrt(discriminant)
    positive = linear >= 0.0
    denominator = jnp.where(positive, linear + root, quadratic)
    safe = jnp.where(before & (denominator != 0.0), denominator, 1.0)
    numerator = jnp.where(positive, constant, linear - root)
    delay = jnp.where(before, numerator / safe, 0.0)
    offset = velocity * delay
    beta = velocity / c
    return _Retarded(
        separation,
        offset,
        delay,
        beta,
        jnp.zeros((3,), dtype=jnp.float64),
        inverse_gamma_sq,
        jnp.dot(beta, beta),
        lag - delay - _distance_change(separation, offset, distance) / c,
        jnp.asarray(True),
    )


def _status_bit(condition: Array, flag: LienardWiechertStatus, /) -> Array:
    return jnp.where(condition, jnp.int32(int(flag)), jnp.int32(0))


def _pair(config: _Config, event: Array, lane: _Lanes, /) -> _Pair:
    """Field of one lane at one observer event (unweighted, without 1/4πε₀)."""
    c = config.speed_of_light
    time, point = event[0], event[1:]
    last = lane.times.shape[0] - 1

    def node_lag(index: Array) -> Array:
        return time - lane.times[index] - _safe_norm(point - lane.positions[index]) / c

    before = node_lag(jnp.int32(0)) < 0.0
    after = node_lag(jnp.int32(last)) > 0.0

    def bisect(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        lower, upper = bounds
        middle = (lower + upper) // 2
        reached = node_lag(middle) >= 0.0
        return jnp.where(reached, middle, lower), jnp.where(reached, upper, middle)

    index, _ = jax.lax.fori_loop(
        0,
        config.bisection_steps,
        bisect,
        (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(last, dtype=jnp.int32)),
    )
    following = index + 1
    step = lane.times[following] - lane.times[index]
    segment = _Segment(
        jnp.where(step > 0.0, step, 1.0),
        lane.positions[following] - lane.positions[index],
        lane.velocities[index],
        lane.velocities[following],
        None if lane.accelerations is None else lane.accelerations[index],
        None if lane.accelerations is None else lane.accelerations[following],
        lane.proper_velocities[index],
        lane.proper_velocities[following],
        None if lane.proper_rates is None else lane.proper_rates[index],
        None if lane.proper_rates is None else lane.proper_rates[following],
    )
    separation = point - lane.positions[index]
    distance = _safe_norm(separation)
    retarded = _segment_retarded(
        config, segment, separation, distance, time - lane.times[index] - distance / c
    )
    reference_time = lane.times[index]
    active = lane.active[index] & lane.active[following]
    match config.history:
        case "refuse":
            extrapolating = jnp.asarray(False)
        case "inertial-extrapolation":
            extrapolating = before
            first_separation = point - lane.positions[0]
            first_distance = _safe_norm(first_separation)
            tau = time - lane.times[0]
            continuation = _extrapolated_retarded(
                tau,
                first_separation,
                first_distance,
                tau - first_distance / c,
                lane.velocities[0],
                lane.inverse_gamma_sq[0],
                before,
                c,
            )
            retarded = jax.tree.map(
                lambda left, right: jnp.where(before, left, right),
                continuation,
                retarded,
            )
            reference_time = jnp.where(before, lane.times[0], reference_time)
            active = jnp.where(before, lane.active[0], active)
        case _:
            assert_never(config.history)
    beta = retarded.beta
    # The retarded solve assumes a subluminal position reconstruction.
    speed_sq = retarded.position_speed_sq
    lane_failed = lane.flags != 0
    evaluated = ~lane_failed & ~after & (~before | extrapolating)
    root_failed = evaluated & ~retarded.solved
    superluminal = evaluated & (speed_sq >= 1.0)
    unsupported = lane.valid & (
        lane_failed | after | (before & ~extrapolating) | root_failed | superluminal
    )
    admitted = lane.valid & evaluated & ~unsupported
    inactive = admitted & ~active
    separation_now = retarded.separation - retarded.offset
    radius = _safe_norm(separation_now)
    excluded = admitted & active & (radius < config.exclusion_radius)
    contributes = admitted & active & ~excluded
    safe_radius = jnp.where(contributes, radius, 1.0)
    direction = separation_now / safe_radius
    inverse_gamma_sq = retarded.inverse_gamma_sq
    transverse = jnp.cross(direction, beta)
    denominator = jnp.where(contributes, 1.0 + jnp.dot(direction, beta), 1.0)
    kappa = (inverse_gamma_sq + jnp.dot(transverse, transverse)) / denominator
    kappa_cubed = jnp.where(contributes, kappa, 1.0) ** 3
    relative = direction - beta
    velocity_field = relative * inverse_gamma_sq / (kappa_cubed * safe_radius**2)
    acceleration_field = jnp.cross(direction, jnp.cross(relative, retarded.beta_rate)) / (
        c * kappa_cubed * safe_radius
    )
    magnetic_field = jnp.cross(direction, velocity_field + acceleration_field) / c
    mismatch = jnp.abs(1.0 - speed_sq - inverse_gamma_sq) / inverse_gamma_sq
    unresolved = admitted & active & (mismatch > _LORENTZ_MISMATCH_LIMIT)
    status = (
        lane.flags
        | _status_bit(before, LienardWiechertStatus.RETARDED_BEFORE_WINDOW)
        | _status_bit(after, LienardWiechertStatus.RETARDED_AFTER_WINDOW)
        | _status_bit(inactive, LienardWiechertStatus.INACTIVE_RETARDED)
        | _status_bit(excluded, LienardWiechertStatus.EXCLUDED_CHARGE)
        | _status_bit(root_failed, LienardWiechertStatus.ROOT_FAILED)
        | _status_bit(superluminal, LienardWiechertStatus.SUPERLUMINAL_INTERPOLANT)
        | _status_bit(unresolved, LienardWiechertStatus.UNRESOLVED_KINEMATICS)
    )
    zero = jnp.zeros((3,), dtype=jnp.float64)
    return _Pair(
        jnp.where(contributes, velocity_field, zero),
        jnp.where(contributes, acceleration_field, zero),
        jnp.where(contributes, magnetic_field, zero),
        jnp.where(lane.valid & ~unsupported, reference_time + retarded.delay, jnp.nan),
        jnp.where(lane.valid, status, jnp.int32(0)),
        contributes,
        unsupported,
        excluded,
        inactive,
        admitted & before,
        jnp.where(contributes, kappa, jnp.inf),
        jnp.where(admitted, jnp.abs(retarded.residual), 0.0),
        jnp.where(admitted & active, mismatch, 0.0),
    )


def _pad_leading(array: Array, multiple: int, /) -> Array:
    padding = (-array.shape[0]) % multiple
    return jnp.pad(array, ((0, padding),) + ((0, 0),) * (array.ndim - 1), mode="edge")


def _chunk(array: Array, size: int, /) -> Array:
    return array.reshape((-1, size, *array.shape[1:]))


def _empty_accumulator(count: int, /) -> _Accumulator:
    vectors = jnp.zeros((count, 3), dtype=jnp.float64)
    reals = jnp.zeros((count,), dtype=jnp.float64)
    counts = jnp.zeros((count,), dtype=jnp.int32)
    return _Accumulator(
        vectors,
        vectors,
        vectors,
        jnp.zeros((count,), dtype=jnp.bool_),
        counts,
        reals,
        counts,
        reals,
        counts,
        jnp.full((count,), jnp.inf, dtype=jnp.float64),
        reals,
        reals,
    )


def _accumulate(total: _Accumulator, pair: _Pair, weights: Array, /) -> _Accumulator:
    """Fold one ``[observers, particles]`` pair block into the observer sums."""
    return _Accumulator(
        total.velocity_field + contract("p,opc->oc", weights, pair.velocity_field),
        total.acceleration_field
        + contract("p,opc->oc", weights, pair.acceleration_field),
        total.magnetic_field + contract("p,opc->oc", weights, pair.magnetic_field),
        total.unsupported | jnp.any(pair.unsupported, axis=1),
        total.status
        | jax.lax.reduce(pair.status, jnp.int32(0), jax.lax.bitwise_or, (1,)),
        total.excluded_charge
        + jnp.sum(jnp.where(pair.excluded, weights[None, :], 0.0), axis=1),
        total.excluded_count + jnp.sum(pair.excluded, axis=1, dtype=jnp.int32),
        total.absent_charge
        + jnp.sum(jnp.where(pair.inactive, weights[None, :], 0.0), axis=1),
        total.extrapolated_count + jnp.sum(pair.extrapolated, axis=1, dtype=jnp.int32),
        jnp.minimum(total.minimum_kappa, jnp.min(pair.kappa, axis=1)),
        jnp.maximum(total.maximum_residual, jnp.max(pair.residual, axis=1)),
        jnp.maximum(total.maximum_mismatch, jnp.max(pair.mismatch, axis=1)),
    )


class PreparedLienardWiechertField(StrictModule, NonTrainableState):
    """Prepared constants and retarded-root termination of one plan.

    ``evaluate(trajectory, observer_events)`` returns the fields of every lane
    at observer events ``[O, 4] = (t, x, y, z)`` in the scale's units.
    """

    __strict_contract__ = True

    plan: LienardWiechertFieldPlan
    field_prefactor: Float64[Scalar]
    termination: NonlinearTermination
    speed_of_light: float = eqx.field(static=True)

    @checked
    def __init__(self, plan: LienardWiechertFieldPlan, /) -> None:
        self.plan = plan
        self.speed_of_light = float(plan.scale.speed_of_light)
        self.field_prefactor = jnp.asarray(
            1.0 / (4.0 * math.pi * float(plan.scale.vacuum_permittivity))
        )
        self.termination = NonlinearTermination(
            absolute_residual=_ROOT_TOLERANCE,
            relative_residual=0.0,
            absolute_step=_ROOT_RESOLUTION,
            relative_step=0.0,
            maximum_steps=plan.resources.maximum_root_steps,
        )

    def resource_estimate(
        self, observer_count: int, particle_count: int, sample_count: int, /
    ) -> LienardWiechertResourceEstimate:
        policy = self.plan.resources
        observers = min(policy.observer_chunk, observer_count)
        particles = min(policy.particle_chunk, particle_count)
        working = _REAL_ITEMSIZE * (
            observers * particles * _PAIR_CHANNELS
            + particles * sample_count * _LANE_CHANNELS
            + observers * _OBSERVER_CHANNELS
        )
        output = (
            observer_count * particle_count * (_REAL_ITEMSIZE + _STATUS_ITEMSIZE)
            + observer_count * _OBSERVER_OUTPUT_CHANNELS * _REAL_ITEMSIZE
        )
        return LienardWiechertResourceEstimate(
            working,
            output,
            policy.maximum_working_bytes,
            policy.maximum_output_bytes,
            observer_count,
            particle_count,
            sample_count,
            observers,
            particles,
            max(sample_count - 2, 0).bit_length(),
        )

    def _refuse(
        self, observer_count: int, particle_count: int, sample_count: int, /
    ) -> LienardWiechertResourceEstimate:
        estimate = self.resource_estimate(observer_count, particle_count, sample_count)
        if estimate.working_bytes > estimate.maximum_working_bytes:
            raise LienardWiechertResourceError(
                f"Liénard–Wiechert fields need {estimate.working_bytes} working bytes "
                f"per observer–particle chunk, above maximum_working_bytes="
                f"{estimate.maximum_working_bytes}."
            )
        if estimate.output_bytes > estimate.maximum_output_bytes:
            raise LienardWiechertResourceError(
                f"Liénard–Wiechert outputs need {estimate.output_bytes} bytes, above "
                f"maximum_output_bytes={estimate.maximum_output_bytes}."
            )
        return estimate

    def _lanes(self, trajectory: ChargedTrajectory, /) -> _Lanes:
        c = self.speed_of_light
        times = jnp.swapaxes(trajectory.times, 0, 1)
        positions = jnp.swapaxes(trajectory.positions, 0, 1)
        proper = jnp.swapaxes(trajectory.proper_velocities, 0, 1)
        inverse_gamma_sq = _inverse_gamma_sq(proper, c)
        inverse_gamma = jnp.sqrt(inverse_gamma_sq)[..., None]
        match self.plan.interpolation:
            case "hermite-cubic":
                rates = accelerations = None
            case "hermite-quintic":
                if trajectory.proper_accelerations is None:
                    raise ValueError("hermite-quintic requires proper_accelerations.")
                rates = jnp.swapaxes(trajectory.proper_accelerations, 0, 1)
                # d/dt (u / γ) = u̇ / γ − u (u·u̇) / (c² γ³)
                accelerations = (
                    rates * inverse_gamma
                    - proper
                    * jnp.sum(proper * rates, axis=-1)[..., None]
                    * inverse_gamma**3
                    / c**2
                )
            case _:
                assert_never(self.plan.interpolation)
        step = times[:, 1:] - times[:, :-1]
        chord = _safe_norm(positions[:, 1:] - positions[:, :-1])
        flags = _status_bit(
            jnp.any(step <= 0.0, axis=1), LienardWiechertStatus.NONMONOTONE_TIME
        ) | _status_bit(
            jnp.any((step > 0.0) & (chord >= c * step), axis=1),
            LienardWiechertStatus.SUPERLUMINAL_SAMPLES,
        )
        return _Lanes(
            times,
            positions,
            proper * inverse_gamma,
            accelerations,
            proper,
            rates,
            inverse_gamma_sq,
            jnp.swapaxes(trajectory.active, 0, 1),
            flags,
            trajectory.charges * trajectory.multiplicities,
            jnp.ones((trajectory.particle_count,), dtype=jnp.bool_),
        )

    def _observer_events(self, observer_events: ArrayLike, /) -> Array:
        events = _float64_array(observer_events, "observer_events")
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError("observer_events must have shape (observers, 4).")
        return parse(
            events, Float64[_ObserverDim, Literal[4]], "observer_events", scope=Scope()
        )

    @checked
    def evaluate(
        self, trajectory: ChargedTrajectory, observer_events: ArrayLike, /
    ) -> LienardWiechertFieldResult:
        """Fields of every lane of ``trajectory`` at ``observer_events``."""
        if trajectory.sample_count < 2:
            raise ValueError(
                "Liénard–Wiechert fields need at least two samples per lane."
            )
        events = self._observer_events(observer_events)
        observer_count = events.shape[0]
        particle_count = trajectory.particle_count
        estimate = self._refuse(observer_count, particle_count, trajectory.sample_count)
        lanes = self._lanes(trajectory)
        plan = self.plan
        config = _Config(
            plan.interpolation,
            plan.history,
            plan.exclusion_radius,
            self.speed_of_light,
            estimate.bisection_steps,
            self.termination,
        )
        observer_chunk, particle_chunk = estimate.observer_chunk, estimate.particle_chunk
        padded = jax.tree.map(lambda array: _pad_leading(array, particle_chunk), lanes)
        # Padding lanes repeat a real lane so every intermediate stays finite;
        # they are invalid and carry no field, status, or accounting.
        padded = padded._replace(valid=jnp.arange(padded.valid.shape[0]) < particle_count)
        lane_chunks = jax.tree.map(lambda array: _chunk(array, particle_chunk), padded)
        event_chunks = _chunk(_pad_leading(events, observer_chunk), observer_chunk)

        def pair_block(block: Array, lane_block: _Lanes) -> _Pair:
            per_lane = jax.vmap(lambda event, lane: _pair(config, event, lane), (None, 0))
            return jax.vmap(per_lane, (0, None))(block, lane_block)

        def observer_block(block: Array) -> tuple[_Accumulator, Array, Array]:
            def particle_step(
                total: _Accumulator, lane_block: _Lanes
            ) -> tuple[_Accumulator, tuple[Array, Array]]:
                pair = pair_block(block, lane_block)
                return _accumulate(total, pair, lane_block.weights), (
                    pair.retarded_time,
                    pair.status,
                )

            total, (retarded, status) = jax.lax.scan(
                particle_step, _empty_accumulator(observer_chunk), lane_chunks
            )
            return total, retarded, status

        totals, retarded, pair_status = jax.lax.map(observer_block, event_chunks)

        def observers(array: Array) -> Array:
            return array.reshape((-1, *array.shape[2:]))[:observer_count]

        def pairs(array: Array) -> Array:
            ordered = jnp.transpose(array, (0, 2, 1, 3))
            flat = ordered.reshape((ordered.shape[0] * ordered.shape[1], -1))
            return flat[:observer_count, :particle_count]

        total = jax.tree.map(observers, totals)
        return self._result(events, total, pairs(retarded), pairs(pair_status), estimate)

    def _result(
        self,
        events: Array,
        total: _Accumulator,
        retarded: Array,
        pair_status: Array,
        estimate: LienardWiechertResourceEstimate,
        /,
    ) -> LienardWiechertFieldResult:
        supported = ~total.unsupported
        mask = supported[:, None]

        def field(array: Array) -> Array:
            return jnp.where(mask, self.field_prefactor * array, jnp.nan)

        velocity = field(total.velocity_field)
        acceleration = field(total.acceleration_field)
        electric = velocity + acceleration
        magnetic = field(total.magnetic_field)
        finite = jnp.all(jnp.isfinite(electric) & jnp.isfinite(magnetic), axis=-1)
        status = total.status | _status_bit(
            supported & ~finite, LienardWiechertStatus.NONFINITE
        )
        unresolved = int(LienardWiechertStatus.UNRESOLVED_KINEMATICS)
        discrete = int(
            LienardWiechertStatus.EXCLUDED_CHARGE
            | LienardWiechertStatus.INACTIVE_RETARDED
        )
        plan = self.plan
        return LienardWiechertFieldResult(
            electric_field=electric,
            magnetic_field=magnetic,
            velocity_field=velocity,
            acceleration_field=acceleration,
            retarded_times=retarded,
            observer_events=events,
            evidence=LienardWiechertEvidence(
                status=status,
                pair_status=pair_status,
                supported=supported,
                finite=finite,
                resolved=supported & finite & ((status & unresolved) == 0),
                derivative_valid=supported & finite & ((status & discrete) == 0),
                excluded_charge=total.excluded_charge,
                excluded_count=total.excluded_count,
                absent_charge=total.absent_charge,
                extrapolated_count=total.extrapolated_count,
                minimum_retardation_factor=total.minimum_kappa,
                maximum_root_residual=total.maximum_residual,
                maximum_lorentz_mismatch=total.maximum_mismatch,
                resource_estimate=estimate,
            ),
            history=plan.history,
            interpolation=plan.interpolation,
            plan_id=plan.plan_id,
        )


__all__ = [
    "LienardWiechertEvidence",
    "LienardWiechertFieldPlan",
    "LienardWiechertFieldResult",
    "LienardWiechertHistory",
    "LienardWiechertInterpolation",
    "LienardWiechertResourceError",
    "LienardWiechertResourceEstimate",
    "LienardWiechertResources",
    "LienardWiechertStatus",
    "PreparedLienardWiechertField",
]
