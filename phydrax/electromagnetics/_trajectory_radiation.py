#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Far-field spectra radiated by charged-particle trajectories in vacuum.

Conventions (shared by every radiation route): phasors ``exp(-i ω t)``,
transient spectra ``F(ω) = ∫ f(t) exp(+i ω t) dt`` over observer time, and the
one-sided spectral energy ``d²W/(dω dΩ) = ε₀ c |r Ẽ|² / π`` for ``ω > 0``.

The field spectrum is the acceleration form of the Liénard–Wiechert far
field,

    r Ẽ(ω) = q / (4 π ε₀ c) ∫ d/dt [ n × (n × β) / κ ] exp(i ω τ(t)) dt,

with retardation factor ``κ = 1 − n·β`` and observer time
``τ(t) = t − n·r(t)/c``. A trajectory sampled at nodes carries piecewise
constant ``β`` between nodes, so the integrand is a sum of jumps ``Δa_j`` of
``a = n × (n × β) / κ`` at the nodes: no radiation is attributed to the window
edges or to activity boundaries. Every route reports the same evidence.
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

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._lorentz import SpectralEmissionCompleteness
from .._physical import ElectromagneticScaleContract
from .._polynomial._orthogonal import legendre_rule_data
from .._spectral._nonuniform_fourier import (
    NonuniformFourierGridEvidence,
    NonuniformFourierType3Plan,
    PreparedNonuniformFourierType3,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, positive_finite_float, positive_integer
from ..ein import contract
from ..typing import (
    Bool,
    Complex128,
    Dim,
    Float64,
    Identifier,
    Int32,
    parse,
    Scalar,
    Scope,
    Size,
    UInt32,
)


TrajectoryRadiationRoute: TypeAlias = Literal[
    "segment-exact", "segment-hermite", "node-gridded"
]
RadiationCoherence: TypeAlias = Literal[
    "coherent", "incoherent", "gaussian-form-factor", "tabulated-form-factor"
]

# Sampling adequacy thresholds reported through ``TrajectoryRadiationStatus``.
# Piecewise-constant amplitudes lose about (ω Δτ)² / 24 of relative accuracy per
# segment; one radian of observer-time phase per segment bounds that at ~4%.
_PHASE_INCREMENT_LIMIT = 1.0
# An amplitude jump above half of the peak amplitude in one step means the
# emission pulse itself is not sampled.
_AMPLITUDE_INCREMENT_LIMIT = 0.5
# Edge jump rates above 1% of the strongest interior jump rate contradict a
# declared complete emission.
_WINDOW_EDGE_LIMIT = 1.0e-2
_ROUNDOFF_JUMP = 64.0 * float(np.finfo(np.float64).eps)
_DIRECTION_DEGENERACY = 1.0e-9
_REAL_ITEMSIZE = 8
_COMPLEX_ITEMSIZE = 16
# Real per-node, per-direction geometry channels: τ, κ, two amplitude and two
# jump components; the Hermite route adds twelve per quadrature point.
_NODE_CHANNELS = 6
_HERMITE_CHANNELS = 12


class TrajectoryRadiationStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    UNRESOLVED_PHASE = 2
    UNRESOLVED_AMPLITUDE = 4
    WINDOW_EDGE_ACCELERATION = 8
    ACTIVITY_TRANSITION = 16
    UNSUPPORTED_NODE = 32
    NONMONOTONE_TIME = 64
    LANE_MISMATCH = 128


class TrajectoryRadiationResourceError(ValueError):
    """A trajectory radiation evaluation exceeds its declared resource policy."""


class _SampleDim(Dim, minimum=1):
    """Trajectory samples along one lane."""


class _ParticleDim(Dim, minimum=1):
    """Particle lanes."""


class _DirectionDim(Dim, minimum=1):
    """Observer directions."""


class _FrequencyDim(Dim, minimum=1):
    """Angular frequencies."""


def _float64_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.dtype != jnp.float64:
        raise TypeError(
            f"{name} must be float64; radiation phases are always float64 and "
            f"{array.dtype} is refused."
        )
    return array


def _uint32_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.dtype != jnp.uint32:
        raise TypeError(f"{name} must be uint32 identity words.")
    return array


class ChargedTrajectory(StrictModule):
    """Sampled charged-particle lanes with per-lane float64 times.

    ``times`` may be one shared vector ``[T]`` or per-lane ``[T, P]``; each lane
    is a proper-time-ordered sequence of samples. ``proper_velocities`` are
    ``u = γ v`` in the scale's velocity units, ``proper_accelerations`` (optional,
    required by the Hermite route) are ``du/dt``. ``charges`` are physical
    charges of one particle of the lane and ``multiplicities`` the number of
    identical particles the lane represents. ``active`` marks samples that
    belong to an existing particle; every dtype except float64 is refused.
    """

    __strict_contract__ = True

    times: Float64[_SampleDim, _ParticleDim]
    positions: Float64[_SampleDim, _ParticleDim, Literal[3]]
    proper_velocities: Float64[_SampleDim, _ParticleDim, Literal[3]]
    proper_accelerations: Float64[_SampleDim, _ParticleDim, Literal[3]] | None
    charges: Float64[_ParticleDim]
    multiplicities: Float64[_ParticleDim]
    active: Bool[_SampleDim, _ParticleDim]
    id_hi: UInt32[_ParticleDim]
    id_lo: UInt32[_ParticleDim]
    sample_count: Size[_SampleDim] = eqx.field(static=True)
    particle_count: Size[_ParticleDim] = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        positions: ArrayLike,
        proper_velocities: ArrayLike,
        charges: ArrayLike,
        multiplicities: ArrayLike,
        active: ArrayLike,
        identities: tuple[ArrayLike, ArrayLike],
        /,
        *,
        proper_accelerations: ArrayLike | None = None,
    ) -> None:
        positions_ = _float64_array(positions, "positions")
        if positions_.ndim != 3 or positions_.shape[2] != 3:
            raise ValueError("positions must have shape (samples, particles, 3).")
        sample_count, particle_count = positions_.shape[0], positions_.shape[1]
        times_ = _float64_array(times, "times")
        if times_.ndim == 1:
            times_ = jnp.broadcast_to(times_[:, None], (times_.shape[0], particle_count))
        active_ = jnp.asarray(active)
        if active_.dtype != jnp.bool_:
            raise TypeError("active must be a boolean array.")
        scope = Scope()
        self.sample_count = parse(
            sample_count, Size[_SampleDim], "sample_count", scope=scope
        )
        self.particle_count = parse(
            particle_count, Size[_ParticleDim], "particle_count", scope=scope
        )
        self.times = parse(
            times_, Float64[_SampleDim, _ParticleDim], "times", scope=scope
        )
        self.positions = parse(
            positions_,
            Float64[_SampleDim, _ParticleDim, Literal[3]],
            "positions",
            scope=scope,
        )
        self.proper_velocities = parse(
            _float64_array(proper_velocities, "proper_velocities"),
            Float64[_SampleDim, _ParticleDim, Literal[3]],
            "proper_velocities",
            scope=scope,
        )
        self.proper_accelerations = (
            None
            if proper_accelerations is None
            else parse(
                _float64_array(proper_accelerations, "proper_accelerations"),
                Float64[_SampleDim, _ParticleDim, Literal[3]],
                "proper_accelerations",
                scope=scope,
            )
        )
        self.charges = parse(
            _float64_array(charges, "charges"),
            Float64[_ParticleDim],
            "charges",
            scope=scope,
        )
        self.multiplicities = parse(
            _float64_array(multiplicities, "multiplicities"),
            Float64[_ParticleDim],
            "multiplicities",
            scope=scope,
        )
        self.active = parse(
            active_, Bool[_SampleDim, _ParticleDim], "active", scope=scope
        )
        hi, lo = identities
        self.id_hi = parse(
            _uint32_array(hi, "identities[0]"),
            UInt32[_ParticleDim],
            "identities[0]",
            scope=scope,
        )
        self.id_lo = parse(
            _uint32_array(lo, "identities[1]"),
            UInt32[_ParticleDim],
            "identities[1]",
            scope=scope,
        )


class RadiationObserverPlan(StrictModule, NonTrainableState):
    """Unit observer directions with the transverse polarization basis.

    For direction ``n`` the basis is ``e1 = normalize(a − (a·n) n)`` for the
    ``reference_axis`` ``a`` and ``e2 = n × e1``, so ``(e1, e2, n)`` is
    right-handed. Directions within ``1e-9`` of ``±a`` have no basis and are
    refused; choose a transverse reference axis for on-axis observers.
    """

    __strict_contract__ = True

    directions: Float64[_DirectionDim, Literal[3]]
    reference_axis: Float64[Literal[3]]
    basis_first: Float64[_DirectionDim, Literal[3]]
    basis_second: Float64[_DirectionDim, Literal[3]]
    direction_count: Size[_DirectionDim] = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(self, directions: ArrayLike, reference_axis: ArrayLike, /) -> None:
        raw = np.asarray(directions, dtype=np.float64)
        axis = np.asarray(reference_axis, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0:
            raise ValueError("directions must have shape (directions, 3).")
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("reference_axis must be one finite three-vector.")
        if not np.all(np.isfinite(raw)):
            raise ValueError("directions must be finite.")
        norms = np.linalg.norm(raw, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("directions must be nonzero.")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm == 0.0:
            raise ValueError("reference_axis must be nonzero.")
        unit = raw / norms[:, None]
        axis = axis / axis_norm
        first = axis[None, :] - (unit @ axis)[:, None] * unit
        first_norm = np.linalg.norm(first, axis=1)
        if np.any(first_norm < _DIRECTION_DEGENERACY):
            raise ValueError(
                "directions parallel to reference_axis have no polarization basis."
            )
        first = first / first_norm[:, None]
        second = np.cross(unit, first)
        scope = Scope()
        self.direction_count = parse(
            unit.shape[0], Size[_DirectionDim], "direction_count", scope=scope
        )
        self.directions = parse(
            jnp.asarray(unit),
            Float64[_DirectionDim, Literal[3]],
            "directions",
            scope=scope,
        )
        self.reference_axis = jnp.asarray(axis)
        self.basis_first = jnp.asarray(first)
        self.basis_second = jnp.asarray(second)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "radiation-observer-plan",
                "directions": array_tree_fingerprint(unit),
                "reference_axis": axis.tolist(),
            }
        )


class TrajectoryRadiationResources(StrictModule, NonTrainableState):
    """Bounded execution policy for trajectory radiation.

    Particle lanes execute in ``lax.scan`` chunks of ``particle_chunk`` vectorized
    lanes; segments of one lane execute in ``lax.scan`` blocks of
    ``segment_block``. The working set of one chunk and the streaming state are
    estimated before execution and refused above ``maximum_working_bytes`` /
    ``maximum_state_bytes``. The ``node-gridded`` route uses
    ``gridded_tolerance`` and ``maximum_grid_points``.
    """

    __strict_contract__ = True

    particle_chunk: int = eqx.field(static=True)
    segment_block: int = eqx.field(static=True)
    maximum_working_bytes: int = eqx.field(static=True)
    maximum_state_bytes: int = eqx.field(static=True)
    gridded_tolerance: float = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    resources_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        particle_chunk: int = 32,
        segment_block: int = 32,
        maximum_working_bytes: int = 2**31,
        maximum_state_bytes: int = 2**31,
        gridded_tolerance: float = 1.0e-10,
        maximum_grid_points: int = 2**24,
    ) -> None:
        self.particle_chunk = positive_integer(particle_chunk, "particle_chunk")
        self.segment_block = positive_integer(segment_block, "segment_block")
        self.maximum_working_bytes = positive_integer(
            maximum_working_bytes, "maximum_working_bytes"
        )
        self.maximum_state_bytes = positive_integer(
            maximum_state_bytes, "maximum_state_bytes"
        )
        self.gridded_tolerance = positive_finite_float(
            gridded_tolerance, "gridded_tolerance"
        )
        self.maximum_grid_points = positive_integer(
            maximum_grid_points, "maximum_grid_points"
        )
        self.resources_id = canonical_fingerprint(
            {
                "kind": "trajectory-radiation-resources",
                "particle_chunk": self.particle_chunk,
                "segment_block": self.segment_block,
                "maximum_working_bytes": self.maximum_working_bytes,
                "maximum_state_bytes": self.maximum_state_bytes,
                "gridded_tolerance": self.gridded_tolerance,
                "maximum_grid_points": self.maximum_grid_points,
            }
        )


class TrajectoryRadiationResourceEstimate(StrictModule):
    """Static byte estimate of one ``accumulate`` call on the largest chunk.

    ``working_bytes`` covers one particle chunk: per-node segment geometry for
    every direction plus the transform working set of one segment block (or
    the gridded Type-3 grids). ``state_bytes`` is the streaming state, which
    holds per-lane spectra for every coherence model except ``"coherent"``.
    """

    working_bytes: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    maximum_working_bytes: int = eqx.field(static=True)
    maximum_state_bytes: int = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    particle_chunk: int = eqx.field(static=True)
    segment_block: int = eqx.field(static=True)
    quadrature_points: int = eqx.field(static=True)


class TrajectoryRadiationPlan(StrictModule, NonTrainableState):
    """Static far-field spectrum request.

    ``route`` selects the segment integration: ``"segment-exact"`` integrates the
    piecewise-constant amplitude of every segment analytically (velocity form
    with explicit activity-boundary terms, algebraically equal to the node jump
    sum), ``"segment-hermite"`` integrates a quintic Hermite reconstruction of
    the position (from sampled velocities and accelerations) with
    ``quadrature_order`` Gauss–Legendre points per segment, and
    ``"node-gridded"`` evaluates the node jump sum with a batched Type-3
    nonuniform Fourier transform over the declared ``observer_time_window``
    (direct evaluation when ``ω_max (τ_hi − τ_lo) ≤ 1``).

    ``coherence`` combines lanes: ``"coherent"`` sums fields, ``"incoherent"``
    sums coherency matrices, ``"gaussian-form-factor"`` spreads each lane's
    particles over a Gaussian bunch with rms sizes ``bunch_sigma`` (form factor
    ``exp(−ω² Σ n_i² σ_i² / (2 c²))``), and ``"tabulated-form-factor"`` uses the
    per-frequency magnitude ``form_factor``. ``emission`` declares whether the
    window holds the complete emission; edge acceleration then contradicts the
    declaration and is reported.
    """

    __strict_contract__ = True

    scale: ElectromagneticScaleContract
    observers: RadiationObserverPlan
    angular_frequencies: Float64[_FrequencyDim]
    coherence: RadiationCoherence = eqx.field(static=True)
    route: TrajectoryRadiationRoute = eqx.field(static=True)
    emission: SpectralEmissionCompleteness = eqx.field(static=True)
    form_factor: Float64[_FrequencyDim] | None
    bunch_sigma: Float64[Literal[3]] | None
    observer_time_window: tuple[float, float] | None = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    resources: TrajectoryRadiationResources
    frequency_count: Size[_FrequencyDim] = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        observers: RadiationObserverPlan,
        angular_frequencies: ArrayLike,
        /,
        *,
        coherence: RadiationCoherence,
        route: TrajectoryRadiationRoute,
        emission: SpectralEmissionCompleteness = "complete",
        form_factor: ArrayLike | None = None,
        bunch_sigma: ArrayLike | None = None,
        observer_time_window: tuple[float, float] | None = None,
        quadrature_order: int = 8,
        resources: TrajectoryRadiationResources | None = None,
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        if not isinstance(observers, RadiationObserverPlan):
            raise TypeError("observers must be a RadiationObserverPlan.")
        frequencies = np.asarray(angular_frequencies, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.shape[0] == 0:
            raise ValueError("angular_frequencies must be a nonempty vector.")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("angular_frequencies must be finite and positive.")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("angular_frequencies must increase strictly.")
        coherence_ = parse(coherence, RadiationCoherence, "coherence")
        route_ = parse(route, TrajectoryRadiationRoute, "route")
        emission_ = parse(emission, SpectralEmissionCompleteness, "emission")
        policy = TrajectoryRadiationResources() if resources is None else resources
        if not isinstance(policy, TrajectoryRadiationResources):
            raise TypeError("resources must be a TrajectoryRadiationResources.")
        factor = _tabulated_form_factor(coherence_, form_factor, frequencies.shape[0])
        sigma = _bunch_sigma(coherence_, bunch_sigma)
        window = _observer_window(route_, observer_time_window)
        order = positive_integer(quadrature_order, "quadrature_order")
        if order < 2:
            raise ValueError("quadrature_order must be at least two.")
        scope = Scope()
        self.frequency_count = parse(
            frequencies.shape[0], Size[_FrequencyDim], "frequency_count", scope=scope
        )
        self.scale = scale
        self.observers = observers
        self.angular_frequencies = parse(
            jnp.asarray(frequencies),
            Float64[_FrequencyDim],
            "angular_frequencies",
            scope=scope,
        )
        self.coherence = coherence_
        self.route = route_
        self.emission = emission_
        self.form_factor = None if factor is None else jnp.asarray(factor)
        self.bunch_sigma = None if sigma is None else jnp.asarray(sigma)
        self.observer_time_window = window
        self.quadrature_order = order
        self.resources = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "trajectory-radiation-plan",
                "scale": scale.scale_id,
                "observers": observers.plan_id,
                "angular_frequencies": array_tree_fingerprint(frequencies),
                "coherence": coherence_,
                "route": route_,
                "emission": emission_,
                "form_factor": None if factor is None else array_tree_fingerprint(factor),
                "bunch_sigma": None if sigma is None else sigma.tolist(),
                "observer_time_window": None if window is None else list(window),
                "quadrature_order": order,
                "resources": policy.resources_id,
            }
        )

    def prepare(self) -> PreparedTrajectoryRadiation:
        return PreparedTrajectoryRadiation(self)


def _tabulated_form_factor(
    coherence: RadiationCoherence, values: ArrayLike | None, count: int, /
) -> np.ndarray | None:
    if coherence != "tabulated-form-factor":
        if values is not None:
            raise ValueError("form_factor is accepted only by tabulated-form-factor.")
        return None
    if values is None:
        raise ValueError("tabulated-form-factor requires form_factor.")
    factor = np.asarray(values, dtype=np.float64)
    if factor.shape != (count,):
        raise ValueError("form_factor must hold one magnitude per angular frequency.")
    if not np.all(np.isfinite(factor)) or np.any(factor < 0.0) or np.any(factor > 1.0):
        raise ValueError("form_factor magnitudes must lie in [0, 1].")
    return factor


def _bunch_sigma(
    coherence: RadiationCoherence, values: ArrayLike | None, /
) -> np.ndarray | None:
    if coherence != "gaussian-form-factor":
        if values is not None:
            raise ValueError("bunch_sigma is accepted only by gaussian-form-factor.")
        return None
    if values is None:
        raise ValueError("gaussian-form-factor requires bunch_sigma.")
    sigma = np.asarray(values, dtype=np.float64)
    if sigma.shape != (3,) or not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("bunch_sigma must be three finite nonnegative rms sizes.")
    return sigma


def _observer_window(
    route: TrajectoryRadiationRoute, window: tuple[float, float] | None, /
) -> tuple[float, float] | None:
    if route != "node-gridded":
        if window is not None:
            raise ValueError("observer_time_window is accepted only by node-gridded.")
        return None
    if window is None:
        raise ValueError("node-gridded requires observer_time_window=(lower, upper).")
    if len(window) != 2:
        raise ValueError("observer_time_window must be (lower, upper).")
    lower = finite_real_scalar(window[0], "observer_time_window[0]")
    upper = finite_real_scalar(window[1], "observer_time_window[1]")
    if not lower < upper:
        raise ValueError("observer_time_window must satisfy lower < upper.")
    return (lower, upper)


class TrajectoryRadiationEvidence(StrictModule):
    """Finiteness, sampling adequacy, and resource evidence of one spectrum.

    ``status`` holds ``TrajectoryRadiationStatus`` bits. ``resolved`` is true
    when the sampling resolves phase and amplitude, no node fell outside a
    gridded box, lane times increase, and (for declared complete emission) the
    window edges carry no acceleration. ``derivative_valid[F, D]`` is false where
    the field is nonfinite, at frequencies whose phase increment is unresolved,
    and everywhere when an activity transition (a discrete event) occurred.
    ``gridded_error_floor[D]`` is the absolute field error bound
    ``tolerance · Σ_j |Δa_j|`` of the Type-3 route.
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    finite: Bool[Scalar]
    resolved: Bool[Scalar]
    derivative_valid: Bool[_FrequencyDim, _DirectionDim]
    minimum_retardation_factor: Float64[Scalar]
    maximum_phase_increment: Float64[Scalar]
    maximum_relative_amplitude_increment: Float64[Scalar]
    window_edge_rate: Float64[Scalar]
    segments_used: Int32[Scalar]
    gridded_error_floor: Float64[_DirectionDim] | None
    resource_estimate: TrajectoryRadiationResourceEstimate = eqx.field(static=True)
    gridded: NonuniformFourierGridEvidence | None = eqx.field(static=True)


class TrajectoryRadiationResult(StrictModule):
    """Far-field spectrum of one trajectory set.

    ``field_spectrum[F, D, 2]`` is the coherent lane sum of ``r Ẽ`` in the
    observer basis ``(e1, e2)``; ``coherency[F, D, 2, 2]`` is ``⟨R_i R_j*⟩``
    under the plan's coherence model, ``spectral_energy = ε₀ c tr(J) / π`` and
    ``stokes = (I, Q, U, V)`` with ``U = 2 Re J₁₂`` and ``V = −2 Im J₁₂``.
    """

    __strict_contract__ = True

    field_spectrum: Complex128[_FrequencyDim, _DirectionDim, Literal[2]]
    spectral_energy: Float64[_FrequencyDim, _DirectionDim]
    coherency: Complex128[_FrequencyDim, _DirectionDim, Literal[2], Literal[2]]
    stokes: Float64[_FrequencyDim, _DirectionDim, Literal[4]]
    angular_frequencies: Float64[_FrequencyDim]
    directions: Float64[_DirectionDim, Literal[3]]
    evidence: TrajectoryRadiationEvidence
    emission: SpectralEmissionCompleteness = eqx.field(static=True)
    coherence: RadiationCoherence = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


class TrajectoryRadiationState(StrictModule):
    """Streaming accumulator between ``accumulate`` calls.

    ``coherent_field`` holds the multiplicity-weighted lane sum, and
    ``particle_fields`` the per-lane spectra required by every coherence model
    except ``"coherent"``. The carried node and segment let the next chunk
    continue each lane exactly; the remaining leaves accumulate evidence.
    """

    __strict_contract__ = True

    coherent_field: Complex128[_FrequencyDim, _DirectionDim, Literal[2]]
    particle_fields: (
        Complex128[_ParticleDim, _FrequencyDim, _DirectionDim, Literal[2]] | None
    )
    node_time: Float64[_ParticleDim]
    node_position: Float64[_ParticleDim, Literal[3]]
    node_proper_velocity: Float64[_ParticleDim, Literal[3]]
    node_proper_acceleration: Float64[_ParticleDim, Literal[3]] | None
    node_active: Bool[_ParticleDim]
    segment_amplitude: Float64[_ParticleDim, _DirectionDim, Literal[2]]
    segment_step: Float64[_ParticleDim]
    segment_active: Bool[_ParticleDim]
    charges: Float64[_ParticleDim]
    multiplicities: Float64[_ParticleDim]
    id_hi: UInt32[_ParticleDim]
    id_lo: UInt32[_ParticleDim]
    started: Bool[Scalar]
    edge_recorded: Bool[_ParticleDim]
    first_jump_rate: Float64[_ParticleDim]
    last_jump_rate: Float64[_ParticleDim]
    maximum_jump_rate: Float64[_ParticleDim]
    minimum_retardation: Float64[Scalar]
    maximum_observer_step: Float64[Scalar]
    maximum_amplitude: Float64[Scalar]
    maximum_jump: Float64[Scalar]
    segments_used: Int32[Scalar]
    activity_transition: Bool[Scalar]
    nonmonotone: Bool[Scalar]
    unsupported: Bool[Scalar]
    lane_mismatch: Bool[Scalar]
    gridded_floor: Float64[_DirectionDim]
    sample_capacity: int = eqx.field(static=True)


class _Observers(NamedTuple):
    directions: Array
    first: Array
    second: Array


class _LaneNodes(NamedTuple):
    times: Array
    positions: Array
    proper_velocities: Array
    proper_accelerations: Array | None
    active: Array


class _LaneCarry(NamedTuple):
    segment_amplitude: Array
    segment_step: Array
    segment_active: Array
    edge_recorded: Array


class _LaneEvidence(NamedTuple):
    minimum_retardation: Array
    maximum_observer_step: Array
    maximum_amplitude: Array
    maximum_jump: Array
    first_jump_rate: Array
    last_jump_rate: Array
    maximum_jump_rate: Array
    edge_recorded: Array
    segments_used: Array
    activity_transition: Array
    nonmonotone: Array
    unsupported: Array
    gridded_floor: Array


class _Segments(NamedTuple):
    """Per-segment geometry of one lane, segment ``i`` joining nodes ``i, i+1``."""

    tau_start: Array
    tau_end: Array
    delta_tau: Array
    delta_t: Array
    kappa: Array
    amplitude: Array
    active: Array


def _lorentz_split(proper: Array, speed_of_light: float, /) -> tuple[Array, Array]:
    """Return ``(1/γ², β)`` from proper velocities ``[..., 3]``."""
    inverse_gamma_sq = 1.0 / (1.0 + jnp.sum(proper * proper, axis=-1) / speed_of_light**2)
    beta = proper * jnp.sqrt(inverse_gamma_sq)[..., None] / speed_of_light
    return inverse_gamma_sq, beta


def _retardation(
    inverse_gamma_sq: Array, beta: Array, observers: _Observers, /
) -> tuple[Array, Array]:
    """Return ``(κ, a)`` for ``β[..., 3]``: ``κ[..., D]`` and ``a[..., D, 2]``.

    ``κ = (1/γ² + |n × β|²) / (1 + n·β)`` avoids the cancellation of ``1 − n·β``
    at large γ; ``a = n × (n × β) / κ = −β_⊥ / κ`` in the observer basis.
    """
    n = observers.directions
    parallel = beta @ n.T
    cross = jnp.cross(n[None, :, :], beta[..., None, :])
    transverse_sq = jnp.sum(cross * cross, axis=-1)
    kappa = (inverse_gamma_sq[..., None] + transverse_sq) / (1.0 + parallel)
    amplitude = (
        -jnp.stack((beta @ observers.first.T, beta @ observers.second.T), axis=-1)
        / kappa[..., None]
    )
    return kappa, amplitude


def _segment_geometry(
    nodes: _LaneNodes, observers: _Observers, speed_of_light: float, /
) -> _Segments:
    tau = (
        nodes.times[:, None] - (nodes.positions @ observers.directions.T) / speed_of_light
    )
    delta_t = nodes.times[1:] - nodes.times[:-1]
    midpoint = 0.5 * (nodes.proper_velocities[1:] + nodes.proper_velocities[:-1])
    inverse_gamma_sq, beta = _lorentz_split(midpoint, speed_of_light)
    kappa, amplitude = _retardation(inverse_gamma_sq, beta, observers)
    both_active = nodes.active[1:] & nodes.active[:-1]
    active = both_active & (delta_t > 0.0)
    return _Segments(
        tau[:-1], tau[1:], tau[1:] - tau[:-1], delta_t, kappa, amplitude, active
    )


def _pad_segments(count: int, block: int, *arrays: Array) -> tuple[Array, ...]:
    padding = (-count) % block
    return tuple(
        jnp.pad(array, ((0, padding),) + ((0, 0),) * (array.ndim - 1)) for array in arrays
    )


def _block_scan(
    body: Callable[..., Array],
    blocks: tuple[Array, ...],
    shape: tuple[int, ...],
    /,
) -> Array:
    def step(total: Array, block: tuple[Array, ...]) -> tuple[Array, None]:
        return total + body(*block), None

    total, _ = jax.lax.scan(step, jnp.zeros(shape, dtype=jnp.complex128), blocks)
    return total


def _segment_exact_field(
    segments: _Segments,
    previous_amplitude: Array,
    previous_active: Array,
    frequencies: Array,
    block: int,
    /,
) -> Array:
    """Velocity form ``−iω Σ_j a_j Δτ_j e^{iωτ_mid} sinc(ωΔτ_j/2)`` plus boundary terms.

    A lane that starts radiating at segment ``i`` contributes
    ``−a_i e^{iωτ_i}`` and one that stops contributes ``+a_{i−1} e^{iωτ_i}``,
    which turns the velocity form into the acceleration (jump) form. Every
    chunk closes its last active segment with ``+a e^{iωτ_end}``; the next chunk
    reopens a carried active segment with ``−a e^{iωτ_start}``, so chunks
    compose exactly and no term is left pending.
    """
    count = segments.active.shape[0]
    shifted_active = jnp.concatenate((previous_active[None], segments.active[:-1]))
    shifted_amplitude = jnp.concatenate(
        (previous_amplitude[None], segments.amplitude[:-1])
    )
    start = segments.active & ~shifted_active
    end = ~segments.active & shifted_active
    node_coefficient = (
        -segments.amplitude * start[:, None, None]
        + shifted_amplitude * end[:, None, None]
    )
    node_coefficient = node_coefficient.at[0].add(-previous_amplitude * previous_active)
    closing = segments.amplitude[-1] * segments.active[-1]
    weighted = (
        segments.amplitude * (segments.delta_tau * segments.active[:, None])[:, :, None]
    )
    tau_mid = 0.5 * (segments.tau_start + segments.tau_end)
    blocks = _pad_segments(
        count,
        block,
        tau_mid,
        segments.delta_tau,
        weighted,
        segments.tau_start,
        node_coefficient,
    )
    blocks = tuple(array.reshape((-1, block, *array.shape[1:])) for array in blocks)
    minus_i_omega = -1j * frequencies.astype(jnp.complex128)

    def body(
        mid: Array,
        delta: Array,
        strength: Array,
        tau_node: Array,
        coefficient: Array,
    ) -> Array:
        phase_mid = jnp.exp(1j * frequencies[:, None, None] * mid[None])
        kernel = phase_mid * jnp.sinc(
            frequencies[:, None, None] * delta[None] / (2.0 * math.pi)
        )
        velocity = contract("f,fbd,bdc->fdc", minus_i_omega, kernel, strength)
        phase_node = jnp.exp(1j * frequencies[:, None, None] * tau_node[None])
        return velocity + contract("fbd,bdc->fdc", phase_node, coefficient)

    shape = (frequencies.shape[0], segments.kappa.shape[1], 2)
    closing_phase = jnp.exp(1j * frequencies[:, None] * segments.tau_end[-1][None, :])
    return _block_scan(body, blocks, shape) + closing_phase[..., None] * closing[None]


def _node_jumps(
    segments: _Segments, previous_amplitude: Array, previous_active: Array, /
) -> tuple[Array, Array]:
    """Return jump strengths ``Δa_i[K, D, 2]`` at node ``i`` and their activity."""
    shifted_active = jnp.concatenate((previous_active[None], segments.active[:-1]))
    shifted_amplitude = jnp.concatenate(
        (previous_amplitude[None], segments.amplitude[:-1])
    )
    jump_active = segments.active & shifted_active
    return (segments.amplitude - shifted_amplitude) * jump_active[:, None, None], (
        jump_active
    )


def _node_direct_field(
    tau: Array, jumps: Array, frequencies: Array, block: int, /
) -> Array:
    count = tau.shape[0]
    tau_b, jumps_b = _pad_segments(count, block, tau, jumps)
    blocks = (
        tau_b.reshape((-1, block, tau.shape[1])),
        jumps_b.reshape((-1, block, *jumps.shape[1:])),
    )

    def body(tau_node: Array, strength: Array) -> Array:
        phase = jnp.exp(1j * frequencies[:, None, None] * tau_node[None])
        return contract("fbd,bdc->fdc", phase, strength)

    return _block_scan(body, blocks, (frequencies.shape[0], tau.shape[1], 2))


def _node_gridded_field(
    prepared: PreparedNonuniformFourierType3,
    tau: Array,
    jumps: Array,
    jump_active: Array,
    frequencies: Array,
    /,
) -> tuple[Array, Array]:
    targets = frequencies[:, None]
    # Nodes without a jump carry zero strength; placing them at the box center
    # keeps the support evidence about real sources only.
    center = prepared.source_center[0]
    sources = jnp.where(jump_active[:, None], tau, center)

    def one_direction(points: Array, strengths: Array) -> tuple[Array, Array]:
        result = prepared.apply(
            points[:, None], strengths.astype(jnp.complex128), targets
        )
        return result.values, jnp.all(result.supported)

    values, supported = jax.vmap(one_direction, in_axes=(1, 1))(sources, jumps)
    return jnp.swapaxes(values, 0, 1), ~jnp.all(supported)


class _HermiteRule(NamedTuple):
    nodes: Array
    weights: Array


def _hermite_basis(
    s: Array, /
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    """Quintic Hermite basis ``(H, H', H'')`` on ``[0, 1]`` for ``(p0, v0, a0, p1, v1, a1)``."""
    s2, s3, s4, s5 = s * s, s**3, s**4, s**5
    value = (
        1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5,
        s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5,
        0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5),
        10.0 * s3 - 15.0 * s4 + 6.0 * s5,
        -4.0 * s3 + 7.0 * s4 - 3.0 * s5,
        0.5 * (s3 - 2.0 * s4 + s5),
    )
    first = (
        -30.0 * s2 + 60.0 * s3 - 30.0 * s4,
        1.0 - 18.0 * s2 + 32.0 * s3 - 15.0 * s4,
        0.5 * (2.0 * s - 9.0 * s2 + 12.0 * s3 - 5.0 * s4),
        30.0 * s2 - 60.0 * s3 + 30.0 * s4,
        -12.0 * s2 + 28.0 * s3 - 15.0 * s4,
        0.5 * (3.0 * s2 - 8.0 * s3 + 5.0 * s4),
    )
    second = (
        -60.0 * s + 180.0 * s2 - 120.0 * s3,
        -36.0 * s + 96.0 * s2 - 60.0 * s3,
        0.5 * (2.0 - 18.0 * s + 36.0 * s2 - 20.0 * s3),
        60.0 * s - 180.0 * s2 + 120.0 * s3,
        -24.0 * s + 84.0 * s2 - 60.0 * s3,
        0.5 * (6.0 * s - 24.0 * s2 + 20.0 * s3),
    )
    return value, first, second


def _cubic_basis(s: Array, /) -> tuple[Array, Array, Array, Array]:
    s2, s3 = s * s, s**3
    return (
        1.0 - 3.0 * s2 + 2.0 * s3,
        s - 2.0 * s2 + s3,
        3.0 * s2 - 2.0 * s3,
        -s2 + s3,
    )


def _segment_hermite_field(
    nodes: _LaneNodes,
    segments: _Segments,
    observers: _Observers,
    rule: _HermiteRule,
    frequencies: Array,
    speed_of_light: float,
    block: int,
    /,
) -> Array:
    """Acceleration form on a quintic Hermite reconstruction of each segment."""
    if nodes.proper_accelerations is None:
        raise ValueError("segment-hermite requires proper_accelerations.")
    c = speed_of_light
    inverse_gamma_sq, _ = _lorentz_split(nodes.proper_velocities, c)
    inverse_gamma = jnp.sqrt(inverse_gamma_sq)[:, None]
    u, u_dot = nodes.proper_velocities, nodes.proper_accelerations
    velocity = u * inverse_gamma
    # d/dt (u / γ) = u̇ / γ − u (u·u̇) / (c² γ³)
    acceleration = (
        u_dot * inverse_gamma
        - u * jnp.sum(u * u_dot, axis=-1)[:, None] * inverse_gamma**3 / c**2
    )
    h = segments.delta_t
    count = h.shape[0]
    s = 0.5 * (1.0 + rule.nodes)
    hv, hd, hdd = _hermite_basis(s)
    cv = _cubic_basis(s)
    ends = (
        nodes.positions[:-1],
        h[:, None] * velocity[:-1],
        h[:, None] ** 2 * acceleration[:-1],
        nodes.positions[1:],
        h[:, None] * velocity[1:],
        h[:, None] ** 2 * acceleration[1:],
    )
    proper_ends = (u[:-1], h[:, None] * u_dot[:-1], u[1:], h[:, None] * u_dot[1:])

    def reconstruct(basis: tuple[Array, ...], data: tuple[Array, ...]) -> Array:
        return sum(
            (b[None, :, None] * d[:, None, :] for b, d in zip(basis, data, strict=True)),
            start=jnp.zeros((count, s.shape[0], 3), dtype=jnp.float64),
        )

    position = reconstruct(hv, ends)
    safe_h = jnp.where(h > 0.0, h, 1.0)[:, None, None]
    beta = reconstruct(hd, ends) / (safe_h * c)
    beta_dot = reconstruct(hdd, ends) / (safe_h**2 * c)
    proper = reconstruct(cv, proper_ends)
    inverse_gamma_sq_q = 1.0 / (1.0 + jnp.sum(proper * proper, axis=-1) / c**2)
    kappa, _ = _retardation(inverse_gamma_sq_q, beta, observers)
    n_dot_beta_dot = beta_dot @ observers.directions.T
    beta_t = jnp.stack((beta @ observers.first.T, beta @ observers.second.T), axis=-1)
    beta_dot_t = jnp.stack(
        (beta_dot @ observers.first.T, beta_dot @ observers.second.T), axis=-1
    )
    # d/dt [−β_⊥ / κ] = −(β̇_⊥ κ + β_⊥ (n·β̇)) / κ²
    integrand = -(beta_dot_t * kappa[..., None] + beta_t * n_dot_beta_dot[..., None]) / (
        kappa[..., None] ** 2
    )
    times = nodes.times[:-1, None] + h[:, None] * s[None, :]
    tau = times[:, :, None] - (position @ observers.directions.T) / c
    weight = (0.5 * h * segments.active)[:, None] * rule.weights[None, :]
    strength = integrand * weight[:, :, None, None]
    tau_b, strength_b = _pad_segments(count, block, tau, strength)
    blocks = (
        tau_b.reshape((-1, block, *tau.shape[1:])),
        strength_b.reshape((-1, block, *strength.shape[1:])),
    )

    def body(tau_block: Array, strength_block: Array) -> Array:
        phase = jnp.exp(1j * frequencies[:, None, None, None] * tau_block[None])
        return contract("fbqd,bqdc->fdc", phase, strength_block)

    return _block_scan(body, blocks, (frequencies.shape[0], kappa.shape[-1], 2))


def _lane_evidence(
    nodes: _LaneNodes,
    segments: _Segments,
    jumps: Array,
    jump_active: Array,
    carry: _LaneCarry,
    started: Array,
    unsupported: Array,
    /,
) -> _LaneEvidence:
    active = segments.active
    active_d = active[:, None]
    kappa_min = jnp.min(jnp.where(active_d, segments.kappa, jnp.inf))
    step_max = jnp.max(jnp.where(active_d, segments.delta_tau, 0.0))
    amplitude_norm = jnp.max(jnp.linalg.norm(segments.amplitude, axis=-1), axis=1)
    amplitude_max = jnp.max(jnp.where(active, amplitude_norm, 0.0))
    jump_norm = jnp.max(jnp.linalg.norm(jumps, axis=-1), axis=1)
    jump_max = jnp.max(jump_norm)
    previous_step = jnp.concatenate((carry.segment_step[None], segments.delta_t[:-1]))
    rate = jnp.where(
        jump_active, jump_norm / (0.5 * (previous_step + segments.delta_t)), 0.0
    )
    any_jump = jnp.any(jump_active)
    count = jump_active.shape[0]
    index = jnp.arange(count)
    first_index = jnp.min(jnp.where(jump_active, index, count))
    last_index = jnp.max(jnp.where(jump_active, index, -1))
    first_rate = rate[jnp.clip(first_index, 0, count - 1)]
    last_rate = rate[jnp.clip(last_index, 0, count - 1)]
    record_first = any_jump & ~carry.edge_recorded
    pair_valid = started | (jnp.arange(count) >= 1)
    transition = jnp.any(pair_valid & (nodes.active[1:] != nodes.active[:-1]))
    nonmonotone = jnp.any(
        nodes.active[1:] & nodes.active[:-1] & (segments.delta_t <= 0.0)
    )
    floor = jnp.sum(jnp.linalg.norm(jumps, axis=-1), axis=0)
    return _LaneEvidence(
        kappa_min,
        step_max,
        amplitude_max,
        jump_max,
        jnp.where(record_first, first_rate, jnp.nan),
        jnp.where(any_jump, last_rate, jnp.nan),
        jnp.max(rate),
        carry.edge_recorded | any_jump,
        jnp.sum(active, dtype=jnp.int32),
        transition,
        nonmonotone,
        unsupported,
        floor,
    )


def _map_lanes(
    lane: Callable[[tuple[Array, ...]], tuple[Array, _LaneCarry, _LaneEvidence]],
    inputs: tuple[Array, ...],
    multiplicities: Array,
    chunk: int,
    keep_fields: bool,
    /,
) -> tuple[Array, Array | None, _LaneCarry, _LaneEvidence]:
    """Run ``lane`` over particle lanes in ``lax.scan`` chunks of ``chunk`` lanes.

    The multiplicity-weighted field sum is carried through the scan, so the
    coherent model never holds per-lane spectra; the other models also return
    them. Padding lanes carry zero weight and are sliced away.
    """
    count = multiplicities.shape[0]
    padding = (-count) % chunk
    padded = tuple(
        jnp.pad(array, ((0, padding),) + ((0, 0),) * (array.ndim - 1))
        for array in (*inputs, multiplicities)
    )
    chunked = tuple(array.reshape((-1, chunk, *array.shape[1:])) for array in padded)

    def body(
        total: Array, block: tuple[Array, ...]
    ) -> tuple[Array, tuple[Array | None, _LaneCarry, _LaneEvidence]]:
        fields, carries, evidence = jax.vmap(lane)(block[:-1])
        total = total + contract("p,pfdc->fdc", block[-1], fields)
        return total, (fields if keep_fields else None, carries, evidence)

    shape = jax.eval_shape(lambda: lane(tuple(array[0] for array in inputs)))[0].shape
    total, (fields, carries, evidence) = jax.lax.scan(
        body, jnp.zeros(shape, dtype=jnp.complex128), chunked
    )

    def unchunk(array: Array) -> Array:
        return array.reshape((-1, *array.shape[2:]))[:count]

    return (
        total,
        None if fields is None else unchunk(fields),
        jax.tree.map(unchunk, carries),
        jax.tree.map(unchunk, evidence),
    )


class PreparedTrajectoryRadiation(StrictModule, NonTrainableState):
    """Prepared constants, quadrature, and gridded transform of one plan.

    ``evaluate`` runs ``initialize``/``accumulate``/``finalize`` on the whole
    trajectory; streaming callers pass consecutive time chunks of the same lanes
    to ``accumulate``. All numerical leaves are float64/complex128.
    """

    __strict_contract__ = True

    plan: TrajectoryRadiationPlan
    prefactor: Float64[Scalar]
    energy_factor: Float64[Scalar]
    hermite_rule: _HermiteRule | None
    type3: PreparedNonuniformFourierType3 | None
    speed_of_light: float = eqx.field(static=True)
    gridded_direct: bool = eqx.field(static=True)

    def __init__(self, plan: TrajectoryRadiationPlan, /) -> None:
        if not isinstance(plan, TrajectoryRadiationPlan):
            raise TypeError("plan must be a TrajectoryRadiationPlan.")
        scale = plan.scale
        c = float(scale.speed_of_light)
        permittivity = float(scale.vacuum_permittivity)
        self.plan = plan
        self.speed_of_light = c
        self.prefactor = jnp.asarray(1.0 / (4.0 * math.pi * permittivity * c))
        self.energy_factor = jnp.asarray(permittivity * c / math.pi)
        match plan.route:
            case "segment-hermite":
                rule = legendre_rule_data(
                    plan.quadrature_order, "gauss", dtype=jnp.float64
                )
                self.hermite_rule = _HermiteRule(rule.nodes, rule.weights)
            case "segment-exact" | "node-gridded":
                self.hermite_rule = None
            case _:
                assert_never(plan.route)
        frequencies = np.asarray(plan.angular_frequencies)
        window = plan.observer_time_window
        if plan.route == "node-gridded" and window is not None:
            lower, upper = window
            duration = upper - lower
            self.gridded_direct = float(frequencies[-1]) * duration <= 1.0
            if self.gridded_direct:
                self.type3 = None
            else:
                omega_center = 0.5 * float(frequencies[0] + frequencies[-1])
                omega_half = max(
                    0.5 * float(frequencies[-1] - frequencies[0]),
                    1.0e-3 * float(frequencies[-1]),
                )
                type3_plan = NonuniformFourierType3Plan(
                    [0.5 * (lower + upper)],
                    [0.5 * duration],
                    [omega_center],
                    [omega_half],
                    tolerance=plan.resources.gridded_tolerance,
                    sign=1,
                    maximum_grid_points=plan.resources.maximum_grid_points,
                )
                self.type3 = PreparedNonuniformFourierType3(type3_plan, dtype=jnp.float64)
        else:
            self.gridded_direct = False
            self.type3 = None

    @property
    def gridded_evidence(self) -> NonuniformFourierGridEvidence | None:
        return None if self.type3 is None else self.type3.evidence

    def _observers(self) -> _Observers:
        observers = self.plan.observers
        return _Observers(
            observers.directions, observers.basis_first, observers.basis_second
        )

    def _quadrature_points(self) -> int:
        return 1 if self.hermite_rule is None else self.hermite_rule.nodes.shape[0]

    def resource_estimate(
        self, particle_count: int, sample_count: int, /
    ) -> TrajectoryRadiationResourceEstimate:
        """Estimate one ``accumulate`` call on ``sample_count`` samples per lane."""
        plan = self.plan
        policy = plan.resources
        frequencies = plan.frequency_count
        directions = plan.observers.direction_count
        quadrature = self._quadrature_points()
        lanes = min(policy.particle_chunk, particle_count)
        spectrum = frequencies * directions * 2 * _COMPLEX_ITEMSIZE
        channels = _NODE_CHANNELS + (
            _HERMITE_CHANNELS * quadrature if plan.route == "segment-hermite" else 0
        )
        geometry = (sample_count + 1) * directions * channels * _REAL_ITEMSIZE
        if self.type3 is None:
            # Phase and kernel tables of one block, [F, block, Q, D] each.
            transform = (
                2
                * frequencies
                * policy.segment_block
                * quadrature
                * directions
                * _COMPLEX_ITEMSIZE
            )
        else:
            # Two payload channels per direction on the Type-3 grids.
            transform = directions * 2 * self.type3.evidence.grid_bytes
        working = lanes * (geometry + transform + spectrum)
        state = spectrum
        if plan.coherence != "coherent":
            state += particle_count * spectrum
        return TrajectoryRadiationResourceEstimate(
            working,
            state,
            policy.maximum_working_bytes,
            policy.maximum_state_bytes,
            particle_count,
            sample_count,
            lanes,
            policy.segment_block,
            quadrature,
        )

    def _refuse(
        self, particle_count: int, sample_count: int, /
    ) -> TrajectoryRadiationResourceEstimate:
        estimate = self.resource_estimate(particle_count, sample_count)
        if estimate.working_bytes > estimate.maximum_working_bytes:
            raise TrajectoryRadiationResourceError(
                f"Trajectory radiation needs {estimate.working_bytes} working bytes "
                f"per particle chunk, above maximum_working_bytes="
                f"{estimate.maximum_working_bytes}."
            )
        if estimate.state_bytes > estimate.maximum_state_bytes:
            raise TrajectoryRadiationResourceError(
                f"Trajectory radiation state needs {estimate.state_bytes} bytes, "
                f"above maximum_state_bytes={estimate.maximum_state_bytes}."
            )
        return estimate

    def initialize(self, trajectory: ChargedTrajectory, /) -> TrajectoryRadiationState:
        """Return the empty state for the lanes of ``trajectory``.

        Only the lane identities, charges, multiplicities, and whether
        accelerations are present are read; samples are consumed by ``accumulate``.
        """
        self._check_trajectory(trajectory)
        self._refuse(trajectory.particle_count, trajectory.sample_count)
        plan = self.plan
        count = trajectory.particle_count
        shape = (plan.frequency_count, plan.observers.direction_count, 2)
        zeros_p = jnp.zeros((count,), dtype=jnp.float64)
        zeros_p3 = jnp.zeros((count, 3), dtype=jnp.float64)
        false_p = jnp.zeros((count,), dtype=jnp.bool_)
        return TrajectoryRadiationState(
            coherent_field=jnp.zeros(shape, dtype=jnp.complex128),
            particle_fields=None
            if plan.coherence == "coherent"
            else jnp.zeros((count, *shape), dtype=jnp.complex128),
            node_time=zeros_p,
            node_position=zeros_p3,
            node_proper_velocity=zeros_p3,
            node_proper_acceleration=None
            if trajectory.proper_accelerations is None
            else zeros_p3,
            node_active=false_p,
            segment_amplitude=jnp.zeros((count, shape[1], 2), dtype=jnp.float64),
            segment_step=zeros_p,
            segment_active=false_p,
            charges=trajectory.charges,
            multiplicities=trajectory.multiplicities,
            id_hi=trajectory.id_hi,
            id_lo=trajectory.id_lo,
            started=jnp.asarray(False),
            edge_recorded=false_p,
            first_jump_rate=zeros_p,
            last_jump_rate=zeros_p,
            maximum_jump_rate=zeros_p,
            minimum_retardation=jnp.asarray(jnp.inf),
            maximum_observer_step=jnp.asarray(0.0),
            maximum_amplitude=jnp.asarray(0.0),
            maximum_jump=jnp.asarray(0.0),
            segments_used=jnp.asarray(0, dtype=jnp.int32),
            activity_transition=jnp.asarray(False),
            nonmonotone=jnp.asarray(False),
            unsupported=jnp.asarray(False),
            lane_mismatch=jnp.asarray(False),
            gridded_floor=jnp.zeros((shape[1],), dtype=jnp.float64),
            sample_capacity=0,
        )

    def _check_trajectory(self, trajectory: ChargedTrajectory, /) -> None:
        if not isinstance(trajectory, ChargedTrajectory):
            raise TypeError("trajectory must be a ChargedTrajectory.")
        if (
            self.plan.route == "segment-hermite"
            and trajectory.proper_accelerations is None
        ):
            raise ValueError("segment-hermite requires proper_accelerations.")

    def _lane_chunk(
        self,
        nodes: _LaneNodes,
        carry: _LaneCarry,
        started: Array,
        /,
    ) -> tuple[Array, _LaneCarry, _LaneEvidence]:
        plan = self.plan
        observers = self._observers()
        frequencies = plan.angular_frequencies
        block = plan.resources.segment_block
        segments = _segment_geometry(nodes, observers, self.speed_of_light)
        jumps, jump_active = _node_jumps(
            segments, carry.segment_amplitude, carry.segment_active
        )
        unsupported = jnp.asarray(False)
        match plan.route:
            case "segment-exact":
                field = _segment_exact_field(
                    segments,
                    carry.segment_amplitude,
                    carry.segment_active,
                    frequencies,
                    block,
                )
            case "segment-hermite":
                if self.hermite_rule is None:
                    raise ValueError("segment-hermite preparation lacks its rule.")
                field = _segment_hermite_field(
                    nodes,
                    segments,
                    observers,
                    self.hermite_rule,
                    frequencies,
                    self.speed_of_light,
                    block,
                )
            case "node-gridded":
                if self.type3 is None:
                    field = _node_direct_field(
                        segments.tau_start, jumps, frequencies, block
                    )
                else:
                    field, unsupported = _node_gridded_field(
                        self.type3, segments.tau_start, jumps, jump_active, frequencies
                    )
            case _:
                assert_never(plan.route)
        evidence = _lane_evidence(
            nodes, segments, jumps, jump_active, carry, started, unsupported
        )
        new_carry = _LaneCarry(
            segments.amplitude[-1],
            segments.delta_t[-1],
            segments.active[-1],
            evidence.edge_recorded,
        )
        return field, new_carry, evidence

    def accumulate(
        self, state: TrajectoryRadiationState, trajectory: ChargedTrajectory, /
    ) -> TrajectoryRadiationState:
        """Fold the next time chunk of the same lanes into ``state``."""
        self._check_trajectory(trajectory)
        if not isinstance(state, TrajectoryRadiationState):
            raise TypeError("state must be a TrajectoryRadiationState.")
        if state.charges.shape[0] != trajectory.particle_count:
            raise ValueError("trajectory lanes must match the state lanes.")
        if (state.node_proper_acceleration is None) != (
            trajectory.proper_accelerations is None
        ):
            raise ValueError("proper_accelerations presence must match the state.")
        plan = self.plan
        prefactor = self.prefactor
        estimate = self._refuse(trajectory.particle_count, trajectory.sample_count)

        def lane(inputs: tuple[Array, ...]) -> tuple[Array, _LaneCarry, _LaneEvidence]:
            (
                times,
                positions,
                proper,
                accelerations,
                active,
                node_time,
                node_position,
                node_proper,
                node_acceleration,
                node_active,
                amplitude,
                step,
                segment_active,
                recorded,
                charge,
            ) = inputs
            nodes = _LaneNodes(
                jnp.concatenate((node_time[None], times)),
                jnp.concatenate((node_position[None], positions)),
                jnp.concatenate((node_proper[None], proper)),
                None
                if plan.route != "segment-hermite"
                else jnp.concatenate((node_acceleration[None], accelerations)),
                jnp.concatenate((node_active[None], active)),
            )
            carry = _LaneCarry(amplitude, step, segment_active, recorded)
            field, new_carry, evidence = self._lane_chunk(nodes, carry, state.started)
            return (prefactor * charge) * field, new_carry, evidence

        accelerations = (
            jnp.zeros_like(trajectory.proper_velocities)
            if trajectory.proper_accelerations is None
            else trajectory.proper_accelerations
        )
        node_acceleration = (
            jnp.zeros_like(state.node_proper_velocity)
            if state.node_proper_acceleration is None
            else state.node_proper_acceleration
        )
        inputs = (
            jnp.swapaxes(trajectory.times, 0, 1),
            jnp.swapaxes(trajectory.positions, 0, 1),
            jnp.swapaxes(trajectory.proper_velocities, 0, 1),
            jnp.swapaxes(accelerations, 0, 1),
            jnp.swapaxes(trajectory.active, 0, 1),
            state.node_time,
            state.node_position,
            state.node_proper_velocity,
            node_acceleration,
            state.node_active,
            state.segment_amplitude,
            state.segment_step,
            state.segment_active,
            state.edge_recorded,
            state.charges,
        )
        weighted, fields, carries, evidence = _map_lanes(
            lane,
            inputs,
            state.multiplicities,
            estimate.particle_chunk,
            state.particle_fields is not None,
        )
        last = trajectory.sample_count - 1
        lane_mismatch = (
            jnp.any(trajectory.id_hi != state.id_hi)
            | jnp.any(trajectory.id_lo != state.id_lo)
            | jnp.any(trajectory.charges != state.charges)
            | jnp.any(trajectory.multiplicities != state.multiplicities)
        )
        first_rate = jnp.where(
            jnp.isnan(evidence.first_jump_rate),
            state.first_jump_rate,
            evidence.first_jump_rate,
        )
        last_rate = jnp.where(
            jnp.isnan(evidence.last_jump_rate),
            state.last_jump_rate,
            evidence.last_jump_rate,
        )
        return TrajectoryRadiationState(
            coherent_field=state.coherent_field + weighted,
            particle_fields=None
            if state.particle_fields is None or fields is None
            else state.particle_fields + fields,
            node_time=trajectory.times[last],
            node_position=trajectory.positions[last],
            node_proper_velocity=trajectory.proper_velocities[last],
            node_proper_acceleration=None
            if trajectory.proper_accelerations is None
            else trajectory.proper_accelerations[last],
            node_active=trajectory.active[last],
            segment_amplitude=carries.segment_amplitude,
            segment_step=carries.segment_step,
            segment_active=carries.segment_active,
            charges=state.charges,
            multiplicities=state.multiplicities,
            id_hi=state.id_hi,
            id_lo=state.id_lo,
            started=jnp.asarray(True),
            edge_recorded=carries.edge_recorded,
            first_jump_rate=first_rate,
            last_jump_rate=last_rate,
            maximum_jump_rate=jnp.maximum(
                state.maximum_jump_rate, evidence.maximum_jump_rate
            ),
            minimum_retardation=jnp.minimum(
                state.minimum_retardation, jnp.min(evidence.minimum_retardation)
            ),
            maximum_observer_step=jnp.maximum(
                state.maximum_observer_step, jnp.max(evidence.maximum_observer_step)
            ),
            maximum_amplitude=jnp.maximum(
                state.maximum_amplitude, jnp.max(evidence.maximum_amplitude)
            ),
            maximum_jump=jnp.maximum(state.maximum_jump, jnp.max(evidence.maximum_jump)),
            segments_used=state.segments_used
            + jnp.sum(evidence.segments_used, dtype=jnp.int32),
            activity_transition=state.activity_transition
            | jnp.any(evidence.activity_transition),
            nonmonotone=state.nonmonotone | jnp.any(evidence.nonmonotone),
            unsupported=state.unsupported | jnp.any(evidence.unsupported),
            lane_mismatch=state.lane_mismatch | lane_mismatch,
            gridded_floor=state.gridded_floor
            + contract(
                "p,pd->d",
                jnp.abs(state.multiplicities * state.charges) * prefactor,
                evidence.gridded_floor,
            ),
            sample_capacity=max(state.sample_capacity, trajectory.sample_count),
        )

    def finalize(self, state: TrajectoryRadiationState, /) -> TrajectoryRadiationResult:
        """Close every lane and combine lanes under the coherence model."""
        if not isinstance(state, TrajectoryRadiationState):
            raise TypeError("state must be a TrajectoryRadiationState.")
        plan = self.plan
        coherent = state.coherent_field
        coherency = _coherency(plan, state, coherent, state.particle_fields)
        evidence = self._evidence(state, coherent, coherency)
        intensity = jnp.real(coherency[..., 0, 0] + coherency[..., 1, 1])
        stokes = jnp.stack(
            (
                intensity,
                jnp.real(coherency[..., 0, 0] - coherency[..., 1, 1]),
                2.0 * jnp.real(coherency[..., 0, 1]),
                -2.0 * jnp.imag(coherency[..., 0, 1]),
            ),
            axis=-1,
        )
        return TrajectoryRadiationResult(
            field_spectrum=coherent,
            spectral_energy=self.energy_factor * intensity,
            coherency=coherency,
            stokes=stokes,
            angular_frequencies=plan.angular_frequencies,
            directions=plan.observers.directions,
            evidence=evidence,
            emission=plan.emission,
            coherence=plan.coherence,
            plan_id=plan.plan_id,
        )

    def _evidence(
        self, state: TrajectoryRadiationState, field: Array, coherency: Array, /
    ) -> TrajectoryRadiationEvidence:
        plan = self.plan
        frequencies = plan.angular_frequencies
        finite_fd = jnp.all(jnp.isfinite(field), axis=-1) & jnp.all(
            jnp.isfinite(coherency), axis=(-2, -1)
        )
        finite = jnp.all(finite_fd)
        phase_limit = (
            float(self._quadrature_points())
            if plan.route == "segment-hermite"
            else _PHASE_INCREMENT_LIMIT
        )
        phase_increment = frequencies * state.maximum_observer_step
        maximum_phase = phase_increment[-1]
        unresolved_phase = maximum_phase > phase_limit
        safe_amplitude = jnp.where(
            state.maximum_amplitude > 0.0, state.maximum_amplitude, 1.0
        )
        amplitude_increment = state.maximum_jump / safe_amplitude
        unresolved_amplitude = amplitude_increment > _AMPLITUDE_INCREMENT_LIMIT
        safe_rate = jnp.where(state.maximum_jump_rate > 0.0, state.maximum_jump_rate, 1.0)
        # Jumps at roundoff level of the amplitude are uniform motion, whose
        # edge "rates" are noise.
        accelerating = state.maximum_jump > _ROUNDOFF_JUMP * safe_amplitude
        edge_rate = jnp.where(
            accelerating,
            jnp.max(
                jnp.where(
                    state.edge_recorded,
                    jnp.maximum(state.first_jump_rate, state.last_jump_rate) / safe_rate,
                    0.0,
                )
            ),
            0.0,
        )
        window_edge = edge_rate > _WINDOW_EDGE_LIMIT
        status = (
            jnp.where(~finite, TrajectoryRadiationStatus.NONFINITE, 0)
            | jnp.where(unresolved_phase, TrajectoryRadiationStatus.UNRESOLVED_PHASE, 0)
            | jnp.where(
                unresolved_amplitude, TrajectoryRadiationStatus.UNRESOLVED_AMPLITUDE, 0
            )
            | jnp.where(
                window_edge, TrajectoryRadiationStatus.WINDOW_EDGE_ACCELERATION, 0
            )
            | jnp.where(
                state.activity_transition,
                TrajectoryRadiationStatus.ACTIVITY_TRANSITION,
                0,
            )
            | jnp.where(state.unsupported, TrajectoryRadiationStatus.UNSUPPORTED_NODE, 0)
            | jnp.where(state.nonmonotone, TrajectoryRadiationStatus.NONMONOTONE_TIME, 0)
            | jnp.where(state.lane_mismatch, TrajectoryRadiationStatus.LANE_MISMATCH, 0)
        ).astype(jnp.int32)
        edge_blocks = window_edge & (plan.emission == "complete")
        resolved = (
            finite
            & ~unresolved_phase
            & ~unresolved_amplitude
            & ~edge_blocks
            & ~state.unsupported
            & ~state.nonmonotone
            & ~state.lane_mismatch
        )
        derivative_valid = (
            finite_fd
            & (phase_increment <= phase_limit)[:, None]
            & ~state.activity_transition
            & ~state.nonmonotone
            & ~state.lane_mismatch
        )
        gridded = self.gridded_evidence
        return TrajectoryRadiationEvidence(
            status=status,
            finite=finite,
            resolved=resolved,
            derivative_valid=derivative_valid,
            minimum_retardation_factor=state.minimum_retardation,
            maximum_phase_increment=maximum_phase,
            maximum_relative_amplitude_increment=amplitude_increment,
            window_edge_rate=edge_rate,
            segments_used=state.segments_used,
            gridded_error_floor=None
            if gridded is None
            else gridded.requested_tolerance * state.gridded_floor,
            resource_estimate=self.resource_estimate(
                state.charges.shape[0], state.sample_capacity
            ),
            gridded=gridded,
        )

    def evaluate(self, trajectory: ChargedTrajectory, /) -> TrajectoryRadiationResult:
        """Radiate the whole trajectory offline."""
        state = self.initialize(trajectory)
        return self.finalize(self.accumulate(state, trajectory))

    def waveform(self, result: TrajectoryRadiationResult, times: ArrayLike, /) -> Array:
        """Observer-time far field ``r E(t)[T, D, 2]`` synthesized from ``field_spectrum``.

        Inverts the one-sided spectrum, ``r E(t) = (1/π) Re ∫₀^∞ F(ω) e^{−iωt} dω``,
        with trapezoid weights on the plan's frequency grid; only the coherent
        model has a field.
        """
        if not isinstance(result, TrajectoryRadiationResult):
            raise TypeError("result must be a TrajectoryRadiationResult.")
        if result.coherence != "coherent":
            raise ValueError("Observer-time waveforms exist only for coherent emission.")
        if result.plan_id != self.plan.plan_id:
            raise ValueError("result was produced by a different plan.")
        sample_times = _float64_array(times, "times")
        if sample_times.ndim != 1:
            raise ValueError("times must be a vector of observer times.")
        frequencies = self.plan.angular_frequencies
        weights = jnp.zeros_like(frequencies)
        if frequencies.shape[0] > 1:
            half = 0.5 * (frequencies[1:] - frequencies[:-1])
            weights = weights.at[:-1].add(half).at[1:].add(half)
        phase = jnp.exp(-1j * sample_times[:, None] * frequencies[None, :])
        return (
            jnp.real(
                contract(
                    "tf,f,fdc->tdc",
                    phase,
                    weights.astype(jnp.complex128),
                    result.field_spectrum,
                )
            )
            / math.pi
        )


def _coherency(
    plan: TrajectoryRadiationPlan,
    state: TrajectoryRadiationState,
    coherent: Array,
    particle_fields: Array | None,
    /,
) -> Array:
    """Return ``J[F, D, 2, 2] = ⟨R_i R_j*⟩`` under the plan's coherence model."""
    outer = contract("fdi,fdj->fdij", coherent, jnp.conj(coherent))
    match plan.coherence:
        case "coherent":
            return outer
        case "incoherent" | "gaussian-form-factor" | "tabulated-form-factor":
            if particle_fields is None:
                raise ValueError("Partially coherent models need per-lane fields.")
            self_terms = contract(
                "p,pfdi,pfdj->fdij",
                state.multiplicities,
                particle_fields,
                jnp.conj(particle_fields),
            )
            match plan.coherence:
                case "incoherent":
                    return self_terms
                case "gaussian-form-factor":
                    if plan.bunch_sigma is None:
                        raise ValueError("gaussian-form-factor lacks bunch_sigma.")
                    spread = (plan.observers.directions**2) @ (plan.bunch_sigma**2)
                    factor_sq = jnp.exp(
                        -(plan.angular_frequencies[:, None] ** 2)
                        * spread[None, :]
                        / float(plan.scale.speed_of_light) ** 2
                    )
                case "tabulated-form-factor":
                    if plan.form_factor is None:
                        raise ValueError("tabulated-form-factor lacks form_factor.")
                    factor_sq = (plan.form_factor**2)[:, None] * jnp.ones(
                        (1, plan.observers.direction_count)
                    )
            weight = factor_sq[..., None, None]
            return (1.0 - weight) * self_terms + weight * outer


__all__ = [
    "ChargedTrajectory",
    "PreparedTrajectoryRadiation",
    "RadiationCoherence",
    "RadiationObserverPlan",
    "TrajectoryRadiationEvidence",
    "TrajectoryRadiationPlan",
    "TrajectoryRadiationResourceError",
    "TrajectoryRadiationResourceEstimate",
    "TrajectoryRadiationResources",
    "TrajectoryRadiationResult",
    "TrajectoryRadiationRoute",
    "TrajectoryRadiationState",
    "TrajectoryRadiationStatus",
]
