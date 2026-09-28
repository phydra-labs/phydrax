#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Far-field trajectory radiation against independent closed-form references.

References: Schott's harmonic decomposition of circular motion (Jackson,
*Classical Electrodynamics*, 3rd ed., problem 14.15; Landau–Lifshitz vol. 2
§74) evaluated with ``phydrax.special.jv``; Liénard's power
``P = q² γ⁶ (|β̇|² − |β × β̇|²) / (6 π ε₀ c)``; the synchrotron tail
``∫ F(x) dx`` from ``phydrax.special.synchrotron_f``; the Liénard–Wiechert
acceleration field; 50-digit ``mpmath`` retardation factors; and the Lorentz
invariance of ``(1/ω²) d²W/(dω dΩ)``.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest

from phydrax import (
    boost_event,
    boost_proper_velocity,
    ElectromagneticScaleContract,
    special,
    transform_spectral_energy,
)
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationCoherence,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
    TrajectoryRadiationResourceError,
    TrajectoryRadiationResources,
    TrajectoryRadiationResult,
    TrajectoryRadiationRoute,
    TrajectoryRadiationStatus,
)


SCALE = ElectromagneticScaleContract.si()
C = float(SCALE.speed_of_light)
EPS0 = float(SCALE.vacuum_permittivity)
Q = float(SCALE.elementary_charge)
OMEGA0 = 1.0e10
PERIOD = 2.0 * np.pi / OMEGA0
SIGMA = 1.0e-9
Z_AXIS = np.array([0.0, 0.0, 1.0])


class Samples:
    """Host samples of one lane: times, positions, proper velocities/accelerations."""

    def __init__(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        proper_velocities: np.ndarray,
        proper_accelerations: np.ndarray,
    ) -> None:
        self.times = times
        self.positions = positions
        self.proper_velocities = proper_velocities
        self.proper_accelerations = proper_accelerations

    def window(self, start: int, stop: int) -> Samples:
        return Samples(
            self.times[start:stop],
            self.positions[start:stop],
            self.proper_velocities[start:stop],
            self.proper_accelerations[start:stop],
        )


def _circular(
    beta: float,
    samples_per_period: int,
    *,
    periods: int = 1,
    offset: float = 0.0,
    extra: int = 0,
) -> Samples:
    """Counterclockwise orbit about +z; nodes at ``(j + offset) dt``."""
    step = PERIOD / samples_per_period
    times = (np.arange(periods * samples_per_period + 1 + extra) + offset) * step
    phase = OMEGA0 * times
    radius = beta * C / OMEGA0
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    zeros = np.zeros_like(times)
    cosine, sine = np.cos(phase), np.sin(phase)
    return Samples(
        times,
        radius * np.stack((cosine, sine, zeros), axis=-1),
        gamma * beta * C * np.stack((-sine, cosine, zeros), axis=-1),
        -gamma * beta * C * OMEGA0 * np.stack((cosine, sine, zeros), axis=-1),
    )


def _periodic_orbit(beta: float, samples_per_period: int, *, periods: int = 1) -> Samples:
    """Whole periods whose piecewise-constant segments are centered on the grid.

    Nodes sit at half steps, ``(j − 1/2) dt`` for ``j = 0..n+1``, so the
    interior node jumps cover exactly the emission times ``[0, periods T₀]``
    and the discrete spectrum at ``m ω₀`` is the Fourier-series coefficient of
    the received field.
    """
    return _circular(beta, samples_per_period, periods=periods, offset=-0.5, extra=1)


def _bump(peak_beta: float, count: int, *, span: float = 8.0) -> Samples:
    """Transverse Gaussian velocity bump ``u_x = U exp(−t²/2σ²)``; complete emission."""
    times = np.linspace(-span * SIGMA, span * SIGMA, count)
    peak = peak_beta / np.sqrt(1.0 - peak_beta**2) * C

    def proper(t: np.ndarray) -> np.ndarray:
        return peak * np.exp(-(t**2) / (2.0 * SIGMA**2))

    def speed(t: np.ndarray) -> np.ndarray:
        u = proper(t)
        return u / np.sqrt(1.0 + (u / C) ** 2)

    nodes, weights = np.polynomial.legendre.leggauss(12)
    lower, upper = times[:-1, None], times[1:, None]
    points = 0.5 * (upper - lower) * nodes[None, :] + 0.5 * (upper + lower)
    increments = 0.5 * (upper[:, 0] - lower[:, 0]) * (speed(points) @ weights)
    x = np.concatenate(([0.0], np.cumsum(increments)))
    u = proper(times)
    zeros = np.zeros_like(times)
    return Samples(
        times,
        np.stack((x, zeros, zeros), axis=-1),
        np.stack((u, zeros, zeros), axis=-1),
        np.stack((-times / SIGMA**2 * u, zeros, zeros), axis=-1),
    )


def _trajectory(
    *lanes: Samples,
    charges: np.ndarray | None = None,
    multiplicities: np.ndarray | None = None,
    active: np.ndarray | None = None,
    identities: tuple[np.ndarray, np.ndarray] | None = None,
    accelerations: bool = True,
    shared_times: bool = False,
) -> ChargedTrajectory:
    count = len(lanes)
    times = (
        lanes[0].times
        if shared_times
        else np.stack([lane.times for lane in lanes], axis=1)
    )
    positions = np.stack([lane.positions for lane in lanes], axis=1)
    return ChargedTrajectory(
        times,
        positions,
        np.stack([lane.proper_velocities for lane in lanes], axis=1),
        np.full(count, Q) if charges is None else charges,
        np.ones(count) if multiplicities is None else multiplicities,
        np.ones(positions.shape[:2], dtype=bool) if active is None else active,
        (
            np.arange(count, dtype=np.uint32),
            np.zeros(count, dtype=np.uint32),
        )
        if identities is None
        else identities,
        proper_accelerations=np.stack(
            [lane.proper_accelerations for lane in lanes], axis=1
        )
        if accelerations
        else None,
    )


def _in_plane_directions(angles: np.ndarray) -> np.ndarray:
    """Directions at polar angle ``angles`` from +z in the x–z plane."""
    return np.stack((np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=-1)


def _evaluate(
    trajectory: ChargedTrajectory,
    directions: np.ndarray,
    frequencies: np.ndarray,
    *,
    route: TrajectoryRadiationRoute = "segment-exact",
    coherence: RadiationCoherence = "coherent",
    emission: Literal["complete", "truncated"] = "complete",
    reference_axis: np.ndarray = Z_AXIS,
    form_factor: np.ndarray | None = None,
    bunch_sigma: np.ndarray | None = None,
    observer_time_window: tuple[float, float] | None = None,
    quadrature_order: int = 8,
) -> TrajectoryRadiationResult:
    plan = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(directions, reference_axis),
        frequencies,
        coherence=coherence,
        route=route,
        emission=emission,
        form_factor=form_factor,
        bunch_sigma=bunch_sigma,
        observer_time_window=observer_time_window,
        quadrature_order=quadrature_order,
    )
    return plan.prepare().evaluate(trajectory)


def _harmonic_power_density(result: TrajectoryRadiationResult) -> np.ndarray:
    """``dP_m/dΩ = 2π S(m ω₀) / T₀²`` of a one-period periodic lane."""
    return 2.0 * np.pi * np.asarray(result.spectral_energy) / PERIOD**2


def _schott(harmonic: np.ndarray, beta: float, theta: np.ndarray) -> np.ndarray:
    """Schott ``dP_m/dΩ`` for circular motion at polar angle ``theta`` (SI)."""
    m = np.asarray(harmonic, dtype=np.float64)[:, None]
    x = jnp.asarray(m * beta * np.sin(theta)[None, :])
    order = jnp.asarray(np.broadcast_to(m, x.shape))
    bessel = np.asarray(special.jv(order, x))
    derivative = 0.5 * np.asarray(special.jv(order - 1.0, x) - special.jv(order + 1.0, x))
    cotangent = np.cos(theta) / np.sin(theta)
    return (
        Q**2
        * OMEGA0**2
        * m**2
        / (8.0 * np.pi**2 * EPS0 * C)
        * (cotangent[None, :] ** 2 * bessel**2 + beta**2 * derivative**2)
    )


def _lienard_circular_power(beta: float) -> float:
    gamma_sq = 1.0 / (1.0 - beta**2)
    return Q**2 * gamma_sq**2 * (beta * OMEGA0) ** 2 / (6.0 * np.pi * EPS0 * C)


def _status(result: TrajectoryRadiationResult) -> TrajectoryRadiationStatus:
    return TrajectoryRadiationStatus(int(result.evidence.status))


ROUTES: tuple[TrajectoryRadiationRoute, ...] = (
    "segment-exact",
    "segment-hermite",
    "node-gridded",
)


def _window(
    route: TrajectoryRadiationRoute, window: tuple[float, float]
) -> tuple[float, float] | None:
    return window if route == "node-gridded" else None


@pytest.mark.parametrize("route", ROUTES)
def test_uniform_motion_radiates_nothing(route: TrajectoryRadiationRoute) -> None:
    gamma = 3.0
    velocity = np.sqrt(1.0 - 1.0 / gamma**2) * C * np.array([0.6, 0.0, 0.8])
    times = np.linspace(0.0, 1.0e-9, 41)
    zeros = np.zeros((times.size, 3))
    lane = Samples(
        times,
        times[:, None] * velocity[None, :],
        np.broadcast_to(gamma * velocity, zeros.shape).copy(),
        zeros,
    )
    directions = _in_plane_directions(np.array([0.1, 0.9, 2.5]))
    result = _evaluate(
        _trajectory(lane),
        directions,
        np.array([1.0e8, 1.0e9, 1.0e10]),
        route=route,
        observer_time_window=_window(route, (-5.0e-9, 5.0e-9)),
    )
    assert np.max(np.abs(np.asarray(result.field_spectrum))) <= 1.0e-12 * Q / (
        4.0 * np.pi * EPS0 * C
    )
    assert _status(result) == TrajectoryRadiationStatus.SUCCESS
    assert bool(result.evidence.resolved)


def test_nonrelativistic_orbit_radiates_the_lienard_larmor_power() -> None:
    beta = 0.01
    cosines, weights = np.polynomial.legendre.leggauss(16)
    theta = np.arccos(cosines)
    result = _evaluate(
        _trajectory(_periodic_orbit(beta, 512), accelerations=False),
        _in_plane_directions(theta),
        OMEGA0 * np.array([1.0, 2.0, 3.0]),
        emission="truncated",
    )
    harmonic_power = 2.0 * np.pi * _harmonic_power_density(result) @ weights
    # Harmonics m ≥ 4 carry O(β⁶) of the power.
    np.testing.assert_allclose(
        np.sum(harmonic_power), _lienard_circular_power(beta), rtol=1.0e-4
    )


def test_nonrelativistic_spectrum_peaks_at_orbit_frequency_with_beta_squared_harmonic() -> (
    None
):
    beta = 0.01
    frequencies = OMEGA0 * np.linspace(0.5, 2.5, 201)
    periods = 8
    orbit = _periodic_orbit(beta, 128, periods=periods)
    result = _evaluate(
        _trajectory(orbit, accelerations=False),
        _in_plane_directions(np.array([np.pi / 2.0])),
        frequencies,
        emission="truncated",
    )
    energy = np.asarray(result.spectral_energy)[:, 0]
    fundamental = np.argmax(energy)
    assert frequencies[fundamental] == pytest.approx(OMEGA0, rel=1.0e-12)

    cosines, weights = np.polynomial.legendre.leggauss(16)
    theta = np.arccos(cosines)
    ratios = []
    for speed in (beta, 2.0 * beta):
        harmonics = _evaluate(
            _trajectory(_periodic_orbit(speed, 512), accelerations=False),
            _in_plane_directions(theta),
            OMEGA0 * np.array([1.0, 2.0]),
            emission="truncated",
        )
        power = _harmonic_power_density(harmonics) @ weights
        reference = _schott(np.array([1.0, 2.0]), speed, theta) @ weights
        np.testing.assert_allclose(power, reference, rtol=1.0e-4)
        ratios.append(power[1] / power[0])
    # P₂/P₁ = (12/5) β² (1 + O(β²)).
    assert ratios[0] == pytest.approx(12.0 / 5.0 * beta**2, rel=2.0e-3)
    assert ratios[1] / ratios[0] == pytest.approx(4.0, rel=2.0e-3)


@pytest.mark.parametrize("route", ("segment-exact", "node-gridded"))
def test_gamma_ten_harmonics_match_schott(route: TrajectoryRadiationRoute) -> None:
    gamma = 10.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    theta = np.pi / 2.0 - np.array([0.0, 0.05, 0.1, 0.3])
    harmonics = np.array([1.0, 3.0, 10.0, 30.0])
    result = _evaluate(
        _trajectory(_periodic_orbit(beta, 2048), accelerations=False),
        _in_plane_directions(theta),
        OMEGA0 * harmonics,
        route=route,
        emission="truncated",
        observer_time_window=_window(route, (-PERIOD, 2.0 * PERIOD)),
    )
    np.testing.assert_allclose(
        _harmonic_power_density(result),
        _schott(harmonics, beta, theta),
        rtol=1.0e-3,
    )


def _composite_gauss(edges: list[float], order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points, scaled = [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        points.append(0.5 * (upper - lower) * nodes + 0.5 * (upper + lower))
        scaled.append(0.5 * (upper - lower) * weights)
    return np.concatenate(points), np.concatenate(scaled)


def test_gamma_ten_harmonic_sum_matches_lienard_total_power() -> None:
    gamma = 10.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    # Elevation above the orbit plane; both hemispheres by symmetry.
    elevation, elevation_weights = _composite_gauss(
        [0.0, 0.03, 0.08, 0.2, 0.5, np.pi / 2.0], 10
    )
    solid_angle = 4.0 * np.pi * np.cos(elevation) * elevation_weights
    # Harmonics 1..40 individually, 41..12001 by Euler–Maclaurin trapezoid.
    last, stride = 12001, 20
    low = np.arange(1.0, 41.0)
    high = np.arange(41.0, last + 1.0, stride)
    high_weights = np.full(high.size, float(stride))
    high_weights[[0, -1]] = 0.5 * stride + 0.5
    harmonics = np.concatenate((low, high))
    harmonic_weights = np.concatenate((np.ones(low.size), high_weights))
    result = _evaluate(
        _trajectory(_periodic_orbit(beta, 32000), accelerations=False),
        _in_plane_directions(np.pi / 2.0 - elevation),
        OMEGA0 * harmonics,
        route="node-gridded",
        emission="truncated",
        observer_time_window=(-PERIOD, 2.0 * PERIOD),
    )
    summed = harmonic_weights @ _harmonic_power_density(result) @ solid_angle
    # Power above harmonic 12001.5 from the synchrotron spectrum
    # P(x) ∝ F(x), ∫₀^∞ F = 8π/(9√3), x = ω/ω_c, ω_c = (3/2) γ³ ω₀.
    lower = (last + 0.5) / (1.5 * gamma**3)
    x, x_weights = _composite_gauss([lower, 12.0, 20.0, 40.0], 24)
    tail = float(x_weights @ np.asarray(special.synchrotron_f(jnp.asarray(x))))
    tail /= 8.0 * np.pi / (9.0 * np.sqrt(3.0))
    lienard = _lienard_circular_power(beta)
    assert summed / lienard + tail == pytest.approx(1.0, abs=2.0e-4)
    # The receding half of the orbit exceeds one radian per step at the
    # highest harmonic, which the evidence reports; the periodic node sum
    # itself stays spectrally accurate.
    assert TrajectoryRadiationStatus.UNRESOLVED_PHASE in _status(result)


def test_polarization_is_circular_on_axis_and_linear_in_plane() -> None:
    orbit = _periodic_orbit(0.3, 512)
    on_axis = _evaluate(
        _trajectory(orbit, accelerations=False),
        np.array([Z_AXIS]),
        np.array([OMEGA0]),
        reference_axis=np.array([1.0, 0.0, 0.0]),
        emission="truncated",
    )
    stokes = np.asarray(on_axis.stokes)[0, 0]
    # e1 = x, e2 = y: the field rotates from e1 toward e2 (counterclockwise
    # about n), which is V = +I with U = 2 Re(R1 R2*), V = −2 Im(R1 R2*).
    assert stokes[3] == pytest.approx(stokes[0], rel=1.0e-9)
    assert abs(stokes[1]) <= 1.0e-9 * stokes[0]
    assert abs(stokes[2]) <= 1.0e-9 * stokes[0]

    in_plane = _evaluate(
        _trajectory(orbit, accelerations=False),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([OMEGA0]),
        emission="truncated",
    )
    stokes = np.asarray(in_plane.stokes)[0, 0]
    # e1 = z, e2 = x × z = −y; the in-plane field is along y, so Q = −I.
    assert stokes[1] == pytest.approx(-stokes[0], rel=1.0e-9)
    assert abs(stokes[3]) <= 1.0e-9 * stokes[0]
    coherency = np.asarray(in_plane.coherency)[0, 0]
    np.testing.assert_allclose(
        np.asarray(in_plane.spectral_energy)[0, 0],
        EPS0 * C * np.real(np.trace(coherency)) / np.pi,
        rtol=1.0e-14,
    )


def _coherence_case(
    coherence: RadiationCoherence,
    count: float,
    *,
    form_factor: np.ndarray | None = None,
    bunch_sigma: np.ndarray | None = None,
) -> np.ndarray:
    result = _evaluate(
        _trajectory(_bump(0.2, 401), multiplicities=np.array([count])),
        _in_plane_directions(np.array([0.4, 1.3])),
        np.array([0.5, 2.0, 4.0]) / SIGMA,
        route="segment-hermite",
        coherence=coherence,
        form_factor=form_factor,
        bunch_sigma=bunch_sigma,
    )
    return np.asarray(result.spectral_energy)


def test_coherence_models_scale_with_particle_number() -> None:
    count = 1000.0
    single = _coherence_case("coherent", 1.0)
    np.testing.assert_allclose(
        _coherence_case("coherent", count), count**2 * single, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        _coherence_case("incoherent", count), count * single, rtol=1.0e-12
    )
    sigma = np.array([2.0e-2, 1.0e-2, 5.0e-2])
    directions = _in_plane_directions(np.array([0.4, 1.3]))
    frequencies = np.array([0.5, 2.0, 4.0]) / SIGMA
    form_sq = np.exp(
        -(frequencies[:, None] ** 2) * ((directions**2) @ sigma**2)[None, :] / C**2
    )
    np.testing.assert_allclose(
        _coherence_case("gaussian-form-factor", count, bunch_sigma=sigma),
        (count * (1.0 - form_sq) + count**2 * form_sq) * single,
        rtol=1.0e-12,
    )
    table = np.array([1.0, 0.5, 0.0])
    np.testing.assert_allclose(
        _coherence_case("tabulated-form-factor", count, form_factor=table),
        (count * (1.0 - table**2) + count**2 * table**2)[:, None] * single,
        rtol=1.0e-12,
    )


def test_identical_lanes_combine_like_one_lane_with_multiplicity() -> None:
    lane = _bump(0.2, 401)
    directions = _in_plane_directions(np.array([0.4, 1.3]))
    frequencies = np.array([0.5, 2.0]) / SIGMA
    for coherence in ("coherent", "incoherent"):
        split = _evaluate(
            _trajectory(lane, lane, lane),
            directions,
            frequencies,
            coherence=coherence,
        )
        merged = _evaluate(
            _trajectory(lane, multiplicities=np.array([3.0])),
            directions,
            frequencies,
            coherence=coherence,
        )
        np.testing.assert_allclose(
            np.asarray(split.spectral_energy),
            np.asarray(merged.spectral_energy),
            rtol=1.0e-12,
        )


def _mp_amplitude(
    proper: tuple[float, ...], direction: tuple[float, ...]
) -> tuple[mpmath.mpf, ...]:
    """``a = n × (n × β) / (1 − n·β)`` at 50 digits from ``u`` in units of c.

    ``n`` is renormalized at 50 digits: a float direction is unit only to
    ``1e-16``, which ``1 − n·β ≈ 1/(2γ²)`` would amplify by ``2γ²``.
    """
    u = [mpmath.mpf(value) for value in proper]
    raw = [mpmath.mpf(value) for value in direction]
    norm = mpmath.sqrt(sum(value * value for value in raw))
    n = [value / norm for value in raw]
    gamma = mpmath.sqrt(1 + sum(value * value for value in u))
    beta = [value / gamma for value in u]
    parallel = sum(a * b for a, b in zip(n, beta, strict=True))
    kappa = 1 - parallel
    return tuple((parallel * n_i - b_i) / kappa for n_i, b_i in zip(n, beta, strict=True))


def test_retardation_factor_is_stable_at_gamma_ten_thousand() -> None:
    mpmath.mp.dps = 50
    gamma = 1.0e4
    kick = 3.0e-5
    speed = np.sqrt(gamma**2 - 1.0)
    before = speed * Z_AXIS
    after = speed * np.array([np.sin(kick), 0.0, np.cos(kick)])
    step = 1.0e-12
    times = step * np.arange(-2.0, 3.0)
    proper = np.stack([before, before, before, after, after]) * C
    segment = 0.5 * (proper[1:] + proper[:-1])
    velocity = segment / np.sqrt(1.0 + np.sum((segment / C) ** 2, axis=1))[:, None]
    positions = np.concatenate((np.zeros((1, 3)), np.cumsum(velocity * step, axis=0)))
    lane = Samples(times, positions, proper, np.zeros_like(proper))
    # κ is conditioned like 1/κ ≈ 2γ² in the direction, so the reference uses
    # exactly the unit directions the observer plan normalized.
    observers = RadiationObserverPlan(
        np.array(
            [Z_AXIS, [np.sin(1.0 / gamma), 0.0, np.cos(1.0 / gamma)], [0.0, 0.5e-4, 1.0]]
        ),
        np.array([1.0, 1.0, 0.0]),
    )
    directions = np.asarray(observers.directions)
    frequency = 1.0e12
    result = (
        TrajectoryRadiationPlan(
            SCALE,
            observers,
            np.array([frequency]),
            coherence="coherent",
            route="segment-exact",
        )
        .prepare()
        .evaluate(_trajectory(lane, accelerations=False))
    )
    kappa_axis = 1.0 / (gamma**2 * (1.0 + np.sqrt(1.0 - 1.0 / gamma**2)))
    assert float(result.evidence.minimum_retardation_factor) == pytest.approx(
        kappa_axis, rel=1.0e-13
    )
    basis = np.stack(
        (np.asarray(observers.basis_first), np.asarray(observers.basis_second)), axis=1
    )
    prefactor = Q / (4.0 * np.pi * EPS0 * C)
    for index, direction in enumerate(directions):
        amplitudes = [
            _mp_amplitude(tuple(value / C for value in u), tuple(direction))
            for u in segment
        ]
        field = np.zeros(3, dtype=np.complex128)
        for node in (1, 2, 3):
            jump = np.array(
                [float(amplitudes[node][k] - amplitudes[node - 1][k]) for k in range(3)]
            )
            tau = times[node] - positions[node] @ direction / C
            field += jump * np.exp(1j * frequency * tau)
        expected = prefactor * basis[index] @ field
        np.testing.assert_allclose(
            np.asarray(result.field_spectrum)[0, index],
            expected,
            rtol=1.0e-11,
            atol=1.0e-11 * np.max(np.abs(expected)),
        )


def test_evidence_reports_unresolved_phase_and_amplitude() -> None:
    orbit = _periodic_orbit(0.5, 16)
    result = _evaluate(
        _trajectory(orbit, accelerations=False),
        _in_plane_directions(np.array([np.pi / 2.0])),
        OMEGA0 * np.array([1.0, 20.0]),
        emission="truncated",
    )
    status = _status(result)
    assert TrajectoryRadiationStatus.UNRESOLVED_PHASE in status
    assert TrajectoryRadiationStatus.UNRESOLVED_AMPLITUDE in status
    assert float(result.evidence.maximum_phase_increment) > 1.0
    assert float(result.evidence.maximum_relative_amplitude_increment) > 0.5
    assert not bool(result.evidence.resolved)
    valid = np.asarray(result.evidence.derivative_valid)
    assert valid[0].all() and not valid[1].any()


def test_evidence_reports_window_edge_acceleration_against_complete_emission() -> None:
    # Along +y the orbit's acceleration at the window edge is transverse.
    directions = np.array([[0.0, 1.0, 0.0]])
    frequencies = np.array([OMEGA0])
    orbit = _trajectory(_periodic_orbit(0.01, 256), accelerations=False)
    complete = _evaluate(orbit, directions, frequencies, emission="complete")
    truncated = _evaluate(orbit, directions, frequencies, emission="truncated")
    for result in (complete, truncated):
        assert TrajectoryRadiationStatus.WINDOW_EDGE_ACCELERATION in _status(result)
        assert float(result.evidence.window_edge_rate) > 0.5
    assert not bool(complete.evidence.resolved)
    assert bool(truncated.evidence.resolved)

    bump = _evaluate(_trajectory(_bump(0.2, 801)), directions, np.array([1.0 / SIGMA]))
    assert _status(bump) == TrajectoryRadiationStatus.SUCCESS
    assert float(bump.evidence.window_edge_rate) < 1.0e-6


def test_evidence_reports_activity_transition_as_nondifferentiable() -> None:
    lane = _bump(0.2, 401)
    active = np.ones((401, 1), dtype=bool)
    active[:100] = False
    result = _evaluate(
        _trajectory(lane, active=active),
        _in_plane_directions(np.array([1.0])),
        np.array([1.0 / SIGMA]),
    )
    assert TrajectoryRadiationStatus.ACTIVITY_TRANSITION in _status(result)
    assert not np.asarray(result.evidence.derivative_valid).any()
    # Jumps begin one node after activation: no radiation is attributed to it.
    reference = _evaluate(
        _trajectory(lane.window(100, 401)),
        _in_plane_directions(np.array([1.0])),
        np.array([1.0 / SIGMA]),
    )
    np.testing.assert_allclose(
        np.asarray(result.field_spectrum),
        np.asarray(reference.field_spectrum),
        rtol=1.0e-12,
    )
    assert int(result.evidence.segments_used) == 300


def test_evidence_reports_nonmonotone_times() -> None:
    lane = _bump(0.2, 101)
    lane.times[50] = lane.times[49]
    result = _evaluate(
        _trajectory(lane),
        _in_plane_directions(np.array([1.0])),
        np.array([1.0 / SIGMA]),
    )
    assert TrajectoryRadiationStatus.NONMONOTONE_TIME in _status(result)
    assert not bool(result.evidence.resolved)


def test_gridded_route_reports_nodes_outside_the_declared_window() -> None:
    result = _evaluate(
        _trajectory(_bump(0.2, 401)),
        _in_plane_directions(np.array([1.0])),
        np.array([0.5, 1.0, 2.0, 4.0]) / SIGMA * 40.0,
        route="node-gridded",
        observer_time_window=(-2.0 * SIGMA, 2.0 * SIGMA),
    )
    assert TrajectoryRadiationStatus.UNSUPPORTED_NODE in _status(result)
    assert not bool(result.evidence.resolved)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("coherence", ("coherent", "incoherent"))
def test_streaming_accumulation_equals_offline_evaluation(
    route: TrajectoryRadiationRoute, coherence: RadiationCoherence
) -> None:
    lanes = (_bump(0.3, 301), _circular(0.4, 100, periods=3))
    trajectory = _trajectory(*lanes, multiplicities=np.array([2.0, 5.0]))
    plan = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(_in_plane_directions(np.array([0.3, 1.2, 2.9])), Z_AXIS),
        np.array([0.5, 1.0, 3.0]) / SIGMA,
        coherence=coherence,
        route=route,
        emission="truncated",
        resources=TrajectoryRadiationResources(particle_chunk=1, segment_block=7),
        observer_time_window=_window(route, (-10.0 * SIGMA, 10.0 * SIGMA)),
    )
    prepared = plan.prepare()
    offline = prepared.evaluate(trajectory)
    state = prepared.initialize(trajectory)
    for start, stop in ((0, 1), (1, 120), (120, 121), (121, 301)):
        chunk = _trajectory(
            *(lane.window(start, stop) for lane in lanes),
            multiplicities=np.array([2.0, 5.0]),
        )
        state = prepared.accumulate(state, chunk)
    streamed = prepared.finalize(state)
    scale = np.max(np.abs(np.asarray(offline.coherency)))
    np.testing.assert_allclose(
        np.asarray(streamed.coherency),
        np.asarray(offline.coherency),
        rtol=0.0,
        atol=1.0e-12 * scale,
    )
    assert int(streamed.evidence.status) == int(offline.evidence.status)
    assert int(streamed.evidence.segments_used) == int(offline.evidence.segments_used)
    assert float(streamed.evidence.window_edge_rate) == pytest.approx(
        float(offline.evidence.window_edge_rate), rel=1.0e-12
    )
    assert float(streamed.evidence.minimum_retardation_factor) == float(
        offline.evidence.minimum_retardation_factor
    )


def test_streaming_reports_lane_identity_mismatch() -> None:
    lane = _bump(0.2, 101)
    plan = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(_in_plane_directions(np.array([1.0])), Z_AXIS),
        np.array([1.0 / SIGMA]),
        coherence="coherent",
        route="segment-exact",
    )
    prepared = plan.prepare()
    state = prepared.initialize(_trajectory(lane.window(0, 50)))
    state = prepared.accumulate(state, _trajectory(lane.window(0, 50)))
    renamed = _trajectory(
        lane.window(49, 101),
        identities=(np.array([7], dtype=np.uint32), np.array([0], dtype=np.uint32)),
    )
    result = prepared.finalize(prepared.accumulate(state, renamed))
    assert TrajectoryRadiationStatus.LANE_MISMATCH in _status(result)
    assert not bool(result.evidence.resolved)


@pytest.mark.parametrize("route", ("segment-exact", "segment-hermite"))
def test_spectral_energy_derivatives_match_finite_differences(
    route: TrajectoryRadiationRoute,
) -> None:
    lane = _bump(0.3, 201)
    plan = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(_in_plane_directions(np.array([0.5, 1.5])), Z_AXIS),
        np.array([0.5, 2.0]) / SIGMA,
        coherence="coherent",
        route=route,
    )
    prepared = plan.prepare()
    positions = jnp.asarray(lane.positions[:, None])
    proper = jnp.asarray(lane.proper_velocities[:, None])
    accelerations = jnp.asarray(lane.proper_accelerations[:, None])

    @eqx.filter_jit
    def energy(scale: jax.Array) -> jax.Array:
        trajectory = ChargedTrajectory(
            jnp.asarray(lane.times),
            scale * positions,
            scale * proper,
            jnp.asarray([Q]),
            jnp.asarray([1.0]),
            jnp.ones((lane.times.size, 1), dtype=jnp.bool_),
            (jnp.zeros(1, dtype=jnp.uint32), jnp.zeros(1, dtype=jnp.uint32)),
            proper_accelerations=scale * accelerations,
        )
        result = prepared.evaluate(trajectory)
        return result.spectral_energy

    point = jnp.asarray(1.0)
    tangent = jnp.asarray(1.0)
    value, derivative = jax.jvp(energy, (point,), (tangent,))
    step = 1.0e-5
    finite = (energy(point + step) - energy(point - step)) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(derivative), np.asarray(finite), rtol=1.0e-6)
    cotangent = jnp.asarray(
        np.random.default_rng(3).normal(size=value.shape) / np.asarray(value)
    )
    _, pullback = jax.vjp(energy, point)
    (adjoint,) = pullback(cotangent)
    assert float(adjoint * tangent) == pytest.approx(
        float(jnp.sum(cotangent * derivative)), rel=1.0e-10
    )


def test_float32_and_malformed_inputs_are_refused() -> None:
    lane = _bump(0.2, 11)
    with pytest.raises(TypeError, match="float64"):
        ChargedTrajectory(
            lane.times.astype(np.float32),
            lane.positions[:, None],
            lane.proper_velocities[:, None],
            np.array([Q]),
            np.array([1.0]),
            np.ones((11, 1), dtype=bool),
            (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
        )
    with pytest.raises(TypeError, match="uint32"):
        ChargedTrajectory(
            lane.times,
            lane.positions[:, None],
            lane.proper_velocities[:, None],
            np.array([Q]),
            np.array([1.0]),
            np.ones((11, 1), dtype=bool),
            (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.uint32)),
        )
    with pytest.raises(ValueError, match="polarization basis"):
        RadiationObserverPlan(np.array([Z_AXIS]), Z_AXIS)


def test_plan_selectors_and_options_are_refused_when_inconsistent() -> None:
    observers = RadiationObserverPlan(_in_plane_directions(np.array([1.0])), Z_AXIS)
    frequencies = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        TrajectoryRadiationPlan(
            SCALE,
            observers,
            frequencies,
            coherence="partially-coherent",  # ty: ignore[invalid-argument-type]
            route="segment-exact",
        )
    with pytest.raises(ValueError, match="bunch_sigma"):
        TrajectoryRadiationPlan(
            SCALE,
            observers,
            frequencies,
            coherence="gaussian-form-factor",
            route="segment-exact",
        )
    with pytest.raises(ValueError, match="form_factor"):
        TrajectoryRadiationPlan(
            SCALE,
            observers,
            frequencies,
            coherence="coherent",
            route="segment-exact",
            form_factor=np.ones(2),
        )
    with pytest.raises(ValueError, match="observer_time_window"):
        TrajectoryRadiationPlan(
            SCALE, observers, frequencies, coherence="coherent", route="node-gridded"
        )
    with pytest.raises(ValueError, match="increase"):
        TrajectoryRadiationPlan(
            SCALE,
            observers,
            np.array([2.0, 1.0]),
            coherence="coherent",
            route="segment-exact",
        )


def test_execution_refusals() -> None:
    lane = _bump(0.2, 101)
    observers = RadiationObserverPlan(_in_plane_directions(np.array([1.0])), Z_AXIS)
    hermite = TrajectoryRadiationPlan(
        SCALE, observers, np.array([1.0]), coherence="coherent", route="segment-hermite"
    ).prepare()
    with pytest.raises(ValueError, match="proper_accelerations"):
        hermite.evaluate(_trajectory(lane, accelerations=False))
    bounded = TrajectoryRadiationPlan(
        SCALE,
        observers,
        np.array([1.0]),
        coherence="coherent",
        route="segment-exact",
        resources=TrajectoryRadiationResources(maximum_working_bytes=1024),
    ).prepare()
    with pytest.raises(TrajectoryRadiationResourceError, match="working bytes"):
        bounded.evaluate(_trajectory(lane))
    incoherent = TrajectoryRadiationPlan(
        SCALE, observers, np.array([1.0]), coherence="incoherent", route="segment-exact"
    ).prepare()
    result = incoherent.evaluate(_trajectory(lane))
    with pytest.raises(ValueError, match="coherent"):
        incoherent.waveform(result, np.array([0.0]))


def test_per_lane_times_match_shared_times_and_separate_lanes() -> None:
    first = _bump(0.2, 201)
    second = _bump(0.35, 201)
    directions = _in_plane_directions(np.array([0.7, 2.0]))
    frequencies = np.array([0.5, 2.0]) / SIGMA
    shared = _evaluate(
        _trajectory(first, second, shared_times=True), directions, frequencies
    )
    per_lane = _evaluate(_trajectory(first, second), directions, frequencies)
    np.testing.assert_array_equal(
        np.asarray(shared.field_spectrum), np.asarray(per_lane.field_spectrum)
    )
    # A lane on its own, shifted and coarser clock adds its own field.
    shifted = _bump(0.35, 201)
    shifted.times = shifted.times + 3.0 * SIGMA
    shifted.positions = shifted.positions + np.array([0.0, 0.0, 0.1])
    combined = _evaluate(_trajectory(first, shifted), directions, frequencies)
    separate = sum(
        np.asarray(_evaluate(_trajectory(lane), directions, frequencies).field_spectrum)
        for lane in (first, shifted)
    )
    np.testing.assert_allclose(
        np.asarray(combined.field_spectrum), separate, rtol=1.0e-13, atol=0.0
    )


def test_gridded_route_matches_exact_route_within_reported_floor() -> None:
    gamma = 10.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    critical = 1.5 * gamma**3 * OMEGA0
    frequencies = np.linspace(OMEGA0, 4.0 * critical, 300)
    trajectory = _trajectory(_periodic_orbit(beta, 1024), accelerations=False)
    directions = _in_plane_directions(np.pi / 2.0 - np.array([0.0, 0.02, 0.1, 0.5]))
    exact = _evaluate(trajectory, directions, frequencies, emission="truncated")
    gridded = _evaluate(
        trajectory,
        directions,
        frequencies,
        route="node-gridded",
        emission="truncated",
        observer_time_window=(-2.0 * PERIOD, 2.0 * PERIOD),
    )
    floor = gridded.evidence.gridded_error_floor
    assert floor is not None
    difference = np.max(
        np.abs(np.asarray(gridded.field_spectrum) - np.asarray(exact.field_spectrum)),
        axis=(0, 2),
    )
    assert np.all(difference <= np.asarray(floor))
    peak = np.max(np.abs(np.asarray(exact.field_spectrum)))
    assert np.all(np.asarray(floor) <= 1.0e-7 * peak)
    # The comparison spans the synchrotron tail up to 4 ω_c; the floor is
    # absolute, so it bounds the error where the field itself is small.
    assert frequencies[-1] > 3.0 * critical
    assert gridded.evidence.gridded is not None


def test_hermite_route_converges_at_fourth_order() -> None:
    gamma = 3.0
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    theta = np.pi / 2.0 - np.array([0.0, 0.2, 0.6])
    harmonics = np.array([1.0, 2.0, 4.0])
    reference = _schott(harmonics, beta, theta)
    errors = []
    for samples in (24, 48, 96):
        result = _evaluate(
            _trajectory(_circular(beta, samples)),
            _in_plane_directions(theta),
            OMEGA0 * harmonics,
            route="segment-hermite",
            emission="truncated",
            quadrature_order=10,
        )
        errors.append(np.max(np.abs(_harmonic_power_density(result) / reference - 1.0)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders >= 3.8)
    assert errors[-1] < 1.0e-5


def _boost_samples(lane: Samples, beta: np.ndarray) -> Samples:
    """Lorentz-transform events, proper velocities, and four-accelerations."""
    velocity = jnp.asarray(beta)
    events = np.asarray(
        boost_event(
            velocity,
            jnp.asarray(np.concatenate((C * lane.times[:, None], lane.positions), 1)),
        )
    )
    proper = np.asarray(
        boost_proper_velocity(velocity, jnp.asarray(lane.proper_velocities / C))
    )
    gamma = np.sqrt(1.0 + np.sum((lane.proper_velocities / C) ** 2, axis=1))
    # A = dU/dτ = (u·u̇/c, γ u̇); u̇' = A'_spatial / γ'.
    four_acceleration = np.concatenate(
        (
            np.sum(lane.proper_velocities * lane.proper_accelerations, axis=1)[:, None]
            / C,
            gamma[:, None] * lane.proper_accelerations,
        ),
        axis=1,
    )
    boosted = np.asarray(boost_event(velocity, jnp.asarray(four_acceleration)))
    boosted_gamma = np.sqrt(1.0 + np.sum(proper**2, axis=1))
    return Samples(
        events[:, 0] / C,
        events[:, 1:],
        proper * C,
        boosted[:, 1:] / boosted_gamma[:, None],
    )


def test_boost_then_radiate_equals_radiate_then_transform() -> None:
    lane = _bump(0.1, 1601)
    directions = np.array(
        [
            Z_AXIS,
            [np.sin(1.0), 0.0, np.cos(1.0)],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, -np.sqrt(0.75)],
        ]
    )
    frequencies = np.array([0.5, 1.0, 2.0, 4.0]) / SIGMA
    reference_axis = np.array([0.0, 1.0, 1.0])
    source = _evaluate(
        _trajectory(lane),
        directions,
        frequencies,
        route="segment-hermite",
        reference_axis=reference_axis,
    )
    assert _status(source) == TrajectoryRadiationStatus.SUCCESS
    beta = np.array([0.0, 0.0, 0.6])
    transformed = transform_spectral_energy(
        jnp.asarray(beta),
        jnp.asarray(frequencies)[:, None],
        jnp.asarray(directions)[None, :, :],
        source.spectral_energy,
        emission=source.emission,
    )
    boosted = _trajectory(_boost_samples(lane, beta))
    for index in range(directions.shape[0]):
        direct = _evaluate(
            boosted,
            np.asarray(transformed.directions)[0, index][None, :],
            np.asarray(transformed.angular_frequencies)[:, index],
            route="segment-hermite",
            reference_axis=reference_axis,
        )
        np.testing.assert_allclose(
            np.asarray(direct.spectral_energy)[:, 0],
            np.asarray(transformed.spectral_energy)[:, index],
            rtol=1.0e-8,
        )


def test_observer_time_waveform_matches_lienard_wiechert_field() -> None:
    lane = _bump(0.05, 1601)
    directions = np.array([[0.0, 1.0, 0.0], [np.sin(0.7), 0.0, np.cos(0.7)]])
    plan = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(directions, Z_AXIS),
        np.linspace(0.05, 12.0, 240) / SIGMA,
        coherence="coherent",
        route="segment-hermite",
    )
    prepared = plan.prepare()
    result = prepared.evaluate(_trajectory(lane))
    u = lane.proper_velocities
    u_dot = lane.proper_accelerations
    gamma = np.sqrt(1.0 + np.sum((u / C) ** 2, axis=1))[:, None]
    beta = u / (gamma * C)
    beta_dot = (
        u_dot / gamma - u * np.sum(u * u_dot, axis=1)[:, None] / (C**2 * gamma**3)
    ) / C
    prefactor = Q / (4.0 * np.pi * EPS0 * C)
    samples = slice(500, 1101, 50)
    for index, direction in enumerate(directions):
        kappa = 1.0 - beta @ direction
        field = (
            prefactor
            * np.cross(direction, np.cross(direction - beta, beta_dot))
            / kappa[:, None] ** 3
        )
        tau = lane.times - lane.positions @ direction / C
        waveform = np.asarray(prepared.waveform(result, tau[samples]))[:, index]
        expected = np.stack(
            (
                field[samples] @ np.asarray(plan.observers.basis_first)[index],
                field[samples] @ np.asarray(plan.observers.basis_second)[index],
            ),
            axis=-1,
        )
        np.testing.assert_allclose(
            waveform, expected, rtol=0.0, atol=1.0e-3 * np.max(np.abs(expected))
        )
