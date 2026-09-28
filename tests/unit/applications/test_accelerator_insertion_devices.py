#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Insertion devices, field maps, and lab-time tracking against independent references.

References: the undulator deflection parameter ``K = e B₀ λ_u / (2π mₑ c)``
with SciPy CODATA 2022 constants, the on-axis resonance
``ω_n = 2nγ² ω_u / (1 + K²/2)`` with only odd harmonics on axis and the
finite-length line width ``Δω/ω_n ≈ 0.886/(nN)`` (FWHM of ``sinc²``) (Clarke,
*The Science and Technology of Undulators and Wigglers*, ch. 3; Kim, AIP Conf.
Proc. 184, 1989); the exact mid-plane magnetostatic invariant
``p_x = −q ∫ B_y dz`` and the path-length delay ``∫ (1/cos θ − 1) dz`` by NumPy
quadrature of the on-axis profile; and the multilinear interpolation bound
``Σ_i h_i² max|∂_i² f| / 8`` of a closed-form vacuum field.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants
from scipy.integrate import cumulative_trapezoid

from phydrax import ElectromagneticScaleContract
from phydrax.applications import accelerator
from phydrax.electromagnetics import RadiationObserverPlan, TrajectoryRadiationPlan


SCALE = ElectromagneticScaleContract.si()
LIGHT = constants.c
CHARGE = constants.e
MASS = constants.m_e
PERIOD = 0.02
GAMMA = 100.0
DEFLECTION = 1.0
MOMENTUM = math.sqrt(GAMMA**2 - 1.0) * MASS * LIGHT
PEAK_FIELD = DEFLECTION * 2.0 * math.pi * MASS * LIGHT / (CHARGE * PERIOD)
WAVENUMBER = 2.0 * math.pi / PERIOD
ORBIT_ANGLE = DEFLECTION / (GAMMA * math.sqrt(1.0 - 1.0 / GAMMA**2))
ORBIT_OFFSET = ORBIT_ANGLE / WAVENUMBER
STEPS_PER_PERIOD = 128
TAIL_RAMPS = 8


def _bunch(
    coordinates: np.ndarray,
    *,
    convention: accelerator.AcceleratorConvention | None = None,
) -> accelerator.AcceleratorBunch:
    count = coordinates.shape[0]
    return accelerator.AcceleratorBunch(
        jnp.asarray(coordinates, dtype=jnp.float64),
        jnp.ones((count,)),
        jnp.arange(count, dtype=jnp.int32),
        reference_rest_energy=MASS * LIGHT**2,
        reference_momentum=MOMENTUM,
        reference_charge=-CHARGE,
        convention=convention,
        bunch_id="electrons",
    )


def _undulator(
    period_count: int, polarization: accelerator.InsertionDevicePolarization, ramp: float
) -> accelerator.InsertionDeviceField:
    return accelerator.InsertionDeviceField(
        PEAK_FIELD,
        PERIOD,
        period_count,
        polarization=polarization,
        center=0.0,
        aperture=(1.0e-3, 1.0e-3),
        ramp_periods=ramp,
    )


def _undulator_plan(
    device: accelerator.InsertionDeviceField,
    *,
    convention: accelerator.AcceleratorConvention | None = None,
) -> accelerator.FieldMapTrackingPlan:
    half = 0.5 * device.period_count * PERIOD + TAIL_RAMPS * device.ramp_width
    time_step = PERIOD / (LIGHT * STEPS_PER_PERIOD)
    return accelerator.FieldMapTrackingPlan(
        accelerator.FieldMapBeamline(SCALE, (device,), element_ids=("undulator",)),
        entrance_plane=-half,
        exit_plane=half,
        time_step=time_step,
        step_count=math.ceil(1.01 * 2.0 * half / (LIGHT * time_step)) + 4,
        convention=convention,
    )


def _track_undulator(
    period_count: int,
    polarization: accelerator.InsertionDevicePolarization = "planar",
    ramp: float = 0.25,
) -> accelerator.FieldMapTrackingResult:
    device = _undulator(period_count, polarization, ramp)
    return accelerator.track_field_map(_undulator_plan(device), _bunch(np.zeros((1, 6))))


@pytest.fixture(scope="module")
def ten_periods() -> accelerator.FieldMapTrackingResult:
    return _track_undulator(10)


@pytest.fixture(scope="module")
def twenty_periods() -> accelerator.FieldMapTrackingResult:
    return _track_undulator(20)


def _fundamental() -> float:
    return 2.0 * GAMMA**2 * (2.0 * math.pi * LIGHT / PERIOD) / (1.0 + DEFLECTION**2 / 2.0)


def _on_axis_spectrum(
    result: accelerator.FieldMapTrackingResult, frequencies: np.ndarray
) -> np.ndarray:
    observers = RadiationObserverPlan(
        np.array([[0.0, 0.0, 1.0]]), np.array([1.0, 0.0, 0.0])
    )
    plan = TrajectoryRadiationPlan(
        SCALE, observers, frequencies, coherence="coherent", route="segment-exact"
    )
    spectrum = plan.prepare().evaluate(result.trajectory)
    assert bool(spectrum.evidence.resolved)
    return np.asarray(spectrum.spectral_energy[:, 0])


def _line(
    result: accelerator.FieldMapTrackingResult, harmonic: int, period_count: int
) -> tuple[float, float, float]:
    """Return ``(peak frequency / nω₁, peak energy, FWHM / nω₁)`` of one line."""
    center = harmonic * _fundamental()
    half_span = 2.0 / (harmonic * period_count)
    frequencies = center * np.linspace(1.0 - half_span, 1.0 + half_span, 801)
    energy = _on_axis_spectrum(result, frequencies)
    peak = int(np.argmax(energy))
    above = np.flatnonzero(energy >= 0.5 * energy[peak])
    assert above[0] > 0 and above[-1] < energy.shape[0] - 1
    width = (frequencies[above[-1]] - frequencies[above[0]]) / center
    return float(frequencies[peak] / center), float(energy[peak]), float(width)


def test_deflection_parameter_matches_definition_and_tracked_orbit_angle(
    ten_periods: accelerator.FieldMapTrackingResult,
) -> None:
    device = _undulator(10, "planar", 0.25)
    resonance = device.resonance(SCALE, GAMMA)
    reference = CHARGE * PEAK_FIELD * PERIOD / (2.0 * math.pi * MASS * LIGHT)
    assert resonance.deflection_parameter == pytest.approx(reference, rel=1.0e-12)
    assert resonance.fundamental_angular_frequency == pytest.approx(
        _fundamental(), rel=1.0e-12
    )
    proper = np.asarray(ten_periods.trajectory.proper_velocities[:, 0])
    angle = np.abs(proper[:, 0]) / np.linalg.norm(proper, axis=-1)
    assert np.max(angle) == pytest.approx(ORBIT_ANGLE, rel=1.0e-3)


def test_on_axis_spectrum_has_resonant_odd_harmonics(
    ten_periods: accelerator.FieldMapTrackingResult,
) -> None:
    first_position, _, _ = _line(ten_periods, 1, 10)
    third_position, third_energy, _ = _line(ten_periods, 3, 10)
    assert first_position == pytest.approx(1.0, abs=3.0e-3)
    assert third_position == pytest.approx(1.0, abs=3.0e-3)
    even = _on_axis_spectrum(ten_periods, np.array([2.0 * _fundamental()]))
    assert even[0] < 1.0e-3 * third_energy


@pytest.mark.parametrize(
    ("harmonic", "period_count"),
    [(1, 10), (3, 10), (1, 20)],
    ids=["fundamental-10", "third-10", "fundamental-20"],
)
def test_line_width_scales_inversely_with_harmonic_and_period_count(
    harmonic: int,
    period_count: int,
    ten_periods: accelerator.FieldMapTrackingResult,
    twenty_periods: accelerator.FieldMapTrackingResult,
) -> None:
    result = ten_periods if period_count == 10 else twenty_periods
    _, _, width = _line(result, harmonic, period_count)
    assert harmonic * period_count * width == pytest.approx(0.886, rel=0.06)


@pytest.mark.parametrize(
    ("polarization", "ramp", "offset_tolerance", "angle_tolerance"),
    [("planar", 0.25, 1.0e-3, 1.0e-5), ("helical", 1.0, 5.0e-3, 1.0e-4)],
    ids=["planar", "helical"],
)
def test_matched_termination_exits_on_axis(
    polarization: accelerator.InsertionDevicePolarization,
    ramp: float,
    offset_tolerance: float,
    angle_tolerance: float,
) -> None:
    result = _track_undulator(10, polarization, ramp)
    assert bool(result.evidence.accepted)
    x, px, y, py = np.asarray(result.bunch.coordinates[0, :4])
    assert abs(x) < offset_tolerance * ORBIT_OFFSET
    assert abs(y) < offset_tolerance * ORBIT_OFFSET
    assert abs(px) < angle_tolerance * ORBIT_ANGLE
    assert abs(py) < angle_tolerance * ORBIT_ANGLE


def test_exit_delay_is_path_length_excess_in_convention_sign() -> None:
    device = _undulator(10, "planar", 0.25)
    late = accelerator.track_field_map(_undulator_plan(device), _bunch(np.zeros((1, 6))))
    early_convention = accelerator.AcceleratorConvention(
        longitudinal_sign="positive-early"
    )
    early = accelerator.track_field_map(
        _undulator_plan(device, convention=early_convention),
        _bunch(np.zeros((1, 6)), convention=early_convention),
    )
    plan = _undulator_plan(device)
    z = np.linspace(plan.entrance_plane, plan.exit_plane, 400001)
    flat, width = 5.0 * PERIOD, device.ramp_width
    window = 0.5 * (np.tanh((z + flat) / width) - np.tanh((z - flat) / width))
    field = PEAK_FIELD * window * np.cos(WAVENUMBER * z)
    angle = CHARGE * cumulative_trapezoid(field, z, initial=0) / MOMENTUM
    delay = np.trapezoid(1.0 / np.sqrt(1.0 - angle**2) - 1.0, z)
    zeta = float(late.bunch.coordinates[0, 4])
    assert zeta == pytest.approx(delay, rel=1.0e-3)
    assert float(early.bunch.coordinates[0, 4]) == pytest.approx(-zeta, rel=1.0e-12)
    assert float(late.bunch.coordinates[0, 5]) == pytest.approx(0.0, abs=1.0e-12)


def test_bend_exit_angle_equals_field_integral() -> None:
    field, length, fringe = 0.02, 0.5, 0.02
    bend = accelerator.DipoleBendField(
        field, length, center=0.0, fringe_width=fringe, aperture=(0.2, 0.01)
    )
    half = 0.5 * length + 20.0 * fringe
    time_step = 2.0e-3 / LIGHT
    plan = accelerator.FieldMapTrackingPlan(
        accelerator.FieldMapBeamline(SCALE, (bend,), element_ids=("bend",)),
        entrance_plane=-half,
        exit_plane=half,
        time_step=time_step,
        step_count=math.ceil(1.05 * 2.0 * half / (LIGHT * time_step)),
    )
    result = accelerator.track_field_map(plan, _bunch(np.zeros((1, 6))))
    assert bool(result.evidence.accepted)
    sine = float(result.bunch.coordinates[0, 1])
    assert sine == pytest.approx(CHARGE * field * length / MOMENTUM, rel=1.0e-7)


def _vacuum_field(y: np.ndarray, z: np.ndarray, /) -> np.ndarray:
    return np.stack(
        (
            np.zeros_like(y),
            PEAK_FIELD * np.cosh(WAVENUMBER * y) * np.cos(WAVENUMBER * z),
            -PEAK_FIELD * np.sinh(WAVENUMBER * y) * np.sin(WAVENUMBER * z),
        ),
        axis=-1,
    )


def test_tabulated_map_matches_analytic_field_within_interpolation_bound() -> None:
    x = np.linspace(-2.0e-3, 2.0e-3, 5)
    y = np.linspace(-2.0e-3, 2.0e-3, 9)
    z = np.linspace(-0.04, 0.04, 161)
    _, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
    table = accelerator.TabulatedFieldMap(x, y, z, _vacuum_field(grid_y, grid_z))
    rng = np.random.default_rng(7)
    queries = np.stack(
        (
            rng.uniform(-2.0e-3, 2.0e-3, 4000),
            rng.uniform(-2.0e-3, 2.0e-3, 4000),
            rng.uniform(-0.04, 0.04, 4000),
        ),
        axis=-1,
    )
    sample = table.external_fields(jnp.asarray(queries), jnp.zeros((4000,)))
    error = np.max(
        np.abs(np.asarray(sample.magnetic) - _vacuum_field(queries[:, 1], queries[:, 2])),
        axis=0,
    )
    spacing_sq = (y[1] - y[0]) ** 2 + (z[1] - z[0]) ** 2
    bound = (
        spacing_sq
        * WAVENUMBER**2
        * PEAK_FIELD
        / 8.0
        * np.array([0.0, np.cosh(WAVENUMBER * y[-1]), np.sinh(WAVENUMBER * y[-1])])
    )
    assert bool(np.all(sample.support))
    assert np.all(error <= bound)
    np.testing.assert_allclose(
        table.magnetic_interpolation_estimate, bound, rtol=0.15, atol=1.0e-15
    )


def test_tabulated_map_support_distinguishes_beyond_from_outside() -> None:
    nodes = np.linspace(-1.0e-3, 1.0e-3, 3)
    z = np.linspace(-0.1, 0.1, 5)
    table = accelerator.TabulatedFieldMap(nodes, nodes, z, np.full((3, 3, 5, 3), 0.25))
    sample = table.external_fields(
        jnp.array([[0.0, 0.0, 0.2], [2.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        jnp.zeros((3,)),
    )
    np.testing.assert_array_equal(np.asarray(sample.support), [True, False, True])
    np.testing.assert_array_equal(np.asarray(sample.magnetic[:2]), 0.0)
    np.testing.assert_allclose(np.asarray(sample.magnetic[2]), 0.25)


def test_lanes_leaving_field_support_are_refused_not_tracked_through_zero_field() -> None:
    nodes = np.linspace(-1.0e-3, 1.0e-3, 3)
    z = np.linspace(-0.1, 0.1, 5)
    table = accelerator.TabulatedFieldMap(nodes, nodes, z, np.zeros((3, 3, 5, 3)))
    time_step = 1.0e-3 / LIGHT
    plan = accelerator.FieldMapTrackingPlan(
        accelerator.FieldMapBeamline(SCALE, (table,), element_ids=("map",)),
        entrance_plane=-0.12,
        exit_plane=0.15,
        time_step=time_step,
        step_count=400,
    )
    coordinates = np.zeros((2, 6))
    coordinates[1, 1] = 0.02
    result = accelerator.track_field_map(plan, _bunch(coordinates))
    evidence = result.evidence
    assert not bool(evidence.accepted)
    assert int(evidence.status) == int(accelerator.FieldMapTrackingStatus.UNSUPPORTED)
    np.testing.assert_array_equal(np.asarray(evidence.supported), [True, False])
    np.testing.assert_array_equal(np.asarray(evidence.exited), [True, False])
    np.testing.assert_array_equal(np.asarray(result.bunch.active), [True, False])
    active = np.asarray(result.trajectory.active)
    assert np.all(active[:, 0])
    assert active[0, 1] and not active[-1, 1]
    last = np.flatnonzero(active[:, 1])[-1]
    positions = np.asarray(result.trajectory.positions[: last + 1, 1])
    assert np.all(np.abs(positions[:, 0]) <= 1.0e-3)


def test_aperture_beyond_analytic_support_is_refused() -> None:
    with pytest.raises(ValueError, match="analytic support"):
        accelerator.InsertionDeviceField(
            PEAK_FIELD,
            PERIOD,
            10,
            polarization="planar",
            center=0.0,
            aperture=(1.0e-2, PERIOD),
            ramp_periods=1.0,
        )
    with pytest.raises(ValueError, match="analytic support"):
        accelerator.InsertionDeviceField(
            PEAK_FIELD,
            PERIOD,
            10,
            polarization="helical",
            center=0.0,
            aperture=(PERIOD, 1.0e-3),
            ramp_periods=1.0,
        )


def test_trajectory_memory_above_budget_is_refused() -> None:
    device = _undulator(10, "planar", 0.25)
    plan = accelerator.FieldMapTrackingPlan(
        accelerator.FieldMapBeamline(SCALE, (device,), element_ids=("undulator",)),
        entrance_plane=-0.2,
        exit_plane=0.2,
        time_step=1.0e-12,
        step_count=2000,
        maximum_trajectory_bytes=1024,
    )
    with pytest.raises(accelerator.FieldMapTrackingResourceError):
        accelerator.track_field_map(plan, _bunch(np.zeros((1, 6))))
