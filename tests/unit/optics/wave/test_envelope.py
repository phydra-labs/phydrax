#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave._envelope import (
    analytic_field_to_envelope,
    envelope_to_analytic_field,
    GaussianPulseEnvelopePlan,
    GaussianPulseEnvelopeStatus,
    prepare_gaussian_pulse_envelope,
    prepare_pulse_envelope_bridge,
    PulseEnvelopeBridgePlan,
    PulseEnvelopeBridgeStatus,
    PulseEnvelopeField,
    sample_gaussian_pulse_envelope,
)
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._pulse_time import PulseTimeSpace


def _plane(shape=(8, 10), *, periodic=True, half_width=5.0):
    specification = FourierAxisSpec if periodic else UniformAxisSpec
    grid = TensorGridPlan(
        tuple(specification(size) for size in shape), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-half_width, -half_width], [half_width, half_width]]))
    return PlaneFieldSpace(
        grid,
        RigidFrame.identity(3),
        "periodic-cell" if periodic else "finite-window",
    )


def _time(size=64, *, periodic=True, half_width=jnp.pi):
    specification = FourierAxisSpec(size) if periodic else UniformAxisSpec(size)
    grid = TensorGridPlan((specification,), axis_names=("time",)).prepare(
        jnp.asarray([[-half_width], [half_width]])
    )
    return PulseTimeSpace(grid, topology="periodic-cell" if periodic else "finite-window")


def test_pulse_time_topology_is_explicit_and_field_shapes_are_closed():
    finite_grid = TensorGridPlan((UniformAxisSpec(9),), axis_names=("time",)).prepare(
        jnp.asarray([[-1.0], [1.0]])
    )
    periodic_grid = TensorGridPlan((FourierAxisSpec(8),), axis_names=("time",)).prepare(
        jnp.asarray([[-1.0], [1.0]])
    )

    with pytest.raises(ValueError, match="periodic-cell"):
        PulseTimeSpace(finite_grid, topology="periodic-cell")
    with pytest.raises(ValueError, match="finite-window"):
        PulseTimeSpace(periodic_grid, topology="finite-window")

    plane = _plane()
    time = PulseTimeSpace(finite_grid, topology="finite-window")
    with pytest.raises(ValueError, match="shape"):
        PulseEnvelopeField(plane, time, jnp.ones((8, 10, 8)), 4.0, 0.0)


def test_exact_grid_aligned_bridge_roundtrips_scalar_and_tangential_fields():
    plane = _plane()
    time_space = _time()
    time = time_space.coordinates
    envelope = 0.7 * jnp.exp(2j * time) + (0.3 - 0.2j) + 0.1j * jnp.exp(-3j * time)
    envelope = jnp.broadcast_to(envelope, plane.shape + time_space.shape)
    prepared = prepare_pulse_envelope_bridge(
        PulseEnvelopeBridgePlan(
            time_space,
            8.0,
            carrier_grid_tolerance=1.0e-12,
            spectral_support_tolerance=1.0e-12,
        )
    )

    for polarization, values in (
        ("scalar", envelope),
        ("tangential", jnp.stack((envelope, 0.5j * envelope), axis=-1)),
    ):
        field = PulseEnvelopeField(
            plane,
            time_space,
            values,
            8.0,
            -0.4,
            polarization=polarization,
        )
        analytic = envelope_to_analytic_field(prepared, field)
        recovered = analytic_field_to_envelope(prepared, analytic.field)

        expected = values * jnp.exp(-8j * time).reshape(
            (1, 1, time_space.size) + (1,) * (values.ndim - 3)
        )
        assert analytic.successful
        assert analytic.status == int(PulseEnvelopeBridgeStatus.SUCCESS)
        assert jnp.allclose(analytic.field.values, expected, rtol=2e-6, atol=2e-6)
        assert recovered.successful
        assert jnp.allclose(recovered.field.values, values, rtol=2e-6, atol=2e-6)
        assert recovered.field.longitudinal_coordinate == -0.4


def test_bridge_refuses_finite_time_off_grid_carriers_and_forbidden_bands():
    finite = _time(periodic=False)
    with pytest.raises(ValueError, match="periodic-cell"):
        PulseEnvelopeBridgePlan(finite, 8.0)

    periodic = _time()
    with pytest.raises(ValueError, match="not aligned"):
        prepare_pulse_envelope_bridge(
            PulseEnvelopeBridgePlan(
                periodic,
                8.125,
                carrier_grid_tolerance=1.0e-12,
            )
        )
    with pytest.raises(ValueError, match="non-Nyquist"):
        prepare_pulse_envelope_bridge(PulseEnvelopeBridgePlan(periodic, 32.0))

    plane = _plane()
    prepared = prepare_pulse_envelope_bridge(
        PulseEnvelopeBridgePlan(periodic, 8.0, spectral_support_tolerance=1.0e-14)
    )
    time = periodic.coordinates
    for offset in (-8, 24):
        values = jnp.exp(-1j * offset * time)
        field = PulseEnvelopeField(
            plane,
            periodic,
            jnp.broadcast_to(values, plane.shape + periodic.shape),
            8.0,
            0.0,
        )
        result = envelope_to_analytic_field(prepared, field)
        assert not result.successful
        assert result.status & int(PulseEnvelopeBridgeStatus.NONPOSITIVE_OR_NYQUIST_BAND)
        assert result.evidence.rejected_spectral_fraction > 0.99


def test_bridge_admits_smooth_gradients_through_complex_field_values():
    plane = _plane(shape=(4, 4))
    time_space = _time(size=32)
    time = time_space.coordinates
    carrier = 6.0
    prepared = prepare_pulse_envelope_bridge(PulseEnvelopeBridgePlan(time_space, carrier))
    mode = jnp.exp(-1j * time)

    def objective(amplitude):
        values = amplitude * jnp.broadcast_to(mode, plane.shape + time_space.shape)
        field = PulseEnvelopeField(plane, time_space, values, carrier, 0.0)
        result = envelope_to_analytic_field(prepared, field)
        return jnp.real(result.field.values[0, 0, 0])

    derivative = jax.grad(objective)(jnp.asarray(0.7))
    assert jnp.isfinite(derivative)
    assert derivative != 0.0


def test_gaussian_sampling_resolves_physical_rms_widths_phase_and_jones_state():
    plane = _plane(shape=(96, 96), half_width=8.0)
    time_space = _time(size=128, half_width=10.0)
    plan = GaussianPulseEnvelopePlan(
        plane,
        time_space,
        12.0,
        peak_amplitude=3.0,
        transverse_center=jnp.asarray([0.0, 0.0]),
        transverse_rms_width=jnp.asarray([1.2, 0.8]),
        temporal_center=0.0,
        temporal_rms_duration=1.5,
        carrier_phase=0.4,
        polarization="tangential",
        jones_vector=jnp.asarray([1.0, 1.0j]),
        boundary_tolerance=1.0e-6,
        spectral_edge_tolerance=1.0e-8,
    )
    result = sample_gaussian_pulse_envelope(prepare_gaussian_pulse_envelope(plan))
    intensity = jnp.sum(jnp.abs(result.field.values) ** 2, axis=-1)
    weights = plane.area_weights[..., None] * time_space.weights[None, None, :]
    normalization = jnp.sum(weights * intensity)
    coordinates = plane.transverse_coordinates
    time = time_space.coordinates
    rms_u = jnp.sqrt(
        jnp.sum(weights * intensity * coordinates[..., 0, None] ** 2) / normalization
    )
    rms_v = jnp.sqrt(
        jnp.sum(weights * intensity * coordinates[..., 1, None] ** 2) / normalization
    )
    rms_t = jnp.sqrt(
        jnp.sum(weights * intensity * time[None, None, :] ** 2) / normalization
    )
    center = result.field.values[plane.shape[0] // 2, plane.shape[1] // 2]
    temporal_peak = center[jnp.argmax(jnp.abs(center[:, 0]))]

    assert result.successful
    assert result.evidence.status == int(GaussianPulseEnvelopeStatus.SUCCESS)
    assert jnp.allclose(jnp.asarray([rms_u, rms_v]), jnp.asarray([1.2, 0.8]), rtol=2e-3)
    assert jnp.allclose(rms_t, 1.5, rtol=2e-3)
    assert jnp.allclose(jnp.abs(temporal_peak[0]), 3.0 / jnp.sqrt(2.0), rtol=2e-3)
    assert jnp.allclose(temporal_peak[1] / temporal_peak[0], 1.0j, rtol=2e-5)
    assert jnp.allclose(jnp.angle(temporal_peak[0]), 0.4, atol=2e-3)


def test_gaussian_sampling_reports_truncated_support_and_spectral_aliasing():
    plane = _plane(shape=(20, 20), periodic=False, half_width=1.0)
    time_space = _time(size=20, periodic=False, half_width=1.0)
    plan = GaussianPulseEnvelopePlan(
        plane,
        time_space,
        4.0,
        peak_amplitude=1.0,
        transverse_center=jnp.zeros(2),
        transverse_rms_width=jnp.asarray([2.0, 2.0]),
        temporal_center=0.0,
        temporal_rms_duration=0.03,
        boundary_tolerance=1.0e-3,
        spectral_edge_tolerance=1.0e-3,
    )
    result = sample_gaussian_pulse_envelope(prepare_gaussian_pulse_envelope(plan))

    assert not result.successful
    assert result.evidence.status & int(
        GaussianPulseEnvelopeStatus.BOUNDARY_SUPPORT_LIMIT
    )
    assert result.evidence.status & int(GaussianPulseEnvelopeStatus.SPECTRAL_EDGE_LIMIT)
    assert result.evidence.boundary_omitted_fraction > 0.1
    assert result.evidence.spectral_edge_fraction > 1.0e-3
