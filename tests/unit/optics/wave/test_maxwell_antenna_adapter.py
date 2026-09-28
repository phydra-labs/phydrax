#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._external_resource import read_bounded_resource, ResourceLimits
from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.interchange._openpmd_laser import (
    OpenPMDLaserEnvelopeImportPolicy,
    OpenPMDLaserEnvelopeProfile,
    read_openpmd_laser_envelope_hdf5,
    write_openpmd_laser_envelope_hdf5,
)
from phydrax.optics.wave import (
    GaussianPulseEnvelopePlan,
    GaussianPulseEnvelopeStatus,
    openpmd_laser_envelope_antenna,
    prepare_gaussian_pulse_envelope,
    pulse_envelope_antenna,
    PulseEnvelopeField,
    sample_focused_gaussian_pulse_envelope,
    sample_gaussian_pulse_envelope,
)
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._pulse_time import PulseTimeSpace


mx = phx.solver.maxwell


def _gaussian_plan(temporal_center: float) -> Any:
    plane_grid = TensorGridPlan(
        (FourierAxisSpec(128), FourierAxisSpec(128)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-60.0, -60.0], [60.0, 60.0]]))
    time_grid = TensorGridPlan((UniformAxisSpec(33),), axis_names=("time",)).prepare(
        jnp.asarray([[-10.0], [10.0]])
    )
    return GaussianPulseEnvelopePlan(
        PlaneFieldSpace(plane_grid, RigidFrame.identity(3), "periodic-cell"),
        PulseTimeSpace(time_grid, topology="finite-window"),
        4.0,
        peak_amplitude=2.0,
        transverse_center=np.asarray([1.0, -2.0]),
        transverse_rms_width=np.asarray([3.0, 4.0]),
        temporal_center=temporal_center,
        temporal_rms_duration=2.0,
        carrier_phase=0.3,
        polarization="tangential",
        jones_vector=np.asarray([1.0, 0.0]),
    )


def test_focused_gaussian_at_zero_distance_is_the_waist_sample() -> None:
    prepared = prepare_gaussian_pulse_envelope(_gaussian_plan(1.0))
    waist = sample_gaussian_pulse_envelope(prepared)
    focused = sample_focused_gaussian_pulse_envelope(
        prepared, focus_distance=0.0, wave_speed=1.0
    )

    np.testing.assert_allclose(focused.field.values, waist.field.values, atol=1e-12)
    assert int(focused.evidence.status) == int(waist.evidence.status)


def test_focused_gaussian_propagates_to_its_waist_under_exact_diffraction() -> None:
    distance = 40.0
    prepared = prepare_gaussian_pulse_envelope(_gaussian_plan(distance))
    focused = sample_focused_gaussian_pulse_envelope(
        prepared, focus_distance=distance, wave_speed=1.0
    )
    waist = sample_gaussian_pulse_envelope(
        prepare_gaussian_pulse_envelope(_gaussian_plan(0.0))
    )
    peak = 16  # t = 0: the delayed pulse peak at the antenna and at the waist
    grid = focused.field.plane_space.coordinate_axes
    spacing = float(grid[0][1] - grid[0][0])
    frequencies = 2.0 * np.pi * np.fft.fftfreq(128, d=spacing)
    kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
    wavenumber = 4.0
    # Exact (nonparaxial) angular-spectrum propagation over the focus distance.
    transfer = np.exp(1j * np.sqrt(wavenumber**2 - kx**2 - ky**2 + 0j) * distance)
    antenna_slice = np.asarray(focused.field.values[:, :, peak, 0])
    propagated = np.fft.ifft2(np.fft.fft2(antenna_slice) * transfer)
    expected = np.asarray(waist.field.values[:, :, peak, 0])

    assert int(focused.evidence.status) == int(GaussianPulseEnvelopeStatus.SUCCESS)
    assert float(focused.field.longitudinal_coordinate) == -distance
    assert np.max(np.abs(propagated - expected)) < 3e-3 * np.max(np.abs(expected))


def _plane_field(rotation: Any, values: Any, height: float = 5.0) -> Any:
    plane_grid = TensorGridPlan(
        (UniformAxisSpec(7), UniformAxisSpec(9)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-3.0, -4.0], [3.0, 4.0]]))
    time_grid = TensorGridPlan((UniformAxisSpec(5),), axis_names=("time",)).prepare(
        jnp.asarray([[0.0], [4.0]])
    )
    return PulseEnvelopeField(
        PlaneFieldSpace(
            plane_grid, RigidFrame(rotation, [0.0, 0.0, height]), "finite-window"
        ),
        PulseTimeSpace(time_grid, topology="finite-window"),
        values,
        3.0,
        0.0,
        polarization="tangential",
    )


def _bridge() -> Any:
    grid = TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(10) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[-5.0, -5.0, 0.0], [5.0, 5.0, 10.0]]))
    return phx.discretization.StructuredCochainBridge(grid)


def test_pulse_envelope_antenna_depends_only_on_the_physical_field() -> None:
    generator = np.random.default_rng(7)
    values = generator.normal(size=(7, 9, 5, 2)) + 1j * generator.normal(
        size=(7, 9, 5, 2)
    )
    bridge = _bridge()
    aligned = pulse_envelope_antenna(_plane_field(np.eye(3), values), bridge)
    # u → −x, v → −y: the same physical field sampled in a rotated frame.
    rotated = pulse_envelope_antenna(
        _plane_field(np.diag([-1.0, -1.0, 1.0]), -values[::-1, ::-1]), bridge
    )
    # u → y, v → x: the plane normal is −z, so the antenna emits toward −z.
    swapped = pulse_envelope_antenna(
        _plane_field([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], values),
        bridge,
    )
    layout = mx.MaxwellCochainLayout(bridge)
    first = aligned.prepare(bridge, layout)
    second = rotated.prepare(bridge, layout)

    assert aligned.direction == "positive" and aligned.plane_coordinate == 5.0
    np.testing.assert_allclose(rotated.first_coordinates, aligned.first_coordinates)
    np.testing.assert_allclose(rotated.electric, aligned.electric, atol=1e-12)
    np.testing.assert_allclose(
        second.sample(1.5).electric_current, first.sample(1.5).electric_current
    )
    assert swapped.direction == "negative"
    np.testing.assert_allclose(
        swapped.electric[..., 0], np.swapaxes(values, 0, 1)[..., 1]
    )


def test_scalar_envelope_has_no_antenna_polarization() -> None:
    plane_grid = TensorGridPlan(
        (UniformAxisSpec(7), UniformAxisSpec(9)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-3.0, -4.0], [3.0, 4.0]]))
    time_grid = TensorGridPlan((UniformAxisSpec(5),), axis_names=("time",)).prepare(
        jnp.asarray([[0.0], [4.0]])
    )
    field = PulseEnvelopeField(
        PlaneFieldSpace(
            plane_grid, RigidFrame(np.eye(3), [0.0, 0.0, 5.0]), "finite-window"
        ),
        PulseTimeSpace(time_grid, topology="finite-window"),
        np.ones((7, 9, 5)),
        3.0,
        0.0,
    )

    with pytest.raises(ValueError, match="Jones vector"):
        pulse_envelope_antenna(field, _bridge())


def test_openpmd_laser_envelope_antenna_round_trip(tmp_path: Path) -> None:
    generator = np.random.default_rng(11)
    values = generator.normal(size=(7, 9, 5, 2)) + 1j * generator.normal(
        size=(7, 9, 5, 2)
    )
    values = values[..., :1] * np.asarray([1.0, 1.0j]) / np.sqrt(2.0)
    source = _plane_field(np.eye(3), values)
    profile = OpenPMDLaserEnvelopeProfile(polarization=(1.0, 1.0j))
    limits = ResourceLimits(2_000_000, 12, 100_000, 128, 0)
    path = tmp_path / "laser.h5"
    # The draft profile stores plane-local records; the embedding is import metadata.
    write_openpmd_laser_envelope_hdf5(
        path, _plane_field(np.eye(3), values, 0.0), profile, limits=limits
    )
    imported = read_openpmd_laser_envelope_hdf5(
        read_bounded_resource(path.name, trusted_root=path.parent, limits=limits),
        OpenPMDLaserEnvelopeImportPolicy(profile),
        frame=RigidFrame(np.eye(3), [0.0, 0.0, 5.0]),
        longitudinal_coordinate=0.0,
    )
    bridge = _bridge()
    direct = pulse_envelope_antenna(source, bridge)
    restored = openpmd_laser_envelope_antenna(imported, bridge)
    layout = mx.MaxwellCochainLayout(bridge)

    assert restored.provenance_id == imported.report.target_id
    assert restored.source_id != direct.source_id
    np.testing.assert_allclose(restored.electric, direct.electric, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        restored.prepare(bridge, layout).sample(2.0).electric_current,
        direct.prepare(bridge, layout).sample(2.0).electric_current,
        rtol=1e-5,
        atol=1e-8,
    )
