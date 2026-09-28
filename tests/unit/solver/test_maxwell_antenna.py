#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-way sampled plane antennas on the compatible Maxwell cochain solver.

References are independent of the antenna: plane-wave energy ``ε c A ∫ E(t)² dt``,
the paraxial Gaussian beam propagated with the Yee lattice's on-axis dispersion
``sin(k̃h/2)/h = sin(ωΔt/2)/(cΔt)`` and diffraction wavenumber ``sin(k̃h)/h``,
and the relativistic Doppler map ``F_lab(ω) = F'(ω/D)``, ``D = γ(1 + β)``, of a
plane wave emitted by a sheet moving along its normal.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    GaussianPulseEnvelopePlan,
    prepare_gaussian_pulse_envelope,
    pulse_envelope_antenna,
    sample_focused_gaussian_pulse_envelope,
)
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._pulse_time import PulseTimeSpace


mx = phx.solver.maxwell


def _column_bridge(cells: int, spacing: float) -> Any:
    """Transversely periodic 2×2 column: a plane wave along ``z``."""
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(cells),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [2 * spacing, 2 * spacing, cells * spacing]]))
    return phx.discretization.StructuredCochainBridge(grid)


def test_three_dimensional_beam_runs_on_declared_magnetic_charge() -> None:
    # A focused 3-D Gaussian has a normal B, so its magnetic sheet has surface
    # divergence; the runtime must track it as charge, not project it.
    grid = TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(size) for size in (48, 48, 64)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [48.0, 48.0, 64.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    plane = PlaneFieldSpace(
        TensorGridPlan(
            (UniformAxisSpec(97), UniformAxisSpec(97)), axis_names=("u", "v")
        ).prepare(jnp.asarray([[-24.0, -24.0], [24.0, 24.0]])),
        RigidFrame(np.eye(3), np.asarray([24.0, 24.0, 8.0])),
        "finite-window",
    )
    time = PulseTimeSpace(
        TensorGridPlan((UniformAxisSpec(289),), axis_names=("time",)).prepare(
            jnp.asarray([[0.0], [72.0]])
        ),
        topology="finite-window",
    )
    beam = GaussianPulseEnvelopePlan(
        plane,
        time,
        2.0 * np.pi / 8.0,
        peak_amplitude=1.0,
        transverse_center=np.zeros(2),
        transverse_rms_width=np.full(2, 4.0),
        temporal_center=36.0 + 12.5,
        temporal_rms_duration=6.0,
        polarization="tangential",
        jones_vector=np.asarray([1.0, 0.0]),
    )
    sampled = sample_focused_gaussian_pulse_envelope(
        prepare_gaussian_pulse_envelope(beam), focus_distance=12.5, wave_speed=1.0
    )
    antenna = pulse_envelope_antenna(sampled.field, bridge)
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(antenna,),
        observers=(mx.MaxwellAntennaWorkObserverPlan(antenna),),
    ).prepare()
    step = 0.95 * float(runtime.stable_dt)
    # Stop at the pulse peak, where the declared surface charge is largest.
    result = mx.solve_compatible_maxwell(
        runtime, runtime.initialize(), 0.0, step, int(36.0 / step)
    )
    source, ledger = runtime.sources[0], runtime.observers[0]
    assert isinstance(source, mx.PreparedSampledPlaneCurrentAntenna)
    assert isinstance(ledger, mx.PreparedMaxwellAntennaWorkObserver)
    declared = np.asarray(result.final_state.auxiliary.magnetic_charge)
    defect = np.asarray(runtime.magnetic_constraint(result.final_state))
    energy = float(runtime.energy(result.final_state))

    assert float(source.evidence.magnetic_closure_defect) > 1e-3
    assert runtime.magnetic_projection_elided
    assert np.max(np.abs(declared)) > 0.0
    assert np.max(np.abs(defect)) < 1e-12 * np.max(np.abs(declared))
    # The trapezoidal ledger of step-end fields agrees with the leapfrog energy to
    # its O((ωΔt)²) consistency error (measured coefficient 1/36 on the plane-wave
    # test; this broadband 3-D beam is bounded by (ωΔt)²/12).
    carrier = 2.0 * np.pi / 8.0
    assert float(ledger.evidence(result.final_state.observations[0]).total_work) == (
        pytest.approx(energy, rel=(carrier * step) ** 2 / 12.0)
    )


def _x_edges(bridge: Any, planes: Any, columns: Any = (0,)) -> np.ndarray:
    """Packed ``E_x`` edge indices at x-cells ``columns``, y-node 0, z-nodes ``planes``."""
    shape = bridge.orientation_shapes[1][0]
    cells, nodes = np.meshgrid(np.asarray(columns), np.asarray(planes), indexing="ij")
    return bridge.orientation_offsets[1][0] + np.ravel_multi_index(
        (cells.T.reshape(-1), np.zeros(cells.size, dtype=np.int64), nodes.T.reshape(-1)),
        shape,
    )


def _uniform_x_polarized(envelope: np.ndarray) -> np.ndarray:
    electric = np.zeros((2, 2, envelope.size, 2), dtype=np.complex128)
    electric[..., 0] = envelope
    return electric


_H = 0.05
_CELLS = 240
_PLANE = 100 * _H
_OMEGA = 2.0 * np.pi  # 20 cells per vacuum wavelength
_TIMES = np.linspace(0.0, 6.0, 601)
_ENVELOPE = np.exp(-(((_TIMES - 2.0) / 0.5) ** 2))


def _plane_wave_run(direction: mx.AntennaEmissionDirection) -> tuple[Any, Any, Any]:
    bridge = _column_bridge(_CELLS, _H)
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        _PLANE,
        [-1.0, 1.0],
        [-1.0, 1.0],
        _TIMES,
        _uniform_x_polarized(_ENVELOPE),
        carrier_angular_frequency=_OMEGA,
        direction=direction,
    )
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        sources=(antenna,),
        observers=(mx.MaxwellAntennaWorkObserverPlan(antenna),),
    ).prepare()
    step = 0.9 * float(runtime.stable_dt)
    result = mx.solve_compatible_maxwell(
        runtime, runtime.initialize(), 0.0, step, int(4.6 / step)
    )
    return bridge, runtime, result


@pytest.fixture(scope="module", params=["positive", "negative"])
def plane_wave_run(request: Any) -> tuple[str, Any, Any, Any]:
    return (request.param, *_plane_wave_run(request.param))


def test_plane_antenna_radiates_only_along_its_emission_direction(
    plane_wave_run: Any,
) -> None:
    direction, bridge, runtime, result = plane_wave_run
    electric = np.asarray(runtime.electric_field(result.final_state))
    nodes = np.arange(_CELLS + 1) * _H
    field = electric[_x_edges(bridge, np.arange(_CELLS + 1))] / _H
    ahead = nodes > _PLANE if direction == "positive" else nodes < _PLANE
    behind = nodes < _PLANE if direction == "positive" else nodes > _PLANE
    forward = np.sum(field[ahead] ** 2)
    backward = np.sum(field[behind] ** 2)
    evidence = runtime.sources[0].evidence

    assert backward / forward < 1e-5
    assert np.max(np.abs(field[ahead])) == pytest.approx(1.0, abs=0.03)
    # A transversely uniform wave has a divergence-free magnetic sheet, so the
    # runtime may elide the magnetic projection.
    assert float(evidence.magnetic_closure_defect) < 1e-14
    assert runtime.magnetic_projection_elided
    assert float(evidence.cells_per_wavelength) == pytest.approx(20.0)


def test_antenna_work_equals_injected_field_energy(plane_wave_run: Any) -> None:
    _, _, runtime, result = plane_wave_run
    ledger = runtime.observers[0].evidence(result.final_state.observations[0])
    energy = float(runtime.energy(result.final_state))
    carrier = np.real(_ENVELOPE * np.exp(-1j * _OMEGA * _TIMES))
    # ε = μ = c = 1: a launched plane pulse carries A ∫ E(t)² dt.
    analytic = (2 * _H) ** 2 * np.trapezoid(carrier**2, _TIMES)
    # The trapezoidal ledger of step-end fields and the leapfrog energy differ by
    # the scheme's O((ωΔt)²) consistency error: halving Δt must quarter it.
    step = 0.9 * float(runtime.stable_dt)
    half = mx.solve_compatible_maxwell(
        runtime, runtime.initialize(), 0.0, 0.5 * step, 2 * int(4.6 / step)
    )
    half_energy = float(runtime.energy(half.final_state))
    half_work = float(runtime.observers[0].value(half.final_state.observations[0]))
    mismatch = (
        abs(float(ledger.total_work) - energy) / energy,
        abs(half_work - half_energy) / half_energy,
    )

    assert mismatch[0] < (_OMEGA * step) ** 2 / 24.0
    assert np.log2(mismatch[0] / mismatch[1]) == pytest.approx(2.0, abs=0.1)
    assert abs(float(ledger.first_power)) < 1e-12 * energy
    assert energy == pytest.approx(analytic, rel=0.02)


def test_paraxial_gaussian_beam_reaches_its_waist_with_gouy_phase() -> None:
    columns, cells, periodic_width = 140, 170, 4.0
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(columns),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(cells),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [columns, 2 * periodic_width, cells]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    omega = 2.0 * np.pi / 12.0
    step = 0.95 * float(phx.solver.CompatibleMaxwellPlan(bridge).prepare().stable_dt)
    # Yee on-axis wavenumber and paraxial diffraction wavenumber on the lattice.
    axial = 2.0 * np.arcsin(np.sin(0.5 * omega * step) / step)
    diffraction = np.sin(axial)
    waist = 18.0
    rayleigh = 0.5 * diffraction * waist**2
    antenna_plane = 10.0
    focus = antenna_plane + 0.5 * rayleigh
    first = np.linspace(0.0, columns, 281)
    offset = first - 0.5 * columns
    distance = focus - antenna_plane
    width = waist * np.sqrt(1.0 + (distance / rayleigh) ** 2)
    profile = np.sqrt(waist / width) * np.exp(
        -((offset / width) ** 2)
        - 0.5j * diffraction * distance * offset**2 / (distance**2 + rayleigh**2)
        + 0.5j * np.arctan(distance / rayleigh)
    )
    times = np.linspace(0.0, 80.0, 801)
    electric = np.zeros((first.size, 2, times.size, 2), dtype=np.complex128)
    electric[..., 0] = profile[:, None, None] * np.exp(-(((times - 36.0) / 12.0) ** 2))
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        antenna_plane,
        first,
        [-10.0, 20.0],
        times,
        electric,
        carrier_angular_frequency=omega,
    )
    planes = np.asarray([16, 34, 52, 70, 88, 106, 124])
    probe = mx.FieldProbePlan("electric", _x_edges(bridge, planes, np.arange(columns)))
    acquisition = mx.MaxwellSpectralAcquisition(
        np.asarray([omega]), sign="positive", measure="time-integral"
    )
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(antenna,), observers=(mx.DFTObserverPlan(probe, acquisition),)
    ).prepare()
    # The run stops before the z = 170 boundary reflection reaches any plane.
    result = mx.solve_compatible_maxwell(
        runtime, runtime.initialize(), 0.0, step, int(190.0 / step)
    )
    phasors = np.asarray(result.observations[0])[0].reshape(planes.size, columns)
    centers = np.arange(columns) + 0.5 - 0.5 * columns
    intensity = np.abs(phasors) ** 2
    widths = 2.0 * np.sqrt(np.sum(centers**2 * intensity, 1) / np.sum(intensity, 1))
    axis = 0.5 * (phasors[:, columns // 2 - 1] + phasors[:, columns // 2])
    gouy = np.angle(axis * np.exp(-1j * axial * (planes - antenna_plane)))
    expected_widths = waist * np.sqrt(1.0 + ((planes - focus) / rayleigh) ** 2)
    expected_gouy = -0.5 * np.arctan((planes - focus) / rayleigh)

    assert runtime.magnetic_projection_elided
    assert planes[np.argmin(widths)] == planes[np.argmin(np.abs(planes - focus))]
    np.testing.assert_allclose(widths, expected_widths, rtol=0.01)
    np.testing.assert_allclose(gouy, expected_gouy, atol=6e-3)


_MOVING_BETA = 0.2
_WAVELENGTH = 2.4e-6


def _moving_run(cells_per_wavelength: int) -> tuple[Any, np.ndarray, np.ndarray, Any]:
    """β = 0.2 plane-wave antenna; forward/backward probe spectra and rest spectrum."""
    scale = phx.ElectromagneticScaleContract.si()
    light = float(scale.speed_of_light)
    spacing = _WAVELENGTH / cells_per_wavelength
    ratio = cells_per_wavelength // 24
    omega = 2.0 * np.pi * light / _WAVELENGTH
    period = 2.0 * np.pi / omega
    doppler = np.sqrt((1.0 + _MOVING_BETA) / (1.0 - _MOVING_BETA))
    bridge = _column_bridge(400 * ratio, spacing)
    times = np.linspace(0.0, 8.0 * period, 801)
    envelope = np.exp(-(((times - 4.0 * period) / (1.2 * period)) ** 2))
    antenna = mx.SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        60 * ratio * spacing,
        np.asarray([-1.0, 1.0]),
        np.asarray([-1.0, 1.0]),
        times,
        _uniform_x_polarized(envelope),
        carrier_angular_frequency=omega,
        beta=_MOVING_BETA,
        scale=scale,
    )
    frequencies = np.linspace(0.3, 2.5, 221) * doppler * omega
    stop = 430 * ratio * spacing / light  # before the far-boundary reflection returns
    acquisition = mx.MaxwellSpectralAcquisition(
        frequencies, sign="positive", measure="time-integral", stop_time=stop
    )
    probe = mx.FieldProbePlan("electric", _x_edges(bridge, [300 * ratio, 30 * ratio]))
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        constitutive=mx.DiagonalMaxwellConstitutivePlan(
            permittivity=float(scale.vacuum_permittivity),
            permeability=float(scale.vacuum_permeability),
        ),
        sources=(antenna,),
        observers=(mx.DFTObserverPlan(probe, acquisition),),
    ).prepare()
    step = 0.95 * float(runtime.stable_dt)
    result = mx.solve_compatible_maxwell(
        runtime, runtime.initialize(), 0.0, step, int(stop / step) + 2
    )
    spectra = np.asarray(result.observations[0]) / spacing
    fine = np.linspace(times[0], times[-1], 20001)
    rest_signal = np.real(np.interp(fine, times, envelope) * np.exp(-1j * omega * fine))
    rest = np.asarray(
        [
            np.trapezoid(rest_signal * np.exp(1j * w * fine), fine)
            for w in frequencies / doppler
        ]
    )
    return runtime, frequencies, spectra, rest


@pytest.fixture(scope="module")
def moving_run() -> tuple[Any, np.ndarray, np.ndarray, Any]:
    return _moving_run(24)


def test_moving_antenna_emits_the_doppler_shifted_wave(moving_run: Any) -> None:
    runtime, frequencies, spectra, rest = moving_run
    forward = spectra[:, 0]
    doppler = np.sqrt((1.0 + _MOVING_BETA) / (1.0 - _MOVING_BETA))
    omega = 2.0 * np.pi * float(phx.ElectromagneticScaleContract.si().speed_of_light)
    omega /= _WAVELENGTH
    rest_centroid = np.sum(frequencies / doppler * np.abs(rest) ** 2) / np.sum(
        np.abs(rest) ** 2
    )
    centroid = np.sum(frequencies * np.abs(forward) ** 2) / np.sum(np.abs(forward) ** 2)
    source = runtime.sources[0]
    assert isinstance(source, mx.PreparedSampledPlaneCurrentAntenna)

    assert float(source.evidence.emitted_carrier_angular_frequency) == pytest.approx(
        doppler * omega
    )
    assert centroid == pytest.approx(doppler * rest_centroid, rel=2e-3)
    # F_lab(ω) = F'(ω/D): the field gains D while the pulse shortens by D.
    assert np.max(np.abs(forward)) == pytest.approx(np.max(np.abs(rest)), rel=0.01)


def test_moving_antenna_backward_leakage_converges_at_second_order(
    moving_run: Any,
) -> None:
    _, _, coarse, _ = moving_run
    _, _, fine, _ = _moving_run(48)
    leakage = [
        np.max(np.abs(spectra[:, 1])) / np.max(np.abs(spectra[:, 0]))
        for spectra in (coarse, fine)
    ]

    assert leakage[0] < 2e-3
    assert np.log2(leakage[0] / leakage[1]) >= 1.8


def test_antenna_refuses_inadmissible_configurations() -> None:
    bridge = _column_bridge(40, 1.0)
    times = np.linspace(0.0, 4.0, 5)
    electric = _uniform_x_polarized(np.ones(times.size))

    def plan(**options: Any) -> Any:
        coordinate = options.pop("plane_coordinate", 20.0)
        return mx.SampledPlaneCurrentAntennaPlan(
            bridge, 2, coordinate, [-1.0, 3.0], [-1.0, 3.0], times, electric, **options
        )

    with pytest.raises(ValueError, match="ElectromagneticScaleContract"):
        plan(beta=0.3)
    with pytest.raises(ValueError, match="vacuum"):
        plan(
            scale=phx.ElectromagneticScaleContract.si(),
            medium=mx.HomogeneousMaxwellExterior(permittivity=2.0),
        )
    with pytest.raises(ValueError, match="nonperiodic"):
        phx.solver.CompatibleMaxwellPlan(
            bridge,
            sources=(
                mx.SampledPlaneCurrentAntennaPlan(
                    bridge, 0, 0.5, [-1.0, 3.0], [-1.0, 50.0], times, electric
                ),
            ),
        ).prepare()
    with pytest.raises(ValueError, match="interior node planes"):
        phx.solver.CompatibleMaxwellPlan(
            bridge, sources=(plan(plane_coordinate=0.0),)
        ).prepare()
    with pytest.raises(ValueError, match="declared homogeneous"):
        phx.solver.CompatibleMaxwellPlan(
            bridge,
            constitutive=mx.DiagonalMaxwellConstitutivePlan(permittivity=2.0),
            sources=(plan(),),
        ).prepare()
