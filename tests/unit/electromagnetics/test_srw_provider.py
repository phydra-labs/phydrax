#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""SRW single-electron oracle: deck translation, import convention, live agreement.

Independent references: CODATA 2022 SI constants from :mod:`scipy.constants`;
SRW's engine wavenumber constant ``2π · 0.80654658 µm⁻¹ per eV``
(``srradint.cpp``); SRW's field normalization
``|E|² = photons/s/0.1%bw/mm²`` for the beam current, which gives one electron
``d²W/(dω dΩ) = 10⁹ ħ (e/I) D² |E|²``; the closed-form insertion-device profile
``B₀ g(u) cos(k_u u)`` with ``g(u) = ½[tanh((u + a)/w) − tanh((u − a)/w)]``.

``tests/data/providers/srw/planar_undulator_field.json`` was produced by the
real SRW (srwpy 4.2.1, CPython 3.12.8, macOS arm64) through :func:`run_srw` on
2026-09-28 (``python srw_driver.py srw_input.json``); its sidecar
``planar_undulator_field.provenance.json`` records the case, the interpreter
digest, and the output digest.

Live tests run when ``PHYDRAX_SRW_PYTHON`` and ``PHYDRAX_SRW_PYTHON_VERSION``
name a pinned interpreter with ``srwpy``; they compare SRW with
:class:`TrajectoryRadiationPlan` on an X1a-tracked ten-period undulator. The
1 % tolerance on the spectral energy covers the leapfrog velocity error
``(k_u c Δt)²/8 ≈ 3·10⁻⁴`` at 128 steps per period, SRW's 10⁻⁵ integration
tolerance, and the cubic interpolation of the exported field table.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants

from phydrax import DimensionalScaleContract, ElectromagneticScaleContract
from phydrax._external_runtime import pin_executable
from phydrax.applications import accelerator
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    read_srw_output,
    run_srw,
    srw_input,
    SRWFieldMapSource,
    SRWProvider,
    TrajectoryRadiationPlan,
    TrajectoryRadiationRoute,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import CHARGE, KILOGRAM, LENGTH, TIME, UnitDefinition


SI = ElectromagneticScaleContract.si()
LIGHT = constants.c
ELECTRON_CHARGE = constants.e
ELECTRON_MASS = constants.m_e
SRW_WAVENUMBER_PER_EV = 2.0 * math.pi * 0.80654658e6
PERIOD = 0.02
GAMMA = 100.0
DEFLECTION = 1.0
PERIODS = 10
PEAK_FIELD = (
    DEFLECTION * 2.0 * math.pi * ELECTRON_MASS * LIGHT / (ELECTRON_CHARGE * PERIOD)
)
MOMENTUM = math.sqrt(GAMMA**2 - 1.0) * ELECTRON_MASS * LIGHT
FUNDAMENTAL = (
    2.0 * GAMMA**2 * (2.0 * math.pi * LIGHT / PERIOD) / (1.0 + DEFLECTION**2 / 2)
)
OFF_AXIS = 0.3 / GAMMA
REFERENCE = Path(__file__).parents[2] / "data" / "providers" / "srw"


def _code_units(length_si: Fraction) -> ElectromagneticScaleContract:
    """Electron-normalized code units: c = e = mₑ = 1 with length unit ``length_si``."""
    light = SI.speed_of_light
    time_si = length_si / light
    mass_si = SI.electron_mass
    charge_si = SI.elementary_charge
    relativity = SI.relativity
    return ElectromagneticScaleContract.code_units(
        DimensionalScaleContract(
            UnitDefinition("L0", LENGTH, "si", length_si),
            UnitDefinition("m_e", KILOGRAM.dimension, "si", mass_si),
            UnitDefinition("T0", TIME, "si", time_si),
        ),
        UnitDefinition("q_e", CHARGE, "si", charge_si),
        gravitational_constant=relativity.gravitational_constant
        * mass_si
        * time_si**2
        / length_si**3,
        speed_of_light=1,
        reduced_planck_constant=SI.reduced_planck_constant
        * time_si
        / (mass_si * length_si**2),
        boltzmann_constant=relativity.boltzmann_constant
        * time_si**2
        / (mass_si * length_si**2),
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=SI.vacuum_permittivity
        * length_si**3
        * mass_si
        / (charge_si**2 * time_si**2),
        constant_set_id="codata-2022",
    )


def _directions() -> np.ndarray:
    return np.array([[0.0, 0.0, 1.0], [math.sin(OFF_AXIS), 0.0, math.cos(OFF_AXIS)]])


def _plan(
    scale: ElectromagneticScaleContract,
    frequencies: np.ndarray,
    *,
    directions: np.ndarray | None = None,
    route: TrajectoryRadiationRoute = "segment-exact",
) -> TrajectoryRadiationPlan:
    return TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(
            _directions() if directions is None else directions,
            np.array([1.0, 0.0, 0.0]),
        ),
        frequencies,
        coherence="coherent",
        route=route,
    )


def _undulator(
    polarization: accelerator.InsertionDevicePolarization = "planar",
) -> accelerator.InsertionDeviceField:
    return accelerator.InsertionDeviceField(
        PEAK_FIELD,
        PERIOD,
        PERIODS,
        polarization=polarization,
        center=0.0,
        aperture=(1.0e-3, 1.0e-3),
        ramp_periods=0.25,
    )


def _tracking(
    elements: tuple[
        accelerator.DipoleBendField
        | accelerator.InsertionDeviceField
        | accelerator.TabulatedFieldMap,
        ...,
    ],
    half: float,
    *,
    steps: int = 2,
    reference_time: float = 0.0,
) -> accelerator.FieldMapTrackingPlan:
    time_step = PERIOD / (LIGHT * 128)
    return accelerator.FieldMapTrackingPlan(
        accelerator.FieldMapBeamline(
            SI, elements, element_ids=tuple(f"e{index}" for index in range(len(elements)))
        ),
        entrance_plane=-half,
        exit_plane=half,
        time_step=time_step,
        step_count=steps,
        reference_time=reference_time,
    )


def _bunch(
    coordinates: tuple[float, ...] = (0.0,) * 6, *, charge: float = -ELECTRON_CHARGE
) -> accelerator.AcceleratorBunch:
    return accelerator.AcceleratorBunch(
        jnp.asarray([coordinates], dtype=jnp.float64),
        jnp.ones((1,)),
        jnp.zeros((1,), dtype=jnp.int32),
        reference_rest_energy=ELECTRON_MASS * LIGHT**2,
        reference_momentum=MOMENTUM,
        reference_charge=charge,
        bunch_id="electron",
    )


def _lane(
    scale: ElectromagneticScaleContract,
    *,
    samples: int = 9,
    charge: float = -1.0,
    times: np.ndarray | None = None,
    lanes: int = 1,
) -> ChargedTrajectory:
    """A tilted uniform electron in code units: u = (0.02, −0.01, 5) c."""
    step = 0.25
    t = 3.0 + step * np.arange(samples) if times is None else times
    proper = np.array([0.02, -0.01, 5.0])
    velocity = proper / math.sqrt(1.0 + proper @ proper)
    start = np.array([1.0e-3, 2.0e-3, -1.0])
    positions = start + (t - t[0])[:, None] * velocity
    return ChargedTrajectory(
        t,
        np.repeat(positions[:, None, :], lanes, axis=1),
        np.broadcast_to(proper, (samples, lanes, 3)).copy(),
        np.full((lanes,), charge),
        np.ones((lanes,)),
        np.ones((samples, lanes), dtype=bool),
        (np.zeros((lanes,), dtype=np.uint32), np.arange(lanes, dtype=np.uint32)),
    )


def _deck(inputs: dict[str, bytes]) -> dict:
    return json.loads(inputs["srw_input.json"])


def test_trajectory_deck_converts_code_units_to_srw_si() -> None:
    length_si = Fraction(1, 1000)
    scale = _code_units(length_si)
    length = float(length_si)
    time = length / LIGHT
    frequencies = np.linspace(2.0, 3.0, 5)
    trajectory = _lane(scale)
    deck = _deck(
        srw_input(_plan(scale, frequencies), trajectory, observation_distance=5.0e4)
    )
    times = 3.0 + 0.25 * np.arange(9)
    proper = np.array([0.02, -0.01, 5.0])
    gamma = math.sqrt(1.0 + proper @ proper)
    beta = proper / gamma
    positions = np.asarray(trajectory.positions[:, 0]) * length
    source = deck["source"]
    assert source["kind"] == "trajectory"
    assert source["ct_start"] == 0.0
    assert source["ct_end"] == pytest.approx(
        LIGHT * (times[-1] - times[0]) * time, rel=1e-12
    )
    np.testing.assert_allclose(source["z"], positions[:, 2], rtol=1e-12)
    np.testing.assert_allclose(source["x"], positions[:, 0], rtol=1e-12)
    np.testing.assert_allclose(source["beta_x"], np.full(9, beta[0]), rtol=1e-12)
    np.testing.assert_allclose(source["beta_z"], np.full(9, beta[2]), rtol=1e-12)
    assert deck["electron"]["gamma"] == pytest.approx(gamma, rel=1e-12)
    # SRW wavenumber E · 2π · 0.80654658 µm⁻¹ equals ω/c in SI.
    energies = frequencies / time / (LIGHT * SRW_WAVENUMBER_PER_EV)
    assert deck["energy_start"] == pytest.approx(energies[0], rel=1e-12)
    assert deck["energy_end"] == pytest.approx(energies[-1], rel=1e-12)
    assert deck["energy_count"] == 5
    origin = np.array([0.0, 0.0, 0.5 * (positions[0, 2] + positions[-1, 2])])
    points = origin + 5.0e4 * length * _directions()
    np.testing.assert_allclose(deck["points"], points, rtol=1e-12, atol=1e-15)
    assert deck["distance"] == pytest.approx(50.0, rel=1e-12)
    # SRW's clock reads zero where the initial uniform motion crosses z = 0.
    origin_time = times[0] * time - positions[0, 2] / (beta[2] * LIGHT)
    assert deck["time_origin"] == pytest.approx(origin_time, rel=1e-12)


def test_tabulated_field_map_deck_samples_the_beamline_in_tesla() -> None:
    device = _undulator()
    half = 0.12
    source = SRWFieldMapSource(
        _tracking((device,), half),
        _bunch((0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.01)),
        "tabulated",
        longitudinal_samples=33,
    )
    deck = _deck(
        srw_input(
            _plan(SI, FUNDAMENTAL * np.linspace(0.9, 1.1, 3)),
            source,
            observation_distance=100.0,
        )
    )
    field = deck["source"]
    nodes = np.linspace(-half, half, 33)
    flat = 0.5 * PERIODS * PERIOD
    width = 0.25 * PERIOD
    window = 0.5 * (np.tanh((nodes + flat) / width) - np.tanh((nodes - flat) / width))
    expected = PEAK_FIELD * window * np.cos(2.0 * math.pi * nodes / PERIOD)
    assert field["shape"] == [1, 1, 33]
    assert field["ranges"] == pytest.approx([0.0, 0.0, 2.0 * half], rel=1e-12)
    assert field["center"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)
    np.testing.assert_allclose(field["by"], expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(field["bx"], np.zeros(33))
    np.testing.assert_allclose(field["bz"], np.zeros(33), atol=1e-12)
    momentum = MOMENTUM * 1.01
    gamma = math.hypot(1.0, momentum / (ELECTRON_MASS * LIGHT))
    beta = math.sqrt(1.0 - gamma**-2)
    electron = deck["electron"]
    assert electron["gamma"] == pytest.approx(gamma, rel=1e-12)
    assert electron["beta_x"] == pytest.approx(beta * 1.0e-4 / 1.01, rel=1e-12)
    assert electron["beta_y"] == 0.0
    assert electron["z"] == pytest.approx(-half, rel=1e-12)
    assert deck["z_start"] == pytest.approx(-half, rel=1e-12)
    assert deck["z_end"] == pytest.approx(half, rel=1e-12)


def test_helical_ideal_undulator_deck_keeps_field_period_and_handedness() -> None:
    device = _undulator("helical")
    deck = _deck(
        srw_input(
            _plan(SI, FUNDAMENTAL * np.linspace(0.9, 1.1, 3)),
            SRWFieldMapSource(_tracking((device,), 0.12), _bunch(), "ideal-undulator"),
            observation_distance=100.0,
        )
    )
    field = deck["source"]
    # On axis (B₀ sin k_u z, B₀ cos k_u z): vertical cosine (SRW symmetry 1) and
    # horizontal sine (symmetry −1).
    planes = [[plane, symmetry] for plane, _, symmetry in field["harmonics"]]
    assert planes == [["v", 1], ["h", -1]]
    amplitudes = [amplitude for _, amplitude, _ in field["harmonics"]]
    assert amplitudes == pytest.approx([PEAK_FIELD, PEAK_FIELD], rel=1e-12)
    assert field["period"] == PERIOD
    assert field["period_count"] == PERIODS
    assert field["center"] == [0.0, 0.0, 0.0]


def _refusal_cases() -> dict[str, Callable[[], object]]:
    si_plan = _plan(SI, FUNDAMENTAL * np.linspace(0.9, 1.1, 3))
    device = _undulator()
    return dict(
        [
            (
                "form-factor",
                lambda: srw_input(
                    TrajectoryRadiationPlan(
                        SI,
                        RadiationObserverPlan(_directions(), np.array([1.0, 0.0, 0.0])),
                        FUNDAMENTAL * np.linspace(0.9, 1.1, 3),
                        coherence="gaussian-form-factor",
                        route="segment-exact",
                        bunch_sigma=np.array([1e-5, 1e-5, 1e-5]),
                    ),
                    SRWFieldMapSource(_tracking((device,), 0.12), _bunch()),
                    observation_distance=100.0,
                ),
            ),
            (
                "nonuniform-frequencies",
                lambda: srw_input(
                    _plan(SI, FUNDAMENTAL * np.array([0.9, 1.0, 1.2])),
                    SRWFieldMapSource(_tracking((device,), 0.12), _bunch()),
                    observation_distance=100.0,
                ),
            ),
            (
                "backward-direction",
                lambda: srw_input(
                    _plan(
                        SI,
                        FUNDAMENTAL * np.linspace(0.9, 1.1, 3),
                        directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.1, -1.0]]),
                    ),
                    SRWFieldMapSource(_tracking((device,), 0.12), _bunch()),
                    observation_distance=100.0,
                ),
            ),
            (
                "two-lanes",
                lambda: srw_input(
                    _plan(_code_units(Fraction(1, 1000)), np.linspace(2.0, 3.0, 3)),
                    _lane(_code_units(Fraction(1, 1000)), lanes=2),
                    observation_distance=1e4,
                ),
            ),
            (
                "positron-lane",
                lambda: srw_input(
                    _plan(_code_units(Fraction(1, 1000)), np.linspace(2.0, 3.0, 3)),
                    _lane(_code_units(Fraction(1, 1000)), charge=1.0),
                    observation_distance=1e4,
                ),
            ),
            (
                "nonuniform-times",
                lambda: srw_input(
                    _plan(_code_units(Fraction(1, 1000)), np.linspace(2.0, 3.0, 3)),
                    _lane(
                        _code_units(Fraction(1, 1000)),
                        samples=4,
                        times=np.array([0.0, 0.25, 0.5, 0.8]),
                    ),
                    observation_distance=1e4,
                ),
            ),
            (
                "late-electron",
                lambda: srw_input(
                    si_plan,
                    SRWFieldMapSource(
                        _tracking((device,), 0.12),
                        _bunch((0.0, 0.0, 0.0, 0.0, 1e-6, 0.0)),
                    ),
                    observation_distance=100.0,
                ),
            ),
            (
                "positron-bunch",
                lambda: srw_input(
                    si_plan,
                    SRWFieldMapSource(
                        _tracking((device,), 0.12), _bunch(charge=ELECTRON_CHARGE)
                    ),
                    observation_distance=100.0,
                ),
            ),
            (
                "electric-map",
                lambda: srw_input(
                    si_plan,
                    SRWFieldMapSource(
                        _tracking(
                            (
                                accelerator.TabulatedFieldMap(
                                    np.linspace(-1e-3, 1e-3, 3),
                                    np.linspace(-1e-3, 1e-3, 3),
                                    np.linspace(-0.1, 0.1, 3),
                                    np.zeros((3, 3, 3, 3)),
                                    electric=np.ones((3, 3, 3, 3)),
                                ),
                            ),
                            0.1,
                        ),
                        _bunch(),
                    ),
                    observation_distance=100.0,
                ),
            ),
            (
                "ideal-undulator-with-bend",
                lambda: srw_input(
                    si_plan,
                    SRWFieldMapSource(
                        _tracking(
                            (
                                device,
                                accelerator.DipoleBendField(
                                    0.01,
                                    0.05,
                                    center=0.3,
                                    fringe_width=0.01,
                                    aperture=(1e-3, 1e-3),
                                ),
                            ),
                            0.12,
                        ),
                        _bunch(),
                        "ideal-undulator",
                    ),
                    observation_distance=100.0,
                ),
            ),
            (
                "table-beyond-support",
                lambda: srw_input(
                    si_plan,
                    SRWFieldMapSource(
                        _tracking((device,), 0.12),
                        _bunch(),
                        "tabulated",
                        transverse_samples=(4, 4),
                        transverse_half_extent=(2e-3, 2e-3),
                    ),
                    observation_distance=100.0,
                ),
            ),
            (
                "two-transverse-samples",
                lambda: SRWFieldMapSource(
                    _tracking((device,), 0.12),
                    _bunch(),
                    "tabulated",
                    transverse_samples=(2, 1),
                    transverse_half_extent=(1e-4, 0.0),
                ),
            ),
        ]
    )


@pytest.mark.parametrize(
    "case",
    [
        "form-factor",
        "nonuniform-frequencies",
        "backward-direction",
        "two-lanes",
        "positron-lane",
        "nonuniform-times",
        "late-electron",
        "positron-bunch",
        "electric-map",
        "ideal-undulator-with-bend",
        "table-beyond-support",
        "two-transverse-samples",
    ],
)
def test_out_of_subset_inputs_are_refused_before_running(case: str) -> None:
    build = _refusal_cases()[case]
    with pytest.raises(ValueError):
        build()


def test_wrong_kinds_are_type_errors() -> None:
    with pytest.raises(TypeError):
        SRWProvider("/usr/bin/python3")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        srw_input(
            _plan(SI, FUNDAMENTAL * np.linspace(0.9, 1.1, 3)),
            _bunch(),  # ty: ignore[invalid-argument-type]
            observation_distance=100.0,
        )


def _reference_plan() -> TrajectoryRadiationPlan:
    return _plan(SI, FUNDAMENTAL * np.linspace(0.96, 1.04, 5))


def test_recorded_srw_field_imports_as_single_electron_spectral_energy() -> None:
    data = (REFERENCE / "planar_undulator_field.json").read_bytes()
    plan = _reference_plan()
    spectrum = read_srw_output(data, plan)
    document = json.loads(data)
    distance = document["distance"]
    current = document["current"]
    directions = _directions()
    expected = np.empty((5, 2))
    for index, wavefront in enumerate(document["wavefronts"]):
        ex = np.asarray(wavefront["ex"][0::2]) + 1j * np.asarray(wavefront["ex"][1::2])
        ey = np.asarray(wavefront["ey"][0::2]) + 1j * np.asarray(wavefront["ey"][1::2])
        n = directions[index]
        ez = -(n[0] * ex + n[1] * ey) / n[2]
        intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
        # photons/0.1%bw/mm² per electron → J s/sr: ħω / (10⁻³ ω) · (10³ D)².
        expected[:, index] = (
            1.0e9 * constants.hbar * ELECTRON_CHARGE / current * distance**2 * intensity
        )
    np.testing.assert_allclose(spectrum.spectral_energy, expected, rtol=1e-9)
    np.testing.assert_allclose(
        spectrum.spectral_energy,
        constants.epsilon_0
        * LIGHT
        * np.sum(np.abs(spectrum.field_spectrum) ** 2, -1)
        / math.pi,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(spectrum.angular_frequencies, plan.angular_frequencies)
    assert spectrum.observation_distance == pytest.approx(100.0, rel=1e-12)
    # A planar undulator radiates horizontally on axis: e2 = ŷ carries nothing.
    assert np.all(np.abs(spectrum.field_spectrum[:, 0, 1]) == 0.0)


def test_recorded_srw_field_is_refused_for_another_plan() -> None:
    data = (REFERENCE / "planar_undulator_field.json").read_bytes()
    with pytest.raises(ValueError, match="requested mesh"):
        read_srw_output(data, _plan(SI, FUNDAMENTAL * np.linspace(0.95, 1.05, 5)))
    with pytest.raises(ValueError, match="observers"):
        read_srw_output(
            data,
            _plan(
                SI,
                FUNDAMENTAL * np.linspace(0.96, 1.04, 5),
                directions=np.array([[0.0, 0.0, 1.0]]),
            ),
        )


def _pinned_srw() -> SRWProvider:
    path = os.environ.get("PHYDRAX_SRW_PYTHON")
    version = os.environ.get("PHYDRAX_SRW_PYTHON_VERSION")
    if path is None or version is None:
        pytest.skip(
            "set PHYDRAX_SRW_PYTHON and PHYDRAX_SRW_PYTHON_VERSION to a pinned "
            "interpreter with srwpy"
        )
    return SRWProvider(pin_executable(path, version=version, license_id="EPICS"))


class _Tracked:
    def __init__(self, polarization: accelerator.InsertionDevicePolarization) -> None:
        device = _undulator(polarization)
        half = 0.5 * PERIODS * PERIOD + 8 * device.ramp_width
        time_step = PERIOD / (LIGHT * 128)
        self.tracking = _tracking(
            (device,),
            half,
            steps=math.ceil(1.01 * 2.0 * half / (LIGHT * time_step)) + 4,
        )
        self.bunch = _bunch()
        self.result = accelerator.track_field_map(self.tracking, self.bunch)
        # On-axis resonance 2γ²ω_u/(1 + K²/2) (planar) or 2γ²ω_u/(1 + K²) (helical).
        fundamental = (
            FUNDAMENTAL
            if polarization == "planar"
            else FUNDAMENTAL * (1.0 + DEFLECTION**2 / 2) / (1.0 + DEFLECTION**2)
        )
        self.plan = _plan(
            SI, fundamental * np.linspace(0.85, 1.15, 61), route="segment-hermite"
        )
        self.reference = self.plan.prepare().evaluate(self.result.trajectory)


@pytest.fixture(scope="module")
def planar() -> _Tracked:
    return _Tracked("planar")


def _relative_error(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.max(np.abs(values - reference), axis=0) / np.max(np.abs(reference), axis=0)


@pytest.mark.parametrize("route", ["tabulated", "trajectory"])
def test_srw_undulator_spectrum_matches_trajectory_radiation(
    planar: _Tracked, route: str, tmp_path: Path
) -> None:
    provider = _pinned_srw()
    source = (
        planar.result.trajectory
        if route == "trajectory"
        else SRWFieldMapSource(planar.tracking, planar.bunch, "tabulated")
    )
    result = run_srw(provider, planar.plan, source, tmp_path, observation_distance=1000.0)
    reference = planar.reference
    assert bool(reference.evidence.resolved)
    energy = _relative_error(
        result.spectrum.spectral_energy, np.asarray(reference.spectral_energy)
    )
    assert np.all(energy < 1.0e-2)
    # Phase-inclusive agreement of the dominant horizontal polarization.
    field = _relative_error(
        result.spectrum.field_spectrum[..., 0],
        np.asarray(reference.field_spectrum)[..., 0],
    )
    assert np.all(field < 1.0e-2)
    report = result.report
    assert report.status == AdapterStatus.DECLARED_LOSS and report.losses
    assert report.source_id == result.output_sha256
    assert result.license_id == "EPICS"
    assert result.provider_version == provider.executable.version
    assert result.executable_sha256 == provider.executable.sha256


def test_srw_helical_undulator_handedness_matches_trajectory_radiation(
    tmp_path: Path,
) -> None:
    provider = _pinned_srw()
    helical = _Tracked("helical")
    source = SRWFieldMapSource(
        helical.tracking, helical.bunch, "tabulated", 4096, (5, 5), (2.0e-4, 2.0e-4)
    )
    result = run_srw(
        provider, helical.plan, source, tmp_path, observation_distance=1000.0
    )
    reference = helical.reference
    field = result.spectrum.field_spectrum[:, 0]
    peak = int(np.argmax(np.asarray(reference.spectral_energy[:, 0])))
    stokes_v = -2.0 * np.imag(field[peak, 0] * np.conj(field[peak, 1]))
    intensity = np.sum(np.abs(field[peak]) ** 2)
    expected = float(reference.stokes[peak, 0, 3] / reference.stokes[peak, 0, 0])
    assert abs(expected) > 0.99
    assert stokes_v / intensity == pytest.approx(expected, abs=1.0e-3)
    energy = _relative_error(
        result.spectrum.spectral_energy, np.asarray(reference.spectral_energy)
    )
    assert np.all(energy < 1.0e-2)
