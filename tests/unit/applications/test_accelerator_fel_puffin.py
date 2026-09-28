#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned Puffin oracle for the one-dimensional time-dependent FEL.

Deck contracts are checked against Puffin's scaled frame computed
independently from its published definitions (Campbell & McNeil, Phys.
Plasmas 19, 093119, 2012; Puffin manual §"Analytic Equations"): Pierce
``ρ = (a_u ω_p/(4ck_u))^(2/3)/γ`` (helical) or with the planar Bessel
coupling, ``l_g = λ_u/(4πρ)``, ``l_c = λ/(4πρ)``, ``κ = a_u/(2ργ_r)``, the
scaled field ``A = eκl_g E/(γ_r mₑc²)`` with intensity ``I = cε₀|E|²``, and
the exact resonance ``λ = λ_u(1 − β_z)/β_z``. The reader is checked on genuine
Puffin records stored in ``tests/data/providers/puffin`` (provenance in
``provenance.json``: Puffin 2.1.0a commit 157f473, the command, and the date)
against the declared seed's analytic slot averages and Puffin's own SI power.
The live oracle compares a seeded amplifier with :class:`FELTimeDependentPlan`.
"""

from __future__ import annotations

import gzip
import math
import os
import re
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants, special

from phydrax import ElectromagneticScaleContract
from phydrax._external_runtime import pin_executable
from phydrax.applications.accelerator import InsertionDeviceField
from phydrax.applications.accelerator.fel import (
    FELBeamSlices,
    FELLoading,
    FELPlan,
    FELPulseSeed,
    FELTimeDependentPlan,
    FELUndulatorLattice,
    FELUndulatorSegment,
    Genesis4GaussianSeed,
    puffin_input,
    PuffinGaussianSeed,
    PuffinProvider,
    read_puffin_power,
    run_puffin,
)
from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.interchange import AdapterStatus
from phydrax.optics.wave import (
    AngularSpectrumPlan,
    PlaneFieldSpace,
    PulseEnvelopeField,
    PulseTimeSpace,
)


SCALE = ElectromagneticScaleContract.si()
LIGHT = constants.c
CHARGE = constants.e
MASS = constants.m_e
PERMITTIVITY = constants.epsilon_0
PERIOD = 0.03
GAMMA = 100.0
SIGMA = 100.0e-6
EMITTANCE = 1.0e-9
# β = σ²γ/ε_n keeps σ = 100 µm with negligible betatron angles (1e-7 rad).
BETA = SIGMA**2 * GAMMA / EMITTANCE
AREA = 2.0 * math.pi * SIGMA**2
DATA = Path(__file__).resolve().parents[2] / "data" / "providers" / "puffin"


def _lattice(
    polarization: str, deflections: tuple[float, ...], periods: int
) -> FELUndulatorLattice:
    segments = []
    for deflection in deflections:
        peak = deflection * 2.0 * math.pi * MASS * LIGHT / (CHARGE * PERIOD)
        device = InsertionDeviceField(
            peak,
            PERIOD,
            periods,
            polarization=polarization,  # ty: ignore[invalid-argument-type]
            center=0.0,
            aperture=(1e-2, 1e-2),
        )
        segments.append(FELUndulatorSegment(device))
    return FELUndulatorLattice(SCALE, tuple(segments), step_length=PERIOD)


def _slices(
    count: int, spacing: float, current: float, *, spread: float = 0.0
) -> FELBeamSlices:
    return FELBeamSlices(
        np.arange(count) * spacing,
        np.full(count, current),
        np.full(count, GAMMA),
        np.full(count, spread),
        np.full((count, 2), EMITTANCE),
        np.full((count, 2), BETA),
        np.zeros((count, 2)),
    )


def _plan(
    lattice: FELUndulatorLattice, *, shot_noise: str = "quiet"
) -> FELTimeDependentPlan:
    wavelength = lattice.resonant_wavelength(GAMMA)
    core = FELPlan(
        lattice,
        wavelength,
        loading=FELLoading(2, 8, shot_noise=shot_noise),  # ty: ignore[invalid-argument-type]
        slice_batch=64,
    )
    return FELTimeDependentPlan(core, slippage="commensurate", boundary="open")


def _namelist(text: bytes) -> dict[str, str]:
    return {
        match.group(1): match.group(2).strip()
        for match in re.finditer(r"^\s*(\w+)\s*=\s*(.+)$", text.decode(), re.MULTILINE)
    }


def _numbers(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float64)


def _pierce(coupling: float, current: float) -> float:
    """``ρ³ = e²κ²n/(8ε₀mₑc²γ³k_u²)`` with ``κ = a_w[JJ]/√2``, ``n = I/(ecA)``."""
    density = current / (CHARGE * LIGHT * AREA)
    wavenumber = 2.0 * math.pi / PERIOD
    return (
        CHARGE**2
        * coupling**2
        * density
        / (8.0 * PERMITTIVITY * MASS * LIGHT**2 * GAMMA**3 * wavenumber**2)
    ) ** (1.0 / 3.0)


def _reference_gamma(wavelength: float, rms_squared: float) -> float:
    beta = 1.0 / (1.0 + wavelength / PERIOD)
    return math.sqrt((1.0 + rms_squared) / (1.0 - beta * beta))


def _gaussian_slots(
    count: int, spacing: float, power: float, center: float, rms: float
) -> np.ndarray:
    edges = (np.arange(count + 1) - 0.5) * spacing
    scaled = (edges - center) / (math.sqrt(2.0) * rms)
    return power * rms * math.sqrt(0.5 * math.pi) * np.diff(special.erf(scaled)) / spacing


# ------------------------------------------------------------------ decks


def test_helical_deck_uses_the_independent_scaled_frame() -> None:
    lattice = _lattice("helical", (1.0,), 40)
    wavelength = PERIOD * 2.0 / (2.0 * GAMMA**2)
    plan = _plan(lattice)
    slices = _slices(32, wavelength, 7.5)
    seed = PuffinGaussianSeed(
        1.0e3, center_position=16.0 * wavelength, rms_length=8.0 * wavelength
    )
    deck = puffin_input(
        plan,
        slices,
        seed=seed,
        steps_per_period=24,
        nodes_per_wavelength=11,
        macroparticles_per_wavelength=12,
        record_periods=8,
    )
    main, beam, seed_file = (
        _namelist(deck[name]) for name in ("puffin.in", "beam.in", "seed.in")
    )
    rho = _pierce(1.0 / math.sqrt(2.0), 7.5)
    gamma_r = _reference_gamma(wavelength, 1.0)
    gain = PERIOD / (4.0 * math.pi * rho)
    cooperation = wavelength / (4.0 * math.pi * rho)
    assert float(main["srho"]) == pytest.approx(rho, rel=1e-12)
    assert float(main["sgamma_r"]) == pytest.approx(gamma_r, rel=1e-12)
    assert float(main["saw"]) == pytest.approx(1.0, rel=1e-12)
    assert float(main["lambda_w"]) == pytest.approx(PERIOD, rel=1e-15)
    assert main["zundType"] == "'helical'"
    assert (main["qOneD"], main["qscaled"], main["q_noise"]) == (
        ".true.",
        ".true.",
        ".false.",
    )
    assert (int(main["nodesPerLambdar"]), int(main["iWriteIntNthSteps"])) == (11, 8 * 24)
    # The seed's field σ is √2 × the power rms; its ±3.75σ extent must stay on
    # the mesh, so the beam head sits behind it by the head margin h̄.
    field_sigma = math.sqrt(2.0) * 8.0 * wavelength / cooperation
    offset = 16.5 * wavelength / cooperation
    margin = 3.75 * field_sigma - offset
    assert margin > 0.0
    length = 32 * wavelength / cooperation
    assert float(seed_file["meanZ2"]) == pytest.approx(margin + offset, rel=1e-12)
    assert _numbers(seed_file["sSigmaF"])[2] == pytest.approx(field_sigma, rel=1e-12)
    assert (
        float(main["sFModelLengthZ2"]) >= margin + length + 40 * wavelength / cooperation
    )
    sigma_bar = SIGMA / math.sqrt(gain * cooperation)
    np.testing.assert_allclose(_numbers(beam["sSigmaE"])[:2], sigma_bar, rtol=1e-12)
    assert _numbers(beam["sLenE"])[2] == pytest.approx(length, rel=1e-12)
    assert float(beam["bcenter"]) == pytest.approx(margin + 0.5 * length, rel=1e-12)
    assert float(beam["Ipk"]) == pytest.approx(7.5, rel=1e-15)
    assert float(beam["gammaf"]) == pytest.approx(GAMMA / gamma_r, rel=1e-12)
    assert (beam["qOneDCold"], int(beam["iMPsZ2PerWave"])) == (".true.", 12)
    # |A|² = (eκl_g/(γ_r mₑc²))² P/(cε₀ area), split over the two circular components.
    coupling = 1.0 / (2.0 * rho * gamma_r)
    scaled = (CHARGE * coupling * gain / (gamma_r * MASS * LIGHT**2)) ** 2 * (
        1.0e3 / (LIGHT * PERMITTIVITY * AREA)
    )
    assert float(seed_file["sA0_X"]) == pytest.approx(0.5 * scaled, rel=1e-9)
    assert float(seed_file["sA0_Y"]) == pytest.approx(0.5 * scaled, rel=1e-9)
    assert deck["puffin.latt"].decode().split() == [
        "UN",
        "'helical'",
        "40",
        "1.0",
        "0.0",
        "24",
        "1.0",
        "1.0",
        "0.0",
        "0.0",
    ]


def test_planar_tapered_deck_maps_modules_bessel_coupling_and_spread() -> None:
    lattice = _lattice("planar", (1.5, 1.4), 20)
    wavelength = PERIOD * (1.0 + 1.5**2 / 2.0) / (2.0 * GAMMA**2)
    plan = _plan(lattice, shot_noise="fawley")
    slices = _slices(16, 2.0 * wavelength, 20.0, spread=1.0e-3)
    deck = puffin_input(plan, slices, seed=None, energy_macroparticles=9)
    main, beam = _namelist(deck["puffin.in"]), _namelist(deck["beam.in"])
    xi = 1.5**2 / (4.0 + 2.0 * 1.5**2)
    orders = special.jv(np.asarray([0.0, 1.0]), np.full(2, xi))
    bessel = float(orders[0] - orders[1])
    rho = _pierce(1.5 / math.sqrt(2.0) * bessel / math.sqrt(2.0), 20.0)
    gamma_r = _reference_gamma(wavelength, 1.5**2 / 2.0)
    assert float(main["srho"]) == pytest.approx(rho, rel=1e-10)
    assert float(main["sgamma_r"]) == pytest.approx(gamma_r, rel=1e-12)
    assert float(main["saw"]) == pytest.approx(1.5, rel=1e-12)
    assert (main["zundType"], main["seed_file"], main["q_noise"]) == (
        "'planepole'",
        "''",
        ".true.",
    )
    assert "seed.in" not in deck
    assert beam["qOneDCold"] == ".false." and int(beam["inmps1DGam"]) == 9
    assert _numbers(beam["sSigmaE"])[5] == pytest.approx(
        1.0e-3 * GAMMA / gamma_r, rel=1e-12
    )
    modules = [line.split() for line in deck["puffin.latt"].decode().splitlines()]
    assert [module[2] for module in modules] == ["20", "20"]
    np.testing.assert_allclose(
        [float(module[3]) for module in modules], [1.0, 1.4 / 1.5], rtol=1e-12
    )


# ---------------------------------------------------------------- refusals


def _refusal_case(name: str) -> tuple[FELTimeDependentPlan, FELBeamSlices]:
    lattice = _lattice("helical", (1.0,), 40)
    wavelength = lattice.resonant_wavelength(GAMMA)
    slices = _slices(16, wavelength, 7.5)
    loading = FELLoading(2, 8, shot_noise="quiet")
    match name:
        case "grid":
            space = PlaneFieldSpace(
                TensorGridPlan(
                    (UniformAxisSpec(9), UniformAxisSpec(9)), axis_names=("x", "y")
                ).prepare(jnp.asarray([[-1e-3, -1e-3], [1e-3, 1e-3]])),
                RigidFrame.identity(3),
                "finite-window",
            )
            core = FELPlan(
                lattice,
                wavelength,
                loading=loading,
                transverse="angular-spectrum",
                field_space=space,
                propagation=AngularSpectrumPlan(4),
            )
            return FELTimeDependentPlan(
                core, slippage="commensurate", boundary="open"
            ), slices
        case "harmonics":
            # Odd planar harmonics couple on axis (helical ones do not).
            planar = _lattice("planar", (1.0,), 40)
            core = FELPlan(
                planar,
                planar.resonant_wavelength(GAMMA),
                loading=loading,
                harmonics=(1, 3),
            )
            return FELTimeDependentPlan(
                core, slippage="commensurate", boundary="open"
            ), slices
        case "periodic":
            core = FELPlan(lattice, wavelength, loading=loading)
            return FELTimeDependentPlan(
                core, slippage="commensurate", boundary="periodic"
            ), slices
        case "drift":
            peak = 2.0 * math.pi * MASS * LIGHT / (CHARGE * PERIOD)
            device = InsertionDeviceField(
                peak,
                PERIOD,
                40,
                polarization="helical",
                center=0.0,
                aperture=(1e-2, 1e-2),
            )
            broken = FELUndulatorLattice(
                SCALE,
                (
                    FELUndulatorSegment(device, drift_length=0.1),
                    FELUndulatorSegment(device),
                ),
                step_length=PERIOD,
            )
            return _plan(broken), slices
        case "periods":
            peak = 2.0 * math.pi * MASS * LIGHT / (CHARGE * PERIOD)
            devices = tuple(
                InsertionDeviceField(
                    peak * PERIOD / period,
                    period,
                    40,
                    polarization="helical",
                    center=0.0,
                    aperture=(1e-2, 1e-2),
                )
                for period in (PERIOD, 1.1 * PERIOD)
            )
            mixed = FELUndulatorLattice(
                SCALE,
                tuple(FELUndulatorSegment(device) for device in devices),
                step_length=PERIOD,
            )
            return _plan(mixed), slices
        case "slices":
            ramp = FELBeamSlices(
                np.arange(16) * wavelength,
                np.linspace(5.0, 7.5, 16),
                np.full(16, GAMMA),
                np.zeros(16),
                np.full((16, 2), EMITTANCE),
                np.full((16, 2), BETA),
                np.zeros((16, 2)),
            )
            return _plan(lattice), ramp
        case "spacing":
            return _plan(lattice), _slices(16, 1.5 * wavelength, 7.5)
        case _:
            raise AssertionError(name)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("grid", "one-dimensional transverse model"),
        ("harmonics", "fundamental only"),
        ("periodic", "open window"),
        ("drift", "refuses breaks"),
        ("periods", "one undulator period"),
        ("slices", "identical slices"),
        ("spacing", "integer multiple of the wavelength"),
    ],
)
def test_out_of_subset_plans_are_refused_before_running(name: str, message: str) -> None:
    plan, slices = _refusal_case(name)
    with pytest.raises(ValueError, match=message):
        puffin_input(plan, slices, seed=None)


def test_pulse_seeds_records_and_sampling_outside_the_subset_are_refused() -> None:
    lattice = _lattice("helical", (1.0,), 40)
    wavelength = lattice.resonant_wavelength(GAMMA)
    slices = _slices(8, wavelength, 7.5)
    plan = _plan(lattice)
    with pytest.raises(ValueError, match="record_periods must divide"):
        puffin_input(plan, slices, seed=None, record_periods=7)
    with pytest.raises(ValueError, match="nodes_per_wavelength >= 9"):
        puffin_input(plan, slices, seed=None, nodes_per_wavelength=8)
    genesis = Genesis4GaussianSeed(1.0, center_position=0.0, rms_length=1e-6, waist=1e-4)
    with pytest.raises(TypeError, match="PuffinGaussianSeed"):
        puffin_input(plan, slices, seed=genesis)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PinnedExecutable"):
        PuffinProvider("puffin")  # ty: ignore[invalid-argument-type]


# ----------------------------------------------------------------- reader


def _reference_case() -> tuple[FELTimeDependentPlan, FELBeamSlices, PuffinGaussianSeed]:
    """The case whose Puffin records are stored under ``tests/data/providers``."""
    lattice = _lattice("helical", (1.0,), 4)
    wavelength = lattice.resonant_wavelength(GAMMA)
    slices = FELBeamSlices(
        np.arange(12) * 2.0 * wavelength,
        np.full(12, 10.0),
        np.full(12, GAMMA),
        np.zeros(12),
        np.full((12, 2), 1e-9),
        np.full((12, 2), 1000.0),
        np.zeros((12, 2)),
    )
    core = FELPlan(lattice, wavelength, loading=FELLoading(2, 8, shot_noise="quiet"))
    plan = FELTimeDependentPlan(core, slippage="spectral", boundary="open")
    seed = PuffinGaussianSeed(
        1.0e3, center_position=14.0 * wavelength, rms_length=3.0 * wavelength
    )
    return plan, slices, seed


def _records(directory: Path) -> tuple[Path, Path]:
    """Decompress the stored genuine Puffin records (``gzip -9 -n`` of the originals)."""
    paths = []
    for index in (0, 1):
        path = directory / f"puffin_integrated_{index}.h5"
        path.write_bytes(gzip.decompress((DATA / f"{path.name}.gz").read_bytes()))
        paths.append(path)
    return paths[0], paths[1]


def test_reader_maps_genuine_records_onto_window_slots(tmp_path: Path) -> None:
    plan, slices, seed = _reference_case()
    wavelength = plan.core.wavelength
    records = _records(tmp_path)
    positions, power = read_puffin_power(
        records, plan, slices, seed=seed, record_periods=4
    )
    np.testing.assert_allclose(positions, [0.0, 4 * PERIOD], rtol=0.0, atol=1e-12)
    # The entrance record is the declared Gaussian pulse averaged over 2λ slots
    # (Puffin's linear mesh interpolation errs by ≲ 1e-3 in the far tails).
    expected = _gaussian_slots(
        12, 2.0 * wavelength, 1.0e3, 14.0 * wavelength, 3.0 * wavelength
    )
    np.testing.assert_allclose(power[0], expected, rtol=2e-3)
    # The exit record against Puffin's own SI power, averaged independently on a
    # fine grid over the window its reference electrons reached: the head sits
    # behind the seed's ±3.75 field-σ extent (field σ = √2 × power rms).
    with h5py.File(records[1], "r") as output:
        info = output["runInfo"].attrs
        dataset = output["powerSI"]
        assert isinstance(dataset, h5py.Dataset)
        nodes = np.asarray(dataset, dtype=np.float64)
        spacing = float(info["sLengthOfElmZ2"])
        cooperation = float(info["Lc"])
        slippage = float(dataset.attrs["zbarInter"])
    mesh = spacing * np.arange(nodes.shape[0])
    extent = 3.75 * math.sqrt(2.0) * 3.0 * wavelength / cooperation
    head = max(0.0, extent - 15.0 * wavelength / cooperation)
    edges = head + slippage + 2.0 * wavelength / cooperation * np.arange(13)
    fine = np.linspace(edges[0], edges[-1], 12 * 400 + 1)
    samples = np.interp(fine, mesh, nodes)
    averages = np.asarray(
        [
            np.trapezoid(
                samples[400 * j : 400 * (j + 1) + 1], fine[400 * j : 400 * (j + 1) + 1]
            )
            for j in range(12)
        ]
    ) / (edges[1] - edges[0])
    # Fine-grid trapezoids of the linear interpolant agree with the exact
    # piecewise-linear integral to ~3e-5 in the faint head slot.
    np.testing.assert_allclose(power[1], averages, rtol=1e-4)
    # Four periods slip the pulse two 2λ slots toward the head.
    assert int(np.argmax(power[1])) == int(np.argmax(power[0])) - 2


def test_reader_refuses_records_of_another_frame_or_schedule(tmp_path: Path) -> None:
    plan, slices, seed = _reference_case()
    records = _records(tmp_path)
    with pytest.raises(ValueError, match="write schedule"):
        read_puffin_power(records[:1], plan, slices, seed=seed, record_periods=4)
    brighter = FELBeamSlices(
        np.asarray(slices.positions),
        np.full(12, 20.0),
        np.asarray(slices.lorentz_factors),
        np.asarray(slices.relative_energy_spreads),
        np.asarray(slices.normalized_emittances),
        np.asarray(slices.beta_functions),
        np.asarray(slices.alpha_functions),
    )
    with pytest.raises(ValueError, match="another Puffin frame"):
        read_puffin_power(records, plan, brighter, seed=seed, record_periods=4)


# ------------------------------------------------------------------- live


def _puffin() -> PuffinProvider:
    executable = os.environ.get("PHYDRAX_PUFFIN")
    version = os.environ.get("PHYDRAX_PUFFIN_VERSION")
    if executable is None or version is None:
        pytest.skip(
            "set PHYDRAX_PUFFIN and PHYDRAX_PUFFIN_VERSION to a pinned Puffin binary"
        )
    return PuffinProvider(
        pin_executable(executable, version=version, license_id="BSD-3-Clause")
    )


def _gain_length(
    positions: np.ndarray, power: np.ndarray, lower: float, upper: float
) -> float:
    inside = (positions >= lower) & (positions <= upper)
    slope = np.polyfit(positions[inside], np.log(power[inside]), 1)[0]
    return float(1.0 / slope)


def test_puffin_oracle_matches_seeded_time_dependent_amplifier(tmp_path: Path) -> None:
    provider = _puffin()
    count, periods = 256, 128
    lattice = _lattice("helical", (1.0,), periods)
    wavelength = lattice.resonant_wavelength(GAMMA)
    # ρ = 0.005: cold 1-D power gain length λ_u/(4π√3ρ) ≈ 9.2 periods.
    rho = 0.005
    plasma = math.sqrt((rho * GAMMA) ** 3) * 4.0 * LIGHT * (2.0 * math.pi / PERIOD)
    current = plasma**2 * PERMITTIVITY * MASS / CHARGE**2 * CHARGE * LIGHT * AREA
    slices = _slices(count, wavelength, current)
    plan = _plan(lattice)
    power, center, rms = 1.0e3, 100.0 * wavelength, 60.0 * wavelength
    # The same Gaussian pulse as a plane-wave X5 envelope with P = ε₀c A|E|²/2.
    zeta = np.arange(count) * wavelength
    amplitude = np.sqrt(2.0 * power / (PERMITTIVITY * LIGHT * AREA)) * np.exp(
        -((zeta - center) ** 2) / (4.0 * rms**2)
    )
    plane = PlaneFieldSpace(
        TensorGridPlan(
            (UniformAxisSpec(2), UniformAxisSpec(2)), axis_names=("x", "y")
        ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]])),
        RigidFrame.identity(3),
        "finite-window",
    )
    time = PulseTimeSpace(
        TensorGridPlan((UniformAxisSpec(count),), axis_names=("t",)).prepare(
            jnp.asarray([[0.0], [(count - 1) * wavelength / LIGHT]])
        ),
        topology="finite-window",
    )
    envelope = PulseEnvelopeField(
        plane,
        time,
        np.broadcast_to(amplitude, (2, 2, count)).astype(np.complex128),
        2.0 * math.pi * LIGHT / wavelength,
        0.0,
    )
    seeded = FELTimeDependentPlan(
        plan.core,
        slippage="commensurate",
        boundary="open",
        pulse_seed=FELPulseSeed(envelope, reference_position=0.0),
    )
    ours = eqx.filter_jit(seeded.solve)(slices, jax.random.key(0))
    theirs = run_puffin(
        provider,
        plan,
        slices,
        tmp_path,
        seed=PuffinGaussianSeed(power, center_position=center, rms_length=rms),
    )
    np.testing.assert_allclose(theirs.positions, ours.positions, rtol=0.0, atol=1e-9)
    mine = np.asarray(ours.power[:, :, 0])
    oracle = np.asarray(theirs.power)
    np.testing.assert_allclose(oracle[0], mine[0], rtol=1e-4)
    # Exponential regime (4.4 to 8.7 gain lengths) at slot 100, ahead of the
    # tail, where Puffin's flat-top edge emission never slips in: averaged and
    # unaveraged growth differ at O(ρ) = 0.5 %; step and mesh errors add ~1 %.
    positions = np.asarray(ours.positions)
    theory = float(ours.scaling.one_dimensional_gain_length[0])
    measured = _gain_length(positions, mine[:, 100], 1.2, 2.4)
    oracle_gain = _gain_length(positions, oracle[:, 100], 1.2, 2.4)
    assert oracle_gain == pytest.approx(measured, rel=0.03)
    assert oracle_gain == pytest.approx(theory, rel=0.05)
    # Saturation: peak power over the head half of the window (the tail half
    # carries Puffin's superradiant edge emission).
    assert np.max(oracle[:, : count // 2]) == pytest.approx(
        np.max(mine[:, : count // 2]), rel=0.05
    )
    assert int(np.argmax(np.max(oracle[:, : count // 2], axis=1))) == pytest.approx(
        int(np.argmax(np.max(mine[:, : count // 2], axis=1))), abs=3
    )
    report = theirs.report
    assert report.status == AdapterStatus.DECLARED_LOSS
    assert report.source_id == theirs.output_sha256
    assert {loss.path for loss in report.losses} >= {"/power", "/power/harmonics"}
    assert (theirs.license_id, theirs.executable_sha256) == (
        "BSD-3-Clause",
        provider.executable.sha256,
    )
