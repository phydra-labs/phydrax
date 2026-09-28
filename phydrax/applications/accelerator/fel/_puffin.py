#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned Puffin oracle for the unaveraged one-dimensional FEL.

Puffin (L. T. Campbell and B. W. J. McNeil, "Puffin: A three dimensional,
unaveraged free electron laser simulation code", Phys. Plasmas 19, 093119,
2012; https://github.com/UKFELs/Puffin; BSD-3-Clause) is used only as a
caller-pinned external executable; no source is copied. Validated live:
Puffin 2.1.0a (commit ``157f473``) built with MPI FFTW and parallel HDF5 from
conda-forge on osx-arm64.

Puffin integrates the unaveraged FEL equations, resolving the radiation
carrier and the electron wiggle, in the scaled frame ``ρ``,
``l_g = λ_u/(4πρ)``, ``l_c = λ/(4πρ)``, ``κ = a_u/(2ργ_r)``,
``z̄ = z/l_g``, ``z̄₂ = (ct − z)/l_c``. :func:`puffin_input` computes that frame
from the plan with the plan's :class:`ElectromagneticScaleContract`: ``ρ`` is
the plan's Pierce parameter, ``a_u`` the peak deflection ``K`` of the first
module, and ``γ_r`` makes Puffin's exact resonance ``λ_u (1 − β_z)/β_z``
(``β_z² = 1 − (1 + a_w²)/γ_r²``) equal the plan wavelength. It writes a
scaled-unit main deck, a flat-top ``simple`` beam, an optional Gaussian seed,
and a lattice file with one ``UN`` module per segment (stepwise ``K`` as the
module tuning ``a_u/a_u₀``). Only the physical lattice length and the peak
current reach Puffin in SI (metres and amperes through
:meth:`ElectromagneticScaleContract.unit_si_map`).

:func:`read_puffin_power` reads Puffin's integrated HDF5 records. The beam
occupies ``z̄₂ ∈ [h̄, h̄ + S Δζ/l_c]`` at the entrance and its reference
electrons advance by ``z̄`` along the undulator, so slot ``j`` of the plan's
``"positive-late"`` window (head first) is
``z̄₂ ∈ [h̄ + z̄ + jΔζ/l_c, h̄ + z̄ + (j + 1)Δζ/l_c]``; the carrier-resolved
power ``|A⊥|² · 2πσ̄_xσ̄_y`` is averaged over it exactly (piecewise-linear
nodes) and converted with ``P = (scaled power) l_g l_c c ε₀
(γ_r mₑc²/(eκl_g))²``. ``h̄`` is the head margin that keeps a declared seed
wholly on Puffin's mesh (Puffin otherwise shifts it silently).

Supported subset (everything else is refused before running): scales
referenced to SI; the ``"one-dimensional"`` transverse model (Puffin's
``qOneD`` with the same slice area ``2πσ_xσ_y``); the fundamental harmonic; an
open window without head padding; undulator modules sharing one period and
polarization with no breaks, thin quadrupoles, phase shifters, or smooth
focusing; no wakes, prebunching, space charge, or plan seeds (a Gaussian pulse
is declared with :class:`PuffinGaussianSeed`); identical, uniformly spaced
slices whose spacing is an integer multiple of the wavelength. Every residual
difference is an :class:`AdapterLoss` on the result report.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, NamedTuple

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...._external_runtime import (
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._validation import finite_real_scalar, positive_finite_float, positive_integer
from ....interchange import AdapterLoss, AdapterReport, AdapterStatus
from ._lattice import undulator_rms_strength_squared
from ._slices import FELBeamSlices
from ._theory import fel_scaling_estimate
from ._time_dependent import FELTimeDependentPlan


_BASENAME = "puffin"
_MAIN = "puffin.in"
_BEAM = "beam.in"
_SEED = "seed.in"
_LATTICE = "puffin.latt"
_SOURCE_FORMAT = "puffin-integrated"
_TARGET_FORMAT = "phydrax-fel-time-dependent-power"
# Puffin's Gaussian seed spans ±gExtEj_G/2 = ±3.75 field σ; a seed reaching past
# the mesh head (z̄₂ < 0) is shifted, so the beam head sits behind that extent.
_SEED_EXTENT = 3.75
# Puffin samples the flat-top beam over total widths sLenE of 6σ̄ transversely
# and 8σ_Γ in energy; only σ̄ enters the one-dimensional area 2πσ̄_xσ̄_y.
_TRANSVERSE_EXTENT = 6.0
_ENERGY_EXTENT = 8.0
# Radiation-mesh margin behind the beam tail after the full slippage, on top of
# Puffin's own unit guard (z̄₂ = 1) in its mesh-length estimate.
_MESH_MARGIN_WAVELENGTHS = 2
_MESH_GUARD = 1.0
# Puffin refuses radiation meshes coarser than λ/8 (chkFSampleLens).
_MINIMUM_NODES = 9
# Puffin evaluates π in single precision (3.1415927…): frame lengths it reports
# differ from the translated frame by 2.8e-8.
_FRAME_TOLERANCE = 1.0e-6


@dataclass(frozen=True, slots=True)
class PuffinProvider:
    """A pinned ``puffin`` binary (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable puffin binary.")


class PuffinGaussianSeed(StrictModule):
    """Gaussian seed pulse ``P(ζ) = P₀ exp(−(ζ − ζ_c)²/(2σ²))`` for Puffin.

    ``center_position`` is in the slice coordinate of the time-dependent plan;
    the pulse is resonant, transversely uniform over the slice area, and
    polarized like the undulator (circular for helical, linear for planar).
    """

    peak_power: float = eqx.field(static=True)
    center_position: float = eqx.field(static=True)
    rms_length: float = eqx.field(static=True)

    def __init__(
        self, peak_power: float, /, *, center_position: float, rms_length: float
    ) -> None:
        self.peak_power = positive_finite_float(peak_power, "peak_power")
        self.center_position = finite_real_scalar(center_position, "center_position")
        self.rms_length = positive_finite_float(rms_length, "rms_length")


class PuffinResult(StrictModule):
    """Puffin's time-resolved power on the plan's window slots.

    ``power[z, j]`` (scale power) is ordered like
    ``FELTimeDependentResult.power`` (head first) at the record positions
    ``positions`` (scale length along the lattice); ``pulse_energy[z]`` is
    ``Σ_j P Δζ / c``. ``output_sha256`` is the SHA-256 of the canonical record
    of every published integrated artifact (path and digest, record order) and
    the report's ``source_id``.
    """

    positions: Array
    power: Array
    pulse_energy: Array
    provider_version: str = eqx.field(static=True)
    executable_sha256: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    output_sha256: str = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)


class _Frame(NamedTuple):
    """Puffin's scaled frame and the translated beam, in scale units."""

    rho: float
    reference_gamma: float
    deflection: float
    period: float
    wavelength: float
    gain_length: float
    cooperation_length: float
    power_scale: float
    spacing: float
    slice_count: int
    beam_length: float
    head_margin: float
    total_slippage: float
    record_steps: int
    total_steps: int
    length_unit: float
    current_unit: float
    undulator_type: str
    modules: tuple[tuple[int, float], ...]


def _number(value: float, /) -> str:
    return repr(float(value))


def _logical(value: bool, /) -> str:
    return ".true." if value else ".false."


def _uniform_value(values: np.ndarray, name: str, /) -> np.ndarray:
    if not np.all(values == values[:1]):
        raise ValueError(f"The Puffin oracle needs identical slices ({name}).")
    return values[0]


def _modules(plan: FELTimeDependentPlan, /) -> tuple[tuple[int, float], ...]:
    lattice = plan.core.lattice
    period = lattice.segments[0].device.period
    modules: list[tuple[int, float]] = []
    for segment, deflection in zip(lattice.segments, lattice.deflections, strict=True):
        if (
            segment.drift_length != 0.0
            or segment.quadrupole_integrated_gradient != 0.0
            or segment.phase_shift != 0.0
            or segment.smooth_focusing_gradient != 0.0
        ):
            raise ValueError(
                "The Puffin oracle refuses breaks, thin quadrupoles, phase shifters, "
                "and smooth focusing."
            )
        if segment.device.period != period:
            raise ValueError("The Puffin oracle needs one undulator period.")
        modules.append((segment.device.period_count, deflection / lattice.deflections[0]))
    return tuple(modules)


def _seed_extent(frame_cooperation: float, seed: PuffinGaussianSeed, /) -> float:
    """Puffin's field σ in z̄₂ (the power rms σ is the field σ over √2)."""
    return math.sqrt(2.0) * seed.rms_length / frame_cooperation


def _frame(
    plan: FELTimeDependentPlan,
    slices: FELBeamSlices,
    seed: PuffinGaussianSeed | None,
    steps_per_period: int,
    record_periods: int,
    /,
) -> _Frame:
    if not isinstance(plan, FELTimeDependentPlan):
        raise TypeError("plan must be an FELTimeDependentPlan.")
    if not isinstance(slices, FELBeamSlices):
        raise TypeError("slices must be FELBeamSlices.")
    if seed is not None and not isinstance(seed, PuffinGaussianSeed):
        raise TypeError("seed must be a PuffinGaussianSeed or None.")
    core = plan.core
    lattice = core.lattice
    scale = lattice.scale
    units = scale.unit_si_map()
    if core.transverse != "one-dimensional":
        raise ValueError("The Puffin oracle needs the one-dimensional transverse model.")
    if core.harmonics != (1,):
        raise ValueError("The Puffin oracle models the fundamental only.")
    if plan.boundary != "open" or plan.head_padding != 0:
        raise ValueError("The Puffin oracle needs an open window without padding.")
    if (
        core.wake is not None
        or core.seed is not None
        or plan.prebunching is not None
        or plan.pulse_seed is not None
        or plan.space_charge is not None
    ):
        raise ValueError(
            "The Puffin oracle refuses wakes, prebunching, space charge, and plan "
            "seeds; declare a PuffinGaussianSeed instead."
        )
    spacing = slices.position_spacing
    if spacing is None:
        raise ValueError("The Puffin oracle needs uniformly spaced slices.")
    sample = spacing / core.wavelength
    if abs(sample - round(sample)) > 1.0e-9 or round(sample) < 1:
        raise ValueError("Slice spacing must be an integer multiple of the wavelength.")
    steps = positive_integer(steps_per_period, "steps_per_period")
    record = positive_integer(record_periods, "record_periods")
    modules = _modules(plan)
    periods = sum(count for count, _ in modules)
    if periods % record != 0:
        raise ValueError("record_periods must divide the lattice's period count.")
    for name, values in (
        ("currents", slices.currents),
        ("lorentz_factors", slices.lorentz_factors),
        ("relative_energy_spreads", slices.relative_energy_spreads),
        ("normalized_emittances", slices.normalized_emittances),
        ("beta_functions", slices.beta_functions),
        ("alpha_functions", slices.alpha_functions),
    ):
        _uniform_value(np.asarray(values), name)
    match lattice.polarization:
        case "helical":
            undulator_type = "helical"
        case "planar":
            undulator_type = "planepole"
        case _:
            assert_never(lattice.polarization)
    light = float(scale.speed_of_light)
    permittivity = float(scale.vacuum_permittivity)
    charge = float(scale.elementary_charge)
    rest_energy = float(scale.electron_mass) * light * light
    period = lattice.segments[0].device.period
    deflection = lattice.deflections[0]
    rms_squared = undulator_rms_strength_squared(deflection, lattice.polarization)
    # Puffin's resonance: λ = λ_u (1 − β_z)/β_z with β_z² = 1 − (1 + a_w²)/γ_r².
    beta_z = 1.0 / (1.0 + core.wavelength / period)
    reference_gamma = math.sqrt((1.0 + rms_squared) / (1.0 - beta_z * beta_z))
    rho = float(
        fel_scaling_estimate(lattice, slices, core.wavelength).pierce_parameter[0]
    )
    gain_length = period / (4.0 * math.pi * rho)
    cooperation_length = core.wavelength / (4.0 * math.pi * rho)
    coupling = deflection / (2.0 * rho * reference_gamma)
    power_scale = (
        gain_length
        * cooperation_length
        * light
        * permittivity
        * (reference_gamma * rest_energy / (charge * coupling * gain_length)) ** 2
    )
    margin = 0.0
    if seed is not None:
        head = float(np.asarray(slices.positions)[0]) - 0.5 * spacing
        center = (seed.center_position - head) / cooperation_length
        margin = max(0.0, _SEED_EXTENT * _seed_extent(cooperation_length, seed) - center)
    return _Frame(
        rho=rho,
        reference_gamma=reference_gamma,
        deflection=deflection,
        period=period,
        wavelength=core.wavelength,
        gain_length=gain_length,
        cooperation_length=cooperation_length,
        power_scale=power_scale,
        spacing=spacing,
        slice_count=slices.slice_count,
        beam_length=slices.slice_count * spacing / cooperation_length,
        head_margin=margin,
        total_slippage=periods * 4.0 * math.pi * rho,
        record_steps=record * steps,
        total_steps=periods * steps,
        length_unit=units["length"][0],
        current_unit=units["current"][0],
        undulator_type=undulator_type,
        modules=modules,
    )


def puffin_input(
    plan: FELTimeDependentPlan,
    slices: FELBeamSlices,
    /,
    *,
    seed: PuffinGaussianSeed | None,
    random_seed: int = 1,
    steps_per_period: int = 30,
    nodes_per_wavelength: int = 17,
    macroparticles_per_wavelength: int = 16,
    energy_macroparticles: int = 15,
    record_periods: int = 1,
) -> dict[str, bytes]:
    """Translate the supported subset into Puffin's main, beam, seed, and lattice files.

    ``steps_per_period`` and ``nodes_per_wavelength`` are Puffin's unaveraged
    integration steps per undulator period and radiation-mesh nodes per
    wavelength; ``macroparticles_per_wavelength`` equispaced macroparticles
    per wavelength sample the flat-top current, with ``energy_macroparticles``
    Gaussian-weighted energies when the slices carry an energy spread.
    Integrated records are written every ``record_periods`` periods.
    """
    frame = _frame(plan, slices, seed, steps_per_period, record_periods)
    nodes = positive_integer(nodes_per_wavelength, "nodes_per_wavelength")
    macroparticles = positive_integer(
        macroparticles_per_wavelength, "macroparticles_per_wavelength"
    )
    energies = positive_integer(energy_macroparticles, "energy_macroparticles")
    random = positive_integer(random_seed, "random_seed")
    if nodes < _MINIMUM_NODES:
        raise ValueError(
            f"Puffin needs nodes_per_wavelength >= {_MINIMUM_NODES} (mesh spacing "
            "at most λ/8)."
        )
    core = plan.core
    gamma = float(np.asarray(slices.lorentz_factors)[0])
    spread = float(np.asarray(slices.relative_energy_spreads)[0])
    emittance = np.asarray(slices.normalized_emittances)[0]
    beta = np.asarray(slices.beta_functions)[0]
    transverse = np.sqrt(emittance * beta / gamma) / math.sqrt(
        frame.gain_length * frame.cooperation_length
    )
    current = float(np.asarray(slices.currents)[0]) * frame.current_unit
    wavelength_bar = 4.0 * math.pi * frame.rho
    # Puffin lengthens (by 10) any mesh not longer than its estimate
    # L̄ + z̄_total (γ_r/γ)² + 1 (calcSamples); the declared mesh exceeds it.
    mesh = (
        frame.head_margin
        + frame.beam_length
        + frame.total_slippage * max(1.0, (frame.reference_gamma / gamma) ** 2)
        + _MESH_GUARD
        + _MESH_MARGIN_WAVELENGTHS * wavelength_bar
    )
    cold = spread == 0.0
    energy_sigma = spread * gamma / frame.reference_gamma
    main = [
        "&MDATA",
        " qOneD = .true.",
        " qFieldEvolve = .true.",
        " qElectronsEvolve = .true.",
        " qElectronFieldCoupling = .true.",
        " qFocussing = .false.",
        " qDiffraction = .false.",
        f" q_noise = {_logical(core.loading.shot_noise == 'fawley')}",
        " qDump = .false.",
        " qResume = .false.",
        " qscaled = .true.",
        " qUndEnds = .false.",
        " qDumpEnd = .false.",
        f" beam_file = '{_BEAM}'",
        f" seed_file = '{_SEED if seed is not None else ''}'",
        f" lattFile = '{_LATTICE}'",
        " iNumNodesX = 1",
        " iNumNodesY = 1",
        f" nodesPerLambdar = {nodes}",
        f" sFModelLengthZ2 = {_number(mesh)}",
        f" srho = {_number(frame.rho)}",
        f" saw = {_number(frame.deflection)}",
        f" sgamma_r = {_number(frame.reference_gamma)}",
        f" lambda_w = {_number(frame.period * frame.length_unit)}",
        f" zundType = '{frame.undulator_type}'",
        f" stepsPerPeriod = {steps_per_period}",
        f" nPeriods = {frame.total_steps // steps_per_period}",
        f" iWriteNthSteps = {frame.total_steps + 1}",
        f" iWriteIntNthSteps = {frame.record_steps}",
        f" iRandSeed = {random}",
        " ioutInfo = 0",
        "/",
        "",
    ]
    beam = [
        "&NBLIST",
        " nbeams = 1",
        " dtype = 'simple'",
        "/",
        "&BLIST",
        " sSigmaE = "
        + ", ".join(
            _number(value)
            for value in (
                transverse[0],
                transverse[1],
                1.0e8,
                1.0,
                1.0,
                energy_sigma if not cold else 1.0,
            )
        ),
        " sLenE = "
        + ", ".join(
            _number(value)
            for value in (
                _TRANSVERSE_EXTENT * transverse[0],
                _TRANSVERSE_EXTENT * transverse[1],
                frame.beam_length,
                _TRANSVERSE_EXTENT,
                _TRANSVERSE_EXTENT,
                _ENERGY_EXTENT * energy_sigma if not cold else 1.0,
            )
        ),
        f" iMPsZ2PerWave = {macroparticles}",
        f" qOneDCold = {_logical(cold)}",
        f" inmps1DGam = {1 if cold else energies}",
        f" Ipk = {_number(current)}",
        f" bcenter = {_number(frame.head_margin + 0.5 * frame.beam_length)}",
        f" gammaf = {_number(gamma / frame.reference_gamma)}",
        " chirp = 0.0",
        " mag = 0.0",
        " qRndEj_G = .false.",
        " qMatched_A = .false.",
        " qFixCharge = .false.",
        " TrLdMeth = 0",
        "/",
        "",
    ]
    lattice = [
        f"UN '{frame.undulator_type}' {count} {_number(tuning)} 0.0 {steps_per_period} "
        "1.0 1.0 0.0 0.0"
        for count, tuning in frame.modules
    ]
    files = {
        _MAIN: "\n".join(main).encode(),
        _BEAM: "\n".join(beam).encode(),
        _LATTICE: ("\n".join(lattice) + "\n").encode(),
    }
    if seed is not None:
        head = float(np.asarray(slices.positions)[0]) - 0.5 * frame.spacing
        area = 2.0 * math.pi * float(transverse[0] * transverse[1])
        # Scaled power is |A⊥|² · 2πσ̄_xσ̄_y; Puffin's sA0 is the cycle-mean
        # |A|² of each polarization (peak field magnitude √(2 sA0)).
        intensity = seed.peak_power / (area * frame.power_scale)
        match plan.core.lattice.polarization:
            case "helical":
                components = (0.5 * intensity, 0.5 * intensity)
            case "planar":
                components = (intensity, 0.0)
            case _:
                assert_never(plan.core.lattice.polarization)
        center = frame.head_margin + (seed.center_position - head) / (
            frame.cooperation_length
        )
        files[_SEED] = "\n".join(
            [
                "&NSLIST",
                " nseeds = 1",
                " dtype = 'simple'",
                "/",
                "&SLIST",
                " freqf = 1.0",
                " ph_sh = 0.0",
                f" sA0_X = {_number(components[0])}",
                f" sA0_Y = {_number(components[1])}",
                " sSigmaF = 1.0, 1.0, "
                + _number(_seed_extent(frame.cooperation_length, seed)),
                " qFlatTop = .false.",
                f" meanZ2 = {_number(center)}",
                " qRndFj_G = .false.",
                " sSigFj_G = 1.0",
                " qMatchS_G = .false.",
                "/",
                "",
            ]
        ).encode()
    return files


def _record_paths(frame: _Frame, /) -> tuple[str, ...]:
    count = frame.total_steps // frame.record_steps + 1
    return tuple(f"{_BASENAME}_integrated_{index}.h5" for index in range(count))


def _slot_averages(
    power: np.ndarray, spacing: float, lower: np.ndarray, width: float, /
) -> np.ndarray:
    """Exact means of the piecewise-linear node power over ``[lower, lower + width]``."""
    cumulative = np.concatenate(
        (
            np.zeros((1,), dtype=np.float64),
            np.cumsum(0.5 * (power[1:] + power[:-1]) * spacing, dtype=np.float64),
        )
    )

    def integral(position: np.ndarray) -> np.ndarray:
        index = np.clip(
            np.floor(position / spacing).astype(np.int64), 0, power.shape[0] - 2
        )
        offset = position - index * spacing
        slope = (power[index + 1] - power[index]) / spacing
        return cumulative[index] + power[index] * offset + 0.5 * slope * offset**2

    return (integral(lower + width) - integral(lower)) / width


def read_puffin_power(
    paths: Sequence[str | Path],
    plan: FELTimeDependentPlan,
    slices: FELBeamSlices,
    /,
    *,
    seed: PuffinGaussianSeed | None,
    steps_per_period: int = 30,
    record_periods: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Read Puffin integrated records into ``(positions, power[z, j])``.

    ``paths`` are Puffin's ``*_integrated_<k>.h5`` records in write order for
    the deck :func:`puffin_input` produced from the same arguments; positions
    are in the plan's length unit and power in its power unit, on the window
    slots head first. Each record's frame (ρ, γ_r, a_u, λ_u, l_c) must match
    the translated frame.
    """
    frame = _frame(plan, slices, seed, steps_per_period, record_periods)
    files = tuple(Path(path) for path in paths)
    if len(files) != frame.total_steps // frame.record_steps + 1:
        raise ValueError("Puffin records do not match the translated write schedule.")
    expected = {
        "rho": frame.rho,
        "gamma_r": frame.reference_gamma,
        "aw": frame.deflection,
        "lambda_w": frame.period * frame.length_unit,
        "Lc": frame.cooperation_length * frame.length_unit,
    }
    positions = np.empty((len(files),), dtype=np.float64)
    power = np.empty((len(files), frame.slice_count), dtype=np.float64)
    for index, file in enumerate(files):
        with h5py.File(file, "r") as output:
            info = output["runInfo"].attrs
            dataset = output["power"]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"{file} has no Puffin power dataset.")
            for name, value in expected.items():
                if not math.isclose(float(info[name]), value, rel_tol=_FRAME_TOLERANCE):
                    raise ValueError(
                        f"{file} was written in another Puffin frame ({name})."
                    )
            if int(info["nX"]) != 1 or int(info["nY"]) != 1:
                raise ValueError(f"{file} is not a one-dimensional Puffin record.")
            nodes = np.asarray(dataset, dtype=np.float64)
            spacing = float(info["sLengthOfElmZ2"])
            slippage = float(dataset.attrs["zbarInter"])
            position = float(dataset.attrs["zTotal"])
            cooperation = float(info["Lc"]) / frame.length_unit
        width = frame.spacing / cooperation
        lower = (
            frame.head_margin
            + slippage
            + width * np.arange(frame.slice_count, dtype=np.float64)
        )
        if nodes.ndim != 1 or lower[-1] + width > spacing * (nodes.shape[0] - 1):
            raise ValueError(f"{file} does not cover the translated window.")
        positions[index] = position / frame.length_unit
        power[index] = frame.power_scale * _slot_averages(nodes, spacing, lower, width)
    return positions, power


def _losses(plan: FELTimeDependentPlan, /) -> tuple[AdapterLoss, ...]:
    losses = [
        AdapterLoss(
            "/power",
            "import",
            "transformed",
            "Puffin resolves the radiation carrier and electron wiggle; its "
            "carrier-resolved power is averaged over each slot of the averaged window.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "/power/harmonics",
            "import",
            "transformed",
            "Puffin has no harmonic restriction: the slot power includes every "
            "harmonic and coherent spontaneous emission (notably from the flat-top "
            "current edges) resolved on its mesh, which the fundamental averaged "
            "model omits.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "/slices/transverse",
            "export",
            "dropped",
            "One-dimensional Puffin has no transverse phase space: emittance and "
            "Twiss parameters enter only through the area 2πσ_xσ_y, so the "
            "betatron contribution to the resonance is dropped.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "/core/lattice/step_length",
            "export",
            "transformed",
            "Puffin integrates steps_per_period unaveraged steps per period and "
            "samples the beam with its own macroparticles per wavelength instead of "
            "the plan's steps and beamlet loading.",
            changes_interpretation=False,
        ),
    ]
    if plan.core.loading.shot_noise == "fawley":
        losses.append(
            AdapterLoss(
                "/core/loading/shot_noise",
                "export",
                "transformed",
                "Puffin adds its own shot noise (McNeil, Poole & Robb 2003) with the "
                "same Poisson statistics, not the plan's Fawley realization.",
                changes_interpretation=False,
            )
        )
    return tuple(losses)


def run_puffin(
    provider: PuffinProvider,
    plan: FELTimeDependentPlan,
    slices: FELBeamSlices,
    destination: str | Path,
    /,
    *,
    seed: PuffinGaussianSeed | None = None,
    random_seed: int = 1,
    steps_per_period: int = 30,
    nodes_per_wavelength: int = 17,
    macroparticles_per_wavelength: int = 16,
    energy_macroparticles: int = 15,
    record_periods: int = 1,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 30,
) -> PuffinResult:
    """Run pinned Puffin (one MPI rank, one thread) and read its power history."""
    if not isinstance(provider, PuffinProvider):
        raise TypeError("provider must be a PuffinProvider.")
    inputs = puffin_input(
        plan,
        slices,
        seed=seed,
        random_seed=random_seed,
        steps_per_period=steps_per_period,
        nodes_per_wavelength=nodes_per_wavelength,
        macroparticles_per_wavelength=macroparticles_per_wavelength,
        energy_macroparticles=energy_macroparticles,
        record_periods=record_periods,
    )
    frame = _frame(plan, slices, seed, steps_per_period, record_periods)
    records = _record_paths(frame)
    artifacts = PinnedFileOutputs(
        str(destination),
        tuple(PinnedFileRequest(path, maximum_output_bytes) for path in records),
        maximum_output_bytes,
    )
    executable = provider.executable
    run = run_pinned_command(
        executable,
        (_MAIN,),
        inputs=inputs,
        timeout=timeout,
        environment={"OMP_NUM_THREADS": "1"},
        artifacts=artifacts,
    ).require_success()
    published = tuple(run.file_artifact(path) for path in records)
    positions, power = read_puffin_power(
        tuple(artifact.location for artifact in published),
        plan,
        slices,
        seed=seed,
        steps_per_period=steps_per_period,
        record_periods=record_periods,
    )
    light = float(plan.core.lattice.scale.speed_of_light)
    energy = np.sum(power, axis=1) * frame.spacing / light
    output = canonical_fingerprint(
        {
            "kind": "puffin-integrated-records",
            "records": [[artifact.path, artifact.sha256] for artifact in published],
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        _SOURCE_FORMAT,
        _TARGET_FORMAT,
        source_id=output,
        target_id=canonical_fingerprint(
            {
                "kind": "puffin-fel-power",
                "plan": plan.plan_id,
                "slices": slices.slices_id,
                "arrays": array_tree_fingerprint((positions, power)),
            }
        ),
        coordinate_mapping=(
            "ρ = plan Pierce parameter; l_g = λ_u/(4πρ); l_c = λ/(4πρ); "
            "κ = K/(2ργ_r) with γ_r from Puffin's exact resonance at λ",
            "slot j -> z̄₂ ∈ h̄ + z̄ + [j, j + 1] Δζ/l_c (head first)",
            "scaled power -> P = power · l_g l_c c ε₀ (γ_r mₑc²/(eκl_g))²",
            "zTotal metres / length unitSI -> positions",
        ),
        preserved_fields=("positions", "power", "pulse_energy"),
        assumptions=(
            "helical seeds are circular with Puffin's co-rotating handedness",
            "the flat-top beam occupies z̄₂ ∈ [h̄, h̄ + SΔζ/l_c] at the entrance",
        ),
        losses=_losses(plan),
    )
    return PuffinResult(
        positions=jnp.asarray(positions),
        power=jnp.asarray(power),
        pulse_energy=jnp.asarray(energy),
        provider_version=executable.version,
        executable_sha256=executable.sha256,
        license_id=executable.license_id,
        output_sha256=output,
        report=report,
    )


__all__ = [
    "PuffinGaussianSeed",
    "PuffinProvider",
    "PuffinResult",
    "puffin_input",
    "read_puffin_power",
    "run_puffin",
]
