#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Full-wave free-electron laser in a Lorentz-boosted frame.

`FELFullWavePlan` composes the self-consistent electromagnetic PIC of the
boosted-frame owner (`phydrax.solver.BoostedFramePlan`) into an FEL: no period
averaging, no slowly varying envelope, no wiggle-averaged coupling. Every
ingredient is an existing owner:

- Frame. A pure boost ``β_b ẑ`` with declared ``γ_b``. At the undulator
  resonance ``γ_b = γ_z = γ/√(1 + a_w²)`` the beam is at rest on average, the
  undulator period contracts to ``λ_u′ = λ_u/γ_b`` and the radiation wavelength
  stretches to ``λ′ = λ γ_b (1 + β_b)``: both are of one scale, so the cost no
  longer grows with ``γ²`` (Vay 2007; Fawley and Vay 2009). The frame evidence
  reports ``γ_b`` against ``γ_z`` and the mean boosted beam velocity.
- Field. A standard (non-Galilean) staggered PSATD grid under the
  boosted-frame run's mandatory numerical-Cherenkov guard, periodic across
  the thin transverse cross section and open along the boost axis: PSATD PML
  layers at both ends absorb the radiation that leaves the interior, whose lab
  energy enters the ledger. The staggered grid keeps the deposited current on
  the edges it was deposited on, so Huygens surfaces can be current-free.
- Undulator. The lattice's X1a `InsertionDeviceField` devices, placed at their
  own centers, are gathered through `BoostedExternalField`: prescribed,
  never deposited, never radiating on the grid.
- Beam. Lab-frame macroparticles (`FELFullWaveBeam`) are injected on the first
  boosted slice on which every one of them is still upstream of the undulator
  support; they reach it ballistically in vacuum (`boost_particles`).
  Neutral beams are electron–positron image pairs: the positron of a pair sees
  ``q v_x = −q² A_x/(γm)``, the same undulator and radiation coupling as its
  electron, so the pair beam of total current ``I`` is the full-wave
  counterpart of a space-charge-free beam of current ``I`` (the periodic
  spectral solver requires a neutral box).
- Seed. `FELFullWaveSeed` is a lab-frame continuous plane wave, polarized in
  the deflection plane of a planar lattice, with a flat top covering the
  slippage interval of every electron. It is launched by a one-way B5
  `SampledPlaneCurrentAntennaPlan` at rest on the lab undulator entrance plane,
  boosted onto the grid (`BoostedFramePlan.boost_antenna`: a sheet moving at
  ``−β_b c``). The run starts before the antenna emits, so no field overlaps
  the beam at initialization.
- Radiation. A1 trajectory radiation of recorded lab tracks
  (`FELFullWaveTracks`: `PICTrackRecorder` lanes carried to the lab with
  per-lane lab times), or the boosted Huygens far field (`FELFullWaveHuygens`)
  relabeled into the lab. The Huygens route is admitted only with ``J = 0`` on
  the box for the whole acquisition window, in vacuum, on the standard grid,
  without a seed antenna (the sheet carries current at every node along the
  axis), and when no transverse periodic image of the radiation reaches the
  box before the window closes; anything else is refused.

Energy ledger. The particle and field four-momentum on a boosted slice, and
the four-momentum the PML absorbs and the antenna injects, transform to the
lab: ``U = γ_b (U′ + β_b c P′_z)``. The undulator is magnetostatic in the lab
and does no work, so the lab beam, grid-field, escaped, and injected energies
balance up to the discretization defect.

Gain. The forward, transversely uniform wave ``(E_x + cB_y)/2``,
``(E_y − cB_x)/2`` is band-limited about ``k′`` and carried to the lab
(``E = γ_b(1 + β_b)E′``); its power gain profile is reported on the co-moving
lab coordinate ``ξ = z − ct`` against the seed amplitude measured on the
grid in the seed's tail margin, which no electron crosses inside the
undulator. For `FELFullWavePlan.flat_top_beam` beams the steady window is the
set of field elements that slipped across flat-top beam only, which is the
time-independent (X5a) gain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._lorentz import boost_matrix, LorentzFrame, LorentzSpectralTransform
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import finite_real_scalar, positive_finite_float, positive_integer
from ....discretization import (
    ChargedParticlePlan,
    ParticlePopulationPlan,
    ParticleSetPlan,
    StructuredCochainBridge,
    StructuredCochainResourcePolicy,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from ....discretization.pic import (
    ChargeConservingCurrentPlan,
    PIC_CODE_RELATIVITY,
    PICChargeModelPlan,
    PICParticleCochainTransferPlan,
    PICSpeciesPlan,
    PICTrackRecorder,
)
from ....electromagnetics import (
    PreparedTrajectoryRadiation,
    TrajectoryRadiationPlan,
    TrajectoryRadiationResult,
)
from ....geometry import Box
from ....solver import (
    BoostedFrameEvidence,
    BoostedFramePlan,
    BoostedFrameState,
    ElectromagneticPICPlan,
    PreparedBoostedFrame,
)
from ....solver.maxwell import (
    HomogeneousMaxwellExterior,
    MaxwellFarFieldPlan,
    MaxwellSpectralAcquisition,
    SampledPlaneCurrentAntennaPlan,
)
from ....solver.maxwell.spectral import (
    PreparedSpectralMaxwell,
    SpectralHuygensBoxPlan,
    SpectralMaxwellPlan,
    SpectralPMLPlan,
)
from ....typing import (
    as_host_array,
    checked,
    ConvertibleToArray,
    Dim,
    Float64,
    HostFloat64,
    Identifier,
    parse,
    Scope,
)
from ._lattice import FELUndulatorLattice, undulator_rms_strength_squared


FELFullWaveBeamSpecies: TypeAlias = Literal["electron", "electron-positron"]

# The X1a window ½[tanh((u + a)/w) − tanh((u − a)/w)] is below 3.4e-4 of its
# flat top four ramp widths beyond the flat top: the undulator support.
_TERMINATION_WIDTHS = 4.0
# Absolute mean-charge limit of the periodic spectral Gauss solve.
_NEUTRALITY_LIMIT = 1.0e-10
# Spline shape order of the charge-conserving deposit and gather.
_SHAPE_ORDER = 1
# Guard cells between the beam envelope and a Huygens surface beyond the
# declared margin: the order-one spline and its Esirkepov path support.
_SHAPE_GUARD = 2
# Interior cells between anything the run must hold and a PML layer: the
# Huygens H stencil and the band-limited antenna sheet reach two cells.
_LAYER_GUARD = 4
# Relative Gaussian width of the gain-envelope band about the boosted radiation
# wavenumber: its spatial kernel is Gaussian, so the envelope of a smooth seed
# ramp is exact to ~2e-5 one wavelength beyond the ramp (a sharp band rings).
_BAND_WIDTH = 0.3
# Absolute floors of the PIC continuity and constraint checks (the PIC defaults).
_CONTINUITY_FLOOR = 1.0e-9
_CONSTRAINT_FLOOR = 1.0e-8


class _MacroparticleDim(Dim, minimum=1):
    """Beam macroparticles."""


class FELFullWaveStatus(IntFlag):
    """Fail-closed status bits of a full-wave FEL run.

    ``STEP_REJECTED``: a PIC step was rejected (the run stalled, radiation
    extraction is withheld); ``NCI_REJECTED``: among them, steps refused by the
    numerical-Cherenkov guard; ``UNDULATOR_UNFINISHED``: a particle is not
    downstream of the lattice support at the end; ``LEDGER_DEFECT``: the lab
    energy ledger exceeds ``ledger_tolerance``; ``RADIATION_UNRESOLVED``: the
    trajectory-radiation evidence carries a nonzero status.
    """

    SUCCESS = 0
    NONFINITE = 1
    STEP_REJECTED = 2
    NCI_REJECTED = 4
    UNDULATOR_UNFINISHED = 8
    LEDGER_DEFECT = 16
    RADIATION_UNRESOLVED = 32


class FELFullWaveBeam(StrictModule):
    """Lab-frame electron macroparticles injected upstream of the undulator.

    ``positions``/``velocities`` are lab positions and velocities at the common
    lab time ``lab_time`` (scale units), ``electrons`` the physical electrons
    each macroparticle carries. Every particle must move downstream and start
    upstream of the lattice support; it reaches the undulator ballistically in
    vacuum. ``species="electron-positron"`` pairs every electron with a
    co-moving positron (a neutral beam). ``flat_top`` optionally declares the
    lab interval ``(tail, head)`` at ``lab_time`` over which the beam is uniform
    and shares one velocity: it defines the steady gain window.
    """

    __strict_contract__ = True

    positions: Float64[_MacroparticleDim, Literal[3]]
    velocities: Float64[_MacroparticleDim, Literal[3]]
    electrons: Float64[_MacroparticleDim]
    lab_time: float = eqx.field(static=True)
    species: FELFullWaveBeamSpecies = eqx.field(static=True)
    flat_top: tuple[float, float] | None = eqx.field(static=True)
    beam_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        positions: ConvertibleToArray,
        velocities: ConvertibleToArray,
        electrons: ConvertibleToArray,
        /,
        *,
        lab_time: float = 0.0,
        species: FELFullWaveBeamSpecies = "electron",
        flat_top: tuple[float, float] | None = None,
    ) -> None:
        scope = Scope()
        position = as_host_array(
            positions,
            HostFloat64[_MacroparticleDim, Literal[3]],
            "positions",
            scope=scope,
        )
        velocity = as_host_array(
            velocities,
            HostFloat64[_MacroparticleDim, Literal[3]],
            "velocities",
            scope=scope,
        )
        count = as_host_array(
            electrons, HostFloat64[_MacroparticleDim], "electrons", scope=scope
        )
        if not all(np.all(np.isfinite(value)) for value in (position, velocity, count)):
            raise ValueError("Beam positions, velocities, and electrons must be finite.")
        if np.any(count <= 0.0):
            raise ValueError("Every macroparticle must carry a positive electron count.")
        if np.any(velocity[:, 2] <= 0.0):
            raise ValueError("Every beam particle must move downstream (v_z > 0).")
        time = finite_real_scalar(lab_time, "lab_time")
        kind = parse(species, FELFullWaveBeamSpecies, "species")
        window: tuple[float, float] | None = None
        if flat_top is not None:
            tail = finite_real_scalar(flat_top[0], "flat_top[0]")
            head = finite_real_scalar(flat_top[1], "flat_top[1]")
            if not tail < head:
                raise ValueError("flat_top must be an increasing (tail, head) interval.")
            window = (tail, head)
        self.positions = jnp.asarray(position)
        self.velocities = jnp.asarray(velocity)
        self.electrons = jnp.asarray(count)
        self.lab_time = time
        self.species = kind
        self.flat_top = window
        self.beam_id = canonical_fingerprint(
            {
                "kind": "fel-full-wave-beam",
                "arrays": array_tree_fingerprint((position, velocity, count)),
                "lab_time": time,
                "species": kind,
                "flat_top": None if window is None else list(window),
            }
        )

    @property
    def macroparticle_count(self) -> int:
        return self.positions.shape[0]


class FELFullWaveSeed(StrictModule, NonTrainableState):
    """Continuous lab-frame seed at the plan wavelength.

    ``amplitude`` is the peak ``E_x`` (scale field units) of the plane wave
    ``E_x = c B_y = E₀ g(ξ) cos(kξ + phase)``, ``ξ = z − ct``. The flat top of
    ``g`` covers every electron's slippage interval plus
    ``margin_wavelengths`` on each side and ramps down over
    ``ramp_wavelengths`` (the ``C^∞`` step ``f(u)/(f(u) + f(1 − u))``,
    ``f(u) = e^{−1/u}``, whose envelope rings neither in the gain band nor
    in the reference). The tail margin, which no electron
    crosses inside the undulator, is the on-grid amplitude reference of the
    gain over its second half, so it spans at least two wavelengths. The wave is launched by a
    one-way sheet antenna at rest on the lab undulator entrance plane. Planar
    lattices only.
    """

    amplitude: float = eqx.field(static=True)
    phase: float = eqx.field(static=True)
    ramp_wavelengths: float = eqx.field(static=True)
    margin_wavelengths: float = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitude: float,
        /,
        *,
        phase: float = 0.0,
        ramp_wavelengths: float = 4.0,
        margin_wavelengths: float = 2.0,
    ) -> None:
        self.amplitude = positive_finite_float(amplitude, "amplitude")
        self.phase = finite_real_scalar(phase, "phase")
        self.ramp_wavelengths = positive_finite_float(
            ramp_wavelengths, "ramp_wavelengths"
        )
        margin = finite_real_scalar(margin_wavelengths, "margin_wavelengths")
        if margin < 2.0:
            raise ValueError(
                "margin_wavelengths must be at least two: the second half of the "
                "tail margin is the seed amplitude reference."
            )
        self.margin_wavelengths = margin
        self.seed_id = canonical_fingerprint(
            {
                "kind": "fel-full-wave-seed",
                "amplitude": self.amplitude,
                "phase": self.phase,
                "ramp_wavelengths": self.ramp_wavelengths,
                "margin_wavelengths": margin,
            }
        )


class _SeedSchedule(NamedTuple):
    """Host timing of the seed antenna.

    ``flat``/``ramp`` shape ``g(ξ)``; ``reference`` is the tail-margin ``ξ``
    interval; ``plane`` the lab antenna plane; ``lab_window`` the lab emission
    times; ``boosted_window`` and ``path`` the boosted active times and the
    boosted sheet positions over them.
    """

    flat: tuple[float, float]
    ramp: float
    reference: tuple[float, float]
    plane: float
    lab_window: tuple[float, float]
    boosted_window: tuple[float, float]
    path: tuple[float, float]


class FELFullWaveTracks(StrictModule, NonTrainableState):
    """A1 trajectory radiation of recorded lab-frame tracks.

    ``lanes`` are beam macroparticle indices; each is recorded in every beam
    species (both members of an electron–positron pair). ``radiation`` is a
    lab-frame `TrajectoryRadiationPlan` in the lattice's scale.
    """

    radiation: PreparedTrajectoryRadiation
    lanes: tuple[int, ...] = eqx.field(static=True)
    tracks_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, radiation: TrajectoryRadiationPlan, lanes: Sequence[int], /
    ) -> None:
        indices = tuple(int(value) for value in lanes)
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("lanes must be distinct nonnegative macroparticle indices.")
        self.radiation = radiation.prepare()
        self.lanes = indices
        self.tracks_id = canonical_fingerprint(
            {
                "kind": "fel-full-wave-tracks",
                "radiation": radiation.plan_id,
                "lanes": list(indices),
            }
        )


class FELFullWaveHuygens(StrictModule, NonTrainableState):
    """Boosted Huygens far field relabeled into the lab.

    ``angular_frequencies`` and unit ``directions`` are boosted-frame samples;
    ``reference_axis`` fixes the far-field polarization basis. The closed box
    encloses the beam's boosted envelope over the run plus ``margin_cells``
    (and the spline support) per axis; the acquisition closes once the last
    radiation of the beam has crossed it.
    """

    angular_frequencies: tuple[float, ...] = eqx.field(static=True)
    directions: tuple[tuple[float, float, float], ...] = eqx.field(static=True)
    reference_axis: tuple[float, float, float] = eqx.field(static=True)
    margin_cells: tuple[int, int, int] = eqx.field(static=True)
    huygens_id: str = eqx.field(static=True)

    def __init__(
        self,
        angular_frequencies: ArrayLike,
        directions: ArrayLike,
        /,
        *,
        reference_axis: Sequence[float] = (1.0, 0.0, 0.0),
        margin_cells: Sequence[int] = (1, 1, 1),
    ) -> None:
        frequencies = np.asarray(angular_frequencies, dtype=np.float64).reshape((-1,))
        if (
            frequencies.size == 0
            or not np.all(np.isfinite(frequencies))
            or np.any(frequencies <= 0.0)
        ):
            raise ValueError("angular_frequencies must be finite and positive.")
        direction = np.asarray(directions, dtype=np.float64)
        if direction.ndim != 2 or direction.shape[1] != 3 or direction.shape[0] == 0:
            raise ValueError("directions must have shape (D, 3).")
        if not np.allclose(np.linalg.norm(direction, axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("directions must be unit vectors.")
        axis = np.asarray(reference_axis, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("reference_axis must be a finite 3-vector.")
        margins = tuple(int(value) for value in margin_cells)
        if len(margins) != 3 or min(margins) < 0:
            raise ValueError("margin_cells must be three nonnegative integers.")
        self.angular_frequencies = tuple(float(value) for value in frequencies)
        self.directions = tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in direction
        )
        self.reference_axis = (float(axis[0]), float(axis[1]), float(axis[2]))
        self.margin_cells = (margins[0], margins[1], margins[2])
        self.huygens_id = canonical_fingerprint(
            {
                "kind": "fel-full-wave-huygens",
                "angular_frequencies": [
                    value.hex() for value in self.angular_frequencies
                ],
                "directions": [[value.hex() for value in row] for row in self.directions],
                "reference_axis": list(self.reference_axis),
                "margin_cells": list(self.margin_cells),
            }
        )


class FELFullWaveFrameEvidence(StrictModule):
    """Boosted-frame choice and resolution of one prepared run (host values).

    ``resonant_lorentz_factor`` is ``γ_z = γ̄/√(1 + a_w²)`` of the mean beam
    energy; ``boosted_beam_velocity`` the mean boosted longitudinal velocity
    ``(β̄_z − β_b)/(1 − β̄_z β_b)`` inside the undulator (zero at resonance).
    ``bunching_resolution`` is the grid Nyquist wavenumber over the boosted
    ponderomotive wavenumber ``k′ + k_u′`` (must exceed one);
    ``deposit_form_factor`` the on-axis spline energy factor
    ``sinc^{2(p+1)}(k′Δz/2)`` of the order-``p`` deposit at ``k′`` averaged
    over sub-cell positions (a particle at rest in the boosted frame radiates
    through the factor of its own sub-cell position). ``grid_origin`` and
    ``grid_spacing`` locate the staggered grid's nodes; ``absorber_cells`` is
    the PML thickness at each end of the boost axis.
    """

    boost_lorentz_factor: float = eqx.field(static=True)
    resonant_lorentz_factor: float = eqx.field(static=True)
    boosted_beam_velocity: float = eqx.field(static=True)
    boosted_wavelength: float = eqx.field(static=True)
    boosted_undulator_period: float = eqx.field(static=True)
    cells_per_wavelength: float = eqx.field(static=True)
    cells_per_undulator_period: float = eqx.field(static=True)
    steps_per_undulator_period: float = eqx.field(static=True)
    bunching_resolution: float = eqx.field(static=True)
    deposit_form_factor: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    stop_time: float = eqx.field(static=True)
    grid_origin: tuple[float, float, float] = eqx.field(static=True)
    grid_spacing: tuple[float, float, float] = eqx.field(static=True)
    absorber_cells: int = eqx.field(static=True)


class FELFullWaveLedger(StrictModule):
    """Lab-frame energy balance (scale energy units).

    ``beam_energy_change`` is the lab kinetic-energy change of every beam
    species, ``field_energy_change`` the lab energy change of the grid field,
    ``escaped_energy`` the lab energy the PML layers absorbed (radiation that
    left the interior), and ``injected_energy`` the lab energy the seed
    antenna delivered. ``defect = beam + field + escaped − injected``;
    ``relative_defect`` divides it by the lab energy the run holds (initial
    beam kinetic energy plus injected energy), the conventional PIC energy
    conservation measure: the beam's lab kinetic energy carries the boosted
    push's discretization error (second order in the step and grid spacing),
    which the seeded exchange must exceed for the ledger to resolve it.
    """

    beam_energy_change: Array
    field_energy_change: Array
    escaped_energy: Array
    injected_energy: Array
    defect: Array
    relative_defect: Array


class FELFullWaveEvidence(StrictModule):
    """Status, frame, guard, and step evidence of a run."""

    status: Array
    frame: FELFullWaveFrameEvidence
    boosted: BoostedFrameEvidence
    accepted_steps: Array
    exited: Array
    finite: Array

    @property
    def successful(self) -> Array:
        return self.status == int(FELFullWaveStatus.SUCCESS)


class FELFullWaveResult(StrictModule):
    """Full-wave FEL solution.

    ``initial_lorentz_factors``/``final_lorentz_factors`` are lab ``γ`` per
    beam species and macroparticle. ``forward_amplitude[z]`` is the lab
    amplitude envelope of the forward, transversely uniform wave about the
    radiation wavenumber on the grid nodes, located at the co-moving lab
    coordinates ``forward_coordinates = z − ct``; ``forward_energy`` holds the
    lab energy of the forward transversely uniform wave at the start and end.
    ``steady_amplitude`` is the mean lab amplitude over the steady window
    (``None`` without window). ``seed_amplitude`` is the lab amplitude of the
    seed measured on the grid over its tail margin; ``gain_profile`` is
    ``|E|²/|E_seed|² − 1`` of a seeded run and ``steady_gain`` its mean over the
    steady window (``None`` without seed or window).
    ``trajectory_spectrum`` and ``huygens_spectrum`` are the lab radiation of
    the declared extraction routes (``None`` when absent or when a rejected
    step left the run incomplete).
    """

    initial_lorentz_factors: Array
    final_lorentz_factors: Array
    forward_amplitude: Array
    forward_coordinates: Array
    forward_energy: Array
    steady_amplitude: Array | None
    seed_amplitude: Array | None
    gain_profile: Array | None
    steady_gain: Array | None
    trajectory_spectrum: TrajectoryRadiationResult | None
    huygens_spectrum: LorentzSpectralTransform | None
    ledger: FELFullWaveLedger
    evidence: FELFullWaveEvidence
    final_state: BoostedFrameState
    plan_id: str = eqx.field(static=True)


class _Kinematics(NamedTuple):
    """Host world-line estimates of every beam particle (lab frame)."""

    entry_time: np.ndarray
    exit_time: np.ndarray
    velocity: np.ndarray
    mean_velocity: np.ndarray
    lorentz_factor: np.ndarray


def _host_boost(
    gamma: float, beta: float, c: float, time: np.ndarray, position: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Boosted ``(t′, z′)`` of lab events along the boost axis."""
    return gamma * (time - beta * position / c), gamma * (position - beta * c * time)


def _sinc(value: float, /) -> float:
    return 1.0 if value == 0.0 else math.sin(value) / value


def _smooth_step(value: np.ndarray, /) -> np.ndarray:
    """``C^∞`` step ``f(u)/(f(u) + f(1 − u))``, ``f(u) = e^{−1/u}``, clipped to [0, 1]."""
    u = np.clip(value, 0.0, 1.0)
    rise = np.where(u > 0.0, np.exp(-1.0 / np.maximum(u, 1.0e-300)), 0.0)
    fall = np.where(u < 1.0, np.exp(-1.0 / np.maximum(1.0 - u, 1.0e-300)), 0.0)
    return rise / (rise + fall)


class FELFullWavePlan(StrictModule, NonTrainableState):
    """Full-wave FEL over a boosted-frame PIC run.

    ``lattice`` binds the undulator devices and the scale, which must be the
    PIC code-unit scale (``c`` of the PIC relativity scale and ``ε₀ = 1``, the
    vacuum of the spectral grid). ``wavelength`` is the lab radiation (seed)
    wavelength ``λ``. ``boost_lorentz_factor`` is ``γ_b``. The grid resolves the
    boosted wavelength with ``cells_per_wavelength`` cells along the boost
    axis; ``transverse_size``/``transverse_cells`` declare the periodic cross
    section centered on the axis (a thin cross section with a transversely
    uniform beam is the one-dimensional limit). ``steps_per_period`` sets the
    time step to the passage time of one boosted undulator period over that
    count. ``absorber_cells`` PSATD PML cells at each end of the boost axis
    absorb radiation leaving the interior (declared normal-incidence
    ``absorber_reflection``). ``seed``, ``tracks``, and ``huygens`` are
    optional; a seed antenna and a Huygens box are refused together.

    ``continuity_tolerance`` and ``constraint_tolerance`` are relative to the
    macroparticle charge density scale ``e·max(electrons)/ΔV`` of the run: the
    PIC charge-continuity and Gauss/magnetic constraints receive
    ``max(relative · scale, absolute floor)`` with the PIC defaults as floors,
    so their roundoff checks follow the beam charge instead of an absolute
    number.
    """

    lattice: FELUndulatorLattice
    frame: BoostedFramePlan
    seed: FELFullWaveSeed | None
    tracks: FELFullWaveTracks | None
    huygens: FELFullWaveHuygens | None
    wavelength: float = eqx.field(static=True)
    transverse_size: tuple[float, float] = eqx.field(static=True)
    transverse_cells: tuple[int, int] = eqx.field(static=True)
    cells_per_wavelength: int = eqx.field(static=True)
    steps_per_period: int = eqx.field(static=True)
    nci_growth_limit: float = eqx.field(static=True)
    nci_energy_fraction: float = eqx.field(static=True)
    ledger_tolerance: float = eqx.field(static=True)
    continuity_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    maximum_entities: int = eqx.field(static=True)
    absorber_cells: int = eqx.field(static=True)
    absorber_reflection: float = eqx.field(static=True)
    undulator_span: tuple[float, float] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        lattice: FELUndulatorLattice,
        wavelength: float,
        /,
        *,
        boost_lorentz_factor: float,
        transverse_size: tuple[float, float],
        transverse_cells: tuple[int, int] = (2, 2),
        cells_per_wavelength: int = 16,
        steps_per_period: int = 32,
        seed: FELFullWaveSeed | None = None,
        tracks: FELFullWaveTracks | None = None,
        huygens: FELFullWaveHuygens | None = None,
        nci_growth_limit: float = 1.0e2,
        nci_energy_fraction: float = 0.25,
        ledger_tolerance: float = 1.0e-3,
        continuity_tolerance: float = 1.0e-7,
        constraint_tolerance: float = 1.0e-6,
        maximum_entities: int = 5_000_000,
        absorber_cells: int = 16,
        absorber_reflection: float = 1.0e-6,
    ) -> None:
        if seed is not None and not isinstance(seed, FELFullWaveSeed):
            raise TypeError("seed must be FELFullWaveSeed or None.")
        if tracks is not None and not isinstance(tracks, FELFullWaveTracks):
            raise TypeError("tracks must be FELFullWaveTracks or None.")
        if huygens is not None and not isinstance(huygens, FELFullWaveHuygens):
            raise TypeError("huygens must be FELFullWaveHuygens or None.")
        scale = lattice.scale
        light = float(scale.speed_of_light)
        if light != float(PIC_CODE_RELATIVITY.speed_of_light) or (
            float(scale.vacuum_permittivity) != 1.0
        ):
            raise ValueError(
                "The full-wave FEL runs in PIC code units: bind the lattice to an "
                "ElectromagneticScaleContract whose speed of light is the PIC "
                "relativity scale's and whose vacuum permittivity is one (the "
                "spectral grid's vacuum)."
            )
        if tracks is not None and (
            tracks.radiation.plan.scale.scale_id != scale.scale_id
        ):
            raise ValueError("The trajectory-radiation plan must use the lattice scale.")
        if seed is not None and lattice.polarization != "planar":
            raise ValueError(
                "The full-wave seed is a linearly polarized plane wave; it couples to "
                "planar lattices only (helical lattices need a circular seed)."
            )
        if seed is not None and huygens is not None:
            raise ValueError(
                "A seed antenna carries sheet current at every node along the boost "
                "axis, so no Huygens surface is current-free: extract seeded runs "
                "through tracks."
            )
        radiation = positive_finite_float(wavelength, "wavelength")
        gamma = positive_finite_float(boost_lorentz_factor, "boost_lorentz_factor")
        if gamma <= 1.0:
            raise ValueError(
                "boost_lorentz_factor must exceed one: the lab frame is not a boost."
            )
        width = tuple(
            positive_finite_float(value, "transverse_size") for value in transverse_size
        )
        cells = tuple(
            positive_integer(value, "transverse_cells") for value in transverse_cells
        )
        if len(width) != 2 or len(cells) != 2:
            raise ValueError("transverse_size and transverse_cells hold two entries.")
        if min(cells) < 2:
            raise ValueError("Every periodic transverse axis needs at least two cells.")
        resolution = positive_integer(cells_per_wavelength, "cells_per_wavelength")
        steps = positive_integer(steps_per_period, "steps_per_period")
        tolerance = positive_finite_float(ledger_tolerance, "ledger_tolerance")
        entities = positive_integer(maximum_entities, "maximum_entities")
        layers = positive_integer(absorber_cells, "absorber_cells")
        if layers < 2:
            raise ValueError("absorber_cells must be at least two.")
        reflection = positive_finite_float(absorber_reflection, "absorber_reflection")
        if reflection >= 1.0:
            raise ValueError("absorber_reflection must be below one.")
        lower, upper = _lattice_span(lattice)
        beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
        frame = BoostedFramePlan(
            LorentzFrame(boost_matrix(jnp.asarray([0.0, 0.0, beta]))),
            Box(
                [0.0, 0.0, 0.5 * (lower + upper)],
                [width[0], width[1], upper - lower],
            ),
            relativity=PIC_CODE_RELATIVITY,
        )
        self.lattice = lattice
        self.frame = frame
        self.seed = seed
        self.tracks = tracks
        self.huygens = huygens
        self.wavelength = radiation
        self.transverse_size = (width[0], width[1])
        self.transverse_cells = (cells[0], cells[1])
        self.cells_per_wavelength = resolution
        self.steps_per_period = steps
        self.nci_growth_limit = float(nci_growth_limit)
        self.nci_energy_fraction = float(nci_energy_fraction)
        self.ledger_tolerance = tolerance
        self.continuity_tolerance = positive_finite_float(
            continuity_tolerance, "continuity_tolerance"
        )
        self.constraint_tolerance = positive_finite_float(
            constraint_tolerance, "constraint_tolerance"
        )
        self.maximum_entities = entities
        self.absorber_cells = layers
        self.absorber_reflection = reflection
        self.undulator_span = (lower, upper)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fel-full-wave-plan",
                "lattice": lattice.lattice_id,
                "frame": frame.plan_id,
                "wavelength": radiation,
                "transverse_size": list(self.transverse_size),
                "transverse_cells": list(self.transverse_cells),
                "cells_per_wavelength": resolution,
                "steps_per_period": steps,
                "seed": None if seed is None else seed.seed_id,
                "tracks": None if tracks is None else tracks.tracks_id,
                "huygens": None if huygens is None else huygens.huygens_id,
                "nci_growth_limit": self.nci_growth_limit,
                "nci_energy_fraction": self.nci_energy_fraction,
                "ledger_tolerance": tolerance,
                "continuity_tolerance": self.continuity_tolerance,
                "constraint_tolerance": self.constraint_tolerance,
                "absorber_cells": layers,
                "absorber_reflection": reflection,
            }
        )

    # -- lattice and frame quantities -------------------------------------------------

    @property
    def speed_of_light(self) -> float:
        return float(self.lattice.scale.speed_of_light)

    @property
    def boosted_wavelength(self) -> float:
        """``λ′ = λ γ_b (1 + β_b)``."""
        return self.wavelength * self.frame.lorentz_factor * (1.0 + self.frame.beta)

    @property
    def boosted_undulator_period(self) -> float:
        """``λ_u′ = λ_u/γ_b`` of the shortest lattice period."""
        return _shortest_period(self.lattice) / self.frame.lorentz_factor

    @property
    def _rms_strength_squared(self) -> float:
        """Largest ``a_w²`` of the lattice (slowest mean longitudinal velocity)."""
        return max(
            undulator_rms_strength_squared(value, self.lattice.polarization)
            for value in self.lattice.deflections
        )

    def resonant_lorentz_factor(self, lorentz_factor: float, /) -> float:
        """``γ_z = γ/√(1 + a_w²)``: the boost that puts the beam at rest."""
        gamma = positive_finite_float(lorentz_factor, "lorentz_factor")
        return gamma / math.sqrt(1.0 + self._rms_strength_squared)

    # -- beam loading -------------------------------------------------------------------

    def flat_top_beam(
        self,
        lorentz_factor: float,
        current: float,
        /,
        *,
        wavelengths: int,
        particles_per_wavelength: int = 8,
        taper_wavelengths: float = 4.0,
        phase_modulation: float = 0.0,
        modulation_phase: float = 0.0,
        species: FELFullWaveBeamSpecies = "electron-positron",
    ) -> FELFullWaveBeam:
        """Quiet-start flat-top beam filling the periodic cross section.

        ``wavelengths`` ponderomotive periods ``2π/(k + k_u)`` of total current
        ``current`` (shared equally by the members of electron–positron pairs)
        at energy ``lorentz_factor``, each holding ``particles_per_wavelength``
        equally spaced phases, with ``sin²`` density tapers of
        ``taper_wavelengths`` radiation wavelengths at both ends. The head sits
        half a period upstream of the lattice support. Phases
        ``θ = ψ − a sin(ψ − φ)`` with ``a = phase_modulation`` and
        ``φ = modulation_phase`` give the fundamental bunching ``J₁(a) e^{−iφ}``.

        One-dimensional limit (planar lattices, two vertical cells): every
        longitudinal particle is replicated on every horizontal cell center and
        sits on the undulator axis vertically, a cell center of the two-cell
        axis. The deposited current is then transversely uniform under any
        common horizontal wiggle, and the vertical natural focusing vanishes on
        the axis.
        """
        if self.lattice.polarization != "planar" or self.transverse_cells[1] != 2:
            raise ValueError(
                "flat_top_beam loads the one-dimensional limit: a planar lattice and "
                "two vertical cells."
            )
        gamma = positive_finite_float(lorentz_factor, "lorentz_factor")
        if gamma <= 1.0:
            raise ValueError("lorentz_factor must exceed one.")
        total = positive_finite_float(current, "current")
        periods = positive_integer(wavelengths, "wavelengths")
        per_period = positive_integer(
            particles_per_wavelength, "particles_per_wavelength"
        )
        if per_period < 2:
            raise ValueError("A quiet start needs at least two particles per period.")
        taper = finite_real_scalar(taper_wavelengths, "taper_wavelengths")
        if taper < 0.0:
            raise ValueError("taper_wavelengths must be nonnegative.")
        modulation = finite_real_scalar(phase_modulation, "phase_modulation")
        offset = finite_real_scalar(modulation_phase, "modulation_phase")
        kind = parse(species, FELFullWaveBeamSpecies, "species")
        c = self.speed_of_light
        beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
        k = 2.0 * math.pi / self.wavelength
        ku = 2.0 * math.pi / _shortest_period(self.lattice)
        period = 2.0 * math.pi / (k + ku)
        head = self.undulator_span[0] - 0.5 * period
        length = periods * period
        tail = head - length
        count = periods * per_period
        psi = 2.0 * math.pi * (np.arange(count, dtype=np.float64) + 0.5) / per_period
        theta = psi - modulation * np.sin(psi - offset)
        z = tail + theta / (k + ku)
        ramp = taper * self.wavelength
        if ramp > 0.0:
            rising = np.clip((z - tail) / ramp, 0.0, 1.0)
            falling = np.clip((head - z) / ramp, 0.0, 1.0)
            density = np.sin(0.5 * np.pi * np.minimum(rising, falling)) ** 2
        else:
            density = np.ones_like(z)
        if np.any(density <= 0.0):
            raise ValueError("The density taper leaves an empty macroparticle.")
        width_x = self.transverse_size[0]
        cells_x = self.transverse_cells[0]
        lower_x, _ = _transverse_lower(self)
        xs = lower_x + (np.arange(cells_x) + 0.5) * width_x / cells_x
        copies = [(x, 0.0) for x in xs]
        positions = np.concatenate(
            [
                np.stack((np.full(count, x), np.full(count, y), z), axis=-1)
                for x, y in copies
            ]
        )
        velocities = np.tile(np.asarray([0.0, 0.0, beta * c]), (positions.shape[0], 1))
        match kind:
            case "electron":
                members = 1.0
            case "electron-positron":
                members = 2.0
            case _:
                assert_never(kind)
        # Current I = e n β c A: electrons per macroparticle over one phase step.
        electrons = (
            total
            / (members * float(self.lattice.scale.elementary_charge) * beta * c)
            * (period / per_period)
            / len(copies)
        )
        weights = np.tile(electrons * density, len(copies))
        return FELFullWaveBeam(
            positions,
            velocities,
            weights,
            lab_time=0.0,
            species=kind,
            flat_top=(tail + ramp, head - ramp),
        )

    # -- preparation -------------------------------------------------------------------

    def prepare(self, beam: FELFullWaveBeam, /) -> PreparedFELFullWave:
        """Size the boosted grid and schedule for ``beam`` and bind the PIC run."""
        return PreparedFELFullWave(self, beam)


def _transverse_lower(plan: FELFullWavePlan, /) -> tuple[float, float]:
    """Lower transverse grid corner that puts the undulator axis at a cell center."""
    values = [
        -0.5 * size + (0.5 * size / cells if cells % 2 == 0 else 0.0)
        for size, cells in zip(plan.transverse_size, plan.transverse_cells, strict=True)
    ]
    return values[0], values[1]


def _lattice_span(lattice: FELUndulatorLattice, /) -> tuple[float, float]:
    """Lab support ``[z_in, z_out]`` of every lattice device."""
    lows, highs = [], []
    for segment in lattice.segments:
        device = segment.device
        half = (
            0.5 * device.period * device.period_count
            + _TERMINATION_WIDTHS * device.ramp_width
        )
        lows.append(device.center - half)
        highs.append(device.center + half)
    return min(lows), max(highs)


def _shortest_period(lattice: FELUndulatorLattice, /) -> float:
    return min(segment.device.period for segment in lattice.segments)


def _kinematics(plan: FELFullWavePlan, beam: FELFullWaveBeam, /) -> _Kinematics:
    """Lab entry/exit times of every particle (ballistic, then mean undulator motion)."""
    c = plan.speed_of_light
    lower, upper = plan.undulator_span
    position = np.asarray(beam.positions)
    velocity = np.asarray(beam.velocities)
    speed2 = np.sum(velocity * velocity, axis=1)
    if np.any(speed2 >= c * c):
        raise ValueError("Beam velocities must be subluminal.")
    if np.any(position[:, 2] >= lower):
        raise ValueError(
            "Every beam particle must start upstream of the lattice support "
            f"(z < {lower:.6g}); it is injected ballistically in vacuum."
        )
    gamma = 1.0 / np.sqrt(1.0 - speed2 / (c * c))
    strength = plan._rms_strength_squared
    if np.any(gamma * gamma <= 1.0 + strength):
        raise ValueError(
            "Beam energies are too low to cross the undulator (γ² ≤ 1 + a_w²)."
        )
    mean = c * np.sqrt(1.0 - (1.0 + strength) / (gamma * gamma))
    entry = beam.lab_time + (lower - position[:, 2]) / velocity[:, 2]
    exit_ = entry + (upper - lower) / mean
    return _Kinematics(entry, exit_, velocity[:, 2], mean, gamma)


class _Schedule(NamedTuple):
    start: float
    interaction_end: float
    stop: float
    step_size: float
    step_count: int


class _Layout(NamedTuple):
    lower: float
    length: float
    cells: int
    spacing: float


class PreparedFELFullWave(StrictModule, NonTrainableState):
    """A full-wave FEL run bound to one beam: grid, schedule, species, and seed."""

    plan: FELFullWavePlan
    beam: FELFullWaveBeam
    boosted: PreparedBoostedFrame
    far_field: MaxwellFarFieldPlan | None
    initial_positions: Array
    initial_proper_velocities: Array
    masses: Array
    frame_evidence: FELFullWaveFrameEvidence
    steady_window: tuple[float, float] | None = eqx.field(static=True)
    reference_window: tuple[float, float] | None = eqx.field(static=True)
    layout: tuple[float, float, int, float] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: FELFullWavePlan, beam: FELFullWaveBeam, /) -> None:
        frame = plan.frame
        c = plan.speed_of_light
        gamma_b, beta_b = frame.lorentz_factor, frame.beta
        kinematics = _kinematics(plan, beam)
        lower, upper = plan.undulator_span
        start_times, _ = _host_boost(gamma_b, beta_b, c, kinematics.entry_time, lower)
        end_times, _ = _host_boost(gamma_b, beta_b, c, kinematics.exit_time, upper)
        seed = _seed_schedule(plan, kinematics)
        # The run starts before the antenna emits: no field overlaps the beam at
        # the PIC initialization.
        start = float(np.min(start_times))
        if seed is not None:
            start = min(start, seed.boosted_window[0])
        interaction_end = float(np.max(end_times))
        boosted = frame.boost_particles(
            np.asarray(beam.positions),
            np.asarray(beam.velocities),
            boosted_time=start,
            lab_times=beam.lab_time,
        )
        spacing = plan.boosted_wavelength / plan.cells_per_wavelength
        passage = plan.boosted_undulator_period / (beta_b * c)
        stop = interaction_end
        if seed is not None:
            stop = max(stop, seed.boosted_window[1])
        initial = np.asarray(boosted.positions)
        envelope = _beam_envelope(plan, beam, kinematics, initial, start, stop)
        if plan.huygens is not None:
            stop = interaction_end + _box_crossing(plan, envelope, spacing) / c
            envelope = _beam_envelope(plan, beam, kinematics, initial, start, stop)
        steady = _steady_window(plan, beam, kinematics)
        reference = None if seed is None else seed.reference
        # The whole seed stays on the interior at the end: an envelope measured
        # next to a wave the PML truncated would carry the truncation's ringing.
        whole = (
            None if seed is None else (seed.flat[0] - seed.ramp, seed.flat[1] + seed.ramp)
        )
        layout = _longitudinal_layout(
            plan, envelope, seed, (steady, reference, whole), stop, spacing
        )
        _refuse_charged_box(plan, beam, layout)
        huygens_box = (
            None
            if plan.huygens is None
            else _huygens_box(plan, envelope, layout, stop - start)
        )
        solver, species = _field_solver(plan, beam, layout, stop, huygens_box, seed)
        # The time step resolves the undulator passage and never exceeds the
        # unaliased PSATD step (thin transverse cells shorten it).
        step = min(passage / plan.steps_per_period, float(solver.stable_step))
        # One step beyond the last requirement: the accumulated run time must not
        # fall short of the acquisition stop by roundoff.
        count = int(math.floor((stop - start) / step)) + 1
        schedule = _Schedule(start, interaction_end, stop, step, count)
        cell = layout.spacing * math.prod(
            size / cells
            for size, cells in zip(
                plan.transverse_size, plan.transverse_cells, strict=True
            )
        )
        density = (
            float(plan.lattice.scale.elementary_charge)
            * float(np.max(np.asarray(beam.electrons)))
            / cell
        )
        run = frame.prepare(
            ElectromagneticPICPlan(
                solver,
                species=species,
                external_fields=tuple(
                    frame.boost_external_field(segment.device)
                    for segment in plan.lattice.segments
                ),
                recorders=_recorders(plan, beam, species, schedule),
                continuity_tolerance=max(
                    _CONTINUITY_FLOOR, plan.continuity_tolerance * density
                ),
                constraint_tolerance=max(
                    _CONSTRAINT_FLOOR, plan.constraint_tolerance * density
                ),
            ),
            nci_growth_limit=plan.nci_growth_limit,
            nci_energy_fraction=plan.nci_energy_fraction,
        )
        masses = float(plan.lattice.scale.electron_mass) * np.asarray(beam.electrons)
        self.plan = plan
        self.beam = beam
        self.boosted = run
        self.far_field = (
            None
            if plan.huygens is None
            else MaxwellFarFieldPlan(
                jnp.asarray(plan.huygens.directions),
                jnp.asarray(plan.huygens.reference_axis),
                HomogeneousMaxwellExterior(),
            )
        )
        self.initial_positions = boosted.positions
        self.initial_proper_velocities = boosted.proper_velocities
        self.masses = jnp.asarray(masses)
        self.frame_evidence = _frame_evidence(plan, beam, kinematics, layout, schedule)
        self.steady_window = steady
        self.reference_window = reference
        self.layout = (layout.lower, layout.length, layout.cells, layout.spacing)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fel-full-wave",
                "plan": plan.plan_id,
                "beam": beam.beam_id,
                "run": run.prepared_id,
                "transform_numerical": solver.transform.plan.numerical_id,
                "transform_execution": solver.transform.plan.execution_id,
                "transform_plan": solver.transform.plan.plan_id,
                "step_size": step,
                "step_count": count,
            }
        )

    @property
    def schedule(self) -> tuple[float, float, int]:
        """``(start, step_size, step_count)`` of the boosted run."""
        evidence = self.frame_evidence
        return evidence.start_time, evidence.step_size, evidence.step_count

    def run(self) -> FELFullWaveResult:
        """Integrate the boosted run and extract lab-frame radiation and ledgers."""
        start, step, count = self.schedule
        initial, final, successful, exchange = _integrate(
            self.boosted,
            self.initial_positions,
            self.initial_proper_velocities,
            self.masses,
            jnp.asarray(start, dtype=jnp.float64),
            step,
            count,
            _species_count(self.beam),
        )
        return self._result(initial, final, successful, exchange)

    # -- post-processing -------------------------------------------------------------

    def _lab_lorentz_factors(self, state: BoostedFrameState, /) -> Array:
        frame = self.plan.frame
        c = self.plan.speed_of_light
        values = []
        for species in state.pic.species:
            proper = frame.lab_proper_velocity(species.particles.proper_velocity)
            values.append(jnp.sqrt(1.0 + jnp.sum(proper * proper, axis=-1) / (c * c)))
        return jnp.stack(values)

    def _beam_energy(self, lorentz_factors: Array, /) -> Array:
        c = self.plan.speed_of_light
        return jnp.sum(self.masses[None, :] * c * c * (lorentz_factors - 1.0))

    def _field_energy(self, state: BoostedFrameState, /) -> Array:
        """Lab energy ``γ_b(U′ + β_b c P′_z)`` of the grid field.

        ``B_x`` and ``B_y`` sit half a cell above ``E_y`` and ``E_x`` along the
        boost axis on the staggered grid; they are moved onto the ``E``
        locations by the band-limited half-cell shift before the product.
        """
        solver = self.boosted.solver
        electric, magnetic = state.pic.field.electric, state.pic.field.magnetic
        volume = float(np.prod(solver.plan.spacing))
        c = self.plan.speed_of_light
        shifted = _half_cell_down(magnetic[..., :2], self.layout[3], axis=2)
        momentum = (
            float(solver.plan.permittivity)
            * volume
            * jnp.sum(
                electric[..., 0] * shifted[..., 1] - electric[..., 1] * shifted[..., 0]
            )
        )
        frame = self.plan.frame
        return frame.lorentz_factor * (
            solver.field_energy(state.pic.field) + frame.beta * c * momentum
        )

    def _forward(self, state: BoostedFrameState, /) -> tuple[Array, Array]:
        """Lab envelope ``|E|`` about ``k′`` and lab energy of the forward k⊥ = 0 wave."""
        solver = self.boosted.solver
        c = self.plan.speed_of_light
        frame = self.plan.frame
        _, _, cells, spacing = self.layout
        electric = jnp.mean(state.pic.field.electric, axis=(0, 1))
        magnetic = _half_cell_down(
            jnp.mean(state.pic.field.magnetic, axis=(0, 1)), spacing, axis=0
        )
        forward = jnp.stack(
            (
                0.5 * (electric[:, 0] + c * magnetic[:, 1]),
                0.5 * (electric[:, 1] - c * magnetic[:, 0]),
            ),
            axis=-1,
        )
        wavenumbers = 2.0 * np.pi * np.fft.fftfreq(cells, d=spacing)
        target = 2.0 * math.pi / self.plan.boosted_wavelength
        window = np.where(
            wavenumbers > 0.0,
            np.exp(-0.5 * ((wavenumbers / target - 1.0) / _BAND_WIDTH) ** 2),
            0.0,
        )
        spectrum = jnp.fft.fft(forward, axis=0)
        analytic = jnp.fft.ifft(jnp.asarray(2.0 * window)[:, None] * spectrum, axis=0)
        doppler = frame.lorentz_factor * (1.0 + frame.beta)
        amplitude = doppler * jnp.sqrt(jnp.sum(jnp.abs(analytic) ** 2, axis=-1))
        area = self.plan.transverse_size[0] * self.plan.transverse_size[1]
        energy = (
            doppler
            * float(solver.plan.permittivity)
            * area
            * spacing
            * jnp.sum(forward * forward)
        )
        return amplitude, energy

    def _window_indices(
        self, window: tuple[float, float] | None, stop: float, /
    ) -> Array | None:
        """Grid nodes holding the forward field elements of a lab ``ξ`` window."""
        if window is None:
            return None
        frame = self.plan.frame
        c = self.plan.speed_of_light
        lower, _, cells, spacing = self.layout
        stretch = frame.lorentz_factor * (1.0 - frame.beta)
        samples = np.arange(window[0], window[1], stretch * spacing)
        if samples.size == 0:
            return None
        # A forward field element keeps its co-moving coordinate ξ = z − ct; the
        # layout holds every window inside the interior at the end of the run.
        boosted = samples / stretch + c * stop
        index = np.floor((boosted - lower) / spacing).astype(np.int64)
        if np.any(index < 0) or np.any(index >= cells):
            raise ValueError("A measurement window left the grid; the layout is wrong.")
        return jnp.asarray(index)

    def _ledger(
        self,
        initial: BoostedFrameState,
        final: BoostedFrameState,
        exchange: Array,
        /,
    ) -> FELFullWaveLedger:
        """Lab energy balance; ``exchange`` holds the boosted absorbed energy,
        absorbed momentum ``P′_z``, and antenna work summed over accepted steps."""
        frame = self.plan.frame
        c = self.plan.speed_of_light
        initial_gamma = self._lab_lorentz_factors(initial)
        final_gamma = self._lab_lorentz_factors(final)
        beam = self._beam_energy(final_gamma) - self._beam_energy(initial_gamma)
        field = self._field_energy(final) - self._field_energy(initial)
        escaped = frame.lorentz_factor * (exchange[0] + frame.beta * c * exchange[1])
        # The one-way sheet launches a transversely uniform forward plane wave:
        # its boosted momentum is W′/c.
        injected = frame.lorentz_factor * (1.0 + frame.beta) * exchange[2]
        defect = beam + field + escaped - injected
        total = self._beam_energy(initial_gamma) + jnp.abs(injected)
        return FELFullWaveLedger(
            beam,
            field,
            escaped,
            injected,
            defect,
            jnp.abs(defect) / total,
        )

    def _result(
        self,
        initial: BoostedFrameState,
        final: BoostedFrameState,
        successful: Array,
        exchange: Array,
        /,
    ) -> FELFullWaveResult:
        plan = self.plan
        frame = plan.frame
        c = plan.speed_of_light
        evidence = self.frame_evidence
        stop = evidence.start_time + evidence.step_count * evidence.step_size
        initial_gamma = self._lab_lorentz_factors(initial)
        final_gamma = self._lab_lorentz_factors(final)
        ledger = self._ledger(initial, final, exchange)
        defect = ledger.defect
        _, initial_energy = self._forward(initial)
        amplitude, energy = self._forward(final)
        lower, _, cells, spacing = self.layout
        nodes = lower + spacing * jnp.arange(cells, dtype=jnp.float64)
        coordinates = frame.lorentz_factor * (1.0 - frame.beta) * (nodes - c * stop)
        steady = self._window_indices(self.steady_window, stop)
        reference = self._window_indices(self.reference_window, stop)
        steady_amplitude = None if steady is None else jnp.mean(amplitude[steady])
        seed_amplitude = None if reference is None else jnp.mean(amplitude[reference])
        gain_profile = (
            None if seed_amplitude is None else (amplitude / seed_amplitude) ** 2 - 1.0
        )
        steady_gain = (
            None
            if gain_profile is None or steady is None
            else jnp.mean(gain_profile[steady])
        )
        accepted = jnp.sum(successful.astype(jnp.int32))
        complete = accepted == evidence.step_count
        exited = self._exited(final)
        finite = (
            jnp.all(jnp.isfinite(final_gamma))
            & jnp.all(jnp.isfinite(final.pic.field.electric))
            & jnp.all(jnp.isfinite(final.pic.field.magnetic))
            & jnp.isfinite(defect)
        )
        trajectory, huygens = self._radiation(final, bool(complete))
        radiation_status = (
            jnp.asarray(0, dtype=jnp.int32)
            if trajectory is None
            else trajectory.evidence.status
        )
        status = (
            jnp.where(finite, 0, int(FELFullWaveStatus.NONFINITE))
            | jnp.where(complete, 0, int(FELFullWaveStatus.STEP_REJECTED))
            | jnp.where(
                final.evidence.nci_rejections > 0,
                int(FELFullWaveStatus.NCI_REJECTED),
                0,
            )
            | jnp.where(exited, 0, int(FELFullWaveStatus.UNDULATOR_UNFINISHED))
            | jnp.where(
                ledger.relative_defect > plan.ledger_tolerance,
                int(FELFullWaveStatus.LEDGER_DEFECT),
                0,
            )
            | jnp.where(
                radiation_status != 0, int(FELFullWaveStatus.RADIATION_UNRESOLVED), 0
            )
        ).astype(jnp.int32)
        return FELFullWaveResult(
            initial_lorentz_factors=initial_gamma,
            final_lorentz_factors=final_gamma,
            forward_amplitude=amplitude,
            forward_coordinates=coordinates,
            forward_energy=jnp.stack((initial_energy, energy)),
            steady_amplitude=steady_amplitude,
            seed_amplitude=seed_amplitude,
            gain_profile=gain_profile,
            steady_gain=steady_gain,
            trajectory_spectrum=trajectory,
            huygens_spectrum=huygens,
            ledger=ledger,
            evidence=FELFullWaveEvidence(
                status=status,
                frame=evidence,
                boosted=final.evidence,
                accepted_steps=accepted,
                exited=exited,
                finite=finite,
            ),
            final_state=final,
            plan_id=plan.plan_id,
        )

    def _exited(self, state: BoostedFrameState, /) -> Array:
        """Every particle's final slice event lies downstream of the lattice support."""
        frame = self.plan.frame
        _, upper = self.plan.undulator_span
        downstream = jnp.asarray(True)
        for species in state.pic.species:
            _, lab = frame.to_lab(state.pic.time, species.particles.position)
            downstream = downstream & jnp.all(lab[:, 2] >= upper)
        return downstream

    def _radiation(
        self, state: BoostedFrameState, complete: bool, /
    ) -> tuple[TrajectoryRadiationResult | None, LorentzSpectralTransform | None]:
        if not complete:
            return None, None
        plan = self.plan
        trajectory = None
        if plan.tracks is not None:
            tracks = self.boosted.lab_trajectory(state, 0, plan.lattice.scale)
            trajectory = plan.tracks.radiation.evaluate(tracks)
        huygens = None
        if self.far_field is not None:
            phasors = self.boosted.solver.huygens_phasors(state.pic.field)[0]
            huygens = self.boosted.lab_far_field(
                state, self.far_field.evaluate(phasors), emission="complete"
            )
        return trajectory, huygens


# -- preparation helpers ----------------------------------------------------------------


def _species_count(beam: FELFullWaveBeam, /) -> int:
    match beam.species:
        case "electron":
            return 1
        case "electron-positron":
            return 2
        case _:
            assert_never(beam.species)


def _seed_schedule(
    plan: FELFullWavePlan, kinematics: _Kinematics, /
) -> _SeedSchedule | None:
    """Antenna on the lab entrance plane emitting ``g(ξ)`` over every slippage
    interval; its lab and boosted emission windows and boosted sheet path."""
    seed = plan.seed
    if seed is None:
        return None
    frame = plan.frame
    gamma_b, beta_b = frame.lorentz_factor, frame.beta
    c = plan.speed_of_light
    lower, upper = plan.undulator_span
    entry_xi = lower - c * kinematics.entry_time
    slip = (upper - lower) * (c / kinematics.mean_velocity - 1.0)
    margin = seed.margin_wavelengths * plan.wavelength
    ramp = seed.ramp_wavelengths * plan.wavelength
    flat = (
        float(np.min(entry_xi - slip)) - margin,
        float(np.max(entry_xi)) + margin,
    )
    # The element ξ leaves the plane z_a at the lab time (z_a − ξ)/c.
    lab = ((lower - (flat[1] + ramp)) / c, (lower - (flat[0] - ramp)) / c)
    times, positions = _host_boost(gamma_b, beta_b, c, np.asarray(lab), lower)
    return _SeedSchedule(
        flat=flat,
        ramp=ramp,
        # The margin half one wavelength or more past the ramp: the envelope
        # estimate there is exact to ~2e-5.
        reference=(flat[0] + 0.5 * margin, flat[0] + margin),
        plane=lower,
        lab_window=lab,
        boosted_window=(float(times[0]), float(times[1])),
        path=(float(positions[1]), float(positions[0])),
    )


def _seed_antenna(
    plan: FELFullWavePlan,
    schedule: _SeedSchedule,
    bridge: StructuredCochainBridge,
    /,
) -> SampledPlaneCurrentAntennaPlan:
    """The lab entrance-plane antenna, boosted onto ``bridge``.

    With ``τ = t − (z − z_a)/c`` the sheet field ``Re[A(τ) e^{−iωτ}]`` is
    ``E₀ g(ξ) cos(kξ + φ)`` for ``A(τ) = E₀ g(z_a − cτ) e^{i(φ + k z_a)}``.
    """
    seed = plan.seed
    if seed is None:
        raise ValueError("The plan declares no seed.")
    c = plan.speed_of_light
    k = 2.0 * math.pi / plan.wavelength
    ramp = schedule.ramp
    lab = schedule.lab_window
    # 32 envelope samples per ramp: the cubic Hermite sheet interpolation of
    # the smooth step is then exact to ~1e-6.
    count = int(math.ceil(32.0 * c * (lab[1] - lab[0]) / ramp)) + 1
    times = np.linspace(lab[0], lab[1], count)
    xi = schedule.plane - c * times
    rising = _smooth_step((xi - (schedule.flat[0] - ramp)) / ramp)
    falling = _smooth_step(((schedule.flat[1] + ramp) - xi) / ramp)
    envelope = rising * falling
    electric = np.zeros((2, 2, count, 2), dtype=np.complex128)
    electric[..., 0] = (
        seed.amplitude * envelope * np.exp(1j * (seed.phase + k * schedule.plane))
    )
    # The aperture samples bracket the whole periodic cross section.
    lower_x, lower_y = _transverse_lower(plan)
    width_x, width_y = plan.transverse_size
    lab_antenna = SampledPlaneCurrentAntennaPlan(
        bridge,
        2,
        schedule.plane,
        np.asarray([lower_x - width_x, lower_x + 2.0 * width_x]),
        np.asarray([lower_y - width_y, lower_y + 2.0 * width_y]),
        times,
        electric,
        carrier_angular_frequency=c * k,
        scale=plan.lattice.scale,
    )
    return plan.frame.boost_antenna(lab_antenna, bridge)


def _boosted_path(
    plan: FELFullWavePlan,
    kinematics: _Kinematics,
    initial: np.ndarray,
    duration: float,
    /,
) -> np.ndarray:
    """Bounds of every particle's boosted ``z′`` over ``duration``.

    The lab longitudinal velocity stays between the undulator mean ``v̄_z`` and
    the free velocity, so the boosted velocity stays between their boosted
    images and ``z′`` between the two linear extrapolations.
    """
    frame = plan.frame
    c = plan.speed_of_light
    beta_b = frame.beta

    def boosted(velocity: np.ndarray, /) -> np.ndarray:
        return (velocity - beta_b * c) / (1.0 - beta_b * velocity / c)

    free = boosted(kinematics.velocity)
    slowed = boosted(kinematics.mean_velocity)
    return np.stack(
        (
            initial + np.minimum(np.minimum(free, slowed), 0.0) * duration,
            initial + np.maximum(np.maximum(free, slowed), 0.0) * duration,
        )
    )


class _Envelope(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray


def _beam_envelope(
    plan: FELFullWavePlan,
    beam: FELFullWaveBeam,
    kinematics: _Kinematics,
    positions: np.ndarray,
    start: float,
    stop: float,
    /,
) -> _Envelope:
    """Boosted bounding box of the beam over ``[start, stop]`` including wiggles.

    ``positions`` are the boosted positions on the first slice.
    """
    path = _boosted_path(plan, kinematics, positions[:, 2], stop - start)
    frame = plan.frame
    c = plan.speed_of_light
    velocities = np.asarray(beam.velocities)
    transverse = velocities[:, :2] / (
        frame.lorentz_factor * (1.0 - frame.beta * velocities[:, 2:3] / c)
    )
    drift = np.abs(transverse) * (stop - start)
    ku = 2.0 * math.pi / _shortest_period(plan.lattice)
    deflection = max(plan.lattice.deflections)
    amplitude = deflection / (float(np.min(kinematics.lorentz_factor)) * ku)
    match plan.lattice.polarization:
        case "planar":
            wiggle = np.asarray([amplitude, 0.0])
        case "helical":
            wiggle = np.asarray([amplitude, amplitude])
        case _:
            assert_never(plan.lattice.polarization)
    low = np.concatenate(
        (np.min(positions[:, :2] - drift, axis=0) - wiggle, [np.min(path)])
    )
    high = np.concatenate(
        (np.max(positions[:, :2] + drift, axis=0) + wiggle, [np.max(path)])
    )
    return _Envelope(low, high)


def _box_crossing(plan: FELFullWavePlan, envelope: _Envelope, spacing: float, /) -> float:
    """Diagonal of the Huygens box around the envelope: the longest crossing path."""
    huygens = plan.huygens
    if huygens is None:
        raise ValueError("The plan declares no Huygens extraction.")
    widths = [
        plan.transverse_size[0] / plan.transverse_cells[0],
        plan.transverse_size[1] / plan.transverse_cells[1],
        spacing,
    ]
    extent = [
        float(envelope.upper[axis] - envelope.lower[axis])
        + 2.0 * (huygens.margin_cells[axis] + _SHAPE_GUARD + 1) * widths[axis]
        for axis in range(3)
    ]
    return math.sqrt(sum(value * value for value in extent))


def _longitudinal_layout(
    plan: FELFullWavePlan,
    envelope: _Envelope,
    seed: _SeedSchedule | None,
    windows: tuple[tuple[float, float] | None, ...],
    stop: float,
    spacing: float,
    /,
) -> _Layout:
    """Boost-axis grid: an interior holding the beam envelope, the seed sheet's
    active path, and the measured ``ξ`` windows at ``stop``, between two PML
    layers that absorb whatever leaves it."""
    c = plan.speed_of_light
    frame = plan.frame
    stretch = frame.lorentz_factor * (1.0 - frame.beta)
    lows, highs = [float(envelope.lower[2])], [float(envelope.upper[2])]
    if seed is not None:
        lows.append(seed.path[0])
        highs.append(seed.path[1])
    for window in windows:
        if window is not None:
            lows.append(window[0] / stretch + c * stop)
            highs.append(window[1] / stretch + c * stop)
    guard = _LAYER_GUARD
    if plan.huygens is not None:
        guard += plan.huygens.margin_cells[2] + _SHAPE_GUARD
    interior = int(math.ceil((max(highs) - min(lows)) / spacing)) + 2 * guard
    count = interior + 2 * plan.absorber_cells
    lower = min(lows) - (guard + plan.absorber_cells) * spacing
    return _Layout(lower, count * spacing, count, spacing)


def _huygens_box(
    plan: FELFullWavePlan,
    envelope: _Envelope,
    layout: _Layout,
    duration: float,
    /,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Node bounds enclosing the envelope; refuse boxes transverse periodic
    images can reach (the boost axis is open)."""
    huygens = plan.huygens
    if huygens is None:
        raise ValueError("The plan declares no Huygens extraction.")
    c = plan.speed_of_light
    lower_x, lower_y = _transverse_lower(plan)
    origins = (lower_x, lower_y, layout.lower)
    widths = (
        plan.transverse_size[0] / plan.transverse_cells[0],
        plan.transverse_size[1] / plan.transverse_cells[1],
        layout.spacing,
    )
    counts = (plan.transverse_cells[0], plan.transverse_cells[1], layout.cells)
    lows, highs = [], []
    for axis in range(3):
        guard = huygens.margin_cells[axis] + _SHAPE_GUARD
        low = (
            int(math.floor((envelope.lower[axis] - origins[axis]) / widths[axis])) - guard
        )
        high = (
            int(math.ceil((envelope.upper[axis] - origins[axis]) / widths[axis])) + guard
        )
        if low < 1 or high > counts[axis] - 1:
            raise ValueError(
                f"The Huygens box does not fit inside the grid along axis {axis}: "
                "enlarge the transverse cross section."
            )
        clearance = (counts[axis] - (high - low)) * widths[axis]
        if axis < 2 and clearance < c * duration:
            raise ValueError(
                f"Periodic images of the radiation reach the Huygens box along axis "
                f"{axis} before the acquisition closes: the grid needs "
                f"{c * duration + (high - low) * widths[axis]:.6g} along this axis, "
                f"it has {counts[axis] * widths[axis]:.6g}."
            )
        lows.append(low)
        highs.append(high)
    return (lows[0], lows[1], lows[2]), (highs[0], highs[1], highs[2])


def _field_solver(
    plan: FELFullWavePlan,
    beam: FELFullWaveBeam,
    layout: _Layout,
    stop: float,
    huygens_box: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
    seed: _SeedSchedule | None,
    /,
) -> tuple[PreparedSpectralMaxwell, tuple[PICSpeciesPlan, ...]]:
    width_x, width_y = plan.transverse_size
    cells_x, cells_y = plan.transverse_cells
    lower_x, lower_y = _transverse_lower(plan)
    grid = TensorGridPlan(
        (
            UniformCellAxisSpec(cells_x, periodic=True),
            UniformCellAxisSpec(cells_y, periodic=True),
            UniformCellAxisSpec(layout.cells, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            [
                [lower_x, lower_y, layout.lower],
                [lower_x + width_x, lower_y + width_y, layout.lower + layout.length],
            ]
        )
    )
    bridge = StructuredCochainBridge(
        grid,
        resources=StructuredCochainResourcePolicy(maximum_entities=plan.maximum_entities),
    )
    scale = plan.lattice.scale
    specific = float(scale.elementary_charge) / float(scale.electron_mass)
    count = beam.macroparticle_count
    members = _species_count(beam)
    names = ("electron", "positron")[:members]
    signs = (-1.0, 1.0)[:members]
    plans, charged = [], []
    for index, (name, sign) in enumerate(zip(names, signs, strict=True)):
        support = ParticleSetPlan(
            jnp.arange(index * count, (index + 1) * count),
            jnp.ones((count,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            ChargedParticlePlan(sign * specific * jnp.ones((count,)), name).prepare(
                support
            )
        )
        plans.append(
            PICSpeciesPlan(
                ParticlePopulationPlan(support),
                PICChargeModelPlan(
                    sign * specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer = PICParticleCochainTransferPlan(bridge, shape_order=_SHAPE_ORDER)
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(ChargeConservingCurrentPlan(value) for value in transfers)
    observers: tuple[SpectralHuygensBoxPlan, ...] = ()
    if huygens_box is not None and plan.huygens is not None:
        observers = (
            SpectralHuygensBoxPlan(
                huygens_box[0],
                huygens_box[1],
                MaxwellSpectralAcquisition(
                    jnp.asarray(plan.huygens.angular_frequencies),
                    sign="positive",
                    measure="time-integral",
                    stop_time=stop,
                ),
                HomogeneousMaxwellExterior(),
            ),
        )
    antennas = () if seed is None else (_seed_antenna(plan, seed, bridge),)
    solver = SpectralMaxwellPlan(
        bridge,
        grid="staggered",
        absorber="psatd-pml",
        pml=SpectralPMLPlan(
            (0, 0, plan.absorber_cells), reflection=plan.absorber_reflection
        ),
        observers=observers,
        antennas=antennas,
    ).prepare(transfers, currents)
    return solver, tuple(plans)


def _recorders(
    plan: FELFullWavePlan,
    beam: FELFullWaveBeam,
    species: tuple[PICSpeciesPlan, ...],
    schedule: _Schedule,
    /,
) -> tuple[PICTrackRecorder, ...]:
    tracks = plan.tracks
    if tracks is None:
        return ()
    count = beam.macroparticle_count
    if max(tracks.lanes) >= count:
        raise ValueError("A recorded lane is not a macroparticle of the beam.")
    lanes = np.asarray(tracks.lanes, dtype=np.int64)
    members = len(species)
    identities = np.concatenate([lanes + index * count for index in range(members)])
    # One pending sample is flushed per step: step_count rows hold every sample.
    return (
        PICTrackRecorder(
            species,
            np.repeat(np.arange(members, dtype=np.int32), lanes.size),
            (
                np.zeros(identities.size, dtype=np.uint32),
                identities.astype(np.uint32),
            ),
            relativity=PIC_CODE_RELATIVITY,
            sample_capacity=schedule.step_count,
        ),
    )


def _refuse_charged_box(
    plan: FELFullWavePlan, beam: FELFullWaveBeam, layout: _Layout, /
) -> None:
    if beam.species != "electron":
        return
    volume = layout.length * plan.transverse_size[0] * plan.transverse_size[1]
    charge = float(plan.lattice.scale.elementary_charge) * float(
        np.sum(np.asarray(beam.electrons))
    )
    if charge / volume > _NEUTRALITY_LIMIT:
        raise ValueError(
            "An electron-only beam leaves a charged periodic box (mean charge density "
            f"{charge / volume:.3e} above {_NEUTRALITY_LIMIT:.0e}); use "
            "species='electron-positron' or test-charge weights."
        )


def _frame_evidence(
    plan: FELFullWavePlan,
    beam: FELFullWaveBeam,
    kinematics: _Kinematics,
    layout: _Layout,
    schedule: _Schedule,
    /,
) -> FELFullWaveFrameEvidence:
    frame = plan.frame
    c = plan.speed_of_light
    weights = np.asarray(beam.electrons)
    mean_gamma = float(np.sum(weights * kinematics.lorentz_factor) / np.sum(weights))
    mean_velocity = float(np.sum(weights * kinematics.mean_velocity) / np.sum(weights))
    beta_z = mean_velocity / c
    boosted_velocity = (beta_z - frame.beta) / (1.0 - beta_z * frame.beta)
    wavenumber = 2.0 * math.pi / plan.boosted_wavelength
    undulator_wavenumber = 2.0 * math.pi / plan.boosted_undulator_period
    nyquist = math.pi / layout.spacing
    form = _sinc(0.5 * wavenumber * layout.spacing) ** (2 * (_SHAPE_ORDER + 1))
    passage = plan.boosted_undulator_period / (frame.beta * c)
    return FELFullWaveFrameEvidence(
        boost_lorentz_factor=frame.lorentz_factor,
        resonant_lorentz_factor=plan.resonant_lorentz_factor(mean_gamma),
        boosted_beam_velocity=boosted_velocity,
        boosted_wavelength=plan.boosted_wavelength,
        boosted_undulator_period=plan.boosted_undulator_period,
        cells_per_wavelength=plan.boosted_wavelength / layout.spacing,
        cells_per_undulator_period=plan.boosted_undulator_period / layout.spacing,
        steps_per_undulator_period=passage / schedule.step_size,
        bunching_resolution=nyquist / (wavenumber + undulator_wavenumber),
        deposit_form_factor=form,
        step_size=schedule.step_size,
        step_count=schedule.step_count,
        grid_shape=(plan.transverse_cells[0], plan.transverse_cells[1], layout.cells),
        start_time=schedule.start,
        stop_time=schedule.stop,
        grid_origin=(*_transverse_lower(plan), layout.lower),
        grid_spacing=(
            plan.transverse_size[0] / plan.transverse_cells[0],
            plan.transverse_size[1] / plan.transverse_cells[1],
            layout.spacing,
        ),
        absorber_cells=plan.absorber_cells,
    )


def _steady_window(
    plan: FELFullWavePlan, beam: FELFullWaveBeam, kinematics: _Kinematics, /
) -> tuple[float, float] | None:
    """Co-moving lab interval ``ξ`` of field elements that crossed flat-top beam only."""
    if beam.flat_top is None:
        return None
    c = plan.speed_of_light
    lower, upper = plan.undulator_span
    velocity = float(np.mean(kinematics.velocity))
    mean = float(np.min(kinematics.mean_velocity))
    tail, head = beam.flat_top

    def entry(position: float, /) -> float:
        return lower - c * (beam.lab_time + (lower - position) / velocity)

    slip = (upper - lower) * (c / mean - 1.0)
    window = (entry(tail), entry(head) - slip)
    return window if window[1] > window[0] else None


# -- runtime ------------------------------------------------------------------------------


def _half_cell_down(values: Array, spacing: float, /, *, axis: int) -> Array:
    """Band-limited values half a cell below their samples along ``axis``.

    The staggered ``B_x``/``B_y`` sit at ``z + h/2``; the phase ``e^{−iqh/2}``
    moves them onto the ``E_x``/``E_y`` nodes (the Nyquist mode, which has no
    unique half-cell value, is dropped).
    """
    count = values.shape[axis]
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    phase = np.exp(-0.5j * wavenumbers * spacing)
    if count % 2 == 0:
        phase[count // 2] = 0.0
    shape = [1] * values.ndim
    shape[axis] = count
    spectrum = jnp.fft.fft(values, axis=axis) * jnp.asarray(phase.reshape(shape))
    return jnp.real(jnp.fft.ifft(spectrum, axis=axis))


@eqx.filter_jit
def _integrate(
    run: PreparedBoostedFrame,
    positions: Array,
    proper_velocities: Array,
    masses: Array,
    start: Array,
    step_size: float,
    step_count: int,
    species_count: int,
) -> tuple[BoostedFrameState, BoostedFrameState, Array, Array]:
    """Run the boosted PIC; also sum the boosted energy and ``P′_z`` the PML
    absorbed and the antenna work over accepted steps."""
    c = run.pic.pusher.speed_of_light
    velocities = (
        proper_velocities
        / jnp.sqrt(1.0 + jnp.sum(proper_velocities**2, axis=-1) / (c * c))[:, None]
    )
    initial = run.initialize(
        tuple(positions for _ in range(species_count)),
        tuple(velocities for _ in range(species_count)),
        step_size,
        time=start,
        masses=tuple(masses for _ in range(species_count)),
    )

    def body(
        carry: tuple[BoostedFrameState, Array], _: None
    ) -> tuple[tuple[BoostedFrameState, Array], Array]:
        state, exchange = carry
        result = run.step_detailed(state, step_size)
        field = result.pic.diagnostics.field
        step = jnp.stack(
            (
                field.absorbed_energy,
                field.absorbed_momentum[2],
                jnp.sum(field.antenna_work),
            )
        )
        exchange = exchange + jnp.where(result.successful, step, 0.0)
        return (result.accepted_state, exchange), result.successful

    (final, exchange), successful = jax.lax.scan(
        body, (initial, jnp.zeros((3,), dtype=jnp.float64)), None, length=step_count
    )
    return initial, final, successful, exchange


__all__ = [
    "FELFullWaveBeam",
    "FELFullWaveBeamSpecies",
    "FELFullWaveEvidence",
    "FELFullWaveFrameEvidence",
    "FELFullWaveHuygens",
    "FELFullWaveLedger",
    "FELFullWavePlan",
    "FELFullWaveResult",
    "FELFullWaveSeed",
    "FELFullWaveStatus",
    "FELFullWaveTracks",
    "PreparedFELFullWave",
]
