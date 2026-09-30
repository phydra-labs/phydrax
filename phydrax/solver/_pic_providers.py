#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned external PIC oracles: WarpX, Smilei, and PIConGPU.

The providers run as caller-pinned executables through
:func:`phydrax.run_pinned_command`; nothing is imported into this process and
no provider source is copied. :func:`pic_oracle_case` binds one Phydrax
`ElectromagneticPICPlan`, the initial conditions its ``initialize`` receives
(positions and physical velocities per species), the step, and the plan's
code-unit `ElectromagneticScaleContract` into a `PICOracleCase`; every SI value
a deck carries is converted with ``scale.unit_si_map()``. Each deck is
generated from the case (never from a second physics model), each run is
bounded (explicit timeout, step, cell, particle, and snapshot counts), and the
provider's ``E``/``B`` snapshots are imported back in the scale's units with
an `AdapterReport` whose losses enumerate what the provider changes.

Scenarios (`PICOracleScenario`) follow the plan's field solver:

- ``"periodic-psatd-plasma"``: a `PreparedSpectralMaxwell` solver (standard,
  Galilean, or averaged-Galilean; constant-J; global FFT; no absorber or
  observers), e.g. the numerical Cherenkov instability of drifting plasma.
- ``"periodic-yee-plasma"``: a `CochainMaxwellPICFieldSolver` (vacuum Yee
  cochain Maxwell with the spline-Whitney Esirkepov current), e.g. cold-plasma
  oscillations.
- ``"single-particle-radiation"``: the Yee case of a plan with one
  `PICTrackRecorder` following the only particle of its species in a uniform,
  static external field; WarpX returns the track as a `ChargedTrajectory`
  (`run_warpx_track`). Smilei and PIConGPU refuse it.
- ``"laser-wakefield-stage"``: a quasi-cylindrical PSATD plan and its focused
  Gaussian pulse (`pic_oracle_wakefield_case`); WarpX RZ launches the pulse
  from an antenna and returns ``thetaMode`` snapshots (`run_warpx_wakefield`,
  pinned ``warpx.rz``: ``PHYDRAX_WARPX_RZ``). Smilei and PIConGPU refuse it.

Common subset, refused with ``ValueError`` before any run: a fully periodic
uniform 3-D box; no PIC processes, particle boundaries, filters, or external
fields; one shape order shared by all species; every macroparticle active and
inside the box; species identifiers that are plain identifiers; ``steps`` a
multiple of ``output_interval``; at most ``2**21`` cells, ``2**18``
macroparticles, ``10**5`` steps, and ``512`` snapshots.

Providers:

- WarpX (Fedeli et al., SC22, doi:10.1109/SC41404.2022.00008;
  BSD-3-Clause-LBNL; https://github.com/BLAST-WarpX/warpx), validated live at
  26.01 (conda-forge ``warpx.3d``, double precision, FFTW). A native
  ``inputs`` deck runs on the pinned ``warpx.3d`` binary; particles use
  ``MultipleParticles`` injection; openPMD HDF5 snapshots are imported with
  `read_openpmd_meshes_hdf5`. Both scenarios; every axis needs more than
  ``shape_order + 3`` cells (WarpX guard cells). Environment:
  ``PHYDRAX_WARPX`` and ``PHYDRAX_WARPX_VERSION``.
- Smilei (Derouillat et al., Comput. Phys. Commun. 222, 351 (2018);
  CECILL-B; https://github.com/SmileiPIC/Smilei), release 5.1. A Python
  namelist with ``numpy`` particle arrays runs on the pinned ``smilei``
  binary; its ``Fields0.h5`` (openPMD 1.0.0-style scalar field records, not
  the 1.1.0 ``E``/``B`` mesh records the openPMD readers own) is imported by
  the bounded :func:`read_smilei_fields`. ``"periodic-yee-plasma"`` only
  (Smilei has no Cartesian PSATD) with ``shape_order == 2`` (Smilei order 2).
  Environment: ``PHYDRAX_SMILEI`` and ``PHYDRAX_SMILEI_VERSION``.
- PIConGPU (Bussmann et al., SC'13, doi:10.1145/2503210.2504564;
  GPL-3.0-or-later, external oracle only;
  https://github.com/ComputationalRadiationPhysics/picongpu), release 0.8.0.
  PIConGPU compiles one binary per setup: `picongpu_input` generates the
  setup's ``.param`` overrides, a pinned per-setup build driver compiles and
  installs ``setup/bin/picongpu``, that binary is pinned by digest and run
  serially, and its openPMD HDF5 snapshots are imported with
  `read_openpmd_meshes_hdf5`. ``"periodic-yee-plasma"`` only, with every
  species a one-per-cell lattice at one in-cell offset, one proper velocity,
  and one weight, and cell counts multiples of the default supercell
  ``(8, 8, 4)`` spanning at least three supercells per axis. Environment: ``PHYDRAX_PICONGPU`` (build driver) and
  ``PHYDRAX_PICONGPU_VERSION``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import assert_never, Literal, TypeAlias

import h5py
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._external_resource import (
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
)
from .._external_runtime import (
    pin_executable,
    PinnedExecutable,
    PinnedFileArtifact,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._validation import positive_finite_float, positive_integer
from ..discretization._tensor_entities import StructuredAxis
from ..discretization.pic import (
    PICShapeOrder,
    PICTrackRecorder,
    QuasiCylindricalGrid,
    RelativisticPusher,
)
from ..electromagnetics._trajectory_radiation import ChargedTrajectory
from ..interchange._openpmd_base import (
    numeric_attribute,
    preflight_hdf5,
    required_text,
    scalar_attribute,
)
from ..interchange._openpmd_mesh import (
    OpenPMDMeshImportPolicy,
    OpenPMDMeshRecord,
    read_openpmd_meshes_hdf5,
)
from ..interchange._openpmd_particles import (
    OpenPMDParticleTrackImportPolicy,
    OpenPMDParticleTrackSelection,
    read_openpmd_particle_tracks_hdf5,
)
from ..interchange._report import AdapterLoss, AdapterReport, AdapterStatus
from ._cochain_pic_field import CochainMaxwellPICFieldSolver
from ._electromagnetic_pic import ElectromagneticPICPlan
from ._maxwell import PreparedDiagonalMaxwellConstitutive
from ._pic_current_source import PreparedPICMaxwellCurrentSource
from .maxwell.spectral._psatd import (
    PreparedSpectralMaxwell,
    SpectralChargeConservation,
    SpectralGrid,
    SpectralMaxwellVariant,
)
from .maxwell.spectral._quasi_cylindrical import PreparedQuasiCylindricalMaxwell


PICOracleScenario: TypeAlias = Literal[
    "periodic-psatd-plasma",
    "periodic-yee-plasma",
    "single-particle-radiation",
    "laser-wakefield-stage",
]
PICOracleCode: TypeAlias = Literal["warpx", "smilei", "picongpu"]

_MAXIMUM_CELLS = 2**21
_MAXIMUM_PARTICLES = 2**18
_MAXIMUM_STEPS = 100_000
_MAXIMUM_SNAPSHOTS = 512
# Relative agreement required between the plan and its declared scale.
_SCALE_TOLERANCE = 1.0e-12
# Relative agreement of provider grid metadata with the case (SI round trips).
_GRID_TOLERANCE = 1.0e-9
# HDF5 structure allowance per snapshot beyond its float64 payload.
_SNAPSHOT_SLACK_BYTES = 1 << 20
_SNAPSHOT_NODES = 4096
_SNAPSHOT_ATTRIBUTES = 65536
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$", re.ASCII)
_OPENPMD_RECORDS: tuple[Literal["E", "B"], Literal["E", "B"]] = ("E", "B")
_WARPX_PREFIX = "diags/fields"
# E and B components of one snapshot.
_FIELD_COMPONENTS = 6
_WARPX_TRACK_PREFIX = "diags/track"
_WARPX_TRACK = f"{_WARPX_TRACK_PREFIX}/openpmd.h5"
# Group-based track series: HDF5 structure per iteration of one small species.
_TRACK_ITERATION_BYTES = 1 << 16
_TRACK_ITERATION_NODES = 64
_TRACK_ITERATION_ATTRIBUTES = 256
# Relative uniformity required of a probed external field.
_EXTERNAL_FIELD_TOLERANCE = 1.0e-12
# WarpX RZ PSATD: axial stencil order psatd.noz and the guard cells it allocates
# (noz axially, 16 radially; each must stay below the cell count), and the
# antenna pre-roll in laser durations (exp(−16) of the peak at t = 0).
_WARPX_RZ_STENCIL = 32
_WARPX_RZ_RADIAL_GUARDS = 16
_WARPX_PREROLL_DURATIONS = 4.0
# E and B components (r, θ, z) of one thetaMode snapshot per mode plane.
_MODAL_COMPONENTS = 6
_SMILEI_OUTPUT = "Fields0.h5"
_SMILEI_ELECTRIC = ("Ex", "Ey", "Ez")
_SMILEI_MAGNETIC = ("Bx_m", "By_m", "Bz_m")
_PICONGPU_SETUP = "setup"
_PICONGPU_BINARY = "setup/bin/picongpu"
_PICONGPU_PARAMS = "setup/include/picongpu/param"
_PICONGPU_SUPERCELL = (8, 8, 4)
_PICONGPU_SHAPES: dict[int, str] = {1: "CIC", 2: "TSC", 3: "PQS"}
_PICONGPU_LICENSE = "GPL-3.0-or-later"
_PICONGPU_BINARY_BYTES = 1 << 30


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleSpecies:
    """One species of a case in the case scale's units.

    ``particle_charge``/``particle_mass`` describe one real particle;
    ``weights[N]`` count real particles per macroparticle; ``positions[N, 3]``
    and ``proper_velocities[N, 3]`` (``γv``) are the initial conditions.
    """

    species_id: str
    particle_charge: float
    particle_mass: float
    positions: np.ndarray
    proper_velocities: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class PICOracleSpectralSolver:
    """PSATD options of a ``"periodic-psatd-plasma"`` case.

    ``stencil_order`` is ``None`` for the infinite-order stencil;
    ``galilean_velocity`` is in the scale velocity unit.
    """

    variant: SpectralMaxwellVariant
    charge_conservation: SpectralChargeConservation
    stencil_order: int | None
    grid: SpectralGrid
    galilean_velocity: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PICOracleTrack:
    """Tracked particle and external field of a ``"single-particle-radiation"`` case.

    ``species`` indexes `PICOracleCase.species`; that species holds exactly the
    tracked macroparticle, whose recorder identity is ``(id_hi, id_lo)``.
    ``electric``/``magnetic`` are the plan's uniform, static external field
    in the scale's units.
    """

    species: int
    identity: tuple[int, int]
    electric: tuple[float, float, float]
    magnetic: tuple[float, float, float]


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleCase:
    """One Phydrax PIC run bound for an external oracle, in the scale's units.

    The box spans ``lower + counts * spacing``; snapshots are taken at
    iterations ``0, output_interval, …, steps``. ``spectral`` is set exactly
    for ``"periodic-psatd-plasma"`` and ``track`` exactly for
    ``"single-particle-radiation"``; ``"laser-wakefield-stage"`` is a
    `PICOracleWakefieldCase`.
    """

    scenario: PICOracleScenario
    scale: ElectromagneticScaleContract
    counts: tuple[int, int, int]
    spacing: tuple[float, float, float]
    lower: tuple[float, float, float]
    shape_order: PICShapeOrder
    pusher: RelativisticPusher
    spectral: PICOracleSpectralSolver | None
    species: tuple[PICOracleSpecies, ...]
    track: PICOracleTrack | None
    step_size: float
    steps: int
    output_interval: int
    plan_id: str
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.scenario == "laser-wakefield-stage" or (
            (self.scenario == "single-particle-radiation") != (self.track is not None)
        ):
            raise ValueError("The case scenario disagrees with its track.")
        spectral = self.spectral
        track = self.track
        object.__setattr__(
            self,
            "case_id",
            canonical_fingerprint(
                {
                    "kind": "pic-oracle-case",
                    "scenario": self.scenario,
                    "scale": self.scale.scale_id,
                    "counts": list(self.counts),
                    "spacing": list(self.spacing),
                    "lower": list(self.lower),
                    "shape_order": self.shape_order,
                    "pusher": self.pusher,
                    "spectral": None
                    if spectral is None
                    else [
                        spectral.variant,
                        spectral.charge_conservation,
                        spectral.stencil_order,
                        spectral.grid,
                        list(spectral.galilean_velocity),
                    ],
                    "species": [
                        [
                            value.species_id,
                            value.particle_charge,
                            value.particle_mass,
                            array_tree_fingerprint(
                                (value.positions, value.proper_velocities, value.weights)
                            ),
                        ]
                        for value in self.species
                    ],
                    "track": None
                    if track is None
                    else [
                        track.species,
                        list(track.identity),
                        list(track.electric),
                        list(track.magnetic),
                    ],
                    "step_size": self.step_size,
                    "steps": self.steps,
                    "output_interval": self.output_interval,
                    "plan": self.plan_id,
                }
            ),
        )

    @property
    def output_iterations(self) -> tuple[int, ...]:
        return tuple(range(0, self.steps + 1, self.output_interval))

    @property
    def cell_count(self) -> int:
        return self.counts[0] * self.counts[1] * self.counts[2]


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleFields:
    """``E``/``B`` snapshots of one oracle run in the case scale's units.

    ``electric[t, i, j, k, c]`` is component ``c`` sampled in cell ``(i, j, k)``
    at in-cell position ``electric_positions[c]`` (cell fractions per axis) and
    time ``times[t] + electric_time_offsets[c]``; likewise ``magnetic``.
    """

    iterations: tuple[int, ...]
    times: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray
    electric_positions: tuple[tuple[float, float, float], ...]
    magnetic_positions: tuple[tuple[float, float, float], ...]
    electric_time_offsets: tuple[float, float, float]
    magnetic_time_offsets: tuple[float, float, float]
    source_format: str


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleResult:
    """Imported snapshots of one pinned oracle run with provenance and losses.

    ``output_sha256`` digests the sorted ``path sha256`` lines of the published
    provider artifacts; ``report.source_id`` equals it and ``report.target_id``
    fingerprints the imported snapshots.
    """

    provider: PICOracleCode
    case_id: str
    fields: PICOracleFields
    provider_version: str
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


@dataclass(frozen=True, slots=True)
class PICOracleLaser:
    """Gaussian laser pulse of a ``"laser-wakefield-stage"`` case, in scale units.

    At ``t = 0`` the pulse is at focus, polarized along ``x``, and travels
    along ``+z``: ``E_x = amplitude exp(−r²/waist² − (z − center)²/length²)
    cos(2π(z − center)/wavelength)`` (``amplitude = a₀ m_e c ω₀/e``).
    """

    amplitude: float
    wavelength: float
    waist: float
    length: float
    center: float

    def __post_init__(self) -> None:
        center = float(self.center)
        if not math.isfinite(center):
            raise ValueError("center must be finite.")
        values = (
            ("amplitude", positive_finite_float(self.amplitude, "amplitude")),
            ("wavelength", positive_finite_float(self.wavelength, "wavelength")),
            ("waist", positive_finite_float(self.waist, "waist")),
            ("length", positive_finite_float(self.length, "length")),
            ("center", center),
        )
        for name, value in values:
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleWakefieldCase:
    """One quasi-cylindrical laser-wakefield PIC run bound for an oracle.

    The grid is Phydrax's `QuasiCylindricalGrid`: radial nodes
    ``(j + ½) radius/radial_count``, periodic axial nodes ``lower + kΔz``, and
    azimuthal modes ``0 … mode_count − 1``; snapshots are taken at iterations
    ``0, output_interval, …, steps``; everything is in the scale's units.
    """

    scale: ElectromagneticScaleContract
    radius: float
    radial_count: int
    lower: float
    axial_spacing: float
    axial_count: int
    mode_count: int
    charge_conservation: SpectralChargeConservation
    shape_order: PICShapeOrder
    pusher: RelativisticPusher
    laser: PICOracleLaser
    species: tuple[PICOracleSpecies, ...]
    step_size: float
    steps: int
    output_interval: int
    plan_id: str
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        laser = self.laser
        object.__setattr__(
            self,
            "case_id",
            canonical_fingerprint(
                {
                    "kind": "pic-oracle-wakefield-case",
                    "scale": self.scale.scale_id,
                    "grid": [
                        self.radius,
                        self.radial_count,
                        self.lower,
                        self.axial_spacing,
                        self.axial_count,
                        self.mode_count,
                    ],
                    "charge_conservation": self.charge_conservation,
                    "shape_order": self.shape_order,
                    "pusher": self.pusher,
                    "laser": [
                        laser.amplitude,
                        laser.wavelength,
                        laser.waist,
                        laser.length,
                        laser.center,
                    ],
                    "species": [
                        [
                            value.species_id,
                            value.particle_charge,
                            value.particle_mass,
                            array_tree_fingerprint(
                                (value.positions, value.proper_velocities, value.weights)
                            ),
                        ]
                        for value in self.species
                    ],
                    "step_size": self.step_size,
                    "steps": self.steps,
                    "output_interval": self.output_interval,
                    "plan": self.plan_id,
                }
            ),
        )

    @property
    def scenario(self) -> PICOracleScenario:
        return "laser-wakefield-stage"

    @property
    def output_iterations(self) -> tuple[int, ...]:
        return tuple(range(0, self.steps + 1, self.output_interval))

    @property
    def upper(self) -> float:
        return self.lower + self.axial_count * self.axial_spacing


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleModalFields:
    """Azimuthal-mode ``E``/``B`` snapshots of one oracle run in scale units.

    ``electric[t, p, j, k, c]`` is the ``(r, θ, z)`` component ``c`` of mode
    plane ``p`` at radius ``radial_coordinates[j]``, axial position
    ``axial_coordinates[k]``, and time ``times[t]``. Mode planes follow openPMD
    ``thetaMode`` with ``imag=+``: mode 0, then the real and imaginary parts of
    modes ``1 … M − 1``, so ``F(θ) = F_0 + Σ_m (Re F_m cos mθ + Im F_m sin mθ)``;
    likewise ``magnetic``.
    """

    iterations: tuple[int, ...]
    times: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray
    radial_coordinates: np.ndarray
    axial_coordinates: np.ndarray
    source_format: str


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleTrackResult:
    """Imported track of one pinned oracle run with provenance and losses.

    ``track`` is the tracked particle in the case scale's units under its
    Phydrax recorder identity; the provenance fields mean what they mean in
    `PICOracleResult`.
    """

    provider: PICOracleCode
    case_id: str
    track: ChargedTrajectory
    provider_version: str
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


@dataclass(frozen=True, slots=True, eq=False)
class PICOracleWakefieldResult:
    """Imported azimuthal-mode snapshots of one pinned wakefield oracle run.

    Provenance fields mean what they mean in `PICOracleResult`.
    """

    provider: PICOracleCode
    case_id: str
    fields: PICOracleModalFields
    provider_version: str
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


def _require_executable(executable: object, description: str, /) -> PinnedExecutable:
    if not isinstance(executable, PinnedExecutable):
        raise TypeError(f"executable must be a PinnedExecutable {description}.")
    return executable


@dataclass(frozen=True, slots=True)
class WarpXProvider:
    """A pinned ``warpx.3d`` binary (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        _require_executable(self.executable, "warpx.3d binary")


@dataclass(frozen=True, slots=True)
class SmileiProvider:
    """A pinned ``smilei`` binary run serially (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        _require_executable(self.executable, "smilei binary")


@dataclass(frozen=True, slots=True)
class PIConGPUProvider:
    """A pinned PIConGPU per-setup build driver (external oracle only).

    PIConGPU compiles one binary per setup. The driver is invoked as
    ``driver setup [cmake_arguments...]`` in the staging directory; it must
    compile ``setup/include/picongpu/param`` against the PIConGPU release that
    ``executable.version``/``executable.license_id`` declare and install
    ``setup/bin/picongpu``. The compiled binary is then pinned by digest.
    """

    executable: PinnedExecutable
    cmake_arguments: tuple[str, ...] = ()
    build_timeout: float = 7200.0

    def __post_init__(self) -> None:
        _require_executable(self.executable, "PIConGPU build driver")
        arguments = tuple(self.cmake_arguments)
        if any(not isinstance(value, str) for value in arguments):
            raise TypeError("cmake_arguments must be strings.")
        object.__setattr__(self, "cmake_arguments", arguments)
        object.__setattr__(
            self,
            "build_timeout",
            positive_finite_float(self.build_timeout, "build_timeout"),
        )


# Case construction ----------------------------------------------------------------


def _close(value: float, reference: float, /) -> bool:
    return math.isclose(value, reference, rel_tol=_SCALE_TOLERANCE, abs_tol=0.0)


def _uniform_axes(
    axes: Sequence[StructuredAxis], /
) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]:
    if len(axes) != 3:
        raise ValueError("PIC oracles need a three-dimensional box.")
    counts: list[int] = []
    spacing: list[float] = []
    lower: list[float] = []
    for axis in axes:
        widths = np.asarray(axis.interval_widths, dtype=np.float64)
        if not axis.periodic:
            raise ValueError("PIC oracles need a fully periodic box.")
        if not np.allclose(widths, widths[0], rtol=1.0e-12, atol=0.0):
            raise ValueError("PIC oracles need uniform cells.")
        counts.append(widths.shape[0])
        spacing.append(float(widths[0]))
        lower.append(float(np.asarray(axis.bounds, dtype=np.float64)[0]))
    return (
        (counts[0], counts[1], counts[2]),
        (spacing[0], spacing[1], spacing[2]),
        (lower[0], lower[1], lower[2]),
    )


def _shape_order(orders: set[int], /) -> PICShapeOrder:
    if len(orders) != 1:
        raise ValueError("PIC oracles need one shape order shared by every species.")
    match orders.pop():
        case 1:
            return 1
        case 2:
            return 2
        case 3:
            return 3
        case other:
            raise ValueError(f"Unsupported PIC shape order {other}.")


@dataclass(frozen=True, slots=True)
class _SolverCase:
    scenario: PICOracleScenario
    counts: tuple[int, int, int]
    spacing: tuple[float, float, float]
    lower: tuple[float, float, float]
    shape_order: PICShapeOrder
    spectral: PICOracleSpectralSolver | None


def _spectral_case(
    solver: PreparedSpectralMaxwell, scale: ElectromagneticScaleContract, /
) -> _SolverCase:
    plan = solver.plan
    if plan.absorber != "none" or plan.pml is not None or plan.observers:
        raise ValueError("PIC oracles refuse spectral absorbers and Huygens observers.")
    if plan.decomposition != "global-fft":
        raise ValueError("PIC oracles need the global-FFT spectral decomposition.")
    if plan.time_dependency != "constant-j":
        raise ValueError("PIC oracles need the constant-J spectral time dependency.")
    if not _close(plan.permittivity, float(scale.vacuum_permittivity)) or not _close(
        plan.permeability, float(scale.vacuum_permeability)
    ):
        raise ValueError("The spectral medium must be the scale's vacuum.")
    return _SolverCase(
        "periodic-psatd-plasma",
        plan.counts,
        plan.spacing,
        plan.origin,
        _shape_order({value.plan.shape_order for value in solver.transfers}),
        PICOracleSpectralSolver(
            plan.variant,
            plan.charge_conservation,
            None if plan.stencil == "infinite-order" else plan.stencil_order,
            plan.grid,
            plan.galilean_velocity,
        ),
    )


def _yee_case(
    solver: CochainMaxwellPICFieldSolver, scale: ElectromagneticScaleContract, /
) -> _SolverCase:
    maxwell = solver.maxwell
    if maxwell.pml is not None or maxwell.boundaries or maxwell.observers:
        raise ValueError("PIC oracles refuse Maxwell CPML, boundaries, and observers.")
    if len(maxwell.sources) != 1 or not isinstance(
        maxwell.sources[0], PreparedPICMaxwellCurrentSource
    ):
        raise ValueError("PIC oracles need the PIC current as the only Maxwell source.")
    medium = maxwell.constitutive
    if not isinstance(medium, PreparedDiagonalMaxwellConstitutive):
        raise ValueError("PIC oracles need a diagonal vacuum Maxwell medium.")
    epsilon = np.asarray(medium.permittivity, dtype=np.float64)
    mu = np.asarray(medium.permeability, dtype=np.float64)
    if not np.allclose(
        epsilon, float(scale.vacuum_permittivity), rtol=_SCALE_TOLERANCE, atol=0.0
    ) or not np.allclose(
        mu, float(scale.vacuum_permeability), rtol=_SCALE_TOLERANCE, atol=0.0
    ):
        raise ValueError("The Maxwell medium must be the scale's vacuum.")
    counts, spacing, lower = _uniform_axes(solver.bridge.grid.structured_axes)
    return _SolverCase(
        "periodic-yee-plasma",
        counts,
        spacing,
        lower,
        _shape_order({value.plan.shape_order for value in solver.transfers}),
        None,
    )


def _box(solved: _SolverCase, /) -> Callable[[np.ndarray], np.ndarray]:
    lower = np.asarray(solved.lower, dtype=np.float64)
    upper = lower + np.asarray(solved.counts) * np.asarray(solved.spacing)

    def inside(points: np.ndarray) -> np.ndarray:
        return np.all((points >= lower) & (points < upper), axis=-1)

    return inside


def _species(
    plan: ElectromagneticPICPlan,
    inside: Callable[[np.ndarray], np.ndarray],
    speed_of_light: float,
    positions: Sequence[ArrayLike],
    velocities: Sequence[ArrayLike],
    masses: Sequence[ArrayLike | None],
    particle_masses: Sequence[float],
    /,
) -> tuple[PICOracleSpecies, ...]:
    result: list[PICOracleSpecies] = []
    for species, position, velocity, mass, particle_mass in zip(
        plan.species, positions, velocities, masses, particle_masses, strict=True
    ):
        if not _IDENTIFIER.match(species.species_id):
            raise ValueError("PIC oracle species identifiers must be plain identifiers.")
        population = species.population.initialize(masses=mass)
        if not np.all(np.asarray(population.active, dtype=np.bool_)):
            raise ValueError("PIC oracles need every macroparticle slot active.")
        points = np.array(position, dtype=np.float64)
        speed = np.array(velocity, dtype=np.float64)
        if points.shape != (species.capacity, 3) or speed.shape != (species.capacity, 3):
            raise ValueError("Positions and velocities must be capacity-by-three.")
        if not np.all(np.isfinite(points)) or not np.all(inside(points)):
            raise ValueError(
                "Initial particles must lie inside the periodic box or cylinder."
            )
        beta2 = np.sum(speed * speed, axis=-1) / speed_of_light**2
        if not np.all(np.isfinite(beta2)) or np.any(beta2 >= 1.0):
            raise ValueError("Initial velocities must be finite and subluminal.")
        real_mass = positive_finite_float(particle_mass, "particle_masses")
        model = species.charge_model
        points.flags.writeable = False
        proper = speed / np.sqrt(1.0 - beta2)[:, None]
        proper.flags.writeable = False
        weights = np.asarray(population.mass, dtype=np.float64) / real_mass
        weights.flags.writeable = False
        result.append(
            PICOracleSpecies(
                species.species_id,
                model.base_specific_charge * model.initial_charge_number * real_mass,
                real_mass,
                points,
                proper,
                weights,
            )
        )
    return tuple(result)


def _bound_species(
    plan: ElectromagneticPICPlan,
    inside: Callable[[np.ndarray], np.ndarray],
    positions: Sequence[ArrayLike],
    velocities: Sequence[ArrayLike],
    masses: Sequence[ArrayLike | None] | None,
    particle_masses: Sequence[float],
    /,
) -> tuple[PICOracleSpecies, ...]:
    count = len(plan.species)
    position_values = tuple(positions)
    velocity_values = tuple(velocities)
    mass_values = (None,) * count if masses is None else tuple(masses)
    real_masses = tuple(particle_masses)
    if not (
        len(position_values)
        == len(velocity_values)
        == len(mass_values)
        == len(real_masses)
        == count
    ):
        raise ValueError("One position, velocity, mass, and particle mass per species.")
    species = _species(
        plan,
        inside,
        float(plan.pusher.speed_of_light),
        position_values,
        velocity_values,
        mass_values,
        real_masses,
    )
    if sum(value.weights.shape[0] for value in species) > _MAXIMUM_PARTICLES:
        raise ValueError(f"PIC oracles are bounded to {_MAXIMUM_PARTICLES} particles.")
    return species


def _schedule(
    step_size: float, steps: int, output_interval: int, /
) -> tuple[float, int, int]:
    step = positive_finite_float(step_size, "step_size")
    total = positive_integer(steps, "steps")
    interval = positive_integer(output_interval, "output_interval")
    if total > _MAXIMUM_STEPS or total % interval != 0:
        raise ValueError(
            f"steps must be at most {_MAXIMUM_STEPS} and a multiple of output_interval."
        )
    if total // interval + 1 > _MAXIMUM_SNAPSHOTS:
        raise ValueError(f"PIC oracles are bounded to {_MAXIMUM_SNAPSHOTS} snapshots.")
    return step, total, interval


def _vector(values: np.ndarray, /) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _uniform_external_field(
    plan: ElectromagneticPICPlan,
    solved: _SolverCase,
    species: tuple[PICOracleSpecies, ...],
    duration: float,
    /,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The plan's summed external field, probed at every cell center and
    initial particle position at ``t = 0``, ``T/2``, and ``T``; it must be
    supported everywhere and uniform and static to ``1e-12`` relative."""
    lower = np.asarray(solved.lower, dtype=np.float64)
    spacing = np.asarray(solved.spacing, dtype=np.float64)
    axes = [
        lower[axis]
        + (np.arange(solved.counts[axis], dtype=np.float64) + 0.5) * spacing[axis]
        for axis in range(3)
    ]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = jnp.asarray(
        np.concatenate((centers, *(value.positions for value in species))),
        dtype=jnp.float64,
    )
    electric = np.zeros((3, points.shape[0], 3), dtype=np.float64)
    magnetic = np.zeros((3, points.shape[0], 3), dtype=np.float64)
    for index, time in enumerate((0.0, 0.5 * duration, duration)):
        times = jnp.full((points.shape[0],), time, dtype=jnp.float64)
        for source in plan.external_fields:
            sample = source.external_fields(points, times)
            if not np.all(np.asarray(sample.support, dtype=np.bool_)):
                raise ValueError("The external field must be supported in the whole box.")
            electric[index] += np.asarray(sample.electric, dtype=np.float64)
            magnetic[index] += np.asarray(sample.magnetic, dtype=np.float64)
    for values in (electric, magnetic):
        size = max(float(np.max(np.abs(values))), float(np.finfo(np.float64).tiny))
        if np.max(np.abs(values - values[0, 0])) > _EXTERNAL_FIELD_TOLERANCE * size:
            raise ValueError(
                "The single-particle-radiation scenario needs a uniform, static "
                "external field."
            )
    return _vector(electric[0, 0]), _vector(magnetic[0, 0])


def _track(
    plan: ElectromagneticPICPlan,
    solved: _SolverCase,
    species: tuple[PICOracleSpecies, ...],
    duration: float,
    /,
) -> PICOracleTrack:
    recorders = plan.recorders
    if len(recorders) != 1 or not isinstance(recorders[0], PICTrackRecorder):
        raise ValueError(
            "The single-particle-radiation scenario needs exactly one PICTrackRecorder."
        )
    recorder = recorders[0]
    if recorder.lane_count != 1:
        raise ValueError("The oracle's track recorder must follow exactly one identity.")
    index = int(np.asarray(recorder.lane_species)[0])
    tracked = plan.species[index]
    if tracked.capacity != 1:
        raise ValueError("The tracked species must hold only the tracked macroparticle.")
    population = tracked.population.initialize()
    identity = (int(np.asarray(recorder.id_hi)[0]), int(np.asarray(recorder.id_lo)[0]))
    if (
        int(np.asarray(population.id_hi)[0]),
        int(np.asarray(population.id_lo)[0]),
    ) != identity:
        raise ValueError("The recorder does not track the tracked species' particle.")
    electric, magnetic = _uniform_external_field(plan, solved, species, duration)
    return PICOracleTrack(index, identity, electric, magnetic)


def pic_oracle_case(
    plan: ElectromagneticPICPlan,
    scale: ElectromagneticScaleContract,
    positions: Sequence[ArrayLike],
    velocities: Sequence[ArrayLike],
    step_size: float,
    /,
    *,
    particle_masses: Sequence[float],
    steps: int,
    output_interval: int,
    masses: Sequence[ArrayLike | None] | None = None,
) -> PICOracleCase:
    """Bind one PIC plan and the arguments of its ``initialize`` for an oracle.

    ``positions``/``velocities``/``masses`` are exactly what
    `ElectromagneticPICPlan.initialize` receives (physical velocities, optional
    macroparticle masses); ``particle_masses[s]`` is the rest mass of one real
    particle of species ``s`` in ``scale`` units, which fixes each provider's
    particle charge, mass, and weight. ``scale`` declares the plan's code
    units: its speed of light must equal the pusher's and its vacuum
    permittivity and permeability the field solver's. A plan with recorders or
    external fields is the ``"single-particle-radiation"`` scenario: a Yee
    cochain solver, one `PICTrackRecorder` following one identity that is the
    only macroparticle of its species, and external fields summing to a
    uniform, static field (probed at every cell center and initial particle
    position at the start, middle, and end of the run).
    """
    if not isinstance(plan, ElectromagneticPICPlan):
        raise TypeError("plan must be an ElectromagneticPICPlan.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    if plan.processes or plan.boundaries is not None or plan.filters:
        raise ValueError(
            "PIC oracles refuse processes, particle boundaries, and filters."
        )
    speed = float(plan.pusher.speed_of_light)
    if not _close(speed, float(scale.speed_of_light)):
        raise ValueError("The scale's speed of light differs from the PIC pusher's.")
    scale.unit_si_map()
    solver = plan.solver
    tracked = bool(plan.recorders or plan.external_fields)
    if isinstance(solver, PreparedSpectralMaxwell) and not tracked:
        solved = _spectral_case(solver, scale)
    elif isinstance(solver, CochainMaxwellPICFieldSolver):
        solved = _yee_case(solver, scale)
    elif tracked:
        raise ValueError(
            "The single-particle-radiation scenario needs the cochain Yee solver."
        )
    else:
        raise ValueError(
            "PIC oracles support the spectral PSATD and cochain Yee field solvers."
        )
    if solved.counts[0] * solved.counts[1] * solved.counts[2] > _MAXIMUM_CELLS:
        raise ValueError(f"PIC oracles are bounded to {_MAXIMUM_CELLS} cells.")
    species = _bound_species(
        plan, _box(solved), positions, velocities, masses, particle_masses
    )
    step, total, interval = _schedule(step_size, steps, output_interval)
    track = _track(plan, solved, species, step * total) if tracked else None
    return PICOracleCase(
        "single-particle-radiation" if tracked else solved.scenario,
        scale,
        solved.counts,
        solved.spacing,
        solved.lower,
        solved.shape_order,
        plan.pusher.method,
        solved.spectral,
        species,
        track,
        step,
        total,
        interval,
        plan.plan_id,
    )


def _cylinder(grid: QuasiCylindricalGrid, /) -> Callable[[np.ndarray], np.ndarray]:
    def inside(points: np.ndarray) -> np.ndarray:
        radius = np.hypot(points[:, 0], points[:, 1])
        return (
            (radius < grid.radius)
            & (points[:, 2] >= grid.lower)
            & (points[:, 2] < grid.upper)
        )

    return inside


def pic_oracle_wakefield_case(
    plan: ElectromagneticPICPlan,
    scale: ElectromagneticScaleContract,
    positions: Sequence[ArrayLike],
    velocities: Sequence[ArrayLike],
    step_size: float,
    /,
    *,
    laser: PICOracleLaser,
    particle_masses: Sequence[float],
    steps: int,
    output_interval: int,
    masses: Sequence[ArrayLike | None] | None = None,
) -> PICOracleWakefieldCase:
    """Bind one quasi-cylindrical laser-wakefield PIC run for an oracle.

    ``plan`` runs a `PreparedQuasiCylindricalMaxwell` solver (standard PSATD,
    spectral charge conservation, no absorber, antennas, or observers, in the
    scale's vacuum) without processes, particle boundaries, filters, recorders,
    or external fields; ``positions``/``velocities``/``masses`` and
    ``particle_masses`` are as in `pic_oracle_case`, with every particle inside
    the cylinder. ``laser`` is the pulse the run adds to its initial field
    (e.g. with ``add_propagating_field``); it must lie in the axial box, and
    the grid needs at least two azimuthal modes to carry it.
    """
    if not isinstance(plan, ElectromagneticPICPlan):
        raise TypeError("plan must be an ElectromagneticPICPlan.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    if not isinstance(laser, PICOracleLaser):
        raise TypeError("laser must be a PICOracleLaser.")
    if (
        plan.processes
        or plan.boundaries is not None
        or plan.filters
        or plan.recorders
        or plan.external_fields
    ):
        raise ValueError(
            "The laser-wakefield-stage scenario refuses processes, particle "
            "boundaries, filters, recorders, and external fields."
        )
    solver = plan.solver
    if not isinstance(solver, PreparedQuasiCylindricalMaxwell):
        raise ValueError(
            "The laser-wakefield-stage scenario needs the quasi-cylindrical PSATD solver."
        )
    declared = solver.plan
    if declared.variant != "standard":
        raise ValueError("The laser-wakefield-stage oracle covers standard PSATD only.")
    if declared.absorber != "none" or declared.antennas or declared.observers:
        raise ValueError(
            "The laser-wakefield-stage oracle refuses radial damping, antennas, and "
            "Huygens observers."
        )
    speed = float(plan.pusher.speed_of_light)
    if not _close(speed, float(scale.speed_of_light)):
        raise ValueError("The scale's speed of light differs from the PIC pusher's.")
    if not _close(declared.permittivity, float(scale.vacuum_permittivity)) or not _close(
        declared.permeability, float(scale.vacuum_permeability)
    ):
        raise ValueError("The quasi-cylindrical medium must be the scale's vacuum.")
    scale.unit_si_map()
    grid = declared.grid
    if grid.mode_count < 2:
        raise ValueError("A linearly polarized laser needs azimuthal modes 0 and 1.")
    if grid.radial_count * grid.axial_count > _MAXIMUM_CELLS:
        raise ValueError(f"PIC oracles are bounded to {_MAXIMUM_CELLS} cells.")
    if not grid.lower <= laser.center < grid.upper:
        raise ValueError("The laser center must lie in the axial box.")
    species = _bound_species(
        plan, _cylinder(grid), positions, velocities, masses, particle_masses
    )
    step, total, interval = _schedule(step_size, steps, output_interval)
    return PICOracleWakefieldCase(
        scale,
        grid.radius,
        grid.radial_count,
        grid.lower,
        grid.axial_spacing,
        grid.axial_count,
        grid.mode_count,
        declared.charge_conservation,
        _shape_order({value.plan.shape_order for value in solver.transfers}),
        plan.pusher.method,
        laser,
        species,
        step,
        total,
        interval,
        plan.plan_id,
    )


# Shared unit conversion -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SI:
    """SI values of one case, every factor from ``scale.unit_si_map()``."""

    length: float
    time: float
    charge: float
    mass: float
    velocity: float
    electric_field: float
    magnetic_field: float

    @classmethod
    def of(cls, case: PICOracleCase | PICOracleWakefieldCase, /) -> _SI:
        units = case.scale.unit_si_map()
        return cls(
            units["length"][0],
            units["time"][0],
            units["charge"][0],
            units["mass"][0],
            units["velocity"][0],
            units["electric_field"][0],
            units["magnetic_field"][0],
        )


def _number(value: float, /) -> str:
    return repr(float(value))


def _numbers(values: np.ndarray, /) -> str:
    return " ".join(repr(float(value)) for value in values.reshape(-1))


def _require_case(case: object, /) -> PICOracleCase:
    if not isinstance(case, PICOracleCase):
        raise TypeError("case must be a PICOracleCase.")
    return case


def _require_any_case(case: object, /) -> PICOracleCase | PICOracleWakefieldCase:
    if not isinstance(case, (PICOracleCase, PICOracleWakefieldCase)):
        raise TypeError("case must be a PICOracleCase or PICOracleWakefieldCase.")
    return case


def _pusher_name(method: RelativisticPusher, code: PICOracleCode, /) -> str:
    match code, method:
        case ("warpx", "boris") | ("smilei", "boris"):
            return "boris"
        case ("warpx", "vay") | ("smilei", "vay"):
            return "vay"
        case "warpx", "higuera-cary":
            return "higuera"
        case "smilei", "higuera-cary":
            return "higueracary"
        case "picongpu", "boris":
            return "Boris"
        case "picongpu", "vay":
            return "Vay"
        case "picongpu", "higuera-cary":
            return "HigueraCary"
        case _:
            raise ValueError(f"Unknown pusher {method!r} for {code!r}.")


# WarpX ------------------------------------------------------------------------------


def _warpx_solver_lines(case: PICOracleCase, /) -> list[str]:
    match case.scenario:
        case "periodic-psatd-plasma":
            spectral = case.spectral
            if spectral is None:
                raise ValueError("A PSATD case must carry its spectral options.")
            order = (
                "inf" if spectral.stencil_order is None else str(spectral.stencil_order)
            )
            collocated = spectral.grid == "collocated"
            match spectral.charge_conservation:
                case "spectral-correction":
                    deposition = "direct" if collocated else "esirkepov"
                    correction, with_rho = 1, 0
                case "vay-deposition":
                    raise ValueError(
                        "WarpX Vay deposition needs local FFTs; the case declares the "
                        "global-FFT PSATD."
                    )
                case "update-with-rho":
                    deposition = "direct" if collocated else "esirkepov"
                    correction, with_rho = 0, 1
                case unreachable:
                    assert_never(unreachable)
            beta = np.asarray(spectral.galilean_velocity) / float(
                case.scale.speed_of_light
            )
            lines = [
                "algo.maxwell_solver = psatd",
                f"warpx.grid_type = {spectral.grid}",
                "psatd.periodic_single_box_fft = 1",
                f"psatd.nox = {order}",
                f"psatd.noy = {order}",
                f"psatd.noz = {order}",
                f"algo.current_deposition = {deposition}",
                f"psatd.current_correction = {correction}",
                f"psatd.update_with_rho = {with_rho}",
                f"psatd.v_galilean = {_numbers(beta)}",
                f"interpolation.galerkin_scheme = {0 if collocated else 1}",
            ]
            match spectral.variant:
                case "standard" | "galilean":
                    return lines
                case "averaged-galilean":
                    return [*lines, "psatd.do_time_averaging = 1"]
                case unreachable:
                    assert_never(unreachable)
        case "periodic-yee-plasma" | "single-particle-radiation":
            return [
                "algo.maxwell_solver = yee",
                "warpx.grid_type = staggered",
                "algo.current_deposition = esirkepov",
                "interpolation.galerkin_scheme = 1",
            ]
        case "laser-wakefield-stage":
            raise ValueError("A laser-wakefield-stage case is a PICOracleWakefieldCase.")
        case unreachable:
            assert_never(unreachable)


def _warpx_species_lines(
    species: Sequence[PICOracleSpecies], si: _SI, light: float, /
) -> list[str]:
    lines = [f"particles.species_names = {' '.join(v.species_id for v in species)}"]
    for value in species:
        name = value.species_id
        points = value.positions * si.length
        momentum = value.proper_velocities / light
        lines += [
            f"{name}.charge = {_number(value.particle_charge * si.charge)}",
            f"{name}.mass = {_number(value.particle_mass * si.mass)}",
            f"{name}.injection_style = MultipleParticles",
            f"{name}.multiple_particles_pos_x = {_numbers(points[:, 0])}",
            f"{name}.multiple_particles_pos_y = {_numbers(points[:, 1])}",
            f"{name}.multiple_particles_pos_z = {_numbers(points[:, 2])}",
            f"{name}.multiple_particles_ux = {_numbers(momentum[:, 0])}",
            f"{name}.multiple_particles_uy = {_numbers(momentum[:, 1])}",
            f"{name}.multiple_particles_uz = {_numbers(momentum[:, 2])}",
            f"{name}.multiple_particles_weight = {_numbers(value.weights)}",
        ]
    return lines


def _warpx_output_lines(case: PICOracleCase, si: _SI, /) -> list[str]:
    track = case.track
    if track is None:
        return [
            "diagnostics.diags_names = fields",
            "fields.diag_type = Full",
            "fields.format = openpmd",
            "fields.openpmd_backend = h5",
            "fields.openpmd_encoding = f",
            f"fields.file_prefix = {_WARPX_PREFIX}",
            "fields.file_min_digits = 6",
            f"fields.intervals = {case.output_interval}",
            "fields.fields_to_plot = Ex Ey Ez Bx By Bz",
            "fields.write_species = 0",
        ]
    electric = np.asarray(track.electric) * si.electric_field
    magnetic = np.asarray(track.magnetic) * si.magnetic_field
    return [
        "particles.E_ext_particle_init_style = constant",
        f"particles.E_external_particle = {_numbers(electric)}",
        "particles.B_ext_particle_init_style = constant",
        f"particles.B_external_particle = {_numbers(magnetic)}",
        "diagnostics.diags_names = track",
        "track.diag_type = Full",
        "track.format = openpmd",
        "track.openpmd_backend = h5",
        "track.openpmd_encoding = g",
        f"track.file_prefix = {_WARPX_TRACK_PREFIX}",
        f"track.intervals = {case.output_interval}",
        "track.fields_to_plot = none",
        f"track.species = {case.species[track.species].species_id}",
    ]


def warpx_preroll_steps(case: PICOracleWakefieldCase, /) -> int:
    """Steps the WarpX antenna emits before the case's ``t = 0``.

    The Gaussian antenna sits at the pulse center with peak time
    ``t_peak = n Δt``, ``n = ⌈4 τ/Δt⌉`` and ``τ = length/c``, so WarpX
    iteration ``n + k`` is the case's step ``k``.
    """
    if not isinstance(case, PICOracleWakefieldCase):
        raise TypeError("case must be a PICOracleWakefieldCase.")
    duration = case.laser.length / float(case.scale.speed_of_light)
    return math.ceil(_WARPX_PREROLL_DURATIONS * duration / case.step_size)


def _warpx_rz_input(case: PICOracleWakefieldCase, /) -> dict[str, bytes]:
    if (
        case.radial_count <= _WARPX_RZ_RADIAL_GUARDS
        or case.axial_count <= _WARPX_RZ_STENCIL
    ):
        raise ValueError(
            f"WarpX RZ PSATD needs more than {_WARPX_RZ_RADIAL_GUARDS} radial and "
            f"{_WARPX_RZ_STENCIL} axial cells (its guard cells)."
        )
    shift = 0.5 * case.axial_spacing
    if any(np.any(value.positions[:, 2] >= case.upper - shift) for value in case.species):
        raise ValueError(
            "WarpX's box ends half a cell below the case's upper axial node; "
            "particles must lie below upper - axial_spacing/2."
        )
    si = _SI.of(case)
    light = float(case.scale.speed_of_light)
    preroll = warpx_preroll_steps(case)
    laser = case.laser
    match case.charge_conservation:
        case "spectral-correction" | "update-with-rho":
            # WarpX's current correction conserves charge only with its periodic
            # single-box global FFT, which RZ lacks; the rho update keeps Gauss.
            pass
        case "vay-deposition":
            raise ValueError("Quasi-cylindrical PSATD has no Vay deposition.")
        case unreachable:
            assert_never(unreachable)
    # Cell-centered output of the collocated grid lands on the case's axial nodes.
    z_lower = (case.lower - shift) * si.length
    z_upper = z_lower + case.axial_count * case.axial_spacing * si.length
    lines = [
        f"max_step = {preroll + case.steps}",
        f"amr.n_cell = {case.radial_count} {case.axial_count}",
        f"amr.max_grid_size = {max(case.radial_count, case.axial_count)}",
        "amr.blocking_factor = 1",
        "amr.max_level = 0",
        "amrex.signal_handling = 0",
        "geometry.dims = RZ",
        f"geometry.prob_lo = 0.0 {_number(z_lower)}",
        f"geometry.prob_hi = {_number(case.radius * si.length)} {_number(z_upper)}",
        f"warpx.n_rz_azimuthal_modes = {case.mode_count}",
        "boundary.field_lo = none damped",
        "boundary.field_hi = none damped",
        "boundary.particle_lo = none absorbing",
        "boundary.particle_hi = reflecting absorbing",
        f"warpx.const_dt = {_number(case.step_size * si.time)}",
        "warpx.use_filter = 0",
        "warpx.verbose = 1",
        f"algo.particle_shape = {case.shape_order}",
        f"algo.particle_pusher = {_pusher_name(case.pusher, 'warpx')}",
        "algo.maxwell_solver = psatd",
        "warpx.grid_type = collocated",
        f"psatd.noz = {_WARPX_RZ_STENCIL}",
        "algo.current_deposition = direct",
        "psatd.current_correction = 0",
        "psatd.update_with_rho = 1",
        *_warpx_species_lines(case.species, si, light),
        "lasers.names = laser",
        f"laser.position = 0.0 0.0 {_number(laser.center * si.length)}",
        "laser.direction = 0.0 0.0 1.0",
        "laser.polarization = 1.0 0.0 0.0",
        f"laser.e_max = {_number(laser.amplitude * si.electric_field)}",
        f"laser.wavelength = {_number(laser.wavelength * si.length)}",
        "laser.profile = Gaussian",
        f"laser.profile_waist = {_number(laser.waist * si.length)}",
        f"laser.profile_duration = {_number(laser.length / light * si.time)}",
        f"laser.profile_t_peak = {_number(preroll * case.step_size * si.time)}",
        "laser.profile_focal_distance = 0.0",
        "diagnostics.diags_names = fields",
        "fields.diag_type = Full",
        "fields.format = openpmd",
        "fields.openpmd_backend = h5",
        "fields.openpmd_encoding = f",
        f"fields.file_prefix = {_WARPX_PREFIX}",
        "fields.file_min_digits = 6",
        f"fields.intervals = {preroll}:{preroll + case.steps}:{case.output_interval}",
        "fields.fields_to_plot = Er Et Ez Br Bt Bz",
        "fields.write_species = 0",
    ]
    return {"inputs": ("\n".join(lines) + "\n").encode()}


def warpx_input(case: PICOracleCase | PICOracleWakefieldCase, /) -> dict[str, bytes]:
    """Translate a case into one WarpX ``inputs`` deck (SI units).

    Cartesian cases run on ``warpx.3d``; a `PICOracleWakefieldCase` runs on
    ``warpx.rz`` (see `warpx_preroll_steps` for its antenna timing).
    """
    case = _require_any_case(case)
    if isinstance(case, PICOracleWakefieldCase):
        return _warpx_rz_input(case)
    if min(case.counts) <= case.shape_order + 3:
        raise ValueError("Every WarpX axis needs more cells than its guard cells.")
    si = _SI.of(case)
    lower = np.asarray(case.lower) * si.length
    upper = lower + np.asarray(case.counts) * np.asarray(case.spacing) * si.length
    periodic = "periodic periodic periodic"
    lines = [
        f"max_step = {case.steps}",
        f"amr.n_cell = {case.counts[0]} {case.counts[1]} {case.counts[2]}",
        f"amr.max_grid_size = {max(case.counts)}",
        "amr.blocking_factor = 1",
        "amr.max_level = 0",
        "amrex.signal_handling = 0",
        "geometry.dims = 3",
        f"geometry.prob_lo = {_numbers(lower)}",
        f"geometry.prob_hi = {_numbers(upper)}",
        f"boundary.field_lo = {periodic}",
        f"boundary.field_hi = {periodic}",
        f"boundary.particle_lo = {periodic}",
        f"boundary.particle_hi = {periodic}",
        f"warpx.const_dt = {_number(case.step_size * si.time)}",
        "warpx.use_filter = 0",
        "warpx.verbose = 1",
        f"algo.particle_shape = {case.shape_order}",
        f"algo.particle_pusher = {_pusher_name(case.pusher, 'warpx')}",
        "algo.field_gathering = energy-conserving",
        *_warpx_solver_lines(case),
        *_warpx_species_lines(case.species, si, float(case.scale.speed_of_light)),
        *_warpx_output_lines(case, si),
    ]
    return {"inputs": ("\n".join(lines) + "\n").encode()}


def _warpx_losses(case: PICOracleCase, /) -> tuple[AdapterLoss, ...]:
    losses = [
        AdapterLoss(
            "initial_field",
            "export",
            "dropped",
            "WarpX starts from zero fields; the Gauss-consistent initial field "
            "Phydrax solves from the initial charge is not imposed (WarpX "
            "initializes self fields only with PML boundaries).",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "fields/E,fields/B",
            "import",
            "transformed",
            "WarpX averages every field component to cell centers before output; "
            "samples are cell-centered averages of the solver fields.",
            changes_interpretation=False,
        ),
    ]
    spectral = case.spectral
    if (
        spectral is not None
        and spectral.grid == "collocated"
        and spectral.charge_conservation != "vay-deposition"
    ):
        losses.append(
            AdapterLoss(
                "current_deposition",
                "export",
                "transformed",
                "WarpX refuses charge-conserving deposition on collocated grids; "
                "direct deposition followed by the declared spectral charge "
                "conservation replaces the spline-Whitney Esirkepov current.",
                changes_interpretation=True,
            )
        )
    return tuple(losses)


def _snapshot_limits(case: PICOracleCase, components: int, /) -> ResourceLimits:
    """Bounds of one snapshot file holding ``components`` float64 grid arrays."""
    return ResourceLimits(
        8 * components * case.cell_count + _SNAPSHOT_SLACK_BYTES,
        16,
        components * case.cell_count + _SNAPSHOT_NODES,
        _SNAPSHOT_ATTRIBUTES,
        1,
    )


def _grid_matches(record: OpenPMDMeshRecord, case: PICOracleCase, /) -> None:
    if record.geometry != "cartesian" or record.axis_labels != ("x", "y", "z"):
        raise ValueError("Oracle field records must be Cartesian x, y, z meshes.")
    if record.components[0].shape != case.counts:
        raise ValueError("Oracle field records do not match the case grid.")
    for spacing, expected, offset, lower in zip(
        record.grid_spacing,
        case.spacing,
        record.grid_global_offset,
        case.lower,
        strict=True,
    ):
        if not math.isclose(spacing, expected, rel_tol=_GRID_TOLERANCE) or not (
            abs(offset - lower) <= _GRID_TOLERANCE * expected * max(case.counts)
        ):
            raise ValueError("Oracle field records do not match the case grid.")


def _positions(record: OpenPMDMeshRecord, /) -> tuple[tuple[float, float, float], ...]:
    return tuple((value[0], value[1], value[2]) for value in record.positions)


def read_pic_oracle_openpmd(
    case: PICOracleCase, resources: Sequence[BoundedResource], /
) -> PICOracleFields:
    """Import one openPMD 1.1.0 HDF5 snapshot per output iteration of ``case``.

    Snapshots are read by `read_openpmd_meshes_hdf5` in the case scale's units;
    each must hold Cartesian ``E`` and ``B`` on the case grid, and component
    positions must agree across snapshots.
    """
    case = _require_case(case)
    iterations = case.output_iterations
    values = tuple(resources)
    if len(values) != len(iterations):
        raise ValueError("One openPMD snapshot per output iteration is required.")
    times: list[float] = []
    electric: list[np.ndarray] = []
    magnetic: list[np.ndarray] = []
    layout: set[tuple[object, ...]] = set()
    source_format = ""
    for iteration, resource in zip(iterations, values, strict=True):
        imported = read_openpmd_meshes_hdf5(
            resource,
            OpenPMDMeshImportPolicy(iteration, records=_OPENPMD_RECORDS),
            scale=case.scale,
        )
        e_record = imported.iteration.record("E")
        b_record = imported.iteration.record("B")
        for record in (e_record, b_record):
            _grid_matches(record, case)
        layout.add(
            (
                _positions(e_record),
                _positions(b_record),
                e_record.time_offset,
                b_record.time_offset,
            )
        )
        times.append(imported.iteration.time)
        electric.append(np.stack(e_record.components, axis=-1))
        magnetic.append(np.stack(b_record.components, axis=-1))
        source_format = imported.report.source_format
    if len(layout) != 1:
        raise ValueError("Oracle field staggering changed between snapshots.")
    e_positions, b_positions, e_offset, b_offset = layout.pop()
    return _fields(
        iterations,
        times,
        electric,
        magnetic,
        e_positions,  # ty: ignore[invalid-argument-type]
        b_positions,  # ty: ignore[invalid-argument-type]
        (float(e_offset),) * 3,  # ty: ignore[invalid-argument-type]
        (float(b_offset),) * 3,  # ty: ignore[invalid-argument-type]
        source_format,
    )


def _fields(
    iterations: tuple[int, ...],
    times: Sequence[float],
    electric: Sequence[np.ndarray],
    magnetic: Sequence[np.ndarray],
    electric_positions: tuple[tuple[float, float, float], ...],
    magnetic_positions: tuple[tuple[float, float, float], ...],
    electric_offsets: tuple[float, ...],
    magnetic_offsets: tuple[float, ...],
    source_format: str,
    /,
) -> PICOracleFields:
    arrays = (
        np.asarray(times, dtype=np.float64),
        np.stack(electric).astype(np.float64),
        np.stack(magnetic).astype(np.float64),
    )
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise ValueError("Oracle snapshots must be finite.")
        array.flags.writeable = False
    return PICOracleFields(
        iterations,
        arrays[0],
        arrays[1],
        arrays[2],
        electric_positions,
        magnetic_positions,
        (electric_offsets[0], electric_offsets[1], electric_offsets[2]),
        (magnetic_offsets[0], magnetic_offsets[1], magnetic_offsets[2]),
        source_format,
    )


def _snapshot_requests(
    limits: ResourceLimits, paths: Sequence[str], maximum_output_bytes: int, /
) -> tuple[tuple[PinnedFileRequest, ...], int]:
    total = limits.max_bytes * len(paths)
    if total > maximum_output_bytes:
        raise ValueError(
            f"The case needs {total} output bytes; maximum_output_bytes is "
            f"{maximum_output_bytes}."
        )
    return tuple(PinnedFileRequest(path, limits.max_bytes) for path in paths), total


def _output_digest(artifacts: Sequence[PinnedFileArtifact], /) -> str:
    lines = sorted(f"{value.path} {value.sha256}" for value in artifacts)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _read_artifacts(
    limits: ResourceLimits, artifacts: Sequence[PinnedFileArtifact], /
) -> tuple[BoundedResource, ...]:
    resources: list[BoundedResource] = []
    for artifact in artifacts:
        location = Path(artifact.location)
        resource = read_bounded_resource(
            location.name, trusted_root=location.parent, limits=limits
        )
        if hashlib.sha256(resource.data).hexdigest() != artifact.sha256:
            raise ValueError("A provider artifact changed after publication.")
        resources.append(resource)
    return tuple(resources)


def _report(
    code: PICOracleCode,
    executable: PinnedExecutable,
    source_format: str,
    target_format: str,
    output_sha256: str,
    target_id: str,
    mapping: tuple[str, ...],
    preserved: tuple[str, ...],
    losses: tuple[AdapterLoss, ...],
    assumptions: tuple[str, ...],
    /,
) -> AdapterReport:
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        f"{code}-{executable.version}:{source_format}",
        target_format,
        source_id=output_sha256,
        target_id=target_id,
        coordinate_mapping=(
            "case scale units -> SI (scale.unit_si_map) -> provider deck",
            *mapping,
        ),
        preserved_fields=preserved,
        assumptions=(
            f"{code} {executable.version} executable sha256 {executable.sha256}",
            *assumptions,
        ),
        losses=losses,
    )


def _result(
    code: PICOracleCode,
    case: PICOracleCase,
    executable: PinnedExecutable,
    fields: PICOracleFields,
    output_sha256: str,
    losses: tuple[AdapterLoss, ...],
    assumptions: tuple[str, ...],
    /,
) -> PICOracleResult:
    target = canonical_fingerprint(
        {
            "kind": "pic-oracle-fields",
            "case": case.case_id,
            "iterations": list(fields.iterations),
            "arrays": array_tree_fingerprint(
                (fields.times, fields.electric, fields.magnetic)
            ),
            "electric_positions": [list(value) for value in fields.electric_positions],
            "magnetic_positions": [list(value) for value in fields.magnetic_positions],
            "electric_time_offsets": list(fields.electric_time_offsets),
            "magnetic_time_offsets": list(fields.magnetic_time_offsets),
        }
    )
    report = _report(
        code,
        executable,
        fields.source_format,
        "PICOracleFields",
        output_sha256,
        target,
        (
            "provider field output -> SI (declared unitSI) -> case scale units",
            "provider axes reordered to x, y, z cell indices of the case box",
        ),
        (
            "grid, box origin, and time step",
            "species charge, mass, weights, positions, and proper velocities",
            "shape order and pusher",
            "E and B snapshots with their in-cell positions and time offsets",
        ),
        losses,
        assumptions,
    )
    return PICOracleResult(
        code,
        case.case_id,
        fields,
        executable.version,
        executable.sha256,
        executable.license_id,
        output_sha256,
        report,
    )


def run_warpx(
    provider: WarpXProvider,
    case: PICOracleCase,
    destination: str | Path,
    /,
    *,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 30,
) -> PICOracleResult:
    """Run pinned WarpX on ``case`` and import its ``E``/``B`` snapshots."""
    if not isinstance(provider, WarpXProvider):
        raise TypeError("provider must be a WarpXProvider.")
    case = _require_case(case)
    if case.track is not None:
        raise ValueError(
            "A single-particle-radiation case yields a track; use run_warpx_track."
        )
    inputs = warpx_input(case)
    paths = tuple(
        f"{_WARPX_PREFIX}/openpmd_{iteration:06d}.h5"
        for iteration in case.output_iterations
    )
    limits = _snapshot_limits(case, _FIELD_COMPONENTS)
    requests, total = _snapshot_requests(limits, paths, maximum_output_bytes)
    run = run_pinned_command(
        provider.executable,
        ("inputs",),
        inputs=inputs,
        timeout=timeout,
        environment={"OMP_NUM_THREADS": "1"},
        artifacts=PinnedFileOutputs(str(destination), requests, total),
    )
    artifacts = tuple(run.file_artifact(path) for path in paths)
    fields = read_pic_oracle_openpmd(case, _read_artifacts(limits, artifacts))
    return _result(
        "warpx",
        case,
        provider.executable,
        fields,
        _output_digest(artifacts),
        _warpx_losses(case),
        ("WarpX SI constants come from its own build (ablastr), not CODATA 2022",),
    )


def _track_limits(case: PICOracleCase, /) -> ResourceLimits:
    """Bounds of one group-based track series of the case's output iterations."""
    snapshots = len(case.output_iterations)
    return ResourceLimits(
        snapshots * _TRACK_ITERATION_BYTES + _SNAPSHOT_SLACK_BYTES,
        16,
        snapshots * _TRACK_ITERATION_NODES + _SNAPSHOT_NODES,
        snapshots * _TRACK_ITERATION_ATTRIBUTES + _SNAPSHOT_ATTRIBUTES,
        1,
    )


def read_pic_oracle_track(
    case: PICOracleCase, resource: BoundedResource, /
) -> ChargedTrajectory:
    """Import the tracked particle of a ``"single-particle-radiation"`` case.

    ``resource`` is one group-based openPMD 1.1.0 particle series read by
    `read_openpmd_particle_tracks_hdf5` in the case scale's units at the case's
    output iterations. The tracked species must hold one particle whose charge,
    mass, weight, and sample times are the case's; the lane is returned under
    the case's recorder identity.
    """
    case = _require_case(case)
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    track = case.track
    if track is None:
        raise ValueError("The case has no tracked particle.")
    species = case.species[track.species]
    iterations = case.output_iterations
    imported = read_openpmd_particle_tracks_hdf5(
        resource,
        OpenPMDParticleTrackImportPolicy(
            OpenPMDParticleTrackSelection(species.species_id, iterations)
        ),
        scale=case.scale,
    )
    trajectory = imported.trajectory
    if trajectory.particle_count != 1:
        raise ValueError("The provider track must hold exactly the tracked particle.")
    charge = float(trajectory.charges[0] * trajectory.multiplicities[0])
    if not math.isclose(
        charge, species.particle_charge * float(species.weights[0]), rel_tol=1.0e-9
    ) or not math.isclose(
        float(imported.masses[0]), species.particle_mass, rel_tol=1.0e-9
    ):
        raise ValueError("The provider track does not carry the case's particle.")
    times = np.asarray(trajectory.times[:, 0], dtype=np.float64)
    expected = np.asarray(iterations, dtype=np.float64) * case.step_size
    if not np.allclose(times, expected, rtol=0.0, atol=_GRID_TOLERANCE * expected[-1]):
        raise ValueError("The provider track is not sampled at the case's iterations.")
    return ChargedTrajectory(
        trajectory.times,
        trajectory.positions,
        trajectory.proper_velocities,
        trajectory.charges,
        trajectory.multiplicities,
        trajectory.active,
        (
            np.asarray([track.identity[0]], dtype=np.uint32),
            np.asarray([track.identity[1]], dtype=np.uint32),
        ),
    )


def _warpx_track_losses() -> tuple[AdapterLoss, ...]:
    return (
        AdapterLoss(
            "initial_field",
            "export",
            "dropped",
            "WarpX starts from zero fields; the Gauss-consistent initial field "
            "Phydrax solves from the initial charge is not imposed.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "track/proper_velocities",
            "import",
            "transformed",
            "WarpX synchronizes the half-step momentum to each output time with a "
            "half push (warpx.synchronize_velocity_for_diagnostics); PICTrackRecorder "
            "samples the time-centered mean (u^{k-1/2} + u^{k+1/2})/2. Both are "
            "second-order estimates of u(t_k).",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "track/id",
            "import",
            "transformed",
            "WarpX's particle id is replaced by the case's recorder identity; the "
            "tracked species holds only that particle.",
            changes_interpretation=False,
        ),
    )


def run_warpx_track(
    provider: WarpXProvider,
    case: PICOracleCase,
    destination: str | Path,
    /,
    *,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 30,
) -> PICOracleTrackResult:
    """Run pinned ``warpx.3d`` on a ``"single-particle-radiation"`` case.

    The uniform external field is applied to particles only (WarpX
    ``constant`` particle fields); the tracked species is written as a
    group-based openPMD series every ``output_interval`` steps and imported by
    `read_pic_oracle_track`.
    """
    if not isinstance(provider, WarpXProvider):
        raise TypeError("provider must be a WarpXProvider.")
    case = _require_case(case)
    if case.track is None:
        raise ValueError("run_warpx_track needs a single-particle-radiation case.")
    inputs = warpx_input(case)
    limits = _track_limits(case)
    requests, total = _snapshot_requests(limits, (_WARPX_TRACK,), maximum_output_bytes)
    run = run_pinned_command(
        provider.executable,
        ("inputs",),
        inputs=inputs,
        timeout=timeout,
        environment={"OMP_NUM_THREADS": "1"},
        artifacts=PinnedFileOutputs(str(destination), requests, total),
    )
    artifact = run.file_artifact(_WARPX_TRACK)
    (resource,) = _read_artifacts(limits, (artifact,))
    track = read_pic_oracle_track(case, resource)
    output_sha256 = _output_digest((artifact,))
    target = canonical_fingerprint(
        {
            "kind": "pic-oracle-track",
            "case": case.case_id,
            "iterations": list(case.output_iterations),
            "arrays": array_tree_fingerprint(
                (
                    track.times,
                    track.positions,
                    track.proper_velocities,
                    track.charges,
                    track.multiplicities,
                )
            ),
            "identity": list(case.track.identity),
        }
    )
    report = _report(
        "warpx",
        provider.executable,
        "openpmd-particle-tracks",
        "ChargedTrajectory",
        output_sha256,
        target,
        (
            "provider particle series -> SI (declared unitSI) -> case scale units",
            "tracked species' only particle -> case recorder identity",
        ),
        (
            "grid, box origin, and time step",
            "species charge, mass, weights, positions, and proper velocities",
            "uniform external E and B",
            "shape order and pusher",
            "tracked particle positions and proper velocities at output times",
        ),
        _warpx_track_losses(),
        ("WarpX SI constants come from its own build (ablastr), not CODATA 2022",),
    )
    return PICOracleTrackResult(
        "warpx",
        case.case_id,
        track,
        provider.executable.version,
        provider.executable.sha256,
        provider.executable.license_id,
        output_sha256,
        report,
    )


def _modal_limits(case: PICOracleWakefieldCase, /) -> ResourceLimits:
    """Bounds of one thetaMode snapshot of E and B."""
    elements = (
        _MODAL_COMPONENTS
        * (2 * case.mode_count - 1)
        * case.radial_count
        * case.axial_count
    )
    return ResourceLimits(
        8 * elements + _SNAPSHOT_SLACK_BYTES,
        16,
        elements + _SNAPSHOT_NODES,
        _SNAPSHOT_ATTRIBUTES,
        1,
    )


def _modal_grid(
    record: OpenPMDMeshRecord, case: PICOracleWakefieldCase, /
) -> tuple[np.ndarray, np.ndarray]:
    """Radial and axial sample coordinates of one WarpX thetaMode record."""
    shape = (2 * case.mode_count - 1, case.radial_count, case.axial_count)
    if record.geometry != "thetaMode" or record.components[0].shape != shape:
        raise ValueError("Oracle field records do not match the case's modal grid.")
    radial_spacing = case.radius / case.radial_count
    expected = (
        (radial_spacing, case.axial_spacing),
        (0.0, case.lower - 0.5 * case.axial_spacing),
    )
    size = max(case.radius, abs(case.lower) + case.axial_count * case.axial_spacing)
    for values, reference in zip(
        (record.grid_spacing, record.grid_global_offset), expected, strict=True
    ):
        if any(
            abs(value - target) > _GRID_TOLERANCE * size
            for value, target in zip(values, reference, strict=True)
        ):
            raise ValueError("Oracle field records do not match the case's modal grid.")
    if any(position != (0.5, 0.5) for position in record.positions):
        raise ValueError("WarpX RZ output must be cell centered.")
    radial = (np.arange(case.radial_count, dtype=np.float64) + 0.5) * radial_spacing
    axial = case.lower + np.arange(case.axial_count, dtype=np.float64) * (
        case.axial_spacing
    )
    return radial, axial


def read_warpx_wakefield(
    case: PICOracleWakefieldCase, resources: Sequence[BoundedResource], /
) -> PICOracleModalFields:
    """Import WarpX RZ openPMD snapshots of a ``"laser-wakefield-stage"`` case.

    One file per case output iteration (WarpX iteration
    ``warpx_preroll_steps(case) + k``) is read by `read_openpmd_meshes_hdf5` in
    the case scale's units; its cell-centered ``thetaMode`` ``E``/``B`` must sit
    on the case's radial and axial nodes. Times are shifted by the pre-roll so
    that ``times[0]`` is the case's ``t = 0``.
    """
    if not isinstance(case, PICOracleWakefieldCase):
        raise TypeError("case must be a PICOracleWakefieldCase.")
    iterations = case.output_iterations
    values = tuple(resources)
    if len(values) != len(iterations):
        raise ValueError("One openPMD snapshot per output iteration is required.")
    preroll = warpx_preroll_steps(case)
    times: list[float] = []
    electric: list[np.ndarray] = []
    magnetic: list[np.ndarray] = []
    radial = axial = np.zeros((0,), dtype=np.float64)
    source_format = ""
    for iteration, resource in zip(iterations, values, strict=True):
        imported = read_openpmd_meshes_hdf5(
            resource,
            OpenPMDMeshImportPolicy(preroll + iteration, records=_OPENPMD_RECORDS),
            scale=case.scale,
        )
        e_record = imported.iteration.record("E")
        b_record = imported.iteration.record("B")
        radial, axial = _modal_grid(e_record, case)
        _modal_grid(b_record, case)
        times.append(imported.iteration.time - preroll * case.step_size)
        electric.append(np.stack(e_record.components, axis=-1))
        magnetic.append(np.stack(b_record.components, axis=-1))
        source_format = imported.report.source_format
    arrays = (
        np.asarray(times, dtype=np.float64),
        np.stack(electric).astype(np.float64),
        np.stack(magnetic).astype(np.float64),
        radial,
        axial,
    )
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise ValueError("Oracle snapshots must be finite.")
        array.flags.writeable = False
    return PICOracleModalFields(iterations, *arrays, source_format)


def _warpx_wakefield_losses(case: PICOracleWakefieldCase, /) -> tuple[AdapterLoss, ...]:
    preroll = warpx_preroll_steps(case)
    return (
        AdapterLoss(
            "laser",
            "export",
            "transformed",
            "WarpX emits the Gaussian pulse from an antenna plane at the pulse "
            f"center (focal distance 0, t_peak = {preroll} steps) instead of "
            "initializing it in the box; the case's t = 0 is WarpX step "
            f"{preroll}, and the plasma sees the pulse's leading tail during the "
            "pre-roll.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "boundaries",
            "export",
            "transformed",
            "The antenna also radiates a pulse toward -z; WarpX's axial field "
            "boundaries are damped and its axial particle boundaries absorbing "
            "(not periodic) so that pulse leaves the box. Particles reaching the "
            "radial wall are reflected.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "field_solver",
            "export",
            "transformed",
            "WarpX RZ PSATD transforms z with local FFTs and a "
            f"psatd.noz = {_WARPX_RZ_STENCIL} stencil on a collocated grid shifted "
            "half an axial cell down; Phydrax uses one global periodic "
            "infinite-order FFT.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "current_deposition",
            "export",
            "transformed",
            "WarpX deposits the direct current with its Verboncoeur on-axis "
            "correction and, since its current correction conserves charge only "
            "with the periodic single-box global FFT RZ lacks, updates E with "
            "psatd.update_with_rho for either spectral charge conservation; "
            "Phydrax deposits the step-mean current at the path midpoint with its "
            f"near-axis volume correction and applies {case.charge_conservation}. "
            "WarpX's rho-update error converges slowly with the step.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "initial_field",
            "export",
            "dropped",
            "WarpX starts from zero self fields; the Gauss-consistent initial field "
            "Phydrax solves from the initial charge is not imposed.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "fields/E,fields/B",
            "import",
            "transformed",
            "WarpX averages the collocated nodal fields to cell centers before "
            "output; samples are two-point averages along r and z.",
            changes_interpretation=False,
        ),
    )


def run_warpx_wakefield(
    provider: WarpXProvider,
    case: PICOracleWakefieldCase,
    destination: str | Path,
    /,
    *,
    timeout: float = 3600.0,
    maximum_output_bytes: int = 1 << 30,
) -> PICOracleWakefieldResult:
    """Run pinned ``warpx.rz`` on a ``"laser-wakefield-stage"`` case.

    The RZ PSATD deck carries the case's particles explicitly and launches the
    laser from an antenna (see `warpx_preroll_steps`); the thetaMode snapshots
    are imported by `read_warpx_wakefield`.
    """
    if not isinstance(provider, WarpXProvider):
        raise TypeError("provider must be a WarpXProvider.")
    if not isinstance(case, PICOracleWakefieldCase):
        raise TypeError("case must be a PICOracleWakefieldCase.")
    inputs = warpx_input(case)
    preroll = warpx_preroll_steps(case)
    paths = tuple(
        f"{_WARPX_PREFIX}/openpmd_{preroll + iteration:06d}.h5"
        for iteration in case.output_iterations
    )
    limits = _modal_limits(case)
    requests, total = _snapshot_requests(limits, paths, maximum_output_bytes)
    run = run_pinned_command(
        provider.executable,
        ("inputs",),
        inputs=inputs,
        timeout=timeout,
        # Contiguous datasets: openPMD-api's default chunks pad small mode planes.
        environment={"OMP_NUM_THREADS": "1", "OPENPMD_HDF5_CHUNKS": "none"},
        artifacts=PinnedFileOutputs(str(destination), requests, total),
    )
    artifacts = tuple(run.file_artifact(path) for path in paths)
    fields = read_warpx_wakefield(case, _read_artifacts(limits, artifacts))
    output_sha256 = _output_digest(artifacts)
    target = canonical_fingerprint(
        {
            "kind": "pic-oracle-modal-fields",
            "case": case.case_id,
            "iterations": list(fields.iterations),
            "arrays": array_tree_fingerprint(
                (
                    fields.times,
                    fields.electric,
                    fields.magnetic,
                    fields.radial_coordinates,
                    fields.axial_coordinates,
                )
            ),
        }
    )
    report = _report(
        "warpx",
        provider.executable,
        fields.source_format,
        "PICOracleModalFields",
        output_sha256,
        target,
        (
            "provider thetaMode output -> SI (declared unitSI) -> case scale units",
            "provider [mode, z, r] planes -> [mode, r, z] on the case's nodes",
            f"provider iteration - {preroll} pre-roll steps -> case iteration",
        ),
        (
            "grid, azimuthal modes, and time step",
            "species charge, mass, weights, positions, and proper velocities",
            "laser amplitude, wavelength, waist, duration, focus, and polarization",
            "shape order, pusher, and spectral charge conservation",
            "E and B azimuthal modes at the case's output iterations",
        ),
        _warpx_wakefield_losses(case),
        ("WarpX SI constants come from its own build (ablastr), not CODATA 2022",),
    )
    return PICOracleWakefieldResult(
        "warpx",
        case.case_id,
        fields,
        provider.executable.version,
        provider.executable.sha256,
        provider.executable.license_id,
        output_sha256,
        report,
    )


# Smilei -----------------------------------------------------------------------------


def _smilei_reference(case: PICOracleCase, /) -> tuple[float, float, float, float]:
    """``(ω_r, L_r, n_r, T_r)`` of Smilei's normalization in SI.

    ``ω_r = c/L`` makes Smilei's length unit ``c/ω_r`` the scale length unit;
    ``n_r = ε₀ m_e ω_r²/e²`` with the SI constants of the CODATA 2022 scale.
    """
    si_scale = ElectromagneticScaleContract.si()
    light = float(si_scale.speed_of_light)
    length = _SI.of(case).length
    omega = light / length
    density = (
        float(si_scale.vacuum_permittivity)
        * float(si_scale.electron_mass)
        * omega**2
        / float(si_scale.elementary_charge) ** 2
    )
    return omega, light / omega, density, 1.0 / omega


def _npy(array: np.ndarray, /) -> bytes:
    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(array, dtype=np.float64), allow_pickle=False)
    return buffer.getvalue()


def smilei_input(case: PICOracleCase | PICOracleWakefieldCase, /) -> dict[str, bytes]:
    """Translate a case into a Smilei namelist and its ``numpy`` particle arrays."""
    case = _require_any_case(case)
    if isinstance(case, PICOracleWakefieldCase):
        raise ValueError(
            "Smilei's AMcylindrical geometry solves Maxwell by FDTD; the oracle has "
            "no quasi-cylindrical PSATD scenario."
        )
    match case.scenario:
        case "periodic-psatd-plasma":
            raise ValueError("Smilei has no Cartesian PSATD solver.")
        case "single-particle-radiation":
            raise ValueError(
                "Smilei writes particle tracks as openPMD 1.0.0 without the ED-PIC "
                "macroWeighted/weightingPower, mass, and weighting records the "
                "openPMD particle-track reader requires."
            )
        case "periodic-yee-plasma":
            pass
        case "laser-wakefield-stage":
            raise ValueError("A laser-wakefield-stage case is a PICOracleWakefieldCase.")
        case unreachable:
            assert_never(unreachable)
    if case.shape_order != 2:
        raise ValueError("Smilei interpolation order 2 needs shape_order == 2.")
    si = _SI.of(case)
    si_scale = ElectromagneticScaleContract.si()
    omega, length, density, time = _smilei_reference(case)
    light = float(case.scale.speed_of_light)
    cells = np.asarray(case.spacing) * si.length / length
    step = case.step_size * si.time / time
    electron_mass = float(si_scale.electron_mass)
    electron_charge = float(si_scale.elementary_charge)
    periodic = '[["periodic"], ["periodic"], ["periodic"]]'
    lines = [
        "import numpy as np",
        "Main(",
        '    geometry="3Dcartesian",',
        "    interpolation_order=2,",
        f"    cell_length=[{', '.join(_number(value) for value in cells)}],",
        "    grid_length=["
        + ", ".join(
            _number(value * count)
            for value, count in zip(cells, case.counts, strict=True)
        )
        + "],",
        "    number_of_patches=[1, 1, 1],",
        f"    timestep={_number(step)},",
        f"    simulation_time={_number((case.steps + 0.5) * step)},",
        f"    EM_boundary_conditions={periodic},",
        '    maxwell_solver="Yee",',
        "    solve_poisson=True,",
        f"    reference_angular_frequency_SI={_number(omega)},",
        f"    print_every={case.steps},",
        ")",
    ]
    inputs: dict[str, bytes] = {}
    volume = length**3
    for species in case.species:
        name = species.species_id
        # Smilei stores integer charge states in units of e.
        charge = species.particle_charge * si.charge / electron_charge
        state = round(charge)
        if abs(charge - state) > 1.0e-9 * max(1.0, abs(charge)):
            raise ValueError("Smilei particle charges must be integer multiples of e.")
        origin = np.asarray(case.lower) * si.length
        points = (species.positions * si.length - origin) / length
        weights = species.weights / (density * volume)
        inputs[f"{name}_position.npy"] = _npy(np.vstack((points.T, weights[None, :])))
        inputs[f"{name}_momentum.npy"] = _npy((species.proper_velocities / light).T)
        lines += [
            "Species(",
            f'    name="{name}",',
            f'    position_initialization=np.load("{name}_position.npy"),',
            f'    momentum_initialization=np.load("{name}_momentum.npy"),',
            f"    mass={_number(species.particle_mass * si.mass / electron_mass)},",
            f"    charge={state},",
            f'    pusher="{_pusher_name(case.pusher, "smilei")}",',
            f"    boundary_conditions={periodic},",
            ")",
        ]
    fields = ", ".join(f'"{value}"' for value in (*_SMILEI_ELECTRIC, *_SMILEI_MAGNETIC))
    lines += [
        "DiagFields(",
        f"    every={case.output_interval},",
        f"    fields=[{fields}],",
        '    datatype="double",',
        ")",
    ]
    inputs["smilei.py"] = ("\n".join(lines) + "\n").encode()
    return inputs


def _smilei_component(
    group: h5py.Group,
    name: str,
    case: PICOracleCase,
    quantity_si: float,
    time_si: float,
    /,
) -> tuple[np.ndarray, tuple[float, float, float], float]:
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Smilei field {name} must be a dataset.")
    labels = tuple(
        value.decode() if isinstance(value, bytes) else str(value)
        for value in np.asarray(dataset.attrs["axisLabels"]).reshape(-1)
    )
    if (
        required_text(dataset.attrs, "geometry") != "cartesian"
        or required_text(dataset.attrs, "dataOrder") != "C"
        or labels != ("x", "y", "z")
    ):
        raise ValueError(f"Smilei field {name} must be a C-ordered x, y, z mesh.")
    stagger = numeric_attribute(dataset.attrs, "position", (3,))
    spacing = numeric_attribute(dataset.attrs, "gridSpacing", (3,))
    spacing_si = spacing * scalar_attribute(dataset.attrs, "gridUnitSI")
    expected = np.asarray(case.spacing) * _SI.of(case).length
    if not np.allclose(spacing_si, expected, rtol=_GRID_TOLERANCE, atol=0.0):
        raise ValueError(f"Smilei field {name} does not match the case grid.")
    origin = numeric_attribute(dataset.attrs, "gridGlobalOffset", (3,))
    if np.any(origin != 0.0) or np.any((stagger != 0.0) & (stagger != 0.5)):
        raise ValueError(f"Smilei field {name} has an unknown staggering.")
    # Smilei writes n + 1 samples per axis at (i + position) * spacing from the
    # lower face; the last one repeats the first on a periodic axis.
    if dataset.shape != tuple(count + 1 for count in case.counts):
        raise ValueError(f"Smilei field {name} does not match the case grid.")
    window = tuple(slice(0, count) for count in case.counts)
    stored = np.asarray(dataset[window], dtype=np.float64)
    values = stored * scalar_attribute(dataset.attrs, "unitSI") / quantity_si
    offset = scalar_attribute(dataset.attrs, "timeOffset") * time_si
    return values, (float(stagger[0]), float(stagger[1]), float(stagger[2])), offset


def read_smilei_fields(
    case: PICOracleCase, resource: BoundedResource, /
) -> PICOracleFields:
    """Import Smilei ``Fields0.h5`` snapshots of ``case`` in its scale's units.

    The bounded HDF5 image is structurally preflighted before any payload is
    read. Each iteration group ``data/<10-digit iteration>`` must hold
    ``Ex, Ey, Ez`` and the time-centered ``Bx_m, By_m, Bz_m`` with Smilei's
    openPMD attributes; values are converted by their ``unitSI`` to SI and by
    ``case.scale.unit_si_map()`` to the scale's units.
    """
    case = _require_case(case)
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    units = case.scale.unit_si_map()
    electric_si = units["electric_field"][0]
    magnetic_si = units["magnetic_field"][0]
    time_unit = units["time"][0]
    with h5py.File(BytesIO(resource.data), "r") as handle:
        inventory = preflight_hdf5(handle, resource.manifest.limits)
        if required_text(handle.attrs, "software") != "Smilei":
            raise ValueError("The file is not Smilei field output.")
        if required_text(handle.attrs, "iterationEncoding") != "groupBased":
            raise ValueError("Smilei field output must be group based.")
        # Stored payloads and their canonical copies each stay within the image cap.
        inventory.require_budget(
            resource.manifest.limits,
            2 * resource.manifest.limits.max_bytes,
            8 * 6 * case.cell_count * len(case.output_iterations),
        )
        times: list[float] = []
        electric: list[np.ndarray] = []
        magnetic: list[np.ndarray] = []
        layout: set[tuple[object, ...]] = set()
        for iteration in case.output_iterations:
            group = handle[f"data/{iteration:010d}"]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"Smilei iteration {iteration} is missing.")
            time_si = scalar_attribute(group.attrs, "timeUnitSI")
            times.append(scalar_attribute(group.attrs, "time") * time_si / time_unit)
            e_parts = [
                _smilei_component(group, name, case, electric_si, time_si / time_unit)
                for name in _SMILEI_ELECTRIC
            ]
            b_parts = [
                _smilei_component(group, name, case, magnetic_si, time_si / time_unit)
                for name in _SMILEI_MAGNETIC
            ]
            electric.append(np.stack([part[0] for part in e_parts], axis=-1))
            magnetic.append(np.stack([part[0] for part in b_parts], axis=-1))
            layout.add(
                (
                    tuple(part[1] for part in e_parts),
                    tuple(part[1] for part in b_parts),
                    tuple(part[2] for part in e_parts),
                    tuple(part[2] for part in b_parts),
                )
            )
    if len(layout) != 1:
        raise ValueError("Smilei field staggering changed between snapshots.")
    e_positions, b_positions, e_offsets, b_offsets = layout.pop()
    return _fields(
        case.output_iterations,
        times,
        electric,
        magnetic,
        e_positions,  # ty: ignore[invalid-argument-type]
        b_positions,  # ty: ignore[invalid-argument-type]
        e_offsets,  # ty: ignore[invalid-argument-type]
        b_offsets,  # ty: ignore[invalid-argument-type]
        "smilei-fields",
    )


def _smilei_losses() -> tuple[AdapterLoss, ...]:
    return (
        AdapterLoss(
            "field_gather",
            "export",
            "transformed",
            "Smilei gathers the Yee fields with its order-2 momentum-conserving "
            "interpolator; Phydrax gathers with spline-Whitney (Galerkin) shapes.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "fields/E,fields/B",
            "import",
            "dropped",
            "Smilei's repeated periodic boundary sample (the last of its n + 1 "
            "samples per axis) is dropped.",
            changes_interpretation=False,
        ),
    )


def run_smilei(
    provider: SmileiProvider,
    case: PICOracleCase,
    destination: str | Path,
    /,
    *,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 30,
) -> PICOracleResult:
    """Run pinned Smilei serially on ``case`` and import its field snapshots."""
    if not isinstance(provider, SmileiProvider):
        raise TypeError("provider must be a SmileiProvider.")
    inputs = smilei_input(case)
    padded = tuple(count + 1 for count in case.counts)
    payload = 6 * 8 * padded[0] * padded[1] * padded[2] * len(case.output_iterations)
    total = payload + _SNAPSHOT_SLACK_BYTES * len(case.output_iterations)
    if total > maximum_output_bytes:
        raise ValueError(
            f"The case needs {total} output bytes; maximum_output_bytes is "
            f"{maximum_output_bytes}."
        )
    run = run_pinned_command(
        provider.executable,
        ("smilei.py",),
        inputs=inputs,
        timeout=timeout,
        environment={"OMP_NUM_THREADS": "1"},
        artifacts=PinnedFileOutputs(
            str(destination), (PinnedFileRequest(_SMILEI_OUTPUT, total),), total
        ),
    )
    artifact = run.file_artifact(_SMILEI_OUTPUT)
    location = Path(artifact.location)
    elements = 6 * padded[0] * padded[1] * padded[2] * len(case.output_iterations)
    resource = read_bounded_resource(
        location.name,
        trusted_root=location.parent,
        limits=ResourceLimits(
            total, 16, elements + _SNAPSHOT_NODES, _SNAPSHOT_ATTRIBUTES, 1
        ),
    )
    if hashlib.sha256(resource.data).hexdigest() != artifact.sha256:
        raise ValueError("A provider artifact changed after publication.")
    return _result(
        "smilei",
        case,
        provider.executable,
        read_smilei_fields(case, resource),
        _output_digest((artifact,)),
        _smilei_losses(),
        (
            "Smilei normalizes to ω_r = c / (scale length unit); decks use CODATA "
            "2022 SI constants while Smilei's output unitSI uses its built-in ones",
        ),
    )


# PIConGPU ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Lattice:
    """One species as a one-per-cell lattice with uniform drift and weight."""

    offset: tuple[float, float, float]
    gamma: float
    direction: tuple[float, float, float]
    density_si: float


def _lattice(case: PICOracleCase, species: PICOracleSpecies, /) -> _Lattice:
    spacing = np.asarray(case.spacing)
    relative = (species.positions - np.asarray(case.lower)) / spacing
    cells = np.floor(relative)
    fraction = relative - cells
    index = (
        cells[:, 0] * case.counts[1] * case.counts[2]
        + cells[:, 1] * case.counts[2]
        + cells[:, 2]
    ).astype(np.int64)
    if species.weights.shape[0] != case.cell_count or np.unique(index).shape[0] != (
        case.cell_count
    ):
        raise ValueError("PIConGPU species must hold exactly one particle per cell.")
    if not np.allclose(fraction, fraction[:1], rtol=0.0, atol=1.0e-9):
        raise ValueError("PIConGPU species need one common in-cell offset.")
    proper = species.proper_velocities
    if not np.allclose(proper, proper[:1], rtol=1.0e-12, atol=0.0) or not np.allclose(
        species.weights, species.weights[0], rtol=1.0e-12, atol=0.0
    ):
        raise ValueError("PIConGPU species need one proper velocity and one weight.")
    light = float(case.scale.speed_of_light)
    u = proper[0] / light
    speed = float(np.linalg.norm(u))
    direction = u / speed if speed > 0.0 else np.asarray((1.0, 0.0, 0.0))
    volume_si = float(np.prod(spacing)) * _SI.of(case).length ** 3
    return _Lattice(
        (float(fraction[0, 0]), float(fraction[0, 1]), float(fraction[0, 2])),
        math.sqrt(1.0 + speed**2),
        (float(direction[0]), float(direction[1]), float(direction[2])),
        float(species.weights[0]) / volume_si,
    )


def _param(body: Sequence[str], includes: Sequence[str], /) -> bytes:
    lines = [
        "// Generated by phydrax.solver.picongpu_input for one oracle case.",
        "#pragma once",
        *(f'#include "{value}"' for value in includes),
        "namespace picongpu",
        "{",
        *(f"    {value}" if value else "" for value in body),
        "} // namespace picongpu",
    ]
    return ("\n".join(lines) + "\n").encode()


def picongpu_input(case: PICOracleCase | PICOracleWakefieldCase, /) -> dict[str, bytes]:
    """Translate a case into PIConGPU ``.param`` overrides of one setup.

    Keys are paths under the staged ``setup`` directory; runtime arguments
    (grid, steps, periodicity, openPMD output) are given by `run_picongpu`.
    """
    case = _require_any_case(case)
    if isinstance(case, PICOracleWakefieldCase):
        raise ValueError("PIConGPU has no quasi-cylindrical geometry.")
    match case.scenario:
        case "periodic-psatd-plasma":
            raise ValueError("The PIConGPU oracle covers the Yee scenario only.")
        case "single-particle-radiation":
            raise ValueError(
                "The PIConGPU oracle creates species as one-per-cell lattices from "
                "a density profile; a single tracked macroparticle is not expressible."
            )
        case "periodic-yee-plasma":
            pass
        case "laser-wakefield-stage":
            raise ValueError("A laser-wakefield-stage case is a PICOracleWakefieldCase.")
        case unreachable:
            assert_never(unreachable)
    # PIConGPU enlarges any axis shorter than three supercells.
    if any(
        count % block != 0 or count < 3 * block
        for count, block in zip(case.counts, _PICONGPU_SUPERCELL, strict=True)
    ):
        raise ValueError(
            "PIConGPU cell counts must be multiples of (8, 8, 4) spanning at least "
            "three supercells per axis."
        )
    si = _SI.of(case)
    si_scale = ElectromagneticScaleContract.si()
    lattices = tuple(_lattice(case, value) for value in case.species)
    base = lattices[0].density_si
    weights = min(float(value.weights[0]) for value in case.species)
    shape = _PICONGPU_SHAPES[case.shape_order]
    names: list[str] = []
    definitions: list[str] = []
    manipulators: list[str] = []
    positions: list[str] = []
    pipeline: list[str] = []
    for index, (species, lattice) in enumerate(zip(case.species, lattices, strict=True)):
        tag = f"Species{index}"
        names.append(tag)
        mass = species.particle_mass * si.mass / float(si_scale.electron_mass)
        charge = -species.particle_charge * si.charge / float(si_scale.elementary_charge)
        definitions += [
            f"value_identifier(float_X, MassRatio{tag}, {_number(mass)});",
            f"value_identifier(float_X, ChargeRatio{tag}, {_number(charge)});",
            f"value_identifier(float_X, DensityRatio{tag}, "
            f"{_number(lattice.density_si / base)});",
            f"using Flags{tag} = MakeSeq_t<",
            "    particlePusher<UsedParticlePusher>,",
            "    shape<UsedParticleShape>,",
            "    interpolation<UsedField2Particle>,",
            "    current<UsedParticleCurrentSolver>,",
            f"    massRatio<MassRatio{tag}>,",
            f"    chargeRatio<ChargeRatio{tag}>,",
            f"    densityRatio<DensityRatio{tag}>>;",
            f'using {tag} = Particles<PMACC_CSTRING("{species.species_id}"), '
            f"Flags{tag}, MakeSeq_t<position<position_pic>, momentum, weighting>>;",
        ]
        direction = ", ".join(_number(value) for value in lattice.direction)
        offset = ", ".join(_number(value) for value in lattice.offset)
        manipulators += [
            f"struct Drift{tag}",
            "{",
            f"    static constexpr float_64 gamma = {_number(lattice.gamma)};",
            f"    static constexpr auto driftDirection = float3_X({direction});",
            "};",
            f"using AssignDrift{tag} = unary::Drift<Drift{tag}, "
            "pmacc::math::operation::Assign>;",
        ]
        positions += [
            f"struct Lattice{tag}",
            "{",
            "    static constexpr uint32_t numParticlesPerCell = 1u;",
            f"    static constexpr auto inCellOffset = float3_X({offset});",
            "};",
            f"using OnePosition{tag} = OnePositionImpl<Lattice{tag}>;",
        ]
        pipeline += [
            f"CreateDensity<densityProfiles::Homogenous, "
            f"startPosition::OnePosition{tag}, {tag}>,",
            f"Manipulate<manipulators::AssignDrift{tag}, {tag}>,",
        ]
    pipeline[-1] = pipeline[-1].rstrip(",")
    spacing = np.asarray(case.spacing) * si.length
    params = f"{_PICONGPU_PARAMS}"
    return {
        f"{params}/precision.param": _param(
            (
                "namespace precisionPIConGPU = precision64Bit;",
                "namespace precisionSqrt = precisionPIConGPU;",
                "namespace precisionExp = precisionPIConGPU;",
                "namespace precisionTrigonometric = precisionPIConGPU;",
            ),
            ("picongpu/simulation_types.hpp",),
        )
        + b'#include "picongpu/unitless/precision.unitless"\n',
        f"{params}/simulation.param": _param(
            (
                "namespace SI",
                "{",
                f"    constexpr float_64 DELTA_T_SI = {_number(case.step_size * si.time)};",
                f"    constexpr float_64 CELL_WIDTH_SI = {_number(spacing[0])};",
                f"    constexpr float_64 CELL_HEIGHT_SI = {_number(spacing[1])};",
                f"    constexpr float_64 CELL_DEPTH_SI = {_number(spacing[2])};",
                f"    constexpr float_64 BASE_DENSITY_SI = {_number(base)};",
                "} // namespace SI",
                "constexpr uint32_t TYPICAL_PARTICLES_PER_CELL = 1u;",
            ),
            (),
        ),
        f"{params}/species.param": _param(
            (
                f"using UsedParticleShape = particles::shapes::{shape};",
                "using UsedField2Particle = FieldToParticleInterpolation<"
                "UsedParticleShape, AssignedTrilinearInterpolation>;",
                "using UsedParticleCurrentSolver = "
                "currentSolver::Esirkepov<UsedParticleShape>;",
                "using UsedParticlePusher = particles::pusher::"
                f"{_pusher_name(case.pusher, 'picongpu')};",
            ),
            (
                "picongpu/algorithms/AssignedTrilinearInterpolation.hpp",
                "picongpu/algorithms/FieldToParticleInterpolation.hpp",
                "picongpu/fields/currentDeposition/Solver.def",
                "picongpu/particles/shapes.hpp",
            ),
        ),
        f"{params}/particle.param": _param(
            (
                "namespace particles",
                "{",
                f"    constexpr float_X MIN_WEIGHTING = {_number(0.5 * weights)};",
                "    namespace manipulators",
                "    {",
                *(f"        {value}" for value in manipulators),
                "    } // namespace manipulators",
                "    namespace startPosition",
                "    {",
                *(f"        {value}" for value in positions),
                "    } // namespace startPosition",
                "} // namespace particles",
            ),
            (
                "picongpu/particles/filter/filter.def",
                "picongpu/particles/manipulators/manipulators.def",
                "picongpu/particles/startPosition/functors.def",
                "pmacc/math/operation.hpp",
            ),
        ),
        f"{params}/speciesDefinition.param": _param(
            (*definitions, f"using VectorAllSpecies = MakeSeq_t<{', '.join(names)}>;"),
            (
                "picongpu/defines.hpp",
                "picongpu/particles/Particles.hpp",
                "pmacc/identifier/value_identifier.hpp",
                "pmacc/meta/String.hpp",
                "pmacc/meta/conversion/MakeSeq.hpp",
                "pmacc/particles/Identifier.hpp",
                "pmacc/particles/traits/FilterByFlag.hpp",
            ),
        ),
        f"{params}/speciesInitialization.param": _param(
            (
                "namespace particles",
                "{",
                "    using InitPipeline = pmacc::mp_list<",
                *(f"        {value}" for value in pipeline),
                "    >;",
                "} // namespace particles",
            ),
            ("picongpu/particles/InitFunctors.hpp",),
        ),
    }


def _picongpu_losses() -> tuple[AdapterLoss, ...]:
    return (
        AdapterLoss(
            "initial_field",
            "export",
            "dropped",
            "PIConGPU starts from zero fields; the Gauss-consistent initial field "
            "Phydrax solves from the initial charge is not imposed.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "field_gather",
            "export",
            "transformed",
            "PIConGPU gathers every staggered component with the full particle "
            "shape; Phydrax gathers with spline-Whitney (Galerkin) shapes.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "species/positions",
            "export",
            "transformed",
            "PIConGPU recreates each species from a homogeneous density as one "
            "particle per cell at the case's common in-cell offset.",
            changes_interpretation=False,
        ),
    )


def run_picongpu(
    provider: PIConGPUProvider,
    case: PICOracleCase,
    destination: str | Path,
    /,
    *,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 30,
) -> PICOracleResult:
    """Compile the case's PIConGPU setup, run it serially, and import its snapshots.

    The pinned driver compiles the staged ``setup`` into ``setup/bin/picongpu``,
    published under ``destination/build``; after its digest is checked against
    the published artifact the binary is pinned and run with openPMD HDF5
    output published under ``destination/output``.
    """
    if not isinstance(provider, PIConGPUProvider):
        raise TypeError("provider must be a PIConGPUProvider.")
    inputs = picongpu_input(case)
    paths = tuple(
        f"openPMD/simData_{iteration:06d}.h5" for iteration in case.output_iterations
    )
    # PIConGPU's fields_all writes E, B, and a charge and energy density per species.
    limits = _snapshot_limits(case, _FIELD_COMPONENTS + 2 * len(case.species))
    requests, total = _snapshot_requests(limits, paths, maximum_output_bytes)
    root = Path(destination)
    build_root = root / "build"
    output_root = root / "output"
    for directory in (build_root, output_root):
        directory.mkdir(parents=True, exist_ok=False)
    built = run_pinned_command(
        provider.executable,
        (_PICONGPU_SETUP, *provider.cmake_arguments),
        inputs=inputs,
        timeout=provider.build_timeout,
        artifacts=PinnedFileOutputs(
            str(build_root),
            (PinnedFileRequest(_PICONGPU_BINARY, _PICONGPU_BINARY_BYTES),),
            _PICONGPU_BINARY_BYTES,
        ),
    )
    compiled = built.file_artifact(_PICONGPU_BINARY)
    # Published artifacts are data files; the compiled binary is made
    # executable and pinned only if its bytes are still the published ones.
    Path(compiled.location).chmod(0o700)
    binary = pin_executable(
        compiled.location,
        version=provider.executable.version,
        license_id=provider.executable.license_id,
    )
    if binary.sha256 != compiled.sha256:
        raise ValueError("The compiled PIConGPU binary changed after publication.")
    run = run_pinned_command(
        binary,
        (
            "-d",
            "1",
            "1",
            "1",
            "-g",
            *(str(count) for count in case.counts),
            "-s",
            str(case.steps),
            "--periodic",
            "1",
            "1",
            "1",
            "--openPMD.period",
            str(case.output_interval),
            "--openPMD.file",
            "simData",
            "--openPMD.ext",
            "h5",
            "--openPMD.source",
            "fields_all",
        ),
        inputs={},
        timeout=timeout,
        environment={"OMP_NUM_THREADS": "1"},
        artifacts=PinnedFileOutputs(str(output_root), requests, total),
    )
    artifacts = tuple(run.file_artifact(path) for path in paths)
    fields = read_pic_oracle_openpmd(case, _read_artifacts(limits, artifacts))
    return _result(
        "picongpu",
        case,
        binary,
        fields,
        _output_digest(artifacts),
        _picongpu_losses(),
        (
            f"PIConGPU compiled by driver {provider.executable.sha256} "
            f"({' '.join(provider.cmake_arguments)})",
            f"compiled binary is used under {_PICONGPU_LICENSE} as an external "
            "oracle only",
        ),
    )


__all__ = [
    "PICOracleCase",
    "PICOracleCode",
    "PICOracleFields",
    "PICOracleLaser",
    "PICOracleModalFields",
    "PICOracleResult",
    "PICOracleScenario",
    "PICOracleSpecies",
    "PICOracleSpectralSolver",
    "PICOracleTrack",
    "PICOracleTrackResult",
    "PICOracleWakefieldCase",
    "PICOracleWakefieldResult",
    "PIConGPUProvider",
    "SmileiProvider",
    "WarpXProvider",
    "pic_oracle_case",
    "pic_oracle_wakefield_case",
    "picongpu_input",
    "read_pic_oracle_openpmd",
    "read_pic_oracle_track",
    "read_smilei_fields",
    "read_warpx_wakefield",
    "run_picongpu",
    "run_smilei",
    "run_warpx",
    "run_warpx_track",
    "run_warpx_wakefield",
    "smilei_input",
    "warpx_input",
    "warpx_preroll_steps",
]
