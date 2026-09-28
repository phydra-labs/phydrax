#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned SRW oracle for single-electron synchrotron and undulator spectra.

SRW, the Synchrotron Radiation Workshop (O. Chubar and P. Elleaume, "Accurate
and efficient computation of synchrotron radiation in the near field region",
Proc. EPAC 1998, p. 1177; https://github.com/ochubar/SRW; EPICS Open License,
SPDX ``EPICS``), is run by a caller-pinned Python interpreter with ``srwpy``
installed, through :func:`phydrax.run_pinned_command`. Nothing is imported
into this process and no SRW source is copied. Validated live against srwpy
4.2.1.

:func:`srw_input` translates a :class:`TrajectoryRadiationPlan` and one
electron source into an SRW driver and a JSON deck; SRW's
``CalcElecFieldSR`` computes the near-field electric field at the point
``o + D n`` for each observer direction ``n``, the source origin
``o = (0, 0, z_mid)`` and the observation distance ``D``;
:func:`read_srw_output` converts it to the far-field convention of
:mod:`phydrax.electromagnetics` (phasors ``exp(−iωt)``,
``F(ω) = ∫ f(t) exp(+iωt) dt``, ``d²W/(dω dΩ) = ε₀ c |r Ẽ|² / π``).

Supported subset (everything else is refused with ``ValueError`` before SRW
runs):

- scales whose units are referenced to SI (their openPMD ``unitSI`` map);
- one electron (charge ``−e``, rest energy ``mₑc²``, multiplicity one);
- plans with ``"coherent"`` or ``"incoherent"`` coherence (identical for one
  lane), uniformly spaced angular frequencies, and forward directions
  (``n_z > 0``);
- a :class:`ChargedTrajectory` source with one lane active at every sample
  and uniform lab-time sampling: SRW integrates the radiation of that
  trajectory (positions and ``β`` on its uniform ``c t`` grid);
- an :class:`SRWFieldMapSource`: a field-map tracking plan whose beamline
  holds only static magnetic elements (insertion devices, dipole bends,
  tabulated maps without an electric table) and a one-electron bunch at
  ``ζ = 0``. SRW integrates its own trajectory from the entrance plane
  through either the beamline sampled onto a uniform table (``"tabulated"``)
  or, for a single insertion device, SRW's ideal undulator
  (``"ideal-undulator"``).

Conversion: SRW's engine maps photon energy to wavenumber with its own
constant ``k = 2π · 0.80654658 µm⁻¹ · E/eV``, so the deck requests
``E = ω / (c · 2π · 0.80654658 µm⁻¹)`` and SRW radiates at exactly the plan's
``ω``. SRW reports ``E`` in ``√(photons/s/0.1%bw/mm²)`` for the beam current
``I``; one electron then emits ``d²W/(dω dΩ) = 10⁹ ħ (e/I) D² |E|²`` (SI,
``D`` in metres), so ``|r Ẽ| = D √(10⁹ π ħ e / (I ε₀ c)) |E|`` with every
constant from :meth:`ElectromagneticScaleContract.si`. SRW returns the
horizontal and vertical components; the longitudinal one follows from
far-field transversality ``n · Ẽ = 0`` before projection onto the plan's
``(e1, e2)``. SRW's phase is paraxial,
``k[z/(2γ²) + ½∫β⊥² dz + |P⊥ − r⊥|²/(2(P_z − z))]``: its clock reads zero
where uniform motion at the initial velocity crosses ``z = 0`` and its Fresnel
term leaves the constant ``D n⊥²/(2 n_z) − n⊥² o_z/(2 n_z²)`` beside the far
field; both are removed, and SRW's field is ``i r Ẽ``, so the imported phase
refers to the Phydrax retarded time ``τ = t − n·r/c`` of the trajectory
(verified live against :class:`TrajectoryRadiationPlan` to 10⁻² rad). Every
residual difference is listed as an :class:`AdapterLoss` on the result
report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, Literal, TYPE_CHECKING, TypeAlias

import jax.numpy as jnp
import numpy as np

from .._external_runtime import (
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._validation import positive_finite_float, positive_integer
from ..interchange._report import AdapterLoss, AdapterReport, AdapterStatus
from ..typing import parse
from ._trajectory_radiation import ChargedTrajectory, TrajectoryRadiationPlan


if TYPE_CHECKING:
    from ..applications.accelerator._beam import AcceleratorBunch
    from ..applications.accelerator._field_map import FieldMapTrackingPlan


SRWFieldRepresentation: TypeAlias = Literal["tabulated", "ideal-undulator"]

_OUTPUT = "srw_field.json"
_SOURCE_FORMAT = "srw-electric-field"
_TARGET_FORMAT = "phydrax-far-field-spectrum"
# SRW normalizes the field to this beam current; any value cancels in the
# single-electron conversion.
_CURRENT = 1.0
# ``√(photons/s/0.1%bw/mm²)``, SRW's ``unitElFld = 1``.
_FIELD_UNIT = 1
# SRW's engine converts photon energy to wavenumber with its own constant,
# k = 2π · 0.80654658 µm⁻¹ · E/eV (srradint.cpp), 9.7 ppm below CODATA 2022.
# Requesting E = ω/(c · k_per_eV) makes SRW radiate at exactly the plan's ω.
_SRW_WAVENUMBER_PER_ELECTRONVOLT = 2.0 * math.pi * 0.80654658e6
_UNIFORM_TOLERANCE = 1.0e-9
_DOCUMENT_KEYS = frozenset(
    ("srwpy_version", "current", "origin", "time_origin", "distance", "wavefronts")
)
_WAVEFRONT_KEYS = frozenset(("point", "energies", "unit", "ex", "ey"))
_IDENTITY_TOLERANCE = 1.0e-12

_DRIVER = b"""
import importlib.metadata, json, sys
from array import array
from srwpy import srwlpy
from srwpy.srwlib import (
    SRWLMagFld3D, SRWLMagFldC, SRWLMagFldH, SRWLMagFldU, SRWLPartBeam,
    SRWLParticle, SRWLPrtTrj, SRWLWfr,
)

deck = json.load(open(sys.argv[1]))
electron = deck["electron"]
source = deck["source"]


def particle():
    return SRWLParticle(
        electron["x"], electron["y"], electron["z"],
        electron["beta_x"], electron["beta_y"], electron["gamma"], 1.0, -1.0,
    )


trajectory = 0
field = 0
if source["kind"] == "trajectory":
    trajectory = SRWLPrtTrj()
    trajectory.allocate(len(source["z"]))
    for name, key in (("arX", "x"), ("arXp", "beta_x"), ("arY", "y"),
                      ("arYp", "beta_y"), ("arZ", "z"), ("arZp", "beta_z")):
        setattr(trajectory, name, array("d", source[key]))
    trajectory.ctStart = source["ct_start"]
    trajectory.ctEnd = source["ct_end"]
    trajectory.partInitCond = particle()
elif source["kind"] == "tabulated":
    shape = source["shape"]
    element = SRWLMagFld3D(
        array("d", source["bx"]), array("d", source["by"]), array("d", source["bz"]),
        shape[0], shape[1], shape[2], *source["ranges"], 1, source["interpolation"],
    )
    field = SRWLMagFldC([element], *(array("d", [value]) for value in source["center"]))
elif source["kind"] == "ideal-undulator":
    harmonics = [SRWLMagFldH(1, plane, amplitude, 0.0, symmetry, 1.0)
                 for plane, amplitude, symmetry in source["harmonics"]]
    element = SRWLMagFldU(harmonics, source["period"], source["period_count"])
    field = SRWLMagFldC([element], *(array("d", [value]) for value in source["center"]))
else:
    raise ValueError("unknown SRW source kind")

precision = [1, deck["relative_precision"], deck["z_start"], deck["z_end"],
             deck["trajectory_points"], 1, 0]
wavefronts = []
for point in deck["points"]:
    wavefront = SRWLWfr()
    wavefront.allocate(deck["energy_count"], 1, 1)
    mesh = wavefront.mesh
    mesh.eStart, mesh.eFin = deck["energy_start"], deck["energy_end"]
    mesh.xStart = mesh.xFin = point[0]
    mesh.yStart = mesh.yFin = point[1]
    mesh.zStart = point[2]
    beam = SRWLPartBeam()
    beam.Iavg = deck["current"]
    beam.partStatMom1 = particle()
    wavefront.partBeam = beam
    srwlpy.CalcElecFieldSR(wavefront, trajectory, field, precision)
    mesh = wavefront.mesh
    wavefronts.append({
        "point": [mesh.xStart, mesh.yStart, mesh.zStart],
        "energies": [mesh.eStart, mesh.eFin, mesh.ne],
        "unit": wavefront.unitElFld,
        "ex": list(wavefront.arEx),
        "ey": list(wavefront.arEy),
    })
json.dump(
    {
        "srwpy_version": importlib.metadata.version("srwpy"),
        "current": deck["current"],
        "origin": deck["origin"],
        "time_origin": deck["time_origin"],
        "distance": deck["distance"],
        "wavefronts": wavefronts,
    },
    open("srw_field.json", "w"),
)
"""


@dataclass(frozen=True, slots=True)
class SRWProvider:
    """A pinned Python interpreter with ``srwpy`` installed (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable Python interpreter.")


@dataclass(frozen=True, slots=True)
class SRWFieldMapSource:
    """One electron entering a field-map beamline, for SRW's own trajectory.

    ``tracking`` supplies the beamline, the entrance and exit planes (SRW's
    integration range), the coordinate convention, and the arrival time
    ``reference_time`` at the entrance plane; ``bunch`` holds exactly one
    active electron at ``ζ = 0``. ``"tabulated"`` samples the beamline on
    ``longitudinal_samples`` uniform planes from entrance to exit and a
    ``transverse_samples`` grid over ``±transverse_half_extent`` (one sample
    per axis tabulates the on-axis field only); SRW interpolates the table
    cubically. ``"ideal-undulator"`` maps a beamline of one insertion device
    onto SRW's ideal undulator of the same peak field, period, and period
    count.
    """

    tracking: FieldMapTrackingPlan
    bunch: AcceleratorBunch
    representation: SRWFieldRepresentation = "tabulated"
    longitudinal_samples: int = 4096
    transverse_samples: tuple[int, int] = (1, 1)
    transverse_half_extent: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        from ..applications.accelerator._beam import AcceleratorBunch
        from ..applications.accelerator._field_map import FieldMapTrackingPlan

        if not isinstance(self.tracking, FieldMapTrackingPlan):
            raise TypeError("tracking must be a FieldMapTrackingPlan.")
        if not isinstance(self.bunch, AcceleratorBunch):
            raise TypeError("bunch must be an AcceleratorBunch.")
        parse(self.representation, SRWFieldRepresentation, "representation")
        if positive_integer(self.longitudinal_samples, "longitudinal_samples") < 4:
            raise ValueError("longitudinal_samples must be at least four.")
        if len(self.transverse_samples) != 2 or len(self.transverse_half_extent) != 2:
            raise ValueError("Transverse sampling needs one entry per transverse axis.")
        for count, extent in zip(
            self.transverse_samples, self.transverse_half_extent, strict=True
        ):
            count_ = positive_integer(count, "transverse_samples")
            if count_ == 1:
                if extent != 0.0:
                    raise ValueError("A single transverse sample has zero extent.")
            elif count_ < 4:
                raise ValueError("Transverse tables need one or at least four samples.")
            else:
                positive_finite_float(extent, "transverse_half_extent")


@dataclass(frozen=True, slots=True)
class SRWFieldSpectrum:
    """SRW's single-electron field in the far-field spectrum convention.

    ``field_spectrum[F, D, 2]`` is ``r Ẽ`` in the plan's observer basis
    ``(e1, e2)`` and ``spectral_energy[F, D]`` is ``ε₀ c |r Ẽ|² / π``, both in
    the plan's scale units at ``angular_frequencies[F]`` and ``directions[D]``;
    ``observation_distance`` is ``D`` in the scale's length unit.
    """

    angular_frequencies: np.ndarray
    directions: np.ndarray
    field_spectrum: np.ndarray
    spectral_energy: np.ndarray
    observation_distance: float


@dataclass(frozen=True, slots=True)
class SRWSpectrumResult:
    """SRW's spectrum with the pinned run identity and the adapter report.

    ``output_sha256`` is the digest of SRW's field artifact (the report's
    ``source_id``); the report's ``target_id`` fingerprints the imported
    arrays, and its losses enumerate every declared approximation.
    """

    spectrum: SRWFieldSpectrum
    provider_version: str
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


@dataclass(frozen=True, slots=True)
class _Units:
    length: float
    time: float
    magnetic: float
    speed_of_light: float
    elementary_charge: float
    rest_energy: float


@dataclass(frozen=True, slots=True)
class _Source:
    """SRW field or trajectory deck, initial electron, and SI integration range."""

    field: dict[str, object]
    electron: dict[str, float]
    z_range: tuple[float, float]
    start_time: float


def _units(scale: ElectromagneticScaleContract, /) -> _Units:
    """Scale-unit SI factors and scale-unit electron constants."""
    factors = scale.unit_si_map()
    light = float(scale.speed_of_light)
    return _Units(
        factors["length"][0],
        factors["time"][0],
        factors["magnetic_field"][0],
        light,
        float(scale.elementary_charge),
        float(scale.electron_mass) * light**2,
    )


def _same(value: float, reference: float, /) -> bool:
    return math.isclose(value, reference, rel_tol=_IDENTITY_TOLERANCE)


def _photon_energies(plan: TrajectoryRadiationPlan, /) -> np.ndarray:
    """SRW photon energies whose SRW wavenumbers are the plan's ``ω/c``."""
    light = float(ElectromagneticScaleContract.si().speed_of_light)
    frequencies = np.asarray(plan.angular_frequencies, dtype=np.float64)
    angular = frequencies / plan.scale.unit_si_map()["time"][0]
    return angular / (light * _SRW_WAVENUMBER_PER_ELECTRONVOLT)


def _check_plan(plan: TrajectoryRadiationPlan, /) -> _Units:
    if not isinstance(plan, TrajectoryRadiationPlan):
        raise TypeError("plan must be a TrajectoryRadiationPlan.")
    match plan.coherence:
        case "coherent" | "incoherent":
            pass
        case "gaussian-form-factor" | "tabulated-form-factor":
            raise ValueError(
                "The SRW oracle computes one electron without a form factor."
            )
        case _:
            assert_never(plan.coherence)
    frequencies = np.asarray(plan.angular_frequencies, dtype=np.float64)
    if frequencies.shape[0] > 1:
        steps = np.diff(frequencies)
        if np.max(np.abs(steps - steps[0])) > _UNIFORM_TOLERANCE * steps[0]:
            raise ValueError("SRW needs uniformly spaced angular frequencies.")
    if np.any(np.asarray(plan.observers.directions)[:, 2] <= 0.0):
        raise ValueError("SRW observes forward directions (n_z > 0) only.")
    return _units(plan.scale)


def _trajectory_source(trajectory: ChargedTrajectory, units: _Units, /) -> _Source:
    if trajectory.particle_count != 1:
        raise ValueError("The SRW oracle radiates one trajectory lane.")
    if not bool(np.all(np.asarray(trajectory.active))):
        raise ValueError("The SRW lane must be active at every sample.")
    if not _same(float(trajectory.multiplicities[0]), 1.0):
        raise ValueError("The SRW lane must have multiplicity one.")
    if not _same(float(trajectory.charges[0]), -units.elementary_charge):
        raise ValueError("The SRW oracle radiates one electron (charge −e).")
    times = np.asarray(trajectory.times[:, 0], dtype=np.float64)
    if times.shape[0] < 3:
        raise ValueError("The SRW lane needs at least three samples.")
    steps = np.diff(times)
    if (
        steps[0] <= 0.0
        or np.max(np.abs(steps - steps[0])) > _UNIFORM_TOLERANCE * steps[0]
    ):
        raise ValueError("SRW needs uniform increasing lab-time samples.")
    positions = np.asarray(trajectory.positions[:, 0], dtype=np.float64) * units.length
    proper = np.asarray(trajectory.proper_velocities[:, 0], dtype=np.float64)
    gamma = np.sqrt(1.0 + np.sum(proper * proper, axis=-1) / units.speed_of_light**2)
    beta = proper / (gamma[:, None] * units.speed_of_light)
    # SRW starts c·t at zero on the initial-condition sample.
    path = units.speed_of_light * (times - times[0]) * units.length
    field: dict[str, object] = {
        "kind": "trajectory",
        "ct_start": 0.0,
        "ct_end": float(path[-1]),
        "x": positions[:, 0].tolist(),
        "y": positions[:, 1].tolist(),
        "z": positions[:, 2].tolist(),
        "beta_x": beta[:, 0].tolist(),
        "beta_y": beta[:, 1].tolist(),
        "beta_z": beta[:, 2].tolist(),
    }
    electron = {
        "x": float(positions[0, 0]),
        "y": float(positions[0, 1]),
        "z": float(positions[0, 2]),
        "beta_x": float(beta[0, 0]),
        "beta_y": float(beta[0, 1]),
        "gamma": float(gamma[0]),
    }
    return _Source(
        field,
        electron,
        (float(positions[0, 2]), float(positions[-1, 2])),
        float(times[0]) * units.time,
    )


def _field_map_electron(source: SRWFieldMapSource, units: _Units, /) -> dict[str, float]:
    bunch = source.bunch
    if bunch.capacity != 1 or not bool(bunch.active[0]) or not bool(bunch.valid[0]):
        raise ValueError("The SRW oracle needs a bunch of exactly one active electron.")
    if not _same(float(bunch.weights[0]), 1.0):
        raise ValueError("The SRW electron must have weight one.")
    if not _same(float(bunch.reference_charge), -units.elementary_charge):
        raise ValueError("The SRW oracle radiates one electron (charge −e).")
    rest = float(bunch.reference_rest_energy)
    if not _same(rest, units.rest_energy):
        raise ValueError("The SRW oracle radiates one electron (rest energy mₑc²).")
    x, px, y, py, zeta, delta = np.asarray(bunch.coordinates[0], dtype=np.float64)
    if zeta != 0.0:
        raise ValueError("The SRW electron must arrive at zeta = 0.")
    relative = 1.0 + delta
    longitudinal_sq = relative**2 - px * px - py * py
    if longitudinal_sq <= 0.0:
        raise ValueError("The SRW electron must move toward the exit plane.")
    reduced = float(bunch.reference_momentum) * units.speed_of_light * relative / rest
    gamma = math.sqrt(1.0 + reduced**2)
    speed = reduced / gamma
    return {
        "x": float(x) * units.length,
        "y": float(y) * units.length,
        "z": source.tracking.entrance_plane * units.length,
        "beta_x": speed * float(px) / relative,
        "beta_y": speed * float(py) / relative,
        "gamma": gamma,
    }


def _tabulated_field(source: SRWFieldMapSource, units: _Units, /) -> dict[str, object]:
    tracking = source.tracking
    entrance, exit_ = tracking.entrance_plane, tracking.exit_plane
    axes = [
        np.linspace(-extent, extent, count) if count > 1 else np.zeros((1,))
        for count, extent in zip(
            source.transverse_samples, source.transverse_half_extent, strict=True
        )
    ]
    axes.append(np.linspace(entrance, exit_, source.longitudinal_samples))
    # SRW tables run fastest in x, then y, then z.
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)
    sample = tracking.beamline.external_fields(
        jnp.asarray(points), jnp.zeros((points.shape[0],), dtype=jnp.float64)
    )
    if not bool(jnp.all(sample.support)):
        raise ValueError("The SRW field table leaves the beamline field support.")
    if bool(jnp.any(sample.electric != 0.0)):
        raise ValueError("SRW represents static magnetic fields only.")
    magnetic = np.asarray(sample.magnetic, dtype=np.float64) * units.magnetic
    return {
        "kind": "tabulated",
        "shape": [axis.shape[0] for axis in axes],
        "ranges": [float(axis[-1] - axis[0]) * units.length for axis in axes],
        "center": [0.0, 0.0, 0.5 * (entrance + exit_) * units.length],
        "interpolation": 3,
        "bx": magnetic[:, 0].tolist(),
        "by": magnetic[:, 1].tolist(),
        "bz": magnetic[:, 2].tolist(),
    }


def _ideal_undulator(source: SRWFieldMapSource, units: _Units, /) -> dict[str, object]:
    from ..applications.accelerator._field_map import InsertionDeviceField

    elements = source.tracking.beamline.elements
    if len(elements) != 1 or not isinstance(elements[0], InsertionDeviceField):
        raise ValueError("The ideal-undulator representation needs one insertion device.")
    device = elements[0]
    amplitude = device.peak_field * units.magnetic
    # On axis the device field is (B₀ sin k_u u, B₀ cos k_u u, 0) (helical) or
    # (0, B₀ cos k_u u, 0) (planar); SRW symmetry 1 is cos and −1 is sin.
    harmonics: list[list[object]] = [["v", amplitude, 1]]
    match device.polarization:
        case "planar":
            pass
        case "helical":
            harmonics.append(["h", amplitude, -1])
        case _:
            assert_never(device.polarization)
    return {
        "kind": "ideal-undulator",
        "harmonics": harmonics,
        "period": device.period * units.length,
        "period_count": device.period_count,
        "center": [0.0, 0.0, device.center * units.length],
    }


def _field_map_source(
    source: SRWFieldMapSource, plan: TrajectoryRadiationPlan, units: _Units, /
) -> _Source:
    from ..applications.accelerator._field_map import (
        DipoleBendField,
        InsertionDeviceField,
        TabulatedFieldMap,
    )

    beamline = source.tracking.beamline
    if beamline.scale.scale_id != plan.scale.scale_id:
        raise ValueError("The beamline and the radiation plan must share one scale.")
    if source.bunch.convention.convention_id != source.tracking.convention.convention_id:
        raise ValueError("Tracking plan and bunch coordinate conventions differ.")
    for element in beamline.elements:
        if isinstance(element, TabulatedFieldMap) and element.electric is not None:
            raise ValueError("SRW represents static magnetic fields only.")
        if not isinstance(
            element, (InsertionDeviceField, DipoleBendField, TabulatedFieldMap)
        ):
            raise ValueError(
                "SRW beamlines hold insertion devices, dipole bends, and tabulated "
                "magnetic maps only."
            )
    electron = _field_map_electron(source, units)
    match source.representation:
        case "tabulated":
            field = _tabulated_field(source, units)
        case "ideal-undulator":
            field = _ideal_undulator(source, units)
        case _:
            assert_never(source.representation)
    tracking = source.tracking
    return _Source(
        field,
        electron,
        (tracking.entrance_plane * units.length, tracking.exit_plane * units.length),
        tracking.reference_time * units.time,
    )


def srw_input(
    plan: TrajectoryRadiationPlan,
    source: ChargedTrajectory | SRWFieldMapSource,
    /,
    *,
    observation_distance: float,
    relative_precision: float = 1.0e-5,
    trajectory_points: int = 20000,
) -> dict[str, bytes]:
    """Translate the supported subset into SRW's driver and JSON deck.

    ``observation_distance`` ``D`` (scale length) places the SRW observation
    point of direction ``n`` at ``o + D n`` with ``o = (0, 0, z_mid)`` the
    middle of the integration range. ``relative_precision`` and
    ``trajectory_points`` are SRW's automatic-undulator integration tolerance
    and trajectory sampling.
    """
    units = _check_plan(plan)
    distance = positive_finite_float(observation_distance, "observation_distance")
    precision = positive_finite_float(relative_precision, "relative_precision")
    points_ = positive_integer(trajectory_points, "trajectory_points")
    if isinstance(source, ChargedTrajectory):
        deck_source = _trajectory_source(source, units)
    elif isinstance(source, SRWFieldMapSource):
        deck_source = _field_map_source(source, plan, units)
    else:
        raise TypeError("source must be a ChargedTrajectory or an SRWFieldMapSource.")
    z_start, z_end = deck_source.z_range
    electron = deck_source.electron
    # SRW's paraxial phase k[z/(2γ²) + ½∫_{z₀}^{z} β⊥² dz] runs a clock that
    # reads zero where uniform motion at the initial velocity crosses z = 0,
    # t₀ − z₀/(β_z c) for the initial sample (t₀, z₀), to O(z₀/γ⁴).
    longitudinal = math.sqrt(
        1.0 - electron["gamma"] ** -2 - electron["beta_x"] ** 2 - electron["beta_y"] ** 2
    )
    light = float(ElectromagneticScaleContract.si().speed_of_light)
    time_origin = deck_source.start_time - electron["z"] / (longitudinal * light)
    origin = np.array([0.0, 0.0, 0.5 * (z_start + z_end)])
    directions = np.asarray(plan.observers.directions, dtype=np.float64)
    points = origin[None, :] + distance * units.length * directions
    energies = _photon_energies(plan)
    deck = {
        "source": deck_source.field,
        "electron": electron,
        "current": _CURRENT,
        "relative_precision": precision,
        "trajectory_points": points_,
        "z_start": z_start,
        "z_end": z_end,
        "origin": origin.tolist(),
        "time_origin": time_origin,
        "distance": distance * units.length,
        "points": points.tolist(),
        "energy_start": float(energies[0]),
        "energy_end": float(energies[-1]),
        "energy_count": energies.shape[0],
    }
    return {
        "srw_driver.py": _DRIVER,
        "srw_input.json": json.dumps(deck).encode(),
    }


def read_srw_output(data: bytes, plan: TrajectoryRadiationPlan, /) -> SRWFieldSpectrum:
    """Convert SRW's field document for ``plan`` into the far-field convention.

    The document must echo the plan's photon energies and observation points;
    anything else is refused as inconsistent with the requested run.
    """
    units = _check_plan(plan)
    document = json.loads(data)
    if not isinstance(document, dict) or not _DOCUMENT_KEYS <= document.keys():
        raise ValueError("SRW output is not a field document.")
    si = ElectromagneticScaleContract.si()
    light = float(si.speed_of_light)
    distance = float(document["distance"])
    origin = np.asarray(document["origin"], dtype=np.float64)
    current = float(document["current"])
    directions = np.asarray(plan.observers.directions, dtype=np.float64)
    wavefronts = document["wavefronts"]
    if len(wavefronts) != directions.shape[0] or origin.shape != (3,) or distance <= 0.0:
        raise ValueError("SRW output does not match the plan's observers.")
    energies = _photon_energies(plan)
    expected = origin[None, :] + distance * directions
    count = energies.shape[0]
    field = np.empty((count, directions.shape[0], 3), dtype=np.complex128)
    for index, wavefront in enumerate(wavefronts):
        if not isinstance(wavefront, dict) or not _WAVEFRONT_KEYS <= wavefront.keys():
            raise ValueError("SRW output wavefronts are malformed.")
        start, end, mesh_count = wavefront["energies"]
        if (
            mesh_count != count
            or not _same(start, float(energies[0]))
            or not _same(end, float(energies[-1]))
            or not np.allclose(wavefront["point"], expected[index], rtol=1e-12, atol=0.0)
            or wavefront["unit"] != _FIELD_UNIT
        ):
            raise ValueError("SRW output does not match the requested mesh.")
        pairs = [np.asarray(wavefront[key], dtype=np.float64) for key in ("ex", "ey")]
        if any(pair.shape != (2 * count,) for pair in pairs):
            raise ValueError("SRW output field arrays do not match the mesh.")
        horizontal, vertical = (pair[0::2] + 1j * pair[1::2] for pair in pairs)
        normal = directions[index]
        longitudinal = -(normal[0] * horizontal + normal[1] * vertical) / normal[2]
        field[:, index] = np.stack((horizontal, vertical, longitudinal), axis=-1)
    if not np.all(np.isfinite(field)):
        raise ValueError("SRW returned a nonfinite field.")
    angular = energies * light * _SRW_WAVENUMBER_PER_ELECTRONVOLT
    time_origin = float(document["time_origin"])
    # SRW's paraxial phase is ω(t − t_SRW − z/c + |P⊥ − r⊥|²/(2(P_z − z) c)) for
    # the observation point P = o + D n and its clock origin t_SRW. Expanded
    # about o, the Fresnel term leaves the constant D n⊥²/(2 n_z) − n⊥² o_z/(2 n_z²)
    # beside the far-field ω τ. SRW's field is i times the acceleration-form r·Ẽ.
    transverse = np.sum(directions[:, :2] ** 2, axis=1)
    normal = directions[:, 2]
    fresnel = 0.5 * transverse * (distance / normal - origin[2] / normal**2)
    delay = fresnel / light - time_origin
    field = -1j * field * np.exp(-1j * angular[:, None, None] * delay[None, :, None])
    amplitude = distance * math.sqrt(
        1.0e9
        * math.pi
        * float(si.reduced_planck_constant)
        * float(si.elementary_charge)
        / (current * float(si.vacuum_permittivity) * light)
    )
    # r·Ẽ in volt-seconds, then in scale units of electric field × length × time.
    electric = plan.scale.unit_si_map()["electric_field"][0]
    field = field * amplitude / (electric * units.length * units.time)
    basis = np.stack(
        (
            np.asarray(plan.observers.basis_first, dtype=np.float64),
            np.asarray(plan.observers.basis_second, dtype=np.float64),
        ),
        axis=1,
    )
    projected = np.sum(field[:, :, None, :] * basis[None], axis=-1)
    permittivity = float(plan.scale.vacuum_permittivity)
    spectral = (
        permittivity
        * units.speed_of_light
        * np.sum(np.abs(projected) ** 2, axis=-1)
        / math.pi
    )
    return SRWFieldSpectrum(
        np.asarray(plan.angular_frequencies, dtype=np.float64),
        directions,
        projected,
        spectral,
        distance / units.length,
    )


def _losses(
    plan: TrajectoryRadiationPlan,
    source: ChargedTrajectory | SRWFieldMapSource,
    observation_distance: float,
) -> tuple[AdapterLoss, ...]:
    directions = np.asarray(plan.observers.directions, dtype=np.float64)
    obliquity = float(np.max(1.0 - directions[:, 2]))
    losses = [
        AdapterLoss(
            "field_spectrum",
            "import",
            "transformed",
            "SRW evaluates the near field at distance "
            f"{observation_distance:.6g} (scale length) from the source origin; it "
            "is read as the far-field r·Ẽ with r = D, so near-field corrections of "
            "order L²/(λD) in the source length L remain.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "field_spectrum.phase",
            "import",
            "transformed",
            "SRW's paraxial clock and Fresnel propagation phase are mapped to the "
            "exact retarded time τ = t − n·r/c; residual phase errors of order "
            "k z/γ⁴ and k D θ⁴ remain, and cross-polarized components off axis "
            "carry SRW's paraxial error of relative order θ.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "field_spectrum.polarization",
            "import",
            "synthesized",
            "SRW reports only E_x and E_y; E_z is reconstructed from far-field "
            f"transversality n·Ẽ = 0 (largest 1 − n_z = {obliquity:.3g}) before "
            "projection onto (e1, e2).",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "field_spectrum.precision",
            "import",
            "transformed",
            "SRW stores the electric field in single precision and normalizes the "
            "photon flux with its own physical constants; the import applies "
            "CODATA 2022.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "evidence",
            "import",
            "unsupported",
            "SRW reports no sampling, resolution, finiteness, or activity evidence "
            "comparable to TrajectoryRadiationEvidence.",
            changes_interpretation=False,
        ),
    ]
    if isinstance(source, ChargedTrajectory):
        losses.append(
            AdapterLoss(
                "trajectory",
                "export",
                "transformed",
                "SRW interpolates the sampled positions and β on the uniform c·t grid "
                "with its own scheme and integrates the near-field radiation integral "
                "adaptively; Phydrax integrates the same samples by its declared route.",
                changes_interpretation=False,
            )
        )
        return tuple(losses)
    losses.append(
        AdapterLoss(
            "trajectory",
            "export",
            "transformed",
            "SRW integrates its own trajectory through the exported field from the "
            "entrance plane; Phydrax tracks the electron with its relativistic pusher.",
            changes_interpretation=False,
        )
    )
    match source.representation:
        case "tabulated":
            losses.append(
                AdapterLoss(
                    "field",
                    "export",
                    "transformed",
                    "The beamline field is sampled on a uniform "
                    f"{source.transverse_samples[0]}×{source.transverse_samples[1]}×"
                    f"{source.longitudinal_samples} table between the entrance and "
                    "exit planes and interpolated cubically by SRW; the field outside "
                    "the table is zero.",
                    changes_interpretation=False,
                )
            )
            if source.transverse_samples == (1, 1):
                losses.append(
                    AdapterLoss(
                        "field.transverse",
                        "export",
                        "dropped",
                        "The table holds the on-axis field, which SRW applies at "
                        "every transverse offset.",
                        changes_interpretation=True,
                    )
                )
        case "ideal-undulator":
            losses.append(
                AdapterLoss(
                    "field",
                    "export",
                    "transformed",
                    "The insertion device's smooth tanh terminations are replaced by "
                    "SRW's ideal-undulator ends of the same peak field, period, and "
                    "period count; other beamline field is absent.",
                    changes_interpretation=True,
                )
            )
        case _:
            assert_never(source.representation)
    return tuple(losses)


def run_srw(
    provider: SRWProvider,
    plan: TrajectoryRadiationPlan,
    source: ChargedTrajectory | SRWFieldMapSource,
    destination: str | Path,
    /,
    *,
    observation_distance: float,
    relative_precision: float = 1.0e-5,
    trajectory_points: int = 20000,
    timeout: float = 600.0,
    maximum_output_bytes: int = 1 << 28,
) -> SRWSpectrumResult:
    """Run pinned SRW on the translated case and import its field spectrum."""
    if not isinstance(provider, SRWProvider):
        raise TypeError("provider must be an SRWProvider.")
    inputs = srw_input(
        plan,
        source,
        observation_distance=observation_distance,
        relative_precision=relative_precision,
        trajectory_points=trajectory_points,
    )
    artifacts = PinnedFileOutputs(
        str(destination),
        (PinnedFileRequest(_OUTPUT, maximum_output_bytes),),
        maximum_output_bytes,
    )
    executable = provider.executable
    run = run_pinned_command(
        executable,
        ("srw_driver.py", "srw_input.json"),
        inputs=inputs,
        timeout=timeout,
        artifacts=artifacts,
    ).require_success()
    artifact = run.file_artifact(_OUTPUT)
    data = Path(artifact.location).read_bytes()
    reported = json.loads(data)["srwpy_version"]
    if reported != executable.version:
        raise ValueError(
            f"The pinned interpreter runs srwpy {reported}, not {executable.version}."
        )
    spectrum = read_srw_output(data, plan)
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        _SOURCE_FORMAT,
        _TARGET_FORMAT,
        source_id=artifact.sha256,
        target_id=canonical_fingerprint(
            {
                "kind": "srw-field-spectrum",
                "plan": plan.plan_id,
                "arrays": array_tree_fingerprint(
                    (spectrum.field_spectrum, spectrum.spectral_energy)
                ),
            }
        ),
        coordinate_mapping=(
            "scale length × unitSI -> SRW metres in the field-map frame (z along the axis)",
            "angular frequency ω -> SRW photon energy ω/(c·2π·0.80654658 µm⁻¹)",
            "scale magnetic field × unitSI -> SRW tesla",
            "SRW (E_x, E_y) with E_z from n·Ẽ = 0 -> r·Ẽ in the plan basis (e1, e2)",
        ),
        preserved_fields=("angular_frequencies", "directions"),
        assumptions=(
            "SRW's E is normalized to √(photons/s/0.1%bw/mm²) for the deck current",
            "one electron radiates d²W/(dω dΩ) = 1e9 ħ (e/I) D² |E|² in SI",
            "SRW's field is i·r·Ẽ with SRW's paraxial clock and Fresnel phase",
        ),
        losses=_losses(plan, source, observation_distance),
    )
    return SRWSpectrumResult(
        spectrum,
        executable.version,
        executable.sha256,
        executable.license_id,
        artifact.sha256,
        report,
    )


__all__ = [
    "read_srw_output",
    "run_srw",
    "srw_input",
    "SRWFieldMapSource",
    "SRWFieldRepresentation",
    "SRWFieldSpectrum",
    "SRWProvider",
    "SRWSpectrumResult",
]
