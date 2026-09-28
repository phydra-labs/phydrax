#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned gyrosynchrotron oracles: Symphony and the Ultimate Fast Gyrosynchrotron Codes.

Both providers are GPL-3.0 projects used only as caller-pinned external
libraries run by a caller-pinned Python interpreter through
:func:`phydrax.run_pinned_command`; nothing is imported into this process and
no provider source is copied. The pinned library bytes are digest-checked on
the host and staged into the run directory, where a generated driver loads the
staged copy.

**Symphony** (Pandya, Zhang, Chandra & Gammie 2016, ApJ 822, 34;
https://github.com/AFD-Illinois/symphony; SPDX ``GPL-3.0-only``): the pinned
file is the ``symphonyPy`` extension module; its companion
``susceptibility_tensor`` package and shared libraries are loaded from the
module's directory (the digest pins the module, not its dynamic dependencies).
Validated live against revision ``a869c6b`` (2022-01-14) built with GSL 2.8.
:func:`symphony_input` maps a :class:`MagnetobremsstrahlungPlan` onto
Symphony's CGS arguments (``ν`` in Hz, ``B`` in gauss, electron density in
cm⁻³) and :func:`read_symphony_output` converts its vacuum Stokes emissivities
``j_ν`` (erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹) and absorptivities ``α_ν`` (cm⁻¹) of ``I``,
``Q`` and ``V`` to the plan's scale: ``j_ω = j_ν / 2π`` per unit angular
frequency and steradian, ``α`` per plan length. Symphony's Stokes basis has
``ê₁`` in the ``k``–``B₀`` plane, as in :class:`FaradayCoefficients`, where
``U`` vanishes for gyrotropic populations.

Symphony supported subset (everything else is refused with ``ValueError``
before running): electron emitters (charge ``−e``, mass ``mₑ``) in a plasma
whose scale is referenced to SI; wave-normal angles in ``(0, π/2)`` (the
pinned revision's integrator fails at exactly perpendicular propagation);
:class:`ThermalJuttnerDistribution` (``MAXWELL_JUETTNER``, ``θₑ = θ``),
:class:`KappaDistribution` (``KAPPA_DIST``, ``w = θ``, no exponential
cutoff) and :class:`PowerLawDistribution` with ``index > 1`` (``POWER_LAW``,
``p = index``). The pinned revision ignores its ``gamma_min``/``gamma_max``
arguments and integrates ``n_s (p − 1) γ^{−p}`` over ``γ ∈ [1, ∞)``; the adapter
chooses ``n_s = N/(u₁^{1−p} − u₂^{1−p})`` so both populations coincide for
``u ≫ 1`` inside ``[u₁, u₂]`` and declares the rest as an interpretation-changing
loss.

**UFGC** (Kuznetsov & Fleishman 2021, ApJ 922, 103;
https://github.com/kuznetsov-radio/gyrosynchrotron; SPDX ``GPL-3.0-only``): the
pinned file is the ``MWTransferArr`` shared library, called through its
``pyGET_MW_SLICE`` entry point. Validated live against revision ``5e014ba``
(2026-08-28) built with ``make`` (clang++, Homebrew libomp). UFGC returns
intensities after radiative transfer through voxels, so
:func:`read_ufgc_output` recovers the mode coefficients from two single-voxel
lines of sight per angle: an optically thick voxel gives the source function
``S_σ = j_σ/κ_σ``, and a voxel whose optical depth is below the double-precision
unit roundoff gives ``j_σ Δz`` exactly (UFGC then evaluates ``1 − e^{−τ}`` as
``τ``). UFGC fluxes (sfu at 1 au for area ``S``) are converted with its fixed
observer distance ``1 au = 1.495978707 × 10¹³ cm`` and ``1 sfu = 10⁻¹⁹ erg s⁻¹
cm⁻² Hz⁻¹``. Its left/right rows are the ordinary/extraordinary modes for
``θ ≤ π/2`` and the reverse otherwise.

UFGC supported subset: electron emitters in a collisionless single-species
electron plasma whose scale is referenced to SI; angles in ``(0, π)``; the
plan route selects UFGC's exact harmonic code (``"harmonic-sum"``, exact
Bessel functions) or its continuous code (``"continuous-harmonic"``,
``Q``-optimized, adaptive energy nodes, no boundary renormalization);
:class:`ThermalJuttnerDistribution` (``THM``) and :class:`KappaDistribution`
(``KAP`` with ``T₀ = κθ mₑc²/((κ − 3/2) k_B)``, so UFGC's ``(κ − 3/2)θ₀`` equals
``κθ``, and ``E_max`` from ``maximum_momentum``) when the emitters are the
plasma electrons (``emitter_density`` equal to the plasma density), and
:class:`PowerLawDistribution` with ``index ≠ 1`` as UFGC's power law over
momentum (``PLP``, ``f(p) ∝ p^{−(index+2)}`` per ``d³p``) above a background
``n₀ = n_plasma − N`` so that UFGC's ``n₀ + n_b`` is the plasma density.
Free–free emission is switched off.

Every SI constant comes from :meth:`ElectromagneticScaleContract.si` and every
plan-unit factor from the plan scale's :meth:`~ElectromagneticScaleContract.unit_si_map`;
gauss follow from the Gaussian energy-density identity ``B_G²/(8π) [erg cm⁻³] =
B_T²/(2μ₀) [J m⁻³]``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy as np

from .._external_runtime import (
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from ..interchange._report import AdapterLoss, AdapterReport, AdapterStatus
from ..typing import as_host_array, ConvertibleToArray, Dim, HostFloat64, parse
from ..units import (
    CENTIMETER,
    derived_unit,
    MASS,
    SECOND,
    SI_REFERENCE_SYSTEM_ID,
    UnitDefinition,
)
from ._cold_plasma import PlasmaWaveMode
from ._magnetobremsstrahlung import (
    KappaDistribution,
    MagnetobremsstrahlungPlan,
    PowerLawDistribution,
    ThermalJuttnerDistribution,
)


class _FrequencyDim(Dim, minimum=1):
    """Angular frequencies of a provider evaluation."""


class _AngleDim(Dim, minimum=1):
    """Wave-normal angles of a provider evaluation."""


_GRAM = UnitDefinition("g", MASS, SI_REFERENCE_SYSTEM_ID, "0.001")
_ERG = derived_unit("erg", ((_GRAM, 1), (CENTIMETER, 2), (SECOND, -2)))
# UFGC's fixed observer distance (1 au, IAU 2012 B2) and flux unit (sfu, cgs).
_UFGC_OBSERVER_DISTANCE_CM = 1.495978707e13
_UFGC_SFU_CGS = 1.0e-19
# Optically thick and thin voxel depths (cm) of the two UFGC lines of sight.
_THICK_DEPTH = 1.0e300
_THIN_DEPTH = 1.0e-25
# Symphony's exponential cutoff γ_cut is disabled by an unreachable value.
_SYMPHONY_NO_CUTOFF = 1.0e300
_DENSITY_TOLERANCE = 1.0e-12
_UFGC_OUTPUT = "ufgc_output.json"
_SYMPHONY_OUTPUT = "symphony_output.json"
_MODES = (PlasmaWaveMode.ORDINARY, PlasmaWaveMode.EXTRAORDINARY)

_UFGC_DRIVER = b"""
import ctypes, json, sys
import numpy as np
from numpy.ctypeslib import ndpointer

deck = json.load(open(sys.argv[1]))
library = ctypes.CDLL("./" + deck["library"])
function = library.pyGET_MW_SLICE
real = ndpointer(dtype=np.float64, flags="F")
function.argtypes = [ndpointer(dtype=np.int32, flags="F")] + [real] * 6
function.restype = ctypes.c_double
integers = np.asfortranarray(np.asarray(deck["integer_parameters"], dtype=np.int32))
reals = np.asfortranarray(np.asarray(deck["pixel_parameters"], dtype=np.float64).T)
voxels = np.asfortranarray(np.asarray(deck["voxel_parameters"], dtype=np.float64).T)
voxels = voxels.reshape((voxels.shape[0], 1, voxels.shape[1]), order="F")
pixels = voxels.shape[2]
frequencies = np.asarray(deck["frequencies_ghz"], dtype=np.float64)
empty = np.zeros(1, dtype=np.float64, order="F")
distribution = np.zeros((1, 1, 1, pixels), dtype=np.float64, order="F")
rows = np.zeros((7, frequencies.size, pixels), dtype=np.float64, order="F")
rows[0] = frequencies[:, None]
status = function(integers, reals, voxels, empty, empty, distribution, rows)
json.dump(
    {
        "status": status,
        "frequencies_ghz": rows[0, :, 0].tolist(),
        "left": rows[1].T.tolist(),
        "right": rows[2].T.tolist(),
        "echo": deck["echo"],
    },
    open("ufgc_output.json", "w"),
)
"""

_SYMPHONY_DRIVER = b"""
import json, os, sys
deck = json.load(open(sys.argv[1]))
sys.path.append(deck["module_directory"])
import symphonyPy as symphony

# The staged, digest-checked module must shadow the one in module_directory.
if os.path.dirname(os.path.abspath(symphony.__file__)) != os.getcwd():
    raise SystemExit("symphonyPy was not loaded from the staged module.")

kinds = {
    "maxwell-juettner": symphony.MAXWELL_JUETTNER,
    "power-law": symphony.POWER_LAW,
    "kappa": symphony.KAPPA_DIST,
}
model = deck["distribution"]
kind = kinds[model["kind"]]
tail = (
    model["theta_e"],
    model["power_law_p"],
    model["gamma_min"],
    model["gamma_max"],
    model["gamma_cutoff"],
    model["kappa"],
    model["kappa_width"],
)
stokes = (symphony.STOKES_I, symphony.STOKES_Q, symphony.STOKES_V)
emission = []
absorption = []
for angle in deck["angles"]:
    for frequency in deck["frequencies_hz"]:
        head = (frequency, deck["magnetic_field_gauss"], deck["electron_density_cgs"], angle, kind)
        emission.append([symphony.j_nu_py(*head, key, *tail) for key in stokes])
        absorption.append(
            [symphony.alpha_nu_py(*head, key, *tail, symphony.SYMPHONY_METHOD) for key in stokes]
        )
json.dump(
    {"emission": emission, "absorption": absorption, "echo": deck["echo"]},
    open("symphony_output.json", "w"),
)
"""


def _pinned(value: object, name: str, /) -> PinnedExecutable:
    if not isinstance(value, PinnedExecutable):
        raise TypeError(f"{name} must be a PinnedExecutable.")
    return value


@dataclass(frozen=True, slots=True)
class SymphonyProvider:
    """A pinned Python interpreter and a pinned ``symphonyPy`` extension module."""

    executable: PinnedExecutable
    module: PinnedExecutable

    def __post_init__(self) -> None:
        _pinned(self.executable, "executable")
        _pinned(self.module, "module")


@dataclass(frozen=True, slots=True)
class UFGCProvider:
    """A pinned Python interpreter (with NumPy) and a pinned UFGC shared library."""

    executable: PinnedExecutable
    library: PinnedExecutable

    def __post_init__(self) -> None:
        _pinned(self.executable, "executable")
        _pinned(self.library, "library")


@dataclass(frozen=True, slots=True)
class SymphonyCoefficients:
    """Symphony's vacuum Stokes coefficients in the plan's scale units.

    ``stokes_emission[A, F, 4]`` is ``(j_I, j_Q, j_U, j_V)`` per unit volume,
    angular frequency and steradian and ``stokes_absorption[A, F, 4]`` is
    ``(α_I, α_Q, α_U, α_V)`` per unit length (the first row of
    `MagnetobremsstrahlungResult.propagation_matrix`), at
    ``angles[A]`` and ``angular_frequencies[F]``.
    """

    angular_frequencies: np.ndarray
    angles: np.ndarray
    stokes_emission: np.ndarray
    stokes_absorption: np.ndarray


@dataclass(frozen=True, slots=True)
class UFGCCoefficients:
    """UFGC's mode emission and absorption coefficients in the plan's scale units.

    ``emission[A, F, 2]`` is ``j_σ`` per unit volume, angular frequency and
    steradian and ``absorption[A, F, 2]`` is ``κ_σ`` per unit length at
    ``angles[A]`` and ``angular_frequencies[F]``; the trailing axis is
    (ordinary, extraordinary). A mode UFGC does not propagate carries zeros.
    """

    angular_frequencies: np.ndarray
    angles: np.ndarray
    emission: np.ndarray
    absorption: np.ndarray

    def select(self, mode: PlasmaWaveMode, values: np.ndarray, /) -> np.ndarray:
        """Pick the ``mode`` column of a ``[..., 2]`` (ordinary, extraordinary) array."""
        return np.asarray(values)[..., _MODES.index(parse(mode, PlasmaWaveMode, "mode"))]


@dataclass(frozen=True, slots=True)
class SymphonyResult:
    """Symphony coefficients with the pinned run identity and the adapter report.

    ``provider_version``, ``executable_sha256`` and ``license_id`` identify the
    pinned ``symphonyPy`` module; ``output_sha256`` is the output artifact
    digest (the report's ``source_id``).
    """

    coefficients: SymphonyCoefficients
    provider_version: str
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


@dataclass(frozen=True, slots=True)
class UFGCResult:
    """UFGC coefficients with the pinned run identity and the adapter report.

    ``provider_version``, ``executable_sha256`` and ``license_id`` identify the
    pinned UFGC library; ``output_sha256`` is the output artifact digest (the
    report's ``source_id``).
    """

    coefficients: UFGCCoefficients
    provider_version: str
    executable_sha256: str
    license_id: str
    output_sha256: str
    report: AdapterReport


@dataclass(frozen=True, slots=True)
class _Units:
    """SI values of the plan's units and the provider CGS units."""

    time: float
    length: float
    energy: float
    magnetic_field: float
    centimeter: float
    erg: float
    gauss_per_tesla: float
    rest_energy: float
    boltzmann: float
    megaelectronvolt: float


def _units(scale: ElectromagneticScaleContract, /) -> _Units:
    factors = scale.unit_si_map()
    si = ElectromagneticScaleContract.si()
    centimeter = float(CENTIMETER.scale_to_reference)
    erg = float(_ERG.scale_to_reference)
    light = float(si.speed_of_light)
    return _Units(
        time=factors["time"][0],
        length=factors["length"][0],
        energy=factors["energy"][0],
        magnetic_field=factors["magnetic_field"][0],
        centimeter=centimeter,
        erg=erg,
        gauss_per_tesla=math.sqrt(
            4.0 * math.pi * centimeter**3 / (erg * float(si.vacuum_permeability))
        ),
        rest_energy=float(si.electron_mass) * light * light,
        boltzmann=float(si.relativity.boltzmann_constant),
        megaelectronvolt=1.0e6 * float(si.elementary_charge),
    )


def _electron_plan(plan: object, /) -> MagnetobremsstrahlungPlan:
    if not isinstance(plan, MagnetobremsstrahlungPlan):
        raise TypeError("plan must be a MagnetobremsstrahlungPlan.")
    if plan.emitter_charge_number != -1.0 or plan.emitter_mass_ratio != 1.0:
        raise ValueError("Gyrosynchrotron providers model electron emitters only.")
    return plan


def _evaluation_grid(
    angular_frequencies: ConvertibleToArray,
    angles: ConvertibleToArray,
    upper_angle: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    omega = as_host_array(
        angular_frequencies, HostFloat64[_FrequencyDim], "angular_frequencies"
    )
    theta = as_host_array(angles, HostFloat64[_AngleDim], "angles")
    if not np.all(np.isfinite(omega)) or np.any(omega <= 0.0):
        raise ValueError("angular_frequencies must be finite and positive.")
    if np.any(np.diff(omega) <= 0.0):
        raise ValueError("angular_frequencies must be strictly increasing.")
    if (
        not np.all(np.isfinite(theta))
        or np.any(theta <= 0.0)
        or np.any(theta >= upper_angle)
    ):
        raise ValueError(f"angles must lie in (0, {upper_angle:.6g}) radians.")
    return np.array(omega, dtype=np.float64), np.array(theta, dtype=np.float64)


def _plasma_electron_density(plan: MagnetobremsstrahlungPlan, /) -> float:
    """Density of the single collisionless electron species UFGC can represent."""
    plasma = plan.plasma
    if (
        plasma.species_count != 1
        or float(plasma.charge_numbers[0]) != -1.0
        or float(plasma.mass_ratios[0]) != 1.0
    ):
        raise ValueError("UFGC represents a plasma of one electron species only.")
    if float(plasma.collision_frequencies[0]) != 0.0:
        raise ValueError("UFGC represents a collisionless plasma only.")
    return float(plasma.densities[0])


def _magnetic_field_gauss(plan: MagnetobremsstrahlungPlan, units: _Units, /) -> float:
    field = float(plan.plasma.magnetic_field_magnitude) * units.magnetic_field
    if not field > 0.0:
        raise ValueError("Gyrosynchrotron providers need a nonzero magnetic field.")
    return field * units.gauss_per_tesla


def _kinetic_energy(momentum: float, units: _Units, /) -> float:
    """Kinetic energy ``(γ − 1) mₑc²`` of normalized momentum ``u`` in MeV."""
    return (
        momentum
        * momentum
        / (1.0 + math.hypot(1.0, momentum))
        * (units.rest_energy / units.megaelectronvolt)
    )


def _same_density(first: float, second: float, /) -> bool:
    return math.isclose(first, second, rel_tol=_DENSITY_TOLERANCE)


def _coefficient_scales(units: _Units, /) -> tuple[float, float]:
    """Plan-unit factors of cgs ``j_ν`` (to ``j_ω``) and cgs ``α``."""
    emission = (
        units.erg
        / units.centimeter**3
        / (2.0 * math.pi)
        / (units.energy / units.length**3)
    )
    return emission, units.length / units.centimeter


def _frequencies_hz(omega: np.ndarray, units: _Units, /) -> np.ndarray:
    return omega / units.time / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Symphony
# ---------------------------------------------------------------------------


def _symphony_model(kind: str, /, **values: float) -> dict[str, float | str]:
    """Symphony's distribution arguments; unused ones keep inert placeholders."""
    model: dict[str, float | str] = {
        "kind": kind,
        "theta_e": 0.0,
        "power_law_p": 0.0,
        "gamma_min": 1.0,
        "gamma_max": 1.0,
        "gamma_cutoff": _SYMPHONY_NO_CUTOFF,
        "kappa": 0.0,
        "kappa_width": 0.0,
    }
    model.update(values)
    return model


def _symphony_distribution(
    plan: MagnetobremsstrahlungPlan, density_cgs: float, /
) -> tuple[dict[str, float | str], float]:
    """Symphony distribution arguments and its electron density (cm⁻³)."""
    distribution = plan.distribution
    match distribution:
        case ThermalJuttnerDistribution():
            theta = float(distribution.temperature)
            return _symphony_model("maxwell-juettner", theta_e=theta), density_cgs
        case KappaDistribution():
            model = _symphony_model(
                "kappa",
                kappa=float(distribution.kappa),
                kappa_width=float(distribution.temperature),
            )
            return model, density_cgs
        case PowerLawDistribution():
            index = float(distribution.index)
            if not index > 1.0:
                raise ValueError("Symphony's power law needs index > 1.")
            lower = float(distribution.minimum_momentum)
            upper = float(distribution.maximum_momentum)
            model = _symphony_model(
                "power-law",
                power_law_p=index,
                gamma_min=math.hypot(1.0, lower),
                gamma_max=math.hypot(1.0, upper),
            )
            return model, density_cgs / (lower ** (1.0 - index) - upper ** (1.0 - index))
        case _:
            raise ValueError(
                "Symphony supports thermal Jüttner, kappa and power-law distributions."
            )


def symphony_input(
    plan: MagnetobremsstrahlungPlan,
    /,
    *,
    angular_frequencies: ConvertibleToArray,
    angles: ConvertibleToArray,
    module_directory: str = ".",
) -> dict[str, bytes]:
    """Translate the supported subset into Symphony's driver and JSON deck.

    ``angular_frequencies`` (strictly increasing, plan time unit⁻¹) and
    ``angles`` (radians in ``(0, π/2)``) span the evaluation grid.
    ``module_directory`` holds the companion ``susceptibility_tensor`` package
    of the staged ``symphonyPy`` module.
    """
    plan = _electron_plan(plan)
    omega, theta = _evaluation_grid(angular_frequencies, angles, 0.5 * math.pi)
    units = _units(plan.plasma.scale)
    density = float(plan.emitter_density) / units.length**3 * units.centimeter**3
    if not density > 0.0:
        raise ValueError("Symphony needs a positive emitter density.")
    model, electron_density = _symphony_distribution(plan, density)
    deck = {
        "module_directory": module_directory,
        "frequencies_hz": _frequencies_hz(omega, units).tolist(),
        "angles": theta.tolist(),
        "magnetic_field_gauss": _magnetic_field_gauss(plan, units),
        "electron_density_cgs": electron_density,
        "distribution": model,
        "echo": {"angular_frequencies": omega.tolist(), "angles": theta.tolist()},
    }
    return {
        "symphony_driver.py": _SYMPHONY_DRIVER,
        "symphony_input.json": json.dumps(deck).encode(),
    }


def read_symphony_output(
    data: bytes, plan: MagnetobremsstrahlungPlan, /
) -> SymphonyCoefficients:
    """Convert Symphony's cgs ``(I, Q, V)`` coefficients to the plan's scale."""
    plan = _electron_plan(plan)
    document = json.loads(data)
    omega = np.asarray(document["echo"]["angular_frequencies"], dtype=np.float64)
    theta = np.asarray(document["echo"]["angles"], dtype=np.float64)
    shape = (theta.size, omega.size, 3)
    emission = np.asarray(document["emission"], dtype=np.float64)
    absorption = np.asarray(document["absorption"], dtype=np.float64)
    if emission.size != math.prod(shape) or absorption.size != math.prod(shape):
        raise ValueError("Symphony output does not match its evaluation grid.")
    if not (np.all(np.isfinite(emission)) and np.all(np.isfinite(absorption))):
        raise ValueError("Symphony returned nonfinite coefficients.")
    emission_scale, absorption_scale = _coefficient_scales(_units(plan.plasma.scale))
    # Stokes U vanishes identically in the basis with ê₁ in the k–B₀ plane.
    zeros = np.zeros(shape[:2] + (1,), dtype=np.float64)
    emission = emission.reshape(shape) * emission_scale
    absorption = absorption.reshape(shape) * absorption_scale
    return SymphonyCoefficients(
        omega,
        theta,
        np.concatenate((emission[..., :2], zeros, emission[..., 2:]), axis=-1),
        np.concatenate((absorption[..., :2], zeros, absorption[..., 2:]), axis=-1),
    )


def _symphony_losses(plan: MagnetobremsstrahlungPlan, /) -> tuple[AdapterLoss, ...]:
    losses = [
        AdapterLoss(
            "plasma",
            "export",
            "dropped",
            "Symphony radiates into vacuum (n = 1): the cold-plasma refractive "
            "indices, mode polarizations and Faraday rotation/conversion of the "
            "plan's dielectric are not represented.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "route",
            "export",
            "transformed",
            f"The plan route {plan.route!r} and its quadrature/harmonic resources "
            "are not transmitted: Symphony sums harmonics n ≤ 30 and integrates "
            "over n beyond, with GSL QAG at relative tolerance 1e-3.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "constants",
            "export",
            "transformed",
            "Symphony evaluates ν_c, prefactors and distributions with its own "
            "CGS constants; the deck uses CODATA 2022.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "stokes.U",
            "import",
            "synthesized",
            "Symphony's U coefficients vanish identically in its basis (ê₁ in the "
            "k–B₀ plane) and are filled with zeros.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "evidence",
            "import",
            "unsupported",
            "Symphony reports no status, harmonic range, quadrature error or tail "
            "evidence comparable to MagnetobremsstrahlungResult.",
            changes_interpretation=False,
        ),
    ]
    distribution = plan.distribution
    match distribution:
        case ThermalJuttnerDistribution() | KappaDistribution():
            losses.append(
                AdapterLoss(
                    "distribution.truncation",
                    "export",
                    "dropped",
                    "Symphony integrates the untruncated distribution; the plan omits "
                    f"a mass fraction ≤ {float(distribution.tail_mass_bound()):.3g} "
                    "beyond its momentum support.",
                    changes_interpretation=False,
                )
            )
        case PowerLawDistribution():
            losses.append(
                AdapterLoss(
                    "distribution",
                    "export",
                    "transformed",
                    "The pinned Symphony ignores gamma_min/gamma_max and integrates "
                    "n_s (p − 1) γ^{-p} over γ ∈ [1, ∞); n_s = N/(u₁^{1−p} − u₂^{1−p}) "
                    "matches the plan's dN/du ∝ u^{-p} only for u ≫ 1 inside "
                    "[u₁, u₂], and electrons below u₁ and above u₂ are added.",
                    changes_interpretation=True,
                )
            )
        case _:
            raise ValueError(
                "Symphony supports thermal Jüttner, kappa and power-law distributions."
            )
    return tuple(losses)


def _verified_bytes(pinned: PinnedExecutable, /) -> bytes:
    data = Path(pinned.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != pinned.sha256:
        raise ValueError(f"The pinned file {pinned.path} no longer matches its digest.")
    return data


def _file_artifacts(
    destination: str | Path, name: str, maximum_output_bytes: int, /
) -> PinnedFileOutputs:
    return PinnedFileOutputs(
        str(destination),
        (PinnedFileRequest(name, maximum_output_bytes),),
        maximum_output_bytes,
    )


def run_symphony(
    provider: SymphonyProvider,
    plan: MagnetobremsstrahlungPlan,
    destination: str | Path,
    /,
    *,
    angular_frequencies: ConvertibleToArray,
    angles: ConvertibleToArray,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 26,
) -> SymphonyResult:
    """Run pinned Symphony on the translated plan and import its Stokes coefficients."""
    if not isinstance(provider, SymphonyProvider):
        raise TypeError("provider must be a SymphonyProvider.")
    module = Path(provider.module.path)
    if not module.name.startswith("symphonyPy."):
        raise ValueError("The pinned Symphony module must be a symphonyPy extension.")
    inputs = symphony_input(
        plan,
        angular_frequencies=angular_frequencies,
        angles=angles,
        module_directory=str(module.parent),
    )
    losses = _symphony_losses(plan)
    inputs[module.name] = _verified_bytes(provider.module)
    run = run_pinned_command(
        provider.executable,
        ("symphony_driver.py", "symphony_input.json"),
        inputs=inputs,
        timeout=timeout,
        max_output_bytes=maximum_output_bytes,
        artifacts=_file_artifacts(destination, _SYMPHONY_OUTPUT, maximum_output_bytes),
    ).require_success()
    artifact = run.file_artifact(_SYMPHONY_OUTPUT)
    coefficients = read_symphony_output(Path(artifact.location).read_bytes(), plan)
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "symphony-stokes-coefficients",
        "phydrax-magnetobremsstrahlung-stokes-coefficients",
        source_id=artifact.sha256,
        target_id=canonical_fingerprint(
            {
                "kind": "symphony-stokes-coefficients",
                "plan": plan.plan_id,
                "arrays": array_tree_fingerprint(
                    (
                        coefficients.angular_frequencies,
                        coefficients.angles,
                        coefficients.stokes_emission,
                        coefficients.stokes_absorption,
                    )
                ),
            }
        ),
        coordinate_mapping=(
            "angular frequency ω / (2π time unitSI) -> Symphony ν in Hz",
            "magnetic field × unitSI × √(4π cm³/(μ₀ erg)) -> Symphony B in gauss",
            "emitter density / (length unitSI)³ × cm³ -> Symphony cm⁻³",
            "Symphony j_ν [erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹] / 2π -> j_ω in plan units",
            "Symphony α_ν [cm⁻¹] -> α per plan length",
        ),
        preserved_fields=("angular_frequencies", "angles", "distribution parameters"),
        assumptions=(
            "Symphony's Stokes basis has ê₁ in the k–B₀ plane (IEEE/IAU V)",
            "absorptivities use Symphony's direct integrator (SYMPHONY_METHOD)",
        ),
        losses=losses,
    )
    return SymphonyResult(
        coefficients,
        provider.module.version,
        provider.module.sha256,
        provider.module.license_id,
        artifact.sha256,
        report,
    )


# ---------------------------------------------------------------------------
# UFGC
# ---------------------------------------------------------------------------

# UFGC voxel-parameter indices (CallingConventions.pdf, Parms).
_DEPTH, _T0, _N0, _FIELD, _ANGLE, _EMISSION_FLAGS, _ENERGY_MODEL = 0, 1, 2, 3, 4, 5, 6
_NB, _KAPPA, _E_MIN, _E_MAX, _DELTA, _PITCH_MODEL, _PROTONS, _ARRAY_KEY = (
    7,
    8,
    9,
    10,
    12,
    14,
    18,
    21,
)
_VOXEL_PARAMETERS = 24
# Gyrosynchrotron only: electron–ion (2) and electron–neutral (4) free–free off.
_GYROSYNCHROTRON_ONLY = 6.0
_THM, _KAP, _PLP = 2.0, 6.0, 7.0


def _require_plasma_emitters(emitter_density: float, plasma_density: float, /) -> None:
    if not _same_density(emitter_density, plasma_density):
        raise ValueError(
            "UFGC's thermal and kappa populations are the plasma electrons: "
            "emitter_density must equal the plasma density."
        )


def _ufgc_population(
    plan: MagnetobremsstrahlungPlan, units: _Units, plasma_density: float, /
) -> dict[int, float]:
    """UFGC voxel parameters of the emitting population (cgs densities)."""
    distribution = plan.distribution
    emitter_density = float(plan.emitter_density)
    to_cgs = units.centimeter**3 / units.length**3
    rest_temperature = units.rest_energy / units.boltzmann
    match distribution:
        case ThermalJuttnerDistribution():
            _require_plasma_emitters(emitter_density, plasma_density)
            return {
                _ENERGY_MODEL: _THM,
                _N0: plasma_density * to_cgs,
                _T0: float(distribution.temperature) * rest_temperature,
            }
        case KappaDistribution():
            _require_plasma_emitters(emitter_density, plasma_density)
            kappa = float(distribution.kappa)
            theta = float(distribution.temperature)
            return {
                _ENERGY_MODEL: _KAP,
                _N0: plasma_density * to_cgs,
                _T0: kappa * theta / (kappa - 1.5) * rest_temperature,
                _KAPPA: kappa,
                _E_MAX: _kinetic_energy(float(distribution.maximum_momentum), units),
            }
        case PowerLawDistribution():
            index = float(distribution.index)
            if index == 1.0:
                raise ValueError("UFGC's momentum power law excludes index 1.")
            if emitter_density > plasma_density:
                raise ValueError(
                    "UFGC counts the emitters in its plasma density: emitter_density "
                    "must not exceed the plasma density."
                )
            return {
                _ENERGY_MODEL: _PLP,
                _N0: (plasma_density - emitter_density) * to_cgs,
                _NB: emitter_density * to_cgs,
                _E_MIN: _kinetic_energy(float(distribution.minimum_momentum), units),
                _E_MAX: _kinetic_energy(float(distribution.maximum_momentum), units),
                _DELTA: index + 2.0,
            }
        case _:
            raise ValueError(
                "UFGC supports thermal Jüttner, kappa and power-law distributions."
            )


def ufgc_input(
    plan: MagnetobremsstrahlungPlan,
    /,
    *,
    angular_frequencies: ConvertibleToArray,
    angles: ConvertibleToArray,
    library_name: str = "MWTransferArr.so",
) -> dict[str, bytes]:
    """Translate the supported subset into UFGC's driver and JSON deck.

    Each angle becomes two single-voxel lines of sight (optically thick, then
    optically negligible) evaluated at every angular frequency.
    """
    plan = _electron_plan(plan)
    omega, theta = _evaluation_grid(angular_frequencies, angles, math.pi)
    units = _units(plan.plasma.scale)
    plasma_density = _plasma_electron_density(plan)
    population = _ufgc_population(plan, units, plasma_density)
    field = _magnetic_field_gauss(plan, units)
    frequencies = _frequencies_hz(omega, units)
    si = ElectromagneticScaleContract.si()
    gyrofrequency = (
        float(si.elementary_charge)
        * field
        / units.gauss_per_tesla
        / (2.0 * math.pi * float(si.electron_mass))
    )
    match plan.route:
        case "harmonic-sum":
            # Exact harmonic code with exact Bessel functions at every frequency.
            boundary = 2.0 * math.ceil(float(frequencies[-1]) / gyrofrequency) + 2.0
        case "continuous-harmonic":
            boundary = 0.0
        case _:
            assert_never(plan.route)
    # Lparms_M: Npix, Nz, Nf, NE, Nmu, Nnodes (adaptive), match_key (off),
    # Qopt_key (on), arr_key (array distribution off), log_key, PK_key, spline_key.
    integers = [2 * theta.size, 1, omega.size, 0, 0, -1, 1, 0, 1, 0, 0, 0]
    pixels = []
    voxels = []
    for angle in theta:
        for depth in (_THICK_DEPTH, _THIN_DEPTH):
            # Rparms: area (cm²), f₀ ≤ 0 (frequencies from RL), Δf, f_C, f_WH.
            pixels.append([1.0, 0.0, 0.0, boundary, boundary])
            voxel = [0.0] * _VOXEL_PARAMETERS
            voxel[_DEPTH] = depth
            voxel[_FIELD] = field
            voxel[_ANGLE] = math.degrees(float(angle))
            voxel[_EMISSION_FLAGS] = _GYROSYNCHROTRON_ONLY
            voxel[_PITCH_MODEL] = 0.0
            # A nonzero proton density only disables UFGC's Saha ionization branch.
            voxel[_PROTONS] = 1.0
            voxel[_ARRAY_KEY] = 1.0
            for key, value in population.items():
                voxel[key] = value
            voxels.append(voxel)
    deck = {
        "library": library_name,
        "integer_parameters": integers,
        "pixel_parameters": pixels,
        "voxel_parameters": voxels,
        "frequencies_ghz": (frequencies / 1.0e9).tolist(),
        "echo": {
            "angular_frequencies": omega.tolist(),
            "angles": theta.tolist(),
            "thick_depth_cm": _THICK_DEPTH,
            "thin_depth_cm": _THIN_DEPTH,
            "area_cm2": 1.0,
        },
    }
    return {
        "ufgc_driver.py": _UFGC_DRIVER,
        "ufgc_input.json": json.dumps(deck).encode(),
    }


def read_ufgc_output(data: bytes, plan: MagnetobremsstrahlungPlan, /) -> UFGCCoefficients:
    """Recover UFGC's mode coefficients from its thick and thin lines of sight."""
    plan = _electron_plan(plan)
    document = json.loads(data)
    if document["status"] != 0:
        raise ValueError(
            f"UFGC rejected the translated parameters ({document['status']})."
        )
    echo = document["echo"]
    omega = np.asarray(echo["angular_frequencies"], dtype=np.float64)
    theta = np.asarray(echo["angles"], dtype=np.float64)
    left = np.asarray(document["left"], dtype=np.float64)
    right = np.asarray(document["right"], dtype=np.float64)
    shape = (2 * theta.size, omega.size)
    if left.shape != shape or right.shape != shape:
        raise ValueError("UFGC output does not match its evaluation grid.")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("UFGC returned nonfinite intensities.")
    # sfu at 1 au for area S -> intensity (erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹).
    intensity = _UFGC_OBSERVER_DISTANCE_CM**2 * _UFGC_SFU_CGS / echo["area_cm2"]
    # For θ ≤ π/2 UFGC's left row is the ordinary mode; beyond, the extraordinary.
    ordinary_first = (theta <= 0.5 * math.pi)[:, None, None, None]
    rows = np.stack((left, right), axis=-1).reshape(theta.size, 2, omega.size, 2)
    modes = np.where(ordinary_first, rows, rows[..., ::-1]) * intensity
    source = modes[:, 0]
    emission = modes[:, 1] / echo["thin_depth_cm"]
    positive = source > 0.0
    absorption = np.where(positive, emission / np.where(positive, source, 1.0), 0.0)
    # UFGC evaluates 1 − e^{−τ} as τ, exactly, only when e^{−τ} rounds to 1.
    if np.any(absorption * echo["thin_depth_cm"] >= 0.5 * np.finfo(np.float64).eps):
        raise ValueError("The thin UFGC voxel is not optically negligible.")
    if np.any(positive & ~(absorption * echo["thick_depth_cm"] > 745.0)):
        raise ValueError("The thick UFGC voxel is not optically thick.")
    emission_scale, absorption_scale = _coefficient_scales(_units(plan.plasma.scale))
    return UFGCCoefficients(
        omega, theta, emission * emission_scale, absorption * absorption_scale
    )


def _ufgc_losses(plan: MagnetobremsstrahlungPlan, /) -> tuple[AdapterLoss, ...]:
    match plan.route:
        case "harmonic-sum":
            route = AdapterLoss(
                "route",
                "export",
                "dropped",
                "UFGC's exact code sums normal-Doppler harmonics s ≥ 1 only; the "
                "plan's anomalous-Doppler (s ≤ 0) resonances and its quadrature and "
                "harmonic-capacity resources are not represented.",
                changes_interpretation=False,
            )
        case "continuous-harmonic":
            route = AdapterLoss(
                "route",
                "export",
                "transformed",
                "UFGC's continuous code (Fleishman & Kuznetsov 2010: approximate "
                "Bessel functions, Q-optimization, adaptive energy nodes) replaces the "
                "plan's exact real-order Bessel integral over (u, cos α).",
                changes_interpretation=False,
            )
        case _:
            assert_never(plan.route)
    return (
        route,
        AdapterLoss(
            "plasma",
            "export",
            "transformed",
            "UFGC recomputes the magnetoionic refractive indices and polarizations "
            "from its own electron density n₀ + n_b; the plan's dielectric object "
            "is not transmitted.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "constants",
            "export",
            "transformed",
            "UFGC evaluates gyrofrequency, plasma frequency and prefactors with its "
            "own CGS constants; the deck uses CODATA 2022.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "coefficients",
            "import",
            "transformed",
            "Mode coefficients are recovered from two single-voxel transfers "
            f"(depths {_THICK_DEPTH:.0e} and {_THIN_DEPTH:.0e} cm): S = j/κ and j Δz.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "polarization",
            "import",
            "dropped",
            "UFGC reports mode intensities only: no Stokes transfer coefficients or "
            "Faraday rotation/conversion are imported.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "evidence",
            "import",
            "unsupported",
            "UFGC reports no per-root status: modes it does not propagate carry zero "
            "coefficients where the plan reports NaN with explicit status flags.",
            changes_interpretation=True,
        ),
    )


def run_ufgc(
    provider: UFGCProvider,
    plan: MagnetobremsstrahlungPlan,
    destination: str | Path,
    /,
    *,
    angular_frequencies: ConvertibleToArray,
    angles: ConvertibleToArray,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 26,
) -> UFGCResult:
    """Run the pinned UFGC library on the translated plan and import its coefficients."""
    if not isinstance(provider, UFGCProvider):
        raise TypeError("provider must be a UFGCProvider.")
    library = Path(provider.library.path)
    inputs = ufgc_input(
        plan,
        angular_frequencies=angular_frequencies,
        angles=angles,
        library_name=library.name,
    )
    inputs[library.name] = _verified_bytes(provider.library)
    run = run_pinned_command(
        provider.executable,
        ("ufgc_driver.py", "ufgc_input.json"),
        inputs=inputs,
        timeout=timeout,
        max_output_bytes=maximum_output_bytes,
        artifacts=_file_artifacts(destination, _UFGC_OUTPUT, maximum_output_bytes),
    ).require_success()
    artifact = run.file_artifact(_UFGC_OUTPUT)
    coefficients = read_ufgc_output(Path(artifact.location).read_bytes(), plan)
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "ufgc-mw-transfer",
        "phydrax-magnetobremsstrahlung-mode-coefficients",
        source_id=artifact.sha256,
        target_id=canonical_fingerprint(
            {
                "kind": "ufgc-mode-coefficients",
                "plan": plan.plan_id,
                "arrays": array_tree_fingerprint(
                    (
                        coefficients.angular_frequencies,
                        coefficients.angles,
                        coefficients.emission,
                        coefficients.absorption,
                    )
                ),
            }
        ),
        coordinate_mapping=(
            "angular frequency ω / (2π time unitSI) -> UFGC ν in GHz",
            "magnetic field × unitSI × √(4π cm³/(μ₀ erg)) -> UFGC B in gauss",
            "wave-normal angle θ -> UFGC viewing angle in degrees",
            "densities / (length unitSI)³ × cm³ -> UFGC n₀, n_b in cm⁻³",
            "UFGC left/right rows -> (ordinary, extraordinary) for θ ≤ π/2, swapped beyond",
            "UFGC j_ν [erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹] / 2π -> j_ω; κ [cm⁻¹] -> per plan length",
        ),
        preserved_fields=("angular_frequencies", "angles", "distribution parameters"),
        assumptions=(
            "each UFGC mode obeys dI/dz = j − κI across one homogeneous voxel",
            "UFGC fluxes are sfu at 1 au for the declared source area",
        ),
        losses=_ufgc_losses(plan),
    )
    return UFGCResult(
        coefficients,
        provider.library.version,
        provider.library.sha256,
        provider.library.license_id,
        artifact.sha256,
        report,
    )


__all__ = [
    "read_symphony_output",
    "read_ufgc_output",
    "run_symphony",
    "run_ufgc",
    "symphony_input",
    "SymphonyCoefficients",
    "SymphonyProvider",
    "SymphonyResult",
    "ufgc_input",
    "UFGCCoefficients",
    "UFGCProvider",
    "UFGCResult",
]
