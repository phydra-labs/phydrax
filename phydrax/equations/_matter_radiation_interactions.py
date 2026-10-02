#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference matter-radiation processes used by electromagnetic showers.

The analytic routes in this module are independent of proprietary transport
providers.  Tabulated Seltzer--Berger data are accepted only through an
explicit, rights-checked manifest; this package does not ship reconstructed or
synthetic NIST values.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest
from ..typing import checked, parse


BremsstrahlungSpectrumRoute: TypeAlias = Literal[
    "bounded", "screened-bethe-heitler", "seltzer-berger"
]

_SI_SCALE = ElectromagneticScaleContract.si()
_ALPHA = float(_SI_SCALE.fine_structure)
_CLASSICAL_ELECTRON_RADIUS_M = float(_SI_SCALE.classical_electron_radius)
_PLANCK_EV_S = float(
    2.0 * np.pi * _SI_SCALE.reduced_planck_constant / _SI_SCALE.elementary_charge
)
_LIGHT_SPEED_M_S = float(_SI_SCALE.speed_of_light)


class SeltzerBergerBremsstrahlungTable(StrictModule, NonTrainableState):
    """Governed differential bremsstrahlung table and normalized row CDFs.

    ``differential_cross_section`` is the source table's nonnegative
    differential value on ``(material, kinetic_energy, photon_fraction)``.
    Its absolute unit is retained by the caller's manifest but cancels during
    spectral sampling.  Rows must have positive trapezoidal integral.  No
    extrapolation is performed.
    """

    kinetic_energy_ev: Array
    photon_fraction: Array
    differential_cross_section: Array
    cumulative_probability: Array
    probability_density: Array
    material_ids: tuple[str, ...] = eqx.field(static=True)
    manifest: ReferenceArtifactManifest = eqx.field(static=True)
    source_url: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        kinetic_energy_ev: ArrayLike,
        photon_fraction: ArrayLike,
        differential_cross_section: ArrayLike,
        material_ids: tuple[str, ...],
        manifest: ReferenceArtifactManifest,
        source_url: str,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        export: bool = False,
    ) -> None:
        energy = np.asarray(kinetic_energy_ev, dtype=np.float64)
        fraction = np.asarray(photon_fraction, dtype=np.float64)
        differential = np.asarray(differential_cross_section, dtype=np.float64)
        materials = tuple(str(value).strip() for value in material_ids)
        url = str(source_url).strip()
        manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            export=export,
        )
        expected = (len(materials), energy.size, fraction.size)
        if (
            energy.ndim != 1
            or energy.size < 2
            or np.any(~np.isfinite(energy))
            or np.any(energy <= 0.0)
            or np.any(np.diff(energy) <= 0.0)
            or fraction.ndim != 1
            or fraction.size < 3
            or np.any(~np.isfinite(fraction))
            or np.any(fraction <= 0.0)
            or np.any(fraction >= 1.0)
            or np.any(np.diff(fraction) <= 0.0)
            or not materials
            or len(set(materials)) != len(materials)
            or any(not value for value in materials)
            or not url
            or differential.shape != expected
            or np.any(~np.isfinite(differential))
            or np.any(differential < 0.0)
        ):
            raise ValueError("Seltzer--Berger table axes or values are invalid.")
        increments = (
            0.5 * (differential[..., 1:] + differential[..., :-1]) * np.diff(fraction)
        )
        cumulative = np.concatenate(
            (
                np.zeros(expected[:-1] + (1,), dtype=np.float64),
                np.cumsum(increments, axis=-1),
            ),
            axis=-1,
        )
        normalization = cumulative[..., -1].copy()
        if np.any(~np.isfinite(normalization)) or np.any(normalization <= 0.0):
            raise ValueError(
                "Every Seltzer--Berger spectrum row must have positive integral."
            )
        cumulative /= normalization[..., None]
        probability_density = differential / normalization[..., None]
        self.kinetic_energy_ev = jnp.asarray(energy)
        self.photon_fraction = jnp.asarray(fraction)
        self.differential_cross_section = jnp.asarray(differential)
        self.cumulative_probability = jnp.asarray(cumulative)
        self.probability_density = jnp.asarray(probability_density)
        self.material_ids = materials
        self.manifest = manifest
        self.source_url = url
        self.table_id = canonical_fingerprint(
            {
                "kind": "seltzer-berger-differential-bremsstrahlung",
                "kinetic_energy_ev": array_tree_fingerprint(energy),
                "photon_fraction": array_tree_fingerprint(fraction),
                "differential_cross_section": array_tree_fingerprint(differential),
                "materials": materials,
                "manifest": manifest.manifest_id,
                "source_url": url,
            }
        )

    @property
    def material_count(self) -> int:
        return len(self.material_ids)

    def sample_fraction(
        self, material_index: ArrayLike, kinetic_energy_ev: ArrayLike, draw: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Invert the linearly energy-interpolated normalized CDF."""
        material = jnp.asarray(material_index, dtype=jnp.int32)
        energy = jnp.asarray(kinetic_energy_ev, dtype=self.kinetic_energy_ev.dtype)
        uniform = jnp.asarray(draw, dtype=energy.dtype)
        shape = jnp.broadcast_shapes(material.shape, energy.shape, uniform.shape)
        material = jnp.broadcast_to(material, shape)
        energy = jnp.broadcast_to(energy, shape)
        uniform = jnp.broadcast_to(uniform, shape)
        supported = (
            (material >= 0)
            & (material < self.material_count)
            & jnp.isfinite(energy)
            & (energy >= self.kinetic_energy_ev[0])
            & (energy <= self.kinetic_energy_ev[-1])
            & jnp.isfinite(uniform)
            & (uniform >= 0.0)
            & (uniform < 1.0)
        )
        safe_material = jnp.clip(material, 0, self.material_count - 1)
        safe_energy = jnp.clip(
            energy, self.kinetic_energy_ev[0], self.kinetic_energy_ev[-1]
        )
        upper_energy = jnp.clip(
            jnp.searchsorted(self.kinetic_energy_ev, safe_energy, side="right"),
            1,
            self.kinetic_energy_ev.size - 1,
        )
        lower_energy = upper_energy - 1
        energy_fraction = (safe_energy - self.kinetic_energy_ev[lower_energy]) / (
            self.kinetic_energy_ev[upper_energy] - self.kinetic_energy_ev[lower_energy]
        )
        lower_cdf = self.cumulative_probability[safe_material, lower_energy]
        upper_cdf = self.cumulative_probability[safe_material, upper_energy]
        cdf = lower_cdf + energy_fraction[..., None] * (upper_cdf - lower_cdf)
        lower_density = self.probability_density[safe_material, lower_energy]
        upper_density = self.probability_density[safe_material, upper_energy]
        density = lower_density + energy_fraction[..., None] * (
            upper_density - lower_density
        )
        upper = jnp.sum(uniform[..., None] >= cdf, axis=-1)
        upper = jnp.clip(upper, 1, self.photon_fraction.size - 1)
        lower = upper - 1
        cdf_left = jnp.take_along_axis(cdf, lower[..., None], axis=-1)[..., 0]
        cdf_right = jnp.take_along_axis(cdf, upper[..., None], axis=-1)[..., 0]
        density_left = jnp.take_along_axis(density, lower[..., None], axis=-1)[..., 0]
        density_right = jnp.take_along_axis(density, upper[..., None], axis=-1)[..., 0]
        width = self.photon_fraction[upper] - self.photon_fraction[lower]
        slope = density_right - density_left
        probability = uniform - cdf_left
        root = jnp.sqrt(
            jnp.maximum(density_left**2 + 2.0 * slope * probability / width, 0.0)
        )
        interpolation = (
            2.0
            * probability
            / jnp.where(
                width * (density_left + root) > 0.0,
                width * (density_left + root),
                1.0,
            )
        )
        sampled = self.photon_fraction[lower] + jnp.clip(interpolation, 0.0, 1.0) * width
        return sampled, supported & (cdf_right > cdf_left)


def _screened_bethe_heitler_fraction(draw: Array, minimum_fraction: Array) -> Array:
    """Inverse CDF of ``(4/3 - 4y/3 + y²)/y`` by fixed bisection."""

    def primitive(value: Array) -> Array:
        return (4.0 / 3.0) * jnp.log(value) - (4.0 / 3.0) * value + 0.5 * value**2

    lower = jnp.clip(minimum_fraction, jnp.finfo(draw.dtype).eps, 1.0)
    target = primitive(lower) + draw * (
        primitive(jnp.asarray(1.0, draw.dtype)) - primitive(lower)
    )

    def bisect(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        left, right = bounds
        middle = 0.5 * (left + right)
        return jnp.where(primitive(middle) < target, middle, left), jnp.where(
            primitive(middle) < target, right, middle
        )

    left, right = jax.lax.fori_loop(0, 48, bisect, (lower, jnp.ones_like(lower)))
    return 0.5 * (left + right)


def sample_bremsstrahlung_fraction(
    route: BremsstrahlungSpectrumRoute,
    draw: ArrayLike,
    kinetic_energy_ev: ArrayLike,
    material_index: ArrayLike,
    minimum_photon_energy_ev: float,
    /,
    *,
    table: SeltzerBergerBremsstrahlungTable | None = None,
) -> tuple[Array, Array]:
    """Sample a photon-energy fraction from the selected physical route.

    The screened Bethe--Heitler route uses the complete-screening Tsai shape.
    The Seltzer--Berger route refuses when a governed table is absent or outside
    support; it never substitutes analytic or fabricated values.
    """
    selected = parse(route, BremsstrahlungSpectrumRoute, "route")
    uniform = jnp.asarray(draw)
    kinetic = jnp.asarray(kinetic_energy_ev, dtype=uniform.dtype)
    material = jnp.asarray(material_index, dtype=jnp.int32)
    threshold = float(minimum_photon_energy_ev)
    if not isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_photon_energy_ev must be finite and positive.")
    minimum_fraction = jnp.clip(threshold / kinetic, 0.0, 1.0)
    required_energy = threshold
    match selected:
        case "bounded":
            sampled = 0.5 * uniform
            supported = kinetic > 0.0
            required_energy = 0.0
        case "screened-bethe-heitler":
            sampled = _screened_bethe_heitler_fraction(uniform, minimum_fraction)
            supported = minimum_fraction < 1.0
        case "seltzer-berger":
            if table is None:
                raise ValueError("seltzer-berger sampling requires a governed table.")
            sampled, supported = table.sample_fraction(material, kinetic, uniform)
            required_energy = 0.0
        case _:
            raise RuntimeError("Unreachable bremsstrahlung route.")
    valid = (
        supported
        & jnp.isfinite(sampled)
        & (uniform >= 0.0)
        & (uniform < 1.0)
        & (kinetic > required_energy)
        & (sampled > 0.0)
        & (sampled < 1.0)
    )
    return jnp.where(valid, sampled, 0.0), valid


def bremsstrahlung_suppression_factor(
    kinetic_energy_ev: ArrayLike,
    photon_energy_ev: ArrayLike,
    /,
    *,
    lpm_energy_ev: float | None = None,
    plasma_energy_ev: float | None = None,
) -> Array:
    """Migdal high-energy and Ter--Mikaelian dielectric suppression factors.

    A disabled mechanism contributes the exact multiplicative identity.  The
    LPM asymptote is ``sqrt(k E_LPM / (E (E-k)))`` capped at one.  Dielectric
    suppression is ``k² / (k² + (gamma hbar omega_p)²)``.
    """
    kinetic = jnp.asarray(kinetic_energy_ev)
    photon = jnp.asarray(photon_energy_ev, dtype=kinetic.dtype)
    total = kinetic + 510998.95069
    lpm = jnp.ones_like(kinetic)
    if lpm_energy_ev is not None:
        scale = float(lpm_energy_ev)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("lpm_energy_ev must be finite and positive when enabled.")
        lpm = jnp.minimum(
            1.0,
            jnp.sqrt(
                jnp.maximum(
                    photon * scale / (total * jnp.maximum(total - photon, photon)), 0.0
                )
            ),
        )
    dielectric = jnp.ones_like(kinetic)
    if plasma_energy_ev is not None:
        plasma = float(plasma_energy_ev)
        if not isfinite(plasma) or plasma <= 0.0:
            raise ValueError("plasma_energy_ev must be finite and positive when enabled.")
        scale = (total / 510998.95069) * plasma
        dielectric = photon**2 / (photon**2 + scale**2)
    return jnp.where((photon > 0.0) & (photon < total), lpm * dielectric, 0.0)


def bethe_heitler_pair_cross_section_m2(
    photon_energy_ev: ArrayLike,
    atomic_number: ArrayLike,
    electron_rest_energy_ev: float = 510998.95069,
    /,
) -> tuple[Array, Array]:
    """Screened Bethe--Heitler nuclear- and electron-field cross sections.

    The high-energy Tsai radiation logarithms use ``183 Z^(-1/3)`` for the
    nuclear field and ``1194 Z^(-2/3)`` for the electron field.  A cubic
    threshold factor enforces exact support at ``2 m_e c²`` while tending to
    one at high energy.  Values are per atom in square meters.
    """
    energy = jnp.asarray(photon_energy_ev)
    z = jnp.asarray(atomic_number, dtype=energy.dtype)
    mass = float(electron_rest_energy_ev)
    if not isfinite(mass) or mass <= 0.0:
        raise ValueError("electron_rest_energy_ev must be finite and positive.")
    supported = (
        (energy > 2.0 * mass) & (z >= 1.0) & jnp.isfinite(energy) & jnp.isfinite(z)
    )
    threshold = jnp.where(supported, (1.0 - (2.0 * mass / energy) ** 2) ** 3, 0.0)
    common = 4.0 * _ALPHA * _CLASSICAL_ELECTRON_RADIUS_M**2
    nuclear_log = jnp.maximum(jnp.log(183.0 * z ** (-1.0 / 3.0)) - 1.0 / 42.0, 0.0)
    electron_log = jnp.maximum(jnp.log(1194.0 * z ** (-2.0 / 3.0)) - 1.0 / 42.0, 0.0)
    nuclear = common * (7.0 / 9.0) * z**2 * nuclear_log * threshold
    electron = common * (7.0 / 9.0) * z * electron_log * threshold
    return jnp.where(supported, nuclear, 0.0), jnp.where(supported, electron, 0.0)


def sample_bethe_heitler_pair_kinetic_energies(
    draw: ArrayLike,
    photon_energy_ev: ArrayLike,
    electron_rest_energy_ev: float = 510998.95069,
    /,
) -> tuple[Array, Array, Array]:
    """Sample the screened Bethe--Heitler energy sharing ``1-4x(1-x)/3``."""
    uniform = jnp.asarray(draw)
    energy = jnp.asarray(photon_energy_ev, dtype=uniform.dtype)
    mass = float(electron_rest_energy_ev)

    def cdf(value: Array) -> Array:
        return (9.0 / 7.0) * (value - (2.0 / 3.0) * value**2 + (4.0 / 9.0) * value**3)

    def bisect(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        left, right = bounds
        middle = 0.5 * (left + right)
        return jnp.where(cdf(middle) < uniform, middle, left), jnp.where(
            cdf(middle) < uniform, right, middle
        )

    left, right = jax.lax.fori_loop(
        0, 48, bisect, (jnp.zeros_like(uniform), jnp.ones_like(uniform))
    )
    fraction = 0.5 * (left + right)
    available = jnp.maximum(energy - 2.0 * mass, 0.0)
    valid = (
        jnp.isfinite(energy)
        & jnp.isfinite(uniform)
        & (uniform >= 0.0)
        & (uniform < 1.0)
        & (energy > 2.0 * mass)
    )
    return available * fraction, available * (1.0 - fraction), valid


class DeltaRayKinematics(StrictModule):
    """Exact two-body lab kinematics for scattering from an electron at rest."""

    primary_kinetic_energy_ev: Array
    secondary_kinetic_energy_ev: Array
    primary_cosine: Array
    secondary_cosine: Array
    four_momentum_residual_ev: Array
    valid: Array


def delta_ray_kinematics(
    incident_kinetic_energy_ev: ArrayLike,
    transferred_energy_ev: ArrayLike,
    particle_kind: ArrayLike,
    electron_rest_energy_ev: float = 510998.95069,
    /,
) -> DeltaRayKinematics:
    """Møller (electron) or Bhabha (positron) delta-ray lab kinematics.

    ``particle_kind`` is zero for an electron and one for a positron.  Møller
    transfer is limited to half the incident kinetic energy because the two
    final electrons are indistinguishable; Bhabha transfer may approach the
    full kinetic energy.  The returned scalar residual evaluates energy and
    longitudinal/transverse momentum closure in eV/c units.
    """
    incident = jnp.asarray(incident_kinetic_energy_ev)
    transfer = jnp.asarray(transferred_energy_ev, dtype=incident.dtype)
    kind = jnp.asarray(particle_kind, dtype=jnp.int32)
    mass = float(electron_rest_energy_ev)
    primary = incident - transfer
    incident_momentum = jnp.sqrt(incident * (incident + 2.0 * mass))
    primary_momentum = jnp.sqrt(jnp.maximum(primary * (primary + 2.0 * mass), 0.0))
    secondary_momentum = jnp.sqrt(jnp.maximum(transfer * (transfer + 2.0 * mass), 0.0))
    primary_cosine = (
        incident_momentum**2 + primary_momentum**2 - secondary_momentum**2
    ) / jnp.where(
        2.0 * incident_momentum * primary_momentum > 0.0,
        2.0 * incident_momentum * primary_momentum,
        1.0,
    )
    secondary_cosine = (
        incident_momentum**2 + secondary_momentum**2 - primary_momentum**2
    ) / jnp.where(
        2.0 * incident_momentum * secondary_momentum > 0.0,
        2.0 * incident_momentum * secondary_momentum,
        1.0,
    )
    primary_cosine = jnp.clip(primary_cosine, -1.0, 1.0)
    secondary_cosine = jnp.clip(secondary_cosine, -1.0, 1.0)
    transverse = primary_momentum * jnp.sqrt(
        jnp.maximum(1.0 - primary_cosine**2, 0.0)
    ) - secondary_momentum * jnp.sqrt(jnp.maximum(1.0 - secondary_cosine**2, 0.0))
    longitudinal = (
        incident_momentum
        - primary_momentum * primary_cosine
        - secondary_momentum * secondary_cosine
    )
    energy_residual = incident - primary - transfer
    residual = jnp.sqrt(energy_residual**2 + transverse**2 + longitudinal**2)
    maximum = jnp.where(kind == 0, 0.5 * incident, incident)
    valid = (
        ((kind == 0) | (kind == 1))
        & jnp.isfinite(incident)
        & jnp.isfinite(transfer)
        & (incident > 0.0)
        & (transfer > 0.0)
        & (transfer <= maximum)
    )
    return DeltaRayKinematics(
        primary, transfer, primary_cosine, secondary_cosine, residual, valid
    )


class AnnihilationKinematics(StrictModule):
    """Two-photon positron annihilation in flight in the electron-rest frame."""

    photon_energies_ev: Array
    photon_cosines: Array
    four_momentum_residual_ev: Array
    valid: Array


def positron_annihilation_in_flight(
    positron_kinetic_energy_ev: ArrayLike,
    com_cosine: ArrayLike,
    electron_rest_energy_ev: float = 510998.95069,
    /,
) -> AnnihilationKinematics:
    """Boost an exactly back-to-back two-photon COM final state to the lab."""
    kinetic = jnp.asarray(positron_kinetic_energy_ev)
    cosine = jnp.asarray(com_cosine, dtype=kinetic.dtype)
    mass = float(electron_rest_energy_ev)
    positron_total = kinetic + mass
    incident_momentum = jnp.sqrt(jnp.maximum(kinetic * (kinetic + 2.0 * mass), 0.0))
    total_energy = positron_total + mass
    beta = incident_momentum / total_energy
    gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta**2, jnp.finfo(kinetic.dtype).tiny))
    com_energy = jnp.sqrt(jnp.maximum(2.0 * mass * total_energy, 0.0)) / 2.0
    energy_1 = gamma * com_energy * (1.0 + beta * cosine)
    energy_2 = gamma * com_energy * (1.0 - beta * cosine)
    cosine_1 = (cosine + beta) / (1.0 + beta * cosine)
    cosine_2 = (-cosine + beta) / (1.0 - beta * cosine)
    transverse_1 = energy_1 * jnp.sqrt(jnp.maximum(1.0 - cosine_1**2, 0.0))
    transverse_2 = energy_2 * jnp.sqrt(jnp.maximum(1.0 - cosine_2**2, 0.0))
    energy_residual = total_energy - energy_1 - energy_2
    longitudinal = incident_momentum - energy_1 * cosine_1 - energy_2 * cosine_2
    transverse = transverse_1 - transverse_2
    residual = jnp.sqrt(energy_residual**2 + longitudinal**2 + transverse**2)
    valid = (
        jnp.isfinite(kinetic)
        & jnp.isfinite(cosine)
        & (kinetic >= 0.0)
        & (cosine >= -1.0)
        & (cosine <= 1.0)
    )
    return AnnihilationKinematics(
        jnp.stack((energy_1, energy_2), axis=-1),
        jnp.stack((cosine_1, cosine_2), axis=-1),
        residual,
        valid,
    )


class AtomicRelaxationResult(StrictModule):
    fluorescence_energy_ev: Array
    auger_energy_ev: Array
    local_deposit_ev: Array
    closure_residual_ev: Array
    valid: Array


def atomic_relaxation(
    vacancy_energy_ev: ArrayLike,
    fluorescence_yield: ArrayLike,
    fluorescence_fraction: ArrayLike,
    draw: ArrayLike,
    /,
) -> AtomicRelaxationResult:
    """Single-vacancy fluorescence/Auger partition with exact energy closure."""
    vacancy = jnp.asarray(vacancy_energy_ev)
    yield_ = jnp.asarray(fluorescence_yield, dtype=vacancy.dtype)
    line_fraction = jnp.asarray(fluorescence_fraction, dtype=vacancy.dtype)
    uniform = jnp.asarray(draw, dtype=vacancy.dtype)
    fluoresces = uniform < yield_
    photon = jnp.where(fluoresces, vacancy * line_fraction, 0.0)
    auger = jnp.where(fluoresces, vacancy - photon, vacancy)
    local = jnp.zeros_like(vacancy)
    residual = vacancy - photon - auger - local
    valid = (
        jnp.isfinite(vacancy)
        & (vacancy >= 0.0)
        & jnp.isfinite(yield_)
        & (yield_ >= 0.0)
        & (yield_ <= 1.0)
        & jnp.isfinite(line_fraction)
        & (line_fraction >= 0.0)
        & (line_fraction <= 1.0)
        & jnp.isfinite(uniform)
        & (uniform >= 0.0)
        & (uniform < 1.0)
    )
    return AtomicRelaxationResult(photon, auger, local, residual, valid)


def cherenkov_yield_in_band(
    step_length_m: ArrayLike,
    beta: ArrayLike,
    refractive_index: ArrayLike,
    wavelength_min_m: float,
    wavelength_max_m: float,
    /,
    *,
    charge_number: float = 1.0,
) -> tuple[Array, Array]:
    """Frank--Tamm photon count and energy over a nondispersive wavelength band."""
    length = jnp.asarray(step_length_m)
    speed = jnp.asarray(beta, dtype=length.dtype)
    index = jnp.asarray(refractive_index, dtype=length.dtype)
    lower, upper, charge = (
        float(wavelength_min_m),
        float(wavelength_max_m),
        float(charge_number),
    )
    if (
        not isfinite(lower)
        or not isfinite(upper)
        or not 0.0 < lower < upper
        or not isfinite(charge)
        or charge == 0.0
    ):
        raise ValueError("Cherenkov band and charge must be finite and physical.")
    factor = jnp.maximum(1.0 - 1.0 / (speed**2 * index**2), 0.0)
    count = (
        2.0 * np.pi * _ALPHA * charge**2 * length * factor * (1.0 / lower - 1.0 / upper)
    )
    energy = (
        np.pi
        * _ALPHA
        * charge**2
        * _PLANCK_EV_S
        * _LIGHT_SPEED_M_S
        * length
        * factor
        * (1.0 / lower**2 - 1.0 / upper**2)
    )
    valid = (
        jnp.isfinite(length)
        & (length >= 0.0)
        & jnp.isfinite(speed)
        & (speed > 0.0)
        & (speed < 1.0)
        & jnp.isfinite(index)
        & (index > 0.0)
    )
    return jnp.where(valid, count, 0.0), jnp.where(valid, energy, 0.0)


def cherenkov_step_spectral_yield(
    step_length: ArrayLike,
    start_beta: ArrayLike,
    end_beta: ArrayLike,
    refractive_index: ArrayLike,
    wavelength: ArrayLike,
    /,
    *,
    charge_number: ArrayLike = 1.0,
) -> tuple[Array, Array]:
    """Dispersive Frank--Tamm photons per unit wavelength emitted over one step.

    The spectral density is ``d2N/(dx dlambda) = 2 pi alpha z^2 / lambda^2
    (1 - 1 / (beta^2 n(lambda)^2))`` where ``beta n > 1`` (Frank and Tamm
    1937).  ``beta`` is linear in path length from ``start_beta`` to
    ``end_beta``, so the path integral is exact: over the above-threshold
    speeds ``[b_lo, b_hi]`` with ``b_lo = max(min beta, 1 / n)`` it equals
    ``L (b_hi - b_lo) / |end_beta - start_beta| (1 - 1 / (n^2 b_lo b_hi))``,
    which reduces to ``L (1 - 1 / (beta^2 n^2))`` at constant speed.
    ``step_length`` and ``wavelength`` share one length unit; the result is
    photons per that unit of wavelength.  Arguments broadcast; the second
    output marks finite, physical inputs, and invalid entries are ``NaN``.
    """
    length = jnp.asarray(step_length, dtype=jnp.float64)
    first = jnp.asarray(start_beta, dtype=jnp.float64)
    last = jnp.asarray(end_beta, dtype=jnp.float64)
    index = jnp.asarray(refractive_index, dtype=jnp.float64)
    lam = jnp.asarray(wavelength, dtype=jnp.float64)
    charge = jnp.asarray(charge_number, dtype=jnp.float64)
    valid = (
        jnp.isfinite(length)
        & (length >= 0.0)
        & jnp.isfinite(first)
        & jnp.isfinite(last)
        & (first >= 0.0)
        & (first < 1.0)
        & (last >= 0.0)
        & (last < 1.0)
        & jnp.isfinite(index)
        & (index > 0.0)
        & jnp.isfinite(lam)
        & (lam > 0.0)
        & jnp.isfinite(charge)
    )
    low = jnp.minimum(first, last)
    high = jnp.maximum(first, last)
    threshold = 1.0 / index
    lower = jnp.clip(threshold, low, high)
    spread = high - low
    # Fraction of the step above threshold; a constant-speed step is all or none.
    above = jnp.where(
        spread > 0.0,
        (high - lower) / jnp.where(spread > 0.0, spread, 1.0),
        (high > threshold).astype(jnp.float64),
    )
    radiating = high > threshold
    factor = jnp.where(
        radiating,
        above * (1.0 - 1.0 / (index * index * lower * high)),
        0.0,
    )
    density = 2.0 * np.pi * _ALPHA * charge**2 * length * factor / (lam * lam)
    return jnp.where(valid, density, jnp.nan), valid


class FoilStackTransitionRadiationPlan(StrictModule, NonTrainableState):
    """Garibian/Cherry formation-zone interference for an arbitrary foil stack.

    Interface positions and signed dielectric jumps define regular or irregular
    stacks.  ``spectral_angular_yield`` coherently sums interface amplitudes,
    propagation phase, and Beer--Lambert amplitude attenuation.  Frequencies
    are photon energies in eV and angles are radians.
    """

    interface_positions_m: Array
    plasma_energy_jump_ev2: Array
    absorption_coefficient_m: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface_positions_m: ArrayLike,
        plasma_energy_jump_ev2: ArrayLike,
        absorption_coefficient_m: ArrayLike,
        /,
    ) -> None:
        positions = np.asarray(interface_positions_m, dtype=np.float64)
        jumps = np.asarray(plasma_energy_jump_ev2, dtype=np.float64)
        absorption = np.asarray(absorption_coefficient_m, dtype=np.float64)
        if (
            positions.ndim != 1
            or positions.size < 2
            or np.any(~np.isfinite(positions))
            or np.any(np.diff(positions) <= 0.0)
            or jumps.shape != positions.shape
            or np.any(~np.isfinite(jumps))
            or absorption.shape != positions.shape
            or np.any(~np.isfinite(absorption))
            or np.any(absorption < 0.0)
        ):
            raise ValueError(
                "Foil-stack interface, jump, or absorption data are invalid."
            )
        self.interface_positions_m = jnp.asarray(positions)
        self.plasma_energy_jump_ev2 = jnp.asarray(jumps)
        self.absorption_coefficient_m = jnp.asarray(absorption)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "garibian-cherry-foil-stack",
                "interfaces": array_tree_fingerprint(positions),
                "plasma_energy_jump_ev2": array_tree_fingerprint(jumps),
                "absorption_coefficient_m": array_tree_fingerprint(absorption),
            }
        )

    def formation_length_m(
        self, photon_energy_ev: ArrayLike, gamma: ArrayLike, angle_rad: ArrayLike, /
    ) -> Array:
        energy = jnp.asarray(photon_energy_ev)
        gamma_ = jnp.asarray(gamma, dtype=energy.dtype)
        angle = jnp.asarray(angle_rad, dtype=energy.dtype)
        return (
            2.0
            * (_PLANCK_EV_S / (2.0 * np.pi))
            * _LIGHT_SPEED_M_S
            / (energy * (gamma_**-2 + angle**2))
        )

    def spectral_angular_yield(
        self, photon_energy_ev: ArrayLike, gamma: ArrayLike, angle_rad: ArrayLike, /
    ) -> Array:
        """Return ``d²N/(dE dtheta²)`` up to the standard azimuth-integrated measure."""
        energy = jnp.asarray(photon_energy_ev)
        gamma_ = jnp.asarray(gamma, dtype=energy.dtype)
        angle = jnp.asarray(angle_rad, dtype=energy.dtype)
        energy, gamma_, angle = jnp.broadcast_arrays(energy, gamma_, angle)
        formation = self.formation_length_m(energy, gamma_, angle)
        distance_to_exit = self.interface_positions_m[-1] - self.interface_positions_m
        phase = self.interface_positions_m / formation[..., None]
        attenuation = jnp.exp(-0.5 * self.absorption_coefficient_m * distance_to_exit)
        denominator = energy[..., None] ** 2 * (
            gamma_[..., None] ** -2 + angle[..., None] ** 2
        )
        amplitude = jnp.sum(
            self.plasma_energy_jump_ev2 / denominator * attenuation * jnp.exp(1j * phase),
            axis=-1,
        )
        return (2.0 * _ALPHA / np.pi) * angle**2 * jnp.abs(amplitude) ** 2 / energy


def longo_shower_profile(
    depth_radiation_lengths: ArrayLike,
    primary_energy_ev: ArrayLike,
    critical_energy_ev: ArrayLike,
    /,
    *,
    incident: Literal["electron", "photon"] = "electron",
) -> Array:
    """Normalized Longo--Sestili gamma profile per radiation length.

    The PDG form uses ``b=1/2`` and ``t_max = ln(E/E_c) - 1/2`` for
    electrons or ``+1/2`` for photons, with ``a = 1 + b t_max``.
    """
    particle = parse(incident, Literal["electron", "photon"], "incident")
    depth = jnp.asarray(depth_radiation_lengths)
    energy = jnp.asarray(primary_energy_ev, dtype=depth.dtype)
    critical = jnp.asarray(critical_energy_ev, dtype=depth.dtype)
    maximum = jnp.log(energy / critical) + (-0.5 if particle == "electron" else 0.5)
    rate = jnp.asarray(0.5, dtype=depth.dtype)
    shape = 1.0 + rate * maximum
    safe_depth = jnp.maximum(depth, jnp.finfo(depth.dtype).tiny)
    log_density = (
        shape * jnp.log(rate)
        + (shape - 1.0) * jnp.log(safe_depth)
        - rate * safe_depth
        - jsp.special.gammaln(shape)
    )
    valid = (
        jnp.isfinite(depth)
        & (depth >= 0.0)
        & jnp.isfinite(energy)
        & jnp.isfinite(critical)
        & (energy > critical)
        & (shape > 0.0)
    )
    return jnp.where(valid, jnp.exp(log_density), 0.0)


__all__ = [
    "AnnihilationKinematics",
    "AtomicRelaxationResult",
    "BremsstrahlungSpectrumRoute",
    "DeltaRayKinematics",
    "FoilStackTransitionRadiationPlan",
    "SeltzerBergerBremsstrahlungTable",
    "atomic_relaxation",
    "bethe_heitler_pair_cross_section_m2",
    "bremsstrahlung_suppression_factor",
    "cherenkov_step_spectral_yield",
    "cherenkov_yield_in_band",
    "delta_ray_kinematics",
    "longo_shower_profile",
    "positron_annihilation_in_flight",
    "sample_bethe_heitler_pair_kinetic_energies",
    "sample_bremsstrahlung_fraction",
]
