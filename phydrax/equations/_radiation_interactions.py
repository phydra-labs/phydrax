#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import (
    conversion_factor,
    derived_unit,
    ELECTRONVOLT,
    JOULE,
    KILOGRAM,
    METER,
    UnitDefinition,
)
from ._diagnostic_photon import (
    DiagnosticPhotonCoefficientRole,
    DiagnosticPhotonCoefficientTable,
    DiagnosticPhotonInterpolationPolicy,
)


_MASS_ATTENUATION_SI = derived_unit("m2/kg", ((METER, 2), (KILOGRAM, -1)))
# Atomic unit of momentum is `m_e * alpha * c`, so a Compton-profile momentum in
# atomic units times the fine-structure constant is that momentum in `m_e c`.
_FINE_STRUCTURE = float(ElectromagneticScaleContract.si().fine_structure)

ComptonKinematics: TypeAlias = Literal["free-electron", "impulse-approximation"]
"""Scattered-photon energy at a sampled Klein–Nishina angle.

`"free-electron"` uses the Compton line of an electron at rest.
`"impulse-approximation"` Doppler-broadens that line with the target electron's
momentum projection drawn from the material's one-parameter Compton profile.
"""


class RadiationInteractionKind(IntEnum):
    PHOTOELECTRIC = 0
    COMPTON = 1
    RAYLEIGH = 2
    PAIR_NUCLEAR = 3
    PAIR_ELECTRON = 4


def sample_sauter_cosine(
    proposal_draws: Array,
    acceptance_draws: Array,
    kinetic_energy: Array,
    electron_rest_energy: float,
    /,
) -> tuple[Array, Array]:
    """Photoelectron polar cosine from the K-shell Sauter (1931) distribution.

    With `nu = 1 - cos(theta)`, `beta` the electron speed, `A = (1 - beta) / beta`,
    and `B = beta * gamma * (gamma - 1) * (gamma - 2) / 2`, the distribution is
    proportional to `nu / (A + nu)**3 * (2 - nu) * (1 / (A + nu) + B)`. The first
    factor is sampled exactly by inversion, `nu = 2 A sqrt(q) / (A + 2 - 2 sqrt(q))`,
    and the remaining factor is accepted against its maximum at `nu = 0`, which
    is `2 (1 / A + B)`. Returns the first accepted cosine over the bounded
    attempt axis and whether any attempt was accepted.
    """
    gamma = 1.0 + kinetic_energy / electron_rest_energy
    beta = jnp.sqrt(kinetic_energy * (kinetic_energy + 2.0 * electron_rest_energy)) / (
        kinetic_energy + electron_rest_energy
    )
    safe_beta = jnp.maximum(beta, jnp.finfo(beta.dtype).tiny)
    a = (1.0 - beta) / safe_beta
    b = 0.5 * beta * gamma * (gamma - 1.0) * (gamma - 2.0)
    root = jnp.sqrt(proposal_draws)
    nu = jnp.clip(2.0 * a * root / (a + 2.0 - 2.0 * root), 0.0, 2.0)
    remainder = (2.0 - nu) * (1.0 / (a + nu) + b)
    ceiling = 2.0 * (1.0 / a + b)
    accepted = acceptance_draws * ceiling <= remainder
    index = jnp.argmax(accepted)
    return 1.0 - nu[index], jnp.any(accepted)


def compton_electron_cosine(
    photon_energy: Array, scattered_energy: Array, photon_cosine: Array, /
) -> Array:
    """Recoil-electron polar cosine from momentum conservation off an electron at rest.

    The electron momentum is the incident photon momentum minus the scattered
    photon momentum, so its component along the incident direction is
    `E - E' cos(theta)` and its magnitude is `sqrt(E**2 + E'**2 - 2 E E' cos(theta))`.
    Its azimuth is opposite to the scattered photon's. Zero transfer returns one.
    """
    squared = (
        photon_energy**2
        + scattered_energy**2
        - 2.0 * photon_energy * scattered_energy * photon_cosine
    )
    magnitude = jnp.sqrt(jnp.maximum(squared, 0.0))
    cosine = (photon_energy - scattered_energy * photon_cosine) / jnp.where(
        magnitude > 0.0, magnitude, 1.0
    )
    return jnp.where(magnitude > 0.0, jnp.clip(cosine, -1.0, 1.0), 1.0)


def sample_compton_profile_momentum(draws: Array, profile_j0: Array, /) -> Array:
    """Electron momentum projection `p_z` in atomic units from a one-parameter profile.

    The profile is `J(p_z) = J0 (1 + 2 J0 |p_z|) exp((1 - (1 + 2 J0 |p_z|)**2) / 2)`
    (Brusa et al. 1996), a unit-normalized even function whose cumulative
    distribution is `exp((1 - (1 + 2 J0 |p_z|)**2) / 2) / 2` on the negative side.
    Inversion gives `|p_z| = (sqrt(1 - 2 ln(2 u)) - 1) / (2 J0)` for `u < 1/2`, with
    the mirrored branch above.
    """
    tiny = jnp.finfo(draws.dtype).tiny
    negative = draws < 0.5
    tail = jnp.clip(jnp.where(negative, draws, 1.0 - draws), tiny, 0.5)
    magnitude = (jnp.sqrt(1.0 - 2.0 * jnp.log(2.0 * tail)) - 1.0) / (2.0 * profile_j0)
    return jnp.where(negative, -magnitude, magnitude)


def doppler_scattered_energy(
    photon_energy: Array,
    photon_cosine: Array,
    momentum_au: Array,
    electron_rest_energy: float,
    /,
) -> tuple[Array, Array]:
    """Scattered energy at fixed angle for a target electron with projection `p_z`.

    In the impulse approximation (Ribberfors 1975) the projection of the bound
    electron momentum on the scattering vector satisfies
    `p_z = (E E' (1 - cos) - m c^2 (E - E')) / (c |q|)` with
    `|q| c = sqrt(E**2 + E'**2 - 2 E E' cos)`. Solving for `E'` with `p` in units
    of `m_e c`, `eps = E / (m c^2)`, and `a = 1 + eps (1 - cos)` gives
    `E' = E [a - p^2 cos + p sqrt((1 - cos)((1 - cos)(1 + eps)^2 + (1 + cos)(1 - p^2)))]
    / (a^2 - p^2)`, which reduces to the Compton line at `p = 0`. The result is
    valid when `|p| < 1` and `0 < E' <= E`; binding thresholds are not applied.
    """
    momentum = momentum_au * _FINE_STRUCTURE
    epsilon = photon_energy / electron_rest_energy
    one_minus = 1.0 - photon_cosine
    a = 1.0 + epsilon * one_minus
    squared = momentum**2
    radicand = one_minus * (
        one_minus * (1.0 + epsilon) ** 2 + (1.0 + photon_cosine) * (1.0 - squared)
    )
    denominator = a**2 - squared
    numerator = (
        a - squared * photon_cosine + momentum * jnp.sqrt(jnp.maximum(radicand, 0.0))
    )
    scattered = photon_energy * numerator / jnp.where(denominator > 0.0, denominator, 1.0)
    valid = (
        (jnp.abs(momentum) < 1.0)
        & (denominator > 0.0)
        & (radicand >= 0.0)
        & (scattered > 0.0)
        & (scattered <= photon_energy)
    )
    return jnp.where(valid, scattered, photon_energy), valid


class RadiationCrossSectionEvaluation(StrictModule):
    coefficients: Array
    total: Array
    supported: Array
    finite: Array
    successful: Array
    library_id: str = eqx.field(static=True)


class RadiationCrossSectionLibrary(StrictModule, NonTrainableState):
    """Macroscopic photon interaction coefficients on one governed energy grid.

    `compton_profile_j0` optionally supplies, per material, the one-parameter
    Compton profile value `J(0)` in atomic units that
    `"impulse-approximation"` Compton kinematics draw the target electron
    momentum from; without it only `"free-electron"` kinematics are available.
    """

    energy: Array
    coefficients: Array
    mass_density_kg_per_m3: Array
    log_interpolation: Array
    compton_profile_j0: Array | None
    material_ids: tuple[str, ...] = eqx.field(static=True)
    process_ids: tuple[str, ...] = eqx.field(static=True)
    energy_unit: UnitDefinition = eqx.field(static=True)
    source_table_ids: tuple[str, ...] = eqx.field(static=True)
    source_provenance_ids: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        photoelectric: DiagnosticPhotonCoefficientTable,
        compton: DiagnosticPhotonCoefficientTable,
        rayleigh: DiagnosticPhotonCoefficientTable,
        mass_density_kg_per_m3: ArrayLike,
        /,
        *,
        compton_profile_j0: ArrayLike | None = None,
        commercial_use: bool = False,
        redistribution: bool = False,
        export: bool = False,
        pair_nuclear: DiagnosticPhotonCoefficientTable | None = None,
        pair_electron: DiagnosticPhotonCoefficientTable | None = None,
    ) -> None:
        required_tables = (photoelectric, compton, rayleigh)
        if (pair_nuclear is None) != (pair_electron is None):
            raise ValueError(
                "Nuclear- and electron-field pair-production tables must be supplied together."
            )
        tables = (
            required_tables
            if pair_nuclear is None or pair_electron is None
            else required_tables + (pair_nuclear, pair_electron)
        )
        if not all(
            isinstance(table, DiagnosticPhotonCoefficientTable) for table in tables
        ):
            raise TypeError(
                "Photon interactions require three DiagnosticPhotonCoefficientTable values."
            )
        if any(
            table.role is not DiagnosticPhotonCoefficientRole.MASS_ATTENUATION
            for table in tables
        ):
            raise ValueError(
                "Photoelectric, Compton, and Rayleigh inputs must be mass-attenuation tables."
            )
        material_ids = photoelectric.material_ids
        energy_grid_id = photoelectric.energy_grid.grid_id
        if any(table.material_ids != material_ids for table in tables[1:]):
            raise ValueError(
                "Photon interaction tables must use one exact ordered material basis."
            )
        if any(table.energy_grid.grid_id != energy_grid_id for table in tables[1:]):
            raise ValueError(
                "Photon interaction tables must use one exact photon energy grid."
            )
        if len({table.table_id for table in tables}) != len(tables):
            raise ValueError(
                "Photon interaction process tables must have distinct identities."
            )
        for table in tables:
            table.provenance.reference.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                export=export,
            )
        density = np.asarray(mass_density_kg_per_m3, dtype=np.float64)
        if (
            density.shape != (len(material_ids),)
            or np.any(~np.isfinite(density))
            or np.any(density <= 0.0)
        ):
            raise ValueError(
                "mass_density_kg_per_m3 must be finite and positive per material."
            )
        profile: np.ndarray | None = None
        if compton_profile_j0 is not None:
            profile = np.asarray(compton_profile_j0, dtype=np.float64)
            if (
                profile.shape != (len(material_ids),)
                or np.any(~np.isfinite(profile))
                or np.any(profile <= 0.0)
            ):
                raise ValueError(
                    "compton_profile_j0 must be finite and positive per material."
                )
        energy = np.asarray(photoelectric.energy_grid.energy_j, dtype=np.float64) * float(
            conversion_factor(JOULE, ELECTRONVOLT)
        )
        mass_coefficients = np.stack(
            tuple(
                np.asarray(table.values, dtype=np.float64)
                * float(conversion_factor(table.unit, _MASS_ATTENUATION_SI))
                for table in tables
            ),
            axis=-1,
        )
        coefficients = density[:, None, None] * mass_coefficients
        if np.any(np.sum(coefficients, axis=-1) <= 0.0):
            raise ValueError(
                "Every material/energy point requires positive total attenuation."
            )
        source_table_ids = (photoelectric.table_id, compton.table_id, rayleigh.table_id)
        source_provenance_ids = (
            photoelectric.provenance.provenance_id,
            compton.provenance.provenance_id,
            rayleigh.provenance.provenance_id,
        )
        log_interpolation = np.asarray(
            tuple(
                table.interpolation is DiagnosticPhotonInterpolationPolicy.LOG_LOG
                for table in tables
            ),
            dtype=np.bool_,
        )
        self.energy = jnp.asarray(energy)
        self.coefficients = jnp.asarray(coefficients)
        self.mass_density_kg_per_m3 = jnp.asarray(density)
        self.log_interpolation = jnp.asarray(log_interpolation)
        self.compton_profile_j0 = None if profile is None else jnp.asarray(profile)
        self.material_ids = material_ids
        self.process_ids = (
            ("photoelectric", "compton", "rayleigh")
            if len(tables) == 3
            else (
                "photoelectric",
                "compton",
                "rayleigh",
                "pair-nuclear",
                "pair-electron",
            )
        )
        self.energy_unit = ELECTRONVOLT
        self.source_table_ids = source_table_ids
        self.source_provenance_ids = source_provenance_ids
        fingerprint: dict[str, object] = {
            "kind": "prepared-governed-photon-interaction-library",
            "energy": array_tree_fingerprint(energy),
            "coefficients": array_tree_fingerprint(coefficients),
            "mass_density_kg_per_m3": array_tree_fingerprint(density),
            "log_interpolation": array_tree_fingerprint(log_interpolation),
            "materials": material_ids,
            "processes": self.process_ids,
            "energy_unit": ELECTRONVOLT.unit_id,
            "source_tables": source_table_ids,
            "source_provenance": source_provenance_ids,
        }
        if profile is not None:
            fingerprint["compton_profile_j0"] = array_tree_fingerprint(profile)
        self.library_id = canonical_fingerprint(fingerprint)

    @property
    def material_count(self) -> int:
        return len(self.material_ids)

    def evaluate(
        self, material_index: ArrayLike, photon_energy: ArrayLike, /
    ) -> RadiationCrossSectionEvaluation:
        material = jnp.asarray(material_index, dtype=jnp.int32)
        energy = jnp.asarray(photon_energy, dtype=self.energy.dtype)
        shape = jnp.broadcast_shapes(material.shape, energy.shape)
        material = jnp.broadcast_to(material, shape)
        energy = jnp.broadcast_to(energy, shape)
        supported = (
            (material >= 0)
            & (material < self.material_count)
            & jnp.isfinite(energy)
            & (energy >= self.energy[0])
            & (energy <= self.energy[-1])
        )
        safe_material = jnp.clip(material, 0, self.material_count - 1)
        safe_energy = jnp.clip(energy, self.energy[0], self.energy[-1])
        upper = jnp.searchsorted(self.energy, safe_energy, side="right")
        upper = jnp.clip(upper, 1, self.energy.size - 1)
        lower = upper - 1
        left_energy = self.energy[lower]
        right_energy = self.energy[upper]
        linear_fraction = (safe_energy - left_energy) / (right_energy - left_energy)
        log_fraction = (jnp.log(safe_energy) - jnp.log(left_energy)) / (
            jnp.log(right_energy) - jnp.log(left_energy)
        )
        left = self.coefficients[safe_material, lower]
        right = self.coefficients[safe_material, upper]
        linear_evaluated = left + linear_fraction[..., None] * (right - left)
        tiny = jnp.finfo(left.dtype).tiny
        log_evaluated = jnp.exp(
            jnp.log(jnp.maximum(left, tiny))
            + log_fraction[..., None]
            * (jnp.log(jnp.maximum(right, tiny)) - jnp.log(jnp.maximum(left, tiny)))
        )
        evaluated = jnp.where(self.log_interpolation, log_evaluated, linear_evaluated)
        total = jnp.sum(evaluated, axis=-1)
        finite = jnp.all(jnp.isfinite(evaluated), axis=-1) & jnp.isfinite(total)
        successful = (
            supported & finite & jnp.all(evaluated >= 0.0, axis=-1) & (total > 0.0)
        )
        return RadiationCrossSectionEvaluation(
            evaluated, total, supported, finite, successful, self.library_id
        )

    def majorant(self, photon_energy: ArrayLike, /) -> Array:
        energy = jnp.asarray(photon_energy, dtype=self.energy.dtype)
        material = jnp.arange(self.material_count, dtype=jnp.int32)
        evaluated = self.evaluate(
            material.reshape((self.material_count,) + (1,) * energy.ndim),
            energy[None, ...],
        )
        return jnp.max(evaluated.total, axis=0)


__all__ = [
    "ComptonKinematics",
    "RadiationCrossSectionEvaluation",
    "RadiationCrossSectionLibrary",
    "RadiationInteractionKind",
    "compton_electron_cosine",
    "doppler_scattered_energy",
    "sample_compton_profile_momentum",
    "sample_sauter_cosine",
]
