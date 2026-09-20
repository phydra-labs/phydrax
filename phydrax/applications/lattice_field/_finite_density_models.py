#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ... import special
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._finite_density import (
    ChemicalChargeConvention,
    FiniteDensityDomain,
    FiniteDensitySourceKind,
    FiniteDensityStatus,
)


class HRGSpectrum(StrictModule, NonTrainableState):
    masses: Array
    degeneracies: Array
    charges: Array
    active: Array
    convention: ChemicalChargeConvention
    source_release: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    interaction_prescription: str = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(
        self,
        masses: ArrayLike,
        degeneracies: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        active: ArrayLike | None = None,
        convention: ChemicalChargeConvention,
        source_release: str,
        checksum: str,
        interaction_prescription: str = "ideal-boltzmann",
    ):
        masses_ = np.asarray(masses, dtype=np.float64)
        degeneracies_ = np.asarray(degeneracies, dtype=np.float64)
        charges_ = np.asarray(charges, dtype=np.float64)
        if (
            masses_.ndim != 1
            or masses_.size < 1
            or degeneracies_.shape != masses_.shape
            or charges_.shape != (masses_.size, 3)
        ):
            raise ValueError("HRG spectrum arrays have incompatible shapes.")
        active_ = (
            np.ones(masses_.shape, dtype=np.bool_)
            if active is None
            else np.asarray(active, dtype=np.bool_)
        )
        if (
            active_.shape != masses_.shape
            or np.any(~np.isfinite(masses_))
            or np.any(masses_[active_] <= 0.0)
            or np.any(~np.isfinite(degeneracies_))
            or np.any(degeneracies_[active_] <= 0.0)
            or np.any(~np.isfinite(charges_))
        ):
            raise ValueError(
                "Active HRG spectrum entries must be finite with positive mass/degeneracy."
            )
        if not isinstance(convention, ChemicalChargeConvention):
            raise TypeError("convention must be ChemicalChargeConvention.")
        source = str(source_release).strip()
        checksum_ = str(checksum).strip()
        interaction = str(interaction_prescription).strip()
        if not source or not checksum_ or interaction != "ideal-boltzmann":
            raise ValueError(
                "The admitted native HRG profile is a pinned ideal-Boltzmann spectrum."
            )
        self.masses = jnp.asarray(masses_)
        self.degeneracies = jnp.asarray(degeneracies_)
        self.charges = jnp.asarray(charges_)
        self.active = jnp.asarray(active_)
        self.convention = convention
        self.source_release = source
        self.checksum = checksum_
        self.interaction_prescription = interaction
        self.spectrum_id = canonical_fingerprint(
            {
                "kind": "hrg-spectrum",
                "arrays": array_tree_fingerprint(
                    (masses_, degeneracies_, charges_, active_)
                ),
                "convention": convention.convention_id,
                "source_release": source,
                "checksum": checksum_,
                "interaction": interaction,
            }
        )


class HRGResult(StrictModule, NonTrainableState):
    pressure_over_temperature4: Array
    densities_over_temperature3: Array
    finite: Array
    source_kind: FiniteDensitySourceKind = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)


def evaluate_ideal_boltzmann_hrg(
    spectrum: HRGSpectrum,
    temperature: ArrayLike,
    chemical_potentials: ArrayLike,
    /,
) -> HRGResult:
    """Evaluate the ideal-Boltzmann HRG pressure and B/Q/S densities."""
    if not isinstance(spectrum, HRGSpectrum):
        raise TypeError("spectrum must be HRGSpectrum.")
    temperature_ = jnp.asarray(temperature, dtype=spectrum.masses.dtype).reshape(())
    chemical = jnp.asarray(chemical_potentials, dtype=temperature_.dtype)
    if chemical.shape != (3,):
        raise ValueError("chemical_potentials must contain B/Q/S values.")
    mass_ratio = spectrum.masses / temperature_
    potential = spectrum.charges @ (chemical / temperature_)
    prefactor = (
        spectrum.degeneracies
        * mass_ratio
        * mass_ratio
        * special.kv(2.0, mass_ratio)
        / (2.0 * jnp.pi**2)
    )
    species_pressure = jnp.where(spectrum.active, prefactor * jnp.cosh(potential), 0.0)
    species_density = jnp.where(
        spectrum.active[:, None],
        prefactor[:, None] * jnp.sinh(potential)[:, None] * spectrum.charges,
        0.0,
    )
    pressure = jnp.sum(species_pressure)
    densities = jnp.sum(species_density, axis=0)
    finite = (
        (temperature_ > 0.0) & jnp.isfinite(pressure) & jnp.all(jnp.isfinite(densities))
    )
    return HRGResult(
        jnp.where(finite, pressure, jnp.nan),
        jnp.where(finite, densities, jnp.nan),
        finite,
        FiniteDensitySourceKind.HRG,
        spectrum.spectrum_id,
    )


class FiniteDensityProviderGrid(StrictModule, NonTrainableState):
    """Pinned resummed or critical-model pressure grid; never a lattice estimate."""

    temperatures: Array
    baryon_chemical_potentials: Array
    regular_pressure_over_temperature4: Array
    singular_pressure_over_temperature4: Array
    valid: Array
    convention: ChemicalChargeConvention
    domain: FiniteDensityDomain
    source_kind: FiniteDensitySourceKind = eqx.field(static=True)
    provider_release: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures: ArrayLike,
        baryon_chemical_potentials: ArrayLike,
        regular_pressure_over_temperature4: ArrayLike,
        singular_pressure_over_temperature4: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        convention: ChemicalChargeConvention,
        domain: FiniteDensityDomain,
        source_kind: FiniteDensitySourceKind,
        provider_release: str,
        checksum: str,
    ):
        temperatures_ = np.asarray(temperatures, dtype=np.float64)
        baryon = np.asarray(baryon_chemical_potentials, dtype=np.float64)
        regular = np.asarray(regular_pressure_over_temperature4, dtype=np.float64)
        singular = np.asarray(singular_pressure_over_temperature4, dtype=np.float64)
        if (
            temperatures_.ndim != 1
            or baryon.ndim != 1
            or temperatures_.size < 2
            or baryon.size < 2
            or np.any(np.diff(temperatures_) <= 0.0)
            or np.any(np.diff(baryon) <= 0.0)
        ):
            raise ValueError("Provider grid axes must be increasing vectors.")
        expected = (temperatures_.size, baryon.size)
        if regular.shape != expected or singular.shape != expected:
            raise ValueError("Provider pressure components must align with grid axes.")
        valid_ = (
            np.ones(expected, dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if (
            valid_.shape != expected
            or np.any(~np.isfinite(regular[valid_]))
            or np.any(~np.isfinite(singular[valid_]))
        ):
            raise ValueError("Provider grid validity or pressure content is invalid.")
        if source_kind not in (
            FiniteDensitySourceKind.RESUMMED,
            FiniteDensitySourceKind.CRITICAL_MODEL,
        ):
            raise ValueError(
                "Provider grids are restricted to resummed or critical-model products."
            )
        if source_kind is FiniteDensitySourceKind.RESUMMED and np.any(singular != 0.0):
            raise ValueError(
                "Resummed provider grids cannot carry a critical singular component."
            )
        release = str(provider_release).strip()
        checksum_ = str(checksum).strip()
        if not release or not checksum_:
            raise ValueError("Provider release and checksum are required.")
        self.temperatures = jnp.asarray(temperatures_)
        self.baryon_chemical_potentials = jnp.asarray(baryon)
        self.regular_pressure_over_temperature4 = jnp.asarray(regular)
        self.singular_pressure_over_temperature4 = jnp.asarray(singular)
        self.valid = jnp.asarray(valid_)
        self.convention = convention
        self.domain = domain
        self.source_kind = source_kind
        self.provider_release = release
        self.checksum = checksum_
        self.grid_id = canonical_fingerprint(
            {
                "kind": "finite-density-provider-grid",
                "axes": array_tree_fingerprint((temperatures_, baryon)),
                "regular": array_tree_fingerprint(regular),
                "singular": array_tree_fingerprint(singular),
                "valid": array_tree_fingerprint(valid_),
                "convention": convention.convention_id,
                "domain": domain.domain_id,
                "source_kind": source_kind.value,
                "release": release,
                "checksum": checksum_,
            }
        )


class ProviderGridResult(StrictModule, NonTrainableState):
    pressure_over_temperature4: Array
    regular_component: Array
    singular_component: Array
    in_domain: Array
    valid: Array
    status: Array
    source_kind: FiniteDensitySourceKind = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)


def evaluate_finite_density_provider_grid(
    grid: FiniteDensityProviderGrid,
    temperature: ArrayLike,
    baryon_chemical_potential: ArrayLike,
    /,
) -> ProviderGridResult:
    """Bilinearly evaluate a provider grid without extrapolation or hole filling."""
    if not isinstance(grid, FiniteDensityProviderGrid):
        raise TypeError("grid must be FiniteDensityProviderGrid.")
    temperature_ = jnp.asarray(temperature, dtype=grid.temperatures.dtype).reshape(())
    baryon = jnp.asarray(baryon_chemical_potential, dtype=temperature_.dtype).reshape(())
    ratios = jnp.asarray([baryon / temperature_, 0.0, 0.0])
    in_domain = grid.domain.contains(temperature_, ratios)
    ti = jnp.searchsorted(grid.temperatures, temperature_, side="right") - 1
    mi = jnp.searchsorted(grid.baryon_chemical_potentials, baryon, side="right") - 1
    ti = jnp.clip(ti, 0, grid.temperatures.size - 2)
    mi = jnp.clip(mi, 0, grid.baryon_chemical_potentials.size - 2)
    t0, t1 = grid.temperatures[ti], grid.temperatures[ti + 1]
    m0, m1 = grid.baryon_chemical_potentials[mi], grid.baryon_chemical_potentials[mi + 1]
    ft = (temperature_ - t0) / (t1 - t0)
    fm = (baryon - m0) / (m1 - m0)

    def interpolate(values):
        return (
            (1.0 - ft) * (1.0 - fm) * values[ti, mi]
            + ft * (1.0 - fm) * values[ti + 1, mi]
            + (1.0 - ft) * fm * values[ti, mi + 1]
            + ft * fm * values[ti + 1, mi + 1]
        )

    regular = interpolate(grid.regular_pressure_over_temperature4)
    singular = interpolate(grid.singular_pressure_over_temperature4)
    corners_valid = (
        grid.valid[ti, mi]
        & grid.valid[ti + 1, mi]
        & grid.valid[ti, mi + 1]
        & grid.valid[ti + 1, mi + 1]
    )
    inside_axes = (
        (temperature_ >= grid.temperatures[0])
        & (temperature_ <= grid.temperatures[-1])
        & (baryon >= grid.baryon_chemical_potentials[0])
        & (baryon <= grid.baryon_chemical_potentials[-1])
    )
    valid = (
        in_domain
        & inside_axes
        & corners_valid
        & jnp.isfinite(regular)
        & jnp.isfinite(singular)
    )
    status = jnp.where(
        valid, int(FiniteDensityStatus.SUCCESS), int(FiniteDensityStatus.OUTSIDE_DOMAIN)
    )
    return ProviderGridResult(
        jnp.where(valid, regular + singular, jnp.nan),
        jnp.where(valid, regular, jnp.nan),
        jnp.where(valid, singular, jnp.nan),
        in_domain & inside_axes,
        valid,
        status.astype(jnp.int32),
        grid.source_kind,
        grid.grid_id,
    )


__all__ = [
    "FiniteDensityProviderGrid",
    "HRGResult",
    "HRGSpectrum",
    "ProviderGridResult",
    "evaluate_finite_density_provider_grid",
    "evaluate_ideal_boltzmann_hrg",
]
