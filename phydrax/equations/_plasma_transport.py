#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._dense_inverse import dense_inverse
from ._chemical_species import ChemicalSpeciesSchema
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ._electrochemistry import FARADAY_CONSTANT
from ._transport_closures import AbstractTransportClosure


class PlasmaTransportEvaluation(StrictModule):
    dynamic_viscosity: Array
    bulk_viscosity: Array
    heavy_thermal_conductivity: Array
    electron_thermal_conductivity: Array
    species_mass_flux: Array
    heavy_heat_flux: Array
    electron_heat_flux: Array
    ambipolar_electric_field: Array
    mass_flux_defect: Array
    current_density: Array
    entropy_production: Array
    maximum_diffusivity: Array
    finite: Array
    successful: Array
    transport_id: str = eqx.field(static=True)


class AmbipolarPlasmaTransportPlan(StrictModule, NonTrainableState):
    """Mass- and current-constrained multicomponent Fick plasma transport."""

    schema: ChemicalSpeciesSchema
    molecular_transport: AbstractTransportClosure
    species_diffusivities: Array
    electron_thermal_conductivity: float = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        molecular_transport: AbstractTransportClosure,
        species_diffusivities: ArrayLike,
        /,
        *,
        electron_thermal_conductivity: float,
    ):
        diffusivities = np.asarray(species_diffusivities, dtype=float)
        electron_conductivity = float(electron_thermal_conductivity)
        if (
            not isinstance(schema, ChemicalSpeciesSchema)
            or not isinstance(molecular_transport, AbstractTransportClosure)
            or diffusivities.shape != (schema.species_count,)
            or np.any(~np.isfinite(diffusivities))
            or np.any(diffusivities <= 0.0)
            or not np.isfinite(electron_conductivity)
            or electron_conductivity < 0.0
            or not np.any(np.asarray(schema.charges) != 0)
        ):
            raise ValueError("Ambipolar plasma transport data are invalid.")
        self.schema = schema
        self.molecular_transport = molecular_transport
        self.species_diffusivities = jnp.asarray(diffusivities)
        self.electron_thermal_conductivity = electron_conductivity
        self.transport_id = canonical_fingerprint(
            {
                "kind": "ambipolar-plasma-transport",
                "schema": schema.schema_id,
                "molecular_transport": molecular_transport.closure_id,
                "species_diffusivities": array_tree_fingerprint(
                    self.species_diffusivities
                ),
                "electron_thermal_conductivity": electron_conductivity,
            }
        )

    def evaluate(
        self,
        density: ArrayLike,
        mass_fractions: ArrayLike,
        mass_fraction_gradient: ArrayLike,
        heavy_temperature: ArrayLike,
        electron_temperature: ArrayLike,
        heavy_temperature_gradient: ArrayLike,
        electron_temperature_gradient: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        args: Any = None,
    ) -> PlasmaTransportEvaluation:
        density_ = jnp.asarray(density)
        mass = jnp.asarray(mass_fractions, dtype=density_.dtype)
        gradient = jnp.asarray(mass_fraction_gradient, dtype=density_.dtype)
        heavy = jnp.asarray(heavy_temperature, dtype=density_.dtype)
        electron = jnp.asarray(electron_temperature, dtype=density_.dtype)
        heavy_gradient = jnp.asarray(heavy_temperature_gradient, dtype=density_.dtype)
        electron_gradient = jnp.asarray(
            electron_temperature_gradient, dtype=density_.dtype
        )
        pressure_ = jnp.asarray(pressure, dtype=density_.dtype)
        dimension = gradient.shape[-1]
        cell_shape = density_.shape
        if (
            mass.shape != cell_shape + (self.schema.species_count,)
            or gradient.shape != cell_shape + (self.schema.species_count, dimension)
            or heavy.shape != cell_shape
            or electron.shape != cell_shape
            or pressure_.shape != cell_shape
            or heavy_gradient.shape != cell_shape + (dimension,)
            or electron_gradient.shape != cell_shape + (dimension,)
        ):
            raise ValueError("Plasma transport arrays have incompatible shapes.")
        diffusivity = self.species_diffusivities.astype(density_.dtype)
        diffusion_shape = (1,) * density_.ndim + (
            self.schema.species_count,
            1,
        )
        diffusivity_field = diffusivity.reshape(diffusion_shape)
        provisional = -density_[..., None, None] * diffusivity_field * gradient
        charge_per_mass = self.schema.charges.astype(
            density_.dtype
        ) / self.schema.molar_masses.astype(density_.dtype)
        charge_scale = jnp.maximum(jnp.max(jnp.abs(charge_per_mass)), 1.0)
        constraint = jnp.stack(
            (
                jnp.ones_like(charge_per_mass),
                charge_per_mass / charge_scale,
            ),
            axis=0,
        )
        weight = (
            density_[..., None]
            * jnp.maximum(mass, jnp.finfo(density_.dtype).tiny)
            * diffusivity
        )
        gram = contract(
            "as,...s,bs->...ab", constraint, weight, constraint, backend="jax"
        )
        constraint_flux = contract(
            "as,...sd->...ad", constraint, provisional, backend="jax"
        )
        multiplier = contract(
            "...ab,...bd->...ad",
            dense_inverse(gram, positive_definite=True),
            constraint_flux,
            backend="jax",
        )
        correction = contract(
            "...s,as,...ad->...sd",
            weight,
            constraint,
            multiplier,
            backend="jax",
        )
        species_flux = provisional - correction
        mass_defect = jnp.sum(species_flux, axis=-2)
        molar_charge_flux = contract(
            "s,...sd->...d", charge_per_mass, species_flux, backend="jax"
        )
        current = FARADAY_CONSTANT * molar_charge_flux
        ambipolar_field = -multiplier[..., 1, :] / charge_scale
        properties = self.molecular_transport.properties(heavy, mass, args)
        heavy_heat = -properties.thermal_conductivity[..., None] * heavy_gradient
        electron_heat = -self.electron_thermal_conductivity * electron_gradient
        entropy = (
            jnp.sum(
                species_flux
                * species_flux
                / jnp.maximum(
                    density_[..., None, None] * diffusivity_field,
                    jnp.finfo(density_.dtype).tiny,
                ),
                axis=(-2, -1),
            )
            + properties.thermal_conductivity
            * jnp.sum(heavy_gradient * heavy_gradient, axis=-1)
            / jnp.maximum(heavy * heavy, jnp.finfo(density_.dtype).tiny)
            + self.electron_thermal_conductivity
            * jnp.sum(electron_gradient * electron_gradient, axis=-1)
            / jnp.maximum(electron * electron, jnp.finfo(density_.dtype).tiny)
        )
        maximum_diffusivity = jnp.maximum(
            jnp.max(diffusivity),
            jnp.maximum(
                properties.dynamic_viscosity
                / jnp.maximum(density_, jnp.finfo(density_.dtype).tiny),
                properties.thermal_conductivity
                / jnp.maximum(
                    density_ * UNIVERSAL_GAS_CONSTANT,
                    jnp.finfo(density_.dtype).tiny,
                ),
            ),
        )
        scale = jnp.maximum(jnp.max(jnp.abs(species_flux), axis=(-2, -1)), 1.0)
        tolerance = 512.0 * jnp.finfo(density_.dtype).eps * scale
        normalized_current = molar_charge_flux / charge_scale
        finite = (
            jnp.isfinite(density_)
            & jnp.all(jnp.isfinite(species_flux), axis=(-2, -1))
            & jnp.all(jnp.isfinite(current), axis=-1)
            & jnp.all(jnp.isfinite(mass_defect), axis=-1)
            & jnp.isfinite(entropy)
        )
        successful = (
            finite
            & (density_ > 0.0)
            & jnp.all(mass >= 0.0, axis=-1)
            & jnp.all(jnp.abs(mass_defect) <= tolerance[..., None], axis=-1)
            & jnp.all(jnp.abs(normalized_current) <= tolerance[..., None], axis=-1)
            & (entropy >= 0.0)
        )
        return PlasmaTransportEvaluation(
            properties.dynamic_viscosity,
            properties.bulk_viscosity,
            properties.thermal_conductivity,
            jnp.asarray(self.electron_thermal_conductivity, dtype=density_.dtype),
            species_flux,
            heavy_heat,
            electron_heat,
            ambipolar_field,
            mass_defect,
            current,
            entropy,
            maximum_diffusivity,
            finite,
            successful,
            self.transport_id,
        )


__all__ = ["AmbipolarPlasmaTransportPlan", "PlasmaTransportEvaluation"]
