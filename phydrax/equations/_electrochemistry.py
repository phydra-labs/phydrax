#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_species import ChemicalSpeciesSchema
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT


FARADAY_CONSTANT = 96485.33212


class ElectrolyteTransportParameters(StrictModule, NonTrainableState):
    schema: ChemicalSpeciesSchema
    diffusivities: Array
    temperature: Array
    permittivity: Array
    parameters_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        diffusivities: ArrayLike,
        temperature: ArrayLike,
        permittivity: ArrayLike,
        /,
    ):
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        diffusivity = jnp.asarray(diffusivities)
        temperature_ = jnp.asarray(temperature, dtype=diffusivity.dtype)
        permittivity_ = jnp.asarray(permittivity, dtype=diffusivity.dtype)
        if (
            diffusivity.shape != (schema.species_count,)
            or temperature_.shape != ()
            or permittivity_.shape != ()
        ):
            raise ValueError("Electrolyte parameter shapes are incompatible.")
        if not bool(
            jnp.all(jnp.isfinite(diffusivity) & (diffusivity > 0.0))
            & jnp.isfinite(temperature_)
            & (temperature_ > 0.0)
            & jnp.isfinite(permittivity_)
            & (permittivity_ > 0.0)
        ):
            raise ValueError("Electrolyte parameters must be finite and positive.")
        self.schema = schema
        self.diffusivities = diffusivity
        self.temperature = temperature_
        self.permittivity = permittivity_
        self.parameters_id = canonical_fingerprint(
            {
                "kind": "electrolyte-transport-parameters",
                "schema": schema.schema_id,
                "diffusivities": array_tree_fingerprint(np.asarray(diffusivity)),
                "temperature": float(temperature_),
                "permittivity": float(permittivity_),
            }
        )


class ElectrochemicalLocalFields(StrictModule):
    concentrations: Array
    chemical_potential: Array
    electrochemical_potential: Array
    charge_density: Array
    chemical_free_energy_density: Array
    osmotic_pressure: Array
    minimum_concentration: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class AbstractElectrochemicalClosure(StrictModule, NonTrainableState, abc.ABC):
    schema: ChemicalSpeciesSchema
    closure_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        concentrations: ArrayLike,
        potential: ArrayLike,
        parameters: ElectrolyteTransportParameters,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
    ) -> ElectrochemicalLocalFields:
        raise NotImplementedError


class IdealDiluteElectrochemicalClosure(AbstractElectrochemicalClosure):
    standard_concentrations: Array

    def __init__(
        self,
        schema: ChemicalSpeciesSchema,
        /,
        *,
        standard_concentrations: ArrayLike | None = None,
    ):
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        if standard_concentrations is None:
            values = np.asarray(
                [
                    schema.phase_specs[
                        int(schema.phase_ids[index])
                    ].standard_concentration
                    for index in range(schema.species_count)
                ],
                dtype=float,
            )
        else:
            values = np.asarray(standard_concentrations, dtype=float)
        if (
            values.shape != (schema.species_count,)
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("standard_concentrations must be finite and positive.")
        self.schema = schema
        self.standard_concentrations = jnp.asarray(values)
        self.closure_id = canonical_fingerprint(
            {
                "kind": "ideal-dilute-electrochemical-closure",
                "schema": schema.schema_id,
                "standard": array_tree_fingerprint(values),
            }
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        potential: ArrayLike,
        parameters: ElectrolyteTransportParameters,
        /,
        *,
        fixed_charge: ArrayLike = 0.0,
    ) -> ElectrochemicalLocalFields:
        concentration = jnp.asarray(concentrations)
        potential_ = jnp.asarray(potential, dtype=concentration.dtype)
        fixed = jnp.asarray(fixed_charge, dtype=concentration.dtype)
        if concentration.ndim < 1 or concentration.shape[-1] != self.schema.species_count:
            raise ValueError("concentrations must end in species axis.")
        if potential_.shape != concentration.shape[:-1] or fixed.shape not in (
            (),
            potential_.shape,
        ):
            raise ValueError("Potential/fixed charge must match concentration nodes.")
        if parameters.schema.schema_id != self.schema.schema_id:
            raise ValueError("Electrolyte parameters and closure schemas must match.")
        positive = concentration > 0.0
        safe = jnp.where(positive, concentration, 1.0)
        ratio = safe / self.standard_concentrations
        thermal = UNIVERSAL_GAS_CONSTANT * parameters.temperature
        chemical = thermal * jnp.log(ratio)
        electric = (
            FARADAY_CONSTANT
            * potential_[..., None]
            * self.schema.charges.astype(concentration.dtype)
        )
        electrochemical = chemical + electric
        free_energy = thermal * jnp.sum(
            xlogy(concentration, concentration / self.standard_concentrations)
            - concentration,
            axis=-1,
        )
        osmotic = thermal * jnp.sum(concentration, axis=-1)
        charge = (
            FARADAY_CONSTANT * contract("...s,s->...", concentration, self.schema.charges)
            + fixed
        )
        minimum = jnp.min(concentration)
        successful = (
            jnp.all(positive)
            & jnp.all(jnp.isfinite(concentration))
            & jnp.all(jnp.isfinite(electrochemical))
            & jnp.all(jnp.isfinite(charge))
            & jnp.all(jnp.isfinite(free_energy))
        )
        return ElectrochemicalLocalFields(
            concentration,
            chemical,
            electrochemical,
            charge,
            free_energy,
            osmotic,
            minimum,
            successful,
            self.closure_id,
        )


__all__ = [
    "AbstractElectrochemicalClosure",
    "ElectrochemicalLocalFields",
    "ElectrolyteTransportParameters",
    "FARADAY_CONSTANT",
    "IdealDiluteElectrochemicalClosure",
]
