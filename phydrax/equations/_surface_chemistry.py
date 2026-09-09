#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._chemical_rates import AbstractChemicalRatePlan, ChemicalRateRuntime
from ._chemical_species import ChemicalSpeciesSchema


class SurfaceSpeciesSchema(StrictModule, NonTrainableState):
    names: tuple[str, ...] = eqx.field(static=True)
    element_composition: Array
    charges: Array
    site_occupancy: Array
    site_capacity: float = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        names: Sequence[str],
        element_composition: ArrayLike,
        charges: ArrayLike,
        site_occupancy: ArrayLike,
        /,
        *,
        site_capacity: float,
    ):
        names_ = tuple(str(value) for value in names)
        elements = np.asarray(element_composition, dtype=float)
        charges_ = np.asarray(charges, dtype=float)
        occupancy = np.asarray(site_occupancy, dtype=float)
        capacity = float(site_capacity)
        if (
            not names_
            or any(not value for value in names_)
            or len(set(names_)) != len(names_)
            or elements.ndim != 2
            or elements.shape[1] != len(names_)
            or charges_.shape != (len(names_),)
            or occupancy.shape != (len(names_),)
            or np.any(~np.isfinite(elements))
            or np.any(elements < 0.0)
            or np.any(~np.isfinite(charges_))
            or np.any(~np.isfinite(occupancy))
            or np.any(occupancy <= 0.0)
            or not np.isfinite(capacity)
            or capacity <= 0.0
        ):
            raise ValueError("Surface species, sites, or capacity are invalid.")
        self.names = names_
        self.element_composition = jnp.asarray(elements)
        self.charges = jnp.asarray(charges_)
        self.site_occupancy = jnp.asarray(occupancy)
        self.site_capacity = capacity
        self.schema_id = canonical_fingerprint(
            {
                "kind": "surface-species-schema",
                "names": names_,
                "elements": array_tree_fingerprint(self.element_composition),
                "charges": array_tree_fingerprint(self.charges),
                "site_occupancy": array_tree_fingerprint(self.site_occupancy),
                "site_capacity": capacity,
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.names)


class SurfaceChemicalState(StrictModule):
    amounts: Array
    temperature: Array
    cumulative_recession_mass: Array
    finite: Array


class GasSurfaceReactionSpec(StrictModule, NonTrainableState):
    rate: AbstractChemicalRatePlan
    gas_stoichiometry: Array
    surface_stoichiometry: Array
    gas_orders: Array
    surface_orders: Array
    reaction_heat: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    reaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        rate: AbstractChemicalRatePlan,
        gas_stoichiometry: ArrayLike,
        surface_stoichiometry: ArrayLike,
        /,
        *,
        gas_orders: ArrayLike | None = None,
        surface_orders: ArrayLike | None = None,
        reaction_heat: float = 0.0,
    ):
        name_ = str(name)
        gas = np.asarray(gas_stoichiometry, dtype=float)
        surface = np.asarray(surface_stoichiometry, dtype=float)
        gas_order = (
            np.maximum(-gas, 0.0)
            if gas_orders is None
            else np.asarray(gas_orders, dtype=float)
        )
        surface_order = (
            np.maximum(-surface, 0.0)
            if surface_orders is None
            else np.asarray(surface_orders, dtype=float)
        )
        heat = float(reaction_heat)
        if (
            not name_
            or not isinstance(rate, AbstractChemicalRatePlan)
            or gas.ndim != 1
            or surface.ndim != 1
            or gas_order.shape != gas.shape
            or surface_order.shape != surface.shape
            or np.any(~np.isfinite(gas))
            or np.any(~np.isfinite(surface))
            or np.any(~np.isfinite(gas_order))
            or np.any(gas_order < 0.0)
            or np.any(~np.isfinite(surface_order))
            or np.any(surface_order < 0.0)
            or not np.isfinite(heat)
        ):
            raise ValueError("Gas-surface reaction definition is invalid.")
        self.name = name_
        self.rate = rate
        self.gas_stoichiometry = jnp.asarray(gas)
        self.surface_stoichiometry = jnp.asarray(surface)
        self.gas_orders = jnp.asarray(gas_order)
        self.surface_orders = jnp.asarray(surface_order)
        self.reaction_heat = heat
        self.reaction_id = canonical_fingerprint(
            {
                "kind": "gas-surface-reaction",
                "name": name_,
                "rate_kind": rate.kind.value,
                "gas_stoichiometry": array_tree_fingerprint(self.gas_stoichiometry),
                "surface_stoichiometry": array_tree_fingerprint(
                    self.surface_stoichiometry
                ),
                "gas_orders": array_tree_fingerprint(self.gas_orders),
                "surface_orders": array_tree_fingerprint(self.surface_orders),
                "reaction_heat": heat,
            }
        )


class GasSurfaceChemicalEvaluation(StrictModule):
    progress_rates: Array
    gas_amount_flux: Array
    surface_amount_rate: Array
    reaction_heat_flux: Array
    site_defect: Array
    element_defect: Array
    charge_defect: Array
    explicit_step: Array
    finite: Array
    successful: Array
    mechanism_id: str = eqx.field(static=True)


class PreparedGasSurfaceMechanism(StrictModule, NonTrainableState):
    gas_schema: ChemicalSpeciesSchema
    surface_schema: SurfaceSpeciesSchema
    reactions: tuple[GasSurfaceReactionSpec, ...]
    gas_stoichiometry: Array
    surface_stoichiometry: Array
    gas_orders: Array
    surface_orders: Array
    reaction_heats: Array
    mechanism_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas_schema: ChemicalSpeciesSchema,
        surface_schema: SurfaceSpeciesSchema,
        reactions: Sequence[GasSurfaceReactionSpec],
        /,
    ):
        reactions_ = tuple(reactions)
        if (
            not isinstance(gas_schema, ChemicalSpeciesSchema)
            or not isinstance(surface_schema, SurfaceSpeciesSchema)
            or not reactions_
            or any(not isinstance(value, GasSurfaceReactionSpec) for value in reactions_)
            or any(
                value.gas_stoichiometry.shape != (gas_schema.species_count,)
                for value in reactions_
            )
            or any(
                value.surface_stoichiometry.shape != (surface_schema.species_count,)
                for value in reactions_
            )
        ):
            raise ValueError("Gas-surface mechanism schemas or reactions are invalid.")
        gas_stoichiometry = jnp.stack(
            tuple(value.gas_stoichiometry for value in reactions_)
        )
        surface_stoichiometry = jnp.stack(
            tuple(value.surface_stoichiometry for value in reactions_)
        )
        gas_element = contract(
            "es,rs->er", gas_schema.element_composition, gas_stoichiometry, backend="jax"
        )
        surface_element = contract(
            "es,rs->er",
            surface_schema.element_composition,
            surface_stoichiometry,
            backend="jax",
        )
        element_defect = gas_element + surface_element
        charge_defect = contract(
            "s,rs->r", gas_schema.charges, gas_stoichiometry, backend="jax"
        ) + contract(
            "s,rs->r", surface_schema.charges, surface_stoichiometry, backend="jax"
        )
        site_defect = contract(
            "s,rs->r", surface_schema.site_occupancy, surface_stoichiometry, backend="jax"
        )
        if (
            float(jnp.max(jnp.abs(element_defect), initial=0.0)) > 1.0e-12
            or float(jnp.max(jnp.abs(charge_defect), initial=0.0)) > 1.0e-12
            or float(jnp.max(jnp.abs(site_defect), initial=0.0)) > 1.0e-12
        ):
            raise ValueError("Gas-surface reactions violate elements, charge, or sites.")
        self.gas_schema = gas_schema
        self.surface_schema = surface_schema
        self.reactions = reactions_
        self.gas_stoichiometry = gas_stoichiometry
        self.surface_stoichiometry = surface_stoichiometry
        self.gas_orders = jnp.stack(tuple(value.gas_orders for value in reactions_))
        self.surface_orders = jnp.stack(
            tuple(value.surface_orders for value in reactions_)
        )
        self.reaction_heats = jnp.asarray(
            tuple(value.reaction_heat for value in reactions_)
        )
        self.mechanism_id = canonical_fingerprint(
            {
                "kind": "prepared-gas-surface-mechanism",
                "gas_schema": gas_schema.schema_id,
                "surface_schema": surface_schema.schema_id,
                "reactions": tuple(value.reaction_id for value in reactions_),
            }
        )

    def evaluate(
        self,
        gas_concentrations: ArrayLike,
        surface_state: SurfaceChemicalState,
        pressure: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ) -> GasSurfaceChemicalEvaluation:
        gas = jnp.asarray(gas_concentrations)
        surface = jnp.asarray(surface_state.amounts, dtype=gas.dtype)
        temperature = jnp.asarray(surface_state.temperature, dtype=gas.dtype)
        pressure_ = jnp.asarray(pressure, dtype=gas.dtype)
        shape = gas.shape[:-1]
        if (
            gas.shape[-1] != self.gas_schema.species_count
            or surface.shape != shape + (self.surface_schema.species_count,)
            or temperature.shape != shape
            or pressure_.shape != shape
        ):
            raise ValueError("Gas-surface state shapes are incompatible.")
        runtime_ = ChemicalRateRuntime() if runtime is None else runtime
        gas_safe = jnp.maximum(gas, jnp.finfo(gas.dtype).tiny)
        coverage = surface / self.surface_schema.site_capacity
        coverage_safe = jnp.maximum(coverage, jnp.finfo(gas.dtype).tiny)
        progress = []
        for index, reaction in enumerate(self.reactions):
            constant = reaction.rate.evaluate(temperature, pressure_, gas, runtime_)
            gas_action = jnp.exp(
                contract(
                    "...s,s->...",
                    jnp.log(gas_safe),
                    self.gas_orders[index],
                    backend="jax",
                )
            )
            surface_action = jnp.exp(
                contract(
                    "...s,s->...",
                    jnp.log(coverage_safe),
                    self.surface_orders[index],
                    backend="jax",
                )
            )
            progress.append(constant * gas_action * surface_action)
        progress_rate = jnp.stack(tuple(progress), axis=-1)
        gas_flux = contract(
            "...r,rs->...s", progress_rate, self.gas_stoichiometry, backend="jax"
        )
        surface_rate = contract(
            "...r,rs->...s", progress_rate, self.surface_stoichiometry, backend="jax"
        )
        heat = contract("...r,r->...", progress_rate, self.reaction_heats, backend="jax")
        site_defect = contract(
            "s,...s->...", self.surface_schema.site_occupancy, surface_rate, backend="jax"
        )
        element_defect = contract(
            "es,...s->...e", self.gas_schema.element_composition, gas_flux, backend="jax"
        ) + contract(
            "es,...s->...e",
            self.surface_schema.element_composition,
            surface_rate,
            backend="jax",
        )
        charge_defect = contract(
            "s,...s->...", self.gas_schema.charges, gas_flux, backend="jax"
        ) + contract(
            "s,...s->...", self.surface_schema.charges, surface_rate, backend="jax"
        )
        consuming = surface_rate < 0.0
        explicit_step = jnp.min(
            jnp.where(
                consuming,
                surface / jnp.maximum(-surface_rate, jnp.finfo(gas.dtype).tiny),
                jnp.inf,
            ),
            axis=-1,
        )
        finite = (
            jnp.all(jnp.isfinite(progress_rate), axis=-1)
            & jnp.all(jnp.isfinite(gas_flux), axis=-1)
            & jnp.all(jnp.isfinite(surface_rate), axis=-1)
        )
        tolerance = (
            512.0
            * jnp.finfo(gas.dtype).eps
            * jnp.maximum(jnp.max(jnp.abs(progress_rate), axis=-1), 1.0)
        )
        successful = (
            finite
            & jnp.all(gas >= 0.0, axis=-1)
            & jnp.all(surface >= 0.0, axis=-1)
            & jnp.all(jnp.abs(element_defect) <= tolerance[..., None], axis=-1)
            & (jnp.abs(charge_defect) <= tolerance)
            & (jnp.abs(site_defect) <= tolerance)
        )
        return GasSurfaceChemicalEvaluation(
            progress_rate,
            gas_flux,
            surface_rate,
            heat,
            site_defect,
            element_defect,
            charge_defect,
            explicit_step,
            finite,
            successful,
            self.mechanism_id,
        )


__all__ = [
    "GasSurfaceChemicalEvaluation",
    "GasSurfaceReactionSpec",
    "PreparedGasSurfaceMechanism",
    "SurfaceChemicalState",
    "SurfaceSpeciesSchema",
]
