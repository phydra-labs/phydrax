#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._chemical_rates import (
    AbstractChemicalRatePlan,
    ChemicalRateRuntime,
    LindemannRatePlan,
    ThirdBodyRatePlan,
)
from ._chemical_species import ChemicalPhaseKind, ChemicalSpeciesSchema
from ._chemical_thermodynamics import (
    AbstractSpeciesThermodynamicsPlan,
    SpeciesThermodynamicEvaluation,
    UNIVERSAL_GAS_CONSTANT,
)


class ChemicalReactionSpec(StrictModule):
    name: str = eqx.field(static=True)
    reactants: tuple[tuple[str, float], ...] = eqx.field(static=True)
    products: tuple[tuple[str, float], ...] = eqx.field(static=True)
    forward_orders: tuple[tuple[str, float], ...] = eqx.field(static=True)
    forward_rate: AbstractChemicalRatePlan
    reverse_rate: AbstractChemicalRatePlan | None
    thermodynamic_reversible: bool = eqx.field(static=True)
    duplicate_group: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        reactants,
        products,
        forward_rate: AbstractChemicalRatePlan,
        /,
        *,
        forward_orders=None,
        reverse_rate: AbstractChemicalRatePlan | None = None,
        thermodynamic_reversible: bool = False,
        duplicate_group: str | None = None,
    ):
        name_ = str(name)
        reactant_values = _normalized_stoichiometry(reactants, "reactants")
        product_values = _normalized_stoichiometry(products, "products")
        order_values = (
            reactant_values
            if forward_orders is None
            else _normalized_stoichiometry(forward_orders, "forward_orders")
        )
        if not name_:
            raise ValueError("Reaction name must be nonempty.")
        if not isinstance(forward_rate, AbstractChemicalRatePlan):
            raise TypeError("forward_rate must implement AbstractChemicalRatePlan.")
        if reverse_rate is not None and not isinstance(
            reverse_rate, AbstractChemicalRatePlan
        ):
            raise TypeError("reverse_rate must implement AbstractChemicalRatePlan.")
        if reverse_rate is not None and thermodynamic_reversible:
            raise ValueError("Choose explicit or thermodynamic reverse rate, not both.")
        if reactant_values == product_values:
            raise ValueError("Reaction must change stoichiometry.")
        group = None if duplicate_group is None else str(duplicate_group)
        if group == "":
            raise ValueError("duplicate_group must be nonempty when provided.")
        self.name = name_
        self.reactants = reactant_values
        self.products = product_values
        self.forward_orders = order_values
        self.forward_rate = forward_rate
        self.reverse_rate = reverse_rate
        self.thermodynamic_reversible = bool(thermodynamic_reversible)
        self.duplicate_group = group


class ChemicalMechanismIR(StrictModule):
    schema: ChemicalSpeciesSchema
    thermodynamics: AbstractSpeciesThermodynamicsPlan
    reactions: tuple[ChemicalReactionSpec, ...]
    name: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        schema: ChemicalSpeciesSchema,
        thermodynamics: AbstractSpeciesThermodynamicsPlan,
        reactions,
        /,
    ):
        name_ = str(name)
        reaction_values = tuple(reactions)
        if not name_:
            raise ValueError("Mechanism name must be nonempty.")
        if not isinstance(schema, ChemicalSpeciesSchema):
            raise TypeError("schema must be ChemicalSpeciesSchema.")
        if not isinstance(thermodynamics, AbstractSpeciesThermodynamicsPlan):
            raise TypeError(
                "thermodynamics must implement AbstractSpeciesThermodynamicsPlan."
            )
        if thermodynamics.schema.schema_id != schema.schema_id:
            raise ValueError("Thermodynamics and mechanism schemas must match exactly.")
        if not reaction_values or any(
            not isinstance(value, ChemicalReactionSpec) for value in reaction_values
        ):
            raise TypeError("reactions must contain ChemicalReactionSpec objects.")
        if len({value.name for value in reaction_values}) != len(reaction_values):
            duplicate_names = [value.name for value in reaction_values]
            duplicate_groups = [value.duplicate_group for value in reaction_values]
            if any(group is None for group in duplicate_groups) or len(
                set(zip(duplicate_names, duplicate_groups, strict=True))
            ) != len(reaction_values):
                raise ValueError(
                    "Duplicate reaction names require unique duplicate groups."
                )
        self.name = name_
        self.schema = schema
        self.thermodynamics = thermodynamics
        self.reactions = reaction_values

    def prepare(self) -> PreparedChemicalMechanism:
        return PreparedChemicalMechanism(self)


class ChemicalMechanismEvidence(StrictModule):
    maximum_element_residual: Array
    maximum_charge_residual: Array
    balanced: Array
    mechanism_id: str = eqx.field(static=True)


class ChemicalRateEvaluation(StrictModule):
    forward_rate_constants: Array
    reverse_rate_constants: Array
    forward_progress_rates: Array
    reverse_progress_rates: Array
    net_progress_rates: Array
    species_amount_rate: Array
    element_residual: Array
    charge_residual: Array
    reactant_margin: Array
    explicit_step_restriction: Array
    thermodynamics: SpeciesThermodynamicEvaluation
    successful: Array
    mechanism_id: str = eqx.field(static=True)


class PreparedChemicalMechanism(StrictModule):
    schema: ChemicalSpeciesSchema
    thermodynamics: AbstractSpeciesThermodynamicsPlan
    reactions: tuple[ChemicalReactionSpec, ...]
    reactant_stoichiometry: Array
    product_stoichiometry: Array
    net_stoichiometry: Array
    forward_orders: Array
    thermodynamic_reverse: Array
    reaction_count: int = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    preparation_evidence: ChemicalMechanismEvidence

    def __init__(self, mechanism: ChemicalMechanismIR, /):
        if not isinstance(mechanism, ChemicalMechanismIR):
            raise TypeError("mechanism must be ChemicalMechanismIR.")
        species_index = {
            name: index for index, name in enumerate(mechanism.schema.species_names)
        }
        reaction_count = len(mechanism.reactions)
        species_count = mechanism.schema.species_count
        reactant = np.zeros((reaction_count, species_count), dtype=float)
        product = np.zeros_like(reactant)
        orders = np.zeros_like(reactant)
        thermodynamic_reverse = np.zeros(reaction_count, dtype=bool)
        for reaction_index, reaction in enumerate(mechanism.reactions):
            for species, coefficient in reaction.reactants:
                if species not in species_index:
                    raise ValueError(
                        f"Reaction {reaction.name!r} references unknown reactant {species!r}."
                    )
                reactant[reaction_index, species_index[species]] = coefficient
            for species, coefficient in reaction.products:
                if species not in species_index:
                    raise ValueError(
                        f"Reaction {reaction.name!r} references unknown product {species!r}."
                    )
                product[reaction_index, species_index[species]] = coefficient
            for species, coefficient in reaction.forward_orders:
                if species not in species_index:
                    raise ValueError(
                        f"Reaction {reaction.name!r} references unknown order species {species!r}."
                    )
                orders[reaction_index, species_index[species]] = coefficient
            thermodynamic_reverse[reaction_index] = reaction.thermodynamic_reversible
            _validate_rate_species_axis(
                reaction.forward_rate, species_count, reaction.name
            )
            if reaction.reverse_rate is not None:
                _validate_rate_species_axis(
                    reaction.reverse_rate, species_count, reaction.name
                )
        net = product - reactant
        element_residual = np.asarray(mechanism.schema.element_composition) @ net.T
        charge_residual = np.asarray(mechanism.schema.charges) @ net.T
        maximum_element = float(np.max(np.abs(element_residual), initial=0.0))
        maximum_charge = float(np.max(np.abs(charge_residual), initial=0.0))
        if maximum_element > 1.0e-12 or maximum_charge > 1.0e-12:
            raise ValueError(
                "Chemical mechanism violates element or charge conservation: "
                f"element={maximum_element:.3e}, charge={maximum_charge:.3e}."
            )
        generated = canonical_fingerprint(
            {
                "kind": "prepared-chemical-mechanism",
                "name": mechanism.name,
                "schema": mechanism.schema.schema_id,
                "thermodynamics": mechanism.thermodynamics.thermodynamics_id,
                "reactant": array_tree_fingerprint(reactant),
                "product": array_tree_fingerprint(product),
                "orders": array_tree_fingerprint(orders),
                "rate_kinds": [
                    value.forward_rate.kind.value for value in mechanism.reactions
                ],
                "reverse": [
                    "thermodynamic"
                    if value.thermodynamic_reversible
                    else None
                    if value.reverse_rate is None
                    else value.reverse_rate.kind.value
                    for value in mechanism.reactions
                ],
            }
        )
        self.schema = mechanism.schema
        self.thermodynamics = mechanism.thermodynamics
        self.reactions = mechanism.reactions
        self.reactant_stoichiometry = jnp.asarray(reactant)
        self.product_stoichiometry = jnp.asarray(product)
        self.net_stoichiometry = jnp.asarray(net)
        self.forward_orders = jnp.asarray(orders)
        self.thermodynamic_reverse = jnp.asarray(thermodynamic_reverse)
        self.reaction_count = reaction_count
        self.mechanism_id = generated
        self.preparation_evidence = ChemicalMechanismEvidence(
            jnp.asarray(maximum_element),
            jnp.asarray(maximum_charge),
            jnp.asarray(True),
            generated,
        )

    def evaluate(
        self,
        concentrations: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ) -> ChemicalRateEvaluation:
        concentration = jnp.asarray(concentrations)
        temperature_value = jnp.asarray(temperature, dtype=concentration.dtype)
        pressure_value = jnp.asarray(pressure, dtype=concentration.dtype)
        if concentration.ndim < 1 or concentration.shape[-1] != self.schema.species_count:
            raise ValueError("concentrations must end in the mechanism species axis.")
        if concentration.shape[:-1] != temperature_value.shape or (
            pressure_value.shape != temperature_value.shape
        ):
            raise ValueError(
                "temperature and pressure must match concentration leading shape."
            )
        runtime_value = (
            ChemicalRateRuntime(
                jnp.zeros((0,), dtype=concentration.dtype),
                jnp.asarray(0.0, dtype=concentration.dtype),
            )
            if runtime is None
            else runtime
        )
        if not isinstance(runtime_value, ChemicalRateRuntime):
            raise TypeError("runtime must be ChemicalRateRuntime.")
        if runtime_value.overpotential.shape not in ((), temperature_value.shape):
            raise ValueError("Runtime overpotential must be scalar or match state shape.")
        thermo = self.thermodynamics.evaluate(temperature_value)
        forward_constants = []
        reverse_constants = []
        for reaction_index, reaction in enumerate(self.reactions):
            forward = reaction.forward_rate.evaluate(
                temperature_value,
                pressure_value,
                concentration,
                runtime_value,
            )
            forward = jnp.broadcast_to(forward, temperature_value.shape)
            if reaction.reverse_rate is not None:
                reverse = reaction.reverse_rate.evaluate(
                    temperature_value,
                    pressure_value,
                    concentration,
                    runtime_value,
                )
                reverse = jnp.broadcast_to(reverse, temperature_value.shape)
            elif reaction.thermodynamic_reversible:
                reverse = self._thermodynamic_reverse_rate(
                    reaction_index,
                    forward,
                    temperature_value,
                    thermo,
                )
            else:
                reverse = jnp.zeros_like(forward)
            forward_constants.append(forward)
            reverse_constants.append(reverse)
        forward_constant = jnp.stack(forward_constants, axis=-1)
        reverse_constant = jnp.stack(reverse_constants, axis=-1)
        forward_mass_action, forward_feasible = _mass_action(
            concentration, self.forward_orders
        )
        reverse_mass_action, reverse_feasible = _mass_action(
            concentration, self.product_stoichiometry
        )
        forward_progress = forward_constant * forward_mass_action
        reverse_progress = reverse_constant * reverse_mass_action
        net_progress = forward_progress - reverse_progress
        species_rate = contract("...r,rs->...s", net_progress, self.net_stoichiometry)
        element_residual = contract(
            "es,...s->...e", self.schema.element_composition, species_rate
        )
        charge_residual = contract("s,...s->...", self.schema.charges, species_rate)
        required = self.forward_orders > 0.0
        safe_margin = jnp.where(
            required,
            concentration[..., None, :],
            jnp.inf,
        )
        reactant_margin = jnp.min(safe_margin, axis=(-2, -1))
        consuming = species_rate < 0.0
        restriction = jnp.min(
            jnp.where(
                consuming,
                concentration
                / jnp.maximum(-species_rate, jnp.finfo(concentration.dtype).tiny),
                jnp.inf,
            ),
            axis=-1,
        )
        scale = jnp.maximum(jnp.max(jnp.abs(species_rate), axis=-1), 1.0)
        tolerance = 256.0 * jnp.finfo(concentration.dtype).eps * scale
        successful = (
            thermo.successful
            & jnp.all(jnp.isfinite(concentration), axis=-1)
            & jnp.all(concentration >= 0.0, axis=-1)
            & jnp.all(
                jnp.isfinite(forward_constant) & (forward_constant >= 0.0),
                axis=-1,
            )
            & jnp.all(
                jnp.isfinite(reverse_constant) & (reverse_constant >= 0.0),
                axis=-1,
            )
            & jnp.all(forward_feasible | (forward_progress == 0.0), axis=-1)
            & jnp.all(reverse_feasible | (reverse_progress == 0.0), axis=-1)
            & jnp.all(
                jnp.abs(element_residual) <= tolerance[..., None],
                axis=-1,
            )
            & (jnp.abs(charge_residual) <= tolerance)
            & jnp.all(jnp.isfinite(species_rate), axis=-1)
        )
        return ChemicalRateEvaluation(
            forward_constant,
            reverse_constant,
            forward_progress,
            reverse_progress,
            net_progress,
            species_rate,
            element_residual,
            charge_residual,
            reactant_margin,
            restriction,
            thermo,
            successful,
            self.mechanism_id,
        )

    def _thermodynamic_reverse_rate(
        self,
        reaction_index,
        forward_rate,
        temperature,
        thermodynamics,
    ):
        net = self.net_stoichiometry[reaction_index]
        delta_gibbs = jnp.sum(net * thermodynamics.molar_gibbs_energy, axis=-1)
        logarithmic_equilibrium = -delta_gibbs / (UNIVERSAL_GAS_CONSTANT * temperature)
        standard_concentrations = []
        phase_ids = tuple(int(value) for value in np.asarray(self.schema.phase_ids))
        for phase_id in phase_ids:
            phase = self.schema.phase_specs[phase_id]
            if phase.kind is ChemicalPhaseKind.GAS:
                assert phase.standard_pressure is not None
                standard = phase.standard_pressure / (
                    UNIVERSAL_GAS_CONSTANT * temperature
                )
            else:
                standard = jnp.full_like(
                    temperature,
                    phase.standard_concentration,
                )
            standard_concentrations.append(standard)
        standard = jnp.stack(standard_concentrations, axis=-1)
        logarithmic_rate_ratio = logarithmic_equilibrium + jnp.sum(
            net * jnp.log(standard), axis=-1
        )
        return forward_rate * jnp.exp(-logarithmic_rate_ratio)


def _normalized_stoichiometry(values, name):
    if isinstance(values, Mapping):
        entries = tuple((str(key), float(value)) for key, value in values.items())
    else:
        entries = tuple((str(key), float(value)) for key, value in values)
    if (
        not entries
        or any(not key for key, _ in entries)
        or len({key for key, _ in entries}) != len(entries)
        or any(not np.isfinite(value) or value < 0.0 for _, value in entries)
        or all(value == 0.0 for _, value in entries)
    ):
        raise ValueError(f"{name} must contain unique nonnegative coefficients.")
    return tuple((key, value) for key, value in entries if value > 0.0)


def _mass_action(concentrations, orders):
    required = orders > 0.0
    expanded = concentrations[..., None, :]
    feasible = jnp.all((~required) | (expanded > 0.0), axis=-1)
    safe = jnp.where(required & (expanded > 0.0), expanded, 1.0)
    product = jnp.prod(jnp.where(required, safe**orders, 1.0), axis=-1)
    return jnp.where(feasible, product, 0.0), feasible


def _validate_rate_species_axis(rate, species_count, reaction_name):
    if isinstance(rate, (ThirdBodyRatePlan, LindemannRatePlan)) and (
        rate.efficiencies.shape != (species_count,)
    ):
        raise ValueError(
            f"Reaction {reaction_name!r} third-body efficiencies must match species."
        )


__all__ = [
    "ChemicalMechanismEvidence",
    "ChemicalMechanismIR",
    "ChemicalRateEvaluation",
    "ChemicalReactionSpec",
    "PreparedChemicalMechanism",
]
