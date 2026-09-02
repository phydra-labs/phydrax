#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_mechanism import (
    ChemicalMechanismIR,
    ChemicalRateEvaluation,
    PreparedChemicalMechanism,
)
from ...equations._chemical_rates import ChemicalRateKind, ChemicalRateRuntime
from ...equations._chemical_species import ChemicalPhaseKind
from ._thermodynamics import ReactingGasModel


_SUPPORTED_GAS_RATE_KINDS = (
    ChemicalRateKind.ARRHENIUS,
    ChemicalRateKind.THIRD_BODY,
    ChemicalRateKind.LINDEMANN,
    ChemicalRateKind.TROE,
    ChemicalRateKind.PLOG,
    ChemicalRateKind.CHEBYSHEV,
)


class ChemicalMechanismFeatureReport(StrictModule, NonTrainableState):
    reaction_count: int = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    rate_kind_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    reversible_reaction_count: int = eqx.field(static=True)
    explicit_reverse_reaction_count: int = eqx.field(static=True)
    third_body_reaction_count: int = eqx.field(static=True)
    falloff_reaction_count: int = eqx.field(static=True)
    pressure_dependent_reaction_count: int = eqx.field(static=True)
    unsupported_features: tuple[str, ...] = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class CompiledMechanismEvidence(StrictModule):
    element_residual: Array
    charge_residual: Array
    energy_residual: Array
    minimum_concentration: Array
    successful: Array
    mechanism_id: str = eqx.field(static=True)


class CompiledMechanismEvaluation(StrictModule):
    rates: ChemicalRateEvaluation
    species_molar_production_rate: Array
    species_mass_production_rate: Array
    heat_release_rate: Array
    evidence: CompiledMechanismEvidence
    mechanism_id: str = eqx.field(static=True)


class CompiledChemicalMechanism(StrictModule, NonTrainableState):
    prepared: PreparedChemicalMechanism
    gas_model: ReactingGasModel
    features: ChemicalMechanismFeatureReport
    conservation_tolerance: float = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)

    def evaluate(
        self,
        concentrations: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ) -> CompiledMechanismEvaluation:
        concentration = jnp.asarray(concentrations)
        if (
            concentration.ndim < 1
            or concentration.shape[-1] != self.prepared.schema.species_count
        ):
            raise ValueError("concentrations must end in the mechanism species axis.")
        cell_shape = concentration.shape[:-1]
        temperature_ = jnp.asarray(temperature, dtype=concentration.dtype)
        pressure_ = jnp.asarray(pressure, dtype=concentration.dtype)
        if temperature_.shape not in ((), cell_shape) or pressure_.shape not in (
            (),
            cell_shape,
        ):
            raise ValueError(
                "temperature and pressure must be scalar or match concentration cells."
            )
        temperature_ = jnp.broadcast_to(temperature_, cell_shape)
        pressure_ = jnp.broadcast_to(pressure_, cell_shape)
        rates = self.prepared.evaluate(
            concentration,
            temperature_,
            pressure_,
            runtime=runtime,
        )
        molar_rate = rates.species_amount_rate
        mass_rate = molar_rate * self.prepared.schema.molar_masses
        species_thermo = self.gas_model.thermodynamics.evaluate(temperature_)
        molar_enthalpy = (
            species_thermo.molar_enthalpy + self.gas_model.formation_molar_enthalpies
        )
        heat_release = -contract("...s,...s->...", molar_rate, molar_enthalpy)
        energy_residual = heat_release + contract(
            "...s,...s->...", molar_rate, molar_enthalpy
        )
        element_residual = contract(
            "es,...s->...e", self.prepared.schema.element_composition, molar_rate
        )
        charge_residual = contract(
            "s,...s->...", self.prepared.schema.charges, molar_rate
        )
        scale = jnp.maximum(jnp.max(jnp.abs(molar_rate), axis=-1), 1.0)
        tolerance = self.conservation_tolerance * scale
        successful = (
            rates.successful
            & species_thermo.successful
            & jnp.all(jnp.isfinite(concentration), axis=-1)
            & jnp.all(concentration >= 0.0, axis=-1)
            & jnp.all(jnp.abs(element_residual) <= tolerance[..., None], axis=-1)
            & (jnp.abs(charge_residual) <= tolerance)
            & (jnp.abs(energy_residual) <= tolerance)
            & jnp.all(jnp.isfinite(mass_rate), axis=-1)
            & jnp.isfinite(heat_release)
        )
        evidence = CompiledMechanismEvidence(
            element_residual,
            charge_residual,
            energy_residual,
            jnp.min(concentration, axis=-1),
            successful,
            self.mechanism_id,
        )
        return CompiledMechanismEvaluation(
            rates,
            molar_rate,
            mass_rate,
            heat_release,
            evidence,
            self.mechanism_id,
        )

    def source_from_density_mass_fractions(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass_fractions: ArrayLike,
        /,
        *,
        runtime: ChemicalRateRuntime | None = None,
    ) -> CompiledMechanismEvaluation:
        density_ = jnp.asarray(density)
        mass = jnp.asarray(mass_fractions, dtype=density_.dtype)
        if mass.shape != density_.shape + (self.prepared.schema.species_count,):
            raise ValueError("mass_fractions must match density and species count.")
        concentrations = density_[..., None] * mass / self.prepared.schema.molar_masses
        return self.evaluate(
            concentrations,
            temperature,
            pressure,
            runtime=runtime,
        )


class ChemicalMechanismCompiler(StrictModule, NonTrainableState):
    """Compile supported gas-phase mechanism features into immutable arrays."""

    maximum_species: int = eqx.field(static=True)
    maximum_reactions: int = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    compiler_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_species: int = 256,
        maximum_reactions: int = 4096,
        conservation_tolerance: float = 1.0e-10,
    ):
        species = int(maximum_species)
        reactions = int(maximum_reactions)
        tolerance = float(conservation_tolerance)
        if species < 2 or reactions < 1:
            raise ValueError("Compiler species/reaction bounds are invalid.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be finite and positive.")
        self.maximum_species = species
        self.maximum_reactions = reactions
        self.conservation_tolerance = tolerance
        self.compiler_id = canonical_fingerprint(
            {
                "kind": "reactive-chemical-mechanism-compiler",
                "maximum_species": species,
                "maximum_reactions": reactions,
                "conservation_tolerance": tolerance,
                "supported_rates": [value.value for value in _SUPPORTED_GAS_RATE_KINDS],
            }
        )

    def inspect(
        self, mechanism: ChemicalMechanismIR, /
    ) -> ChemicalMechanismFeatureReport:
        if not isinstance(mechanism, ChemicalMechanismIR):
            raise TypeError("mechanism must be ChemicalMechanismIR.")
        counts = {kind: 0 for kind in ChemicalRateKind}
        unsupported: list[str] = []
        if mechanism.schema.species_count > self.maximum_species:
            unsupported.append(
                f"species-count:{mechanism.schema.species_count}>{self.maximum_species}"
            )
        if len(mechanism.reactions) > self.maximum_reactions:
            unsupported.append(
                f"reaction-count:{len(mechanism.reactions)}>{self.maximum_reactions}"
            )
        if any(phase is not ChemicalPhaseKind.GAS for phase in mechanism.schema.phases):
            unsupported.append("non-gas-phase")
        for reaction in mechanism.reactions:
            counts[reaction.forward_rate.kind] += 1
            if reaction.forward_rate.kind not in _SUPPORTED_GAS_RATE_KINDS:
                unsupported.append(
                    f"reaction:{reaction.name}:forward:{reaction.forward_rate.kind.value}"
                )
            if reaction.reverse_rate is not None:
                counts[reaction.reverse_rate.kind] += 1
                if reaction.reverse_rate.kind not in _SUPPORTED_GAS_RATE_KINDS:
                    unsupported.append(
                        f"reaction:{reaction.name}:reverse:{reaction.reverse_rate.kind.value}"
                    )
        kind_counts = tuple(
            (kind.value, counts[kind]) for kind in ChemicalRateKind if counts[kind] > 0
        )
        third_body = counts[ChemicalRateKind.THIRD_BODY]
        falloff = counts[ChemicalRateKind.LINDEMANN] + counts[ChemicalRateKind.TROE]
        pressure_dependent = (
            counts[ChemicalRateKind.PLOG] + counts[ChemicalRateKind.CHEBYSHEV]
        )
        reversible = sum(
            reaction.thermodynamic_reversible for reaction in mechanism.reactions
        )
        explicit_reverse = sum(
            reaction.reverse_rate is not None for reaction in mechanism.reactions
        )
        unsupported_values = tuple(dict.fromkeys(unsupported))
        report_id = canonical_fingerprint(
            {
                "kind": "chemical-mechanism-feature-report",
                "compiler": self.compiler_id,
                "mechanism": mechanism.name,
                "schema": mechanism.schema.schema_id,
                "counts": list(kind_counts),
                "unsupported": list(unsupported_values),
            }
        )
        return ChemicalMechanismFeatureReport(
            len(mechanism.reactions),
            mechanism.schema.species_count,
            kind_counts,
            reversible,
            explicit_reverse,
            third_body,
            falloff,
            pressure_dependent,
            unsupported_values,
            not unsupported_values,
            report_id,
        )

    def compile(
        self,
        mechanism: ChemicalMechanismIR,
        /,
        *,
        gas_model: ReactingGasModel | None = None,
    ) -> CompiledChemicalMechanism:
        report = self.inspect(mechanism)
        if not report.supported:
            features = ", ".join(report.unsupported_features)
            raise ValueError(f"Unsupported reacting-flow mechanism features: {features}.")
        model = (
            ReactingGasModel(mechanism.schema, mechanism.thermodynamics)
            if gas_model is None
            else gas_model
        )
        if not isinstance(model, ReactingGasModel):
            raise TypeError("gas_model must be ReactingGasModel or None.")
        if (
            model.schema.schema_id != mechanism.schema.schema_id
            or model.thermodynamics.thermodynamics_id
            != mechanism.thermodynamics.thermodynamics_id
        ):
            raise ValueError("Compiler mechanism and gas model must match exactly.")
        prepared = ChemicalMechanismIR(
            mechanism.name,
            mechanism.schema,
            model.mechanism_thermodynamics(),
            mechanism.reactions,
        ).prepare()
        mechanism_id = canonical_fingerprint(
            {
                "kind": "compiled-reactive-chemical-mechanism",
                "compiler": self.compiler_id,
                "prepared": prepared.mechanism_id,
                "gas_model": model.model_id,
                "features": report.report_id,
                "forward_rate_parameters": array_tree_fingerprint(
                    tuple(reaction.forward_rate for reaction in mechanism.reactions)
                ),
                "reverse_rate_parameters": array_tree_fingerprint(
                    tuple(
                        reaction.reverse_rate
                        for reaction in mechanism.reactions
                        if reaction.reverse_rate is not None
                    )
                ),
            }
        )
        return CompiledChemicalMechanism(
            prepared,
            model,
            report,
            self.conservation_tolerance,
            mechanism_id,
        )


__all__ = [
    "ChemicalMechanismCompiler",
    "ChemicalMechanismFeatureReport",
    "CompiledChemicalMechanism",
    "CompiledMechanismEvaluation",
    "CompiledMechanismEvidence",
]
