#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.lattice_boltzmann._species import SpeciesLatticeBoltzmannState
from ..discretization.lattice_boltzmann._thermal import ThermalLatticeBoltzmannState


class ReactiveLocalStepResult(StrictModule):
    species_amount: Array
    sensible_energy: Array
    extent_increment: Array
    element_residual: Array
    energy_residual: Array
    iterations: Array
    successful: Array


class ReactiveLocalStepper(Protocol):
    def step(
        self,
        species_amount: Array,
        sensible_energy: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> ReactiveLocalStepResult: ...


class ReactiveSpeciesLatticeBoltzmannState(StrictModule):
    thermal: ThermalLatticeBoltzmannState
    species: SpeciesLatticeBoltzmannState
    reaction_extent: Array
    element_inventory: Array
    successful: Array


class ReactiveSpeciesLatticeBoltzmannDiagnostics(StrictModule):
    maximum_element_residual: Array
    maximum_energy_residual: Array
    minimum_species_amount: Array
    minimum_temperature_margin: Array
    maximum_iterations: Array
    successful: Array


class ReactiveSpeciesLatticeBoltzmannStepResult(StrictModule):
    candidate_state: ReactiveSpeciesLatticeBoltzmannState
    accepted_state: ReactiveSpeciesLatticeBoltzmannState
    diagnostics: ReactiveSpeciesLatticeBoltzmannDiagnostics
    successful: Array


class ReactiveSpeciesCouplingSchedulePlan(StrictModule, NonTrainableState):
    """Fixed-schedule Strang reaction around one coupled thermal/species transport step."""

    reaction_substeps: int = eqx.field(static=True)
    element_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        reaction_substeps: int = 1,
        element_tolerance: float = 1.0e-10,
        energy_tolerance: float = 1.0e-10,
    ):
        substeps = int(reaction_substeps)
        etol = float(element_tolerance)
        htol = float(energy_tolerance)
        if substeps <= 0 or etol <= 0.0 or htol <= 0.0:
            raise ValueError(
                "Reactive schedule substeps and tolerances must be positive."
            )
        self.reaction_substeps = substeps
        self.element_tolerance = etol
        self.energy_tolerance = htol
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-species-lattice-boltzmann-schedule",
                "reaction_substeps": substeps,
                "element_tolerance": etol,
                "energy_tolerance": htol,
            }
        )

    def _reaction_half(
        self,
        species: Array,
        energy: Array,
        extent: Array,
        step_size: Array,
        stepper: ReactiveLocalStepper,
        args: Any,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        dt = 0.5 * jnp.asarray(step_size) / float(self.reaction_substeps)

        def body(_, carry):
            (
                current_species,
                current_energy,
                current_extent,
                success,
                max_element,
                max_energy,
                max_iterations,
            ) = carry
            result = stepper.step(current_species, current_energy, dt, args)
            local_success = (
                jnp.all(result.successful)
                & jnp.all(jnp.isfinite(result.species_amount))
                & jnp.all(jnp.isfinite(result.sensible_energy))
                & jnp.all(result.species_amount >= 0.0)
            )
            accepted_species = jnp.where(
                local_success, result.species_amount, current_species
            )
            accepted_energy = jnp.where(
                local_success, result.sensible_energy, current_energy
            )
            accepted_extent = jnp.where(
                local_success, current_extent + result.extent_increment, current_extent
            )
            return (
                accepted_species,
                accepted_energy,
                accepted_extent,
                success & local_success,
                jnp.maximum(max_element, jnp.max(jnp.abs(result.element_residual))),
                jnp.maximum(max_energy, jnp.max(jnp.abs(result.energy_residual))),
                jnp.maximum(max_iterations, jnp.max(result.iterations)),
            )

        zero = jnp.zeros((), dtype=energy.dtype)
        return jax.lax.fori_loop(
            0,
            self.reaction_substeps,
            body,
            (
                species,
                energy,
                extent,
                jnp.asarray(True),
                zero,
                zero,
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    def advance(
        self,
        state: ReactiveSpeciesLatticeBoltzmannState,
        step_size: Array,
        reaction_stepper: ReactiveLocalStepper,
        transport_step: Callable[
            [ThermalLatticeBoltzmannState, SpeciesLatticeBoltzmannState, Any],
            tuple[ThermalLatticeBoltzmannState, SpeciesLatticeBoltzmannState, Array],
        ],
        /,
        *,
        args: Any = None,
        temperature_margin: Array | float = jnp.inf,
    ) -> ReactiveSpeciesLatticeBoltzmannStepResult:
        if not isinstance(state, ReactiveSpeciesLatticeBoltzmannState):
            raise TypeError("state must be ReactiveSpeciesLatticeBoltzmannState.")
        species_amount = jnp.sum(state.species.populations, axis=-1)
        sensible_energy = jnp.sum(state.thermal.populations, axis=-1)
        first = self._reaction_half(
            species_amount,
            sensible_energy,
            state.reaction_extent,
            step_size,
            reaction_stepper,
            args,
        )
        (
            first_species,
            first_energy,
            first_extent,
            first_success,
            first_element,
            first_energy_residual,
            first_iterations,
        ) = first
        species_delta = first_species - species_amount
        energy_delta = first_energy - sensible_energy
        species_populations = (
            state.species.populations
            + species_delta[..., :, None] / state.species.populations.shape[-1]
        )
        thermal_populations = (
            state.thermal.populations
            + energy_delta[..., None] / state.thermal.populations.shape[-1]
        )
        thermal_input = eqx.tree_at(
            lambda value: value.populations, state.thermal, thermal_populations
        )
        species_input = eqx.tree_at(
            lambda value: value.populations, state.species, species_populations
        )
        transported_thermal, transported_species, transport_success = transport_step(
            thermal_input, species_input, args
        )
        transported_species_amount = jnp.sum(transported_species.populations, axis=-1)
        transported_energy = jnp.sum(transported_thermal.populations, axis=-1)
        second = self._reaction_half(
            transported_species_amount,
            transported_energy,
            first_extent,
            step_size,
            reaction_stepper,
            args,
        )
        (
            second_species,
            second_energy,
            second_extent,
            second_success,
            second_element,
            second_energy_residual,
            second_iterations,
        ) = second
        final_species_delta = second_species - transported_species_amount
        final_energy_delta = second_energy - transported_energy
        candidate_species = eqx.tree_at(
            lambda value: value.populations,
            transported_species,
            transported_species.populations
            + final_species_delta[..., :, None]
            / transported_species.populations.shape[-1],
        )
        candidate_thermal = eqx.tree_at(
            lambda value: value.populations,
            transported_thermal,
            transported_thermal.populations
            + final_energy_delta[..., None] / transported_thermal.populations.shape[-1],
        )
        maximum_element = jnp.maximum(first_element, second_element)
        maximum_energy = jnp.maximum(first_energy_residual, second_energy_residual)
        margin = jnp.asarray(temperature_margin, dtype=transported_energy.dtype)
        successful = (
            state.successful
            & first_success
            & second_success
            & jnp.all(transport_success)
            & (maximum_element <= self.element_tolerance)
            & (maximum_energy <= self.energy_tolerance)
            & jnp.all(second_species >= 0.0)
            & jnp.all(jnp.isfinite(second_energy))
            & jnp.all(margin > 0.0)
        )
        candidate = ReactiveSpeciesLatticeBoltzmannState(
            candidate_thermal,
            candidate_species,
            second_extent,
            state.element_inventory,
            successful,
        )
        accepted = jax.tree.map(
            lambda proposed, old: jnp.where(successful, proposed, old), candidate, state
        )
        diagnostics = ReactiveSpeciesLatticeBoltzmannDiagnostics(
            maximum_element,
            maximum_energy,
            jnp.min(second_species),
            jnp.min(margin),
            jnp.maximum(first_iterations, second_iterations),
            successful,
        )
        return ReactiveSpeciesLatticeBoltzmannStepResult(
            candidate, accepted, diagnostics, successful
        )


__all__ = [
    "ReactiveLocalStepResult",
    "ReactiveLocalStepper",
    "ReactiveSpeciesCouplingSchedulePlan",
    "ReactiveSpeciesLatticeBoltzmannDiagnostics",
    "ReactiveSpeciesLatticeBoltzmannState",
    "ReactiveSpeciesLatticeBoltzmannStepResult",
]
