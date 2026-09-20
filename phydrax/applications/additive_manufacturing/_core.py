#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded thermal-mechanical additive manufacturing workflows."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...manufacturing import GaussianMovingSource, MaterialActivationState
from ...qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class DEDState:
    temperature_k: Array
    activation: MaterialActivationState
    time_s: Array
    deposited_mass_kg: Array
    supplied_energy_j: Array


@dataclass(frozen=True, slots=True)
class DEDStepResult:
    candidate: DEDState
    accepted: DEDState
    accepted_step: Array
    energy_defect_j: Array
    maximum_temperature_k: Array


@dataclass(frozen=True, slots=True)
class DEDProcessPlan:
    coordinates_m: Array
    volumes_m3: Array
    conductivity_graph_w_k: Array
    density_kg_m3: float
    heat_capacity_j_kg_k: float
    convection_w_m2_k: float
    exposed_area_m2: Array
    ambient_temperature_k: float
    thermal_expansion_k_inv: float
    elastic_modulus_pa: float

    @classmethod
    def create(
        cls,
        coordinates_m: ArrayLike,
        volumes_m3: ArrayLike,
        conductivity_graph_w_k: ArrayLike,
        /,
        *,
        density_kg_m3: float,
        heat_capacity_j_kg_k: float,
        convection_w_m2_k: float,
        exposed_area_m2: ArrayLike,
        ambient_temperature_k: float,
        thermal_expansion_k_inv: float,
        elastic_modulus_pa: float,
    ) -> DEDProcessPlan:
        return cls(
            jnp.asarray(coordinates_m),
            jnp.asarray(volumes_m3),
            jnp.asarray(conductivity_graph_w_k),
            float(density_kg_m3),
            float(heat_capacity_j_kg_k),
            float(convection_w_m2_k),
            jnp.asarray(exposed_area_m2),
            float(ambient_temperature_k),
            float(thermal_expansion_k_inv),
            float(elastic_modulus_pa),
        )

    def __post_init__(self) -> None:
        count = self.coordinates_m.shape[0]
        if (
            self.coordinates_m.ndim != 2
            or self.volumes_m3.shape != (count,)
            or self.conductivity_graph_w_k.shape != (count, count)
            or self.exposed_area_m2.shape != (count,)
        ):
            raise ValueError("DED geometry arrays do not align.")
        if (
            min(
                self.density_kg_m3,
                self.heat_capacity_j_kg_k,
                self.ambient_temperature_k,
                self.elastic_modulus_pa,
            )
            <= 0.0
        ):
            raise ValueError("DED material properties must be positive.")

    def step(
        self,
        state: DEDState,
        source: GaussianMovingSource,
        center_m: ArrayLike,
        step_size_s: ArrayLike,
        /,
        *,
        activation_selection: ArrayLike | None = None,
        added_mass_kg_s: float = 0.0,
    ) -> DEDStepResult:
        step = jnp.asarray(step_size_s)
        activation = (
            state.activation
            if activation_selection is None
            else state.activation.activate(activation_selection, state.time_s)
        )
        active = activation.active.astype(state.temperature_k.dtype)
        source_density = source.evaluate(self.coordinates_m, center_m) * active
        source_power = source_density * self.volumes_m3
        conduction = (
            -(
                self.conductivity_graph_w_k @ state.temperature_k
                - jnp.sum(self.conductivity_graph_w_k, axis=1) * state.temperature_k
            )
            * active
        )
        convection = (
            -self.convection_w_m2_k
            * self.exposed_area_m2
            * (state.temperature_k - self.ambient_temperature_k)
            * active
        )
        capacity = self.density_kg_m3 * self.heat_capacity_j_kg_k * self.volumes_m3
        rate = (source_power + conduction + convection) / jnp.maximum(capacity, 1.0e-30)
        candidate_temperature = state.temperature_k + step * rate
        supplied = jnp.sum(source_power) * step
        loss = -jnp.sum(convection) * step
        stored = jnp.sum(capacity * (candidate_temperature - state.temperature_k))
        defect = stored - supplied + loss - jnp.sum(conduction) * step
        candidate = DEDState(
            candidate_temperature,
            activation,
            state.time_s + step,
            state.deposited_mass_kg + float(added_mass_kg_s) * step,
            state.supplied_energy_j + supplied,
        )
        accepted = (
            jnp.all(jnp.isfinite(candidate_temperature))
            & (step > 0.0)
            & jnp.all(candidate_temperature > 0.0)
        )
        committed_activation = MaterialActivationState(
            jnp.where(accepted, candidate.activation.active, state.activation.active),
            jnp.where(
                accepted,
                candidate.activation.activation_time_s,
                state.activation.activation_time_s,
            ),
        )
        committed = DEDState(
            jnp.where(accepted, candidate.temperature_k, state.temperature_k),
            committed_activation,
            jnp.where(accepted, candidate.time_s, state.time_s),
            jnp.where(accepted, candidate.deposited_mass_kg, state.deposited_mass_kg),
            jnp.where(accepted, candidate.supplied_energy_j, state.supplied_energy_j),
        )
        return DEDStepResult(
            candidate, committed, accepted, defect, jnp.max(candidate_temperature)
        )

    def constrained_thermal_stress_pa(
        self, temperature_k: ArrayLike, reference_temperature_k: float, /
    ) -> Array:
        return (
            -self.elastic_modulus_pa
            * self.thermal_expansion_k_inv
            * (jnp.asarray(temperature_k) - float(reference_temperature_k))
        )


def additive_manufacturing_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        (
            "additive-manufacturing.ded-macro-thermal",
            "moving-source-activation-graph-conduction",
        ),
        ("additive-manufacturing.ded-thermal-mechanical", "constrained-thermal-stress"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("energy-ledger", "mass-ledger", "path-replay", "refinement"),
        )
        for name, formulation in specs
    )


__all__ = [
    "DEDProcessPlan",
    "DEDState",
    "DEDStepResult",
    "additive_manufacturing_candidate_profiles",
]
