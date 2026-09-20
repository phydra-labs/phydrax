#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative moving-source DED process–thermal–material workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...ein import contract
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ...manufacturing import ManufacturingRuntime, ManufacturingRuntimeState
from ...materials import MaterialState, SpatialICMEModel, SpatialMaterialField


@dataclass(frozen=True, slots=True)
class SpatialDEDState:
    runtime: ManufacturingRuntimeState
    temperature_k: Array
    phase_fractions: Array


@dataclass(frozen=True, slots=True)
class SpatialDEDStep:
    state: SpatialDEDState
    effective_material_properties: Array
    constrained_thermal_stress_pa: Array
    thermal_residual_norm: Array
    energy_balance_residual_j: Array
    mass_balance_residual_kg: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class SpatialDEDWorkflow:
    runtime: ManufacturingRuntime
    heat_capacity_j_k: Array
    conductance_laplacian_w_k: Array
    convection_conductance_w_k: Array
    ambient_temperature_k: float
    thermal_expansion_k_inv: float
    elastic_modulus_pa: float
    reference_temperature_k: float
    icme_model: SpatialICMEModel

    @classmethod
    def create(
        cls,
        runtime: ManufacturingRuntime,
        heat_capacity_j_k: ArrayLike,
        conductance_laplacian_w_k: ArrayLike,
        convection_conductance_w_k: ArrayLike,
        icme_model: SpatialICMEModel,
        /,
        *,
        ambient_temperature_k: float,
        thermal_expansion_k_inv: float,
        elastic_modulus_pa: float,
        reference_temperature_k: float,
        tolerance: float = 1e-10,
    ) -> SpatialDEDWorkflow:
        capacity = np.asarray(heat_capacity_j_k, dtype=np.float64)
        conductance = np.asarray(conductance_laplacian_w_k, dtype=np.float64)
        convection = np.asarray(convection_conductance_w_k, dtype=np.float64)
        cells = runtime.control_volumes_m3.size
        if capacity.shape != (cells,) or np.any(capacity <= 0):
            raise ValueError("DED heat capacities must be positive and cell aligned.")
        if conductance.shape != (cells, cells):
            raise ValueError("DED conductance Laplacian has incompatible shape.")
        if not np.allclose(conductance, conductance.T, atol=tolerance, rtol=0):
            raise ValueError("DED conductance Laplacian must be symmetric.")
        if not np.allclose(conductance.sum(axis=0), 0, atol=tolerance, rtol=0):
            raise ValueError("DED conductance Laplacian must conserve internal heat.")
        if convection.shape != (cells,) or np.any(convection < 0):
            raise ValueError(
                "DED convection conductance must be non-negative and aligned."
            )
        if ambient_temperature_k <= 0 or elastic_modulus_pa <= 0:
            raise ValueError(
                "DED ambient temperature and elastic modulus must be positive."
            )
        return cls(
            runtime,
            jnp.asarray(capacity),
            jnp.asarray(conductance),
            jnp.asarray(convection),
            float(ambient_temperature_k),
            float(thermal_expansion_k_inv),
            float(elastic_modulus_pa),
            float(reference_temperature_k),
            icme_model,
        )

    def advance(
        self,
        state: SpatialDEDState,
        end_time_s: float,
        equilibrium_phase_fractions: ArrayLike,
        /,
    ) -> SpatialDEDStep:
        temperature = jnp.asarray(state.temperature_k)
        if temperature.shape != self.heat_capacity_j_k.shape:
            raise ValueError("DED temperature does not match workflow cells.")
        runtime_step = self.runtime.advance(state.runtime, end_time_s)
        dt = float(end_time_s - state.runtime.time_s)
        heat_increment = (
            runtime_step.state.supplied_energy_j - state.runtime.supplied_energy_j
        )
        matrix = (
            jnp.diag(self.heat_capacity_j_k / dt + self.convection_conductance_w_k)
            + self.conductance_laplacian_w_k
        )
        right = (
            self.heat_capacity_j_k / dt * temperature
            + heat_increment / dt
            + self.convection_conductance_w_k * self.ambient_temperature_k
        )
        thermal = solve(
            LinearSystem(DenseLinearOperator(matrix)),
            right,
            policy=LinearSolvePolicy(DenseLU()),
        )
        thermal_residual = matrix @ thermal.value - right
        thermal_residual_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(thermal_residual), thermal_residual))
        )
        stored = contract("q,q->", self.heat_capacity_j_k, thermal.value - temperature)
        convection_loss = dt * contract(
            "q,q->",
            self.convection_conductance_w_k,
            thermal.value - self.ambient_temperature_k,
        )
        energy_balance = stored + convection_loss - jnp.sum(heat_increment)

        material_state = MaterialState(
            temperature,
            jnp.full_like(temperature, 101325.0),
            state.phase_fractions,
        )
        material_field = SpatialMaterialField.create(
            self.runtime.coordinates_m,
            self.runtime.control_volumes_m3,
            material_state,
        )
        icme = self.icme_model.advance(
            material_field,
            thermal.value,
            equilibrium_phase_fractions,
            dt,
        )
        stress = (
            -self.elastic_modulus_pa
            * self.thermal_expansion_k_inv
            * (thermal.value - self.reference_temperature_k)
        )
        next_state = SpatialDEDState(
            runtime_step.state,
            thermal.value,
            icme.field.state.phase_fractions,
        )
        successful = (
            thermal.successful
            & jnp.all(jnp.isfinite(thermal.value))
            & jnp.all(thermal.value > 0)
            & icme.field.state.admissible
        )
        return SpatialDEDStep(
            next_state,
            icme.effective_properties,
            stress,
            thermal_residual_norm,
            energy_balance,
            runtime_step.mass_balance_residual_kg,
            successful,
        )


__all__ = ["SpatialDEDState", "SpatialDEDStep", "SpatialDEDWorkflow"]
