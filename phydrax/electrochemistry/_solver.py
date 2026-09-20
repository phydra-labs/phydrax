#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Spatial porous-electrode current and species transport workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ._core import FARADAY_C_MOL, GAS_CONSTANT_J_MOL_K


@dataclass(frozen=True, slots=True)
class PorousElectrodeState:
    concentration_mol_m3: Array
    potential_v: Array


@dataclass(frozen=True, slots=True)
class PorousElectrodeStep:
    state: PorousElectrodeState
    faradaic_current_a: Array
    reaction_rate_mol_s: Array
    species_balance_residual_mol: Array
    current_residual_norm_a: Array
    nonlinear_iterations: int
    successful: Array


@dataclass(frozen=True, slots=True)
class PorousElectrodeSystem:
    cell_volumes_m3: Array
    species_transport_generators_s_inv: Array
    potential_operator_s: Array
    reaction_stoichiometry: Array
    reaction_area_m2: Array
    exchange_current_density_a_m2: Array
    equilibrium_potential_v: Array
    temperature_k: Array
    electron_count: int
    transfer_coefficient: float

    @classmethod
    def create(
        cls,
        cell_volumes_m3: ArrayLike,
        species_transport_generators_s_inv: ArrayLike,
        potential_operator_s: ArrayLike,
        reaction_stoichiometry: ArrayLike,
        reaction_area_m2: ArrayLike,
        exchange_current_density_a_m2: ArrayLike,
        equilibrium_potential_v: ArrayLike,
        temperature_k: ArrayLike,
        electron_count: int,
        /,
        *,
        transfer_coefficient: float = 0.5,
        tolerance: float = 1e-10,
    ) -> PorousElectrodeSystem:
        volumes = np.asarray(cell_volumes_m3, dtype=float)
        transport = np.asarray(species_transport_generators_s_inv, dtype=float)
        potential = np.asarray(potential_operator_s, dtype=float)
        stoichiometry = np.asarray(reaction_stoichiometry, dtype=float)
        area = np.asarray(reaction_area_m2, dtype=float)
        exchange = np.asarray(exchange_current_density_a_m2, dtype=float)
        equilibrium = np.asarray(equilibrium_potential_v, dtype=float)
        temperature = np.asarray(temperature_k, dtype=float)
        cells = volumes.size
        if volumes.ndim != 1 or cells == 0 or np.any(volumes <= 0):
            raise ValueError("Porous-electrode volumes must be a positive vector.")
        if transport.ndim != 3 or transport.shape[1:] != (cells, cells):
            raise ValueError("Species transport generators have incompatible shape.")
        for generator in transport:
            if not np.allclose(volumes @ generator, 0, atol=tolerance, rtol=tolerance):
                raise ValueError(
                    "Every species transport generator must conserve inventory."
                )
            if np.any(generator - np.diag(np.diag(generator)) < -tolerance):
                raise ValueError(
                    "Species transport generators must be positivity preserving."
                )
        if potential.shape != (cells, cells) or not np.allclose(
            potential, potential.T, atol=tolerance, rtol=0
        ):
            raise ValueError(
                "Porous-electrode potential operator must be symmetric and square."
            )
        if stoichiometry.shape != (transport.shape[0],):
            raise ValueError(
                "Reaction stoichiometry must align with transported species."
            )
        if any(
            value.shape != (cells,)
            for value in (area, exchange, equilibrium, temperature)
        ):
            raise ValueError("Porous-electrode reaction fields must be cell aligned.")
        if np.any(area < 0) or np.any(exchange < 0) or np.any(temperature <= 0):
            raise ValueError("Porous-electrode reaction fields are inadmissible.")
        if electron_count == 0 or not 0 < transfer_coefficient < 1:
            raise ValueError(
                "Electrochemical electron count or transfer coefficient is invalid."
            )
        return cls(
            jnp.asarray(volumes),
            jnp.asarray(transport),
            jnp.asarray(potential),
            jnp.asarray(stoichiometry),
            jnp.asarray(area),
            jnp.asarray(exchange),
            jnp.asarray(equilibrium),
            jnp.asarray(temperature),
            int(electron_count),
            float(transfer_coefficient),
        )

    def _faradaic(self, potential: Array) -> tuple[Array, Array]:
        thermal = (
            self.electron_count
            * FARADAY_C_MOL
            / (GAS_CONSTANT_J_MOL_K * self.temperature_k)
        )
        overpotential = potential - self.equilibrium_potential_v
        anodic = jnp.exp(self.transfer_coefficient * thermal * overpotential)
        cathodic = jnp.exp(-(1 - self.transfer_coefficient) * thermal * overpotential)
        current_density = self.exchange_current_density_a_m2 * (anodic - cathodic)
        derivative = (
            self.exchange_current_density_a_m2
            * thermal
            * (
                self.transfer_coefficient * anodic
                + (1 - self.transfer_coefficient) * cathodic
            )
        )
        return self.reaction_area_m2 * current_density, self.reaction_area_m2 * derivative

    def advance(
        self,
        state: PorousElectrodeState,
        applied_current_a: ArrayLike,
        step_size_s: float,
        /,
        *,
        nonlinear_iterations: int = 20,
        current_tolerance_a: float = 1e-10,
    ) -> PorousElectrodeStep:
        concentration = jnp.asarray(state.concentration_mol_m3)
        potential = jnp.asarray(state.potential_v)
        applied = jnp.asarray(applied_current_a)
        species = self.species_transport_generators_s_inv.shape[0]
        cells = self.cell_volumes_m3.size
        if concentration.shape != (cells, species):
            raise ValueError("Porous-electrode concentrations have incompatible shape.")
        if potential.shape != (cells,) or applied.shape != (cells,) or step_size_s <= 0:
            raise ValueError("Porous-electrode electrical state or step size is invalid.")
        if bool(jnp.any(concentration < 0)):
            raise ValueError("Porous-electrode concentrations must be non-negative.")
        completed = 0
        for iteration in range(nonlinear_iterations):
            faradaic, derivative = self._faradaic(potential)
            residual = self.potential_operator_s @ potential + faradaic - applied
            if bool(jnp.max(jnp.abs(residual)) <= current_tolerance_a):
                completed = iteration
                break
            jacobian = self.potential_operator_s + jnp.diag(derivative)
            correction = solve(
                LinearSystem(DenseLinearOperator(jacobian)),
                -residual,
                policy=LinearSolvePolicy(DenseLU()),
            )
            potential = potential + correction.value
            completed = iteration + 1
        faradaic, _ = self._faradaic(potential)
        current_residual = self.potential_operator_s @ potential + faradaic - applied
        current_norm = jnp.sqrt(
            jnp.real(contract("i,i->", jnp.conj(current_residual), current_residual))
        )
        reaction_rate = faradaic / (self.electron_count * FARADAY_C_MOL)
        source_amount_rate = reaction_rate[:, None] * self.reaction_stoichiometry[None, :]
        next_species = []
        balance = []
        for species_index in range(species):
            generator = self.species_transport_generators_s_inv[species_index]
            matrix = jnp.eye(cells) - float(step_size_s) * generator
            right = concentration[:, species_index] + float(step_size_s) * (
                source_amount_rate[:, species_index] / self.cell_volumes_m3
            )
            transported = solve(
                LinearSystem(DenseLinearOperator(matrix)),
                right,
                policy=LinearSolvePolicy(DenseLU()),
            )
            next_species.append(transported.value)
            balance.append(
                contract(
                    "q,q->",
                    self.cell_volumes_m3,
                    transported.value - concentration[:, species_index],
                )
                - float(step_size_s) * jnp.sum(source_amount_rate[:, species_index])
            )
        next_concentration = jnp.stack(next_species, axis=1)
        species_balance = jnp.stack(balance)
        successful = (
            jnp.all(next_concentration >= -1e-12)
            & jnp.all(jnp.isfinite(next_concentration))
            & (current_norm <= current_tolerance_a)
        )
        return PorousElectrodeStep(
            PorousElectrodeState(next_concentration, potential),
            faradaic,
            reaction_rate,
            species_balance,
            current_norm,
            completed,
            successful,
        )


__all__ = ["PorousElectrodeState", "PorousElectrodeStep", "PorousElectrodeSystem"]
