#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..equations._chemical_species import ChemicalPhaseKind
from ..equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ..equations._homogeneous_thermodynamics import (
    HomogeneousHelmholtzPlan,
    ZeroResidualHelmholtzTerm,
)
from ..optim import (
    Bounds,
    MinimizationProblem,
    minimize,
    NonlinearConstraint,
    OptimizationStatus,
    OptimizationTermination,
    PrimalDualInteriorPoint,
)


class ChemicalEquilibriumEnsemble(StrEnum):
    TP = "tp"
    TV = "tv"
    HP = "hp"
    UV = "uv"
    SP = "sp"
    SV = "sv"


class ChemicalEquilibriumEvidence(StrictModule):
    element_residual: Array
    charge_residual: Array
    conserved_property_residual: Array
    stationarity_norm: Array
    objective_change: Array
    active_species: Array
    active_phases: Array
    derivative_valid: Array
    successful: Array
    equilibrium_id: str = eqx.field(static=True)


class ChemicalEquilibriumResult(StrictModule):
    species_amount: Array
    mole_fraction: Array
    phase_fraction: Array
    temperature: Array
    pressure: Array
    volume: Array
    chemical_potential: Array
    enthalpy: Array
    internal_energy: Array
    entropy: Array
    objective: Array
    solver_status: Array
    evidence: ChemicalEquilibriumEvidence
    equilibrium_id: str = eqx.field(static=True)


class ChemicalEquilibriumThermodynamicState(StrictModule):
    mole_fraction: Array
    phase_fraction: Array
    chemical_potential: Array
    enthalpy: Array
    internal_energy: Array
    entropy: Array
    gibbs: Array
    helmholtz: Array
    volume: Array
    pressure: Array
    successful: Array


class ChemicalEquilibriumPlan(StrictModule):
    """Ideal-phase constrained chemical equilibrium in one declared ensemble.

    Gas phases use ideal partial-pressure activities. Liquid and solid phases use
    ideal within-phase activities and negligible volume. Surface phases are outside
    this profile because their conserved site measures require a separate reactor.
    """

    thermodynamics: HomogeneousHelmholtzPlan
    ensemble: ChemicalEquilibriumEnsemble = eqx.field(static=True)
    balance_matrix: Array
    balance_nullspace: Array
    gas_species: Array
    species_phase_indices: Array
    phase_standard_pressure: Array
    active_balance_rows: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    equilibrium_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        ensemble: ChemicalEquilibriumEnsemble | str,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_steps: int = 200,
    ):
        if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
            raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
        if not isinstance(thermodynamics.residual, ZeroResidualHelmholtzTerm):
            raise TypeError("Chemical equilibrium currently requires ideal phases.")
        try:
            ensemble_ = ChemicalEquilibriumEnsemble(ensemble)
        except ValueError as failure:
            raise ValueError("Unknown chemical-equilibrium ensemble.") from failure
        tolerance_ = float(tolerance)
        steps = int(maximum_steps)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0 or steps <= 0:
            raise ValueError("Equilibrium tolerance and maximum_steps are invalid.")
        schema = thermodynamics.schema
        if any(kind is ChemicalPhaseKind.SURFACE for kind in schema.phases):
            raise ValueError(
                "Surface species require a site-balanced surface-equilibrium profile."
            )
        gas = np.asarray(
            tuple(kind is ChemicalPhaseKind.GAS for kind in schema.phases), dtype=bool
        )
        if not np.any(gas):
            raise ValueError("Chemical equilibrium requires at least one gas species.")
        full_balance = np.concatenate(
            (
                np.asarray(schema.element_composition, dtype=float),
                np.asarray(schema.charges, dtype=float)[None, :],
            ),
            axis=0,
        )
        active_rows = _independent_rows(full_balance, tolerance_)
        balance = full_balance[np.asarray(active_rows, dtype=np.int32)]
        _, _, right_vectors = np.linalg.svd(balance, full_matrices=True)
        nullspace = right_vectors[len(active_rows) :].T
        phase_pressure = np.ones((schema.phase_count,), dtype=float)
        for index, phase in enumerate(schema.phase_specs):
            if phase.kind is ChemicalPhaseKind.GAS:
                phase_pressure[index] = float(phase.standard_pressure)
        self.thermodynamics = thermodynamics
        self.ensemble = ensemble_
        self.balance_matrix = jnp.asarray(balance)
        self.balance_nullspace = jnp.asarray(nullspace)
        self.gas_species = jnp.asarray(gas)
        self.species_phase_indices = schema.phase_ids
        self.phase_standard_pressure = jnp.asarray(phase_pressure)
        self.active_balance_rows = active_rows
        self.tolerance = tolerance_
        self.maximum_steps = steps
        self.equilibrium_id = canonical_fingerprint(
            {
                "kind": "ideal-phase-chemical-equilibrium",
                "thermodynamics": thermodynamics.model_id,
                "ensemble": ensemble_.value,
                "active_rows": list(active_rows),
                "balance": array_tree_fingerprint(balance),
                "balance_nullspace": array_tree_fingerprint(nullspace),
                "gas_species": gas.tolist(),
                "phase_standard_pressure": phase_pressure.tolist(),
                "tolerance": tolerance_,
                "maximum_steps": steps,
            }
        )

    def evaluate_state(self, amount: Array, temperature: Array, pressure: Array, /):
        tiny = jnp.finfo(amount.dtype).tiny
        phase_amount = jnp.stack(
            tuple(
                jnp.sum(jnp.where(self.species_phase_indices == phase, amount, 0.0))
                for phase in range(self.thermodynamics.schema.phase_count)
            )
        )
        total_amount = jnp.sum(amount)
        mole_fraction = amount / jnp.maximum(total_amount, tiny)
        species_phase_amount = phase_amount[self.species_phase_indices]
        phase_fraction = amount / jnp.maximum(species_phase_amount, tiny)
        gas_total = jnp.sum(jnp.where(self.gas_species, amount, 0.0))
        gas_fraction = amount / jnp.maximum(gas_total, tiny)
        species_thermo = self.thermodynamics.thermodynamics.evaluate(temperature)
        standard_gibbs = species_thermo.molar_gibbs_energy
        standard_entropy = species_thermo.molar_entropy
        standard_enthalpy = species_thermo.molar_enthalpy
        standard_internal = species_thermo.molar_internal_energy
        standard_pressure = self.phase_standard_pressure[self.species_phase_indices]
        activity = jnp.where(
            self.gas_species,
            gas_fraction * pressure / standard_pressure,
            phase_fraction,
        )
        log_activity = jnp.log(jnp.maximum(activity, tiny))
        chemical = standard_gibbs + UNIVERSAL_GAS_CONSTANT * temperature * log_activity
        enthalpy = jnp.sum(amount * standard_enthalpy)
        internal = jnp.sum(amount * standard_internal)
        entropy = jnp.sum(
            amount * (standard_entropy - UNIVERSAL_GAS_CONSTANT * log_activity)
        )
        gibbs = jnp.sum(amount * chemical)
        helmholtz = internal - temperature * entropy
        volume = gas_total * UNIVERSAL_GAS_CONSTANT * temperature / pressure
        successful = (
            species_thermo.successful
            & jnp.all(jnp.isfinite(chemical))
            & jnp.isfinite(enthalpy)
            & jnp.isfinite(internal)
            & jnp.isfinite(entropy)
            & jnp.isfinite(volume)
            & (temperature > 0.0)
            & (pressure > 0.0)
            & (gas_total > 0.0)
        )
        return ChemicalEquilibriumThermodynamicState(
            mole_fraction,
            phase_fraction,
            chemical,
            enthalpy,
            internal,
            entropy,
            gibbs,
            helmholtz,
            volume,
            pressure,
            successful,
        )

    def solve(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        initial_species_amount: ArrayLike,
        /,
    ) -> ChemicalEquilibriumResult:
        temperature_ = jnp.asarray(temperature)
        pressure_ = jnp.asarray(pressure, dtype=temperature_.dtype)
        initial = jnp.asarray(initial_species_amount, dtype=temperature_.dtype)
        count = self.thermodynamics.schema.species_count
        if temperature_.shape != () or pressure_.shape != ():
            raise ValueError("Chemical equilibrium solves one thermodynamic state.")
        if initial.shape != (count,):
            raise ValueError("initial_species_amount must match species_count.")
        if not jnp.issubdtype(initial.dtype, jnp.inexact):
            raise TypeError("initial_species_amount must have inexact dtype.")
        amount_scale = jnp.maximum(jnp.sum(initial), 1.0)
        scaled_initial = initial / amount_scale
        initial_state = self.evaluate_state(initial, temperature_, pressure_)
        fixed_volume = initial_state.volume
        ensemble = self.ensemble
        variable_temperature = ensemble not in (
            ChemicalEquilibriumEnsemble.TP,
            ChemicalEquilibriumEnsemble.TV,
        )
        reaction_dimension = self.balance_nullspace.shape[1]
        variables = jnp.zeros(
            (reaction_dimension + int(variable_temperature),), dtype=initial.dtype
        )
        minimum_temperature = self.thermodynamics.thermodynamics.minimum_temperature
        maximum_temperature = self.thermodynamics.thermodynamics.maximum_temperature
        objective_scale = amount_scale * jnp.where(
            ensemble
            in (
                ChemicalEquilibriumEnsemble.HP,
                ChemicalEquilibriumEnsemble.UV,
            ),
            UNIVERSAL_GAS_CONSTANT,
            UNIVERSAL_GAS_CONSTANT * temperature_,
        )

        def decode(value):
            reaction_coordinate = value[:reaction_dimension]
            scaled_amount = scaled_initial + self.balance_nullspace @ reaction_coordinate
            amount = amount_scale * scaled_amount
            solved_temperature = (
                temperature_ * jnp.exp(value[-1])
                if variable_temperature
                else temperature_
            )
            if ensemble in (
                ChemicalEquilibriumEnsemble.TV,
                ChemicalEquilibriumEnsemble.UV,
                ChemicalEquilibriumEnsemble.SV,
            ):
                gas_total = jnp.sum(jnp.where(self.gas_species, amount, 0.0))
                solved_pressure = (
                    gas_total * UNIVERSAL_GAS_CONSTANT * solved_temperature / fixed_volume
                )
            else:
                solved_pressure = pressure_
            state = self.evaluate_state(amount, solved_temperature, solved_pressure)
            return amount, solved_temperature, solved_pressure, state

        def physical_objective(value):
            _, _, _, state = decode(value)
            if ensemble is ChemicalEquilibriumEnsemble.TP:
                return state.gibbs
            if ensemble is ChemicalEquilibriumEnsemble.TV:
                return state.helmholtz
            if ensemble in (
                ChemicalEquilibriumEnsemble.HP,
                ChemicalEquilibriumEnsemble.UV,
            ):
                return -state.entropy
            if ensemble is ChemicalEquilibriumEnsemble.SP:
                return state.enthalpy
            return state.internal_energy

        def objective(value, _):
            return physical_objective(value) / objective_scale

        def scaled_amount_constraint(value, _):
            return scaled_initial + self.balance_nullspace @ value[:reaction_dimension]

        constraints = [
            NonlinearConstraint(
                scaled_amount_constraint,
                lower=jnp.zeros_like(scaled_initial),
                upper=jnp.full_like(scaled_initial, jnp.inf),
            )
        ]
        conserved_name = "none"
        conserved_target = jnp.asarray(0.0, dtype=initial.dtype)
        if ensemble is ChemicalEquilibriumEnsemble.HP:
            conserved_name, conserved_target = "enthalpy", initial_state.enthalpy
        elif ensemble is ChemicalEquilibriumEnsemble.UV:
            conserved_name, conserved_target = "internal", initial_state.internal_energy
        elif ensemble in (
            ChemicalEquilibriumEnsemble.SP,
            ChemicalEquilibriumEnsemble.SV,
        ):
            conserved_name, conserved_target = "entropy", initial_state.entropy
        property_scale = jnp.maximum(jnp.abs(conserved_target), 1.0)
        if conserved_name != "none":

            def conserved(value, _):
                _, _, _, state = decode(value)
                selected = (
                    state.enthalpy
                    if conserved_name == "enthalpy"
                    else state.internal_energy
                    if conserved_name == "internal"
                    else state.entropy
                )
                return jnp.asarray((selected / property_scale,))

            target = jnp.asarray((conserved_target / property_scale,))
            constraints.append(NonlinearConstraint(conserved, lower=target, upper=target))
        lower = jnp.full_like(variables, -jnp.inf)
        upper = jnp.full_like(variables, jnp.inf)
        if variable_temperature:
            lower = lower.at[-1].set(jnp.log(minimum_temperature / temperature_))
            upper = upper.at[-1].set(jnp.log(maximum_temperature / temperature_))
        if variables.size == 0:
            solved_parameters = variables
            solver_status = jnp.asarray(int(OptimizationStatus.SUCCESS), dtype=jnp.int32)
            solver_successful = jnp.asarray(True)
            stationarity_norm = jnp.asarray(0.0, dtype=initial.dtype)
        else:
            solved = minimize(
                MinimizationProblem(
                    objective,
                    bounds=Bounds(lower, upper),
                    constraints=tuple(constraints),
                ),
                variables,
                method=PrimalDualInteriorPoint(
                    mode="dense-filter",
                    max_dense_dimension=max(
                        32, variables.size + len(self.active_balance_rows)
                    ),
                ),
                termination=OptimizationTermination(
                    absolute_optimality=self.tolerance,
                    relative_optimality=0.0,
                    maximum_steps=self.maximum_steps,
                ),
            )
            solved_parameters = solved.parameters
            solver_status = solved.status
            solver_successful = solved.successful
            stationarity_norm = solved.diagnostics.final_optimality_norm
        amount, solved_temperature, solved_pressure, state = decode(solved_parameters)
        full_balance = jnp.concatenate(
            (
                self.thermodynamics.schema.element_composition.astype(amount.dtype),
                self.thermodynamics.schema.charges.astype(amount.dtype)[None, :],
            ),
            axis=0,
        )
        full_residual = contract("es,s->e", full_balance, amount - initial)
        element_count = self.thermodynamics.schema.element_count
        active_threshold = jnp.sqrt(jnp.finfo(amount.dtype).eps) * amount_scale
        active = amount > active_threshold
        active_phases = jnp.stack(
            tuple(
                jnp.any(active & (self.species_phase_indices == phase))
                for phase in range(self.thermodynamics.schema.phase_count)
            )
        )
        selected_property = (
            state.enthalpy
            if conserved_name == "enthalpy"
            else state.internal_energy
            if conserved_name == "internal"
            else state.entropy
            if conserved_name == "entropy"
            else jnp.asarray(0.0, dtype=amount.dtype)
        )
        property_residual = selected_property - conserved_target
        initial_objective = physical_objective(variables)
        final_objective = physical_objective(solved_parameters)
        balance_scale = jnp.maximum(
            jnp.abs(contract("es,s->e", full_balance, initial)), 1.0
        )
        successful = (
            solver_successful
            & state.successful
            & jnp.all(amount >= 0.0)
            & jnp.all(jnp.abs(full_residual) <= self.tolerance * balance_scale)
            & (
                (conserved_name == "none")
                | (jnp.abs(property_residual) <= self.tolerance * property_scale)
            )
            & (
                final_objective
                <= initial_objective
                + self.tolerance * jnp.maximum(jnp.abs(initial_objective), 1.0)
            )
        )
        derivative_valid = successful & jnp.all(active)
        evidence = ChemicalEquilibriumEvidence(
            full_residual[:element_count],
            full_residual[element_count],
            property_residual,
            stationarity_norm,
            final_objective - initial_objective,
            active,
            active_phases,
            derivative_valid,
            successful,
            self.equilibrium_id,
        )
        return ChemicalEquilibriumResult(
            amount,
            state.mole_fraction,
            state.phase_fraction,
            solved_temperature,
            solved_pressure,
            state.volume,
            state.chemical_potential,
            state.enthalpy,
            state.internal_energy,
            state.entropy,
            final_objective,
            solver_status,
            evidence,
            self.equilibrium_id,
        )


def _independent_rows(matrix: np.ndarray, tolerance: float) -> tuple[int, ...]:
    selected: list[int] = []
    rank = 0
    for index in range(matrix.shape[0]):
        candidate = matrix[np.asarray((*selected, index), dtype=np.int32)]
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=tolerance))
        if candidate_rank > rank:
            selected.append(index)
            rank = candidate_rank
    if not selected:
        raise ValueError("Equilibrium requires at least one independent balance row.")
    return tuple(selected)


__all__ = [
    "ChemicalEquilibriumEnsemble",
    "ChemicalEquilibriumEvidence",
    "ChemicalEquilibriumPlan",
    "ChemicalEquilibriumResult",
    "ChemicalEquilibriumThermodynamicState",
]
