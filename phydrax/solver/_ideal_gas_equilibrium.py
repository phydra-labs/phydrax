#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
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
    OptimizationTermination,
    PrimalDualInteriorPoint,
)


class IdealGasEquilibriumEvidence(StrictModule):
    balance_residual: Array
    charge_residual: Array
    stationarity_norm: Array
    gibbs_change: Array
    active_mask: Array
    successful: Array
    equilibrium_id: str = eqx.field(static=True)


class IdealGasEquilibriumResult(StrictModule):
    species_amount: Array
    mole_fraction: Array
    volume: Array
    chemical_potential: Array
    gibbs_energy: Array
    solver_status: Array
    evidence: IdealGasEquilibriumEvidence
    equilibrium_id: str = eqx.field(static=True)


class IdealGasGibbsEquilibriumPlan(StrictModule):
    """Fixed-temperature, fixed-pressure ideal-gas Gibbs minimization."""

    thermodynamics: HomogeneousHelmholtzPlan
    balance_matrix: Array
    active_balance_rows: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    equilibrium_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_steps: int = 200,
    ) -> None:
        if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
            raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
        if not isinstance(thermodynamics.residual, ZeroResidualHelmholtzTerm):
            raise TypeError("Ideal gas equilibrium requires a zero residual model.")
        tolerance_value = float(tolerance)
        steps = int(maximum_steps)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if steps <= 0:
            raise ValueError("maximum_steps must be positive.")
        schema = thermodynamics.schema
        full_balance = np.concatenate(
            (
                np.asarray(schema.element_composition, dtype=float),
                np.asarray(schema.charges, dtype=float)[None, :],
            ),
            axis=0,
        )
        active_rows = _independent_rows(full_balance, tolerance_value)
        balance = full_balance[np.asarray(active_rows, dtype=np.int32)]
        generated = canonical_fingerprint(
            {
                "kind": "ideal-gas-gibbs-equilibrium",
                "thermodynamics": thermodynamics.model_id,
                "active_rows": list(active_rows),
                "balance": array_tree_fingerprint(balance),
                "tolerance": tolerance_value,
                "maximum_steps": steps,
            }
        )
        self.thermodynamics = thermodynamics
        self.balance_matrix = jnp.asarray(balance)
        self.active_balance_rows = active_rows
        self.tolerance = tolerance_value
        self.maximum_steps = steps
        self.equilibrium_id = generated

    def solve(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        initial_species_amount: ArrayLike,
        /,
    ) -> IdealGasEquilibriumResult:
        temperature_value = jnp.asarray(temperature)
        pressure_value = jnp.asarray(pressure)
        initial = jnp.asarray(initial_species_amount)
        count = self.thermodynamics.schema.species_count
        if temperature_value.shape != () or pressure_value.shape != ():
            raise ValueError(
                "Ideal gas equilibrium currently solves one state at a time."
            )
        if initial.shape != (count,):
            raise ValueError("initial_species_amount must have shape (species_count,).")
        if not jnp.issubdtype(initial.dtype, jnp.inexact):
            raise TypeError("initial_species_amount must have inexact dtype.")
        balance_target = contract("es,s->e", self.balance_matrix, initial)
        initial_total = jnp.sum(initial)
        scale = jnp.maximum(initial_total, jnp.asarray(1.0, dtype=initial.dtype))
        scaled_initial = initial / scale
        scaled_target = balance_target / scale
        standard_gibbs = self.thermodynamics.ideal.standard_gibbs_energy(
            temperature_value
        )
        standard_pressure = self.thermodynamics.ideal.standard_pressure
        tiny = jnp.finfo(initial.dtype).tiny

        def objective(amount, _):
            total = jnp.sum(amount)
            fraction = amount / jnp.maximum(total, tiny)
            logarithm = jnp.log(
                jnp.maximum(fraction, tiny) * pressure_value / standard_pressure
            )
            chemical = standard_gibbs + (
                UNIVERSAL_GAS_CONSTANT * temperature_value * logarithm
            )
            return scale * jnp.sum(jnp.where(amount > 0.0, amount * chemical, 0.0))

        constraint = NonlinearConstraint(
            lambda amount, _: contract("es,s->e", self.balance_matrix, amount),
            lower=scaled_target,
            upper=scaled_target,
        )
        problem = MinimizationProblem(
            objective,
            bounds=Bounds(
                jnp.zeros_like(scaled_initial), jnp.full_like(scaled_initial, jnp.inf)
            ),
            constraints=(constraint,),
        )
        solved = minimize(
            problem,
            scaled_initial,
            method=PrimalDualInteriorPoint(
                mode="dense-filter",
                max_dense_dimension=max(32, count + len(self.active_balance_rows)),
            ),
            termination=OptimizationTermination(
                absolute_optimality=self.tolerance,
                relative_optimality=0.0,
                maximum_steps=self.maximum_steps,
            ),
        )
        amount = solved.parameters * scale
        total = jnp.sum(amount)
        fraction = amount / jnp.maximum(total, tiny)
        volume = total * UNIVERSAL_GAS_CONSTANT * temperature_value / pressure_value
        chemical = standard_gibbs + UNIVERSAL_GAS_CONSTANT * temperature_value * jnp.log(
            jnp.maximum(fraction, tiny) * pressure_value / standard_pressure
        )
        gibbs = jnp.sum(jnp.where(amount > 0.0, amount * chemical, 0.0))
        initial_fraction = initial / jnp.maximum(initial_total, tiny)
        initial_chemical = standard_gibbs + (
            UNIVERSAL_GAS_CONSTANT
            * temperature_value
            * jnp.log(
                jnp.maximum(initial_fraction, tiny) * pressure_value / standard_pressure
            )
        )
        initial_gibbs = jnp.sum(jnp.where(initial > 0.0, initial * initial_chemical, 0.0))
        balance_residual = (
            contract("es,s->e", self.balance_matrix, amount) - balance_target
        )
        element_rows = self.thermodynamics.schema.element_count
        full_balance = jnp.concatenate(
            (
                self.thermodynamics.schema.element_composition.astype(amount.dtype),
                self.thermodynamics.schema.charges.astype(amount.dtype)[None, :],
            ),
            axis=0,
        )
        full_residual = contract("es,s->e", full_balance, amount - initial)
        element_residual = full_residual[:element_rows]
        charge_residual = full_residual[element_rows]
        active = amount > jnp.sqrt(jnp.finfo(amount.dtype).eps) * scale
        stationarity = solved.diagnostics.final_optimality_norm
        successful = (
            solved.successful
            & jnp.isfinite(gibbs)
            & (
                gibbs
                <= initial_gibbs
                + self.tolerance * jnp.maximum(jnp.abs(initial_gibbs), 1.0)
            )
            & jnp.all(
                jnp.abs(balance_residual)
                <= self.tolerance * jnp.maximum(jnp.abs(balance_target), 1.0)
            )
            & jnp.all(amount >= 0.0)
        )
        evidence = IdealGasEquilibriumEvidence(
            element_residual,
            charge_residual,
            stationarity,
            gibbs - initial_gibbs,
            active,
            successful,
            self.equilibrium_id,
        )
        return IdealGasEquilibriumResult(
            amount,
            fraction,
            volume,
            chemical,
            gibbs,
            solved.status,
            evidence,
            self.equilibrium_id,
        )


def _independent_rows(matrix: np.ndarray, tolerance: float) -> tuple[int, ...]:
    selected = []
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
    "IdealGasEquilibriumEvidence",
    "IdealGasEquilibriumResult",
    "IdealGasGibbsEquilibriumPlan",
]
