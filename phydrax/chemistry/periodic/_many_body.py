#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Diagonal quasiparticle GW roots and resonant/full Bethe--Salpeter manifolds."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import Bisection, NonlinearTermination, scalar_root, ScalarRootProblem
from ...units import UnitDefinition
from ..excited import RandomPhaseApproximationPlan
from ..excited._tda import ExcitedStateManifoldPlan, TammDancoffPlan


DiagonalSelfEnergy = Callable[[int, Array], Array]


class GWQuasiparticleResult(StrictModule, NonTrainableState):
    mean_field_energies: Array
    quasiparticle_energies: Array
    renormalization_factors: Array
    residuals: Array
    successful_roots: Array
    successful: Array
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_field_energies,
        quasiparticle_energies,
        renormalization_factors,
        residuals,
        successful_roots,
        energy_unit,
        plan_id,
        /,
    ):
        mean_field = jnp.asarray(mean_field_energies)
        quasiparticle = jnp.asarray(quasiparticle_energies, dtype=mean_field.dtype)
        factors = jnp.asarray(renormalization_factors, dtype=mean_field.real.dtype)
        residuals_ = jnp.asarray(residuals, dtype=mean_field.real.dtype)
        roots = jnp.asarray(successful_roots, dtype=bool)
        if any(
            value.shape != mean_field.shape
            for value in (quasiparticle, factors, residuals_, roots)
        ) or not isinstance(energy_unit, UnitDefinition):
            raise ValueError("GW quasiparticle arrays or energy unit do not align.")
        self.mean_field_energies = mean_field
        self.quasiparticle_energies = quasiparticle
        self.renormalization_factors = factors
        self.residuals = residuals_
        self.successful_roots = roots
        self.successful = (
            jnp.all(roots)
            & jnp.all(jnp.isfinite(quasiparticle))
            & jnp.all(jnp.isfinite(factors))
            & jnp.all(jnp.isfinite(residuals_))
        )
        self.energy_unit = energy_unit
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "gw-quasiparticle-result",
                "plan": self.plan_id,
                "energy_unit": energy_unit.unit_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "mean_field": np.asarray(mean_field),
                        "quasiparticle": np.asarray(quasiparticle),
                        "renormalization": np.asarray(factors),
                        "residuals": np.asarray(residuals_),
                    }
                ),
            }
        )


class DiagonalGWPlan(StrictModule, NonTrainableState):
    mean_field_energies: Array
    mean_field_xc_expectations: Array
    brackets: Array
    self_energy: DiagonalSelfEnergy = eqx.field(static=True)
    self_energy_definition_id: str = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_field_energies: ArrayLike,
        mean_field_xc_expectations: ArrayLike,
        brackets: ArrayLike,
        self_energy: DiagonalSelfEnergy,
        self_energy_definition_id: str,
        energy_unit: UnitDefinition,
        /,
        *,
        residual_tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
    ):
        energies = jnp.asarray(mean_field_energies)
        xc = jnp.asarray(mean_field_xc_expectations, dtype=energies.dtype)
        brackets_ = jnp.asarray(brackets, dtype=energies.real.dtype)
        definition = str(self_energy_definition_id).strip()
        tolerance = float(residual_tolerance)
        maximum = int(maximum_iterations)
        if (
            energies.ndim != 1
            or xc.shape != energies.shape
            or brackets_.shape != (energies.size, 2)
            or bool(jnp.any(brackets_[:, 1] <= brackets_[:, 0]))
            or not callable(self_energy)
            or not definition
            or not isinstance(energy_unit, UnitDefinition)
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or maximum <= 0
        ):
            raise ValueError(
                "GW energies, brackets, self-energy definition, or solve policy is invalid."
            )
        self.mean_field_energies = energies
        self.mean_field_xc_expectations = xc
        self.brackets = brackets_
        self.self_energy = self_energy
        self.self_energy_definition_id = definition
        self.residual_tolerance = tolerance
        self.maximum_iterations = maximum
        self.energy_unit = energy_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diagonal-gw-plan",
                "self_energy_definition": definition,
                "residual_tolerance": tolerance,
                "maximum_iterations": maximum,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "mean_field": np.asarray(energies),
                        "xc": np.asarray(xc),
                        "brackets": np.asarray(brackets_),
                    }
                ),
            }
        )

    def evaluate(self, /) -> GWQuasiparticleResult:
        values = []
        factors = []
        residuals = []
        successful = []
        for index in range(int(self.mean_field_energies.size)):
            state_index = index

            def equation(energy, _, state_index=state_index):
                return (
                    energy
                    - self.mean_field_energies[state_index]
                    - jnp.real(self.self_energy(state_index, energy))
                    + self.mean_field_xc_expectations[state_index]
                )

            result = scalar_root(
                ScalarRootProblem(
                    equation,
                    bracket=(
                        self.brackets[state_index, 0],
                        self.brackets[state_index, 1],
                    ),
                    problem_id=(f"gw-quasiparticle:{self.plan_id}:{state_index}"),
                ),
                method=Bisection(),
                termination=NonlinearTermination(
                    absolute_residual=self.residual_tolerance,
                    relative_residual=0.0,
                    maximum_steps=self.maximum_iterations,
                    maximum_evaluations=2 * self.maximum_iterations + 4,
                    maximum_linear_iterations=1,
                ),
            )
            derivative = jax.grad(
                lambda energy, state_index=state_index: jnp.real(
                    self.self_energy(state_index, energy)
                )
            )(result.root)
            values.append(result.root)
            factors.append(1.0 / (1.0 - derivative))
            residuals.append(jnp.abs(equation(result.root, None)))
            successful.append(result.successful)
        return GWQuasiparticleResult(
            self.mean_field_energies,
            jnp.stack(tuple(values)),
            jnp.stack(tuple(factors)),
            jnp.stack(tuple(residuals)),
            jnp.stack(tuple(successful)),
            self.energy_unit,
            self.plan_id,
        )


class BetheSalpeterPlan(StrictModule, NonTrainableState):
    transition_energies: Array
    screened_direct: Array
    bare_exchange: Array
    coupling: Array
    transition_dipoles: Array
    ground_state_energy: Array
    energy_unit: UnitDefinition
    transition_dipole_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_energies: ArrayLike,
        screened_direct: ArrayLike,
        bare_exchange: ArrayLike,
        transition_dipoles: ArrayLike,
        ground_state_energy: ArrayLike,
        energy_unit: UnitDefinition,
        transition_dipole_unit: UnitDefinition,
        /,
        *,
        coupling: ArrayLike | None = None,
    ):
        transition_input = jnp.asarray(transition_energies)
        transitions = jnp.real(transition_input)
        direct_input = jnp.asarray(screened_direct)
        exchange_input = jnp.asarray(bare_exchange)
        coupling_input = (
            jnp.zeros_like(direct_input) if coupling is None else jnp.asarray(coupling)
        )
        kernel_dtype = jnp.result_type(
            transitions.dtype,
            direct_input.dtype,
            exchange_input.dtype,
            coupling_input.dtype,
        )
        direct = direct_input.astype(kernel_dtype)
        exchange = exchange_input.astype(kernel_dtype)
        coupling_ = coupling_input.astype(kernel_dtype)
        dipoles = jnp.asarray(transition_dipoles).astype(
            jnp.result_type(kernel_dtype, jnp.asarray(transition_dipoles).dtype)
        )
        count = int(transitions.size)
        if (
            transitions.shape != (count,)
            or direct.shape != (count, count)
            or exchange.shape != direct.shape
            or coupling_.shape != direct.shape
            or dipoles.shape != (count, 3)
            or bool(jnp.any(jnp.abs(jnp.imag(transition_input)) > 1.0e-12))
            or bool(jnp.any(transitions <= 0.0))
            or not np.allclose(
                np.asarray(direct),
                np.asarray(jnp.conj(direct.T)),
                rtol=0.0,
                atol=1.0e-10,
            )
            or not np.allclose(
                np.asarray(exchange),
                np.asarray(jnp.conj(exchange.T)),
                rtol=0.0,
                atol=1.0e-10,
            )
            or not np.allclose(
                np.asarray(coupling_),
                np.asarray(coupling_.T),
                rtol=0.0,
                atol=1.0e-10,
            )
            or not bool(jnp.all(jnp.isfinite(direct)))
            or not bool(jnp.all(jnp.isfinite(exchange)))
            or not bool(jnp.all(jnp.isfinite(coupling_)))
            or not bool(jnp.all(jnp.isfinite(dipoles)))
            or not isinstance(energy_unit, UnitDefinition)
            or not isinstance(transition_dipole_unit, UnitDefinition)
        ):
            raise ValueError(
                "BSE transitions, Hermitian kernels, dipoles, or units are invalid."
            )
        self.transition_energies = transitions
        self.screened_direct = direct
        self.bare_exchange = exchange
        self.coupling = coupling_
        self.transition_dipoles = dipoles
        self.ground_state_energy = jnp.asarray(
            ground_state_energy, dtype=transitions.dtype
        ).reshape(())
        self.energy_unit = energy_unit
        self.transition_dipole_unit = transition_dipole_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bethe-salpeter-plan",
                "energy_unit": energy_unit.unit_id,
                "transition_dipole_unit": transition_dipole_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "transitions": np.asarray(transitions),
                        "direct": np.asarray(direct),
                        "exchange": np.asarray(exchange),
                        "coupling": np.asarray(coupling_),
                        "dipoles": np.asarray(dipoles),
                    }
                ),
            }
        )

    @property
    def resonant_matrix(self) -> Array:
        return (
            jnp.diag(self.transition_energies) + self.bare_exchange - self.screened_direct
        )

    def tda(self, root_count: int, /):
        return TammDancoffPlan(
            ExcitedStateManifoldPlan(root_count, spin_sector="singlet"),
            self.resonant_matrix,
            self.transition_dipoles,
            self.ground_state_energy,
            self.energy_unit,
            self.transition_dipole_unit,
        ).solve()

    def full(self, root_count: int, /):
        return RandomPhaseApproximationPlan(
            self.resonant_matrix,
            self.coupling,
            self.transition_dipoles,
            self.ground_state_energy,
            root_count,
            "bse",
            self.energy_unit,
            self.transition_dipole_unit,
            spin_sector="singlet",
        ).solve()


__all__ = [
    "BetheSalpeterPlan",
    "DiagonalGWPlan",
    "GWQuasiparticleResult",
]
