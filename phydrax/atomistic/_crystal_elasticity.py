#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Crystal stress, elastic stability, and NVE evidence for native many-body solids."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import AbstractPreparedParticleNeighborhood, PeriodicCell
from ..linalg import DenseLinearOperator, OperatorProperties
from ..linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ..units import UnitDefinition
from ._many_body import ManyBodyPotential
from ._potential_program import PreparedAtomisticPotentialProgram


_VOIGT_COMPONENTS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class CrystalElasticityResult(StrictModule, NonTrainableState):
    """Energy, symmetric stress, elastic tensor, and Born stability evidence."""

    energy: Array
    stress: Array
    elastic_tensor: Array
    voigt_stiffness: Array
    stability_eigenvalues: Array
    maximum_stress_antisymmetry: Array
    maximum_elastic_symmetry_residual: Array
    minimum_stability_eigenvalue: Array
    stable: Array
    successful: Array
    energy_unit: UnitDefinition
    stress_unit: UnitDefinition
    potential_kind_ids: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy,
        stress,
        elastic_tensor,
        voigt_stiffness,
        stability_eigenvalues,
        stress_residual,
        elastic_residual,
        stability_tolerance,
        successful,
        energy_unit,
        stress_unit,
        potential_kind_ids,
        system_id,
        /,
    ):
        self.energy = jnp.asarray(energy).reshape(())
        self.stress = jnp.asarray(stress, dtype=self.energy.dtype).reshape((3, 3))
        self.elastic_tensor = jnp.asarray(
            elastic_tensor, dtype=self.energy.dtype
        ).reshape((3, 3, 3, 3))
        self.voigt_stiffness = jnp.asarray(
            voigt_stiffness, dtype=self.energy.dtype
        ).reshape((6, 6))
        self.stability_eigenvalues = jnp.asarray(
            stability_eigenvalues, dtype=self.energy.dtype
        ).reshape((6,))
        self.maximum_stress_antisymmetry = jnp.asarray(
            stress_residual, dtype=self.energy.dtype
        ).reshape(())
        self.maximum_elastic_symmetry_residual = jnp.asarray(
            elastic_residual, dtype=self.energy.dtype
        ).reshape(())
        self.minimum_stability_eigenvalue = jnp.min(self.stability_eigenvalues)
        self.stable = self.minimum_stability_eigenvalue > float(stability_tolerance)
        self.successful = jnp.asarray(successful, dtype=bool).reshape(()) & jnp.all(
            jnp.isfinite(self.stability_eigenvalues)
        )
        self.energy_unit = energy_unit
        self.stress_unit = stress_unit
        self.potential_kind_ids = tuple(str(value) for value in potential_kind_ids)
        self.system_id = str(system_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "crystal-elasticity-result",
                "system": self.system_id,
                "potential_kinds": list(self.potential_kind_ids),
                "energy_unit": energy_unit.unit_id,
                "stress_unit": stress_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(self.energy),
                        "stress": np.asarray(self.stress),
                        "elastic_tensor": np.asarray(self.elastic_tensor),
                        "voigt": np.asarray(self.voigt_stiffness),
                        "stability": np.asarray(self.stability_eigenvalues),
                        "successful": np.asarray(self.successful),
                    }
                ),
            }
        )


class CrystalElasticityPlan(StrictModule, NonTrainableState):
    """Fixed-neighborhood homogeneous-strain response for EAM/SW/Tersoff crystals."""

    potential: PreparedAtomisticPotentialProgram
    neighborhood: AbstractPreparedParticleNeighborhood
    equilibrium_fractional_positions: Array
    strain_step: float = eqx.field(static=True)
    stability_tolerance: float = eqx.field(static=True)
    stress_symmetry_tolerance: float = eqx.field(static=True)
    elastic_symmetry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        potential: PreparedAtomisticPotentialProgram,
        neighborhood: AbstractPreparedParticleNeighborhood,
        equilibrium_fractional_positions: ArrayLike,
        /,
        *,
        strain_step: float = 1.0e-4,
        stability_tolerance: float = 0.0,
        stress_symmetry_tolerance: float = 1.0e-9,
        elastic_symmetry_tolerance: float = 1.0e-8,
    ):
        if not isinstance(potential, PreparedAtomisticPotentialProgram):
            raise TypeError("potential must be PreparedAtomisticPotentialProgram.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be AbstractPreparedParticleNeighborhood.")
        if not potential.plan.terms or not all(
            isinstance(term, ManyBodyPotential) for term in potential.plan.terms
        ):
            raise ValueError(
                "Crystal elasticity is bounded to native EAM, SW, and Tersoff programs."
            )
        if not potential.plan.capabilities.cell_derivative:
            raise ValueError("Every potential term must support cell differentiation.")
        cell = potential.system.cell
        if cell is None or cell.rank != 3 or not cell.fully_periodic:
            raise ValueError("Crystal elasticity requires a fully periodic rank-3 cell.")
        fractional = np.asarray(equilibrium_fractional_positions)
        if fractional.shape != (potential.system.capacity, 3) or np.any(
            ~np.isfinite(fractional)
        ):
            raise ValueError(
                "equilibrium_fractional_positions must have shape (N,3) and be finite."
            )
        step = float(strain_step)
        stability = float(stability_tolerance)
        stress_tolerance = float(stress_symmetry_tolerance)
        elastic_tolerance = float(elastic_symmetry_tolerance)
        if (
            not isfinite(step)
            or step <= 0.0
            or any(
                not isfinite(value) or value < 0.0
                for value in (stability, stress_tolerance, elastic_tolerance)
            )
        ):
            raise ValueError("Strain step and elasticity tolerances are invalid.")
        self.potential = potential
        self.neighborhood = neighborhood
        self.equilibrium_fractional_positions = jnp.asarray(fractional)
        self.strain_step = step
        self.stability_tolerance = stability
        self.stress_symmetry_tolerance = stress_tolerance
        self.elastic_symmetry_tolerance = elastic_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "crystal-elasticity-plan",
                "potential": potential.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "cell": cell.cell_id,
                "strain_step": step,
                "stability_tolerance": stability,
                "stress_symmetry_tolerance": stress_tolerance,
                "elastic_symmetry_tolerance": elastic_tolerance,
                "fractional": array_tree_fingerprint(fractional),
            }
        )

    def evaluate(self, /) -> CrystalElasticityResult:
        cell = self.potential.system.cell
        if cell is None:
            raise RuntimeError("Validated periodic cell unexpectedly absent.")
        reference_vectors = np.asarray(cell.vectors)
        fractional = np.asarray(self.equilibrium_fractional_positions)
        identity = np.eye(3, dtype=reference_vectors.dtype)

        def strain_matrix(voigt: np.ndarray) -> np.ndarray:
            return np.asarray(
                (
                    (voigt[0], 0.5 * voigt[5], 0.5 * voigt[4]),
                    (0.5 * voigt[5], voigt[1], 0.5 * voigt[3]),
                    (0.5 * voigt[4], 0.5 * voigt[3], voigt[2]),
                ),
                dtype=voigt.dtype,
            )

        all_successful = True

        def energy_of_strain(voigt: np.ndarray) -> float:
            nonlocal all_successful
            deformation = identity + strain_matrix(voigt)
            vectors = np.einsum("ij,kj->ki", deformation, reference_vectors)
            deformed_cell = PeriodicCell(
                vectors,
                origin=np.asarray(cell.origin),
                periodic_axes=cell.periodic_axes,
                maximum_condition_number=max(
                    cell.certified_condition_number,
                    float(np.linalg.cond(vectors)),
                ),
            )
            positions = deformed_cell.cartesian(fractional)
            neighborhood = self.neighborhood.build(positions)
            evaluation = self.potential.evaluate(
                positions,
                neighborhood,
                cell=deformed_cell,
            )
            all_successful = all_successful and bool(evaluation.successful)
            return float(evaluation.energy)

        origin = np.zeros((6,), dtype=reference_vectors.dtype)
        energy = energy_of_strain(origin)
        step = self.strain_step
        stress_voigt = np.zeros((6,), dtype=float)
        stiffness = np.zeros((6, 6), dtype=float)
        unit_vectors = np.eye(6)
        plus_energy = np.zeros((6,), dtype=float)
        minus_energy = np.zeros((6,), dtype=float)
        for index in range(6):
            plus_energy[index] = energy_of_strain(step * unit_vectors[index])
            minus_energy[index] = energy_of_strain(-step * unit_vectors[index])
            stress_voigt[index] = (plus_energy[index] - minus_energy[index]) / (
                2.0 * step * cell.volume
            )
            stiffness[index, index] = (
                plus_energy[index] - 2.0 * energy + minus_energy[index]
            ) / (step**2 * cell.volume)
        for row in range(6):
            for column in range(row + 1, 6):
                pp = energy_of_strain(step * (unit_vectors[row] + unit_vectors[column]))
                pm = energy_of_strain(step * (unit_vectors[row] - unit_vectors[column]))
                mp = energy_of_strain(step * (-unit_vectors[row] + unit_vectors[column]))
                mm = energy_of_strain(-step * (unit_vectors[row] + unit_vectors[column]))
                value = (pp - pm - mp + mm) / (4.0 * step**2 * cell.volume)
                stiffness[row, column] = value
                stiffness[column, row] = value
        stress = jnp.asarray(
            (
                (stress_voigt[0], stress_voigt[5], stress_voigt[4]),
                (stress_voigt[5], stress_voigt[1], stress_voigt[3]),
                (stress_voigt[4], stress_voigt[3], stress_voigt[2]),
            )
        )
        stiffness_array = jnp.asarray(stiffness)
        elastic = jnp.zeros((3, 3, 3, 3), dtype=stiffness_array.dtype)
        for row, (i, j) in enumerate(_VOIGT_COMPONENTS):
            for column, (k, l) in enumerate(_VOIGT_COMPONENTS):
                elastic = elastic.at[i, j, k, l].set(stiffness_array[row, column])
                elastic = elastic.at[j, i, k, l].set(stiffness_array[row, column])
                elastic = elastic.at[i, j, l, k].set(stiffness_array[row, column])
                elastic = elastic.at[j, i, l, k].set(stiffness_array[row, column])
        elastic = 0.5 * (elastic + jnp.transpose(elastic, (2, 3, 0, 1)))
        stability_eigenvalues = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    stiffness_array,
                    properties=OperatorProperties(
                        self_adjoint=True,
                        evidence={"self_adjoint": "construction"},
                    ),
                )
            ),
            policy=EigenSolvePolicy(DenseEigh(), count=6, which="smallest-algebraic"),
        ).eigenvalues.real
        stress_residual = jnp.max(jnp.abs(stress - stress.T), initial=0.0)
        elastic_residual = jnp.max(
            jnp.abs(elastic - jnp.transpose(elastic, (1, 0, 2, 3))),
            initial=0.0,
        )
        elastic_residual = jnp.maximum(
            elastic_residual,
            jnp.max(
                jnp.abs(elastic - jnp.transpose(elastic, (2, 3, 0, 1))),
                initial=0.0,
            ),
        )
        successful = (
            all_successful
            and np.isfinite(energy)
            and np.all(np.isfinite(stiffness))
            and float(stress_residual) <= self.stress_symmetry_tolerance
            and float(elastic_residual) <= self.elastic_symmetry_tolerance
        )
        units = self.potential.system.plan.units
        kinds = tuple(term.kind.value for term in self.potential.plan.terms)
        return CrystalElasticityResult(
            energy,
            stress,
            elastic,
            stiffness_array,
            stability_eigenvalues,
            stress_residual,
            elastic_residual,
            self.stability_tolerance,
            successful,
            units.scale.energy_unit,
            units.pressure_unit,
            kinds,
            self.potential.system.plan.system_id,
        )


class CrystalNVEEvidence(StrictModule, NonTrainableState):
    """Raw bounded NVE conservation evidence; it does not run an integrator."""

    times: Array
    total_energies: Array
    linear_momenta: Array
    relative_energy_drift: Array
    maximum_energy_excursion: Array
    momentum_drift: Array
    successful: Array
    energy_unit: UnitDefinition
    momentum_unit: UnitDefinition
    trajectory_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        total_energies: ArrayLike,
        linear_momenta: ArrayLike,
        energy_unit: UnitDefinition,
        momentum_unit: UnitDefinition,
        /,
        *,
        trajectory_id: str,
        maximum_relative_energy_drift: float,
        maximum_momentum_drift: float,
    ):
        time = np.asarray(times, dtype=float)
        energy = np.asarray(total_energies, dtype=float)
        momentum = np.asarray(linear_momenta, dtype=float)
        if (
            time.ndim != 1
            or time.size < 3
            or energy.shape != time.shape
            or momentum.shape != (time.size, 3)
        ):
            raise ValueError(
                "NVE evidence requires T>=3 times, energies, and (T,3) momenta."
            )
        if (
            np.any(~np.isfinite(time))
            or np.any(~np.isfinite(energy))
            or np.any(~np.isfinite(momentum))
            or np.any(np.diff(time) <= 0.0)
        ):
            raise ValueError("NVE samples must be finite and strictly time ordered.")
        energy_scale = max(float(np.max(np.abs(energy))), np.finfo(float).tiny)
        slope = float(np.polyfit(time - time[0], energy, 1)[0])
        relative_drift = abs(slope) * float(time[-1] - time[0]) / energy_scale
        excursion = float(np.max(np.abs(energy - energy[0])) / energy_scale)
        momentum_drift = float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=1), initial=0.0)
        )
        energy_limit = float(maximum_relative_energy_drift)
        momentum_limit = float(maximum_momentum_drift)
        if any(
            not isfinite(value) or value < 0.0 for value in (energy_limit, momentum_limit)
        ):
            raise ValueError("NVE conservation limits must be finite and nonnegative.")
        self.times = jnp.asarray(time)
        self.total_energies = jnp.asarray(energy)
        self.linear_momenta = jnp.asarray(momentum)
        self.relative_energy_drift = jnp.asarray(relative_drift)
        self.maximum_energy_excursion = jnp.asarray(excursion)
        self.momentum_drift = jnp.asarray(momentum_drift)
        self.successful = jnp.asarray(
            relative_drift <= energy_limit and momentum_drift <= momentum_limit
        )
        self.energy_unit = energy_unit
        self.momentum_unit = momentum_unit
        self.trajectory_id = str(trajectory_id)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "crystal-nve-evidence",
                "trajectory": self.trajectory_id,
                "energy_unit": energy_unit.unit_id,
                "momentum_unit": momentum_unit.unit_id,
                "energy_limit": energy_limit,
                "momentum_limit": momentum_limit,
                "arrays": array_tree_fingerprint(
                    {"times": time, "energies": energy, "momenta": momentum}
                ),
            }
        )


__all__ = ["CrystalElasticityPlan", "CrystalElasticityResult", "CrystalNVEEvidence"]
