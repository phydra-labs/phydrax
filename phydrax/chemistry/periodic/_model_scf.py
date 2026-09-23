#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native restricted periodic Hubbard mean field over the canonical H/S pencil."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._periodic_cell import PeriodicCell
from ...discretization._reciprocal import ReciprocalMeshPlan
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)
from ...operators.periodic import PreparedPeriodicOrbitalPencil
from ...units import UnitDefinition
from .._state import PeriodicElectronicSectorPlan
from ._orbital_model import PeriodicHubbardMeanFieldPlan


class PeriodicSCFResult(StrictModule, NonTrainableState):
    energy: Array
    free_energy: Array
    entropy: Array
    chemical_potential: Array
    orbital_energies: Array
    occupations: Array
    coefficients: Array
    density_matrices: Array
    populations: Array
    residual: Array
    iterations: Array
    successful: Array
    energy_unit: UnitDefinition
    cell_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)
    pencil_id: str = eqx.field(static=True)
    mean_field_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy: ArrayLike,
        free_energy: ArrayLike,
        entropy: ArrayLike,
        chemical_potential: ArrayLike,
        orbital_energies: ArrayLike,
        occupations: ArrayLike,
        coefficients: ArrayLike,
        density_matrices: ArrayLike,
        populations: ArrayLike,
        residual: ArrayLike,
        iterations: int,
        successful: ArrayLike,
        energy_unit: UnitDefinition,
        /,
        *,
        cell_id: str,
        mesh_id: str,
        sector_id: str,
        pencil_id: str,
        mean_field_id: str,
    ):
        orbital = jnp.asarray(orbital_energies)
        occupation = jnp.asarray(occupations, dtype=orbital.real.dtype)
        coefficient = jnp.asarray(coefficients)
        density = jnp.asarray(density_matrices)
        population = jnp.asarray(populations, dtype=orbital.real.dtype)
        if (
            orbital.ndim != 2
            or occupation.shape != orbital.shape
            or coefficient.shape != (orbital.shape[0], orbital.shape[1], orbital.shape[1])
            or density.shape != coefficient.shape
            or population.shape != (orbital.shape[1],)
            or not isinstance(energy_unit, UnitDefinition)
        ):
            raise ValueError(
                "Periodic SCF orbital, coefficient, density, or population arrays do not align."
            )
        self.energy = jnp.asarray(energy, dtype=orbital.real.dtype).reshape(())
        self.free_energy = jnp.asarray(free_energy, dtype=orbital.real.dtype).reshape(())
        self.entropy = jnp.asarray(entropy, dtype=orbital.real.dtype).reshape(())
        self.chemical_potential = jnp.asarray(
            chemical_potential, dtype=orbital.real.dtype
        ).reshape(())
        self.orbital_energies = orbital
        self.occupations = occupation
        self.coefficients = coefficient
        self.density_matrices = density
        self.populations = population
        self.residual = jnp.asarray(residual, dtype=orbital.real.dtype).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.energy_unit = energy_unit
        self.cell_id = str(cell_id)
        self.mesh_id = str(mesh_id)
        self.sector_id = str(sector_id)
        self.pencil_id = str(pencil_id)
        self.mean_field_id = str(mean_field_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-scf-result",
                "cell": self.cell_id,
                "mesh": self.mesh_id,
                "sector": self.sector_id,
                "pencil": self.pencil_id,
                "mean_field": self.mean_field_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(self.energy),
                        "free_energy": np.asarray(self.free_energy),
                        "entropy": np.asarray(self.entropy),
                        "chemical_potential": np.asarray(self.chemical_potential),
                        "orbital_energies": np.asarray(orbital),
                        "occupations": np.asarray(occupation),
                        "coefficients": np.asarray(coefficient),
                        "density_matrices": np.asarray(density),
                        "populations": np.asarray(population),
                        "residual": np.asarray(self.residual),
                    }
                ),
            }
        )


class NativePeriodicSCFPlan(StrictModule, NonTrainableState):
    cell: PeriodicCell
    mesh: ReciprocalMeshPlan
    sector: PeriodicElectronicSectorPlan
    pencil: PreparedPeriodicOrbitalPencil
    mean_field: PeriodicHubbardMeanFieldPlan
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        mesh: ReciprocalMeshPlan,
        sector: PeriodicElectronicSectorPlan,
        pencil: PreparedPeriodicOrbitalPencil,
        mean_field: PeriodicHubbardMeanFieldPlan,
        /,
        *,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-10,
        maximum_iterations: int = 256,
        damping: float = 0.25,
    ):
        if not isinstance(cell, PeriodicCell) or not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError(
                "Native periodic SCF requires typed cell and reciprocal mesh."
            )
        if (
            not isinstance(sector, PeriodicElectronicSectorPlan)
            or not isinstance(pencil, PreparedPeriodicOrbitalPencil)
            or not isinstance(mean_field, PeriodicHubbardMeanFieldPlan)
        ):
            raise TypeError(
                "Native periodic SCF requires sector, prepared pencil, and mean-field plans."
            )
        mesh.require_cell(cell)
        if pencil.plan.basis.cell_id != cell.cell_id:
            raise ValueError("SCF cell and orbital pencil cell do not match.")
        if mean_field.basis_id != pencil.plan.basis.basis_id:
            raise ValueError("SCF mean field belongs to a different orbital basis.")
        if mean_field.energy_unit != pencil.plan.energy_unit:
            raise ValueError("SCF pencil and mean-field energy units do not match.")
        if sector.charge_per_cell != 0.0 or sector.spin_magnetization != 0.0:
            raise ValueError(
                "Restricted native periodic SCF requires a neutral zero-magnetization sector."
            )
        if sector.electron_count > 2.0 * pencil.plan.basis.orbital_count:
            raise ValueError(
                "Periodic electron count exceeds orbital occupation capacity."
            )
        if not np.isclose(
            float(np.sum(np.asarray(mean_field.reference_populations))),
            sector.electron_count,
            atol=1.0e-10,
        ):
            raise ValueError(
                "Periodic reference populations must sum to electrons per cell."
            )
        smearing = float(smearing_energy)
        tolerance = float(convergence_tolerance)
        iterations = int(maximum_iterations)
        damping_ = float(damping)
        if (
            not isfinite(smearing)
            or smearing < 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or iterations <= 0
            or not isfinite(damping_)
            or not 0.0 <= damping_ < 1.0
        ):
            raise ValueError(
                "Periodic SCF smearing, tolerance, iteration, or damping policy is invalid."
            )
        self.cell = cell
        self.mesh = mesh
        self.sector = sector
        self.pencil = pencil
        self.mean_field = mean_field
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_iterations = iterations
        self.damping = damping_
        self.provider_id = canonical_fingerprint(
            {
                "kind": "native-periodic-scf-plan",
                "cell": cell.cell_id,
                "mesh": mesh.mesh_id,
                "sector": sector.sector_id,
                "pencil": pencil.prepared_id,
                "mean_field": mean_field.plan_id,
                "smearing_energy": smearing,
                "convergence_tolerance": tolerance,
                "maximum_iterations": iterations,
                "damping": damping_,
            }
        )

    def _occupations(self, energies: Array, /) -> tuple[Array, Array, Array]:
        target = self.sector.electron_count
        if self.smearing_energy == 0.0:
            occupied = int(round(target / 2.0))
            if abs(2.0 * occupied - target) > 1.0e-12 or occupied < 1:
                raise ValueError(
                    "Zero-smearing periodic SCF requires a positive even electron count."
                )
            occupations = jnp.zeros_like(energies.real).at[:, :occupied].set(2.0)
            highest = jnp.max(energies[:, occupied - 1].real)
            if occupied < energies.shape[1]:
                lowest = jnp.min(energies[:, occupied].real)
                if float(highest) >= float(lowest) - 1.0e-12 * max(
                    float(jnp.max(jnp.abs(energies.real))), 1.0
                ):
                    raise ValueError(
                        "Zero-smearing periodic SCF requires an insulating band gap."
                    )
                chemical = 0.5 * (highest + lowest)
            else:
                chemical = highest
            return occupations, chemical, jnp.asarray(0.0, dtype=energies.real.dtype)
        lower = float(jnp.min(energies.real)) - 80.0 * self.smearing_energy
        upper = float(jnp.max(energies.real)) + 80.0 * self.smearing_energy
        weights = self.mesh.weights[:, None]
        for _ in range(180):
            chemical = 0.5 * (lower + upper)
            occupation = 2.0 / (
                1.0 + jnp.exp((energies.real - chemical) / self.smearing_energy)
            )
            if float(jnp.sum(weights * occupation)) > target:
                upper = chemical
            else:
                lower = chemical
        chemical = jnp.asarray(0.5 * (lower + upper), dtype=energies.real.dtype)
        occupations = 2.0 / (
            1.0 + jnp.exp((energies.real - chemical) / self.smearing_energy)
        )
        probability = jnp.clip(0.5 * occupations, 1.0e-15, 1.0 - 1.0e-15)
        entropy = -2.0 * jnp.sum(
            weights
            * (
                probability * jnp.log(probability)
                + (1.0 - probability) * jnp.log(1.0 - probability)
            )
        )
        return occupations, chemical, entropy

    def evaluate(self, /) -> PeriodicSCFResult:
        evaluation = self.pencil.evaluate(self.mesh.fractional_points)
        if not bool(evaluation.successful):
            raise ValueError(
                "Periodic SCF overlap pencil failed positivity/condition admission."
            )
        hamiltonian = evaluation.hamiltonians
        overlap = evaluation.overlaps
        orbital_count = self.pencil.plan.basis.orbital_count
        populations = jnp.asarray(self.mean_field.reference_populations)
        previous_energy = jnp.asarray(jnp.inf, dtype=populations.dtype)
        residual = jnp.asarray(jnp.inf, dtype=populations.dtype)
        energies = jnp.zeros(
            (self.mesh.fractional_points.shape[0], orbital_count), dtype=populations.dtype
        )
        coefficients = jnp.zeros(
            (self.mesh.fractional_points.shape[0], orbital_count, orbital_count),
            dtype=hamiltonian.dtype,
        )
        occupations = jnp.zeros_like(energies)
        density = jnp.zeros_like(coefficients)
        chemical = jnp.asarray(0.0, dtype=populations.dtype)
        entropy = jnp.asarray(0.0, dtype=populations.dtype)
        total_energy = jnp.asarray(jnp.nan, dtype=populations.dtype)
        converged = False
        completed = 0
        operator_properties = OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        )
        metric_properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        for iteration in range(self.maximum_iterations):
            potential = jnp.diag(
                self.mean_field.onsite_hubbard
                * (populations - self.mean_field.reference_populations)
            )
            solved = eigensolve(
                GeneralizedEigenproblem(
                    DenseLinearOperator(
                        hamiltonian + potential[None, :, :],
                        properties=operator_properties,
                    ),
                    DenseLinearOperator(overlap, properties=metric_properties),
                    problem_id=f"{self.provider_id}:{iteration}",
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(), count=orbital_count, which="smallest-algebraic"
                ),
            )
            if not bool(jnp.all(solved.successful)):
                raise RuntimeError("Native periodic generalized eigensolve failed.")
            energies = solved.eigenvalues.real
            coefficients = solved.eigenvectors
            occupations, chemical, entropy = self._occupations(energies)
            density = contract(
                "kn,kan,kbn->kab",
                occupations,
                coefficients,
                jnp.conj(coefficients),
                backend="jax",
            )
            proposed = jnp.real(
                contract(
                    "k,kab,kba->a", self.mesh.weights, density, overlap, backend="jax"
                )
            )
            mixed = (1.0 - self.damping) * proposed + self.damping * populations
            one_electron = jnp.real(
                contract(
                    "k,kab,kba->", self.mesh.weights, density, hamiltonian, backend="jax"
                )
            )
            delta = mixed - self.mean_field.reference_populations
            interaction = 0.5 * jnp.sum(self.mean_field.onsite_hubbard * delta * delta)
            total_energy = one_electron + interaction + self.mean_field.ionic_energy
            residual = jnp.maximum(
                jnp.max(jnp.abs(mixed - populations)),
                jnp.abs(total_energy - previous_energy),
            )
            populations = mixed
            previous_energy = total_energy
            completed = iteration + 1
            if bool(residual <= self.convergence_tolerance):
                converged = True
                break
        return PeriodicSCFResult(
            total_energy,
            total_energy - self.smearing_energy * entropy,
            entropy,
            chemical,
            energies,
            occupations,
            coefficients,
            density,
            populations,
            residual,
            completed,
            converged,
            self.pencil.plan.energy_unit,
            cell_id=self.cell.cell_id,
            mesh_id=self.mesh.mesh_id,
            sector_id=self.sector.sector_id,
            pencil_id=self.pencil.plan.pencil_id,
            mean_field_id=self.mean_field.plan_id,
        )


__all__ = ["NativePeriodicSCFPlan", "PeriodicSCFResult"]
