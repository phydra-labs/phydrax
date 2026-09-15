#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Spin-resolved k-point periodic AO SCF with metallic Fermi smearing."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from ...ein import contract
from ...units import UnitDefinition
from ._model_scf import (
    _hermitian_eigh,
    KPointMeshPlan,
    PeriodicAOModelPlan,
    PeriodicElectronicSectorPlan,
)


class SpinPeriodicSCFResult(StrictModule, NonTrainableState):
    energy: Array
    free_energy: Array
    entropy: Array
    chemical_potentials: Array
    orbital_energies: Array
    occupations: Array
    density_matrices: Array
    spin_populations: Array
    residual: Array
    iterations: Array
    successful: Array
    energy_unit: UnitDefinition
    cell_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy,
        free_energy,
        entropy,
        chemical_potentials,
        orbital_energies,
        occupations,
        density_matrices,
        spin_populations,
        residual,
        iterations,
        successful,
        energy_unit,
        cell_id,
        mesh_id,
        sector_id,
        model_id,
        /,
    ):
        orbital = jnp.asarray(orbital_energies)
        occupation = jnp.asarray(occupations, dtype=orbital.real.dtype)
        density = jnp.asarray(density_matrices)
        populations = jnp.asarray(spin_populations, dtype=orbital.real.dtype)
        if (
            orbital.ndim != 3
            or orbital.shape[0] != 2
            or occupation.shape != orbital.shape
            or density.shape != (2, orbital.shape[1], orbital.shape[2], orbital.shape[2])
            or populations.shape != (2, orbital.shape[2])
            or not isinstance(energy_unit, UnitDefinition)
        ):
            raise ValueError("Spin periodic SCF arrays or units do not align.")
        self.energy = jnp.asarray(energy, dtype=orbital.real.dtype).reshape(())
        self.free_energy = jnp.asarray(free_energy, dtype=orbital.real.dtype).reshape(())
        self.entropy = jnp.asarray(entropy, dtype=orbital.real.dtype).reshape(())
        self.chemical_potentials = jnp.asarray(
            chemical_potentials, dtype=orbital.real.dtype
        ).reshape((2,))
        self.orbital_energies = orbital
        self.occupations = occupation
        self.density_matrices = density
        self.spin_populations = populations
        self.residual = jnp.asarray(residual, dtype=orbital.real.dtype).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.energy_unit = energy_unit
        self.cell_id = str(cell_id)
        self.mesh_id = str(mesh_id)
        self.sector_id = str(sector_id)
        self.model_id = str(model_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "spin-periodic-scf-result",
                "cell": self.cell_id,
                "mesh": self.mesh_id,
                "sector": self.sector_id,
                "model": self.model_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(self.energy),
                        "free_energy": np.asarray(self.free_energy),
                        "entropy": np.asarray(self.entropy),
                        "chemical_potentials": np.asarray(self.chemical_potentials),
                        "orbital_energies": np.asarray(orbital),
                        "occupations": np.asarray(occupation),
                        "density": np.asarray(density),
                        "populations": np.asarray(populations),
                        "residual": np.asarray(self.residual),
                    }
                ),
            }
        )


def _spin_occupations(energies, target, weights, smearing):
    if smearing == 0.0:
        occupied = int(round(target))
        if (
            abs(occupied - target) > 1.0e-12
            or occupied < 0
            or occupied > energies.shape[1]
        ):
            raise ValueError(
                "Zero-smearing spin occupations require an integer band count."
            )
        occupations = jnp.zeros_like(energies.real).at[:, :occupied].set(1.0)
        chemical = energies[:, max(occupied - 1, 0)].real.max()
        return occupations, chemical, jnp.asarray(0.0, dtype=energies.real.dtype)
    lower = float(jnp.min(energies.real)) - 80.0 * smearing
    upper = float(jnp.max(energies.real)) + 80.0 * smearing
    for _ in range(180):
        chemical = 0.5 * (lower + upper)
        occupation = 1.0 / (1.0 + jnp.exp((energies.real - chemical) / smearing))
        count = float(jnp.sum(weights[:, None] * occupation))
        if count > target:
            upper = chemical
        else:
            lower = chemical
    chemical = jnp.asarray(0.5 * (lower + upper), dtype=energies.real.dtype)
    occupations = 1.0 / (1.0 + jnp.exp((energies.real - chemical) / smearing))
    probability = jnp.clip(occupations, 1.0e-15, 1.0 - 1.0e-15)
    entropy = -jnp.sum(
        weights[:, None]
        * (
            probability * jnp.log(probability)
            + (1.0 - probability) * jnp.log(1.0 - probability)
        )
    )
    return occupations, chemical, entropy


class SpinPeriodicSCFPlan(StrictModule, NonTrainableState):
    cell: PeriodicCell
    mesh: KPointMeshPlan
    sector: PeriodicElectronicSectorPlan
    model: PeriodicAOModelPlan
    reference_spin_populations: Array
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        mesh: KPointMeshPlan,
        sector: PeriodicElectronicSectorPlan,
        model: PeriodicAOModelPlan,
        /,
        *,
        reference_spin_populations: ArrayLike | None = None,
        smearing_energy: float = 0.01,
        convergence_tolerance: float = 1.0e-9,
        maximum_iterations: int = 256,
        damping: float = 0.25,
    ):
        if (
            not isinstance(cell, PeriodicCell)
            or not all(cell.periodic_axes)
            or not isinstance(mesh, KPointMeshPlan)
            or not isinstance(sector, PeriodicElectronicSectorPlan)
            or not isinstance(model, PeriodicAOModelPlan)
        ):
            raise TypeError(
                "Spin periodic SCF requires typed 3D cell, mesh, sector, and model."
            )
        if sector.charge_per_cell != 0.0:
            raise ValueError("Native spin periodic SCF requires a neutral cell.")
        alpha_count = 0.5 * (sector.electron_count + sector.spin_magnetization)
        beta_count = 0.5 * (sector.electron_count - sector.spin_magnetization)
        if (
            min(alpha_count, beta_count) < 0.0
            or max(alpha_count, beta_count) > model.orbital_count
        ):
            raise ValueError("Spin populations exceed periodic orbital capacity.")
        reference = (
            jnp.stack(
                (
                    model.reference_populations * alpha_count / sector.electron_count,
                    model.reference_populations * beta_count / sector.electron_count,
                )
            )
            if reference_spin_populations is None
            else jnp.asarray(reference_spin_populations)
        )
        if (
            reference.shape != (2, model.orbital_count)
            or not np.isclose(float(jnp.sum(reference[0])), alpha_count, atol=1.0e-10)
            or not np.isclose(float(jnp.sum(reference[1])), beta_count, atol=1.0e-10)
        ):
            raise ValueError("Reference spin populations must match sector populations.")
        smearing = float(smearing_energy)
        tolerance = float(convergence_tolerance)
        maximum = int(maximum_iterations)
        damping_ = float(damping)
        if (
            not isfinite(smearing)
            or smearing < 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or maximum <= 0
            or not isfinite(damping_)
            or not 0.0 <= damping_ < 1.0
        ):
            raise ValueError(
                "Spin periodic SCF smearing, tolerance, or iteration policy is invalid."
            )
        self.cell = cell
        self.mesh = mesh
        self.sector = sector
        self.model = model
        self.reference_spin_populations = reference
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_iterations = maximum
        self.damping = damping_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spin-periodic-scf-plan",
                "cell": cell.cell_id,
                "mesh": mesh.mesh_id,
                "sector": sector.sector_id,
                "model": model.model_id,
                "smearing_energy": smearing,
                "convergence_tolerance": tolerance,
                "maximum_iterations": maximum,
                "damping": damping_,
                "reference_spin_populations": array_tree_fingerprint(
                    np.asarray(reference)
                ),
            }
        )

    def evaluate(self, /) -> SpinPeriodicSCFResult:
        hamiltonian, overlap = self.model.bloch_matrices(self.mesh)
        k_count = int(self.mesh.weights.size)
        orbital_count = self.model.orbital_count
        populations = self.reference_spin_populations
        alpha_target = 0.5 * (self.sector.electron_count + self.sector.spin_magnetization)
        beta_target = 0.5 * (self.sector.electron_count - self.sector.spin_magnetization)
        targets = (alpha_target, beta_target)
        energies = jnp.zeros((2, k_count, orbital_count), dtype=hamiltonian.real.dtype)
        occupations = jnp.zeros_like(energies)
        density = jnp.zeros(
            (2, k_count, orbital_count, orbital_count), dtype=hamiltonian.dtype
        )
        chemical = jnp.zeros((2,), dtype=hamiltonian.real.dtype)
        entropy = jnp.asarray(0.0, dtype=hamiltonian.real.dtype)
        previous_energy = jnp.asarray(jnp.inf, dtype=hamiltonian.real.dtype)
        residual = jnp.asarray(jnp.inf, dtype=hamiltonian.real.dtype)
        total_energy = jnp.asarray(jnp.nan, dtype=hamiltonian.real.dtype)
        converged = False
        completed = 0
        orthogonalizers = []
        for kpoint in range(k_count):
            overlap_values, overlap_vectors = _hermitian_eigh(overlap[kpoint])
            if bool(jnp.min(overlap_values) <= 1.0e-10):
                raise ValueError("Spin periodic overlap is rank deficient.")
            orthogonalizers.append(
                overlap_vectors
                @ jnp.diag(overlap_values**-0.5)
                @ jnp.conj(overlap_vectors.T)
            )
        for iteration in range(self.maximum_iterations):
            entropy = jnp.asarray(0.0, dtype=hamiltonian.real.dtype)
            for spin in range(2):
                other = 1 - spin
                potential = jnp.diag(
                    self.model.onsite_hubbard
                    * (populations[other] - self.reference_spin_populations[other])
                )
                coefficients_by_k = []
                values_by_k = []
                for kpoint in range(k_count):
                    orthogonalizer = orthogonalizers[kpoint]
                    values, vectors = _hermitian_eigh(
                        jnp.conj(orthogonalizer.T)
                        @ (hamiltonian[kpoint] + potential)
                        @ orthogonalizer
                    )
                    values_by_k.append(values.real)
                    coefficients_by_k.append(orthogonalizer @ vectors)
                spin_energies = jnp.stack(tuple(values_by_k))
                coefficients = jnp.stack(tuple(coefficients_by_k))
                spin_occupations, spin_chemical, spin_entropy = _spin_occupations(
                    spin_energies, targets[spin], self.mesh.weights, self.smearing_energy
                )
                spin_density = contract(
                    "kn,kan,kbn->kab",
                    spin_occupations,
                    coefficients,
                    jnp.conj(coefficients),
                )
                energies = energies.at[spin].set(spin_energies)
                occupations = occupations.at[spin].set(spin_occupations)
                density = density.at[spin].set(spin_density)
                chemical = chemical.at[spin].set(spin_chemical)
                entropy = entropy + spin_entropy
            proposed = jnp.real(
                contract("k,skab,kba->sa", self.mesh.weights, density, overlap)
            )
            mixed = self.damping * populations + (1.0 - self.damping) * proposed
            one_electron = jnp.real(
                contract("k,skab,kba->", self.mesh.weights, density, hamiltonian)
            )
            delta = mixed - self.reference_spin_populations
            interaction = jnp.sum(self.model.onsite_hubbard * delta[0] * delta[1])
            total_energy = one_electron + interaction + self.model.ionic_energy
            residual = jnp.maximum(
                jnp.max(jnp.abs(mixed - populations), initial=0.0),
                jnp.abs(total_energy - previous_energy),
            )
            populations = mixed
            previous_energy = total_energy
            completed = iteration + 1
            if bool(residual <= self.convergence_tolerance):
                converged = True
                break
        free_energy = total_energy - self.smearing_energy * entropy
        return SpinPeriodicSCFResult(
            total_energy,
            free_energy,
            entropy,
            chemical,
            energies,
            occupations,
            density,
            populations,
            residual,
            completed,
            converged,
            self.model.energy_unit,
            self.cell.cell_id,
            self.mesh.mesh_id,
            self.sector.sector_id,
            self.model.model_id,
        )


__all__ = ["SpinPeriodicSCFPlan", "SpinPeriodicSCFResult"]
