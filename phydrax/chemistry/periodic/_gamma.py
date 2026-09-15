#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded Gamma-point plane-wave FFTDF and Gaussian GDF mean fields."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...operators.quantum.gaussian import FactorizedERITensor
from ...units import UnitDefinition
from ._model_scf import _hermitian_eigh


class GammaSCFResult(StrictModule, NonTrainableState):
    total_energy: Array
    free_energy: Array
    orbital_energies: Array
    occupations: Array
    density: Array
    residual: Array
    iterations: Array
    successful: Array
    backend: str = eqx.field(static=True)
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_energy,
        free_energy,
        orbital_energies,
        occupations,
        density,
        residual,
        iterations,
        successful,
        backend,
        energy_unit,
        plan_id,
        /,
    ):
        energies = jnp.asarray(orbital_energies)
        occupations_ = jnp.asarray(occupations, dtype=energies.real.dtype)
        density_ = jnp.asarray(density)
        backend_ = str(backend).strip()
        if (
            occupations_.shape != energies.shape
            or density_.ndim not in (1, 2, 3)
            or not backend_
            or not isinstance(energy_unit, UnitDefinition)
        ):
            raise ValueError(
                "Gamma SCF orbitals, density, backend, or units are invalid."
            )
        self.total_energy = jnp.asarray(total_energy, dtype=energies.real.dtype).reshape(
            ()
        )
        self.free_energy = jnp.asarray(free_energy, dtype=energies.real.dtype).reshape(())
        self.orbital_energies = energies
        self.occupations = occupations_
        self.density = density_
        self.residual = jnp.asarray(residual, dtype=energies.real.dtype).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.backend = backend_
        self.energy_unit = energy_unit
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "gamma-scf-result",
                "backend": backend_,
                "plan": self.plan_id,
                "energy_unit": energy_unit.unit_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "total_energy": np.asarray(self.total_energy),
                        "free_energy": np.asarray(self.free_energy),
                        "orbital_energies": np.asarray(energies),
                        "occupations": np.asarray(occupations_),
                        "density": np.asarray(density_),
                        "residual": np.asarray(self.residual),
                    }
                ),
            }
        )


def _gamma_occupations(energies, electron_count, smearing):
    count = float(electron_count)
    if smearing == 0.0:
        occupied = int(round(0.5 * count))
        if abs(2.0 * occupied - count) > 1.0e-12 or occupied > energies.size:
            raise ValueError(
                "Zero-smearing Gamma SCF requires an even admissible electron count."
            )
        occupations = jnp.zeros_like(energies.real).at[:occupied].set(2.0)
        chemical = energies[max(occupied - 1, 0)].real
        return occupations, chemical, jnp.asarray(0.0, dtype=energies.real.dtype)
    lower = float(jnp.min(energies.real)) - 80.0 * smearing
    upper = float(jnp.max(energies.real)) + 80.0 * smearing
    for _ in range(180):
        chemical = 0.5 * (lower + upper)
        occupations = 2.0 / (1.0 + jnp.exp((energies.real - chemical) / smearing))
        if float(jnp.sum(occupations)) > count:
            upper = chemical
        else:
            lower = chemical
    chemical = jnp.asarray(0.5 * (lower + upper), dtype=energies.real.dtype)
    occupations = 2.0 / (1.0 + jnp.exp((energies.real - chemical) / smearing))
    probability = jnp.clip(0.5 * occupations, 1.0e-15, 1.0 - 1.0e-15)
    entropy = -2.0 * jnp.sum(
        probability * jnp.log(probability)
        + (1.0 - probability) * jnp.log(1.0 - probability)
    )
    return occupations, chemical, entropy


class GammaFFTDFPlan(StrictModule, NonTrainableState):
    cell_vectors: Array
    ionic_potential: Array
    electron_count: float = eqx.field(static=True)
    ionic_energy: float = eqx.field(static=True)
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_vectors: ArrayLike,
        ionic_potential: ArrayLike,
        electron_count: float,
        ionic_energy: float,
        energy_unit: UnitDefinition,
        /,
        *,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
        damping: float = 0.3,
        maximum_grid_points: int = 512,
    ):
        cell = jnp.asarray(cell_vectors)
        potential = jnp.asarray(ionic_potential, dtype=cell.dtype)
        electrons = float(electron_count)
        ionic = float(ionic_energy)
        smearing = float(smearing_energy)
        tolerance = float(convergence_tolerance)
        maximum = int(maximum_iterations)
        damping_ = float(damping)
        capacity = int(maximum_grid_points)
        if (
            cell.shape != (3, 3)
            or abs(float(np.linalg.det(np.asarray(cell)))) <= np.finfo(float).eps
            or potential.ndim != 3
            or potential.size > capacity
            or electrons <= 0.0
            or electrons > 2.0 * potential.size
            or any(
                not isfinite(value) for value in (ionic, smearing, tolerance, damping_)
            )
            or smearing < 0.0
            or tolerance <= 0.0
            or maximum <= 0
            or not 0.0 <= damping_ < 1.0
            or not isinstance(energy_unit, UnitDefinition)
        ):
            raise ValueError(
                "Gamma FFTDF cell, grid, electrons, or SCF policy is invalid."
            )
        self.cell_vectors = cell
        self.ionic_potential = potential
        self.electron_count = electrons
        self.ionic_energy = ionic
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_iterations = maximum
        self.damping = damping_
        self.maximum_grid_points = capacity
        self.energy_unit = energy_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gamma-fftdf-plan",
                "electron_count": electrons,
                "ionic_energy": ionic,
                "smearing_energy": smearing,
                "convergence_tolerance": tolerance,
                "maximum_iterations": maximum,
                "damping": damping_,
                "maximum_grid_points": capacity,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {"cell": np.asarray(cell), "ionic_potential": np.asarray(potential)}
                ),
            }
        )

    def evaluate(self, /) -> GammaSCFResult:
        shape = self.ionic_potential.shape
        size = int(self.ionic_potential.size)
        cell = self.cell_vectors
        determinant = jnp.sum(cell[0] * jnp.cross(cell[1], cell[2]))
        volume = jnp.abs(determinant)
        inverse = (
            jnp.stack(
                (
                    jnp.cross(cell[1], cell[2]),
                    jnp.cross(cell[2], cell[0]),
                    jnp.cross(cell[0], cell[1]),
                ),
                axis=1,
            )
            / determinant
        )
        integer_axes = tuple(jnp.fft.fftfreq(count) * count for count in shape)
        integer_grid = jnp.stack(jnp.meshgrid(*integer_axes, indexing="ij"), axis=-1)
        wavevectors = 2.0 * jnp.pi * contract("...i,ji->...j", integer_grid, inverse)
        squared = jnp.sum(wavevectors**2, axis=-1)
        fourier_indices = np.stack(np.unravel_index(np.arange(size), shape), axis=1)
        grid_indices = np.stack(np.unravel_index(np.arange(size), shape), axis=1)
        phase = (
            2.0
            * np.pi
            * sum(
                np.outer(grid_indices[:, axis], fourier_indices[:, axis]) / shape[axis]
                for axis in range(3)
            )
        )
        fourier = jnp.asarray(np.exp(-1.0j * phase) / np.sqrt(size))
        kinetic = jnp.conj(fourier.T) @ jnp.diag(0.5 * squared.reshape((-1,))) @ fourier
        grid_weight = volume / size
        density = jnp.full((size,), self.electron_count / volume, dtype=cell.dtype)
        previous_energy = jnp.asarray(jnp.inf, dtype=cell.dtype)
        residual = jnp.asarray(jnp.inf, dtype=cell.dtype)
        total = jnp.asarray(jnp.nan, dtype=cell.dtype)
        free = total
        energies = jnp.zeros((size,), dtype=cell.dtype)
        occupations = jnp.zeros_like(energies)
        converged = False
        completed = 0
        for iteration in range(self.maximum_iterations):
            density_grid = density.reshape(shape)
            density_fourier = jnp.fft.fftn(density_grid)
            coulomb_kernel = jnp.where(squared > 0.0, 4.0 * jnp.pi / squared, 0.0)
            hartree_potential = jnp.real(
                jnp.fft.ifftn(coulomb_kernel * density_fourier)
            ).reshape((-1,))
            safe_density = jnp.maximum(density, jnp.finfo(density.dtype).tiny ** 0.25)
            exchange_coefficient = -0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)
            exchange_energy_density = exchange_coefficient * safe_density ** (4.0 / 3.0)
            exchange_potential = (
                (4.0 / 3.0) * exchange_coefficient * safe_density ** (1.0 / 3.0)
            )
            local_potential = (
                self.ionic_potential.reshape((-1,))
                + hartree_potential
                + exchange_potential
            )
            hamiltonian = kinetic + jnp.diag(local_potential)
            energies, orbitals = _hermitian_eigh(hamiltonian)
            occupations, _, entropy = _gamma_occupations(
                energies, self.electron_count, self.smearing_energy
            )
            proposed = (
                jnp.real(
                    contract("n,gn,gn->g", occupations, orbitals, jnp.conj(orbitals))
                )
                / grid_weight
            )
            mixed = self.damping * density + (1.0 - self.damping) * proposed
            one_body = jnp.real(
                contract(
                    "n,gn,gh,hn->",
                    occupations,
                    jnp.conj(orbitals),
                    kinetic + jnp.diag(self.ionic_potential.reshape((-1,))),
                    orbitals,
                )
            )
            hartree_energy = 0.5 * grid_weight * jnp.sum(mixed * hartree_potential)
            exchange_energy = grid_weight * jnp.sum(exchange_energy_density)
            total = one_body + hartree_energy + exchange_energy + self.ionic_energy
            free = total - self.smearing_energy * entropy
            residual = jnp.maximum(
                jnp.max(jnp.abs(mixed - density), initial=0.0),
                jnp.abs(total - previous_energy),
            )
            density = mixed
            previous_energy = total
            completed = iteration + 1
            if bool(residual <= self.convergence_tolerance):
                converged = True
                break
        return GammaSCFResult(
            total,
            free,
            energies,
            occupations,
            density.reshape(shape),
            residual,
            completed,
            converged,
            "fftdf-lda-x",
            self.energy_unit,
            self.plan_id,
        )


DensityFunctionalMatrix = Callable[[Array], tuple[Array, Array]]


class GammaGDFPlan(StrictModule, NonTrainableState):
    one_body: Array
    overlap: Array
    factors: FactorizedERITensor
    electron_count: float = eqx.field(static=True)
    nuclear_energy: float = eqx.field(static=True)
    exact_exchange_fraction: float = eqx.field(static=True)
    density_functional: DensityFunctionalMatrix | None = eqx.field(static=True)
    density_functional_id: str | None = eqx.field(static=True)
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        one_body: ArrayLike,
        overlap: ArrayLike,
        factors: FactorizedERITensor,
        electron_count: float,
        nuclear_energy: float,
        energy_unit: UnitDefinition,
        /,
        *,
        exact_exchange_fraction: float = 1.0,
        density_functional: DensityFunctionalMatrix | None = None,
        density_functional_id: str | None = None,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
        damping: float = 0.2,
    ):
        one = jnp.asarray(one_body)
        overlap_ = jnp.asarray(overlap, dtype=one.dtype)
        electrons = float(electron_count)
        nuclear = float(nuclear_energy)
        exchange = float(exact_exchange_fraction)
        functional_id = (
            None if density_functional_id is None else str(density_functional_id).strip()
        )
        smearing = float(smearing_energy)
        tolerance = float(convergence_tolerance)
        maximum = int(maximum_iterations)
        damping_ = float(damping)
        count = one.shape[0] if one.ndim == 2 else -1
        if (
            one.shape != (count, count)
            or overlap_.shape != one.shape
            or not isinstance(factors, FactorizedERITensor)
            or factors.orbital_count != count
            or electrons <= 0.0
            or electrons > 2.0 * count
            or not 0.0 <= exchange <= 1.0
            or (density_functional is None) != (functional_id is None)
            or density_functional is not None
            and not callable(density_functional)
            or any(
                not isfinite(value) for value in (nuclear, smearing, tolerance, damping_)
            )
            or smearing < 0.0
            or tolerance <= 0.0
            or maximum <= 0
            or not 0.0 <= damping_ < 1.0
            or not isinstance(energy_unit, UnitDefinition)
        ):
            raise ValueError(
                "Gamma GDF integrals, electrons, functional, or SCF policy is invalid."
            )
        self.one_body = one
        self.overlap = overlap_
        self.factors = factors
        self.electron_count = electrons
        self.nuclear_energy = nuclear
        self.exact_exchange_fraction = exchange
        self.density_functional = density_functional
        self.density_functional_id = functional_id
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_iterations = maximum
        self.damping = damping_
        self.energy_unit = energy_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gamma-gdf-plan",
                "factors": factors.tensor_id,
                "electron_count": electrons,
                "nuclear_energy": nuclear,
                "exact_exchange_fraction": exchange,
                "density_functional": functional_id,
                "smearing_energy": smearing,
                "convergence_tolerance": tolerance,
                "maximum_iterations": maximum,
                "damping": damping_,
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {"one_body": np.asarray(one), "overlap": np.asarray(overlap_)}
                ),
            }
        )

    def evaluate(self, /) -> GammaSCFResult:
        overlap_values, overlap_vectors = _hermitian_eigh(self.overlap)
        if bool(jnp.min(overlap_values) <= 1.0e-10):
            raise ValueError("Gamma GDF overlap is rank deficient.")
        orthogonalizer = (
            overlap_vectors @ jnp.diag(overlap_values**-0.5) @ jnp.conj(overlap_vectors.T)
        )
        energies, transformed = _hermitian_eigh(
            jnp.conj(orthogonalizer.T) @ self.one_body @ orthogonalizer
        )
        coefficients = orthogonalizer @ transformed
        occupations, _, entropy = _gamma_occupations(
            energies, self.electron_count, self.smearing_energy
        )
        density = contract(
            "pi,i,qi->pq", coefficients, occupations, jnp.conj(coefficients)
        )
        previous_energy = jnp.asarray(jnp.inf, dtype=one_real_dtype(self.one_body))
        residual = jnp.asarray(jnp.inf, dtype=previous_energy.dtype)
        total = jnp.asarray(jnp.nan, dtype=previous_energy.dtype)
        free = total
        converged = False
        completed = 0
        fock = self.one_body
        for iteration in range(self.maximum_iterations):
            coulomb = self.factors.coulomb(density)
            exchange = self.factors.exchange(density)
            if self.density_functional is None:
                functional_energy = jnp.asarray(0.0, dtype=previous_energy.dtype)
                functional_potential = jnp.zeros_like(self.one_body)
            else:
                functional_energy, functional_potential = self.density_functional(density)
            fock = (
                self.one_body
                + coulomb
                - 0.5 * self.exact_exchange_fraction * exchange
                + functional_potential
            )
            energies, transformed = _hermitian_eigh(
                jnp.conj(orthogonalizer.T) @ fock @ orthogonalizer
            )
            coefficients = orthogonalizer @ transformed
            occupations, _, entropy = _gamma_occupations(
                energies, self.electron_count, self.smearing_energy
            )
            proposed = contract(
                "pi,i,qi->pq", coefficients, occupations, jnp.conj(coefficients)
            )
            mixed = self.damping * density + (1.0 - self.damping) * proposed
            coulomb_new = self.factors.coulomb(mixed)
            exchange_new = self.factors.exchange(mixed)
            if self.density_functional is not None:
                functional_energy, _ = self.density_functional(mixed)
            total = jnp.real(
                contract("ab,ab->", mixed, self.one_body)
                + 0.5 * contract("ab,ab->", mixed, coulomb_new)
                - 0.25
                * self.exact_exchange_fraction
                * contract("ab,ab->", mixed, exchange_new)
                + functional_energy
                + self.nuclear_energy
            )
            free = total - self.smearing_energy * entropy
            residual = jnp.maximum(
                jnp.max(jnp.abs(mixed - density), initial=0.0),
                jnp.abs(total - previous_energy),
            )
            density = mixed
            previous_energy = total
            completed = iteration + 1
            if bool(residual <= self.convergence_tolerance):
                converged = True
                break
        return GammaSCFResult(
            total,
            free,
            energies,
            occupations,
            density,
            residual,
            completed,
            converged,
            "gdf-hf" if self.density_functional is None else "gdf-hybrid",
            self.energy_unit,
            self.plan_id,
        )


def one_real_dtype(value):
    return jnp.asarray(value).real.dtype


__all__ = ["GammaFFTDFPlan", "GammaGDFPlan", "GammaSCFResult"]
