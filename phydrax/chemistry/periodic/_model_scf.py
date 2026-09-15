#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic electronic sectors, k meshes, provider contracts, and native AO SCF."""

from __future__ import annotations

import abc
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ...units import UnitDefinition


class PeriodicElectronicSectorPlan(StrictModule, NonTrainableState):
    electron_count: float = eqx.field(static=True)
    spin_magnetization: float = eqx.field(static=True)
    charge_per_cell: float = eqx.field(static=True)
    background_policy: str = eqx.field(static=True)
    sector_id: str = eqx.field(static=True)

    def __init__(
        self,
        electron_count: float,
        /,
        *,
        spin_magnetization: float = 0.0,
        charge_per_cell: float = 0.0,
        background_policy: str = "forbid-charged-cell",
    ):
        electrons = float(electron_count)
        magnetization = float(spin_magnetization)
        charge = float(charge_per_cell)
        policy = str(background_policy).strip()
        if any(not isfinite(value) for value in (electrons, magnetization, charge)):
            raise ValueError(
                "Periodic electron, magnetization, and charge values must be finite."
            )
        if electrons <= 0.0 or abs(magnetization) > electrons or not policy:
            raise ValueError("Periodic electronic sector is invalid.")
        if charge != 0.0 and policy == "forbid-charged-cell":
            raise ValueError(
                "Charged periodic cells require an explicit background policy."
            )
        self.electron_count = electrons
        self.spin_magnetization = magnetization
        self.charge_per_cell = charge
        self.background_policy = policy
        self.sector_id = canonical_fingerprint(
            {
                "kind": "periodic-electronic-sector",
                "electron_count": electrons,
                "spin_magnetization": magnetization,
                "charge_per_cell": charge,
                "background_policy": policy,
            }
        )


class KPointMeshPlan(StrictModule, NonTrainableState):
    fractional_points: Array
    weights: Array
    mesh_shape: tuple[int, int, int] = eqx.field(static=True)
    shift: tuple[float, float, float] = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        fractional_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        mesh_shape: tuple[int, int, int],
        shift: tuple[float, float, float],
    ):
        points = jnp.asarray(fractional_points)
        weights_ = jnp.asarray(weights, dtype=points.dtype)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or weights_.shape != (points.shape[0],)
        ):
            raise ValueError("k points and weights must have shapes (K,3) and (K,).")
        if (
            np.any(~np.isfinite(np.asarray(points)))
            or np.any(~np.isfinite(np.asarray(weights_)))
            or np.any(np.asarray(weights_) <= 0.0)
        ):
            raise ValueError("k points must be finite and weights positive finite.")
        if not np.isclose(float(np.sum(np.asarray(weights_))), 1.0, atol=1.0e-12):
            raise ValueError("k-point weights must sum to one.")
        shape = tuple(int(value) for value in mesh_shape)
        shift_ = tuple(float(value) for value in shift)
        if len(shape) != 3 or any(value <= 0 for value in shape) or len(shift_) != 3:
            raise ValueError("k mesh shape/shift are invalid.")
        self.fractional_points = points
        self.weights = weights_
        self.mesh_shape = shape
        self.shift = shift_
        self.mesh_id = canonical_fingerprint(
            {
                "kind": "k-point-mesh",
                "mesh_shape": list(shape),
                "shift": list(shift_),
                "arrays": array_tree_fingerprint(
                    {"points": np.asarray(points), "weights": np.asarray(weights_)}
                ),
            }
        )

    @classmethod
    def monkhorst_pack(
        cls,
        mesh_shape: tuple[int, int, int],
        /,
        *,
        shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dtype=np.float64,
    ) -> KPointMeshPlan:
        shape = tuple(int(value) for value in mesh_shape)
        shift_ = tuple(float(value) for value in shift)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("Monkhorst--Pack mesh dimensions must be positive.")
        axes = [
            (np.arange(count, dtype=dtype) + 0.5) / count - 0.5 + shift_[axis] / count
            for axis, count in enumerate(shape)
        ]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
        weights = np.full((mesh.shape[0],), 1.0 / mesh.shape[0], dtype=dtype)
        return cls(mesh, weights, mesh_shape=shape, shift=shift_)


class PeriodicAOModelPlan(StrictModule, NonTrainableState):
    translations: Array
    hamiltonian_blocks: Array
    overlap_blocks: Array
    onsite_hubbard: Array
    reference_populations: Array
    ionic_energy: Array
    energy_unit: UnitDefinition
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        translations: ArrayLike,
        hamiltonian_blocks: ArrayLike,
        overlap_blocks: ArrayLike,
        onsite_hubbard: ArrayLike,
        reference_populations: ArrayLike,
        ionic_energy: ArrayLike,
        energy_unit: UnitDefinition,
        /,
    ):
        translations_ = jnp.asarray(translations, dtype=jnp.int32)
        hamiltonian = jnp.asarray(hamiltonian_blocks)
        overlap = jnp.asarray(overlap_blocks, dtype=hamiltonian.dtype)
        if translations_.ndim != 2 or translations_.shape[1] != 3:
            raise ValueError("Periodic translations must have shape (R,3).")
        if (
            hamiltonian.ndim != 3
            or hamiltonian.shape != overlap.shape
            or (hamiltonian.shape[0] != translations_.shape[0])
            or hamiltonian.shape[1] != hamiltonian.shape[2]
        ):
            raise ValueError(
                "Periodic Hamiltonian/overlap blocks must have shape (R,A,A)."
            )
        orbital_count = int(hamiltonian.shape[1])
        hubbard = jnp.asarray(onsite_hubbard, dtype=hamiltonian.dtype)
        reference = jnp.asarray(reference_populations, dtype=hamiltonian.dtype)
        if hubbard.shape != (orbital_count,) or reference.shape != (orbital_count,):
            raise ValueError(
                "Onsite Hubbard and reference populations require one value per AO."
            )
        if np.any(np.asarray(hubbard) < 0.0) or np.any(~np.isfinite(np.asarray(hubbard))):
            raise ValueError("Onsite Hubbard values must be finite and non-negative.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        zero_rows = np.flatnonzero(np.all(np.asarray(translations_) == 0, axis=1))
        if zero_rows.size != 1:
            raise ValueError(
                "Periodic AO model requires exactly one zero translation block."
            )
        self.translations = translations_
        self.hamiltonian_blocks = hamiltonian
        self.overlap_blocks = overlap
        self.onsite_hubbard = hubbard
        self.reference_populations = reference
        self.ionic_energy = jnp.asarray(ionic_energy, dtype=hamiltonian.dtype).reshape(())
        self.energy_unit = energy_unit
        self.model_id = canonical_fingerprint(
            {
                "kind": "periodic-ao-model",
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "translations": np.asarray(translations_),
                        "hamiltonian": np.asarray(hamiltonian),
                        "overlap": np.asarray(overlap),
                        "onsite_hubbard": np.asarray(hubbard),
                        "reference_populations": np.asarray(reference),
                        "ionic_energy": np.asarray(self.ionic_energy),
                    }
                ),
            }
        )

    @property
    def orbital_count(self) -> int:
        return int(self.hamiltonian_blocks.shape[1])

    def bloch_matrices(self, mesh: KPointMeshPlan, /) -> tuple[Array, Array]:
        phase = jnp.exp(
            2.0j
            * jnp.pi
            * contract("kd,rd->kr", mesh.fractional_points, self.translations)
        )
        hamiltonian = contract("kr,rab->kab", phase, self.hamiltonian_blocks)
        overlap = contract("kr,rab->kab", phase, self.overlap_blocks)
        hamiltonian_residual = float(
            jnp.max(jnp.abs(hamiltonian - jnp.conj(jnp.swapaxes(hamiltonian, -1, -2))))
        )
        overlap_residual = float(
            jnp.max(jnp.abs(overlap - jnp.conj(jnp.swapaxes(overlap, -1, -2))))
        )
        if max(hamiltonian_residual, overlap_residual) > 1.0e-10:
            raise ValueError(
                "Periodic real-space blocks do not produce Hermitian Bloch matrices."
            )
        return hamiltonian, overlap


class PeriodicSCFResult(StrictModule, NonTrainableState):
    energy: Array
    free_energy: Array
    entropy: Array
    chemical_potential: Array
    orbital_energies: Array
    occupations: Array
    density_matrices: Array
    populations: Array
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
        energy: ArrayLike,
        free_energy: ArrayLike,
        entropy: ArrayLike,
        chemical_potential: ArrayLike,
        orbital_energies: ArrayLike,
        occupations: ArrayLike,
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
        model_id: str,
    ):
        orbital = jnp.asarray(orbital_energies)
        occupation = jnp.asarray(occupations, dtype=orbital.real.dtype)
        density = jnp.asarray(density_matrices)
        population = jnp.asarray(populations, dtype=occupation.dtype)
        if orbital.ndim != 2 or occupation.shape != orbital.shape:
            raise ValueError(
                "Periodic orbital energies and occupations must align (k,band)."
            )
        if density.shape != (orbital.shape[0], orbital.shape[1], orbital.shape[1]):
            raise ValueError("Periodic density matrices must align with k-point AO axes.")
        self.energy = jnp.asarray(energy, dtype=occupation.dtype).reshape(())
        self.free_energy = jnp.asarray(free_energy, dtype=occupation.dtype).reshape(())
        self.entropy = jnp.asarray(entropy, dtype=occupation.dtype).reshape(())
        self.chemical_potential = jnp.asarray(
            chemical_potential, dtype=occupation.dtype
        ).reshape(())
        self.orbital_energies = orbital
        self.occupations = occupation
        self.density_matrices = density
        self.populations = population
        self.residual = jnp.asarray(residual, dtype=occupation.dtype).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.energy_unit = energy_unit
        self.cell_id = cell_id
        self.mesh_id = mesh_id
        self.sector_id = sector_id
        self.model_id = model_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-scf-result",
                "cell": cell_id,
                "mesh": mesh_id,
                "sector": sector_id,
                "model": model_id,
                "energy_unit": energy_unit.unit_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(self.energy),
                        "free_energy": np.asarray(self.free_energy),
                        "entropy": np.asarray(self.entropy),
                        "chemical_potential": np.asarray(self.chemical_potential),
                        "orbital_energies": np.asarray(orbital),
                        "occupations": np.asarray(occupation),
                        "density_matrices": np.asarray(density),
                        "populations": np.asarray(population),
                        "residual": np.asarray(self.residual),
                        "iterations": np.asarray(self.iterations),
                    }
                ),
            }
        )


class AbstractPeriodicElectronicProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(self) -> PeriodicSCFResult:
        raise NotImplementedError


PeriodicEvaluator = Callable[[], PeriodicSCFResult]


class CallablePeriodicElectronicProvider(AbstractPeriodicElectronicProvider):
    evaluator: PeriodicEvaluator
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: PeriodicEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(self) -> PeriodicSCFResult:
        result = self.evaluator()
        if not isinstance(result, PeriodicSCFResult):
            raise TypeError("Periodic provider returned an invalid result.")
        return result


def _hermitian_eigh(matrix: Array, /) -> tuple[Array, Array]:
    result = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                matrix,
                properties=OperatorProperties(
                    self_adjoint=True, evidence={"self_adjoint": "construction"}
                ),
            )
        ),
        policy=EigenSolvePolicy(
            DenseEigh(), count=int(matrix.shape[0]), which="smallest-algebraic"
        ),
    )
    if not bool(result.successful):
        raise RuntimeError("Periodic Hermitian eigensolve failed.")
    return result.eigenvalues, result.eigenvectors


class NativePeriodicSCFPlan(AbstractPeriodicElectronicProvider):
    cell: PeriodicCell
    mesh: KPointMeshPlan
    sector: PeriodicElectronicSectorPlan
    model: PeriodicAOModelPlan
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        mesh: KPointMeshPlan,
        sector: PeriodicElectronicSectorPlan,
        model: PeriodicAOModelPlan,
        /,
        *,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-10,
        maximum_iterations: int = 256,
        damping: float = 0.25,
    ):
        if not isinstance(cell, PeriodicCell) or not all(cell.periodic_axes):
            raise ValueError("Native periodic SCF requires a fully periodic 3D cell.")
        if (
            not isinstance(mesh, KPointMeshPlan)
            or not isinstance(sector, PeriodicElectronicSectorPlan)
            or not isinstance(model, PeriodicAOModelPlan)
        ):
            raise TypeError("Periodic mesh, sector, and model plans are required.")
        if sector.charge_per_cell != 0.0:
            raise ValueError("Native periodic SCF currently supports neutral cells only.")
        if sector.spin_magnetization != 0.0:
            raise ValueError(
                "Native periodic SCF currently supports zero spin magnetization."
            )
        if sector.electron_count > 2.0 * model.orbital_count:
            raise ValueError("Periodic electron count exceeds AO occupation capacity.")
        if not np.isclose(
            float(np.sum(np.asarray(model.reference_populations))),
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
            any(not isfinite(value) or value < 0.0 for value in (smearing, tolerance))
            or tolerance <= 0.0
        ):
            raise ValueError("Periodic smearing/convergence values are invalid.")
        if iterations <= 0 or not isfinite(damping_) or damping_ < 0.0 or damping_ >= 1.0:
            raise ValueError("Periodic iteration count/damping is invalid.")
        self.cell = cell
        self.mesh = mesh
        self.sector = sector
        self.model = model
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
                "model": model.model_id,
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
            if abs(2.0 * occupied - target) > 1.0e-12:
                raise ValueError(
                    "Zero-smearing periodic SCF requires an even electron count."
                )
            if occupied < 1:
                raise ValueError(
                    "Zero-smearing periodic SCF requires at least one occupied band."
                )
            occupations = jnp.zeros_like(energies.real).at[:, :occupied].set(2.0)
            highest_occupied = jnp.max(energies[:, occupied - 1].real)
            if occupied < energies.shape[1]:
                lowest_unoccupied = jnp.min(energies[:, occupied].real)
                gap_tolerance = 1.0e-12 * max(
                    float(jnp.max(jnp.abs(energies.real))),
                    1.0,
                )
                if float(highest_occupied) >= float(lowest_unoccupied) - gap_tolerance:
                    raise ValueError(
                        "Zero-smearing periodic SCF requires an insulating band gap; "
                        "use finite Fermi--Dirac smearing for overlapping bands."
                    )
                chemical_potential = 0.5 * (highest_occupied + lowest_unoccupied)
            else:
                chemical_potential = highest_occupied
            return (
                occupations,
                chemical_potential,
                jnp.asarray(0.0, dtype=energies.real.dtype),
            )
        lower = float(jnp.min(energies.real)) - 50.0 * self.smearing_energy
        upper = float(jnp.max(energies.real)) + 50.0 * self.smearing_energy
        weights = self.mesh.weights[:, None]
        for _ in range(160):
            chemical = 0.5 * (lower + upper)
            occupation = 2.0 / (
                1.0 + jnp.exp((energies.real - chemical) / self.smearing_energy)
            )
            count = float(jnp.sum(weights * occupation))
            if count > target:
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

    def evaluate(self) -> PeriodicSCFResult:
        hamiltonian, overlap = self.model.bloch_matrices(self.mesh)
        k_count = int(self.mesh.fractional_points.shape[0])
        orbital_count = self.model.orbital_count
        populations = jnp.asarray(self.model.reference_populations)
        previous_energy = jnp.asarray(jnp.inf, dtype=populations.dtype)
        residual = jnp.asarray(jnp.inf, dtype=populations.dtype)
        energies = jnp.zeros((k_count, orbital_count), dtype=populations.dtype)
        coefficients = jnp.zeros(
            (k_count, orbital_count, orbital_count), dtype=hamiltonian.dtype
        )
        occupations = jnp.zeros_like(energies)
        density = jnp.zeros_like(coefficients)
        chemical = jnp.asarray(0.0, dtype=populations.dtype)
        entropy = jnp.asarray(0.0, dtype=populations.dtype)
        converged = False
        completed = 0
        total_energy = jnp.asarray(jnp.nan, dtype=populations.dtype)
        for iteration in range(self.maximum_iterations):
            potential = jnp.diag(
                self.model.onsite_hubbard
                * (populations - self.model.reference_populations)
            )
            for kpoint in range(k_count):
                overlap_values, overlap_vectors = _hermitian_eigh(overlap[kpoint])
                if bool(jnp.min(overlap_values) <= 1.0e-10):
                    raise ValueError(
                        "Periodic overlap matrix is numerically rank deficient."
                    )
                orthogonalizer = (
                    overlap_vectors
                    @ jnp.diag(overlap_values**-0.5)
                    @ jnp.conj(overlap_vectors.T)
                )
                values, transformed = _hermitian_eigh(
                    jnp.conj(orthogonalizer.T)
                    @ (hamiltonian[kpoint] + potential)
                    @ orthogonalizer
                )
                energies = energies.at[kpoint].set(values.real)
                coefficients = coefficients.at[kpoint].set(orthogonalizer @ transformed)
            occupations, chemical, entropy = self._occupations(energies)
            density = contract(
                "kn,kan,kbn->kab",
                occupations,
                coefficients,
                jnp.conj(coefficients),
            )
            proposed_populations = jnp.real(
                contract("k,kab,kba->a", self.mesh.weights, density, overlap)
            )
            mixed = (
                1.0 - self.damping
            ) * proposed_populations + self.damping * populations
            one_electron = jnp.real(
                contract("k,kab,kba->", self.mesh.weights, density, hamiltonian)
            )
            delta = mixed - self.model.reference_populations
            interaction = 0.5 * jnp.sum(self.model.onsite_hubbard * delta * delta)
            total_energy = one_electron + interaction + self.model.ionic_energy
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
        free_energy = total_energy - self.smearing_energy * entropy
        return PeriodicSCFResult(
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
            cell_id=self.cell.cell_id,
            mesh_id=self.mesh.mesh_id,
            sector_id=self.sector.sector_id,
            model_id=self.model.model_id,
        )


__all__ = [
    "AbstractPeriodicElectronicProvider",
    "CallablePeriodicElectronicProvider",
    "KPointMeshPlan",
    "NativePeriodicSCFPlan",
    "PeriodicAOModelPlan",
    "PeriodicElectronicSectorPlan",
    "PeriodicSCFResult",
]
