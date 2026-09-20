#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Restricted and collinear periodic Hubbard SCF on canonical H/S pencils."""

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import ReciprocalMeshPlan
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)
from ...units import UnitDefinition
from .._result import ElectronicEnergyLedger
from .._state import PeriodicElectronicSectorPlan
from ._orbital_model import PeriodicHubbardMeanFieldPlan, PreparedPeriodicOrbitalPencil


SpinReferenceKind = Literal["restricted", "collinear"]


class SpinPeriodicSCFEvidence(StrictModule, NonTrainableState):
    """Independent fixed-point, spectral, metric-count, and thermodynamic evidence."""

    energy_residual: Array
    density_residual: Array
    commutator_residual: Array
    eigenpair_residual: Array
    electron_count_residual: Array
    spin_count_residual: Array
    free_energy_residual: Array
    successful: Array
    tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_residual: ArrayLike,
        density_residual: ArrayLike,
        commutator_residual: ArrayLike,
        eigenpair_residual: ArrayLike,
        electron_count_residual: ArrayLike,
        spin_count_residual: ArrayLike,
        free_energy_residual: ArrayLike,
        successful: ArrayLike,
        tolerance: float,
        /,
    ):
        values = jnp.asarray(
            (
                energy_residual,
                density_residual,
                commutator_residual,
                eigenpair_residual,
                electron_count_residual,
                spin_count_residual,
                free_energy_residual,
            )
        ).reshape((7,))
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Spin SCF evidence tolerance must be finite and positive.")
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(values >= 0.0)
        admitted = (
            jnp.asarray(successful, dtype=jnp.bool_).reshape(())
            & finite
            & (values[0] <= tolerance_)
            & (values[1] <= tolerance_)
            & (values[2] <= 10.0 * tolerance_)
            & (values[3] <= 10.0 * tolerance_)
            & (values[4] <= 10.0 * tolerance_)
            & (values[5] <= 10.0 * tolerance_)
            & (values[6] <= 10.0 * tolerance_)
        )
        (
            self.energy_residual,
            self.density_residual,
            self.commutator_residual,
            self.eigenpair_residual,
            self.electron_count_residual,
            self.spin_count_residual,
            self.free_energy_residual,
        ) = tuple(values)
        self.successful = admitted
        self.tolerance = tolerance_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "spin-periodic-scf-evidence",
                "tolerance": tolerance_.hex(),
                "successful": bool(admitted),
                "residuals": array_tree_fingerprint(np.asarray(values)),
            }
        )


class SpinPeriodicSCFResult(StrictModule, NonTrainableState):
    energy: Array
    free_energy: Array
    entropy: Array
    chemical_potentials: Array
    orbital_energies: Array
    occupations: Array
    coefficients: Array
    density_matrices: Array
    spin_populations: Array
    energy_ledger: ElectronicEnergyLedger
    evidence: SpinPeriodicSCFEvidence
    iterations: Array
    successful: Array
    energy_unit: UnitDefinition
    reference_kind: SpinReferenceKind = eqx.field(static=True)
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
        chemical_potentials: ArrayLike,
        orbital_energies: ArrayLike,
        occupations: ArrayLike,
        coefficients: ArrayLike,
        density_matrices: ArrayLike,
        spin_populations: ArrayLike,
        energy_ledger: ElectronicEnergyLedger,
        evidence: SpinPeriodicSCFEvidence,
        iterations: int,
        energy_unit: UnitDefinition,
        /,
        *,
        reference_kind: SpinReferenceKind,
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
        populations = jnp.asarray(spin_populations, dtype=orbital.real.dtype)
        if (
            orbital.ndim != 3
            or orbital.shape[0] != 2
            or occupation.shape != orbital.shape
            or coefficient.shape
            != (2, orbital.shape[1], orbital.shape[2], orbital.shape[2])
            or density.shape != coefficient.shape
            or populations.shape != (2, orbital.shape[2])
        ):
            raise ValueError("Spin periodic SCF arrays do not align.")
        if not isinstance(energy_ledger, ElectronicEnergyLedger):
            raise TypeError("energy_ledger must be ElectronicEnergyLedger.")
        if not isinstance(evidence, SpinPeriodicSCFEvidence):
            raise TypeError("evidence must be SpinPeriodicSCFEvidence.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        if energy_ledger.energy_unit != energy_unit:
            raise ValueError("Spin SCF ledger and result energy units differ.")
        if reference_kind not in ("restricted", "collinear"):
            raise ValueError("Spin reference_kind must be restricted or collinear.")
        iterations_ = int(iterations)
        if iterations_ <= 0:
            raise ValueError("Spin SCF result must report at least one iteration.")
        energy_ = jnp.asarray(energy, dtype=orbital.real.dtype).reshape(())
        free_ = jnp.asarray(free_energy, dtype=orbital.real.dtype).reshape(())
        entropy_ = jnp.asarray(entropy, dtype=orbital.real.dtype).reshape(())
        chemical = jnp.asarray(chemical_potentials, dtype=orbital.real.dtype).reshape(
            (2,)
        )
        identifiers = tuple(
            str(value).strip()
            for value in (cell_id, mesh_id, sector_id, pencil_id, mean_field_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Spin SCF identities must be non-empty.")
        finite = all(
            bool(jnp.all(jnp.isfinite(value)))
            for value in (
                energy_,
                free_,
                entropy_,
                chemical,
                orbital,
                occupation,
                coefficient,
                density,
                populations,
            )
        )
        self.energy = energy_
        self.free_energy = free_
        self.entropy = entropy_
        self.chemical_potentials = chemical
        self.orbital_energies = orbital
        self.occupations = occupation
        self.coefficients = coefficient
        self.density_matrices = density
        self.spin_populations = populations
        self.energy_ledger = energy_ledger
        self.evidence = evidence
        self.iterations = jnp.asarray(iterations_, dtype=jnp.int32)
        self.successful = evidence.successful & finite
        self.energy_unit = energy_unit
        self.reference_kind = reference_kind
        (
            self.cell_id,
            self.mesh_id,
            self.sector_id,
            self.pencil_id,
            self.mean_field_id,
        ) = identifiers
        self.result_id = canonical_fingerprint(
            {
                "kind": "spin-periodic-scf-result",
                "reference": reference_kind,
                "cell": self.cell_id,
                "mesh": self.mesh_id,
                "sector": self.sector_id,
                "pencil": self.pencil_id,
                "mean_field": self.mean_field_id,
                "ledger": energy_ledger.ledger_id,
                "evidence": evidence.evidence_id,
                "iterations": iterations_,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energy": np.asarray(energy_),
                        "free_energy": np.asarray(free_),
                        "entropy": np.asarray(entropy_),
                        "chemical_potentials": np.asarray(chemical),
                        "orbital_energies": np.asarray(orbital),
                        "occupations": np.asarray(occupation),
                        "coefficients": np.asarray(coefficient),
                        "density_matrices": np.asarray(density),
                        "spin_populations": np.asarray(populations),
                    }
                ),
            }
        )


def _spin_occupations(
    energies: Array,
    target: float,
    weights: Array,
    smearing: float,
    /,
) -> tuple[Array, Array, Array]:
    if smearing == 0.0:
        occupied = int(round(target))
        if (
            abs(occupied - target) > 1.0e-12
            or occupied < 0
            or occupied > energies.shape[1]
        ):
            raise ValueError(
                "Zero-smearing spin occupations require an admissible integer band count."
            )
        occupations = jnp.zeros_like(energies.real)
        if occupied:
            occupations = occupations.at[:, :occupied].set(1.0)
            highest = jnp.max(energies[:, occupied - 1].real)
        else:
            highest = jnp.min(energies.real) - 1.0
        if occupied < energies.shape[1]:
            lowest = jnp.min(energies[:, occupied].real)
            scale = jnp.maximum(jnp.max(jnp.abs(energies.real)), 1.0)
            if occupied and float(highest) >= float(lowest - 1.0e-12 * scale):
                raise ValueError(
                    "Zero-smearing spin SCF requires an insulating band gap."
                )
            chemical = lowest - 1.0 if not occupied else 0.5 * (highest + lowest)
        else:
            chemical = highest
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
    mesh: ReciprocalMeshPlan
    sector: PeriodicElectronicSectorPlan
    pencil: PreparedPeriodicOrbitalPencil
    mean_field: PeriodicHubbardMeanFieldPlan
    reference_spin_populations: Array
    reference_kind: SpinReferenceKind = eqx.field(static=True)
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: ReciprocalMeshPlan,
        sector: PeriodicElectronicSectorPlan,
        pencil: PreparedPeriodicOrbitalPencil,
        mean_field: PeriodicHubbardMeanFieldPlan,
        /,
        *,
        reference_kind: SpinReferenceKind = "collinear",
        reference_spin_populations: ArrayLike | None = None,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-10,
        maximum_iterations: int = 256,
        damping: float = 0.25,
    ):
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be ReciprocalMeshPlan.")
        if not isinstance(sector, PeriodicElectronicSectorPlan):
            raise TypeError("sector must be PeriodicElectronicSectorPlan.")
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil):
            raise TypeError("pencil must be PreparedPeriodicOrbitalPencil.")
        if not isinstance(mean_field, PeriodicHubbardMeanFieldPlan):
            raise TypeError("mean_field must be PeriodicHubbardMeanFieldPlan.")
        mesh.require_cell(pencil.plan.basis.cell)
        if mean_field.basis_id != pencil.plan.basis.basis_id:
            raise ValueError("Spin SCF mean field belongs to a different orbital basis.")
        if mean_field.energy_unit != pencil.plan.energy_unit:
            raise ValueError("Spin SCF pencil and mean-field energy units differ.")
        if sector.charge_per_cell != 0.0:
            raise ValueError("Spin periodic SCF requires a neutral electronic sector.")
        if reference_kind not in ("restricted", "collinear"):
            raise ValueError("reference_kind must be restricted or collinear.")
        if reference_kind == "restricted" and sector.spin_magnetization != 0.0:
            raise ValueError("Restricted periodic SCF requires zero spin magnetization.")
        orbital_count = pencil.plan.basis.orbital_count
        alpha_count = 0.5 * (sector.electron_count + sector.spin_magnetization)
        beta_count = 0.5 * (sector.electron_count - sector.spin_magnetization)
        if (
            min(alpha_count, beta_count) < 0.0
            or max(alpha_count, beta_count) > orbital_count
        ):
            raise ValueError("Spin populations exceed periodic orbital capacity.")
        total_reference = np.asarray(mean_field.reference_populations)
        if not np.isclose(
            float(np.sum(total_reference)), sector.electron_count, atol=1.0e-10
        ):
            raise ValueError(
                "Mean-field reference populations must sum to electron count."
            )
        reference = (
            np.stack(
                (
                    total_reference * alpha_count / sector.electron_count,
                    total_reference * beta_count / sector.electron_count,
                )
            )
            if reference_spin_populations is None
            else np.asarray(reference_spin_populations)
        )
        if (
            reference.shape != (2, orbital_count)
            or np.any(~np.isfinite(reference))
            or np.any(reference < 0.0)
            or not np.isclose(float(np.sum(reference[0])), alpha_count, atol=1.0e-10)
            or not np.isclose(float(np.sum(reference[1])), beta_count, atol=1.0e-10)
        ):
            raise ValueError("Reference spin populations must match sector populations.")
        if reference_kind == "restricted" and not np.allclose(
            reference[0], reference[1], rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("Restricted reference populations must be spin symmetric.")
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
                "Spin periodic SCF smearing, tolerance, iterations, or damping is invalid."
            )
        self.mesh = mesh
        self.sector = sector
        self.pencil = pencil
        self.mean_field = mean_field
        self.reference_spin_populations = jnp.asarray(reference)
        self.reference_kind = reference_kind
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_iterations = maximum
        self.damping = damping_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spin-periodic-scf-plan",
                "cell": mesh.cell_id,
                "mesh": mesh.mesh_id,
                "sector": sector.sector_id,
                "pencil": pencil.prepared_id,
                "mean_field": mean_field.plan_id,
                "reference_kind": reference_kind,
                "smearing_energy": smearing.hex(),
                "convergence_tolerance": tolerance.hex(),
                "maximum_iterations": maximum,
                "damping": damping_.hex(),
                "reference_spin_populations": array_tree_fingerprint(reference),
            }
        )

    def evaluate(self, /) -> SpinPeriodicSCFResult:
        evaluation = self.pencil.evaluate(self.mesh.fractional_points)
        if not bool(evaluation.successful):
            raise ValueError(
                "Spin SCF overlap pencil failed positivity/condition admission."
            )
        hamiltonian = evaluation.hamiltonians
        overlap = evaluation.overlaps
        weights = self.mesh.weights
        k_count = weights.size
        orbital_count = self.pencil.plan.basis.orbital_count
        alpha_target = 0.5 * (self.sector.electron_count + self.sector.spin_magnetization)
        beta_target = 0.5 * (self.sector.electron_count - self.sector.spin_magnetization)
        targets = (alpha_target, beta_target)
        populations = self.reference_spin_populations
        energies = jnp.zeros((2, k_count, orbital_count), dtype=hamiltonian.real.dtype)
        coefficients = jnp.zeros(
            (2, k_count, orbital_count, orbital_count), dtype=hamiltonian.dtype
        )
        occupations = jnp.zeros_like(energies)
        density = jnp.zeros_like(coefficients)
        chemical = jnp.zeros((2,), dtype=hamiltonian.real.dtype)
        entropy = jnp.asarray(0.0, dtype=hamiltonian.real.dtype)
        previous_energy = jnp.asarray(jnp.inf, dtype=hamiltonian.real.dtype)
        energy_residual = jnp.asarray(jnp.inf, dtype=hamiltonian.real.dtype)
        density_residual = jnp.asarray(jnp.inf, dtype=hamiltonian.real.dtype)
        total_energy = jnp.asarray(jnp.nan, dtype=hamiltonian.real.dtype)
        interaction = jnp.asarray(0.0, dtype=hamiltonian.real.dtype)
        one_electron = jnp.asarray(0.0, dtype=hamiltonian.real.dtype)
        converged = False
        completed = 0
        focks = jnp.zeros((2,) + hamiltonian.shape, dtype=hamiltonian.dtype)
        operator_properties = OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
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
            entropy = jnp.asarray(0.0, dtype=hamiltonian.real.dtype)
            for spin in range(2):
                other = 1 - spin
                potential = jnp.diag(
                    self.mean_field.onsite_hubbard
                    * (populations[other] - self.reference_spin_populations[other])
                )
                fock = hamiltonian + potential[None, :, :]
                solved = eigensolve(
                    GeneralizedEigenproblem(
                        DenseLinearOperator(fock, properties=operator_properties),
                        DenseLinearOperator(overlap, properties=metric_properties),
                        problem_id=f"{self.plan_id}:{iteration}:{spin}",
                    ),
                    policy=EigenSolvePolicy(
                        DenseEigh(),
                        count=orbital_count,
                        which="smallest-algebraic",
                    ),
                )
                if not bool(jnp.all(solved.successful)):
                    raise RuntimeError("Spin periodic generalized eigensolve failed.")
                spin_energies = solved.eigenvalues.real
                spin_coefficients = solved.eigenvectors
                spin_occupations, spin_chemical, spin_entropy = _spin_occupations(
                    spin_energies,
                    targets[spin],
                    weights,
                    self.smearing_energy,
                )
                spin_density = contract(
                    "kn,kan,kbn->kab",
                    spin_occupations,
                    spin_coefficients,
                    jnp.conj(spin_coefficients),
                    backend="jax",
                )
                focks = focks.at[spin].set(fock)
                energies = energies.at[spin].set(spin_energies)
                coefficients = coefficients.at[spin].set(spin_coefficients)
                occupations = occupations.at[spin].set(spin_occupations)
                density = density.at[spin].set(spin_density)
                chemical = chemical.at[spin].set(spin_chemical)
                entropy = entropy + spin_entropy
            proposed = jnp.real(
                contract("k,skab,kba->sa", weights, density, overlap, backend="jax")
            )
            mixed = self.damping * populations + (1.0 - self.damping) * proposed
            one_electron = jnp.real(
                contract("k,skab,kba->", weights, density, hamiltonian, backend="jax")
            )
            delta = mixed - self.reference_spin_populations
            interaction = jnp.sum(self.mean_field.onsite_hubbard * delta[0] * delta[1])
            total_energy = one_electron + interaction + self.mean_field.ionic_energy
            energy_residual = jnp.abs(total_energy - previous_energy)
            density_residual = jnp.max(jnp.abs(mixed - populations), initial=0.0)
            populations = mixed
            previous_energy = total_energy
            completed = iteration + 1
            if bool(
                jnp.maximum(energy_residual, density_residual)
                <= self.convergence_tolerance
            ):
                converged = True
                break
        free_energy = total_energy - self.smearing_energy * entropy
        eigenpair = contract(
            "skab,skbn->skan", focks, coefficients, backend="jax"
        ) - contract(
            "kab,skbn,skn->skan",
            overlap,
            coefficients,
            energies,
            backend="jax",
        )
        eigenpair_residual = jnp.max(jnp.abs(eigenpair), initial=0.0)
        commutator = contract(
            "skab,skbc,kcd->skad", focks, density, overlap, backend="jax"
        ) - contract("kab,skbc,skcd->skad", overlap, density, focks, backend="jax")
        commutator_residual = jnp.max(jnp.abs(commutator), initial=0.0)
        weighted_spin_counts = jnp.sum(weights[None, :, None] * occupations, axis=(1, 2))
        target_counts = jnp.asarray(targets, dtype=weighted_spin_counts.dtype)
        spin_count_residual = jnp.max(
            jnp.abs(weighted_spin_counts - target_counts), initial=0.0
        )
        electron_count_residual = jnp.abs(
            jnp.sum(weighted_spin_counts) - self.sector.electron_count
        )
        free_energy_residual = jnp.abs(
            free_energy - (total_energy - self.smearing_energy * entropy)
        )
        ledger = ElectronicEnergyLedger(
            ("one-electron", "hubbard", "ionic"),
            jnp.asarray((one_electron, interaction, self.mean_field.ionic_energy)),
            total_energy,
            self.pencil.plan.energy_unit,
        )
        evidence = SpinPeriodicSCFEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            eigenpair_residual,
            electron_count_residual,
            spin_count_residual,
            free_energy_residual,
            converged & (ledger.closure_residual <= self.convergence_tolerance),
            self.convergence_tolerance,
        )
        return SpinPeriodicSCFResult(
            total_energy,
            free_energy,
            entropy,
            chemical,
            energies,
            occupations,
            coefficients,
            density,
            populations,
            ledger,
            evidence,
            completed,
            self.pencil.plan.energy_unit,
            reference_kind=self.reference_kind,
            cell_id=self.mesh.cell_id,
            mesh_id=self.mesh.mesh_id,
            sector_id=self.sector.sector_id,
            pencil_id=self.pencil.plan.pencil_id,
            mean_field_id=self.mean_field.plan_id,
        )


__all__ = [
    "SpinPeriodicSCFEvidence",
    "SpinPeriodicSCFPlan",
    "SpinPeriodicSCFResult",
    "SpinReferenceKind",
]
