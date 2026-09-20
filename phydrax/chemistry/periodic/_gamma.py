#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Governed Gamma GDF-RHF and bounded candidate local-GTH FFTDF mean fields."""

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
from ...discretization import PeriodicCell
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)
from ...operators.quantum.gaussian import FactorizedERITensor
from ...units import ENERGY, LENGTH, UnitDefinition
from .._result import ElectronicEnergyLedger
from ._electrostatics import GTHPseudopotentialPlan
from ._source import PeriodicProvenanceManifest


GammaSCFClassification = Literal[
    "production-supplied-gdf-rhf", "candidate-local-gth-lda-x"
]


class GammaSCFEvidence(StrictModule, NonTrainableState):
    """Energy, density, metric, spectral, and governed-factor evidence."""

    energy_residual: Array
    density_residual: Array
    commutator_residual: Array
    eigenpair_residual: Array
    electron_count_residual: Array
    free_energy_residual: Array
    factorization_residual: Array | None
    successful: Array
    convergence_tolerance: float = eqx.field(static=True)
    factorization_tolerance: float | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_residual: ArrayLike,
        density_residual: ArrayLike,
        commutator_residual: ArrayLike,
        eigenpair_residual: ArrayLike,
        electron_count_residual: ArrayLike,
        free_energy_residual: ArrayLike,
        factorization_residual: ArrayLike | None,
        successful: ArrayLike,
        convergence_tolerance: float,
        /,
        *,
        factorization_tolerance: float | None = None,
    ):
        residuals = jnp.asarray(
            (
                energy_residual,
                density_residual,
                commutator_residual,
                eigenpair_residual,
                electron_count_residual,
                free_energy_residual,
            )
        ).reshape((6,))
        convergence = float(convergence_tolerance)
        factorization = (
            None
            if factorization_residual is None
            else jnp.asarray(factorization_residual, dtype=residuals.dtype).reshape(())
        )
        factorization_limit = (
            None if factorization_tolerance is None else float(factorization_tolerance)
        )
        if not isfinite(convergence) or convergence <= 0.0:
            raise ValueError(
                "Gamma SCF convergence tolerance must be finite and positive."
            )
        if (factorization is None) != (factorization_limit is None):
            raise ValueError(
                "Gamma factorization residual and tolerance must be supplied together."
            )
        if factorization_limit is not None and (
            not isfinite(factorization_limit) or factorization_limit < 0.0
        ):
            raise ValueError(
                "Gamma factorization tolerance must be finite and non-negative."
            )
        finite = jnp.all(jnp.isfinite(residuals)) & jnp.all(residuals >= 0.0)
        factor_ok = jnp.asarray(True)
        if factorization is not None:
            factor_ok = (
                jnp.isfinite(factorization)
                & (factorization >= 0.0)
                & (factorization <= factorization_limit)
            )
        admitted = (
            jnp.asarray(successful, dtype=jnp.bool_).reshape(())
            & finite
            & factor_ok
            & (residuals[0] <= convergence)
            & (residuals[1] <= convergence)
            & (residuals[2] <= 10.0 * convergence)
            & (residuals[3] <= 10.0 * convergence)
            & (residuals[4] <= 10.0 * convergence)
            & (residuals[5] <= 10.0 * convergence)
        )
        (
            self.energy_residual,
            self.density_residual,
            self.commutator_residual,
            self.eigenpair_residual,
            self.electron_count_residual,
            self.free_energy_residual,
        ) = tuple(residuals)
        self.factorization_residual = factorization
        self.successful = admitted
        self.convergence_tolerance = convergence
        self.factorization_tolerance = factorization_limit
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "gamma-scf-evidence",
                "convergence_tolerance": convergence.hex(),
                "factorization_tolerance": (
                    None if factorization_limit is None else factorization_limit.hex()
                ),
                "factorization_residual": (
                    None if factorization is None else float(factorization).hex()
                ),
                "successful": bool(admitted),
                "residuals": array_tree_fingerprint(np.asarray(residuals)),
            }
        )


class GammaSCFResult(StrictModule, NonTrainableState):
    total_energy: Array
    free_energy: Array
    entropy: Array
    chemical_potential: Array
    orbital_energies: Array
    occupations: Array
    coefficients: Array
    density: Array
    energy_ledger: ElectronicEnergyLedger
    evidence: GammaSCFEvidence
    iterations: Array
    successful: Array
    backend: str = eqx.field(static=True)
    classification: GammaSCFClassification = eqx.field(static=True)
    source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    energy_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        total_energy: ArrayLike,
        free_energy: ArrayLike,
        entropy: ArrayLike,
        chemical_potential: ArrayLike,
        orbital_energies: ArrayLike,
        occupations: ArrayLike,
        coefficients: ArrayLike,
        density: ArrayLike,
        energy_ledger: ElectronicEnergyLedger,
        evidence: GammaSCFEvidence,
        iterations: int,
        backend: str,
        classification: GammaSCFClassification,
        source_manifest_ids: tuple[str, ...],
        energy_unit: UnitDefinition,
        plan_id: str,
        /,
    ):
        energies = jnp.asarray(orbital_energies)
        occupations_ = jnp.asarray(occupations, dtype=energies.real.dtype)
        coefficients_ = jnp.asarray(coefficients)
        density_ = jnp.asarray(density)
        backend_ = str(backend).strip()
        sources = tuple(str(value).strip() for value in source_manifest_ids)
        if (
            energies.ndim != 1
            or occupations_.shape != energies.shape
            or coefficients_.shape != (energies.size, energies.size)
            or density_.ndim not in (1, 2, 3)
            or not backend_
            or classification
            not in ("production-supplied-gdf-rhf", "candidate-local-gth-lda-x")
            or not sources
            or any(not value for value in sources)
            or len(set(sources)) != len(sources)
        ):
            raise ValueError(
                "Gamma SCF orbitals, density, classification, or sources are invalid."
            )
        if not isinstance(energy_ledger, ElectronicEnergyLedger):
            raise TypeError("energy_ledger must be ElectronicEnergyLedger.")
        if not isinstance(evidence, GammaSCFEvidence):
            raise TypeError("evidence must be GammaSCFEvidence.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("Gamma energy_unit must have energy dimension.")
        if energy_ledger.energy_unit != energy_unit:
            raise ValueError("Gamma ledger and result energy units differ.")
        iterations_ = int(iterations)
        plan = str(plan_id).strip()
        if iterations_ <= 0 or not plan:
            raise ValueError("Gamma SCF result iterations and plan identity are invalid.")
        total = jnp.asarray(total_energy, dtype=energies.real.dtype).reshape(())
        free = jnp.asarray(free_energy, dtype=energies.real.dtype).reshape(())
        entropy_ = jnp.asarray(entropy, dtype=energies.real.dtype).reshape(())
        chemical = jnp.asarray(chemical_potential, dtype=energies.real.dtype).reshape(())
        finite = all(
            bool(jnp.all(jnp.isfinite(value)))
            for value in (
                total,
                free,
                entropy_,
                chemical,
                energies,
                occupations_,
                coefficients_,
                density_,
            )
        )
        self.total_energy = total
        self.free_energy = free
        self.entropy = entropy_
        self.chemical_potential = chemical
        self.orbital_energies = energies
        self.occupations = occupations_
        self.coefficients = coefficients_
        self.density = density_
        self.energy_ledger = energy_ledger
        self.evidence = evidence
        self.iterations = jnp.asarray(iterations_, dtype=jnp.int32)
        self.successful = evidence.successful & finite
        self.backend = backend_
        self.classification = classification
        self.source_manifest_ids = sources
        self.energy_unit = energy_unit
        self.plan_id = plan
        self.result_id = canonical_fingerprint(
            {
                "kind": "gamma-scf-result",
                "backend": backend_,
                "classification": classification,
                "sources": list(sources),
                "plan": plan,
                "energy_unit": energy_unit.unit_id,
                "ledger": energy_ledger.ledger_id,
                "evidence": evidence.evidence_id,
                "iterations": iterations_,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "total_energy": np.asarray(total),
                        "free_energy": np.asarray(free),
                        "entropy": np.asarray(entropy_),
                        "chemical_potential": np.asarray(chemical),
                        "orbital_energies": np.asarray(energies),
                        "occupations": np.asarray(occupations_),
                        "coefficients": np.asarray(coefficients_),
                        "density": np.asarray(density_),
                    }
                ),
            }
        )


def _gamma_occupations(
    energies: Array, electron_count: float, smearing: float, /
) -> tuple[Array, Array, Array]:
    count = float(electron_count)
    if smearing == 0.0:
        occupied = int(round(0.5 * count))
        if (
            abs(2.0 * occupied - count) > 1.0e-12
            or occupied < 1
            or occupied > energies.size
        ):
            raise ValueError(
                "Zero-smearing Gamma SCF requires a positive even admissible electron count."
            )
        occupations = jnp.zeros_like(energies.real).at[:occupied].set(2.0)
        highest = energies[occupied - 1].real
        if occupied < energies.size:
            lowest = energies[occupied].real
            scale = jnp.maximum(jnp.max(jnp.abs(energies.real)), 1.0)
            if float(highest) >= float(lowest - 1.0e-12 * scale):
                raise ValueError("Zero-smearing Gamma SCF requires an insulating gap.")
            chemical = 0.5 * (highest + lowest)
        else:
            chemical = highest
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


def _hermitian_eigh(matrix: Array, problem_id: str, /) -> tuple[Array, Array]:
    solve = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                matrix,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
            ),
            problem_id=problem_id,
        ),
        policy=EigenSolvePolicy(
            DenseEigh(), count=matrix.shape[0], which="smallest-algebraic"
        ),
    )
    if not bool(solve.successful):
        raise RuntimeError("Gamma Hermitian eigensolve failed.")
    return solve.eigenvalues.real, solve.eigenvectors


def _generalized_eigh(
    matrix: Array, overlap: Array, problem_id: str, /
) -> tuple[Array, Array]:
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
    solve = eigensolve(
        GeneralizedEigenproblem(
            DenseLinearOperator(matrix, properties=operator_properties),
            DenseLinearOperator(overlap, properties=metric_properties),
            problem_id=problem_id,
        ),
        policy=EigenSolvePolicy(
            DenseEigh(), count=matrix.shape[0], which="smallest-algebraic"
        ),
    )
    if not bool(solve.successful):
        raise RuntimeError("Gamma generalized Hermitian eigensolve failed.")
    return solve.eigenvalues.real, solve.eigenvectors


def _matrix_residuals(
    fock: Array,
    overlap: Array,
    coefficients: Array,
    energies: Array,
    density: Array,
    /,
) -> tuple[Array, Array]:
    eigenpair = fock @ coefficients - overlap @ (coefficients * energies[None, :])
    commutator = fock @ density @ overlap - overlap @ density @ fock
    return (
        jnp.max(jnp.abs(eigenpair), initial=0.0),
        jnp.max(jnp.abs(commutator), initial=0.0),
    )


class GammaFFTDFPlan(StrictModule, NonTrainableState):
    """Candidate local-GTH/Dirac-exchange grid SCF; nonlocal GTH is refused."""

    cell: PeriodicCell
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    ionic_positions_fractional: Array
    pseudopotentials: tuple[GTHPseudopotentialPlan, ...]
    electron_count: float = eqx.field(static=True)
    ionic_energy: float = eqx.field(static=True)
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    maximum_grid_points: int = eqx.field(static=True)
    energy_unit: UnitDefinition
    length_unit: UnitDefinition
    classification: GammaSCFClassification = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        grid_shape: tuple[int, int, int],
        ionic_positions_fractional: ArrayLike,
        pseudopotentials: tuple[GTHPseudopotentialPlan, ...],
        electron_count: float,
        ionic_energy: float,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        /,
        *,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
        damping: float = 0.3,
        maximum_grid_points: int = 512,
    ):
        if (
            not isinstance(cell, PeriodicCell)
            or cell.rank != 3
            or not cell.fully_periodic
        ):
            raise TypeError(
                "Gamma FFTDF requires a fully periodic three-dimensional cell."
            )
        shape = tuple(grid_shape)
        pseudopotentials_ = tuple(pseudopotentials)
        positions = np.asarray(ionic_positions_fractional)
        electrons = float(electron_count)
        ionic = float(ionic_energy)
        smearing = float(smearing_energy)
        tolerance = float(convergence_tolerance)
        maximum = int(maximum_iterations)
        damping_ = float(damping)
        capacity = int(maximum_grid_points)
        size = int(np.prod(shape)) if len(shape) == 3 else 0
        if (
            len(shape) != 3
            or any(value < 2 for value in shape)
            or size > capacity
            or not pseudopotentials_
            or any(
                not isinstance(value, GTHPseudopotentialPlan)
                for value in pseudopotentials_
            )
            or positions.shape != (len(pseudopotentials_), 3)
            or np.any(~np.isfinite(positions))
            or electrons <= 0.0
            or electrons > 2.0 * size
            or any(
                not isfinite(value) for value in (ionic, smearing, tolerance, damping_)
            )
            or smearing < 0.0
            or tolerance <= 0.0
            or maximum <= 0
            or not 0.0 <= damping_ < 1.0
        ):
            raise ValueError(
                "Gamma FFTDF grid, ions, electrons, or SCF policy is invalid."
            )
        if any(value.channels for value in pseudopotentials_):
            raise ValueError(
                "Candidate Gamma FFTDF implements local GTH only and refuses nonlocal channels."
            )
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("Gamma FFTDF energy_unit must have energy dimension.")
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise TypeError("Gamma FFTDF length_unit must have length dimension.")
        if any(
            value.energy_unit != energy_unit or value.length_unit != length_unit
            for value in pseudopotentials_
        ):
            raise ValueError("Gamma FFTDF and GTH units must match exactly.")
        if not np.isclose(
            sum(value.ionic_charge for value in pseudopotentials_),
            electrons,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Candidate local-GTH FFTDF requires a neutral valence cell.")
        self.cell = cell
        self.grid_shape = shape
        self.ionic_positions_fractional = jnp.asarray(positions)
        self.pseudopotentials = pseudopotentials_
        self.electron_count = electrons
        self.ionic_energy = ionic
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_iterations = maximum
        self.damping = damping_
        self.maximum_grid_points = capacity
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.classification = "candidate-local-gth-lda-x"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gamma-fftdf-plan",
                "classification": self.classification,
                "cell": cell.cell_id,
                "grid_shape": list(shape),
                "pseudopotentials": [value.plan_id for value in pseudopotentials_],
                "electron_count": electrons.hex(),
                "ionic_energy": ionic.hex(),
                "smearing_energy": smearing.hex(),
                "convergence_tolerance": tolerance.hex(),
                "maximum_iterations": maximum,
                "damping": damping_.hex(),
                "maximum_grid_points": capacity,
                "energy_unit": energy_unit.unit_id,
                "length_unit": length_unit.unit_id,
                "ionic_positions": array_tree_fingerprint(positions),
            }
        )

    def evaluate(self, /) -> GammaSCFResult:
        shape = self.grid_shape
        size = int(np.prod(shape))
        cell = jnp.asarray(self.cell.vectors)
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
        wavevectors = (
            2.0 * jnp.pi * contract("...i,ji->...j", integer_grid, inverse, backend="jax")
        )
        wavevectors_flat = wavevectors.reshape((-1, 3))
        squared = jnp.sum(wavevectors**2, axis=-1)
        squared_flat = squared.reshape((-1,))
        ionic_cartesian = self.ionic_positions_fractional @ cell
        ionic_spectrum = jnp.zeros(
            (size,), dtype=jnp.result_type(cell.dtype, jnp.complex64)
        )
        source_manifest_ids = []
        for position, pseudopotential in zip(
            ionic_cartesian, self.pseudopotentials, strict=True
        ):
            local = pseudopotential.local_reciprocal(squared_flat)
            if not bool(local.successful):
                raise RuntimeError("Governed GTH local component evaluation failed.")
            ionic_spectrum = ionic_spectrum + local.values * jnp.exp(
                -1.0j * (wavevectors_flat @ position)
            )
            source_manifest_ids.append(pseudopotential.source_manifest.manifest_id)
        ionic_potential = (
            jnp.real(jnp.fft.ifftn(ionic_spectrum.reshape(shape))) * size / volume
        ).reshape((-1,))
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
        kinetic = jnp.conj(fourier.T) @ jnp.diag(0.5 * squared_flat) @ fourier
        overlap = jnp.eye(size, dtype=kinetic.dtype)
        grid_weight = volume / size
        density_grid = jnp.full((size,), self.electron_count / volume, dtype=cell.dtype)
        previous_energy = jnp.asarray(jnp.inf, dtype=cell.dtype)
        energy_residual = jnp.asarray(jnp.inf, dtype=cell.dtype)
        density_residual = jnp.asarray(jnp.inf, dtype=cell.dtype)
        total = jnp.asarray(jnp.nan, dtype=cell.dtype)
        free = total
        energies = jnp.zeros((size,), dtype=cell.dtype)
        occupations = jnp.zeros_like(energies)
        coefficients = jnp.eye(size, dtype=kinetic.dtype)
        density_matrix = jnp.zeros_like(kinetic)
        fock = kinetic + jnp.diag(ionic_potential)
        one_body = jnp.asarray(0.0, dtype=cell.dtype)
        hartree_energy = jnp.asarray(0.0, dtype=cell.dtype)
        exchange_energy = jnp.asarray(0.0, dtype=cell.dtype)
        entropy = jnp.asarray(0.0, dtype=cell.dtype)
        chemical = jnp.asarray(0.0, dtype=cell.dtype)
        converged = False
        completed = 0
        for iteration in range(self.maximum_iterations):
            density_fourier = jnp.fft.fftn(density_grid.reshape(shape))
            coulomb_kernel = jnp.where(squared > 0.0, 4.0 * jnp.pi / squared, 0.0)
            hartree_potential = jnp.real(
                jnp.fft.ifftn(coulomb_kernel * density_fourier)
            ).reshape((-1,))
            safe_density = jnp.maximum(
                density_grid, jnp.finfo(density_grid.dtype).tiny ** 0.25
            )
            exchange_coefficient = -0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)
            exchange_energy_density = exchange_coefficient * safe_density ** (4.0 / 3.0)
            exchange_potential = (
                (4.0 / 3.0) * exchange_coefficient * safe_density ** (1.0 / 3.0)
            )
            local_potential = ionic_potential + hartree_potential + exchange_potential
            fock = kinetic + jnp.diag(local_potential)
            energies, coefficients = _generalized_eigh(
                fock, overlap, f"{self.plan_id}:{iteration}"
            )
            occupations, chemical, entropy = _gamma_occupations(
                energies, self.electron_count, self.smearing_energy
            )
            density_matrix = contract(
                "pi,i,qi->pq",
                coefficients,
                occupations,
                jnp.conj(coefficients),
                backend="jax",
            )
            proposed = jnp.real(jnp.diag(density_matrix)) / grid_weight
            mixed = self.damping * density_grid + (1.0 - self.damping) * proposed
            one_body = jnp.real(
                contract(
                    "ab,ba->",
                    density_matrix,
                    kinetic + jnp.diag(ionic_potential),
                    backend="jax",
                )
            )
            hartree_energy = 0.5 * grid_weight * jnp.sum(mixed * hartree_potential)
            exchange_energy = grid_weight * jnp.sum(exchange_energy_density)
            total = one_body + hartree_energy + exchange_energy + self.ionic_energy
            free = total - self.smearing_energy * entropy
            energy_residual = jnp.abs(total - previous_energy)
            density_residual = jnp.max(jnp.abs(mixed - density_grid), initial=0.0)
            density_grid = mixed
            previous_energy = total
            completed = iteration + 1
            if bool(
                jnp.maximum(energy_residual, density_residual)
                <= self.convergence_tolerance
            ):
                converged = True
                break
        eigenpair_residual, commutator_residual = _matrix_residuals(
            fock, overlap, coefficients, energies, density_matrix
        )
        electron_count_residual = jnp.abs(
            jnp.real(contract("ab,ba->", density_matrix, overlap, backend="jax"))
            - self.electron_count
        )
        free_energy_residual = jnp.abs(free - (total - self.smearing_energy * entropy))
        ledger = ElectronicEnergyLedger(
            ("one-electron", "hartree", "dirac-exchange", "ionic"),
            jnp.asarray((one_body, hartree_energy, exchange_energy, self.ionic_energy)),
            total,
            self.energy_unit,
        )
        evidence = GammaSCFEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            eigenpair_residual,
            electron_count_residual,
            free_energy_residual,
            None,
            converged & (ledger.closure_residual <= self.convergence_tolerance),
            self.convergence_tolerance,
        )
        return GammaSCFResult(
            total,
            free,
            entropy,
            chemical,
            energies,
            occupations,
            coefficients,
            density_grid.reshape(shape),
            ledger,
            evidence,
            completed,
            "bounded-grid-local-gth-dirac-exchange",
            self.classification,
            tuple(sorted(set(source_manifest_ids))),
            self.energy_unit,
            self.plan_id,
        )


class GammaGDFPlan(StrictModule, NonTrainableState):
    """Production RHF postprocessing of governed supplied one- and two-electron data."""

    one_body: Array
    overlap: Array
    factors: FactorizedERITensor
    electron_count: float = eqx.field(static=True)
    nuclear_energy: float = eqx.field(static=True)
    source_manifest: PeriodicProvenanceManifest
    smearing_energy: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_factorization_residual: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    classification: GammaSCFClassification = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        one_body: ArrayLike,
        overlap: ArrayLike,
        factors: FactorizedERITensor,
        electron_count: float,
        nuclear_energy: float,
        energy_unit: UnitDefinition,
        source_manifest: PeriodicProvenanceManifest,
        /,
        *,
        smearing_energy: float = 0.0,
        convergence_tolerance: float = 1.0e-9,
        maximum_factorization_residual: float = 1.0e-8,
        maximum_iterations: int = 200,
        damping: float = 0.2,
    ):
        one = jnp.asarray(one_body)
        overlap_ = jnp.asarray(overlap, dtype=one.dtype)
        electrons = float(electron_count)
        nuclear = float(nuclear_energy)
        smearing = float(smearing_energy)
        tolerance = float(convergence_tolerance)
        factorization_tolerance = float(maximum_factorization_residual)
        maximum = int(maximum_iterations)
        damping_ = float(damping)
        count = one.shape[0] if one.ndim == 2 else -1
        if (
            one.shape != (count, count)
            or overlap_.shape != one.shape
            or not isinstance(factors, FactorizedERITensor)
            or factors.orbital_count != count
            or not bool(jnp.all(jnp.isfinite(one)))
            or not bool(jnp.all(jnp.isfinite(overlap_)))
            or not bool(jnp.all(jnp.isfinite(factors.factors)))
            or not np.allclose(
                np.asarray(one), np.asarray(jnp.conj(one.T)), rtol=0.0, atol=1.0e-12
            )
            or not np.allclose(
                np.asarray(overlap_),
                np.asarray(jnp.conj(overlap_.T)),
                rtol=0.0,
                atol=1.0e-12,
            )
            or electrons <= 0.0
            or electrons > 2.0 * count
            or any(
                not isfinite(value)
                for value in (
                    nuclear,
                    smearing,
                    tolerance,
                    factorization_tolerance,
                    damping_,
                )
            )
            or smearing < 0.0
            or tolerance <= 0.0
            or factorization_tolerance < 0.0
            or maximum <= 0
            or not 0.0 <= damping_ < 1.0
        ):
            raise ValueError("Gamma GDF integrals, electrons, or SCF policy are invalid.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise TypeError("Gamma GDF energy_unit must have energy dimension.")
        if not isinstance(source_manifest, PeriodicProvenanceManifest):
            raise TypeError(
                "Gamma GDF source_manifest must be PeriodicProvenanceManifest."
            )
        if source_manifest.source_id != factors.source_id:
            raise ValueError(
                "Gamma GDF factor source must match the governed source manifest."
            )
        if float(factors.residual_bound) > factorization_tolerance:
            raise ValueError(
                "Gamma GDF factorization residual exceeds its admitted bound."
            )
        overlap_values, _ = _hermitian_eigh(
            overlap_, f"gamma-gdf-overlap:{source_manifest.manifest_id}"
        )
        if float(jnp.min(overlap_values)) <= 1.0e-10:
            raise ValueError("Gamma GDF overlap must be positive definite.")
        self.one_body = one
        self.overlap = overlap_
        self.factors = factors
        self.electron_count = electrons
        self.nuclear_energy = nuclear
        self.source_manifest = source_manifest
        self.smearing_energy = smearing
        self.convergence_tolerance = tolerance
        self.maximum_factorization_residual = factorization_tolerance
        self.maximum_iterations = maximum
        self.damping = damping_
        self.energy_unit = energy_unit
        self.classification = "production-supplied-gdf-rhf"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gamma-gdf-rhf-plan",
                "classification": self.classification,
                "factors": factors.tensor_id,
                "source_manifest": source_manifest.manifest_id,
                "electron_count": electrons.hex(),
                "nuclear_energy": nuclear.hex(),
                "smearing_energy": smearing.hex(),
                "convergence_tolerance": tolerance.hex(),
                "maximum_factorization_residual": factorization_tolerance.hex(),
                "maximum_iterations": maximum,
                "damping": damping_.hex(),
                "energy_unit": energy_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {"one_body": np.asarray(one), "overlap": np.asarray(overlap_)}
                ),
            }
        )

    def evaluate(self, /) -> GammaSCFResult:
        energies, coefficients = _generalized_eigh(
            self.one_body, self.overlap, f"{self.plan_id}:initial"
        )
        occupations, chemical, entropy = _gamma_occupations(
            energies, self.electron_count, self.smearing_energy
        )
        density = contract(
            "pi,i,qi->pq",
            coefficients,
            occupations,
            jnp.conj(coefficients),
            backend="jax",
        )
        previous_energy = jnp.asarray(jnp.inf, dtype=self.one_body.real.dtype)
        energy_residual = jnp.asarray(jnp.inf, dtype=previous_energy.dtype)
        density_residual = jnp.asarray(jnp.inf, dtype=previous_energy.dtype)
        total = jnp.asarray(jnp.nan, dtype=previous_energy.dtype)
        free = total
        one_electron = jnp.asarray(0.0, dtype=previous_energy.dtype)
        coulomb_energy = jnp.asarray(0.0, dtype=previous_energy.dtype)
        exchange_energy = jnp.asarray(0.0, dtype=previous_energy.dtype)
        fock = self.one_body
        converged = False
        completed = 0
        for iteration in range(self.maximum_iterations):
            coulomb = self.factors.coulomb(density)
            exchange = self.factors.exchange(density)
            fock = self.one_body + coulomb - 0.5 * exchange
            energies, coefficients = _generalized_eigh(
                fock, self.overlap, f"{self.plan_id}:{iteration}"
            )
            occupations, chemical, entropy = _gamma_occupations(
                energies, self.electron_count, self.smearing_energy
            )
            proposed = contract(
                "pi,i,qi->pq",
                coefficients,
                occupations,
                jnp.conj(coefficients),
                backend="jax",
            )
            mixed = self.damping * density + (1.0 - self.damping) * proposed
            coulomb_new = self.factors.coulomb(mixed)
            exchange_new = self.factors.exchange(mixed)
            one_electron = jnp.real(
                contract("ab,ba->", mixed, self.one_body, backend="jax")
            )
            coulomb_energy = 0.5 * jnp.real(
                contract("ab,ba->", mixed, coulomb_new, backend="jax")
            )
            exchange_energy = -0.25 * jnp.real(
                contract("ab,ba->", mixed, exchange_new, backend="jax")
            )
            total = one_electron + coulomb_energy + exchange_energy + self.nuclear_energy
            free = total - self.smearing_energy * entropy
            energy_residual = jnp.abs(total - previous_energy)
            density_residual = jnp.max(jnp.abs(mixed - density), initial=0.0)
            density = mixed
            previous_energy = total
            completed = iteration + 1
            if bool(
                jnp.maximum(energy_residual, density_residual)
                <= self.convergence_tolerance
            ):
                converged = True
                break
        eigenpair_residual, commutator_residual = _matrix_residuals(
            fock, self.overlap, coefficients, energies, density
        )
        electron_count_residual = jnp.abs(
            jnp.real(contract("ab,ba->", density, self.overlap, backend="jax"))
            - self.electron_count
        )
        free_energy_residual = jnp.abs(free - (total - self.smearing_energy * entropy))
        ledger = ElectronicEnergyLedger(
            ("one-electron", "coulomb", "exact-exchange", "nuclear"),
            jnp.asarray(
                (
                    one_electron,
                    coulomb_energy,
                    exchange_energy,
                    self.nuclear_energy,
                )
            ),
            total,
            self.energy_unit,
        )
        evidence = GammaSCFEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            eigenpair_residual,
            electron_count_residual,
            free_energy_residual,
            self.factors.residual_bound,
            converged & (ledger.closure_residual <= self.convergence_tolerance),
            self.convergence_tolerance,
            factorization_tolerance=self.maximum_factorization_residual,
        )
        return GammaSCFResult(
            total,
            free,
            entropy,
            chemical,
            energies,
            occupations,
            coefficients,
            density,
            ledger,
            evidence,
            completed,
            "governed-supplied-density-fitted-rhf",
            self.classification,
            (self.source_manifest.manifest_id,),
            self.energy_unit,
            self.plan_id,
        )


__all__ = [
    "GammaFFTDFPlan",
    "GammaGDFPlan",
    "GammaSCFClassification",
    "GammaSCFEvidence",
    "GammaSCFResult",
]
