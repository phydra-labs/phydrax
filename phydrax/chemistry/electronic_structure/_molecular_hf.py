#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""General-basis RHF, UHF, ROHF, and GHF with explicit SCF evidence."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan
from ...ein import contract
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ...nonlinear import Bisection, NonlinearTermination, scalar_root, ScalarRootProblem
from ...operators.quantum.gaussian import (
    electron_repulsion_tensor,
    FactorizedERITensor,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion_energy,
    overlap_matrix,
    PreparedDirectJK,
    PreparedGaussianBasis,
)
from ...units import BOHR, conversion_factor, HARTREE
from .._context import ElectronicInitialGuessState
from .._model import ElectronicReferenceKind
from .._state import MolecularElectronicSectorPlan, PreparedMolecularElectronicSector
from ._mean_field import (
    ElectronicOccupationKind,
    ElectronicOccupationPlan,
    GeneralizedMeanFieldState,
    InitialGuessKind,
    InitialGuessPlan,
    RestrictedMeanFieldState,
    SCFAccelerationKind,
    SCFAccelerationPlan,
    SCFConvergenceEvidence,
    SCFConvergencePlan,
    SCFStabilityPlan,
    SCFStabilityResult,
    UnrestrictedMeanFieldState,
)


class MolecularGradientResult(StrictModule):
    gradient: Array
    forces: Array
    pulay: Array
    one_electron: Array
    two_electron: Array
    nuclear: Array
    successful: Array
    state_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _hermitian_eigh(matrix: Array, count: int | None = None, /) -> tuple[Array, Array]:
    dimension = matrix.shape[0]
    solve_ = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                0.5 * (matrix + jnp.conj(matrix.T)),
                properties=OperatorProperties(
                    self_adjoint=True, evidence={"self_adjoint": "construction"}
                ),
            )
        ),
        policy=EigenSolvePolicy(
            DenseEigh(),
            count=dimension if count is None else int(count),
            which="smallest-algebraic",
        ),
    )
    if not bool(solve_.successful):
        raise RuntimeError("Mean-field Hermitian eigensolve failed.")
    return solve_.eigenvalues, solve_.eigenvectors


def _orthogonalizer(overlap: Array, tolerance: float, /) -> tuple[Array, int]:
    eigenvalues, eigenvectors = _hermitian_eigh(overlap)
    retained = np.flatnonzero(np.asarray(eigenvalues) > tolerance)
    if retained.size == 0:
        raise ValueError("Gaussian overlap has no retained numerical rank.")
    vectors = eigenvectors[:, retained]
    return vectors / jnp.sqrt(eigenvalues[retained])[None, :], int(
        overlap.shape[0] - retained.size
    )


def _diagonalize(fock: Array, orthogonalizer: Array, /) -> tuple[Array, Array]:
    values, vectors = _hermitian_eigh(jnp.conj(orthogonalizer.T) @ fock @ orthogonalizer)
    return values, orthogonalizer @ vectors


def _density(coefficients: Array, occupations: Array, /) -> Array:
    return contract("pi,i,qi->pq", coefficients, occupations, jnp.conj(coefficients))


def _commutator_norm(fock: Array, density: Array, overlap: Array, /) -> Array:
    residual = fock @ density @ overlap - overlap @ density @ fock
    return jnp.max(jnp.abs(residual), initial=0.0)


def _integer_occupations(
    orbital_count: int, electron_count: int, maximum: float, /
) -> Array:
    maximum_electrons = int(round(maximum)) * orbital_count
    if electron_count < 0 or electron_count > maximum_electrons:
        raise ValueError("Electron count exceeds retained orbital capacity.")
    full = electron_count // int(round(maximum))
    remainder = electron_count - full * int(round(maximum))
    values = jnp.zeros((orbital_count,), dtype=jnp.float64)
    values = values.at[:full].set(maximum)
    if remainder:
        values = values.at[full].set(float(remainder))
    return values


def _occupations(
    plan: ElectronicOccupationPlan,
    energies: Array,
    electron_count: int,
    maximum: float,
    /,
) -> tuple[Array, Array, Array]:
    orbital_count = energies.size
    if plan.kind in (
        ElectronicOccupationKind.INTEGER,
        ElectronicOccupationKind.MAXIMUM_OVERLAP,
    ):
        occupation = _integer_occupations(orbital_count, electron_count, maximum).astype(
            energies.dtype
        )
        occupied = np.flatnonzero(np.asarray(occupation) > 0.0)
        chemical = energies[int(occupied[-1])] if occupied.size else energies[0]
        return occupation, chemical, jnp.asarray(0.0, dtype=energies.dtype)
    if plan.kind is ElectronicOccupationKind.EXPLICIT:
        occupation = jnp.asarray(plan.explicit_occupations, dtype=energies.dtype)
        if occupation.shape != energies.shape:
            raise ValueError("Explicit occupations do not align with retained orbitals.")
        if bool(jnp.any(occupation > maximum)) or not np.isclose(
            float(jnp.sum(occupation)), electron_count, atol=1.0e-10
        ):
            raise ValueError("Explicit occupations violate capacity or electron count.")
        occupied = np.flatnonzero(np.asarray(occupation) > 0.0)
        chemical = energies[int(occupied[-1])] if occupied.size else energies[0]
        return occupation, chemical, jnp.asarray(0.0, dtype=energies.dtype)
    temperature = plan.smearing_energy
    lower = jnp.min(energies) - 80.0 * temperature
    upper = jnp.max(energies) + 80.0 * temperature
    problem = ScalarRootProblem(
        lambda chemical, _: (
            jnp.sum(maximum / (1.0 + jnp.exp((energies - chemical) / temperature)))
            - electron_count
        ),
        bracket=(lower, upper),
        problem_id=f"mean-field-chemical-potential:{plan.plan_id}",
    )
    root = scalar_root(
        problem,
        method=Bisection(),
        termination=NonlinearTermination(
            absolute_residual=1.0e-12,
            relative_residual=0.0,
            maximum_steps=200,
            maximum_evaluations=404,
            maximum_linear_iterations=1,
        ),
    )
    if not bool(root.successful):
        raise RuntimeError("Fermi-Dirac chemical-potential solve failed.")
    occupation = maximum / (1.0 + jnp.exp((energies - root.root) / temperature))
    probability = jnp.clip(occupation / maximum, 1.0e-15, 1.0 - 1.0e-15)
    entropy = -maximum * jnp.sum(
        probability * jnp.log(probability)
        + (1.0 - probability) * jnp.log(1.0 - probability)
    )
    return occupation, root.root, entropy


def _maximum_overlap_occupations(
    coefficients: Array,
    energies: Array,
    previous_coefficients: Array,
    previous_occupations: Array,
    overlap: Array,
    electron_count: int,
    maximum: float,
    /,
) -> tuple[Array, Array, Array]:
    occupied = np.flatnonzero(np.asarray(previous_occupations) > 1.0e-12)
    if occupied.size == 0:
        raise ValueError("Maximum-overlap occupations require an occupied reference.")
    orbital_overlaps = (
        jnp.conj(previous_coefficients[:, occupied].T) @ overlap @ coefficients
    )
    scores = jnp.sum(jnp.abs(orbital_overlaps) ** 2, axis=0)
    order = np.argsort(-np.asarray(scores), kind="stable")
    occupations = jnp.zeros((coefficients.shape[1],), dtype=scores.dtype)
    remaining = int(electron_count)
    for index in order:
        population = min(int(round(maximum)), remaining)
        if population <= 0:
            break
        occupations = occupations.at[int(index)].set(float(population))
        remaining -= population
    if remaining:
        raise ValueError("Maximum-overlap occupation exceeds orbital capacity.")
    selected = np.flatnonzero(np.asarray(occupations) > 0.0)
    chemical = jnp.max(energies[selected]) if selected.size else energies[0]
    return occupations, chemical, jnp.asarray(0.0, dtype=scores.dtype)


def _updated_occupations(
    plan: ElectronicOccupationPlan,
    energies: Array,
    coefficients: Array,
    previous_coefficients: Array,
    previous_occupations: Array,
    overlap: Array,
    electron_count: int,
    maximum: float,
    /,
) -> tuple[Array, Array, Array]:
    if plan.kind is ElectronicOccupationKind.MAXIMUM_OVERLAP:
        return _maximum_overlap_occupations(
            coefficients,
            energies,
            previous_coefficients,
            previous_occupations,
            overlap,
            electron_count,
            maximum,
        )
    return _occupations(plan, energies, electron_count, maximum)


def _level_shift_fock(
    fock: Array,
    overlap: Array,
    coefficients: Array,
    occupations: Array,
    maximum: float,
    shift: float,
    /,
) -> Array:
    if shift == 0.0:
        return fock
    virtual_weights = jnp.clip(1.0 - occupations / maximum, 0.0, 1.0)
    virtual_projector = contract(
        "pi,i,qi->pq",
        coefficients,
        virtual_weights,
        jnp.conj(coefficients),
    )
    return fock + shift * overlap @ virtual_projector @ overlap


def _diis_extrapolate(
    focks: list[Array], errors: list[Array], maximum_space: int, /
) -> Array:
    if len(focks) != len(errors) or len(focks) < 2:
        raise ValueError("DIIS requires at least two aligned Fock/error pairs.")
    focks_ = focks[-maximum_space:]
    errors_ = errors[-maximum_space:]
    count = len(focks_)
    matrix = jnp.zeros((count + 1, count + 1), dtype=focks_[0].real.dtype)
    for left in range(count):
        for right in range(count):
            matrix = matrix.at[left, right].set(
                jnp.real(jnp.vdot(errors_[left], errors_[right]))
            )
    matrix = matrix.at[count, :count].set(-1.0)
    matrix = matrix.at[:count, count].set(-1.0)
    right = jnp.zeros((count + 1,), dtype=matrix.dtype).at[count].set(-1.0)
    result = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )
    if not bool(result.successful):
        raise RuntimeError("SCF DIIS coefficient solve failed.")
    return sum(
        (result.value[index] * fock for index, fock in enumerate(focks_)),
        start=jnp.zeros_like(focks_[0]),
    )


class MolecularHartreeFockPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    basis: PreparedGaussianBasis
    sector: PreparedMolecularElectronicSector
    reference: ElectronicReferenceKind = eqx.field(static=True)
    convergence: SCFConvergencePlan
    acceleration: SCFAccelerationPlan
    occupations: ElectronicOccupationPlan
    guess: InitialGuessPlan
    stability: SCFStabilityPlan
    linear_dependence_tolerance: float = eqx.field(static=True)
    factorized_eri: FactorizedERITensor | None
    direct_jk: PreparedDirectJK | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        basis: PreparedGaussianBasis,
        sector: MolecularElectronicSectorPlan | PreparedMolecularElectronicSector,
        reference: ElectronicReferenceKind,
        /,
        *,
        convergence: SCFConvergencePlan | None = None,
        acceleration: SCFAccelerationPlan | None = None,
        occupations: ElectronicOccupationPlan | None = None,
        guess: InitialGuessPlan | None = None,
        stability: SCFStabilityPlan | None = None,
        linear_dependence_tolerance: float = 1.0e-9,
        factorized_eri: FactorizedERITensor | None = None,
        direct_jk: PreparedDirectJK | None = None,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if (
            not isinstance(basis, PreparedGaussianBasis)
            or basis.system_id != system.system_id
        ):
            raise ValueError("basis must be prepared for the Hartree-Fock system.")
        sector_ = (
            sector.prepare(system)
            if isinstance(sector, MolecularElectronicSectorPlan)
            else sector
        )
        if (
            not isinstance(sector_, PreparedMolecularElectronicSector)
            or sector_.system_id != system.system_id
        ):
            raise ValueError(
                "Electronic sector must be prepared for the Hartree-Fock system."
            )
        if not isinstance(reference, ElectronicReferenceKind):
            raise TypeError("reference must be ElectronicReferenceKind.")
        if (
            reference is ElectronicReferenceKind.RESTRICTED
            and sector_.alpha_electron_count != sector_.beta_electron_count
        ):
            raise ValueError("RHF requires equal alpha and beta electron counts.")
        if reference is ElectronicReferenceKind.NONCOLLINEAR:
            raise ValueError(
                "Noncollinear references belong to Kohn-Sham spin-density plans."
            )
        convergence_ = SCFConvergencePlan() if convergence is None else convergence
        acceleration_ = SCFAccelerationPlan() if acceleration is None else acceleration
        occupation_ = ElectronicOccupationPlan() if occupations is None else occupations
        guess_ = InitialGuessPlan() if guess is None else guess
        stability_ = SCFStabilityPlan() if stability is None else stability
        if not isinstance(convergence_, SCFConvergencePlan) or not isinstance(
            acceleration_, SCFAccelerationPlan
        ):
            raise TypeError("convergence and acceleration must be typed SCF plans.")
        if not isinstance(occupation_, ElectronicOccupationPlan) or not isinstance(
            guess_, InitialGuessPlan
        ):
            raise TypeError("occupations and guess must be typed mean-field plans.")
        if guess_.kind in (InitialGuessKind.SAD, InitialGuessKind.SAP):
            raise NotImplementedError(
                "SAD/SAP require a governed atomic guess provider; use an explicit guess state."
            )
        if not isinstance(stability_, SCFStabilityPlan):
            raise TypeError("stability must be SCFStabilityPlan.")
        if reference is not ElectronicReferenceKind.RESTRICTED and (
            stability_.require_internal or stability_.require_external
        ):
            raise NotImplementedError(
                "Native orbital-Hessian stability requirements currently apply only to RHF."
            )
        tolerance = float(linear_dependence_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("linear_dependence_tolerance must be positive finite.")
        if factorized_eri is not None and direct_jk is not None:
            raise ValueError("Choose at most one factorized or direct J/K route.")
        if (
            factorized_eri is not None
            and factorized_eri.orbital_count != basis.basis_function_count
        ):
            raise ValueError("Factorized ERIs do not align with the Gaussian basis.")
        if direct_jk is not None and direct_jk.basis.prepared_id != basis.prepared_id:
            raise ValueError("Direct J/K plan belongs to another Gaussian basis.")
        self.system = system
        self.basis = basis
        self.sector = sector_
        self.reference = reference
        self.convergence = convergence_
        self.acceleration = acceleration_
        self.occupations = occupation_
        self.guess = guess_
        self.stability = stability_
        self.linear_dependence_tolerance = tolerance
        self.factorized_eri = factorized_eri
        self.direct_jk = direct_jk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-hartree-fock-plan",
                "system": system.system_id,
                "basis": basis.prepared_id,
                "sector": sector_.prepared_id,
                "reference": reference.value,
                "convergence": convergence_.plan_id,
                "acceleration": acceleration_.plan_id,
                "occupations": occupation_.plan_id,
                "guess": guess_.plan_id,
                "stability": stability_.plan_id,
                "linear_dependence_tolerance": tolerance,
                "factorized_eri": None
                if factorized_eri is None
                else factorized_eri.tensor_id,
                "direct_jk": None if direct_jk is None else direct_jk.prepared_id,
            }
        )

    def _integrals(self, positions: Array, /):
        charges = jnp.asarray(self.system.atomic_numbers, dtype=positions.dtype)
        overlap = overlap_matrix(self.basis, positions)
        core = kinetic_matrix(self.basis, positions) + nuclear_attraction_matrix(
            self.basis, positions, charges
        )
        nuclear = nuclear_repulsion_energy(positions, charges)
        dense = (
            electron_repulsion_tensor(self.basis, positions)
            if self.factorized_eri is None and self.direct_jk is None
            else None
        )
        return overlap, core, nuclear, dense

    def _jk(self, positions: Array, density: Array, dense: Array | None, /):
        if self.factorized_eri is not None:
            return (
                self.factorized_eri.coulomb(density),
                self.factorized_eri.exchange(density),
                jnp.asarray(True),
            )
        if self.direct_jk is not None:
            result = self.direct_jk.evaluate(positions, density)
            return result.coulomb, result.exchange, result.successful
        if dense is None:
            raise RuntimeError("Dense ERIs are unexpectedly absent.")
        return (
            contract("cd,abcd->ab", density, dense),
            contract("cd,acbd->ab", density, dense),
            jnp.asarray(True),
        )

    def _initial_coefficients(
        self,
        core: Array,
        overlap: Array,
        orthogonalizer: Array,
        /,
    ):
        guess_kind = self.guess.kind
        if guess_kind in (
            InitialGuessKind.CORE,
            InitialGuessKind.ZERO,
            InitialGuessKind.EXPLICIT,
            InitialGuessKind.PROJECTED,
        ):
            guess_hamiltonian = core
        elif guess_kind is InitialGuessKind.HUCKEL:
            diagonal = jnp.diag(core)
            guess_hamiltonian = 0.875 * (diagonal[:, None] + diagonal[None, :]) * overlap
            guess_hamiltonian = guess_hamiltonian.at[jnp.diag_indices(core.shape[0])].set(
                diagonal
            )
        else:
            raise RuntimeError("Unknown native molecular initial guess.")
        return _diagonalize(guess_hamiltonian, orthogonalizer)

    def _validated_initial_guess(
        self, initial_guess: ElectronicInitialGuessState | None, /
    ) -> Array | None:
        requires = self.guess.kind in (
            InitialGuessKind.EXPLICIT,
            InitialGuessKind.PROJECTED,
        )
        if requires != (initial_guess is not None):
            raise ValueError(
                "Explicit/projected guess plans and initial guess state must be supplied together."
            )
        if initial_guess is None:
            return None
        if (
            initial_guess.basis_id != self.basis.prepared_id
            or initial_guess.sector_id != self.sector.prepared_id
            or initial_guess.guess_id != self.guess.source_id
        ):
            raise ValueError(
                "Initial guess basis, sector, or source identity differs from the SCF plan."
            )
        return initial_guess.density

    def solve_atomic_units(
        self,
        positions: ArrayLike,
        /,
        *,
        initial_guess: ElectronicInitialGuessState | None = None,
    ):
        coordinate = jnp.asarray(positions)
        guess_density = self._validated_initial_guess(initial_guess)
        overlap, core, nuclear, dense = self._integrals(coordinate)
        orthogonalizer, _ = _orthogonalizer(overlap, self.linear_dependence_tolerance)
        if self.reference is ElectronicReferenceKind.RESTRICTED:
            state = self._solve_restricted(
                coordinate,
                overlap,
                core,
                nuclear,
                dense,
                orthogonalizer,
                guess_density,
            )
            if not (self.stability.require_internal or self.stability.require_external):
                return state
            stability = self.stability_analysis(coordinate, state)
            stable = (
                stability.successful
                & (
                    stability.internal_stable
                    | ~jnp.asarray(self.stability.require_internal)
                )
                & (
                    stability.external_stable
                    | ~jnp.asarray(self.stability.require_external)
                )
            )
            evidence = SCFConvergenceEvidence(
                state.evidence.energy_residual,
                state.evidence.density_residual,
                state.evidence.commutator_residual,
                state.evidence.electron_count_residual,
                state.evidence.spin_residual,
                state.evidence.iterations,
                state.evidence.converged,
                stable,
                state.evidence.finite,
                state.evidence.plan_id,
            )
            return RestrictedMeanFieldState(
                state.density,
                state.coefficients,
                state.orbital_energies,
                state.occupations,
                state.fock,
                state.overlap,
                state.electronic_energy,
                state.total_energy,
                state.entropy,
                state.free_energy,
                evidence,
                reference=state.reference,
            )
        if self.reference is ElectronicReferenceKind.UNRESTRICTED:
            return self._solve_unrestricted(
                coordinate,
                overlap,
                core,
                nuclear,
                dense,
                orthogonalizer,
                guess_density,
            )
        if self.reference is ElectronicReferenceKind.RESTRICTED_OPEN_SHELL:
            return self._solve_rohf(
                coordinate,
                overlap,
                core,
                nuclear,
                dense,
                orthogonalizer,
                guess_density,
            )
        return self._solve_generalized(
            coordinate,
            overlap,
            core,
            nuclear,
            dense,
            orthogonalizer,
            guess_density,
        )

    def _solve_restricted(
        self, positions, overlap, core, nuclear, dense, orthogonalizer, guess_density
    ):
        energies, coefficients = self._initial_coefficients(core, overlap, orthogonalizer)
        occupations, _, entropy = _occupations(
            self.occupations, energies, self.sector.electron_count, 2.0
        )
        density = _density(coefficients, occupations)
        if guess_density is not None:
            if guess_density.shape != density.shape:
                raise ValueError("Restricted initial density has the wrong shape.")
            density = jnp.asarray(guess_density, dtype=density.dtype)
        elif self.guess.kind is InitialGuessKind.ZERO:
            density = jnp.zeros_like(density)
        previous_energy = jnp.asarray(jnp.inf, dtype=core.real.dtype)
        fock_history: list[Array] = []
        error_history: list[Array] = []
        energy_residual = density_residual = commutator_residual = jnp.asarray(
            jnp.inf, dtype=core.real.dtype
        )
        converged = False
        finite = jnp.asarray(True)
        electronic = jnp.asarray(jnp.nan, dtype=core.real.dtype)
        fock = core
        completed = 0
        for iteration in range(self.convergence.maximum_iterations):
            coulomb, exchange, jk_success = self._jk(positions, density, dense)
            fock = core + coulomb - 0.5 * exchange
            error = fock @ density @ overlap - overlap @ density @ fock
            fock_history.append(fock)
            error_history.append(error)
            diagonal_fock = fock
            if (
                SCFAccelerationKind.DIIS in self.acceleration.schedule
                and iteration >= self.acceleration.diis_start
                and len(fock_history) >= 2
            ):
                diagonal_fock = _diis_extrapolate(
                    fock_history, error_history, self.acceleration.diis_space
                )
            diagonal_fock = _level_shift_fock(
                diagonal_fock,
                overlap,
                coefficients,
                occupations,
                2.0,
                self.acceleration.level_shift,
            )
            previous_coefficients = coefficients
            previous_occupations = occupations
            energies, coefficients = _diagonalize(diagonal_fock, orthogonalizer)
            occupations, _, entropy = _updated_occupations(
                self.occupations,
                energies,
                coefficients,
                previous_coefficients,
                previous_occupations,
                overlap,
                self.sector.electron_count,
                2.0,
            )
            proposed = _density(coefficients, occupations)
            mixed = (
                self.acceleration.damping * density
                + (1.0 - self.acceleration.damping) * proposed
                if SCFAccelerationKind.DAMPING in self.acceleration.schedule
                and iteration < self.acceleration.diis_start
                else proposed
            )
            coulomb_new, exchange_new, new_success = self._jk(positions, mixed, dense)
            electronic = contract("ab,ab->", mixed, core) + 0.5 * contract(
                "ab,ab->", mixed, coulomb_new - 0.5 * exchange_new
            )
            total = electronic + nuclear
            energy_residual = jnp.abs(total - previous_energy)
            density_residual = jnp.max(jnp.abs(mixed - density), initial=0.0)
            commutator_residual = _commutator_norm(fock, mixed, overlap)
            finite = finite & jk_success & new_success & jnp.isfinite(total)
            density = mixed
            previous_energy = total
            completed = iteration + 1
            if bool(
                finite
                & (energy_residual <= self.convergence.energy_tolerance)
                & (density_residual <= self.convergence.density_tolerance)
                & (commutator_residual <= self.convergence.commutator_tolerance)
            ):
                converged = True
                break
        electron_residual = jnp.abs(
            jnp.real(contract("ab,ba->", density, overlap)) - self.sector.electron_count
        )
        stable = not (self.stability.require_internal or self.stability.require_external)
        evidence = SCFConvergenceEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            electron_residual,
            jnp.asarray(0.0, dtype=core.real.dtype),
            jnp.asarray(completed, dtype=jnp.int32),
            jnp.asarray(converged),
            jnp.asarray(stable),
            finite,
            self.convergence.plan_id,
        )
        free = electronic + nuclear - self.occupations.smearing_energy * entropy
        return RestrictedMeanFieldState(
            density,
            coefficients,
            energies,
            occupations,
            fock,
            overlap,
            electronic,
            electronic + nuclear,
            entropy,
            free,
            evidence,
        )

    def _solve_unrestricted(
        self, positions, overlap, core, nuclear, dense, orthogonalizer, guess_density
    ):
        alpha_energies, alpha_coefficients = self._initial_coefficients(
            core, overlap, orthogonalizer
        )
        beta_energies, beta_coefficients = alpha_energies, alpha_coefficients
        alpha_occupations, _, alpha_entropy = _occupations(
            self.occupations, alpha_energies, self.sector.alpha_electron_count, 1.0
        )
        beta_occupations, _, beta_entropy = _occupations(
            self.occupations, beta_energies, self.sector.beta_electron_count, 1.0
        )
        alpha_density = _density(alpha_coefficients, alpha_occupations)
        beta_density = _density(beta_coefficients, beta_occupations)
        if guess_density is not None:
            if guess_density.shape != (2,) + alpha_density.shape:
                raise ValueError(
                    "Unrestricted initial density must have shape (2, AO, AO)."
                )
            alpha_density = jnp.asarray(guess_density[0], dtype=alpha_density.dtype)
            beta_density = jnp.asarray(guess_density[1], dtype=beta_density.dtype)
        elif self.guess.kind is InitialGuessKind.ZERO:
            alpha_density = jnp.zeros_like(alpha_density)
            beta_density = jnp.zeros_like(beta_density)
        previous_energy = jnp.asarray(jnp.inf, dtype=core.real.dtype)
        fock_history: list[Array] = []
        error_history: list[Array] = []
        energy_residual = density_residual = commutator_residual = jnp.asarray(
            jnp.inf, dtype=core.real.dtype
        )
        finite = jnp.asarray(True)
        converged = False
        completed = 0
        alpha_fock = beta_fock = core
        electronic = jnp.asarray(jnp.nan, dtype=core.real.dtype)
        for iteration in range(self.convergence.maximum_iterations):
            total_density = alpha_density + beta_density
            coulomb, _, j_success = self._jk(positions, total_density, dense)
            _, alpha_exchange, ka_success = self._jk(positions, alpha_density, dense)
            _, beta_exchange, kb_success = self._jk(positions, beta_density, dense)
            alpha_fock = core + coulomb - alpha_exchange
            beta_fock = core + coulomb - beta_exchange
            alpha_error = (
                alpha_fock @ alpha_density @ overlap
                - overlap @ alpha_density @ alpha_fock
            )
            beta_error = (
                beta_fock @ beta_density @ overlap - overlap @ beta_density @ beta_fock
            )
            block_fock = jnp.block(
                [[alpha_fock, jnp.zeros_like(core)], [jnp.zeros_like(core), beta_fock]]
            )
            block_error = jnp.block(
                [[alpha_error, jnp.zeros_like(core)], [jnp.zeros_like(core), beta_error]]
            )
            fock_history.append(block_fock)
            error_history.append(block_error)
            alpha_diagonal_fock = alpha_fock
            beta_diagonal_fock = beta_fock
            if (
                SCFAccelerationKind.DIIS in self.acceleration.schedule
                and iteration >= self.acceleration.diis_start
                and len(fock_history) >= 2
            ):
                extrapolated = _diis_extrapolate(
                    fock_history, error_history, self.acceleration.diis_space
                )
                count = core.shape[0]
                alpha_diagonal_fock = extrapolated[:count, :count]
                beta_diagonal_fock = extrapolated[count:, count:]
            alpha_diagonal_fock = _level_shift_fock(
                alpha_diagonal_fock,
                overlap,
                alpha_coefficients,
                alpha_occupations,
                1.0,
                self.acceleration.level_shift,
            )
            beta_diagonal_fock = _level_shift_fock(
                beta_diagonal_fock,
                overlap,
                beta_coefficients,
                beta_occupations,
                1.0,
                self.acceleration.level_shift,
            )
            previous_alpha_coefficients = alpha_coefficients
            previous_beta_coefficients = beta_coefficients
            previous_alpha_occupations = alpha_occupations
            previous_beta_occupations = beta_occupations
            alpha_energies, alpha_coefficients = _diagonalize(
                alpha_diagonal_fock, orthogonalizer
            )
            beta_energies, beta_coefficients = _diagonalize(
                beta_diagonal_fock, orthogonalizer
            )
            alpha_occupations, _, alpha_entropy = _updated_occupations(
                self.occupations,
                alpha_energies,
                alpha_coefficients,
                previous_alpha_coefficients,
                previous_alpha_occupations,
                overlap,
                self.sector.alpha_electron_count,
                1.0,
            )
            beta_occupations, _, beta_entropy = _updated_occupations(
                self.occupations,
                beta_energies,
                beta_coefficients,
                previous_beta_coefficients,
                previous_beta_occupations,
                overlap,
                self.sector.beta_electron_count,
                1.0,
            )
            proposed_alpha = _density(alpha_coefficients, alpha_occupations)
            proposed_beta = _density(beta_coefficients, beta_occupations)
            if (
                SCFAccelerationKind.DAMPING in self.acceleration.schedule
                and iteration < self.acceleration.diis_start
            ):
                proposed_alpha = (
                    self.acceleration.damping * alpha_density
                    + (1.0 - self.acceleration.damping) * proposed_alpha
                )
                proposed_beta = (
                    self.acceleration.damping * beta_density
                    + (1.0 - self.acceleration.damping) * proposed_beta
                )
            total_new = proposed_alpha + proposed_beta
            coulomb_new, _, j_new = self._jk(positions, total_new, dense)
            _, alpha_exchange_new, ka_new = self._jk(positions, proposed_alpha, dense)
            _, beta_exchange_new, kb_new = self._jk(positions, proposed_beta, dense)
            electronic = (
                contract("ab,ab->", total_new, core)
                + 0.5 * contract("ab,ab->", total_new, coulomb_new)
                - 0.5 * contract("ab,ab->", proposed_alpha, alpha_exchange_new)
                - 0.5 * contract("ab,ab->", proposed_beta, beta_exchange_new)
            )
            total_energy = electronic + nuclear
            energy_residual = jnp.abs(total_energy - previous_energy)
            density_residual = jnp.maximum(
                jnp.max(jnp.abs(proposed_alpha - alpha_density), initial=0.0),
                jnp.max(jnp.abs(proposed_beta - beta_density), initial=0.0),
            )
            commutator_residual = jnp.maximum(
                _commutator_norm(alpha_fock, proposed_alpha, overlap),
                _commutator_norm(beta_fock, proposed_beta, overlap),
            )
            finite = (
                finite
                & j_success
                & ka_success
                & kb_success
                & j_new
                & ka_new
                & kb_new
                & jnp.isfinite(total_energy)
            )
            alpha_density, beta_density = proposed_alpha, proposed_beta
            previous_energy = total_energy
            completed = iteration + 1
            if bool(
                finite
                & (energy_residual <= self.convergence.energy_tolerance)
                & (density_residual <= self.convergence.density_tolerance)
                & (commutator_residual <= self.convergence.commutator_tolerance)
            ):
                converged = True
                break
        total_density = alpha_density + beta_density
        electron_residual = jnp.abs(
            jnp.real(contract("ab,ba->", total_density, overlap))
            - self.sector.electron_count
        )
        spin_population = jnp.real(
            contract("ab,ba->", alpha_density - beta_density, overlap)
        )
        spin_residual = jnp.abs(
            spin_population
            - (self.sector.alpha_electron_count - self.sector.beta_electron_count)
        )
        stable = not (self.stability.require_internal or self.stability.require_external)
        evidence = SCFConvergenceEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            electron_residual,
            spin_residual,
            jnp.asarray(completed, dtype=jnp.int32),
            jnp.asarray(converged),
            jnp.asarray(stable),
            finite,
            self.convergence.plan_id,
        )
        entropy = alpha_entropy + beta_entropy
        free = electronic + nuclear - self.occupations.smearing_energy * entropy
        return UnrestrictedMeanFieldState(
            alpha_density,
            beta_density,
            alpha_coefficients,
            beta_coefficients,
            alpha_energies,
            beta_energies,
            alpha_occupations,
            beta_occupations,
            alpha_fock,
            beta_fock,
            overlap,
            electronic,
            electronic + nuclear,
            entropy,
            free,
            evidence,
        )

    def _solve_rohf(
        self, positions, overlap, core, nuclear, dense, orthogonalizer, guess_density
    ):
        if self.occupations.kind is not ElectronicOccupationKind.INTEGER:
            raise ValueError("ROHF currently requires integer alpha/beta occupations.")
        _, coefficients = self._initial_coefficients(core, overlap, orthogonalizer)
        orbital_count = coefficients.shape[1]
        alpha_occupations = _integer_occupations(
            orbital_count, self.sector.alpha_electron_count, 1.0
        ).astype(core.real.dtype)
        beta_occupations = _integer_occupations(
            orbital_count, self.sector.beta_electron_count, 1.0
        ).astype(core.real.dtype)
        alpha_density = _density(coefficients, alpha_occupations)
        beta_density = _density(coefficients, beta_occupations)
        if guess_density is not None:
            if guess_density.shape != (2,) + alpha_density.shape:
                raise ValueError("ROHF initial density must have shape (2, AO, AO).")
            alpha_density = jnp.asarray(guess_density[0], dtype=alpha_density.dtype)
            beta_density = jnp.asarray(guess_density[1], dtype=beta_density.dtype)
        elif self.guess.kind is InitialGuessKind.ZERO:
            alpha_density = jnp.zeros_like(alpha_density)
            beta_density = jnp.zeros_like(beta_density)
        closed_count = self.sector.beta_electron_count
        open_count = self.sector.alpha_electron_count - self.sector.beta_electron_count
        closed = jnp.arange(closed_count, dtype=jnp.int32)
        open_ = jnp.arange(closed_count, closed_count + open_count, dtype=jnp.int32)
        virtual = jnp.arange(closed_count + open_count, orbital_count, dtype=jnp.int32)
        previous_energy = jnp.asarray(jnp.inf, dtype=core.real.dtype)
        energy_residual = density_residual = commutator_residual = jnp.asarray(
            jnp.inf, dtype=core.real.dtype
        )
        fock_history: list[Array] = []
        error_history: list[Array] = []
        electronic = jnp.asarray(jnp.nan, dtype=core.real.dtype)
        alpha_fock = beta_fock = core
        alpha_energies = beta_energies = jnp.zeros(
            (orbital_count,), dtype=core.real.dtype
        )
        converged = False
        finite = jnp.asarray(True)
        completed = 0
        for iteration in range(self.convergence.maximum_iterations):
            total_density = alpha_density + beta_density
            coulomb, _, success_j = self._jk(positions, total_density, dense)
            _, alpha_exchange, success_a = self._jk(positions, alpha_density, dense)
            _, beta_exchange, success_b = self._jk(positions, beta_density, dense)
            alpha_fock = core + coulomb - alpha_exchange
            beta_fock = core + coulomb - beta_exchange
            alpha_mo = jnp.conj(coefficients.T) @ alpha_fock @ coefficients
            beta_mo = jnp.conj(coefficients.T) @ beta_fock @ coefficients
            average_mo = 0.5 * (alpha_mo + beta_mo)
            effective_mo = average_mo
            if closed.size and open_.size:
                effective_mo = effective_mo.at[jnp.ix_(closed, open_)].set(
                    beta_mo[jnp.ix_(closed, open_)]
                )
                effective_mo = effective_mo.at[jnp.ix_(open_, closed)].set(
                    jnp.conj(beta_mo[jnp.ix_(closed, open_)].T)
                )
            if open_.size and virtual.size:
                effective_mo = effective_mo.at[jnp.ix_(open_, virtual)].set(
                    alpha_mo[jnp.ix_(open_, virtual)]
                )
                effective_mo = effective_mo.at[jnp.ix_(virtual, open_)].set(
                    jnp.conj(alpha_mo[jnp.ix_(open_, virtual)].T)
                )
            effective = (
                overlap @ coefficients @ effective_mo @ jnp.conj(coefficients.T) @ overlap
            )
            error = (
                effective @ total_density @ overlap - overlap @ total_density @ effective
            )
            fock_history.append(effective)
            error_history.append(error)
            if (
                SCFAccelerationKind.DIIS in self.acceleration.schedule
                and iteration >= self.acceleration.diis_start
                and len(fock_history) >= 2
            ):
                effective = _diis_extrapolate(
                    fock_history, error_history, self.acceleration.diis_space
                )
            effective = _level_shift_fock(
                effective,
                overlap,
                coefficients,
                alpha_occupations + beta_occupations,
                2.0,
                self.acceleration.level_shift,
            )
            _, proposed_coefficients = _diagonalize(effective, orthogonalizer)
            proposed_alpha = _density(proposed_coefficients, alpha_occupations)
            proposed_beta = _density(proposed_coefficients, beta_occupations)
            if (
                SCFAccelerationKind.DAMPING in self.acceleration.schedule
                and iteration < self.acceleration.diis_start
            ):
                proposed_alpha = (
                    self.acceleration.damping * alpha_density
                    + (1.0 - self.acceleration.damping) * proposed_alpha
                )
                proposed_beta = (
                    self.acceleration.damping * beta_density
                    + (1.0 - self.acceleration.damping) * proposed_beta
                )
            total_new = proposed_alpha + proposed_beta
            coulomb_new, _, success_j_new = self._jk(positions, total_new, dense)
            _, alpha_exchange_new, success_a_new = self._jk(
                positions, proposed_alpha, dense
            )
            _, beta_exchange_new, success_b_new = self._jk(
                positions, proposed_beta, dense
            )
            electronic = (
                contract("ab,ab->", total_new, core)
                + 0.5 * contract("ab,ab->", total_new, coulomb_new)
                - 0.5 * contract("ab,ab->", proposed_alpha, alpha_exchange_new)
                - 0.5 * contract("ab,ab->", proposed_beta, beta_exchange_new)
            )
            total_energy = electronic + nuclear
            energy_residual = jnp.abs(total_energy - previous_energy)
            density_residual = jnp.maximum(
                jnp.max(jnp.abs(proposed_alpha - alpha_density), initial=0.0),
                jnp.max(jnp.abs(proposed_beta - beta_density), initial=0.0),
            )
            commutator_residual = jnp.max(jnp.abs(error), initial=0.0)
            finite = (
                finite
                & success_j
                & success_a
                & success_b
                & success_j_new
                & success_a_new
                & success_b_new
                & jnp.isfinite(total_energy)
            )
            coefficients = proposed_coefficients
            alpha_density, beta_density = proposed_alpha, proposed_beta
            previous_energy = total_energy
            completed = iteration + 1
            if bool(
                finite
                & (energy_residual <= self.convergence.energy_tolerance)
                & (density_residual <= self.convergence.density_tolerance)
                & (commutator_residual <= self.convergence.commutator_tolerance)
            ):
                converged = True
                break
        alpha_energies = jnp.real(
            jnp.diag(jnp.conj(coefficients.T) @ alpha_fock @ coefficients)
        )
        beta_energies = jnp.real(
            jnp.diag(jnp.conj(coefficients.T) @ beta_fock @ coefficients)
        )
        total_density = alpha_density + beta_density
        electron_residual = jnp.abs(
            jnp.real(contract("ab,ba->", total_density, overlap))
            - self.sector.electron_count
        )
        spin_residual = jnp.abs(
            jnp.real(contract("ab,ba->", alpha_density - beta_density, overlap))
            - (self.sector.alpha_electron_count - self.sector.beta_electron_count)
        )
        evidence = SCFConvergenceEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            electron_residual,
            spin_residual,
            jnp.asarray(completed, dtype=jnp.int32),
            jnp.asarray(converged),
            jnp.asarray(
                not (self.stability.require_internal or self.stability.require_external)
            ),
            finite,
            self.convergence.plan_id,
        )
        return UnrestrictedMeanFieldState(
            alpha_density,
            beta_density,
            coefficients,
            coefficients,
            alpha_energies,
            beta_energies,
            alpha_occupations,
            beta_occupations,
            alpha_fock,
            beta_fock,
            overlap,
            electronic,
            electronic + nuclear,
            jnp.asarray(0.0, dtype=core.real.dtype),
            electronic + nuclear,
            evidence,
            reference=ElectronicReferenceKind.RESTRICTED_OPEN_SHELL,
        )

    def _solve_generalized(
        self, positions, overlap, core, nuclear, dense, orthogonalizer, guess_density
    ):
        count = core.shape[0]
        zero = jnp.zeros_like(core)
        spin_overlap = jnp.block([[overlap, zero], [zero, overlap]])
        spin_core = jnp.block([[core, zero], [zero, core]])
        spin_orthogonalizer = jnp.block(
            [
                [orthogonalizer, jnp.zeros_like(orthogonalizer)],
                [jnp.zeros_like(orthogonalizer), orthogonalizer],
            ]
        )
        spatial_energies, spatial_coefficients = self._initial_coefficients(
            core, overlap, orthogonalizer
        )
        coefficient_zero = jnp.zeros_like(spatial_coefficients)
        coefficients = jnp.block(
            [
                [spatial_coefficients, coefficient_zero],
                [coefficient_zero, spatial_coefficients],
            ]
        )
        energies = jnp.concatenate((spatial_energies, spatial_energies))
        initial_order = jnp.argsort(energies)
        energies = energies[initial_order]
        coefficients = coefficients[:, initial_order]
        occupations, _, _ = _occupations(
            self.occupations, energies, self.sector.electron_count, 1.0
        )
        density = _density(coefficients, occupations)
        if guess_density is not None:
            if guess_density.shape != density.shape:
                raise ValueError(
                    "Generalized initial density has the wrong spinor shape."
                )
            density = jnp.asarray(guess_density, dtype=density.dtype)
        elif self.guess.kind is InitialGuessKind.ZERO:
            density = jnp.zeros_like(density)
        previous_energy = jnp.asarray(jnp.inf, dtype=core.real.dtype)
        energy_residual = density_residual = commutator_residual = jnp.asarray(
            jnp.inf, dtype=core.real.dtype
        )
        converged = False
        finite = jnp.asarray(True)
        fock = spin_core
        electronic = jnp.asarray(jnp.nan, dtype=core.real.dtype)
        completed = 0
        fock_history: list[Array] = []
        error_history: list[Array] = []
        for iteration in range(self.convergence.maximum_iterations):
            blocks = density.reshape((2, count, 2, count)).transpose((0, 2, 1, 3))
            total_density = blocks[0, 0] + blocks[1, 1]
            coulomb, _, j_success = self._jk(positions, total_density, dense)
            exchange_blocks = []
            exchange_success = jnp.asarray(True)
            for spin_left in range(2):
                row = []
                for spin_right in range(2):
                    _, exchange, success = self._jk(
                        positions, blocks[spin_left, spin_right], dense
                    )
                    row.append(exchange)
                    exchange_success = exchange_success & success
                exchange_blocks.append(row)
            fock = jnp.block(
                [
                    [core + coulomb - exchange_blocks[0][0], -exchange_blocks[0][1]],
                    [-exchange_blocks[1][0], core + coulomb - exchange_blocks[1][1]],
                ]
            )
            error = fock @ density @ spin_overlap - spin_overlap @ density @ fock
            fock_history.append(fock)
            error_history.append(error)
            diagonal_fock = fock
            if (
                SCFAccelerationKind.DIIS in self.acceleration.schedule
                and iteration >= self.acceleration.diis_start
                and len(fock_history) >= 2
            ):
                diagonal_fock = _diis_extrapolate(
                    fock_history, error_history, self.acceleration.diis_space
                )
            diagonal_fock = _level_shift_fock(
                diagonal_fock,
                spin_overlap,
                coefficients,
                occupations,
                1.0,
                self.acceleration.level_shift,
            )
            previous_coefficients = coefficients
            previous_occupations = occupations
            energies, coefficients = _diagonalize(diagonal_fock, spin_orthogonalizer)
            occupations, _, _ = _updated_occupations(
                self.occupations,
                energies,
                coefficients,
                previous_coefficients,
                previous_occupations,
                spin_overlap,
                self.sector.electron_count,
                1.0,
            )
            proposed = _density(coefficients, occupations)
            if (
                SCFAccelerationKind.DAMPING in self.acceleration.schedule
                and iteration < self.acceleration.diis_start
            ):
                proposed = (
                    self.acceleration.damping * density
                    + (1.0 - self.acceleration.damping) * proposed
                )
            electronic = 0.5 * jnp.real(contract("ab,ba->", proposed, spin_core + fock))
            total_energy = electronic + nuclear
            energy_residual = jnp.abs(total_energy - previous_energy)
            density_residual = jnp.max(jnp.abs(proposed - density), initial=0.0)
            commutator_residual = _commutator_norm(fock, proposed, spin_overlap)
            finite = finite & j_success & exchange_success & jnp.isfinite(total_energy)
            density = proposed
            previous_energy = total_energy
            completed = iteration + 1
            if bool(
                finite
                & (energy_residual <= self.convergence.energy_tolerance)
                & (density_residual <= self.convergence.density_tolerance)
                & (commutator_residual <= self.convergence.commutator_tolerance)
            ):
                converged = True
                break
        electron_residual = jnp.abs(
            jnp.real(contract("ab,ba->", density, spin_overlap))
            - self.sector.electron_count
        )
        evidence = SCFConvergenceEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            electron_residual,
            jnp.asarray(0.0, dtype=core.real.dtype),
            jnp.asarray(completed, dtype=jnp.int32),
            jnp.asarray(converged),
            jnp.asarray(
                not (self.stability.require_internal or self.stability.require_external)
            ),
            finite,
            self.convergence.plan_id,
        )
        return GeneralizedMeanFieldState(
            density,
            coefficients,
            energies,
            occupations,
            fock,
            spin_overlap,
            electronic,
            electronic + nuclear,
            evidence,
        )

    def stability_analysis(
        self,
        positions: ArrayLike,
        state: RestrictedMeanFieldState,
        /,
    ) -> SCFStabilityResult:
        if not isinstance(state, RestrictedMeanFieldState) or (
            state.reference is not ElectronicReferenceKind.RESTRICTED
        ):
            raise TypeError("Current stability analysis requires a restricted state.")
        if self.occupations.kind is not ElectronicOccupationKind.INTEGER:
            raise ValueError("Orbital-Hessian stability requires integer occupations.")
        coordinate = jnp.asarray(positions)
        charges = jnp.asarray(self.system.atomic_numbers, dtype=coordinate.dtype)
        core = kinetic_matrix(self.basis, coordinate) + nuclear_attraction_matrix(
            self.basis, coordinate, charges
        )
        eri = electron_repulsion_tensor(self.basis, coordinate)
        occupied = tuple(np.flatnonzero(np.asarray(state.occupations) > 1.0))
        virtual = tuple(np.flatnonzero(np.asarray(state.occupations) < 1.0e-12))
        pairs = tuple(
            (virtual_, occupied_) for virtual_ in virtual for occupied_ in occupied
        )
        if not pairs:
            empty = jnp.zeros((0,), dtype=core.real.dtype)
            return SCFStabilityResult(
                empty,
                empty,
                jnp.asarray(True),
                jnp.asarray(True),
                state.evidence.converged,
                self.stability.plan_id,
                state.state_id,
            )
        orbital_count = state.coefficients.shape[1]
        jnp.eye(orbital_count, dtype=state.coefficients.dtype)

        def rotation(parameters, sign=1.0):
            generator = jnp.zeros(
                (orbital_count, orbital_count), dtype=state.coefficients.dtype
            )
            for index, (virtual_, occupied_) in enumerate(pairs):
                value = sign * parameters[index]
                generator = generator.at[virtual_, occupied_].set(value)
                generator = generator.at[occupied_, virtual_].set(-jnp.conj(value))
            return jsp.linalg.expm(generator)

        def restricted_energy(parameters):
            coefficients = state.coefficients @ rotation(parameters)
            occupied_coefficients = coefficients[:, : len(occupied)]
            density = 2.0 * occupied_coefficients @ jnp.conj(occupied_coefficients.T)
            coulomb = contract("cd,abcd->ab", density, eri)
            exchange = contract("cd,acbd->ab", density, eri)
            return jnp.real(
                contract("ab,ab->", density, core)
                + 0.5 * contract("ab,ab->", density, coulomb - 0.5 * exchange)
            )

        def unrestricted_energy(parameters):
            alpha_coefficients = state.coefficients @ rotation(parameters, 1.0)
            beta_coefficients = state.coefficients @ rotation(parameters, -1.0)
            alpha_occupied = alpha_coefficients[:, : len(occupied)]
            beta_occupied = beta_coefficients[:, : len(occupied)]
            alpha = alpha_occupied @ jnp.conj(alpha_occupied.T)
            beta = beta_occupied @ jnp.conj(beta_occupied.T)
            total = alpha + beta
            coulomb = contract("cd,abcd->ab", total, eri)
            alpha_exchange = contract("cd,acbd->ab", alpha, eri)
            beta_exchange = contract("cd,acbd->ab", beta, eri)
            return jnp.real(
                contract("ab,ab->", total, core)
                + 0.5 * contract("ab,ab->", total, coulomb)
                - 0.5 * contract("ab,ab->", alpha, alpha_exchange)
                - 0.5 * contract("ab,ab->", beta, beta_exchange)
            )

        zero = jnp.zeros((len(pairs),), dtype=core.real.dtype)
        internal_hessian = jax.hessian(restricted_energy)(zero)
        external_hessian = jax.hessian(unrestricted_energy)(zero)
        internal_values, _ = _hermitian_eigh(internal_hessian)
        external_values, _ = _hermitian_eigh(external_hessian)
        tolerance = self.stability.eigenvalue_tolerance
        internal_stable = jnp.min(internal_values) >= -tolerance
        external_stable = jnp.min(external_values) >= -tolerance
        finite = (
            state.evidence.converged
            & jnp.all(jnp.isfinite(internal_values))
            & jnp.all(jnp.isfinite(external_values))
        )
        return SCFStabilityResult(
            internal_values,
            external_values,
            internal_stable,
            external_stable,
            finite,
            self.stability.plan_id,
            state.state_id,
        )

    def analytic_gradient_atomic_units(
        self, positions: ArrayLike, state, /
    ) -> MolecularGradientResult:
        coordinate = jnp.asarray(positions)
        if isinstance(state, GeneralizedMeanFieldState):
            raise ValueError(
                "GHF analytic gradients require spinor response and are not admitted."
            )
        charges = jnp.asarray(self.system.atomic_numbers, dtype=coordinate.dtype)
        derivative_overlap = jax.jacfwd(lambda value: overlap_matrix(self.basis, value))(
            coordinate
        )
        derivative_core = jax.jacfwd(
            lambda value: (
                kinetic_matrix(self.basis, value)
                + nuclear_attraction_matrix(self.basis, value, charges)
            )
        )(coordinate)
        derivative_eri = jax.jacfwd(
            lambda value: electron_repulsion_tensor(self.basis, value)
        )(coordinate)
        derivative_nuclear = jax.jacfwd(
            lambda value: nuclear_repulsion_energy(value, charges)
        )(coordinate)
        if isinstance(state, RestrictedMeanFieldState):
            density = state.density
            energy_weighted = contract(
                "pi,i,qi->pq",
                state.coefficients,
                state.occupations * state.orbital_energies,
                jnp.conj(state.coefficients),
            )
            one = contract("ab,abNx->Nx", density, derivative_core)
            coulomb = 0.5 * contract("ab,cd,abcdNx->Nx", density, density, derivative_eri)
            exchange = -0.25 * contract(
                "ac,bd,abcdNx->Nx", density, density, derivative_eri
            )
            two = coulomb + exchange
        else:
            alpha, beta = state.alpha_density, state.beta_density
            density = alpha + beta
            energy_weighted = contract(
                "pi,i,qi->pq",
                state.alpha_coefficients,
                state.alpha_occupations * state.alpha_orbital_energies,
                jnp.conj(state.alpha_coefficients),
            ) + contract(
                "pi,i,qi->pq",
                state.beta_coefficients,
                state.beta_occupations * state.beta_orbital_energies,
                jnp.conj(state.beta_coefficients),
            )
            one = contract("ab,abNx->Nx", density, derivative_core)
            coulomb = 0.5 * contract("ab,cd,abcdNx->Nx", density, density, derivative_eri)
            exchange = -0.5 * (
                contract("ac,bd,abcdNx->Nx", alpha, alpha, derivative_eri)
                + contract("ac,bd,abcdNx->Nx", beta, beta, derivative_eri)
            )
            two = coulomb + exchange
        pulay = -contract("ab,abNx->Nx", energy_weighted, derivative_overlap)
        gradient = jnp.real(one + two + pulay + derivative_nuclear)
        finite = state.evidence.converged & jnp.all(jnp.isfinite(gradient))
        return MolecularGradientResult(
            gradient,
            -gradient,
            jnp.real(pulay),
            jnp.real(one),
            jnp.real(two),
            jnp.real(derivative_nuclear),
            finite,
            state.state_id,
            self.plan_id,
        )

    def evaluate(self, positions: ArrayLike, /):
        coordinate = jnp.asarray(positions)
        length_to_bohr = float(
            conversion_factor(self.system.units.scale.length_unit, BOHR)
        )
        energy_from_hartree = float(
            conversion_factor(HARTREE, self.system.units.scale.energy_unit)
        )
        coordinate_bohr = coordinate * length_to_bohr
        state = self.solve_atomic_units(coordinate_bohr)
        gradient = self.analytic_gradient_atomic_units(coordinate_bohr, state)
        force_factor = energy_from_hartree * length_to_bohr
        return state, gradient.forces * force_factor


__all__ = ["MolecularGradientResult", "MolecularHartreeFockPlan"]
