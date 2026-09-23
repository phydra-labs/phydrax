#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""General-basis RKS and UKS with moving atom-centered LDA/GGA grids."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan
from ...ein import contract
from ...operators.quantum.gaussian import (
    ao_gradients,
    ao_values,
    electron_repulsion_tensor,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion_energy,
    overlap_matrix,
    PreparedGaussianBasis,
    range_separated_electron_repulsion_tensor,
)
from ...units import BOHR, conversion_factor, HARTREE
from .._context import ElectronicInitialGuessState
from .._method import DensityFunctionalPlan
from .._model import ElectronicReferenceKind
from .._state import MolecularElectronicSectorPlan, PreparedMolecularElectronicSector
from ._functional import NativeXCFunctional
from ._grid import MolecularDFTGridPlan, PreparedMolecularDFTGrid
from ._mean_field import (
    ElectronicOccupationPlan,
    InitialGuessKind,
    InitialGuessPlan,
    mean_field_owner_id,
    RestrictedMeanFieldState,
    SCFAccelerationKind,
    SCFAccelerationPlan,
    SCFConvergenceEvidence,
    SCFConvergencePlan,
    UnrestrictedMeanFieldState,
)
from ._molecular_hf import (
    _commutator_norm,
    _density,
    _diagonalize,
    _diis_extrapolate,
    _level_shift_fock,
    _occupations,
    _orthogonalizer,
    _updated_occupations,
    MolecularGradientResult,
)


class MolecularKohnShamPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    basis: PreparedGaussianBasis
    sector: PreparedMolecularElectronicSector
    reference: ElectronicReferenceKind = eqx.field(static=True)
    functional: NativeXCFunctional
    grid: PreparedMolecularDFTGrid
    convergence: SCFConvergencePlan
    acceleration: SCFAccelerationPlan
    occupations: ElectronicOccupationPlan
    guess: InitialGuessPlan
    linear_dependence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        basis: PreparedGaussianBasis,
        sector: MolecularElectronicSectorPlan | PreparedMolecularElectronicSector,
        functional: DensityFunctionalPlan | NativeXCFunctional,
        grid: MolecularDFTGridPlan | PreparedMolecularDFTGrid,
        reference: ElectronicReferenceKind = ElectronicReferenceKind.RESTRICTED,
        /,
        *,
        convergence: SCFConvergencePlan | None = None,
        acceleration: SCFAccelerationPlan | None = None,
        occupations: ElectronicOccupationPlan | None = None,
        guess: InitialGuessPlan | None = None,
        linear_dependence_tolerance: float = 1.0e-9,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if (
            not isinstance(basis, PreparedGaussianBasis)
            or basis.system_id != system.system_id
        ):
            raise ValueError("basis must be prepared for the Kohn-Sham system.")
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
                "Electronic sector must be prepared for the Kohn-Sham system."
            )
        functional_ = (
            functional
            if isinstance(functional, NativeXCFunctional)
            else NativeXCFunctional(functional)
        )
        grid_ = grid.prepare() if isinstance(grid, MolecularDFTGridPlan) else grid
        if (
            not isinstance(grid_, PreparedMolecularDFTGrid)
            or grid_.plan.system.system_id != system.system_id
        ):
            raise ValueError("DFT grid must be prepared for the Kohn-Sham system.")
        if reference not in (
            ElectronicReferenceKind.RESTRICTED,
            ElectronicReferenceKind.UNRESTRICTED,
        ):
            raise ValueError("Native molecular Kohn-Sham currently supports RKS and UKS.")
        if (
            reference is ElectronicReferenceKind.RESTRICTED
            and sector_.alpha_electron_count != sector_.beta_electron_count
        ):
            raise ValueError("RKS requires equal alpha and beta populations.")
        convergence_ = SCFConvergencePlan() if convergence is None else convergence
        acceleration_ = SCFAccelerationPlan() if acceleration is None else acceleration
        occupations_ = ElectronicOccupationPlan() if occupations is None else occupations
        guess_ = InitialGuessPlan() if guess is None else guess
        tolerance = float(linear_dependence_tolerance)
        if not isinstance(convergence_, SCFConvergencePlan) or not isinstance(
            acceleration_, SCFAccelerationPlan
        ):
            raise TypeError("convergence and acceleration must be typed SCF plans.")
        if not isinstance(occupations_, ElectronicOccupationPlan) or not isinstance(
            guess_, InitialGuessPlan
        ):
            raise TypeError("occupations and guess must be typed mean-field plans.")
        if guess_.kind in (InitialGuessKind.SAD, InitialGuessKind.SAP):
            raise NotImplementedError(
                "SAD/SAP require a governed atomic guess provider; use an explicit guess state."
            )
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("linear_dependence_tolerance must be positive finite.")
        self.system = system
        self.basis = basis
        self.sector = sector_
        self.reference = reference
        self.functional = functional_
        self.grid = grid_
        self.convergence = convergence_
        self.acceleration = acceleration_
        self.occupations = occupations_
        self.guess = guess_
        self.linear_dependence_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-kohn-sham-plan",
                "system": system.system_id,
                "basis": basis.prepared_id,
                "sector": sector_.prepared_id,
                "reference": reference.value,
                "functional": functional_.functional_id,
                "grid": grid_.prepared_id,
                "convergence": convergence_.plan_id,
                "acceleration": acceleration_.plan_id,
                "occupations": occupations_.plan_id,
                "guess": guess_.plan_id,
                "linear_dependence_tolerance": tolerance,
            }
        )

    def exchange_tensor(
        self, positions: Array, full_eri: Array | None = None, /
    ) -> Array:
        full = (
            electron_repulsion_tensor(self.basis, positions)
            if full_eri is None
            else full_eri
        )
        result = self.functional.plan.exact_exchange_fraction * full
        if self.functional.plan.long_range_exchange_fraction:
            long_range = range_separated_electron_repulsion_tensor(
                self.basis,
                positions,
                self.functional.plan.range_separation,
            )
            result = (
                result + self.functional.plan.long_range_exchange_fraction * long_range
            )
        return result

    def _initial_coefficients(
        self,
        core: Array,
        overlap: Array,
        orthogonalizer: Array,
        /,
    ):
        if self.guess.kind is InitialGuessKind.HUCKEL:
            diagonal = jnp.diag(core)
            guess_hamiltonian = 0.875 * (diagonal[:, None] + diagonal[None, :]) * overlap
            guess_hamiltonian = guess_hamiltonian.at[jnp.diag_indices(core.shape[0])].set(
                diagonal
            )
        else:
            guess_hamiltonian = core
        return _diagonalize(guess_hamiltonian, orthogonalizer)

    def _fixed_quantities(self, positions: Array, /):
        charges = jnp.asarray(self.system.atomic_numbers, dtype=positions.dtype)
        overlap = overlap_matrix(self.basis, positions)
        core = kinetic_matrix(self.basis, positions) + nuclear_attraction_matrix(
            self.basis, positions, charges
        )
        eri = electron_repulsion_tensor(self.basis, positions)
        exchange_eri = self.exchange_tensor(positions, eri)
        nuclear = nuclear_repulsion_energy(positions, charges)
        grid = self.grid.evaluate(positions)
        ao = ao_values(self.basis, positions, grid.points)
        gradient = ao_gradients(self.basis, positions, grid.points)
        return overlap, core, eri, exchange_eri, nuclear, grid, ao, gradient

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
                "Initial guess basis, sector, or source identity differs from the KS plan."
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
        (
            overlap,
            core,
            eri,
            exchange_eri,
            nuclear,
            grid,
            ao,
            gradient,
        ) = self._fixed_quantities(coordinate)
        if not bool(grid.successful):
            raise ValueError("Moving molecular DFT grid failed geometric admission.")
        orthogonalizer, _ = _orthogonalizer(overlap, self.linear_dependence_tolerance)
        if self.reference is ElectronicReferenceKind.RESTRICTED:
            return self._solve_restricted(
                coordinate,
                overlap,
                core,
                eri,
                exchange_eri,
                nuclear,
                grid.weights,
                ao,
                gradient,
                orthogonalizer,
                guess_density,
            )
        return self._solve_unrestricted(
            coordinate,
            overlap,
            core,
            eri,
            exchange_eri,
            nuclear,
            grid.weights,
            ao,
            gradient,
            orthogonalizer,
            guess_density,
        )

    def _solve_restricted(
        self,
        coordinate,
        overlap,
        core,
        eri,
        exchange_eri,
        nuclear,
        weights,
        ao,
        gradient,
        orthogonalizer,
        guess_density,
    ):
        orbital_energies, coefficients = self._initial_coefficients(
            core, overlap, orthogonalizer
        )
        occupations, _, entropy = _occupations(
            self.occupations, orbital_energies, self.sector.electron_count, 2.0
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
        electronic = xc_energy = jnp.asarray(jnp.nan, dtype=core.real.dtype)
        fock = core
        converged = False
        completed = 0
        finite = jnp.asarray(True)
        for iteration in range(self.convergence.maximum_iterations):
            alpha = beta = 0.5 * density
            xc_energy, alpha_potential, beta_potential = (
                self.functional.potential_matrices(alpha, beta, ao, gradient, weights)
            )
            potential = 0.5 * (alpha_potential + beta_potential)
            coulomb = contract("cd,abcd->ab", density, eri)
            exchange = contract("cd,acbd->ab", density, exchange_eri)
            fock = core + coulomb - 0.5 * exchange + potential
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
            orbital_energies, coefficients = _diagonalize(diagonal_fock, orthogonalizer)
            occupations, _, entropy = _updated_occupations(
                self.occupations,
                orbital_energies,
                coefficients,
                previous_coefficients,
                previous_occupations,
                overlap,
                self.sector.electron_count,
                2.0,
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
            alpha_new = beta_new = 0.5 * proposed
            xc_energy, _, _ = self.functional.potential_matrices(
                alpha_new, beta_new, ao, gradient, weights
            )
            coulomb_new = contract("cd,abcd->ab", proposed, eri)
            exchange_new = contract("cd,acbd->ab", proposed, exchange_eri)
            electronic = (
                contract("ab,ab->", proposed, core)
                + 0.5 * contract("ab,ab->", proposed, coulomb_new)
                - 0.25 * contract("ab,ab->", proposed, exchange_new)
                + xc_energy
            )
            total = electronic + nuclear
            energy_residual = jnp.abs(total - previous_energy)
            density_residual = jnp.max(jnp.abs(proposed - density), initial=0.0)
            commutator_residual = _commutator_norm(fock, proposed, overlap)
            finite = finite & jnp.isfinite(total) & jnp.all(jnp.isfinite(fock))
            density = proposed
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
        evidence = SCFConvergenceEvidence(
            energy_residual,
            density_residual,
            commutator_residual,
            electron_residual,
            jnp.asarray(0.0, dtype=core.real.dtype),
            jnp.asarray(completed, dtype=jnp.int32),
            jnp.asarray(converged),
            jnp.asarray(True),
            finite,
            self.convergence.plan_id,
        )
        free = electronic + nuclear - self.occupations.smearing_energy * entropy
        return RestrictedMeanFieldState(
            density,
            coefficients,
            orbital_energies,
            occupations,
            fock,
            overlap,
            electronic,
            electronic + nuclear,
            entropy,
            free,
            evidence,
            mean_field_owner_id(self.plan_id, coordinate),
        )

    def _solve_unrestricted(
        self,
        coordinate,
        overlap,
        core,
        eri,
        exchange_eri,
        nuclear,
        weights,
        ao,
        gradient,
        orthogonalizer,
        guess_density,
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
        alpha = _density(alpha_coefficients, alpha_occupations)
        beta = _density(beta_coefficients, beta_occupations)
        if guess_density is not None:
            if guess_density.shape != (2,) + alpha.shape:
                raise ValueError(
                    "Unrestricted initial density must have shape (2, AO, AO)."
                )
            alpha = jnp.asarray(guess_density[0], dtype=alpha.dtype)
            beta = jnp.asarray(guess_density[1], dtype=beta.dtype)
        elif self.guess.kind is InitialGuessKind.ZERO:
            alpha = jnp.zeros_like(alpha)
            beta = jnp.zeros_like(beta)
        previous_energy = jnp.asarray(jnp.inf, dtype=core.real.dtype)
        energy_residual = density_residual = commutator_residual = jnp.asarray(
            jnp.inf, dtype=core.real.dtype
        )
        alpha_fock = beta_fock = core
        electronic = jnp.asarray(jnp.nan, dtype=core.real.dtype)
        converged = False
        finite = jnp.asarray(True)
        completed = 0
        fock_history: list[Array] = []
        error_history: list[Array] = []
        for iteration in range(self.convergence.maximum_iterations):
            total_density = alpha + beta
            coulomb = contract("cd,abcd->ab", total_density, eri)
            alpha_exchange = contract("cd,acbd->ab", alpha, exchange_eri)
            beta_exchange = contract("cd,acbd->ab", beta, exchange_eri)
            _, alpha_potential, beta_potential = self.functional.potential_matrices(
                alpha, beta, ao, gradient, weights
            )
            alpha_fock = core + coulomb - alpha_exchange + alpha_potential
            beta_fock = core + coulomb - beta_exchange + beta_potential
            alpha_error = alpha_fock @ alpha @ overlap - overlap @ alpha @ alpha_fock
            beta_error = beta_fock @ beta @ overlap - overlap @ beta @ beta_fock
            zero = jnp.zeros_like(core)
            block_fock = jnp.block([[alpha_fock, zero], [zero, beta_fock]])
            block_error = jnp.block([[alpha_error, zero], [zero, beta_error]])
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
                    self.acceleration.damping * alpha
                    + (1.0 - self.acceleration.damping) * proposed_alpha
                )
                proposed_beta = (
                    self.acceleration.damping * beta
                    + (1.0 - self.acceleration.damping) * proposed_beta
                )
            total_new = proposed_alpha + proposed_beta
            coulomb_new = contract("cd,abcd->ab", total_new, eri)
            alpha_exchange_new = contract("cd,acbd->ab", proposed_alpha, exchange_eri)
            beta_exchange_new = contract("cd,acbd->ab", proposed_beta, exchange_eri)
            xc_energy, _, _ = self.functional.potential_matrices(
                proposed_alpha, proposed_beta, ao, gradient, weights
            )
            electronic = (
                contract("ab,ab->", total_new, core)
                + 0.5 * contract("ab,ab->", total_new, coulomb_new)
                - 0.5 * contract("ab,ab->", proposed_alpha, alpha_exchange_new)
                - 0.5 * contract("ab,ab->", proposed_beta, beta_exchange_new)
                + xc_energy
            )
            total = electronic + nuclear
            energy_residual = jnp.abs(total - previous_energy)
            density_residual = jnp.maximum(
                jnp.max(jnp.abs(proposed_alpha - alpha), initial=0.0),
                jnp.max(jnp.abs(proposed_beta - beta), initial=0.0),
            )
            commutator_residual = jnp.maximum(
                _commutator_norm(alpha_fock, proposed_alpha, overlap),
                _commutator_norm(beta_fock, proposed_beta, overlap),
            )
            finite = (
                finite
                & jnp.isfinite(total)
                & jnp.all(jnp.isfinite(alpha_fock))
                & jnp.all(jnp.isfinite(beta_fock))
            )
            alpha, beta = proposed_alpha, proposed_beta
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
        total_density = alpha + beta
        electron_residual = jnp.abs(
            jnp.real(contract("ab,ba->", total_density, overlap))
            - self.sector.electron_count
        )
        spin_residual = jnp.abs(
            jnp.real(contract("ab,ba->", alpha - beta, overlap))
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
            jnp.asarray(True),
            finite,
            self.convergence.plan_id,
        )
        entropy = alpha_entropy + beta_entropy
        free = electronic + nuclear - self.occupations.smearing_energy * entropy
        return UnrestrictedMeanFieldState(
            alpha,
            beta,
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
            mean_field_owner_id(self.plan_id, coordinate),
        )

    def analytic_gradient_atomic_units(
        self, positions: ArrayLike, state, /
    ) -> MolecularGradientResult:
        coordinate = jnp.asarray(positions)
        if state.owner_id != mean_field_owner_id(self.plan_id, coordinate):
            raise ValueError("Gradient state belongs to another KS plan or geometry.")
        if isinstance(state, RestrictedMeanFieldState):
            alpha = beta = 0.5 * state.density
            energy_weighted = contract(
                "pi,i,qi->pq",
                state.coefficients,
                state.occupations * state.orbital_energies,
                jnp.conj(state.coefficients),
            )
        elif isinstance(state, UnrestrictedMeanFieldState):
            alpha, beta = state.alpha_density, state.beta_density
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
        else:
            raise TypeError("Kohn-Sham gradient requires an RKS or UKS state.")
        charges = jnp.asarray(self.system.atomic_numbers, dtype=coordinate.dtype)

        def stationary_energy(value):
            overlap = overlap_matrix(self.basis, value)
            del overlap
            core = kinetic_matrix(self.basis, value) + nuclear_attraction_matrix(
                self.basis, value, charges
            )
            eri = electron_repulsion_tensor(self.basis, value)
            exchange_eri = self.exchange_tensor(value, eri)
            grid = self.grid.evaluate(value)
            ao = ao_values(self.basis, value, grid.points)
            gradient = ao_gradients(self.basis, value, grid.points)
            total_density = alpha + beta
            coulomb = contract("cd,abcd->ab", total_density, eri)
            alpha_exchange = contract("cd,acbd->ab", alpha, exchange_eri)
            beta_exchange = contract("cd,acbd->ab", beta, exchange_eri)
            xc = self.functional.energy(alpha, beta, ao, gradient, grid.weights)
            return jnp.real(
                contract("ab,ab->", total_density, core)
                + 0.5 * contract("ab,ab->", total_density, coulomb)
                - 0.5 * contract("ab,ab->", alpha, alpha_exchange)
                - 0.5 * contract("ab,ab->", beta, beta_exchange)
                + xc
                + nuclear_repulsion_energy(value, charges)
            )

        partial = jax.grad(stationary_energy)(coordinate)
        derivative_overlap = jax.jacfwd(lambda value: overlap_matrix(self.basis, value))(
            coordinate
        )
        pulay = -jnp.real(contract("ab,abNx->Nx", energy_weighted, derivative_overlap))
        total = partial + pulay
        zero = jnp.zeros_like(total)
        successful = state.evidence.converged & jnp.all(jnp.isfinite(total))
        return MolecularGradientResult(
            total,
            -total,
            pulay,
            partial,
            zero,
            zero,
            successful,
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
        return state, gradient.forces * energy_from_hartree * length_to_bohr


__all__ = ["MolecularKohnShamPlan"]
