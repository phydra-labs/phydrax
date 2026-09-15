#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native restricted Hartree--Fock over contracted s-Gaussian orbitals."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticSystemPlan
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ...operators.quantum.gaussian import (
    molecular_integrals,
    nuclear_point_charge_energy,
    point_charge_potential_matrix,
    PreparedGaussianBasis,
)
from ...units import BOHR, conversion_factor, derived_unit, ELEMENTARY_CHARGE, HARTREE
from .._calculation import ElectronicCalculationPlan, make_electronic_evaluation
from .._kernel import ElectronicKernelEvaluation
from .._model import ElectronicMethodFamily, ElectronicReferenceKind
from .._provider import (
    AbstractElectronicProvider,
    AbstractPreparedElectronicCalculation,
    ElectronicCapabilityError,
    ElectronicProviderCapabilities,
)
from .._result import (
    ElectronicCalculationStatus,
    ElectronicConvergenceEvidence,
    ElectronicEvaluation,
    ElectronicWorkEvidence,
)
from .._task import ElectronicProperty
from ..excited._tda import ExcitedStateManifoldPlan, TammDancoffPlan


class SCFState(StrictModule, NonTrainableState):
    density: Array
    coefficients: Array
    orbital_energies: Array
    electronic_energy: Array
    total_energy: Array
    residual: Array
    iterations: Array
    converged: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        coefficients: ArrayLike,
        orbital_energies: ArrayLike,
        electronic_energy: ArrayLike,
        total_energy: ArrayLike,
        residual: ArrayLike,
        iterations: int,
        converged: ArrayLike,
        /,
    ):
        density_ = jnp.asarray(density)
        coefficients_ = jnp.asarray(coefficients, dtype=density_.dtype)
        orbital_energies_ = jnp.asarray(orbital_energies, dtype=density_.dtype)
        if density_.ndim != 2 or density_.shape[0] != density_.shape[1]:
            raise ValueError("SCF density must be square.")
        if coefficients_.shape != density_.shape or orbital_energies_.shape != (
            density_.shape[0],
        ):
            raise ValueError("SCF coefficient and orbital axes must match the AO basis.")
        self.density = density_
        self.coefficients = coefficients_
        self.orbital_energies = orbital_energies_
        self.electronic_energy = jnp.asarray(
            electronic_energy, dtype=density_.dtype
        ).reshape(())
        self.total_energy = jnp.asarray(total_energy, dtype=density_.dtype).reshape(())
        self.residual = jnp.asarray(residual, dtype=density_.dtype).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.converged = jnp.asarray(converged, dtype=bool).reshape(())
        self.state_id = canonical_fingerprint(
            {
                "kind": "native-rhf-state",
                "arrays": array_tree_fingerprint(
                    {
                        "density": np.asarray(density_),
                        "coefficients": np.asarray(coefficients_),
                        "orbital_energies": np.asarray(orbital_energies_),
                        "electronic_energy": np.asarray(self.electronic_energy),
                        "total_energy": np.asarray(self.total_energy),
                        "residual": np.asarray(self.residual),
                        "iterations": np.asarray(self.iterations),
                        "converged": np.asarray(self.converged),
                    }
                ),
            }
        )


def _symmetric_eigh(matrix: Array, /) -> tuple[Array, Array]:
    count = int(matrix.shape[0])
    result = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                0.5 * (matrix + matrix.T),
                properties=OperatorProperties(
                    self_adjoint=True, evidence={"self_adjoint": "construction"}
                ),
            )
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=count, which="smallest-algebraic"),
    )
    if not bool(result.successful):
        raise RuntimeError("Native symmetric eigensolve failed.")
    return result.eigenvalues, result.eigenvectors


class NativeRHFPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    basis: PreparedGaussianBasis
    electron_count: int = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    linear_dependence_tolerance: float = eqx.field(static=True)
    force_displacement: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        basis: PreparedGaussianBasis,
        electron_count: int,
        /,
        *,
        convergence_tolerance: float = 1.0e-10,
        maximum_iterations: int = 128,
        damping: float = 0.25,
        linear_dependence_tolerance: float = 1.0e-9,
        force_displacement: float = 1.0e-4,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if not isinstance(basis, PreparedGaussianBasis):
            raise TypeError("basis must be PreparedGaussianBasis.")
        if basis.system_id != system.system_id:
            raise ValueError("Gaussian basis belongs to another system.")
        if not bool(np.all(np.asarray(system.active_mask))):
            raise ValueError(
                "Native RHF currently requires an unpadded all-active system."
            )
        if not bool(np.all(np.asarray(system.element_mask))):
            raise ValueError("Native RHF requires every active site to be an element.")
        electrons = int(electron_count)
        if electrons <= 0 or electrons % 2 or electrons > 2 * basis.basis_function_count:
            raise ValueError(
                "Native RHF requires a positive even electron count within AO capacity."
            )
        tolerance = float(convergence_tolerance)
        iterations = int(maximum_iterations)
        damping_ = float(damping)
        rank_tolerance = float(linear_dependence_tolerance)
        displacement = float(force_displacement)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (tolerance, rank_tolerance, displacement)
        ):
            raise ValueError("RHF tolerances and displacement must be positive finite.")
        if iterations <= 0 or not isfinite(damping_) or damping_ < 0.0 or damping_ >= 1.0:
            raise ValueError("RHF iteration count or damping is invalid.")
        self.system = system
        self.basis = basis
        self.electron_count = electrons
        self.convergence_tolerance = tolerance
        self.maximum_iterations = iterations
        self.damping = damping_
        self.linear_dependence_tolerance = rank_tolerance
        self.force_displacement = displacement
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-rhf-plan",
                "system": system.system_id,
                "basis": basis.prepared_id,
                "electron_count": electrons,
                "convergence_tolerance": tolerance,
                "maximum_iterations": iterations,
                "damping": damping_,
                "linear_dependence_tolerance": rank_tolerance,
                "force_displacement": displacement,
            }
        )

    def solve_atomic_units(
        self,
        positions_bohr: ArrayLike,
        /,
        *,
        embedding_positions_bohr: ArrayLike | None = None,
        embedding_charges: ArrayLike | None = None,
    ) -> SCFState:
        positions = jnp.asarray(positions_bohr)
        charges = jnp.asarray(self.system.atomic_numbers, dtype=positions.dtype)
        integrals = molecular_integrals(self.basis, positions, charges)
        core_hamiltonian = integrals.core_hamiltonian
        nuclear_repulsion = integrals.nuclear_repulsion
        if (embedding_positions_bohr is None) != (embedding_charges is None):
            raise ValueError("Embedding positions and charges must be provided together.")
        if embedding_positions_bohr is not None and embedding_charges is not None:
            point_positions = jnp.asarray(embedding_positions_bohr, dtype=positions.dtype)
            point_charges = jnp.asarray(embedding_charges, dtype=positions.dtype)
            core_hamiltonian = core_hamiltonian + point_charge_potential_matrix(
                self.basis,
                positions,
                point_positions,
                point_charges,
            )
            nuclear_repulsion = nuclear_repulsion + nuclear_point_charge_energy(
                positions,
                charges,
                point_positions,
                point_charges,
            )
        overlap_values, overlap_vectors = _symmetric_eigh(integrals.overlap)
        if bool(jnp.min(overlap_values) <= self.linear_dependence_tolerance):
            raise ValueError("Gaussian overlap matrix is numerically rank deficient.")
        orthogonalizer = (
            overlap_vectors @ jnp.diag(overlap_values**-0.5) @ overlap_vectors.T
        )
        occupied = self.electron_count // 2
        density = jnp.zeros_like(integrals.overlap)
        previous_energy = jnp.asarray(jnp.inf, dtype=positions.dtype)
        converged = False
        residual = jnp.asarray(jnp.inf, dtype=positions.dtype)
        coefficients = jnp.eye(self.basis.basis_function_count, dtype=positions.dtype)
        orbital_energies = jnp.zeros(
            (self.basis.basis_function_count,), dtype=positions.dtype
        )
        electronic_energy = jnp.asarray(jnp.nan, dtype=positions.dtype)
        completed = 0
        for iteration in range(self.maximum_iterations):
            coulomb = contract("cd,abcd->ab", density, integrals.electron_repulsion)
            exchange = contract("cd,acbd->ab", density, integrals.electron_repulsion)
            fock = core_hamiltonian + coulomb - 0.5 * exchange
            transformed = orthogonalizer.T @ fock @ orthogonalizer
            orbital_energies, transformed_coefficients = _symmetric_eigh(transformed)
            coefficients = orthogonalizer @ transformed_coefficients
            occupied_coefficients = coefficients[:, :occupied]
            proposed_density = 2.0 * occupied_coefficients @ occupied_coefficients.T
            mixed_density = (
                1.0 - self.damping
            ) * proposed_density + self.damping * density
            electronic_energy = 0.5 * contract(
                "ab,ab->", mixed_density, core_hamiltonian + fock
            )
            density_residual = jnp.max(jnp.abs(mixed_density - density))
            energy_residual = jnp.abs(electronic_energy - previous_energy)
            residual = jnp.maximum(density_residual, energy_residual)
            density = mixed_density
            previous_energy = electronic_energy
            completed = iteration + 1
            if bool(residual <= self.convergence_tolerance):
                converged = True
                break
        coulomb = contract("cd,abcd->ab", density, integrals.electron_repulsion)
        exchange = contract("cd,acbd->ab", density, integrals.electron_repulsion)
        fock = core_hamiltonian + coulomb - 0.5 * exchange
        orbital_energies, transformed_coefficients = _symmetric_eigh(
            orthogonalizer.T @ fock @ orthogonalizer
        )
        coefficients = orthogonalizer @ transformed_coefficients
        electronic_energy = 0.5 * contract("ab,ab->", density, core_hamiltonian + fock)
        total_energy = electronic_energy + nuclear_repulsion
        return SCFState(
            density,
            coefficients,
            orbital_energies,
            electronic_energy,
            total_energy,
            residual,
            completed,
            converged,
        )

    def evaluate(
        self, positions: ArrayLike, /, *, dipole: bool = False
    ) -> tuple[ElectronicKernelEvaluation, SCFState]:
        coordinate = jnp.asarray(positions)
        length_to_bohr = float(
            conversion_factor(self.system.units.scale.length_unit, BOHR)
        )
        energy_from_hartree = float(
            conversion_factor(HARTREE, self.system.units.scale.energy_unit)
        )
        coordinate_bohr = coordinate * length_to_bohr
        state = self.solve_atomic_units(coordinate_bohr)
        all_converged = bool(state.converged)
        forces = jnp.zeros_like(coordinate)
        for atom in range(int(coordinate.shape[0])):
            for component in range(3):
                displacement = (
                    jnp.zeros_like(coordinate)
                    .at[atom, component]
                    .set(self.force_displacement)
                )
                plus = self.solve_atomic_units(
                    (coordinate + displacement) * length_to_bohr
                )
                minus = self.solve_atomic_units(
                    (coordinate - displacement) * length_to_bohr
                )
                all_converged = (
                    all_converged and bool(plus.converged) and bool(minus.converged)
                )
                derivative = (
                    (plus.total_energy - minus.total_energy)
                    * energy_from_hartree
                    / (2.0 * self.force_displacement)
                )
                forces = forces.at[atom, component].set(-derivative)
        dipole_value = None
        if dipole:
            charges = jnp.asarray(self.system.atomic_numbers, dtype=coordinate.dtype)
            integrals = molecular_integrals(self.basis, coordinate_bohr, charges)
            electronic = -contract("ab,xab->x", state.density, integrals.dipole)
            nuclear = contract("a,ax->x", charges, coordinate_bohr)
            charge_factor = float(
                conversion_factor(ELEMENTARY_CHARGE, self.system.units.charge_unit)
            )
            length_from_bohr = float(
                conversion_factor(BOHR, self.system.units.scale.length_unit)
            )
            dipole_value = (electronic + nuclear) * charge_factor * length_from_bohr
        kernel = ElectronicKernelEvaluation(
            state.total_energy * energy_from_hartree,
            forces,
            all_converged,
            dipole=dipole_value,
            iterations=state.iterations,
            residual=state.residual,
        )
        return kernel, state


def rhf_tamm_dancoff(
    plan: NativeRHFPlan,
    positions: ArrayLike,
    state: SCFState,
    manifold: ExcitedStateManifoldPlan,
    /,
) -> TammDancoffPlan:
    if not isinstance(plan, NativeRHFPlan) or not isinstance(state, SCFState):
        raise TypeError("plan and state must be native RHF values.")
    if not isinstance(manifold, ExcitedStateManifoldPlan):
        raise TypeError("manifold must be ExcitedStateManifoldPlan.")
    if not bool(state.converged):
        raise ValueError("TDA requires a converged RHF parent state.")
    if manifold.spin_sector != "singlet":
        raise ValueError(
            "Native RHF TDA currently supports the singlet spin sector only."
        )
    if manifold.symmetry_sector is not None:
        raise ValueError("Native RHF TDA does not implement symmetry-sector projection.")
    coordinate = jnp.asarray(positions)
    coordinate_bohr = coordinate * float(
        conversion_factor(plan.system.units.scale.length_unit, BOHR)
    )
    charges = jnp.asarray(plan.system.atomic_numbers, dtype=coordinate.dtype)
    integrals = molecular_integrals(plan.basis, coordinate_bohr, charges)
    coefficients = state.coefficients
    eri_mo = contract(
        "pi,qj,rk,sl,pqrs->ijkl",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        integrals.electron_repulsion,
    )
    dipole_mo = contract(
        "pi,xpq,qj->xij",
        coefficients,
        integrals.dipole,
        coefficients,
    )
    occupied = plan.electron_count // 2
    orbital_count = plan.basis.basis_function_count
    virtual_count = orbital_count - occupied
    dimension = occupied * virtual_count
    if dimension <= 0 or manifold.root_count > dimension:
        raise ValueError("TDA roots exceed the RHF occupied--virtual response space.")
    response = jnp.zeros((dimension, dimension), dtype=coordinate.dtype)
    transition = jnp.zeros((dimension, 3), dtype=coordinate.dtype)
    for occupied_left in range(occupied):
        for virtual_left in range(virtual_count):
            a = occupied + virtual_left
            left_index = occupied_left * virtual_count + virtual_left
            transition = transition.at[left_index].set(
                jnp.sqrt(2.0) * dipole_mo[:, occupied_left, a]
            )
            for occupied_right in range(occupied):
                for virtual_right in range(virtual_count):
                    b = occupied + virtual_right
                    right_index = occupied_right * virtual_count + virtual_right
                    diagonal = jnp.where(
                        (occupied_left == occupied_right) & (a == b),
                        state.orbital_energies[a] - state.orbital_energies[occupied_left],
                        0.0,
                    )
                    coupling = (
                        2.0 * eri_mo[occupied_left, a, occupied_right, b]
                        - eri_mo[
                            occupied_left,
                            occupied_right,
                            a,
                            b,
                        ]
                    )
                    response = response.at[left_index, right_index].set(
                        diagonal + coupling
                    )
    dipole_unit = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
    return TammDancoffPlan(
        manifold,
        response,
        transition,
        state.total_energy,
        HARTREE,
        dipole_unit,
    )


class PreparedNativeRHFCalculation(AbstractPreparedElectronicCalculation):
    calculation: ElectronicCalculationPlan
    plan: NativeRHFPlan
    capabilities: ElectronicProviderCapabilities
    provider_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        calculation: ElectronicCalculationPlan,
        plan: NativeRHFPlan,
        capabilities: ElectronicProviderCapabilities,
        provider_id: str,
        /,
    ):
        self.calculation = calculation
        self.plan = plan
        self.capabilities = capabilities
        self.provider_id = provider_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-rhf-calculation",
                "calculation": calculation.calculation_id,
                "plan": plan.plan_id,
                "provider": provider_id,
            }
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> ElectronicEvaluation:
        if cell_vectors is not None:
            raise ValueError(
                "Native molecular RHF does not accept periodic cell vectors."
            )
        request = self.calculation.task
        kernel, state = self.plan.evaluate(
            positions, dipole=request.requires(ElectronicProperty.DIPOLE)
        )
        status = (
            ElectronicCalculationStatus.SUCCESS
            if bool(kernel.successful)
            else ElectronicCalculationStatus.ELECTRONIC_NOT_CONVERGED
        )
        return make_electronic_evaluation(
            self.calculation,
            self.provider_id,
            positions,
            kernel.energy,
            forces=kernel.forces,
            dipole=kernel.dipole,
            convergence=ElectronicConvergenceEvidence(
                kernel.successful,
                iterations=kernel.iterations,
                density_residual=kernel.residual,
                message=(
                    "native-rhf-converged"
                    if bool(kernel.successful)
                    else "native-rhf-not-converged"
                ),
            ),
            work=ElectronicWorkEvidence(
                energy_evaluations=1 + 6 * int(self.calculation.system.particle_ids.size),
                force_evaluations=1,
                property_evaluations=int(kernel.dipole is not None),
            ),
            status=status,
            source_unit_ids=(("energy", HARTREE.unit_id), ("length", BOHR.unit_id)),
            artifact_ids=(state.state_id,),
        )


class NativeRHFProvider(AbstractElectronicProvider):
    plan: NativeRHFPlan
    model_chemistry_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    capabilities: ElectronicProviderCapabilities

    def __init__(self, plan: NativeRHFPlan, model_chemistry_id: str, /):
        if not isinstance(plan, NativeRHFPlan):
            raise TypeError("plan must be NativeRHFPlan.")
        model_id = str(model_chemistry_id).strip()
        if not model_id:
            raise ValueError("model_chemistry_id must be non-empty.")
        capabilities = ElectronicProviderCapabilities.molecular_ground_state(
            (
                ElectronicProperty.ENERGY,
                ElectronicProperty.FORCES,
                ElectronicProperty.DIPOLE,
            ),
            (ElectronicReferenceKind.RESTRICTED,),
            families=(ElectronicMethodFamily.HARTREE_FOCK,),
            conservative_forces=True,
            differentiable=False,
            execution="host",
            concurrency="serial",
        )
        self.plan = plan
        self.model_chemistry_id = model_id
        self.capabilities = capabilities
        self.provider_id = canonical_fingerprint(
            {
                "kind": "native-rhf-provider",
                "plan": plan.plan_id,
                "model_chemistry": model_id,
                "capabilities": capabilities.capabilities_id,
            }
        )

    def prepare(
        self, calculation: ElectronicCalculationPlan, /
    ) -> PreparedNativeRHFCalculation:
        self.capabilities.require(calculation)
        if calculation.system.system_id != self.plan.system.system_id:
            raise ValueError("Native RHF plan belongs to another system.")
        if calculation.state.electron_count != self.plan.electron_count:
            raise ValueError(
                "Native RHF electron count differs from the calculation state."
            )
        if calculation.model_chemistry.model_chemistry_id != self.model_chemistry_id:
            raise ValueError("Native RHF provider is bound to another model chemistry.")
        if (
            calculation.model_chemistry.method.family
            is not ElectronicMethodFamily.HARTREE_FOCK
        ):
            raise ElectronicCapabilityError(
                "Native RHF requires Hartree--Fock model chemistry."
            )
        if (
            calculation.model_chemistry.method.reference
            is not ElectronicReferenceKind.RESTRICTED
        ):
            raise ElectronicCapabilityError(
                "Native RHF supports restricted references only."
            )
        if calculation.task.requires(ElectronicProperty.HESSIAN):
            raise ElectronicCapabilityError(
                "Use MolecularHessianPlan with native RHF forces."
            )
        return PreparedNativeRHFCalculation(
            calculation, self.plan, self.capabilities, self.provider_id
        )


__all__ = [
    "NativeRHFPlan",
    "NativeRHFProvider",
    "PreparedNativeRHFCalculation",
    "SCFState",
    "rhf_tamm_dancoff",
]
