#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit stationary-density CPHF/CPKS response without differentiating SCF iterations."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare as prepare_linear,
    RHSLayout,
    solve as solve_linear,
)
from ...nonlinear import NonlinearSystemProblem, root_solution_jvp
from ...operators.quantum.gaussian import (
    ao_gradients,
    ao_values,
    dipole_integrals,
    electron_repulsion_tensor,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion_energy,
    overlap_matrix,
)
from ._mean_field import mean_field_owner_id, RestrictedMeanFieldState
from ._molecular_hf import MolecularHartreeFockPlan
from ._molecular_ks import MolecularKohnShamPlan


class MeanFieldResponseResult(StrictModule):
    density_response: Array
    dipole_response: Array
    polarizability: Array
    response_residuals: Array
    condition_estimates: Array
    successful: Array
    route: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class MeanFieldHessianResult(StrictModule):
    raw_hessian: Array
    hessian: Array
    antisymmetry_residual: Array
    density_response: Array
    response_residuals: Array
    condition_estimates: Array
    successful: Array
    route: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class MeanFieldResponsePlan(StrictModule, NonTrainableState):
    residual_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        residual_tolerance: float = 1.0e-8,
        condition_limit: float = 1.0e12,
    ):
        residual = float(residual_tolerance)
        condition = float(condition_limit)
        if residual <= 0.0 or condition <= 1.0:
            raise ValueError("Response residual and condition limits are invalid.")
        self.residual_tolerance = residual
        self.condition_limit = condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mean-field-response-plan",
                "residual_tolerance": residual,
                "condition_limit": condition,
            }
        )

    @staticmethod
    def _diagonalize(fock: Array, overlap: Array, /) -> tuple[Array, Array]:
        # This map is differentiated by CPHF/CPKS. The policy eigensolver does not
        # currently expose eigenvector JVPs, so the differentiable Hermitian JAX
        # primitive is required here rather than used as an ungoverned solve.
        overlap_values, overlap_vectors = jnp.linalg.eigh(
            0.5 * (overlap + jnp.conj(overlap.T))
        )
        inverse_sqrt = (
            overlap_vectors @ jnp.diag(overlap_values**-0.5) @ jnp.conj(overlap_vectors.T)
        )
        values, vectors = jnp.linalg.eigh(jnp.conj(inverse_sqrt.T) @ fock @ inverse_sqrt)
        return values, inverse_sqrt @ vectors

    def _fock_orbitals(
        self,
        calculation: MolecularHartreeFockPlan | MolecularKohnShamPlan,
        density: Array,
        positions: Array,
        field: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        basis = calculation.basis
        charges = jnp.asarray(calculation.system.atomic_numbers, dtype=positions.dtype)
        overlap = overlap_matrix(basis, positions)
        core = kinetic_matrix(basis, positions) + nuclear_attraction_matrix(
            basis, positions, charges
        )
        core = core + contract("x,xab->ab", field, dipole_integrals(basis, positions))
        eri = electron_repulsion_tensor(basis, positions)
        coulomb = contract("cd,abcd->ab", density, eri)
        if isinstance(calculation, MolecularHartreeFockPlan):
            exchange = contract("cd,acbd->ab", density, eri)
            fock = core + coulomb - 0.5 * exchange
        else:
            exchange_eri = calculation.exchange_tensor(positions, eri)
            exchange = contract("cd,acbd->ab", density, exchange_eri)
            grid = calculation.grid.evaluate(positions)
            ao = ao_values(basis, positions, grid.points)
            gradient = ao_gradients(basis, positions, grid.points)
            alpha = beta = 0.5 * density
            _, alpha_potential, beta_potential = (
                calculation.functional.potential_matrices(
                    alpha, beta, ao, gradient, grid.weights
                )
            )
            fock = (
                core + coulomb - 0.5 * exchange + 0.5 * (alpha_potential + beta_potential)
            )
        energies, coefficients = self._diagonalize(fock, overlap)
        return overlap, core, eri, fock, energies, coefficients

    def _density_map(
        self,
        calculation: MolecularHartreeFockPlan | MolecularKohnShamPlan,
        state: RestrictedMeanFieldState,
        density: Array,
        positions: Array,
        field: Array,
        /,
    ) -> Array:
        *_, coefficients = self._fock_orbitals(calculation, density, positions, field)
        return contract(
            "pi,i,qi->pq",
            coefficients,
            state.occupations,
            jnp.conj(coefficients),
        )

    def electric_response(
        self,
        calculation: MolecularHartreeFockPlan | MolecularKohnShamPlan,
        positions: ArrayLike,
        state: RestrictedMeanFieldState,
        /,
    ) -> MeanFieldResponseResult:
        if not isinstance(calculation, (MolecularHartreeFockPlan, MolecularKohnShamPlan)):
            raise TypeError("Response requires a native molecular HF or KS plan.")
        if not isinstance(state, RestrictedMeanFieldState) or not bool(
            state.evidence.converged
        ):
            raise ValueError("Response requires a converged restricted mean-field state.")
        if state.owner_id != mean_field_owner_id(calculation.plan_id, positions):
            raise ValueError("Response state belongs to another plan or geometry.")
        coordinate = jnp.asarray(positions)
        zero_field = jnp.zeros((3,), dtype=coordinate.dtype)
        problem = NonlinearSystemProblem(
            lambda density, field: (
                self._density_map(calculation, state, density, coordinate, field)
                - density
            ),
            problem_id=f"mean-field-electric-response:{self.plan_id}:{calculation.plan_id}",
        )
        responses = []
        residuals = []
        conditions = []
        successful = jnp.asarray(True)
        for axis in range(3):
            tangent = jnp.zeros((3,), dtype=coordinate.dtype).at[axis].set(1.0)
            derivative = root_solution_jvp(
                problem,
                state.density,
                zero_field,
                tangent,
            )
            responses.append(derivative.value)
            residuals.append(derivative.evidence.residual_norm)
            conditions.append(derivative.evidence.condition_estimate)
            successful = successful & derivative.evidence.successful
        density_response = jnp.stack(tuple(responses))
        dipoles = dipole_integrals(calculation.basis, coordinate)
        dipole_response = -contract("xab,yab->yx", dipoles, density_response)
        polarizability = 0.5 * (dipole_response + dipole_response.T)
        residual_values = jnp.stack(tuple(residuals))
        condition_values = jnp.stack(tuple(conditions))
        successful = (
            successful
            & jnp.all(residual_values <= self.residual_tolerance)
            & jnp.all(condition_values <= self.condition_limit)
            & jnp.all(jnp.isfinite(polarizability))
        )
        route = "cphf" if isinstance(calculation, MolecularHartreeFockPlan) else "cpks"
        return MeanFieldResponseResult(
            density_response,
            dipole_response,
            polarizability,
            residual_values,
            condition_values,
            successful,
            route,
            self.plan_id,
            state.state_id,
        )

    def _stationary_energy(
        self,
        calculation: MolecularHartreeFockPlan | MolecularKohnShamPlan,
        density: Array,
        positions: Array,
        /,
    ) -> Array:
        zero_field = jnp.zeros((3,), dtype=positions.dtype)
        _, core, eri, _, _, _ = self._fock_orbitals(
            calculation, density, positions, zero_field
        )
        coulomb = contract("cd,abcd->ab", density, eri)
        if isinstance(calculation, MolecularHartreeFockPlan):
            exchange_eri = eri
            xc = jnp.asarray(0.0, dtype=positions.dtype)
        else:
            exchange_eri = calculation.exchange_tensor(positions, eri)
            grid = calculation.grid.evaluate(positions)
            ao = ao_values(calculation.basis, positions, grid.points)
            gradient = ao_gradients(calculation.basis, positions, grid.points)
            xc = calculation.functional.energy(
                0.5 * density,
                0.5 * density,
                ao,
                gradient,
                grid.weights,
            )
        exchange = contract("cd,acbd->ab", density, exchange_eri)
        charges = jnp.asarray(calculation.system.atomic_numbers, dtype=positions.dtype)
        return jnp.real(
            contract("ab,ab->", density, core)
            + 0.5 * contract("ab,ab->", density, coulomb)
            - 0.25 * contract("ab,ab->", density, exchange)
            + xc
            + nuclear_repulsion_energy(positions, charges)
        )

    def _stationary_gradient(
        self,
        calculation: MolecularHartreeFockPlan | MolecularKohnShamPlan,
        state: RestrictedMeanFieldState,
        density: Array,
        positions: Array,
        /,
    ) -> Array:
        partial = jax.grad(
            lambda value: self._stationary_energy(calculation, density, value)
        )(positions)
        zero_field = jnp.zeros((3,), dtype=positions.dtype)
        overlap, _, _, _, energies, coefficients = self._fock_orbitals(
            calculation, density, positions, zero_field
        )
        del overlap
        energy_weighted = contract(
            "pi,i,qi->pq",
            coefficients,
            state.occupations * energies,
            jnp.conj(coefficients),
        )
        derivative_overlap = jax.jacfwd(
            lambda value: overlap_matrix(calculation.basis, value)
        )(positions)
        pulay = -contract("ab,abNx->Nx", energy_weighted, derivative_overlap)
        return jnp.real(partial + pulay)

    def nuclear_hessian(
        self,
        calculation: MolecularHartreeFockPlan | MolecularKohnShamPlan,
        positions: ArrayLike,
        state: RestrictedMeanFieldState,
        /,
    ) -> MeanFieldHessianResult:
        if not isinstance(calculation, (MolecularHartreeFockPlan, MolecularKohnShamPlan)):
            raise TypeError("Hessian response requires a native HF or KS plan.")
        if not isinstance(state, RestrictedMeanFieldState) or not bool(
            state.evidence.converged
        ):
            raise ValueError("Hessian response requires a converged restricted state.")
        if state.owner_id != mean_field_owner_id(calculation.plan_id, positions):
            raise ValueError(
                "Hessian response state belongs to another plan or geometry."
            )
        coordinate = jnp.asarray(positions)
        zero_field = jnp.zeros((3,), dtype=coordinate.dtype)
        density_coordinates = state.density.reshape((-1,))
        position_coordinates = coordinate.reshape((-1,))
        density_shape = state.density.shape
        position_shape = coordinate.shape
        coordinate_count = coordinate.size

        def residual_flat(current_density, current_positions):
            density = current_density.reshape(density_shape)
            positions_ = current_positions.reshape(position_shape)
            return (
                self._density_map(
                    calculation,
                    state,
                    density,
                    positions_,
                    zero_field,
                )
                - density
            ).reshape((-1,))

        density_jacobian = jax.jacfwd(
            lambda value: residual_flat(value, position_coordinates)
        )(density_coordinates)
        position_jacobian = jax.jacfwd(
            lambda value: residual_flat(density_coordinates, value)
        )(position_coordinates)
        prepared = prepare_linear(
            LeastSquaresProblem(DenseLinearOperator(density_jacobian)),
            LinearSolvePolicy(DenseSVD()),
        )
        linear = solve_linear(
            prepared,
            -position_jacobian,
            rhs_layout=RHSLayout((coordinate_count,)),
        )
        density_response_columns = linear.value
        response_residuals = jnp.sqrt(
            jnp.sum(
                jnp.abs(density_jacobian @ density_response_columns + position_jacobian)
                ** 2,
                axis=0,
            )
        )

        def gradient_flat(current_density, current_positions):
            return self._stationary_gradient(
                calculation,
                state,
                current_density.reshape(density_shape),
                current_positions.reshape(position_shape),
            ).reshape((-1,))

        gradient_density_jacobian = jax.jacfwd(
            lambda value: gradient_flat(value, position_coordinates)
        )(density_coordinates)
        explicit_gradient_jacobian = jax.jacfwd(
            lambda value: gradient_flat(density_coordinates, value)
        )(position_coordinates)
        raw_matrix = (
            explicit_gradient_jacobian
            + gradient_density_jacobian @ density_response_columns
        )
        symmetric_matrix = 0.5 * (raw_matrix + raw_matrix.T)
        raw = raw_matrix.reshape(position_shape + position_shape)
        hessian = symmetric_matrix.reshape(position_shape + position_shape)
        antisymmetry = jnp.max(jnp.abs(raw_matrix - raw_matrix.T), initial=0.0)
        density_response = density_response_columns.T.reshape(
            position_shape + density_shape
        )
        condition = jnp.max(jnp.asarray(linear.diagnostics.condition_estimate))
        condition_values = jnp.full(position_shape, condition)
        residual_values = response_residuals.reshape(position_shape)
        successful = (
            jnp.all(linear.successful)
            & jnp.all(residual_values <= self.residual_tolerance)
            & (condition <= self.condition_limit)
            & jnp.all(jnp.isfinite(hessian))
        )
        route = (
            "cphf-hessian"
            if isinstance(calculation, MolecularHartreeFockPlan)
            else "cpks-hessian"
        )
        return MeanFieldHessianResult(
            raw,
            hessian,
            antisymmetry,
            density_response,
            residual_values,
            condition_values,
            successful,
            route,
            self.plan_id,
            state.state_id,
        )


__all__ = [
    "MeanFieldHessianResult",
    "MeanFieldResponsePlan",
    "MeanFieldResponseResult",
]
