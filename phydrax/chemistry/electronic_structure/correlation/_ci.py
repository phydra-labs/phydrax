#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded determinant FCI and CASCI over explicit molecular-orbital spaces."""

from __future__ import annotations

from itertools import combinations
from math import comb

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._orbital import MolecularOrbitalIntegralStore


def _annihilate(bits: int, orbital: int, /) -> tuple[int, int] | None:
    mask = 1 << orbital
    if not bits & mask:
        return None
    phase = -1 if (bits & (mask - 1)).bit_count() % 2 else 1
    return bits ^ mask, phase


def _create(bits: int, orbital: int, /) -> tuple[int, int] | None:
    mask = 1 << orbital
    if bits & mask:
        return None
    phase = -1 if (bits & (mask - 1)).bit_count() % 2 else 1
    return bits | mask, phase


def _determinants(
    spatial_orbitals: int,
    alpha_electrons: int,
    beta_electrons: int,
    /,
) -> tuple[int, ...]:
    alpha = tuple(combinations(range(spatial_orbitals), alpha_electrons))
    beta = tuple(combinations(range(spatial_orbitals), beta_electrons))
    return tuple(
        sum(1 << value for value in alpha_)
        | sum(1 << (spatial_orbitals + value) for value in beta_)
        for alpha_ in alpha
        for beta_ in beta
    )


def _spin_orbital_integrals(one: np.ndarray, two: np.ndarray, /):
    spatial = one.shape[0]
    spin_count = 2 * spatial
    one_spin = np.zeros((spin_count, spin_count), dtype=one.dtype)
    antisymmetrized = np.zeros((spin_count,) * 4, dtype=two.dtype)
    for p in range(spin_count):
        spin_p, orbital_p = divmod(p, spatial)
        for q in range(spin_count):
            spin_q, orbital_q = divmod(q, spatial)
            if spin_p == spin_q:
                one_spin[p, q] = one[orbital_p, orbital_q]
            for r in range(spin_count):
                spin_r, orbital_r = divmod(r, spatial)
                for s in range(spin_count):
                    spin_s, orbital_s = divmod(s, spatial)
                    direct = (
                        two[orbital_p, orbital_r, orbital_q, orbital_s]
                        if spin_p == spin_r and spin_q == spin_s
                        else 0.0
                    )
                    exchange = (
                        two[orbital_p, orbital_s, orbital_q, orbital_r]
                        if spin_p == spin_s and spin_q == spin_r
                        else 0.0
                    )
                    antisymmetrized[p, q, r, s] = direct - exchange
    return one_spin, antisymmetrized


def _hamiltonian_matrix(
    determinants: tuple[int, ...],
    one_body: np.ndarray,
    antisymmetrized: np.ndarray,
    scalar_energy: float,
    /,
) -> np.ndarray:
    dimension = len(determinants)
    spin_orbitals = one_body.shape[0]
    index = {bits: ordinal for ordinal, bits in enumerate(determinants)}
    hamiltonian = np.zeros(
        (dimension, dimension), dtype=np.result_type(one_body, antisymmetrized)
    )
    for column, determinant in enumerate(determinants):
        hamiltonian[column, column] += scalar_energy
        for q in range(spin_orbitals):
            removed_q = _annihilate(determinant, q)
            if removed_q is None:
                continue
            state_q, phase_q = removed_q
            for p in range(spin_orbitals):
                created_p = _create(state_q, p)
                if created_p is None:
                    continue
                state_p, phase_p = created_p
                row = index.get(state_p)
                if row is not None:
                    hamiltonian[row, column] += phase_q * phase_p * one_body[p, q]
        for r in range(spin_orbitals):
            removed_r = _annihilate(determinant, r)
            if removed_r is None:
                continue
            state_r, phase_r = removed_r
            for s in range(spin_orbitals):
                removed_s = _annihilate(state_r, s)
                if removed_s is None:
                    continue
                state_s, phase_s = removed_s
                for q in range(spin_orbitals):
                    created_q = _create(state_s, q)
                    if created_q is None:
                        continue
                    state_q, phase_q = created_q
                    for p in range(spin_orbitals):
                        created_p = _create(state_q, p)
                        if created_p is None:
                            continue
                        state_p, phase_p = created_p
                        row = index.get(state_p)
                        if row is not None:
                            hamiltonian[row, column] += (
                                0.25
                                * phase_r
                                * phase_s
                                * phase_q
                                * phase_p
                                * antisymmetrized[p, q, r, s]
                            )
    return 0.5 * (hamiltonian + hamiltonian.T.conj())


class CASCIResult(StrictModule, NonTrainableState):
    energies: Array
    coefficients: Array
    hamiltonian_residuals: Array
    determinant_bits: tuple[int, ...] = eqx.field(static=True)
    active_orbitals: tuple[int, ...] = eqx.field(static=True)
    successful: Array
    plan_id: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies,
        coefficients,
        residuals,
        determinant_bits: tuple[int, ...],
        active_orbitals: tuple[int, ...],
        successful,
        plan_id: str,
        store_id: str,
        /,
    ):
        energies_ = jnp.asarray(energies)
        coefficients_ = jnp.asarray(coefficients)
        residuals_ = jnp.asarray(residuals, dtype=energies_.real.dtype)
        self.energies = energies_
        self.coefficients = coefficients_
        self.hamiltonian_residuals = residuals_
        self.determinant_bits = determinant_bits
        self.active_orbitals = active_orbitals
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.store_id = str(store_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "casci-result",
                "plan": self.plan_id,
                "store": self.store_id,
                "determinants": list(determinant_bits),
                "active_orbitals": list(active_orbitals),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(energies_),
                        "coefficients": np.asarray(coefficients_),
                        "residuals": np.asarray(residuals_),
                    }
                ),
            }
        )


class CASCIPlan(StrictModule, NonTrainableState):
    active_orbitals: tuple[int, ...] = eqx.field(static=True)
    active_alpha_electrons: int = eqx.field(static=True)
    active_beta_electrons: int = eqx.field(static=True)
    root_count: int = eqx.field(static=True)
    maximum_determinants: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        active_orbitals,
        active_alpha_electrons: int,
        active_beta_electrons: int,
        /,
        *,
        root_count: int = 1,
        maximum_determinants: int = 100_000,
        residual_tolerance: float = 1.0e-9,
    ):
        active = tuple(active_orbitals)
        alpha = int(active_alpha_electrons)
        beta = int(active_beta_electrons)
        roots = int(root_count)
        capacity = int(maximum_determinants)
        tolerance = float(residual_tolerance)
        if not active or len(set(active)) != len(active) or min(active) < 0:
            raise ValueError("CAS active orbitals must be unique non-negative indices.")
        if alpha < 0 or beta < 0 or alpha > len(active) or beta > len(active):
            raise ValueError("CAS active spin populations exceed active orbitals.")
        determinant_count = comb(len(active), alpha) * comb(len(active), beta)
        if (
            determinant_count <= 0
            or determinant_count > capacity
            or roots <= 0
            or roots > determinant_count
        ):
            raise ValueError("CAS determinant or root count exceeds its capacity.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("CAS residual tolerance must be positive finite.")
        self.active_orbitals = active
        self.active_alpha_electrons = alpha
        self.active_beta_electrons = beta
        self.root_count = roots
        self.maximum_determinants = capacity
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "casci-plan",
                "active_orbitals": list(active),
                "active_alpha_electrons": alpha,
                "active_beta_electrons": beta,
                "root_count": roots,
                "maximum_determinants": capacity,
                "residual_tolerance": tolerance,
            }
        )

    def evaluate(self, integrals: MolecularOrbitalIntegralStore, /) -> CASCIResult:
        if not isinstance(integrals, MolecularOrbitalIntegralStore):
            raise TypeError("integrals must be MolecularOrbitalIntegralStore.")
        if max(self.active_orbitals) >= integrals.partition.orbital_count:
            raise ValueError("CAS active orbital exceeds the integral store.")
        active = np.asarray(self.active_orbitals, dtype=np.int32)
        frozen = tuple(
            value
            for value in integrals.partition.frozen_occupied
            if value not in set(self.active_orbitals)
        )
        one = np.asarray(integrals.one_body)
        two = np.asarray(integrals.dense_two_body())
        core_energy = float(integrals.nuclear_repulsion)
        for i in frozen:
            core_energy += 2.0 * float(one[i, i])
            for j in frozen:
                core_energy += 2.0 * float(two[i, i, j, j]) - float(two[i, j, j, i])
        effective_one = one[np.ix_(active, active)].copy()
        for p, orbital_p in enumerate(active):
            for q, orbital_q in enumerate(active):
                for i in frozen:
                    effective_one[p, q] += (
                        2.0 * two[orbital_p, orbital_q, i, i]
                        - two[orbital_p, i, i, orbital_q]
                    )
        active_two = two[np.ix_(active, active, active, active)]
        one_spin, antisymmetrized = _spin_orbital_integrals(effective_one, active_two)
        determinants = _determinants(
            len(self.active_orbitals),
            self.active_alpha_electrons,
            self.active_beta_electrons,
        )
        hamiltonian = jnp.asarray(
            _hamiltonian_matrix(determinants, one_spin, antisymmetrized, core_energy)
        )
        solve = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    hamiltonian,
                    properties=OperatorProperties(
                        self_adjoint=True,
                        evidence={"self_adjoint": "construction"},
                    ),
                )
            ),
            policy=EigenSolvePolicy(
                DenseEigh(), count=self.root_count, which="smallest-algebraic"
            ),
        )
        residuals = jnp.sqrt(
            jnp.sum(
                jnp.abs(
                    hamiltonian @ solve.eigenvectors
                    - solve.eigenvectors * solve.eigenvalues[None, :]
                )
                ** 2,
                axis=0,
            )
        )
        successful = solve.successful & jnp.all(residuals <= self.residual_tolerance)
        return CASCIResult(
            solve.eigenvalues,
            solve.eigenvectors,
            residuals,
            determinants,
            self.active_orbitals,
            successful,
            self.plan_id,
            integrals.store_id,
        )


class FCIPlan(StrictModule, NonTrainableState):
    alpha_electrons: int = eqx.field(static=True)
    beta_electrons: int = eqx.field(static=True)
    root_count: int = eqx.field(static=True)
    maximum_determinants: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        alpha_electrons: int,
        beta_electrons: int,
        /,
        *,
        root_count: int = 1,
        maximum_determinants: int = 100_000,
        residual_tolerance: float = 1.0e-9,
    ):
        alpha = int(alpha_electrons)
        beta = int(beta_electrons)
        roots = int(root_count)
        capacity = int(maximum_determinants)
        tolerance = float(residual_tolerance)
        if (
            alpha < 0
            or beta < 0
            or roots <= 0
            or capacity <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("FCI electron, root, capacity, or tolerance is invalid.")
        self.alpha_electrons = alpha
        self.beta_electrons = beta
        self.root_count = roots
        self.maximum_determinants = capacity
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fci-plan",
                "alpha_electrons": alpha,
                "beta_electrons": beta,
                "root_count": roots,
                "maximum_determinants": capacity,
                "residual_tolerance": tolerance,
            }
        )

    def evaluate(self, integrals: MolecularOrbitalIntegralStore, /) -> CASCIResult:
        if integrals.partition.frozen_occupied or integrals.partition.frozen_virtual:
            raise ValueError("FCI requires an integral store without frozen orbitals.")
        active = tuple(range(integrals.partition.orbital_count))
        plan = CASCIPlan(
            active,
            self.alpha_electrons,
            self.beta_electrons,
            root_count=self.root_count,
            maximum_determinants=self.maximum_determinants,
            residual_tolerance=self.residual_tolerance,
        )
        result = plan.evaluate(integrals)
        return CASCIResult(
            result.energies,
            result.coefficients,
            result.hamiltonian_residuals,
            result.determinant_bits,
            result.active_orbitals,
            result.successful,
            self.plan_id,
            result.store_id,
        )


__all__ = ["CASCIPlan", "CASCIResult", "FCIPlan"]
