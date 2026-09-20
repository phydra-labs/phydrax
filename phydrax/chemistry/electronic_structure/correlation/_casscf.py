#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded state-averaged CASSCF orbital optimization over explicit CASCI spaces."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.linalg import expm

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ._ci import CASCIPlan, CASCIResult
from ._orbital import MolecularOrbitalIntegralStore


class CASSCFResult(StrictModule, NonTrainableState):
    casci: CASCIResult
    orbital_rotation: Array
    macro_energies: Array
    orbital_gradient_norms: Array
    iterations: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        casci: CASCIResult,
        orbital_rotation,
        macro_energies,
        orbital_gradient_norms,
        iterations: int,
        successful,
        plan_id: str,
        /,
    ):
        if not isinstance(casci, CASCIResult):
            raise TypeError("casci must be CASCIResult.")
        rotation = jnp.asarray(orbital_rotation)
        energies = jnp.asarray(macro_energies, dtype=rotation.real.dtype)
        gradients = jnp.asarray(orbital_gradient_norms, dtype=rotation.real.dtype)
        self.casci = casci
        self.orbital_rotation = rotation
        self.macro_energies = energies
        self.orbital_gradient_norms = gradients
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "casscf-result",
                "plan": self.plan_id,
                "casci": casci.result_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "orbital_rotation": np.asarray(rotation),
                        "macro_energies": np.asarray(energies),
                        "orbital_gradient_norms": np.asarray(gradients),
                        "iterations": np.asarray(self.iterations),
                    }
                ),
            }
        )


class CASSCFPlan(StrictModule, NonTrainableState):
    casci: CASCIPlan
    state_weights: Array
    maximum_macro_iterations: int = eqx.field(static=True)
    orbital_gradient_tolerance: float = eqx.field(static=True)
    finite_difference_step: float = eqx.field(static=True)
    descent_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        casci: CASCIPlan,
        /,
        *,
        state_weights=None,
        maximum_macro_iterations: int = 32,
        orbital_gradient_tolerance: float = 1.0e-6,
        finite_difference_step: float = 1.0e-4,
        descent_step: float = 0.1,
    ):
        if not isinstance(casci, CASCIPlan):
            raise TypeError("casci must be CASCIPlan.")
        weights = (
            np.full((casci.root_count,), 1.0 / casci.root_count)
            if state_weights is None
            else np.asarray(state_weights, dtype=np.float64)
        )
        iterations = int(maximum_macro_iterations)
        tolerance = float(orbital_gradient_tolerance)
        difference = float(finite_difference_step)
        step = float(descent_step)
        if (
            weights.shape != (casci.root_count,)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
            or iterations <= 0
            or any(
                not isfinite(value) or value <= 0.0
                for value in (tolerance, difference, step)
            )
        ):
            raise ValueError(
                "CASSCF weights, iterations, tolerance, or steps are invalid."
            )
        self.casci = casci
        self.state_weights = jnp.asarray(weights)
        self.maximum_macro_iterations = iterations
        self.orbital_gradient_tolerance = tolerance
        self.finite_difference_step = difference
        self.descent_step = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "casscf-plan",
                "casci": casci.plan_id,
                "state_weights": weights.tolist(),
                "maximum_macro_iterations": iterations,
                "orbital_gradient_tolerance": tolerance,
                "finite_difference_step": difference,
                "descent_step": step,
            }
        )

    @staticmethod
    def _rotate_store(
        store: MolecularOrbitalIntegralStore,
        rotation: np.ndarray,
        /,
    ) -> MolecularOrbitalIntegralStore:
        unitary = np.asarray(expm(rotation))
        one = np.asarray(store.one_body)
        two = np.asarray(store.dense_two_body())
        rotated_one = unitary.T.conj() @ one @ unitary
        rotated_two = np.asarray(
            contract(
                "pi,qj,rk,sl,pqrs->ijkl",
                jnp.asarray(unitary.conj()),
                jnp.asarray(unitary),
                jnp.asarray(unitary.conj()),
                jnp.asarray(unitary),
                jnp.asarray(two),
            )
        )
        coefficients = np.asarray(store.coefficients) @ unitary
        energies = np.real(
            np.diag(
                unitary.T.conj() @ np.diag(np.asarray(store.orbital_energies)) @ unitary
            )
        )
        return MolecularOrbitalIntegralStore(
            rotated_one,
            energies,
            coefficients,
            store.nuclear_repulsion,
            store.reference_energy,
            store.partition,
            store.source_basis_id,
            two_body=rotated_two,
        )

    def evaluate(self, integrals: MolecularOrbitalIntegralStore, /) -> CASSCFResult:
        if not isinstance(integrals, MolecularOrbitalIntegralStore):
            raise TypeError("integrals must be MolecularOrbitalIntegralStore.")
        orbital_count = integrals.partition.orbital_count
        active_set = set(self.casci.active_orbitals)
        pairs = tuple(
            (active, external)
            for active in self.casci.active_orbitals
            for external in range(orbital_count)
            if external not in active_set
        )
        if not pairs:
            casci = self.casci.evaluate(integrals)
            return CASSCFResult(
                casci,
                jnp.eye(orbital_count),
                jnp.asarray([jnp.sum(self.state_weights * casci.energies)]),
                jnp.asarray([0.0]),
                0,
                casci.successful,
                self.plan_id,
            )
        current = integrals
        cumulative = np.eye(orbital_count)
        energies = []
        gradient_norms = []
        final_casci = self.casci.evaluate(current)
        converged = False
        for _ in range(self.maximum_macro_iterations):
            final_casci = self.casci.evaluate(current)
            state_average = float(
                np.sum(np.asarray(self.state_weights) * np.asarray(final_casci.energies))
            )
            energies.append(state_average)
            gradient = np.zeros((len(pairs),), dtype=np.float64)
            for index, (active, external) in enumerate(pairs):
                generator = np.zeros((orbital_count, orbital_count), dtype=np.float64)
                generator[external, active] = self.finite_difference_step
                generator[active, external] = -self.finite_difference_step
                plus = self.casci.evaluate(self._rotate_store(current, generator))
                minus = self.casci.evaluate(self._rotate_store(current, -generator))
                plus_energy = float(
                    np.sum(np.asarray(self.state_weights) * np.asarray(plus.energies))
                )
                minus_energy = float(
                    np.sum(np.asarray(self.state_weights) * np.asarray(minus.energies))
                )
                gradient[index] = (plus_energy - minus_energy) / (
                    2.0 * self.finite_difference_step
                )
            norm = float(np.linalg.norm(gradient))
            gradient_norms.append(norm)
            if norm <= self.orbital_gradient_tolerance:
                converged = True
                break
            update = np.zeros((orbital_count, orbital_count), dtype=np.float64)
            for value, (active, external) in zip(gradient, pairs, strict=True):
                update[external, active] = -self.descent_step * value
                update[active, external] = self.descent_step * value
            unitary = expm(update)
            cumulative = cumulative @ unitary
            current = self._rotate_store(current, update)
        successful = converged and bool(final_casci.successful)
        return CASSCFResult(
            final_casci,
            cumulative,
            np.asarray(energies),
            np.asarray(gradient_norms),
            len(energies),
            successful,
            self.plan_id,
        )


__all__ = ["CASSCFPlan", "CASSCFResult"]
