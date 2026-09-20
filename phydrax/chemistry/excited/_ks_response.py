#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Restricted adiabatic TDDFT from exact real/imaginary orbital Hessians."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...operators.quantum.gaussian import dipole_integrals
from ...units import BOHR, derived_unit, ELEMENTARY_CHARGE, HARTREE
from ..electronic_structure._mean_field import RestrictedMeanFieldState
from ..electronic_structure._molecular_ks import MolecularKohnShamPlan
from ._rpa import RandomPhaseApproximationPlan
from ._tda import ExcitedStateManifoldPlan, TammDancoffPlan


class KohnShamExcitedResponsePlan(StrictModule, NonTrainableState):
    a_matrix: Array
    b_matrix: Array
    transition_dipoles: Array
    ground_state_energy: Array
    occupied_count: int = eqx.field(static=True)
    virtual_count: int = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kohn_sham: MolecularKohnShamPlan,
        positions: ArrayLike,
        state: RestrictedMeanFieldState,
        /,
        *,
        dipole_ao: ArrayLike | None = None,
    ):
        if not isinstance(kohn_sham, MolecularKohnShamPlan):
            raise TypeError("kohn_sham must be MolecularKohnShamPlan.")
        if not isinstance(state, RestrictedMeanFieldState) or not bool(
            state.evidence.converged
        ):
            raise ValueError("TDDFT requires a converged restricted KS state.")
        occupation_host = np.asarray(state.occupations)
        occupied_indices = tuple(np.flatnonzero(occupation_host > 1.0))
        virtual_indices = tuple(np.flatnonzero(occupation_host < 1.0e-12))
        if len(occupied_indices) + len(virtual_indices) != occupation_host.size:
            raise ValueError("Native TDDFT requires integer closed-shell occupations.")
        if not occupied_indices or not virtual_indices:
            raise ValueError("TDDFT requires occupied and virtual orbital spaces.")
        coordinate = jnp.asarray(positions)
        (
            _,
            core,
            eri,
            exchange_eri,
            nuclear,
            grid,
            ao,
            ao_gradient,
        ) = kohn_sham._fixed_quantities(coordinate)
        coefficients = state.coefficients
        orbital_count = coefficients.shape[1]
        pairs = tuple(
            (virtual, occupied)
            for occupied in occupied_indices
            for virtual in virtual_indices
        )

        def rotated_density(parameters, imaginary):
            generator = jnp.zeros(
                (orbital_count, orbital_count),
                dtype=jnp.result_type(coefficients.dtype, jnp.complex64),
            )
            for index, (virtual, occupied) in enumerate(pairs):
                value = parameters[index]
                if imaginary:
                    value = 1.0j * value
                    conjugate_value = value
                else:
                    conjugate_value = -value
                generator = generator.at[virtual, occupied].set(value)
                generator = generator.at[occupied, virtual].set(conjugate_value)
            rotated = coefficients.astype(generator.dtype) @ jsp.linalg.expm(generator)
            occupied_coefficients = rotated[:, jnp.asarray(occupied_indices)]
            return 2.0 * occupied_coefficients @ jnp.conj(occupied_coefficients.T)

        def energy(parameters, imaginary):
            density = rotated_density(parameters, imaginary)
            alpha = beta = 0.5 * density
            coulomb = contract("cd,abcd->ab", density, eri)
            exchange = contract("cd,acbd->ab", density, exchange_eri)
            xc = kohn_sham.functional.energy(
                alpha,
                beta,
                ao,
                ao_gradient,
                grid.weights,
            )
            return jnp.real(
                contract("ab,ab->", density, core)
                + 0.5 * contract("ab,ab->", density, coulomb)
                - 0.25 * contract("ab,ab->", density, exchange)
                + xc
                + nuclear
            )

        zero = jnp.zeros((len(pairs),), dtype=coordinate.dtype)
        real_hessian = jax.hessian(lambda value: energy(value, False))(zero)
        imaginary_hessian = jax.hessian(lambda value: energy(value, True))(zero)
        a_matrix = 0.125 * (real_hessian + imaginary_hessian)
        b_matrix = 0.125 * (real_hessian - imaginary_hessian)
        a_matrix = 0.5 * (a_matrix + jnp.conj(a_matrix.T))
        b_matrix = 0.5 * (b_matrix + jnp.conj(b_matrix.T))
        dipoles = (
            dipole_integrals(kohn_sham.basis, coordinate)
            if dipole_ao is None
            else jnp.asarray(dipole_ao)
        )
        if dipoles.shape != (3, coefficients.shape[0], coefficients.shape[0]):
            raise ValueError("TDDFT AO dipoles must have shape (3, AO, AO).")
        dipole_mo = contract(
            "pi,xpq,qj->xij",
            jnp.conj(coefficients),
            dipoles,
            coefficients,
        )
        transition = jnp.stack(
            tuple(
                jnp.sqrt(2.0) * dipole_mo[:, occupied, virtual]
                for virtual, occupied in pairs
            )
        )
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.transition_dipoles = transition
        self.ground_state_energy = state.total_energy
        self.occupied_count = len(occupied_indices)
        self.virtual_count = len(virtual_indices)
        self.functional_id = kohn_sham.functional.functional_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kohn-sham-excited-response-plan",
                "kohn_sham": kohn_sham.plan_id,
                "state": state.state_id,
                "functional": self.functional_id,
                "occupied_count": self.occupied_count,
                "virtual_count": self.virtual_count,
                "arrays": array_tree_fingerprint(
                    {
                        "a": np.asarray(a_matrix),
                        "b": np.asarray(b_matrix),
                        "transition_dipoles": np.asarray(transition),
                    }
                ),
            }
        )

    def tda(self, root_count: int, /) -> TammDancoffPlan:
        dipole_unit = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
        return TammDancoffPlan(
            ExcitedStateManifoldPlan(root_count, spin_sector="singlet"),
            self.a_matrix,
            self.transition_dipoles,
            self.ground_state_energy,
            HARTREE,
            dipole_unit,
        )

    def tddft(self, root_count: int, /) -> RandomPhaseApproximationPlan:
        dipole_unit = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
        return RandomPhaseApproximationPlan(
            self.a_matrix,
            self.b_matrix,
            self.transition_dipoles,
            self.ground_state_energy,
            root_count,
            "tddft",
            HARTREE,
            dipole_unit,
            spin_sector="singlet",
        )


__all__ = ["KohnShamExcitedResponsePlan"]
