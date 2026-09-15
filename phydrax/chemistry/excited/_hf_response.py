#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Restricted Hartree–Fock singlet/triplet TDA and full TDHF builders."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...operators.quantum.gaussian import dipole_integrals, electron_repulsion_tensor
from ...units import BOHR, derived_unit, ELEMENTARY_CHARGE, HARTREE
from ..electronic_structure._mean_field import RestrictedMeanFieldState
from ..electronic_structure._molecular_hf import MolecularHartreeFockPlan
from ._rpa import RandomPhaseApproximationPlan
from ._tda import ExcitedStateManifoldPlan, TammDancoffPlan


class HartreeFockExcitedResponsePlan(StrictModule, NonTrainableState):
    a_matrix: Array
    b_matrix: Array
    transition_dipoles: Array
    ground_state_energy: Array
    spin_sector: str = eqx.field(static=True)
    occupied_count: int = eqx.field(static=True)
    virtual_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hartree_fock: MolecularHartreeFockPlan,
        state: RestrictedMeanFieldState,
        positions_bohr: ArrayLike,
        /,
        *,
        spin_sector: str = "singlet",
    ):
        if not isinstance(hartree_fock, MolecularHartreeFockPlan):
            raise TypeError("hartree_fock must be MolecularHartreeFockPlan.")
        if not isinstance(state, RestrictedMeanFieldState) or not bool(
            state.evidence.converged
        ):
            raise ValueError("Excited response requires a converged restricted HF state.")
        spin = str(spin_sector).strip().lower()
        if spin not in ("singlet", "triplet"):
            raise ValueError(
                "Restricted HF response spin sector must be singlet or triplet."
            )
        occupied = int(jnp.count_nonzero(state.occupations > 1.0))
        orbital_count = int(state.coefficients.shape[1])
        virtual = orbital_count - occupied
        if occupied <= 0 or virtual <= 0:
            raise ValueError("HF response requires occupied and virtual orbitals.")
        positions = jnp.asarray(positions_bohr)
        eri_ao = electron_repulsion_tensor(hartree_fock.basis, positions)
        dipole_ao = dipole_integrals(hartree_fock.basis, positions)
        coefficients = state.coefficients
        eri = contract(
            "pi,qj,rk,sl,pqrs->ijkl",
            jnp.conj(coefficients),
            coefficients,
            jnp.conj(coefficients),
            coefficients,
            eri_ao,
        )
        dipole = contract(
            "pi,xpq,qj->xij",
            jnp.conj(coefficients),
            dipole_ao,
            coefficients,
        )
        dimension = occupied * virtual
        a_matrix = jnp.zeros((dimension, dimension), dtype=eri.dtype)
        b_matrix = jnp.zeros_like(a_matrix)
        transition = jnp.zeros((dimension, 3), dtype=eri.dtype)
        for i in range(occupied):
            for a_offset in range(virtual):
                a = occupied + a_offset
                ia = i * virtual + a_offset
                if spin == "singlet":
                    transition = transition.at[ia].set(jnp.sqrt(2.0) * dipole[:, i, a])
                for j in range(occupied):
                    for b_offset in range(virtual):
                        b = occupied + b_offset
                        jb = j * virtual + b_offset
                        diagonal = jnp.where(
                            (i == j) & (a == b),
                            state.orbital_energies[a] - state.orbital_energies[i],
                            0.0,
                        )
                        if spin == "singlet":
                            a_coupling = 2.0 * eri[i, a, j, b] - eri[i, j, a, b]
                            b_coupling = 2.0 * eri[i, a, b, j] - eri[i, b, a, j]
                        else:
                            a_coupling = -eri[i, j, a, b]
                            b_coupling = -eri[i, b, a, j]
                        a_matrix = a_matrix.at[ia, jb].set(diagonal + a_coupling)
                        b_matrix = b_matrix.at[ia, jb].set(b_coupling)
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.transition_dipoles = transition
        self.ground_state_energy = state.total_energy
        self.spin_sector = spin
        self.occupied_count = occupied
        self.virtual_count = virtual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hartree-fock-excited-response",
                "hartree_fock": hartree_fock.plan_id,
                "state": state.state_id,
                "spin_sector": spin,
                "occupied_count": occupied,
                "virtual_count": virtual,
            }
        )

    def tda(self, root_count: int, /) -> TammDancoffPlan:
        dipole_unit = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
        return TammDancoffPlan(
            ExcitedStateManifoldPlan(root_count, spin_sector=self.spin_sector),
            self.a_matrix,
            self.transition_dipoles,
            self.ground_state_energy,
            HARTREE,
            dipole_unit,
        )

    def tdhf(self, root_count: int, /) -> RandomPhaseApproximationPlan:
        dipole_unit = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
        return RandomPhaseApproximationPlan(
            self.a_matrix,
            self.b_matrix,
            self.transition_dipoles,
            self.ground_state_energy,
            root_count,
            "tdhf",
            HARTREE,
            dipole_unit,
            spin_sector=self.spin_sector,
        )


__all__ = ["HartreeFockExcitedResponsePlan"]
