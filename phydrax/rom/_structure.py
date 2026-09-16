#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class SymplecticReduction(StrictModule, NonTrainableState):
    basis: Array
    symplectic_form: Array
    reduced_symplectic_form: Array
    defect: Array
    reduction_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: ArrayLike,
        symplectic_form: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-8,
    ):
        values = jnp.asarray(basis)
        form = jnp.asarray(symplectic_form)
        if (
            values.ndim != 2
            or form.shape != (values.shape[0], values.shape[0])
            or values.shape[1] % 2
        ):
            raise ValueError("Symplectic basis/form shapes are invalid.")
        reduced_rank = values.shape[1]
        half = reduced_rank // 2
        canonical = jnp.block(
            [
                [jnp.zeros((half, half)), jnp.eye(half)],
                [-jnp.eye(half), jnp.zeros((half, half))],
            ]
        ).astype(values.dtype)
        reduced = jnp.conj(values.T) @ form @ values
        defect = jnp.max(jnp.abs(reduced - canonical))
        if float(np.asarray(defect)) > float(tolerance):
            raise ValueError("Basis is not symplectic within tolerance.")
        self.basis = values
        self.symplectic_form = form
        self.reduced_symplectic_form = canonical
        self.defect = defect
        self.reduction_id = canonical_fingerprint(
            {
                "kind": "symplectic-reduction",
                "content": array_tree_fingerprint({"basis": values, "form": form})[
                    "sha256"
                ],
            }
        )

    def project_hamiltonian_matrix(self, matrix: ArrayLike, /) -> Array:
        value = jnp.asarray(matrix)
        symplectic_inverse = -self.reduced_symplectic_form
        return (
            symplectic_inverse
            @ jnp.conj(self.basis.T)
            @ self.symplectic_form
            @ value
            @ self.basis
        )


class PortHamiltonianReduction(StrictModule, NonTrainableState):
    interconnection: Array
    dissipation: Array
    energy_metric: Array
    input_matrix: Array
    trial_basis: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        interconnection: ArrayLike,
        dissipation: ArrayLike,
        energy_metric: ArrayLike,
        input_matrix: ArrayLike,
        trial_basis: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-8,
    ):
        inter = jnp.asarray(interconnection)
        dissip = jnp.asarray(dissipation)
        energy = jnp.asarray(energy_metric)
        inputs = jnp.asarray(input_matrix)
        basis = jnp.asarray(trial_basis)
        n = inter.shape[0]
        if (
            inter.shape != (n, n)
            or dissip.shape != (n, n)
            or energy.shape != (n, n)
            or inputs.ndim != 2
            or inputs.shape[0] != n
            or basis.ndim != 2
            or basis.shape[0] != n
        ):
            raise ValueError("Port-Hamiltonian arrays have incompatible shape.")
        tolerance_ = float(tolerance)
        skew_defect = float(np.max(np.abs(np.asarray(inter + jnp.conj(inter.T)))))
        dissipation_eigen = np.linalg.eigvalsh(
            np.asarray(0.5 * (dissip + jnp.conj(dissip.T)))
        )
        energy_eigen = np.linalg.eigvalsh(np.asarray(0.5 * (energy + jnp.conj(energy.T))))
        if (
            skew_defect > tolerance_
            or np.min(dissipation_eigen) < -tolerance_
            or np.min(energy_eigen) <= tolerance_
        ):
            raise ValueError("Port-Hamiltonian structure is not valid.")
        gram = jnp.conj(basis.T) @ energy @ basis
        test = energy @ basis @ jnp.linalg.inv(gram)
        reduced_inter = jnp.conj(test.T) @ inter @ test
        reduced_dissip = jnp.conj(test.T) @ dissip @ test
        reduced_energy = gram
        reduced_input = jnp.conj(test.T) @ inputs
        self.interconnection = 0.5 * (reduced_inter - jnp.conj(reduced_inter.T))
        self.dissipation = 0.5 * (reduced_dissip + jnp.conj(reduced_dissip.T))
        self.energy_metric = 0.5 * (reduced_energy + jnp.conj(reduced_energy.T))
        self.input_matrix = reduced_input
        self.trial_basis = basis
        self.model_id = canonical_fingerprint(
            {
                "kind": "port-hamiltonian-reduction",
                "content": array_tree_fingerprint(
                    {"j": inter, "r": dissip, "q": energy, "b": inputs, "basis": basis}
                )["sha256"],
            }
        )

    def system_matrix(self) -> Array:
        return (self.interconnection - self.dissipation) @ self.energy_metric


class DissipativeStructureEvidence(StrictModule, NonTrainableState):
    energy_change: Array
    invariant_defect: Array
    valid: Array
    structure_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        energy_change: ArrayLike,
        invariant_defect: ArrayLike,
        /,
        *,
        structure_id: str,
        tolerance: float,
    ):
        change = jnp.asarray(energy_change)
        defect = jnp.asarray(invariant_defect)
        identifier = str(structure_id)
        tolerance_ = float(tolerance)
        if not identifier or not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Structure evidence identity or tolerance is invalid.")
        self.energy_change = change
        self.invariant_defect = defect
        self.valid = jnp.all(change <= tolerance_) & jnp.all(
            jnp.abs(defect) <= tolerance_
        )
        self.structure_id = identifier
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "dissipative-structure-evidence",
                "structure": identifier,
                "tolerance": tolerance_,
            }
        )


__all__ = [
    "DissipativeStructureEvidence",
    "PortHamiltonianReduction",
    "SymplecticReduction",
]
