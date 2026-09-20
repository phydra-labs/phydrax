#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Full non-Hermitian TDHF/TDDFT response eigenproblem with X/Y semantics."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator
from ...linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
)
from ...units import (
    BOHR,
    conversion_factor,
    derived_unit,
    ELEMENTARY_CHARGE,
    HARTREE,
    UnitDefinition,
)
from ._manifold import ElectronicManifoldResult
from ._representation import RPAStateRepresentation


class RandomPhaseApproximationPlan(StrictModule, NonTrainableState):
    a_matrix: Array
    b_matrix: Array
    basis_transition_dipoles: Array
    ground_state_energy: Array
    root_count: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    spin_sector: str = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    imaginary_tolerance: float = eqx.field(static=True)
    degeneracy_tolerance: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    transition_dipole_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        a_matrix: ArrayLike,
        b_matrix: ArrayLike,
        basis_transition_dipoles: ArrayLike,
        ground_state_energy: ArrayLike,
        root_count: int,
        method: str,
        energy_unit: UnitDefinition,
        transition_dipole_unit: UnitDefinition,
        /,
        *,
        spin_sector: str = "singlet",
        residual_tolerance: float = 1.0e-8,
        imaginary_tolerance: float = 1.0e-8,
        degeneracy_tolerance: float = 1.0e-6,
    ):
        a = jnp.asarray(a_matrix)
        b = jnp.asarray(b_matrix, dtype=a.dtype)
        dipoles = jnp.asarray(basis_transition_dipoles, dtype=a.dtype)
        roots = int(root_count)
        method_ = str(method).strip()
        spin = str(spin_sector).strip()
        tolerances = (
            float(residual_tolerance),
            float(imaginary_tolerance),
            float(degeneracy_tolerance),
        )
        if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
            raise ValueError("RPA A and B matrices must be aligned square arrays.")
        if dipoles.shape != (a.shape[0], 3) or roots <= 0 or roots > a.shape[0]:
            raise ValueError("RPA transition basis or root count is invalid.")
        if (
            not method_
            or not spin
            or any(not isfinite(value) or value < 0.0 for value in tolerances)
            or tolerances[0] <= 0.0
        ):
            raise ValueError("RPA method, spin, or tolerances are invalid.")
        if not isinstance(energy_unit, UnitDefinition) or not isinstance(
            transition_dipole_unit, UnitDefinition
        ):
            raise TypeError("RPA energy and transition-dipole units must be typed.")
        self.a_matrix = a
        self.b_matrix = b
        self.basis_transition_dipoles = dipoles
        self.ground_state_energy = jnp.asarray(
            ground_state_energy, dtype=a.real.dtype
        ).reshape(())
        self.root_count = roots
        self.method = method_
        self.spin_sector = spin
        self.residual_tolerance, self.imaginary_tolerance, self.degeneracy_tolerance = (
            tolerances
        )
        self.energy_unit = energy_unit
        self.transition_dipole_unit = transition_dipole_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "random-phase-approximation-plan",
                "root_count": roots,
                "method": method_,
                "spin_sector": spin,
                "residual_tolerance": tolerances[0],
                "imaginary_tolerance": tolerances[1],
                "degeneracy_tolerance": tolerances[2],
                "energy_unit": energy_unit.unit_id,
                "transition_dipole_unit": transition_dipole_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "a": np.asarray(a),
                        "b": np.asarray(b),
                        "basis_transition_dipoles": np.asarray(dipoles),
                        "ground_state_energy": np.asarray(self.ground_state_energy),
                    }
                ),
            }
        )

    def solve(self, /) -> ElectronicManifoldResult:
        a = self.a_matrix
        b = self.b_matrix
        response = jnp.block([[a, b], [-jnp.conj(b), -jnp.conj(a)]])
        solve = general_eigensolve(
            GeneralEigenproblem(DenseLinearOperator(response)),
            policy=GeneralEigenSolvePolicy(
                DenseSchurQZ(),
                selection=GeneralEigenSelection.all(),
            ),
        )
        values_host = np.asarray(solve.eigenvalues)
        positive = np.flatnonzero(
            (values_host.real > self.imaginary_tolerance)
            & (np.abs(values_host.imag) <= self.imaginary_tolerance)
        )
        if positive.size < self.root_count:
            raise ValueError("RPA response does not contain enough positive real roots.")
        selected = positive[np.argsort(values_host.real[positive])[: self.root_count]]
        values = jnp.real(solve.eigenvalues[selected])
        right = jnp.asarray(solve.right_eigenvector_coordinates)[:, selected]
        left = jnp.asarray(solve.left_eigenvector_coordinates)[:, selected]
        dimension = a.shape[0]
        x = right[:dimension]
        y = right[dimension:]
        symplectic = jnp.real(jnp.sum(jnp.conj(x) * x - jnp.conj(y) * y, axis=0))
        scale = jnp.sqrt(jnp.abs(symplectic))
        x = x / scale[None, :]
        y = y / scale[None, :]
        left = left * scale[None, :]
        right_normalized = jnp.concatenate((x, y), axis=0)
        representation = RPAStateRepresentation(x, y, left)
        residuals = jnp.sqrt(
            jnp.sum(
                jnp.abs(response @ right_normalized - right_normalized * values[None, :])
                ** 2,
                axis=0,
            )
        )
        transition = jnp.conj((x + y).T) @ self.basis_transition_dipoles
        atomic_dipole = derived_unit("e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1)))
        values_atomic = values * float(conversion_factor(self.energy_unit, HARTREE))
        transition_atomic = transition * float(
            conversion_factor(self.transition_dipole_unit, atomic_dipole)
        )
        oscillator = (
            (2.0 / 3.0) * values_atomic * jnp.sum(jnp.abs(transition_atomic) ** 2, axis=1)
        )
        values_np = np.asarray(values)
        clusters = []
        start = 0
        for index in range(1, self.root_count):
            if abs(values_np[index] - values_np[index - 1]) > self.degeneracy_tolerance:
                clusters.append(tuple(range(start, index)))
                start = index
        clusters.append(tuple(range(start, self.root_count)))
        successful = (
            solve.successful
            & jnp.all(residuals <= self.residual_tolerance)
            & jnp.all(representation.symplectic_norms > 0.0)
            & jnp.all(jnp.isfinite(oscillator))
        )
        return ElectronicManifoldResult(
            values,
            self.ground_state_energy + values,
            representation,
            transition,
            oscillator,
            residuals,
            successful,
            self.method,
            self.spin_sector,
            tuple(clusters),
            self.energy_unit,
            self.transition_dipole_unit,
        )


__all__ = ["RandomPhaseApproximationPlan"]
