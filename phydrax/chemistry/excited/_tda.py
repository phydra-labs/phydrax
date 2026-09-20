#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tamm--Dancoff excited-state manifolds, tracking, and finite-difference couplings."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ...units import (
    BOHR,
    conversion_factor,
    derived_unit,
    ELEMENTARY_CHARGE,
    HARTREE,
    UnitDefinition,
)
from ._manifold import ElectronicManifoldResult
from ._representation import TDAStateRepresentation
from ._tracking import track_excited_states


class ExcitedStateManifoldPlan(StrictModule, NonTrainableState):
    root_count: int = eqx.field(static=True)
    spin_sector: str = eqx.field(static=True)
    symmetry_sector: str | None = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    degeneracy_absolute: float = eqx.field(static=True)
    degeneracy_relative: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_count: int,
        /,
        *,
        spin_sector: str = "singlet",
        symmetry_sector: str | None = None,
        residual_tolerance: float = 1.0e-8,
        degeneracy_absolute: float = 1.0e-6,
        degeneracy_relative: float = 1.0e-8,
    ):
        roots = int(root_count)
        spin = str(spin_sector).strip()
        symmetry = None if symmetry_sector is None else str(symmetry_sector).strip()
        values = tuple(
            float(value)
            for value in (
                residual_tolerance,
                degeneracy_absolute,
                degeneracy_relative,
            )
        )
        if roots <= 0 or not spin or (symmetry_sector is not None and not symmetry):
            raise ValueError("Excited-state root/spin/symmetry values are invalid.")
        if (
            any(not isfinite(value) or value < 0.0 for value in values)
            or values[0] <= 0.0
        ):
            raise ValueError("Excited-state tolerances are invalid.")
        self.root_count = roots
        self.spin_sector = spin
        self.symmetry_sector = symmetry
        (
            self.residual_tolerance,
            self.degeneracy_absolute,
            self.degeneracy_relative,
        ) = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "excited-state-manifold-plan",
                "root_count": roots,
                "spin_sector": spin,
                "symmetry_sector": symmetry,
                "residual_tolerance": values[0],
                "degeneracy_absolute": values[1],
                "degeneracy_relative": values[2],
            }
        )


class TammDancoffPlan(StrictModule, NonTrainableState):
    manifold: ExcitedStateManifoldPlan
    response_matrix: Array
    basis_transition_dipoles: Array
    ground_state_energy: Array
    energy_unit: UnitDefinition
    transition_dipole_unit: UnitDefinition
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        manifold: ExcitedStateManifoldPlan,
        response_matrix: ArrayLike,
        basis_transition_dipoles: ArrayLike,
        ground_state_energy: ArrayLike,
        energy_unit: UnitDefinition,
        transition_dipole_unit: UnitDefinition,
        /,
    ):
        if not isinstance(manifold, ExcitedStateManifoldPlan):
            raise TypeError("manifold must be ExcitedStateManifoldPlan.")
        matrix = jnp.asarray(response_matrix)
        dipoles = jnp.asarray(basis_transition_dipoles, dtype=matrix.dtype)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("TDA response_matrix must be square.")
        if dipoles.shape != (matrix.shape[0], 3):
            raise ValueError(
                "One Cartesian transition dipole is required per TDA basis vector."
            )
        if manifold.root_count > matrix.shape[0]:
            raise ValueError("Requested roots exceed TDA response dimension.")
        if not isinstance(energy_unit, UnitDefinition) or not isinstance(
            transition_dipole_unit, UnitDefinition
        ):
            raise TypeError("TDA units must be UnitDefinition values.")
        self.manifold = manifold
        self.response_matrix = matrix
        self.basis_transition_dipoles = dipoles
        self.ground_state_energy = jnp.asarray(
            ground_state_energy, dtype=matrix.dtype
        ).reshape(())
        self.energy_unit = energy_unit
        self.transition_dipole_unit = transition_dipole_unit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tamm-dancoff-plan",
                "manifold": manifold.plan_id,
                "energy_unit": energy_unit.unit_id,
                "transition_dipole_unit": transition_dipole_unit.unit_id,
                "arrays": array_tree_fingerprint(
                    {
                        "response_matrix": np.asarray(matrix),
                        "basis_transition_dipoles": np.asarray(dipoles),
                        "ground_state_energy": np.asarray(self.ground_state_energy),
                    }
                ),
            }
        )

    def solve(
        self,
        /,
        *,
        parent: ElectronicManifoldResult | None = None,
    ) -> ElectronicManifoldResult:
        root_count = self.manifold.root_count
        solve = eigensolve(
            Eigenproblem(
                DenseLinearOperator(
                    0.5 * (self.response_matrix + jnp.conj(self.response_matrix.T)),
                    properties=OperatorProperties(
                        self_adjoint=True,
                        evidence={"self_adjoint": "construction"},
                    ),
                )
            ),
            policy=EigenSolvePolicy(
                DenseEigh(), count=root_count, which="smallest-algebraic"
            ),
        )
        values = solve.eigenvalues
        vectors = solve.eigenvectors
        threshold = (
            self.manifold.degeneracy_absolute
            + self.manifold.degeneracy_relative
            * max(float(jnp.max(jnp.abs(values), initial=0.0)), 1.0)
        )
        values_host = np.asarray(values)
        clusters: list[tuple[int, ...]] = []
        start = 0
        for index in range(1, root_count):
            if abs(float(values_host[index] - values_host[index - 1])) > threshold:
                clusters.append(tuple(range(start, index)))
                start = index
        clusters.append(tuple(range(start, root_count)))

        def build_result(values_, vectors_, clusters_, successful_):
            residuals_ = jnp.sqrt(
                jnp.sum(
                    jnp.abs(self.response_matrix @ vectors_ - vectors_ * values_[None, :])
                    ** 2,
                    axis=0,
                )
            )
            transition_dipoles_ = jnp.conj(vectors_.T) @ self.basis_transition_dipoles
            atomic_dipole_unit = derived_unit(
                "e*bohr", ((ELEMENTARY_CHARGE, 1), (BOHR, 1))
            )
            energy_atomic = values_ * float(conversion_factor(self.energy_unit, HARTREE))
            dipole_atomic = transition_dipoles_ * float(
                conversion_factor(self.transition_dipole_unit, atomic_dipole_unit)
            )
            oscillator_ = (
                (2.0 / 3.0) * energy_atomic * jnp.sum(jnp.abs(dipole_atomic) ** 2, axis=1)
            )
            successful_ = (
                successful_
                & jnp.all(residuals_ <= self.manifold.residual_tolerance)
                & jnp.all(values_ > 0.0)
            )
            return ElectronicManifoldResult(
                values_,
                self.ground_state_energy + values_,
                TDAStateRepresentation(vectors_),
                transition_dipoles_,
                oscillator_,
                residuals_,
                successful_,
                "tda",
                self.manifold.spin_sector,
                clusters_,
                self.energy_unit,
                self.transition_dipole_unit,
                symmetry_sector=self.manifold.symmetry_sector,
            )

        result = build_result(
            values,
            vectors,
            tuple(clusters),
            solve.successful,
        )
        if parent is None:
            return result
        tracking = track_excited_states(parent, result, minimum_overlap=0.5)
        alignment = tracking.alignment
        vectors = vectors @ alignment
        values = jnp.real(jnp.diag(jnp.conj(alignment.T) @ jnp.diag(values) @ alignment))
        return build_result(
            values,
            vectors,
            parent.clusters,
            solve.successful & tracking.successful,
        )


class NonadiabaticCouplingResult(StrictModule, NonTrainableState):
    derivative_couplings: Array
    energy_weighted_couplings: Array
    antisymmetry_residual: Array
    successful: Array
    inverse_length_unit: UnitDefinition
    energy_weighted_unit: UnitDefinition
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        derivative_couplings: ArrayLike,
        energy_weighted_couplings: ArrayLike,
        antisymmetry_residual: ArrayLike,
        successful: ArrayLike,
        inverse_length_unit: UnitDefinition,
        energy_weighted_unit: UnitDefinition,
        /,
    ):
        derivative = jnp.asarray(derivative_couplings)
        weighted = jnp.asarray(energy_weighted_couplings, dtype=derivative.dtype)
        if (
            derivative.ndim != 2
            or derivative.shape[0] != derivative.shape[1]
            or weighted.shape != derivative.shape
        ):
            raise ValueError("Nonadiabatic coupling matrices must be square and aligned.")
        self.derivative_couplings = derivative
        self.energy_weighted_couplings = weighted
        self.antisymmetry_residual = jnp.asarray(
            antisymmetry_residual, dtype=derivative.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.inverse_length_unit = inverse_length_unit
        if not isinstance(energy_weighted_unit, UnitDefinition):
            raise TypeError("energy_weighted_unit must be UnitDefinition.")
        self.energy_weighted_unit = energy_weighted_unit
        self.result_id = canonical_fingerprint(
            {
                "kind": "nonadiabatic-coupling-result",
                "unit": inverse_length_unit.unit_id,
                "energy_weighted_unit": energy_weighted_unit.unit_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "derivative": np.asarray(derivative),
                        "weighted": np.asarray(weighted),
                        "antisymmetry_residual": np.asarray(self.antisymmetry_residual),
                    }
                ),
            }
        )


def finite_difference_nonadiabatic_coupling(
    reference: ElectronicManifoldResult,
    plus: ElectronicManifoldResult,
    minus: ElectronicManifoldResult,
    displacement: float,
    length_unit: UnitDefinition,
    /,
    *,
    antisymmetry_tolerance: float = 1.0e-6,
) -> NonadiabaticCouplingResult:
    step = float(displacement)
    if not isfinite(step) or step <= 0.0:
        raise ValueError("Nonadiabatic coupling displacement must be positive finite.")
    reference_representation = reference.representation
    plus_representation = plus.representation
    minus_representation = minus.representation
    if (
        not isinstance(reference_representation, TDAStateRepresentation)
        or not isinstance(plus_representation, TDAStateRepresentation)
        or not isinstance(minus_representation, TDAStateRepresentation)
    ):
        raise TypeError(
            "Finite-difference couplings require TDA manifold representations."
        )
    reference_amplitudes = reference_representation.amplitudes
    plus_amplitudes = plus_representation.amplitudes
    minus_amplitudes = minus_representation.amplitudes
    if any(
        value.shape != reference_amplitudes.shape
        for value in (plus_amplitudes, minus_amplitudes)
    ):
        raise ValueError("Excited-state manifolds must share one amplitude layout.")
    if any(
        value.energy_unit.unit_id != reference.energy_unit.unit_id
        or value.electric_dipole_unit.unit_id != reference.electric_dipole_unit.unit_id
        for value in (plus, minus)
    ):
        raise ValueError("Excited-state manifold units must remain unchanged.")
    derivative_vectors = (plus_amplitudes - minus_amplitudes) / (2.0 * step)
    coupling = contract(
        "ai,aj->ij",
        jnp.conj(reference_amplitudes),
        derivative_vectors,
    )
    coupling = 0.5 * (coupling - jnp.conj(coupling.T))
    energy_gap = (
        reference.excitation_energies[None, :] - reference.excitation_energies[:, None]
    )
    weighted = coupling * energy_gap
    residual = jnp.max(jnp.abs(coupling + jnp.conj(coupling.T)), initial=0.0)
    inverse_length = derived_unit(f"1/{length_unit.symbol}", ((length_unit, -1),))
    energy_weighted_unit = derived_unit(
        f"{reference.energy_unit.symbol}/{length_unit.symbol}",
        ((reference.energy_unit, 1), (length_unit, -1)),
    )
    successful = (
        bool(reference.successful)
        and bool(plus.successful)
        and bool(minus.successful)
        and bool(residual <= float(antisymmetry_tolerance))
    )
    return NonadiabaticCouplingResult(
        coupling,
        weighted,
        residual,
        successful,
        inverse_length,
        energy_weighted_unit,
    )


__all__ = [
    "ExcitedStateManifoldPlan",
    "NonadiabaticCouplingResult",
    "TammDancoffPlan",
    "finite_difference_nonadiabatic_coupling",
]
