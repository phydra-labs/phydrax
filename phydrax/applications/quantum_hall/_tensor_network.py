#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conserved-charge tensor-network execution for finite Hall cylinders."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...tensor_network import (
    abelian_finite_dmrg,
    abelian_product_mps,
    AbelianDMRGEvidence,
    AbelianMatrixProductState,
    lower_quantum_lattice_to_abelian_mpo,
    QuantumLatticeAbelianMPOPolicy,
    QuantumLatticeAbelianMPOResult,
)
from ._cylinder import PreparedHallCylinderHamiltonian


class HallCylinderDMRGPlan(StrictModule, NonTrainableState):
    prepared: PreparedHallCylinderHamiltonian
    initial_occupations: tuple[int, ...] = eqx.field(static=True)
    maximum_sweeps: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    descent_step: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_mpo_tensor_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedHallCylinderHamiltonian,
        initial_occupations: Sequence[int],
        /,
        *,
        maximum_sweeps: int = 20,
        maximum_bond_dimension: int = 128,
        descent_step: float = 0.05,
        residual_tolerance: float = 1.0e-8,
        maximum_mpo_tensor_elements: int = 100_000_000,
    ):
        if not isinstance(prepared, PreparedHallCylinderHamiltonian):
            raise TypeError("prepared must be PreparedHallCylinderHamiltonian.")
        occupations = tuple(int(value) for value in initial_occupations)
        sweeps = int(maximum_sweeps)
        bond = int(maximum_bond_dimension)
        step = float(descent_step)
        tolerance = float(residual_tolerance)
        elements = int(maximum_mpo_tensor_elements)
        if (
            len(occupations) != prepared.plan.orbital_count
            or any(value not in (0, 1) for value in occupations)
            or sum(occupations) != prepared.plan.particle_count
        ):
            raise ValueError(
                "Initial cylinder occupations must select the declared particle sector."
            )
        if (
            sweeps < 1
            or bond < 1
            or not isfinite(step)
            or step <= 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or elements < 1
        ):
            raise ValueError("Cylinder DMRG numerical controls are invalid.")
        self.prepared = prepared
        self.initial_occupations = occupations
        self.maximum_sweeps = sweeps
        self.maximum_bond_dimension = bond
        self.descent_step = step
        self.residual_tolerance = tolerance
        self.maximum_mpo_tensor_elements = elements
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hall-cylinder-dmrg-plan",
                "prepared": prepared.prepared_id,
                "initial_occupations": occupations,
                "maximum_sweeps": sweeps,
                "maximum_bond_dimension": bond,
                "descent_step": step,
                "residual_tolerance": tolerance,
                "maximum_mpo_tensor_elements": elements,
            }
        )


class HallCylinderDMRGResult(StrictModule):
    state: AbelianMatrixProductState
    evidence: AbelianDMRGEvidence
    lowering: QuantumLatticeAbelianMPOResult
    omitted_interaction_norm: jnp.ndarray
    successful: jnp.ndarray
    result_id: str = eqx.field(static=True)


def solve_hall_cylinder_dmrg(
    plan: HallCylinderDMRGPlan,
    /,
) -> HallCylinderDMRGResult:
    if not isinstance(plan, HallCylinderDMRGPlan):
        raise TypeError("plan must be HallCylinderDMRGPlan.")
    term_count = len(plan.prepared.prepared_lattice.monomials)
    lowering = lower_quantum_lattice_to_abelian_mpo(
        plan.prepared.prepared_lattice,
        QuantumLatticeAbelianMPOPolicy(
            maximum_bond_dimension=max(1, term_count),
            maximum_tensor_elements=plan.maximum_mpo_tensor_elements,
        ),
    )
    physical_legs = tuple(tensor.layout.legs[2] for tensor in lowering.operator.tensors)
    local_states = tuple(
        jnp.asarray((1.0, 0.0)) if occupation == 0 else jnp.asarray((0.0, 1.0))
        for occupation in plan.initial_occupations
    )
    initial = abelian_product_mps(
        local_states,
        physical_legs,
        plan.initial_occupations,
    )
    state, evidence = abelian_finite_dmrg(
        initial,
        lowering.operator,
        maximum_sweeps=plan.maximum_sweeps,
        maximum_bond_dimension=plan.maximum_bond_dimension,
        descent_step=plan.descent_step,
        residual_tolerance=plan.residual_tolerance,
    )
    successful = evidence.valid & evidence.converged
    return HallCylinderDMRGResult(
        state,
        evidence,
        lowering,
        plan.prepared.omitted_coefficient_norm,
        successful,
        canonical_fingerprint(
            {
                "kind": "hall-cylinder-dmrg-result",
                "plan": plan.plan_id,
                "lowering": lowering.evidence.lowering_id,
                "final_energy": float(np.asarray(evidence.energies[-1])),
                "final_residual": float(np.asarray(evidence.residual_norms[-1])),
            }
        ),
    )


__all__ = [
    "HallCylinderDMRGPlan",
    "HallCylinderDMRGResult",
    "solve_hall_cylinder_dmrg",
]
