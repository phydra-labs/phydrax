#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..tensor_network import (
    build_local_term_mpo,
    FiniteLocalTerm,
    FixedStructureMPOCoefficients,
    MatrixProductOperator,
    TensorNetworkPrecisionPolicy,
)
from ._local_hamiltonian import FixedGridLocalHamiltonian, LocalHamiltonian


class LocalHamiltonianMPOPolicy(StrictModule):
    """Exact product-factor lowering and bond-capacity contract."""

    maximum_bond_dimension: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_bond_dimension: int = 1024,
        hermiticity_tolerance: float = 1e-10,
    ):
        if isinstance(maximum_bond_dimension, bool) or not isinstance(
            maximum_bond_dimension, Integral
        ):
            raise TypeError("maximum_bond_dimension must be a positive integer.")
        if int(maximum_bond_dimension) <= 0:
            raise ValueError("maximum_bond_dimension must be positive.")
        tolerance = float(hermiticity_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("hermiticity_tolerance must be finite and non-negative.")
        self.maximum_bond_dimension = int(maximum_bond_dimension)
        self.hermiticity_tolerance = tolerance


class LocalHamiltonianMPOEvidence(StrictModule):
    """Exactness, factorization, Hermiticity, and bond evidence."""

    hermiticity_residual: Array
    hermitian: Array
    finite: Array
    valid: Array
    exact: bool = eqx.field(static=True)
    all_factored: bool = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    chain_order: tuple[str, ...] = eqx.field(static=True)


class LocalHamiltonianMPOResult(StrictModule):
    """Exact MPO sum and ordered single-term MPO coefficient basis."""

    operator: MatrixProductOperator
    basis_operators: tuple[MatrixProductOperator, ...]
    evidence: LocalHamiltonianMPOEvidence
    hamiltonian_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)


def _finite_term(
    hamiltonian: LocalHamiltonian,
    term_index: int,
    chain_order: tuple[str, ...],
    local_dimensions: tuple[int, ...],
    /,
) -> FiniteLocalTerm:
    term = hamiltonian.terms[term_index]
    positions = tuple(chain_order.index(wire) for wire in term.target_wire_ids)
    if term.product_factors is None:
        if len(positions) != 1:
            raise ValueError(
                "Multi-site local Hamiltonian terms require exact product_factors for MPO lowering."
            )
        factors = (term.generator,)
    else:
        factors = term.product_factors
    position_to_factor = {
        position: factor for position, factor in zip(positions, factors, strict=True)
    }
    start = min(positions)
    stop = max(positions)
    dtype = term.generator.dtype
    operators = tuple(
        position_to_factor.get(
            position,
            jnp.eye(local_dimensions[position], dtype=dtype),
        )
        for position in range(start, stop + 1)
    )
    return FiniteLocalTerm(start, operators)


def lower_local_hamiltonian_to_mpo(
    hamiltonian: LocalHamiltonian,
    chain_order: Sequence[str] | None = None,
    /,
    *,
    policy: LocalHamiltonianMPOPolicy | None = None,
    precision: TensorNetworkPrecisionPolicy | None = None,
) -> LocalHamiltonianMPOResult:
    """Lower exactly factored local terms into a declared finite-chain ordering."""

    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be a LocalHamiltonian.")
    order = (
        hamiltonian.layout.wire_ids
        if chain_order is None
        else tuple(str(wire) for wire in chain_order)
    )
    if len(order) != hamiltonian.layout.wire_count or set(order) != set(
        hamiltonian.layout.wire_ids
    ):
        raise ValueError("chain_order must be a permutation of register wire IDs.")
    selected = LocalHamiltonianMPOPolicy() if policy is None else policy
    if not isinstance(selected, LocalHamiltonianMPOPolicy):
        raise TypeError("policy must be a LocalHamiltonianMPOPolicy or None.")
    if len(hamiltonian.terms) > selected.maximum_bond_dimension:
        raise ValueError("Exact MPO sum exceeds maximum_bond_dimension.")
    dimensions = tuple(
        hamiltonian.layout.local_dimensions[hamiltonian.layout.wire_index(wire)]
        for wire in order
    )
    finite_terms = tuple(
        _finite_term(hamiltonian, index, order, dimensions)
        for index in range(len(hamiltonian.terms))
    )
    combined = build_local_term_mpo(
        dimensions,
        finite_terms,
        hermiticity_tolerance=selected.hermiticity_tolerance,
        precision=precision,
    )
    basis = tuple(
        build_local_term_mpo(
            dimensions,
            (term,),
            hermiticity_tolerance=selected.hermiticity_tolerance,
            precision=precision,
        ).operator
        for term in finite_terms
    )
    maximum_bond = max((1,) + combined.operator.bond_dimensions)
    finite = jnp.isfinite(combined.evidence.hermiticity_residual)
    valid = (
        hamiltonian.valid
        & finite
        & combined.evidence.hermitian
        & (maximum_bond <= selected.maximum_bond_dimension)
    )
    evidence = LocalHamiltonianMPOEvidence(
        combined.evidence.hermiticity_residual,
        combined.evidence.hermitian,
        finite,
        valid,
        True,
        True,
        maximum_bond,
        order,
    )
    lowering_id = canonical_fingerprint(
        {
            "kind": "local-hamiltonian-mpo-lowering",
            "hamiltonian": hamiltonian.hamiltonian_id,
            "chain_order": list(order),
            "maximum_bond_dimension": selected.maximum_bond_dimension,
            "precision": combined.operator.precision.policy_id,
        }
    )
    return LocalHamiltonianMPOResult(
        combined.operator,
        basis,
        evidence,
        hamiltonian.hamiltonian_id,
        lowering_id,
    )


def fixed_grid_local_hamiltonian_mpo_coefficients(
    schedule: FixedGridLocalHamiltonian,
    lowering: LocalHamiltonianMPOResult,
    /,
) -> FixedStructureMPOCoefficients:
    """Bind interval coefficients to an exact per-term MPO basis."""

    if not isinstance(schedule, FixedGridLocalHamiltonian):
        raise TypeError("schedule must be a FixedGridLocalHamiltonian.")
    if not isinstance(lowering, LocalHamiltonianMPOResult):
        raise TypeError("lowering must be a LocalHamiltonianMPOResult.")
    if schedule.hamiltonian.hamiltonian_id != lowering.hamiltonian_id:
        raise ValueError("Schedule Hamiltonian does not match the MPO lowering.")
    nodal_coefficients = jnp.concatenate(
        (schedule.coefficients, schedule.coefficients[-1:]),
        axis=0,
    )
    return FixedStructureMPOCoefficients(
        lowering.basis_operators,
        nodal_coefficients,
    )


__all__ = [
    "LocalHamiltonianMPOEvidence",
    "LocalHamiltonianMPOPolicy",
    "LocalHamiltonianMPOResult",
    "fixed_grid_local_hamiltonian_mpo_coefficients",
    "lower_local_hamiltonian_to_mpo",
]
