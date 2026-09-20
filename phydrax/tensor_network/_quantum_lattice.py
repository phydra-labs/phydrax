#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit exact-MPO target lowering for canonical quantum lattices."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum.lattice._compile import PreparedQuantumLattice
from ..operators.quantum.lattice._strings import monomial_product_factors
from ._core import MatrixProductOperator
from ._environments import mpo_hermiticity_residual
from ._mpo import add_mpo, scale_mpo


class QuantumLatticeMPOPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_bond_dimension: int,
        maximum_tensor_elements: int,
        hermiticity_tolerance: float = 1e-10,
    ):
        bond = int(maximum_bond_dimension)
        elements = int(maximum_tensor_elements)
        tolerance = float(hermiticity_tolerance)
        if bond < 1 or elements < 1 or tolerance < 0.0 or not jnp.isfinite(tolerance):
            raise ValueError("MPO resource limits and tolerance are invalid.")
        self.maximum_bond_dimension = bond
        self.maximum_tensor_elements = elements
        self.hermiticity_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-mpo-policy",
                "maximum_bond_dimension": bond,
                "maximum_tensor_elements": elements,
                "hermiticity_tolerance": tolerance,
            }
        )


class QuantumLatticeMPOEvidence(StrictModule):
    hermiticity_residual: Array
    hermitian: Array
    exact: bool = eqx.field(static=True)
    bond_dimension: int = eqx.field(static=True)
    tensor_elements: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)


class QuantumLatticeMPOResult(StrictModule):
    operator: MatrixProductOperator
    evidence: QuantumLatticeMPOEvidence


def _forecast_elements(dimensions: tuple[int, ...], term_count: int, /) -> int:
    if len(dimensions) == 1:
        return dimensions[0] * dimensions[0]
    return (
        term_count * dimensions[0] ** 2
        + term_count * dimensions[-1] ** 2
        + term_count * term_count * sum(value * value for value in dimensions[1:-1])
    )


def lower_quantum_lattice_to_mpo(
    prepared: PreparedQuantumLattice,
    policy: QuantumLatticeMPOPolicy,
    /,
) -> QuantumLatticeMPOResult:
    """Build the selected MPO target; no target registry or dispatcher is used."""
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    if not isinstance(policy, QuantumLatticeMPOPolicy):
        raise TypeError("policy must be QuantumLatticeMPOPolicy.")
    term_count = len(prepared.monomials)
    forecast = _forecast_elements(prepared.specification.local_dimensions, term_count)
    if term_count > policy.maximum_bond_dimension:
        raise ValueError(
            "Exact sum MPO exceeds maximum_bond_dimension before allocation."
        )
    if forecast > policy.maximum_tensor_elements:
        raise ValueError(
            "Exact sum MPO exceeds maximum_tensor_elements before allocation."
        )
    summands = []
    for monomial in prepared.monomials:
        local = monomial_product_factors(prepared, monomial)
        product = MatrixProductOperator(
            tuple(matrix[None, :, :, None] for matrix in local)
        )
        summands.append(scale_mpo(product, monomial.coefficient))
    operator = summands[0]
    for summand in summands[1:]:
        operator = add_mpo(operator, summand)
    actual_elements = sum(tensor.size for tensor in operator.tensors)
    if actual_elements > policy.maximum_tensor_elements:
        raise ValueError("Constructed MPO exceeds its predeclared tensor-element limit.")
    residual = mpo_hermiticity_residual(operator)
    hermitian = (
        prepared.specification.self_adjoint
        & jnp.isfinite(residual)
        & (residual <= policy.hermiticity_tolerance)
    )
    evidence = QuantumLatticeMPOEvidence(
        hermiticity_residual=residual,
        hermitian=hermitian,
        exact=True,
        bond_dimension=max((1,) + operator.bond_dimensions),
        tensor_elements=actual_elements,
        prepared_id=prepared.prepared_id,
        policy_id=policy.policy_id,
        lowering_id=canonical_fingerprint(
            {
                "kind": "quantum-lattice-mpo-lowering",
                "prepared": prepared.prepared_id,
                "policy": policy.policy_id,
                "term_count": term_count,
                "tensor_elements": actual_elements,
            }
        ),
    )
    return QuantumLatticeMPOResult(operator, evidence)


__all__ = [
    "QuantumLatticeMPOEvidence",
    "QuantumLatticeMPOPolicy",
    "QuantumLatticeMPOResult",
    "lower_quantum_lattice_to_mpo",
]
