#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit LocalHamiltonian target lowering for canonical quantum lattices."""

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._register import HilbertRegisterLayout
from ..operators.quantum.lattice._compile import PreparedQuantumLattice
from ..operators.quantum.lattice._strings import monomial_product_factors
from ._local_hamiltonian import LocalHamiltonian, LocalHamiltonianTerm


class LocalHamiltonianQuantumLatticePolicy(StrictModule):
    """Hard per-term ambient support limit for this explicit target."""

    maximum_term_matrix_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_term_matrix_elements: int):
        limit = int(maximum_term_matrix_elements)
        if limit < 1:
            raise ValueError("maximum_term_matrix_elements must be positive.")
        self.maximum_term_matrix_elements = limit
        self.policy_id = canonical_fingerprint(
            {
                "kind": "local-hamiltonian-quantum-lattice-policy",
                "maximum_term_matrix_elements": limit,
            }
        )


class LocalHamiltonianQuantumLatticeEvidence(StrictModule):
    term_matrix_elements: tuple[int, ...] = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)


class LocalHamiltonianQuantumLatticeResult(StrictModule):
    hamiltonian: LocalHamiltonian
    evidence: LocalHamiltonianQuantumLatticeEvidence = eqx.field(static=True)


def _active_product(
    prepared: PreparedQuantumLattice, monomial, /
) -> tuple[tuple[str, ...], tuple[jnp.ndarray, ...]]:
    factors = monomial_product_factors(prepared, monomial)
    active = tuple(
        index
        for index, (space, matrix) in enumerate(
            zip(prepared.specification.spaces, factors, strict=True)
        )
        if not np.allclose(
            np.asarray(matrix), np.eye(space.dimension), rtol=1e-12, atol=1e-12
        )
    )
    if not active:
        active = (0,)
    return (
        tuple(prepared.specification.site_ids[index] for index in active),
        tuple(factors[index] for index in active),
    )


def _kronecker(values, /):
    result = jnp.asarray([[1.0 + 0.0j]])
    for value in values:
        result = jnp.kron(result, value)
    return result


def lower_quantum_lattice_to_local_hamiltonian(
    prepared: PreparedQuantumLattice,
    policy: LocalHamiltonianQuantumLatticePolicy,
    /,
) -> LocalHamiltonianQuantumLatticeResult:
    """Lower exactly to the existing finite-register LocalHamiltonian target."""
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    if not isinstance(policy, LocalHamiltonianQuantumLatticePolicy):
        raise TypeError("policy must be LocalHamiltonianQuantumLatticePolicy.")
    if not prepared.specification.self_adjoint:
        raise ValueError(
            "LocalHamiltonian lowering requires a self-adjoint specification."
        )
    layout = HilbertRegisterLayout(
        prepared.specification.site_ids, prepared.specification.local_dimensions
    )
    groups: dict[str, list] = {}
    for monomial in prepared.monomials:
        groups.setdefault(monomial.source_term_id, []).append(monomial)
    terms = []
    matrix_elements = []
    for term_id, monomials in groups.items():
        target_ids, factors = _active_product(prepared, monomials[0])
        target_dimension = prod(
            prepared.specification.space(site).dimension for site in target_ids
        )
        entries = target_dimension * target_dimension
        if entries > policy.maximum_term_matrix_elements:
            raise ValueError(
                f"LocalHamiltonian term requires {entries} matrix elements, exceeding "
                f"the admitted limit {policy.maximum_term_matrix_elements}."
            )
        generator = monomials[0].coefficient * _kronecker(factors)
        for monomial in monomials[1:]:
            other_ids, other_factors = _active_product(prepared, monomial)
            if other_ids != target_ids:
                raise ValueError(
                    "Adjoint components produced inconsistent target support."
                )
            generator = generator + monomial.coefficient * _kronecker(other_factors)
        terms.append(
            LocalHamiltonianTerm(
                generator,
                target_ids,
                term_id=f"{prepared.prepared_id}:{term_id}",
            )
        )
        matrix_elements.append(entries)
    hamiltonian = LocalHamiltonian(layout, tuple(terms))
    evidence = LocalHamiltonianQuantumLatticeEvidence(
        term_matrix_elements=tuple(matrix_elements),
        exact=True,
        prepared_id=prepared.prepared_id,
        policy_id=policy.policy_id,
        lowering_id=canonical_fingerprint(
            {
                "kind": "local-hamiltonian-quantum-lattice-lowering",
                "prepared": prepared.prepared_id,
                "policy": policy.policy_id,
                "term_matrix_elements": matrix_elements,
            }
        ),
    )
    return LocalHamiltonianQuantumLatticeResult(hamiltonian, evidence)


__all__ = [
    "LocalHamiltonianQuantumLatticeEvidence",
    "LocalHamiltonianQuantumLatticePolicy",
    "LocalHamiltonianQuantumLatticeResult",
    "lower_quantum_lattice_to_local_hamiltonian",
]
