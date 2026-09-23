#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact conserved-charge MPO lowering for canonical quantum lattices."""

from __future__ import annotations

from collections import Counter
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..operators.quantum._abelian_charge import AbelianGroup
from ..operators.quantum.lattice._compile import PreparedQuantumLattice
from ..operators.quantum.lattice._strings import monomial_product_factors
from ._abelian import AbelianLeg, AbelianTensor, AbelianTensorLayout
from ._abelian_core import AbelianMatrixProductOperator


class QuantumLatticeAbelianMPOPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)
    matrix_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_bond_dimension: int,
        maximum_tensor_elements: int,
        matrix_tolerance: float = 1.0e-12,
    ):
        bond = int(maximum_bond_dimension)
        elements = int(maximum_tensor_elements)
        tolerance = float(matrix_tolerance)
        if bond < 1 or elements < 1 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Abelian MPO resource limits and tolerance are invalid.")
        self.maximum_bond_dimension = bond
        self.maximum_tensor_elements = elements
        self.matrix_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "quantum-lattice-abelian-mpo-policy",
                "maximum_bond_dimension": bond,
                "maximum_tensor_elements": elements,
                "matrix_tolerance": tolerance,
            }
        )


class QuantumLatticeAbelianMPOEvidence(StrictModule):
    exact: bool = eqx.field(static=True)
    bond_dimension: int = eqx.field(static=True)
    tensor_elements: int = eqx.field(static=True)
    charge_labels: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)


class QuantumLatticeAbelianMPOResult(StrictModule):
    operator: AbelianMatrixProductOperator
    evidence: QuantumLatticeAbelianMPOEvidence


def _matrix_charge_delta(
    matrix: np.ndarray,
    charges: np.ndarray,
    tolerance: float,
    /,
) -> tuple[int, ...]:
    rows, columns = np.nonzero(np.abs(matrix) > tolerance)
    if rows.size == 0:
        return (0,) * charges.shape[1]
    differences = charges[rows] - charges[columns]
    unique = np.unique(differences, axis=0)
    if unique.shape[0] != 1:
        raise ValueError("A local MPO factor mixes distinct conserved-charge changes.")
    return tuple(int(value) for value in unique[0])


def _catalog(charges: tuple[tuple[int, ...], ...], /):
    counts = Counter(charges)
    ordered = tuple(sorted(counts))
    capacities = tuple(counts[value] for value in ordered)
    offsets = {}
    used = Counter()
    starts = {}
    start = 0
    for charge, capacity in zip(ordered, capacities, strict=True):
        starts[charge] = start
        start += capacity
    for term, charge in enumerate(charges):
        offsets[term] = starts[charge] + used[charge]
        used[charge] += 1
    return ordered, capacities, offsets


def lower_quantum_lattice_to_abelian_mpo(
    prepared: PreparedQuantumLattice,
    policy: QuantumLatticeAbelianMPOPolicy,
    /,
) -> QuantumLatticeAbelianMPOResult:
    if not isinstance(prepared, PreparedQuantumLattice):
        raise TypeError("prepared must be PreparedQuantumLattice.")
    if not isinstance(policy, QuantumLatticeAbelianMPOPolicy):
        raise TypeError("policy must be QuantumLatticeAbelianMPOPolicy.")
    spaces = prepared.specification.spaces
    labels = spaces[0].charge_labels
    if not labels or any(space.charge_labels != labels for space in spaces):
        raise ValueError(
            "Abelian MPO lowering requires one ordered charge roster on every site."
        )
    group = AbelianGroup((None,) * len(labels))
    local_products = tuple(
        tuple(np.asarray(value) for value in monomial_product_factors(prepared, monomial))
        for monomial in prepared.monomials
    )
    term_count = len(local_products)
    if term_count > policy.maximum_bond_dimension:
        raise ValueError("Abelian MPO term channels exceed maximum_bond_dimension.")
    site_count = len(spaces)
    cumulative = [[tuple(0 for _ in labels)] for _ in range(term_count)]
    for term, products in enumerate(local_products):
        running = np.zeros((len(labels),), dtype=np.int64)
        for site, matrix in enumerate(products):
            delta = np.asarray(
                _matrix_charge_delta(
                    matrix,
                    np.asarray(spaces[site].charges),
                    policy.matrix_tolerance,
                ),
                dtype=np.int64,
            )
            running = running + delta
            cumulative[term].append(tuple(int(-value) for value in running))
        if cumulative[term][-1] != group.zero:
            raise ValueError("An Abelian MPO monomial has nonzero total charge change.")
    cut_catalogs = []
    cut_offsets = []
    for cut in range(site_count + 1):
        if cut in (0, site_count):
            cut_catalogs.append(((group.zero,), (1,)))
            cut_offsets.append({term: 0 for term in range(term_count)})
        else:
            charges = tuple(cumulative[term][cut] for term in range(term_count))
            catalog, capacities, offsets = _catalog(charges)
            cut_catalogs.append((catalog, capacities))
            cut_offsets.append(offsets)
    forecast = 0
    for site, space in enumerate(spaces):
        left_size = sum(cut_catalogs[site][1])
        right_size = sum(cut_catalogs[site + 1][1])
        forecast += left_size * space.dimension * space.dimension * right_size
    if forecast > policy.maximum_tensor_elements:
        raise ValueError("Abelian MPO exceeds maximum_tensor_elements before allocation.")
    tensors = []
    for site, space in enumerate(spaces):
        left_charges, left_capacities = cut_catalogs[site]
        right_charges, right_capacities = cut_catalogs[site + 1]
        left_leg = AbelianLeg(group, left_charges, left_capacities, orientation=1)
        right_leg = AbelianLeg(group, right_charges, right_capacities, orientation=-1)
        physical_charges = tuple(
            tuple(int(value) for value in row) for row in np.asarray(space.charges)
        )
        output_leg = AbelianLeg(
            group,
            physical_charges,
            (1,) * space.dimension,
            orientation=-1,
        )
        input_leg = AbelianLeg(
            group,
            physical_charges,
            (1,) * space.dimension,
            orientation=1,
        )
        dense = np.zeros(
            (left_leg.size, space.dimension, space.dimension, right_leg.size),
            dtype=np.complex128,
        )
        for term, monomial in enumerate(prepared.monomials):
            coefficient = complex(np.asarray(monomial.coefficient)) if site == 0 else 1.0
            dense[
                cut_offsets[site][term],
                :,
                :,
                cut_offsets[site + 1][term],
            ] += coefficient * local_products[term][site]
        layout = AbelianTensorLayout((left_leg, output_leg, input_leg, right_leg))
        tensors.append(AbelianTensor.from_dense(layout, jnp.asarray(dense)))
    operator = AbelianMatrixProductOperator(tuple(tensors))
    actual = sum(
        prod(tensor.layout.block_shapes[index])
        for tensor in operator.tensors
        for index in range(len(tensor.blocks))
    )
    evidence = QuantumLatticeAbelianMPOEvidence(
        True,
        max((1,) + tuple(sum(value[1]) for value in cut_catalogs)),
        actual,
        labels,
        prepared.prepared_id,
        policy.policy_id,
        canonical_fingerprint(
            {
                "kind": "quantum-lattice-abelian-mpo-lowering",
                "prepared": prepared.prepared_id,
                "policy": policy.policy_id,
                "term_count": term_count,
                "tensor_elements": actual,
            }
        ),
    )
    return QuantumLatticeAbelianMPOResult(operator, evidence)


__all__ = [
    "QuantumLatticeAbelianMPOEvidence",
    "QuantumLatticeAbelianMPOPolicy",
    "QuantumLatticeAbelianMPOResult",
    "lower_quantum_lattice_to_abelian_mpo",
]
