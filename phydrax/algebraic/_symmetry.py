#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._exact_integer import smith_normal_decomposition, smith_rank


if TYPE_CHECKING:
    from ._system import SparsePolynomialSupport, SparsePolynomialSystem


class ExponentLatticeScalingEvidence(StrictModule, NonTrainableState):
    """Exact support-lattice evidence for diagonal complex scaling symmetries.

    This records algebraic evidence only. It does not certify solution completeness,
    remove variables, or authorize quotienting a root set.
    """

    support_id: str = eqx.field(static=True)
    relation_matrix: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    lattice_rank: int = eqx.field(static=True)
    free_rank: int = eqx.field(static=True)
    free_generators: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    torsion_orders: tuple[int, ...] = eqx.field(static=True)
    torsion_generators: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    status: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        support_id: str,
        relation_matrix: Sequence[Sequence[int]],
        lattice_rank: int,
        free_generators: Sequence[Sequence[int]],
        torsion_orders: Sequence[int],
        torsion_generators: Sequence[Sequence[int]],
    ):
        support_id_ = str(support_id)
        relations = tuple(tuple(row) for row in relation_matrix)
        free = tuple(tuple(row) for row in free_generators)
        orders = tuple(torsion_orders)
        torsion = tuple(tuple(row) for row in torsion_generators)
        rank = int(lattice_rank)
        variable_count = (
            len(relations[0])
            if relations
            else len(free[0])
            if free
            else len(torsion[0])
            if torsion
            else rank
        )
        if not support_id_:
            raise ValueError("Scaling evidence requires a support identity.")
        if rank < 0 or rank > variable_count:
            raise ValueError("Exponent-lattice rank is inconsistent with its columns.")
        if any(len(row) != variable_count for row in relations + free + torsion):
            raise ValueError("Exponent-lattice evidence rows must share one width.")
        if len(free) != variable_count - rank:
            raise ValueError("Free scaling generators must span the exact kernel rank.")
        if len(orders) != len(torsion) or any(order <= 1 for order in orders):
            raise ValueError(
                "Torsion generators require aligned orders greater than one."
            )
        self.support_id = support_id_
        self.relation_matrix = relations
        self.lattice_rank = rank
        self.free_rank = variable_count - rank
        self.free_generators = free
        self.torsion_orders = orders
        self.torsion_generators = torsion
        self.status = "exact_support_lattice"
        self.backend = "phydrax-exact-smith-normal-decomposition"
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "exponent-lattice-scaling-evidence",
                "support": support_id_,
                "relations": [list(row) for row in relations],
                "lattice_rank": rank,
                "free_generators": [list(row) for row in free],
                "torsion_orders": list(orders),
                "torsion_generators": [list(row) for row in torsion],
                "status": self.status,
                "backend": self.backend,
            }
        )

    @property
    def connected_dimension(self) -> int:
        return self.free_rank


def _support_of(
    value: SparsePolynomialSupport | SparsePolynomialSystem,
    /,
) -> SparsePolynomialSupport:
    from ._system import SparsePolynomialSupport, SparsePolynomialSystem

    if isinstance(value, SparsePolynomialSupport):
        return value
    if isinstance(value, SparsePolynomialSystem):
        return value.support
    raise TypeError("Scaling-symmetry analysis requires polynomial support or a system.")


def _exponent_relations(
    support: SparsePolynomialSupport,
    /,
) -> tuple[tuple[int, ...], ...]:
    equation_indices = np.asarray(support.equation_indices)
    exponents = np.asarray(support.exponents)
    relations: list[tuple[int, ...]] = []
    for equation in range(support.equation_count):
        rows = exponents[equation_indices == equation]
        reference = rows[0]
        relations.extend(tuple(row - reference) for row in rows[1:])
    return tuple(relations)


def analyze_exponent_lattice_scaling(
    value: SparsePolynomialSupport | SparsePolynomialSystem,
    /,
) -> ExponentLatticeScalingEvidence:
    """Compute exact free and finite-component diagonal scaling evidence.

    Coefficients are intentionally not inspected: the evidence concerns the complete
    declared support, so explicitly zero coefficient slots cannot silently strengthen
    the reported symmetry.
    """
    support = _support_of(value)
    relations = _exponent_relations(support)
    variable_count = support.variable_count
    if not relations:
        rank = 0
        right = np.eye(variable_count, dtype=object)
        diagonal: tuple[int, ...] = ()
    else:
        smith, _, right = smith_normal_decomposition(relations)
        rank = smith_rank(smith)
        diagonal = tuple(abs(int(smith[index, index])) for index in range(rank))
    free_generators = tuple(
        tuple(int(right[row, column]) for row in range(variable_count))
        for column in range(rank, variable_count)
    )
    torsion_positions = tuple(
        index for index, invariant in enumerate(diagonal) if invariant > 1
    )
    torsion_orders = tuple(diagonal[index] for index in torsion_positions)
    torsion_generators = tuple(
        tuple(int(right[row, column]) for row in range(variable_count))
        for column in torsion_positions
    )
    return ExponentLatticeScalingEvidence(
        support_id=support.support_id,
        relation_matrix=relations,
        lattice_rank=rank,
        free_generators=free_generators,
        torsion_orders=torsion_orders,
        torsion_generators=torsion_generators,
    )


__all__ = [
    "analyze_exponent_lattice_scaling",
    "ExponentLatticeScalingEvidence",
]
