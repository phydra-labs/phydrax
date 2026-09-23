#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._filtration import CellFiltration
from ._homology import compute_homology, FiniteFieldBasis
from ._persistence import compute_persistence, PersistenceResult
from ._resources import TopologyResourceError, TopologyResourcePolicy


class TerminalCocycleAnnotation(StrictModule, NonTrainableState):
    """Terminal cohomology basis and compatible essential interval indices."""

    basis: FiniteFieldBasis
    essential_pair_indices: Array

    def __init__(
        self,
        basis: FiniteFieldBasis,
        essential_pair_indices,
        /,
    ):
        indices = jnp.asarray(essential_pair_indices, dtype=jnp.int32)
        if indices.shape != (basis.generator_count,):
            raise ValueError(
                "Terminal cocycle annotations must align with basis generators."
            )
        self.basis = basis
        self.essential_pair_indices = indices


class PersistentCohomologyResult(StrictModule, NonTrainableState):
    """Exact persistence intervals with terminal cocycle representatives."""

    persistence: PersistenceResult
    terminal_cocycles: tuple[FiniteFieldBasis, ...]
    annotations: tuple[TerminalCocycleAnnotation, ...]
    field: PrimeField
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        persistence: PersistenceResult,
        terminal_cocycles: tuple[FiniteFieldBasis, ...],
        annotations: tuple[TerminalCocycleAnnotation, ...],
        field: PrimeField,
        /,
    ):
        self.persistence = persistence
        self.terminal_cocycles = terminal_cocycles
        self.annotations = annotations
        self.field = field
        self.result_id = canonical_fingerprint(
            {
                "kind": "persistent-cohomology-result",
                "persistence": persistence.result_id,
                "field": field.field_id,
                "terminal_cocycles": [value.basis_id for value in terminal_cocycles],
                "annotations": [
                    {
                        "basis": value.basis.basis_id,
                        "pairs": np.asarray(value.essential_pair_indices).tolist(),
                    }
                    for value in annotations
                ],
            }
        )


def _finite_field_matmul(
    left: np.ndarray,
    right: np.ndarray,
    modulus: int,
    /,
) -> np.ndarray:
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            total = 0
            for inner in range(left.shape[1]):
                total = (
                    total + int(left[row, inner]) * int(right[inner, column])
                ) % modulus
            result[row, column] = total
    return result


def _finite_field_inverse(
    matrix: np.ndarray,
    field: PrimeField,
    /,
) -> np.ndarray:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise RuntimeError("Terminal cocycle pairing matrix must be square.")
    augmented = [
        [
            *(int(matrix[row, column]) % field.modulus for column in range(size)),
            *(1 if row == column else 0 for column in range(size)),
        ]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if augmented[row][column] % field.modulus
            ),
            None,
        )
        if pivot is None:
            raise RuntimeError(
                "Terminal cocycles do not pair nondegenerately with essential cycles."
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = field.inverse(augmented[column][column])
        augmented[column] = [field.multiply(scale, value) for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column] % field.modulus
            if factor:
                augmented[row] = [
                    field.subtract(value, field.multiply(factor, pivot_value))
                    for value, pivot_value in zip(
                        augmented[row],
                        augmented[column],
                        strict=True,
                    )
                ]
    return np.asarray([row[size:] for row in augmented], dtype=np.int64).reshape(
        (size, size)
    )


def _basis_coordinates(
    basis: FiniteFieldBasis,
    filtration: CellFiltration,
    /,
) -> np.ndarray:
    layout = filtration.complex.layout
    inverse = np.asarray(layout.ambient_to_compact[basis.degree], dtype=np.int32)
    coordinates = np.zeros(
        (basis.generator_count, layout.counts[basis.degree]),
        dtype=np.int64,
    )
    for cell, generator, coefficient in zip(
        np.asarray(basis.cell_indices),
        np.asarray(basis.generator_indices),
        np.asarray(basis.coefficients),
        strict=True,
    ):
        compact = int(inverse[int(cell)])
        if compact < 0:
            raise RuntimeError("Terminal cocycle addresses a cell outside its layout.")
        coordinates[int(generator), compact] = int(coefficient)
    return coordinates


def _essential_cycle_coordinates(
    persistence: PersistenceResult,
    filtration: CellFiltration,
    degree: int,
    pair_indices: np.ndarray,
    /,
) -> np.ndarray:
    representatives = persistence.pairing.representatives
    if representatives is None:
        raise RuntimeError("Persistent cohomology requires birth-cycle representatives.")
    layout = filtration.complex.layout
    inverse = np.asarray(layout.ambient_to_compact[degree], dtype=np.int32)
    order_degrees = np.asarray(persistence.pairing.order_degrees)
    order_ambient = np.asarray(persistence.pairing.order_ambient_indices)
    pair_to_row = {int(pair): row for row, pair in enumerate(pair_indices)}
    coordinates = np.zeros(
        (pair_indices.size, layout.counts[degree]),
        dtype=np.int64,
    )
    for order_index, pair_index, coefficient in zip(
        np.asarray(representatives.cell_order_indices),
        np.asarray(representatives.pair_indices),
        np.asarray(representatives.coefficients),
        strict=True,
    ):
        row = pair_to_row.get(int(pair_index))
        if row is None:
            continue
        order = int(order_index)
        if int(order_degrees[order]) != degree:
            raise RuntimeError(
                "Persistent cycle representative has an inconsistent degree."
            )
        compact = int(inverse[int(order_ambient[order])])
        if compact < 0:
            raise RuntimeError("Persistent cycle addresses a cell outside its layout.")
        coordinates[row, compact] = (
            coordinates[row, compact] + int(coefficient)
        ) % persistence.pairing.field.modulus
    return coordinates


def _persistence_dual_cocycles(
    basis: FiniteFieldBasis,
    persistence: PersistenceResult,
    filtration: CellFiltration,
    pair_indices: np.ndarray,
    field: PrimeField,
    resources: TopologyResourcePolicy,
    /,
) -> FiniteFieldBasis:
    generator_count = basis.generator_count
    cell_count = filtration.complex.layout.counts[basis.degree]
    representative_entries = generator_count * cell_count
    operations = 2 * generator_count * generator_count * cell_count + generator_count**3
    if representative_entries > resources.max_representative_entries:
        raise TopologyResourceError(
            "Persistent cocycle alignment exceeds max_representative_entries."
        )
    if operations > resources.max_operations:
        raise TopologyResourceError(
            "Persistent cocycle alignment exceeds max_operations."
        )
    cocycles = _basis_coordinates(basis, filtration)
    cycles = _essential_cycle_coordinates(
        persistence,
        filtration,
        basis.degree,
        pair_indices,
    )
    pairing = _finite_field_matmul(cocycles, cycles.T, field.modulus)
    change_of_basis = _finite_field_inverse(pairing, field)
    dual = _finite_field_matmul(change_of_basis, cocycles, field.modulus)
    vectors = tuple(
        {
            cell: int(coefficient)
            for cell, coefficient in enumerate(row)
            if int(coefficient) % field.modulus
        }
        for row in dual
    )
    return FiniteFieldBasis(
        basis.degree,
        "cochain",
        vectors,
        filtration.complex.layout,
        filtration.complex.topology,
        field,
    )


def compute_persistent_cohomology(
    filtration: CellFiltration,
    /,
    *,
    coefficients: PrimeField,
    max_degree: int | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> PersistentCohomologyResult:
    """Compute field-equivalent intervals and exact cocycles at the terminal complex."""
    policy = TopologyResourcePolicy() if resources is None else resources
    if not isinstance(policy, TopologyResourcePolicy):
        raise TypeError("resources must be a TopologyResourcePolicy.")
    maximum = filtration.max_degree if max_degree is None else int(max_degree)
    persistence = compute_persistence(
        filtration,
        coefficients=coefficients,
        max_degree=maximum,
        representatives="cycles",
        resources=policy,
    )
    terminal = compute_homology(
        filtration.complex,
        coefficients=coefficients,
        degrees=tuple(range(maximum + 1)),
        representatives="cocycles",
        resources=policy,
    )
    raw_cocycles = tuple(
        value.cocycles for value in terminal.degrees if value.cocycles is not None
    )
    degrees = np.asarray(persistence.pairing.degrees)
    finite = np.asarray(persistence.pairing.has_finite_death)
    cocycles = []
    annotations = []
    for basis in raw_cocycles:
        candidates = np.flatnonzero((degrees == basis.degree) & ~finite).astype(np.int32)
        if candidates.size != basis.generator_count:
            raise RuntimeError(
                "Terminal cocycle dimension does not match essential intervals."
            )
        aligned = _persistence_dual_cocycles(
            basis,
            persistence,
            filtration,
            candidates,
            coefficients,
            policy,
        )
        cocycles.append(aligned)
        annotations.append(TerminalCocycleAnnotation(aligned, candidates))
    return PersistentCohomologyResult(
        persistence,
        tuple(cocycles),
        tuple(annotations),
        coefficients,
    )


__all__ = [
    "PersistentCohomologyResult",
    "TerminalCocycleAnnotation",
    "compute_persistent_cohomology",
]
