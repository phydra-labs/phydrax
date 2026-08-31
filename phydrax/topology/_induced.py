#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._homology import FiniteFieldBasis, HomologyResult
from ._maps import CellularChainMap


class FiniteFieldCoordinateMap(StrictModule, NonTrainableState):
    """Exact finite-field coordinates for one induced linear map."""

    matrix: Array
    degree: int = eqx.field(static=True)
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    field: PrimeField
    source_basis_id: str = eqx.field(static=True)
    target_basis_id: str = eqx.field(static=True)
    map_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        matrix: Sequence[Sequence[int]] | np.ndarray,
        field: PrimeField,
        /,
        *,
        source_basis_id: str,
        target_basis_id: str,
    ):
        values = np.asarray(matrix, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError("Finite-field coordinate maps must be matrices.")
        values %= field.modulus
        self.matrix = jnp.asarray(values)
        self.degree = int(degree)
        self.source_dimension = int(values.shape[1])
        self.target_dimension = int(values.shape[0])
        self.field = field
        self.source_basis_id = str(source_basis_id)
        self.target_basis_id = str(target_basis_id)
        self.map_id = canonical_fingerprint(
            {
                "kind": "finite-field-coordinate-map",
                "degree": int(degree),
                "field": field.field_id,
                "source_basis": self.source_basis_id,
                "target_basis": self.target_basis_id,
                "matrix": array_tree_fingerprint(values),
            }
        )


class InducedTopologyMap(StrictModule, NonTrainableState):
    """Exact induced homology and cohomology coordinates in named bases."""

    chain_map_id: str = eqx.field(static=True)
    homology_maps: tuple[FiniteFieldCoordinateMap, ...]
    cohomology_maps: tuple[FiniteFieldCoordinateMap, ...]
    field: PrimeField
    induced_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        chain_map_id: str,
        homology_maps: Sequence[FiniteFieldCoordinateMap],
        cohomology_maps: Sequence[FiniteFieldCoordinateMap],
        field: PrimeField,
        /,
    ):
        homology = tuple(homology_maps)
        cohomology = tuple(cohomology_maps)
        if tuple(value.degree for value in homology) != tuple(
            value.degree for value in cohomology
        ):
            raise ValueError("Induced homology and cohomology degrees must align.")
        self.chain_map_id = str(chain_map_id)
        self.homology_maps = homology
        self.cohomology_maps = cohomology
        self.field = field
        self.induced_map_id = canonical_fingerprint(
            {
                "kind": "induced-topology-map",
                "chain_map": self.chain_map_id,
                "field": field.field_id,
                "homology": [value.map_id for value in homology],
                "cohomology": [value.map_id for value in cohomology],
            }
        )


def _basis_vectors(
    basis: FiniteFieldBasis,
    layout,
    /,
) -> tuple[dict[int, int], ...]:
    vectors = [dict() for _ in range(basis.generator_count)]
    ambient_to_compact = np.asarray(layout.ambient_to_compact[basis.degree])
    for ambient, generator, coefficient in zip(
        np.asarray(basis.cell_indices),
        np.asarray(basis.generator_indices),
        np.asarray(basis.coefficients),
        strict=True,
    ):
        compact = int(ambient_to_compact[int(ambient)])
        if compact < 0:
            raise ValueError(
                "Homology basis addresses a cell outside its compact layout."
            )
        vectors[int(generator)][compact] = int(coefficient)
    return tuple(vectors)


def _pairing(
    cocycles: Sequence[dict[int, int]],
    cycles: Sequence[dict[int, int]],
    field: PrimeField,
    /,
) -> np.ndarray:
    matrix = np.zeros((len(cocycles), len(cycles)), dtype=np.int64)
    for row, cocycle in enumerate(cocycles):
        for column, cycle in enumerate(cycles):
            value = 0
            for cell, coefficient in cycle.items():
                value = field.add(
                    value,
                    field.multiply(cocycle.get(cell, 0), coefficient),
                )
            matrix[row, column] = value
    return matrix


def _solve_square(
    matrix: np.ndarray, right: np.ndarray, field: PrimeField, /
) -> np.ndarray:
    coefficients = np.asarray(matrix, dtype=object) % field.modulus
    values = np.asarray(right, dtype=object) % field.modulus
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError("Finite-field basis pairing must be square.")
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != coefficients.shape[0]:
        raise ValueError("Finite-field coordinate solve dimensions do not align.")
    size = coefficients.shape[0]
    augmented = np.concatenate((coefficients, values), axis=1)
    for pivot_column in range(size):
        pivot_row = next(
            (
                row
                for row in range(pivot_column, size)
                if int(augmented[row, pivot_column]) % field.modulus
            ),
            None,
        )
        if pivot_row is None:
            raise ValueError("Homology/cohomology basis pairing is singular.")
        if pivot_row != pivot_column:
            augmented[[pivot_column, pivot_row]] = augmented[[pivot_row, pivot_column]]
        inverse = field.inverse(int(augmented[pivot_column, pivot_column]))
        augmented[pivot_column] = [
            field.multiply(int(value), inverse) for value in augmented[pivot_column]
        ]
        for row in range(size):
            if row == pivot_column:
                continue
            factor = int(augmented[row, pivot_column]) % field.modulus
            if factor:
                augmented[row] = [
                    field.subtract(int(left), field.multiply(factor, int(right_value)))
                    for left, right_value in zip(
                        augmented[row], augmented[pivot_column], strict=True
                    )
                ]
    return np.asarray(augmented[:, size:], dtype=np.int64)


def _map_vector(matrix, vector: dict[int, int], field: PrimeField, /) -> dict[int, int]:
    output: dict[int, int] = {}
    columns = matrix.columns()
    for column, coefficient in vector.items():
        for row, value in columns[column].items():
            resolved = field.add(output.get(row, 0), field.multiply(coefficient, value))
            if resolved:
                output[row] = resolved
            elif row in output:
                del output[row]
    return output


def compute_induced_topology_map(
    chain_map: CellularChainMap,
    source: HomologyResult,
    target: HomologyResult,
    /,
) -> InducedTopologyMap:
    """Compute induced homology and cohomology matrices in explicit paired bases."""
    if source.field.field_id != target.field.field_id:
        raise ValueError("Induced maps require one shared coefficient field.")
    field = source.field
    if source.source_id != chain_map.source.subcomplex_id:
        raise ValueError("Source homology result does not match the chain map.")
    if target.source_id != chain_map.target.subcomplex_id:
        raise ValueError("Target homology result does not match the chain map.")
    source_by_degree = {value.degree: value for value in source.degrees}
    target_by_degree = {value.degree: value for value in target.degrees}
    common = tuple(sorted(set(source_by_degree) & set(target_by_degree)))
    homology_maps = []
    cohomology_maps = []
    for degree in common:
        source_degree = source_by_degree[degree]
        target_degree = target_by_degree[degree]
        if (
            source_degree.cycles is None
            or source_degree.cocycles is None
            or target_degree.cycles is None
            or target_degree.cocycles is None
        ):
            raise ValueError(
                "Induced maps require cycle and cocycle representatives in both results."
            )
        source_cycles = _basis_vectors(source_degree.cycles, chain_map.source.layout)
        source_cocycles = _basis_vectors(source_degree.cocycles, chain_map.source.layout)
        target_cycles = _basis_vectors(target_degree.cycles, chain_map.target.layout)
        target_cocycles = _basis_vectors(target_degree.cocycles, chain_map.target.layout)
        source_pairing = _pairing(source_cocycles, source_cycles, field)
        target_pairing = _pairing(target_cocycles, target_cycles, field)
        mapped_cycles = tuple(
            _map_vector(chain_map.degree_maps[degree], cycle, field)
            for cycle in source_cycles
        )
        evaluations = _pairing(target_cocycles, mapped_cycles, field)
        homology_coordinates = _solve_square(target_pairing, evaluations, field)
        pullback_evaluations = evaluations.T
        cohomology_coordinates = _solve_square(
            source_pairing.T,
            pullback_evaluations,
            field,
        )
        homology_maps.append(
            FiniteFieldCoordinateMap(
                degree,
                homology_coordinates,
                field,
                source_basis_id=source_degree.cycles.basis_id,
                target_basis_id=target_degree.cycles.basis_id,
            )
        )
        cohomology_maps.append(
            FiniteFieldCoordinateMap(
                degree,
                cohomology_coordinates,
                field,
                source_basis_id=target_degree.cocycles.basis_id,
                target_basis_id=source_degree.cocycles.basis_id,
            )
        )
    return InducedTopologyMap(
        chain_map.map_id,
        homology_maps,
        cohomology_maps,
        field,
    )


def induced_homology_coordinates(
    degree_maps,
    source_layout,
    target_layout,
    source: HomologyResult,
    target: HomologyResult,
    /,
) -> tuple[FiniteFieldCoordinateMap, ...]:
    """Compute only covariant homology coordinates for exact quotient maps."""
    if source.field.field_id != target.field.field_id:
        raise ValueError("Induced homology coordinates require one coefficient field.")
    field = source.field
    source_by_degree = {value.degree: value for value in source.degrees}
    target_by_degree = {value.degree: value for value in target.degrees}
    output = []
    for degree in sorted(set(source_by_degree) & set(target_by_degree)):
        source_degree = source_by_degree[degree]
        target_degree = target_by_degree[degree]
        if (
            source_degree.cycles is None
            or target_degree.cycles is None
            or target_degree.cocycles is None
        ):
            raise ValueError(
                "Induced homology coordinates require source cycles and paired "
                "target cycle/cocycle bases."
            )
        source_cycles = _basis_vectors(source_degree.cycles, source_layout)
        target_cycles = _basis_vectors(target_degree.cycles, target_layout)
        target_cocycles = _basis_vectors(target_degree.cocycles, target_layout)
        target_pairing = _pairing(target_cocycles, target_cycles, field)
        mapped = tuple(
            _map_vector(degree_maps[degree], cycle, field) for cycle in source_cycles
        )
        coordinates = _solve_square(
            target_pairing,
            _pairing(target_cocycles, mapped, field),
            field,
        )
        output.append(
            FiniteFieldCoordinateMap(
                degree,
                coordinates,
                field,
                source_basis_id=source_degree.cycles.basis_id,
                target_basis_id=target_degree.cycles.basis_id,
            )
        )
    return tuple(output)


__all__ = [
    "FiniteFieldCoordinateMap",
    "InducedTopologyMap",
    "compute_induced_topology_map",
    "induced_homology_coordinates",
]
