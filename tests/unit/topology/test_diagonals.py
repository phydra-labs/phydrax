#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import combinations

import numpy as np
import pytest

from phydrax.discretization._cell_complex import (
    cubical_cell_complex,
    simplicial_cell_complex,
)
from phydrax.discretization._topology import CellComplexTopology, OrientedIncidence
from phydrax.topology import (
    cell_vertex_support,
    CellDiagonalApproximation,
    cup_product,
    PrimeField,
)
from phydrax.topology._diagonals import alexander_whitney_diagonal, serre_diagonal


def _simplex_levels(dimension: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32)
        for degree in range(dimension + 1)
    )


def _diagonal_matrix(
    topology: CellComplexTopology,
    components: tuple[CellDiagonalApproximation, ...],
    degree: int,
) -> np.ndarray:
    counts = tuple(entity.count for entity in topology.entity_sets)
    rows = sum(counts[p] * counts[degree - p] for p in range(degree + 1))
    matrix = np.zeros((rows, counts[degree]), dtype=np.int64)
    offset = 0
    for p in range(degree + 1):
        component = next(
            value
            for value in components
            if value.source_degree == degree and value.left_degree == p
        )
        addresses = np.asarray(component.left_cells) * counts[degree - p] + np.asarray(
            component.right_cells
        )
        np.add.at(
            matrix,
            (offset + addresses, np.asarray(component.source_cells)),
            np.asarray(component.coefficients),
        )
        offset += counts[p] * counts[degree - p]
    return matrix


def _tensor_boundary(topology: CellComplexTopology, degree: int) -> np.ndarray:
    counts = tuple(entity.count for entity in topology.entity_sets)
    boundaries = tuple(
        incidence.scipy_boundary().toarray().astype(np.int64)
        for incidence in topology.incidences
    )
    source_sizes = [counts[p] * counts[degree - p] for p in range(degree + 1)]
    target_sizes = [counts[p] * counts[degree - 1 - p] for p in range(degree)]
    source_offsets = np.concatenate((np.asarray([0]), np.cumsum(source_sizes)))
    target_offsets = np.concatenate((np.asarray([0]), np.cumsum(target_sizes)))
    result = np.zeros((sum(target_sizes), sum(source_sizes)), dtype=np.int64)
    for p in range(degree + 1):
        q = degree - p
        columns = slice(source_offsets[p], source_offsets[p + 1])
        if p:
            rows = slice(target_offsets[p - 1], target_offsets[p])
            result[rows, columns] += np.kron(
                boundaries[p - 1], np.eye(counts[q], dtype=np.int64)
            )
        if q:
            rows = slice(target_offsets[p], target_offsets[p + 1])
            result[rows, columns] += (-1) ** p * np.kron(
                np.eye(counts[p], dtype=np.int64), boundaries[q - 1]
            )
    return result


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
def test_alexander_whitney_is_an_integral_chain_map(dimension: int) -> None:
    levels = _simplex_levels(dimension)
    topology = simplicial_cell_complex(levels)
    diagonal = alexander_whitney_diagonal(topology, cell_vertex_support(topology, levels))
    for degree in range(1, dimension + 1):
        np.testing.assert_array_equal(
            _tensor_boundary(topology, degree)
            @ _diagonal_matrix(topology, diagonal, degree),
            _diagonal_matrix(topology, diagonal, degree - 1)
            @ topology.incidences[degree - 1].scipy_boundary().toarray(),
        )


@pytest.mark.parametrize(
    ("shape", "periodic"),
    (
        ((2,), (False,)),
        ((2, 3), (False, True)),
        ((1, 1), (True, True)),
        ((2, 2, 2), (True, False, True)),
    ),
)
def test_serre_is_an_integral_chain_map(
    shape: tuple[int, ...],
    periodic: tuple[bool, ...],
) -> None:
    cubical = cubical_cell_complex(shape, periodic=periodic)
    diagonal = serre_diagonal(cubical)
    for degree in range(1, len(shape) + 1):
        np.testing.assert_array_equal(
            _tensor_boundary(cubical.topology, degree)
            @ _diagonal_matrix(cubical.topology, diagonal, degree),
            _diagonal_matrix(cubical.topology, diagonal, degree - 1)
            @ cubical.topology.incidences[degree - 1].scipy_boundary().toarray(),
        )


def _cup(
    topology: CellComplexTopology,
    diagonal: tuple[CellDiagonalApproximation, ...],
    left: np.ndarray,
    right: np.ndarray,
    p: int,
    q: int,
) -> np.ndarray:
    component = next(
        value for value in diagonal if value.left_degree == p and value.right_degree == q
    )
    return np.asarray(
        cup_product(
            left,
            right,
            component,
            coefficients=PrimeField(101),
            left_topology_id=topology.topology_id,
            right_topology_id=topology.topology_id,
        ),
        dtype=np.int64,
    )


@pytest.mark.parametrize("simplicial", (True, False))
def test_cup_associativity_and_graded_leibniz(simplicial: bool) -> None:
    if simplicial:
        levels = _simplex_levels(3)
        topology = simplicial_cell_complex(levels)
        diagonal = alexander_whitney_diagonal(
            topology, cell_vertex_support(topology, levels)
        )
    else:
        cubical = cubical_cell_complex((2, 2, 2), periodic=(True, False, True))
        topology = cubical.topology
        diagonal = serre_diagonal(cubical)
    counts = tuple(entity.count for entity in topology.entity_sets)
    boundaries = tuple(
        incidence.scipy_boundary().toarray().astype(np.int64)
        for incidence in topology.incidences
    )
    values = tuple((np.arange(count, dtype=np.int64) * 17 + 9) % 101 for count in counts)
    for p in range(3):
        for q in range(3 - p):
            left, right = values[p], values[q][::-1]
            lhs = boundaries[p + q].T @ _cup(topology, diagonal, left, right, p, q)
            rhs = _cup(topology, diagonal, boundaries[p].T @ left, right, p + 1, q) + (
                -1
            ) ** p * _cup(topology, diagonal, left, boundaries[q].T @ right, p, q + 1)
            np.testing.assert_array_equal(lhs % 101, rhs % 101)
    a, b, c = values[1], values[1][::-1], values[1] + 3
    np.testing.assert_array_equal(
        _cup(topology, diagonal, _cup(topology, diagonal, a, b, 1, 1), c, 2, 1),
        _cup(topology, diagonal, a, _cup(topology, diagonal, b, c, 1, 1), 1, 2),
    )


def test_periodic_torus_generators_have_nonzero_skew_cup_product() -> None:
    cubical = cubical_cell_complex((1, 1), periodic=(True, True))
    topology = cubical.topology
    diagonal = serre_diagonal(cubical)
    x = np.asarray([1, 0], dtype=np.int64)
    y = np.asarray([0, 1], dtype=np.int64)
    boundary = topology.incidences[1].scipy_boundary().toarray()
    np.testing.assert_array_equal(boundary.T @ x, np.asarray([0]))
    np.testing.assert_array_equal(boundary.T @ y, np.asarray([0]))
    np.testing.assert_array_equal(_cup(topology, diagonal, x, y, 1, 1), np.asarray([1]))
    np.testing.assert_array_equal(_cup(topology, diagonal, y, x, 1, 1), np.asarray([100]))
    np.testing.assert_array_equal(_cup(topology, diagonal, x, x, 1, 1), np.asarray([0]))
    np.testing.assert_array_equal(_cup(topology, diagonal, y, y, 1, 1), np.asarray([0]))


def test_alexander_whitney_respects_independent_cell_reorientation() -> None:
    levels = _simplex_levels(2)
    topology = simplicial_cell_complex(levels)
    orientation = (
        np.ones((3,), dtype=np.int64),
        np.asarray([-1, 1, -1]),
        np.asarray([-1]),
    )
    incidences = []
    for degree, original in enumerate(topology.incidences, start=1):
        signs = (
            np.asarray(original.signs)
            * orientation[degree - 1][np.asarray(original.relation.source_indices)]
            * orientation[degree][np.asarray(original.relation.target_indices)]
        )
        incidences.append(
            OrientedIncidence(
                degree,
                topology.entity_sets[degree - 1],
                topology.entity_sets[degree],
                original.relation,
                signs,
            )
        )
    reoriented = CellComplexTopology(topology.entity_sets, incidences)
    original_diagonal = alexander_whitney_diagonal(
        topology, cell_vertex_support(topology, levels)
    )
    changed_diagonal = alexander_whitney_diagonal(
        reoriented, cell_vertex_support(reoriented, levels)
    )
    np.testing.assert_array_equal(
        _cup(
            reoriented,
            changed_diagonal,
            np.asarray([2, 3, 4]) * orientation[1],
            np.asarray([5, 6, 7]) * orientation[1],
            1,
            1,
        ),
        _cup(
            topology,
            original_diagonal,
            np.asarray([2, 3, 4]),
            np.asarray([5, 6, 7]),
            1,
            1,
        )
        * orientation[2]
        % 101,
    )


def test_alexander_whitney_refuses_unrelated_support() -> None:
    levels = _simplex_levels(2)
    first = simplicial_cell_complex(levels, topology_id="first")
    second = simplicial_cell_complex(levels, topology_id="second")
    with pytest.raises(ValueError, match="topology_id"):
        alexander_whitney_diagonal(first, cell_vertex_support(second, levels))
