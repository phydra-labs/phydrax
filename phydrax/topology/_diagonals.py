#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..discretization._cell_complex import CubicalCellComplex
from ..discretization._topology import CellComplexTopology
from ._advanced import CellDiagonalApproximation
from ._complex import CellVertexSupport


def _simplex_support(
    topology: CellComplexTopology, support: CellVertexSupport, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be a CellComplexTopology.")
    if not isinstance(support, CellVertexSupport):
        raise TypeError("support must be a CellVertexSupport.")
    if support.topology_id != topology.topology_id:
        raise ValueError("Vertex support and diagonal must share topology_id.")
    levels = []
    for degree, (entity, relation) in enumerate(
        zip(topology.entity_sets, support.relations, strict=True)
    ):
        cells: list[set[int]] = [set() for _ in range(entity.count)]
        valid = np.asarray(relation.valid, dtype=np.bool_)
        vertices = np.asarray(relation.source_indices)[valid]
        targets = np.asarray(relation.target_indices)[valid]
        for vertex, target in zip(vertices, targets, strict=True):
            cells[int(target)].add(int(vertex))
        rows = tuple(tuple(sorted(cell)) for cell in cells)
        if any(len(row) != degree + 1 for row in rows):
            raise ValueError("Alexander–Whitney requires simplicial vertex supports.")
        if len(set(rows)) != len(rows):
            raise ValueError("Alexander–Whitney requires distinct simplices per degree.")
        levels.append(rows)
    return tuple(levels)


def _simplex_orientations(
    topology: CellComplexTopology,
    levels: tuple[tuple[tuple[int, ...], ...], ...],
    /,
) -> tuple[np.ndarray, ...]:
    orientations = [np.ones((len(levels[0]),), dtype=np.int32)]
    for degree, incidence in enumerate(topology.incidences, start=1):
        lower_lookup = {row: index for index, row in enumerate(levels[degree - 1])}
        actual: dict[tuple[int, int], int] = {}
        valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
        lower = np.asarray(incidence.relation.source_indices)[valid]
        upper = np.asarray(incidence.relation.target_indices)[valid]
        signs = np.asarray(incidence.signs)[valid]
        for lower_cell, upper_cell, sign in zip(lower, upper, signs, strict=True):
            key = (int(lower_cell), int(upper_cell))
            if key in actual:
                raise ValueError("Simplicial diagonals require one route per face.")
            actual[key] = int(sign)
        oriented = np.zeros((len(levels[degree]),), dtype=np.int32)
        expected_count = 0
        for upper_cell, simplex in enumerate(levels[degree]):
            for removed in range(degree + 1):
                face = simplex[:removed] + simplex[removed + 1 :]
                if face not in lower_lookup:
                    raise ValueError("Simplicial support must be face closed.")
                lower_cell = lower_lookup[face]
                key = (lower_cell, upper_cell)
                if key not in actual:
                    raise ValueError("Simplicial incidence is missing a boundary face.")
                orientation = (
                    actual[key]
                    * (-1 if removed % 2 else 1)
                    * int(orientations[degree - 1][lower_cell])
                )
                if oriented[upper_cell] and oriented[upper_cell] != orientation:
                    raise ValueError("Incidence does not orient the declared simplex.")
                oriented[upper_cell] = orientation
                expected_count += 1
        if len(actual) != expected_count:
            raise ValueError("Simplicial incidence contains an undeclared face.")
        orientations.append(oriented)
    return tuple(orientations)


def alexander_whitney_diagonal(
    topology: CellComplexTopology, support: CellVertexSupport, /
) -> tuple[CellDiagonalApproximation, ...]:
    """Return all components of Δ[v₀,…,vₖ] = Σₚ[v₀,…,vₚ]⊗[vₚ,…,vₖ].

    Global vertex indices define the total order. Incidence, not support-route
    ordering, supplies each simplex orientation. Components are ordered by
    source degree and then left degree; each records its right degree.
    """
    levels = _simplex_support(topology, support)
    orientations = _simplex_orientations(topology, levels)
    lookups = tuple({row: index for index, row in enumerate(level)} for level in levels)
    result = []
    for source_degree, level in enumerate(levels):
        for left_degree in range(source_degree + 1):
            right_degree = source_degree - left_degree
            source_cells = np.arange(len(level), dtype=np.int32)
            left_cells = np.asarray(
                [lookups[left_degree][row[: left_degree + 1]] for row in level],
                dtype=np.int32,
            )
            right_cells = np.asarray(
                [lookups[right_degree][row[left_degree:]] for row in level],
                dtype=np.int32,
            )
            coefficients = (
                orientations[source_degree]
                * orientations[left_degree][left_cells]
                * orientations[right_degree][right_cells]
            )
            result.append(
                CellDiagonalApproximation(
                    topology,
                    source_degree,
                    left_degree,
                    right_degree,
                    source_cells,
                    left_cells,
                    right_cells,
                    coefficients,
                )
            )
    return tuple(result)


def _cubical_face_indices(
    cubical: CubicalCellComplex,
    degree: int,
    axes: tuple[int, ...],
    indices: np.ndarray,
    /,
) -> np.ndarray:
    block = cubical.orientations[degree].index(axes)
    shape = cubical.orientation_shapes[degree][block]
    wrapped = indices.copy()
    for axis, periodic in enumerate(cubical.periodic):
        if periodic:
            wrapped[:, axis] %= cubical.shape[axis]
    return (
        np.ravel_multi_index(tuple(wrapped[:, axis] for axis in range(len(shape))), shape)
        + cubical.orientation_offsets[degree][block]
    ).astype(np.int32)


def serre_diagonal(
    cubical: CubicalCellComplex, /
) -> tuple[CellDiagonalApproximation, ...]:
    """Tensor-product interval diagonal with the Koszul shuffle signs.

    Each interval factor has Δe = v₀⊗e + e⊗v₁. Choosing the left-varying
    axes A leaves the right face at the upper endpoint on A and the left
    face at the lower endpoint on its complement. Periodic identifications
    preserve the two occurrences even when their cell indices coincide.
    """
    if not isinstance(cubical, CubicalCellComplex):
        raise TypeError("cubical must be a CubicalCellComplex.")
    result = []
    for source_degree, orientation_blocks in enumerate(cubical.orientations):
        for left_degree in range(source_degree + 1):
            right_degree = source_degree - left_degree
            sources = []
            lefts = []
            rights = []
            signs = []
            for block, axes in enumerate(orientation_blocks):
                offset = cubical.orientation_offsets[source_degree][block]
                shape = cubical.orientation_shapes[source_degree][block]
                count = int(np.prod(shape, dtype=np.int64))
                indices = np.asarray(cubical.cell_multi_indices[source_degree])[
                    offset : offset + count
                ]
                source_cells = np.arange(offset, offset + count, dtype=np.int32)
                for left_axes in combinations(axes, left_degree):
                    right_axes = tuple(axis for axis in axes if axis not in left_axes)
                    right_indices = indices.copy()
                    for axis in left_axes:
                        right_indices[:, axis] += 1
                    inversions = sum(a > b for a in left_axes for b in right_axes)
                    sources.append(source_cells)
                    lefts.append(
                        _cubical_face_indices(cubical, left_degree, left_axes, indices)
                    )
                    rights.append(
                        _cubical_face_indices(
                            cubical, right_degree, right_axes, right_indices
                        )
                    )
                    signs.append(
                        np.full((count,), -1 if inversions % 2 else 1, dtype=np.int32)
                    )
            result.append(
                CellDiagonalApproximation(
                    cubical.topology,
                    source_degree,
                    left_degree,
                    right_degree,
                    np.concatenate(sources),
                    np.concatenate(lefts),
                    np.concatenate(rights),
                    np.concatenate(signs),
                )
            )
    return tuple(result)


__all__ = ["alexander_whitney_diagonal", "serre_diagonal"]
