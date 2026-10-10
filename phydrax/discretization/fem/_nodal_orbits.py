#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reference-owner nodal orbit labels, never physical nearest-node pairing."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from .._coordinate_enclosure import (
    _reference_image,
    owning_tabulator_source,
    source_basis,
)
from .._reference_cell import reference_cell_topology


if TYPE_CHECKING:
    from .._cell_mesh import CellMesh
    from ._reference import FiniteElementSpec


def _labels(element: FiniteElementSpec) -> tuple[tuple[Fraction, ...], ...]:
    from .._cell_geometry import _CoordinateTabulator
    from ._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from ._spectral_hp_completion import HybridReferenceFamily

    tabulator = element.tabulator
    topology = reference_cell_topology(element.cell_kind)
    if tabulator is None and element.family == "Lagrange":
        from .._coordinate_enclosure import coordinate_source_signature
        from ._reference import lagrange_element

        canonical = lagrange_element(element.cell_kind, element.degree)
        actual_nodes = np.asarray(element.reference_nodes)
        canonical_nodes = np.asarray(canonical.reference_nodes)
        if (
            canonical.tabulator is not None
            or coordinate_source_signature(element)
            != coordinate_source_signature(canonical)
            or actual_nodes.shape != canonical_nodes.shape
            or actual_nodes.dtype != canonical_nodes.dtype
            or not np.array_equal(
                actual_nodes.view(np.uint8), canonical_nodes.view(np.uint8)
            )
        ):
            raise ValueError(
                "A built-in nodal orbit must retain its canonical reference source."
            )
        return tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in canonical.reference_nodes
        )
    if isinstance(tabulator, _CoordinateTabulator):
        degree = tabulator.degree
        if degree == 1:
            return tuple(
                tuple(Fraction(float(value)) for value in row)
                for row in topology.vertices
            )
        if element.cell_kind == "pyramid":
            return tuple(
                (
                    Fraction(2 * i + k, 2 * degree),
                    Fraction(2 * j + k, 2 * degree),
                    Fraction(k, degree),
                )
                for k in range(degree + 1)
                for i in range(degree - k + 1)
                for j in range(degree - k + 1)
            )
        indices = tuple(product(range(degree + 1), repeat=topology.dimension))
        if element.cell_kind in (
            "triangle",
            "tetrahedron",
        ) or element.cell_kind.startswith("simplex:"):
            indices = tuple(row for row in indices if sum(row) <= degree)
        elif element.cell_kind == "prism":
            indices = tuple(row for row in indices if row[0] + row[1] <= degree)
        return tuple(tuple(Fraction(value, degree) for value in row) for row in indices)
    owner = owning_tabulator_source(tabulator)
    if isinstance(owner, SimplexNodalFamily):
        return tuple(
            tuple(Fraction(value, owner.order) for value in row[1:])
            for row in owner.multiindices
        )
    if isinstance(owner, ReferenceNodalFamily):
        return tuple(
            tuple(
                Fraction(value, order)
                for value, order in zip(row, owner.orders, strict=True)
            )
            for row in product(*(range(order + 1) for order in owner.orders))
        )
    if isinstance(owner, HybridReferenceFamily):
        return owner.nodal_reference_labels()
    raise ValueError(
        "Periodic H1 numbering requires an authoritative nodal reference owner."
    )


def _convention(element: FiniteElementSpec, dimension: int, arity: int) -> str:
    from .._cell_geometry import _CoordinateTabulator
    from ._high_order import ReferenceNodalFamily, SimplexNodalFamily
    from ._spectral_hp_completion import HybridReferenceFamily

    if element.tabulator is None and element.family == "Lagrange":
        # _labels has authenticated the actual implemented reference element.
        return "equispaced"
    if isinstance(element.tabulator, _CoordinateTabulator):
        return "equispaced"
    owner = owning_tabulator_source(element.tabulator)
    if isinstance(owner, ReferenceNodalFamily):
        return owner.node_set
    if isinstance(owner, SimplexNodalFamily):
        return "gauss-lobatto" if dimension == 1 else "warp-and-blend"
    if isinstance(owner, HybridReferenceFamily):
        if dimension == 1:
            return "gauss-lobatto"
        if dimension == 2:
            match arity:
                case 3:
                    return "warp-and-blend"
                case 4:
                    return "gauss-lobatto"
                case _:
                    raise ValueError(
                        "Hybrid nodal traces require actual triangular or quadrilateral facets."
                    )
        raise ValueError("Hybrid nodal orbit conventions require an edge or face.")
    raise ValueError("A nodal orbit must retain its actual reference-source convention.")


def nodal_orbit_keys(
    mesh: CellMesh, elements: tuple[FiniteElementSpec, ...]
) -> tuple[
    dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]], dict[tuple[int, int], str]
]:
    from .._cell_geometry import coordinate_lagrange_element
    from .._periodic_topology import _lifted_loops
    from ._generic import _h1_trace_positions

    loops = {
        degree: {
            int(row): tuple(int(value) for value in corners)
            for rows, group in _lifted_loops(mesh, degree)
            for row, corners in zip(rows, group, strict=True)
        }
        for degree in range(1, mesh.topological_dimension)
    }
    lookup = {
        degree: {tuple(sorted(corners)): row for row, corners in group.items()}
        for degree, group in loops.items()
    }
    result: dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]] = {}
    families: dict[tuple[int, int], str] = {}
    for block, element in zip(mesh.blocks, elements, strict=True):
        reference = reference_cell_topology(block.cell_kind)
        basis = source_basis(coordinate_lagrange_element(block.cell_kind, 1))
        if basis is None:
            raise ValueError("Nodal corner labels lack their canonical reference source.")
        weights = tuple(
            _reference_image(basis, block.cell_kind, point) for point in _labels(element)
        )
        vertices = np.asarray(block.vertices, dtype=np.int64)
        for degree in range(1, mesh.topological_dimension):
            entities = np.asarray(
                [
                    [
                        lookup[degree][tuple(sorted(int(row[i]) for i in corners))]
                        for corners in reference.entities[degree]
                    ]
                    for row in vertices
                ],
                dtype=np.int64,
            )
            positions = _h1_trace_positions(mesh, element, vertices, degree, entities)
            for local_entity, dofs in enumerate(element.entity_dofs[degree]):
                if not dofs:
                    continue
                for cell, row in enumerate(vertices):
                    entity = int(entities[cell, local_entity])
                    loop = loops[degree][entity]
                    slots = [int(np.flatnonzero(row == vertex)[0]) for vertex in loop]
                    keys: list[tuple[Fraction, ...] | None] = [None] * len(dofs)
                    for local_dof, position in zip(
                        dofs, positions[local_entity][cell], strict=True
                    ):
                        keys[int(position)] = tuple(
                            weights[local_dof][slot] for slot in slots
                        )
                    if any(key is None for key in keys):
                        raise ValueError(
                            "Canonical nodal labels do not cover their owning entity."
                        )
                    complete = tuple(key for key in keys if key is not None)
                    previous = result.get((degree, entity))
                    if previous is not None and previous != complete:
                        raise ValueError(
                            "Shared nodal reference labels are incompatible."
                        )
                    result[degree, entity] = complete
                    convention = _convention(
                        element, degree, len(reference.entities[degree][local_entity])
                    )
                    old_convention = families.get((degree, entity))
                    if old_convention is not None and old_convention != convention:
                        raise ValueError(
                            "Shared nodal traces have different scientific reference sources."
                        )
                    families[degree, entity] = convention
    return result, families
