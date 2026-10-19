#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reference-owner nodal orbit labels, never physical nearest-node pairing."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np

from phydrax.discretization.fem._hybrid_reference import HybridReferenceFamily

from ..._resource_ticket import HostBatchTicket
from ...typing import checked
from .. import (
    _cell_geometry as cell_geometry_owner,
    _coordinate_enclosure as coordinate_enclosure_owner,
    _periodic_topology as periodic_topology_owner,
)
from .._cell_mesh import CellMesh
from .._coordinate_budget import CoordinateEnclosureBudget
from .._coordinate_enclosure import (
    coordinate_estimate_bank,
    owning_tabulator_source,
    reference_image,
    require_rational_reference_scalar,
    source_basis,
    source_expression_estimates,
)
from .._reference_cell import reference_cell_topology, ReferenceCellTopology
from . import (
    _assembly as assembly_owner,
    _high_order as high_order_owner,
    _reference as reference_owner,
)
from ._reference_types import FiniteElementSpec


def _builtin_lagrange_labels(
    element: FiniteElementSpec,
) -> tuple[tuple[Fraction, ...], ...]:
    canonical = reference_owner.lagrange_element(element.cell_kind, element.degree)
    actual_nodes = np.asarray(element.reference_nodes)
    canonical_nodes = np.asarray(canonical.reference_nodes)
    if (
        canonical.tabulator is not None
        or coordinate_enclosure_owner.coordinate_source_signature(element)
        != coordinate_enclosure_owner.coordinate_source_signature(canonical)
        or actual_nodes.shape != canonical_nodes.shape
        or actual_nodes.dtype != canonical_nodes.dtype
        or not np.array_equal(actual_nodes.view(np.uint8), canonical_nodes.view(np.uint8))
    ):
        raise ValueError(
            "nodal_orbit.reference_source must match its built-in element; got different source metadata or nodes"
        )
    return tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in canonical.reference_nodes
    )


def _coordinate_labels(
    element: FiniteElementSpec,
    tabulator: cell_geometry_owner.CoordinateTabulator,
    topology: ReferenceCellTopology,
) -> tuple[tuple[Fraction, ...], ...]:
    degree = tabulator.degree
    if degree == 1:
        return tuple(
            tuple(Fraction(float(value)) for value in row) for row in topology.vertices
        )
    match element.cell_kind:
        case "pyramid":
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
        case kind:
            indices = tuple(product(range(degree + 1), repeat=topology.dimension))
            if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
                indices = tuple(row for row in indices if sum(row) <= degree)
            elif kind == "prism":
                indices = tuple(row for row in indices if row[0] + row[1] <= degree)
            return tuple(
                tuple(Fraction(value, degree) for value in row) for row in indices
            )


@checked
def labels(element: FiniteElementSpec) -> tuple[tuple[Fraction, ...], ...]:
    """Return reference labels authenticated against the nodal source."""
    tabulator = element.tabulator
    topology = reference_cell_topology(element.cell_kind)
    if tabulator is None and element.family == "Lagrange":
        return _builtin_lagrange_labels(element)
    match tabulator:
        case cell_geometry_owner.CoordinateTabulator():
            return _coordinate_labels(element, tabulator, topology)
        case _:
            owner = owning_tabulator_source(tabulator)
    match owner:
        case high_order_owner.SimplexNodalFamily():
            return tuple(
                tuple(Fraction(value, owner.order) for value in row[1:])
                for row in owner.multiindices
            )
        case high_order_owner.ReferenceNodalFamily():
            return tuple(
                tuple(
                    Fraction(value, order)
                    for value, order in zip(row, owner.orders, strict=True)
                )
                for row in product(*(range(order + 1) for order in owner.orders))
            )
        case HybridReferenceFamily():
            return owner.nodal_reference_labels()
        case _:
            raise ValueError(
                "nodal_orbit.reference_source must declare nodal labels; got an unsupported source"
            )


def _convention(element: FiniteElementSpec, dimension: int, arity: int) -> str:

    if element.tabulator is None and element.family == "Lagrange":
        # labels authenticated the implemented reference element.
        return "equispaced"
    match element.tabulator:
        case cell_geometry_owner.CoordinateTabulator():
            return "equispaced"
        case _:
            owner = owning_tabulator_source(element.tabulator)
    match owner:
        case high_order_owner.ReferenceNodalFamily():
            return owner.node_set
        case high_order_owner.SimplexNodalFamily():
            return "gauss-lobatto" if dimension == 1 else "warp-and-blend"
        case HybridReferenceFamily():
            match dimension, arity:
                case 1, _:
                    return "gauss-lobatto"
                case 2, 3:
                    return "warp-and-blend"
                case 2, 4:
                    return "gauss-lobatto"
                case 2, _:
                    raise ValueError(
                        f"nodal_orbit.facet_arity must be 3 or 4; got {arity}"
                    )
                case _:
                    raise ValueError(
                        f"nodal_orbit.dimension must be 1 or 2; got {dimension}"
                    )
        case _:
            raise ValueError(
                "nodal_orbit.reference_source must declare its node convention; got an unsupported source"
            )


def _register_entity_labels(
    result: dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]],
    families: dict[tuple[int, int], str],
    element: FiniteElementSpec,
    degree: int,
    entity: int,
    arity: int,
    row: np.ndarray,
    loop: tuple[int, ...],
    dofs: tuple[int, ...],
    positions: np.ndarray,
    weights: tuple[tuple[Fraction, ...], ...],
) -> None:
    slots = [int(np.flatnonzero(row == vertex)[0]) for vertex in loop]
    keys: list[tuple[Fraction, ...] | None] = [None] * len(dofs)
    for local_dof, position in zip(dofs, positions, strict=True):
        keys[int(position)] = tuple(weights[local_dof][slot] for slot in slots)
    if any(key is None for key in keys):
        raise ValueError(
            "nodal_orbit.labels must cover the entity; got missing positions"
        )
    complete = tuple(key for key in keys if key is not None)
    previous = result.get((degree, entity))
    if previous is not None and previous != complete:
        raise ValueError(
            "nodal_orbit.labels must agree on shared entities; got different reference labels"
        )
    result[degree, entity] = complete
    convention = _convention(element, degree, arity)
    old_convention = families.get((degree, entity))
    if old_convention is not None and old_convention != convention:
        raise ValueError(
            "nodal_orbit.reference_source must agree on shared traces; got different node conventions"
        )
    families[degree, entity] = convention


# Lifted entity identity and reference trace permutations follow deterministic block/cell traversal.
def _nodal_orbit_keys(
    mesh: CellMesh,
    elements: tuple[FiniteElementSpec, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]], dict[tuple[int, int], str]
]:

    loops = {
        degree: {
            int(row): tuple(int(value) for value in corners)
            for rows, group in periodic_topology_owner.lifted_loops(mesh, degree)
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
        basis = source_basis(
            cell_geometry_owner.coordinate_lagrange_element(block.cell_kind, 1),
            budget=budget,
        )
        if basis is None:
            raise ValueError(
                "nodal_orbit.corner_source must provide reference expressions; got an unsupported source"
            )
        weights = tuple(
            tuple(
                require_rational_reference_scalar(value)
                for value in reference_image(basis, block.cell_kind, point, budget=budget)
            )
            for point in labels(element)
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
            positions = assembly_owner.h1_trace_positions(
                mesh, element, vertices, degree, entities
            )
            for local_entity, dofs in enumerate(element.entity_dofs[degree]):
                if not dofs:
                    continue
                for cell in range(vertices.shape[0]):
                    # CellBlock owns rank-two vertex rows; slice/reshape preserves a rank-one array view.
                    row = vertices[cell : cell + 1].reshape((-1,))
                    entity = int(entities[cell, local_entity])
                    loop = loops[degree][entity]
                    _register_entity_labels(
                        result,
                        families,
                        element,
                        degree,
                        entity,
                        len(reference.entities[degree][local_entity]),
                        row,
                        loop,
                        dofs,
                        positions[local_entity][cell],
                        weights,
                    )
    return result, families


@checked
def nodal_orbit_keys(
    mesh: CellMesh,
    elements: tuple[FiniteElementSpec, ...],
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[
    dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]], dict[tuple[int, int], str]
]:
    """Prepare reference nodal orbit labels within a source-derived work bound."""

    if budget is not None:
        return _nodal_orbit_keys(mesh, elements, budget=budget)
    sources = []
    evaluation_work = 0
    for block, element in zip(mesh.blocks, elements, strict=True):
        basis = source_expression_estimates(
            cell_geometry_owner.coordinate_lagrange_element(block.cell_kind, 1)
        )
        if basis is None:
            raise ValueError(
                "nodal_orbit.corner_source must provide reference expressions; got an unsupported source"
            )
        sources.extend(basis)
        evaluation_work += len(labels(element)) * sum(value.terms for value in basis)
    estimate = coordinate_estimate_bank(tuple(sources))
    maximum_work = estimate.work_upper + evaluation_work
    with HostBatchTicket(maximum_work, bytes_upper=estimate.bytes_upper) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=estimate.bytes_upper,
        )
        with ledger.activate():
            return _nodal_orbit_keys(mesh, elements, budget=ledger)
