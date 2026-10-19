#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-element measures of admitted periodic quotient cells."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..linalg._small_batched import cofactor_determinant_2, cofactor_determinant_3
from ..typing import checked
from . import _cell_geometry as cell_geometry_owner, _cell_mesh as cell_mesh_owner
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    coordinate_corner_images,
    coordinate_expressions,
    prepared_coordinate_source_bank,
    RationalPolynomial,
)
from ._exact_algebraic_geometry import (
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
    reserve_algebraic_work,
)
from ._periodic_cell import PeriodicCell
from ._periodic_exact import PeriodicIsometryGroup
from ._periodic_topology import PeriodicMeasureReport


if TYPE_CHECKING:
    from .fem._cell_map import PreparedFiniteElementCellMap


@eqx.filter_jit
def _integrate_orbit_block(
    cell_map: PreparedFiniteElementCellMap,
    coordinates: Array,
    points: Array,
    weights: Array,
) -> tuple[Array, Array]:
    """Integrate one prepared support with the cell-map owner's full evidence."""
    point_count = weights.shape[0]
    evaluation = cell_map.evaluate(
        coordinates,
        jnp.repeat(jnp.arange(cell_map.cell_count, dtype=jnp.int32), point_count),
        jnp.tile(points, (cell_map.cell_count, 1)),
    )
    density = evaluation.measure.reshape((cell_map.cell_count, point_count))
    return density @ weights, jnp.all(evaluation.valid.reshape(density.shape), axis=1)


def _exact_affine_orbit_measures(
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicMeasureReport:
    """Integrate constant exact determinants before rounding the published report."""
    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "periodic_orbit_measures.mesh must retain periodic topology; got none"
        )
    dimension = mesh.ambient_dimension
    if mesh.topological_dimension != dimension or dimension not in (2, 3):
        raise ValueError(
            "periodic_orbit_measures.geometry must use full-dimensional 2D or 3D simplices; got incompatible dimensions"
        )
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    source = prepared_coordinate_source_bank(geometry, budget=budget)
    measures: list[Fraction | RadicalElement] = []
    valid = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind != ("triangle" if dimension == 2 else "tetrahedron"):
            raise ValueError(
                f"periodic_orbit_measures.cell_kind must be a full-dimensional simplex; got {block.cell_kind!r}"
            )
        for row in np.asarray(route):
            local = tuple(source[int(index)] for index in row if index >= 0)
            expressions = coordinate_expressions(element, local, budget=budget)
            if expressions is None or any(
                isinstance(value, RationalPolynomial)
                or any(sum(index) > 1 for index in value)
                for value in expressions
            ):
                raise ValueError(
                    "periodic_orbit_measures.geometry must use affine coordinate maps; got a nonaffine source"
                )
            corners = coordinate_corner_images(element, local, budget=budget)
            if corners is None:
                raise ValueError(
                    "periodic_orbit_measures.geometry must expose corner images; got no corner capability"
                )
            reserve_algebraic_work(
                corners, 64 * dimension, 32 * dimension, dimension + 1, budget=budget
            )
            values = np.asarray(corners, dtype=object)
            edges = values[1:] - values[0]
            determinant = (
                cofactor_determinant_2(edges[0], edges[1])
                if dimension == 2
                else cofactor_determinant_3(edges[0], edges[1], edges[2])
            )
            measures.append(abs(determinant) / math.factorial(dimension))
            valid.append(determinant > 0)
    reserve_algebraic_work(
        tuple(measures), 4 * len(measures), 4 * len(measures), 2, budget=budget
    )
    total = sum(measures, Fraction(0))

    def rounded(value: Fraction | RadicalElement) -> float:
        return (
            value.rounded(budget=budget)[0]
            if isinstance(value, RadicalElement)
            else float(value)
        )

    lattice = None
    defect = None
    if isinstance(periodic.cell, PeriodicCell):
        vectors = np.asarray(
            [
                [Fraction(float(value)) for value in row]
                for row in np.asarray(periodic.cell.vectors)
            ],
            dtype=object,
        )
        reserve_algebraic_work(
            tuple(vectors.flat),
            64 * dimension,
            32 * dimension,
            dimension,
            budget=budget,
        )
        lattice_exact = abs(
            cofactor_determinant_2(vectors[0], vectors[1])
            if dimension == 2
            else cofactor_determinant_3(vectors[0], vectors[1], vectors[2])
        )
        lattice = float(lattice_exact)
        if periodic.cell.fully_periodic and periodic.cell.rank == dimension:
            defect = rounded(abs(total - lattice_exact) / lattice_exact)
    return PeriodicMeasureReport(
        orbit_measures=jnp.asarray(
            [rounded(value) for value in measures], dtype=jnp.float64
        ),
        valid=jnp.asarray(valid, dtype=jnp.bool_),
        total_measure=jnp.asarray(rounded(total), dtype=jnp.float64),
        lattice_measure=lattice,
        relative_coverage_defect=defect,
        periodic_topology_id=periodic.periodic_topology_id,
        geometry_layout_id=geometry.geometry_layout_id,
    )


@checked
def periodic_orbit_measures(
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PeriodicMeasureReport:
    """Integrate the measure of every quotient top cell exactly once.

    Each lifted top cell is the unique published member of its orbit, so the
    finite-element coordinate map of its local lift integrates the orbit once.
    Quadrature uses the reference rule for the coordinate element's determinant degree.
    Declared algebraic affine simplices instead evaluate source determinants,
    signs, and their sum in the coefficient field before rounding this report.
    """
    # FEM transfer preparation imports periodic meshing through surface source support.
    # Resolve this one dependency only after both meshing and FEM modules initialize.
    from .fem import (
        _cell_map as cell_map_owner,
        _generic as generic_owner,
        _reference as reference_owner,
        _reference_rules as reference_rules_owner,
    )

    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "periodic_orbit_measures.mesh must retain a periodic topology; got a nonperiodic carrier"
        )
    spec = (
        cell_geometry_owner.CellGeometrySpec.affine(mesh)
        if geometry is None
        else geometry
    )
    if isinstance(spec.exact_source, ExactAlgebraicCellGeometrySource):
        if budget is None:
            from ..geometry._embedding_state import (
                coordinate_certificate_request,
                MeshCertificateLimits,
            )

            with coordinate_certificate_request(MeshCertificateLimits()) as owned:
                return _exact_affine_orbit_measures(mesh, spec, budget=owned)
        return _exact_affine_orbit_measures(mesh, spec, budget=budget)
    discretization = generic_owner.FiniteElementPlan(
        mesh,
        generic_owner.FiniteElementFieldSpec(
            "periodic-measure",
            {
                block.name: reference_owner.discontinuous_element(block.cell_kind, 0)
                for block in mesh.blocks
            },
        ),
        coordinate_spec=spec,
    ).prepare()
    coordinates = discretization.default_runtime.coordinates
    # The cell-map owner validates one coordinate-element identity and preserves
    # block/cell order on its homogeneous whole-support route. Do not replace
    # distinct restricted source actions with a common degree or corner carrier.
    homogeneous = (
        len({element.element_id for element in discretization.coordinate_elements}) == 1
    )
    block_indices = (None,) if homogeneous else tuple(range(len(mesh.blocks)))
    measures = []
    valid = []
    for index in block_indices:
        block = mesh.blocks[0 if index is None else index]
        cell_map = cell_map_owner.PreparedFiniteElementCellMap(discretization, index)
        points, weights = reference_rules_owner.degree_aware_reference_rule(
            block.cell_kind,
            cell_map.reference_dimension * cell_map.coordinate_element.degree,
        )
        block_measures, block_valid = _integrate_orbit_block(
            cell_map, coordinates, points, weights
        )
        measures.append(block_measures)
        valid.append(block_valid)
    orbit_measures = jnp.concatenate(measures)
    total = jnp.sum(orbit_measures)
    cell = periodic.cell
    match cell:
        case PeriodicCell():
            lattice = cell.cell_measure
            comparable = (
                cell.fully_periodic
                and cell.rank == mesh.topological_dimension == mesh.ambient_dimension
            )
        case PeriodicIsometryGroup():
            lattice = None
            comparable = False
        case _:
            raise TypeError(
                "periodic_orbit_measures.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )
    return PeriodicMeasureReport(
        orbit_measures=orbit_measures,
        valid=jnp.concatenate(valid),
        total_measure=total,
        lattice_measure=lattice,
        relative_coverage_defect=(
            abs(float(total) - lattice) / lattice
            if comparable and lattice is not None
            else None
        ),
        periodic_topology_id=periodic.periodic_topology_id,
        geometry_layout_id=discretization.default_runtime.geometry_layout_id,
    )


__all__ = ["periodic_orbit_measures"]
