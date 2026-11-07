#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-coordinate maps attached to individual mesh cells."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations
from typing import cast

import numpy as np

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    BernsteinDomain,
    evaluate_polynomial,
    Expression,
    expression_equal,
    linear_combinations,
    normalize_pyramid_reference_expression,
    polynomial_bounds,
    pyramid_reference_to_box_arguments,
    PyramidApexNormalization,
    RationalPolynomial,
)
from ..discretization._exact_algebraic_geometry import coefficient_endpoint, ExactScalar
from ..discretization._reference_cell import reference_cell_topology


class UnsupportedEmbeddingMap(ValueError):
    """The exact source map has no certified representation for this proof route."""


@cache
def _reference_rows(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    topology = reference_cell_topology(kind)
    return tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(topology.vertices, dtype=np.float64)
    )


@cache
def _reference_facets(kind: str, dimension: int, /) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in face)
        for face in reference_cell_topology(kind).entities[dimension - 1]
    )


_ZERO = Fraction(0)


@cache
def _unit_simplex_corners(dimension: int, /) -> frozenset[tuple[Fraction, ...]]:
    return frozenset(
        tuple(Fraction(int(axis == corner)) for axis in range(dimension))
        for corner in range(-1, dimension)
    )


def _linear_simplex_chart(
    coordinates: tuple[Expression, ...],
    references: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    /,
) -> (
    tuple[
        tuple[tuple[ExactScalar, ...], ...],
        tuple[ExactScalar, ...],
        tuple[tuple[ExactScalar, ...], ...],
        tuple[tuple[float, float], ...],
    ]
    | None
):
    """Exact corners, chart and outward bounds of rational degree-one simplex maps.

    On the unit simplex the corner images are the constant term plus at most one
    linear coefficient, the affine map through them is the polynomial itself (the
    chart `_solve_affine` returns), and the degree-one Bernstein controls are
    exactly the corner images.  `None` for any other map.
    """
    if len(references) != dimension + 1 or frozenset(references) != _unit_simplex_corners(
        dimension
    ):
        return None
    polynomials: list[Mapping[tuple[int, ...], ExactScalar]] = []
    for expression in coordinates:
        if isinstance(expression, RationalPolynomial) or any(
            sum(exponent) > 1 or not isinstance(value, Fraction)
            for exponent, value in expression.items()
        ):
            return None
        polynomials.append(expression)
    zero = (0,) * dimension
    units = tuple(
        tuple(int(axis == column) for axis in range(dimension))
        for column in range(dimension)
    )
    offset = tuple(polynomial.get(zero, _ZERO) for polynomial in polynomials)
    matrix = tuple(
        tuple(polynomial.get(unit, _ZERO) for unit in units) for polynomial in polynomials
    )
    corners = tuple(
        tuple(
            sum(
                (row[column] for column, value in enumerate(point) if value),
                offset[axis],
            )
            for axis, row in enumerate(matrix)
        )
        for point in references
    )
    bounds = tuple(
        (
            coefficient_endpoint(min(values), upper=False, saturate=True),
            coefficient_endpoint(max(values), upper=True, saturate=True),
        )
        for values in zip(*corners, strict=True)
    )
    return corners, offset, matrix, bounds


# Elements whose effective basis composes a per-cell reference chart.
_CELL_CHARTED = (
    PerCellRestrictedCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
)


def _straight_simplex_frame(
    element: ScalarCoordinateElement,
    references: tuple[tuple[Fraction, ...], ...],
    dimension: int,
    width: int,
    /,
) -> tuple[tuple[int, ...], int, tuple[int, ...]] | None:
    """Corner DOFs of an element mapping every cell affinely through them.

    A cell-independent basis of total degree at most one that is nodal at the
    unit-simplex corners maps each cell onto the simplex spanned by its corner
    coordinate rows: reference corner `k` lands exactly on the row of DOF
    `corner_dofs[k]`.  Returns `(corner_dofs, origin, units)` with the reference
    rows of the origin and of each unit vector, or `None` for any other element
    (whose cells keep their exact per-cell expressions).
    """
    if (
        isinstance(element, _CELL_CHARTED)
        or width != dimension + 1
        or len(references) != dimension + 1
        or frozenset(references) != _unit_simplex_corners(dimension)
    ):
        return None
    try:
        basis = coordinate_effective_basis(element)
    except (ValueError, ZeroDivisionError):
        return None
    if len(basis) != width or any(
        isinstance(function, RationalPolynomial)
        or any(
            sum(exponent) > 1 or not isinstance(value, Fraction)
            for exponent, value in function.items()
        )
        for function in basis
    ):
        return None
    corner_dofs: list[int] = []
    for point in references:
        values = tuple(evaluate_polynomial(function, point) for function in basis)
        nodal = tuple(dof for dof, value in enumerate(values) if value)
        if len(nodal) != 1 or values[nodal[0]] != 1:
            return None
        corner_dofs.append(nodal[0])
    origin = references.index((_ZERO,) * dimension)
    units = tuple(
        references.index(
            tuple(Fraction(int(axis == column)) for axis in range(dimension))
        )
        for column in range(dimension)
    )
    return tuple(corner_dofs), origin, units


def _pyramid_box_reference(
    point: tuple[Fraction, ...], /
) -> tuple[Fraction, Fraction, Fraction]:
    if len(point) != 3:
        raise UnsupportedEmbeddingMap(
            "Pyramid corners require three reference coordinates."
        )
    if point[2] == 1:
        if point[:2] != (Fraction(1, 2), Fraction(1, 2)):
            raise UnsupportedEmbeddingMap("A pyramid apex has noncanonical coordinates.")
        return Fraction(0), Fraction(0), Fraction(1)
    if point[2] == 0:
        return point[0], point[1], Fraction(0)
    raise UnsupportedEmbeddingMap(
        "Pyramid corners are outside the canonical reference strata."
    )


def _map_domain(kind: str, dimension: int) -> str:
    if kind in ("interval", "triangle", "tetrahedron") or kind.startswith("simplex:"):
        return "simplex"
    if kind == "prism":
        return "prism"
    if kind == "pyramid":
        if dimension != 3:
            raise UnsupportedEmbeddingMap(
                "A pyramid reference map must be three-dimensional."
            )
        return "pyramid"
    if kind in ("quadrilateral", "hexahedron") or kind.startswith("tensor:"):
        return "box"
    raise UnsupportedEmbeddingMap(f"No reference-domain map for {kind!r}.")


def _solve_affine(
    references: tuple[tuple[Fraction, ...], ...],
    images: tuple[tuple[ExactScalar, ...], ...],
) -> tuple[tuple[ExactScalar, ...], tuple[tuple[ExactScalar, ...], ...]]:
    dimension = len(references[0])
    ambient = len(images[0])
    for selected in combinations(range(len(references)), dimension + 1):
        rows = [[Fraction(1), *references[index]] for index in selected]
        inverse_rhs = [
            [images[index][axis] for axis in range(ambient)] for index in selected
        ]
        augmented = [row + rhs for row, rhs in zip(rows, inverse_rhs, strict=True)]
        for column in range(dimension + 1):
            pivot = next(
                (row for row in range(column, dimension + 1) if augmented[row][column]),
                None,
            )
            if pivot is None:
                break
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            scale = augmented[column][column]
            augmented[column] = [value / scale for value in augmented[column]]
            for row in range(dimension + 1):
                if row == column:
                    continue
                factor = augmented[row][column]
                if factor:
                    augmented[row] = [
                        value - factor * unit
                        for value, unit in zip(
                            augmented[row], augmented[column], strict=True
                        )
                    ]
        else:
            coefficients = tuple(
                tuple(
                    augmented[row][dimension + 1 + axis] for row in range(dimension + 1)
                )
                for axis in range(ambient)
            )
            return (
                tuple(row[0] for row in coefficients),
                tuple(
                    tuple(row[column + 1] for column in range(dimension))
                    for row in coefficients
                ),
            )
    raise UnsupportedEmbeddingMap("Reference vertices do not determine an affine chart.")


@cache
def _affine_exponents(
    dimension: int, /
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    return (0,) * dimension, tuple(
        tuple(int(index == column) for index in range(dimension))
        for column in range(dimension)
    )


def _affine_expressions(
    offset: tuple[ExactScalar, ...], matrix: tuple[tuple[ExactScalar, ...], ...]
) -> tuple[dict[tuple[int, ...], ExactScalar], ...]:
    zero, units = _affine_exponents(len(matrix[0]))
    result = []
    for axis, row in enumerate(matrix):
        expression: dict[tuple[int, ...], ExactScalar] = {}
        if offset[axis]:
            expression[zero] = offset[axis]
        for exponent, coefficient in zip(units, row, strict=True):
            if coefficient:
                expression[exponent] = coefficient
        result.append(expression)
    return tuple(result)


def _evaluate(expression: Expression, point: tuple[Fraction, ...]) -> ExactScalar:
    if isinstance(expression, RationalPolynomial):
        denominator = evaluate_polynomial(expression.denominator, point)
        if not denominator:
            raise UnsupportedEmbeddingMap(
                "Exact coordinate-map denominator vanishes at a reference corner."
            )
        return evaluate_polynomial(expression.numerator, point) / denominator
    return evaluate_polynomial(expression, point)


@dataclass(frozen=True, slots=True)
class EmbeddingCell:
    """Exact source-map image and canonical local vertex strata for one cell."""

    global_id: int
    block_name: str
    local_index: int
    kind: str
    dimension: int
    domain: str
    vertex_indices: tuple[int, ...]
    vertex_ids: tuple[int, ...]
    reference_vertices: tuple[tuple[Fraction, ...], ...]
    coordinate_expressions: tuple[Expression, ...] | None
    corner_images: tuple[tuple[ExactScalar, ...], ...]
    affine_offset: tuple[ExactScalar, ...] | None
    affine_matrix: tuple[tuple[ExactScalar, ...], ...] | None
    image_lower: tuple[float, ...]
    image_upper: tuple[float, ...]
    facet_routes: tuple[tuple[int, ...], ...] = ()

    pyramid_normalizations: tuple[PyramidApexNormalization, ...] | None = None
    pyramid_reference_arguments: tuple[Expression, ...] | None = None
    # Present only on charts prepared directly from these complete operands.
    # Derived/reindexed/shifted charts do not carry mesh-reuse authority.
    owning_mesh_id: str | None = None
    owning_geometry_id: str | None = None

    @property
    def coordinates(self) -> tuple[Expression, ...]:
        """Materialize affine polynomials only for a consumer that needs them."""
        if self.coordinate_expressions is not None:
            return self.coordinate_expressions
        if self.affine_offset is None or self.affine_matrix is None:
            raise UnsupportedEmbeddingMap(
                "An implicit affine chart needs its exact frame."
            )
        return _affine_expressions(self.affine_offset, self.affine_matrix)

    @property
    def proof_domain(self) -> BernsteinDomain:
        domain = "box" if self.pyramid_normalizations is not None else self.domain
        return cast(BernsteinDomain, domain)

    @property
    def proof_coordinates(self) -> tuple[Expression, ...]:
        if self.pyramid_normalizations is None:
            return self.coordinates
        return tuple(
            normalization.expression for normalization in self.pyramid_normalizations
        )

    @property
    def affine(self) -> bool:
        return self.affine_offset is not None and self.affine_matrix is not None


def _polyhedral_face_routes(
    mesh: CellMesh, cell_row: int, vertex_row: tuple[int, ...], /
) -> tuple[tuple[int, ...], ...]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise UnsupportedEmbeddingMap(
            "A variable-topology cell has no exact face incidence."
        )
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    local_index = {vertex: index for index, vertex in enumerate(vertex_row)}
    routes = []
    start, stop = cell_offsets[cell_row], cell_offsets[cell_row + 1]
    # Face loops are stored once; the cell's face sign orients each loop
    # outward so every cell lists shared faces with opposite orientation.
    for face, sign in zip(cell_faces[start:stop], cell_signs[start:stop], strict=True):
        vertices = face_vertices[face_offsets[face] : face_offsets[face + 1]]
        try:
            route = tuple(local_index[int(vertex)] for vertex in vertices)
        except KeyError as error:
            raise UnsupportedEmbeddingMap(
                "A cell face references a vertex outside its cell."
            ) from error
        routes.append(route if sign > 0 else route[::-1])
    if not routes:
        raise UnsupportedEmbeddingMap("A polyhedron has no declared boundary faces.")
    return tuple(routes)


def _cell_expressions(
    element: ScalarCoordinateElement,
    local: tuple[tuple[ExactScalar, ...], ...],
    row: int,
    basis: tuple[Expression, ...] | None = None,
) -> tuple[Expression, ...]:
    if basis is None:
        basis = coordinate_effective_basis(element, cell_row=row)
    if not basis:
        raise UnsupportedEmbeddingMap(
            "Coordinate element has no exact effective source basis."
        )
    coefficients = tuple(
        tuple(local[dof][axis] for dof in range(len(local)))
        for axis in range(len(local[0]))
    )
    try:
        return linear_combinations(basis, coefficients)
    except (ValueError, ZeroDivisionError) as error:
        raise UnsupportedEmbeddingMap(
            "Exact coordinate basis composition is unavailable."
        ) from error


_CellChart = tuple[
    tuple[Expression, ...] | None,
    tuple[tuple[ExactScalar, ...], ...],
    tuple[ExactScalar, ...] | None,
    tuple[tuple[ExactScalar, ...], ...] | None,
    tuple[tuple[float, float], ...],
    tuple[PyramidApexNormalization, ...] | None,
    tuple[Expression, ...] | None,
]


def _straight_cell_chart(
    local: tuple[tuple[Fraction, ...], ...],
    frame: tuple[tuple[int, ...], int, tuple[int, ...]],
    /,
) -> _CellChart:
    """Chart of a straight simplex read from its exact corner coordinate rows.

    The corner images are the routed rows themselves, the affine chart is the
    origin row plus the unit-corner differences, and its degree-one Bernstein
    controls are the corners, so no per-cell polynomial is composed.
    """
    corner_dofs, origin, units = frame
    corners = tuple(local[dof] for dof in corner_dofs)
    offset = corners[origin]
    matrix = tuple(
        tuple(corners[unit][axis] - value for unit in units)
        for axis, value in enumerate(offset)
    )
    bounds = tuple(
        (
            coefficient_endpoint(min(values), upper=False, saturate=True),
            coefficient_endpoint(max(values), upper=True, saturate=True),
        )
        for values in zip(*corners, strict=True)
    )
    return (
        None,
        corners,
        offset,
        matrix,
        bounds,
        None,
        None,
    )


def _mapped_cell_chart(
    element: ScalarCoordinateElement,
    local: tuple[tuple[ExactScalar, ...], ...],
    row: int,
    kind: str,
    dimension: int,
    domain: str,
    references: tuple[tuple[Fraction, ...], ...],
    ambient: int,
    basis: tuple[Expression, ...] | None = None,
    /,
) -> _CellChart:
    """Chart of a cell from its exact composed per-cell coordinate expressions."""
    coordinates = _cell_expressions(element, local, row, basis)
    if len(coordinates) != ambient:
        raise UnsupportedEmbeddingMap(
            "Exact source map ambient dimension does not match the mesh."
        )
    pyramid_normalizations: tuple[PyramidApexNormalization, ...] | None = None
    pyramid_reference_arguments: tuple[Expression, ...] | None = None
    proof_references = references
    proof_coordinates = coordinates
    proof_domain = domain
    if kind == "pyramid":
        pyramid_normalizations = tuple(
            normalize_pyramid_reference_expression(expression)
            for expression in coordinates
        )
        if any(
            normalization.apex_trace is None or normalization.apex_value is None
            for normalization in pyramid_normalizations
        ):
            raise UnsupportedEmbeddingMap(
                "Pyramid source map has no exact constant apex continuation."
            )
        pyramid_reference_arguments = pyramid_reference_to_box_arguments()
        proof_references = tuple(_pyramid_box_reference(point) for point in references)
        proof_coordinates = tuple(
            normalization.expression for normalization in pyramid_normalizations
        )
        proof_domain = "box"
    linear_chart = (
        _linear_simplex_chart(coordinates, references, dimension)
        if proof_domain == "simplex"
        else None
    )
    offset: tuple[ExactScalar, ...] | None
    matrix: tuple[tuple[ExactScalar, ...], ...] | None
    if linear_chart is None:
        corners = tuple(
            tuple(_evaluate(expression, point) for expression in proof_coordinates)
            for point in proof_references
        )
        offset, matrix = _solve_affine(references, corners)
        bounds = tuple(
            polynomial_bounds(expression, cast(BernsteinDomain, proof_domain), dimension)
            for expression in proof_coordinates
        )
    else:
        corners, offset, matrix, bounds = linear_chart
    affine_expressions = _affine_expressions(offset, matrix)
    if not (
        len(affine_expressions) == len(coordinates)
        and all(
            expression_equal(source, target)
            for source, target in zip(coordinates, affine_expressions, strict=True)
        )
    ):
        offset = None
        matrix = None
    else:
        # Exact source-expression equality, not corner interpolation, licenses
        # this polynomial representation. Retaining a removable pyramid
        # denominator would obstruct strict boundary-stratum sign proofs.
        coordinates = affine_expressions
    return (
        coordinates,
        corners,
        offset,
        matrix,
        bounds,
        pyramid_normalizations,
        pyramid_reference_arguments,
    )


def prepare_embedding_cells(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[EmbeddingCell, ...]:
    """Bind exact source-coordinate charts to mesh-order cell identities."""
    return tuple(_iter_embedding_cells(mesh, geometry))


def _iter_embedding_cells(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> Iterator[EmbeddingCell]:
    """Bind exact source-coordinate charts to mesh-order cell identities.

    Each block's connectivity, routes and identities are read once as plain rows
    and streamed cell by cell.  Straight simplices of a nodal degree-one element
    take their chart directly from the exact corner rows; every other fixed-
    reference cell composes its exact per-cell expressions.
    """
    from ..discretization._cell_geometry_validity import cell_geometry_id

    owning_mesh_id = mesh.mesh_id
    owning_geometry_id = cell_geometry_id(geometry)
    elements, routes, _ = geometry.resolve(mesh)
    bank = geometry.source_coordinates()
    vertex_global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64).tolist()
    ambient = mesh.ambient_dimension
    block_offset = 0
    for block, element, block_route in zip(mesh.blocks, elements, routes, strict=True):
        rows = np.asarray(block.vertices, dtype=np.int64)
        route_rows = np.asarray(block_route, dtype=np.int64)
        active = np.asarray(block.vertex_valid, dtype=np.bool_)
        ids = np.asarray(block.global_ids, dtype=np.int64)
        if (
            rows.ndim != 2
            or route_rows.ndim != 2
            or rows.shape[0] != route_rows.shape[0]
            or active.shape != rows.shape
            or ids.shape != (rows.shape[0],)
        ):
            raise UnsupportedEmbeddingMap(
                "Cell connectivity and coordinate routes are not row-aligned."
            )
        kind = str(block.cell_kind)
        name = str(block.name)
        dimension = block.topological_dimension
        vertex_element = isinstance(element, CellVertexGeometryElement)
        frame = None
        if not vertex_element:
            if not isinstance(element, ScalarCoordinateElement):
                raise UnsupportedEmbeddingMap(
                    "Fixed-reference embedding requires a scalar coordinate element."
                )
            facet_routes = _reference_facets(kind, dimension)
            domain = _map_domain(kind, dimension)
            references = _reference_rows(kind)
            if domain == "simplex":
                frame = _straight_simplex_frame(
                    element, references, dimension, route_rows.shape[1]
                )
        chart_bases: dict[bytes, tuple[Expression, ...]] = {}
        for local_index in range(rows.shape[0]):
            row = rows[local_index : local_index + 1].reshape(-1)
            route = route_rows[local_index : local_index + 1].reshape(-1)
            valid = active[local_index : local_index + 1].reshape(-1)
            cell_id = int(ids[local_index])
            active_row = tuple(
                int(vertex) for vertex, keep in zip(row, valid, strict=True) if keep
            )
            if vertex_element:
                if len(route) != len(valid):
                    raise UnsupportedEmbeddingMap(
                        "Vertex-based coordinate routes must align with active cell vertices."
                    )
                active_route = tuple(
                    int(index) for index, keep in zip(route, valid, strict=True) if keep
                )
                if len(active_row) != len(active_route):
                    raise UnsupportedEmbeddingMap(
                        "Active cell vertices and coordinate routes disagree."
                    )
                if kind == "polygon":
                    facet_routes = tuple(
                        (index, (index + 1) % len(active_row))
                        for index in range(len(active_row))
                    )
                else:
                    facet_routes = _polyhedral_face_routes(
                        mesh, block_offset + local_index, active_row
                    )
                corners = tuple(bank[index] for index in active_route)
                if len(corners) < (3 if kind == "polygon" else 4):
                    raise UnsupportedEmbeddingMap(
                        "Variable-topology cell has too few active vertices."
                    )
                chart: _CellChart = (
                    (),
                    corners,
                    None,
                    None,
                    tuple(
                        (
                            coefficient_endpoint(
                                min(point[axis] for point in corners), upper=False
                            ),
                            coefficient_endpoint(
                                max(point[axis] for point in corners), upper=True
                            ),
                        )
                        for axis in range(ambient)
                    ),
                    None,
                    None,
                )
                cell_references: tuple[tuple[Fraction, ...], ...] = ()
                cell_domain = "polyhedron" if kind == "polyhedron" else "polygon"
            else:
                if not all(valid):
                    raise UnsupportedEmbeddingMap(
                        "Padded fixed-reference cells are not supported by this map."
                    )
                if len(references) != len(active_row):
                    raise UnsupportedEmbeddingMap(
                        "Reference corners do not match cell vertex incidence."
                    )
                local = tuple(bank[index] for index in route)
                if not local or any(len(value) != len(local[0]) for value in local):
                    raise UnsupportedEmbeddingMap(
                        "Coordinate source rows have inconsistent ambient dimension."
                    )
                if (
                    frame is not None
                    and len(local[0]) == ambient
                    and all(
                        isinstance(value, Fraction) for point in local for value in point
                    )
                ):
                    chart = _straight_cell_chart(
                        cast(tuple[tuple[Fraction, ...], ...], local), frame
                    )
                else:
                    basis = None
                    if isinstance(element, PerCellRestrictedCellGeometryElement):
                        from ..discretization._cell_geometry_validity import _map_digest

                        chart_key = _map_digest(
                            kind, element.chart_arguments(local_index)
                        )
                        if chart_key is not None:
                            basis = chart_bases.get(chart_key)
                            if basis is None:
                                basis = coordinate_effective_basis(
                                    element, cell_row=local_index
                                )
                                chart_bases[chart_key] = basis
                    chart = _mapped_cell_chart(
                        element,
                        local,
                        local_index,
                        kind,
                        dimension,
                        domain,
                        references,
                        ambient,
                        basis,
                    )
                cell_references = references
                cell_domain = domain
            coordinates, corners, offset, matrix, bounds, normalizations, arguments = (
                chart
            )
            lower = tuple(value[0] for value in bounds)
            upper = tuple(value[1] for value in bounds)
            if not all(math.isfinite(value) for value in (*lower, *upper)):
                raise UnsupportedEmbeddingMap(
                    "Exact source image has no finite binary64 broad-phase bound."
                )
            yield EmbeddingCell(
                int(cell_id),
                name,
                local_index,
                kind,
                dimension,
                cell_domain,
                active_row,
                tuple(vertex_global_ids[index] for index in active_row),
                cell_references,
                coordinates,
                corners,
                offset,
                matrix,
                lower,
                upper,
                facet_routes,
                normalizations,
                arguments,
                owning_mesh_id,
                owning_geometry_id,
            )
        block_offset += block.cell_count


__all__ = ["EmbeddingCell", "UnsupportedEmbeddingMap", "prepare_embedding_cells"]
