#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-coordinate maps attached to individual mesh cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import cast

import numpy as np

from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
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


def _reference_rows(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    topology = reference_cell_topology(kind)
    return tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(topology.vertices, dtype=np.float64)
    )


def _pyramid_box_reference(
    point: tuple[Fraction, ...], /
) -> tuple[Fraction, Fraction, Fraction]:
    if len(point) != 3:
        raise UnsupportedEmbeddingMap("Pyramid corners require three reference coordinates.")
    if point[2] == 1:
        if point[:2] != (Fraction(1, 2), Fraction(1, 2)):
            raise UnsupportedEmbeddingMap("A pyramid apex has noncanonical coordinates.")
        return Fraction(0), Fraction(0), Fraction(1)
    if point[2] == 0:
        return point[0], point[1], Fraction(0)
    raise UnsupportedEmbeddingMap("Pyramid corners are outside the canonical reference strata.")


def _map_domain(kind: str, dimension: int) -> str:
    if kind in ("interval", "triangle", "tetrahedron") or kind.startswith("simplex:"):
        return "simplex"
    if kind == "prism":
        return "prism"
    if kind == "pyramid":
        if dimension != 3:
            raise UnsupportedEmbeddingMap("A pyramid reference map must be three-dimensional.")
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


def _affine_expressions(
    offset: tuple[ExactScalar, ...], matrix: tuple[tuple[ExactScalar, ...], ...]
) -> tuple[dict[tuple[int, ...], ExactScalar], ...]:
    dimension = len(matrix[0])
    zero = (0,) * dimension
    result = []
    for axis, row in enumerate(matrix):
        expression: dict[tuple[int, ...], ExactScalar] = {}
        if offset[axis]:
            expression[zero] = offset[axis]
        for column, coefficient in enumerate(row):
            if coefficient:
                exponent = tuple(int(index == column) for index in range(dimension))
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
    coordinates: tuple[Expression, ...]
    corner_images: tuple[tuple[ExactScalar, ...], ...]
    affine_offset: tuple[ExactScalar, ...] | None
    affine_matrix: tuple[tuple[ExactScalar, ...], ...] | None
    image_lower: tuple[float, ...]
    image_upper: tuple[float, ...]
    facet_routes: tuple[tuple[int, ...], ...] = ()

    pyramid_normalizations: tuple[PyramidApexNormalization, ...] | None = None
    pyramid_reference_arguments: tuple[Expression, ...] | None = None

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
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    local_index = {vertex: index for index, vertex in enumerate(vertex_row)}
    routes = []
    for face in cell_faces[cell_offsets[cell_row] : cell_offsets[cell_row + 1]]:
        vertices = face_vertices[face_offsets[face] : face_offsets[face + 1]]
        try:
            route = tuple(local_index[int(vertex)] for vertex in vertices)
        except KeyError as error:
            raise UnsupportedEmbeddingMap(
                "A cell face references a vertex outside its cell."
            ) from error
        routes.append(route)
    if not routes:
        raise UnsupportedEmbeddingMap("A polyhedron has no declared boundary faces.")
    return tuple(routes)


def _cell_expressions(
    element: ScalarCoordinateElement,
    route: np.ndarray,
    source_bank: tuple[tuple[ExactScalar, ...], ...],
    row: int,
) -> tuple[Expression, ...]:
    basis = coordinate_effective_basis(element, cell_row=row)
    if not basis:
        raise UnsupportedEmbeddingMap(
            "Coordinate element has no exact effective source basis."
        )
    local = tuple(source_bank[int(index)] for index in route)
    if not local or any(len(value) != len(local[0]) for value in local):
        raise UnsupportedEmbeddingMap(
            "Coordinate source rows have inconsistent ambient dimension."
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


def prepare_embedding_cells(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[EmbeddingCell, ...]:
    """Bind exact effective coordinate expressions to mesh-order cell identities."""
    elements, routes, _ = geometry.resolve(mesh)
    bank = geometry.source_coordinates()
    vertex_global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    result: list[EmbeddingCell] = []
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
        for local_index in range(rows.shape[0]):
            row = rows[local_index, :]
            route = route_rows[local_index, :]
            valid = active[local_index, :]
            cell_id = ids[local_index]
            active_row = tuple(int(value) for value in row[valid])
            kind = str(block.cell_kind)
            dimension = block.topological_dimension
            pyramid_normalizations: tuple[
                PyramidApexNormalization, ...
            ] | None = None
            pyramid_reference_arguments: tuple[Expression, ...] | None = None
            if isinstance(element, CellVertexGeometryElement):
                if route.shape != valid.shape:
                    raise UnsupportedEmbeddingMap(
                        "Vertex-based coordinate routes must align with active cell vertices."
                    )
                active_route = tuple(int(value) for value in route[valid])
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
                references: tuple[tuple[Fraction, ...], ...] = ()
                coordinates: tuple[Expression, ...] = ()
                offset = None
                matrix = None
                bounds: tuple[tuple[float, float], ...] = tuple(
                    (
                        coefficient_endpoint(
                            min(point[axis] for point in corners), upper=False
                        ),
                        coefficient_endpoint(
                            max(point[axis] for point in corners), upper=True
                        ),
                    )
                    for axis in range(mesh.ambient_dimension)
                )
                cell_domain = "polyhedron" if kind == "polyhedron" else "polygon"
            else:
                if not isinstance(element, ScalarCoordinateElement):
                    raise UnsupportedEmbeddingMap(
                        "Fixed-reference embedding requires a scalar coordinate element."
                    )
                if not np.all(valid):
                    raise UnsupportedEmbeddingMap(
                        "Padded fixed-reference cells are not supported by this map."
                    )
                facet_routes = tuple(
                    tuple(int(index) for index in face)
                    for face in reference_cell_topology(kind).entities[dimension - 1]
                )
                domain = _map_domain(kind, dimension)
                cell_domain = domain
                references = _reference_rows(kind)
                if len(references) != len(active_row):
                    raise UnsupportedEmbeddingMap(
                        "Reference corners do not match cell vertex incidence."
                    )
                coordinates = _cell_expressions(element, route, bank, local_index)
                if len(coordinates) != mesh.ambient_dimension:
                    raise UnsupportedEmbeddingMap(
                        "Exact source map ambient dimension does not match the mesh."
                    )
                proof_references = references
                proof_coordinates = coordinates
                proof_domain = domain
                if kind == "pyramid":
                    pyramid_normalizations = tuple(
                        normalize_pyramid_reference_expression(expression)
                        for expression in coordinates
                    )
                    if any(
                        normalization.apex_trace is None
                        or normalization.apex_value is None
                        for normalization in pyramid_normalizations
                    ):
                        raise UnsupportedEmbeddingMap(
                            "Pyramid source map has no exact constant apex continuation."
                        )
                    pyramid_reference_arguments = (
                        pyramid_reference_to_box_arguments()
                    )
                    proof_references = tuple(
                        _pyramid_box_reference(point) for point in references
                    )
                    proof_coordinates = tuple(
                        normalization.expression
                        for normalization in pyramid_normalizations
                    )
                    proof_domain = "box"
                corners = tuple(
                    tuple(_evaluate(expression, point) for expression in proof_coordinates)
                    for point in proof_references
                )
                offset, matrix = _solve_affine(references, corners)
                affine_expressions = _affine_expressions(offset, matrix)
                if not (
                    len(affine_expressions) == len(coordinates)
                    and all(
                        expression_equal(source, target)
                        for source, target in zip(
                            coordinates, affine_expressions, strict=True
                        )
                    )
                ):
                    offset = None
                    matrix = None
                bounds = tuple(
                    polynomial_bounds(
                        expression,
                        cast(BernsteinDomain, proof_domain),
                        dimension,
                    )
                    for expression in proof_coordinates
                )
            lower = tuple(value[0] for value in bounds)
            upper = tuple(value[1] for value in bounds)
            if not all(np.isfinite((*lower, *upper))):
                raise UnsupportedEmbeddingMap(
                    "Exact source image has no finite binary64 broad-phase bound."
                )
            result.append(
                EmbeddingCell(
                    int(cell_id),
                    str(block.name),
                    local_index,
                    kind,
                    dimension,
                    cell_domain,
                    active_row,
                    tuple(int(vertex_global_ids[index]) for index in active_row),
                    references,
                    coordinates,
                    corners,
                    offset,
                    matrix,
                    lower,
                    upper,
                    facet_routes,
                    pyramid_normalizations,
                    pyramid_reference_arguments,
                )
            )
        block_offset += block.cell_count
    return tuple(result)


__all__ = ["EmbeddingCell", "UnsupportedEmbeddingMap", "prepare_embedding_cells"]
