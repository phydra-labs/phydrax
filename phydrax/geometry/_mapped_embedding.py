#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous source-map embedding decisions with bounded interval subdivision.

A positive determinant is not an injectivity theorem. Local injectivity here is
proved by strict monotonicity of one fixed linear projection of the physical
Jacobian on the convex reference cell. Volume meshes are decided by the
boundary-degree theorem when its premises are proved; otherwise pair separation
uses whole image bounds, adjacent codimension-one patches use certified
tangent-cone contacts, and remaining shared topology is removed only by a
checked strict separating-plane proof.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import cast, NamedTuple, TYPE_CHECKING

import numpy as np

from ..discretization._coordinate_enclosure import (
    _box_corner_lattice,
    _COORDINATE_BUDGET,
    _expression_control_net,
    _reserve_polynomial,
    affine_arguments,
    constant,
    coordinate_expressions,
    Expression,
    expression_add as add,
    expression_bernstein_coefficients as bernstein_coefficients,
    expression_bounds as polynomial_bounds,
    expression_compose as compose,
    expression_derivative as derivative,
    expression_evaluate as evaluate,
    expression_linear_combinations,
    expression_node_count as bernstein_node_count,
    expression_physical_jacobian as physical_jacobian,
    expression_reference_evaluate,
    expression_scale as scale,
    expression_sum as sum_polynomials,
    ExpressionComposition as ArgumentComposition,
    multi_affine_coordinates,
    outward,
    Polynomial,
    prepared_coordinate_source_bank,
    RationalPolynomial,
    restrict_chart_expressions,
)
from ..discretization._reference_cell import reference_cell_topology


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometryElement, CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ..discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateLiveStorage,
    )
    from ._mesh_certificates import _EmbeddingState, MeshCertificateLimits


@dataclass(frozen=True)
class _Cell:
    coordinates: tuple[Expression, ...]
    domain: str
    kind: str
    dimension: int
    vertices: tuple[int, ...]
    reference_vertices: tuple[tuple[Fraction, ...], ...]
    # Exact images of ``reference_vertices``, present only when the source map
    # has degree at most one in every axis of its box (multi-affine).
    corner_images: tuple[tuple[Fraction, ...], ...] | None = None


def _corner_lattice(cell: _Cell) -> list[tuple[Fraction, ...]]:
    """Multi-affine corner images indexed by their reference-axis bitmask."""
    if cell.corner_images is None:
        raise ValueError("A corner lattice requires a multi-affine cell.")
    masks, _ = _box_corner_lattice(cell.kind)
    corners: list[tuple[Fraction, ...]] = [()] * len(masks)
    for mask, image in zip(masks, cell.corner_images, strict=True):
        corners[mask] = image
    return corners


def _domain(kind: str) -> str:
    return (
        "simplex"
        if kind in ("triangle", "tetrahedron")
        else "prism"
        if kind == "prism"
        else "box"
    )


@cache
def _vertices(kind: str) -> tuple[tuple[Fraction, ...], ...]:
    topology = reference_cell_topology(kind)
    if kind == "pyramid":
        return tuple(
            tuple(Fraction(value) for value in point)
            for point in (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            )
        )
    return tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(topology.vertices)
    )


def _identity_reference_map(origin: np.ndarray, matrix: np.ndarray) -> bool:
    return (
        not np.any(origin)
        and matrix.shape == (origin.size, origin.size)
        and all(
            value == int(i == j)
            for i, row in enumerate(matrix)
            for j, value in enumerate(row)
        )
    )


def _bounds(
    cell: _Cell, origin: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    identity = _identity_reference_map(origin, matrix)
    if identity and cell.corner_images is not None:
        # Degree-one Bernstein controls of a multi-affine map are its corner images.
        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            budget.reserve(len(cell.corner_images) * len(cell.coordinates))
        columns = tuple(zip(*cell.corner_images, strict=True))
        return (
            np.asarray(
                [outward(min(column), -math.inf) for column in columns], dtype=np.float64
            ),
            np.asarray(
                [outward(max(column), math.inf) for column in columns], dtype=np.float64
            ),
        )
    composition = (
        None if identity else ArgumentComposition(affine_arguments(origin, matrix))
    )
    intervals = np.asarray(
        [
            polynomial_bounds(
                value if composition is None else composition(value),
                cell.domain,
                cell.dimension,
            )
            for value in cell.coordinates
        ],
        dtype=np.float64,
    )
    return intervals[:, 0], intervals[:, 1]


def _rounded_matrix(values: tuple[tuple[Fraction, ...], ...]) -> np.ndarray | None:
    maximum = Fraction(float(np.finfo(np.float64).max))
    if any(abs(value) > maximum for row in values for value in row):
        return None
    return np.asarray(
        [[float(value) for value in row] for row in values], dtype=np.float64
    )


def _midpoint_projection(
    midpoint: np.ndarray | None, dimension: int
) -> np.ndarray | None:
    """Fixed binary64 pseudo-inverse of a full-rank midpoint Jacobian."""
    if midpoint is None or np.linalg.matrix_rank(midpoint) != dimension:
        return None
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    return projection if rank == dimension else None


def _jacobian_midpoint(
    cell: _Cell,
    jacobian: tuple[tuple[Expression, ...], ...] | None = None,
) -> np.ndarray | None:
    if jacobian is None:
        jacobian = physical_jacobian(cell.coordinates, cell.kind, cell.dimension)
    if jacobian is None:
        return None
    point = (
        tuple(Fraction(1, cell.dimension + 1) for _ in range(cell.dimension))
        if cell.domain == "simplex"
        else (Fraction(1, 3), Fraction(1, 3), Fraction(1, 2))
        if cell.domain == "prism"
        else (Fraction(1, 2),) * cell.dimension
    )
    values = tuple(tuple(evaluate(entry, point) for entry in row) for row in jacobian)
    return _rounded_matrix(values)


def _projected_jacobian(
    cell: _Cell,
    projection: np.ndarray,
    jacobian: tuple[tuple[Expression, ...], ...] | None = None,
) -> tuple[tuple[Expression, ...], ...] | None:
    if jacobian is None:
        jacobian = physical_jacobian(cell.coordinates, cell.kind, cell.dimension)
    if jacobian is None:
        return None
    weights = tuple(tuple(Fraction(float(value)) for value in row) for row in projection)
    columns = tuple(
        expression_linear_combinations(tuple(row[j] for row in jacobian), weights)
        for j in range(cell.dimension)
    )
    return tuple(tuple(column[i] for column in columns) for i in range(cell.dimension))


def _correlated_bernstein_spd(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: str,
    dimension: int,
    composition: ArgumentComposition | None,
    symmetric_parts: dict[tuple[int, int], Expression],
) -> bool:
    """Prove strict monotonicity by convexity of symmetric positive-definite nets."""
    if dimension not in (2, 3):
        return False
    if any(isinstance(entry, RationalPolynomial) for row in jacobian for entry in row):
        # Different rational denominators do not share Bernstein convex weights.
        return False
    entries: list[Expression] = [jacobian[i][i] for i in range(dimension)]
    for i in range(dimension):
        for j in range(i + 1, dimension):
            symmetric = symmetric_parts.get((i, j))
            if symmetric is None:
                symmetric = expression_linear_combinations(
                    (jacobian[i][j], jacobian[j][i]), ((Fraction(1, 2), Fraction(1, 2)),)
                )[0]
                symmetric_parts[i, j] = symmetric
            entries.append(symmetric)
    polynomials: list[Polynomial] = []
    for entry in entries:
        value = entry if composition is None else composition(entry)
        if isinstance(value, RationalPolynomial):
            return False
        polynomials.append(value)
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(dimension * sum(len(value) for value in polynomials))
    match domain:
        case "box":
            degrees = tuple(
                max((index[axis] for value in polynomials for index in value), default=0)
                for axis in range(dimension)
            )
        case "simplex":
            degrees = (
                max((sum(index) for value in polynomials for index in value), default=0),
            )
        case "prism":
            degrees = (
                max(
                    (sum(index[:2]) for value in polynomials for index in value),
                    default=0,
                ),
                max((index[2] for value in polynomials for index in value), default=0),
            )
        case _:
            raise ValueError(
                "A correlated Bernstein proof requires a canonical reference domain."
            )
    if not any(degrees):
        # A constant net has one exact control, independent of the domain.
        # Preserve the same principal-minor proof without preparing identity
        # monomial weights separately for every matrix entry.
        if budget is not None:
            budget.reserve(len(polynomials))
        controls = tuple(
            (value.get((0,) * dimension, Fraction(0)),) for value in polynomials
        )
    else:
        controls = tuple(
            _expression_control_net(value, domain, degrees, dimension)
            for value in polynomials
        )
    if budget is not None:
        budget.reserve(sum(len(net) for net in controls))
        bits = max(
            (
                abs(value.numerator).bit_length() + value.denominator.bit_length()
                for net in controls
                for value in net
            ),
            default=1,
        )
        _reserve_polynomial(0, 32, 0, 24 * bits + 4)
    for values in zip(*controls, strict=True):
        extra = None if dimension == 2 else (values[2], values[4], values[5])
        if not _spd_minors(values[0], values[1], values[dimension], extra):
            return False
    return True


def _spd_minors(
    a: Fraction,
    d: Fraction,
    b: Fraction,
    extra: tuple[Fraction, Fraction, Fraction] | None,
) -> bool:
    """Positive leading principal minors of one exact symmetric control matrix.

    ``a``, ``d`` are the first two diagonal entries and ``b`` their coupling;
    ``extra`` carries ``(f, c, e)``: the third diagonal entry and its couplings
    to the first and second rows.
    """
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(1)
    if a <= 0:
        return False
    if budget is not None:
        budget.reserve(4)
    minor = a * d - b * b
    if minor <= 0:
        return False
    if extra is not None:
        f, c, e = extra
        if budget is not None:
            budget.reserve(12)
        if f * minor - a * e * e - d * c * c + 2 * b * c * e <= 0:
            return False
    return True


def _multi_affine_monotone(cell: _Cell) -> bool:
    """Prove the complete multi-affine Jacobian net strictly monotone.

    Physical edge columns are prepared once and used both for the midpoint
    projection and for every exact corner control. Sparse projection omits
    only proved zero coefficients. Every symmetric control matrix retains its
    full exact principal-minor checks without speculative cache-key visits.
    """
    dimension = cell.dimension
    if cell.corner_images is None or dimension not in (2, 3):
        return False
    corners = _corner_lattice(cell)
    count, edges, ambient = len(corners), len(corners) // 2, len(corners[0])
    budget = _COORDINATE_BUDGET.get()
    image_numerator, image_common = 1, 1
    if budget is not None:
        budget.reserve(count * ambient)
        for image in corners:
            for value in image:
                image_numerator = max(image_numerator, abs(value.numerator).bit_length())
                image_common = math.lcm(image_common, value.denominator)
        _reserve_polynomial(
            ambient * dimension * (2 * edges + 1),
            edges * dimension * ambient,
            dimension,
            image_numerator + image_common.bit_length() + 2,
        )
    physical = tuple(
        {
            mask: {
                row: value
                for row in range(ambient)
                if (value := corners[mask | (1 << axis)][row] - corners[mask][row])
            }
            for mask in range(count)
            if not (mask >> axis) & 1
        }
        for axis in range(dimension)
    )
    zero = Fraction(0)
    midpoint = _rounded_matrix(
        tuple(
            tuple(
                sum((edge.get(row, zero) for edge in physical[axis].values()), zero)
                / edges
                for axis in range(dimension)
            )
            for row in range(ambient)
        )
    )
    projection = _midpoint_projection(midpoint, dimension)
    if projection is None:
        return False
    if budget is not None:
        budget.reserve(dimension * ambient)
    weights = tuple(tuple(Fraction(float(value)) for value in row) for row in projection)
    weight_numerator, weight_common = 1, 1
    sparse_weights = []
    for row in weights:
        sparse = []
        for axis, value in enumerate(row):
            if budget is not None:
                weight_numerator = max(
                    weight_numerator, abs(value.numerator).bit_length()
                )
                weight_common = math.lcm(weight_common, value.denominator)
            if value:
                sparse.append((axis, value))
        sparse_weights.append(tuple(sparse))
    pairs = tuple((i, j) for i in range(dimension) for j in range(i + 1, dimension))
    if budget is not None:
        products = sum(
            sum(axis in edge for axis, _ in row)
            for columns in physical
            for edge in columns.values()
            for row in sparse_weights
        )
        _reserve_polynomial(
            products,
            edges * dimension * dimension + count * len(pairs),
            dimension,
            image_numerator
            + image_common.bit_length()
            + weight_numerator
            + weight_common.bit_length()
            + ambient.bit_length()
            + 2,
        )
    differences = tuple(
        {
            mask: tuple(
                sum((weight * edge[axis] for axis, weight in row if axis in edge), zero)
                for row in sparse_weights
            )
            for mask, edge in columns.items()
        }
        for columns in physical
    )
    if budget is not None:
        budget.reserve(2 * count * len(pairs))
    controls = []
    for mask in range(count):
        column = tuple(
            differences[axis][mask & ~(1 << axis)] for axis in range(dimension)
        )
        controls.append(
            (
                tuple(column[i][i] for i in range(dimension)),
                tuple((column[j][i] + column[i][j]) / 2 for i, j in pairs),
            )
        )
    if budget is not None:
        budget.reserve(count * (dimension + len(pairs)))
        bits = max(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for diagonal, coupling in controls
            for value in (*diagonal, *coupling)
        )
        _reserve_polynomial(0, 32, 0, 24 * bits + 4)
    for diagonal, coupling in controls:
        extra = None if dimension == 2 else (diagonal[2], coupling[1], coupling[2])
        if not _spd_minors(diagonal[0], diagonal[1], coupling[0], extra):
            return False
    return True


def _strict_spd(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: str,
    dimension: int,
    origin: np.ndarray,
    matrix: np.ndarray,
    symmetric_parts: dict[tuple[int, int], Expression],
) -> bool:
    composition = (
        None
        if _identity_reference_map(origin, matrix)
        else ArgumentComposition(affine_arguments(origin, matrix))
    )
    if _correlated_bernstein_spd(
        jacobian, domain, dimension, composition, symmetric_parts
    ):
        return True
    # The symmetric part is shared by rows i and j; enclose it once per piece.
    magnitudes: dict[tuple[int, int], float] = {}
    for i in range(dimension):
        entry = jacobian[i][i]
        diagonal = polynomial_bounds(
            entry if composition is None else composition(entry), domain, dimension
        )[0]
        radius = 0.0
        for j in range(dimension):
            if i != j:
                pair = (min(i, j), max(i, j))
                magnitude = magnitudes.get(pair)
                if magnitude is None:
                    symmetric = symmetric_parts.get(pair)
                    if symmetric is None:
                        symmetric = expression_linear_combinations(
                            (jacobian[pair[0]][pair[1]], jacobian[pair[1]][pair[0]]),
                            ((Fraction(1, 2), Fraction(1, 2)),),
                        )[0]
                        symmetric_parts[pair] = symmetric
                    interval = polynomial_bounds(
                        symmetric if composition is None else composition(symmetric),
                        domain,
                        dimension,
                    )
                    magnitude = magnitudes[pair] = max(abs(value) for value in interval)
                radius = float(np.nextafter(radius + magnitude, math.inf))
        if not diagonal > radius:
            return False
    return True


def _children(
    cell: _Cell, origin: np.ndarray, matrix: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    from ..discretization._cell_geometry_validity import (
        _prism_child_maps,
        _simplex_child_maps,
    )

    if cell.domain == "simplex":
        origins, matrices = _simplex_child_maps(cell.dimension)
    elif cell.domain == "prism":
        origins, matrices = _prism_child_maps()
    else:
        # Binary splitting bounds candidate growth and keeps the work budget
        # independent of the ambient dimension.
        axis = int(np.argmax(np.linalg.norm(matrix, axis=0)))
        first = np.eye(cell.dimension, dtype=np.float64)
        first[axis, axis] = 0.5
        second = np.zeros((cell.dimension,), dtype=np.float64)
        second[axis] = 0.5
        origins = np.stack((np.zeros_like(second), second))
        matrices = np.stack((first, first))
    return tuple(
        (origin + matrix @ start, matrix @ child)
        for start, child in zip(origins, matrices, strict=True)
    )


def _injective(
    cell: _Cell, limits: MeshCertificateLimits, state: _EmbeddingState
) -> str | None:
    if cell.corner_images is not None:
        if state.subdivision_pieces >= limits.maximum_subdivision_pieces:
            return "local_injectivity_piece_budget"
        if _multi_affine_monotone(cell):
            state.subdivision_pieces += 1
            return None
    physical = physical_jacobian(cell.coordinates, cell.kind, cell.dimension)
    if physical is None:
        return "local_injectivity_premise"
    budget = _COORDINATE_BUDGET.get()
    affine_key: tuple[tuple[Fraction, ...], ...] | None = None
    if budget is not None:
        rows = []
        constant = True
        for row in physical:
            values = []
            for entry in row:
                if isinstance(entry, RationalPolynomial):
                    constant = False
                    break
                budget.reserve(len(entry))
                if any(any(index) for index in entry):
                    constant = False
                    break
                values.append(entry.get((0,) * cell.dimension, Fraction(0)))
            if not constant:
                break
            rows.append(tuple(values))
        if constant:
            affine_key = tuple(rows)
            budget.reserve(1)
            if affine_key in budget.affine_injectivity_cache:
                if state.subdivision_pieces >= limits.maximum_subdivision_pieces:
                    return "local_injectivity_piece_budget"
                state.subdivision_pieces += 1
                return None
    projection = _midpoint_projection(_jacobian_midpoint(cell, physical), cell.dimension)
    if projection is None:
        return "local_injectivity_premise"
    jacobian = _projected_jacobian(cell, projection, physical)
    if jacobian is None:
        return "rational_denominator_premise"
    active = [
        (
            np.zeros((cell.dimension,), dtype=np.float64),
            np.eye(cell.dimension, dtype=np.float64),
            0,
        )
    ]
    symmetric_parts: dict[tuple[int, int], Expression] = {}
    while active:
        origin, matrix, depth = active.pop()
        if state.subdivision_pieces >= limits.maximum_subdivision_pieces:
            return "local_injectivity_piece_budget"
        state.subdivision_pieces += 1
        if _strict_spd(
            jacobian, cell.domain, cell.dimension, origin, matrix, symmetric_parts
        ):
            continue
        if depth == limits.maximum_subdivision_depth:
            return "local_injectivity_subdivision_depth"
        active.extend(
            (start, child, depth + 1) for start, child in _children(cell, origin, matrix)
        )
    if budget is not None and affine_key is not None:
        # The original projected-Jacobian/SPD proof above earned this theorem.
        # Translation and reference-domain subdivision cannot change a
        # constant physical Jacobian, so identical matrices reuse that proof.
        budget.retain_basis((affine_key,))
        budget.affine_injectivity_cache.add(affine_key)
    return None


def _signed_image(cell: _Cell, normal: np.ndarray, offset: Fraction) -> Expression:
    return add(
        sum_polynomials(
            tuple(
                scale(value, Fraction(weight))
                for value, weight in zip(cell.coordinates, normal, strict=True)
            )
        ),
        constant(offset, cell.dimension),
    )


def _strict_side(
    cell: _Cell,
    polynomial: Expression,
    shared: set[int],
    *,
    require_shared_support: bool = True,
) -> bool:
    coefficients = bernstein_coefficients(polynomial, cell.domain, cell.dimension)
    if min(coefficients) < 0 or max(coefficients) <= 0:
        return False
    vertices = (
        cell.vertices
        if cell.kind != "pyramid"
        else (*cell.vertices[:4], *([cell.vertices[4]] * 4))
    )
    for vertex, point in zip(vertices, cell.reference_vertices, strict=True):
        value = expression_reference_evaluate(polynomial, point, cell.domain)
        if vertex in shared:
            if value:
                return False
        elif require_shared_support and value <= 0:
            return False
    return True


class _PlaneControls(NamedTuple):
    """Exact same-degree coordinate nets retained for one separating operation."""

    coordinates: tuple[tuple[int, ...], ...]
    denominator: int
    degree_support: tuple[tuple[tuple[int, ...], ...], ...]
    coefficient_bits: int


def _prepare_plane_controls(cell: _Cell) -> _PlaneControls | None:
    # Rational controls are ratios, not linear coordinate nets. Their owning
    # numerator/denominator reduction remains on the original expression route.
    if any(isinstance(value, RationalPolynomial) for value in cell.coordinates):
        return None
    polynomials = tuple(
        value for value in cell.coordinates if not isinstance(value, RationalPolynomial)
    )

    def degree(index: tuple[int, ...]) -> tuple[int, ...]:
        return (
            (sum(index),)
            if cell.domain == "simplex"
            else (sum(index[:2]), index[2])
            if cell.domain == "prism"
            else index
        )

    degrees = tuple(
        tuple(
            max((degree(index)[axis] for index in value), default=0)
            for axis in range(len(degree((0,) * cell.dimension)))
        )
        for value in polynomials
    )
    if not degrees or len(set(degrees)) != 1 or max(degrees[0]) <= 1:
        return None
    indices = tuple(dict.fromkeys(index for value in polynomials for index in value))
    from ..discretization._coordinate_enclosure import _reserve_polynomial

    source_denominator = math.lcm(
        *(
            coefficient.denominator
            for value in polynomials
            for coefficient in value.values()
        )
    )
    source_bits = (
        max(
            (
                abs(coefficient.numerator).bit_length()
                for value in polynomials
                for coefficient in value.values()
            ),
            default=1,
        )
        + source_denominator.bit_length()
    )
    support_terms = sum(
        sum(degree(index)[axis] == maximum for index in indices)
        for axis, maximum in enumerate(degrees[0])
    ) * len(polynomials)
    _reserve_polynomial(support_terms, support_terms, cell.dimension, source_bits)
    support = tuple(
        tuple(
            tuple(
                0
                if index not in value
                else value[index].numerator
                * (source_denominator // value[index].denominator)
                for value in polynomials
            )
            for index in indices
            if degree(index)[axis] == maximum
        )
        for axis, maximum in enumerate(degrees[0])
    )
    controls = tuple(
        bernstein_coefficients(value, cell.domain, cell.dimension)
        for value in polynomials
    )
    denominator = math.lcm(
        *(value.denominator for column in controls for value in column)
    )
    bits = (
        max(
            (
                abs(value.numerator).bit_length()
                for column in controls
                for value in column
            ),
            default=1,
        )
        + denominator.bit_length()
    )
    count = sum(len(column) for column in controls)
    _reserve_polynomial(count, count, cell.dimension, bits)
    integers = tuple(
        tuple(value.numerator * (denominator // value.denominator) for value in column)
        for column in controls
    )
    return _PlaneControls(integers, denominator, support, bits)


def _plane_preserves_degree(prepared: _PlaneControls, normal: np.ndarray) -> bool:
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(sum(len(group) for group in prepared.degree_support) * len(normal))
    # Exact cancellation can lower the original scalar Bernstein degree. Keep
    # that route unchanged rather than replacing it with an elevated net.
    denominator = math.lcm(*(Fraction(value).denominator for value in normal))
    integers = tuple(
        Fraction(value).numerator * (denominator // Fraction(value).denominator)
        for value in normal
    )
    return all(
        any(
            sum(weight * value for weight, value in zip(integers, row, strict=True))
            for row in group
        )
        for group in prepared.degree_support
    )


def _prepared_plane_side(
    cell: _Cell,
    prepared: _PlaneControls,
    normal: np.ndarray,
    offset: Fraction,
    vertex_values: np.ndarray,
    shared: set[int],
) -> tuple[bool, bool]:
    from ..discretization._coordinate_enclosure import _reserve_polynomial

    weights = tuple(Fraction(value) for value in normal)
    denominator = math.lcm(offset.denominator, *(value.denominator for value in weights))
    bits = (
        prepared.coefficient_bits
        + max(
            abs(offset.numerator).bit_length(),
            *(abs(value.numerator).bit_length() for value in weights),
        )
        + denominator.bit_length()
        + (len(weights) + 1).bit_length()
    )
    nodes = len(prepared.coordinates[0])
    _reserve_polynomial(nodes * (2 * len(weights) + 1), nodes, cell.dimension, bits)
    integers = tuple(
        value.numerator * (denominator // value.denominator) for value in weights
    )
    constant = (
        offset.numerator * (denominator // offset.denominator) * prepared.denominator
    )
    # Every control has the same strictly positive denominator. Integer signs
    # are precisely the signs of the original normalized Fraction controls.
    coefficients = tuple(
        constant
        + sum(weight * value for weight, value in zip(integers, row, strict=True))
        for row in zip(*prepared.coordinates, strict=True)
    )
    if min(coefficients) < 0 or max(coefficients) <= 0:
        return False, False
    vertices = (
        cell.vertices
        if cell.kind != "pyramid"
        else (*cell.vertices[:4], *([cell.vertices[4]] * 4))
    )
    minimal = True
    for vertex, value in zip(vertices, vertex_values, strict=True):
        if vertex in shared:
            if value:
                return False, False
        elif value <= 0:
            minimal = False
    return minimal, True


def _multi_affine_trace_equal(
    first: _Cell, second: _Cell, shared: list[int]
) -> bool | None:
    """Exact trace equality of multi-affine box cells on all common entities, or None.

    A multi-affine map restricted to a box edge or square face is the interpolant
    of its corner images. When every common entity of ``first`` is an entity of
    ``second`` whose square face pairs the same diagonal vertices, the full
    restrictions agree exactly when the shared corner images agree. Any other
    common entity is left to the generic restriction proof.
    """
    if (
        first.corner_images is None
        or second.corner_images is None
        or first.kind != second.kind
        or first.dimension != second.dimension
    ):
        return None
    budget = _COORDINATE_BUDGET.get()
    first_reference, second_reference = (
        first.reference_vertices,
        second.reference_vertices,
    )
    topology = reference_cell_topology(first.kind)
    if budget is not None:
        budget.reserve(
            len(shared) * (len(first.coordinates) + 2 * first.dimension)
            + sum(
                len(entity)
                for degree in range(1, first.dimension)
                for entity in topology.entities[degree]
            )
        )
    for degree in range(1, first.dimension):
        for entity in topology.entities[degree]:
            keys = sorted(first.vertices[i] for i in entity)
            if not set(keys).issubset(shared):
                continue
            partners = []
            for cell, reference in ((first, first_reference), (second, second_reference)):
                corners = [reference[cell.vertices.index(key)] for key in keys]
                free = [
                    axis
                    for axis in range(cell.dimension)
                    if len({corner[axis] for corner in corners}) > 1
                ]
                if len(free) != degree:
                    return None
                # A square face is one square symmetry apart in both cells exactly
                # when the first key has the same diagonal partner in both.
                partners.append(
                    tuple(
                        key
                        for key, corner in zip(keys, corners, strict=True)
                        if all(corner[axis] != corners[0][axis] for axis in free)
                    )
                )
            if degree == 2 and (partners[0] != partners[1] or len(partners[0]) != 1):
                return None
    for key in shared:
        if (
            first.corner_images[first.vertices.index(key)]
            != second.corner_images[second.vertices.index(key)]
        ):
            return False
    return True


def _multi_affine_axis_separation(first: _Cell, second: _Cell, shared: set[int]) -> bool:
    """Exact coordinate-axis separating plane of multi-affine box cells.

    The signed plane value of a multi-affine map is multi-affine, so its
    degree-one Bernstein controls are its corner values. This is the generic
    separating-plane acceptance for the coordinate normals, decided on exact
    corner images; ``False`` leaves every other normal to the generic proof.
    """
    if first.corner_images is None or second.corner_images is None:
        return False
    a, b = first.corner_images, second.corner_images
    ambient = len(a[0])
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(4 * ambient * (len(a) + len(b)))
    anchor = next(iter(shared), None)

    def side(values: list[Fraction], vertices: tuple[int, ...], strict: bool) -> bool:
        if not any(values):
            return False
        for value, vertex in zip(values, vertices, strict=True):
            if vertex in shared:
                if value:
                    return False
            elif strict and value <= 0:
                return False
        return True

    for axis in range(ambient):
        for sign in (1, -1):
            left = [sign * corner[axis] for corner in a]
            right = [sign * corner[axis] for corner in b]
            offset = (
                sign * a[first.vertices.index(anchor)][axis]
                if anchor is not None
                else (max(left) + min(right)) / 2
            )
            below = [offset - value for value in left]
            above = [value - offset for value in right]
            if any(value < 0 for value in below) or any(value < 0 for value in above):
                continue
            if (
                side(below, first.vertices, True) and side(above, second.vertices, False)
            ) or (
                side(below, first.vertices, False) and side(above, second.vertices, True)
            ):
                return True
    return False


def _trace_equal(first: _Cell, second: _Cell) -> bool:
    shared = sorted(set(first.vertices).intersection(second.vertices))
    if not shared:
        return True
    decided = _multi_affine_trace_equal(first, second, shared)
    if decided is not None:
        return decided
    first_reference = reference_cell_topology(first.kind)
    second_reference = reference_cell_topology(second.kind)
    # Every common canonical subentity, including vertex-only contacts, must
    # have the same full source expression in the same oriented parameter chart.
    for degree in range(first.dimension):
        entities = first_reference.entities[degree]
        for entity in entities:
            keys = [first.vertices[i] for i in entity]
            if not set(keys).issubset(shared):
                continue
            ordered = sorted(keys)
            a = np.asarray(first_reference.vertices, dtype=np.float64)[
                [first.vertices.index(key) for key in ordered]
            ]
            b = np.asarray(second_reference.vertices, dtype=np.float64)[
                [second.vertices.index(key) for key in ordered]
            ]
            if degree == 0:
                left = tuple(
                    expression_reference_evaluate(
                        value, tuple(Fraction(float(x)) for x in a[0]), first.domain
                    )
                    for value in first.coordinates
                )
                right = tuple(
                    expression_reference_evaluate(
                        value, tuple(Fraction(float(x)) for x in b[0]), second.domain
                    )
                    for value in second.coordinates
                )
            else:
                if len(ordered) not in (2, 3, 4):
                    continue
                kind = (
                    "interval"
                    if degree == 1
                    else "triangle"
                    if len(ordered) == 3
                    else "quadrilateral"
                )
                # Select an independent corner basis; canonical order is shared
                # by both restrictions, never selected by physical proximity.
                columns = (
                    (1,)
                    if degree == 1
                    else next(
                        (
                            pair
                            for pair in ((1, 2), (1, 3), (2, 3))
                            if max(pair) < len(ordered)
                            and np.linalg.matrix_rank((a[list(pair)] - a[0]).T) == degree
                            and np.linalg.matrix_rank((b[list(pair)] - b[0]).T) == degree
                        ),
                        (),
                    )
                )
                if not columns:
                    return False
                left = restrict_chart_expressions(
                    first.coordinates, first.kind, kind, a[0], (a[list(columns)] - a[0]).T
                )
                right = restrict_chart_expressions(
                    second.coordinates,
                    second.kind,
                    kind,
                    b[0],
                    (b[list(columns)] - b[0]).T,
                )
            if left != right:
                return False
    return True


def _separating_plane(
    first: _Cell,
    second: _Cell,
    prepared_controls: tuple[_PlaneControls | None, _PlaneControls | None] | None = None,
) -> bool:
    from contextlib import nullcontext

    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    budget = _COORDINATE_BUDGET.get()
    shared = set(first.vertices).intersection(second.vertices)
    a = np.asarray(
        [
            [
                expression_reference_evaluate(value, point, first.domain)
                for value in first.coordinates
            ]
            for point in first.reference_vertices
        ],
        dtype=object,
    )
    b = np.asarray(
        [
            [
                expression_reference_evaluate(value, point, second.domain)
                for value in second.coordinates
            ]
            for point in second.reference_vertices
        ],
        dtype=object,
    )
    ambient = a.shape[1]
    normals = [
        np.sum(b, axis=0) / len(b) - np.sum(a, axis=0) / len(a),
        *np.asarray(
            [[Fraction(i == j) for j in range(ambient)] for i in range(ambient)],
            dtype=object,
        ),
    ]

    def cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ),
            dtype=object,
        )

    if ambient == 2:
        normals.extend(
            np.asarray((-(q - p)[1], (q - p)[0]))
            for corners in (a, b)
            for p, q in zip(corners, np.roll(corners, -1, axis=0), strict=True)
        )
    elif ambient == 3:
        from itertools import combinations

        for first_vertex, second_vertex in combinations(sorted(shared), 2):
            tangent = (
                a[first.vertices.index(second_vertex)]
                - a[first.vertices.index(first_vertex)]
            )
            normals.extend(
                cross(tangent, axis)
                for axis in np.asarray(
                    [[Fraction(i == j) for j in range(3)] for i in range(3)], dtype=object
                )
            )
        topology = reference_cell_topology(first.kind)
        normals.extend(
            cross(a[face[1]] - a[face[0]], a[face[2]] - a[face[0]])
            for face in topology.entities[2]
            if len(face) >= 3
        )
        topology = reference_cell_topology(second.kind)
        normals.extend(
            cross(b[face[1]] - b[face[0]], b[face[2]] - b[face[0]])
            for face in topology.entities[2]
            if len(face) >= 3
        )
        if first.kind == second.kind == "triangle":
            normal_a = cross(a[1] - a[0], a[2] - a[0])
            normal_b = cross(b[1] - b[0], b[2] - b[0])
            tangent_normal = (
                normal_a + normal_b if normal_a @ normal_b >= 0 else normal_a - normal_b
            )
            edge_normals = [
                cross(tangent_normal, q - p)
                for corners in (a, b)
                for p, q in zip(corners, np.roll(corners, -1, axis=0), strict=True)
            ]
            normals.extend(edge_normals)
            normals.extend(x + y for x in edge_normals for y in edge_normals)
            normals.extend(x - y for x in edge_normals for y in edge_normals)
    # Contacts on an edge/vertex may require a separator in the cone spanned
    # by two supporting face normals, rather than either face normal alone.
    # Every candidate still proves a full source-expression side inequality.
    with budget.temporary_scope() if budget is not None else nullcontext():
        from ..discretization._coordinate_enclosure import _reserve_polynomial

        if budget is not None:
            budget.reserve(3 * len(normals) * ambient + 2 * (a.size + b.size))
        normal_denominator = math.lcm(
            *(Fraction(value).denominator for normal in normals for value in normal)
        )
        coordinate_denominator = math.lcm(
            *(value.denominator for corners in (a, b) for row in corners for value in row)
        )
        normal_bits = (
            max(
                abs(Fraction(value).numerator).bit_length()
                for normal in normals
                for value in normal
            )
            + normal_denominator.bit_length()
            + 2
        )
        coordinate_bits = (
            max(
                abs(value.numerator).bit_length()
                for corners in (a, b)
                for row in corners
                for value in row
            )
            + coordinate_denominator.bit_length()
        )
        count = len(normals) ** 2 * ambient + a.size + b.size
        _reserve_polynomial(
            count, count, first.dimension, max(normal_bits, coordinate_bits)
        )
        # One common positive scale preserves every base vector AND all cone
        # sums/differences. Independently normalizing each vector would not.
        base_normals = tuple(
            np.asarray(
                [
                    Fraction(value).numerator
                    * (normal_denominator // Fraction(value).denominator)
                    for value in normal
                ],
                dtype=object,
            )
            for normal in normals
        )
        normals = list(base_normals)
        normals.extend(
            x + y
            for index, x in enumerate(base_normals)
            for y in base_normals[index + 1 :]
        )
        normals.extend(
            x - y
            for index, x in enumerate(base_normals)
            for y in base_normals[index + 1 :]
        )
        a = np.asarray(
            [
                [
                    value.numerator * (coordinate_denominator // value.denominator)
                    for value in row
                ]
                for row in a
            ],
            dtype=object,
        )
        b = np.asarray(
            [
                [
                    value.numerator * (coordinate_denominator // value.denominator)
                    for value in row
                ]
                for row in b
            ],
            dtype=object,
        )
        prepared_a, prepared_b = (
            (_prepare_plane_controls(first), _prepare_plane_controls(second))
            if prepared_controls is None
            else prepared_controls
        )
        for normal in normals:
            with budget.temporary_scope() if budget is not None else nullcontext():
                if not np.any(normal):
                    continue
                if budget is not None:
                    budget.reserve(2 * (a.size + b.size))
                projected_a, projected_b = a @ (2 * normal), b @ (2 * normal)
                for sign in (1, -1):
                    n = sign * 2 * normal
                    left, right = sign * projected_a, sign * projected_b
                    if budget is not None:
                        budget.reserve(ambient + 2 * (len(left) + len(right)))
                    if shared:
                        vertex = next(iter(shared))
                        anchor = a[first.vertices.index(vertex)]
                        integer_offset = -sum(
                            weight * value
                            for weight, value in zip(n, anchor, strict=True)
                        )
                    else:
                        # The factor two makes this exact midpoint integral.
                        integer_offset = -(max(left) + min(right)) // 2
                    corners_a, corners_b = -left - integer_offset, right + integer_offset
                    # This is only an exact negative vertex test. Every plane
                    # that passes still needs its full source-control proof.
                    if any(value < 0 for value in corners_a) or any(
                        value < 0 for value in corners_b
                    ):
                        continue
                    offset = Fraction(integer_offset, coordinate_denominator)
                    prepared = (
                        prepared_a is not None
                        and prepared_b is not None
                        and _plane_preserves_degree(prepared_a, normal)
                        and _plane_preserves_degree(prepared_b, normal)
                    )
                    if prepared and prepared_a is not None and prepared_b is not None:
                        minimal_a, permissive_a = _prepared_plane_side(
                            first, prepared_a, -n, -offset, corners_a, shared
                        )
                        minimal_b, permissive_b = _prepared_plane_side(
                            second, prepared_b, n, offset, corners_b, shared
                        )
                        if (minimal_a and permissive_b) or (permissive_a and minimal_b):
                            return True
                        continue
                    pa = scale(_signed_image(first, n, offset), -1)
                    pb = _signed_image(second, n, offset)
                    # One side's zero set must lie on the shared reference entities.
                    # The other side may have a larger planar boundary support: its
                    # independent injectivity and exact shared trace exclude extra
                    # preimages of the common contact.
                    if (
                        _strict_side(first, pa, shared)
                        and _strict_side(second, pb, shared, require_shared_support=False)
                    ) or (
                        _strict_side(first, pa, shared, require_shared_support=False)
                        and _strict_side(second, pb, shared)
                    ):
                        return True
    return False


def _inside(point: np.ndarray, cell: _Cell, margin: float = 0.0) -> bool:
    if np.any(point <= margin) or np.any(point >= 1.0 - margin):
        return False
    if cell.domain == "simplex":
        return bool(np.sum(point) < 1.0 - margin)
    if cell.domain == "prism":
        return bool(np.sum(point[:2]) < 1.0 - margin)
    return True


def _overlap_witness(
    first: _Cell, second: _Cell, source: np.ndarray | None = None
) -> bool:
    if (
        len(first.coordinates) != first.dimension
        or len(second.coordinates) != second.dimension
    ):
        return False
    ambient = second.dimension
    if source is None:
        source = np.full(
            (first.dimension,),
            1.0 / (first.dimension + 1) if first.domain == "simplex" else 0.25,
            dtype=np.float64,
        )
    if not _inside(source, first):
        return False
    target = tuple(
        evaluate(value, tuple(Fraction(float(x)) for x in source))
        for value in first.coordinates
    )
    point = np.full(
        (ambient,),
        1.0 / (ambient + 1) if second.domain == "simplex" else 0.5,
        dtype=np.float64,
    )
    jacobian = tuple(
        tuple(derivative(value, axis) for axis in range(ambient))
        for value in second.coordinates
    )
    for _ in range(20):
        center = tuple(Fraction(float(x)) for x in point)
        residual = np.asarray(
            [
                float(evaluate(value, center) - q)
                for value, q in zip(second.coordinates, target, strict=True)
            ],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [[float(evaluate(entry, center)) for entry in row] for row in jacobian],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(matrix) != ambient:
            return False
        step = np.linalg.solve(matrix, residual)
        point -= step
        if np.max(np.abs(step)) < 1.0e-12:
            break
    if not _inside(point, second):
        return False
    radius = (
        min(
            float(np.min(point)),
            float(np.min(1.0 - point)),
            (1.0 - float(np.sum(point))) / ambient
            if second.domain == "simplex"
            else (1.0 - float(np.sum(point[:2]))) / 2.0
            if second.domain == "prism"
            else 1.0,
        )
        * 0.125
    )
    if radius <= 0:
        return False
    center = tuple(Fraction(float(x)) for x in point)
    midpoint = np.asarray(
        [[float(evaluate(entry, center)) for entry in row] for row in jacobian],
        dtype=np.float64,
    )
    preconditioner = np.linalg.solve(midpoint, np.eye(ambient, dtype=np.float64))
    arguments = ArgumentComposition(
        affine_arguments(
            point - radius, np.eye(ambient, dtype=np.float64) * (2.0 * radius)
        )
    )
    residual_polynomials = tuple(
        add(value, constant(-q, ambient))
        for value, q in zip(second.coordinates, target, strict=True)
    )
    for i in range(ambient):
        transformed = sum_polynomials(
            tuple(
                scale(value, Fraction(float(preconditioner[i, k])))
                for k, value in enumerate(residual_polynomials)
            )
        )
        displacement = abs(evaluate(transformed, center))
        gain = Fraction(0)
        for j in range(ambient):
            entry = sum_polynomials(
                tuple(
                    scale(jacobian[k][j], Fraction(float(preconditioner[i, k])))
                    for k in range(ambient)
                )
            )
            remainder = add(constant(int(i == j), ambient), scale(entry, -1))
            coefficients = bernstein_coefficients(arguments(remainder), "box", ambient)
            gain += max(abs(value) for value in coefficients)
        if displacement + gain * Fraction(radius) >= Fraction(radius):
            return False
    return True


def _crossing_witness(first: _Cell, second: _Cell) -> bool:
    """An interval-Newton inclusion witnesses an interior transverse contact."""
    from itertools import combinations

    from ..discretization._coordinate_enclosure import axes

    ambient = len(first.coordinates)
    if ambient > first.dimension + second.dimension or ambient != len(second.coordinates):
        return False
    point_a = np.full(
        (first.dimension,),
        1.0 / (first.dimension + 1) if first.domain == "simplex" else 0.5,
        dtype=np.float64,
    )
    point_b = np.full(
        (second.dimension,),
        1.0 / (second.dimension + 1) if second.domain == "simplex" else 0.5,
        dtype=np.float64,
    )
    derivatives_a = tuple(
        tuple(derivative(value, axis) for axis in range(first.dimension))
        for value in first.coordinates
    )
    derivatives_b = tuple(
        tuple(derivative(value, axis) for axis in range(second.dimension))
        for value in second.coordinates
    )
    for _ in range(20):
        a = tuple(Fraction(float(value)) for value in point_a)
        b = tuple(Fraction(float(value)) for value in point_b)
        residual = np.asarray(
            [
                float(evaluate(x, a) - evaluate(y, b))
                for x, y in zip(first.coordinates, second.coordinates, strict=True)
            ],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [
                [float(evaluate(entry, a)) for entry in row_a]
                + [-float(evaluate(entry, b)) for entry in row_b]
                for row_a, row_b in zip(derivatives_a, derivatives_b, strict=True)
            ],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(matrix) != ambient:
            return False
        update, _, rank, _ = np.linalg.lstsq(matrix, residual, rcond=None)
        if rank != ambient:
            return False
        point_a -= update[: first.dimension]
        point_b -= update[first.dimension :]
        if np.max(np.abs(update)) < 1.0e-12:
            break
    if not _inside(point_a, first) or not _inside(point_b, second):
        return False
    selected = next(
        (
            columns
            for columns in combinations(range(matrix.shape[1]), ambient)
            if np.linalg.matrix_rank(matrix[:, columns]) == ambient
        ),
        None,
    )
    if selected is None:
        return False
    margins = [
        float(np.min(point_a)),
        float(np.min(1.0 - point_a)),
        float(np.min(point_b)),
        float(np.min(1.0 - point_b)),
    ]
    for point, cell in ((point_a, first), (point_b, second)):
        if cell.domain == "simplex":
            margins.append((1.0 - float(np.sum(point))) / cell.dimension)
        elif cell.domain == "prism":
            margins.append((1.0 - float(np.sum(point[:2]))) / 2.0)
    radius = min(margins) * 0.125
    variables = axes(ambient)
    parameters = np.concatenate((point_a, point_b))
    from ..discretization._coordinate_enclosure import add as polynomial_add

    arguments = tuple(
        polynomial_add(
            constant(Fraction(float(value)), ambient),
            variables[selected.index(i)] if i in selected else {},
        )
        for i, value in enumerate(parameters)
    )
    difference = tuple(
        add(
            compose(a, arguments[: first.dimension]),
            scale(compose(b, arguments[first.dimension :]), -1),
        )
        for a, b in zip(first.coordinates, second.coordinates, strict=True)
    )
    origin = (Fraction(0),) * ambient
    jacobian = tuple(
        tuple(derivative(value, axis) for axis in range(ambient)) for value in difference
    )
    midpoint = np.asarray(
        [[float(evaluate(entry, origin)) for entry in row] for row in jacobian],
        dtype=np.float64,
    )
    if np.linalg.matrix_rank(midpoint) != ambient:
        return False
    preconditioner = np.linalg.solve(midpoint, np.eye(ambient, dtype=np.float64))
    cube = ArgumentComposition(
        affine_arguments(
            np.full((ambient,), -radius, dtype=np.float64),
            np.eye(ambient, dtype=np.float64) * (2.0 * radius),
        )
    )
    for i in range(ambient):
        residual = sum_polynomials(
            tuple(
                scale(value, Fraction(float(preconditioner[i, k])))
                for k, value in enumerate(difference)
            )
        )
        displacement = abs(evaluate(residual, origin))
        gain = Fraction(0)
        for j in range(ambient):
            entry = sum_polynomials(
                tuple(
                    scale(jacobian[k][j], Fraction(float(preconditioner[i, k])))
                    for k in range(ambient)
                )
            )
            remainder = add(constant(int(i == j), ambient), scale(entry, -1))
            coefficients = bernstein_coefficients(cube(remainder), "box", ambient)
            gain += max(abs(value) for value in coefficients)
        if displacement + gain * Fraction(radius) >= Fraction(radius):
            return False
    return True


def _adjacent_triangle_injectivity(
    first: _Cell, second: _Cell, limits: MeshCertificateLimits, maximum_work: int
) -> tuple[bool, int]:
    """Prove physical projection injectivity on a checked convex two-cell chart."""
    if first.kind != "triangle" or second.kind != "triangle":
        return False, 0
    shared = set(first.vertices).intersection(second.vertices)
    if len(shared) != 2:
        return False, 0
    midpoint = _jacobian_midpoint(first)
    if midpoint is None:
        return False, 0
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    if rank != 2:
        return False, 0
    projected = []
    for cell in (first, second):
        corners = [
            tuple(
                expression_reference_evaluate(value, point, cell.domain)
                for value in cell.coordinates
            )
            for point in cell.reference_vertices
        ]
        projected.append(
            tuple(
                tuple(
                    sum(
                        (
                            Fraction(float(weight)) * value
                            for weight, value in zip(row, point, strict=True)
                        ),
                        Fraction(0),
                    )
                    for row in projection
                )
                for point in corners
            )
        )
    identifiers = sorted(shared)
    a, b = (projected[0][first.vertices.index(identifier)] for identifier in identifiers)
    c = projected[0][
        next(i for i, identifier in enumerate(first.vertices) if identifier not in shared)
    ]
    d = projected[1][
        next(
            i for i, identifier in enumerate(second.vertices) if identifier not in shared
        )
    ]
    for identifier in identifiers:
        if (
            projected[0][first.vertices.index(identifier)]
            != projected[1][second.vertices.index(identifier)]
        ):
            return False, 0

    def turn(
        x: tuple[Fraction, ...], y: tuple[Fraction, ...], z: tuple[Fraction, ...]
    ) -> Fraction:
        return (y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0])

    side = turn(a, b, c)
    if not side or side * turn(a, b, d) >= 0:
        return False, 0
    boundary = (c, a, d, b)
    if any(
        side * turn(boundary[i], boundary[(i + 1) % 4], boundary[(i + 2) % 4]) < 0
        for i in range(4)
    ):
        return False, 0
    transforms = []
    for corners in projected:
        matrix = tuple(
            tuple(corners[j + 1][i] - corners[0][i] for j in range(2)) for i in range(2)
        )
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if not determinant:
            return False, 0
        transforms.append(
            (
                (matrix[1][1] / determinant, -matrix[0][1] / determinant),
                (-matrix[1][0] / determinant, matrix[0][0] / determinant),
            )
        )
    work = 0
    for cell, transform in zip((first, second), transforms, strict=True):
        local = _projected_jacobian(cell, projection)
        if local is None:
            return False, work
        jacobian = tuple(
            tuple(
                sum_polynomials(
                    tuple(scale(local[i][k], transform[k][j]) for k in range(2))
                )
                for j in range(2)
            )
            for i in range(2)
        )
        active = [(np.zeros((2,), dtype=np.float64), np.eye(2, dtype=np.float64), 0)]
        symmetric_parts: dict[tuple[int, int], Expression] = {}
        while active:
            origin, piece, depth = active.pop()
            if work >= maximum_work:
                return False, work
            work += 1
            if _strict_spd(jacobian, "simplex", 2, origin, piece, symmetric_parts):
                continue
            if depth == limits.maximum_subdivision_depth:
                return False, work
            children = _children(cell, origin, piece)
            if len(active) + len(children) > maximum_work - work:
                return False, work
            active.extend((start, child, depth + 1) for start, child in children)
    return True, work


def _adjacent_tensor_injectivity(
    first: _Cell, second: _Cell, limits: MeshCertificateLimits, maximum_work: int
) -> tuple[bool, int]:
    """Continuous piecewise monotonicity on a convex two-cell tensor atlas."""
    shared = np.asarray(
        sorted(set(first.vertices).intersection(second.vertices)), dtype=np.int64
    )
    gluing = _tensor_gluing(first, second, shared)
    if gluing is None:
        return False, 0
    matrix, offset = gluing
    dimension = first.dimension
    reference = np.asarray(reference_cell_topology(first.kind).vertices, dtype=np.float64)
    face = reference[[first.vertices.index(int(value)) for value in shared]]
    fixed = np.flatnonzero(np.ptp(face, axis=0) == 0)
    if fixed.size != 1:
        return False, 0
    axis = int(fixed[0])
    origin = np.zeros(dimension, dtype=np.float64)
    origin[axis] = face[0, axis]
    tangent = np.eye(dimension, dtype=np.float64)[
        :, [i for i in range(dimension) if i != axis]
    ]
    trace_kind = "interval" if dimension == 2 else "quadrilateral"
    left = restrict_chart_expressions(
        first.coordinates, first.kind, trace_kind, origin, tangent
    )
    right = restrict_chart_expressions(
        second.coordinates,
        second.kind,
        trace_kind,
        matrix.T @ (origin - offset),
        matrix.T @ tangent,
    )
    if left != right:
        return False, 0
    midpoint = _jacobian_midpoint(first)
    if midpoint is None:
        return False, 0
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    if rank != dimension:
        return False, 0
    work = 0
    for cell, transform in (
        (first, np.eye(dimension, dtype=np.float64)),
        (second, matrix),
    ):
        local = _projected_jacobian(cell, projection)
        if local is None:
            return False, work
        jacobian = tuple(
            tuple(
                sum_polynomials(
                    tuple(
                        scale(local[i][k], Fraction(float(transform[j, k])))
                        for k in range(dimension)
                    )
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        active = [
            (
                np.zeros(dimension, dtype=np.float64),
                np.eye(dimension, dtype=np.float64),
                0,
            )
        ]
        symmetric_parts: dict[tuple[int, int], Expression] = {}
        while active:
            start, piece, depth = active.pop()
            if work >= maximum_work:
                return False, work
            work += 1
            if _strict_spd(
                jacobian, cell.domain, dimension, start, piece, symmetric_parts
            ):
                continue
            if depth == limits.maximum_subdivision_depth:
                return False, work
            children = _children(cell, start, piece)
            if len(active) + len(children) > maximum_work - work:
                return False, work
            active.extend(
                (child_start, child, depth + 1) for child_start, child in children
            )
    return True, work


def _pair_decision(
    first: _Cell,
    second: _Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    prepared_controls: tuple[_PlaneControls | None, _PlaneControls | None] | None = None,
    *,
    authoritative_first_traces: tuple[tuple[int, ...], ...] = (),
    tangent_cone: bool = True,
) -> tuple[str, int]:
    if not _trace_equal(first, second):
        return "mapped_trace_mismatch", 0
    if _multi_affine_axis_separation(
        first, second, set(first.vertices).intersection(second.vertices)
    ):
        return "separated", 0
    from ._affine_mapped_contact import affine_simplex_contact

    affine = affine_simplex_contact(
        first,
        second,
        maximum_work,
        authoritative_first_traces=authoritative_first_traces,
    )
    if affine is not None:
        return affine
    cone_work = 0
    if tangent_cone:
        from ._tangent_cone_contact import tangent_cone_contact

        contact = tangent_cone_contact(first, second, limits, maximum_work)
        if contact is not None:
            proved, cone_work = contact
            if proved:
                return "separated", cone_work
    if _separating_plane(first, second, prepared_controls):
        return "separated", cone_work
    adjacent, adjacency_work = _adjacent_triangle_injectivity(
        first, second, limits, maximum_work - cone_work
    )
    adjacency_work += cone_work
    if adjacent:
        return "separated", adjacency_work
    tensor_adjacent, tensor_work = _adjacent_tensor_injectivity(
        first, second, limits, maximum_work - adjacency_work
    )
    adjacency_work += tensor_work
    if tensor_adjacent:
        return "separated", adjacency_work
    from ._prism_embedding import adjacent_prism_injectivity

    prism_adjacent, prism_work = adjacent_prism_injectivity(
        first, second, limits, maximum_work - adjacency_work
    )
    adjacency_work += prism_work
    if prism_adjacent:
        return "separated", adjacency_work
    if (
        _overlap_witness(first, second)
        or _overlap_witness(second, first)
        or _crossing_witness(first, second)
    ):
        return "overlap", adjacency_work
    for parameter in (0.1, 0.2, 0.3, 0.4):
        source_a = np.full((first.dimension,), parameter, dtype=np.float64)
        source_b = np.full((second.dimension,), parameter, dtype=np.float64)
        if _overlap_witness(first, second, source_a) or _overlap_witness(
            second, first, source_b
        ):
            return "overlap", adjacency_work
    from contextlib import nullcontext

    budget = _COORDINATE_BUDGET.get()
    # Each entry owns two affine charts (four NumPy owners), two chart tuples,
    # its queue tuple and depth. This upper includes the queue's block slots.
    chart_bytes = 1024 + 32 * max(first.dimension, second.dimension) ** 2
    with nullcontext() if budget is None else budget.live_storage() as live:
        if live is not None:
            live.set_bound(1024 + 2 * chart_bytes)
        identity_a = (
            np.zeros((first.dimension,), dtype=np.float64),
            np.eye(first.dimension, dtype=np.float64),
        )
        identity_b = (
            np.zeros((second.dimension,), dtype=np.float64),
            np.eye(second.dimension, dtype=np.float64),
        )
        active = deque(((identity_a, identity_b, 0),))
        work = adjacency_work
        undecided: str | None = None
        while active:
            with nullcontext() if budget is None else budget.temporary_scope():
                a, b, depth = active.popleft()
                work += 1
                if work > maximum_work:
                    return "mapped_intersection_piece_budget", work - 1
                if budget is not None:
                    # Two bound pairs, their base owners, and node-local Cell
                    # dictionaries stay live until this node's last predicate.
                    budget.reserve(
                        0,
                        2048 + 128 * max(len(first.coordinates), len(second.coordinates)),
                    )
                lo_a, hi_a = _bounds(first, *a)
                lo_b, hi_b = _bounds(second, *b)
                if np.any(hi_a < lo_b) or np.any(hi_b < lo_a):
                    del a, b, lo_a, hi_a, lo_b, hi_b
                    if live is not None:
                        live.set_bound(1024 + chart_bytes * (len(active) + 1))
                    continue
                if depth > 0:
                    composition_a, composition_b = (
                        ArgumentComposition(affine_arguments(*a)),
                        ArgumentComposition(affine_arguments(*b)),
                    )
                    piece_a = _Cell(
                        tuple(composition_a(value) for value in first.coordinates),
                        first.domain,
                        first.kind,
                        first.dimension,
                        first.vertices,
                        first.reference_vertices,
                    )
                    piece_b = _Cell(
                        tuple(composition_b(value) for value in second.coordinates),
                        second.domain,
                        second.kind,
                        second.dimension,
                        second.vertices,
                        second.reference_vertices,
                    )
                    if (
                        _overlap_witness(piece_a, piece_b)
                        or _overlap_witness(piece_b, piece_a)
                        or _crossing_witness(piece_a, piece_b)
                    ):
                        return "overlap", work
                    del piece_a, piece_b, composition_a, composition_b
                if depth == limits.maximum_subdivision_depth:
                    undecided = "mapped_intersection_subdivision_depth"
                    del a, b, lo_a, hi_a, lo_b, hi_b
                    if live is not None:
                        live.set_bound(1024 + chart_bytes * (len(active) + 1))
                    continue
                if live is not None:
                    live.set_bound(
                        1024
                        + chart_bytes
                        * (len(active) + 2 + 2 ** max(first.dimension, second.dimension))
                    )
                if np.max(hi_a - lo_a) >= np.max(hi_b - lo_b):
                    children = tuple(
                        (child, b, depth + 1) for child in _children(first, *a)
                    )
                else:
                    children = tuple(
                        (a, child, depth + 1) for child in _children(second, *b)
                    )
                if len(active) + len(children) > maximum_work - work:
                    return "mapped_intersection_piece_budget", work
                active.extend(children)
                del children
            del a, b, lo_a, hi_a, lo_b, hi_b
            if live is not None:
                live.set_bound(1024 + chart_bytes * (len(active) + 1))
        return "separated" if undecided is None else undecided, work


def _interval_trace_identity(
    first: _Cell, second: _Cell, vertices: np.ndarray
) -> bool | None:
    """Exact edge trace identity of polynomial cells, or None for another owner.

    Composing a total-degree-``p`` polynomial with an affine edge map gives a
    univariate polynomial of degree at most ``p``. Two such restrictions agree
    identically exactly when they agree at ``p + 1`` distinct exact parameters.
    """
    polynomials: list[tuple[Polynomial, ...]] = []
    for cell in (first, second):
        if cell.kind == "pyramid":
            return None
        coordinates: list[Polynomial] = []
        for value in cell.coordinates:
            if isinstance(value, RationalPolynomial):
                return None
            coordinates.append(value)
        polynomials.append(tuple(coordinates))
    # Maximum total degree of the actual composed source polynomials, never a
    # nominal element degree; affine edge restriction cannot exceed it.
    degree = max(
        (
            sum(index)
            for coordinates in polynomials
            for value in coordinates
            for index in value
        ),
        default=0,
    )
    # Both edge parameters run from the facet's first nominal vertex to its
    # second, so opposite local cell orientations share one exact chart.
    ends = []
    for cell in (first, second):
        reference = reference_cell_topology(cell.kind).vertices
        ends.append(
            tuple(
                tuple(
                    Fraction(float(value))
                    for value in reference[cell.vertices.index(int(vertex))]
                )
                for vertex in vertices
            )
        )
    for step in range(degree + 1):
        parameter = Fraction(step, max(degree, 1))
        values = []
        for restricted, (start, end) in zip(polynomials, ends, strict=True):
            point = tuple(
                a + parameter * (b - a) for a, b in zip(start, end, strict=True)
            )
            values.append(tuple(evaluate(value, point) for value in restricted))
        if values[0] != values[1]:
            return False
    return True


def _multi_affine_trace_identity(
    first: _Cell, second: _Cell, vertices: np.ndarray
) -> bool | None:
    """Exact facet trace identity of multi-affine box cells, or None for another owner.

    A multi-affine map restricted to an edge or square face of its box is
    multi-affine on that facet and equals the interpolant of its corner values.
    When both cells order the facet vertices around the same square, the two
    restrictions agree identically exactly when the exact corner images of the
    shared vertices agree.
    """
    if (
        first.corner_images is None
        or second.corner_images is None
        or vertices.size not in (2, 4)
    ):
        return None
    budget = _COORDINATE_BUDGET.get()
    images = []
    for cell, corner_images in (
        (first, first.corner_images),
        (second, second.corner_images),
    ):
        rows = [cell.vertices.index(int(vertex)) for vertex in vertices]
        if vertices.size == 4:
            if budget is not None:
                budget.reserve(2 * cell.dimension)
            # Opposite corners of the ordered square share their midpoint.
            reference = cell.reference_vertices
            if any(
                reference[rows[0]][axis] + reference[rows[2]][axis]
                != reference[rows[1]][axis] + reference[rows[3]][axis]
                for axis in range(cell.dimension)
            ):
                return None
        images.append(tuple(corner_images[row] for row in rows))
    if budget is not None:
        budget.reserve(vertices.size * len(first.coordinates))
    return images[0] == images[1]


def _trace_continuity(
    state: _EmbeddingState,
    mesh: CellMesh,
    cells: list[_Cell],
    cell_groups: np.ndarray | None = None,
    *,
    shared_charts: bool = False,
    geometry: CellGeometrySpec | None = None,
) -> None:
    """Prove exact facet trace identity of every interior facet.

    With ``shared_charts`` many cells share one exact corner-image record, as the
    block charts of restricted children do; a multi-affine identity decided for
    two records at the same local facet corners is then reused, not re-proved.
    """
    from contextlib import nullcontext

    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET
    from ._mesh_certificates import _mesh_facets

    facets = _mesh_facets(mesh, geometry=geometry)
    state.checks.append("mapped_trace_continuity")
    budget = _COORDINATE_BUDGET.get()
    decided: dict[tuple[int, ...], bool | None] = {}
    for group in np.flatnonzero(facets.counts == 2):
        occurrences = np.flatnonzero(facets.group == group)
        if (
            cell_groups is not None
            and cell_groups[int(facets.cells[occurrences[0]])]
            != cell_groups[int(facets.cells[occurrences[1]])]
        ):
            continue
        # Restrictions are needed only for this facet; the original cell
        # expressions stay retained by the enclosing proof's source bank.
        with nullcontext() if budget is None else budget.temporary_scope():
            first = cells[int(facets.cells[occurrences[0]])]
            second = cells[int(facets.cells[occurrences[1]])]
            vertices = facets.rows[occurrences[0]]
            vertices = vertices[vertices >= 0]
            entities = np.asarray((group,), dtype=np.int64)
            identity = (
                _interval_trace_identity(first, second, vertices)
                if vertices.size == 2
                else None
            )
            if identity is None:
                if (
                    shared_charts
                    and first.corner_images is not None
                    and second.corner_images is not None
                ):
                    key = (
                        id(first.corner_images),
                        id(second.corner_images),
                        *(first.vertices.index(int(vertex)) for vertex in vertices),
                        -1,
                        *(second.vertices.index(int(vertex)) for vertex in vertices),
                    )
                    if key not in decided:
                        decided[key] = _multi_affine_trace_identity(
                            first, second, vertices
                        )
                    identity = decided[key]
                else:
                    identity = _multi_affine_trace_identity(first, second, vertices)
            if identity is not None:
                if not identity:
                    state.add("mapped_trace_mismatch", "violated", "facet", entities)
                continue
            restrictions = []
            for cell in (first, second):
                reference = np.asarray(
                    reference_cell_topology(cell.kind).vertices, dtype=np.float64
                )
                corners = reference[
                    [cell.vertices.index(int(vertex)) for vertex in vertices]
                ]
                if vertices.size == 1:
                    source = tuple(Fraction(float(value)) for value in corners[0])
                    if cell.kind == "pyramid" and source[2] == 1:
                        source = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
                    restrictions.append(
                        tuple(evaluate(value, source) for value in cell.coordinates)
                    )
                    continue
                matrix = (
                    (corners[[1]] - corners[0]).T
                    if vertices.size == 2
                    else (corners[[1, 2 if vertices.size == 3 else 3]] - corners[0]).T
                )
                restrictions.append(
                    restrict_chart_expressions(
                        cell.coordinates,
                        cell.kind,
                        "interval"
                        if vertices.size == 2
                        else "triangle"
                        if vertices.size == 3
                        else "quadrilateral",
                        corners[0],
                        matrix,
                    )
                )
            if any(value is None for value in restrictions):
                state.add("mapped_trace_denominator", "unresolved", "facet", entities)
            elif restrictions[0] != restrictions[1]:
                state.add("mapped_trace_mismatch", "violated", "facet", entities)


def _tensor_gluing(
    first: _Cell, second: _Cell, shared: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if first.kind not in ("quadrilateral", "hexahedron") or second.kind != first.kind:
        return None
    dimension = first.dimension
    if shared.size != 2 ** (dimension - 1):
        return None
    a = np.asarray(reference_cell_topology(first.kind).vertices, dtype=np.float64)[
        [first.vertices.index(int(vertex)) for vertex in shared]
    ]
    b = np.asarray(reference_cell_topology(second.kind).vertices, dtype=np.float64)[
        [second.vertices.index(int(vertex)) for vertex in shared]
    ]
    fixed_a = np.flatnonzero(np.ptp(a, axis=0) == 0)
    fixed_b = np.flatnonzero(np.ptp(b, axis=0) == 0)
    if fixed_a.size != 1 or fixed_b.size != 1:
        return None
    ia, ib = int(fixed_a[0]), int(fixed_b[0])
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[ia, ib] = (1.0 if a[0, ia] == 1.0 else -1.0) * (
        1.0 if b[0, ib] == 0.0 else -1.0
    )
    for axis in range(dimension):
        if axis == ib:
            continue
        delta = b - b[0]
        candidates = np.flatnonzero(
            (np.abs(delta[:, axis]) == 1.0) & (np.count_nonzero(delta, axis=1) == 1)
        )
        if candidates.size != 1:
            return None
        row = int(candidates[0])
        matrix[:, axis] = (a[row] - a[0]) / delta[row, axis]
    offset = a[0] - matrix @ b[0]
    if not np.array_equal(b @ matrix.T + offset, a):
        return None
    return matrix, offset


def _common_atlas_extension(
    cells: list[_Cell],
    charts: dict[int, tuple[np.ndarray, np.ndarray]],
    lower: np.ndarray,
    spans: np.ndarray,
    state: _EmbeddingState,
    limits: MeshCertificateLimits,
) -> bool:
    """Prove one actual polynomial map is injective beyond an occupied atlas.

    This extension premise permits cavities and nonconvex reference occupancy;
    merely matching nodal values, or positive Jacobians on occupied cells, does
    not establish it.
    """
    common: tuple[Expression, ...] | None = None
    for cell_id, (matrix, offset) in charts.items():
        arguments = affine_arguments(-matrix.T @ offset, matrix.T)
        expression = tuple(
            compose(value, arguments) for value in cells[cell_id].coordinates
        )
        if common is None:
            common = expression
        elif common != expression:
            return False
    if common is None:
        return False
    dimension = lower.size
    chart = affine_arguments(lower, np.diag(spans))
    extension = _Cell(
        tuple(compose(value, chart) for value in common),
        "box",
        cells[next(iter(charts))].kind,
        dimension,
        (),
        (),
    )
    if any(
        bernstein_node_count(value, "box", dimension) > limits.maximum_bernstein_nodes
        for value in extension.coordinates
    ):
        return False
    return _injective(extension, limits, state) is None


def _tensor_component_injectivity(
    state: _EmbeddingState,
    mesh: CellMesh,
    cells: list[_Cell],
    limits: MeshCertificateLimits,
) -> np.ndarray:
    """A complete rectangular reference atlas has a convex domain.

    Strict monotonicity of its continuous piecewise map proves injectivity of
    the *whole* component, including diagonally adjacent curved faces.
    """
    from ._mesh_certificates import _mesh_facets

    labels = np.full((len(cells),), -1, dtype=np.int64)
    facets = _mesh_facets(mesh)
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in cells]
    for group in np.flatnonzero(facets.counts == 2):
        occurrences = np.flatnonzero(facets.group == group)
        a, b = (int(value) for value in facets.cells[occurrences])
        shared = facets.rows[occurrences[0]]
        shared = shared[shared >= 0]
        adjacency[a].append((b, shared))
        adjacency[b].append((a, shared))
    visited: set[int] = set()
    for root, root_cell in enumerate(cells):
        if root in visited or root_cell.kind not in ("quadrilateral", "hexahedron"):
            continue
        dimension = root_cell.dimension
        charts = {
            root: (
                np.eye(dimension, dtype=np.float64),
                np.zeros((dimension,), dtype=np.float64),
            )
        }
        active = [root]
        consistent = True
        while active:
            a = active.pop()
            outer, shift = charts[a]
            for b, shared in adjacency[a]:
                gluing = _tensor_gluing(cells[a], cells[b], shared)
                if gluing is None:
                    consistent = False
                    continue
                matrix, offset = gluing
                chart = (outer @ matrix, shift + outer @ offset)
                if b in charts:
                    old = charts[b]
                    consistent &= bool(
                        np.array_equal(old[0], chart[0])
                        and np.array_equal(old[1], chart[1])
                    )
                else:
                    charts[b] = chart
                    active.append(b)
        visited.update(charts)
        if not consistent:
            continue
        positions = []
        vertices_by_position: dict[tuple[float, ...], int] = {}
        positions_by_vertex: dict[int, tuple[float, ...]] = {}
        for cell_id, (matrix, offset) in charts.items():
            reference = np.asarray(
                reference_cell_topology(cells[cell_id].kind).vertices, dtype=np.float64
            )
            points = reference @ matrix.T + offset
            positions.append(tuple(np.min(points, axis=0).tolist()))
            for vertex, point in zip(cells[cell_id].vertices, points, strict=True):
                position = tuple(point.tolist())
                if (
                    position in vertices_by_position
                    and vertices_by_position[position] != vertex
                ):
                    consistent = False
                if (
                    vertex in positions_by_vertex
                    and positions_by_vertex[vertex] != position
                ):
                    consistent = False
                vertices_by_position[position] = vertex
                positions_by_vertex[vertex] = position
        if not consistent or len(set(positions)) != len(positions):
            continue
        origins = np.asarray(positions, dtype=np.float64)
        spans = np.max(origins, axis=0) - np.min(origins, axis=0) + 1.0
        if not np.all(origins == np.round(origins)):
            continue
        if np.prod(spans) != len(charts):
            if _common_atlas_extension(
                cells, charts, np.min(origins, axis=0), spans, state, limits
            ):
                labels[list(charts)] = root
                state.checks.append("mapped_injective_atlas_extension")
            continue
        midpoint = _jacobian_midpoint(root_cell)
        if midpoint is None or np.linalg.matrix_rank(midpoint) != dimension:
            continue
        projection, _, rank, _ = np.linalg.lstsq(
            midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
        )
        if rank != dimension:
            continue
        for cell_id, (matrix, _) in charts.items():
            local = _projected_jacobian(cells[cell_id], projection)
            if local is None:
                consistent = False
                break
            # A signed permutation is orthogonal exactly.
            jacobian = tuple(
                tuple(
                    sum_polynomials(
                        tuple(
                            scale(local[i][k], Fraction(float(matrix[j, k])))
                            for k in range(dimension)
                        )
                    )
                    for j in range(dimension)
                )
                for i in range(dimension)
            )
            active_pieces = [
                (
                    np.zeros((dimension,), dtype=np.float64),
                    np.eye(dimension, dtype=np.float64),
                    0,
                )
            ]
            symmetric_parts: dict[tuple[int, int], Expression] = {}
            while active_pieces:
                origin, transform, depth = active_pieces.pop()
                if (
                    state.subdivision_pieces >= limits.maximum_subdivision_pieces
                    or depth > limits.maximum_subdivision_depth
                ):
                    consistent = False
                    break
                state.subdivision_pieces += 1
                if _strict_spd(
                    jacobian,
                    cells[cell_id].domain,
                    dimension,
                    origin,
                    transform,
                    symmetric_parts,
                ):
                    continue
                if depth == limits.maximum_subdivision_depth:
                    consistent = False
                    break
                active_pieces.extend(
                    (start, child, depth + 1)
                    for start, child in _children(cells[cell_id], origin, transform)
                )
            if not consistent:
                break
        if consistent:
            labels[list(charts)] = root
            state.checks.append("mapped_convex_reference_atlas")
    return labels


def _straight_boundary_images(mesh: CellMesh, cells: list[_Cell]) -> bool:
    """Prove that the actual mapped boundary equals its polygonal carrier."""
    from ..discretization._coordinate_enclosure import restrict_coordinates
    from ._mesh_certificates import _mesh_facets

    if any(
        isinstance(value, RationalPolynomial)
        for cell in cells
        for value in cell.coordinates
    ):
        return False
    budget = _COORDINATE_BUDGET.get()
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    for cell in cells:
        if cell.corner_images is not None:
            if budget is not None:
                budget.reserve(len(cell.corner_images) * len(cell.coordinates))
            if any(
                value != Fraction(float(coordinate))
                for image, vertex in zip(cell.corner_images, cell.vertices, strict=True)
                for value, coordinate in zip(image, points[vertex], strict=True)
            ):
                return False
            continue
        reference = cell.reference_vertices
        vertices = (
            cell.vertices
            if cell.kind != "pyramid"
            else (*cell.vertices[:4], *([cell.vertices[4]] * 4))
        )
        for vertex, point in zip(vertices, reference, strict=True):
            if any(
                evaluate(value, point) != Fraction(float(coordinate))
                for value, coordinate in zip(
                    cell.coordinates, points[vertex], strict=True
                )
            ):
                return False
    facets = _mesh_facets(mesh)
    for occurrence in np.flatnonzero(facets.boundary):
        cell = cells[int(facets.cells[occurrence])]
        vertices = facets.rows[occurrence]
        vertices = vertices[vertices >= 0]
        if cell.corner_images is not None and vertices.size in (2, 4):
            # A multi-affine restriction interpolates its corner images, which
            # equal the mesh points: an edge image is straight, and a square face
            # image is the planar quadrilateral exactly when its four points are
            # coplanar, because the plane residual is bilinear on the face.
            if vertices.size == 4:
                if budget is not None:
                    budget.reserve(5 * len(cell.coordinates) + 8)
                p = [
                    tuple(Fraction(float(value)) for value in point)
                    for point in points[vertices]
                ]
                u = tuple(b - a for a, b in zip(p[0], p[1], strict=True))
                v = tuple(b - a for a, b in zip(p[0], p[3], strict=True))
                w = tuple(b - a for a, b in zip(p[0], p[2], strict=True))
                normal = (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
                if not any(normal) or sum(
                    (a * b for a, b in zip(normal, w, strict=True)), Fraction(0)
                ):
                    return False
            continue
        reference = np.asarray(
            reference_cell_topology(cell.kind).vertices, dtype=np.float64
        )
        corners = reference[[cell.vertices.index(int(vertex)) for vertex in vertices]]
        matrix = (
            (corners[[1]] - corners[0]).T
            if vertices.size == 2
            else (corners[[1, 2 if vertices.size == 3 else 3]] - corners[0]).T
        )
        source_polynomials = tuple(
            value
            for value in cell.coordinates
            if not isinstance(value, RationalPolynomial)
        )
        if len(source_polynomials) != len(cell.coordinates):
            return False
        mapped = restrict_coordinates(source_polynomials, cell.kind, corners[0], matrix)
        if mapped is None:
            return False
        if vertices.size <= 3:
            if any(sum(index) > 1 for polynomial in mapped for index in polynomial):
                return False
            continue
        # A planar, injective bilinear face has the exact quadrilateral image:
        # all four boundary restrictions are straight and the Jordan degree is 1.
        if any(
            any(exponent > 1 for exponent in index)
            for polynomial in mapped
            for index in polynomial
        ):
            return False
        p = [
            tuple(Fraction(float(value)) for value in point) for point in points[vertices]
        ]
        u = tuple(b - a for a, b in zip(p[0], p[1], strict=True))
        v = tuple(b - a for a, b in zip(p[0], p[3], strict=True))
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        if not any(normal):
            return False
        plane = add(
            sum_polynomials(
                tuple(
                    scale(value, weight)
                    for value, weight in zip(mapped, normal, strict=True)
                )
            ),
            constant(
                -sum((a * b for a, b in zip(normal, p[0], strict=True)), Fraction(0)), 2
            ),
        )
        if plane:
            return False
    return True


def _plane_controls_storage_upper(
    controls: _PlaneControls, ledger: CoordinateEnclosureBudget, /
) -> int:
    """Meter the immutable local control net before it escapes a node scope."""
    size = sys.getsizeof(controls) + 128
    for bank in controls:
        ledger.reserve(1)
        size += sys.getsizeof(bank) + 128
    for net in controls.coordinates:
        ledger.reserve(1 + len(net))
        size += sys.getsizeof(net) + 128
        size += sum(sys.getsizeof(value) + 128 for value in net)
    for support in controls.degree_support:
        ledger.reserve(1)
        size += sys.getsizeof(support) + 128
        for indices in support:
            ledger.reserve(1 + len(indices))
            size += sys.getsizeof(indices) + 128
            size += sum(sys.getsizeof(value) + 128 for value in indices)
    return size


def _certify_affine_parent_surface_contact(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    limits: MeshCertificateLimits,
    /,
) -> bool:
    """Replace a complete exact nested partition by its original parent atlas."""
    from ..discretization._coordinate_enclosure import (
        coordinate_corner_images,
        coordinate_partition_unity_reference_chain,
        evaluate,
        prepared_coordinate_source_bank,
        RationalPolynomial,
    )
    from ._mesh_certificates import _exact_source_predicate_view, _pairwise_contacts

    origin = geometry.restriction_source
    if origin is None:
        return False
    elements, routes, _ = geometry.resolve(mesh)
    bank = prepared_coordinate_source_bank(geometry)
    reference = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    parent_vertices: dict[int, tuple[int, ...]] = {}
    parent_corners: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    children: dict[int, list[tuple[tuple[Fraction, ...], ...]]] = {}
    target_images: list[tuple[Fraction, ...] | None] = [None] * mesh.coordinates.shape[0]
    numeric = np.asarray(mesh.coordinates, dtype=np.float64)
    reference_actions: dict[
        str, tuple[CellGeometryElement, tuple[tuple[Fraction, ...], ...]]
    ] = {}
    for block, element, routes_ in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind != "triangle":
            return False
        parent_ids = np.asarray(origin.block_parent_cell_ids[block.name], dtype=np.int64)
        parent_rows = np.asarray(
            origin.block_parent_vertex_ids[block.name], dtype=np.int64
        )
        target_rows = np.asarray(block.vertices, dtype=np.int64)
        for parent, vertices, target_vertices, route in zip(
            parent_ids.tolist(),
            parent_rows.tolist(),
            target_rows.tolist(),
            np.asarray(routes_).tolist(),
            strict=True,
        ):
            local = tuple(bank[int(index)] for index in route)
            prepared = reference_actions.get(element.element_id)
            if prepared is None:
                try:
                    root, arguments = coordinate_partition_unity_reference_chain(element)
                except ValueError:
                    return False
                if any(
                    isinstance(value, RationalPolynomial)
                    or any(sum(index) > 1 for index in value)
                    for value in arguments
                ):
                    return False
                polynomial_arguments = cast(tuple[Polynomial, ...], arguments)
                exact_reference = tuple(
                    tuple(evaluate(value, point) for value in polynomial_arguments)
                    for point in reference
                )
                prepared = root, exact_reference
                reference_actions[element.element_id] = prepared
            root, exact_reference = prepared
            exact_parent = coordinate_corner_images(root, local)
            if exact_parent is None or len(exact_parent) != 3:
                return False
            physical = []
            interpolation_work = 0
            for point in exact_reference:
                weights = (1 - sum(point, Fraction(0)), *point)
                nonzero = tuple(index for index, weight in enumerate(weights) if weight)
                if len(nonzero) == 1 and weights[nonzero[0]] == 1:
                    physical.append(exact_parent[nonzero[0]])
                    continue
                interpolation_work += len(nonzero) * mesh.ambient_dimension
                physical.append(
                    tuple(
                        sum(
                            (
                                weights[index] * exact_parent[index][axis]
                                for index in nonzero
                            ),
                            Fraction(0),
                        )
                        for axis in range(mesh.ambient_dimension)
                    )
                )
            budget = _COORDINATE_BUDGET.get()
            if budget is not None:
                budget.reserve(interpolation_work)
            for vertex, image in zip(target_vertices, physical, strict=True):
                if not np.array_equal(
                    np.asarray(
                        tuple(float(value) for value in image), dtype=np.float64
                    ).view(np.uint64),
                    numeric[int(vertex)].view(np.uint64),
                ):
                    raise ValueError(
                        "Nested affine carrier is not the RNE of its exact source map."
                    )
                previous = target_images[int(vertex)]
                if previous is not None and previous != image:
                    raise ValueError(
                        "Nested affine children disagree on an exact shared vertex."
                    )
                target_images[int(vertex)] = image
            identifier = int(parent)
            vertex_ids = tuple(int(value) for value in vertices)
            known_vertices = parent_vertices.setdefault(identifier, vertex_ids)
            known_corners = parent_corners.setdefault(identifier, exact_parent)
            if known_vertices != vertex_ids or known_corners != exact_parent:
                raise ValueError(
                    "Nested affine children disagree on their scientific parent map."
                )
            children.setdefault(identifier, []).append(exact_reference)
    if any(value is None for value in target_images):
        return False
    compact: dict[int, int] = {}
    points: list[tuple[Fraction, ...]] = []
    triangles: list[tuple[int, int, int]] = []
    identifiers: list[int] = []
    for parent in sorted(children):
        partitions = children[parent]
        total = Fraction(0)
        for triangle in partitions:
            a, b, c = triangle
            twice = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if twice <= 0 or any(
                value < 0 or sum(point, Fraction(0)) > 1
                for point in triangle
                for value in point
            ):
                return False
            total += twice
        if total != 1:
            return False
        for first in range(len(partitions)):
            for second in range(first + 1, len(partitions)):
                separated = False
                for subject, other in (
                    (partitions[first], partitions[second]),
                    (partitions[second], partitions[first]),
                ):
                    for edge in range(3):
                        a, b = subject[edge], subject[(edge + 1) % 3]
                        budget = _COORDINATE_BUDGET.get()
                        if budget is not None:
                            budget.reserve(3)
                        sides = tuple(
                            (b[0] - a[0]) * (point[1] - a[1])
                            - (b[1] - a[1]) * (point[0] - a[0])
                            for point in other
                        )
                        if max(sides) <= 0:
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    return False
        row = []
        for vertex, point in zip(
            parent_vertices[parent], parent_corners[parent], strict=True
        ):
            position = compact.get(vertex)
            if position is None:
                position = len(points)
                compact[vertex] = position
                points.append(point)
            elif points[position] != point:
                raise ValueError(
                    "Nested affine parents disagree on an exact shared vertex."
                )
            row.append(position)
        triangles.append((row[0], row[1], row[2]))
        identifiers.append(parent)
    predicate_points, binary = _exact_source_predicate_view(
        np.asarray(points, dtype=object)
    )
    if binary:
        state.checks.append("exact_source_binary64_value_equivalence")
    _pairwise_contacts(
        state,
        predicate_points,
        np.asarray(triangles, dtype=np.int64),
        np.asarray(identifiers, dtype=np.int64),
        "cell",
        "cell_contact",
        limits,
    )
    state.checks.extend(
        (
            "mapped_affine_source_maps",
            "mapped_exact_shared_corner_incidence",
            "mapped_cell_contact",
            "mapped_exact_affine_surface_contact",
            "mapped_affine_parent_partition",
        )
    )
    return True


def _certify_affine_source_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
) -> bool:
    """Prove actual complete affine simplex maps on their exact shared vertices.

    A mapped coefficient bank may have nonbinary corner images. Its correctly
    rounded mesh is not the source: affine incidence/contact predicates consume
    the authenticated exact images, without changing the public mapped binding.
    """
    from ..discretization._coordinate_enclosure import (
        coordinate_corner_images,
        coordinate_expressions,
        RationalPolynomial,
        source_expressions,
    )
    from ._mesh_certificates import (
        _exact_source_predicate_view,
        _mesh_facets,
        _surface_embedding,
        _volume_embedding,
    )

    dimension = mesh.topological_dimension
    surface = dimension == 2 and mesh.ambient_dimension == 3
    if (dimension != mesh.ambient_dimension and not surface) or dimension not in (2, 3):
        return False
    kind = "triangle" if dimension == 2 else "tetrahedron"
    if any(block.cell_kind != kind for block in mesh.blocks):
        return False
    if geometry.restriction_source is not None:
        if surface and _certify_affine_parent_surface_contact(
            state, mesh, geometry, limits
        ):
            return True
        # Restricted children are not independent scientific authority. The
        # general restricted route certifies the complete declared roots and
        # their exact partition before promoting child embedding.
        return False
    elements, routes, _ = geometry.resolve(mesh)
    bank = prepared_coordinate_source_bank(geometry)
    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(0, 128 + 8 * mesh.coordinates.shape[0])
    images: list[tuple[Fraction, ...] | None] = [None] * mesh.coordinates.shape[0]
    numeric = np.asarray(mesh.coordinates, dtype=np.float64)
    cursor = 0

    def affine(values: tuple[Expression, ...], /) -> bool:
        return not any(
            isinstance(value, RationalPolynomial)
            or any(sum(index) > 1 for index in value)
            for value in values
        )

    for block, element, routes_ in zip(mesh.blocks, elements, routes, strict=True):
        expressions = source_expressions(element)
        if expressions is None:
            return False
        # A higher-degree element is admitted cell by cell when its complete
        # coordinate action is exactly affine (straight Lagrange nodes).
        element_affine = affine(expressions)
        if element_affine and ledger is not None:
            polynomial_expressions = cast(tuple[Polynomial, ...], expressions)
            ledger.reserve(sum(len(value) for value in polynomial_expressions))
        for vertices, route in zip(
            np.asarray(block.vertices), np.asarray(routes_), strict=True
        ):
            local = tuple(bank[int(index)] for index in route)
            if not element_affine:
                action = coordinate_expressions(element, local)
                if action is None or not affine(action):
                    return False
            corners_ = coordinate_corner_images(element, local)
            if corners_ is None:
                return False
            corners = corners_
            for vertex, image in zip(vertices, corners, strict=True):
                if ledger is not None:
                    ledger.reserve(len(image))
                if not np.array_equal(
                    np.asarray(
                        tuple(float(value) for value in image), dtype=np.float64
                    ).view(np.uint64),
                    numeric[int(vertex)].view(np.uint64),
                ):
                    raise ValueError(
                        "Affine source corner carrier is not the RNE of its actual complete map."
                    )
                previous = images[int(vertex)]
                if previous is not None and previous != image:
                    state.add(
                        "mapped_trace_mismatch",
                        "violated",
                        "cell",
                        cell_ids[cursor : cursor + 1],
                    )
                    return True
                images[int(vertex)] = image
            columns = tuple(
                tuple(
                    corner[axis] - corners[0][axis]
                    for axis in range(mesh.ambient_dimension)
                )
                for corner in corners[1:]
            )
            if ledger is not None:
                ledger.reserve(
                    dimension * mesh.ambient_dimension
                    + (9 if surface else 3 if dimension == 2 else 14)
                )
            if surface:
                a, b = columns
                cross = (
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                )
                determinant = int(any(value != 0 for value in cross))
            elif dimension == 2:
                determinant = (
                    columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]
                )
            else:
                a, b, c = columns
                determinant = (
                    a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
                )
            if determinant <= 0:
                state.add(
                    "mapped_local_orientation",
                    "violated",
                    "cell",
                    cell_ids[cursor : cursor + 1],
                )
                return True
            cursor += 1
    if any(image is None for image in images):
        # Unused carrier rows have no authored cell map and must not be promoted
        # to source authority by copying numerical coordinates.
        return False
    exact = tuple(image for image in images if image is not None)
    if ledger is not None:
        ledger.retain_basis(exact)
    state.checks.extend(
        (
            "mapped_affine_source_maps",
            "mapped_exact_shared_corner_incidence",
            "mapped_cell_contact",
        )
    )
    if surface:
        state.checks.append("mapped_exact_affine_surface_contact")
        predicate_points, binary = _exact_source_predicate_view(
            np.asarray(exact, dtype=object)
        )
        if binary:
            state.checks.append("exact_source_binary64_value_equivalence")
        _surface_embedding(state, mesh, predicate_points, cell_ids, limits)
    else:
        predicate_points, binary = _exact_source_predicate_view(
            np.asarray(exact, dtype=object)
        )
        if binary:
            state.checks.append("exact_source_binary64_value_equivalence")
        _volume_embedding(state, mesh, predicate_points, _mesh_facets(mesh), limits)
    return True


def certify_mapped_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
) -> None:
    """Keep root cells, bounds and local control caches live for this proof."""
    from contextlib import nullcontext

    ledger = _COORDINATE_BUDGET.get()
    with nullcontext() if ledger is None else ledger.temporary_scope():
        if _certify_affine_source_embedding(state, mesh, geometry, cell_ids, limits):
            return
    with nullcontext() if ledger is None else ledger.temporary_scope():
        with nullcontext() if ledger is None else ledger.live_storage() as live:
            _certify_mapped_embedding(state, mesh, geometry, cell_ids, limits, live)


def _certify_mapped_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    live: CoordinateLiveStorage | None,
) -> None:
    from contextlib import nullcontext

    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        _has_cartesian_reference_chain,
        coordinate_scope_key,
        PreparedCoordinateCells,
    )
    from ._mesh_certificates import _candidate_pairs

    ledger = _COORDINATE_BUDGET.get()

    state.checks.extend(("mapped_local_injectivity", "mapped_cell_contact"))
    elements, routes, _ = geometry.resolve(mesh)

    points = prepared_coordinate_source_bank(geometry)
    cells = []
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    if geometry.restriction_source is not None and all(
        _has_cartesian_reference_chain(element) for element in elements
    ):
        from ._restricted_embedding import certify_restricted_embedding

        certify_restricted_embedding(state, mesh, geometry, cell_ids, limits)
        return
    if ledger is not None:
        total = cell_ids.size
        # Cell dictionaries, box owners/views, stacked bounds, and the actual
        # local lists/dictionaries survive all per-cell temporary scopes.
        if live is not None:
            live.set_bound(
                1024 + total * (1536 + 64 * mesh.ambient_dimension), work=total
            )
        else:
            ledger.reserve(total)
    descriptors = []
    prepared_coordinates = []
    prepared_ids = []
    offset = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        kind = block.cell_kind
        domain = _domain(kind)
        for row, vertices in zip(
            np.asarray(route), np.asarray(block.vertices), strict=True
        ):
            with ledger.temporary_scope() if ledger is not None else nullcontext():
                local = tuple(points[index] for index in row)
                prepared = multi_affine_coordinates(element, local)
                polynomials: tuple[Expression, ...] | None
                if prepared is None:
                    polynomials, corner_images = (
                        coordinate_expressions(element, local),
                        None,
                    )
                else:
                    polynomials, corner_images = prepared
                if polynomials is None:
                    state.add(
                        "coordinate_source_enclosure",
                        "unresolved",
                        "cell",
                        cell_ids[offset : offset + 1],
                    )
                    return
                count = max(
                    bernstein_node_count(value, domain, mesh.topological_dimension)
                    for value in polynomials
                )
                if count > limits.maximum_bernstein_nodes:
                    state.add(
                        "mapped_bernstein_node_budget",
                        "unresolved",
                        "cell",
                        cell_ids[offset : offset + 1],
                    )
                    return
                cell = _Cell(
                    polynomials,
                    domain,
                    kind,
                    mesh.topological_dimension,
                    tuple(vertices.tolist()),
                    _vertices(kind),
                    corner_images,
                )
                root_lower, root_upper = _bounds(
                    cell,
                    np.zeros((cell.dimension,), dtype=np.float64),
                    np.eye(cell.dimension, dtype=np.float64),
                )
                if not (
                    np.all(np.isfinite(root_lower)) and np.all(np.isfinite(root_upper))
                ):
                    state.add(
                        "mapped_coordinate_bound_range",
                        "unresolved",
                        "cell",
                        cell_ids[offset : offset + 1],
                    )
                    return
                reason = _injective(cell, limits, state)
                if reason is not None:
                    state.add(reason, "unresolved", "cell", cell_ids[offset : offset + 1])
                    children = _children(
                        cell,
                        np.zeros((cell.dimension,), dtype=np.float64),
                        np.eye(cell.dimension, dtype=np.float64),
                    )
                    pieces = [
                        _Cell(
                            tuple(
                                map(
                                    ArgumentComposition(affine_arguments(start, matrix)),
                                    cell.coordinates,
                                )
                            ),
                            cell.domain,
                            cell.kind,
                            cell.dimension,
                            cell.vertices,
                            cell.reference_vertices,
                        )
                        for start, matrix in children
                    ]
                    for a in range(len(pieces)):
                        for b in range(a + 1, len(pieces)):
                            if state.candidate_pairs >= limits.maximum_candidate_pairs:
                                state.add(
                                    "mapped_self_contact_candidate_budget",
                                    "unresolved",
                                    "cell",
                                    cell_ids[offset : offset + 1],
                                )
                                break
                            state.candidate_pairs += 1
                            if _crossing_witness(pieces[a], pieces[b]):
                                state.add(
                                    "mapped_self_contact",
                                    "violated",
                                    "cell",
                                    cell_ids[offset : offset + 1],
                                )
            if ledger is not None:
                # These exact source records remain strongly owned by cells;
                # derived single-cell workspace does not survive this stage.
                ledger.retain_basis(cell.coordinates)
                ledger.retain_basis(cell.reference_vertices)
                ledger.retain_basis(cell.vertices)
                if cell.corner_images is not None:
                    ledger.retain_basis(cell.corner_images)
            if ledger is not None:
                descriptors.append((kind, element, local, cell.vertices))
                prepared_coordinates.append(cell.coordinates)
                prepared_ids.append(int(cell_ids[offset]))
            cells.append(cell)
            boxes.append((root_lower, root_upper))
            offset += 1
    if ledger is not None:
        key = coordinate_scope_key(mesh, geometry)
        prepared = PreparedCoordinateCells(
            tuple(prepared_ids),
            tuple(descriptors),
            tuple(prepared_coordinates),
        )
        ledger.retain_basis(
            (key, prepared, prepared.cell_ids, prepared.descriptors, prepared.coordinates)
        )
        ledger.prepared_cell_cache[key] = prepared
    with ledger.temporary_scope() if ledger is not None else nullcontext():
        _trace_continuity(state, mesh, cells, geometry=geometry)
        straight_boundary = (
            state.clean
            and mesh.topological_dimension == mesh.ambient_dimension
            and mesh.topological_dimension in (2, 3)
            and _straight_boundary_images(mesh, cells)
        )
    if straight_boundary:
        from ._mesh_certificates import _mesh_facets, _volume_embedding

        state.checks.append("mapped_straight_boundary_degree")
        _volume_embedding(
            state,
            mesh,
            np.asarray(mesh.coordinates, dtype=np.float64),
            _mesh_facets(mesh),
            limits,
        )
        return
    from ._boundary_degree_embedding import certify_boundary_degree_embedding

    with ledger.temporary_scope() if ledger is not None else nullcontext():
        decided = certify_boundary_degree_embedding(state, mesh, cells, cell_ids, limits)
    if decided:
        return
    with ledger.temporary_scope() if ledger is not None else nullcontext():
        atlas_labels = _tensor_component_injectivity(state, mesh, cells, limits)
    lower = np.stack([value[0] for value in boxes])
    upper = np.stack([value[1] for value in boxes])
    unbounded = ~(np.all(np.isfinite(lower), axis=1) & np.all(np.isfinite(upper), axis=1))
    if np.any(unbounded):
        state.add(
            "mapped_coordinate_bound_range", "unresolved", "cell", cell_ids[unbounded]
        )
        return
    first, second, exceeded = _candidate_pairs(
        lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
    )
    state.candidate_pairs += first.size
    if exceeded:
        state.add("mapped_candidate_pair_budget", "unresolved", "mesh")
    prepared_controls: dict[int, _PlaneControls | None] = {}
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        if atlas_labels[a] >= 0 and atlas_labels[a] == atlas_labels[b]:
            continue
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            trace_equal = _trace_equal(cells[a], cells[b])
        if not trace_equal:
            state.add("mapped_trace_mismatch", "violated", "cell", cell_ids[[a, b]])
            continue
        for index in (a, b):
            if index in prepared_controls:
                continue
            with ledger.temporary_scope() if ledger is not None else nullcontext():
                prepared = _prepare_plane_controls(cells[index])
                if prepared is not None and ledger is not None and live is not None:
                    # This cache is local to the proof, not a retained history
                    # bank. Its actual immutable payload escapes this scope.
                    live.set_bound(
                        live.bound + _plane_controls_storage_upper(prepared, ledger)
                    )
            prepared_controls[index] = prepared
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            decision, work = _pair_decision(
                cells[a],
                cells[b],
                limits,
                limits.maximum_subdivision_pieces - state.subdivision_pieces,
                (prepared_controls[a], prepared_controls[b]),
            )
        state.subdivision_pieces += work
        if decision == "overlap":
            state.add("mapped_cell_overlap", "violated", "cell", cell_ids[[a, b]])
        elif decision != "separated":
            state.add(
                decision,
                "violated" if decision == "mapped_trace_mismatch" else "unresolved",
                "cell",
                cell_ids[[a, b]],
            )
