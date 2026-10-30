# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Conforming hex-core closure with explicit pyramid and tetrahedron transitions."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from typing import final, Literal, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometryRestrictionSource,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    bernstein_coefficients,
    compose_expression,
    constant,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
    expression_equal,
    linear_combinations,
    multiply,
    Polynomial,
    RationalPolynomial,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem import lagrange_element
from ..typing import checked, ConvertibleToArray
from ._dual_extraction import _RestrictedCoordinateMap
from ._hex_cut_generation import (
    _is_tetrahedron_block,
    _require_global_embedding,
    _require_tetrahedral_global_embedding,
    _tetrahedral_mesh_incidence,
    GlobalEmbeddingCertificate,
)
from ._hex_generation import (
    _compose_restricted_dual_chart,
    _trilinear_chart,
    extract_volume_hexes,
)


Point3: TypeAlias = tuple[Fraction, Fraction, Fraction]
ExactPoint3: TypeAlias = tuple[ExactScalar, ExactScalar, ExactScalar]


def _fraction_point3(values: Sequence[float], /) -> Point3:
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


def _exact_coordinate_coefficients(
    values: Sequence[Sequence[object]],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    rows: list[tuple[ExactScalar, ...]] = []
    for row in values:
        exact_row: list[ExactScalar] = []
        for value in row:
            if isinstance(value, (Fraction, RadicalElement)):
                exact_row.append(value)
            elif isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                exact_row.append(Fraction(int(value)))
            elif isinstance(value, (float, np.floating)):
                exact_row.append(Fraction.from_float(float(value)))
            else:
                raise TypeError(
                    "Exact mixed-family charts require rational or real-algebraic source coordinates."
                )
        rows.append(tuple(exact_row))
    return tuple(rows)


def _exact_coordinate_bank_key(
    values: Sequence[Sequence[ExactScalar]], /
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    return tuple(
        tuple(
            ("fraction", value.numerator, value.denominator)
            if isinstance(value, Fraction)
            else ("algebraic", str(value))
            for value in row
        )
        for row in values
    )

def _determinant3(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    third: tuple[int, int, int],
) -> int:
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _orient3d_sign(points: np.ndarray, vertices: tuple[int, int, int, int], /) -> int:
    exact = tuple(
        tuple(Fraction(float(value)) for value in points[row])
        for row in vertices
    )
    origin, second, third, fourth = exact
    first_edge = tuple(second[axis] - origin[axis] for axis in range(3))
    second_edge = tuple(third[axis] - origin[axis] for axis in range(3))
    third_edge = tuple(fourth[axis] - origin[axis] for axis in range(3))
    determinant = (
        first_edge[0] * (second_edge[1] * third_edge[2] - second_edge[2] * third_edge[1])
        - first_edge[1] * (second_edge[0] * third_edge[2] - second_edge[2] * third_edge[0])
        + first_edge[2] * (second_edge[0] * third_edge[1] - second_edge[1] * third_edge[0])
    )
    return 1 if determinant > 0 else -1 if determinant < 0 else 0


@final
class HexDominantConstruction(StrictModule, NonTrainableState):
    """All output-family transitions, ancestry, and whole-cell validity."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    global_embedding: GlobalEmbeddingCertificate
    vertex_supports: np.ndarray
    core_cell_ids: np.ndarray
    parent_offsets: np.ndarray
    parent_cells: np.ndarray
    parent_cell_ids: np.ndarray
    hex_cell_count: int = eqx.field(static=True)
    pyramid_cell_count: int = eqx.field(static=True)
    tetrahedron_cell_count: int = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_validity_id: str = eqx.field(static=True)
    reference_chart_determinant_bounds: tuple[
        tuple[Fraction, Fraction], ...
    ] = eqx.field(static=True)
    source_reference_coverage: tuple[Fraction, ...] = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: CellValidityCertificate,
        global_embedding: GlobalEmbeddingCertificate,
        vertex_supports: ConvertibleToArray,
        core_cell_ids: ConvertibleToArray,
        parent_offsets: ConvertibleToArray,
        parent_cells: ConvertibleToArray,
        parent_cell_ids: ConvertibleToArray,
        hex_cell_count: int,
        pyramid_cell_count: int,
        tetrahedron_cell_count: int,
        source: CellMesh,
        source_geometry_id: str,
        source_validity_id: str,
        reference_chart_determinant_bounds: Sequence[
            tuple[Fraction, Fraction]
        ],
        source_reference_coverage: Sequence[Fraction],
        construction_id: str,
        /,
    ) -> None:
        core = np.asarray(core_cell_ids, dtype=np.int64)
        offsets = np.asarray(parent_offsets, dtype=np.int64)
        parents = np.asarray(parent_cells, dtype=np.int64)
        parent_ids = np.asarray(parent_cell_ids, dtype=np.int64)
        supports = np.asarray(vertex_supports, dtype=np.int64)
        chart_bounds = tuple(
            (Fraction(lower), Fraction(upper))
            for lower, upper in reference_chart_determinant_bounds
        )
        reference_coverage = tuple(
            Fraction(value) for value in source_reference_coverage
        )
        counts = (int(hex_cell_count), int(pyramid_cell_count), int(tetrahedron_cell_count))
        cell_count = sum(counts)
        if not isinstance(global_embedding, GlobalEmbeddingCertificate):
            raise TypeError("global_embedding must be a GlobalEmbeddingCertificate.")
        if (
            global_embedding.status != "certified"
            or global_embedding.validity.certificate_id != validity.certificate_id
        ):
            raise ValueError("Hex-dominant global embedding must bind its current validity.")
        global_embedding.binding.require(mesh, geometry)
        if not source.blocks or any(
            not _is_tetrahedron_block(block) for block in source.blocks
        ):
            raise ValueError("Hex-dominant ancestry requires tetrahedral source blocks.")
        source_cell_count = sum(block.cell_count for block in source.blocks)
        if mesh.topological_dimension != 3 or not mesh.blocks:
            raise ValueError("Hex-dominant construction requires a nonempty volume mesh.")
        output_families = ("hexahedron", "pyramid", "tetrahedron")
        if any(
            reference_cell_topology(block.cell_kind).family not in output_families
            for block in mesh.blocks
        ):
            raise ValueError("Hex-dominant transitions may contain only hexes, pyramids, and tetrahedra.")
        actual = {
            family: sum(
                block.cell_count
                for block in mesh.blocks
                if reference_cell_topology(block.cell_kind).family == family
            )
            for family in output_families
        }
        if actual != dict(zip(output_families, counts, strict=True)):
            raise ValueError("Hex-dominant family counts must match actual mesh blocks.")
        if (
            core.ndim != 1
            or not core.size
            or np.unique(core).size != core.size
            or offsets.shape != (cell_count + 1,)
            or offsets[0] != 0
            or offsets[-1] != parents.size
            or np.any(offsets[1:] <= offsets[:-1])
            or np.any((parents < 0) | (parents >= source_cell_count))
            or parents.shape != (cell_count,)
            or parent_ids.shape != (cell_count,)
            or np.any(parent_ids < 0)
            or supports.shape != (mesh.coordinates.shape[0], 4)
            or np.any(supports < -1)
            or len(chart_bounds) != cell_count
            or any(lower <= 0 or upper < lower for lower, upper in chart_bounds)
            or len(reference_coverage) != source_cell_count
            or any(value != Fraction(1, 6) for value in reference_coverage)
            or np.any(np.asarray(validity.status) != int(CellValidityStatus.CERTIFIED_VALID))
        ):
            raise ValueError("Hex-dominant ancestry and whole-cell proof banks must align.")
        if not all((source_geometry_id, source_validity_id, construction_id)):
            raise ValueError("Hex-dominant source and proof identities must be non-empty.")
        self.mesh = mesh
        self.geometry = geometry
        self.validity = validity
        self.global_embedding = global_embedding
        self.core_cell_ids = core
        self.vertex_supports = supports
        self.parent_offsets = offsets
        self.parent_cells = parents
        self.parent_cell_ids = parent_ids
        self.hex_cell_count, self.pyramid_cell_count, self.tetrahedron_cell_count = counts
        self.source_mesh_id = source.mesh_id
        self.source_topology_id = source.topology_id
        self.source_geometry_id = str(source_geometry_id)
        self.source_validity_id = str(source_validity_id)
        self.reference_chart_determinant_bounds = chart_bounds
        self.source_reference_coverage = reference_coverage
        self.construction_id = str(construction_id)


def _pyramid_orientation(
    points: np.ndarray,
    base: tuple[int, int, int, int],
    apex: int,
    reference_sign: int,
    /,
) -> tuple[int, int, int, int, int]:
    first = _orient3d_sign(points, (base[0], base[1], base[2], apex))
    second = _orient3d_sign(points, (base[0], base[2], base[3], apex))
    if first == 0 or second == 0 or first != second:
        raise ValueError("A clipped PLC quad face does not form a nondegenerate pyramid transition.")
    if first != reference_sign:
        return (base[0], base[3], base[2], base[1], apex)
    return (*base, apex)


def _tetrahedron_chart(
    vertices: tuple[Point3, Point3, Point3, Point3],
    /,
) -> tuple[Polynomial, Polynomial, Polynomial]:
    return (
        {
            (0, 0, 0): vertices[0][0],
            (1, 0, 0): vertices[1][0] - vertices[0][0],
            (0, 1, 0): vertices[2][0] - vertices[0][0],
            (0, 0, 1): vertices[3][0] - vertices[0][0],
        },
        {
            (0, 0, 0): vertices[0][1],
            (1, 0, 0): vertices[1][1] - vertices[0][1],
            (0, 1, 0): vertices[2][1] - vertices[0][1],
            (0, 0, 1): vertices[3][1] - vertices[0][1],
        },
        {
            (0, 0, 0): vertices[0][2],
            (1, 0, 0): vertices[1][2] - vertices[0][2],
            (0, 1, 0): vertices[2][2] - vertices[0][2],
            (0, 0, 1): vertices[3][2] - vertices[0][2],
        },
    )


def _pyramid_shape_functions(
    *,
    collapsed: bool,
) -> tuple[Expression, Expression, Expression, Expression, Expression]:
    if collapsed:
        first, second, height = axes(3)
        one = constant(Fraction(1), 3)
        lower = add(one, scale(height, Fraction(-1)))
        return (
            multiply(multiply(lower, add(one, scale(first, -1))), add(one, scale(second, -1))),
            multiply(multiply(lower, first), add(one, scale(second, -1))),
            multiply(multiply(lower, first), second),
            multiply(multiply(lower, add(one, scale(first, -1))), second),
            height,
        )
    first, second, height = axes(3)
    one = constant(Fraction(1), 3)
    half = Fraction(1, 2)
    denominator = add(one, scale(height, Fraction(-1)))
    shifted_first = add(first, scale(height, -half))
    shifted_second = add(second, scale(height, -half))
    lower_first = add(denominator, scale(shifted_first, -1))
    lower_second = add(denominator, scale(shifted_second, -1))
    return (
        RationalPolynomial(multiply(lower_first, lower_second), denominator),
        RationalPolynomial(multiply(shifted_first, lower_second), denominator),
        RationalPolynomial(multiply(shifted_first, shifted_second), denominator),
        RationalPolynomial(multiply(lower_first, shifted_second), denominator),
        height,
    )


def _pyramid_chart(
    vertices: Sequence[ExactPoint3],
    /,
    *,
    collapsed: bool = False,
) -> tuple[Expression, Expression, Expression]:
    shapes = _pyramid_shape_functions(collapsed=collapsed)
    return (
        sum_polynomials(
            tuple(scale(shapes[row], vertices[row][0]) for row in range(5)),
            3,
        ),
        sum_polynomials(
            tuple(scale(shapes[row], vertices[row][1]) for row in range(5)),
            3,
        ),
        sum_polynomials(
            tuple(scale(shapes[row], vertices[row][2]) for row in range(5)),
            3,
        ),
    )


def _divide_by_one_minus_height(
    polynomial: Polynomial,
    /,
) -> Polynomial | None:
    groups: dict[tuple[int, int], dict[int, ExactScalar]] = {}
    for powers, coefficient in polynomial.items():
        if len(powers) != 3:
            return None
        if coefficient == 0:
            continue
        group = groups.setdefault((powers[0], powers[1]), {})
        group[powers[2]] = coefficient
    quotient: Polynomial = {}
    for (first_power, second_power), coefficients in groups.items():
        degree = max(coefficients)
        previous = coefficients.get(0, Fraction(0))
        if previous != 0:
            quotient[(first_power, second_power, 0)] = previous
        for height_power in range(1, degree):
            current = coefficients.get(height_power, Fraction(0)) + previous
            if current != 0:
                quotient[(first_power, second_power, height_power)] = current
            previous = current
        if coefficients.get(degree, Fraction(0)) + previous != 0:
            return None
    return quotient


def _pyramid_collapsed_rational_extension(
    expression: RationalPolynomial,
    /,
) -> Polynomial | None:
    """Prove a pyramid-root rational chart extends polynomially to its apex."""
    denominator = {
        tuple(int(value) for value in powers): coefficient
        for powers, coefficient in expression.denominator.items()
        if coefficient != 0
    }
    if not denominator or any(
        len(powers) != 3 or powers[0] != 0 or powers[1] != 0
        for powers in denominator
    ):
        return None
    degree = max(powers[2] for powers in denominator)
    scale_factor = denominator.get((0, 0, 0), Fraction(0))
    if degree == 0 or scale_factor == 0:
        return None
    expected_denominator = {
        (0, 0, power): scale_factor
        * Fraction((-1) ** power * comb(degree, power))
        for power in range(degree + 1)
    }
    if denominator != expected_denominator:
        return None

    first, second, height = axes(3)
    unit = constant(Fraction(1), 3)
    width = add(unit, scale(height, Fraction(-1)))
    half_height = scale(height, Fraction(1, 2))
    collapsed_chart = (
        add(multiply(width, first), half_height),
        add(multiply(width, second), half_height),
        height,
    )
    numerator = compose_expression(expression.numerator, collapsed_chart)
    if isinstance(numerator, RationalPolynomial):
        return None
    extension: Polynomial = dict(numerator)
    for _ in range(degree):
        quotient = _divide_by_one_minus_height(extension)
        if quotient is None:
            return None
        extension = quotient
    extension = {
        powers: coefficient / scale_factor
        for powers, coefficient in extension.items()
        if coefficient != 0
    }
    apex_terms: dict[tuple[int, int], ExactScalar] = {}
    for (first_power, second_power, _), coefficient in extension.items():
        key = (first_power, second_power)
        apex_terms[key] = apex_terms.get(key, Fraction(0)) + coefficient
    if any(
        key != (0, 0) and coefficient != 0
        for key, coefficient in apex_terms.items()
    ):
        return None
    return extension


def _chart_jacobian_determinant(
    chart: tuple[Expression, Expression, Expression],
    /,
) -> Expression:
    jacobian = tuple(
        tuple(derivative(component, axis) for axis in range(3))
        for component in chart
    )
    minor0 = add(
        multiply(jacobian[1][1], jacobian[2][2]),
        scale(multiply(jacobian[1][2], jacobian[2][1]), -1),
    )
    minor1 = add(
        multiply(jacobian[1][0], jacobian[2][2]),
        scale(multiply(jacobian[1][2], jacobian[2][0]), -1),
    )
    minor2 = add(
        multiply(jacobian[1][0], jacobian[2][1]),
        scale(multiply(jacobian[1][1], jacobian[2][0]), -1),
    )
    return add(
        add(
            multiply(jacobian[0][0], minor0),
            scale(multiply(jacobian[0][1], minor1), -1),
        ),
        multiply(jacobian[0][2], minor2),
    )




def _require_polynomial_expression(
    expression: Expression,
    /,
) -> Polynomial:
    if isinstance(expression, RationalPolynomial):
        raise ValueError("Exact mixed-family reference charts require polynomials.")
    return expression
def _pyramid_reduced_chart_determinant(
    vertices: tuple[Point3, Point3, Point3, Point3, Point3],
    /,
) -> Polynomial:
    base = _face_chart((vertices[0], vertices[1], vertices[2], vertices[3]))
    base_3d: tuple[Polynomial, Polynomial, Polynomial] = (
        {
            (exponent[0], exponent[1], 0): coefficient
            for exponent, coefficient in base[0].items()
        },
        {
            (exponent[0], exponent[1], 0): coefficient
            for exponent, coefficient in base[1].items()
        },
        {
            (exponent[0], exponent[1], 0): coefficient
            for exponent, coefficient in base[2].items()
        },
    )
    jacobian = tuple(
        (
            derivative(value, 0),
            derivative(value, 1),
            add(
                constant(vertices[4][axis], 3),
                scale(base_3d[axis], Fraction(-1)),
            ),
        )
        for axis, value in enumerate(base_3d)
    )
    minor0 = add(
        multiply(jacobian[1][1], jacobian[2][2]),
        scale(multiply(jacobian[1][2], jacobian[2][1]), -1),
    )
    minor1 = add(
        multiply(jacobian[1][0], jacobian[2][2]),
        scale(multiply(jacobian[1][2], jacobian[2][0]), -1),
    )
    minor2 = add(
        multiply(jacobian[1][0], jacobian[2][1]),
        scale(multiply(jacobian[1][1], jacobian[2][0]), -1),
    )
    determinant = add(
        add(
            multiply(jacobian[0][0], minor0),
            scale(multiply(jacobian[0][1], minor1), -1),
        ),
        multiply(jacobian[0][2], minor2),
    )
    return _require_polynomial_expression(determinant)


def _positive_chart_bounds(
    cell_kind: str,
    chart: tuple[Expression, Expression, Expression],
    cell_id: int,
    /,
    *,
    pyramid_vertices: Sequence[Point3] | None = None,
) -> tuple[Fraction, Fraction]:
    family = reference_cell_topology(cell_kind).family
    if family == "pyramid":
        if pyramid_vertices is None or len(pyramid_vertices) != 5:
            raise ValueError("Pyramid chart proof requires its five exact source corners.")
        pyramid_points = (
            pyramid_vertices[0],
            pyramid_vertices[1],
            pyramid_vertices[2],
            pyramid_vertices[3],
            pyramid_vertices[4],
        )
        determinant: Expression = _pyramid_reduced_chart_determinant(
            pyramid_points
        )
    else:
        determinant = _chart_jacobian_determinant(chart)
    if isinstance(determinant, RationalPolynomial):
        raise ValueError(
            f"Exact {cell_kind} chart determinant for cell ID {cell_id} "
            "is rational rather than a polynomial."
        )
    domain: Literal["simplex", "box"] = (
        "simplex" if family == "tetrahedron" else "box"
    )
    raw_controls = bernstein_coefficients(determinant, domain, 3)
    controls: tuple[Fraction, ...] = tuple(
        value for value in raw_controls if isinstance(value, Fraction)
    )
    if (
        len(controls) != len(raw_controls)
        or not controls
        or min(controls) <= 0
    ):
        raise ValueError(
            f"Exact {cell_kind} reference chart for cell ID {cell_id} lacks "
            "a strictly positive whole-cell Bernstein determinant bound."
        )
    return min(controls), max(controls)


def _reference_chart_volume(
    polynomial: Polynomial,
    domain: Literal["simplex", "box"],
    /,
) -> Fraction:
    total = Fraction(0)
    for exponent, coefficient in polynomial.items():
        if not isinstance(coefficient, Fraction):
            raise ValueError("Mixed-family reference chart volume must be rational.")
        if domain == "simplex":
            degree = sum(exponent)
            total += coefficient * Fraction(
                factorial(exponent[0])
                * factorial(exponent[1])
                * factorial(exponent[2]),
                factorial(degree + 3),
            )
        else:
            total += coefficient * Fraction(
                1,
                (exponent[0] + 1) * (exponent[1] + 1) * (exponent[2] + 1),
            )
    return total


def _face_chart(
    vertices: Sequence[Point3],
    /,
) -> tuple[Polynomial, Polynomial, Polynomial]:
    if len(vertices) == 3:
        return (
            {
                (0, 0): vertices[0][0],
                (1, 0): vertices[1][0] - vertices[0][0],
                (0, 1): vertices[2][0] - vertices[0][0],
            },
            {
                (0, 0): vertices[0][1],
                (1, 0): vertices[1][1] - vertices[0][1],
                (0, 1): vertices[2][1] - vertices[0][1],
            },
            {
                (0, 0): vertices[0][2],
                (1, 0): vertices[1][2] - vertices[0][2],
                (0, 1): vertices[2][2] - vertices[0][2],
            },
        )
    if len(vertices) != 4:
        raise ValueError("Hex-dominant common faces must be triangles or quadrilaterals.")
    result: list[Polynomial] = []
    for axis in range(3):
        q00 = vertices[0][axis]
        q10 = vertices[1][axis]
        q11 = vertices[2][axis]
        q01 = vertices[3][axis]
        result.append(
            {
                (0, 0): q00,
                (1, 0): q10 - q00,
                (0, 1): q01 - q00,
                (1, 1): q00 - q10 + q11 - q01,
            }
        )
    return (result[0], result[1], result[2])


def _canonical_face_cycle(
    cycle: tuple[int, ...],
    vertex_ids: np.ndarray,
    /,
) -> tuple[int, ...]:
    start = min(range(len(cycle)), key=lambda row: int(vertex_ids[cycle[row]]))
    forward = (*cycle[start:], *cycle[:start])
    reverse = (forward[0], *reversed(forward[1:]))
    return (
        forward
        if int(vertex_ids[forward[1]]) < int(vertex_ids[forward[-1]])
        else reverse
    )



def _face_key3(vertices: tuple[int, int, int], /) -> tuple[int, int, int]:
    ordered = sorted(vertices)
    return (ordered[0], ordered[1], ordered[2])


def _edge_key2(first: int, second: int, /) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)

@dataclass(frozen=True, slots=True)
class _SourceCellChart:
    base_element: ScalarCoordinateElement
    root_cell_id: int
    parent_to_root: tuple[Expression, Expression, Expression]
    source_route: tuple[int, ...]
    physical_expressions: tuple[Expression, Expression, Expression]


def _source_cell_chart(
    source: CellMesh,
    geometry: CellGeometrySpec,
    cell_row: int,
    /,
) -> _SourceCellChart:
    elements, routes, _ = geometry.resolve(source)
    block_index = next(
        block_index
        for block_index, block in enumerate(source.blocks)
        if sum(item.cell_count for item in source.blocks[: block_index + 1])
        > cell_row
    )
    block_offset = sum(block.cell_count for block in source.blocks[:block_index])
    local_row = cell_row - block_offset
    element = elements[block_index]
    if not isinstance(element, ScalarCoordinateElement):
        raise ValueError("Mixed-family composition requires scalar coordinate elements.")
    source_cell_id = int(source.blocks[block_index].global_ids[local_row])
    root_element, root_cell_id, parent_to_root = _compose_restricted_dual_chart(
        element,
        local_row,
        source_cell_id,
        axes(3),
    )
    if len(parent_to_root) != 3:
        raise ValueError("Source cell chart must retain all three reference coordinates.")
    route = tuple(
        int(value)
        for value in np.asarray(routes[block_index], dtype=np.int64)[local_row].tolist()
    )
    coefficients = _exact_coordinate_coefficients(
        geometry.source_coordinates(np.asarray(route, dtype=np.int64))
    )
    basis = coordinate_effective_basis(root_element)
    if not coefficients or any(len(row) != 3 for row in coefficients):
        raise ValueError("Mixed-family source coordinates must be three-dimensional.")
    if len(basis) != len(coefficients):
        raise ValueError(
            "Root source coordinate basis and local route differ during mixed-family composition."
        )
    components = (
        tuple(coefficients[dof][0] for dof in range(len(coefficients))),
        tuple(coefficients[dof][1] for dof in range(len(coefficients))),
        tuple(coefficients[dof][2] for dof in range(len(coefficients))),
    )
    root_expressions = linear_combinations(basis, components)
    physical_expressions = tuple(
        compose_expression(expression, parent_to_root)
        for expression in root_expressions
    )
    if len(physical_expressions) != 3:
        raise ValueError("Mixed-family source maps must preserve three physical coordinates.")
    return _SourceCellChart(
        root_element,
        int(root_cell_id),
        (parent_to_root[0], parent_to_root[1], parent_to_root[2]),
        route,
        (
            physical_expressions[0],
            physical_expressions[1],
            physical_expressions[2],
        ),
    )


def _source_cell_coordinate_expressions(
    source: CellMesh,
    geometry: CellGeometrySpec,
    cell_row: int,
    /,
) -> tuple[Expression, Expression, Expression]:
    return _source_cell_chart(source, geometry, cell_row).physical_expressions


@dataclass(frozen=True, slots=True)
class _ComposedCell:
    cell_kind: str
    vertices: tuple[int, ...]
    source_cell_row: int
    source_root_cell_id: int
    global_id: int
    source_chart: tuple[Expression, ...]
    target_chart: tuple[Expression, ...]
    physical_chart: tuple[Expression, ...]
    base_element: ScalarCoordinateElement
    source_route: tuple[int, ...]
    chart_bounds: tuple[Fraction, Fraction]
    reference_volume: Fraction
    pyramid_vertex_family: bool


@final
class _HexDominantCoordinateMap(StrictModule, NonTrainableState):
    """Retain exact parent controls and exact mixed-cell vertex images."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    rounded_vertices: np.ndarray
    exact_vertex_images: tuple[ExactPoint3, ...] = eqx.field(static=True)
    exact_coordinate_bank: tuple[ExactPoint3, ...] = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        exact_vertex_images: tuple[ExactPoint3, ...],
        /,
    ) -> None:
        if not isinstance(source_mesh, CellMesh) or not isinstance(
            target_mesh, CellMesh
        ):
            raise TypeError("Mixed coordinate maps require parent and target CellMesh values.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        source_geometry.resolve(source_mesh)
        if len(exact_vertex_images) != target_mesh.coordinates.shape[0] or any(
            len(point) != 3
            or any(not isinstance(value, (Fraction, RadicalElement)) for value in point)
            for point in exact_vertex_images
        ):
            raise ValueError(
                "Exact mixed-family vertex images must cover the target mesh."
            )
        parent_coefficients = _exact_coordinate_coefficients(
            source_geometry.source_coordinates()
        )
        if any(len(point) != 3 for point in parent_coefficients):
            raise ValueError(
                "Mixed-family parent coefficient rows must be three-dimensional."
            )
        exact_bank = tuple(
            (point[0], point[1], point[2]) for point in parent_coefficients
        ) + exact_vertex_images
        parent_values = np.asarray(source_geometry.coordinates, dtype=np.float64)
        rounded_vertices = np.array(
            target_mesh.coordinates, dtype=np.float64, copy=True
        )
        expected = np.concatenate((parent_values, rounded_vertices), axis=0)
        rounded_bank = np.asarray(exact_bank, dtype=np.float64)
        if rounded_bank.shape != expected.shape or not np.array_equal(
            rounded_bank, expected
        ):
            raise ValueError(
                "Exact mixed-family coefficient bank does not round to its retained controls."
            )
        rounded_vertices.setflags(write=False)
        parent_map = source_geometry.source_map
        source_id = (
            source_mesh.mesh_id if parent_map is None else parent_map.source_id
        )
        source_revision = (
            source_mesh.numeric_version
            if parent_map is None
            else parent_map.source_revision
        )
        if not source_id or not source_revision:
            raise ValueError("Mixed coordinate maps require stable source identity.")
        self.source_mesh = source_mesh
        self.source_geometry = source_geometry
        self.rounded_vertices = rounded_vertices
        self.exact_vertex_images = exact_vertex_images
        self.exact_coordinate_bank = exact_bank
        self.source_mesh_id = source_mesh.mesh_id
        self.source_topology_id = source_mesh.topology_id
        self.target_mesh_id = target_mesh.mesh_id
        self.target_topology_id = target_mesh.topology_id
        self.target_geometry_id = target_mesh.geometry_id
        self.source_id = source_id
        self.source_revision = source_revision
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "hex-dominant-exact-mixed-coordinate-map",
                "source_mesh": source_mesh.mesh_id,
                "source_topology": source_mesh.topology_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "parent_source_map": None
                if parent_map is None
                else parent_map.source_map_id,
                "target_mesh": target_mesh.mesh_id,
                "target_topology": target_mesh.topology_id,
                "target_geometry": target_mesh.geometry_id,
                "source": (source_id, source_revision),
                "exact_coordinate_bank": _exact_coordinate_bank_key(exact_bank),
                "exact_vertex_images": _exact_coordinate_bank_key(
                    exact_vertex_images
                ),
            }
        )

    @classmethod
    def from_stored_fields(
        cls, value: "_HexDominantCoordinateMap", target_mesh: CellMesh, /
    ) -> "_HexDominantCoordinateMap":
        if type(value) is not cls:
            raise TypeError(
                "Stored mixed coordinate maps must be _HexDominantCoordinateMap."
            )
        if (
            target_mesh.mesh_id != value.target_mesh_id
            or target_mesh.topology_id != value.target_topology_id
            or target_mesh.geometry_id != value.target_geometry_id
        ):
            raise ValueError(
                "Stored mixed coordinate maps bind a different target mesh."
            )
        renewed = cls(
            value.source_mesh,
            value.source_geometry,
            target_mesh,
            value.exact_vertex_images,
        )
        if (
            renewed.source_map_id != value.source_map_id
            or renewed.exact_coordinate_bank != value.exact_coordinate_bank
            or not np.array_equal(
                renewed.rounded_vertices, value.rounded_vertices
            )
        ):
            raise ValueError(
                "Stored mixed coordinate fields do not renew their exact identity."
            )
        return renewed

    def source_coordinates(
        self, indices: ConvertibleToArray | None = None, /
    ) -> tuple[ExactPoint3, ...]:
        if indices is None:
            return self.exact_coordinate_bank
        rows = np.asarray(indices)
        if rows.ndim != 1 or rows.dtype.kind not in "iu":
            raise TypeError(
                "Mixed coordinate source indices must be an integer vector."
            )
        row_ids = np.asarray(rows, dtype=np.int64)
        if np.any((row_ids < 0) | (row_ids >= len(self.exact_coordinate_bank))):
            raise IndexError(
                "Mixed coordinate source index lies outside the exact coefficient bank."
            )
        return tuple(self.exact_coordinate_bank[int(row)] for row in row_ids.tolist())


def _source_reference_point(
    support: tuple[int, ...],
    cell_row: int,
    source_cells: np.ndarray,
    /,
) -> Point3:
    cell = tuple(int(value) for value in source_cells[cell_row])
    references = reference_cell_topology("tetrahedron").vertices
    local: dict[int, Point3] = {
        vertex: _fraction_point3(references[corner])
        for corner, vertex in enumerate(cell)
    }
    if not support or any(vertex not in local for vertex in support):
        raise ValueError(
            "Hex-dominant source support lies outside its retained tetrahedron ancestry."
        )
    return (
        sum((local[vertex][0] for vertex in support), Fraction(0)) / len(support),
        sum((local[vertex][1] for vertex in support), Fraction(0)) / len(support),
        sum((local[vertex][2] for vertex in support), Fraction(0)) / len(support),
    )

def _validate_source_geometry_faces(
    source: CellMesh,
    expressions: tuple[tuple[Expression, Expression, Expression], ...],
    /,
) -> None:
    if not source.blocks or any(
        not _is_tetrahedron_block(block) for block in source.blocks
    ):
        raise ValueError("Exact source chart conformity requires tetrahedral blocks.")
    _, faces, _ = _tetrahedral_mesh_incidence(source)
    cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    face_ids = np.asarray(source.entity_set(2).entity_ids, dtype=np.int64)
    face_lookup: dict[tuple[int, int, int], int] = {
        _face_key3((int(face[0]), int(face[1]), int(face[2]))): int(
            face_ids[row]
        )
        for row, face in enumerate(faces.tolist())
    }
    local_faces = reference_cell_topology("tetrahedron").entities[2]
    incidences: dict[
        tuple[int, int, int],
        list[tuple[int, tuple[int, int, int]]],
    ] = {}
    for cell_row, cell in enumerate(cells.tolist()):
        for local_face in local_faces:
            cycle = (
                int(cell[local_face[0]]),
                int(cell[local_face[1]]),
                int(cell[local_face[2]]),
            )
            key = _face_key3(cycle)
            incidences.setdefault(key, []).append((cell_row, cycle))
    for key, adjacent in incidences.items():
        if len(adjacent) > 2:
            raise ValueError(
                f"Curved source chart conformity is undefined at nonmanifold face ID "
                f"{face_lookup[key]}."
            )
        if len(adjacent) != 2:
            continue
        canonical_ids = _face_key3(
            (
                int(vertex_ids[key[0]]),
                int(vertex_ids[key[1]]),
                int(vertex_ids[key[2]]),
            )
        )
        reference_by_id: list[tuple[Point3, Point3, Point3]] = []
        for cell_row, _ in adjacent:
            cell = tuple(int(value) for value in cells[cell_row])
            local_by_id: dict[int, Point3] = {
                int(vertex_ids[vertex]): _fraction_point3(
                    reference_cell_topology("tetrahedron").vertices[local]
                )
                for local, vertex in enumerate(cell)
            }
            reference_by_id.append(
                (
                    local_by_id[canonical_ids[0]],
                    local_by_id[canonical_ids[1]],
                    local_by_id[canonical_ids[2]],
                )
            )
        face_map = _face_chart(reference_by_id[0])
        left = tuple(
            compose_expression(value, face_map)
            for value in expressions[adjacent[0][0]]
        )
        right = tuple(
            compose_expression(value, _face_chart(reference_by_id[1]))
            for value in expressions[adjacent[1][0]]
        )
        if any(
            not expression_equal(first, second)
            for first, second in zip(left, right, strict=True)
        ):
            raise ValueError(
                f"Exact source coordinate charts disagree across source face ID "
                f"{face_lookup[key]}."
            )

def _exact_dual_coordinates(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    scaffold: CellMesh,
    vertex_supports: np.ndarray,
    /,
) -> tuple[
    np.ndarray,
    tuple[tuple[Expression, Expression, Expression], ...],
    dict[int, ExactPoint3],
]:
    if not source.blocks or any(
        not _is_tetrahedron_block(block) for block in source.blocks
    ):
        raise ValueError("Hex-dominant source mapping requires tetrahedral blocks.")
    _tetrahedral_mesh_incidence(source)
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    expressions = tuple(
        _source_cell_coordinate_expressions(source, source_geometry, row)
        for row in range(source_cells.shape[0])
    )
    _validate_source_geometry_faces(source, expressions)
    coordinates = np.asarray(scaffold.coordinates, dtype=np.float64).copy()
    exact_images: dict[int, ExactPoint3] = {}
    for vertex_row, raw_support in enumerate(np.asarray(vertex_supports).tolist()):
        support = tuple(int(value) for value in raw_support if value >= 0)
        candidates = tuple(
            parent
            for parent, cell in enumerate(source_cells.tolist())
            if set(support) <= set(map(int, cell))
        )
        if not candidates:
            raise ValueError(
                f"Hex-dominant support row {vertex_row} has no source tetrahedron ancestor."
            )
        images: tuple[ExactPoint3, ...] = tuple(
            (
                evaluate_expression(
                    expressions[parent][0],
                    _source_reference_point(support, parent, source_cells),
                ),
                evaluate_expression(
                    expressions[parent][1],
                    _source_reference_point(support, parent, source_cells),
                ),
                evaluate_expression(
                    expressions[parent][2],
                    _source_reference_point(support, parent, source_cells),
                ),
            )
            for parent in candidates
        )
        if any(image != images[0] for image in images[1:]):
            raise ValueError(
                "Adjacent exact source charts disagree at shared dual support "
                f"vertex row {vertex_row}."
            )
        exact_images[vertex_row] = images[0]
        rounded = np.asarray([float(value) for value in images[0]], dtype=np.float64)
        if not np.all(np.isfinite(rounded)):
            raise ValueError("Exact source chart composition produced a nonfinite dual vertex.")
        coordinates[vertex_row] = rounded
    return coordinates, expressions, exact_images


def _expression_chart3(
    values: Sequence[Expression],
    /,
) -> tuple[Expression, Expression, Expression]:
    if len(values) != 3:
        raise ValueError("Exact three-dimensional charts require three coordinate expressions.")
    return (values[0], values[1], values[2])


def _compose_mixed_geometry(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    scaffold: CellMesh,
    vertex_supports: np.ndarray,
    coordinates: np.ndarray,
    source_expression_tuple: tuple[
        tuple[Expression, Expression, Expression], ...
    ],
    exact_vertex_images: dict[int, ExactPoint3],
    cell_specs: tuple[tuple[str, tuple[int, ...], int, int], ...],
    /,
) -> tuple[
        CellMesh,
        CellGeometrySpec,
        CellValidityCertificate,
        np.ndarray,
        tuple[tuple[Fraction, Fraction], ...],
        tuple[Fraction, ...],
    ]:
    if (
        len(source.blocks) != 1
        or not _is_tetrahedron_block(source.blocks[0])
    ):
        raise ValueError(
            "Mixed-family composition requires source tetrahedra in one topological block."
        )
    _tetrahedral_mesh_incidence(source)
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    source_cell_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64)
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    support_rows = np.asarray(vertex_supports, dtype=np.int64)
    source_chart_records = tuple(
        _source_cell_chart(source, source_geometry, row)
        for row in range(source_cells.shape[0])
    )
    coordinates = np.asarray(coordinates, dtype=np.float64).copy()
    target_families = {"hexahedron", "pyramid", "tetrahedron"}
    records: list[_ComposedCell] = []
    for kind, raw_vertices, parent, global_id in cell_specs:
        vertices = tuple(int(value) for value in raw_vertices)
        family = reference_cell_topology(kind).family
        if family not in target_families:
            raise ValueError(f"Unsupported hex-dominant target family {kind!r}.")
        source_references: list[Point3] = []
        for target_vertex in vertices:
            support = tuple(
                int(value)
                for value in support_rows[target_vertex].tolist()
                if int(value) >= 0
            )
            source_references.append(
                _source_reference_point(support, parent, source_cells)
            )
        parent_vertices = tuple(source_references)
        if family == "hexahedron":
            source_chart = _expression_chart3(_trilinear_chart(parent_vertices))
            determinant_chart = source_chart
        elif family == "pyramid":
            if len(parent_vertices) != 5:
                raise ValueError("Pyramid transition charts require five exact corners.")
            source_chart = _pyramid_chart(parent_vertices)
            determinant_chart = _pyramid_chart(parent_vertices, collapsed=True)
        else:
            if len(parent_vertices) != 4:
                raise ValueError("Tetrahedral transition charts require four exact corners.")
            tetrahedron_vertices = (
                parent_vertices[0],
                parent_vertices[1],
                parent_vertices[2],
                parent_vertices[3],
            )
            source_chart = _expression_chart3(
                _tetrahedron_chart(tetrahedron_vertices)
            )
            determinant_chart = source_chart
        chart_bounds = _positive_chart_bounds(
            kind,
            determinant_chart,
            global_id,
            pyramid_vertices=parent_vertices if family == "pyramid" else None,
        )
        full_determinant = _require_polynomial_expression(
            _chart_jacobian_determinant(determinant_chart)
        )
        chart_domain: Literal["simplex", "box"] = (
            "simplex" if family == "tetrahedron" else "box"
        )
        reference_volume = _reference_chart_volume(full_determinant, chart_domain)
        if reference_volume <= 0:
            raise ValueError(
                f"Exact {kind} reference chart for cell ID {global_id} has nonpositive orientation."
            )

        root_record = source_chart_records[parent]
        base_element = root_record.base_element
        parent_to_base = root_record.parent_to_root
        if len(parent_to_base) != 3:
            raise ValueError("Source cell chart must retain all three reference coordinates.")
        target_chart = _expression_chart3(
            tuple(
                compose_expression(argument, source_chart)
                for argument in parent_to_base
            )
        )
        source_route = root_record.source_route
        source_root_cell_id = root_record.root_cell_id
        if len(source_route) != int(base_element.local_dof_count):
            raise ValueError(
                "Source chart composition changed the exact root coordinate basis width."
            )
        physical_chart = _expression_chart3(
            tuple(
                compose_expression(expression, source_chart)
                for expression in source_expression_tuple[parent]
            )
        )
        for source_reference, target_vertex in zip(
            parent_vertices, vertices, strict=True
        ):
            exact_image: ExactPoint3 = (
                evaluate_expression(source_expression_tuple[parent][0], source_reference),
                evaluate_expression(source_expression_tuple[parent][1], source_reference),
                evaluate_expression(source_expression_tuple[parent][2], source_reference),
            )
            if exact_image != exact_vertex_images[target_vertex]:
                raise ValueError(
                    f"Exact {kind} chart does not reproduce target vertex row {target_vertex}."
                )
        pyramid_vertex_family = False
        if family == "pyramid":
            vertex_physical = tuple(
                exact_vertex_images[target_vertex] for target_vertex in vertices
            )
            vertex_chart = _pyramid_chart(vertex_physical)
            pyramid_vertex_family = all(
                expression_equal(actual, expected)
                for actual, expected in zip(
                    physical_chart, vertex_chart, strict=True
                )
            )
            if not pyramid_vertex_family:
                raise ValueError(
                    f"Curved source chart for pyramid transition cell ID {global_id} "
                    "has no exact apex-regular family map."
                )
        records.append(
            _ComposedCell(
                kind,
                vertices,
                parent,
                source_root_cell_id,
                int(global_id),
                tuple(source_chart),
                tuple(target_chart),
                tuple(physical_chart),
                base_element,
                source_route,
                chart_bounds,
                reference_volume,
                pyramid_vertex_family,
            )
        )


    source_reference_coverage: list[Fraction] = [
        Fraction(0) for _ in range(source_cells.shape[0])
    ]
    for record in records:
        source_reference_coverage[record.source_cell_row] += record.reference_volume
    expected_source_reference_volume = Fraction(1, 6)
    for parent, volume in enumerate(source_reference_coverage):
        if volume != expected_source_reference_volume:
            raise ValueError(
                f"Hex-dominant mixed templates cover source tetrahedron "
                f"{int(source_cell_ids[parent])} with exact reference volume {volume}, "
                f"not {expected_source_reference_volume}."
            )
    groups: list[list[_ComposedCell]] = []
    group_rows: dict[tuple[str, str], int] = {}
    for record in records:
        rational = any(
            isinstance(argument, RationalPolynomial)
            for argument in record.target_chart
        )
        if rational:
            domain: Literal["simplex", "box"] = (
                "simplex"
                if reference_cell_topology(record.cell_kind).family == "tetrahedron"
                else "box"
            )
            for argument in record.target_chart:
                if not isinstance(argument, RationalPolynomial):
                    continue
                denominator_controls = bernstein_coefficients(
                    argument.denominator, domain, 3
                )
                denominator_has_strict_sign = bool(denominator_controls) and (
                    all(value > 0 for value in denominator_controls)
                    or all(value < 0 for value in denominator_controls)
                )
                if not denominator_has_strict_sign and (
                    record.cell_kind != "pyramid"
                    or _pyramid_collapsed_rational_extension(argument) is None
                ):
                    raise ValueError(
                        f"Exact {record.cell_kind} chart for output cell ID "
                        f"{record.global_id} has an unproved rational denominator."
                    )
        key = (
            record.cell_kind,
            "pyramid-vertex-family"
            if record.pyramid_vertex_family
            else record.base_element.element_id,
        )
        group = group_rows.get(key)
        if group is None:
            group = len(groups)
            group_rows[key] = group
            groups.append([])
        groups[group].append(record)
    target_blocks: list[CellBlock] = []
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    parent_cell_ids: dict[str, np.ndarray] = {}
    parent_vertex_ids: dict[str, np.ndarray] = {}
    source_block_names: dict[str, str] = {}
    flattened_records: list[_ComposedCell] = []
    source_geometry_coordinates = np.asarray(source_geometry.coordinates, dtype=np.float64)
    pyramid_geometry_offset = source_geometry_coordinates.shape[0]
    pyramid_geometry_coordinates = (
        np.concatenate((source_geometry_coordinates, coordinates), axis=0)
        if any(record.pyramid_vertex_family for record in records)
        else source_geometry_coordinates
    )
    for group_index, group in enumerate(groups):
        first = group[0]
        name = f"{first.cell_kind}-composed-{group_index}"
        target_ids = np.asarray([record.global_id for record in group], dtype=np.int64)
        target_cells = np.asarray([record.vertices for record in group], dtype=np.int64)
        target_blocks.append(
            CellBlock(
                name,
                first.cell_kind,
                target_cells,
                global_ids=target_ids,
            )
        )
        if first.pyramid_vertex_family:
            elements[name] = lagrange_element("pyramid", 1)
            routes[name] = pyramid_geometry_offset + target_cells
        else:
            elements[name] = PerCellRestrictedCellGeometryElement(
                first.base_element,
                first.cell_kind,
                tuple(record.global_id for record in group),
                tuple(record.source_root_cell_id for record in group),
                tuple(record.target_chart for record in group),
            )
            routes[name] = np.stack(
                [np.asarray(record.source_route, dtype=np.int64) for record in group]
            )
        parent_rows = np.asarray(
            [record.source_cell_row for record in group], dtype=np.int64
        )
        parent_cell_ids[name] = source_cell_ids[parent_rows]
        parent_vertex_ids[name] = np.asarray(
            [
                source_vertex_ids[source_cells[int(parent)]]
                for parent in parent_rows.tolist()
            ],
            dtype=np.int64,
        )
        source_block_names[name] = source.blocks[0].name
        flattened_records.extend(group)

    target_mesh = CellMesh(
        coordinates,
        tuple(target_blocks),
        vertex_global_ids=np.asarray(scaffold.vertex_global_ids, dtype=np.int64),
        numeric_version=source.numeric_version,
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source.topology_id,
        parent_cell_ids,
        parent_vertex_ids,
        block_source_blocks=source_block_names,
    )
    if any(record.pyramid_vertex_family for record in records):
        exact_vertex_image_bank = tuple(
            exact_vertex_images[row] for row in range(target_mesh.coordinates.shape[0])
        )
        restricted_map = _HexDominantCoordinateMap(
            source,
            source_geometry,
            target_mesh,
            exact_vertex_image_bank,
        )
    elif source_geometry.source_map is None:
        restricted_map = None
    else:
        restricted_map = _RestrictedCoordinateMap(
            source, source_geometry, target_mesh
        )
    geometry = CellGeometrySpec(
        elements,
        routes,
        pyramid_geometry_coordinates,
        source_map=restricted_map,
        source_mesh=target_mesh if restricted_map is not None else None,
        restriction_source=restriction,
    )

    target_connectivity = target_mesh.connectivity
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise ValueError("Mixed-family output requires explicit polyhedral face connectivity.")
    target_face_offsets = np.asarray(
        target_connectivity.face_vertex_offsets, dtype=np.int64
    )
    target_face_vertices = np.asarray(
        target_connectivity.face_vertex_values, dtype=np.int64
    )
    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    target_face_ids = np.asarray(target_mesh.entity_set(2).entity_ids, dtype=np.int64)
    target_face_lookup: dict[tuple[int, ...], int] = {}
    for row, (start, stop) in enumerate(
        zip(target_face_offsets[:-1], target_face_offsets[1:], strict=True)
    ):
        vertices = tuple(
            int(value)
            for value in target_face_vertices[int(start) : int(stop)].tolist()
        )
        target_face_lookup[tuple(sorted(vertices))] = int(target_face_ids[row])
    records_by_id = {record.global_id: record for record in flattened_records}
    face_records: dict[
        tuple[int, ...],
        list[tuple[_ComposedCell, tuple[Point3, ...]]],
    ] = {}
    for block in target_mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        for cell_row, global_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            record = records_by_id[int(global_id)]
            for local_face in topology.entities[2]:
                cycle = tuple(record.vertices[int(index)] for index in local_face)
                canonical = _canonical_face_cycle(cycle, target_vertex_ids)
                reference_by_vertex: dict[int, Point3] = {
                    record.vertices[local]: _fraction_point3(
                        topology.vertices[local]
                    )
                    for local in range(len(record.vertices))
                }
                face_records.setdefault(tuple(sorted(canonical)), []).append(
                    (
                        record,
                        tuple(reference_by_vertex[vertex] for vertex in canonical),
                    )
                )
    for face_key, adjacent in face_records.items():
        if len(adjacent) > 2:
            raise ValueError(
                f"Composed hex-dominant topology is nonmanifold at face ID "
                f"{target_face_lookup[face_key]}."
            )
        if len(adjacent) != 2:
            continue
        left_record, left_points = adjacent[0]
        right_record, right_points = adjacent[1]
        left_map = tuple(
            compose_expression(expression, _face_chart(left_points))
            for expression in left_record.physical_chart
        )
        right_map = tuple(
            compose_expression(expression, _face_chart(right_points))
            for expression in right_record.physical_chart
        )
        if any(
            not expression_equal(left, right)
            for left, right in zip(left_map, right_map, strict=True)
        ):
            raise ValueError(
                f"Exact mixed-family coordinate restrictions disagree on output "
                f"face ID {target_face_lookup[face_key]}."
            )

    flattened_records = [
        records_by_id[int(identifier)]
        for block in target_mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist()
    ]
    parent_rows = np.asarray(
        [record.source_cell_row for record in flattened_records], dtype=np.int64
    )
    policy = CellValidityPolicy(
        relative_determinant_floor=0.0,
        relative_planarity_tolerance=source_validity.policy.relative_planarity_tolerance,
    )
    validity = certify_cell_geometry_validity(
        geometry,
        mesh=target_mesh,
        policy=policy,
    )
    status = np.asarray(validity.status)
    invalid_rows = tuple(
        int(row)
        for row in np.flatnonzero(status != int(CellValidityStatus.CERTIFIED_VALID))
    )
    if invalid_rows:
        invalid_output_ids = tuple(
            flattened_records[row].global_id for row in invalid_rows
        )
        invalid_root_ids = tuple(
            int(source_cell_ids[flattened_records[row].source_cell_row])
            for row in invalid_rows
        )
        invalid_statuses = tuple(int(status[row]) for row in invalid_rows)
        raise ValueError(
            "Hex-dominant geometry lacks canonical positive whole-cell validity "
            f"for output cell IDs {invalid_output_ids}, root source cell IDs "
            f"{invalid_root_ids}, statuses {invalid_statuses}."
        )
    parents = parent_rows
    chart_bounds = tuple(record.chart_bounds for record in flattened_records)
    return (
        target_mesh,
        geometry,
        validity,
        parents,
        chart_bounds,
        tuple(source_reference_coverage),
    )
@checked
def extract_hex_dominant(
    source: CellMesh,
    core_cell_ids: ConvertibleToArray,
    /,
    *,
    source_geometry: CellGeometrySpec | None = None,
) -> HexDominantConstruction:
    """Close selected affine tetrahedral hex cores with conforming mixed templates.

    Each selected tetrahedron contributes its vertex-star dual hexes. Unselected
    tetrahedra are coned from their cell center to a shared face partition: fully
    split faces receive three pyramid bases, partially split faces receive a
    triangular fan, and untouched faces remain triangles. Edge midpoint marks are
    global to the source complex, so every core interface is face-conforming.
    """
    if (
        source.topological_dimension != 3
        or source.ambient_dimension != 3
        or not source.blocks
        or len(source.blocks) != 1
        or not _is_tetrahedron_block(source.blocks[0])
    ):
        raise ValueError("Hex-dominant closure requires one three-dimensional tetrahedron block.")
    if source_geometry is None:
        source_geometry = CellGeometrySpec.affine(source)
    elif not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec or None.")
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    if source_validity.invalid_count or source_validity.unresolved_count:
        raise ValueError("Hex-dominant source tetrahedra must be certified positive.")
    _require_tetrahedral_global_embedding(source, source_geometry, source_validity)

    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    source_cell_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64)
    requested = np.asarray(core_cell_ids)
    if requested.ndim != 1 or requested.dtype.kind not in "iu":
        raise TypeError("core_cell_ids must be an integer vector of source tetrahedron IDs.")
    core_ids = np.asarray(requested, dtype=np.int64)
    if not core_ids.size or np.unique(core_ids).size != core_ids.size:
        raise ValueError("Hex-dominant core IDs must be a nonempty unique source-cell selection.")
    row_by_id = {int(identifier): row for row, identifier in enumerate(source_cell_ids.tolist())}
    if any(int(identifier) not in row_by_id for identifier in core_ids.tolist()):
        raise ValueError("Hex-dominant core IDs must name source tetrahedra.")
    core_rows = np.asarray([row_by_id[int(identifier)] for identifier in core_ids], dtype=np.int64)
    if core_rows.size == source_cells.shape[0]:
        raise ValueError("An all-hex request must use a genuine all-hex generation route.")
    selected = np.zeros((source_cells.shape[0],), dtype=np.bool_)
    selected[core_rows] = True

    dual = extract_volume_hexes(source)
    dual_cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in dual.mesh.blocks]
    )
    dual_parents = np.asarray(dual.parent_cells, dtype=np.int64)
    core_mask = selected[dual_parents]
    core_hexes = dual_cells[core_mask]
    core_parent_rows = dual_parents[core_mask]
    coordinates, source_expression_tuple, exact_vertex_images = _exact_dual_coordinates(
        source,
        source_geometry,
        dual.mesh,
        np.asarray(dual.vertex_supports, dtype=np.int64),
    )
    edge_rows, face_rows, face_incidence = _tetrahedral_mesh_incidence(source)
    edge_lookup: dict[tuple[int, int], int] = {
        _edge_key2(int(edge[0]), int(edge[1])): row
        for row, edge in enumerate(edge_rows.tolist())
    }
    face_lookup: dict[tuple[int, int, int], int] = {
        _face_key3((int(face[0]), int(face[1]), int(face[2]))): row
        for row, face in enumerate(face_rows.tolist())
    }
    if np.any(face_incidence > 2):
        raise ValueError("Hex-dominant transitions require a manifold tetrahedral face complex.")

    original = source.coordinates.shape[0]
    edge_count = edge_rows.shape[0]
    face_count = face_rows.shape[0]
    marked_edges: set[int] = set()
    for row in core_rows.tolist():
        cell = source_cells[row]
        for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            marked_edges.add(
                edge_lookup[_edge_key2(int(cell[left]), int(cell[right]))]
            )

    face_pieces: list[tuple[tuple[int, ...], ...]] = []
    for face_row, face_values in enumerate(face_rows.tolist()):
        face = tuple(int(value) for value in face_values)
        face_edges = (
            edge_lookup[_edge_key2(face[0], face[1])],
            edge_lookup[_edge_key2(face[1], face[2])],
            edge_lookup[_edge_key2(face[2], face[0])],
        )
        split = tuple(edge in marked_edges for edge in face_edges)
        center = original + edge_count + face_row
        if all(split):
            midpoints = tuple(original + edge for edge in face_edges)
            face_pieces.append(
                tuple(
                    (face[vertex], midpoints[vertex], center, midpoints[(vertex + 2) % 3])
                    for vertex in range(3)
                )
            )
        elif any(split):
            ring: list[int] = []
            for vertex in range(3):
                ring.append(face[vertex])
                if split[vertex]:
                    ring.append(original + face_edges[vertex])
            face_pieces.append(
                tuple(
                    (ring[index], ring[(index + 1) % len(ring)], center)
                    for index in range(len(ring))
                )
            )
        else:
            face_pieces.append((face,))

    pyramid_rows: list[tuple[int, int, int, int, int]] = []
    pyramid_parents: list[int] = []
    tetrahedron_rows: list[tuple[int, int, int, int]] = []
    tetrahedron_parents: list[int] = []
    reference_pyramid = reference_cell_topology("pyramid")
    ref_points = np.asarray(reference_pyramid.vertices, dtype=np.float64)
    reference_sign = _orient3d_sign(ref_points, (0, 1, 2, 4))
    for cell_row, cell_values in enumerate(source_cells.tolist()):
        if selected[cell_row]:
            continue
        cell = tuple(int(value) for value in cell_values)
        center = original + edge_count + face_count + cell_row
        local_faces = (
            (cell[0], cell[1], cell[2]),
            (cell[0], cell[1], cell[3]),
            (cell[0], cell[2], cell[3]),
            (cell[1], cell[2], cell[3]),
        )
        for local_face in local_faces:
            face_row = face_lookup[
                _face_key3((local_face[0], local_face[1], local_face[2]))
            ]
            for piece in face_pieces[face_row]:
                if len(piece) == 4:
                    pyramid_rows.append(
                        _pyramid_orientation(
                            coordinates, tuple(piece), center, reference_sign
                        )
                    )
                    pyramid_parents.append(cell_row)
                elif len(piece) == 3:
                    tetrahedron = (*piece, center)
                    sign = _orient3d_sign(coordinates, tetrahedron)
                    if sign == 0:
                        raise ValueError("A hex-dominant tetrahedron transition is degenerate.")
                    if sign < 0:
                        tetrahedron = (tetrahedron[1], tetrahedron[0], tetrahedron[2], tetrahedron[3])
                    tetrahedron_rows.append(tetrahedron)
                    tetrahedron_parents.append(cell_row)
                else:
                    raise AssertionError("Transition face templates contain triangles or quads only.")

    cell_specs: list[tuple[str, tuple[int, ...], int, int]] = []
    next_cell_id = 0
    for vertices, parent in zip(core_hexes.tolist(), core_parent_rows.tolist(), strict=True):
        cell_specs.append(
            ("hexahedron", tuple(int(value) for value in vertices), int(parent), next_cell_id)
        )
        next_cell_id += 1
    for vertices, parent in zip(pyramid_rows, pyramid_parents, strict=True):
        cell_specs.append(
            ("pyramid", tuple(int(value) for value in vertices), int(parent), next_cell_id)
        )
        next_cell_id += 1
    for vertices, parent in zip(tetrahedron_rows, tetrahedron_parents, strict=True):
        cell_specs.append(
            ("tetrahedron", tuple(int(value) for value in vertices), int(parent), next_cell_id)
        )
        next_cell_id += 1
    (
        mesh,
        geometry,
        validity,
        parents,
        chart_bounds,
        source_reference_coverage,
    ) = _compose_mixed_geometry(
        source,
        source_geometry,
        source_validity,
        dual.mesh,
        np.asarray(dual.vertex_supports, dtype=np.int64),
        coordinates,
        source_expression_tuple,
        exact_vertex_images,
        tuple(cell_specs),
    )
    global_embedding = _require_global_embedding(
        mesh, geometry, validity, "Hex-dominant mixed-family output"
    )
    parent_offsets = np.arange(parents.size + 1, dtype=np.int64)
    parent_ids = source_cell_ids[parents]
    source_geometry_identity = cell_geometry_id(source_geometry)
    construction_id = canonical_fingerprint(
        {
            "kind": "native-hex-dominant-transition-construction",
            "source_topology": source.topology_id,
            "source_geometry": source_geometry_identity,
            "source_validity": source_validity.certificate_id,
            "reference_chart_determinant_bounds": tuple(
                tuple((value.numerator, value.denominator) for value in bounds)
                for bounds in chart_bounds
            ),
            "source_reference_coverage": tuple(
                (value.numerator, value.denominator)
                for value in source_reference_coverage
            ),
            "core_cell_ids": np.sort(core_ids),
            "mesh": mesh.mesh_id,
            "parent_cells": parents,
            "parent_cell_ids": parent_ids,
            "global_embedding": global_embedding.certificate_id,
        }
    )
    return HexDominantConstruction(
        mesh,
        geometry,
        validity,
        global_embedding,
        np.asarray(dual.vertex_supports, dtype=np.int64),
        core_ids,
        parent_offsets,
        parents,
        parent_ids,
        core_hexes.shape[0],
        len(pyramid_rows),
        len(tetrahedron_rows),
        source,
        source_geometry_identity,
        source_validity.certificate_id,
        chart_bounds,
        source_reference_coverage,
        construction_id,
    )


__all__ = ["HexDominantConstruction", "extract_hex_dominant"]
