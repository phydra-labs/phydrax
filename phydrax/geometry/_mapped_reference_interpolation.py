#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous metric bounds for mapped-root source charts."""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from typing import cast, TYPE_CHECKING

import numpy as np
from jax import Array

from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_mesh import CellMesh, PolyhedralBlock
from ..discretization._coordinate_enclosure import (
    compose_expression,
    coordinate_map_expressions,
    CoordinateSourceBank,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
    ExpressionInput,
    multiply,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    RestrictedCoordinateMapPreparation,
    subtract,
)
from ..discretization._exact_algebraic_geometry import RadicalElement
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._embedding_state import MeshCertificateFinding
from ._mapped_reference_boundary import MappedReferenceBoundarySource
from ._rational import fraction_endpoint
from ._source_boundary_records import SourceBoundaryChartWitness


if TYPE_CHECKING:
    from ._affine_plc_boundary import AffinePlcBoundarySource
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

Point2 = tuple[Fraction, Fraction]
Triangle = tuple[Point2, Point2, Point2]
LinePoints = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
_Point3Exact = tuple[ExactScalar, ExactScalar, ExactScalar]


def _source_coordinate_bank(
    values: tuple[tuple[object, ...], ...], /
) -> CoordinateSourceBank:
    if any(
        not isinstance(value, (Fraction, RadicalElement))
        for row in values
        for value in row
    ):
        raise TypeError("Source coordinate rows must retain exact scalar coefficients.")
    return cast(CoordinateSourceBank, values)


def _coordinate_expression(value: ExpressionInput, /) -> Expression:
    if isinstance(value, RationalPolynomial):
        return value
    return {monomial: coefficient for monomial, coefficient in value.items()}


def _triangle_points(values: tuple[tuple[Fraction, ...], ...], /) -> Triangle:
    if len(values) != 3 or any(len(point) != 2 for point in values):
        raise ValueError(
            "A mapped surface chart must retain three exact two-dimensional points."
        )
    return (
        (values[0][0], values[0][1]),
        (values[1][0], values[1][1]),
        (values[2][0], values[2][1]),
    )


def _line_points(values: tuple[tuple[Fraction, ...], ...], /) -> LinePoints:
    if len(values) != 2 or len(values[0]) != len(values[1]):
        raise ValueError(
            "A mapped curve chart must retain two same-dimensional exact points."
        )
    return values[0], values[1]


StandardTriangle: Triangle = (
    (Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)


@dataclass(frozen=True, slots=True)
class SurfaceChartFidelity:
    """Continuous source/target bounds in canonical witness row order."""

    cell_global_ids: tuple[int, ...]
    chart_rows: tuple[int, ...]
    occurrence_paths: tuple[tuple[str, ...], ...]
    deviation_bounds: np.ndarray
    mesh_to_source_lower: float
    mesh_to_source_upper: float
    source_to_mesh_lower: float
    source_to_mesh_upper: float
    findings: tuple[MeshCertificateFinding, ...]
    normal_deviation_bounds: np.ndarray | None
    maximum_normal_deviation_lower: float | None


def _vector_bounds(
    expressions: tuple[Expression, ...], triangle: Triangle, /
) -> tuple[Fraction, Fraction] | None:
    arguments: tuple[Expression, ...] = _chart_arguments(triangle)
    component_lowers: list[Fraction] = []
    component_uppers: list[Fraction] = []
    for expression in expressions:
        restricted = compose_expression(expression, arguments)
        low, high = polynomial_bounds(restricted, "simplex", 2)
        if not math.isfinite(low) or not math.isfinite(high):
            return None
        lower = Fraction.from_float(float(low))
        upper = Fraction.from_float(float(high))
        if lower > upper:
            return None
        component_lowers.append(
            Fraction(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
        )
        component_uppers.append(max(abs(lower), abs(upper)))
    return (
        fraction_sqrt_interval(
            sum((value * value for value in component_lowers), Fraction(0))
        )[0],
        fraction_sqrt_interval(
            sum((value * value for value in component_uppers), Fraction(0))
        )[1],
    )


def _exact_point_distance_bounds(
    expressions: tuple[Expression, ...], point: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    components = tuple(
        evaluate_expression(expression, point) for expression in expressions
    )
    lows: list[Fraction] = []
    highs: list[Fraction] = []
    for component in components:
        low = (
            Fraction.from_float(float(component))
            if isinstance(component, Fraction)
            else None
        )
        if low is not None:
            lower = upper = low
        else:
            from ..discretization._exact_algebraic_geometry import coefficient_endpoint

            lower = Fraction(coefficient_endpoint(component, upper=False))
            upper = Fraction(coefficient_endpoint(component, upper=True))
        lows.append(Fraction(0) if lower <= 0 <= upper else min(abs(lower), abs(upper)))
        highs.append(max(abs(lower), abs(upper)))
    return (
        fraction_sqrt_interval(sum((value * value for value in lows), Fraction(0)))[0],
        fraction_sqrt_interval(sum((value * value for value in highs), Fraction(0)))[1],
    )


def _affine(expressions: tuple[Expression, ...], /) -> bool:
    return all(
        isinstance(expression, dict) and all(sum(power) <= 1 for power in expression)
        for expression in expressions
    )


def _split_triangle(triangle: Triangle, /) -> tuple[Triangle, Triangle] | None:
    best_edge: tuple[int, int, int, Point2] | None = None
    best_length = Fraction(-1)
    edges: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for first, last, third in edges:
        a, b = triangle[first], triangle[last]
        middle: Point2 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        rounded = tuple(float(value) for value in middle)
        if rounded == tuple(float(value) for value in a) or rounded == tuple(
            float(value) for value in b
        ):
            continue
        length = sum(((b[axis] - a[axis]) ** 2 for axis in range(2)), Fraction(0))
        if length > best_length:
            best_edge = (first, last, third, middle)
            best_length = length
    if best_edge is None:
        return None
    first, last, third, middle = best_edge
    return (
        (triangle[first], middle, triangle[third]),
        (middle, triangle[last], triangle[third]),
    )


def _supremum_bounds(
    expressions: tuple[Expression, ...],
    tolerance: Fraction,
    /,
) -> tuple[Fraction, Fraction] | None:
    if all(
        not expression.numerator
        if isinstance(expression, RationalPolynomial)
        else not expression
        for expression in expressions
    ):
        return Fraction(0), Fraction(0)
    if _affine(expressions):
        values = tuple(
            _exact_point_distance_bounds(expressions, point) for point in StandardTriangle
        )
        return max(lower for lower, _ in values), max(upper for _, upper in values)
    initial = _vector_bounds(expressions, StandardTriangle)
    if initial is None:
        return None
    queue: list[tuple[Fraction, int, Triangle, Fraction]] = [
        (-initial[1], 0, StandardTriangle, initial[0])
    ]
    serial = 1
    maximum_lower = initial[0]
    while queue:
        maximum_upper = -queue[0][0]
        if maximum_lower > tolerance or maximum_upper <= tolerance:
            return maximum_lower, maximum_upper
        _, _, triangle, _ = heapq.heappop(queue)
        children = _split_triangle(triangle)
        if children is None:
            return maximum_lower, maximum_upper
        for child in children:
            bounds = _vector_bounds(expressions, child)
            if bounds is None:
                return None
            maximum_lower = max(maximum_lower, bounds[0])
            heapq.heappush(queue, (-bounds[1], serial, child, bounds[0]))
            serial += 1
    return maximum_lower, Fraction(0)


def _affine_coordinates(
    corners: tuple[
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
    ],
    /,
) -> tuple[Polynomial, Polynomial, Polynomial]:
    zero_power: tuple[int, ...] = (0, 0)
    first_power: tuple[int, ...] = (1, 0)
    second_power: tuple[int, ...] = (0, 1)
    return (
        {
            zero_power: corners[0][0],
            first_power: corners[1][0] - corners[0][0],
            second_power: corners[2][0] - corners[0][0],
        },
        {
            zero_power: corners[0][1],
            first_power: corners[1][1] - corners[0][1],
            second_power: corners[2][1] - corners[0][1],
        },
        {
            zero_power: corners[0][2],
            first_power: corners[1][2] - corners[0][2],
            second_power: corners[2][2] - corners[0][2],
        },
    )


def _chart_arguments(
    points: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Polynomial, ...]:
    if (
        len(points) != 3
        or not points[0]
        or any(len(point) != len(points[0]) for point in points)
    ):
        raise ValueError("A triangle chart requires three equal-dimensional points.")
    zero_power: tuple[int, ...] = (0, 0)
    first_power: tuple[int, ...] = (1, 0)
    second_power: tuple[int, ...] = (0, 1)
    return tuple(
        {
            zero_power: points[0][axis],
            first_power: points[1][axis] - points[0][axis],
            second_power: points[2][axis] - points[0][axis],
        }
        for axis in range(len(points[0]))
    )


def _line_chart_arguments(points: LinePoints, /) -> tuple[Expression, ...]:
    if len(points[0]) != len(points[1]):
        raise ValueError("Exact line-chart endpoints must have equal dimensions.")
    constant_power: tuple[int, ...] = (0,)
    first_power: tuple[int, ...] = (1,)
    return tuple(
        {
            constant_power: points[0][axis],
            first_power: points[1][axis] - points[0][axis],
        }
        for axis in range(len(points[0]))
    )


def _mapped_chart_maps(
    mesh: CellMesh,
    witness: SourceBoundaryChartWitness,
    row: int,
    target_row: tuple[int, int],
    source_base: tuple[Expression, ...],
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    coordinates: CoordinateSourceBank,
    target_vertex_coordinates: dict[int, _Point3Exact] | None,
    target_maps: OrderedDict[tuple[int, int], tuple[Expression, ...]],
    root_preparations: dict[int, RestrictedCoordinateMapPreparation],
    /,
) -> tuple[tuple[Expression, ...], tuple[Expression, ...]]:
    parameters = _triangle_points(witness.source_parameter_chart(row))
    source_arguments = _chart_arguments(parameters)
    source_map = tuple(
        compose_expression(expression, source_arguments) for expression in source_base
    )
    if len(source_map) != 3:
        raise ValueError(
            "Mapped root surface charts must evaluate in three physical dimensions."
        )

    block_index, cell_row = target_row
    if np.asarray(witness.target_face_vertex_global_ids).size:
        if not isinstance(mesh.blocks[block_index], PolyhedralBlock):
            raise TypeError(
                "Polyhedral target chart ancestry requires a polyhedron block."
            )
        if target_vertex_coordinates is None:
            raise ValueError(
                "Polyhedral target coordinates were not prepared for this chart bank."
            )
        vertex_ids = np.asarray(
            witness.target_face_vertex_global_ids[row], dtype=np.int64
        )
        if vertex_ids.shape != (3,):
            raise ValueError(
                "Polyhedral source chart must retain three target vertex IDs."
            )
        parent = (
            target_vertex_coordinates[int(vertex_ids[0])],
            target_vertex_coordinates[int(vertex_ids[1])],
            target_vertex_coordinates[int(vertex_ids[2])],
        )
        target_reference = _triangle_points(witness.target_reference_chart(row))
        target_corners: list[_Point3Exact] = []
        for u, v in target_reference:
            weights = (Fraction(1) - u - v, u, v)
            target_corners.append(
                (
                    sum(
                        (weights[index] * parent[index][0] for index in range(3)),
                        Fraction(0),
                    ),
                    sum(
                        (weights[index] * parent[index][1] for index in range(3)),
                        Fraction(0),
                    ),
                    sum(
                        (weights[index] * parent[index][2] for index in range(3)),
                        Fraction(0),
                    ),
                )
            )
        target_map = _affine_coordinates(
            (target_corners[0], target_corners[1], target_corners[2])
        )
        return source_map, target_map

    route = np.asarray(routes[block_index][cell_row], dtype=np.int64)
    local = tuple(coordinates[int(index)] for index in route)
    element = elements[block_index]
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError("Mapped target charts require scalar coordinate elements.")
    target_base = target_maps.get(target_row)
    if target_base is None:
        preparation = root_preparations.get(block_index)
        if preparation is None and isinstance(element, PerCellRestrictedCellGeometryElement):
            preparation = RestrictedCoordinateMapPreparation(
                element, coordinates, np.asarray(routes[block_index], dtype=np.int64),
                mode="source",
            )
            root_preparations[block_index] = preparation
        target_base = None if preparation is None else preparation.expressions(cell_row)
        if target_base is None:
            target_base = coordinate_map_expressions(element, local, cell_row=cell_row)
        target_maps[target_row] = target_base
        if len(target_maps) > 64:
            target_maps.popitem(last=False)
    else:
        target_maps.move_to_end(target_row)
    target_reference = witness.target_reference_chart(row)
    if witness.reference_dimension != mesh.topological_dimension or any(
        len(point) != witness.reference_dimension for point in target_reference
    ):
        raise ValueError("Target face charts must retain the parent reference dimension.")
    target_arguments = _chart_arguments(target_reference)
    target_map = tuple(
        compose_expression(expression, target_arguments) for expression in target_base
    )
    if len(target_map) != 3:
        raise ValueError(
            "Mapped target surface charts must evaluate in three physical dimensions."
        )
    return source_map, target_map


def _cross_expressions(
    expressions: tuple[Expression, ...], /
) -> tuple[Expression, Expression, Expression]:
    first = tuple(derivative(expressions[axis], 0) for axis in range(3))
    second = tuple(derivative(expressions[axis], 1) for axis in range(3))
    return (
        subtract(multiply(first[1], second[2]), multiply(first[2], second[1])),
        subtract(multiply(first[2], second[0]), multiply(first[0], second[2])),
        subtract(multiply(first[0], second[1]), multiply(first[1], second[0])),
    )


def _zero_expression(expression: Expression, /) -> bool:
    return (
        not expression.numerator
        if isinstance(expression, RationalPolynomial)
        else not expression
    )


def _normal_cross_interval(
    expressions: tuple[Expression, ...],
    triangle: Triangle,
    /,
) -> tuple[tuple[Fraction, Fraction], ...] | None:
    lower = np.empty((3, 2), dtype=np.float64)
    upper = np.empty((3, 2), dtype=np.float64)
    arguments = _chart_arguments(triangle)
    for axis, expression in enumerate(expressions):
        for parameter in range(2):
            restricted = compose_expression(derivative(expression, parameter), arguments)
            low, high = polynomial_bounds(restricted, "simplex", 2)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                return None
            lower[axis, parameter] = low
            upper[axis, parameter] = high
    from ._source_fidelity import _normal_cross_interval

    return _normal_cross_interval(lower, upper, 1)


def _normal_angle_bounds(
    source_map: tuple[Expression, ...],
    target_map: tuple[Expression, ...],
    triangle: Triangle,
    /,
) -> tuple[Fraction, Fraction] | None:
    from ..discretization._exact_algebraic_geometry import coefficient_endpoint
    from ._source_fidelity import (
        _acos_lower,
        _acos_upper,
        _normal_cosine_interval_bounds,
    )

    source_cross = _cross_expressions(source_map)
    target_cross = _cross_expressions(target_map)
    if all(_zero_expression(value) for value in source_cross):
        return None
    if all(_zero_expression(value) for value in target_cross):
        return None
    if all(
        _zero_expression(subtract(first, second))
        for first, second in zip(source_cross, target_cross, strict=True)
    ):
        source_interval = _normal_cross_interval(source_map, triangle)
        target_interval = _normal_cross_interval(target_map, triangle)
        if (
            source_interval is not None
            and target_interval is not None
            and _normal_cosine_interval_bounds(source_interval, target_interval)
            is not None
        ):
            return Fraction(0), Fraction(0)
    negative_target = tuple(
        multiply(value, {(0, 0): Fraction(-1)}) for value in target_cross
    )
    if all(
        _zero_expression(subtract(first, negative))
        for first, negative in zip(source_cross, negative_target, strict=True)
    ):
        source_interval = _normal_cross_interval(source_map, triangle)
        target_interval = _normal_cross_interval(target_map, triangle)
        if (
            source_interval is not None
            and target_interval is not None
            and _normal_cosine_interval_bounds(source_interval, target_interval)
            is not None
        ):
            return _acos_lower(Fraction(-1)), _acos_upper(Fraction(-1))

    if _affine(source_map) and _affine(target_map):
        point = (Fraction(0), Fraction(0))
        source_derivatives: tuple[
            tuple[ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar],
        ] = (
            (
                evaluate_expression(derivative(source_map[0], 0), point),
                evaluate_expression(derivative(source_map[0], 1), point),
            ),
            (
                evaluate_expression(derivative(source_map[1], 0), point),
                evaluate_expression(derivative(source_map[1], 1), point),
            ),
            (
                evaluate_expression(derivative(source_map[2], 0), point),
                evaluate_expression(derivative(source_map[2], 1), point),
            ),
        )
        target_derivatives: tuple[
            tuple[ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar],
        ] = (
            (
                evaluate_expression(derivative(target_map[0], 0), point),
                evaluate_expression(derivative(target_map[0], 1), point),
            ),
            (
                evaluate_expression(derivative(target_map[1], 0), point),
                evaluate_expression(derivative(target_map[1], 1), point),
            ),
            (
                evaluate_expression(derivative(target_map[2], 0), point),
                evaluate_expression(derivative(target_map[2], 1), point),
            ),
        )

        def exact_cross(
            values: tuple[
                tuple[ExactScalar, ExactScalar],
                tuple[ExactScalar, ExactScalar],
                tuple[ExactScalar, ExactScalar],
            ],
        ) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
            first = (values[0][0], values[1][0], values[2][0])
            second = (values[0][1], values[1][1], values[2][1])
            return (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )

        source_exact = exact_cross(source_derivatives)
        target_exact = exact_cross(target_derivatives)
        source_interval = tuple(
            (
                Fraction(coefficient_endpoint(value, upper=False)),
                Fraction(coefficient_endpoint(value, upper=True)),
            )
            for value in source_exact
        )
        cosine = _normal_cosine_interval_bounds(
            source_interval,
            tuple(
                (
                    Fraction(coefficient_endpoint(value, upper=False)),
                    Fraction(coefficient_endpoint(value, upper=True)),
                )
                for value in target_exact
            ),
        )
        if cosine is None:
            return None
        return _acos_lower(cosine[1]), _acos_upper(cosine[0])

    source_interval = _normal_cross_interval(source_map, triangle)
    target_interval = _normal_cross_interval(target_map, triangle)
    if source_interval is None or target_interval is None:
        return None
    cosine = _normal_cosine_interval_bounds(source_interval, target_interval)
    if cosine is None:
        return None
    return _acos_lower(cosine[1]), _acos_upper(cosine[0])


def _normal_bounds(
    source_map: tuple[Expression, ...],
    target_map: tuple[Expression, ...],
    maximum_deviation: float,
    /,
) -> tuple[Fraction, Fraction] | None:
    from ._source_fidelity import _acos_upper

    limit = Fraction.from_float(float(maximum_deviation))
    initial = _normal_angle_bounds(source_map, target_map, StandardTriangle)
    initial_valid = initial is not None
    if initial is None:
        if _affine(source_map) and _affine(target_map):
            return None
        initial = (Fraction(0), _acos_upper(Fraction(-1)))
    pieces: list[tuple[Triangle, Fraction, Fraction, bool]] = [
        (StandardTriangle, initial[0], initial[1], initial_valid)
    ]
    while True:
        lower_bound = max((piece[1] for piece in pieces), default=Fraction(0))
        upper_bound = max((piece[2] for piece in pieces), default=Fraction(0))
        if lower_bound > limit:
            return lower_bound, upper_bound
        if all(piece[3] for piece in pieces) and upper_bound <= limit:
            return lower_bound, upper_bound
        index = max(range(len(pieces)), key=lambda row: pieces[row][2])
        triangle, _, _, _ = pieces[index]
        children = _split_triangle(triangle)
        if children is None:
            return None
        child_values: list[tuple[Triangle, Fraction, Fraction, bool]] = []
        for child in children:
            bounds = _normal_angle_bounds(source_map, target_map, child)
            if bounds is None:
                child_values.append(
                    (child, Fraction(0), _acos_upper(Fraction(-1)), False)
                )
            else:
                child_values.append((child, bounds[0], bounds[1], True))
        pieces[index : index + 1] = child_values


def certify_surface_chart_maps(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MappedReferenceBoundarySource
    | PiecewiseLinearBoundarySource
    | AffinePlcBoundarySource,
    witness: SourceBoundaryChartWitness,
    target_rows: tuple[tuple[int, int], ...],
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SurfaceChartFidelity:
    """Bound both directed errors through exact source and target chart maps."""
    from ._affine_plc_boundary import AffinePlcBoundarySource
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

    if len(target_rows) != len(witness.source_chart_ids):
        raise ValueError("Mapped source and target chart rows must align.")
    elements, routes, _ = geometry.resolve(mesh)
    coordinates = _source_coordinate_bank(geometry.source_coordinates())
    polyhedral = np.asarray(witness.target_face_vertex_global_ids).size != 0
    if polyhedral:
        from ._source_fidelity import _polyhedral_vertex_source_coordinates

        target_vertex_coordinates = _polyhedral_vertex_source_coordinates(mesh, geometry)
    else:
        target_vertex_coordinates = None
    tolerance_fraction = Fraction.from_float(float(tolerance))
    row_count = len(target_rows)
    deviations = np.full((row_count, 2), (0.0, math.inf), dtype=np.float64)
    normal_bounds = (
        None
        if maximum_normal_deviation is None
        else np.full((row_count,), math.inf, dtype=np.float64)
    )
    findings: list[MeshCertificateFinding] = []
    lower_max = Fraction(0)
    upper_max: Fraction | float = Fraction(0)
    normal_lower_max = Fraction(0)
    target_ids = tuple(int(value) for value in witness.entity_global_ids)
    source_rows: list[int] = []
    source_map_cache: dict[str, tuple[Expression, ...]] = {}
    # This bounded scheduling window is local to these exact geometry operands;
    # eviction only repeats the same composition and does not affect the proof.
    target_maps: OrderedDict[tuple[int, int], tuple[Expression, ...]] = OrderedDict()
    root_preparations: dict[int, RestrictedCoordinateMapPreparation] = {}
    for row, target_row in enumerate(target_rows):
        chart_row = source.chart_row(2, witness.source_chart_ids[row])
        source_rows.append(
            int(
                source.source_rows[chart_row]
                if isinstance(
                    source, (PiecewiseLinearBoundarySource, AffinePlcBoundarySource)
                )
                else source.polygon_source_facet_rows[chart_row]
            )
        )
        try:
            chart_id = witness.source_chart_ids[row]
            source_base = source_map_cache.get(chart_id)
            if source_base is None:
                source_base = tuple(
                    _coordinate_expression(value)
                    for value in source.chart_coordinate_expressions(2, chart_id)
                )
                source_map_cache[chart_id] = source_base
            source_map, target_map = _mapped_chart_maps(
                mesh,
                witness,
                row,
                target_row,
                source_base,
                elements,
                routes,
                coordinates,
                target_vertex_coordinates,
                target_maps,
                root_preparations,
            )
            difference = tuple(
                subtract(source_value, target_value)
                for source_value, target_value in zip(source_map, target_map, strict=True)
            )
            if len(difference) != 3:
                raise ValueError(
                    "Mapped source/target difference must be three-dimensional."
                )
            bounds = _supremum_bounds(difference, tolerance_fraction)
        except (
            ArithmeticError,
            FloatingPointError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            bounds = None
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "cell",
                    (target_ids[row],),
                    detail=f"Mapped root/target chart enclosure is unavailable: {error}",
                )
            )
        if bounds is None:
            upper_max = math.inf
            continue
        lower, upper = bounds
        lower_max = max(lower_max, lower)
        upper_max = max(upper_max, upper)
        deviations[row] = (
            fraction_endpoint(lower, upper=False, saturate=True),
            fraction_endpoint(upper, upper=True, saturate=True),
        )
        if lower > tolerance_fraction:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "violated",
                    "cell",
                    (target_ids[row],),
                    detail="A continuous mapped source/target distance lower bound exceeds tolerance.",
                )
            )
        elif upper > tolerance_fraction:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "unresolved",
                    "cell",
                    (target_ids[row],),
                    detail="A continuous mapped source/target distance upper bound exceeds tolerance.",
                )
            )
        if normal_bounds is not None and maximum_normal_deviation is not None:
            try:
                normal = _normal_bounds(
                    source_map, target_map, float(maximum_normal_deviation)
                )
            except (
                ArithmeticError,
                FloatingPointError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                normal = None
            if normal is None:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_range",
                        "unresolved",
                        "cell",
                        (target_ids[row],),
                        detail="Mapped source/target chart normals lack a finite continuous interval enclosure.",
                    )
                )
                continue
            normal_lower, normal_upper = normal
            normal_lower_max = max(normal_lower_max, normal_lower)
            normal_bounds[row] = fraction_endpoint(
                normal_upper, upper=True, saturate=True
            )
            normal_limit = Fraction.from_float(float(maximum_normal_deviation))
            if normal_lower > normal_limit:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_limit",
                        "violated",
                        "cell",
                        (target_ids[row],),
                        detail="A mapped oriented-normal lower bound exceeds the requested limit.",
                    )
                )
            elif normal_upper > normal_limit:
                findings.append(
                    MeshCertificateFinding(
                        "normal_deviation_limit",
                        "unresolved",
                        "cell",
                        (target_ids[row],),
                        detail="A mapped oriented-normal upper bound exceeds the requested limit.",
                    )
                )
    lower_float = fraction_endpoint(lower_max, upper=False, saturate=True)
    upper_float = (
        fraction_endpoint(upper_max, upper=True, saturate=True)
        if isinstance(upper_max, Fraction)
        else math.inf
    )
    return SurfaceChartFidelity(
        target_ids,
        tuple(source_rows),
        witness.occurrence_paths,
        deviations,
        lower_float,
        upper_float,
        lower_float,
        upper_float,
        tuple(findings),
        normal_bounds,
        None
        if maximum_normal_deviation is None
        else fraction_endpoint(normal_lower_max, upper=False, saturate=True),
    )


def _curve_vector_bounds(
    expressions: tuple[Expression, ...], first: Fraction, last: Fraction, /
) -> tuple[Fraction, Fraction] | None:
    component_lowers: list[Fraction] = []
    component_uppers: list[Fraction] = []
    argument = _line_chart_arguments(((first,), (last,)))
    for expression in expressions:
        restricted = compose_expression(expression, argument)
        low, high = polynomial_bounds(restricted, "box", 1)
        if not math.isfinite(low) or not math.isfinite(high):
            return None
        lower = Fraction.from_float(float(low))
        upper = Fraction.from_float(float(high))
        component_lowers.append(
            Fraction(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
        )
        component_uppers.append(max(abs(lower), abs(upper)))
    return (
        fraction_sqrt_interval(
            sum((value * value for value in component_lowers), Fraction(0))
        )[0],
        fraction_sqrt_interval(
            sum((value * value for value in component_uppers), Fraction(0))
        )[1],
    )


def _exact_curve_point_distance_bounds(
    expressions: tuple[Expression, ...], parameter: Fraction, /
) -> tuple[Fraction, Fraction]:
    from ..discretization._exact_algebraic_geometry import coefficient_endpoint

    lower_components: list[Fraction] = []
    upper_components: list[Fraction] = []
    for expression in expressions:
        value = evaluate_expression(expression, (parameter,))
        lower = Fraction(coefficient_endpoint(value, upper=False))
        upper = Fraction(coefficient_endpoint(value, upper=True))
        lower_components.append(
            Fraction(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
        )
        upper_components.append(max(abs(lower), abs(upper)))
    return (
        fraction_sqrt_interval(
            sum((value * value for value in lower_components), Fraction(0))
        )[0],
        fraction_sqrt_interval(
            sum((value * value for value in upper_components), Fraction(0))
        )[1],
    )


def _curve_supremum_bounds(
    expressions: tuple[Expression, ...], tolerance: Fraction, /
) -> tuple[Fraction, Fraction] | None:
    if all(
        not expression.numerator
        if isinstance(expression, RationalPolynomial)
        else not expression
        for expression in expressions
    ):
        return Fraction(0), Fraction(0)
    if _affine(expressions):
        values = tuple(
            _exact_curve_point_distance_bounds(expressions, point)
            for point in (Fraction(0), Fraction(1))
        )
        return max(lower for lower, _ in values), max(upper for _, upper in values)
    initial = _curve_vector_bounds(expressions, Fraction(0), Fraction(1))
    if initial is None:
        return None
    queue: list[tuple[Fraction, int, Fraction, Fraction]] = [
        (-initial[1], 0, Fraction(0), Fraction(1))
    ]
    serial = 1
    maximum_lower = initial[0]
    while queue:
        maximum_upper = -queue[0][0]
        if maximum_lower > tolerance or maximum_upper <= tolerance:
            return maximum_lower, maximum_upper
        _, _, first, last = heapq.heappop(queue)
        middle = (first + last) / 2
        if float(middle) in (float(first), float(last)):
            return maximum_lower, maximum_upper
        for lower_parameter, upper_parameter in ((first, middle), (middle, last)):
            bounds = _curve_vector_bounds(expressions, lower_parameter, upper_parameter)
            if bounds is None:
                return None
            maximum_lower = max(maximum_lower, bounds[0])
            heapq.heappush(queue, (-bounds[1], serial, lower_parameter, upper_parameter))
            serial += 1
    return maximum_lower, Fraction(0)


def certify_mapped_reference_curve_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MappedReferenceBoundarySource | PiecewiseLinearBoundarySource,
    witness: SourceBoundaryChartWitness,
    target_rows: tuple[tuple[int, int], ...],
    /,
    *,
    tolerance: float,
    maximum_normal_deviation: float | None,
) -> SurfaceChartFidelity:
    """Bound both directed errors on exact root and target curve charts."""
    from ..discretization._cell_mesh import CellBlock
    from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource

    if len(target_rows) != len(witness.source_chart_ids):
        raise ValueError("Mapped source and target curve rows must align.")
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = _source_coordinate_bank(geometry.source_coordinates())
    limit = Fraction.from_float(float(tolerance))
    count = len(target_rows)
    deviations = np.full((count, 2), (0.0, math.inf), dtype=np.float64)
    normal_bounds = (
        None if maximum_normal_deviation is None else np.full((count,), math.inf)
    )
    target_ids = tuple(int(value) for value in witness.entity_global_ids)
    source_rows: list[int] = []
    findings: list[MeshCertificateFinding] = []
    lower_max = Fraction(0)
    upper_max: Fraction | float = Fraction(0)
    source_map_cache: dict[str, tuple[Expression, ...]] = {}
    for row, (block_index, cell_row) in enumerate(target_rows):
        chart_row = source.chart_row(1, witness.source_chart_ids[row])
        source_rows.append(
            int(
                source.entity_ids[chart_row]
                if isinstance(source, PiecewiseLinearBoundarySource)
                else source.segment_source_entity_ids[chart_row]
            )
        )
        try:
            block = mesh.blocks[block_index]
            if not isinstance(block, CellBlock):
                raise TypeError(
                    "Mapped curve target geometry requires standard reference-cell maps."
                )
            raw_source_parameters = witness.source_parameter_chart(row)
            source_points = _line_points(raw_source_parameters)
            if len(source_points[0]) != 1:
                raise ValueError("Mapped root curve charts must be one-dimensional.")
            source_argument = _line_chart_arguments(
                ((source_points[0][0],), (source_points[1][0],))
            )
            chart_id = witness.source_chart_ids[row]
            source_base = source_map_cache.get(chart_id)
            if source_base is None:
                source_base = tuple(
                    _coordinate_expression(value)
                    for value in source.chart_coordinate_expressions(1, chart_id)
                )
                source_map_cache[chart_id] = source_base
            source_map = tuple(
                compose_expression(value, source_argument) for value in source_base
            )
            local = tuple(
                coefficients[int(index)]
                for index in np.asarray(routes[block_index][cell_row], dtype=np.int64)
            )
            element = elements[block_index]
            if not isinstance(element, ScalarCoordinateElement):
                raise TypeError(
                    "Mapped curve target requires a scalar coordinate element."
                )
            target_base = coordinate_map_expressions(element, local, cell_row=cell_row)
            target_reference = _line_points(witness.target_reference_chart(row))
            target_argument = _line_chart_arguments(target_reference)
            target_map = tuple(
                compose_expression(value, target_argument) for value in target_base
            )
            difference = tuple(
                subtract(source_value, target_value)
                for source_value, target_value in zip(source_map, target_map, strict=True)
            )
            if len(difference) != 3:
                raise ValueError(
                    "Mapped source/target curve maps must evaluate in three dimensions."
                )
            bounds = _curve_supremum_bounds(difference, limit)
        except (
            ArithmeticError,
            FloatingPointError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            bounds = None
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "cell",
                    (target_ids[row],),
                    detail=f"Mapped root/target curve enclosure is unavailable: {error}",
                )
            )
        if bounds is None:
            upper_max = math.inf
            continue
        lower, upper = bounds
        lower_max = max(lower_max, lower)
        upper_max = max(upper_max, upper)
        deviations[row] = (
            fraction_endpoint(lower, upper=False, saturate=True),
            fraction_endpoint(upper, upper=True, saturate=True),
        )
        if lower > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "violated",
                    "cell",
                    (target_ids[row],),
                    detail="A continuous mapped source/target curve distance lower bound exceeds tolerance.",
                )
            )
        elif upper > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "unresolved",
                    "cell",
                    (target_ids[row],),
                    detail="A continuous mapped source/target curve distance upper bound exceeds tolerance.",
                )
            )
    if normal_bounds is not None:
        findings.append(
            MeshCertificateFinding(
                "normal_deviation_range",
                "unresolved",
                "mesh",
                detail="A one-dimensional mapped source chart does not define an oriented surface normal.",
            )
        )
    lower_float = fraction_endpoint(lower_max, upper=False, saturate=True)
    upper_float = (
        fraction_endpoint(upper_max, upper=True, saturate=True)
        if isinstance(upper_max, Fraction)
        else math.inf
    )
    return SurfaceChartFidelity(
        target_ids,
        tuple(source_rows),
        witness.occurrence_paths,
        deviations,
        lower_float,
        upper_float,
        lower_float,
        upper_float,
        tuple(findings),
        normal_bounds,
        None,
    )


__all__ = [
    "SurfaceChartFidelity",
    "certify_mapped_reference_curve_charts",
    "certify_surface_chart_maps",
]
