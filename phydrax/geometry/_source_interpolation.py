#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous source-chart interpolation bounds for native fidelity proofs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from jax import Array

from .._interpolation._bspline_extraction import (
    PreparedSourceSpans,
    RationalBSplineSurfaceSpan,
)
from ..discretization._cell_geometry import CellGeometrySpec, ScalarCoordinateElement
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._coordinate_enclosure import (
    add,
    compose_expression,
    coordinate_map_expressions,
    evaluate_expression,
    ExactScalar,
    Expression,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    scale,
    subtract,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    coefficient_interval,
)
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._curve_boundary import ParametricCurveBoundarySource, SegmentChartCurve
from ._domain_boundary import MeshingDomainBoundarySource
from ._embedding_state import MeshCertificateFinding
from ._interval_enclosure import IntervalExtensionError, PreparedIntervalFunction
from ._meshing_domain import SourcePreparation
from ._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from ._rational import fraction_endpoint, rational_vector
from ._source_boundary_records import SourceBoundaryChartWitness


if TYPE_CHECKING:
    from ._meshing_domain import _PlacedSurfacePatch
    from .brep._patches import RevolutionSurface


def _has_analytic_source_program(surface: object, /) -> bool:
    """Whether a direct or placed analytic carrier has a source-bound program."""
    from ._meshing_domain import _PlacedSurfacePatch
    from .brep._patches import (
        CircleCurve,
        CylinderPatch,
        LineCurve,
        RevolutionSurface,
        SpherePatch,
        TorusPatch,
    )

    native = surface.source if isinstance(surface, _PlacedSurfacePatch) else surface
    return isinstance(native, (CylinderPatch, SpherePatch, TorusPatch)) or (
        isinstance(native, RevolutionSurface)
        and isinstance(native.curve, (LineCurve, CircleCurve))
    )


def _revolution_axis_split(
    revolution_axis: tuple[ExactScalar, ...], vector: tuple[ExactScalar, ...], /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...], tuple[ExactScalar, ...]]:
    """Axial, radial and axis-cross parts of ``vector`` about a retained axis."""
    projection = sum((vector[i] * revolution_axis[i] for i in range(3)), Fraction(0))
    axial = tuple(projection * revolution_axis[i] for i in range(3))
    radial = tuple(vector[i] - axial[i] for i in range(3))
    cross = tuple(
        revolution_axis[(i + 1) % 3] * radial[(i + 2) % 3]
        - revolution_axis[(i + 2) % 3] * radial[(i + 1) % 3]
        for i in range(3)
    )
    return axial, radial, cross


def _revolution_axis(
    native: RevolutionSurface, /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]]:
    # These are the retained source coefficients. The stored normalized
    # binary axis need not have exactly unit squared length.
    return (
        rational_vector(np.asarray(native.axis_origin, dtype=np.float64)),
        rational_vector(np.asarray(native.axis_direction, dtype=np.float64)),
    )


def _line_revolution_source_vectors(
    native: RevolutionSurface, /
) -> tuple[tuple[ExactScalar, ...], ...]:
    """Exact ``(P0, ..., P5)`` of a line revolution's unplaced source map.

    ``S(u, v) = P0 + v P1 + cos(u) (P2 + v P3) + sin(u) (P4 + v P5)``.
    """
    from .brep._patches import _require_exact_numeric_shadow, LineCurve

    curve = native.curve
    if not isinstance(curve, LineCurve):
        raise TypeError("A line revolution requires a LineCurve profile.")
    _require_exact_numeric_shadow(curve.origin, curve.origin_exact, "Line origin")
    _require_exact_numeric_shadow(
        curve.direction, curve.direction_exact, "Line direction"
    )
    axis_origin, revolution_axis = _revolution_axis(native)
    relative = tuple(curve.origin_exact[i] - axis_origin[i] for i in range(3))
    axial, radial, cross = _revolution_axis_split(revolution_axis, relative)
    axial_rate, radial_rate, cross_rate = _revolution_axis_split(
        revolution_axis, curve.direction_exact
    )
    return (
        tuple(axis_origin[i] + axial[i] for i in range(3)),
        axial_rate,
        radial,
        radial_rate,
        cross,
        cross_rate,
    )


def _placed_source_vectors(
    surface: _PlacedSurfacePatch,
    vectors: tuple[tuple[ExactScalar, ...], ...],
    /,
) -> tuple[tuple[ExactScalar, ...], ...]:
    """Apply an exact placement to a point (first) and direction vectors."""
    from .brep._patches import _require_exact_numeric_shadow

    matrix = surface.placement.matrix
    linear = tuple(tuple(matrix[i][j] for j in range(3)) for i in range(3))
    offset = tuple(matrix[i][3] for i in range(3))
    _require_exact_numeric_shadow(surface.linear, linear, "Placed source linear")
    _require_exact_numeric_shadow(surface.offset, offset, "Placed source offset")
    return tuple(
        tuple(
            sum(
                (linear[i][j] * vector[j] for j in range(3)),
                offset[i] if index == 0 else Fraction(0),
            )
            for i in range(3)
        )
        for index, vector in enumerate(vectors)
    )


def exact_line_revolution_vectors(
    surface: object, /
) -> tuple[tuple[ExactScalar, ...], ...] | None:
    """Exact harmonic vectors of a direct or placed line revolution, else ``None``.

    ``S(u, v) = P0 + v P1 + cos(u) (P2 + v P3) + sin(u) (P4 + v P5)`` in
    the carrier's physical frame.
    """
    from ._meshing_domain import _PlacedSurfacePatch
    from .brep._patches import LineCurve, RevolutionSurface

    native = surface.source if isinstance(surface, _PlacedSurfacePatch) else surface
    if not isinstance(native, RevolutionSurface) or not isinstance(
        native.curve, LineCurve
    ):
        return None
    vectors = _line_revolution_source_vectors(native)
    if isinstance(surface, _PlacedSurfacePatch):
        return _placed_source_vectors(surface, vectors)
    return vectors


def _prepare_analytic_source_program(
    surface: object, order: int, /
) -> PreparedIntervalFunction | None:
    """Prepare exact-coefficient point or derivative bounds for analytic carriers.

    The numeric coefficient vector is only a JAX execution shadow. Its captured
    interval constants enclose the authored exact scalars, including nonbinary
    radii, so point and derivative bounds are for the source, not the shadow.
    """
    import jax
    import jax.numpy as jnp

    from ._interval_enclosure import prepare_interval_function
    from ._meshing_domain import _PlacedSurfacePatch
    from .brep._patches import (
        _require_exact_numeric_shadow,
        _validate_surface_source_shadows,
        CircleCurve,
        CylinderPatch,
        LineCurve,
        RevolutionSurface,
        SpherePatch,
        TorusPatch,
    )

    native = surface.source if isinstance(surface, _PlacedSurfacePatch) else surface
    line_revolution = isinstance(native, RevolutionSurface) and isinstance(
        native.curve, LineCurve
    )
    circle_revolution = isinstance(native, RevolutionSurface) and isinstance(
        native.curve, CircleCurve
    )
    if not isinstance(native, (CylinderPatch, SpherePatch, TorusPatch)) and not (
        line_revolution or circle_revolution
    ):
        return None
    if order not in (0, 1, 2):
        raise ValueError("Analytic source interval derivative order must be 0, 1, or 2.")
    radii: tuple[ExactScalar, ...] = ()
    if line_revolution:
        vectors = _line_revolution_source_vectors(native)
    elif circle_revolution:
        curve = native.curve
        if not isinstance(curve, CircleCurve):
            raise RuntimeError("Analytic source classification is inconsistent.")
        _require_exact_numeric_shadow(curve.center, curve.center_exact, "Circle center")
        _require_exact_numeric_shadow(
            curve.first_axis, curve.first_axis_exact, "Circle first axis"
        )
        _require_exact_numeric_shadow(
            curve.second_axis, curve.second_axis_exact, "Circle second axis"
        )
        _require_exact_numeric_shadow(curve.radius, curve.radius_exact, "Circle radius")
        axis_origin, revolution_axis = _revolution_axis(native)
        relative = tuple(curve.center_exact[i] - axis_origin[i] for i in range(3))
        components = (
            _revolution_axis_split(revolution_axis, relative),
            _revolution_axis_split(
                revolution_axis,
                tuple(curve.radius_exact * value for value in curve.first_axis_exact),
            ),
            _revolution_axis_split(
                revolution_axis,
                tuple(curve.radius_exact * value for value in curve.second_axis_exact),
            ),
        )
        vectors = (
            tuple(axis_origin[i] + components[0][0][i] for i in range(3)),
            components[1][0],
            components[2][0],
            components[0][1],
            components[1][1],
            components[2][1],
            components[0][2],
            components[1][2],
            components[2][2],
        )
    else:
        match native:
            case CylinderPatch():
                origin = native.origin_exact
                radii = (native.radius_exact,)
            case TorusPatch():
                origin = native.center_exact
                radii = (native.major_radius_exact, native.minor_radius_exact)
            case SpherePatch():
                origin = rational_vector(np.asarray(native.center, dtype=np.float64))
                radii = (Fraction.from_float(float(native.radius)),)
            case _:
                raise RuntimeError("Analytic source classification is inconsistent.")
        if isinstance(native, SpherePatch):
            vectors = (
                origin,
                rational_vector(np.asarray(native.first_axis, dtype=np.float64)),
                rational_vector(np.asarray(native.second_axis, dtype=np.float64)),
                rational_vector(np.asarray(native.axis, dtype=np.float64)),
            )
        else:
            vectors = (
                origin,
                native.first_axis_exact,
                native.second_axis_exact,
                native.axis_exact,
            )
        _validate_surface_source_shadows(native)
    if isinstance(surface, _PlacedSurfacePatch):
        vectors = _placed_source_vectors(surface, vectors)
    exact = (*sum(vectors, ()), *radii)
    nominal = np.asarray(tuple(float(value) for value in exact), dtype=np.float64)
    intervals = tuple(coefficient_interval(value, bits=128) for value in exact)
    lower = np.asarray(
        tuple(fraction_endpoint(value[0], upper=False) for value in intervals),
        dtype=np.float64,
    )
    upper = np.asarray(
        tuple(fraction_endpoint(value[1], upper=True) for value in intervals),
        dtype=np.float64,
    )
    coefficients = jnp.asarray(nominal)

    def evaluate(uv: Array) -> Array:
        longitude, latitude = uv[0], uv[1]
        if line_revolution:
            return (
                coefficients[:3]
                + latitude * coefficients[3:6]
                + (coefficients[6:9] + latitude * coefficients[9:12]) * jnp.cos(longitude)
                + (coefficients[12:15] + latitude * coefficients[15:18])
                * jnp.sin(longitude)
            )
        if circle_revolution:
            cos_lat, sin_lat = jnp.cos(latitude), jnp.sin(latitude)
            axial = (
                coefficients[:3]
                + cos_lat * coefficients[3:6]
                + sin_lat * coefficients[6:9]
            )
            radial = (
                coefficients[9:12]
                + cos_lat * coefficients[12:15]
                + sin_lat * coefficients[15:18]
            )
            cross = (
                coefficients[18:21]
                + cos_lat * coefficients[21:24]
                + sin_lat * coefficients[24:27]
            )
            return axial + jnp.cos(longitude) * radial + jnp.sin(longitude) * cross
        radial = (
            jnp.cos(longitude) * coefficients[3:6]
            + jnp.sin(longitude) * coefficients[6:9]
        )
        if isinstance(native, CylinderPatch):
            return (
                coefficients[:3]
                + coefficients[12] * radial
                + latitude * coefficients[9:12]
            )
        if isinstance(native, SpherePatch):
            return coefficients[:3] + coefficients[12] * (
                jnp.cos(latitude) * radial + jnp.sin(latitude) * coefficients[9:12]
            )
        return (
            coefficients[:3]
            + (coefficients[12] + coefficients[13] * jnp.cos(latitude)) * radial
            + coefficients[13] * jnp.sin(latitude) * coefficients[9:12]
        )

    derivative: Callable[[Array], Array] = evaluate
    for _ in range(order):
        derivative = jax.jacfwd(derivative)
    return prepare_interval_function(
        derivative, 2, constant_bounds=((nominal, lower, upper),)
    )


def _exact_rational_span_interpolation_bounds(
    spans: PreparedSourceSpans[RationalBSplineSurfaceSpan],
    charts: tuple[tuple[tuple[Fraction, Fraction], ...], ...],
    /,
) -> np.ndarray:
    from ._meshing_domain import _exact_spline_interpolation_bounds

    return _exact_spline_interpolation_bounds(
        np.asarray(charts, dtype=np.object_), spans, charts
    )


def _exact_rational_surface_point_bounds(
    spans: PreparedSourceSpans[RationalBSplineSurfaceSpan],
    chart: tuple[Fraction, Fraction],
    /,
) -> tuple[tuple[Fraction, Fraction], ...] | None:
    u, v = chart
    enclosures = []
    for span in spans.spans:
        (u_lower, u_upper), (v_lower, v_upper) = span.parameter_bounds
        if not (u_lower <= u <= u_upper and v_lower <= v <= v_upper):
            continue
        enclosures.append(
            tuple(coefficient_interval(value) for value in span.evaluate_exact(chart))
        )
    if not enclosures:
        return None
    return tuple(
        (
            min(enclosure[component][0] for enclosure in enclosures),
            max(enclosure[component][1] for enclosure in enclosures),
        )
        for component in range(len(enclosures[0]))
    )


def _exact_rational_surface_chart_breakpoints(
    spans: PreparedSourceSpans[RationalBSplineSurfaceSpan],
    origin: tuple[Fraction, Fraction],
    direction: tuple[Fraction, Fraction],
    first: Fraction,
    last: Fraction,
    /,
) -> tuple[Fraction, ...] | None:
    values = {first, last}
    for span in spans.spans:
        for axis in range(2):
            if direction[axis] == 0:
                continue
            lower, upper = span.parameter_bounds[axis]
            if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
                return None
            for boundary in (lower, upper):
                parameter = (boundary - origin[axis]) / direction[axis]
                if first < parameter < last:
                    values.add(parameter)
    return tuple(sorted(values))


def _fraction_float_interval(value: Fraction, /) -> tuple[float, float]:
    represented = float(value)
    if not math.isfinite(represented):
        raise ValueError("Exact chart coordinate exceeds finite interval resolution.")
    rounded = Fraction.from_float(represented)
    lower = (
        float(np.nextafter(represented, -math.inf)) if rounded > value else represented
    )
    upper = float(np.nextafter(represented, math.inf)) if rounded < value else represented
    return lower, upper


@dataclass(frozen=True, slots=True)
class SourceChartFidelity:
    """Computed curve-chart deviation bounds in witness row order."""

    cell_global_ids: tuple[int, ...]
    chart_rows: tuple[int, ...]
    occurrence_paths: tuple[tuple[str, ...], ...]
    deviation_bounds: np.ndarray
    mesh_to_source_lower: float
    mesh_to_source_upper: float
    source_to_mesh_lower: float
    source_to_mesh_upper: float
    findings: tuple[MeshCertificateFinding, ...]
    target_corners: tuple[tuple[tuple[ExactScalar, ...], ...], ...] = ()
    target_affine: tuple[bool, ...] = ()


def _exact_line_data(
    source: ParametricCurveBoundarySource, chart_row: int, /
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]] | None:
    from .brep._source_correspondence import _affine_curve_coefficients

    curve = source.curves[chart_row]
    if isinstance(curve, SegmentChartCurve):
        return curve.origin_exact, curve.direction_exact
    return _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension)


def _transform_exact_surface_map(
    expressions: tuple[Polynomial, ...],
    matrix: tuple[tuple[ExactScalar, ...], ...],
    /,
) -> tuple[Polynomial, ...]:
    def transformed_row(row: int, /) -> Polynomial:
        result: Polynomial = {(0, 0): matrix[row][3]}
        for column in range(3):
            result = add(result, scale(expressions[column], matrix[row][column]))
        return result

    return tuple(transformed_row(row) for row in range(3))


def exact_surface_coordinate_expressions(
    surface: object, /
) -> tuple[Polynomial, ...] | None:
    """Return a source-owned exact map for supported affine/bilinear carriers."""
    from ._meshing_domain import _PlacedSurfacePatch
    from .brep._patches import OffsetSurface, PlanePatch, RuledSurface
    from .brep._placed import PlacedSurface
    from .reconstruction._core import _ReconstructedTrianglePatch

    if isinstance(surface, _PlacedSurfacePatch):
        source_map = exact_surface_coordinate_expressions(surface.source)
        if source_map is None:
            return None
        return _transform_exact_surface_map(source_map, surface.placement.matrix)
    if isinstance(surface, PlacedSurface):
        source_map = exact_surface_coordinate_expressions(surface.definition)
        if source_map is None:
            return None
        return _transform_exact_surface_map(source_map, surface.affine.matrix)
    if isinstance(surface, OffsetSurface):
        base_map = exact_surface_coordinate_expressions(surface.base)
        if base_map is None or not _exact_surface_map_is_affine(base_map):
            return None
        first = tuple(expression.get((1, 0), Fraction(0)) for expression in base_map)
        second = tuple(expression.get((0, 1), Fraction(0)) for expression in base_map)
        from ..discretization._exact_algebraic_geometry import (
            RadicalElement,
            rational_coefficient,
        )
        from ._rational import rational_cross

        normal = rational_cross(first, second)
        squared_length = sum((value * value for value in normal), Fraction(0))
        if squared_length == 0:
            return None
        try:
            length = RadicalElement.sqrt(
                rational_coefficient(
                    squared_length, role="offset surface normal squared norm"
                )
            )
        except ValueError:
            return None
        orientation = surface.frame_sign if surface.frame_sign != 0 else 1
        distance = surface.distance_exact * orientation
        return tuple(
            add(expression, {(0, 0): distance * normal[axis] / length})
            for axis, expression in enumerate(base_map)
        )
    if isinstance(surface, _ReconstructedTrianglePatch):
        origin, first, second = surface.exact_corners
        return tuple(
            {
                (0, 0): origin[axis],
                (1, 0): first[axis] - origin[axis],
                (0, 1): second[axis] - origin[axis],
            }
            for axis in range(3)
        )
    if isinstance(surface, PlanePatch):
        origin = rational_vector(np.asarray(surface.origin))
        first = rational_vector(np.asarray(surface.first_axis))
        second = rational_vector(np.asarray(surface.second_axis))
        return tuple(
            {
                (0, 0): origin[axis],
                (1, 0): first[axis],
                (0, 1): second[axis],
            }
            for axis in range(3)
        )
    if not isinstance(surface, RuledSurface):
        return None
    from .brep._source_correspondence import _affine_curve_coefficients

    origin = (Fraction(0), Fraction(0), Fraction(0))
    first = _affine_curve_coefficients(surface.first, origin)
    second = _affine_curve_coefficients(surface.second, origin)
    if first is None or second is None:
        return None
    first_origin, first_direction = first
    second_origin, second_direction = second
    return tuple(
        {
            (0, 0): first_origin[axis],
            (1, 0): first_direction[axis],
            (0, 1): second_origin[axis] - first_origin[axis],
            (1, 1): second_direction[axis] - first_direction[axis],
        }
        for axis in range(3)
    )


def _exact_surface_map_is_affine(expressions: tuple[Polynomial, ...], /) -> bool:
    return all(
        all(
            sum(power) <= 1 or coefficient == 0
            for power, coefficient in expression.items()
        )
        for expression in expressions
    )


def exact_affine_boundary_source(query: MeshingDomainBoundarySource, /) -> bool:
    """Prove that every selected exact-represented source chart is affine."""
    if not isinstance(query, MeshingDomainBoundarySource):
        raise TypeError(
            "Exact affine source proof requires a MeshingDomainBoundarySource."
        )
    query.require_current()
    if query.domain.accuracy != "exact_represented" or not query.patches:
        return False
    for patch in query.patches:
        expressions = exact_surface_coordinate_expressions(
            query.domain.patches[patch].surface
        )
        if expressions is None or not _exact_surface_map_is_affine(expressions):
            return False
    return True


def _exact_surface_interpolation_bound(
    expressions: tuple[Polynomial, ...],
    chart: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> float:
    arguments: tuple[Polynomial, ...] = tuple(
        {
            (0, 0): chart[0][axis],
            (1, 0): chart[1][axis] - chart[0][axis],
            (0, 1): chart[2][axis] - chart[0][axis],
        }
        for axis in range(2)
    )
    mapped = tuple(compose_expression(value, arguments) for value in expressions)
    standard = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    corners = tuple(
        tuple(evaluate_expression(value, point) for value in mapped) for point in standard
    )
    residuals: list[Expression] = []
    for axis, expression in enumerate(mapped):
        affine: Polynomial = {
            (0, 0): corners[0][axis],
            (1, 0): corners[1][axis] - corners[0][axis],
            (0, 1): corners[2][axis] - corners[0][axis],
        }
        residuals.append(subtract(expression, affine))
    half = Fraction(1, 2)
    subtriangles = (
        (
            (Fraction(0), Fraction(0)),
            (half, Fraction(0)),
            (Fraction(0), half),
        ),
        (
            (half, Fraction(0)),
            (Fraction(1), Fraction(0)),
            (half, half),
        ),
        (
            (Fraction(0), half),
            (half, half),
            (Fraction(0), Fraction(1)),
        ),
        (
            (half, Fraction(0)),
            (half, half),
            (Fraction(0), half),
        ),
    )
    maximum_upper = Fraction(0)
    for triangle in subtriangles:
        sub_arguments: tuple[Polynomial, ...] = tuple(
            {
                (0, 0): triangle[0][axis],
                (1, 0): triangle[1][axis] - triangle[0][axis],
                (0, 1): triangle[2][axis] - triangle[0][axis],
            }
            for axis in range(2)
        )
        component_errors: list[Fraction] = []
        for residual in residuals:
            sub_residual = compose_expression(residual, sub_arguments)
            lower, upper = polynomial_bounds(sub_residual, "simplex", 2)
            if not math.isfinite(lower) or not math.isfinite(upper):
                return math.inf
            component_errors.append(
                max(abs(Fraction.from_float(lower)), abs(Fraction.from_float(upper)))
            )
        maximum_upper = max(
            maximum_upper,
            fraction_sqrt_interval(
                sum((value * value for value in component_errors), Fraction(0))
            )[1],
        )
    return fraction_endpoint(maximum_upper, upper=True, saturate=True)


def _has_unexpanded_exact_surface_carrier(surface: object, /) -> bool:
    from ._meshing_domain import _PlacedSurfacePatch
    from .brep._intersection_curve import CurveTrimSegment
    from .brep._patches import (
        ExtrusionSurface,
        OffsetSurface,
        RevolutionSurface,
        RuledSurface,
        SurfaceMappedCurve,
    )
    from .brep._placed import PlacedCurve, PlacedSurface
    from .brep._root_bindings import _source_coefficients_are_binary

    def matrix_has_nonbinary(matrix: tuple[tuple[ExactScalar, ...], ...], /) -> bool:
        for row in matrix:
            for value in row:
                try:
                    if Fraction.from_float(float(value)) != value:
                        return True
                except (OverflowError, TypeError, ValueError):
                    return True
        return False

    def curve_has_unexpanded_exact_carrier(curve: object, /) -> bool:
        if isinstance(curve, PlacedCurve):
            return matrix_has_nonbinary(curve.affine.matrix) or (
                curve_has_unexpanded_exact_carrier(curve.definition)
            )
        if isinstance(curve, CurveTrimSegment):
            return curve_has_unexpanded_exact_carrier(curve.curve)
        if isinstance(curve, SurfaceMappedCurve):
            return _has_unexpanded_exact_surface_carrier(curve.surface)
        return not _source_coefficients_are_binary(curve)

    if isinstance(surface, _PlacedSurfacePatch):
        return matrix_has_nonbinary(surface.placement.matrix) or (
            _has_unexpanded_exact_surface_carrier(surface.source)
        )
    if isinstance(surface, PlacedSurface):
        return matrix_has_nonbinary(surface.affine.matrix) or (
            _has_unexpanded_exact_surface_carrier(surface.definition)
        )
    if isinstance(surface, OffsetSurface):
        return not _source_coefficients_are_binary(surface) or (
            _has_unexpanded_exact_surface_carrier(surface.base)
        )
    if isinstance(surface, (ExtrusionSurface, RevolutionSurface)):
        return not _source_coefficients_are_binary(surface) or (
            curve_has_unexpanded_exact_carrier(surface.curve)
        )
    if isinstance(surface, RuledSurface):
        return curve_has_unexpanded_exact_carrier(
            surface.first
        ) or curve_has_unexpanded_exact_carrier(surface.second)
    return not _source_coefficients_are_binary(surface)


def _outward_parameter(value: Fraction, *, lower: bool) -> float:
    result = float(value)
    if not math.isfinite(result):
        return result
    represented = Fraction.from_float(result)
    if lower and represented > value:
        return float(np.nextafter(result, -math.inf))
    if not lower and represented < value:
        return float(np.nextafter(result, math.inf))
    return result


def _coordinate_interval(
    expression: Polynomial | RationalPolynomial,
    /,
) -> tuple[float, float]:
    return polynomial_bounds(expression, "box", 1)


def _target_subinterval_expressions(
    expressions: tuple[Polynomial | RationalPolynomial, ...],
    target_first: tuple[Fraction, ...],
    target_last: tuple[Fraction, ...],
    first: Fraction,
    last: Fraction,
    /,
) -> tuple[Polynomial | RationalPolynomial, ...]:
    arguments: tuple[Polynomial, ...] = tuple(
        {
            (0,): lower + (upper - lower) * first,
            (1,): (upper - lower) * (last - first),
        }
        for lower, upper in zip(target_first, target_last, strict=True)
    )
    return tuple(compose_expression(value, arguments) for value in expressions)


def _line_difference_expressions(
    line: tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]],
    first: Fraction,
    last: Fraction,
    target: tuple[Polynomial | RationalPolynomial, ...],
    /,
) -> tuple[Polynomial | RationalPolynomial, ...]:
    origin, direction = line
    source: tuple[Polynomial, ...] = tuple(
        {
            (0,): point + slope * first,
            (1,): slope * (last - first),
        }
        for point, slope in zip(origin, direction, strict=True)
    )
    return tuple(
        subtract(source_value, target_value)
        for source_value, target_value in zip(source, target, strict=True)
    )


def _line_endpoint_bracket_padding(
    source: ParametricCurveBoundarySource,
    chart_row: int,
    first: Fraction,
    last: Fraction,
    direction: tuple[ExactScalar, ...],
    /,
) -> Fraction:
    """Bound an uncertain source endpoint's exact line displacement."""
    low, high = sorted((first, last))
    parameter_error = max(
        (
            max(
                abs(Fraction.from_float(lower) - shadow),
                abs(Fraction.from_float(upper) - shadow),
            )
            for shadow_float, (lower, upper) in zip(
                source.parameter_ranges[chart_row],
                source.endpoint_parameter_bounds[chart_row],
                strict=True,
            )
            for shadow in (Fraction.from_float(shadow_float),)
            if lower != upper
            and low <= Fraction.from_float(upper)
            and high >= Fraction.from_float(lower)
        ),
        default=Fraction(0),
    )
    if not parameter_error:
        return Fraction(0)
    squared = sum(
        (
            max(
                abs(Fraction(coefficient_endpoint(value, upper=False))),
                abs(Fraction(coefficient_endpoint(value, upper=True))),
            )
            ** 2
            for value in direction
        ),
        Fraction(0),
    )
    return parameter_error * fraction_sqrt_interval(squared)[1]


def _distance_bounds(
    intervals: tuple[tuple[float, float], ...], /
) -> tuple[Fraction, Fraction | float]:
    if any(not math.isfinite(value) for interval in intervals for value in interval):
        return Fraction(0), math.inf
    component_lowers: list[Fraction] = []
    component_uppers: list[Fraction] = []
    for lower, upper in intervals:
        low, high = Fraction.from_float(lower), Fraction.from_float(upper)
        component_lowers.append(
            Fraction(0) if low <= 0 <= high else min(abs(low), abs(high))
        )
        component_uppers.append(max(abs(low), abs(high)))
    lower = fraction_sqrt_interval(
        sum((value * value for value in component_lowers), Fraction(0))
    )[0]
    upper = fraction_sqrt_interval(
        sum((value * value for value in component_uppers), Fraction(0))
    )[1]
    return lower, upper


def _polynomial_distance_bounds(
    expressions: tuple[Polynomial | RationalPolynomial, ...], /
) -> tuple[Fraction, Fraction | float]:
    if all(
        not expression.numerator
        if isinstance(expression, RationalPolynomial)
        else not expression
        for expression in expressions
    ):
        return Fraction(0), Fraction(0)
    intervals = tuple(_coordinate_interval(expression) for expression in expressions)
    return _distance_bounds(intervals)


def _interval_difference_bounds(
    target: tuple[tuple[float, float], ...],
    source: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[Fraction, Fraction | float]:
    source_lower, source_upper = source
    if (
        np.any(~np.isfinite(source_lower))
        or np.any(~np.isfinite(source_upper))
        or any(not math.isfinite(value) for pair in target for value in pair)
    ):
        return Fraction(0), math.inf
    differences = tuple(
        (
            target_lower - float(source_high),
            target_upper - float(source_low),
        )
        for (target_lower, target_upper), source_low, source_high in zip(
            target, source_lower, source_upper, strict=True
        )
    )
    return _distance_bounds(differences)


def _curve_chart_partition(
    source: ParametricCurveBoundarySource,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[int, ...]:
    chart_rows = {chart_id: row for row, chart_id in enumerate(source.chart_ids)}
    if len(chart_rows) != len(source.chart_ids):
        raise ValueError("Curve source chart identities are not unique.")
    grouped: dict[int, list[tuple[Fraction, Fraction]]] = {}
    rows: list[int] = []
    for witness_row, (chart_id, entity_id, occurrence) in enumerate(
        zip(
            witness.source_chart_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
    ):
        parameters = witness.source_parameter_chart(witness_row)
        row = chart_rows.get(chart_id)
        if row is None:
            raise ValueError(
                "Curve chart witness names a source chart absent from the query."
            )
        if (
            source.curve_entity_ids[row] != entity_id
            or source.occurrence_paths[row] != occurrence
        ):
            raise ValueError(
                "Curve chart witness changes its source entity or occurrence path."
            )
        first, last = parameters[0][0], parameters[1][0]
        if first == last:
            raise ValueError("Curve chart witness contains a degenerate source interval.")
        grouped.setdefault(row, []).append((min(first, last), max(first, last)))
        rows.append(row)
    if set(grouped) != set(range(len(source.curves))):
        raise ValueError("Curve chart witness omits or adds a source chart.")
    for row, intervals in grouped.items():
        source_first, source_last = (
            Fraction.from_float(float(value)) for value in source.parameter_ranges[row]
        )
        expected = (min(source_first, source_last), max(source_first, source_last))
        ordered = sorted(intervals)
        if ordered[0][0] != expected[0]:
            raise ValueError(
                "Curve chart witness does not start at the authored source endpoint."
            )
        endpoint = expected[0]
        for first, last in ordered:
            if first != endpoint:
                raise ValueError(
                    "Curve chart intervals have a gap, overlap, or reversed occurrence."
                )
            endpoint = last
        if endpoint != expected[1]:
            raise ValueError(
                "Curve chart witness omits part of an authored source interval."
            )
    return tuple(rows)


def certify_parametric_curve_chart(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: ParametricCurveBoundarySource,
    witness: SourceBoundaryChartWitness,
    target_rows: tuple[tuple[int, int], ...],
    tolerance: float,
    /,
) -> SourceChartFidelity:
    """Bound both directed deviations on witnessed target edges."""
    if mesh.ambient_dimension != source.ambient_dimension:
        raise ValueError("Curve chart fidelity requires the source physical dimension.")
    if (
        witness.mesh_id != mesh.mesh_id
        or witness.topology_id != mesh.topology_id
        or witness.geometry_id != cell_geometry_id(geometry)
        or witness.source_id != source.source_id
        or witness.source_revision != source.source_revision
        or witness.source_query_id != source.query_id
        or witness.chart_dimension != 1
        or witness.reference_dimension != mesh.topological_dimension
        or len(target_rows) != len(witness.source_chart_ids)
    ):
        raise ValueError(
            "Curve chart witness is bound to a different source, mesh, or map."
        )
    source.require_current()
    chart_rows = _curve_chart_partition(source, witness)
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    references = tuple(
        witness.target_reference_chart(row)
        for row in range(len(witness.source_chart_ids))
    )
    source_parameters = tuple(
        witness.source_parameter_chart(row)
        for row in range(len(witness.source_chart_ids))
    )
    parent_cells = tuple(int(value) for value in witness.cell_global_ids)
    target_entities = tuple(int(value) for value in witness.entity_global_ids)
    deviations = np.zeros((len(target_entities), 2), dtype=np.float64)
    upper_max: Fraction | float = Fraction(0)
    findings: list[MeshCertificateFinding] = []
    limit = Fraction.from_float(float(tolerance))
    for witness_row, (parent_cell, target_entity, chart_row) in enumerate(
        zip(parent_cells, target_entities, chart_rows, strict=True)
    ):
        block_index, cell_row = target_rows[witness_row]

        target_reference = references[witness_row]
        if (
            len(target_reference[0]) != mesh.topological_dimension
            or target_reference[0] == target_reference[1]
        ):
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "cell",
                    (target_entity,),
                    detail="The witnessed target edge has a degenerate parent-reference map.",
                )
            )
            deviations[witness_row, 1] = math.inf
            continue
        local = tuple(
            coefficients[int(index)]
            for index in np.asarray(routes[block_index][cell_row])
        )
        try:
            element = elements[block_index]
            if not isinstance(element, ScalarCoordinateElement):
                raise TypeError("Target curve map requires a scalar coordinate element.")
            target_expressions = coordinate_map_expressions(
                element, local, cell_row=cell_row
            )
            if len(target_expressions) != source.ambient_dimension:
                raise ValueError(
                    "Target coordinate dimension differs from its curve source."
                )
        except (TypeError, ValueError, IndexError) as error:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "cell",
                    (target_entity,),
                    detail=f"Target curve coordinate map is unavailable: {error}",
                )
            )
            deviations[witness_row, 1] = math.inf
            continue
        target_first, target_last = target_reference
        parameter_first, parameter_last = (
            source_parameters[witness_row][0][0],
            source_parameters[witness_row][1][0],
        )
        exact_line = _exact_line_data(source, chart_row)
        pending = [(Fraction(0), Fraction(1))]
        row_upper: Fraction | float = Fraction(0)
        row_unresolved = False
        while pending:
            first, last = pending.pop()
            target = _target_subinterval_expressions(
                target_expressions, target_first, target_last, first, last
            )
            source_first = parameter_first + (parameter_last - parameter_first) * first
            source_last = parameter_first + (parameter_last - parameter_first) * last
            try:
                if exact_line is not None:
                    difference = _line_difference_expressions(
                        exact_line, source_first, source_last, target
                    )
                    lower, upper = _polynomial_distance_bounds(difference)
                    padding = _line_endpoint_bracket_padding(
                        source, chart_row, source_first, source_last, exact_line[1]
                    )
                    lower = max(Fraction(0), lower - padding)
                    upper += padding
                else:
                    source_lower_parameter = _outward_parameter(
                        min(source_first, source_last), lower=True
                    )
                    source_upper_parameter = _outward_parameter(
                        max(source_first, source_last), lower=False
                    )
                    source_bounds = source.chart_interval_bounds(
                        source.chart_ids[chart_row],
                        source_lower_parameter,
                        source_upper_parameter,
                    )
                    target_bounds = tuple(_coordinate_interval(value) for value in target)
                    lower, upper = _interval_difference_bounds(
                        target_bounds, source_bounds
                    )
            except (ArithmeticError, FloatingPointError, TypeError, ValueError):
                lower, upper = Fraction(0), math.inf
            if not math.isfinite(float(upper)):
                row_upper = math.inf
                row_unresolved = True
                break
            if upper <= limit:
                row_upper = max(row_upper, upper)
                continue
            if lower > limit:
                row_unresolved = True
                row_upper = math.inf
                break
            middle = (first + last) / 2
            first_parameter = parameter_first + (parameter_last - parameter_first) * first
            middle_parameter = (
                parameter_first + (parameter_last - parameter_first) * middle
            )
            last_parameter = parameter_first + (parameter_last - parameter_first) * last
            first_float = float(first_parameter)
            middle_float = float(middle_parameter)
            last_float = float(last_parameter)
            if (
                not min(first_float, last_float)
                < middle_float
                < max(first_float, last_float)
            ):
                row_unresolved = True
                row_upper = math.inf
                break
            pending.extend(((middle, last), (first, middle)))
        upper_bound = (
            fraction_endpoint(row_upper, upper=True, saturate=True)
            if isinstance(row_upper, Fraction)
            else math.inf
        )
        deviations[witness_row] = (0.0, upper_bound)
        upper_max = max(upper_max, row_upper)
        if row_unresolved:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_distance_unresolved",
                    "unresolved",
                    "cell",
                    (target_entity,),
                    detail="Continuous chart distance bounds did not resolve the requested tolerance.",
                )
            )
        elif row_upper > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "unresolved",
                    "cell",
                    (target_entity,),
                    detail="A continuous source/target chart distance upper bound exceeds tolerance.",
                )
            )
    lower_float = 0.0
    upper_float = (
        fraction_endpoint(upper_max, upper=True, saturate=True)
        if isinstance(upper_max, Fraction)
        else math.inf
    )
    return SourceChartFidelity(
        target_entities,
        chart_rows,
        witness.occurrence_paths,
        deviations,
        lower_float,
        upper_float,
        lower_float,
        upper_float,
        tuple(findings),
    )


def certify_piecewise_linear_curve_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: PiecewiseLinearBoundarySource,
    witness: SourceBoundaryChartWitness,
    target_rows: tuple[tuple[int, int], ...],
    tolerance: float,
    /,
) -> SourceChartFidelity:
    """Continuously bound exact PWL source segments against witnessed target edges."""
    if (
        mesh.ambient_dimension != source.ambient_dimension
        or witness.chart_dimension != 1
        or witness.reference_dimension != mesh.topological_dimension
        or len(target_rows) != len(witness.source_chart_ids)
    ):
        raise ValueError(
            "PWL curve chart witness changes the target or source dimension."
        )
    source.require_current()
    source.validate_chart_witness(witness)
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    cell_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    target_id_values = np.asarray(witness.entity_global_ids, dtype=np.int64)
    if cell_ids.shape != (len(target_rows),) or target_id_values.shape != (
        len(target_rows),
    ):
        raise ValueError("PWL curve target identity rows do not align with the witness.")
    target_entities = tuple(
        int(target_id_values[row]) for row in range(target_id_values.size)
    )
    chart_rows = tuple(
        source.chart_row(1, witness.source_chart_ids[row])
        for row in range(len(witness.source_chart_ids))
    )
    references = tuple(
        witness.target_reference_chart(row)
        for row in range(len(witness.source_chart_ids))
    )
    parameters = tuple(
        witness.source_parameter_chart(row)
        for row in range(len(witness.source_chart_ids))
    )
    deviations = np.zeros((len(target_rows), 2), dtype=np.float64)
    upper_max: Fraction | float = Fraction(0)
    findings: list[MeshCertificateFinding] = []
    limit = Fraction.from_float(float(tolerance))
    for row in range(len(target_rows)):
        block_index, cell_row = target_rows[row]
        target_entity = target_entities[row]
        row_upper: Fraction | float = Fraction(0)
        row_unresolved = False
        try:
            block = mesh.blocks[block_index]
            if not isinstance(block, CellBlock):
                raise TypeError(
                    "PWL curve target maps require standard reference-cell blocks."
                )
            block_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
            if (
                cell_row < 0
                or cell_row >= block_cell_ids.size
                or int(block_cell_ids[cell_row]) != int(cell_ids[row])
            ):
                raise ValueError(
                    "PWL curve target row changes its global mesh-cell identity."
                )
            target_first, target_last = references[row]
            if (
                len(target_first) != mesh.topological_dimension
                or len(target_last) != mesh.topological_dimension
                or target_first == target_last
            ):
                raise ValueError("The witnessed target-reference edge is degenerate.")
            route = np.asarray(routes[block_index][cell_row], dtype=np.int64)
            local = tuple(
                coefficients[int(route[route_row])] for route_row in range(route.shape[0])
            )
            element = elements[block_index]
            if not isinstance(element, ScalarCoordinateElement):
                raise TypeError(
                    "PWL curve target maps require scalar coordinate elements."
                )
            target_expressions = coordinate_map_expressions(
                element, local, cell_row=cell_row
            )
            if len(target_expressions) != source.ambient_dimension:
                raise ValueError(
                    "Target coordinate dimension differs from its PWL curve source."
                )
            source_first, source_last = (
                parameters[row][0][0],
                parameters[row][1][0],
            )
            chart_id = witness.source_chart_ids[row]
            chart_expressions_raw = source.chart_coordinate_expressions(1, chart_id)
            if len(chart_expressions_raw) != source.ambient_dimension:
                raise ValueError(
                    "PWL source chart coordinate map has the wrong physical dimension."
                )
            chart_expressions: tuple[Polynomial, ...] = tuple(
                {
                    power: Fraction(coefficient)
                    for power, coefficient in expression.items()
                }
                for expression in chart_expressions_raw
            )
            pending = [(Fraction(0), Fraction(1))]
            while pending:
                first, last = pending.pop()
                target_piece = _target_subinterval_expressions(
                    target_expressions,
                    target_first,
                    target_last,
                    first,
                    last,
                )
                parameter_first = source_first + (source_last - source_first) * first
                parameter_last = source_first + (source_last - source_first) * last
                source_argument: tuple[Polynomial, ...] = (
                    {
                        (0,): parameter_first,
                        (1,): parameter_last - parameter_first,
                    },
                )
                source_piece = tuple(
                    compose_expression(expression, source_argument)
                    for expression in chart_expressions
                )
                differences = tuple(
                    subtract(target_value, source_value)
                    for target_value, source_value in zip(
                        target_piece, source_piece, strict=True
                    )
                )
                lower, upper = _polynomial_distance_bounds(differences)
                if isinstance(upper, Fraction):
                    row_upper = max(row_upper, upper)
                else:
                    row_upper = math.inf
                    row_unresolved = True
                    break
                if upper <= limit:
                    continue
                if lower > limit:
                    row_unresolved = True
                    row_upper = math.inf
                    break
                middle = (first + last) / 2
                if not float(first) < float(middle) < float(last):
                    row_unresolved = True
                    row_upper = math.inf
                    break
                pending.extend(((middle, last), (first, middle)))
        except (ArithmeticError, FloatingPointError, IndexError, TypeError, ValueError):
            row_upper = math.inf
            row_unresolved = True
        upper_bound = (
            fraction_endpoint(row_upper, upper=True, saturate=True)
            if isinstance(row_upper, Fraction)
            else math.inf
        )
        deviations[row] = (0.0, upper_bound)
        if isinstance(row_upper, Fraction):
            upper_max = max(upper_max, row_upper)
        else:
            upper_max = math.inf
        if row_unresolved:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_distance_unresolved",
                    "unresolved",
                    "cell",
                    (target_entity,),
                    detail="PWL source/target interval bounds did not resolve the requested tolerance.",
                )
            )
        elif row_upper > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "unresolved",
                    "cell",
                    (target_entity,),
                    detail="The continuous PWL source/target deviation upper bound exceeds tolerance.",
                )
            )
    upper_float = (
        fraction_endpoint(upper_max, upper=True, saturate=True)
        if isinstance(upper_max, Fraction)
        else math.inf
    )
    return SourceChartFidelity(
        target_entities,
        chart_rows,
        witness.occurrence_paths,
        deviations,
        0.0,
        upper_float,
        0.0,
        upper_float,
        tuple(findings),
    )


def certify_domain_surface_charts(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MeshingDomainBoundarySource,
    witness: SourceBoundaryChartWitness,
    patch_rows: np.ndarray,
    target_rows: tuple[tuple[int, int], ...],
    /,
    *,
    tolerance: float,
    plane_patches: frozenset[int] = frozenset(),
    preparation: SourcePreparation | None = None,
) -> SourceChartFidelity:
    """Bound the two directed errors on complete affine material-chart pieces.

    Rows of ``plane_patches`` are charted in their patch's `exact_plane_chart`.
    ``preparation`` is the caller's `SourcePreparation` of the source domain.
    """
    from ..discretization._cell_geometry import CellGeometrySpec as _CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh as _CellMesh
    from ._domain_boundary import exact_plane_chart
    from ._source_fidelity import (
        _affine_triangle_arguments,
        _affine_volume_triangle_arguments,
        _box_gap_upper,
        _exact_triangle2,
        _exact_triangle3,
        _polyhedral_vertex_source_coordinates,
    )

    if not isinstance(mesh, _CellMesh) or not isinstance(geometry, _CellGeometrySpec):
        raise TypeError(
            "Surface chart interpolation requires canonical mesh/map operands."
        )
    if len(patch_rows) != len(target_rows) or len(target_rows) != len(
        witness.source_chart_ids
    ):
        raise ValueError("Surface chart source and target rows must align.")
    cell_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    if cell_ids.shape != (len(target_rows),):
        raise ValueError(
            "Surface chart witness must bind every target row by its global cell ID."
        )
    for cell_id, (block_index, cell_row) in zip(
        cell_ids.tolist(), target_rows, strict=True
    ):
        if int(mesh.blocks[block_index].global_ids[cell_row]) != int(cell_id):
            raise ValueError(
                "Surface chart target row changes its actual global mesh-cell identity."
            )
    domain = source.domain
    preparation = SourcePreparation.of(domain, preparation)
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    polyhedral = np.asarray(witness.target_face_vertex_global_ids).shape[0] != 0
    target_vertex_coordinates = (
        _polyhedral_vertex_source_coordinates(mesh, geometry) if polyhedral else {}
    )
    charts = tuple(
        _exact_triangle2(witness.source_parameter_chart(row))
        for row in range(len(witness.source_chart_ids))
    )
    deviations = np.zeros((len(cell_ids), 2), dtype=np.float64)
    deviations[:, 1] = math.inf
    target_corners: list[tuple[tuple[ExactScalar, ...], ...]] = [() for _ in cell_ids]
    target_affine: list[bool] = [False for _ in cell_ids]
    findings: list[MeshCertificateFinding] = []
    exact_surface_maps: dict[int, tuple[Polynomial, ...]] = {}
    patch_bounds: dict[int, np.ndarray] = {}
    source_point_intervals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    grouped_rows: dict[int, list[int]] = {}
    adaptive_patches: set[int] = set()
    for row, patch in enumerate(patch_rows):
        grouped_rows.setdefault(int(patch), []).append(row)
    patch_local_row_by_witness_row = {
        row: local_row
        for rows in grouped_rows.values()
        for local_row, row in enumerate(rows)
    }
    for patch, rows in grouped_rows.items():
        surface = domain.patches[patch].surface
        try:
            if patch in plane_patches:
                plane_chart = exact_plane_chart(domain, patch)
                if plane_chart is None:
                    raise ValueError("Source patch lost its exact plane chart.")
                exact_surface = plane_chart.coordinate_expressions()
            else:
                exact_surface = exact_surface_coordinate_expressions(surface)
            spans = None
            unexpanded_exact_carrier = False
            if exact_surface is None:
                spans = preparation.surface_spans(surface)
                unexpanded_exact_carrier = _has_unexpanded_exact_surface_carrier(surface)
                if _has_analytic_source_program(surface):
                    # Both programs bind the exact coefficient intervals.
                    preparation.derivative_program(surface, 0)
                    preparation.derivative_program(surface, 2)
                    unexpanded_exact_carrier = False
        except (ArithmeticError, FloatingPointError, TypeError, ValueError) as error:
            for _ in rows:
                findings.append(
                    MeshCertificateFinding(
                        "source_chart_interpolation",
                        "unresolved",
                        "source_facet",
                        (patch,),
                        detail=f"Exact source-chart capability is unavailable: {error}",
                    )
                )
            continue
        if exact_surface is None and unexpanded_exact_carrier and spans is None:
            for _ in rows:
                findings.append(
                    MeshCertificateFinding(
                        "source_chart_interpolation",
                        "unresolved",
                        "source_facet",
                        (patch,),
                        detail="An exact-backed source carrier has no retained exact coordinate map; rounded dynamic values are not a proof.",
                    )
                )
            continue
        if exact_surface is not None:
            interpolation = np.empty((len(rows),), dtype=np.float64)
            lower = np.empty((3 * len(rows), 3), dtype=np.float64)
            upper = np.empty_like(lower)
            try:
                for local_row, row in enumerate(rows):
                    chart = charts[row]
                    interpolation[local_row] = _exact_surface_interpolation_bound(
                        exact_surface, chart
                    )
                    for corner, point in enumerate(chart):
                        for axis, expression in enumerate(exact_surface):
                            value = evaluate_expression(expression, point)
                            target = 3 * local_row + corner
                            lower[target, axis] = coefficient_endpoint(value, upper=False)
                            upper[target, axis] = coefficient_endpoint(value, upper=True)
            except (ArithmeticError, FloatingPointError, TypeError, ValueError):
                interpolation.fill(math.inf)
                lower.fill(-math.inf)
                upper.fill(math.inf)
            exact_surface_maps[patch] = exact_surface
        else:
            try:
                if spans is not None:
                    interpolation = _exact_rational_span_interpolation_bounds(
                        spans, tuple(charts[row] for row in rows)
                    )
                    if (
                        interpolation.shape != (len(rows),)
                        or np.any(~np.isfinite(interpolation))
                        or np.any(interpolation < 0.0)
                    ):
                        raise ValueError(
                            "Exact span interpolation bounds are malformed or nonfinite."
                        )
                    lower = np.empty((3 * len(rows), 3), dtype=np.float64)
                    upper = np.empty_like(lower)
                    for local_row, row in enumerate(rows):
                        for corner, point in enumerate(charts[row]):
                            point_bounds = _exact_rational_surface_point_bounds(
                                spans, point
                            )
                            if point_bounds is None or len(point_bounds) != 3:
                                raise ValueError(
                                    "Exact span source corner has no finite ordered point box."
                                )
                            target_row = 3 * local_row + corner
                            lower[target_row] = tuple(
                                fraction_endpoint(bound[0], upper=False)
                                for bound in point_bounds
                            )
                            upper[target_row] = tuple(
                                fraction_endpoint(bound[1], upper=True)
                                for bound in point_bounds
                            )
                    patch_bounds[patch] = interpolation
                    source_point_intervals[patch] = (lower, upper)
                    continue

                parameter_lowers: list[tuple[float, float]] = []
                parameter_uppers: list[tuple[float, float]] = []
                for row in rows:
                    for point in charts[row]:
                        u_interval = _fraction_float_interval(point[0])
                        v_interval = _fraction_float_interval(point[1])
                        parameter_lowers.append((u_interval[0], v_interval[0]))
                        parameter_uppers.append((u_interval[1], v_interval[1]))
                # Generic carriers are bounded once each row's budget is known.
                interpolation = np.zeros((len(rows),), dtype=np.float64)
                adaptive_patches.add(patch)
                program = preparation.derivative_program(surface, 0)
                lower, upper = program.evaluate(
                    np.asarray(parameter_lowers, dtype=np.float64),
                    np.asarray(parameter_uppers, dtype=np.float64),
                )
                lower, upper = (
                    np.asarray(lower, dtype=np.float64),
                    np.asarray(upper, dtype=np.float64),
                )
            except (
                IntervalExtensionError,
                NotImplementedError,
                TypeError,
                ValueError,
                FloatingPointError,
            ) as error:
                for _ in rows:
                    findings.append(
                        MeshCertificateFinding(
                            "source_chart_interpolation",
                            "unresolved",
                            "source_facet",
                            (patch,),
                            detail=f"Continuous source chart interpolation is unavailable: {error}",
                        )
                    )
                continue
        if (
            interpolation.shape != (len(rows),)
            or np.any(~np.isfinite(interpolation))
            or np.any(interpolation < 0.0)
            or lower.shape != (3 * len(rows), 3)
            or upper.shape != lower.shape
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
        ):
            for _ in rows:
                findings.append(
                    MeshCertificateFinding(
                        "source_chart_interpolation",
                        "unresolved",
                        "source_facet",
                        (patch,),
                        detail="Source chart interpolation enclosures are malformed or nonfinite.",
                    )
                )
            continue
        patch_bounds[patch] = interpolation
        source_point_intervals[patch] = (lower, upper)
    limit = Fraction.from_float(float(tolerance))
    upper_max: Fraction | float = Fraction(0)
    adaptive_rows: dict[int, list[tuple[int, Fraction, Fraction | float]]] = {}

    def record_deviation(row: int, deviation: Fraction | float, /) -> None:
        nonlocal upper_max
        deviation_upper = (
            fraction_endpoint(deviation, upper=True, saturate=True)
            if isinstance(deviation, Fraction)
            else math.inf
        )
        deviations[row] = (0.0, deviation_upper)
        upper_max = max(upper_max, deviation)
        if deviation > limit:
            findings.append(
                MeshCertificateFinding(
                    "source_fidelity_tolerance",
                    "unresolved",
                    "cell",
                    (int(cell_ids[row]),),
                    detail="A continuous two-directed chart deviation upper bound exceeds tolerance.",
                )
            )

    for row, (patch_raw, (block_index, cell_row)) in enumerate(
        zip(patch_rows, target_rows, strict=True)
    ):
        patch = int(patch_raw)
        if patch not in patch_bounds:
            continue
        try:
            if polyhedral:
                if mesh.blocks[block_index].cell_kind != "polyhedron":
                    raise ValueError("Polygonal target charts require polyhedron blocks.")
                parent_vertex_ids = np.asarray(
                    witness.target_face_vertex_global_ids[row], dtype=np.int64
                )
                parent_corners = (
                    target_vertex_coordinates[int(parent_vertex_ids[0])],
                    target_vertex_coordinates[int(parent_vertex_ids[1])],
                    target_vertex_coordinates[int(parent_vertex_ids[2])],
                )
                target_reference = _exact_triangle2(witness.target_reference_chart(row))
                corners = tuple(
                    tuple(
                        parent_corners[0][axis]
                        + target_reference[corner][0]
                        * (parent_corners[1][axis] - parent_corners[0][axis])
                        + target_reference[corner][1]
                        * (parent_corners[2][axis] - parent_corners[0][axis])
                        for axis in range(3)
                    )
                    for corner in range(3)
                )
                target_interpolation = Fraction(0)
                is_affine = True
            else:
                arguments: tuple[Polynomial, ...]
                if mesh.topological_dimension == 2:
                    target_reference = _exact_triangle2(
                        witness.target_reference_chart(row)
                    )
                    arguments = _affine_triangle_arguments(target_reference)
                elif mesh.topological_dimension == 3:
                    target_reference = _exact_triangle3(
                        witness.target_reference_chart(row)
                    )
                    arguments = _affine_volume_triangle_arguments(target_reference)
                else:
                    raise ValueError(
                        "Surface chart target maps require 2-D or 3-D parent references."
                    )
                local = tuple(
                    coefficients[int(index)]
                    for index in np.asarray(routes[block_index][cell_row])
                )
                element = elements[block_index]
                if not isinstance(element, ScalarCoordinateElement):
                    raise TypeError(
                        "Target surface map requires a scalar coordinate element."
                    )
                physical = coordinate_map_expressions(element, local, cell_row=cell_row)
                if len(physical) != 3:
                    raise ValueError(
                        "The physical target map must take values in three-space."
                    )
                mapped = tuple(compose_expression(value, arguments) for value in physical)
                barycentric_corners = (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(0), Fraction(1)),
                )
                corners = tuple(
                    tuple(evaluate_expression(value, point) for value in mapped)
                    for point in barycentric_corners
                )
                residual_components: list[Fraction] = []
                target_map_unbounded = False
                is_affine = True
                for axis, expression in enumerate(mapped):
                    is_affine = (
                        is_affine
                        and isinstance(expression, dict)
                        and all(sum(power) <= 1 for power in expression)
                    )
                    affine: Polynomial = {
                        (0, 0): corners[0][axis],
                        (1, 0): corners[1][axis] - corners[0][axis],
                        (0, 1): corners[2][axis] - corners[0][axis],
                    }
                    residual = subtract(expression, affine)
                    if (
                        isinstance(residual, RationalPolynomial)
                        and not residual.numerator
                    ):
                        residual_components.append(Fraction(0))
                        continue
                    if isinstance(residual, dict) and not residual:
                        residual_components.append(Fraction(0))
                        continue
                    bound_lower, bound_upper = polynomial_bounds(residual, "simplex", 2)
                    if not math.isfinite(bound_lower) or not math.isfinite(bound_upper):
                        target_map_unbounded = True
                        continue
                    residual_components.append(
                        max(
                            abs(Fraction.from_float(bound_lower)),
                            abs(Fraction.from_float(bound_upper)),
                        )
                    )
                if target_map_unbounded:
                    target_interpolation = math.inf
                else:
                    target_interpolation = fraction_sqrt_interval(
                        sum((value * value for value in residual_components), Fraction(0))
                    )[1]
        except (
            ArithmeticError,
            FloatingPointError,
            TypeError,
            ValueError,
            IndexError,
        ) as error:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_target_map",
                    "unresolved",
                    "cell",
                    (int(cell_ids[row]),),
                    detail=f"Continuous target boundary map is unavailable: {error}",
                )
            )
            continue
        target_corners[row] = corners
        target_affine[row] = is_affine
        exact_surface = exact_surface_maps.get(patch)
        exact_source = (
            None
            if exact_surface is None
            else tuple(
                tuple(
                    evaluate_expression(expression, point) for expression in exact_surface
                )
                for point in charts[row]
            )
        )
        endpoint_error = Fraction(0)
        local_row = patch_local_row_by_witness_row[row]
        lower, upper = source_point_intervals[patch]
        for corner in range(3):
            if exact_source is None:
                component_errors = tuple(
                    _box_gap_upper(
                        float(lower[3 * local_row + corner, axis]),
                        float(upper[3 * local_row + corner, axis]),
                        value,
                    )
                    for axis, value in enumerate(corners[corner])
                )
            else:
                component_errors = tuple(
                    max(
                        abs(
                            Fraction(
                                coefficient_endpoint(target - source_value, upper=True)
                            )
                        ),
                        abs(
                            Fraction(
                                coefficient_endpoint(target - source_value, upper=False)
                            )
                        ),
                    )
                    for target, source_value in zip(
                        corners[corner], exact_source[corner], strict=True
                    )
                )
            endpoint_error = max(
                endpoint_error,
                fraction_sqrt_interval(
                    sum((value * value for value in component_errors), Fraction(0))
                )[1],
            )
        if exact_surface is not None and _exact_surface_map_is_affine(exact_surface):
            source_interpolation = Fraction(0)
        elif patch in adaptive_patches:
            adaptive_rows.setdefault(patch, []).append(
                (row, endpoint_error, target_interpolation)
            )
            continue
        else:
            source_interpolation = Fraction.from_float(
                float(patch_bounds[patch][local_row])
            )
        record_deviation(
            row, source_interpolation + endpoint_error + target_interpolation
        )
    for patch, pending in adaptive_rows.items():
        try:
            adaptive_bounds = domain.adaptive_interpolation_bounds(
                patch,
                tuple(charts[row] for row, _, _ in pending),
                tuple(
                    limit - endpoint_error - target_interpolation
                    if isinstance(target_interpolation, Fraction)
                    else Fraction(-1)
                    for _, endpoint_error, target_interpolation in pending
                ),
                preparation=preparation,
            )
        except (
            ArithmeticError,
            FloatingPointError,
            IntervalExtensionError,
            NotImplementedError,
            TypeError,
            ValueError,
        ) as error:
            findings.extend(
                MeshCertificateFinding(
                    "source_chart_interpolation",
                    "unresolved",
                    "source_facet",
                    (patch,),
                    detail=f"Continuous source chart interpolation is unavailable: {error}",
                )
                for _ in pending
            )
            continue
        for (row, endpoint_error, target_interpolation), bound in zip(
            pending, adaptive_bounds, strict=True
        ):
            if bound is None:
                findings.append(
                    MeshCertificateFinding(
                        "source_chart_interpolation",
                        "unresolved",
                        "source_facet",
                        (patch,),
                        detail="Source chart interpolation enclosures are malformed or nonfinite.",
                    )
                )
                continue
            record_deviation(row, bound + endpoint_error + target_interpolation)
    upper_float = (
        fraction_endpoint(upper_max, upper=True, saturate=True)
        if isinstance(upper_max, Fraction)
        else math.inf
    )
    return SourceChartFidelity(
        tuple(int(value) for value in witness.entity_global_ids),
        tuple(int(value) for value in patch_rows),
        witness.occurrence_paths,
        deviations,
        0.0,
        upper_float,
        0.0,
        upper_float,
        tuple(findings),
        tuple(target_corners),
        tuple(target_affine),
    )


__all__ = [
    "SourceChartFidelity",
    "certify_domain_surface_charts",
    "certify_parametric_curve_chart",
]
