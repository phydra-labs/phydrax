#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import comb, isfinite
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..._interpolation import rational_bspline_curve_spans, rational_bspline_surface_spans
from ...algebraic._exact_univariate import isolate_real_roots
from ...discretization._coordinate_enclosure import (
    ExactScalar,
    Polynomial,
    RationalPolynomial,
    add,
    axes,
    compose,
    constant,
    derivative,
    multiply,
    polynomial_bounds,
    scale,
    subtract,
)
from ._intersection_kernel import krawczyk
from ._patches import (
    AbstractCurve,
    BSplineCurve,
    BSplineSurfacePatch,
    ExactLineCurve,
    LineCurve,
)
from ._projection_contracts import BRepProjectionStatus
from ._tspline import TSplineSurfacePatch
if TYPE_CHECKING:
    from ._model import BRepModel
    from ._projection_contracts import BRepProjectionPolicy


type _SurfaceBounds = tuple[Fraction, Fraction, Fraction, Fraction]
type _HomogeneousNet = tuple[tuple[tuple[ExactScalar, ...], ...], ...]
type _RawSurfacePiece = tuple[tuple[object, ...], _SurfaceBounds, _HomogeneousNet]
type _ChartBox2 = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


class _PolynomialEvaluator:
    def __init__(self, system: _PolynomialSystem, *, jacobian: bool) -> None:
        self._system = system
        self._jacobian = jacobian

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        box_lower = tuple(float(value) for value in lower[0])
        box_upper = tuple(float(value) for value in upper[0])
        if not self._jacobian:
            bounds = tuple(
                _poly_interval(value, box_lower, box_upper)
                for value in self._system._values
            )
            return (
                np.asarray([[value[0] for value in bounds]], dtype=np.float64),
                np.asarray([[value[1] for value in bounds]], dtype=np.float64),
            )
        bounds = tuple(
            tuple(
                _poly_interval(value, box_lower, box_upper)
                for value in row
            )
            for row in self._system._jacobian
        )
        return (
            np.asarray([[[value[0] for value in row] for row in bounds]], dtype=np.float64),
            np.asarray([[[value[1] for value in row] for row in bounds]], dtype=np.float64),
        )


class _PolynomialSystem:
    def __init__(
        self,
        values: tuple[Polynomial, Polynomial],
        jacobian: tuple[tuple[Polynomial, Polynomial], ...],
    ) -> None:
        self._values = values
        self._jacobian = jacobian
        self.value = _PolynomialEvaluator(self, jacobian=False)
        self.jacobian = _PolynomialEvaluator(self, jacobian=True)

    def point_jacobians(self, points: np.ndarray, /) -> np.ndarray:
        rows = []
        for point in points:
            exact = tuple(Fraction.from_float(float(value)) for value in point)
            rows.append(
                [
                    [float(_eval_polynomial(self._jacobian[row][axis], exact)) for axis in range(2)]
                    for row in range(2)
                ]
            )
        return np.asarray(rows, dtype=np.float64)
def _curve_value_box(
    coordinates: tuple[RationalPolynomial, ...],
    lower: float,
    upper: float,
    /,
) -> tuple[tuple[float, float], ...]:
    argument = (_affine_argument(Fraction.from_float(lower), Fraction.from_float(upper), 1, 0),)
    return tuple(
        polynomial_bounds(
            RationalPolynomial(
                compose(dict(value.numerator), argument, dimension=1),
                compose(dict(value.denominator), argument, dimension=1),
            ),
            "box",
            1,
        )
        for value in coordinates
    )


def _local_float_interval(
    lower: Fraction, upper: Fraction, /
) -> tuple[float, float]:
    return (
        max(0.0, float(np.nextafter(float(lower), -np.inf))),
        min(1.0, float(np.nextafter(float(upper), np.inf))),
    )


def _unclamped_float_interval(
    lower: Fraction, upper: Fraction, /
) -> tuple[float, float]:
    return (
        float(np.nextafter(float(lower), -np.inf)),
        float(np.nextafter(float(upper), np.inf)),
    )


def _chart_box(
    segment: _CurveSegment,
    lower: Fraction,
    upper: Fraction,
    /,
) -> _ChartBox2 | None:
    if segment.chart_coordinates is None:
        return None
    local_lower, local_upper = _local_float_interval(lower, upper)
    bounds = _curve_value_box(segment.chart_coordinates, local_lower, local_upper)
    if any(not isfinite(first) or not isfinite(last) for first, last in bounds):
        return None
    return (
        (
            Fraction.from_float(bounds[0][0]),
            Fraction.from_float(bounds[0][1]),
        ),
        (
            Fraction.from_float(bounds[1][0]),
            Fraction.from_float(bounds[1][1]),
        ),
    )


def _curve_root_intervals(
    polynomial: Polynomial,
    segment: _CurveSegment,
    distance_squared: RationalPolynomial,
    policy: BRepProjectionPolicy,
    /,
) -> tuple[
    tuple[tuple[tuple[Fraction, Fraction], int, tuple[float, float]], ...],
    tuple[_Unknown, ...],
    bool,
]:
    coefficients = _root_coefficients(polynomial)
    if coefficients is None:
        return (), (_Unknown(0.0, np.inf, None, segment.piece_id, ("unrepresentable-univariate-coefficients",)),), False
    if coefficients == (Fraction(0),):
        return (), (), True
    span_width = abs(segment.parameter_bounds[1] - segment.parameter_bounds[0])
    tolerance = min(
        Fraction(1),
        Fraction.from_float(float(policy.parametric_tolerance)) / max(span_width, Fraction(1)),
    )
    while True:
        roots = isolate_real_roots(coefficients, tolerance=tolerance)
        selected = []
        failed = False
        unresolved: list[_Unknown] = []
        representability_blocked = False
        for lower, upper, multiplicity in roots:
            if upper < 0 or lower > 1:
                continue
            interval = max(Fraction(0), lower), min(Fraction(1), upper)
            if interval[0] > interval[1]:
                continue
            floats = _local_float_interval(*interval)
            distance_bounds = _distance_bounds(
                distance_squared,
                (floats[0],),
                (floats[1],),
            )
            coordinate_bounds = _curve_value_box(
                segment.coordinates, floats[0], floats[1]
            )
            diameter = _diameter(coordinate_bounds)
            chart_diameter = (
                0.0
                if segment.chart_coordinates is None
                else _diameter(
                    _curve_value_box(
                        segment.chart_coordinates, floats[0], floats[1]
                    )
                )
            )
            param_width = float(span_width * (interval[1] - interval[0]))
            distance_width = distance_bounds[1] - distance_bounds[0]
            if (
                param_width > policy.parametric_tolerance
                or chart_diameter > policy.parametric_tolerance
                or diameter > policy.ambiguity_tolerance
                or distance_width > policy.ambiguity_tolerance
            ):
                failed = True
                if floats[0] == floats[1] or np.nextafter(floats[0], np.inf) >= floats[1]:
                    representability_blocked = True
                    unresolved.append(
                        _Unknown(distance_bounds[0], distance_bounds[1], None, segment.piece_id, ("curve-root-resolution", interval, multiplicity))
                    )
            selected.append((interval, multiplicity, distance_bounds))
        if not failed:
            return tuple(selected), tuple(unresolved), False
        if representability_blocked:
            return tuple(selected), tuple(unresolved), False
        next_tolerance = tolerance / 2
        if next_tolerance == 0 or float(next_tolerance) == 0.0:
            return tuple(selected), tuple(unresolved) or (_Unknown(0.0, np.inf, None, segment.piece_id, ("curve-root-binary64-resolution",)),), False
        tolerance = next_tolerance

def _univariate_zero_locations(
    polynomial: Polynomial, policy: BRepProjectionPolicy, /
) -> tuple[Fraction, ...] | None:
    coefficients = _root_coefficients(polynomial)
    if coefficients is None:
        return None
    if coefficients == (Fraction(0),):
        return ()
    roots = isolate_real_roots(
        coefficients,
        tolerance=Fraction.from_float(float(policy.parametric_tolerance)),
    )
    locations = []
    for lower, upper, _ in roots:
        if lower == upper and 0 < lower < 1:
            locations.append(lower)
        elif upper > 0 and lower < 1:
            return None
    return tuple(locations)


def _partition_pcurve_segment(
    segment: _CurveSegment,
    pieces: tuple[_SplinePiece, ...],
    policy: BRepProjectionPolicy,
    /,
) -> tuple[tuple[_CurveSegment, _SplinePiece], ...] | None:
    if segment.chart_coordinates is None:
        return None
    u_expr, v_expr = segment.chart_coordinates
    denominator_u = dict(u_expr.denominator)
    denominator_v = dict(v_expr.denominator)
    cuts = {Fraction(0), Fraction(1)}
    for piece in pieces:
        for value, denominator, lower, upper in (
            (dict(u_expr.numerator), denominator_u, piece.bounds[0], piece.bounds[1]),
            (dict(v_expr.numerator), denominator_v, piece.bounds[2], piece.bounds[3]),
        ):
            for boundary in (lower, upper):
                locations = _univariate_zero_locations(
                    subtract(value, scale(denominator, boundary)), policy
                )
                if locations is None:
                    return None
                cuts.update(locations)
    ordered = tuple(sorted(cuts))
    result: list[tuple[_CurveSegment, _SplinePiece]] = []
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        if lower >= upper:
            continue
        middle = (lower + upper) / 2
        uv = (
            _eval_rational(u_expr, (middle,)),
            _eval_rational(v_expr, (middle,)),
        )
        owners = [
            piece
            for piece in pieces
            if piece.bounds[0] <= uv[0] <= piece.bounds[1]
            and piece.bounds[2] <= uv[1] <= piece.bounds[3]
        ]
        if not owners:
            return None
        for piece in owners:
            a, b = segment.parameter_bounds
            local_bounds = (a + (b - a) * lower, a + (b - a) * upper)
            first_local = _affine_argument(lower, upper, 1, 0)
            uv_restricted = (
                RationalPolynomial(
                    compose(dict(u_expr.numerator), (first_local,), dimension=1),
                    compose(dict(u_expr.denominator), (first_local,), dimension=1),
                ),
                RationalPolynomial(
                    compose(dict(v_expr.numerator), (first_local,), dimension=1),
                    compose(dict(v_expr.denominator), (first_local,), dimension=1),
                ),
            )
            u0, u1, v0, v1 = piece.bounds
            local_u = RationalPolynomial(
                subtract(dict(uv_restricted[0].numerator), scale(dict(uv_restricted[0].denominator), u0)),
                scale(dict(uv_restricted[0].denominator), u1 - u0),
            )
            local_v = RationalPolynomial(
                subtract(dict(uv_restricted[1].numerator), scale(dict(uv_restricted[1].denominator), v0)),
                scale(dict(uv_restricted[1].denominator), v1 - v0),
            )
            homogeneous = tuple(
                compose(value, (local_u, local_v), dimension=1)
                for value in piece.homogeneous
            )
            coords = _rational_map(homogeneous)
            chart = (
                RationalPolynomial(
                    compose(dict(u_expr.numerator), (first_local,), dimension=1),
                    compose(dict(u_expr.denominator), (first_local,), dimension=1),
                ),
                RationalPolynomial(
                    compose(dict(v_expr.numerator), (first_local,), dimension=1),
                    compose(dict(v_expr.denominator), (first_local,), dimension=1),
                ),
            )
            result.append(
                (
                    _curve_segment(
                        (piece.piece_id, "trim", segment.piece_id, local_bounds),
                        local_bounds,
                        coords,
                        chart_coordinates=chart,
                    ),
                    piece,
                )
            )
    return tuple(result)


def _face_trim_segments(
    model: BRepModel,
    face: int,
    pieces: tuple[_SplinePiece, ...],
    policy: BRepProjectionPolicy,
    /,
) -> tuple[tuple[_CurveSegment, _SplinePiece], ...] | None:
    geometry = model.geometry
    face_segments: list[tuple[_CurveSegment, _SplinePiece]] = []
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            first_float, last_float = ranges[edge]
            pcurve = geometry.pcurves[coedge]
            segments = _pcurve_segments(
                pcurve,
                Fraction.from_float(float(first_float)),
                Fraction.from_float(float(last_float)),
            )
            if segments is None:
                return None
            for segment in segments:
                scoped_segment = _curve_segment(
                    ("face-coedge", face, coedge, edge, segment.piece_id),
                    segment.parameter_bounds,
                    segment.coordinates,
                    chart_coordinates=segment.chart_coordinates,
                )
                partitioned = _partition_pcurve_segment(
                    scoped_segment, pieces, policy
                )
                if partitioned is None:
                    return None
                face_segments.extend(partitioned)
    return tuple(face_segments)


@dataclass(frozen=True, slots=True)
class _SplinePiece:
    piece_id: tuple[object, ...]
    bounds: _SurfaceBounds
    homogeneous: tuple[Polynomial, Polynomial, Polynomial, Polynomial]
    coordinates: tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]
    normal_squared: Polynomial


@dataclass(frozen=True, slots=True)
class _CurveSegment:
    piece_id: tuple[object, ...]
    parameter_bounds: tuple[Fraction, Fraction]
    coordinates: tuple[RationalPolynomial, ...]
    chart_coordinates: tuple[RationalPolynomial, RationalPolynomial] | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    point: np.ndarray
    parameters: np.ndarray
    lower: float
    upper: float
    status: int
    piece_id: tuple[object, ...]
    root_box: tuple[tuple[Fraction, Fraction], ...]
    exact_parameters: tuple[ExactScalar, ...]
    root_id: tuple[object, ...]
    root_system: _PolynomialSystem | None = None
    root_piece_bounds: _SurfaceBounds | None = None
    root_box_unique: bool = False
    surface_piece_id: tuple[object, ...] | None = None
    exact_root: bool = False
    exact_point: tuple[ExactScalar, ...] | None = None


@dataclass(frozen=True, slots=True)
class _Unknown:
    lower: float
    upper: float
    chart_box: _ChartBox2 | None
    piece_id: tuple[object, ...]
    root_id: tuple[object, ...]
    surface_piece_id: tuple[object, ...] | None = None
    exact_parameters: tuple[ExactScalar, ...] | None = None
    exact_root: bool = False
    root_system: _PolynomialSystem | None = None
    root_piece_bounds: _SurfaceBounds | None = None


@dataclass(frozen=True, slots=True)
class SplineProjection:
    point: np.ndarray
    parameters: np.ndarray
    lower: float
    upper: float
    status: int
    piece_id: tuple[object, ...] | None = None
    root_id: tuple[object, ...] | None = None


@dataclass(frozen=True, slots=True)
class CurveProjection:
    point: np.ndarray
    parameter: float
    lower: float
    upper: float
    status: int
    piece_id: tuple[object, ...] | None = None
    root_id: tuple[object, ...] | None = None


_NAN3 = np.full((3,), np.nan, dtype=np.float64)
_NAN2 = np.full((2,), np.nan, dtype=np.float64)


def _bernstein_power(controls: tuple[tuple[ExactScalar, ...], ...]) -> Polynomial:
    degree_u, degree_v = len(controls) - 1, len(controls[0]) - 1
    result: Polynomial = {}
    for i, j in product(range(degree_u + 1), range(degree_v + 1)):
        value = controls[i][j]
        factor = comb(degree_u, i) * comb(degree_v, j)
        for a in range(degree_u - i + 1):
            first = i + a
            first_coefficient = (-1 if a & 1 else 1) * comb(degree_u - i, a)
            for b in range(degree_v - j + 1):
                exponent = (first, j + b)
                coefficient = (
                    value
                    * factor
                    * first_coefficient
                    * (-1 if b & 1 else 1)
                    * comb(degree_v - j, b)
                )
                if coefficient:
                    result[exponent] = result.get(exponent, Fraction(0)) + coefficient
    return {index: value for index, value in result.items() if value}


def _bernstein_curve_power(controls: tuple[ExactScalar, ...]) -> Polynomial:
    degree = len(controls) - 1
    result: Polynomial = {}
    for i, value in enumerate(controls):
        for offset in range(degree - i + 1):
            exponent = (i + offset,)
            coefficient = (
                value
                * comb(degree, i)
                * (-1 if offset & 1 else 1)
                * comb(degree - i, offset)
            )
            if coefficient:
                result[exponent] = result.get(exponent, Fraction(0)) + coefficient
    return {index: value for index, value in result.items() if value}


def _affine_argument(
    lower: Fraction, upper: Fraction, dimension: int, axis: int
) -> Polynomial:
    variables = axes(dimension)
    return add(
        constant(lower, dimension),
        scale(variables[axis], upper - lower),
    )



def _chart_coordinate_to_local(
    expression: RationalPolynomial,
    lower: Fraction,
    upper: Fraction,
    /,
) -> RationalPolynomial:
    denominator = dict(expression.denominator)
    return RationalPolynomial(
        subtract(dict(expression.numerator), scale(denominator, lower)),
        scale(denominator, upper - lower),
    )

def _restrict_surface_polynomial(
    polynomial: Polynomial,
    source_bounds: tuple[Fraction, Fraction, Fraction, Fraction],
    bounds: tuple[Fraction, Fraction, Fraction, Fraction],
) -> Polynomial:
    u0, u1, v0, v1 = source_bounds
    a0, a1, b0, b1 = bounds
    local_u0, local_u1 = (a0 - u0) / (u1 - u0), (a1 - u0) / (u1 - u0)
    local_v0, local_v1 = (b0 - v0) / (v1 - v0), (b1 - v0) / (v1 - v0)
    return compose(
        polynomial,
        (
            _affine_argument(local_u0, local_u1, 2, 0),
            _affine_argument(local_v0, local_v1, 2, 1),
        ),
        dimension=2,
    )


def _rational_map(
    homogeneous: tuple[Polynomial | RationalPolynomial, ...], /
) -> tuple[RationalPolynomial, ...]:
    weight = homogeneous[-1]
    weight_numerator = (
        dict(weight.numerator) if isinstance(weight, RationalPolynomial) else weight
    )
    if not weight_numerator:
        raise ValueError("An exact spline piece has an identically zero rational weight.")
    weight_denominator = (
        dict(weight.denominator)
        if isinstance(weight, RationalPolynomial)
        else constant(1, len(next(iter(weight_numerator))))
    )
    dimension = len(next(iter(weight_numerator)))
    result = []
    for value in homogeneous[:-1]:
        numerator = (
            dict(value.numerator) if isinstance(value, RationalPolynomial) else value
        )
        denominator = (
            dict(value.denominator)
            if isinstance(value, RationalPolynomial)
            else constant(1, dimension)
        )
        result.append(
            RationalPolynomial(
                multiply(numerator, weight_denominator),
                multiply(denominator, weight_numerator),
            )
        )
    return tuple(result)

def _surface_pieces(model: BRepModel, face: int, /) -> tuple[_SplinePiece, ...]:
    patch = model.patches[face]
    raw_pieces: list[_RawSurfacePiece] = []
    if isinstance(patch, TSplineSurfacePatch):
        patch.validate_exact_transfer_chain()
        for extraction in patch.realization.extractions:
            bounds: _SurfaceBounds = extraction.parameter_bounds
            rows = patch.realization.exact_homogeneous_piece(
                extraction.piece_id, patch.exact_target_homogeneous
            )
            net: _HomogeneousNet = tuple(
                tuple(
                    tuple(rows[i * 4 + j][axis] for axis in range(4))
                    for j in range(4)
                )
                for i in range(4)
            )
            raw_pieces.append(
                (("tspline", extraction.cell_id, extraction.piece_id), bounds, net)
            )
        domain = patch.mesh.parameter_domain
        full_u = (domain[0], domain[1])
        full_v = (domain[2], domain[3])
    elif isinstance(patch, BSplineSurfacePatch):
        spans = rational_bspline_surface_spans(
            np.asarray(patch.control_points, dtype=np.float64).tolist(),
            np.asarray(patch.weights, dtype=np.float64).tolist(),
            np.asarray(patch.u_knots, dtype=np.float64).tolist(),
            np.asarray(patch.v_knots, dtype=np.float64).tolist(),
            patch.u_degree,
            patch.v_degree,
            exact_control_points=patch.exact_control_points,
            exact_weights=patch.exact_weights,
            exact_u_knots=patch.exact_u_knots,
            exact_v_knots=patch.exact_v_knots,
        )
        for span in spans:
            u_lower, u_upper = span.parameter_bounds[0]
            v_lower, v_upper = span.parameter_bounds[1]
            if (
                not isinstance(u_lower, Fraction)
                or not isinstance(u_upper, Fraction)
                or not isinstance(v_lower, Fraction)
                or not isinstance(v_upper, Fraction)
            ):
                return ()
            bounds: _SurfaceBounds = (u_lower, u_upper, v_lower, v_upper)
            raw_pieces.append(
                (
                    ("bspline", *span.span_indices, span.coefficient_rows),
                    bounds,
                    span.homogeneous_bernstein_coefficients,
                )
            )
        full_u = (
            patch.exact_u_knots[patch.u_degree],
            patch.exact_u_knots[patch.control_points.shape[0]],
        )
        full_v = (
            patch.exact_v_knots[patch.v_degree],
            patch.exact_v_knots[patch.control_points.shape[1]],
        )
    else:
        return ()
    face_bounds: _SurfaceBounds = (
        max(full_u[0], Fraction.from_float(float(model.parameter_bounds[face, 0, 0]))),
        min(full_u[1], Fraction.from_float(float(model.parameter_bounds[face, 1, 0]))),
        max(full_v[0], Fraction.from_float(float(model.parameter_bounds[face, 0, 1]))),
        min(full_v[1], Fraction.from_float(float(model.parameter_bounds[face, 1, 1]))),
    )
    if face_bounds[0] >= face_bounds[1] or face_bounds[2] >= face_bounds[3]:
        return ()

    pieces: list[_SplinePiece] = []
    for piece_id, source_bounds, controls in raw_pieces:
        bounds: _SurfaceBounds = (
            max(source_bounds[0], face_bounds[0]),
            min(source_bounds[1], face_bounds[1]),
            max(source_bounds[2], face_bounds[2]),
            min(source_bounds[3], face_bounds[3]),
        )
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            continue

        def restrict_axis(axis: int, /) -> Polynomial:
            coefficients = tuple(
                tuple(controls[i][j][axis] for j in range(len(controls[i])))
                for i in range(len(controls))
            )
            return _restrict_surface_polynomial(
                _bernstein_power(coefficients), source_bounds, bounds
            )

        homogeneous = (
            restrict_axis(0),
            restrict_axis(1),
            restrict_axis(2),
            restrict_axis(3),
        )
        mapped_coordinates = _rational_map(homogeneous)
        coordinates = (
            mapped_coordinates[0],
            mapped_coordinates[1],
            mapped_coordinates[2],
        )
        x, y, z, weight = homogeneous
        weight_u, weight_v = derivative(weight, 0), derivative(weight, 1)
        du = (
            subtract(multiply(derivative(x, 0), weight), multiply(x, weight_u)),
            subtract(multiply(derivative(y, 0), weight), multiply(y, weight_u)),
            subtract(multiply(derivative(z, 0), weight), multiply(z, weight_u)),
        )
        dv = (
            subtract(multiply(derivative(x, 1), weight), multiply(x, weight_v)),
            subtract(multiply(derivative(y, 1), weight), multiply(y, weight_v)),
            subtract(multiply(derivative(z, 1), weight), multiply(z, weight_v)),
        )
        normal = (
            subtract(multiply(du[1], dv[2]), multiply(du[2], dv[1])),
            subtract(multiply(du[2], dv[0]), multiply(du[0], dv[2])),
            subtract(multiply(du[0], dv[1]), multiply(du[1], dv[0])),
        )
        normal_squared: Polynomial = {}
        for component in normal:
            normal_squared = add(normal_squared, multiply(component, component))
        pieces.append(
            _SplinePiece(
                piece_id,
                bounds,
                homogeneous,
                coordinates,
                normal_squared,
            )
        )
    return tuple(pieces)


def _piece_query_data(
    piece: _SplinePiece, query: tuple[Fraction, Fraction, Fraction], /
) -> tuple[tuple[Polynomial, Polynomial], tuple[tuple[Polynomial, Polynomial], ...], RationalPolynomial]:
    x, y, z, weight = piece.homogeneous
    offset = tuple(
        subtract(component, scale(weight, value)) for component, value in zip((x, y, z), query, strict=True)
    )
    weight_u, weight_v = derivative(weight, 0), derivative(weight, 1)
    du = tuple(
        subtract(multiply(derivative(component, 0), weight), multiply(component, weight_u))
        for component in (x, y, z)
    )
    dv = tuple(
        subtract(multiply(derivative(component, 1), weight), multiply(component, weight_v))
        for component in (x, y, z)
    )
    stationary = []
    for direction in (du, dv):
        value: Polynomial = {}
        for first, second in zip(offset, direction, strict=True):
            value = add(value, multiply(first, second))
        stationary.append(value)
    jacobian = tuple(
        tuple(derivative(stationary[row], axis) for axis in range(2))
        for row in range(2)
    )
    distance_numerator: Polynomial = {}
    for value in offset:
        distance_numerator = add(distance_numerator, multiply(value, value))
    return (stationary[0], stationary[1]), jacobian, RationalPolynomial(
        distance_numerator, multiply(weight, weight)
    )


def _eval_polynomial(polynomial: Polynomial, values: tuple[Fraction, ...], /) -> ExactScalar:
    result: ExactScalar = Fraction(0)
    for exponents, coefficient in polynomial.items():
        term = coefficient
        for value, exponent in zip(values, exponents, strict=True):
            term *= value**exponent
        result += term
    return result


def _constant_rational_value(
    expression: RationalPolynomial, /
) -> ExactScalar | None:
    numerator = dict(expression.numerator)
    denominator = dict(expression.denominator)
    if not denominator:
        return None
    pivot = next(iter(denominator))
    value = numerator.get(pivot, Fraction(0)) / denominator[pivot]
    indices = set(numerator) | set(denominator)
    if all(
        numerator.get(index, Fraction(0))
        == value * denominator.get(index, Fraction(0))
        for index in indices
    ):
        return value
    return None
def _eval_rational(
    expression: RationalPolynomial, values: tuple[Fraction, ...], /
) -> ExactScalar:
    numerator = _eval_polynomial(dict(expression.numerator), values)
    denominator = _eval_polynomial(dict(expression.denominator), values)
    if denominator == 0:
        raise ZeroDivisionError("An exact spline query reached a zero rational denominator.")
    return numerator / denominator


def _poly_interval(
    polynomial: Polynomial,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    /,
) -> tuple[float, float]:
    dimension = len(lower)
    arguments = tuple(
        _affine_argument(
            Fraction.from_float(float(lower[axis])),
            Fraction.from_float(float(upper[axis])),
            dimension,
            axis,
        )
        for axis in range(dimension)
    )
    restricted = compose(polynomial, arguments, dimension=dimension)
    return polynomial_bounds(restricted, "box", dimension)


def _poly_excludes_zero(polynomial: Polynomial, /) -> bool:
    return not polynomial


def _surface_value_box(
    coordinates: tuple[RationalPolynomial, ...],
    lower: tuple[float, float],
    upper: tuple[float, float],
    /,
) -> tuple[tuple[float, float], ...]:
    arguments = tuple(
        _affine_argument(
            Fraction.from_float(float(lower[axis])),
            Fraction.from_float(float(upper[axis])),
            2,
            axis,
        )
        for axis in range(2)
    )
    return tuple(
        polynomial_bounds(
            RationalPolynomial(
                compose(dict(expression.numerator), arguments, dimension=2),
                compose(dict(expression.denominator), arguments, dimension=2),
            ),
            "box",
            2,
        )
        for expression in coordinates
    )


def _sqrt_bounds(squared: tuple[float, float], /) -> tuple[float, float]:
    lower, upper = squared
    if not isfinite(upper) or upper < 0.0:
        return 0.0, np.inf
    low_value = max(0.0, lower)
    lo = float(np.sqrt(low_value))
    hi = float(np.sqrt(max(0.0, upper)))
    if lo * lo > low_value:
        lo = float(np.nextafter(lo, -np.inf))
    if hi * hi < upper:
        hi = float(np.nextafter(hi, np.inf))
    return max(0.0, float(np.nextafter(lo, -np.inf))), float(np.nextafter(hi, np.inf))


def _distance_bounds(
    expression: RationalPolynomial,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    /,
) -> tuple[float, float]:
    dimension = len(lower)
    arguments = tuple(
        _affine_argument(
            Fraction.from_float(float(lower[axis])),
            Fraction.from_float(float(upper[axis])),
            dimension,
            axis,
        )
        for axis in range(dimension)
    )
    restricted = RationalPolynomial(
        compose(dict(expression.numerator), arguments, dimension=dimension),
        compose(dict(expression.denominator), arguments, dimension=dimension),
    )
    squared = polynomial_bounds(restricted, "box", dimension)
    return _sqrt_bounds(squared)


def _diameter(bounds: tuple[tuple[float, float], ...], /) -> float:
    if any(not isfinite(lower) or not isfinite(upper) for lower, upper in bounds):
        return np.inf
    squared = sum((upper - lower) ** 2 for lower, upper in bounds)
    return float(np.nextafter(np.sqrt(squared), np.inf))


def _point_from_coordinates(
    coordinates: tuple[RationalPolynomial, ...], values: tuple[Fraction, ...], /
) -> np.ndarray:
    return np.asarray([float(_eval_rational(value, values)) for value in coordinates], dtype=np.float64)


def _distance_value_bounds(
    expression: RationalPolynomial, values: tuple[Fraction, ...], /
) -> tuple[float, float]:
    coords = tuple(float(value) for value in values)
    return _distance_bounds(expression, coords, coords)


def _parameter_box(
    piece: _SplinePiece,
    lower: tuple[float, float],
    upper: tuple[float, float],
    /,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    u0, u1, v0, v1 = piece.bounds
    return (
        (
            u0 + (u1 - u0) * Fraction.from_float(lower[0]),
            u0 + (u1 - u0) * Fraction.from_float(upper[0]),
        ),
        (
            v0 + (v1 - v0) * Fraction.from_float(lower[1]),
            v0 + (v1 - v0) * Fraction.from_float(upper[1]),
        ),
    )




def _trim_box_classification(
    model: BRepModel,
    face: int,
    box: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    full_rectangle: Callable[[BRepModel, int], bool],
    /,
) -> int | None:
    lower = np.nextafter(
        np.asarray([float(box[0][0]), float(box[1][0])], dtype=np.float64),
        -np.inf,
    )
    upper = np.nextafter(
        np.asarray([float(box[0][1]), float(box[1][1])], dtype=np.float64),
        np.inf,
    )
    domain = model.trim_domains[face]
    if domain is None:
        if full_rectangle(model, face):
            return 1
        return None
    result = domain.classify_boxes(np.stack((lower, upper))[None, ...])
    if not bool(np.asarray(result.resolved[0])):
        return None
    return int(bool(np.asarray(result.inside[0])))


def _trim_point_classification(
    trim_classify: Callable[[BRepModel, int, np.ndarray], int | None],
    model: BRepModel,
    face: int,
    uv: np.ndarray,
    /,
) -> int | None:
    return trim_classify(model, face, uv)


def _make_candidate(
    point: np.ndarray,
    parameters: np.ndarray,
    lower: float,
    upper: float,
    status: int,
    piece_id: tuple[object, ...],
    root_box: tuple[tuple[Fraction, Fraction], ...],
    exact_parameters: tuple[ExactScalar, ...],
    root_id: tuple[object, ...],
    /,
    *,
    root_system: _PolynomialSystem | None = None,
    root_piece_bounds: _SurfaceBounds | None = None,
    surface_piece_id: tuple[object, ...] | None = None,
    exact_root: bool = False,
    exact_point: tuple[ExactScalar, ...] | None = None,
    root_box_unique: bool = False,
) -> _Candidate:
    return _Candidate(
        point,
        parameters,
        lower,
        upper,
        status,
        piece_id,
        root_box,
        exact_parameters,
        root_id,
        root_system,
        root_piece_bounds,
        root_box_unique,
        surface_piece_id,
        exact_root,
        exact_point,
    )


def _curve_segment(
    piece_id: tuple[object, ...],
    parameter_bounds: tuple[Fraction, Fraction],
    coordinates: tuple[RationalPolynomial, ...],
    /,
    chart_coordinates: tuple[RationalPolynomial, RationalPolynomial] | None = None,
) -> _CurveSegment:
    return _CurveSegment(
        piece_id,
        parameter_bounds,
        coordinates,
        chart_coordinates,
    )


def _curve_query_data(
    segment: _CurveSegment,
    query: tuple[Fraction, ...],
    /,
) -> tuple[RationalPolynomial, Polynomial]:
    deltas = tuple(
        subtract(value, constant(component, 1))
        for value, component in zip(segment.coordinates, query, strict=True)
    )
    distance: object = constant(0, 1)
    stationarity: object = constant(0, 1)
    for coordinate, delta in zip(segment.coordinates, deltas, strict=True):
        distance = add(distance, multiply(delta, delta))
        stationarity = add(stationarity, multiply(delta, derivative(coordinate, 0)))
    distance_polynomial = (
        distance
        if isinstance(distance, RationalPolynomial)
        else RationalPolynomial(distance, constant(1, 1))
    )
    stationarity_polynomial = (
        dict(stationarity.numerator)
        if isinstance(stationarity, RationalPolynomial)
        else dict(stationarity)
    )
    return distance_polynomial, stationarity_polynomial


def _root_coefficients(polynomial: Polynomial, /) -> tuple[Fraction, ...] | None:
    if not polynomial:
        return (Fraction(0),)
    if any(len(index) != 1 for index in polynomial):
        return None
    degree = max(index[0] for index in polynomial)
    values = [Fraction(0) for _ in range(degree + 1)]
    for index, coefficient in polynomial.items():
        if not isinstance(coefficient, Fraction):
            return None
        values[index[0]] = coefficient
    return tuple(values)


def _curve_segments_for_curve(
    curve: AbstractCurve,
    first: Fraction,
    last: Fraction,
    /,
) -> tuple[_CurveSegment, ...] | None:
    if isinstance(curve, BSplineCurve):
        spans = rational_bspline_curve_spans(
            np.asarray(curve.control_points, dtype=np.float64).tolist(),
            np.asarray(curve.weights, dtype=np.float64).tolist(),
            np.asarray(curve.knots, dtype=np.float64).tolist(),
            curve.degree,
            exact_control_points=curve.exact_control_points,
            exact_weights=curve.exact_weights,
            exact_knots=curve.exact_knots,
        )
        segments = []
        for span in spans:
            span_first, span_last = span.parameter_bounds
            if not isinstance(span_first, Fraction) or not isinstance(span_last, Fraction):
                return None
            lower, upper = max(first, span_first), min(last, span_last)
            if lower >= upper:
                continue
            width = span_last - span_first
            a, b = (lower - span_first) / width, (upper - span_first) / width
            homogeneous = tuple(
                compose(
                    _bernstein_curve_power(tuple(row[axis] for row in span.homogeneous_bernstein_coefficients)),
                    (_affine_argument(a, b, 1, 0),),
                    dimension=1,
                )
                for axis in range(curve.ambient_dimension + 1)
            )
            coordinates = _rational_map(homogeneous)
            segments.append(
                _curve_segment(
                    ("bspline-curve", span.span_index, lower, upper),
                    (lower, upper),
                    coordinates,
                )
            )
        return tuple(segments)
    if isinstance(curve, LineCurve):
        origin = curve.origin_exact
        direction = curve.direction_exact
    elif isinstance(curve, ExactLineCurve):
        origin = curve.first_exact
        direction = tuple(last_value - first_value for first_value, last_value in zip(curve.first_exact, curve.last_exact, strict=True))
    else:
        return None
    parameter = _affine_argument(first, last, 1, 0)
    coordinates = tuple(
        RationalPolynomial(
            add(constant(origin[axis], 1), scale(parameter, direction[axis])),
            constant(1, 1),
        )
        for axis in range(curve.ambient_dimension)
    )
    return (_curve_segment(("line-curve", first, last), (first, last), coordinates),)


def _all_curve_candidates(
    segments: tuple[_CurveSegment, ...],
    query: tuple[Fraction, ...],
    policy: BRepProjectionPolicy,
    /,
    *,
    endpoint_status: Callable[[Fraction], int],
) -> tuple[list[_Candidate], list[_Unknown], bool]:
    candidates: list[_Candidate] = []
    unknowns: list[_Unknown] = []
    nonisolated = False
    for segment in segments:
        distance_squared, stationarity = _curve_query_data(segment, query)
        if not stationarity:
            lower, upper = _distance_bounds(distance_squared, (0.0,), (1.0,))
            nonisolated = True
            unknowns.append(
                _Unknown(
                    lower,
                    upper,
                    None,
                    segment.piece_id,
                    ("nonisolated-curve", segment.piece_id),
                )
            )
            continue
        roots, unresolved, continuum = _curve_root_intervals(
            stationarity, segment, distance_squared, policy
        )
        unknowns.extend(unresolved)
        nonisolated |= continuum
        local_candidates = [
            (Fraction(0), Fraction(0), "endpoint", 1),
            (Fraction(1), Fraction(1), "endpoint", 1),
        ]
        local_candidates.extend(
            (bounds[0], bounds[1], "root", multiplicity)
            for bounds, multiplicity, _ in roots
        )
        for lower_local, upper_local, kind, multiplicity in local_candidates:
            middle = (lower_local + upper_local) / 2
            local_float_lower, local_float_upper = _local_float_interval(
                lower_local, upper_local
            )
            distance_lower, distance_upper = _distance_bounds(
                distance_squared,
                (local_float_lower,),
                (local_float_upper,),
            )
            parameter_first, parameter_last = segment.parameter_bounds
            exact_parameter = parameter_first + (parameter_last - parameter_first) * middle
            parameter = float(exact_parameter)
            point = _point_from_coordinates(segment.coordinates, (middle,))
            parameter_box = (
                parameter_first + (parameter_last - parameter_first) * lower_local,
                parameter_first + (parameter_last - parameter_first) * upper_local,
            )
            if segment.chart_coordinates is None:
                exact_parameters = (exact_parameter,)
                parameters = np.asarray((parameter, np.nan), dtype=np.float64)
                selected_status = (
                    endpoint_status(exact_parameter)
                    if kind == "endpoint"
                    else int(BRepProjectionStatus.UNIQUE)
                )
                root_box = (parameter_box,)
            else:
                exact_uv = tuple(
                    _eval_rational(value, (middle,))
                    for value in segment.chart_coordinates
                )
                exact_parameters = exact_uv
                parameters = np.asarray(
                    tuple(float(value) for value in exact_uv), dtype=np.float64
                )
                selected_status = int(BRepProjectionStatus.SEAM)
                root_box = _chart_box(
                    segment, lower_local, upper_local
                )
                if root_box is None:
                    unknowns.append(
                        _Unknown(
                            distance_lower,
                            distance_upper,
                            None,
                            segment.piece_id,
                            (segment.piece_id, "chart-box-unresolved", parameter_box),
                        )
                    )
                    continue
            root_id = (segment.piece_id, kind, parameter_box, multiplicity)
            candidates.append(
                _make_candidate(
                    point,
                    parameters,
                    distance_lower,
                    distance_upper,
                    selected_status,
                    segment.piece_id,
                    root_box,
                    exact_parameters,
                    root_id,
                    exact_root=lower_local == upper_local,
                    exact_point=(
                        tuple(
                            _eval_rational(value, (middle,))
                            for value in segment.coordinates
                        )
                        if lower_local == upper_local
                        else None
                    ),
                )
            )
    return candidates, unknowns, nonisolated


def project_bspline_curve(
    curve: BSplineCurve,
    parameter_range: tuple[float, float],
    query: np.ndarray,
    policy: BRepProjectionPolicy,
    /,
    *,
    edge_range: tuple[float, float],
) -> CurveProjection:
    first, last = (Fraction.from_float(float(value)) for value in parameter_range)
    segments = _curve_segments_for_curve(curve, first, last)
    if segments is None or not segments:
        return CurveProjection(_NAN3.copy(), np.nan, 0.0, np.inf, int(BRepProjectionStatus.UNRESOLVED))
    candidates, unknowns, nonisolated = _all_curve_candidates(
        segments,
        tuple(Fraction.from_float(float(value)) for value in query),
        policy,
        endpoint_status=lambda parameter: int(BRepProjectionStatus.SEAM)
        if parameter in tuple(Fraction.from_float(float(value)) for value in edge_range)
        else int(BRepProjectionStatus.UNIQUE),
    )
    if not candidates:
        if nonisolated and unknowns:
            lower = min(item.lower for item in unknowns)
            upper = min(item.upper for item in unknowns)
            if upper - lower <= policy.ambiguity_tolerance:
                return CurveProjection(
                    _NAN3.copy(),
                    np.nan,
                    lower,
                    upper,
                    int(BRepProjectionStatus.NON_ISOLATED),
                )
            return CurveProjection(
                _NAN3.copy(),
                np.nan,
                lower,
                np.inf,
                int(BRepProjectionStatus.UNRESOLVED),
            )
        return CurveProjection(_NAN3.copy(), np.nan, 0.0, np.inf, int(BRepProjectionStatus.UNRESOLVED))
    best = min(candidates, key=lambda candidate: (candidate.upper, candidate.piece_id, candidate.root_id))
    possible_unknowns = [
        item
        for item in unknowns
        if item.lower <= best.upper + policy.ambiguity_tolerance
    ]
    nonisolated_bounds = [
        item
        for item in possible_unknowns
        if "nonisolated" in repr(item.root_id).lower()
        and item.upper - item.lower <= policy.ambiguity_tolerance
        and item.upper >= best.lower - policy.ambiguity_tolerance
    ]
    if nonisolated_bounds:
        return CurveProjection(
            _NAN3.copy(),
            np.nan,
            min(best.lower, *(item.lower for item in nonisolated_bounds)),
            min(best.upper, *(item.upper for item in nonisolated_bounds)),
            int(BRepProjectionStatus.NON_ISOLATED),
            best.piece_id,
            best.root_id,
        )
    if possible_unknowns:
        return CurveProjection(
            _NAN3.copy(),
            np.nan,
            min(best.lower, *(item.lower for item in possible_unknowns)),
            best.upper,
            int(BRepProjectionStatus.UNRESOLVED),
            best.piece_id,
            best.root_id,
        )
    other_minima = [
        candidate
        for candidate in candidates
        if candidate is not best
        and candidate.lower <= best.upper + policy.ambiguity_tolerance
        and candidate.upper >= best.lower - policy.ambiguity_tolerance
        and not _candidate_is_duplicate(candidate, best, policy)
    ]
    status = int(BRepProjectionStatus.AMBIGUOUS) if other_minima else best.status
    return CurveProjection(
        best.point,
        float(best.parameters[0]),
        min(candidate.lower for candidate in candidates),
        best.upper,
        status,
        best.piece_id,
        best.root_id,
    )



def _curve_segments_for_piece_edge(
    piece: _SplinePiece,
    fixed_axis: int,
    fixed_value: Fraction,
    /,
) -> tuple[_CurveSegment, ...]:
    varying_axis = 1 - fixed_axis
    bounds = piece.bounds
    variable_bounds = (bounds[0], bounds[1]) if varying_axis == 0 else (bounds[2], bounds[3])
    args = [constant(0, 1), constant(0, 1)]
    args[fixed_axis] = constant(Fraction(0) if fixed_value == bounds[fixed_axis * 2] else Fraction(1), 1)
    args[varying_axis] = axes(1)[0]
    homogeneous = tuple(
        compose(value, tuple(args), dimension=1) for value in piece.homogeneous
    )
    coordinates = _rational_map(homogeneous)
    chart_constant = fixed_value
    chart_variable = RationalPolynomial(
        add(constant(variable_bounds[0], 1), scale(axes(1)[0], variable_bounds[1] - variable_bounds[0])),
        constant(1, 1),
    )
    chart = (
        (RationalPolynomial(constant(chart_constant, 1), constant(1, 1)), chart_variable)
        if fixed_axis == 0
        else (chart_variable, RationalPolynomial(constant(chart_constant, 1), constant(1, 1)))
    )
    segment = _curve_segment(
        (piece.piece_id, "piece-edge", fixed_axis, fixed_value),
        variable_bounds,
        coordinates,
        chart_coordinates=chart,
    )
    return (segment,)


def _same_curve_map_on_interval(
    first: _CurveSegment,
    second: _CurveSegment,
    lower: Fraction,
    upper: Fraction,
    /,
) -> bool:
    def restrict(
        segment: _CurveSegment, /
    ) -> tuple[RationalPolynomial, ...] | None:
        first_parameter, last_parameter = segment.parameter_bounds
        if lower < first_parameter or upper > last_parameter:
            return None
        width = last_parameter - first_parameter
        local_lower = (lower - first_parameter) / width
        local_upper = (upper - first_parameter) / width
        argument = (_affine_argument(local_lower, local_upper, 1, 0),)
        return tuple(
            RationalPolynomial(
                compose(dict(value.numerator), argument, dimension=1),
                compose(dict(value.denominator), argument, dimension=1),
            )
            for value in segment.coordinates
        )

    first_restricted = restrict(first)
    second_restricted = restrict(second)
    if (
        first_restricted is None
        or second_restricted is None
        or len(first_restricted) != len(second_restricted)
    ):
        return False
    return all(
        not subtract(
            multiply(dict(first_value.numerator), dict(second_value.denominator)),
            multiply(dict(second_value.numerator), dict(first_value.denominator)),
        )
        for first_value, second_value in zip(
            first_restricted, second_restricted, strict=True
        )
    )


def _pcurve_segments(
    curve: AbstractCurve,
    first: Fraction,
    last: Fraction,
    /,
) -> tuple[_CurveSegment, ...] | None:
    if isinstance(curve, BSplineCurve):
        spans = rational_bspline_curve_spans(
            np.asarray(curve.control_points, dtype=np.float64).tolist(),
            np.asarray(curve.weights, dtype=np.float64).tolist(),
            np.asarray(curve.knots, dtype=np.float64).tolist(),
            curve.degree,
            exact_control_points=curve.exact_control_points,
            exact_weights=curve.exact_weights,
            exact_knots=curve.exact_knots,
        )
        segments = []
        for span in spans:
            span_first, span_last = span.parameter_bounds
            if not isinstance(span_first, Fraction) or not isinstance(span_last, Fraction):
                return None
            lower, upper = max(first, span_first), min(last, span_last)
            if lower >= upper:
                continue
            a, b = (lower - span_first) / (span_last - span_first), (upper - span_first) / (span_last - span_first)
            homogeneous = tuple(
                compose(
                    _bernstein_curve_power(tuple(row[axis] for row in span.homogeneous_bernstein_coefficients)),
                    (_affine_argument(a, b, 1, 0),),
                    dimension=1,
                )
                for axis in range(3)
            )
            coords = _rational_map(homogeneous)
            segments.append(
                _curve_segment(
                    ("pcurve-bspline", span.span_index, lower, upper),
                    (lower, upper),
                    coords,
                    chart_coordinates=(coords[0], coords[1]),
                )
            )
        return tuple(segments)
    if isinstance(curve, LineCurve):
        origin, direction = curve.origin_exact, curve.direction_exact
    elif isinstance(curve, ExactLineCurve):
        origin = curve.first_exact
        direction = tuple(b - a for a, b in zip(curve.first_exact, curve.last_exact, strict=True))
    else:
        return None
    parameter = _affine_argument(first, last, 1, 0)
    coordinates = tuple(
        RationalPolynomial(add(constant(origin[axis], 1), scale(parameter, direction[axis])), constant(1, 1))
        for axis in range(2)
    )
    return (
        _curve_segment(
            ("pcurve-line", first, last),
            (first, last),
            coordinates,
            chart_coordinates=(coordinates[0], coordinates[1]),
        ),
    )




def _surface_candidates(
    piece: _SplinePiece,
    query: tuple[Fraction, Fraction, Fraction],
    policy: BRepProjectionPolicy,
    /,
    *,
    model: BRepModel,
    face: int,
    trim_classify: Callable[[BRepModel, int, np.ndarray], int | None],
    full_rectangle: Callable[[BRepModel, int], bool],
) -> tuple[list[_Candidate], list[_Unknown]]:
    stationary, jacobian, distance_squared = _piece_query_data(piece, query)
    if all(_poly_excludes_zero(value) for value in stationary):
        chart_box = (
            (piece.bounds[0], piece.bounds[1]),
            (piece.bounds[2], piece.bounds[3]),
        )
        if _trim_box_classification(model, face, chart_box, full_rectangle) == 0:
            return [], []
        constant_distance = _constant_rational_value(distance_squared)
        if constant_distance is not None and constant_distance >= 0:
            distance_expression = RationalPolynomial(
                constant(constant_distance, 2), constant(1, 2)
            )
            lower, upper = _distance_bounds(
                distance_expression, (0.0, 0.0), (0.0, 0.0)
            )
        else:
            lower, upper = _distance_bounds(
                distance_squared, (0.0, 0.0), (1.0, 1.0)
            )
        return [], [
            _Unknown(
                lower,
                upper,
                chart_box,
                piece.piece_id,
                ("stationary-set-nonisolated", piece.piece_id),
            )
        ]
    system = _PolynomialSystem(stationary, jacobian)
    pending: list[tuple[tuple[float, float], tuple[float, float], tuple[object, ...]]] = [
        ((0.0, 0.0), (1.0, 1.0), ())
    ]
    candidates: list[_Candidate] = []
    unknowns: list[_Unknown] = []

    def enqueue_certified_neighborhood(
        lower: tuple[float, float],
        upper: tuple[float, float],
        branch: tuple[object, ...],
    ) -> bool:
        if ("certified-neighborhood",) in branch:
            return False
        spans = (
            piece.bounds[1] - piece.bounds[0],
            piece.bounds[3] - piece.bounds[2],
        )
        margins = (
            max(
                upper[0] - lower[0],
                float(policy.parametric_tolerance / spans[0]),
            ),
            max(
                upper[1] - lower[1],
                float(policy.parametric_tolerance / spans[1]),
            ),
        )
        expanded_lower = (
            max(0.0, lower[0] - margins[0]),
            max(0.0, lower[1] - margins[1]),
        )
        expanded_upper = (
            min(1.0, upper[0] + margins[0]),
            min(1.0, upper[1] + margins[1]),
        )
        if expanded_lower == lower and expanded_upper == upper:
            return False
        proof = krawczyk(
            system,
            np.asarray((expanded_lower,)),
            np.asarray((expanded_upper,)),
        )
        if not bool(proof.certified[0]):
            return False
        expanded_branch: tuple[object, ...] = (
            *branch,
            ("certified-neighborhood",),
        )
        pending.append((expanded_lower, expanded_upper, expanded_branch))
        return True

    while pending:
        lower, upper, branch = pending.pop()
        excluded = any(
            (bounds := _poly_interval(value, lower, upper))[0] > 0.0
            or bounds[1] < 0.0
            for value in stationary
        )
        if excluded:
            continue
        proof = krawczyk(system, np.asarray((lower,)), np.asarray((upper,)))
        if bool(proof.excluded[0]):
            continue
        root_lower = (float(proof.lower[0, 0]), float(proof.lower[0, 1]))
        root_upper = (float(proof.upper[0, 0]), float(proof.upper[0, 1]))
        certified = bool(proof.certified[0])
        if certified:
            contract_lower, contract_upper = root_lower, root_upper
            while True:
                global_box = _parameter_box(piece, contract_lower, contract_upper)
                physical_bounds = _surface_value_box(
                    piece.coordinates,
                    contract_lower,
                    contract_upper,
                )
                distance_lower, distance_upper = _distance_bounds(
                    distance_squared,
                    contract_lower,
                    contract_upper,
                )
                if (
                    max(global_box[0][1] - global_box[0][0], global_box[1][1] - global_box[1][0])
                    <= Fraction.from_float(float(policy.parametric_tolerance))
                    and _diameter(physical_bounds) <= policy.ambiguity_tolerance
                    and distance_upper - distance_lower <= policy.ambiguity_tolerance
                ):
                    break
                refined = krawczyk(system, np.asarray((contract_lower,)), np.asarray((contract_upper,)))
                # The existing certified box already proves there is exactly one
                # root.  Intersecting with the Krawczyk image is a safe enclosure
                # refinement even if roundoff prevents a second strict inclusion.
                next_lower = (
                    max(contract_lower[0], float(refined.lower[0, 0])),
                    max(contract_lower[1], float(refined.lower[0, 1])),
                )
                next_upper = (
                    min(contract_upper[0], float(refined.upper[0, 0])),
                    min(contract_upper[1], float(refined.upper[0, 1])),
                )
                if any(next_lower[axis] > next_upper[axis] for axis in range(2)):
                    certified = False
                    break
                if next_lower == contract_lower and next_upper == contract_upper:
                    certified = False
                    break
                contract_lower, contract_upper = next_lower, next_upper
            if certified:
                global_box = _parameter_box(piece, contract_lower, contract_upper)
                trim_box = _trim_box_classification(model, face, global_box, full_rectangle)
                uv_exact = (
                    (global_box[0][0] + global_box[0][1]) / 2,
                    (global_box[1][0] + global_box[1][1]) / 2,
                )
                uv_float = np.asarray([float(value) for value in uv_exact], dtype=np.float64)
                point = _point_from_coordinates(piece.coordinates, tuple(Fraction.from_float(float(value)) for value in contract_lower))
                # The published representative and exact bound use the same in-domain parameter point.
                midpoint_local = tuple(
                    (Fraction.from_float(contract_lower[axis]) + Fraction.from_float(contract_upper[axis])) / 2
                    for axis in range(2)
                )
                exact_stationary = all(
                    _eval_polynomial(value, midpoint_local) == 0
                    for value in stationary
                )
                point = _point_from_coordinates(piece.coordinates, midpoint_local)
                exact_distance = _distance_value_bounds(distance_squared, midpoint_local)
                lower_distance = min(distance_lower, exact_distance[0])
                upper_distance = exact_distance[1]
                root_box = global_box
                root_id = (piece.piece_id, "stationary", root_box, branch)
                if trim_box == 0:
                    continue
                if trim_box is None:
                    unknowns.append(
                        _Unknown(
                            lower_distance,
                            upper_distance,
                            global_box,
                            piece.piece_id,
                            root_id,
                            surface_piece_id=piece.piece_id,
                            exact_parameters=uv_exact,
                            exact_root=exact_stationary,
                            root_system=system,
                            root_piece_bounds=piece.bounds,
                        )
                    )
                    continue
                trim_state = _trim_point_classification(trim_classify, model, face, uv_float)
                if trim_state == 0:
                    continue
                if trim_state is None:
                    unknowns.append(
                        _Unknown(
                            lower_distance,
                            upper_distance,
                            global_box,
                            piece.piece_id,
                            root_id,
                            surface_piece_id=piece.piece_id,
                            exact_parameters=uv_exact,
                            exact_root=exact_stationary,
                            root_system=system,
                            root_piece_bounds=piece.bounds,
                        )
                    )
                    continue
                normal_bounds = _poly_interval(piece.normal_squared, contract_lower, contract_upper)
                if normal_bounds[1] == 0.0:
                    status = int(BRepProjectionStatus.SINGULAR)
                elif normal_bounds[0] > 0.0:
                    status = int(BRepProjectionStatus.SEAM) if trim_state < 0 else int(BRepProjectionStatus.UNIQUE)
                else:
                    status = int(BRepProjectionStatus.UNRESOLVED)
                    unknowns.append(
                        _Unknown(
                            lower_distance,
                            upper_distance,
                            global_box,
                            piece.piece_id,
                            (root_id, "normal-rank-unresolved"),
                            surface_piece_id=piece.piece_id,
                            root_system=system,
                            root_piece_bounds=piece.bounds,
                        )
                    )
                    continue
                exact_stationary = all(
                    _eval_polynomial(value, midpoint_local) == 0
                    for value in stationary
                )
                exact_point = (
                    tuple(
                        _eval_rational(value, midpoint_local)
                        for value in piece.coordinates
                    )
                    if exact_stationary
                    else None
                )
                candidates.append(
                    _make_candidate(
                        point,
                        np.asarray((float(uv_exact[0]), float(uv_exact[1])), dtype=np.float64),
                        lower_distance,
                        upper_distance,
                        status,
                        piece.piece_id,
                        root_box,
                        uv_exact,
                        root_id,
                        root_system=system,
                        root_piece_bounds=piece.bounds,
                        root_box_unique=True,
                        surface_piece_id=piece.piece_id,
                        exact_root=exact_stationary,
                        exact_point=exact_point,
                    )
                )
                continue
        chart_box = _parameter_box(piece, lower, upper)
        physical_bounds = _surface_value_box(piece.coordinates, lower, upper)
        physical_diameter = _diameter(physical_bounds)
        chart_width = max(
            chart_box[0][1] - chart_box[0][0],
            chart_box[1][1] - chart_box[1][0],
        )
        if (
            chart_width <= Fraction.from_float(float(policy.parametric_tolerance))
            or physical_diameter <= policy.ambiguity_tolerance
        ):
            trim_box = _trim_box_classification(model, face, chart_box, full_rectangle)
            if trim_box == 0:
                continue
            if enqueue_certified_neighborhood(lower, upper, branch):
                continue
            lower_distance, upper_distance = _distance_bounds(distance_squared, lower, upper)
            unknowns.append(
                _Unknown(
                    lower_distance,
                    upper_distance,
                    chart_box,
                    piece.piece_id,
                    (piece.piece_id, "unresolved", chart_box, branch),
                    surface_piece_id=piece.piece_id,
                    root_system=system,
                    root_piece_bounds=piece.bounds,
                )
            )
            continue
        widths = ((upper[0] - lower[0]) * float(piece.bounds[1] - piece.bounds[0]), (upper[1] - lower[1]) * float(piece.bounds[3] - piece.bounds[2]))
        axis = int(widths[1] > widths[0])
        midpoint = 0.5 * lower[axis] + 0.5 * upper[axis]
        if not lower[axis] < midpoint < upper[axis]:
            global_box = _parameter_box(piece, lower, upper)
            trim_box = _trim_box_classification(model, face, global_box, full_rectangle)
            if trim_box == 0:
                continue
            if enqueue_certified_neighborhood(lower, upper, branch):
                continue
            distance_lower, distance_upper = _distance_bounds(distance_squared, lower, upper)
            unknowns.append(
                _Unknown(
                    distance_lower,
                    distance_upper,
                    global_box,
                    piece.piece_id,
                    (piece.piece_id, "binary64-split-resolution", branch),
                    surface_piece_id=piece.piece_id,
                    root_system=system,
                    root_piece_bounds=piece.bounds,
                )
            )
            continue
        split_upper = float(np.nextafter(midpoint, np.inf))
        split_lower = float(np.nextafter(midpoint, -np.inf))
        left_upper = (
            (split_upper, upper[1])
            if axis == 0
            else (upper[0], split_upper)
        )
        right_lower = (
            (split_lower, lower[1])
            if axis == 0
            else (lower[0], split_lower)
        )
        left_branch: tuple[object, ...] = (
            *branch,
            (axis, 0, Fraction.from_float(midpoint)),
        )
        right_branch: tuple[object, ...] = (
            *branch,
            (axis, 1, Fraction.from_float(midpoint)),
        )
        pending.append((right_lower, upper, right_branch))
        pending.append((lower, left_upper, left_branch))
    return candidates, unknowns


def _piece_boundary_candidates(
    pieces: tuple[_SplinePiece, ...],
    query: tuple[Fraction, Fraction, Fraction],
    policy: BRepProjectionPolicy,
    /,
    *,
    model: BRepModel,
    face: int,
    full_rectangle: Callable[[BRepModel, int], bool],
) -> tuple[list[_Candidate], list[_Unknown]]:
    candidates: list[_Candidate] = []
    unknowns: list[_Unknown] = []

    stationary_systems: dict[tuple[object, ...], _PolynomialSystem] = {}

    def system_for(piece: _SplinePiece, /) -> _PolynomialSystem:
        system = stationary_systems.get(piece.piece_id)
        if system is None:
            values, jacobian, _ = _piece_query_data(piece, query)
            system = _PolynomialSystem(values, jacobian)
            stationary_systems[piece.piece_id] = system
        return system

    def has_inward_descent(
        parameters: tuple[ExactScalar, ExactScalar],
        exact_point: tuple[ExactScalar, ...],
        /,
    ) -> bool:
        if not isinstance(parameters[0], Fraction) or not isinstance(
            parameters[1], Fraction
        ):
            return False
        rational_parameters = parameters[0], parameters[1]
        for adjacent in pieces:
            u0, u1, v0, v1 = adjacent.bounds
            if not (
                u0 <= rational_parameters[0] <= u1
                and v0 <= rational_parameters[1] <= v1
            ):
                continue
            local = (
                (rational_parameters[0] - u0) / (u1 - u0),
                (rational_parameters[1] - v0) / (v1 - v0),
            )
            adjacent_point = tuple(
                _eval_rational(value, local) for value in adjacent.coordinates
            )
            if adjacent_point != exact_point:
                continue
            stationary = system_for(adjacent)._values
            for axis, lower, upper in (
                (0, u0, u1),
                (1, v0, v1),
            ):
                derivative_value = _eval_polynomial(stationary[axis], local)
                if (
                    lower < rational_parameters[axis] < upper
                    and derivative_value != 0
                ):
                    return True
                if rational_parameters[axis] == lower and derivative_value < 0:
                    return True
                if rational_parameters[axis] == upper and derivative_value > 0:
                    return True
        return False
    def has_edge_root_descent(
        piece: _SplinePiece,
        fixed_axis: int,
        fixed_value: Fraction,
        edge_segment: _CurveSegment,
        parameter_interval: tuple[Fraction, Fraction],
        /,
    ) -> bool:
        for adjacent in pieces:
            bounds = adjacent.bounds
            fixed_lower = bounds[fixed_axis * 2]
            fixed_upper = bounds[fixed_axis * 2 + 1]
            if fixed_value not in (fixed_lower, fixed_upper):
                continue
            varying_axis = 1 - fixed_axis
            varying_lower = bounds[varying_axis * 2]
            varying_upper = bounds[varying_axis * 2 + 1]
            if (
                parameter_interval[0] < varying_lower
                or parameter_interval[1] > varying_upper
            ):
                continue
            adjacent_segment = (
                edge_segment
                if adjacent.piece_id == piece.piece_id
                else _curve_segments_for_piece_edge(
                    adjacent, fixed_axis, fixed_value
                )[0]
            )
            if not _same_curve_map_on_interval(
                edge_segment,
                adjacent_segment,
                parameter_interval[0],
                parameter_interval[1],
            ):
                continue
            fixed_local = (fixed_value - fixed_lower) / (
                fixed_upper - fixed_lower
            )
            varying_local = (
                (parameter_interval[0] - varying_lower)
                / (varying_upper - varying_lower),
                (parameter_interval[1] - varying_lower)
                / (varying_upper - varying_lower),
            )
            fixed_float_lower, fixed_float_upper = _local_float_interval(
                fixed_local, fixed_local
            )
            varying_float_lower, varying_float_upper = _local_float_interval(
                *varying_local
            )
            local_lower = (
                (fixed_float_lower, varying_float_lower)
                if fixed_axis == 0
                else (varying_float_lower, fixed_float_lower)
            )
            local_upper = (
                (fixed_float_upper, varying_float_upper)
                if fixed_axis == 0
                else (varying_float_upper, fixed_float_upper)
            )
            stationary = system_for(adjacent)._values
            derivative_bounds = _poly_interval(
                stationary[fixed_axis], local_lower, local_upper
            )
            if fixed_value == fixed_lower and derivative_bounds[1] < 0.0:
                return True
            if fixed_value == fixed_upper and derivative_bounds[0] > 0.0:
                return True
        return False

    for piece in pieces:
        u0, u1, v0, v1 = piece.bounds
        surface_system = system_for(piece)
        surface_stationary = surface_system._values
        for fixed_axis, fixed_value in ((0, u0), (0, u1), (1, v0), (1, v1)):
            segment = _curve_segments_for_piece_edge(
                piece, fixed_axis, fixed_value
            )[0]
            distance_squared, stationarity = _curve_query_data(segment, query)
            chart_coordinates = segment.chart_coordinates
            if chart_coordinates is None:
                lower, upper = _distance_bounds(distance_squared, (0.0,), (1.0,))
                unknowns.append(
                    _Unknown(
                        lower,
                        upper,
                        None,
                        segment.piece_id,
                        (segment.piece_id, "missing-edge-chart"),
                    )
                )
                continue
            local_edge_coordinates = (
                _chart_coordinate_to_local(chart_coordinates[0], u0, u1),
                _chart_coordinate_to_local(chart_coordinates[1], v0, v1),
            )
            restricted_normal = compose(
                surface_stationary[fixed_axis],
                local_edge_coordinates,
                dimension=1,
            )
            normal_stationarity_zero = (
                not restricted_normal.numerator
                if isinstance(restricted_normal, RationalPolynomial)
                else not restricted_normal
            )
            roots, unresolved, continuum = _curve_root_intervals(stationarity, segment, distance_squared, policy)
            unknowns.extend(unresolved)
            if continuum:
                lower, upper = _distance_bounds(distance_squared, (0.0,), (1.0,))
                unknowns.append(_Unknown(lower, upper, None, segment.piece_id, (segment.piece_id, "nonisolated")))
                continue
            local_candidates = [(Fraction(0), Fraction(0), "endpoint", 1), (Fraction(1), Fraction(1), "endpoint", 1)]
            local_candidates.extend((bounds[0], bounds[1], "root", multiplicity) for bounds, multiplicity, _ in roots)
            for local_lower, local_upper, kind, multiplicity in local_candidates:
                middle = (local_lower + local_upper) / 2
                chart_exact = (
                    _eval_rational(chart_coordinates[0], (middle,)),
                    _eval_rational(chart_coordinates[1], (middle,)),
                )
                uv = np.asarray([float(value) for value in chart_exact], dtype=np.float64)
                local_float_lower, local_float_upper = _local_float_interval(
                    local_lower, local_upper
                )
                distance_lower, distance_upper = _distance_bounds(
                    distance_squared,
                    (local_float_lower,),
                    (local_float_upper,),
                )
                point = _point_from_coordinates(segment.coordinates, (middle,))
                root_box = _chart_box(segment, local_lower, local_upper)
                parameter_box = (
                    segment.parameter_bounds[0]
                    + (segment.parameter_bounds[1] - segment.parameter_bounds[0])
                    * local_lower,
                    segment.parameter_bounds[0]
                    + (segment.parameter_bounds[1] - segment.parameter_bounds[0])
                    * local_upper,
                )
                root_id = (segment.piece_id, kind, parameter_box, multiplicity)
                if isinstance(chart_exact[0], Fraction) and isinstance(
                    chart_exact[1], Fraction
                ):
                    local_exact = (
                        (chart_exact[0] - u0) / (u1 - u0),
                        (chart_exact[1] - v0) / (v1 - v0),
                    )
                else:
                    local_exact = None
                exact_edge_point = (
                    tuple(
                        _eval_rational(value, (middle,))
                        for value in segment.coordinates
                    )
                    if local_lower == local_upper
                    else None
                )
                full_stationary_at_exact = (
                    exact_edge_point is not None
                    and local_exact is not None
                    and all(
                        _eval_polynomial(value, local_exact) == 0
                        for value in surface_stationary
                    )
                )
                edge_root_system = (
                    surface_system
                    if full_stationary_at_exact
                    or (kind == "root" and normal_stationarity_zero)
                    else None
                )
                if root_box is None:
                    unknowns.append(
                        _Unknown(
                            distance_lower,
                            distance_upper,
                            None,
                            segment.piece_id,
                            (root_id, "chart-box-unresolved"),
                            surface_piece_id=piece.piece_id,
                            exact_parameters=chart_exact,
                            exact_root=local_lower == local_upper,
                            root_system=edge_root_system,
                            root_piece_bounds=(
                                piece.bounds
                                if edge_root_system is not None
                                else None
                            ),
                        )
                    )
                    continue
                trim = _trim_box_classification(
                    model, face, root_box, full_rectangle
                )
                if trim == 0:
                    continue
                if trim is None:
                    unknowns.append(
                        _Unknown(
                            distance_lower,
                            distance_upper,
                            root_box,
                            segment.piece_id,
                            root_id,
                            surface_piece_id=piece.piece_id,
                            exact_parameters=chart_exact,
                            exact_root=local_lower == local_upper,
                            root_system=edge_root_system,
                            root_piece_bounds=(
                                piece.bounds
                                if edge_root_system is not None
                                else None
                            ),
                        )
                    )
                    continue
                if (
                    (
                        exact_edge_point is not None
                        and has_inward_descent(chart_exact, exact_edge_point)
                    )
                    or has_edge_root_descent(
                        piece,
                        fixed_axis,
                        fixed_value,
                        segment,
                        parameter_box,
                    )
                ):
                    continue
                candidates.append(
                    _make_candidate(
                        point,
                        uv,
                        distance_lower,
                        distance_upper,
                        int(BRepProjectionStatus.UNIQUE),
                        segment.piece_id,
                        root_box,
                        chart_exact,
                        root_id,
                        surface_piece_id=piece.piece_id,
                        exact_root=local_lower == local_upper,
                        exact_point=exact_edge_point,
                        root_system=edge_root_system,
                        root_piece_bounds=(
                            piece.bounds
                            if edge_root_system is not None
                            else None
                        ),
                    )
                )
    return candidates, unknowns


def _trim_boundary_candidates(
    model: BRepModel,
    face: int,
    pieces: tuple[_SplinePiece, ...],
    query: tuple[Fraction, Fraction, Fraction],
    policy: BRepProjectionPolicy,
    /,
) -> tuple[list[_Candidate], list[_Unknown]]:
    segments = _face_trim_segments(model, face, pieces, policy)
    if segments is None:
        return [], [_Unknown(0.0, np.inf, None, ("trim", face), ("unsupported-trim-piece-composition", face))]
    candidates: list[_Candidate] = []
    unknowns: list[_Unknown] = []
    for segment, _piece in segments:
        distance_squared, stationarity = _curve_query_data(segment, query)
        chart_coordinates = segment.chart_coordinates
        if chart_coordinates is None:
            lower, upper = _distance_bounds(distance_squared, (0.0,), (1.0,))
            unknowns.append(
                _Unknown(
                    lower,
                    upper,
                    None,
                    segment.piece_id,
                    (segment.piece_id, "missing-trim-chart"),
                )
            )
            continue
        roots, unresolved, continuum = _curve_root_intervals(stationarity, segment, distance_squared, policy)
        unknowns.extend(unresolved)
        if continuum:
            lower, upper = _distance_bounds(distance_squared, (0.0,), (1.0,))
            unknowns.append(_Unknown(lower, upper, None, segment.piece_id, (segment.piece_id, "nonisolated-trim")))
            continue
        local_candidates = [(Fraction(0), Fraction(0), "endpoint", 1), (Fraction(1), Fraction(1), "endpoint", 1)]
        local_candidates.extend((bounds[0], bounds[1], "root", multiplicity) for bounds, multiplicity, _ in roots)
        for local_lower, local_upper, kind, multiplicity in local_candidates:
            middle = (local_lower + local_upper) / 2
            chart_exact = (
                _eval_rational(chart_coordinates[0], (middle,)),
                _eval_rational(chart_coordinates[1], (middle,)),
            )
            point = _point_from_coordinates(segment.coordinates, (middle,))
            local_float_lower, local_float_upper = _local_float_interval(
                local_lower, local_upper
            )
            distance_lower, distance_upper = _distance_bounds(
                distance_squared,
                (local_float_lower,),
                (local_float_upper,),
            )
            parameter_box = (
                segment.parameter_bounds[0]
                + (segment.parameter_bounds[1] - segment.parameter_bounds[0])
                * local_lower,
                segment.parameter_bounds[0]
                + (segment.parameter_bounds[1] - segment.parameter_bounds[0])
                * local_upper,
            )
            root_box = _chart_box(segment, local_lower, local_upper)
            root_id = (segment.piece_id, kind, parameter_box, multiplicity)
            if root_box is None:
                unknowns.append(
                    _Unknown(
                        distance_lower,
                        distance_upper,
                        None,
                        segment.piece_id,
                        (root_id, "chart-box-unresolved"),
                        surface_piece_id=_piece.piece_id,
                        exact_parameters=chart_exact,
                        exact_root=local_lower == local_upper,
                    )
                )
                continue
            candidates.append(
                _make_candidate(
                    point,
                    np.asarray([float(value) for value in chart_exact], dtype=np.float64),
                    distance_lower,
                    distance_upper,
                    int(BRepProjectionStatus.SEAM),
                    segment.piece_id,
                    root_box,
                    chart_exact,
                    root_id,
                    surface_piece_id=_piece.piece_id,
                    exact_root=local_lower == local_upper,
                    exact_point=(
                        tuple(
                            _eval_rational(value, (middle,))
                            for value in segment.coordinates
                        )
                        if local_lower == local_upper
                        else None
                    ),
                )
            )
    return candidates, unknowns


def _same_root_system(
    first: _PolynomialSystem | None,
    second: _PolynomialSystem | None,
    /,
) -> bool:
    return first is second or (
        first is not None
        and second is not None
        and first._values == second._values
        and first._jacobian == second._jacobian
    )


def _candidate_is_duplicate(
    first: _Candidate,
    second: _Candidate,
    policy: BRepProjectionPolicy,
    /,
) -> bool:
    if first.piece_id == second.piece_id and first.root_id == second.root_id:
        return True
    # Parameter and point representatives are not root identity evidence.
    if (
        first.exact_root
        and second.exact_root
        and first.exact_parameters == second.exact_parameters
        and first.exact_point is not None
        and first.exact_point == second.exact_point
        and first.surface_piece_id is not None
        and first.surface_piece_id == second.surface_piece_id
    ):
        return True
    if (
        first.root_system is None
        or second.root_system is None
        or not _same_root_system(first.root_system, second.root_system)
        or first.surface_piece_id is None
        or first.surface_piece_id != second.surface_piece_id
        or first.root_piece_bounds is None
        or first.root_piece_bounds != second.root_piece_bounds
        or len(first.root_box) != 2
        or len(second.root_box) != 2
    ):
        return False
    if first.root_box_unique and second.root_box_unique and (
        all(
            a[0] <= b[0] and b[1] <= a[1]
            for a, b in zip(first.root_box, second.root_box, strict=True)
        )
        or all(
            b[0] <= a[0] and a[1] <= b[1]
            for a, b in zip(first.root_box, second.root_box, strict=True)
        )
    ):
        return True
    # Both root enclosures lie in this union. A Krawczyk inclusion proves
    # they contain the same unique root, even when a root is on a piece edge.
    union = (
        (
            min(first.root_box[0][0], second.root_box[0][0]),
            max(first.root_box[0][1], second.root_box[0][1]),
        ),
        (
            min(first.root_box[1][0], second.root_box[1][0]),
            max(first.root_box[1][1], second.root_box[1][1]),
        ),
    )
    u0, u1, v0, v1 = first.root_piece_bounds
    spans = (u1 - u0, v1 - v0)
    local_lower = (
        (union[0][0] - u0) / spans[0],
        (union[1][0] - v0) / spans[1],
    )
    local_upper = (
        (union[0][1] - u0) / spans[0],
        (union[1][1] - v0) / spans[1],
    )
    lower_u, upper_u = _unclamped_float_interval(local_lower[0], local_upper[0])
    lower_v, upper_v = _unclamped_float_interval(local_lower[1], local_upper[1])
    margins = (
        max(upper_u - lower_u, float(policy.parametric_tolerance / spans[0])),
        max(upper_v - lower_v, float(policy.parametric_tolerance / spans[1])),
    )
    lower = np.asarray((lower_u - margins[0], lower_v - margins[1]), dtype=np.float64)
    upper = np.asarray((upper_u + margins[0], upper_v + margins[1]), dtype=np.float64)
    return bool(
        krawczyk(first.root_system, lower[None, :], upper[None, :]).certified[0]
    )


def _unknown_is_same_candidate_root(
    unknown: _Unknown,
    candidate: _Candidate,
    policy: BRepProjectionPolicy,
    /,
) -> bool:
    distance_bounds_overlap = (
        candidate.lower <= unknown.upper + policy.ambiguity_tolerance
        and candidate.upper >= unknown.lower - policy.ambiguity_tolerance
    )
    if not distance_bounds_overlap:
        return False
    if (
        unknown.exact_root
        and candidate.exact_root
        and unknown.surface_piece_id is not None
        and unknown.surface_piece_id == candidate.surface_piece_id
        and unknown.exact_parameters == candidate.exact_parameters
    ):
        return True
    if (
        unknown.chart_box is None
        or candidate.root_system is None
        or not _same_root_system(unknown.root_system, candidate.root_system)
        or unknown.surface_piece_id is None
        or unknown.surface_piece_id != candidate.surface_piece_id
        or unknown.root_piece_bounds is None
        or unknown.root_piece_bounds != candidate.root_piece_bounds
        or len(candidate.root_box) != 2
    ):
        return False
    # A candidate may cover an unknown region only by containment in its
    # unique-root box or by a Krawczyk inclusion on their full union.
    if candidate.root_box_unique and all(
        candidate.root_box[axis][0] <= unknown.chart_box[axis][0]
        and unknown.chart_box[axis][1] <= candidate.root_box[axis][1]
        for axis in range(2)
    ):
        return True
    u0, u1, v0, v1 = unknown.root_piece_bounds
    union = (
        (
            min(unknown.chart_box[0][0], candidate.root_box[0][0]),
            max(unknown.chart_box[0][1], candidate.root_box[0][1]),
        ),
        (
            min(unknown.chart_box[1][0], candidate.root_box[1][0]),
            max(unknown.chart_box[1][1], candidate.root_box[1][1]),
        ),
    )
    spans = (u1 - u0, v1 - v0)
    local_lower = (
        (union[0][0] - u0) / spans[0],
        (union[1][0] - v0) / spans[1],
    )
    local_upper = (
        (union[0][1] - u0) / spans[0],
        (union[1][1] - v0) / spans[1],
    )
    if any(local_lower[axis] > local_upper[axis] for axis in range(2)):
        return False
    lower_u, upper_u = _unclamped_float_interval(local_lower[0], local_upper[0])
    lower_v, upper_v = _unclamped_float_interval(local_lower[1], local_upper[1])
    source_spans = (spans[0], spans[1])
    margins = (
        max(
            upper_u - lower_u,
            float(policy.parametric_tolerance / source_spans[0]),
        ),
        max(
            upper_v - lower_v,
            float(policy.parametric_tolerance / source_spans[1]),
        ),
    )
    lower = np.asarray(
        (lower_u - margins[0], lower_v - margins[1]),
        dtype=np.float64,
    )
    upper = np.asarray(
        (upper_u + margins[0], upper_v + margins[1]),
        dtype=np.float64,
    )
    return bool(
        krawczyk(candidate.root_system, lower[None, :], upper[None, :]).certified[0]
    )


def _unique_candidates(
    values: list[_Candidate], policy: BRepProjectionPolicy, /
) -> list[_Candidate]:
    result: list[_Candidate] = []
    for candidate in values:
        duplicate = next(
            (
                row
                for row, prior in enumerate(result)
                if _candidate_is_duplicate(candidate, prior, policy)
            ),
            None,
        )
        if duplicate is None:
            result.append(candidate)
        elif candidate.upper < result[duplicate].upper:
            result[duplicate] = candidate
    return result


def project_spline_face(
    model: BRepModel,
    face: int,
    query: np.ndarray,
    policy: BRepProjectionPolicy,
    /,
    *,
    trim_classify: Callable[[BRepModel, int, np.ndarray], int | None],
    full_rectangle: Callable[[BRepModel, int], bool],
) -> SplineProjection:
    pieces = _surface_pieces(model, face)
    if not pieces:
        return SplineProjection(_NAN3.copy(), _NAN2.copy(), 0.0, np.inf, int(BRepProjectionStatus.UNRESOLVED))
    exact_query = (
        Fraction.from_float(float(query[0])),
        Fraction.from_float(float(query[1])),
        Fraction.from_float(float(query[2])),
    )
    candidates: list[_Candidate] = []
    unknowns: list[_Unknown] = []
    for piece in pieces:
        interior, unresolved = _surface_candidates(
            piece,
            exact_query,
            policy,
            model=model,
            face=face,
            trim_classify=trim_classify,
            full_rectangle=full_rectangle,
        )
        candidates.extend(interior)
        unknowns.extend(unresolved)
    boundary, boundary_unknowns = _piece_boundary_candidates(
        pieces,
        exact_query,
        policy,
        model=model,
        face=face,
        full_rectangle=full_rectangle,
    )
    candidates.extend(boundary)
    unknowns.extend(boundary_unknowns)
    trim_boundary, trim_unknowns = _trim_boundary_candidates(
        model, face, pieces, exact_query, policy
    )
    candidates.extend(trim_boundary)
    unknowns.extend(trim_unknowns)
    candidates = _unique_candidates(candidates, policy)
    if not candidates:
        nonisolated = [
            value
            for value in unknowns
            if "nonisolated" in repr(value.root_id).lower()
            and value.upper - value.lower <= policy.ambiguity_tolerance
        ]
        if nonisolated:
            return SplineProjection(
                _NAN3.copy(),
                _NAN2.copy(),
                min(value.lower for value in nonisolated),
                min(value.upper for value in nonisolated),
                int(BRepProjectionStatus.NON_ISOLATED),
            )
        if unknowns:
            return SplineProjection(_NAN3.copy(), _NAN2.copy(), min(value.lower for value in unknowns), np.inf, int(BRepProjectionStatus.UNRESOLVED))
        return SplineProjection(_NAN3.copy(), _NAN2.copy(), 0.0, np.inf, int(BRepProjectionStatus.UNRESOLVED))
    best = min(candidates, key=lambda value: (value.upper, value.piece_id, value.root_id))
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            if (
                first.root_system is None
                or first.root_system is not second.root_system
                or first.root_piece_bounds != second.root_piece_bounds
                or len(first.root_box) != 2
                or len(second.root_box) != 2
                or _candidate_is_duplicate(first, second, policy)
            ):
                continue
            overlaps = all(
                max(a[0], b[0]) <= min(a[1], b[1])
                for a, b in zip(first.root_box, second.root_box, strict=True)
            )
            competes = (
                first.lower <= best.upper + policy.ambiguity_tolerance
                and second.lower <= best.upper + policy.ambiguity_tolerance
                and first.upper >= best.lower - policy.ambiguity_tolerance
                and second.upper >= best.lower - policy.ambiguity_tolerance
            )
            if overlaps and competes:
                return SplineProjection(
                    _NAN3.copy(),
                    _NAN2.copy(),
                    min(first.lower, second.lower, best.lower),
                    min(first.upper, second.upper, best.upper),
                    int(BRepProjectionStatus.UNRESOLVED),
                    best.piece_id,
                    best.root_id,
                )
    possible_unknowns = []
    for unknown in unknowns:
        if unknown.lower > best.upper + policy.ambiguity_tolerance:
            continue
        if any(
            _unknown_is_same_candidate_root(unknown, candidate, policy)
            for candidate in candidates
        ):
            continue
        possible_unknowns.append(unknown)
    nonisolated = [
        value
        for value in possible_unknowns
        if "nonisolated" in repr(value.root_id).lower()
        and value.upper - value.lower <= policy.ambiguity_tolerance
        and value.lower <= best.upper + policy.ambiguity_tolerance
        and value.upper >= best.lower - policy.ambiguity_tolerance
    ]
    if nonisolated:
        return SplineProjection(
            _NAN3.copy(),
            _NAN2.copy(),
            min(best.lower, *(value.lower for value in nonisolated)),
            min(best.upper, *(value.upper for value in nonisolated)),
            int(BRepProjectionStatus.NON_ISOLATED),
            best.piece_id,
            best.root_id,
        )
    if possible_unknowns:
        return SplineProjection(
            _NAN3.copy(),
            _NAN2.copy(),
            min(best.lower, *(value.lower for value in possible_unknowns)),
            best.upper,
            int(BRepProjectionStatus.UNRESOLVED),
            best.piece_id,
            best.root_id,
        )
    co_minima = [
        candidate
        for candidate in candidates
        if candidate is not best
        and candidate.lower <= best.upper + policy.ambiguity_tolerance
        and candidate.upper >= best.lower - policy.ambiguity_tolerance
        and not _candidate_is_duplicate(candidate, best, policy)
    ]
    if co_minima:
        return SplineProjection(
            best.point,
            best.parameters,
            min(candidate.lower for candidate in candidates),
            best.upper,
            int(BRepProjectionStatus.AMBIGUOUS),
            best.piece_id,
            best.root_id,
        )
    status = best.status
    if status not in (
        int(BRepProjectionStatus.SEAM),
        int(BRepProjectionStatus.SINGULAR),
        int(BRepProjectionStatus.AMBIGUOUS),
    ):
        trim = trim_classify(model, face, best.parameters)
        if trim is None:
            return SplineProjection(_NAN3.copy(), _NAN2.copy(), best.lower, best.upper, int(BRepProjectionStatus.UNRESOLVED), best.piece_id, best.root_id)
        if trim < 0:
            status = int(BRepProjectionStatus.SEAM)
    return SplineProjection(
        best.point,
        best.parameters,
        min(candidate.lower for candidate in candidates),
        best.upper,
        status,
        best.piece_id,
        best.root_id,
    )


__all__ = ["CurveProjection", "SplineProjection", "project_bspline_curve", "project_spline_face"]
