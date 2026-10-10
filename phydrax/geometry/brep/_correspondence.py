#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Whole-edge source correspondence, separate from sampled evaluation residuals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from ._intersection import RootEndpoint
from ._intersection_curve import (
    _affine_curve_coefficients,
    _period_offset,
    _period_scalar_bounds,
    _PeriodOffset,
    AffinePCurve,
    BernsteinCurvePiece,
    coefficient_enclosures,
    curve_pieces_for_interval,
    CurveEvaluator,
    CurvePiece,
    CurveRange,
    IntersectionCurve,
    IntersectionPCurve,
    PeriodicPCurve,
    surface_pieces,
    SurfaceEvaluator,
    SurfacePiece,
    SurfaceRegion,
)
from ._patches import (
    _rational_piece_jet_bounds,
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    RationalBezierPiece,
)
from ._placed import PlacedCurve, PlacedSurface, same_source_pose
from ._sphere_membership import _square_root_interval


@dataclass(frozen=True, slots=True)
class CurveSurfaceCorrespondence:
    """Full closed edge-interval enclosure, or explicit unproved subintervals."""

    deviation_bound: float
    unresolved: tuple[tuple[float, float], ...]
    intervals_processed: int

    @property
    def complete(self) -> bool:
        return not self.unresolved and bool(np.isfinite(self.deviation_bound))


class CurveSurfaceCorrespondenceError(ValueError):
    """A face/coedge geometric closure request failed with its full interval work."""

    face: int
    coedge: int
    evidence: CurveSurfaceCorrespondence
    tolerance: float

    def __init__(
        self,
        face: int,
        coedge: int,
        evidence: CurveSurfaceCorrespondence,
        tolerance: float,
        /,
    ) -> None:
        self.face = face
        self.coedge = coedge
        self.evidence = evidence
        self.tolerance = tolerance
        outcome = "unresolved" if not evidence.complete else "out-of-tolerance"
        super().__init__(
            f"Face {face} coedge {coedge} whole-edge correspondence is {outcome}: "
            f"bound={evidence.deviation_bound}, physical tolerance={tolerance}, "
            f"intervals processed={evidence.intervals_processed}, "
            f"unresolved intervals={len(evidence.unresolved)}."
        )


class _CorrespondenceSystem(StrictModule):
    curve: CurveEvaluator | None
    pcurve: CurveEvaluator
    surface: SurfaceEvaluator
    point: Array

    def residual(self, parameter: Array, /) -> Array:
        value = parameter[0]
        expected = self.point if self.curve is None else self.curve.evaluate(value)
        return self.surface.evaluate(self.pcurve.evaluate(value)) - expected


class _IntervalEvaluator(Protocol):
    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]: ...


class _RationalCurveSurfaceJets:
    """Canonical native chain rule with homogeneous-hull rational curve jets.

    Expanded rational spline basis AD is not used to enclose the fitted UV or
    XYZ jets. Native generating surface programs and their source coefficient
    intervals remain authoritative. Hooks may account bounded numeric work.
    """

    def __init__(
        self,
        curve: RationalBezierPiece | None,
        pcurve: RationalBezierPiece,
        surface: SurfaceEvaluator,
        *,
        point: np.ndarray,
        prepare: Callable[
            [
                Callable[[Array], Array],
                int,
                tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
            ],
            _IntervalEvaluator,
        ]
        | None = None,
        observe_curve_jet: Callable[[], None] | None = None,
    ) -> None:
        self.curve, self.pcurve, self.point = curve, pcurve, point
        self.observe_curve_jet = observe_curve_jet
        coefficients = coefficient_enclosures(surface)

        def program(function: Callable[[Array], Array]) -> _IntervalEvaluator:
            if prepare is None:
                return prepare_interval_function(
                    function, 2, batch_capacity=1, constant_bounds=coefficients
                )
            return prepare(function, 2, coefficients)

        self.value = program(surface.evaluate)
        self.jacobian = program(jax.jacfwd(surface.evaluate))
        self.hessian = program(jax.jacfwd(jax.jacfwd(surface.evaluate)))

    def _curve_jet(
        self, piece: RationalBezierPiece, first: float, last: float, order: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.observe_curve_jet is not None:
            self.observe_curve_jet()
        return _rational_piece_jet_bounds(piece, first, last, order=order)

    def jet(self, first: float, last: float, order: int) -> tuple[np.ndarray, np.ndarray]:
        if order not in (0, 1, 2) or isinstance(order, bool):
            raise ValueError(
                "Curve/surface residual jets support orders zero, one and two."
            )
        uv = self._curve_jet(self.pcurve, first, last, 0)
        expected = (
            self._curve_jet(self.curve, first, last, order)
            if self.curve is not None
            else ((self.point, self.point) if order == 0 else (np.zeros(3), np.zeros(3)))
        )
        if order == 0:
            value = self.value.evaluate(uv[0][None], uv[1][None])
            return interval_subtract((value[0][0], value[1][0]), expected)
        jacobian = self.jacobian.evaluate(uv[0][None], uv[1][None])
        direction = self._curve_jet(self.pcurve, first, last, order)
        result = (np.zeros(3), np.zeros(3))
        for axis in range(2):
            result = interval_add(
                result,
                interval_multiply(
                    (jacobian[0][0, :, axis], jacobian[1][0, :, axis]),
                    (direction[0][axis], direction[1][axis]),
                ),
            )
        if order == 2:
            first_jet = self._curve_jet(self.pcurve, first, last, 1)
            hessian = self.hessian.evaluate(uv[0][None], uv[1][None])
            for i in range(2):
                for j in range(2):
                    product = interval_multiply(
                        (first_jet[0][i], first_jet[1][i]),
                        (first_jet[0][j], first_jet[1][j]),
                    )
                    result = interval_add(
                        result,
                        interval_multiply(
                            (hessian[0][0, :, i, j], hessian[1][0, :, i, j]), product
                        ),
                    )
        return interval_subtract(result, expected)


def _bernstein_rational_piece(curve: BernsteinCurvePiece) -> RationalBezierPiece:
    return RationalBezierPiece(
        np.asarray(curve.controls),
        (curve.parameter_domain,),
        (0,),
        np.asarray(curve.controls_lower),
        np.asarray(curve.controls_upper),
    )


class _CorrespondenceJetMap:
    """One chain-rule order in the existing batched correspondence protocol."""

    def __init__(self, jets: _RationalCurveSurfaceJets, order: int) -> None:
        self.jets, self.order = jets, order

    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        values = [
            self.jets.jet(float(a[0]), float(b[0]), self.order)
            for a, b in zip(lower, upper, strict=True)
        ]
        shape = (len(values), 3) + (1,) * self.order
        return np.asarray([value[0] for value in values]).reshape(shape), np.asarray(
            [value[1] for value in values]
        ).reshape(shape)


@dataclass(frozen=True, slots=True)
class _PreparedCorrespondence:
    value: PreparedIntervalFunction | _CorrespondenceJetMap
    derivative: PreparedIntervalFunction | _CorrespondenceJetMap
    second_derivative: PreparedIntervalFunction | _CorrespondenceJetMap


def _prepare(system: _CorrespondenceSystem, /) -> _PreparedCorrespondence:
    if isinstance(system.pcurve, BernsteinCurvePiece) and (
        system.curve is None or isinstance(system.curve, BernsteinCurvePiece)
    ):
        jets = _RationalCurveSurfaceJets(
            None if system.curve is None else _bernstein_rational_piece(system.curve),
            _bernstein_rational_piece(system.pcurve),
            system.surface,
            point=np.asarray(system.point),
        )
        return _PreparedCorrespondence(
            _CorrespondenceJetMap(jets, 0),
            _CorrespondenceJetMap(jets, 1),
            _CorrespondenceJetMap(jets, 2),
        )
    coefficients = coefficient_enclosures(system)
    return _PreparedCorrespondence(
        prepare_interval_function(
            system.residual, 1, batch_capacity=64, constant_bounds=coefficients
        ),
        prepare_interval_function(
            jax.jacfwd(system.residual),
            1,
            batch_capacity=64,
            constant_bounds=coefficients,
        ),
        prepare_interval_function(
            jax.jacfwd(jax.jacfwd(system.residual)),
            1,
            batch_capacity=64,
            constant_bounds=coefficients,
        ),
    )


def _coupled_definition(
    curve: IntersectionCurve,
    pcurve: IntersectionPCurve,
    surface: AbstractSurfacePatch,
    /,
) -> bool:
    from ._model import _carrier_payload

    supporting = curve.first.patch if pcurve.side == "first" else curve.second.patch
    return (
        curve.fully_certified
        and pcurve.curve.branch_id == curve.branch_id
        and not pcurve.reversed
        and canonical_fingerprint(_carrier_payload(supporting))
        == canonical_fingerprint(_carrier_payload(surface))
    )


def _fraction_norm_upper(components: tuple[Fraction, ...], /) -> float:
    squared = sum((value * value for value in components), Fraction(0))
    return (
        0.0
        if squared == 0
        else float(np.nextafter(np.sqrt(np.nextafter(float(squared), np.inf)), np.inf))
    )


def _plane_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: PlanePatch,
    first: float,
    last: float,
    point: np.ndarray,
    /,
) -> float | None:
    """Exact affine coefficients or a common positive rational basis convex hull."""
    origin = tuple(Fraction(float(value)) for value in np.asarray(surface.origin))
    axes = tuple(
        tuple(Fraction(float(value)) for value in np.asarray(axis))
        for axis in (surface.first_axis, surface.second_axis)
    )
    expected, parameters = (
        _affine_curve_coefficients(curve, point),
        _affine_curve_coefficients(pcurve, np.zeros((2,), dtype=np.float64)),
    )
    if expected is not None and parameters is not None:
        constant = tuple(
            origin[dimension]
            + sum(
                (axes[axis][dimension] * parameters[0][axis] for axis in range(2)),
                Fraction(0),
            )
            - expected[0][dimension]
            for dimension in range(3)
        )
        derivative = tuple(
            sum(
                (axes[axis][dimension] * parameters[1][axis] for axis in range(2)),
                Fraction(0),
            )
            - expected[1][dimension]
            for dimension in range(3)
        )
        return max(
            _fraction_norm_upper(
                tuple(
                    a + Fraction(parameter) * b
                    for a, b in zip(constant, derivative, strict=True)
                )
            )
            for parameter in (first, last)
        )
    if not isinstance(curve, BSplineCurve) or not isinstance(pcurve, BSplineCurve):
        return None
    if (
        curve.degree != pcurve.degree
        or not np.array_equal(np.asarray(curve.knots), np.asarray(pcurve.knots))
        or not np.array_equal(np.asarray(curve.weights), np.asarray(pcurve.weights))
        or np.any(np.asarray(curve.weights) <= 0.0)
    ):
        return None
    residuals = [
        tuple(
            origin[dimension]
            + sum(
                (axes[axis][dimension] * Fraction(float(uv[axis])) for axis in range(2)),
                Fraction(0),
            )
            - Fraction(float(xyz[dimension]))
            for dimension in range(3)
        )
        for xyz, uv in zip(
            np.asarray(curve.control_points),
            np.asarray(pcurve.control_points),
            strict=True,
        )
    ]
    return max(_fraction_norm_upper(residual) for residual in residuals)


def _circle_correspondence(
    curve: AbstractCurve | None, pcurve: AbstractCurve, surface: AbstractSurfacePatch, /
) -> float | None:
    """Exact constant/cosine/sine source coefficients over the entire curve."""

    def harmonic(
        source: AbstractCurve,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]] | None:
        if isinstance(source, CircleCurve):
            radius = Fraction(float(source.radius))
            return (
                tuple(Fraction(float(value)) for value in np.asarray(source.center)),
                tuple(
                    radius * Fraction(float(value))
                    for value in np.asarray(source.first_axis)
                ),
                tuple(
                    radius * Fraction(float(value))
                    for value in np.asarray(source.second_axis)
                ),
            )
        if not isinstance(source, AffinePCurve) or not isinstance(
            source.curve, AbstractCurve
        ):
            return None
        coefficients = harmonic(source.curve)
        if coefficients is None:
            return None
        matrix, offset = source.matrix, source.offset

        def image(vector: tuple[Fraction, ...], translate: bool) -> tuple[Fraction, ...]:
            return tuple(
                (offset[axis] if translate else Fraction(0))
                + sum(
                    (
                        coefficient * value
                        for coefficient, value in zip(row, vector, strict=True)
                    ),
                    Fraction(0),
                )
                for axis, row in enumerate(matrix)
            )

        return (
            image(coefficients[0], True),
            image(coefficients[1], False),
            image(coefficients[2], False),
        )

    if not isinstance(curve, CircleCurve):
        return None
    radius = Fraction(float(curve.radius))
    center = tuple(Fraction(float(value)) for value in np.asarray(curve.center))
    cosine = tuple(
        radius * Fraction(float(value)) for value in np.asarray(curve.first_axis)
    )
    sine = tuple(
        radius * Fraction(float(value)) for value in np.asarray(curve.second_axis)
    )
    if isinstance(surface, PlanePatch):
        coefficients = harmonic(pcurve)
        if coefficients is None:
            return None
        axes = tuple(
            tuple(Fraction(float(value)) for value in np.asarray(axis))
            for axis in (surface.first_axis, surface.second_axis)
        )
        origin = tuple(Fraction(float(value)) for value in np.asarray(surface.origin))
        expected_center, expected_cosine, expected_sine = (
            tuple(
                (origin[dimension] if component == 0 else Fraction(0))
                + sum(
                    (axes[axis][dimension] * vector[axis] for axis in range(2)),
                    Fraction(0),
                )
                for dimension in range(3)
            )
            for component, vector in enumerate(coefficients)
        )
    elif isinstance(surface, CylinderPatch):
        coefficients = _affine_curve_coefficients(
            pcurve, np.zeros((2,), dtype=np.float64)
        )
        if (
            coefficients is None
            or coefficients[0][0] != 0
            or coefficients[1] != (Fraction(1), Fraction(0))
        ):
            return None
        height = coefficients[0][1]
        expected_center = tuple(
            Fraction(float(a)) + height * Fraction(float(b))
            for a, b in zip(
                np.asarray(surface.origin), np.asarray(surface.axis), strict=True
            )
        )
        cylinder_radius = Fraction(float(surface.radius))
        expected_cosine = tuple(
            cylinder_radius * Fraction(float(value))
            for value in np.asarray(surface.first_axis)
        )
        expected_sine = tuple(
            cylinder_radius * Fraction(float(value))
            for value in np.asarray(surface.second_axis)
        )
    else:
        return None
    bound = 0.0
    for actual, expected in (
        (center, expected_center),
        (cosine, expected_cosine),
        (sine, expected_sine),
    ):
        contribution = _fraction_norm_upper(
            tuple(a - b for a, b in zip(actual, expected, strict=True))
        )
        bound = (
            bound
            if contribution == 0.0
            else float(np.nextafter(bound + contribution, np.inf))
        )
    return bound


def _norm_upper(components: np.ndarray, /) -> float:
    if np.any(np.isnan(components)):
        return np.inf
    squared = np.float64(0.0)
    for component in components:
        term = np.nextafter(component * component, np.inf)
        squared = np.nextafter(squared + term, np.inf)
    return float(np.nextafter(np.sqrt(squared), np.inf))


def _spline_isoline_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: BSplineSurfacePatch,
    /,
) -> bool:
    """Exact endpoint tensor-basis restriction, with the original edge parameter."""
    if not isinstance(curve, BSplineCurve) or not isinstance(pcurve, LineCurve):
        return False
    origin, direction = np.asarray(pcurve.origin), np.asarray(pcurve.direction)
    for fixed in (0, 1):
        moving = 1 - fixed
        if direction[fixed] != 0.0 or direction[moving] != 1.0 or origin[moving] != 0.0:
            continue
        knots = np.asarray(surface.u_knots if fixed == 0 else surface.v_knots)
        degree = surface.u_degree if fixed == 0 else surface.v_degree
        if origin[fixed] == knots[degree] and np.all(
            knots[: degree + 1] == knots[degree]
        ):
            endpoint = 0
        elif origin[fixed] == knots[-degree - 1] and np.all(
            knots[-degree - 1 :] == knots[-degree - 1]
        ):
            endpoint = -1
        else:
            continue
        controls = np.asarray(surface.control_points)
        weights = np.asarray(surface.weights)
        profile = controls[endpoint, :, :] if fixed == 0 else controls[:, endpoint, :]
        profile_weights = weights[endpoint, :] if fixed == 0 else weights[:, endpoint]
        profile_knots = np.asarray(surface.v_knots if fixed == 0 else surface.u_knots)
        profile_degree = surface.v_degree if fixed == 0 else surface.u_degree
        edge_weights = np.asarray(curve.weights)
        if (
            curve.degree != profile_degree
            or not np.array_equal(np.asarray(curve.knots), profile_knots)
            or not np.array_equal(np.asarray(curve.control_points), profile)
            or edge_weights.shape != profile_weights.shape
        ):
            continue
        if np.any(profile_weights <= 0.0) or np.any(edge_weights == 0.0):
            continue
        ratio = Fraction(float(edge_weights[0])) / Fraction(float(profile_weights[0]))
        if all(
            Fraction(float(edge)) == ratio * Fraction(float(face))
            for edge, face in zip(edge_weights, profile_weights, strict=True)
        ):
            return True
    return False


def placed_deviation_bound(rotation: np.ndarray, deviation: float, /) -> float:
    """Exact Gram/Gershgorin operator bound for an authored binary source pose."""
    if deviation == 0.0:
        return 0.0
    columns = tuple(
        tuple(Fraction(float(rotation[row, column])) for row in range(3))
        for column in range(3)
    )
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
            for second in columns
        )
        for first in columns
    )
    squared = max(sum((abs(value) for value in row), Fraction(0)) for row in gram)
    scale = _square_root_interval(squared)[1]
    return float(np.nextafter(scale * deviation, np.inf))


def certify_curve_surface(
    curve: AbstractCurve | IntersectionCurve | None,
    pcurve: AbstractCurve | IntersectionPCurve,
    surface: AbstractSurfacePatch,
    parameter_box: np.ndarray,
    first: float,
    last: float,
    /,
    *,
    point: np.ndarray,
    tolerance: float,
    maximum_intervals: int = 65536,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
) -> CurveSurfaceCorrespondence:
    """Enclose ``surface(pcurve(t))-curve(t)`` over every source parameter.

    Exact generating support and a certified coupled atlas establish an
    intersection branch's represented correspondence directly. Other families
    use source-aware gather-free Bernstein/analytic pieces and the mean-value
    enclosure of the residual. A center value is itself interval-enclosed;
    neither sampled values nor local numerical convergence prove this result.
    Interval transcendental evaluations retain the owning four-ulp libm premise.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0 or maximum_intervals < 1:
        raise ValueError(
            "Correspondence tolerance must be nonnegative and interval budget positive."
        )
    if isinstance(curve, PlacedCurve) and isinstance(surface, PlacedSurface):
        if not same_source_pose(curve, surface):
            return CurveSurfaceCorrespondence(np.inf, ((first, last),), 0)
        scale = placed_deviation_bound(np.asarray(curve.rotation), 1.0)
        original = certify_curve_surface(
            curve.definition,
            pcurve,
            surface.definition,
            parameter_box,
            first,
            last,
            point=point,
            tolerance=tolerance / scale,
            maximum_intervals=maximum_intervals,
            endpoint_roots=endpoint_roots,
        )
        return CurveSurfaceCorrespondence(
            placed_deviation_bound(np.asarray(curve.rotation), original.deviation_bound),
            original.unresolved,
            original.intervals_processed,
        )
    if isinstance(curve, IntersectionCurve):
        if isinstance(pcurve, IntersectionPCurve) and _coupled_definition(
            curve, pcurve, surface
        ):
            curve.bounding_box(first, last, endpoint_roots=endpoint_roots)
            pcurve.enclosure(first, last, endpoint_roots=endpoint_roots)
            return CurveSurfaceCorrespondence(0.0, (), 0)
        return CurveSurfaceCorrespondence(np.inf, ((first, last),), 0)
    if isinstance(pcurve, IntersectionPCurve):
        return CurveSurfaceCorrespondence(np.inf, ((first, last),), 0)
    pcurve.validate_query_range(first, last)
    if curve is not None:
        curve.validate_query_range(first, last)
    if isinstance(curve, AbstractCurve) and isinstance(pcurve, AbstractCurve):
        from ._intersection import _exact_source_lift

        if _exact_source_lift(
            curve, pcurve, SurfaceRegion(surface, parameter_box), first, last
        ):
            return CurveSurfaceCorrespondence(0.0, (), 1)
    if isinstance(surface, BSplineSurfacePatch) and _spline_isoline_correspondence(
        curve, pcurve, surface
    ):
        return CurveSurfaceCorrespondence(0.0, (), 1)
    circle_bound = _circle_correspondence(curve, pcurve, surface)
    if circle_bound is not None:
        return CurveSurfaceCorrespondence(circle_bound, (), 1)
    if isinstance(surface, PlanePatch):
        bound = _plane_correspondence(curve, pcurve, surface, first, last, point)
        if bound is not None:
            return CurveSurfaceCorrespondence(bound, (), 1)
    if tolerance == 0.0:
        # Exact roots require a represented identity above. Subdivision can
        # bound a residual but cannot turn independently rounded carriers into
        # the same symbolic source, so do not spend a candidate budget trying.
        return CurveSurfaceCorrespondence(np.inf, ((first, last),), 0)
    curve_range = None if curve is None else CurveRange(curve, first, last)
    curve_segments = (
        ()
        if curve is None or curve_range is None
        else curve_pieces_for_interval(curve, curve_range.first, curve_range.last)
    )
    parameter_range = CurveRange(pcurve, first, last)
    parameter_segments = curve_pieces_for_interval(
        pcurve, parameter_range.first, parameter_range.last
    )
    patches = surface_pieces(SurfaceRegion(surface, parameter_box))
    breaks = sorted(
        {
            first,
            last,
            *(item.lower for item in curve_segments),
            *(item.upper for item in curve_segments),
            *(item.lower for item in parameter_segments),
            *(item.upper for item in parameter_segments),
        }
    )
    pending = [(start, end) for start, end in zip(breaks[:-1], breaks[1:], strict=True)]
    cache: dict[tuple[int, int, int], _PreparedCorrespondence] = {}

    def enclose(
        intervals: list[tuple[float, float]],
    ) -> dict[tuple[float, float], float | None]:
        return _residual_interval_bounds(
            intervals,
            curve,
            point,
            curve_segments,
            parameter_segments,
            patches,
            cache,
        )

    # Breadth-first levels enclose every interval of one bisection depth in
    # shared interval-program batches. The depth-first ledger below replays
    # them, so evidence, order and the interval budget are unchanged.
    outcomes: dict[tuple[float, float], float | None] = {}
    level = list(pending)
    while level and len(outcomes) + len(level) <= maximum_intervals:
        outcomes.update(enclose(level))
        level = [
            child
            for start, end in level
            for child in _bisected_interval(start, end, outcomes[start, end], tolerance)
        ]
    unresolved: list[tuple[float, float]] = []
    processed = 0
    maximum_bound = 0.0
    while pending and processed < maximum_intervals:
        start, end = pending.pop()
        processed += 1
        if (start, end) not in outcomes:
            outcomes.update(enclose([(start, end)]))
        bound = outcomes[start, end]
        if bound is None:
            unresolved.append((start, end))
            continue
        midpoint = 0.5 * (start + end)
        if np.isfinite(bound) and bound <= tolerance:
            maximum_bound = max(maximum_bound, bound)
        elif start < midpoint < end:
            pending.extend(((start, midpoint), (midpoint, end)))
        else:
            unresolved.append((start, end))
    unresolved.extend(pending)
    return CurveSurfaceCorrespondence(
        np.inf if unresolved else maximum_bound, tuple(unresolved), processed
    )


def _bisected_interval(
    start: float, end: float, bound: float | None, tolerance: float, /
) -> tuple[tuple[float, float], ...]:
    """Children of an enclosed interval that the depth-first ledger would bisect."""
    midpoint = 0.5 * (start + end)
    if bound is None or (np.isfinite(bound) and bound <= tolerance):
        return ()
    if start < midpoint < end:
        return ((start, midpoint), (midpoint, end))
    return ()


def _residual_interval_bounds(
    intervals: list[tuple[float, float]],
    curve: AbstractCurve | None,
    point: np.ndarray,
    curve_segments: tuple[CurvePiece[AbstractCurve], ...],
    parameter_segments: tuple[CurvePiece[AbstractCurve], ...],
    patches: tuple[SurfacePiece, ...],
    cache: dict[tuple[int, int, int], _PreparedCorrespondence],
    /,
) -> dict[tuple[float, float], float | None]:
    """Mean-value/Taylor residual bounds of closed intervals; None without support.

    Intervals sharing one prepared curve/p-curve/surface piece triple are
    enclosed as one batch. Interval programs act row by row on fixed-capacity
    chunks, so each row is enclosed exactly as it would be alone.
    """
    bounds: dict[tuple[float, float], float | None] = {}
    groups: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    for start, end in intervals:
        midpoint = 0.5 * (start + end)
        parameter_piece = next(
            item for item in parameter_segments if item.lower <= midpoint <= item.upper
        )
        curve_piece = (
            None
            if curve is None
            else next(
                item for item in curve_segments if item.lower <= midpoint <= item.upper
            )
        )
        uv_box = parameter_piece.evaluator.bounding_box(start, end)
        possible = [
            patch
            for patch in patches
            if np.all(uv_box[1] >= patch.lower) and np.all(uv_box[0] <= patch.upper)
        ]
        if not possible:
            bounds[start, end] = None
            continue
        bounds[start, end] = 0.0
        for patch in possible:
            key = (
                -1 if curve_piece is None else curve_piece.index,
                parameter_piece.index,
                patch.index,
            )
            if key not in cache:
                cache[key] = _prepare(
                    _CorrespondenceSystem(
                        None if curve_piece is None else curve_piece.evaluator,
                        parameter_piece.evaluator,
                        patch.evaluator,
                        jnp.asarray(point, dtype=jnp.float64),
                    )
                )
            groups.setdefault(key, []).append((start, end))
    for key, members in groups.items():
        prepared = cache[key]
        starts = np.asarray([[start] for start, _ in members], dtype=np.float64)
        ends = np.asarray([[end] for _, end in members], dtype=np.float64)
        centers = np.asarray(
            [[0.5 * (start + end)] for start, end in members], dtype=np.float64
        )
        low, high = prepared.value.evaluate(centers, centers)
        derivative_low, derivative_high = prepared.derivative.evaluate(starts, ends)
        center_low, center_high = prepared.derivative.evaluate(centers, centers)
        second_low, second_high = prepared.second_derivative.evaluate(starts, ends)
        for row, (start, end) in enumerate(members):
            midpoint = 0.5 * (start + end)
            magnitude = np.maximum(np.abs(low[row]), np.abs(high[row]))
            derivative = np.maximum(
                np.abs(derivative_low[row, :, 0]), np.abs(derivative_high[row, :, 0])
            )
            radius = np.nextafter(max(midpoint - start, end - midpoint), np.inf)
            error = np.nextafter(
                magnitude + np.nextafter(derivative * radius, np.inf), np.inf
            )
            center_derivative = np.maximum(
                np.abs(center_low[row, :, 0]), np.abs(center_high[row, :, 0])
            )
            second_derivative = np.maximum(
                np.abs(second_low[row, :, 0, 0]), np.abs(second_high[row, :, 0, 0])
            )
            quadratic_radius = np.nextafter(
                0.5 * np.nextafter(radius * radius, np.inf), np.inf
            )
            linear_error = np.nextafter(center_derivative * radius, np.inf)
            second_error = np.nextafter(second_derivative * quadratic_radius, np.inf)
            taylor_error = np.nextafter(
                magnitude + np.nextafter(linear_error + second_error, np.inf), np.inf
            )
            previous = bounds[start, end]
            if previous is None:
                raise RuntimeError("A supported residual interval lost its bound.")
            bounds[start, end] = max(
                previous, _norm_upper(np.minimum(error, taylor_error))
            )
    return bounds


# ------------------------------------------------ exact carrier parameter maps


type ExactParameter = Fraction | _PeriodOffset

# (cos, sin) of k quarter turns: the only phases whose frame rotation is exact.
_QUARTER_TURNS: tuple[tuple[Fraction, Fraction], ...] = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
    (Fraction(-1), Fraction(0)),
    (Fraction(0), Fraction(-1)),
)


def exact_parameter_parts(value: ExactParameter, /) -> tuple[Fraction, Fraction]:
    """``(rational, turns)`` of ``rational + turns * 2*pi``."""
    return (
        (value, Fraction(0))
        if isinstance(value, Fraction)
        else (value.rational, value.turns)
    )


def exact_parameter_affine(
    value: ExactParameter, scale: Fraction, offset: ExactParameter, /
) -> ExactParameter:
    """``scale * value + offset`` in Q + Q*(2*pi), canonicalized to a rational when possible."""
    rational, turns = exact_parameter_parts(value)
    offset_rational, offset_turns = exact_parameter_parts(offset)
    return _period_offset(
        rational * scale + offset_rational, turns * scale + offset_turns
    )


def exact_parameter_sign(value: ExactParameter, /) -> int | None:
    """Exact sign; ``None`` only when the directed 2*pi enclosure cannot decide it."""
    rational, turns = exact_parameter_parts(value)
    if not turns:
        return (rational > 0) - (rational < 0)
    lower, upper = _period_scalar_bounds(value)
    if float(lower) > 0.0:
        return 1
    if float(upper) < 0.0:
        return -1
    return None


@dataclass(frozen=True, slots=True)
class CurveParameterMap:
    """Exact carrier identity ``first(scale * s + offset) == second(s)`` for every ``s``.

    ``offset`` lies in Q + Q*(2*pi); a transcendental phase is never rationalized,
    and parameterization equality alone never implies a world-image identity.
    """

    scale: Fraction
    offset: ExactParameter

    def __post_init__(self) -> None:
        if not isinstance(self.scale, Fraction) or not self.scale:
            raise ValueError(
                "A carrier parameter map needs a nonzero exact rational scale."
            )
        if not isinstance(self.offset, (Fraction, _PeriodOffset)):
            raise TypeError("A carrier parameter offset must be exact in Q + Q*(2*pi).")

    def apply(self, value: ExactParameter, /) -> ExactParameter:
        return exact_parameter_affine(value, self.scale, self.offset)

    def inverse(self) -> CurveParameterMap:
        return CurveParameterMap(
            1 / self.scale,
            exact_parameter_affine(self.offset, -1 / self.scale, Fraction(0)),
        )


def _exact_vector(values: Array | np.ndarray, /) -> tuple[Fraction, ...]:
    return tuple(
        Fraction(float(value))
        for value in np.asarray(values, dtype=np.float64).reshape(-1)
    )


def _binary_vector(values: tuple[Fraction, ...], /) -> np.ndarray | None:
    """Binary64 values only when every exact coefficient is represented without rounding."""
    floats = np.asarray([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(floats)) or any(
        Fraction(float(number)) != value
        for number, value in zip(floats, values, strict=True)
    ):
        return None
    return floats


def _space_line(
    curve: AbstractCurve | IntersectionCurve, /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    if not isinstance(curve, (LineCurve, BSplineCurve)):
        return None
    return _affine_curve_coefficients(
        curve, np.zeros((curve.ambient_dimension,), dtype=np.float64)
    )


def _circle_frame(
    curve: CircleCurve, /
) -> tuple[tuple[Fraction, ...], Fraction, tuple[Fraction, ...], tuple[Fraction, ...]]:
    return (
        _exact_vector(curve.center),
        Fraction(float(curve.radius)),
        _exact_vector(curve.first_axis),
        _exact_vector(curve.second_axis),
    )


def _projective(vector: tuple[Fraction, ...], /) -> tuple[Fraction, ...] | None:
    pivot = next((index for index, value in enumerate(vector) if value), None)
    return None if pivot is None else tuple(value / vector[pivot] for value in vector)


def curve_support_key(curve: AbstractCurve | IntersectionCurve, /) -> tuple[object, ...]:
    """Exact world-image support key: equal keys are necessary for a carrier map.

    Lines use their exact projective direction and pivot-plane point; circles
    use exact center, radius and projective plane normal. Every other family is
    keyed by its complete carrier payload, so only identical sources compare.
    """
    from ._model import _carrier_payload

    line = _space_line(curve)
    pivot = (
        None
        if line is None
        else next((index for index, value in enumerate(line[1]) if value), None)
    )
    if line is not None and pivot is not None:
        origin, direction = line
        step = origin[pivot] / direction[pivot]
        return (
            "line",
            _projective(direction),
            tuple(
                point - step * slope
                for point, slope in zip(origin, direction, strict=True)
            ),
        )
    if isinstance(curve, CircleCurve):
        center, radius, first, second = _circle_frame(curve)
        if len(center) != 3:
            return ("circle", center, radius)
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        return ("circle", center, radius, _projective(normal))
    return ("carrier", canonical_fingerprint(_carrier_payload(curve)))


def _line_map(
    first: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    second: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    /,
) -> CurveParameterMap | None:
    (origin_a, direction_a), (origin_b, direction_b) = first, second
    pivot = next((index for index, value in enumerate(direction_a) if value), None)
    if pivot is None:
        return None
    scale = direction_b[pivot] / direction_a[pivot]
    offset = (origin_b[pivot] - origin_a[pivot]) / direction_a[pivot]
    if (
        not scale
        or any(b != scale * a for a, b in zip(direction_a, direction_b, strict=True))
        or any(
            b != a + offset * d
            for a, b, d in zip(origin_a, origin_b, direction_a, strict=True)
        )
    ):
        return None
    return CurveParameterMap(scale, offset)


def _circle_map(first: CircleCurve, second: CircleCurve, /) -> CurveParameterMap | None:
    """``second(s) == first(sigma*s + k*pi/2)`` by exact frame algebra, or None."""
    center_a, radius_a, cosine_a, sine_a = _circle_frame(first)
    center_b, radius_b, cosine_b, sine_b = _circle_frame(second)
    if center_a != center_b or radius_a != radius_b:
        return None
    for quarter, (cosine, sine) in enumerate(_QUARTER_TURNS):
        if cosine_b != tuple(
            cosine * a + sine * b for a, b in zip(cosine_a, sine_a, strict=True)
        ):
            continue
        rotated = tuple(
            cosine * b - sine * a for a, b in zip(cosine_a, sine_a, strict=True)
        )
        for sigma in (Fraction(1), Fraction(-1)):
            if sine_b == tuple(sigma * value for value in rotated):
                return CurveParameterMap(
                    sigma, _period_offset(Fraction(0), Fraction(quarter, 4))
                )
    return None


def prove_curve_correspondence(
    first: AbstractCurve | IntersectionCurve,
    second: AbstractCurve | IntersectionCurve,
    /,
) -> CurveParameterMap | None:
    """Prove ``first(scale*s + offset) == second(s)`` from exact source coefficients.

    Identical carrier payloads are the identity. Lines (including degree-one
    rational splines) use exact affine coefficients; circles use exact center,
    radius and quarter-turn/reflection frame algebra. A coincident support
    whose parameter map is not exactly representable returns None.
    """
    from ._model import _carrier_payload

    if first.ambient_dimension != second.ambient_dimension:
        return None
    if canonical_fingerprint(_carrier_payload(first)) == canonical_fingerprint(
        _carrier_payload(second)
    ):
        return CurveParameterMap(Fraction(1), Fraction(0))
    lines = _space_line(first), _space_line(second)
    if lines[0] is not None and lines[1] is not None:
        return _line_map(lines[0], lines[1])
    if isinstance(first, CircleCurve) and isinstance(second, CircleCurve):
        return _circle_map(first, second)
    return None


def _reparameterized_spline(
    curve: BSplineCurve, scale: Fraction, offset: Fraction, /
) -> BSplineCurve | None:
    knots = _binary_vector(
        tuple((knot - offset) / scale for knot in _exact_vector(curve.knots))
    )
    if knots is None:
        return None
    controls, weights = np.asarray(curve.control_points), np.asarray(curve.weights)
    if scale < 0:
        knots, controls, weights = knots[::-1], controls[::-1], weights[::-1]
    return BSplineCurve(controls, weights, knots, curve.degree)


def _reparameterized_circle(
    curve: CircleCurve, scale: Fraction, offset: ExactParameter, /
) -> CircleCurve | None:
    rational, turns = exact_parameter_parts(offset)
    if abs(scale) != 1 or rational or (4 * turns).denominator != 1:
        return None
    cosine, sine = _QUARTER_TURNS[int(4 * turns) % 4]
    _, _, first, second = _circle_frame(curve)
    cosine_axis = _binary_vector(
        tuple(cosine * a + sine * b for a, b in zip(first, second, strict=True))
    )
    sine_axis = _binary_vector(
        tuple(scale * (cosine * b - sine * a) for a, b in zip(first, second, strict=True))
    )
    if cosine_axis is None or sine_axis is None:
        return None
    return CircleCurve(np.asarray(curve.center), cosine_axis, sine_axis, curve.radius)


def reparameterize_pcurve(
    curve: AbstractCurve | IntersectionPCurve,
    parameter_map: CurveParameterMap,
    /,
) -> AbstractCurve | IntersectionPCurve | None:
    """The same exact family evaluating ``curve(scale*t + offset)``, or None.

    No carrier is refitted: line origins/directions, spline knots and circle
    quarter-turn frames are transported only when every binary64 coefficient is
    exact. Affine and periodic UV operations keep their original gauge.
    """
    scale, offset = parameter_map.scale, parameter_map.offset
    if scale == 1 and offset == 0:
        return curve
    match curve:
        case AffinePCurve():
            inner = (
                reparameterize_pcurve(curve.curve, parameter_map)
                if isinstance(curve.curve, AbstractCurve)
                else None
            )
            return (
                AffinePCurve(inner, curve.matrix, curve.offset)
                if isinstance(inner, AbstractCurve)
                else None
            )
        case PeriodicPCurve():
            source = curve.source_curve
            inner = (
                reparameterize_pcurve(source, parameter_map)
                if isinstance(source, AbstractCurve)
                else None
            )
            return (
                PeriodicPCurve(inner, curve.patch, curve.period_shifts)
                if isinstance(inner, AbstractCurve)
                else None
            )
        case LineCurve():
            if not isinstance(offset, Fraction):
                return None
            origin, direction = (
                _exact_vector(curve.origin),
                _exact_vector(curve.direction),
            )
            moved = _binary_vector(
                tuple(a + offset * d for a, d in zip(origin, direction, strict=True))
            )
            scaled = _binary_vector(tuple(scale * d for d in direction))
            return None if moved is None or scaled is None else LineCurve(moved, scaled)
        case CircleCurve():
            return _reparameterized_circle(curve, scale, offset)
        case BSplineCurve():
            return (
                _reparameterized_spline(curve, scale, offset)
                if isinstance(offset, Fraction)
                else None
            )
        case _:
            return None


__all__ = [
    "CurveParameterMap",
    "CurveSurfaceCorrespondence",
    "certify_curve_surface",
    "curve_support_key",
    "exact_parameter_affine",
    "exact_parameter_parts",
    "exact_parameter_sign",
    "prove_curve_correspondence",
    "reparameterize_pcurve",
]
