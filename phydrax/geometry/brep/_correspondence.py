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

import phydrax.geometry.brep._carrier_pieces as carrier_owner

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import rational_vector
from ._carrier_pieces import CurvePiece, SurfacePiece
from ._coordinate_request import coordinate_source_request
from ._intersection import exact_source_lift, RootEndpoint
from ._intersection_curve import (
    affine_curve_coefficients,
    AffinePCurve,
    curve_pieces_for_interval,
    CurveEvaluator,
    CurveRange,
    IntersectionCurve,
    IntersectionPCurve,
    PeriodicPCurve,
    surface_pieces,
    SurfaceRegion,
)
from ._interval_source import evaluate_interval
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinCurvePiece,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    coefficient_enclosures,
    LineCurve,
    PlanePatch,
    rational_piece_jets,
    RationalBezierPiece,
)
from ._placed import PlacedCurve, PlacedSurface, same_source_pose
from ._request_preparations import TrimRequestPreparations
from ._source_correspondence import (
    circle_correspondence,
    norm_upper,
    placed_deviation_bound,
    plane_correspondence,
)
from ._trim_helpers import period_offset, period_scalar_bounds, PeriodOffset


@dataclass(frozen=True, slots=True)
class CurveSurfaceCorrespondence:
    """Full closed edge-interval enclosure, or explicit unproved subintervals."""

    deviation_bound: float
    unresolved: tuple[tuple[float, float], ...]
    intervals_processed: int

    @classmethod
    def identity(cls, intervals_processed: int = 0) -> CurveSurfaceCorrespondence:
        """Return correspondence proved by a generating carrier identity."""
        return cls(0.0, (), intervals_processed)

    @classmethod
    def enclosed(
        cls, deviation_bound: float, intervals_processed: int
    ) -> CurveSurfaceCorrespondence:
        """Return a whole-interval deviation enclosure."""
        return cls(deviation_bound, (), intervals_processed)

    @classmethod
    def unproved(
        cls,
        intervals: tuple[tuple[float, float], ...],
        intervals_processed: int = 0,
    ) -> CurveSurfaceCorrespondence:
        """Return the source intervals whose correspondence remains unresolved."""
        return cls(np.inf, intervals, intervals_processed)

    def rescale(self, rotation: np.ndarray) -> CurveSurfaceCorrespondence:
        """Transport the deviation enclosure through a placed carrier's linear map."""
        return CurveSurfaceCorrespondence(
            placed_deviation_bound(rotation, self.deviation_bound),
            self.unresolved,
            self.intervals_processed,
        )

    @property
    def complete(self) -> bool:
        return not self.unresolved and bool(np.isfinite(self.deviation_bound))


class _CorrespondenceSystem(StrictModule):
    curve: CurveEvaluator | None
    pcurve: CurveEvaluator
    surface: AbstractSurfacePatch
    point: Array

    def residual(self, parameter: Array, /) -> Array:
        value = parameter[0]
        expected = self.point if self.curve is None else self.curve.evaluate(value)
        return self.surface.evaluate(self.pcurve.evaluate(value)) - expected


class _IntervalEvaluator(Protocol):
    def evaluate(
        self, lower: np.ndarray, upper: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class _SurfaceJetPrograms:
    surface: AbstractSurfacePatch
    value: _IntervalEvaluator
    jacobian: _IntervalEvaluator
    hessian: _IntervalEvaluator


class RationalCurveSurfaceJets:
    """Enclose curve/surface chain-rule jets with homogeneous rational hulls.

    Native surface programs and coefficient intervals define the generating
    surface. Hooks account for bounded numerical work.
    """

    @checked
    def __init__(
        self,
        curve: RationalBezierPiece | None,
        pcurve: RationalBezierPiece,
        surface: AbstractSurfacePatch,
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
        surface_programs: _SurfaceJetPrograms | None = None,
    ) -> None:
        self.curve, self.pcurve, self.point = curve, pcurve, point
        self.observe_curve_jet = observe_curve_jet
        if surface_programs is not None:
            if surface_programs.surface is not surface:
                raise ValueError("Surface jet programs must bind the same source patch.")
            programs = surface_programs
        else:
            coefficients = coefficient_enclosures(surface)

            def program(function: Callable[[Array], Array]) -> _IntervalEvaluator:
                if prepare is None:
                    return prepare_interval_function(
                        function, 2, batch_capacity=1, constant_bounds=coefficients
                    )
                return prepare(function, 2, coefficients)

            programs = _SurfaceJetPrograms(
                surface, program(surface.evaluate),
                program(jax.jacfwd(surface.evaluate)),
                program(jax.jacfwd(jax.jacfwd(surface.evaluate))),
            )
        self.surface_programs = programs
        self.value, self.jacobian, self.hessian = (
            programs.value, programs.jacobian, programs.hessian
        )

    def _curve_jets(
        self, piece: RationalBezierPiece, first: float, last: float, order: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if self.observe_curve_jet is not None:
            self.observe_curve_jet()
        return rational_piece_jets(piece, first, last, order)

    @checked
    def jet(
        self,
        first: float,
        last: float,
        order: int,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                    order,
                )
            ) as budget:
                return self.jet(
                    first, last, order, budget=budget, preparations=preparations
                )
        if order not in (0, 1, 2) or isinstance(order, bool):
            raise ValueError(
                f"RationalCurveSurfaceJets.order must be zero, one or two; got {order!r}"
            )
        return self._jet_orders(
            first, last, (order,), budget=budget, preparations=preparations
        )[0]

    def _jet_orders(
        self,
        first: float,
        last: float,
        orders: tuple[int, ...],
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Share lower-order quotient hulls and surface jets within one query."""
        maximum = max(orders)
        uv_jets = self._curve_jets(self.pcurve, first, last, maximum)
        uv = uv_jets[0]
        expected = (
            self._curve_jets(self.curve, first, last, maximum)
            if self.curve is not None
            else (
                (self.point, self.point),
                *((np.zeros(3), np.zeros(3)) for _ in range(maximum)),
            )
        )
        value = (
            evaluate_interval(
                self.value, uv[0][None], uv[1][None],
                budget=budget, preparations=preparations,
            )
            if 0 in orders else None
        )
        jacobian = (
            evaluate_interval(
                self.jacobian, uv[0][None], uv[1][None],
                budget=budget, preparations=preparations,
            )
            if maximum > 0 else None
        )
        hessian = (
            evaluate_interval(
                self.hessian, uv[0][None], uv[1][None],
                budget=budget, preparations=preparations,
            )
            if maximum == 2 else None
        )
        return tuple(
            curve_surface_chain_jet(
                order,
                expected[order],
                value=None if value is None else (value[0][0], value[1][0]),
                jacobian=None if jacobian is None else (jacobian[0][0], jacobian[1][0]),
                direction=None if order == 0 else uv_jets[order],
                hessian=None if hessian is None else (hessian[0][0], hessian[1][0]),
                velocity=None if order != 2 else uv_jets[1],
            )
            for order in orders
        )

    def residual_jets_batch(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        centers: np.ndarray,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """All four Taylor premises, without retaining state across queries."""
        rows: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = []
        for start, end, center in zip(starts, ends, centers, strict=True):
            center_value, center_derivative = self._jet_orders(
                float(center[0]), float(center[0]), (0, 1),
                budget=budget, preparations=preparations,
            )
            derivative, second = self._jet_orders(
                float(start[0]), float(end[0]), (1, 2),
                budget=budget, preparations=preparations,
            )
            rows.append((center_value, derivative, center_derivative, second))

        def batch(index: int, order: int) -> tuple[np.ndarray, np.ndarray]:
            shape = (len(rows), 3) + (1,) * order
            return (
                np.asarray([row[index][0] for row in rows]).reshape(shape),
                np.asarray([row[index][1] for row in rows]).reshape(shape),
            )

        return batch(0, 0), batch(1, 1), batch(2, 1), batch(3, 2)


type _JetInterval = tuple[np.ndarray, np.ndarray]


def curve_surface_chain_jet(
    order: int,
    expected: _JetInterval,
    *,
    value: _JetInterval | None = None,
    jacobian: _JetInterval | None = None,
    direction: _JetInterval | None = None,
    hessian: _JetInterval | None = None,
    velocity: _JetInterval | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(S o uv)^(order) - expected`` from enclosed surface/p-curve jet rows.

    ``value``/``jacobian``/``hessian`` enclose the surface over the p-curve's
    order-zero box; ``direction`` is the p-curve jet of ``order`` and
    ``velocity`` its first jet (second order only).
    """
    if order == 0:
        if value is None:
            raise ValueError("curve_surface_chain_jet.value must enclose order zero")
        lower, upper = interval_subtract(value, expected)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if jacobian is None or direction is None:
        raise ValueError(
            "curve_surface_chain_jet.jacobian and direction must enclose positive orders"
        )
    result = (np.zeros(3), np.zeros(3))
    for axis in range(2):
        result = interval_add(
            result,
            interval_multiply(
                (jacobian[0][:, axis], jacobian[1][:, axis]),
                (direction[0][axis], direction[1][axis]),
            ),
        )
    if order == 2:
        if hessian is None or velocity is None:
            raise ValueError(
                "curve_surface_chain_jet.hessian and velocity must enclose order two"
            )
        for i in range(2):
            for j in range(2):
                product = interval_multiply(
                    (velocity[0][i], velocity[1][i]),
                    (velocity[0][j], velocity[1][j]),
                )
                result = interval_add(
                    result,
                    interval_multiply((hessian[0][:, i, j], hessian[1][:, i, j]), product),
                )
    lower, upper = interval_subtract(result, expected)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


@checked
def bernstein_rational_piece(curve: BernsteinCurvePiece) -> RationalBezierPiece:
    return RationalBezierPiece(
        np.asarray(curve.controls),
        (curve.parameter_domain,),
        (0,),
        np.asarray(curve.controls_lower),
        np.asarray(curve.controls_upper),
    )


class _CorrespondenceJetMap:
    """One chain-rule order in the existing batched correspondence protocol."""

    def __init__(self, jets: RationalCurveSurfaceJets, order: int) -> None:
        self.jets, self.order = jets, order

    def evaluate_scope(
        self, lower: np.ndarray, upper: np.ndarray, /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = [
            self.jets.jet(
                float(a[0]), float(b[0]), self.order,
                budget=budget, preparations=preparations,
            )
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


def _prepare(
    system: _CorrespondenceSystem,
    /,
    surface_programs: _SurfaceJetPrograms | None = None,
) -> _PreparedCorrespondence:
    match (system.curve, system.pcurve):
        case (
            (BernsteinCurvePiece() | None) as curve,
            BernsteinCurvePiece() as pcurve,
        ):
            jets = RationalCurveSurfaceJets(
                None if curve is None else bernstein_rational_piece(curve),
                bernstein_rational_piece(pcurve),
                system.surface,
                point=np.asarray(system.point),
                surface_programs=surface_programs,
            )
            return _PreparedCorrespondence(
                _CorrespondenceJetMap(jets, 0),
                _CorrespondenceJetMap(jets, 1),
                _CorrespondenceJetMap(jets, 2),
            )
        case _:
            pass
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

    supporting = curve.first.patch if pcurve.side == "first" else curve.second.patch
    return (
        curve.fully_certified
        and pcurve.curve.branch_id == curve.branch_id
        and not pcurve.reversed
        and canonical_fingerprint(carrier_owner.carrier_payload(supporting))
        == canonical_fingerprint(carrier_owner.carrier_payload(surface))
    )


def _spline_isoline_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: BSplineSurfacePatch,
    /,
) -> bool:
    """Prove an endpoint tensor-basis restriction using the source edge parameter."""
    match (curve, pcurve):
        case (BSplineCurve() as spline, LineCurve() as parameter_line):
            curve, pcurve = spline, parameter_line
        case _:
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


class _CorrespondenceWorkset(StrictModule):
    curve_segments: tuple[CurvePiece[AbstractCurve], ...]
    parameter_segments: tuple[CurvePiece[AbstractCurve], ...]
    patches: tuple[SurfacePiece[AbstractSurfacePatch], ...]
    pending: list[tuple[float, float]]


def _prepare_correspondence_workset(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: AbstractSurfacePatch,
    parameter_box: np.ndarray,
    first: float,
    last: float,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> _CorrespondenceWorkset:
    curve_range = None if curve is None else CurveRange(curve, first, last)
    curve_segments = (
        ()
        if curve is None or curve_range is None
        else curve_pieces_for_interval(
            curve, curve_range.first, curve_range.last, coordinate_budget=budget
        )
    )
    parameter_range = CurveRange(pcurve, first, last)
    parameter_segments = curve_pieces_for_interval(
        pcurve, parameter_range.first, parameter_range.last, coordinate_budget=budget
    )
    patches = surface_pieces(
        SurfaceRegion(surface, parameter_box, budget=budget, preparations=preparations),
        budget=budget,
    )
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
    return _CorrespondenceWorkset(curve_segments, parameter_segments, patches, pending)


def _generating_correspondence(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: AbstractSurfacePatch,
    parameter_box: np.ndarray,
    first: float,
    last: float,
    point: np.ndarray,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> CurveSurfaceCorrespondence | None:
    if curve is not None and exact_source_lift(
        curve,
        pcurve,
        SurfaceRegion(surface, parameter_box, budget=budget, preparations=preparations),
        first,
        last,
    ):
        return CurveSurfaceCorrespondence.identity(1)
    match surface:
        case BSplineSurfacePatch() if _spline_isoline_correspondence(
            curve, pcurve, surface
        ):
            return CurveSurfaceCorrespondence.identity(1)
        case _:
            pass
    circle_bound = circle_correspondence(curve, pcurve, surface)
    if circle_bound is not None:
        return CurveSurfaceCorrespondence.enclosed(circle_bound, 1)
    match surface:
        case PlanePatch():
            bound = plane_correspondence(curve, pcurve, surface, first, last, point)
            if bound is not None:
                return CurveSurfaceCorrespondence.enclosed(bound, 1)
        case _:
            pass
    return None


def _execute_correspondence_workset(
    workset: _CorrespondenceWorkset,
    curve: AbstractCurve | None,
    point: np.ndarray,
    tolerance: float,
    maximum_intervals: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> CurveSurfaceCorrespondence:
    pending = workset.pending
    cache: dict[tuple[int, int, int], _PreparedCorrespondence] = {}
    surface_programs: dict[int, _SurfaceJetPrograms] = {}

    def enclose(
        intervals: list[tuple[float, float]],
    ) -> dict[tuple[float, float], float | None]:
        return _residual_interval_bounds(
            intervals,
            curve,
            point,
            workset.curve_segments,
            workset.parameter_segments,
            workset.patches,
            cache,
            preparations=preparations,
            budget=budget,
            tolerance=tolerance,
            surface_programs=surface_programs,
        )

    # Batched breadth-first enclosures are replayed through the depth-first
    # ledger, retaining its evidence order and interval-budget boundary.
    outcomes: dict[tuple[float, float], float | None] = {}
    level = list(pending)
    while level and len(outcomes) + len(level) <= maximum_intervals:
        outcomes.update(enclose(level))
        unsupported = tuple(interval for interval in level if outcomes[interval] is None)
        if unsupported:
            return CurveSurfaceCorrespondence.unproved(unsupported, len(outcomes))
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
    return (
        CurveSurfaceCorrespondence.unproved(tuple(unresolved), processed)
        if unresolved
        else CurveSurfaceCorrespondence.enclosed(maximum_bound, processed)
    )



def _correspondence_key(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: AbstractSurfacePatch,
    parameter_box: np.ndarray,
    first: float,
    last: float,
    point: np.ndarray,
    maximum_intervals: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> str:
    # iter_array_bytes uses at most a 65536-byte noncontiguous-array buffer.
    budget.reserve(0, 65536)
    # The residual's expected value is the actual curve when present; its
    # fallback point participates only in the collapsed point-carrier case.
    expected_point = point if curve is None else None
    # Only collapse decomposition domains after a fresh, whole-range outward
    # enclosure proves that both supplied domains contain the same p-curve.
    # Mere overlap or a toleranced containment test cannot authorize reuse.
    uv_box = np.asarray(
        pcurve.bounding_box(
            first, last, preparations=preparations, budget=budget
        ),
        dtype=np.float64,
    )
    covered_box = (
        uv_box
        if np.all(uv_box[0] >= parameter_box[0])
        and np.all(uv_box[1] <= parameter_box[1])
        else parameter_box
    )
    carrier_owner.admit_carrier_fingerprint(
        (
            curve, pcurve, surface, covered_box, first, last, expected_point,
            maximum_intervals,
        ),
        budget,
    )
    return canonical_fingerprint(
        {
            "algorithm": "closed-source-directed-mean-value-taylor-v2",
            "curve": None if curve is None else carrier_owner.carrier_payload(curve),
            "pcurve": carrier_owner.carrier_payload(pcurve),
            "surface": carrier_owner.carrier_payload(surface),
            "parameter_box": covered_box,
            "range": (first, last),
            "point": expected_point,
            "maximum_intervals": maximum_intervals,
        }
    )


def _certify_native_curve_surface(
    curve: AbstractCurve | None,
    pcurve: AbstractCurve,
    surface: AbstractSurfacePatch,
    parameter_box: np.ndarray,
    first: float,
    last: float,
    point: np.ndarray,
    tolerance: float,
    maximum_intervals: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> CurveSurfaceCorrespondence:
    pcurve.validate_query_range(first, last)
    if curve is not None:
        curve.validate_query_range(first, last)
    generating = _generating_correspondence(
        curve, pcurve, surface, parameter_box, first, last, point, preparations, budget
    )
    budget.ticket.checkpoint()
    if generating is not None:
        return generating
    if tolerance == 0.0:
        # Subdivision encloses residuals but cannot prove a generating identity
        # between independently rounded source carriers.
        return CurveSurfaceCorrespondence.unproved(((first, last),))
    with budget.temporary_scope():
        key = _correspondence_key(
            curve, pcurve, surface, parameter_box, first, last, point,
            maximum_intervals, preparations, budget,
        )
        cached = budget.brep_correspondence_cache.get(key)
        if cached is not None and cached.complete and cached.deviation_bound <= tolerance:
            budget.ticket.checkpoint()
            return cached
        workset = _prepare_correspondence_workset(
            curve, pcurve, surface, parameter_box, first, last, preparations, budget
        )
        result = _execute_correspondence_workset(
            workset, curve, point, tolerance, maximum_intervals, preparations, budget
        )
        budget.ticket.checkpoint()
        if result.complete:
            budget.retain_owner((key, result))
            budget.ticket.checkpoint()
            budget.brep_correspondence_cache[key] = result
        return result


@checked
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
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> CurveSurfaceCorrespondence:
    """Enclose ``surface(pcurve(t))-curve(t)`` over every source parameter.

    Generating support and a certified coupled atlas establish an
    intersection branch's represented correspondence directly. Other families
    use source-aware gather-free Bernstein/analytic pieces and the mean-value
    enclosure of the residual. A center value is itself interval-enclosed;
    neither sampled values nor local numerical convergence prove this result.
    Interval transcendental evaluations assume the documented four-ulp libm bound.

    Complete native proofs are retained only by the supplied request budget.
    Reuse binds the live carriers, closed source range, parameter box, point and
    controls; range and generating-source checks still precede every lookup.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                curve,
                pcurve,
                surface,
                parameter_box,
                first,
                last,
            )
        ) as budget:
            return certify_curve_surface(
                curve,
                pcurve,
                surface,
                parameter_box,
                first,
                last,
                point=point,
                tolerance=tolerance,
                maximum_intervals=maximum_intervals,
                endpoint_roots=endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
    if not np.isfinite(tolerance) or tolerance < 0.0 or maximum_intervals < 1:
        raise ValueError(
            f"CurveSurfaceCorrespondence.controls must have nonnegative tolerance and positive interval budget; got tolerance={tolerance}, maximum_intervals={maximum_intervals}"
        )
    match (curve, pcurve, surface):
        case (PlacedCurve(), _, PlacedSurface()):
            if not same_source_pose(curve, surface):
                return CurveSurfaceCorrespondence.unproved(((first, last),))
            rotation = np.asarray(curve.rotation)
            scale = placed_deviation_bound(rotation, 1.0)
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
                preparations=preparations,
                budget=budget,
            )
            return original.rescale(rotation)
        case (IntersectionCurve(), IntersectionPCurve(), _) if _coupled_definition(
            curve, pcurve, surface
        ):
            curve.bounding_box(
                first,
                last,
                endpoint_roots=endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
            pcurve.enclosure(
                first,
                last,
                endpoint_roots=endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
            return CurveSurfaceCorrespondence.identity()
        case (IntersectionCurve(), _, _) | (_, IntersectionPCurve(), _):
            return CurveSurfaceCorrespondence.unproved(((first, last),))
        case (
            (AbstractCurve() | None) as native_curve,
            AbstractCurve() as native_pcurve,
            _,
        ):
            return _certify_native_curve_surface(
                native_curve,
                native_pcurve,
                surface,
                parameter_box,
                first,
                last,
                point,
                tolerance,
                maximum_intervals,
                preparations,
                budget,
            )
        case _:
            raise TypeError(
                "certify_curve_surface.carriers must satisfy the declared source families; "
                f"got {type(curve).__name__}, {type(pcurve).__name__}"
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
    patches: tuple[SurfacePiece[AbstractSurfacePatch], ...],
    cache: dict[tuple[int, int, int], _PreparedCorrespondence],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
    tolerance: float,
    surface_programs: dict[int, _SurfaceJetPrograms] | None = None,
) -> dict[tuple[float, float], float | None]:
    """Mean-value/Taylor bounds; None without support or with a proved violation.

    Intervals sharing one prepared curve/p-curve/surface piece triple are
    enclosed as one batch. Interval programs act row by row on fixed-capacity
    chunks, so each row is enclosed exactly as it would be alone.
    """
    if surface_programs is None:
        surface_programs = {}
    bounds: dict[tuple[float, float], float] = {}
    contradicted: set[tuple[float, float]] = set()
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
        parameter_curve = parameter_piece.evaluator
        uv_box = type(parameter_curve).bounding_box(
            parameter_curve, start, end, preparations=preparations, budget=budget
        )
        possible = [
            patch
            for patch in patches
            if np.all(uv_box[1] >= patch.lower) and np.all(uv_box[0] <= patch.upper)
        ]
        if not possible:
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
                    ),
                    surface_programs.get(patch.index),
                )
                value = cache[key].value
                if isinstance(value, _CorrespondenceJetMap):
                    surface_programs[patch.index] = value.jets.surface_programs
            groups.setdefault(key, []).append((start, end))
    for key, members in groups.items():
        prepared = cache[key]
        starts = np.asarray([[start] for start, _ in members], dtype=np.float64)
        ends = np.asarray([[end] for _, end in members], dtype=np.float64)
        centers = np.asarray(
            [[0.5 * (start + end)] for start, end in members], dtype=np.float64
        )
        if (
            isinstance(prepared.value, _CorrespondenceJetMap)
            and isinstance(prepared.derivative, _CorrespondenceJetMap)
            and isinstance(prepared.second_derivative, _CorrespondenceJetMap)
            and prepared.value.jets is prepared.derivative.jets
            and prepared.value.jets is prepared.second_derivative.jets
        ):
            (
                (low, high),
                (derivative_low, derivative_high),
                (center_low, center_high),
                (second_low, second_high),
            ) = prepared.value.jets.residual_jets_batch(
                starts, ends, centers, budget=budget, preparations=preparations
            )
        else:
            low, high = evaluate_interval(
                prepared.value, centers, centers, budget=budget, preparations=preparations
            )
            derivative_low, derivative_high = evaluate_interval(
                prepared.derivative, starts, ends, budget=budget, preparations=preparations
            )
            center_low, center_high = evaluate_interval(
                prepared.derivative, centers, centers, budget=budget, preparations=preparations
            )
            second_low, second_high = evaluate_interval(
                prepared.second_derivative, starts, ends,
                budget=budget, preparations=preparations,
            )
        for row, (start, end) in enumerate(members):
            if (start, end) in contradicted:
                continue
            # A single source point outside the requested residual ball
            # disproves a whole-domain upper-bound claim. These are outward
            # center intervals, not sampled residuals. The supporting piece
            # must own that entire center UV enclosure.
            if np.any(low[row] > tolerance) or np.any(high[row] < -tolerance):
                parameter_piece = next(
                    item for item in parameter_segments if item.index == key[1]
                )
                patch = next(item for item in patches if item.index == key[2])
                midpoint = 0.5 * (start + end)
                uv = parameter_piece.evaluator.bounding_box(
                    midpoint, midpoint, preparations=preparations, budget=budget
                )
                if np.all(uv[0] >= patch.lower) and np.all(uv[1] <= patch.upper):
                    contradicted.add((start, end))
                    continue
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
            bounds[start, end] = max(
                previous, norm_upper(np.minimum(error, taylor_error))
            )
    return {
        interval: None if interval in contradicted else bounds.get(interval)
        for interval in intervals
    }


# ------------------------------------------------ carrier parameter maps


type ExactParameter = Fraction | PeriodOffset

# (cos, sin) of k quarter turns: frame rotations using only rational arithmetic.
_QUARTER_TURNS: tuple[tuple[Fraction, Fraction], ...] = (
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
    (Fraction(-1), Fraction(0)),
    (Fraction(0), Fraction(-1)),
)


@checked
def exact_parameter_parts(value: ExactParameter, /) -> tuple[Fraction, Fraction]:
    """``(rational, turns)`` of ``rational + turns * 2*pi``."""
    return (
        (value, Fraction(0))
        if isinstance(value, Fraction)
        else (value.rational, value.turns)
    )


@checked
def exact_parameter_affine(
    value: ExactParameter, scale: Fraction, offset: ExactParameter, /
) -> ExactParameter:
    """``scale * value + offset`` in Q + Q*(2*pi), canonicalized to a rational when possible."""
    rational, turns = exact_parameter_parts(value)
    offset_rational, offset_turns = exact_parameter_parts(offset)
    return period_offset(rational * scale + offset_rational, turns * scale + offset_turns)


@checked
def exact_parameter_sign(value: ExactParameter, /) -> int | None:
    """Return the sign, or None if the directed 2*pi enclosure cannot decide it."""
    rational, turns = exact_parameter_parts(value)
    if not turns:
        return (rational > 0) - (rational < 0)
    lower, upper = period_scalar_bounds(value)
    if float(lower) > 0.0:
        return 1
    if float(upper) < 0.0:
        return -1
    return None


@dataclass(frozen=True, slots=True)
class CurveParameterMap:
    """Prove carrier identity ``first(scale * s + offset) == second(s)`` for every ``s``.

    ``offset`` lies in Q + Q*(2*pi); a transcendental phase is never rationalized,
    and parameterization equality alone never implies a world-image identity.
    """

    scale: Fraction
    offset: ExactParameter

    def __post_init__(self) -> None:
        if not isinstance(self.scale, Fraction) or not self.scale:
            raise ValueError(
                f"CurveParameterMap.scale must be a nonzero rational scale; got {self.scale!r}"
            )
        if not isinstance(self.offset, (Fraction, PeriodOffset)):
            raise TypeError(
                f"CurveParameterMap.offset must belong to Q + Q*(2*pi); got {self.offset!r}"
            )

    def apply(self, value: ExactParameter, /) -> ExactParameter:
        return exact_parameter_affine(value, self.scale, self.offset)

    def inverse(self) -> CurveParameterMap:
        return CurveParameterMap(
            1 / self.scale,
            exact_parameter_affine(self.offset, -1 / self.scale, Fraction(0)),
        )


def _binary_vector(values: tuple[Fraction, ...], /) -> np.ndarray | None:
    """Return binary64 values only when every coefficient is represented without rounding."""
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
    match curve:
        case LineCurve() | BSplineCurve():
            return affine_curve_coefficients(
                curve, np.zeros((curve.ambient_dimension,), dtype=np.float64)
            )
        case _:
            return None


def _circle_frame(
    curve: CircleCurve, /
) -> tuple[tuple[Fraction, ...], Fraction, tuple[Fraction, ...], tuple[Fraction, ...]]:
    return (
        rational_vector(np.asarray(curve.center, dtype=np.float64).reshape(-1)),
        Fraction(float(curve.radius)),
        rational_vector(np.asarray(curve.first_axis, dtype=np.float64).reshape(-1)),
        rational_vector(np.asarray(curve.second_axis, dtype=np.float64).reshape(-1)),
    )


def _projective(vector: tuple[Fraction, ...], /) -> tuple[Fraction, ...] | None:
    pivot = next((index for index, value in enumerate(vector) if value), None)
    return None if pivot is None else tuple(value / vector[pivot] for value in vector)


@checked
def curve_support_key(curve: AbstractCurve | IntersectionCurve, /) -> tuple[object, ...]:
    """Return a world-image support key necessary for a carrier map.

    Lines use rational projective directions and pivot-plane points; circles
    use rational centers, radii and projective plane normals. Every other family is
    keyed by its carrier payload, so only identical sources compare.
    """

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
    return ("carrier", canonical_fingerprint(carrier_owner.carrier_payload(curve)))


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
    """Return a rational frame proof of ``second(s) == first(sigma*s + k*pi/2)``."""
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
                    sigma, period_offset(Fraction(0), Fraction(quarter, 4))
                )
    return None


@checked
def prove_curve_correspondence(
    first: AbstractCurve | IntersectionCurve,
    second: AbstractCurve | IntersectionCurve,
    /,
) -> CurveParameterMap | None:
    """Prove ``first(scale*s + offset) == second(s)`` from source coefficients.

    Identical carrier payloads are the identity. Lines (including degree-one
    rational splines) use rational affine coefficients; circles use rational centers,
    radius and quarter-turn/reflection frame algebra. A coincident support
    whose parameter map is not exactly representable returns None.
    """

    if first.ambient_dimension != second.ambient_dimension:
        return None
    if canonical_fingerprint(
        carrier_owner.carrier_payload(first)
    ) == canonical_fingerprint(carrier_owner.carrier_payload(second)):
        return CurveParameterMap(Fraction(1), Fraction(0))
    lines = _space_line(first), _space_line(second)
    if lines[0] is not None and lines[1] is not None:
        return _line_map(lines[0], lines[1])
    match (first, second):
        case (CircleCurve() as first_circle, CircleCurve() as second_circle):
            return _circle_map(first_circle, second_circle)
        case _:
            return None


def _reparameterized_spline(
    curve: BSplineCurve, scale: Fraction, offset: Fraction, /
) -> BSplineCurve | None:
    knots = _binary_vector(
        tuple(
            (knot - offset) / scale
            for knot in rational_vector(
                np.asarray(curve.knots, dtype=np.float64).reshape(-1)
            )
        )
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


@checked
def reparameterize_pcurve(
    curve: AbstractCurve | IntersectionPCurve,
    parameter_map: CurveParameterMap,
    /,
) -> AbstractCurve | IntersectionPCurve | None:
    """Return the same family evaluating ``curve(scale*t + offset)``, or None.

    No carrier is refitted: line origins/directions, spline knots and circle
    quarter-turn frames are transported only when every binary64 coefficient is
    representable without rounding. Affine and periodic UV operations keep their source gauge.
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
                rational_vector(np.asarray(curve.origin, dtype=np.float64).reshape(-1)),
                rational_vector(
                    np.asarray(curve.direction, dtype=np.float64).reshape(-1)
                ),
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
