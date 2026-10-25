"""Native source-trim mean-value premises, independent of point sampling."""

import math
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_operands import (
    _immutable_array_layout,
    seal_coordinate_operand_array,
)
from phydrax.geometry import _domain_chart_chain as charts
from phydrax.geometry._interval_enclosure import (
    interval_add,
    interval_multiply,
    interval_subtract,
)
from phydrax.geometry.brep import (
    _intersection as intersections,
    _intersection_curve as curves,
)
from phydrax.geometry.brep._constructors import NormalizedTrimCurve
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.geometry.brep._intersection_kernel import krawczyk
from phydrax.geometry.brep._placed import PlacedCurve
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


geometry = phx.geometry


def _two_chart_branch(periodic: bool) -> curves.IntersectionCurve:
    """Construct real source equations and obtain every graph witness natively."""
    second = geometry.SurfaceRegion(
        geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((-2, -2), (2, 2)),
    )
    if periodic:
        first = geometry.SurfaceRegion(
            geometry.CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0),
            ((0.0, -1), (math.tau, 1)),
        )
        parameters = ((5.5, math.tau), (0.0, 0.5))

        def node(t: float) -> tuple[float, float, float, float]:
            return (t, 0.0, math.cos(t), math.sin(t))

        lower = np.asarray(((5.4, -0.1, -1.2, -1.2), (-0.1, -0.1, -1.2, -1.2)))
        upper = np.asarray(((6.4, 0.1, 1.2, 1.2), (0.6, 0.1, 1.2, 1.2)))
        shifts = np.asarray(((-math.tau, 0, 0, 0), (0, 0, 0, 0)))
    else:
        first = geometry.SurfaceRegion(
            geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 0, 1)),
            ((-1, -1), (2, 1)),
        )
        parameters = ((0.0, 0.25), (0.25, 1.0))

        def node(t: float) -> tuple[float, float, float, float]:
            return (t, 0.0, t, 0.0)

        # The free coordinate x=u needs strict inclusion in a wider dependent
        # box; graph-axis and source-use ranges remain unchanged.
        lower = np.asarray(((-0.1, -0.1, -0.2, -0.1), (0.15, -0.1, 0.05, -0.1)))
        upper = np.asarray(((0.35, 0.1, 0.45, 0.1), (1.1, 0.1, 1.2, 0.1)))
        shifts = np.zeros((2, 4))
    prepared = intersections.prepare(curves.SurfacePairSystem(first.patch, second.patch), 4, 1)
    with coordinate_source_request((first, second, lower, upper)) as budget:
        witnesses = [
            krawczyk(prepared, lo[None], hi[None], parameter_axis=0, budget=budget)
            for lo, hi in zip(lower, upper, strict=True)
        ]
        assert all(bool(witness.certified[0]) for witness in witnesses)
        branch = curves.IntersectionCurve(
            first, second,
            chart_axes=np.zeros(2, dtype=np.int32),
            chart_start=np.asarray([node(a) for a, _ in parameters]),
            chart_end=np.asarray([node(b) for _, b in parameters]),
            chart_pieces=np.zeros((2, 2), dtype=np.int32),
            box_lower=lower, box_upper=upper,
            preconditioners=np.stack([w.preconditioner[0] for w in witnesses]),
            contraction=np.asarray([w.contraction[0] for w in witnesses]),
            certified=np.ones(2, dtype=np.bool_), transition_shifts=shifts,
            closed=False, start_kind="boundary", end_kind="boundary", budget=budget,
        )
    assert branch.fully_certified
    return branch


@pytest.fixture(scope="module")
def planar_branch() -> curves.IntersectionCurve:
    return _two_chart_branch(False)


@pytest.fixture(scope="module")
def periodic_branch() -> curves.IntersectionCurve:
    return _two_chart_branch(True)


def _arc(
    carrier: geometry.CurveTrimSegment | NormalizedTrimCurve,
    preparations: TrimRequestPreparations,
    *,
    certified: bool = False,
    reverse: bool = False,
) -> charts._SourceTrimArc:
    return charts._SourceTrimArc(
        carrier, Fraction(int(reverse)), Fraction(int(not reverse)),
        "test-native-trim", preparations, -1 if reverse else 1, False,
        mean_value_certified=certified,
    )


def _direct(
    arc: charts._SourceTrimArc,
    curve: tuple[np.ndarray, np.ndarray],
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    start = arc.bounds(Fraction(0), Fraction(0), budget)
    span = interval_subtract(arc.bounds(Fraction(1), Fraction(1), budget), start)
    result = interval_subtract(curve, interval_add(start, interval_multiply(charts._WHOLE_ARC.t, span)))
    return np.asarray(result[0]), np.asarray(result[1])


def test_piecewise_c1_shared_root_seam_proves_mean_value(
    planar_branch: curves.IntersectionCurve,
) -> None:
    segment = geometry.CurveTrimSegment(planar_branch.p_curve("first"), 0.0, 2.0)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((segment,)) as budget:
        assert not segment.is_c1_on(0.0, 1.0, preparations=preparations, budget=budget)
        assert charts._source_piecewise_trim_continuity(segment, preparations, budget, {})
        arc = _arc(segment, preparations, certified=True)
        cache = {}
        curve, jet = charts._arc_jets(arc, charts._WHOLE_ARC, cache, budget, derivative=True)
        result, _ = charts._chord_deviation(arc, charts._WHOLE_ARC, curve, jet, cache, budget)
        direct = _direct(arc, curve, budget)
    # Exact source: u(t)=t/2 for t<=1/2, 3t/2-1/2 afterwards.
    # Its chord deviation has precisely the range [-1/4, 0]. Unequal
    # one-sided velocities are allowed; a missing C0 join would not be.
    assert result[0][0, 0] <= -0.25 and result[1][0, 0] >= 0.0
    assert result[1][0, 0] - result[0][0, 0] < direct[1][0, 0] - direct[0][0, 0]


@pytest.mark.parametrize("first,last", ((0.0, 1.0), (1.0, 2.0)))
def test_periodic_seam_endpoint_uses_restricted_chart_support(
    periodic_branch: curves.IntersectionCurve, first: float, last: float,
) -> None:
    segment = geometry.CurveTrimSegment(periodic_branch.p_curve("first"), first, last)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((segment,)) as budget:
        assert charts._source_piecewise_trim_continuity(segment, preparations, budget, {})


def test_periodic_uv_jump_refuses_even_with_finite_piece_jets(
    periodic_branch: curves.IntersectionCurve,
) -> None:
    segment = geometry.CurveTrimSegment(periodic_branch.p_curve("first"), 0.0, 2.0)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((segment,)) as budget:
        assert all(np.all(np.isfinite(part)) for part in segment.derivative_bounds(
            0.0, 1.0, preparations=preparations, budget=budget
        ))
        with pytest.raises(charts.SourceChartMapError, match="source_trim_uv_transition_discontinuous"):
            charts._source_piecewise_trim_continuity(segment, preparations, budget, {})
        # The same atlas is C0 in physical space, including the period transport.
        charts._source_intersection_continuity(
            periodic_branch, 0.0, 2.0, 0, 1, preparations, budget, {}
        )


def test_uncertified_c0_spline_keeps_direct_enclosure() -> None:
    # Legal multiplicity-degree knots are C0, not C1. This arc has not been
    # issued continuity evidence, so even valid piece jets cannot tighten it.
    source = geometry.BSplineCurve(
        ((0, 0), (0, 0), (2, 0)), (1, 1, 1),
        (0, 0, 0.5, 1, 1), 1,
    )
    segment = geometry.CurveTrimSegment(source, 0.0, 1.0)
    preparations = TrimRequestPreparations()
    arc = _arc(segment, preparations)
    with coordinate_source_request((segment,)) as budget:
        assert not segment.is_c1_on(0.0, 1.0, preparations=preparations, budget=budget)
        cache = {}
        curve, jet = charts._arc_jets(arc, charts._WHOLE_ARC, cache, budget, derivative=True)
        assert jet is not None
        assert all(np.all(np.isfinite(part)) for part in jet)
        result, _ = charts._chord_deviation(arc, charts._WHOLE_ARC, curve, jet, cache, budget)
        expected = _direct(arc, curve, budget)
        np.testing.assert_array_equal(result, expected)
    # Exact piecewise residual: -2t before 1/2, 2t-2 afterwards, range [-1,0].
    assert result[0][0, 0] <= -1.0 and result[1][0, 0] >= 0.0


@pytest.mark.parametrize("reverse", (False, True))
def test_root_defined_period_endpoint_keeps_fixed_affine_parameter(reverse: bool) -> None:
    circle = geometry.CircleCurve((0, 0), (1, 0), (0, 1), 1.0)
    endpoint = NativePeriodEndpoint(circle, turns=Fraction(1, 2))
    segment = geometry.CurveTrimSegment(circle, 0.0, endpoint.parameter, last_root=endpoint)
    patch = geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    carrier = NormalizedTrimCurve(segment, jnp.zeros(2), jnp.ones(2), patch, None, None, None, endpoint)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((carrier,)) as budget:
        assert carrier.is_c1_on(0.0, 1.0, preparations=preparations, budget=budget)
        whole_jet = carrier.derivative_bounds(0.0, 1.0, preparations=preparations, budget=budget)
        assert all(np.all(np.isfinite(part)) for part in whole_jet)
        lower, upper = segment.carrier_parameter_enclosure(1.0, 1.0, preparations=preparations, budget=budget)
        assert lower < upper  # pi is an exact source atom, not its binary representative.
        arc = _arc(carrier, preparations, certified=True, reverse=reverse)
        cache = {}
        curve, jet = charts._arc_jets(arc, charts._WHOLE_ARC, cache, budget, derivative=True)
        result, _ = charts._chord_deviation(arc, charts._WHOLE_ARC, curve, jet, cache, budget)
    # Full half-circle y residual is sin(pi*t), exactly [0,1], both orientations.
    assert result[0][0, 1] <= 0.0 and result[1][0, 1] >= 1.0


@pytest.mark.parametrize("placed", (False, True))
def test_unproved_c0_native_closed_edge_refuses_before_gap_jets(placed: bool) -> None:
    source = geometry.BSplineCurve(
        ((0, 0, 0), (0, 0, 0), (0.001, 0, 0)),
        (1, 1, 1), (0, 0, 0.5, 1, 1), 1,
    )
    if placed:
        source = PlacedCurve(source, np.eye(3), np.zeros(3))
    preparations = TrimRequestPreparations()
    vertices = tuple(
        charts._SourceAuthoredVertex(np.asarray(point), None, str(i), preparations)
        for i, point in enumerate(((0.0, 0.0, 0.0), (0.001, 0.0, 0.0)))
    )
    edge = charts._SourceClosedEdge(
        source, (0.0, 1.0), (None, None), (vertices[0], vertices[1]),
        (0.0, 0.0), "c0-edge", preparations,
    )
    with coordinate_source_request((source,)) as budget:
        assert all(np.all(np.isfinite(part)) for part in edge.derivative(Fraction(0), Fraction(1), budget))
        with pytest.raises(charts.SourceChartMapError, match="source_gap_native_edge_continuity_unproved"):
            charts._source_closed_edge_continuity(edge, budget, {})


@pytest.mark.parametrize("reverse", (False, True))
def test_gap_residual_keeps_root_defined_native_and_trim_endpoints(reverse: bool) -> None:
    native = geometry.CircleCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0)
    pcurve = geometry.CircleCurve((0, 0), (1, 0), (0, 1), 1.0)
    native_root = NativePeriodEndpoint(native, turns=Fraction(1, 2))
    trim_root = NativePeriodEndpoint(pcurve, turns=Fraction(1, 2))
    segment = geometry.CurveTrimSegment(
        pcurve, 0.0, trim_root.parameter, last_root=trim_root
    )
    patch = geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    preparations = TrimRequestPreparations()
    with coordinate_source_request((native, segment, patch)) as budget:
        points = seal_coordinate_operand_array(
            np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float64),
            budget=budget,
            copy=True,
        )
        vertices = tuple(
            charts._SourceAuthoredVertex(
                point,
                None,
                str(i),
                preparations,
                {"layout": charts._AuthoredPointLayout(_immutable_array_layout(point))},
            )
            for i, point in enumerate(points)
        )
        native_edge = charts._SourceClosedEdge(
            native, (0.0, native_root.parameter), (None, native_root),
            (vertices[0], vertices[1]), (0.0, 0.0), "root-defined-half-circle", preparations,
        )
        arc = _arc(segment, preparations, certified=True, reverse=reverse)
        gap = charts._SourceGapArc(
            arc, native_edge, Fraction(int(reverse)), Fraction(int(not reverse)),
            patch, 1.0e-12,
        )
        charts._source_closed_edge_continuity(native_edge, budget, {})
        whole_value, whole_jet = gap.residuals(
            charts._WHOLE_ARC, {}, budget, value=True, jacobian=True
        )
        # Both exact traces are the same half-circle with the same exact pi
        # endpoint atom, not two independently rounded endpoint parameters.
        for enclosure in (whole_value, whole_jet):
            assert np.all(np.isfinite(enclosure))
            assert np.all(enclosure[0] <= 0.0) and np.all(enclosure[1] >= 0.0)
        for t in (Fraction(0), Fraction(1)):
            value, _ = gap.residuals(
                charts._EdgeRows(
                    np.zeros(1, dtype=np.int64), (t,), (t,),
                    (np.asarray([[float(t)]]), np.asarray([[float(t)]])),
                    (np.ones((1, 1)), np.ones((1, 1))),
                ),
                {}, budget, value=True, jacobian=False,
            )
            assert np.all(value[0] <= 0.0) and np.all(value[1] >= 0.0)


def test_stored_atlas_certificate_bits_do_not_establish_continuity(
    planar_branch: curves.IntersectionCurve,
) -> None:
    corrupted = eqx.tree_at(
        lambda value: value.chart_start_lower,
        planar_branch, planar_branch.chart_start_lower + 0.01,
    )
    assert corrupted.fully_certified
    segment = geometry.CurveTrimSegment(corrupted.p_curve("first"), 0.0, 2.0)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((segment,)) as budget:
        with pytest.raises(charts.SourceChartMapError, match="source_trim_atlas_continuity_unproved"):
            charts._source_piecewise_trim_continuity(segment, preparations, budget, {})


@pytest.mark.parametrize("dimension", (2, 3))
def test_discontinuous_spline_is_rejected_by_native_carrier_owner(dimension: int) -> None:
    start, end = (0.0,) * dimension, (0.001, *(0.0,) * (dimension - 1))
    with pytest.raises(ValueError, match="spline.knot_multiplicities"):
        geometry.BSplineCurve(
            (start, start, end, end), (1, 1, 1, 1),
            (0, 0, 0.5, 0.5, 1, 1), 1,
        )
