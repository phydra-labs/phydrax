#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry import (
    AbstractTrimCurve,
    box_boundary_atlas,
    CurveTrimLoop,
    PolygonTrimLoop,
    TrimClassification,
    TrimDomain,
)
from phydrax.geometry._atlas import _affine_chord_within_tolerance, _certified_arcs
from phydrax.geometry.brep import (
    BRepTessellationPolicy,
    BSplineCurve,
    BSplineSurfacePatch,
    CurveTrimSegment,
    intersect_surface_regions,
    LineCurve,
    PlanePatch,
    SpherePatch,
    SurfaceRegion,
    TrimRootEndpoint,
)
from phydrax.geometry.brep._carrier_pieces import AffineTrimChordEvidence
from phydrax.geometry.brep._constructors import NormalizedTrimCurve
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection import TrimIntersectionRoot
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


class _UncertifiedAffineTrim(AbstractTrimCurve):
    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, 1.0

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        return jnp.stack((parameters, jnp.zeros_like(parameters)), axis=-1)

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        return np.asarray(((first, 0.0), (last, 0.0)), dtype=np.float64)


def test_prescribed_curve_cover_retains_source_partition_and_chords() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    curves = tuple(
        CurveTrimSegment(LineCurve(first, second - first), 0.0, 1.0)
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    )
    # Each unit segment has enclosure diagonal one; tolerance two admits one arc.
    loop = CurveTrimLoop(
        curves,
        tolerance=2.0,
        maximum_arcs=4,
        arc_parameters=tuple(np.asarray((0.0, 1.0), dtype=np.float64) for _ in curves),
        chord_vertices=vertices,
    )
    np.testing.assert_array_equal(loop.arc_curves, np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(loop.arc_first, np.zeros(4, dtype=np.float64))
    np.testing.assert_array_equal(loop.arc_last, np.ones(4, dtype=np.float64))
    np.testing.assert_array_equal(loop.source_chords, vertices)
    np.testing.assert_array_equal(loop.chords, vertices)
    assert loop.closure_gap == 0.0
    refined = CurveTrimLoop(
        curves,
        tolerance=2.0,
        maximum_arcs=8,
        arc_parameters=tuple(
            np.asarray((0.0, 0.5, 1.0), dtype=np.float64) for _ in curves
        ),
    )
    assert TrimDomain(loop).trim_id != TrimDomain(refined).trim_id


def test_polygon_box_classification_retains_closed_boundary_refusal() -> None:
    domain = TrimDomain(PolygonTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))))
    result = domain.classify_boxes(
        np.asarray(
            (
                ((0.25, 0.25), (0.75, 0.75)),
                ((1.25, 0.25), (1.75, 0.75)),
                ((0.0, 0.25), (0.25, 0.75)),
            ),
            dtype=np.float64,
        )
    )
    np.testing.assert_array_equal(result.inside, (True, False, False))
    np.testing.assert_array_equal(result.resolved, (True, True, False))
    np.testing.assert_array_equal(result.refinements, (0, 0, 0))


def test_boundary_atlas_host_metadata_retains_declared_values() -> None:
    atlas = box_boundary_atlas(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((2.0, 4.0, 6.0), dtype=np.float64),
        source_id="box-source",
    )
    np.testing.assert_array_equal(atlas.source_entity_ids, np.arange(6, dtype=np.int32))
    np.testing.assert_array_equal(atlas.orientation, (-1.0, 1.0, 1.0, -1.0, -1.0, 1.0))
    assert atlas.source_id == "box-source"
    assert atlas.physical_tags == ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def test_boundary_atlas_refuses_noncanonical_source_identifier() -> None:
    with pytest.raises(ValueError, match="BoundaryAtlas.source_id"):
        box_boundary_atlas(
            np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            np.asarray((2.0, 4.0, 6.0), dtype=np.float64),
            source_id=" box-source ",
        )


def test_trim_base_keeps_unestablished_continuity_and_period_semantics() -> None:
    trim = _UncertifiedAffineTrim()
    assert not trim.is_c1_on(0.0, 1.0)
    assert trim.native_period_symbol() is None
    assert trim.coefficient_enclosures() == ()


class _NearAffineTrim(AbstractTrimCurve):
    @property
    def parameter_interval(self) -> tuple[float, float]:
        return 0.0, 1.0

    def evaluate(
        self,
        parameters: Array,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> Array:
        return jnp.stack((parameters, 1e-12 * parameters * (1.0 - parameters)), axis=-1)

    def enclosure(
        self,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
    ) -> np.ndarray:
        parameters = (Fraction(first), Fraction(last))
        values = tuple(Fraction(1e-12) * value * (1 - value) for value in parameters)
        maximum = Fraction(1e-12) / 4 if first <= 0.5 <= last else max(values)
        return np.asarray(
            (
                (first, np.nextafter(float(min(values)), -np.inf)),
                (last, np.nextafter(float(maximum), np.inf)),
            ),
            dtype=np.float64,
        )


def _normalized_square_curves() -> tuple[NormalizedTrimCurve, ...]:
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    patch = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    return tuple(
        NormalizedTrimCurve(
            curve=CurveTrimSegment(LineCurve(first, second - first), 0.0, 1.0),
            lower=jnp.asarray((0.0, 0.0), dtype=jnp.float64),
            extent=jnp.asarray((1.0, 1.0), dtype=jnp.float64),
            patch=patch,
            start_root=None,
            end_root=None,
            start_endpoint=None,
            end_endpoint=None,
        )
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    )


def test_normalized_affine_chord_proof_admits_witnessed_tolerance_without_subdivision() -> (
    None
):
    curves = _normalized_square_curves()
    loop = CurveTrimLoop(
        curves,
        tolerance=4.263256414560601e-14,
        maximum_arcs=4,
    )
    assert loop.curves == curves
    np.testing.assert_array_equal(loop.arc_curves, np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(loop.arc_first, np.zeros(4))
    np.testing.assert_array_equal(loop.arc_last, np.ones(4))
    np.testing.assert_array_equal(
        loop.chords, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    assert np.all(np.linalg.norm(loop.arc_upper - loop.arc_lower, axis=1) >= 1.0)


def test_near_affine_curve_keeps_generic_enclosure_subdivision() -> None:
    curve = _NearAffineTrim()
    with coordinate_source_request((curve,)) as budget:
        with pytest.raises(ValueError, match="CurveTrimLoop.maximum_arcs"):
            _certified_arcs(curve, 0, 0.0, 1.0, 4.263256414560601e-14, 4, budget=budget)


@pytest.mark.parametrize("forgery", ("source-id", "affine-coordinate", "endpoint-box"))
def test_affine_chord_proof_refuses_forged_source_or_endpoint_evidence(
    forgery: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    curve = _normalized_square_curves()[0]
    owner_method = NormalizedTrimCurve.affine_chord_evidence

    def forged(
        self: NormalizedTrimCurve,
        first: float,
        last: float,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
    ) -> AffineTrimChordEvidence | None:
        evidence = owner_method(
            self, first, last, budget=budget, preparations=preparations
        )
        assert evidence is not None
        match forgery:
            case "source-id":
                return replace(evidence, source_id="forged-carrier")
            case "affine-coordinate":
                return replace(evidence, nominal_start=(Fraction(99), Fraction(0)))
            case "endpoint-box":
                return replace(
                    evidence,
                    start_lower=np.asarray((99.0, 0.0)),
                    start_upper=np.asarray((100.0, 0.0)),
                )
            case _:
                raise AssertionError(
                    f"atlas_fixture.forgery must name a supported regression case; got {forgery!r}"
                )

    monkeypatch.setattr(NormalizedTrimCurve, "affine_chord_evidence", forged)
    with coordinate_source_request((curve,)) as budget:
        with pytest.raises(
            ValueError, match="affine_chord_evidence|AffineTrimChordEvidence.endpoints"
        ):
            _certified_arcs(curve, 0, 0.0, 1.0, 4.263256414560601e-14, 4, budget=budget)


def _normalized_line_trim(
    segment: CurveTrimSegment,
    /,
    *,
    lower: tuple[float, float] = (0.0, 0.0),
    extent: tuple[float, float] = (1.0, 1.0),
) -> NormalizedTrimCurve:
    return NormalizedTrimCurve(
        curve=segment,
        lower=jnp.asarray(lower, dtype=jnp.float64),
        extent=jnp.asarray(extent, dtype=jnp.float64),
        patch=PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        start_root=None,
        end_root=None,
        start_endpoint=None,
        end_endpoint=None,
    )


def test_affine_chord_proof_retains_nonunit_normalization_and_reversed_parameters() -> (
    None
):
    segment = CurveTrimSegment(LineCurve((1.0, 2.0), (2.0, 4.0)), 2.0, 4.0, reversed=True)
    curve = _normalized_line_trim(segment, lower=(3.0, 6.0), extent=(2.0, 4.0))
    with coordinate_source_request((curve,)) as budget:
        evidence = curve.affine_chord_evidence(0.0, 1.0, budget=budget)
        assert evidence is not None
        assert evidence.nominal_start == (Fraction(3), Fraction(3))
        assert evidence.nominal_end == (Fraction(1), Fraction(1))
        arcs = _certified_arcs(
            curve, 0, 0.0, 1.0, 4.263256414560601e-14, 1, budget=budget
        )
    assert len(arcs) == 1
    np.testing.assert_array_equal(arcs[0][3], (3.0, 3.0))
    assert segment.first == 2.0 and segment.last == 4.0 and segment.reversed


def test_affine_chord_proof_retains_genuine_endpoint_root_uncertainty() -> None:
    line = LineCurve((0.0, 0.0), (3.0, 0.0))
    crossing = LineCurve((1.0, 0.0), (0.0, 1.0))
    root = TrimIntersectionRoot(
        CurveTrimSegment(line, -1.0, 1.0),
        CurveTrimSegment(crossing, -1.0, 1.0),
        parameter_lower=np.asarray((0.6, 0.4)),
        parameter_upper=np.asarray((0.75, 0.6)),
    )
    endpoint = TrimRootEndpoint(root, "first")
    segment = CurveTrimSegment(line, 1.0 / 3.0, 1.0, first_root=endpoint)
    curve = _normalized_line_trim(segment, extent=(3.0, 1.0))
    with coordinate_source_request((curve,)) as budget:
        evidence = curve.affine_chord_evidence(0.0, 1.0, budget=budget)
        assert evidence is not None
        assert evidence.endpoint_root_ids == (root.root_id,)
        assert evidence.start_lower[0] < evidence.start_upper[0]
        arcs = _certified_arcs(
            curve, 0, 0.0, 1.0, 4.263256414560601e-14, 1, budget=budget
        )
        assert len(arcs) == 1
        assert not _affine_chord_within_tolerance(
            curve,
            0.0,
            1.0,
            arcs[0][3],
            np.asarray((1.0, 0.0)),
            1e-30,
            budget=budget,
            preparations=None,
        )


def _spline_evidence_curve(family: str, scale: float, /) -> BSplineCurve:
    match family:
        case "affine-degree-one":
            controls = scale * np.asarray(((2.0, -2.0), (1.0, 0.317)))
            weights = np.asarray((1.0, 1.0))
            knots = np.asarray((2.0, 2.0, 4.0, 4.0))
            degree = 1
        case "unequal-positive-weights":
            controls = scale * np.asarray(((2.0, -2.0), (1.0, 0.317)))
            weights = np.asarray((1.0, 2.0))
            knots = np.asarray((2.0, 2.0, 4.0, 4.0))
            degree = 1
        case "nonlinear-degree-two":
            controls = scale * np.asarray(((2.0, -2.0), (1.5, 0.2), (1.0, 0.317)))
            weights = np.asarray((1.0, 1.0, 1.0))
            knots = np.asarray((2.0, 2.0, 2.0, 4.0, 4.0, 4.0))
            degree = 2
        case "three-control-degree-one":
            controls = scale * np.asarray(((2.0, -2.0), (1.5, 0.2), (1.0, 0.317)))
            weights = np.asarray((1.0, 1.0, 1.0))
            knots = np.asarray((2.0, 2.0, 3.0, 4.0, 4.0))
            degree = 1
        case "unclamped-degree-one":
            controls = scale * np.asarray(((2.0, -2.0), (1.0, 0.317)))
            weights = np.asarray((1.0, 1.0))
            knots = np.asarray((1.0, 2.0, 4.0, 5.0))
            degree = 1
        case _:
            raise AssertionError(
                f"atlas_fixture.family must name a supported spline family; got {family!r}"
            )
    return BSplineCurve(controls, weights, knots, degree)


@pytest.mark.parametrize(
    "family",
    (
        "affine-degree-one",
        "unequal-positive-weights",
        "nonlinear-degree-two",
        "three-control-degree-one",
        "unclamped-degree-one",
    ),
)
def test_atlas_affine_spline_capability_requires_exact_source_predicate(
    family: str,
) -> None:
    scale = float(np.ldexp(1.0, 20))
    spline = _spline_evidence_curve(family, scale)
    segment = CurveTrimSegment(spline, 2.0, 4.0)
    curve = _normalized_line_trim(
        segment, lower=(-2.0 * scale, -2.0 * scale), extent=(4.0 * scale, 4.0 * scale)
    )
    endpoints = np.asarray(
        jax.device_get(curve.evaluate(jnp.asarray((0.0, 1.0), dtype=jnp.float64)))
    )
    with coordinate_source_request((curve,)) as budget:
        evidence = curve.affine_chord_evidence(0.0, 1.0, budget=budget)
        admitted = _affine_chord_within_tolerance(
            curve,
            0.0,
            1.0,
            endpoints[0],
            endpoints[1],
            0.01,
            budget=budget,
            preparations=None,
        )
        match family:
            case "affine-degree-one":
                assert evidence is not None
                assert admitted
                arcs = _certified_arcs(curve, 0, 0.0, 1.0, 0.01, 1, budget=budget)
                assert len(arcs) == 1
                assert arcs[0][1:3] == (0.0, 1.0)
                np.testing.assert_array_equal(arcs[0][3], endpoints[0])
                assert curve.curve is segment and segment.curve is spline
            case (
                "unequal-positive-weights"
                | "nonlinear-degree-two"
                | "three-control-degree-one"
                | "unclamped-degree-one"
            ):
                assert evidence is None
                assert not admitted
            case _:
                raise AssertionError(
                    f"atlas_fixture.family must name a supported spline family; got {family!r}"
                )


@pytest.mark.parametrize(
    ("query", "inside", "resolved", "refinements"),
    (
        (((-0.1, -0.1), (0.1, 0.1)), False, False, 0),
        (((0.05, 0.8), (0.1, 0.85)), True, True, 1),
    ),
    ids=("junction-overlap", "non-junction-refinement"),
)
def test_curve_box_exclusion_refuses_junction_without_wasted_refinements(
    query: tuple[tuple[float, float], tuple[float, float]],
    inside: bool,
    resolved: bool,
    refinements: int,
) -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)), dtype=np.float64
    )
    curves = tuple(
        CurveTrimSegment(LineCurve(first, second - first), 0.0, 1.0)
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    )
    domain = TrimDomain(CurveTrimLoop(curves, tolerance=2.0, maximum_arcs=4))
    result = domain.classify_boxes(
        np.asarray((query,), dtype=np.float64),
        maximum_depth=1,
        maximum_refinements=2,
    )
    np.testing.assert_array_equal(result.inside, (inside,))
    np.testing.assert_array_equal(result.resolved, (resolved,))
    np.testing.assert_array_equal(result.refinements, (refinements,))


def _charged_implicit_circle_domain() -> TrimDomain:
    """Retain the cap circle on a spline plane whose coefficient bounds need work."""
    sphere = SurfaceRegion(
        SpherePatch(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 1.0
        ),
        np.asarray(((0.0, -np.pi / 2), (2 * np.pi, np.pi / 2)), dtype=np.float64),
    )
    plane = SurfaceRegion(
        BSplineSurfacePatch(
            np.asarray(
                (
                    ((-2.0, -2.0, 0.3), (-2.0, 2.0, 0.3)),
                    ((2.0, -2.0, 0.3), (2.0, 2.0, 0.3)),
                ),
                dtype=np.float64,
            ),
            np.ones((2, 2), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
            1,
            1,
        ),
        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
    )
    result = intersect_surface_regions(sphere, plane)
    assert result.complete and len(result.curves) == 1
    branch = result.curves[0]
    assert branch.closed
    match branch.second.patch:
        case BSplineSurfacePatch():
            pass
        case _:
            raise AssertionError(
                "atlas_fixture.circle_surface must retain its spline generating surface; got another patch family"
            )
    pcurve = branch.p_curve("second")
    first, last = pcurve.parameter_interval
    policy = BRepTessellationPolicy(realize=False)
    return TrimDomain(
        CurveTrimLoop(
            (CurveTrimSegment(pcurve, first, last),),
            tolerance=1.0 / policy.trim_samples_per_edge,
            maximum_arcs=policy.maximum_triangles * 3,
            relative_closure_tolerance=1e-9,
        )
    )


def _interior_arc_probe(loop: CurveTrimLoop, /) -> np.ndarray:
    """Choose a positive-width chord-interior box outside every junction band."""
    for start, end, lower, upper in zip(
        loop.source_chords,
        np.roll(loop.source_chords, -1, axis=0),
        loop.arc_lower,
        loop.arc_upper,
        strict=True,
    ):
        center = 0.5 * (start + end)
        half_width = float(np.max(np.abs(end - start))) / 8
        if half_width == 0.0:
            continue
        query = np.stack((center - half_width, center + half_width))
        junction_disjoint = np.any(
            (query[1] < loop.junction_lower) | (query[0] > loop.junction_upper), axis=1
        )
        if np.all(junction_disjoint):
            assert np.all(query[1] >= lower) and np.all(query[0] <= upper)
            return query
    raise AssertionError(
        "atlas_fixture.circle_probe must lie inside an arc outside junction bands; got no separated probe"
    )


def test_implicit_trim_containment_refuses_exhausted_caller_budget() -> None:
    domain = _charged_implicit_circle_domain()
    match domain.outer:
        case CurveTrimLoop() as loop:
            query = _interior_arc_probe(loop)
        case _:
            raise AssertionError(
                "atlas_fixture.trim_loop must retain its source curve; got another loop kind"
            )
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with pytest.raises(CoordinateEnclosureResourceError) as refused:
            with budget.measurement():
                domain.classify_boxes(
                    query[None, :],
                    budget=budget,
                    preparations=TrimRequestPreparations(),
                )
        assert refused.value.limit == 0
        assert refused.value.requested > 0
        assert refused.value.completed == 0
        assert budget.work_units == 0
        assert refused.value.phase_evidence is not None
        assert refused.value.phase_evidence.work_units == 0


@pytest.mark.parametrize(
    ("query", "inside"),
    (
        (((0.4, 0.4), (0.6, 0.6)), True),
        (((1.4, 0.4), (1.6, 0.6)), False),
    ),
    ids=("inside", "outside"),
)
def test_disjoint_certified_spline_arcs_reuse_earned_cover_with_zero_budget(
    query: tuple[tuple[float, float], tuple[float, float]], inside: bool
) -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    curves = tuple(
        CurveTrimSegment(
            BSplineCurve(
                np.stack((first, second)),
                np.ones(2, dtype=np.float64),
                np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
                1,
            ),
            0.0,
            1.0,
        )
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    )
    loop = CurveTrimLoop(curves, tolerance=2.0, maximum_arcs=4)
    box = np.asarray(query, dtype=np.float64)
    assert np.all(np.any((box[1] < loop.arc_lower) | (box[0] > loop.arc_upper), axis=1))
    polygon = TrimDomain(PolygonTrimLoop(vertices)).classify_boxes(box[None, :])
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        result = TrimDomain(loop).classify_boxes(
            box[None, :], budget=budget, preparations=TrimRequestPreparations()
        )
        assert budget.work_units == 0
    np.testing.assert_array_equal(result.inside, (inside,))
    np.testing.assert_array_equal(result.resolved, (True,))
    np.testing.assert_array_equal(result.refinements, (0,))
    np.testing.assert_array_equal(result.inside, polygon.inside)
    np.testing.assert_array_equal(result.resolved, polygon.resolved)


def test_overlapping_cells_share_new_arc_splits_per_refinement_level() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)), dtype=np.float64
    )
    curves = tuple(
        CurveTrimSegment(LineCurve(first, second - first), 0.0, 1.0)
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    )
    domain = TrimDomain(CurveTrimLoop(curves, tolerance=2.0, maximum_arcs=4))
    queries = np.asarray(
        (
            ((0.05, 0.8), (0.1, 0.85)),
            ((0.05, 0.8), (0.1, 0.85)),
            ((0.125, 0.375), (0.15, 0.4)),
            ((0.125, 0.375), (0.15, 0.4)),
        ),
        dtype=np.float64,
    )
    independent = tuple(
        domain.classify_boxes(query[None, :], maximum_depth=2, maximum_refinements=10)
        for query in queries
    )
    preparations = TrimRequestPreparations()
    with coordinate_source_request((domain, queries), rows=queries.shape[0]) as budget:
        shared = domain.classify_boxes(
            queries,
            maximum_depth=2,
            maximum_refinements=10,
            budget=budget,
            preparations=preparations,
        )
        repeated = domain.classify_boxes(
            queries,
            maximum_depth=2,
            maximum_refinements=10,
            budget=budget,
            preparations=preparations,
        )
    np.testing.assert_array_equal(
        shared.inside, np.concatenate([row.inside for row in independent])
    )
    np.testing.assert_array_equal(
        shared.resolved, np.concatenate([row.resolved for row in independent])
    )
    np.testing.assert_array_equal(shared.refinements, (1, 0, 1, 0))
    np.testing.assert_array_equal(repeated.inside, shared.inside)
    np.testing.assert_array_equal(repeated.resolved, shared.resolved)
    np.testing.assert_array_equal(repeated.refinements, np.zeros(4, dtype=np.int32))


def _diamond_domain(offset: float = 0.0) -> TrimDomain:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)), dtype=np.float64
    ) + np.asarray((offset, 0.0))
    curves = tuple(
        CurveTrimSegment(LineCurve(first, second - first), 0.0, 1.0)
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
    )
    return TrimDomain(CurveTrimLoop(curves, tolerance=2.0, maximum_arcs=4))


# Repeated, nearby and on-boundary probes against several arcs and a junction.
_DIAMOND_PROBES = np.asarray(
    (
        (0.5, 0.52),
        (0.5, 0.52),
        (0.5, 0.48),
        (0.5, 1.49),
        (0.5, 1.51),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (0.0, 1.0),
        (0.0, 0.0),
        (-0.75, 1.2),
    ),
    dtype=np.float64,
)


_CLASSIFICATION_FIELDS = ("inside", "boundary", "resolved", "winding", "refinements")


def _assert_same_classification(
    actual: TrimClassification, expected: dict[str, np.ndarray]
) -> None:
    for name in _CLASSIFICATION_FIELDS:
        np.testing.assert_array_equal(getattr(actual, name), expected[name])


def _independent_classification(
    domain: TrimDomain, maximum_depth: int
) -> dict[str, np.ndarray]:
    """Classify each probe in its own fresh request."""
    rows = [
        domain.classify(point[None, :], maximum_depth=maximum_depth)
        for point in _DIAMOND_PROBES
    ]
    return {
        name: np.concatenate([np.asarray(getattr(row, name)) for row in rows])
        for name in _CLASSIFICATION_FIELDS
    }


@pytest.mark.parametrize("maximum_depth", [0, 1, 8])
def test_point_classification_retains_request_arc_boxes_without_changing_queries(
    maximum_depth: int,
) -> None:
    domain = _diamond_domain()
    independent = _independent_classification(domain, maximum_depth)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((domain, _DIAMOND_PROBES)) as budget:
        # Earned box-classification facts and a shallower point pass share the
        # same request arc tree; every later query still decides on its own.
        domain.classify_boxes(
            np.asarray((((0.05, 0.8), (0.1, 0.85)),), dtype=np.float64),
            maximum_depth=2,
            maximum_refinements=10,
            budget=budget,
            preparations=preparations,
        )
        domain.classify(
            _DIAMOND_PROBES, maximum_depth=0, budget=budget, preparations=preparations
        )
        for _ in range(2):
            shared = domain.classify(
                _DIAMOND_PROBES,
                maximum_depth=maximum_depth,
                budget=budget,
                preparations=preparations,
            )
            _assert_same_classification(shared, independent)
        # A different (shifted) source loop in the same request earns its own
        # boxes; retained facts are bound to the loop object, not its shape.
        shifted = _diamond_domain(0.25)
        _assert_same_classification(
            shifted.classify(
                _DIAMOND_PROBES,
                maximum_depth=maximum_depth,
                budget=budget,
                preparations=preparations,
            ),
            _independent_classification(shifted, maximum_depth),
        )


def test_point_classification_retains_private_copies_of_reused_provider_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = _diamond_domain()
    enclosure = CurveTrimSegment.enclosure
    scratch = np.empty((2, 2), dtype=np.float64)

    def reused_buffer(
        self: CurveTrimSegment, first: float, last: float, /, **kwargs: Any
    ) -> np.ndarray:
        # A provider returning one mutable scratch buffer for every interval.
        scratch[...] = enclosure(self, first, last, **kwargs)
        return scratch

    monkeypatch.setattr(CurveTrimSegment, "enclosure", reused_buffer)
    independent = _independent_classification(domain, 8)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((domain, _DIAMOND_PROBES)) as budget:
        for _ in range(2):
            _assert_same_classification(
                domain.classify(
                    _DIAMOND_PROBES,
                    maximum_depth=8,
                    budget=budget,
                    preparations=preparations,
                ),
                independent,
            )


def test_point_classification_refuses_nonfinite_provider_boxes_every_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = _diamond_domain()
    query = np.asarray(((0.5, 0.52),), dtype=np.float64)

    def unbounded(
        self: CurveTrimSegment, first: float, last: float, /, **kwargs: Any
    ) -> np.ndarray:
        return np.asarray(((-np.inf, -np.inf), (np.inf, np.inf)), dtype=np.float64)

    monkeypatch.setattr(CurveTrimSegment, "enclosure", unbounded)
    preparations = TrimRequestPreparations()
    with coordinate_source_request((domain, query)) as budget:
        for _ in range(2):
            with pytest.raises(ValueError, match=r"finite \(2, 2\) boxes"):
                domain.classify(query, budget=budget, preparations=preparations)
        # Box classification keeps its bounded unresolved refusal for the same
        # unretained provider bounds.
        boxes = domain.classify_boxes(
            np.asarray((((0.45, 0.5), (0.55, 0.54)),), dtype=np.float64),
            maximum_depth=2,
            maximum_refinements=4,
            budget=budget,
            preparations=preparations,
        )
    np.testing.assert_array_equal(boxes.resolved, (False,))


def test_curved_box_classification_shares_request_chords_without_changing_decisions() -> None:
    domain = _charged_implicit_circle_domain()
    match domain.outer:
        case CurveTrimLoop() as loop:
            probe = _interior_arc_probe(loop)
        case _:
            raise AssertionError(
                "atlas_fixture.trim_loop must retain its source curve; got another loop kind"
            )
    center = 0.5 * (probe[0] + probe[1])
    half = 0.5 * (probe[1] - probe[0])
    # Shrinking and shifted boxes revisit left/right descendants whose first
    # source parameters coincide with earlier visited parents.
    queries = np.asarray(
        [
            np.stack((center + shift * half - half / scale, center + shift * half + half / scale))
            for scale in (1.0, 2.0, 8.0, 64.0)
            for shift in (-1.0, 0.0, 1.0)
        ],
        dtype=np.float64,
    )
    independent = [
        domain.classify_boxes(query[None, :], maximum_depth=12, maximum_refinements=512)
        for query in queries
    ]
    preparations = TrimRequestPreparations()
    with coordinate_source_request((domain, queries), rows=queries.shape[0]) as budget:
        shared = [
            domain.classify_boxes(
                query[None, :],
                maximum_depth=12,
                maximum_refinements=512,
                budget=budget,
                preparations=preparations,
            )
            for query in queries
        ]
        repeated = [
            domain.classify_boxes(
                query[None, :],
                maximum_depth=12,
                maximum_refinements=512,
                budget=budget,
                preparations=preparations,
            )
            for query in queries
        ]
    for fresh, first, again in zip(independent, shared, repeated, strict=True):
        np.testing.assert_array_equal(first.inside, fresh.inside)
        np.testing.assert_array_equal(first.resolved, fresh.resolved)
        np.testing.assert_array_equal(again.inside, fresh.inside)
        np.testing.assert_array_equal(again.resolved, fresh.resolved)
        np.testing.assert_array_equal(again.refinements, 0)
    assert any(bool(row.resolved[0]) for row in shared)
