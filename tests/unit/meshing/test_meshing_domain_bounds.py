#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry import (
    BSplineCurve,
    LineCurve,
    MeshingDomain,
    MeshingDomainBoundarySource,
    MeshingSurfacePatch,
    PatchCurveUse,
    PlanePatch,
    SpherePatch,
)
from phydrax.geometry._domain_chart_chain import (
    boundary_cell_ribbon_bounds,
    sampling_trim,
    verify_chart_chain,
)
from phydrax.geometry.brep._constructors import NormalizedTrimCurve
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment
from phydrax.meshing.qualification import sphere_surface
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.meshing_domain import (
    c0_tensor_spline_trim_domain,
    polygon_boundary_domain,
    SOURCE_ROUNDOFF_ATOL,
    SPHERE_PROJECTION_RTOL,
    square_chart_boundary_source,
)


def test_boundary_source_samples_and_distances_match_the_sphere() -> None:
    source = MeshingDomainBoundarySource(sphere_surface().domain, (0,), resolution=24)
    samples = source.boundary_samples(10_000)
    queries = np.asarray([[0.0, 0.0, 2.0], [0.3, -0.2, 0.1], [0.6, 0.6, 0.0]])
    distance = source.boundary_distance(queries)
    exact = np.abs(np.linalg.norm(queries, axis=1) - 1.0)

    assert samples.complete
    np.testing.assert_allclose(
        np.linalg.norm(samples.points, axis=1), 1.0, atol=SOURCE_ROUNDOFF_ATOL
    )
    assert source.boundary_samples(10).complete is False
    np.testing.assert_allclose(
        distance.upper, exact, rtol=SPHERE_PROJECTION_RTOL, atol=SOURCE_ROUNDOFF_ATOL
    )


def test_continuous_chart_cover_certifies_the_oriented_square_chain() -> None:
    complete = square_chart_boundary_source().boundary_chart_cover(64)
    assert complete.complete and complete.semantics == "certified"
    assert np.max(complete.deviation_bounds) < SOURCE_ROUNDOFF_ATOL
    assert not complete.findings


def test_continuous_chart_cover_refuses_an_omitted_triangle() -> None:
    source = square_chart_boundary_source()
    cells = source.chart_triangulations[0][3]
    missing = eqx.tree_at(
        lambda value: value.chart_triangulations[0][3], source, cells[:1]
    ).boundary_chart_cover(64)
    assert not missing.complete
    assert missing.semantics == "sampled"
    assert tuple(f.check for f in missing.findings) == ("source_trim_coverage_premise",)
    assert tuple(f.status for f in missing.findings) == ("unresolved",)
    assert missing.resource_counts == tuple(sorted(missing.resource_counts))
    assert tuple(row[0] for row in missing.resource_counts[:3]) == (
        "source_chart_restrictions:0",
        "source_trim_arcs:0",
        "source_trim_chain_pairs:0",
    )


def test_continuous_chart_cover_refuses_unproved_source_restrictions() -> None:
    source = square_chart_boundary_source()
    required = np.asarray((False, False, True, False), dtype=np.bool_)
    unproved = eqx.tree_at(
        lambda value: value.chart_triangulations[0][6], source, required
    ).boundary_chart_cover(64)
    assert not unproved.complete
    assert any(
        finding.check == "source_chart_restriction_authority"
        for finding in unproved.findings
    )
    assert tuple(f.check for f in unproved.findings) == (
        "source_chart_restriction_authority",
        "source_trim_coverage_premise",
    )
    assert tuple(f.status for f in unproved.findings) == ("unresolved", "unresolved")
    assert unproved.resource_counts == (("source_trim_arcs:0", 4, 65_536),)


def test_continuous_chart_cover_keeps_restriction_edge_refusal_phase() -> None:
    source = square_chart_boundary_source()
    invalid_edges = np.empty((0, 1), dtype=np.int64)
    refused = eqx.tree_at(
        lambda value: value.chart_triangulations[0][8], source, invalid_edges
    ).boundary_chart_cover(64)
    assert not refused.complete
    assert tuple(f.check for f in refused.findings) == (
        "source_chart_restriction_authority",
        "source_trim_coverage_premise",
    )
    assert tuple(f.status for f in refused.findings) == ("unresolved", "unresolved")
    assert refused.resource_counts == (("source_trim_arcs:0", 4, 65_536),)


def test_sphere_interpolation_bound_covers_interior_not_only_nodes() -> None:
    domain = sphere_surface().domain
    charts = np.asarray([[[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]]])
    corners = domain.evaluate(np.zeros((3,), dtype=np.int64), charts[0])
    weights = np.asarray(
        [
            [first / 20, second / 20, 1 - (first + second) / 20]
            for first in range(21)
            for second in range(21 - first)
        ]
    )
    exact = domain.evaluate(
        np.zeros((weights.shape[0],), dtype=np.int64), weights @ charts[0]
    )
    error = np.linalg.norm(exact - weights @ corners, axis=1)
    bound = domain.interpolation_bounds(0, charts)[0]
    assert 0.0 < np.max(error) <= bound < 0.025


def test_rotated_sphere_normal_bound_uses_sharp_physical_gram_evidence() -> None:
    base = sphere_surface().domain
    root = np.sqrt(0.5)
    domain = MeshingDomain(
        (
            MeshingSurfacePatch(
                SpherePatch((0, 0, 0), (root, root, 0), (-root, root, 0), (0, 0, 1), 1),
                base.patches[0].loops,
            ),
        ),
        base.curves,
        2,
        source_id="rotated-sphere",
        source_revision="r1",
    )
    charts = np.asarray([[[0, 0], [0.2, 0], [0, 0.2]]], dtype=np.float64)
    bound = domain.normal_turn_bounds(0, charts)[0]
    # A rigid rotation cannot change the unit-sphere normal Lipschitz bound.
    assert 0.4 <= bound < 0.400000001


def test_sampled_source_projection_respects_holes_and_outer_trims() -> None:
    outer = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    hole = np.asarray(((0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)), dtype=np.float64)
    source = MeshingDomainBoundarySource(
        polygon_boundary_domain((outer, hole)), (0,), resolution=5
    )
    queries = np.asarray(
        ((0.5, 0.5, 0.0), (-0.1, 0.5, 0.0), (0.2, 0.5, 1.0)), dtype=np.float64
    )
    distance = source.boundary_distance(queries)
    assert distance.semantics == "sampled"
    np.testing.assert_allclose(
        distance.upper,
        (0.1, 0.1, 1.0),
        rtol=SOURCE_ROUNDOFF_ATOL,
        atol=SOURCE_ROUNDOFF_ATOL,
    )


def test_boundary_samples_retain_narrow_trim_without_interior_grid_points() -> None:
    diamond = np.asarray(
        ((0.5, 0.0), (0.51, 0.5), (0.5, 1.0), (0.49, 0.5)), dtype=np.float64
    )
    source = MeshingDomainBoundarySource(
        polygon_boundary_domain((diamond,)), (0,), resolution=2
    )
    samples = source.boundary_samples(16)
    assert samples.complete
    for corner in diamond:
        assert np.any(np.all(samples.points[:, :2] == corner, axis=1))
    distance = source.boundary_distance(np.asarray(((0.5, 0.5, 1.0),), dtype=np.float64))
    np.testing.assert_allclose(
        distance.upper, (1.0,), rtol=SOURCE_ROUNDOFF_ATOL, atol=SOURCE_ROUNDOFF_ATOL
    )


def test_exact_affine_spline_trim_boundary_resolves_without_subdivision() -> None:

    polygon = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)), dtype=np.float64
    )
    plane = PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    uses = []
    for index, first in enumerate(polygon):
        last = polygon[(index + 1) % polygon.shape[0]]
        span = 1.0 if index % 2 == 0 else 2.0
        curve = BSplineCurve(
            np.stack((first, last)),
            np.ones((2,), dtype=np.float64),
            np.asarray((0.0, 0.0, span, span), dtype=np.float64),
            1,
        )
        carrier = NormalizedTrimCurve(
            CurveTrimSegment(curve, 0.0, span),
            jnp.zeros((2,), dtype=jnp.float64),
            jnp.ones((2,), dtype=jnp.float64),
            plane,
            None,
            None,
            None,
            None,
        )
        uses.append(PatchCurveUse(index, curve, 0.0, span, trim_curve=carrier))
    source = MeshingSurfacePatch(plane, (tuple(uses),))

    with coordinate_reference_budget() as budget:
        trim = sampling_trim(source, 6, 0.0, budget=budget)
    classified = trim.classify(
        np.asarray(((0.5, 2.0), (0.5, 1.0), (0.5, 2.1)), dtype=np.float64),
        maximum_depth=0,
    )
    np.testing.assert_array_equal(classified.resolved, (True, True, True))
    np.testing.assert_array_equal(classified.boundary, (True, False, False))
    np.testing.assert_array_equal(classified.inside, (False, True, False))
    np.testing.assert_array_equal(classified.refinements, 0)


def test_nonlinear_spline_trim_membership_uses_original_curve_not_chord_fit() -> None:

    top = BSplineCurve(
        np.asarray(((1.0, 2.0), (0.5, 2.2), (0.0, 2.0)), dtype=np.float64),
        np.ones((3,), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        2,
    )
    source = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 2.0)), 0.0, 1.0),
                PatchCurveUse(2, top, 0.0, 1.0),
                PatchCurveUse(3, LineCurve((0.0, 2.0), (0.0, -2.0)), 0.0, 1.0),
            ),
        ),
    )

    with coordinate_reference_budget() as budget:
        trim = sampling_trim(source, 3, 0.0, budget=budget)
    points = np.asarray(((0.5, 2.095), (0.5, 2.105)), dtype=np.float64)
    classified = trim.classify(points)
    np.testing.assert_array_equal(classified.resolved, (True, True))
    np.testing.assert_array_equal(classified.inside, (True, False))


def test_c0_tensor_source_interpolation_encloses_actual_chord_without_claiming_hessian() -> (
    None
):
    domain = c0_tensor_spline_trim_domain()
    triangle = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0)))
    surface = domain.patches[0].surface
    assert not surface.is_c1_on(((0.0, 0.0), (2.0, 1.0)))
    low, high = surface.derivative_bounds(((0.0, 0.0), (2.0, 1.0)), order=2)
    assert np.any(~np.isfinite(low)) and np.any(~np.isfinite(high))
    bound = domain.interpolation_bounds(0, triangle[None])[0]
    assert np.isfinite(bound) and 0.5 <= bound < 0.51
    barycentric = np.asarray(
        [
            (first / 16, second / 16, (16 - first - second) / 16)
            for first in range(17)
            for second in range(17 - first)
        ]
    )
    nodal = domain.evaluate(np.zeros(3, dtype=np.int64), triangle)
    charts = barycentric @ triangle
    actual = domain.evaluate(np.zeros(charts.shape[0], dtype=np.int64), charts)
    error = np.linalg.norm(actual - barycentric @ nodal, axis=1)
    assert np.all(error <= bound)


def test_c0_tensor_source_trim_ribbon_has_finite_certified_source_chain() -> None:

    domain = c0_tensor_spline_trim_domain()
    charts = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    points = domain.evaluate(np.zeros(4, dtype=np.int64), charts)
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    boundary = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)
    provenance = np.asarray([(0, row, 0.0, 1.0) for row in range(4)])
    findings, resources = [], []
    required = np.zeros((charts.shape[0],), dtype=np.bool_)

    with coordinate_reference_budget(queries_upper=2 * boundary.shape[0]) as budget:
        valid, _, ribbon, _ = verify_chart_chain(
            domain,
            0,
            charts,
            points,
            cells,
            boundary,
            provenance,
            required,
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            findings,
            resources,
            budget=budget,
        )
    assert valid and not findings
    assert ribbon.shape == (4,) and np.all(np.isfinite(ribbon))
    assert np.max(ribbon) >= 0.5


def test_tensor_source_first_jet_batch_preserves_closed_knot_queries() -> None:
    order = 1
    surface = c0_tensor_spline_trim_domain().patches[0].surface
    boxes = np.asarray(
        (
            ((0.0, 0.0), (2.0, 1.0)),
            ((1.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (1.0, 1.0)),
            ((1.0, 0.0), (1.5, 1.0)),
            ((0.2, 0.1), (0.8, 0.9)),
        )
    )
    individual = [surface.derivative_bounds(box, order=order) for box in boxes]
    lower, upper = surface.derivative_bounds_batch(boxes, order=order)
    np.testing.assert_array_equal(lower, np.stack([value[0] for value in individual]))
    np.testing.assert_array_equal(upper, np.stack([value[1] for value in individual]))
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert lower[1, 1, 0] <= 0.0 and upper[1, 1, 0] >= 1.0


def test_tensor_source_second_jet_batch_preserves_closed_knot_queries() -> None:
    order = 2
    surface = c0_tensor_spline_trim_domain().patches[0].surface
    boxes = np.asarray(
        (
            ((0.0, 0.0), (2.0, 1.0)),
            ((1.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (1.0, 1.0)),
            ((1.0, 0.0), (1.5, 1.0)),
            ((0.2, 0.1), (0.8, 0.9)),
        )
    )
    individual = [surface.derivative_bounds(box, order=order) for box in boxes]
    lower, upper = surface.derivative_bounds_batch(boxes, order=order)
    np.testing.assert_array_equal(lower, np.stack([value[0] for value in individual]))
    np.testing.assert_array_equal(upper, np.stack([value[1] for value in individual]))
    assert np.all(np.isneginf(lower[:4])) and np.all(np.isposinf(upper[:4]))
    assert np.all(np.isfinite(lower[4])) and np.all(np.isfinite(upper[4]))


@pytest.mark.parametrize(
    "edge_order",
    (slice(None), slice(None, None, -1)),
    ids=("forward-edges", "reversed-edges"),
)
def test_boundary_ribbons_apply_only_to_incident_cells(edge_order: slice) -> None:
    cells = np.asarray(((0, 1, 2), (2, 1, 3), (4, 5, 6), (7, 8, 9)), dtype=np.int64)
    boundary = np.asarray(((0, 1), (1, 3), (4, 5)), dtype=np.int64)
    actual = boundary_cell_ribbon_bounds(
        cells, boundary[:, edge_order], np.asarray((0.25, 0.75, 0.5))
    )
    np.testing.assert_array_equal(actual, (0.25, 0.75, 0.5, 0.0))


def test_boundary_ribbons_refuse_unaligned_edge_bounds() -> None:
    with pytest.raises(ValueError, match="MeshingDomain.boundary_ribbon_bounds"):
        boundary_cell_ribbon_bounds(
            np.asarray(((0, 1, 2),), dtype=np.int64),
            np.asarray(((0, 1), (1, 2)), dtype=np.int64),
            np.asarray((0.25,)),
        )
