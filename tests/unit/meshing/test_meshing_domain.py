#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import numpy as np
import pytest

import phydrax as phx
from examples._native_surface_sources import capped_cylinder, folded_plates, sphere, torus
from phydrax.geometry import (
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    LineCurve,
    MeshingDomain,
    MeshingDomainBoundarySource,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
    PlanePatch,
    SpherePatch,
)
from phydrax.meshing._domain import compile_surface_domain


M = phx.meshing


def _scope(domain: Any, dimension: int, entities: Any, revision: str = "r1") -> Any:
    return M.MeshingScope(
        domain.source_id,
        revision,
        M.MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(entities, dtype=np.int64),
    )


def _square(curves: tuple[int, int, int, int], shift: float = 0.0) -> MeshingSurfacePatch:
    """Unit square patch of the plane z = 0 whose right side may be shifted."""

    bottom, right, top, left = curves
    return MeshingSurfacePatch(
        PlanePatch((shift, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(bottom, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(right, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
                PatchCurveUse(top, LineCurve((0.0, 1.0), (1.0, 0.0)), 1.0, 0.0),
                PatchCurveUse(left, LineCurve((0.0, 0.0), (0.0, 1.0)), 1.0, 0.0),
            ),
        ),
    )


def test_strata_identities_and_incidence_follow_declarations() -> None:
    can = capped_cylinder().domain
    seam_uses = can.curve_uses(2)

    assert can.corner_count == 2
    assert [can.entity_id(dimension, 1) for dimension in (0, 1, 2)] == [
        "r1:corner:1",
        "r1:curve:1",
        "r1:surface:1",
    ]
    assert can.entity_set_id(1) == "can:curves"
    assert [patch for patch, _ in seam_uses] == [0, 0]
    assert {(use.first, use.last) for _, use in seam_uses} == {(0.0, 1.5), (1.5, 0.0)}
    assert [patch for patch, _ in can.curve_uses(0)] == [0, 1]
    np.testing.assert_array_equal(can.patch_curves(0), (0, 1, 2))
    # Every patch bounds the interior on its negative (inner) side only.
    np.testing.assert_array_equal(can.patch_regions, [[0, -1], [0, -1], [0, -1]])
    np.testing.assert_allclose(can.corner_points, [[0.8, 0.0, 0.0], [0.8, 0.0, 1.5]])


def test_oriented_normals_report_regularity_and_pole_singularity() -> None:
    domain = sphere().domain
    charts = np.asarray([[0.3, 0.2], [2.0, -1.0], [1.0, 0.5 * np.pi]])
    normals, regular = domain.oriented_normals(np.zeros((3,), dtype=np.int64), charts)
    points = domain.evaluate(np.zeros((3,), dtype=np.int64), charts)

    np.testing.assert_array_equal(regular, (True, True, False))
    np.testing.assert_allclose(normals[:2], points[:2], atol=1.0e-12)
    np.testing.assert_array_equal(normals[2], 0.0)


def test_projection_returns_closest_points_with_status() -> None:
    domain = torus().domain
    queries = np.asarray([[2.9, 0.0, 0.1], [0.0, -2.3, -0.4]])
    seeds = np.asarray([[0.1, 0.2], [4.5, 5.5]])
    projection = domain.project(queries, np.zeros((2,), dtype=np.int64), seeds)
    ring = np.hypot(queries[:, 0], queries[:, 1]) - 2.0
    exact = np.abs(np.hypot(ring, queries[:, 2]) - 0.6)

    np.testing.assert_array_equal(projection.converged, (True, True))
    np.testing.assert_allclose(projection.distances, exact, rtol=1.0e-9)
    np.testing.assert_allclose(
        np.linalg.norm(projection.points - queries, axis=1), exact, rtol=1.0e-9
    )


def test_shared_curve_that_patches_do_not_reproduce_is_refused() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    first = _square((0, 1, 2, 3))
    second = MeshingSurfacePatch(
        PlanePatch((1.0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(4, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(5, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
                PatchCurveUse(6, LineCurve((0.0, 1.0), (1.0, 0.0)), 1.0, 0.0),
                PatchCurveUse(
                    1,
                    BSplineCurve(
                        ((0.0, 0.0), (1.0e-3, 0.5), (0.0, 1.0)),
                        (1.0, 1.0, 1.0),
                        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                        2,
                    ),
                    1.0,
                    0.0,
                ),
            ),
        ),
    )
    extra = (MeshingDomainCurve(1, 4), MeshingDomainCurve(4, 5), MeshingDomainCurve(5, 2))

    with pytest.raises(ValueError, match="does not reproduce curve 1"):
        MeshingDomain(
            (first, second), curves + extra, 6, source_id="gap", source_revision="r1"
        )


def test_inconsistently_oriented_neighbors_are_refused() -> None:
    plates = folded_plates().domain
    flipped = MeshingSurfacePatch(
        plates.patches[1].surface, plates.patches[1].loops, reversed=True
    )

    with pytest.raises(ValueError, match="inconsistent orientations"):
        MeshingDomain(
            (plates.patches[0], flipped),
            plates.curves,
            6,
            source_id="plates",
            source_revision="r1",
        )


def test_pole_side_and_region_closure_are_validated() -> None:
    patch = SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    half = 0.5 * np.pi
    not_a_pole = (
        PatchPoleUse(0, (0.0, -1.0), (2.0 * np.pi, -1.0)),
        PatchCurveUse(0, LineCurve((2.0 * np.pi, 0.0), (0.0, 1.0)), -1.0, half),
        PatchPoleUse(1, (2.0 * np.pi, half), (0.0, half)),
        PatchCurveUse(0, LineCurve((0.0, 0.0), (0.0, 1.0)), half, -1.0),
    )
    with pytest.raises(ValueError, match="does not collapse"):
        MeshingDomain(
            (MeshingSurfacePatch(patch, (not_a_pole,)),),
            (MeshingDomainCurve(0, 1),),
            2,
            source_id="band",
            source_revision="r1",
        )
    plates = folded_plates().domain
    with pytest.raises(ValueError, match="not bounded by a closed surface"):
        MeshingDomain(
            plates.patches,
            plates.curves,
            6,
            source_id="plates",
            source_revision="r1",
            regions=(MeshingDomainRegion("open", ((0, 1), (1, 1))),),
        )


def test_domain_identity_tracks_geometry_and_revision() -> None:
    base = sphere().domain
    moved = MeshingDomain(
        (
            MeshingSurfacePatch(
                SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.5),
                base.patches[0].loops,
            ),
        ),
        base.curves,
        2,
        source_id="sphere",
        source_revision="r1",
    )

    assert sphere().domain.domain_id == base.domain_id
    assert moved.domain_id != base.domain_id
    assert sphere(revision="r2").domain.domain_id != base.domain_id


def test_compiled_constraints_share_curves_and_corners() -> None:
    can = capped_cylinder().domain
    surfaces = _scope(can, 2, (0, 1, 2))
    seam = _scope(can, 1, (2,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        surfaces,
        size_controls=(
            M.UniformSizeControl(surfaces, 0.4, strength=M.SizeControlStrength.SOFT),
            M.UniformSizeControl(seam, 0.1, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(
            M.ProtectedFeature(
                _scope(can, 2, (1,)), M.FeatureKind.SURFACE, maximum_deviation=0.01
            ),
        ),
    )
    compiled = compile_surface_domain(can, specification)
    junctions = {
        junction.name: junction.endpoints for junction in compiled.curve_request.junctions
    }

    np.testing.assert_array_equal(compiled.curves, (0, 1, 2))
    np.testing.assert_array_equal(compiled.corners, (0, 1))
    np.testing.assert_allclose(compiled.curve_sizes, (0.4, 0.4, 0.1))
    # The bottom cap's fidelity bound reaches the circle it shares.
    np.testing.assert_allclose(compiled.curve_deviations, (0.01, np.inf, np.inf))
    assert junctions == {
        "corner:0": ((0, "end"), (0, "start"), (2, "start")),
        "corner:1": ((1, "end"), (1, "start"), (2, "end")),
    }


def test_stale_control_scope_is_refused_before_compilation() -> None:
    domain = sphere().domain
    surfaces = _scope(domain, 2, (0,))
    stale = _scope(domain, 2, (0,), revision="r0")

    with pytest.raises(ValueError, match="share the top-level source binding"):
        M.SurfaceMeshingSpec(
            M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
            surfaces,
            size_controls=(M.UniformSizeControl(stale, 0.3),),
        )


@pytest.mark.parametrize(
    ("dimension", "entities", "kind"),
    (
        (1, (1,), M.FeatureKind.CURVE),
        (0, (1,), M.FeatureKind.CORNER),
        (2, (2,), M.FeatureKind.SURFACE),
        (1, (99,), M.FeatureKind.CURVE),
        (2, (1,), M.FeatureKind.CORNER),
    ),
)
def test_hard_protected_source_scope_refuses_unrepresented_entities(
    dimension: int,
    entities: tuple[int, ...],
    kind: M.FeatureKind,
) -> None:
    domain = capped_cylinder().domain
    selected = _scope(domain, 2, (1,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(M.UniformSizeControl(selected, 0.4),),
        protected_features=(
            M.ProtectedFeature(
                _scope(domain, dimension, entities), kind, maximum_deviation=0.01
            ),
        ),
    )
    from phydrax.meshing._domain import surface_domain_issues

    assert surface_domain_issues(domain, specification)
    with pytest.raises(M.MeshingFailure) as refused:
        compile_surface_domain(domain, specification)
    assert refused.value.category is M.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED
    assert refused.value.provider_code == "protected_source_scope"


def test_selected_closed_curve_keeps_both_authored_endpoint_incidents() -> None:
    domain = capped_cylinder().domain
    selected = _scope(domain, 2, (1,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(M.UniformSizeControl(selected, 0.4),),
        protected_features=(
            M.ProtectedFeature(_scope(domain, 0, (0,)), M.FeatureKind.CORNER),
            M.ProtectedFeature(
                _scope(domain, 1, (0,)), M.FeatureKind.CURVE, maximum_deviation=0.01
            ),
        ),
    )
    compiled = compile_surface_domain(domain, specification)
    np.testing.assert_array_equal(compiled.curves, (0,))
    np.testing.assert_array_equal(compiled.corners, (0,))
    np.testing.assert_array_equal(compiled.curve_deviations, (0.01,))
    assert compiled.curve_request.junctions[0].endpoints == ((0, "end"), (0, "start"))


def test_boundary_source_samples_and_distances_match_the_sphere() -> None:
    source = MeshingDomainBoundarySource(sphere().domain, (0,), resolution=24)
    samples = source.boundary_samples(10_000)
    queries = np.asarray([[0.0, 0.0, 2.0], [0.3, -0.2, 0.1], [0.6, 0.6, 0.0]])
    distance = source.boundary_distance(queries)
    exact = np.abs(np.linalg.norm(queries, axis=1) - 1.0)

    assert samples.complete
    np.testing.assert_allclose(np.linalg.norm(samples.points, axis=1), 1.0, atol=1e-12)
    assert source.boundary_samples(10).complete is False
    np.testing.assert_allclose(distance.upper, exact, rtol=1.0e-8, atol=1.0e-12)


def test_continuous_chart_cover_refuses_an_omitted_triangle() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    domain = MeshingDomain(
        (_square((0, 1, 2, 3)),), curves, 4, source_id="square", source_revision="r1"
    )
    charts = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points = domain.evaluate(np.zeros((4,), dtype=np.int64), charts)
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    boundary = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    provenance = np.asarray(
        [[0, 0, 0, 1], [0, 1, 0, 1], [0, 2, 1, 0], [0, 3, 1, 0]], dtype=np.float64
    )
    restrictions = (
        np.zeros((charts.shape[0],), dtype=np.bool_),
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
    )
    complete = MeshingDomainBoundarySource(
        domain,
        (0,),
        chart_triangulations=(
            (0, charts, points, cells, boundary, provenance, *restrictions),
        ),
    ).boundary_chart_cover(64)
    missing = MeshingDomainBoundarySource(
        domain,
        (0,),
        chart_triangulations=(
            (0, charts, points, cells[:1], boundary, provenance, *restrictions),
        ),
    ).boundary_chart_cover(64)
    required = np.asarray((False, False, True, False), dtype=np.bool_)
    unproved = MeshingDomainBoundarySource(
        domain,
        (0,),
        chart_triangulations=(
            (
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
            ),
        ),
    ).boundary_chart_cover(64)
    assert complete.complete and complete.semantics == "certified"
    assert np.max(complete.deviation_bounds) < 1e-12
    assert not unproved.complete
    assert any(
        finding.check == "source_chart_restriction_authority"
        for finding in unproved.findings
    )
    assert not missing.complete


def test_sphere_interpolation_bound_covers_interior_not_only_nodes() -> None:
    domain = sphere().domain
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


def test_localized_source_curvature_preserves_physical_principal_curvature() -> None:
    controls = np.asarray(
        [
            [[0, 0, 0], [0, 1, 0]],
            [[0.5, 0, 0], [0.5, 1, 0]],
            [[1, 0, 1], [1, 1, 1]],
        ],
        dtype=np.float64,
    )
    patch = BSplineSurfacePatch(
        controls,
        np.ones((3, 2), dtype=np.float64),
        np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float64),
        np.asarray([0, 0, 1, 1], dtype=np.float64),
        2,
        1,
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, _square((0, 1, 2, 3)).loops),),
        tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0))),
        4,
        source_id="parabolic-sheet",
        source_revision="r1",
    )
    scope = _scope(domain, 2, (0,))
    angle = 0.5
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 10, strength=M.SizeControlStrength.SOFT),
            M.CurvatureSizeControl(scope, angle),
        ),
    )
    compiled = compile_surface_domain(domain, specification)
    charts = np.asarray([[0.1, 0.5], [0.8, 0.5]], dtype=np.float64)
    points = domain.evaluate(np.zeros((2,), dtype=np.int64), charts)
    sizes = compiled.source_sizing.evaluate(domain, 0, charts, points)
    # Independent graph formula for z=u²: kappa=2/(1+4u²)^(3/2).
    expected = np.sin(angle / 2) * (1 + 4 * charts[:, 0] ** 2) ** 1.5
    np.testing.assert_allclose(sizes, expected, rtol=1e-10)


def test_source_sizing_rejects_contradictory_hard_targets_before_growth() -> None:
    domain = sphere().domain
    scope = _scope(domain, 2, (0,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 0.3),
            M.UniformSizeControl(scope, 0.4),
        ),
    )
    with pytest.raises(ValueError, match="hard size-control targets conflict"):
        compile_surface_domain(domain, specification)


def test_source_sizing_refuses_scratch_before_materializing_query_plan() -> None:
    domain = sphere().domain
    scope = _scope(domain, 2, (0,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(M.UniformSizeControl(scope, 0.3),),
        limits=M.MeshingLimits(maximum_scratch_bytes=1024),
    )
    with pytest.raises(M.MeshingFailure) as refused:
        compile_surface_domain(domain, specification)
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED


def test_rotated_sphere_normal_bound_uses_sharp_physical_gram_evidence() -> None:
    base = sphere().domain
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


def test_source_branch_root_survives_curve_use_and_physical_atlas_lowering() -> None:
    import jax.numpy as jnp

    from phydrax.geometry import intersect_surface_regions, SurfaceRegion
    from phydrax.geometry._meshing_domain import _PhysicalCurveMap
    from phydrax.geometry.brep._intersection import (
        BranchRootEndpoint,
        CurveSurfaceIntersectionRoot,
    )

    first = SurfaceRegion(
        PlanePatch(
            np.asarray((-1.0, 0.0, 0.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
        np.asarray(((0.0, 0.0), (1.0, 1.0))),
    )
    second = SurfaceRegion(
        PlanePatch(
            np.asarray((0.0, -1.0, -1.0)),
            np.asarray((0.0, 3.0, 0.0)),
            np.asarray((0.0, 0.0, 2.0)),
        ),
        np.asarray(((0.0, 0.0), (1.0, 1.0))),
    )
    intersection = intersect_surface_regions(first, second)
    assert intersection.complete
    (branch,) = intersection.curves
    edge = LineCurve(np.asarray((-1.0, 0.37, 0.0)), np.asarray((2.0, 0.0, 0.0)))
    root = CurveSurfaceIntersectionRoot(
        edge,
        second,
        parameter_lower=np.asarray((0.49, 0.45, 0.49)),
        parameter_upper=np.asarray((0.51, 0.46, 0.51)),
    )
    image = np.asarray((0.5, 0.37, 1.37 / 3.0, 0.5))
    charts = np.flatnonzero(
        np.all(
            (np.asarray(branch.box_lower) <= image)
            & (image <= np.asarray(branch.box_upper)),
            axis=1,
        )
    )
    endpoint = BranchRootEndpoint(
        root,
        branch,
        int(charts[0]),
        source_pcurve=LineCurve(np.asarray((0.0, 0.37)), np.asarray((1.0, 0.0))),
        source_side="first",
        source_first=0.0,
        source_last=1.0,
    )
    lower, upper = endpoint.parameter_enclosure()
    nominal = 0.5 * (lower + upper)
    use = PatchCurveUse(
        0,
        branch.p_curve("first"),
        nominal,
        float(branch.num_charts),
        first_root=endpoint,
    )
    mapping = _PhysicalCurveMap(
        (branch,),
        np.asarray(((nominal, float(branch.num_charts)),)),
        ((endpoint, None),),
    )
    point = np.asarray(
        mapping.map(jnp.asarray((0,), dtype=jnp.int32), jnp.asarray(((0.0,),)))
    )
    # Independent source-plane/edge intersection, including transported scalar
    # root uncertainty rather than a replacement nominal junction definition.
    np.testing.assert_allclose(point[0], (0.0, 0.37, 0.0), rtol=0.0, atol=1.0e-10)
    assert use.first_root is endpoint
    assert np.all(np.isfinite(mapping.endpoint_errors))
    assert mapping.endpoint_errors[0, 0] > 0.0


def _polygon_boundary_domain(polygons: tuple[np.ndarray, ...]) -> MeshingDomain:
    loops: list[tuple[PatchCurveUse, ...]] = []
    curves: list[MeshingDomainCurve] = []
    corner = 0
    for polygon in polygons:
        uses = []
        for index, first in enumerate(polygon):
            last = polygon[(index + 1) % polygon.shape[0]]
            uses.append(
                PatchCurveUse(
                    corner + index,
                    LineCurve(first, last - first),
                    0.0,
                    1.0,
                )
            )
            curves.append(
                MeshingDomainCurve(
                    corner + index,
                    corner + (index + 1) % polygon.shape[0],
                )
            )
        loops.append(tuple(uses))
        corner += polygon.shape[0]
    return MeshingDomain(
        (MeshingSurfacePatch(PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), tuple(loops)),),
        tuple(curves),
        corner,
        source_id="authored-trim",
        source_revision="trim-r1",
    )


def test_sampled_source_projection_respects_holes_and_outer_trims() -> None:
    outer = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    hole = np.asarray(((0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)), dtype=np.float64)
    source = MeshingDomainBoundarySource(
        _polygon_boundary_domain((outer, hole)), (0,), resolution=5
    )
    queries = np.asarray(
        ((0.5, 0.5, 0.0), (-0.1, 0.5, 0.0), (0.2, 0.5, 1.0)), dtype=np.float64
    )
    distance = source.boundary_distance(queries)
    assert distance.semantics == "sampled"
    np.testing.assert_allclose(distance.upper, (0.1, 0.1, 1.0), rtol=1e-12, atol=1e-12)


def test_boundary_samples_retain_narrow_trim_without_interior_grid_points() -> None:
    diamond = np.asarray(
        ((0.5, 0.0), (0.51, 0.5), (0.5, 1.0), (0.49, 0.5)), dtype=np.float64
    )
    source = MeshingDomainBoundarySource(
        _polygon_boundary_domain((diamond,)), (0,), resolution=2
    )
    samples = source.boundary_samples(16)
    assert samples.complete
    for corner in diamond:
        assert np.any(np.all(samples.points[:, :2] == corner, axis=1))
    distance = source.boundary_distance(np.asarray(((0.5, 0.5, 1.0),), dtype=np.float64))
    np.testing.assert_allclose(distance.upper, (1.0,), rtol=1e-12, atol=1e-12)


def test_curved_trim_queries_survive_original_source_recipe_restoration() -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.geometry.brep import NativePeriodEndpoint
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    outer = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    hole = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.2)
    patch = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(
                    0,
                    outer,
                    0.0,
                    2 * np.pi,
                    first_root=NativePeriodEndpoint(outer, turns=0),
                    last_root=NativePeriodEndpoint(outer, turns=1),
                ),
            ),
            (
                PatchCurveUse(
                    1,
                    hole,
                    2 * np.pi,
                    0.0,
                    first_root=NativePeriodEndpoint(hole, turns=1),
                    last_root=NativePeriodEndpoint(hole, turns=0),
                ),
            ),
        ),
    )
    domain = MeshingDomain(
        (patch,),
        (MeshingDomainCurve(0, 0), MeshingDomainCurve(1, 1)),
        2,
        source_id="authored-annulus",
        source_revision="circle-r1",
    )
    source = MeshingDomainBoundarySource(domain, (0,), resolution=4)
    register_meshing_source_artifacts()
    validate_meshing_source_closure(source)
    recipe = model_structure_recipe(source)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(source, recipe, prefix="trim-source"),
        prefix="trim-source",
    )
    assert isinstance(restored, MeshingDomainBoundarySource)
    validate_meshing_source_closure(restored)
    assert restored.source_id == source.source_id
    assert restored.source_revision == source.source_revision
    assert restored.source_scope_id == source.source_scope_id
    assert isinstance(restored.domain.patches[0].loops[0][0].pcurve, CircleCurve)
    queries = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.99 * np.cos(np.pi / 4), 0.99 * np.sin(np.pi / 4), 0.5),
            (0.15, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        restored.boundary_distance(queries).upper,
        (0.2, 0.5, 0.05),
        rtol=1e-12,
        atol=1e-12,
    )
    radii = np.linalg.norm(restored.boundary_samples(100).points[:, :2], axis=1)
    assert np.all(radii >= 0.2 - 1e-14)
    assert np.all(radii <= 1.0 + 1e-14)


def test_exact_affine_spline_trim_boundary_resolves_without_subdivision() -> None:
    import jax.numpy as jnp

    from phydrax.geometry._meshing_domain import _sampling_trim
    from phydrax.geometry.brep._constructors import _NormalizedTrimCurve
    from phydrax.geometry.brep._intersection_curve import CurveTrimSegment

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
        carrier = _NormalizedTrimCurve(
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
    trim = _sampling_trim(source, 6, 0.0)
    classified = trim.classify(
        np.asarray(((0.5, 2.0), (0.5, 1.0), (0.5, 2.1)), dtype=np.float64),
        maximum_depth=0,
    )
    np.testing.assert_array_equal(classified.resolved, (True, True, True))
    np.testing.assert_array_equal(classified.boundary, (True, False, False))
    np.testing.assert_array_equal(classified.inside, (False, True, False))
    np.testing.assert_array_equal(classified.refinements, 0)


def test_nonlinear_spline_trim_membership_uses_original_curve_not_chord_fit() -> None:
    from phydrax.geometry._meshing_domain import _sampling_trim

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
    trim = _sampling_trim(source, 3, 0.0)
    points = np.asarray(((0.5, 2.095), (0.5, 2.105)), dtype=np.float64)
    classified = trim.classify(points)
    np.testing.assert_array_equal(classified.resolved, (True, True))
    np.testing.assert_array_equal(classified.inside, (True, False))


@pytest.mark.parametrize("kind", (M.FeatureKind.CORNER, M.FeatureKind.CURVE))
@pytest.mark.parametrize(
    "feature_revision", ("r1", "r0"), ids=("current-feature", "stale-feature")
)
def test_planar_source_corner_protection_requires_the_authored_dimension(
    kind: M.FeatureKind,
    feature_revision: str,
) -> None:
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id="protected-square",
    )
    source = M.NativePlanarSource(region, "r1")
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "square-region",
        np.asarray((0,)),
    )
    corner = M.MeshingScope(
        source.source_id,
        feature_revision,
        M.MeshingEntityKind.GEOMETRY,
        0,
        "square-corners",
        np.asarray((0,)),
    )
    feature = M.ProtectedFeature(corner, kind, maximum_deviation=0.0)

    def specification() -> M.SurfaceMeshingSpec:
        return M.SurfaceMeshingSpec(
            M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                M.UniformSizeControl(scope, 0.3, strength=M.SizeControlStrength.SOFT),
            ),
            protected_features=(feature,),
            planar_embedding=phx.geometry.PlanarEmbedding(
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            ),
        )

    if feature_revision != source.source_revision:
        with pytest.raises(ValueError, match="top-level source binding"):
            specification()
        return
    request = specification()
    provider = M.NativeMeshingProvider(
        M.NativeMeshingOptions("planar_constrained_delaunay")
    )
    if kind is M.FeatureKind.CURVE:
        with pytest.raises(M.MeshingFailure):
            provider.plan(
                source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
            )
        return
    result = provider.plan(
        source,
        request,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()
    assert (
        dict(result.compliance.achieved)[
            f"protected:{feature.feature_id}:maximum_deviation"
        ]
        == 0.0
    )
    assert result.compliance.passed
    points = np.asarray(result.mesh.coordinates)
    assert np.any(np.all(points == (0.0, 0.0), axis=1))


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_unknown_size_control_entity_refuses_before_source_queries(
    dimension: int,
) -> None:
    from phydrax.meshing._domain import surface_domain_issues

    domain = capped_cylinder().domain
    selected = _scope(domain, 2, (0, 1, 2))
    unknown = _scope(domain, dimension, (99,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(
            M.UniformSizeControl(selected, 0.4),
            M.UniformSizeControl(unknown, 0.2),
        ),
    )
    assert surface_domain_issues(domain, specification)
    with pytest.raises(M.MeshingFailure) as refused:
        compile_surface_domain(domain, specification)
    assert refused.value.category is M.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED
    assert refused.value.provider_code == "unknown_source_entity"


def test_source_neighborhood_refuses_at_the_original_shared_query_root() -> None:
    from phydrax.meshing._domain import SourcePatchNeighborhood
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    domain = capped_cylinder().domain
    original_id = domain.domain_id
    corners = np.asarray(domain.corner_points).copy()
    limits = M.MeshingLimits(maximum_geometry_queries=1)
    with pytest.raises(M.MeshingFailure) as refused:
        with native_volume_execution_budget(
            limits, stage=M.MeshingStageKind.SOURCE_INSPECTION
        ):
            SourcePatchNeighborhood(domain, 1)
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert domain.domain_id == original_id
    np.testing.assert_array_equal(domain.corner_points, corners)


def _c0_tensor_spline_trim_domain() -> MeshingDomain:
    controls = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            [[2.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
        ],
    )
    surface = BSplineSurfacePatch(
        controls,
        np.ones((3, 2)),
        [0.0, 0.0, 1.0, 2.0, 2.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    corners = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    loop = tuple(
        PatchCurveUse(
            row,
            BSplineCurve.bezier(
                (head, 0.5 * (head + corners[(row + 1) % 4]), corners[(row + 1) % 4]),
                [1.0, 1.0, 1.0],
            ),
            0.0,
            1.0,
        )
        for row, head in enumerate(corners)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (loop,)),),
        tuple(MeshingDomainCurve(row, (row + 1) % 4) for row in range(4)),
        4,
        source_id="authored-c0-tensor-trim",
        source_revision="original-knot-bank",
    )


def test_c0_tensor_source_interpolation_encloses_actual_chord_without_claiming_hessian() -> (
    None
):
    domain = _c0_tensor_spline_trim_domain()
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
    from phydrax.geometry._meshing_domain import _verify_chart_chain

    domain = _c0_tensor_spline_trim_domain()
    charts = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    points = domain.evaluate(np.zeros(4, dtype=np.int64), charts)
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    boundary = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)
    provenance = np.asarray([(0, row, 0.0, 1.0) for row in range(4)])
    findings, resources = [], []
    required = np.zeros((charts.shape[0],), dtype=np.bool_)
    valid, _, ribbon, _ = _verify_chart_chain(
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
    )
    assert valid and not findings
    assert ribbon.shape == (4,) and np.all(np.isfinite(ribbon))
    assert np.max(ribbon) >= 0.5


@pytest.mark.parametrize(
    "order", (1, 2), ids=("closed-first-jets", "truthful-global-second-jets")
)
def test_tensor_source_bound_batch_preserves_individual_closed_knot_queries(
    order: int,
) -> None:
    surface = _c0_tensor_spline_trim_domain().patches[0].surface
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
    if order == 1:
        assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
        assert lower[1, 1, 0] <= 0.0 and upper[1, 1, 0] >= 1.0
    else:
        assert np.all(np.isneginf(lower[:4])) and np.all(np.isposinf(upper[:4]))
        assert np.all(np.isfinite(lower[4])) and np.all(np.isfinite(upper[4]))
