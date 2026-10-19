#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import numpy as np
import pytest

from phydrax.geometry import (
    BSplineCurve,
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
    PlanePatch,
    SpherePatch,
)
from phydrax.meshing.qualification import (
    capped_cylinder_surface,
    folded_plates_surface,
    sphere_surface,
    torus_surface,
)
from tests._support.meshing_domain import (
    SOURCE_ROUNDOFF_ATOL,
    square_patch,
    TORUS_PROJECTION_RTOL,
)


def test_strata_identities_and_incidence_follow_declarations() -> None:
    can = capped_cylinder_surface().domain
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
    domain = sphere_surface().domain
    charts = np.asarray([[0.3, 0.2], [2.0, -1.0], [1.0, 0.5 * np.pi]])
    normals, regular = domain.oriented_normals(np.zeros((3,), dtype=np.int64), charts)
    points = domain.evaluate(np.zeros((3,), dtype=np.int64), charts)

    np.testing.assert_array_equal(regular, (True, True, False))
    np.testing.assert_allclose(normals[:2], points[:2], atol=SOURCE_ROUNDOFF_ATOL)
    np.testing.assert_array_equal(normals[2], 0.0)


def test_projection_returns_closest_points_with_status() -> None:
    domain = torus_surface().domain
    queries = np.asarray([[2.9, 0.0, 0.1], [0.0, -2.3, -0.4]])
    seeds = np.asarray([[0.1, 0.2], [4.5, 5.5]])
    projection = domain.project(queries, np.zeros((2,), dtype=np.int64), seeds)
    ring = np.hypot(queries[:, 0], queries[:, 1]) - 2.0
    exact = np.abs(np.hypot(ring, queries[:, 2]) - 0.6)

    np.testing.assert_array_equal(projection.converged, (True, True))
    np.testing.assert_allclose(projection.distances, exact, rtol=TORUS_PROJECTION_RTOL)
    np.testing.assert_allclose(
        np.linalg.norm(projection.points - queries, axis=1),
        exact,
        rtol=TORUS_PROJECTION_RTOL,
    )


def test_shared_curve_that_patches_do_not_reproduce_is_refused() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    first = square_patch((0, 1, 2, 3))
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

    with pytest.raises(ValueError, match="MeshingDomain.curve_geometry"):
        MeshingDomain(
            (first, second), curves + extra, 6, source_id="gap", source_revision="r1"
        )


def test_inconsistently_oriented_neighbors_are_refused() -> None:
    plates = folded_plates_surface().domain
    flipped = MeshingSurfacePatch(
        plates.patches[1].surface, plates.patches[1].loops, reversed=True
    )

    with pytest.raises(ValueError, match="MeshingDomain.curve_orientation"):
        MeshingDomain(
            (plates.patches[0], flipped),
            plates.curves,
            6,
            source_id="plates",
            source_revision="r1",
        )


def test_pole_side_geometry_is_validated() -> None:
    patch = SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    half = 0.5 * np.pi
    not_a_pole = (
        PatchPoleUse(0, (0.0, -1.0), (2.0 * np.pi, -1.0)),
        PatchCurveUse(0, LineCurve((2.0 * np.pi, 0.0), (0.0, 1.0)), -1.0, half),
        PatchPoleUse(1, (2.0 * np.pi, half), (0.0, half)),
        PatchCurveUse(0, LineCurve((0.0, 0.0), (0.0, 1.0)), half, -1.0),
    )
    with pytest.raises(ValueError, match="MeshingDomain.pole_geometry"):
        MeshingDomain(
            (MeshingSurfacePatch(patch, (not_a_pole,)),),
            (MeshingDomainCurve(0, 1),),
            2,
            source_id="band",
            source_revision="r1",
        )


def test_region_surface_closure_is_validated() -> None:
    plates = folded_plates_surface().domain
    with pytest.raises(ValueError, match="MeshingDomain.region_boundary"):
        MeshingDomain(
            plates.patches,
            plates.curves,
            6,
            source_id="plates",
            source_revision="r1",
            regions=(MeshingDomainRegion("open", ((0, 1), (1, 1))),),
        )


def test_domain_identity_tracks_geometry_and_revision() -> None:
    base = sphere_surface().domain
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

    assert sphere_surface().domain.domain_id == base.domain_id
    assert moved.domain_id != base.domain_id
    assert sphere_surface(revision="r2").domain.domain_id != base.domain_id


def test_domain_identifiers_strip_whitespace_before_identity_binding() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    patch = square_patch((0, 1, 2, 3))
    stripped = MeshingDomain(
        (patch,), curves, 4, source_id="  square  ", source_revision="  r1  "
    )
    canonical = MeshingDomain(
        (patch,), curves, 4, source_id="square", source_revision="r1"
    )
    assert stripped.source_id == "square"
    assert stripped.source_revision == "r1"
    assert stripped.domain_id == canonical.domain_id
    assert MeshingDomainRegion("  material  ", ((0, 1),)).name == "material"


def test_content_identity_detects_domain_tree_surgery() -> None:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    domain = MeshingDomain(
        (square_patch((0, 1, 2, 3)),), curves, 4, source_id="square", source_revision="r1"
    )
    changed = eqx.tree_at(
        lambda value: value.patches[0].surface,
        domain,
        PlanePatch((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    )
    assert domain.content_domain_id() == domain.domain_id
    assert changed.domain_id == domain.domain_id
    assert changed.content_domain_id() != domain.domain_id


def test_curve_range_refusal_precedes_later_root_kind_admission() -> None:
    with pytest.raises(ValueError, match="PatchCurveUse.first"):
        PatchCurveUse(
            0,
            LineCurve((0.0, 0.0), (1.0, 0.0)),
            np.nan,
            1.0,
            last_root="invalid-root",  # ty: ignore[invalid-argument-type]
        )


def test_pole_geometry_refusal_precedes_trim_kind_admission() -> None:
    with pytest.raises(ValueError, match="PatchPoleUse.parameter_range"):
        PatchPoleUse(
            0,
            (0.0, 0.0),
            (0.0, 0.0),
            trim_curve="invalid-trim",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    "scope_state", ("inactive", "work-exhausted", "queries-exhausted")
)
def test_plc_preparation_cannot_replace_explicit_scope_with_ambient_capacity(
    scope_state: str,
) -> None:
    import phydrax as phx
    from phydrax._meshcore import meshcore_available, MeshcoreStatus, NativeExecutionScope
    from phydrax.discretization import reference_cell_topology
    from phydrax.meshing._volume_generation import prepare_plc_source

    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    topology = reference_cell_topology("tetrahedron")
    complex_ = phx.meshing.PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.arange(len(topology.entities[2]), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(topology.entities[2]), 1)),
        ("material",),
    )
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=7,
        maximum_geometry_queries=11,
        maximum_scratch_bytes=1 << 20,
        maximum_wall_seconds=120.0,
    )
    lowered = limits.native_limits()
    owner = NativeExecutionScope(lowered)
    inactive = NativeExecutionScope(lowered)
    identity = complex_.complex_id
    with owner:
        if scope_state == "inactive":
            explicit = inactive
            error = RuntimeError
            message = "NativeExecutionScope.remaining"
        else:
            explicit = owner
            work = limits.maximum_work_units if scope_state == "work-exhausted" else 0
            queries = (
                limits.maximum_geometry_queries
                if scope_state == "queries-exhausted"
                else 0
            )
            with owner.batch(work, queries_upper=queries) as ticket:
                ticket.consume(work, geometry_queries=queries)
            error = phx.meshing.MeshingFailure
            message = "source-renewal"
        before = owner.remaining()
        with pytest.raises(error, match=message) as refusal:
            prepare_plc_source(
                complex_,
                "explicit-scope-tetrahedron",
                "r1",
                phx.SpatialCoordinateContract.si(),
                limits=limits,
                scope=explicit,
            )
        if isinstance(refusal.value, phx.meshing.MeshingFailure):
            assert (
                refusal.value.category
                is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
            )
        after = owner.remaining()
        assert after.remaining_work_units == before.remaining_work_units
        assert after.remaining_geometry_queries == before.remaining_geometry_queries
        assert after.remaining_managed_bytes == before.remaining_managed_bytes
        assert complex_.complex_id == identity
        assert owner.limits is lowered
    assert inactive.evidence is None
    assert owner.completed_evidence().status is MeshcoreStatus.OK
