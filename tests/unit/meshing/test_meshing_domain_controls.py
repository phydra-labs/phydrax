#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import (
    BSplineSurfacePatch,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
)
from phydrax.meshing._domain import (
    compile_surface_domain,
    SourcePatchNeighborhood,
    surface_domain_issues,
)
from phydrax.meshing._volume_generation import native_volume_execution_budget
from phydrax.meshing.qualification import capped_cylinder_surface, sphere_surface
from tests._support.meshing_domain import (
    meshing_scope,
    planar_protection_scenario,
    PRINCIPAL_CURVATURE_RTOL,
    square_patch,
)


M = phx.meshing


def test_compiled_constraints_share_curves_and_corners() -> None:
    can = capped_cylinder_surface().domain
    surfaces = meshing_scope(can, 2, (0, 1, 2))
    seam = meshing_scope(can, 1, (2,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        surfaces,
        size_controls=(
            M.UniformSizeControl(surfaces, 0.4, strength=M.SizeControlStrength.SOFT),
            M.UniformSizeControl(seam, 0.1, strength=M.SizeControlStrength.SOFT),
        ),
        protected_features=(
            M.ProtectedFeature(
                meshing_scope(can, 2, (1,)), M.FeatureKind.SURFACE, maximum_deviation=0.01
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
    domain = sphere_surface().domain
    surfaces = meshing_scope(domain, 2, (0,))
    stale = meshing_scope(domain, 2, (0,), revision="r0")

    with pytest.raises(ValueError, match="MeshingSpec.controls"):
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
    domain = capped_cylinder_surface().domain
    selected = meshing_scope(domain, 2, (1,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(M.UniformSizeControl(selected, 0.4),),
        protected_features=(
            M.ProtectedFeature(
                meshing_scope(domain, dimension, entities), kind, maximum_deviation=0.01
            ),
        ),
    )

    assert surface_domain_issues(domain, specification)
    with pytest.raises(M.MeshingFailure) as refused:
        compile_surface_domain(domain, specification)
    assert refused.value.category is M.MeshingFailureCategory.SCOPE_RESOLUTION_FAILED
    assert refused.value.provider_code == "protected_source_scope"


def test_selected_closed_curve_keeps_both_authored_endpoint_incidents() -> None:
    domain = capped_cylinder_surface().domain
    selected = meshing_scope(domain, 2, (1,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(M.UniformSizeControl(selected, 0.4),),
        protected_features=(
            M.ProtectedFeature(meshing_scope(domain, 0, (0,)), M.FeatureKind.CORNER),
            M.ProtectedFeature(
                meshing_scope(domain, 1, (0,)),
                M.FeatureKind.CURVE,
                maximum_deviation=0.01,
            ),
        ),
    )
    compiled = compile_surface_domain(domain, specification)
    np.testing.assert_array_equal(compiled.curves, (0,))
    np.testing.assert_array_equal(compiled.corners, (0,))
    np.testing.assert_array_equal(compiled.curve_deviations, (0.01,))
    assert compiled.curve_request.junctions[0].endpoints == ((0, "end"), (0, "start"))


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
        (MeshingSurfacePatch(patch, square_patch((0, 1, 2, 3)).loops),),
        tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0))),
        4,
        source_id="parabolic-sheet",
        source_revision="r1",
    )
    scope = meshing_scope(domain, 2, (0,))
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
    limits = specification.limits
    ticket = HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    )
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    with ticket, budget.activate():
        sizes = compiled.source_sizing.evaluate(domain, 0, charts, points, budget=budget)
    # Independent graph formula for z=u²: kappa=2/(1+4u²)^(3/2).
    expected = np.sin(angle / 2) * (1 + 4 * charts[:, 0] ** 2) ** 1.5
    np.testing.assert_allclose(sizes, expected, rtol=PRINCIPAL_CURVATURE_RTOL)


def test_source_sizing_rejects_contradictory_hard_targets_before_growth() -> None:
    domain = sphere_surface().domain
    scope = meshing_scope(domain, 2, (0,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            M.UniformSizeControl(scope, 0.3),
            M.UniformSizeControl(scope, 0.4),
        ),
    )
    with pytest.raises(ValueError, match="resolve_size_controls.controls"):
        compile_surface_domain(domain, specification)


def test_source_sizing_refuses_scratch_before_materializing_query_plan() -> None:
    domain = sphere_surface().domain
    scope = meshing_scope(domain, 2, (0,))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(M.UniformSizeControl(scope, 0.3),),
        limits=M.MeshingLimits(maximum_scratch_bytes=1024),
    )
    with pytest.raises(M.MeshingFailure) as refused:
        compile_surface_domain(domain, specification)
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_unknown_size_control_entity_refuses_before_source_queries(
    dimension: int,
) -> None:

    domain = capped_cylinder_surface().domain
    selected = meshing_scope(domain, 2, (0, 1, 2))
    unknown = meshing_scope(domain, dimension, (99,))
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

    domain = capped_cylinder_surface().domain
    original_id = domain.domain_id
    corners = np.asarray(domain.corner_points).copy()
    limits = M.MeshingLimits(maximum_geometry_queries=1)
    with pytest.raises(M.MeshingFailure) as refused:
        with native_volume_execution_budget(
            limits, stage=M.MeshingStageKind.SOURCE_INSPECTION
        ) as scope:
            ticket = scope.batch(
                limits.maximum_work_units,
                queries_upper=limits.maximum_geometry_queries,
                bytes_upper=limits.maximum_scratch_bytes,
            )
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket, budget.activate():
                SourcePatchNeighborhood(domain, 1, budget=budget)
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert domain.domain_id == original_id
    np.testing.assert_array_equal(domain.corner_points, corners)


@pytest.mark.parametrize(
    "kind", (M.FeatureKind.CORNER, M.FeatureKind.CURVE), ids=("corner", "curve")
)
def test_planar_source_protection_rejects_stale_binding(kind: M.FeatureKind) -> None:
    with pytest.raises(ValueError, match="MeshingSpec.controls"):
        planar_protection_scenario(kind, "r0")


def test_planar_source_curve_protection_rejects_corner_dimension() -> None:
    scenario = planar_protection_scenario(M.FeatureKind.CURVE)
    provider = M.NativeMeshingProvider(
        M.NativeMeshingOptions("planar_constrained_delaunay")
    )
    with pytest.raises(M.MeshingFailure):
        provider.plan(
            scenario.source,
            scenario.specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )


def test_planar_source_corner_protection_retains_authored_corner() -> None:
    scenario = planar_protection_scenario(M.FeatureKind.CORNER)
    provider = M.NativeMeshingProvider(
        M.NativeMeshingOptions("planar_constrained_delaunay")
    )
    result = provider.plan(
        scenario.source,
        scenario.specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()
    assert (
        dict(result.compliance.achieved)[
            f"protected:{scenario.feature.feature_id}:maximum_deviation"
        ]
        == 0.0
    )
    assert result.compliance.passed
    points = np.asarray(result.mesh.coordinates)
    assert np.any(np.all(points == (0.0, 0.0), axis=1))


@pytest.mark.parametrize("dimension", (2, 3), ids=("surface-region", "volume-region"))
def test_disabled_region_refuses_before_source_sizing(dimension: int) -> None:
    domain = capped_cylinder_surface().domain
    selected = meshing_scope(domain, 2, (0, 1, 2))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(M.UniformSizeControl(selected, 0.4),),
        region_controls=(
            M.RegionControl(
                meshing_scope(domain, dimension, (0,)),
                "disabled",
                "material",
                M.RegionRole.FLUID,
                meshing_enabled=False,
            ),
        ),
        # Sizing cannot be prepared with this limit; semantic refusal must precede it.
        limits=M.MeshingLimits(maximum_scratch_bytes=1),
    )
    with pytest.raises(ValueError, match="disabled for meshing"):
        compile_surface_domain(domain, specification)


def test_curve_patch_refuses_declared_names_with_wrong_source_incidence() -> None:
    domain = capped_cylinder_surface().domain
    selected = meshing_scope(domain, 2, (0, 1, 2))
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        selected,
        size_controls=(M.UniformSizeControl(selected, 0.4),),
        region_controls=tuple(
            M.RegionControl(
                meshing_scope(domain, 2, (index,)),
                name,
                f"material-{name}",
                M.RegionRole.FLUID,
            )
            for index, name in enumerate(("side", "bottom", "top"))
        ),
        patch_controls=(
            M.PatchControl(
                "bottom-edge", meshing_scope(domain, 1, (0,)), ("side", "top")
            ),
        ),
        limits=M.MeshingLimits(maximum_scratch_bytes=1),
    )
    with pytest.raises(ValueError, match="adjacent_region_names"):
        compile_surface_domain(domain, specification)
