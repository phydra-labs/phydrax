"""Consumer identity, topology, and evidence contracts of meshing fixtures."""

from collections import Counter

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.discretization import (
    AdaptiveSimplexPolicy,
    CellGeometrySpec,
    ExactPlcCellGeometrySource,
    TensorGridPlan,
    UniformAxisSpec,
)
from phydrax.geometry import (
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PlanePatch,
)
from phydrax.geometry._embedding_state import (
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from phydrax.geometry._global_embedding import EmbeddingCounts, GlobalEmbeddingCertificate
from phydrax.geometry._mesh_certificates import MeshCertificateBinding
from phydrax.meshing import (
    certify_cell_mesh,
    MeshAdaptationRoute,
    MeshingEntityKind,
    MeshingLimits,
    NativePlcSource,
    RegionRole,
    RegionSeed,
    SizeControlStrength,
    UniformSizeControl,
    VolumeFillStrategy,
)
from phydrax.meshing._mesh_part import MeshPart
from tests._support.cell_meshes import unit_square
from tests._support.meshing_fixtures import (
    adaptation_policy,
    box_corners,
    box_plc,
    certified_carrier,
    domain_scope,
    finding_checks,
    mesh_scope,
    surface_mesh,
    surface_specification,
    volume_mesh,
    volume_specification,
)


# Box coordinates are dyadic and only a handful of 3x3 determinant and
# summation operations contribute to the volume: 128 eps is a conservative bound.
_BOX_VOLUME_ROUNDOFF = 128 * np.finfo(np.float64).eps


def _plane_domain() -> MeshingDomain:
    uv = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    uses = tuple(
        PatchCurveUse(i, LineCurve(uv[i], uv[(i + 1) % 4] - uv[i]), 0.0, 1.0)
        for i in range(4)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), (uses,)),),
        tuple(MeshingDomainCurve(i, (i + 1) % 4) for i in range(4)),
        4,
        source_id="fixture-plane",
        source_revision="fixture-r7",
    )


def test_mesh_scope_retains_mesh_revision_entity_set_and_sparse_ids() -> None:
    mesh = unit_square(cell_global_ids=np.asarray((10, 30), dtype=np.int64))
    scope = mesh_scope(mesh, 2, (30,))
    assert scope.source_id == mesh.mesh_id
    assert scope.source_revision == mesh.numeric_version
    assert scope.entity_kind is MeshingEntityKind.MESH
    assert scope.entity_dimension == 2
    assert scope.entity_set_id == mesh.entity_set(2).entity_set_id
    np.testing.assert_array_equal(scope.entity_ids, (30,))
    np.testing.assert_array_equal(mesh_scope(mesh, 2).entity_ids, (10, 30))


def test_domain_scope_retains_geometry_source_revision_and_entity_indices() -> None:
    domain = _plane_domain()
    scope = domain_scope(domain, 1, (1, 3))
    assert scope.source_id == domain.source_id
    assert scope.source_revision == domain.source_revision
    assert scope.entity_kind is MeshingEntityKind.GEOMETRY
    assert scope.entity_dimension == 1
    assert scope.entity_set_id == domain.entity_set_id(1)
    np.testing.assert_array_equal(scope.entity_ids, (1, 3))


def test_surface_specification_binds_all_patches_size_strength_and_limits() -> None:
    domain = _plane_domain()
    limits = MeshingLimits(maximum_vertices=123)
    spec = surface_specification(
        domain, 0.25, limits=limits, strength=SizeControlStrength.HARD
    )
    assert spec.target.topological_dimension == 2 and spec.target.ambient_dimension == 3
    assert spec.target.cell_families.required == ("triangle",)
    assert (
        spec.scope.source_id == domain.source_id
        and spec.scope.source_revision == domain.source_revision
    )
    np.testing.assert_array_equal(spec.scope.entity_ids, (0,))
    assert len(spec.size_controls) == 1
    control = spec.size_controls[0]
    assert isinstance(control, UniformSizeControl)
    assert control.scope.scope_id == spec.scope.scope_id
    assert control.target_size == 0.25
    assert control.strength is SizeControlStrength.HARD
    assert spec.limits.limits_id == limits.limits_id


def test_volume_specification_binds_facet_scope_seed_and_limits() -> None:
    source = NativePlcSource(box_plc(), "fixture-box", "fixture-r9")
    limits = MeshingLimits(maximum_vertices=123)
    seed = RegionSeed(
        np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
        "solid",
        "material",
        RegionRole.SOLID,
    )
    spec = volume_specification(source, size=0.5, limits=limits, region_seeds=(seed,))
    assert spec.target.topological_dimension == 3 and spec.target.ambient_dimension == 3
    assert spec.target.cell_families.required == ("tetrahedron",)
    assert spec.fill_strategy is VolumeFillStrategy.SIMPLEX
    assert spec.boundary_scope.source_id == source.source_id
    assert spec.boundary_scope.source_revision == source.source_revision
    assert spec.boundary_scope.entity_set_id == "fixture-box-facets"
    np.testing.assert_array_equal(spec.boundary_scope.entity_ids, np.arange(6))
    control = spec.size_controls[0]
    assert isinstance(control, UniformSizeControl)
    assert control.scope.scope_id == spec.boundary_scope.scope_id
    assert control.target_size == 0.5
    assert control.strength is SizeControlStrength.SOFT
    assert spec.region_seeds[0].seed_id == seed.seed_id
    assert spec.limits.limits_id == limits.limits_id


def test_surface_mesh_retains_request_physical_frame_and_certification() -> None:
    domain = _plane_domain()
    spec = surface_specification(domain, 1.0)
    contract = SpatialCoordinateContract(
        SpatialCoordinateContract.si().length_unit, reference_frame="fixture-frame"
    )
    result = surface_mesh(domain, spec, coordinate_contract=contract)
    assert result.mesh.topological_dimension == 2 and result.mesh.ambient_dimension == 3
    assert all(block.cell_kind == "triangle" for block in result.mesh.blocks)
    assert result.coordinate_contract.spatial_id == contract.spatial_id
    assert result.audit.passed and result.compliance.passed
    assert result.compliance.specification_id == spec.specification_id
    assert result.certification is not None and result.certification.passed
    assert result.certification.mesh_id == result.mesh.mesh_id
    np.testing.assert_array_equal(np.asarray(result.mesh.coordinates)[:, 2], 0.0)


def test_volume_mesh_retains_request_physical_frame_and_unit_box_coverage() -> None:
    source = NativePlcSource(box_plc(), "fixture-box", "fixture-r9")
    spec = volume_specification(source, size=4.0)
    contract = SpatialCoordinateContract(
        SpatialCoordinateContract.si().length_unit, reference_frame="fixture-frame"
    )
    result = volume_mesh(source, spec, coordinate_contract=contract)
    assert all(block.cell_kind == "tetrahedron" for block in result.mesh.blocks)
    assert result.coordinate_contract.spatial_id == contract.spatial_id
    assert result.audit.passed and result.compliance.passed
    assert result.compliance.specification_id == spec.specification_id
    assert result.certification is not None and result.certification.passed
    exact_source = result.geometry.exact_source
    assert isinstance(exact_source, ExactPlcCellGeometrySource)
    # A PLC's declared physical identity belongs to its retained source, while
    # certification request source IDs describe optional fidelity queries.
    assert exact_source.domain_source_id == source.source_id
    assert exact_source.domain_source_revision == source.source_revision
    points = np.asarray(result.mesh.coordinates)
    cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
    corners = points[cells]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6
    assert np.all(volumes > 0)
    assert abs(volumes.sum() - 1.0) <= _BOX_VOLUME_ROUNDOFF


def test_certified_carrier_retains_the_cell_producer_and_refuses_tensor_carriers() -> (
    None
):
    result = certify_cell_mesh(unit_square(), SpatialCoordinateContract.si())
    assert certified_carrier(MeshPart("cells", result)) is result
    grid = TensorGridPlan(
        (UniformAxisSpec(2), UniformAxisSpec(2)), axis_names=("x", "y")
    ).prepare(np.asarray(((0, 0), (1, 1)), dtype=np.float64))
    part = MeshPart("grid", grid, coordinate_contract=SpatialCoordinateContract.si())
    with pytest.raises(TypeError, match="meshing_fixtures.part.carrier"):
        certified_carrier(part)


def test_box_corners_retains_binary_corner_order_and_nonunit_bounds() -> None:
    expected = np.asarray(
        (
            (-1, 2, 4),
            (3, 2, 4),
            (-1, 5, 4),
            (3, 5, 4),
            (-1, 2, 6),
            (3, 2, 6),
            (-1, 5, 6),
            (3, 5, 6),
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(box_corners((-1, 2, 4), (3, 5, 6)), expected)
    with pytest.raises(ValueError, match="meshing_fixtures.box"):
        box_corners((0, 0, 0), (1, 0, 1))


def test_box_plc_has_closed_outward_chain_and_named_interior() -> None:
    plc = box_plc(region_id="fluid", boundary="fixed")
    assert plc.region_ids == ("fluid",) and plc.boundary == "fixed"
    np.testing.assert_array_equal(plc.facet_regions, np.tile((-1, 0), (6, 1)))
    np.testing.assert_array_equal(plc.polygon_facets, np.arange(6))
    chain: Counter[tuple[int, int]] = Counter()
    for start, stop in zip(
        plc.polygon_offsets[:-1], plc.polygon_offsets[1:], strict=True
    ):
        loop = plc.polygon_vertices[start:stop]
        corners = plc.vertices[loop]
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        assert normal @ (corners.mean(axis=0) - 0.5) > 0
        for left, right in zip(loop, np.roll(loop, -1), strict=True):
            chain[(int(left), int(right))] += 1
    assert len(chain) == 24
    assert all(count == chain[(right, left)] for (left, right), count in chain.items())


@pytest.mark.parametrize(
    "route",
    (
        MeshAdaptationRoute.NATIVE_BISECTION,
        MeshAdaptationRoute.DEVICE_BISECTION,
        MeshAdaptationRoute.DEVICE_METRIC_2D,
    ),
)
def test_adaptation_policy_retains_route_and_device_capacity_requirement(
    route: MeshAdaptationRoute,
) -> None:
    scope = mesh_scope(unit_square(), 1)
    limits = MeshingLimits(maximum_vertices=123)
    policy = adaptation_policy(route, limits=limits, protected_scopes=(scope,))
    assert policy.route is route
    assert policy.limits.limits_id == limits.limits_id
    assert policy.protected_scopes[0].scope_id == scope.scope_id
    if route is MeshAdaptationRoute.NATIVE_BISECTION:
        assert policy.device_policy is None
    else:
        assert policy.device_policy is not None
        capacity = AdaptiveSimplexPolicy()
        assert adaptation_policy(route, device_policy=capacity).device_policy is capacity


def test_finding_checks_filters_each_finding_not_the_overall_status() -> None:
    mesh = unit_square()
    geometry = CellGeometrySpec.affine(mesh)
    binding = MeshCertificateBinding(mesh, geometry, "affine", MeshCertificateLimits())
    findings = (
        MeshCertificateFinding("cell_contact", "violated", "cell", (0,)),
        MeshCertificateFinding("cell_contact_capacity", "unresolved", "cell", (1,)),
        MeshCertificateFinding("cell_contact", "violated", "cell", (1,)),
    )
    certificate = GlobalEmbeddingCertificate(
        binding,
        "fixture-validity",
        findings,
        ("cell_contact", "cell_contact_capacity"),
        counts=EmbeddingCounts(
            cell_count=2,
            boundary_facet_count=4,
            shell_count=1,
            candidate_pair_count=1,
            ray_test_count=0,
            subdivision_piece_count=0,
        ),
    )
    assert certificate.status == "violated"
    assert finding_checks(certificate, "violated") == {"cell_contact"}
    assert finding_checks(certificate, "unresolved") == {"cell_contact_capacity"}
