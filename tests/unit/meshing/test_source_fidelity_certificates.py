#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible source fidelity certificates contracts."""

import math

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax.geometry._mesh_certificates import MeshCertificateBinding
from phydrax.geometry._source_fidelity import (
    ImplicitProjectionBoundarySource,
    ImplicitProjectionCoverageEvidence,
    MappedDomainBoundarySource,
)
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.units import METER
from tests._support.mesh_certificate_fixtures import (
    independent_mapped_shear_domain,
    octahedron,
    parabolic_coordinate_mesh,
    ParabolicChartSource,
    ParabolicTriangleSource,
    SampledSphere,
)
from tests._support.meshing_fixtures import finding_checks
from tests._support.recipes import restore_with_identity


@pytest.mark.parametrize(
    ("tolerance", "status"),
    [
        pytest.param(0.8, "certified", id="loose"),
        pytest.param(0.1, "violated", id="tight"),
    ],
)
def test_exact_sdf_source_fidelity_is_two_sided_and_certified(
    tolerance: float, status: str
) -> None:
    surface = octahedron()
    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    source = phx.geometry.ImplicitBoundarySource(
        sphere, source_id="unit-sphere", spacing=0.2
    )
    certificate = phx.geometry.certify_source_fidelity(
        surface,
        phx.discretization.CellGeometrySpec.affine(surface),
        source,
        tolerance=tolerance,
    )
    # Independent reference: the octahedron face centers lie 1 - 1/sqrt(3) from
    # the sphere, the largest deviation in both directions.
    hausdorff = 1.0 - 1.0 / math.sqrt(3.0)

    assert certificate.status == status
    assert certificate.mesh_to_source_semantics == "certified"
    assert certificate.source_to_mesh_semantics == "certified"
    assert (
        certificate.source_to_mesh_lower <= hausdorff <= certificate.source_to_mesh_upper
    )
    assert (
        certificate.mesh_to_source_lower <= hausdorff <= certificate.mesh_to_source_upper
    )
    assert certificate.binding.source_revision == source.source_revision


def test_sampled_source_bounds_never_certify_fidelity() -> None:
    surface = octahedron()
    certificate = phx.geometry.certify_source_fidelity(
        surface,
        phx.discretization.CellGeometrySpec.affine(surface),
        SampledSphere(),
        tolerance=10.0,
    )

    assert certificate.mesh_to_source_semantics == "sampled"
    assert certificate.status == "unresolved"


@pytest.mark.parametrize("degree", [2, 3], ids=["quadratic", "canonical-source-cubic"])
def test_curved_source_fidelity_uses_continuous_coordinate_and_source_covers(
    degree: int,
) -> None:
    source = ParabolicTriangleSource()
    mesh, geometry = parabolic_coordinate_mesh(source, degree=degree)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.25,
    )

    assert certificate.status == "certified"
    assert certificate.binding.coordinate_scope == "mapped"
    assert certificate.mesh_to_source_semantics == "certified"
    assert certificate.source_to_mesh_semantics == "certified"
    assert certificate.mesh_to_source_upper <= 0.25
    assert certificate.source_to_mesh_upper <= 0.25
    assert certificate.binding.source_revision == source.source_revision


def test_curved_chart_taylor_evidence_certifies_requested_hard_accuracy() -> None:
    source = ParabolicChartSource(curvature=1e-4)
    mesh, geometry = parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=2e-4,
        limits=phx.geometry.MeshCertificateLimits(
            maximum_source_samples=1,
            maximum_subdivision_depth=0,
        ),
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper <= 2e-4
    assert certificate.source_to_mesh_upper <= 2e-4
    assert certificate.binding.source_revision == source.source_revision


@pytest.mark.parametrize(
    ("bias", "bow"),
    [pytest.param(0.3, 0.0, id="biased"), pytest.param(0.0, 1.2, id="bowed-interior")],
)
def test_curved_offsource_coordinates_have_certified_distance_counterexamples(
    bias: float, bow: float
) -> None:
    source = ParabolicChartSource(curvature=1e-4)
    mesh, geometry = parabolic_coordinate_mesh(source, bias=bias, bow=bow)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.05,
    )

    assert certificate.status == "violated"
    assert "boundary_deviation" in finding_checks(certificate, "violated")
    assert certificate.mesh_to_source_lower > certificate.tolerance


def test_curved_uncertified_source_never_promotes_coordinate_bounds() -> None:
    source = ParabolicTriangleSource(certified=False)
    mesh, geometry = parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.25,
    )

    assert certificate.status == "unresolved"
    assert certificate.mesh_to_source_semantics == "sampled"
    assert certificate.source_to_mesh_semantics == "sampled"


def test_curved_source_cover_budget_refusal_reports_unbounded_direction() -> None:
    source = ParabolicTriangleSource()
    mesh, geometry = parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.01,
        limits=phx.geometry.MeshCertificateLimits(maximum_source_samples=1),
    )

    assert certificate.status == "unresolved"
    assert "mesh_sample_capacity" in finding_checks(certificate, "unresolved")
    assert "source_sample_capacity" in finding_checks(certificate, "unresolved")
    assert math.isinf(certificate.mesh_to_source_upper)
    assert math.isinf(certificate.source_to_mesh_upper)


@pytest.mark.parametrize(
    ("budget", "finding"),
    [
        pytest.param({"maximum_subdivision_depth": 0}, "subdivision_depth", id="depth"),
        pytest.param(
            {"maximum_subdivision_pieces": 1}, "subdivision_capacity", id="pieces"
        ),
        pytest.param(
            {"maximum_bernstein_nodes": 1}, "bernstein_capacity", id="bernstein"
        ),
        pytest.param(
            {"maximum_distance_evaluations": 1}, "distance_capacity", id="queries"
        ),
    ],
)
def test_curved_enclosure_resource_limits_remain_explicitly_unresolved(
    budget: dict[str, int],
    finding: str,
) -> None:
    source = ParabolicTriangleSource()
    mesh, geometry = parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.01,
        limits=phx.geometry.MeshCertificateLimits(**budget),
    )

    assert certificate.status == "unresolved"
    assert finding in finding_checks(certificate, "unresolved")
    assert certificate.mesh_to_source_upper > certificate.tolerance


def test_curved_source_fidelity_refines_an_affine_face_cover_without_relaxing_tolerance() -> (
    None
):
    source = ParabolicTriangleSource()
    mesh, _ = parabolic_coordinate_mesh(source)
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        source,
        tolerance=0.25,
        sample_order=4,
    )

    assert certificate.status == "certified"
    assert certificate.binding.coordinate_scope == "affine"
    assert certificate.mesh_to_source_upper <= certificate.tolerance
    assert certificate.source_to_mesh_upper <= certificate.tolerance


def test_curved_analytic_source_has_independent_degree_one_projection_coverage() -> None:

    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
        source_id="independent-sphere",
    )
    mesh = octahedron().with_coordinates(
        radius * np.asarray(octahedron().coordinates),
        numeric_version="source-scale",
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.49 * radius,
    )

    assert certificate.status == "certified"
    assert certificate.source_to_mesh_upper <= certificate.tolerance
    assert certificate.projection_coverage is not None
    assert certificate.projection_coverage.status == "certified"
    assert certificate.projection_coverage.source_state_id == profile.state_id
    assert certificate.projection_coverage.source_reach_lower == profile.reach_lower
    assert certificate.projection_coverage.fiber_crossing_classes.count("interior") == 1


@pytest.fixture(scope="module")
def projection_replay_context() -> tuple[
    AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
]:
    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
        source_id="independent-sphere",
        source_revision="retained-sphere-revision",
    )
    surface = octahedron()
    mesh = surface.with_coordinates(
        radius * np.asarray(surface.coordinates), numeric_version="source-scale"
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.49 * radius,
    )
    assert certificate.status == "certified"
    assert certificate.projection_coverage is not None
    register_meshing_source_artifacts()
    return profile, certificate.projection_coverage


def test_projection_coverage_stored_replay_preserves_raw_evidence_and_profile_binding(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
) -> None:
    profile, evidence = projection_replay_context
    rebuilt_profile = reconstruct_stored_fields(profile)
    assert rebuilt_profile is not profile
    assert tree_content_equal(rebuilt_profile, profile)
    rebuilt = reconstruct_stored_fields(evidence, bindings={"profile": rebuilt_profile})
    assert rebuilt is not evidence
    assert tree_content_equal(rebuilt, evidence)
    assert rebuilt.evidence_id == evidence.evidence_id
    assert rebuilt.profile_id == profile.profile_id
    assert (rebuilt.binding.source_id, rebuilt.binding.source_revision) == (
        profile.source_id,
        profile.source_revision,
    )


def test_projection_coverage_stored_replay_requires_retained_profile(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
) -> None:
    _, evidence = projection_replay_context
    with pytest.raises(TypeError, match=r"ImplicitProjectionCoverageEvidence\.profile"):
        reconstruct_stored_fields(evidence)


def test_projection_coverage_stored_replay_refuses_same_namespace_mismatched_profile(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
) -> None:
    profile, evidence = projection_replay_context
    coordinates = profile.coordinate_contract
    changed_profile = AnalyticImplicitProfile(
        profile.geometry,
        phx.SpatialCoordinateContract(coordinates.length_unit, reference_frame="scanner"),
        tube_radius=profile.tube_radius,
        cover_radius=profile.cover_radius,
        source_id=profile.source_id,
        source_revision=profile.source_revision,
    )
    assert changed_profile.source_id == profile.source_id
    assert changed_profile.source_revision == profile.source_revision
    assert tree_content_equal(changed_profile.center, profile.center)
    assert changed_profile.radius == profile.radius
    assert changed_profile.family == profile.family
    assert changed_profile.profile_id != profile.profile_id
    with pytest.raises(
        ValueError, match=r"ImplicitProjectionCoverageEvidence\.profile_id"
    ):
        reconstruct_stored_fields(evidence, bindings={"profile": changed_profile})


@pytest.mark.parametrize("field", ("source_id", "source_revision"))
def test_projection_coverage_stored_replay_refuses_forged_source_binding(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    field: str,
) -> None:
    profile, evidence = projection_replay_context
    forged_binding = restore_with_identity(evidence.binding, field, "forged-source")
    assert isinstance(forged_binding, MeshCertificateBinding)
    forged = eqx.tree_at(lambda value: value.binding, evidence, replace=forged_binding)
    with pytest.raises(ValueError, match=r"ImplicitProjectionCoverageEvidence\.binding"):
        reconstruct_stored_fields(forged, bindings={"profile": profile})


@pytest.mark.parametrize(
    ("field", "replacement", "error_field"),
    (
        pytest.param("profile_id", "forged-profile", "profile_id", id="profile_id"),
        pytest.param("evidence_id", "forged-evidence", "stored_fields", id="evidence_id"),
        pytest.param("status", "unresolved", "stored_fields", id="status"),
    ),
)
def test_projection_coverage_stored_replay_refuses_forged_identity_or_status(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    field: str,
    replacement: str,
    error_field: str,
) -> None:
    profile, evidence = projection_replay_context
    forged = restore_with_identity(evidence, field, replacement)
    assert isinstance(forged, ImplicitProjectionCoverageEvidence)
    with pytest.raises(
        ValueError, match=rf"ImplicitProjectionCoverageEvidence\.{error_field}"
    ):
        reconstruct_stored_fields(forged, bindings={"profile": profile})


@pytest.mark.parametrize("field", ("facets", "field_bounds", "fiber_value_bounds"))
def test_projection_coverage_stored_replay_refuses_changed_raw_evidence(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    field: str,
) -> None:
    profile, evidence = projection_replay_context
    changed = np.array(getattr(evidence, field), copy=True)
    changed.flat[0] = np.nextafter(changed.flat[0], np.float64(np.inf))
    forged = eqx.tree_at(lambda value: getattr(value, field), evidence, replace=changed)
    with pytest.raises(
        ValueError, match=r"ImplicitProjectionCoverageEvidence\.stored_fields"
    ):
        reconstruct_stored_fields(forged, bindings={"profile": profile})


def test_curved_source_projection_refuses_a_missing_closed_mesh_premise() -> None:

    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
    )
    closed = octahedron()
    mesh = phx.discretization.CellMesh.from_triangles(
        radius * np.asarray(closed.coordinates),
        np.asarray(closed.blocks[0].vertices, dtype=np.int64)[:-1],
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.49 * radius,
    )

    assert certificate.status == "unresolved"
    assert math.isinf(certificate.source_to_mesh_upper)
    assert certificate.projection_coverage is not None
    assert certificate.projection_coverage.status == "unresolved"


def test_curved_source_projection_reports_a_proved_distance_counterexample() -> None:

    radius = 0.371
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    profile = AnalyticImplicitProfile(
        implicit,
        phx.SpatialCoordinateContract.si(),
        tube_radius=0.49 * radius,
    )
    closed = octahedron()
    mesh = phx.discretization.CellMesh.from_triangles(
        3.0 * radius * np.asarray(closed.coordinates),
        np.asarray(closed.blocks[0].vertices, dtype=np.int64),
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile),
        tolerance=0.1 * radius,
    )

    assert certificate.status == "violated"
    assert certificate.projection_coverage is not None
    assert certificate.projection_coverage.status == "violated"
    assert any(
        finding.check == "projection_source_orientation_or_distance"
        and finding.status == "violated"
        for finding in certificate.findings
    )


def test_true_bilinear_mapped_source_boundary_equality_certifies_exact_zero() -> None:

    mesh, geometry, domain = independent_mapped_shear_domain()
    source = MappedDomainBoundarySource(domain, np.asarray((0, 0), dtype=np.int64))
    certificate = phx.geometry.certify_source_fidelity(
        mesh,
        geometry,
        source,
        tolerance=0.0,
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0
    assert certificate.domain_coverage is not None
    assert certificate.domain_coverage.status == "certified"
    assert certificate.binding.source_revision == domain.source_revision
