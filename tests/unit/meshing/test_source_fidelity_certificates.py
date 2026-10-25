#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible source fidelity certificates contracts."""

import math
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax.geometry._mesh_certificates import MeshCertificateBinding
from phydrax.geometry._source_fidelity import (
    DomainBoundaryFidelityEvidence,
    ImplicitProjectionBoundarySource,
    ImplicitProjectionCoverageEvidence,
    issue_domain_fidelity_source,
    MappedDomainBoundarySource,
    prepare_domain_boundary_fidelity,
    SourceFidelityPredecessor,
)
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
from phydrax.units import METER
from tests._support.coordinate_references import coordinate_reference_budget
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


_PROJECTION_HOST_BANKS = (
    "source_center", "facets", "facet_vertices", "facet_ids", "facet_components",
    "piece_triangles", "piece_barycentric_vertices", "piece_coordinate_bounds",
    "piece_owners", "field_bounds", "gradient_bounds", "normal_bounds",
    "fiber_value_bounds", "fiber_gradient_bounds", "fiber_candidate_facets",
)


def test_projection_coverage_owns_immutable_host_banks_without_producer_aliases(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
) -> None:
    import jax

    from phydrax._fingerprint import array_tree_fingerprint
    from phydrax._host_storage import immutable_host_array_layout

    profile, evidence = projection_replay_context
    producer = eqx.tree_at(
        lambda value: tuple(getattr(value, name) for name in _PROJECTION_HOST_BANKS),
        evidence,
        replace=tuple(np.array(getattr(evidence, name), copy=True) for name in _PROJECTION_HOST_BANKS),
    )
    before = array_tree_fingerprint(producer)
    restored = ImplicitProjectionCoverageEvidence.from_stored_fields(producer, profile=profile)
    assert restored is not evidence
    assert restored.evidence_id == evidence.evidence_id
    assert array_tree_fingerprint(restored) == before
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(producer)
    leaves = jax.tree_util.tree_leaves(restored)
    for name in _PROJECTION_HOST_BANKS:
        bank, original = getattr(restored, name), getattr(producer, name)
        assert type(bank) is np.ndarray
        assert bank.dtype == original.dtype and bank.shape == original.shape
        assert any(leaf is bank for leaf in leaves)
        assert immutable_host_array_layout(bank) is not None
        assert not np.shares_memory(bank, original)
        expected = bank.tobytes()
        if original.size:
            original.flat[0] += 1
        assert bank.tobytes() == expected
        for alias in (bank, bank.view(), bank.reshape(-1), np.asarray(bank)):
            with pytest.raises(ValueError):
                alias.setflags(write=True)
            if alias.size:
                with pytest.raises(ValueError):
                    alias.flat[0] = 0
        owner = bank
        while isinstance(owner, np.ndarray):
            with pytest.raises(ValueError):
                owner.setflags(write=True)
            owner = owner.base
        assert type(owner) is bytes
        assert memoryview(owner).readonly
    again = ImplicitProjectionCoverageEvidence.from_stored_fields(restored, profile=profile)
    assert again.evidence_id == restored.evidence_id
    for name in _PROJECTION_HOST_BANKS:
        # The center is owned by the retained profile, not by the input evidence.
        if name != "source_center":
            assert getattr(again, name) is getattr(restored, name)


def test_projection_coverage_archive_restores_numeric_banks_and_content_identity(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    tmp_path: Path,
) -> None:
    import json
    import subprocess
    import sys

    from phydrax._fingerprint import array_tree_fingerprint
    from phydrax._host_storage import immutable_host_array_layout
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    profile, evidence = projection_replay_context
    receipt = write_meshing_source_closure(
        tmp_path / "projection-banks.npz", (ImplicitProjectionBoundarySource(profile), evidence)
    )
    cold_source, cold_evidence = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert cold_evidence is not evidence
    assert cold_evidence.evidence_id == evidence.evidence_id
    assert array_tree_fingerprint(cold_evidence) == array_tree_fingerprint(evidence)
    renewed = ImplicitProjectionCoverageEvidence.from_stored_fields(
        cold_evidence, profile=cold_source.profile
    )
    for name in _PROJECTION_HOST_BANKS:
        bank = getattr(renewed, name)
        assert immutable_host_array_layout(bank) is not None
        np.testing.assert_array_equal(bank, getattr(evidence, name))
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import json, sys, numpy as np, jax; "
            "from phydrax.lifecycle import read_meshing_source_closure; "
            "from phydrax._host_storage import immutable_host_array_layout; "
            "from phydrax._fingerprint import array_tree_fingerprint; "
            "source, evidence = read_meshing_source_closure(sys.argv[1], expected_content_id=sys.argv[2]); "
            "banks = [leaf for leaf in jax.tree_util.tree_leaves(evidence) if type(leaf) is np.ndarray]; "
            "assert len(banks) == 15; "
            "assert all(immutable_host_array_layout(bank) is not None for bank in banks); "
            "print(json.dumps([evidence.evidence_id, array_tree_fingerprint(evidence)]))",
            str(receipt.path), receipt.content_id,
        ],
        check=True, capture_output=True, text=True, timeout=60,
    )
    assert json.loads(completed.stdout) == [
        evidence.evidence_id, array_tree_fingerprint(evidence)
    ]


@pytest.mark.parametrize("sealed", (False, True), ids=("mutable-producer", "immutable-proof"))
def test_projection_publication_scan_work_and_storage_exact_bound(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    sealed: bool,
) -> None:
    from phydrax._host_storage import immutable_host_array_layout
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.meshing._native_preparation import (
        _native_source_array_content,
        _native_source_array_scan_cost,
    )

    _, evidence = projection_replay_context
    banks = tuple(getattr(evidence, name) for name in _PROJECTION_HOST_BANKS)
    if not sealed:
        banks = tuple(np.array(bank, copy=True) for bank in banks)
        for bank in banks:
            bank.setflags(write=False)
    # An unrelated genuinely mutable bank remains scanned under both paths.
    mutable = (np.arange(17, dtype=np.int64),) + tuple(
        bank for bank in banks if immutable_host_array_layout(bank) is None
    )
    work, storage = _native_source_array_scan_cost(mutable)
    assert work == (18 if sealed else 18 + sum(1 + bank.size for bank in banks))

    def scan(work_limit: int, storage_limit: int) -> tuple[str, int, int]:
        ticket = HostBatchTicket(work_limit, bytes_upper=storage_limit)
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work_limit, maximum_memory_bytes=storage_limit
        )
        with ticket, budget.activate():
            digest = _native_source_array_content(mutable, budget)
            return digest, budget.work_units, budget.peak_bytes_upper

    first = scan(work, storage)
    assert first[1:] == (work, storage)
    assert scan(work, storage) == first  # A separate request earns the same scan.
    with pytest.raises(CoordinateEnclosureResourceError, match="coefficient_work"):
        scan(work - 1, storage)
    with pytest.raises(CoordinateEnclosureResourceError, match="polynomial_storage"):
        scan(work, storage - 1)
    mutable[0][-1] += 1
    assert scan(work, storage)[0] != first[0]


def test_projection_sphere_banks_remove_large_mutable_publication_scan(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    record_property: Callable[[str, object], None],
) -> None:
    from phydrax._host_storage import immutable_host_array_layout
    from phydrax.meshing._native_preparation import _native_source_array_scan_cost

    profile, _ = projection_replay_context
    octahedral = octahedron()
    points = np.asarray(octahedral.coordinates).tolist()
    faces = np.asarray(octahedral.blocks[0].vertices).tolist()
    # Independent spherical octahedron subdivision, not a design proposal or
    # adaptation route. Every physical facet receives its own real proof.
    for _ in range(4):
        midpoints: dict[tuple[int, int], int] = {}
        children = []
        for a, b, c in faces:
            mids = []
            for first, second in ((a, b), (b, c), (c, a)):
                edge = (min(first, second), max(first, second))
                if edge not in midpoints:
                    point = np.asarray(points[first]) + np.asarray(points[second])
                    point /= np.linalg.norm(point)
                    midpoints[edge] = len(points)
                    points.append(point.tolist())
                mids.append(midpoints[edge])
            ab, bc, ca = mids
            children.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        faces = children
    mesh = phx.discretization.CellMesh.from_triangles(
        profile.radius * np.asarray(points), np.asarray(faces, dtype=np.int32)
    )
    certificate = phx.geometry.certify_source_fidelity(
        mesh, phx.discretization.CellGeometrySpec.affine(mesh),
        ImplicitProjectionBoundarySource(profile), tolerance=profile.tube_radius,
    )
    assert certificate.status == "certified"
    evidence = certificate.projection_coverage
    assert evidence is not None
    banks = tuple(getattr(evidence, name) for name in _PROJECTION_HOST_BANKS)
    before_work, before_storage = _native_source_array_scan_cost(banks)
    remaining = tuple(bank for bank in banks if immutable_host_array_layout(bank) is None)
    after_work, after_storage = _native_source_array_scan_cost(remaining)
    assert before_work - after_work > 90_307
    assert (after_work, after_storage) == (0, 0)
    record_property("proof_bank_scan_work_before", before_work)
    record_property("proof_bank_scan_work_after", after_work)
    record_property("proof_bank_scan_storage_before", before_storage)
    record_property("proof_bank_scan_storage_after", after_storage)
    record_property("proof_facets", evidence.facets.shape[0])
    record_property("proof_pieces", evidence.piece_triangles.shape[0])


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


def test_exact_domain_boundary_fidelity_retains_issued_root_and_target_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _source_fidelity as owner

    original_issuance = owner.is_issued_domain_coverage
    authenticated_budgets: list[object] = []

    def authenticate(
        mesh: phx.discretization.CellMesh,
        geometry: phx.discretization.CellGeometrySpec,
        domain: phx.geometry.PiecewiseLinearDomain | phx.geometry.MappedReferenceDomain,
        regions: np.ndarray,
        embedding: phx.geometry.GlobalEmbeddingCertificate,
        limits: phx.geometry.MeshCertificateLimits,
        certificate: phx.geometry.DomainCoverageCertificate,
        *,
        budget: phx.discretization.CoordinateEnclosureBudget | None = None,
    ) -> bool:
        authenticated_budgets.append(budget)
        return original_issuance(
            mesh, geometry, domain, regions, embedding, limits, certificate, budget=budget
        )

    monkeypatch.setattr(owner, "is_issued_domain_coverage", authenticate)
    mesh, geometry, domain = independent_mapped_shear_domain()
    regions = np.zeros(sum(block.cell_count for block in mesh.blocks), dtype=np.int64)
    source = MappedDomainBoundarySource(domain, regions)
    limits = phx.geometry.MeshCertificateLimits()
    with coordinate_reference_budget() as budget:
        validity = phx.discretization.certify_cell_geometry_validity(
            geometry, mesh=mesh, budget=budget
        )
        embedding = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, limits=limits, budget=budget
        )
        coverage = phx.geometry.certify_domain_coverage(
            mesh, geometry, domain, regions, embedding=embedding,
            limits=limits, budget=budget,
        )
        fidelity = phx.geometry.certify_source_fidelity(
            mesh, geometry, source, tolerance=0.0, limits=limits, budget=budget,
        )
        root = SourceFidelityPredecessor(
            mesh, geometry, source, domain, regions, embedding, coverage, fidelity, limits
        )
        issue_domain_fidelity_source(root, budget=budget)
        target = mesh.with_coordinates(mesh.coordinates, numeric_version="boundary-equality-target")
        target_validity = phx.discretization.certify_cell_geometry_validity(
            geometry, mesh=target, budget=budget
        )
        target_embedding = phx.geometry.certify_global_embedding(
            target, geometry, target_validity, limits=limits, budget=budget
        )
        target_coverage = phx.geometry.certify_domain_coverage(
            target, geometry, domain, regions, embedding=target_embedding,
            limits=limits, budget=budget,
        )
        renewed = prepare_domain_boundary_fidelity(
            root, target, geometry, source, domain, regions, target_embedding,
            target_coverage, limits, tolerance=0.0, sample_order=4, budget=budget,
        )
        assert renewed is not None
        assert renewed.mesh_to_source_upper == fidelity.mesh_to_source_upper
        assert renewed.source_to_mesh_upper == fidelity.source_to_mesh_upper
        assert renewed.boundary_equality is not None
        assert renewed.boundary_equality.predecessor is root
        assert renewed.boundary_equality.target_coverage is target_coverage
        renewed.binding.require(target, geometry)
        restored = DomainBoundaryFidelityEvidence.from_stored_fields(
            renewed.boundary_equality
        )
        assert restored.evidence_id == renewed.boundary_equality.evidence_id
        # A reconstructed owner has no process-local issuance, even when all
        # scientific IDs agree. Its retained root must be earned on this same
        # cumulative ledger rather than granted a fresh helper allowance.
        cold_mesh = mesh.with_coordinates(
            mesh.coordinates, numeric_version=mesh.numeric_version
        )
        cold_root = SourceFidelityPredecessor(
            cold_mesh, geometry, source, domain, regions,
            embedding, coverage, fidelity, limits,
        )
        before = budget.work_units
        cold_renewed = prepare_domain_boundary_fidelity(
            cold_root, target, geometry, source, domain, regions, target_embedding,
            target_coverage, limits, tolerance=0.0, sample_order=4, budget=budget,
        )
        assert cold_renewed is not None
        assert cold_renewed.boundary_equality is not None
        assert cold_renewed.boundary_equality.predecessor.mesh is cold_mesh
        assert cold_renewed.boundary_equality.predecessor.fidelity is not fidelity
        assert budget.work_units > before
        # Static query controls are input too, even with identical source IDs
        # and numerical array banks.
        foreign = MappedDomainBoundarySource(
            domain, regions, covering_radius=2.0 * source.covering_radius
        )
        assert prepare_domain_boundary_fidelity(
            root, target, geometry, foreign, domain, regions, target_embedding,
            target_coverage, limits, tolerance=0.0, sample_order=4, budget=budget,
        ) is None
        assert len(authenticated_budgets) >= 3
        assert all(current is budget for current in authenticated_budgets)


def test_boundary_equality_does_not_discard_required_target_projection() -> None:
    radius = 0.371
    surface = octahedron()
    points = radius * np.asarray(surface.coordinates)
    faces = np.asarray(surface.blocks[0].vertices, dtype=np.int32)
    points = np.vstack((points, np.zeros((1, 3))))
    cells = np.column_stack(
        (np.full(faces.shape[0], points.shape[0] - 1, dtype=np.int32), faces)
    )
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[negative] = cells[negative][:, (0, 2, 1, 3)]
    mesh = phx.discretization.CellMesh.from_tetrahedra(points, cells)
    limits = phx.geometry.MeshCertificateLimits()
    tolerance = 0.49 * radius
    implicit = phx.geometry.Sphere((0.0, 0.0, 0.0), radius).compile(
        coordinate_unit=METER
    )
    with coordinate_reference_budget(queries_upper=100_000) as budget:
        profile = AnalyticImplicitProfile(
            implicit, phx.SpatialCoordinateContract.si(),
            tube_radius=tolerance, budget=budget,
        )
        query = ImplicitProjectionBoundarySource(profile, budget=budget)
        domain = phx.geometry.PiecewiseLinearDomain(
            points,
            faces,
            np.tile(np.asarray((0, -1), dtype=np.int64), (faces.shape[0], 1)),
            ("ball",),
            source_id=query.source_id,
        )
        regions = np.zeros(cells.shape[0], dtype=np.int64)
        geometry = phx.discretization.CellGeometrySpec.affine(mesh, budget=budget)
        validity = phx.discretization.certify_cell_geometry_validity(
            geometry, mesh=mesh, budget=budget
        )
        embedding = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, limits=limits, budget=budget
        )
        coverage = phx.geometry.certify_domain_coverage(
            mesh, geometry, domain, regions, embedding=embedding,
            limits=limits, budget=budget,
        )
        fidelity = phx.geometry.certify_source_fidelity(
            mesh, geometry, query, tolerance=tolerance, limits=limits, budget=budget
        )
        assert fidelity.projection_coverage is not None
        root = SourceFidelityPredecessor(
            mesh, geometry, query, domain, regions, embedding, coverage, fidelity, limits
        )
        issue_domain_fidelity_source(root, budget=budget)
        target_points = np.vstack((points, np.mean(points[cells[0]], axis=0)))
        children = np.repeat(cells[:1], 4, axis=0)
        children[np.arange(4), np.arange(4)] = points.shape[0]
        target_cells = np.vstack((children, cells[1:]))
        target = phx.discretization.CellMesh.from_tetrahedra(
            target_points, target_cells, numeric_version="new-volume-epoch"
        )
        target_geometry = phx.discretization.CellGeometrySpec.affine(target, budget=budget)
        target_regions = np.zeros(target_cells.shape[0], dtype=np.int64)
        target_validity = phx.discretization.certify_cell_geometry_validity(
            target_geometry, mesh=target, budget=budget
        )
        target_embedding = phx.geometry.certify_global_embedding(
            target, target_geometry, target_validity, limits=limits, budget=budget
        )
        target_coverage = phx.geometry.certify_domain_coverage(
            target, target_geometry, domain, target_regions, embedding=target_embedding,
            limits=limits, budget=budget,
        )
        assert prepare_domain_boundary_fidelity(
            root, target, target_geometry, query, domain, target_regions, target_embedding,
            target_coverage, limits, tolerance=tolerance, sample_order=4, budget=budget,
        ) is None
        before = budget.ticket.geometry_queries
        renewed = phx.geometry.certify_source_fidelity(
            target, target_geometry, query, tolerance=tolerance, limits=limits, budget=budget
        )
        assert budget.ticket.geometry_queries > before
        assert renewed.status == "certified"
        assert renewed.projection_coverage is not None
        renewed.projection_coverage.binding.require(target, target_geometry)
        assert renewed.projection_coverage.binding.mesh_id != fidelity.projection_coverage.binding.mesh_id

        moved = target.with_coordinates(
            np.asarray(target.coordinates) + np.asarray((3.0 * radius, 0.0, 0.0)),
            numeric_version="outside-source",
        )
        moved_geometry = phx.discretization.CellGeometrySpec.affine(moved, budget=budget)
        rejected_target = phx.geometry.certify_source_fidelity(
            moved, moved_geometry, query, tolerance=tolerance,
            limits=limits, budget=budget,
        )
        assert rejected_target.status != "certified"
        other_implicit = phx.geometry.Sphere((3.0 * radius, 0.0, 0.0), radius).compile(
            coordinate_unit=METER
        )
        other_profile = AnalyticImplicitProfile(
            other_implicit, phx.SpatialCoordinateContract.si(),
            tube_radius=tolerance, budget=budget,
        )
        rejected_source = phx.geometry.certify_source_fidelity(
            target, target_geometry, ImplicitProjectionBoundarySource(other_profile, budget=budget),
            tolerance=tolerance, limits=limits, budget=budget,
        )
        assert rejected_source.status != "certified"
        regional_domain = phx.geometry.PiecewiseLinearDomain(
            points, faces, domain.facet_regions, ("ball", "foreign"),
            source_id=query.source_id,
        )
        rejected_regions = phx.geometry.certify_domain_coverage(
            target, target_geometry, regional_domain, np.ones_like(target_regions),
            embedding=target_embedding, limits=limits, budget=budget,
        )
        assert rejected_regions.status != "certified"


def test_projection_source_retains_original_discovery_through_cold_archive(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    tmp_path: Path,
) -> None:
    from phydrax.geometry.implicit import (
        AdaptiveImplicitSurfacePolicy,
        discover_adaptive_implicit_surface,
    )
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    profile, _ = projection_replay_context
    margin = profile.reach_lower / 4
    domain = np.asarray(profile.boundary_bounds) + np.asarray(
        ((-margin,) * 3, (margin,) * 3)
    )
    discovered = discover_adaptive_implicit_surface(
        profile.geometry,
        domain=domain,
        policy=AdaptiveImplicitSurfacePolicy(
            enclosure="interval",
            initial_level=2,
            minimum_surface_level=3,
            maximum_level=5,
            maximum_boxes=32768,
            maximum_evaluations=2_097_152,
            maximum_faces=32768,
            maximum_intersection_pairs=1_048_576,
        ),
        source_id=profile.source_id,
    )
    source = ImplicitProjectionBoundarySource(profile, discovery=discovered.volume)
    assert source.discovery is discovered.volume
    receipt = write_meshing_source_closure(tmp_path / "original-discovery.npz", source)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored.discovery is not source.discovery
    assert restored.discovery.query_id == discovered.volume.query_id
    assert tree_content_equal(restored.discovery, discovered.volume)
    np.testing.assert_array_equal(restored.discovery.domain, domain)
    np.testing.assert_array_equal(
        restored.discovery.classify_points(profile.center[None]).classes,
        discovered.volume.classify_points(profile.center[None]).classes,
    )


@pytest.mark.parametrize("fault", ("foreign-source", "stale-domain", "false-enclosure"))
def test_projection_source_refuses_foreign_or_changed_discovery(
    projection_replay_context: tuple[
        AnalyticImplicitProfile, ImplicitProjectionCoverageEvidence
    ],
    fault: str,
) -> None:
    from phydrax.geometry.implicit._adaptive_discovery import ImplicitVolumeQuery

    profile, _ = projection_replay_context
    domain = np.asarray(profile.boundary_bounds)
    values = profile.field_bounds.boxes(domain[:1], domain[1:])
    query = ImplicitVolumeQuery(
        profile.field_bounds,
        domain[:1],
        domain[1:],
        values.value_lower,
        values.value_upper,
        np.asarray((0,), dtype=np.int64),
        domain,
        maximum_level=1,
    )
    if fault == "foreign-source":
        foreign = phx.geometry.Sphere(
            tuple(np.asarray(profile.center) + 3.0 * profile.radius), profile.radius
        ).compile(coordinate_unit=METER)
        foreign_profile = AnalyticImplicitProfile(
            foreign, profile.coordinate_contract
        )
        query = eqx.tree_at(lambda value: value.bounds, query, foreign_profile.field_bounds)
    elif fault == "stale-domain":
        query = eqx.tree_at(
            lambda value: value.domain, query, query.domain + profile.radius
        )
    else:
        query = eqx.tree_at(
            lambda value: (value.leaf_value_lower, value.leaf_value_upper),
            query,
            (np.asarray((1.0,)), np.asarray((2.0,))),
        )
    with pytest.raises(ValueError, match="ImplicitVolumeQuery"):
        ImplicitProjectionBoundarySource(profile, discovery=query)
