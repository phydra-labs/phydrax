import pickle
from typing import Literal

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._global_embedding import (
    certify_global_embedding,
    ValidatedGlobalEmbedding,
)
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry.simplicial import SegmentMesh
from phydrax.meshing._certification import MeshCertificationReport
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    CurveMeshingSpec,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_curve import prepare_native_curve
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativeCurveSource


def _planar_inputs() -> tuple[CellMesh, CellGeometrySpec, PiecewiseLinearDomain]:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    mesh = CellMesh(
        points,
        (CellBlock("triangle", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64)),),
    )
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64),
        np.asarray(((0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("material",),
        source_id="triangle-source",
    )
    return mesh, CellGeometrySpec.affine(mesh), domain


@pytest.mark.parametrize(
    "missing",
    ("domain", "cell_regions", "junction_vertices", "protected_features"),
)
def test_planar_request_refuses_missing_scientific_premise(
    missing: Literal["domain", "cell_regions", "junction_vertices", "protected_features"],
) -> None:
    mesh, geometry, domain = _planar_inputs()
    with pytest.raises(ValueError):
        MeshCertificationInputs(
            mesh,
            geometry,
            MeshCertificationSchedule("planar"),
            source_id=domain.source_id,
            source_revision=domain.source_revision,
            domain=None if missing == "domain" else domain,
            cell_regions=None
            if missing == "cell_regions"
            else np.asarray((0,), dtype=np.int64),
            junction_vertices=None
            if missing == "junction_vertices"
            else np.asarray((), dtype=np.int64),
            protected_features=None if missing == "protected_features" else (),
        )


@pytest.mark.parametrize(
    "assignment", ((-1,), (1,), ()), ids=("negative", "unknown-region", "missing-cell")
)
def test_domain_request_refuses_invalid_material_assignment(
    assignment: tuple[int, ...],
) -> None:
    mesh, geometry, domain = _planar_inputs()
    with pytest.raises(ValueError):
        MeshCertificationInputs(
            mesh,
            geometry,
            MeshCertificationSchedule("planar"),
            source_id=domain.source_id,
            source_revision=domain.source_revision,
            domain=domain,
            cell_regions=np.asarray(assignment, dtype=np.int64),
            junction_vertices=np.asarray((), dtype=np.int64),
            protected_features=(),
        )


def test_domain_request_owns_immutable_material_assignment() -> None:
    mesh, geometry, domain = _planar_inputs()
    assignments = np.asarray((0,), dtype=np.int64)
    request = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=assignments,
        junction_vertices=np.asarray((), dtype=np.int64),
        protected_features=(),
    )
    assignments[0] = 1
    request.require_current(mesh, geometry, domain=domain)
    assert request.cell_regions is not None
    np.testing.assert_array_equal(request.cell_regions, (0,))
    assert not request.cell_regions.flags.writeable


def test_acceptance_renews_without_inventing_absent_metric_quality() -> None:
    mesh, geometry, domain = _planar_inputs()
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        junction_vertices=np.asarray((), dtype=np.int64),
        protected_features=(),
    )
    report = MeshCertificationReport(inputs)
    assert report.passed
    assert np.isnan(report.audit.quality.minimum_metric_quality)
    report.revalidate(mesh, geometry)

    elements, routes, coordinates = geometry.resolve(mesh)
    moved = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(mesh.blocks, elements, strict=True)
        },
        {block.name: route for block, route in zip(mesh.blocks, routes, strict=True)},
        coordinates * 2.0,
    )
    with pytest.raises(ValueError):
        report.revalidate(mesh, moved)


def test_acceptance_reuses_only_an_embedding_bound_to_its_inputs() -> None:
    mesh, geometry, domain = _planar_inputs()
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        junction_vertices=np.asarray((), dtype=np.int64),
        protected_features=(),
    )
    prepared = certify_global_embedding(
        mesh, geometry, certify_cell_geometry_validity(geometry, mesh=mesh)
    )
    report = MeshCertificationReport(inputs, embedding=prepared)
    assert report.passed
    assert report.embedding is prepared
    assert report.report_id == MeshCertificationReport(inputs).report_id
    report.revalidate(mesh, geometry, embedding=prepared)

    elements, routes, coordinates = geometry.resolve(mesh)
    moved = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(mesh.blocks, elements, strict=True)
        },
        {block.name: route for block, route in zip(mesh.blocks, routes, strict=True)},
        coordinates * 2.0,
    )
    foreign = certify_global_embedding(
        mesh, moved, certify_cell_geometry_validity(moved, mesh=mesh)
    )
    with pytest.raises(ValueError, match="MeshCertificateBinding"):
        MeshCertificationReport(inputs, embedding=foreign)
    with pytest.raises(ValueError, match="MeshCertificateBinding"):
        report.revalidate(mesh, geometry, embedding=foreign)


def test_operation_local_embedding_reuse_rejects_foreign_and_serialization() -> None:
    mesh, geometry, domain = _planar_inputs()
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        junction_vertices=np.asarray((), dtype=np.int64),
        protected_features=(),
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    proof = ValidatedGlobalEmbedding(mesh, geometry, validity)
    report = MeshCertificationReport(inputs, validated_embedding=proof)
    assert report.passed
    assert report.embedding is proof.certificate
    assert report.report_id == MeshCertificationReport(inputs).report_id
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(proof)
    other_mesh, other_geometry, _ = _planar_inputs()
    other_validity = certify_cell_geometry_validity(other_geometry, mesh=other_mesh)
    other = ValidatedGlobalEmbedding(other_mesh, other_geometry, other_validity)
    with pytest.raises(ValueError, match="different exact"):
        MeshCertificationReport(inputs, validated_embedding=other)


def test_validated_embedding_accepts_only_its_own_certificate_in_hand() -> None:
    mesh, geometry, domain = _planar_inputs()
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        junction_vertices=np.asarray((), dtype=np.int64),
        protected_features=(),
    )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    proof = ValidatedGlobalEmbedding(mesh, geometry, validity)
    # A route that hands over the same proof twice keeps one validated proof.
    report = MeshCertificationReport(
        inputs, embedding=proof.certificate, validated_embedding=proof
    )
    assert report.passed
    assert report.embedding is proof.certificate
    assert report.report_id == MeshCertificationReport(inputs).report_id
    # Any other in-hand proof is still refused next to a validated embedding.
    other = certify_global_embedding(mesh, geometry, validity)
    assert other is not proof.certificate
    with pytest.raises(ValueError, match="cannot be combined"):
        MeshCertificationReport(inputs, embedding=other, validated_embedding=proof)
    with pytest.raises(ValueError, match="cannot be combined"):
        MeshCertificationReport(inputs, validity=validity, validated_embedding=proof)


def test_local_mesh_validity_does_not_replace_source_domain_coverage() -> None:
    source_mesh, _, domain = _planar_inputs()
    mesh = CellMesh(source_mesh.coordinates * 0.5, source_mesh.blocks)
    geometry = CellGeometrySpec.affine(mesh)
    inputs = MeshCertificationInputs(
        mesh,
        geometry,
        MeshCertificationSchedule("planar"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain=domain,
        cell_regions=np.asarray((0,), dtype=np.int64),
        junction_vertices=np.asarray((), dtype=np.int64),
        protected_features=(),
    )
    report = MeshCertificationReport(inputs)
    assert report.audit.passed
    assert report.embedding.status == "certified"
    assert report.coverage is not None
    assert report.coverage.status != "certified"
    assert not report.passed


@pytest.mark.parametrize(
    ("overall_tolerance", "scoped_tolerance"),
    ((0.25, 0.25), (0.25, 0.01), (0.0, -0.0)),
    ids=("same-bound", "stricter-feature-bound", "signed-zero-identity"),
)
def test_scoped_fidelity_retains_its_own_request_binding(
    overall_tolerance: float,
    scoped_tolerance: float,
) -> None:
    segments = SegmentMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64),
        np.asarray(((0, 1),), dtype=np.int32),
        source_id="scoped-fidelity-line",
    )
    source = NativeCurveSource(
        segments,
        "revision",
        curve_entity_ids=np.asarray((73,), dtype=np.int64),
        vertex_entity_ids=np.asarray((91, 92), dtype=np.int64),
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        np.asarray(source.chart_scope_entity_ids, dtype=np.int64),
    )
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 2.0, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=0.25,
    )
    construction = prepare_native_curve(
        source,
        specification,
        NativeMeshingOptions("curve_arc_length"),
        SpatialCoordinateContract.si(),
    ).execute()
    request = construction.certification
    query, witness = request.fidelity_source, request.fidelity_witness
    assert query is not None and witness is not None
    inputs = MeshCertificationInputs(
        construction.mesh,
        construction.geometry,
        request.schedule,
        source_id=request.source_id,
        source_revision=request.source_revision,
        source=query,
        fidelity_tolerance=overall_tolerance,
        fidelity_witness=witness,
        junction_vertices=request.junction_vertices,
        protected_features=request.protected_features,
        scoped_fidelity=((query, witness, scoped_tolerance),),
        feature_source=request.feature_source,
    )
    report = MeshCertificationReport(inputs, organization=construction.organization)
    scoped = report.scoped_fidelity[0]
    assert scoped.status == "certified", scoped.findings
    scoped.require_inputs(
        construction.mesh,
        construction.geometry,
        query,
        tolerance=scoped_tolerance,
        maximum_normal_deviation=None,
        witness=witness,
    )
    independent = certify_source_fidelity(
        construction.mesh,
        construction.geometry,
        query,
        tolerance=scoped_tolerance,
        witness=witness,
    )
    assert scoped.certificate_id == independent.certificate_id


def test_live_surface_proof_retains_exact_control_identity() -> None:
    from phydrax.geometry._source_fidelity_certificate import SourceFidelityWitness
    from phydrax.geometry.brep import brep_box
    from phydrax.meshing._certification import _ComputedSurfaceAcceptance
    from phydrax.meshing._contracts import SurfaceMeshingSpec
    from phydrax.meshing.providers._native import NativeMeshingProvider

    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="surface-reuse-controls",
    )
    scope = MeshingScope.whole(model, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 1.0),),
        maximum_deviation=0.05,
    )
    construction = NativeMeshingProvider().plan(model, specification).prepared.execute()
    request = construction.certification
    witness = request.fidelity_witness
    assert witness is not None

    def inputs(
        tolerance: float,
        selected_witness: SourceFidelityWitness,
    ) -> MeshCertificationInputs:
        return MeshCertificationInputs(
            construction.mesh,
            construction.geometry,
            request.schedule,
            source_id=request.source_id,
            source_revision=request.source_revision,
            domain=request.domain,
            cell_regions=request.cell_regions,
            source=request.fidelity_source,
            fidelity_tolerance=tolerance,
            maximum_normal_deviation=request.maximum_normal_deviation,
            junction_vertices=request.junction_vertices,
            protected_features=request.protected_features,
            scoped_fidelity=request.scoped_fidelity,
            fidelity_witness=selected_witness,
            feature_source=request.feature_source,
        )

    original = inputs(0.0, witness)
    computed = _ComputedSurfaceAcceptance(
        original,
        construction.organization,
        validity_parent=None,
        validated=(),
    )
    changed = inputs(-0.0, witness)
    assert not computed.matches(changed)
    renewed = MeshCertificationReport(
        changed,
        organization=construction.organization,
        computed_surface=computed,
    )
    independent = MeshCertificationReport(changed, organization=construction.organization)
    assert renewed.passed and independent.passed
    assert renewed.report_id == independent.report_id
    assert renewed.fidelity is not None and computed.report.fidelity is not None
    assert renewed.fidelity.certificate_id != computed.report.fidelity.certificate_id
