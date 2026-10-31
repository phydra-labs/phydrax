from typing import Literal

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.meshing._certification import MeshCertificationReport
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._certification_schedule import MeshCertificationSchedule


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
