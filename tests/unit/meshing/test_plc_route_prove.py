"""The native PLC tetrahedral route: candidates and the native degree certificates."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from benchmarks._meshing_surfaces import icosphere as _icosphere
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.meshing import CellMeshingCandidate, CellMeshingResult
from phydrax.meshing._certification import MeshCertificationReport
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, write_model_artifact
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingPlan, NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativePlcSource


pytestmark = pytest.mark.meshcore

_REPOSITORY = Path(__file__).resolve().parents[3]


def _plan(
    vertices: np.ndarray,
    faces: np.ndarray,
    facet_regions: np.ndarray,
    seeds: tuple[RegionSeed, ...],
    size: float,
    /,
) -> NativeMeshingPlan:
    count = faces.shape[0]
    complex_ = PiecewiseLinearComplex(
        vertices,
        faces,
        np.arange(count, dtype=np.int64),
        facet_regions,
        tuple(seed.region_name for seed in seeds),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"v{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"f{row}" for row in range(count)),
        facet_occurrence_paths=((),) * count,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),) * len(seeds),
        boundary="fixed",
    )
    source = NativePlcSource(complex_, "plc", "r1")
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, size, strength=SizeControlStrength.SOFT),
        ),
        region_seeds=seeds,
    )
    return NativeMeshingProvider(
        NativeMeshingOptions("plc_tetrahedral", volume_schedule=NativeVolumeSchedule())
    ).plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())


def _icosphere_plan() -> NativeMeshingPlan:
    vertices, faces = _icosphere(1)
    return _plan(
        vertices,
        faces,
        np.tile(np.asarray([[-1, 0]], dtype=np.int64), (faces.shape[0], 1)),
        (
            RegionSeed(
                np.zeros((3,), dtype=np.float64),
                "ball",
                "ball-material",
                RegionRole.SOLID,
            ),
        ),
        0.5,
    )


def _bipyramid_plan() -> NativeMeshingPlan:
    # Two regions of one triangular bipyramid: each facet's normal points
    # toward its first region (-1 is the exterior).
    vertices = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((1, 2, 3), (0, 3, 2), (0, 1, 3), (2, 1, 4), (0, 4, 1), (0, 2, 4), (0, 1, 2)),
        dtype=np.int64,
    )
    regions = np.asarray(
        ((-1, 0), (-1, 0), (-1, 0), (-1, 1), (-1, 1), (-1, 1), (0, 1)), dtype=np.int64
    )
    return _plan(
        vertices,
        faces,
        regions,
        (
            RegionSeed(
                np.asarray((0.0, 0.25, 0.25), dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.0, 0.25, -0.25), dtype=np.float64),
                "lower",
                "lower-material",
                RegionRole.SOLID,
            ),
        ),
        0.4,
    )


def test_unproved_execution_returns_an_uncertified_candidate() -> None:
    candidate = _icosphere_plan().execute(prove=False)
    assert isinstance(candidate, CellMeshingCandidate)
    assert not isinstance(candidate, CellMeshingResult)
    assert candidate.mesh.blocks[0].cell_count > 0


def test_proving_the_candidate_certifies_by_the_native_degree_theorems() -> None:
    plan = _icosphere_plan()
    result = plan.execute(prove=False).prove()
    certification = result.certification
    assert certification is not None and certification.passed
    assert certification.embedding.evaluated_checks == (
        "cell_validity",
        "tetrahedral_degree",
    )
    assert certification.embedding.witnesses == ()
    assert certification.coverage is not None
    assert certification.coverage.coverage_method == "tetrahedral_degree"
    direct = plan.execute()
    assert direct.certification is not None
    assert direct.certification.report_id == certification.report_id


def test_stored_degree_certificates_reprove_natively() -> None:
    from phydrax.geometry._domain_certification import DomainCoverageCertificate
    from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate

    certification = _icosphere_plan().execute().certification
    assert certification is not None and certification.coverage is not None
    embedding = GlobalEmbeddingCertificate.from_stored_fields(certification.embedding)
    assert embedding.certificate_id == certification.embedding.certificate_id
    coverage = DomainCoverageCertificate.from_stored_fields(certification.coverage)
    assert coverage.certificate_id == certification.coverage.certificate_id


def _recertified(
    result: CellMeshingResult, points: np.ndarray, cells: np.ndarray, regions: np.ndarray
) -> MeshCertificationReport:
    """Certify a changed mesh of the route's source under the route's schedule."""
    certification = result.certification
    assert certification is not None
    inputs = certification.inputs
    mesh = CellMesh.from_simplices(points, cells, dimension=3, block_name="tampered")
    return MeshCertificationReport(
        MeshCertificationInputs(
            mesh,
            CellGeometrySpec.affine(mesh),
            inputs.schedule,
            source_id=inputs.source_id,
            source_revision=inputs.source_revision,
            domain=inputs.domain,
            cell_regions=regions,
        )
    )


def _orientation(points: np.ndarray, cell: np.ndarray, /) -> float:
    corners = points[cell]
    return float(np.linalg.det(corners[1:] - corners[0]))


def _boundary_cell(cells: np.ndarray, /) -> tuple[int, int]:
    """A cell with a boundary face whose four vertices all lie in other cells."""
    faces = np.sort(cells[:, ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))], axis=2)
    keys, counts = np.unique(faces.reshape((-1, 3)), axis=0, return_counts=True)
    single = {tuple(row) for row in keys[counts == 1].tolist()}
    for row, cell in enumerate(cells.tolist()):
        others = np.delete(cells, row, axis=0)
        if not set(cell) <= set(others.reshape(-1).tolist()):
            continue
        for local in range(4):
            if tuple(faces[row, local].tolist()) in single:
                return row, local
    raise AssertionError("The route mesh has no removable boundary cell.")


def _tampered(
    result: CellMeshingResult, kind: str, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = result.mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64).copy()
    certification = result.certification
    assert certification is not None and certification.inputs.cell_regions is not None
    regions = np.asarray(certification.inputs.cell_regions, dtype=np.int64).copy()
    if kind == "wrong-region-label":
        # Every cell of the two-region bipyramid has the other region beside it.
        regions[0] = 1 - regions[0]
        return points, cells, regions
    row, local = _boundary_cell(cells)
    match kind:
        case "flipped-cell":
            cells[row] = cells[row, (1, 0, 2, 3)]
        case "missing-boundary-face":
            cells = np.delete(cells, row, axis=0)
            regions = np.delete(regions, row)
        case "cell-outside-surface":
            # Reflect the opposite vertex through the boundary face.
            face = np.delete(cells[row], local)
            apex = 2.0 * points[face].mean(axis=0) - points[cells[row, local]]
            points = np.concatenate((points, apex[None, :]))
            outside = np.append(face, points.shape[0] - 1)
            if _orientation(points, outside) < 0.0:
                outside = outside[(1, 0, 2, 3),]
            cells = np.concatenate((cells, outside[None, :]))
            regions = np.append(regions, regions[row])
        case _:
            raise AssertionError(kind)
    return points, cells, regions


@pytest.mark.parametrize("plan", (_icosphere_plan, _bipyramid_plan))
def test_recertified_route_mesh_certifies_by_the_degree_theorems(
    plan: Callable[[], NativeMeshingPlan],
) -> None:
    result = plan().execute()
    certification = result.certification
    assert certification is not None and certification.inputs.cell_regions is not None
    report = _recertified(
        result,
        np.asarray(result.mesh.coordinates, dtype=np.float64),
        np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64),
        np.asarray(certification.inputs.cell_regions, dtype=np.int64),
    )
    assert report.status == "certified"
    assert report.embedding.evaluated_checks == ("cell_validity", "tetrahedral_degree")
    assert report.coverage is not None
    assert report.coverage.coverage_method == "tetrahedral_degree"


@pytest.mark.parametrize(
    ("kind", "plan", "checks"),
    (
        ("flipped-cell", _icosphere_plan, {"cell_validity", "tetrahedral_orientation"}),
        ("missing-boundary-face", _icosphere_plan, {"region_measure"}),
        ("cell-outside-surface", _icosphere_plan, {"region_measure"}),
        ("wrong-region-label", _bipyramid_plan, {"region_measure"}),
    ),
    ids=("flipped-cell", "missing-boundary-face", "cell-outside-surface", "wrong-region"),
)
def test_route_certificate_rejects_a_tampered_mesh(
    kind: str, plan: Callable[[], NativeMeshingPlan], checks: set[str]
) -> None:
    result = plan().execute()
    report = _recertified(result, *_tampered(result, kind))
    assert report.status != "certified"
    observed = {finding.check for finding in report.findings}
    assert observed & checks, observed


def _restored_in_a_fresh_process(path: Path, /) -> list[str]:
    code = (
        "import sys\n"
        "from phydrax.meshing._model_codec import (\n"
        "    MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value)\n"
        "from phydrax.meshing._result import CellMeshingResult\n"
        "restored = read_model_artifact(sys.argv[1], expected_type=CellMeshingResult,\n"
        "    restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS)\n"
        "report = restored.certification\n"
        "print(restored.result_id)\n"
        "print(report.report_id)\n"
        "print(','.join(report.embedding.evaluated_checks))\n"
        "print(report.coverage.coverage_method)\n"
        "print(report.embedding.validity.native_affine_tetrahedra)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_REPOSITORY), *filter(None, (environment.get("PYTHONPATH"),)))
    )
    completed = subprocess.run(
        (sys.executable, "-c", code, str(path)),
        cwd=_REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().splitlines()[-5:]


def test_degree_certified_result_round_trips_in_a_fresh_process(tmp_path: Path) -> None:
    original = _bipyramid_plan().execute()
    report = original.certification
    assert report is not None and report.coverage is not None
    assert report.embedding.validity.native_affine_tetrahedra
    path = write_model_artifact(
        tmp_path / "plc.phx", original, limits=MODEL_ARCHIVE_LIMITS
    )
    assert _restored_in_a_fresh_process(path) == [
        original.result_id,
        report.report_id,
        "cell_validity,tetrahedral_degree",
        "tetrahedral_degree",
        "True",
    ]
