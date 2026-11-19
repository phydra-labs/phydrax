"""The native PLC tetrahedral route: candidates and the native degree certificates."""

from __future__ import annotations

import numpy as np

from benchmarks.native_meshing import _icosphere
from phydrax._physical import SpatialCoordinateContract
from phydrax.meshing import CellMeshingCandidate, CellMeshingResult
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingPlan, NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativePlcSource


def _plan() -> NativeMeshingPlan:
    vertices, faces = _icosphere(1)
    count = faces.shape[0]
    complex_ = PiecewiseLinearComplex(
        vertices,
        faces,
        np.arange(count, dtype=np.int64),
        np.tile(np.asarray([[-1, 0]], dtype=np.int64), (count, 1)),
        ("ball",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"v{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"f{row}" for row in range(count)),
        facet_occurrence_paths=((),) * count,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativePlcSource(complex_, "icosphere", "r1")
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
            UniformSizeControl(scope, 0.5, strength=SizeControlStrength.SOFT),
        ),
        region_seeds=(
            RegionSeed(
                np.zeros((3,), dtype=np.float64),
                "ball",
                "ball-material",
                RegionRole.SOLID,
            ),
        ),
    )
    return NativeMeshingProvider(
        NativeMeshingOptions("plc_tetrahedral", volume_schedule=NativeVolumeSchedule())
    ).plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())


def test_unproved_execution_returns_an_uncertified_candidate() -> None:
    candidate = _plan().execute(prove=False)
    assert isinstance(candidate, CellMeshingCandidate)
    assert not isinstance(candidate, CellMeshingResult)
    assert candidate.mesh.blocks[0].cell_count > 0


def test_proving_the_candidate_certifies_by_the_native_degree_theorems() -> None:
    plan = _plan()
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

    certification = _plan().execute().certification
    assert certification is not None and certification.coverage is not None
    embedding = GlobalEmbeddingCertificate.from_stored_fields(certification.embedding)
    assert embedding.certificate_id == certification.embedding.certificate_id
    coverage = DomainCoverageCertificate.from_stored_fields(certification.coverage)
    assert coverage.certificate_id == certification.coverage.certificate_id
