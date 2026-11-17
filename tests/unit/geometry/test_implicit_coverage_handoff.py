#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import copy
from dataclasses import dataclass, replace

import equinox as eqx
import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._domain_certification import (
    certify_domain_coverage,
    DomainCoverageCertificate,
)
from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry._source_fidelity_certificate import SourceFidelityCertificate
from phydrax.geometry.implicit import _source_charts as chart_owner
from phydrax.geometry.implicit._adaptive_discovery import (
    AdaptiveImplicitBoundarySource,
    AdaptiveImplicitSurface,
)
from phydrax.geometry.implicit._source_charts import (
    _AdaptiveImplicitBoundaryCarrier,
    _ImplicitChartHandoff,
    AdaptiveImplicitRootChartMap,
    AdaptiveImplicitSourceQuery,
    prepare_adaptive_implicit_root_charts,
)
from phydrax.meshing._certification import MeshCertificationReport
from phydrax.meshing._certification_inputs import MeshCertificationInputs
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from tests.unit.geometry.test_implicit_source_charts import (
    _adaptive_source,
    _chart_witness,
)
from tests.unit.meshing.test_native_implicit_routes import _adaptive_volume_construction


@dataclass(frozen=True, slots=True)
class _CoverageCase:
    mesh: CellMesh
    geometry: CellGeometrySpec
    domain: AdaptiveImplicitSurface
    source: AdaptiveImplicitSourceQuery
    fidelity: SourceFidelityCertificate
    embedding: GlobalEmbeddingCertificate
    regions: np.ndarray
    charts: AdaptiveImplicitRootChartMap
    original: DomainCoverageCertificate


@pytest.fixture(scope="module")
def coverage_case() -> _CoverageCase:
    construction, domain = _adaptive_volume_construction(
        NativeVolumeSchedule(radius_edge_bound=8.0, minimum_dihedral_degrees=0.0)
    )
    request = construction.certification
    inputs = MeshCertificationInputs(
        construction.mesh,
        construction.geometry,
        request.schedule,
        source_id=request.source_id,
        source_revision=request.source_revision,
        domain=request.domain,
        cell_regions=request.cell_regions,
        source=request.fidelity_source,
        fidelity_tolerance=request.fidelity_tolerance,
        maximum_normal_deviation=request.maximum_normal_deviation,
        junction_vertices=request.junction_vertices,
        protected_features=request.protected_features,
        feature_source=request.feature_source,
        scoped_fidelity=request.scoped_fidelity,
        fidelity_witness=request.fidelity_witness,
        source_coverage_witness=request.source_coverage_witness,
    )
    report = MeshCertificationReport(inputs, organization=construction.organization)
    fidelity = report.fidelity
    coverage = report.coverage
    if fidelity is None or coverage is None:
        raise AssertionError("Implicit volume must retain fidelity and coverage.")
    source = coverage.source
    if not isinstance(
        source, (AdaptiveImplicitBoundarySource, _AdaptiveImplicitBoundaryCarrier)
    ):
        raise AssertionError("Implicit coverage must retain its adaptive source.")
    witness = fidelity.witness
    if not isinstance(witness, SourceBoundaryChartWitness):
        raise AssertionError("Implicit fidelity must retain its chart witness.")
    if coverage.domain is not domain:
        raise AssertionError("The construction must retain its live discovery domain.")
    charts = prepare_adaptive_implicit_root_charts(
        construction.mesh,
        construction.geometry,
        source,
        witness,
        tolerance=fidelity.tolerance,
        maximum_normal_deviation=fidelity.maximum_normal_deviation,
    )
    return _CoverageCase(
        construction.mesh,
        construction.geometry,
        domain,
        source,
        fidelity,
        report.embedding,
        np.asarray(coverage.cell_regions, dtype=np.int64),
        charts,
        coverage,
    )


def _coverage(
    case: _CoverageCase,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: AdaptiveImplicitSourceQuery,
    fidelity: SourceFidelityCertificate,
    charts: AdaptiveImplicitRootChartMap | None,
    /,
) -> DomainCoverageCertificate:
    return certify_domain_coverage(
        mesh,
        geometry,
        case.domain,
        case.regions,
        embedding=case.embedding,
        source=source,
        fidelity=fidelity,
        root_charts=charts,
    )


def test_live_handoff_matches_independent_coverage(coverage_case: _CoverageCase) -> None:
    case = coverage_case
    independent = _coverage(
        case, case.mesh, case.geometry, case.source, case.fidelity, None
    )
    shared = _coverage(
        case, case.mesh, case.geometry, case.source, case.fidelity, case.charts
    )
    assert independent.status == shared.status == "certified"
    assert (
        shared.certificate_id
        == independent.certificate_id
        == case.original.certificate_id
    )
    assert shared.source_region_degree_ids == independent.source_region_degree_ids
    assert shared.source_distance_upper == independent.source_distance_upper
    assert shared.findings == independent.findings
    assert shared.coverage_method == independent.coverage_method


def test_fidelity_does_not_offer_an_explicit_map(coverage_case: _CoverageCase) -> None:
    case = coverage_case
    handoff = _ImplicitChartHandoff()
    fidelity = certify_source_fidelity(
        case.mesh,
        case.geometry,
        case.source,
        tolerance=case.fidelity.tolerance,
        maximum_normal_deviation=case.fidelity.maximum_normal_deviation,
        witness=case.fidelity.witness,
        root_charts=case.charts,
        _root_chart_handoff=handoff,
    )
    assert fidelity.status == "certified"
    assert fidelity.certificate_id == case.fidelity.certificate_id
    assert handoff.root_charts is None


def test_fidelity_does_not_offer_an_incomplete_map(
    coverage_case: _CoverageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = coverage_case
    incomplete = case.charts
    row = next(
        i
        for i, face in enumerate(incomplete._faces)
        if face.evidence.bound == "root_chart"
    )
    faces = list(incomplete._faces)
    faces[row] = replace(faces[row], pieces=())
    monkeypatch.setattr(incomplete, "_faces", tuple(faces))
    handoff = _ImplicitChartHandoff()
    fidelity = certify_source_fidelity(
        case.mesh,
        case.geometry,
        case.source,
        tolerance=case.fidelity.tolerance,
        maximum_normal_deviation=case.fidelity.maximum_normal_deviation,
        witness=case.fidelity.witness,
        root_charts=incomplete,
        _root_chart_handoff=handoff,
    )
    assert fidelity.status == "certified"
    assert handoff.root_charts is None
    calls: list[int] = []
    original = chart_owner.prepare_adaptive_implicit_root_charts

    def counted(
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: AdaptiveImplicitSourceQuery,
        witness: SourceBoundaryChartWitness,
        /,
        *,
        frame_id: str | None = None,
        tolerance: float | None = None,
        maximum_normal_deviation: float | None = None,
    ) -> AdaptiveImplicitRootChartMap:
        calls.append(1)
        return original(
            mesh,
            geometry,
            source,
            witness,
            frame_id=frame_id,
            tolerance=tolerance,
            maximum_normal_deviation=maximum_normal_deviation,
        )

    monkeypatch.setattr(chart_owner, "prepare_adaptive_implicit_root_charts", counted)
    independent = _coverage(case, case.mesh, case.geometry, case.source, fidelity, None)
    empty = _coverage(
        case, case.mesh, case.geometry, case.source, fidelity, handoff.root_charts
    )
    assert len(calls) == 2
    assert independent.status == empty.status == "certified"
    assert independent.certificate_id == empty.certificate_id


@pytest.mark.parametrize(
    "changed",
    (
        "mesh",
        "geometry",
        "source",
        "witness",
        "frame",
        "control",
        "signed_zero",
        "pieces",
        "map",
    ),
)
def test_live_handoff_refuses_changed_premise(
    coverage_case: _CoverageCase, changed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = coverage_case
    mesh, geometry, source, fidelity = (
        case.mesh,
        case.geometry,
        case.source,
        case.fidelity,
    )
    charts = case.charts
    if changed == "mesh":
        mesh = copy.copy(mesh)
        assert mesh is not case.mesh and mesh.mesh_id == case.mesh.mesh_id
    elif changed == "geometry":
        geometry = copy.copy(geometry)
        assert geometry is not case.geometry
    elif changed == "source":
        source = copy.copy(source)
        assert source is not case.source and source.source_id == case.source.source_id
    elif changed == "witness":
        witness = fidelity.witness
        if not isinstance(witness, SourceBoundaryChartWitness):
            raise AssertionError("The chart witness was lost.")
        rebuilt = copy.copy(witness)
        assert rebuilt is not witness and rebuilt.witness_id == witness.witness_id
        fidelity = eqx.tree_at(lambda value: value.witness, fidelity, rebuilt)
    elif changed == "frame":
        monkeypatch.setattr(charts, "frame_id", "different-live-coordinate-frame")
    elif changed == "control":
        monkeypatch.setattr(
            charts, "tolerance", float(np.nextafter(fidelity.tolerance, np.inf))
        )
    elif changed == "signed_zero":
        fidelity = eqx.tree_at(lambda value: value.tolerance, fidelity, 0.0)
        monkeypatch.setattr(charts, "tolerance", -0.0)
    elif changed == "map":
        charts = copy.copy(charts)
        assert charts is not case.charts and charts.witness_id == case.charts.witness_id
    else:
        row = next(
            i
            for i, face in enumerate(charts._faces)
            if face.evidence.bound == "root_chart"
        )
        faces = list(charts._faces)
        faces[row] = replace(faces[row], pieces=())
        monkeypatch.setattr(charts, "_faces", tuple(faces))
    with pytest.raises(ValueError, match="root charts"):
        _coverage(case, mesh, geometry, source, fidelity, charts)


def test_failed_fidelity_leaves_handoff_empty() -> None:
    _, source, _, mesh, affine, _ = _adaptive_source()
    elements, routes, coordinates = affine.resolve(mesh)
    block = mesh.blocks[0]
    if not isinstance(block, CellBlock):
        raise AssertionError("The fixture requires triangular cells.")
    singular = CellGeometrySpec(
        {block.name: elements[0]},
        {block.name: routes[0]},
        np.zeros_like(coordinates, dtype=np.float64),
    )
    witness = _chart_witness(mesh, singular, source)
    handoff = _ImplicitChartHandoff()
    fidelity = certify_source_fidelity(
        mesh,
        singular,
        source,
        tolerance=0.5,
        witness=witness,
        _root_chart_handoff=handoff,
    )
    assert fidelity.status == "unresolved"
    assert handoff.root_charts is None
