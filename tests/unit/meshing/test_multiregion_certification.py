from dataclasses import replace
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    cad_mesh_solid,
    CadSolidResult,
    SurfaceFindingKind,
    VerdictStatus,
    verify_chart_partition,
    verify_surface_embedding,
    verify_tetrahedra,
    verify_volume,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._projection_contracts import brep_entity_id
from phydrax.geometry.multiregion_surface import MultiRegionSurfaceTopology
from phydrax.meshing._certification import (
    _audit_findings,
    _multiregion_proof,
    MeshCertificationReport,
)
from phydrax.meshing._native_cad_certificate import NativeCadSurfaceWitness
from phydrax.meshing._native_cad_records import (
    _native_cad_regional_surface,
    _RegionalSurfaceRows,
)
from phydrax.meshing._native_construction import NativeMeshOrganization
from phydrax.meshing._plc_source import (
    _require_unambiguous_cad_copies,
    PiecewiseLinearComplex,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers._native_cad_bundle import cad_geometry_bundle
from phydrax.meshing.providers._native_cad_volume import _cad_plc_data
from phydrax.meshing.providers._native_volume import _face_order_sign


@pytest.mark.parametrize(
    ("vertices", "expected"),
    (
        ((3, 5, 7), 1),
        ((5, 7, 3), 1),
        ((7, 3, 5), 1),
        ((3, 7, 5), -1),
        ((7, 5, 3), -1),
        ((5, 3, 7), -1),
    ),
)
def test_recovered_face_orientation_transports_exact_vertex_parity(
    vertices: tuple[int, int, int],
    expected: int,
) -> None:
    assert _face_order_sign((3, 5, 7), vertices) == expected


def test_recovered_face_orientation_rejects_other_vertex_identity() -> None:
    with pytest.raises(ValueError, match="vertex identities"):
        _face_order_sign((3, 5, 7), (3, 5, 8))


@pytest.fixture(scope="module")
def split_box_model() -> BRepModel:
    pytest.importorskip("OCP")
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    from phydrax._physical import SpatialCoordinateContract
    from phydrax.interchange import import_occt_shape

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(
            BRepPrimAPI_MakeBox(gp_Pnt(x, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
        )
    splitter.Perform()
    assert not splitter.HasErrors()
    return import_occt_shape(
        splitter.Shape(), coordinate_contract=SpatialCoordinateContract.si()
    ).model


@pytest.fixture(scope="module")
def multiregion_surface(
    split_box_model: BRepModel,
) -> tuple[MeshCertificationReport, NativeMeshOrganization, CellMeshingResult]:
    model = split_box_model
    scope = phx.meshing.MeshingScope.whole(model, 2)
    spec = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
        maximum_deviation=0.05,
    )
    result = phx.meshing.NativeMeshingProvider().plan(model, spec).execute(prove=True)
    report = result.certification
    assert report is not None and report.multiregion_surface is not None
    organization = NativeMeshOrganization(
        boundary=report.boundary,
        patches=report.patches,
        zones=report.zones,
        labels=report.labels,
        attributes=report.attributes,
        associations=report.associations,
        multiregion_surface=report.multiregion_surface,
    )
    return report, organization, result


def test_missing_multiregion_evidence_does_not_excuse_junction(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    report, organization, _ = multiregion_surface
    assert report.multiregion_evidence is not None
    assert report.multiregion_evidence.accepted
    assert (
        _multiregion_proof(report.inputs, replace(organization, multiregion_surface=None))
        is None
    )
    without = _audit_findings(report.inputs, report.audit, None)
    assert any(finding.check == "nonmanifold_facets" for finding in without)
    qualified = _audit_findings(report.inputs, report.audit, report.multiregion_evidence)
    assert {f.check for f in without} - {
        "nonmanifold_facets",
        "nonmanifold_edges",
        "nonmanifold_vertices",
    } == {f.check for f in qualified}
    assert report.embedding.status == "certified"
    report.revalidate(report.inputs.mesh, report.inputs.geometry)


def test_multiregion_proof_refuses_missing_source_association(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    report, organization, _ = multiregion_surface
    with pytest.raises(ValueError, match="unique source association"):
        _multiregion_proof(report.inputs, replace(organization, associations=()))


@pytest.mark.parametrize("defect", ("inverted", "open", "wrong-region"))
def test_multiregion_proof_refuses_altered_region_topology(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
    defect: Literal["inverted", "open", "wrong-region"],
) -> None:
    report, organization, _ = multiregion_surface
    original = report.multiregion_surface
    assert original is not None
    faces = original.host_faces().copy()
    labels = original.host_face_labels().copy()
    face_ids = np.asarray(original.face_global_ids[: original.face_count], dtype=np.int64)
    if defect == "inverted":
        faces[0] = faces[0, ::-1]
    elif defect == "open":
        faces, labels, face_ids = faces[:-1], labels[:-1], face_ids[:-1]
    else:
        labels[0] = labels[0, ::-1]
    with pytest.raises(ValueError):
        malformed = MultiRegionSurfaceTopology(
            original.plan,
            faces,
            labels,
            original.region_ids,
            original.region_kinds,
            vertex_count=original.vertex_count,
            vertex_global_ids=np.asarray(
                original.vertex_global_ids[: original.vertex_count]
            ),
            face_global_ids=face_ids,
            epoch=original.epoch,
            domain=original.domain,
        )
        _multiregion_proof(
            report.inputs, replace(organization, multiregion_surface=malformed)
        )


@pytest.fixture(scope="module")
def cad_plc(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> PiecewiseLinearComplex:
    report, _, result = multiregion_surface
    domain = report.inputs.feature_source
    charts = result.surface_source
    assert isinstance(domain, MeshingDomain) and isinstance(charts, SurfaceSourceCharts)
    return _cad_plc_data(domain, result, charts, "conforming", 0.05).complex


def _rebuild_plc(
    plc: PiecewiseLinearComplex,
    /,
    *,
    charts: SurfaceSourceCharts | None,
    vertex_rows: np.ndarray | None,
    segment_edge_ids: np.ndarray | None,
    polygon_cell_ids: np.ndarray | None,
    tolerance: float | None,
    facet_regions: np.ndarray | None = None,
    drop_proof: bool = False,
) -> PiecewiseLinearComplex:
    offsets = np.asarray(plc.polygon_offsets)
    loops = tuple(
        np.asarray(plc.polygon_vertices)[first:last]
        for first, last in zip(offsets[:-1], offsets[1:], strict=True)
    )
    return PiecewiseLinearComplex(
        plc.vertices,
        loops,
        plc.polygon_facets,
        plc.facet_regions if facet_regions is None else facet_regions,
        plc.region_ids,
        segments=plc.segments,
        vertex_ids=plc.vertex_ids,
        vertex_occurrence_paths=plc.vertex_occurrence_paths,
        facet_ids=plc.facet_ids,
        facet_occurrence_paths=plc.facet_occurrence_paths,
        segment_ids=plc.segment_ids,
        segment_occurrence_paths=plc.segment_occurrence_paths,
        region_occurrence_paths=plc.region_occurrence_paths,
        boundary=plc.boundary,
        geometry=plc.geometry,
        facet_planes=plc.facet_planes,
        facet_plane_tolerances=plc.facet_plane_tolerances,
        cad_source_charts=charts,
        cad_fidelity_tolerance=tolerance,
        cad_embedding=plc.cad_embedding if charts is not None else None,
        cad_fidelity=plc.cad_fidelity if charts is not None and not drop_proof else None,
        cad_maximum_normal_deviation=plc.cad_maximum_normal_deviation
        if charts is not None
        else None,
        cad_vertex_rows=vertex_rows,
        cad_segment_edge_ids=segment_edge_ids,
        cad_polygon_cell_ids=polygon_cell_ids,
    )


def test_generic_plc_still_refuses_unproven_occurrence_join(
    cad_plc: PiecewiseLinearComplex,
) -> None:
    with pytest.raises(ValueError, match="occurrence path"):
        _rebuild_plc(
            cad_plc,
            charts=None,
            vertex_rows=None,
            segment_edge_ids=None,
            polygon_cell_ids=None,
            tolerance=None,
        )


@pytest.mark.parametrize(
    "defect",
    (
        "edge",
        "vertex",
        "polygon",
        "region",
        "stale-charts",
        "association",
        "no-fidelity",
        "no-proof",
    ),
)
def test_cad_plc_refuses_forged_ancestry(
    cad_plc: PiecewiseLinearComplex,
    defect: Literal[
        "edge",
        "vertex",
        "polygon",
        "region",
        "stale-charts",
        "association",
        "no-fidelity",
        "no-proof",
    ],
) -> None:
    charts = cad_plc.cad_source_charts
    vertices = cad_plc.cad_vertex_rows
    edges = cad_plc.cad_segment_edge_ids
    polygons = cad_plc.cad_polygon_cell_ids
    tolerance = cad_plc.cad_fidelity_tolerance
    facet_regions = None
    assert charts is not None and vertices is not None and edges is not None
    assert polygons is not None and tolerance is not None
    if defect == "edge":
        edges = edges.copy()
        edges[0] = int(edges.max()) + 1
    elif defect == "vertex":
        vertices = vertices.copy()
        vertices[0] = (vertices[0] + 1) % len(cad_plc.vertex_ids)
    elif defect == "polygon":
        polygons = polygons.copy()
        polygons[0] = int(polygons.max()) + 1
    elif defect == "region":
        facet_regions = np.array(cad_plc.facet_regions, copy=True)
        facet_regions[0] = facet_regions[0, ::-1]
    elif defect == "stale-charts":
        patches = np.array(charts.root_patches, copy=True)
        patches[0] = (patches[0] + 1) % len(charts.domain.patches)
        charts = eqx.tree_at(lambda value: value.root_patches, charts, patches)
    elif defect == "association":
        charts = eqx.tree_at(
            lambda value: value.associations, charts, charts.associations[:-1]
        )
    elif defect == "no-fidelity":
        tolerance = None
    with pytest.raises((ValueError, TypeError)):
        _rebuild_plc(
            cad_plc,
            charts=charts,
            vertex_rows=vertices,
            segment_edge_ids=edges,
            polygon_cell_ids=polygons,
            tolerance=tolerance,
            facet_regions=facet_regions,
            drop_proof=defect == "no-proof",
        )


def test_cad_plc_restores_exact_scientific_authority(
    cad_plc: PiecewiseLinearComplex,
) -> None:
    rebuilt = PiecewiseLinearComplex.from_stored_fields(cad_plc)
    assert rebuilt.complex_id == cad_plc.complex_id
    assert rebuilt.cad_source_charts is not None
    assert rebuilt.cad_segment_edge_ids is not None
    assert not rebuilt.cad_segment_edge_ids.flags.writeable


def test_cad_plc_rejects_ambiguous_repeated_shared_copy(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    from phydrax.geometry.brep._model import (
        BRepAssemblyGraph,
        BRepAssemblyUse,
        BRepGeometry,
    )

    report, _, _ = multiregion_surface
    domain = report.inputs.feature_source
    assert isinstance(domain, MeshingDomain) and domain.brep_authority is not None
    source = domain.brep_authority
    geometry = source.geometry
    graph = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("first", solid_index=0),
            BRepAssemblyUse("repeated", solid_index=0),
            BRepAssemblyUse("neighbor", solid_index=1),
        ),
        definitions=(),
        solid_count=2,
    )
    repeated = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=graph,
    )
    with pytest.raises(ValueError, match="Repeated coincident"):
        _require_unambiguous_cad_copies(
            repeated, source.source_id, source.source_revision
        )


def test_cad_plc_handoffs_fresh_surface_proof(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
    cad_plc: PiecewiseLinearComplex,
) -> None:
    import copy
    import pickle

    from phydrax.meshing import _plc_source
    from phydrax.meshing._model_codec import _same_model_content

    report, _, result = multiregion_surface
    charts = result.surface_source
    domain = report.inputs.feature_source
    assert isinstance(charts, SurfaceSourceCharts)
    assert isinstance(domain, MeshingDomain)
    # Exercise the producer path that must compute a missing fidelity proof.
    without_fidelity = eqx.tree_at(lambda value: value.fidelity, report, None)
    source = eqx.tree_at(lambda value: value.certification, result, without_fidelity)
    rebuilt = _cad_plc_data(domain, source, charts, "conforming", 0.05).complex
    assert _same_model_content(rebuilt.cad_embedding, cad_plc.cad_embedding)
    assert _same_model_content(rebuilt.cad_fidelity, cad_plc.cad_fidelity)
    assert rebuilt.complex_id == cad_plc.complex_id
    assert rebuilt.cad_embedding is not None
    assert rebuilt.cad_fidelity_tolerance is not None
    junctions = np.asarray(
        rebuilt.cad_embedding.binding.junction_vertices, dtype=np.int64
    )
    artifact = _plc_source._ValidatedCadSurface(
        charts,
        rebuilt.cad_fidelity_tolerance,
        rebuilt.cad_maximum_normal_deviation,
        junctions,
    )
    assert all(value is not artifact for value in vars(rebuilt).values())
    with pytest.raises(ValueError, match="different proof operands"):
        artifact.require(
            charts,
            rebuilt.cad_fidelity_tolerance + 1.0,
            rebuilt.cad_maximum_normal_deviation,
            junctions,
        )
    with pytest.raises(ValueError, match="different proof operands"):
        artifact.require(
            charts,
            rebuilt.cad_fidelity_tolerance,
            rebuilt.cad_maximum_normal_deviation,
            np.append(junctions, -1),
        )
    with pytest.raises(ValueError, match="different proof operands"):
        artifact.require(
            copy.copy(charts),
            rebuilt.cad_fidelity_tolerance,
            rebuilt.cad_maximum_normal_deviation,
            junctions,
        )
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(artifact)
    # The constructor's verification binds only this exact, unchanged object.
    assert PiecewiseLinearComplex.from_stored_fields(rebuilt) is rebuilt
    # A copy or pytree rebuild is untrusted even in this process and must replay.
    for untrusted in (
        jax.tree_util.tree_map(lambda leaf: leaf, rebuilt),
        copy.copy(rebuilt),
    ):
        assert untrusted is not rebuilt
        restored = PiecewiseLinearComplex.from_stored_fields(untrusted)
        assert restored is not untrusted
        assert restored.complex_id == rebuilt.complex_id
        assert restored.cad_embedding is not rebuilt.cad_embedding
        assert _same_model_content(restored.cad_embedding, rebuilt.cad_embedding)
        # That full replay reproduced this live value exactly, so it is now
        # verified and is not proved a second time.
        assert PiecewiseLinearComplex.from_stored_fields(untrusted) is untrusted
    # An in-place content change voids the reuse: the full replay refuses it.
    vertices = rebuilt.vertices
    chain = [vertices]
    while isinstance(chain[-1].base, np.ndarray):
        chain.append(chain[-1].base)
    original = float(vertices[0, 0])
    for array in reversed(chain):
        array.setflags(write=True)
    vertices[0, 0] = original + 0.125
    try:
        with pytest.raises(ValueError):
            PiecewiseLinearComplex.from_stored_fields(rebuilt)
    finally:
        vertices[0, 0] = original
        for array in chain:
            array.setflags(write=False)
    assert PiecewiseLinearComplex.from_stored_fields(rebuilt) is rebuilt


def test_cad_curve_ancestry_replay_binds_its_exact_live_charts(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy

    import phydrax.geometry._surface_source_support as support

    _, _, result = multiregion_surface
    charts = result.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    replays: list[None] = []
    original = support._cad_root_edges

    def counted(value: SurfaceSourceCharts, /) -> object:
        replays.append(None)
        return original(value)

    monkeypatch.setattr(support, "_cad_root_edges", counted)
    first = support.validate_cad_curve_ancestry(charts, 0.05)
    count = len(replays)
    first[-1] = ()
    again = support.validate_cad_curve_ancestry(charts, 0.05)
    # The same live, unchanged charts at the same tolerance reuse the replay,
    # and callers own their result mappings.
    assert len(replays) == count
    assert -1 not in again
    # Another tolerance or a copy of the charts replays in full.
    support.validate_cad_curve_ancestry(charts, 0.1)
    assert len(replays) == count + 1
    copied = support.validate_cad_curve_ancestry(copy.copy(charts), 0.05)
    assert len(replays) == count + 2
    assert copied == again


def test_native_surface_proofs_revalidate_scientific_content(
    multiregion_surface: tuple[
        MeshCertificationReport, NativeMeshOrganization, CellMeshingResult
    ],
) -> None:
    from phydrax._meshcore import meshcore_identity
    from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
    from phydrax.meshing._certification import _ComputedSurfaceAcceptance
    from phydrax.meshing._model_codec import _same_model_content
    from phydrax.meshing._native_construction import NativeMeshConstruction

    report, _, _ = multiregion_surface
    domain = report.inputs.feature_source
    assert isinstance(domain, MeshingDomain)
    model = domain.brep_authority
    assert model is not None
    scope = phx.meshing.MeshingScope.whole(model, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
        maximum_deviation=0.05,
    )
    plan = phx.meshing.NativeMeshingProvider().plan(model, specification)
    construction = plan.prepared.execute()
    assert isinstance(construction, NativeMeshConstruction)
    computed: list[_ComputedSurfaceAcceptance] = []
    result = _accept_native_construction(
        construction,
        specification,
        _NATIVE_PROVIDER,
        meshcore_identity(),
        plan.plan_id,
        plan.route,
        computed_surface_out=computed,
    )
    assert len(computed) == 1
    charts = result.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    accepted_domain = charts.domain
    live = _cad_plc_data(
        accepted_domain,
        result,
        charts,
        "conforming",
        0.05,
        computed_surface=computed[0],
    ).complex
    renewed = PiecewiseLinearComplex.from_stored_fields(
        jax.tree_util.tree_map(lambda leaf: leaf, live)
    )
    assert renewed.complex_id == live.complex_id
    assert _same_model_content(renewed.cad_embedding, live.cad_embedding)
    assert _same_model_content(renewed.cad_fidelity, live.cad_fidelity)
    # The live handoff is process-local and bound to its actual scientific owners.
    import copy
    import pickle

    from phydrax.discretization._surface_chart_records import SurfaceChartWitness

    acceptance = computed[0]
    changed = eqx.tree_at(
        lambda value: value.mesh, acceptance.inputs, copy.copy(acceptance.inputs.mesh)
    )
    assert not acceptance.matches(changed)
    changed = eqx.tree_at(
        lambda value: value.geometry,
        acceptance.inputs,
        copy.copy(acceptance.inputs.geometry),
    )
    assert not acceptance.matches(changed)
    assert acceptance.inputs.source is not None
    changed = eqx.tree_at(
        lambda value: value.source,
        acceptance.inputs,
        copy.copy(acceptance.inputs.source),
    )
    assert not acceptance.matches(changed)
    witness = acceptance.inputs.fidelity_witness
    assert isinstance(witness, SurfaceChartWitness)
    altered_charts = np.asarray(witness.charts).copy()
    altered_charts[0, 0, 0] += 0.25
    altered_witness = eqx.tree_at(lambda value: value.charts, witness, altered_charts)
    changed = eqx.tree_at(
        lambda value: value.fidelity_witness, acceptance.inputs, altered_witness
    )
    assert not acceptance.matches(changed)
    assert acceptance.inputs.fidelity_tolerance is not None
    changed = eqx.tree_at(
        lambda value: value.fidelity_tolerance,
        acceptance.inputs,
        acceptance.inputs.fidelity_tolerance + 1.0,
    )
    assert not acceptance.matches(changed)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(acceptance)
    _, _, another_result = multiregion_surface
    assert another_result.surface_source is not None
    with pytest.raises(ValueError, match="another report"):
        _cad_plc_data(
            domain,
            another_result,
            another_result.surface_source,
            "conforming",
            0.05,
            computed_surface=acceptance,
        )


class _NativeSplitBox(NamedTuple):
    domain: MeshingDomain
    mesh: CellMesh
    solid: CadSolidResult
    witness: NativeCadSurfaceWitness
    rows: _RegionalSurfaceRows
    interface: np.ndarray


@pytest.fixture(scope="module")
def native_split_box(split_box_model: BRepModel) -> _NativeSplitBox:
    domain = MeshingDomain.from_brep(split_box_model)
    model = cad_geometry_bundle(domain).model
    # Exercise the native construction itself without changing provider
    # admission: this component does not yet prove BRep material occupancy.
    solid = cad_mesh_solid(
        model,
        np.full(model.surface_nodes.size, 0.4),
        np.full(model.surface_nodes.size, 0.05),
        np.full(model.curve_nodes.size, 0.05),
        radius_edge_bound=8.0,
        min_dihedral_degrees=0.0,
    )
    assert len(solid.volume.tetrahedra) > 0, solid.volume.stats
    mesh = CellMesh.from_simplices(
        solid.volume.points,
        solid.volume.tetrahedra,
        dimension=3,
        numeric_version=domain.source_revision,
    )
    np.testing.assert_array_equal(mesh.coordinates, solid.volume.points)
    np.testing.assert_array_equal(mesh.blocks[0].vertices, solid.volume.tetrahedra)
    witness = NativeCadSurfaceWitness(mesh, solid.surface, model.loop_surfaces.size)
    rows = _native_cad_regional_surface(
        mesh, domain, witness, native_surface=solid.surface
    )
    interface = np.flatnonzero(np.all(rows.regions >= 0, axis=1))
    assert interface.size > 1
    return _NativeSplitBox(domain, mesh, solid, witness, rows, interface)


def test_native_split_box_regional_boundary_and_exact_volume(
    native_split_box: _NativeSplitBox,
) -> None:
    domain, mesh, solid, witness, rows, interface = native_split_box
    points, cells = np.asarray(mesh.coordinates), np.asarray(mesh.blocks[0].vertices)
    cell_regions = solid.volume.tetrahedron_regions
    assert len(domain.regions) == 2
    np.testing.assert_array_equal(np.unique(cell_regions), [0, 1])
    assert not verify_chart_partition(
        len(domain.patches), witness.chart_patches, witness.chart_uv, witness.chords()
    ).size
    replay = _native_cad_regional_surface(mesh, domain, witness)
    np.testing.assert_array_equal(replay.patches, rows.patches)
    np.testing.assert_array_equal(replay.regions, rows.regions)

    # The shared patch is one actual BRep face occurrence. Its sign in each
    # solid, not the labels on incident tetrahedra, fixes the declaration.
    (patch,) = np.unique(rows.patches[interface])
    source, identifier, occurrence = rows.patch_occurrences[patch]
    assert source == domain.source_indices[2][patch]
    assert occurrence == domain.source_occurrences[2][patch]
    assert identifier == brep_entity_id(
        domain.source_id, domain.source_revision, 2, source, occurrence_path=occurrence
    )
    authority = domain.brep_authority
    assert isinstance(authority, BRepModel)
    topo = authority.topology
    for region, solid_id in enumerate(domain.region_source_indices):
        source_slot = topo.solid_faces[solid_id].index(source)
        sign = topo.solid_face_orientations[solid_id][source_slot]
        assert np.all(rows.regions[interface, 0 if sign == 1 else 1] == region)

    aggregate = verify_surface_embedding(points, rows.triangles)
    assert aggregate.status is VerdictStatus.VIOLATED
    assert SurfaceFindingKind.EDGE in aggregate.findings["kind"]
    exterior_rows, exterior = rows.outward(None)
    assert not np.intersect1d(exterior_rows, interface).size
    assert len(exterior_rows) + len(interface) == len(rows.triangles)
    assert verify_surface_embedding(points, exterior).status is VerdictStatus.CERTIFIED

    interface_uses = []
    for region in range(len(domain.regions)):
        original, outward = rows.outward(region)
        verdict = verify_surface_embedding(points, outward)
        assert verdict.status is VerdictStatus.CERTIFIED, verdict.findings
        offsets = np.searchsorted(original, interface)
        np.testing.assert_array_equal(original[offsets], interface)
        interface_uses.append(outward[offsets])
    np.testing.assert_array_equal(interface_uses[0], interface_uses[1][:, ::-1])

    tiling = verify_tetrahedra(points, cells)
    assert tiling.status is VerdictStatus.CERTIFIED, tiling
    assert tiling.boundary_faces == len(exterior_rows)
    coverage = verify_volume(
        points,
        cells,
        rows.triangles,
        cell_regions=cell_regions,
        surface_regions=rows.regions,
    )
    assert coverage.certified, coverage
    assert coverage.boundary_faces == len(exterior_rows)


@pytest.mark.parametrize(
    "defect",
    (
        "swapped-sides",
        "reversed-triangle",
        "reversed-chart-and-triangle",
        "wrong-chart-patch",
        "wrong-occurrence",
    ),
)
def test_native_split_box_refuses_invalid_chart_incidence(
    native_split_box: _NativeSplitBox, defect: str
) -> None:
    domain, mesh, solid, witness, rows, interface = native_split_box
    surface = solid.surface
    triangle = int(interface[0])
    chart_row = int(np.flatnonzero(witness.chart_to_mesh == triangle)[0])
    patch = int(rows.patches[triangle])
    match = rf"triangle row {triangle}\."
    if defect == "swapped-sides":
        sides = surface.facet_regions.copy()
        sides[triangle] = sides[triangle, ::-1]
        surface = surface._replace(facet_regions=sides)
    elif defect in ("reversed-triangle", "reversed-chart-and-triangle"):
        triangles = surface.triangles.copy()
        triangles[triangle] = triangles[triangle, ::-1]
        surface = surface._replace(triangles=triangles)
        match = rf"chart row {chart_row}.*oriented boundary triangle row {triangle}\."
        if defect == "reversed-chart-and-triangle":
            chart = surface.chart
            chart_triangles, uv = chart.triangles.copy(), chart.uv.copy()
            chart_triangles[chart_row] = chart_triangles[chart_row, ::-1]
            uv[chart_row] = uv[chart_row, ::-1]
            surface = surface._replace(
                chart=chart._replace(triangles=chart_triangles, uv=uv)
            )
            match = (
                rf"chart row {chart_row} reverses or degenerates source patch row {patch}"
            )
        witness = NativeCadSurfaceWitness(mesh, surface, witness.loop_count)
    elif defect == "wrong-chart-patch":
        patches = surface.chart.patches.copy()
        wrong_patch = next(
            row
            for row, face in enumerate(domain.patches)
            if row != patch and face.orientation == domain.patches[patch].orientation
        )
        patches[chart_row] = wrong_patch
        witness = NativeCadSurfaceWitness(
            mesh,
            surface._replace(chart=surface.chart._replace(patches=patches)),
            witness.loop_count,
        )
        findings = verify_chart_partition(
            len(domain.patches), witness.chart_patches, witness.chart_uv, witness.chords()
        )
        assert np.any(np.isin(findings["patch"], [patch, wrong_patch]))
    else:
        vertices, edges, faces = domain.source_occurrences
        wrong_faces = (*faces[:patch], ("wrong-occurrence",), *faces[patch + 1 :])
        domain = eqx.tree_at(
            lambda value: value.source_occurrences, domain, (vertices, edges, wrong_faces)
        )
        match = "MeshingDomain content changed"
    with pytest.raises(ValueError, match=match):
        _native_cad_regional_surface(mesh, domain, witness, native_surface=surface)


@pytest.mark.parametrize("defect", ("missing", "duplicate"))
def test_native_split_box_refuses_invalid_physical_facet_rows(
    native_split_box: _NativeSplitBox, defect: str
) -> None:
    domain, mesh, solid, witness, rows, interface = native_split_box
    surface = solid.surface
    triangle = int(interface[0])
    chart_row = int(np.flatnonzero(witness.chart_to_mesh == triangle)[0])
    chart = surface.chart
    if defect == "missing":
        mapping = chart.chart_to_mesh.copy()
        mapping[chart_row] = -1
        surface = surface._replace(chart=chart._replace(chart_to_mesh=mapping))
        match = rf"triangle row {triangle} has 0 chart images"
    else:
        duplicate = len(rows.triangles)
        chart = chart._replace(
            triangles=np.concatenate((chart.triangles, chart.triangles[[chart_row]])),
            patches=np.append(chart.patches, chart.patches[chart_row]),
            uv=np.concatenate((chart.uv, chart.uv[[chart_row]])),
            chart_to_mesh=np.append(chart.chart_to_mesh, duplicate),
            edge_chord=np.concatenate((chart.edge_chord, chart.edge_chord[[chart_row]])),
        )
        surface = surface._replace(
            triangles=np.concatenate((surface.triangles, surface.triangles[[triangle]])),
            chart=chart,
        )
        match = rf"triangle row {duplicate} duplicates a facet"
    changed = NativeCadSurfaceWitness(mesh, surface, witness.loop_count)
    with pytest.raises(ValueError, match=match):
        _native_cad_regional_surface(mesh, domain, changed)


@pytest.mark.parametrize(
    "defect",
    (
        "swapped",
        "reversed",
        "undeclared",
        "duplicate",
        "missing-exterior",
        "wrong-cell-label",
    ),
)
def test_native_split_box_rejects_wrong_exact_regional_coverage(
    native_split_box: _NativeSplitBox, defect: str
) -> None:
    _, mesh, solid, _, rows, interface = native_split_box
    points, cells = np.asarray(mesh.coordinates), np.asarray(mesh.blocks[0].vertices)
    cell_regions = solid.volume.tetrahedron_regions.copy()
    triangles, sides = rows.triangles.copy(), rows.regions.copy()
    row = int(rows.outward(None)[0][0] if defect == "missing-exterior" else interface[0])
    # Only for checking the exact diagnostic row, not for declaring any facet.
    incident = np.flatnonzero(
        np.count_nonzero(np.isin(cells, triangles[row]), axis=1) == 3
    )
    assert incident.size == (1 if defect == "missing-exterior" else 2)
    if defect == "swapped":
        sides[row] = sides[row, ::-1]
    elif defect == "reversed":
        triangles[row] = triangles[row, ::-1]
    elif defect in ("undeclared", "missing-exterior"):
        triangles = np.delete(triangles, row, axis=0)
        sides = np.delete(sides, row, axis=0)
    elif defect == "duplicate":
        triangles = np.concatenate((triangles, triangles[[row]]))
        sides = np.concatenate((sides, sides[[row]]))
    else:
        cell_regions[0] = 1 - cell_regions[0]

    coverage = verify_volume(
        points, cells, triangles, cell_regions=cell_regions, surface_regions=sides
    )
    assert not coverage.certified
    if defect == "duplicate":
        assert coverage.unmatched_surface_triangles == 1
        # The last declaration in the key group is used; the original row
        # is the extra one in the native verifier's ordering.
        assert coverage.first_unmatched_surface_triangle == row
    elif defect == "missing-exterior":
        assert coverage.unmatched_boundary_faces == 1
        assert coverage.first_unmatched_cell == incident[0]
    elif defect == "wrong-cell-label":
        assert coverage.region_mismatches == 4
        assert coverage.first_region_mismatch_cell == 0
    else:
        assert coverage.region_mismatches == 1
        assert coverage.first_region_mismatch_cell == incident[0]


def test_native_split_box_v2_reports_edges_of_missing_interface_facet(
    native_split_box: _NativeSplitBox,
) -> None:
    _, mesh, _, _, rows, interface = native_split_box
    points = np.asarray(mesh.coordinates)
    omitted = interface[0]
    a, b, c = (int(vertex) for vertex in rows.triangles[omitted])
    open_edges = sorted(tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a)))
    for region in (0, 1):
        original, outward = rows.outward(region)
        verdict = verify_surface_embedding(points, outward[original != omitted])
        assert verdict.status is VerdictStatus.VIOLATED
        findings = verdict.findings[verdict.findings["kind"] == SurfaceFindingKind.EDGE]
        assert list(zip(findings["first"], findings["second"], strict=True)) == open_edges


def test_native_split_box_rejects_inverted_tetrahedron_in_union(
    native_split_box: _NativeSplitBox,
) -> None:
    _, mesh, _, _, _, _ = native_split_box
    points, cells = np.asarray(mesh.coordinates), np.array(mesh.blocks[0].vertices)
    cells[0, [0, 1]] = cells[0, [1, 0]]
    verdict = verify_tetrahedra(points, cells)
    assert verdict.status is VerdictStatus.VIOLATED
    assert verdict.nonpositive_cells == 1
    assert verdict.first_nonpositive_cell == 0
