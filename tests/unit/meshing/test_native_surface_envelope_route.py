#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
    SurfaceEnvelopeDistanceWitness,
)
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry.reconstruction._core import (
    ReconstructedGeometrySource,
    ReconstructionReport,
)
from phydrax.geometry.reconstruction._sheet import (
    AuthoredTriangleSheet,
    TriangleSheetRepairEvidence,
)
from phydrax.geometry.simplicial._regions import MeshRegion
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature, RegionControl
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._surface_envelope import (
    prepare_native_surface_envelope,
    SurfaceEnvelopeSource,
)
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_options import NativeMeshingOptions


def _reconstructed_tetrahedron() -> ReconstructedGeometrySource:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
        dtype=np.int32,
    )
    region = MeshRegion(
        jnp.asarray(vertices),
        jnp.asarray(faces),
        feature_id="consumer-surface-envelope-tetrahedron",
    )
    report = ReconstructionReport(
        source_kind="point_cloud",
        algorithm="screened-poisson-reconstruction",
        input_digest="a" * 64,
        input_points=4,
        retained_points=4,
        output_vertices=4,
        output_cells=4,
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=(0.0, 0.0, 0.0),
        parameters=(("screening", "4.0"),),
    )
    return ReconstructedGeometrySource(region, report)


def _request(
    *,
    protected_features: tuple[tuple[int, bool], ...] = (),
) -> tuple[
    ReconstructedGeometrySource,
    SurfaceEnvelopeSource,
    VolumeMeshingSpec,
    NativeMeshingOptions,
    SpatialCoordinateContract,
]:
    original = _reconstructed_tetrahedron()
    contract = SpatialCoordinateContract.si()
    envelope_source_id = f"{original.source_id}:unsigned-distance-envelope"
    source = SurfaceEnvelopeSource(
        original,
        source_id=envelope_source_id,
        coordinate_contract=contract,
        offset=1.5,
        spacing=0.4,
        certificate_spacing=0.4,
        maximum_deviation=3.0,
    )
    specification, options = _specification(source, protected_features)
    return original, source, specification, options, contract


def _specification(
    source: SurfaceEnvelopeSource,
    protected_features: tuple[tuple[int, bool], ...],
    /,
) -> tuple[VolumeMeshingSpec, NativeMeshingOptions]:
    boundary_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    region_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    region_control = RegionControl(
        region_scope,
        source.domain.region_ids[0],
        "envelope-material",
        RegionRole.SOLID,
    )

    feature_kinds = {
        0: FeatureKind.CORNER,
        1: FeatureKind.CURVE,
        2: FeatureKind.SURFACE,
    }
    features = tuple(
        ProtectedFeature(
            MeshingScope(
                source.source_id,
                source.source_revision,
                MeshingEntityKind.GEOMETRY,
                dimension,
                source.scope_entity_set_id(dimension),
                source.scope_entity_ids(dimension)[:1],
            ),
            feature_kinds[dimension],
            hard=hard,
        )
        for dimension, hard in protected_features
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        protected_features=features,
        maximum_deviation=0.0,
        region_controls=(region_control,),
    )
    options = NativeMeshingOptions(
        "surface_envelope_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=1.0e6,
            minimum_dihedral_degrees=0.0,
        ),
    )
    return specification, options


def test_surface_envelope_route_retains_original_and_derived_region_authority() -> None:
    original, source, specification, options, contract = _request()
    assert source.source_id == f"{original.source_id}:unsigned-distance-envelope"
    assert source.source_id != original.source_id
    assert source.source_revision != original.source_revision
    assert source.domain.source_id == source.source_id
    assert source.region_id == source.domain.region_ids[0]
    assert source.region_id != str(original.persistent_region_id)
    assert source.result.surface_triangles.shape[0] == source.scope_entity_ids(2).size
    np.testing.assert_array_equal(
        source.native_vertex_rows,
        np.arange(source.result.surface_vertices.shape[0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        source.native_facet_rows,
        np.arange(source.result.surface_triangles.shape[0], dtype=np.int64),
    )
    assert (
        np.unique(source.native_vertex_entity_ids).size == source.native_vertex_rows.size
    )
    assert np.unique(source.native_facet_entity_ids).size == source.native_facet_rows.size

    construction = prepare_native_surface_envelope(
        source,
        specification,
        options,
        contract,
    ).execute()

    mesh = construction.mesh
    assert mesh.topological_dimension == 3
    assert mesh.ambient_dimension == 3
    assert tuple(block.cell_kind for block in mesh.blocks) == ("simplex:3",)
    assert mesh.blocks[0].cell_count > 0
    vertex_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    target_rows_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        )
    }
    target_ids_by_source_key = {
        (key, occurrence): int(target_id)
        for target_id, key, occurrence, role in zip(
            np.asarray(vertex_association.target_global_ids, dtype=np.int64).tolist(),
            vertex_association.source_entity_ids,
            vertex_association.source_occurrence_paths,
            vertex_association.source_roles,
            strict=True,
        )
        if role is GeometrySourceEntityRole.VERTEX
    }
    target_rows = [
        target_rows_by_id[
            target_ids_by_source_key[source.source_entity_key_for_mesh_row(0, row)]
        ]
        for row in range(source.domain.vertices.shape[0])
    ]
    np.testing.assert_array_equal(
        np.asarray(mesh.coordinates, dtype=np.float64)[target_rows],
        np.asarray(source.domain.vertices, dtype=np.float64),
    )

    certification = construction.certification
    assert certification.source_id == source.source_id
    assert certification.source_revision == source.source_revision
    assert isinstance(certification.fidelity_source, PiecewiseLinearBoundarySource)
    assert certification.fidelity_source.owner is source
    assert certification.fidelity_source.entity_dimension == 2
    assert certification.fidelity_source.is_full_scope
    assert certification.fidelity_tolerance == 0.0
    derived_witness = certification.fidelity_witness
    assert isinstance(derived_witness, SourceBoundaryChartWitness)
    assert derived_witness.source_id == source.source_id
    assert derived_witness.source_revision == source.source_revision
    assert derived_witness.chart_dimension == 2
    certification.fidelity_source.validate_chart_witness(derived_witness)

    assert len(certification.scoped_fidelity) == 1
    envelope_query, witness, tolerance = certification.scoped_fidelity[0]
    assert envelope_query is source.original_boundary_query
    assert isinstance(witness, SurfaceEnvelopeDistanceWitness)
    assert witness.source_id == original.source_id
    assert witness.source_revision == original.source_revision
    assert witness.offset == source.offset
    assert witness.spacing == source.spacing
    assert tolerance == source.maximum_deviation
    np.testing.assert_array_equal(witness.bounds, source.result.bounds)
    witness.require_current(mesh, construction.geometry, envelope_query)

    domain = construction.certification.domain
    assert domain is source.domain
    assert domain.source_id == source.source_id
    assert domain.source_revision == source.source_revision
    np.testing.assert_array_equal(
        construction.certification.cell_regions,
        np.zeros((mesh.blocks[0].cell_count,), dtype=np.int64),
    )

    cell_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(3).entity_set_id
    )
    region_id = int(source.scope_entity_ids(3)[0])
    region_key = source.source_entity_key_for_scope(3, region_id)
    assert cell_association.complete
    assert cell_association.exact
    assert cell_association.source_roles == (GeometrySourceEntityRole.REGION,) * (
        mesh.blocks[0].cell_count
    )
    assert (
        cell_association.source_entity_ids == (region_key[0],) * mesh.blocks[0].cell_count
    )
    assert construction.organization.zones[0].name == domain.region_ids[0]
    assert construction.organization.zones[0].material_id == "envelope-material"
    assert construction.organization.boundary is not None
    assert construction.organization.boundary.metadata.source_id == source.source_id
    assert (
        construction.organization.boundary.metadata.source_revision
        == source.source_revision
    )
    assert construction.compliance.passed
    assert all(
        label.name != "internal_curve" for label in construction.organization.labels
    )


def test_surface_envelope_hard_point_curve_and_face_features_keep_exact_witnesses() -> (
    None
):
    _, source, specification, options, contract = _request(
        protected_features=((0, True), (1, True), (2, True))
    )
    construction = prepare_native_surface_envelope(
        source, specification, options, contract
    ).execute()
    certification = construction.certification
    assert certification.feature_source is source
    assert len(certification.scoped_fidelity) == 4
    for feature, (query, witness, tolerance) in zip(
        specification.protected_features,
        certification.scoped_fidelity[1:],
        strict=True,
    ):
        dimension = feature.scope.entity_dimension
        assert isinstance(query, PiecewiseLinearBoundarySource)
        assert query.entity_dimension == dimension
        np.testing.assert_array_equal(query.entity_ids, feature.scope.entity_ids)
        assert tolerance == feature.maximum_deviation == 0.0
        if dimension == 0:
            assert isinstance(witness, SourceBoundaryPointWitness)
            witness.require_current(construction.mesh, construction.geometry, query)
        else:
            assert isinstance(witness, SourceBoundaryChartWitness)
            assert witness.chart_dimension == dimension
            query.validate_chart_witness(witness)
        certificate = certify_source_fidelity(
            construction.mesh,
            construction.geometry,
            query,
            tolerance=tolerance,
            witness=witness,
        )
        assert (
            max(
                certificate.mesh_to_source_upper,
                certificate.source_to_mesh_upper,
            )
            <= tolerance
        )


def test_surface_envelope_soft_features_are_measured_not_mandatory_scopes() -> None:
    _, source, specification, options, contract = _request(
        protected_features=((1, False),)
    )
    construction = prepare_native_surface_envelope(
        source, specification, options, contract
    ).execute()
    certification = construction.certification
    assert certification.feature_source is source
    assert len(certification.scoped_fidelity) == 1
    feature = specification.protected_features[0]
    key = f"protected:{feature.feature_id}:maximum_deviation"
    achieved = dict(construction.compliance.achieved)
    requested = dict(construction.compliance.requested)
    assert requested[key] == feature.maximum_deviation
    assert achieved[key] == 0.0
    assert key not in construction.compliance.issues
    warning_key = f"soft_feature:{feature.feature_id}:warning"
    assert achieved[warning_key] == 0.0


def test_envelope_chart_query_rejects_a_witness_with_an_omitted_source_facet() -> None:
    _, source, specification, options, contract = _request(
        protected_features=((2, True),)
    )
    construction = prepare_native_surface_envelope(
        source, specification, options, contract
    ).execute()
    query, witness, _ = construction.certification.scoped_fidelity[1]
    assert isinstance(query, PiecewiseLinearBoundarySource)
    assert query.entity_dimension == 2
    selected = np.asarray(source.scope_entity_ids(2), dtype=np.int64)[:2]
    assert selected.size == 2
    broader_query = PiecewiseLinearBoundarySource(source, 2, entity_ids=selected)
    assert isinstance(witness, SourceBoundaryChartWitness)
    assert witness.entity_global_ids.size == 1
    with pytest.raises(ValueError):
        broader_query.validate_chart_witness(witness)


def test_envelope_point_query_rejects_a_witness_with_an_omitted_source_vertex() -> None:
    _, source, specification, options, contract = _request()
    construction = prepare_native_surface_envelope(
        source, specification, options, contract
    ).execute()
    point_ids = np.asarray(source.scope_entity_ids(0), dtype=np.int64)[:2]
    assert point_ids.size == 2
    query = PiecewiseLinearBoundarySource(source, 0, entity_ids=point_ids)
    association = next(
        item
        for item in construction.organization.associations
        if item.target_entity_set_id == construction.mesh.entity_set(0).entity_set_id
    )
    target_by_key = {
        (key, occurrence): int(target_id)
        for target_id, key, occurrence, role in zip(
            np.asarray(association.target_global_ids, dtype=np.int64).tolist(),
            association.source_entity_ids,
            association.source_occurrence_paths,
            association.source_roles,
            strict=True,
        )
        if role is GeometrySourceEntityRole.VERTEX
    }
    key = (query.source_entity_keys[0], query.source_occurrence_paths[0])
    incomplete = SourceBoundaryPointWitness(
        mesh_id=construction.mesh.mesh_id,
        geometry_id=cell_geometry_id(construction.geometry),
        topology_id=construction.mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.query_id,
        vertex_global_ids=np.asarray((target_by_key[key],), dtype=np.int64),
        source_entity_ids=(key[0],),
        occurrence_paths=(key[1],),
    )
    with pytest.raises(ValueError, match="omits or adds a selected source vertex"):
        incomplete.require_current(construction.mesh, construction.geometry, query)


def test_envelope_source_scopes_are_the_derived_plc_entity_banks() -> None:
    _, source, *_ = _request()
    assert source.plc_source.source_id == source.source_id
    assert source.plc_source.source_revision == source.source_revision
    for dimension in range(4):
        row_ids = np.asarray(
            source.scope_entity_ids_by_mesh_row[dimension], dtype=np.int64
        )
        np.testing.assert_array_equal(
            np.sort(row_ids), source.scope_entity_ids(dimension)
        )
        for row, identifier in enumerate(row_ids.tolist()):
            assert source.source_entity_key_for_mesh_row(dimension, row) == (
                source.plc_source.source_entity_key_for_scope(dimension, int(identifier))
            )


def test_envelope_currentness_refuses_a_retained_surface_the_kernel_does_not_reproduce() -> (
    None
):
    _, source, *_ = _request()
    source.require_current()
    moved = np.array(source.result.surface_vertices, dtype=np.float64)
    moved[0, 0] = np.nextafter(moved[0, 0], np.inf)
    tampered = eqx.tree_at(
        lambda envelope: envelope.result,
        source,
        source.result._replace(surface_vertices=moved, volume_vertices=moved),
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        tampered.require_current()


def test_envelope_currentness_trusts_only_a_reproduction_record_of_its_own_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.reconstruction import _envelope

    _, source, *_ = _request()
    runs: list[int] = []
    kernel = _envelope.surface_envelope

    def counted(*args: Any, **kwargs: Any) -> Any:
        runs.append(1)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(_envelope, "surface_envelope", counted)
    # The record the source retained binds its input and output: no kernel run.
    source.require_current()
    assert runs == []
    # A record of other content proves nothing here: the kernel reruns and the
    # genuine result still reproduces.
    foreign = _envelope.SurfaceEnvelopeReproduction(
        input_id=source.reproduction.input_id,
        output_id=source.reproduction.input_id,
    )
    mismatched = eqx.tree_at(lambda envelope: envelope.reproduction, source, foreign)
    mismatched.require_current()
    assert runs == [1]
    # A retained output the record does not bind is refused even though the
    # record itself is genuine.
    moved = np.array(source.result.surface_vertices, dtype=np.float64)
    moved[0, 0] = np.nextafter(moved[0, 0], np.inf)
    tampered = eqx.tree_at(
        lambda envelope: envelope.result,
        source,
        source.result._replace(surface_vertices=moved, volume_vertices=moved),
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        tampered.require_current()
    assert runs == [1, 1]


def test_surface_envelope_source_refuses_an_insufficient_distance_bound() -> None:
    original = _reconstructed_tetrahedron()
    with pytest.raises(MeshingFailure, match="cannot certify"):
        SurfaceEnvelopeSource(
            original,
            source_id=f"{original.source_id}:unsigned-distance-envelope",
            coordinate_contract=SpatialCoordinateContract.si(),
            offset=1.5,
            spacing=0.4,
            certificate_spacing=0.4,
            maximum_deviation=1.5,
        )
    # The offset must dominate the native sampling error sqrt(3) h + h_c.
    with pytest.raises(ValueError, match="must exceed"):
        SurfaceEnvelopeSource(
            original,
            source_id=f"{original.source_id}:unsigned-distance-envelope",
            coordinate_contract=SpatialCoordinateContract.si(),
            offset=0.02,
            spacing=0.16,
            certificate_spacing=0.08,
            maximum_deviation=3.0,
        )


def _authored_open_sheet() -> tuple[np.ndarray, np.ndarray]:
    """Open 3x3 dyadic grid with a hole, a crack, a flip, a duplicate, and a second component."""
    corner = {}
    vertices: list[tuple[float, float, float]] = []
    for j in range(4):
        for i in range(4):
            corner[(i, j)] = len(vertices)
            vertices.append((2.0 * i, 2.0 * j, 0.0))
    # Unwelded crack: cell (1, 2) uses bitwise-coincident copies of its x=2 corners.
    crack = {corner[(1, 2)]: len(vertices), corner[(1, 3)]: len(vertices) + 1}
    vertices.extend(((2.0, 4.0, 0.0), (2.0, 6.0, 0.0)))
    triangles: list[tuple[int, int, int]] = []
    for j in range(3):
        for i in range(3):
            if (i, j) == (1, 1):
                continue  # interior hole
            quad = (
                corner[(i, j)],
                corner[(i + 1, j)],
                corner[(i + 1, j + 1)],
                corner[(i, j + 1)],
            )
            if (i, j) == (1, 2):
                quad = tuple(crack.get(vertex, vertex) for vertex in quad)
            triangles.extend(((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])))
    triangles[5] = triangles[5][::-1]  # inconsistent winding
    triangles.append(triangles[0])  # exact duplicate face
    first = len(vertices)
    vertices.extend(((1.0, 1.0, 0.75), (2.0, 1.0, 0.75), (1.0, 2.0, 0.75)))
    triangles.append((first, first + 1, first + 2))  # second component
    # Exact dyadic scaling keeps the envelope small enough for a focused fill.
    return 0.25 * np.asarray(vertices, dtype=np.float64), np.asarray(
        triangles, dtype=np.int64
    )


def _point_triangle_distances(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Brute-force Euclidean distance from every point to a triangle soup (Ericson regions)."""
    p = points[:, None, :]
    a, b, c = (triangles[None, :, k, :] for k in range(3))
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, axis=-1), np.sum(ac * ap, axis=-1)
    bp, cp = p - b, p - c
    d3, d4 = np.sum(ab * bp, axis=-1), np.sum(ac * bp, axis=-1)
    d5, d6 = np.sum(ab * cp, axis=-1), np.sum(ac * cp, axis=-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = va + vb + vc
        v = vb / denominator
        w = vc / denominator
        closest = a + v[..., None] * ab + w[..., None] * ac
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        on_ab = a + t_ab[..., None] * ab
        on_ac = a + t_ac[..., None] * ac
        on_bc = b + t_bc[..., None] * (c - b)
    # Ericson's Voronoi-region priority: a, b, edge ab, c, edge ac, edge bc, interior.
    conditions = (
        (d1 <= 0.0) & (d2 <= 0.0),
        (d3 >= 0.0) & (d4 <= d3),
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        (d6 >= 0.0) & (d5 <= d6),
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
    )
    choices = (a, b, on_ab, c, on_ac, on_bc)
    for condition, choice in zip(reversed(conditions), reversed(choices), strict=True):
        closest = np.where(
            condition[..., None], np.broadcast_to(choice, closest.shape), closest
        )
    return np.min(np.linalg.norm(p - closest, axis=-1), axis=1)


def _triangle_samples(triangles: np.ndarray, divisions: int) -> np.ndarray:
    weights = (
        np.asarray(
            [
                (i, j, divisions - i - j)
                for i in range(divisions + 1)
                for j in range(divisions + 1 - i)
            ],
            dtype=np.float64,
        )
        / divisions
    )
    return np.einsum("wk,tkd->twd", weights, triangles).reshape((-1, 3))


def test_surface_envelope_wraps_an_authored_imperfect_open_sheet() -> None:
    vertices, triangles = _authored_open_sheet()
    sheet = AuthoredTriangleSheet(vertices, triangles, feature_id="authored-open-sheet")
    assert sheet.repair == TriangleSheetRepairEvidence(
        authored_vertices=21,
        authored_triangles=18,
        repaired_vertices=19,
        repaired_triangles=17,
        repaired_edges=35,
        welded_vertices=2,
        removed_duplicate_triangles=1,
        flipped_triangles=1,
        boundary_edges=19,
        components=2,
    )
    assert sheet.report.repair_interpretation == "authored_sheet_topology_repair"
    assert not sheet.report.watertight
    assert sheet.report.winding_consistent
    assert sheet.report.connected_components == 2
    assert sheet.report.euler_characteristic == 1
    assert dict(sheet.report.parameters)["welded_vertices"] == "2"
    assert "19 boundary edges remain open" in sheet.report.warnings[0]
    np.testing.assert_array_equal(sheet.vertices, vertices)
    np.testing.assert_array_equal(sheet.triangles, triangles)
    # Repair moves no point: the repaired triangles are the authored triangle set.
    authored_keys = {
        tuple(sorted(map(tuple, vertices[face].tolist()))) for face in triangles
    }
    repaired_keys = {
        tuple(sorted(map(tuple, sheet.repaired_vertices[face].tolist())))
        for face in sheet.repaired_triangles
    }
    assert authored_keys == repaired_keys
    oriented_edges = [
        (int(face[local]), int(face[(local + 1) % 3]))
        for face in sheet.repaired_triangles
        for local in range(3)
    ]
    assert len(set(oriented_edges)) == len(oriented_edges)

    contract = SpatialCoordinateContract.si()
    source = SurfaceEnvelopeSource(
        sheet,
        source_id="authored-open-sheet-envelope",
        coordinate_contract=contract,
        offset=1.5,
        spacing=0.5,
        certificate_spacing=0.5,
        maximum_deviation=3.0,
    )
    assert source.original is not sheet
    assert source.original.source_revision == sheet.source_revision
    assert source.original_boundary_query.source_id == sheet.source_id
    bounds = np.asarray(source.result.bounds, dtype=np.float64)
    assert 0.0 < bounds[2] <= 1.5 <= bounds[1] and bounds[0] <= 3.0

    # Independent oracle against the AUTHORED triangles (including the duplicate and flip).
    envelope_triangles = np.asarray(source.domain.vertices, dtype=np.float64)[
        np.asarray(source.domain.facets, dtype=np.int64)
    ]
    authored_triangles = vertices[triangles]
    roundoff = 1.0e-12
    input_to_envelope = _point_triangle_distances(
        _triangle_samples(authored_triangles, 8), envelope_triangles
    )
    assert np.max(input_to_envelope) <= bounds[0] + roundoff
    envelope_points = np.concatenate(
        (
            np.asarray(source.domain.vertices, dtype=np.float64),
            envelope_triangles.mean(axis=1),
        )
    )
    envelope_to_input = _point_triangle_distances(envelope_points, authored_triangles)
    assert np.min(envelope_to_input) >= bounds[2] - roundoff
    assert np.max(envelope_to_input) <= bounds[1] + roundoff

    specification, options = _specification(source, ())
    construction = prepare_native_surface_envelope(
        source, specification, options, contract
    ).execute()
    assert tuple(block.cell_kind for block in construction.mesh.blocks) == ("simplex:3",)
    assert construction.mesh.blocks[0].cell_count > 0
    assert construction.compliance.passed
    achieved = dict(construction.compliance.achieved)
    assert achieved["source_envelope_maximum_deviation"] == max(bounds[0], bounds[1])
    assert achieved["source_envelope_input_to_envelope_upper"] == bounds[0]
    assert achieved["source_envelope_envelope_to_input_upper"] == bounds[1]
    assert achieved["source_envelope_envelope_to_input_lower"] == bounds[2]
    for name, value in sheet.repair.quantities:
        assert achieved[f"source_topology_repair:{name}"] == float(value)
    distance_rows = [
        witness
        for _, witness, _ in construction.certification.scoped_fidelity
        if isinstance(witness, SurfaceEnvelopeDistanceWitness)
    ]
    assert len(distance_rows) == 1
    assert distance_rows[0].source_id == sheet.source_id
    assert distance_rows[0].source_revision == sheet.source_revision


def _mobius_strip() -> tuple[np.ndarray, np.ndarray]:
    count = 6
    top: list[tuple[float, float, float]] = []
    bottom: list[tuple[float, float, float]] = []
    for row in range(count):
        angle = 2.0 * np.pi * row / count
        radial = np.asarray((np.cos(angle), np.sin(angle), 0.0))
        width = 0.3 * (
            np.cos(angle / 2.0) * radial
            + np.sin(angle / 2.0) * np.asarray((0.0, 0.0, 1.0))
        )
        top.append(tuple((2.0 * radial + width).tolist()))
        bottom.append(tuple((2.0 * radial - width).tolist()))
    vertices = np.asarray(top + bottom, dtype=np.float64)
    triangles = []
    for row in range(count):
        upper, lower = row, count + row
        if row + 1 < count:
            next_upper, next_lower = row + 1, count + row + 1
        else:  # half twist: the strip closes with top and bottom exchanged
            next_upper, next_lower = count, 0
        triangles.extend(((upper, lower, next_lower), (upper, next_lower, next_upper)))
    return vertices, np.asarray(triangles, dtype=np.int64)


def test_authored_sheet_refuses_non_orientable_and_non_manifold_topology() -> None:
    vertices, triangles = _mobius_strip()
    with pytest.raises(MeshingFailure, match="non-orientable") as refused:
        AuthoredTriangleSheet(vertices, triangles, feature_id="authored-mobius-strip")
    assert refused.value.stage == "topology_repair"
    assert refused.value.evidence.logical_findings == ("non_orientable_component",)
    assert refused.value.entity_ids == tuple(range(12))

    fan = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    with pytest.raises(
        MeshingFailure, match="more than two distinct triangles"
    ) as refused:
        AuthoredTriangleSheet(
            fan,
            np.asarray(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int64),
            feature_id="authored-non-manifold-fan",
        )
    assert refused.value.evidence.logical_findings == ("non_manifold_edge",)
    assert refused.value.entity_ids == (0, 1, 2)
