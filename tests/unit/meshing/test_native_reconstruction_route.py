from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry.reconstruction._boundary import ReconstructedTriangleBoundarySource
from phydrax.geometry.reconstruction._core import (
    ReconstructedGeometrySource,
    ReconstructionReport,
)
from phydrax.geometry.simplicial._regions import MeshRegion
from phydrax.meshing._association import GeometryAssociationKind
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_reconstruction import (
    prepare_native_reconstruction,
)


def _reconstructed_source(
    *,
    feature_id: str = "native-reconstructed-tetrahedron",
    report: ReconstructionReport | None = None,
) -> ReconstructedGeometrySource:
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
    region = MeshRegion(jnp.asarray(vertices), jnp.asarray(faces), feature_id=feature_id)
    source_report = report or ReconstructionReport(
        source_kind="point_cloud",
        algorithm="screened-poisson-reconstruction",
        input_digest="a" * 64,
        input_points=128,
        retained_points=128,
        output_vertices=4,
        output_cells=4,
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=(0.0, 0.0, 0.0),
        parameters=(("screening", "4.0"),),
        warnings=("One closure patch is inferred beyond input coverage.",),
        repair_interpretation="poisson_inferred_closure",
    )
    return ReconstructedGeometrySource(region, source_report)


def _edge_key(first: int, last: int, /) -> tuple[int, int]:
    return (first, last) if first <= last else (last, first)


def _triangle_key(first: int, second: int, third: int, /) -> tuple[int, int, int]:
    if first > second:
        first, second = second, first
    if second > third:
        second, third = third, second
    if first > second:
        first, second = second, first
    return first, second, third


def _specification(
    source: ReconstructedGeometrySource,
    *,
    size_controls: tuple[UniformSizeControl, ...] = (),
    size_compliance: SizeCompliancePolicy | None = None,
) -> SurfaceMeshingSpec:
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    return SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=size_controls,
        size_compliance=size_compliance,
        maximum_deviation=0.0,
    )


def test_reconstructed_native_route_preserves_source_identity_and_metadata() -> None:
    source = _reconstructed_source()
    region = source.source
    assert isinstance(region, MeshRegion)
    duplicate = ReconstructedGeometrySource(source.source, source.report)
    assert source.source_id == duplicate.source_id
    assert source.source_revision == duplicate.source_revision
    assert source.scope_entity_set_id(2) == duplicate.scope_entity_set_id(2)
    np.testing.assert_array_equal(
        source.scope_entity_ids(2), duplicate.scope_entity_ids(2)
    )
    source_keys = tuple(
        source.source_entity_key_for_scope(2, int(identifier))
        for identifier in source.scope_entity_ids(2)
    )
    assert len(set(source_keys)) == 4
    domain = source.meshing_domain
    assert domain is not None
    triangle_indices = domain.source_indices[2]
    assert set(source_keys) == {
        (
            source_entity_id(
                source.source_id, source.source_revision, "reconstructed_triangle", index
            ),
            (),
        )
        for index in triangle_indices
    }
    assert source.scope_entity_ids(0).shape == (4,)
    assert source.scope_entity_ids(1).shape == (6,)
    assert source.scope_entity_ids(3).shape == (1,)
    assert domain.source_id == source.source_id
    assert domain.source_revision == source.source_revision
    assert domain.domain_id
    for dimension in range(4):
        np.testing.assert_array_equal(
            domain.scope_entity_ids(dimension), source.scope_entity_ids(dimension)
        )
    assert tuple(domain.source_indices[0]) == tuple(source.persistent_vertex_ids.tolist())
    assert tuple(domain.source_indices[1]) == tuple(source.persistent_edge_ids.tolist())
    assert tuple(domain.source_indices[2]) == tuple(source.persistent_facet_ids.tolist())
    assert domain.region_source_indices == (source.persistent_region_id,)
    for row in range(source.persistent_facet_ids.size):
        scope_id = source.scope_entity_id_for_mesh_row(2, row)
        key = source.source_entity_key_for_mesh_row(2, row)
        assert source.source_entity_key_for_scope(2, scope_id) == key
        assert domain.source_entity_key_for_scope(2, scope_id) == key
        assert source.source_row_for_scope(2, scope_id) == row

    restored = ReconstructedGeometrySource(
        source.source,
        source.report,
        persistent_vertex_ids=source.persistent_vertex_ids,
        persistent_edge_ids=source.persistent_edge_ids,
        persistent_facet_ids=source.persistent_facet_ids,
        persistent_region_id=source.persistent_region_id,
    )
    assert restored.source_revision == source.source_revision
    assert restored.meshing_domain is not None
    assert restored.meshing_domain.domain_id == domain.domain_id

    changed_report = ReconstructedGeometrySource(
        source.source,
        replace(source.report, warnings=("Different reconstruction evidence.",)),
    )
    assert changed_report.source_id == source.source_id
    assert changed_report.source_revision != source.source_revision
    assert changed_report.scope_entity_set_id(2) != source.scope_entity_set_id(2)
    changed_vertices = np.asarray(region.vertices, dtype=np.float64).copy()
    changed_vertices[0, 0] = -0.125
    changed_geometry = ReconstructedGeometrySource(
        MeshRegion(
            jnp.asarray(changed_vertices),
            region.faces,
            feature_id="native-reconstructed-tetrahedron",
        ),
        source.report,
    )
    assert changed_geometry.source_id == source.source_id
    assert changed_geometry.source_revision != source.source_revision
    assert changed_geometry.scope_entity_set_id(2) != source.scope_entity_set_id(2)
    assert not np.array_equal(
        changed_geometry.scope_entity_ids(2), source.scope_entity_ids(2)
    )

    query = ReconstructedTriangleBoundarySource(source)
    distance = query.boundary_distance(np.asarray(((0.0, 0.0, -1.0),)))
    assert distance.semantics == "certified"
    np.testing.assert_array_equal(distance.lower, np.asarray((1.0,)))
    np.testing.assert_array_equal(distance.upper, np.asarray((1.0,)))
    samples = query.boundary_cover(0.5)
    assert samples.complete
    assert samples.semantics == "certified"
    assert samples.points.shape[0] > 2
    assert samples.covering_radius.shape == (samples.points.shape[0],)
    assert samples.point_error.shape == samples.covering_radius.shape
    assert np.all(samples.covering_radius + samples.point_error <= 0.5)
    chart_cover = query.boundary_chart_cover()
    assert chart_cover.complete
    assert chart_cover.semantics == "certified"
    assert chart_cover.source_id == source.source_id
    assert chart_cover.source_revision == source.source_revision
    assert chart_cover.simplices.shape == (4, 3, 3)
    np.testing.assert_array_equal(chart_cover.deviation_bounds, np.zeros((4,)))

    specification = _specification(source)
    contract = SpatialCoordinateContract.si()
    construction = prepare_native_reconstruction(
        source,
        specification,
        NativeMeshingOptions("reconstructed_surface"),
        contract,
    ).execute()

    assert construction.mesh.topological_dimension == 2
    assert construction.mesh.ambient_dimension == 3
    assert all(block.cell_kind == "triangle" for block in construction.mesh.blocks)
    np.testing.assert_array_equal(
        construction.geometry.coordinates,
        construction.mesh.coordinates,
    )
    assert construction.organization.boundary is not None
    boundary = construction.organization.boundary
    assert boundary.mesh is construction.mesh
    assert boundary.metadata.source_id == source.source_id
    assert boundary.metadata.source_revision == source.source_revision
    assert boundary.metadata.coordinate_contract.spatial_id == contract.spatial_id
    assert boundary.metadata.cell_tags == ("reconstructed-surface",) * 4
    assert "poisson_inferred_closure" in boundary.metadata.provenance

    cell_set = construction.mesh.entity_set(2)
    cell_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == cell_set.entity_set_id
    )
    assert cell_association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
    assert cell_association.source_id == source.source_id
    assert cell_association.source_revision == source.source_revision
    assert cell_association.exact
    assert cell_association.complete
    assert set(cell_association.source_entity_ids) == {key[0] for key in source_keys}
    assert set(cell_association.source_occurrence_paths) == {
        key[1] for key in source_keys
    }

    assert len(construction.organization.labels) == 1
    assert construction.organization.labels[0].name == "reconstructed-surface"
    np.testing.assert_array_equal(
        construction.organization.labels[0].scope.entity_ids,
        cell_set.entity_ids,
    )
    assert len(construction.organization.patches) == 1
    assert construction.organization.patches[0].name == "reconstructed-surface"
    assert construction.compliance.passed
    certification = construction.certification
    assert certification.schedule.route == "surface"
    assert certification.source_id == source.source_id
    assert certification.source_revision == source.source_revision
    assert certification.fidelity_source is not None
    assert certification.fidelity_source.source_id == source.source_id
    assert certification.fidelity_source.source_revision == source.source_revision
    assert certification.fidelity_tolerance == 0.0
    assert (
        certification.fidelity_source.boundary_distance(
            np.asarray(((0.0, 0.0, -1.0),))
        ).semantics
        == "certified"
    )
    assert source.report.repair_interpretation == "poisson_inferred_closure"


def test_reconstruction_chart_witness_revalidates_persistent_triangle_keys() -> None:
    source = _reconstructed_source(feature_id="native-reconstructed-chart-witness")
    construction = prepare_native_reconstruction(
        source,
        _specification(source),
        NativeMeshingOptions("reconstructed_surface"),
        SpatialCoordinateContract.si(),
    ).execute()

    fidelity_source = construction.certification.fidelity_source
    witness = construction.certification.fidelity_witness
    domain = source.meshing_domain
    assert fidelity_source is not None
    assert witness is not None
    assert isinstance(witness, SourceBoundaryChartWitness)
    assert domain is not None
    domain_query = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    assert witness.chart_dimension == 2
    assert witness.reference_dimension == 2
    assert witness.source_query_id == domain_query.source_scope_id
    assert witness.source_chart_ids == witness.source_entity_ids
    assert set(witness.source_entity_ids) == {
        domain.entity_id(2, row) for row in range(len(domain.patches))
    }
    witness.require_current(construction.mesh, construction.geometry, fidelity_source)


def test_reconstructed_query_prepares_its_domain_cover_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _domain_boundary
    from phydrax.geometry._source_fidelity import _reconstructed_domain_query

    source = _reconstructed_source(feature_id="native-reconstructed-prepared-domain")
    domain = source.meshing_domain
    assert domain is not None
    query = ReconstructedTriangleBoundarySource(source)
    reference = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    assert query.domain_refusal is None
    assert query.domain_query is not None and query.domain_cover is not None
    assert query.domain_query.source_scope_id == reference.source_scope_id
    expected = reference.boundary_chart_cover()
    np.testing.assert_array_equal(query.domain_cover.simplices, expected.simplices)
    np.testing.assert_array_equal(
        query.domain_cover.deviation_bounds, expected.deviation_bounds
    )
    assert query.domain_cover.restriction_ids == expected.restriction_ids

    # Validation reuses the prepared query: no domain chart cover is recomputed.
    def refuse(self: MeshingDomainBoundarySource, /) -> None:
        raise AssertionError("the prepared domain cover was recomputed")

    monkeypatch.setattr(
        _domain_boundary.MeshingDomainBoundarySource, "boundary_chart_cover", refuse
    )
    assert _reconstructed_domain_query(query) is query.domain_query
    monkeypatch.undo()

    # A source retaining a domain of other content no longer carries the proof.
    other = _reconstructed_source(feature_id="native-reconstructed-other-domain")
    assert other.meshing_domain is not None
    swapped = eqx.tree_at(
        lambda value: value.source.meshing_domain, query, other.meshing_domain
    )
    with pytest.raises(ValueError, match="persistent source vertex/facet IDs"):
        _reconstructed_domain_query(swapped)


def test_reconstructed_persistent_entities_survive_row_permutation() -> None:
    source = _reconstructed_source(feature_id="native-reconstructed-permutation")
    original_region = source.source
    assert isinstance(original_region, MeshRegion)
    original_vertices = np.asarray(original_region.vertices, dtype=np.float64)
    original_faces = np.asarray(original_region.faces, dtype=np.int64)
    vertex_permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
    inverse_permutation = np.empty_like(vertex_permutation)
    inverse_permutation[vertex_permutation] = np.arange(vertex_permutation.size)
    face_permutation = np.asarray((2, 0, 3, 1), dtype=np.int64)
    permuted_region = MeshRegion(
        jnp.asarray(original_vertices[vertex_permutation]),
        jnp.asarray(inverse_permutation[original_faces[face_permutation]]),
        feature_id=original_region.feature_id,
    )
    permuted_vertex_ids = source.persistent_vertex_ids[vertex_permutation]
    edge_ids_by_vertices = {
        _edge_key(
            int(source.persistent_vertex_ids[int(edge[0])]),
            int(source.persistent_vertex_ids[int(edge[1])]),
        ): int(identifier)
        for edge, identifier in zip(
            np.asarray(original_region.triangle_mesh.topology.edges).tolist(),
            source.persistent_edge_ids.tolist(),
            strict=True,
        )
    }
    permuted_edge_ids = np.asarray(
        [
            edge_ids_by_vertices[
                _edge_key(
                    int(permuted_vertex_ids[int(edge[0])]),
                    int(permuted_vertex_ids[int(edge[1])]),
                )
            ]
            for edge in np.asarray(
                permuted_region.triangle_mesh.topology.edges, dtype=np.int64
            ).tolist()
        ],
        dtype=np.int64,
    )
    facet_ids_by_vertices = {
        _triangle_key(
            int(source.persistent_vertex_ids[int(face[0])]),
            int(source.persistent_vertex_ids[int(face[1])]),
            int(source.persistent_vertex_ids[int(face[2])]),
        ): int(identifier)
        for face, identifier in zip(
            original_faces.tolist(),
            source.persistent_facet_ids.tolist(),
            strict=True,
        )
    }
    permuted_facet_ids = np.asarray(
        [
            facet_ids_by_vertices[
                _triangle_key(
                    int(permuted_vertex_ids[int(face[0])]),
                    int(permuted_vertex_ids[int(face[1])]),
                    int(permuted_vertex_ids[int(face[2])]),
                )
            ]
            for face in np.asarray(permuted_region.faces, dtype=np.int64).tolist()
        ],
        dtype=np.int64,
    )
    transformed = ReconstructedGeometrySource(
        permuted_region,
        source.report,
        persistent_vertex_ids=permuted_vertex_ids,
        persistent_edge_ids=permuted_edge_ids,
        persistent_facet_ids=permuted_facet_ids,
        persistent_region_id=source.persistent_region_id,
    )

    assert transformed.source_id == source.source_id
    assert transformed.source_revision == source.source_revision
    assert transformed.meshing_domain is not None
    for dimension in range(4):
        np.testing.assert_array_equal(
            transformed.scope_entity_ids(dimension), source.scope_entity_ids(dimension)
        )
    assert set(transformed.persistent_vertex_ids.tolist()) == set(
        source.persistent_vertex_ids.tolist()
    )
    assert set(transformed.persistent_edge_ids.tolist()) == set(
        source.persistent_edge_ids.tolist()
    )
    assert set(transformed.persistent_facet_ids.tolist()) == set(
        source.persistent_facet_ids.tolist()
    )


def test_reconstructed_route_red_refines_without_leaving_source_triangles() -> None:
    source = _reconstructed_source(feature_id="native-reconstructed-refinement")
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    control = UniformSizeControl(scope, 0.5)
    specification = _specification(
        source,
        size_controls=(control,),
        size_compliance=SizeCompliancePolicy(absolute_tolerance=0.6),
    )
    construction = prepare_native_reconstruction(
        source,
        specification,
        NativeMeshingOptions("reconstructed_surface"),
        SpatialCoordinateContract.si(),
    ).execute()

    assert construction.mesh.entity_set(2).count == 16
    assert construction.compliance.passed
    fidelity_source = construction.certification.fidelity_source
    fidelity_witness = construction.certification.fidelity_witness
    assert fidelity_source is not None
    assert fidelity_witness is not None
    assert isinstance(fidelity_witness, SourceBoundaryChartWitness)
    fidelity_witness.require_current(
        construction.mesh, construction.geometry, fidelity_source
    )

    association = next(
        row
        for row in construction.organization.associations
        if row.target_entity_set_id == construction.mesh.entity_set(2).entity_set_id
    )
    assert association.exact
    assert association.complete
    assert set(association.source_entity_ids) == {
        source.source_entity_key_for_scope(2, int(identifier))[0]
        for identifier in source.scope_entity_ids(2)
    }
    refined_distance = fidelity_source.boundary_distance(
        np.asarray(construction.mesh.coordinates, dtype=np.float64)
    )
    assert refined_distance.semantics == "certified"
    np.testing.assert_array_equal(refined_distance.lower, np.zeros((10,)))
    np.testing.assert_array_equal(refined_distance.upper, np.zeros((10,)))


def test_reconstructed_public_plan_certifies_continuous_source_boundary() -> None:
    source = _reconstructed_source(feature_id="native-reconstructed-public-route")
    contract = SpatialCoordinateContract.si()
    plan = NativeMeshingProvider(NativeMeshingOptions("reconstructed_surface")).plan(
        source, _specification(source), coordinate_contract=contract
    )

    assert plan.route == "reconstructed_surface"
    result = plan.execute()
    certification = result.certification
    boundary = result.boundary
    assert certification is not None
    assert certification.status == "certified"
    assert certification.passed
    assert certification.findings == ()
    assert certification.fidelity is not None
    assert certification.fidelity.coverage is not None
    assert certification.fidelity.coverage.complete
    source_query = certification.inputs.source
    assert isinstance(source_query, ReconstructedTriangleBoundarySource)
    assert certification.fidelity.witness is not None
    assert isinstance(certification.fidelity.witness, SourceBoundaryChartWitness)
    certification.fidelity.witness.require_current(
        result.mesh, result.geometry, source_query
    )
    assert boundary is not None
    assert boundary.mesh is result.mesh
    assert boundary.metadata.source_id == source.source_id
    assert boundary.metadata.source_revision == source.source_revision
    assert boundary.metadata.coordinate_contract.spatial_id == contract.spatial_id
    assert (
        boundary.metadata.cell_tags
        == ("reconstructed-surface",) * result.mesh.entity_set(2).count
    )
    assert result.audit.passed
    assert result.compliance.passed
    assert result.trace.successful


def test_reconstructed_domain_evaluates_patches_in_one_stacked_call() -> None:
    from phydrax.geometry._meshing_domain import StackedSurfaceEvaluation

    source = _reconstructed_source(feature_id="native-reconstructed-stacked")
    domain = source.meshing_domain
    assert domain is not None
    assert all(
        isinstance(patch.surface, StackedSurfaceEvaluation) for patch in domain.patches
    )
    rng = np.random.default_rng(3)
    rows = rng.integers(0, len(domain.patches), size=97)
    charts = rng.random((97, 2))
    charts[::5] = (1.0, 0.0)
    values = domain.evaluate(rows, charts)
    for row, patch in enumerate(rows):
        chart = charts[row]
        single = np.asarray(
            domain.patches[int(patch)].surface.evaluate(jnp.asarray(chart[None, :]))
        )[0]
        assert values[row].tobytes() == single.tobytes()
