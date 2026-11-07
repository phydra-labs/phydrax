from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization.finite_volume import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.geometry._source_fidelity import validate_source_fidelity_witness
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.meshing import (
    BoundaryLayerControl,
    BoundaryLayerRoute,
    CellFamilyPolicy,
    CellMeshingTarget,
    LayerSchedule,
    MeshingEntityKind,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingScope,
    NativeMeshingOptions,
    NativeMeshingProvider,
    prepare_boundary_layers,
    RegionRole,
    RegionSeed,
    SizeControlStrength,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._association import GeometryAssociation, GeometrySourceEntityRole
from phydrax.meshing._audit import audit_cell_mesh
from phydrax.meshing._boundary_layer import BoundaryLayerMesh
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_generation import NativePolyhedralSchedule
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_sources import (
    NativeLayerCoreSource,
    NativePolyhedralSource,
)
from tests._support.layer_curving_fixtures import (
    _mapped_core_cap_complex,
    _mapped_cube_source_for_layer,
    _mapped_cube_surface_charts,
    _mapped_cube_wall,
)


_COORDINATES = SpatialCoordinateContract.si()


class _LayerPolyhedralFixture(NamedTuple):
    layers: BoundaryLayerMesh
    control: BoundaryLayerControl
    complex: PiecewiseLinearComplex
    vertex_layer_ids: np.ndarray
    charts: SurfaceSourceCharts
    power_source: NativePolyhedralSource


def _bind_layer_source(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    vertex_layer_ids: np.ndarray,
    charts: SurfaceSourceCharts,
    core_power_source: NativePolyhedralSource | None,
    /,
) -> NativeLayerCoreSource:
    layer_cell_count = sum(block.cell_count for block in layers.mesh.blocks)
    cap = layers.cap
    if cap is None:
        raise ValueError(
            "Positive polyhedral core fixture requires the accepted layer cap."
        )
    cap_cell_count = sum(block.cell_count for block in cap.blocks)
    return NativeLayerCoreSource(
        layers,
        complex_,
        charts.source_id,
        charts.source_revision,
        vertex_layer_ids=vertex_layer_ids,
        cap_polygon_ids=np.arange(cap_cell_count, dtype=np.int64),
        layer_regions=np.zeros((layer_cell_count,), dtype=np.int64),
        region_entity_ids=np.asarray(complex_.region_entity_ids, dtype=np.int64),
        region_occurrence_paths=complex_.region_occurrence_paths,
        region_ids=complex_.region_ids,
        core_region_map=np.arange(len(complex_.region_ids), dtype=np.int64),
        fidelity_source=charts.boundary_source,
        source_wall_charts=charts,
        source_boundary_charts=charts,
        maximum_deviation=0.0,
        core_facet_source_ids=complex_.facet_ids,
        core_facet_source_occurrence_paths=complex_.facet_occurrence_paths,
        core_power_source=core_power_source,
    )


def _make_fixture() -> _LayerPolyhedralFixture:
    mapped_source, domain, reference_mesh = _mapped_cube_source_for_layer()
    wall = _mapped_cube_wall(mapped_source, domain, physical=False)
    _, charts = _mapped_cube_surface_charts(
        wall, mapped_source.source_id, mapped_source.source_revision
    )
    schedule = LayerSchedule.geometric(1, 0.02, growth_rate=1.2)
    # Native layer controls are realized on the wall mesh; the layer/core route
    # binds them to the accepted layers by control identity.
    control = BoundaryLayerControl(
        MeshingScope.whole(wall, 2),
        schedule,
        route=BoundaryLayerRoute.ADVANCING,
    )
    layers = prepare_boundary_layers(wall, control)
    complex_, vertex_layer_ids = _mapped_core_cap_complex(
        layers,
        mapped_source.reference.complex,
        np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64),
    )
    power_source = NativePolyhedralSource(
        complex_,
        charts.source_id,
        charts.source_revision,
        sites=np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64),
        weights=np.zeros((1,), dtype=np.float64),
        site_ids=("authored-core-site",),
        fidelity_source=charts.boundary_source,
        maximum_deviation=0.0,
    )
    return _LayerPolyhedralFixture(
        layers, control, complex_, vertex_layer_ids, charts, power_source
    )


def _source_and_specification(
    fixture: _LayerPolyhedralFixture,
    /,
    *,
    include_power_source: bool,
) -> tuple[NativeLayerCoreSource, VolumeMeshingSpec]:
    layers, _, complex_, vertex_layer_ids, charts, power_source = fixture
    source = _bind_layer_source(
        layers,
        complex_,
        vertex_layer_ids,
        charts,
        power_source if include_power_source else None,
    )
    boundary = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        np.asarray(complex_.facet_entity_ids, dtype=np.int64),
    )
    layer_kinds = tuple(sorted({block.cell_kind for block in layers.mesh.blocks}))
    target = CellMeshingTarget(
        3,
        3,
        CellFamilyPolicy(
            required=("polyhedron",),
            allowed_transitions=layer_kinds,
            allow_mixed=True,
        ),
    )
    specification = VolumeMeshingSpec(
        target,
        boundary,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(
                boundary,
                2.0,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                complex_.region_ids[0],
                "core-material",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.0,
    )
    return source, specification


def _options() -> NativeMeshingOptions:
    return NativeMeshingOptions(
        "layer_core",
        volume_schedule=NativeVolumeSchedule(),
        polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
    )


def test_layer_core_refuses_polyhedral_fill_without_authored_power_source() -> None:
    fixture = _make_fixture()
    source, specification = _source_and_specification(fixture, include_power_source=False)

    with pytest.raises(MeshingFailure) as failure:
        NativeMeshingProvider(_options()).plan(
            source, specification, coordinate_contract=_COORDINATES
        )

    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    assert "authored core power source" in str(failure.value)


def test_layer_core_refuses_soft_volume_shape_strength() -> None:
    fixture = _make_fixture()
    source, specification = _source_and_specification(fixture, include_power_source=True)
    options = NativeMeshingOptions(
        "layer_core",
        volume_schedule=NativeVolumeSchedule(shape_strength=SizeControlStrength.SOFT),
        polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
    )

    with pytest.raises(MeshingFailure) as failure:
        NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=_COORDINATES
        )

    assert failure.value.category is MeshingFailureCategory.UNSUPPORTED_COMBINATION
    assert "hard NativeVolumeSchedule shape strength" in str(failure.value)


def test_accepted_layer_cap_shares_polyhedral_core_and_fv_recovers_box_measure() -> None:
    fixture = _make_fixture()
    source, specification = _source_and_specification(fixture, include_power_source=True)
    assert fixture.layers.control_id == fixture.control.control_id
    assert specification.boundary_scope.entity_dimension == 2
    result = (
        NativeMeshingProvider(_options())
        .plan(source, specification, coordinate_contract=_COORDINATES)
        .execute()
    )

    assert result.certification is not None and result.certification.passed
    assert result.compliance.passed
    assert result.mesh.periodic_topology is None
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    assert any(block.cell_kind == "polyhedron" for block in result.mesh.blocks)
    labels = {label.name: label for label in result.labels}
    layer_ids = np.asarray(labels["boundary-layer"].scope.entity_ids, dtype=np.int64)
    core_ids = np.asarray(labels["core"].scope.entity_ids, dtype=np.int64)
    assert layer_ids.size == sum(block.cell_count for block in fixture.layers.mesh.blocks)
    assert core_ids.size == 1
    assert any(zone.name == fixture.complex.region_ids[0] for zone in result.zones)
    # Layer cells keep exact lineage to the accepted layer result; every other
    # association binds the layer/core source revision.
    bindings = {
        (association.source_id, association.source_revision)
        for association in result.associations
    }
    assert (source.source_id, source.source_revision) in bindings
    assert all(
        binding == (source.source_id, source.source_revision)
        or binding[0] == fixture.layers.result_id
        for binding in bindings
    )

    cap = fixture.layers.cap
    if cap is None:
        raise ValueError("Accepted layer cap disappeared before conformity checking.")
    cap_count = sum(block.cell_count for block in cap.blocks)
    patches = {patch.name: patch for patch in result.patches}
    interface_ids = np.asarray(
        patches["layer-core-interface"].scope.entity_ids, dtype=np.int64
    )
    assert interface_ids.size == cap_count
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_row_by_id = {int(identifier): row for row, identifier in enumerate(face_ids)}
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    for identifier in interface_ids:
        face = face_row_by_id[int(identifier)]
        assert neighbor[face] >= 0
        incident = (int(cell_ids[owner[face]]), int(cell_ids[neighbor[face]]))
        assert (incident[0] in layer_ids and incident[1] in core_ids) or (
            incident[1] in layer_ids and incident[0] in core_ids
        )

    polygon_rows = np.asarray(fixture.complex.polygon_vertices, dtype=np.int64).reshape(
        (-1, 3)
    )
    source_vertex_rows = np.asarray(source.vertex_layer_ids, dtype=np.int64)
    mesh_vertex_ids = np.asarray(result.mesh.vertex_global_ids, dtype=np.int64)
    expected_cap_keys = {
        tuple(sorted(mesh_vertex_ids[source_vertex_rows[polygon]].tolist()))
        for polygon in polygon_rows[:cap_count]
    }
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    face_by_key = {
        tuple(sorted(mesh_vertex_ids[face_vertices[start:stop]].tolist())): int(
            identifier
        )
        for start, stop, identifier in zip(
            face_offsets[:-1], face_offsets[1:], face_ids, strict=True
        )
    }
    assert {face_by_key[key] for key in expected_cap_keys} == set(interface_ids.tolist())

    finite_volume = prepare_polyhedral_finite_volume_geometry(result.mesh)
    assert finite_volume.mesh_id == result.mesh.mesh_id
    cell_volumes = np.asarray(finite_volume.cell_volumes, dtype=np.float64)
    assert np.all(cell_volumes > 0.0)
    # The root source is the unit cube; this analytic oracle is independent of
    # the generated cell decomposition and FV divergence formulas.
    assert float(np.sum(cell_volumes)) == pytest.approx(1.0, rel=0.0, abs=1.0e-10)
    exterior_faces = np.flatnonzero(neighbor < 0)
    face_centers = np.asarray(finite_volume.face_centers, dtype=np.float64)[
        exterior_faces
    ]
    exterior_measures = np.asarray(finite_volume.face_measures, dtype=np.float64)[
        exterior_faces
    ]
    plane_distance = np.minimum(np.abs(face_centers), np.abs(face_centers - 1.0))
    on_box_planes = plane_distance <= 1.0e-12
    np.testing.assert_array_equal(np.sum(on_box_planes, axis=1), 1)
    for axis in range(3):
        for coordinate in (0.0, 1.0):
            selected = np.isclose(
                face_centers[:, axis], coordinate, rtol=0.0, atol=1.0e-12
            )
            assert float(np.sum(exterior_measures[selected])) == pytest.approx(
                1.0, rel=0.0, abs=1.0e-10
            )


def test_polyhedral_core_audit_refuses_an_unreplayed_source_edge_face_overlay() -> None:
    fixture = _make_fixture()
    source, specification = _source_and_specification(fixture, include_power_source=True)
    result = (
        NativeMeshingProvider(_options())
        .plan(source, specification, coordinate_contract=_COORDINATES)
        .execute()
    )
    certification = result.certification
    assert certification is not None
    inputs = certification.inputs
    if inputs.source is None or inputs.fidelity_witness is None:
        raise AssertionError("The polyhedral core fixture requires a fidelity witness.")
    token = validate_source_fidelity_witness(
        result.mesh, result.geometry, inputs.source, inputs.fidelity_witness
    )
    face_set = result.mesh.entity_set(2).entity_set_id
    facets = next(
        value
        for value in result.associations
        if value.target_entity_set_id == face_set
        and (value.source_id, value.source_revision)
        == (source.source_id, source.source_revision)
    )
    # Polyhedral constructions emit no source-edge face overlay; an injected
    # one cannot be replayed through reference-cell core charts.
    overlay = GeometryAssociation(
        facets.association_kind,
        facets.source_id,
        facets.source_revision,
        face_set,
        np.asarray(facets.target_global_ids, dtype=np.int64)[:1],
        facets.source_entity_ids[:1],
        np.zeros((1,), dtype=np.float64),
        exact=True,
        source_roles=(GeometrySourceEntityRole.EDGE,),
        source_occurrence_paths=(
            None
            if facets.source_occurrence_paths is None
            else facets.source_occurrence_paths[:1]
        ),
    )

    def overlay_issues(associations: tuple[GeometryAssociation, ...]) -> list[str]:
        report = audit_cell_mesh(
            result.mesh,
            result.geometry,
            associations=associations,
            validated_fidelity=(token,),
            association_source=source,
            layer_core_root_geometry=inputs.layer_core_root_geometry,
            certification_inputs=inputs,
        )
        return [
            issue
            for issue in report.issues
            if issue.startswith("association_source_witness_detail:")
            and '"stage":"edge_face_overlay"' in issue
        ]

    assert overlay_issues(result.associations) == []
    refused = overlay_issues((*result.associations, overlay))
    assert len(refused) == 1
    assert "admits no source-edge face overlay" in refused[0]
