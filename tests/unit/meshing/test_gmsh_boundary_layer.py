from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.geometry.surface import SurfaceMetadata, SurfaceModel


pytestmark = [
    pytest.mark.meshing_gmsh,
    pytest.mark.skipif(
        find_spec("gmsh") is None, reason="optional gmsh package is not installed"
    ),
]
requires_meshcore = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)

_COORDINATES = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
_SCHEDULE = phx.meshing.LayerSchedule.geometric(3, 0.02, growth_rate=1.25)
_IMPORT_LIMITS = phx.interchange.ResourceLimits(
    max_bytes=1 << 24,
    max_depth=64,
    max_nodes=100_000,
    max_attributes=2_000_000,
    max_losses=8,
)

# Empirical comparison ceilings inherited from the pre-remediation ball-wall
# residual and generated-layer thickness/growth scenarios in this module.
# Source fixtures: ball_in_box (OCCT sphere radius 0.4 mm), _SCHEDULE, and
# test_planar_provider_layers_lower_to_a_measured_gmsh_boundary_layer_field.
_CAD_RESIDUAL_RELATIVE_TOLERANCE = 1.0e-10
_CAD_RESIDUAL_ABSOLUTE_TOLERANCE = 1.0e-12
_NATIVE_LAYER_RELATIVE_TOLERANCE = 1.0e-6
# The first-layer ceiling matches the adapter's declared BoundaryLayer-field
# measurement allowance. Growth retains the ball-wall fixture's ratio ceiling.
_GMSH_FIELD_THICKNESS_RELATIVE_TOLERANCE = 1.0e-2
_GMSH_FIELD_GROWTH_RELATIVE_TOLERANCE = 2.0e-2
_PLANAR_FIELD_RELATIVE_TOLERANCE = 1.0e-9


def _persist(shape: Any, path: Any, **options: Any) -> Any:
    # OCCT constructs only the external comparison fixture, not the native model.
    phx.interchange.persist_occt_shape(
        shape,
        path,
        coordinate_contract=_COORDINATES,
        linear_deflection=options.get("linear_deflection", 0.01),
        angular_deflection=options.get("angular_deflection", 0.1),
    )
    return phx.interchange.read_cad(
        path,
        phx.interchange.CadImportPolicy(
            _COORDINATES,
            _IMPORT_LIMITS,
            tessellation=phx.geometry.BRepTessellationPolicy(
                linear_deflection=options.get("linear_deflection", 0.01),
                angular_deflection=options.get("angular_deflection", 0.1),
            ),
        ),
        trusted_root=path.parent,
        source_length_unit=_COORDINATES.length_unit,
    ).model


def _ball_in_box(path: Any) -> Any:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(-1.0, -1.0, -1.0), 2.0, 2.0, 2.0).Shape()
    ball = BRepPrimAPI_MakeSphere(gp_Pnt(0.0, 0.0, 0.0), 0.4).Shape()
    return _persist(BRepAlgoAPI_Cut(box, ball).Shape(), path)


def _box(path: Any) -> Any:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    return _persist(
        BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape(),
        path,
        linear_deflection=0.03,
        angular_deflection=0.15,
    )


def _faces(source: Any, predicate: Any) -> Any:
    assert source.geometry is not None
    selected = []
    for face, patch in enumerate(source.patches):
        lower, upper = np.asarray(source.parameter_bounds[face])
        first, second = np.meshgrid(
            np.linspace(lower[0], upper[0], 3),
            np.linspace(lower[1], upper[1], 3),
        )
        parameters = np.column_stack((first.ravel(), second.ravel()))
        points = np.asarray(patch.evaluate(parameters))
        if np.all(predicate(points[None, :, :])):
            selected.append(face)
    return tuple(selected)


def _face_scope(provider: Any, source: Any, indices: Any) -> Any:
    return provider.entity_scope(
        source, tuple(source.face_ids[index] for index in indices)
    )


def _layered_spec(provider: Any, source: Any, control: Any) -> Any:
    whole = provider.whole_scope(source, 3)
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("pyramid", "hexahedron"),
                allow_mixed=True,
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(whole, 0.25, maximum_growth_rate=10.0),
        ),
        layer_controls=(control,),
    )


def _ball_control(provider: Any, source: Any, route: Any, corner: Any) -> Any:
    ball = _faces(source, lambda triangles: np.linalg.norm(triangles, axis=2) < 0.5)
    return phx.meshing.BoundaryLayerControl(
        _face_scope(provider, source, ball),
        _SCHEDULE,
        route=route,
        volume_scope=provider.whole_scope(source, 3),
        corner=corner,
    )


def _face_rows(mesh: Any) -> Any:
    connectivity = mesh.connectivity
    offsets = np.asarray(connectivity.face_vertex_offsets)
    values = np.asarray(connectivity.face_vertex_values)
    return [
        values[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    ]


def _assert_layered_organization(result: Any) -> None:
    zones = {zone.name: zone for zone in result.zones}
    assert set(zones) == {"boundary-layer", "core"}
    cell_kinds = np.concatenate(
        [
            np.full((block.cell_count,), block.cell_kind, dtype=object)
            for block in result.mesh.blocks
        ]
    )
    core_cells = np.asarray(zones["core"].scope.entity_ids)
    assert set(cell_kinds[core_cells]) == {"tetrahedron"}
    layer_cells = np.asarray(zones["boundary-layer"].scope.entity_ids)
    assert "prism" in set(cell_kinds[layer_cells])
    patches = {patch.name: patch for patch in result.patches}
    assert set(patches) == {"wall", "layer-core-interface", "outer"}
    connectivity = result.mesh.connectivity
    owners = np.asarray(connectivity.face_owner)
    neighbors = np.asarray(connectivity.face_neighbor)
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids)
    row_of = {int(identifier): row for row, identifier in enumerate(face_ids)}

    def face_rows(name: Any) -> Any:
        return np.asarray(
            [row_of[int(value)] for value in np.asarray(patches[name].scope.entity_ids)]
        )

    interface = face_rows("layer-core-interface")
    rows = _face_rows(result.mesh)
    assert all(len(rows[face]) == 3 for face in interface)
    zone_of = np.zeros((cell_kinds.size,), dtype=np.int8)
    zone_of[core_cells] = 1
    np.testing.assert_array_equal(
        np.sort(
            np.stack((zone_of[owners[interface]], zone_of[neighbors[interface]]), 1), 1
        ),
        np.tile((0, 1), (interface.size, 1)),
    )
    assert np.all(neighbors[face_rows("wall")] < 0)
    assert np.all(neighbors[face_rows("outer")] < 0)
    assert result.audit.passed and result.compliance.passed


def _assert_curved_wall_association(result: Any, source: Any) -> None:
    boundary = result.boundary
    assert boundary is not None
    association = result.associations[0]
    cells = boundary.mesh.entity_set(2)
    np.testing.assert_array_equal(association.target_global_ids, cells.entity_ids)
    assert association.complete and not association.exact
    assert association.provenance is phx.meshing.GeometryAssociationProvenance.PROVIDER
    corners = np.asarray(boundary.mesh.coordinates)[
        np.asarray(boundary.mesh.blocks[0].vertices)
    ]
    centroids = np.mean(corners, axis=1)
    wall = np.linalg.norm(centroids, axis=1) < 0.5
    ball_faces = _faces(source, lambda points: np.linalg.norm(points, axis=2) < 0.5)
    np.testing.assert_array_equal(
        np.isin(np.asarray(association.source_indices), ball_faces), wall
    )
    # Sphere radius is fixed by the OCCT fixture; centroid sagitta is independent
    # of the source tessellation and of the provider's retained classification.
    np.testing.assert_allclose(
        np.asarray(association.residuals)[wall],
        0.4 - np.linalg.norm(centroids[wall], axis=1),
        rtol=_CAD_RESIDUAL_RELATIVE_TOLERANCE,
        atol=_CAD_RESIDUAL_ABSOLUTE_TOLERANCE,
    )
    assert np.max(np.asarray(association.residuals)[wall]) > (
        4.0 * source.report.linear_deflection
    )
    owned = {entity.index: entity for entity in source.qualified_face_ids}
    expected_entities = tuple(
        owned[int(index)] for index in np.asarray(association.source_indices)
    )
    assert association.source_entities.occurrence_paths() == tuple(
        entity.occurrence_path for entity in expected_entities
    )
    assert association.source_entities.entity_ids() == tuple(
        phx.geometry.brep_entity_id(
            source.source_revision,
            2,
            entity.index,
            occurrence_path=entity.occurrence_path,
        )
        for entity in expected_entities
    )


@requires_meshcore
def test_advancing_layers_around_a_curved_wall_fill_a_conforming_certified_core(
    tmp_path: Any,
) -> None:
    provider = phx.meshing.GmshProvider()
    source = _ball_in_box(tmp_path / "ball.brep")
    control = _ball_control(
        provider,
        source,
        phx.meshing.BoundaryLayerRoute.ADVANCING,
        phx.meshing.BoundaryLayerCornerPolicy.FAN,
    )

    result = provider.plan(source, _layered_spec(provider, source, control)).execute()

    _assert_layered_organization(result)
    _assert_curved_wall_association(result, source)
    achieved = dict(result.compliance.achieved)
    key = f"layer:{control.control_id}"
    for index, thickness in enumerate(_SCHEDULE.thicknesses):
        assert achieved[f"{key}:thickness:{index}"] == pytest.approx(
            thickness, rel=_NATIVE_LAYER_RELATIVE_TOLERANCE
        )
    assert achieved[f"{key}:growth:1"] == pytest.approx(
        1.25, rel=_NATIVE_LAYER_RELATIVE_TOLERANCE
    )
    assert achieved["fixed_boundary_bitwise"] == 1.0
    assert all(association.complete for association in result.associations)
    certificate = phx.discretization.certify_cell_geometry_validity(result.mesh)
    assert certificate.certified_valid_count == sum(
        block.cell_count for block in result.mesh.blocks
    )


@requires_meshcore
def test_provider_boundary_layer_extrusion_meets_measured_layer_compliance(
    tmp_path: Any,
) -> None:
    provider = phx.meshing.GmshProvider()
    source = _ball_in_box(tmp_path / "ball.brep")
    control = _ball_control(
        provider,
        source,
        phx.meshing.BoundaryLayerRoute.PROVIDER,
        phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
    )

    result = provider.plan(source, _layered_spec(provider, source, control)).execute()

    _assert_layered_organization(result)
    _assert_curved_wall_association(result, source)
    achieved = dict(result.compliance.achieved)
    key = f"layer:{control.control_id}"
    assert achieved[f"{key}:layer_count"] == 3
    assert achieved[f"{key}:first_layer_thickness"] == pytest.approx(
        0.02, rel=_GMSH_FIELD_THICKNESS_RELATIVE_TOLERANCE
    )
    assert achieved[f"{key}:growth:1"] == pytest.approx(
        1.25, rel=_GMSH_FIELD_GROWTH_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    ("route", "corner", "wall_predicate", "message"),
    (
        (
            phx.meshing.BoundaryLayerRoute.PROVIDER,
            phx.meshing.BoundaryLayerCornerPolicy.FAN,
            lambda triangles: np.linalg.norm(triangles, axis=2) < 0.5,
            "fan templates",
        ),
        (
            phx.meshing.BoundaryLayerRoute.ADVANCING,
            phx.meshing.BoundaryLayerCornerPolicy.FAN,
            lambda triangles: np.isclose(triangles[:, :, 2], -1.0),
            "closed wall components",
        ),
    ),
)
def test_layered_volume_routes_reject_unsupported_requests_before_generation(
    tmp_path: Any, route: Any, corner: Any, wall_predicate: Any, message: Any
) -> None:
    provider = phx.meshing.GmshProvider()
    source = _ball_in_box(tmp_path / "ball.brep")
    control = phx.meshing.BoundaryLayerControl(
        _face_scope(provider, source, _faces(source, wall_predicate)),
        _SCHEDULE,
        route=route,
        volume_scope=provider.whole_scope(source, 3),
        corner=corner,
    )

    with pytest.raises(phx.meshing.MeshingFailure, match=message) as failure:
        provider.plan(source, _layered_spec(provider, source, control))

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )


@pytest.fixture(scope="module")
def cad_extrusion_request() -> tuple[Any, ...]:
    provider = phx.meshing.GmshProvider()
    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_COORDINATES,
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.03,
            angular_deflection=0.15,
        ),
    )
    schedule = phx.meshing.LayerSchedule(np.asarray((0.08, 0.10, 0.12), dtype=np.float64))
    control = phx.meshing.BoundaryLayerControl(
        _face_scope(
            provider,
            source,
            _faces(source, lambda triangles: np.isclose(triangles[:, :, 2], 0.0)),
        ),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=provider.whole_scope(source, 3),
    )
    return provider, source, schedule, control


def test_cad_extrusion_requires_explicit_preparation(
    cad_extrusion_request: tuple[Any, ...],
) -> None:
    provider, source, _, control = cad_extrusion_request
    with pytest.raises(
        phx.meshing.MeshingFailure, match="prepare_boundary_layer_extrusion"
    ):
        provider.plan(source, _layered_spec(provider, source, control))


@pytest.fixture(scope="module")
def prepared_cad_extrusion(
    cad_extrusion_request: tuple[Any, ...],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Any, ...]:
    provider, source, schedule, _ = cad_extrusion_request
    tmp_path = tmp_path_factory.mktemp("cad-extrusion")
    domain = phx.geometry.MeshingDomain.from_brep(source)
    wall_faces = set(
        _faces(source, lambda triangles: np.isclose(triangles[:, :, 2], 0.0))
    )
    wall_slots = domain.scope_indices(2)
    native_wall = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(
            tuple(
                wall_slots[row]
                for row, face in enumerate(domain.source_indices[2])
                if face in wall_faces
            ),
            dtype=np.int64,
        ),
    )
    native_volume = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        domain.entity_set_id(3),
        np.asarray(domain.scope_indices(3), dtype=np.int64),
    )
    control = phx.meshing.BoundaryLayerControl(
        native_wall,
        schedule,
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=native_volume,
    )

    extrusion = phx.meshing.prepare_boundary_layer_extrusion(
        source, control, destination=tmp_path / "split.brep"
    )

    return provider, extrusion


def test_cad_extrusion_partitions_an_exact_slab(
    prepared_cad_extrusion: tuple[Any, ...],
) -> None:
    _, extrusion = prepared_cad_extrusion
    assert len(extrusion.layer_solid_ids) == 1 and len(extrusion.core_solid_ids) == 1
    assert np.asarray(extrusion.slab_volumes) == pytest.approx(0.3)
    assert extrusion.control.route is phx.meshing.BoundaryLayerRoute.EXACT_SWEEP


def test_cad_extrusion_meshes_as_an_exact_sweep(
    prepared_cad_extrusion: tuple[Any, ...],
) -> None:
    provider, extrusion = prepared_cad_extrusion
    part = extrusion.source
    whole = provider.whole_scope(part, 3)
    layers = provider.entity_scope(part, part.solid_ids[extrusion.layer_solid_ids[0]])
    core = provider.entity_scope(part, part.solid_ids[extrusion.core_solid_ids[0]])
    native_control = extrusion.control
    assert native_control.volume_scope is not None
    assert native_control.cap_scope is not None
    # Native extrusion names definition rows; author comparison-provider
    # scopes from those entities rather than reinterpreting its numeric namespace.
    sweep = phx.meshing.BoundaryLayerControl(
        provider.entity_scope(
            part,
            tuple(
                part.face_ids[int(row)]
                for row in np.asarray(native_control.wall_scope.entity_ids)
            ),
        ),
        native_control.schedule,
        route=native_control.route,
        volume_scope=provider.entity_scope(
            part,
            tuple(
                part.solid_ids[int(row)]
                for row in np.asarray(native_control.volume_scope.entity_ids)
            ),
        ),
        cap_scope=provider.entity_scope(
            part,
            tuple(
                part.face_ids[int(row)]
                for row in np.asarray(native_control.cap_scope.entity_ids)
            ),
        ),
        collision=native_control.collision,
        corner=native_control.corner,
        feature_angle=native_control.feature_angle,
        minimum_thickness_fraction=native_control.minimum_thickness_fraction,
        growth_rate_bounds=native_control.growth_rate_bounds,
        maximum_corner_stretch=native_control.maximum_corner_stretch,
        smoothing_iterations=native_control.smoothing_iterations,
        core_maximum_size=native_control.core_maximum_size,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"), allow_mixed=True
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole, 0.24, minimum_size=0.04, maximum_size=0.5, maximum_growth_rate=10.0
            ),
        ),
        region_controls=(
            phx.meshing.RegionControl(
                layers, "layers", "air", phx.meshing.RegionRole.FLUID
            ),
            phx.meshing.RegionControl(
                core, "core", "air-core", phx.meshing.RegionRole.FLUID
            ),
        ),
        patch_controls=(
            phx.meshing.PatchControl(
                "cap",
                # ty: ignore[invalid-argument-type]
                sweep.cap_scope,
                ("core", "layers"),
            ),
        ),
        layer_controls=(sweep,),
    )

    result = provider.plan(part, specification).execute()

    prisms = result.mesh.block("prisms")
    heights = np.asarray(result.mesh.coordinates)[np.asarray(prisms.vertices)][:, :, 2]
    np.testing.assert_allclose(np.unique(np.round(heights, 12)), (0.0, 0.08, 0.18, 0.3))
    assert result.audit.passed and result.compliance.passed


@requires_meshcore
def test_core_fill_keeps_the_native_cap_bitwise_and_conforming() -> None:
    provider = phx.meshing.GmshProvider()
    t = (1.0 + 5.0**0.5) / 2.0
    base = np.asarray(
        (
            (-1, t, 0),
            (1, t, 0),
            (-1, -t, 0),
            (1, -t, 0),
            (0, -1, t),
            (0, 1, t),
            (0, -1, -t),
            (0, 1, -t),
            (t, 0, -1),
            (t, 0, 1),
            (-t, 0, -1),
            (-t, 0, 1),
        ),
        dtype=np.float64,
    )
    points = 0.4 * base / np.linalg.norm(base, axis=1, keepdims=True)
    faces = np.asarray(
        (
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        )
    )
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.sum(normals * corners.mean(axis=1), axis=1) < 0.0
    faces[inward] = faces[inward][:, ::-1]
    wall = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "triangles", "triangle", faces, global_ids=np.arange(20)
            ),
        ),
    )
    cells = wall.entity_set(2)
    control = phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope(
            wall.mesh_id,
            wall.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            2,
            cells.entity_set_id,
            cells.entity_ids,
        ),
        # ty: ignore[invalid-argument-type]
        phx.meshing.LayerSchedule((0.02, 0.03)),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    layers = phx.meshing.prepare_boundary_layers(wall, control)
    outer_points = np.asarray(
        [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    outer_faces = np.asarray(
        (
            (0, 1, 3),
            (0, 3, 2),
            (4, 6, 7),
            (4, 7, 5),
            (0, 4, 5),
            (0, 5, 1),
            (2, 3, 7),
            (2, 7, 6),
            (0, 2, 6),
            (0, 6, 4),
            (1, 5, 7),
            (1, 7, 3),
        )
    )
    outer = SurfaceModel.from_triangles(
        outer_points,
        outer_faces,
        SurfaceMetadata(
            source_id="outer-box",
            source_revision="0",
            coordinate_contract=_COORDINATES,
            provenance=("test",),
        ),
        repair_orientation=True,
    )

    result = provider.fill_boundary_layer_core(layers, outer, maximum_size=0.5)

    coordinates = np.asarray(result.mesh.coordinates)
    layer_points = np.asarray(layers.mesh.coordinates)
    np.testing.assert_array_equal(coordinates[: layer_points.shape[0]], layer_points)
    np.testing.assert_array_equal(
        coordinates[np.asarray(layers.cap_vertices)],
        # ty: ignore[unresolved-attribute]
        np.asarray(layers.cap.coordinates),
    )
    _assert_layered_organization(result)


def test_planar_provider_layers_lower_to_a_measured_gmsh_boundary_layer_field(
    tmp_path: Any,
) -> None:
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5))),
        ((0, 1, 2, 3),),
        feature_id="channel",
    )
    source = phx.geometry.partition_planar(
        phx.geometry.PlanarPartitionPlan(
            _COORDINATES,
            embedding,
            (
                phx.geometry.PlanarPartitionOperand(
                    "channel", region, phx.geometry.BRepPartitionRole.REGION
                ),
            ),
            phx.geometry.BRepPartitionPolicy(("channel",)),
        ),
        destination=tmp_path / "channel.brep",
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 2)
    schedule = phx.meshing.LayerSchedule.geometric(4, 0.01, growth_rate=1.2)
    control = phx.meshing.BoundaryLayerControl(
        provider.whole_scope(source, 1),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.PROVIDER,
        corner=phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2,
            2,
            phx.meshing.CellFamilyPolicy(
                required=("triangle", "quadrilateral"), allow_mixed=True
            ),
        ),
        scope,
        planar_embedding=embedding,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
        region_controls=tuple(
            phx.meshing.RegionControl(
                provider.entity_scope(source, value.entity_ids),
                value.name,
                "air",
                phx.meshing.RegionRole.FLUID,
            )
            for value in source.regions
        ),
        patch_controls=tuple(
            phx.meshing.PatchControl(
                value.name,
                provider.entity_scope(source, value.entity_ids),
                value.adjacent_region_ids,
            )
            for value in source.patches
        ),
        layer_controls=(control,),
    )

    result = provider.plan(source, specification).execute()

    achieved = dict(result.compliance.achieved)
    key = f"layer:{control.control_id}"
    assert achieved[f"{key}:layer_count"] == 4
    assert achieved[f"{key}:first_layer_thickness"] == pytest.approx(
        0.01, rel=_PLANAR_FIELD_RELATIVE_TOLERANCE
    )
    for index in range(1, 4):
        assert achieved[f"{key}:growth:{index}"] == pytest.approx(
            1.2, rel=_PLANAR_FIELD_RELATIVE_TOLERANCE
        )
    assert {block.cell_kind for block in result.mesh.blocks} == {
        "triangle",
        "quadrilateral",
    }
    assert result.compliance.passed


def test_boundary_association_rebinds_owned_faces_by_preserved_cell_ids() -> None:
    from phydrax.meshing._canonical import canonicalize_cell_mesh
    from phydrax.meshing.providers._gmsh_evidence import boundary_association

    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=_COORDINATES
    )
    bottom = source.qualified_face_ids[
        _faces(source, lambda points: np.isclose(points[:, :, 2], 0.0))[0]
    ]
    top = source.qualified_face_ids[
        _faces(source, lambda points: np.isclose(points[:, :, 2], 1.0))[0]
    ]
    # Equal facet coordinates cannot establish identity: the provider assigns
    # each cell its own face, including a sampled approximation residual of one.
    points = np.asarray(
        ((0.2, 0.2, 1.0), (0.8, 0.2, 1.0), (0.2, 0.8, 1.0)), dtype=np.float64
    )
    surface = SurfaceModel.from_triangles(
        np.concatenate((points, points)),
        np.asarray(((0, 1, 2), (3, 5, 4)), dtype=np.int32),
        SurfaceMetadata(
            source_id=source.source_id,
            source_revision=source.source_revision,
            coordinate_contract=_COORDINATES,
            provenance=("test",),
        ),
        cell_global_ids=np.asarray((91, 17), dtype=np.int64),
        repair_orientation=True,
        orient_closed_outward=False,
    )
    boundary = canonicalize_cell_mesh(surface.mesh)
    association, _, attribute = boundary_association(
        source, boundary, {91: bottom, 17: top}
    )

    np.testing.assert_array_equal(association.target_global_ids, (17, 91))
    np.testing.assert_array_equal(association.source_indices, (top.index, bottom.index))
    np.testing.assert_allclose(
        association.residuals, (0.0, 1.0), atol=_CAD_RESIDUAL_ABSOLUTE_TOLERANCE
    )
    np.testing.assert_array_equal(attribute.values, (top.index, bottom.index))
    assert association.complete and not association.exact
    assert association.provenance is phx.meshing.GeometryAssociationProvenance.PROVIDER


@pytest.mark.parametrize("owner_ids", ((17,), (17, 91, 100)), ids=("missing", "extra"))
def test_boundary_association_refuses_incomplete_cell_ownership(
    owner_ids: tuple[int, ...],
) -> None:
    from phydrax.meshing.providers._gmsh_evidence import boundary_association

    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=_COORDINATES
    )
    boundary = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
        cell_global_ids=np.asarray((91,), dtype=np.int64),
    )
    with pytest.raises(
        phx.meshing.MeshingFailure, match="bind every boundary cell ID exactly once"
    ) as failure:
        boundary_association(
            source,
            boundary,
            {identifier: source.qualified_face_ids[0] for identifier in owner_ids},
        )
    assert failure.value.category is phx.meshing.MeshingFailureCategory.ASSOCIATION_FAILED
