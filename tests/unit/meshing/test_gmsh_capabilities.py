from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest

import phydrax as phx


pytestmark = [
    pytest.mark.meshing_gmsh,
    pytest.mark.skipif(
        find_spec("gmsh") is None, reason="optional gmsh package is not installed"
    ),
]

_CONTRACT = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
_IMPORT_POLICY = phx.interchange.CadImportPolicy(
    _CONTRACT,
    phx.interchange.ResourceLimits(
        max_bytes=1 << 24,
        max_depth=64,
        max_nodes=100_000,
        max_attributes=2_000_000,
        max_losses=8,
    ),
    tessellation=phx.geometry.BRepTessellationPolicy(
        linear_deflection=0.01, angular_deflection=0.2
    ),
)


def _face(points: Any) -> Any:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.gp import gp_Pnt

    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in points:
        polygon.Add(gp_Pnt(float(x), float(y), 0.0))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def _persist(shape: Any, path: Any, *, overwrite: Any = False) -> Any:
    # OCCT constructs the external comparison fixture, never the native source model.
    phx.interchange.persist_occt_shape(
        shape,
        path,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.01,
        angular_deflection=0.2,
        overwrite=overwrite,
    )
    return phx.interchange.read_cad(
        path,
        _IMPORT_POLICY,
        trusted_root=path.parent,
        source_length_unit=_CONTRACT.length_unit,
    ).model


def _surface_spec(provider: Any, source: Any, size: Any, **kwargs: Any) -> Any:
    scope = provider.whole_scope(source, 2)
    return phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, size),),
        **kwargs,
    )


def _volume_spec(
    provider: Any, source: Any, size: Any, *, order: Any = 1, deterministic: Any = True
) -> Any:
    scope = provider.whole_scope(source, 3)
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=order,
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(scope, size),),
        deterministic=deterministic,
    )


def _mesh_edges(result: Any) -> Any:
    points = np.asarray(result.mesh.coordinates)
    edges = np.asarray(result.mesh.connectivity.edges)
    return points[edges[:, 0]], points[edges[:, 1]]


def _achieved(result: Any, prefix: Any) -> Any:
    return {
        key.rsplit(":", 1)[-1]: value
        for key, value in result.compliance.achieved
        if key.startswith(prefix)
    }


def _square_metric_control(hx: Any, hy: Any, mode: Any) -> Any:
    corners = np.array(
        [[-0.01, -0.01, 0.0], [1.01, -0.01, 0.0], [1.01, 1.01, 0.0], [-0.01, 1.01, 0.0]]
    )
    background = phx.discretization.CellMesh(
        corners,
        (
            phx.discretization.CellBlock(
                "triangles", "triangle", np.array([[0, 1, 2], [0, 2, 3]])
            ),
        ),
    )
    vertices = background.entity_set(0)
    metric = phx.meshing.MeshMetricField(
        phx.meshing.MeshingScope(
            background.mesh_id,
            background.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            vertices.entity_ids,
        ),
        np.broadcast_to(np.diag([1.0 / hx**2, 1.0 / hy**2, 1.0 / hx**2]), (4, 3, 3)),
        minimum_size=min(hx, hy),
        maximum_size=max(hx, hy),
        maximum_anisotropy=max(hx, hy) / min(hx, hy),
    )
    return phx.meshing.BackgroundMetricControl(
        background, metric, _CONTRACT, mode=mode, maximum_metric_edge_length=2.0
    )


def test_real_anisotropic_background_metric_stretches_cells_along_the_metric(
    tmp_path: Any,
) -> None:
    square = _persist(_face(((0, 0), (1, 0), (1, 1), (0, 1))), tmp_path / "square.brep")
    provider = phx.meshing.GmshProvider(
        phx.meshing.GmshOptions(algorithm_2d=phx.meshing.GmshSurfaceAlgorithm.BAMG)
    )
    control = _square_metric_control(
        0.2, 0.04, phx.meshing.BackgroundMetricMode.ANISOTROPIC
    )

    result = provider.plan(
        square, _surface_spec(provider, square, 0.5, background_metric=control)
    ).execute()

    first, second = _mesh_edges(result)
    extent = np.abs(second - first)
    metric = _achieved(result, "background_metric:")
    assert result.audit.passed and result.compliance.passed
    assert 3.0 < np.mean(extent[:, 0]) / np.mean(extent[:, 1]) < 7.0
    assert 0.75 < metric["mean_edge_length"] < 1.35
    assert metric["maximum_edge_length"] <= 2.0
    assert metric["unit_edge_fraction"] > 0.9


def test_gmsh_preflight_routes_anisotropic_volume_metrics_away(tmp_path: Any) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    cube = phx.geometry.BRepSource(
        _persist(BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(), tmp_path / "cube.brep")
    )
    provider = phx.meshing.GmshProvider()
    control = _square_metric_control(
        0.2, 0.04, phx.meshing.BackgroundMetricMode.ANISOTROPIC
    )

    report = provider.validate(
        cube, _volume_spec(provider, cube, 0.5), background_metric=control
    )

    assert not report.supported
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(cube, _volume_spec(provider, cube, 0.5), background_metric=control)
    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )


def _channel_walls(provider: Any, source: Any) -> Any:
    geometry = source.geometry
    assert geometry is not None
    walls = {0.45: [], 0.55: []}
    for index, curve_index in enumerate(geometry.edge_curves):
        if curve_index < 0:
            points = np.asarray(geometry.vertex_points)[
                list(geometry.edge_vertices[index])
            ]
        else:
            interval = np.asarray(geometry.edge_ranges[index], dtype=np.float64)
            parameters = np.linspace(interval[0], interval[1], 3)
            points = np.asarray(geometry.curves[curve_index].evaluate(parameters))
        for height, selected in walls.items():
            if np.allclose(points[:, 1], height, atol=1.0e-9, rtol=0.0):
                selected.append(source.edge_ids[index])
    assert all(len(selected) == 1 for selected in walls.values())
    return tuple(provider.entity_scope(source, walls[height]) for height in (0.45, 0.55))


def test_real_proximity_control_refines_a_narrow_channel(tmp_path: Any) -> None:
    outline = (
        (0, 0),
        (1, 0),
        (1, 0.45),
        (2, 0.45),
        (2, 0),
        (3, 0),
        (3, 1),
        (2, 1),
        (2, 0.55),
        (1, 0.55),
        (1, 1),
        (0, 1),
    )
    source = _persist(_face(outline), tmp_path / "dumbbell.brep")
    provider = phx.meshing.GmshProvider()
    lower, upper = _channel_walls(provider, source)
    tolerance = phx.meshing.SizeCompliancePolicy(relative_tolerance=0.5)

    def channel_edges(result: Any) -> Any:
        first, second = _mesh_edges(result)
        midpoint = 0.5 * (first + second)
        inside = (midpoint[:, 0] > 1.1) & (midpoint[:, 0] < 1.9)
        return np.linalg.norm(second - first, axis=1)[inside]

    coarse = provider.plan(
        source, _surface_spec(provider, source, 0.3, size_compliance=tolerance)
    ).execute()
    specification = _surface_spec(provider, source, 0.3, size_compliance=tolerance)
    refined_specification = phx.meshing.SurfaceMeshingSpec(
        specification.target,
        specification.scope,
        size_controls=(
            *specification.size_controls,
            phx.meshing.ProximitySizeControl(lower, upper, 4),
        ),
        size_compliance=tolerance,
    )
    refined = provider.plan(source, refined_specification).execute()

    evidence = _achieved(refined, "proximity:")
    assert refined.compliance.passed
    assert evidence["gap"] == pytest.approx(0.1)
    assert np.max(channel_edges(refined)) <= 0.1 / 4 * 1.5
    assert np.max(channel_edges(coarse)) > 0.1 / 4 * 1.5
    assert evidence["maximum_edge_ratio"] <= 1.5


def test_real_protected_free_curve_is_embedded_as_mesh_edges(tmp_path: Any) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, _face(((0, 0), (1, 0), (1, 1), (0, 1))))
    start, stop = np.array([0.2, 0.3, 0.0]), np.array([0.8, 0.65, 0.0])
    builder.Add(compound, BRepBuilderAPI_MakeEdge(gp_Pnt(*start), gp_Pnt(*stop)).Edge())
    source = _persist(compound, tmp_path / "embedded.brep")
    provider = phx.meshing.GmshProvider()
    free = next(
        index for index, owners in enumerate(source.topology.edge_faces) if not owners
    )
    feature = phx.meshing.ProtectedFeature(
        provider.entity_scope(source, source.edge_ids[free]),
        phx.meshing.FeatureKind.CURVE,
    )

    result = provider.plan(
        source, _surface_spec(provider, source, 0.2, protected_features=(feature,))
    ).execute()

    first, second = _mesh_edges(result)
    direction = stop - start

    def on_curve(points: Any) -> Any:
        parameter = (points - start) @ direction / (direction @ direction)
        residual = np.linalg.norm(start + parameter[:, None] * direction - points, axis=1)
        return (parameter > -1.0e-12) & (parameter < 1.0 + 1.0e-12) & (residual < 1.0e-9)

    covered = on_curve(first) & on_curve(second)
    evidence = _achieved(result, f"protected:{feature.feature_id}:")
    assert result.compliance.passed
    assert np.sum(np.linalg.norm(second - first, axis=1)[covered]) == pytest.approx(
        np.linalg.norm(direction)
    )
    assert evidence["embedded_entity_count"] == 1.0
    assert evidence["maximum_deviation"] <= 1.0e-12


def _stl_cube(path: Any) -> Any:
    corners = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    lines = ["solid cube"]
    for a, b, c, d in quads:
        for triangle in ((a, b, c), (a, c, d)):
            points = corners[list(triangle)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            lines.append("facet normal {} {} {}".format(*normal))
            lines.append("outer loop")
            lines.extend("vertex {} {} {}".format(*point) for point in points)
            lines.extend(("endloop", "endfacet"))
    lines.append("endsolid cube")
    path.write_text("\n".join(lines))
    return phx.geometry.import_surface(
        path, phx.geometry.SurfaceImportPolicy(phx.units.METER, allow_lossy=True)
    ).model


def test_real_discrete_stl_remesh_is_audited_and_preserves_topology(
    tmp_path: Any,
) -> None:
    source = _stl_cube(tmp_path / "cube.stl")
    provider = phx.meshing.GmshProvider()
    surface = _surface_spec(provider, source, 0.2)
    source_mesh = source.mesh
    assert source_mesh is not None
    specification = phx.meshing.SurfaceRemeshingSpec(surface, source_mesh.mesh_id)
    reconstruction = phx.meshing.SurfaceReconstructionControl(np.deg2rad(40.0), 1.0e-6)

    result = provider.plan(source, specification, reconstruction=reconstruction).execute()

    achieved = dict(result.compliance.achieved)
    boundary = result.boundary
    assert boundary is not None
    boundary_audit = boundary.audit()
    assert result.audit.passed and result.compliance.passed
    assert boundary_audit.valid and boundary_audit.closed
    assert result.mesh.blocks[0].cell_count > 12 * 10
    assert achieved["surface_euler_characteristic"] == 2.0
    assert achieved["surface_maximum_remesh_to_source_distance"] <= 1.0e-6
    assert achieved["surface_maximum_source_to_remesh_distance"] <= 1.0e-6
    assert result.associations[0].complete
    with pytest.raises(TypeError):
        provider.validate(source_mesh, specification, reconstruction=reconstruction)


def _tetrahedron_rule(count: Any) -> Any:
    """Collapsed Gauss rule on the reference tetrahedron (Duffy transform)."""
    nodes, weights = np.polynomial.legendre.leggauss(count)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    u, v, w = (
        axis.reshape(-1) for axis in np.meshgrid(nodes, nodes, nodes, indexing="ij")
    )
    weight = np.prod(
        np.stack(np.meshgrid(weights, weights, weights, indexing="ij")), axis=0
    ).reshape(-1)
    points = np.stack((u, v * (1.0 - u), w * (1.0 - u) * (1.0 - v)), axis=1)
    return points, weight * (1.0 - u) ** 2 * (1.0 - v)


@pytest.mark.parametrize("order", (2, 3))
def test_real_curved_cylinder_is_certified_by_native_gmsh_quality(
    tmp_path: Any, order: Any
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

    source = phx.geometry.BRepSource(
        _persist(BRepPrimAPI_MakeCylinder(1.0, 1.0).Shape(), tmp_path / "cylinder.brep")
    )
    provider = phx.meshing.GmshProvider(
        phx.meshing.GmshOptions(
            high_order_optimization=phx.meshing.GmshHighOrderOptimization.FAST_CURVING
        )
    )

    result = provider.plan(
        source, _volume_spec(provider, source, 0.6, order=order)
    ).execute()

    achieved = dict(result.compliance.achieved)
    elements, routes, coordinates = result.geometry.resolve(result.mesh)
    element = elements[0]
    assert isinstance(element, phx.discretization.FiniteElementSpec)
    nodes = np.asarray(coordinates)[np.asarray(routes[0])]
    points, weights = _tetrahedron_rule(6)
    _, gradients = element.tabulate(points)
    jacobians = np.matmul(np.swapaxes(nodes, 1, 2)[:, None], np.asarray(gradients)[None])
    curved_volume = float(np.sum(np.linalg.det(jacobians) @ weights))
    corners = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    straight_volume = float(
        np.sum(np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))) / 6.0
    )
    reference_vertices = np.vstack((np.zeros((1, 3)), np.eye(3)))
    values, _ = element.tabulate(reference_vertices)
    assert result.audit.passed and result.compliance.passed
    assert element.degree == order
    assert achieved["gmsh_minimum_scaled_inverse_condition"] > 0.0
    assert achieved["gmsh_minimum_scaled_inverse_gradient_error"] > 0.0
    assert achieved["minimum_curved_jacobian_determinant"] > 0.0
    # The curved map interpolates the canonical corners and recovers the CAD volume.
    np.testing.assert_allclose(np.asarray(values) @ nodes, corners, atol=1.0e-12)
    assert abs(curved_volume - np.pi) < 0.1 * abs(straight_volume - np.pi)


def test_real_session_import_cache_is_bound_to_immutable_native_revisions(
    tmp_path: Any,
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    path = tmp_path / "cube.brep"
    source = phx.geometry.BRepSource(
        _persist(BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(), path)
    )
    provider = phx.meshing.GmshProvider()

    with provider.open_session() as session:
        first = session.execute(
            provider.plan(source, _volume_spec(provider, source, 0.5))
        )
        second = session.execute(
            provider.plan(source, _volume_spec(provider, source, 0.4))
        )
        assert (session.import_cache_misses, session.import_cache_hits) == (1, 1)
        assert session.cached_source_revisions == (source.report.source_revision,)

        replaced = phx.geometry.BRepSource(
            _persist(
                BRepPrimAPI_MakeBox(2.0, 1.0, 1.0).Shape(),
                path,
                overwrite=True,
            )
        )
        assert replaced.report.source_revision != source.report.source_revision
        third = session.execute(
            provider.plan(replaced, _volume_spec(provider, replaced, 0.5))
        )
        assert (session.import_cache_misses, session.import_cache_hits) == (2, 1)
        assert session.cached_source_revisions == tuple(
            sorted((source.report.source_revision, replaced.report.source_revision))
        )

        fourth = session.execute(
            provider.plan(source, _volume_spec(provider, source, 0.5))
        )
        assert (session.import_cache_misses, session.import_cache_hits) == (2, 2)

    assert all(result.compliance.passed for result in (first, second, third, fourth))
    assert third.mesh.mesh_id != first.mesh.mesh_id
    assert fourth.mesh.mesh_id == first.mesh.mesh_id


def test_real_hxt_threads_follow_the_explicit_resource_policy(tmp_path: Any) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    source = phx.geometry.BRepSource(
        _persist(BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(), tmp_path / "box.brep")
    )
    provider = phx.meshing.GmshProvider(
        phx.meshing.GmshOptions(
            algorithm_3d=phx.meshing.GmshVolumeAlgorithm.HXT,
            num_threads=2,
            optimize_netgen=True,
        )
    )

    assert not provider.validate(source, _volume_spec(provider, source, 0.3)).supported
    plan = provider.plan(source, _volume_spec(provider, source, 0.3, deterministic=False))
    with provider.open_session() as session:
        with pytest.raises(phx.meshing.MeshingFailure) as failure:
            session.execute(plan)
    result = plan.execute()

    assert failure.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert result.compliance.passed and not result.runtime.deterministic
    assert phx.meshing.MeshingStageKind.OPTIMIZATION in {
        stage.stage for stage in result.trace.stages
    }
