from importlib.util import find_spec

import build123d as bd
import numpy as np
import pytest

import phydrax as phx


pytestmark = [
    pytest.mark.meshing_gmsh,
    pytest.mark.skipif(
        find_spec("gmsh") is None, reason="optional gmsh package is not installed"
    ),
]


def _source(path, shape=None):
    bd.export_step(
        bd.Box(1.0, 1.0, 1.0) if shape is None else shape, path, unit=bd.Unit.MM
    )
    model = phx.geometry.import_brep(
        path,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    return phx.geometry.BRepSource(model) if shape is None else model


@pytest.mark.parametrize("geometry_order", (1, 2))
def test_real_gmsh_volume_result_is_audited_associated_and_solver_ready(
    tmp_path,
    geometry_order,
):
    if find_spec("gmsh") is None:
        pytest.skip("optional gmsh package is not installed")
    source = _source(tmp_path / "cube.step")
    provider = phx.meshing.GmshProvider(
        phx.meshing.GmshOptions(
            coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER)
        )
    )
    scope = provider.whole_scope(source, 3)
    size = phx.meshing.UniformSizeControl(
        scope,
        0.3,
        maximum_growth_rate=2.0,
    )
    target = phx.meshing.CellMeshingTarget(
        3,
        3,
        phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        geometry_order=geometry_order,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        target,
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(size,),
    )

    result = provider.plan(source, specification).execute()

    assert result.audit.passed
    assert result.compliance.passed
    assert result.trace.successful
    assert result.boundary is not None
    assert result.associations[0].complete
    assert result.geometry.elements[0].degree == geometry_order
    assert result.mesh.blocks[0].cell_kind == "tetrahedron"
    assert result.coordinate_contract.length_unit == phx.units.MILLIMETER
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {"tetrahedra": phx.discretization.lagrange_element("tetrahedron", 1)},
    )
    prepared = phx.discretization.FiniteElementPlan(result.mesh, (field,)).prepare()
    assert prepared.mesh.mesh_id == result.mesh.mesh_id


def _provider():
    return phx.meshing.GmshProvider(
        phx.meshing.GmshOptions(
            coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        )
    )


def _scope(source, dimension, identifiers):
    return phx.meshing.MeshingScope(
        source.report.source_id,
        source.report.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        f"{source.report.source_revision}:brep:{dimension}",
        np.asarray(identifiers, dtype=np.int64),
    )


def _face_scope(source, axis, coordinate):
    points = np.asarray(source.model.mesh_vertices)
    triangles = points[np.asarray(source.model.mesh_faces)]
    face_ids = np.asarray(source.model.triangle_face_ids)
    selected = np.unique(
        face_ids[np.all(np.isclose(triangles[:, :, axis], coordinate), axis=1)]
    )
    assert selected.size == 1
    return _scope(source, 2, selected)


def _edge_scope(source, axis, coordinate):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS

    from phydrax.geometry.brep._occt import _explore_unique, read_occt_shape

    shape, _, _ = read_occt_shape(source.report.source_id)
    selected = []
    for index, edge in enumerate(_explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)):
        curve = BRepAdaptor_Curve(edge)
        values = [
            curve.Value(float(value))
            for value in np.linspace(curve.FirstParameter(), curve.LastParameter(), 3)
        ]
        points = np.asarray([(point.X(), point.Y(), point.Z()) for point in values])
        if np.allclose(points[:, axis], coordinate):
            selected.append(index)
    assert len(selected) == 1
    return _scope(source, 1, selected)


def _specification(
    provider, source, dimension, policy, *, order=2, layers=(), periodic=()
):
    scope = provider.whole_scope(source, dimension)
    target = phx.meshing.CellMeshingTarget(dimension, 3, policy, geometry_order=order)
    size = phx.meshing.UniformSizeControl(
        scope, 0.25, minimum_size=0.05, maximum_size=0.5, maximum_growth_rate=10.0
    )
    if dimension == 2:
        return phx.meshing.SurfaceMeshingSpec(
            target, scope, size_controls=(size,), periodic_constraints=periodic
        )
    return phx.meshing.VolumeMeshingSpec(
        target,
        scope,
        phx.meshing.VolumeFillStrategy.SWEEP
        if layers
        else phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(size,),
        layer_controls=layers,
        periodic_constraints=periodic,
    )


def _assert_curved_geometry(result):
    elements, routes, coordinates = result.geometry.resolve(result.mesh)
    for element, route in zip(elements, routes, strict=True):
        # Evaluate canonical maps, independently of Gmsh's node ordering and audit.
        center = np.mean(np.asarray(element.reference_nodes), axis=0, keepdims=True)
        _, gradients = element.tabulate(center)
        jacobians = np.einsum(
            "cna,qnd->cqad",
            np.asarray(coordinates)[np.asarray(route)],
            np.asarray(gradients),
        )
        if element.topological_dimension == 3:
            assert np.all(np.linalg.det(jacobians) > 0.0)
        else:
            assert np.all(
                np.linalg.norm(np.cross(jacobians[..., 0], jacobians[..., 1]), axis=-1)
                > 0.0
            )
    assert dict(result.compliance.achieved)["minimum_curved_jacobian_determinant"] > 0.0


@pytest.mark.parametrize("dimension", (2, 3))
def test_real_periodic_planar_and_volume_meshes_match_all_quadratic_nodes(
    tmp_path, dimension
):
    provider = _provider()
    source = _source(
        tmp_path / "periodic.step", bd.Rectangle(1, 1) if dimension == 2 else None
    )
    selector = _edge_scope if dimension == 2 else _face_scope
    transform = np.eye(4)
    transform[0, 3] = 1.0
    constraint = phx.meshing.PeriodicConstraint(
        selector(source, 0, -0.5),
        selector(source, 0, 0.5),
        transform,
        tolerance=1.0e-9,
    )
    kind = "triangle" if dimension == 2 else "tetrahedron"
    spec = _specification(
        provider,
        source,
        dimension,
        phx.meshing.CellFamilyPolicy(required=(kind,)),
        periodic=(constraint,),
    )

    result = provider.plan(source, spec).execute()

    assert result.compliance.passed and result.audit.passed
    coordinates = np.asarray(result.geometry.coordinates)
    left = coordinates[np.isclose(coordinates[:, 0], -0.5)]
    right = coordinates[np.isclose(coordinates[:, 0], 0.5)]
    mapped = left + transform[:3, 3]
    distances = np.linalg.norm(mapped[:, None] - right[None], axis=-1)
    assert left.shape == right.shape
    assert np.all(np.sum(distances <= constraint.tolerance, axis=1) == 1)
    achieved = dict(result.compliance.achieved)
    assert (
        achieved[f"periodic:{constraint.constraint_id}:maximum_residual"]
        <= constraint.tolerance
    )
    assert achieved[f"periodic:{constraint.constraint_id}:node_pairs"] == left.shape[0]
    assert result.associations[0].complete
    _assert_curved_geometry(result)


@pytest.mark.parametrize("mixed", (False, True))
def test_real_planar_quadrilateral_and_mixed_output_keeps_quadratic_maps(tmp_path, mixed):
    provider = _provider()
    # An odd boundary subdivision admits mixed recombination; the rectangle
    # can legitimately become all-quadrilateral even with partial recombination.
    shape = (
        bd.Polygon((0.0, 0.0), (0.6, 0.0), (0.3, np.sqrt(3.0) * 0.3))
        if mixed
        else bd.Rectangle(1.0, 0.8)
    )
    source = _source(tmp_path / "surface.step", shape)
    policy = (
        phx.meshing.CellFamilyPolicy(
            required=("triangle", "quadrilateral"), allow_mixed=True
        )
        if mixed
        else phx.meshing.CellFamilyPolicy(required=("quadrilateral",))
    )
    result = provider.plan(source, _specification(provider, source, 2, policy)).execute()

    assert {block.cell_kind for block in result.mesh.blocks} == set(policy.required)
    assert (
        result.associations[0].target_entity_set_id
        == result.mesh.entity_set(2).entity_set_id
    )
    assert result.associations[0].complete
    assert result.boundary is not None
    assert result.audit.passed and result.compliance.passed
    _assert_curved_geometry(result)


@pytest.mark.parametrize("kind", ("prism", "hexahedron"))
def test_real_thin_region_sweep_has_exact_layer_interfaces_and_high_order_geometry(
    tmp_path, kind
):
    provider = _provider()
    source = _source(tmp_path / "sweep.step")
    control = phx.meshing.ThinRegionLayerControl(
        _face_scope(source, 2, -0.5),
        _face_scope(source, 2, 0.5),
        provider.whole_scope(source, 3),
        4,
    )
    result = provider.plan(
        source,
        _specification(
            provider,
            source,
            3,
            phx.meshing.CellFamilyPolicy(required=(kind,)),
            layers=(control,),
        ),
    ).execute()

    assert {block.cell_kind for block in result.mesh.blocks} == {kind}
    assert np.unique(
        np.round(np.asarray(result.mesh.coordinates)[:, 2], 10)
    ) == pytest.approx(np.linspace(-0.5, 0.5, 5))
    assert dict(result.compliance.achieved)["layer_count"] == 4
    assert result.associations[0].complete
    assert result.audit.passed and result.compliance.passed
    _assert_curved_geometry(result)


def test_real_prism_layers_realize_geometric_thickness_schedule(tmp_path):
    provider = _provider()
    source = _source(tmp_path / "prism.step")
    first = 1.0 / (1.0 + 1.2 + 1.2**2 + 1.2**3)
    control = phx.meshing.PrismLayerControl(
        _face_scope(source, 2, -0.5),
        provider.whole_scope(source, 3),
        4,
        first,
        growth_rate=1.2,
    )
    result = provider.plan(
        source,
        _specification(
            provider,
            source,
            3,
            phx.meshing.CellFamilyPolicy(required=("prism",)),
            layers=(control,),
        ),
    ).execute()

    expected = -0.5 + np.concatenate(([0.0], np.cumsum(first * 1.2 ** np.arange(4))))
    assert np.unique(
        np.round(np.asarray(result.mesh.coordinates)[:, 2], 10)
    ) == pytest.approx(expected)
    achieved = dict(result.compliance.achieved)
    assert achieved["first_layer_thickness"] == pytest.approx(first)
    assert achieved["layer_growth_rate"] == pytest.approx(1.2)
    assert achieved["layer_count"] == 4
    _assert_curved_geometry(result)


def test_real_partial_prism_layer_contract_is_rejected_not_reported_as_compliant(
    tmp_path,
):
    provider = _provider()
    source = _source(tmp_path / "partial.step")
    control = phx.meshing.PrismLayerControl(
        _face_scope(source, 2, -0.5),
        provider.whole_scope(source, 3),
        2,
        0.05,
        growth_rate=1.0,
    )
    plan = provider.plan(
        source,
        _specification(
            provider,
            source,
            3,
            phx.meshing.CellFamilyPolicy(required=("prism",)),
            layers=(control,),
        ),
    )
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        plan.execute()
    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
    with provider.open_session() as session:
        assert not session.closed


def test_real_curved_tetrahedron_audit_detects_inversion_with_valid_corners():
    import gmsh

    from phydrax.meshing.providers._gmsh import _audit_jacobians, _element_rows

    with _provider().open_session():
        gmsh.model.add("curved-inversion")
        entity = gmsh.model.addDiscreteEntity(3)
        element_type = gmsh.model.mesh.getElementType("Tetrahedron", 2)
        _, _, _, count, reference, _ = gmsh.model.mesh.getElementProperties(element_type)
        points = np.asarray(reference).reshape((-1, 3)).copy()
        points[4] = (0.5, 0.0, 2.0)
        tags = np.arange(1, count + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(3, entity, tags, points.reshape(-1))
        gmsh.model.mesh.addElementsByType(entity, element_type, [1], tags)
        assert np.linalg.det((points[1:4] - points[0]).T) > 0.0
        with pytest.raises(phx.meshing.MeshingFailure) as failure:
            _audit_jacobians(gmsh, _element_rows(gmsh, 3, 2))
        assert failure.value.category is phx.meshing.MeshingFailureCategory.AUDIT_FAILED


def test_real_gmsh_session_releases_global_ownership_after_body_failure():
    import gmsh

    provider = _provider()
    with pytest.raises(RuntimeError, match="body failure"):
        with provider.open_session() as session:
            gmsh.model.add("failed-session")
            raise RuntimeError("body failure")
    assert session.closed
    assert not gmsh.isInitialized()
    with provider.open_session() as reopened:
        gmsh.model.add("reopened-session")
        assert gmsh.model.getCurrent() == "reopened-session"
        assert not reopened.closed


def test_open_cad_model_is_rejected_for_volume_meshing_without_weakening_solid_source(
    tmp_path,
):
    provider = _provider()
    model = _source(tmp_path / "open-sheet.step", bd.Rectangle(1, 1))
    specification = _specification(
        provider,
        model,
        3,
        phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(model, specification)

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
