from importlib.util import find_spec

import build123d as bd
import numpy as np
import pytest
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

import phydrax as phx


pytestmark = [
    pytest.mark.meshing_gmsh,
    pytest.mark.skipif(
        find_spec("gmsh") is None, reason="optional gmsh package is not installed"
    ),
]


def _planar_face(points):
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in points:
        polygon.Add(gp_Pnt(float(x), float(y), 0.0))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def _source(path, shape=None):
    persisted = path.with_suffix(".brep")
    model = phx.geometry.persist_occt_shape(
        BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape() if shape is None else shape,
        persisted,
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
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
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 3)
    size = phx.meshing.UniformSizeControl(
        scope,
        0.3,
        maximum_growth_rate=None,
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
    assert result.coordinate_contract.spatial_id == source.coordinate_contract.spatial_id
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {"tetrahedra": phx.discretization.lagrange_element("tetrahedron", 1)},
    )
    prepared = phx.discretization.FiniteElementPlan(result.mesh, (field,)).prepare()
    assert prepared.mesh.mesh_id == result.mesh.mesh_id


def _provider():
    return phx.meshing.GmshProvider()


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
        tmp_path / "periodic.step",
        bd.Rectangle(1, 1).wrapped if dimension == 2 else None,
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
        _planar_face(((0.0, 0.0), (0.6, 0.0), (0.3, np.sqrt(3.0) * 0.3)))
        if mixed
        else _planar_face(((0.0, 0.0), (1.0, 0.0), (1.0, 0.8), (0.0, 0.8)))
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


def test_real_whole_volume_sweep_has_exact_prism_schedule(tmp_path):
    provider = _provider()
    source = _source(tmp_path / "sweep.step")
    first = 1.0 / (1.0 + 1.2 + 1.2**2 + 1.2**3)
    schedule = phx.meshing.LayerSchedule.geometric(
        4,
        first,
        growth_rate=1.2,
    )
    control = phx.meshing.SweptLayerControl(
        _face_scope(source, 2, -0.5),
        _face_scope(source, 2, 0.5),
        provider.whole_scope(source, 3),
        schedule,
    )
    result = provider.plan(
        source,
        _specification(
            provider,
            source,
            3,
            phx.meshing.CellFamilyPolicy(required=("prism",)),
            order=1,
            layers=(control,),
        ),
    ).execute()

    expected = -0.5 + np.concatenate(([0.0], np.cumsum(schedule.thicknesses)))
    np.testing.assert_allclose(
        np.unique(np.round(np.asarray(result.mesh.coordinates)[:, 2], 10)),
        expected,
    )
    assert {block.cell_kind for block in result.mesh.blocks} == {"prism"}
    assert isinstance(
        result.mesh.connectivity,
        phx.discretization.PolyhedralConnectivity,
    )
    attributes = {attribute.name: attribute for attribute in result.attributes}
    assert set(np.asarray(attributes["layer_index"].values)) == {0, 1, 2, 3}
    assert set(np.asarray(attributes["layer_control_index"].values)) == {0}
    achieved = dict(result.compliance.achieved)
    assert achieved["first_layer_thickness"] == pytest.approx(first)
    assert achieved["layer_growth_rate"] == pytest.approx(1.2)
    assert achieved["layer_count"] == 4
    assert result.associations[0].complete
    assert result.audit.passed and result.compliance.passed
    _assert_curved_geometry(result)


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
    model = _source(
        tmp_path / "open-sheet.brep",
        _planar_face(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
    )
    scope = _scope(model, 3, (0,))
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                0.25,
                minimum_size=0.05,
                maximum_size=0.5,
                maximum_growth_rate=10.0,
            ),
        ),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(model, specification)

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )


def test_gmsh_preflight_rejects_unlowered_local_curvature_and_proximity(tmp_path):
    source = _source(tmp_path / "local-sizing.step")
    provider = _provider()
    whole = provider.whole_scope(source, 2)
    first = provider.entity_scope(source, source.model.face_ids[0])
    second = provider.entity_scope(source, source.model.face_ids[1])
    target = phx.meshing.CellMeshingTarget(
        2,
        3,
        phx.meshing.CellFamilyPolicy(required=("triangle",)),
    )
    curvature = phx.meshing.SurfaceMeshingSpec(
        target,
        whole,
        size_controls=(phx.meshing.CurvatureSizeControl(first, np.pi / 12.0),),
    )
    proximity = phx.meshing.SurfaceMeshingSpec(
        target,
        whole,
        size_controls=(phx.meshing.ProximitySizeControl(first, second, 3),),
    )

    for specification in (curvature, proximity):
        with pytest.raises(phx.meshing.MeshingFailure) as failure:
            provider.plan(source, specification)
        assert (
            failure.value.category
            is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
        )


def test_real_gmsh_hard_size_compliance_has_no_factor_two_allowance(tmp_path):
    source = _source(tmp_path / "hard-size.step")
    provider = _provider()
    scope = provider.whole_scope(source, 3)
    control = phx.meshing.UniformSizeControl(
        scope,
        1.2,
        minimum_size=1.1,
        maximum_size=2.0,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(control,),
        size_compliance=phx.meshing.SizeCompliancePolicy(absolute_tolerance=0.05),
    )

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(source, specification).execute()

    assert failure.value.category is phx.meshing.MeshingFailureCategory.COMPLIANCE_FAILED


def _two_box_shape(*, conformal):
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    left = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
    right = BRepPrimAPI_MakeBox(gp_Pnt(1.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
    if conformal:
        splitter = BOPAlgo_Splitter()
        splitter.AddArgument(left)
        splitter.AddArgument(right)
        splitter.Perform()
        if splitter.HasErrors():
            raise RuntimeError("OCCT failed to build the conformal partition fixture.")
        return splitter.Shape()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, left)
    builder.Add(compound, right)
    return compound


def _semantic_source(path, *, conformal=True):
    model = phx.geometry.persist_occt_shape(
        _two_box_shape(conformal=conformal),
        path,
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    assert model.report.num_solids == 2
    return model


def _semantic_specification(
    provider,
    source,
    *,
    order=1,
    size_controls=None,
    interface_required=True,
):
    whole = provider.whole_scope(source, 3)
    left_scope = provider.entity_scope(source, source.solid_ids[0])
    right_scope = provider.entity_scope(source, source.solid_ids[1])
    controls = (
        (
            phx.meshing.UniformSizeControl(
                whole,
                0.28,
                minimum_size=0.06,
                maximum_size=0.55,
                maximum_growth_rate=10.0,
            ),
        )
        if size_controls is None
        else size_controls
    )
    regions = (
        phx.meshing.RegionControl(
            left_scope,
            "left",
            "material-left",
            phx.meshing.RegionRole.SOLID,
        ),
        phx.meshing.RegionControl(
            right_scope,
            "right",
            "material-right",
            phx.meshing.RegionRole.SOLID,
        ),
    )
    interface_faces = tuple(
        source.face_ids[index]
        for index, owners in enumerate(source.topology.face_solids)
        if len(owners) == 2
    )
    if not interface_faces and interface_required:
        raise ValueError("The semantic fixture requires one shared interface face.")
    patch_controls = (
        ()
        if not interface_faces
        else (
            phx.meshing.PatchControl(
                "left-right",
                provider.entity_scope(source, interface_faces),
                ("left", "right"),
                required=interface_required,
            ),
        )
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=order,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=tuple(controls),
        region_controls=regions,
        patch_controls=patch_controls,
    )
    return specification, regions


@pytest.mark.parametrize("geometry_order", (1, 2))
def test_real_semantic_multi_region_mesh_preserves_ownership_and_interface(
    tmp_path, geometry_order
):
    provider = _provider()
    source = _semantic_source(tmp_path / f"partition-{geometry_order}.brep")
    specification, regions = _semantic_specification(
        provider, source, order=geometry_order
    )

    result = provider.plan(source, specification).execute()

    region_zones = tuple(
        zone for zone in result.zones if zone.role is phx.meshing.MeshZoneRole.REGION
    )
    assert {zone.name for zone in region_zones} == {"left", "right"}
    assert {zone.material_id for zone in region_zones} == {
        control.material_id for control in regions
    }
    assert {zone.region_role for zone in region_zones} == {phx.meshing.RegionRole.SOLID}
    np.testing.assert_array_equal(
        np.sort(
            np.concatenate(
                tuple(np.asarray(zone.scope.entity_ids) for zone in region_zones)
            )
        ),
        np.sort(np.asarray(result.mesh.entity_set(3).entity_ids)),
    )
    assert len(result.patches) == 1
    patch = result.patches[0]
    assert patch.name == "left-right"
    assert patch.scope.entity_set_id == result.mesh.entity_set(2).entity_set_id
    assert set(patch.adjacent_zone_ids) == {zone.zone_id for zone in region_zones}
    assert result.boundary is not None
    assert set(np.asarray(patch.scope.entity_ids)).isdisjoint(
        set(np.asarray(result.boundary.mesh.entity_set(2).entity_ids))
    )
    assert {association.target_entity_set_id for association in result.associations} == {
        result.mesh.entity_set(2).entity_set_id,
        result.mesh.entity_set(3).entity_set_id,
    }
    assert all(
        association.complete and association.exact for association in result.associations
    )
    assert result.audit.passed and result.compliance.passed
    if geometry_order == 2:
        _assert_curved_geometry(result)


def test_real_persisted_semantic_source_replays_with_stable_identity(tmp_path):
    provider = _provider()
    path = tmp_path / "replay-partition.brep"
    source = _semantic_source(path)
    specification, _ = _semantic_specification(provider, source)
    plan = provider.plan(source, specification)

    first = plan.execute()
    second = plan.execute()

    assert first.result_id == second.result_id
    assert first.mesh.mesh_id == second.mesh.mesh_id
    assert first.provenance.semantic_id == second.provenance.semantic_id

    phx.geometry.persist_occt_shape(
        bd.Box(3.0, 1.0, 1.0).wrapped,
        path,
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        overwrite=True,
    )
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        plan.execute()
    assert failure.value.category is phx.meshing.MeshingFailureCategory.INVALID_SOURCE


def test_semantic_region_ownership_conflict_is_rejected_by_contract(tmp_path):
    provider = _provider()
    source = _semantic_source(tmp_path / "ownership-conflict.brep")
    whole = provider.whole_scope(source, 3)
    left = provider.entity_scope(source, source.solid_ids[0])
    size = phx.meshing.UniformSizeControl(
        whole,
        0.3,
        minimum_size=0.05,
        maximum_size=0.6,
        maximum_growth_rate=10.0,
    )
    regions = (
        phx.meshing.RegionControl(
            left, "first", "first-material", phx.meshing.RegionRole.SOLID
        ),
        phx.meshing.RegionControl(
            left, "second", "second-material", phx.meshing.RegionRole.SOLID
        ),
    )

    with pytest.raises(ValueError, match="Region control scopes must be disjoint"):
        phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            whole,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(size,),
            region_controls=regions,
        )


def test_real_nonconformal_touching_solids_are_rejected_before_generation(tmp_path):
    provider = _provider()
    source = _semantic_source(
        tmp_path / "ambiguous-touching-partition.brep", conformal=False
    )
    specification, _ = _semantic_specification(provider, source, interface_required=False)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(source, specification).execute()

    assert failure.value.category is phx.meshing.MeshingFailureCategory.INVALID_SOURCE


def test_real_solid_scoped_uniform_size_refines_only_selected_region(tmp_path):
    provider = _provider()
    source = _semantic_source(tmp_path / "scoped-size-partition.brep")
    whole = provider.whole_scope(source, 3)
    right = provider.entity_scope(source, source.solid_ids[1])
    baseline = phx.meshing.UniformSizeControl(
        whole,
        0.38,
        minimum_size=0.06,
        maximum_size=0.65,
        maximum_growth_rate=10.0,
        strength=phx.meshing.SizeControlStrength.SOFT,
    )
    refinement = phx.meshing.UniformSizeControl(
        right,
        0.14,
        minimum_size=0.04,
        maximum_size=0.32,
        maximum_growth_rate=10.0,
    )
    specification, _ = _semantic_specification(
        provider,
        source,
        size_controls=(baseline, refinement),
    )

    result = provider.plan(source, specification).execute()
    achieved = dict(result.compliance.achieved)

    assert achieved["size_field_count"] == 2.0
    assert (
        achieved[f"size:{refinement.control_id}:maximum_edge"]
        < achieved[f"size:{baseline.control_id}:maximum_edge"]
    )
    assert result.compliance.passed
