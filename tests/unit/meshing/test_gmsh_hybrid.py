from importlib.util import find_spec

import numpy as np
import pytest

import phydrax as phx


pytestmark = [
    pytest.mark.meshing_gmsh,
    pytest.mark.skipif(
        find_spec("gmsh") is None, reason="optional gmsh package is not installed"
    ),
]


def _split_boxes(first_origin, first_size, second_origin, second_size):
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    first = BRepPrimAPI_MakeBox(gp_Pnt(*first_origin), *first_size).Shape()
    second = BRepPrimAPI_MakeBox(gp_Pnt(*second_origin), *second_size).Shape()
    splitter = BOPAlgo_Splitter()
    splitter.AddArgument(first)
    splitter.AddArgument(second)
    splitter.Perform()
    if splitter.HasErrors():
        raise RuntimeError("OCCT failed to build the conformal hybrid fixture.")
    return splitter.Shape()


def _source(path, shape):
    return phx.geometry.persist_occt_shape(
        shape,
        path,
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        linear_deflection=0.03,
        angular_deflection=0.15,
    )


def _face_indices(source, axis, coordinate):
    points = np.asarray(source.mesh_vertices)
    triangles = points[np.asarray(source.mesh_faces)]
    face_ids = np.asarray(source.triangle_face_ids)
    return tuple(
        int(value)
        for value in np.unique(
            face_ids[np.all(np.isclose(triangles[:, :, axis], coordinate), axis=1)]
        )
    )


def _scope(provider, source, dimension, indices):
    entities = {2: source.face_ids, 3: source.solid_ids}[dimension]
    return provider.entity_scope(source, tuple(entities[index] for index in indices))


def _slab_fixture(path):
    shape = _split_boxes(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.3),
        (0.0, 0.0, 0.3),
        (1.0, 1.0, 0.7),
    )
    source = _source(path, shape)
    source_face = _face_indices(source, 2, 0.0)
    target_face = _face_indices(source, 2, 0.3)
    assert len(source_face) == len(target_face) == 1
    swept_solid = source.topology.face_solids[source_face[0]]
    assert len(swept_solid) == 1
    swept = swept_solid[0]
    core = next(index for index in range(source.topology.num_solids) if index != swept)
    return source, source_face[0], target_face[0], swept, core


def _hybrid_specification(
    provider,
    source,
    source_face,
    target_face,
    swept,
    core,
    schedule,
    *,
    geometry_order=1,
    policy=None,
    patch_face=None,
):
    whole = provider.whole_scope(source, 3)
    swept_scope = _scope(provider, source, 3, (swept,))
    core_scope = _scope(provider, source, 3, (core,))
    control = phx.meshing.SweptLayerControl(
        _scope(provider, source, 2, (source_face,)),
        _scope(provider, source, 2, (target_face,)),
        swept_scope,
        schedule,
    )
    regions = (
        phx.meshing.RegionControl(
            swept_scope,
            "swept-slab",
            "swept-material",
            phx.meshing.RegionRole.SOLID,
        ),
        phx.meshing.RegionControl(
            core_scope,
            "tetra-core",
            "core-material",
            phx.meshing.RegionRole.SOLID,
        ),
    )
    families = (
        phx.meshing.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        if policy is None
        else policy
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            families,
            geometry_order=geometry_order,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole,
                0.24,
                minimum_size=0.04,
                maximum_size=0.5,
                maximum_growth_rate=10.0,
            ),
        ),
        region_controls=regions,
        patch_controls=(
            phx.meshing.PatchControl(
                "slab-core",
                _scope(
                    provider,
                    source,
                    2,
                    (target_face if patch_face is None else patch_face,),
                ),
                ("swept-slab", "tetra-core"),
            ),
        ),
        layer_controls=(control,),
    )
    return specification, control


def _polyhedral_faces(connectivity):
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int32)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    return tuple(
        values[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )


def test_real_full_width_prism_slab_meets_tetra_core_on_exact_triangles(tmp_path):
    provider = phx.meshing.GmshProvider()
    source, source_face, target_face, swept, core = _slab_fixture(
        tmp_path / "hybrid-slab.brep"
    )
    schedule = phx.meshing.LayerSchedule((0.08, 0.10, 0.12))
    specification, control = _hybrid_specification(
        provider,
        source,
        source_face,
        target_face,
        swept,
        core,
        schedule,
    )

    result = provider.plan(source, specification).execute()

    assert [block.cell_kind for block in result.mesh.blocks] == [
        "prism",
        "tetrahedron",
    ]
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, phx.discretization.PolyhedralConnectivity)
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in result.mesh.blocks)
    )
    np.testing.assert_array_equal(cell_ids, np.arange(cell_ids.size))
    prism_block = result.mesh.block("prisms")
    prism_points = np.asarray(result.mesh.coordinates)[np.asarray(prism_block.vertices)]
    np.testing.assert_allclose(
        np.unique(np.round(prism_points[:, :, 2], 12)),
        np.asarray((0.0, 0.08, 0.18, 0.3)),
    )

    face_rows = _polyhedral_faces(connectivity)
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids)
    face_by_id = {int(identifier): row for row, identifier in enumerate(face_ids)}
    interface_rows = tuple(
        face_by_id[int(identifier)]
        for identifier in np.asarray(result.patches[0].scope.entity_ids)
    )
    assert interface_rows
    assert all(len(face_rows[row]) == 3 for row in interface_rows)
    cell_kinds = np.concatenate(
        tuple(
            np.full((block.cell_count,), block.cell_kind, dtype=object)
            for block in result.mesh.blocks
        )
    )
    owners = np.asarray(connectivity.face_owner)
    neighbours = np.asarray(connectivity.face_neighbour)
    assert all(
        {str(cell_kinds[owners[row]]), str(cell_kinds[neighbours[row]])}
        == {"prism", "tetrahedron"}
        for row in interface_rows
    )

    region_zones = tuple(
        zone for zone in result.zones if zone.role is phx.meshing.MeshZoneRole.REGION
    )
    assert {zone.name for zone in region_zones} == {"swept-slab", "tetra-core"}
    assert all(
        association.complete and association.exact for association in result.associations
    )
    attributes = {attribute.name: attribute for attribute in result.attributes}
    assert set(np.asarray(attributes["layer_index"].values)) == {0, 1, 2}
    assert set(np.asarray(attributes["layer_control_index"].values)) == {0}
    achieved = dict(result.compliance.achieved)
    key = f"layer:{control.control_id}"
    assert achieved[f"{key}:thickness:0"] == pytest.approx(0.08)
    assert achieved[f"{key}:thickness:1"] == pytest.approx(0.10)
    assert achieved[f"{key}:thickness:2"] == pytest.approx(0.12)
    assert achieved[f"{key}:growth:1"] == pytest.approx(1.25)
    assert achieved[f"{key}:maximum_alignment_residual"] <= 1.0e-12
    assert achieved[f"{key}:maximum_interface_residual"] <= 1.0e-12
    assert achieved["layer_interface_compliance"] == 1.0
    assert result.audit.passed and result.compliance.passed

    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", 1),
            "tetrahedra": phx.discretization.lagrange_element("tetrahedron", 1),
        },
    )
    prepared = phx.discretization.FiniteElementPlan(
        result.mesh,
        (field,),
        coordinate_spec=result.geometry,
    ).prepare()
    assert prepared.mesh.mesh_id == result.mesh.mesh_id


def test_swept_lateral_face_adjoining_unswept_volume_is_rejected_before_generation(
    tmp_path,
):
    provider = phx.meshing.GmshProvider()
    source = _source(
        tmp_path / "invalid-side.brep",
        _split_boxes(
            (0.0, 0.0, 0.0),
            (0.3, 1.0, 1.0),
            (0.3, 0.0, 0.0),
            (0.7, 1.0, 1.0),
        ),
    )
    swept_face = _face_indices(source, 0, 0.0)
    assert len(swept_face) == 1
    swept = source.topology.face_solids[swept_face[0]][0]
    core = next(index for index in range(source.topology.num_solids) if index != swept)
    source_cap = next(
        face
        for face in _face_indices(source, 2, 0.0)
        if swept in source.topology.face_solids[face]
    )
    target_cap = next(
        face
        for face in _face_indices(source, 2, 1.0)
        if swept in source.topology.face_solids[face]
    )
    shared_face = _face_indices(source, 0, 0.3)
    assert len(shared_face) == 1
    specification, _ = _hybrid_specification(
        provider,
        source,
        source_cap,
        target_cap,
        swept,
        core,
        phx.meshing.LayerSchedule((0.25, 0.25, 0.25, 0.25)),
        patch_face=shared_face[0],
    )

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(source, specification)

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )


def test_schedule_total_mismatch_fails_before_gmsh_mesh_generation(tmp_path, monkeypatch):
    import gmsh

    provider = phx.meshing.GmshProvider()
    source, source_face, target_face, swept, core = _slab_fixture(
        tmp_path / "mismatched-schedule.brep"
    )
    specification, _ = _hybrid_specification(
        provider,
        source,
        source_face,
        target_face,
        swept,
        core,
        phx.meshing.LayerSchedule((0.08, 0.10, 0.11)),
    )

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("mesh generation must not run for a mismatched schedule")

    monkeypatch.setattr(gmsh.model.mesh, "generate", unexpected_generation)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(source, specification).execute()

    assert (
        failure.value.category is phx.meshing.MeshingFailureCategory.INVALID_SPECIFICATION
    )


@pytest.mark.parametrize(
    ("geometry_order", "policy"),
    (
        (
            2,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"), allow_mixed=True
            ),
        ),
        (
            1,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron", "pyramid"), allow_mixed=True
            ),
        ),
    ),
)
def test_hybrid_sweep_rejects_high_order_and_transition_cell_claims(
    tmp_path,
    geometry_order,
    policy,
):
    provider = phx.meshing.GmshProvider()
    source, source_face, target_face, swept, core = _slab_fixture(
        tmp_path / f"unsupported-{geometry_order}-{len(policy.required)}.brep"
    )
    specification, _ = _hybrid_specification(
        provider,
        source,
        source_face,
        target_face,
        swept,
        core,
        phx.meshing.LayerSchedule((0.1, 0.1, 0.1)),
        geometry_order=geometry_order,
        policy=policy,
    )

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        provider.plan(source, specification)

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
