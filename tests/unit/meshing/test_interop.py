import meshio
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_ordering import (
    MESHIO_CELL_TYPES,
    meshio_reference_nodes,
    reference_node_permutation,
)
from phydrax.interchange._mesh_arrays import (
    MeshArrayArtifact,
    MeshArrayAssociation,
    MeshArrayBlock,
    MeshArrayField,
)
from phydrax.meshing._interop import export_mesh_array_artifact


def _mesh():
    return phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
        np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
        vertex_global_ids=np.asarray((30, 10, 40, 20), dtype=np.int64),
        cell_global_ids=np.asarray((200, 100), dtype=np.int64),
    )


def _scope(mesh, dimension, ids=None):
    entity_set = mesh.entity_set(dimension)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        dimension,
        entity_set.entity_set_id,
        entity_set.entity_ids if ids is None else ids,
    )


def _policy(*, allow_lossy=False):
    return phx.meshing.MeshInteropPolicy(
        phx.SpatialCoordinateContract.si(), allow_lossy=allow_lossy
    )


def test_vtu_roundtrip_preserves_ids_and_attributes_by_entity_identity(tmp_path):
    mesh = _mesh()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    attributes = (
        phx.meshing.MeshAttribute(
            "point_value",
            phx.meshing.MeshAttributeRole.USER,
            _scope(mesh, 0),
            np.asarray((1, 2, 3, 4)),
        ),
        phx.meshing.MeshAttribute(
            "material",
            phx.meshing.MeshAttributeRole.MATERIAL,
            _scope(mesh, 2),
            np.asarray((7, 9)),
        ),
    )
    policy = _policy(allow_lossy=True)
    exported = phx.meshing.export_cell_mesh(
        tmp_path / "mesh.vtu", mesh, geometry, policy, attributes=attributes
    )
    imported = phx.meshing.import_cell_mesh(exported.path, policy)
    assert exported.report.valid and imported.report.valid
    np.testing.assert_array_equal(imported.mesh.vertex_global_ids, mesh.vertex_global_ids)
    np.testing.assert_array_equal(
        imported.mesh.blocks[0].global_ids, mesh.blocks[0].global_ids
    )
    np.testing.assert_allclose(imported.geometry.coordinates[:, :2], mesh.coordinates)
    for original, restored in zip(attributes, imported.attributes, strict=True):
        assert restored.name == original.name
        np.testing.assert_array_equal(
            restored.scope.entity_ids, original.scope.entity_ids
        )
        np.testing.assert_array_equal(restored.values, original.values)
    losses = {loss.path for loss in exported.report.losses}
    assert {"attribute.material.role", "coordinate_contract", "block_names"} <= losses
    assert {"point_value", "material"} <= set(exported.report.preserved_fields)


def test_external_metadata_loss_requires_permission_before_writing(tmp_path):
    mesh = _mesh()
    path = tmp_path / "mesh.vtu"
    with pytest.raises(ValueError, match="coordinate_contract"):
        phx.meshing.export_cell_mesh(
            path, mesh, phx.discretization.CellGeometrySpec.affine(mesh), _policy()
        )
    assert not path.exists()


def test_native_roundtrip_preserves_organization_units_and_lower_entity_ids():
    original = _mesh()
    mesh = phx.discretization.CellMesh(
        original.coordinates,
        original.blocks,
        vertex_global_ids=original.vertex_global_ids,
        entity_global_ids={1: np.asarray((901, 905, 902, 904, 903))},
        numeric_version="revision-7",
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    attribute = phx.meshing.MeshAttribute(
        "edge_length",
        phx.meshing.MeshAttributeRole.GEOMETRY_CLASSIFICATION,
        _scope(mesh, 1, np.asarray((901, 904))),
        np.asarray((1.5, 2.5)),
        unit=phx.units.METER,
    )
    zone = phx.meshing.MeshZone(
        "boundary",
        phx.meshing.MeshZoneRole.BOUNDARY,
        _scope(mesh, 1, np.asarray((901, 904))),
    )
    label = phx.meshing.MeshLabel("selected", _scope(mesh, 2, np.asarray((200,))))
    policy = phx.meshing.MeshInteropPolicy(
        phx.SpatialCoordinateContract(phx.units.METER, reference_frame="lab")
    )
    artifact, report = export_mesh_array_artifact(
        mesh, geometry, policy, attributes=(attribute,), zones=(zone,), labels=(label,)
    )
    restored = phx.meshing.import_cell_mesh(artifact, policy)
    assert report.status == phx.interchange.AdapterStatus.LOSSLESS
    assert restored.report.status == phx.interchange.AdapterStatus.LOSSLESS
    assert restored.mesh.mesh_id == mesh.mesh_id
    assert restored.mesh.numeric_version == "revision-7"
    assert restored.mesh.blocks[0].name == mesh.blocks[0].name
    assert (
        restored.artifact.coordinate_contract.spatial_id
        == policy.coordinate_contract.spatial_id
    )
    assert restored.attributes[0].attribute_id == attribute.attribute_id
    assert restored.zones[0].zone_id == zone.zone_id
    assert restored.labels[0].label_id == label.label_id
    for dimension in range(3):
        np.testing.assert_array_equal(
            restored.mesh.entity_set(dimension).entity_ids,
            mesh.entity_set(dimension).entity_ids,
        )


@pytest.mark.parametrize("cell_type", tuple(MESHIO_CELL_TYPES))
def test_native_roundtrip_preserves_shared_reference_ordering_and_geometry_ids(cell_type):
    kind, order, corners = MESHIO_CELL_TYPES[cell_type]
    points = meshio_reference_nodes(cell_type)
    vertex_ids = np.arange(corners, dtype=np.int64)[::-1] + 100
    mesh = phx.discretization.CellMesh(
        points[:corners],
        (
            phx.discretization.CellBlock(
                "named-block",
                kind,
                np.arange(corners, dtype=np.int32)[None, :],
                global_ids=np.asarray((500,)),
            ),
        ),
        vertex_global_ids=vertex_ids,
    )
    element = phx.discretization.lagrange_element(kind, order)
    permutation = reference_node_permutation(cell_type, element.reference_nodes)
    curved_points = points.copy()
    if order > 1:
        curved_points[corners:, 0] += 0.03125
    geometry = phx.discretization.CellGeometrySpec(
        {"named-block": element}, {"named-block": permutation[None, :]}, curved_points
    )
    point_ids = np.concatenate(
        (vertex_ids, np.arange(points.shape[0] - corners, dtype=np.int64) + 1000)
    )
    artifact, report = export_mesh_array_artifact(
        mesh, geometry, _policy(), point_global_ids=point_ids
    )
    restored = phx.meshing.import_cell_mesh(artifact, _policy())
    assert report.status == phx.interchange.AdapterStatus.LOSSLESS
    np.testing.assert_array_equal(
        artifact.blocks[0].connectivity, np.arange(points.shape[0])[None, :]
    )
    np.testing.assert_array_equal(restored.artifact.point_global_ids, point_ids)
    np.testing.assert_array_equal(
        restored.geometry.geometry_dofs[0], permutation[None, :]
    )
    np.testing.assert_array_equal(restored.geometry.coordinates, curved_points)
    np.testing.assert_array_equal(restored.mesh.vertex_global_ids, vertex_ids)


def test_native_reordered_geometry_keeps_vertex_ids_and_separate_vertex_coordinates():
    mesh = _mesh()
    permutation = np.asarray((2, 0, 3, 1))
    inverse = np.argsort(permutation)
    geometry = phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: phx.discretization.lagrange_element("triangle", 1)},
        {mesh.blocks[0].name: inverse[np.asarray(mesh.blocks[0].vertices)]},
        np.asarray(mesh.coordinates)[permutation] + 0.5,
    )
    artifact, _ = export_mesh_array_artifact(mesh, geometry, _policy())
    restored = phx.meshing.import_cell_mesh(artifact, _policy())
    np.testing.assert_array_equal(
        artifact.point_global_ids, np.asarray(mesh.vertex_global_ids)[permutation]
    )
    np.testing.assert_array_equal(restored.mesh.vertex_global_ids, mesh.vertex_global_ids)
    np.testing.assert_array_equal(restored.mesh.coordinates, mesh.coordinates)
    np.testing.assert_array_equal(restored.geometry.coordinates, geometry.coordinates)


def test_native_coordinate_contract_mismatch_is_rejected():
    mesh = _mesh()
    artifact, _ = export_mesh_array_artifact(
        mesh, phx.discretization.CellGeometrySpec.affine(mesh), _policy()
    )
    different = phx.meshing.MeshInteropPolicy(
        phx.SpatialCoordinateContract(phx.units.METER, reference_frame="other")
    )
    with pytest.raises(ValueError, match="coordinate contract"):
        phx.meshing.import_cell_mesh(artifact, different)


def test_export_rejects_stale_attribute_binding_and_reserved_identity_name(tmp_path):
    mesh = _mesh()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    valid_scope = _scope(mesh, 0)
    foreign_scope = phx.meshing.MeshingScope(
        "foreign",
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        valid_scope.entity_set_id,
        valid_scope.entity_ids,
    )
    foreign = phx.meshing.MeshAttribute(
        "field", phx.meshing.MeshAttributeRole.USER, foreign_scope, np.ones(4)
    )
    with pytest.raises(ValueError, match="binding"):
        export_mesh_array_artifact(mesh, geometry, _policy(), attributes=(foreign,))
    reserved = phx.meshing.MeshAttribute(
        "phydrax_point_global_ids",
        phx.meshing.MeshAttributeRole.USER,
        valid_scope,
        np.ones(4),
    )
    with pytest.raises(ValueError, match="reserved"):
        phx.meshing.export_cell_mesh(
            tmp_path / "reserved.vtu",
            mesh,
            geometry,
            _policy(allow_lossy=True),
            attributes=(reserved,),
        )


def test_partial_attribute_is_preserved_natively_and_reported_externally(tmp_path):
    mesh = _mesh()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    attribute = phx.meshing.MeshAttribute(
        "selection",
        phx.meshing.MeshAttributeRole.USER,
        _scope(mesh, 0, np.asarray((10, 30))),
        np.asarray((7, 8)),
    )
    artifact, _ = export_mesh_array_artifact(
        mesh, geometry, _policy(), attributes=(attribute,)
    )
    restored = phx.meshing.import_cell_mesh(artifact, _policy())
    np.testing.assert_array_equal(restored.attributes[0].scope.entity_ids, (10, 30))
    np.testing.assert_array_equal(restored.attributes[0].values, (7, 8))
    exported = phx.meshing.export_cell_mesh(
        tmp_path / "partial.vtu",
        mesh,
        geometry,
        _policy(allow_lossy=True),
        attributes=(attribute,),
    )
    assert "attribute.selection" in {loss.path for loss in exported.report.losses}
    assert "selection" not in exported.report.preserved_fields
    assert "selection" not in meshio.read(exported.path).point_data


def test_external_codec_field_loss_is_reported(tmp_path):
    mesh = _mesh()
    exported = phx.meshing.export_cell_mesh(
        tmp_path / "mesh.stl",
        mesh,
        phx.discretization.CellGeometrySpec.affine(mesh),
        _policy(allow_lossy=True),
    )
    losses = {loss.path for loss in exported.report.losses}
    assert "point_data.phydrax_point_global_ids" in losses
    assert "cell_data.phydrax_cell_global_ids" in losses


def test_import_gmsh_named_regions_require_explicit_loss_permission(tmp_path):
    path = tmp_path / "regions.msh"
    meshio.write(
        path,
        meshio.Mesh(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            [("triangle", np.asarray(((0, 1, 2),)))],
            field_data={"wall": np.asarray((7, 2))},
            cell_data={
                "gmsh:physical": [np.asarray((7,))],
                "gmsh:geometrical": [np.asarray((1,))],
            },
        ),
        file_format="gmsh22",
    )
    with pytest.raises(ValueError, match="field_data"):
        phx.meshing.read_mesh_array_artifact(path, _policy())
    artifact, report = phx.meshing.read_mesh_array_artifact(
        path, _policy(allow_lossy=True)
    )
    assert "field_data" in {loss.path for loss in report.losses}
    assert artifact.fields[0].association == MeshArrayAssociation.CELL


def test_high_order_point_field_projection_is_not_silently_lossless(tmp_path):
    path = tmp_path / "high-order.vtu"
    points = meshio_reference_nodes("triangle6")
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("triangle6", np.arange(6)[None, :])],
            point_data={"temperature": np.arange(6, dtype=float)},
        ),
    )
    with pytest.raises(ValueError, match="nonvertex_values"):
        phx.meshing.import_cell_mesh(path, _policy())
    restored = phx.meshing.import_cell_mesh(path, _policy(allow_lossy=True))
    np.testing.assert_array_equal(restored.artifact.fields[0].values, np.arange(6))
    assert "field.temperature.nonvertex_values" in {
        loss.path for loss in restored.report.losses
    }


def test_import_orders_mixed_surface_blocks_without_changing_cell_fields(tmp_path):
    path = tmp_path / "mixed.vtu"
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0)))
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [
                ("quad", np.asarray(((0, 1, 2, 3),))),
                ("triangle", np.asarray(((1, 4, 2),))),
            ],
            cell_data={
                "value": [np.asarray((7,)), np.asarray((9,))],
                "phydrax_cell_global_ids": [np.asarray((80,)), np.asarray((20,))],
            },
        ),
    )
    restored = phx.meshing.import_cell_mesh(path, _policy())
    assert [block.cell_kind for block in restored.mesh.blocks] == [
        "triangle",
        "quadrilateral",
    ]
    assert {
        int(attribute.scope.entity_ids[0]): int(attribute.values[0])
        for attribute in restored.attributes
    } == {80: 7, 20: 9}


def _artifact(blocks, *, fields=()):
    return MeshArrayArtifact(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
        np.arange(4),
        blocks,
        phx.SpatialCoordinateContract.si(),
        source_id="test",
        source_format="native",
        fields=fields,
    )


def test_artifact_rejects_ambiguous_block_and_field_identity():
    first = MeshArrayBlock(
        "same", "triangle", "triangle", 1, np.asarray(((0, 1, 2),)), np.asarray((10,))
    )
    duplicate_name = MeshArrayBlock(
        "same", "triangle", "triangle", 1, np.asarray(((1, 3, 2),)), np.asarray((20,))
    )
    duplicate_id = MeshArrayBlock(
        "other", "triangle", "triangle", 1, np.asarray(((1, 3, 2),)), np.asarray((10,))
    )
    with pytest.raises(ValueError, match="block names"):
        _artifact((first, duplicate_name))
    with pytest.raises(ValueError, match="IDs.*across blocks"):
        _artifact((first, duplicate_id))
    undeclared = MeshArrayField(
        "a", MeshArrayAssociation.CELL, np.asarray((7,)), block_name="missing"
    )
    with pytest.raises(ValueError, match="undeclared block"):
        _artifact((first,), fields=(undeclared,))
    repeated = MeshArrayField(
        "a", MeshArrayAssociation.CELL, np.asarray((7,)), block_name="same"
    )
    with pytest.raises(ValueError, match="field names"):
        _artifact((first,), fields=(repeated, repeated))


@pytest.mark.parametrize(
    "invalid", [np.asarray(((0.0, 1.5, 2.0),)), np.asarray(((False, True, True),))]
)
def test_artifact_rejects_noninteger_connectivity_without_truncation(invalid):
    with pytest.raises(TypeError, match="integers"):
        MeshArrayBlock("a", "triangle", "triangle", 1, invalid, np.asarray((10,)))


def test_artifact_owns_arrays_and_rejects_malformed_reference_width():
    connectivity = np.asarray(((0, 1, 2),))
    ids = np.asarray((10,))
    values = np.asarray((7.0,))
    block = MeshArrayBlock("a", "triangle", "triangle", 1, connectivity, ids)
    field = MeshArrayField("material", MeshArrayAssociation.CELL, values, block_name="a")
    artifact = _artifact((block,), fields=(field,))
    fingerprint = artifact.artifact_id
    connectivity[:] = 0
    ids[:] = 99
    values[:] = 9
    np.testing.assert_array_equal(artifact.blocks[0].connectivity, ((0, 1, 2),))
    np.testing.assert_array_equal(artifact.blocks[0].global_ids, (10,))
    np.testing.assert_array_equal(artifact.fields[0].values, (7,))
    assert artifact.artifact_id == fingerprint
    for array in (
        artifact.points,
        artifact.point_global_ids,
        block.connectivity,
        block.global_ids,
        field.values,
    ):
        assert not array.flags.writeable
    malformed = MeshArrayBlock(
        "bad", "triangle", "triangle6", 2, np.asarray(((0, 1, 2),)), np.asarray((1,))
    )
    with pytest.raises(ValueError, match="connectivity width"):
        phx.meshing.import_cell_mesh(_artifact((malformed,)), _policy())


def test_disjoint_same_name_cell_attributes_survive_native_and_file_roundtrip(tmp_path):
    mesh = _mesh()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    attributes = tuple(
        phx.meshing.MeshAttribute(
            "region",
            phx.meshing.MeshAttributeRole.USER,
            _scope(mesh, 2, np.asarray((identifier,))),
            np.asarray((value,)),
        )
        for identifier, value in ((200, 5), (100, 9))
    )
    artifact, _ = export_mesh_array_artifact(
        mesh, geometry, _policy(), attributes=attributes
    )
    native = phx.meshing.import_cell_mesh(artifact, _policy())
    assert [attribute.attribute_id for attribute in native.attributes] == [
        attribute.attribute_id for attribute in attributes
    ]
    exported = phx.meshing.export_cell_mesh(
        tmp_path / "regions.vtu",
        mesh,
        geometry,
        _policy(allow_lossy=True),
        attributes=attributes,
    )
    restored = phx.meshing.import_cell_mesh(exported.path, _policy())
    np.testing.assert_array_equal(restored.attributes[0].scope.entity_ids, (100, 200))
    np.testing.assert_array_equal(restored.attributes[0].values, (9, 5))


def test_tensor_attributes_preserve_native_shape_and_declare_file_loss(tmp_path):
    mesh = _mesh()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    attribute = phx.meshing.MeshAttribute(
        "tensor", phx.meshing.MeshAttributeRole.USER, _scope(mesh, 2), values
    )
    artifact, _ = export_mesh_array_artifact(
        mesh, geometry, _policy(), attributes=(attribute,)
    )
    restored = phx.meshing.import_cell_mesh(artifact, _policy())
    np.testing.assert_array_equal(restored.attributes[0].values, values)
    exported = phx.meshing.export_cell_mesh(
        tmp_path / "tensor.vtu",
        mesh,
        geometry,
        _policy(allow_lossy=True),
        attributes=(attribute,),
    )
    assert "attribute.tensor" in {loss.path for loss in exported.report.losses}
    assert "tensor" not in exported.report.preserved_fields


def test_generated_geometry_ids_keep_unsorted_vertex_ids_without_collision():
    points = meshio_reference_nodes("triangle6")
    mesh = phx.discretization.CellMesh.from_triangles(
        points[:3], np.asarray(((0, 1, 2),)), vertex_global_ids=np.asarray((5, 0, 2))
    )
    element = phx.discretization.lagrange_element("triangle", 2)
    block = mesh.blocks[0]
    geometry = phx.discretization.CellGeometrySpec(
        {block.name: element},
        {
            block.name: reference_node_permutation("triangle6", element.reference_nodes)[
                None, :
            ]
        },
        points,
    )
    artifact, report = export_mesh_array_artifact(mesh, geometry, _policy())
    np.testing.assert_array_equal(artifact.point_global_ids[:3], (5, 0, 2))
    assert np.unique(artifact.point_global_ids).size == 6
    assert "high_order_point_global_ids" in {loss.path for loss in report.losses}
