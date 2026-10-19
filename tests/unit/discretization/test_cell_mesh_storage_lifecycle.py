#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax.discretization import (
    CellBlock,
    CellMesh,
    CellMeshStorage,
    PeriodicCell,
    PeriodicMeshTopology,
    reference_cell_topology,
)
from tests._support.cell_meshes import unit_square


@pytest.fixture
def stored_mesh() -> CellMesh:
    source = unit_square(
        vertex_global_ids=np.asarray((31, 11, 41, 21), dtype=np.int64),
        cell_global_ids=np.asarray((73, 23), dtype=np.int64),
    )
    entities = source.topology.entity_sets
    storage = CellMeshStorage(
        tuple(entity.count for entity in entities),
        tuple(entity.entity_ids for entity in entities),
        tuple(np.zeros(entity.count, dtype=np.int32) for entity in entities),
        partition_index=0,
        partition_count=1,
        logical_topology_id=source.topology_id,
        logical_geometry_id=source.geometry_id,
        evidence_id=source.mesh_id,
        logical_arrays=(("coordinates", source.coordinates),),
        local_coordinates=source.coordinates,
        local_blocks=source.blocks,
    )
    return CellMesh(source.coordinates, source.blocks, storage=storage)


def test_corner_storage_reconstruction_preserves_coordinate_and_entity_identity(
    stored_mesh: CellMesh,
) -> None:
    storage = stored_mesh.storage
    assert storage is not None
    restored = CellMeshStorage.from_stored_fields(storage)
    assert restored.storage_id == storage.storage_id
    assert restored.local_geometry_source_id == storage.local_geometry_source_id
    assert restored.local_coordinate_id == storage.local_coordinate_id
    assert restored.geometry_source_blocks == storage.geometry_source_blocks
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(storage)
    np.testing.assert_array_equal(
        restored.coordinate_global_ids, storage.coordinate_global_ids
    )
    np.testing.assert_array_equal(restored.coordinate_owner, storage.coordinate_owner)
    for degree, ids in enumerate(storage.entity_global_ids):
        np.testing.assert_array_equal(restored.entity_global_ids[degree], ids)
        np.testing.assert_array_equal(
            restored.entity_owned[degree], storage.entity_owned[degree]
        )


def test_storage_bound_mesh_reconstruction_preserves_logical_ids(
    stored_mesh: CellMesh,
) -> None:
    restored = CellMesh.from_stored_fields(stored_mesh)
    assert restored.mesh_id == stored_mesh.mesh_id
    assert restored.topology_id == stored_mesh.topology_id
    assert restored.geometry_id == stored_mesh.geometry_id
    assert restored.geometry_layout_id == stored_mesh.geometry_layout_id
    assert restored.storage is stored_mesh.storage
    np.testing.assert_array_equal(restored.coordinates, stored_mesh.coordinates)
    np.testing.assert_array_equal(restored.blocks[0].global_ids, (73, 23))


def test_serial_mesh_reconstruction_preserves_nonmonotone_source_ids() -> None:
    source = unit_square(
        vertex_global_ids=np.asarray((31, 11, 41, 21), dtype=np.int64),
        cell_global_ids=np.asarray((73, 23), dtype=np.int64),
    )
    restored = CellMesh.from_stored_fields(source)
    assert restored.mesh_id == source.mesh_id
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(source)
    np.testing.assert_array_equal(restored.vertex_global_ids, (31, 11, 41, 21))
    np.testing.assert_array_equal(restored.blocks[0].global_ids, (73, 23))
    np.testing.assert_array_equal(restored.blocks[0].vertices, source.blocks[0].vertices)


def test_face_loop_reconstruction_preserves_polyhedral_incidence() -> None:
    reference = reference_cell_topology("hexahedron")
    source = CellMesh.from_polyhedra(
        np.asarray(reference.vertices, dtype=np.float64),
        (tuple(np.asarray(face, dtype=np.int32) for face in reference.entities[2]),),
        cell_global_ids=np.asarray((802,), dtype=np.int64),
    )
    restored = CellMesh.from_stored_fields(source)
    assert restored.mesh_id == source.mesh_id
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(source)
    for degree in range(4):
        np.testing.assert_array_equal(
            restored.entity_set(degree).entity_ids, source.entity_set(degree).entity_ids
        )
    for degree in range(1, 4):
        np.testing.assert_array_equal(
            restored.topology.incidences[degree - 1].signs,
            source.topology.incidences[degree - 1].signs,
        )


def test_mixed_polyhedral_grouping_retains_named_source_cell_ids() -> None:
    reference = reference_cell_topology("hexahedron")
    first = np.asarray(reference.vertices, dtype=np.float64)
    points = np.concatenate((first, first + np.asarray((2.0, 0.0, 0.0))))
    faces = tuple(np.asarray(face, dtype=np.int32) for face in reference.entities[2])
    source = CellMesh.from_mixed_3d(
        points,
        (),
        polyhedra={"z-bank": (tuple(face + 8 for face in faces),), "a-bank": (faces,)},
        polyhedral_cell_global_ids={
            "z-bank": np.asarray((901,), dtype=np.int64),
            "a-bank": np.asarray((107,), dtype=np.int64),
        },
    )
    assert tuple(block.name for block in source.blocks) == ("a-bank", "z-bank")
    np.testing.assert_array_equal(source.block("a-bank").global_ids, (107,))
    np.testing.assert_array_equal(source.block("z-bank").global_ids, (901,))
    restored = CellMesh.from_stored_fields(source)
    assert restored.mesh_id == source.mesh_id
    np.testing.assert_array_equal(restored.entity_set(3).entity_ids, (107, 901))


def test_storage_reconstruction_refuses_duplicate_stored_entity_ids(
    stored_mesh: CellMesh,
) -> None:
    storage = stored_mesh.storage
    assert storage is not None
    duplicate = storage.entity_global_ids[0].at[1].set(storage.entity_global_ids[0][0])
    damaged = eqx.tree_at(
        lambda value: value.entity_global_ids,
        storage,
        (duplicate, *storage.entity_global_ids[1:]),
    )
    with pytest.raises(ValueError, match=r"CellMeshStorage\.entity_global_ids"):
        CellMeshStorage.from_stored_fields(damaged)


def test_storage_bound_mesh_reconstruction_refuses_changed_corner_coordinates(
    stored_mesh: CellMesh,
) -> None:
    damaged = eqx.tree_at(
        lambda value: value.coordinates,
        stored_mesh,
        stored_mesh.coordinates.at[0, 0].add(0.125),
    )
    with pytest.raises(ValueError, match=r"CellMesh\.coordinates"):
        CellMesh.from_stored_fields(damaged)


@pytest.mark.parametrize(
    ("kind", "rows", "valid"),
    (
        ("triangle", ((2, 0, 1), (1, 3, 2)), ((True,) * 3,) * 2),
        (
            "polyhedron",
            ((0, 1, 2, 3, 0), (1, 2, 3, 4, 0)),
            ((True, True, True, True, False),) * 2,
        ),
    ),
    ids=("triangle", "padded-polyhedron"),
)
def test_block_reconstruction_preserves_rows_masks_and_nonmonotone_ids(
    kind: str,
    rows: tuple[tuple[int, ...], ...],
    valid: tuple[tuple[bool, ...], ...],
) -> None:
    source = CellBlock(
        "source-bank",
        kind,
        np.asarray(rows, dtype=np.int32),
        vertex_valid=np.asarray(valid, dtype=np.bool_),
        global_ids=np.asarray((71, 11), dtype=np.int64),
    )
    restored = CellBlock.from_stored_fields(source)
    assert restored.block_id == source.block_id
    assert restored.name == source.name
    assert restored.cell_kind == source.cell_kind
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(source)
    np.testing.assert_array_equal(restored.vertices, source.vertices)
    np.testing.assert_array_equal(restored.vertex_valid, source.vertex_valid)
    np.testing.assert_array_equal(restored.global_ids, (71, 11))


def test_block_reconstruction_refuses_duplicate_stored_global_ids() -> None:
    source = unit_square().blocks[0]
    assert isinstance(source, CellBlock)
    damaged = eqx.tree_at(
        lambda value: value.global_ids,
        source,
        source.global_ids.at[1].set(source.global_ids[0]),
    )
    with pytest.raises(ValueError, match=r"CellBlock\.global_ids"):
        CellBlock.from_stored_fields(damaged)


@pytest.fixture
def periodic_mesh() -> CellMesh:
    lifted = unit_square(cell_global_ids=np.asarray((73, 23), dtype=np.int64))
    periodic = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(2, dtype=np.float64)),
        np.zeros(lifted.coordinates.shape[0], dtype=np.int64),
        np.asarray(lifted.coordinates, dtype=np.int64),
    )
    return CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        periodic_topology=periodic,
    )


def test_stored_lift_reconstruction_retains_physical_entities_not_quotient_entities(
    periodic_mesh: CellMesh,
) -> None:
    descriptor = periodic_mesh.periodic_topology
    assert descriptor is not None
    lifted = CellMesh.lifted_from_stored_fields(periodic_mesh)
    assert lifted.periodic_topology is None
    assert lifted.topology_id == descriptor.lifted_topology_id
    assert lifted.topology.entity_sets[0].count == 4
    assert descriptor.quotient.entity_sets[0].count == 1
    assert lifted.numeric_version == periodic_mesh.numeric_version
    for degree, entities in enumerate(periodic_mesh.topology.entity_sets):
        np.testing.assert_array_equal(
            lifted.topology.entity_sets[degree].entity_ids, entities.entity_ids
        )
    np.testing.assert_array_equal(lifted.blocks[0].global_ids, (73, 23))
    np.testing.assert_array_equal(lifted.coordinates, periodic_mesh.coordinates)


def test_periodic_mesh_reconstruction_rebinds_quotient_to_validated_stored_lift(
    periodic_mesh: CellMesh,
) -> None:
    restored = CellMesh.from_stored_fields(periodic_mesh)
    source_descriptor = periodic_mesh.periodic_topology
    descriptor = restored.periodic_topology
    assert source_descriptor is not None
    assert descriptor is not None
    assert restored.mesh_id == periodic_mesh.mesh_id
    assert descriptor.periodic_topology_id == source_descriptor.periodic_topology_id
    for degree, entities in enumerate(source_descriptor.quotient.entity_sets):
        np.testing.assert_array_equal(
            descriptor.quotient.entity_sets[degree].entity_ids, entities.entity_ids
        )


def test_stored_lift_reconstruction_refuses_corrupted_physical_vertex_ids(
    periodic_mesh: CellMesh,
) -> None:
    vertices = periodic_mesh.topology.entity_sets[0].entity_ids
    damaged = eqx.tree_at(
        lambda value: value.topology.entity_sets[0].entity_ids,
        periodic_mesh,
        vertices.at[1].set(vertices[0]),
    )
    with pytest.raises(ValueError, match=r"CellMesh\.vertex_global_ids"):
        CellMesh.lifted_from_stored_fields(damaged)


def test_stored_lift_reconstruction_refuses_implicit_storage_materialization(
    stored_mesh: CellMesh,
) -> None:
    with pytest.raises(ValueError, match=r"CellMesh\.storage"):
        CellMesh.lifted_from_stored_fields(stored_mesh)
