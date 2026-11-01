#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

import phydrax as phx
from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._periodic_topology import _entity_vertex_rows
from phydrax.discretization._view_support import OwnerLocalCellGeometry
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._distribution import (
    OwnerLocalCellShardRows,
    prepare_owner_local_cell_shard,
)


def _three_owner_group(name: str, /) -> ExecutionGroup:
    devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
    ranks = tuple(device.process_index for device in devices)
    if (
        jax.process_count() != 3
        or len(devices) != 3
        or ranks != (0, 1, 2)
        or len(jax.local_devices()) != 1
    ):
        pytest.skip("Owner-local shards need three processes with one device each.")
    return ExecutionGroup(
        ExecutionGroupSpec(
            name,
            ranks,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("shard", 3),),
        ),
        devices,
    )


def _square_mesh(count: int, /) -> MeshPart:
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.asarray([(x, y) for y in axis for x in axis], dtype=np.float64)
    triangles = []
    for j in range(count):
        for i in range(count):
            lower = j * (count + 1) + i
            upper = lower + count + 1
            triangles.extend(((lower, lower + 1, upper + 1), (lower, upper + 1, upper)))
    mesh = CellMesh(
        points,
        (CellBlock("triangles", "triangle", np.asarray(triangles, dtype=np.int64)),),
        numeric_version="shard-square",
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return MeshPart("shard-square", result)


def _serial_tables(part: MeshPart, /) -> dict[str, Any]:
    assert isinstance(part.carrier, phx.meshing.CellMeshingResult)
    mesh = part.carrier.mesh
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = vertex_ids[np.asarray(mesh.blocks[0].vertices, dtype=np.int64)]
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    edge_rows = _entity_vertex_rows(mesh)[1]
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_by_key = {
        tuple(
            sorted(int(vertex_ids[vertex]) for vertex in np.asarray(edge_rows[row]))
        ): int(edge_ids[row])
        for row in range(edge_ids.size)
    }
    incident: dict[tuple[int, int], list[int]] = {}
    for cell, vertices in zip(cell_ids.tolist(), cells.tolist(), strict=True):
        for first, second in ((0, 1), (1, 2), (0, 2)):
            key = tuple(sorted((vertices[first], vertices[second])))
            incident.setdefault(key, []).append(cell)
    return {
        "vertex_ids": vertex_ids,
        "coordinates": np.asarray(mesh.coordinates, dtype=np.float64),
        "cells": dict(zip(cell_ids.tolist(), cells.tolist(), strict=True)),
        "edge_by_key": edge_by_key,
        "incident": incident,
    }


def _face_halo(
    owned: set[int], incident: dict[tuple[int, int], list[int]], /
) -> set[int]:
    return {
        other
        for cells in incident.values()
        if len(cells) == 2 and owned.intersection(cells)
        for other in cells
        if other not in owned
    }


def _view(assembly: Any, /) -> OwnerLocalCellGeometry:
    (part,) = assembly.parts
    view = part.carrier
    assert isinstance(view, OwnerLocalCellGeometry)
    return view


def _check_view(
    view: OwnerLocalCellGeometry,
    serial: dict[str, Any],
    owner_of: dict[int, int],
    rank: int,
    /,
) -> None:
    owned = {cell for cell, owner in owner_of.items() if owner == rank}
    halo = _face_halo(owned, serial["incident"])
    cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
    assert not view.serial_source_view
    assert set(cell_ids.tolist()) == owned | halo
    assert set(cell_ids[np.asarray(view.owned, dtype=np.bool_)].tolist()) == owned
    np.testing.assert_array_equal(
        np.asarray(view.cell_owner_ranks),
        [owner_of[int(cell)] for cell in cell_ids],
    )
    vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
    resident_vertices = {
        vertex for cell in cell_ids.tolist() for vertex in serial["cells"][cell]
    }
    assert set(vertex_ids.tolist()) == resident_vertices
    serial_row = {int(vertex): row for row, vertex in enumerate(serial["vertex_ids"])}
    np.testing.assert_array_equal(
        np.asarray(view.vertex_coordinates),
        serial["coordinates"][[serial_row[int(vertex)] for vertex in vertex_ids]],
    )
    minimum_owner = {}
    for cell, owner in owner_of.items():
        for vertex in serial["cells"][cell]:
            minimum_owner[vertex] = min(owner, minimum_owner.get(vertex, owner))
    np.testing.assert_array_equal(
        np.asarray(view.vertex_owner_ranks),
        [minimum_owner[int(vertex)] for vertex in vertex_ids],
    )
    # Resident edges keep the serial mesh's global entity IDs.
    _, edge_ids = view.entity_binding(1)
    loops = view.entity_vertex_loops(1, np.asarray(edge_ids))
    for identifier, loop in zip(np.asarray(edge_ids).tolist(), loops, strict=True):
        key = tuple(sorted(int(vertex_ids[row]) for row in np.asarray(loop)))
        assert serial["edge_by_key"][key] == identifier
    expected_edges = {
        serial["edge_by_key"][tuple(sorted((vertices[a], vertices[b])))]
        for cell in cell_ids.tolist()
        for vertices in (serial["cells"][cell],)
        for a, b in ((0, 1), (1, 2), (0, 2))
    }
    assert set(np.asarray(edge_ids).tolist()) == expected_edges
    # The face-neighbor graph of owned cells is exact.
    graph_ids = np.asarray(view.owned_cell_ids, dtype=np.int64)
    offsets = np.asarray(view.cell_neighbor_offsets, dtype=np.int64)
    neighbor_ids = np.asarray(view.cell_neighbor_ids, dtype=np.int64)
    for row, cell in enumerate(graph_ids.tolist()):
        expected = {
            other
            for cells in serial["incident"].values()
            if cell in cells and len(cells) == 2
            for other in cells
            if other != cell
        }
        assert set(neighbor_ids[offsets[row] : offsets[row + 1]].tolist()) == expected
    mesh, geometry = view.reconstruct_local_geometry()
    if cell_ids.size:
        assert mesh is not None and geometry is not None
        assert mesh.blocks[0].cell_count == cell_ids.size
    else:
        assert mesh is None


def test_owner_local_cell_shards_partition_cells_with_exact_face_halos() -> None:
    group = _three_owner_group("owner-local-cell-shard")
    rank = jax.process_index()
    part = _square_mesh(4)
    serial = _serial_tables(part)
    assert isinstance(part.carrier, phx.meshing.CellMeshingResult)
    cell_ids = np.asarray(part.carrier.mesh.blocks[0].global_ids, dtype=np.int64)
    centroids = np.mean(
        serial["coordinates"][np.asarray(part.carrier.mesh.blocks[0].vertices)], axis=1
    )
    owners = np.minimum((centroids[:, 0] * 3.0).astype(np.int32), 2)
    distribution = phx.meshing.MeshDistribution(
        part, phx.discretization.CellPartition(owners, 3), halo_width=1
    )
    rows = distribution.owner_local_cell_shard_rows(rank)
    assembly = prepare_owner_local_cell_shard(
        part.name,
        part.name,
        part.part_id,
        phx.SpatialCoordinateContract.si(),
        "triangles",
        "triangle",
        rows,
        group,
        axis_name="owner_local_cell_shard_test",
    )
    view = _view(assembly)
    owner_of = dict(zip(cell_ids.tolist(), owners.tolist(), strict=True))
    _check_view(view, serial, owner_of, rank)
    owned_counts = np.asarray(
        multihost_utils.process_allgather(
            np.asarray((int(np.count_nonzero(np.asarray(view.owned))),), np.int32)
        )
    ).reshape((-1,))
    assert int(np.sum(owned_counts)) == cell_ids.size
    identities = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(
                bytes.fromhex(
                    canonical_fingerprint(
                        (
                            view.source_topology_id,
                            view.source_geometry_id,
                            view.source_map_id,
                            view.distribution_id,
                        )
                    )
                ),
                dtype=np.uint8,
            )
        )
    )
    assert np.all(identities == identities[0])


def test_owner_local_cell_shard_admits_a_zero_resident_owner() -> None:
    group = _three_owner_group("owner-local-cell-shard-empty")
    rank = jax.process_index()
    part = _square_mesh(3)
    serial = _serial_tables(part)
    assert isinstance(part.carrier, phx.meshing.CellMeshingResult)
    cell_ids = np.asarray(part.carrier.mesh.blocks[0].global_ids, dtype=np.int64)
    centroids = np.mean(
        serial["coordinates"][np.asarray(part.carrier.mesh.blocks[0].vertices)], axis=1
    )
    owners = (centroids[:, 1] > 0.5).astype(np.int32)
    distribution = phx.meshing.MeshDistribution(
        part, phx.discretization.CellPartition(owners, 2), halo_width=1
    )
    rows = (
        distribution.owner_local_cell_shard_rows(rank)
        if rank < 2
        else OwnerLocalCellShardRows(
            np.empty((0,), np.int64),
            np.empty((0, 3), np.int64),
            np.empty((0,), np.int64),
            np.empty((0, 2), np.float64),
            (np.empty((0, 2), np.int64),),
            (np.empty((0,), np.int64),),
        )
    )
    assembly = prepare_owner_local_cell_shard(
        part.name,
        part.name,
        part.part_id,
        phx.SpatialCoordinateContract.si(),
        "triangles",
        "triangle",
        rows,
        group,
        axis_name="owner_local_cell_shard_empty_test",
    )
    view = _view(assembly)
    owner_of = dict(zip(cell_ids.tolist(), owners.tolist(), strict=True))
    _check_view(view, serial, owner_of, rank)
    from phydrax.discretization.finite_volume._side_trace import (
        finite_volume_field_space_id,
    )

    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        view, field_name="empty-owner-field"
    ).prepare()
    assert geometry.cell_count == view.cell_ids.size
    if rank == 2:
        assert geometry.mesh is None
        assert geometry.state_shape == (0, 1)
        assert geometry.face_blocks == ()
        with pytest.raises(ValueError, match="published cell field"):
            finite_volume_field_space_id(geometry)
        with pytest.raises(ValueError, match="discrete support"):
            geometry.integration_domain("cell")
        with pytest.raises(TypeError, match="CellMesh"):
            phx.discretization.prepare_unstructured_conservative_remap(
                geometry, geometry, provenance="empty-owner-refusal"
            )
        with pytest.raises(ValueError, match="resident cell topology"):
            phx.graph.facet_adjacency(geometry)
    else:
        assert geometry.cell_space is not None
        assert finite_volume_field_space_id(geometry) == canonical_fingerprint(
            {
                "kind": "finite-volume-field-space",
                "discretization": geometry.prepared_id,
                "field": geometry.cell_space.name,
                "components": list(geometry.component_names),
            }
        )
