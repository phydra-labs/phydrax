from __future__ import annotations

from itertools import permutations, product
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import MeshcoreError
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._level_set import LevelSetEvidence, LevelSetMeshAdaptation


M = phx.meshing


def _scope(
    mesh: phx.discretization.CellMesh, degree: int, ids: np.ndarray
) -> M.MeshingScope:
    return M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        degree,
        mesh.entity_set(degree).entity_set_id,
        ids,
    )


def _tetrahedron(vertex_ids: tuple[int, ...] | None = None) -> M.CellMeshingResult:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        vertex_global_ids=None
        if vertex_ids is None
        else np.asarray(vertex_ids, dtype=np.int64),
    )
    return M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _bipyramid() -> phx.discretization.CellMesh:
    return phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            (
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
        np.asarray(((2, 3, 0, 4), (2, 1, 3, 4)), dtype=np.int32),
    )


def _request(
    source: M.CellMeshingResult,
    values: np.ndarray,
    *,
    previous: LevelSetEvidence | None = None,
    accept_topology_change: bool = False,
) -> LevelSetMeshAdaptation:
    ids = np.asarray(source.mesh.vertex_global_ids)
    return LevelSetMeshAdaptation(
        _scope(source.mesh, 0, ids),
        values[np.argsort(ids)],
        previous=previous,
        accept_topology_change=accept_topology_change,
    )


def _execute(
    source: M.CellMeshingResult,
    request: LevelSetMeshAdaptation,
    *,
    protected_scopes: tuple[M.MeshingScope, ...] = (),
) -> M.MeshAdaptationResult:
    return M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            request,
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_LEVEL_SET, protected_scopes=protected_scopes
            ),
        )
    )


def _inventory(mesh: phx.discretization.CellMesh, density: np.ndarray) -> float:
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = np.asarray(mesh.coordinates)[cells]
    frames = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
    volumes = np.abs(np.linalg.det(frames)) / 6.0
    return float(np.sum(volumes * np.mean(density[cells], axis=1)))


def test_exact_zero_samples_define_shared_interface_without_duplicate_roots() -> None:
    mesh = _bipyramid()
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    adapted = _execute(source, _request(source, np.asarray(mesh.coordinates)[:, 0]))
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(adapted.target.mesh)
    assert evidence.inserted_roots == 0
    assert evidence.exact_zero_samples == 3
    assert evidence.maximum_root_residual == 0.0
    assert np.all(np.abs(np.asarray(evidence.interface_orientations)) == 1)
    np.testing.assert_array_equal(
        evidence.zero_vertex_global_ids, np.asarray(mesh.vertex_global_ids)[2:]
    )
    np.testing.assert_array_equal(np.sort(evidence.cell_sides), (-1, 1))
    interface_ids = np.asarray(evidence.interface_facet_global_ids)
    assert interface_ids.size == 1
    rows = np.flatnonzero(
        np.isin(np.asarray(adapted.target.mesh.entity_set(2).entity_ids), interface_ids)
    )
    connectivity = adapted.target.mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise ValueError("The level-set fixture must publish tetrahedral connectivity.")
    corners = np.asarray(adapted.target.mesh.coordinates)[
        np.asarray(connectivity.faces)[rows]
    ]
    np.testing.assert_array_equal(corners[..., 0], 0.0)
    normals = (
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        * np.asarray(evidence.interface_orientations)[:, None]
    )
    assert np.all(normals[:, 0] > 0.0)
    np.testing.assert_array_equal(normals[:, 1:], 0.0)
    assert {zone.name for zone in adapted.target.zones} == {
        evidence.inside_region_id,
        evidence.outside_region_id,
    }
    assert adapted.target.patches[0].adjacent_zone_ids == tuple(
        sorted(zone.zone_id for zone in adapted.target.zones)
    )


@pytest.mark.parametrize(
    "vertex_ids", [None, (10, 3, 20, 7)], ids=["canonical-ids", "nonpositional-ids"]
)
def test_native_inserted_roots_conform_and_preserve_conservative_inventory(
    vertex_ids: tuple[int, ...] | None,
) -> None:
    source = _tetrahedron(vertex_ids)
    points = np.asarray(source.mesh.coordinates)
    adapted = _execute(source, _request(source, np.sum(points, axis=1) - 0.5))
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    assert evidence.inserted_roots == 3
    assert evidence.maximum_root_residual == 0.0
    target = adapted.target.mesh
    root_rows = np.flatnonzero(
        np.isin(
            np.asarray(target.vertex_global_ids),
            np.asarray(evidence.zero_vertex_global_ids),
        )
    )
    assert set(map(tuple, np.asarray(target.coordinates)[root_rows])) == {
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    }
    assert evidence.topology_signature == (1, 1, 1, 1, 2)
    transfer = adapted.transfer
    assert transfer is not None and transfer.conservative and transfer.preserves_linear
    density = 2.0 + points @ np.asarray((0.2, 0.3, 0.4), dtype=np.float64)
    target_density = np.asarray(transfer.apply(density))
    np.testing.assert_allclose(
        target_density,
        2.0 + np.asarray(target.coordinates) @ np.asarray((0.2, 0.3, 0.4)),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        _inventory(target, target_density),
        _inventory(source.mesh, density),
        rtol=0.0,
        atol=1e-14,
    )
    assert adapted.parent_reference_vertices.shape == (target.entity_set(3).count, 4, 3)
    assert np.all(adapted.parent_cells == 0)


def test_moving_zero_set_transfers_every_field_and_history_payload() -> None:
    source = _tetrahedron((10, 3, 20, 7))
    initial_mesh = source.mesh
    points = np.asarray(source.mesh.coordinates)
    gradients = np.asarray(
        (((0.2, 0.3, 0.4), (-0.1, 0.2, 0.3)), ((0.4, -0.2, 0.1), (0.3, 0.1, -0.2))),
        dtype=np.float64,
    )
    offsets = np.asarray(((2.0, 3.0), (4.0, 5.0)), dtype=np.float64)
    fields = offsets + np.einsum("vd,hfd->vhf", points, gradients)
    initial = fields.copy()
    previous = None
    for position in (0.5, 0.25):
        points = np.asarray(source.mesh.coordinates)
        adapted = _execute(
            source,
            _request(source, np.sum(points, axis=1) - position, previous=previous),
        )
        transfer = adapted.transfer
        assert (
            transfer is not None and transfer.conservative and transfer.preserves_linear
        )
        fields = np.asarray(transfer.apply(fields))
        target_points = np.asarray(adapted.target.mesh.coordinates)
        expected = offsets + np.einsum("vd,hfd->vhf", target_points, gradients)
        np.testing.assert_allclose(fields, expected, rtol=0.0, atol=1e-14)
        for history in range(offsets.shape[0]):
            for field in range(offsets.shape[1]):
                np.testing.assert_allclose(
                    _inventory(adapted.target.mesh, fields[:, history, field]),
                    _inventory(initial_mesh, initial[:, history, field]),
                    rtol=0.0,
                    atol=1e-14,
                )
        previous = adapted.evidence
        assert isinstance(previous, LevelSetEvidence)
        assert not previous.topology_changed
        source = adapted.target


def test_archived_moving_zero_set_replays_current_evidence_and_transfer(
    tmp_path: Path,
) -> None:
    source = _tetrahedron((10, 3, 20, 7))
    first = _execute(
        source,
        _request(source, np.sum(np.asarray(source.mesh.coordinates), axis=1) - 0.5),
    )
    previous = first.evidence
    assert isinstance(previous, LevelSetEvidence)
    values = np.sum(np.asarray(first.target.mesh.coordinates), axis=1) - 0.25
    request = _request(first.target, values, previous=previous)
    receipt = write_meshing_source_closure(tmp_path / "moving-zero-set", request)
    restored = read_meshing_source_closure(
        tmp_path / "moving-zero-set",
        expected_content_id=receipt.content_id,
    )
    assert isinstance(restored, LevelSetMeshAdaptation)
    assert restored.request_id == request.request_id
    assert restored.previous is not None
    restored.previous.require_current(first.target.mesh)
    np.testing.assert_array_equal(restored.values, request.values)
    np.testing.assert_array_equal(restored.previous.counters, previous.counters)
    second = _execute(first.target, restored)
    evidence = second.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(second.target.mesh)
    assert not evidence.topology_changed
    transfer = second.transfer
    assert transfer is not None and transfer.conservative and transfer.preserves_linear
    points = np.asarray(first.target.mesh.coordinates)
    fields = np.stack((2.0 + points[:, 0], 3.0 + points[:, 1]), axis=1)
    carried = np.asarray(transfer.apply(fields))
    target_points = np.asarray(second.target.mesh.coordinates)
    np.testing.assert_allclose(
        carried,
        np.stack((2.0 + target_points[:, 0], 3.0 + target_points[:, 1]), axis=1),
        rtol=0.0,
        atol=1e-14,
    )
    for field in range(fields.shape[1]):
        np.testing.assert_allclose(
            _inventory(second.target.mesh, carried[:, field]),
            _inventory(first.target.mesh, fields[:, field]),
            rtol=0.0,
            atol=1e-14,
        )


@pytest.mark.parametrize("change", ("counters", "partition"))
def test_level_set_evidence_refuses_changed_scientific_payload(change: str) -> None:
    source = _tetrahedron()
    adapted = _execute(
        source,
        _request(source, np.sum(np.asarray(source.mesh.coordinates), axis=1) - 0.5),
    )
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(adapted.target.mesh)
    if change == "counters":
        changed = eqx.tree_at(
            lambda value: value.counters,
            evidence,
            evidence.counters.at[4].add(1),
        )
    else:
        changed = eqx.tree_at(
            lambda value: value.cell_sides,
            evidence,
            -evidence.cell_sides,
        )
    with pytest.raises(ValueError, match="Level-set evidence"):
        changed.require_current(adapted.target.mesh)
    evidence.require_current(adapted.target.mesh)


def test_moving_zero_set_component_deletion_requires_accepted_topology_event() -> None:
    source = _tetrahedron()
    first = _execute(
        source,
        _request(source, np.sum(np.asarray(source.mesh.coordinates), axis=1) - 0.5),
    )
    previous = first.evidence
    assert isinstance(previous, LevelSetEvidence)
    positive = np.ones((first.target.mesh.coordinates.shape[0],), dtype=np.float64)
    request = _request(first.target, positive, previous=previous)
    with pytest.raises(ValueError, match="changes topology"):
        _execute(first.target, request)
    previous.require_current(first.target.mesh)
    accepted = _execute(
        first.target,
        _request(first.target, positive, previous=previous, accept_topology_change=True),
    )
    evidence = accepted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(accepted.target.mesh)
    assert accepted.target.mesh.mesh_id == first.target.mesh.mesh_id
    assert accepted.target.mesh.numeric_version != first.target.mesh.numeric_version
    assert evidence.topology_changed and evidence.topology_event_accepted
    assert evidence.topology_signature == (0, 1, 0, 0, -1)
    assert evidence.interface_facet_global_ids.size == 0
    np.testing.assert_array_equal(evidence.cell_sides, 1)
    assert not accepted.target.patches
    assert {zone.name for zone in accepted.target.zones} == {evidence.outside_region_id}
    transfer = accepted.transfer
    assert transfer is not None and transfer.conservative
    density = 2.0 + np.asarray(first.target.mesh.coordinates)[:, 1]
    np.testing.assert_allclose(
        _inventory(accepted.target.mesh, np.asarray(transfer.apply(density))),
        _inventory(first.target.mesh, density),
        rtol=0.0,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="stale"):
        previous.require_current(accepted.target.mesh)
    with pytest.raises(ValueError, match="stale"):
        M.prepare_mesh_adaptation(
            accepted.target,
            _request(
                accepted.target, positive, previous=previous, accept_topology_change=True
            ),
            policy=M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_LEVEL_SET),
        )


def test_source_material_assignments_survive_level_set_partition_as_orthogonal_labels() -> (
    None
):
    mesh = _bipyramid()
    ids = np.asarray(mesh.entity_set(3).entity_ids)
    zones = tuple(
        M.MeshZone(name, M.MeshZoneRole.REGION, _scope(mesh, 3, ids[index : index + 1]))
        for index, name in enumerate(("material-left", "material-right"))
    )
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si(), zones=zones)
    adapted = _execute(source, _request(source, np.asarray(mesh.coordinates)[:, 1] - 0.5))
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    assert {zone.name for zone in adapted.target.zones} == {zone.name for zone in zones}
    assert {label.name for label in adapted.target.labels} == {
        evidence.inside_region_id,
        evidence.outside_region_id,
    }
    parent_by_id = dict(
        zip(
            np.concatenate(
                [np.asarray(block.global_ids) for block in adapted.target.mesh.blocks]
            ),
            adapted.parent_cells,
            strict=True,
        )
    )
    for index, zone in enumerate(adapted.target.zones):
        expected_parent = next(i for i, old in enumerate(zones) if old.name == zone.name)
        assert {
            int(parent_by_id[int(identifier)])
            for identifier in np.asarray(zone.scope.entity_ids)
        } == {expected_parent}, index


def test_protected_crossing_is_refused_without_mutating_source() -> None:
    source = _tetrahedron()
    mesh_id = source.mesh.mesh_id
    protected = _scope(source.mesh, 1, np.asarray(source.mesh.entity_set(1).entity_ids))
    with pytest.raises(MeshcoreError, match="refused source edge"):
        _execute(
            source,
            _request(source, np.sum(np.asarray(source.mesh.coordinates), axis=1) - 0.5),
            protected_scopes=(protected,),
        )
    assert source.mesh.mesh_id == mesh_id
    np.testing.assert_array_equal(
        source.mesh.coordinates,
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


def test_periodic_edge_root_orbits_preserve_quotient_and_inventory() -> None:
    vertices = tuple(product(range(3), range(2), range(2)))
    rows = {vertex: row for row, vertex in enumerate(vertices)}
    points = np.asarray(vertices, dtype=np.float64)
    cells = []
    for x in range(2):
        for axes in permutations(range(3)):
            vertex = np.asarray((x, 0, 0), dtype=np.int32)
            local = [rows[(int(vertex[0]), int(vertex[1]), int(vertex[2]))]]
            for axis in axes:
                vertex = vertex.copy()
                vertex[axis] += 1
                local.append(rows[(int(vertex[0]), int(vertex[1]), int(vertex[2]))])
            corners = points[local]
            if np.linalg.det(corners[1:] - corners[:1]) < 0.0:
                local[1], local[2] = local[2], local[1]
            cells.append(local)
    lifted = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(cells, dtype=np.int32)
    )
    representatives = np.asarray(
        [rows[(0, y, z)] if x == 2 else rows[(x, y, z)] for x, y, z in vertices],
        dtype=np.int32,
    )
    shifts = np.asarray([[1 if x == 2 else 0] for x, _, _ in vertices], dtype=np.int32)
    lattice = phx.discretization.PeriodicCell(
        np.asarray(((2.0, 0.0, 0.0),), dtype=np.float64)
    )
    descriptor = phx.discretization.PeriodicMeshTopology(
        lifted, lattice, representatives, shifts
    )
    mesh = phx.discretization.CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        periodic_topology=descriptor,
    )
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    adapted = _execute(source, _request(source, points[:, 1] - 0.5))
    target = adapted.target.mesh
    periodic = target.periodic_topology
    assert periodic is not None
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    assert evidence.topology_signature == (1, 1, 1, 0, 2)
    target_points = np.asarray(target.coordinates)
    root_mask = np.isin(
        np.asarray(target.vertex_global_ids), np.asarray(evidence.zero_vertex_global_ids)
    )
    np.testing.assert_array_equal(
        root_mask, root_mask[np.asarray(periodic.vertex_representatives)]
    )
    np.testing.assert_array_equal(target_points[root_mask, 1], 0.5)
    transfer = adapted.transfer
    assert transfer is not None and transfer.conservative
    density = 2.0 + points[:, 2]
    np.testing.assert_allclose(
        _inventory(target, np.asarray(transfer.apply(density))), 5.0, rtol=0.0, atol=1e-13
    )


def test_tangential_zero_face_is_retained_without_inventing_an_inside_region() -> None:
    mesh = _bipyramid()
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    adapted = _execute(
        source, _request(source, np.abs(np.asarray(mesh.coordinates)[:, 0]))
    )
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    np.testing.assert_array_equal(evidence.cell_sides, 1)
    assert evidence.interface_facet_global_ids.size == 1
    np.testing.assert_array_equal(evidence.interface_orientations, 0)
    assert evidence.topology_signature == (0, 2, 1, 1, 2)
    assert {zone.name for zone in adapted.target.zones} == {evidence.outside_region_id}
    patch = adapted.target.patches[0]
    assert patch.name == evidence.interface_id
    np.testing.assert_array_equal(
        patch.scope.entity_ids, evidence.interface_facet_global_ids
    )
    assert not patch.adjacent_zone_ids


def test_isolated_exact_zero_is_recorded_as_a_zero_dimensional_component() -> None:
    source = _tetrahedron()
    adapted = _execute(
        source, _request(source, np.asarray((0.0, 1.0, 1.0, 1.0), dtype=np.float64))
    )
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    assert evidence.topology_signature == (0, 1, 1, 1, 0)
    np.testing.assert_array_equal(
        evidence.zero_vertex_global_ids, np.asarray(source.mesh.vertex_global_ids)[:1]
    )
    assert evidence.interface_facet_global_ids.size == 0
