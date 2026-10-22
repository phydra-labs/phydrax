from __future__ import annotations

import math
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    exact_orient3d,
    level_set_split_3d,
    LevelSetOutputMode,
    LevelSetSplitRequest,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    homogeneous_barycentric_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import affine_vertex_positions
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._exact_algebraic_geometry import (
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
)
from phydrax.discretization._exact_rational_geometry import rational_source_layouts
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._level_set import LevelSetEvidence, LevelSetMeshAdaptation
from tests._support.cell_meshes import incompatible_simplex
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mesh_certificate_fixtures import (
    affine_tetrahedral_root_restrictions,
    TETRAHEDRON,
)
from tests._support.meshing_fixtures import mesh_scope


M = phx.meshing


# Fixed empirical allowances from test_native_level_set.py
# at a6827a9e5; no meshing acceptance policy is relaxed.
# Affine density 2 + (0.2, 0.3, 0.4).x and volume-weighted unit-simplex inventories.
_SIMPLEX_TRANSFER_ATOL = 1e-14
# Periodic edge-root transfer of the fixture inventory 5.0 over its quotient cells.
_PERIODIC_INVENTORY_ATOL = 1e-13


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
        mesh_scope(source.mesh, 0, ids),
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
    assert evidence.maximum_carrier_deviation == 0.0
    assert np.asarray(evidence.root_vertex_global_ids).size == 0
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
        raise ValueError(
            "adapted.target.mesh.connectivity must be tetrahedral; got non-tetrahedral connectivity."
        )
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
    assert evidence.maximum_carrier_deviation == 0.0
    np.testing.assert_array_equal(evidence.root_weights, np.full((3, 2), 0.5))
    np.testing.assert_array_equal(evidence.root_deviations, 0.0)
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
        atol=_SIMPLEX_TRANSFER_ATOL,
    )
    np.testing.assert_allclose(
        _inventory(target, target_density),
        _inventory(source.mesh, density),
        rtol=0.0,
        atol=_SIMPLEX_TRANSFER_ATOL,
    )
    assert adapted.parent_reference_vertices.shape == (target.entity_set(3).count, 4, 3)
    assert np.all(adapted.parent_cells == 0)


def test_root_parameter_survives_overflowing_endpoint_sum() -> None:
    # |f(a)| + |f(b)| overflows binary64; each crossing still has its representable
    # midpoint root on every platform.
    source = _tetrahedron()
    values = np.asarray((-1.5e308, 1.5e308, 1.5e308, 1.5e308), dtype=np.float64)
    adapted = _execute(source, _request(source, values))
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    assert evidence.inserted_roots == 3
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
        np.testing.assert_allclose(
            fields, expected, rtol=0.0, atol=_SIMPLEX_TRANSFER_ATOL
        )
        for history in range(offsets.shape[0]):
            for field in range(offsets.shape[1]):
                np.testing.assert_allclose(
                    _inventory(adapted.target.mesh, fields[:, history, field]),
                    _inventory(initial_mesh, initial[:, history, field]),
                    rtol=0.0,
                    atol=_SIMPLEX_TRANSFER_ATOL,
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
        atol=_SIMPLEX_TRANSFER_ATOL,
    )
    for field in range(fields.shape[1]):
        np.testing.assert_allclose(
            _inventory(second.target.mesh, carried[:, field]),
            _inventory(first.target.mesh, fields[:, field]),
            rtol=0.0,
            atol=_SIMPLEX_TRANSFER_ATOL,
        )


@pytest.mark.parametrize("change", ("counters", "partition", "roots"))
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
    elif change == "partition":
        changed = eqx.tree_at(
            lambda value: value.cell_sides,
            evidence,
            -evidence.cell_sides,
        )
    else:
        changed = eqx.tree_at(
            lambda value: value.root_weights,
            evidence,
            evidence.root_weights.at[0, 0].set(0.25),
        )
    with pytest.raises(ValueError, match=r"LevelSetEvidence\."):
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
    # Rational root targets publish one block per coefficient action, and a
    # multi-block CellMesh binds its numeric revision into mesh_id; the
    # unchanged content is its topology, carrier bytes and source geometry.
    assert accepted.target.mesh.topology_id == first.target.mesh.topology_id
    np.testing.assert_array_equal(
        np.asarray(accepted.target.mesh.coordinates).view(np.uint64),
        np.asarray(first.target.mesh.coordinates).view(np.uint64),
    )
    assert cell_geometry_id(accepted.target.geometry) == cell_geometry_id(
        first.target.geometry
    )
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
        atol=_SIMPLEX_TRANSFER_ATOL,
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
        M.MeshZone(
            name, M.MeshZoneRole.REGION, mesh_scope(mesh, 3, ids[index : index + 1])
        )
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
    protected = mesh_scope(
        source.mesh, 1, np.asarray(source.mesh.entity_set(1).entity_ids)
    )
    with pytest.raises(MeshcoreError, match=r"LevelSetMeshAdaptation\.native_split"):
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
        _inventory(target, np.asarray(transfer.apply(density))),
        5.0,
        rtol=0.0,
        atol=_PERIODIC_INVENTORY_ATOL,
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


@pytest.mark.parametrize(
    "values", ((1.0, -1.0, 1.0, 1.0), (2.0, -0.3, 3.0, 0.7)), ids=["midpoint", "skewed"]
)
def test_generic_tetrahedron_inserts_exact_source_owned_roots(
    values: tuple[float, ...],
) -> None:
    # A13: these roots are not exactly representable on their source edges.
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            ((0.1, 0.2, 0.3), (0.7, 0.5, 0.9), (0.2, 0.8, 0.1), (0.6, 0.1, 0.7)),
            dtype=np.float64,
        ),
        np.asarray(((0, 2, 1, 3),), dtype=np.int32),
    )
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    field = np.asarray(values, dtype=np.float64)
    adapted = _execute(source, _request(source, field))
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(adapted.target.mesh)
    assert evidence.inserted_roots == 3
    target = adapted.target.mesh
    source_ids = np.asarray(source.mesh.vertex_global_ids)
    points = np.asarray(source.mesh.coordinates)
    target_ids = np.asarray(target.vertex_global_ids)
    exact_distances = []
    for root, (first, second), (h_first, h_second), deviation in zip(
        np.asarray(evidence.root_vertex_global_ids),
        np.asarray(evidence.root_source_global_ids),
        np.asarray(evidence.root_weights),
        np.asarray(evidence.root_deviations),
        strict=True,
    ):
        a = int(np.flatnonzero(source_ids == first)[0])
        b = int(np.flatnonzero(source_ids == second)[0])
        assert (h_first, h_second) == (abs(field[b]), abs(field[a]))
        weight_a, weight_b = Fraction(float(h_first)), Fraction(float(h_second))
        assert Fraction(field[a]) * weight_a + Fraction(field[b]) * weight_b == 0
        exact = [
            (weight_a * Fraction(points[a, k]) + weight_b * Fraction(points[b, k]))
            / (weight_a + weight_b)
            for k in range(3)
        ]
        carrier = np.asarray(target.coordinates)[
            int(np.flatnonzero(target_ids == root)[0])
        ]
        # Python's Fraction-to-float conversion is correctly rounded (ties to even).
        assert tuple(carrier) == tuple(float(value) for value in exact)
        distance = sum(
            (Fraction(float(c)) - s) ** 2 for c, s in zip(carrier, exact, strict=True)
        )
        assert Fraction(float(deviation)) ** 2 >= distance
        exact_distances.append(distance)
    assert any(distance > 0 for distance in exact_distances)
    assert evidence.maximum_carrier_deviation == float(
        np.max(np.asarray(evidence.root_deviations))
    )
    cells = np.concatenate([np.asarray(block.vertices) for block in target.blocks])
    corners = np.asarray(target.coordinates)[cells]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )


def _exact_positions(
    source: M.CellMeshingResult, *records: LevelSetEvidence
) -> dict[int, tuple[Fraction, ...]]:
    """Exact positions of the binary64 source vertices and every inserted root."""
    positions = {
        int(identifier): tuple(Fraction(float(value)) for value in point)
        for identifier, point in zip(
            np.asarray(source.mesh.vertex_global_ids),
            np.asarray(source.mesh.coordinates),
            strict=True,
        )
    }
    for record in records:
        for root, (first, second), (h_first, h_second) in zip(
            np.asarray(record.root_vertex_global_ids),
            np.asarray(record.root_source_global_ids),
            np.asarray(record.root_weights),
            strict=True,
        ):
            weight_a, weight_b = Fraction(float(h_first)), Fraction(float(h_second))
            positions[int(root)] = tuple(
                (weight_a * x + weight_b * y) / (weight_a + weight_b)
                for x, y in zip(
                    positions[int(first)], positions[int(second)], strict=True
                )
            )
    return positions


def _require_certified_roots(
    adapted: M.MeshAdaptationResult, exact: dict[int, tuple[Fraction, ...]]
) -> None:
    """Published carriers are the nearest binary64 roots within their bounds."""
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    target = adapted.target.mesh
    ids = np.asarray(target.vertex_global_ids)
    coordinates = np.asarray(target.coordinates)
    for root, deviation in zip(
        np.asarray(evidence.root_vertex_global_ids),
        np.asarray(evidence.root_deviations),
        strict=True,
    ):
        point = exact[int(root)]
        carrier = coordinates[int(np.flatnonzero(ids == root)[0])]
        # Python's Fraction-to-float conversion is correctly rounded (ties to even).
        rounded = np.asarray([float(value) for value in point], dtype=np.float64)
        np.testing.assert_array_equal(carrier.view(np.uint64), rounded.view(np.uint64))
        distance = sum(
            (Fraction(float(c)) - s) ** 2 for c, s in zip(carrier, point, strict=True)
        )
        assert Fraction(float(deviation)) ** 2 >= distance
        assert (float(deviation) == 0.0) == (distance == 0)


def _root(evidence: LevelSetEvidence, first: int, second: int) -> tuple[int, float]:
    for root, sources, deviation in zip(
        np.asarray(evidence.root_vertex_global_ids),
        np.asarray(evidence.root_source_global_ids),
        np.asarray(evidence.root_deviations),
        strict=True,
    ):
        if set(sources.tolist()) == {first, second}:
            return int(root), float(deviation)
    raise AssertionError(
        "native_split.root_source_global_ids must contain source edge "
        f"{(first, second)}; got no inserted root"
    )


@pytest.mark.parametrize(
    ("second_values", "edge", "expected", "carrier_based"),
    (
        # Rational 0 and 1/3 with values -1, 2: root 1/9, whose carrier is the
        # exactly representable rounded-endpoint root; its deviation is not 0.
        ({"origin": -1.0}, ("origin", "third"), Fraction(1, 9), None),
        # Rational 1/3 and 1 with homogeneous weights (3, 8): root 9/11, whose
        # nearest binary64 value differs from the rounded-endpoint root.
        (
            {"rest": -8.0, "unit": 3.0},
            ("third", "unit"),
            Fraction(9, 11),
            0.8181818181818181,
        ),
    ),
    ids=["one-ninth", "nine-elevenths"],
)
def test_repeated_nondyadic_roots_use_exact_source_geometry(
    second_values: dict[str, float],
    edge: tuple[str, str],
    expected: Fraction,
    carrier_based: float | None,
) -> None:
    source = _tetrahedron()
    first = _execute(source, _request(source, np.asarray((-1.0, 2.0, 2.0, 2.0))))
    first_evidence = first.evidence
    assert isinstance(first_evidence, LevelSetEvidence)
    assert first_evidence.source_geometry_id == cell_geometry_id(source.geometry)
    exact = _exact_positions(source, first_evidence)
    _require_certified_roots(first, exact)
    source_ids = np.asarray(source.mesh.vertex_global_ids)
    named = {
        "origin": int(source_ids[0]),
        "unit": int(source_ids[1]),
        "third": next(
            identifier
            for identifier, point in exact.items()
            if point == (Fraction(1, 3), Fraction(0), Fraction(0))
        ),
    }
    middle = first.target.mesh
    middle_ids = np.asarray(middle.vertex_global_ids)
    values = np.full(middle_ids.shape, second_values.get("rest", 2.0))
    for name in ("origin", "unit"):
        if name in second_values:
            values[middle_ids == named[name]] = second_values[name]
    second = _execute(
        first.target,
        _request(
            first.target, values, previous=first_evidence, accept_topology_change=True
        ),
    )
    evidence = second.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(second.target.mesh)
    assert evidence.source_geometry_id == cell_geometry_id(first.target.geometry)
    exact = _exact_positions(source, first_evidence, evidence)
    _require_certified_roots(second, exact)
    root, deviation = _root(evidence, named[edge[0]], named[edge[1]])
    assert exact[root] == (expected, Fraction(0), Fraction(0))
    carrier = np.asarray(second.target.mesh.coordinates)[
        int(np.flatnonzero(np.asarray(second.target.mesh.vertex_global_ids) == root)[0])
    ]
    assert carrier[0].hex() == float(expected).hex()
    assert deviation > 0.0
    if carrier_based is not None:
        assert carrier_based == float((3 * Fraction(float(Fraction(1, 3))) + 8) / 11)
        assert carrier[0] != carrier_based
    # Every target cell lies in its source parent: retained vertices are parent
    # corners and every root's source edge is a parent edge.
    parents = np.concatenate([np.asarray(block.vertices) for block in middle.blocks])
    root_sources = dict(
        zip(
            np.asarray(evidence.root_vertex_global_ids).tolist(),
            map(tuple, np.asarray(evidence.root_source_global_ids).tolist()),
            strict=True,
        )
    )
    target = second.target.mesh
    target_ids = np.asarray(target.vertex_global_ids)
    for cell, parent in zip(
        np.concatenate([np.asarray(block.vertices) for block in target.blocks]),
        second.parent_cells,
        strict=True,
    ):
        corners = set(middle_ids[parents[int(parent)]].tolist())
        for vertex in target_ids[cell].tolist():
            assert set(root_sources.get(vertex, (vertex,))) <= corners
    transfer = second.transfer
    assert transfer is not None and transfer.conservative and transfer.preserves_linear
    density = 2.0 + np.asarray(middle.coordinates) @ np.asarray((0.2, 0.3, 0.4))
    np.testing.assert_allclose(
        _inventory(target, np.asarray(transfer.apply(density))),
        _inventory(middle, density),
        rtol=0.0,
        atol=_SIMPLEX_TRANSFER_ATOL,
    )


def _native_request(
    mesh: phx.discretization.CellMesh,
    values: np.ndarray,
    mode: LevelSetOutputMode,
    *,
    carriers: bool,
) -> LevelSetSplitRequest:
    connectivity = mesh.connectivity
    assert isinstance(connectivity, TetrahedralConnectivity)
    return LevelSetSplitRequest(
        points=np.ascontiguousarray(mesh.coordinates, dtype=np.float64),
        values=np.ascontiguousarray(values, dtype=np.float64),
        cells=np.ascontiguousarray(mesh.blocks[0].vertices, dtype=np.int32),
        faces=np.ascontiguousarray(connectivity.faces, dtype=np.int32),
        edges=np.ascontiguousarray(connectivity.edges, dtype=np.int32),
        protected_edges=np.zeros((connectivity.edges.shape[0],), dtype=np.int8),
        max_vertices=10,
        max_cells=32,
        work_limit=1_000_000,
        cavity_limit=32,
        output_mode=mode,
        output_points=np.full((10, 3), -7.0) if carriers else None,
        output_cells=np.full((32, 4), -7, dtype=np.int32),
        output_parents=np.full((32,), -7, dtype=np.int32),
        output_sources=np.full((10, 2), -7, dtype=np.int32),
        output_weights=np.full((10, 2), -7.0),
        output_deviations=np.full((10,), -7.0) if carriers else None,
        counter_slots=np.zeros((6,), dtype=np.int64),
    )


def test_native_topology_only_split_publishes_no_geometry_outputs() -> None:
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(
            ((0.1, 0.2, 0.3), (0.7, 0.5, 0.9), (0.2, 0.8, 0.1), (0.6, 0.1, 0.7)),
            dtype=np.float64,
        ),
        np.asarray(((0, 2, 1, 3),), dtype=np.int32),
    )
    values = np.asarray((2.0, -0.3, 3.0, 0.7), dtype=np.float64)
    with NativeExecutionScope(
        NativeExecutionLimits(1_000_000, 1000, 1000, 64 * 1024 * 1024, math.inf)
    ):
        full = _native_request(
            mesh, values, LevelSetOutputMode.FULL_CARRIERS, carriers=True
        )
        topology = _native_request(
            mesh, values, LevelSetOutputMode.TOPOLOGY_ONLY, carriers=False
        )
        full_run = level_set_split_3d(full)
        topology_run = level_set_split_3d(topology)
        assert full_run.status is MeshcoreStatus.OK
        assert topology_run.status is MeshcoreStatus.OK
        assert topology_run.output_mode is LevelSetOutputMode.TOPOLOGY_ONLY
        assert topology.output_points is None and topology.output_deviations is None
        vertices, cells = topology_run.counters.vertices, topology_run.counters.cells
        assert (vertices, cells) == (full_run.counters.vertices, full_run.counters.cells)
        assert topology_run.counters.inserted == full_run.counters.inserted == 3
        assert topology_run.counters.work_units < full_run.counters.work_units
        for name in ("output_cells", "output_parents", "output_sources"):
            np.testing.assert_array_equal(getattr(topology, name), getattr(full, name))
        np.testing.assert_array_equal(
            topology.output_weights.view(np.uint64), full.output_weights.view(np.uint64)
        )
        for mode, carriers in (
            (LevelSetOutputMode.TOPOLOGY_ONLY, True),
            (LevelSetOutputMode.FULL_CARRIERS, False),
        ):
            mismatched = _native_request(mesh, values, mode, carriers=carriers)
            with pytest.raises(ValueError, match=r"LevelSetSplitRequest\.output_points"):
                level_set_split_3d(mismatched)
            np.testing.assert_array_equal(mismatched.counter_slots, 0)
            np.testing.assert_array_equal(mismatched.output_cells, -7)


_UNIT_CORNERS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _third_vertex_source() -> M.CellMeshingResult:
    """Unit tetrahedron whose first vertex is the rational source point (1/3, 0, 0)."""
    corners = np.asarray(_UNIT_CORNERS, dtype=np.float64)
    carrier = corners.copy()
    carrier[0, 0] = float(Fraction(1, 3))
    mesh = phx.discretization.CellMesh(
        carrier,
        (
            phx.discretization.CellBlock(
                "solid",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(4, dtype=np.int64),
    )
    # Corner 0 is (2 P0 + P1) / 3 of the input unit tetrahedron.
    element = homogeneous_barycentric_element(
        coordinate_lagrange_element("tetrahedron", 1),
        np.asarray(
            (
                (2.0, 1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"solid": element},
        {"solid": np.asarray(((0, 1, 2, 3),), dtype=np.int32)},
        corners,
    )
    return M.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )


def test_exact_source_root_replaces_rounded_endpoint_duplicate() -> None:
    # Rational edge 1/3 -> 1 with values (-1, 2**55): weights (2**55, 1). The rational
    # root rounds to the successor of float(1/3); the root of the rounded
    # endpoint rounds back onto vertex 0's own carrier.
    far = 2.0**55
    third = float(Fraction(1, 3))
    assert float((Fraction(far) * Fraction(third) + 1) / (Fraction(far) + 1)) == third
    assert float((Fraction(far) / 3 + 1) / (Fraction(far) + 1)) == math.nextafter(
        third, math.inf
    )
    source = _third_vertex_source()
    values = np.asarray((-1.0, far, far, far), dtype=np.float64)
    with NativeExecutionScope(
        NativeExecutionLimits(1_000_000, 1000, 1000, 64 * 1024 * 1024, math.inf)
    ):
        # Full carriers take the binary64 input as the exact source and must
        # refuse the duplicate carrier without publishing anything.
        full = _native_request(
            source.mesh, values, LevelSetOutputMode.FULL_CARRIERS, carriers=True
        )
        assert level_set_split_3d(full).status is MeshcoreStatus.DEGENERATE_INPUT
        np.testing.assert_array_equal(full.output_points, -7.0)
        np.testing.assert_array_equal(full.output_cells, -7)
        # Topology alone is independent of carrier rounding.
        topology = _native_request(
            source.mesh, values, LevelSetOutputMode.TOPOLOGY_ONLY, carriers=False
        )
        run = level_set_split_3d(topology)
        assert run.status is MeshcoreStatus.OK
        assert run.counters.inserted == 3
    # The public route certifies the rational carriers as distinct and positively
    # oriented, so it is not refused as a native duplicate; the deterministic target
    # audit then welds the one-ulp-separated vertices under its default
    # coincidence tolerance and publishes nothing.
    with pytest.raises(M.MeshingFailure, match="coincident_vertices") as refused:
        _execute(source, _request(source, values))
    assert refused.value.category is M.MeshingFailureCategory.AUDIT_FAILED


def test_algebraic_source_roots_round_from_field_values_and_repeat() -> None:
    # A declared algebraic tetrahedron with a sqrt(2) corner is a nested source:
    # its roots stay in the field and only correctly rounded carriers publish.
    with coordinate_reference_budget() as budget:
        zero, one = RadicalElement(), RadicalElement(1)
        bank = (
            (zero, zero, zero),
            (RadicalElement.sqrt(2), zero, zero),
            (zero, one, zero),
            (zero, zero, one),
        )
        cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        elements = {"solid": coordinate_lagrange_element("tetrahedron", 1)}
        routes = {"solid": cells}
        exact_source = ExactAlgebraicCellGeometrySource(
            bank,
            rational_source_layouts(elements, routes),
            source_identity="level-set-radical-tetrahedron",
            maximum_work=budget.maximum_work_units,
            maximum_bits=1024,
            budget=budget,
        )
        points = exact_source.representatives(budget=budget)
        geometry = phx.discretization.CellGeometrySpec(
            elements, routes, points, exact_source=exact_source, budget=budget
        )
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "solid", "tetrahedron", cells, global_ids=np.asarray((0,), dtype=np.int64)
            ),
        ),
        vertex_global_ids=np.arange(4, dtype=np.int64),
    )
    source = M.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    first = _execute(source, _request(source, np.asarray((-1.0, 2.0, 2.0, 2.0))))
    evidence = first.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(first.target.mesh)
    assert isinstance(
        first.target.geometry.exact_source, ExactAlgebraicCellGeometrySource
    )
    root, deviation = _root(evidence, 0, 1)
    exact = RadicalElement.sqrt(2) / 3
    target = first.target.mesh
    ids = np.asarray(target.vertex_global_ids)
    carrier = np.asarray(target.coordinates)[int(np.flatnonzero(ids == root)[0])]
    assert tuple(carrier.tolist()) == (float(exact), 0.0, 0.0)
    lower, upper = (exact - RadicalElement(Fraction(float(carrier[0])))).interval(
        bits=256
    )
    assert 0 < max(abs(lower), abs(upper)) <= Fraction(deviation)
    # The algebraic target is itself an admitted source of a further insertion.
    values = np.full(ids.shape, -8.0)
    values[ids == 1] = 3.0
    second = _execute(
        first.target,
        _request(first.target, values, previous=evidence, accept_topology_change=True),
    )
    repeated = second.evidence
    assert isinstance(repeated, LevelSetEvidence)
    repeated.require_current(second.target.mesh)
    assert repeated.source_geometry_id == cell_geometry_id(first.target.geometry)
    assert repeated.inserted_roots == 3
    assert np.all(np.asarray(repeated.root_deviations) > 0.0)


def _require_exact_composed_roots(
    source: M.CellMeshingResult, adapted: M.MeshAdaptationResult
) -> None:
    """Roots are correctly rounded exact images over the source geometry."""
    evidence = adapted.evidence
    assert isinstance(evidence, LevelSetEvidence)
    evidence.require_current(adapted.target.mesh)
    assert evidence.source_geometry_id == cell_geometry_id(source.geometry)
    with coordinate_reference_budget() as budget:
        before = affine_vertex_positions(source.mesh, source.geometry, budget=budget)
        after = affine_vertex_positions(
            adapted.target.mesh, adapted.target.geometry, budget=budget
        )
    ids = np.asarray(source.mesh.vertex_global_ids)
    rows = {int(identifier): row for row, identifier in enumerate(ids)}
    target_ids = np.asarray(adapted.target.mesh.vertex_global_ids)
    coordinates = np.asarray(adapted.target.mesh.coordinates)
    for identifier, row in rows.items():
        target_row = int(np.flatnonzero(target_ids == identifier)[0])
        assert after[target_row] == before[row]
        np.testing.assert_array_equal(
            coordinates[target_row].view(np.uint64),
            np.asarray(source.mesh.coordinates)[row].view(np.uint64),
        )
    for vertex, (first, second), (h_first, h_second), deviation in zip(
        np.asarray(evidence.root_vertex_global_ids),
        np.asarray(evidence.root_source_global_ids),
        np.asarray(evidence.root_weights),
        np.asarray(evidence.root_deviations),
        strict=True,
    ):
        weight_a, weight_b = Fraction(float(h_first)), Fraction(float(h_second))
        exact = tuple(
            (weight_a * x + weight_b * y) / (weight_a + weight_b)
            for x, y in zip(
                before[rows[int(first)]], before[rows[int(second)]], strict=True
            )
        )
        row = int(np.flatnonzero(target_ids == vertex)[0])
        assert after[row] == exact
        assert tuple(coordinates[row].tolist()) == tuple(float(value) for value in exact)
        distance = sum(
            (Fraction(float(c)) - s) ** 2
            for c, s in zip(coordinates[row], exact, strict=True)
        )
        assert Fraction(float(deviation)) ** 2 >= distance
        assert (float(deviation) == 0.0) == (distance == 0)
    cells = np.concatenate(
        [np.asarray(block.vertices) for block in adapted.target.mesh.blocks]
    )
    corners = coordinates[cells]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )


def test_rational_bisection_charts_are_admitted_exact_level_set_parents() -> None:
    # Uniform bisection publishes exact rational P1 charts over the input
    # root; a zero set through them composes those charts, so every root is the
    # correctly rounded exact image over the scientific root geometry.
    root = M.certify_cell_mesh(
        incompatible_simplex(3), phx.SpatialCoordinateContract.si()
    )
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            root,
            M.MarkedMeshAdaptation(
                np.asarray((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            ),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=M.BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        )
    )
    source = refined.target
    assert {type(element) for element in source.geometry.elements} == {
        PolynomialComposedCellGeometryElement
    }
    ids = np.asarray(source.mesh.vertex_global_ids)
    values = np.ones(ids.shape)
    values[ids == 0] = -1.0
    _require_exact_composed_roots(source, _execute(source, _request(source, values)))


def test_restricted_parents_compose_exact_level_set_children_and_repeat() -> None:
    # Restricted parents keep their own coordinate owner: every distinct child
    # chart is published in its own block, preserved cells keep the parent map,
    # and the conservative transfer measures the restricted maps.
    mesh, geometry = affine_tetrahedral_root_restrictions(TETRAHEDRON + 5.0)
    source = M.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    values = np.full((8,), 2.0)
    values[0] = -1.0
    first = _execute(source, _request(source, values))
    _require_exact_composed_roots(source, first)
    elements = {type(element) for element in first.target.geometry.elements}
    assert elements == {
        PolynomialComposedCellGeometryElement,
        RestrictedCellGeometryElement,
    }
    assert source.mesh.blocks[0].name in first.target.geometry.block_names
    evidence = first.evidence
    assert isinstance(evidence, LevelSetEvidence)
    ids = np.asarray(first.target.mesh.vertex_global_ids)
    repeated = np.full(ids.shape, -8.0)
    repeated[ids == 1] = 3.0
    second = _execute(
        first.target,
        _request(first.target, repeated, previous=evidence, accept_topology_change=True),
    )
    _require_exact_composed_roots(first.target, second)
    for before, after in ((source, first), (first.target, second)):
        transfer = after.transfer
        assert transfer is not None and transfer.conservative
        density = 2.0 + np.asarray(before.mesh.coordinates) @ np.asarray((0.2, 0.3, 0.4))
        np.testing.assert_allclose(
            _inventory(after.target.mesh, np.asarray(transfer.apply(density))),
            _inventory(before.mesh, density),
            rtol=0.0,
            atol=_SIMPLEX_TRANSFER_ATOL,
        )
