from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._periodic import stage_periodic_topology_edit
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._tetra_metric import (
    execute_tetra_metric_adaptation,
    TetraMetricOutcome,
)
from phydrax.meshing.providers._native_periodic_tet import (
    tet_mesh_periodic_identity_map,
    tet_mesh_periodic_state,
)


def _periodic_tetra_source() -> CellMeshingResult:
    shape = 3
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    lattice = np.stack(
        np.meshgrid(*(np.arange(shape) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    coordinates = lattice.astype(np.float64) / 2.0
    indices = np.arange(shape, dtype=np.int64)
    representatives = vertex_rows[
        indices[:, None, None] % 2,
        indices[None, :, None] % 2,
        indices[None, None, :] % 2,
    ].reshape(-1)
    shifts = np.stack(
        np.meshgrid(*(indices // 2 for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    tetrahedra = []
    for cube in np.ndindex((shape - 1,) * 3):
        for order in permutations(range(3)):
            point = np.asarray(cube, dtype=np.int64)
            path = [point.copy()]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                path.append(point)
            tetrahedron = np.asarray(
                [vertex_rows[tuple(row)] for row in path], dtype=np.int32
            )
            if np.linalg.det(
                coordinates[tetrahedron[1:]] - coordinates[tetrahedron[0]]
            ) < 0.0:
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    vertex_ids = np.arange(shape**3, dtype=np.int64)
    block = CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(tetrahedra, dtype=np.int32),
        global_ids=np.arange(3000, 3000 + len(tetrahedra), dtype=np.int64),
    )
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    descriptor = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.eye(3, dtype=np.float64)),
        representatives,
        shifts,
    )
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=descriptor,
    )
    return certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _execute(
    source: CellMeshingResult, metric: np.ndarray
) -> TetraMetricOutcome:
    mesh = source.mesh
    return execute_tetra_metric_adaptation(
        source,
        metric,
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.int64),
        protected_edges=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.bool_),
        fixed_vertices=np.zeros(mesh.vertex_global_ids.size, dtype=np.bool_),
        predicate_mode=PredicateMode.EXACT,
        topology_operations=True,
        relocation=True,
    )


def test_native_tetra_metric_commits_complete_periodic_split_orbits() -> None:
    source = _periodic_tetra_source()
    metric = np.broadcast_to(
        np.diag((16.0, 9.0, 4.0)),
        (source.mesh.vertex_global_ids.size, 3, 3),
    ).copy()

    outcome = _execute(source, metric)

    assert outcome.evidence.splits > 0
    assert outcome.evidence.minimum_metric_quality > 0.0
    assert outcome.evidence.maximum_region_volume_error <= 1.0e-12
    source_row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(source.mesh.vertex_global_ids.tolist())
    }
    for row, target_coordinates in enumerate(outcome.edit.coordinates):
        valid = np.asarray(outcome.edit.stencil_valid[row], dtype=np.bool_)
        source_ids = np.asarray(outcome.edit.stencil_sources[row, valid], dtype=np.int64)
        weights = np.asarray(outcome.edit.stencil_weights[row, valid], dtype=np.float64)
        source_rows = np.asarray(
            [source_row_by_id[int(identifier)] for identifier in source_ids],
            dtype=np.int64,
        )
        reconstructed = weights @ np.asarray(source.mesh.coordinates)[source_rows]
        np.testing.assert_allclose(
            target_coordinates,
            reconstructed,
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        )
    target = stage_periodic_topology_edit(source.mesh, outcome.edit)
    assert target is not None and target.periodic_topology is not None
    assert all(isinstance(block, CellBlock) for block in target.blocks)
    cell_blocks = tuple(
        block for block in target.blocks if isinstance(block, CellBlock)
    )
    target.periodic_topology.require_lifted(
        target.topology, target.lifted_topology_id, cell_blocks
    )
    target.periodic_topology.require_coordinates(target.coordinates)
    target_state = tet_mesh_periodic_state(
        target,
        np.zeros(target.entity_set(3).entity_ids.size, dtype=np.int32),
    )
    assert target_state.identity_map.periodic_topology_id == (
        target.periodic_topology.periodic_topology_id
    )
    assert target_state.quotient_cells


def test_periodic_quotient_ids_continue_once_and_new_entities_take_fresh_ids() -> None:
    source = _periodic_tetra_source().mesh
    topology = source.periodic_topology
    assert topology is not None
    lifted = CellMesh(
        source.coordinates, source.blocks, vertex_global_ids=source.vertex_global_ids
    )
    orbit_rows = np.asarray(topology.orbit_indices, dtype=np.int64)[
        : int(topology.lifted_offsets[1])
    ]
    source_ids = np.asarray(topology.quotient.entity_sets[0].entity_ids, dtype=np.int64)
    ancestry = source_ids[orbit_rows].copy()
    ancestry[orbit_rows == 0] = -1  # quotient row 0 is new
    ancestry[np.flatnonzero(orbit_rows == 1)[0]] = source_ids[2]  # row 1 is ambiguous
    ancestry[orbit_rows == 3] = source_ids[4]  # rows 3 and 4 both claim one ID

    def rebuilt() -> PeriodicMeshTopology:
        return PeriodicMeshTopology(
            lifted,
            topology.cell,
            topology.vertex_representatives,
            topology.vertex_shifts,
            entity_allocator_next_ids={0: 1000},
            quotient_id_ancestry={0: ancestry},
        )

    first, second = rebuilt(), rebuilt()
    ids = np.asarray(first.quotient.entity_sets[0].entity_ids, dtype=np.int64)
    np.testing.assert_array_equal(ids, second.quotient.entity_sets[0].entity_ids)
    np.testing.assert_array_equal(ids[[2, 5, 6, 7]], source_ids[[2, 5, 6, 7]])
    # New, ambiguous, and doubly claimed entities retire their IDs in quotient order.
    np.testing.assert_array_equal(ids[[0, 1, 3, 4]], [1000, 1001, 1002, 1003])
    assert first.allocator_next_ids[0] == 1004
    with pytest.raises(ValueError, match="allocator"):
        PeriodicMeshTopology(
            lifted,
            topology.cell,
            topology.vertex_representatives,
            topology.vertex_shifts,
            quotient_id_ancestry={0: ancestry},
        )


def test_native_tetra_metric_rejects_non_equivariant_periodic_spd_data() -> None:
    source = _periodic_tetra_source()
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64),
        (source.mesh.vertex_global_ids.size, 3, 3),
    ).copy()
    metric[-1, 0, 0] = 2.0

    with pytest.raises(ValueError, match="equivariantly"):
        _execute(source, metric)


def test_periodic_tet_provider_accepts_simplex_3_cell_mesh() -> None:
    source_mesh = _periodic_tetra_source().mesh
    source_topology = source_mesh.periodic_topology
    assert source_topology is not None
    block = source_mesh.blocks[0]
    lifted = CellMesh.from_simplices(
        source_mesh.coordinates,
        block.vertices,
        dimension=3,
        vertex_global_ids=source_mesh.vertex_global_ids,
        cell_global_ids=block.global_ids,
    )
    periodic_topology = PeriodicMeshTopology(
        lifted,
        source_topology.cell,
        source_topology.vertex_representatives,
        source_topology.vertex_shifts,
    )
    mesh = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        periodic_topology=periodic_topology,
    )

    identity_map = tet_mesh_periodic_identity_map(mesh)
    state = tet_mesh_periodic_state(mesh, None)

    assert identity_map.periodic_topology_id == periodic_topology.periodic_topology_id
    assert state.identity_map.periodic_topology_id == identity_map.periodic_topology_id
    assert state.quotient_cells


def test_native_periodic_volume_route_adapts_metric_over_tetra_orbits() -> None:
    source_mesh = _periodic_tetra_source().mesh
    source_topology = source_mesh.periodic_topology
    assert source_topology is not None
    source_id = "periodic-tetra-adaptation"
    source_revision = "revision-1"
    source = phx.meshing.NativePeriodicSource(
        source_mesh, source_id, source_revision
    )
    volume_scope = phx.meshing.MeshingScope(
        source_id,
        source_revision,
        phx.meshing.MeshingEntityKind.MESH,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    quotient_vertices = source_topology.quotient.entity_sets[0]
    size_scope = phx.meshing.MeshingScope(
        source_id,
        source_revision,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        quotient_vertices.entity_set_id,
        quotient_vertices.entity_ids,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        volume_scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(size_scope, 0.25),),
    )
    result = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("periodic_delaunay")
    ).plan(
        source,
        specification,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    ).execute()

    target = result.mesh
    target_topology = target.periodic_topology
    assert target_topology is not None
    assert target_topology.vertex_orbit_count > source_topology.vertex_orbit_count
    target_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    target_topology.require_coordinates(target.coordinates)
    assert result.compliance.issues == ()


