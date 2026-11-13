from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._geometry_predicates import PredicateMode
from phydrax._meshcore import TetMeshEditStatus
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._periodic import (
    _prepare_periodic_stencil_transport,
    _transport_periodic_stencil,
    _transport_periodic_vertex_ids,
    stage_periodic_topology_edit,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._tetra_metric import (
    execute_tetra_metric_adaptation,
    TetraMetricOutcome,
)
from phydrax.meshing.providers._native_periodic_tet import (
    tet_mesh_periodic_identity_map,
    tet_mesh_periodic_state,
)


def _periodic_tetra_source(basis: np.ndarray | None = None) -> CellMeshingResult:
    shape = 3
    vertex_rows = np.arange(shape**3, dtype=np.int64).reshape((shape,) * 3)
    lattice = np.stack(
        np.meshgrid(*(np.arange(shape) for _ in range(3)), indexing="ij"), axis=-1
    ).reshape((-1, 3))
    coordinates = lattice.astype(np.float64) / 2.0
    if basis is None:
        basis = np.eye(3, dtype=np.float64)
    coordinates = coordinates @ basis.T
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
            if (
                np.linalg.det(coordinates[tetrahedron[1:]] - coordinates[tetrahedron[0]])
                < 0.0
            ):
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
        PeriodicCell(basis.T),
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


def test_periodic_stencil_transport_uses_source_identity_and_exact_weights() -> None:
    source = _periodic_tetra_source().mesh
    prepared = _prepare_periodic_stencil_transport(source)
    assert prepared.source_mesh_id == source.mesh_id
    zero = np.zeros(3, dtype=np.int64)
    shift = np.array([1, 0, 0], dtype=np.int64)
    stencil = {4: Fraction(3, 4), 5: Fraction(1, 4)}
    point = np.array([0.0, 0.5, 0.625], dtype=np.float64)
    assert _transport_periodic_vertex_ids(prepared, (4, 5), shift) == (22, 23)
    assert _transport_periodic_vertex_ids(prepared, (22,), shift) is None
    assert _transport_periodic_vertex_ids(prepared, (1000,), zero) is None
    lifted = _transport_periodic_stencil(prepared, stencil, point, shift)
    assert lifted is not None
    assert lifted[0] == {22: Fraction(3, 4), 23: Fraction(1, 4)}
    np.testing.assert_array_equal(lifted[1], [1.0, 0.5, 0.625])
    unchanged = _transport_periodic_stencil(prepared, stencil, point, zero)
    assert unchanged is not None
    assert unchanged[0] == stencil
    assert _transport_periodic_stencil(prepared, stencil, point, -shift) is None
    exact = [
        [Fraction(float(value)) for value in row]
        for row in np.asarray(source.coordinates)
    ]
    exact[22][2] += Fraction(1, 8)
    changed = _prepare_periodic_stencil_transport(
        source, tuple(tuple(row) for row in exact)
    )
    assert _transport_periodic_stencil(changed, stencil, point, shift) is None


def test_periodic_stencil_transport_refuses_noncommuting_sheared_rounding() -> None:
    basis = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    source = _periodic_tetra_source(basis).mesh
    prepared = _prepare_periodic_stencil_transport(source)
    assert prepared.source_mesh_id == source.mesh_id
    assert prepared.source_mesh_id != _periodic_tetra_source().mesh.mesh_id
    y_weight = Fraction(5, 2**54)
    z_weight = Fraction(1, 16)
    stencil = {4: 1 - y_weight - z_weight, 7: y_weight, 5: z_weight}
    point = np.array(
        [0.0, float(Fraction(1, 2) + Fraction(5, 2**55)), 0.53125], dtype=np.float64
    )
    shift = np.array([1, 0, 0], dtype=np.int64)
    assert _transport_periodic_vertex_ids(prepared, (4, 7, 5), shift) == (22, 25, 23)
    assert (
        _transport_periodic_stencil(prepared, stencil, point, np.zeros(3, dtype=np.int64))
        is not None
    )
    assert _transport_periodic_stencil(prepared, stencil, point, shift) is None


def test_periodic_coupled_model_lifts_one_consistent_source_choice_per_orbit() -> None:
    from phydrax.meshing import _tetra_coupled as coupled
    from phydrax.meshing._tetra_metric import (
        _apply_relocation_batch,
        _locate_source_weights,
    )
    from tests.unit.meshing.test_tetra_metric import _state

    state = _state(_periodic_tetra_source(), np.eye(3, dtype=np.float64))
    try:
        search = coupled._prepare_search(state)
        vertices = (4, 13, 22)
        details = coupled._vertex_details(search, vertices)
        assert details is not None
        choices = tuple(detail.source_cells for detail in details)
        combinations = tuple(coupled._source_choices(search, details, choices))
        assert combinations
        assert len(combinations) <= len(choices[0]) * len(choices[1])
        displacement = np.asarray((0.0, 1.0 / 32.0, 0.0), dtype=np.float64)
        goals = tuple(
            search.prepared.points[detail.row] + displacement for detail in details
        )
        queries = tuple(
            (
                Fraction(float(point[0])),
                Fraction(float(point[1])),
                Fraction(float(point[2])),
            )
            for point in goals
        )
        selected = next(
            records
            for records in combinations
            if all(
                _locate_source_weights(query, (record,)) is not None
                for query, record in zip(queries, records, strict=True)
            )
        )
        binding = coupled._CoupledBinding(
            state.source_identity,
            int(state.snapshot.generation),
            (4, 13),
            vertices,
            tuple(record[0] for record in selected),
        )
        incident = [
            index
            for index, edge in enumerate(search.edges)
            if search.measured[index] and any(vertex in edge for vertex in vertices)
        ]
        model = coupled._model_rows(
            search, details, selected, incident, binding, phx.optim.ConvexSolvePolicy()
        )
        assert model is not None
        equality = np.asarray(model[2], dtype=np.float64)
        assert np.linalg.matrix_rank(equality) == equality.shape[0]
        candidate = coupled._quantize(search, details, selected, np.tile(displacement, 3))
        assert candidate is not None
        proposals = candidate[0]
        assert {proposal["vertex"] for proposal in proposals} == set(vertices)
        assert _apply_relocation_batch(state, proposals) is TetMeshEditStatus.APPLIED
        rows = {int(vertex): row for row, vertex in enumerate(state.snapshot.vertex_ids)}
        for vertex, goal in zip(vertices, goals, strict=True):
            np.testing.assert_array_equal(state.snapshot.points[rows[vertex]], goal)
        np.testing.assert_array_equal(
            state.snapshot.points[rows[22]] - state.snapshot.points[rows[4]],
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        )
    finally:
        state.native.close()


def _execute(source: CellMeshingResult, metric: np.ndarray) -> TetraMetricOutcome:
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


def test_periodic_overlapping_relocation_requires_the_complete_lifted_orbit() -> None:
    from phydrax.meshing._tetra_metric import (
        _apply_relocation_batch,
        _assemble_native_edit,
        _locate_source_weights,
        _prepare_native_state,
    )

    source = _periodic_tetra_source()
    mesh = source.mesh
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (mesh.vertex_global_ids.size, 3, 3)
    ).copy()
    state = _prepare_native_state(
        mesh,
        source.geometry,
        source.coordinate_contract,
        metric,
        cell_classes=np.zeros(mesh.entity_set(3).entity_ids.size, dtype=np.int64),
        facet_classes=np.zeros(mesh.entity_set(2).entity_ids.size, dtype=np.int64),
        edge_classes=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.int64),
        protected_edges=np.zeros(mesh.entity_set(1).entity_ids.size, dtype=np.bool_),
        fixed_vertices=np.zeros(mesh.vertex_global_ids.size, dtype=np.bool_),
    )
    try:
        snapshot = state.snapshot
        ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
        cells = ids[np.asarray(snapshot.tetrahedra, dtype=np.int64)]
        cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
        proposals: list[dict[str, Any]] = []
        for vertex in (4, 13, 22):
            point = np.asarray(snapshot.points)[np.flatnonzero(ids == vertex)[0]].copy()
            point[1] += 1.0 / 32.0
            exact = (
                Fraction.from_float(float(point[0])),
                Fraction.from_float(float(point[1])),
                Fraction.from_float(float(point[2])),
            )
            located = _locate_source_weights(exact, state.source_records)
            assert located is not None
            _, vertices, weights = located
            stencil = {v: w for v, w in zip(vertices, weights, strict=True) if w}
            proposals.append(
                {
                    "vertex": vertex,
                    "point": point,
                    "metric": np.eye(3, dtype=np.float64),
                    "stencil": stencil,
                    "current_stencil": stencil.copy(),
                    "star_cell_ids": tuple(
                        int(c) for c in cell_ids[np.any(cells == vertex, axis=1)]
                    ),
                }
            )
        assert set(proposals[0]["star_cell_ids"]) & set(proposals[1]["star_cell_ids"])
        assert set(proposals[1]["star_cell_ids"]) & set(proposals[2]["star_cell_ids"])
        old_target = state.target_mesh
        old_periodic_state = state.periodic_state
        old_points = np.asarray(snapshot.points).copy()
        old_windings = {v: w.copy() for v, w in state.periodic_vertex_windings.items()}
        assert (
            _apply_relocation_batch(state, proposals[:2])
            is TetMeshEditStatus.REJECTED_TOPOLOGY
        )
        assert state.snapshot.generation == snapshot.generation
        np.testing.assert_array_equal(state.snapshot.points, old_points)
        assert state.target_mesh is old_target
        assert state.periodic_state is old_periodic_state
        assert all(
            np.array_equal(state.periodic_vertex_windings[v], w)
            for v, w in old_windings.items()
        )

        assert _apply_relocation_batch(state, proposals) is TetMeshEditStatus.APPLIED
        assert state.target_mesh is not old_target
        assert state.periodic_state is not old_periodic_state
        edit, _ = _assemble_native_edit(state)
        target = stage_periodic_topology_edit(mesh, edit)
        assert target is not None and target.periodic_topology is not None
        target.periodic_topology.require_coordinates(target.coordinates)
        target_rows = {
            int(v): i for i, v in enumerate(np.asarray(target.vertex_global_ids))
        }
        for proposal in proposals:
            vertex = proposal["vertex"]
            np.testing.assert_array_equal(
                target.coordinates[target_rows[vertex]], proposal["point"]
            )
            assert state.vertex_stencils[vertex] == proposal["stencil"]
        np.testing.assert_array_equal(
            target.coordinates[target_rows[22]] - target.coordinates[target_rows[4]],
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        )
        assert all(
            np.array_equal(state.periodic_vertex_windings[v], w)
            for v, w in old_windings.items()
        )
    finally:
        state.native.close()


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
    cell_blocks = tuple(block for block in target.blocks if isinstance(block, CellBlock))
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


def test_native_tetra_metric_orients_sorted_simplex_3_quotient_cells() -> None:
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
    mesh = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        periodic_topology=PeriodicMeshTopology(
            lifted,
            source_topology.cell,
            source_topology.vertex_representatives,
            source_topology.vertex_shifts,
        ),
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)

    def volumes(cells: np.ndarray, /) -> np.ndarray:
        corners = coordinates[cells]
        return np.linalg.det(corners[:, 1:] - corners[:, :1])

    # Precondition: the canonical simplicial rows are sorted, and sorting
    # reverses some positively oriented lifted cells.
    assert np.all(volumes(np.asarray(block.vertices, dtype=np.int64)) > 0.0)
    assert np.any(volumes(np.sort(np.asarray(block.vertices, dtype=np.int64))) < 0.0)

    state = tet_mesh_periodic_state(mesh, None)
    row_of = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids).tolist())
    }
    members = {
        (int(orbit.representative_vertex_id), tuple(member.winding.tolist())): row_of[
            int(member.vertex_id)
        ]
        for orbit in state.orbits
        for member in orbit.members
    }
    corners = np.asarray(
        [
            [
                members[(int(vertex), tuple(winding))]
                for vertex, winding in zip(
                    cell.representative_vertices.tolist(),
                    cell.winding.tolist(),
                    strict=True,
                )
            ]
            for cell in state.quotient_cells
        ],
        dtype=np.int64,
    )
    assert np.all(volumes(corners) > 0.0)
    metric = np.broadcast_to(
        np.diag((16.0, 9.0, 4.0)), (mesh.vertex_global_ids.size, 3, 3)
    ).copy()
    # The native kernel admits the oriented quotient cells; it refused the
    # sorted-row orientation as invalid input before. ``simplex:3`` has the
    # tetrahedron's reference order, so the same operations commit as on
    # identical ``tetrahedron`` blocks.
    outcome = _execute(
        certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()), metric
    )
    reference = _execute(_periodic_tetra_source(), metric)

    assert outcome.evidence.passes >= 1
    assert outcome.evidence.maximum_region_volume_error <= 1.0e-12
    assert reference.evidence.splits > 0
    for name in (
        "status",
        "passes",
        "splits",
        "collapses",
        "flips23",
        "flips32",
        "ring_reconnections",
        "relocations",
        "rejected_topology",
    ):
        assert getattr(outcome.evidence, name) == getattr(reference.evidence, name)


def test_native_periodic_volume_route_adapts_metric_over_tetra_orbits() -> None:
    source_mesh = _periodic_tetra_source().mesh
    source_topology = source_mesh.periodic_topology
    assert source_topology is not None
    source_id = "periodic-tetra-adaptation"
    source_revision = "revision-1"
    source = phx.meshing.NativePeriodicSource(source_mesh, source_id, source_revision)
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
        # Equilateral tetrahedra do not tile space, so quotient-edge quantiles
        # cannot all equal the target; the unit-metric band accepts lengths in
        # [target / sqrt(2), target * sqrt(2)], i.e. relative deviation below 0.5.
        size_compliance=phx.meshing.SizeCompliancePolicy(relative_tolerance=0.5),
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("periodic_delaunay")
        )
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )

    target = result.mesh
    target_topology = target.periodic_topology
    assert target_topology is not None
    assert target_topology.vertex_orbit_count > source_topology.vertex_orbit_count
    target_topology.require_lifted(
        target.topology, target.lifted_topology_id, target.blocks
    )
    target_topology.require_coordinates(target.coordinates)
    assert result.compliance.issues == ()
    for _, achieved in result.compliance.achieved:
        assert 0.25 / np.sqrt(2.0) <= achieved <= 0.25 * np.sqrt(2.0)
    # Surviving quotient vertices keep their IDs; new ones are fresh lineage IDs.
    source_vertex_ids = set(map(int, np.asarray(quotient_vertices.entity_ids)))
    target_vertex_ids = set(
        map(int, np.asarray(target_topology.quotient.entity_sets[0].entity_ids))
    )
    assert source_vertex_ids < target_vertex_ids
    assert min(target_vertex_ids - source_vertex_ids) > max(source_vertex_ids)
    for degree, entities in enumerate(target_topology.quotient.entity_sets):
        assert target_topology.allocator_next_ids[degree] > int(
            np.max(np.asarray(entities.entity_ids))
        )
