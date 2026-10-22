# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex_parts,
    refine_adaptive_simplex_parts,
)
from phydrax.graph._partition import GraphPartitionPlan, partition_graph
from phydrax.meshing._distribution import SimplexNeighborhoodWorkset
from phydrax.meshing._distribution_migration import (
    accept_ownerlocal_migration,
    ownerlocal_forest_marks,
    ownerlocal_simplex_graph,
    ownerlocal_solver_cell_owners,
    ownerlocal_solver_forest,
    OwnerLocalMigrationState,
    prepare_ownerlocal_migration,
    restart_solver_target_owners,
)
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
    _require_instance,
    _require_present,
)


_CONTRACT_KEYS = (
    "forest_101",
    "forest_102",
    "forest_104",
    "forest_105",
    "forest_111",
    "forest_112",
    "forest_113",
    "forest_116",
    "forest_117",
    "forest_118",
    "forest_119",
    "forest_120",
    "forest_126",
    "forest_128",
    "forest_129",
    "forest_130",
    "forest_132",
    "forest_136",
    "forest_140",
    "forest_143",
    "forest_148",
    "forest_151",
    "forest_153",
    "forest_154",
    "forest_155",
    "forest_157",
    "forest_158",
    "forest_161",
    "forest_40",
    "forest_42",
    "forest_44",
    "forest_46",
    "forest_80",
    "forest_81",
    "forest_83",
    "forest_85",
    "forest_92",
    "forest_94",
)


def _forest(scenario: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    capacity = 16 if scenario == "forest_overflow" else 32
    points = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rows = np.asarray([[0, 1, 2, 5], [0, 2, 3, 5], [0, 3, 4, 5], [0, 4, 1, 5]], np.int32)
    layout = AdaptiveSimplexLayout(
        3,
        3,
        vertex_capacity=32,
        cell_capacity=capacity,
        protected_edge_capacity=4,
        maximum_closure_iterations=32,
        maximum_coarsening_passes=2,
    )

    def piece(indices: np.ndarray) -> AdaptiveSimplexState:
        ids = np.unique(rows[indices]).astype(np.int64)
        local = np.searchsorted(ids, rows[indices]).astype(np.int32)
        n = len(indices)
        return adaptive_simplex_state(
            layout,
            coordinates=points[ids],
            vertex_ids=ids,
            vertex_active=np.ones(len(ids), np.bool_),
            vertex_parents=np.full((len(ids), 2), -1, np.int32),
            vertex_protected=np.zeros(len(ids), np.bool_),
            cells=local,
            tuples=local,
            tags=np.full(n, 3, np.int32),
            blocks=np.zeros(n, np.int32),
            generations=np.zeros(n, np.int32),
            parents=np.full(n, -1, np.int32),
            children=np.full((n, 2), -1, np.int32),
            bisection_vertices=np.full(n, -1, np.int32),
            cell_ids=indices.astype(np.int64) + 10,
            cell_active=np.ones(n, np.bool_),
            cell_classes=np.full(n, 7, np.int32),
            facet_classes=np.zeros((n, 4), np.int32),
            protected_edges=np.empty((0, 2), np.int32),
            next_vertex_id=6,
            next_cell_id=14,
        )

    source = jax.tree_util.tree_map(
        lambda *a: jnp.stack(a),
        piece(np.arange(2, dtype=np.int32)),
        piece(np.arange(2, 4, dtype=np.int32)),
    )
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0, 1), (1, 0)))
    first = refine_adaptive_simplex_parts(layout, parts, source, source.mesh.cell_active)
    checks.setdefault("forest_40", []).append(
        bool(np.all(np.asarray(first.report.status) == 0))
    )
    undo = coarsen_adaptive_simplex_parts(
        layout, parts, first.state, first.state.mesh.cell_active
    )
    checks.setdefault("forest_42", []).append(
        bool(np.all(np.asarray(undo.report.status) == 0))
    )
    second = refine_adaptive_simplex_parts(
        layout, parts, undo.state, undo.state.mesh.cell_active
    )
    checks.setdefault("forest_44", []).append(
        bool(np.all(np.asarray(second.report.status) == 0))
    )
    split = second.state
    checks.setdefault("forest_46", []).append(
        bool(np.any(np.asarray(split.vertex_removal) >= 0))
    )
    split = eqx.tree_at(
        lambda state: (state.vertex_protected, state.protected_codes),
        split,
        (
            split.vertex_removal >= 0,
            jnp.broadcast_to(
                jnp.asarray([1, *[np.iinfo(np.int64).max] * 3], jnp.int64), (2, 4)
            ),
        ),
    )
    mesh = split.mesh
    P, C = mesh.cell_ids.shape
    V = mesh.vertex_ids.shape[1]
    W = 4
    vids = jnp.take_along_axis(mesh.vertex_ids[:, :, None], mesh.cells, axis=1)
    coords = jax.vmap(lambda x, r: x[r])(mesh.coordinates, mesh.cells)
    rank = jnp.arange(P, dtype=jnp.int32)[:, None]
    co = jnp.where(mesh.cell_ids >= 0, rank, -1)
    vo = np.full((P, V), -1, np.int32)
    for p in range(P):
        for v, identifier in enumerate(np.asarray(mesh.vertex_ids[p])):
            if identifier >= 0:
                vo[p, v] = min(
                    (q for q in range(P) if identifier in np.asarray(mesh.vertex_ids[q]))
                )
    work = SimplexNeighborhoodWorkset(
        jnp.where(mesh.cell_active, mesh.cell_ids, jnp.iinfo(jnp.int64).max),
        vids,
        coords,
        jnp.broadcast_to(rank, (P, C)),
        split.cell_classes,
        mesh.facet_neighbors < 0,
        mesh.cell_active,
        mesh.cell_ids,
        mesh.cell_ids[:, :, None],
        jnp.full((P, C, W, W), -1, jnp.int64),
        jnp.zeros((P, C, W, W), jnp.float64),
        jnp.zeros(P, jnp.int32),
    )
    history = (
        mesh.cell_ids + 2**54,
        split.retired,
        mesh.cell_ids[:, :, None] * jnp.asarray([2.0, 3.0], jnp.float64),
    )
    vhistory = (mesh.vertex_ids + 2**54, split.vertex_removal, mesh.coordinates)
    original = OwnerLocalMigrationState(
        work,
        jnp.where(mesh.cell_active, mesh.cell_ids.astype(jnp.float64), 0),
        coords,
        mesh.cell_ids,
        jnp.zeros((P, C, W), jnp.int32),
        split,
        co,
        jnp.asarray(vo),
        history,
        vhistory,
    )
    targets = jnp.broadcast_to((jnp.arange(C, dtype=jnp.int32) % 2)[None, :], (P, C))
    prepared = prepare_ownerlocal_migration(
        parts, original, targets, halo_width=1, vertex_capacity=16
    )
    accepted = accept_ownerlocal_migration(prepared)
    if scenario == "forest_overflow":
        checks.setdefault("forest_80", []).append(
            bool(
                np.all(
                    np.asarray(prepared.evidence.status)
                    & int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
                )
            )
        )
        _record_array(checks, "forest_81", prepared.evidence.accepted, [False, False])
        for old, new in zip(
            jax.tree_util.tree_leaves(original),
            jax.tree_util.tree_leaves(accepted),
            strict=True,
        ):
            _record_array(checks, "forest_83", old, new)
    else:
        _record_array(checks, "forest_85", prepared.evidence.accepted, [True, True])
        placed = jax.device_put(
            original, NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
        )
        graph = ownerlocal_simplex_graph(
            parts,
            placed,
            vertex_weights=jnp.where(
                placed.workset.cell_valid, 1 + placed.workset.cell_ids % 3, 0
            ).astype(jnp.int64),
        )
        graph_result = partition_graph(
            graph, GraphPartitionPlan(2, maximum_imbalance=1.5)
        )
        checks.setdefault("forest_92", []).append(bool(graph_result.evidence.accepted))
        raw_proposal, proposal_status = restart_solver_target_owners(
            parts,
            _require_present(placed.forest),
            placed.workset,
            _require_instance(graph_result.parts, jax.Array),
        )
        _record_array(checks, "forest_94", proposal_status, [0, 0])
        for p in range(P):
            active_rows = np.asarray(placed.workset.cell_valid[p])
            proposal_by_id = dict(
                zip(
                    np.asarray(placed.workset.cell_ids[p])[active_rows],
                    np.asarray(graph_result.parts[p])[active_rows],
                    strict=True,
                )
            )
            active_raw = np.asarray(_require_present(placed.forest).mesh.cell_active[p])
            expected_proposal = [
                proposal_by_id[identifier]
                for identifier in np.asarray(
                    _require_present(placed.forest).mesh.cell_ids[p]
                )[active_raw]
            ]
            _record_array(
                checks,
                "forest_101",
                np.asarray(raw_proposal[p])[active_raw],
                expected_proposal,
            )
        _record_array(
            checks,
            "forest_102",
            np.asarray(raw_proposal)[
                ~np.asarray(_require_present(placed.forest).mesh.cell_active)
            ],
            -1,
        )
        forest = _require_present(accepted.forest)
        checks.setdefault("forest_104", []).append(bool(forest is not None))
        _record_array(
            checks,
            "forest_105",
            np.sort(
                np.asarray(forest.mesh.cell_ids)[np.asarray(forest.mesh.cell_ids) >= 0]
            ),
            np.sort(
                np.asarray(split.mesh.cell_ids)[np.asarray(split.mesh.cell_ids) >= 0]
            ),
        )
        for p in range(P):
            ids = np.asarray(forest.mesh.cell_ids[p])
            active = ids >= 0
            _record_array(
                checks,
                "forest_111",
                np.asarray(accepted.cell_history[0][p])[active],
                ids[active] + 2**54,
            )
            _record_array(
                checks,
                "forest_112",
                np.asarray(accepted.cell_history[1][p])[active],
                np.asarray(forest.retired[p])[active],
            )
            _record_array(
                checks,
                "forest_113",
                np.asarray(accepted.cell_history[2][p])[active],
                ids[active, None] * [2.0, 3.0],
            )
            vertices = np.asarray(forest.mesh.vertex_ids[p])
            active = vertices >= 0
            _record_array(
                checks,
                "forest_116",
                np.asarray(accepted.vertex_history[0][p])[active],
                vertices[active] + 2**54,
            )
            _record_array(
                checks,
                "forest_117",
                np.asarray(accepted.vertex_history[1][p])[active],
                np.asarray(forest.vertex_removal[p])[active],
            )
            _record_array(
                checks,
                "forest_118",
                np.asarray(accepted.vertex_history[2][p])[active],
                np.asarray(forest.mesh.coordinates[p])[active],
            )
        _record_array(checks, "forest_119", forest.clocks, split.clocks)
        _record_array(checks, "forest_120", forest.counters, split.counters)
        for p in range(P):
            codes = np.asarray(forest.protected_codes[p])
            codes = codes[codes != np.iinfo(np.int64).max]
            if codes.size:
                endpoints = np.asarray(forest.mesh.vertex_ids[p])[
                    np.stack((codes // V, codes % V), axis=1)
                ]
                _record_array(checks, "forest_126", endpoints, [[0, 1]])
            removed = np.asarray(forest.vertex_removal[p]) >= 0
            checks.setdefault("forest_128", []).append(
                bool(np.all(np.asarray(forest.vertex_protected[p])[removed]))
            )
        _record_array(
            checks, "forest_129", prepared.evidence.forest_cells_before, [10, 10]
        )
        _record_array(checks, "forest_130", prepared.evidence.forest_cells_after, [20, 0])
        solver_owners, solver_owner_status = ownerlocal_solver_cell_owners(
            prepared.target_parts, accepted
        )
        _record_array(checks, "forest_132", solver_owner_status, [0, 0])
        for p in range(P):
            active = np.asarray(forest.mesh.cell_active[p])
            scientific_ids = np.asarray(forest.mesh.cell_ids[p])[active]
            _record_array(
                checks,
                "forest_136",
                np.asarray(solver_owners[p])[active],
                (scientific_ids - 22) % 2,
            )
        (
            solver_forest,
            solver_cell_history,
            solver_vertex_history,
            solver_forest_status,
        ) = ownerlocal_solver_forest(prepared.target_parts, accepted)
        _record_array(checks, "forest_140", solver_forest_status, [0, 0])
        for p in range(P):
            allocated = np.asarray(solver_forest.mesh.cell_ids[p]) >= 0
            _record_array(
                checks,
                "forest_143",
                np.asarray(solver_cell_history[0][p])[allocated],
                np.asarray(solver_forest.mesh.cell_ids[p])[allocated] + 2**54,
            )
            resident_ids = np.asarray(accepted.workset.cell_ids[p])[
                np.asarray(accepted.workset.cell_valid[p])
            ]
            checks.setdefault("forest_148", []).append(
                bool(
                    np.all(
                        np.isin(resident_ids, np.asarray(solver_forest.mesh.cell_ids[p]))
                    )
                )
            )
        old_owned = np.asarray(work.cell_valid) & (
            np.asarray(work.cell_owner) == np.arange(P)[:, None]
        )
        new_owned = np.asarray(accepted.workset.cell_valid) & (
            np.asarray(accepted.workset.cell_owner) == np.arange(P)[:, None]
        )
        checks.setdefault("forest_151", []).append(
            bool(
                np.sum(np.asarray(original.cell_fields)[old_owned])
                == np.sum(np.asarray(accepted.cell_fields)[new_owned])
            )
        )
        marks, mark_status = ownerlocal_forest_marks(
            prepared.target_parts, accepted, accepted.workset.cell_valid
        )
        _record_array(checks, "forest_153", mark_status, [0, 0])
        _record_array(checks, "forest_154", marks, forest.mesh.cell_active)
        checks.setdefault("forest_155", []).append(
            bool(prepared.forest_parts is not None)
        )
        execution_parts = _require_present(prepared.forest_parts)
        assert isinstance(execution_parts, AdaptiveSimplexParts)
        coarsened = coarsen_adaptive_simplex_parts(
            layout, execution_parts, forest, marks
        )
        checks.setdefault("forest_157", []).append(
            bool(np.all(np.asarray(coarsened.report.status) == 0))
        )
        _record_array(
            checks,
            "forest_158",
            np.sort(
                np.asarray(coarsened.state.mesh.cell_ids)[
                    np.asarray(coarsened.state.mesh.cell_active)
                ]
            ),
            [10, 11, 12, 13],
        )
        _record_array(checks, "forest_161", coarsened.report.operations, [4, 4])
        print(
            "Retired/removed records, int64/bool/vector history, conservation and cross-owner sibling coarsening preserved"
        )
    return checks


def forest_positive() -> dict[str, Any]:
    return _forest("positive")


def forest_forest_overflow() -> dict[str, Any]:
    return _forest("forest_overflow")
