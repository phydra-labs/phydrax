# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex_parts,
    refine_adaptive_simplex_parts,
)
from phydrax.meshing._distribution_forest import inherit_solver_cell_owners
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
)


_CONTRACT_KEYS = (
    "solver_34",
    "solver_39",
    "solver_43",
    "solver_47",
    "solver_63",
    "solver_65",
    "solver_66",
    "solver_73",
    "solver_74",
    "solver_78",
)


def _prepare_solver_source(
    checks: dict[str, Any],
) -> tuple[
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    jax.Array,
    AdaptiveSimplexState,
]:
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
        vertex_capacity=64,
        cell_capacity=64,
        protected_edge_capacity=1,
        maximum_closure_iterations=64,
        maximum_coarsening_passes=1,
    )

    def piece(indices: np.ndarray) -> AdaptiveSimplexState:
        ids = np.unique(rows[indices]).astype(np.int64)
        local = np.searchsorted(ids, rows[indices]).astype(np.int32)
        count = indices.size
        return adaptive_simplex_state(
            layout,
            coordinates=points[ids],
            vertex_ids=ids,
            vertex_active=np.ones(ids.size, np.bool_),
            vertex_parents=np.full((ids.size, 2), -1, np.int32),
            vertex_protected=np.zeros(ids.size, np.bool_),
            cells=local,
            tuples=local,
            tags=np.full(count, 3, np.int32),
            blocks=np.zeros(count, np.int32),
            generations=np.zeros(count, np.int32),
            parents=np.full(count, -1, np.int32),
            children=np.full((count, 2), -1, np.int32),
            bisection_vertices=np.full(count, -1, np.int32),
            cell_ids=indices.astype(np.int64) + 10,
            cell_active=np.ones(count, np.bool_),
            cell_classes=np.full(count, 7, np.int32),
            facet_classes=np.zeros((count, 4), np.int32),
            protected_edges=np.empty((0, 2), np.int32),
            next_vertex_id=6,
            next_cell_id=14,
        )

    source = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        piece(np.arange(2, dtype=np.int32)),
        piece(np.arange(2, 4, dtype=np.int32)),
    )
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0, 1), (1, 0)))
    first = refine_adaptive_simplex_parts(layout, parts, source, source.mesh.cell_active)
    _record_array(checks, "solver_34", first.report.status, [0, 0])
    initial = first.state
    bank = jnp.where(initial.mesh.cell_active, initial.mesh.cell_ids % 2, -1).astype(
        jnp.int32
    )
    coarse = coarsen_adaptive_simplex_parts(
        layout, parts, initial, initial.mesh.cell_active
    )
    _record_array(checks, "solver_39", coarse.report.status, [0, 0])
    current = coarse.state
    return (layout, parts, initial, bank, current)


def solver_refine() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    update = refine_adaptive_simplex_parts(
        layout, parts, initial, initial.mesh.cell_active
    )
    _record_array(checks, "solver_43", update.report.status, [0, 0])
    current = update.state
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    _record_array(checks, "solver_65", status, [0, 0])
    _record_array(
        checks, "solver_66", np.asarray(owners)[~np.asarray(current.mesh.cell_active)], -1
    )
    for rank in range(2):
        for slot in np.flatnonzero(np.asarray(current.mesh.cell_active[rank])):
            ancestor = slot
            while not bool(initial.mesh.cell_active[rank, ancestor]):
                ancestor = int(current.parents[rank, ancestor]) // 2
                assert ancestor >= 0
                checks.setdefault("solver_73", []).append(bool(ancestor >= 0))
            checks.setdefault("solver_74", []).append(
                bool(int(owners[rank, slot]) == int(bank[rank, ancestor]))
            )
    return checks


def solver_coarsen() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    _record_array(checks, "solver_65", status, [0, 0])
    _record_array(
        checks, "solver_66", np.asarray(owners)[~np.asarray(current.mesh.cell_active)], -1
    )
    _record_array(
        checks, "solver_78", np.asarray(owners)[np.asarray(current.mesh.cell_active)], 0
    )
    return checks


def solver_mixed() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    update = refine_adaptive_simplex_parts(
        layout, parts, current, current.mesh.cell_active
    )
    _record_array(checks, "solver_47", update.report.status, [0, 0])
    current = update.state
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    _record_array(checks, "solver_65", status, [0, 0])
    _record_array(
        checks, "solver_66", np.asarray(owners)[~np.asarray(current.mesh.cell_active)], -1
    )
    _record_array(
        checks, "solver_78", np.asarray(owners)[np.asarray(current.mesh.cell_active)], 0
    )
    return checks


def solver_missing_owner() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    bank = bank.at[1, 2].set(-1)
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    checks.setdefault("solver_63", []).append(
        bool(np.all(np.asarray(status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)))
    )
    return checks


def solver_incomplete_support() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    initial = eqx.tree_at(
        lambda state: state.mesh.cell_active,
        initial,
        initial.mesh.cell_active.at[1, 2].set(False),
    )
    bank = bank.at[1, 2].set(-1)
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    checks.setdefault("solver_63", []).append(
        bool(np.all(np.asarray(status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)))
    )
    return checks


def solver_contradictory_parent() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    initial = eqx.tree_at(
        lambda state: state.parents, initial, initial.parents.at[1, 2].set(2)
    )
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    checks.setdefault("solver_63", []).append(
        bool(np.all(np.asarray(status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)))
    )
    return checks


def solver_overlapping_source() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    layout, parts, initial, bank, current = _prepare_solver_source(checks)
    initial = eqx.tree_at(
        lambda state: state.mesh.cell_active,
        initial,
        initial.mesh.cell_active.at[1, 0].set(True),
    )
    bank = bank.at[1, 0].set(0)
    owners, status = inherit_solver_cell_owners(parts, current, initial, bank)
    checks.setdefault("solver_63", []).append(
        bool(np.all(np.asarray(status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)))
    )
    return checks
