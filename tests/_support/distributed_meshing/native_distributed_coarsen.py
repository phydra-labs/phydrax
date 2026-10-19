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
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
)


_CONTRACT_KEYS = (
    "coarsen_100",
    "coarsen_105",
    "coarsen_107",
    "coarsen_108",
    "coarsen_31",
    "coarsen_41",
    "coarsen_42",
    "coarsen_43",
    "coarsen_44",
    "coarsen_45",
    "coarsen_46",
    "coarsen_49",
    "coarsen_50",
    "coarsen_51",
    "coarsen_52",
    "coarsen_53",
    "coarsen_69",
    "coarsen_70",
    "coarsen_71",
    "coarsen_72",
    "coarsen_73",
    "coarsen_74",
    "coarsen_75",
    "coarsen_79",
    "coarsen_80",
    "coarsen_81",
    "coarsen_83",
    "coarsen_85",
    "coarsen_86",
    "coarsen_87",
    "coarsen_97",
    "coarsen_99",
)


def complete_family_coarsening(device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
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
    rows = np.asarray(
        [[0, 1, 2, 5], [0, 2, 3, 5], [0, 3, 4, 5], [0, 4, 1, 5]], dtype=np.int32
    )
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
        n = len(indices)
        return adaptive_simplex_state(
            layout,
            coordinates=points[ids],
            vertex_ids=ids,
            vertex_active=np.ones(len(ids), dtype=np.bool_),
            vertex_parents=np.full((len(ids), 2), -1, dtype=np.int32),
            vertex_protected=np.zeros(len(ids), dtype=np.bool_),
            cells=local,
            tuples=local,
            tags=np.full(n, 3, dtype=np.int32),
            blocks=np.zeros(n, dtype=np.int32),
            generations=np.zeros(n, dtype=np.int32),
            parents=np.full(n, -1, dtype=np.int32),
            children=np.full((n, 2), -1, dtype=np.int32),
            bisection_vertices=np.full(n, -1, dtype=np.int32),
            cell_ids=indices.astype(np.int64) + 10,
            cell_active=np.ones(n, dtype=np.bool_),
            cell_classes=np.full(n, 7, dtype=np.int32),
            facet_classes=np.zeros((n, 4), dtype=np.int32),
            protected_edges=np.empty((0, 2), dtype=np.int32),
            next_vertex_id=6,
            next_cell_id=14,
        )

    pieces = [
        piece(np.arange(p * 4 // count, (p + 1) * 4 // count, dtype=np.int32))
        for p in range(count)
    ]
    source = jax.tree_util.tree_map(lambda *v: jnp.stack(v), *pieces)
    parts = AdaptiveSimplexParts(
        jax.devices("cpu")[:count],
        neighbor_pairs=tuple(
            ((s, t) for s in range(count) for t in range(count) if s != t)
        ),
    )
    refined = refine_adaptive_simplex_parts(
        layout, parts, source, source.mesh.cell_active
    )
    checks.setdefault("coarsen_31", []).append(
        bool(np.all(np.asarray(refined.report.status) == 0))
    )
    split = refined.state
    parts = AdaptiveSimplexParts(
        jax.devices("cpu")[:count],
        neighbor_pairs=tuple(
            (route for p in range(count - 1) for route in ((p, p + 1), (p + 1, p)))
        ),
    )
    full = coarsen_adaptive_simplex_parts(layout, parts, split, split.mesh.cell_active)
    checks.setdefault("coarsen_41", []).append(
        bool(np.all(np.asarray(full.report.status) == 0))
    )
    checks.setdefault("coarsen_42", []).append(
        bool(np.all(np.asarray(full.report.operations) == 4))
    )
    checks.setdefault("coarsen_43", []).append(
        bool(np.all(np.asarray(full.report.vertices) == 1))
    )
    checks.setdefault("coarsen_44", []).append(
        bool(np.all(np.asarray(full.report.accepted) == 8))
    )
    _record_array(
        checks, "coarsen_45", full.state.mesh.cell_active, source.mesh.cell_active
    )
    _record_array(
        checks, "coarsen_46", full.state.mesh.vertex_active, source.mesh.vertex_active
    )
    for p in range(count):
        active = np.asarray(full.state.mesh.cell_active[p])
        _record_array(
            checks,
            "coarsen_49",
            full.state.mesh.cell_ids[p][active],
            source.mesh.cell_ids[p][active],
        )
        _record_array(
            checks,
            "coarsen_50",
            full.state.mesh.cells[p][active],
            source.mesh.cells[p][active],
        )
        _record_array(
            checks,
            "coarsen_51",
            full.state.mesh.coordinates[p][np.asarray(full.state.mesh.vertex_active[p])],
            source.mesh.coordinates[p][np.asarray(source.mesh.vertex_active[p])],
        )
    _record_array(checks, "coarsen_52", full.state.cursors, split.cursors)
    _record_array(checks, "coarsen_53", full.state.retired, split.mesh.cell_active)
    mid = int(np.flatnonzero(np.asarray(split.mesh.vertex_ids[-1]) == 6)[0])
    leaf = int(np.flatnonzero(np.asarray(split.mesh.cell_active[-1]))[0])
    for scenario in ("partial", "class", "facet", "protected"):
        state = split
        marks = split.mesh.cell_active
        if scenario == "partial":
            marks = marks.at[-1, leaf].set(False)
        if scenario == "class":
            state = eqx.tree_at(
                lambda s: s.cell_classes, state, state.cell_classes.at[-1, leaf].set(9)
            )
        if scenario == "facet":
            columns = np.flatnonzero(np.asarray(state.mesh.cells[-1, leaf]) != mid)
            equatorial = next(
                (
                    int(c)
                    for c in columns
                    if int(state.mesh.vertex_ids[-1, state.mesh.cells[-1, leaf, c]])
                    not in (0, 5)
                )
            )
            state = eqx.tree_at(
                lambda s: s.facet_classes,
                state,
                state.facet_classes.at[-1, leaf, equatorial].set(11),
            )
        if scenario == "protected":
            state = eqx.tree_at(
                lambda s: s.vertex_protected,
                state,
                state.vertex_protected.at[-1, mid].set(True),
            )
        update = coarsen_adaptive_simplex_parts(layout, parts, state, marks)
        checks.setdefault("coarsen_69", []).append(
            bool(np.all(np.asarray(update.report.status) == 0))
        )
        checks.setdefault("coarsen_70", []).append(
            bool(np.all(np.asarray(update.report.operations) == 0))
        )
        checks.setdefault("coarsen_71", []).append(
            bool(np.all(np.asarray(update.report.accepted) == 0))
        )
        _record_array(
            checks, "coarsen_72", update.state.mesh.cell_active, state.mesh.cell_active
        )
        _record_array(
            checks,
            "coarsen_73",
            update.state.mesh.vertex_active,
            state.mesh.vertex_active,
        )
        _record_array(checks, "coarsen_74", update.state.retired, state.retired)
        _record_array(checks, "coarsen_75", update.report.rejected, marks)
    bad = eqx.tree_at(
        lambda s: s.mesh.cells,
        split,
        split.mesh.cells.at[-1, 0].set(
            split.mesh.cells[-1, 0, jnp.asarray([1, 0, 2, 3], dtype=jnp.int32)]
        ),
    )
    failed = coarsen_adaptive_simplex_parts(layout, parts, bad, bad.mesh.cell_active)
    checks.setdefault("coarsen_79", []).append(
        bool(
            np.all(
                np.asarray(failed.report.status)
                & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
            )
        )
    )
    checks.setdefault("coarsen_80", []).append(
        bool(np.all(np.asarray(failed.report.operations) == 0))
    )
    _record_array(checks, "coarsen_81", failed.report.rejected, bad.mesh.cell_active)
    cleared = eqx.tree_at(
        lambda s: s.clocks, failed.state, failed.state.clocks.at[:, 2].set(0)
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(cleared), jax.tree_util.tree_leaves(bad), strict=True
    ):
        _record_array(checks, "coarsen_83", left, right)
    refused = coarsen_adaptive_simplex_parts(
        layout, parts, failed.state, failed.state.mesh.cell_active
    )
    checks.setdefault("coarsen_85", []).append(
        bool(np.all(np.asarray(refused.report.failed)))
    )
    checks.setdefault("coarsen_86", []).append(
        bool(np.all(np.asarray(refused.report.iterations) == 0))
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(refused.state),
        jax.tree_util.tree_leaves(failed.state),
        strict=True,
    ):
        _record_array(checks, "coarsen_87", left, right)
    offset = jnp.arange(count, dtype=jnp.int64)[:, None] * 100
    independent = eqx.tree_at(
        lambda s: (s.mesh.vertex_ids, s.mesh.cell_ids, s.cursors),
        source,
        (
            jnp.where(source.mesh.vertex_ids >= 0, source.mesh.vertex_ids + offset, -1),
            jnp.where(source.mesh.cell_ids >= 0, source.mesh.cell_ids + offset, -1),
            source.cursors.at[:, 2].set(count * 100 + 6).at[:, 3].set(count * 100 + 14),
        ),
    )
    isolated = AdaptiveSimplexParts(jax.devices("cpu")[:count], neighbor_pairs=())
    first = refine_adaptive_simplex_parts(
        layout, isolated, independent, independent.mesh.cell_active
    )
    deep = refine_adaptive_simplex_parts(
        layout, isolated, first.state, first.state.mesh.cell_active
    )
    checks.setdefault("coarsen_97", []).append(
        bool(np.all(np.asarray(deep.report.status) == 0))
    )
    limited = coarsen_adaptive_simplex_parts(
        layout, isolated, deep.state, deep.state.mesh.cell_active
    )
    checks.setdefault("coarsen_99", []).append(
        bool(
            np.all(
                np.asarray(limited.report.status) == int(AdaptiveSimplexStatus.PASS_LIMIT)
            )
        )
    )
    checks.setdefault("coarsen_100", []).append(
        bool(np.all(np.asarray(limited.report.iterations) == 1))
    )
    finished = limited
    for _ in range(8):
        finished = coarsen_adaptive_simplex_parts(
            layout, isolated, finished.state, finished.state.mesh.cell_active
        )
        if np.all(np.asarray(finished.report.status) == 0):
            break
        checks.setdefault("coarsen_105", []).append(
            bool(
                np.all(
                    np.asarray(finished.report.status)
                    == int(AdaptiveSimplexStatus.PASS_LIMIT)
                )
            )
        )
    else:
        raise AssertionError("bounded isolated forest did not finish")
    _record_array(
        checks,
        "coarsen_107",
        finished.state.mesh.cell_active,
        independent.mesh.cell_active,
    )
    _record_array(
        checks,
        "coarsen_108",
        finished.state.mesh.vertex_active,
        independent.mesh.vertex_active,
    )
    print(
        f"COARSEN {count} devices: source IDs/arrays restored; remote vetoes; unique midpoint; retired slots; rollback/terminal; truthful pass limit"
    )
    return checks
