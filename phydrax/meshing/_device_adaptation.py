#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device adaptation epochs of certified simplex meshes.

`prepare_adaptive_simplex` binds one certified source result (and the
`BisectionHierarchy` of earlier adaptations) to a fixed-capacity device state:
the labels, closure, and coarsening families are exactly those of the host
bisection route. Refinement and coarsening then run as compiled device calls
(`refine_adaptive_simplex`, `coarsen_adaptive_simplex`) any number of times, and
solvers consume the masked layout directly. `commit_adaptive_simplex` performs
one device-to-host transfer and builds the canonical target, its complete
lineage, the sparse P1 transfer, and the next hierarchy through the same host
edit assembly as the host route, so a device epoch and a host adaptation with
the same marks commit byte-identical meshes.

Cell lineage relative to the epoch source follows bisection-tree containment:
every cell is a node ``(root, depth, path)`` of its root's binary bisection
tree, so a target cell inside a source cell is refined from it and a source
cell inside a target cell is coarsened into it, including cells re-created
after a coarsening inside one epoch.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    adaptive_simplex_bucket,
    adaptive_simplex_state,
    AdaptiveSimplexCounter,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex,
    refine_adaptive_simplex,
)
from ._adaptation import (
    _adaptation_result,
    _finalize_native,
    _RouteOutcome,
    _unchanged,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    prepare_mesh_adaptation,
    PreparedMeshAdaptation,
)
from ._bisection import (
    _bound_start,
    _Cells,
    _Coarsening,
    _edit,
    _Front,
    _global_vertices,
    _Growth,
    _labelled_start,
    _prepared_request,
    _prepared_source,
    _Records,
    _Refinement,
    _resolved_stencil,
    _reverse_supports,
    _SHIFT,
    _Source,
    _Start,
    _target_hierarchy,
    BisectionEvidence,
    BisectionHierarchy,
)
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._device_status import (
    DeviceEpoch,
    require_applied_status,
    require_committed_status,
)
from ._lineage import MeshTransitionKind
from ._result import CellMeshingResult
from ._topology_edit import key_rows


# Bisection-tree paths are int64 bit strings below their root.
_MAXIMUM_DEPTH = 62


class _Anchor(NamedTuple):
    """Host description of the epoch source shared by preparation and commit."""

    source: _Source
    start: _Start
    cell_ids: np.ndarray
    slot_origins: np.ndarray
    root_origins: np.ndarray
    anchor_slots: np.ndarray


@final
class PreparedAdaptiveSimplex(StrictModule, NonTrainableState):
    """One certified source epoch bound to a device capacity bucket.

    ``adaptation`` is the prepared transaction (source, policy, resolved
    protection and organization); ``layout`` the static compile identity;
    ``state`` the initial device state. Every device call with this layout
    reuses one compiled executable, whatever the source topology or cycle.
    """

    adaptation: PreparedMeshAdaptation
    layout: AdaptiveSimplexLayout
    state: AdaptiveSimplexState
    anchor: _Anchor
    source_topology_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        adaptation: PreparedMeshAdaptation,
        layout: AdaptiveSimplexLayout,
        state: AdaptiveSimplexState,
        anchor: _Anchor,
        /,
    ):
        self.adaptation = adaptation
        self.layout = layout
        self.state = state
        self.anchor = anchor
        self.source_topology_id = adaptation.source.mesh.topology_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-adaptive-simplex",
                "adaptation": adaptation.prepared_id,
                "layout": layout.signature_id,
            }
        )

    def cell_marks(self, cell_ids: np.ndarray, /) -> np.ndarray:
        """Slot mask of the prepared cells descending from the given source cells."""

        identifiers = np.asarray(cell_ids, dtype=np.int64)
        marks = np.zeros((self.layout.cell_capacity,), dtype=np.bool_)
        origins = self.anchor.slot_origins
        marks[: origins.size] = np.isin(origins, identifiers) & (origins >= 0)
        return marks


def _record_generations(active_generations: np.ndarray, children: np.ndarray, /):
    """Generations of records: one less than their first child, bottom-up."""

    generations = active_generations.copy()
    known = generations >= 0
    records = np.flatnonzero(children[:, 0] >= 0)
    while not np.all(known[records]):
        ready = records[~known[records] & known[children[records, 0]]]
        if ready.size == 0:
            raise ValueError("The bisection forest has an unreachable record.")
        generations[ready] = generations[children[ready, 0]] - 1
        known[ready] = True
    return generations


def _facet_classes(
    rows: np.ndarray, facet_keys: np.ndarray, facet_classes: np.ndarray, /
) -> np.ndarray:
    """Source facet class + 1 of every local facet; 0 for facets new to the source."""

    count, width = rows.shape
    columns = np.asarray([[j for j in range(width) if j != i] for i in range(width)])
    keys = np.sort(rows[:, columns], axis=2).reshape((-1, width - 1))
    found = key_rows(np.sort(facet_keys, axis=1), keys)
    classes = np.where(found >= 0, facet_classes[np.maximum(found, 0)] + 1, 0)
    return classes.reshape((count, width)).astype(np.int32)


def _forest(start: _Start, source: _Source, request, /):
    """Active cells and records of the start, merged into ID-ordered slots."""

    cells, records = start.front.cells, start.records
    active_count, record_count = cells.ids.size, records.parent_ids.size
    ids = np.concatenate((cells.ids, records.parent_ids))
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    count = ids.size

    def merged(active_values, record_values):
        return np.concatenate((active_values, record_values))[order]

    active = merged(
        np.ones((active_count,), np.bool_), np.zeros((record_count,), np.bool_)
    )
    record_slots = np.searchsorted(ids, records.parent_ids)
    children = np.full((count, 2), -1, dtype=np.int64)
    children[record_slots] = np.searchsorted(ids, records.child_ids)
    parents = np.full((count,), -1, dtype=np.int64)
    parents[children[record_slots, 0]] = 2 * record_slots
    parents[children[record_slots, 1]] = 2 * record_slots + 1
    bisection_vertices = np.full((count,), -1, dtype=np.int64)
    bisection_vertices[record_slots] = records.vertices
    origins = merged(cells.origins, np.full((record_count,), -1, np.int64))
    source_rows = np.searchsorted(source.cells.ids, np.maximum(origins, 0))
    cell_classes = np.where(active, request.cell_classes[source_rows], 0)
    rows = merged(cells.rows, records.rows)
    facet_classes = _facet_classes(rows, request.facet_keys, request.facet_classes)
    return {
        "ids": ids,
        "active": active,
        "rows": rows,
        "tuples": merged(cells.tuples, records.tuples),
        "tags": merged(cells.tags, records.tags),
        "blocks": merged(cells.blocks, records.blocks),
        "generations": _record_generations(
            merged(cells.generations, np.full((record_count,), -1, np.int64)), children
        ),
        "parents": parents,
        "children": children,
        "bisection_vertices": bisection_vertices,
        "origins": origins,
        "cell_classes": np.where(active, cell_classes, 0),
        "facet_classes": np.where(active[:, None], facet_classes, 0),
    }


def _vertices(start: _Start, source: _Source, request, /):
    """Source vertices by global ID, then vertices issued by the start."""

    count, dimension = source.vertex_ids.size, source.dimension
    growth = start.front.growth
    sources, weights = _resolved_stencil(growth, count, dimension)
    created = np.sum(
        np.where(
            (sources >= 0)[..., None],
            source.coordinates[np.maximum(sources, 0)] * weights[..., None],
            0.0,
        ),
        axis=1,
    )
    issued = count + np.arange(growth.levels.size, dtype=np.int64)
    total = count + growth.levels.size
    parents = np.full((total, 2), -1, dtype=np.int64)
    records = start.records
    parents[records.vertices, 0] = records.tuples[:, 0]
    parents[records.vertices, 1] = records.tuples[
        np.arange(records.tags.size), records.tags
    ]
    protected = np.zeros((total,), dtype=np.bool_)
    protected[:count] = request.protected_vertices
    return {
        "coordinates": np.concatenate((source.coordinates, created)),
        "ids": np.concatenate(
            (source.vertex_ids, _global_vertices(issued, source, start.next_vertex))
        ),
        "parents": parents,
        "protected": protected,
    }


def _prepared_simplex(adaptation: PreparedMeshAdaptation, /) -> PreparedAdaptiveSimplex:
    policy = adaptation.policy
    device = policy.device_policy
    constraints = adaptation.constraints
    mesh = adaptation.source.mesh
    source = _prepared_source(mesh)
    empty = np.zeros((0,), dtype=np.int64)
    request = _prepared_request(
        source,
        empty,
        empty,
        constraints.protected_edge_keys,
        constraints.protected_vertex_ids,
        constraints.cell_classes,
        constraints.facet_classes,
        policy.maximum_closure_iterations,
    )
    hierarchy = adaptation.request.hierarchy
    start = (
        _labelled_start(source, request, policy.compatibility)
        if hierarchy is None
        else _bound_start(source, request, hierarchy)
    )
    forest = _forest(start, source, request)
    vertices = _vertices(start, source, request)
    vertex_capacity, cell_capacity = device.capacities(
        vertices["ids"].size, forest["ids"].size
    )
    protected = request.protected_codes
    layout = AdaptiveSimplexLayout(
        source.dimension,
        mesh.ambient_dimension,
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        protected_edge_capacity=adaptive_simplex_bucket(protected.size, 1.0),
        maximum_closure_iterations=policy.maximum_closure_iterations,
        maximum_coarsening_passes=device.maximum_coarsening_passes,
    )
    state = adaptive_simplex_state(
        layout,
        coordinates=vertices["coordinates"],
        vertex_ids=vertices["ids"],
        vertex_active=np.ones(vertices["ids"].shape, dtype=np.bool_),
        vertex_parents=vertices["parents"],
        vertex_protected=vertices["protected"],
        cells=forest["rows"],
        tuples=forest["tuples"],
        tags=forest["tags"],
        blocks=forest["blocks"],
        generations=forest["generations"],
        parents=forest["parents"],
        children=forest["children"],
        bisection_vertices=forest["bisection_vertices"],
        cell_ids=forest["ids"],
        cell_active=forest["active"],
        cell_classes=forest["cell_classes"],
        facet_classes=forest["facet_classes"],
        protected_edges=np.stack((protected // _SHIFT, protected % _SHIFT), axis=1),
        next_vertex_id=start.next_vertex + start.front.growth.levels.size,
        next_cell_id=start.front.next_cell,
    )
    origins = forest["origins"]
    roots = np.where(origins != forest["ids"], origins, -1)
    anchor = _Anchor(
        source,
        start,
        forest["ids"],
        origins,
        np.where(forest["parents"] < 0, roots, -1),
        np.flatnonzero(np.isin(forest["ids"], source.cells.ids)),
    )
    return PreparedAdaptiveSimplex(adaptation, layout, state, anchor)


def prepare_adaptive_simplex(
    source: CellMeshingResult,
    /,
    *,
    policy: MeshAdaptationPolicy,
    hierarchy: BisectionHierarchy | None = None,
) -> PreparedAdaptiveSimplex:
    """Validate, label, and pad one certified simplex source into a device state.

    ``policy.route`` must be DEVICE_BISECTION; ``policy.device_policy`` fixes the
    capacity bucket. Labels are the host longest-edge Maubach labels (or the
    supplied hierarchy's); incompatible labellings are rejected unless the
    policy requests the explicit uniform barycentric refinement. Protection and
    organization classes are resolved exactly as for the host route.
    """

    if not isinstance(policy, MeshAdaptationPolicy):
        raise TypeError("policy must be MeshAdaptationPolicy.")
    if policy.route is not MeshAdaptationRoute.DEVICE_BISECTION:
        raise ValueError("Adaptive simplex epochs require the DEVICE_BISECTION route.")
    adaptation = prepare_mesh_adaptation(
        source, MarkedMeshAdaptation(hierarchy=hierarchy), policy=policy
    )
    return _prepared_simplex(adaptation)


def _node_keys(parents: np.ndarray, generations: np.ndarray, /) -> np.ndarray:
    """``(root, depth, path)`` of every slot in its root's bisection tree."""

    count = parents.size
    root = np.arange(count, dtype=np.int64)
    depth = np.zeros((count,), dtype=np.int64)
    path = np.zeros((count,), dtype=np.int64)
    for generation in np.unique(generations):
        chosen = np.flatnonzero((generations == generation) & (parents >= 0))
        parent, ordinal = parents[chosen] // 2, parents[chosen] % 2
        root[chosen] = root[parent]
        depth[chosen] = depth[parent] + 1
        path[chosen] = 2 * path[parent] + ordinal
    if np.any(depth > _MAXIMUM_DEPTH):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"Device bisection trees deeper than {_MAXIMUM_DEPTH} cannot be committed.",
            stage="device-commit",
        )
    return np.stack((root, depth, path), axis=1)


def _ancestors(keys: np.ndarray, queries: np.ndarray, table: np.ndarray, /, *, strict):
    """Row in ``table`` of the nearest (strict) ancestor-or-self of each query."""

    found = np.full((queries.shape[0],), -1, dtype=np.int64)
    if table.shape[0] == 0 or queries.shape[0] == 0:
        return found
    nodes = keys[queries]
    for shift in range(1 if strict else 0, int(np.max(nodes[:, 1])) + 1):
        open_ = (found < 0) & (nodes[:, 1] >= shift)
        ancestor = np.stack(
            (nodes[:, 0], nodes[:, 1] - shift, nodes[:, 2] >> shift), axis=1
        )
        rows = key_rows(table, ancestor[open_])
        found[np.flatnonzero(open_)] = rows
    return found


class _Lineage(NamedTuple):
    origins: np.ndarray
    removed_ids: np.ndarray
    link_targets: np.ndarray
    source_cells: np.ndarray


class _HostEpoch(NamedTuple):
    """Host arrays of one device epoch (single part or merged parts)."""

    cell_ids: np.ndarray
    cell_active: np.ndarray
    cells: np.ndarray
    tuples: np.ndarray
    tags: np.ndarray
    blocks: np.ndarray
    generations: np.ndarray
    parents: np.ndarray
    children: np.ndarray
    bisection_vertices: np.ndarray
    retired: np.ndarray
    refine_rejected: np.ndarray
    coarsen_marked: np.ndarray
    vertex_ids: np.ndarray
    vertex_active: np.ndarray
    coordinates: np.ndarray
    vertex_parents: np.ndarray
    vertex_levels: np.ndarray
    vertex_removal: np.ndarray
    cursors: np.ndarray
    counters: np.ndarray
    flags: int


def _host_epoch(state: AdaptiveSimplexState, /) -> _HostEpoch:
    """One device-to-host transfer of a single-part state."""

    host = jax.device_get(state)
    mesh = host.mesh
    return _HostEpoch(
        np.asarray(mesh.cell_ids, dtype=np.int64),
        np.asarray(mesh.cell_active, dtype=np.bool_),
        np.asarray(mesh.cells, dtype=np.int64),
        np.asarray(host.tuples, dtype=np.int64),
        np.asarray(host.tags, dtype=np.int64),
        np.asarray(host.blocks, dtype=np.int64),
        np.asarray(host.generations, dtype=np.int64),
        np.asarray(host.parents, dtype=np.int64),
        np.asarray(host.children, dtype=np.int64),
        np.asarray(host.bisection_vertices, dtype=np.int64),
        np.asarray(host.retired, dtype=np.bool_),
        np.asarray(host.refine_rejected, dtype=np.bool_),
        np.asarray(host.coarsen_marked, dtype=np.bool_),
        np.asarray(mesh.vertex_ids, dtype=np.int64),
        np.asarray(mesh.vertex_active, dtype=np.bool_),
        # Exact widening: host predicates resolve the device coordinates.
        np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(host.vertex_parents, dtype=np.int64),
        np.asarray(host.vertex_levels, dtype=np.int64),
        np.asarray(host.vertex_removal, dtype=np.int64),
        np.asarray(host.cursors, dtype=np.int64),
        np.asarray(host.counters, dtype=np.int64),
        int(np.asarray(host.status_flags)),
    )


def _cell_lineage(anchor: _Anchor, host: _HostEpoch, count: int, /):
    """Source cell containing every slot, and the target cell of removed sources."""

    ids = np.asarray(host.cell_ids)[:count]
    active = np.asarray(host.cell_active)[:count]
    keys = _node_keys(
        np.asarray(host.parents, dtype=np.int64)[:count],
        np.asarray(host.generations, dtype=np.int64)[:count],
    )
    anchors = anchor.anchor_slots
    slots = np.arange(count, dtype=np.int64)
    container = _ancestors(keys, slots, keys[anchors], strict=False)
    roots = np.full((count,), -1, dtype=np.int64)
    roots[: anchor.root_origins.size] = anchor.root_origins
    # A trailing -1 absorbs slots without an anchor ancestor (row -1).
    anchor_ids = np.concatenate((ids[anchors], np.full((1,), -1, dtype=np.int64)))
    inside = np.where(container >= 0, anchor_ids[container], roots[keys[:, 0]])
    targets = np.flatnonzero(active)
    covering = np.zeros((anchors.size,), dtype=np.bool_)
    covering[container[targets][container[targets] >= 0]] = True
    removed = anchors[~active[anchors] & ~covering]
    within = _ancestors(keys, removed, keys[targets], strict=True)
    if np.any(within < 0):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "A removed source cell lies in no target cell of the device epoch.",
            stage="device-commit",
        )
    return _Lineage(
        np.where(inside >= 0, inside, ids),
        ids[removed],
        ids[targets[within]],
        inside,
    )


def _growth(anchor: _Anchor, host: _HostEpoch, count: int, /) -> _Growth:
    """Growth of the start followed by the device-issued midpoints."""

    initial = anchor.start.front.growth
    width = anchor.source.dimension + 1
    first = anchor.source.vertex_ids.size + initial.levels.size
    issued = np.arange(first, count, dtype=np.int64)
    parents = np.full((issued.size, width), -1, dtype=np.int64)
    parents[:, :2] = np.asarray(host.vertex_parents, dtype=np.int64)[issued]
    weights = np.zeros((issued.size, width), dtype=np.float64)
    weights[:, :2] = 0.5
    levels = np.asarray(host.vertex_levels, dtype=np.int64)[issued]
    return _Growth(
        np.concatenate((initial.parents, parents)),
        np.concatenate((initial.weights, weights)),
        np.concatenate((initial.levels, levels)),
    )


def _supports(anchor: _Anchor, host: _HostEpoch, /) -> np.ndarray:
    """Surviving vertices spanning each removed source vertex (reverse passes)."""

    count = anchor.source.vertex_ids.size
    removal = np.asarray(host.vertex_removal, dtype=np.int64)[:count]
    parents = np.asarray(host.vertex_parents, dtype=np.int64)[:count]
    steps = []
    for index in np.unique(removal[removal >= 0]):
        vertices = np.flatnonzero(removal == index)
        steps.append((vertices, parents[vertices, 0], parents[vertices, 1]))
    return _reverse_supports(steps, count, anchor.source.dimension)


class _Committed(NamedTuple):
    edit: object
    hierarchy: BisectionHierarchy
    evidence: BisectionEvidence
    refined: bool
    coarsened: bool
    partial: bool
    pass_limited: bool


def _committed(prepared: PreparedAdaptiveSimplex, host: _HostEpoch, /):
    anchor = prepared.anchor
    source, start = anchor.source, anchor.start
    vertex_count = int(host.cursors[0])
    cell_count = int(host.cursors[1])
    active = np.asarray(host.cell_active)[:cell_count]
    retired = np.asarray(host.retired)[:cell_count]
    ids = np.asarray(host.cell_ids)[:cell_count]
    rows = np.asarray(host.cells, dtype=np.int64)[:cell_count]
    tuples = np.asarray(host.tuples, dtype=np.int64)[:cell_count]
    tags = np.asarray(host.tags, dtype=np.int64)[:cell_count]
    blocks = np.asarray(host.blocks, dtype=np.int64)[:cell_count]
    generations = np.asarray(host.generations, dtype=np.int64)[:cell_count]
    lineage = _cell_lineage(anchor, host, cell_count)
    target = np.flatnonzero(active)
    cells = _Cells(
        ids[target],
        rows[target],
        tuples[target],
        tags[target],
        blocks[target],
        lineage.origins[target],
        generations[target],
    )
    record = np.flatnonzero(~active & ~retired)
    children = np.asarray(host.children, dtype=np.int64)[:cell_count][record]
    records = _Records(
        ids[record],
        blocks[record],
        rows[record],
        tuples[record],
        tags[record],
        ids[children],
        np.asarray(host.bisection_vertices, dtype=np.int64)[:cell_count][record],
    )
    growth = _growth(anchor, host, vertex_count)
    removed_vertices = np.flatnonzero(~np.asarray(host.vertex_active)[:vertex_count])
    counters = np.asarray(host.counters, dtype=np.int64)
    rejected_refinements = np.unique(
        lineage.origins[np.asarray(host.refine_rejected)[:cell_count]]
    )
    marked = np.asarray(host.coarsen_marked)[:cell_count] & ~retired
    rejected_coarsenings = ids[marked]
    prepared_slots = anchor.cell_ids.size
    restored = np.flatnonzero(active[:prepared_slots] & (anchor.slot_origins < 0))
    coarsening = _Coarsening(
        _Cells(
            ids[restored],
            rows[restored],
            tuples[restored],
            tags[restored],
            blocks[restored],
            ids[restored],
            generations[restored],
        ),
        lineage.removed_ids,
        lineage.link_targets,
        np.zeros((0,), dtype=np.int64),
        removed_vertices,
        _supports(anchor, host),
        int(counters[AdaptiveSimplexCounter.COARSENING_PASSES]),
        rejected_coarsenings,
    )
    front = _Front(
        cells,
        growth,
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        source.vertex_ids.size,
        int(host.cursors[3]),
    )
    refinement = _Refinement(
        front,
        records,
        rejected_refinements,
        int(counters[AdaptiveSimplexCounter.ADMISSIBILITY_TESTS]),
        int(counters[AdaptiveSimplexCounter.CLOSURE_ITERATIONS]),
    )
    edit, retired_tables = _edit(source, start, refinement, coarsening, cells)
    hierarchy = _target_hierarchy(source, start, cells, records, retired_tables, front)
    evidence = BisectionEvidence(
        requested_refinements=int(counters[AdaptiveSimplexCounter.REQUESTED_REFINEMENTS]),
        accepted_refinements=int(counters[AdaptiveSimplexCounter.ACCEPTED_REFINEMENTS]),
        rejected_refinement_ids=rejected_refinements,
        admissibility_tests=int(counters[AdaptiveSimplexCounter.ADMISSIBILITY_TESTS]),
        bisections=int(counters[AdaptiveSimplexCounter.BISECTIONS]),
        closure_iterations=int(counters[AdaptiveSimplexCounter.CLOSURE_ITERATIONS]),
        created_vertices=growth.levels.size,
        maximum_generation=int(np.max(cells.generations)),
        initially_compatible=start.compatible,
        incompatible_facets=start.incompatible,
        uniform_refinement_applied=start.uniform,
        requested_coarsenings=int(counters[AdaptiveSimplexCounter.REQUESTED_COARSENINGS]),
        coarsened_vertices=int(counters[AdaptiveSimplexCounter.COARSENED_VERTICES]),
        coarsening_passes=int(counters[AdaptiveSimplexCounter.COARSENING_PASSES]),
        restored_cells=restored.size,
        rejected_coarsening_ids=rejected_coarsenings,
    )
    flags = int(host.flags)
    return _Committed(
        edit,
        hierarchy,
        evidence,
        evidence.bisections > 0 or start.uniform,
        evidence.coarsened_vertices > 0,
        rejected_refinements.size > 0 or rejected_coarsenings.size > 0,
        bool(flags & AdaptiveSimplexStatus.PASS_LIMIT),
    )


def _outcome(prepared: PreparedAdaptiveSimplex, host: _HostEpoch, /):
    require_committed_status(
        host.flags,
        host.coordinates,
        host.cells[host.cell_active],
        DeviceEpoch.BISECTION,
    )
    committed = _committed(prepared, host)
    adaptation = prepared.adaptation
    if not committed.refined and not committed.coarsened:
        status = (
            MeshAdaptationStatus.PASS_LIMIT
            if committed.pass_limited
            else MeshAdaptationStatus.PARTIAL
            if committed.partial
            else MeshAdaptationStatus.UNCHANGED
        )
        return _unchanged(
            adaptation, committed.evidence, committed.hierarchy, status=status
        )
    match (committed.refined, committed.coarsened):
        case (True, False):
            kind = MeshTransitionKind.REFINE
        case (False, True):
            kind = MeshTransitionKind.COARSEN
        case _:
            kind = MeshTransitionKind.REMESH
    native = _finalize_native(
        adaptation, committed.edit, kind, conservative=not committed.coarsened
    )
    if committed.pass_limited:
        status = MeshAdaptationStatus.PASS_LIMIT
    elif committed.partial:
        status = MeshAdaptationStatus.PARTIAL
    else:
        status = MeshAdaptationStatus.COMPLETE
    return _RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        committed.evidence,
        committed.hierarchy,
    )


def _checked_state(prepared: PreparedAdaptiveSimplex, state: AdaptiveSimplexState, /):
    if not isinstance(prepared, PreparedAdaptiveSimplex):
        raise TypeError("prepared must be PreparedAdaptiveSimplex.")
    if not isinstance(state, AdaptiveSimplexState):
        raise TypeError("state must be AdaptiveSimplexState.")
    if state.mesh.signature_id != prepared.layout.mesh_signature_id:
        raise ValueError("The state does not belong to this prepared epoch.")


def commit_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex, state: AdaptiveSimplexState, /
) -> MeshAdaptationResult:
    """Commit one device epoch: one transfer, canonical target, lineage, evidence.

    The target is assembled, organization inherited by exact IDs, certified,
    and bound to its `CellMeshTransition`, `MeshLineage`, and sparse P1
    transfer exactly as the host bisection route does; the result's
    ``hierarchy`` prepares the next epoch. The epoch's cumulative
    ``status_flags`` decide acceptance: a terminal flag raises its
    `MeshingFailure` (capacity or closure bound: RESOURCE_EXHAUSTED, protected
    conflict: INVALID_SPECIFICATION, invalid geometry: QUALITY_REJECTED),
    NEEDS_HOST_RESOLUTION requires every committed cell to be certified
    positively oriented by exact host predicates (else QUALITY_REJECTED), and
    a pass-limited coarsening commits with status PASS_LIMIT.
    """

    _checked_state(prepared, state)
    started = time.monotonic()
    outcome = _outcome(prepared, _host_epoch(state))
    return _adaptation_result(prepared.adaptation, outcome, started)


def _execute_device_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    """One refine-then-coarsen device epoch of a marked request, then commit."""

    request = prepared.request
    if request.coarsen_cell_ids.size and request.hierarchy is None:
        raise ValueError(
            "Coarsening requires the BisectionHierarchy of a previous bisection."
        )
    simplex = _prepared_simplex(prepared)
    layout = simplex.layout
    refined = refine_adaptive_simplex(
        layout, simplex.state, simplex.cell_marks(np.asarray(request.refine_cell_ids))
    )
    require_applied_status(int(refined.report.status), DeviceEpoch.BISECTION)
    coarsened = coarsen_adaptive_simplex(
        layout,
        refined.state,
        simplex.cell_marks(np.asarray(request.coarsen_cell_ids)),
    )
    return _outcome(simplex, _host_epoch(coarsened.state))


@final
class PartitionedAdaptiveSimplex(StrictModule, NonTrainableState):
    """Part-sharded view of one prepared epoch: each part refines its owned cells.

    Ownership is the policy's `MeshDistribution` (space-filling-curve, graph,
    or provider ownership of the source cells); cells created by uniform
    subdivision follow their source cell. ``layout`` is the per-part capacity
    bucket, ``parts`` the device mesh, and ``states`` the stacked local states.
    Repartitioning happens only at commit, through the distribution transition
    of the adaptation result.
    """

    prepared: PreparedAdaptiveSimplex
    layout: AdaptiveSimplexLayout
    parts: AdaptiveSimplexParts
    states: AdaptiveSimplexState
    slot_origins: np.ndarray
    partitioned_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedAdaptiveSimplex,
        layout: AdaptiveSimplexLayout,
        parts: AdaptiveSimplexParts,
        states: AdaptiveSimplexState,
        slot_origins: np.ndarray,
        /,
    ):
        self.prepared = prepared
        self.layout = layout
        self.parts = parts
        self.states = states
        self.slot_origins = slot_origins
        self.partitioned_id = canonical_fingerprint(
            {
                "kind": "partitioned-adaptive-simplex",
                "prepared": prepared.prepared_id,
                "layout": layout.signature_id,
                "distribution": prepared.adaptation.policy.distribution.distribution_id,
            }
        )

    def cell_marks(self, cell_ids: np.ndarray, /) -> np.ndarray:
        """Per-part slot masks of the prepared cells descending from source cells."""

        identifiers = np.asarray(cell_ids, dtype=np.int64)
        return np.isin(self.slot_origins, identifiers) & (self.slot_origins >= 0)


def _part_arrays(base: _HostEpoch, prepared: PreparedAdaptiveSimplex, slots, /):
    """Owned cells of one part and their vertices, in global-ID slot order."""

    vertices = np.unique(base.cells[slots])
    local = np.full((base.vertex_ids.size,), -1, dtype=np.int64)
    local[vertices] = np.arange(vertices.size)
    codes = np.asarray(prepared.state.protected_codes, dtype=np.int64)
    codes = codes[codes != np.iinfo(np.int64).max]
    capacity = prepared.layout.vertex_capacity
    ends = local[np.stack((codes // capacity, codes % capacity), axis=1)]
    return {
        "vertices": vertices,
        "cells": local[base.cells[slots]],
        "tuples": local[base.tuples[slots]],
        "protected_edges": ends[np.all(ends >= 0, axis=1)],
    }


def partition_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> PartitionedAdaptiveSimplex:
    """Split one prepared epoch by its distribution into stacked per-part states.

    Each part holds its owned active cells and their vertices with the global
    IDs and cursors of the epoch; ``devices`` default to the first devices of
    the process. Records of earlier bisections stay with the prepared epoch
    and rejoin the parts at commit.
    """

    if not isinstance(prepared, PreparedAdaptiveSimplex):
        raise TypeError("prepared must be PreparedAdaptiveSimplex.")
    distribution = prepared.adaptation.policy.distribution
    if distribution is None:
        raise ValueError("Partitioned device epochs require a policy distribution.")
    part_count = distribution.partition.part_count
    chosen = tuple(jax.devices()[:part_count] if devices is None else devices)
    if len(chosen) != part_count:
        raise ValueError(f"The distribution needs {part_count} devices.")
    base = _host_epoch(prepared.state)
    count = prepared.anchor.cell_ids.size
    origins = prepared.anchor.slot_origins
    native = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    owners = np.asarray(distribution.partition.cell_owner, dtype=np.int64)
    order = np.argsort(native, kind="stable")
    position = np.searchsorted(native[order], np.maximum(origins, 0))
    slot_owner = np.where(base.cell_active[:count], owners[order][position], -1)
    owned = tuple(np.flatnonzero(slot_owner == part) for part in range(part_count))
    pieces = tuple(_part_arrays(base, prepared, slots) for slots in owned)
    policy = prepared.adaptation.policy
    vertex_capacity, cell_capacity = policy.device_policy.capacities(
        max(piece["vertices"].size for piece in pieces),
        max(slots.size for slots in owned),
    )
    layout = AdaptiveSimplexLayout(
        prepared.layout.dimension,
        prepared.layout.ambient_dimension,
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        protected_edge_capacity=prepared.layout.protected_edge_capacity,
        maximum_closure_iterations=prepared.layout.maximum_closure_iterations,
        maximum_coarsening_passes=prepared.layout.maximum_coarsening_passes,
    )
    coordinates = np.asarray(prepared.state.mesh.coordinates)
    classes = np.asarray(prepared.state.cell_classes)
    facet_classes = np.asarray(prepared.state.facet_classes)
    protected = np.asarray(prepared.state.vertex_protected)
    states = []
    slot_origins = np.full((part_count, cell_capacity), -1, dtype=np.int64)
    for part, (slots, piece) in enumerate(zip(owned, pieces, strict=True)):
        vertices = piece["vertices"]
        size = slots.size
        states.append(
            adaptive_simplex_state(
                layout,
                coordinates=coordinates[vertices],
                vertex_ids=base.vertex_ids[vertices],
                vertex_active=np.ones(vertices.shape, dtype=np.bool_),
                vertex_parents=np.full((vertices.size, 2), -1, dtype=np.int64),
                vertex_protected=protected[vertices],
                cells=piece["cells"],
                tuples=piece["tuples"],
                tags=base.tags[slots],
                blocks=base.blocks[slots],
                generations=base.generations[slots],
                parents=np.full((size,), -1, dtype=np.int64),
                children=np.full((size, 2), -1, dtype=np.int64),
                bisection_vertices=np.full((size,), -1, dtype=np.int64),
                cell_ids=base.cell_ids[slots],
                cell_active=np.ones((size,), dtype=np.bool_),
                cell_classes=classes[slots],
                facet_classes=facet_classes[slots],
                protected_edges=piece["protected_edges"],
                next_vertex_id=int(base.cursors[2]),
                next_cell_id=int(base.cursors[3]),
            )
        )
        slot_origins[part, :size] = origins[slots]
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
    return PartitionedAdaptiveSimplex(
        prepared, layout, AdaptiveSimplexParts(chosen), stacked, slot_origins
    )


def _merged_epoch(
    partitioned: PartitionedAdaptiveSimplex, parts: AdaptiveSimplexState, /
) -> _HostEpoch:
    """Merge part states into the prepared epoch's slots (IDs issue the slots)."""

    prepared = partitioned.prepared
    base = _host_epoch(prepared.state)
    host = jax.device_get(parts)
    vertex_count, cell_count = int(base.cursors[0]), int(base.cursors[1])
    vertex_base, cell_base = int(base.cursors[2]), int(base.cursors[3])
    cursors = np.asarray(host.cursors, dtype=np.int64)
    vertex_total = vertex_count + int(cursors[0, 2]) - vertex_base
    cell_total = cell_count + int(cursors[0, 3]) - cell_base

    def grown(values, total, fill):
        result = np.full((total,) + values.shape[1:], fill, dtype=values.dtype)
        result[: min(total, values.shape[0])] = values[:total]
        return result

    fields = base._asdict()
    arrays = {
        name: grown(fields[name][:cell_count], cell_total, fill)
        for name, fill in (
            ("cell_ids", -1),
            ("cell_active", False),
            ("cells", 0),
            ("tuples", 0),
            ("tags", 1),
            ("blocks", 0),
            ("generations", 0),
            ("parents", -1),
            ("children", -1),
            ("bisection_vertices", -1),
            ("retired", False),
            ("refine_rejected", False),
            ("coarsen_marked", False),
        )
    }
    arrays.update(
        {
            name: grown(fields[name][:vertex_count], vertex_total, fill)
            for name, fill in (
                ("vertex_ids", -1),
                ("vertex_active", False),
                ("coordinates", 0.0),
                ("vertex_parents", -1),
                ("vertex_levels", 0),
                ("vertex_removal", -1),
            )
        }
    )
    base_vertex_ids = base.vertex_ids[:vertex_count]
    base_cell_ids = base.cell_ids[:cell_count]

    def global_slots(identifiers, prepared_ids, prepared_count, issued_base):
        return np.where(
            identifiers < issued_base,
            np.searchsorted(prepared_ids, identifiers),
            prepared_count + identifiers - issued_base,
        )

    def mapped_slots(values, slots):
        return np.where(values >= 0, slots[np.maximum(values, 0)], -1)

    for part in range(partitioned.parts.part_count):
        local_vertices, local_cells = int(cursors[part, 0]), int(cursors[part, 1])
        mesh = host.mesh
        vertex_ids = np.asarray(mesh.vertex_ids[part, :local_vertices], dtype=np.int64)
        vertex_slots = global_slots(
            vertex_ids, base_vertex_ids, vertex_count, vertex_base
        )
        cell_ids = np.asarray(mesh.cell_ids[part, :local_cells], dtype=np.int64)
        cell_slots = global_slots(cell_ids, base_cell_ids, cell_count, cell_base)

        issued = vertex_ids >= vertex_base
        target = vertex_slots[issued]
        arrays["vertex_ids"][target] = vertex_ids[issued]
        arrays["vertex_active"][target] = True
        arrays["coordinates"][target] = np.asarray(
            mesh.coordinates[part, :local_vertices], dtype=np.float64
        )[issued]
        arrays["vertex_parents"][target] = mapped_slots(
            np.asarray(host.vertex_parents[part, :local_vertices], np.int64)[issued],
            vertex_slots,
        )
        arrays["vertex_levels"][target] = np.asarray(
            host.vertex_levels[part, :local_vertices]
        )[issued]
        for name, values in (
            ("cell_ids", cell_ids),
            ("cell_active", mesh.cell_active[part, :local_cells]),
            (
                "cells",
                mapped_slots(
                    np.asarray(mesh.cells[part, :local_cells], np.int64), vertex_slots
                ),
            ),
            (
                "tuples",
                mapped_slots(
                    np.asarray(host.tuples[part, :local_cells], np.int64), vertex_slots
                ),
            ),
            ("tags", host.tags[part, :local_cells]),
            ("blocks", host.blocks[part, :local_cells]),
            ("generations", host.generations[part, :local_cells]),
        ):
            arrays[name][cell_slots] = values
        bisected = np.asarray(host.children[part, :local_cells, 0]) >= 0
        arrays["children"][cell_slots[bisected]] = mapped_slots(
            np.asarray(host.children[part, :local_cells], np.int64)[bisected],
            cell_slots,
        )
        arrays["bisection_vertices"][cell_slots[bisected]] = mapped_slots(
            np.asarray(host.bisection_vertices[part, :local_cells], np.int64)[bisected],
            vertex_slots,
        )
        packed = np.asarray(host.parents[part, :local_cells], dtype=np.int64)
        created = (cell_ids >= cell_base) & (packed >= 0)
        arrays["parents"][cell_slots[created]] = (
            2 * cell_slots[packed[created] // 2] + packed[created] % 2
        )
    return _HostEpoch(
        **arrays,
        cursors=np.asarray(
            (vertex_total, cell_total, cursors[0, 2], cursors[0, 3]), dtype=np.int64
        ),
        counters=np.asarray(host.counters[0], dtype=np.int64),
        # A terminal flag on any part rejects the whole epoch.
        flags=int(np.bitwise_or.reduce(np.asarray(host.status_flags, dtype=np.int64))),
    )


def commit_partitioned_adaptive_simplex(
    partitioned: PartitionedAdaptiveSimplex, states: AdaptiveSimplexState, /
) -> MeshAdaptationResult:
    """Commit a part-sharded epoch: one transfer, merge by global IDs, commit.

    Part-independent IDs make the merged epoch identical to the single-part
    epoch with the same marks, so the committed target, lineage, hierarchy,
    and distribution transition are those of `commit_adaptive_simplex`. The
    union of the parts' ``status_flags`` decides acceptance as there: a
    terminal flag on any part rejects the whole epoch.
    """

    if not isinstance(partitioned, PartitionedAdaptiveSimplex):
        raise TypeError("partitioned must be PartitionedAdaptiveSimplex.")
    if not isinstance(states, AdaptiveSimplexState) or (
        states.mesh.cells.shape[:2]
        != (partitioned.parts.part_count, partitioned.layout.cell_capacity)
    ):
        raise ValueError("states must be the stacked part states of this epoch.")
    started = time.monotonic()
    prepared = partitioned.prepared
    outcome = _outcome(prepared, _merged_epoch(partitioned, states))
    return _adaptation_result(prepared.adaptation, outcome, started)


__all__ = [
    "PartitionedAdaptiveSimplex",
    "PreparedAdaptiveSimplex",
    "commit_adaptive_simplex",
    "commit_partitioned_adaptive_simplex",
    "partition_adaptive_simplex",
    "prepare_adaptive_simplex",
]
