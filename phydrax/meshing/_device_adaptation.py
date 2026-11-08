#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device adaptation epochs of certified simplex meshes.

`prepare_adaptive_simplex` binds one certified source result (and the
`BisectionHierarchy` of earlier adaptations) to an initial device capacity
bucket. The labels, closure, and coarsening families replay the host bisection
route. Refinement and coarsening use compiled device kernels; when refinement
needs more capacity, its host wrapper grows the state and layout and retries
the unchanged update. Each update returns its current `layout` for continued
calls, while solvers consume the masked state directly. `commit_adaptive_simplex`
performs one device-to-host transfer and builds the canonical target, complete
lineage, sparse P1 transfer, and next hierarchy through the same host edit
assembly as the host route, so matching logical marks commit byte-identical
meshes.

Cell lineage relative to the epoch source follows bisection-tree containment:
every cell is a node ``(root, depth, path)`` of its root's binary bisection
tree, so a target cell inside a source cell is refined from it and a source
cell inside a target cell is coarsened into it, including cells re-created
after a coarsening inside one epoch.
"""

from __future__ import annotations

import time
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from math import factorial
from typing import Any, final, NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from .._execution_runtime import ExecutionGroup
    from ..discretization._view_support import OwnerLocalCellGeometry
    from ..graph._partition import GraphPartitionPlan
    from ._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from ._periodic import OwnerLocalPeriodicCandidate

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..discretization.fem import FiniteElementTopologyTransfer
from ..linalg._exact import determinant_exact
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
    _canonical,
    _Cells,
    _Coarsening,
    _edit,
    _EpochSubdivisions,
    _Flags,
    _Front,
    _global_vertices,
    _Growth,
    _labelled_start,
    _longest_edge_labels,
    _prepared_request,
    _prepared_source,
    _Records,
    _Refinement,
    _refinement_ends,
    _resolved_stencil,
    _resubdivided,
    _reverse_supports,
    _select,
    _SHIFT,
    _Source,
    _Start,
    _target_hierarchy,
    _undone_epoch_subdivisions,
    BisectionEvidence,
    BisectionHierarchy,
)
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._device_status import (
    DeviceEpoch,
    require_applied_status,
    require_committed_status,
)
from ._distribution_forest import (
    extract_simplex_forest,
    PeriodicSimplexForest,
    reconstruct_adaptive_simplex_state,
    SimplexForestRecords,
)
from ._lineage import (
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)
from ._result import CellMeshingResult
from ._topology_edit import _entity_lineage, entity_keys, EntityRelations, key_rows


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
    ) -> None:
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

    def cell_marks(
        self,
        cell_ids: np.ndarray,
        /,
        *,
        layout: AdaptiveSimplexLayout | None = None,
    ) -> np.ndarray:
        """Slot mask for prepared origins, padded to the current execution layout."""

        current = self.layout if layout is None else layout
        if (
            not isinstance(current, AdaptiveSimplexLayout)
            or current.dimension != self.layout.dimension
            or current.ambient_dimension != self.layout.ambient_dimension
            or current.coordinate_dtype != self.layout.coordinate_dtype
            or current.protected_edge_capacity != self.layout.protected_edge_capacity
            or current.vertex_capacity < self.layout.vertex_capacity
            or current.cell_capacity < self.layout.cell_capacity
        ):
            raise ValueError("layout must be a grown layout of this prepared epoch.")
        identifiers = np.asarray(cell_ids, dtype=np.int64)
        marks = np.zeros((current.cell_capacity,), dtype=np.bool_)
        origins = self.anchor.slot_origins
        marks[: origins.size] = np.isin(origins, identifiers) & (origins >= 0)
        return marks


def _record_generations(active_generations: np.ndarray, children: np.ndarray, /) -> Any:
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
    columns = np.asarray(
        [[j for j in range(width) if j != i] for i in range(width)],
        dtype=np.intp,
    )
    keys = np.sort(rows[:, columns], axis=2).reshape((-1, width - 1))
    found = key_rows(np.sort(facet_keys, axis=1), keys)
    classes = np.where(found >= 0, facet_classes[np.maximum(found, 0)] + 1, 0)
    return classes.reshape((count, width)).astype(np.int32)


def _forest(start: _Start, source: _Source, request: Any, /) -> Any:
    """Active cells and records of the start, merged into ID-ordered slots."""

    cells, records = start.front.cells, start.records
    active_count, record_count = cells.ids.size, records.parent_ids.size
    ids = np.concatenate((cells.ids, records.parent_ids))
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    count = ids.size

    def merged(active_values: Any, record_values: Any) -> Any:
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


def _vertices(start: _Start, source: _Source, request: Any, /) -> Any:
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
    parents[records.vertices, 1] = _refinement_ends(
        records.tuples, records.tags, dimension
    )
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


def _simplex_request(adaptation: PreparedMeshAdaptation, source: _Source, /) -> Any:
    """Protection and organization classes of one prepared device source."""

    constraints = adaptation.constraints
    empty = np.zeros((0,), dtype=np.int64)
    return _prepared_request(
        source,
        empty,
        empty,
        constraints.protected_edge_keys,
        constraints.protected_vertex_ids,
        constraints.cell_classes,
        constraints.facet_classes,
    )


def _prepared_simplex(adaptation: PreparedMeshAdaptation, /) -> PreparedAdaptiveSimplex:
    policy = adaptation.policy
    device = policy.device_policy
    mesh = adaptation.source.mesh
    source = _prepared_source(mesh)
    request = _simplex_request(adaptation, source)
    # ty: ignore[unresolved-attribute]
    hierarchy = adaptation.request.hierarchy
    start = (
        _labelled_start(source, request, policy.compatibility)
        if hierarchy is None
        # Restored barycentric parents are subdivided again: device epochs may
        # mark any active cell for bisection.
        else _resubdivided(
            # ty: ignore[invalid-argument-type]
            _bound_start(source, request, hierarchy),
            request,
            policy.compatibility,
            source.dimension,
        )
    )
    forest = _forest(start, source, request)
    vertices = _vertices(start, source, request)
    # ty: ignore[unresolved-attribute]
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


def _ancestors(
    keys: np.ndarray, queries: np.ndarray, table: np.ndarray, /, *, strict: Any
) -> Any:
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
    cell_classes: np.ndarray
    facet_classes: np.ndarray
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
        np.asarray(host.cell_classes, dtype=np.int64),
        np.asarray(host.facet_classes, dtype=np.int64),
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


def _cell_lineage(anchor: _Anchor, host: _HostEpoch, count: int, /) -> Any:
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


def _supports(
    anchor: _Anchor,
    host: _HostEpoch,
    subdivision_steps: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
    /,
) -> np.ndarray:
    """Surviving vertices spanning each removed source vertex (reverse passes).

    Undone subdivisions follow the device bisection passes.
    """

    count, width = anchor.source.vertex_ids.size, anchor.source.dimension + 1
    removal = np.asarray(host.vertex_removal, dtype=np.int64)[:count]
    parents = np.full((count, width), -1, dtype=np.int64)
    parents[:, :2] = np.asarray(host.vertex_parents, dtype=np.int64)[:count]
    steps = []
    for index in np.unique(removal[removal >= 0]):
        vertices = np.flatnonzero(removal == index)
        steps.append((vertices, parents[vertices]))
    steps.extend(subdivision_steps)
    return _reverse_supports(steps, count, anchor.source.dimension)


class _Committed(NamedTuple):
    edit: object
    hierarchy: BisectionHierarchy
    evidence: BisectionEvidence
    refined: bool
    coarsened: bool
    partial: bool


def _epoch_subdivisions(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    cells: _Cells,
    target: np.ndarray,
    restored: np.ndarray,
    lineage: _Lineage,
    vertex_count: int,
    /,
) -> _EpochSubdivisions | None:
    """Host undo of the barycentric subdivisions the device coarsening exposed.

    The compiled passes coarsen bisection families only; the recorded
    subdivisions are then removed by the host rule on the active cells, whose
    facet classes the device maintained. Parents the device restored count as
    marked, as host-restored parents do.
    """

    anchor = prepared.anchor
    source, subdivisions = anchor.source, anchor.start.subdivisions
    if subdivisions.parent_ids.size == 0:
        return None
    count = int(host.cursors[1])
    width = source.dimension + 1
    columns = np.asarray(
        [[j for j in range(width) if j != i] for i in range(width)], dtype=np.intp
    )
    keys, first = np.unique(
        np.sort(cells.rows[:, columns], axis=2).reshape((-1, width - 1)),
        axis=0,
        return_index=True,
    )
    facet_classes = np.asarray(host.facet_classes)[:count][target].reshape((-1,))
    marked = np.asarray(host.coarsen_marked)[:count].copy()
    marked[restored] = True
    order = np.argsort(cells.ids, kind="stable")
    flags = _Flags(
        marked[target][order],
        np.zeros(order.shape, dtype=np.bool_),
        np.asarray(host.cell_classes)[:count][target][order],
    )
    return _undone_epoch_subdivisions(
        source,
        _simplex_request(prepared.adaptation, source),
        subdivisions,
        _select(cells, order),
        flags,
        # Device facet classes are source class + 1 (0: created by bisection).
        (keys, facet_classes[first] - 1),
        vertex_count,
        lineage.removed_ids,
        lineage.link_targets,
    )


def _committed(prepared: PreparedAdaptiveSimplex, host: _HostEpoch, /) -> Any:
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
    restored_count = restored.size
    removed_ids, link_targets = lineage.removed_ids, lineage.link_targets
    subdivisions, subdivision_steps = start.subdivisions, ()
    undone = _epoch_subdivisions(
        prepared, host, cells, target, restored, lineage, vertex_count
    )
    if undone is not None:
        cells, subdivisions = undone.cells, undone.subdivisions
        subdivision_steps = undone.steps
        removed_ids, link_targets = undone.removed_ids, undone.link_targets
        removed_vertices = np.union1d(
            removed_vertices, np.concatenate([step[0] for step in subdivision_steps])
        )
        rejected_coarsenings = rejected_coarsenings[
            ~np.isin(rejected_coarsenings, undone.undone_children)
        ]
        restored_count += undone.restored_ids.size - int(
            np.count_nonzero(np.isin(ids[restored], undone.undone_children))
        )
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
        removed_ids,
        link_targets,
        np.zeros((0,), dtype=np.int64),
        removed_vertices,
        _supports(anchor, host, subdivision_steps),
        int(counters[AdaptiveSimplexCounter.COARSENING_PASSES]) + len(subdivision_steps),
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
    hierarchy = _target_hierarchy(
        source, start, cells, records, subdivisions, retired_tables, front
    )
    subdivision_vertices = sum(step[0].size for step in subdivision_steps)
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
        coarsened_vertices=int(counters[AdaptiveSimplexCounter.COARSENED_VERTICES])
        + subdivision_vertices,
        coarsening_passes=coarsening.passes,
        restored_cells=restored_count,
        rejected_coarsening_ids=rejected_coarsenings,
    )
    return _Committed(
        edit,
        hierarchy,
        evidence,
        evidence.bisections > 0 or start.uniform,
        evidence.coarsened_vertices > 0,
        rejected_refinements.size > 0 or rejected_coarsenings.size > 0,
    )


def _outcome(prepared: PreparedAdaptiveSimplex, host: _HostEpoch, /) -> Any:
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
            MeshAdaptationStatus.PARTIAL
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
    status = (
        MeshAdaptationStatus.PARTIAL
        if committed.partial
        else MeshAdaptationStatus.COMPLETE
    )
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


def _checked_state(
    prepared: PreparedAdaptiveSimplex, state: AdaptiveSimplexState, /
) -> None:
    if not isinstance(prepared, PreparedAdaptiveSimplex):
        raise TypeError("prepared must be PreparedAdaptiveSimplex.")
    if not isinstance(state, AdaptiveSimplexState):
        raise TypeError("state must be AdaptiveSimplexState.")
    mesh = state.mesh
    layout = prepared.layout
    current = AdaptiveSimplexLayout(
        layout.dimension,
        layout.ambient_dimension,
        vertex_capacity=mesh.vertex_capacity,
        cell_capacity=mesh.cell_capacity,
        protected_edge_capacity=layout.protected_edge_capacity,
        coordinate_dtype=np.dtype(mesh.coordinates.dtype),
    )
    if (
        mesh.signature_id != current.mesh_signature_id
        or current.coordinate_dtype != layout.coordinate_dtype
        or mesh.vertex_capacity < layout.vertex_capacity
        or mesh.cell_capacity < layout.cell_capacity
        or state.protected_codes.shape != (layout.protected_edge_capacity,)
    ):
        raise ValueError("The state does not belong to this prepared epoch.")


def commit_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex, state: AdaptiveSimplexState, /
) -> MeshAdaptationResult:
    """Commit one device epoch: one transfer, canonical target, lineage, evidence.

    The target is assembled, organization inherited by exact IDs, certified,
    and bound to its `CellMeshTransition`, `MeshLineage`, and sparse P1
    transfer exactly as the host bisection route does; the result's
    ``hierarchy`` prepares the next epoch. The epoch's cumulative
    ``status_flags`` decide acceptance: protected conflicts raise
    `MeshingFailure` (INVALID_SPECIFICATION), invalid geometry is rejected
    (QUALITY_REJECTED), and ``NEEDS_HOST_RESOLUTION`` requires every committed
    cell to be certified positively oriented by exact host predicates.
    """

    _checked_state(prepared, state)
    started = time.monotonic()
    outcome = _outcome(prepared, _host_epoch(state))
    return _adaptation_result(prepared.adaptation, outcome, started)


def _execute_device_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    """One refine-then-coarsen device epoch of a marked request, then commit."""

    request = prepared.request
    if not isinstance(request, MarkedMeshAdaptation):
        raise TypeError("The device bisection route requires MarkedMeshAdaptation.")
    if request.coarsen_cell_ids.size and request.hierarchy is None:
        raise ValueError(
            "Coarsening requires the BisectionHierarchy of a previous bisection."
        )
    simplex = _prepared_simplex(prepared)
    layout = simplex.layout
    refined = refine_adaptive_simplex(
        layout,
        simplex.state,
        simplex.cell_marks(np.asarray(request.refine_cell_ids)),
    )
    require_applied_status(int(refined.report.status), DeviceEpoch.BISECTION)
    coarsen_marks = np.zeros((refined.layout.cell_capacity,), dtype=np.bool_)
    original_marks = simplex.cell_marks(np.asarray(request.coarsen_cell_ids))
    coarsen_marks[: original_marks.size] = original_marks
    coarsened = coarsen_adaptive_simplex(
        refined.layout,
        refined.state,
        coarsen_marks,
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
    ) -> None:
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
                # ty: ignore[unresolved-attribute]
                "distribution": prepared.adaptation.policy.distribution.distribution_id,
            }
        )

    def cell_marks(
        self,
        cell_ids: np.ndarray,
        /,
        *,
        layout: AdaptiveSimplexLayout | None = None,
    ) -> np.ndarray:
        """Per-part prepared-origin mask, padded to the current layout."""

        current = self.layout if layout is None else layout
        if (
            not isinstance(current, AdaptiveSimplexLayout)
            or current.dimension != self.layout.dimension
            or current.ambient_dimension != self.layout.ambient_dimension
            or current.coordinate_dtype != self.layout.coordinate_dtype
            or current.protected_edge_capacity != self.layout.protected_edge_capacity
            or current.vertex_capacity < self.layout.vertex_capacity
            or current.cell_capacity < self.layout.cell_capacity
        ):
            raise ValueError("layout must be a grown layout of this partitioned epoch.")
        identifiers = np.asarray(cell_ids, dtype=np.int64)
        marks = np.zeros(
            (self.slot_origins.shape[0], current.cell_capacity), dtype=np.bool_
        )
        origins = self.slot_origins
        marks[:, : origins.shape[1]] = np.isin(origins, identifiers) & (origins >= 0)
        return marks


class OwnerLocalAdaptivePartTransition(NamedTuple):
    """One accepted part-local adaptive topology transition.

    The source and target meshes are owner-local views retaining global vertex
    and cell IDs. `state` keeps the device execution ancestry/cursors needed for
    restart; `lineage` and the exact vertex stencil expose the accepted local
    source map without gathering a global candidate mesh.
    """

    part_index: int
    source_distribution_id: str
    source_part_id: str
    source_mesh_id: str | None
    source_topology_id: str
    source_geometry_id: str
    source_geometry_layout_id: str
    coordinate_contract_id: str
    source_periodic_topology_id: str | None
    source_mesh: CellMesh | None
    target_mesh: CellMesh | None
    lineage: MeshLineage | None
    vertex_stencil: VertexInterpolationStencil | None
    vertex_transfer: FiniteElementTopologyTransfer | None
    source_cell_ids: np.ndarray
    target_cell_ids: np.ndarray
    target_cell_owner_ranks: np.ndarray
    target_cell_parent_ids: np.ndarray
    target_vertex_parent_ids: np.ndarray
    source_cell_measures: tuple[Fraction, ...]
    target_cell_measures: tuple[Fraction, ...]
    source_partition_weights: np.ndarray
    state: AdaptiveSimplexState
    slot_origins: np.ndarray
    status_flags: int
    counters: np.ndarray
    transition_id: str


class PartitionedAdaptiveSimplexResult(NamedTuple):
    """Collectively accepted owner-local transitions; no merged target mesh."""

    source_distribution_id: str
    source_part_id: str
    source_mesh_id: str
    source_topology_id: str
    source_geometry_id: str
    coordinate_contract_id: str
    status: MeshAdaptationStatus
    status_flags: np.ndarray
    parts: tuple[OwnerLocalAdaptivePartTransition, ...]
    result_id: str


def _owner_local_exact_cell_measures(mesh: CellMesh, /) -> tuple[Fraction, ...]:
    dimension = mesh.topological_dimension
    denominator = factorial(dimension)
    measures: dict[int, Fraction] = {}
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        if vertices.shape[1] != dimension + 1:
            raise ValueError(
                "Owner-local measures require full-dimensional simplex cells."
            )
        for cell_id, rows in zip(identifiers.tolist(), vertices, strict=True):
            corners = [
                tuple(Fraction(float(value)) for value in points[int(row)].tolist())
                for row in np.asarray(rows, dtype=np.int64).reshape(-1)
            ]
            origin = corners[0]
            jacobian = tuple(
                tuple(
                    corners[column + 1][axis] - origin[axis] for axis in range(dimension)
                )
                for column in range(dimension)
            )
            determinant = determinant_exact(jacobian)
            if determinant == 0:
                raise ValueError("Owner-local target simplex has zero exact measure.")
            measures[int(cell_id)] = abs(determinant) / denominator
    return tuple(
        measures[int(identifier)]
        for identifier in np.asarray(mesh.entity_set(dimension).entity_ids)
    )


def _owner_local_parent_ids(
    state: AdaptiveSimplexState, /
) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    """Project executor slots to the scientific cell and vertex parent DAGs."""
    vertex_count, cell_count = map(int, np.asarray(state.cursors)[:2])
    vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)[:vertex_count]
    cell_ids = np.asarray(state.mesh.cell_ids, dtype=np.int64)[:cell_count]
    vertices = np.asarray(state.vertex_parents, dtype=np.int64)[:vertex_count]
    cells = np.asarray(state.parents, dtype=np.int64)[:cell_count]
    if (
        vertices.shape != (vertex_count, 2)
        or np.any(vertices < -1)
        or np.any(vertices >= vertex_count)
        or np.any(np.any(vertices < 0, axis=1) != np.all(vertices < 0, axis=1))
        or np.any(cells < -1)
        or np.any(cells >= 2 * cell_count)
    ):
        raise ValueError("Owner-local parent slots escape their scientific domains.")
    return (
        {
            int(identifier): -1 if code < 0 else int(cell_ids[int(code) // 2])
            for identifier, code in zip(cell_ids, cells, strict=True)
        },
        {
            int(vertex_ids[row]): (
                -1 if vertices[row, 0] < 0 else int(vertex_ids[vertices[row, 0]]),
                -1 if vertices[row, 1] < 0 else int(vertex_ids[vertices[row, 1]]),
            )
            for row in range(vertex_count)
        },
    )


def _owner_local_vertex_stencil(
    source: CellMesh,
    target: CellMesh,
    vertex_parent_ids: dict[int, tuple[int, int]],
    /,
) -> tuple[VertexInterpolationStencil, FiniteElementTopologyTransfer]:
    source_mesh_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    source_coordinates = np.asarray(source.coordinates, dtype=np.float64)
    source_order = np.argsort(source_mesh_ids, kind="stable")
    source_ids = source_mesh_ids[source_order]
    coordinates = source_coordinates[source_order]
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    memo: dict[int, tuple[tuple[int, Fraction], ...]] = {}
    active: set[int] = set()

    def exact_row(identifier: int) -> tuple[tuple[int, Fraction], ...]:
        cached = memo.get(identifier)
        if cached is not None:
            return cached
        if identifier in source_rows:
            result = ((identifier, Fraction(1)),)
            memo[identifier] = result
            return result
        parents = vertex_parent_ids.get(identifier)
        if parents is None or identifier in active:
            raise ValueError(
                "An owner-local target vertex has no acyclic source ancestry."
            )
        if any(parent < 0 for parent in parents):
            raise ValueError("A created owner-local vertex lacks two source parents.")
        active.add(identifier)
        combined: dict[int, Fraction] = {}
        for parent in parents:
            for source_id, weight in exact_row(parent):
                combined[source_id] = combined.get(source_id, Fraction(0)) + weight / 2
        active.remove(identifier)
        result = tuple(
            (source_id, weight)
            for source_id, weight in sorted(combined.items())
            if weight
        )
        if not result or sum((weight for _, weight in result), Fraction(0)) != 1:
            raise ValueError("An owner-local vertex source row is not constant-exact.")
        memo[identifier] = result
        return result

    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    exact_rows = tuple(exact_row(int(identifier)) for identifier in target_ids)
    width = max(1, max(len(row) for row in exact_rows))
    source_routes = np.full((target_ids.size, width), -1, dtype=np.int64)
    weights = np.zeros((target_ids.size, width), dtype=np.float64)
    valid = np.zeros((target_ids.size, width), dtype=np.bool_)
    exact_weights = [[Fraction(0) for _ in range(width)] for _ in range(target_ids.size)]
    for row, entries in enumerate(exact_rows):
        for column, (source_id, exact_weight) in enumerate(entries):
            position = source_rows.get(source_id)
            if position is None:
                raise ValueError(
                    "An owner-local stencil references an absent source vertex."
                )
            source_routes[row, column] = source_id
            weights[row, column] = float(exact_weight)
            valid[row, column] = True
            exact_weights[row][column] = exact_weight
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        target_ids,
        source_routes,
        weights,
        valid,
        exact_weights=tuple(tuple(row) for row in exact_weights),
    )
    transfer = stencil.as_transfer(
        source_ids,
        source_topology_id=source.topology_id,
        target_topology_id=target.topology_id,
        preserves_linear=True,
        source_coordinates=coordinates,
        target_coordinates=np.asarray(target.coordinates, dtype=np.float64),
    )
    return stencil, transfer


def _owner_local_lineage(
    source: CellMesh,
    target: CellMesh,
    stencil: VertexInterpolationStencil,
    parents: dict[int, int],
    /,
) -> MeshLineage:
    target_ids = np.asarray(stencil.target_global_ids, dtype=np.int64)
    source_rows = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid = np.asarray(stencil.valid, dtype=np.bool_)
    exact = stencil.exact_weights
    stencil_support = {
        int(identifier): {
            int(source_id)
            for source_id, weight, active in zip(
                source_rows[row], exact[row], valid[row], strict=True
            )
            if active and weight
        }
        for row, identifier in enumerate(target_ids.tolist())
    }
    records = []
    for dimension in range(source.topological_dimension + 1):
        if dimension == source.topological_dimension:
            source_cell_ids = np.asarray(
                source.entity_set(dimension).entity_ids, dtype=np.int64
            )
            target_entity_ids = np.asarray(
                target.entity_set(dimension).entity_ids, dtype=np.int64
            )
            source_cell_set = set(map(int, source_cell_ids))
            target_cell_set = set(map(int, target_entity_ids))

            def ancestor(identifier: int, candidates: set[int]) -> int:
                seen: set[int] = set()
                current = parents[identifier]
                while current >= 0:
                    if current in candidates:
                        return current
                    if current in seen:
                        raise ValueError("Owner-local cell lineage contains a cycle.")
                    seen.add(current)
                    current = parents[current]
                return -1

            links: list[tuple[int, int, int]] = []
            for identifier in sorted(target_cell_set - source_cell_set):
                parent = ancestor(identifier, source_cell_set)
                if parent >= 0:
                    links.append(
                        (parent, identifier, int(EntityLineageKind.REFINED_FROM))
                    )
            for identifier in sorted(source_cell_set - target_cell_set):
                parent = ancestor(identifier, target_cell_set)
                if parent >= 0:
                    links.append(
                        (identifier, parent, int(EntityLineageKind.COARSENED_INTO))
                    )
            covered_source = {row[0] for row in links} | (
                source_cell_set & target_cell_set
            )
            covered_target = {row[1] for row in links} | (
                source_cell_set & target_cell_set
            )
            if covered_source != source_cell_set or covered_target != target_cell_set:
                raise ValueError(
                    "Owner-local nested lineage does not cover both cell domains."
                )
            relations = EntityRelations(
                dimension,
                np.asarray([row[0] for row in links], dtype=np.int64).reshape((-1, 1)),
                np.asarray([row[1] for row in links], dtype=np.int64).reshape((-1, 1)),
                np.asarray([row[2] for row in links], dtype=np.int32),
            )
            records.append(_entity_lineage(source, target, relations))
            continue
        source_keys = entity_keys(source, dimension)
        target_keys = entity_keys(target, dimension)
        source_ids = np.asarray(source.entity_set(dimension).entity_ids, dtype=np.int64)
        target_entity_ids = np.asarray(
            target.entity_set(dimension).entity_ids, dtype=np.int64
        )
        source_by_key = {
            tuple(int(value) for value in row if int(value) >= 0): int(identifier)
            for row, identifier in zip(source_keys, source_ids, strict=True)
        }
        relation_source_keys: list[tuple[int, ...]] = []
        relation_target_keys: list[tuple[int, ...]] = []
        relation_kinds: list[int] = []
        for target_row, row in enumerate(target_keys):
            key = tuple(int(value) for value in row if int(value) >= 0)
            if dimension == 0 or key in source_by_key:
                continue
            support = set().union(*(stencil_support[int(vertex)] for vertex in key))
            parent_key = tuple(sorted(support))
            if len(parent_key) != dimension + 1 or parent_key not in source_by_key:
                continue
            relation_source_keys.append(parent_key)
            relation_target_keys.append(key)
            relation_kinds.append(int(EntityLineageKind.REFINED_FROM))
        width = dimension + 1
        relations = EntityRelations(
            dimension,
            np.asarray(relation_source_keys, dtype=np.int64).reshape((-1, width)),
            np.asarray(relation_target_keys, dtype=np.int64).reshape((-1, width)),
            np.asarray(relation_kinds, dtype=np.int32),
        )
        records.append(_entity_lineage(source, target, relations))
    return MeshLineage(source.topology_id, target.topology_id, tuple(records))


def _owner_local_origins(
    state: AdaptiveSimplexState,
    source_cell_ids: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = state.mesh
    cell_count = int(np.asarray(state.cursors)[1])
    cell_ids = np.asarray(mesh.cell_ids, dtype=np.int64)[:cell_count]
    cell_active = np.asarray(mesh.cell_active, dtype=np.bool_)[:cell_count]
    parent_codes = np.asarray(state.parents, dtype=np.int64)[:cell_count]
    source_ids = set(int(value) for value in np.asarray(source_cell_ids, dtype=np.int64))
    memo: dict[int, int] = {}
    active: set[int] = set()

    def origin(row: int) -> int:
        if row in memo:
            return memo[row]
        identifier = int(cell_ids[row])
        if identifier in source_ids:
            memo[row] = identifier
            return identifier
        if row in active:
            raise ValueError("Owner-local cell ancestry contains a cycle.")
        code = int(parent_codes[row])
        if code < 0:
            return -1
        parent = code // 2
        if parent < 0 or parent >= cell_count:
            raise ValueError("Owner-local parent cell index is out of range.")
        active.add(row)
        result = origin(parent)
        active.remove(row)
        memo[row] = result
        return result

    active_rows = np.flatnonzero(cell_active)
    origins = np.asarray([origin(int(row)) for row in active_rows], dtype=np.int64)
    parents = np.asarray(
        [
            -1 if parent_codes[row] < 0 else cell_ids[int(parent_codes[row]) // 2]
            for row in active_rows
        ],
        dtype=np.int64,
    )
    return cell_ids[active_rows], origins, parents


def _owner_local_mesh_subset(
    source: CellMesh,
    source_cell_ids: np.ndarray,
    /,
    *,
    numeric_version: str,
) -> CellMesh | None:
    """Retain exact cell identities and their canonically ordered vertex closure."""
    requested = set(int(value) for value in np.asarray(source_cell_ids, dtype=np.int64))
    selected_source: list[tuple[int, np.ndarray, np.ndarray]] = []
    used_source_vertices: list[np.ndarray] = []
    selected_ids: set[int] = set()
    for block_index, block in enumerate(source.blocks):
        ids = np.asarray(block.global_ids, dtype=np.int64)
        rows = np.flatnonzero(np.isin(ids, tuple(sorted(requested))))
        if rows.size:
            vertices = np.asarray(block.vertices, dtype=np.int64)[rows]
            selected_source.append((block_index, vertices, ids[rows]))
            used_source_vertices.append(vertices.reshape((-1,)))
            selected_ids.update(map(int, ids[rows]))
    if selected_ids != requested:
        raise ValueError("Owner-local cell selection escapes its current mesh.")
    if not selected_source:
        return None
    source_rows = np.unique(np.concatenate(used_source_vertices))
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    source_rows = source_rows[np.argsort(source_vertex_ids[source_rows], kind="stable")]
    source_local = np.full((source_vertex_ids.size,), -1, dtype=np.int64)
    source_local[source_rows] = np.arange(source_rows.size, dtype=np.int64)
    source_blocks = tuple(
        CellBlock(
            source.blocks[block_index].name,
            source.blocks[block_index].cell_kind,
            source_local[vertices].astype(np.int32),
            global_ids=ids,
        )
        for block_index, vertices, ids in selected_source
    )
    return CellMesh(
        np.asarray(source.coordinates, dtype=np.float64)[source_rows],
        source_blocks,
        vertex_global_ids=source_vertex_ids[source_rows],
        numeric_version=numeric_version,
    )


def _owner_local_part_meshes(
    source: CellMesh,
    source_cell_ids: np.ndarray,
    target_state: AdaptiveSimplexState,
    part_index: int,
    /,
    *,
    numeric_version: str,
) -> tuple[CellMesh | None, CellMesh | None]:
    source_mesh = _owner_local_mesh_subset(
        source,
        source_cell_ids,
        numeric_version=f"{numeric_version}:source:{part_index}",
    )
    if source_mesh is None:
        return None, None
    host = target_state
    vertex_count = int(np.asarray(host.cursors)[0])
    cell_count = int(np.asarray(host.cursors)[1])
    target_vertices = np.flatnonzero(
        np.asarray(host.mesh.vertex_active, dtype=np.bool_)[:vertex_count]
    )
    target_cells = np.flatnonzero(
        np.asarray(host.mesh.cell_active, dtype=np.bool_)[:cell_count]
    )
    if not target_cells.size:
        return source_mesh, None
    vertex_ids = np.asarray(host.mesh.vertex_ids, dtype=np.int64)[target_vertices]
    order = np.argsort(vertex_ids, kind="stable")
    target_vertices = target_vertices[order]
    vertex_ids = vertex_ids[order]
    local = np.full((vertex_count,), -1, dtype=np.int64)
    local[target_vertices] = np.arange(target_vertices.size, dtype=np.int64)
    state_cells = np.asarray(host.mesh.cells, dtype=np.int64)[:cell_count]
    cell_ids = np.asarray(host.mesh.cell_ids, dtype=np.int64)[:cell_count]
    blocks = np.asarray(host.blocks, dtype=np.int64)[:cell_count]
    target_blocks = tuple(
        CellBlock(
            source.blocks[block_index].name,
            source.blocks[block_index].cell_kind,
            local[state_cells[rows]].astype(np.int32),
            global_ids=cell_ids[rows],
        )
        for block_index in range(len(source.blocks))
        if (rows := target_cells[blocks[target_cells] == block_index]).size
    )
    if not target_blocks:
        raise ValueError("Owner-local target cells lack source-family blocks.")
    target_mesh = CellMesh(
        np.asarray(host.mesh.coordinates, dtype=np.float64)[target_vertices],
        target_blocks,
        vertex_global_ids=vertex_ids,
        numeric_version=f"{numeric_version}:target:{part_index}",
    )
    return source_mesh, target_mesh


def _owner_local_part_transition(
    prepared: PreparedAdaptiveSimplex,
    partitioned: PartitionedAdaptiveSimplex,
    part_index: int,
    states: AdaptiveSimplexState,
    host_states: AdaptiveSimplexState,
    /,
) -> OwnerLocalAdaptivePartTransition:
    adaptation = prepared.adaptation
    distribution = adaptation.policy.distribution
    if distribution is None:
        raise ValueError("Owner-local adaptive commits require a source distribution.")
    source_result = adaptation.source
    source_mesh = source_result.mesh
    source_cell_rows = np.asarray(distribution.owned_rows[part_index], dtype=np.int64)
    source_cell_ids = np.asarray(distribution.cell_global_ids, dtype=np.int64)[
        source_cell_rows
    ]
    host_part = jax.tree_util.tree_map(lambda value: value[part_index], host_states)
    local_state = jax.tree_util.tree_map(lambda value: value[part_index], states)
    source_local_mesh, target_mesh = _owner_local_part_meshes(
        source_mesh,
        source_cell_ids,
        host_part,
        part_index,
        numeric_version=f"owner-local:{prepared.prepared_id}",
    )
    slot_origins = np.full((states.mesh.cell_capacity,), -1, dtype=np.int64)
    slot_origins[: partitioned.slot_origins.shape[1]] = np.asarray(
        partitioned.slot_origins[part_index], dtype=np.int64
    )
    status_flags = int(np.asarray(host_part.status_flags))
    counters = np.asarray(host_part.counters, dtype=np.int64)
    if source_local_mesh is None or target_mesh is None:
        if source_cell_ids.size or target_mesh is not None:
            raise ValueError("A nonempty owner-local adaptation lost its local mesh.")
        empty = np.zeros((0,), dtype=np.int64)
        return OwnerLocalAdaptivePartTransition(
            part_index,
            distribution.distribution_id,
            distribution.part.part_id,
            source_mesh.mesh_id,
            source_mesh.topology_id,
            cell_geometry_id(source_result.geometry),
            source_result.geometry.geometry_layout_id,
            source_result.coordinate_contract.spatial_id,
            None
            if source_mesh.periodic_topology is None
            else source_mesh.periodic_topology.periodic_topology_id,
            None,
            None,
            None,
            None,
            None,
            empty,
            empty,
            empty,
            empty,
            empty.reshape((0, 2)),
            (),
            (),
            empty.astype(np.float64),
            local_state,
            slot_origins,
            status_flags,
            counters,
            canonical_fingerprint(
                {
                    "kind": "owner-local-adaptive-part",
                    "source_distribution": distribution.distribution_id,
                    "part_index": part_index,
                    "status_flags": status_flags,
                }
            ),
        )
    cell_parents, vertex_parents = _owner_local_parent_ids(host_part)
    stencil, transfer = _owner_local_vertex_stencil(
        source_local_mesh, target_mesh, vertex_parents
    )
    target_cell_ids, origins, parents = _owner_local_origins(host_part, source_cell_ids)
    lineage = _owner_local_lineage(source_local_mesh, target_mesh, stencil, cell_parents)
    source_cell_order = np.asarray(
        source_local_mesh.entity_set(source_mesh.topological_dimension).entity_ids,
        dtype=np.int64,
    )
    source_measures = _owner_local_exact_cell_measures(source_local_mesh)
    target_order = np.asarray(
        target_mesh.entity_set(target_mesh.topological_dimension).entity_ids,
        dtype=np.int64,
    )
    target_measures = _owner_local_exact_cell_measures(target_mesh)
    parents_by_id = {
        int(identifier): int(parent)
        for identifier, parent in zip(target_cell_ids, parents, strict=True)
    }
    target_parents = np.asarray(
        [parents_by_id[int(identifier)] for identifier in target_order],
        dtype=np.int64,
    )
    state_vertex_count = int(np.asarray(host_part.cursors)[0])
    state_vertex_ids = np.asarray(host_part.mesh.vertex_ids, dtype=np.int64)[
        :state_vertex_count
    ]
    state_vertex_parents = np.asarray(host_part.vertex_parents, dtype=np.int64)[
        :state_vertex_count
    ]
    state_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(state_vertex_ids.tolist())
    }
    target_vertex_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_parent_ids = np.full((target_vertex_ids.size, 2), -1, dtype=np.int64)
    for target_row, identifier in enumerate(target_vertex_ids.tolist()):
        state_row = state_vertex_rows[int(identifier)]
        for column, parent_row in enumerate(state_vertex_parents[state_row]):
            if parent_row >= 0:
                target_vertex_parent_ids[target_row, column] = state_vertex_ids[
                    int(parent_row)
                ]
    source_partition_weight_by_id = {
        int(identifier): float(weight)
        for identifier, weight in zip(
            np.asarray(distribution.cell_global_ids, dtype=np.int64),
            np.asarray(distribution.cell_weights, dtype=np.float64),
            strict=True,
        )
    }
    target_owner_ranks = np.full(target_order.shape, part_index, dtype=np.int32)
    source_partition_weights = np.asarray(
        [
            source_partition_weight_by_id[int(identifier)]
            for identifier in source_cell_order
        ],
        dtype=np.float64,
    )
    return OwnerLocalAdaptivePartTransition(
        part_index,
        distribution.distribution_id,
        distribution.part.part_id,
        source_mesh.mesh_id,
        source_mesh.topology_id,
        cell_geometry_id(source_result.geometry),
        source_result.geometry.geometry_layout_id,
        source_result.coordinate_contract.spatial_id,
        None
        if source_mesh.periodic_topology is None
        else source_mesh.periodic_topology.periodic_topology_id,
        source_local_mesh,
        target_mesh,
        lineage,
        stencil,
        transfer,
        source_cell_order,
        target_order,
        target_owner_ranks,
        target_parents,
        target_vertex_parent_ids,
        source_measures,
        target_measures,
        source_partition_weights,
        local_state,
        slot_origins,
        status_flags,
        counters,
        canonical_fingerprint(
            {
                "kind": "owner-local-adaptive-part",
                "source_distribution": distribution.distribution_id,
                "source_part": distribution.part.part_id,
                "part_index": part_index,
                "source_local_mesh": source_local_mesh.mesh_id,
                "target_local_mesh": target_mesh.mesh_id,
                "lineage": lineage.lineage_id,
                "vertex_stencil": stencil.stencil_id,
                "status_flags": status_flags,
            }
        ),
    )


def _part_arrays(
    base: _HostEpoch, prepared: PreparedAdaptiveSimplex, slots: Any, /
) -> Any:
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
    # ty: ignore[unresolved-attribute]
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


def commit_partitioned_adaptive_simplex(
    partitioned: PartitionedAdaptiveSimplex, states: AdaptiveSimplexState, /
) -> PartitionedAdaptiveSimplexResult:
    """Accept part-local results without assembling a global target mesh."""

    if not isinstance(partitioned, PartitionedAdaptiveSimplex):
        raise TypeError("partitioned must be PartitionedAdaptiveSimplex.")
    if not isinstance(states, AdaptiveSimplexState):
        raise TypeError("states must be a stacked AdaptiveSimplexState.")
    part_count = partitioned.parts.part_count
    mesh = states.mesh
    if (
        states.cursors.shape != (part_count, 4)
        or mesh.cells.ndim != 3
        or mesh.cells.shape[0] != part_count
        or mesh.cells.shape[1] != mesh.cell_capacity
        or states.protected_codes.shape
        != (part_count, partitioned.layout.protected_edge_capacity)
    ):
        raise ValueError("states must be the stacked parts of this adaptive epoch.")
    current = AdaptiveSimplexLayout(
        partitioned.layout.dimension,
        partitioned.layout.ambient_dimension,
        vertex_capacity=mesh.vertex_capacity,
        cell_capacity=mesh.cell_capacity,
        protected_edge_capacity=partitioned.layout.protected_edge_capacity,
        coordinate_dtype=np.dtype(mesh.coordinates.dtype),
    )
    if (
        mesh.signature_id != current.mesh_signature_id
        or current.coordinate_dtype != partitioned.layout.coordinate_dtype
        or mesh.vertex_capacity < partitioned.layout.vertex_capacity
        or mesh.cell_capacity < partitioned.layout.cell_capacity
    ):
        raise ValueError("states are stale for this partitioned adaptive epoch.")
    prepared = partitioned.prepared
    adaptation = prepared.adaptation
    distribution = adaptation.policy.distribution
    if distribution is None:
        raise ValueError("Owner-local adaptive commits require a source distribution.")
    source_mesh = adaptation.source.mesh
    if source_mesh.periodic_topology is not None:
        raise ValueError(
            "Periodic owner-local commits require an exact quotient and orbit-closure witness."
        )
    host_states = jax.device_get(states)
    for part_index in range(part_count):
        vertex_count, cell_count = map(
            int, np.asarray(host_states.cursors[part_index], dtype=np.int64)[:2]
        )
        active = np.asarray(
            host_states.mesh.cell_active[part_index, :cell_count], dtype=np.bool_
        )
        require_committed_status(
            int(np.asarray(host_states.status_flags[part_index])),
            np.asarray(host_states.mesh.coordinates[part_index, :vertex_count]),
            np.asarray(host_states.mesh.cells[part_index, :cell_count])[active],
            DeviceEpoch.BISECTION,
        )
    parts = tuple(
        _owner_local_part_transition(
            prepared, partitioned, part_index, states, host_states
        )
        for part_index in range(part_count)
    )
    flags = np.asarray([part.status_flags for part in parts], dtype=np.int64)
    rejected_refinement = np.asarray(host_states.refine_rejected, dtype=np.bool_)
    rejected_coarsening = np.asarray(host_states.coarsen_marked, dtype=np.bool_)
    retired = np.asarray(host_states.retired, dtype=np.bool_)
    partial = bool(np.any(rejected_refinement) or np.any(rejected_coarsening & ~retired))
    changed = any(
        part.source_mesh is not None
        and part.target_mesh is not None
        and part.source_mesh.topology_id != part.target_mesh.topology_id
        for part in parts
    )
    status = (
        MeshAdaptationStatus.PARTIAL
        if partial
        else MeshAdaptationStatus.COMPLETE
        if changed
        else MeshAdaptationStatus.UNCHANGED
    )
    result_id = canonical_fingerprint(
        {
            "kind": "partitioned-adaptive-simplex-result",
            "prepared": prepared.prepared_id,
            "distribution": distribution.distribution_id,
            "status": status.value,
            "parts": tuple(part.transition_id for part in parts),
            "flags": flags,
        }
    )
    return PartitionedAdaptiveSimplexResult(
        distribution.distribution_id,
        distribution.part.part_id,
        source_mesh.mesh_id,
        source_mesh.topology_id,
        cell_geometry_id(adaptation.source.geometry),
        adaptation.source.coordinate_contract.spatial_id,
        status,
        flags,
        parts,
        result_id,
    )


class OwnerLocalPeriodicSizeStatistic(NamedTuple):
    control_id: str
    statistic: str
    target_size: float
    lower_squared_length: Fraction
    upper_squared_length: Fraction
    interpolation: Fraction
    achieved: float
    below: bool
    above: bool
    accepted: bool


class OwnerLocalPeriodicSizeEvidence(NamedTuple):
    specification_id: str
    control_ids: tuple[str, ...]
    edge_count: int
    minimum_squared_length: Fraction
    minimum_edge_key: tuple[int, ...]
    maximum_squared_length: Fraction
    maximum_edge_key: tuple[int, ...]
    statistics: tuple[OwnerLocalPeriodicSizeStatistic, ...]
    issues: tuple[str, ...]
    too_small: bool
    too_large: bool
    selected_edge_key: tuple[int, ...] | None
    evidence_id: str

    @property
    def compliant(self) -> bool:
        return not self.issues


class _OwnerLocalPeriodicEdgeEpoch(NamedTuple):
    evidence: OwnerLocalPeriodicSizeEvidence
    edge_owners: dict[tuple[int, ...], tuple[int, ...]]


def _owner_local_periodic_edge_key(
    first_root: int,
    first_winding: tuple[int, ...],
    second_root: int,
    second_winding: tuple[int, ...],
    /,
) -> tuple[int, ...]:
    """Canonical quotient identity of one lifted periodic edge."""
    if len(first_winding) != len(second_winding):
        raise ValueError("Periodic edge winding dimensions disagree.")
    endpoints = sorted(
        (
            (int(first_root), tuple(int(value) for value in first_winding)),
            (int(second_root), tuple(int(value) for value in second_winding)),
        )
    )
    if endpoints[0] == endpoints[1]:
        raise ValueError("A periodic edge must have distinct lifted endpoints.")
    anchor = endpoints[0][1]
    relative = tuple(
        value - anchor[axis]
        for _, winding in endpoints
        for axis, value in enumerate(winding)
    )
    return (
        endpoints[0][0],
        endpoints[1][0],
        *relative,
    )


def _owner_local_periodic_cell_key(
    vertex_root_ids: Sequence[int],
    vertex_windings: Sequence[Sequence[int]],
    /,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Canonical target cell key using the periodic anchor-corner convention."""
    roots = tuple(int(value) for value in vertex_root_ids)
    windings = tuple(tuple(int(value) for value in row) for row in vertex_windings)
    if not roots or len(roots) != len(windings) or not windings[0]:
        raise ValueError("Periodic cell key requires aligned root/winding corners.")
    dimension = len(windings[0])
    if any(len(row) != dimension for row in windings):
        raise ValueError("Periodic cell winding rows have inconsistent dimensions.")
    corners = tuple(zip(roots, windings, strict=True))
    if len(set(corners)) != len(corners):
        raise ValueError("Periodic cell repeats an exact quotient corner image.")
    anchor = min(corners)[1]
    ordered = tuple(
        sorted(
            (
                root,
                tuple(value - anchor[axis] for axis, value in enumerate(winding)),
            )
            for root, winding in corners
        )
    )
    return (
        tuple(root for root, _ in ordered),
        tuple(relative for _, relative in ordered),
        anchor,
    )


def _owner_local_fraction_wire(value: Fraction, /) -> bytes:
    numerator = str(value.numerator).encode("ascii")
    denominator = str(value.denominator).encode("ascii")
    return (
        len(numerator).to_bytes(8, "little")
        + numerator
        + len(denominator).to_bytes(8, "little")
        + denominator
    )


def _owner_local_fraction_from_wire(value: bytes, /) -> Fraction:
    if len(value) < 16:
        raise ValueError("Exact metric fraction payload is truncated.")
    numerator_size = int.from_bytes(value[:8], "little")
    numerator_end = 8 + numerator_size
    if numerator_end + 8 > len(value):
        raise ValueError("Exact metric fraction numerator is truncated.")
    denominator_size = int.from_bytes(value[numerator_end : numerator_end + 8], "little")
    denominator_start = numerator_end + 8
    denominator_end = denominator_start + denominator_size
    if denominator_end != len(value):
        raise ValueError("Exact metric fraction denominator has a malformed extent.")
    return Fraction(
        int(value[8:numerator_end].decode("ascii")),
        int(value[denominator_start:denominator_end].decode("ascii")),
    )


def _owner_local_fraction_order_statistic(
    values: tuple[Fraction, ...],
    order: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> Fraction:
    """Select one exact global order statistic from hash-sharded rational rows."""
    from ._distribution import _owner_group_allgather

    local_start = 0
    local_stop = len(values)
    target = int(order)
    while True:
        local_count = local_stop - local_start
        count_records = _owner_group_allgather(
            np.asarray((local_count,), dtype=np.int64),
            execution_group,
            axis_name,
        )
        counts = np.asarray(count_records, dtype=np.int64).reshape((-1,))
        total = int(np.sum(counts, dtype=np.int64))
        if total <= 0 or not 0 <= target < total:
            raise ValueError("Exact periodic edge selection has an invalid rank.")
        local_pivot = values[local_start + local_count // 2] if local_count else None
        local_payload = (
            b"" if local_pivot is None else _owner_local_fraction_wire(local_pivot)
        )
        payload_sizes = _owner_group_allgather(
            np.asarray((len(local_payload),), dtype=np.int64),
            execution_group,
            axis_name,
        )
        payload_width = max(1, int(np.max(payload_sizes, initial=0)))
        padded_payload = np.zeros((payload_width,), dtype=np.uint8)
        if local_payload:
            padded_payload[: len(local_payload)] = np.frombuffer(
                local_payload, dtype=np.uint8
            )
        pivot_records = _owner_group_allgather(padded_payload, execution_group, axis_name)
        pivots = []
        for rank, count in enumerate(counts.tolist()):
            if count:
                length = int(np.asarray(payload_sizes)[rank, 0])
                payload = np.asarray(pivot_records[rank, :length], dtype=np.uint8)
                pivots.append(
                    (
                        _owner_local_fraction_from_wire(payload.tobytes()),
                        int(count),
                    )
                )
        pivots.sort(key=lambda item: item[0])
        weighted_middle = (total - 1) // 2
        cumulative = 0
        pivot = pivots[-1][0]
        for candidate, weight in pivots:
            cumulative += weight
            if cumulative > weighted_middle:
                pivot = candidate
                break
        local_left = bisect_left(values, pivot, local_start, local_stop)
        local_right = bisect_right(values, pivot, local_start, local_stop)
        local_partition = np.asarray(
            (local_left - local_start, local_right - local_left),
            dtype=np.int64,
        )
        partition_records = _owner_group_allgather(
            local_partition, execution_group, axis_name
        )
        totals = np.sum(np.asarray(partition_records, dtype=np.int64), axis=0)
        less, equal = int(totals[0]), int(totals[1])
        if target < less:
            local_stop = local_left
            continue
        if target < less + equal:
            return pivot
        target -= less + equal
        local_start = local_right


def _owner_local_sqrt_mix_compare(
    first_squared: Fraction,
    second_squared: Fraction,
    interpolation: Fraction,
    bound: Fraction,
    /,
) -> int:
    """Compare an exact linear quantile of two square roots to a rational bound."""
    if bound < 0:
        return 1
    first_weight = Fraction(1) - interpolation
    second_weight = interpolation
    first_term = first_weight * first_weight * first_squared
    second_term = second_weight * second_weight * second_squared
    if first_weight == 0:
        difference = second_term - bound * bound
        return (difference > 0) - (difference < 0)
    if second_weight == 0:
        difference = first_term - bound * bound
        return (difference > 0) - (difference < 0)
    remainder = bound * bound - first_term - second_term
    if remainder < 0:
        return 1
    cross_lower = 4 * first_weight * first_weight * second_weight * second_weight
    cross_value = cross_lower * first_squared * second_squared
    remainder_squared = remainder * remainder
    return (cross_value > remainder_squared) - (cross_value < remainder_squared)


def _owner_local_periodic_edge_records(
    mesh: CellMesh | None,
    cell_owner_ranks: np.ndarray,
    vertex_root_ids: np.ndarray,
    vertex_windings: np.ndarray,
    exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...],
    local_rank: int,
    /,
) -> dict[tuple[int, ...], Fraction]:
    if mesh is None:
        if np.asarray(cell_owner_ranks).size or vertex_root_ids.size:
            raise ValueError("An empty periodic rank has nonempty local entity rows.")
        return {}
    dimension = mesh.topological_dimension
    if dimension not in (2, 3) or mesh.ambient_dimension != dimension:
        raise ValueError(
            "Owner-local periodic metric edges require full-dimensional simplices."
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    roots = np.asarray(vertex_root_ids, dtype=np.int64)
    windings = np.asarray(vertex_windings, dtype=np.int64)
    owners = np.asarray(cell_owner_ranks, dtype=np.int32)
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if (
        roots.shape != vertex_ids.shape
        or windings.shape != (vertex_ids.size, dimension)
        or len(exact_vertex_coordinates) != vertex_ids.size
        or owners.shape != cell_ids.shape
        or any(
            len(row) != dimension or any(not isinstance(value, Fraction) for value in row)
            for row in exact_vertex_coordinates
        )
    ):
        raise ValueError(
            "Periodic exact vertex/cell metadata does not align with the local mesh."
        )
    owner_by_cell = {
        int(identifier): int(owner)
        for identifier, owner in zip(cell_ids, owners, strict=True)
    }
    edge_records: dict[tuple[int, ...], Fraction] = {}
    for block in mesh.blocks:
        if block.cell_kind != ("triangle" if dimension == 2 else "tetrahedron"):
            raise ValueError("Owner-local periodic metric adaptation requires simplices.")
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell_row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            if owner_by_cell.get(int(cell_id)) != local_rank:
                continue
            cell = cells[cell_row]
            for first, second in combinations(range(dimension + 1), 2):
                first_row = int(cell[first])
                second_row = int(cell[second])
                key = _owner_local_periodic_edge_key(
                    int(roots[first_row]),
                    tuple(int(value) for value in windings[first_row]),
                    int(roots[second_row]),
                    tuple(int(value) for value in windings[second_row]),
                )
                squared = sum(
                    (
                        (
                            exact_vertex_coordinates[first_row][axis]
                            - exact_vertex_coordinates[second_row][axis]
                        )
                        ** 2
                        for axis in range(dimension)
                    ),
                    Fraction(0),
                )
                squared_float = float(squared)
                if not np.isfinite(squared_float) or squared_float <= 0.0:
                    raise ValueError(
                        "Exact periodic edge size is not representable in binary64."
                    )
                if squared <= 0:
                    raise ValueError(
                        "Periodic quotient edges require positive exact length."
                    )
                previous = edge_records.get(key)
                if previous is not None and previous != squared:
                    raise ValueError(
                        "One exact periodic edge key has inconsistent lifted metric lengths."
                    )
                edge_records[key] = squared
    return edge_records


def _owner_local_periodic_edge_shard(
    edge_records: dict[tuple[int, ...], Fraction],
    dimension: int,
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[dict[tuple[int, ...], Fraction], dict[tuple[int, ...], tuple[int, ...]]]:
    from ._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    if part_count <= 0:
        raise ValueError("Periodic edge sharding needs a nonempty owner group.")
    if dimension not in (2, 3):
        raise ValueError("Periodic edge sharding requires dimension two or three.")
    key_width = 2 + 2 * dimension
    local_rows: list[tuple[int, tuple[int, ...], bytes]] = []
    local_error = None
    try:
        for key, squared in edge_records.items():
            if len(key) != key_width or squared <= 0:
                raise ValueError("Periodic edge shard records have inconsistent extents.")
            fingerprint = canonical_fingerprint(
                {"kind": "owner-local-periodic-edge-shard", "key": key}
            )
            owner = int(fingerprint[:16], 16) % part_count
            key_bytes = np.asarray(key, dtype="<i8").tobytes()
            fraction = _owner_local_fraction_wire(squared)
            row = key_bytes + len(fraction).to_bytes(8, "little") + fraction
            local_rows.append((owner, key, row))
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic edge shard packing"
    )
    local_width = max(
        (len(row) for _, _, row in local_rows),
        default=key_width * 8 + 8,
    )
    widths = np.asarray(
        _owner_group_allgather(
            np.asarray((local_width,), dtype=np.int64),
            execution_group,
            axis_name,
        ),
        dtype=np.int64,
    ).reshape((-1,))
    width = max(int(np.max(widths, initial=0)), key_width * 8 + 8)
    outgoing_lists: list[list[np.ndarray]] = [[] for _ in range(part_count)]
    local_shard_rows: list[np.ndarray] = []
    for owner, _, row in local_rows:
        padded = np.zeros((width,), dtype=np.uint8)
        padded[: len(row)] = np.frombuffer(row, dtype=np.uint8)
        if owner == local_rank:
            local_shard_rows.append(padded)
        else:
            outgoing_lists[owner].append(padded)
    outgoing = tuple(
        np.stack(rows).astype(np.uint8, copy=False)
        if rows
        else np.zeros((0, width), dtype=np.uint8)
        for rows in outgoing_lists
    )
    incoming = _owner_exchange_message_banks(
        outgoing, local_rank, execution_group, axis_name
    )
    records: dict[tuple[int, ...], Fraction] = {}
    owners_by_key: dict[tuple[int, ...], set[int]] = {}
    parse_error = None
    try:
        banks = {local_rank: np.stack(local_shard_rows)} if local_shard_rows else {}
        banks.update(incoming)
        key_bytes = key_width * 8
        for sender, bank in banks.items():
            for row in np.asarray(bank, dtype=np.uint8):
                key = tuple(
                    int(value)
                    for value in np.frombuffer(row[:key_bytes].tobytes(), dtype="<i8")
                )
                payload_size = int.from_bytes(
                    row[key_bytes : key_bytes + 8].tobytes(), "little"
                )
                payload_start = key_bytes + 8
                payload_stop = payload_start + payload_size
                if payload_stop > row.size:
                    raise ValueError("Periodic edge fraction payload is truncated.")
                squared = _owner_local_fraction_from_wire(
                    row[payload_start:payload_stop].tobytes()
                )
                previous = records.get(key)
                if previous is not None and previous != squared:
                    raise ValueError(
                        "Periodic edge orbit images disagree on exact squared length."
                    )
                records[key] = squared
                owners_by_key.setdefault(key, set()).add(int(sender))
    except Exception as error:
        parse_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        parse_error, execution_group, axis_name, "periodic edge shard reduction"
    )
    return records, {key: tuple(sorted(owners)) for key, owners in owners_by_key.items()}


def _owner_local_periodic_size_epoch(
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    execution_group: ExecutionGroup,
    local_rank: int,
    /,
    *,
    mesh_override: CellMesh | None = None,
    cell_owner_ranks_override: np.ndarray | None = None,
    vertex_root_ids_override: np.ndarray | None = None,
    vertex_windings_override: np.ndarray | None = None,
    exact_vertex_coordinates_override: (tuple[tuple[Fraction, ...], ...] | None) = None,
    use_override: bool = False,
    axis_name: str = "periodic_owner_size",
) -> _OwnerLocalPeriodicEdgeEpoch:
    from .._execution_runtime import ExecutionGroup
    from ._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from ._distribution import _owner_group_allgather, _owner_group_fail_if_any
    from ._periodic import OwnerLocalPeriodicCandidate
    from ._sizing import UniformSizeControl

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if not isinstance(candidate, OwnerLocalPeriodicCandidate):
        raise TypeError("candidate must be OwnerLocalPeriodicCandidate.")
    if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
        raise TypeError("specification must be SurfaceMeshingSpec or VolumeMeshingSpec.")
    members = execution_group.spec.process_indices
    if (
        len(members) != candidate.part_count
        or tuple(device.process_index for device in execution_group.devices) != members
        or not execution_group.is_member
        or not 0 <= local_rank < candidate.part_count
        or candidate.rank != local_rank
        or members[local_rank] != jax.process_index()
    ):
        raise ValueError("Periodic metric ranks do not match their registered group.")
    top_scope = (
        specification.scope
        if isinstance(specification, SurfaceMeshingSpec)
        else specification.boundary_scope
    )
    controls = tuple(specification.size_controls)
    local_error = None
    control_ids: tuple[str, ...] = ()
    target_size: float | None = None
    minimum_sizes: list[float] = []
    maximum_sizes: list[float] = []
    control_values: tuple[UniformSizeControl, ...] = ()
    try:
        control_values = tuple(
            control for control in controls if isinstance(control, UniformSizeControl)
        )
        if not controls or len(control_values) != len(controls):
            raise ValueError(
                "Owner-local periodic metric adaptation requires uniform size controls."
            )
        if (
            specification.protected_features
            or specification.region_controls
            or specification.patch_controls
            or specification.layer_controls
            or (
                isinstance(specification, VolumeMeshingSpec)
                and (specification.region_seeds or specification.hole_seeds)
            )
        ):
            raise ValueError(
                "Owner-local periodic size adaptation does not admit unbound feature, "
                "region, patch, layer, or seed controls."
            )
        if any(
            control.scope.scope_id != top_scope.scope_id
            or control.scope.source_id != candidate.source_id
            or control.scope.source_revision != candidate.source_revision
            for control in control_values
        ):
            raise ValueError(
                "Owner-local periodic size controls must bind the complete common "
                "fundamental-domain scope."
            )
        target_sizes = {float(control.target_size) for control in control_values}
        if len(target_sizes) != 1:
            raise ValueError(
                "Overlapping whole-domain periodic size controls prescribe unequal targets."
            )
        target_size = target_sizes.pop()
        minimum_sizes = [
            float(control.minimum_size)
            for control in control_values
            if control.minimum_size is not None
        ]
        maximum_sizes = [
            float(control.maximum_size)
            for control in control_values
            if control.maximum_size is not None
        ]
        if minimum_sizes and maximum_sizes and max(minimum_sizes) > min(maximum_sizes):
            raise ValueError(
                "Overlapping periodic size bounds have an empty intersection."
            )
        control_ids = tuple(control.control_id for control in control_values)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic size-control validation"
    )
    if local_error is not None or target_size is None:
        raise ValueError(
            "Owner-local periodic size-control validation failed collectively."
        )
    request_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-size-request",
            "source": candidate.source_id,
            "source_revision": candidate.source_revision,
            "source_manifest": candidate.source_manifest_id,
            "distribution": candidate.source_distribution_id,
            "source_part": candidate.source_part_id,
            "authored_mesh": candidate.source_mesh_id,
            "source_topology": candidate.source_topology_id,
            "source_geometry": candidate.source_geometry_id,
            "coordinate_contract": candidate.coordinate_contract_id,
            "periodic_topology": candidate.periodic_topology_id,
            "face_coverage": candidate.face_coverage_id,
            "periodic_cell": candidate.periodic_cell_id,
            "specification": specification.specification_id,
            "controls": control_ids,
            "target_size": target_size,
            "size_policy": specification.size_compliance.policy_id,
            "group": execution_group.spec.group_id,
            "parts": candidate.part_count,
        }
    )
    request_records = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(request_id), dtype=np.uint8),
        execution_group,
        axis_name,
    )
    if any(
        np.asarray(record, dtype=np.uint8).tobytes().hex() != request_id
        for record in request_records
    ):
        raise ValueError("Periodic metric ranks supplied different size requests.")
    local_error = None
    try:
        if use_override:
            if (
                cell_owner_ranks_override is None
                or vertex_root_ids_override is None
                or vertex_windings_override is None
                or exact_vertex_coordinates_override is None
            ):
                raise ValueError(
                    "Periodic target metric rows require full exact metadata."
                )
            metric_mesh = mesh_override
            metric_cell_owners = cell_owner_ranks_override
            metric_root_ids = vertex_root_ids_override
            metric_windings = vertex_windings_override
            metric_exact_coordinates = exact_vertex_coordinates_override
        else:
            metric_mesh = candidate.local_mesh
            metric_cell_owners = np.asarray(candidate.cell_owner_ranks, dtype=np.int32)
            metric_root_ids = np.asarray(candidate.vertex_source_ids, dtype=np.int64)
            metric_windings = np.asarray(
                candidate.vertex_absolute_windings, dtype=np.int64
            )
            metric_exact_coordinates = candidate.exact_vertex_coordinates
        local_records = _owner_local_periodic_edge_records(
            metric_mesh,
            np.asarray(metric_cell_owners, dtype=np.int32),
            np.asarray(metric_root_ids, dtype=np.int64),
            np.asarray(metric_windings, dtype=np.int64),
            metric_exact_coordinates,
            local_rank,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        local_records = {}
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic edge metric input"
    )
    shard_records, edge_owners = _owner_local_periodic_edge_shard(
        local_records,
        candidate.dimension,
        local_rank,
        candidate.part_count,
        execution_group,
        axis_name,
    )
    local_values = tuple(sorted(shard_records.values()))
    count_rows = _owner_group_allgather(
        np.asarray((len(local_values),), dtype=np.int64),
        execution_group,
        axis_name,
    )
    edge_count = int(np.sum(np.asarray(count_rows, dtype=np.int64), dtype=np.int64))
    if edge_count <= 0:
        raise ValueError("Periodic size compliance requires at least one quotient edge.")
    order_cache: dict[int, Fraction] = {}

    def order_value(order: int, /) -> Fraction:
        cached = order_cache.get(order)
        if cached is None:
            cached = _owner_local_fraction_order_statistic(
                local_values, order, execution_group, axis_name
            )
            order_cache[order] = cached
        return cached

    minimum_squared = order_value(0)
    maximum_squared = order_value(edge_count - 1)

    def extreme_key(squared: Fraction, /) -> tuple[int, ...]:
        candidates = [key for key, value in shard_records.items() if value == squared]
        local_key = (
            min(candidates)
            if candidates
            else tuple(0 for _ in range(2 + 2 * candidate.dimension))
        )
        local_record = np.asarray((int(bool(candidates)), *local_key), dtype=np.int64)
        gathered = _owner_group_allgather(local_record, execution_group, axis_name)
        valid = [
            tuple(int(value) for value in row[1:])
            for row in np.asarray(gathered, dtype=np.int64)
            if int(row[0])
        ]
        if not valid:
            raise ValueError("Exact periodic size extreme has no quotient edge key.")
        return min(valid)

    minimum_key = extreme_key(minimum_squared)
    maximum_key = extreme_key(maximum_squared)
    statistics: list[OwnerLocalPeriodicSizeStatistic] = []
    issues: list[str] = []
    too_large = False
    too_small = False
    tolerance = specification.size_compliance
    for control in control_values:
        target = Fraction.from_float(float(control.target_size))
        allowed_float = (
            tolerance.absolute_tolerance
            + tolerance.relative_tolerance * float(control.target_size)
        )
        if not np.isfinite(allowed_float):
            raise ValueError("Periodic size compliance tolerance overflows binary64.")
        allowed = Fraction.from_float(float(allowed_float))
        for name in tolerance.target_statistics:
            quantile = Fraction(1, 2) if name == "p50" else Fraction(19, 20)
            position = quantile * (edge_count - 1)
            lower_order = position.numerator // position.denominator
            upper_order = (
                position.numerator + position.denominator - 1
            ) // position.denominator
            interpolation = position - lower_order
            lower_squared = order_value(lower_order)
            upper_squared = order_value(upper_order)
            achieved = float(
                np.sqrt(float(lower_squared))
                + float(interpolation)
                * (np.sqrt(float(upper_squared)) - np.sqrt(float(lower_squared)))
            )
            if not np.isfinite(achieved):
                raise ValueError(
                    "Periodic size statistic is not representable in binary64."
                )
            lower_bound = max(Fraction(0), target - allowed)
            upper_bound = target + allowed
            below = (
                _owner_local_sqrt_mix_compare(
                    lower_squared,
                    upper_squared,
                    interpolation,
                    lower_bound,
                )
                < 0
            )
            above = (
                _owner_local_sqrt_mix_compare(
                    lower_squared,
                    upper_squared,
                    interpolation,
                    upper_bound,
                )
                > 0
            )
            accepted = not below and not above
            statistics.append(
                OwnerLocalPeriodicSizeStatistic(
                    control.control_id,
                    name,
                    float(control.target_size),
                    lower_squared,
                    upper_squared,
                    interpolation,
                    achieved,
                    below,
                    above,
                    accepted,
                )
            )
            if below:
                too_small = True
                issues.append(
                    f"periodic size {name} statistic {achieved:g} is below the "
                    f"declared target interval for control {control.control_id}."
                )
            if above:
                too_large = True
                issues.append(
                    f"periodic size {name} statistic {achieved:g} is above the "
                    f"declared target interval for control {control.control_id}."
                )
        if control.minimum_size is not None:
            minimum = Fraction.from_float(float(control.minimum_size))
            if minimum_squared < minimum * minimum:
                too_small = True
                issues.append(
                    f"periodic edge minimum violates control {control.control_id}."
                )
        if control.maximum_size is not None:
            maximum = Fraction.from_float(float(control.maximum_size))
            if maximum_squared > maximum * maximum:
                too_large = True
                issues.append(
                    f"periodic edge maximum violates control {control.control_id}."
                )
    selected_edge_key = maximum_key if too_large else None
    statistic_payload = tuple(
        (
            value.control_id,
            value.statistic,
            value.target_size,
            (
                value.lower_squared_length.numerator,
                value.lower_squared_length.denominator,
            ),
            (
                value.upper_squared_length.numerator,
                value.upper_squared_length.denominator,
            ),
            (value.interpolation.numerator, value.interpolation.denominator),
            value.achieved,
            value.below,
            value.above,
            value.accepted,
        )
        for value in statistics
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-size-evidence",
            "source": candidate.source_id,
            "source_revision": candidate.source_revision,
            "source_manifest": candidate.source_manifest_id,
            "distribution": candidate.source_distribution_id,
            "source_part": candidate.source_part_id,
            "authored_mesh": candidate.source_mesh_id,
            "source_topology": candidate.source_topology_id,
            "source_geometry": candidate.source_geometry_id,
            "coordinate_contract": candidate.coordinate_contract_id,
            "periodic_topology": candidate.periodic_topology_id,
            "face_coverage": candidate.face_coverage_id,
            "periodic_cell": candidate.periodic_cell_id,
            "specification": specification.specification_id,
            "group": execution_group.spec.group_id,
            "controls": control_ids,
            "edge_count": edge_count,
            "minimum_squared": (
                minimum_squared.numerator,
                minimum_squared.denominator,
            ),
            "minimum_key": minimum_key,
            "maximum_squared": (
                maximum_squared.numerator,
                maximum_squared.denominator,
            ),
            "maximum_key": maximum_key,
            "statistics": statistic_payload,
            "issues": tuple(issues),
            "too_small": too_small,
            "too_large": too_large,
            "selected_edge": selected_edge_key,
        }
    )
    evidence = OwnerLocalPeriodicSizeEvidence(
        specification.specification_id,
        control_ids,
        edge_count,
        minimum_squared,
        minimum_key,
        maximum_squared,
        maximum_key,
        tuple(statistics),
        tuple(issues),
        too_small,
        too_large,
        selected_edge_key,
        evidence_id,
    )
    return _OwnerLocalPeriodicEdgeEpoch(evidence, edge_owners)


def _owner_local_size_statistic_compare(
    left: OwnerLocalPeriodicSizeStatistic,
    right: OwnerLocalPeriodicSizeStatistic,
    /,
) -> int:
    """Exact comparison of two nonnegative linearly interpolated edge lengths."""

    def sign(value: Fraction) -> int:
        return int(value > 0) - int(value < 0)

    def rational_plus_root(
        rational: Fraction, coefficient: Fraction, radicand: Fraction
    ) -> int:
        if coefficient == 0 or radicand == 0:
            return sign(rational)
        if rational == 0 or sign(rational) == sign(coefficient):
            return sign(coefficient)
        squared = rational * rational - coefficient * coefficient * radicand
        return sign(rational) * sign(squared)

    a, b = Fraction(1) - left.interpolation, left.interpolation
    c, d = Fraction(1) - right.interpolation, right.interpolation
    x, y = left.lower_squared_length, left.upper_squared_length
    u, v = right.lower_squared_length, right.upper_squared_length
    rational = a * a * x + b * b * y - c * c * u - d * d * v
    positive_root = 4 * a * a * b * b * x * y
    negative_root = 4 * c * c * d * d * u * v
    left_sign = rational_plus_root(rational, Fraction(1), positive_root)
    if left_sign < 0:
        return -1
    if left_sign == 0:
        return -int(negative_root > 0)
    return rational_plus_root(
        rational * rational + positive_root - negative_root,
        2 * rational,
        positive_root,
    )


def _owner_local_periodic_coarsen_is_admissible(
    before: OwnerLocalPeriodicSizeEvidence,
    after: OwnerLocalPeriodicSizeEvidence,
    /,
) -> bool:
    if (after.too_small and not before.too_small) or (
        after.too_large and not before.too_large
    ):
        return False
    if (
        before.too_small
        and after.too_small
        and (after.minimum_squared_length < before.minimum_squared_length)
    ):
        return False
    if (
        before.too_large
        and after.too_large
        and (after.maximum_squared_length > before.maximum_squared_length)
    ):
        return False
    if len(before.statistics) != len(after.statistics):
        return False
    for old, new in zip(before.statistics, after.statistics, strict=True):
        if (new.below and not old.below) or (new.above and not old.above):
            return False
        if old.below and new.below and _owner_local_size_statistic_compare(new, old) < 0:
            return False
        if old.above and new.above and _owner_local_size_statistic_compare(new, old) > 0:
            return False
    return True


class _OwnerLocalPeriodicState(NamedTuple):
    layout: AdaptiveSimplexLayout
    state: AdaptiveSimplexState
    source_mesh: CellMesh | None
    source_cell_ids: np.ndarray
    source_measures: dict[int, Fraction]
    exact_coordinates: dict[int, tuple[Fraction, ...]]
    vertex_root_ids: dict[int, int]
    vertex_windings: dict[int, tuple[int, ...]]
    vertex_owner_ranks: dict[int, int]


def _owner_local_periodic_state(
    candidate: OwnerLocalPeriodicCandidate,
    local_rank: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
    *,
    live_forest: PeriodicSimplexForest | None = None,
) -> _OwnerLocalPeriodicState:
    from .._execution_runtime import ExecutionGroup
    from ._distribution import _owner_group_allgather, _owner_group_fail_if_any
    from ._periodic import OwnerLocalPeriodicCandidate

    if not isinstance(candidate, OwnerLocalPeriodicCandidate):
        raise TypeError("candidate must be OwnerLocalPeriodicCandidate.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    dimension = int(candidate.dimension)
    local_mesh = candidate.local_mesh
    vertex_ids = np.asarray(candidate.vertex_ids, dtype=np.int64)
    vertex_owners = np.asarray(candidate.vertex_owner_ranks, dtype=np.int32)
    cell_ids = np.asarray(candidate.cell_ids, dtype=np.int64)
    cell_owners = np.asarray(candidate.cell_owner_ranks, dtype=np.int32)
    vertex_max = int(np.max(vertex_ids[vertex_owners == local_rank], initial=-1))
    cell_max = int(np.max(cell_ids[cell_owners == local_rank], initial=-1))
    maxima = np.asarray(
        _owner_group_allgather(
            np.asarray((vertex_max, cell_max), dtype=np.int64),
            execution_group,
            axis_name,
        ),
        dtype=np.int64,
    )
    maximum_vertex_id = int(np.max(maxima[:, 0], initial=-1))
    maximum_cell_id = int(np.max(maxima[:, 1], initial=-1))
    local_error = (
        "Periodic source identifiers leave no representable int64 successor."
        if max(maximum_vertex_id, maximum_cell_id) >= np.iinfo(np.int64).max
        else None
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source ID range"
    )
    next_vertex_id = maximum_vertex_id + 1
    next_cell_id = maximum_cell_id + 1
    local_error = None
    source_mesh: CellMesh | None = None
    source_cell_ids = np.zeros((0,), dtype=np.int64)
    source_measures: dict[int, Fraction] = {}
    exact_coordinates: dict[int, tuple[Fraction, ...]] = {}
    root_ids_by_vertex: dict[int, int] = {}
    windings_by_vertex: dict[int, tuple[int, ...]] = {}
    owner_ranks_by_vertex: dict[int, int] = {}
    state = None
    layout = None
    try:
        if dimension not in (2, 3):
            raise ValueError(
                "Owner-local periodic bisection requires dimension two or three."
            )
        if local_mesh is None:
            selected_blocks: tuple[CellBlock, ...] = ()
            used_vertices = np.zeros((0,), dtype=np.int64)
        else:
            if (
                not isinstance(local_mesh, CellMesh)
                or local_mesh.topological_dimension != dimension
                or local_mesh.ambient_dimension != dimension
            ):
                raise ValueError(
                    "Periodic candidate local mesh has the wrong dimensions."
                )
            owner_by_id = {
                int(identifier): int(owner)
                for identifier, owner in zip(cell_ids, cell_owners, strict=True)
            }
            selected: list[tuple[str, str, np.ndarray, np.ndarray]] = []
            used = []
            for block in local_mesh.blocks:
                rows = np.asarray(block.vertices, dtype=np.int64)
                identifiers = np.asarray(block.global_ids, dtype=np.int64)
                if block.cell_kind != ("triangle" if dimension == 2 else "tetrahedron"):
                    raise ValueError("Periodic candidate contains a non-simplex block.")
                owned_rows = np.asarray(
                    [
                        row
                        for row, identifier in enumerate(identifiers.tolist())
                        if owner_by_id[int(identifier)] == local_rank
                    ],
                    dtype=np.int64,
                )
                if owned_rows.size:
                    cell_vertices = rows[owned_rows]
                    selected.append(
                        (
                            block.name,
                            block.cell_kind,
                            cell_vertices,
                            identifiers[owned_rows],
                        )
                    )
                    used.append(cell_vertices.reshape((-1,)))
            if selected:
                old_vertex_ids = np.asarray(local_mesh.vertex_global_ids, dtype=np.int64)
                used_vertices = np.unique(np.concatenate(used))
                used_vertices = used_vertices[
                    np.argsort(old_vertex_ids[used_vertices], kind="stable")
                ]
                old_to_local = np.full((old_vertex_ids.size,), -1, dtype=np.int64)
                old_to_local[used_vertices] = np.arange(
                    used_vertices.size, dtype=np.int64
                )
                selected_blocks = tuple(
                    CellBlock(
                        name,
                        kind,
                        old_to_local[vertices].astype(np.int32),
                        global_ids=identifiers,
                    )
                    for name, kind, vertices, identifiers in selected
                )
                source_mesh = CellMesh(
                    np.asarray(local_mesh.coordinates, dtype=np.float64)[used_vertices],
                    selected_blocks,
                    vertex_global_ids=old_vertex_ids[used_vertices],
                    numeric_version=(
                        f"owner-local-periodic-source:{candidate.candidate_id}:"
                        f"{local_rank}"
                    ),
                )
                source_cell_ids = np.asarray(
                    source_mesh.entity_set(dimension).entity_ids, dtype=np.int64
                )
                exact_bank = candidate.exact_vertex_coordinates
                root_bank = np.asarray(candidate.vertex_source_ids, dtype=np.int64)
                winding_bank = np.asarray(
                    candidate.vertex_absolute_windings, dtype=np.int64
                )
                candidate_vertex_ids = np.asarray(candidate.vertex_ids, dtype=np.int64)
                row_by_id = {
                    int(identifier): row
                    for row, identifier in enumerate(candidate_vertex_ids.tolist())
                }
                source_vertex_ids = np.asarray(
                    source_mesh.vertex_global_ids, dtype=np.int64
                )
                source_roots = []
                source_windings = []
                for source_row, identifier in enumerate(source_vertex_ids.tolist()):
                    candidate_row = row_by_id[int(identifier)]
                    exact_row = tuple(exact_bank[candidate_row])
                    if tuple(float(value) for value in exact_row) != tuple(
                        np.asarray(source_mesh.coordinates, dtype=np.float64)[
                            source_row
                        ].tolist()
                    ):
                        raise ValueError(
                            "Periodic source exact coordinates disagree with the local mesh."
                        )
                    exact_coordinates[int(identifier)] = exact_row
                    owner_ranks_by_vertex[int(identifier)] = int(
                        vertex_owners[candidate_row]
                    )
                    source_roots.append(int(root_bank[candidate_row]))
                    root_ids_by_vertex[int(identifier)] = int(root_bank[candidate_row])
                    source_windings.append(
                        tuple(int(value) for value in winding_bank[candidate_row])
                    )
                    windings_by_vertex[int(identifier)] = source_windings[-1]
                source_metadata = _owner_local_periodic_cell_metadata(
                    source_mesh,
                    np.asarray(source_roots, dtype=np.int64),
                    np.asarray(source_windings, dtype=np.int64).reshape((-1, dimension)),
                    dimension,
                )
                candidate_cell_ids = np.asarray(candidate.cell_ids, dtype=np.int64)
                candidate_offsets = np.asarray(candidate.cell_key_offsets, dtype=np.int64)
                candidate_key_source_ids = np.asarray(
                    candidate.cell_key_source_ids, dtype=np.int64
                )
                candidate_relative_windings = np.asarray(
                    candidate.cell_key_relative_windings, dtype=np.int64
                )
                candidate_cell_windings = np.asarray(
                    candidate.cell_windings, dtype=np.int64
                )
                if (
                    candidate_offsets.shape != (candidate_cell_ids.size + 1,)
                    or candidate_offsets[0] != 0
                    or np.any(np.diff(candidate_offsets) != dimension + 1)
                    or candidate_offsets[-1] != candidate_key_source_ids.size
                    or candidate_relative_windings.shape
                    != (candidate_key_source_ids.size, dimension)
                    or candidate_cell_windings.shape
                    != (candidate_cell_ids.size, dimension)
                ):
                    raise ValueError(
                        "Periodic source cell quotient-key banks are malformed."
                    )
                candidate_cell_row = {
                    int(identifier): row
                    for row, identifier in enumerate(candidate_cell_ids.tolist())
                }
                for source_row, identifier in enumerate(
                    source_metadata.cell_ids.tolist()
                ):
                    candidate_row = candidate_cell_row[int(identifier)]
                    start, stop = map(
                        int, candidate_offsets[candidate_row : candidate_row + 2]
                    )
                    if (
                        stop - start != dimension + 1
                        or not np.array_equal(
                            candidate_key_source_ids[start:stop],
                            source_metadata.cell_key_source_ids[
                                source_metadata.cell_key_offsets[
                                    source_row
                                ] : source_metadata.cell_key_offsets[source_row + 1]
                            ],
                        )
                        or not np.array_equal(
                            candidate_relative_windings[start:stop],
                            source_metadata.cell_key_relative_windings[
                                source_metadata.cell_key_offsets[
                                    source_row
                                ] : source_metadata.cell_key_offsets[source_row + 1]
                            ],
                        )
                        or not np.array_equal(
                            candidate_cell_windings[candidate_row],
                            source_metadata.cell_windings[source_row],
                        )
                    ):
                        raise ValueError(
                            "Periodic source cell keys disagree with its exact vertex orbit."
                        )
                candidate_measures = {
                    int(identifier): value
                    for identifier, value in zip(
                        candidate_cell_ids.tolist(),
                        candidate.exact_cell_measures,
                        strict=True,
                    )
                }
                source_measures = {
                    int(identifier): candidate_measures[int(identifier)]
                    for identifier in source_cell_ids.tolist()
                }
                exact_source_measures = _owner_local_periodic_exact_cell_measures(
                    source_mesh, exact_coordinates, dimension
                )
                if exact_source_measures != tuple(
                    source_measures[int(identifier)]
                    for identifier in source_metadata.cell_ids.tolist()
                ):
                    raise ValueError(
                        "Periodic source exact measures disagree with its coordinate bank."
                    )
        if source_mesh is None:
            cells = np.zeros((0, dimension + 1), dtype=np.int32)
            tuples = np.zeros((0, dimension + 1), dtype=np.int32)
            tags = np.zeros((0,), dtype=np.int32)
            blocks = np.zeros((0,), dtype=np.int32)
            generations = np.zeros((0,), dtype=np.int32)
            state_vertex_ids = np.zeros((0,), dtype=np.int64)
            coordinates = np.zeros((0, dimension), dtype=np.float64)
        else:
            from ._bisection import _prepared_source

            prepared = _prepared_source(source_mesh)
            tuples, tags = _longest_edge_labels(
                prepared.cells, prepared.coordinates, dimension
            )
            cells = np.asarray(prepared.cells.rows, dtype=np.int32)
            blocks = np.asarray(prepared.cells.blocks, dtype=np.int32)
            generations = np.asarray(prepared.cells.generations, dtype=np.int32)
            state_vertex_ids = np.asarray(prepared.vertex_ids, dtype=np.int64)
            coordinates = np.asarray(prepared.coordinates, dtype=np.float64)
            source_cell_ids = np.asarray(prepared.cells.ids, dtype=np.int64)
        layout = AdaptiveSimplexLayout(
            dimension,
            dimension,
            vertex_capacity=adaptive_simplex_bucket(max(state_vertex_ids.size, 1), 1.0),
            cell_capacity=adaptive_simplex_bucket(max(source_cell_ids.size, 1), 1.0),
            protected_edge_capacity=1,
        )
        cell_count = source_cell_ids.size
        vertex_count = state_vertex_ids.size
        state = adaptive_simplex_state(
            layout,
            coordinates=coordinates,
            vertex_ids=state_vertex_ids,
            vertex_active=np.ones((vertex_count,), dtype=np.bool_),
            vertex_parents=np.full((vertex_count, 2), -1, dtype=np.int64),
            vertex_protected=np.ones((vertex_count,), dtype=np.bool_),
            cells=cells,
            tuples=np.asarray(tuples, dtype=np.int32),
            tags=np.asarray(tags, dtype=np.int32),
            blocks=np.asarray(blocks, dtype=np.int32),
            generations=np.asarray(generations, dtype=np.int32),
            parents=np.full((cell_count,), -1, dtype=np.int64),
            children=np.full((cell_count, 2), -1, dtype=np.int64),
            bisection_vertices=np.full((cell_count,), -1, dtype=np.int64),
            cell_ids=source_cell_ids,
            cell_active=np.ones((cell_count,), dtype=np.bool_),
            cell_classes=np.zeros((cell_count,), dtype=np.int32),
            facet_classes=np.zeros((cell_count, dimension + 1), dtype=np.int32),
            protected_edges=np.zeros((0, 2), dtype=np.int64),
            next_vertex_id=next_vertex_id,
            next_cell_id=next_cell_id,
        )
        if live_forest is not None:
            records = live_forest.records
            if (
                live_forest.source_id != candidate.source_id
                or live_forest.source_revision != candidate.source_revision
                or live_forest.periodic_cell_id != candidate.periodic_cell_id
                or records.dimension != dimension
            ):
                raise ValueError(
                    "Live periodic family belongs to another source or quotient."
                )
            fields = records.cell_field_map()
            forest_ids = np.asarray(records.cell_ids, dtype=np.int64)
            active_ids = forest_ids[np.asarray(fields["active"], dtype=np.bool_)]
            if not np.array_equal(active_ids, source_cell_ids):
                raise ValueError(
                    "Live periodic families do not cover the current owned cells."
                )
            forest_vertices = np.asarray(records.vertex_ids, dtype=np.int64)
            for row, identifier in enumerate(forest_vertices):
                key = int(identifier)
                exact = live_forest.exact_vertex_coordinates[row]
                root = int(live_forest.vertex_source_ids[row])
                winding = tuple(
                    map(int, np.asarray(live_forest.vertex_absolute_windings)[row])
                )
                if (
                    key not in exact_coordinates
                    or exact_coordinates[key] != exact
                    or root_ids_by_vertex[key] != root
                    or windings_by_vertex[key] != winding
                ):
                    raise ValueError(
                        "Live periodic family geometry or winding differs from the current mesh."
                    )
            layout = AdaptiveSimplexLayout(
                dimension,
                dimension,
                vertex_capacity=adaptive_simplex_bucket(
                    max(forest_vertices.size, 1), 1.0
                ),
                cell_capacity=adaptive_simplex_bucket(max(forest_ids.size, 1), 1.0),
                protected_edge_capacity=max(records.protected_edge_ids.shape[0], 1),
            )
            state = reconstruct_adaptive_simplex_state(records, state, layout)
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic adaptive source state"
    )
    if local_error is not None or state is None or layout is None:
        raise ValueError("Owner-local periodic initial state failed collectively.")
    return _OwnerLocalPeriodicState(
        layout,
        state,
        source_mesh,
        source_cell_ids,
        source_measures,
        exact_coordinates,
        root_ids_by_vertex,
        windings_by_vertex,
        owner_ranks_by_vertex,
    )


def _owner_local_allocate_exact_ids(
    keys: Sequence[tuple[int, ...]],
    key_width: int,
    first_identifier: int,
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
    *,
    group_key_width: int | None = None,
) -> tuple[
    dict[tuple[int, ...], int],
    int,
    dict[tuple[int, ...], int],
]:
    """Assign collision-free IDs in exact range-shard order."""
    from ._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    group_width = key_width if group_key_width is None else group_key_width
    local_error = None
    if (
        part_count <= 0
        or not 0 <= local_rank < part_count
        or key_width <= 0
        or not 1 <= group_width <= key_width
        or first_identifier < 0
        or first_identifier > np.iinfo(np.int64).max
    ):
        local_error = "Periodic exact-ID allocation inputs are invalid."
    local_keys = tuple(sorted(set(tuple(int(value) for value in key) for key in keys)))
    if any(
        len(key) != key_width
        or not key
        or key[0] < 0
        or any(
            value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max
            for value in key
        )
        for key in local_keys
    ):
        local_error = "Periodic exact-ID key rows are malformed or outside int64."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic exact-ID input"
    )
    layout = _owner_group_allgather(
        np.asarray(
            (key_width, group_width, first_identifier, part_count),
            dtype=np.int64,
        ),
        execution_group,
        axis_name,
    )
    layout_values = np.asarray(layout, dtype=np.int64)
    if layout_values.shape != (part_count, 4) or np.any(
        layout_values != layout_values[0]
    ):
        raise ValueError("Periodic exact-ID allocation inputs differ across ranks.")
    outgoing_rows: list[list[tuple[int, ...]]] = [[] for _ in range(part_count)]
    local_rows: list[tuple[int, ...]] = []
    for local_index, key in enumerate(local_keys):
        bucket = (int(key[0]) * part_count) // (1 << 63)
        if not 0 <= bucket < part_count:
            raise ValueError("Periodic exact-ID key is outside its range shard.")
        row = (local_index, *key)
        if bucket == local_rank:
            local_rows.append(row)
        else:
            outgoing_rows[bucket].append(row)
    outgoing = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, key_width + 1))
        if owner != local_rank
        else np.zeros((0, key_width + 1), dtype=np.int64)
        for owner, rows in enumerate(outgoing_rows)
    )
    incoming = _owner_exchange_message_banks(
        outgoing, local_rank, execution_group, axis_name
    )
    requests_by_sender: dict[int, list[tuple[int, ...]]] = {}
    shard_keys = set()
    for row in local_rows:
        key = tuple(row[1:])
        shard_keys.add(key)
        requests_by_sender.setdefault(local_rank, []).append(row)
    for sender, bank in incoming.items():
        for row in np.asarray(bank, dtype=np.int64):
            key = tuple(int(value) for value in row[1:])
            shard_keys.add(key)
            requests_by_sender.setdefault(int(sender), []).append((int(row[0]), *key))
    ordered_keys = tuple(sorted(shard_keys))
    counts = np.asarray(
        _owner_group_allgather(
            np.asarray((len(ordered_keys),), dtype=np.int64),
            execution_group,
            axis_name,
        ),
        dtype=np.int64,
    ).reshape((-1,))
    if counts.shape != (part_count,) or np.any(counts < 0):
        raise ValueError("Periodic exact-ID shard counts are malformed.")
    prefix = int(np.sum(counts[:local_rank], dtype=np.int64))
    total = int(np.sum(counts, dtype=np.int64))
    next_identifier = first_identifier + total
    local_error = (
        "Periodic exact IDs exceed the stable int64 identifier range."
        if next_identifier > np.iinfo(np.int64).max
        else None
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic exact-ID range"
    )
    shard_ids = {
        key: first_identifier + prefix + row for row, key in enumerate(ordered_keys)
    }
    group_ids: dict[tuple[int, ...], int] = {}
    group_by_prefix: dict[tuple[int, ...], int] = {}
    for key in ordered_keys:
        prefix_key = key[:group_width]
        group_identifier = group_by_prefix.setdefault(prefix_key, shard_ids[key])
        group_ids[key] = group_identifier
    response_rows: list[list[tuple[int, ...]]] = [[] for _ in range(part_count)]
    local_result: dict[tuple[int, ...], int] = {}
    local_group_result: dict[tuple[int, ...], int] = {}
    for sender, rows in requests_by_sender.items():
        for local_index, *key_values in rows:
            key = tuple(key_values)
            identifier = shard_ids[key]
            group_identifier = group_ids[key]
            if sender == local_rank:
                local_result[key] = identifier
                local_group_result[key] = group_identifier
            else:
                response_rows[sender].append(
                    (local_index, *key, identifier, group_identifier)
                )
    responses = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, key_width + 3))
        if owner != local_rank
        else np.zeros((0, key_width + 3), dtype=np.int64)
        for owner, rows in enumerate(response_rows)
    )
    returned = _owner_exchange_message_banks(
        responses, local_rank, execution_group, axis_name
    )
    reply_error = None
    try:
        for bank in returned.values():
            for row in np.asarray(bank, dtype=np.int64):
                local_index = int(row[0])
                key = tuple(int(value) for value in row[1 : key_width + 1])
                identifier = int(row[-2])
                group_identifier = int(row[-1])
                if (
                    local_index < 0
                    or local_index >= len(local_keys)
                    or key != local_keys[local_index]
                ):
                    raise ValueError(
                        "Periodic exact-ID reply does not match its request."
                    )
                local_result[key] = identifier
                local_group_result[key] = group_identifier
        if set(local_result) != set(local_keys) or set(local_group_result) != set(
            local_keys
        ):
            raise ValueError("Periodic exact-ID allocation omitted a local key.")
    except Exception as error:
        reply_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        reply_error, execution_group, axis_name, "periodic exact-ID reply"
    )
    return local_result, next_identifier, local_group_result


def _reindex_owner_local_periodic_state(
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    next_vertex_id: int,
    next_cell_id: int,
    /,
) -> AdaptiveSimplexState:
    """Sort global IDs after collective allocation and remap every local slot."""
    from ..discretization._adaptive_simplex import _state, _work

    work = _work(state)
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    cursors = np.asarray(work.cursors, dtype=np.int64).copy()
    vertex_count, cell_count = int(cursors[0]), int(cursors[1])
    vertex_ids = np.asarray(work.vertex_ids, dtype=np.int64)
    cell_ids = np.asarray(work.cell_ids, dtype=np.int64)
    vertex_order = np.concatenate(
        (
            np.argsort(vertex_ids[:vertex_count], kind="stable"),
            np.arange(vertex_count, vertex_capacity, dtype=np.int64),
        )
    )
    cell_order = np.concatenate(
        (
            np.argsort(cell_ids[:cell_count], kind="stable"),
            np.arange(cell_count, cell_capacity, dtype=np.int64),
        )
    )
    if (
        np.unique(vertex_ids[:vertex_count]).size != vertex_count
        or np.unique(cell_ids[:cell_count]).size != cell_count
    ):
        raise ValueError(
            "Periodic adaptive target IDs must be unique in each local state."
        )
    vertex_slot = np.empty((vertex_capacity,), dtype=np.int64)
    vertex_slot[vertex_order] = np.arange(vertex_capacity, dtype=np.int64)
    cell_slot = np.empty((cell_capacity,), dtype=np.int64)
    cell_slot[cell_order] = np.arange(cell_capacity, dtype=np.int64)

    def reorder_rows(values: Any, order: np.ndarray) -> np.ndarray:
        return np.asarray(values)[order].copy()

    coordinates = reorder_rows(work.coordinates, vertex_order)
    vertex_ids = reorder_rows(work.vertex_ids, vertex_order)
    vertex_active = reorder_rows(work.vertex_active, vertex_order)
    vertex_parents = reorder_rows(work.vertex_parents, vertex_order)
    vertex_levels = reorder_rows(work.vertex_levels, vertex_order)
    vertex_removal = reorder_rows(work.vertex_removal, vertex_order)
    vertex_protected = reorder_rows(work.vertex_protected, vertex_order)
    valid_vertex_parents = vertex_parents[:vertex_count] >= 0
    vertex_parents[:vertex_count][valid_vertex_parents] = vertex_slot[
        vertex_parents[:vertex_count][valid_vertex_parents]
    ]
    valid_removal = vertex_removal[:vertex_count] >= 0
    vertex_removal[:vertex_count][valid_removal] = vertex_slot[
        vertex_removal[:vertex_count][valid_removal]
    ]

    cells = reorder_rows(work.cells, cell_order)
    tuples = reorder_rows(work.tuples, cell_order)
    cell_ids = reorder_rows(work.cell_ids, cell_order)
    cell_active = reorder_rows(work.cell_active, cell_order)
    tags = reorder_rows(work.tags, cell_order)
    blocks = reorder_rows(work.blocks, cell_order)
    generations = reorder_rows(work.generations, cell_order)
    parents = reorder_rows(work.parents, cell_order)
    children = reorder_rows(work.children, cell_order)
    bisection_vertices = reorder_rows(work.bisection_vertices, cell_order)
    retired = reorder_rows(work.retired, cell_order)
    cell_classes = reorder_rows(work.cell_classes, cell_order)
    facet_classes = reorder_rows(work.facet_classes, cell_order)
    refine_rejected = reorder_rows(work.refine_rejected, cell_order)
    coarsen_marked = reorder_rows(work.coarsen_marked, cell_order)
    cells[:cell_count] = vertex_slot[cells[:cell_count]]
    tuples[:cell_count] = vertex_slot[tuples[:cell_count]]
    valid_parents = parents[:cell_count] >= 0
    parent_rows = parents[:cell_count][valid_parents] // 2
    parent_ordinal = parents[:cell_count][valid_parents] % 2
    parents[:cell_count][valid_parents] = 2 * cell_slot[parent_rows] + parent_ordinal
    valid_children = children[:cell_count] >= 0
    children[:cell_count][valid_children] = cell_slot[
        children[:cell_count][valid_children]
    ]
    valid_midpoints = bisection_vertices[:cell_count] >= 0
    bisection_vertices[:cell_count][valid_midpoints] = vertex_slot[
        bisection_vertices[:cell_count][valid_midpoints]
    ]
    protected_codes = np.asarray(work.protected_codes, dtype=np.int64)
    sentinel = np.iinfo(np.int64).max
    selected_codes = protected_codes[protected_codes != sentinel]
    if selected_codes.size:
        first = selected_codes // vertex_capacity
        second = selected_codes % vertex_capacity
        first, second = vertex_slot[first], vertex_slot[second]
        selected_codes = np.sort(
            np.minimum(first, second) * vertex_capacity + np.maximum(first, second)
        )
        protected_codes = np.full(
            (layout.protected_edge_capacity,), sentinel, dtype=np.int64
        )
        protected_codes[: selected_codes.size] = selected_codes
    cursors[2] = int(next_vertex_id)
    cursors[3] = int(next_cell_id)
    return _state(
        work._replace(
            coordinates=jnp.asarray(coordinates),
            vertex_ids=jnp.asarray(vertex_ids),
            vertex_active=jnp.asarray(vertex_active),
            cells=jnp.asarray(cells),
            cell_ids=jnp.asarray(cell_ids),
            cell_active=jnp.asarray(cell_active),
            tuples=jnp.asarray(tuples),
            tags=jnp.asarray(tags),
            blocks=jnp.asarray(blocks),
            generations=jnp.asarray(generations),
            parents=jnp.asarray(parents),
            children=jnp.asarray(children),
            bisection_vertices=jnp.asarray(bisection_vertices),
            retired=jnp.asarray(retired),
            cell_classes=jnp.asarray(cell_classes),
            facet_classes=jnp.asarray(facet_classes),
            vertex_parents=jnp.asarray(vertex_parents),
            vertex_levels=jnp.asarray(vertex_levels),
            vertex_removal=jnp.asarray(vertex_removal),
            vertex_protected=jnp.asarray(vertex_protected),
            protected_codes=jnp.asarray(protected_codes),
            refine_rejected=jnp.asarray(refine_rejected),
            coarsen_marked=jnp.asarray(coarsen_marked),
            cursors=jnp.asarray(cursors),
        )
    )


def _owner_local_periodic_route_split_keys(
    local_keys: Sequence[tuple[int, ...]],
    edge_owners: dict[tuple[int, ...], tuple[int, ...]],
    dimension: int,
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[tuple[int, ...], ...]:
    """Route newly split quotient-edge keys only to current edge owners."""
    from ._distribution import (
        _owner_exchange_message_banks,
        _owner_group_fail_if_any,
    )

    key_width = 2 + 2 * dimension
    if dimension not in (2, 3) or part_count <= 0:
        raise ValueError("Periodic edge closure dimensions or owner count are invalid.")
    keys = tuple(sorted(set(tuple(int(value) for value in key) for key in local_keys)))
    local_error = None
    if any(len(key) != key_width or key[0] < 0 for key in keys):
        local_error = "Periodic split edge keys have an invalid exact extent."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic split-edge key input"
    )
    requests: list[list[tuple[int, ...]]] = [[] for _ in range(part_count)]
    local_requests: list[tuple[int, ...]] = []
    for key in keys:
        shard = (
            int(
                canonical_fingerprint(
                    {"kind": "owner-local-periodic-edge-shard", "key": key}
                )[:16],
                16,
            )
            % part_count
        )
        if shard == local_rank:
            local_requests.append(key)
        else:
            requests[shard].append(key)
    outgoing = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, key_width))
        if owner != local_rank
        else np.zeros((0, key_width), dtype=np.int64)
        for owner, rows in enumerate(requests)
    )
    incoming = _owner_exchange_message_banks(
        outgoing, local_rank, execution_group, axis_name
    )
    keys_by_owner: list[set[tuple[int, ...]]] = [set() for _ in range(part_count)]
    for key in local_requests:
        owners = edge_owners.get(key)
        if owners is None:
            continue
        for owner in owners:
            keys_by_owner[int(owner)].add(key)
    for sender, bank in incoming.items():
        del sender
        for row in np.asarray(bank, dtype=np.int64):
            key = tuple(int(value) for value in row.tolist())
            owners = edge_owners.get(key)
            if owners is None:
                continue
            for owner in owners:
                keys_by_owner[int(owner)].add(key)
    response_banks = tuple(
        np.asarray(sorted(rows), dtype=np.int64).reshape((-1, key_width))
        if owner != local_rank
        else np.zeros((0, key_width), dtype=np.int64)
        for owner, rows in enumerate(keys_by_owner)
    )
    replies = _owner_exchange_message_banks(
        response_banks, local_rank, execution_group, axis_name
    )
    result = set(keys_by_owner[local_rank])
    for bank in replies.values():
        result.update(
            tuple(int(value) for value in row.tolist())
            for row in np.asarray(bank, dtype=np.int64)
        )
    return tuple(sorted(result))


def _owner_local_periodic_route_vertex_roots(
    requested_roots: Sequence[int],
    available_roots: Sequence[int],
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[int, ...]:
    """Route quotient vertex-orbit requests to ranks holding active images."""
    from ._distribution import (
        _owner_exchange_message_banks,
        _owner_group_fail_if_any,
    )

    local_requested = tuple(sorted(set(int(value) for value in requested_roots)))
    local_available = tuple(sorted(set(int(value) for value in available_roots)))
    local_error = None
    if (
        part_count <= 0
        or not 0 <= local_rank < part_count
        or any(
            value < 0 or value > np.iinfo(np.int64).max
            for value in (*local_requested, *local_available)
        )
    ):
        local_error = "Periodic vertex-orbit root requests are malformed."
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic coarsen-root input"
    )
    outgoing_rows: list[list[tuple[int, int]]] = [[] for _ in range(part_count)]
    local_rows: list[tuple[int, int]] = []
    for kind, roots in ((0, local_available), (1, local_requested)):
        for root in roots:
            bucket = (root * part_count) // (1 << 63)
            if not 0 <= bucket < part_count:
                raise ValueError("Periodic vertex root lies outside its owner shard.")
            row = (kind, root)
            if bucket == local_rank:
                local_rows.append(row)
            else:
                outgoing_rows[bucket].append(row)
    outgoing = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 2))
        if owner != local_rank
        else np.zeros((0, 2), dtype=np.int64)
        for owner, rows in enumerate(outgoing_rows)
    )
    incoming = _owner_exchange_message_banks(
        outgoing, local_rank, execution_group, axis_name
    )
    owners_by_root: dict[int, set[int]] = {}
    request_roots: set[int] = set()
    banks = {local_rank: np.asarray(local_rows, dtype=np.int64).reshape((-1, 2))}
    banks.update(incoming)
    for sender, bank in banks.items():
        for kind, root in np.asarray(bank, dtype=np.int64):
            if int(kind) == 0:
                owners_by_root.setdefault(int(root), set()).add(int(sender))
            elif int(kind) == 1:
                request_roots.add(int(root))
            else:
                raise ValueError("Periodic vertex-orbit route has an unknown message.")
    roots_by_owner: list[set[int]] = [set() for _ in range(part_count)]
    for root in request_roots:
        for owner in owners_by_root.get(root, ()):
            roots_by_owner[owner].add(root)
    response_banks = tuple(
        np.asarray(sorted(roots), dtype=np.int64).reshape((-1, 1))
        if owner != local_rank
        else np.zeros((0, 1), dtype=np.int64)
        for owner, roots in enumerate(roots_by_owner)
    )
    replies = _owner_exchange_message_banks(
        response_banks, local_rank, execution_group, axis_name
    )
    result = set(roots_by_owner[local_rank])
    for bank in replies.values():
        result.update(int(row[0]) for row in np.asarray(bank, dtype=np.int64))
    return tuple(sorted(result))


def _owner_local_periodic_active_midpoint_roots(
    state: AdaptiveSimplexState,
    vertex_root_ids: dict[int, int],
    /,
) -> tuple[int, ...]:
    vertex_count, cell_count = map(int, np.asarray(state.cursors, dtype=np.int64)[:2])
    vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)[:vertex_count]
    vertex_active = np.asarray(state.mesh.vertex_active, dtype=np.bool_)[:vertex_count]
    vertex_parents = np.asarray(state.vertex_parents, dtype=np.int64)[:vertex_count]
    vertex_protected = np.asarray(state.vertex_protected, dtype=np.bool_)[:vertex_count]
    cell_active = np.asarray(state.mesh.cell_active, dtype=np.bool_)[:cell_count]
    cells = np.asarray(state.mesh.cells, dtype=np.int64)[:cell_count]
    used_slots = set(int(value) for value in cells[cell_active].reshape((-1,)).tolist())
    return tuple(
        sorted(
            vertex_root_ids[int(vertex_ids[row])]
            for row in range(vertex_count)
            if (
                row in used_slots
                and vertex_active[row]
                and not vertex_protected[row]
                and np.all(vertex_parents[row] >= 0)
            )
        )
    )


def _owner_local_periodic_mark_coarsen_roots(
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    roots: Sequence[int],
    vertex_root_ids: dict[int, int],
    /,
) -> tuple[Array, bool]:
    requested = set(int(value) for value in roots)
    cell_count = int(np.asarray(state.cursors, dtype=np.int64)[1])
    active = np.asarray(state.mesh.cell_active, dtype=np.bool_)[:cell_count]
    cells = np.asarray(state.mesh.cells, dtype=np.int64)[:cell_count]
    vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)
    marks = np.zeros((layout.cell_capacity,), dtype=np.bool_)
    for row in np.flatnonzero(active):
        marks[row] = any(
            vertex_root_ids[int(vertex_ids[int(slot)])] in requested
            for slot in cells[row]
            if int(vertex_ids[int(slot)]) in vertex_root_ids
        )
    return jnp.asarray(marks), bool(np.any(marks))


def _owner_local_periodic_coarsen_orbit(
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    root_id: int,
    vertex_root_ids: dict[int, int],
    next_vertex_id: int,
    next_cell_id: int,
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[AdaptiveSimplexState, bool, str | None]:
    from ._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    def collective_reindex(value: AdaptiveSimplexState) -> AdaptiveSimplexState:
        local_error = None
        try:
            result = _reindex_owner_local_periodic_state(
                layout, value, next_vertex_id, next_cell_id
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            result = None
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic coarsen reindex"
        )
        if result is None:
            raise ValueError("Periodic coarsen state reindex failed collectively.")
        return result

    trial_state = state
    pending_roots: tuple[int, ...] = (int(root_id),)
    changed_any = False
    while True:
        before_roots = _owner_local_periodic_active_midpoint_roots(
            trial_state, vertex_root_ids
        )
        local_requested = _owner_local_periodic_route_vertex_roots(
            pending_roots,
            before_roots,
            local_rank,
            part_count,
            execution_group,
            axis_name,
        )
        request_counts = np.asarray(
            _owner_group_allgather(
                np.asarray((len(local_requested),), dtype=np.int64),
                execution_group,
                axis_name,
            ),
            dtype=np.int64,
        ).reshape((-1,))
        if not np.any(request_counts):
            return trial_state, changed_any, None
        marks, marked = _owner_local_periodic_mark_coarsen_roots(
            layout, trial_state, local_requested, vertex_root_ids
        )
        local_error = (
            "A periodic coarsen owner has no active cell in its midpoint orbit."
            if local_requested and not marked
            else None
        )
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic orbit coarsen marking"
        )
        local_error = None
        try:
            update = coarsen_adaptive_simplex(layout, trial_state, marks)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            update = None
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic orbit coarsening"
        )
        if update is None:
            raise ValueError("Periodic orbit coarsening failed collectively.")
        flags = int(np.asarray(jax.device_get(update.report.status)))
        failure_flags = int(
            AdaptiveSimplexStatus.PROTECTED_CONFLICT
            | AdaptiveSimplexStatus.INVALID_GEOMETRY
        )
        local_invalid = bool(flags & failure_flags)
        host_state = jax.device_get(update.state)
        vertex_count, cell_count = map(
            int, np.asarray(host_state.cursors, dtype=np.int64)[:2]
        )
        active_cells = np.asarray(
            host_state.mesh.cell_active[:cell_count], dtype=np.bool_
        )
        if not local_invalid:
            try:
                require_committed_status(
                    int(np.asarray(host_state.status_flags)),
                    np.asarray(host_state.mesh.coordinates[:vertex_count]),
                    np.asarray(host_state.mesh.cells[:cell_count])[active_cells],
                    DeviceEpoch.BISECTION,
                )
            except Exception:
                local_invalid = True
        invalid_counts = np.asarray(
            _owner_group_allgather(
                np.asarray((int(local_invalid),), dtype=np.int64),
                execution_group,
                axis_name,
            ),
            dtype=np.int64,
        ).reshape((-1,))
        if np.any(invalid_counts):
            return (
                state,
                False,
                "A derived periodic midpoint orbit fails coarsening quality.",
            )
        trial_state = update.state
        active_after = _owner_local_periodic_active_midpoint_roots(
            trial_state, vertex_root_ids
        )
        before_counts: dict[int, int] = {}
        after_counts: dict[int, int] = {}
        for root in before_roots:
            before_counts[root] = before_counts.get(root, 0) + 1
        for root in active_after:
            after_counts[root] = after_counts.get(root, 0) + 1
        changed_roots = {
            root
            for root, count in before_counts.items()
            if after_counts.get(root, 0) < count
        }
        removed_count = len(before_roots) - len(active_after)
        removed_counts = np.asarray(
            _owner_group_allgather(
                np.asarray((removed_count,), dtype=np.int64),
                execution_group,
                axis_name,
            ),
            dtype=np.int64,
        ).reshape((-1,))
        if not np.any(removed_counts):
            return state, False, "No complete derived midpoint family was coarsenable."
        changed_any = True
        remaining_requested = set(local_requested).intersection(after_counts)
        pending_roots = tuple(sorted(changed_roots | remaining_requested))
        pending_counts = np.asarray(
            _owner_group_allgather(
                np.asarray((len(pending_roots),), dtype=np.int64),
                execution_group,
                axis_name,
            ),
            dtype=np.int64,
        ).reshape((-1,))
        if not np.any(pending_counts):
            return collective_reindex(trial_state), changed_any, None
        trial_state = collective_reindex(trial_state)


def _owner_local_periodic_mark_edges(
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    keys: Sequence[tuple[int, ...]],
    vertex_root_ids: dict[int, int],
    vertex_windings: dict[int, tuple[int, ...]],
    /,
) -> tuple[AdaptiveSimplexState, Array, bool]:
    dimension = layout.dimension
    requested = set(tuple(int(value) for value in key) for key in keys)
    width = 2 + 2 * dimension
    if any(len(key) != width for key in requested):
        raise ValueError("Periodic marked edge keys have the wrong dimension.")
    cell_count = int(np.asarray(state.cursors, dtype=np.int64)[1])
    work_cells = np.asarray(state.mesh.cells, dtype=np.int64)[:cell_count]
    cell_active = np.asarray(state.mesh.cell_active, dtype=np.bool_)[:cell_count]
    vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)
    tuples = np.asarray(state.tuples, dtype=np.int32).copy()
    tags = np.asarray(state.tags, dtype=np.int32).copy()
    marks = np.zeros((layout.cell_capacity,), dtype=np.bool_)
    for cell_row in np.flatnonzero(cell_active):
        cell = work_cells[cell_row]
        matching = []
        for first, second in combinations(range(dimension + 1), 2):
            first_id = int(vertex_ids[int(cell[first])])
            second_id = int(vertex_ids[int(cell[second])])
            key = _owner_local_periodic_edge_key(
                vertex_root_ids[first_id],
                vertex_windings[first_id],
                vertex_root_ids[second_id],
                vertex_windings[second_id],
            )
            if key in requested:
                matching.append(
                    (
                        (min(first_id, second_id), max(first_id, second_id)),
                        int(cell[first]),
                        int(cell[second]),
                    )
                )
        if not matching:
            continue
        _, first_slot, last_slot = min(matching)
        remaining = sorted(
            (int(value) for value in cell if int(value) not in (first_slot, last_slot)),
            key=lambda value: int(vertex_ids[value]),
        )
        desired = np.asarray((first_slot, *remaining, last_slot), dtype=np.int64)[None, :]
        tag = np.asarray((dimension,), dtype=np.int64)
        canonical, _ = _canonical(
            desired,
            tag,
            np.zeros((1,), dtype=np.bool_),
            dimension,
        )
        tuples[cell_row] = canonical[0].astype(np.int32)
        tags[cell_row] = dimension
        marks[cell_row] = True
    changed = bool(np.any(marks))
    if changed:
        state = eqx.tree_at(
            lambda value: (value.tuples, value.tags),
            state,
            (jnp.asarray(tuples), jnp.asarray(tags)),
        )
    return state, jnp.asarray(marks), changed


def _owner_local_periodic_current_mesh(
    state: AdaptiveSimplexState,
    source_mesh: CellMesh | None,
    vertex_root_ids: dict[int, int],
    vertex_windings: dict[int, tuple[int, ...]],
    vertex_owner_ranks: dict[int, int],
    exact_vertex_coordinates: dict[int, tuple[Fraction, ...]],
    local_rank: int,
    numeric_version: str,
    /,
) -> tuple[
    CellMesh | None,
    np.ndarray,
    np.ndarray,
    tuple[tuple[Fraction, ...], ...],
    np.ndarray,
    np.ndarray,
]:
    cell_count = int(np.asarray(state.cursors, dtype=np.int64)[1])
    vertex_count = int(np.asarray(state.cursors, dtype=np.int64)[0])
    active = np.asarray(state.mesh.cell_active, dtype=np.bool_)[:cell_count]
    active_rows = np.flatnonzero(active)
    if not active_rows.size:
        dimension = int(state.mesh.cells.shape[1] - 1)
        return (
            None,
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            (),
            np.zeros((0, dimension), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )
    if source_mesh is None:
        raise ValueError("A nonempty periodic state requires its source block table.")
    slot_cells = np.asarray(state.mesh.cells, dtype=np.int64)[:cell_count]
    state_cell_ids = np.asarray(state.mesh.cell_ids, dtype=np.int64)[:cell_count]
    state_blocks = np.asarray(state.blocks, dtype=np.int64)[:cell_count]
    state_vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)[:vertex_count]
    used_vertices = np.unique(slot_cells[active_rows].reshape((-1,)))
    used_vertices = used_vertices[
        np.argsort(state_vertex_ids[used_vertices], kind="stable")
    ]
    local_vertex_rows = np.full((vertex_count,), -1, dtype=np.int64)
    local_vertex_rows[used_vertices] = np.arange(used_vertices.size, dtype=np.int64)
    blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            local_vertex_rows[slot_cells[rows]].astype(np.int32),
            global_ids=state_cell_ids[rows],
        )
        for block_index, block in enumerate(source_mesh.blocks)
        if (rows := active_rows[state_blocks[active_rows] == block_index]).size
    )
    if not blocks:
        raise ValueError("Owner-local periodic state lost its source cell family.")
    target_vertex_ids = state_vertex_ids[used_vertices]
    target_coordinates = np.asarray(state.mesh.coordinates, dtype=np.float64)[
        used_vertices
    ]
    target = CellMesh(
        target_coordinates,
        blocks,
        vertex_global_ids=target_vertex_ids,
        numeric_version=numeric_version,
    )
    roots = np.asarray(
        [vertex_root_ids[int(identifier)] for identifier in target_vertex_ids],
        dtype=np.int64,
    )
    windings = np.asarray(
        [vertex_windings[int(identifier)] for identifier in target_vertex_ids],
        dtype=np.int64,
    ).reshape((-1, target.topological_dimension))
    exact = tuple(
        exact_vertex_coordinates[int(identifier)]
        for identifier in target_vertex_ids.tolist()
    )
    cell_owners = np.full(
        target.entity_set(target.topological_dimension).count,
        local_rank,
        dtype=np.int32,
    )
    target_vertex_owners = np.asarray(
        [vertex_owner_ranks[int(identifier)] for identifier in target_vertex_ids],
        dtype=np.int32,
    )
    return (
        target,
        cell_owners,
        roots,
        exact,
        windings,
        target_vertex_owners,
    )


def _owner_local_periodic_assign_new_ids(
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    old_vertex_ids: set[int],
    old_cell_ids: set[int],
    vertex_root_ids: dict[int, int],
    vertex_windings: dict[int, tuple[int, ...]],
    vertex_owner_ranks: dict[int, int],
    exact_vertex_coordinates: dict[int, tuple[Fraction, ...]],
    cell_windings: dict[int, tuple[int, ...]],
    next_vertex_id: int,
    next_cell_id: int,
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[
    AdaptiveSimplexState,
    dict[int, int],
    dict[int, tuple[int, ...]],
    dict[int, int],
    dict[int, tuple[Fraction, ...]],
    dict[int, tuple[int, ...]],
    int,
    int,
    tuple[tuple[int, ...], ...],
]:
    """Rekey exact edge images and inherit root ownership from canonical endpoints."""
    from ..discretization._adaptive_simplex import _state, _work
    from ._distribution import _owner_group_fail_if_any

    vertex_count, cell_count = map(int, np.asarray(state.cursors, dtype=np.int64)[:2])
    vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)
    vertex_parents = np.asarray(state.vertex_parents, dtype=np.int64)
    exact_midpoints: dict[int, tuple[Fraction, ...]] = {}
    vertex_keys: dict[int, tuple[int, ...]] = {}
    midpoint_owner_ranks: dict[int, int] = {}
    local_error = None
    dimension = layout.dimension
    try:
        for row in range(vertex_count):
            temporary_id = int(vertex_ids[row])
            if temporary_id in old_vertex_ids:
                continue
            parents = vertex_parents[row]
            if (
                parents.shape != (2,)
                or np.any(parents < 0)
                or np.any(parents >= vertex_count)
            ):
                raise ValueError("New periodic midpoint lacks two valid parents.")
            parent_ids = tuple(int(value) for value in vertex_ids[parents].tolist())
            first_id, second_id = parent_ids
            first_exact = exact_vertex_coordinates[first_id]
            second_exact = exact_vertex_coordinates[second_id]
            midpoint = tuple(
                (first_exact[axis] + second_exact[axis]) / 2 for axis in range(dimension)
            )
            projected = tuple(float(value) for value in midpoint)
            if not np.all(np.isfinite(np.asarray(projected, dtype=np.float64))):
                raise ValueError("Periodic midpoint is not representable in binary64.")
            if projected in (
                tuple(float(value) for value in first_exact),
                tuple(float(value) for value in second_exact),
            ):
                raise ValueError(
                    "Periodic midpoint is indistinguishable from a fixed endpoint."
                )
            first_winding = vertex_windings[first_id]
            second_winding = vertex_windings[second_id]
            edge_key = _owner_local_periodic_edge_key(
                vertex_root_ids[first_id],
                first_winding,
                vertex_root_ids[second_id],
                second_winding,
            )
            endpoints = sorted(
                (
                    (
                        vertex_root_ids[first_id],
                        first_winding,
                        first_id,
                    ),
                    (
                        vertex_root_ids[second_id],
                        second_winding,
                        second_id,
                    ),
                )
            )
            anchor_winding = tuple(int(value) for value in endpoints[0][1])
            midpoint_owner_ranks[temporary_id] = vertex_owner_ranks[int(endpoints[0][2])]
            vertex_keys[temporary_id] = (*edge_key, *anchor_winding)
            exact_midpoints[temporary_id] = midpoint
        if len(set(vertex_keys.values())) != len(vertex_keys):
            raise ValueError(
                "One owner-local split created a duplicate periodic vertex image."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic midpoint ancestry"
    )
    vertex_key_width = 2 + 3 * dimension
    edge_key_width = 2 + 2 * dimension
    vertex_id_by_key, next_vertex_id, vertex_root_by_key = (
        _owner_local_allocate_exact_ids(
            tuple(vertex_keys.values()),
            vertex_key_width,
            next_vertex_id,
            local_rank,
            part_count,
            execution_group,
            axis_name,
            group_key_width=edge_key_width,
        )
    )
    work = _work(state)
    coordinates = np.asarray(work.coordinates, dtype=np.float64).copy()
    state_vertex_ids = np.asarray(work.vertex_ids, dtype=np.int64).copy()
    for temporary_id in vertex_keys:
        vertex_root_ids.pop(temporary_id, None)
        vertex_windings.pop(temporary_id, None)
        exact_vertex_coordinates.pop(temporary_id, None)
        vertex_owner_ranks.pop(temporary_id, None)
    for row in range(vertex_count):
        temporary_id = int(state_vertex_ids[row])
        key = vertex_keys.get(temporary_id)
        if key is None:
            continue
        identifier = vertex_id_by_key[key]
        state_vertex_ids[row] = identifier
        vertex_root_ids[identifier] = vertex_root_by_key[key]
        vertex_windings[identifier] = tuple(int(value) for value in key[-dimension:])
        vertex_owner_ranks[identifier] = midpoint_owner_ranks[temporary_id]
        midpoint = exact_midpoints[temporary_id]
        exact_vertex_coordinates[identifier] = midpoint
        coordinates[row] = np.asarray(
            [float(value) for value in midpoint], dtype=np.float64
        )
    state = _state(
        work._replace(
            coordinates=jnp.asarray(coordinates),
            vertex_ids=jnp.asarray(state_vertex_ids),
        )
    )
    cell_count = int(np.asarray(state.cursors, dtype=np.int64)[1])
    state_cell_ids = np.asarray(state.mesh.cell_ids, dtype=np.int64)
    state_cells = np.asarray(state.mesh.cells, dtype=np.int64)
    cell_keys: dict[int, tuple[int, ...]] = {}
    cell_anchors: dict[int, tuple[int, ...]] = {}
    local_error = None
    try:
        for row in range(cell_count):
            temporary_id = int(state_cell_ids[row])
            if temporary_id in old_cell_ids:
                continue
            corners = state_cells[row]
            roots = tuple(
                vertex_root_ids[int(state_vertex_ids[int(slot)])] for slot in corners
            )
            windings = tuple(
                vertex_windings[int(state_vertex_ids[int(slot)])] for slot in corners
            )
            quotient_roots, relative, anchor = _owner_local_periodic_cell_key(
                roots, windings
            )
            relative_flat = tuple(
                value for row_values in relative for value in row_values
            )
            cell_keys[temporary_id] = (*quotient_roots, *relative_flat)
            cell_anchors[temporary_id] = anchor
        if len(set(cell_keys.values())) != len(cell_keys):
            raise ValueError(
                "One owner-local bisection created duplicate periodic cell orbits."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic child-cell identity"
    )
    cell_key_width = (dimension + 1) * (dimension + 1)
    cell_id_by_key, next_cell_id, _ = _owner_local_allocate_exact_ids(
        tuple(cell_keys.values()),
        cell_key_width,
        next_cell_id,
        local_rank,
        part_count,
        execution_group,
        axis_name,
    )
    work = _work(state)
    state_cell_ids = np.asarray(work.cell_ids, dtype=np.int64).copy()
    for temporary_id in cell_keys:
        cell_windings.pop(temporary_id, None)
    for row in range(cell_count):
        temporary_id = int(state_cell_ids[row])
        key = cell_keys.get(temporary_id)
        if key is None:
            continue
        identifier = cell_id_by_key[key]
        state_cell_ids[row] = identifier
        cell_windings[identifier] = cell_anchors[temporary_id]
    cursors = np.asarray(work.cursors, dtype=np.int64).copy()
    cursors[2] = next_vertex_id
    cursors[3] = next_cell_id
    state = _state(
        work._replace(
            vertex_ids=jnp.asarray(state_vertex_ids),
            cell_ids=jnp.asarray(state_cell_ids),
            cursors=jnp.asarray(cursors),
        )
    )
    if (
        np.unique(state_vertex_ids[:vertex_count]).size != vertex_count
        or np.unique(state_cell_ids[:cell_count]).size != cell_count
    ):
        local_error = "Periodic exact-ID allocation produced local duplicate IDs."
    else:
        local_error = None
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic target ID uniqueness"
    )
    split_keys = tuple(sorted(set(vertex_keys.values())))
    state = _reindex_owner_local_periodic_state(
        layout, state, next_vertex_id, next_cell_id
    )
    return (
        state,
        vertex_root_ids,
        vertex_windings,
        vertex_owner_ranks,
        exact_vertex_coordinates,
        cell_windings,
        next_vertex_id,
        next_cell_id,
        tuple(tuple(int(value) for value in key[:edge_key_width]) for key in split_keys),
    )


class OwnerLocalPeriodicCellMetadata(NamedTuple):
    cell_ids: np.ndarray
    cell_key_offsets: np.ndarray
    cell_key_source_ids: np.ndarray
    cell_key_relative_windings: np.ndarray
    cell_windings: np.ndarray


class OwnerLocalPeriodicAdaptiveExecution(NamedTuple):
    """Ephemeral metric evidence before provider target-view acceptance.

    Target vertex banks are row-aligned with `target_vertex_ids`; midpoint orbit
    owners inherit the canonical first endpoint root owner and require provider
    sharded-directory validation. `target_owned_*` identify this carrier;
    `target_local_*` remain unset until exact support/ghosts and face/CSR proof.
    A supplied graph plan is an explicit repartition request, not a receipt.
    """

    candidate_id: str
    specification_id: str
    source_mesh_id: str
    source_local_mesh_id: str | None
    source_local_topology_id: str | None
    source_local_geometry_id: str | None
    source_owned_mesh_id: str | None
    target_local_mesh_id: str | None
    target_local_topology_id: str | None
    target_local_geometry_id: str | None
    target_owned_mesh_id: str | None
    target_owned_topology_id: str | None
    status: MeshAdaptationStatus
    refusal: str | None
    part_transition: OwnerLocalAdaptivePartTransition | None
    target_vertex_ids: np.ndarray
    vertex_source_ids: np.ndarray
    vertex_absolute_windings: np.ndarray
    vertex_owner_ranks: np.ndarray
    exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...]
    cell_metadata: OwnerLocalPeriodicCellMetadata
    exact_cell_measures: tuple[Fraction, ...]
    size_evidence: OwnerLocalPeriodicSizeEvidence
    graph_partition_plan: GraphPartitionPlan | None
    live_forest: PeriodicSimplexForest | None
    result_id: str


class OwnerLocalPeriodicAdaptiveResult(NamedTuple):
    """Compact, rank-local replay evidence for an accepted periodic adaptation.

    The target `OwnerLocalCellGeometry` is persisted separately; this record
    binds its accepted local view and stores the exact source-to-target action,
    never the adaptive execution state or full mesh payload.
    """

    candidate_id: str
    specification_id: str
    executor_result_id: str
    status: MeshAdaptationStatus
    refusal: str | None
    source_distribution_id: str
    source_part_id: str
    source_id: str
    source_revision: str
    source_mesh_id: str
    source_topology_id: str
    source_geometry_id: str
    source_geometry_layout_id: str
    coordinate_contract_id: str
    source_periodic_topology_id: str | None
    source_local_mesh_id: str | None
    source_local_topology_id: str | None
    source_local_geometry_id: str | None
    source_owned_mesh_id: str | None
    target_local_mesh_id: str | None
    target_local_topology_id: str | None
    target_local_geometry_id: str | None
    target_owned_mesh_id: str | None
    target_owned_topology_id: str | None
    target_topology_id: str
    target_geometry_id: str
    target_source_map_id: str | None
    target_source_context_id: str | None
    target_distribution_id: str
    target_part_id: str
    target_owner_rank: int
    target_local_view_id: str
    target_owner_directory_id: str
    target_periodic_face_coverage_id: str | None
    target_periodic_topology_id: str | None
    graph_partition_plan: GraphPartitionPlan | None
    transition_id: str | None
    lineage: MeshLineage | None
    vertex_stencil: VertexInterpolationStencil | None
    vertex_transfer: FiniteElementTopologyTransfer | None
    source_cell_ids: np.ndarray
    target_cell_ids: np.ndarray
    target_cell_owner_ranks: np.ndarray
    target_cell_parent_ids: np.ndarray
    target_vertex_parent_ids: np.ndarray
    source_cell_measures: tuple[Fraction, ...]
    target_cell_measures: tuple[Fraction, ...]
    source_partition_weights: np.ndarray
    status_flags: int
    counters: np.ndarray
    target_vertex_ids: np.ndarray
    vertex_source_ids: np.ndarray
    vertex_absolute_windings: np.ndarray
    vertex_owner_ranks: np.ndarray
    exact_vertex_coordinates: tuple[tuple[Fraction, ...], ...]
    cell_metadata: OwnerLocalPeriodicCellMetadata
    size_evidence: OwnerLocalPeriodicSizeEvidence
    live_forest: PeriodicSimplexForest | None
    result_id: str

    @classmethod
    def from_execution(
        cls,
        execution: OwnerLocalPeriodicAdaptiveExecution,
        current_view: OwnerLocalCellGeometry,
        /,
    ) -> OwnerLocalPeriodicAdaptiveResult:
        from ..discretization._view_support import OwnerLocalCellGeometry

        if not isinstance(execution, OwnerLocalPeriodicAdaptiveExecution):
            raise TypeError("execution must be OwnerLocalPeriodicAdaptiveExecution.")
        if not isinstance(current_view, OwnerLocalCellGeometry):
            raise TypeError("current_view must be OwnerLocalCellGeometry.")
        current_view.reconstruct_local_geometry()
        return cls._from_execution_validated(execution, current_view)

    @classmethod
    def _from_execution_validated(
        cls,
        execution: OwnerLocalPeriodicAdaptiveExecution,
        current_view: OwnerLocalCellGeometry,
        /,
    ) -> OwnerLocalPeriodicAdaptiveResult:
        from ..discretization._view_support import OwnerLocalCellGeometry

        if not isinstance(execution, OwnerLocalPeriodicAdaptiveExecution):
            raise TypeError("execution must be OwnerLocalPeriodicAdaptiveExecution.")
        if not isinstance(current_view, OwnerLocalCellGeometry):
            raise TypeError("current_view must be OwnerLocalCellGeometry.")
        if current_view.source_mesh_id != execution.source_mesh_id:
            raise ValueError("The accepted local view changed its authored source scope.")
        if current_view.owner_rank < 0:
            raise ValueError("The accepted local view has an invalid owner rank.")
        if execution.status.converged and (
            current_view.periodic_face_coverage_id is None
            or current_view.periodic_topology_id is None
        ):
            raise ValueError(
                "A converged periodic execution requires accepted target face coverage."
            )
        part = execution.part_transition
        if part is None:
            empty = np.zeros((0,), dtype=np.int64)
            source_distribution_id = current_view.distribution_id
            source_part_id = current_view.part_id
            source_topology_id = current_view.source_topology_id
            source_geometry_id = current_view.source_geometry_id
            source_geometry_layout_id = current_view.geometry_layout_id
            coordinate_contract_id = current_view.coordinate_contract_id
            source_periodic_topology_id = current_view.periodic_topology_id
            transition_id = None
            lineage = None
            stencil = None
            transfer = None
            source_cell_ids = empty.copy()
            target_cell_ids = empty.copy()
            target_cell_owner_ranks = np.zeros((0,), dtype=np.int32)
            target_cell_parent_ids = empty.copy()
            target_vertex_parent_ids = np.zeros((0, 2), dtype=np.int64)
            source_cell_measures: tuple[Fraction, ...] = ()
            target_cell_measures: tuple[Fraction, ...] = ()
            source_partition_weights = np.zeros((0,), dtype=np.int64)
            status_flags = 0
            counters = np.zeros((0,), dtype=np.int64)
        else:
            if (
                part.part_index != current_view.owner_rank
                or part.source_mesh_id
                != (
                    execution.source_owned_mesh_id
                    if execution.source_owned_mesh_id is not None
                    else execution.source_local_mesh_id
                )
                or (
                    part.target_mesh is None
                    and execution.target_owned_mesh_id is not None
                )
                or (
                    part.target_mesh is not None
                    and part.target_mesh.mesh_id != execution.target_owned_mesh_id
                )
            ):
                raise ValueError("The local transition carrier identities are stale.")
            if not np.array_equal(
                np.asarray(execution.target_vertex_ids, dtype=np.int64),
                np.asarray(
                    ()
                    if part.target_mesh is None
                    else part.target_mesh.vertex_global_ids,
                    dtype=np.int64,
                ),
            ):
                raise ValueError(
                    "Target vertex IDs differ from the accepted part carrier."
                )
            if not np.array_equal(
                np.asarray(execution.cell_metadata.cell_ids, dtype=np.int64),
                np.asarray(part.target_cell_ids, dtype=np.int64),
            ):
                raise ValueError(
                    "Target cell keys differ from the accepted part carrier."
                )
            source_distribution_id = part.source_distribution_id
            source_part_id = part.source_part_id
            source_topology_id = part.source_topology_id
            source_geometry_id = part.source_geometry_id
            source_geometry_layout_id = part.source_geometry_layout_id
            coordinate_contract_id = part.coordinate_contract_id
            source_periodic_topology_id = part.source_periodic_topology_id
            transition_id = part.transition_id
            lineage = part.lineage
            stencil = part.vertex_stencil
            transfer = part.vertex_transfer
            source_cell_ids = np.asarray(part.source_cell_ids, dtype=np.int64).copy()
            target_cell_ids = np.asarray(part.target_cell_ids, dtype=np.int64).copy()
            target_cell_owner_ranks = np.asarray(
                part.target_cell_owner_ranks, dtype=np.int32
            ).copy()
            target_cell_parent_ids = np.asarray(
                part.target_cell_parent_ids, dtype=np.int64
            ).copy()
            target_vertex_parent_ids = np.asarray(
                part.target_vertex_parent_ids, dtype=np.int64
            ).copy()
            source_cell_measures = tuple(part.source_cell_measures)
            target_cell_measures = tuple(part.target_cell_measures)
            source_partition_weights = np.asarray(
                part.source_partition_weights, dtype=np.int64
            ).copy()
            status_flags = int(part.status_flags)
            counters = np.asarray(part.counters, dtype=np.int64).copy()
        if (
            source_partition_weights.shape != source_cell_ids.shape
            or source_partition_weights.dtype != np.dtype(np.int64)
            or np.any(source_partition_weights <= 0)
        ):
            raise ValueError(
                "Periodic source partition work must be positive integer units per cell."
            )
        target_vertex_ids = np.asarray(execution.target_vertex_ids, dtype=np.int64).copy()
        vertex_source_ids = np.asarray(execution.vertex_source_ids, dtype=np.int64).copy()
        vertex_absolute_windings = np.asarray(
            execution.vertex_absolute_windings, dtype=np.int64
        ).copy()
        vertex_owner_ranks = np.asarray(
            execution.vertex_owner_ranks, dtype=np.int32
        ).copy()
        exact_vertex_coordinates = tuple(
            tuple(row) for row in execution.exact_vertex_coordinates
        )
        if (
            vertex_source_ids.shape != target_vertex_ids.shape
            or vertex_owner_ranks.shape != target_vertex_ids.shape
            or vertex_absolute_windings.ndim != 2
            or vertex_absolute_windings.shape[0] != target_vertex_ids.size
            or len(exact_vertex_coordinates) != target_vertex_ids.size
        ):
            raise ValueError("Periodic target vertex identity banks do not align.")
        if execution.exact_cell_measures != tuple(target_cell_measures):
            raise ValueError(
                "Periodic exact cell measures differ from the part transition."
            )
        target_local_view_id = current_view.local_view_id
        target_owner_directory_id = current_view.distribution_id
        target_topology_id = current_view.source_topology_id
        target_geometry_id = current_view.source_geometry_id
        target_source_map_id = current_view.source_map_id
        target_source_context_id = current_view.source_context_id
        plan_id = (
            None
            if execution.graph_partition_plan is None
            else execution.graph_partition_plan.plan_id
        )
        result_id = canonical_fingerprint(
            {
                "kind": "owner-local-periodic-adaptive-result",
                "executor": execution.result_id,
                "candidate": execution.candidate_id,
                "specification": execution.specification_id,
                "source": (
                    execution.source_mesh_id,
                    source_distribution_id,
                    source_part_id,
                    current_view.source_id,
                    current_view.source_revision,
                    source_topology_id,
                    source_geometry_id,
                    source_geometry_layout_id,
                    coordinate_contract_id,
                    source_periodic_topology_id,
                    execution.source_local_mesh_id,
                    execution.source_local_topology_id,
                    execution.source_local_geometry_id,
                    execution.source_owned_mesh_id,
                ),
                "target_view": (
                    target_local_view_id,
                    current_view.local_mesh_id,
                    current_view.local_topology_id,
                    current_view.local_geometry_id,
                    current_view.distribution_id,
                    current_view.part_id,
                    current_view.owner_rank,
                    target_owner_directory_id,
                    current_view.periodic_face_coverage_id,
                    current_view.periodic_topology_id,
                    target_topology_id,
                    target_geometry_id,
                    target_source_map_id,
                    target_source_context_id,
                ),
                "transition": transition_id,
                "lineage": None if lineage is None else lineage.lineage_id,
                "stencil": None if stencil is None else stencil.stencil_id,
                "vertex_transfer": (None if transfer is None else transfer.transfer_id),
                "source_cell_ids": source_cell_ids,
                "target_cell_ids": target_cell_ids,
                "target_cell_parent_ids": target_cell_parent_ids,
                "target_cell_owner_ranks": target_cell_owner_ranks,
                "target_vertex_parent_ids": target_vertex_parent_ids,
                "source_cell_measures": tuple(
                    (value.numerator, value.denominator) for value in source_cell_measures
                ),
                "target_cell_measures": tuple(
                    (value.numerator, value.denominator) for value in target_cell_measures
                ),
                "source_partition_weights": source_partition_weights,
                "status_flags": status_flags,
                "counters": counters,
                "target_vertices": (
                    target_vertex_ids,
                    vertex_source_ids,
                    vertex_absolute_windings,
                    vertex_owner_ranks,
                    tuple(
                        tuple((value.numerator, value.denominator) for value in row)
                        for row in exact_vertex_coordinates
                    ),
                ),
                "target_cells": (
                    execution.cell_metadata.cell_ids,
                    execution.cell_metadata.cell_key_offsets,
                    execution.cell_metadata.cell_key_source_ids,
                    execution.cell_metadata.cell_key_relative_windings,
                    execution.cell_metadata.cell_windings,
                ),
                "size_evidence": execution.size_evidence.evidence_id,
                "graph_partition_plan": plan_id,
                "live_forest": None
                if execution.live_forest is None
                else execution.live_forest.forest_id,
                "status": execution.status.value,
                "refusal": execution.refusal,
            }
        )
        result = cls(
            execution.candidate_id,
            execution.specification_id,
            execution.result_id,
            execution.status,
            execution.refusal,
            source_distribution_id,
            source_part_id,
            current_view.source_id,
            current_view.source_revision,
            execution.source_mesh_id,
            source_topology_id,
            source_geometry_id,
            source_geometry_layout_id,
            coordinate_contract_id,
            source_periodic_topology_id,
            execution.source_local_mesh_id,
            execution.source_local_topology_id,
            execution.source_local_geometry_id,
            execution.source_owned_mesh_id,
            current_view.local_mesh_id,
            current_view.local_topology_id,
            current_view.local_geometry_id,
            execution.target_owned_mesh_id,
            execution.target_owned_topology_id,
            target_topology_id,
            target_geometry_id,
            target_source_map_id,
            target_source_context_id,
            current_view.distribution_id,
            current_view.part_id,
            int(current_view.owner_rank),
            target_local_view_id,
            target_owner_directory_id,
            current_view.periodic_face_coverage_id,
            current_view.periodic_topology_id,
            execution.graph_partition_plan,
            transition_id,
            lineage,
            stencil,
            transfer,
            source_cell_ids,
            target_cell_ids,
            target_cell_owner_ranks,
            target_cell_parent_ids,
            target_vertex_parent_ids,
            source_cell_measures,
            target_cell_measures,
            source_partition_weights,
            status_flags,
            counters,
            target_vertex_ids,
            vertex_source_ids,
            vertex_absolute_windings,
            vertex_owner_ranks,
            exact_vertex_coordinates,
            execution.cell_metadata,
            execution.size_evidence,
            execution.live_forest,
            result_id,
        )
        if result._computed_result_id() != result.result_id:
            raise ValueError("Periodic adaptive result identity omits stored evidence.")
        return result

    def _computed_result_id(self, /) -> str:
        plan_id = (
            None
            if self.graph_partition_plan is None
            else self.graph_partition_plan.plan_id
        )
        return canonical_fingerprint(
            {
                "kind": "owner-local-periodic-adaptive-result",
                "executor": self.executor_result_id,
                "candidate": self.candidate_id,
                "specification": self.specification_id,
                "source": (
                    self.source_mesh_id,
                    self.source_distribution_id,
                    self.source_part_id,
                    self.source_id,
                    self.source_revision,
                    self.source_topology_id,
                    self.source_geometry_id,
                    self.source_geometry_layout_id,
                    self.coordinate_contract_id,
                    self.source_periodic_topology_id,
                    self.source_local_mesh_id,
                    self.source_local_topology_id,
                    self.source_local_geometry_id,
                    self.source_owned_mesh_id,
                ),
                "target_view": (
                    self.target_local_view_id,
                    self.target_local_mesh_id,
                    self.target_local_topology_id,
                    self.target_local_geometry_id,
                    self.target_distribution_id,
                    self.target_part_id,
                    self.target_owner_rank,
                    self.target_owner_directory_id,
                    self.target_periodic_face_coverage_id,
                    self.target_periodic_topology_id,
                    self.target_topology_id,
                    self.target_geometry_id,
                    self.target_source_map_id,
                    self.target_source_context_id,
                ),
                "transition": self.transition_id,
                "lineage": None if self.lineage is None else self.lineage.lineage_id,
                "stencil": (
                    None
                    if self.vertex_stencil is None
                    else self.vertex_stencil.stencil_id
                ),
                "vertex_transfer": (
                    None
                    if self.vertex_transfer is None
                    else self.vertex_transfer.transfer_id
                ),
                "source_cell_ids": self.source_cell_ids,
                "target_cell_ids": self.target_cell_ids,
                "target_cell_parent_ids": self.target_cell_parent_ids,
                "target_cell_owner_ranks": self.target_cell_owner_ranks,
                "target_vertex_parent_ids": self.target_vertex_parent_ids,
                "source_cell_measures": tuple(
                    (value.numerator, value.denominator)
                    for value in self.source_cell_measures
                ),
                "target_cell_measures": tuple(
                    (value.numerator, value.denominator)
                    for value in self.target_cell_measures
                ),
                "source_partition_weights": self.source_partition_weights,
                "status_flags": self.status_flags,
                "counters": self.counters,
                "target_vertices": (
                    self.target_vertex_ids,
                    self.vertex_source_ids,
                    self.vertex_absolute_windings,
                    self.vertex_owner_ranks,
                    tuple(
                        tuple((value.numerator, value.denominator) for value in row)
                        for row in self.exact_vertex_coordinates
                    ),
                ),
                "target_cells": (
                    self.cell_metadata.cell_ids,
                    self.cell_metadata.cell_key_offsets,
                    self.cell_metadata.cell_key_source_ids,
                    self.cell_metadata.cell_key_relative_windings,
                    self.cell_metadata.cell_windings,
                ),
                "size_evidence": self.size_evidence.evidence_id,
                "graph_partition_plan": plan_id,
                "live_forest": None
                if self.live_forest is None
                else self.live_forest.forest_id,
                "status": self.status.value,
                "refusal": self.refusal,
            }
        )

    def rebind_after_ownership_migration(
        self,
        current_view: OwnerLocalCellGeometry,
        graph_partition_plan: GraphPartitionPlan,
        execution_group: ExecutionGroup,
        /,
        *,
        axis_name: str = "periodic_adaptive_rebind",
    ) -> OwnerLocalPeriodicAdaptiveResult:
        """Rebind after an owner-group-wide, geometry-preserving migration.

        The supplied plan is the one whose partition labels drove the migration
        (the caller binds those labels to it), so it replaces the stored plan as
        the current placement request. The owner group itself cannot change:
        both plans must address the current owner directory's part count. An
        adaptation placed without an explicit plan records the supplied plan.

        Only the target view identities are rebound. The rank-local transition
        rows (lineage, vertex stencil, cell ancestry) stay the adaptation's
        execution record on its original placement; a field transfer through
        the adaptation runs on that placement and is then repartitioned.
        """
        from .._execution_runtime import ExecutionGroup
        from ..discretization._view_support import OwnerLocalCellGeometry
        from ..graph._partition import GraphPartitionPlan

        if not isinstance(current_view, OwnerLocalCellGeometry):
            raise TypeError("current_view must be OwnerLocalCellGeometry.")
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        if not isinstance(graph_partition_plan, GraphPartitionPlan):
            raise TypeError("graph_partition_plan must be GraphPartitionPlan.")

        def rebuilt_plan(plan: GraphPartitionPlan, /) -> GraphPartitionPlan:
            return GraphPartitionPlan(
                len(plan.target_shares),
                target_shares=np.asarray(plan.target_shares, dtype=np.float64),
                maximum_imbalance=plan.maximum_imbalance,
                require_nonempty=plan.require_nonempty,
                refinement_passes=plan.refinement_passes,
            )

        supplied_plan = rebuilt_plan(graph_partition_plan)
        stored = self.graph_partition_plan
        stored_plan = None if stored is None else rebuilt_plan(stored)
        if (
            (
                stored is not None
                and stored_plan is not None
                and (
                    not _owner_local_periodic_values_equal(stored, stored_plan)
                    or stored_plan.part_count != supplied_plan.part_count
                )
            )
            or not _owner_local_periodic_values_equal(graph_partition_plan, supplied_plan)
            or supplied_plan.part_count != current_view.owner_directory_data.part_count
            or current_view.owner_directory_data.rank != current_view.owner_rank
        ):
            raise ValueError("Ownership migration plan or owner group changed.")
        current_view.require_owner_directory_current(
            execution_group,
            axis_name=axis_name,
        )
        current_view.reconstruct_local_geometry()
        if (
            current_view.source_id != self.source_id
            or current_view.source_revision != self.source_revision
            or current_view.source_mesh_id != self.source_mesh_id
            or current_view.source_topology_id != self.target_topology_id
            or current_view.source_geometry_id != self.target_geometry_id
            or current_view.coordinate_contract_id != self.coordinate_contract_id
            or current_view.specification_id != self.specification_id
            or current_view.periodic_topology_id != self.target_periodic_topology_id
            or current_view.periodic_face_coverage_id
            != self.target_periodic_face_coverage_id
        ):
            raise ValueError(
                "Ownership migration changed target source geometry or periodic proof."
            )
        from ._distribution_migration import migrate_periodic_simplex_forest

        live_forest = self.live_forest
        if live_forest is not None:
            live_forest = migrate_periodic_simplex_forest(
                live_forest,
                current_view.owned_cell_ids,
                execution_group,
                axis_name=f"{axis_name}_live_families",
            )
        rebound = self._replace(
            live_forest=live_forest,
            target_local_mesh_id=current_view.local_mesh_id,
            target_local_topology_id=current_view.local_topology_id,
            target_local_geometry_id=current_view.local_geometry_id,
            target_distribution_id=current_view.distribution_id,
            target_part_id=current_view.part_id,
            target_owner_rank=int(current_view.owner_rank),
            target_local_view_id=current_view.local_view_id,
            target_owner_directory_id=current_view.distribution_id,
            target_periodic_face_coverage_id=current_view.periodic_face_coverage_id,
            target_source_map_id=current_view.source_map_id,
            target_source_context_id=current_view.source_context_id,
            graph_partition_plan=supplied_plan,
            result_id="",
        )
        return rebound._replace(result_id=rebound._computed_result_id())

    def revalidate(
        self,
        candidate: OwnerLocalPeriodicCandidate,
        specification: SurfaceMeshingSpec | VolumeMeshingSpec,
        execution_group: ExecutionGroup,
        local_rank: int,
        current_view: OwnerLocalCellGeometry,
        graph_partition_plan: GraphPartitionPlan | None,
        /,
    ) -> OwnerLocalPeriodicAdaptiveResult:
        from .._execution_runtime import ExecutionGroup
        from ..discretization._view_support import OwnerLocalCellGeometry
        from ..graph._partition import GraphPartitionPlan
        from ._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
        from ._periodic import OwnerLocalPeriodicCandidate

        if not isinstance(current_view, OwnerLocalCellGeometry):
            raise TypeError("current_view must be OwnerLocalCellGeometry.")
        if not isinstance(candidate, OwnerLocalPeriodicCandidate):
            raise TypeError("candidate must be OwnerLocalPeriodicCandidate.")
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError(
                "specification must be SurfaceMeshingSpec or VolumeMeshingSpec."
            )
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be ExecutionGroup.")
        if specification.specification_id != self.specification_id:
            raise ValueError(
                "The current periodic specification differs from the recipe."
            )
        if (
            candidate.source_id != self.source_id
            or candidate.source_revision != self.source_revision
            or candidate.source_topology_id != self.target_topology_id
            or candidate.source_geometry_id != self.target_geometry_id
            or candidate.source_distribution_id != self.target_distribution_id
            or candidate.source_part_id != self.target_part_id
            or candidate.coordinate_contract_id != self.coordinate_contract_id
            or candidate.periodic_topology_id != self.target_periodic_topology_id
        ):
            raise ValueError(
                "Current periodic source, geometry, or ownership differs from accepted state."
            )
        if self.graph_partition_plan is None:
            if graph_partition_plan is not None:
                raise ValueError(
                    "A stored no-plan result cannot be replayed with a plan."
                )
        else:
            if not isinstance(self.graph_partition_plan, GraphPartitionPlan):
                raise TypeError("Stored graph partition plan has the wrong type.")
            if graph_partition_plan is None or not isinstance(
                graph_partition_plan, GraphPartitionPlan
            ):
                raise ValueError("The current graph partition plan is absent or untyped.")
            stored_plan = GraphPartitionPlan(
                len(self.graph_partition_plan.target_shares),
                target_shares=np.asarray(
                    self.graph_partition_plan.target_shares, dtype=np.float64
                ),
                maximum_imbalance=self.graph_partition_plan.maximum_imbalance,
                require_nonempty=self.graph_partition_plan.require_nonempty,
                refinement_passes=self.graph_partition_plan.refinement_passes,
            )
            current_plan = GraphPartitionPlan(
                len(graph_partition_plan.target_shares),
                target_shares=np.asarray(
                    graph_partition_plan.target_shares, dtype=np.float64
                ),
                maximum_imbalance=graph_partition_plan.maximum_imbalance,
                require_nonempty=graph_partition_plan.require_nonempty,
                refinement_passes=graph_partition_plan.refinement_passes,
            )
            if (
                not _owner_local_periodic_values_equal(
                    self.graph_partition_plan, stored_plan
                )
                or not _owner_local_periodic_values_equal(
                    graph_partition_plan, current_plan
                )
                or not _owner_local_periodic_values_equal(stored_plan, current_plan)
            ):
                raise ValueError(
                    "The current graph partition plan differs from the recipe."
                )
        if current_view.owner_rank != local_rank:
            raise ValueError("The renewed periodic view belongs to another rank.")
        current_view.require_owner_directory_current(
            execution_group,
            axis_name="periodic_adaptive_replay_directory",
        )
        current_view.reconstruct_local_geometry()
        if (
            current_view.source_id != self.source_id
            or current_view.source_revision != self.source_revision
            or current_view.local_view_id != self.target_local_view_id
            or current_view.local_mesh_id != self.target_local_mesh_id
            or current_view.local_topology_id != self.target_local_topology_id
            or current_view.local_geometry_id != self.target_local_geometry_id
            or current_view.source_mesh_id != self.source_mesh_id
            or current_view.distribution_id != self.target_distribution_id
            or current_view.part_id != self.target_part_id
            or current_view.distribution_id != self.target_owner_directory_id
            or current_view.periodic_face_coverage_id
            != self.target_periodic_face_coverage_id
            or current_view.periodic_topology_id != self.target_periodic_topology_id
            or current_view.source_topology_id != self.target_topology_id
            or current_view.source_geometry_id != self.target_geometry_id
            or current_view.source_map_id != self.target_source_map_id
            or current_view.source_context_id != self.target_source_context_id
        ):
            raise ValueError(
                "The restored periodic target view is stale for this result."
            )
        from ._distribution import _owner_group_fail_if_any
        from ._distribution_migration import migrate_periodic_simplex_forest

        forest = self.live_forest
        local_error = None
        rebuilt = None
        try:
            if forest is None:
                raise ValueError(
                    "Accepted periodic adaptation lacks live scientific families."
                )
            rebuilt = _restored_periodic_live_forest(forest)
            _validate_owner_local_periodic_forest(rebuilt, candidate)
            if (
                rebuilt.forest_id != forest.forest_id
                or self._computed_result_id() != self.result_id
            ):
                raise ValueError("Accepted periodic family or result identity changed.")
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            "periodic_live_restore",
            "live periodic family mathematics",
        )
        if rebuilt is None:
            raise ValueError("Live periodic family reconstruction failed collectively.")
        canonical = migrate_periodic_simplex_forest(
            rebuilt,
            current_view.owned_cell_ids,
            execution_group,
            axis_name="periodic_live_restore_replicas",
            validate_replicas=True,
        )
        _owner_group_fail_if_any(
            None
            if canonical.forest_id == rebuilt.forest_id
            else "Split-owner family replicas disagree.",
            execution_group,
            "periodic_live_restore_agreement",
            "live periodic family replicas",
        )
        epoch = _owner_local_periodic_size_epoch(
            candidate,
            specification,
            execution_group,
            local_rank,
            axis_name="periodic_current_size",
        )
        if not epoch.evidence.compliant:
            raise ValueError(
                "Restored periodic geometry fails its current exact size constraints."
            )
        return self


def _restored_periodic_live_forest(
    forest: PeriodicSimplexForest, /
) -> PeriodicSimplexForest:
    """Replay structural and exact family mathematics, not only stored IDs."""
    stored = forest.records
    records = SimplexForestRecords(
        stored.cell_ids,
        stored.cell_owner_ranks,
        dict(stored.cell_fields),
        stored.vertex_ids,
        stored.vertex_owner_ranks,
        dict(stored.vertex_fields),
        stored.protected_edge_ids,
        stored.dimension,
        stored.ambient_dimension,
        stored.next_vertex_id,
        stored.next_cell_id,
    )
    return PeriodicSimplexForest(
        records,
        forest.vertex_source_ids,
        forest.vertex_absolute_windings,
        forest.exact_vertex_coordinates,
        forest.block_names,
        forest.source_id,
        forest.source_revision,
        forest.periodic_cell_id,
    )


def _validate_owner_local_periodic_forest(
    forest: PeriodicSimplexForest,
    candidate: OwnerLocalPeriodicCandidate | OwnerLocalCellGeometry,
    /,
) -> None:
    """Check owned leaves directly against the independently certified current mesh."""
    from ..discretization._view_support import OwnerLocalCellGeometry
    from ._periodic import PeriodicPointOrbits
    from .providers._native_periodic import NativePeriodicSource

    if isinstance(candidate, OwnerLocalCellGeometry):
        rank = candidate.owner_rank
        source = candidate.source_value
        if not isinstance(source, NativePeriodicSource) or not isinstance(
            source.domain, PeriodicPointOrbits
        ):
            raise TypeError("Owner-local periodic families require their point source.")
        periodic_cell_ids = (source.domain.cell.cell_id,)
        mesh, _ = candidate.reconstruct_local_geometry()
    else:
        rank = candidate.rank
        periodic_cell_ids = (candidate.periodic_cell_id,)
        mesh = candidate.local_mesh
    records = forest.records
    if (
        forest.source_id != candidate.source_id
        or forest.source_revision != candidate.source_revision
        or forest.periodic_cell_id not in periodic_cell_ids
    ):
        raise ValueError("Periodic live families belong to another source or lattice.")
    fields = records.cell_field_map()
    owned = np.asarray(fields["active"], dtype=np.bool_) & (
        np.asarray(records.cell_owner_ranks) == rank
    )
    identifiers = np.asarray(records.cell_ids)[owned]
    if not np.array_equal(identifiers, np.asarray(candidate.owned_cell_ids)):
        raise ValueError("Periodic live family leaves do not cover accepted ownership.")
    if mesh is None:
        if identifiers.size:
            raise ValueError("Nonempty periodic family has no accepted local geometry.")
        return
    if candidate.exact_vertex_coordinates is None:
        raise ValueError("Periodic family binding requires exact current coordinates.")
    candidate_vertex_ids = np.asarray(candidate.vertex_ids, dtype=np.int64)
    candidate_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(candidate_vertex_ids)
    }
    forest_vertex_ids = np.asarray(records.vertex_ids, dtype=np.int64)
    forest_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(forest_vertex_ids)
    }
    mesh_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    mesh_cells = {
        int(identifier): (block.name, tuple(map(int, mesh_vertex_ids[row])))
        for block in mesh.blocks
        for identifier, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    for row in np.flatnonzero(owned):
        vertices = tuple(map(int, np.asarray(fields["cells"])[row]))
        expected = (forest.block_names[int(np.asarray(fields["blocks"])[row])], vertices)
        if mesh_cells.get(int(np.asarray(records.cell_ids)[row])) != expected:
            raise ValueError(
                "Periodic live cell geometry differs from its accepted leaf."
            )
        for identifier in vertices:
            old = forest_vertex_rows[identifier]
            current = candidate_vertex_rows[identifier]
            if (
                forest.exact_vertex_coordinates[old]
                != tuple(candidate.exact_vertex_coordinates[current])
                or int(forest.vertex_source_ids[old])
                != int(candidate.vertex_source_ids[current])
                or not np.array_equal(
                    np.asarray(forest.vertex_absolute_windings)[old],
                    np.asarray(candidate.vertex_absolute_windings)[current],
                )
            ):
                raise ValueError("Periodic live source coordinates or winding changed.")


def _periodic_transfer_mesh(
    view: OwnerLocalCellGeometry,
    identifiers: np.ndarray,
    /,
) -> tuple[CellMesh | None, dict[int, tuple[Fraction, ...]]]:
    """Bind an owned transition domain to current exact, lifted geometry."""
    if not np.array_equal(
        np.sort(identifiers),
        np.sort(np.asarray(view.owned_cell_ids, dtype=np.int64)),
    ):
        raise ValueError("Periodic transfer cells differ from current ownership.")
    local_mesh, _ = view.reconstruct_local_geometry()
    if local_mesh is None:
        if identifiers.size:
            raise ValueError("Nonempty periodic transfer lacks current geometry.")
        return None, {}
    mesh = _owner_local_mesh_subset(
        local_mesh, identifiers, numeric_version="periodic-transfer-validation"
    )
    if mesh is None:
        return None, {}
    if not np.array_equal(
        identifiers,
        np.asarray(
            mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
        ),
    ):
        raise ValueError("Periodic transfer cell rows are not in canonical order.")
    vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
    bank = view.exact_vertex_coordinates
    if (
        not isinstance(bank, tuple)
        or len(bank) != vertex_ids.size
        or any(
            not isinstance(row, tuple)
            or len(row) != mesh.ambient_dimension
            or any(not isinstance(value, Fraction) for value in row)
            for row in bank
        )
    ):
        raise ValueError("Periodic transfer requires current exact rational vertices.")
    return mesh, {
        int(identifier): tuple(row)
        for identifier, row in zip(vertex_ids, bank, strict=True)
    }


def _periodic_transfer_frontiers(
    source: CellMesh,
    target: CellMesh,
    source_exact: dict[int, tuple[Fraction, ...]],
    target_exact: dict[int, tuple[Fraction, ...]],
    /,
) -> tuple[
    dict[int, tuple[str, tuple[int, ...]]],
    dict[int, tuple[Fraction, ...]],
]:
    """Bind shared scientific IDs to both actual frontier geometries."""
    cells: dict[int, tuple[str, tuple[int, ...]]] = {}
    points: dict[int, tuple[Fraction, ...]] = {}
    for mesh, exact in ((source, source_exact), (target, target_exact)):
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        for identifier in map(int, vertex_ids):
            point = exact[identifier]
            if identifier in points and points[identifier] != point:
                raise ValueError("Periodic transfer moved a retained exact vertex.")
            points[identifier] = point
        for block in mesh.blocks:
            for identifier, row in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            ):
                identifier = int(identifier)
                geometry = (block.name, tuple(map(int, vertex_ids[row])))
                if identifier in cells and cells[identifier] != geometry:
                    raise ValueError(
                        "Periodic transfer changed a retained cell's closure."
                    )
                cells[identifier] = geometry
    return cells, points


def _periodic_transfer_parent_banks(
    source: OwnerLocalCellGeometry,
    target: OwnerLocalCellGeometry,
    current_cells: dict[int, tuple[str, tuple[int, ...]]],
    current_coordinates: dict[int, tuple[Fraction, ...]],
    /,
) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    """Use both current frontiers; coarsening need not retain retired children."""
    cells: dict[int, int] = {}
    vertices: dict[int, tuple[int, int]] = {}
    coordinates: dict[int, tuple[Fraction, ...]] = {}
    for view in (source, target):
        previous = view.adaptive_result
        if previous is None:
            continue
        if not isinstance(previous, OwnerLocalPeriodicAdaptiveResult):
            raise TypeError("Periodic transfer has an unsupported family authority.")
        if previous.live_forest is None:
            raise ValueError("Periodic transfer lacks its current live families.")
        forest = _restored_periodic_live_forest(previous.live_forest)
        _validate_owner_local_periodic_forest(forest, view)
        if forest.forest_id != previous.live_forest.forest_id:
            raise ValueError("Periodic transfer family identity does not reproduce.")
        records = forest.records
        fields = records.cell_field_map()
        forest_cells = np.asarray(fields["cells"])
        forest_blocks = np.asarray(fields["blocks"])
        for row, (identifier, parent) in enumerate(
            zip(
                np.asarray(records.cell_ids),
                np.asarray(fields["parent_ids"]),
                strict=True,
            )
        ):
            identifier, parent = int(identifier), int(parent)
            if identifier in current_cells and current_cells[identifier] != (
                forest.block_names[int(forest_blocks[row])],
                tuple(map(int, forest_cells[row])),
            ):
                raise ValueError(
                    "Periodic family ancestor differs from current geometry."
                )
            if identifier in cells and cells[identifier] != parent:
                raise ValueError("Periodic transfer changed a shared cell's ancestry.")
            cells[identifier] = parent
        vertex_fields = records.vertex_field_map()
        for identifier, parents, point in zip(
            np.asarray(records.vertex_ids),
            np.asarray(vertex_fields["parents"]),
            forest.exact_vertex_coordinates,
            strict=True,
        ):
            identifier = int(identifier)
            if (
                identifier in current_coordinates
                and current_coordinates[identifier] != point
            ):
                raise ValueError(
                    "Periodic family ancestor has different exact coordinates."
                )
            pair = (int(parents[0]), int(parents[1]))
            if identifier in vertices and (
                vertices[identifier] != pair or coordinates[identifier] != point
            ):
                raise ValueError("Periodic transfer changed a shared vertex's ancestry.")
            vertices[identifier] = pair
            coordinates[identifier] = point
    return cells, vertices


def _require_owner_local_periodic_transfer(
    result: OwnerLocalPeriodicAdaptiveResult,
    source: OwnerLocalCellGeometry,
    target: OwnerLocalCellGeometry,
    /,
) -> None:
    """Replay historical transfer mathematics against both current input views."""
    if (
        result.source_topology_id != source.source_topology_id
        or result.source_geometry_id != source.source_geometry_id
        or result.target_topology_id != target.source_topology_id
        or result.target_geometry_id != target.source_geometry_id
        or not _owner_local_periodic_values_equal(result, target.adaptive_result)
    ):
        raise ValueError("Periodic transfer belongs to different scientific geometry.")
    source_mesh, source_exact = _periodic_transfer_mesh(
        source, np.asarray(result.source_cell_ids, dtype=np.int64)
    )
    target_mesh, target_exact = _periodic_transfer_mesh(
        target, np.asarray(result.target_cell_ids, dtype=np.int64)
    )
    if source_mesh is None or target_mesh is None:
        if (
            source_mesh is not None
            or target_mesh is not None
            or result.lineage is not None
            or result.vertex_stencil is not None
            or result.vertex_transfer is not None
            or result.source_cell_measures
            or result.target_cell_measures
        ):
            raise ValueError(
                "Empty periodic transfer carries nonempty scientific claims."
            )
        return
    dimension = source_mesh.topological_dimension
    if (
        _owner_local_periodic_exact_cell_measures(source_mesh, source_exact, dimension)
        != result.source_cell_measures
        or _owner_local_periodic_exact_cell_measures(target_mesh, target_exact, dimension)
        != result.target_cell_measures
    ):
        raise ValueError("Periodic transfer measures differ from exact current geometry.")
    current_cells, current_coordinates = _periodic_transfer_frontiers(
        source_mesh, target_mesh, source_exact, target_exact
    )
    cells, vertices = _periodic_transfer_parent_banks(
        source, target, current_cells, current_coordinates
    )
    stencil, transfer = _owner_local_vertex_stencil(source_mesh, target_mesh, vertices)
    lineage = _owner_local_lineage(source_mesh, target_mesh, stencil, cells)
    if (
        not _owner_local_periodic_values_equal(stencil, result.vertex_stencil)
        or not _owner_local_periodic_values_equal(transfer, result.vertex_transfer)
        or not _owner_local_periodic_values_equal(lineage, result.lineage)
    ):
        raise ValueError("Periodic transfer claims differ from exact family ancestry.")
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    if (
        not np.array_equal(target_ids, np.asarray(result.target_vertex_ids))
        or tuple(target_exact[int(identifier)] for identifier in target_ids)
        != result.exact_vertex_coordinates
        or not np.array_equal(
            np.asarray(result.target_cell_parent_ids),
            np.asarray(
                [cells[int(identifier)] for identifier in result.target_cell_ids],
                dtype=np.int64,
            ),
        )
        or not np.array_equal(
            np.asarray(result.target_vertex_parent_ids),
            np.asarray(
                [vertices[int(identifier)] for identifier in target_ids], dtype=np.int64
            ),
        )
    ):
        raise ValueError("Periodic transfer target coordinates or parent rows changed.")


def _owner_local_periodic_values_equal(first: Any, second: Any, /) -> bool:
    if type(first) is not type(second):
        return False
    try:
        first_leaves, first_structure = jax.tree_util.tree_flatten(first)
        second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    except Exception:
        return False
    if first_structure != second_structure or len(first_leaves) != len(second_leaves):
        return False
    for left, right in zip(first_leaves, second_leaves, strict=True):
        if (
            hasattr(left, "shape")
            and hasattr(left, "dtype")
            and hasattr(right, "shape")
            and hasattr(right, "dtype")
        ):
            first_array = np.asarray(left)
            second_array = np.asarray(right)
            if (
                first_array.dtype != second_array.dtype
                or first_array.shape != second_array.shape
                or not np.array_equal(first_array, second_array)
            ):
                return False
            continue
        try:
            equal = left == right
        except Exception:
            return False
        if isinstance(equal, np.ndarray):
            if not bool(np.all(equal)):
                return False
        elif not bool(equal):
            return False
    return True


def _owner_local_periodic_cell_metadata(
    mesh: CellMesh | None,
    vertex_root_ids: np.ndarray,
    vertex_windings: np.ndarray,
    dimension: int,
    /,
) -> OwnerLocalPeriodicCellMetadata:
    if mesh is None:
        if np.asarray(vertex_root_ids).size or np.asarray(vertex_windings).size:
            raise ValueError("An empty periodic mesh has nonempty vertex metadata.")
        return OwnerLocalPeriodicCellMetadata(
            np.zeros((0,), dtype=np.int64),
            np.zeros((1,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, dimension), dtype=np.int64),
            np.zeros((0, dimension), dtype=np.int64),
        )
    if mesh.topological_dimension != dimension:
        raise ValueError("Periodic cell metadata dimension disagrees with its mesh.")
    roots = np.asarray(vertex_root_ids, dtype=np.int64)
    windings = np.asarray(vertex_windings, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if roots.shape != vertex_ids.shape or windings.shape != (
        vertex_ids.size,
        dimension,
    ):
        raise ValueError("Periodic cell metadata vertex rows do not align.")
    records: dict[
        int, tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]
    ] = {}
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for identifier, cell in zip(identifiers.tolist(), cells, strict=True):
            cell_rows = np.asarray(cell, dtype=np.int64).reshape((-1,))
            cell_roots = tuple(int(value) for value in roots[cell_rows].reshape((-1,)))
            cell_winding_rows = windings[cell_rows, :].reshape((-1, dimension))
            cell_windings = tuple(
                tuple(int(value) for value in row) for row in cell_winding_rows
            )
            records[int(identifier)] = _owner_local_periodic_cell_key(
                cell_roots,
                cell_windings,
            )
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if set(records) != set(int(value) for value in cell_ids.tolist()):
        raise ValueError("Periodic cell keys do not cover the complete local mesh.")
    ordered = tuple(records[int(identifier)] for identifier in cell_ids.tolist())
    corner_count = dimension + 1
    offsets = np.arange(
        0,
        (cell_ids.size + 1) * corner_count,
        corner_count,
        dtype=np.int64,
    )
    source_ids = np.asarray(
        [root for roots_row, _, _ in ordered for root in roots_row],
        dtype=np.int64,
    )
    relative_windings = np.asarray(
        [wind for _, relative_row, _ in ordered for wind in relative_row],
        dtype=np.int64,
    ).reshape((-1, dimension))
    cell_windings = np.asarray(
        [anchor for _, _, anchor in ordered], dtype=np.int64
    ).reshape((-1, dimension))
    return OwnerLocalPeriodicCellMetadata(
        cell_ids,
        offsets,
        source_ids,
        relative_windings,
        cell_windings,
    )


def _owner_local_periodic_exact_cell_measures(
    mesh: CellMesh | None,
    exact_vertex_coordinates: dict[int, tuple[Fraction, ...]],
    dimension: int,
    /,
) -> tuple[Fraction, ...]:
    if mesh is None:
        return ()
    if mesh.topological_dimension != dimension:
        raise ValueError("Exact periodic measures require matching cell dimension.")
    denomin = factorial(dimension)
    measure_by_cell: dict[int, Fraction] = {}
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        for cell_id, rows in zip(identifiers.tolist(), vertices, strict=True):
            corners = [
                exact_vertex_coordinates[int(vertex_ids[int(row)])]
                for row in np.asarray(rows, dtype=np.int64).reshape(-1)
            ]
            origin = corners[0]
            jacobian = tuple(
                tuple(
                    corners[column + 1][axis] - origin[axis] for axis in range(dimension)
                )
                for column in range(dimension)
            )
            determinant = determinant_exact(jacobian)
            if determinant == 0:
                raise ValueError("Exact periodic target simplex is degenerate.")
            measure_by_cell[int(cell_id)] = abs(determinant) / denomin
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    if set(measure_by_cell) != set(int(value) for value in cell_ids.tolist()):
        raise ValueError("Exact periodic measures omit a local target cell.")
    return tuple(measure_by_cell[int(identifier)] for identifier in cell_ids.tolist())


def _owner_local_periodic_finish(
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    source_mesh: CellMesh | None,
    source_cell_ids: np.ndarray,
    source_measures: dict[int, Fraction],
    vertex_root_ids: dict[int, int],
    vertex_windings: dict[int, tuple[int, ...]],
    vertex_owner_ranks: dict[int, int],
    exact_vertex_coordinates: dict[int, tuple[Fraction, ...]],
    size_evidence: OwnerLocalPeriodicSizeEvidence,
    local_rank: int,
    /,
    *,
    execution_group: ExecutionGroup,
    axis_name: str,
    status: MeshAdaptationStatus,
    refusal: str | None = None,
    graph_partition_plan: GraphPartitionPlan | None = None,
) -> OwnerLocalPeriodicAdaptiveExecution:
    dimension = layout.dimension
    (
        target_mesh,
        target_cell_owners,
        target_roots,
        target_exact,
        target_windings,
        target_vertex_owners,
    ) = _owner_local_periodic_current_mesh(
        state,
        source_mesh,
        vertex_root_ids,
        vertex_windings,
        vertex_owner_ranks,
        exact_vertex_coordinates,
        local_rank,
        f"owner-local-periodic-target:{candidate.candidate_id}:{status.value}",
    )
    target_vertex_ids = (
        np.zeros((0,), dtype=np.int64)
        if target_mesh is None
        else np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    )
    from ..discretization._periodic_topology import _entity_vertex_rows
    from ._distribution import (
        _owner_group_fail_if_any,
        _owner_group_minimum_owner_ranks,
    )

    owner_error = None
    candidate_rows: list[tuple[int, int]] = []
    try:
        if target_mesh is not None:
            cell_vertex_rows = _entity_vertex_rows(target_mesh)[dimension]
            if len(cell_vertex_rows) != target_cell_owners.size:
                raise ValueError(
                    "Periodic target cells and vertex ownership rows do not align."
                )
            for cell_vertices, owner in zip(
                cell_vertex_rows, target_cell_owners, strict=True
            ):
                candidate_rows.extend(
                    (
                        int(target_vertex_ids[int(vertex_row)]),
                        int(owner),
                    )
                    for vertex_row in np.asarray(cell_vertices, dtype=np.int64)
                )
    except Exception as error:
        owner_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        owner_error,
        execution_group,
        axis_name,
        "periodic target vertex ownership incidence",
    )
    owner_candidates = np.asarray(candidate_rows, dtype=np.int64).reshape((-1, 2))
    target_vertex_owners = _owner_group_minimum_owner_ranks(
        target_vertex_ids,
        owner_candidates,
        local_rank,
        execution_group,
        f"{axis_name}_vertex_owners",
    )
    target_metadata = _owner_local_periodic_cell_metadata(
        target_mesh, target_roots, target_windings, dimension
    )
    exact_target_measures = _owner_local_periodic_exact_cell_measures(
        target_mesh, exact_vertex_coordinates, dimension
    )
    source_ids = np.asarray(source_cell_ids, dtype=np.int64)
    source_cell_measures = tuple(
        source_measures[int(identifier)] for identifier in source_ids.tolist()
    )
    if sum(source_cell_measures, Fraction(0)) != sum(exact_target_measures, Fraction(0)):
        raise ValueError("Owner-local periodic bisection changed exact domain measure.")
    target_vertex_parents = np.full(
        (
            0 if target_mesh is None else np.asarray(target_mesh.vertex_global_ids).size,
            2,
        ),
        -1,
        dtype=np.int64,
    )
    target_cell_ids = np.zeros((0,), dtype=np.int64)
    target_cell_origins = np.zeros((0,), dtype=np.int64)
    target_cell_parent_ids = np.zeros((0,), dtype=np.int64)
    target_cell_owners = np.asarray(target_cell_owners, dtype=np.int32)
    lineage = None
    stencil = None
    transfer = None
    if source_mesh is None:
        if target_mesh is not None or source_ids.size:
            raise ValueError("An empty periodic owner lost its source partition.")
        source_weights = np.zeros((0,), dtype=np.float64)
    else:
        if target_mesh is None:
            raise ValueError("Periodic bisection removed every local source cell.")
        cell_parents, vertex_parents = _owner_local_parent_ids(state)
        stencil, transfer = _owner_local_vertex_stencil(
            source_mesh, target_mesh, vertex_parents
        )
        target_cell_ids, target_cell_origins, target_cell_parent_ids = (
            _owner_local_origins(state, source_ids)
        )
        lineage = _owner_local_lineage(
            source_mesh,
            target_mesh,
            stencil,
            cell_parents,
        )
        target_cell_order = np.asarray(
            target_mesh.entity_set(dimension).entity_ids, dtype=np.int64
        )
        if not np.array_equal(target_cell_order, target_cell_ids):
            raise ValueError("Periodic target cell ancestry order is not canonical.")
        source_measure_by_id = {
            int(identifier): measure
            for identifier, measure in zip(
                source_ids.tolist(), source_cell_measures, strict=True
            )
        }
        target_measure_by_id = {
            int(identifier): measure
            for identifier, measure in zip(
                target_cell_ids.tolist(), exact_target_measures, strict=True
            )
        }
        source_coverage = {int(identifier): Fraction(0) for identifier in source_ids}
        target_coverage = {int(identifier): Fraction(0) for identifier in target_cell_ids}
        cell_lineage = lineage.entity_lineage(dimension)
        for source_id, target_id, kind in zip(
            np.asarray(cell_lineage.source_global_ids, dtype=np.int64),
            np.asarray(cell_lineage.target_global_ids, dtype=np.int64),
            np.asarray(cell_lineage.relation_kinds, dtype=np.int32),
            strict=True,
        ):
            measure = (
                source_measure_by_id[int(source_id)]
                if kind == int(EntityLineageKind.COARSENED_INTO)
                else target_measure_by_id[int(target_id)]
            )
            source_coverage[int(source_id)] += measure
            target_coverage[int(target_id)] += measure
        if (
            source_coverage != source_measure_by_id
            or target_coverage != target_measure_by_id
        ):
            raise ValueError(
                "Periodic nested lineage changed exact physical cell coverage."
            )
        weight_ids = np.asarray(candidate.owned_cell_ids, dtype=np.int64)
        if (
            weight_ids.ndim != 1
            or np.unique(weight_ids).size != weight_ids.size
            or not np.array_equal(np.sort(weight_ids), np.sort(source_ids))
        ):
            raise ValueError(
                "Periodic source partition work rows do not cover the owned cells."
            )
        # Physical measures remain geometric evidence; graph work is unit per cell.
        source_weights = np.ones(source_ids.shape, dtype=np.int64)
        state_vertex_count = int(np.asarray(state.cursors, dtype=np.int64)[0])
        state_vertex_ids = np.asarray(state.mesh.vertex_ids, dtype=np.int64)[
            :state_vertex_count
        ]
        state_vertex_row = {
            int(identifier): row
            for row, identifier in enumerate(state_vertex_ids.tolist())
        }
        parent_slots = np.asarray(state.vertex_parents, dtype=np.int64)
        for target_row, identifier in enumerate(
            np.asarray(target_mesh.vertex_global_ids, dtype=np.int64).tolist()
        ):
            parents = parent_slots[state_vertex_row[int(identifier)]]
            if np.all(parents < 0):
                continue
            if np.any(parents < 0) or np.any(parents >= state_vertex_count):
                raise ValueError(
                    "Periodic target vertex ancestry has invalid parent slots."
                )
            target_vertex_parents[target_row] = state_vertex_ids[parents]
    target_cell_count = target_cell_ids.size
    target_cell_owners = np.asarray(target_cell_owners, dtype=np.int32)
    if target_cell_owners.shape != (target_cell_count,):
        raise ValueError("Periodic target cell owners do not align with cell lineage.")
    target_cell_count_in_state = int(np.asarray(state.cursors, dtype=np.int64)[1])
    active_rows = np.flatnonzero(
        np.asarray(state.mesh.cell_active, dtype=np.bool_)[:target_cell_count_in_state]
    )
    slot_origins = np.full((layout.cell_capacity,), -1, dtype=np.int64)
    if active_rows.size != target_cell_origins.size:
        raise ValueError("Periodic cell-origin rows do not align with the active state.")
    slot_origins[active_rows] = target_cell_origins
    status_flags = int(np.asarray(state.status_flags))
    counters = np.asarray(state.counters, dtype=np.int64)
    transition_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-adaptive-part",
            "candidate": candidate.candidate_id,
            "source_distribution": candidate.source_distribution_id,
            "source_part": candidate.source_part_id,
            "source_mesh": candidate.source_local_mesh_id,
            "source_scope": candidate.source_mesh_id,
            "source_owned_mesh": (None if source_mesh is None else source_mesh.mesh_id),
            "source_local_topology": candidate.source_local_topology_id,
            "source_local_geometry": candidate.source_local_geometry_id,
            "source_topology": candidate.source_topology_id,
            "target_owned_mesh": None if target_mesh is None else target_mesh.mesh_id,
            "target_owned_topology": (
                None if target_mesh is None else target_mesh.topology_id
            ),
            "target_vertex_owners": target_vertex_owners,
            "target_vertex_ids": target_vertex_ids,
            "stencil": None
            if stencil is None
            else tuple(
                tuple(
                    (
                        int(source_id),
                        rational.numerator,
                        rational.denominator,
                    )
                    for source_id, rational, active in zip(
                        stencil.source_global_ids[row],
                        (
                            rational_coefficient(
                                weight, role="Owner-local periodic stencil weight"
                            )
                            for weight in stencil.exact_weights[row]
                        ),
                        stencil.valid[row],
                        strict=True,
                    )
                    if active
                )
                for row in range(len(stencil.target_global_ids))
            ),
            "cell_metadata": (
                target_metadata.cell_ids,
                target_metadata.cell_key_offsets,
                target_metadata.cell_key_source_ids,
                target_metadata.cell_key_relative_windings,
                target_metadata.cell_windings,
            ),
            "graph_partition_plan": (
                None if graph_partition_plan is None else graph_partition_plan.plan_id
            ),
            "size_evidence": size_evidence.evidence_id,
            "status": status.value,
            "refusal": refusal,
        }
    )
    part_transition = OwnerLocalAdaptivePartTransition(
        local_rank,
        candidate.source_distribution_id,
        candidate.source_part_id,
        (candidate.source_local_mesh_id if source_mesh is None else source_mesh.mesh_id),
        candidate.source_topology_id,
        candidate.source_geometry_id,
        candidate.geometry_layout_id,
        candidate.coordinate_contract_id,
        candidate.periodic_topology_id,
        source_mesh,
        target_mesh,
        lineage,
        stencil,
        transfer,
        source_ids,
        target_cell_ids,
        target_cell_owners,
        target_cell_parent_ids,
        target_vertex_parents,
        source_cell_measures,
        exact_target_measures,
        source_weights,
        state,
        slot_origins,
        status_flags,
        counters,
        transition_id,
    )
    records = extract_simplex_forest(
        state,
        cell_owner_ranks=np.full(state.mesh.cell_capacity, local_rank, dtype=np.int32),
        vertex_owner_ranks=np.asarray(
            [
                vertex_owner_ranks.get(int(identifier), local_rank)
                for identifier in np.asarray(state.mesh.vertex_ids)
            ],
            dtype=np.int32,
        ),
    )
    forest_vertices = np.asarray(records.vertex_ids, dtype=np.int64)
    live_forest = PeriodicSimplexForest(
        records,
        np.asarray(
            [vertex_root_ids[int(identifier)] for identifier in forest_vertices],
            dtype=np.int64,
        ),
        np.asarray(
            [vertex_windings[int(identifier)] for identifier in forest_vertices],
            dtype=np.int64,
        ).reshape((-1, dimension)),
        tuple(
            exact_vertex_coordinates[int(identifier)] for identifier in forest_vertices
        ),
        tuple(block.name for block in source_mesh.blocks)
        if source_mesh is not None
        else ("periodic",),
        candidate.source_id,
        candidate.source_revision,
        candidate.periodic_cell_id,
    )
    result_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-adaptive-execution",
            "candidate": candidate.candidate_id,
            "specification": specification.specification_id,
            "status": status.value,
            "refusal": refusal,
            "transition": transition_id,
            "source_mesh_id": candidate.source_mesh_id,
            "source_local_mesh_id": candidate.source_local_mesh_id,
            "source_local_topology_id": candidate.source_local_topology_id,
            "source_local_geometry_id": candidate.source_local_geometry_id,
            "source_owned_mesh_id": (
                None if source_mesh is None else source_mesh.mesh_id
            ),
            "target_owned_mesh_id": (
                None if target_mesh is None else target_mesh.mesh_id
            ),
            "target_owned_topology_id": (
                None if target_mesh is None else target_mesh.topology_id
            ),
            "target_vertex_owners": target_vertex_owners,
            "target_vertex_ids": target_vertex_ids,
            "graph_partition_plan": (
                None if graph_partition_plan is None else graph_partition_plan.plan_id
            ),
            "size": size_evidence.evidence_id,
        }
    )
    return OwnerLocalPeriodicAdaptiveExecution(
        candidate.candidate_id,
        specification.specification_id,
        candidate.source_mesh_id,
        candidate.source_local_mesh_id,
        candidate.source_local_topology_id,
        candidate.source_local_geometry_id,
        None if source_mesh is None else source_mesh.mesh_id,
        None,
        None,
        None,
        None if target_mesh is None else target_mesh.mesh_id,
        None if target_mesh is None else target_mesh.topology_id,
        status,
        refusal,
        part_transition,
        target_vertex_ids,
        target_roots,
        target_windings,
        target_vertex_owners,
        target_exact,
        target_metadata,
        exact_target_measures,
        size_evidence,
        graph_partition_plan,
        live_forest,
        result_id,
    )


def _owner_local_periodic_refusal(
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    evidence: OwnerLocalPeriodicSizeEvidence,
    refusal: str,
    status: MeshAdaptationStatus,
    /,
    *,
    graph_partition_plan: GraphPartitionPlan | None = None,
) -> OwnerLocalPeriodicAdaptiveExecution:
    dimension = int(candidate.dimension)
    metadata = OwnerLocalPeriodicCellMetadata(
        np.zeros((0,), dtype=np.int64),
        np.zeros((1,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, dimension), dtype=np.int64),
        np.zeros((0, dimension), dtype=np.int64),
    )
    result_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-adaptation-refusal",
            "candidate": candidate.candidate_id,
            "source_mesh_id": candidate.source_mesh_id,
            "source_local_mesh_id": candidate.source_local_mesh_id,
            "source_local_topology_id": candidate.source_local_topology_id,
            "source_local_geometry_id": candidate.source_local_geometry_id,
            "specification": specification.specification_id,
            "evidence": evidence.evidence_id,
            "status": status.value,
            "graph_partition_plan": (
                None if graph_partition_plan is None else graph_partition_plan.plan_id
            ),
            "refusal": refusal,
        }
    )
    return OwnerLocalPeriodicAdaptiveExecution(
        candidate.candidate_id,
        specification.specification_id,
        candidate.source_mesh_id,
        candidate.source_local_mesh_id,
        candidate.source_local_topology_id,
        candidate.source_local_geometry_id,
        None,
        None,
        None,
        None,
        None,
        None,
        status,
        refusal,
        None,
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, dimension), dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
        (),
        metadata,
        (),
        evidence,
        graph_partition_plan,
        None,
        result_id,
    )


def _owner_local_periodic_refusal_result(
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    evidence: OwnerLocalPeriodicSizeEvidence,
    refusal: str,
    status: MeshAdaptationStatus,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
    *,
    graph_partition_plan: GraphPartitionPlan | None = None,
) -> OwnerLocalPeriodicAdaptiveExecution:
    result = _owner_local_periodic_refusal(
        candidate,
        specification,
        evidence,
        refusal,
        status,
        graph_partition_plan=graph_partition_plan,
    )
    return _owner_local_periodic_globalize_result(
        result, candidate, specification, execution_group, axis_name
    )


def _owner_local_periodic_try_coarsen(
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    source_mesh: CellMesh | None,
    vertex_root_ids: dict[int, int],
    vertex_windings: dict[int, tuple[int, ...]],
    vertex_owner_ranks: dict[int, int],
    exact_vertex_coordinates: dict[int, tuple[Fraction, ...]],
    current_epoch: _OwnerLocalPeriodicEdgeEpoch,
    split_history: set[tuple[int, ...]],
    next_vertex_id: int,
    next_cell_id: int,
    local_rank: int,
    part_count: int,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[AdaptiveSimplexState, _OwnerLocalPeriodicEdgeEpoch, bool, str | None]:
    from ._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    last_refusal: str | None = None
    attempted: set[int] = set()
    while True:
        local_candidates = tuple(
            root
            for root in _owner_local_periodic_active_midpoint_roots(
                state, vertex_root_ids
            )
            if root not in attempted
        )
        # Stable numeric issuance depends on placement; order midpoint orbits by
        # their exact lattice-unshifted coordinates instead.
        import json

        lattice = tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in np.asarray(candidate.periodic_cell.vectors, dtype=np.float64)
        )
        root_points: dict[int, tuple[Fraction, ...]] = {}
        for identifier, root in vertex_root_ids.items():
            if root not in local_candidates:
                continue
            point = exact_vertex_coordinates[identifier]
            winding = vertex_windings[identifier]
            unshifted = tuple(
                point[axis]
                - sum(
                    (
                        int(winding[index]) * lattice[index][axis]
                        for index in range(layout.dimension)
                    ),
                    Fraction(0),
                )
                for axis in range(layout.dimension)
            )
            previous = root_points.setdefault(root, unshifted)
            if previous != unshifted:
                raise ValueError(
                    "Periodic midpoint images disagree on their exact orbit root."
                )
        local_choice = min(
            ((point, root) for root, point in root_points.items()), default=None
        )
        payload = json.dumps(
            None
            if local_choice is None
            else (
                [(value.numerator, value.denominator) for value in local_choice[0]],
                local_choice[1],
            ),
            separators=(",", ":"),
        ).encode("ascii")
        lengths = np.asarray(
            _owner_group_allgather(
                np.asarray((len(payload),), dtype=np.int64), execution_group, axis_name
            ),
            dtype=np.int64,
        ).reshape((-1,))
        width = int(np.max(lengths))
        encoded = np.zeros((width,), dtype=np.uint8)
        encoded[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        payloads = np.asarray(
            _owner_group_allgather(encoded, execution_group, axis_name), dtype=np.uint8
        ).reshape((part_count, width))
        choices = []
        for row, length in zip(payloads, lengths, strict=True):
            value = json.loads(row[: int(length)].tobytes().decode("ascii"))
            if value is not None:
                choices.append(
                    (tuple(Fraction(*item) for item in value[0]), int(value[1]))
                )
        if not choices:
            return (
                state,
                current_epoch,
                False,
                last_refusal
                or "No derived periodic midpoint orbit remains eligible for coarsening.",
            )
        selected_root = min(choices)[1]
        attempted.add(selected_root)
        trial_state, changed, trial_refusal = _owner_local_periodic_coarsen_orbit(
            layout,
            state,
            selected_root,
            vertex_root_ids,
            next_vertex_id,
            next_cell_id,
            local_rank,
            part_count,
            execution_group,
            axis_name,
        )
        if not changed:
            last_refusal = trial_refusal or (
                "A derived periodic midpoint orbit was not completely coarsenable."
            )
            continue
        local_error = None
        try:
            (
                target_mesh,
                target_cell_owners,
                target_roots,
                target_exact,
                target_windings,
                _target_vertex_owners,
            ) = _owner_local_periodic_current_mesh(
                trial_state,
                source_mesh,
                vertex_root_ids,
                vertex_windings,
                vertex_owner_ranks,
                exact_vertex_coordinates,
                local_rank,
                f"owner-local-periodic-coarsen-trial:{candidate.candidate_id}",
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            target_mesh = None
            target_cell_owners = np.zeros((0,), dtype=np.int32)
            target_roots = np.zeros((0,), dtype=np.int64)
            target_exact = ()
            target_windings = np.zeros((0, layout.dimension), dtype=np.int64)
            _target_vertex_owners = np.zeros((0,), dtype=np.int32)
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic coarsen target mesh"
        )
        trial_epoch = _owner_local_periodic_size_epoch(
            candidate,
            specification,
            execution_group,
            local_rank,
            mesh_override=target_mesh,
            cell_owner_ranks_override=target_cell_owners,
            vertex_root_ids_override=target_roots,
            vertex_windings_override=target_windings,
            exact_vertex_coordinates_override=target_exact,
            use_override=True,
            axis_name=axis_name,
        )
        selected_after = trial_epoch.evidence.selected_edge_key
        recreates_split_edge = False
        if selected_after is not None:
            history_matches = np.asarray(
                _owner_group_allgather(
                    np.asarray(
                        (int(selected_after in split_history),),
                        dtype=np.int64,
                    ),
                    execution_group,
                    axis_name,
                ),
                dtype=np.int64,
            ).reshape((-1,))
            recreates_split_edge = bool(np.any(history_matches))
        if recreates_split_edge:
            last_refusal = (
                "Coarsening would recreate an edge selected for refinement "
                "in this metric epoch."
            )
            attempted.add(selected_root)
            continue
        if _owner_local_periodic_coarsen_is_admissible(
            current_epoch.evidence, trial_epoch.evidence
        ):
            return trial_state, trial_epoch, True, None
        last_refusal = (
            "Coarsening would move exact target-size evidence outside its "
            "admissible monotone interval."
        )
        attempted.add(selected_root)


def _adapt_owner_local_periodic(
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    execution_group: ExecutionGroup,
    local_rank: int,
    /,
    *,
    axis_name: str = "periodic_owner_adaptation",
    graph_partition_plan: GraphPartitionPlan | None = None,
    live_forest: PeriodicSimplexForest | None = None,
) -> OwnerLocalPeriodicAdaptiveExecution:
    """Refine/coarsen fixed-site periodic Delaunay cells under exact size evidence.

    The owner group never gathers a target mesh. `graph_partition_plan` is used
    only when explicitly supplied; repartition/migration follows target-view
    construction and exact periodic face/CSR acceptance in the provider.
    """

    from .._execution_runtime import ExecutionGroup
    from ..graph._partition import GraphPartitionPlan
    from ._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
    from ._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )
    from ._periodic import OwnerLocalPeriodicCandidate

    if not isinstance(candidate, OwnerLocalPeriodicCandidate):
        raise TypeError("candidate must be OwnerLocalPeriodicCandidate.")
    if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
        raise TypeError("specification must be SurfaceMeshingSpec or VolumeMeshingSpec.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    part_count = int(candidate.part_count)
    local_error = None
    if graph_partition_plan is not None and (
        not isinstance(graph_partition_plan, GraphPartitionPlan)
        or graph_partition_plan.part_count != part_count
    ):
        local_error = (
            "Periodic repartition requires an explicit GraphPartitionPlan "
            "matching the current owner group."
        )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic repartition plan"
    )
    plan_id = None if graph_partition_plan is None else graph_partition_plan.plan_id
    plan_token = canonical_fingerprint(
        {
            "kind": "periodic-owner-repartition-request",
            "source": candidate.source_id,
            "source_revision": candidate.source_revision,
            "source_manifest": candidate.source_manifest_id,
            "distribution": candidate.source_distribution_id,
            "source_part": candidate.source_part_id,
            "authored_mesh": candidate.source_mesh_id,
            "dimension": candidate.dimension,
            "face_coverage": candidate.face_coverage_id,
            "source_topology": candidate.source_topology_id,
            "source_geometry": candidate.source_geometry_id,
            "coordinate_contract": candidate.coordinate_contract_id,
            "periodic_topology": candidate.periodic_topology_id,
            "periodic_cell": candidate.periodic_cell_id,
            "specification": specification.specification_id,
            "group": execution_group.spec.group_id,
            "plan": plan_id,
            "parts": part_count,
        }
    )
    plan_records = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(plan_token), dtype=np.uint8),
        execution_group,
        axis_name,
    )
    if any(
        np.asarray(record, dtype=np.uint8).tobytes().hex() != plan_token
        for record in plan_records
    ):
        raise ValueError("Periodic repartition plans differ across owner ranks.")
    source = _owner_local_periodic_state(
        candidate, local_rank, execution_group, axis_name, live_forest=live_forest
    )
    layout = source.layout
    state = source.state
    vertex_root_ids = dict(source.vertex_root_ids)
    vertex_windings = dict(source.vertex_windings)
    vertex_owner_ranks = dict(source.vertex_owner_ranks)
    exact_vertex_coordinates = dict(source.exact_coordinates)
    cell_ids = np.asarray(candidate.cell_ids, dtype=np.int64)
    cell_windings_array = np.asarray(candidate.cell_windings, dtype=np.int64)
    local_error = (
        "Periodic source cell winding rows do not align with cell IDs."
        if cell_windings_array.shape != (cell_ids.size, layout.dimension)
        else None
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic source cell windings"
    )
    cell_windings = {
        int(identifier): tuple(int(value) for value in cell_windings_array[row])
        for row, identifier in enumerate(cell_ids.tolist())
    }
    epoch = _owner_local_periodic_size_epoch(
        candidate,
        specification,
        execution_group,
        local_rank,
        axis_name=axis_name,
    )
    if epoch.evidence.compliant:
        local_error = None
        try:
            result = _owner_local_periodic_finish(
                candidate,
                specification,
                layout,
                state,
                source.source_mesh,
                source.source_cell_ids,
                source.source_measures,
                vertex_root_ids,
                vertex_windings,
                vertex_owner_ranks,
                exact_vertex_coordinates,
                epoch.evidence,
                local_rank,
                execution_group=execution_group,
                axis_name=f"{axis_name}_target_vertex_owners",
                status=MeshAdaptationStatus.UNCHANGED,
                graph_partition_plan=graph_partition_plan,
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            result = None
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "periodic unchanged transition"
        )
        if result is None:
            raise ValueError("Periodic unchanged transition failed collectively.")
        return _owner_local_periodic_globalize_result(
            result, candidate, specification, execution_group, axis_name
        )
    next_vertex_id = int(np.asarray(state.cursors, dtype=np.int64)[2])
    next_cell_id = int(np.asarray(state.cursors, dtype=np.int64)[3])
    changed = False
    split_history: set[tuple[int, ...]] = set()
    while not epoch.evidence.compliant:
        if epoch.evidence.too_small:
            state, epoch, coarsened, coarsen_refusal = _owner_local_periodic_try_coarsen(
                candidate,
                specification,
                layout,
                state,
                source.source_mesh,
                vertex_root_ids,
                vertex_windings,
                vertex_owner_ranks,
                exact_vertex_coordinates,
                epoch,
                split_history,
                next_vertex_id,
                next_cell_id,
                local_rank,
                part_count,
                execution_group,
                axis_name,
            )
            if coarsened:
                changed = True
                continue
            refusal = coarsen_refusal or "; ".join(epoch.evidence.issues)
            return _owner_local_periodic_refusal_result(
                candidate,
                specification,
                epoch.evidence,
                refusal,
                MeshAdaptationStatus.STALLED,
                execution_group,
                axis_name,
                graph_partition_plan=graph_partition_plan,
            )
        selected_edge = epoch.evidence.selected_edge_key
        if not epoch.evidence.too_large or selected_edge is None:
            return _owner_local_periodic_refusal_result(
                candidate,
                specification,
                epoch.evidence,
                "; ".join(epoch.evidence.issues),
                MeshAdaptationStatus.STALLED,
                execution_group,
                axis_name,
                graph_partition_plan=graph_partition_plan,
            )
        pending_keys: tuple[tuple[int, ...], ...] = (selected_edge,)
        first_request = True
        while True:
            local_requests = _owner_local_periodic_route_split_keys(
                pending_keys,
                epoch.edge_owners,
                layout.dimension,
                local_rank,
                part_count,
                execution_group,
                axis_name,
            )
            request_counts = np.asarray(
                _owner_group_allgather(
                    np.asarray((len(local_requests),), dtype=np.int64),
                    execution_group,
                    axis_name,
                ),
                dtype=np.int64,
            ).reshape((-1,))
            any_requests = bool(np.any(request_counts))
            if not any_requests:
                if first_request:
                    return _owner_local_periodic_refusal_result(
                        candidate,
                        specification,
                        epoch.evidence,
                        "The selected periodic quotient edge has no current owner.",
                        MeshAdaptationStatus.NUMERICAL_STOP,
                        execution_group,
                        axis_name,
                        graph_partition_plan=graph_partition_plan,
                    )
                break
            old_vertex_count, old_cell_count = map(
                int, np.asarray(state.cursors, dtype=np.int64)[:2]
            )
            old_vertex_ids = set(
                int(value)
                for value in np.asarray(state.mesh.vertex_ids, dtype=np.int64)[
                    :old_vertex_count
                ].tolist()
            )
            old_cell_ids = set(
                int(value)
                for value in np.asarray(state.mesh.cell_ids, dtype=np.int64)[
                    :old_cell_count
                ].tolist()
            )
            local_error = None
            try:
                state, marks, marked = _owner_local_periodic_mark_edges(
                    layout,
                    state,
                    local_requests,
                    vertex_root_ids,
                    vertex_windings,
                )
                if local_requests and not marked:
                    raise ValueError(
                        "A routed periodic edge owner has no matching active cell."
                    )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
                marked = False
                marks = jnp.zeros((layout.cell_capacity,), dtype=jnp.bool_)
            _owner_group_fail_if_any(
                local_error, execution_group, axis_name, "periodic edge marking"
            )
            local_error = None
            try:
                # Empty owners still participate in every collective and
                # validate the accepted epoch, but have no local bisection.
                active_cells = np.asarray(state.mesh.cell_active, dtype=np.bool_)
                if np.any(active_cells):
                    update = refine_adaptive_simplex(layout, state, marks)
                    require_applied_status(
                        int(np.asarray(jax.device_get(update.report.status))),
                        DeviceEpoch.BISECTION,
                    )
                    next_layout, next_state = update.layout, update.state
                else:
                    if np.any(np.asarray(marks, dtype=np.bool_)):
                        raise ValueError("An empty periodic owner cannot mark a cell.")
                    require_applied_status(
                        int(np.asarray(state.status_flags)), DeviceEpoch.BISECTION
                    )
                    next_layout, next_state = layout, state
                host_state = jax.device_get(next_state)
                vertex_count, cell_count = map(
                    int, np.asarray(host_state.cursors, dtype=np.int64)[:2]
                )
                active = np.asarray(
                    host_state.mesh.cell_active[:cell_count], dtype=np.bool_
                )
                require_committed_status(
                    int(np.asarray(host_state.status_flags)),
                    np.asarray(host_state.mesh.coordinates[:vertex_count]),
                    np.asarray(host_state.mesh.cells[:cell_count])[active],
                    DeviceEpoch.BISECTION,
                )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
                next_state = None
            _owner_group_fail_if_any(
                local_error, execution_group, axis_name, "periodic bisection acceptance"
            )
            if next_state is None:
                raise ValueError("Periodic bisection failed collectively.")
            layout = next_layout
            state = next_state
            local_error = None
            try:
                assigned = _owner_local_periodic_assign_new_ids(
                    layout,
                    state,
                    old_vertex_ids,
                    old_cell_ids,
                    vertex_root_ids,
                    vertex_windings,
                    vertex_owner_ranks,
                    exact_vertex_coordinates,
                    cell_windings,
                    next_vertex_id,
                    next_cell_id,
                    local_rank,
                    part_count,
                    execution_group,
                    axis_name,
                )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
                assigned = None
            _owner_group_fail_if_any(
                local_error, execution_group, axis_name, "periodic exact ID allocation"
            )
            if assigned is None:
                raise ValueError("Periodic exact-ID allocation failed collectively.")
            (
                state,
                vertex_root_ids,
                vertex_windings,
                vertex_owner_ranks,
                exact_vertex_coordinates,
                cell_windings,
                next_vertex_id,
                next_cell_id,
                split_keys,
            ) = assigned
            split_history.update(split_keys)
            split_counts = np.asarray(
                _owner_group_allgather(
                    np.asarray((len(split_keys),), dtype=np.int64),
                    execution_group,
                    axis_name,
                ),
                dtype=np.int64,
            ).reshape((-1,))
            if not np.any(split_counts):
                return _owner_local_periodic_refusal_result(
                    candidate,
                    specification,
                    epoch.evidence,
                    "A marked periodic edge produced no representable midpoint.",
                    MeshAdaptationStatus.NUMERICAL_STOP,
                    execution_group,
                    axis_name,
                    graph_partition_plan=graph_partition_plan,
                )
            if marked and not split_keys:
                local_error = (
                    "A local marked periodic edge produced no local split midpoint."
                )
            else:
                local_error = None
            _owner_group_fail_if_any(
                local_error, execution_group, axis_name, "periodic split progress"
            )
            local_error = None
            try:
                (
                    target_mesh,
                    target_cell_owners,
                    target_roots,
                    target_exact,
                    target_windings,
                    _target_vertex_owners,
                ) = _owner_local_periodic_current_mesh(
                    state,
                    source.source_mesh,
                    vertex_root_ids,
                    vertex_windings,
                    vertex_owner_ranks,
                    exact_vertex_coordinates,
                    local_rank,
                    f"owner-local-periodic-epoch:{candidate.candidate_id}",
                )
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
                target_mesh = None
                target_cell_owners = np.zeros((0,), dtype=np.int32)
                target_roots = np.zeros((0,), dtype=np.int64)
                target_exact = ()
                target_windings = np.zeros((0, layout.dimension), dtype=np.int64)
                _target_vertex_owners = np.zeros((0,), dtype=np.int32)
            _owner_group_fail_if_any(
                local_error, execution_group, axis_name, "periodic target mesh"
            )
            epoch = _owner_local_periodic_size_epoch(
                candidate,
                specification,
                execution_group,
                local_rank,
                mesh_override=target_mesh,
                cell_owner_ranks_override=target_cell_owners,
                vertex_root_ids_override=target_roots,
                vertex_windings_override=target_windings,
                exact_vertex_coordinates_override=target_exact,
                use_override=True,
                axis_name=axis_name,
            )
            changed = True
            pending_keys = split_keys
            first_request = False
        if epoch.evidence.compliant:
            break
    local_error = None
    try:
        result = _owner_local_periodic_finish(
            candidate,
            specification,
            layout,
            state,
            source.source_mesh,
            source.source_cell_ids,
            source.source_measures,
            vertex_root_ids,
            vertex_windings,
            vertex_owner_ranks,
            exact_vertex_coordinates,
            epoch.evidence,
            local_rank,
            execution_group=execution_group,
            axis_name=f"{axis_name}_target_vertex_owners",
            status=(
                MeshAdaptationStatus.COMPLETE
                if changed
                else MeshAdaptationStatus.UNCHANGED
            ),
            graph_partition_plan=graph_partition_plan,
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        result = None
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic adaptive commit"
    )
    if result is None:
        raise ValueError("Periodic adaptive commit failed collectively.")
    return _owner_local_periodic_globalize_result(
        result, candidate, specification, execution_group, axis_name
    )


def _owner_local_periodic_globalize_result(
    result: OwnerLocalPeriodicAdaptiveExecution,
    candidate: OwnerLocalPeriodicCandidate,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> OwnerLocalPeriodicAdaptiveExecution:
    from ._distribution import _owner_group_allgather

    if result.part_transition is None:
        local_id = result.result_id
    else:
        local_id = result.part_transition.transition_id
    records = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(local_id), dtype=np.uint8),
        execution_group,
        axis_name,
    )
    part_ids = tuple(
        np.asarray(record, dtype=np.uint8).tobytes().hex() for record in records
    )
    result_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-adaptive-global-execution",
            "source": candidate.source_id,
            "source_revision": candidate.source_revision,
            "source_manifest": candidate.source_manifest_id,
            "distribution": candidate.source_distribution_id,
            "source_part": candidate.source_part_id,
            "authored_mesh": candidate.source_mesh_id,
            "source_topology": candidate.source_topology_id,
            "source_geometry": candidate.source_geometry_id,
            "coordinate_contract": candidate.coordinate_contract_id,
            "periodic_topology": candidate.periodic_topology_id,
            "face_coverage": candidate.face_coverage_id,
            "periodic_cell": candidate.periodic_cell_id,
            "specification": specification.specification_id,
            "parts": part_ids,
            "status": result.status.value,
            "refusal": result.refusal,
            "size": result.size_evidence.evidence_id,
            "graph_partition_plan": (
                None
                if result.graph_partition_plan is None
                else result.graph_partition_plan.plan_id
            ),
        }
    )
    return result._replace(result_id=result_id)


__all__ = [
    "OwnerLocalAdaptivePartTransition",
    "OwnerLocalPeriodicAdaptiveExecution",
    "OwnerLocalPeriodicAdaptiveResult",
    "OwnerLocalPeriodicCellMetadata",
    "PartitionedAdaptiveSimplexResult",
    "PartitionedAdaptiveSimplex",
    "PreparedAdaptiveSimplex",
    "commit_adaptive_simplex",
    "commit_partitioned_adaptive_simplex",
    "partition_adaptive_simplex",
    "prepare_adaptive_simplex",
]
