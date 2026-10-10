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

import hashlib
import time
from collections.abc import Callable, Sequence
from itertools import combinations
from math import prod
from typing import Any, final, NamedTuple, TYPE_CHECKING, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
    logical_array_value_collection_digest,
)
from .._meshcore import (
    current_native_host_workspace,
    exact_orient2d,
    MeshcoreError,
    MeshcoreUnavailableError,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    _restored_facet_classes,
    _work,
    adaptive_simplex_bucket,
    adaptive_simplex_state,
    AdaptiveSimplexCounter,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex,
    coarsen_adaptive_simplex_parts,
    MaskedSimplexMesh,
    refine_adaptive_simplex,
    refine_adaptive_simplex_parts,
)
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellGeometryStorageProjection,
)
from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
from ..discretization._cell_mesh import CellBlock, CellMesh, CellMeshStorage
from ..geometry._collective_domain_coverage import certify_collective_premises
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    PiecewiseLinearDomain,
)
from ._adaptation import (
    _adaptation_result,
    _compliance,
    _finalize_native,
    _region_boundary_transition,
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
from ._assembly import MeshPart
from ._association import (
    BRepAssociationTransfer,
    PlcAssociationTransfer,
    SurfaceAssociationTransfer,
)
from ._bisection import (
    _Cells,
    _Coarsening,
    _edit,
    _Flags,
    _Front,
    _global_vertices,
    _Growth,
    _prepared_request,
    _prepared_source,
    _prepared_start,
    _Records,
    _Refinement,
    _relations,
    _resolved_stencil,
    _reverse_supports,
    _SHIFT,
    _simplex_keys,
    _Source,
    _Start,
    _target_hierarchy,
    _uniform_charge,
    _uniform_execution,
    _uniform_inverse,
    _UniformAllowance,
    BisectionEvidence,
    BisectionHierarchy,
    BisectionUniformRefinement,
)
from ._canonical import certify_owner_local_cell_mesh
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._device_status import (
    DeviceEpoch,
    require_applied_status,
    require_committed_status,
)
from ._distribution import (
    _rows_of,
    AffineBisectionSourceWitness,
    expand_partitioned_simplex_neighborhood,
    prepare_owner_local_distribution_transition,
    SimplexNeighborhoodWorkset,
)
from ._distribution_migration import _WORKSET_FIELDS, SimplexGraphRestartProposal
from ._lineage import (
    CellMeshTransition,
    EntityLineage,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)
from ._measurements import NativeExecutionRecord
from ._organization import (
    lower_mesh_organization,
    MeshAttributeProjection,
    prepare_mesh_attribute_projections,
    prepare_mesh_organization_scopes,
)
from ._publication_lowering import PreparedPublicationLowering, PublicationProjection
from ._result import (
    CellMeshingResult,
    CollectiveMeshEvidence,
    CollectiveMeshStorageBinding,
    require_original_meshing_source,
)
from ._scope import _local_logical_lookup, MeshScopeProjection


if TYPE_CHECKING:
    from ._initial_certification import InitialCollectiveMeshEvidence
    from ._restart_distribution import (
        _SimplexRestartInventory,
        SimplexRestartRepack,
        SimplexRestartRepackProof,
    )
from ._topology_edit import (
    _entity_lineage,
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    nested_reference_vertices,
    TopologyEditBlock,
)


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
    block_indices: np.ndarray | None = None
    source_blocks: tuple[CellBlock, ...] | None = None


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
    execution_evidence: NativeExecutionRecord | None
    source_topology_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        adaptation: PreparedMeshAdaptation,
        layout: AdaptiveSimplexLayout,
        state: AdaptiveSimplexState,
        anchor: _Anchor,
        /,
        *,
        execution_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        _require_preparation_receipt(adaptation, execution_evidence)
        self.adaptation = adaptation
        self.layout = layout
        self.state = state
        self.anchor = anchor
        self.execution_evidence = execution_evidence
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
    if start.uniform_refinement is not None and start.uniform:
        _uniform_charge(rows.size, rows.nbytes * 4)
        arrays = start.uniform_refinement.host_arrays()
        columns = tuple(
            tuple(index for index in range(source.dimension + 1) if index != opposite)
            for opposite in range(source.dimension + 1)
        )
        for parent, siblings in enumerate(arrays["child_ids"]):
            for child_index, child in enumerate(siblings):
                slot = np.searchsorted(ids, child)
                for opposite, local in enumerate(columns):
                    weights = arrays["barycentric_weights"][
                        parent, child_index, list(local)
                    ]
                    for facet, original_local in enumerate(
                        combinations(range(source.dimension + 1), source.dimension)
                    ):
                        original_opposite = next(
                            index
                            for index in range(source.dimension + 1)
                            if index not in original_local
                        )
                        if np.all(weights[:, original_opposite] == 0.0):
                            facet_classes[slot, opposite] = (
                                arrays["parent_facet_classes"][parent, facet] + 1
                            )
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


def _require_preparation_receipt(
    adaptation: PreparedMeshAdaptation, receipt: NativeExecutionRecord | None, /
) -> None:
    if receipt is None:
        return
    if type(receipt) is not NativeExecutionRecord:
        raise TypeError("Preparation requires its exact ended native record.")
    receipt.require_valid()
    if receipt.owner_id != adaptation.prepared_id or int(np.asarray(receipt.status)) != 0:
        raise ValueError(
            "Preparation receipt does not bind this successful scientific operation."
        )
    phase = receipt
    while phase is not None:
        if phase.owner_id != adaptation.prepared_id:
            raise ValueError(
                "Preparation receipt includes a phase from a different scientific operation."
            )
        if int(np.asarray(phase.status)) != 0:
            raise ValueError("Preparation receipt includes a failed preparation phase.")
        if phase.preparation_evidence is None and (
            int(np.asarray(phase.source_preparation_work_units))
            or int(np.asarray(phase.source_preparation_geometry_queries))
            or float(np.asarray(phase.preparation_seconds))
        ):
            raise ValueError(
                "Adaptation preparation requires actual ended preparation records, not source-field aliases."
            )
        phase = phase.preparation_evidence


def _phase_allowance(
    adaptation: PreparedMeshAdaptation, receipt: NativeExecutionRecord | None, /
) -> _UniformAllowance:
    _require_preparation_receipt(adaptation, receipt)
    limits = adaptation.policy.limits
    work = 0 if receipt is None else int(np.asarray(receipt.total_work_units))
    queries = 0 if receipt is None else int(np.asarray(receipt.total_geometry_queries))
    seconds = 0.0 if receipt is None else float(np.asarray(receipt.total_elapsed_seconds))
    if (
        work >= limits.maximum_work_units
        or queries > limits.maximum_geometry_queries
        or seconds >= limits.maximum_wall_seconds
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Actual completed preparation exhausted the original operation allowance.",
            stage="device-preparation-admission",
        )
    return _UniformAllowance(
        limits.maximum_work_units - work,
        limits.maximum_geometry_queries - queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds - seconds,
        limits.maximum_cavity_cells,
    )


def _retain_preparation(
    value: PreparedAdaptiveSimplex | PartitionedAdaptiveSimplex, /
) -> None:
    workspace = current_native_host_workspace()
    if workspace is None:
        raise RuntimeError(
            "Prepared source retention lost its canonical native workspace."
        )
    workspace.retain_owner(value)


def _prepared_simplex(adaptation: PreparedMeshAdaptation, /) -> PreparedAdaptiveSimplex:
    limits = adaptation.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance) as budget:
        prepared = _run_prepared_simplex(adaptation)
    if budget.evidence is None:
        return prepared
    return PreparedAdaptiveSimplex(
        adaptation,
        prepared.layout,
        prepared.state,
        prepared.anchor,
        execution_evidence=NativeExecutionRecord(
            budget.evidence, owner_id=adaptation.prepared_id
        ),
    )


def _run_prepared_simplex(
    adaptation: PreparedMeshAdaptation, /
) -> PreparedAdaptiveSimplex:
    policy = adaptation.policy
    device = policy.device_policy
    constraints = adaptation.constraints
    mesh = adaptation.source.mesh
    if mesh.storage is not None:
        from ._initial_certification import InitialCollectiveMeshEvidence

        initial_evidence = adaptation.source.collective_evidence
        if not isinstance(initial_evidence, InitialCollectiveMeshEvidence):
            raise ValueError(
                "Owner-local source preparation requires its actual native initial theorem."
            )
        initial_evidence.require_passed()
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
    if not isinstance(adaptation.request, MarkedMeshAdaptation):
        raise TypeError("Device simplex preparation requires its marked request.")
    hierarchy = adaptation.request.hierarchy
    if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError(
            "Device simplex preparation requires its canonical bisection hierarchy."
        )
    start = _prepared_start(
        source,
        request,
        policy.compatibility,
        hierarchy,
        _UniformAllowance(
            policy.limits.maximum_work_units,
            policy.limits.maximum_geometry_queries,
            policy.limits.maximum_cells,
            policy.limits.maximum_vertices,
            policy.limits.maximum_scratch_bytes,
            policy.limits.maximum_wall_seconds,
            policy.limits.maximum_cavity_cells,
        ),
        adaptation.source,
        future_refinement=True,
    )
    forest = _forest(start, source, request)
    vertices = _vertices(start, source, request)
    if policy.distribution is None:
        # ty: ignore[unresolved-attribute]
        vertex_capacity, cell_capacity = device.capacities(
            vertices["ids"].size, forest["ids"].size
        )
    else:
        # The dense forest is immutable preparation, not one owner's execution
        # bucket. Partitioning below reapplies the exact requested per-owner caps.
        # ty: ignore[unresolved-attribute]
        requested_vertices = (
            adaptive_simplex_bucket(vertices["ids"].size, device.growth_factor)
            if device.vertex_capacity is None
            else device.vertex_capacity
        )
        # ty: ignore[unresolved-attribute]
        requested_cells = (
            adaptive_simplex_bucket(forest["ids"].size, device.growth_factor)
            if device.cell_capacity is None
            else device.cell_capacity
        )
        vertex_capacity = max(
            requested_vertices, adaptive_simplex_bucket(vertices["ids"].size, 1.0)
        )
        cell_capacity = max(
            requested_cells, adaptive_simplex_bucket(forest["ids"].size, 1.0)
        )
    protected = request.protected_codes
    layout = AdaptiveSimplexLayout(
        source.dimension,
        mesh.ambient_dimension,
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        protected_edge_capacity=adaptive_simplex_bucket(protected.size, 1.0),
        maximum_closure_iterations=policy.maximum_closure_iterations,
        # ty: ignore[unresolved-attribute]
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
    if start.records.parent_ids.size:
        depth = int(np.max(forest["generations"]))

        def restore_source_classes(
            step: jax.Array,
            fields: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            classes, facets = fields
            selected = (
                ~state.mesh.cell_active
                & (state.children[:, 0] >= 0)
                & (state.generations == depth - step)
            )
            first = jnp.maximum(state.children[:, 0], 0)
            restored_facets = _restored_facet_classes(
                _work(state)._replace(facet_classes=facets)
            )
            return (
                jnp.where(selected, classes[first], classes),
                jnp.where(selected[:, None], restored_facets, facets),
            )

        fields = jax.lax.fori_loop(
            0,
            depth + 1,
            restore_source_classes,
            (state.cell_classes, state.facet_classes),
        )
        state = eqx.tree_at(
            lambda value: (value.cell_classes, value.facet_classes), state, fields
        )
    if mesh.storage is not None:
        evidence = adaptation.source.collective_evidence
        if evidence is None:
            raise ValueError(
                "Initial scientific root allocation requires actual global identity banks."
            )
        issuers = jnp.asarray(
            (
                jnp.max(evidence.entity_ids[0]) + 1,
                jnp.max(evidence.entity_ids[-1]) + 1,
            ),
            dtype=jnp.int64,
        )
        state = eqx.tree_at(
            lambda value: value.cursors, state, state.cursors.at[2:].set(issuers)
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
    limits = policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance) as budget:
        adaptation = prepare_mesh_adaptation(
            source,
            MarkedMeshAdaptation(hierarchy=hierarchy),
            policy=policy,
        )
        prepared = _prepared_simplex(adaptation)
    if budget.evidence is None:
        return prepared
    return PreparedAdaptiveSimplex(
        adaptation,
        prepared.layout,
        prepared.state,
        prepared.anchor,
        execution_evidence=NativeExecutionRecord(
            budget.evidence, owner_id=adaptation.prepared_id
        ),
    )


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
    """Host arrays of one process-addressable device epoch."""

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
    cell_classes: np.ndarray
    facet_classes: np.ndarray
    vertex_protected: np.ndarray
    protected_codes: np.ndarray


def _host_epoch(state: AdaptiveSimplexState, /) -> _HostEpoch:
    """Transfer one part into an owned, inspectable host snapshot."""

    host = jax.device_get(state)
    mesh = host.mesh
    return _HostEpoch(
        np.array(mesh.cell_ids, dtype=np.int64, copy=True),
        np.array(mesh.cell_active, dtype=np.bool_, copy=True),
        np.array(mesh.cells, dtype=np.int64, copy=True),
        np.array(host.tuples, dtype=np.int64, copy=True),
        np.array(host.tags, dtype=np.int64, copy=True),
        np.array(host.blocks, dtype=np.int64, copy=True),
        np.array(host.generations, dtype=np.int64, copy=True),
        np.array(host.parents, dtype=np.int64, copy=True),
        np.array(host.children, dtype=np.int64, copy=True),
        np.array(host.bisection_vertices, dtype=np.int64, copy=True),
        np.array(host.retired, dtype=np.bool_, copy=True),
        np.array(host.refine_rejected, dtype=np.bool_, copy=True),
        np.array(host.coarsen_marked, dtype=np.bool_, copy=True),
        np.array(mesh.vertex_ids, dtype=np.int64, copy=True),
        np.array(mesh.vertex_active, dtype=np.bool_, copy=True),
        # Exact widening: host predicates resolve the device coordinates.
        np.array(mesh.coordinates, dtype=np.float64, copy=True),
        np.array(host.vertex_parents, dtype=np.int64, copy=True),
        np.array(host.vertex_levels, dtype=np.int64, copy=True),
        np.array(host.vertex_removal, dtype=np.int64, copy=True),
        np.array(host.cursors, dtype=np.int64, copy=True),
        np.array(host.counters, dtype=np.int64, copy=True),
        int(np.asarray(host.status_flags)),
        np.array(host.cell_classes, dtype=np.int32, copy=True),
        np.array(host.facet_classes, dtype=np.int32, copy=True),
        np.array(host.vertex_protected, dtype=np.bool_, copy=True),
        np.array(host.protected_codes, dtype=np.int64, copy=True),
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


def _supports(anchor: _Anchor, host: _HostEpoch, /) -> tuple[np.ndarray, np.ndarray]:
    """Surviving vertices and reference weights spanning each removed source vertex."""

    count = anchor.source.vertex_ids.size
    removal = np.asarray(host.vertex_removal, dtype=np.int64)[:count]
    parents = np.asarray(host.vertex_parents, dtype=np.int64)[:count]
    steps = []
    for index in np.unique(removal[removal >= 0]):
        vertices = np.flatnonzero(removal == index)
        steps.append((vertices, parents[vertices, 0], parents[vertices, 1]))
    return _reverse_supports(steps, count, anchor.source.dimension)


class _Committed(NamedTuple):
    edit: CellTopologyEdit
    hierarchy: BisectionHierarchy
    evidence: BisectionEvidence
    refined: bool
    coarsened: bool
    partial: bool
    pass_limited: bool
    host_execution: NativeExecutionRecord | None


def _owned_retired_entities(
    source: _Source,
    keys: tuple[np.ndarray, ...],
    identifiers: tuple[np.ndarray, ...],
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Retain original issued IDs for entities whose actual vertices are local."""
    result = []
    for table, ids in zip(keys, identifiers, strict=True):
        owned = np.all(np.isin(table, source.vertex_ids), axis=1)
        result.append((np.searchsorted(source.vertex_ids, table[owned]), ids[owned]))
    return tuple(result)


def _owned_uniform_refinement(
    lineage: BisectionUniformRefinement | None,
    owned_ids: np.ndarray,
    records: _Records,
    /,
) -> BisectionUniformRefinement | None:
    if lineage is None:
        return None
    _uniform_charge(
        records.parent_ids.size + owned_ids.size,
        192 * (records.parent_ids.size + owned_ids.size),
    )
    branches = {
        int(parent): tuple(int(child) for child in children)
        for parent, children in zip(records.parent_ids, records.child_ids, strict=True)
    }
    owned = set(owned_ids.tolist())
    selected = []
    for root, children in enumerate(np.asarray(lineage.child_ids)):
        leaves: set[int] = set()
        parent = int(np.asarray(lineage.parent_ids)[root])
        pending = (
            [parent]
            if parent in owned or parent in branches
            else [int(child) for child in children]
        )
        visited: set[int] = set()
        while pending:
            _uniform_charge(1, 192)
            identifier = pending.pop()
            if identifier in visited:
                raise ValueError(
                    "Uniform ownership contains a repeated binary descendant identity."
                )
            visited.add(identifier)
            descendants = branches.get(identifier)
            if descendants is None:
                leaves.add(identifier)
            else:
                pending.extend(descendants)
        overlap = leaves & owned
        if overlap and overlap != leaves:
            raise MeshingFailure(
                MeshingFailureCategory.LINEAGE_FAILED,
                "A distributed part does not own the complete actual uniform source sibling tree.",
                stage="uniform-source-ownership",
                entity_ids=(int(np.asarray(lineage.parent_ids)[root]),),
            )
        if overlap:
            selected.append(root)
    return lineage.select(np.asarray(selected, dtype=np.int64))


def _committed_uniform_inverse(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    cells: _Cells,
    coarsening: _Coarsening,
    /,
) -> tuple[_Cells, _Coarsening, NativeExecutionRecord | None]:
    if prepared.anchor.start.uniform_refinement is None:
        return cells, coarsening, None
    baseline = np.asarray(jax.device_get(prepared.state.counters), dtype=np.int64)
    final = np.asarray(host.counters, dtype=np.int64)
    if baseline.shape != final.shape or np.any(baseline < 0) or np.any(final < baseline):
        raise ValueError(
            "Device commit counters must retain a nonnegative exact prepared-source baseline."
        )
    split_work = int(
        final[AdaptiveSimplexCounter.BISECTIONS]
        - baseline[AdaptiveSimplexCounter.BISECTIONS]
    )
    limits = prepared.adaptation.policy.limits
    if split_work > limits.maximum_work_units:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Device splits exhausted the original transaction work allowance before its uniform inverse.",
            stage="uniform-source-commit",
        )
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance) as budget:
        cells, coarsening = _execute_uniform_commit(prepared, host, cells, coarsening)
    record = (
        None
        if budget.evidence is None
        else NativeExecutionRecord(
            budget.evidence,
            owner_id=prepared.adaptation.prepared_id,
        )
    )
    return cells, coarsening, record


def _execute_uniform_commit(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    cells: _Cells,
    coarsening: _Coarsening,
    /,
) -> tuple[_Cells, _Coarsening]:
    from ._bisection import _joined, _Request, _select

    source, start = prepared.anchor.source, prepared.anchor.start
    uniform = start.uniform_refinement
    if uniform is None:
        return cells, coarsening
    count = int(host.cursors[1])
    active = np.flatnonzero(host.cell_active[:count])
    _uniform_charge(count, count * 32)
    marks = np.array(host.coarsen_marked[:count], dtype=np.bool_, copy=True)
    children = np.asarray(host.children[:count], dtype=np.int64)
    retired = np.asarray(host.retired[:count], dtype=np.bool_)
    for generation in np.unique(np.asarray(host.generations[:count]))[::-1]:
        parents = np.flatnonzero(
            (np.asarray(host.generations[:count]) == generation)
            & np.all(children >= 0, axis=1)
        )
        _uniform_charge(parents.size, parents.nbytes)
        pair = children[parents]
        marks[parents] |= np.all(marks[pair] & retired[pair], axis=1)
    requested = np.asarray(host.cell_ids[:count])[
        marks & np.asarray(host.cell_active[:count])
    ]
    if requested.size == 0:
        return cells, coarsening
    width = source.dimension + 1
    columns = np.asarray(
        [
            [index for index in range(width) if index != opposite]
            for opposite in range(width)
        ],
        dtype=np.int64,
    )
    _uniform_charge(
        cells.rows.size,
        4 * cells.ids.size * width * source.dimension * 8
        + 2 * cells.ids.size * width * 8,
    )
    keys = np.sort(cells.rows[:, columns], axis=2).reshape((-1, source.dimension))
    encoded = np.asarray(host.facet_classes)[active].reshape(-1)
    protected = np.asarray(
        host.vertex_protected[: source.vertex_ids.size], dtype=np.bool_
    )
    codes = np.asarray(host.protected_codes, dtype=np.int64)
    unique, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    if np.any(encoded != encoded[first][inverse]):
        raise ValueError(
            "Incident device facets disagree on their actual organization class."
        )
    facet_classes = encoded[first] - 1
    request = _Request(
        np.zeros((0,), dtype=np.int64),
        requested,
        codes,
        protected,
        np.zeros(source.cells.ids.shape, dtype=np.int64),
        unique,
        facet_classes,
        prepared.adaptation.policy.maximum_closure_iterations,
    )
    flags = _Flags(
        marks[active] | np.isin(cells.ids, coarsening.restored.ids),
        np.zeros(cells.ids.shape, dtype=np.bool_),
        np.asarray(host.cell_classes)[active],
    )
    updated = _uniform_inverse(
        source, start, request, cells, flags, (unique, facet_classes), coarsening
    )
    original_parents = np.setdiff1d(updated.undone, coarsening.undone)
    if original_parents.size:
        selected = np.searchsorted(np.asarray(uniform.parent_ids), original_parents)
        removed = np.asarray(uniform.child_ids)[selected].reshape(-1)
        cells = _joined(
            _select(cells, ~np.isin(cells.ids, removed)),
            _select(updated.restored, np.isin(updated.restored.ids, original_parents)),
        )
        cells = _select(cells, np.argsort(cells.ids, kind="stable"))
    return cells, updated


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
    if anchor.block_indices is not None:
        mapping = np.asarray(anchor.block_indices, dtype=np.int64)
        if mapping.ndim != 1:
            raise ValueError(
                "Source block projection requires one actual raw presentation axis."
            )
        _uniform_charge(cell_count + mapping.size, 32 * cell_count + 16 * mapping.size)
        used = active | ~retired
        source_blocks = blocks[used]
        if np.any((source_blocks < 0) | (source_blocks >= mapping.size)):
            raise ValueError(
                "An actual raw cell or binary record has an undeclared source presentation block."
            )
        projected = mapping[source_blocks]
        if np.any((projected < 0) | (projected >= len(source.mesh.blocks))):
            raise ValueError(
                "An actual raw cell or binary record lacks its resident scientific source block."
            )
        # The source owner binds this map from real SCI block incidence. Only
        # the host commit representation changes; compiled blocks stay intact.
        blocks = np.full(blocks.shape, -1, dtype=np.int64)
        blocks[used] = projected
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
        *_supports(anchor, host),
        int(counters[AdaptiveSimplexCounter.COARSENING_PASSES]),
        rejected_coarsenings,
    )
    # The compiled binary summary remains unchanged. Compatibility siblings
    # have one additional real inverse at this documented host commit boundary.
    cells, coarsening, host_execution = _committed_uniform_inverse(
        prepared, host, cells, coarsening
    )
    rejected_coarsenings = coarsening.rejected_ids
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
    request = prepared.adaptation.request
    prior = request.hierarchy if isinstance(request, MarkedMeshAdaptation) else None
    if prior is not None and not isinstance(prior, BisectionHierarchy):
        raise TypeError(
            "Device simplex commit requires its canonical scientific hierarchy."
        )
    hierarchy = _target_hierarchy(
        source, start, cells, records, retired_tables, front, prior=prior
    )
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
        coarsened_vertices=coarsening.removed_vertices.size,
        coarsening_passes=coarsening.passes,
        restored_cells=coarsening.restored.ids.size,
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
        host_execution,
    )


def _outcome(prepared: PreparedAdaptiveSimplex, host: _HostEpoch, /) -> Any:
    allowance = _phase_allowance(prepared.adaptation, prepared.execution_evidence)
    with _uniform_execution(allowance) as budget:
        _retain_preparation(prepared)
        baseline = np.asarray(jax.device_get(prepared.state.counters), dtype=np.int64)
        if (
            baseline.shape != host.counters.shape
            or np.any(baseline < 0)
            or np.any(host.counters < baseline)
        ):
            raise ValueError(
                "Device commit counters lost their exact prepared-source baseline."
            )
        budget.charge(
            work=int(
                host.counters[AdaptiveSimplexCounter.BISECTIONS]
                - baseline[AdaptiveSimplexCounter.BISECTIONS]
            )
        )
        outcome = _run_outcome(prepared, host)
    if budget.evidence is None:
        return outcome
    target = outcome.target.with_execution_evidence(
        NativeExecutionRecord(
            budget.evidence,
            preparation_evidence=prepared.execution_evidence,
            owner_id=prepared.adaptation.prepared_id,
        ),
    )
    return outcome._replace(target=target)


def _run_outcome(prepared: PreparedAdaptiveSimplex, host: _HostEpoch, /) -> _RouteOutcome:
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
        adaptation,
        committed.edit,
        kind,
        conservative=not committed.coarsened,
        bisection_hierarchy=committed.hierarchy,
    )
    from ._bisection import _rebind_bisection_presentation_blocks

    hierarchy = _rebind_bisection_presentation_blocks(committed.hierarchy, native.target)
    if committed.host_execution is not None:
        native = native._replace(
            target=native.target.with_execution_evidence(committed.host_execution)
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
        hierarchy,
    )


def _checked_state(
    prepared: PreparedAdaptiveSimplex, state: AdaptiveSimplexState, /
) -> None:
    if not isinstance(prepared, PreparedAdaptiveSimplex):
        raise TypeError("prepared must be PreparedAdaptiveSimplex.")
    prepared.adaptation.source.mesh.require_dense("Serial adaptive simplex commit")
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
    limits = prepared.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance) as budget:
        outcome = _run_device_bisection_route(prepared)
    if budget.evidence is None:
        return outcome
    target = outcome.target.with_execution_evidence(
        NativeExecutionRecord(budget.evidence, owner_id=prepared.prepared_id),
    )
    return outcome._replace(target=target)


def _run_device_bisection_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:

    request = prepared.request
    # ty: ignore[unresolved-attribute]
    if request.coarsen_cell_ids.size and request.hierarchy is None:
        raise ValueError(
            "Coarsening requires the BisectionHierarchy of a previous bisection."
        )
    simplex = _prepared_simplex(prepared)
    layout = simplex.layout
    refined = refine_adaptive_simplex(
        layout,
        simplex.state,
        # ty: ignore[unresolved-attribute]
        simplex.cell_marks(np.asarray(request.refine_cell_ids)),
    )
    require_applied_status(int(refined.report.status), DeviceEpoch.BISECTION)
    coarsened = coarsen_adaptive_simplex(
        layout,
        refined.state,
        # ty: ignore[unresolved-attribute]
        simplex.cell_marks(np.asarray(request.coarsen_cell_ids)),
    )
    return _outcome(simplex, _host_epoch(coarsened.state))


def restore_partitioned_mesh_state(
    storage: CellMeshStorage,
    evidence: CollectiveMeshEvidence,
    /,
) -> tuple[AdaptiveSimplexLayout, AdaptiveSimplexState, jax.Array]:
    """Restore and recheck producer-owned raw state before accepted publication.

    The input is a storage/proof pair, not a successful result. Domain and
    source certification producers can consume this numerical premise without
    a circular dependency on the result they are preparing.
    """
    if not isinstance(storage, CellMeshStorage) or not isinstance(
        evidence, CollectiveMeshEvidence
    ):
        raise TypeError(
            "Raw state restoration requires canonical storage and its consumed proof."
        )
    if (
        evidence.topology_id != storage.logical_topology_id
        or evidence.geometry_id != storage.logical_geometry_id
        or evidence.evidence_id != storage.evidence_id
        or evidence.partition_count != storage.partition_count
    ):
        raise ValueError(
            "Retained raw state proof differs from its immutable storage binding."
        )
    values = dict(storage.logical_arrays)
    mesh_names = (
        "coordinates",
        "vertex_ids",
        "vertex_active",
        "cells",
        "cell_ids",
        "cell_active",
        "facet_neighbors",
    )
    state_names = (
        "vertex_half_facets",
        "tuples",
        "tags",
        "blocks",
        "generations",
        "parents",
        "children",
        "bisection_vertices",
        "retired",
        "cell_classes",
        "facet_classes",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
        "protected_codes",
        "refine_rejected",
        "coarsen_marked",
        "cursors",
        "clocks",
        "counters",
    )
    names = (*mesh_names, *state_names, "source_exterior", "layout_parameters")
    missing = tuple(name for name in names if f"epoch/{name}" not in values)
    if missing:
        raise ValueError(
            f"Accepted mesh lacks retained numerical epoch arrays: {missing}."
        )
    arrays = {name: values[f"epoch/{name}"] for name in names}
    if any(value.shape[0] != storage.partition_count for value in arrays.values()):
        raise ValueError(
            "Retained epoch arrays disagree with accepted partition coverage."
        )
    accepted = dict(evidence.logical_arrays)
    retained = {f"epoch/{name}": arrays[name] for name in names}
    if any(name not in accepted for name in retained) or (
        logical_array_value_collection_digest(retained)
        != logical_array_value_collection_digest(
            {name: accepted[name] for name in retained}
        )
    ):
        raise ValueError(
            "Retained numerical forest differs from its accepted collective witness."
        )

    def restore(part: dict[str, jax.Array]) -> AdaptiveSimplexState:
        mesh = MaskedSimplexMesh(*(part[name] for name in mesh_names))
        return AdaptiveSimplexState(mesh, **{name: part[name] for name in state_names})

    state = eqx.filter_vmap(restore)(arrays)
    width = arrays["cells"].shape[-1]
    parameters = arrays["layout_parameters"]
    if parameters.shape != (storage.partition_count, 2) or parameters.dtype != jnp.int64:
        raise ValueError("Retained layout parameters must be int64 partition-by-two.")
    bounds = np.asarray(jax.device_get(jnp.max(parameters, axis=0)), dtype=np.int64)
    if not bool(jax.device_get(jnp.all(parameters == jnp.asarray(bounds)))):
        raise ValueError("Retained parts disagree on their declared iteration controls.")
    layout = AdaptiveSimplexLayout(
        width - 1,
        arrays["coordinates"].shape[-1],
        vertex_capacity=arrays["vertex_ids"].shape[1],
        cell_capacity=arrays["cell_ids"].shape[1],
        protected_edge_capacity=arrays["protected_codes"].shape[1],
        maximum_closure_iterations=int(bounds[0]),
        maximum_coarsening_passes=int(bounds[1]),
    )
    from ._distribution import replay_logical_bisection_checks

    checks = replay_logical_bisection_checks(
        evidence.compiled_states,
        evidence.initial_states,
        evidence.source_exterior,
        axis_name=evidence.axis_name,
        neighbor_pairs=evidence.neighbor_pairs,
    )
    if not bool(jax.device_get(jnp.all(checks))) or not bool(
        jax.device_get(jnp.array_equal(checks, evidence.partition_checks))
    ):
        raise ValueError(
            "Restored numerical forest does not reproduce its actual raw certificates."
        )
    return layout, state, arrays["source_exterior"]


def _retained_partition_states(
    source: CellMeshingResult, /
) -> tuple[AdaptiveSimplexLayout, AdaptiveSimplexState, jax.Array]:
    """Consume accepted identity, then reuse the independently validated raw getter."""
    storage = source.mesh.storage
    evidence = source.collective_evidence
    if storage is None or not isinstance(evidence, CollectiveMeshEvidence):
        raise ValueError(
            "Retained epoch restoration requires accepted owner-local storage."
        )
    if evidence.mesh_id != source.mesh.mesh_id:
        raise ValueError(
            "Retained source identity differs from its accepted collective proof."
        )
    evidence.require_passed()
    return restore_partitioned_mesh_state(storage, evidence)


def validate_partitioned_mesh_evidence(
    storage: CellMeshStorage,
    evidence: CollectiveMeshEvidence,
    /,
) -> None:
    """Replay mathematical and native source premises on the current hardware."""
    limits = evidence.preparation.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance):
        workspace = current_native_host_workspace()
        if workspace is None:
            raise RuntimeError(
                "Collective source replay requires its original bounded workspace."
            )
        workspace.retain_owner((storage, evidence))
        _run_validate_partitioned_mesh_evidence(storage, evidence)


def _run_validate_partitioned_mesh_evidence(
    storage: CellMeshStorage,
    evidence: CollectiveMeshEvidence,
    /,
) -> None:
    layout, states, _ = restore_partitioned_mesh_state(storage, evidence)
    if layout.signature_id != evidence.layout.signature_id:
        raise ValueError(
            "Restored collective layout differs from its actual numerical declaration."
        )
    _validate_restored_initial_source(evidence)
    renewed = CollectiveMeshEvidence(evidence, evidence.compiled_states)
    if (
        renewed.evidence_id != evidence.evidence_id
        or renewed.source_evidence_id != evidence.source_evidence_id
        or renewed.global_organization_id != evidence.global_organization_id
        or renewed.coordinate_geometry_id != evidence.coordinate_geometry_id
        or renewed.mesh_id != evidence.mesh_id
        or renewed.global_entity_counts != evidence.global_entity_counts
        or logical_array_value_collection_digest(dict(renewed.logical_arrays))
        != logical_array_value_collection_digest(dict(evidence.logical_arrays))
    ):
        raise ValueError(
            "Restored collective theorem differs from its independently replayed source mathematics."
        )


def _canonical_retained_coordinates(
    storage: CellMeshStorage, state: AdaptiveSimplexState, /
) -> AdaptiveSimplexState:
    arrays = dict(storage.logical_arrays)
    count = storage.global_entity_counts[0]
    identifiers = arrays["vertex_global_ids"]
    table = jnp.where(
        jnp.arange(identifiers.shape[0]) < count,
        identifiers,
        jnp.iinfo(jnp.int64).max,
    )
    rows = jnp.searchsorted(table, state.mesh.vertex_ids)
    safe = jnp.minimum(rows, table.shape[0] - 1)
    found = (rows < table.shape[0]) & (table[safe] == state.mesh.vertex_ids)
    coordinates = arrays["coordinates"][safe]
    return eqx.tree_at(
        lambda value: value.mesh.coordinates,
        state,
        jnp.where(
            (state.mesh.vertex_active & found)[..., None],
            coordinates,
            state.mesh.coordinates,
        ),
    )


def _validate_restored_initial_source(evidence: CollectiveMeshEvidence, /) -> None:
    """Bind the saved producer input to authored roots or an actual prior epoch."""
    source = evidence.source
    initial = evidence.initial_states
    prepared = evidence.preparation
    if prepared.source.result_id != source.result_id:
        raise ValueError("Restored preparation differs from its actual accepted source.")
    if "placement/requested_target_owners" in dict(evidence.logical_arrays):
        from ._restart_distribution import validate_simplex_restart_repack

        receipt = _restart_receipt_from_evidence(evidence)
        validate_simplex_restart_repack(receipt, receipt.requested_target_owners)
        return
    from ._initial_certification import InitialCollectiveMeshEvidence

    if source.mesh.storage is None:
        simplex = _prepared_simplex(prepared)
        if tuple(block.block_id for block in evidence.raw_source_blocks) != tuple(
            block.block_id for block in source.mesh.blocks
        ):
            raise ValueError(
                "Restored raw banks differ from their actual whole-source declaration."
            )
        origins = jnp.where(initial.mesh.cell_active, initial.mesh.cell_ids, -1)
        _validate_initial_partition_source(
            simplex,
            initial.mesh.cell_ids.shape[0],
            initial,
            evidence.source_exterior,
            _source_entity_tables(prepared),
            origins,
            layout=evidence.layout,
        )
        return
    if isinstance(source.collective_evidence, InitialCollectiveMeshEvidence):
        _validate_initial_source_roots(
            source.collective_evidence, initial, evidence.source_exterior
        )
        if tuple(block.block_id for block in evidence.raw_source_blocks) != tuple(
            block.block_id for block in source.mesh.blocks
        ):
            raise ValueError(
                "Restored initial raw banks differ from their actual native source declaration."
            )
        return
    old_layout, retained, _ = _retained_partition_states(source)
    compatible = _prepare_restored_original_roots(prepared, old_layout, retained)
    if compatible is not None:
        if compatible.layout.signature_id != evidence.layout.signature_id:
            raise ValueError(
                "Restored compatible root preparation differs from its original bounded layout."
            )
        expected = compatible.states
        if logical_array_value_collection_digest(
            {"exterior": compatible.source_exterior}
        ) != logical_array_value_collection_digest(
            {"exterior": evidence.source_exterior}
        ):
            raise ValueError(
                "Restored compatible roots changed their original physical-boundary law."
            )
        declarations = compatible.prepared.anchor.source_blocks
        if declarations is None or tuple(
            block.block_id for block in declarations
        ) != tuple(block.block_id for block in evidence.raw_source_blocks):
            raise ValueError(
                "Restored preparation changed its actual numeric source bank declaration."
            )
        owners = dict(evidence.logical_arrays).get("placement/initial_solver_cell_owners")
        if owners is None or logical_array_value_collection_digest(
            {"owners": owners}
        ) != logical_array_value_collection_digest(
            {"owners": compatible.solver_cell_owners}
        ):
            raise ValueError(
                "Restored preparation changed actual minimum source-leaf solver ownership."
            )
    else:
        if tuple(block.block_id for block in evidence.raw_source_blocks) != tuple(
            block.block_id for block in _raw_source_blocks(source)
        ):
            raise ValueError(
                "Restored raw banks differ from their actual retained numeric declaration."
            )
        expected = _resized_epoch(retained, evidence.layout)
        protected, codes, capacity = _retained_protection(prepared, expected)
        if capacity != evidence.layout.protected_edge_capacity:
            raise ValueError(
                "Restored protection does not match the declared source layout."
            )
        expected = eqx.tree_at(
            lambda value: (value.vertex_protected, value.protected_codes),
            expected,
            (protected, codes),
        )
        source_storage = source.mesh.storage
        if source_storage is None:
            raise ValueError("Retained source lost its accepted logical storage.")
        expected = _canonical_retained_coordinates(source_storage, expected)
    actual = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_flatten_with_path(initial)[0]
    }
    reference = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_flatten_with_path(expected)[0]
    }
    if logical_array_value_collection_digest(
        actual
    ) != logical_array_value_collection_digest(reference):
        raise ValueError(
            "Restored initial forest differs from its immutable accepted predecessor."
        )


def _validate_initial_source_roots(
    evidence: InitialCollectiveMeshEvidence,
    initial: AdaptiveSimplexState,
    exterior: jax.Array,
    /,
) -> None:
    """Consume original native coordinates, cells, owners and physical facets."""
    evidence.require_current()
    arrays = dict(evidence.logical_arrays)
    mesh = initial.mesh
    count = mesh.cell_ids.shape[0]
    table = jnp.where(
        evidence.entity_ids[-1] >= 0, evidence.entity_ids[-1], jnp.iinfo(jnp.int64).max
    )
    rows = jnp.minimum(jnp.searchsorted(table, mesh.cell_ids), table.shape[0] - 1)
    corners = jax.vmap(lambda part: part.mesh.vertex_ids[part.mesh.cells])(initial)
    source_corners = arrays["cell_vertices"][rows]
    source_owners = evidence.entity_owners[-1][rows]
    rank = jnp.arange(count, dtype=jnp.int32)[:, None]
    valid = jnp.all(
        ~mesh.cell_active
        | (
            (table[rows] == mesh.cell_ids)
            & (source_owners == rank)
            & jnp.all(corners == source_corners, axis=-1)
        )
    )
    valid &= jnp.sum(mesh.cell_active) == evidence.global_entity_counts[-1]
    vertex_table = jnp.where(
        evidence.entity_ids[0] >= 0, evidence.entity_ids[0], jnp.iinfo(jnp.int64).max
    )
    positions = jnp.minimum(
        jnp.searchsorted(vertex_table, mesh.vertex_ids), vertex_table.shape[0] - 1
    )
    valid &= jnp.all(
        ~mesh.vertex_active
        | (
            (vertex_table[positions] == mesh.vertex_ids)
            & jnp.all(mesh.coordinates == arrays["coordinates"][positions], axis=-1)
        )
    )
    allocated = mesh.cell_ids >= 0
    valid &= jnp.all(
        ~allocated
        | (
            mesh.cell_active
            & (initial.parents < 0)
            & jnp.all(initial.children < 0, axis=-1)
            & ~initial.retired
            & (initial.generations == 0)
        )
    )
    width = mesh.cells.shape[-1]
    columns = jnp.asarray(
        tuple(
            tuple(index for index in range(width) if index != opposite)
            for opposite in range(width)
        ),
        dtype=jnp.int32,
    )
    facets = jnp.sort(corners[:, :, columns], axis=-1)
    facet_keys = jnp.where(
        evidence.entity_keys[-2] >= 0, evidence.entity_keys[-2], jnp.iinfo(jnp.int64).max
    )
    facet_rows = _key_positions(facet_keys, facets.reshape((-1, width - 1))).reshape(
        mesh.cells.shape
    )
    physical = arrays["initial/edge_physical"][jnp.maximum(facet_rows, 0)]
    valid &= jnp.all(
        ~mesh.cell_active[..., None] | ((facet_rows >= 0) & (exterior == physical))
    )
    if not bool(jax.device_get(valid)):
        raise ValueError(
            "Initial raw roots differ from the actual independent native source theorem."
        )


@final
class PartitionedAdaptiveSimplex(StrictModule, NonTrainableState):
    """Part-sharded view of one prepared epoch: each part refines its owned cells.

    Ownership is the policy's `MeshDistribution` (space-filling-curve, graph,
    or provider ownership of the source cells); cells created by uniform
    subdivision follow their source cell. ``layout`` is the per-part capacity
    bucket, ``parts`` the device mesh, and ``states`` the stacked local states.
    Publication retains ownership and exposes its actual sparse transition.
    Repartitioning requires a separately prepared complete-state migration.
    """

    prepared: PreparedAdaptiveSimplex
    layout: AdaptiveSimplexLayout
    parts: AdaptiveSimplexParts
    states: AdaptiveSimplexState
    source_exterior: jax.Array
    source_entities: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ]
    slot_origins: np.ndarray | jax.Array
    partitioned_id: str = eqx.field(static=True)
    source_witness: AffineBisectionSourceWitness | None
    source_cache_id: str = eqx.field(static=True)
    initial_state_id: str = eqx.field(static=True)
    solver_cell_owners: jax.Array
    placement_arrays: tuple[tuple[str, jax.Array], ...]
    solver_neighborhood: SimplexNeighborhoodWorkset | None
    solver_forest: AdaptiveSimplexState | None
    restart_repack: SimplexRestartRepack | None
    restart_proof: SimplexRestartRepackProof | None
    execution_evidence: NativeExecutionRecord | None

    def __init__(
        self,
        prepared: PreparedAdaptiveSimplex,
        layout: AdaptiveSimplexLayout,
        parts: AdaptiveSimplexParts,
        states: AdaptiveSimplexState,
        source_exterior: jax.Array,
        slot_origins: np.ndarray | jax.Array,
        /,
        *,
        solver_cell_owners: jax.Array | None = None,
        placement_arrays: tuple[tuple[str, jax.Array], ...] = (),
        solver_neighborhood: SimplexNeighborhoodWorkset | None = None,
        solver_forest: AdaptiveSimplexState | None = None,
        restart_repack: SimplexRestartRepack | None = None,
        restart_proof: SimplexRestartRepackProof | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
        _validated_source: PartitionedAdaptiveSimplex | None = None,
    ) -> None:
        _require_preparation_receipt(prepared.adaptation, execution_evidence)
        if _validated_source is not None:
            if not isinstance(_validated_source, PartitionedAdaptiveSimplex):
                raise TypeError(
                    "Receipt rebinding requires the actual validated partitioned source."
                )
            supplied = (
                prepared,
                layout,
                parts,
                states,
                source_exterior,
                slot_origins,
                solver_cell_owners,
                placement_arrays,
                solver_neighborhood,
                solver_forest,
                restart_repack,
                restart_proof,
            )
            retained = (
                _validated_source.prepared,
                _validated_source.layout,
                _validated_source.parts,
                _validated_source.states,
                _validated_source.source_exterior,
                _validated_source.slot_origins,
                _validated_source.solver_cell_owners,
                _validated_source.placement_arrays,
                _validated_source.solver_neighborhood,
                _validated_source.solver_forest,
                _validated_source.restart_repack,
                _validated_source.restart_proof,
            )
            if any(
                left is not right for left, right in zip(supplied, retained, strict=True)
            ):
                raise ValueError(
                    "Receipt rebinding cannot alter any validated scientific input."
                )
            if isinstance(slot_origins, np.ndarray) and slot_origins.flags.writeable:
                raise ValueError(
                    "Receipt rebinding requires immutable source-origin ownership."
                )
            self.prepared = prepared
            self.layout = layout
            self.parts = parts
            self.states = states
            self.source_exterior = source_exterior
            self.slot_origins = slot_origins
            self.solver_cell_owners = _validated_source.solver_cell_owners
            self.placement_arrays = placement_arrays
            self.solver_neighborhood = solver_neighborhood
            self.solver_forest = solver_forest
            self.restart_repack = restart_repack
            self.restart_proof = restart_proof
            self.execution_evidence = execution_evidence
            self.source_entities = _validated_source.source_entities
            self.source_witness = _validated_source.source_witness
            self.source_cache_id = _validated_source.source_cache_id
            self.initial_state_id = _validated_source.initial_state_id
            self.partitioned_id = _validated_source.partitioned_id
            return
        entities = _source_entity_tables(prepared.adaptation)
        if solver_cell_owners is None:
            solver_cell_owners = _source_solver_cell_owners(prepared.adaptation, states)
        if (
            solver_cell_owners.shape != states.mesh.cell_ids.shape
            or solver_cell_owners.dtype != jnp.int32
        ):
            raise ValueError(
                "Solver ownership requires an exact int32 raw-forest cell axis."
            )
        if not bool(
            jax.device_get(
                jnp.all(
                    ~states.mesh.cell_active
                    | (
                        (solver_cell_owners >= 0)
                        & (solver_cell_owners < parts.part_count)
                    )
                )
            )
        ):
            raise ValueError(
                "A prepared active forest row lacks its actual solver owner."
            )
        accepted = prepared.adaptation.source.mesh.storage is not None
        supported = accepted or (
            prepared.anchor.start.uniform_refinement is not None
            or not prepared.anchor.start.front.growth.levels.size
        )
        if supported and not accepted:
            failure: MeshingFailure | None = None
            try:
                _validate_initial_partition_source(
                    prepared,
                    parts.part_count,
                    states,
                    source_exterior,
                    entities,
                    slot_origins,
                    layout=layout,
                )
            except (ValueError, IndexError) as error:
                failure = MeshingFailure(
                    MeshingFailureCategory.LINEAGE_FAILED,
                    str(error),
                    stage="distributed-source-preparation",
                )
            _require_collective_local_success(failure)
        cache_id = _source_cache_digest(entities)
        state_id = logical_array_value_collection_digest(
            {
                jax.tree_util.keystr(path): value
                for path, value in jax.tree_util.tree_flatten_with_path(states)[0]
            }
        )
        for value in jax.tree_util.tree_leaves(prepared.anchor):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        if isinstance(slot_origins, np.ndarray):
            slot_origins.setflags(write=False)
        for table in entities:
            for value in table:
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
        self.prepared = prepared
        self.execution_evidence = execution_evidence
        self.layout = layout
        self.parts = parts
        self.states = states
        self.source_exterior = source_exterior
        self.source_entities = entities
        self.slot_origins = slot_origins
        self.source_witness = (
            AffineBisectionSourceWitness(
                prepared.adaptation.source,
                uniform_refinement=prepared.anchor.start.uniform_refinement,
            )
            if supported
            and (
                prepared.adaptation.source.certification is not None
                or prepared.adaptation.source.collective_evidence is not None
            )
            else None
        )
        self.source_cache_id = cache_id
        self.initial_state_id = state_id
        self.solver_cell_owners = solver_cell_owners
        self.placement_arrays = placement_arrays
        self.solver_neighborhood = solver_neighborhood
        self.solver_forest = solver_forest
        self.restart_repack = restart_repack
        self.restart_proof = restart_proof
        self.partitioned_id = canonical_fingerprint(
            {
                "kind": "partitioned-adaptive-simplex",
                "prepared": prepared.prepared_id,
                "layout": layout.signature_id,
                # ty: ignore[unresolved-attribute]
                "distribution": prepared.adaptation.policy.distribution.distribution_id,
            }
        )

    def cell_marks(self, cell_ids: np.ndarray | jax.Array, /) -> jax.Array:
        """Per-part slot masks of the prepared cells descending from source cells."""

        identifiers = jnp.asarray(cell_ids, dtype=jnp.int64)
        origins = jnp.asarray(self.slot_origins, dtype=jnp.int64)
        return jnp.isin(origins, identifiers) & (origins >= 0)


def _resized_epoch(
    states: AdaptiveSimplexState, layout: AdaptiveSimplexLayout, /
) -> AdaptiveSimplexState:
    """Resize only padding while retaining every issued scientific identity."""
    vertex_fields = (
        "vertex_half_facets",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
    )
    cell_fields = (
        "tuples",
        "tags",
        "blocks",
        "generations",
        "parents",
        "children",
        "bisection_vertices",
        "retired",
        "cell_classes",
        "facet_classes",
    )

    def resize(part: AdaptiveSimplexState) -> AdaptiveSimplexState:
        def fit(value: jax.Array, count: int, fill: int | bool = 0) -> jax.Array:
            result = value[:count]
            return jnp.pad(
                result,
                (
                    (0, max(count - result.shape[0], 0)),
                    *((0, 0) for _ in result.shape[1:]),
                ),
                constant_values=fill,
            )

        v, c = layout.vertex_capacity, layout.cell_capacity
        mesh = MaskedSimplexMesh(
            fit(part.mesh.coordinates, v),
            fit(part.mesh.vertex_ids, v, -1),
            fit(part.mesh.vertex_active, v, False),
            fit(part.mesh.cells, c),
            fit(part.mesh.cell_ids, c, -1),
            fit(part.mesh.cell_active, c, False),
            fit(part.mesh.facet_neighbors, c, -1),
        )
        sentinel = jnp.iinfo(jnp.int64).max
        codes = part.protected_codes
        old = part.mesh.vertex_capacity
        codes = jnp.where(codes != sentinel, (codes // old) * v + codes % old, sentinel)
        fields = {
            name: fit(
                getattr(part, name),
                v,
                -1
                if name in ("vertex_half_facets", "vertex_parents", "vertex_removal")
                else 0,
            )
            for name in vertex_fields
        }
        fields.update(
            {
                name: fit(
                    getattr(part, name),
                    c,
                    -1 if name in ("parents", "children", "bisection_vertices") else 0,
                )
                for name in cell_fields
            }
        )
        return AdaptiveSimplexState(
            mesh,
            **fields,
            protected_codes=fit(codes, layout.protected_edge_capacity, sentinel),
            refine_rejected=jnp.zeros((c,), dtype=jnp.bool_),
            coarsen_marked=jnp.zeros((c,), dtype=jnp.bool_),
            cursors=part.cursors,
            clocks=part.clocks.at[2].set(0),
            counters=jnp.zeros_like(part.counters),
        )

    return eqx.filter_vmap(resize)(states)


def _retained_protection(
    prepared: PreparedMeshAdaptation, states: AdaptiveSimplexState, /
) -> tuple[jax.Array, jax.Array, int]:
    """Apply requested scientific scopes on logical IDs, preserving prior constraints."""
    evidence = prepared.source.collective_evidence
    storage = prepared.source.mesh.storage
    if storage is None:
        raise ValueError(
            "Retained protection requires actual owner-local numerical storage."
        )
    if evidence is None:
        raise ValueError("Retained protection requires actual collective source routing.")
    vertex_flags = states.vertex_protected
    parts, cells, width = states.mesh.cells.shape
    capacity = states.mesh.vertex_ids.shape[1]
    columns = jnp.asarray(tuple(combinations(range(width), 2)), dtype=jnp.int32)
    edge_slots = states.mesh.cells[:, :, columns].reshape((parts, -1, 2))
    edge_ids = jnp.take_along_axis(
        states.mesh.vertex_ids,
        edge_slots.reshape((parts, -1)),
        axis=1,
    ).reshape(edge_slots.shape)
    edge_flags = jnp.zeros(edge_slots.shape[:-1], dtype=jnp.bool_)
    arrays = dict(storage.logical_arrays)
    for scope in prepared.policy.protected_scopes:
        degree = scope.entity_dimension
        if (
            scope.source_id != prepared.source.mesh.mesh_id
            or scope.source_revision != prepared.source.mesh.numeric_version
            or scope.entity_set_id
            != prepared.source.mesh.entity_set(degree).entity_set_id
        ):
            raise ValueError(
                "Protected scopes must bind the exact accepted source epoch."
            )
        ids = evidence.entity_ids[degree][: evidence.global_entity_counts[degree]]
        order = jnp.argsort(ids, stable=True)
        positions = _key_positions(ids[order, None], scope.entity_ids[:, None])
        if bool(jax.device_get(jnp.any(positions < 0))):
            raise ValueError(
                "Protected scope selects absent accepted logical source entities."
            )
        rows = order[positions]
        keys = (
            arrays["cell_vertices"][rows]
            if degree == width - 1
            else evidence.entity_keys[degree][rows]
        )
        vertex_flags = vertex_flags | jnp.any(
            states.mesh.vertex_ids[:, :, None, None] == keys[None, None, :, :],
            axis=(-2, -1),
        )
        if degree:
            ends = jnp.any(
                edge_ids[:, :, :, None, None] == keys[None, None, None, :, :], axis=-1
            )
            edge_flags = edge_flags | jnp.any(jnp.all(ends, axis=2), axis=-1)
    sentinel = jnp.iinfo(jnp.int64).max
    low, high = jnp.min(edge_slots, axis=-1), jnp.max(edge_slots, axis=-1)
    codes = jnp.where(
        edge_flags & jnp.repeat(states.mesh.cell_active, columns.shape[0], axis=1),
        low.astype(jnp.int64) * capacity + high,
        sentinel,
    )
    codes = jnp.sort(jnp.concatenate((states.protected_codes, codes), axis=1), axis=1)
    unique = (codes != sentinel) & jnp.concatenate(
        (
            jnp.ones((parts, 1), dtype=jnp.bool_),
            codes[:, 1:] != codes[:, :-1],
        ),
        axis=1,
    )
    order = jnp.argsort(~unique, axis=1, stable=True)
    codes = jnp.take_along_axis(codes, order, axis=1)
    count = int(jax.device_get(jnp.max(jnp.sum(unique, axis=1, dtype=jnp.int64))))
    edges = adaptive_simplex_bucket(max(count, 1), 1.0)
    codes = codes[:, :edges]
    if codes.shape[1] < edges:
        codes = jnp.pad(
            codes, ((0, 0), (0, edges - codes.shape[1])), constant_values=sentinel
        )
    return vertex_flags, codes, edges


def prepare_partitioned_mesh_adaptation(
    prepared: PreparedMeshAdaptation, /
) -> PartitionedAdaptiveSimplex:
    if not isinstance(prepared, PreparedMeshAdaptation):
        raise TypeError("prepared must be PreparedMeshAdaptation.")
    limits = prepared.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance) as budget:
        partitioned = _prepare_partitioned_mesh_adaptation(prepared)
    if budget.evidence is None:
        return partitioned
    return _bind_partitioned_preparation_receipt(
        partitioned, NativeExecutionRecord(budget.evidence, owner_id=prepared.prepared_id)
    )


def _bind_partitioned_preparation_receipt(
    source: PartitionedAdaptiveSimplex,
    record: NativeExecutionRecord,
    /,
) -> PartitionedAdaptiveSimplex:
    return PartitionedAdaptiveSimplex(
        source.prepared,
        source.layout,
        source.parts,
        source.states,
        source.source_exterior,
        source.slot_origins,
        solver_cell_owners=source.solver_cell_owners,
        placement_arrays=source.placement_arrays,
        solver_neighborhood=source.solver_neighborhood,
        solver_forest=source.solver_forest,
        restart_repack=source.restart_repack,
        restart_proof=source.restart_proof,
        execution_evidence=record,
        _validated_source=source,
    )


def _prepare_partitioned_mesh_adaptation(
    prepared: PreparedMeshAdaptation, /
) -> PartitionedAdaptiveSimplex:
    """Prepare the canonical distributed route from dense or accepted logical state."""
    if not isinstance(prepared, PreparedMeshAdaptation):
        raise TypeError("prepared must be PreparedMeshAdaptation.")
    if (
        prepared.policy.route is not MeshAdaptationRoute.DEVICE_BISECTION
        or not isinstance(prepared.request, MarkedMeshAdaptation)
    ):
        raise ValueError(
            "Distributed simplex adaptation requires a marked DEVICE_BISECTION request."
        )
    distribution = prepared.policy.distribution
    device = prepared.policy.device_policy
    if distribution is None or device is None:
        raise ValueError(
            "Distributed simplex adaptation requires explicit distribution and device policies."
        )
    source = prepared.source
    if source.mesh.storage is None:
        return partition_adaptive_simplex(_prepared_simplex(prepared))
    from ._initial_certification import InitialCollectiveMeshEvidence

    if isinstance(source.collective_evidence, InitialCollectiveMeshEvidence):
        return _prepare_initial_partition(prepared)
    old_layout, retained, _ = _retained_partition_states(source)
    storage = source.mesh.storage
    if distribution.partition.part_count != storage.partition_count:
        raise ValueError(
            "Changed ownership requires completed scientific-ID forest migration before adaptation."
        )
    compatible = _prepare_restored_original_roots(prepared, old_layout, retained)
    if compatible is not None:
        return compatible
    allocated = np.asarray(
        jax.device_get(jnp.max(retained.cursors[:, :2], axis=0)), dtype=np.int64
    )
    vertices, cells = device.capacities(int(allocated[0]), int(allocated[1]))
    layout = AdaptiveSimplexLayout(
        old_layout.dimension,
        old_layout.ambient_dimension,
        vertex_capacity=vertices,
        cell_capacity=cells,
        protected_edge_capacity=old_layout.protected_edge_capacity,
        maximum_closure_iterations=prepared.policy.maximum_closure_iterations,
        maximum_coarsening_passes=device.maximum_coarsening_passes,
    )
    states = _resized_epoch(retained, layout)
    vertex_flags, protected_codes, edge_capacity = _retained_protection(prepared, states)
    if edge_capacity != layout.protected_edge_capacity:
        layout = AdaptiveSimplexLayout(
            layout.dimension,
            layout.ambient_dimension,
            vertex_capacity=layout.vertex_capacity,
            cell_capacity=layout.cell_capacity,
            protected_edge_capacity=edge_capacity,
            maximum_closure_iterations=layout.maximum_closure_iterations,
            maximum_coarsening_passes=layout.maximum_coarsening_passes,
        )
    states = eqx.tree_at(
        lambda state: (state.vertex_protected, state.protected_codes),
        states,
        (vertex_flags, protected_codes),
    )
    arrays = dict(storage.logical_arrays)
    states = _canonical_retained_coordinates(storage, states)

    def exterior(
        ids: jax.Array,
        active: jax.Array,
        packet_ids: jax.Array,
        packet_valid: jax.Array,
        flags: jax.Array,
    ) -> jax.Array:
        table = jnp.where(packet_valid, packet_ids, jnp.iinfo(jnp.int64).max)
        positions = jnp.searchsorted(table, ids)
        positions = jnp.minimum(positions, table.size - 1)
        return active[:, None] & flags[positions]

    source_exterior = jax.vmap(exterior)(
        states.mesh.cell_ids,
        states.mesh.cell_active,
        arrays["closure/cell_ids"],
        arrays["closure/cell_valid"],
        arrays["closure/cell_exterior"],
    )
    part_count = storage.partition_count
    chosen = tuple(jax.devices()[:part_count])
    if len(chosen) != part_count:
        raise ValueError(f"Retained ownership requires {part_count} available devices.")
    parts = AdaptiveSimplexParts(
        chosen,
        neighbor_pairs=tuple(
            (left, right)
            for left in range(part_count)
            for right in range(part_count)
            if left != right
        ),
    )
    simplex = _source_simplex_on_states(prepared, layout, states)
    origins = jnp.where(states.mesh.cell_active, states.mesh.cell_ids, -1)
    return PartitionedAdaptiveSimplex(
        simplex, layout, parts, states, source_exterior, origins
    )


def _prepare_restored_original_roots(
    prepared: PreparedMeshAdaptation,
    old_layout: AdaptiveSimplexLayout,
    retained: AdaptiveSimplexState,
    /,
) -> PartitionedAdaptiveSimplex | None:
    limits = prepared.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    with _uniform_execution(allowance):
        workspace = current_native_host_workspace()
        if workspace is None:
            raise RuntimeError(
                "Compatible original roots require their actual native source workspace."
            )
        workspace.retain_owner((prepared, retained))
        result = _run_prepare_restored_original_roots(prepared, old_layout, retained)
        if result is None:
            result = _run_prepare_restored_mixed_roots(prepared, old_layout, retained)
        if result is not None:
            workspace.retain_owner(result)
        return result


def _merge_prepared_root_part(
    old: _HostEpoch,
    seed: _HostEpoch,
    seed_slots: np.ndarray,
    restored: np.ndarray,
    lineage: BisectionUniformRefinement,
    remap: dict[int, int],
    protected: np.ndarray,
    codes: np.ndarray,
    old_capacity: int,
    seed_edges: np.ndarray,
    old_solver: np.ndarray,
    root_solver: dict[int, tuple[int, int]],
    /,
) -> tuple[_HostEpoch, np.ndarray, np.ndarray]:
    """Retain untouched actual families and replace only authenticated roots."""
    old_count = int(old.cursors[1])
    roots = np.arange(old_count, dtype=np.int64)
    for slot in range(old_count):
        parent = old.parents[slot] // 2
        if parent >= 0:
            roots[slot] = roots[parent]
    parents = {
        int(child): int(parent)
        for parent, children in zip(
            np.asarray(lineage.parent_ids), np.asarray(lineage.child_ids), strict=True
        )
        for child in children
    }
    original_roots = np.asarray(
        [
            parents.get(int(identifier), int(identifier))
            for identifier in old.cell_ids[roots]
        ]
    )
    kept = np.flatnonzero(~np.isin(original_roots, restored))
    components = ((old, kept, False), (seed, seed_slots, True))
    cell_ids = np.concatenate([part.cell_ids[slots] for part, slots, _ in components])
    if np.unique(cell_ids).size != cell_ids.size:
        raise ValueError(
            "Mixed root preparation collided with a retained scientific cell ID."
        )
    order = np.argsort(cell_ids, kind="stable")
    cell_ids = cell_ids[order]
    owners: dict[int, tuple[_HostEpoch, int, bool]] = {}
    vertex_flags: dict[int, bool] = {}

    def identifier(part: _HostEpoch, slot: int, incoming: bool) -> int:
        value = int(part.vertex_ids[slot])
        return remap.get(value, value) if incoming else value

    for part, slots, incoming in components:
        used = set(part.cells[slots].reshape(-1).tolist())
        used.update(part.tuples[slots].reshape(-1).tolist())
        used.update(
            part.bisection_vertices[slots][part.bisection_vertices[slots] >= 0].tolist()
        )
        pending = list(used)
        while pending:
            slot = pending.pop()
            for parent in part.vertex_parents[slot]:
                if parent >= 0 and int(parent) not in used:
                    used.add(int(parent))
                    pending.append(int(parent))
        for slot in sorted(used):
            key = identifier(part, slot, incoming)
            owners.setdefault(key, (part, slot, incoming))
            flag = part.vertex_protected[slot] if incoming else protected[slot]
            vertex_flags[key] = vertex_flags.get(key, False) or bool(flag)
    vertex_ids = np.asarray(sorted(owners), dtype=np.int64)
    vertex_count, count = vertex_ids.size, cell_ids.size
    values = {
        name: []
        for name in (
            "cells",
            "tuples",
            "tags",
            "blocks",
            "generations",
            "parents",
            "children",
            "bisection_vertices",
            "cell_active",
            "retired",
            "refine_rejected",
            "coarsen_marked",
            "cell_classes",
            "facet_classes",
        )
    }
    origins, solver = [], []
    for part, slots, incoming in components:
        for slot in slots:
            for name in values:
                value = getattr(part, name)[slot]
                if name in ("cells", "tuples"):
                    ids = np.asarray(
                        [identifier(part, int(index), incoming) for index in value]
                    )
                    value = np.searchsorted(vertex_ids, ids)
                elif name == "parents":
                    if value >= 0:
                        value = (
                            2 * np.searchsorted(cell_ids, part.cell_ids[value // 2])
                            + value % 2
                        )
                elif name == "children":
                    value = np.asarray(
                        [
                            np.searchsorted(cell_ids, part.cell_ids[index])
                            if index >= 0
                            else -1
                            for index in value
                        ],
                        dtype=np.int64,
                    )
                elif name == "bisection_vertices" and value >= 0:
                    value = np.searchsorted(
                        vertex_ids, identifier(part, int(value), incoming)
                    )
                values[name].append(value)
            origin = int(part.cell_ids[slot])
            if incoming:
                origin = int(root_solver[int(part.cell_ids[slot])][0])
                owner = int(root_solver[int(part.cell_ids[slot])][1])
            else:
                owner = int(old_solver[slot])
            origins.append(origin if part.cell_active[slot] else -1)
            solver.append(owner)
    fields = {
        name: np.asarray(value, dtype=getattr(old, name).dtype).reshape(
            (cell_ids.size, *getattr(old, name).shape[1:])
        )[order]
        for name, value in values.items()
    }
    coordinates, active, vertex_parents, levels, removal = [], [], [], [], []
    for key in vertex_ids:
        part, slot, incoming = owners[int(key)]
        coordinates.append(part.coordinates[slot])
        active.append(part.vertex_active[slot])
        vertex_parents.append(
            [
                np.searchsorted(vertex_ids, identifier(part, int(parent), incoming))
                if parent >= 0
                else -1
                for parent in part.vertex_parents[slot]
            ]
        )
        levels.append(part.vertex_levels[slot])
        removal.append(part.vertex_removal[slot])
    actual_codes = codes[codes != np.iinfo(np.int64).max]
    ends = np.stack((actual_codes // old_capacity, actual_codes % old_capacity), axis=1)
    edges = np.concatenate((old.vertex_ids[ends], seed_edges))
    edges = edges[np.all(np.isin(edges, vertex_ids), axis=1)]
    compact = np.searchsorted(vertex_ids, edges)
    protected_codes = np.unique(compact[:, 0] * vertex_count + compact[:, 1])
    cursors = np.asarray(
        (
            vertex_count,
            count,
            max(int(old.cursors[2]), int(seed.cursors[2])),
            max(int(old.cursors[3]), int(seed.cursors[3])),
        ),
        dtype=np.int64,
    )
    base = _HostEpoch(
        cell_ids,
        fields["cell_active"],
        fields["cells"],
        fields["tuples"],
        fields["tags"],
        fields["blocks"],
        fields["generations"],
        fields["parents"],
        fields["children"],
        fields["bisection_vertices"],
        fields["retired"],
        fields["refine_rejected"],
        fields["coarsen_marked"],
        vertex_ids,
        np.asarray(active),
        np.asarray(coordinates, dtype=np.float64).reshape(
            (vertex_count, old.coordinates.shape[1])
        ),
        np.asarray(vertex_parents, dtype=np.int64).reshape((vertex_count, 2)),
        np.asarray(levels),
        np.asarray(removal),
        cursors,
        old.counters,
        old.flags,
        fields["cell_classes"],
        fields["facet_classes"],
        np.asarray([vertex_flags[int(key)] for key in vertex_ids]),
        protected_codes,
    )
    return (
        base,
        np.asarray(origins, dtype=np.int64)[order],
        np.asarray(solver, dtype=np.int32)[order],
    )


def _run_prepare_restored_mixed_roots(
    prepared: PreparedMeshAdaptation,
    old_layout: AdaptiveSimplexLayout,
    retained: AdaptiveSimplexState,
    /,
) -> PartitionedAdaptiveSimplex | None:
    """Reconcile a real restored cohort with untouched live uniform families."""
    from ._bisection import _reconcile_uniform_refinement, _select

    if not isinstance(prepared.request, MarkedMeshAdaptation):
        raise TypeError("Distributed simplex preparation requires its marked request.")
    hierarchy = prepared.request.hierarchy
    if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError(
            "Distributed simplex preparation requires its canonical bisection hierarchy."
        )
    evidence = prepared.source.collective_evidence
    lineage = (
        evidence.uniform_refinement
        if isinstance(evidence, CollectiveMeshEvidence)
        else None
    )
    if hierarchy is None or lineage is None:
        return None
    scientific = lineage.source
    require_original_meshing_source(prepared.source)
    source = _prepared_source(scientific.mesh)
    storage = prepared.source.mesh.storage
    if storage is None:
        return None
    ids = np.asarray(jax.device_get(dict(storage.logical_arrays)["cell_global_ids"]))[
        : storage.global_entity_counts[-1]
    ]
    restored = np.intersect1d(ids, np.asarray(lineage.parent_ids))
    if restored.size == 0:
        return None
    # Uniform compatibility is a cohort law: when any restored root is refined,
    # regenerate every restored sibling family needed for conforming interfaces.
    expanded = restored
    positions = np.searchsorted(source.cells.ids, restored)
    if np.any(positions >= source.cells.ids.size) or not np.array_equal(
        source.cells.ids[positions], restored
    ):
        raise ValueError(
            "Restored mixed roots lack their original scientific cell authority."
        )
    bank = lineage.host_arrays()
    parent_rows = np.searchsorted(bank["parent_ids"], restored)
    if not np.array_equal(
        source.vertex_ids[source.cells.rows[positions]], bank["parent_rows"][parent_rows]
    ):
        raise ValueError(
            "Restored mixed roots changed their original source corner columns."
        )
    flags, codes, _ = _retained_protection(prepared, retained)
    solver = np.asarray(jax.device_get(_source_solver_cell_owners(prepared, retained)))
    leaves = jax.tree_util.tree_leaves(retained)
    _uniform_charge(
        sum(value.size for value in leaves if isinstance(value, jax.Array)),
        sum(value.nbytes for value in leaves if isinstance(value, jax.Array)),
    )
    hosts = tuple(
        _host_epoch(_addressable_part_state(retained, part))
        for part in range(storage.partition_count)
    )
    restored_tuples = np.empty((restored.size, source.dimension + 1), dtype=np.int64)
    restored_tags = np.empty(restored.size, dtype=np.int64)
    restored_levels = np.empty(restored.size, dtype=np.int64)
    restored_facets = np.empty((restored.size, source.dimension + 1), dtype=np.int32)
    root_owner, root_solver = {}, {}
    protected_vertices, protected_edges = [], []
    for part, host in enumerate(hosts):
        active = np.flatnonzero(host.cell_active)
        for slot in active[np.isin(host.cell_ids[active], restored)]:
            key = int(host.cell_ids[slot])
            row = np.searchsorted(restored, key)
            if key in root_owner or host.parents[slot] >= 0:
                raise ValueError(
                    "Restored original roots have conflicting actual source ownership."
                )
            if (
                host.generations[slot] != bank["parent_levels"][parent_rows[row]]
                or host.cell_classes[slot] != bank["parent_classes"][parent_rows[row]]
            ):
                raise ValueError(
                    "Restored roots changed their actual original level or class."
                )
            if not np.array_equal(
                host.vertex_ids[host.cells[slot]], bank["parent_rows"][parent_rows[row]]
            ):
                raise ValueError(
                    "Restored raw root incidence differs from original SCI authority."
                )
            root_owner[key], root_solver[key] = part, int(solver[part, slot])
            restored_tuples[row] = host.vertex_ids[host.tuples[slot]]
            restored_tags[row] = host.tags[slot]
            restored_levels[row] = host.generations[slot]
            restored_facets[row] = host.facet_classes[slot]
        actual_flags = np.asarray(jax.device_get(flags[part])) & host.vertex_active
        protected_vertices.extend(
            host.vertex_ids[
                actual_flags & np.isin(host.vertex_ids, source.vertex_ids)
            ].tolist()
        )
        actual = np.asarray(jax.device_get(codes[part]))
        actual = actual[actual != np.iinfo(np.int64).max]
        edges = host.vertex_ids[
            np.stack(
                (
                    actual // old_layout.vertex_capacity,
                    actual % old_layout.vertex_capacity,
                ),
                axis=1,
            )
        ]
        protected_edges.extend(
            edges[np.all(np.isin(edges, source.vertex_ids), axis=1)].tolist()
        )
    if set(root_owner) != set(restored.tolist()):
        raise ValueError("Mixed preparation lacks a uniquely retained original root.")
    width = source.dimension + 1
    facet_keys = entity_keys(source.mesh, source.dimension - 1)
    facet_classes = np.full(facet_keys.shape[0], -1, dtype=np.int64)
    columns = np.asarray(
        tuple(combinations(range(width), source.dimension)), dtype=np.int64
    )
    rows = key_rows(
        facet_keys,
        np.sort(bank["parent_rows"][:, columns], axis=2).reshape((-1, source.dimension)),
    )
    if np.any(rows < 0):
        raise ValueError("Mixed preparation lost original source facet SCI.")
    np.maximum.at(facet_classes, rows, bank["parent_facet_classes"].reshape(-1))
    classes = bank["parent_classes"][parent_rows]
    empty = np.zeros(0, dtype=np.int64)
    source = source._replace(
        cells=_select(source.cells, positions)._replace(
            tuples=np.searchsorted(source.vertex_ids, restored_tuples),
            tags=restored_tags,
            generations=restored_levels,
        )
    )
    request = _prepared_request(
        source,
        empty,
        empty,
        np.unique(np.asarray(protected_edges, dtype=np.int64).reshape((-1, 2)), axis=0),
        np.unique(np.asarray(protected_vertices, dtype=np.int64)),
        classes,
        facet_classes,
        prepared.policy.maximum_closure_iterations,
        source_cell_classes=classes,
    )
    limits = prepared.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    issuer_values = np.max(np.asarray(jax.device_get(retained.cursors))[:, 2:], axis=0)
    issuers = (int(issuer_values[0]), int(issuer_values[1]))
    retired = []
    for keys, identifiers in zip(
        hierarchy.retired_entity_keys, hierarchy.retired_entity_ids, strict=True
    ):
        keys, identifiers = np.asarray(keys), np.asarray(identifiers)
        eligible = np.all(np.isin(keys, source.vertex_ids), axis=1)
        retired.append(
            (np.searchsorted(source.vertex_ids, keys[eligible]), identifiers[eligible])
        )
    from ._bisection import _reprepare_admitted_uniform

    expanded_rows = np.searchsorted(restored, expanded)
    expanded_source = source._replace(cells=_select(source.cells, expanded_rows))
    expanded_classes = classes[expanded_rows]
    expanded_request = _prepared_request(
        expanded_source,
        empty,
        empty,
        np.stack(
            (
                request.protected_codes // _SHIFT,
                request.protected_codes % _SHIFT,
            ),
            axis=1,
        ),
        np.unique(np.asarray(protected_vertices, dtype=np.int64)),
        expanded_classes,
        facet_classes,
        prepared.policy.maximum_closure_iterations,
        source_cell_classes=expanded_classes,
    )
    start = _reprepare_admitted_uniform(
        expanded_source,
        expanded_request,
        lineage,
        allowance,
        issuers,
        tuple(retired),
    )
    if start.uniform_refinement is None:
        raise ValueError(
            "Admitted uniform re-preparation lost its actual source action law."
        )
    combined, remap = _reconcile_uniform_refinement(
        start.uniform_refinement, lineage, expanded
    )
    count = restored.size
    vertex_count = source.vertex_ids.size
    seed = _HostEpoch(
        np.asarray(source.cells.ids, dtype=np.int64),
        np.ones((count,), dtype=np.bool_),
        np.asarray(source.cells.rows, dtype=np.int64),
        np.asarray(source.cells.tuples, dtype=np.int64),
        np.asarray(source.cells.tags, dtype=np.int64),
        np.asarray(source.cells.blocks, dtype=np.int64),
        np.asarray(source.cells.generations, dtype=np.int64),
        np.full((count,), -1, dtype=np.int64),
        np.full((count, 2), -1, dtype=np.int64),
        np.full((count,), -1, dtype=np.int64),
        np.zeros((count,), dtype=np.bool_),
        np.zeros((count,), dtype=np.bool_),
        np.zeros((count,), dtype=np.bool_),
        np.asarray(source.vertex_ids, dtype=np.int64),
        np.ones((vertex_count,), dtype=np.bool_),
        np.asarray(source.coordinates, dtype=np.float64),
        np.full((vertex_count, 2), -1, dtype=np.int64),
        np.zeros((vertex_count,), dtype=hosts[0].vertex_levels.dtype),
        np.full((vertex_count,), -1, dtype=hosts[0].vertex_removal.dtype),
        np.asarray((vertex_count, count, issuers[0], issuers[1]), dtype=np.int64),
        hosts[0].counters,
        0,
        np.asarray(classes, dtype=hosts[0].cell_classes.dtype),
        restored_facets,
        np.isin(source.vertex_ids, protected_vertices),
        empty,
    )
    root_seed = seed
    forest = _forest(start, expanded_source, expanded_request)
    vertices = _vertices(start, expanded_source, expanded_request)
    generated_count = forest["ids"].size
    generated_vertex_count = vertices["ids"].size
    generated = _HostEpoch(
        forest["ids"],
        forest["active"],
        forest["rows"],
        forest["tuples"],
        forest["tags"],
        forest["blocks"],
        forest["generations"],
        forest["parents"],
        forest["children"],
        forest["bisection_vertices"],
        np.zeros(generated_count, dtype=np.bool_),
        np.zeros(generated_count, dtype=np.bool_),
        np.zeros(generated_count, dtype=np.bool_),
        vertices["ids"],
        np.ones(generated_vertex_count, dtype=np.bool_),
        vertices["coordinates"],
        vertices["parents"],
        np.zeros(generated_vertex_count, dtype=hosts[0].vertex_levels.dtype),
        np.full(generated_vertex_count, -1, dtype=hosts[0].vertex_removal.dtype),
        np.asarray(
            (
                generated_vertex_count,
                generated_count,
                start.next_vertex + start.front.growth.levels.size,
                start.front.next_cell,
            ),
            dtype=np.int64,
        ),
        hosts[0].counters,
        0,
        forest["cell_classes"],
        forest["facet_classes"],
        vertices["protected"],
        empty,
    )
    generated_solver = {
        int(cell): (int(root), root_solver[int(root)])
        for cell, root in zip(forest["ids"], forest["origins"], strict=True)
    }
    expanded_edges = expanded_source.vertex_ids[
        np.stack(
            (
                expanded_request.protected_codes // _SHIFT,
                expanded_request.protected_codes % _SHIFT,
            ),
            axis=1,
        )
    ]
    seed, seed_origins, seed_solver_owners = _merge_prepared_root_part(
        root_seed,
        generated,
        np.arange(generated_count, dtype=np.int64),
        expanded,
        lineage,
        remap,
        root_seed.vertex_protected,
        np.full((0,), np.iinfo(np.int64).max, dtype=np.int64),
        root_seed.vertex_ids.size,
        expanded_edges,
        np.asarray(
            [root_solver[int(identifier)] for identifier in root_seed.cell_ids],
            dtype=np.int32,
        ),
        generated_solver,
    )
    declarations = list(_raw_source_blocks(prepared.source))
    declaration_ids = {block.block_id: index for index, block in enumerate(declarations)}
    incoming_blocks = []
    for block in _scientific_simplex_blocks(scientific.mesh):
        index = declaration_ids.get(block.block_id)
        if index is None:
            index = len(declarations)
            declarations.append(block)
            declaration_ids[block.block_id] = index
        incoming_blocks.append(index)
    seed = seed._replace(blocks=np.asarray(incoming_blocks, dtype=np.int64)[seed.blocks])
    incoming_owner = np.asarray(
        [root_owner[int(origin)] for origin in seed_origins],
        dtype=np.int64,
    )
    incoming_solver = {
        int(identifier): (int(origin), int(owner))
        for identifier, origin, owner in zip(
            seed.cell_ids, seed_origins, seed_solver_owners, strict=True
        )
    }
    edges = source.vertex_ids[
        np.stack(
            (request.protected_codes // _SHIFT, request.protected_codes % _SHIFT), axis=1
        )
    ]
    merged = tuple(
        _merge_prepared_root_part(
            host,
            seed,
            np.flatnonzero(incoming_owner == part),
            restored,
            lineage,
            remap,
            np.asarray(jax.device_get(flags[part])),
            np.asarray(jax.device_get(codes[part])),
            old_layout.vertex_capacity,
            edges,
            solver[part],
            incoming_solver,
        )
        for part, host in enumerate(hosts)
    )
    workspace = current_native_host_workspace()
    if workspace is None:
        raise RuntimeError(
            "Mixed source preparation requires its actual storage workspace."
        )
    workspace.retain_owner((seed, merged, combined))
    device = prepared.policy.device_policy
    if device is None:
        raise ValueError(
            "Mixed root preparation requires its original device capacities."
        )
    vertex_capacity, cell_capacity = device.capacities(
        max(base.vertex_ids.size for base, _, _ in merged),
        max(base.cell_ids.size for base, _, _ in merged),
    )
    edge_capacity = adaptive_simplex_bucket(
        max(1, max(base.protected_codes.size for base, _, _ in merged)), 1.0
    )
    layout = AdaptiveSimplexLayout(
        source.dimension,
        scientific.mesh.ambient_dimension,
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        protected_edge_capacity=edge_capacity,
        maximum_closure_iterations=prepared.policy.maximum_closure_iterations,
        maximum_coarsening_passes=device.maximum_coarsening_passes,
    )
    chosen = tuple(jax.devices()[: storage.partition_count])
    if len(chosen) != storage.partition_count:
        raise ValueError("Mixed roots require their actual retained device ownership.")
    parts = AdaptiveSimplexParts(
        chosen,
        neighbor_pairs=tuple(
            (left, right)
            for left in range(storage.partition_count)
            for right in range(storage.partition_count)
            if left != right
        ),
    )
    sharding = jax.sharding.NamedSharding(
        parts.mesh, jax.sharding.PartitionSpec(parts.axis_name)
    )
    states = []
    origins = np.full((storage.partition_count, cell_capacity), -1, dtype=np.int64)
    owners = np.full(origins.shape, -1, dtype=np.int32)
    exterior = np.zeros((*origins.shape, width), dtype=np.bool_)
    opposite = np.asarray(
        tuple(
            tuple(column for column in range(width) if column != facet)
            for facet in range(width)
        ),
        dtype=np.int64,
    )
    all_keys = np.concatenate(
        [
            np.sort(
                base.vertex_ids[base.cells[base.cell_active]][:, opposite], axis=2
            ).reshape((-1, source.dimension))
            for base, _, _ in merged
        ]
    )
    unique, occurrences = np.unique(all_keys, axis=0, return_counts=True)
    for part, (base, actual_origins, actual_owners) in enumerate(merged):
        selected = np.arange(base.cell_ids.size, dtype=np.int64)
        piece = _part_arrays(
            base,
            None,
            selected,
            protected_codes=base.protected_codes,
            protected_vertex_capacity=base.vertex_ids.size,
        )
        state = _owned_part_state(layout, base, selected, piece)
        size = selected.size
        vertices = piece["vertices"]
        state = eqx.tree_at(
            lambda value: (
                value.counters,
                value.retired,
                value.refine_rejected,
                value.coarsen_marked,
                value.vertex_levels,
                value.vertex_removal,
            ),
            state,
            (
                jnp.asarray(base.counters),
                jnp.pad(jnp.asarray(base.retired), (0, cell_capacity - size)),
                jnp.pad(jnp.asarray(base.refine_rejected), (0, cell_capacity - size)),
                jnp.pad(jnp.asarray(base.coarsen_marked), (0, cell_capacity - size)),
                jnp.pad(
                    jnp.asarray(
                        base.vertex_levels[vertices], dtype=state.vertex_levels.dtype
                    ),
                    (0, vertex_capacity - vertices.size),
                ),
                jnp.pad(
                    jnp.asarray(
                        base.vertex_removal[vertices], dtype=state.vertex_removal.dtype
                    ),
                    (0, vertex_capacity - vertices.size),
                    constant_values=-1,
                ),
            ),
        )
        states.append(state)
        origins[part, :size], owners[part, :size] = actual_origins, actual_owners
        keys = np.sort(base.vertex_ids[base.cells][:, opposite], axis=2).reshape(
            (-1, source.dimension)
        )
        rows = key_rows(unique, keys)
        valid = rows >= 0
        flags = np.zeros(rows.shape, dtype=np.bool_)
        flags[valid] = occurrences[rows[valid]] == 1
        exterior[part, :size] = flags.reshape((size, width)) & base.cell_active[:, None]
    stacked = jax.tree_util.tree_map(
        lambda *values: jax.device_put(jnp.stack(values), sharding), *states
    )
    stacked = _physical_packed_publication_states(stacked, combined)
    all_ids = np.concatenate([base.cell_ids for base, _, _ in merged])
    all_origins = np.concatenate([value for _, value, _ in merged])
    order = np.argsort(all_ids, kind="stable")
    all_ids, all_origins = all_ids[order], all_origins[order]
    start = start._replace(uniform_refinement=combined)
    roots = np.where(all_ids != all_origins, all_origins, -1)
    anchor = _Anchor(
        source,
        start,
        all_ids,
        all_origins,
        roots,
        np.flatnonzero(np.isin(all_ids, restored)),
        source_blocks=tuple(declarations),
    )
    simplex = PreparedAdaptiveSimplex(
        prepared, layout, _addressable_part_state(stacked, 0), anchor
    )
    placed_owners = jax.device_put(jnp.asarray(owners), sharding)
    return PartitionedAdaptiveSimplex(
        simplex,
        layout,
        parts,
        stacked,
        jax.device_put(jnp.asarray(exterior), sharding),
        jax.device_put(jnp.asarray(origins), sharding),
        solver_cell_owners=placed_owners,
        placement_arrays=(("placement/initial_solver_cell_owners", placed_owners),),
    )


def _run_prepare_restored_original_roots(
    prepared: PreparedMeshAdaptation,
    old_layout: AdaptiveSimplexLayout,
    retained: AdaptiveSimplexState,
    /,
) -> PartitionedAdaptiveSimplex | None:
    """Prepare genuine full-inverse roots before allocating bounded owner states."""
    if not isinstance(prepared.request, MarkedMeshAdaptation):
        raise TypeError("Distributed simplex preparation requires its marked request.")
    hierarchy = prepared.request.hierarchy
    if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError(
            "Distributed simplex preparation requires its canonical bisection hierarchy."
        )
    if (
        hierarchy is None
        or hierarchy.record_parent_ids.size
        or hierarchy.uniform_refinement is not None
    ):
        return None
    evidence = prepared.source.collective_evidence
    if (
        isinstance(evidence, CollectiveMeshEvidence)
        and evidence.uniform_refinement is not None
        and np.intersect1d(
            np.asarray(prepared.request.refine_cell_ids, dtype=np.int64),
            np.asarray(evidence.uniform_refinement.parent_ids, dtype=np.int64),
        ).size
    ):
        # Regenerate only explicitly marked restored roots; the mixed preparer
        # keeps every other genuine parent compact.
        return None
    scientific = require_original_meshing_source(prepared.source)
    if not isinstance(scientific, CellMeshingResult):
        return None
    source = _prepared_source(scientific.mesh)
    storage = prepared.source.mesh.storage
    if storage is None:
        return None
    arrays = dict(storage.logical_arrays)
    root_ids = np.asarray(jax.device_get(arrays["cell_global_ids"]))[
        : storage.global_entity_counts[-1]
    ]
    if not np.array_equal(root_ids, source.cells.ids):
        return None
    root_corners = np.asarray(jax.device_get(arrays["cell_vertices"]))[: root_ids.size]
    if not np.array_equal(root_corners, source.vertex_ids[source.cells.rows]):
        return None
    vertex_ids = np.asarray(jax.device_get(arrays["vertex_global_ids"]))[
        : storage.global_entity_counts[0]
    ]
    coordinates = np.asarray(jax.device_get(arrays["coordinates"]))[: vertex_ids.size]
    if not np.array_equal(vertex_ids, source.vertex_ids) or not np.array_equal(
        coordinates, source.coordinates
    ):
        return None
    active = np.asarray(jax.device_get(retained.mesh.cell_active))
    if np.any(np.asarray(jax.device_get(retained.parents))[active] >= 0):
        return None
    flags, codes, _ = _retained_protection(prepared, retained)
    host = jax.device_get(retained)
    cell_classes = np.zeros(root_ids.size, dtype=np.int64)
    root_generations = np.empty(root_ids.size, dtype=np.int64)
    root_tuples = np.empty((root_ids.size, source.dimension + 1), dtype=np.int64)
    root_tags = np.empty(root_ids.size, dtype=np.int64)
    root_solver_owners = np.empty(root_ids.size, dtype=np.int32)
    retained_solver_owners = np.asarray(
        jax.device_get(_source_solver_cell_owners(prepared, retained))
    )
    facet_keys = entity_keys(source.mesh, source.dimension - 1)
    facet_classes = np.zeros(facet_keys.shape[0], dtype=np.int64)
    seen = np.zeros(root_ids.size, dtype=np.bool_)
    protected_vertices, protected_edges = [], []
    columns = np.asarray(
        tuple(
            tuple(column for column in range(source.dimension + 1) if column != opposite)
            for opposite in range(source.dimension + 1)
        ),
        dtype=np.int64,
    )
    for part in range(active.shape[0]):
        slots = np.flatnonzero(active[part])
        ids = np.asarray(host.mesh.cell_ids[part])[slots]
        positions = np.searchsorted(root_ids, ids)
        if (
            np.any(positions >= root_ids.size)
            or not np.array_equal(root_ids[positions], ids)
            or np.any(seen[positions])
        ):
            raise ValueError(
                "Full-inverse preparation lost unique original SCI root ownership."
            )
        seen[positions] = True
        cell_classes[positions] = np.asarray(host.cell_classes[part])[slots]
        root_generations[positions] = np.asarray(host.generations[part])[slots]
        root_tuples[positions] = np.asarray(host.mesh.vertex_ids[part])[
            np.asarray(host.tuples[part])[slots]
        ]
        root_tags[positions] = np.asarray(host.tags[part])[slots]
        root_solver_owners[positions] = retained_solver_owners[part, slots]
        corners = np.asarray(host.mesh.vertex_ids[part])[
            np.asarray(host.mesh.cells[part])[slots]
        ]
        keys = np.sort(corners[:, columns], axis=2).reshape((-1, source.dimension))
        rows = key_rows(facet_keys, keys)
        values = np.asarray(host.facet_classes[part])[slots].reshape(-1)
        if np.any(rows < 0):
            raise ValueError("Full-inverse root facets differ from original SCI keys.")
        np.maximum.at(facet_classes, rows, values)
        ids_bank = np.asarray(host.mesh.vertex_ids[part])
        protected_vertices.extend(
            ids_bank[
                np.asarray(jax.device_get(flags[part]))
                & np.asarray(host.mesh.vertex_active[part])
            ].tolist()
        )
        actual_codes = np.asarray(jax.device_get(codes[part]))
        actual_codes = actual_codes[actual_codes != np.iinfo(np.int64).max]
        ends = np.stack(
            (
                actual_codes // old_layout.vertex_capacity,
                actual_codes % old_layout.vertex_capacity,
            ),
            axis=1,
        )
        protected_edges.extend(ids_bank[ends].tolist())
    if not np.all(seen):
        raise ValueError("Full-inverse preparation lacks an original SCI root.")
    empty = np.zeros(0, dtype=np.int64)
    source = source._replace(
        cells=source.cells._replace(
            tuples=np.searchsorted(source.vertex_ids, root_tuples),
            tags=root_tags,
            generations=root_generations,
        )
    )
    request = _prepared_request(
        source,
        empty,
        empty,
        np.unique(np.asarray(protected_edges, dtype=np.int64).reshape((-1, 2)), axis=0),
        np.unique(np.asarray(protected_vertices, dtype=np.int64)),
        cell_classes,
        np.maximum(facet_classes - 1, -1),
        prepared.policy.maximum_closure_iterations,
    )
    limits = prepared.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    # A complete inverse has cut over to genuine original roots. Historical
    # uniform evidence remains in the accepted theorem, but re-expanding those
    # children here would consume execution capacity for absent topology.
    start = _prepared_start(
        source,
        request,
        prepared.policy.compatibility,
        hierarchy,
        allowance,
        scientific,
        future_refinement=True,
    )
    forest, vertices = _forest(start, source, request), _vertices(start, source, request)
    count, vertex_count = forest["ids"].size, vertices["ids"].size
    base = _HostEpoch(
        forest["ids"],
        forest["active"],
        forest["rows"],
        forest["tuples"],
        forest["tags"],
        forest["blocks"],
        forest["generations"],
        forest["parents"],
        forest["children"],
        forest["bisection_vertices"],
        np.zeros(count, dtype=np.bool_),
        np.zeros(count, dtype=np.bool_),
        np.zeros(count, dtype=np.bool_),
        vertices["ids"],
        np.ones(vertex_count, dtype=np.bool_),
        vertices["coordinates"],
        vertices["parents"],
        np.zeros(vertex_count, dtype=np.int64),
        np.full(vertex_count, -1, dtype=np.int64),
        np.asarray(
            (
                vertex_count,
                count,
                start.next_vertex + start.front.growth.levels.size,
                start.front.next_cell,
            )
        ),
        np.zeros_like(np.asarray(host.counters[0])),
        0,
        forest["cell_classes"],
        forest["facet_classes"],
        vertices["protected"],
        empty,
    )
    tables = _source_entity_tables(prepared)
    ids, owners = (np.asarray(jax.device_get(value)) for value in tables[-1][1:])
    order = np.argsort(ids, kind="stable")
    slot_owner = owners[order][np.searchsorted(ids[order], forest["origins"])]
    parts_count = storage.partition_count
    slots = tuple(np.flatnonzero(slot_owner == part) for part in range(parts_count))
    edges = np.stack(
        (request.protected_codes // _SHIFT, request.protected_codes % _SHIFT), axis=1
    )
    encoded = edges[:, 0] * vertex_count + edges[:, 1]
    pieces = tuple(
        _part_arrays(
            base,
            None,
            selected,
            protected_codes=encoded,
            protected_vertex_capacity=vertex_count,
        )
        for selected in slots
    )
    device = prepared.policy.device_policy
    if device is None:
        raise ValueError(
            "Full-inverse preparation requires its original device capacities."
        )
    vertex_capacity, cell_capacity = device.capacities(
        max(piece["vertices"].size for piece in pieces),
        max(selected.size for selected in slots),
    )
    edge_capacity = adaptive_simplex_bucket(
        max(1, max((piece["protected_edges"].shape[0] for piece in pieces), default=0)),
        1.0,
    )
    layout = AdaptiveSimplexLayout(
        source.dimension,
        scientific.mesh.ambient_dimension,
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        protected_edge_capacity=edge_capacity,
        maximum_closure_iterations=prepared.policy.maximum_closure_iterations,
        maximum_coarsening_passes=device.maximum_coarsening_passes,
    )
    states = []
    for part, (selected, piece) in enumerate(zip(slots, pieces, strict=True)):
        state = _owned_part_state(layout, base, selected, piece)
        state = eqx.tree_at(lambda value: value.counters, state, retained.counters[part])
        states.append(state)
    chosen = tuple(jax.devices()[:parts_count])
    if len(chosen) != parts_count:
        raise ValueError(
            "Full-inverse preparation requires its original device ownership."
        )
    parts = AdaptiveSimplexParts(
        chosen,
        neighbor_pairs=tuple(
            (left, right)
            for left in range(parts_count)
            for right in range(parts_count)
            if left != right
        ),
    )
    sharding = jax.sharding.NamedSharding(
        parts.mesh, jax.sharding.PartitionSpec(parts.axis_name)
    )
    stacked = jax.tree_util.tree_map(
        lambda *values: jax.device_put(jnp.stack(values), sharding), *states
    )
    if start.uniform_refinement is not None:
        stacked = _physical_packed_publication_states(stacked, start.uniform_refinement)
    keys = np.sort(forest["rows"][:, columns], axis=2)
    _, inverse, occurrences = np.unique(
        keys.reshape((-1, source.dimension)),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    boundary = occurrences[inverse].reshape((count, source.dimension + 1)) == 1
    exterior = np.zeros(
        (parts_count, cell_capacity, source.dimension + 1), dtype=np.bool_
    )
    origins = np.full((parts_count, cell_capacity), -1, dtype=np.int64)
    solver_owners = np.full((parts_count, cell_capacity), -1, dtype=np.int32)
    for part, selected in enumerate(slots):
        exterior[part, : selected.size] = boundary[selected]
        origins[part, : selected.size] = forest["origins"][selected]
        solver_owners[part, : selected.size] = root_solver_owners[
            np.searchsorted(root_ids, forest["origins"][selected])
        ]
    roots = np.where(forest["origins"] != forest["ids"], forest["origins"], -1)
    anchor = _Anchor(
        source,
        start,
        forest["ids"],
        forest["origins"],
        np.where(forest["parents"] < 0, roots, -1),
        np.flatnonzero(np.isin(forest["ids"], source.cells.ids)),
        source_blocks=_scientific_simplex_blocks(scientific.mesh),
    )
    simplex = PreparedAdaptiveSimplex(
        prepared, layout, _addressable_part_state(stacked, 0), anchor
    )
    placed_owners = jax.device_put(jnp.asarray(solver_owners), sharding)
    return PartitionedAdaptiveSimplex(
        simplex,
        layout,
        parts,
        stacked,
        jax.device_put(jnp.asarray(exterior), sharding),
        jax.device_put(jnp.asarray(origins), sharding),
        solver_cell_owners=placed_owners,
        placement_arrays=(("placement/initial_solver_cell_owners", placed_owners),),
    )


def _scientific_simplex_blocks(mesh: CellMesh, /) -> tuple[CellBlock, ...]:
    """Retain authored block identity while rejecting non-simplex bank declarations."""
    blocks: list[CellBlock] = []
    for block in mesh.blocks:
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Device simplex source requires canonical cell block declarations."
            )
        blocks.append(block)
    return tuple(blocks)


def _raw_source_blocks(source: CellMeshingResult, /) -> tuple[CellBlock, ...]:
    """Read the actual raw numeric bank declaration, never the regrouped target."""
    evidence = source.collective_evidence
    if isinstance(evidence, CollectiveMeshEvidence):
        return evidence.publication_source_blocks
    return _scientific_simplex_blocks(source.mesh)


def _source_simplex_on_states(
    prepared: PreparedMeshAdaptation,
    layout: AdaptiveSimplexLayout,
    states: AdaptiveSimplexState,
    /,
    *,
    raw_source_blocks: tuple[CellBlock, ...] | None = None,
    uniform_refinement: BisectionUniformRefinement | None = None,
) -> PreparedAdaptiveSimplex:
    """Bind source-local metadata to an actual current addressable forest."""
    local_source = _prepared_source(prepared.source.mesh)
    first = states.mesh.cell_ids.addressable_shards[0].index[0]
    if not isinstance(first, slice):
        raise ValueError(
            "Current raw execution requires its explicit leading part rectangle."
        )
    part = 0 if first.start is None else first.start
    local_state = _addressable_part_state(states, part)
    width = local_source.dimension + 1
    empty = np.zeros((0,), dtype=np.int64)
    hierarchy = (
        prepared.request.hierarchy
        if isinstance(prepared.request, MarkedMeshAdaptation)
        else None
    )
    if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError(
            "Distributed simplex preparation requires its canonical bisection hierarchy."
        )
    lineage = None if hierarchy is None else hierarchy.uniform_refinement
    evidence = prepared.source.collective_evidence
    if isinstance(evidence, CollectiveMeshEvidence):
        lineage = evidence.uniform_refinement
    if uniform_refinement is not None:
        lineage = uniform_refinement
    start = _Start(
        _Front(
            local_source.cells,
            _Growth(
                np.zeros((0, width), dtype=np.int64),
                np.zeros((0, width), dtype=np.float64),
                empty,
            ),
            np.zeros((0, 2), dtype=np.int64),
            empty,
            local_source.vertex_ids.size,
            int(jax.device_get(jnp.max(states.cursors[:, 3]))),
        ),
        _Records(
            empty,
            empty,
            np.zeros((0, width), dtype=np.int64),
            np.zeros((0, width), dtype=np.int64),
            empty,
            np.zeros((0, 2), dtype=np.int64),
            empty,
        ),
        np.zeros((local_source.cells.ids.size,), dtype=np.bool_),
        int(jax.device_get(jnp.max(states.cursors[:, 2]))),
        None,
        0,
        False,
        tuple(
            (np.zeros((0, degree + 1), dtype=np.int64), empty)
            for degree in range(1, local_source.dimension)
        )
        if hierarchy is None
        else _owned_retired_entities(
            local_source,
            tuple(np.asarray(value) for value in hierarchy.retired_entity_keys),
            tuple(np.asarray(value) for value in hierarchy.retired_entity_ids),
        ),
        lineage,
    )
    ids = local_source.cells.ids
    anchor = _Anchor(
        local_source,
        start,
        ids,
        ids,
        np.full(ids.shape, -1, dtype=np.int64),
        np.arange(ids.size, dtype=np.int64),
        source_blocks=(
            _raw_source_blocks(prepared.source)
            if raw_source_blocks is None
            else raw_source_blocks
        ),
    )
    return PreparedAdaptiveSimplex(prepared, layout, local_state, anchor)


def commit_partitioned_mesh_restart(
    repack: SimplexRestartRepack,
    prepared: PreparedMeshAdaptation,
    /,
) -> tuple[MeshAdaptationResult, ...]:
    """Publish an independently validated changed-count accepted source epoch."""
    from ._restart_distribution import (
        SimplexRestartRepack,
    )

    if not isinstance(repack, SimplexRestartRepack) or not isinstance(
        prepared, PreparedMeshAdaptation
    ):
        raise TypeError(
            "Restart publication requires the actual numerical repack and admitted preparation."
        )
    if (
        repack.source_result is None
        or prepared.source.result_id != repack.source_result.result_id
    ):
        raise ValueError(
            "Restart preparation must bind the exact saved accepted source result."
        )
    if not isinstance(prepared.request, MarkedMeshAdaptation) or (
        prepared.request.refine_cell_ids.size or prepared.request.coarsen_cell_ids.size
    ):
        raise ValueError(
            "Restart ownership publication requires its exact placement-only marked request."
        )
    policy = prepared.policy.partition_policy
    if policy is None or policy.part_count != repack.parts.part_count:
        raise ValueError(
            "Restart placement must satisfy the exact requested new owner count."
        )
    from ._restart_distribution import SimplexRestartRepackProof

    proof = SimplexRestartRepackProof(repack)
    source = repack.source_result
    storage = source.mesh.storage
    evidence = source.collective_evidence
    if storage is None or not isinstance(evidence, CollectiveMeshEvidence):
        raise ValueError(
            "Accepted restart requires complete retained source forest science."
        )
    validate_partitioned_mesh_evidence(storage, evidence)
    source_arrays = dict(storage.logical_arrays)
    dimension = repack.layout.dimension
    width = dimension + 1
    cell_count = storage.global_entity_counts[-1]
    logical_ids = source_arrays["cell_global_ids"]
    id_table = jnp.where(
        jnp.arange(logical_ids.shape[0]) < cell_count,
        logical_ids,
        jnp.iinfo(jnp.int64).max,
    )
    rows = jnp.searchsorted(id_table, repack.states.mesh.cell_ids)
    safe_rows = jnp.minimum(rows, id_table.shape[0] - 1)
    active = repack.states.mesh.cell_active
    if not bool(
        jax.device_get(
            jnp.all(
                ~active
                | (
                    (rows < id_table.shape[0])
                    & (id_table[safe_rows] == repack.states.mesh.cell_ids)
                )
            )
        )
    ):
        raise ValueError(
            "Restart active records are absent from the accepted logical topology."
        )
    columns = jnp.asarray(
        tuple(
            tuple(column for column in range(width) if column != opposite)
            for opposite in range(width)
        ),
        dtype=jnp.int32,
    )
    logical_cells = source_arrays["cell_vertices"]
    logical_keys = jnp.sort(logical_cells[:, columns], axis=2).reshape((-1, dimension))
    logical_valid = jnp.repeat(jnp.arange(logical_cells.shape[0]) < cell_count, width)
    facet_table = source_arrays[f"entity_vertices_{dimension - 1}"]
    facet_count = storage.global_entity_counts[dimension - 1]
    searchable_facets = jnp.where(
        jnp.arange(facet_table.shape[0])[:, None] < facet_count,
        facet_table,
        jnp.iinfo(jnp.int64).max,
    )
    logical_facet_rows = _key_positions(searchable_facets, logical_keys)
    incidence = (
        jnp.zeros((facet_table.shape[0],), dtype=jnp.int32)
        .at[jnp.maximum(logical_facet_rows, 0)]
        .add(logical_valid.astype(jnp.int32))
    )
    target_cells = logical_cells[safe_rows]
    target_keys = jnp.sort(target_cells[..., columns], axis=-1).reshape((-1, dimension))
    target_facet_rows = _key_positions(searchable_facets, target_keys).reshape(
        (*repack.states.mesh.cell_ids.shape, width)
    )
    if not bool(jax.device_get(jnp.all(~active[..., None] | (target_facet_rows >= 0)))):
        raise ValueError(
            "Restart cells reference a facet absent from accepted logical topology."
        )
    exterior = active[..., None] & (incidence[jnp.maximum(target_facet_rows, 0)] == 1)
    exterior = jax.device_put(exterior, repack.states.mesh.cell_ids.sharding)
    arrays: dict[str, jax.Array] = {
        "placement/requested_target_owners": repack.requested_target_owners,
        "placement/saved_cell_owners": repack.saved_cell_owners,
        "placement/saved_vertex_owners": repack.saved_vertex_owners,
        "placement/cell_owners": repack.cell_owners,
        "placement/vertex_owners": repack.vertex_owners,
        "placement/cell_saved_locations": repack.cell_saved_locations,
        "placement/vertex_saved_locations": repack.vertex_saved_locations,
        "placement/saved_cursors": repack.saved_cursors,
        "placement/saved_clocks": repack.saved_clocks,
        "placement/saved_counters": repack.saved_counters,
        "placement/initial_solver_cell_owners": repack.solver_cell_owners,
        "placement/status": repack.status,
        "placement/cells_before": repack.cells_before,
        "placement/cells_after": repack.cells_after,
        "placement/vertices_before": repack.vertices_before,
        "placement/vertices_after": repack.vertices_after,
    }
    if repack.saved_solver_cell_owners is None or repack.cell_migration_counts is None:
        raise ValueError(
            "Accepted restart lacks its actual old-to-new solver traffic proof."
        )
    arrays["placement/saved_solver_cell_owners"] = repack.saved_solver_cell_owners
    arrays["placement/cell_migration_counts"] = repack.cell_migration_counts
    for family, values in (
        ("saved_cell_history", repack.saved_cell_history),
        ("saved_vertex_history", repack.saved_vertex_history),
        ("cell_history", repack.cell_history),
        ("vertex_history", repack.vertex_history),
    ):
        for index, value in enumerate(values):
            arrays[f"placement/{family}/{index}"] = value
    replicated = jax.sharding.NamedSharding(
        repack.parts.mesh, jax.sharding.PartitionSpec()
    )
    graph = repack.graph_proposal
    if graph is not None:
        arrays["placement/graph/parts"] = graph.parts
        arrays["placement/graph/vertex_weights"] = graph.vertex_weights
        arrays["placement/graph/edge_weights"] = graph.edge_weights
        arrays["placement/graph/maximum_imbalance"] = jax.device_put(
            np.asarray(graph.maximum_imbalance, dtype=np.float64),
            replicated,
        )
        for name in _WORKSET_FIELDS:
            arrays[f"placement/graph/workset/{name}"] = getattr(graph.workset, name)
    arrays["placement/proof_content_id"] = jax.device_put(
        np.frombuffer(bytes.fromhex(proof.content_id), dtype=np.uint8).copy(),
        replicated,
    )
    allowance = _phase_allowance(prepared, None)
    with _uniform_execution(allowance) as budget:
        simplex = _source_simplex_on_states(prepared, repack.layout, repack.states)
        _retain_preparation(simplex)
        workspace = current_native_host_workspace()
        if workspace is None:
            raise RuntimeError(
                "Restart publication lost its actual native host workspace."
            )
        workspace.retain_owner((repack, prepared))
        partitioned = PartitionedAdaptiveSimplex(
            simplex,
            repack.layout,
            repack.parts,
            repack.states,
            exterior,
            jnp.where(repack.states.mesh.cell_active, repack.states.mesh.cell_ids, -1),
            solver_cell_owners=repack.solver_cell_owners,
            placement_arrays=tuple(sorted(arrays.items())),
            restart_repack=repack,
            restart_proof=proof,
        )
    if budget.evidence is not None:
        partitioned = _bind_partitioned_preparation_receipt(
            partitioned,
            NativeExecutionRecord(budget.evidence, owner_id=prepared.prepared_id),
        )
    return commit_partitioned_adaptive_simplex(partitioned, repack.states)


def _restart_receipt_from_evidence(
    evidence: CollectiveMeshEvidence, /
) -> _SimplexRestartInventory:
    """Reconstruct historical scientific inputs without demanding their old devices."""
    from ._restart_distribution import _SimplexRestartInventory

    source = evidence.source
    storage = source.mesh.storage
    source_evidence = source.collective_evidence
    if storage is None or not isinstance(source_evidence, CollectiveMeshEvidence):
        raise ValueError(
            "Restart proof requires its unchanged actual accepted predecessor."
        )
    _, saved, _ = restore_partitioned_mesh_state(storage, source_evidence)
    arrays = dict(evidence.logical_arrays)
    state = evidence.initial_states
    count = evidence.partition_count

    def values(family: str) -> tuple[jax.Array, ...]:
        prefix = f"placement/{family}/"
        names = tuple(name for name in arrays if name.startswith(prefix))
        indices = sorted(int(name.removeprefix(prefix)) for name in names)
        if indices != list(range(len(indices))):
            raise ValueError(
                "Saved scientific histories have a missing or duplicate declaration."
            )
        return tuple(arrays[f"{prefix}{index}"] for index in indices)

    graph = None
    if "placement/graph/parts" in arrays:
        graph = SimplexGraphRestartProposal(
            SimplexNeighborhoodWorkset(
                *(arrays[f"placement/graph/workset/{name}"] for name in _WORKSET_FIELDS)
            ),
            arrays["placement/graph/vertex_weights"],
            arrays["placement/graph/edge_weights"],
            arrays["placement/graph/parts"],
            arrays["placement/requested_target_owners"],
            float(
                np.asarray(jax.device_get(arrays["placement/graph/maximum_imbalance"]))
            ),
        )
    return _SimplexRestartInventory(
        evidence.axis_name,
        count,
        evidence.neighbor_pairs,
        evidence.layout,
        state,
        arrays["placement/cell_owners"],
        arrays["placement/vertex_owners"],
        values("cell_history"),
        values("vertex_history"),
        arrays["placement/status"],
        arrays["placement/cells_before"],
        arrays["placement/cells_after"],
        arrays["placement/vertices_before"],
        arrays["placement/vertices_after"],
        arrays["placement/saved_cursors"],
        arrays["placement/saved_clocks"],
        arrays["placement/saved_counters"],
        saved,
        arrays["placement/saved_cell_owners"],
        arrays["placement/saved_vertex_owners"],
        source,
        arrays["placement/cell_saved_locations"],
        arrays["placement/vertex_saved_locations"],
        arrays["placement/requested_target_owners"],
        values("saved_cell_history"),
        values("saved_vertex_history"),
        arrays["placement/initial_solver_cell_owners"],
        arrays["placement/saved_solver_cell_owners"],
        arrays["placement/cell_migration_counts"],
        graph,
    )


def execute_partitioned_mesh_adaptation(
    partitioned: PartitionedAdaptiveSimplex, /
) -> tuple[MeshAdaptationResult, ...]:
    if not isinstance(partitioned, PartitionedAdaptiveSimplex):
        raise TypeError("partitioned must be PartitionedAdaptiveSimplex.")
    preparation = partitioned.prepared.adaptation
    receipt = partitioned.execution_evidence
    if receipt is None:
        receipt = partitioned.prepared.execution_evidence
    with _uniform_execution(_phase_allowance(preparation, receipt)) as budget:
        _retain_preparation(partitioned)
        results = _execute_partitioned_mesh_adaptation(partitioned)
    if budget.evidence is None:
        return results
    record = NativeExecutionRecord(
        budget.evidence, preparation_evidence=receipt, owner_id=preparation.prepared_id
    )
    return _bind_collective_execution_receipt(preparation, results, record)


def _bind_collective_execution_receipt(
    preparation: PreparedMeshAdaptation,
    results: tuple[MeshAdaptationResult, ...],
    record: NativeExecutionRecord,
    /,
) -> tuple[MeshAdaptationResult, ...]:
    """Use canonical constructors without copying the actual raw proof holder."""
    rebound = []
    for result in results:
        target = result.target.with_execution_evidence(record)
        outcome = _RouteOutcome(
            result.status,
            target,
            result.transition,
            result.lineage,
            result.stencil,
            result.transfer,
            result.metric,
            result.evidence,
            result.hierarchy,
            result.common_refinement,
        )
        rebound.append(
            MeshAdaptationResult(
                preparation,
                outcome,
                result.compliance,
                result.distribution,
                result.elapsed_seconds,
            )
        )
    return tuple(rebound)


def _execute_partitioned_mesh_adaptation(
    partitioned: PartitionedAdaptiveSimplex, /
) -> tuple[MeshAdaptationResult, ...]:
    """Execute the exact prepared marked request and publish addressable results."""
    if not isinstance(partitioned, PartitionedAdaptiveSimplex):
        raise TypeError("partitioned must be PartitionedAdaptiveSimplex.")
    request = partitioned.prepared.adaptation.request
    if not isinstance(request, MarkedMeshAdaptation):
        raise TypeError("Partitioned simplex execution requires its marked request.")
    refine_marks = partitioned.cell_marks(request.refine_cell_ids)
    # Restored parents expanded during bounded preparation have already consumed
    # that exact refinement request; do not bisect every canonical child again.
    refine_marks &= partitioned.states.mesh.cell_ids == jnp.asarray(
        partitioned.slot_origins, dtype=jnp.int64
    )
    refined = refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        refine_marks,
    )
    coarsened = coarsen_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        refined.state,
        partitioned.cell_marks(request.coarsen_cell_ids),
    )
    return commit_partitioned_adaptive_simplex(partitioned, coarsened.state)


def _part_arrays(
    base: _HostEpoch,
    prepared: PreparedAdaptiveSimplex | None,
    slots: Any,
    /,
    *,
    protected_codes: np.ndarray | None = None,
    protected_vertex_capacity: int | None = None,
) -> Any:
    """Owned cells of one part and their vertices, in global-ID slot order."""

    vertices = np.unique(base.cells[slots])
    local = np.full((base.vertex_ids.size,), -1, dtype=np.int64)
    local[vertices] = np.arange(vertices.size)
    if protected_codes is None:
        if prepared is None:
            raise ValueError(
                "Owned host preparation requires its actual protection encoding."
            )
        protected_codes = np.asarray(prepared.state.protected_codes, dtype=np.int64)
    codes = protected_codes
    codes = codes[codes != np.iinfo(np.int64).max]
    if protected_vertex_capacity is None:
        if prepared is None:
            raise ValueError(
                "Owned host preparation requires its actual protection capacity."
            )
        protected_vertex_capacity = prepared.layout.vertex_capacity
    capacity = protected_vertex_capacity
    ends = local[np.stack((codes // capacity, codes % capacity), axis=1)]
    cell_slots = np.full(base.cell_ids.shape, -1, dtype=np.int64)
    cell_slots[slots] = np.arange(slots.size, dtype=np.int64)
    parent_slots = base.parents[slots] // 2
    parents = np.where(
        base.parents[slots] >= 0,
        2 * cell_slots[np.maximum(parent_slots, 0)] + base.parents[slots] % 2,
        -1,
    )
    raw_children = base.children[slots]
    children = np.where(raw_children >= 0, cell_slots[np.maximum(raw_children, 0)], -1)
    if np.any(
        (base.parents[slots] >= 0) & (cell_slots[np.maximum(parent_slots, 0)] < 0)
    ) or np.any((raw_children >= 0) & (children < 0)):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "A distributed owner must retain the complete actual binary source family.",
            stage="distributed-source-ownership",
        )
    return {
        "vertices": vertices,
        "cells": local[base.cells[slots]],
        "tuples": local[base.tuples[slots]],
        "protected_edges": ends[np.all(ends >= 0, axis=1)],
        "parents": parents,
        "children": children,
        "bisection_vertices": np.where(
            base.bisection_vertices[slots] >= 0,
            local[np.maximum(base.bisection_vertices[slots], 0)],
            -1,
        ),
        "vertex_parents": np.where(
            base.vertex_parents[vertices] >= 0,
            local[np.maximum(base.vertex_parents[vertices], 0)],
            -1,
        ),
    }


def _owned_part_state(
    layout: AdaptiveSimplexLayout,
    base: _HostEpoch,
    slots: np.ndarray,
    piece: dict[str, np.ndarray],
    /,
) -> AdaptiveSimplexState:
    """Allocate complete source-owned simplex families through the state owner."""
    vertices = piece["vertices"]
    return adaptive_simplex_state(
        layout,
        coordinates=base.coordinates[vertices],
        vertex_ids=base.vertex_ids[vertices],
        vertex_active=base.vertex_active[vertices],
        vertex_parents=piece["vertex_parents"],
        vertex_protected=base.vertex_protected[vertices],
        cells=piece["cells"],
        tuples=piece["tuples"],
        tags=base.tags[slots],
        blocks=base.blocks[slots],
        generations=base.generations[slots],
        parents=piece["parents"],
        children=piece["children"],
        bisection_vertices=piece["bisection_vertices"],
        cell_ids=base.cell_ids[slots],
        cell_active=base.cell_active[slots],
        cell_classes=base.cell_classes[slots],
        facet_classes=base.facet_classes[slots],
        protected_edges=piece["protected_edges"],
        next_vertex_id=int(base.cursors[2]),
        next_cell_id=int(base.cursors[3]),
    )


def _prepare_initial_partition(
    adaptation: PreparedMeshAdaptation, /
) -> PartitionedAdaptiveSimplex:
    """Initialize raw owned roots from actual native initial publication banks."""
    source = adaptation.source
    storage = source.mesh.storage
    distribution = adaptation.policy.distribution
    device = adaptation.policy.device_policy
    if storage is None or distribution is None or device is None:
        raise ValueError(
            "Native initial root preparation requires its exact storage and execution policy."
        )
    if distribution.partition.part_count != storage.partition_count:
        raise ValueError(
            "Initial owner changes require an accepted complete-state migration."
        )
    prepared = _prepared_simplex(adaptation)
    if prepared.anchor.start.uniform or prepared.anchor.start.front.growth.levels.size:
        raise ValueError(
            "Native initial root labels require their complete preclosure forest, not root-only allocation."
        )
    base = _host_epoch(prepared.state)
    owned = np.asarray(storage.entity_global_ids[-1])[
        np.asarray(storage.entity_owned[-1])
    ]
    slots = np.flatnonzero(np.isin(prepared.anchor.slot_origins, owned))
    piece = _part_arrays(base, prepared, slots)
    capacities = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                (
                    piece["vertices"].size,
                    slots.size,
                    prepared.layout.protected_edge_capacity,
                ),
                dtype=np.int64,
            ),
            tiled=False,
        )
    )
    maximum = np.max(capacities.reshape((-1, 3)), axis=0)
    vertices, cells = device.capacities(int(maximum[0]), int(maximum[1]))
    layout = AdaptiveSimplexLayout(
        prepared.layout.dimension,
        prepared.layout.ambient_dimension,
        vertex_capacity=vertices,
        cell_capacity=cells,
        protected_edge_capacity=int(maximum[2]),
        maximum_closure_iterations=adaptation.policy.maximum_closure_iterations,
        maximum_coarsening_passes=device.maximum_coarsening_passes,
    )
    state = _owned_part_state(layout, base, slots, piece)
    width = source.mesh.topological_dimension + 1
    columns = np.asarray(
        tuple(
            tuple(index for index in range(width) if index != opposite)
            for opposite in range(width)
        ),
        dtype=np.int32,
    )
    keys = np.sort(base.vertex_ids[base.cells[slots]][:, columns], axis=2)
    facet_rows = key_rows(
        entity_keys(source.mesh, width - 2), keys.reshape((-1, width - 1))
    )
    physical = storage.local_physical_boundary_facets
    if physical is None:
        raise ValueError(
            "Initial root exterior requires actual global physical-boundary receipts."
        )
    flags = np.asarray(physical)[facet_rows].reshape((slots.size, width))
    exterior = np.zeros((layout.cell_capacity, width), dtype=np.bool_)
    exterior[: slots.size] = flags
    origins = np.full((layout.cell_capacity,), -1, dtype=np.int64)
    origins[: slots.size] = prepared.anchor.slot_origins[slots]
    initial_evidence = source.collective_evidence
    if initial_evidence is None:
        raise ValueError("Native initial root allocation lost its actual source proof.")
    placement = dict(initial_evidence.logical_arrays)["closure/cell_ids"].sharding
    if not isinstance(placement, jax.sharding.NamedSharding):
        raise ValueError(
            "Native initial execution requires explicit globally named placement."
        )
    parts = AdaptiveSimplexParts(
        tuple(placement.mesh.devices.flat),
        axis_name=placement.mesh.axis_names[0],
        neighbor_pairs=tuple(
            (left, right)
            for left in range(storage.partition_count)
            for right in range(storage.partition_count)
            if left != right
        ),
    )

    def placed(value: jax.Array | np.ndarray) -> jax.Array:
        local = np.asarray(jax.device_get(value))[None]
        return jax.make_array_from_process_local_data(
            placement,
            local,
            (storage.partition_count, *local.shape[1:]),
        )

    return PartitionedAdaptiveSimplex(
        prepared,
        layout,
        parts,
        jax.tree_util.tree_map(placed, state),
        placed(exterior),
        placed(origins),
    )


def _validate_initial_partition_source(
    prepared: PreparedAdaptiveSimplex,
    partition_count: int,
    states: AdaptiveSimplexState,
    exterior: jax.Array,
    entities: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    origins: np.ndarray | jax.Array,
    /,
    *,
    layout: AdaptiveSimplexLayout,
) -> None:
    """Bind initial owner roots and constraints to the actual accepted source."""

    source = _prepared_source(prepared.adaptation.source.mesh)
    anchor = prepared.anchor.source
    if any(
        not np.array_equal(actual, expected)
        for actual, expected in (
            (anchor.vertex_ids, source.vertex_ids),
            (anchor.coordinates, source.coordinates),
            (anchor.cells.ids, source.cells.ids),
            (anchor.cells.rows, source.cells.rows),
        )
    ):
        raise ValueError(
            "Prepared source anchors differ from the accepted source content."
        )
    if (
        states.mesh.cell_ids.shape[0] != partition_count
        or exterior.shape != states.mesh.cells.shape
        or exterior.dtype != jnp.bool_
        or origins.shape != states.mesh.cell_ids.shape
    ):
        raise ValueError(
            "Initial partition shapes differ from the prepared owner layout."
        )
    if states.mesh.signature_id != layout.mesh_signature_id:
        raise ValueError(
            "Initial partition state differs from its declared owner layout."
        )
    base = _host_epoch(prepared.state)
    authoritative_exterior = np.asarray(prepared.state.mesh.facet_neighbors) < 0
    source_ids, owner = entities[-1][1:]
    if not isinstance(source_ids, np.ndarray) or not isinstance(owner, np.ndarray):
        raise ValueError("Authored initial roots require host source entity routing.")
    slot_owner = _source_forest_slot_owners(prepared, base, source_ids, owner)
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError("Initial source parts require a leading owner slice.")
        first = 0 if selection.start is None else selection.start
        last = partition_count if selection.stop is None else selection.stop
        for part in range(first, last):
            host = _host_epoch(_addressable_part_state(states, part))
            slots = np.flatnonzero(slot_owner == part)
            piece = _part_arrays(base, prepared, slots)
            expected = _host_epoch(_owned_part_state(layout, base, slots, piece))
            if any(
                not np.array_equal(actual, reference)
                for actual, reference in zip(host, expected, strict=True)
            ):
                raise ValueError(
                    "Initial owner families or constraints differ from the accepted source."
                )
            received_origins = (
                np.asarray(jax.device_get(_addressable_part_array(origins, part)))
                if isinstance(origins, jax.Array)
                else origins[part]
            )
            expected_origins = np.pad(
                prepared.anchor.slot_origins[slots],
                (0, layout.cell_capacity - slots.size),
                constant_values=-1,
            )
            if not np.array_equal(received_origins, expected_origins):
                raise ValueError(
                    "Initial owner origins differ from their exact source family."
                )
            received_exterior = np.asarray(
                jax.device_get(_addressable_part_array(exterior, part))
            )
            if not np.array_equal(
                received_exterior[: slots.size], authoritative_exterior[slots]
            ) or np.any(received_exterior[slots.size :]):
                raise ValueError(
                    "Initial exterior flags differ from the authoritative source facets."
                )


def _source_cache_digest(
    tables: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    /,
) -> str:
    """Fingerprint source routing from canonical logical leaves, not host copies."""
    return logical_array_value_collection_digest(
        {
            f"{degree}/{name}": jnp.asarray(value)
            for degree, table in enumerate(tables)
            for name, value in zip(("keys", "ids", "owners"), table, strict=True)
        }
    )


def _source_entity_tables(
    prepared: PreparedMeshAdaptation, /
) -> tuple[
    tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
]:
    """Prepare immutable source closure routing before distributed execution."""

    accepted = prepared.source
    if accepted.mesh.storage is not None:
        evidence = accepted.collective_evidence
        if evidence is None:
            raise ValueError(
                "Accepted source routing requires its collective numerical witness."
            )
        evidence.require_passed()
        return tuple(
            (keys[:count], identifiers[:count], owners[:count])
            for keys, identifiers, owners, count in zip(
                evidence.entity_keys,
                evidence.entity_ids,
                evidence.entity_owners,
                evidence.global_entity_counts,
                strict=True,
            )
        )
    source = _prepared_source(prepared.source.mesh)
    distribution = prepared.policy.distribution
    if distribution is None:
        raise ValueError("Source entity routing requires a prepared distribution.")
    cell_ids = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    order = np.argsort(cell_ids, kind="stable")
    cell_owner = np.asarray(distribution.partition.cell_owner, dtype=np.int32)[order][
        np.searchsorted(cell_ids[order], source.cells.ids)
    ]
    tables = []
    dimension = source.dimension
    for degree in range(dimension + 1):
        keys = entity_keys(source.mesh, degree)
        identifiers = np.asarray(
            source.mesh.entity_set(degree).entity_ids, dtype=np.int64
        )
        owners = np.full(
            identifiers.shape, distribution.partition.part_count, dtype=np.int32
        )
        if degree == dimension:
            owners[key_rows(keys, source.cells.ids[:, None])] = cell_owner
        else:
            columns = np.asarray(
                tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32
            )
            occurrences = np.sort(
                source.vertex_ids[source.cells.rows[:, columns]], axis=2
            )
            rows = key_rows(keys, occurrences.reshape((-1, degree + 1)))
            np.minimum.at(owners, rows, np.repeat(cell_owner, columns.shape[0]))
        if np.any(owners == distribution.partition.part_count):
            raise ValueError("An accepted source entity has no owning incident cell.")
        for values in (keys, identifiers, owners):
            values.setflags(write=False)
        tables.append((keys, identifiers, owners))
    return tuple(tables)


def _source_solver_cell_owners(
    prepared: PreparedMeshAdaptation,
    states: AdaptiveSimplexState,
    /,
) -> jax.Array:
    """Match raw active identities to the accepted scientific solver-owner bank."""

    storage = prepared.source.mesh.storage
    evidence = prepared.source.collective_evidence
    if storage is not None and isinstance(evidence, CollectiveMeshEvidence):
        count = evidence.global_entity_counts[-1]
        identifiers = evidence.entity_ids[-1][:count]
        ownership = dict(storage.logical_arrays)["cell_owners"][:count]
        if ownership.shape != identifiers.shape or ownership.dtype != jnp.int32:
            raise ValueError(
                "Accepted logical cells lost their scientific solver-owner bank."
            )
    else:
        entities = _source_entity_tables(prepared)
        identifiers, ownership = (jnp.asarray(value) for value in entities[-1][1:])
    order = jnp.argsort(identifiers, stable=True)
    table = identifiers[order]
    positions = jnp.minimum(
        jnp.searchsorted(table, states.mesh.cell_ids), table.shape[0] - 1
    )
    if not bool(
        jax.device_get(
            jnp.all(~states.mesh.cell_active | (table[positions] == states.mesh.cell_ids))
        )
    ):
        raise ValueError(
            "Prepared active forest rows are absent from their actual solver ownership bank."
        )
    return jnp.where(states.mesh.cell_active, ownership[order[positions]], -1).astype(
        jnp.int32
    )


def _source_forest_slot_owners(
    prepared: PreparedAdaptiveSimplex,
    base: _HostEpoch,
    source_ids: np.ndarray,
    source_owners: np.ndarray,
    /,
) -> np.ndarray:
    """Assign only complete declared binary and packed source trees."""
    count = prepared.anchor.cell_ids.size
    origins = prepared.anchor.slot_origins
    order = np.argsort(source_ids, kind="stable")
    positions = np.searchsorted(source_ids[order], np.maximum(origins, 0))
    active = base.cell_active[:count]
    owner = np.full((count,), -1, dtype=np.int64)
    owner[active] = source_owners[order][positions[active]]
    roots = _node_keys(base.parents[:count], base.generations[:count])[:, 0]
    for root in np.unique(roots):
        member = roots == root
        owners = np.unique(owner[member & active])
        if owners.size != 1:
            raise MeshingFailure(
                MeshingFailureCategory.LINEAGE_FAILED,
                "A distributed owner does not own the complete declared binary source tree.",
                stage="distributed-source-ownership",
                entity_ids=(int(base.cell_ids[root]),),
            )
        owner[member] = owners[0]
    for part in np.unique(owner):
        _owned_uniform_refinement(
            prepared.anchor.start.uniform_refinement,
            base.cell_ids[:count][active & (owner == part)],
            prepared.anchor.start.records,
        )
    return owner


def partition_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> PartitionedAdaptiveSimplex:
    if not isinstance(prepared, PreparedAdaptiveSimplex):
        raise TypeError("prepared must be PreparedAdaptiveSimplex.")
    allowance = _phase_allowance(prepared.adaptation, prepared.execution_evidence)
    with _uniform_execution(allowance) as budget:
        _retain_preparation(prepared)
        partitioned = _partition_adaptive_simplex(prepared, devices=devices)
    if budget.evidence is None:
        return partitioned
    return _bind_partitioned_preparation_receipt(
        partitioned,
        NativeExecutionRecord(
            budget.evidence,
            preparation_evidence=prepared.execution_evidence,
            owner_id=prepared.adaptation.prepared_id,
        ),
    )


def _partition_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> PartitionedAdaptiveSimplex:
    """Split one prepared epoch by its distribution into stacked per-part states.

    Each part holds its owned active cells and their vertices with the global
    IDs and cursors of the epoch; ``devices`` default to the first global devices.
    Initial source forests and protections are validated against accepted source
    content, including complete declared packed sibling trees.
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
    origins = prepared.anchor.slot_origins
    native = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    owners = np.asarray(distribution.partition.cell_owner, dtype=np.int64)
    slot_owner = _source_forest_slot_owners(prepared, base, native, owners)
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
        maximum_closure_iterations=prepared.layout.maximum_closure_iterations,
        maximum_coarsening_passes=prepared.layout.maximum_coarsening_passes,
    )
    states = []
    slot_origins = np.full((part_count, cell_capacity), -1, dtype=np.int64)
    source_exterior = np.zeros(
        (part_count, cell_capacity, prepared.layout.dimension + 1), dtype=np.bool_
    )
    global_exterior = np.asarray(prepared.state.mesh.facet_neighbors) < 0
    for part, (slots, piece) in enumerate(zip(owned, pieces, strict=True)):
        size = slots.size
        states.append(_owned_part_state(layout, base, slots, piece))
        slot_origins[part, :size] = origins[slots]
        source_exterior[part, :size] = global_exterior[slots]
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *states)
    neighbor_pairs = tuple(
        (left, right)
        for left in range(part_count)
        for right in range(part_count)
        if left != right
        and np.intersect1d(
            base.vertex_ids[pieces[left]["vertices"]],
            base.vertex_ids[pieces[right]["vertices"]],
            assume_unique=True,
        ).size
    )
    parts = AdaptiveSimplexParts(chosen, neighbor_pairs=neighbor_pairs)
    sharding = jax.sharding.NamedSharding(
        parts.mesh, jax.sharding.PartitionSpec(parts.axis_name)
    )

    def placed(value: jax.Array) -> jax.Array:
        def local(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "Root placement requires its concrete addressable array rectangle."
                )
            return np.asarray(jax.device_get(value[index]))

        return jax.make_array_from_callback(value.shape, sharding, local)

    return PartitionedAdaptiveSimplex(
        prepared,
        layout,
        parts,
        jax.tree_util.tree_map(placed, stacked),
        jax.make_array_from_callback(
            source_exterior.shape, sharding, lambda index: source_exterior[index]
        ),
        slot_origins,
    )


class _LocalCommit(NamedTuple):
    partition_index: int
    source: CellMesh
    committed: _Committed
    cell_exterior: np.ndarray | None = None


def _addressable_part_state(
    states: AdaptiveSimplexState, partition_index: int, /
) -> AdaptiveSimplexState:
    """Extract one process-addressable part without touching remote shards."""

    def local(value: jax.Array) -> jax.Array:
        return _addressable_part_array(value, partition_index)

    return jax.tree_util.tree_map(local, states)


def _addressable_part_array(value: jax.Array, partition_index: int, /) -> jax.Array:
    for shard in value.addressable_shards:
        selection = shard.index[0]
        if isinstance(selection, slice):
            first = 0 if selection.start is None else selection.start
            last = value.shape[0] if selection.stop is None else selection.stop
            if first <= partition_index < last:
                return shard.data[partition_index - first]
    raise ValueError("The requested adaptive part is not process-addressable.")


def _source_entity_values(
    entities: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    degree: int,
    queries: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower only the requested source entities from globally logical routing."""
    keys, identifiers, owners = entities[degree]
    table = jnp.asarray(keys, dtype=jnp.int64)
    order = jnp.lexsort(table.T[::-1])
    positions = _key_positions(table[order], jnp.asarray(queries, dtype=jnp.int64))
    if bool(jax.device_get(jnp.any(positions < 0))):
        raise ValueError(
            "A local source entity is absent from accepted logical topology."
        )
    rows = order[positions]
    return (
        np.asarray(jax.device_get(jnp.asarray(identifiers)[rows]), dtype=np.int64),
        np.asarray(jax.device_get(jnp.asarray(owners)[rows]), dtype=np.int32),
    )


def _source_closure(
    partitioned: PartitionedAdaptiveSimplex, root_ids: np.ndarray, /
) -> _Source:
    """Lower only the active predecessor cells consumed by one local closure."""

    accepted = partitioned.prepared.adaptation.source
    cell_ids = np.unique(root_ids[root_ids >= 0])
    storage = accepted.mesh.storage
    if accepted.mesh.periodic_topology is not None:
        from ._collective_geometry import periodic_source_closure_cells

        cell_ids = periodic_source_closure_cells(
            accepted.mesh,
            cell_ids,
            cell_capacity=partitioned.layout.cell_capacity,
            vertex_capacity=partitioned.layout.vertex_capacity,
        )
    if storage is None:
        original = partitioned.prepared.anchor.source
        rows = np.searchsorted(original.cells.ids, cell_ids)
        if np.any(rows >= original.cells.ids.size) or not np.array_equal(
            original.cells.ids[rows], cell_ids
        ):
            raise ValueError(
                "Neighborhood ancestry references absent accepted source cells."
            )
        corners = original.vertex_ids[original.cells.rows[rows]]
        identifiers = np.unique(corners)
        points = original.coordinates[np.searchsorted(original.vertex_ids, identifiers)]
    else:
        arrays = dict(storage.logical_arrays)
        corners, present = _local_logical_lookup(
            arrays["cell_global_ids"][: storage.global_entity_counts[-1]],
            arrays["cell_vertices"][: storage.global_entity_counts[-1]],
            cell_ids,
        )
        if not np.all(present):
            raise ValueError(
                "Neighborhood ancestry references absent accepted logical source cells."
            )
        identifiers = np.unique(corners)
        points, present = _local_logical_lookup(
            arrays["vertex_global_ids"][: storage.global_entity_counts[0]],
            arrays["coordinates"][: storage.global_entity_counts[0]],
            identifiers,
        )
        if not np.all(present):
            raise ValueError(
                "Accepted logical cell ancestry references absent source vertices."
            )
    block = accepted.mesh.blocks[0]
    cells = np.searchsorted(identifiers, corners).astype(np.int32)
    blocks = (CellBlock(block.name, block.cell_kind, cells, global_ids=cell_ids),)
    probe = CellMesh(points, blocks, vertex_global_ids=identifiers)
    entity_ids = {
        degree: _source_entity_values(
            partitioned.source_entities, degree, entity_keys(probe, degree)
        )[0]
        for degree in range(1, accepted.mesh.topological_dimension)
    }
    mesh = CellMesh(
        points,
        blocks,
        vertex_global_ids=identifiers,
        entity_global_ids=entity_ids,
    )
    if accepted.mesh.periodic_topology is not None:
        from ._collective_geometry import project_source_periodic_topology

        periodic = project_source_periodic_topology(accepted.mesh, mesh)
        mesh = CellMesh(
            points,
            blocks,
            vertex_global_ids=identifiers,
            entity_global_ids=entity_ids,
            periodic_topology=periodic,
        )
    return _prepared_source(mesh)


def _predecessor_reverse_supports(
    partitioned: PartitionedAdaptiveSimplex,
    source: _Source,
    target_vertices: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the retained midpoint DAG for vertices removed by this epoch."""
    lineage = partitioned.prepared.anchor.start.uniform_refinement
    if lineage is not None:
        from ._collective_organization import _original_vertex_ancestry

        support_ids, support_weights, valid = jax.vmap(
            lambda state: _original_vertex_ancestry(state, lineage),
        )(partitioned.states)
        if not bool(jax.device_get(jnp.all(valid))):
            raise ValueError(
                "Packed predecessor lost its actual full barycentric source support."
            )
        table = partitioned.states.mesh.vertex_ids.reshape((-1, 1))
        keys, order, count = _canonical_rows(
            table,
            partitioned.states.mesh.vertex_active.reshape(-1),
        )
        positions = _key_positions(keys[:count], jnp.asarray(source.vertex_ids[:, None]))
        if bool(jax.device_get(jnp.any(positions < 0))):
            raise ValueError("Packed source closure lost a retained predecessor vertex.")
        selected = order[positions]
        width = source.dimension + 1
        _uniform_charge(
            source.vertex_ids.size * width, source.vertex_ids.size * width * 16
        )
        ids = np.array(
            jax.device_get(support_ids.reshape((-1, width))[selected]),
            dtype=np.int64,
            copy=True,
        )
        weights = np.array(
            jax.device_get(support_weights.reshape((-1, width))[selected]),
            dtype=np.float64,
            copy=True,
        )
        present = np.isin(source.vertex_ids, target_vertices)
        ids[present] = -1
        weights[present] = 0.0
        ids[present, 0] = source.vertex_ids[present]
        weights[present, 0] = 1.0
        rows = key_rows(
            source.vertex_ids[:, None], np.maximum(ids, 0).reshape((-1, 1))
        ).reshape(ids.shape)
        rows = np.where(weights != 0.0, rows, -1)
        if np.any((weights != 0.0) & (rows < 0)):
            raise ValueError("Packed reverse support lost an original scientific vertex.")
        return rows, weights
    initial = partitioned.states
    vertices = initial.mesh.vertex_ids.shape[1]
    parents = initial.vertex_parents
    identifiers = jnp.take_along_axis(
        initial.mesh.vertex_ids,
        jnp.clip(parents.reshape((parents.shape[0], -1)), 0, vertices - 1),
        axis=1,
    ).reshape(parents.shape)
    identifiers = jnp.where(parents >= 0, identifiers, -1)
    table = initial.mesh.vertex_ids.reshape((-1, 1))
    valid = initial.mesh.vertex_active.reshape((-1,))
    keys, order, count = _canonical_rows(table, valid)
    positions = _key_positions(
        keys[:count], jnp.asarray(source.vertex_ids[:, None], dtype=jnp.int64)
    )
    if bool(jax.device_get(jnp.any(positions < 0))):
        raise ValueError(
            "Source closure vertices are absent from retained predecessor state."
        )
    selected = order[positions]
    parent_ids = np.asarray(
        jax.device_get(identifiers.reshape((-1, 2))[selected]), dtype=np.int64
    )
    levels = np.asarray(
        jax.device_get(initial.vertex_levels.reshape((-1,))[selected]), dtype=np.int64
    )
    removed = np.flatnonzero(~np.isin(source.vertex_ids, target_vertices))
    steps = []
    for level in np.sort(np.unique(levels[removed]))[::-1]:
        slots = removed[levels[removed] == level]
        references = key_rows(
            source.vertex_ids[:, None], parent_ids[slots].reshape((-1, 1))
        ).reshape((-1, 2))
        if np.any(references < 0):
            raise ValueError(
                "A removed predecessor vertex lacks its accepted midpoint support."
            )
        steps.append((slots, references[:, 0], references[:, 1]))
    return _reverse_supports(steps, source.vertex_ids.size, source.dimension)


def _neighborhood_commit(
    partitioned: PartitionedAdaptiveSimplex,
    local: _LocalCommit,
    neighborhood: SimplexNeighborhoodWorkset,
    /,
) -> _LocalCommit:
    """Bind received ghost packets to exact source-supported lineage and geometry."""

    def host(value: jax.Array) -> np.ndarray:
        return np.asarray(
            jax.device_get(_addressable_part_array(value, local.partition_index))
        )

    valid = host(neighborhood.cell_valid).astype(np.bool_)
    cell_ids = host(neighborhood.cell_ids)[valid]
    order = np.argsort(cell_ids, kind="stable")
    cell_ids = cell_ids[order]
    global_rows = host(neighborhood.cell_vertices)[valid][order]
    corners = host(neighborhood.cell_coordinates)[valid][order]
    roots = host(neighborhood.root_cell_ids)[valid][order]
    predecessor_cells = host(neighborhood.source_cell_ids)[valid][order]
    supports = host(neighborhood.source_vertex_ids)[valid][order]
    coefficients = host(neighborhood.source_weights)[valid][order]
    exterior = host(neighborhood.cell_exterior)[valid][order]
    source = _source_closure(partitioned, predecessor_cells.reshape((-1,)))
    dimension = source.dimension
    width = dimension + 1
    vertices, first = np.unique(global_rows.reshape((-1,)), return_index=True)
    points = corners.reshape((-1, corners.shape[-1]))[first]
    stencil_sources = supports.reshape((-1, width))[first]
    stencil_weights = coefficients.reshape((-1, width))[first]
    stencil_valid = (stencil_sources >= 0) & (stencil_weights > 0.0)
    stencil_sources = np.where(stencil_valid, stencil_sources, -1)
    new = ~np.isin(vertices, source.vertex_ids)
    issued = vertices[new]
    all_ids = np.concatenate((source.vertex_ids, issued))
    compact = np.empty(vertices.shape, dtype=np.int64)
    compact[~new] = np.searchsorted(source.vertex_ids, vertices[~new])
    compact[new] = source.vertex_ids.size + np.arange(issued.size, dtype=np.int64)
    target_rows = np.searchsorted(vertices, global_rows)
    split_sources = np.where(
        stencil_valid[new],
        np.searchsorted(source.vertex_ids, np.maximum(stencil_sources[new], 0)),
        -1,
    )
    empty = np.zeros((0,), dtype=np.int64)
    empty_cells = source.cells._replace(
        ids=empty,
        rows=np.zeros((0, width), dtype=np.int64),
        tuples=np.zeros((0, width), dtype=np.int64),
        tags=empty,
        blocks=empty,
        origins=empty,
        generations=empty,
    )
    removed_rows, target_columns = np.nonzero(
        (predecessor_cells >= 0) & (roots[:, None] < 0)
    )
    removed_ids = predecessor_cells[removed_rows, target_columns]
    link_targets = cell_ids[removed_rows]
    reverse_sources, reverse_weights = _predecessor_reverse_supports(
        partitioned, source, vertices
    )
    coarsening = _Coarsening(
        empty_cells,
        removed_ids,
        link_targets,
        empty,
        np.flatnonzero(~np.isin(source.vertex_ids, vertices)),
        reverse_sources,
        reverse_weights,
        0,
        empty,
    )
    cells = _Cells(
        cell_ids,
        compact[target_rows],
        compact[target_rows],
        np.zeros(cell_ids.shape, dtype=np.int64),
        np.zeros(cell_ids.shape, dtype=np.int64),
        np.where(roots >= 0, roots, cell_ids),
        np.zeros(cell_ids.shape, dtype=np.int64),
    )
    tables = tuple(
        (
            _simplex_keys(source.cells.rows, degree + 1),
            _simplex_keys(cells.rows, degree + 1),
        )
        for degree in range(1, dimension)
    )
    start = partitioned.prepared.anchor.start
    relations = _relations(
        source,
        start,
        cells,
        (split_sources, stencil_weights[new]),
        coarsening,
        tables,
    )
    synthetic = np.concatenate(
        (source.vertex_ids, start.next_vertex + np.arange(issued.size, dtype=np.int64))
    )

    def actual(keys: np.ndarray) -> np.ndarray:
        return all_ids[np.searchsorted(synthetic, keys)]

    relations = tuple(
        relation._replace(
            source_keys=actual(relation.source_keys),
            target_keys=actual(relation.target_keys),
        )
        if relation.dimension < dimension
        else relation
        for relation in relations
    )
    refined = roots >= 0
    root_rows = np.searchsorted(source.cells.ids, roots[refined])
    parent_corners = source.vertex_ids[source.cells.rows[root_rows]]
    witnesses = NestedReferenceWitnesses(
        cell_ids[refined],
        roots[refined],
        nested_reference_vertices(
            stencil_sources[target_rows[refined]],
            stencil_weights[target_rows[refined]],
            parent_corners,
        ),
    )
    removed_source_rows = np.searchsorted(source.cells.ids, removed_ids)
    merged_target_rows = np.searchsorted(cell_ids, link_targets)
    reverse_ids = np.where(
        reverse_sources >= 0, source.vertex_ids[np.maximum(reverse_sources, 0)], -1
    )
    coarsened_witnesses = NestedReferenceWitnesses(
        removed_ids,
        link_targets,
        nested_reference_vertices(
            reverse_ids[source.cells.rows[removed_source_rows]],
            reverse_weights[source.cells.rows[removed_source_rows]],
            global_rows[merged_target_rows],
        ),
    )
    block = local.committed.edit.blocks[0]
    if not isinstance(block, TopologyEditBlock):
        raise ValueError("Affine simplex closure requires a fixed-family topology edit.")
    edit = local.committed.edit._replace(
        coordinates=points,
        vertex_global_ids=vertices,
        blocks=(block._replace(cells=target_rows.astype(np.int32), cell_ids=cell_ids),),
        stencil_sources=stencil_sources,
        stencil_weights=stencil_weights,
        stencil_valid=stencil_valid,
        relations=relations,
        refinement=witnesses,
        coarsening=coarsened_witnesses,
    )
    return _LocalCommit(
        local.partition_index,
        source.mesh,
        local.committed._replace(edit=edit),
        exterior,
    )


def _local_anchor(
    original: _Anchor,
    entities: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    initial: _HostEpoch,
    /,
) -> _Anchor:
    """Lower the source closure while retaining every scientific entity ID."""

    global_source = original.source.mesh
    declarations = (
        original.source_blocks
        if original.source_blocks is not None
        else tuple(global_source.blocks)
    )
    vertex_count, cell_count = (int(value) for value in initial.cursors[:2])
    vertices = initial.vertex_ids[:vertex_count]
    active = np.flatnonzero(initial.cell_active[:cell_count])
    rows = initial.cells[active]
    raw_groups = np.unique(initial.blocks[active])
    if np.any(raw_groups < 0) or np.any(raw_groups >= len(declarations)):
        raise ValueError(
            "A local source cell lacks its actual declared presentation bank."
        )
    block_indices = np.full(len(declarations), -1, dtype=np.int64)
    blocks = []
    for local_group, raw_group in enumerate(raw_groups):
        definition = declarations[int(raw_group)]
        selected = active[initial.blocks[active] == raw_group]
        blocks.append(
            CellBlock(
                definition.name,
                definition.cell_kind,
                initial.cells[selected],
                global_ids=initial.cell_ids[selected],
            )
        )
        block_indices[raw_group] = local_group
    probe = CellMesh(
        initial.coordinates[:vertex_count], tuple(blocks), vertex_global_ids=vertices
    )
    entity_ids = {
        degree: _source_entity_values(entities, degree, entity_keys(probe, degree))[0]
        for degree in range(1, global_source.topological_dimension)
    }
    mesh = CellMesh(
        initial.coordinates[:vertex_count],
        probe.blocks,
        vertex_global_ids=vertices,
        entity_global_ids=entity_ids,
    )
    source = _prepared_source(mesh)
    cells = _Cells(
        initial.cell_ids[active],
        rows,
        initial.tuples[active],
        initial.tags[active],
        block_indices[initial.blocks[active]],
        initial.cell_ids[active],
        initial.generations[active],
    )
    dimension = source.dimension
    empty = np.zeros((0,), dtype=np.int64)
    growth = _Growth(
        np.zeros((0, dimension + 1), dtype=np.int64),
        np.zeros((0, dimension + 1), dtype=np.float64),
        empty,
    )
    record_slots = np.flatnonzero(
        ~initial.cell_active[:cell_count] & ~initial.retired[:cell_count]
    )
    if np.any(initial.blocks[record_slots] < 0) or np.any(
        initial.blocks[record_slots] >= block_indices.size
    ):
        raise ValueError(
            "A local source record lacks its actual declared presentation bank."
        )
    if np.any(block_indices[initial.blocks[record_slots]] < 0):
        raise ValueError(
            "A local binary source family lacks a resident descendant presentation bank."
        )
    records = _Records(
        initial.cell_ids[record_slots],
        block_indices[initial.blocks[record_slots]],
        initial.cells[record_slots],
        initial.tuples[record_slots],
        initial.tags[record_slots],
        initial.cell_ids[initial.children[record_slots]],
        initial.bisection_vertices[record_slots],
    )
    front = _Front(
        cells,
        growth,
        np.zeros((0, 2), dtype=np.int64),
        empty,
        vertex_count,
        int(initial.cursors[3]),
    )
    start = _Start(
        front,
        records,
        np.zeros((active.size,), dtype=np.bool_),
        int(initial.cursors[2]),
        original.start.compatible,
        original.start.incompatible,
        False,
        _owned_retired_entities(
            source,
            tuple(original.source.vertex_ids[keys] for keys, _ in original.start.retired),
            tuple(ids for _, ids in original.start.retired),
        ),
        _owned_uniform_refinement(
            original.start.uniform_refinement,
            cells.ids,
            records,
        ),
    )
    ids = initial.cell_ids[:cell_count]
    origins = np.where(initial.cell_active[:cell_count], ids, -1)
    return _Anchor(
        source,
        start,
        ids,
        origins,
        np.full(ids.shape, -1, dtype=np.int64),
        active,
        block_indices,
        tuple(blocks),
    )


def _rebind_issued_vertices(
    committed: _Committed, anchor: _Anchor, host: _HostEpoch, /
) -> _Committed:
    """Bind host witnesses to canonical IDs issued by distributed ordering."""

    count = int(host.cursors[0])
    original = anchor.source.vertex_ids.size
    synthetic = np.concatenate(
        (
            anchor.source.vertex_ids,
            anchor.start.next_vertex + np.arange(count - original, dtype=np.int64),
        )
    )
    actual = host.vertex_ids[:count]

    def identifiers(values: np.ndarray | jax.Array) -> np.ndarray:
        array = np.asarray(values, dtype=np.int64)
        valid = array >= 0
        present = valid & np.isin(array, actual)
        positions = np.searchsorted(synthetic, np.maximum(array, 0))
        safe = np.minimum(positions, synthetic.size - 1)
        provisional = (
            valid & ~present & (positions < synthetic.size) & (synthetic[safe] == array)
        )
        return np.where(valid, np.where(provisional, actual[safe], array), -1)

    edit = committed.edit
    rebound = edit._replace(
        vertex_global_ids=identifiers(edit.vertex_global_ids),
        relations=tuple(
            relation._replace(target_keys=identifiers(relation.target_keys))
            if relation.dimension < anchor.source.dimension
            else relation
            for relation in edit.relations
        ),
        prescribed_entity_ids=tuple(
            prescribed._replace(keys=identifiers(prescribed.keys))
            for prescribed in edit.prescribed_entity_ids
        ),
    )
    hierarchy = committed.hierarchy
    uniform = hierarchy.uniform_refinement
    if uniform is not None:
        arrays = uniform.host_arrays()
        for name in ("parent_rows", "parent_vertices", "child_vertices"):
            arrays[name] = identifiers(arrays[name])
        uniform = BisectionUniformRefinement(
            uniform.dimension, *arrays.values(), source=uniform.source
        )
    forest = BisectionHierarchy(
        hierarchy.dimension,
        hierarchy.cell_global_ids,
        identifiers(hierarchy.ordered_vertices),
        hierarchy.tags,
        hierarchy.generations,
        scientific_cell_ids=hierarchy.scientific_cell_ids,
        scientific_block_ids=hierarchy.scientific_block_ids,
        record_parent_ids=hierarchy.record_parent_ids,
        record_parent_blocks=hierarchy.record_parent_blocks,
        record_parent_rows=identifiers(hierarchy.record_parent_rows),
        record_parent_vertices=identifiers(hierarchy.record_parent_vertices),
        record_parent_tags=hierarchy.record_parent_tags,
        record_child_ids=hierarchy.record_child_ids,
        record_vertex_ids=identifiers(hierarchy.record_vertex_ids),
        retired_entity_keys=tuple(
            identifiers(keys) for keys in hierarchy.retired_entity_keys
        ),
        retired_entity_ids=hierarchy.retired_entity_ids,
        next_vertex_id=int(host.cursors[2]),
        next_cell_id=int(host.cursors[3]),
        uniform_refinement=uniform,
    )
    return committed._replace(edit=rebound, hierarchy=forest)


def _require_exact_edge_restrictions(
    host: _HostEpoch, initial_vertices: int, /
) -> np.ndarray:
    """Resolve sampled midpoint residuals; exact maps retain nonzero roundoff signs."""

    count = int(host.cursors[0])
    if not initial_vertices <= count <= host.coordinates.shape[0]:
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "The affine restriction witness has invalid allocated vertex bounds.",
            stage="distributed-affine-witness",
        )
    if not np.all(np.isfinite(host.coordinates[:count])):
        raise MeshingFailure(
            MeshingFailureCategory.QUALITY_REJECTED,
            "The affine restriction witness contains non-finite coordinates.",
            stage="distributed-affine-witness",
        )
    slots = np.arange(initial_vertices, count, dtype=np.int64)
    if not slots.size:
        return np.zeros((0, host.coordinates.shape[1]), dtype=np.int8)
    endpoints = host.vertex_parents[slots]
    if (
        np.any(endpoints < 0)
        or np.any(endpoints >= slots[:, None])
        or np.any(endpoints[:, 0] == endpoints[:, 1])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "An issued vertex lacks an earlier, distinct parent edge.",
            stage="distributed-affine-witness",
        )
    shape = (slots.size, host.coordinates.shape[1], 2)
    first = np.zeros(shape, dtype=np.float64)
    second = np.ones(shape, dtype=np.float64)
    middle = np.full(shape, 0.5, dtype=np.float64)
    first[..., 1] = host.coordinates[endpoints[:, 0]]
    second[..., 1] = host.coordinates[endpoints[:, 1]]
    middle[..., 1] = host.coordinates[slots]
    try:
        signs = exact_orient2d(first, second, middle)
    except MeshcoreUnavailableError as error:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "The required native exact midpoint backend is unavailable: " + str(error),
            stage="distributed-affine-witness",
        ) from error
    except (ValueError, MeshcoreError) as error:
        raise MeshingFailure(
            MeshingFailureCategory.QUALITY_REJECTED,
            "The native exact midpoint witness could not be certified: " + str(error),
            stage="distributed-affine-witness",
        ) from error
    expected = (
        0.5 * host.coordinates[endpoints[:, 0]] + 0.5 * host.coordinates[endpoints[:, 1]]
    )
    if not np.array_equal(host.coordinates[slots], expected):
        raise MeshingFailure(
            MeshingFailureCategory.QUALITY_REJECTED,
            "Issued coordinates do not match their declared midpoint sampling operation.",
            stage="distributed-reference-restriction",
        )
    return np.asarray(signs, dtype=np.int8)


def _require_partitioned_exact_restrictions(
    layout: AdaptiveSimplexLayout,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    /,
) -> tuple[jax.Array, jax.Array]:
    """Consume native decisions and retain every resolved source-sample residual."""

    failure: MeshingFailure | None = None
    valid_packets: dict[int, np.ndarray] = {}
    sign_packets: dict[int, np.ndarray] = {}
    try:
        for shard in states.mesh.cell_ids.addressable_shards:
            selection = shard.index[0]
            if not isinstance(selection, slice):
                raise ValueError(
                    "Adaptive part shards require a leading partition slice."
                )
            first = 0 if selection.start is None else selection.start
            last = (
                states.mesh.cell_ids.shape[0]
                if selection.stop is None
                else selection.stop
            )
            for part in range(first, last):
                host = _host_epoch(_addressable_part_state(states, part))
                cursor = _addressable_part_array(initial_states.cursors, part)
                initial_vertices = int(jax.device_get(cursor[0]))
                require_committed_status(
                    host.flags,
                    host.coordinates,
                    host.cells[host.cell_active],
                    DeviceEpoch.BISECTION,
                )
                signs = _require_exact_edge_restrictions(host, initial_vertices)
                valid = np.zeros((layout.vertex_capacity,), dtype=np.bool_)
                packet = np.zeros(
                    (layout.vertex_capacity, layout.ambient_dimension), dtype=np.int8
                )
                count = int(host.cursors[0])
                valid[initial_vertices:count] = True
                packet[initial_vertices:count] = signs
                valid_packets[part] = valid
                sign_packets[part] = packet
    except MeshingFailure as error:
        failure = error
    except ValueError as error:
        failure = MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            str(error),
            stage="distributed-affine-witness",
        )
    _require_collective_local_success(failure)
    sharding = states.mesh.cell_ids.sharding

    def placed(packets: dict[int, np.ndarray], shape: tuple[int, ...]) -> jax.Array:
        def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "Native proof placement requires its concrete addressable array rectangle."
                )
            leading = index[0]
            first = 0 if leading.start is None else leading.start
            last = shape[0] if leading.stop is None else leading.stop
            return np.stack(
                tuple(packets[part][index[1:]] for part in range(first, last))
            )

        return jax.make_array_from_callback(shape, sharding, addressable)

    shape = (states.mesh.cell_ids.shape[0], layout.vertex_capacity)
    return placed(valid_packets, shape), placed(
        sign_packets, (*shape, layout.ambient_dimension)
    )


def _scientific_preparation_source(
    preparation: PreparedMeshAdaptation, /
) -> CellMeshingResult | InitialCollectiveMeshEvidence:
    request = preparation.request
    hierarchy = request.hierarchy if isinstance(request, MarkedMeshAdaptation) else None
    if hierarchy is not None and not isinstance(hierarchy, BisectionHierarchy):
        raise TypeError(
            "Packed collective source requires its actual bisection hierarchy."
        )
    lineage = None if hierarchy is None else hierarchy.uniform_refinement
    return require_original_meshing_source(
        preparation.source if lineage is None else lineage.source,
    )


def _packed_publication_states(
    preparation: PreparedMeshAdaptation,
    layout: AdaptiveSimplexLayout,
    initial: AdaptiveSimplexState,
    compiled: AdaptiveSimplexState,
    exterior: jax.Array,
    lineage: BisectionUniformRefinement | None,
    /,
    *,
    raw_source_blocks: tuple[CellBlock, ...],
    initial_solver_cell_owners: jax.Array,
) -> tuple[AdaptiveSimplexState, jax.Array, tuple[CellBlock, ...]]:
    if lineage is None:
        return compiled, exterior, raw_source_blocks
    limits = preparation.policy.limits
    allowance = _UniformAllowance(
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cells,
        limits.maximum_vertices,
        limits.maximum_scratch_bytes,
        limits.maximum_wall_seconds,
        limits.maximum_cavity_cells,
    )
    declarations = list(raw_source_blocks)
    declared_ids = {block.block_id: index for index, block in enumerate(declarations)}
    original_blocks = []
    for block in _scientific_simplex_blocks(lineage.source.mesh):
        index = declared_ids.get(block.block_id)
        if index is None:
            index = len(declarations)
            declarations.append(block)
            declared_ids[block.block_id] = index
        original_blocks.append(index)
    publication_blocks = np.asarray(original_blocks, dtype=np.int64)
    with _uniform_execution(allowance):
        published, boundary = _run_packed_publication_states(
            preparation,
            layout,
            initial,
            compiled,
            exterior,
            lineage,
            raw_source_blocks=raw_source_blocks,
            initial_solver_cell_owners=initial_solver_cell_owners,
            publication_blocks=publication_blocks,
        )
        return (
            _physical_packed_publication_states(published, lineage),
            boundary,
            tuple(declarations),
        )


def _physical_packed_publication_states(
    states: AdaptiveSimplexState,
    lineage: BisectionUniformRefinement,
    /,
) -> AdaptiveSimplexState:
    """Release exact-construction scratch, retaining the real source owner."""
    workspace = current_native_host_workspace()
    if workspace is None:
        raise RuntimeError(
            "Physical packed publication lost its original host storage owner."
        )
    workspace.retain_owner(lineage.source)
    retained = workspace.bound
    try:
        physical = _run_physical_packed_publication_states(states, lineage)
        workspace.retain_owner(physical)
        return physical
    finally:
        workspace.set_bound(retained)


def _run_physical_packed_publication_states(
    states: AdaptiveSimplexState,
    lineage: BisectionUniformRefinement,
    /,
) -> AdaptiveSimplexState:
    """Evaluate the retained full action chain, not merged binary64 weights."""
    from ..discretization._cell_geometry import (
        _require_full_p1_source,
        BarycentricCellGeometryElement,
    )
    from ..discretization._coordinate_enclosure import (
        coordinate_corner_images,
        prepared_coordinate_source_bank,
        rounded_point,
    )
    from ._collective_geometry import (
        _collective_action_stacks,
        _original_p1_vertex_images,
    )

    _original_p1_vertex_images(lineage.source)
    original = lineage.source
    elements, routes, _ = original.geometry.resolve(original.mesh)
    bank = prepared_coordinate_source_bank(original.geometry)
    sources = {
        int(identifier): (_require_full_p1_source(element), np.asarray(route)[row])
        for block, element, route in zip(
            original.mesh.blocks, elements, routes, strict=True
        )
        for row, identifier in enumerate(block.global_ids)
    }
    roots, counts, stacks = _collective_action_stacks(states, lineage)
    coordinates: dict[int, np.ndarray] = {}
    incident = {}
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "Physical publication requires addressable owner rectangles."
            )
        first = 0 if selection.start is None else selection.start
        last = states.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        for part in range(first, last):
            state = _addressable_part_state(states, part)
            host = _host_epoch(state)
            root_ids = np.asarray(
                jax.device_get(_addressable_part_array(roots, part)), dtype=np.int64
            )
            action_counts = np.asarray(
                jax.device_get(_addressable_part_array(counts, part)), dtype=np.int32
            )
            actions = np.asarray(
                jax.device_get(_addressable_part_array(stacks, part)), dtype=np.float64
            )
            _uniform_charge(
                state.mesh.coordinates.size + actions.size,
                state.mesh.coordinates.size * 8
                + actions.nbytes
                + root_ids.nbytes
                + action_counts.nbytes,
            )
            carrier = np.array(host.coordinates, copy=True)
            for slot in np.flatnonzero(host.cell_active):
                root = int(root_ids[slot])
                if root not in sources:
                    raise ValueError(
                        "A physical action has no actual original SCI source element."
                    )
                element, route = sources[root]
                for action in reversed(actions[slot, : action_counts[slot]]):
                    element = BarycentricCellGeometryElement(element, action)
                images = coordinate_corner_images(
                    element, tuple(bank[int(index)] for index in route)
                )
                if images is None:
                    raise ValueError(
                        "A full P1 action lost its exact scientific coefficient law."
                    )
                for vertex, image in zip(host.cells[slot], images, strict=True):
                    identifier = int(host.vertex_ids[vertex])
                    if identifier in incident and incident[identifier] != image:
                        raise ValueError(
                            "Incident full actions disagree on their exact physical carrier."
                        )
                    incident[identifier] = image
                    carrier[vertex] = rounded_point(image)
            coordinates[part] = carrier

    def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
        if index is None:
            raise ValueError("Physical publication requires concrete addressable slices.")
        leading = index[0]
        first = 0 if leading.start is None else leading.start
        last = states.mesh.coordinates.shape[0] if leading.stop is None else leading.stop
        return np.stack(
            tuple(coordinates[part][index[1:]] for part in range(first, last))
        )

    physical = jax.make_array_from_callback(
        states.mesh.coordinates.shape, states.mesh.coordinates.sharding, addressable
    )
    return eqx.tree_at(lambda value: value.mesh.coordinates, states, physical)


def _run_packed_publication_states(
    preparation: PreparedMeshAdaptation,
    layout: AdaptiveSimplexLayout,
    initial: AdaptiveSimplexState,
    compiled: AdaptiveSimplexState,
    exterior: jax.Array,
    lineage: BisectionUniformRefinement | None,
    /,
    *,
    raw_source_blocks: tuple[CellBlock, ...],
    initial_solver_cell_owners: jax.Array,
    publication_blocks: np.ndarray,
) -> tuple[AdaptiveSimplexState, jax.Array]:
    """Derive the host inverse independently; never mutate compiled receipts."""
    if lineage is None:
        return compiled, exterior
    original = _source_simplex_on_states(
        preparation,
        layout,
        initial,
        raw_source_blocks=raw_source_blocks,
        uniform_refinement=lineage,
    )
    entities = _source_entity_tables(preparation)
    source_evidence = preparation.source.collective_evidence
    source_ids = (
        source_evidence.entity_ids[-1][: source_evidence.global_entity_counts[-1]]
        if source_evidence is not None
        else jnp.asarray(
            np.concatenate(
                [np.asarray(block.global_ids) for block in preparation.source.mesh.blocks]
            )
        )
    )
    if bool(
        jax.device_get(
            jnp.any(
                initial.mesh.cell_active & ~jnp.isin(initial.mesh.cell_ids, source_ids)
            )
        )
    ):
        seed = _logical_publication(
            preparation.source,
            initial,
            initial_solver_cell_owners,
            exterior,
            layout,
            original_source=lineage.source,
            uniform_refinement=lineage,
            numeric_version=f"adaptation:{preparation.prepared_id}",
        )
        entities = tuple(
            (keys[:count], identifiers[:count], owners[:count])
            for keys, identifiers, owners, count in zip(
                seed.entity_keys,
                seed.entity_ids,
                seed.entity_owners,
                seed.counts,
                strict=True,
            )
        )
    arrays = lineage.host_arrays()
    packets: dict[int, AdaptiveSimplexState] = {}
    exterior_packets: dict[int, np.ndarray] = {}
    width = layout.dimension + 1
    columns = np.asarray(
        tuple(
            tuple(index for index in range(width) if index != opposite)
            for opposite in range(width)
        ),
        dtype=np.int64,
    )
    source = lineage.source.mesh
    source_rows_count = sum(block.vertices.shape[0] for block in source.blocks)
    _uniform_charge(source_rows_count, source_rows_count * width * (width * 8 + 32))
    source_rows = np.concatenate(
        tuple(
            np.asarray(source.vertex_global_ids)[np.asarray(block.vertices)]
            for block in source.blocks
        )
    )
    source_facets = np.sort(source_rows[:, columns], axis=2).reshape((-1, width - 1))
    _, facet_inverse, facet_counts = np.unique(
        source_facets,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    original_exterior = (facet_counts[facet_inverse] == 1).reshape((-1, width))
    source_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in source.blocks)
    )
    source_order = np.argsort(source_cell_ids)
    original_rows = source_order[
        np.searchsorted(source_cell_ids[source_order], arrays["parent_ids"])
    ]
    for shard in compiled.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "Packed publication requires explicit addressable part rectangles."
            )
        first = 0 if selection.start is None else selection.start
        last = (
            compiled.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        )
        for part in range(first, last):
            local = _addressable_part_state(compiled, part)
            host = _host_epoch(local)
            baseline = _host_epoch(_addressable_part_state(initial, part))
            anchor = _local_anchor(original.anchor, entities, baseline)
            prepared = PreparedAdaptiveSimplex(
                preparation,
                layout,
                _addressable_part_state(initial, part),
                anchor,
            )
            committed = _committed(prepared, host)
            target_ids = np.asarray(committed.hierarchy.cell_global_ids)
            initial_ids = baseline.cell_ids[baseline.cell_active]
            restored = np.flatnonzero(
                np.isin(arrays["parent_ids"], target_ids)
                & ~np.isin(arrays["parent_ids"], initial_ids)
            )
            if restored.size == 0:
                packets[part] = local
                exterior_packets[part] = np.asarray(
                    jax.device_get(_addressable_part_array(exterior, part))
                )
                continue
            _uniform_charge(
                sum(value.size for value in host if isinstance(value, np.ndarray)),
                sum(value.nbytes for value in host if isinstance(value, np.ndarray))
                + layout.cell_capacity * width,
            )
            values = {
                name: np.array(value, copy=True)
                for name, value in zip(
                    host._fields,
                    host,
                    strict=True,
                )
                if isinstance(value, np.ndarray)
            }
            boundary = np.array(
                jax.device_get(_addressable_part_array(exterior, part)), copy=True
            )
            for root in restored:
                siblings = np.flatnonzero(
                    np.isin(values["cell_ids"], arrays["child_ids"][root])
                )
                if siblings.size != arrays["child_ids"].shape[1]:
                    raise ValueError(
                        "Packed inverse publication lost its actual complete sibling slots."
                    )
                slot = int(siblings[0])
                values["cell_active"][siblings] = False
                values["retired"][siblings] = True
                boundary[siblings] = False
                vertices = values["vertex_ids"]
                order = np.argsort(vertices)
                rows = order[
                    np.searchsorted(vertices[order], arrays["parent_rows"][root])
                ]
                tuples = order[
                    np.searchsorted(vertices[order], arrays["parent_vertices"][root])
                ]
                if not np.array_equal(vertices[rows], arrays["parent_rows"][root]):
                    raise ValueError(
                        "Packed inverse publication lost an original scientific vertex."
                    )
                values["cell_ids"][slot] = arrays["parent_ids"][root]
                values["cells"][slot] = rows
                values["tuples"][slot] = tuples
                for name, field in (
                    ("tags", "parent_tags"),
                    ("generations", "parent_levels"),
                    ("cell_classes", "parent_classes"),
                    ("facet_classes", "parent_facet_classes"),
                ):
                    values[name][slot] = arrays[field][root]
                values["blocks"][slot] = publication_blocks[
                    int(arrays["parent_blocks"][root])
                ]
                values["facet_classes"][slot] = (
                    arrays["parent_facet_classes"][root][::-1] + 1
                )
                values["parents"][slot] = -1
                values["children"][slot] = -1
                values["bisection_vertices"][slot] = -1
                values["cell_active"][slot] = True
                values["retired"][slot] = False
                values["coarsen_marked"][slot] = False
                boundary[slot] = original_exterior[original_rows[root]]
            values["vertex_active"][:] = False
            values["vertex_active"][np.unique(values["cells"][values["cell_active"]])] = (
                True
            )
            accepted_ids = values["cell_ids"][values["cell_active"]]
            if not np.array_equal(np.sort(accepted_ids), np.sort(target_ids)):
                raise ValueError(
                    "Packed publication differs from the actual admitted host inverse topology."
                )
            vertex_count, cell_count = (int(value) for value in host.cursors[:2])
            order = np.argsort(values["cell_ids"][:cell_count], kind="stable")
            inverse = np.empty(order.shape, dtype=np.int64)
            inverse[order] = np.arange(cell_count, dtype=np.int64)
            for name in (
                "cell_ids",
                "cell_active",
                "cells",
                "tuples",
                "tags",
                "blocks",
                "generations",
                "parents",
                "children",
                "bisection_vertices",
                "retired",
                "refine_rejected",
                "coarsen_marked",
                "cell_classes",
                "facet_classes",
            ):
                values[name] = values[name][:cell_count][order]
            parent = values["parents"]
            values["parents"] = np.where(
                parent >= 0, 2 * inverse[np.maximum(parent, 0) // 2] + parent % 2, -1
            )
            children = values["children"]
            values["children"] = np.where(
                children >= 0, inverse[np.maximum(children, 0)], -1
            )
            boundary[:cell_count] = boundary[:cell_count][order]
            for name in (
                "coordinates",
                "vertex_ids",
                "vertex_active",
                "vertex_parents",
                "vertex_protected",
            ):
                values[name] = values[name][:vertex_count]
            codes = values["protected_codes"]
            codes = codes[
                (codes >= 0) & (codes < layout.vertex_capacity * layout.vertex_capacity)
            ]
            projected = adaptive_simplex_state(
                layout,
                coordinates=values["coordinates"],
                vertex_ids=values["vertex_ids"],
                vertex_active=values["vertex_active"],
                vertex_parents=values["vertex_parents"],
                vertex_protected=values["vertex_protected"],
                cells=values["cells"],
                tuples=values["tuples"],
                tags=values["tags"],
                blocks=values["blocks"],
                generations=values["generations"],
                parents=values["parents"],
                children=values["children"],
                bisection_vertices=values["bisection_vertices"],
                cell_ids=values["cell_ids"],
                cell_active=values["cell_active"],
                cell_classes=values["cell_classes"],
                facet_classes=values["facet_classes"],
                protected_edges=np.stack(
                    (codes // layout.vertex_capacity, codes % layout.vertex_capacity),
                    axis=1,
                ),
                next_vertex_id=int(host.cursors[2]),
                next_cell_id=int(host.cursors[3]),
            )
            projected = eqx.tree_at(
                lambda value: (
                    value.retired,
                    value.refine_rejected,
                    value.coarsen_marked,
                    value.vertex_levels,
                    value.vertex_removal,
                    value.clocks,
                    value.counters,
                ),
                projected,
                (
                    jnp.pad(
                        jnp.asarray(values["retired"]),
                        (0, layout.cell_capacity - cell_count),
                    ),
                    jnp.pad(
                        jnp.asarray(values["refine_rejected"]),
                        (0, layout.cell_capacity - cell_count),
                    ),
                    jnp.pad(
                        jnp.asarray(values["coarsen_marked"]),
                        (0, layout.cell_capacity - cell_count),
                    ),
                    local.vertex_levels,
                    local.vertex_removal,
                    local.clocks,
                    local.counters,
                ),
            )
            packets[part], exterior_packets[part] = projected, boundary

    def place(value: jax.Array, values: dict[int, np.ndarray]) -> jax.Array:
        def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "Packed publication requires concrete addressable array slices."
                )
            leading = index[0]
            first = 0 if leading.start is None else leading.start
            last = value.shape[0] if leading.stop is None else leading.stop
            return np.stack(tuple(values[part][index[1:]] for part in range(first, last)))

        return jax.make_array_from_callback(value.shape, value.sharding, addressable)

    leaves = {part: jax.tree_util.tree_leaves(state) for part, state in packets.items()}
    projected_leaves = tuple(
        place(
            value,
            {
                part: np.asarray(jax.device_get(items[index]))
                for part, items in leaves.items()
            },
        )
        for index, value in enumerate(jax.tree_util.tree_leaves(compiled))
    )
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(compiled), projected_leaves
    ), place(exterior, exterior_packets)


def _collective_publication_state(
    evidence: CollectiveMeshEvidence, /
) -> AdaptiveSimplexState:
    arrays = dict(evidence.logical_arrays)
    mesh_names = (
        "coordinates",
        "vertex_ids",
        "vertex_active",
        "cells",
        "cell_ids",
        "cell_active",
        "facet_neighbors",
    )
    state_names = (
        "vertex_half_facets",
        "tuples",
        "tags",
        "blocks",
        "generations",
        "parents",
        "children",
        "bisection_vertices",
        "retired",
        "cell_classes",
        "facet_classes",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
        "protected_codes",
        "refine_rejected",
        "coarsen_marked",
        "cursors",
        "clocks",
        "counters",
    )
    values = {name: arrays[f"epoch/{name}"] for name in (*mesh_names, *state_names)}

    def restore(part: dict[str, jax.Array]) -> AdaptiveSimplexState:
        return AdaptiveSimplexState(
            MaskedSimplexMesh(*(part[name] for name in mesh_names)),
            **{name: part[name] for name in state_names},
        )

    return eqx.filter_vmap(restore)(values)


def _packed_neighborhood(
    partitioned: PartitionedAdaptiveSimplex,
    compiled: AdaptiveSimplexState,
    evidence: CollectiveMeshEvidence,
    /,
) -> SimplexNeighborhoodWorkset:
    from ._distribution import (
        _simplex_neighborhood_seed,
        expand_simplex_neighborhood_workset,
    )

    lineage = evidence.uniform_refinement
    if lineage is None:
        raise ValueError(
            "Packed neighborhood requires its retained uniform source authority."
        )
    published = _collective_publication_state(evidence)
    arrays = dict(evidence.logical_arrays)
    parent_ids = np.asarray(lineage.parent_ids)
    child_ids = np.asarray(lineage.child_ids)
    capacity = partitioned.layout.cell_capacity
    prepared_uniform = partitioned.prepared.anchor.start.uniform
    if prepared_uniform:
        from ._collective_organization import _original_vertex_ancestry

        sibling_parent = {
            int(child): int(parent)
            for parent, children in zip(parent_ids, child_ids, strict=True)
            for child in children
        }
    packets: dict[int, SimplexNeighborhoodWorkset] = {}
    for shard in compiled.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "Packed neighborhood requires explicit addressable part rectangles."
            )
        first = 0 if selection.start is None else selection.start
        last = (
            compiled.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        )
        for part in range(first, last):
            raw = _addressable_part_state(compiled, part)
            initial = _addressable_part_state(partitioned.states, part)
            seed = _simplex_neighborhood_seed(
                raw,
                initial,
                _addressable_part_array(partitioned.source_exterior, part),
                jnp.asarray(part, dtype=jnp.int32),
                capacity,
            )
            seed_leaves = tuple(getattr(seed, name) for name in _WORKSET_FIELDS)
            _uniform_charge(
                sum(value.size for value in seed_leaves),
                sum(value.size * value.dtype.itemsize for value in seed_leaves),
            )
            host = jax.device_get(seed)
            values = {
                name: np.array(getattr(host, name), copy=True) for name in _WORKSET_FIELDS
            }
            final = _host_epoch(_addressable_part_state(published, part))
            baseline = _host_epoch(initial)
            slots = np.flatnonzero(final.cell_active)
            order = slots[np.argsort(final.cell_ids[slots], kind="stable")]
            ids = final.cell_ids[order]
            raw_ids = np.asarray(host.cell_ids)
            positions = np.searchsorted(raw_ids, ids)
            safe = np.minimum(positions, capacity - 1)
            matched = raw_ids[safe] == ids
            for name in _WORKSET_FIELDS:
                if name != "status":
                    values[name][: ids.size] = np.asarray(getattr(host, name))[safe]
            values["cell_ids"][:] = np.iinfo(np.int64).max
            values["cell_ids"][: ids.size] = ids
            values["cell_valid"][:] = False
            values["cell_valid"][: ids.size] = True
            values["cell_vertices"][: ids.size] = final.vertex_ids[final.cells[order]]
            values["cell_coordinates"][: ids.size] = final.coordinates[final.cells[order]]
            values["cell_classes"][: ids.size] = final.cell_classes[order]
            values["cell_owner"][: ids.size] = part
            boundary = np.asarray(
                jax.device_get(
                    _addressable_part_array(arrays["epoch/source_exterior"], part)
                )
            )
            values["cell_exterior"][: ids.size] = boundary[order]
            roots = np.arange(baseline.cell_ids.size, dtype=np.int64)
            for slot in range(int(baseline.cursors[1])):
                parent = baseline.parents[slot] // 2
                if parent >= 0:
                    roots[slot] = roots[parent]
            for row in np.flatnonzero(~matched):
                root = np.searchsorted(parent_ids, ids[row])
                if root >= parent_ids.size or parent_ids[root] != ids[row]:
                    raise ValueError(
                        "Publication issued a cell absent from raw output and packed parent authority."
                    )
                predecessor = baseline.cell_ids[
                    baseline.cell_active
                    & np.isin(baseline.cell_ids[roots], child_ids[root])
                ]
                if (
                    predecessor.size == 0
                    or predecessor.size > values["source_cell_ids"].shape[1]
                ):
                    raise ValueError(
                        "Packed neighborhood lost its complete actual predecessor source support."
                    )
                values["root_cell_ids"][row] = -1
                values["source_cell_ids"][row] = -1
                values["source_cell_ids"][row, : predecessor.size] = predecessor
                values["source_vertex_ids"][row] = -1
                values["source_weights"][row] = 0.0
                values["source_vertex_ids"][row, :, 0] = values["cell_vertices"][row]
                values["source_weights"][row, :, 0] = 1.0
            if prepared_uniform:
                support_ids, support_weights, passed = _original_vertex_ancestry(
                    raw, lineage
                )
                if not bool(jax.device_get(passed)):
                    raise ValueError(
                        "Prepared uniform source lost its original full coefficient support."
                    )
                _uniform_charge(
                    support_ids.size + support_weights.size,
                    support_ids.nbytes + support_weights.nbytes,
                )
                raw_vertices = np.asarray(jax.device_get(raw.mesh.vertex_ids))
                vertex_order = np.argsort(raw_vertices, kind="stable")
                queries = values["cell_vertices"][: ids.size]
                positions = np.searchsorted(raw_vertices[vertex_order], queries)
                if np.any(positions >= raw_vertices.size) or not np.array_equal(
                    raw_vertices[vertex_order][positions], queries
                ):
                    raise ValueError(
                        "Prepared source support lacks a published original-law vertex."
                    )
                vertex_rows = vertex_order[positions]
                source_ids = np.asarray(jax.device_get(support_ids))
                source_weights = np.asarray(jax.device_get(support_weights))
                regenerated_parents = set(
                    partitioned.prepared.anchor.source.cells.ids.tolist()
                )
                regenerated_children = set(
                    partitioned.prepared.anchor.start.front.cells.ids.tolist()
                )
                for row in range(ids.size):
                    if (
                        matched[row]
                        and int(values["root_cell_ids"][row]) not in regenerated_children
                    ):
                        continue
                    parent = (
                        sibling_parent.get(int(values["root_cell_ids"][row]))
                        if matched[row]
                        else int(ids[row])
                    )
                    if parent is None or parent not in parent_ids:
                        raise ValueError(
                            "Prepared uniform cell lacks its original accepted predecessor SCI."
                        )
                    if parent not in regenerated_parents:
                        continue
                    values["source_vertex_ids"][row] = source_ids[vertex_rows[row]]
                    values["source_weights"][row] = source_weights[vertex_rows[row]]
                    values["root_cell_ids"][row] = parent
                    values["source_cell_ids"][row] = -1
                    values["source_cell_ids"][row, 0] = parent
            packets[part] = SimplexNeighborhoodWorkset(
                *(jnp.asarray(values[name]) for name in _WORKSET_FIELDS)
            )

    def place(value: jax.Array, blocks: dict[int, jax.Array]) -> jax.Array:
        shape = (partitioned.parts.part_count, *value.shape)

        def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "Packed neighborhood requires concrete addressable packet slices."
                )
            leading = index[0]
            first = 0 if leading.start is None else leading.start
            last = shape[0] if leading.stop is None else leading.stop
            return np.stack(
                tuple(
                    np.asarray(jax.device_get(blocks[part]))[index[1:]]
                    for part in range(first, last)
                )
            )

        return jax.make_array_from_callback(
            shape, compiled.mesh.cell_ids.sharding, addressable
        )

    exemplar = next(iter(packets.values()))
    workset = SimplexNeighborhoodWorkset(
        *(
            place(
                getattr(exemplar, name),
                {part: getattr(packet, name) for part, packet in packets.items()},
            )
            for name in _WORKSET_FIELDS
        )
    )
    distribution = partitioned.prepared.adaptation.policy.distribution
    if distribution is None:
        raise ValueError("Packed neighborhood lost its accepted ownership policy.")
    return expand_simplex_neighborhood_workset(
        partitioned.parts,
        workset,
        halo_width=distribution.halo_width,
        vertex_capacity=partitioned.layout.vertex_capacity,
    )


def _addressable_commits(
    partitioned: PartitionedAdaptiveSimplex, states: AdaptiveSimplexState, /
) -> tuple[_LocalCommit, ...]:
    """Prepare exact local topology/lineage records, never a merged global mesh."""

    anchor = partitioned.prepared.anchor
    premise = _scientific_preparation_source(partitioned.prepared.adaptation)
    seed_entities = partitioned.source_entities
    if anchor.start.uniform or anchor.start.front.growth.levels.size:
        lineage = anchor.start.uniform_refinement
        if not anchor.start.uniform or lineage is None:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "Distributed source growth requires its actual canonical uniform source law.",
                stage="distributed-device-commit",
            )
        seed = _logical_publication(
            partitioned.prepared.adaptation.source,
            partitioned.states,
            partitioned.solver_cell_owners,
            partitioned.source_exterior,
            partitioned.layout,
            original_source=premise if isinstance(premise, CellMeshingResult) else None,
            uniform_refinement=anchor.start.uniform_refinement,
            numeric_version=f"adaptation:{partitioned.prepared.adaptation.prepared_id}",
        )
        seed_entities = tuple(
            (keys[:count], identifiers[:count], owners[:count])
            for keys, identifiers, owners, count in zip(
                seed.entity_keys,
                seed.entity_ids,
                seed.entity_owners,
                seed.counts,
                strict=True,
            )
        )
    if isinstance(premise, CellMeshingResult) and len(premise.mesh.blocks) != 1:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Distributed source preparation requires a single actual simplex coordinate block.",
            stage="distributed-device-commit",
        )
    commits = []
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError("Adaptive part shards require a leading partition slice.")
        first = 0 if selection.start is None else selection.start
        last = states.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        for part in range(first, last):
            local = _addressable_part_state(states, part)
            initial = _host_epoch(_addressable_part_state(partitioned.states, part))
            host = _host_epoch(local)
            require_committed_status(
                host.flags,
                host.coordinates,
                host.cells[host.cell_active],
                DeviceEpoch.BISECTION,
            )
            local_anchor = _local_anchor(
                partitioned.prepared.anchor, seed_entities, initial
            )
            prepared = PreparedAdaptiveSimplex(
                partitioned.prepared.adaptation,
                partitioned.layout,
                _addressable_part_state(partitioned.states, part),
                local_anchor,
            )
            committed = _rebind_issued_vertices(
                _committed(prepared, host), local_anchor, host
            )
            commits.append(_LocalCommit(part, local_anchor.source.mesh, committed))
    return tuple(commits)


class _LogicalPublication(NamedTuple):
    arrays: tuple[tuple[str, jax.Array], ...]
    counts: tuple[int, ...]
    entity_keys: tuple[jax.Array, ...]
    entity_ids: tuple[jax.Array, ...]
    entity_owners: tuple[jax.Array, ...]
    topology_id: str
    geometry_id: str
    geometry_projection: CellGeometryStorageProjection | None = None
    publication_projection: PublicationProjection | None = None
    scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None
    attribute_projections: tuple[tuple[str, MeshAttributeProjection], ...] | None = None
    restart_proof: SimplexRestartRepackProof | None = None
    storage_binding: CollectiveMeshStorageBinding | None = None
    surface_transfer: SurfaceAssociationTransfer | None = None


def _canonical_rows(
    keys: jax.Array, valid: jax.Array, /
) -> tuple[jax.Array, jax.Array, int]:
    """Bounded device ordering with a canonical active prefix and explicit count."""

    sentinel = jnp.iinfo(jnp.int64).max
    padded = jnp.where(valid[:, None], keys, sentinel)
    order = jnp.lexsort(padded.T[::-1])
    ordered = padded[order]
    fresh = jnp.concatenate(
        (
            jnp.asarray([True], dtype=jnp.bool_),
            jnp.any(ordered[1:] != ordered[:-1], axis=1),
        )
    ) & jnp.all(ordered != sentinel, axis=1)
    count = int(jax.device_get(jnp.sum(fresh, dtype=jnp.int64)))
    selected = jnp.nonzero(fresh, size=keys.shape[0], fill_value=keys.shape[0] - 1)[0]
    prefix = jnp.arange(keys.shape[0]) < count
    return jnp.where(prefix[:, None], ordered[selected], -1), order[selected], count


def _key_positions(table: jax.Array, queries: jax.Array, /) -> jax.Array:
    """Lexicographic device lookup of fixed-width global-ID entity keys."""

    low = jnp.zeros((queries.shape[0],), dtype=jnp.int64)
    high = jnp.full(low.shape, table.shape[0], dtype=jnp.int64)

    def search(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        left, right = bounds
        middle = (left + right) // 2
        row = table[jnp.minimum(middle, table.shape[0] - 1)]
        equal = row == queries
        earlier_equal = jnp.concatenate(
            (
                jnp.ones((queries.shape[0], 1), dtype=jnp.bool_),
                jnp.cumprod(equal[:, :-1], axis=1, dtype=jnp.int32).astype(jnp.bool_),
            ),
            axis=1,
        )
        less = jnp.any((row < queries) & earlier_equal, axis=1)
        return jnp.where(less, middle + 1, left), jnp.where(less, right, middle)

    low, _ = jax.lax.fori_loop(0, table.shape[0].bit_length() + 1, search, (low, high))
    clipped = jnp.minimum(low, table.shape[0] - 1)
    return jnp.where(jnp.all(table[clipped] == queries, axis=1), clipped, -1)


def _inherited_entity_ids(
    source: CellMeshingResult,
    degree: int,
    table: jax.Array,
    count: int,
    partition_count: int,
    original_source: CellMeshingResult | None,
    /,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Restore retired scientific entity identities from consumed predecessor leaves."""
    from ._initial_certification import InitialCollectiveMeshEvidence

    inherited = jnp.full((table.shape[0],), -1, dtype=jnp.int64)
    high = jnp.asarray(-1, dtype=jnp.int64)
    ledger_keys = []
    ledger_ids = []
    while True:
        if source.collective_evidence is None:
            keys = jnp.asarray(entity_keys(source.mesh, degree), dtype=jnp.int64)
            identifiers = jnp.asarray(
                source.mesh.entity_set(degree).entity_ids, dtype=jnp.int64
            )
        else:
            evidence = source.collective_evidence
            keys = evidence.entity_keys[degree][: evidence.global_entity_counts[degree]]
            identifiers = evidence.entity_ids[degree][
                : evidence.global_entity_counts[degree]
            ]
        ledger_keys.append(keys)
        ledger_ids.append(identifiers)
        order = jnp.lexsort(keys.T[::-1])
        positions = _key_positions(keys[order], table)
        restored = identifiers[order[jnp.maximum(positions, 0)]]
        inherited = jnp.where((inherited < 0) & (positions >= 0), restored, inherited)
        high = jnp.maximum(high, jnp.max(identifiers))
        evidence = source.collective_evidence
        if evidence is None or isinstance(evidence, InitialCollectiveMeshEvidence):
            if (
                original_source is not None
                and source.result_id != original_source.result_id
            ):
                source, original_source = original_source, None
                continue
            break
        if not isinstance(evidence, CollectiveMeshEvidence):
            raise TypeError("Collective source authority has an unknown theorem kind.")
        source = evidence.source
    active = jnp.arange(table.shape[0]) < count
    new = (inherited < 0) & active
    ids = jnp.where(
        active,
        jnp.where(inherited >= 0, inherited, high + jnp.cumsum(new, dtype=jnp.int64)),
        -1,
    )
    history_keys = jnp.concatenate(ledger_keys)
    history_ids = jnp.concatenate(ledger_ids)
    padding = (-history_keys.shape[0]) % partition_count
    history_keys = jnp.pad(history_keys, ((0, padding), (0, 0)), constant_values=-1)
    history_ids = jnp.pad(history_ids, (0, padding), constant_values=-1)
    keys, order, history_count = _canonical_rows(history_keys, history_ids >= 0)
    active_table = jnp.where(table >= 0, table, jnp.iinfo(jnp.int64).max)
    removed = (jnp.arange(keys.shape[0]) < history_count) & (
        _key_positions(active_table, keys) < 0
    )
    retired_keys, retired_order, _ = _canonical_rows(keys, removed)
    retired_ids = jnp.where(
        retired_keys[:, 0] >= 0, history_ids[order[retired_order]], -1
    )
    return ids, retired_keys, retired_ids


class _LogicalPresentation(NamedTuple):
    names: tuple[str, ...]
    kinds: tuple[str, ...]
    groups: jax.Array
    counts: tuple[int, ...]
    order: jax.Array


def _logical_array_tree_fingerprint(
    value: jax.Array, logical_shape: tuple[int, ...], /
) -> dict[str, object]:
    """Stream one logical JAX array through the serial array fingerprint recipe."""

    if (
        not logical_shape
        or len(logical_shape) != value.ndim
        or logical_shape[1:] != value.shape[1:]
        or logical_shape[0] < 0
        or logical_shape[0] > value.shape[0]
    ):
        raise ValueError(
            "Logical array fingerprints may trim only a nonempty leading capacity axis."
        )
    dtype = np.dtype(value.dtype)
    metadata = canonical_json(
        {
            "path": "<root>",
            "dtype": dtype.str,
            "shape": list(logical_shape),
        }
    ).encode("ascii")
    byte_count = prod(logical_shape) * dtype.itemsize
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(byte_count.to_bytes(8, "big"))
    intervals: dict[tuple[int, int], jax.Device] = {}
    for device, index in value.sharding.devices_indices_map(value.shape).items():
        if (
            len(index) != value.ndim
            or not isinstance(index[0], slice)
            or any(
                not isinstance(entry, slice) or entry.indices(size) != (0, size, 1)
                for entry, size in zip(index[1:], value.shape[1:], strict=True)
            )
        ):
            raise ValueError(
                "Logical topology hashing requires contiguous leading-axis shards."
            )
        start, stop, step = index[0].indices(value.shape[0])
        if step != 1:
            raise ValueError("Logical topology hashing does not admit strided shards.")
        key = (start, stop)
        previous = intervals.get(key)
        if previous is None or (device.process_index, device.id) < (
            previous.process_index,
            previous.id,
        ):
            intervals[key] = device
    cursor = 0
    for start, stop in sorted(intervals):
        if start != cursor or stop < start:
            raise ValueError(
                "Logical topology shards overlap or leave leading-axis coverage gaps."
            )
        cursor = stop
    if cursor != value.shape[0]:
        raise ValueError("Logical topology shards leave incomplete global coverage.")
    local = {shard.device: shard.data for shard in value.addressable_shards}
    trailing = prod(value.shape[1:])
    chunk_elements = max(1, (1 << 20) // dtype.itemsize)
    for (start, end), device in sorted(intervals.items()):
        stop = min(end, logical_shape[0])
        count = max(0, stop - start) * trailing
        source = device.process_index == jax.process_index()
        for first in range(0, count, chunk_elements):
            last = min(first + chunk_elements, count)
            if source:
                shard = local.get(device)
                if shard is None:
                    raise ValueError(
                        "The canonical topology shard owner has no addressable payload."
                    )
                payload = np.asarray(shard.reshape(-1)[first:last])
            else:
                payload = np.empty((last - first,), dtype=dtype)
            if jax.process_count() > 1 and not value.is_fully_addressable:
                payload = multihost_utils.broadcast_one_to_all(payload, is_source=source)
            digest.update(np.ascontiguousarray(payload).tobytes(order="C"))
    return {
        "signature": [
            {
                "path": "<root>",
                "shape": list(logical_shape),
                "dtype": dtype.str,
            }
        ],
        "sha256": digest.hexdigest(),
    }


def _publication_block_declarations(
    accepted_source: CellMeshingResult,
    original_source: CellMeshingResult | None,
    /,
) -> tuple[CellBlock, ...]:
    evidence = accepted_source.collective_evidence
    raw = (
        evidence.publication_source_blocks
        if isinstance(evidence, CollectiveMeshEvidence)
        else accepted_source.mesh.blocks
    )
    declarations: list[CellBlock] = []
    for block in raw:
        if not isinstance(block, CellBlock):
            raise TypeError(
                "Logical presentation requires actual simplex source block declarations."
            )
        declarations.append(block)
    if original_source is not None:
        declared = {block.block_id for block in declarations}
        for block in original_source.mesh.blocks:
            if not isinstance(block, CellBlock):
                raise TypeError(
                    "Original presentation requires actual simplex source blocks."
                )
            if block.block_id not in declared:
                declarations.append(block)
                declared.add(block.block_id)
    if not declarations:
        raise ValueError("Logical presentation requires at least one source block.")
    return tuple(declarations)


def _presentation_roots(
    premise: CellMeshingResult | InitialCollectiveMeshEvidence, /
) -> tuple[tuple[str, str, str], ...]:
    if isinstance(premise, CellMeshingResult):
        elements, _, _ = premise.geometry.resolve(premise.mesh)
        return tuple(
            (block.name, block.cell_kind, element.element_id)
            for block, element in zip(premise.mesh.blocks, elements, strict=True)
        )
    from ..discretization._cell_geometry import coordinate_lagrange_element

    element = coordinate_lagrange_element("triangle", 1)
    return (("surface", "triangle", element.element_id),)


def _declaration_presentations(
    declarations: tuple[CellBlock, ...],
    roots: tuple[tuple[str, str, str], ...],
    /,
) -> tuple[tuple[int, str], ...]:
    root_rows = {name: index for index, (name, _, _) in enumerate(roots)}
    root_ids = {name: identifier for name, _, identifier in roots}
    marker = "/coefficient-action/"
    result = []
    for block in declarations:
        root_name = block.name
        element_id = root_ids.get(root_name)
        prefix, separator, suffix = block.name.rpartition(marker)
        if (
            separator
            and prefix in root_rows
            and len(suffix) == 64
            and all(character in "0123456789abcdef" for character in suffix)
        ):
            root_name, element_id = prefix, suffix
        if element_id is None or root_name not in root_rows:
            raise ValueError(
                "A raw presentation block has no original coefficient-action owner."
            )
        result.append((root_rows[root_name], element_id))
    return tuple(result)


def _barycentric_element_id(source_id: str, action: np.ndarray, /) -> str:
    weights = np.asarray(action, dtype=np.float64)
    return canonical_fingerprint(
        {
            "kind": "full-barycentric-cell-geometry",
            "source": source_id,
            "weights": array_tree_fingerprint(weights),
        }
    )


def _local_presentation_keys(
    state: AdaptiveSimplexState,
    declarations: tuple[tuple[int, str], ...],
    uniform_actions: dict[int, tuple[int, str, str]],
    /,
) -> np.ndarray:
    host = _host_epoch(state)
    capacity, width = host.cells.shape
    keys = np.full((capacity, 33), -1, dtype=np.int64)
    for slot in np.flatnonzero(host.cell_active):
        current = int(slot)
        actions = []
        for _ in range(_MAXIMUM_DEPTH + 1):
            parent = int(host.parents[current]) // 2
            if parent < 0:
                break
            corners = host.vertex_ids[host.cells[current]]
            original = host.vertex_ids[host.cells[parent]]
            midpoint = host.vertex_ids[host.bisection_vertices[parent]]
            ordered = host.vertex_ids[host.tuples[parent]]
            endpoints = np.asarray(
                (ordered[0], ordered[host.tags[parent]]), dtype=np.int64
            )
            action = np.where(
                corners[:, None] == midpoint,
                0.5 * np.any(original[None, :, None] == endpoints[None, None, :], axis=2),
                corners[:, None] == original[None, :],
            ).astype(np.float64)
            actions.append(action)
            current = parent
        else:
            raise ValueError(
                "A presentation action chain exceeds the device depth bound."
            )
        block = int(host.blocks[current])
        if block < 0 or block >= len(declarations):
            raise ValueError(
                "A logical target cell has no declared source presentation block."
            )
        root, element_id = declarations[block]
        uniform = uniform_actions.get(int(host.cell_ids[current]))
        if uniform is not None:
            uniform_root, original_id, action_id = uniform
            if root == uniform_root and element_id == original_id:
                element_id = action_id
        if any(action.shape != (width, width) for action in actions):
            raise ValueError("A coefficient action lost its complete simplex basis.")
        for action in reversed(actions):
            element_id = _barycentric_element_id(element_id, action)
        keys[slot, 0] = root
        keys[slot, 1:] = np.frombuffer(bytes.fromhex(element_id), dtype=np.uint8).astype(
            np.int64
        )
    return keys


def _replicated_host_rows(value: jax.Array, /) -> np.ndarray:
    if not value.is_fully_addressable:
        sharding = value.sharding
        if not isinstance(sharding, jax.sharding.NamedSharding):
            raise ValueError(
                "Logical presentation metadata requires a named distributed sharding."
            )
        value = jax.device_put(
            value,
            jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec()),
        )
    return np.asarray(jax.device_get(value), dtype=np.int64)


def _logical_presentation(
    accepted_source: CellMeshingResult,
    premise: CellMeshingResult | InitialCollectiveMeshEvidence,
    original_source: CellMeshingResult | None,
    states: AdaptiveSimplexState,
    cell_keys: jax.Array,
    cell_order: jax.Array,
    cells: int,
    uniform_refinement: BisectionUniformRefinement | None,
    /,
) -> _LogicalPresentation:
    roots = _presentation_roots(premise)
    declarations = _publication_block_declarations(accepted_source, original_source)
    declaration_presentations = _declaration_presentations(declarations, roots)
    uniform_actions: dict[int, tuple[int, str, str]] = {}
    if uniform_refinement is not None:
        arrays = uniform_refinement.host_arrays()
        blocks = np.asarray(arrays["parent_blocks"], dtype=np.int64)
        children = np.asarray(arrays["child_ids"], dtype=np.int64)
        actions = np.asarray(arrays["barycentric_weights"], dtype=np.float64)
        if (
            blocks.ndim != 1
            or children.ndim != 2
            or actions.ndim != 4
            or children.shape != actions.shape[:2]
            or blocks.shape != children.shape[:1]
        ):
            raise ValueError(
                "Uniform presentation actions have incompatible scientific axes."
            )
        for row in range(blocks.shape[0]):
            root = int(blocks[row])
            if root < 0 or root >= len(roots):
                raise ValueError(
                    "Uniform presentation names an absent original source block."
                )
            original_id = roots[root][2]
            for child in range(children.shape[1]):
                uniform_actions[int(children[row, child])] = (
                    root,
                    original_id,
                    _barycentric_element_id(original_id, actions[row, child]),
                )
    packets: dict[int, np.ndarray] = {}
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "Presentation metadata requires a leading partition rectangle."
            )
        first = 0 if selection.start is None else selection.start
        last = states.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        for part in range(first, last):
            packets.setdefault(
                part,
                _local_presentation_keys(
                    _addressable_part_state(states, part),
                    declaration_presentations,
                    uniform_actions,
                ),
            )

    def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
        if index is None:
            raise ValueError(
                "Presentation placement requires its concrete addressable rectangle."
            )
        leading = index[0]
        if not isinstance(leading, slice):
            raise ValueError("Presentation placement requires a leading slice.")
        first = 0 if leading.start is None else leading.start
        last = states.mesh.cell_ids.shape[0] if leading.stop is None else leading.stop
        return np.stack(tuple(packets[part] for part in range(first, last)))[index[1:]]

    key_shape = (*states.mesh.cell_ids.shape, 33)
    placed = jax.make_array_from_callback(
        key_shape, states.mesh.cell_ids.sharding, addressable
    )
    flat = placed.reshape((-1, 33))
    active = states.mesh.cell_active.reshape(-1)
    unique, _, count = _canonical_rows(flat, active)
    host_unique = _replicated_host_rows(unique[:count])
    root_names = tuple(name for name, _, _ in roots)
    root_kinds = tuple(kind for _, kind, _ in roots)
    root_ids = tuple(identifier for _, _, identifier in roots)
    records = []
    for row in host_unique:
        root = int(row[0])
        if root < 0 or root >= len(roots):
            raise ValueError("A presentation key names an absent original source block.")
        element_id = bytes(np.asarray(row[1:], dtype=np.uint8)).hex()
        name = (
            root_names[root]
            if element_id == root_ids[root]
            else f"{root_names[root]}/coefficient-action/{element_id}"
        )
        records.append((name, root_kinds[root], row))
    records.sort(key=lambda record: record[0])
    names = tuple(record[0] for record in records)
    kinds = tuple(record[1] for record in records)
    table = np.asarray([record[2] for record in records], dtype=np.int64)
    key_order = np.lexsort(table.T[::-1])
    table_array = jnp.asarray(table[key_order], dtype=jnp.int64)
    group_by_row = jnp.asarray(key_order, dtype=jnp.int64)
    sharding = placed.sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
        replicated = jax.sharding.NamedSharding(
            sharding.mesh, jax.sharding.PartitionSpec()
        )
        table_array = jax.device_put(table_array, replicated)
        group_by_row = jax.device_put(group_by_row, replicated)
    positions = _key_positions(table_array, flat)
    slot_groups = jnp.where(positions >= 0, group_by_row[jnp.maximum(positions, 0)], -1)
    groups = slot_groups[cell_order]
    prefix = jnp.arange(cell_keys.shape[0]) < cells
    counts = tuple(
        int(jax.device_get(jnp.sum(prefix & (groups == group), dtype=jnp.int64)))
        for group in range(len(names))
    )
    if any(group_count <= 0 for group_count in counts) or sum(counts) != cells:
        raise ValueError("Logical presentation grouping lost an active target cell.")
    within = cell_keys[:, 0]
    if isinstance(premise, CellMeshingResult):
        original_blocks = {block.name: block for block in premise.mesh.blocks}
        for group, name in enumerate(names):
            block = original_blocks.get(name)
            if block is None:
                continue
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            order = np.argsort(identifiers, kind="stable")
            sorted_ids = jnp.asarray(identifiers[order], dtype=jnp.int64)
            positions = jnp.searchsorted(sorted_ids, cell_keys[:, 0])
            safe = jnp.minimum(positions, sorted_ids.shape[0] - 1)
            member = prefix & (groups == group)
            if bool(
                jax.device_get(jnp.any(member & (sorted_ids[safe] != cell_keys[:, 0])))
            ):
                raise ValueError(
                    "An unchanged presentation block contains a foreign source cell."
                )
            rows = jnp.asarray(order, dtype=jnp.int64)[safe]
            within = jnp.where(member, rows, within)
    _, presentation_order, ordered_count = _canonical_rows(
        jnp.stack((groups, within), axis=1), prefix
    )
    if ordered_count != cells:
        raise ValueError("Presentation ordering lost a target cell.")
    return _LogicalPresentation(names, kinds, groups, counts, presentation_order)


def _logical_boundary_masks(
    cell_rows: jax.Array,
    cells: int,
    entity_tables: tuple[jax.Array, ...],
    counts: tuple[int, ...],
    /,
) -> tuple[jax.Array, ...]:
    dimension = cell_rows.shape[1] - 1
    columns = jnp.asarray(
        tuple(
            tuple(column for column in range(dimension + 1) if column != opposite)
            for opposite in range(dimension + 1)
        ),
        dtype=jnp.int32,
    )
    cell_valid = jnp.arange(cell_rows.shape[0]) < cells
    facet_keys = jnp.sort(cell_rows[:, columns], axis=2).reshape((-1, dimension))
    facet_valid = jnp.repeat(cell_valid, dimension + 1)
    facet_table = jnp.where(
        entity_tables[-2] >= 0,
        entity_tables[-2],
        jnp.iinfo(jnp.int64).max,
    )
    facet_rows = _key_positions(facet_table, facet_keys)
    incidence = (
        jnp.zeros((facet_table.shape[0],), dtype=jnp.int32)
        .at[jnp.maximum(facet_rows, 0)]
        .add(facet_valid.astype(jnp.int32))
    )
    facet_boundary = (jnp.arange(facet_table.shape[0]) < counts[-2]) & (incidence == 1)
    masks = []
    for degree in range(dimension - 1):
        local = np.asarray(
            tuple(combinations(range(dimension), degree + 1)), dtype=np.int32
        )
        keys = entity_tables[-2][:, local].reshape((-1, degree + 1))
        valid = jnp.repeat(facet_boundary, local.shape[0])
        table = jnp.where(
            entity_tables[degree] >= 0,
            entity_tables[degree],
            jnp.iinfo(jnp.int64).max,
        )
        rows = _key_positions(table, keys)
        membership = (
            jnp.zeros((table.shape[0],), dtype=jnp.int32)
            .at[jnp.maximum(rows, 0)]
            .add(valid.astype(jnp.int32))
        )
        masks.append((jnp.arange(table.shape[0]) < counts[degree]) & (membership > 0))
    masks.append(facet_boundary)
    masks.append(jnp.zeros((cell_rows.shape[0],), dtype=jnp.bool_))
    return tuple(masks)


def _first_appearance_order(
    keys: jax.Array,
    valid: jax.Array,
    table: jax.Array,
    count: int,
    /,
) -> jax.Array:
    rows = _key_positions(jnp.where(table >= 0, table, jnp.iinfo(jnp.int64).max), keys)
    if bool(jax.device_get(jnp.any(valid & (rows < 0)))):
        raise ValueError("A logical simplex route references an absent entity key.")
    sentinel = keys.shape[0]
    first = (
        jnp.full((table.shape[0],), sentinel, dtype=jnp.int64)
        .at[jnp.maximum(rows, 0)]
        .min(jnp.where(valid, jnp.arange(keys.shape[0]), sentinel))
    )
    order = jnp.argsort(first, stable=True)
    if bool(jax.device_get(jnp.any(first[order[:count]] == sentinel))):
        raise ValueError("First-appearance ordering lost a logical simplex entity.")
    return order


def _logical_entity_orders(
    top_rows: jax.Array,
    cells: int,
    entity_tables: tuple[jax.Array, ...],
    counts: tuple[int, ...],
    /,
) -> tuple[jax.Array, ...]:
    dimension = top_rows.shape[1] - 1
    cell_valid = jnp.arange(top_rows.shape[0]) < cells
    orders: list[jax.Array] = [jnp.arange(entity_tables[0].shape[0], dtype=jnp.int64)]
    if dimension == 2:
        routes = jnp.asarray(((0, 1), (1, 2), (2, 0)), dtype=jnp.int32)
        edge_keys = jnp.sort(top_rows[:, routes], axis=2).reshape((-1, 2))
        orders.append(
            _first_appearance_order(
                edge_keys,
                jnp.repeat(cell_valid, 3),
                entity_tables[1],
                counts[1],
            )
        )
    elif dimension == 3:
        face_routes = jnp.asarray(
            ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
            dtype=jnp.int32,
        )
        face_keys = jnp.sort(top_rows[:, face_routes], axis=2).reshape((-1, 3))
        face_order = _first_appearance_order(
            face_keys,
            jnp.repeat(cell_valid, 4),
            entity_tables[2],
            counts[2],
        )
        faces = entity_tables[2][face_order]
        edge_routes = jnp.asarray(((0, 1), (1, 2), (2, 0)), dtype=jnp.int32)
        edge_keys = jnp.sort(faces[:, edge_routes], axis=2).reshape((-1, 2))
        orders.append(
            _first_appearance_order(
                edge_keys,
                jnp.repeat(
                    jnp.arange(faces.shape[0]) < counts[2],
                    3,
                ),
                entity_tables[1],
                counts[1],
            )
        )
        orders.append(face_order)
    else:
        raise ValueError(
            "Distributed simplex publication supports dimensions two and three."
        )
    orders.append(jnp.arange(top_rows.shape[0], dtype=jnp.int64))
    return tuple(orders)


def _logical_multiblock_topology_id(
    dimension: int,
    vertex_keys: jax.Array,
    cell_keys: jax.Array,
    cell_rows: jax.Array,
    entity_tables: tuple[jax.Array, ...],
    entity_ids: tuple[jax.Array, ...],
    counts: tuple[int, ...],
    presentation: _LogicalPresentation,
    /,
) -> str:
    cell_capacity = cell_keys.shape[0]
    cell_prefix = jnp.arange(cell_capacity) < counts[-1]
    canonical_blocks = []
    for group, (name, kind, count) in enumerate(
        zip(
            presentation.names,
            presentation.kinds,
            presentation.counts,
            strict=True,
        )
    ):
        selected = jnp.nonzero(
            cell_prefix & (presentation.groups == group),
            size=cell_capacity,
            fill_value=0,
        )[0]
        block_ids = cell_keys[selected, 0]
        block_rows = cell_rows[selected]
        valid = jnp.ones(block_rows.shape, dtype=jnp.bool_)
        canonical_blocks.append(
            {
                "name": name,
                "cell_kind": kind,
                "global_ids": _logical_array_tree_fingerprint(block_ids, (count,)),
                "global_vertices": _logical_array_tree_fingerprint(
                    block_rows, (count, dimension + 1)
                ),
                "vertex_valid": _logical_array_tree_fingerprint(
                    valid, (count, dimension + 1)
                ),
            }
        )
    top_ids = cell_keys[presentation.order, 0]
    top_rows = cell_rows[presentation.order]
    canonical_boundary_masks = _logical_boundary_masks(
        cell_rows, counts[-1], entity_tables, counts
    )
    entity_orders = _logical_entity_orders(top_rows, counts[-1], entity_tables, counts)
    logical_entity_ids = tuple(
        (top_ids if degree == dimension else identifiers[entity_orders[degree]])
        for degree, identifiers in enumerate(entity_ids)
    )
    boundary_masks = tuple(
        mask if degree == dimension else mask[entity_orders[degree]]
        for degree, mask in enumerate(canonical_boundary_masks)
    )
    names = (
        ("vertices", "edges", "cells")
        if dimension == 2
        else ("vertices", "edges", "faces", "cells")
    )
    entity_set_ids = []
    for degree, (identifiers, boundary, count) in enumerate(
        zip(logical_entity_ids, boundary_masks, counts, strict=True)
    ):
        subset_id = canonical_fingerprint(
            {
                "kind": "entity-subset",
                "name": "boundary",
                "mask": _logical_array_tree_fingerprint(boundary, (count,)),
            }
        )
        active = jnp.ones(identifiers.shape, dtype=jnp.bool_)
        entity_set_ids.append(
            canonical_fingerprint(
                {
                    "kind": "entity-set",
                    "name": names[degree],
                    "intrinsic_dimension": degree,
                    "entity_ids": _logical_array_tree_fingerprint(identifiers, (count,)),
                    "active_mask": _logical_array_tree_fingerprint(active, (count,)),
                    "subsets": [subset_id],
                }
            )
        )
    incidence_ids = []
    for degree in range(1, dimension + 1):
        upper_keys = (
            jnp.sort(top_rows, axis=1) if degree == dimension else entity_tables[degree]
        )
        upper_ids = top_ids if degree == dimension else entity_ids[degree]
        lower_ids = entity_ids[degree - 1]
        positions = np.arange(degree + 1, dtype=np.int32)
        columns = jnp.asarray(
            np.stack([positions[positions != removed] for removed in range(degree + 1)]),
            dtype=jnp.int32,
        )
        faces = upper_keys[:, columns].reshape((-1, degree))
        lower_table = jnp.where(
            entity_tables[degree - 1] >= 0,
            entity_tables[degree - 1],
            jnp.iinfo(jnp.int64).max,
        )
        rows = _key_positions(lower_table, faces)
        valid = jnp.repeat(jnp.arange(upper_keys.shape[0]) < counts[degree], degree + 1)
        if bool(jax.device_get(jnp.any(valid & (rows < 0)))):
            raise ValueError("Logical simplex incidence is not face closed.")
        signs = jnp.broadcast_to(
            jnp.asarray(np.where(positions % 2, -1, 1), dtype=jnp.int64),
            (upper_keys.shape[0], degree + 1),
        )
        if degree == dimension:
            inversions = jnp.zeros((top_rows.shape[0],), dtype=jnp.int64)
            for first, second in combinations(range(dimension + 1), 2):
                inversions += top_rows[:, first] > top_rows[:, second]
            orientation = jnp.where(inversions % 2, -1, 1)
            signs = signs * orientation[:, None]
        incidence = jnp.stack(
            (
                lower_ids[jnp.maximum(rows, 0)],
                jnp.repeat(upper_ids, degree + 1),
                signs.reshape(-1),
            ),
            axis=1,
        )
        sentinel = jnp.iinfo(jnp.int64).max
        padded = jnp.where(valid[:, None], incidence, sentinel)
        order = jnp.lexsort(padded.T[::-1])
        ordered = padded[order]
        incidence_count = counts[degree] * (degree + 1)
        incidence_ids.append(
            canonical_fingerprint(
                {
                    "kind": "oriented-incidence",
                    "degree": degree,
                    "lower": entity_set_ids[degree - 1],
                    "upper": entity_set_ids[degree],
                    "canonical_incidence": _logical_array_tree_fingerprint(
                        ordered, (incidence_count, 3)
                    ),
                }
            )
        )
    complex_id = canonical_fingerprint(
        {
            "kind": "cell-complex-topology",
            "entity_sets": entity_set_ids,
            "incidences": incidence_ids,
        }
    )
    return canonical_fingerprint(
        {
            "kind": "cell-mesh-topology",
            "topological_dimension": dimension,
            "vertex_global_ids": _logical_array_tree_fingerprint(
                vertex_keys[:, 0], (counts[0],)
            ),
            "blocks": canonical_blocks,
            "cell_complex": complex_id,
        }
    )


def _logical_publication(
    accepted_source: CellMeshingResult,
    states: AdaptiveSimplexState,
    owners: jax.Array,
    source_exterior: jax.Array,
    layout: AdaptiveSimplexLayout,
    /,
    *,
    placement_arrays: tuple[tuple[str, jax.Array], ...] = (),
    original_source: CellMeshingResult | None,
    uniform_refinement: BisectionUniformRefinement | None = None,
    numeric_version: str | None = None,
) -> _LogicalPublication:
    """Order and compact numerical global arrays on devices, not on a host."""

    mesh = states.mesh
    parts = mesh.cells.shape[0]
    width = mesh.cells.shape[2]
    part_vertices = jnp.take_along_axis(
        mesh.vertex_ids, mesh.cells.reshape((parts, -1)), axis=1
    )
    cell_vertices = part_vertices.reshape((-1, width))
    cell_keys, cell_order, cells = _canonical_rows(
        mesh.cell_ids.reshape((-1, 1)), mesh.cell_active.reshape((-1,))
    )
    cell_rows = cell_vertices[cell_order]
    physical_owners = jnp.broadcast_to(
        jnp.arange(parts, dtype=jnp.int32)[:, None], mesh.cell_ids.shape
    )
    cell_owners = physical_owners.reshape((-1,))[cell_order]
    solver_cell_owners = owners.reshape((-1,))[cell_order]
    used_vertex_ids = jnp.where(
        jnp.repeat(mesh.cell_active, width, axis=1),
        cell_vertices.reshape(mesh.cell_ids.shape[0], -1),
        -1,
    ).reshape(-1)
    vertex_valid = mesh.vertex_active.reshape(-1) & jnp.isin(
        mesh.vertex_ids.reshape(-1), used_vertex_ids
    )
    vertex_keys, vertex_order, vertices = _canonical_rows(
        mesh.vertex_ids.reshape((-1, 1)), vertex_valid
    )
    vertex_positions = _key_positions(
        jnp.where(vertex_keys >= 0, vertex_keys, jnp.iinfo(jnp.int64).max),
        cell_vertices.reshape((-1, 1)),
    ).reshape((-1, width))
    incident_owners = jnp.where(
        mesh.cell_active.reshape((-1, 1)),
        physical_owners.reshape((-1, 1)),
        parts,
    )
    vertex_owners = jnp.full((vertex_keys.shape[0],), parts, dtype=jnp.int32)
    vertex_owners = vertex_owners.at[jnp.maximum(vertex_positions, 0)].min(
        jnp.broadcast_to(incident_owners, vertex_positions.shape)
    )
    vertex_owners = jnp.where(
        jnp.arange(vertex_keys.shape[0]) < vertices, vertex_owners, -1
    )
    coordinates = mesh.coordinates.reshape((-1, mesh.coordinates.shape[-1]))[
        vertex_order
    ].astype(jnp.float64)
    entity_tables = [vertex_keys]
    entity_ids = [vertex_keys[:, 0]]
    entity_owners = [vertex_owners]
    counts = [vertices]
    arrays = {
        "cell_global_ids": cell_keys[:, 0],
        "cell_vertices": cell_rows,
        "coordinates": coordinates,
        "vertex_global_ids": vertex_keys[:, 0],
    }
    shapes = {
        "cell_global_ids": (cells,),
        "cell_vertices": (cells, width),
        "coordinates": (vertices, mesh.coordinates.shape[-1]),
        "vertex_global_ids": (vertices,),
    }
    premise = require_original_meshing_source(
        accepted_source if original_source is None else original_source
    )
    source = accepted_source.mesh
    for degree in range(1, width - 1):
        columns = np.asarray(
            tuple(combinations(range(width), degree + 1)), dtype=np.int32
        )
        keys = jnp.sort(cell_rows[:, columns], axis=2).reshape((-1, degree + 1))
        valid = jnp.repeat(jnp.arange(cell_keys.shape[0]) < cells, columns.shape[0])
        table, order, count = _canonical_rows(keys, valid)
        ids, retired_keys, retired_ids = _inherited_entity_ids(
            accepted_source,
            degree,
            table,
            count,
            parts,
            original_source,
        )
        arrays[f"epoch/retired_entity_keys_{degree}"] = retired_keys
        arrays[f"epoch/retired_entity_ids_{degree}"] = retired_ids
        routing = jnp.repeat(cell_owners, columns.shape[0])[order]
        entity_tables.append(table)
        entity_ids.append(ids)
        entity_owners.append(routing)
        counts.append(count)
        arrays[f"entity_global_ids_{degree}"] = ids
        arrays[f"entity_vertices_{degree}"] = table
        shapes[f"entity_global_ids_{degree}"] = (count,)
        shapes[f"entity_vertices_{degree}"] = (count, degree + 1)
    entity_tables.append(cell_keys)
    entity_ids.append(cell_keys[:, 0])
    entity_owners.append(cell_owners)
    counts.append(cells)
    presentation = _logical_presentation(
        accepted_source,
        premise,
        original_source,
        states,
        cell_keys,
        cell_order,
        cells,
        uniform_refinement,
    )
    sharding = states.mesh.cell_ids.sharding
    arrays = {name: jax.device_put(value, sharding) for name, value in arrays.items()}
    topology_digest = logical_array_value_collection_digest(
        {
            name: value
            for name, value in arrays.items()
            if name != "coordinates" and not name.startswith("epoch/")
        },
        logical_shapes={
            name: shape for name, shape in shapes.items() if name != "coordinates"
        },
    )
    if len(presentation.names) == 1:
        topology_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-topology",
                "dimension": source.topological_dimension,
                "blocks": [(presentation.names[0], presentation.kinds[0])],
                "arrays": topology_digest,
            }
        )
    else:
        topology_id = _logical_multiblock_topology_id(
            source.topological_dimension,
            vertex_keys,
            cell_keys,
            cell_rows,
            tuple(entity_tables),
            tuple(entity_ids),
            tuple(counts),
            presentation,
        )
    target_numeric_version = (
        accepted_source.mesh.numeric_version
        if numeric_version is None
        else str(numeric_version)
    )
    if len(presentation.names) == 1:
        coordinate_digest = logical_array_value_collection_digest(
            {"coordinates": arrays["coordinates"]},
            logical_shapes={"coordinates": shapes["coordinates"]},
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry",
                "topology": topology_id,
                "ambient_dimension": source.ambient_dimension,
                "coordinates": coordinate_digest,
            }
        )
    else:
        geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry-layout",
                "topology": topology_id,
                "ambient_dimension": source.ambient_dimension,
                "coordinate_count": vertices,
                "coordinate_dtype": str(np.dtype(coordinates.dtype)),
            }
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "cell-mesh-geometry",
                "layout": geometry_layout_id,
                "coordinates": _logical_array_tree_fingerprint(
                    arrays["coordinates"], shapes["coordinates"]
                ),
                "numeric_version": target_numeric_version,
            }
        )
    for degree, identifiers in enumerate(entity_ids):
        arrays[f"entity_active_{degree}"] = jax.device_put(
            jnp.arange(identifiers.shape[0]) < counts[degree], sharding
        )
    arrays["cell_owners"] = jax.device_put(solver_cell_owners, sharding)
    arrays.update(
        {
            "epoch/coordinates": mesh.coordinates,
            "epoch/vertex_ids": mesh.vertex_ids,
            "epoch/vertex_active": mesh.vertex_active,
            "epoch/cells": mesh.cells,
            "epoch/cell_ids": mesh.cell_ids,
            "epoch/cell_active": mesh.cell_active,
            "epoch/facet_neighbors": mesh.facet_neighbors,
            "epoch/vertex_half_facets": states.vertex_half_facets,
            "epoch/tuples": states.tuples,
            "epoch/tags": states.tags,
            "epoch/blocks": states.blocks,
            "epoch/generations": states.generations,
            "epoch/parents": states.parents,
            "epoch/children": states.children,
            "epoch/bisection_vertices": states.bisection_vertices,
            "epoch/retired": states.retired,
            "epoch/cell_classes": states.cell_classes,
            "epoch/facet_classes": states.facet_classes,
            "epoch/vertex_parents": states.vertex_parents,
            "epoch/vertex_levels": states.vertex_levels,
            "epoch/vertex_removal": states.vertex_removal,
            "epoch/vertex_protected": states.vertex_protected,
            "epoch/protected_codes": states.protected_codes,
            "epoch/refine_rejected": states.refine_rejected,
            "epoch/coarsen_marked": states.coarsen_marked,
            "epoch/cursors": states.cursors,
            "epoch/clocks": states.clocks,
            "epoch/counters": states.counters,
            "epoch/source_exterior": source_exterior,
            "epoch/layout_parameters": jax.device_put(
                jnp.broadcast_to(
                    jnp.asarray(
                        (
                            layout.maximum_closure_iterations,
                            layout.maximum_coarsening_passes,
                        ),
                        dtype=jnp.int64,
                    ),
                    (parts, 2),
                ),
                sharding,
            ),
        }
    )
    arrays.update(placement_arrays)
    arrays["epoch/solver_cell_owners"] = jax.device_put(owners, sharding)
    return _LogicalPublication(
        tuple(sorted(arrays.items())),
        tuple(counts),
        tuple(jax.device_put(value, sharding) for value in entity_tables),
        tuple(jax.device_put(value, sharding) for value in entity_ids),
        tuple(jax.device_put(value, sharding) for value in entity_owners),
        topology_id,
        geometry_id,
    )


def _collective_local_commits(
    partitioned: PartitionedAdaptiveSimplex, states: AdaptiveSimplexState, /
) -> tuple[_LocalCommit, ...]:
    """All processes reject before exposing any locally resolved transition."""

    failure: MeshingFailure | None = None
    commits: tuple[_LocalCommit, ...] = ()
    try:
        commits = _addressable_commits(partitioned, states)
    except MeshingFailure as error:
        failure = error
    except ValueError as error:
        failure = MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            str(error),
            stage="distributed-device-commit",
        )
    _require_collective_local_success(failure)
    return commits


def _require_collective_local_success(failure: MeshingFailure | None, /) -> None:
    """Reduce failure categories after an owning-process transaction barrier."""

    categories = tuple(MeshingFailureCategory)
    code = 0 if failure is None else categories.index(failure.category) + 1
    verdicts = np.asarray(multihost_utils.process_allgather(np.int32(code), tiled=False))
    rejected = verdicts[verdicts > 0]
    if rejected.size:
        selected = categories[int(np.min(rejected)) - 1]
        if failure is not None and failure.category is selected:
            raise failure
        raise MeshingFailure(
            selected,
            "An owning process rejected the exact topology, lineage, or geometry "
            "resolution of the distributed epoch; no part was published.",
            stage="distributed-device-commit",
        )


def _lower_collective_geometry(
    local: _LocalCommit,
    logical: _LogicalPublication,
    original: CellMeshingResult | InitialCollectiveMeshEvidence,
    /,
) -> tuple[
    tuple[CellBlock, ...],
    CellGeometrySpec,
    np.ndarray,
    np.ndarray,
    dict[str, str],
    np.ndarray,
]:
    """Lower only this closure's exact source coefficients and reference charts."""
    from ..discretization._cell_geometry import (
        _require_full_p1_source,
        BarycentricCellGeometryElement,
        CellGeometryRestrictionSource,
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from ..discretization._cell_geometry_validity import cell_geometry_id

    if logical.geometry_projection is None:
        raise ValueError(
            "Local geometry lowering requires its consumed collective projection."
        )
    arrays = dict(logical.geometry_projection.addressable_arrays(local.partition_index))
    edit = local.committed.edit
    fixed: list[TopologyEditBlock] = []
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError(
                "Simplex source geometry cannot lower a polyhedral topology edit."
            )
        fixed.append(block)
    target_ids = np.concatenate(tuple(block.cell_ids for block in fixed))
    target_rows = np.concatenate(tuple(block.cells for block in fixed))
    if isinstance(original, CellMeshingResult):
        root_elements, _, _ = original.geometry.resolve(original.mesh)
        roots = []
        for block, element in zip(original.mesh.blocks, root_elements, strict=True):
            roots.append((block.name, block.cell_kind, _require_full_p1_source(element)))
        source_geometry_id = cell_geometry_id(original.geometry)
        source_topology_id = original.mesh.topology_id
    else:
        roots = [("surface", "triangle", coordinate_lagrange_element("triangle", 1))]
        source_geometry_id = original.coordinate_geometry_id
        source_topology_id = original.topology_id
    elements = {}
    rows = {}
    ids = {}
    routes = {}
    parent_cells = {}
    parent_vertices = {}
    source_banks = {}
    kinds = {}
    for root_name, root_kind, root_element in roots:
        table = arrays[f"geometry/cell_ids/{root_name}"]
        table = jnp.where(table >= 0, table, jnp.iinfo(jnp.int64).max)
        positions = _key_positions(
            table[:, None], jnp.asarray(target_ids[:, None], dtype=jnp.int64)
        )
        found = np.asarray(jax.device_get(positions), dtype=np.int64)
        selected = np.flatnonzero(found >= 0)
        positions = jnp.asarray(found[selected], dtype=jnp.int64)
        actions = np.asarray(
            jax.device_get(arrays[f"geometry/action_weights/{root_name}"][positions]),
            dtype=np.float64,
        )
        action_counts = np.asarray(
            jax.device_get(arrays[f"geometry/action_counts/{root_name}"][positions]),
            dtype=np.int32,
        )
        if (
            actions.ndim != 4
            or actions.shape[2:]
            != (root_element.local_dof_count, root_element.local_dof_count)
            or action_counts.shape != (selected.size,)
            or np.any(action_counts < 0)
            or np.any(action_counts > actions.shape[1])
        ):
            raise ValueError(
                "Collective full action stacks lost their source columns or declared depth."
            )
        coefficient_routes = np.asarray(
            jax.device_get(arrays[f"geometry/routes/{root_name}"][positions]),
            dtype=np.int64,
        )
        parents = np.asarray(
            jax.device_get(arrays[f"geometry/parent_cell_ids/{root_name}"][positions]),
            dtype=np.int64,
        )
        corners = np.asarray(
            jax.device_get(arrays[f"geometry/parent_vertex_ids/{root_name}"][positions]),
            dtype=np.int64,
        )
        for index, target in enumerate(selected):
            element = root_element
            for action in reversed(actions[index, : action_counts[index]]):
                element = BarycentricCellGeometryElement(element, action)
            name = (
                root_name
                if action_counts[index] == 0
                else f"{root_name}/coefficient-action/{element.element_id}"
            )
            elements[name] = element
            kinds[name] = root_kind
            source_banks[name] = root_name
            rows.setdefault(name, []).append(target_rows[target])
            ids.setdefault(name, []).append(target_ids[target])
            routes.setdefault(name, []).append(coefficient_routes[index])
            parent_cells.setdefault(name, []).append(parents[index])
            parent_vertices.setdefault(name, []).append(corners[index])
    if sum(len(values) for values in ids.values()) != target_ids.size:
        raise ValueError("A local cell has no unique accepted scientific source chart.")
    # Canonical block order matches the serial commit owner (sorted by name), so
    # one logical target has one topology and geometry identity on every route.
    names = tuple(sorted(elements))
    if isinstance(original, CellMeshingResult):
        for block in original.mesh.blocks:
            if block.name not in ids:
                continue
            original_positions = {
                int(identifier): row for row, identifier in enumerate(block.global_ids)
            }
            order = np.argsort(
                np.asarray(
                    [
                        original_positions[int(identifier)]
                        for identifier in ids[block.name]
                    ],
                    dtype=np.int64,
                ),
                kind="stable",
            )
            for groups in (rows, ids, routes, parent_cells, parent_vertices):
                groups[block.name] = [groups[block.name][index] for index in order]
    coordinate_ids = np.unique(
        np.concatenate(tuple(np.asarray(routes[name]).reshape(-1) for name in names))
    )
    coordinate_table = arrays["geometry/coordinate_ids"]
    positions = _key_positions(
        jnp.where(coordinate_table >= 0, coordinate_table, jnp.iinfo(jnp.int64).max)[
            :, None
        ],
        jnp.asarray(coordinate_ids[:, None], dtype=jnp.int64),
    )
    if bool(jax.device_get(jnp.any(positions < 0))):
        raise ValueError("A scientific source chart references an absent coordinate DOF.")
    coordinates = np.asarray(
        jax.device_get(arrays["geometry/coordinates"][positions]), dtype=np.float64
    )
    owners = np.asarray(
        jax.device_get(arrays["geometry/coordinate_owners"][positions]), dtype=np.int32
    )
    exact_source = None
    if isinstance(original, CellMeshingResult):
        from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
        from ..discretization._exact_power_geometry import (
            ExactPowerCellGeometryLinearActionSource,
            ExactPowerCellGeometryRestrictionSource,
            ExactPowerCellGeometrySource,
        )

        _uniform_charge(coordinate_ids.size, coordinates.nbytes + coordinate_ids.nbytes)
        if np.any(coordinate_ids < 0) or np.any(
            coordinate_ids >= original.geometry.coordinates.shape[0]
        ):
            raise ValueError(
                "A collective coefficient SCI is absent from its original source bank."
            )
        if not np.array_equal(
            coordinates, np.asarray(original.geometry.coordinates)[coordinate_ids]
        ):
            raise ValueError(
                "A collective physical coefficient bank differs from its original SCI routes."
            )
        source = original.geometry.exact_source
        if isinstance(source, ExactPlcCellGeometrySource):
            _uniform_charge(coordinate_ids.size, coordinate_ids.size * 32)
            exact_source = source._with_witnesses(
                source.vertex_strata[coordinate_ids],
                source.vertex_rows[coordinate_ids],
                source.vertex_parameters[coordinate_ids],
            )
        elif isinstance(
            source,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            _uniform_charge(coordinate_ids.size, coordinate_ids.size * 24)
            exact_source = ExactPowerCellGeometryRestrictionSource(
                source,
                np.zeros((0, 4), dtype=np.float64),
                np.stack(
                    (coordinate_ids, np.full(coordinate_ids.shape, -1, dtype=np.int64)),
                    axis=1,
                ),
                np.full(coordinate_ids.shape, -1, dtype=np.int64),
            )
    blocks = tuple(
        CellBlock(
            name,
            kinds[name],
            np.asarray(rows[name], dtype=np.int32),
            global_ids=np.asarray(ids[name], dtype=np.int64),
        )
        for name in names
    )
    origin = CellGeometryRestrictionSource(
        source_geometry_id,
        source_topology_id,
        {name: np.asarray(parent_cells[name], dtype=np.int64) for name in names},
        {name: np.asarray(parent_vertices[name], dtype=np.int64) for name in names},
        block_source_blocks=source_banks,
    )
    geometry = CellGeometrySpec(
        elements,
        {
            name: np.searchsorted(
                coordinate_ids, np.asarray(routes[name], dtype=np.int64)
            ).astype(np.int32)
            for name in names
        },
        coordinates,
        restriction_source=origin,
        exact_source=exact_source,
    )
    order = key_rows(
        target_ids[:, None],
        np.concatenate(tuple(np.asarray(block.global_ids) for block in blocks))[:, None],
    )
    return blocks, geometry, coordinate_ids, owners, source_banks, order


def _published_cell_mesh(
    local: _LocalCommit,
    logical: _LogicalPublication,
    evidence: CollectiveMeshEvidence,
    partitioned: PartitionedAdaptiveSimplex,
    /,
) -> CellMesh:
    """Lower a local closure with globally accepted IDs, ownership, and evidence."""

    edit = local.committed.edit
    original = _scientific_preparation_source(partitioned.prepared.adaptation)
    (
        blocks,
        local_geometry,
        coordinate_ids,
        coordinate_owners,
        source_banks,
        cell_order,
    ) = _lower_collective_geometry(local, logical, original)
    probe = CellMesh(edit.coordinates, blocks, vertex_global_ids=edit.vertex_global_ids)
    projection = logical.publication_projection
    if projection is None:
        raise ValueError(
            "Local publication requires its consumed fixed-capacity entity projection."
        )
    projection.require_source(logical.arrays)
    projected = dict(projection.addressable_arrays(local.partition_index))
    ids = []
    owners = []
    for degree in range(probe.topological_dimension + 1):
        valid = np.asarray(
            jax.device_get(projected[f"entity/{degree}/valid"]), dtype=np.bool_
        )
        keys = np.asarray(
            jax.device_get(projected[f"entity/{degree}/keys"]), dtype=np.int64
        )[valid]
        queries = entity_keys(probe, degree)
        positions = key_rows(keys, queries)
        if np.any(positions < 0):
            raise ValueError(
                "A local closure entity is absent from its consumed publication projection."
            )
        identifiers = np.asarray(
            jax.device_get(projected[f"entity/{degree}/ids"]), dtype=np.int64
        )[valid]
        ownership = np.asarray(
            jax.device_get(projected[f"entity/{degree}/owners"]), dtype=np.int32
        )[valid]
        ids.append(identifiers[positions])
        owners.append(ownership[positions])
        if degree == 0:
            coordinates = np.asarray(
                jax.device_get(projected["coordinates"]), dtype=np.float64
            )[valid]
            if not np.array_equal(coordinates[positions], edit.coordinates):
                raise ValueError(
                    "Local coordinates differ from their consumed logical geometry projection."
                )
    if local.cell_exterior is None:
        raise ValueError("Publication requires consumed neighborhood exterior witnesses.")
    columns = np.asarray(
        [
            [
                column
                for column in range(probe.topological_dimension + 1)
                if column != opposite
            ]
            for opposite in range(probe.topological_dimension + 1)
        ],
        dtype=np.int32,
    )
    cell_vertices = edit.vertex_global_ids[
        np.concatenate(tuple(np.asarray(block.vertices) for block in blocks))
    ]
    facets = np.sort(cell_vertices[:, columns], axis=2)
    facet_rows = key_rows(
        entity_keys(probe, probe.topological_dimension - 1),
        facets.reshape((-1, probe.topological_dimension)),
    )
    physical = np.zeros(
        (probe.entity_set(probe.topological_dimension - 1).count,), dtype=np.bool_
    )
    local_exterior = local.cell_exterior[cell_order]
    np.logical_or.at(physical, facet_rows, local_exterior.reshape((-1,)))
    incidence = np.bincount(facet_rows, minlength=physical.size)
    complete = np.all(
        (incidence[facet_rows].reshape(local_exterior.shape) == 2) | local_exterior,
        axis=1,
    )
    storage = CellMeshStorage(
        logical.counts,
        ids,
        owners,
        partition_index=local.partition_index,
        partition_count=partitioned.parts.part_count,
        local_coordinates=edit.coordinates,
        local_blocks=blocks,
        global_coordinate_count=(
            original.geometry.coordinates.shape[0]
            if isinstance(original, CellMeshingResult)
            else original.global_entity_counts[0]
        ),
        coordinate_global_ids=coordinate_ids,
        coordinate_owner=coordinate_owners,
        local_geometry=local_geometry,
        geometry_source_blocks=source_banks,
        logical_coordinate_geometry_id=evidence.coordinate_geometry_id,
        geometry_projection=logical.geometry_projection,
        local_physical_boundary_facets=physical,
        local_neighborhood_complete=complete,
        neighborhood_depth=(
            partitioned.prepared.adaptation.policy.distribution.halo_width
            if partitioned.prepared.adaptation.policy.distribution is not None
            else 0
        ),
        logical_topology_id=logical.topology_id,
        logical_geometry_id=logical.geometry_id,
        evidence_id=evidence.evidence_id,
        logical_arrays=logical.arrays,
    )
    return CellMesh(
        edit.coordinates,
        blocks,
        storage=storage,
        numeric_version=f"adaptation:{partitioned.prepared.adaptation.prepared_id}",
    )


def _local_entity_lineage(
    partitioned: PartitionedAdaptiveSimplex,
    local: _LocalCommit,
    mesh: CellMesh,
    relation: EntityRelations,
    /,
) -> EntityLineage:
    """Shard owned creation/deletion without declaring absent ghost rows deleted."""

    record = _entity_lineage(local.source, mesh, relation)
    degree = record.dimension
    deleted = np.asarray(record.deleted_source_ids, dtype=np.int64)
    if deleted.size:
        keys = entity_keys(local.source, degree)
        local_ids = np.asarray(local.source.entity_set(degree).entity_ids, dtype=np.int64)
        rows = key_rows(local_ids[:, None], deleted[:, None])
        _, source_owner = _source_entity_values(
            partitioned.source_entities, degree, keys[rows]
        )
        deleted = deleted[source_owner == local.partition_index]
    storage = mesh.storage
    if storage is None:
        raise ValueError("Local lineage requires accepted owner-local storage.")
    target_ids = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
    target_owned = np.asarray(storage.entity_owned[degree], dtype=np.bool_)
    created = np.asarray(record.created_target_ids, dtype=np.int64)
    created = created[np.isin(created, target_ids[target_owned])]
    return EntityLineage(
        degree,
        partitioned.prepared.adaptation.source.mesh.entity_set(degree).entity_set_id,
        mesh.entity_set(degree).entity_set_id,
        record.source_global_ids,
        record.target_global_ids,
        record.relation_kinds,
        created_target_ids=created,
        deleted_source_ids=deleted,
    )


class _LocalTarget(NamedTuple):
    local: _LocalCommit
    mesh: CellMesh
    lineage: MeshLineage
    geometry: CellGeometrySpec


def _local_target(
    partitioned: PartitionedAdaptiveSimplex,
    local: _LocalCommit,
    logical: _LogicalPublication,
    evidence: CollectiveMeshEvidence,
    /,
) -> _LocalTarget:
    """Lower one owner-local target carrier and its exact lineage."""
    prepared = partitioned.prepared.adaptation
    mesh = _published_cell_mesh(local, logical, evidence, partitioned)
    lineage = MeshLineage(
        prepared.source.mesh.topology_id,
        mesh.topology_id,
        tuple(
            _local_entity_lineage(partitioned, local, mesh, relation)
            for relation in local.committed.edit.relations
        ),
    )
    if mesh.storage is None:
        raise ValueError("Accepted local publication lost its coordinate-map storage.")
    return _LocalTarget(local, mesh, lineage, mesh.storage.restore_geometry())


def _collective_plc_premises(
    partitioned: PartitionedAdaptiveSimplex,
    target: _LocalTarget,
    evidence: CollectiveMeshEvidence,
    /,
) -> tuple[GlobalEmbeddingCertificate, DomainCoverageCertificate] | None:
    """New-target embedding and domain coverage for represented-PLC transfer.

    Collective: the selector depends only on the shared policy and source, so
    every owner enters the producer together. Other transfers need no premise.
    """
    prepared = partitioned.prepared.adaptation
    if not prepared.source.associations or not isinstance(
        prepared.policy.association_transfer, PlcAssociationTransfer
    ):
        return None
    original = _scientific_preparation_source(prepared)
    if not isinstance(original, CellMeshingResult) or original.certification is None:
        raise ValueError(
            "Distributed PLC transfer requires the original serial embedding and domain-coverage theorem."
        )
    certification = original.certification
    domain = certification.request.domain
    if (
        certification.embedding is None
        or certification.coverage is None
        or not isinstance(domain, PiecewiseLinearDomain)
        or certification.request.cell_regions is None
    ):
        raise ValueError(
            "Distributed PLC transfer requires the original serial embedding and domain-coverage theorem."
        )
    storage = target.mesh.storage
    if storage is None:
        raise ValueError("Accepted local publication lost its coordinate-map storage.")
    _, states, _ = restore_partitioned_mesh_state(storage, evidence)
    _, embedding, coverage = certify_collective_premises(
        target.mesh,
        target.geometry,
        domain,
        original.mesh,
        original.geometry,
        certification.embedding,
        certification.coverage,
        np.asarray(certification.request.cell_regions),
        states,
        evidence.initial_states,
        logical_arrays=storage.logical_arrays,
        source_partition_id=storage.evidence_id,
    )
    return embedding, coverage


def _local_adaptation_result(
    partitioned: PartitionedAdaptiveSimplex,
    staged: _LocalTarget,
    premises: tuple[GlobalEmbeddingCertificate, DomainCoverageCertificate] | None,
    logical: _LogicalPublication,
    evidence: CollectiveMeshEvidence,
    started: float,
    /,
) -> MeshAdaptationResult:
    """Bind one certified local publication to the existing transition owners."""

    prepared = partitioned.prepared.adaptation
    local, mesh, lineage, geometry = staged
    committed = local.committed
    edit = committed.edit
    original = _scientific_preparation_source(prepared)
    patches, zones, labels, attributes = lower_mesh_organization(
        original,
        mesh,
        dict(logical.arrays),
        scope_projections=logical.scope_projections,
        attribute_projections=logical.attribute_projections,
    )
    region_boundaries = _region_boundary_transition(
        prepared.source, mesh, lineage, patches, zones, labels
    )
    region_evidence = None
    if prepared.source.region_evidence is not None:
        from ._compartments import revalidate_region_evidence

        renewal = revalidate_region_evidence(
            prepared.source,
            mesh,
            geometry,
            zones,
            patches,
            lineage=lineage,
            limits=prepared.policy.limits,
        )
        zones, patches, region_evidence = (
            renewal.zones,
            renewal.patches,
            renewal.region_evidence,
        )
    associations = ()
    if prepared.source.associations:
        transfer = prepared.policy.association_transfer
        if isinstance(transfer, PlcAssociationTransfer):
            if premises is None:
                raise ValueError(
                    "Distributed PLC transfer requires its collective target premises."
                )
            associations = transfer.propagate(
                prepared.source,
                lineage,
                mesh,
                geometry=geometry,
                embedding=premises[0],
                coverage=premises[1],
            )
        elif isinstance(transfer, BRepAssociationTransfer):
            associations = transfer.propagate(prepared.source, lineage, mesh)
        elif isinstance(transfer, SurfaceAssociationTransfer):
            # The collectively receipted transfer replaces the policy's unbound one.
            if logical.surface_transfer is None:
                raise ValueError(
                    "Distributed surface transfer requires its collective source receipts."
                )
            associations = logical.surface_transfer.propagate(
                prepared.source, lineage, mesh, geometry=geometry
            )
        else:
            raise ValueError(
                "Associated distributed source requires its authoritative association transfer."
            )
    target = certify_owner_local_cell_mesh(
        mesh,
        prepared.source,
        evidence,
        audit_policy=prepared.policy.audit_policy,
        geometry=geometry,
        patches=patches,
        zones=zones,
        labels=labels,
        attributes=attributes,
        associations=associations,
        region_evidence=region_evidence,
        region_boundary_evidence=region_boundaries,
        scope_projections=logical.scope_projections,
        attribute_projections=logical.attribute_projections,
        storage_binding=logical.storage_binding,
        collective_certificates=premises,
    )
    stencil = VertexInterpolationStencil(
        prepared.source.mesh.entity_set(0).entity_set_id,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
        edit.stencil_sources,
        edit.stencil_weights,
        edit.stencil_valid,
        preserves_constants=True,
    )
    kind = (
        MeshTransitionKind.REMESH
        if committed.refined and committed.coarsened
        else MeshTransitionKind.REFINE
        if committed.refined
        else MeshTransitionKind.COARSEN
        if committed.coarsened
        else MeshTransitionKind.PARTITION
    )
    transition = CellMeshTransition(
        prepared.source.mesh.mesh_id,
        prepared.source.mesh.topology_id,
        target,
        lineage,
        kind,
        vertex_stencil=stencil,
        parents=edit.refinement,
    )
    transfer = stencil.as_transfer(
        local.source.vertex_global_ids,
        source_topology_id=prepared.source.mesh.topology_id,
        target_topology_id=mesh.topology_id,
        preserves_linear=True,
        source_coordinates=local.source.coordinates,
        target_coordinates=mesh.coordinates,
    )
    status = (
        MeshAdaptationStatus.PASS_LIMIT
        if committed.pass_limited
        else MeshAdaptationStatus.PARTIAL
        if committed.partial
        else MeshAdaptationStatus.COMPLETE
    )
    from ._bisection import _rebind_bisection_presentation_blocks

    hierarchy = _rebind_bisection_presentation_blocks(committed.hierarchy, target)
    outcome = _RouteOutcome(
        status,
        target,
        transition,
        lineage,
        stencil,
        transfer,
        None,
        committed.evidence,
        hierarchy,
    )
    elapsed = time.monotonic() - started
    source_distribution = prepared.policy.distribution
    partition_policy = prepared.policy.partition_policy
    if source_distribution is None or partition_policy is None:
        raise ValueError("Local publication requires the prepared distribution policy.")
    source_evidence = prepared.source.collective_evidence
    if source_evidence is None:
        source_ids = partitioned.source_entities[mesh.topological_dimension][1]
        source_owners = partitioned.source_entities[mesh.topological_dimension][2]
        source_count = source_ids.shape[0]
    else:
        source_ids = source_evidence.entity_ids[-1]
        source_owners = source_evidence.entity_owners[-1]
        source_count = source_evidence.global_entity_counts[-1]
    cells = lineage.entity_lineage(mesh.topological_dimension)
    source_table = jnp.asarray(source_ids[:source_count], dtype=jnp.int64)
    source_order = jnp.argsort(source_table, stable=True)
    source_rows, defined = _local_logical_lookup(
        source_table[source_order],
        source_order,
        np.asarray(cells.source_global_ids, dtype=np.int64),
    )
    if not np.all(defined):
        raise ValueError(
            "A cell lineage occurrence is absent from the accepted predecessor."
        )
    distribution = prepare_owner_local_distribution_transition(
        source_distribution,
        MeshPart(source_distribution.part.name, target),
        lineage,
        source_rows,
        _rows_of(
            np.asarray(mesh.entity_set(mesh.topological_dimension).entity_ids),
            cells.target_global_ids,
        ),
        source_ids,
        source_owners,
        policy=partition_policy,
        restart_repack=partitioned.restart_repack,
        restart_proof=logical.restart_proof,
    )
    return MeshAdaptationResult(
        prepared, outcome, _compliance(prepared, outcome, elapsed), distribution, elapsed
    )


def _collective_status(statuses: jax.Array, /) -> int:
    """Union integer status bits using backend-supported scalar reductions."""

    shifts = jnp.arange(8, dtype=jnp.int32)
    bits = (statuses[:, None] >> shifts) & 1
    present = jnp.sum(bits, axis=0, dtype=jnp.int32) > 0
    flags = jnp.sum(present.astype(jnp.int32) << shifts, dtype=jnp.int32)
    return int(jax.device_get(flags))


def commit_partitioned_adaptive_simplex(
    partitioned: PartitionedAdaptiveSimplex, states: AdaptiveSimplexState, /
) -> tuple[MeshAdaptationResult, ...]:
    if not isinstance(partitioned, PartitionedAdaptiveSimplex):
        raise TypeError("partitioned must be PartitionedAdaptiveSimplex.")
    if not isinstance(states, AdaptiveSimplexState):
        raise TypeError("states must be AdaptiveSimplexState.")
    preparation = partitioned.prepared.adaptation
    receipt = partitioned.execution_evidence
    if receipt is None:
        receipt = partitioned.prepared.execution_evidence
    allowance = _phase_allowance(preparation, receipt)
    with _uniform_execution(allowance) as budget:
        _retain_preparation(partitioned)
        workspace = current_native_host_workspace()
        if workspace is None:
            raise RuntimeError(
                "Compiled source retention lost its actual native workspace."
            )
        workspace.retain_owner(states)
        baseline = partitioned.states.counters
        if baseline.shape != states.counters.shape or not bool(
            jax.device_get(jnp.all(baseline >= 0) & jnp.all(states.counters >= baseline))
        ):
            raise ValueError(
                "Collective compiled counters lost their actual prepared-source baseline."
            )
        budget.charge(
            work=int(
                jax.device_get(
                    jnp.sum(
                        states.counters[:, AdaptiveSimplexCounter.BISECTIONS]
                        - baseline[:, AdaptiveSimplexCounter.BISECTIONS],
                        dtype=jnp.int64,
                    )
                )
            )
        )
        results = _run_partitioned_adaptive_simplex_commit(partitioned, states)
    if budget.evidence is None:
        return results
    record = NativeExecutionRecord(
        budget.evidence, preparation_evidence=receipt, owner_id=preparation.prepared_id
    )
    return _bind_collective_execution_receipt(preparation, results, record)


def _run_partitioned_adaptive_simplex_commit(
    partitioned: PartitionedAdaptiveSimplex, states: AdaptiveSimplexState, /
) -> tuple[MeshAdaptationResult, ...]:
    """Collectively publish every process-addressable accepted refinement closure.

    Raw forest, source-map and semantic banks are checked before projection.
    Fixed-capacity collective receipts then lower the exact received closure;
    no host-global mesh or variable-shaped global lookup is used by a local
    constructor. The tuple contains only this process's addressable partitions.
    """

    if not isinstance(partitioned, PartitionedAdaptiveSimplex):
        raise TypeError("partitioned must be PartitionedAdaptiveSimplex.")
    if not isinstance(states, AdaptiveSimplexState) or (
        states.mesh.cells.shape[:2]
        != (partitioned.parts.part_count, partitioned.layout.cell_capacity)
    ):
        raise ValueError("states must be the stacked part states of this epoch.")
    started = time.monotonic()
    flags = _collective_status(states.status_flags)
    require_applied_status(flags, DeviceEpoch.BISECTION)
    source = partitioned.prepared.adaptation.source
    try:
        original = _scientific_preparation_source(partitioned.prepared.adaptation)
    except ValueError as error:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            str(error),
            stage="distributed-device-commit",
        ) from error
    if isinstance(original, CellMeshingResult):
        if original.certification is None or original.certification.embedding is None:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "Distributed publication requires the original source's scientific embedding theorem.",
                stage="distributed-device-commit",
            )
    else:
        original.require_passed()
    if (
        partitioned.source_witness is None
        or source.boundary is not None
        or source.mesh.periodic_topology is not None
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "Distributed source restriction requires an actual source-map witness "
            "and a nonperiodic, explicitly bound source boundary model.",
            stage="distributed-device-commit",
        )
    local = _collective_local_commits(partitioned, states)
    evidence = CollectiveMeshEvidence(partitioned, states)
    limits = partitioned.prepared.adaptation.policy.limits
    counts = evidence.global_entity_counts
    resources = (
        ("vertices", counts[0], limits.maximum_vertices),
        ("cells", counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            counts[-1] * len(counts),
            limits.maximum_connectivity_entries,
        ),
    )
    if any(actual > maximum for _, actual, maximum in resources):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Accepted logical topology exceeds the exact requested global resource controls.",
            stage="distributed-device-commit",
            requested=tuple(
                (f"maximum_{name}", maximum) for name, _, maximum in resources
            ),
            achieved=tuple((name, actual) for name, actual, _ in resources),
        )
    logical = _LogicalPublication(
        evidence.logical_arrays,
        evidence.global_entity_counts,
        evidence.entity_keys,
        evidence.entity_ids,
        evidence.entity_owners,
        evidence.topology_id,
        evidence.geometry_id,
    )
    distribution = partitioned.prepared.adaptation.policy.distribution
    if distribution is None:
        raise ValueError("Distributed publication lost its prepared ownership.")
    if evidence.uniform_refinement is None:
        neighborhood = expand_partitioned_simplex_neighborhood(
            partitioned.parts,
            states,
            partitioned.states,
            partitioned.source_exterior,
            halo_width=distribution.halo_width,
            cell_capacity=partitioned.layout.cell_capacity,
            vertex_capacity=partitioned.layout.vertex_capacity,
        )
    else:
        neighborhood = _packed_neighborhood(partitioned, states, evidence)
    neighborhood_flags = _collective_status(neighborhood.status)
    require_applied_status(neighborhood_flags, DeviceEpoch.BISECTION)
    coordinate_table = jnp.where(
        logical.entity_keys[0] >= 0,
        logical.entity_keys[0],
        jnp.iinfo(jnp.int64).max,
    )
    coordinate_rows = _key_positions(
        coordinate_table,
        neighborhood.cell_vertices.reshape((-1, 1)),
    )
    closure_vertices = jnp.repeat(
        neighborhood.cell_valid, neighborhood.cell_vertices.shape[-1]
    )
    if bool(jax.device_get(jnp.any(closure_vertices & (coordinate_rows < 0)))):
        raise ValueError(
            "A closure vertex is absent from accepted logical coordinate ownership."
        )
    canonical_coordinates = dict(logical.arrays)["coordinates"][
        jnp.maximum(coordinate_rows, 0)
    ].reshape(neighborhood.cell_coordinates.shape)
    neighborhood = eqx.tree_at(
        lambda work: work.cell_coordinates,
        neighborhood,
        canonical_coordinates,
    )
    logical = logical._replace(
        arrays=tuple(
            sorted(
                (
                    *logical.arrays,
                    ("closure/cell_ids", neighborhood.cell_ids),
                    ("closure/cell_vertices", neighborhood.cell_vertices),
                    ("closure/cell_coordinates", neighborhood.cell_coordinates),
                    ("closure/cell_classes", neighborhood.cell_classes),
                    ("closure/status", neighborhood.status),
                    ("closure/cell_exterior", neighborhood.cell_exterior),
                    ("closure/cell_valid", neighborhood.cell_valid),
                    ("closure/cell_owner", neighborhood.cell_owner),
                    ("closure/root_cell_ids", neighborhood.root_cell_ids),
                    ("closure/source_cell_ids", neighborhood.source_cell_ids),
                    ("closure/source_vertex_ids", neighborhood.source_vertex_ids),
                    ("closure/source_weights", neighborhood.source_weights),
                )
            )
        )
    )
    projection = PreparedPublicationLowering(
        original,
        evidence,
        neighborhood,
        vertex_capacity=partitioned.layout.vertex_capacity,
    ).execute()
    scopes = prepare_mesh_organization_scopes(
        original,
        evidence,
        projection,
        f"adaptation:{partitioned.prepared.adaptation.prepared_id}",
    )
    attributes = prepare_mesh_attribute_projections(scopes)
    storage_binding = CollectiveMeshStorageBinding(
        evidence, logical.arrays, logical.counts
    )
    transfer = partitioned.prepared.adaptation.policy.association_transfer
    surface_transfer = (
        transfer.with_receipts(
            transfer.support.prepare_receipts(logical.arrays, projection.geometry)
        )
        if isinstance(transfer, SurfaceAssociationTransfer)
        else None
    )
    logical = logical._replace(
        geometry_projection=projection.geometry,
        publication_projection=projection,
        scope_projections=scopes,
        attribute_projections=attributes,
        restart_proof=partitioned.restart_proof,
        storage_binding=storage_binding,
        surface_transfer=surface_transfer,
    )
    targets = _collective_local_phase(
        lambda: tuple(
            _local_target(
                partitioned,
                _neighborhood_commit(partitioned, owned, neighborhood),
                logical,
                evidence,
            )
            for owned in local
        )
    )
    premises = _collective_local_phase(
        lambda: tuple(
            _collective_plc_premises(partitioned, target, evidence) for target in targets
        )
    )
    return _collective_local_phase(
        lambda: tuple(
            _local_adaptation_result(
                partitioned, target, premise, logical, evidence, started
            )
            for target, premise in zip(targets, premises, strict=True)
        )
    )


_Phase = TypeVar("_Phase")


def _collective_local_phase(
    action: Callable[[], tuple[_Phase, ...]], /
) -> tuple[_Phase, ...]:
    """Run one owner-local phase; every process rejects if any owner failed."""
    values: tuple[_Phase, ...] = ()
    failure: MeshingFailure | None = None
    try:
        values = action()
    except MeshingFailure as error:
        failure = error
    except ValueError as error:
        failure = MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            str(error),
            stage="distributed-device-commit",
        )
    _require_collective_local_success(failure)
    return values


__all__ = [
    "PartitionedAdaptiveSimplex",
    "PreparedAdaptiveSimplex",
    "commit_adaptive_simplex",
    "commit_partitioned_adaptive_simplex",
    "commit_partitioned_mesh_restart",
    "prepare_partitioned_mesh_adaptation",
    "execute_partitioned_mesh_adaptation",
    "restore_partitioned_mesh_state",
    "validate_partitioned_mesh_evidence",
    "partition_adaptive_simplex",
    "prepare_adaptive_simplex",
]
