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
one device-to-host transfer and builds the target, its entity
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

import fractions as _owner_fractions
import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from itertools import combinations
from math import prod
from typing import Any, assert_never, final, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

import phydrax.discretization._adaptive_simplex as _dependency_phydrax_discretization__adaptive_simplex
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry as _owner_phydrax_discretization__cell_geometry
import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._cell_geometry_validity as _owner_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._cell_mesh as _dependency_phydrax_discretization__cell_mesh
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._coordinate_enclosure as _owner_phydrax_discretization__coordinate_enclosure
import phydrax.discretization._exact_plc_geometry as _owner_phydrax_discretization__exact_plc_geometry
import phydrax.discretization._exact_power_geometry as _owner_phydrax_discretization__exact_power_geometry
import phydrax.geometry._collective_domain_coverage as _dependency_phydrax_geometry__collective_domain_coverage
import phydrax.geometry._domain_coverage as _dependency_phydrax_geometry__mesh_certificates
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.meshing._adaptation as _dependency_phydrax_meshing__adaptation
import phydrax.meshing._adaptation as _owner_phydrax_meshing__adaptation
import phydrax.meshing._association as _dependency_phydrax_meshing__association
import phydrax.meshing._bisection as _dependency_phydrax_meshing__bisection
import phydrax.meshing._bisection as _owner_phydrax_meshing__bisection
import phydrax.meshing._bisection_kernel_host as bisection_host
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._collective_failure as collective_failure_owner
import phydrax.meshing._collective_geometry as _owner_phydrax_meshing__collective_geometry
import phydrax.meshing._collective_organization as _owner_phydrax_meshing__collective_organization
import phydrax.meshing._compartments as _owner_phydrax_meshing__compartments
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._device_status as _dependency_phydrax_meshing__device_status
import phydrax.meshing._distribution as _dependency_phydrax_meshing__distribution
import phydrax.meshing._distribution as _owner_phydrax_meshing__distribution
import phydrax.meshing._distribution_migration as _dependency_phydrax_meshing__distribution_migration
import phydrax.meshing._distribution_migration as _owner_phydrax_meshing__distribution_migration
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._initial_certification as _owner_phydrax_meshing__initial_certification
import phydrax.meshing._lineage as _dependency_phydrax_meshing__lineage
import phydrax.meshing._lineage_records as _dependency_phydrax_meshing__lineage_records
import phydrax.meshing._mesh_part as mesh_part_owner
import phydrax.meshing._organization as _dependency_phydrax_meshing__organization
import phydrax.meshing._organization_records as _dependency_phydrax_meshing__organization_records
import phydrax.meshing._publication_lowering as _dependency_phydrax_meshing__publication_lowering
import phydrax.meshing._restart_distribution as _owner_phydrax_meshing__restart_distribution
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit
from phydrax.discretization._reference_cell_types import CellBlock

from .._execution_array import _project_local_leading_axes
from .._execution_runtime import ExecutionRuntime
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
    logical_array_value_collection_digest,
)
from .._meshcore import (
    exact_orient2d,
    MeshcoreError,
    MeshcoreStatus,
    MeshcoreUnavailableError,
    NativeExecutionScope,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._entity_keys import key_positions
from ._measurements import NativeExecutionRecord


# Bisection-tree paths are int64 bit strings below their root.
_MAXIMUM_DEPTH = 62


class _Anchor(NamedTuple):
    """Host description of the epoch source shared by preparation and commit."""

    source: bisection_host.Source
    start: bisection_host.Start
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

    adaptation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout
    state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    anchor: _Anchor
    execution_evidence: NativeExecutionRecord | None
    source_topology_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        adaptation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
        layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
        state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
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
            raise ValueError(
                "device_adaptation.record_generations.ready must be reachable from active leaves; got the bisection forest has an unreachable record"
            )
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
    found = entity_keys_owner.key_rows(np.sort(facet_keys, axis=1), keys)
    classes = np.where(found >= 0, facet_classes[np.maximum(found, 0)] + 1, 0)
    return classes.reshape((count, width)).astype(np.int32)


def _forest(
    start: bisection_host.Start,
    source: bisection_host.Source,
    request: Any,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Any:
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
        bisection_host.uniform_count(rows.size, rows.nbytes * 4, budget=budget)
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


def _vertices(
    start: bisection_host.Start,
    source: bisection_host.Source,
    request: Any,
    /,
) -> Any:
    """Source vertices by global ID, then vertices issued by the start."""

    count, dimension = source.vertex_ids.size, source.dimension
    growth = start.front.growth
    sources, weights = bisection_host.resolved_stencil(growth, count, dimension)
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
            (
                source.vertex_ids,
                bisection_host.global_vertices(issued, source, start.next_vertex),
            )
        ),
        "parents": parents,
        "protected": protected,
    }


def _require_preparation_receipt(
    adaptation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    receipt: NativeExecutionRecord | None,
    /,
) -> None:
    if receipt is None:
        return
    if type(receipt) is not NativeExecutionRecord:
        raise TypeError(
            "device_adaptation.require_preparation_receipt.NativeExecutionRecord must provide its ended native execution record; got an absent or incompatible value"
        )
    receipt.require_valid()
    if receipt.owner_id != adaptation.prepared_id or int(np.asarray(receipt.status)) != 0:
        raise ValueError(
            "device_adaptation.require_preparation_receipt.execution_record must bind this successful scientific operation; got an incompatible value"
        )
    phase = receipt
    while phase is not None:
        if phase.owner_id != adaptation.prepared_id:
            raise ValueError(
                "device_adaptation.require_preparation_receipt.phase must exclude a phase from a different scientific operation; got preparation execution record includes a phase from a different scientific operation"
            )
        if int(np.asarray(phase.status)) != 0:
            raise ValueError(
                "device_adaptation.require_preparation_receipt.phase must exclude a failed preparation phase; got preparation execution record includes a failed preparation phase"
            )
        if phase.preparation_evidence is None and (
            int(np.asarray(phase.source_preparation_work_units))
            or int(np.asarray(phase.source_preparation_geometry_queries))
            or float(np.asarray(phase.preparation_seconds))
        ):
            raise ValueError(
                "device_adaptation.require_preparation_receipt.phase must provide ended preparation records, not source-field aliases; got an absent or incompatible value"
            )
        phase = phase.preparation_evidence


def _phase_allowance(
    adaptation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    receipt: NativeExecutionRecord | None,
    /,
) -> bisection_host.UniformAllowance:
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "device_adaptation.phase_allowance.work must leave a positive operation allowance; got completed preparation exhausted the operation allowance",
            stage="device-preparation-admission",
        )
    return bisection_host.UniformAllowance.from_limits(
        limits,
        work=limits.maximum_work_units - work,
        geometry_queries=limits.maximum_geometry_queries - queries,
        seconds=limits.maximum_wall_seconds - seconds,
    )


def _retain_preparation(
    value: PreparedAdaptiveSimplex | PartitionedAdaptiveSimplex,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    budget.retain_basis((value,))


@contextmanager
@checked
def device_coordinate_budget(
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    /,
    *,
    scope: NativeExecutionScope,
) -> Iterator[
    _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
]:
    ticket = scope.batch(0)
    ticket.settle()
    budget = (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
    )
    with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
        yield budget


def _run_prepared_simplex(
    adaptation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PreparedAdaptiveSimplex:
    policy = adaptation.policy
    device = policy.device_policy
    constraints = adaptation.constraints
    mesh = adaptation.source.mesh
    if mesh.storage is not None:
        initial_evidence = adaptation.source.collective_evidence
        if not isinstance(
            initial_evidence,
            _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
        ):
            raise ValueError(
                "device_adaptation.run_prepared_simplex.initial_evidence must provide its native initial certification; got an absent or incompatible value"
            )
        initial_evidence.require_passed()
    source = bisection_host.prepared_source(mesh)
    empty = np.zeros((0,), dtype=np.int64)
    request = bisection_host.prepared_request(
        source,
        empty,
        empty,
        constraints.protected_edge_keys,
        constraints.protected_vertex_ids,
        constraints.cell_classes,
        constraints.facet_classes,
        policy.maximum_closure_iterations,
    )
    if not isinstance(
        adaptation.request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
    ):
        raise TypeError(
            "device_adaptation.run_prepared_simplex.adaptation must provide its marked request; got an absent or incompatible value"
        )
    hierarchy = adaptation.request.hierarchy
    if hierarchy is not None and not isinstance(
        hierarchy, _dependency_phydrax_meshing__bisection.BisectionHierarchy
    ):
        raise TypeError(
            "device_adaptation.run_prepared_simplex.hierarchy must provide its bisection hierarchy; got an absent or incompatible value"
        )
    start = bisection_host.prepared_start(
        source,
        request,
        policy.compatibility,
        hierarchy,
        bisection_host.UniformAllowance.from_limits(policy.limits),
        adaptation.source,
        future_refinement=True,
        scope=scope,
        budget=budget,
    )
    forest = _forest(start, source, request, scope=scope, budget=budget)
    vertices = _vertices(start, source, request)
    if policy.distribution is None:
        # ty: ignore[unresolved-attribute]
        vertex_capacity, cell_capacity = device.capacities(
            vertices["ids"].size, forest["ids"].size
        )
    else:
        # The dense forest is immutable preparation, not one owner's execution
        # bucket. Partitioning below reapplies the requested per-owner capacities.
        # ty: ignore[unresolved-attribute]
        requested_vertices = (
            _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
                vertices["ids"].size, device.growth_factor
            )
            if device.vertex_capacity is None
            else device.vertex_capacity
        )
        # ty: ignore[unresolved-attribute]
        requested_cells = (
            _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
                forest["ids"].size, device.growth_factor
            )
            if device.cell_capacity is None
            else device.cell_capacity
        )
        vertex_capacity = max(
            requested_vertices,
            _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
                vertices["ids"].size, 1.0
            ),
        )
        cell_capacity = max(
            requested_cells,
            _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
                forest["ids"].size, 1.0
            ),
        )
    protected = request.protected_codes
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
        source.dimension,
        mesh.ambient_dimension,
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        protected_edge_capacity=_dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
            protected.size, 1.0
        ),
        maximum_closure_iterations=policy.maximum_closure_iterations,
        # ty: ignore[unresolved-attribute]
        maximum_coarsening_passes=device.maximum_coarsening_passes,
    )
    state = _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_state(
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
        protected_edges=np.stack(
            (
                protected // bisection_host.SHIFT,
                protected % bisection_host.SHIFT,
            ),
            axis=1,
        ),
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
            restored_facets = _dependency_phydrax_discretization__adaptive_simplex.restored_facet_classes(
                _dependency_phydrax_discretization__adaptive_simplex.simplex_work_bound(
                    state
                )._replace(facet_classes=facets)
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
                "device_adaptation.run_prepared_simplex.evidence must provide global identity banks; got an absent or incompatible value"
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


@checked
def prepare_adaptive_simplex(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    /,
    *,
    policy: _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy,
    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy | None = None,
) -> PreparedAdaptiveSimplex:
    """Validate, label, and pad one certified simplex source into a device state.

    ``policy.route`` must be DEVICE_BISECTION; ``policy.device_policy`` fixes the
    capacity bucket. Labels are the host longest-edge Maubach labels (or the
    supplied hierarchy's); incompatible labellings are rejected unless the
    policy requests the explicit uniform barycentric refinement. Protection and
    organization classes are resolved exactly as for the host route.
    """

    if (
        policy.route
        is not _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.DEVICE_BISECTION
    ):
        raise ValueError(
            "device_adaptation.prepare_adaptive_simplex.policy must use the DEVICE_BISECTION route; got adaptive simplex epochs require the device_bisection route"
        )
    limits = policy.limits
    allowance = bisection_host.UniformAllowance.from_limits(limits)
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        adaptation = _dependency_phydrax_meshing__adaptation.prepare_mesh_adaptation(
            source,
            _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation(
                hierarchy=hierarchy
            ),
            policy=policy,
        )
        with (
            bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
            budget.ticket.native_target(),
        ):
            prepared = _run_prepared_simplex(adaptation, scope=scope, budget=budget)
    if scope.evidence is None:
        return prepared
    return PreparedAdaptiveSimplex(
        adaptation,
        prepared.layout,
        prepared.state,
        prepared.anchor,
        execution_evidence=NativeExecutionRecord(
            scope.evidence, owner_id=adaptation.prepared_id
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"device_adaptation.node_keys.depth must not exceed {_MAXIMUM_DEPTH}; got deeper ancestry",
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
        rows = entity_keys_owner.key_rows(table, ancestor[open_])
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

    @classmethod
    def seed(
        cls,
        source: bisection_host.Source,
        classes: np.ndarray,
        facets: np.ndarray,
        issuers: tuple[int, int],
        template: _HostEpoch,
        protected_vertices: np.ndarray,
        /,
    ) -> _HostEpoch:
        """Construct authenticated roots with no binary ancestry."""
        count, vertices = source.cells.ids.size, source.vertex_ids.size
        return cls(
            source.cells.ids,
            np.ones(count, dtype=np.bool_),
            source.cells.rows,
            source.cells.tuples,
            source.cells.tags,
            source.cells.blocks,
            source.cells.generations,
            np.full(count, -1, dtype=np.int64),
            np.full((count, 2), -1, dtype=np.int64),
            np.full(count, -1, dtype=np.int64),
            np.zeros(count, dtype=np.bool_),
            np.zeros(count, dtype=np.bool_),
            np.zeros(count, dtype=np.bool_),
            source.vertex_ids,
            np.ones(vertices, dtype=np.bool_),
            source.coordinates,
            np.full((vertices, 2), -1, dtype=np.int64),
            np.zeros(vertices, dtype=template.vertex_levels.dtype),
            np.full(vertices, -1, dtype=template.vertex_removal.dtype),
            np.asarray((vertices, count, *issuers), dtype=np.int64),
            template.counters,
            0,
            np.asarray(classes, dtype=template.cell_classes.dtype),
            facets,
            np.isin(source.vertex_ids, protected_vertices),
            np.zeros(0, dtype=np.int64),
        )

    @classmethod
    def from_forest(
        cls,
        forest: dict[str, np.ndarray],
        vertices: dict[str, np.ndarray],
        start: bisection_host.Start,
        counters: np.ndarray,
        /,
        *,
        template: _HostEpoch | None = None,
    ) -> _HostEpoch:
        """Construct an ID-ordered host epoch from admitted forest banks."""
        count, vertex_count = forest["ids"].size, vertices["ids"].size
        return cls(
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
            np.zeros(
                vertex_count,
                dtype=np.int64 if template is None else template.vertex_levels.dtype,
            ),
            np.full(
                vertex_count,
                -1,
                dtype=np.int64 if template is None else template.vertex_removal.dtype,
            ),
            np.asarray(
                (
                    vertex_count,
                    count,
                    start.next_vertex + start.front.growth.levels.size,
                    start.front.next_cell,
                ),
                dtype=np.int64,
            ),
            counters,
            0,
            forest["cell_classes"],
            forest["facet_classes"],
            vertices["protected"],
            np.zeros(0, dtype=np.int64),
        )


def _host_epoch(
    state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState, /
) -> _HostEpoch:
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
        # Lossless widening: host predicates resolve the device coordinates.
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "device_adaptation.cell_lineage.within must lie inside a surviving target cell; got a removed source cell lies in no target cell of the device epoch",
            stage="device-commit",
        )
    return _Lineage(
        np.where(inside >= 0, inside, ids),
        ids[removed],
        ids[targets[within]],
        inside,
    )


def _growth(anchor: _Anchor, host: _HostEpoch, count: int, /) -> bisection_host.Growth:
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
    return bisection_host.Growth(
        np.concatenate((initial.parents, parents)),
        np.concatenate((initial.weights, weights)),
        np.concatenate((initial.levels, levels)),
    )


def _supports(
    anchor: _Anchor,
    host: _HostEpoch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Surviving vertices and reference weights spanning each removed source vertex."""

    count = anchor.source.vertex_ids.size
    removal = np.asarray(host.vertex_removal, dtype=np.int64)[:count]
    parents = np.asarray(host.vertex_parents, dtype=np.int64)[:count]
    steps = []
    for index in np.unique(removal[removal >= 0]):
        vertices = np.flatnonzero(removal == index)
        steps.append((vertices, parents[vertices, 0], parents[vertices, 1]))
    return bisection_host.reverse_supports(steps, count, anchor.source.dimension)


class _Committed(NamedTuple):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy
    evidence: _dependency_phydrax_meshing__bisection.BisectionEvidence
    refined: bool
    coarsened: bool
    partial: bool
    pass_limited: bool


def _owned_retired_entities(
    source: bisection_host.Source,
    keys: tuple[np.ndarray, ...],
    identifiers: tuple[np.ndarray, ...],
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Retain issued IDs for entities whose vertices are local."""
    result = []
    for table, ids in zip(keys, identifiers, strict=True):
        owned = np.all(np.isin(table, source.vertex_ids), axis=1)
        result.append((np.searchsorted(source.vertex_ids, table[owned]), ids[owned]))
    return tuple(result)


def _owned_uniform_refinement(
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None,
    owned_ids: np.ndarray,
    records: bisection_host.Records,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None:
    if lineage is None:
        return None
    bisection_host.uniform_count(
        records.parent_ids.size + owned_ids.size,
        192 * (records.parent_ids.size + owned_ids.size),
        budget=budget,
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
            bisection_host.uniform_count(1, 192, budget=budget)
            identifier = pending.pop()
            if identifier in visited:
                raise ValueError(
                    "device_adaptation.owned_uniform_refinement.identifier must exclude a repeated binary descendant identity; got uniform ownership contains a repeated binary descendant identity"
                )
            visited.add(identifier)
            descendants = branches.get(identifier)
            if descendants is None:
                leaves.add(identifier)
            else:
                pending.extend(descendants)
        overlap = leaves & owned
        if overlap and overlap != leaves:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
                "device_adaptation.owned_uniform_refinement.overlap must own the uniform source sibling tree; got an incompatible value",
                stage="uniform-source-ownership",
                entity_ids=(int(np.asarray(lineage.parent_ids)[root]),),
            )
        if overlap:
            selected.append(root)
    return lineage.select(np.asarray(selected, dtype=np.int64))


def _committed_uniform_inverse(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    cells: bisection_host.Cells,
    coarsening: bisection_host.Coarsening,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    bisection_host.Cells,
    bisection_host.Coarsening,
]:
    if prepared.anchor.start.uniform_refinement is None:
        return cells, coarsening
    baseline = np.asarray(jax.device_get(prepared.state.counters), dtype=np.int64)
    final_arrays = np.asarray(host.counters, dtype=np.int64)
    if (
        baseline.shape != final_arrays.shape
        or np.any(baseline < 0)
        or np.any(final_arrays < baseline)
    ):
        raise ValueError(
            "device_adaptation.committed_uniform_inverse.baseline must retain a nonnegative prepared-source baseline; got an incompatible value"
        )
    split_work = int(
        final_arrays[
            _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS
        ]
        - baseline[
            _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS
        ]
    )
    limits = prepared.adaptation.policy.limits
    if split_work > limits.maximum_work_units:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "device_adaptation.committed_uniform_inverse.split_work must leave work for the uniform inverse; got device splits exhausted the transaction work allowance before its uniform inverse",
            stage="uniform-source-commit",
        )
    cells, coarsening = _execute_uniform_commit(
        prepared, host, cells, coarsening, scope=scope, budget=budget
    )
    return cells, coarsening


def _execute_uniform_commit(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    cells: bisection_host.Cells,
    coarsening: bisection_host.Coarsening,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    bisection_host.Cells,
    bisection_host.Coarsening,
]:

    source, start = prepared.anchor.source, prepared.anchor.start
    uniform = start.uniform_refinement
    if uniform is None:
        return cells, coarsening
    count = int(host.cursors[1])
    active = np.flatnonzero(host.cell_active[:count])
    bisection_host.uniform_count(count, count * 32, budget=budget)
    marks = np.array(host.coarsen_marked[:count], dtype=np.bool_, copy=True)
    children = np.asarray(host.children[:count], dtype=np.int64)
    retired = np.asarray(host.retired[:count], dtype=np.bool_)
    for generation in np.unique(np.asarray(host.generations[:count]))[::-1]:
        parents = np.flatnonzero(
            (np.asarray(host.generations[:count]) == generation)
            & np.all(children >= 0, axis=1)
        )
        bisection_host.uniform_count(parents.size, parents.nbytes, budget=budget)
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
    bisection_host.uniform_count(
        cells.rows.size,
        4 * cells.ids.size * width * source.dimension * 8
        + 2 * cells.ids.size * width * 8,
        budget=budget,
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
            "device_adaptation.execute_uniform_commit.encoded must agree on their organization class; got conflicting values"
        )
    facet_classes = encoded[first] - 1
    request = bisection_host.Request(
        np.zeros((0,), dtype=np.int64),
        requested,
        codes,
        protected,
        np.zeros(source.cells.ids.shape, dtype=np.int64),
        unique,
        facet_classes,
        prepared.adaptation.policy.maximum_closure_iterations,
    )
    flags = bisection_host.Flags(
        marks[active] | np.isin(cells.ids, coarsening.restored.ids),
        np.zeros(cells.ids.shape, dtype=np.bool_),
        np.asarray(host.cell_classes)[active],
    )
    updated = bisection_host.uniform_inverse(
        source,
        start,
        request,
        cells,
        flags,
        (unique, facet_classes),
        coarsening,
        scope=scope,
        budget=budget,
    )
    original_parents = np.setdiff1d(updated.undone, coarsening.undone)
    if original_parents.size:
        selected = np.searchsorted(np.asarray(uniform.parent_ids), original_parents)
        removed = np.asarray(uniform.child_ids)[selected].reshape(-1)
        cells = bisection_host.joined(
            bisection_host.select(cells, ~np.isin(cells.ids, removed)),
            bisection_host.select(
                updated.restored, np.isin(updated.restored.ids, original_parents)
            ),
        )
        cells = bisection_host.select(cells, np.argsort(cells.ids, kind="stable"))
    return cells, updated


def _retained_scientific_block_authority(
    prepared: PreparedAdaptiveSimplex, /
) -> tuple[np.ndarray, np.ndarray]:
    """Bind every retained initial cell to its authored source block ancestry."""

    initial = _host_epoch(prepared.state)
    count = int(initial.cursors[1])
    identifiers = np.asarray(initial.cell_ids[:count], dtype=np.int64)
    original = _scientific_preparation_source(prepared.adaptation)
    match original:
        case _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence():
            # Initial surface publication has one declared surface cell block;
            # authored patch incidence is retained in its organization banks.
            return identifiers, np.zeros(identifiers.shape, dtype=np.int32)
        case _dependency_phydrax_meshing__result.CellMeshingResult():
            authority = {
                int(identifier): block_index
                for block_index, block in enumerate(original.mesh.blocks)
                for identifier in block.global_ids
            }
        case _:
            raise TypeError(
                "device_adaptation.retained_source.source must carry a supported source certification; got unknown source provenance"
            )
    uniform = prepared.anchor.start.uniform_refinement
    if uniform is not None:
        for parent, children in zip(uniform.parent_ids, uniform.child_ids, strict=True):
            parent_id = int(parent)
            if parent_id not in authority:
                raise ValueError(
                    "retained_uniform.parent must bind an authored source cell; got "
                    f"{parent_id}"
                )
            for child in children:
                authority[int(child)] = authority[parent_id]
    blocks = np.empty(identifiers.shape, dtype=np.int32)
    for slot, identifier in enumerate(identifiers):
        ancestor = slot
        for _ in range(_MAXIMUM_DEPTH + 1):
            ancestor_id = int(initial.cell_ids[ancestor])
            if ancestor_id in authority:
                blocks[slot] = authority[ancestor_id]
                break
            parent = int(initial.parents[ancestor]) // 2
            if parent < 0 or parent >= count:
                raise ValueError(
                    "retained_cell.ancestry must reach its authored source block; got "
                    f"{int(identifier)}"
                )
            ancestor = parent
        else:
            raise ValueError(
                "retained_cell.ancestry must fit the admitted depth bound; got "
                f"{int(identifier)}"
            )
    return identifiers, blocks


@final
class _CommittedForest(StrictModule):
    """Active cells, retained binary records, and reverse supports for the host commit."""

    cells: bisection_host.Cells
    records: bisection_host.Records
    growth: bisection_host.Growth
    counters: np.ndarray
    rejected_refinements: np.ndarray
    coarsening: bisection_host.Coarsening


def _resolve_committed_forest(
    prepared: PreparedAdaptiveSimplex,
    anchor: _Anchor,
    source: bisection_host.Source,
    start: bisection_host.Start,
    host: _HostEpoch,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CommittedForest:
    """Active cells, retained binary records, and reverse supports for the host commit."""
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
                "device_adaptation.resolve_committed_forest.mapping must provide one raw presentation axis; got an absent or incompatible value"
            )
        bisection_host.uniform_count(
            cell_count + mapping.size, 32 * cell_count + 16 * mapping.size, budget=budget
        )
        used = active | ~retired
        source_blocks = blocks[used]
        if np.any((source_blocks < 0) | (source_blocks >= mapping.size)):
            raise ValueError(
                "device_adaptation.resolve_committed_forest.source_blocks must declare a source presentation block; got an raw cell or binary record has an undeclared source presentation block"
            )
        projected = mapping[source_blocks]
        if np.any((projected < 0) | (projected >= len(source.mesh.blocks))):
            raise ValueError(
                "device_adaptation.resolve_committed_forest.projected must provide its resident scientific source block; got missing its resident scientific source block"
            )
        # The source owner binds this map from real SCI block incidence. Only
        # the host commit representation changes; compiled blocks stay intact.
        blocks = np.full(blocks.shape, -1, dtype=np.int64)
        blocks[used] = projected
    generations = np.asarray(host.generations, dtype=np.int64)[:cell_count]
    lineage = _cell_lineage(anchor, host, cell_count)
    target = np.flatnonzero(active)
    cells = bisection_host.Cells(
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
    records = bisection_host.Records(
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
    coarsening = bisection_host.Coarsening(
        bisection_host.Cells(
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
        *_supports(anchor, host, scope=scope, budget=budget),
        int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.COARSENING_PASSES
            ]
        ),
        rejected_coarsenings,
    )
    return _CommittedForest(
        cells, records, growth, counters, rejected_refinements, coarsening
    )


def _committed(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Any:
    anchor = prepared.anchor
    source, start = anchor.source, anchor.start
    phase = _resolve_committed_forest(
        prepared=prepared,
        anchor=anchor,
        source=source,
        start=start,
        host=host,
        scope=scope,
        budget=budget,
    )
    cells = phase.cells
    records = phase.records
    growth = phase.growth
    counters = phase.counters
    rejected_refinements = phase.rejected_refinements
    coarsening = phase.coarsening
    # The compiled binary summary remains unchanged. Compatibility siblings
    # have one additional real inverse at this documented host commit boundary.
    cells, coarsening = _committed_uniform_inverse(
        prepared, host, cells, coarsening, scope=scope, budget=budget
    )
    rejected_coarsenings = coarsening.rejected_ids
    front = bisection_host.Front(
        cells,
        growth,
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        source.vertex_ids.size,
        int(host.cursors[3]),
    )
    refinement = bisection_host.Refinement(
        front,
        records,
        rejected_refinements,
        int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.ADMISSIBILITY_TESTS
            ]
        ),
        int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.CLOSURE_ITERATIONS
            ]
        ),
    )
    edit, retired_tables = bisection_host.build_bisection_edit(
        source, start, refinement, coarsening, cells, scope=scope, budget=budget
    )
    request = prepared.adaptation.request
    prior = (
        request.hierarchy
        if isinstance(
            request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
        )
        else None
    )
    if prior is not None and not isinstance(
        prior, _dependency_phydrax_meshing__bisection.BisectionHierarchy
    ):
        raise TypeError(
            "device_adaptation.committed.prior must provide its scientific hierarchy; got an absent or incompatible value"
        )
    source_authority = (
        _retained_scientific_block_authority(prepared)
        if prior is None and prepared.adaptation.source.mesh.storage is not None
        else None
    )
    hierarchy = bisection_host.target_hierarchy(
        source,
        start,
        cells,
        records,
        retired_tables,
        front,
        prior=prior,
        source_authority=source_authority,
    )
    evidence = _dependency_phydrax_meshing__bisection.BisectionEvidence(
        requested_refinements=int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.REQUESTED_REFINEMENTS
            ]
        ),
        accepted_refinements=int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.ACCEPTED_REFINEMENTS
            ]
        ),
        rejected_refinement_ids=rejected_refinements,
        admissibility_tests=int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.ADMISSIBILITY_TESTS
            ]
        ),
        bisections=int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS
            ]
        ),
        closure_iterations=int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.CLOSURE_ITERATIONS
            ]
        ),
        created_vertices=growth.levels.size,
        maximum_generation=int(np.max(cells.generations)),
        initially_compatible=start.compatible,
        incompatible_facets=start.incompatible,
        uniform_refinement_applied=start.uniform,
        requested_coarsenings=int(
            counters[
                _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.REQUESTED_COARSENINGS
            ]
        ),
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
        bool(
            flags
            & _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexStatus.PASS_LIMIT
        ),
    )


def _outcome(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Any:
    allowance = _phase_allowance(prepared.adaptation, prepared.execution_evidence)
    with bisection_host.uniform_budget(allowance, scope=scope, budget=budget):
        _retain_preparation(prepared, scope=scope, budget=budget)
        baseline = np.asarray(jax.device_get(prepared.state.counters), dtype=np.int64)
        if (
            baseline.shape != host.counters.shape
            or np.any(baseline < 0)
            or np.any(host.counters < baseline)
        ):
            raise ValueError(
                "device_adaptation.outcome.baseline must retain their prepared-source baseline; got missing their prepared-source baseline"
            )
    work = int(
        host.counters[
            _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS
        ]
        - baseline[
            _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS
        ]
    )
    with scope.batch(work) as ticket:
        ticket.consume(work)
    return _run_outcome(prepared, host, scope=scope, budget=budget)


def _run_outcome(
    prepared: PreparedAdaptiveSimplex,
    host: _HostEpoch,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__adaptation.RouteOutcome:
    _dependency_phydrax_meshing__device_status.require_committed_status(
        host.flags,
        host.coordinates,
        host.cells[host.cell_active],
        _dependency_phydrax_meshing__device_status.DeviceEpoch.BISECTION,
    )
    adaptation = prepared.adaptation
    allowance = _phase_allowance(adaptation, prepared.execution_evidence)
    with (
        bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
        budget.ticket.native_target(),
    ):
        committed = _committed(prepared, host, scope=scope, budget=budget)
    if not committed.refined and not committed.coarsened:
        status = (
            _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.PASS_LIMIT
            if committed.pass_limited
            else _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.PARTIAL
            if committed.partial
            else _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.UNCHANGED
        )
        return _dependency_phydrax_meshing__adaptation.unchanged(
            adaptation, committed.evidence, committed.hierarchy, status=status
        )
    match (committed.refined, committed.coarsened):
        case (True, False):
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REFINE
        case (False, True):
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
        case _:
            kind = _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH
    native = _dependency_phydrax_meshing__adaptation.finalize_native(
        adaptation,
        committed.edit,
        kind,
        conservative=not committed.coarsened,
        bisection_hierarchy=committed.hierarchy,
        scope=scope,
        budget=budget,
    )

    with bisection_host.uniform_budget(allowance, scope=scope, budget=budget):
        hierarchy = bisection_host.rebind_bisection_presentation_blocks(
            committed.hierarchy, native.target, scope=scope, budget=budget
        )
    if committed.pass_limited:
        status = _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.PASS_LIMIT
    elif committed.partial:
        status = _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.PARTIAL
    else:
        status = _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.COMPLETE
    return _dependency_phydrax_meshing__adaptation.RouteOutcome(
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


@checked
def _checked_state(
    prepared: PreparedAdaptiveSimplex,
    state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> None:
    prepared.adaptation.source.mesh.require_dense("Serial adaptive simplex commit")
    if state.mesh.signature_id != prepared.layout.mesh_signature_id:
        raise ValueError(
            "device_adaptation.checked_state.state must belong to this prepared epoch; got an incompatible value"
        )


@checked
def commit_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex,
    state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> _dependency_phydrax_meshing__adaptation.MeshAdaptationResult:
    """Commit one device epoch: one transfer, target, lineage, evidence.

    The target is assembled, organization inherited by scientific IDs, certified,
    and bound to its `CellMeshTransition`, `MeshLineage`, and sparse P1
    transfer exactly as the host bisection route does; the result's
    ``hierarchy`` prepares the next epoch. The epoch's cumulative
    ``status_flags`` decide acceptance: a terminal flag raises its
    `MeshingFailure` (capacity or closure bound: RESOURCE_EXHAUSTED, protected
    conflict: INVALID_SPECIFICATION, invalid geometry: QUALITY_REJECTED),
    NEEDS_HOST_RESOLUTION requires every committed cell to be certified
    positively oriented by host exact predicates (else QUALITY_REJECTED), and
    a pass-limited coarsening commits with status PASS_LIMIT.
    """

    _checked_state(prepared, state)
    started = time.monotonic()
    limits = prepared.adaptation.policy.limits
    allowance = _phase_allowance(prepared.adaptation, prepared.execution_evidence)
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        outcome = _outcome(prepared, _host_epoch(state), scope=scope, budget=budget)
    if scope.evidence is not None:
        outcome = outcome._replace(
            target=outcome.target.with_execution_evidence(
                NativeExecutionRecord(
                    scope.evidence,
                    preparation_evidence=prepared.execution_evidence,
                    owner_id=prepared.adaptation.prepared_id,
                )
            )
        )
    return _dependency_phydrax_meshing__adaptation.adaptation_result(
        prepared.adaptation, outcome, started
    )


@checked
def execute_device_bisection_route(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__adaptation.RouteOutcome:
    """One refine-then-coarsen device epoch of a marked request, then commit."""
    return _run_device_bisection_route(prepared, scope=scope, budget=budget)


def _run_device_bisection_route(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__adaptation.RouteOutcome:
    request = prepared.request
    # ty: ignore[unresolved-attribute]
    if request.coarsen_cell_ids.size and request.hierarchy is None:
        raise ValueError(
            "device_adaptation.run_device_bisection_route.request must provide the BisectionHierarchy of a previous bisection; got an absent or incompatible value"
        )
    allowance = _phase_allowance(prepared, None)
    with (
        bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
        budget.ticket.native_target(),
    ):
        simplex = _run_prepared_simplex(prepared, scope=scope, budget=budget)
    layout = simplex.layout
    refined = (
        _dependency_phydrax_discretization__adaptive_simplex.refine_adaptive_simplex(
            layout,
            simplex.state,
            # ty: ignore[unresolved-attribute]
            simplex.cell_marks(np.asarray(request.refine_cell_ids)),
        )
    )
    _dependency_phydrax_meshing__device_status.require_applied_status(
        int(refined.report.status),
        _dependency_phydrax_meshing__device_status.DeviceEpoch.BISECTION,
    )
    coarsened = (
        _dependency_phydrax_discretization__adaptive_simplex.coarsen_adaptive_simplex(
            layout,
            refined.state,
            # ty: ignore[unresolved-attribute]
            simplex.cell_marks(np.asarray(request.coarsen_cell_ids)),
        )
    )
    return _outcome(simplex, _host_epoch(coarsened.state), scope=scope, budget=budget)


@checked
def restore_partitioned_mesh_state(
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
) -> tuple[
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    jax.Array,
]:
    """Restore and recheck producer-owned raw state before accepted publication.

    The input is a storage/proof pair, not a successful result. Domain and
    source certification producers can consume this numerical source binding without
    a circular dependency on the result they are preparing.
    """
    if (
        evidence.topology_id != storage.logical_topology_id
        or evidence.geometry_id != storage.logical_geometry_id
        or evidence.evidence_id != storage.evidence_id
        or evidence.partition_count != storage.partition_count
    ):
        raise ValueError(
            "device_adaptation.restore_partitioned_mesh_state.evidence must match its immutable storage binding; got mismatched retained raw state proof"
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
            f"device_adaptation.restore_partitioned_mesh_state.epoch must include the epoch banks; got missing {missing}"
        )
    arrays = {name: values[f"epoch/{name}"] for name in names}
    if any(value.shape[0] != storage.partition_count for value in arrays.values()):
        raise ValueError(
            "device_adaptation.restore_partitioned_mesh_state.any must agree with accepted partition coverage; got conflicting values"
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
            "device_adaptation.restore_partitioned_mesh_state.any must match its accepted collective evidence; got mismatched retained numerical forest"
        )

    def restore(
        part: dict[str, jax.Array],
    ) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
        mesh = _dependency_phydrax_discretization__adaptive_simplex.MaskedSimplexMesh(
            *(part[name] for name in mesh_names)
        )
        return _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState(
            mesh, **{name: part[name] for name in state_names}
        )

    state = eqx.filter_vmap(restore)(arrays)
    width = arrays["cells"].shape[-1]
    parameters = arrays["layout_parameters"]
    if parameters.shape != (storage.partition_count, 2) or parameters.dtype != jnp.int64:
        raise ValueError(
            "device_adaptation.restore_partitioned_mesh_state.parameters must be int64 partition-by-two; got an incompatible value"
        )
    bounds = np.asarray(jax.device_get(jnp.max(parameters, axis=0)), dtype=np.int64)
    if not bool(jax.device_get(jnp.all(parameters == jnp.asarray(bounds)))):
        raise ValueError(
            "device_adaptation.restore_partitioned_mesh_state.parameters must agree on their declared iteration controls; got conflicting values"
        )
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
        width - 1,
        arrays["coordinates"].shape[-1],
        vertex_capacity=arrays["vertex_ids"].shape[1],
        cell_capacity=arrays["cell_ids"].shape[1],
        protected_edge_capacity=arrays["protected_codes"].shape[1],
        maximum_closure_iterations=int(bounds[0]),
        maximum_coarsening_passes=int(bounds[1]),
    )

    checks = _owner_phydrax_meshing__distribution.replay_logical_bisection_checks(
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
            "device_adaptation.restore_partitioned_mesh_state.checks must reproduce its raw certificates; got an incompatible value"
        )
    return layout, state, arrays["source_exterior"]


@checked
def retained_partition_states(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> tuple[
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    jax.Array,
]:
    """Consume accepted identity, then reuse the independently validated raw getter."""
    storage = source.mesh.storage
    evidence = source.collective_evidence
    if storage is None or not isinstance(
        evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
    ):
        raise ValueError(
            "device_adaptation.retained_partition_states.storage must provide accepted owner-local storage; got an absent or incompatible value"
        )
    if evidence.mesh_id != source.mesh.mesh_id:
        raise ValueError(
            "device_adaptation.retained_partition_states.evidence must match its accepted collective proof; got mismatched retained source identity"
        )
    evidence.require_passed()
    return restore_partitioned_mesh_state(storage, evidence)


@checked
def validate_partitioned_mesh_evidence(
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
) -> None:
    """Replay mathematical and native source premises on the current hardware."""
    limits = evidence.preparation.policy.limits
    allowance = bisection_host.UniformAllowance.from_limits(limits)
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        with (
            bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
            budget.ticket.native_target(),
        ):
            budget.retain_basis((storage, evidence))
            _run_validate_partitioned_mesh_evidence(
                storage, evidence, scope=scope, budget=budget
            )


def _run_validate_partitioned_mesh_evidence(
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    if evidence.migration_record is not None:
        # Admit device handles only for explicit replay, never for persisted records.
        parts = evidence.migration_record.source.placement.rebind()
        sharding = jax.sharding.NamedSharding(parts.mesh, jax.sharding.PartitionSpec())

        def place(value: Any) -> Any:
            return (
                jax.device_put(value, sharding) if isinstance(value, jax.Array) else value
            )

        storage, evidence = jax.tree.map(place, (storage, evidence))
        budget.retain_basis((storage, evidence))
    layout, states, _ = restore_partitioned_mesh_state(storage, evidence)
    if layout.signature_id != evidence.layout.signature_id:
        raise ValueError(
            "device_adaptation.run_validate_partitioned_mesh_evidence.layout must match its numerical declaration; got mismatched restored collective layout"
        )
    _validate_restored_initial_source(evidence, scope=scope, budget=budget)
    renewed = _dependency_phydrax_meshing__result.CollectiveMeshEvidence(
        evidence, evidence.compiled_states, scope=scope, budget=budget
    )
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
            "device_adaptation.run_validate_partitioned_mesh_evidence.renewed must match its independently replayed source mathematics; got mismatched restored collective certification"
        )


@jax.jit
def _canonical_retained_coordinate_values(
    identifiers: jax.Array,
    count: jax.Array,
    vertex_ids: jax.Array,
    active: jax.Array,
    source_coordinates: jax.Array,
    prior_coordinates: jax.Array,
) -> jax.Array:
    table = jnp.where(
        jnp.arange(identifiers.shape[0]) < count, identifiers, jnp.iinfo(jnp.int64).max
    )
    rows = jnp.searchsorted(table, vertex_ids)
    safe = jnp.minimum(rows, table.shape[0] - 1)
    found = (rows < table.shape[0]) & (table[safe] == vertex_ids)
    return jnp.where(
        (active & found)[..., None], source_coordinates[safe], prior_coordinates
    )


def _canonical_retained_coordinates(
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    arrays = dict(storage.logical_arrays)
    count = storage.global_entity_counts[0]
    coordinates = _canonical_retained_coordinate_values(
        arrays["vertex_global_ids"],
        jnp.asarray(count, dtype=jnp.int64),
        state.mesh.vertex_ids,
        state.mesh.vertex_active,
        arrays["coordinates"],
        state.mesh.coordinates,
    )
    return eqx.tree_at(lambda value: value.mesh.coordinates, state, coordinates)


def _require_migration_source_roots(
    migration: _owner_phydrax_meshing__distribution_migration.PartitionedEpochMigration,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Bind the premigration initial roots to their retained scientific source."""
    before = migration.source_partitioned
    source = before.prepared.adaptation.source
    if source.mesh.storage is None:
        # The retained routing banks are device leaves; validation reads them on host.
        entities, slot_origins = jax.device_get(
            (before.source_entities, before.slot_origins)
        )
        _validate_initial_partition_source(
            before.prepared,
            before.parts.part_count,
            before.states,
            before.source_exterior,
            entities,
            slot_origins,
            layout=before.layout,
            scope=scope,
            budget=budget,
        )
    else:
        prior = _dependency_phydrax_meshing__result.CollectiveMeshEvidence(
            before, migration.saved_states, scope=scope, budget=budget
        )
        _validate_restored_initial_source(prior, scope=scope, budget=budget)


def _validate_restored_initial_source(
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Bind the saved producer input to authored roots or an prior epoch."""
    source = evidence.source
    initial = evidence.initial_states
    prepared = evidence.preparation
    if prepared.source.result_id != source.result_id:
        raise ValueError(
            "device_adaptation.validate_restored_initial_source.source must match its accepted source; got mismatched restored preparation"
        )
    if evidence.migration_record is not None:
        migration = evidence.migration_record.replay(scope=scope, budget=budget)
        _require_migration_source_roots(migration, scope=scope, budget=budget)
        return
    if "placement/requested_target_owners" in dict(evidence.logical_arrays):
        receipt = _restart_receipt_from_evidence(evidence)
        _owner_phydrax_meshing__restart_distribution.validate_simplex_restart_repack(
            receipt, receipt.requested_target_owners
        )
        return

    if source.mesh.storage is None:
        simplex = _run_prepared_simplex(prepared, scope=scope, budget=budget)
        if tuple(block.block_id for block in evidence.raw_source_blocks) != tuple(
            block.block_id for block in source.mesh.blocks
        ):
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.tuple must match their whole-source declaration; got mismatched restored raw banks"
            )
        origins = jnp.where(initial.mesh.cell_active, initial.mesh.cell_ids, -1)
        _validate_initial_partition_source(
            simplex,
            initial.mesh.cell_ids.shape[0],
            initial,
            evidence.source_exterior,
            source_entity_tables(prepared),
            origins,
            layout=evidence.layout,
            scope=scope,
            budget=budget,
        )
        return
    if isinstance(
        source.collective_evidence,
        _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    ):
        _validate_initial_source_roots(
            source.collective_evidence, initial, evidence.source_exterior
        )
        if tuple(block.block_id for block in evidence.raw_source_blocks) != tuple(
            block.block_id for block in source.mesh.blocks
        ):
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.tuple must match their native source declaration; got mismatched restored initial raw banks"
            )
        return
    old_layout, retained, _ = retained_partition_states(source)
    compatible = _run_prepare_restored_roots(
        prepared, old_layout, retained, scope=scope, budget=budget
    )
    if compatible is not None:
        if compatible.layout.signature_id != evidence.layout.signature_id:
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.compatible must match its bounded layout; got mismatched restored compatible root preparation"
            )
        expected = compatible.states
        if logical_array_value_collection_digest(
            {"exterior": compatible.source_exterior}
        ) != logical_array_value_collection_digest(
            {"exterior": evidence.source_exterior}
        ):
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.logical_array_value_collection_digest must preserve their physical-boundary law; got changed their physical-boundary law"
            )
        declarations = compatible.prepared.anchor.source_blocks
        if declarations is None or tuple(
            block.block_id for block in declarations
        ) != tuple(block.block_id for block in evidence.raw_source_blocks):
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.declarations must preserve its numeric source bank declaration; got changed its numeric source bank declaration"
            )
        owners = dict(evidence.logical_arrays).get("placement/initial_solver_cell_owners")
        if owners is None or logical_array_value_collection_digest(
            {"owners": owners}
        ) != logical_array_value_collection_digest(
            {"owners": compatible.solver_cell_owners}
        ):
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.owners must preserve minimum source-leaf solver ownership; got changed minimum source-leaf solver ownership"
            )
    else:
        if tuple(block.block_id for block in evidence.raw_source_blocks) != tuple(
            block.block_id for block in _raw_source_blocks(source)
        ):
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.tuple must match their retained numeric declaration; got mismatched restored raw banks"
            )
        expected = _resized_epoch(retained, evidence.layout)
        protected, codes, capacity = _retained_protection(prepared, expected)
        if capacity != evidence.layout.protected_edge_capacity:
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.capacity must match the declared source layout; got an incompatible value"
            )
        expected = eqx.tree_at(
            lambda value: (value.vertex_protected, value.protected_codes),
            expected,
            (protected, codes),
        )
        source_storage = source.mesh.storage
        if source_storage is None:
            raise ValueError(
                "device_adaptation.validate_restored_initial_source.source_storage must retain its accepted logical storage; got missing its accepted logical storage"
            )
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
            "device_adaptation.validate_restored_initial_source.logical_array_value_collection_digest must match its immutable accepted predecessor; got mismatched restored initial forest"
        )


def _validate_initial_source_roots(
    evidence: _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    initial: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    exterior: jax.Array,
    /,
) -> None:
    """Consume native coordinates, cells, owners and physical facets."""
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
    facet_rows = key_positions(facet_keys, facets.reshape((-1, width - 1))).reshape(
        mesh.cells.shape
    )
    physical = arrays["initial/edge_physical"][jnp.maximum(facet_rows, 0)]
    valid &= jnp.all(
        ~mesh.cell_active[..., None] | ((facet_rows >= 0) & (exterior == physical))
    )
    if not bool(jax.device_get(valid)):
        raise ValueError(
            "device_adaptation.validate_initial_source_roots.valid must match the independent native source certification; got mismatched initial raw roots"
        )


def _partitioned_solver_owners(
    prepared: PreparedAdaptiveSimplex,
    parts: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    solver_cell_owners: jax.Array | None,
    /,
) -> jax.Array:
    """Validate solver routing with the existing single device reduction."""
    if solver_cell_owners is None:
        solver_cell_owners = source_solver_cell_owners(prepared, states)
    if (
        solver_cell_owners.shape != states.mesh.cell_ids.shape
        or solver_cell_owners.dtype != np.int32
    ):
        raise ValueError(
            "device_adaptation.init__.solver_cell_owners must provide an int32 raw-forest cell axis; got an absent or incompatible value"
        )
    # States may be global arrays spanning other processes; reduce on device.
    if not bool(
        jax.device_get(
            jnp.all(
                ~states.mesh.cell_active
                | ((solver_cell_owners >= 0) & (solver_cell_owners < parts.part_count))
            )
        )
    ):
        raise ValueError(
            "device_adaptation.init__.solver_cell_owners must provide its solver owner; got missing its solver owner"
        )
    return solver_cell_owners


def _validate_partitioned_source(
    prepared: PreparedAdaptiveSimplex,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    parts: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    source_exterior: jax.Array,
    entities: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    slot_origins: np.ndarray | jax.Array,
    supported: bool,
    accepted: bool,
    migration: _owner_phydrax_meshing__distribution_migration.PartitionedEpochMigration
    | None,
    rebalance_proposal: _owner_phydrax_meshing__distribution.PartitionedRebalanceProposal
    | None,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Authenticate initial or migrated roots before identity preparation."""
    if supported and not accepted and migration is None:
        failure: _leaf_phydrax_meshing__failures.MeshingFailure | None = None
        try:
            with budget.measurement():
                _validate_initial_partition_source(
                    prepared,
                    parts.part_count,
                    states,
                    source_exterior,
                    entities,
                    slot_origins,
                    layout=layout,
                    scope=scope,
                    budget=budget,
                )
        except (ValueError, IndexError) as error:
            failure = _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
                str(error),
                stage="distributed-source-preparation",
            )
        collective_failure_owner.collective_failure_barrier(
            failure,
            message="An owning process rejected the exact topology, lineage, or geometry resolution of the distributed epoch; no part was published.",
            stage="distributed-device-commit",
        )
    if migration is not None:
        _require_migration_source_roots(migration, scope=scope, budget=budget)
        _owner_phydrax_meshing__distribution_migration.validate_migrated_epoch(
            migration,
            prepared,
            migration.states,
            states,
            source_exterior,
            np.asarray(slot_origins, dtype=np.int64),
        )
        if rebalance_proposal is None:
            raise ValueError(
                "partitioned_migration.proposal must bind the requested ownership; got absent evidence"
            )


@final
class PartitionedAdaptiveSimplex(StrictModule, NonTrainableState):
    """Part-sharded view of one prepared epoch: each part refines its owned cells.

    Ownership is the policy's `MeshDistribution` (space-filling-curve, graph,
    or provider ownership of the source cells); cells created by uniform
    subdivision follow their source cell. ``layout`` is the per-part capacity
    bucket, ``parts`` the device mesh, and ``states`` the stacked local states.
    Publication automatically rebalances through whole-state migration when
    the inherited ownership exceeds the partition policy's balance bound.
    """

    prepared: PreparedAdaptiveSimplex
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout
    parts: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    source_exterior: jax.Array
    source_entities: tuple[tuple[jax.Array, jax.Array, jax.Array], ...]
    slot_origins: jax.Array
    partitioned_id: str = eqx.field(static=True)
    source_witness: (
        _dependency_phydrax_meshing__distribution.AffineBisectionSourceWitness | None
    )
    source_cache_id: str = eqx.field(static=True)
    initial_state_id: str = eqx.field(static=True)
    solver_cell_owners: jax.Array
    placement_arrays: tuple[tuple[str, jax.Array], ...]
    solver_neighborhood: (
        _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset | None
    )
    solver_forest: (
        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState | None
    )
    restart_repack: (
        _owner_phydrax_meshing__restart_distribution.SimplexRestartRepack | None
    )
    restart_proof: (
        _owner_phydrax_meshing__restart_distribution.SimplexRestartRepackProof | None
    )
    migration: (
        _owner_phydrax_meshing__distribution_migration.PartitionedEpochMigration | None
    )
    rebalance_proposal: (
        _owner_phydrax_meshing__distribution.PartitionedRebalanceProposal | None
    )
    execution_evidence: NativeExecutionRecord | None

    # Result reuse preserves object identity; fresh routing and root validation precede identity publication.
    @checked
    def __init__(
        self,
        prepared: PreparedAdaptiveSimplex,
        layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
        parts: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts,
        states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
        source_exterior: jax.Array,
        slot_origins: np.ndarray | jax.Array,
        /,
        *,
        solver_cell_owners: jax.Array | None = None,
        placement_arrays: tuple[tuple[str, jax.Array], ...] = (),
        solver_neighborhood: _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset
        | None = None,
        solver_forest: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
        | None = None,
        restart_repack: _owner_phydrax_meshing__restart_distribution.SimplexRestartRepack
        | None = None,
        restart_proof: _owner_phydrax_meshing__restart_distribution.SimplexRestartRepackProof
        | None = None,
        migration: _owner_phydrax_meshing__distribution_migration.PartitionedEpochMigration
        | None = None,
        rebalance_proposal: _owner_phydrax_meshing__distribution.PartitionedRebalanceProposal
        | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
        _validated_source: PartitionedAdaptiveSimplex | None = None,
        scope: NativeExecutionScope,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        _require_preparation_receipt(prepared.adaptation, execution_evidence)
        if _validated_source is not None:
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
                migration,
                rebalance_proposal,
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
                _validated_source.migration,
                _validated_source.rebalance_proposal,
            )
            if any(
                left is not right for left, right in zip(supplied, retained, strict=True)
            ):
                raise ValueError(
                    "device_adaptation.init__.any must not alter any validated scientific input; got a prohibited change"
                )
            if isinstance(slot_origins, np.ndarray) and slot_origins.flags.writeable:
                raise ValueError(
                    "device_adaptation.init__.slot_origins must provide immutable source-origin ownership; got an absent or incompatible value"
                )
            self.prepared = prepared
            self.layout = layout
            self.parts = parts
            self.states = states
            self.source_exterior = source_exterior
            self.slot_origins = _validated_source.slot_origins
            self.solver_cell_owners = _validated_source.solver_cell_owners
            self.placement_arrays = placement_arrays
            self.solver_neighborhood = solver_neighborhood
            self.solver_forest = solver_forest
            self.restart_repack = restart_repack
            self.restart_proof = restart_proof
            self.migration = migration
            self.rebalance_proposal = rebalance_proposal
            self.execution_evidence = execution_evidence
            self.source_entities = _validated_source.source_entities
            self.source_witness = _validated_source.source_witness
            self.source_cache_id = _validated_source.source_cache_id
            self.initial_state_id = _validated_source.initial_state_id
            self.partitioned_id = _validated_source.partitioned_id
            return
        entities = source_entity_tables(prepared.adaptation)
        solver_cell_owners = _partitioned_solver_owners(
            prepared, parts, states, solver_cell_owners
        )
        accepted = prepared.adaptation.source.mesh.storage is not None
        supported = accepted or (
            prepared.anchor.start.uniform_refinement is not None
            or not prepared.anchor.start.front.growth.levels.size
        )
        _validate_partitioned_source(
            prepared,
            layout,
            parts,
            states,
            source_exterior,
            entities,
            slot_origins,
            supported,
            accepted,
            migration,
            rebalance_proposal,
            scope=scope,
            budget=budget,
        )
        cache_id = source_cache_digest(entities)
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
        # Accepted origins are a global device bank. Preserve its placement:
        # converting it to NumPy would fetch non-addressable owners on this rank.
        # Only the host-prepared NumPy bank crosses the host/device boundary.
        origins = (
            jnp.asarray(slot_origins, dtype=jnp.int64)
            if isinstance(slot_origins, jax.Array)
            else np.asarray(slot_origins, dtype=np.int64)
        )
        self.source_entities = jax.tree_util.tree_map(
            lambda value: (
                value if isinstance(value, jax.Array) else jax.device_put(value)
            ),
            entities,
        )
        self.slot_origins = (
            origins if isinstance(origins, jax.Array) else jax.device_put(origins)
        )
        self.source_witness = (
            migration.source_partitioned.source_witness
            if migration is not None
            else _dependency_phydrax_meshing__distribution.AffineBisectionSourceWitness(
                prepared.adaptation.source,
                uniform_refinement=prepared.anchor.start.uniform_refinement,
                scope=scope,
                budget=budget,
            )
            if supported
            else None
        )
        if prepared.adaptation.source.associations and isinstance(
            prepared.adaptation.policy.association_transfer,
            _dependency_phydrax_meshing__association.PlcAssociationTransfer,
        ):
            scientific = _scientific_preparation_source(prepared.adaptation)
            if (
                not isinstance(
                    scientific, _dependency_phydrax_meshing__result.CellMeshingResult
                )
                or scientific.certification is None
                or scientific.certification.coverage is None
                or scientific.certification.embedding is None
            ):
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SPECIFICATION,
                    "partitioned_source.plc_transfer must carry embedding and domain coverage; got absent certification",
                    stage="distributed-source-preparation",
                )
        self.source_cache_id = cache_id
        self.initial_state_id = state_id
        self.solver_cell_owners = solver_cell_owners
        self.placement_arrays = placement_arrays
        self.solver_neighborhood = solver_neighborhood
        self.solver_forest = solver_forest
        self.restart_repack = restart_repack
        self.restart_proof = restart_proof
        self.migration = migration
        self.rebalance_proposal = rebalance_proposal
        self.partitioned_id = canonical_fingerprint(
            {
                "kind": "partitioned-adaptive-simplex",
                "prepared": prepared.prepared_id,
                "layout": layout.signature_id,
                # ty: ignore[unresolved-attribute]
                "distribution": prepared.adaptation.policy.distribution.distribution_id,
                "rebalance": None
                if rebalance_proposal is None
                else rebalance_proposal.proposal_id,
            }
        )

    def cell_marks(self, cell_ids: np.ndarray | jax.Array, /) -> jax.Array:
        """Per-part slot masks of the prepared cells descending from source cells."""

        identifiers = jnp.asarray(cell_ids, dtype=jnp.int64)
        origins = jnp.asarray(self.slot_origins, dtype=jnp.int64)
        return jnp.isin(origins, identifiers) & (origins >= 0)


def _adaptive_execution_parts(
    chosen: tuple[jax.Device, ...],
    *,
    neighbor_pairs: tuple[tuple[int, int], ...],
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts:
    """Reuse a compatible declared root axis for internally placed adaptive owners."""
    mesh = ExecutionRuntime.current().root_group.mesh
    if mesh.devices.ndim == 1 and tuple(mesh.devices.flat) == chosen:
        return _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts(
            chosen, axis_name=mesh.axis_names[0], neighbor_pairs=neighbor_pairs
        )
    return _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts(
        chosen, neighbor_pairs=neighbor_pairs
    )


@jax.jit
def _retained_exterior_values(
    ids: jax.Array,
    active: jax.Array,
    packet_ids: jax.Array,
    packet_valid: jax.Array,
    flags: jax.Array,
) -> jax.Array:
    def exterior(
        local_ids: jax.Array,
        local_active: jax.Array,
        local_packet_ids: jax.Array,
        local_packet_valid: jax.Array,
        local_flags: jax.Array,
    ) -> jax.Array:
        table = jnp.where(local_packet_valid, local_packet_ids, jnp.iinfo(jnp.int64).max)
        positions = jnp.minimum(jnp.searchsorted(table, local_ids), table.size - 1)
        return local_active[:, None] & local_flags[positions]

    return jax.vmap(exterior)(ids, active, packet_ids, packet_valid, flags)


def _resized_epoch(
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    /,
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    """Resize only padding while retaining every issued scientific identity."""
    return _resized_epoch_values(
        states,
        vertex_capacity=layout.vertex_capacity,
        cell_capacity=layout.cell_capacity,
        protected_edge_capacity=layout.protected_edge_capacity,
        output_shardings=tuple(value.sharding for value in jax.tree.leaves(states)),
    )


@eqx.filter_jit
def _resized_epoch_values(
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    *,
    vertex_capacity: int,
    cell_capacity: int,
    protected_edge_capacity: int,
    output_shardings: tuple[jax.sharding.Sharding, ...],
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    """Resize dynamic epoch leaves per capacity bucket and input placement."""

    def resize(
        part: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    ) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
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

        v, c = vertex_capacity, cell_capacity
        mesh = _dependency_phydrax_discretization__adaptive_simplex.MaskedSimplexMesh(
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
        return _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState(
            mesh,
            vertex_half_facets=fit(part.vertex_half_facets, v, -1),
            vertex_parents=fit(part.vertex_parents, v, -1),
            vertex_levels=fit(part.vertex_levels, v),
            vertex_removal=fit(part.vertex_removal, v, -1),
            vertex_protected=fit(part.vertex_protected, v),
            tuples=fit(part.tuples, c),
            tags=fit(part.tags, c),
            blocks=fit(part.blocks, c),
            generations=fit(part.generations, c),
            parents=fit(part.parents, c, -1),
            children=fit(part.children, c, -1),
            bisection_vertices=fit(part.bisection_vertices, c, -1),
            retired=fit(part.retired, c),
            cell_classes=fit(part.cell_classes, c),
            facet_classes=fit(part.facet_classes, c),
            protected_codes=fit(codes, protected_edge_capacity, sentinel),
            # Retain the vmap owner axis instead of broadcasting fresh reset leaves.
            refine_rejected=fit(part.refine_rejected & False, c, False),
            coarsen_marked=fit(part.coarsen_marked & False, c, False),
            cursors=part.cursors,
            clocks=part.clocks.at[2].set(0),
            counters=part.counters & jnp.int64(0),
        )

    resized = eqx.filter_vmap(resize)(states)
    leaves, structure = jax.tree.flatten(resized)
    return jax.tree.unflatten(
        structure,
        tuple(
            jax.lax.with_sharding_constraint(value, sharding)
            for value, sharding in zip(leaves, output_shardings, strict=True)
        ),
    )


@jax.jit
def _retained_protection_edges(
    vertex_ids: jax.Array, cells: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    parts, _, width = cells.shape
    columns = jnp.asarray(tuple(combinations(range(width), 2)), dtype=jnp.int32)
    slots = cells[:, :, columns].reshape((parts, -1, 2))
    ids = jnp.take_along_axis(vertex_ids, slots.reshape((parts, -1)), axis=1)
    return slots, ids.reshape(slots.shape), jnp.zeros(slots.shape[:-1], dtype=jnp.bool_)


@jax.jit
def _retained_scope_keys(
    identifiers: jax.Array,
    count: jax.Array,
    selected: jax.Array,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    table = jnp.where(
        jnp.arange(identifiers.shape[0]) < count, identifiers, jnp.iinfo(jnp.int64).max
    )
    order = jnp.argsort(table, stable=True)
    positions = key_positions(table[order, None], selected[:, None])
    missing = jnp.any((positions < 0) | (positions >= count))
    return keys[order[positions]], missing


@jax.jit
def _retained_vertex_scope_flags(
    vertex_ids: jax.Array, flags: jax.Array, keys: jax.Array
) -> jax.Array:
    return flags | jnp.any(
        vertex_ids[:, :, None, None] == keys[None, None, :, :], axis=(-2, -1)
    )


@jax.jit
def _retained_edge_scope_flags(
    edge_ids: jax.Array, flags: jax.Array, keys: jax.Array
) -> jax.Array:
    ends = jnp.any(edge_ids[:, :, :, None, None] == keys[None, None, None, :, :], axis=-1)
    return flags | jnp.any(jnp.all(ends, axis=2), axis=-1)


@jax.jit
def _retained_protection_values(
    slots: jax.Array,
    edge_flags: jax.Array,
    active: jax.Array,
    prior_codes: jax.Array,
    vertex_capacity: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    parts = slots.shape[0]
    columns = slots.shape[1] // active.shape[1]
    sentinel = jnp.iinfo(jnp.int64).max
    low, high = jnp.min(slots, axis=-1), jnp.max(slots, axis=-1)
    codes = jnp.where(
        edge_flags & jnp.repeat(active, columns, axis=1),
        low.astype(jnp.int64) * vertex_capacity + high,
        sentinel,
    )
    codes = jnp.sort(jnp.concatenate((prior_codes, codes), axis=1), axis=1)
    unique = (codes != sentinel) & jnp.concatenate(
        (jnp.ones((parts, 1), dtype=jnp.bool_), codes[:, 1:] != codes[:, :-1]), axis=1
    )
    order = jnp.argsort(~unique, axis=1, stable=True)
    codes = jnp.take_along_axis(codes, order, axis=1)
    return codes, jnp.max(jnp.sum(unique, axis=1, dtype=jnp.int64))


@jax.jit(static_argnames=("capacity",))
def _retained_protection_bucket(codes: jax.Array, *, capacity: int) -> jax.Array:
    codes = codes[:, :capacity]
    return jnp.pad(
        codes,
        ((0, 0), (0, max(capacity - codes.shape[1], 0))),
        constant_values=jnp.iinfo(jnp.int64).max,
    )


def _retained_protection(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> tuple[jax.Array, jax.Array, int]:
    """Apply authored scopes while preserving retained constraints.

    Numerical entry points receive only live array leaves and dynamic counts;
    the final shape uses the same declared protected-edge capacity bucket.
    Cold source/epoch/entity admission remains at this owner before use.
    """
    evidence = prepared.source.collective_evidence
    storage = prepared.source.mesh.storage
    if storage is None:
        raise ValueError(
            "device_adaptation.retained_protection.storage must provide owner-local numerical storage; got an absent or incompatible value"
        )
    if evidence is None:
        raise ValueError(
            "device_adaptation.retained_protection.evidence must provide collective source routing; got an absent or incompatible value"
        )
    vertex_flags = states.vertex_protected
    width = states.mesh.cells.shape[-1]
    capacity = states.mesh.vertex_ids.shape[1]
    edge_slots, edge_ids, edge_flags = _retained_protection_edges(
        states.mesh.vertex_ids, states.mesh.cells
    )
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
                "device_adaptation.retained_protection.scope must bind the accepted source epoch; got an incompatible value"
            )
        keys, missing = _retained_scope_keys(
            evidence.entity_ids[degree],
            jnp.asarray(evidence.global_entity_counts[degree], dtype=jnp.int64),
            scope.entity_ids,
            arrays["cell_vertices"]
            if degree == width - 1
            else evidence.entity_keys[degree],
        )
        if bool(jax.device_get(missing)):
            raise ValueError(
                "device_adaptation.retained_protection.positions must select existing accepted source entities; got protected scope selects absent accepted logical source entities"
            )
        vertex_flags = _retained_vertex_scope_flags(
            states.mesh.vertex_ids, vertex_flags, keys
        )
        if degree:
            edge_flags = _retained_edge_scope_flags(edge_ids, edge_flags, keys)
    codes, count_value = _retained_protection_values(
        edge_slots,
        edge_flags,
        states.mesh.cell_active,
        states.protected_codes,
        jnp.asarray(capacity, dtype=jnp.int64),
    )
    count = int(jax.device_get(count_value))
    edges = _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
        max(count, 1), 1.0
    )
    codes = _retained_protection_bucket(codes, capacity=edges)
    return vertex_flags, codes, edges


@checked
def prepare_partitioned_mesh_adaptation(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation, /
) -> PartitionedAdaptiveSimplex:
    limits = prepared.policy.limits
    allowance = bisection_host.UniformAllowance.from_limits(limits)
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        partitioned = _prepare_partitioned_mesh_adaptation(
            prepared, scope=scope, budget=budget
        )
    if scope.evidence is None:
        return partitioned
    return _bind_partitioned_preparation_receipt(
        partitioned,
        NativeExecutionRecord(scope.evidence, owner_id=prepared.prepared_id),
        scope=scope,
        budget=budget,
    )


def _bind_partitioned_preparation_receipt(
    source: PartitionedAdaptiveSimplex,
    record: NativeExecutionRecord,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
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
        migration=source.migration,
        rebalance_proposal=source.rebalance_proposal,
        execution_evidence=record,
        _validated_source=source,
        scope=scope,
        budget=budget,
    )


def _prepare_partitioned_mesh_adaptation(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex:
    """Prepare the distributed route from dense or accepted logical state."""
    if (
        prepared.policy.route
        is not _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.DEVICE_BISECTION
        or not isinstance(
            prepared.request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
        )
    ):
        raise ValueError(
            "device_adaptation.prepare_partitioned_mesh_adaptation.prepared must provide a marked DEVICE_BISECTION request; got an absent or incompatible value"
        )
    distribution = prepared.policy.distribution
    device = prepared.policy.device_policy
    if distribution is None or device is None:
        raise ValueError(
            "device_adaptation.prepare_partitioned_mesh_adaptation.distribution must provide explicit distribution and device policies; got an absent or incompatible value"
        )
    source = prepared.source
    allowance = _phase_allowance(prepared, None)
    if source.mesh.storage is None:
        with (
            bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
            budget.ticket.native_target(),
        ):
            simplex = _run_prepared_simplex(prepared, scope=scope, budget=budget)
        return partition_adaptive_simplex(simplex)

    if isinstance(
        source.collective_evidence,
        _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    ):
        with (
            bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
            budget.ticket.native_target(),
        ):
            return _prepare_initial_partition(prepared, scope=scope, budget=budget)
    old_layout, retained, _ = retained_partition_states(source)
    storage = source.mesh.storage
    if distribution.partition.part_count != storage.partition_count:
        raise ValueError(
            "device_adaptation.prepare_partitioned_mesh_adaptation.storage must provide completed scientific-ID forest migration before adaptation; got an absent or incompatible value"
        )
    compatible = _prepare_restored_roots(
        prepared, old_layout, retained, scope=scope, budget=budget
    )
    if compatible is not None:
        return compatible
    allocated = np.asarray(
        jax.device_get(jnp.max(retained.cursors[:, :2], axis=0)), dtype=np.int64
    )
    vertices, cells = device.capacities(int(allocated[0]), int(allocated[1]))
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
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
        layout = (
            _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
                layout.dimension,
                layout.ambient_dimension,
                vertex_capacity=layout.vertex_capacity,
                cell_capacity=layout.cell_capacity,
                protected_edge_capacity=edge_capacity,
                maximum_closure_iterations=layout.maximum_closure_iterations,
                maximum_coarsening_passes=layout.maximum_coarsening_passes,
            )
        )
    states = eqx.tree_at(
        lambda state: (state.vertex_protected, state.protected_codes),
        states,
        (vertex_flags, protected_codes),
    )
    arrays = dict(storage.logical_arrays)
    states = _canonical_retained_coordinates(storage, states)

    source_exterior = _retained_exterior_values(
        states.mesh.cell_ids,
        states.mesh.cell_active,
        arrays["closure/cell_ids"],
        arrays["closure/cell_valid"],
        arrays["closure/cell_exterior"],
    )
    part_count = storage.partition_count
    chosen = tuple(jax.devices()[:part_count])
    if len(chosen) != part_count:
        raise ValueError(
            f"device_adaptation.retained_ownership.devices must provide {part_count} devices; got insufficient device availability"
        )
    parts = _adaptive_execution_parts(
        chosen,
        neighbor_pairs=tuple(
            (left, right)
            for left in range(part_count)
            for right in range(part_count)
            if left != right
        ),
    )
    with (
        bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
        budget.ticket.native_target(),
    ):
        simplex = _source_simplex_on_states(
            prepared, layout, states, scope=scope, budget=budget
        )
        origins = jnp.where(states.mesh.cell_active, states.mesh.cell_ids, -1)
        return PartitionedAdaptiveSimplex(
            simplex,
            layout,
            parts,
            states,
            source_exterior,
            origins,
            scope=scope,
            budget=budget,
        )


def _run_prepare_restored_roots(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex | None:
    budget.retain_basis((prepared, retained))
    inventory: _OriginalRootInventory | _MixedRootInventory | None = (
        _admit_original_root_inventory(prepared, retained)
    )
    if inventory is None:
        inventory = _admit_mixed_root_inventory(prepared, retained)
    match inventory:
        case _OriginalRootInventory():
            result = _prepare_original_root_forest(
                prepared, old_layout, retained, inventory, scope=scope, budget=budget
            )
        case _MixedRootInventory():
            result = _prepare_mixed_root_cohort(
                prepared, old_layout, retained, inventory, scope=scope, budget=budget
            )
        case None:
            return None
        case _ as invalid:
            assert_never(invalid)
    budget.retain_basis((result,))
    return result


def _prepare_restored_roots(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex | None:
    limits = prepared.policy.limits
    allowance = bisection_host.UniformAllowance.from_limits(limits)
    with (
        bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
        budget.ticket.native_target(),
    ):
        return _run_prepare_restored_roots(
            prepared, old_layout, retained, scope=scope, budget=budget
        )


@final
class _RootCellBank(StrictModule):
    """Cell topology and routing remapped to the merged ID-ordered banks."""

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
    origins: np.ndarray
    solver: np.ndarray


def _remapped_root_cells(
    part: _HostEpoch,
    slots: np.ndarray,
    incoming: bool,
    vertex_ids: np.ndarray,
    cell_ids: np.ndarray,
    remap: dict[int, int],
    old_solver: np.ndarray,
    root_solver: dict[int, tuple[int, int]],
    /,
) -> _RootCellBank:
    identifiers = np.asarray(
        [
            remap.get(int(value), int(value)) if incoming else int(value)
            for value in part.vertex_ids
        ],
        dtype=np.int64,
    )
    vertex_rows = np.searchsorted(vertex_ids, identifiers)
    parents = part.parents[slots]
    parents = np.where(
        parents >= 0,
        2 * np.searchsorted(cell_ids, part.cell_ids[np.maximum(parents, 0) // 2])
        + parents % 2,
        parents,
    )
    children = part.children[slots]
    children = np.where(
        children >= 0,
        np.searchsorted(cell_ids, part.cell_ids[np.maximum(children, 0)]),
        -1,
    )
    bisected = part.bisection_vertices[slots]
    bisected = np.where(bisected >= 0, vertex_rows[np.maximum(bisected, 0)], bisected)
    if incoming:
        routing = np.asarray(
            [root_solver[int(value)] for value in part.cell_ids[slots]], dtype=np.int64
        ).reshape((-1, 2))
        origins, solver = routing[:, 0], routing[:, 1].astype(np.int32)
    else:
        origins, solver = part.cell_ids[slots], old_solver[slots]
    return _RootCellBank(
        part.cell_active[slots],
        vertex_rows[part.cells[slots]],
        vertex_rows[part.tuples[slots]],
        part.tags[slots],
        part.blocks[slots],
        part.generations[slots],
        parents,
        children,
        bisected,
        part.retired[slots],
        part.refine_rejected[slots],
        part.coarsen_marked[slots],
        part.cell_classes[slots],
        part.facet_classes[slots],
        np.where(part.cell_active[slots], origins, -1),
        solver,
    )


def _merged_root_cells(
    components: tuple[tuple[_HostEpoch, np.ndarray, bool], ...],
    vertex_ids: np.ndarray,
    cell_ids: np.ndarray,
    order: np.ndarray,
    remap: dict[int, int],
    old_solver: np.ndarray,
    root_solver: dict[int, tuple[int, int]],
    /,
) -> _RootCellBank:
    banks = tuple(
        _remapped_root_cells(
            part, slots, incoming, vertex_ids, cell_ids, remap, old_solver, root_solver
        )
        for part, slots, incoming in components
    )
    return _RootCellBank(
        np.concatenate([bank.cell_active for bank in banks])[order],
        np.concatenate([bank.cells for bank in banks])[order],
        np.concatenate([bank.tuples for bank in banks])[order],
        np.concatenate([bank.tags for bank in banks])[order],
        np.concatenate([bank.blocks for bank in banks])[order],
        np.concatenate([bank.generations for bank in banks])[order],
        np.concatenate([bank.parents for bank in banks])[order],
        np.concatenate([bank.children for bank in banks])[order],
        np.concatenate([bank.bisection_vertices for bank in banks])[order],
        np.concatenate([bank.retired for bank in banks])[order],
        np.concatenate([bank.refine_rejected for bank in banks])[order],
        np.concatenate([bank.coarsen_marked for bank in banks])[order],
        np.concatenate([bank.cell_classes for bank in banks])[order],
        np.concatenate([bank.facet_classes for bank in banks])[order],
        np.concatenate([bank.origins for bank in banks])[order],
        np.concatenate([bank.solver for bank in banks])[order],
    )


def _merge_prepared_root_part(
    old: _HostEpoch,
    seed: _HostEpoch,
    seed_slots: np.ndarray,
    restored: np.ndarray,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    remap: dict[int, int],
    protected: np.ndarray,
    codes: np.ndarray,
    old_capacity: int,
    seed_edges: np.ndarray,
    old_solver: np.ndarray,
    root_solver: dict[int, tuple[int, int]],
    /,
) -> tuple[_HostEpoch, np.ndarray, np.ndarray]:
    """Retain untouched families and replace only authenticated roots."""
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
            "device_adaptation.merge_prepared_root_part.cell_ids must use cell IDs distinct from retained families; got mixed root preparation collided with a retained scientific cell id"
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
    bank = _merged_root_cells(
        components, vertex_ids, cell_ids, order, remap, old_solver, root_solver
    )
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
        bank.cell_active,
        bank.cells,
        bank.tuples,
        bank.tags,
        bank.blocks,
        bank.generations,
        bank.parents,
        bank.children,
        bank.bisection_vertices,
        bank.retired,
        bank.refine_rejected,
        bank.coarsen_marked,
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
        bank.cell_classes,
        bank.facet_classes,
        np.asarray([vertex_flags[int(key)] for key in vertex_ids]),
        protected_codes,
    )
    return base, bank.origins.astype(np.int64), bank.solver.astype(np.int32)


@final
class _MixedRootRequest(StrictModule):
    """Authenticated root columns, ownership, protection, and source request."""

    source: bisection_host.Source
    request: bisection_host.Request
    flags: jax.Array
    codes: jax.Array
    solver: np.ndarray
    hosts: tuple[_HostEpoch, ...]
    restored_facets: np.ndarray
    classes: np.ndarray
    protected_vertices: list[int]
    root_owner: dict[int, int]
    root_solver: dict[int, int]
    width: int = eqx.field(static=True)


def _restored_root_request(
    source: bisection_host.Source,
    protected_edges: list[list[int]],
    protected_vertices: list[int],
    cell_classes: np.ndarray,
    facet_classes: np.ndarray,
    maximum_closure_iterations: int,
    /,
    *,
    source_cell_classes: np.ndarray | None = None,
) -> bisection_host.Request:
    empty = np.zeros(0, dtype=np.int64)
    return bisection_host.prepared_request(
        source,
        empty,
        empty,
        np.unique(np.asarray(protected_edges, dtype=np.int64).reshape((-1, 2)), axis=0),
        np.unique(np.asarray(protected_vertices, dtype=np.int64)),
        cell_classes,
        facet_classes,
        maximum_closure_iterations,
        source_cell_classes=source_cell_classes,
    )


def _authenticate_mixed_root_request(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    source: bisection_host.Source,
    restored: np.ndarray,
    positions: np.ndarray,
    bank: dict[str, np.ndarray],
    parent_rows: np.ndarray,
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedRootRequest:
    """Authenticated root columns, ownership, protection, and source request."""
    flags, codes, _ = _retained_protection(prepared, retained)
    solver = np.asarray(jax.device_get(source_solver_cell_owners(prepared, retained)))
    leaves = jax.tree_util.tree_leaves(retained)
    bisection_host.uniform_count(
        sum(value.size for value in leaves if isinstance(value, jax.Array)),
        sum(value.nbytes for value in leaves if isinstance(value, jax.Array)),
        budget=budget,
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
                    "device_adaptation.authenticate_mixed_root_request.key must have conflicting source ownership; got incompatible axes"
                )
            if (
                host.generations[slot] != bank["parent_levels"][parent_rows[row]]
                or host.cell_classes[slot] != bank["parent_classes"][parent_rows[row]]
            ):
                raise ValueError(
                    "device_adaptation.authenticate_mixed_root_request.slot must preserve their level or class; got changed their level or class"
                )
            if not np.array_equal(
                host.vertex_ids[host.cells[slot]], bank["parent_rows"][parent_rows[row]]
            ):
                raise ValueError(
                    "device_adaptation.authenticate_mixed_root_request.host must match SCI provenance; got mismatched restored raw root incidence"
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
        raise ValueError(
            "device_adaptation.authenticate_mixed_root_request.set must provide a uniquely retained root; got missing a uniquely retained root"
        )
    width = source.dimension + 1
    facet_keys = entity_keys_owner.entity_keys(source.mesh, source.dimension - 1)
    facet_classes = np.full(facet_keys.shape[0], -1, dtype=np.int64)
    columns = np.asarray(
        tuple(combinations(range(width), source.dimension)), dtype=np.int64
    )
    rows = entity_keys_owner.key_rows(
        facet_keys,
        np.sort(bank["parent_rows"][:, columns], axis=2).reshape((-1, source.dimension)),
    )
    if np.any(rows < 0):
        raise ValueError(
            "device_adaptation.authenticate_mixed_root_request.rows must retain source facet SCI; got missing source facet SCI"
        )
    np.maximum.at(facet_classes, rows, bank["parent_facet_classes"].reshape(-1))
    classes = bank["parent_classes"][parent_rows]
    source = source._replace(
        cells=bisection_host.select(source.cells, positions)._replace(
            tuples=np.searchsorted(source.vertex_ids, restored_tuples),
            tags=restored_tags,
            generations=restored_levels,
        )
    )
    request = _restored_root_request(
        source,
        protected_edges,
        protected_vertices,
        classes,
        facet_classes,
        prepared.policy.maximum_closure_iterations,
        source_cell_classes=classes,
    )
    return _MixedRootRequest(
        source,
        request,
        flags,
        codes,
        solver,
        hosts,
        restored_facets,
        classes,
        protected_vertices,
        root_owner,
        root_solver,
        width,
    )


@final
class _MixedRootCohort(StrictModule):
    """Regenerated families reconciled with the retained forest and identity banks."""

    start: bisection_host.Start
    combined: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    merged: tuple[tuple[_HostEpoch, np.ndarray, np.ndarray], ...]
    declarations: list[CellBlock]


def _regenerate_mixed_root_cohort(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    restored: np.ndarray,
    scientific: _dependency_phydrax_meshing__result.CellMeshingResult,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    phase: _MixedRootRequest,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _MixedRootCohort:
    """Regenerated families reconciled with the retained forest and identity banks."""
    source = phase.source
    request = phase.request
    flags = phase.flags
    codes = phase.codes
    solver = phase.solver
    hosts = phase.hosts
    restored_facets = phase.restored_facets
    classes = phase.classes
    protected_vertices = phase.protected_vertices
    root_owner = phase.root_owner
    root_solver = phase.root_solver
    limits = prepared.policy.limits
    allowance = bisection_host.UniformAllowance.from_limits(limits)
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

    start = bisection_host.reprepare_admitted_uniform(
        source,
        request,
        lineage,
        allowance,
        issuers,
        tuple(retired),
        scope=scope,
        budget=budget,
    )
    if start.uniform_refinement is None:
        raise ValueError(
            "device_adaptation.regenerate_mixed_root_cohort.start must retain its source action law; got missing its source action law"
        )
    combined, remap = _owner_phydrax_meshing__bisection.reconcile_uniform_refinement(
        start.uniform_refinement, lineage, restored, scope=scope, budget=budget
    )
    seed = _HostEpoch.seed(
        source,
        classes,
        restored_facets,
        issuers,
        hosts[0],
        np.asarray(protected_vertices, dtype=np.int64),
    )
    root_seed = seed
    forest = _forest(start, source, request, scope=scope, budget=budget)
    vertices = _vertices(start, source, request)
    generated_count = forest["ids"].size
    generated = _HostEpoch.from_forest(
        forest, vertices, start, hosts[0].counters, template=hosts[0]
    )
    generated_solver = {
        int(cell): (int(root), root_solver[int(root)])
        for cell, root in zip(forest["ids"], forest["origins"], strict=True)
    }
    seed_edges = source.vertex_ids[
        np.stack(
            (
                request.protected_codes // bisection_host.SHIFT,
                request.protected_codes % bisection_host.SHIFT,
            ),
            axis=1,
        )
    ]
    seed, seed_origins, seed_solver_owners = _merge_prepared_root_part(
        root_seed,
        generated,
        np.arange(generated_count, dtype=np.int64),
        restored,
        lineage,
        remap,
        root_seed.vertex_protected,
        np.full((0,), np.iinfo(np.int64).max, dtype=np.int64),
        root_seed.vertex_ids.size,
        seed_edges,
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
            (
                request.protected_codes // bisection_host.SHIFT,
                request.protected_codes % bisection_host.SHIFT,
            ),
            axis=1,
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
    budget.retain_basis((seed, merged, combined))
    return _MixedRootCohort(start, combined, merged, declarations)


@final
class _RestoredRootPlacement(StrictModule):
    """Capacity-bounded owner placement with restored boundary and solver routing."""

    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout
    parts: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts
    sharding: jax.sharding.NamedSharding
    stacked: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    origins: np.ndarray
    owners: np.ndarray
    exterior: np.ndarray


def _place_mixed_root_cohort(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    source: bisection_host.Source,
    scientific: _dependency_phydrax_meshing__result.CellMeshingResult,
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    merged: tuple[tuple[_HostEpoch, np.ndarray, np.ndarray], ...],
    width: int,
    combined: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _RestoredRootPlacement:
    """Capacity-bounded owner placement with restored boundary and solver routing."""
    device = prepared.policy.device_policy
    if device is None:
        raise ValueError(
            "device_adaptation.place_mixed_root_cohort.device must provide its device capacities; got an absent or incompatible value"
        )
    vertex_capacity, cell_capacity = device.capacities(
        max(base.vertex_ids.size for base, _, _ in merged),
        max(base.cell_ids.size for base, _, _ in merged),
    )
    edge_capacity = (
        _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
            max(1, max(base.protected_codes.size for base, _, _ in merged)), 1.0
        )
    )
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
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
        raise ValueError(
            "device_adaptation.place_mixed_root_cohort.len must retain the declared device ownership; got mixed roots require their retained device ownership"
        )
    parts = _adaptive_execution_parts(
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
        rows = entity_keys_owner.key_rows(unique, keys)
        valid = rows >= 0
        flags = np.zeros(rows.shape, dtype=np.bool_)
        flags[valid] = occurrences[rows[valid]] == 1
        exterior[part, :size] = flags.reshape((size, width)) & base.cell_active[:, None]
    stacked = jax.tree_util.tree_map(
        lambda *values: jax.device_put(jnp.stack(values), sharding), *states
    )
    stacked = _physical_packed_publication_states(
        stacked, combined, scope=scope, budget=budget
    )
    return _RestoredRootPlacement(
        layout, parts, sharding, stacked, origins, owners, exterior
    )


@final
class _RestoredBisectionRequest(StrictModule):
    """Marked request and hierarchy narrowed for restored root preparation."""

    request: _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy | None


def _restored_bisection_request(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
) -> _RestoredBisectionRequest:
    match prepared.request:
        case _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation() as request:
            match request.hierarchy:
                case (
                    _dependency_phydrax_meshing__bisection.BisectionHierarchy() as hierarchy
                ):
                    return _RestoredBisectionRequest(request, hierarchy)
                case None:
                    return _RestoredBisectionRequest(request, None)
                case _:
                    raise TypeError(
                        "device_adaptation.restored_bisection_hierarchy.prepared must provide its bisection hierarchy; got an absent or incompatible value"
                    )
        case _:
            raise TypeError(
                "device_adaptation.restored_bisection_hierarchy.prepared must provide its marked request; got an absent or incompatible value"
            )


@final
class _MixedRootInventory(StrictModule):
    """Authenticated restored root inventory and scientific source banks."""

    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    scientific: _dependency_phydrax_meshing__result.CellMeshingResult
    source: bisection_host.Source
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage
    restored: np.ndarray
    positions: np.ndarray
    bank: dict[str, np.ndarray]
    parent_rows: np.ndarray


def _admit_mixed_root_inventory(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
) -> _MixedRootInventory | None:
    hierarchy = _restored_bisection_request(prepared).hierarchy
    evidence = prepared.source.collective_evidence
    lineage = (
        evidence.uniform_refinement
        if isinstance(
            evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
        )
        else None
    )
    if hierarchy is None or lineage is None:
        return None
    scientific = lineage.source
    _dependency_phydrax_meshing__result.require_original_meshing_source(prepared.source)
    source = bisection_host.prepared_source(scientific.mesh)
    storage = prepared.source.mesh.storage
    if storage is None:
        return None
    ids = np.asarray(jax.device_get(dict(storage.logical_arrays)["cell_global_ids"]))[
        : storage.global_entity_counts[-1]
    ]
    restored = np.intersect1d(ids, np.asarray(lineage.parent_ids))
    if restored.size == 0:
        return None
    positions = np.searchsorted(source.cells.ids, restored)
    if np.any(positions >= source.cells.ids.size) or not np.array_equal(
        source.cells.ids[positions], restored
    ):
        raise ValueError(
            "device_adaptation.run_prepare_restored_mixed_roots.positions must provide their scientific source cell provenance; got missing their scientific source cell provenance"
        )
    bank = lineage.host_arrays()
    parent_rows = np.searchsorted(bank["parent_ids"], restored)
    if not np.array_equal(
        source.vertex_ids[source.cells.rows[positions]], bank["parent_rows"][parent_rows]
    ):
        raise ValueError(
            "device_adaptation.run_prepare_restored_mixed_roots.parent_rows must preserve their source corner columns; got changed their source corner columns"
        )
    return _MixedRootInventory(
        hierarchy,
        lineage,
        scientific,
        source,
        storage,
        restored,
        positions,
        bank,
        parent_rows,
    )


def _prepare_mixed_root_cohort(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    inventory: _MixedRootInventory,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex:
    hierarchy = inventory.hierarchy
    lineage = inventory.lineage
    scientific = inventory.scientific
    source = inventory.source
    storage = inventory.storage
    restored = inventory.restored
    positions = inventory.positions
    bank = inventory.bank
    parent_rows = inventory.parent_rows
    phase = _authenticate_mixed_root_request(
        prepared=prepared,
        retained=retained,
        old_layout=old_layout,
        source=source,
        restored=restored,
        positions=positions,
        bank=bank,
        parent_rows=parent_rows,
        storage=storage,
        budget=budget,
    )
    source = phase.source
    width = phase.width
    phase = _regenerate_mixed_root_cohort(
        prepared=prepared,
        retained=retained,
        hierarchy=hierarchy,
        lineage=lineage,
        restored=restored,
        scientific=scientific,
        old_layout=old_layout,
        phase=phase,
        scope=scope,
        budget=budget,
    )
    start = phase.start
    combined = phase.combined
    merged = phase.merged
    declarations = phase.declarations
    phase = _place_mixed_root_cohort(
        prepared=prepared,
        source=source,
        scientific=scientific,
        storage=storage,
        merged=merged,
        width=width,
        combined=combined,
        scope=scope,
        budget=budget,
    )
    layout = phase.layout
    parts = phase.parts
    sharding = phase.sharding
    stacked = phase.stacked
    origins = phase.origins
    owners = phase.owners
    exterior = phase.exterior
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
        scope=scope,
        budget=budget,
    )


@final
class _OriginalRootRequest(StrictModule):
    """Authenticated full-inverse roots, protection, facets, and solver ownership."""

    source: bisection_host.Source
    request: bisection_host.Request
    host: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    root_solver_owners: np.ndarray
    columns: np.ndarray


def _authenticate_original_root_request(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    source: bisection_host.Source,
    root_ids: np.ndarray,
    active: np.ndarray,
) -> _OriginalRootRequest:
    """Authenticated full-inverse roots, protection, facets, and solver ownership."""
    flags, codes, _ = _retained_protection(prepared, retained)
    host = jax.device_get(retained)
    cell_classes = np.zeros(root_ids.size, dtype=np.int64)
    root_generations = np.empty(root_ids.size, dtype=np.int64)
    root_tuples = np.empty((root_ids.size, source.dimension + 1), dtype=np.int64)
    root_tags = np.empty(root_ids.size, dtype=np.int64)
    root_solver_owners = np.empty(root_ids.size, dtype=np.int32)
    retained_solver_owners = np.asarray(
        jax.device_get(source_solver_cell_owners(prepared, retained))
    )
    facet_keys = entity_keys_owner.entity_keys(source.mesh, source.dimension - 1)
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
                "device_adaptation.authenticate_original_root_request.positions must retain unique SCI root ownership; got missing unique SCI root ownership"
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
        rows = entity_keys_owner.key_rows(facet_keys, keys)
        values = np.asarray(host.facet_classes[part])[slots].reshape(-1)
        if np.any(rows < 0):
            raise ValueError(
                "device_adaptation.authenticate_original_root_request.rows must match SCI keys; got mismatched full-inverse root facets"
            )
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
        raise ValueError(
            "device_adaptation.authenticate_original_root_request.seen must provide an SCI root; got missing an SCI root"
        )
    source = source._replace(
        cells=source.cells._replace(
            tuples=np.searchsorted(source.vertex_ids, root_tuples),
            tags=root_tags,
            generations=root_generations,
        )
    )
    request = _restored_root_request(
        source,
        protected_edges,
        protected_vertices,
        cell_classes,
        np.maximum(facet_classes - 1, -1),
        prepared.policy.maximum_closure_iterations,
    )
    return _OriginalRootRequest(source, request, host, root_solver_owners, columns)


@final
class _OriginalRootPlacement(StrictModule):
    """Prepared binary forest placed under the retained root capacities and ownership."""

    start: bisection_host.Start
    forest: dict[str, np.ndarray]
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout
    parts: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts
    sharding: jax.sharding.NamedSharding
    stacked: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    origins: np.ndarray
    solver_owners: np.ndarray
    exterior: np.ndarray


def _place_original_root_forest(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    scientific: _dependency_phydrax_meshing__result.CellMeshingResult,
    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy,
    root_ids: np.ndarray,
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage,
    phase: _OriginalRootRequest,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _OriginalRootPlacement:
    """Prepared binary forest placed under the retained root capacities and ownership."""
    source = phase.source
    request = phase.request
    host = phase.host
    root_solver_owners = phase.root_solver_owners
    columns = phase.columns
    limits = prepared.policy.limits
    allowance = bisection_host.UniformAllowance.from_limits(limits)
    # A inverse has cut over to roots. Historical
    # uniform evidence remains in the accepted certification, but re-expanding those
    # children here would consume execution capacity for absent topology.
    start = bisection_host.prepared_start(
        source,
        request,
        prepared.policy.compatibility,
        hierarchy,
        allowance,
        scientific,
        future_refinement=True,
        scope=scope,
        budget=budget,
    )
    forest, vertices = (
        _forest(start, source, request, scope=scope, budget=budget),
        _vertices(start, source, request),
    )
    count, vertex_count = forest["ids"].size, vertices["ids"].size
    base = _HostEpoch.from_forest(
        forest, vertices, start, np.zeros_like(np.asarray(host.counters[0]))
    )
    tables = source_entity_tables(prepared)
    ids, owners = (np.asarray(jax.device_get(value)) for value in tables[-1][1:])
    order = np.argsort(ids, kind="stable")
    slot_owner = owners[order][np.searchsorted(ids[order], forest["origins"])]
    parts_count = storage.partition_count
    slots = tuple(np.flatnonzero(slot_owner == part) for part in range(parts_count))
    edges = np.stack(
        (
            request.protected_codes // bisection_host.SHIFT,
            request.protected_codes % bisection_host.SHIFT,
        ),
        axis=1,
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
            "device_adaptation.place_original_root_forest.device must provide its device capacities; got an absent or incompatible value"
        )
    vertex_capacity, cell_capacity = device.capacities(
        max(piece["vertices"].size for piece in pieces),
        max(selected.size for selected in slots),
    )
    edge_capacity = (
        _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_bucket(
            max(
                1, max((piece["protected_edges"].shape[0] for piece in pieces), default=0)
            ),
            1.0,
        )
    )
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
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
            "device_adaptation.place_original_root_forest.parts_count must provide its device ownership; got an absent or incompatible value"
        )
    parts = _adaptive_execution_parts(
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
        stacked = _physical_packed_publication_states(
            stacked, start.uniform_refinement, scope=scope, budget=budget
        )
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
    return _OriginalRootPlacement(
        start, forest, layout, parts, sharding, stacked, origins, solver_owners, exterior
    )


@final
class _OriginalRootInventory(StrictModule):
    """Authenticated restored root inventory and scientific source banks."""

    hierarchy: _dependency_phydrax_meshing__bisection.BisectionHierarchy
    scientific: _dependency_phydrax_meshing__result.CellMeshingResult
    source: bisection_host.Source
    storage: _dependency_phydrax_discretization__cell_mesh.CellMeshStorage
    root_ids: np.ndarray
    active: np.ndarray


def _admit_original_root_inventory(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
) -> _OriginalRootInventory | None:
    marked = _restored_bisection_request(prepared)
    hierarchy = marked.hierarchy
    if (
        hierarchy is None
        or hierarchy.record_parent_ids.size
        or hierarchy.uniform_refinement is not None
    ):
        return None
    evidence = prepared.source.collective_evidence
    if (
        isinstance(evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence)
        and evidence.uniform_refinement is not None
        and np.intersect1d(
            np.asarray(marked.request.refine_cell_ids, dtype=np.int64),
            np.asarray(evidence.uniform_refinement.parent_ids, dtype=np.int64),
        ).size
    ):
        # Regenerate only explicitly marked restored roots; the mixed preparer
        # keeps every other parent compact.
        return None
    scientific = _dependency_phydrax_meshing__result.require_original_meshing_source(
        prepared.source
    )
    if not isinstance(scientific, _dependency_phydrax_meshing__result.CellMeshingResult):
        return None
    source = bisection_host.prepared_source(scientific.mesh)
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
    return _OriginalRootInventory(
        hierarchy, scientific, source, storage, root_ids, active
    )


def _prepare_original_root_forest(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    old_layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    retained: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    inventory: _OriginalRootInventory,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex:
    hierarchy = inventory.hierarchy
    scientific = inventory.scientific
    source = inventory.source
    storage = inventory.storage
    root_ids = inventory.root_ids
    active = inventory.active
    phase = _authenticate_original_root_request(
        prepared=prepared,
        retained=retained,
        old_layout=old_layout,
        source=source,
        root_ids=root_ids,
        active=active,
    )
    source = phase.source
    phase = _place_original_root_forest(
        prepared=prepared,
        retained=retained,
        scientific=scientific,
        hierarchy=hierarchy,
        root_ids=root_ids,
        storage=storage,
        phase=phase,
        scope=scope,
        budget=budget,
    )
    start = phase.start
    forest = phase.forest
    layout = phase.layout
    parts = phase.parts
    sharding = phase.sharding
    stacked = phase.stacked
    origins = phase.origins
    solver_owners = phase.solver_owners
    exterior = phase.exterior
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
        scope=scope,
        budget=budget,
    )


def _scientific_simplex_blocks(
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh, /
) -> tuple[CellBlock, ...]:
    """Retain authored block identity while rejecting non-simplex bank declarations."""
    blocks: list[CellBlock] = []
    for block in mesh.blocks:
        if not isinstance(block, CellBlock):
            raise TypeError(
                "device_adaptation.scientific_simplex_blocks.block must provide cell block declarations; got an absent or incompatible value"
            )
        blocks.append(block)
    return tuple(blocks)


def _raw_source_blocks(
    source: _dependency_phydrax_meshing__result.CellMeshingResult, /
) -> tuple[CellBlock, ...]:
    """Read the raw numeric bank declaration, never the regrouped target."""
    evidence = source.collective_evidence
    if isinstance(evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence):
        return evidence.publication_source_blocks
    return _scientific_simplex_blocks(source.mesh)


def _source_simplex_on_states(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    raw_source_blocks: tuple[CellBlock, ...] | None = None,
    uniform_refinement: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PreparedAdaptiveSimplex:
    """Bind source-local metadata to an current addressable forest."""
    local_source = bisection_host.prepared_source(prepared.source.mesh)
    first = states.mesh.cell_ids.addressable_shards[0].index[0]
    if not isinstance(first, slice):
        raise ValueError(
            "device_adaptation.source_simplex_on_states.first must provide its explicit leading part rectangle; got an absent or incompatible value"
        )
    part = 0 if first.start is None else first.start
    local_state = _addressable_part_state(states, part)
    width = local_source.dimension + 1
    empty = np.zeros((0,), dtype=np.int64)
    hierarchy = (
        prepared.request.hierarchy
        if isinstance(
            prepared.request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
        )
        else None
    )
    if hierarchy is not None and not isinstance(
        hierarchy, _dependency_phydrax_meshing__bisection.BisectionHierarchy
    ):
        raise TypeError(
            "device_adaptation.source_simplex_on_states.hierarchy must provide its bisection hierarchy; got an absent or incompatible value"
        )
    lineage = None if hierarchy is None else hierarchy.uniform_refinement
    evidence = prepared.source.collective_evidence
    if isinstance(evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence):
        lineage = evidence.uniform_refinement
    if uniform_refinement is not None:
        lineage = uniform_refinement
    start = bisection_host.Start(
        bisection_host.Front(
            local_source.cells,
            bisection_host.Growth(
                np.zeros((0, width), dtype=np.int64),
                np.zeros((0, width), dtype=np.float64),
                empty,
            ),
            np.zeros((0, 2), dtype=np.int64),
            empty,
            local_source.vertex_ids.size,
            int(jax.device_get(jnp.max(states.cursors[:, 3]))),
        ),
        bisection_host.Records(
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


@checked
def commit_partitioned_mesh_restart(
    repack: _owner_phydrax_meshing__restart_distribution.SimplexRestartRepack,
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
) -> _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult:
    """Publish an independently validated changed-count accepted source epoch."""

    if (
        repack.source_result is None
        or prepared.source.result_id != repack.source_result.result_id
    ):
        raise ValueError(
            "device_adaptation.commit_partitioned_mesh_restart.repack must bind the saved accepted source result; got an incompatible value"
        )
    if not isinstance(
        prepared.request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
    ) or (
        prepared.request.refine_cell_ids.size or prepared.request.coarsen_cell_ids.size
    ):
        raise ValueError(
            "device_adaptation.commit_partitioned_mesh_restart.prepared must provide its placement-only marked request; got an absent or incompatible value"
        )
    policy = prepared.policy.partition_policy
    if policy is None or policy.part_count != repack.parts.part_count:
        raise ValueError(
            "device_adaptation.commit_partitioned_mesh_restart.policy must satisfy the requested owner count; got an incompatible value"
        )

    proof = _owner_phydrax_meshing__restart_distribution.SimplexRestartRepackProof(repack)
    parts = repack.execution_parts()
    source = repack.source_result
    storage = source.mesh.storage
    evidence = source.collective_evidence
    if storage is None or not isinstance(
        evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
    ):
        raise ValueError(
            "device_adaptation.commit_partitioned_mesh_restart.storage must provide retained source forest science; got an absent or incompatible value"
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
            "device_adaptation.commit_partitioned_mesh_restart.active must exist in the accepted logical topology; got restart active records are absent from the accepted logical topology"
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
    logical_facet_rows = key_positions(searchable_facets, logical_keys)
    incidence = (
        jnp.zeros((facet_table.shape[0],), dtype=jnp.int32)
        .at[jnp.maximum(logical_facet_rows, 0)]
        .add(logical_valid.astype(jnp.int32))
    )
    target_cells = logical_cells[safe_rows]
    target_keys = jnp.sort(target_cells[..., columns], axis=-1).reshape((-1, dimension))
    target_facet_rows = key_positions(searchable_facets, target_keys).reshape(
        (*repack.states.mesh.cell_ids.shape, width)
    )
    if not bool(jax.device_get(jnp.all(~active[..., None] | (target_facet_rows >= 0)))):
        raise ValueError(
            "device_adaptation.commit_partitioned_mesh_restart.target_facet_rows must reference facets present in the accepted logical topology; got restart cells reference a facet absent from accepted logical topology"
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
            "device_adaptation.commit_partitioned_mesh_restart.repack must provide its old-to-new solver traffic proof; got missing its old-to-new solver traffic proof"
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
    replicated = jax.sharding.NamedSharding(parts.mesh, jax.sharding.PartitionSpec())
    graph = repack.graph_proposal
    if graph is not None:
        arrays["placement/graph/parts"] = graph.parts
        arrays["placement/graph/vertex_weights"] = graph.vertex_weights
        arrays["placement/graph/edge_weights"] = graph.edge_weights
        arrays["placement/graph/maximum_imbalance"] = jax.device_put(
            np.asarray(graph.maximum_imbalance, dtype=np.float64),
            replicated,
        )
        for name, value in graph.workset.arrays().items():
            arrays[f"placement/graph/workset/{name}"] = value
    arrays["placement/proof_content_id"] = jax.device_put(
        np.frombuffer(bytes.fromhex(proof.content_id), dtype=np.uint8).copy(),
        replicated,
    )
    allowance = _phase_allowance(prepared, None)
    limits = prepared.policy.limits
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
        bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
        budget.ticket.native_target(),
    ):
        simplex = _source_simplex_on_states(
            prepared, repack.layout, repack.states, scope=scope, budget=budget
        )
        _retain_preparation(simplex, scope=scope, budget=budget)
        budget.retain_basis((repack, prepared))
        partitioned = PartitionedAdaptiveSimplex(
            simplex,
            repack.layout,
            parts,
            repack.states,
            exterior,
            jnp.where(repack.states.mesh.cell_active, repack.states.mesh.cell_ids, -1),
            solver_cell_owners=repack.solver_cell_owners,
            placement_arrays=tuple(sorted(arrays.items())),
            restart_repack=repack,
            restart_proof=proof,
            scope=scope,
            budget=budget,
        )
    if scope.evidence is not None:
        partitioned = _bind_partitioned_preparation_receipt(
            partitioned,
            NativeExecutionRecord(scope.evidence, owner_id=prepared.prepared_id),
            scope=scope,
            budget=budget,
        )
    return commit_partitioned_adaptive_simplex(partitioned, repack.states)


def _restart_receipt_from_evidence(
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence, /
) -> _owner_phydrax_meshing__restart_distribution.SimplexRestartRepack:
    """Reconstruct historical scientific inputs without demanding their old devices."""

    source = evidence.source
    storage = source.mesh.storage
    source_evidence = source.collective_evidence
    if storage is None or not isinstance(
        source_evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
    ):
        raise ValueError(
            "device_adaptation.restart_receipt_from_evidence.storage must provide its unchanged accepted predecessor; got an absent or incompatible value"
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
                "device_adaptation.values.indices must have a missing or duplicate declaration; got incompatible axes"
            )
        return tuple(arrays[f"{prefix}{index}"] for index in indices)

    graph = None
    if "placement/graph/parts" in arrays:
        graph = _dependency_phydrax_meshing__distribution_migration.SimplexGraphRestartProposal(
            _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset(
                *(
                    arrays[f"placement/graph/workset/{name}"]
                    for name in _dependency_phydrax_meshing__distribution_migration.WORKSET_FIELDS
                )
            ),
            arrays["placement/graph/vertex_weights"],
            arrays["placement/graph/edge_weights"],
            arrays["placement/graph/parts"],
            arrays["placement/requested_target_owners"],
            float(
                np.asarray(jax.device_get(arrays["placement/graph/maximum_imbalance"]))
            ),
        )
    return _owner_phydrax_meshing__restart_distribution.SimplexRestartRepack(
        _owner_phydrax_meshing__restart_distribution.SimplexRestartPlacement(
            evidence.axis_name, count, evidence.neighbor_pairs
        ),
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


@checked
def execute_partitioned_mesh_adaptation(
    partitioned: PartitionedAdaptiveSimplex, /
) -> _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult:
    preparation = partitioned.prepared.adaptation
    receipt = partitioned.execution_evidence
    if receipt is None:
        receipt = partitioned.prepared.execution_evidence
    allowance = _phase_allowance(preparation, receipt)
    limits = preparation.policy.limits
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        with bisection_host.uniform_budget(allowance, scope=scope, budget=budget):
            _retain_preparation(partitioned, scope=scope, budget=budget)
        results = _execute_partitioned_mesh_adaptation(
            partitioned, scope=scope, budget=budget
        )
    if scope.evidence is None:
        return results
    record = NativeExecutionRecord(
        scope.evidence, preparation_evidence=receipt, owner_id=preparation.prepared_id
    )
    return _bind_collective_execution_receipt(
        preparation, results, record, scope=scope, budget=budget
    )


def _bind_collective_execution_receipt(
    preparation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    results: _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult,
    record: NativeExecutionRecord,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult:
    """Use constructors without copying the raw proof holder."""
    rebound = []
    for result in results.local_results:
        # Binding an Equinox method traverses the collective PyTree before this
        # host-only operation can perform its shallow root reconstruction.
        target = (
            _dependency_phydrax_meshing__result.CellMeshingResult.with_execution_evidence(
                result.target, record
            )
        )
        outcome = _dependency_phydrax_meshing__adaptation.RouteOutcome(
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
            _dependency_phydrax_meshing__adaptation.MeshAdaptationResult(
                preparation,
                outcome,
                result.compliance,
                result.distribution,
                result.elapsed_seconds,
            )
        )
    return _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult(
        tuple(rebound), results.identity
    )


def _execute_partitioned_mesh_adaptation(
    partitioned: PartitionedAdaptiveSimplex,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult:
    """Execute the admitted marked request and publish addressable results."""
    request = partitioned.prepared.adaptation.request
    if not isinstance(
        request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
    ):
        raise TypeError(
            "device_adaptation.execute_partitioned_mesh_adaptation.request must provide its marked request; got an absent or incompatible value"
        )
    refine_marks = partitioned.cell_marks(request.refine_cell_ids)
    # Restored parents expanded during bounded preparation have already consumed
    # that refinement request; do not bisect every child again.
    refine_marks &= partitioned.states.mesh.cell_ids == jnp.asarray(
        partitioned.slot_origins, dtype=jnp.int64
    )
    refined = _dependency_phydrax_discretization__adaptive_simplex.refine_adaptive_simplex_parts(
        partitioned.layout,
        partitioned.parts,
        partitioned.states,
        refine_marks,
    )
    coarsened = _dependency_phydrax_discretization__adaptive_simplex.coarsen_adaptive_simplex_parts(
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
                "device_adaptation.part_arrays.prepared must provide its protection encoding; got an absent or incompatible value"
            )
        protected_codes = np.asarray(prepared.state.protected_codes, dtype=np.int64)
    codes = protected_codes
    codes = codes[codes != np.iinfo(np.int64).max]
    if protected_vertex_capacity is None:
        if prepared is None:
            raise ValueError(
                "device_adaptation.part_arrays.prepared must provide its protection capacity; got an absent or incompatible value"
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "device_adaptation.part_arrays.raw_children must retain the binary source family; got an incompatible value",
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
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    base: _HostEpoch,
    slots: np.ndarray,
    piece: dict[str, np.ndarray],
    /,
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    """Allocate source-owned simplex families through the state owner."""
    vertices = piece["vertices"]
    return _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_state(
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
    adaptation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex:
    """Initialize raw owned roots from native initial publication banks."""
    source = adaptation.source
    storage = source.mesh.storage
    distribution = adaptation.policy.distribution
    device = adaptation.policy.device_policy
    if storage is None or distribution is None or device is None:
        raise ValueError(
            "device_adaptation.prepare_initial_partition.storage must provide its storage and execution policy; got an absent or incompatible value"
        )
    if distribution.partition.part_count != storage.partition_count:
        raise ValueError(
            "device_adaptation.prepare_initial_partition.storage must carry an accepted whole-state migration; got initial owner changes require an accepted whole-state migration"
        )
    prepared = _run_prepared_simplex(adaptation, scope=scope, budget=budget)
    if prepared.anchor.start.uniform or prepared.anchor.start.front.growth.levels.size:
        raise ValueError(
            "device_adaptation.prepare_initial_partition.prepared must allocate the preclosure forest rather than only root labels; got native initial root labels require their preclosure forest, not root-only allocation"
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
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
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
    facet_rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(source.mesh, width - 2),
        keys.reshape((-1, width - 1)),
    )
    physical = storage.local_physical_boundary_facets
    if physical is None:
        raise ValueError(
            "device_adaptation.prepare_initial_partition.physical must provide global physical-boundary receipts; got an absent or incompatible value"
        )
    flags = np.asarray(physical)[facet_rows].reshape((slots.size, width))
    exterior = np.zeros((layout.cell_capacity, width), dtype=np.bool_)
    exterior[: slots.size] = flags
    origins = np.full((layout.cell_capacity,), -1, dtype=np.int64)
    origins[: slots.size] = prepared.anchor.slot_origins[slots]
    initial_evidence = source.collective_evidence
    if initial_evidence is None:
        raise ValueError(
            "device_adaptation.prepare_initial_partition.initial_evidence must retain its source proof; got missing its source proof"
        )
    placement = dict(initial_evidence.logical_arrays)["closure/cell_ids"].sharding
    if not isinstance(placement, jax.sharding.NamedSharding):
        raise ValueError(
            "device_adaptation.prepare_initial_partition.placement must provide explicit globally named placement; got an absent or incompatible value"
        )
    parts = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexParts(
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
        scope=scope,
        budget=budget,
    )


def _validate_initial_partition_source(
    prepared: PreparedAdaptiveSimplex,
    partition_count: int,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    exterior: jax.Array,
    entities: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    origins: np.ndarray | jax.Array,
    /,
    *,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Bind initial owner roots and constraints to the accepted source."""

    source = bisection_host.prepared_source(prepared.adaptation.source.mesh)
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
            "device_adaptation.validate_initial_partition_source.any must match the accepted source content; got mismatched prepared source anchors"
        )
    if (
        states.mesh.cell_ids.shape[0] != partition_count
        or exterior.shape != states.mesh.cells.shape
        or exterior.dtype != jnp.bool_
        or origins.shape != states.mesh.cell_ids.shape
    ):
        raise ValueError(
            "device_adaptation.validate_initial_partition_source.partition_count must match the prepared owner layout; got mismatched initial partition shapes"
        )
    if states.mesh.signature_id != layout.mesh_signature_id:
        raise ValueError(
            "device_adaptation.validate_initial_partition_source.layout must match its declared owner layout; got mismatched initial partition state"
        )
    base = _host_epoch(prepared.state)
    authoritative_exterior = np.asarray(prepared.state.mesh.facet_neighbors) < 0
    source_ids, owner = entities[-1][1:]
    if not isinstance(source_ids, np.ndarray) or not isinstance(owner, np.ndarray):
        raise ValueError(
            "device_adaptation.validate_initial_partition_source.source_ids must provide host entity routing; got authored initial roots require host source entity routing"
        )
    slot_owner = _source_forest_slot_owners(
        prepared, base, source_ids, owner, scope=scope, budget=budget
    )
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "device_adaptation.validate_initial_partition_source.selection must declare a leading owner slice; got initial source parts require a leading owner slice"
            )
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
                    "device_adaptation.validate_initial_partition_source.any must match the accepted source; got mismatched initial owner families or constraints"
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
                    "device_adaptation.validate_initial_partition_source.received_origins must match their source family; got mismatched initial owner origins"
                )
            received_exterior = np.asarray(
                jax.device_get(_addressable_part_array(exterior, part))
            )
            if not np.array_equal(
                received_exterior[: slots.size], authoritative_exterior[slots]
            ) or np.any(received_exterior[slots.size :]):
                raise ValueError(
                    "device_adaptation.validate_initial_partition_source.received_exterior must match the source facets; got mismatched initial exterior flags"
                )


def source_cache_digest(
    tables: tuple[
        tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
    ],
    /,
) -> str:
    """Fingerprint source routing from logical leaves, not host copies."""
    return logical_array_value_collection_digest(
        {
            f"{degree}/{name}": jnp.asarray(value)
            for degree, table in enumerate(tables)
            for name, value in zip(("keys", "ids", "owners"), table, strict=True)
        }
    )


@checked
def source_entity_tables(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation, /
) -> tuple[
    tuple[np.ndarray | jax.Array, np.ndarray | jax.Array, np.ndarray | jax.Array], ...
]:
    """Prepare immutable source closure routing before distributed execution."""

    accepted = prepared.source
    if accepted.mesh.storage is not None:
        evidence = accepted.collective_evidence
        if evidence is None:
            raise ValueError(
                "device_adaptation.source_entity_tables.evidence must provide its collective numerical evidence; got an absent or incompatible value"
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
    source = bisection_host.prepared_source(prepared.source.mesh)
    distribution = prepared.policy.distribution
    if distribution is None:
        raise ValueError(
            "device_adaptation.source_entity_tables.distribution must provide a prepared distribution; got an absent or incompatible value"
        )
    cell_ids = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    order = np.argsort(cell_ids, kind="stable")
    cell_owner = np.asarray(distribution.partition.cell_owner, dtype=np.int32)[order][
        np.searchsorted(cell_ids[order], source.cells.ids)
    ]
    tables = []
    dimension = source.dimension
    for degree in range(dimension + 1):
        keys = entity_keys_owner.entity_keys(source.mesh, degree)
        identifiers = np.asarray(
            source.mesh.entity_set(degree).entity_ids, dtype=np.int64
        )
        owners = np.full(
            identifiers.shape, distribution.partition.part_count, dtype=np.int32
        )
        if degree == dimension:
            owners[entity_keys_owner.key_rows(keys, source.cells.ids[:, None])] = (
                cell_owner
            )
        else:
            columns = np.asarray(
                tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32
            )
            occurrences = np.sort(
                source.vertex_ids[source.cells.rows[:, columns]], axis=2
            )
            rows = entity_keys_owner.key_rows(keys, occurrences.reshape((-1, degree + 1)))
            np.minimum.at(owners, rows, np.repeat(cell_owner, columns.shape[0]))
        if np.any(owners == distribution.partition.part_count):
            raise ValueError(
                "device_adaptation.source_entity_tables.owners must have an incident accepted cell; got an accepted source entity has no incident cell"
            )
        for values in (keys, identifiers, owners):
            values.setflags(write=False)
        tables.append((keys, identifiers, owners))
    return tuple(tables)


@jax.jit
def _source_solver_owner_values(
    identifiers: jax.Array,
    ownership: jax.Array,
    cell_ids: jax.Array,
    cell_active: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    order = jnp.argsort(identifiers, stable=True)
    table = identifiers[order]
    positions = jnp.minimum(jnp.searchsorted(table, cell_ids), table.shape[0] - 1)
    accepted = ~cell_active | (table[positions] == cell_ids)
    values = jnp.where(cell_active, ownership[order[positions]], -1)
    return values, accepted


@checked
def source_solver_cell_owners(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation
    | PreparedAdaptiveSimplex,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> jax.Array:
    """Resolve active source ownership through prepared preclosure origins."""

    cell_ids = states.mesh.cell_ids
    forest = None
    match prepared:
        case PreparedAdaptiveSimplex():
            forest = prepared
            adaptation = prepared.adaptation
        case _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation():
            adaptation = prepared
        case _:
            raise TypeError(
                "device_adaptation.source_solver_cell_owners.prepared must provide a scientific preparation or its prepared forest; got an unsupported preparation"
            )
    storage = adaptation.source.mesh.storage
    evidence = adaptation.source.collective_evidence
    if storage is not None and isinstance(
        evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
    ):
        count = evidence.global_entity_counts[-1]
        identifiers = evidence.entity_ids[-1][:count]
        ownership = dict(storage.logical_arrays)["cell_owners"][:count]
        if ownership.shape != identifiers.shape or ownership.dtype != jnp.int32:
            raise ValueError(
                "device_adaptation.source_solver_cell_owners.ownership must retain their scientific solver-owner bank; got missing their scientific solver-owner bank"
            )
    else:
        entities = source_entity_tables(adaptation)
        identifiers, ownership = (jnp.asarray(value) for value in entities[-1][1:])
    values, accepted = _source_solver_owner_values(
        identifiers, ownership, cell_ids, states.mesh.cell_active
    )
    if forest is not None:
        born = states.mesh.cell_active & ~accepted
        origins, accepted_origins = _source_solver_owner_values(
            jnp.asarray(forest.anchor.cell_ids),
            jnp.asarray(forest.anchor.slot_origins),
            cell_ids,
            born,
        )
        inherited, accepted_inherited = _source_solver_owner_values(
            identifiers, ownership, origins, born
        )
        values = jnp.where(born, inherited, values)
        accepted |= accepted_origins & accepted_inherited
    if not bool(jax.device_get(jnp.all(accepted))):
        raise ValueError(
            "device_adaptation.source_solver_cell_owners.table must exist in the solver ownership bank; got prepared active forest rows are absent from their solver ownership bank"
        )
    return values


def _source_forest_slot_owners(
    prepared: PreparedAdaptiveSimplex,
    base: _HostEpoch,
    source_ids: np.ndarray,
    source_owners: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Assign only declared binary and packed source trees."""
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
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
                "device_adaptation.source_forest_slot_owners.owners must own the declared binary source tree; got an incompatible value",
                stage="distributed-source-ownership",
                entity_ids=(int(base.cell_ids[root]),),
            )
        owner[member] = owners[0]
    for part in np.unique(owner):
        _owned_uniform_refinement(
            prepared.anchor.start.uniform_refinement,
            base.cell_ids[:count][active & (owner == part)],
            prepared.anchor.start.records,
            scope=scope,
            budget=budget,
        )
    return owner


@checked
def partition_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
) -> PartitionedAdaptiveSimplex:
    allowance = _phase_allowance(prepared.adaptation, prepared.execution_evidence)
    limits = prepared.adaptation.policy.limits
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
        bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
        budget.ticket.native_target(),
    ):
        _retain_preparation(prepared, scope=scope, budget=budget)
        partitioned = _partition_adaptive_simplex(
            prepared, devices=devices, scope=scope, budget=budget
        )
    if scope.evidence is None:
        return partitioned
    return _bind_partitioned_preparation_receipt(
        partitioned,
        NativeExecutionRecord(
            scope.evidence,
            preparation_evidence=prepared.execution_evidence,
            owner_id=prepared.adaptation.prepared_id,
        ),
        scope=scope,
        budget=budget,
    )


def _partition_adaptive_simplex(
    prepared: PreparedAdaptiveSimplex,
    /,
    *,
    devices: Sequence[jax.Device] | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PartitionedAdaptiveSimplex:
    """Split one prepared epoch by its distribution into stacked per-part states.

    Each part holds its owned active cells and their vertices with the global
    IDs and cursors of the epoch; ``devices`` default to the first global devices.
    Initial source forests and protections are validated against accepted source
    content, including declared packed sibling trees.
    """

    distribution = prepared.adaptation.policy.distribution
    if distribution is None:
        raise ValueError(
            "device_adaptation.partition_adaptive_simplex.distribution must provide a policy distribution; got partitioned device epochs require a policy distribution"
        )
    part_count = distribution.partition.part_count
    chosen = tuple(jax.devices()[:part_count] if devices is None else devices)
    if len(chosen) != part_count:
        raise ValueError(
            f"device_adaptation.partition.devices must provide {part_count} devices; got insufficient device availability"
        )
    base = _host_epoch(prepared.state)
    origins = prepared.anchor.slot_origins
    native = np.asarray(distribution.cell_global_ids, dtype=np.int64)
    owners = np.asarray(distribution.partition.cell_owner, dtype=np.int64)
    slot_owner = _source_forest_slot_owners(
        prepared, base, native, owners, scope=scope, budget=budget
    )
    owned = tuple(np.flatnonzero(slot_owner == part) for part in range(part_count))
    pieces = tuple(_part_arrays(base, prepared, slots) for slots in owned)
    policy = prepared.adaptation.policy
    # ty: ignore[unresolved-attribute]
    vertex_capacity, cell_capacity = policy.device_policy.capacities(
        max(piece["vertices"].size for piece in pieces),
        max(slots.size for slots in owned),
    )
    layout = _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout(
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
    parts = _adaptive_execution_parts(chosen, neighbor_pairs=neighbor_pairs)
    sharding = jax.sharding.NamedSharding(
        parts.mesh, jax.sharding.PartitionSpec(parts.axis_name)
    )

    def placed(value: jax.Array) -> jax.Array:
        def local(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "device_adaptation.local.index must provide its concrete addressable array rectangle; got an absent or incompatible value"
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
        scope=scope,
        budget=budget,
    )


class _LocalCommit(NamedTuple):
    partition_index: int
    source: _dependency_phydrax_discretization__cell_mesh.CellMesh
    committed: _Committed
    cell_exterior: np.ndarray | None = None


def _addressable_part_state(
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    partition_index: int,
    /,
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    """Extract one process-addressable part without touching remote shards."""

    leaves, structure = jax.tree_util.tree_flatten(states)
    payloads = tuple(
        _addressable_part_payload(value, partition_index) for value in leaves
    )
    return jax.tree_util.tree_unflatten(structure, _project_local_leading_axes(payloads))


def _addressable_part_payload(
    value: jax.Array, partition_index: int, /
) -> tuple[jax.Array, int, jax.Device]:
    for shard in value.addressable_shards:
        selection = shard.index[0]
        if isinstance(selection, slice):
            first = 0 if selection.start is None else selection.start
            last = value.shape[0] if selection.stop is None else selection.stop
            if first <= partition_index < last:
                return shard.data, partition_index - first, shard.device
    raise ValueError(
        "device_adaptation.addressable_part_array.value must be process-addressable; got the requested adaptive part is not process-addressable"
    )


def _addressable_part_array(value: jax.Array, partition_index: int, /) -> jax.Array:
    data, local_index, _ = _addressable_part_payload(value, partition_index)
    return data[local_index]


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
    positions = key_positions(table[order], jnp.asarray(queries, dtype=jnp.int64))
    if bool(jax.device_get(jnp.any(positions < 0))):
        raise ValueError(
            "device_adaptation.source_entity_values.positions must exist in the accepted logical topology; got a local source entity is absent from accepted logical topology"
        )
    rows = order[positions]
    return (
        np.asarray(jax.device_get(jnp.asarray(identifiers)[rows]), dtype=np.int64),
        np.asarray(jax.device_get(jnp.asarray(owners)[rows]), dtype=np.int32),
    )


def _source_closure(
    partitioned: PartitionedAdaptiveSimplex,
    root_ids: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> bisection_host.Source:
    """Lower only the active predecessor cells consumed by one local closure."""

    accepted = partitioned.prepared.adaptation.source
    cell_ids = np.unique(root_ids[root_ids >= 0])
    storage = accepted.mesh.storage
    if accepted.mesh.periodic_topology is not None:
        cell_ids = (
            _owner_phydrax_meshing__collective_geometry.periodic_source_closure_cells(
                accepted.mesh,
                cell_ids,
                cell_capacity=partitioned.layout.cell_capacity,
                vertex_capacity=partitioned.layout.vertex_capacity,
                budget=budget,
            )
        )
    if storage is None:
        original = partitioned.prepared.anchor.source
        rows = np.searchsorted(original.cells.ids, cell_ids)
        if np.any(rows >= original.cells.ids.size) or not np.array_equal(
            original.cells.ids[rows], cell_ids
        ):
            raise ValueError(
                "device_adaptation.source_closure.rows must reference existing accepted source cells; got neighborhood ancestry references absent accepted source cells"
            )
        corners = original.vertex_ids[original.cells.rows[rows]]
        identifiers = np.unique(corners)
        points = original.coordinates[np.searchsorted(original.vertex_ids, identifiers)]
    else:
        arrays = dict(storage.logical_arrays)
        corners, present = _dependency_phydrax_meshing__scope.local_logical_lookup(
            arrays["cell_global_ids"][: storage.global_entity_counts[-1]],
            arrays["cell_vertices"][: storage.global_entity_counts[-1]],
            cell_ids,
        )
        if not np.all(present):
            raise ValueError(
                "device_adaptation.source_closure.present must reference existing accepted logical cells; got neighborhood ancestry references absent accepted logical source cells"
            )
        identifiers = np.unique(corners)
        points, present = _dependency_phydrax_meshing__scope.local_logical_lookup(
            arrays["vertex_global_ids"][: storage.global_entity_counts[0]],
            arrays["coordinates"][: storage.global_entity_counts[0]],
            identifiers,
        )
        if not np.all(present):
            raise ValueError(
                "device_adaptation.source_closure.present must reference existing source vertices; got accepted logical cell ancestry references absent source vertices"
            )
    block = accepted.mesh.blocks[0]
    cells = np.searchsorted(identifiers, corners).astype(np.int32)
    blocks = (CellBlock(block.name, block.cell_kind, cells, global_ids=cell_ids),)
    probe = _dependency_phydrax_discretization__cell_mesh.CellMesh(
        points, blocks, vertex_global_ids=identifiers
    )
    entity_ids = {
        degree: _source_entity_values(
            partitioned.source_entities,
            degree,
            entity_keys_owner.entity_keys(probe, degree),
        )[0]
        for degree in range(1, accepted.mesh.topological_dimension)
    }
    mesh = _dependency_phydrax_discretization__cell_mesh.CellMesh(
        points,
        blocks,
        vertex_global_ids=identifiers,
        entity_global_ids=entity_ids,
    )
    if accepted.mesh.periodic_topology is not None:
        periodic = (
            _owner_phydrax_meshing__collective_geometry.project_source_periodic_topology(
                accepted.mesh, mesh, budget=budget
            )
        )
        mesh = _dependency_phydrax_discretization__cell_mesh.CellMesh(
            points,
            blocks,
            vertex_global_ids=identifiers,
            entity_global_ids=entity_ids,
            periodic_topology=periodic,
        )
    return bisection_host.prepared_source(mesh)


def _predecessor_reverse_supports(
    partitioned: PartitionedAdaptiveSimplex,
    source: bisection_host.Source,
    target_vertices: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the retained midpoint DAG for vertices removed by this epoch."""
    lineage = partitioned.prepared.anchor.start.uniform_refinement
    if lineage is not None:
        support_ids, support_weights, valid = jax.vmap(
            lambda state: (
                _owner_phydrax_meshing__collective_organization.original_vertex_ancestry(
                    state, lineage
                )
            ),
        )(partitioned.states)
        if not bool(jax.device_get(jnp.all(valid))):
            raise ValueError(
                "device_adaptation.predecessor_reverse_supports.valid must retain its full barycentric source support; got missing its full barycentric source support"
            )
        table = partitioned.states.mesh.vertex_ids.reshape((-1, 1))
        keys, order, count = canonical_rows(
            table,
            partitioned.states.mesh.vertex_active.reshape(-1),
        )
        positions = key_positions(keys[:count], jnp.asarray(source.vertex_ids[:, None]))
        if bool(jax.device_get(jnp.any(positions < 0))):
            raise ValueError(
                "device_adaptation.predecessor_reverse_supports.positions must retain a retained predecessor vertex; got missing a retained predecessor vertex"
            )
        selected = order[positions]
        width = source.dimension + 1
        bisection_host.uniform_count(
            source.vertex_ids.size * width,
            source.vertex_ids.size * width * 16,
            budget=budget,
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
        rows = entity_keys_owner.key_rows(
            source.vertex_ids[:, None], np.maximum(ids, 0).reshape((-1, 1))
        ).reshape(ids.shape)
        rows = np.where(weights != 0.0, rows, -1)
        if np.any((weights != 0.0) & (rows < 0)):
            raise ValueError(
                "device_adaptation.predecessor_reverse_supports.weights must retain an scientific vertex; got missing an scientific vertex"
            )
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
    keys, order, count = canonical_rows(table, valid)
    positions = key_positions(
        keys[:count], jnp.asarray(source.vertex_ids[:, None], dtype=jnp.int64)
    )
    if bool(jax.device_get(jnp.any(positions < 0))):
        raise ValueError(
            "device_adaptation.predecessor_reverse_supports.positions must exist in the retained predecessor state; got source closure vertices are absent from retained predecessor state"
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
        references = entity_keys_owner.key_rows(
            source.vertex_ids[:, None], parent_ids[slots].reshape((-1, 1))
        ).reshape((-1, 2))
        if np.any(references < 0):
            raise ValueError(
                "device_adaptation.predecessor_reverse_supports.references must provide its accepted midpoint support; got missing its accepted midpoint support"
            )
        steps.append((slots, references[:, 0], references[:, 1]))
    return bisection_host.reverse_supports(
        steps, source.vertex_ids.size, source.dimension
    )


def _neighborhood_commit(
    partitioned: PartitionedAdaptiveSimplex,
    local: _LocalCommit,
    neighborhood: _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _LocalCommit:
    """Bind received ghost packets to source-supported lineage and geometry."""

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
    source = _source_closure(
        partitioned, predecessor_cells.reshape((-1,)), scope=scope, budget=budget
    )
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
    global_vertices = jnp.where(
        neighborhood.cell_valid[..., None], neighborhood.cell_vertices, -1
    ).reshape((-1,))
    globally_present = np.asarray(
        jax.device_get(
            jnp.isin(
                jnp.asarray(source.vertex_ids),
                global_vertices,
            )
        ),
        dtype=np.bool_,
    )
    surviving_predecessor_vertices = source.vertex_ids[globally_present]
    reverse_sources, reverse_weights = _predecessor_reverse_supports(
        partitioned, source, surviving_predecessor_vertices, scope=scope, budget=budget
    )
    coarsening = bisection_host.Coarsening(
        empty_cells,
        removed_ids,
        link_targets,
        empty,
        np.flatnonzero(~globally_present),
        reverse_sources,
        reverse_weights,
        0,
        empty,
    )
    cells = bisection_host.Cells(
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
            bisection_host.simplex_keys(source.cells.rows, degree + 1),
            bisection_host.simplex_keys(cells.rows, degree + 1),
        )
        for degree in range(1, dimension)
    )
    start = partitioned.prepared.anchor.start
    relations = bisection_host.relations(
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
    witnesses = transition_types_owner.NestedReferenceWitnesses(
        cell_ids[refined],
        roots[refined],
        _dependency_phydrax_meshing__topology_edit.nested_reference_vertices(
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
    coarsened_witnesses = transition_types_owner.NestedReferenceWitnesses(
        removed_ids,
        link_targets,
        _dependency_phydrax_meshing__topology_edit.nested_reference_vertices(
            reverse_ids[source.cells.rows[removed_source_rows]],
            reverse_weights[source.cells.rows[removed_source_rows]],
            global_rows[merged_target_rows],
        ),
    )
    block = local.committed.edit.blocks[0]
    if not isinstance(
        block, _dependency_phydrax_meshing__topology_edit.TopologyEditBlock
    ):
        raise ValueError(
            "device_adaptation.neighborhood_commit.block must provide a fixed-family topology edit; got an absent or incompatible value"
        )
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
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
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
            "device_adaptation.local_anchor.raw_groups must provide its declared presentation bank; got missing its declared presentation bank"
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
    probe = _dependency_phydrax_discretization__cell_mesh.CellMesh(
        initial.coordinates[:vertex_count], tuple(blocks), vertex_global_ids=vertices
    )
    entity_ids = {
        degree: _source_entity_values(
            entities,
            degree,
            entity_keys_owner.entity_keys(probe, degree),
        )[0]
        for degree in range(1, global_source.topological_dimension)
    }
    mesh = _dependency_phydrax_discretization__cell_mesh.CellMesh(
        initial.coordinates[:vertex_count],
        probe.blocks,
        vertex_global_ids=vertices,
        entity_global_ids=entity_ids,
    )
    source = bisection_host.prepared_source(mesh)
    cells = bisection_host.Cells(
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
    growth = bisection_host.Growth(
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
            "device_adaptation.local_anchor.record_slots must provide its declared presentation bank; got missing its declared presentation bank"
        )
    if np.any(block_indices[initial.blocks[record_slots]] < 0):
        raise ValueError(
            "device_adaptation.local_anchor.block_indices must provide a resident descendant presentation bank; got missing a resident descendant presentation bank"
        )
    records = bisection_host.Records(
        initial.cell_ids[record_slots],
        block_indices[initial.blocks[record_slots]],
        initial.cells[record_slots],
        initial.tuples[record_slots],
        initial.tags[record_slots],
        initial.cell_ids[initial.children[record_slots]],
        initial.bisection_vertices[record_slots],
    )
    front = bisection_host.Front(
        cells,
        growth,
        np.zeros((0, 2), dtype=np.int64),
        empty,
        vertex_count,
        int(initial.cursors[3]),
    )
    start = bisection_host.Start(
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
            scope=scope,
            budget=budget,
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
    """Bind host witnesses to IDs issued by distributed ordering."""

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
        # Fresh host records use the synthetic namespace; coincident target IDs
        # do not change the meaning of those source keys.
        positions = np.searchsorted(synthetic, np.maximum(array, 0))
        safe = np.minimum(positions, synthetic.size - 1)
        provisional = valid & (positions < synthetic.size) & (synthetic[safe] == array)
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
        uniform = _dependency_phydrax_meshing__bisection.BisectionUniformRefinement(
            uniform.dimension, *arrays.values(), source=uniform.source
        )
    forest = _dependency_phydrax_meshing__bisection.BisectionHierarchy(
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
    host: _HostEpoch,
    initial_vertices: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Resolve sampled midpoint residuals; coefficient maps retain nonzero roundoff signs."""

    count = int(host.cursors[0])
    if not initial_vertices <= count <= host.coordinates.shape[0]:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "device_adaptation.require_exact_edge_restrictions.initial_vertices must bound every allocated restriction vertex; got the affine restriction support has invalid allocated vertex bounds",
            stage="distributed-affine-witness",
        )
    if not np.all(np.isfinite(host.coordinates[:count])):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.QUALITY_REJECTED,
            "device_adaptation.require_exact_edge_restrictions.host must exclude non-finite coordinates; got the affine restriction support contains non-finite coordinates",
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "device_adaptation.require_exact_edge_restrictions.endpoints must provide an earlier, distinct parent edge; got missing an earlier, distinct parent edge",
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
        with budget.measurement():
            signs = exact_orient2d(first, second, middle)
    except MeshcoreUnavailableError as error:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            f"device_adaptation.midpoint.backend must provide native orientation predicates; got {error}",
            stage="distributed-affine-witness",
        ) from error
    except MeshcoreError as error:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            if error.status in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)
            else _leaf_phydrax_meshing__failures.MeshingFailureCategory.QUALITY_REJECTED,
            f"device_adaptation.midpoint.coordinates must certify affine midpoint placement; got {error}",
            stage="distributed-affine-witness",
        ) from error
    except ValueError as error:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.QUALITY_REJECTED,
            f"device_adaptation.midpoint.coordinates must certify affine midpoint placement; got {error}",
            stage="distributed-affine-witness",
        ) from error
    expected = (
        0.5 * host.coordinates[endpoints[:, 0]] + 0.5 * host.coordinates[endpoints[:, 1]]
    )
    if not np.array_equal(host.coordinates[slots], expected):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.QUALITY_REJECTED,
            "device_adaptation.require_exact_edge_restrictions.expected must match their declared midpoint sampling operation; got incompatible values",
            stage="distributed-reference-restriction",
        )
    return np.asarray(signs, dtype=np.int8)


@checked
def require_partitioned_exact_restrictions(
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    initial_states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[jax.Array, jax.Array]:
    """Consume native decisions and retain every resolved source-sample residual."""

    failure: _leaf_phydrax_meshing__failures.MeshingFailure | None = None
    valid_packets: dict[int, np.ndarray] = {}
    sign_packets: dict[int, np.ndarray] = {}
    try:
        with budget.measurement():
            for shard in states.mesh.cell_ids.addressable_shards:
                selection = shard.index[0]
                if not isinstance(selection, slice):
                    raise ValueError(
                        "device_adaptation.require_partitioned_exact_restrictions.selection must declare a leading partition slice; got adaptive part shards require a leading partition slice"
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
                    _dependency_phydrax_meshing__device_status.require_committed_status(
                        host.flags,
                        host.coordinates,
                        host.cells[host.cell_active],
                        _dependency_phydrax_meshing__device_status.DeviceEpoch.BISECTION,
                    )
                    signs = _require_exact_edge_restrictions(
                        host, initial_vertices, scope=scope, budget=budget
                    )
                    valid = np.zeros((layout.vertex_capacity,), dtype=np.bool_)
                    packet = np.zeros(
                        (layout.vertex_capacity, layout.ambient_dimension), dtype=np.int8
                    )
                    count = int(host.cursors[0])
                    valid[initial_vertices:count] = True
                    packet[initial_vertices:count] = signs
                    valid_packets[part] = valid
                    sign_packets[part] = packet
    except _leaf_phydrax_meshing__failures.MeshingFailure as error:
        failure = error
    except ValueError as error:
        failure = _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            str(error),
            stage="distributed-affine-witness",
        )
    collective_failure_owner.collective_failure_barrier(
        failure,
        message="An owning process rejected the exact topology, lineage, or geometry resolution of the distributed epoch; no part was published.",
        stage="distributed-device-commit",
    )
    sharding = states.mesh.cell_ids.sharding

    def placed(packets: dict[int, np.ndarray], shape: tuple[int, ...]) -> jax.Array:
        def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "device_adaptation.addressable.index must provide its concrete addressable array rectangle; got an absent or incompatible value"
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
    preparation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation, /
) -> (
    _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence
):
    request = preparation.request
    hierarchy = (
        request.hierarchy
        if isinstance(
            request, _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation
        )
        else None
    )
    if hierarchy is not None and not isinstance(
        hierarchy, _dependency_phydrax_meshing__bisection.BisectionHierarchy
    ):
        raise TypeError(
            "device_adaptation.scientific_preparation_source.hierarchy must provide its bisection hierarchy; got an absent or incompatible value"
        )
    lineage = None if hierarchy is None else hierarchy.uniform_refinement
    return _dependency_phydrax_meshing__result.require_original_meshing_source(
        preparation.source if lineage is None else lineage.source,
    )


@checked
def packed_publication_states(
    preparation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    initial: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    compiled: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    exterior: jax.Array,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None,
    /,
    *,
    raw_source_blocks: tuple[CellBlock, ...],
    initial_solver_cell_owners: jax.Array,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    jax.Array,
    tuple[CellBlock, ...],
]:
    if lineage is None:
        return compiled, exterior, raw_source_blocks
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
        scope=scope,
        budget=budget,
    )
    return (
        _physical_packed_publication_states(
            published, lineage, scope=scope, budget=budget
        ),
        boundary,
        tuple(declarations),
    )


def _physical_packed_publication_states(
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    """Release coefficient-construction scratch while retaining the source."""
    with budget.temporary_scope():
        budget.retain_basis((lineage.source,))
        physical = _run_physical_packed_publication_states(
            states, lineage, scope=scope, budget=budget
        )
        budget.retain_basis((physical,))
        return physical


def _run_physical_packed_publication_states(
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
    """Evaluate the retained full action chain, not merged binary64 weights."""

    _owner_phydrax_meshing__collective_geometry.original_p1_vertex_images(
        lineage.source, budget=budget
    )
    original = lineage.source
    elements, routes, _ = original.geometry.resolve(original.mesh, budget=budget)
    bank = _owner_phydrax_discretization__coordinate_enclosure.prepared_coordinate_source_bank(
        original.geometry, budget=budget
    )
    sources = {
        int(identifier): (
            _owner_phydrax_discretization__cell_geometry.require_full_p1_source(element),
            np.asarray(route)[row],
        )
        for block, element, route in zip(
            original.mesh.blocks, elements, routes, strict=True
        )
        for row, identifier in enumerate(block.global_ids)
    }
    roots, counts, stacks = (
        _owner_phydrax_meshing__collective_geometry.collective_action_stacks(
            states, lineage, budget=budget
        )
    )
    coordinates: dict[int, np.ndarray] = {}
    incident = {}
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "device_adaptation.run_physical_packed_publication_states.selection must provide addressable owner rectangles; got an absent or incompatible value"
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
            bisection_host.uniform_count(
                state.mesh.coordinates.size + actions.size,
                state.mesh.coordinates.size * 8
                + actions.nbytes
                + root_ids.nbytes
                + action_counts.nbytes,
                budget=budget,
            )
            carrier = np.array(host.coordinates, copy=True)
            for slot in np.flatnonzero(host.cell_active):
                root = int(root_ids[slot])
                if root not in sources:
                    raise ValueError(
                        "device_adaptation.run_physical_packed_publication_states.root must bind a source element; got a physical action has no sci source element"
                    )
                element, route = sources[root]
                for action in reversed(actions[slot, : action_counts[slot]]):
                    element = _owner_phydrax_discretization__cell_geometry.BarycentricCellGeometryElement(
                        element, action
                    )
                images = _owner_phydrax_discretization__coordinate_enclosure.coordinate_corner_images(
                    element, tuple(bank[int(index)] for index in route), budget=budget
                )
                if images is None:
                    raise ValueError(
                        "device_adaptation.run_physical_packed_publication_states.images must retain its scientific coefficient law; got missing its scientific coefficient law"
                    )
                for vertex, image in zip(host.cells[slot], images, strict=True):
                    identifier = int(host.vertex_ids[vertex])
                    if identifier in incident and incident[identifier] != image:
                        raise ValueError(
                            "device_adaptation.run_physical_packed_publication_states.identifier must agree on their physical carrier; got conflicting values"
                        )
                    incident[identifier] = image
                    carrier[vertex] = (
                        _owner_phydrax_discretization__coordinate_enclosure.rounded_point(
                            image
                        )
                    )
            coordinates[part] = carrier

    def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
        if index is None:
            raise ValueError(
                "device_adaptation.addressable.index must provide concrete addressable slices; got an absent or incompatible value"
            )
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


@final
class _PackedRootRows(StrictModule):
    """Restored parent slots authenticated against the admitted host inverse topology."""

    values: dict[str, np.ndarray]
    boundary: np.ndarray


def _restore_packed_root_rows(
    host: _HostEpoch,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    exterior: jax.Array,
    part: int,
    restored: np.ndarray,
    arrays: dict[str, np.ndarray],
    publication_blocks: np.ndarray,
    original_exterior: np.ndarray,
    original_rows: np.ndarray,
    target_ids: np.ndarray,
    width: int,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PackedRootRows:
    """Restored parent slots authenticated against the admitted host inverse topology."""
    bisection_host.uniform_count(
        sum(value.size for value in host if isinstance(value, np.ndarray)),
        sum(value.nbytes for value in host if isinstance(value, np.ndarray))
        + layout.cell_capacity * width,
        budget=budget,
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
        siblings = np.flatnonzero(np.isin(values["cell_ids"], arrays["child_ids"][root]))
        if siblings.size != arrays["child_ids"].shape[1]:
            raise ValueError(
                "device_adaptation.restore_packed_root_rows.siblings must retain its sibling slots; got missing its sibling slots"
            )
        slot = int(siblings[0])
        values["cell_active"][siblings] = False
        values["retired"][siblings] = True
        boundary[siblings] = False
        vertices = values["vertex_ids"]
        order = np.argsort(vertices)
        rows = order[np.searchsorted(vertices[order], arrays["parent_rows"][root])]
        tuples = order[np.searchsorted(vertices[order], arrays["parent_vertices"][root])]
        if not np.array_equal(vertices[rows], arrays["parent_rows"][root]):
            raise ValueError(
                "device_adaptation.restore_packed_root_rows.vertices must retain an scientific vertex; got missing an scientific vertex"
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
        values["blocks"][slot] = publication_blocks[int(arrays["parent_blocks"][root])]
        values["facet_classes"][slot] = arrays["parent_facet_classes"][root][::-1] + 1
        values["parents"][slot] = -1
        values["children"][slot] = -1
        values["bisection_vertices"][slot] = -1
        values["cell_active"][slot] = True
        values["retired"][slot] = False
        values["coarsen_marked"][slot] = False
        boundary[slot] = original_exterior[original_rows[root]]
    values["vertex_active"][:] = False
    values["vertex_active"][np.unique(values["cells"][values["cell_active"]])] = True
    accepted_ids = values["cell_ids"][values["cell_active"]]
    if not np.array_equal(np.sort(accepted_ids), np.sort(target_ids)):
        raise ValueError(
            "device_adaptation.restore_packed_root_rows.accepted_ids must match the admitted host inverse topology; got mismatched packed publication"
        )
    return _PackedRootRows(values, boundary)


@final
class _PackedRootProjection(StrictModule):
    """Compact restored ancestry into the device bucket while retaining compiled counters and clocks."""

    projected: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    boundary: np.ndarray


def _project_packed_root_state(
    host: _HostEpoch,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    local: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    values: dict[str, np.ndarray],
    boundary: np.ndarray,
) -> _PackedRootProjection:
    """Compact restored ancestry into the device bucket while retaining compiled counters and clocks."""
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
    values["children"] = np.where(children >= 0, inverse[np.maximum(children, 0)], -1)
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
    projected = (
        _dependency_phydrax_discretization__adaptive_simplex.adaptive_simplex_state(
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
    return _PackedRootProjection(projected, boundary)


def _run_packed_publication_states(
    preparation: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    initial: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    compiled: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    exterior: jax.Array,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement | None,
    /,
    *,
    raw_source_blocks: tuple[CellBlock, ...],
    initial_solver_cell_owners: jax.Array,
    publication_blocks: np.ndarray,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState, jax.Array
]:
    """Derive the host inverse independently; never mutate compiled receipts."""
    if lineage is None:
        return compiled, exterior
    original = _source_simplex_on_states(
        preparation,
        layout,
        initial,
        raw_source_blocks=raw_source_blocks,
        uniform_refinement=lineage,
        scope=scope,
        budget=budget,
    )
    entities = source_entity_tables(preparation)
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
        seed = logical_publication(
            preparation.source,
            initial,
            initial_solver_cell_owners,
            exterior,
            layout,
            original_source=lineage.source,
            uniform_refinement=lineage,
            numeric_version=f"adaptation:{preparation.prepared_id}",
            scope=scope,
            budget=budget,
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
    packets: dict[
        int, _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState
    ] = {}
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
    bisection_host.uniform_count(
        source_rows_count, source_rows_count * width * (width * 8 + 32), budget=budget
    )
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
                "device_adaptation.run_packed_publication_states.selection must provide explicit addressable part rectangles; got an absent or incompatible value"
            )
        first = 0 if selection.start is None else selection.start
        last = (
            compiled.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        )
        for part in range(first, last):
            local = _addressable_part_state(compiled, part)
            host = _host_epoch(local)
            baseline = _host_epoch(_addressable_part_state(initial, part))
            anchor = _local_anchor(
                original.anchor, entities, baseline, scope=scope, budget=budget
            )
            prepared = PreparedAdaptiveSimplex(
                preparation,
                layout,
                _addressable_part_state(initial, part),
                anchor,
            )
            committed = _committed(prepared, host, scope=scope, budget=budget)
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
            phase = _restore_packed_root_rows(
                host=host,
                layout=layout,
                exterior=exterior,
                part=part,
                restored=restored,
                arrays=arrays,
                publication_blocks=publication_blocks,
                original_exterior=original_exterior,
                original_rows=original_rows,
                target_ids=target_ids,
                width=width,
                budget=budget,
            )
            values = phase.values
            boundary = phase.boundary
            phase = _project_packed_root_state(
                host=host, layout=layout, local=local, values=values, boundary=boundary
            )
            projected = phase.projected
            boundary = phase.boundary
            packets[part], exterior_packets[part] = projected, boundary

    def place(value: jax.Array, values: dict[int, np.ndarray]) -> jax.Array:
        def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "device_adaptation.addressable.index must provide concrete addressable array slices; got an absent or incompatible value"
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
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence, /
) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
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

    def restore(
        part: dict[str, jax.Array],
    ) -> _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState:
        return _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState(
            _dependency_phydrax_discretization__adaptive_simplex.MaskedSimplexMesh(
                *(part[name] for name in mesh_names)
            ),
            **{name: part[name] for name in state_names},
        )

    return eqx.filter_vmap(restore)(values)


@final
class _PackedUniformSupport(StrictModule):
    """Uniform cells bound to the retained source coefficient support and predecessor IDs."""

    values: dict[str, np.ndarray]


def _bind_packed_uniform_support(
    partitioned: PartitionedAdaptiveSimplex,
    raw: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    lineage: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement,
    parent_ids: np.ndarray,
    sibling_parent: dict[int, int],
    matched: np.ndarray,
    values: dict[str, np.ndarray],
    ids: np.ndarray,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PackedUniformSupport:
    """Uniform cells bound to the retained source coefficient support and predecessor IDs."""
    support_ids, support_weights, passed = (
        _owner_phydrax_meshing__collective_organization.original_vertex_ancestry(
            raw, lineage
        )
    )
    if not bool(jax.device_get(passed)):
        raise ValueError(
            "device_adaptation.bind_packed_uniform_support.passed must retain its full coefficient support; got missing its full coefficient support"
        )
    bisection_host.uniform_count(
        support_ids.size + support_weights.size,
        support_ids.nbytes + support_weights.nbytes,
        budget=budget,
    )
    raw_vertices = np.asarray(jax.device_get(raw.mesh.vertex_ids))
    vertex_order = np.argsort(raw_vertices, kind="stable")
    queries = values["cell_vertices"][: ids.size]
    positions = np.searchsorted(raw_vertices[vertex_order], queries)
    if np.any(positions >= raw_vertices.size) or not np.array_equal(
        raw_vertices[vertex_order][positions], queries
    ):
        raise ValueError(
            "device_adaptation.bind_packed_uniform_support.positions must provide a published retained-source vertex; got missing a published retained-source vertex"
        )
    vertex_rows = vertex_order[positions]
    source_ids = np.asarray(jax.device_get(support_ids))
    source_weights = np.asarray(jax.device_get(support_weights))
    regenerated_parents = set(partitioned.prepared.anchor.source.cells.ids.tolist())
    regenerated_children = set(partitioned.prepared.anchor.start.front.cells.ids.tolist())
    for row in range(ids.size):
        if matched[row] and int(values["root_cell_ids"][row]) not in regenerated_children:
            continue
        parent = (
            sibling_parent.get(int(values["root_cell_ids"][row]))
            if matched[row]
            else int(ids[row])
        )
        if parent is None or parent not in parent_ids:
            raise ValueError(
                "device_adaptation.bind_packed_uniform_support.parent must provide its accepted predecessor SCI; got missing its accepted predecessor SCI"
            )
        if parent not in regenerated_parents:
            continue
        values["source_vertex_ids"][row] = source_ids[vertex_rows[row]]
        values["source_weights"][row] = source_weights[vertex_rows[row]]
        values["root_cell_ids"][row] = parent
        values["source_cell_ids"][row] = -1
        values["source_cell_ids"][row, 0] = parent
    return _PackedUniformSupport(values)


def _restore_packed_predecessors(
    baseline: _HostEpoch,
    values: dict[str, np.ndarray],
    matched: np.ndarray,
    ids: np.ndarray,
    parent_ids: np.ndarray,
    child_ids: np.ndarray,
    /,
) -> None:
    """Bind unmatched restored parents to their retained predecessor support."""
    roots = np.arange(baseline.cell_ids.size, dtype=np.int64)
    for slot in range(int(baseline.cursors[1])):
        parent = baseline.parents[slot] // 2
        if parent >= 0:
            roots[slot] = roots[parent]
    for row in np.flatnonzero(~matched):
        root = np.searchsorted(parent_ids, ids[row])
        if root >= parent_ids.size or parent_ids[root] != ids[row]:
            raise ValueError(
                "device_adaptation.packed_neighborhood.root must exist in raw output or the retained parent bank; got publication issued a cell absent from raw output and retained parent provenance"
            )
        predecessor = baseline.cell_ids[
            baseline.cell_active & np.isin(baseline.cell_ids[roots], child_ids[root])
        ]
        if predecessor.size == 0 or predecessor.size > values["source_cell_ids"].shape[1]:
            raise ValueError(
                "device_adaptation.packed_neighborhood.predecessor must retain its predecessor source support; got missing its predecessor source support"
            )
        values["root_cell_ids"][row] = -1
        values["source_cell_ids"][row] = -1
        values["source_cell_ids"][row, : predecessor.size] = predecessor
        values["source_vertex_ids"][row] = -1
        values["source_weights"][row] = 0.0
        values["source_vertex_ids"][row, :, 0] = values["cell_vertices"][row]
        values["source_weights"][row, :, 0] = 1.0


# Predecessor restoration is host-only; packet transfers and collective placement retain their order.
def _packed_neighborhood(
    partitioned: PartitionedAdaptiveSimplex,
    compiled: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset:

    lineage = evidence.uniform_refinement
    if lineage is None:
        raise ValueError(
            "device_adaptation.packed_neighborhood.lineage must provide its retained uniform source provenance; got an absent or incompatible value"
        )
    published = _collective_publication_state(evidence)
    arrays = dict(evidence.logical_arrays)
    parent_ids = np.asarray(lineage.parent_ids)
    child_ids = np.asarray(lineage.child_ids)
    capacity = partitioned.layout.cell_capacity
    prepared_uniform = partitioned.prepared.anchor.start.uniform
    if prepared_uniform:
        sibling_parent = {
            int(child): int(parent)
            for parent, children in zip(parent_ids, child_ids, strict=True)
            for child in children
        }
    packets: dict[
        int, _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset
    ] = {}
    for shard in compiled.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "device_adaptation.packed_neighborhood.selection must provide explicit addressable part rectangles; got an absent or incompatible value"
            )
        first = 0 if selection.start is None else selection.start
        last = (
            compiled.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        )
        for part in range(first, last):
            raw = _addressable_part_state(compiled, part)
            initial = _addressable_part_state(partitioned.states, part)
            seed = _owner_phydrax_meshing__distribution.simplex_neighborhood_seed(
                raw,
                initial,
                _addressable_part_array(partitioned.source_exterior, part),
                jnp.asarray(part, dtype=jnp.int32),
                capacity,
            )
            seed_leaves = tuple(seed.arrays().values())
            bisection_host.uniform_count(
                sum(value.size for value in seed_leaves),
                sum(value.size * value.dtype.itemsize for value in seed_leaves),
                budget=budget,
            )
            host = jax.device_get(seed)
            values = {
                name: np.array(value, copy=True) for name, value in host.arrays().items()
            }
            final_arrays = _host_epoch(_addressable_part_state(published, part))
            baseline = _host_epoch(initial)
            slots = np.flatnonzero(final_arrays.cell_active)
            order = slots[np.argsort(final_arrays.cell_ids[slots], kind="stable")]
            ids = final_arrays.cell_ids[order]
            raw_ids = np.asarray(host.cell_ids)
            positions = np.searchsorted(raw_ids, ids)
            safe = np.minimum(positions, capacity - 1)
            matched = raw_ids[safe] == ids
            for name, value in host.arrays().items():
                if name != "status":
                    values[name][: ids.size] = np.asarray(value)[safe]
            values["cell_ids"][:] = np.iinfo(np.int64).max
            values["cell_ids"][: ids.size] = ids
            values["cell_valid"][:] = False
            values["cell_valid"][: ids.size] = True
            values["cell_vertices"][: ids.size] = final_arrays.vertex_ids[
                final_arrays.cells[order]
            ]
            values["cell_coordinates"][: ids.size] = final_arrays.coordinates[
                final_arrays.cells[order]
            ]
            values["cell_classes"][: ids.size] = final_arrays.cell_classes[order]
            values["cell_owner"][: ids.size] = part
            boundary = np.asarray(
                jax.device_get(
                    _addressable_part_array(arrays["epoch/source_exterior"], part)
                )
            )
            values["cell_exterior"][: ids.size] = boundary[order]
            _restore_packed_predecessors(
                baseline, values, matched, ids, parent_ids, child_ids
            )
            if prepared_uniform:
                phase = _bind_packed_uniform_support(
                    partitioned=partitioned,
                    raw=raw,
                    lineage=lineage,
                    parent_ids=parent_ids,
                    sibling_parent=sibling_parent,
                    matched=matched,
                    values=values,
                    ids=ids,
                    budget=budget,
                )
                values = phase.values
            packets[part] = (
                _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset(
                    *(
                        jnp.asarray(values[name])
                        for name in _dependency_phydrax_meshing__distribution_migration.WORKSET_FIELDS
                    )
                )
            )

    def place(value: jax.Array, blocks: dict[int, jax.Array]) -> jax.Array:
        shape = (partitioned.parts.part_count, *value.shape)

        def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                raise ValueError(
                    "device_adaptation.addressable.index must provide concrete addressable packet slices; got an absent or incompatible value"
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
    workset = _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset(
        *(
            place(
                value, {part: packet.arrays()[name] for part, packet in packets.items()}
            )
            for name, value in exemplar.arrays().items()
        )
    )
    distribution = partitioned.prepared.adaptation.policy.distribution
    if distribution is None:
        raise ValueError(
            "device_adaptation.packed_neighborhood.distribution must retain its accepted ownership policy; got missing its accepted ownership policy"
        )
    return _owner_phydrax_meshing__distribution.expand_simplex_neighborhood_workset(
        partitioned.parts,
        workset,
        halo_width=distribution.halo_width,
        vertex_capacity=partitioned.layout.vertex_capacity,
    )


def _addressable_commits(
    partitioned: PartitionedAdaptiveSimplex,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_LocalCommit, ...]:
    """Prepare owner-local topology and lineage records for the received closure."""

    anchor = partitioned.prepared.anchor
    premise = _scientific_preparation_source(partitioned.prepared.adaptation)
    seed_entities = partitioned.source_entities
    if anchor.start.uniform or anchor.start.front.growth.levels.size:
        lineage = anchor.start.uniform_refinement
        if not anchor.start.uniform or lineage is None:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SPECIFICATION,
                "device_adaptation.addressable_commits.lineage must provide its uniform source law; got an absent or incompatible value",
                stage="distributed-device-commit",
            )
        seed = logical_publication(
            partitioned.prepared.adaptation.source,
            partitioned.states,
            partitioned.solver_cell_owners,
            partitioned.source_exterior,
            partitioned.layout,
            original_source=premise
            if isinstance(premise, _dependency_phydrax_meshing__result.CellMeshingResult)
            else None,
            uniform_refinement=anchor.start.uniform_refinement,
            numeric_version=f"adaptation:{partitioned.prepared.adaptation.prepared_id}",
            scope=scope,
            budget=budget,
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
    if (
        isinstance(premise, _dependency_phydrax_meshing__result.CellMeshingResult)
        and len(premise.mesh.blocks) != 1
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SPECIFICATION,
            "device_adaptation.addressable_commits.source_binding must provide a single simplex coordinate block; got an absent or incompatible value",
            stage="distributed-device-commit",
        )
    initial_states = partitioned.states
    if partitioned.migration is not None:
        states = partitioned.migration.solver_forest
        initial_states = partitioned.migration.solver_initial_states
    commits = []
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "device_adaptation.addressable_commits.selection must satisfy the declared source, topology, and ownership constraints; got adaptive part shards require a leading partition slice"
            )
        first = 0 if selection.start is None else selection.start
        last = states.mesh.cell_ids.shape[0] if selection.stop is None else selection.stop
        for part in range(first, last):
            local = _addressable_part_state(states, part)
            initial = _host_epoch(_addressable_part_state(initial_states, part))
            host = _host_epoch(local)
            _dependency_phydrax_meshing__device_status.require_committed_status(
                host.flags,
                host.coordinates,
                host.cells[host.cell_active],
                _dependency_phydrax_meshing__device_status.DeviceEpoch.BISECTION,
            )
            local_anchor = _local_anchor(
                partitioned.prepared.anchor,
                seed_entities,
                initial,
                scope=scope,
                budget=budget,
            )
            prepared = PreparedAdaptiveSimplex(
                partitioned.prepared.adaptation,
                partitioned.layout,
                _addressable_part_state(initial_states, part),
                local_anchor,
            )
            committed = _rebind_issued_vertices(
                _committed(prepared, host, scope=scope, budget=budget), local_anchor, host
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
    geometry_projection: (
        _dependency_phydrax_discretization__cell_geometry.CellGeometryStorageProjection
        | None
    ) = None
    publication_projection: (
        _dependency_phydrax_meshing__publication_lowering.PublicationProjection | None
    ) = None
    scope_projections: (
        tuple[tuple[str, _dependency_phydrax_meshing__scope.MeshScopeProjection], ...]
        | None
    ) = None
    attribute_projections: (
        tuple[
            tuple[
                str,
                _dependency_phydrax_meshing__organization_records.MeshAttributeProjection,
            ],
            ...,
        ]
        | None
    ) = None
    restart_proof: (
        _owner_phydrax_meshing__restart_distribution.SimplexRestartRepackProof | None
    ) = None
    storage_binding: (
        _dependency_phydrax_meshing__result.CollectiveMeshStorageBinding | None
    ) = None
    surface_transfer: (
        _dependency_phydrax_meshing__association.SurfaceAssociationTransfer | None
    ) = None
    presentation: _LogicalPresentation | None = None


@jax.jit
def _canonical_row_values(
    keys: jax.Array, valid: jax.Array, /
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Order a capacity-sized key table without specializing on its active count."""

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
    count = jnp.sum(fresh, dtype=jnp.int64)
    selected = jnp.nonzero(fresh, size=keys.shape[0], fill_value=keys.shape[0] - 1)[0]
    prefix = jnp.arange(keys.shape[0]) < count
    return jnp.where(prefix[:, None], ordered[selected], -1), order[selected], count


def canonical_rows(
    keys: jax.Array, valid: jax.Array, /
) -> tuple[jax.Array, jax.Array, int]:
    """Bounded device ordering with an active prefix and explicit host count."""
    rows, order, count = _canonical_row_values(keys, valid)
    return rows, order, int(jax.device_get(count))


@partial(jax.jit, static_argnames=("partition_count",))
def _inherited_entity_values(
    table: jax.Array,
    count: int,
    ledger_keys: tuple[jax.Array, ...],
    ledger_ids: tuple[jax.Array, ...],
    *,
    partition_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    inherited = jnp.full((table.shape[0],), -1, dtype=jnp.int64)
    high = jnp.asarray(-1, dtype=jnp.int64)
    for keys, identifiers in zip(ledger_keys, ledger_ids, strict=True):
        order = jnp.lexsort(keys.T[::-1])
        positions = key_positions(keys[order], table)
        restored = identifiers[order[jnp.maximum(positions, 0)]]
        inherited = jnp.where((inherited < 0) & (positions >= 0), restored, inherited)
        high = jnp.maximum(high, jnp.max(identifiers))
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
    keys, order, history_count = _canonical_row_values(history_keys, history_ids >= 0)
    active_table = jnp.where(table >= 0, table, jnp.iinfo(jnp.int64).max)
    removed = (jnp.arange(keys.shape[0]) < history_count) & (
        key_positions(active_table, keys) < 0
    )
    retired_keys, retired_order, _ = _canonical_row_values(keys, removed)
    retired_ids = jnp.where(
        retired_keys[:, 0] >= 0, history_ids[order[retired_order]], -1
    )
    return ids, retired_keys, retired_ids


def _inherited_entity_ids(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    degree: int,
    table: jax.Array,
    count: int,
    partition_count: int,
    original_source: _dependency_phydrax_meshing__result.CellMeshingResult | None,
    /,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Restore retired scientific entity identities from consumed predecessor leaves."""

    ledger_keys = []
    ledger_ids = []
    while True:
        if source.collective_evidence is None:
            keys = jnp.asarray(
                entity_keys_owner.entity_keys(source.mesh, degree),
                dtype=jnp.int64,
            )
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
        evidence = source.collective_evidence
        if evidence is None or isinstance(
            evidence,
            _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
        ):
            if (
                original_source is not None
                and source.result_id != original_source.result_id
            ):
                source, original_source = original_source, None
                continue
            break
        if not isinstance(
            evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
        ):
            raise TypeError(
                "device_adaptation.inherited_entity_ids.evidence must have a supported scientific evidence kind; got collective source provenance has an unknown certification kind"
            )
        source = evidence.source
    return _inherited_entity_values(
        table,
        count,
        tuple(ledger_keys),
        tuple(ledger_ids),
        partition_count=partition_count,
    )


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
            "device_adaptation.logical_array_tree_fingerprint.logical_shape must trim only a nonempty leading capacity axis; got logical array fingerprints may trim only a nonempty leading capacity axis"
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
                "device_adaptation.logical_array_tree_fingerprint.any must provide contiguous leading-axis shards; got an absent or incompatible value"
            )
        start, stop, step = index[0].indices(value.shape[0])
        if step != 1:
            raise ValueError(
                "device_adaptation.logical_array_tree_fingerprint.step must admit strided shards; got an incompatible value"
            )
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
                "device_adaptation.logical_array_tree_fingerprint.start must cover the leading axis without overlaps or gaps; got logical topology shards overlap or leave leading-axis coverage gaps"
            )
        cursor = stop
    if cursor != value.shape[0]:
        raise ValueError(
            "device_adaptation.logical_array_tree_fingerprint.cursor must cover the logical topology without gaps; got logical topology shards leave incomplete global coverage"
        )
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
                        "device_adaptation.logical_array_tree_fingerprint.shard must have an addressable payload; got the topology shard owner has no addressable payload"
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
    accepted_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    original_source: _dependency_phydrax_meshing__result.CellMeshingResult | None,
    /,
) -> tuple[CellBlock, ...]:
    evidence = accepted_source.collective_evidence
    raw = (
        evidence.publication_source_blocks
        if isinstance(
            evidence, _dependency_phydrax_meshing__result.CollectiveMeshEvidence
        )
        else accepted_source.mesh.blocks
    )
    declarations: list[CellBlock] = []
    for block in raw:
        if not isinstance(block, CellBlock):
            raise TypeError(
                "device_adaptation.publication_block_declarations.block must provide simplex source block declarations; got an absent or incompatible value"
            )
        declarations.append(block)
    if original_source is not None:
        declared = {block.block_id for block in declarations}
        for block in original_source.mesh.blocks:
            if not isinstance(block, CellBlock):
                raise TypeError(
                    "device_adaptation.publication_block_declarations.block must provide simplex source blocks; got an absent or incompatible value"
                )
            if block.block_id not in declared:
                declarations.append(block)
                declared.add(block.block_id)
    if not declarations:
        raise ValueError(
            "device_adaptation.publication_block_declarations.declarations must provide at least one source block; got an absent or incompatible value"
        )
    return tuple(declarations)


def _presentation_roots(
    premise: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[tuple[str, str, str], ...]:
    if isinstance(premise, _dependency_phydrax_meshing__result.CellMeshingResult):
        elements, _, _ = premise.geometry.resolve(premise.mesh, budget=budget)
        return tuple(
            (block.name, block.cell_kind, element.element_id)
            for block, element in zip(premise.mesh.blocks, elements, strict=True)
        )

    element = _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
        "triangle", 1
    )
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
                "device_adaptation.declaration_presentations.element_id must bind a coefficient-action owner; got a raw presentation block has no coefficient-action owner"
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


def _composed_presentation_element(
    source_element: _dependency_phydrax_meshing__adaptation.FullP1Element,
    actions: Sequence[np.ndarray],
    /,
    *,
    packed: bool,
    uniform: bool,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__adaptation.FullP1Element:
    """Compose a binary epoch or one uniform-plus-binary source chart."""

    if not actions:
        return source_element
    width = source_element.local_dof_count
    if any(action.shape != (width, width) for action in actions):
        raise ValueError(
            "device_adaptation.composed_presentation_element.any must retain its simplex basis; got missing its simplex basis"
        )
    if not packed:
        composed = np.eye(width, dtype=np.float64)
        for action in reversed(actions):
            composed = action @ composed
        return (
            source_element
            if np.array_equal(composed, np.eye(width, dtype=np.float64))
            else _owner_phydrax_meshing__adaptation.apply_full_p1_action(
                source_element, composed, budget=budget
            )
        )
    exact = np.asarray(
        [
            [_owner_fractions.Fraction(int(row == column)) for column in range(width)]
            for row in range(width)
        ],
        dtype=object,
    )
    binary = actions
    if uniform:
        subset_rows = []
        for row in actions[-1]:
            support = np.flatnonzero(row)
            if support.size == 0 or not np.all(row[support] == 1.0 / support.size):
                raise ValueError(
                    "presentation.uniform_action must contain declared subset rows; got nonuniform support"
                )
            subset_rows.append(
                [
                    _owner_fractions.Fraction(1, support.size)
                    if column in support
                    else _owner_fractions.Fraction(0)
                    for column in range(width)
                ]
            )
        exact = np.asarray(subset_rows, dtype=object)
        binary = actions[:-1]
    for action in reversed(binary):
        dyadic = np.asarray(
            [
                [_owner_fractions.Fraction(float(value)) for value in row]
                for row in action
            ],
            dtype=object,
        )
        exact = dyadic @ exact
    return _owner_phydrax_meshing__adaptation.uniform_rational_p1_chart(
        source_element, tuple(tuple(row) for row in exact), budget=budget
    )


def _local_presentation_keys(
    state: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    declarations: tuple[tuple[int, str], ...],
    uniform_actions: dict[
        int,
        tuple[
            int, _dependency_phydrax_meshing__adaptation.FullP1Element, np.ndarray | None
        ],
    ],
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
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
                "device_adaptation.local_presentation_keys.state must stay within the declared capacity; got a presentation action chain exceeds the device depth bound"
            )
        block = int(host.blocks[current])
        if block < 0 or block >= len(declarations):
            raise ValueError(
                "device_adaptation.local_presentation_keys.block must declare a source presentation block; got a logical target cell has no declared source presentation block"
            )
        root, element_id = declarations[block]
        uniform = uniform_actions.get(int(host.cell_ids[current]))
        if uniform is not None:
            uniform_root, source_element, uniform_action = uniform
            if root != uniform_root:
                raise ValueError(
                    "presentation.source_block must retain the uniform parent owner; "
                    f"got root {root} instead of {uniform_root}"
                )
            if uniform_action is not None:
                actions.append(uniform_action)
            element_id = _composed_presentation_element(
                source_element,
                actions,
                packed=True,
                uniform=uniform_action is not None,
                scope=scope,
                budget=budget,
            ).element_id
        elif uniform_actions:
            raise ValueError(
                "presentation.source_root must name a retained uniform cell; "
                f"got {int(host.cell_ids[current])}"
            )
        elif actions:
            if any(action.shape != (width, width) for action in actions):
                raise ValueError(
                    "device_adaptation.local_presentation_keys.any must retain its simplex basis; got missing its simplex basis"
                )
            composed = np.eye(width, dtype=np.float64)
            for action in reversed(actions):
                composed = action @ composed
            if not np.array_equal(composed, np.eye(width, dtype=np.float64)):
                element_id = _barycentric_element_id(element_id, composed)
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
                "device_adaptation.replicated_host_rows.sharding must provide a named distributed sharding; got an absent or incompatible value"
            )
        value = jax.device_put(
            value,
            jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec()),
        )
    return np.asarray(jax.device_get(value), dtype=np.int64)


@final
class _PresentationSources(StrictModule):
    """Source block identity and retained coefficient actions used by presentation grouping."""

    roots: tuple[tuple[str, str, str], ...]
    declaration_presentations: tuple[tuple[int, str], ...]
    uniform_actions: dict[
        int,
        tuple[
            int, _dependency_phydrax_meshing__adaptation.FullP1Element, np.ndarray | None
        ],
    ]


def _prepare_presentation_sources(
    accepted_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    premise: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    original_source: _dependency_phydrax_meshing__result.CellMeshingResult | None,
    uniform_refinement: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _PresentationSources:
    """Source block identity and retained coefficient actions used by presentation grouping."""
    roots = _presentation_roots(premise, budget=budget)
    declarations = _publication_block_declarations(accepted_source, original_source)
    declaration_presentations = _declaration_presentations(declarations, roots)
    uniform_actions: dict[
        int,
        tuple[
            int, _dependency_phydrax_meshing__adaptation.FullP1Element, np.ndarray | None
        ],
    ] = {}
    if uniform_refinement is not None:
        source = uniform_refinement.source
        source_elements, _, _ = source.geometry.resolve(source.mesh, budget=budget)
        arrays = uniform_refinement.host_arrays()
        parents = np.asarray(arrays["parent_ids"], dtype=np.int64)
        blocks = np.asarray(arrays["parent_blocks"], dtype=np.int64)
        children = np.asarray(arrays["child_ids"], dtype=np.int64)
        actions = np.asarray(arrays["barycentric_weights"], dtype=np.float64)
        if (
            blocks.ndim != 1
            or parents.shape != blocks.shape
            or children.ndim != 2
            or actions.ndim != 4
            or children.shape != actions.shape[:2]
            or blocks.shape != children.shape[:1]
        ):
            raise ValueError(
                "device_adaptation.prepare_presentation_sources.blocks must have incompatible scientific axes; got incompatible axes"
            )
        for row in range(blocks.shape[0]):
            root = int(blocks[row])
            if root < 0 or root >= len(roots):
                raise ValueError(
                    "device_adaptation.prepare_presentation_sources.root must name an existing source block; got uniform presentation names an absent source block"
                )
            source_element = (
                _owner_phydrax_discretization__cell_geometry.require_full_p1_source(
                    source_elements[root]
                )
            )
            if (
                source.mesh.blocks[root].name != roots[root][0]
                or source_element.element_id != roots[root][2]
            ):
                raise ValueError(
                    "presentation.uniform_source must match its coefficient owner; "
                    "got an incompatible source block"
                )
            uniform_actions[int(parents[row])] = (root, source_element, None)
            for child in range(children.shape[1]):
                uniform_actions[int(children[row, child])] = (
                    root,
                    source_element,
                    actions[row, child],
                )
    return _PresentationSources(roots, declaration_presentations, uniform_actions)


def _logical_presentation(
    accepted_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    premise: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    original_source: _dependency_phydrax_meshing__result.CellMeshingResult | None,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    cell_keys: jax.Array,
    cell_order: jax.Array,
    cells: int,
    uniform_refinement: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    | None,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _LogicalPresentation:
    phase = _prepare_presentation_sources(
        accepted_source=accepted_source,
        premise=premise,
        original_source=original_source,
        uniform_refinement=uniform_refinement,
        budget=budget,
    )
    roots = phase.roots
    declaration_presentations = phase.declaration_presentations
    uniform_actions = phase.uniform_actions
    packets: dict[int, np.ndarray] = {}
    for shard in states.mesh.cell_ids.addressable_shards:
        selection = shard.index[0]
        if not isinstance(selection, slice):
            raise ValueError(
                "device_adaptation.logical_presentation.selection must provide a leading partition rectangle; got an absent or incompatible value"
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
                    scope=scope,
                    budget=budget,
                ),
            )

    def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
        if index is None:
            raise ValueError(
                "device_adaptation.addressable.index must provide its concrete addressable rectangle; got an absent or incompatible value"
            )
        leading = index[0]
        if not isinstance(leading, slice):
            raise ValueError(
                "device_adaptation.addressable.leading must provide a leading slice; got an absent or incompatible value"
            )
        first = 0 if leading.start is None else leading.start
        last = states.mesh.cell_ids.shape[0] if leading.stop is None else leading.stop
        return np.stack(tuple(packets[part] for part in range(first, last)))[index[1:]]

    key_shape = (*states.mesh.cell_ids.shape, 33)
    placed = jax.make_array_from_callback(
        key_shape, states.mesh.cell_ids.sharding, addressable
    )
    flat = placed.reshape((-1, 33))
    active = states.mesh.cell_active.reshape(-1)
    unique, _, count = canonical_rows(flat, active)
    host_unique = _replicated_host_rows(unique[:count])
    root_names = tuple(name for name, _, _ in roots)
    root_kinds = tuple(kind for _, kind, _ in roots)
    root_ids = tuple(identifier for _, _, identifier in roots)
    records = []
    for row in host_unique:
        root = int(row[0])
        if root < 0 or root >= len(roots):
            raise ValueError(
                "device_adaptation.logical_presentation.root must name an existing source block; got a presentation key names an absent source block"
            )
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
    positions = key_positions(table_array, flat)
    slot_groups = jnp.where(positions >= 0, group_by_row[jnp.maximum(positions, 0)], -1)
    groups = slot_groups[cell_order]
    prefix = jnp.arange(cell_keys.shape[0]) < cells
    counts = tuple(
        int(jax.device_get(jnp.sum(prefix & (groups == group), dtype=jnp.int64)))
        for group in range(len(names))
    )
    if any(group_count <= 0 for group_count in counts) or sum(counts) != cells:
        raise ValueError(
            "device_adaptation.logical_presentation.any must retain an active target cell; got missing an active target cell"
        )
    within = cell_keys[:, 0]
    if isinstance(premise, _dependency_phydrax_meshing__result.CellMeshingResult):
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
                    "device_adaptation.logical_presentation.member must exclude a foreign source cell; got an unchanged presentation block contains a foreign source cell"
                )
            rows = jnp.asarray(order, dtype=jnp.int64)[safe]
            within = jnp.where(member, rows, within)
    _, presentation_order, ordered_count = canonical_rows(
        jnp.stack((groups, within), axis=1), prefix
    )
    if ordered_count != cells:
        raise ValueError(
            "device_adaptation.logical_presentation.ordered_count must retain a target cell; got missing a target cell"
        )
    return _LogicalPresentation(names, kinds, groups, counts, presentation_order)


@jax.jit
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
    facet_rows = key_positions(facet_table, facet_keys)
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
        rows = key_positions(table, keys)
        membership = (
            jnp.zeros((table.shape[0],), dtype=jnp.int32)
            .at[jnp.maximum(rows, 0)]
            .add(valid.astype(jnp.int32))
        )
        masks.append((jnp.arange(table.shape[0]) < counts[degree]) & (membership > 0))
    masks.append(facet_boundary)
    masks.append(jnp.zeros((cell_rows.shape[0],), dtype=jnp.bool_))
    return tuple(masks)


@jax.jit
def _first_appearance_values(
    keys: jax.Array,
    valid: jax.Array,
    table: jax.Array,
    count: int,
    /,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rows = key_positions(jnp.where(table >= 0, table, jnp.iinfo(jnp.int64).max), keys)
    absent = jnp.any(valid & (rows < 0))
    sentinel = keys.shape[0]
    first = (
        jnp.full((table.shape[0],), sentinel, dtype=jnp.int64)
        .at[jnp.maximum(rows, 0)]
        .min(jnp.where(valid, jnp.arange(keys.shape[0]), sentinel))
    )
    order = jnp.argsort(first, stable=True)
    missing = jnp.any((jnp.arange(table.shape[0]) < count) & (first[order] == sentinel))
    return order, absent, missing


def _first_appearance_order(
    keys: jax.Array,
    valid: jax.Array,
    table: jax.Array,
    count: int,
    /,
) -> jax.Array:
    order, absent, missing = _first_appearance_values(keys, valid, table, count)
    if bool(jax.device_get(absent)):
        raise ValueError(
            "device_adaptation.first_appearance_order.valid must reference an existing entity key; got a logical simplex route references an absent entity key"
        )
    if bool(jax.device_get(missing)):
        raise ValueError(
            "device_adaptation.first_appearance_order.sentinel must retain a logical simplex entity; got missing a logical simplex entity"
        )
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
            "device_adaptation.logical_entity_orders.dimension must have topological dimension two or three; got distributed simplex publication supports dimensions two and three"
        )
    orders.append(jnp.arange(top_rows.shape[0], dtype=jnp.int64))
    return tuple(orders)


@jax.jit
def _multiblock_block_values(
    cell_keys: jax.Array,
    cell_rows: jax.Array,
    groups: jax.Array,
    group: int,
    cell_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    selected = jnp.nonzero(
        (jnp.arange(cell_keys.shape[0]) < cell_count) & (groups == group),
        size=cell_keys.shape[0],
        fill_value=0,
    )[0]
    rows = cell_rows[selected]
    return cell_keys[selected, 0], rows, jnp.ones(rows.shape, dtype=jnp.bool_)


@jax.jit
def _multiblock_top_values(
    cell_keys: jax.Array, cell_rows: jax.Array, order: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return cell_keys[order, 0], cell_rows[order]


@jax.jit
def _multiblock_entity_values(
    top_ids: jax.Array,
    entity_ids: tuple[jax.Array, ...],
    orders: tuple[jax.Array, ...],
    masks: tuple[jax.Array, ...],
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    dimension = len(entity_ids) - 1
    identities = tuple(
        top_ids if degree == dimension else values[orders[degree]]
        for degree, values in enumerate(entity_ids)
    )
    boundary = tuple(
        mask if degree == dimension else mask[orders[degree]]
        for degree, mask in enumerate(masks)
    )
    active = tuple(jnp.ones(values.shape, dtype=jnp.bool_) for values in identities)
    return identities, boundary, active


@partial(jax.jit, static_argnames=("top_entity",))
def _multiblock_incidence_values(
    upper_keys: jax.Array,
    upper_ids: jax.Array,
    lower_keys: jax.Array,
    lower_ids: jax.Array,
    top_rows: jax.Array,
    upper_count: int,
    *,
    top_entity: bool,
) -> tuple[jax.Array, jax.Array]:
    degree = upper_keys.shape[1] - 1
    positions = np.arange(degree + 1, dtype=np.int32)
    columns = jnp.asarray(
        np.stack([positions[positions != removed] for removed in range(degree + 1)]),
        dtype=jnp.int32,
    )
    faces = upper_keys[:, columns].reshape((-1, degree))
    lower_table = jnp.where(lower_keys >= 0, lower_keys, jnp.iinfo(jnp.int64).max)
    rows = key_positions(lower_table, faces)
    valid = jnp.repeat(jnp.arange(upper_keys.shape[0]) < upper_count, degree + 1)
    absent = jnp.any(valid & (rows < 0))
    signs = jnp.broadcast_to(
        jnp.asarray(np.where(positions % 2, -1, 1), dtype=jnp.int64),
        (upper_keys.shape[0], degree + 1),
    )
    if top_entity:
        inversions = jnp.zeros((top_rows.shape[0],), dtype=jnp.int64)
        for first, second in combinations(range(degree + 1), 2):
            inversions += top_rows[:, first] > top_rows[:, second]
        signs = signs * jnp.where(inversions % 2, -1, 1)[:, None]
    incidence = jnp.stack(
        (
            lower_ids[jnp.maximum(rows, 0)],
            jnp.repeat(upper_ids, degree + 1),
            signs.reshape(-1),
        ),
        axis=1,
    )
    padded = jnp.where(valid[:, None], incidence, jnp.iinfo(jnp.int64).max)
    order = jnp.lexsort(padded.T[::-1])
    return padded[order], absent


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
    """Hash multiblock topology arrays prepared by fixed-capacity numerical joins."""
    canonical_blocks = []
    for group, (name, kind, count) in enumerate(
        zip(
            presentation.names,
            presentation.kinds,
            presentation.counts,
            strict=True,
        )
    ):
        block_ids, block_rows, valid = _multiblock_block_values(
            cell_keys, cell_rows, presentation.groups, group, counts[-1]
        )
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
    top_ids, top_rows = _multiblock_top_values(cell_keys, cell_rows, presentation.order)
    canonical_boundary_masks = _logical_boundary_masks(
        cell_rows, counts[-1], entity_tables, counts
    )
    entity_orders = _logical_entity_orders(top_rows, counts[-1], entity_tables, counts)
    logical_entity_ids, boundary_masks, active_masks = _multiblock_entity_values(
        top_ids, entity_ids, entity_orders, canonical_boundary_masks
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
        active = active_masks[degree]
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
        ordered, absent = _multiblock_incidence_values(
            upper_keys,
            upper_ids,
            entity_tables[degree - 1],
            lower_ids,
            top_rows,
            counts[degree],
            top_entity=degree == dimension,
        )
        if bool(jax.device_get(absent)):
            raise ValueError(
                "device_adaptation.logical_multiblock_topology_id.valid must be face closed; got logical simplex incidence is not face closed"
            )
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


@final
class _LogicalTopologyCompaction(StrictModule):
    """Device-ordered active topology, shared-vertex ownership, and coordinate banks."""

    mesh: _dependency_phydrax_discretization__adaptive_simplex.MaskedSimplexMesh
    parts: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    cell_keys: jax.Array
    cell_order: jax.Array
    cells: int = eqx.field(static=True)
    cell_rows: jax.Array
    cell_owners: jax.Array
    solver_cell_owners: jax.Array
    vertex_keys: jax.Array
    vertices: int = eqx.field(static=True)
    vertex_owners: jax.Array
    coordinates: jax.Array


def _compact_logical_topology(
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    owners: jax.Array,
    publication_cell_owners: jax.Array | None,
) -> _LogicalTopologyCompaction:
    """Device-ordered active topology, shared-vertex ownership, and coordinate banks."""
    mesh = states.mesh
    (
        cell_keys,
        cell_order,
        cells,
        cell_rows,
        cell_owners,
        solver_cell_owners,
        vertex_keys,
        vertices,
        vertex_owners,
        coordinates,
    ) = _compact_logical_values(mesh, owners, publication_cell_owners)
    return _LogicalTopologyCompaction(
        mesh,
        mesh.cells.shape[0],
        mesh.cells.shape[2],
        cell_keys,
        cell_order,
        int(jax.device_get(cells)),
        cell_rows,
        cell_owners,
        solver_cell_owners,
        vertex_keys,
        int(jax.device_get(vertices)),
        vertex_owners,
        coordinates,
    )


@jax.jit
def _compact_logical_values(
    mesh: _dependency_phydrax_discretization__adaptive_simplex.MaskedSimplexMesh,
    owners: jax.Array,
    publication_cell_owners: jax.Array | None,
) -> tuple[jax.Array, ...]:
    parts = mesh.cells.shape[0]
    width = mesh.cells.shape[2]
    part_vertices = jnp.take_along_axis(
        mesh.vertex_ids, mesh.cells.reshape((parts, -1)), axis=1
    )
    cell_vertices = part_vertices.reshape((-1, width))
    cell_keys, cell_order, cells = _canonical_row_values(
        mesh.cell_ids.reshape((-1, 1)), mesh.cell_active.reshape((-1,))
    )
    cell_rows = cell_vertices[cell_order]
    physical_owners = jnp.broadcast_to(
        jnp.arange(parts, dtype=jnp.int32)[:, None], mesh.cell_ids.shape
    )
    publication_owners = (
        physical_owners if publication_cell_owners is None else publication_cell_owners
    )
    cell_owners = publication_owners.reshape((-1,))[cell_order]
    solver_cell_owners = owners.reshape((-1,))[cell_order]
    used_vertex_ids = jnp.where(
        jnp.repeat(mesh.cell_active, width, axis=1),
        cell_vertices.reshape(mesh.cell_ids.shape[0], -1),
        -1,
    ).reshape(-1)
    vertex_valid = mesh.vertex_active.reshape(-1) & jnp.isin(
        mesh.vertex_ids.reshape(-1), used_vertex_ids
    )
    vertex_keys, vertex_order, vertices = _canonical_row_values(
        mesh.vertex_ids.reshape((-1, 1)), vertex_valid
    )
    vertex_positions = key_positions(
        jnp.where(vertex_keys >= 0, vertex_keys, jnp.iinfo(jnp.int64).max),
        cell_vertices.reshape((-1, 1)),
    ).reshape((-1, width))
    incident_owners = jnp.where(
        mesh.cell_active.reshape((-1, 1)),
        publication_owners.reshape((-1, 1)),
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
    return (
        cell_keys,
        cell_order,
        cells,
        cell_rows,
        cell_owners,
        solver_cell_owners,
        vertex_keys,
        vertices,
        vertex_owners,
        coordinates,
    )


@final
class _LogicalPublicationIdentity(StrictModule):
    """Topology and geometry fingerprints using the serial identity payloads and reduction order."""

    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)


def _logical_publication_identity(
    accepted_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    source: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    arrays: dict[str, jax.Array],
    shapes: dict[str, tuple[int, ...]],
    presentation: _LogicalPresentation,
    vertex_keys: jax.Array,
    cell_keys: jax.Array,
    cell_rows: jax.Array,
    entity_tables: list[jax.Array],
    entity_ids: list[jax.Array],
    counts: list[int],
    numeric_version: str | None,
    vertices: int,
    coordinates: jax.Array,
) -> _LogicalPublicationIdentity:
    """Topology and geometry fingerprints using the serial identity payloads and reduction order."""
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
    return _LogicalPublicationIdentity(topology_id, geometry_id)


@checked
def logical_publication(
    accepted_source: _dependency_phydrax_meshing__result.CellMeshingResult,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    owners: jax.Array,
    source_exterior: jax.Array,
    layout: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexLayout,
    /,
    *,
    placement_arrays: tuple[tuple[str, jax.Array], ...] = (),
    original_source: _dependency_phydrax_meshing__result.CellMeshingResult | None,
    uniform_refinement: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    | None = None,
    numeric_version: str | None = None,
    publication_cell_owners: jax.Array | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _LogicalPublication:
    """Order and compact numerical global arrays on devices, not on a host."""

    phase = _compact_logical_topology(
        states=states, owners=owners, publication_cell_owners=publication_cell_owners
    )
    mesh = phase.mesh
    parts = phase.parts
    width = phase.width
    cell_keys = phase.cell_keys
    cell_order = phase.cell_order
    cells = phase.cells
    cell_rows = phase.cell_rows
    cell_owners = phase.cell_owners
    solver_cell_owners = phase.solver_cell_owners
    vertex_keys = phase.vertex_keys
    vertices = phase.vertices
    vertex_owners = phase.vertex_owners
    coordinates = phase.coordinates
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
    premise = _dependency_phydrax_meshing__result.require_original_meshing_source(
        accepted_source if original_source is None else original_source
    )
    source = accepted_source.mesh
    for degree in range(1, width - 1):
        columns = np.asarray(
            tuple(combinations(range(width), degree + 1)), dtype=np.int32
        )
        keys = jnp.sort(cell_rows[:, columns], axis=2).reshape((-1, degree + 1))
        valid = jnp.repeat(jnp.arange(cell_keys.shape[0]) < cells, columns.shape[0])
        table, order, count = canonical_rows(keys, valid)
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
        scope=scope,
        budget=budget,
    )
    sharding = states.mesh.cell_ids.sharding
    arrays = {name: jax.device_put(value, sharding) for name, value in arrays.items()}
    phase = _logical_publication_identity(
        accepted_source=accepted_source,
        source=source,
        arrays=arrays,
        shapes=shapes,
        presentation=presentation,
        vertex_keys=vertex_keys,
        cell_keys=cell_keys,
        cell_rows=cell_rows,
        entity_tables=entity_tables,
        entity_ids=entity_ids,
        counts=counts,
        numeric_version=numeric_version,
        vertices=vertices,
        coordinates=coordinates,
    )
    topology_id = phase.topology_id
    geometry_id = phase.geometry_id
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
    arrays["presentation/cell_groups"] = jax.device_put(presentation.groups, sharding)
    arrays["presentation/cell_order"] = jax.device_put(presentation.order, sharding)
    arrays["epoch/solver_cell_owners"] = jax.device_put(owners, sharding)
    return _LogicalPublication(
        tuple(sorted(arrays.items())),
        tuple(counts),
        tuple(jax.device_put(value, sharding) for value in entity_tables),
        tuple(jax.device_put(value, sharding) for value in entity_ids),
        tuple(jax.device_put(value, sharding) for value in entity_owners),
        topology_id,
        geometry_id,
        presentation=presentation,
    )


def _collective_resource_failure(
    error: MeshcoreError
    | _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
    /,
) -> _leaf_phydrax_meshing__failures.MeshingFailure:
    match error:
        case MeshcoreError():
            category = (
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
                if error.status
                in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)
                else _leaf_phydrax_meshing__failures.MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
            )
        case _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError():
            category = (
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            )
        case value:
            assert_never(value)
    failure = _leaf_phydrax_meshing__failures.MeshingFailure(
        category, str(error), stage="distributed-device-commit"
    )
    failure.__cause__ = error
    return failure


@final
class _CollectiveGeometryRoots(StrictModule):
    """Source coefficient owners and accepted target rows for collective geometry lowering."""

    arrays: dict[str, jax.Array]
    target_ids: np.ndarray
    target_rows: np.ndarray
    roots: list[tuple[str, str, _dependency_phydrax_meshing__adaptation.FullP1Element]]
    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)


def _prepare_collective_geometry_roots(
    local: _LocalCommit,
    logical: _LogicalPublication,
    original: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectiveGeometryRoots:
    """Source coefficient owners and accepted target rows for collective geometry lowering."""
    if logical.geometry_projection is None:
        raise ValueError(
            "device_adaptation.prepare_collective_geometry_roots.logical must provide its consumed collective projection; got an absent or incompatible value"
        )
    arrays = dict(logical.geometry_projection.addressable_arrays(local.partition_index))
    edit = local.committed.edit
    fixed: list[_dependency_phydrax_meshing__topology_edit.TopologyEditBlock] = []
    for block in edit.blocks:
        if not isinstance(
            block, _dependency_phydrax_meshing__topology_edit.TopologyEditBlock
        ):
            raise ValueError(
                "device_adaptation.prepare_collective_geometry_roots.block must not lower a polyhedral topology edit; got a prohibited change"
            )
        fixed.append(block)
    target_ids = np.concatenate(tuple(block.cell_ids for block in fixed))
    target_rows = np.concatenate(tuple(block.cells for block in fixed))
    roots: list[
        tuple[str, str, _dependency_phydrax_meshing__adaptation.FullP1Element]
    ] = []
    if isinstance(original, _dependency_phydrax_meshing__result.CellMeshingResult):
        root_elements, _, _ = original.geometry.resolve(original.mesh, budget=budget)
        for block, element in zip(original.mesh.blocks, root_elements, strict=True):
            roots.append(
                (
                    block.name,
                    block.cell_kind,
                    _owner_phydrax_discretization__cell_geometry.require_full_p1_source(
                        element
                    ),
                )
            )
        source_geometry_id = (
            _owner_phydrax_discretization__cell_geometry_validity.cell_geometry_id(
                original.geometry
            )
        )
        source_topology_id = original.mesh.topology_id
    else:
        roots.append(
            (
                "surface",
                "triangle",
                _owner_phydrax_discretization__cell_geometry.coordinate_lagrange_element(
                    "triangle", 1
                ),
            )
        )
        source_geometry_id = original.coordinate_geometry_id
        source_topology_id = original.topology_id
    return _CollectiveGeometryRoots(
        arrays, target_ids, target_rows, roots, source_geometry_id, source_topology_id
    )


@final
class _CollectiveExactSource(StrictModule):
    """Restrict source coefficients while retaining geometry evidence."""

    exact_source: (
        _owner_phydrax_discretization__cell_geometry.ExactCellGeometrySource | None
    )


def _lower_collective_exact_source(
    original: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    coordinate_ids: np.ndarray,
    coordinates: np.ndarray,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectiveExactSource:
    """Restrict source coefficients while retaining geometry evidence."""
    exact_source = None
    if isinstance(original, _dependency_phydrax_meshing__result.CellMeshingResult):
        bisection_host.uniform_count(
            coordinate_ids.size, coordinates.nbytes + coordinate_ids.nbytes, budget=budget
        )
        if np.any(coordinate_ids < 0) or np.any(
            coordinate_ids >= original.geometry.coordinates.shape[0]
        ):
            raise ValueError(
                "device_adaptation.lower_collective_exact_source.coordinate_ids must exist in the source coefficient bank; got a collective coefficient sci is absent from its source bank"
            )
        if not np.array_equal(
            coordinates, np.asarray(original.geometry.coordinates)[coordinate_ids]
        ):
            raise ValueError(
                "device_adaptation.lower_collective_exact_source.coordinates must match its SCI routes; got mismatched a collective physical coefficient bank"
            )
        source = original.geometry.exact_source
        match source:
            case _owner_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource():
                bisection_host.uniform_count(
                    coordinate_ids.size, coordinate_ids.size * 32, budget=budget
                )
                exact_source = source.with_witnesses(
                    source.vertex_strata[coordinate_ids],
                    source.vertex_rows[coordinate_ids],
                    source.vertex_parameters[coordinate_ids],
                    vertex_parameter_ratios=(
                        None
                        if source.vertex_parameter_ratios is None
                        else tuple(
                            source.vertex_parameter_ratios[index]
                            for index in coordinate_ids.tolist()
                        )
                    ),
                    budget=budget,
                )
            case (
                _owner_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometrySource()
                | _owner_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource()
                | _owner_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryLinearActionSource()
            ):
                bisection_host.uniform_count(
                    coordinate_ids.size, coordinate_ids.size * 24, budget=budget
                )
                exact_source = _owner_phydrax_discretization__exact_power_geometry.ExactPowerCellGeometryRestrictionSource(
                    source,
                    np.zeros((0, 4), dtype=np.float64),
                    np.stack(
                        (
                            coordinate_ids,
                            np.full(coordinate_ids.shape, -1, dtype=np.int64),
                        ),
                        axis=1,
                    ),
                    np.full(coordinate_ids.shape, -1, dtype=np.int64),
                )
            case _:
                pass
    return _CollectiveExactSource(exact_source)


@final
class _CollectiveCoefficientActions(StrictModule):
    """Full coefficient action stacks grouped by source identity and presentation element."""

    elements: dict[str, _dependency_phydrax_meshing__adaptation.FullP1Element]
    rows: dict[str, list[np.ndarray]]
    ids: dict[str, list[np.int64]]
    routes: dict[str, list[np.ndarray]]
    parent_cells: dict[str, list[np.int64]]
    parent_vertices: dict[str, list[np.ndarray]]
    source_banks: dict[str, str]
    kinds: dict[str, str]


def _lower_collective_coefficient_actions(
    roots: list[tuple[str, str, _dependency_phydrax_meshing__adaptation.FullP1Element]],
    arrays: dict[str, jax.Array],
    target_ids: np.ndarray,
    target_rows: np.ndarray,
    uniform_refinement: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    | None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CollectiveCoefficientActions:
    """Full coefficient action stacks grouped by source identity and presentation element."""
    elements = {}
    rows = {}
    ids = {}
    routes = {}
    parent_cells = {}
    parent_vertices = {}
    source_banks = {}
    kinds = {}
    uniform_actions = (
        None
        if uniform_refinement is None
        else np.asarray(uniform_refinement.barycentric_weights, dtype=np.float64)
    )
    uniform_parents = (
        None
        if uniform_refinement is None
        else np.asarray(uniform_refinement.parent_ids, dtype=np.int64)
    )
    for root_name, root_kind, root_element in roots:
        table = arrays[f"geometry/cell_ids/{root_name}"]
        table = jnp.where(table >= 0, table, jnp.iinfo(jnp.int64).max)
        positions = key_positions(
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
                "device_adaptation.lower_collective_coefficient_actions.actions must retain their source columns or declared depth; got missing their source columns or declared depth"
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
            stack = actions[index, : action_counts[index]]
            uniform = False
            if uniform_actions is not None and uniform_parents is not None and stack.size:
                parent_rows = np.flatnonzero(uniform_parents == parents[index])
                if parent_rows.size != 1:
                    raise ValueError(
                        "presentation.uniform_parent must bind one retained source cell; "
                        f"got {parent_rows.size} matches for {parents[index]}"
                    )
                uniform = any(
                    np.array_equal(stack[-1], action)
                    for action in uniform_actions[parent_rows[0]]
                )
            element = _composed_presentation_element(
                root_element,
                tuple(stack),
                packed=uniform_refinement is not None,
                uniform=uniform,
                scope=scope,
                budget=budget,
            )
            name = (
                root_name
                if element.element_id == root_element.element_id
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
        raise ValueError(
            "device_adaptation.lower_collective_coefficient_actions.sum must bind one accepted source chart per local cell; got a local cell has no unique accepted scientific source chart"
        )
    return _CollectiveCoefficientActions(
        elements, rows, ids, routes, parent_cells, parent_vertices, source_banks, kinds
    )


def _lower_collective_geometry(
    local: _LocalCommit,
    logical: _LogicalPublication,
    original: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    uniform_refinement: _dependency_phydrax_meshing__bisection.BisectionUniformRefinement
    | None,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    tuple[CellBlock, ...],
    _owner_phydrax_discretization__cell_geometry.CellGeometrySpec,
    np.ndarray,
    np.ndarray,
    dict[str, str],
    np.ndarray,
]:
    """Lower only this closure's source coefficients and reference charts."""

    phase = _prepare_collective_geometry_roots(
        local=local, logical=logical, original=original, budget=budget
    )
    arrays = phase.arrays
    target_ids = phase.target_ids
    target_rows = phase.target_rows
    roots = phase.roots
    source_geometry_id = phase.source_geometry_id
    source_topology_id = phase.source_topology_id
    phase = _lower_collective_coefficient_actions(
        roots=roots,
        arrays=arrays,
        target_ids=target_ids,
        target_rows=target_rows,
        uniform_refinement=uniform_refinement,
        scope=scope,
        budget=budget,
    )
    elements = phase.elements
    rows = phase.rows
    ids = phase.ids
    routes = phase.routes
    parent_cells = phase.parent_cells
    parent_vertices = phase.parent_vertices
    source_banks = phase.source_banks
    kinds = phase.kinds
    # block order matches the serial commit owner (sorted by name), so
    # one logical target has one topology and geometry identity on every route.
    names = tuple(sorted(elements))
    if isinstance(original, _dependency_phydrax_meshing__result.CellMeshingResult):
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
            rows[block.name] = [rows[block.name][index] for index in order]
            ids[block.name] = [ids[block.name][index] for index in order]
            routes[block.name] = [routes[block.name][index] for index in order]
            parent_cells[block.name] = [
                parent_cells[block.name][index] for index in order
            ]
            parent_vertices[block.name] = [
                parent_vertices[block.name][index] for index in order
            ]
    coordinate_ids = np.unique(
        np.concatenate(tuple(np.asarray(routes[name]).reshape(-1) for name in names))
    )
    coordinate_table = arrays["geometry/coordinate_ids"]
    positions = key_positions(
        jnp.where(coordinate_table >= 0, coordinate_table, jnp.iinfo(jnp.int64).max)[
            :, None
        ],
        jnp.asarray(coordinate_ids[:, None], dtype=jnp.int64),
    )
    if bool(jax.device_get(jnp.any(positions < 0))):
        raise ValueError(
            "device_adaptation.lower_collective_geometry.positions must reference existing coordinate DOFs; got a scientific source chart references an absent coordinate dof"
        )
    coordinates = np.asarray(
        jax.device_get(arrays["geometry/coordinates"][positions]), dtype=np.float64
    )
    owners = np.asarray(
        jax.device_get(arrays["geometry/coordinate_owners"][positions]), dtype=np.int32
    )
    phase = _lower_collective_exact_source(
        original=original,
        coordinate_ids=coordinate_ids,
        coordinates=coordinates,
        budget=budget,
    )
    exact_source = phase.exact_source
    blocks = tuple(
        CellBlock(
            name,
            kinds[name],
            np.asarray(rows[name], dtype=np.int32),
            global_ids=np.asarray(ids[name], dtype=np.int64),
        )
        for name in names
    )
    origin = _owner_phydrax_discretization__cell_geometry.CellGeometryRestrictionSource(
        source_geometry_id,
        source_topology_id,
        {name: np.asarray(parent_cells[name], dtype=np.int64) for name in names},
        {name: np.asarray(parent_vertices[name], dtype=np.int64) for name in names},
        block_source_blocks=source_banks,
    )
    geometry = _owner_phydrax_discretization__cell_geometry.CellGeometrySpec(
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
        budget=budget,
    )
    order = entity_keys_owner.key_rows(
        target_ids[:, None],
        np.concatenate(tuple(np.asarray(block.global_ids) for block in blocks))[:, None],
    )
    return blocks, geometry, coordinate_ids, owners, source_banks, order


def _published_cell_mesh(
    local: _LocalCommit,
    logical: _LogicalPublication,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    partitioned: PartitionedAdaptiveSimplex,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_discretization__cell_mesh.CellMesh:
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
    ) = _lower_collective_geometry(
        local, logical, original, evidence.uniform_refinement, scope=scope, budget=budget
    )
    probe = _dependency_phydrax_discretization__cell_mesh.CellMesh(
        edit.coordinates, blocks, vertex_global_ids=edit.vertex_global_ids
    )
    projection = logical.publication_projection
    if projection is None:
        raise ValueError(
            "device_adaptation.published_cell_mesh.projection must provide its consumed fixed-capacity entity projection; got an absent or incompatible value"
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
        queries = entity_keys_owner.entity_keys(probe, degree)
        positions = entity_keys_owner.key_rows(keys, queries)
        if np.any(positions < 0):
            raise ValueError(
                "device_adaptation.published_cell_mesh.positions must exist in the consumed publication projection; got a local closure entity is absent from its consumed publication projection"
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
                    "device_adaptation.published_cell_mesh.coordinates must match their consumed logical geometry projection; got mismatched local coordinates"
                )
    if local.cell_exterior is None:
        raise ValueError(
            "device_adaptation.published_cell_mesh.local must provide consumed neighborhood exterior witnesses; got an absent or incompatible value"
        )
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
    facet_rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(probe, probe.topological_dimension - 1),
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
    storage = _dependency_phydrax_discretization__cell_mesh.CellMeshStorage(
        logical.counts,
        ids,
        owners,
        partition_index=local.partition_index,
        partition_count=partitioned.parts.part_count,
        local_coordinates=edit.coordinates,
        local_blocks=blocks,
        global_coordinate_count=(
            original.geometry.coordinates.shape[0]
            if isinstance(original, _dependency_phydrax_meshing__result.CellMeshingResult)
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
    return _dependency_phydrax_discretization__cell_mesh.CellMesh(
        edit.coordinates,
        blocks,
        storage=storage,
        numeric_version=f"adaptation:{partitioned.prepared.adaptation.prepared_id}",
    )


def _local_entity_lineage(
    partitioned: PartitionedAdaptiveSimplex,
    local: _LocalCommit,
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    relation: _dependency_phydrax_meshing__topology_edit.EntityRelations,
    /,
) -> _dependency_phydrax_meshing__lineage_records.EntityLineage:
    """Shard owned creation/deletion without declaring absent ghost rows deleted."""

    record = _dependency_phydrax_meshing__topology_edit.entity_lineage(
        local.source, mesh, relation
    )
    degree = record.dimension
    deleted = np.asarray(record.deleted_source_ids, dtype=np.int64)
    if deleted.size:
        keys = entity_keys_owner.entity_keys(local.source, degree)
        local_ids = np.asarray(local.source.entity_set(degree).entity_ids, dtype=np.int64)
        rows = entity_keys_owner.key_rows(local_ids[:, None], deleted[:, None])
        _, source_owner = _source_entity_values(
            partitioned.source_entities, degree, keys[rows]
        )
        deleted = deleted[source_owner == local.partition_index]
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "device_adaptation.local_entity_lineage.storage must provide accepted owner-local storage; got an absent or incompatible value"
        )
    target_ids = np.asarray(storage.entity_global_ids[degree], dtype=np.int64)
    target_owned = np.asarray(storage.entity_owned[degree], dtype=np.bool_)
    created = np.asarray(record.created_target_ids, dtype=np.int64)
    created = created[np.isin(created, target_ids[target_owned])]
    return _dependency_phydrax_meshing__lineage_records.EntityLineage(
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
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec


def _local_target(
    partitioned: PartitionedAdaptiveSimplex,
    local: _LocalCommit,
    logical: _LogicalPublication,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _LocalTarget:
    """Lower one owner-local target carrier and its lineage."""
    prepared = partitioned.prepared.adaptation
    mesh = _published_cell_mesh(
        local, logical, evidence, partitioned, scope=scope, budget=budget
    )
    lineage = _dependency_phydrax_meshing__lineage_records.MeshLineage(
        prepared.source.mesh.topology_id,
        mesh.topology_id,
        tuple(
            _local_entity_lineage(partitioned, local, mesh, relation)
            for relation in local.committed.edit.relations
        ),
    )
    if mesh.storage is None:
        raise ValueError(
            "device_adaptation.local_target.mesh must retain its coordinate-map storage; got missing its coordinate-map storage"
        )
    return _LocalTarget(
        local, mesh, lineage, mesh.storage.restore_geometry(budget=budget)
    )


def _collective_plc_premises(
    partitioned: PartitionedAdaptiveSimplex,
    target: _LocalTarget,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> (
    tuple[
        global_embedding_owner.GlobalEmbeddingCertificate,
        _dependency_phydrax_geometry__mesh_certificates.DomainCoverageCertificate,
    ]
    | None
):
    """New-target embedding and domain coverage for represented-PLC transfer.

    Collective: the selector depends only on the shared policy and source, so
    every owner enters the producer together. Other transfer routes do not consume these certificates.
    """
    prepared = partitioned.prepared.adaptation
    if not prepared.source.associations or not isinstance(
        prepared.policy.association_transfer,
        _dependency_phydrax_meshing__association.PlcAssociationTransfer,
    ):
        return None
    original = _scientific_preparation_source(prepared)
    if (
        not isinstance(original, _dependency_phydrax_meshing__result.CellMeshingResult)
        or original.certification is None
    ):
        raise ValueError(
            "device_adaptation.collective_plc_premises.source must provide the serial embedding and domain-coverage certificates; got an absent or incompatible value"
        )
    certification = original.certification
    domain = certification.request.domain
    if (
        certification.embedding is None
        or certification.coverage is None
        or not isinstance(
            domain, _dependency_phydrax_geometry__mesh_certificates.PiecewiseLinearDomain
        )
        or certification.request.cell_regions is None
    ):
        raise ValueError(
            "device_adaptation.collective_plc_premises.certification must provide the serial embedding and domain-coverage certificates; got an absent or incompatible value"
        )
    storage = target.mesh.storage
    if storage is None:
        raise ValueError(
            "device_adaptation.collective_plc_premises.storage must retain its coordinate-map storage; got missing its coordinate-map storage"
        )
    _, states, _ = restore_partitioned_mesh_state(storage, evidence)
    _, embedding, coverage = (
        _dependency_phydrax_geometry__collective_domain_coverage.certify_collective_premises(
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
            budget=budget,
        )
    )
    return embedding, coverage


@final
class _CertifiedLocalTarget(StrictModule):
    """Certified owner-local target with organization, associations, and renewed region evidence."""

    target: _dependency_phydrax_meshing__result.CellMeshingResult


def _certify_local_adaptation_target(
    prepared: _dependency_phydrax_meshing__adaptation.PreparedMeshAdaptation,
    original: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    mesh: _dependency_phydrax_discretization__cell_mesh.CellMesh,
    lineage: _dependency_phydrax_meshing__lineage_records.MeshLineage,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
    logical: _LogicalPublication,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    premises: tuple[
        global_embedding_owner.GlobalEmbeddingCertificate,
        _dependency_phydrax_geometry__mesh_certificates.DomainCoverageCertificate,
    ]
    | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CertifiedLocalTarget:
    """Certified owner-local target with organization, associations, and renewed region evidence."""
    patches, zones, labels, attributes = (
        _dependency_phydrax_meshing__organization.lower_mesh_organization(
            original,
            mesh,
            dict(logical.arrays),
            scope_projections=logical.scope_projections,
            attribute_projections=logical.attribute_projections,
        )
    )
    region_boundaries = (
        _dependency_phydrax_meshing__adaptation.region_boundary_transition(
            prepared.source, mesh, lineage, patches, zones, labels
        )
    )
    region_evidence = None
    if prepared.source.region_evidence is not None:
        renewal = _owner_phydrax_meshing__compartments.revalidate_region_evidence(
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
        match transfer:
            case _dependency_phydrax_meshing__association.PlcAssociationTransfer():
                if premises is None:
                    raise ValueError(
                        "device_adaptation.certify_local_adaptation_target.premises must provide its collective target premises; got an absent or incompatible value"
                    )
                associations = transfer.propagate(
                    prepared.source,
                    lineage,
                    mesh,
                    geometry=geometry,
                    embedding=premises[0],
                    coverage=premises[1],
                )
            case _dependency_phydrax_meshing__association.BRepAssociationTransfer():
                associations = transfer.propagate(prepared.source, lineage, mesh)
            case _dependency_phydrax_meshing__association.SurfaceAssociationTransfer():
                if logical.surface_transfer is None:
                    raise ValueError(
                        "device_adaptation.certify_local_adaptation_target.logical must provide its collective source receipts; got an absent or incompatible value"
                    )
                associations = logical.surface_transfer.propagate(
                    prepared.source, lineage, mesh, geometry=geometry
                )
            case _:
                raise ValueError(
                    "device_adaptation.certify_local_adaptation_target.prepared must provide its association transfer; got an absent or incompatible value"
                )
    target = _dependency_phydrax_meshing__canonical.certify_owner_local_cell_mesh(
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
        budget=budget,
    )
    return _CertifiedLocalTarget(target)


def _local_adaptation_result(
    partitioned: PartitionedAdaptiveSimplex,
    staged: _LocalTarget,
    premises: tuple[
        global_embedding_owner.GlobalEmbeddingCertificate,
        _dependency_phydrax_geometry__mesh_certificates.DomainCoverageCertificate,
    ]
    | None,
    logical: _LogicalPublication,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    started: float,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__adaptation.MeshAdaptationResult:
    """Bind one certified local publication to the existing transition owners."""

    prepared = partitioned.prepared.adaptation
    local, mesh, lineage, geometry = staged
    committed = local.committed
    edit = committed.edit
    original = _scientific_preparation_source(prepared)
    phase = _certify_local_adaptation_target(
        prepared=prepared,
        original=original,
        mesh=mesh,
        lineage=lineage,
        geometry=geometry,
        logical=logical,
        evidence=evidence,
        premises=premises,
        budget=budget,
    )
    target = phase.target
    stencil = _dependency_phydrax_meshing__lineage_records.VertexInterpolationStencil(
        prepared.source.mesh.entity_set(0).entity_set_id,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
        edit.stencil_sources,
        edit.stencil_weights,
        edit.stencil_valid,
        preserves_constants=True,
    )
    kind = (
        _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REMESH
        if committed.refined and committed.coarsened
        else _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.REFINE
        if committed.refined
        else _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.COARSEN
        if committed.coarsened
        else _dependency_phydrax_meshing__lineage_records.MeshTransitionKind.PARTITION
    )
    transition = _dependency_phydrax_meshing__lineage.CellMeshTransition(
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
        _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.PASS_LIMIT
        if committed.pass_limited
        else _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.PARTIAL
        if committed.partial
        else _dependency_phydrax_meshing__adaptation.MeshAdaptationStatus.COMPLETE
    )

    hierarchy = bisection_host.rebind_bisection_presentation_blocks(
        committed.hierarchy, target, scope=scope, budget=budget
    )
    outcome = _dependency_phydrax_meshing__adaptation.RouteOutcome(
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
        raise ValueError(
            "device_adaptation.local_adaptation_result.source_distribution must provide the prepared distribution policy; got an absent or incompatible value"
        )
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
    source_rows, defined = _dependency_phydrax_meshing__scope.local_logical_lookup(
        source_table[source_order],
        source_order,
        np.asarray(cells.source_global_ids, dtype=np.int64),
    )
    if not np.all(defined):
        raise ValueError(
            "device_adaptation.local_adaptation_result.defined must exist in the accepted predecessor; got a cell lineage occurrence is absent from the accepted predecessor"
        )
    distribution = _dependency_phydrax_meshing__distribution.prepare_owner_local_distribution_transition(
        source_distribution,
        mesh_part_owner.MeshPart(source_distribution.part.name, target),
        lineage,
        source_rows,
        _dependency_phydrax_meshing__distribution.rows_of(
            np.asarray(mesh.entity_set(mesh.topological_dimension).entity_ids),
            cells.target_global_ids,
        ),
        source_ids,
        source_owners,
        policy=partition_policy,
        restart_repack=partitioned.restart_repack,
        restart_proof=logical.restart_proof,
    )
    return _dependency_phydrax_meshing__adaptation.MeshAdaptationResult(
        prepared,
        outcome,
        _dependency_phydrax_meshing__adaptation.compliance(prepared, outcome, elapsed),
        distribution,
        elapsed,
    )


def host_process_status_union(statuses: jax.Array, /) -> int:
    """Union integer status bits using backend-supported scalar reductions."""

    shifts = jnp.arange(8, dtype=jnp.int32)
    bits = (statuses[:, None] >> shifts) & 1
    present = jnp.sum(bits, axis=0, dtype=jnp.int32) > 0
    flags = jnp.sum(present.astype(jnp.int32) << shifts, dtype=jnp.int32)
    return int(jax.device_get(flags))


def _collective_identity_rows(
    partitioned: PartitionedAdaptiveSimplex,
    packets: dict[int, np.ndarray],
    width: int,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> np.ndarray:
    """Merge fixed-capacity owner metadata, deduplicated in row order."""
    local_capacity = max((rows.shape[0] for rows in packets.values()), default=0)
    capacity = max(
        1,
        int(np.max(multihost_utils.process_allgather(np.int64(local_capacity)))),
    )
    shape = (partitioned.parts.part_count, capacity, width)
    bisection_host.uniform_count(
        prod(shape), prod(shape) * np.dtype(np.int64).itemsize * 3, budget=budget
    )
    padded = {
        part: np.pad(
            rows,
            ((0, capacity - rows.shape[0]), (0, 0)),
            constant_values=-1,
        )[None]
        for part, rows in packets.items()
    }

    def addressable(index: tuple[slice, ...] | None) -> np.ndarray:
        if index is None:
            raise ValueError(
                "collective_identity.index must provide an addressable rectangle; got None"
            )
        start = 0 if index[0].start is None else index[0].start
        stop = shape[0] if index[0].stop is None else index[0].stop
        return np.concatenate(tuple(padded[part] for part in range(start, stop)), axis=0)

    placed = jax.make_array_from_callback(
        shape, partitioned.states.mesh.cell_ids.sharding, addressable
    )
    rows = _replicated_host_rows(placed).reshape((-1, width))
    return np.unique(rows[rows[:, 0] >= 0], axis=0)


def _collective_entity_set_ids(
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence, /
) -> tuple[str, ...]:
    """Fingerprint the global entity sets without constructing a dense mesh."""
    dimension = len(evidence.global_entity_counts) - 1
    arrays = dict(evidence.logical_arrays)
    rows = arrays["cell_vertices"]
    counts = evidence.global_entity_counts
    masks = _logical_boundary_masks(rows, counts[-1], evidence.entity_keys, counts)
    presentation_order = arrays["presentation/cell_order"]
    orders = _logical_entity_orders(
        rows[presentation_order], counts[-1], evidence.entity_keys, counts
    )
    names = (
        ("vertices", "edges", "cells")
        if dimension == 2
        else ("vertices", "edges", "faces", "cells")
    )
    result = []
    for degree, (ids, mask, count) in enumerate(
        zip(evidence.entity_ids, masks, counts, strict=True)
    ):
        order = presentation_order if degree == dimension else orders[degree]
        ids, mask = ids[order], mask[order]
        subset = canonical_fingerprint(
            {
                "kind": "entity-subset",
                "name": "boundary",
                "mask": _logical_array_tree_fingerprint(mask, (count,)),
            }
        )
        result.append(
            canonical_fingerprint(
                {
                    "kind": "entity-set",
                    "name": names[degree],
                    "intrinsic_dimension": degree,
                    "entity_ids": _logical_array_tree_fingerprint(ids, (count,)),
                    "active_mask": _logical_array_tree_fingerprint(
                        jnp.ones(ids.shape, dtype=jnp.bool_), (count,)
                    ),
                    "subsets": [subset],
                }
            )
        )
    return tuple(result)


def _collective_adaptation_identity(
    partitioned: PartitionedAdaptiveSimplex,
    results: tuple[_dependency_phydrax_meshing__adaptation.MeshAdaptationResult, ...],
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__result.CollectiveMeshIdentity:
    """Bind owner-local lineage and forest records to the serial identity recipes."""
    packets: dict[int, _dependency_phydrax_meshing__lineage_records.MeshLineage] = {}
    hierarchies: dict[int, _dependency_phydrax_meshing__bisection.BisectionHierarchy] = {}
    block_routes: dict[int, np.ndarray] = {}
    for result in results:
        storage = result.target.mesh.storage
        lineage = result.lineage
        if storage is None or lineage is None:
            raise ValueError(
                "device_adaptation.partitioned_result must carry storage and lineage; got missing evidence"
            )
        match result.hierarchy:
            case _dependency_phydrax_meshing__bisection.BisectionHierarchy() as hierarchy:
                hierarchies[storage.partition_index] = hierarchy
            case _:
                raise ValueError(
                    "partitioned_result.hierarchy must be a bisection forest; got incompatible evidence"
                )
        packets[storage.partition_index] = lineage
        block_routes[storage.partition_index] = np.asarray(
            [
                evidence.presentation_names.index(block.name)
                for block in result.target.mesh.blocks
            ],
            dtype=np.int64,
        )
    target_sets = _collective_entity_set_ids(evidence)
    entities = []
    for degree in range(len(target_sets)):
        relations = {}
        creations = {}
        deletions = {}
        for part, local_lineage in packets.items():
            relation = local_lineage.entity_lineage(degree)
            relations[part] = np.stack(
                (
                    np.asarray(relation.source_global_ids, dtype=np.int64),
                    np.asarray(relation.target_global_ids, dtype=np.int64),
                    np.asarray(relation.relation_kinds, dtype=np.int64),
                ),
                axis=1,
            )
            creations[part] = np.asarray(relation.created_target_ids, dtype=np.int64)[
                :, None
            ]
            deletions[part] = np.asarray(relation.deleted_source_ids, dtype=np.int64)[
                :, None
            ]
        rows = _collective_identity_rows(
            partitioned, relations, 3, scope=scope, budget=budget
        )
        entities.append(
            _dependency_phydrax_meshing__lineage_records.EntityLineage(
                degree,
                partitioned.prepared.adaptation.source.mesh.entity_set(
                    degree
                ).entity_set_id,
                target_sets[degree],
                rows[:, 0],
                rows[:, 1],
                rows[:, 2],
                created_target_ids=_collective_identity_rows(
                    partitioned, creations, 1, scope=scope, budget=budget
                )[:, 0],
                deleted_source_ids=_collective_identity_rows(
                    partitioned, deletions, 1, scope=scope, budget=budget
                )[:, 0],
            )
        )
    lineage = _dependency_phydrax_meshing__lineage_records.MeshLineage(
        partitioned.prepared.adaptation.source.mesh.topology_id,
        evidence.topology_id,
        tuple(entities),
    )
    hierarchy = _collective_adaptation_hierarchy(
        partitioned, hierarchies, block_routes, scope=scope, budget=budget
    )
    return _dependency_phydrax_meshing__result.CollectiveMeshIdentity(
        evidence.topology_id, lineage.lineage_id, hierarchy.hierarchy_id
    )


def _collective_adaptation_hierarchy(
    partitioned: PartitionedAdaptiveSimplex,
    hierarchies: dict[int, _dependency_phydrax_meshing__bisection.BisectionHierarchy],
    block_routes: dict[int, np.ndarray],
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__bisection.BisectionHierarchy:
    dimension = partitioned.layout.dimension
    width = dimension + 1
    active = {}
    parents = {}
    scientific = {}
    for part, hierarchy in hierarchies.items():
        active[part] = np.column_stack(
            (
                hierarchy.cell_global_ids,
                hierarchy.ordered_vertices,
                hierarchy.tags,
                hierarchy.generations,
            )
        ).astype(np.int64)
        parents[part] = np.column_stack(
            (
                hierarchy.record_parent_ids,
                block_routes[part][
                    np.asarray(hierarchy.record_parent_blocks, dtype=np.int64)
                ],
                hierarchy.record_parent_rows,
                hierarchy.record_parent_vertices,
                hierarchy.record_parent_tags,
                hierarchy.record_child_ids,
                hierarchy.record_vertex_ids,
            )
        ).astype(np.int64)
        scientific[part] = np.column_stack(
            (
                hierarchy.scientific_cell_ids,
                hierarchy.scientific_block_ids,
            )
        ).astype(np.int64)
    active_rows = _collective_identity_rows(
        partitioned, active, width + 3, scope=scope, budget=budget
    )
    parent_rows = _collective_identity_rows(
        partitioned, parents, 2 * width + 6, scope=scope, budget=budget
    )
    scientific_rows = _collective_identity_rows(
        partitioned, scientific, 2, scope=scope, budget=budget
    )
    retired = tuple(
        _collective_identity_rows(
            partitioned,
            {
                part: np.column_stack(
                    (
                        hierarchy.retired_entity_ids[degree - 1],
                        hierarchy.retired_entity_keys[degree - 1],
                    )
                ).astype(np.int64)
                for part, hierarchy in hierarchies.items()
            },
            degree + 2,
            scope=scope,
            budget=budget,
        )
        for degree in range(1, dimension)
    )
    counters = (
        np.asarray(
            multihost_utils.process_allgather(
                np.asarray(
                    (
                        max((h.next_vertex_id for h in hierarchies.values()), default=0),
                        max((h.next_cell_id for h in hierarchies.values()), default=0),
                    ),
                    dtype=np.int64,
                )
            )
        )
        .reshape((-1, 2))
        .max(axis=0)
    )
    uniform = partitioned.prepared.anchor.start.uniform_refinement
    if uniform is not None and np.all(
        np.isin(np.asarray(uniform.parent_ids), active_rows[:, 0])
    ):
        uniform = None
    return _dependency_phydrax_meshing__bisection.BisectionHierarchy(
        dimension,
        active_rows[:, 0],
        active_rows[:, 1 : 1 + width],
        active_rows[:, -2],
        active_rows[:, -1],
        record_parent_ids=parent_rows[:, 0],
        record_parent_blocks=parent_rows[:, 1],
        record_parent_rows=parent_rows[:, 2 : 2 + width],
        record_parent_vertices=parent_rows[:, 2 + width : 2 + 2 * width],
        record_parent_tags=parent_rows[:, -4],
        record_child_ids=parent_rows[:, -3:-1],
        record_vertex_ids=parent_rows[:, -1],
        retired_entity_keys=tuple(rows[:, 1:] for rows in retired),
        retired_entity_ids=tuple(rows[:, 0] for rows in retired),
        next_vertex_id=int(counters[0]),
        next_cell_id=int(counters[1]),
        uniform_refinement=uniform,
        scientific_cell_ids=scientific_rows[:, 0],
        scientific_block_ids=scientific_rows[:, 1],
    )


def _rebalance_partitioned_epoch(
    partitioned: PartitionedAdaptiveSimplex,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    PartitionedAdaptiveSimplex,
    _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
]:
    """Migrate an imbalanced accepted raw epoch before owner-local publication."""

    policy = partitioned.prepared.adaptation.policy.partition_policy
    if policy is None:
        raise ValueError(
            "partitioned_publication.policy must retain its partition policy; got none"
        )
    count = evidence.global_entity_counts[-1]
    owners = evidence.entity_owners[-1][:count]
    weights = jnp.bincount(owners, length=policy.part_count).astype(jnp.float64)
    balanced = jnp.all(weights > 0.0) & (
        jnp.max(weights) / jnp.mean(weights) <= policy.maximum_imbalance
    )
    if bool(jax.device_get(balanced)):
        return partitioned, states, evidence
    if partitioned.migration is not None:
        raise ValueError(
            "partitioned_migration.imbalance must meet the declared bound; got an unbalanced migrated epoch"
        )
    bisection_host.uniform_count(
        states.mesh.cell_ids.size + states.mesh.vertex_ids.size,
        states.mesh.cell_ids.size * 8,
        budget=budget,
    )
    migration_state = (
        _owner_phydrax_meshing__distribution_migration.prepare_ownerlocal_migration_state(
            partitioned,
            states,
            cell_field_ids=states.mesh.cell_ids,
            cell_fields=jnp.zeros((*states.mesh.cell_ids.shape, 0), dtype=jnp.float64),
            vertex_field_ids=states.mesh.vertex_ids,
            vertex_fields=jnp.zeros(
                (*states.mesh.vertex_ids.shape, 0), dtype=jnp.float64
            ),
            cell_epochs=states.generations.astype(jnp.int64),
        )
    )
    proposal = (
        _owner_phydrax_meshing__distribution.prepare_partitioned_rebalance_proposal(
            partitioned, migration_state, policy
        )
    )
    migration = _owner_phydrax_meshing__distribution_migration.migrate_partitioned_epoch(
        partitioned, states, migration_state, proposal.requested_target_owners
    )
    placement = tuple(
        sorted(
            (
                *partitioned.placement_arrays,
                (
                    "placement/initial_solver_cell_owners",
                    migration.initial_solver_cell_owners,
                ),
                *(
                    (f"rebalance/{name}", value)
                    for name, value in _owner_phydrax_meshing__distribution.partitioned_rebalance_proposal_arrays(
                        proposal
                    ).items()
                ),
            )
        )
    )
    moved = PartitionedAdaptiveSimplex(
        partitioned.prepared,
        partitioned.layout,
        migration.parts,
        migration.initial_states,
        migration.source_exterior,
        migration.slot_origins,
        solver_cell_owners=migration.initial_solver_cell_owners,
        placement_arrays=placement,
        solver_neighborhood=migration.solver_neighborhood,
        solver_forest=migration.solver_forest,
        migration=migration,
        rebalance_proposal=proposal,
        execution_evidence=partitioned.execution_evidence,
        scope=scope,
        budget=budget,
    )
    accepted = _dependency_phydrax_meshing__result.CollectiveMeshEvidence(
        moved, migration.states, scope=scope, budget=budget
    )
    if accepted.topology_id != evidence.topology_id:
        raise ValueError(
            "partitioned_migration.topology must preserve the adapted scientific topology; got changed identity"
        )
    return moved, migration.states, accepted


@checked
def commit_partitioned_adaptive_simplex(
    partitioned: PartitionedAdaptiveSimplex,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
) -> _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult:
    preparation = partitioned.prepared.adaptation
    receipt = partitioned.execution_evidence
    if receipt is None:
        receipt = partitioned.prepared.execution_evidence
    allowance = _phase_allowance(preparation, receipt)
    limits = preparation.policy.limits
    with (
        bisection_host.uniform_execution(allowance) as scope,
        device_coordinate_budget(limits, scope=scope) as budget,
    ):
        with bisection_host.uniform_budget(allowance, scope=scope, budget=budget):
            _retain_preparation(partitioned, scope=scope, budget=budget)
            budget.retain_basis((states,))
        baseline = partitioned.states.counters
        if baseline.shape != states.counters.shape or not bool(
            jax.device_get(jnp.all(baseline >= 0) & jnp.all(states.counters >= baseline))
        ):
            raise ValueError(
                "device_adaptation.commit_partitioned_adaptive_simplex.baseline must retain their prepared-source baseline; got missing their prepared-source baseline"
            )
        work = int(
            jax.device_get(
                jnp.sum(
                    states.counters[
                        :,
                        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS,
                    ]
                    - baseline[
                        :,
                        _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexCounter.BISECTIONS,
                    ],
                    dtype=jnp.int64,
                )
            )
        )
        with scope.batch(work) as ticket:
            ticket.consume(work)
        with (
            bisection_host.uniform_budget(allowance, scope=scope, budget=budget),
            budget.ticket.native_target(),
        ):
            results = _run_partitioned_adaptive_simplex_commit(
                partitioned, states, scope=scope, budget=budget
            )
    if scope.evidence is None:
        return results
    record = NativeExecutionRecord(
        scope.evidence, preparation_evidence=receipt, owner_id=preparation.prepared_id
    )
    return _bind_collective_execution_receipt(
        preparation, results, record, scope=scope, budget=budget
    )


@final
class _CommittedNeighborhood(StrictModule):
    """Validated closure coordinates and logical publication arrays for the accepted epoch."""

    logical: _LogicalPublication
    neighborhood: _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset


def _prepare_committed_neighborhood(
    partitioned: PartitionedAdaptiveSimplex,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CommittedNeighborhood:
    """Validated closure coordinates and logical publication arrays for the accepted epoch."""
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
        raise ValueError(
            "device_adaptation.prepare_committed_neighborhood.distribution must retain its prepared ownership; got missing its prepared ownership"
        )
    if partitioned.migration is not None:
        neighborhood = partitioned.migration.solver_neighborhood
    elif evidence.uniform_refinement is None:
        neighborhood = _dependency_phydrax_meshing__distribution.expand_partitioned_simplex_neighborhood(
            partitioned.parts,
            states,
            partitioned.states,
            partitioned.source_exterior,
            halo_width=distribution.halo_width,
            cell_capacity=partitioned.layout.cell_capacity,
            vertex_capacity=partitioned.layout.vertex_capacity,
        )
    else:
        neighborhood = _packed_neighborhood(
            partitioned, states, evidence, scope=scope, budget=budget
        )
    neighborhood_flags = host_process_status_union(neighborhood.status)
    _dependency_phydrax_meshing__device_status.require_applied_status(
        neighborhood_flags,
        _dependency_phydrax_meshing__device_status.DeviceEpoch.BISECTION,
    )
    coordinate_table = jnp.where(
        logical.entity_keys[0] >= 0,
        logical.entity_keys[0],
        jnp.iinfo(jnp.int64).max,
    )
    coordinate_rows = key_positions(
        coordinate_table,
        neighborhood.cell_vertices.reshape((-1, 1)),
    )
    closure_vertices = jnp.repeat(
        neighborhood.cell_valid, neighborhood.cell_vertices.shape[-1]
    )
    if bool(jax.device_get(jnp.any(closure_vertices & (coordinate_rows < 0)))):
        raise ValueError(
            "device_adaptation.prepare_committed_neighborhood.closure_vertices must exist in accepted coordinate ownership; got a closure vertex is absent from accepted logical coordinate ownership"
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
    return _CommittedNeighborhood(logical, neighborhood)


@final
class _CommittedPublication(StrictModule):
    """Consumed geometry, organization, storage, and association projections for local publication."""

    logical: _LogicalPublication


def _lower_committed_publication(
    partitioned: PartitionedAdaptiveSimplex,
    original: _dependency_phydrax_meshing__result.CellMeshingResult
    | _owner_phydrax_meshing__initial_certification.InitialCollectiveMeshEvidence,
    evidence: _dependency_phydrax_meshing__result.CollectiveMeshEvidence,
    neighborhood: _dependency_phydrax_meshing__distribution.SimplexNeighborhoodWorkset,
    logical: _LogicalPublication,
) -> _CommittedPublication:
    """Consumed geometry, organization, storage, and association projections for local publication."""
    projection = (
        _dependency_phydrax_meshing__publication_lowering.PreparedPublicationLowering(
            original,
            evidence,
            neighborhood,
            vertex_capacity=partitioned.layout.vertex_capacity,
        ).execute()
    )
    scopes = _dependency_phydrax_meshing__organization.prepare_mesh_organization_scopes(
        original,
        evidence,
        projection,
        f"adaptation:{partitioned.prepared.adaptation.prepared_id}",
    )
    attributes = _dependency_phydrax_meshing__organization_records.prepare_mesh_attribute_projections(
        scopes
    )
    storage_binding = _dependency_phydrax_meshing__result.CollectiveMeshStorageBinding(
        evidence, logical.arrays, logical.counts
    )
    transfer = partitioned.prepared.adaptation.policy.association_transfer
    surface_transfer = (
        transfer.with_receipts(
            transfer.support.prepare_receipts(logical.arrays, projection.geometry)
        )
        if isinstance(
            transfer, _dependency_phydrax_meshing__association.SurfaceAssociationTransfer
        )
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
    return _CommittedPublication(logical)


def _run_partitioned_adaptive_simplex_commit(
    partitioned: PartitionedAdaptiveSimplex,
    states: _dependency_phydrax_discretization__adaptive_simplex.AdaptiveSimplexState,
    /,
    *,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult:
    """Collectively publish every process-addressable accepted refinement closure.

    Raw forest, source-map and semantic banks are checked before projection.
    Fixed-capacity collective packets then lower the received closure;
    no host-global mesh or variable-shaped global lookup is used by a local
    constructor. The tuple contains only this process's addressable partitions.
    """

    if states.mesh.cells.shape[:2] != (
        partitioned.parts.part_count,
        partitioned.layout.cell_capacity,
    ):
        raise ValueError(
            "device_adaptation.run_partitioned_adaptive_simplex_commit.states must be the stacked part states of this epoch; got an incompatible value"
        )
    started = time.monotonic()
    flags = host_process_status_union(states.status_flags)
    _dependency_phydrax_meshing__device_status.require_applied_status(
        flags, _dependency_phydrax_meshing__device_status.DeviceEpoch.BISECTION
    )
    source = partitioned.prepared.adaptation.source
    original = _scientific_preparation_source(partitioned.prepared.adaptation)
    if (
        partitioned.source_witness is None
        or source.boundary is not None
        or source.mesh.periodic_topology is not None
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.INVALID_SPECIFICATION,
            "device_adaptation.run_partitioned_adaptive_simplex_commit.partitioned must provide source-map support and a nonperiodic, explicitly bound source boundary model; got an absent or incompatible value",
            stage="distributed-device-commit",
        )
    evidence = _dependency_phydrax_meshing__result.CollectiveMeshEvidence(
        partitioned, states, scope=scope, budget=budget
    )
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "device_adaptation.run_partitioned_adaptive_simplex_commit.any must stay within the declared capacity; got accepted logical topology exceeds the requested global resource controls",
            stage="distributed-device-commit",
            requested=tuple(
                (f"maximum_{name}", maximum) for name, _, maximum in resources
            ),
            achieved=tuple((name, actual) for name, actual, _ in resources),
        )
    partitioned, states, evidence = _rebalance_partitioned_epoch(
        partitioned, states, evidence, scope=scope, budget=budget
    )
    local = _collective_local_phase(
        lambda: _addressable_commits(partitioned, states, scope=scope, budget=budget),
        budget=budget,
    )
    phase = _prepare_committed_neighborhood(
        partitioned=partitioned,
        states=states,
        evidence=evidence,
        scope=scope,
        budget=budget,
    )
    logical = phase.logical
    neighborhood = phase.neighborhood
    phase = _lower_committed_publication(
        partitioned=partitioned,
        original=original,
        evidence=evidence,
        neighborhood=neighborhood,
        logical=logical,
    )
    logical = phase.logical
    targets = _collective_local_phase(
        lambda: tuple(
            _local_target(
                partitioned,
                _neighborhood_commit(
                    partitioned, owned, neighborhood, scope=scope, budget=budget
                ),
                logical,
                evidence,
                scope=scope,
                budget=budget,
            )
            for owned in local
        ),
        budget=budget,
    )
    premises = _collective_local_phase(
        lambda: tuple(
            _collective_plc_premises(
                partitioned, target, evidence, scope=scope, budget=budget
            )
            for target in targets
        ),
        budget=budget,
    )
    results = _collective_local_phase(
        lambda: tuple(
            _local_adaptation_result(
                partitioned,
                target,
                premise,
                logical,
                evidence,
                started,
                scope=scope,
                budget=budget,
            )
            for target, premise in zip(targets, premises, strict=True)
        ),
        budget=budget,
    )
    return _dependency_phydrax_meshing__result.PartitionedMeshAdaptationResult(
        results,
        _collective_adaptation_identity(
            partitioned, results, evidence, scope=scope, budget=budget
        ),
    )


_Phase = TypeVar("_Phase")


def _collective_local_phase(
    action: Callable[[], tuple[_Phase, ...]],
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_Phase, ...]:
    """Run one owner-local phase; every process rejects if any owner failed."""
    values: tuple[_Phase, ...] = ()
    failure: _leaf_phydrax_meshing__failures.MeshingFailure | None = None
    try:
        with budget.measurement():
            values = action()
    except _leaf_phydrax_meshing__failures.MeshingFailure as error:
        failure = error
    except (
        MeshcoreError,
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError,
    ) as error:
        failure = _collective_resource_failure(error)
    except ValueError as error:
        failure = _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            str(error),
            stage="distributed-device-commit",
        )
    collective_failure_owner.collective_failure_barrier(
        failure,
        message="An owning process rejected the exact topology, lineage, or geometry "
        "resolution of the distributed epoch; no part was published.",
        stage="distributed-device-commit",
    )
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
