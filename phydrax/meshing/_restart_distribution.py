#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Repack saved simplex forest records for a changed execution topology.

Source recipes, accepted certificates, and predecessor proofs remain with the
checkpoint or epoch owners. This module returns placement inventory for their
numerical stable-ID records.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from enum import IntEnum, IntFlag
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec, Sharding

from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    masked_simplex_facet_neighbors,
    MaskedSimplexMesh,
    vertex_half_facets,
)
from ..typing import checked

# Module references defer symbol resolution across mutually dependent execution owners.
# Deferred symbol lookup: _restart_distribution <-> _distribution_migration cycle.
from . import (
    _device_adaptation as device_adaptation_owner,
    _distribution_migration as migration_owner,
)
from ._distribution_forest import (
    forest_cell_arrays,
    forest_vertex_arrays,
    ids,
    members,
    simplex_forest_arrays,
    slots,
    validated_forest_roots,
)
from ._result import CellMeshingResult, CollectiveMeshEvidence


_SENTINEL = jnp.iinfo(jnp.int64).max
_INVALID = int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
_CAPACITY = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)


class SimplexRestartInvalidReason(IntFlag):
    """Closed provenance bits for restart INVALID_GEOMETRY predicates."""

    CELL_AUTHORITY = 1 << 0
    VERTEX_AUTHORITY = 1 << 1
    CELL_OWNER_RANGE = 1 << 2
    VERTEX_OWNER_RANGE = 1 << 3
    SAVED_STATUS = 1 << 4
    CELL_PARENT_RANGE = 1 << 5
    CELL_VERTEX_RANGE = 1 << 6
    TUPLE_VERTEX_RANGE = 1 << 7
    BISECTION_VERTEX_RANGE = 1 << 8
    CELL_CHILD_RANGE = 1 << 9
    VERTEX_PARENT_RANGE = 1 << 10
    REQUIRED_VERTEX_SLOT = 1 << 11
    REQUIRED_VERTEX_ID = 1 << 12
    CELL_PARENT_ID = 1 << 13
    CELL_CHILD_ID = 1 << 14
    BISECTION_VERTEX_ID = 1 << 15
    VERTEX_PARENT_ID = 1 << 16
    PROTECTED_ENDPOINT_ID = 1 << 17
    NONFINITE_COORDINATE = 1 << 18
    PROTECTED_CODE_RANGE = 1 << 19
    ALLOCATOR_CURSOR = 1 << 20
    FOREST_ROOT = 1 << 21
    PROPOSED_OWNER_RANGE = 1 << 22
    DUPLICATE_RECORD_CONFLICT = 1 << 23
    VERTEX_CLOSURE_LIMIT = 1 << 24
    MISSING_CELL_VERTEX = 1 << 25
    MISSING_TUPLE_VERTEX = 1 << 26
    MISSING_BISECTION_VERTEX = 1 << 27
    MISSING_CELL_PARENT = 1 << 28
    MISSING_CELL_CHILD = 1 << 29
    MISSING_VERTEX_PARENT = 1 << 30
    CELL_COUNT = 1 << 31
    VERTEX_COUNT = 1 << 32
    VERTEX_REMOVAL_HISTORY = 1 << 33


class SimplexRestartRecordRole(IntEnum):
    CELL = 1
    VERTEX = 2


class SimplexRestartRecordDtype(IntEnum):
    BOOL = 1
    INT32 = 2
    INT64 = 3
    FLOAT32 = 4
    FLOAT64 = 5
    UINT32 = 6
    UINT64 = 7
    INT8 = 8
    INT16 = 9
    UINT8 = 10
    UINT16 = 11
    FLOAT16 = 12
    BFLOAT16 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15


class SimplexRestartEquality(IntEnum):
    ELEMENTWISE_EQUAL = 1


def _first_conflict(left: Array, right: Array) -> Array:
    """Lexicographic minimum; all -1 is the absent witness."""
    a = jnp.where(left[0] < 0, _SENTINEL, left)
    b = jnp.where(right[0] < 0, _SENTINEL, right)
    index = jnp.argmax(a != b)
    return jnp.where(a[index] <= b[index], left, right)


def _packet_conflict(
    current: dict[str, Array],
    incoming: dict[str, Array],
    selected: Array,
    role: SimplexRestartRecordRole,
) -> Array:
    """First key/field/component mismatch under exact elementwise equality.

    Columns: role, scientific ID, sorted field index, flattened component,
    dtype enum, source rank low, source rank high, equality enum.
    Diagnostic source ranks are excluded from scientific comparison.
    """
    keys = jnp.concatenate(
        (current["ids"], jnp.where(selected, incoming["ids"], _SENTINEL))
    )
    order = jnp.argsort(keys, stable=True)
    keys = keys[order]
    duplicate = (keys[1:] == keys[:-1]) & (keys[1:] != _SENTINEL)
    owners = jnp.concatenate(
        (current["_source_rank"], incoming["_source_rank"])
    )[order]
    witness = jnp.full((8,), -1, jnp.int64)
    for field, name in enumerate(sorted(key for key in current if key != "_source_rank")):
        values = jnp.concatenate((current[name], incoming[name]))[order]
        different = (values[1:] != values[:-1]).reshape((keys.size - 1, -1))
        mismatch = duplicate[:, None] & different
        row = jnp.argmax(jnp.any(mismatch, axis=1))
        same_key = keys[:-1] == keys[row]
        component = jnp.argmax(jnp.any(mismatch & same_key[:, None], axis=0))
        matches = same_key & mismatch[:, component]
        lower = jnp.minimum(owners[:-1], owners[1:])
        upper = jnp.maximum(owners[:-1], owners[1:])
        low = jnp.min(jnp.where(matches, lower, _SENTINEL))
        row = jnp.argmin(jnp.where(matches & (lower == low), upper, _SENTINEL))
        candidate = jnp.asarray(
            [
                int(role), keys[row], field, component,
                int(SimplexRestartRecordDtype[values.dtype.name.upper()]),
                jnp.minimum(owners[row], owners[row + 1]),
                jnp.maximum(owners[row], owners[row + 1]),
                int(SimplexRestartEquality.ELEMENTWISE_EQUAL),
            ],
            jnp.int64,
        )
        witness = _first_conflict(
            witness, jnp.where(jnp.any(mismatch), candidate, -1)
        )
    return witness


def validate_restart_duplicate_conflicts(
    masks: Array, conflicts: Array, source_count: int,
) -> None:
    """Bind closed typed witnesses to the existing duplicate-refusal bit."""
    if conflicts.shape != (*masks.shape, 8) or conflicts.dtype != jnp.int64:
        raise ValueError("restart.duplicate_record_conflicts must be int64[target,8]")
    present = (masks & int(SimplexRestartInvalidReason.DUPLICATE_RECORD_CONFLICT)) != 0
    valid = jnp.where(
        present,
        ((conflicts[:, 0] == int(SimplexRestartRecordRole.CELL))
         | (conflicts[:, 0] == int(SimplexRestartRecordRole.VERTEX)))
        & jnp.all(conflicts[:, 1:4] >= 0, axis=1)
        & (conflicts[:, 4] >= 1)
        & (conflicts[:, 4] <= max(SimplexRestartRecordDtype))
        & (conflicts[:, 5] >= 0)
        & (conflicts[:, 5] <= conflicts[:, 6])
        & (conflicts[:, 6] < source_count)
        & (conflicts[:, 7] == int(SimplexRestartEquality.ELEMENTWISE_EQUAL)),
        jnp.all(conflicts == -1, axis=1),
    )
    if not bool(jax.device_get(jnp.all(valid))):
        raise ValueError("restart.duplicate_record_conflicts must match duplicate reason and closed schema")

def _invalid_evidence(predicate: Array, reason: SimplexRestartInvalidReason) -> Array:
    """Carry diagnostic bits above the unchanged low-byte status internally."""
    return jnp.where(
        predicate, jnp.int64(_INVALID | (int(reason) << 8)), jnp.int64(0)
    )


@eqx.filter_jit
def _invalid_reason_summary(masks: Array) -> tuple[Array, Array, Array]:
    bits = jnp.asarray(tuple(int(bit) for bit in SimplexRestartInvalidReason), jnp.int64)
    present = (masks[:, None] & bits[None, :]) != 0
    counts = jnp.sum(present, axis=0, dtype=jnp.int32)
    first = jnp.min(
        jnp.where(present, jnp.arange(masks.shape[0])[:, None], masks.shape[0]),
        axis=0,
    )
    # CPU multi-process collectives support arithmetic sums, not int64 OR.
    # Every closed reason is a distinct bit, so summing present bits is exact.
    mask = jnp.sum(jnp.where(counts > 0, bits, 0), dtype=jnp.int64)
    return mask, counts, jnp.where(counts > 0, first, -1)


def validate_restart_invalid_reasons(status: Array, masks: Array) -> None:
    """Reject unknown bits and disagreement with the collective scalar status."""
    if masks.shape != status.shape or masks.dtype != jnp.int64:
        raise ValueError("restart.invalid_reason_masks must be int64 and match status shape")
    known = sum(int(bit) for bit in SimplexRestartInvalidReason)
    valid = jnp.all((masks >= 0) & ((masks & ~known) == 0))
    valid &= jnp.all(((status & _INVALID) != 0) == jnp.any(masks != 0))
    if not bool(jax.device_get(valid)):
        raise ValueError("restart.invalid_reason_masks must use known bits consistent with status")


@final
class SimplexRestartPlacement(StrictModule, NonTrainableState):
    """Saved execution metadata without requiring its former physical devices."""

    axis_name: str = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)


@final
class SimplexRestartRepack(StrictModule, NonTrainableState):
    """Repacked scientific records with live or saved execution placement."""

    parts: AdaptiveSimplexParts | SimplexRestartPlacement
    layout: AdaptiveSimplexLayout
    states: AdaptiveSimplexState
    cell_owners: Array
    vertex_owners: Array
    cell_history: tuple[Array, ...]
    vertex_history: tuple[Array, ...]
    status: Array
    cells_before: Array
    cells_after: Array
    vertices_before: Array
    vertices_after: Array
    saved_cursors: Array
    saved_clocks: Array
    saved_counters: Array
    saved_states: AdaptiveSimplexState
    saved_cell_owners: Array
    saved_vertex_owners: Array
    source_result: CellMeshingResult | None
    cell_saved_locations: Array
    vertex_saved_locations: Array
    requested_target_owners: Array
    saved_cell_history: tuple[Array, ...]
    saved_vertex_history: tuple[Array, ...]
    solver_cell_owners: Array
    saved_solver_cell_owners: Array | None
    cell_migration_counts: Array | None
    invalid_reason_masks: Array
    duplicate_record_conflicts: Array
    graph_proposal: migration_owner.SimplexGraphRestartProposal | None = None

    def invalid_reason_summary(self) -> tuple[Array, Array, Array]:
        """Return global mask, affected-target counts and first target indices.

        Counts describe target ranks, not distinct corrupt scientific records.
        Indices are target ranks; absent reasons have index -1.
        """
        return _invalid_reason_summary(self.invalid_reason_masks)

    def duplicate_record_schema(self) -> dict[str, object]:
        """Decode typed witnesses without materializing scientific arrays."""
        return {
            "columns": (
                "role", "scientific_id", "field_index", "component_index",
                "dtype", "source_rank_low", "source_rank_high", "equality",
            ),
            "absent": -1,
            "roles": {role.name: int(role) for role in SimplexRestartRecordRole},
            "dtypes": {dtype.name: int(dtype) for dtype in SimplexRestartRecordDtype},
            "equality": {
                "ELEMENTWISE_EQUAL": int(SimplexRestartEquality.ELEMENTWISE_EQUAL),
            },
            "fields": {
                # Vertex activity is normalized from authenticated source records
                # before exact packet comparison; scientific Boolean payloads are not.
                "CELL": sorted((
                    "ids", "cells", "tuples", "active", "parents", "ordinal",
                    "children", "bisection", *forest_cell_arrays(self.saved_states),
                    *(f"payload_{i}" for i in range(len(self.saved_cell_history))),
                )),
                "VERTEX": sorted((
                    "ids", "coordinates", "active", "parents",
                    *forest_vertex_arrays(self.saved_states),
                    *(f"payload_{i}" for i in range(len(self.saved_vertex_history))),
                )),
            },
            "partition_metadata": {
                "VERTEX.active": "any authenticated source active for scientific ID",
                "source_active_masks": "saved_states.mesh.vertex_active",
                "source_ids": "saved_states.mesh.vertex_ids",
                "source_owners": "saved_vertex_owners",
                "VERTEX.vertex_removal": "active: -1; inactive: earliest cumulative coarsening event",
                "source_removal_events": "saved_states.vertex_removal",
                "source_history_clocks": "saved_states.clocks",
            },
        }

    def execution_parts(self) -> AdaptiveSimplexParts:
        """Return the live execution placement required for publication."""
        match self.parts:
            case AdaptiveSimplexParts() as parts:
                return parts
            case SimplexRestartPlacement():
                raise ValueError(
                    "restart.parts must provide live execution devices; got saved placement"
                )


def _restart_local_phase(collective: bool) -> AbstractContextManager[None]:
    if collective:
        from ..lifecycle._meshing_sources import meshing_source_local_restore_phase

        return meshing_source_local_restore_phase()
    return nullcontext()


@final
class SimplexRestartRepackProof(StrictModule, NonTrainableState):
    """All-rank numerical proof consumed later without owner-local collectives."""

    arrays: tuple[tuple[str, Array], ...]
    source: CellMeshingResult
    layout: AdaptiveSimplexLayout
    content_id: str = eqx.field(static=True)
    source_result_id: str = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    historical_placement: bool = eqx.field(static=True)
    execution_sharding: Sharding = eqx.field(static=True)
    constructor: Callable[..., None] = eqx.field(static=True)
    content_constructor: Callable[..., str] = eqx.field(static=True)

    @checked
    def __init__(
        self, receipt: SimplexRestartRepack, /, *, collective: bool = False
    ) -> None:
        identity = simplex_restart_repack_content_id(receipt, collective=collective)
        with _restart_local_phase(collective):
            source = receipt.source_result
            if source is None:
                raise ValueError("restart.source_result must be present; got None")
            arrays = tuple(sorted({
                **_restart_repack_arrays(receipt),
                "diagnostic/invalid_reason_masks": receipt.invalid_reason_masks,
                "diagnostic/duplicate_record_conflicts": receipt.duplicate_record_conflicts,
            }.items()))
            historical = collective and isinstance(receipt.parts, SimplexRestartPlacement)
            parts = receipt.parts if historical else SimplexRestartRepack.execution_parts(receipt)
            self.arrays = arrays
            self.source = source
            self.layout = receipt.layout
            self.content_id = identity
            self.source_result_id = source.result_id
            self.axis_name = parts.axis_name
            self.partition_count = parts.part_count
            self.neighbor_pairs = parts.neighbor_pairs
            self.historical_placement = historical
            self.execution_sharding = receipt.states.mesh.cell_ids.sharding
            self.constructor = type(self).__init__
            self.content_constructor = simplex_restart_repack_content_id

    @checked
    def require_receipt(self, receipt: SimplexRestartRepack, /) -> None:
        """Require immutable array/source bindings using only local metadata."""
        parts = (
            receipt.parts
            if self.historical_placement
            else SimplexRestartRepack.execution_parts(receipt)
        )
        if (
            type(self).__init__ is not self.constructor
            or simplex_restart_repack_content_id is not self.content_constructor
            or self.historical_placement
            != isinstance(receipt.parts, SimplexRestartPlacement)
        ):
            raise ValueError(
                "restart.proof must retain its actual constructor and placement kind; got a changed reconstruction"
            )
        if (
            receipt.source_result is not self.source
            or receipt.source_result.result_id != self.source_result_id
            or receipt.layout is not self.layout
            or receipt.parts.axis_name != self.axis_name
            or receipt.parts.part_count != self.partition_count
            or receipt.parts.neighbor_pairs != self.neighbor_pairs
        ):
            raise ValueError(
                "restart.proof must bind the same source and placement; got mismatched source, layout, or owners"
            )
        if self.historical_placement:
            placement = self.execution_sharding
        else:
            if not isinstance(parts, AdaptiveSimplexParts):
                raise ValueError(
                    "restart.proof must retain actual live execution parts."
                )
            placement = NamedSharding(parts.mesh, PartitionSpec(self.axis_name))
        if not receipt.states.mesh.cell_ids.sharding.is_equivalent_to(
            placement, receipt.states.mesh.cell_ids.ndim
        ):
            raise ValueError(
                "restart.states must use the validated execution sharding; got different device placement"
            )
        actual = {
            **_restart_repack_arrays(receipt),
            "diagnostic/invalid_reason_masks": receipt.invalid_reason_masks,
            "diagnostic/duplicate_record_conflicts": receipt.duplicate_record_conflicts,
        }
        if len(actual) != len(self.arrays) or any(
            value.is_deleted() or name not in actual or actual[name] is not value
            for name, value in self.arrays
        ):
            raise ValueError(
                "restart.arrays must retain validated numerical references; got different array bindings"
            )


@eqx.filter_jit
def _restart_solver_owners(active: Array, locations: Array, requested: Array) -> Array:
    parts = jnp.clip(locations[:, :, 0], 0, requested.shape[0] - 1)
    slots = jnp.clip(locations[:, :, 1], 0, requested.shape[1] - 1)
    found = jnp.all(locations >= 0, axis=-1)
    return jnp.where(active & found, requested[parts, slots], -1).astype(jnp.int32)


def _source_solver_owners(
    source: CellMeshingResult, saved: AdaptiveSimplexState, *, collective: bool = False
) -> Array:
    """Bind saved solver senders to the accepted source entity bank."""
    with _restart_local_phase(collective):
        evidence = source.collective_evidence
        storage = source.mesh.storage
        if evidence is None or storage is None:
            raise ValueError(
                "restart.source_result must retain collective mesh evidence and storage; got missing source evidence"
            )
        count = saved.mesh.cell_ids.shape[0]
        if (
            evidence.partition_count != count
            or storage.partition_count != count
            or evidence.mesh_id != source.mesh.mesh_id
            or evidence.evidence_id != storage.evidence_id
        ):
            raise ValueError(
                "restart.saved_states must match the accepted source placement; got mismatched dimensions"
            )
        identifiers, owners = evidence.entity_ids[-1], evidence.entity_owners[-1]
        total = evidence.global_entity_counts[-1]
    valid = jnp.arange(identifiers.shape[0]) < total
    order = jnp.argsort(jnp.where(valid, identifiers, _SENTINEL), stable=True)
    keys = jnp.where(valid[order], identifiers[order], _SENTINEL)
    position = jnp.minimum(jnp.searchsorted(keys, saved.mesh.cell_ids), keys.shape[0] - 1)
    matched = keys[position] == saved.mesh.cell_ids
    selected = owners[order[position]]
    invalid = jnp.any(
        saved.mesh.cell_active & (~matched | (selected < 0) | (selected >= count))
    )
    invalid |= jnp.any((keys[1:] == keys[:-1]) & (keys[1:] != _SENTINEL))
    invalid |= jnp.sum(saved.mesh.cell_active, dtype=jnp.int64) != total
    rejected = bool(jax.device_get(invalid))
    with _restart_local_phase(collective):
        if rejected:
            raise ValueError(
                "restart.saved_states must cover accepted source ownership; got missing active records"
            )
    return jnp.where(saved.mesh.cell_active, selected, -1).astype(jnp.int32)


@eqx.filter_jit
def _restart_cell_traffic(
    saved: AdaptiveSimplexState,
    authority: Array,
    solver_owners: Array,
    requested: Array,
    count: int,
) -> Array:
    old_count = saved.mesh.cell_ids.shape[0]
    active = saved.mesh.cell_active & (
        authority == jnp.arange(old_count, dtype=jnp.int32)[:, None]
    )
    senders = jnp.where(active, solver_owners, old_count)
    receivers = jnp.where(active, requested, count)
    return (
        jnp.zeros((old_count, count), jnp.int64)
        .at[senders, receivers]
        .add(
            active.astype(jnp.int64),
            mode="drop",
        )
    )


def _numerical_digest(tree: object, *, collective: bool = False) -> str:
    """Compare numerical leaves through their logical-byte collection digest."""
    with _restart_local_phase(collective):
        arrays = {
            "value" + jax.tree_util.keystr(path): value
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
            if isinstance(value, Array)
        }
    return logical_array_value_collection_digest(arrays)


@eqx.filter_jit
def logical_restart_routes(vertex_ids: Array, /) -> Array:
    """Replay shared allocated-ID routes without assuming one device per saved row."""
    count = vertex_ids.shape[0]
    keys = jnp.sort(jnp.where(vertex_ids >= 0, vertex_ids, _SENTINEL), axis=1)

    def source_row(source: Array, mask: Array) -> Array:
        def target_row(target: Array, row_mask: Array) -> Array:
            positions = jnp.minimum(
                jnp.searchsorted(keys[target], keys[source]), keys.shape[1] - 1
            )
            shared = (source != target) & jnp.any(
                (keys[source] != _SENTINEL) & (keys[target, positions] == keys[source])
            )
            return row_mask.at[source, target].set(shared)

        return jax.lax.fori_loop(0, count, target_row, mask)

    return jax.lax.fori_loop(
        0, count, source_row, jnp.zeros((count, count), dtype=jnp.bool_)
    )


@checked
def validate_simplex_restart_repack(
    receipt: SimplexRestartRepack,
    requested_target_owners: Array,
    /,
    *,
    collective: bool = False,
) -> None:
    """Replay every repack premise with agreed local phases in common cold mode."""
    with _restart_local_phase(collective):
        source = receipt.source_result
        if source is None:
            raise ValueError(
                "restart.source_result must provide an accepted source; got None"
            )
        storage, evidence = source.mesh.storage, source.collective_evidence
        if storage is None or not isinstance(evidence, CollectiveMeshEvidence):
            raise ValueError(
                "restart.source_result must retain forest storage and collective evidence; got missing retained content"
            )
    saved_layout, _, _ = device_adaptation_owner.restore_partitioned_mesh_state(
        storage, evidence
    )
    # Keep the raw forest, not the packed publication view, as the replay input.
    accepted_saved = evidence.compiled_states
    saved_digest = _numerical_digest(receipt.saved_states, collective=collective)
    accepted_digest = _numerical_digest(accepted_saved, collective=collective)
    with _restart_local_phase(collective):
        if saved_digest != accepted_digest:
            raise ValueError(
                "restart.saved_states must match accepted source evidence; got different forest content"
            )
    requested_digest = _numerical_digest(requested_target_owners, collective=collective)
    retained_digest = _numerical_digest(
        receipt.requested_target_owners, collective=collective
    )
    with _restart_local_phase(collective):
        if requested_digest != retained_digest:
            raise ValueError(
                "restart.requested_target_owners must match the retained proposal; got different numerical owners"
            )
        placement = receipt.parts
        count = placement.part_count
        if (
            requested_target_owners.shape != receipt.saved_states.mesh.cell_ids.shape
            or requested_target_owners.dtype != jnp.int32
            or receipt.states.mesh.cell_ids.shape != (count, receipt.layout.cell_capacity)
            or receipt.states.mesh.vertex_ids.shape != (count, receipt.layout.vertex_capacity)
            or receipt.layout.dimension != saved_layout.dimension
            or receipt.layout.ambient_dimension != saved_layout.ambient_dimension
            or receipt.layout.coordinate_dtype != saved_layout.coordinate_dtype
        ):
            raise ValueError(
                "restart.layout must match source and target execution dimensions; got incompatible capacities or coordinates"
            )
    expected = _repack(
        receipt.layout,
        count,
        receipt.saved_states,
        receipt.saved_cell_owners,
        receipt.saved_vertex_owners,
        requested_target_owners,
        receipt.saved_cell_history,
        receipt.saved_vertex_history,
    )
    with _restart_local_phase(collective):
        if expected[0].mesh.signature_id != receipt.states.mesh.signature_id:
            raise ValueError(
                "restart.states must match the repacked layout signature; got different coordinate or cell capacity"
            )
        actual = (
            receipt.states, receipt.cell_owners, receipt.vertex_owners,
            receipt.cell_history, receipt.vertex_history, receipt.status,
            receipt.invalid_reason_masks,
            receipt.duplicate_record_conflicts,
        )
    expected_digest = _numerical_digest(expected, collective=collective)
    actual_digest = _numerical_digest(actual, collective=collective)
    with _restart_local_phase(collective):
        if expected_digest != actual_digest:
            raise ValueError(
                "restart.states must match replayed records, histories, and status; got different numerical content"
            )
        states, co, vo, _, _, status, _, _ = expected
    rejected = bool(jax.device_get(jnp.any(status != 0)))
    validate_restart_invalid_reasons(receipt.status, receipt.invalid_reason_masks)
    validate_restart_duplicate_conflicts(
        receipt.invalid_reason_masks, receipt.duplicate_record_conflicts,
        receipt.saved_states.mesh.cell_ids.shape[0],
    )
    with _restart_local_phase(collective):
        if rejected:
            local_status = [
                (int(code), AdaptiveSimplexStatus(int(code)).name)
                for shard in status.addressable_shards
                for code in np.asarray(shard.data).reshape(-1)
            ]
            raise ValueError(
                "restart.status must pass history packet validation; got nonzero status "
                f"{local_status}"
            )
    cells = _saved_locations(
        receipt.saved_states.mesh.cell_ids, receipt.saved_cell_owners, states.mesh.cell_ids,
    )
    vertices = _saved_locations(
        receipt.saved_states.mesh.vertex_ids, receipt.saved_vertex_owners, states.mesh.vertex_ids,
    )
    before_ranks = jnp.arange(
        receipt.saved_states.mesh.cell_ids.shape[0], dtype=jnp.int32
    )[:, None]
    after_ranks = jnp.arange(count, dtype=jnp.int32)[:, None]
    controls = (
        receipt.saved_states.cursors, receipt.saved_states.clocks, receipt.saved_states.counters,
    )
    counts = (
        jnp.sum(
            (receipt.saved_states.mesh.cell_ids >= 0)
            & (receipt.saved_cell_owners == before_ranks), axis=1, dtype=jnp.int64,
        ),
        jnp.sum(states.mesh.cell_ids >= 0, axis=1, dtype=jnp.int64),
        jnp.sum(
            (receipt.saved_states.mesh.vertex_ids >= 0)
            & (receipt.saved_vertex_owners == before_ranks), axis=1, dtype=jnp.int64,
        ),
        jnp.sum(
            (states.mesh.vertex_ids >= 0) & (vo == after_ranks), axis=1, dtype=jnp.int64,
        ),
    )
    expected_digest = _numerical_digest(
        (cells, vertices, controls, counts), collective=collective
    )
    actual_digest = _numerical_digest(
        (
            receipt.cell_saved_locations, receipt.vertex_saved_locations,
            (receipt.saved_cursors, receipt.saved_clocks, receipt.saved_counters),
            (receipt.cells_before, receipt.cells_after, receipt.vertices_before, receipt.vertices_after),
        ),
        collective=collective,
    )
    with _restart_local_phase(collective):
        if expected_digest != actual_digest:
            raise ValueError(
                "restart.saved_locations must match replayed locations, controls, and counts; got different retained content"
            )
    saved_solver = _source_solver_owners(source, receipt.saved_states, collective=collective)
    solver = _restart_solver_owners(states.mesh.cell_active, cells, requested_target_owners)
    traffic = _restart_cell_traffic(
        receipt.saved_states, receipt.saved_cell_owners, saved_solver,
        requested_target_owners, count,
    )
    expected_digest = _numerical_digest((saved_solver, solver, traffic), collective=collective)
    actual_digest = _numerical_digest(
        (receipt.saved_solver_cell_owners, receipt.solver_cell_owners, receipt.cell_migration_counts),
        collective=collective,
    )
    with _restart_local_phase(collective):
        if expected_digest != actual_digest:
            raise ValueError(
                "restart.cell_migration_counts must match solver ownership correspondence; got different owners or traffic"
            )
    if isinstance(placement, AdaptiveSimplexParts):
        routes = migration_owner.prepare_migration_routes(
            placement, migration_owner.forest_migration_routes(placement, receipt.states),
        )
        neighbor_pairs = routes.neighbor_pairs
    else:
        mask = np.asarray(
            jax.device_get(logical_restart_routes(receipt.states.mesh.vertex_ids)),
            dtype=np.bool_,
        )
        with _restart_local_phase(collective):
            neighbor_pairs = tuple(
                (int(source), int(target)) for source, target in np.argwhere(mask)
            )
    with _restart_local_phase(collective):
        if neighbor_pairs != placement.neighbor_pairs:
            raise ValueError(
                "restart.parts.neighbor_pairs must match shared allocated vertices; got different execution routes"
            )
        graph = receipt.graph_proposal
        if graph is not None and graph.parts.shape[0] != count:
            raise ValueError(
                "restart.graph_proposal must bind its executed projected owners; got different target ownership"
            )
    if graph is not None:
        graph_digest = _numerical_digest(graph.requested_target_owners, collective=collective)
        requested_digest = _numerical_digest(requested_target_owners, collective=collective)
        with _restart_local_phase(collective):
            if graph_digest != requested_digest:
                raise ValueError(
                    "restart.graph_proposal must bind its executed projected owners; got different target ownership"
                )
        migration_owner.replay_graph_restart_proposal(
            graph, receipt.saved_states, axis_name=placement.axis_name, collective=collective
        )


def _restart_repack_arrays(
    receipt: SimplexRestartRepack,
) -> dict[str, Array]:
    """Return the numerical inventory shared by identity and local binding."""
    old_solver = receipt.saved_solver_cell_owners
    traffic = receipt.cell_migration_counts
    if receipt.source_result is None or old_solver is None or traffic is None:
        raise ValueError(
            "restart.source_result must bind source and solver transport; got missing source, owners, or traffic"
        )
    arrays = {
        "proposal/target_owners": receipt.requested_target_owners,
        "source/raw_cell_owners": receipt.saved_cell_owners,
        "source/raw_vertex_owners": receipt.saved_vertex_owners,
        "source/solver_cell_owners": old_solver,
        "source/cursors": receipt.saved_cursors,
        "source/clocks": receipt.saved_clocks,
        "source/counters": receipt.saved_counters,
        "target/raw_cell_owners": receipt.cell_owners,
        "target/raw_vertex_owners": receipt.vertex_owners,
        "target/solver_cell_owners": receipt.solver_cell_owners,
        "target/cell_saved_locations": receipt.cell_saved_locations,
        "target/vertex_saved_locations": receipt.vertex_saved_locations,
        "target/cell_migration_counts": traffic,
        "target/status": receipt.status,
        "counts/cells_before": receipt.cells_before,
        "counts/cells_after": receipt.cells_after,
        "counts/vertices_before": receipt.vertices_before,
        "counts/vertices_after": receipt.vertices_after,
    }
    for prefix, state in (("source", receipt.saved_states), ("target", receipt.states)):
        arrays.update(
            {
                f"{prefix}/{name}": value
                for name, value in simplex_forest_arrays(state).items()
            }
        )
    for prefix, cells, vertices in (
        ("source", receipt.saved_cell_history, receipt.saved_vertex_history),
        ("target", receipt.cell_history, receipt.vertex_history),
    ):
        arrays.update(
            {f"{prefix}/cell_history/{index}": value for index, value in enumerate(cells)}
        )
        arrays.update(
            {
                f"{prefix}/vertex_history/{index}": value
                for index, value in enumerate(vertices)
            }
        )
    graph = receipt.graph_proposal
    if graph is not None:
        arrays.update(
            {
                "graph/parts": graph.parts,
                "graph/vertex_weights": graph.vertex_weights,
                "graph/edge_weights": graph.edge_weights,
                **{
                    f"graph/workset/{name}": value
                    for name, value in graph.workset.arrays().items()
                },
            }
        )
    return arrays


@checked
def simplex_restart_repack_content_id(
    receipt: SimplexRestartRepack, /, *, collective: bool = False
) -> str:
    """Validate the scientific placement inventory and compute its exact content ID."""
    validate_simplex_restart_repack(
        receipt, receipt.requested_target_owners, collective=collective
    )
    with _restart_local_phase(collective):
        source = receipt.source_result
        if source is None:
            raise ValueError(
                "restart.source_result must be present after validation; got None"
            )
        arrays = _restart_repack_arrays(receipt)
        placement = receipt.parts
    numerical_content = logical_array_value_collection_digest(arrays)
    with _restart_local_phase(collective):
        content_id = canonical_fingerprint(
            {
                "kind": "complete-simplex-restart-repack",
                "source_result": source.result_id,
                "source_mesh": source.mesh.mesh_id,
                "source_coordinate_contract": source.coordinate_contract.spatial_id,
                "source_layout": receipt.saved_states.mesh.signature_id,
                "target_layout": receipt.layout.signature_id,
                "old_partition_count": receipt.saved_states.mesh.cell_ids.shape[0],
                "new_partition_count": placement.part_count,
                "axis_name": placement.axis_name,
                "neighbor_pairs": placement.neighbor_pairs,
                "proposal_route": None
                if receipt.graph_proposal is None
                else {
                    "kind": "native-graph",
                    "plan": receipt.graph_proposal.plan.plan_id,
                },
                "numerical_content": numerical_content,
            }
        )
    return content_id


@eqx.filter_jit
def restart_forest_target_owners(
    ancestor_cell_ids: Array,
    repacked_cell_ids: Array,
    repacked_cell_owners: Array,
    /,
) -> tuple[Array, Array]:
    """Route a saved theorem's initial forest by retained scientific cell IDs.

    The ancestor forest can have different capacities and old part count.
    Every allocated ancestor record must have exactly one record-owner entry
    in the repacked forest; unknown/duplicated records refuse the whole mapping.
    These owners permit a second repack of the old initial states and
    their geometry/exterior payloads, not a relabeling of the predecessor proof.
    """
    if (
        ancestor_cell_ids.ndim != 2
        or repacked_cell_ids.ndim != 2
        or repacked_cell_owners.shape != repacked_cell_ids.shape
    ):
        raise ValueError(
            "restart.forest_ids must provide part and capacity axes; got incompatible array shapes"
        )
    if (
        ancestor_cell_ids.dtype != jnp.int64
        or repacked_cell_ids.dtype != jnp.int64
        or repacked_cell_owners.dtype != jnp.int32
    ):
        raise TypeError(
            "restart.forest_ids must use int64 IDs and int32 owners; got incompatible dtypes"
        )
    count = repacked_cell_ids.shape[0]
    result = jnp.full(ancestor_cell_ids.shape, count, jnp.int32)
    occurrences = jnp.zeros(ancestor_cell_ids.shape, jnp.int32)

    def visit(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        result, occurrences = carry
        keys = jnp.where(
            repacked_cell_ids[index] >= 0, repacked_cell_ids[index], _SENTINEL
        )
        left = jnp.searchsorted(keys, ancestor_cell_ids, side="left")
        right = jnp.searchsorted(keys, ancestor_cell_ids, side="right")
        matches = jnp.where(ancestor_cell_ids >= 0, right - left, 0).astype(jnp.int32)
        return jnp.minimum(
            result, jnp.where(matches > 0, index, count)
        ), occurrences + matches

    result, occurrences = jax.lax.fori_loop(0, count, visit, (result, occurrences))
    invalid = jnp.any((ancestor_cell_ids >= 0) & (occurrences != 1))
    invalid |= jnp.any(
        (repacked_cell_ids >= 0)
        & (repacked_cell_owners != jnp.arange(count, dtype=jnp.int32)[:, None])
    )
    return jnp.where(ancestor_cell_ids >= 0, result, -1).astype(jnp.int32), jnp.where(
        invalid, _INVALID, 0
    ).astype(jnp.int32)


@eqx.filter_jit
def _saved_locations(saved_ids: Array, owners: Array, target_ids: Array) -> Array:
    """Map new slots to their saved record-owner part and slot."""
    result = jnp.full((*target_ids.shape, 2), -1, jnp.int32)

    def visit(index: int, result: Array) -> Array:
        keys = jnp.where(saved_ids[index] >= 0, saved_ids[index], _SENTINEL)
        slots = jnp.minimum(jnp.searchsorted(keys, target_ids), keys.shape[0] - 1)
        found = (
            (target_ids >= 0)
            & (keys[slots] == target_ids)
            & (owners[index, slots] == index)
        )
        return jnp.where(
            found[:, :, None],
            jnp.stack((jnp.full_like(slots, index), slots), axis=-1).astype(jnp.int32),
            result,
        )

    return jax.lax.fori_loop(0, saved_ids.shape[0], visit, result)


def _merge_packet(
    current: dict[str, Array],
    packet: dict[str, Array],
    selected: Array,
    *,
    diagnostic: bool = False,
) -> tuple[dict[str, Array], Array]:
    capacity = current["ids"].shape[0]
    keys = jnp.concatenate(
        (current["ids"], jnp.where(selected, packet["ids"], _SENTINEL))
    )
    order = jnp.argsort(keys, stable=True)
    ordered = keys[order]
    unique = (ordered != _SENTINEL) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), ordered[1:] != ordered[:-1])
    )
    count = jnp.sum(unique, dtype=jnp.int32)
    pick = jnp.nonzero(unique, size=capacity, fill_value=keys.shape[0] - 1)[0]
    result = {
        name: jnp.concatenate((current[name], packet[name]))[order[pick]]
        for name in current
    }
    result["ids"] = jnp.where(jnp.arange(capacity) < count, ordered[pick], _SENTINEL)
    duplicate = (ordered[1:] == ordered[:-1]) & (ordered[1:] != _SENTINEL)
    conflict = jnp.asarray(False)
    for name in current:
        if name == "_source_rank":
            continue
        values = jnp.concatenate((current[name], packet[name]))[order]
        different = values[1:] != values[:-1]
        if different.ndim > 1:
            different = jnp.any(different, axis=tuple(range(1, different.ndim)))
        conflict |= jnp.any(duplicate & different)
    flags = jnp.where(count > capacity, _CAPACITY, 0) | _invalid_evidence(
        conflict, SimplexRestartInvalidReason.DUPLICATE_RECORD_CONFLICT
    )
    return result, flags if diagnostic else flags & 255


def _empty(packet: dict[str, Array], capacity: int) -> dict[str, Array]:
    return {
        name: jnp.full((capacity, *value.shape[1:]), _SENTINEL, jnp.int64)
        if name == "ids"
        else jnp.zeros((capacity, *value.shape[1:]), value.dtype)
        for name, value in packet.items()
    }


def packet(
    raw: AdaptiveSimplexState,
    cells: tuple[Array, ...],
    vertices: tuple[Array, ...],
    *,
    diagnostic: bool = False,
) -> tuple[dict[str, Array], dict[str, Array], Array]:
    mesh = raw.mesh
    c, v = mesh.cell_ids.shape[0], mesh.vertex_ids.shape[0]
    cell = {
        "ids": mesh.cell_ids,
        "cells": ids(mesh.vertex_ids, mesh.cells),
        "tuples": ids(mesh.vertex_ids, raw.tuples),
        "active": mesh.cell_active,
        "parents": ids(mesh.cell_ids, raw.parents // 2),
        "ordinal": raw.parents % 2,
        "children": ids(mesh.cell_ids, raw.children),
        "bisection": ids(mesh.vertex_ids, raw.bisection_vertices),
        **forest_cell_arrays(raw),
        **{f"payload_{i}": value for i, value in enumerate(cells)},
    }
    vertex = {
        "ids": mesh.vertex_ids,
        "coordinates": mesh.coordinates,
        "active": mesh.vertex_active,
        "parents": ids(mesh.vertex_ids, raw.vertex_parents),
        **forest_vertex_arrays(raw),
        **{f"payload_{i}": value for i, value in enumerate(vertices)},
    }
    reason = SimplexRestartInvalidReason
    invalid = _invalid_evidence(
        jnp.any((mesh.cell_ids >= 0) & ((raw.parents < -1) | (raw.parents >= 2 * c))),
        reason.CELL_PARENT_RANGE,
    )
    for references, bit in (
        (mesh.cells, reason.CELL_VERTEX_RANGE),
        (raw.tuples, reason.TUPLE_VERTEX_RANGE),
        (raw.bisection_vertices, reason.BISECTION_VERTEX_RANGE),
    ):
        mask = (mesh.cell_ids >= 0).reshape((c,) + (1,) * (references.ndim - 1))
        invalid |= _invalid_evidence(
            jnp.any(mask & ((references < -1) | (references >= v))), bit
        )
    invalid |= _invalid_evidence(
        jnp.any((mesh.cell_ids >= 0)[:, None] & ((raw.children < -1) | (raw.children >= c))),
        reason.CELL_CHILD_RANGE,
    )
    invalid |= _invalid_evidence(
        jnp.any((mesh.vertex_ids >= 0)[:, None]
                & ((raw.vertex_parents < -1) | (raw.vertex_parents >= v))),
        reason.VERTEX_PARENT_RANGE,
    )
    allocated = mesh.cell_ids >= 0
    invalid |= _invalid_evidence(
        jnp.any(allocated[:, None] & ((mesh.cells < 0) | (raw.tuples < 0))),
        reason.REQUIRED_VERTEX_SLOT,
    )
    invalid |= _invalid_evidence(
        jnp.any(allocated[:, None] & ((cell["cells"] < 0) | (cell["tuples"] < 0))),
        reason.REQUIRED_VERTEX_ID,
    )
    for source_slots, references, bit in (
        (raw.parents // 2, cell["parents"], reason.CELL_PARENT_ID),
        (raw.children, cell["children"], reason.CELL_CHILD_ID),
        (raw.bisection_vertices, cell["bisection"], reason.BISECTION_VERTEX_ID),
    ):
        mask = allocated.reshape((c,) + (1,) * (source_slots.ndim - 1))
        invalid |= _invalid_evidence(
            jnp.any(mask & (source_slots >= 0) & (references < 0)), bit
        )
    invalid |= _invalid_evidence(
        jnp.any((mesh.vertex_ids >= 0)[:, None]
                & (raw.vertex_parents >= 0) & (vertex["parents"] < 0)),
        reason.VERTEX_PARENT_ID,
    )
    protected = raw.protected_codes != _SENTINEL
    endpoints = jnp.stack((raw.protected_codes // v, raw.protected_codes % v), axis=1)
    invalid |= _invalid_evidence(
        jnp.any(protected[:, None]
                & (ids(mesh.vertex_ids, jnp.clip(endpoints, 0, v - 1)) < 0)),
        reason.PROTECTED_ENDPOINT_ID,
    )
    invalid |= _invalid_evidence(
        jnp.any((mesh.vertex_ids >= 0) & ~jnp.all(jnp.isfinite(mesh.coordinates), axis=1)),
        reason.NONFINITE_COORDINATE,
    )
    invalid |= _invalid_evidence(
        jnp.any((raw.protected_codes != _SENTINEL)
                & ((raw.protected_codes < 0) | (raw.protected_codes >= v * v))),
        reason.PROTECTED_CODE_RANGE,
    )
    invalid |= _invalid_evidence(
        (raw.cursors[2] <= jnp.max(mesh.vertex_ids))
        | (raw.cursors[3] <= jnp.max(mesh.cell_ids)),
        reason.ALLOCATOR_CURSOR,
    )
    return cell, vertex, invalid if diagnostic else invalid & 255


def missing(references: Array, identifiers: Array, valid: Array) -> Array:
    keys = jnp.where(identifiers >= 0, identifiers, _SENTINEL)
    pos = jnp.minimum(jnp.searchsorted(keys, references), identifiers.shape[0] - 1)
    mask = valid.reshape((valid.shape[0],) + (1,) * (references.ndim - 1))
    return jnp.any(mask & (references >= 0) & (identifiers[pos] != references))


def _source_vertex_activity(identifiers: Array, active: Array) -> Array:
    """Union partition-local activity by ID, never by storage slot or rank.

    The caller must authenticate the complete owner/halo inventory first.
    Saved masks remain untouched as the diagnostic source evidence.
    """
    if active.shape != identifiers.shape or active.dtype != jnp.bool_:
        raise ValueError("restart vertex active must be Boolean with the source ID shape")
    keys = jnp.where(identifiers >= 0, identifiers, _SENTINEL).reshape(-1)
    order = jnp.argsort(keys, stable=True)
    ordered = keys[order]
    groups = jnp.cumsum(
        jnp.concatenate((jnp.ones(1, jnp.int32), (ordered[1:] != ordered[:-1]).astype(jnp.int32)))
    ) - 1
    union = jnp.zeros(keys.shape, jnp.bool_).at[groups].max(active.reshape(-1)[order])
    restored = jnp.zeros(keys.shape, jnp.bool_).at[order].set(union[groups])
    return (restored.reshape(identifiers.shape) & (identifiers >= 0))


def _source_vertex_removal(
    old: AdaptiveSimplexState, authoritative: Array, active: Array,
) -> tuple[Array, Array]:
    """Canonical partition removal events on the cumulative coarsening clock.

    Event stamps precede the completed-pass clock. Equal stamps from different
    owners denote the same ordering position, so their scalar result is identical.
    """
    identifiers, removal = old.mesh.vertex_ids, old.vertex_removal
    present = identifiers >= 0
    clocks = old.clocks[:, 1, None]
    invalid = jnp.any(
        present & (
            (removal < -1) | ((removal >= 0) & (removal >= clocks))
            | (authoritative & old.mesh.vertex_active & (removal >= 0))
        )
    ) | jnp.any(clocks < 0)
    keys = jnp.where(present, identifiers, _SENTINEL).reshape(-1)
    order = jnp.argsort(keys, stable=True)
    ordered = keys[order]
    groups = jnp.cumsum(
        jnp.concatenate((jnp.ones(1, jnp.int32), (ordered[1:] != ordered[:-1]).astype(jnp.int32)))
    ) - 1
    absent = jnp.iinfo(removal.dtype).max
    events = jnp.where(present & (removal >= 0), removal, absent).reshape(-1)[order]
    earliest = jnp.full(keys.shape, absent, removal.dtype).at[groups].min(events)
    restored = jnp.full(keys.shape, absent, removal.dtype).at[order].set(earliest[groups])
    restored = restored.reshape(identifiers.shape)
    invalid |= jnp.any(present & ~active & (restored == absent))
    return jnp.where(present & ~active, restored, -1), invalid


@eqx.filter_jit
def _repack(
    layout: AdaptiveSimplexLayout,
    count: int,
    old: AdaptiveSimplexState,
    cell_owners: Array,
    vertex_owners: Array,
    requested: Array,
    ch: tuple[Array, ...],
    vh: tuple[Array, ...],
) -> tuple[
    AdaptiveSimplexState, Array, Array, tuple[Array, ...], tuple[Array, ...], Array, Array, Array
]:
    old_count = old.mesh.cell_ids.shape[0]
    ranks = jnp.arange(old_count, dtype=jnp.int32)[:, None]
    authoritative = (old.mesh.cell_ids >= 0) & (cell_owners == ranks)
    vertex_authoritative = (old.mesh.vertex_ids >= 0) & (vertex_owners == ranks)

    def complete_authority(identifiers: Array, owners: Array, mask: Array) -> Array:
        order = jnp.argsort(
            jnp.where(mask, identifiers, _SENTINEL).reshape(-1), stable=True
        )
        keys = jnp.where(mask, identifiers, _SENTINEL).reshape(-1)[order]
        source_rank = jnp.broadcast_to(ranks, identifiers.shape).reshape(-1)[order]
        position = jnp.minimum(jnp.searchsorted(keys, identifiers), keys.shape[0] - 1)
        unique = jnp.all((keys[1:] != keys[:-1]) | (keys[1:] == _SENTINEL))
        covered = jnp.all(
            (identifiers < 0)
            | ((keys[position] == identifiers) & (source_rank[position] == owners))
        )
        return unique & covered

    reason = SimplexRestartInvalidReason
    invalid = _invalid_evidence(
        ~complete_authority(old.mesh.cell_ids, cell_owners, authoritative),
        reason.CELL_AUTHORITY,
    )
    invalid |= _invalid_evidence(
        ~complete_authority(old.mesh.vertex_ids, vertex_owners, vertex_authoritative),
        reason.VERTEX_AUTHORITY,
    )
    invalid |= _invalid_evidence(
        jnp.any((old.mesh.cell_ids >= 0) & ((cell_owners < 0) | (cell_owners >= old_count))),
        reason.CELL_OWNER_RANGE,
    )
    invalid |= _invalid_evidence(
        jnp.any((old.mesh.vertex_ids >= 0) & ((vertex_owners < 0) | (vertex_owners >= old_count))),
        reason.VERTEX_OWNER_RANGE,
    )
    invalid |= _invalid_evidence(jnp.any(old.clocks[:, 2] != 0), reason.SAVED_STATUS)
    invalid |= _invalid_evidence(
        jnp.any(old.mesh.vertex_active & (old.mesh.vertex_ids < 0)),
        reason.REQUIRED_VERTEX_ID,
    )
    # Only partition metadata is normalized. Owner coverage/range failures above
    # remain collective refusals; no unvalidated union can become a receipt.
    source_vertex_activity = _source_vertex_activity(
        old.mesh.vertex_ids, old.mesh.vertex_active
    )
    source_vertex_removal, removal_invalid = _source_vertex_removal(
        old, vertex_authoritative, source_vertex_activity,
    )
    invalid |= _invalid_evidence(removal_invalid, reason.VERTEX_REMOVAL_HISTORY)

    def local(
        rank: Array,
    ) -> tuple[
        AdaptiveSimplexState, Array, Array, tuple[Array, ...], tuple[Array, ...], Array, Array
    ]:
        first = jax.tree_util.tree_map(lambda value: value[0], old)
        cell, vertex, _ = packet(
            first, tuple(value[0] for value in ch), tuple(value[0] for value in vh)
        )
        cell["_source_rank"] = jnp.zeros(cell["ids"].shape, jnp.int32)
        vertex["_source_rank"] = jnp.zeros(vertex["ids"].shape, jnp.int32)
        resident = _empty(cell, layout.cell_capacity)
        vertices = _empty(vertex, layout.vertex_capacity)
        status = invalid
        conflict = jnp.full((8,), -1, jnp.int64)

        def cell_visit(
            index: int, carry: tuple[dict[str, Array], Array, Array]
        ) -> tuple[dict[str, Array], Array, Array]:
            resident, status, conflict = carry
            raw = jax.tree_util.tree_map(lambda value: value[index], old)
            local_packet, _, flags = packet(
                raw,
                tuple(value[index] for value in ch),
                tuple(value[index] for value in vh),
                diagnostic=True,
            )
            roots, valid = validated_forest_roots(raw)
            target = jnp.full(raw.mesh.cell_ids.shape, count, jnp.int32)
            proposal = requested[index]
            active = raw.mesh.cell_active & (cell_owners[index] == index)
            status |= flags | _invalid_evidence(~valid, reason.FOREST_ROOT)
            status |= _invalid_evidence(
                jnp.any(active & ((proposal < 0) | (proposal >= count))),
                reason.PROPOSED_OWNER_RANGE,
            )
            target = target.at[roots].min(jnp.where(active, proposal, count))[roots]
            target = jnp.where(target < count, target, raw.mesh.cell_ids[roots] % count)
            selected = (
                (raw.mesh.cell_ids >= 0)
                & (cell_owners[index] == index)
                & (target == rank)
            )
            local_packet["_source_rank"] = jnp.full(local_packet["ids"].shape, index, jnp.int32)
            conflict = _first_conflict(
                conflict, _packet_conflict(resident, local_packet, selected, SimplexRestartRecordRole.CELL)
            )
            resident, flags = _merge_packet(resident, local_packet, selected, diagnostic=True)
            return resident, status | flags, conflict

        resident, status, conflict = jax.lax.fori_loop(
            0, old_count, cell_visit, (resident, status, conflict)
        )
        used = resident["ids"] != _SENTINEL
        required = jnp.concatenate(
            (
                jnp.where(used[:, None], resident["cells"], -1).reshape(-1),
                jnp.where(used[:, None], resident["tuples"], -1).reshape(-1),
                jnp.where(used, resident["bisection"], -1),
            )
        )

        def vertex_round(
            carry: tuple[dict[str, Array], Array, Array, Array, Array],
        ) -> tuple[dict[str, Array], Array, Array, Array, Array]:
            vertices, status, iteration, _, conflict = carry
            before = vertices["ids"]
            refs = jnp.sort(
                jnp.concatenate(
                    (
                        required,
                        jnp.where(
                            (before != _SENTINEL)[:, None], vertices["parents"], -1
                        ).reshape(-1),
                        jnp.where(before != _SENTINEL, before, -1),
                    )
                )
            )

            def visit(
                index: int, carry: tuple[dict[str, Array], Array, Array]
            ) -> tuple[dict[str, Array], Array, Array]:
                result, status, conflict = carry
                raw = jax.tree_util.tree_map(lambda value: value[index], old)
                _, local_packet, flags = packet(
                    raw,
                    tuple(value[index] for value in ch),
                    tuple(value[index] for value in vh),
                    diagnostic=True,
                )
                v = raw.mesh.vertex_ids.shape[0]
                codes = raw.protected_codes
                edges = jnp.stack((codes // v, codes % v), axis=1)
                edges = jnp.where(
                    (codes != _SENTINEL)[:, None],
                    ids(raw.mesh.vertex_ids, jnp.clip(edges, 0, v - 1)),
                    -1,
                )
                touching = jnp.any(members(refs, edges), axis=1)
                edge_refs = jnp.sort(jnp.where(touching[:, None], edges, -1).reshape(-1))
                selected = (local_packet["ids"] >= 0) & (
                    members(refs, local_packet["ids"])
                    | members(edge_refs, local_packet["ids"])
                    | (local_packet["ids"] % count == rank)
                )
                local_packet["active"] = source_vertex_activity[index]
                local_packet["vertex_removal"] = source_vertex_removal[index]
                local_packet["_source_rank"] = jnp.full(local_packet["ids"].shape, index, jnp.int32)
                conflict = _first_conflict(
                    conflict, _packet_conflict(result, local_packet, selected, SimplexRestartRecordRole.VERTEX)
                )
                result, merged_status = _merge_packet(
                    result, local_packet, selected, diagnostic=True
                )
                return result, status | flags | merged_status, conflict

            vertices, status, conflict = jax.lax.fori_loop(
                0, old_count, visit, (vertices, status, conflict)
            )
            return vertices, status, iteration + 1, jnp.any(vertices["ids"] != before), conflict

        vertices, status, _, changed, conflict = jax.lax.while_loop(
            lambda carry: carry[3] & (carry[2] <= old.mesh.vertex_ids.size),
            vertex_round,
            (vertices, status, jnp.asarray(0, jnp.int32), jnp.asarray(True), conflict),
        )
        status |= _invalid_evidence(changed, reason.VERTEX_CLOSURE_LIMIT)
        cids = jnp.where(used, resident["ids"], -1)
        vids = jnp.where(vertices["ids"] != _SENTINEL, vertices["ids"], -1)
        for name, bit in (
            ("cells", reason.MISSING_CELL_VERTEX),
            ("tuples", reason.MISSING_TUPLE_VERTEX),
            ("bisection", reason.MISSING_BISECTION_VERTEX),
        ):
            status |= _invalid_evidence(missing(resident[name], vids, used), bit)
        for name, bit in (
            ("parents", reason.MISSING_CELL_PARENT),
            ("children", reason.MISSING_CELL_CHILD),
        ):
            status |= _invalid_evidence(missing(resident[name], cids, used), bit)
        status |= _invalid_evidence(
            missing(vertices["parents"], vids, vids >= 0), reason.MISSING_VERTEX_PARENT
        )
        rows = jnp.maximum(slots(vids, resident["cells"]), 0)
        active = used & resident["active"]
        neighbors = masked_simplex_facet_neighbors(rows, active)
        mesh = MaskedSimplexMesh(
            vertices["coordinates"],
            vids,
            (vids >= 0) & vertices["active"],
            rows,
            cids,
            active,
            neighbors,
        )
        codes = jnp.full((layout.protected_edge_capacity,), _SENTINEL, jnp.int64)

        def constraints(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            codes, status = carry
            raw = jax.tree_util.tree_map(lambda value: value[index], old)
            v = raw.mesh.vertex_ids.shape[0]
            edges = jnp.stack((raw.protected_codes // v, raw.protected_codes % v), axis=1)
            edges = jnp.where(
                (raw.protected_codes != _SENTINEL)[:, None],
                ids(raw.mesh.vertex_ids, jnp.clip(edges, 0, v - 1)),
                -1,
            )
            present = jnp.all(
                members(jnp.where(vids >= 0, vids, _SENTINEL), edges), axis=1
            )
            local_slots = slots(vids, edges).astype(jnp.int64)
            incoming = jnp.where(
                present,
                jnp.min(local_slots, axis=1) * layout.vertex_capacity
                + jnp.max(local_slots, axis=1),
                _SENTINEL,
            )
            ordered = jnp.sort(jnp.concatenate((codes, incoming)))
            fresh = (ordered != _SENTINEL) & jnp.concatenate(
                (jnp.ones(1, jnp.bool_), ordered[1:] != ordered[:-1])
            )
            pick = jnp.nonzero(
                fresh, size=codes.shape[0], fill_value=ordered.shape[0] - 1
            )[0]
            n = jnp.sum(fresh, dtype=jnp.int32)
            return jnp.where(
                jnp.arange(codes.shape[0]) < n, ordered[pick], _SENTINEL
            ), status | jnp.where(n > codes.shape[0], _CAPACITY, 0)

        codes, status = jax.lax.fori_loop(0, old_count, constraints, (codes, status))
        parent = slots(cids, resident["parents"])

        def accumulate(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            counters, status = carry
            incoming = jnp.where(index % count == rank, old.counters[index], 0)
            overflow = jnp.any((incoming < 0) | (counters > _SENTINEL - incoming))
            return counters + incoming, status | jnp.where(overflow, _CAPACITY, 0)

        counters, status = jax.lax.fori_loop(
            0,
            old_count,
            accumulate,
            (jnp.zeros(old.counters.shape[1:], jnp.int64), status),
        )
        clocks = jnp.max(old.clocks, axis=0).at[2].set(status & 255)
        state = AdaptiveSimplexState(
            mesh,
            vertex_half_facets=vertex_half_facets(
                rows, active, neighbors, layout.vertex_capacity
            ),
            tuples=jnp.maximum(slots(vids, resident["tuples"]), 0),
            parents=jnp.where(parent >= 0, parent * 2 + resident["ordinal"], -1),
            children=slots(cids, resident["children"]),
            bisection_vertices=slots(vids, resident["bisection"]),
            vertex_parents=slots(vids, vertices["parents"]),
            protected_codes=codes,
            cursors=jnp.stack(
                (
                    jnp.sum(vids >= 0, dtype=jnp.int64),
                    jnp.sum(used, dtype=jnp.int64),
                    jnp.max(old.cursors[:, 2]),
                    jnp.max(old.cursors[:, 3]),
                )
            ),
            clocks=clocks,
            counters=counters,
            tags=resident["tags"],
            blocks=resident["blocks"],
            generations=resident["generations"],
            retired=resident["retired"],
            cell_classes=resident["cell_classes"],
            facet_classes=resident["facet_classes"],
            refine_rejected=resident["refine_rejected"],
            coarsen_marked=resident["coarsen_marked"],
            vertex_levels=vertices["vertex_levels"],
            vertex_removal=vertices["vertex_removal"],
            vertex_protected=vertices["vertex_protected"],
        )
        return (
            state,
            jnp.where(used, rank, -1).astype(jnp.int32),
            jnp.where(vids >= 0, vids % count, -1).astype(jnp.int32),
            tuple(resident[f"payload_{i}"] for i in range(len(ch))),
            tuple(vertices[f"payload_{i}"] for i in range(len(vh))),
            status,
            conflict,
        )

    states, co, vo, ch, vh, status, conflicts = jax.vmap(local)(jnp.arange(count, dtype=jnp.int32))
    before = jnp.sum(authoritative, dtype=jnp.int64)
    after = jnp.sum(states.mesh.cell_ids >= 0, dtype=jnp.int64)
    vb = jnp.sum(vertex_authoritative, dtype=jnp.int64)
    va = jnp.sum(
        (states.mesh.vertex_ids >= 0)
        & (vo == jnp.arange(count, dtype=jnp.int32)[:, None]),
        dtype=jnp.int64,
    )
    count_flags = _invalid_evidence(before != after, reason.CELL_COUNT)
    count_flags |= _invalid_evidence(vb != va, reason.VERTEX_COUNT)
    reason_masks = (status | count_flags) >> 8
    verdict = (jnp.bitwise_or.reduce(status) | count_flags) & 255
    status = jnp.full((count,), verdict, jnp.int32)
    states = eqx.tree_at(
        lambda value: value.clocks, states, states.clocks.at[:, 2].set(verdict)
    )
    return states, co, vo, ch, vh, status, reason_masks, conflicts


@checked
def repack_simplex_restart(
    target_group: ExecutionGroup,
    layout: AdaptiveSimplexLayout,
    saved: AdaptiveSimplexState,
    cell_owners: Array,
    vertex_owners: Array,
    requested_target_owners: Array,
    /,
    *,
    cell_history: tuple[Array, ...] = (),
    vertex_history: tuple[Array, ...] = (),
    source_result: CellMeshingResult | None = None,
    graph_proposal: migration_owner.SimplexGraphRestartProposal | None = None,
    route_metadata_capacity_bytes: int = 1 << 20,
) -> SimplexRestartRepack:
    """Reassemble all saved logical records on the target execution devices.

    ``requested_target_owners`` is the placement proposal in saved raw-cell ID
    order. When it comes from the native GRAPH route, ``graph_proposal`` binds
    the executed partition and must hold this proposal; restart
    validation replays it. A family stays together; its smallest active proposed
    owner wins.
    Entirely inactive families use their root scientific ID. Cumulative counters
    retain their total, clocks use their global maximum and allocator ID cursors
    retain their maximum. Saved rank controls are also returned unchanged.
    Any numerical refusal applies collectively; no result acceptance is implied.
    """
    if len(target_group.mesh.axis_names) != 1 or not target_group.is_member:
        raise ValueError(
            "restart.target_group must have one axis and include this process; got incompatible execution membership"
        )
    old_shape = saved.mesh.cell_ids.shape
    if (
        len(old_shape) != 2
        or cell_owners.shape != old_shape
        or requested_target_owners.shape != old_shape
        or vertex_owners.shape != saved.mesh.vertex_ids.shape
    ):
        raise ValueError(
            "restart.saved_owners must align with saved raw capacities; got incompatible owner or proposal shapes"
        )
    if (
        cell_owners.dtype != jnp.int32
        or vertex_owners.dtype != jnp.int32
        or requested_target_owners.dtype != jnp.int32
    ):
        raise TypeError(
            "restart.requested_target_owners must use int32 with raw owners; got incompatible dtypes"
        )
    if (
        saved.mesh.cells.shape[-1] - 1 != layout.dimension
        or saved.mesh.ambient_dimension != layout.ambient_dimension
        or saved.mesh.coordinates.dtype.name != layout.coordinate_dtype
    ):
        raise ValueError(
            "restart.layout must preserve saved geometry and coordinate dtype; got incompatible coordinate layout"
        )
    if any(value.shape[:2] != old_shape for value in cell_history) or any(
        value.shape[:2] != saved.mesh.vertex_ids.shape for value in vertex_history
    ):
        raise ValueError(
            "restart.saved_history must bind every allocated raw record; got incompatible history shapes"
        )
    if (
        graph_proposal is not None
        and graph_proposal.requested_target_owners is not requested_target_owners
    ):
        raise ValueError(
            "restart.graph_proposal must bind its executed projected owners; got different target ownership"
        )
    axis = target_group.mesh.axis_names[0]
    count = len(target_group.devices)
    route_budget = operator.index(route_metadata_capacity_bytes)
    if isinstance(route_metadata_capacity_bytes, bool) or route_budget < count * count:
        raise ValueError(
            "restart.route_metadata_capacity_bytes must cover shared-vertex routing; got insufficient byte capacity"
        )
    parts = AdaptiveSimplexParts(target_group.devices, axis_name=axis, neighbor_pairs=())
    output = _repack(
        layout,
        count,
        saved,
        cell_owners,
        vertex_owners,
        requested_target_owners,
        cell_history,
        vertex_history,
    )
    logical_states = output[0]
    cell_locations = _saved_locations(
        saved.mesh.cell_ids, cell_owners, logical_states.mesh.cell_ids
    )
    vertex_locations = _saved_locations(
        saved.mesh.vertex_ids, vertex_owners, logical_states.mesh.vertex_ids
    )
    solver_owners = _restart_solver_owners(
        logical_states.mesh.cell_active, cell_locations, requested_target_owners
    )
    # Correspondence math runs in the saved logical array placement;
    # the single explicit lowering then moves all new-slot records together.
    output, cell_locations, vertex_locations, solver_owners = jax.tree_util.tree_map(
        lambda value: jax.device_put(
            value, NamedSharding(parts.mesh, PartitionSpec(axis))
        ),
        (output, cell_locations, vertex_locations, solver_owners),
    )
    states, co, vo, ch, vh, status, reason_masks, conflicts = output
    routed_parts = migration_owner.prepare_migration_routes(
        parts, migration_owner.forest_migration_routes(parts, states)
    )
    if not isinstance(routed_parts, AdaptiveSimplexParts):
        raise ValueError("restart.parts must retain actual live execution placement.")
    parts = routed_parts
    ranks = jnp.arange(old_shape[0], dtype=jnp.int32)[:, None]
    new_ranks = jnp.arange(count, dtype=jnp.int32)[:, None]
    saved_solver_owners = (
        None if source_result is None else _source_solver_owners(source_result, saved)
    )
    traffic = (
        None
        if saved_solver_owners is None
        else _restart_cell_traffic(
            saved,
            cell_owners,
            saved_solver_owners,
            requested_target_owners,
            count,
        )
    )
    return SimplexRestartRepack(
        parts=parts,
        layout=layout,
        states=states,
        cell_owners=co,
        vertex_owners=vo,
        cell_history=ch,
        vertex_history=vh,
        status=status,
        invalid_reason_masks=reason_masks,
        duplicate_record_conflicts=conflicts,
        cells_before=jnp.sum(
            (saved.mesh.cell_ids >= 0) & (cell_owners == ranks), axis=1, dtype=jnp.int64
        ),
        cells_after=jnp.sum(states.mesh.cell_ids >= 0, axis=1, dtype=jnp.int64),
        vertices_before=jnp.sum(
            (saved.mesh.vertex_ids >= 0) & (vertex_owners == ranks),
            axis=1,
            dtype=jnp.int64,
        ),
        vertices_after=jnp.sum(
            (states.mesh.vertex_ids >= 0) & (vo == new_ranks), axis=1, dtype=jnp.int64
        ),
        saved_cursors=saved.cursors,
        saved_clocks=saved.clocks,
        saved_counters=saved.counters,
        saved_states=saved,
        saved_cell_owners=cell_owners,
        saved_vertex_owners=vertex_owners,
        source_result=source_result,
        cell_saved_locations=cell_locations,
        vertex_saved_locations=vertex_locations,
        requested_target_owners=requested_target_owners,
        saved_cell_history=cell_history,
        saved_vertex_history=vertex_history,
        solver_cell_owners=solver_owners,
        saved_solver_cell_owners=saved_solver_owners,
        cell_migration_counts=traffic,
        graph_proposal=graph_proposal,
    )
