#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared bounded streamed execution of nonlinear additive sparse relations.

A streamed relation evaluates a pure per-event callback over bounded edge
fragments, accumulates its additive messages into seeded receiver states, and
applies a complete receiver epilogue once, after the receiver's final
fragment. Only the declared receiver outputs and explicitly requested edge
outputs escape a tile; edge messages, aggregates and epilogue internals are
tile-local workspace.

A fragment aggregator replaces the per-event callback when a consumer owns a
specialized whole-fragment accumulation (for example an accelerated kernel):
it receives one prepared `StreamedFragment` (receiver-slot and fragment-local
source routing, both prepared once per topology), the fragment's local edge
rows, the loop-invariant source rows and parameters, and the seeded
high/correction state of every receiver slot, and returns the updated seeded
state. The substrate still owns tiling, spill across fragments, the single
epilogue per receiver, replay, poisoning and output commits;
`StreamedFragment.reduce` is the canonical seeded additive reduction for any
per-lane values the aggregator does not accumulate itself.

The reference execution is ordinary JAX: forward, JVP, VJP, reverse-over-reverse
and forward-over-reverse transforms are JAX's own derivatives of the tiled
primal. Each tile body is rematerialized inside the shared checkpointed scan's
replay boundary (a nested checkpoint keeps replay through a second reverse
pass), so first-order reverse, force-loss parameter gradients and coordinate
HVPs retain one bounded spill carry per replay boundary while graph-wide
inputs and parameters remain loop-invariant scan constants. Transform memory
is certified by compiled-executable analysis, not by this structure.
"""

from __future__ import annotations

from math import ceil, prod
from types import ModuleType
from typing import assert_never, final, Literal, NamedTuple, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._model import register_artifact_value
from .._numerics._checkpointed_scan import checkpointed_scan, CheckpointedScanMode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import normalized_identifier
from ..typing import parse
from ._execution import KeyGroupAccumulation, reduce_key_groups, RelationAccumulation
from ._key_groups import KeyGroupEvidence, KeyGroupPlan, KeyGroupState
from ._relation import _traced, EdgeRelation, RowRelation, SparseRelation


StreamedDirection: TypeAlias = Literal["target", "source"]
"""Receiver role of a prepared streamed relation: relation targets or sources."""


class StreamedEdgeFunction(Protocol):
    """Pure per-event callback returning a declared message (and edge output).

    The callback sees one event: ``parameters`` and the source, receiver and
    edge rows of that event, without a leading event axis. It returns the
    declared message PyTree, or ``(message, edge_output)`` when the payload
    requests edge outputs. It must not depend on other events.
    """

    def __call__(
        self,
        parameters: PyTree[Array],
        source: PyTree[Array],
        receiver: PyTree[Array],
        edge: PyTree[Array],
        /,
    ) -> PyTree[Array]: ...


class StreamedReceiverEpilogue(Protocol):
    """Complete per-receiver epilogue over the receiver's additive aggregate."""

    def __call__(
        self,
        parameters: PyTree[Array],
        receiver: PyTree[Array],
        aggregate: PyTree[Array],
        /,
    ) -> PyTree[Array]: ...


class StreamedFragmentAggregator(Protocol):
    """Pure whole-fragment additive accumulation into seeded receiver slots.

    ``sources`` are the loop-invariant source rows (leading source axis, gather
    by ``fragment.source_ids`` or ``fragment.lane_sources``), ``receivers`` the
    rows of the fragment's receiver slots ``[receiver_tile, ...]`` and
    ``edges`` the fragment's lane rows ``[edge_tile, ...]``; unusable lanes
    carry a usable lane's edge row, so local geometry stays domain-safe, and
    must be masked by ``fragment.lane_use``. ``seed`` is the declared message
    PyTree with one `KeyGroupAccumulation` ``[receiver_tile, ...]`` per leaf.
    The aggregator returns the updated accumulations in the same structure
    (seed plus this fragment's usable lanes, under the plan's accumulation),
    or ``(accumulations, edge_output)`` with lane-major edge outputs when the
    payload requests them. It never sees a fragment without usable lanes.
    """

    def __call__(
        self,
        parameters: PyTree[Array],
        sources: PyTree[Array],
        receivers: PyTree[Array],
        edges: PyTree[Array],
        fragment: StreamedFragment,
        seed: PyTree[KeyGroupAccumulation],
        /,
    ) -> (
        PyTree[KeyGroupAccumulation] | tuple[PyTree[KeyGroupAccumulation], PyTree[Array]]
    ): ...


class _LeafSpecs(NamedTuple):
    tree: PyTreeDef
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[np.dtype, ...]
    records: tuple[tuple[str, tuple[int, ...], str], ...]


def _leaf_specs(name: str, tree: PyTree[jax.ShapeDtypeStruct], /) -> _LeafSpecs:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    if not path_leaves:
        raise ValueError(f"{name} must declare at least one array leaf.")
    shapes: list[tuple[int, ...]] = []
    dtypes: list[np.dtype] = []
    records: list[tuple[str, tuple[int, ...], str]] = []
    for path, leaf in path_leaves:
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            raise TypeError(f"{name} leaves must be jax.ShapeDtypeStruct values.")
        dtype = np.dtype(leaf.dtype)
        if dtype.kind not in "iufc":
            raise TypeError(f"{name} leaves must have numeric, non-boolean dtypes.")
        shape = tuple(leaf.shape)
        shapes.append(shape)
        dtypes.append(dtype)
        records.append((jax.tree_util.keystr(path) or "<root>", shape, dtype.name))
    return _LeafSpecs(treedef, tuple(shapes), tuple(dtypes), tuple(records))


def _elements(shapes: tuple[tuple[int, ...], ...], /) -> int:
    return sum(prod(shape) for shape in shapes)


def _bytes(shapes: tuple[tuple[int, ...], ...], dtypes: tuple[np.dtype, ...], /) -> int:
    return sum(
        prod(shape) * dtype.itemsize for shape, dtype in zip(shapes, dtypes, strict=True)
    )


def _conform(
    name: str,
    value: PyTree[Array],
    specs: tuple[PyTreeDef, tuple[tuple[int, ...], ...], tuple[np.dtype, ...]],
    leading: tuple[int, ...],
    /,
) -> list[Array]:
    """Admit traced callback results against their declared payload, never cast."""
    treedef, shapes, dtypes = specs
    leaves, actual = jax.tree_util.tree_flatten(value)
    if actual != treedef:
        raise TypeError(f"{name} returned structure {actual}; declared {treedef}.")
    arrays: list[Array] = []
    for leaf, shape, dtype in zip(leaves, shapes, dtypes, strict=True):
        array = jnp.asarray(leaf)
        if array.shape != leading + shape:
            raise ValueError(
                f"{name} returned shape {array.shape[len(leading) :]}; declared {shape}."
            )
        if array.dtype != dtype:
            raise TypeError(f"{name} returned dtype {array.dtype}; declared {dtype}.")
        arrays.append(array)
    return arrays


@final
class StreamedPayloadSpec(StrictModule):
    """Declared per-event message/edge output and per-receiver output payloads.

    Every leaf is a ``jax.ShapeDtypeStruct`` for ONE event or ONE receiver,
    without a leading axis. Messages are additive and accumulate in their
    declared dtype. Callbacks are never probed; traced results must match.
    """

    message_tree: PyTreeDef = eqx.field(static=True)
    message_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    message_dtypes: tuple[np.dtype, ...] = eqx.field(static=True)
    output_tree: PyTreeDef = eqx.field(static=True)
    output_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    output_dtypes: tuple[np.dtype, ...] = eqx.field(static=True)
    edge_output_tree: PyTreeDef | None = eqx.field(static=True)
    edge_output_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    edge_output_dtypes: tuple[np.dtype, ...] = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        message: PyTree[jax.ShapeDtypeStruct],
        output: PyTree[jax.ShapeDtypeStruct],
        edge_output: PyTree[jax.ShapeDtypeStruct] | None = None,
    ) -> None:
        message_specs = _leaf_specs("message", message)
        output_specs = _leaf_specs("output", output)
        edge_specs = (
            None if edge_output is None else _leaf_specs("edge_output", edge_output)
        )
        self.message_tree = message_specs.tree
        self.message_shapes = message_specs.shapes
        self.message_dtypes = message_specs.dtypes
        self.output_tree = output_specs.tree
        self.output_shapes = output_specs.shapes
        self.output_dtypes = output_specs.dtypes
        self.edge_output_tree = None if edge_specs is None else edge_specs.tree
        self.edge_output_shapes = () if edge_specs is None else edge_specs.shapes
        self.edge_output_dtypes = () if edge_specs is None else edge_specs.dtypes
        self.payload_id = canonical_fingerprint(
            {
                "type": "streamed-payload",
                "message": [list(record) for record in message_specs.records],
                "output": [list(record) for record in output_specs.records],
                "edge_output": None
                if edge_specs is None
                else [list(record) for record in edge_specs.records],
            }
        )

    @property
    def event_elements(self) -> int:
        """Elements of one event's message plus requested edge output."""
        return _elements(self.message_shapes) + _elements(self.edge_output_shapes)

    @property
    def output_elements(self) -> int:
        return _elements(self.output_shapes)

    @property
    def message_bytes(self) -> int:
        return _bytes(self.message_shapes, self.message_dtypes)

    @property
    def output_bytes(self) -> int:
        return _bytes(self.output_shapes, self.output_dtypes)

    @property
    def edge_output_bytes(self) -> int:
        return _bytes(self.edge_output_shapes, self.edge_output_dtypes)


def _positive(name: str, value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


@final
class StreamedRelationPlan(StrictModule):
    """Static tiling, channel cap, accumulation and replay of streamed relations.

    ``receiver_tile`` bounds the receivers owning epilogues in one tile and
    ``edge_tile`` the events of one fragment. A receiver whose events exceed
    the remaining fragment is split across consecutive tiles; its seeded
    high/correction accumulator is carried and its epilogue committed only at
    its final fragment. ``channel_capacity`` caps the elements of one event's
    message plus edge output and of one receiver output. ``replay`` selects the
    shared checkpointed-scan rematerialization of tile bodies; ``transpose``
    additionally prepares the independent source-major schedule.
    """

    receiver_tile: int = eqx.field(static=True)
    edge_tile: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    replay: CheckpointedScanMode = eqx.field(static=True)
    replay_block_size: int | None = eqx.field(static=True)
    transpose: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        receiver_tile: int = 256,
        edge_tile: int = 8192,
        channel_capacity: int = 4096,
        accumulation: RelationAccumulation = "fast",
        replay: CheckpointedScanMode = "step",
        replay_block_size: int | None = None,
        transpose: bool = False,
    ) -> None:
        receivers = _positive("receiver_tile", receiver_tile)
        edges = _positive("edge_tile", edge_tile)
        channels = _positive("channel_capacity", channel_capacity)
        accumulation = parse(accumulation, RelationAccumulation, "accumulation")
        replay = parse(replay, CheckpointedScanMode, "replay")
        match replay:
            case "block":
                if replay_block_size is None:
                    raise ValueError("Block replay requires replay_block_size.")
                block = _positive("replay_block_size", replay_block_size)
            case "step" | "full":
                if replay_block_size is not None:
                    raise ValueError("Only block replay accepts replay_block_size.")
                block = None
            case "scheduled":
                raise ValueError(
                    "Scheduled replay needs a schedule bound to the prepared tile "
                    "count; streamed relations admit full, step, or block replay."
                )
            case _:
                assert_never(replay)
        if not isinstance(transpose, bool):
            raise TypeError("transpose must be a bool.")
        self.receiver_tile = receivers
        self.edge_tile = edges
        self.channel_capacity = channels
        self.accumulation = accumulation
        self.replay = replay
        self.replay_block_size = block
        self.transpose = transpose
        self.plan_id = canonical_fingerprint(
            {
                "type": "streamed-relation-plan",
                "receiver_tile": receivers,
                "edge_tile": edges,
                "channel_capacity": channels,
                "accumulation": accumulation,
                "replay": replay,
                "replay_block_size": block,
                "transpose": transpose,
            }
        )

    def prepare(
        self,
        relation: SparseRelation,
        *,
        owner_id: str,
        epoch: ArrayLike = 0,
        stable_route_ids: ArrayLike | None = None,
        receiver_valid: ArrayLike | None = None,
        source_valid: ArrayLike | None = None,
    ) -> PreparedStreamedRelation:
        """Prepare receiver-major (and optionally source-major) tile schedules.

        Each schedule is built at effective widths ``R = min(receiver_tile,
        receivers of that direction)`` and ``F = min(edge_tile, route
        capacity)`` (each at least one), recorded on the schedule; the plan
        and its identity are unchanged. Works on concrete host topology (exact
        tile count, duplicate stable IDs refused) and on traced topology inside
        a compiled rebuild (static tile bound ``receivers // R + capacity // F
        + 1``, failures reported by evidence and poisoned outputs). Events into
        or from an invalid endpoint are absent. ``owner_id`` names the topology
        owner's schema; ``epoch`` is its dynamic topology epoch.
        """
        match relation:
            case RowRelation():
                edge = relation.as_edge_relation()
            case EdgeRelation():
                edge = relation
            case _:
                raise TypeError("relation must be an EdgeRelation or RowRelation.")
        owner = normalized_identifier(owner_id, "owner_id")
        route_shape = relation.route_shape
        with jax.ensure_compile_time_eval():
            ids = _route_ids(stable_route_ids, route_shape)
            receiver_mask = _endpoint_mask(
                "receiver_valid", receiver_valid, relation.output_shape
            )
            source_mask = _endpoint_mask(
                "source_valid", source_valid, relation.input_shape
            )
            epoch_array = jnp.asarray(epoch)
            if epoch_array.ndim != 0 or not jnp.issubdtype(
                epoch_array.dtype, jnp.integer
            ):
                raise ValueError("epoch must be an integer scalar.")
            # Requested tiles are upper bounds: a schedule is built at widths the
            # immutable topology can actually fill, so small relations carry no
            # padding. Widths depend only on static extents, never on route
            # content, so plan identity, event order and epoch reuse are kept.
            edge_width = max(min(self.edge_tile, edge.capacity), 1)
            target_schedule = _prepare_schedule(
                receivers=edge.target_indices,
                senders=edge.source_indices,
                valid=edge.valid,
                stable_ids=ids,
                receiver_valid=receiver_mask,
                sender_valid=source_mask,
                receiver_tile=max(min(self.receiver_tile, edge.target_size), 1),
                edge_tile=edge_width,
            )
            source_schedule = (
                _prepare_schedule(
                    receivers=edge.source_indices,
                    senders=edge.target_indices,
                    valid=edge.valid,
                    stable_ids=ids,
                    receiver_valid=source_mask,
                    sender_valid=receiver_mask,
                    receiver_tile=max(min(self.receiver_tile, edge.source_size), 1),
                    edge_tile=edge_width,
                )
                if self.transpose
                else None
            )
        topology = (edge.source_indices, edge.target_indices, edge.valid)
        content_id = (
            None
            if _traced(*topology, ids, receiver_mask, source_mask)
            else _content_id(relation, edge, ids, receiver_mask, source_mask)
        )
        binding = StreamedTopologyBinding(
            owner_id=owner,
            content_id=content_id,
            epoch=epoch_array.astype(jnp.int32),
        )
        return PreparedStreamedRelation(
            plan=self,
            schedule=target_schedule,
            transpose_schedule=source_schedule,
            binding=binding,
            input_shape=relation.input_shape,
            output_shape=relation.output_shape,
            route_shape=route_shape,
            direction="target",
        )


register_artifact_value("phydrax.sparse:StreamedRelationPlan", StreamedRelationPlan)
# Relations are registered here rather than in ``_relation``: that leaf module is
# imported before the model artifact registry during package initialization.
register_artifact_value("phydrax.sparse:EdgeRelation", EdgeRelation)
register_artifact_value("phydrax.sparse:RowRelation", RowRelation)


def _route_ids(value: ArrayLike | None, route_shape: tuple[int, ...], /) -> Array:
    count = prod(route_shape)
    if value is None:
        return jnp.arange(count, dtype=jnp.int32)
    ids = jnp.asarray(value)
    if not jnp.issubdtype(ids.dtype, jnp.integer):
        raise TypeError("stable_route_ids must have an integer dtype.")
    if ids.shape != route_shape:
        raise ValueError(
            f"stable_route_ids must have the route shape {route_shape}; got {ids.shape}."
        )
    return ids.reshape((count,))


def _endpoint_mask(
    name: str, value: ArrayLike | None, shape: tuple[int, ...], /
) -> Array:
    count = prod(shape)
    if value is None:
        return jnp.ones((count,), dtype=jnp.bool_)
    mask = jnp.asarray(value)
    if mask.dtype != jnp.bool_:
        raise TypeError(f"{name} must be a boolean mask.")
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {mask.shape}.")
    return mask.reshape((count,))


def _content_id(
    relation: SparseRelation,
    edge: EdgeRelation,
    ids: Array,
    receiver_mask: Array,
    source_mask: Array,
    /,
) -> str:
    source, target, valid, stable, receivers, sources = jax.device_get(
        (
            edge.source_indices,
            edge.target_indices,
            edge.valid,
            ids,
            receiver_mask,
            source_mask,
        )
    )
    return canonical_fingerprint(
        {
            "type": "streamed-topology",
            "relation": type(relation).__name__,
            "route_shape": list(relation.route_shape),
            "source_size": edge.source_size,
            "target_size": edge.target_size,
            "source_indices": np.where(valid, source, 0),
            "target_indices": np.where(valid, target, 0),
            "valid": valid,
            "stable_route_ids": stable,
            "receiver_valid": receivers,
            "source_valid": sources,
        }
    )


@final
class StreamedTopologyBinding(NonTrainableState, StrictModule):
    """Topology owner, host content identity and dynamic epoch of a schedule.

    ``content_id`` fingerprints concrete route indices, masks and stable IDs;
    it is ``None`` for traced topology, whose identity is the owner's schema
    and epoch. Equal shapes or capacities never imply the same topology.
    """

    epoch: Array
    owner_id: str = eqx.field(static=True)
    content_id: str | None = eqx.field(static=True)

    @property
    def binding_id(self) -> str:
        return canonical_fingerprint(
            {
                "type": "streamed-topology-binding",
                "owner_id": self.owner_id,
                "content_id": self.content_id,
            }
        )


@final
class StreamedSchedule(NonTrainableState, StrictModule):
    """Receiver-major bounded tiles of one direction of a sparse relation.

    Events are ordered by receiver and then stable route ID (``route_groups``
    holds the reversible canonical permutation and ``receiver_offsets`` the
    row offsets). Tile ``t`` owns receiver slots ``[t, :receiver_tile]`` and
    event lanes ``[t, :edge_tile]``. A receiver open at the end of tile ``t``
    (``open_out``) continues in slot zero of tile ``t + 1``
    (``continuation_in``); ``receiver_commit`` holds the receiver index at the
    receiver's final fragment and ``receiver_count`` (dropped) elsewhere.

    Fragment routing for whole-fragment aggregators is prepared with the
    tiles: lanes are receiver-slot major, so slot ``r`` owns lanes
    ``receiver_lane_offsets[t, r] + [0, receiver_lane_counts[t, r])``. The
    fragment's distinct sources are ``fragment_source_ids[t]`` (ascending,
    ``fragment_source_valid``; padding is source zero with no lanes);
    ``lane_local_sources`` maps each lane to its fragment source and
    ``source_lanes`` lists lanes source-major, with source ``s`` owning
    positions ``source_lane_offsets[t, s] + [0, source_lane_counts[t, s])``.
    """

    route_groups: KeyGroupState
    receiver_offsets: Array
    receiver_ids: Array
    receiver_slot_valid: Array
    receiver_commit: Array
    receiver_group_slots: Array
    receiver_has_group: Array
    continuation_in: Array
    open_out: Array
    open_slot: Array
    tile_active: Array
    lane_routes: Array
    lane_sources: Array
    lane_slots: Array
    lane_valid: Array
    receiver_lane_offsets: Array
    receiver_lane_counts: Array
    fragment_source_ids: Array
    fragment_source_valid: Array
    lane_local_sources: Array
    source_lanes: Array
    source_lane_offsets: Array
    source_lane_counts: Array
    tile_groups: KeyGroupState | None
    active_routes: Array
    maximum_receiver_degree: Array
    fragmented_receivers: Array
    tiles_used: Array
    duplicate_stable_ids: Array
    schedule_complete: Array
    successful: Array
    tile_count: int = eqx.field(static=True)
    receiver_count: int = eqx.field(static=True)
    sender_count: int = eqx.field(static=True)
    route_capacity: int = eqx.field(static=True)
    receiver_tile: int = eqx.field(static=True)
    edge_tile: int = eqx.field(static=True)

    @property
    def storage_bytes(self) -> int:
        """Bytes of every prepared schedule array (persistent topology)."""
        return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(self))


class _TileExtent[T: (Array, np.ndarray)](NamedTuple):
    """Tile extents in the array kind of the stream prefix (device or host)."""

    count: T
    begin: T
    end: T
    open_last: T
    next_receiver: T
    next_offset: T


def _tile_extent[T: (Array, np.ndarray)](
    xp: ModuleType,
    prefix: T,
    receiver: T | int,
    offset: T | int,
    receiver_tile: int,
    edge_tile: int,
    receiver_count: int,
    /,
) -> _TileExtent[T]:
    """Greedy fill of one tile from stream position ``(receiver, offset)``.

    A tile takes whole receivers until ``receiver_tile`` receivers complete or
    ``edge_tile`` events are consumed; the receiver crossing the event limit
    stays open. Every non-final tile therefore completes ``receiver_tile``
    receivers or holds exactly ``edge_tile`` events, so a relation of N
    receivers and E events needs at most ``N // R + E // F + 1`` tiles.
    """
    begin = xp.add(prefix[receiver], offset)
    limit = xp.minimum(receiver + receiver_tile, receiver_count)
    completed_end = prefix[limit]
    fits = completed_end - begin <= edge_tile
    split_end = begin + edge_tile
    split_receiver = xp.clip(
        xp.searchsorted(prefix, split_end, side="right") - 1, 0, receiver_count
    )
    split_offset = split_end - prefix[split_receiver]
    split_open = split_offset > 0
    return _TileExtent(
        count=xp.where(
            fits,
            limit - receiver,
            split_receiver - receiver + split_open.astype(np.int32),
        ),
        begin=begin,
        end=xp.where(fits, completed_end, split_end),
        open_last=xp.logical_and(xp.logical_not(fits), split_open),
        next_receiver=xp.where(fits, limit, split_receiver),
        next_offset=xp.where(fits, 0, split_offset),
    )


def _tile_starts(
    prefix: Array,
    receiver_count: int,
    route_capacity: int,
    receiver_tile: int,
    edge_tile: int,
    /,
) -> tuple[Array, Array, Array]:
    """Return tile start receivers/offsets and whether the stream completed."""
    if isinstance(prefix, jax_core.Tracer):
        bound = receiver_count // receiver_tile + route_capacity // edge_tile + 1

        def step(
            state: tuple[Array, Array], _: None
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            receiver, offset = state
            extent = _tile_extent(
                jnp, prefix, receiver, offset, receiver_tile, edge_tile, receiver_count
            )
            return (
                (
                    extent.next_receiver.astype(jnp.int32),
                    extent.next_offset.astype(jnp.int32),
                ),
                state,
            )

        zero = jnp.zeros((), dtype=jnp.int32)
        (final_receiver, _), (starts, offsets) = jax.lax.scan(
            step, (zero, zero), None, length=bound
        )
        return starts, offsets, final_receiver == receiver_count
    # Concrete topology: the exact greedy tile sequence is host preparation.
    host_prefix = np.asarray(jax.device_get(prefix), dtype=np.int64)
    start_receivers: list[int] = []
    start_offsets: list[int] = []
    receiver, offset = 0, 0
    while receiver < receiver_count:
        start_receivers.append(receiver)
        start_offsets.append(offset)
        extent = _tile_extent(
            np, host_prefix, receiver, offset, receiver_tile, edge_tile, receiver_count
        )
        receiver, offset = int(extent.next_receiver), int(extent.next_offset)
    return (
        jnp.asarray(np.asarray(start_receivers, dtype=np.int32)),
        jnp.asarray(np.asarray(start_offsets, dtype=np.int32)),
        jnp.asarray(True),
    )


def _prepare_schedule(
    *,
    receivers: Array,
    senders: Array,
    valid: Array,
    stable_ids: Array,
    receiver_valid: Array,
    sender_valid: Array,
    receiver_tile: int,
    edge_tile: int,
) -> StreamedSchedule:
    route_capacity = receivers.shape[0]
    receiver_count = receiver_valid.shape[0]
    sender_count = sender_valid.shape[0]
    safe_receivers = jnp.where(valid, receivers, 0)
    safe_senders = jnp.where(valid, senders, 0)
    active = valid
    if route_capacity:
        active = valid & receiver_valid[safe_receivers] & sender_valid[safe_senders]
    route_groups = KeyGroupPlan(
        route_capacity,
        max(min(route_capacity, receiver_count), 1),
        max(receiver_count - 1, 0),
    ).build(safe_receivers, active, stable_ids=stable_ids)
    degree = (
        jnp.zeros((receiver_count,), dtype=jnp.int32)
        .at[safe_receivers]
        .add(active.astype(jnp.int32))
    )
    prefix = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(degree, dtype=jnp.int32))
    )
    if receiver_count == 0:
        starts = jnp.zeros((0,), dtype=jnp.int32)
        offsets = jnp.zeros((0,), dtype=jnp.int32)
        complete = jnp.asarray(True)
    else:
        starts, offsets, complete = _tile_starts(
            prefix, receiver_count, route_capacity, receiver_tile, edge_tile
        )
    extent = _tile_extent(
        jnp, prefix, starts, offsets, receiver_tile, edge_tile, receiver_count
    )
    tile_count = starts.shape[0]
    tile_active = starts < receiver_count

    slots = jnp.arange(receiver_tile, dtype=jnp.int32)
    slot_valid = slots[None, :] < extent.count[:, None]
    first_receiver = jnp.minimum(starts, max(receiver_count - 1, 0))
    receiver_ids = jnp.where(
        slot_valid, starts[:, None] + slots[None, :], first_receiver[:, None]
    )
    open_slot = jnp.maximum(extent.count - 1, 0)
    last_open = extent.open_last[:, None] & (slots[None, :] == open_slot[:, None])
    commit = slot_valid & ~last_open
    if receiver_count:
        commit = commit & receiver_valid[receiver_ids]
    receiver_commit = jnp.where(commit, receiver_ids, receiver_count)

    lanes = jnp.arange(edge_tile, dtype=jnp.int32)
    position = extent.begin[:, None] + lanes[None, :]
    lane_valid = position < extent.end[:, None]
    if route_capacity:
        route = route_groups.storage_to_logical[jnp.clip(position, 0, route_capacity - 1)]
        lane_routes = jnp.where(lane_valid, route, 0)
        lane_sources = jnp.where(lane_valid, safe_senders[route], 0)
        lane_slots = jnp.where(lane_valid, safe_receivers[route] - starts[:, None], 0)
    else:
        lane_routes = jnp.zeros(lane_valid.shape, dtype=jnp.int32)
        lane_sources = lane_routes
        lane_slots = lane_routes

    if tile_count:
        tile_groups = KeyGroupPlan(
            edge_tile, receiver_tile, receiver_tile - 1, case_shape=(tile_count,)
        ).build(lane_slots, lane_valid)
        lookup = tile_groups.lookup(jnp.broadcast_to(slots, (tile_count, receiver_tile)))
        group_slots, has_group = lookup.group_slots, lookup.supported
        # Valid lanes are a receiver-slot-major prefix of each tile, so a slot
        # group's storage start is its first lane.
        lane_offsets = jnp.where(
            has_group,
            jnp.take_along_axis(tile_groups.group_starts, group_slots, axis=1),
            0,
        )
        lane_counts = jnp.where(
            has_group,
            jnp.take_along_axis(tile_groups.group_counts, group_slots, axis=1),
            0,
        )
        source_groups = KeyGroupPlan(
            edge_tile, edge_tile, max(sender_count - 1, 0), case_shape=(tile_count,)
        ).build(lane_sources, lane_valid)
        source_valid = source_groups.group_active
        source_ids = jnp.where(source_valid, source_groups.group_keys, 0)
        lane_local_sources = jnp.where(lane_valid, source_groups.item_group_slots, 0)
        source_lanes = source_groups.storage_to_logical
        source_offsets = source_groups.group_starts
        source_counts = source_groups.group_counts
        groups_successful = jnp.all(tile_groups.evidence.successful) & jnp.all(
            source_groups.evidence.successful
        )
    else:
        tile_groups = None
        group_slots = jnp.zeros((0, receiver_tile), dtype=jnp.int32)
        has_group = jnp.zeros((0, receiver_tile), dtype=jnp.bool_)
        lane_offsets = lane_counts = group_slots
        lane_local_sources = source_lanes = source_offsets = source_counts = jnp.zeros(
            (0, edge_tile), dtype=jnp.int32
        )
        source_ids = source_offsets
        source_valid = jnp.zeros((0, edge_tile), dtype=jnp.bool_)
        groups_successful = jnp.asarray(True)

    duplicates = route_groups.evidence.duplicate_stable_ids
    successful = route_groups.evidence.successful & complete & groups_successful
    if not isinstance(successful, jax_core.Tracer) and not bool(successful):
        raise ValueError(
            "Streamed relation preparation refused: "
            f"{int(duplicates)} duplicate stable route IDs among valid routes."
        )
    return StreamedSchedule(
        route_groups=route_groups,
        receiver_offsets=prefix,
        receiver_ids=receiver_ids.astype(jnp.int32),
        receiver_slot_valid=slot_valid,
        receiver_commit=receiver_commit.astype(jnp.int32),
        receiver_group_slots=group_slots,
        receiver_has_group=has_group,
        continuation_in=offsets > 0,
        open_out=extent.open_last,
        open_slot=open_slot.astype(jnp.int32),
        tile_active=tile_active,
        lane_routes=lane_routes.astype(jnp.int32),
        lane_sources=lane_sources.astype(jnp.int32),
        lane_slots=lane_slots.astype(jnp.int32),
        lane_valid=lane_valid,
        receiver_lane_offsets=lane_offsets.astype(jnp.int32),
        receiver_lane_counts=lane_counts.astype(jnp.int32),
        fragment_source_ids=source_ids.astype(jnp.int32),
        fragment_source_valid=source_valid,
        lane_local_sources=lane_local_sources.astype(jnp.int32),
        source_lanes=source_lanes.astype(jnp.int32),
        source_lane_offsets=source_offsets.astype(jnp.int32),
        source_lane_counts=source_counts.astype(jnp.int32),
        tile_groups=tile_groups,
        active_routes=route_groups.evidence.active_items,
        maximum_receiver_degree=jnp.max(degree, initial=0),
        # A receiver is fragmented once: at the tile where it first stays open.
        fragmented_receivers=jnp.sum(
            extent.open_last & tile_active & ~((offsets > 0) & (open_slot == 0)),
            dtype=jnp.int32,
        ),
        tiles_used=jnp.sum(tile_active, dtype=jnp.int32),
        duplicate_stable_ids=duplicates,
        schedule_complete=complete,
        successful=successful,
        tile_count=tile_count,
        receiver_count=receiver_count,
        sender_count=sender_count,
        route_capacity=route_capacity,
        receiver_tile=receiver_tile,
        edge_tile=edge_tile,
    )


@final
class StreamedRelationResources(StrictModule):
    """Substrate-declared logical byte bounds of one streamed evaluation.

    These count arrays the substrate owns: persistent schedules, one bounded
    edge fragment and receiver tile, the spill carry, retained replay
    boundaries, admitted outputs and their staging, and the single live
    cotangent accumulator of differentiable inputs under reverse mode
    (parameters, rows and the dynamic array leaves of array-bearing callbacks).
    Callback-internal intermediates and compiler temporaries are NOT included;
    certify a transformed executable with compiled memory analysis.
    ``replay_tiles_in_flight`` tiles retain callback residuals during replay
    (one for step replay, the block size for block replay, every tile for full
    replay, whose ``replay_boundary_bytes`` is therefore undeclared).
    """

    tile_count: int = eqx.field(static=True)
    receiver_tile: int = eqx.field(static=True)
    edge_tile: int = eqx.field(static=True)
    persistent_schedule_bytes: int = eqx.field(static=True)
    edge_workspace_bytes: int = eqx.field(static=True)
    receiver_workspace_bytes: int = eqx.field(static=True)
    spill_carry_bytes: int = eqx.field(static=True)
    replay_boundary_bytes: int | None = eqx.field(static=True)
    replay_tiles_in_flight: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    output_staging_bytes: int = eqx.field(static=True)
    cotangent_accumulator_bytes: int = eqx.field(static=True)
    replay: CheckpointedScanMode = eqx.field(static=True)

    @property
    def resources_id(self) -> str:
        return canonical_fingerprint(
            {
                "type": "streamed-relation-resources",
                "tile_count": self.tile_count,
                "receiver_tile": self.receiver_tile,
                "edge_tile": self.edge_tile,
                "persistent_schedule_bytes": self.persistent_schedule_bytes,
                "edge_workspace_bytes": self.edge_workspace_bytes,
                "receiver_workspace_bytes": self.receiver_workspace_bytes,
                "spill_carry_bytes": self.spill_carry_bytes,
                "replay_boundary_bytes": self.replay_boundary_bytes,
                "replay_tiles_in_flight": self.replay_tiles_in_flight,
                "output_bytes": self.output_bytes,
                "output_staging_bytes": self.output_staging_bytes,
                "cotangent_accumulator_bytes": self.cotangent_accumulator_bytes,
                "replay": self.replay,
            }
        )


@final
class StreamedRelationEvidence(NonTrainableState, StrictModule):
    """Runtime schedule, numerical and topology evidence of one evaluation.

    ``successful`` requires a complete schedule without duplicate stable IDs
    and finite accumulations and committed outputs. A failed evaluation
    poisons inexact outputs with NaN, and their derivatives with respect to
    every differentiable input (including inputs the outputs do not read), so
    derivatives are invalid, never zero.
    """

    active_routes: Array
    evaluated_routes: Array
    committed_receivers: Array
    tiles_used: Array
    fragmented_receivers: Array
    maximum_receiver_degree: Array
    duplicate_stable_ids: Array
    schedule_complete: Array
    finite: Array
    successful: Array
    binding: StreamedTopologyBinding
    tile_count: int = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    direction: StreamedDirection = eqx.field(static=True)


@final
class StreamedRelationResult(StrictModule):
    """Receiver outputs, requested edge outputs, evidence and declared bounds."""

    receiver_outputs: PyTree[Array]
    edge_outputs: PyTree[Array] | None
    evidence: StreamedRelationEvidence
    resources: StreamedRelationResources


class _Tile(NamedTuple):
    receiver_ids: Array
    receiver_commit: Array
    receiver_group_slots: Array
    receiver_has_group: Array
    open_out: Array
    open_slot: Array
    tile_active: Array
    lane_routes: Array
    lane_sources: Array
    lane_slots: Array
    lane_valid: Array
    receiver_lane_offsets: Array
    receiver_lane_counts: Array
    fragment_source_ids: Array
    fragment_source_valid: Array
    lane_local_sources: Array
    source_lanes: Array
    source_lane_offsets: Array
    source_lane_counts: Array
    groups: KeyGroupState


class _TileOutputs(NamedTuple):
    outputs: tuple[Array, ...]
    edge_outputs: tuple[Array, ...]
    edge_index: Array
    evaluated: Array
    finite: Array


type _Spill = tuple[tuple[Array, Array], ...]


def _row_mask(mask: Array, value: Array, /) -> Array:
    return mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))


def _gather(tree: PyTree[Array], indices: Array, /) -> PyTree[Array]:
    return jax.tree.map(lambda value: value[indices], tree)


def _leading_rows(
    name: str, tree: PyTree[ArrayLike], leading: tuple[int, ...], /
) -> PyTree[Array]:
    count = prod(leading)

    def rows(value: ArrayLike) -> Array:
        array = jnp.asarray(value)
        if array.shape[: len(leading)] != leading:
            raise ValueError(
                f"{name} leaves must begin with shape {leading}; got {array.shape}."
            )
        return array.reshape((count,) + array.shape[len(leading) :])

    return jax.tree.map(rows, tree)


def _row_bytes(tree: PyTree[Array], /) -> int:
    return sum(
        prod(leaf.shape[1:]) * leaf.dtype.itemsize for leaf in jax.tree.leaves(tree)
    )


def _tree_bytes(tree: PyTree[Array], /) -> int:
    return sum(
        leaf.size * leaf.dtype.itemsize
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )


def _tile_kernel_groups(stacked: KeyGroupState, plan: KeyGroupPlan, /) -> KeyGroupState:
    """Rebind one scan-sliced tile of stacked tile groups to the tile plan."""
    evidence = stacked.evidence
    return KeyGroupState(
        plan=plan,
        item_keys=stacked.item_keys,
        item_valid=stacked.item_valid,
        stable_ids=stacked.stable_ids,
        storage_to_logical=stacked.storage_to_logical,
        logical_to_storage=stacked.logical_to_storage,
        sorted_keys=stacked.sorted_keys,
        sorted_item_valid=stacked.sorted_item_valid,
        item_group_slots=stacked.item_group_slots,
        group_keys=stacked.group_keys,
        group_active=stacked.group_active,
        group_starts=stacked.group_starts,
        group_counts=stacked.group_counts,
        evidence=KeyGroupEvidence(
            requested_items=evidence.requested_items,
            active_items=evidence.active_items,
            invalid_keys=evidence.invalid_keys,
            duplicate_stable_ids=evidence.duplicate_stable_ids,
            required_groups=evidence.required_groups,
            group_capacity=evidence.group_capacity,
            maximum_group_size=evidence.maximum_group_size,
            member_capacity=evidence.member_capacity,
            group_overflow=evidence.group_overflow,
            member_overflow=evidence.member_overflow,
            successful=evidence.successful,
        ),
    )


def _reduce_slots(
    groups: KeyGroupState,
    receiver_group_slots: Array,
    receiver_has_group: Array,
    lane_use: Array,
    values: Array,
    seed: KeyGroupAccumulation,
    accumulation: RelationAccumulation,
    /,
) -> tuple[KeyGroupAccumulation, Array]:
    """Seeded per-slot sums of one fragment's lanes; slots without lanes keep seeds."""
    edge_tile = groups.plan.item_capacity
    receiver_tile = groups.plan.group_capacity
    if values.shape[:1] != (edge_tile,):
        raise ValueError(
            f"Fragment values must lead with {edge_tile} lanes; got {values.shape}."
        )
    slot_shape = (receiver_tile,) + values.shape[1:]
    if seed.high.shape != slot_shape or seed.high.dtype != values.dtype:
        raise ValueError(
            f"Fragment seeds must be {values.dtype}{list(slot_shape)}; got "
            f"{seed.high.dtype}{list(seed.high.shape)}."
        )
    zeros = jnp.zeros(slot_shape, values.dtype)
    group_present = _row_mask(groups.group_active, zeros)
    # Groups are slot keys in ascending order; seed each group with its slot.
    group_slot = jnp.where(groups.group_active, groups.group_keys, 0)
    reduced, evidence = reduce_key_groups(
        groups,
        values,
        accumulation=accumulation,
        initial=KeyGroupAccumulation(
            jnp.where(group_present, seed.high[group_slot], zeros),
            jnp.where(group_present, seed.correction[group_slot], zeros),
        ),
        value_valid=lane_use,
    )
    present = _row_mask(receiver_has_group, zeros)
    return (
        KeyGroupAccumulation(
            jnp.where(present, reduced.high[receiver_group_slots], seed.high),
            jnp.where(present, reduced.correction[receiver_group_slots], seed.correction),
        ),
        jnp.all(evidence.finite),
    )


@final
class StreamedFragment(NonTrainableState, StrictModule):
    """Prepared routing of one bounded fragment (or, stacked, of every tile).

    Extents are the plan's ``receiver_tile`` R slots and ``edge_tile`` F
    lanes; fragment-local sources number at most F. Every index is in range.

    - ``receiver_ids`` [R]: receiver of each slot (slot zero may continue the
      previous fragment's receiver). Lanes are slot-major in stable order:
      slot ``r`` owns lanes ``receiver_lane_offsets[r] + [0,
      receiver_lane_counts[r])``.
    - ``lane_routes``/``lane_sources``/``lane_slots`` [F]: original route,
      relation source and receiver slot of each lane; ``lane_use`` is
      structural validity AND the runtime route mask.
    - ``source_ids`` [F]: distinct sources of the fragment, ascending
      (``source_valid``; padding is source zero with no lanes);
      ``lane_local_sources`` [F] maps lanes onto them; ``source_lanes`` [F]
      lists lanes source-major, source ``s`` owning positions
      ``source_lane_offsets[s] + [0, source_lane_counts[s])``.

    `reduce` is the substrate's canonical seeded additive reduction of
    per-lane values into the slots, under the plan's ``accumulation``.
    """

    receiver_ids: Array
    receiver_lane_offsets: Array
    receiver_lane_counts: Array
    lane_routes: Array
    lane_sources: Array
    lane_slots: Array
    lane_use: Array
    source_ids: Array
    source_valid: Array
    lane_local_sources: Array
    source_lanes: Array
    source_lane_offsets: Array
    source_lane_counts: Array
    receiver_groups: KeyGroupState
    receiver_group_slots: Array
    receiver_has_group: Array
    accumulation: RelationAccumulation = eqx.field(static=True)

    def reduce(
        self, values: ArrayLike, seed: KeyGroupAccumulation, /
    ) -> KeyGroupAccumulation:
        """Add usable lanes' ``values[F, ...]`` to ``seed[R, ...]`` by receiver slot."""
        if not isinstance(seed, KeyGroupAccumulation):
            raise TypeError("seed must be a KeyGroupAccumulation.")
        if self.lane_slots.ndim != 1:
            raise ValueError("reduce applies to one fragment, not the stacked schedule.")
        receiver_tile = self.receiver_ids.shape[0]
        groups = _tile_kernel_groups(
            self.receiver_groups,
            KeyGroupPlan(self.lane_slots.shape[0], receiver_tile, receiver_tile - 1),
        )
        reduced, _ = _reduce_slots(
            groups,
            self.receiver_group_slots,
            self.receiver_has_group,
            self.lane_use,
            jnp.asarray(values),
            seed,
            self.accumulation,
        )
        return reduced


def _input_link(trees: tuple[object, ...], /) -> Array:
    """An exact zero whose derivatives of every order reach each input leaf.

    ``trees`` are the evaluation's differentiable inputs, including the
    dynamic array leaves of array-bearing callbacks. Non-finite entries are
    masked first, so the link is an exact zero even for padding NaN rows. The
    final ``sin`` keeps every derivative order structurally present (a linear
    link has an identically zero Hessian, so HVPs and force-loss mixed
    derivatives of a failed evaluation would stay finite).
    """
    link = jnp.zeros(())
    for leaf in jax.tree.leaves(trees):
        if not isinstance(leaf, jax.Array) or not jnp.issubdtype(leaf.dtype, jnp.inexact):
            continue
        safe = jnp.where(jnp.isfinite(leaf), leaf, jnp.zeros((), leaf.dtype))
        touch = jnp.sum(safe - jax.lax.stop_gradient(safe))
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
            touch = jnp.real(touch) + jnp.imag(touch)
        link = link + touch.astype(link.dtype)
    return jnp.sin(link)


def _poison(value: Array, successful: Array, link: Array, /) -> Array:
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        return value
    # A NaN factor poisons the value and every derivative through it, so a
    # failed evaluation yields invalid derivatives rather than masked zeros or
    # finite values independent of the failed part. The zero-valued input link
    # carries that NaN to inputs the output does not depend on (a parameter
    # read only by a failed message under a residual-only epilogue). An
    # admitted evaluation multiplies by exactly one with a zero-scaled link
    # tangent, leaving values and derivatives unchanged.
    real = jnp.finfo(value.dtype).dtype
    scale = jnp.where(successful, jnp.zeros((), real), jnp.nan)
    return value * (1 + scale * link.astype(real))


def _finite_rows(values: list[Array], rows: Array, /) -> Array:
    """Whether every inexact value is finite on the selected leading rows."""
    finite = jnp.asarray(True)
    for value in values:
        if jnp.issubdtype(value.dtype, jnp.inexact):
            selected = jnp.where(
                _row_mask(rows, value), value, jnp.zeros((), value.dtype)
            )
            finite = finite & jnp.all(jnp.isfinite(selected))
    return finite


class _StreamedKernel(NamedTuple):
    """One evaluation's static payload/plan facts and dynamic closure data.

    Exactly one of ``edge_function`` (per-event messages, substrate-reduced)
    and ``fragment_aggregate`` (whole-fragment seeded accumulation) is set.
    """

    payload: StreamedPayloadSpec
    plan: StreamedRelationPlan
    edge_function: StreamedEdgeFunction | None
    fragment_aggregate: StreamedFragmentAggregator | None
    epilogue: StreamedReceiverEpilogue
    parameters: PyTree[Array]
    sources: PyTree[Array]
    receivers: PyTree[Array]
    edges: PyTree[Array]
    edge_active: Array
    receiver_count: int
    route_capacity: int

    def groups(self, tile: _Tile, /) -> KeyGroupState:
        receiver_tile = tile.receiver_ids.shape[0]
        return _tile_kernel_groups(
            tile.groups,
            KeyGroupPlan(tile.lane_slots.shape[0], receiver_tile, receiver_tile - 1),
        )

    def seeds(self, tile: _Tile, spill: _Spill, /) -> list[KeyGroupAccumulation]:
        """Slot-zero seeds from the previous fragment's open receiver."""
        # A continuing receiver owns slot zero; an exhausted spill is exactly
        # zero, so seeding slot zero is inert for a fresh receiver.
        receiver_tile = tile.receiver_ids.shape[0]
        seeds = []
        for (high, correction), shape, dtype in zip(
            spill, self.payload.message_shapes, self.payload.message_dtypes, strict=True
        ):
            zeros = jnp.zeros((receiver_tile,) + shape, dtype)
            seeds.append(
                KeyGroupAccumulation(zeros.at[0].set(high), zeros.at[0].set(correction))
            )
        return seeds

    def edge_specs(
        self,
    ) -> tuple[PyTreeDef, tuple[tuple[int, ...], ...], tuple[np.dtype, ...]] | None:
        payload = self.payload
        if payload.edge_output_tree is None:
            return None
        return (
            payload.edge_output_tree,
            payload.edge_output_shapes,
            payload.edge_output_dtypes,
        )

    def zero_edge_outputs(self, edge_tile: int, /) -> list[Array]:
        payload = self.payload
        return [
            jnp.zeros((edge_tile,) + shape, dtype)
            for shape, dtype in zip(
                payload.edge_output_shapes, payload.edge_output_dtypes
            )
        ]

    def events(self, tile: _Tile, lane_use: Array, /) -> tuple[list[Array], list[Array]]:
        """Evaluate one fragment; unusable lanes replay a usable representative."""
        payload = self.payload
        edge_tile = lane_use.shape[0]
        message_specs = (
            payload.message_tree,
            payload.message_shapes,
            payload.message_dtypes,
        )
        edge_specs = self.edge_specs()
        if self.edge_function is None:
            raise RuntimeError("Per-event streaming needs an edge function.")
        edge_function = self.edge_function

        def skipped() -> tuple[list[Array], list[Array]]:
            # No usable event: callbacks never see invalid/inactive dummy data.
            return (
                [
                    jnp.zeros((edge_tile,) + shape, dtype)
                    for shape, dtype in zip(
                        payload.message_shapes, payload.message_dtypes
                    )
                ],
                self.zero_edge_outputs(edge_tile),
            )

        if self.route_capacity == 0:
            return skipped()

        def evaluated() -> tuple[list[Array], list[Array]]:
            representative = jnp.argmax(lane_use)
            routes = jnp.where(
                lane_use, tile.lane_routes, tile.lane_routes[representative]
            )
            sources = jnp.where(
                lane_use, tile.lane_sources, tile.lane_sources[representative]
            )
            slots = jnp.where(lane_use, tile.lane_slots, tile.lane_slots[representative])

            def event(
                source: PyTree[Array], receiver: PyTree[Array], edge: PyTree[Array]
            ) -> PyTree[Array]:
                return edge_function(self.parameters, source, receiver, edge)

            result = jax.vmap(event)(
                _gather(self.sources, sources),
                _gather(self.receivers, tile.receiver_ids[slots]),
                _gather(self.edges, routes),
            )
            if edge_specs is None:
                return _conform("edge_function", result, message_specs, (edge_tile,)), []
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "edge_function must return (message, edge_output) when edge "
                    "outputs are requested."
                )
            return (
                _conform("edge_function message", result[0], message_specs, (edge_tile,)),
                _conform(
                    "edge_function edge_output", result[1], edge_specs, (edge_tile,)
                ),
            )

        return jax.lax.cond(jnp.any(lane_use), evaluated, skipped)

    def accumulate(
        self,
        tile: _Tile,
        messages: list[Array],
        lane_use: Array,
        seeds: list[KeyGroupAccumulation],
        /,
    ) -> tuple[list[KeyGroupAccumulation], Array]:
        """Seeded per-receiver-slot sums of per-event messages."""
        groups = self.groups(tile)
        accumulations: list[KeyGroupAccumulation] = []
        finite = jnp.asarray(True)
        for values, seed in zip(messages, seeds, strict=True):
            reduced, reduced_finite = _reduce_slots(
                groups,
                tile.receiver_group_slots,
                tile.receiver_has_group,
                lane_use,
                values,
                seed,
                self.plan.accumulation,
            )
            accumulations.append(reduced)
            finite = finite & reduced_finite
        return accumulations, finite

    def fragment(
        self, tile: _Tile, lane_use: Array, seeds: list[KeyGroupAccumulation], /
    ) -> tuple[list[KeyGroupAccumulation], list[Array], Array]:
        """Whole-fragment seeded accumulation by the consumer's aggregator."""
        payload = self.payload
        edge_tile = lane_use.shape[0]
        receiver_tile = tile.receiver_ids.shape[0]
        edge_specs = self.edge_specs()
        if self.fragment_aggregate is None:
            raise RuntimeError("Fragment streaming needs a fragment aggregator.")
        aggregate = self.fragment_aggregate

        def skipped() -> tuple[list[Array], list[Array], list[Array]]:
            # No usable lane: the aggregator never sees dummy data; seeds pass.
            return (
                [seed.high for seed in seeds],
                [seed.correction for seed in seeds],
                self.zero_edge_outputs(edge_tile),
            )

        if self.route_capacity == 0:
            high, correction, edge_values = skipped()
        else:

            def evaluated() -> tuple[list[Array], list[Array], list[Array]]:
                representative = jnp.argmax(lane_use)
                routes = jnp.where(
                    lane_use, tile.lane_routes, tile.lane_routes[representative]
                )
                fragment = StreamedFragment(
                    receiver_ids=tile.receiver_ids,
                    receiver_lane_offsets=tile.receiver_lane_offsets,
                    receiver_lane_counts=tile.receiver_lane_counts,
                    lane_routes=tile.lane_routes,
                    lane_sources=tile.lane_sources,
                    lane_slots=tile.lane_slots,
                    lane_use=lane_use,
                    source_ids=tile.fragment_source_ids,
                    source_valid=tile.fragment_source_valid,
                    lane_local_sources=tile.lane_local_sources,
                    source_lanes=tile.source_lanes,
                    source_lane_offsets=tile.source_lane_offsets,
                    source_lane_counts=tile.source_lane_counts,
                    receiver_groups=self.groups(tile),
                    receiver_group_slots=tile.receiver_group_slots,
                    receiver_has_group=tile.receiver_has_group,
                    accumulation=self.plan.accumulation,
                )
                result = aggregate(
                    self.parameters,
                    self.sources,
                    _gather(self.receivers, tile.receiver_ids),
                    _gather(self.edges, routes),
                    fragment,
                    payload.message_tree.unflatten(seeds),
                )
                edge_result: list[Array] = []
                if edge_specs is not None:
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise TypeError(
                            "fragment_aggregate must return (accumulations, edge_output) "
                            "when edge outputs are requested."
                        )
                    result, edge_tree = result
                    edge_result = _conform(
                        "fragment_aggregate edge_output",
                        edge_tree,
                        edge_specs,
                        (edge_tile,),
                    )
                leaves = payload.message_tree.flatten_up_to(result)
                if not all(isinstance(leaf, KeyGroupAccumulation) for leaf in leaves):
                    raise TypeError(
                        "fragment_aggregate must return one KeyGroupAccumulation per "
                        "declared message leaf."
                    )
                specs = (
                    payload.message_tree,
                    payload.message_shapes,
                    payload.message_dtypes,
                )
                return (
                    _conform(
                        "fragment_aggregate high",
                        payload.message_tree.unflatten([leaf.high for leaf in leaves]),
                        specs,
                        (receiver_tile,),
                    ),
                    _conform(
                        "fragment_aggregate correction",
                        payload.message_tree.unflatten(
                            [leaf.correction for leaf in leaves]
                        ),
                        specs,
                        (receiver_tile,),
                    ),
                    edge_result,
                )

            high, correction, edge_values = jax.lax.cond(
                jnp.any(lane_use), evaluated, skipped
            )
        finite = jnp.asarray(True)
        for value_high, value_correction in zip(high, correction, strict=True):
            if jnp.issubdtype(value_high.dtype, jnp.inexact):
                finite = finite & jnp.all(jnp.isfinite(value_high + value_correction))
        accumulations = [
            KeyGroupAccumulation(value_high, value_correction)
            for value_high, value_correction in zip(high, correction, strict=True)
        ]
        return accumulations, edge_values, finite

    def finish(
        self, tile: _Tile, aggregates: list[Array], /
    ) -> tuple[list[Array], Array]:
        """Run the epilogue on committing receiver slots only.

        A slot that does not commit (an open partial fragment, a masked
        receiver or padding) never presents its own aggregate to the epilogue:
        a partial sum may lie outside the epilogue's domain (``log`` of a
        partial zero) and its masked derivative would still be NaN. Such slots
        replay a committing slot of the tile and are dropped at commit; a tile
        without committing slots skips the epilogue. Committed empty rows keep
        their lawful zero-aggregate epilogue.
        """
        payload = self.payload
        receiver_tile = tile.receiver_ids.shape[0]
        committed = tile.receiver_commit < self.receiver_count

        def receiver(row: PyTree[Array], aggregate: PyTree[Array]) -> PyTree[Array]:
            return self.epilogue(self.parameters, row, aggregate)

        def evaluated() -> list[Array]:
            representative = jnp.argmax(committed)
            receiver_ids = jnp.where(
                committed, tile.receiver_ids, tile.receiver_ids[representative]
            )
            safe = [
                jnp.where(_row_mask(committed, value), value, value[representative])
                for value in aggregates
            ]
            return _conform(
                "epilogue",
                jax.vmap(receiver)(
                    _gather(self.receivers, receiver_ids),
                    payload.message_tree.unflatten(safe),
                ),
                (payload.output_tree, payload.output_shapes, payload.output_dtypes),
                (receiver_tile,),
            )

        def skipped() -> list[Array]:
            return [
                jnp.zeros((receiver_tile,) + shape, dtype)
                for shape, dtype in zip(payload.output_shapes, payload.output_dtypes)
            ]

        outputs = jax.lax.cond(jnp.any(committed), evaluated, skipped)
        return outputs, _finite_rows(outputs, committed)

    def tile(self, spill: _Spill, tile: _Tile, /) -> tuple[_Spill, _TileOutputs]:
        def run() -> tuple[_Spill, _TileOutputs]:
            lane_use = (
                tile.lane_valid & self.edge_active[tile.lane_routes]
                if self.route_capacity
                else jnp.zeros_like(tile.lane_valid)
            )
            seeds = self.seeds(tile, spill)
            if self.fragment_aggregate is None:
                messages, edge_values = self.events(tile, lane_use)
                accumulations, accumulated = self.accumulate(
                    tile, messages, lane_use, seeds
                )
            else:
                accumulations, edge_values, accumulated = self.fragment(
                    tile, lane_use, seeds
                )
            next_spill = tuple(
                (
                    jnp.where(
                        tile.open_out,
                        accumulation.high[tile.open_slot],
                        jnp.zeros((), accumulation.high.dtype),
                    ),
                    jnp.where(
                        tile.open_out,
                        accumulation.correction[tile.open_slot],
                        jnp.zeros((), accumulation.correction.dtype),
                    ),
                )
                for accumulation in accumulations
            )
            outputs, finished = self.finish(
                tile, [accumulation.value for accumulation in accumulations]
            )
            # Requested edge outputs of active events are admitted results; an
            # unusable lane replays a usable one and is masked, so it is inert.
            emitted = _finite_rows(edge_values, lane_use)
            return next_spill, _TileOutputs(
                outputs=tuple(outputs),
                edge_outputs=tuple(
                    jnp.where(
                        _row_mask(lane_use, value), value, jnp.zeros((), value.dtype)
                    )
                    for value in edge_values
                ),
                edge_index=jnp.where(lane_use, tile.lane_routes, self.route_capacity),
                evaluated=jnp.sum(lane_use, dtype=jnp.int32),
                finite=accumulated & finished & emitted,
            )

        def idle() -> tuple[_Spill, _TileOutputs]:
            # Static-bound tiles beyond a traced schedule's last tile.
            receiver_tile = tile.receiver_ids.shape[0]
            edge_tile = tile.lane_routes.shape[0]
            payload = self.payload
            return spill, _TileOutputs(
                outputs=tuple(
                    jnp.zeros((receiver_tile,) + shape, dtype)
                    for shape, dtype in zip(payload.output_shapes, payload.output_dtypes)
                ),
                edge_outputs=tuple(
                    jnp.zeros((edge_tile,) + shape, dtype)
                    for shape, dtype in zip(
                        payload.edge_output_shapes, payload.edge_output_dtypes
                    )
                ),
                edge_index=jnp.full((edge_tile,), self.route_capacity, dtype=jnp.int32),
                evaluated=jnp.zeros((), dtype=jnp.int32),
                finite=jnp.asarray(True),
            )

        return jax.lax.cond(tile.tile_active, run, idle)


@final
class PreparedStreamedRelation(StrictModule):
    """One direction of a prepared streamed relation bound to its topology."""

    plan: StreamedRelationPlan
    schedule: StreamedSchedule
    transpose_schedule: StreamedSchedule | None
    binding: StreamedTopologyBinding
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    route_shape: tuple[int, ...] = eqx.field(static=True)
    direction: StreamedDirection = eqx.field(static=True)

    @property
    def execution_id(self) -> str:
        """Static execution identity: plan, topology binding, direction, tiles."""
        return canonical_fingerprint(
            {
                "type": "streamed-relation",
                "plan_id": self.plan.plan_id,
                "binding_id": self.binding.binding_id,
                "direction": self.direction,
                "input_shape": list(self.input_shape),
                "output_shape": list(self.output_shape),
                "route_shape": list(self.route_shape),
                "tile_count": self.schedule.tile_count,
                "transpose_tile_count": None
                if self.transpose_schedule is None
                else self.transpose_schedule.tile_count,
            }
        )

    def transpose(self) -> PreparedStreamedRelation:
        """Stream into relation sources using the independently prepared schedule."""
        if self.transpose_schedule is None:
            raise ValueError(
                "This relation was prepared without transpose=True; no source-major "
                "schedule exists."
            )
        direction: StreamedDirection = (
            "source" if self.direction == "target" else "target"
        )
        return PreparedStreamedRelation(
            plan=self.plan,
            schedule=self.transpose_schedule,
            transpose_schedule=self.schedule,
            binding=self.binding,
            input_shape=self.output_shape,
            output_shape=self.input_shape,
            route_shape=self.route_shape,
            direction=direction,
        )

    def resources(
        self,
        payload: StreamedPayloadSpec,
        parameters: PyTree[Array],
        source_data: PyTree[ArrayLike],
        receiver_data: PyTree[ArrayLike],
        edge_data: PyTree[ArrayLike],
        /,
        *,
        callbacks: PyTree[object] = None,
        fragment_aggregation: bool = False,
    ) -> StreamedRelationResources:
        """Declared substrate bytes of one evaluation with these inputs.

        ``callbacks`` are the edge function (or fragment aggregator) and
        epilogue the evaluation will use: the dynamic array leaves of
        array-bearing callbacks (for example `jax.tree_util.Partial` weights)
        are differentiable inputs counted in ``cotangent_accumulator_bytes``.
        ``fragment_aggregation`` selects `evaluate_fragments` staging (source
        rows gathered once per distinct fragment source, receiver rows once per
        slot) instead of per-event gathers.
        """
        if not isinstance(fragment_aggregation, bool):
            raise TypeError("fragment_aggregation must be a bool.")
        return self._resources(
            payload,
            parameters,
            _leading_rows("source_data", source_data, self.input_shape),
            _leading_rows("receiver_data", receiver_data, self.output_shape),
            _leading_rows("edge_data", edge_data, self.route_shape),
            callbacks,
            fragment_aggregation,
        )

    def _resources(
        self,
        payload: StreamedPayloadSpec,
        parameters: PyTree[Array],
        sources: PyTree[Array],
        receivers: PyTree[Array],
        edges: PyTree[Array],
        callbacks: PyTree[object],
        fragment_aggregation: bool,
        /,
    ) -> StreamedRelationResources:
        schedule = self.schedule
        receiver_tile, edge_tile = schedule.receiver_tile, schedule.edge_tile
        tiles = schedule.tile_count
        spill = 2 * payload.message_bytes
        # Per-event gathers stage source, receiver and edge rows per lane; a
        # fragment aggregator stages at most one source row per lane and its
        # receiver rows once per slot (counted in the receiver workspace).
        edge_workspace = edge_tile * (
            _row_bytes(sources)
            + (0 if fragment_aggregation else _row_bytes(receivers))
            + _row_bytes(edges)
            + payload.message_bytes
            + payload.edge_output_bytes
        )
        # Seeded high/correction in group and receiver space, the aggregate,
        # gathered receiver rows and the epilogue outputs of one tile.
        receiver_workspace = receiver_tile * (
            5 * payload.message_bytes + _row_bytes(receivers) + payload.output_bytes
        )
        match self.plan.replay:
            case "step":
                boundary: int | None = tiles * spill
                in_flight = min(1, tiles)
            case "block":
                block = self.plan.replay_block_size or 1
                boundary = ceil(tiles / block) * spill
                in_flight = min(block, tiles)
            case "full":
                boundary = None
                in_flight = tiles
            case "scheduled":
                raise ValueError(
                    "Scheduled replay is not admitted for streamed relations."
                )
            case _:
                assert_never(self.plan.replay)
        persistent = schedule.storage_bytes + (
            0
            if self.transpose_schedule is None
            else self.transpose_schedule.storage_bytes
        )
        return StreamedRelationResources(
            tile_count=tiles,
            receiver_tile=receiver_tile,
            edge_tile=edge_tile,
            persistent_schedule_bytes=persistent,
            edge_workspace_bytes=edge_workspace,
            receiver_workspace_bytes=receiver_workspace,
            spill_carry_bytes=spill,
            replay_boundary_bytes=boundary,
            replay_tiles_in_flight=in_flight,
            output_bytes=prod(self.output_shape) * payload.output_bytes
            + prod(self.route_shape) * payload.edge_output_bytes,
            output_staging_bytes=tiles
            * (
                receiver_tile * payload.output_bytes
                + edge_tile * payload.edge_output_bytes
            ),
            cotangent_accumulator_bytes=_tree_bytes(parameters)
            + _tree_bytes(sources)
            + _tree_bytes(receivers)
            + _tree_bytes(edges)
            + _tree_bytes(callbacks),
            replay=self.plan.replay,
        )

    def evaluate(
        self,
        payload: StreamedPayloadSpec,
        edge_function: StreamedEdgeFunction,
        epilogue: StreamedReceiverEpilogue,
        parameters: PyTree[Array],
        source_data: PyTree[ArrayLike],
        receiver_data: PyTree[ArrayLike],
        edge_data: PyTree[ArrayLike],
        *,
        edge_active: ArrayLike | None = None,
    ) -> StreamedRelationResult:
        """Gather, evaluate, accumulate and commit every receiver exactly once.

        ``source_data`` leaves lead with this direction's input shape,
        ``receiver_data`` with its output shape and ``edge_data`` with the
        route shape (original route order). ``edge_active`` is a runtime route
        mask that never changes the schedule. Inactive, invalid and padded
        lanes replay a usable lane of the same fragment and are masked from
        accumulation; a fragment without usable lanes skips its callbacks
        through ``lax.cond`` (under ``vmap`` that skip becomes a select, so a
        batched caller's slot-zero route data must then be domain-safe).
        Receiver outputs lead with the output shape; requested edge outputs
        lead with the route shape and are zero on unused routes.
        """
        return self._evaluate(
            payload,
            edge_function,
            None,
            epilogue,
            parameters,
            source_data,
            receiver_data,
            edge_data,
            edge_active,
        )

    def evaluate_fragments(
        self,
        payload: StreamedPayloadSpec,
        aggregate: StreamedFragmentAggregator,
        epilogue: StreamedReceiverEpilogue,
        parameters: PyTree[Array],
        source_data: PyTree[ArrayLike],
        receiver_data: PyTree[ArrayLike],
        edge_data: PyTree[ArrayLike],
        *,
        edge_active: ArrayLike | None = None,
    ) -> StreamedRelationResult:
        """Stream whole-fragment seeded aggregation, then commit epilogues once.

        Inputs, outputs, replay, spill and poisoning are exactly those of
        `evaluate`; ``aggregate`` (a `StreamedFragmentAggregator`) replaces the
        per-event callback and its substrate reduction. Each fragment's seeded
        accumulations carry an open receiver into the next fragment, and the
        epilogue sees ``high + correction`` after the receiver's final one.
        """
        return self._evaluate(
            payload,
            None,
            aggregate,
            epilogue,
            parameters,
            source_data,
            receiver_data,
            edge_data,
            edge_active,
        )

    def fragments(self, edge_active: ArrayLike | None = None, /) -> StreamedFragment:
        """Every prepared fragment's routing, stacked with a leading tile axis.

        ``lane_use`` combines structural lane validity with ``edge_active``.
        Slice one tile (``jax.tree.map(lambda x: x[t], ...)``) before calling
        `StreamedFragment.reduce` or a fragment kernel.
        """
        schedule = self.schedule
        if schedule.tile_groups is None:
            raise ValueError("This relation has no prepared fragments.")
        active = self._edge_active(edge_active)
        return StreamedFragment(
            receiver_ids=schedule.receiver_ids,
            receiver_lane_offsets=schedule.receiver_lane_offsets,
            receiver_lane_counts=schedule.receiver_lane_counts,
            lane_routes=schedule.lane_routes,
            lane_sources=schedule.lane_sources,
            lane_slots=schedule.lane_slots,
            lane_use=schedule.lane_valid & active[schedule.lane_routes]
            if schedule.route_capacity
            else schedule.lane_valid,
            source_ids=schedule.fragment_source_ids,
            source_valid=schedule.fragment_source_valid,
            lane_local_sources=schedule.lane_local_sources,
            source_lanes=schedule.source_lanes,
            source_lane_offsets=schedule.source_lane_offsets,
            source_lane_counts=schedule.source_lane_counts,
            receiver_groups=schedule.tile_groups,
            receiver_group_slots=schedule.receiver_group_slots,
            receiver_has_group=schedule.receiver_has_group,
            accumulation=self.plan.accumulation,
        )

    def _edge_active(self, edge_active: ArrayLike | None, /) -> Array:
        route_capacity = self.schedule.route_capacity
        if edge_active is None:
            return jnp.ones((route_capacity,), dtype=jnp.bool_)
        active = jnp.asarray(edge_active)
        if active.dtype != jnp.bool_ or active.shape != self.route_shape:
            raise ValueError(
                f"edge_active must be a boolean mask of shape {self.route_shape}."
            )
        return active.reshape((route_capacity,))

    def _evaluate(
        self,
        payload: StreamedPayloadSpec,
        edge_function: StreamedEdgeFunction | None,
        aggregate: StreamedFragmentAggregator | None,
        epilogue: StreamedReceiverEpilogue,
        parameters: PyTree[Array],
        source_data: PyTree[ArrayLike],
        receiver_data: PyTree[ArrayLike],
        edge_data: PyTree[ArrayLike],
        edge_active: ArrayLike | None,
        /,
    ) -> StreamedRelationResult:
        if not isinstance(payload, StreamedPayloadSpec):
            raise TypeError("payload must be a StreamedPayloadSpec.")
        width = max(payload.event_elements, payload.output_elements)
        if width > self.plan.channel_capacity:
            raise ValueError(
                f"Payload needs {width} channel elements; the plan admits "
                f"{self.plan.channel_capacity}."
            )
        sources = _leading_rows("source_data", source_data, self.input_shape)
        receivers = _leading_rows("receiver_data", receiver_data, self.output_shape)
        edges = _leading_rows("edge_data", edge_data, self.route_shape)
        schedule = self.schedule
        route_capacity = schedule.route_capacity
        active = self._edge_active(edge_active)
        callbacks = (edge_function, aggregate, epilogue)
        resources = self._resources(
            payload,
            parameters,
            sources,
            receivers,
            edges,
            callbacks,
            aggregate is not None,
        )
        kernel = _StreamedKernel(
            payload=payload,
            plan=self.plan,
            edge_function=edge_function,
            fragment_aggregate=aggregate,
            epilogue=epilogue,
            parameters=parameters,
            sources=sources,
            receivers=receivers,
            edges=edges,
            edge_active=active,
            receiver_count=schedule.receiver_count,
            route_capacity=route_capacity,
        )
        staged, evaluated, finite = self._stream(kernel)
        # Every public output carries one admission: a failed schedule or any
        # non-finite accumulation, committed output or requested edge output
        # poisons all of them, so consumers that never read the evidence still
        # cannot use finite values or derivatives from a failed evaluation.
        successful = schedule.successful & finite
        link = _input_link((parameters, sources, receivers, edges, callbacks))
        receiver_outputs = [
            _poison(
                jnp.zeros((schedule.receiver_count,) + shape, dtype)
                .at[schedule.receiver_commit.reshape((-1,))]
                .add(
                    value.reshape((-1,) + shape),
                    mode="drop",
                )
                .reshape(self.output_shape + shape),
                successful,
                link,
            )
            for value, shape, dtype in zip(
                staged.outputs, payload.output_shapes, payload.output_dtypes, strict=True
            )
        ]
        edge_outputs: PyTree[Array] | None = None
        if payload.edge_output_tree is not None:
            edge_outputs = payload.edge_output_tree.unflatten(
                [
                    _poison(
                        jnp.zeros((route_capacity,) + shape, dtype)
                        .at[staged.edge_index.reshape((-1,))]
                        .add(value.reshape((-1,) + shape), mode="drop")
                        .reshape(self.route_shape + shape),
                        successful,
                        link,
                    )
                    for value, shape, dtype in zip(
                        staged.edge_outputs,
                        payload.edge_output_shapes,
                        payload.edge_output_dtypes,
                        strict=True,
                    )
                ]
            )
        evidence = StreamedRelationEvidence(
            active_routes=schedule.active_routes,
            evaluated_routes=evaluated,
            committed_receivers=jnp.sum(
                schedule.receiver_commit < schedule.receiver_count, dtype=jnp.int32
            ),
            tiles_used=schedule.tiles_used,
            fragmented_receivers=schedule.fragmented_receivers,
            maximum_receiver_degree=schedule.maximum_receiver_degree,
            duplicate_stable_ids=schedule.duplicate_stable_ids,
            schedule_complete=schedule.schedule_complete,
            finite=finite,
            successful=successful,
            binding=self.binding,
            tile_count=schedule.tile_count,
            accumulation=self.plan.accumulation,
            direction=self.direction,
        )
        return StreamedRelationResult(
            receiver_outputs=payload.output_tree.unflatten(receiver_outputs),
            edge_outputs=edge_outputs,
            evidence=evidence,
            resources=resources,
        )

    def _stream(self, kernel: _StreamedKernel, /) -> tuple[_TileOutputs, Array, Array]:
        """Scan tiles under the declared replay; returns staged tile outputs."""
        schedule = self.schedule
        payload = kernel.payload
        if schedule.tile_count == 0 or schedule.tile_groups is None:
            empty = _TileOutputs(
                outputs=tuple(
                    jnp.zeros((0,) + shape, dtype)
                    for shape, dtype in zip(payload.output_shapes, payload.output_dtypes)
                ),
                edge_outputs=tuple(
                    jnp.zeros((0,) + shape, dtype)
                    for shape, dtype in zip(
                        payload.edge_output_shapes, payload.edge_output_dtypes
                    )
                ),
                edge_index=jnp.zeros((0,), dtype=jnp.int32),
                evaluated=jnp.zeros((0,), dtype=jnp.int32),
                finite=jnp.zeros((0,), dtype=jnp.bool_),
            )
            return empty, jnp.zeros((), dtype=jnp.int32), jnp.asarray(True)
        tiles = _Tile(
            receiver_ids=schedule.receiver_ids,
            receiver_commit=schedule.receiver_commit,
            receiver_group_slots=schedule.receiver_group_slots,
            receiver_has_group=schedule.receiver_has_group,
            open_out=schedule.open_out,
            open_slot=schedule.open_slot,
            tile_active=schedule.tile_active,
            lane_routes=schedule.lane_routes,
            lane_sources=schedule.lane_sources,
            lane_slots=schedule.lane_slots,
            lane_valid=schedule.lane_valid,
            receiver_lane_offsets=schedule.receiver_lane_offsets,
            receiver_lane_counts=schedule.receiver_lane_counts,
            fragment_source_ids=schedule.fragment_source_ids,
            fragment_source_valid=schedule.fragment_source_valid,
            lane_local_sources=schedule.lane_local_sources,
            source_lanes=schedule.source_lanes,
            source_lane_offsets=schedule.source_lane_offsets,
            source_lane_counts=schedule.source_lane_counts,
            groups=schedule.tile_groups,
        )
        spill = tuple(
            (jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))
            for shape, dtype in zip(payload.message_shapes, payload.message_dtypes)
        )
        # Linearizing a checkpoint hoists its known primal out of the replay
        # boundary, so a singly checkpointed tile is ordinary code inside the
        # first derivative and reverse-over-reverse (force-loss parameter
        # gradients) would retain every tile's callback residuals. The tile
        # carries its own checkpoint inside the scan's replay boundary; full
        # replay deliberately keeps residuals.
        body = kernel.tile if self.plan.replay == "full" else jax.checkpoint(kernel.tile)
        _, staged = checkpointed_scan(
            body,
            spill,
            tiles,
            length=schedule.tile_count,
            mode=self.plan.replay,
            block_size=self.plan.replay_block_size,
        )
        return staged, jnp.sum(staged.evaluated), jnp.all(staged.finite)


__all__ = [
    "PreparedStreamedRelation",
    "StreamedDirection",
    "StreamedEdgeFunction",
    "StreamedFragment",
    "StreamedFragmentAggregator",
    "StreamedPayloadSpec",
    "StreamedReceiverEpilogue",
    "StreamedRelationEvidence",
    "StreamedRelationPlan",
    "StreamedRelationResources",
    "StreamedRelationResult",
    "StreamedSchedule",
    "StreamedTopologyBinding",
]
