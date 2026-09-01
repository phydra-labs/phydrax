#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...sparse import EdgeRelation
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import reverse_complement, SequenceBatch
from ._assembly import certify_supplied_dag, DAGCertificate


class VariationGraphStatus(IntEnum):
    SUCCESS = 0
    INVALID_GRAPH = 1
    INVALID_PATH = 2
    REPEATED_NODE = 3
    CAPACITY_EXCEEDED = 4
    OUT_OF_RANGE = 5
    EMPTY_INPUT = 6


def _graph_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "supplied variation-DAG validation",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.GRAPH,
        conditioning_statement=(
            "The node sequences, native edge relation, and proposed topological order are "
            "supplied by the caller."
        ),
        truncation_statement=(
            "Validation covers every valid relation route; graph discovery is outside the claim."
        ),
        capacity_semantics="Node and edge capacities are fixed and no valid route is truncated.",
        assumptions=("Every active node contains a non-empty encoded sequence.",),
        nondifferentiable_outputs=("relation", "certificate", "status"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _coordinate_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact variation-graph path coordinates",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement="Each supplied path is a directed walk in the certified DAG.",
        truncation_statement="All supplied path slots are validated; none are silently dropped.",
        capacity_semantics="Path count and path-node width are fixed by array shape.",
        assumptions=("Graph node valid masks are left-prefix masks.",),
        nondifferentiable_outputs=(
            "path_node_indices",
            "node_starts",
            "node_lengths",
            "status",
        ),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _lookup_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact graph-path coordinate lookup",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.ARRAY,
        conditioning_statement="Coordinates are zero-based offsets on one validated graph path.",
        truncation_statement="Each query is answered or explicitly marked out of range.",
        capacity_semantics="Query capacity is fixed by the broadcast query shape.",
        assumptions=(),
        nondifferentiable_outputs=("node_indices", "node_offsets", "status"),
        input_dtype="int32",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _decode_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact variation-graph path decoding",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.SEQUENCE,
        conditioning_statement="Paths and coordinates were validated against the same graph.",
        truncation_statement="A path exceeding output capacity fails rather than truncating.",
        capacity_semantics="Output sequence width is a caller-declared fixed capacity.",
        assumptions=(
            "Reverse-oriented nodes use the graph alphabet complement mapping.",
        ),
        nondifferentiable_outputs=("token_codes", "valid_mask", "status"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


class VariationGraph(StrictModule):
    """Sequence-labeled DAG backed by the single native edge-relation type."""

    node_sequences: SequenceBatch
    relation: EdgeRelation
    method_contract: BioinformaticsMethodContract
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_sequences: SequenceBatch,
        relation: EdgeRelation,
        method_contract: BioinformaticsMethodContract,
        graph_id: str,
        /,
    ):
        node_capacity = node_sequences.token_codes.shape[0]
        if relation.source_size != node_capacity or relation.target_size != node_capacity:
            raise ValueError("Variation-graph relation spaces must match node capacity.")
        if not graph_id:
            raise ValueError("graph_id must be non-empty.")
        self.node_sequences = node_sequences
        self.relation = relation
        self.method_contract = method_contract
        self.graph_id = graph_id


class VariationGraphBuildResult(StrictModule):
    graph: VariationGraph
    certificate: DAGCertificate
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


class GraphPathCoordinates(StrictModule):
    """Fixed-capacity coordinate index for supplied paths through one graph."""

    path_ids: Array
    path_node_indices: Array
    path_node_valid: Array
    path_node_reverse: Array
    node_starts: Array
    node_lengths: Array
    path_lengths: Array
    case_valid: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    graph_id: str = eqx.field(static=True)


class GraphPathLocationResult(StrictModule):
    node_indices: Array
    node_offsets: Array
    node_reverse: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    graph_id: str = eqx.field(static=True)


class GraphPathDecodeResult(StrictModule):
    sequences: SequenceBatch
    path_lengths: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    graph_id: str = eqx.field(static=True)


def build_supplied_variation_graph(
    node_sequences: SequenceBatch,
    source_indices: ArrayLike,
    target_indices: ArrayLike,
    edge_valid: ArrayLike,
    topological_order: ArrayLike,
    order_valid: ArrayLike,
    /,
    *,
    graph_id: str | None = None,
) -> VariationGraphBuildResult:
    """Validate a caller-supplied fixed-capacity variation DAG without discovering it."""
    node_capacity = node_sequences.token_codes.shape[0]
    relation = EdgeRelation(
        source_indices,
        target_indices,
        source_size=node_capacity,
        target_size=node_capacity,
        valid=edge_valid,
    )
    identity = graph_id or canonical_fingerprint(
        {
            "kind": "supplied-variation-dag",
            "node_capacity": node_capacity,
            "node_width": node_sequences.token_codes.shape[1],
            "edge_capacity": relation.capacity,
            "alphabet": node_sequences.alphabet.fingerprint,
        }
    )
    contract = _graph_contract()
    graph = VariationGraph(node_sequences, relation, contract, identity)
    certificate = certify_supplied_dag(
        relation,
        node_sequences.case_mask,
        topological_order,
        order_valid,
        graph_id=identity,
    )
    nonempty_nodes = jnp.all(
        (~node_sequences.case_mask) | jnp.any(node_sequences.valid_mask, axis=1)
    )
    valid = certificate.valid & nonempty_nodes
    status = jnp.where(
        valid,
        int(VariationGraphStatus.SUCCESS),
        int(VariationGraphStatus.INVALID_GRAPH),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        (
            jnp.sum(node_sequences.case_mask, dtype=jnp.int32),
            jnp.sum(relation.valid, dtype=jnp.int32),
            jnp.sum(
                node_sequences.case_mask & (~jnp.any(node_sequences.valid_mask, axis=1)),
                dtype=jnp.int32,
            ),
            jnp.asarray(~certificate.valid, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return VariationGraphBuildResult(
        graph, certificate, valid, status, evidence, contract
    )


def index_graph_paths(
    graph: VariationGraph,
    certificate: DAGCertificate,
    path_ids: ArrayLike,
    path_node_indices: ArrayLike,
    path_node_valid: ArrayLike,
    path_valid: ArrayLike,
    /,
    *,
    path_node_reverse: ArrayLike | None = None,
) -> GraphPathCoordinates:
    """Build exact path offsets after validating every supplied directed transition."""
    ids = jnp.asarray(path_ids, dtype=jnp.int32)
    nodes = jnp.asarray(path_node_indices, dtype=jnp.int32)
    node_valid = jnp.asarray(path_node_valid, dtype=bool)
    cases = jnp.asarray(path_valid, dtype=bool)
    reverse = (
        jnp.zeros_like(nodes, dtype=bool)
        if path_node_reverse is None
        else jnp.asarray(path_node_reverse, dtype=bool)
    )
    if nodes.ndim != 2 or nodes.shape[1] < 1:
        raise ValueError("path_node_indices must be a rank-2 positive-width array.")
    if node_valid.shape != nodes.shape or reverse.shape != nodes.shape:
        raise ValueError("Path-node masks and orientations must match path nodes.")
    if ids.shape != (nodes.shape[0],) or cases.shape != ids.shape:
        raise ValueError("path_ids and path_valid must match path capacity.")
    if certificate.graph_id != graph.graph_id:
        raise ValueError("DAG certificate and graph identities do not match.")

    node_capacity = graph.relation.source_size
    width = nodes.shape[1]
    in_bounds = (nodes >= 0) & (nodes < node_capacity)
    safe_nodes = jnp.clip(nodes, 0, max(node_capacity - 1, 0))
    prefix = jnp.cumprod(node_valid.astype(jnp.int32), axis=1).astype(bool)
    prefix_ok = jnp.all(prefix == node_valid, axis=1)
    member_ok = jnp.all(
        (~node_valid) | (in_bounds & graph.node_sequences.case_mask[safe_nodes]),
        axis=1,
    )
    node_count = jnp.sum(node_valid, axis=1, dtype=jnp.int32)
    same = safe_nodes[:, :, None] == safe_nodes[:, None, :]
    both = node_valid[:, :, None] & node_valid[:, None, :]
    lower = jnp.arange(width)[:, None] > jnp.arange(width)[None, :]
    repeated = jnp.any(same & both & lower[None, :, :], axis=(1, 2))

    if width > 1:
        transition = node_valid[:, :-1] & node_valid[:, 1:]
        edge_match = (
            (safe_nodes[:, :-1, None] == graph.relation.source_indices[None, None, :])
            & (safe_nodes[:, 1:, None] == graph.relation.target_indices[None, None, :])
            & graph.relation.valid[None, None, :]
        )
        edge_ok = jnp.all((~transition) | jnp.any(edge_match, axis=2), axis=1)
    else:
        edge_ok = jnp.ones_like(cases)

    valid_lengths = jnp.sum(graph.node_sequences.valid_mask, axis=1, dtype=jnp.int32)
    lengths = jnp.where(node_valid, valid_lengths[safe_nodes], 0)
    starts = jnp.cumsum(lengths, axis=1) - lengths
    path_lengths = jnp.sum(lengths, axis=1, dtype=jnp.int32)
    case_valid = (
        cases
        & certificate.valid
        & prefix_ok
        & member_ok
        & (node_count > 0)
        & (~repeated)
        & edge_ok
    )
    valid = jnp.all((~cases) | case_valid)
    status = jnp.where(
        jnp.any(cases & repeated),
        int(VariationGraphStatus.REPEATED_NODE),
        jnp.where(
            valid,
            jnp.where(
                jnp.any(cases),
                int(VariationGraphStatus.SUCCESS),
                int(VariationGraphStatus.EMPTY_INPUT),
            ),
            int(VariationGraphStatus.INVALID_PATH),
        ),
    ).astype(jnp.int32)
    contract = _coordinate_contract()
    evidence = jnp.asarray(
        (
            jnp.sum(cases, dtype=jnp.int32),
            jnp.sum(case_valid, dtype=jnp.int32),
            jnp.sum(cases & repeated, dtype=jnp.int32),
            jnp.sum(cases & (~edge_ok), dtype=jnp.int32),
            jnp.sum(cases & (~prefix_ok), dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return GraphPathCoordinates(
        ids,
        nodes,
        node_valid,
        reverse,
        starts,
        lengths,
        path_lengths,
        case_valid,
        valid,
        status,
        evidence,
        contract,
        graph.graph_id,
    )


def locate_graph_path_positions(
    coordinates: GraphPathCoordinates,
    path_indices: ArrayLike,
    positions: ArrayLike,
    /,
) -> GraphPathLocationResult:
    """Map path offsets to graph-node indices and orientation-aware node offsets."""
    path = jnp.asarray(path_indices, dtype=jnp.int32)
    position = jnp.asarray(positions, dtype=jnp.int32)
    path, position = jnp.broadcast_arrays(path, position)
    path_capacity = coordinates.path_ids.shape[0]
    path_in_bounds = (path >= 0) & (path < path_capacity)
    safe_path = jnp.clip(path, 0, max(path_capacity - 1, 0))
    position_valid = (
        path_in_bounds
        & coordinates.case_valid[safe_path]
        & (position >= 0)
        & (position < coordinates.path_lengths[safe_path])
    )
    starts = coordinates.node_starts[safe_path]
    lengths = coordinates.node_lengths[safe_path]
    contains = (
        coordinates.path_node_valid[safe_path]
        & (position[..., None] >= starts)
        & (position[..., None] < (starts + lengths))
    )
    slot = jnp.argmax(contains.astype(jnp.int32), axis=-1)
    node = jnp.take_along_axis(
        coordinates.path_node_indices[safe_path], slot[..., None], axis=-1
    )[..., 0]
    reverse = jnp.take_along_axis(
        coordinates.path_node_reverse[safe_path], slot[..., None], axis=-1
    )[..., 0]
    start = jnp.take_along_axis(starts, slot[..., None], axis=-1)[..., 0]
    length = jnp.take_along_axis(lengths, slot[..., None], axis=-1)[..., 0]
    forward_offset = position - start
    offset = jnp.where(reverse, length - 1 - forward_offset, forward_offset)
    node = jnp.where(position_valid, node, -1)
    offset = jnp.where(position_valid, offset, -1)
    reverse = reverse & position_valid
    status = jnp.where(
        position_valid,
        int(VariationGraphStatus.SUCCESS),
        int(VariationGraphStatus.OUT_OF_RANGE),
    ).astype(jnp.int32)
    contract = _lookup_contract()
    evidence = jnp.asarray(
        (
            jnp.sum(position_valid, dtype=jnp.int32),
            jnp.sum(~position_valid, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return GraphPathLocationResult(
        node,
        offset,
        reverse,
        position_valid,
        status,
        evidence,
        contract,
        coordinates.graph_id,
    )


def decode_graph_paths(
    graph: VariationGraph,
    coordinates: GraphPathCoordinates,
    /,
    *,
    output_capacity: int,
) -> GraphPathDecodeResult:
    """Decode validated graph paths into fixed-width sequences without truncation."""
    capacity = int(output_capacity)
    if capacity < 1:
        raise ValueError("output_capacity must be positive.")
    if coordinates.graph_id != graph.graph_id:
        raise ValueError("Path-coordinate and graph identities do not match.")
    path_capacity, width = coordinates.path_node_indices.shape
    read_width = graph.node_sequences.token_codes.shape[1]
    safe_nodes = jnp.clip(
        coordinates.path_node_indices,
        0,
        max(graph.relation.source_size - 1, 0),
    )
    regular = graph.node_sequences.token_codes[safe_nodes]
    reversed_tokens = reverse_complement(graph.node_sequences).token_codes[safe_nodes]
    node_tokens = jnp.where(
        coordinates.path_node_reverse[:, :, None], reversed_tokens, regular
    )
    within_capacity = coordinates.path_lengths <= capacity
    case_valid = coordinates.case_valid & within_capacity
    pad_code = graph.node_sequences.alphabet.code(
        graph.node_sequences.alphabet.pad_symbol
    )

    def write_one(
        tokens: Array,
        node_valid: Array,
        starts: Array,
        lengths: Array,
        accepted: Array,
    ) -> Array:
        output = jnp.full(
            (capacity,), pad_code, dtype=graph.node_sequences.token_codes.dtype
        )

        def body(flat_index: int, current: Array) -> Array:
            node_slot = flat_index // read_width
            position = flat_index % read_width
            write = accepted & node_valid[node_slot] & (position < lengths[node_slot])
            destination = starts[node_slot] + position
            safe_destination = jnp.clip(destination, 0, capacity - 1)
            previous = current[safe_destination]
            value = jnp.where(write, tokens[node_slot, position], previous)
            return current.at[safe_destination].set(value)

        return jax.lax.fori_loop(0, width * read_width, body, output)

    output_tokens = jax.vmap(write_one)(
        node_tokens,
        coordinates.path_node_valid,
        coordinates.node_starts,
        coordinates.node_lengths,
        case_valid,
    )
    output_mask = (
        jnp.arange(capacity, dtype=jnp.int32)[None, :] < coordinates.path_lengths[:, None]
    ) & case_valid[:, None]
    sequences = SequenceBatch(
        coordinates.path_ids,
        output_tokens,
        output_mask,
        case_valid,
        jnp.zeros_like(output_mask),
        graph.node_sequences.alphabet,
    )
    valid = jnp.all((~coordinates.case_valid) | case_valid)
    status = jnp.where(
        valid,
        jnp.where(
            jnp.any(case_valid),
            int(VariationGraphStatus.SUCCESS),
            int(VariationGraphStatus.EMPTY_INPUT),
        ),
        int(VariationGraphStatus.CAPACITY_EXCEEDED),
    ).astype(jnp.int32)
    contract = _decode_contract()
    evidence = jnp.asarray(
        (
            jnp.sum(coordinates.case_valid, dtype=jnp.int32),
            jnp.sum(case_valid, dtype=jnp.int32),
            jnp.sum(coordinates.case_valid & (~within_capacity), dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return GraphPathDecodeResult(
        sequences,
        coordinates.path_lengths,
        valid,
        status,
        evidence,
        contract,
        graph.graph_id,
    )


__all__ = [
    "GraphPathCoordinates",
    "GraphPathDecodeResult",
    "GraphPathLocationResult",
    "VariationGraph",
    "VariationGraphBuildResult",
    "VariationGraphStatus",
    "build_supplied_variation_graph",
    "decode_graph_paths",
    "index_graph_paths",
    "locate_graph_path_positions",
]
