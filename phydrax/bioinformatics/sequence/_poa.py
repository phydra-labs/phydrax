#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import heapq
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax.combinatorial import ShortestPathSpace
from phydrax.sparse import EdgeRelation, RowRelation

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._alphabet import AlphabetPlan
from ._batch import SequenceBatch
from ._motifs import _observation_support


POA_MATCH = 0
POA_INSERT = 1
POA_DELETE = 2

POA_STATUS_VALID = 0
POA_STATUS_CAPACITY_EXCEEDED = 1
POA_STATUS_INFEASIBLE = 2


def _poa_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "bounded supplied-DAG partial-order alignment",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Exact affine-gap dynamic programming over every path in the supplied "
            "acyclic graph and every active query position."
        ),
        truncation_statement="No graph branch, query position, or traceback state is pruned.",
        capacity_semantics=(
            "Graph vertices, edges, incoming width, query length, and decoded path "
            "must fit the declared bounds before the kernel executes."
        ),
        assumptions=("Node labels and query tokens share one encoded alphabet.",),
        nondifferentiable_outputs=(
            "node_indices",
            "query_indices",
            "operations",
            "status",
        ),
    )


def _topology(
    relation: EdgeRelation, vertex_count: int, /
) -> tuple[np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    source = np.asarray(relation.source_indices)
    target = np.asarray(relation.target_indices)
    valid = np.asarray(relation.valid, dtype=bool)
    indegree = np.zeros((vertex_count,), dtype=np.int32)
    outdegree = np.zeros((vertex_count,), dtype=np.int32)
    outgoing: list[list[int]] = [[] for _ in range(vertex_count)]
    incoming: list[list[int]] = [[] for _ in range(vertex_count)]
    for edge in np.nonzero(valid)[0].tolist():
        left = int(source[edge])
        right = int(target[edge])
        indegree[right] += 1
        outdegree[left] += 1
        outgoing[left].append(right)
        incoming[right].append(left)
    remaining = indegree.copy()
    ready = [int(value) for value in np.nonzero(remaining == 0)[0].tolist()]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        vertex = heapq.heappop(ready)
        order.append(vertex)
        for right in sorted(outgoing[vertex]):
            remaining[right] -= 1
            if remaining[right] == 0:
                heapq.heappush(ready, right)
    if len(order) != vertex_count:
        raise ValueError("Partial-order alignment requires an acyclic supplied graph.")
    return np.asarray(order, dtype=np.int32), incoming, indegree, outdegree


class PartialOrderGraph(StrictModule):
    """A supplied, alphabet-labelled DAG backed by native sparse path primitives."""

    node_tokens: Array
    relation: EdgeRelation
    topological_order: Array
    predecessors: RowRelation
    start_mask: Array
    end_mask: Array
    adjacency: Array
    path_space: ShortestPathSpace
    alphabet: AlphabetPlan = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)
    incoming_width: int = eqx.field(static=True)

    def __init__(
        self,
        node_tokens: ArrayLike,
        relation: EdgeRelation,
        alphabet: AlphabetPlan,
        *,
        start_mask: ArrayLike | None = None,
        end_mask: ArrayLike | None = None,
        graph_id: str = "partial-order-graph",
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be an EdgeRelation.")
        tokens = jnp.asarray(node_tokens)
        if tokens.ndim != 1 or not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("node_tokens must be a one-dimensional integer array.")
        vertex_count = int(tokens.shape[0])
        if vertex_count <= 0:
            raise ValueError("A partial-order graph requires at least one node.")
        if relation.source_size != vertex_count or relation.target_size != vertex_count:
            raise ValueError("Graph relation spaces must match the node count.")
        concrete_tokens = (
            None if isinstance(tokens, jax_core.Tracer) else np.asarray(tokens)
        )
        if concrete_tokens is None:
            raise TypeError(
                "Graph topology and node labels must be concrete at preparation."
            )
        if np.any(concrete_tokens < 0) or np.any(concrete_tokens >= alphabet.size):
            raise ValueError("node_tokens contains a code outside the alphabet.")
        _, scorable = _observation_support(alphabet)
        if np.any(~np.asarray(scorable)[concrete_tokens]):
            raise ValueError("POA nodes cannot contain gap, pad, or nonscorable symbols.")

        order, incoming, indegree, outdegree = _topology(relation, vertex_count)
        inferred_start = indegree == 0
        inferred_end = outdegree == 0
        starts = (
            inferred_start if start_mask is None else np.asarray(start_mask, dtype=bool)
        )
        ends = inferred_end if end_mask is None else np.asarray(end_mask, dtype=bool)
        if starts.shape != (vertex_count,) or ends.shape != (vertex_count,):
            raise ValueError(
                "start_mask and end_mask must have one entry per graph node."
            )
        if not np.any(starts) or not np.any(ends):
            raise ValueError(
                "A partial-order graph needs at least one start and one end node."
            )
        if np.any(starts & ~inferred_start) or np.any(ends & ~inferred_end):
            raise ValueError(
                "Declared starts and ends must be topological sources and sinks."
            )

        source_vertex = vertex_count
        sink_vertex = vertex_count + 1
        original_source = np.asarray(relation.source_indices, dtype=np.int32)
        original_target = np.asarray(relation.target_indices, dtype=np.int32)
        original_valid = np.asarray(relation.valid, dtype=bool)
        start_nodes = np.nonzero(starts)[0].astype(np.int32)
        end_nodes = np.nonzero(ends)[0].astype(np.int32)
        augmented_source = jnp.asarray(
            np.concatenate(
                (
                    original_source,
                    np.full(start_nodes.shape, source_vertex, dtype=np.int32),
                    end_nodes,
                )
            )
        )
        augmented_target = jnp.asarray(
            np.concatenate(
                (
                    original_target,
                    start_nodes,
                    np.full(end_nodes.shape, sink_vertex, dtype=np.int32),
                )
            )
        )
        augmented_valid = jnp.asarray(
            np.concatenate(
                (
                    original_valid,
                    np.ones(start_nodes.shape, dtype=bool),
                    np.ones(end_nodes.shape, dtype=bool),
                )
            )
        )
        augmented = EdgeRelation(
            augmented_source,
            augmented_target,
            source_size=vertex_count + 2,
            target_size=vertex_count + 2,
            valid=augmented_valid,
        )
        path_space = ShortestPathSpace(augmented, source_vertex, sink_vertex)
        if not path_space.acyclic:
            raise ValueError(
                "Partial-order alignment requires an acyclic supplied graph."
            )

        incoming_with_source = [
            ([source_vertex] if starts[vertex] else []) + sorted(incoming[vertex])
            for vertex in range(vertex_count)
        ]
        width = max(len(values) for values in incoming_with_source)
        predecessor_indices = np.zeros((vertex_count, width), dtype=np.int32)
        predecessor_valid = np.zeros((vertex_count, width), dtype=bool)
        for vertex, values in enumerate(incoming_with_source):
            predecessor_indices[vertex, : len(values)] = values
            predecessor_valid[vertex, : len(values)] = True
        predecessors = RowRelation(
            predecessor_indices,
            source_size=vertex_count + 1,
            valid=predecessor_valid,
        )
        adjacency = np.zeros((vertex_count, vertex_count), dtype=bool)
        adjacency[original_source[original_valid], original_target[original_valid]] = True
        identifier = str(graph_id).strip()
        if not identifier:
            raise ValueError("graph_id must be non-empty.")

        self.node_tokens = tokens.astype(jnp.int32)
        self.relation = relation
        self.topological_order = jnp.asarray(order)
        self.predecessors = predecessors
        self.start_mask = jnp.asarray(starts)
        self.end_mask = jnp.asarray(ends)
        self.adjacency = jnp.asarray(adjacency)
        self.path_space = path_space
        self.alphabet = alphabet
        self.graph_id = identifier
        self.incoming_width = width
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "partial-order-graph",
                "graph_id": identifier,
                "alphabet": alphabet.fingerprint,
                "node_tokens": array_tree_fingerprint(tokens),
                "relation": array_tree_fingerprint(
                    (
                        relation.source_indices,
                        relation.target_indices,
                        relation.valid,
                    )
                ),
                "starts": tuple(bool(value) for value in starts),
                "ends": tuple(bool(value) for value in ends),
            }
        )

    @property
    def node_count(self) -> int:
        return int(self.node_tokens.shape[0])

    @property
    def edge_count(self) -> int:
        return self.relation.capacity


class PartialOrderAlignmentPlan(StrictModule):
    """Resource bounds and affine scoring for exact supplied-DAG alignment."""

    maximum_nodes: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_incoming_width: int = eqx.field(static=True)
    maximum_query_length: int = eqx.field(static=True)
    maximum_alignment_length: int = eqx.field(static=True)
    match_score: float = eqx.field(static=True)
    mismatch_score: float = eqx.field(static=True)
    gap_open_score: float = eqx.field(static=True)
    gap_extend_score: float = eqx.field(static=True)
    tie_policy: str = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(
        self,
        maximum_nodes: int,
        maximum_edges: int,
        maximum_incoming_width: int,
        maximum_query_length: int,
        *,
        maximum_alignment_length: int | None = None,
        match_score: float = 2.0,
        mismatch_score: float = -2.0,
        gap_open_score: float = -3.0,
        gap_extend_score: float = -1.0,
    ):
        raw_bounds = (
            maximum_nodes,
            maximum_edges,
            maximum_incoming_width,
            maximum_query_length,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_bounds
        ):
            raise TypeError("POA resource bounds must be integers.")
        if any(int(value) <= 0 for value in raw_bounds):
            raise ValueError("POA resource bounds must be positive.")
        alignment_bound = (
            int(maximum_nodes) + int(maximum_query_length)
            if maximum_alignment_length is None
            else int(maximum_alignment_length)
        )
        if alignment_bound <= 0:
            raise ValueError("maximum_alignment_length must be positive.")
        scores = (
            float(match_score),
            float(mismatch_score),
            float(gap_open_score),
            float(gap_extend_score),
        )
        if any(not np.isfinite(value) for value in scores):
            raise ValueError("POA scores must be finite.")
        self.maximum_nodes = int(maximum_nodes)
        self.maximum_edges = int(maximum_edges)
        self.maximum_incoming_width = int(maximum_incoming_width)
        self.maximum_query_length = int(maximum_query_length)
        self.maximum_alignment_length = alignment_bound
        self.match_score, self.mismatch_score = scores[:2]
        self.gap_open_score, self.gap_extend_score = scores[2:]
        self.tie_policy = "lowest-predecessor-then-match-insert-delete"
        self.method_contract = _poa_contract()


class PartialOrderAlignmentPath(StrictModule):
    node_indices: Array
    query_indices: Array
    operations: Array
    valid: Array
    length: Array


class PartialOrderAlignmentEvidence(StrictModule):
    """Capacity, source-to-sink, decoding, and score feasibility evidence."""

    acyclic: Array
    capacity_sufficient: Array
    reaches_source: Array
    reaches_sink: Array
    graph_edges_feasible: Array
    query_consumed: Array
    expected_query_length: Array
    score_residual: Array


class PartialOrderAlignmentResult(StrictModule):
    score: Array
    path: PartialOrderAlignmentPath
    valid: Array
    status: Array
    evidence: PartialOrderAlignmentEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _compatibility_table(alphabet: AlphabetPlan, match: float, mismatch: float) -> Array:
    support, scorable = _observation_support(alphabet)
    support = support.astype(jnp.float32)
    totals = jnp.sum(support, axis=1, keepdims=True)
    distributions = jnp.where(totals > 0.0, support / jnp.maximum(totals, 1.0), 0.0)
    compatibility = distributions @ distributions.T
    score = mismatch + (match - mismatch) * compatibility
    return jnp.where(scorable[:, None] & scorable[None, :], score, -jnp.inf)


def _capacity_sufficient(
    graph: PartialOrderGraph,
    sequences: SequenceBatch,
    plan: PartialOrderAlignmentPlan,
    /,
) -> bool:
    return (
        graph.node_count <= plan.maximum_nodes
        and graph.edge_count <= plan.maximum_edges
        and graph.incoming_width <= plan.maximum_incoming_width
        and sequences.sequence_capacity <= plan.maximum_query_length
        and graph.node_count + sequences.sequence_capacity
        <= plan.maximum_alignment_length
    )


def _poa_single(
    graph: PartialOrderGraph,
    tokens: Array,
    mask: Array,
    plan: PartialOrderAlignmentPlan,
    score_table: Array,
    /,
) -> tuple[Array, PartialOrderAlignmentPath, Array, Array, Array, Array, Array]:
    n = int(tokens.shape[0])
    vertices = graph.node_count
    source_node = vertices
    width = graph.incoming_width
    dtype = score_table.dtype
    values = jnp.full((n + 1, vertices + 1, 3), -jnp.inf, dtype=dtype)
    predecessor_node = jnp.full((n + 1, vertices + 1, 3), -1, dtype=jnp.int32)
    predecessor_class = jnp.full((n + 1, vertices + 1, 3), -1, dtype=jnp.int8)
    pointer_kind = jnp.full((n + 1, vertices + 1, 3), -1, dtype=jnp.int8)
    values = values.at[0, source_node, POA_MATCH].set(0.0)

    def delete_cell(row, vertex):
        predecessors = graph.predecessors.source_indices[vertex]
        predecessor_valid = graph.predecessors.valid[vertex]
        candidates = row[predecessors, :]
        gap_cost = jnp.where(
            jnp.arange(3)[None, :] == POA_DELETE,
            plan.gap_extend_score,
            plan.gap_open_score,
        )
        flattened = (candidates + gap_cost).reshape((-1,))
        flattened = jnp.where(jnp.repeat(predecessor_valid, 3), flattened, -jnp.inf)
        selected = jnp.argmax(flattened)
        slot = selected // 3
        state_class = selected % 3
        return (
            flattened[selected],
            predecessors[slot],
            state_class.astype(jnp.int8),
        )

    for order_index in range(vertices):
        vertex = graph.topological_order[order_index]
        score, previous_node, previous_class = delete_cell(values[0], vertex)
        values = values.at[0, vertex, POA_DELETE].set(score)
        predecessor_node = predecessor_node.at[0, vertex, POA_DELETE].set(previous_node)
        predecessor_class = predecessor_class.at[0, vertex, POA_DELETE].set(
            previous_class
        )
        pointer_kind = pointer_kind.at[0, vertex, POA_DELETE].set(POA_DELETE)

    for position in range(1, n + 1):
        token = tokens[position - 1]
        row = jnp.full((vertices + 1, 3), -jnp.inf, dtype=dtype)
        row_node = jnp.full((vertices + 1, 3), -1, dtype=jnp.int32)
        row_class = jnp.full((vertices + 1, 3), -1, dtype=jnp.int8)
        row_kind = jnp.full((vertices + 1, 3), -1, dtype=jnp.int8)

        source_candidates = values[position - 1, source_node, :] + jnp.where(
            jnp.arange(3) == POA_INSERT,
            plan.gap_extend_score,
            plan.gap_open_score,
        )
        selected_source = jnp.argmax(source_candidates)
        row = row.at[source_node, POA_INSERT].set(source_candidates[selected_source])
        row_node = row_node.at[source_node, POA_INSERT].set(source_node)
        row_class = row_class.at[source_node, POA_INSERT].set(
            selected_source.astype(jnp.int8)
        )
        row_kind = row_kind.at[source_node, POA_INSERT].set(POA_INSERT)

        for order_index in range(vertices):
            vertex = graph.topological_order[order_index]
            predecessors = graph.predecessors.source_indices[vertex]
            predecessor_valid = graph.predecessors.valid[vertex]
            match_candidates = values[position - 1, predecessors, :].reshape((-1,))
            match_candidates = jnp.where(
                jnp.repeat(predecessor_valid, 3), match_candidates, -jnp.inf
            )
            selected_match = jnp.argmax(match_candidates)
            match_slot = selected_match // 3
            match_class = selected_match % 3
            row = row.at[vertex, POA_MATCH].set(
                match_candidates[selected_match]
                + score_table[graph.node_tokens[vertex], token]
            )
            row_node = row_node.at[vertex, POA_MATCH].set(predecessors[match_slot])
            row_class = row_class.at[vertex, POA_MATCH].set(match_class.astype(jnp.int8))
            row_kind = row_kind.at[vertex, POA_MATCH].set(POA_MATCH)

            insert_candidates = values[position - 1, vertex, :] + jnp.where(
                jnp.arange(3) == POA_INSERT,
                plan.gap_extend_score,
                plan.gap_open_score,
            )
            selected_insert = jnp.argmax(insert_candidates)
            row = row.at[vertex, POA_INSERT].set(insert_candidates[selected_insert])
            row_node = row_node.at[vertex, POA_INSERT].set(vertex)
            row_class = row_class.at[vertex, POA_INSERT].set(
                selected_insert.astype(jnp.int8)
            )
            row_kind = row_kind.at[vertex, POA_INSERT].set(POA_INSERT)

            delete_score, delete_node, delete_class = delete_cell(row, vertex)
            row = row.at[vertex, POA_DELETE].set(delete_score)
            row_node = row_node.at[vertex, POA_DELETE].set(delete_node)
            row_class = row_class.at[vertex, POA_DELETE].set(delete_class)
            row_kind = row_kind.at[vertex, POA_DELETE].set(POA_DELETE)

        active = mask[position - 1]
        values = values.at[position].set(jnp.where(active, row, values[position - 1]))
        predecessor_node = predecessor_node.at[position].set(
            jnp.where(
                active,
                row_node,
                jnp.broadcast_to(
                    jnp.arange(vertices + 1, dtype=jnp.int32)[:, None],
                    (vertices + 1, 3),
                ),
            )
        )
        predecessor_class = predecessor_class.at[position].set(
            jnp.where(
                active,
                row_class,
                jnp.broadcast_to(
                    jnp.arange(3, dtype=jnp.int8)[None, :], (vertices + 1, 3)
                ),
            )
        )
        pointer_kind = pointer_kind.at[position].set(
            jnp.where(
                active,
                row_kind,
                jnp.full((vertices + 1, 3), 3, dtype=jnp.int8),
            )
        )

    terminal_nodes = jnp.nonzero(graph.end_mask, size=vertices, fill_value=0)[0]
    terminal_valid = jnp.arange(vertices) < jnp.sum(graph.end_mask)
    terminal_values = values[n, terminal_nodes, :].reshape((-1,))
    terminal_values = jnp.where(jnp.repeat(terminal_valid, 3), terminal_values, -jnp.inf)
    terminal_choice = jnp.argmax(terminal_values)
    terminal_slot = terminal_choice // 3
    state_node = terminal_nodes[terminal_slot].astype(jnp.int32)
    state_class = (terminal_choice % 3).astype(jnp.int32)
    score = terminal_values[terminal_choice]

    capacity = n + vertices
    reverse_node = jnp.full((capacity,), -1, dtype=jnp.int32)
    reverse_query = jnp.full((capacity,), -1, dtype=jnp.int32)
    reverse_operation = jnp.full((capacity,), -1, dtype=jnp.int8)

    def trace_step(_, state):
        t, node, operation, alive, count, out_node, out_query, out_operation = state
        at_source = alive & (t == 0) & (node == source_node) & (operation == POA_MATCH)
        kind = jnp.where(
            at_source,
            jnp.asarray(-1, dtype=jnp.int8),
            pointer_kind[t, node, operation],
        )
        skip = alive & (kind == 3)
        write_state = alive & ~skip & ~at_source
        destination = capacity - 1 - count
        safe_destination = jnp.maximum(destination, 0)
        reported_node = jnp.where(operation == POA_INSERT, -1, node)
        reported_query = jnp.where(operation == POA_DELETE, -1, t - 1)
        out_node = out_node.at[safe_destination].set(
            jnp.where(write_state, reported_node, out_node[safe_destination])
        )
        out_query = out_query.at[safe_destination].set(
            jnp.where(write_state, reported_query, out_query[safe_destination])
        )
        out_operation = out_operation.at[safe_destination].set(
            jnp.where(
                write_state, operation.astype(jnp.int8), out_operation[safe_destination]
            )
        )
        count = count + write_state.astype(jnp.int32)
        previous_node = jnp.where(at_source, node, predecessor_node[t, node, operation])
        previous_class = jnp.where(
            at_source,
            operation,
            predecessor_class[t, node, operation].astype(jnp.int32),
        )
        previous_t = jnp.where(skip | (operation != POA_DELETE), t - 1, t)
        next_alive = alive & ~at_source & (previous_node >= 0) & (previous_class >= 0)
        return (
            jnp.where(next_alive, previous_t, t),
            jnp.where(next_alive, previous_node, node),
            jnp.where(next_alive, previous_class, operation),
            next_alive,
            count,
            out_node,
            out_query,
            out_operation,
        )

    traced = jax.lax.fori_loop(
        0,
        capacity + 1,
        trace_step,
        (
            jnp.asarray(n, dtype=jnp.int32),
            state_node,
            state_class,
            jnp.isfinite(score),
            jnp.asarray(0, dtype=jnp.int32),
            reverse_node,
            reverse_query,
            reverse_operation,
        ),
    )
    path_length = traced[4]
    packed_source = jnp.arange(capacity, dtype=jnp.int32) + (capacity - path_length)
    packed_source = jnp.clip(packed_source, 0, max(capacity - 1, 0))
    node_path = traced[5][packed_source]
    query_path = traced[6][packed_source]
    operation_path = traced[7][packed_source]
    path_valid = jnp.arange(capacity) < path_length
    node_path = jnp.where(path_valid, node_path, -1)
    query_path = jnp.where(path_valid, query_path, -1)
    operation_path = jnp.where(path_valid, operation_path, -1).astype(jnp.int8)
    path = PartialOrderAlignmentPath(
        node_path, query_path, operation_path, path_valid, path_length
    )

    node_event = path_valid & (node_path >= 0)
    node_count = jnp.sum(node_event, dtype=jnp.int32)
    node_event_positions = jnp.nonzero(node_event, size=capacity, fill_value=0)[0]
    packed_nodes = jnp.where(
        jnp.arange(capacity) < node_count,
        node_path[node_event_positions],
        -1,
    )
    node_positions = jnp.arange(capacity)
    edge_checks = jnp.where(
        (node_positions > 0) & (node_positions < node_count),
        graph.adjacency[
            jnp.maximum(packed_nodes[jnp.maximum(node_positions - 1, 0)], 0),
            jnp.maximum(packed_nodes, 0),
        ],
        True,
    )
    graph_feasible = jnp.all(edge_checks)
    first_node = packed_nodes[0]
    last_node = packed_nodes[jnp.maximum(node_count - 1, 0)]
    reaches_source = (node_count > 0) & graph.start_mask[jnp.maximum(first_node, 0)]
    reaches_sink = (node_count > 0) & graph.end_mask[jnp.maximum(last_node, 0)]
    query_consumed = jnp.sum(path_valid & (query_path >= 0), dtype=jnp.int32)

    previous_operation = jnp.concatenate(
        (jnp.asarray([-1], dtype=jnp.int8), operation_path[:-1])
    )
    matched_score = (
        score_table[
            graph.node_tokens[jnp.maximum(node_path, 0)],
            tokens[jnp.maximum(query_path, 0)],
        ]
        if n > 0
        else jnp.zeros((capacity,), dtype=score.dtype)
    )
    gap_score = jnp.where(
        operation_path == previous_operation,
        plan.gap_extend_score,
        plan.gap_open_score,
    )
    step_score = jnp.where(operation_path == POA_MATCH, matched_score, gap_score)
    recomputed = jnp.sum(jnp.where(path_valid, step_score, 0.0))
    score_residual = jnp.abs(recomputed - score)
    return (
        score,
        path,
        reaches_source,
        reaches_sink,
        graph_feasible,
        query_consumed,
        score_residual,
    )


def _failure_result(
    graph: PartialOrderGraph,
    sequences: SequenceBatch,
    plan: PartialOrderAlignmentPlan,
    /,
) -> PartialOrderAlignmentResult:
    records = sequences.record_capacity
    capacity = graph.node_count + sequences.sequence_capacity
    path = PartialOrderAlignmentPath(
        jnp.full((records, capacity), -1, dtype=jnp.int32),
        jnp.full((records, capacity), -1, dtype=jnp.int32),
        jnp.full((records, capacity), -1, dtype=jnp.int8),
        jnp.zeros((records, capacity), dtype=bool),
        jnp.zeros((records,), dtype=jnp.int32),
    )
    case = jnp.asarray(sequences.case_mask, dtype=bool)
    expected = jnp.sum(sequences.valid_mask, axis=1, dtype=jnp.int32)
    evidence = PartialOrderAlignmentEvidence(
        jnp.ones((records,), dtype=bool),
        jnp.zeros((records,), dtype=bool),
        jnp.zeros((records,), dtype=bool),
        jnp.zeros((records,), dtype=bool),
        jnp.zeros((records,), dtype=bool),
        jnp.zeros((records,), dtype=jnp.int32),
        expected,
        jnp.full((records,), jnp.inf),
    )
    return PartialOrderAlignmentResult(
        jnp.full((records,), -jnp.inf),
        path,
        jnp.zeros((records,), dtype=bool),
        jnp.where(case, POA_STATUS_CAPACITY_EXCEEDED, POA_STATUS_INFEASIBLE).astype(
            jnp.int32
        ),
        evidence,
        plan.method_contract,
    )


def align_partial_order(
    graph: PartialOrderGraph,
    sequences: SequenceBatch,
    plan: PartialOrderAlignmentPlan,
    /,
) -> PartialOrderAlignmentResult:
    """Align every populated query record to its best path through a supplied DAG."""
    if not isinstance(graph, PartialOrderGraph):
        raise TypeError("graph must be a PartialOrderGraph.")
    if not isinstance(sequences, SequenceBatch):
        raise TypeError("sequences must be a SequenceBatch.")
    if not isinstance(plan, PartialOrderAlignmentPlan):
        raise TypeError("plan must be a PartialOrderAlignmentPlan.")
    if graph.alphabet.fingerprint != sequences.alphabet.fingerprint:
        raise ValueError("POA graph and sequence alphabets must match.")
    capacity_ok = _capacity_sufficient(graph, sequences, plan)
    if not capacity_ok:
        return _failure_result(graph, sequences, plan)

    score_table = _compatibility_table(
        graph.alphabet, plan.match_score, plan.mismatch_score
    )
    score, path, begins, ends, graph_ok, consumed, residual = jax.vmap(
        lambda token, mask: _poa_single(graph, token, mask, plan, score_table)
    )(sequences.token_codes, sequences.valid_mask)
    case = jnp.asarray(sequences.case_mask, dtype=bool)
    expected = jnp.sum(sequences.valid_mask, axis=1, dtype=jnp.int32)
    tolerance = 32.0 * jnp.finfo(score.dtype).eps * jnp.maximum(jnp.abs(score), 1.0)
    valid = (
        case
        & jnp.isfinite(score)
        & begins
        & ends
        & graph_ok
        & (consumed == expected)
        & (residual <= tolerance)
    )
    evidence = PartialOrderAlignmentEvidence(
        jnp.ones(case.shape, dtype=bool),
        jnp.ones(case.shape, dtype=bool) & case,
        begins & case,
        ends & case,
        graph_ok & case,
        consumed,
        expected,
        residual,
    )
    status = jnp.where(
        valid,
        POA_STATUS_VALID,
        POA_STATUS_INFEASIBLE,
    ).astype(jnp.int32)
    return PartialOrderAlignmentResult(
        jnp.where(case, score, -jnp.inf),
        path,
        valid,
        status,
        evidence,
        plan.method_contract,
    )


__all__ = [
    "POA_DELETE",
    "POA_INSERT",
    "POA_MATCH",
    "POA_STATUS_CAPACITY_EXCEEDED",
    "POA_STATUS_INFEASIBLE",
    "POA_STATUS_VALID",
    "PartialOrderAlignmentEvidence",
    "PartialOrderAlignmentPath",
    "PartialOrderAlignmentPlan",
    "PartialOrderAlignmentResult",
    "PartialOrderGraph",
    "align_partial_order",
]
