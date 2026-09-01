#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import reverse_complement, SequenceBatch


class AssemblyStatus(IntEnum):
    """Status codes shared by fixed-capacity assembly kernels."""

    SUCCESS = 0
    INVALID_CANDIDATE = 1
    CAPACITY_EXCEEDED = 2
    INVALID_DAG_ORDER = 3
    REPEATED_NODE = 4
    CYCLE = 5
    INVALID_PATH = 6
    NON_UNITIG_PATH = 7
    EMPTY_INPUT = 8


def _supplied_candidate_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "supplied overlap-candidate boundary",
        MethodKind.HEURISTIC,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.SET,
        conditioning_statement=(
            "The candidate relation and its provenance are supplied by the caller."
        ),
        truncation_statement=(
            "This boundary makes no completeness claim about candidate retrieval."
        ),
        capacity_semantics=(
            "Every supplied route occupies one explicit slot; inputs are never truncated."
        ),
        assumptions=("Candidate retrieval was performed outside this kernel.",),
        nondifferentiable_outputs=(
            "source_indices",
            "target_indices",
            "source_reverse",
            "target_reverse",
        ),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _overlap_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact supplied-candidate overlap scoring",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.GRAPH,
        conditioning_statement=(
            "Encoded symbols are compared exactly after the declared orientation transform."
        ),
        truncation_statement=(
            "The longest overlap is exact within each supplied candidate; candidate retrieval "
            "is outside the claim."
        ),
        capacity_semantics=(
            "Read length and edge capacities are fixed by input shapes; no accepted route is "
            "silently dropped."
        ),
        assumptions=(
            "Sequence valid masks are left-prefix masks.",
            "All reads use one alphabet with a defined reverse complement.",
        ),
        nondifferentiable_outputs=("relation", "overlap_lengths", "orientations"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _dag_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "supplied topological-order certification",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Every active node must occur exactly once in the supplied topological order."
        ),
        truncation_statement="All valid graph routes are certified; there is no truncation.",
        capacity_semantics="The order capacity equals the graph node capacity.",
        assumptions=("The graph relation is directed from source to target.",),
        nondifferentiable_outputs=("topological_order", "node_rank", "status"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _decode_contract(*, unitig: bool) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "certified supplied-unitig decoding"
        if unitig
        else "certified supplied-path decoding",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.SEQUENCE,
        conditioning_statement=(
            "Decoded paths must use certified DAG edges with exactly matching orientations and "
            "overlap symbols."
        ),
        truncation_statement=(
            "A path exceeding output capacity fails as a whole and is never truncated."
        ),
        capacity_semantics=(
            "Path count, path width, read length, and output length are fixed capacities."
        ),
        assumptions=(
            "Supplied unitigs form a complete maximal non-branching node partition."
            if unitig
            else "Supplied paths are intended graph walks without repeated nodes.",
        ),
        nondifferentiable_outputs=("token_codes", "valid_mask", "status"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


class SuppliedOverlapCandidates(StrictModule, NonTrainableState):
    """Fixed-capacity externally retrieved overlap candidates.

    Candidate retrieval is deliberately not implemented here: production retrieval is a
    heuristic or external database boundary, while scoring each supplied route is exact.
    """

    relation: EdgeRelation
    source_reverse: Array
    target_reverse: Array
    method_contract: BioinformaticsMethodContract
    provenance: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        source_reverse: ArrayLike,
        target_reverse: ArrayLike,
        /,
        *,
        read_capacity: int,
        valid: ArrayLike | None = None,
        provenance: str = "caller-supplied",
        candidate_id: str | None = None,
        method_contract: BioinformaticsMethodContract | None = None,
    ):
        relation = EdgeRelation(
            source_indices,
            target_indices,
            source_size=int(read_capacity),
            target_size=int(read_capacity),
            valid=valid,
        )
        source_orientation = jnp.asarray(source_reverse, dtype=bool)
        target_orientation = jnp.asarray(target_reverse, dtype=bool)
        if source_orientation.shape != relation.route_shape:
            raise ValueError("source_reverse must match candidate route capacity.")
        if target_orientation.shape != relation.route_shape:
            raise ValueError("target_reverse must match candidate route capacity.")
        provenance_ = str(provenance).strip()
        if not provenance_:
            raise ValueError("Candidate provenance must be non-empty.")
        identity = candidate_id or canonical_fingerprint(
            {
                "kind": "supplied-overlap-candidates",
                "read_capacity": relation.source_size,
                "routes": array_tree_fingerprint(
                    (
                        relation.source_indices,
                        relation.target_indices,
                        relation.valid,
                        source_orientation,
                        target_orientation,
                    )
                ),
                "provenance": provenance_,
            }
        )
        if not identity:
            raise ValueError("candidate_id must be non-empty.")
        self.relation = relation
        self.source_reverse = source_orientation
        self.target_reverse = target_orientation
        self.method_contract = method_contract or _supplied_candidate_contract()
        self.provenance = provenance_
        self.candidate_id = identity


class OverlapGraph(StrictModule):
    """A read-overlap graph whose sole topology is a native edge relation."""

    relation: EdgeRelation
    overlap_lengths: Array
    source_reverse: Array
    target_reverse: Array
    node_valid: Array
    method_contract: BioinformaticsMethodContract
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        overlap_lengths: ArrayLike,
        source_reverse: ArrayLike,
        target_reverse: ArrayLike,
        node_valid: ArrayLike,
        method_contract: BioinformaticsMethodContract,
        graph_id: str,
        /,
    ):
        if relation.source_size != relation.target_size:
            raise ValueError("An overlap graph requires one shared read-node space.")
        overlap = jnp.asarray(overlap_lengths, dtype=jnp.int32)
        source_orientation = jnp.asarray(source_reverse, dtype=bool)
        target_orientation = jnp.asarray(target_reverse, dtype=bool)
        active = jnp.asarray(node_valid, dtype=bool)
        if overlap.shape != relation.route_shape:
            raise ValueError("overlap_lengths must match the edge capacity.")
        if source_orientation.shape != relation.route_shape:
            raise ValueError("source_reverse must match the edge capacity.")
        if target_orientation.shape != relation.route_shape:
            raise ValueError("target_reverse must match the edge capacity.")
        if active.shape != (relation.source_size,):
            raise ValueError("node_valid must match the read-node capacity.")
        if not graph_id:
            raise ValueError("graph_id must be non-empty.")
        self.relation = relation
        self.overlap_lengths = overlap
        self.source_reverse = source_orientation
        self.target_reverse = target_orientation
        self.node_valid = active
        self.method_contract = method_contract
        self.graph_id = graph_id


class OverlapScoringResult(StrictModule):
    graph: OverlapGraph
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


class DAGCertificate(StrictModule, NonTrainableState):
    """Array-valued certificate that a supplied node order covers one directed DAG."""

    topological_order: Array
    order_valid: Array
    node_rank: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    graph_id: str = eqx.field(static=True)


class SuppliedAssemblyPaths(StrictModule, NonTrainableState):
    """Fixed-width caller-supplied paths; discovery remains an external boundary."""

    path_ids: Array
    node_indices: Array
    node_valid: Array
    node_reverse: Array
    path_valid: Array
    path_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        path_ids: ArrayLike,
        node_indices: ArrayLike,
        node_valid: ArrayLike,
        node_reverse: ArrayLike,
        path_valid: ArrayLike,
        /,
        *,
        path_set_id: str | None = None,
    ):
        ids = jnp.asarray(path_ids, dtype=jnp.int32)
        nodes = jnp.asarray(node_indices, dtype=jnp.int32)
        valid = jnp.asarray(node_valid, dtype=bool)
        reverse = jnp.asarray(node_reverse, dtype=bool)
        active = jnp.asarray(path_valid, dtype=bool)
        if nodes.ndim != 2 or nodes.shape[1] < 1:
            raise ValueError(
                "node_indices must have shape (path_capacity, positive_width)."
            )
        if valid.shape != nodes.shape or reverse.shape != nodes.shape:
            raise ValueError("Path node masks and orientations must match node_indices.")
        if ids.shape != (nodes.shape[0],) or active.shape != ids.shape:
            raise ValueError("path_ids and path_valid must match path capacity.")
        identity = path_set_id or canonical_fingerprint(
            {
                "kind": "supplied-assembly-paths",
                "paths": array_tree_fingerprint((ids, nodes, valid, reverse, active)),
            }
        )
        if not identity:
            raise ValueError("path_set_id must be non-empty.")
        self.path_ids = ids
        self.node_indices = nodes
        self.node_valid = valid
        self.node_reverse = reverse
        self.path_valid = active
        self.path_set_id = identity


class AssemblyDecodeResult(StrictModule):
    sequences: SequenceBatch
    path_lengths: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


def _longest_overlaps(
    source_tokens: Array,
    target_tokens: Array,
    source_lengths: Array,
    target_lengths: Array,
    route_valid: Array,
    /,
) -> Array:
    edge_capacity, read_width = source_tokens.shape
    positions = jnp.arange(read_width, dtype=jnp.int32)[None, :]

    def body(length: int, best: Array) -> Array:
        length_ = jnp.asarray(length, dtype=jnp.int32)
        source_position = source_lengths[:, None] - length_ + positions
        safe_source = jnp.clip(source_position, 0, max(read_width - 1, 0))
        source_symbol = jnp.take_along_axis(source_tokens, safe_source, axis=1)
        compared = positions < length_
        symbols_match = jnp.all((~compared) | (source_symbol == target_tokens), axis=1)
        feasible = (
            route_valid
            & (length_ <= source_lengths)
            & (length_ <= target_lengths)
            & symbols_match
        )
        return jnp.where(feasible, length_, best)

    initial = jnp.zeros((edge_capacity,), dtype=jnp.int32)
    return jax.lax.fori_loop(1, read_width + 1, body, initial)


def score_supplied_overlaps(
    reads: SequenceBatch,
    candidates: SuppliedOverlapCandidates,
    /,
    *,
    minimum_overlap: int,
) -> OverlapScoringResult:
    """Exactly score every externally supplied oriented suffix-prefix candidate."""
    threshold = int(minimum_overlap)
    if threshold < 1:
        raise ValueError("minimum_overlap must be positive.")
    if reads.token_codes.ndim != 2:
        raise ValueError("reads must contain a rank-2 token array.")
    read_capacity, read_width = reads.token_codes.shape
    if candidates.relation.source_size != read_capacity:
        raise ValueError("Candidate read capacity must match SequenceBatch capacity.")

    relation = candidates.relation
    route_valid = relation.valid
    safe_source = jnp.where(route_valid, relation.source_indices, 0)
    safe_target = jnp.where(route_valid, relation.target_indices, 0)
    active_route = (
        route_valid
        & reads.case_mask[safe_source]
        & reads.case_mask[safe_target]
        & (safe_source != safe_target)
    )

    reverse_reads = reverse_complement(reads)
    source_regular = reads.token_codes[safe_source]
    source_reversed = reverse_reads.token_codes[safe_source]
    target_regular = reads.token_codes[safe_target]
    target_reversed = reverse_reads.token_codes[safe_target]
    source_tokens = jnp.where(
        candidates.source_reverse[:, None], source_reversed, source_regular
    )
    target_tokens = jnp.where(
        candidates.target_reverse[:, None], target_reversed, target_regular
    )
    lengths = jnp.sum(reads.valid_mask, axis=1, dtype=jnp.int32)
    source_lengths = lengths[safe_source]
    target_lengths = lengths[safe_target]
    overlaps = _longest_overlaps(
        source_tokens,
        target_tokens,
        source_lengths,
        target_lengths,
        active_route,
    )

    same_route = (
        (relation.source_indices[:, None] == relation.source_indices[None, :])
        & (relation.target_indices[:, None] == relation.target_indices[None, :])
        & (candidates.source_reverse[:, None] == candidates.source_reverse[None, :])
        & (candidates.target_reverse[:, None] == candidates.target_reverse[None, :])
        & route_valid[:, None]
        & route_valid[None, :]
    )
    earlier = (
        jnp.arange(relation.capacity)[:, None] > jnp.arange(relation.capacity)[None, :]
    )
    duplicate = jnp.any(same_route & earlier, axis=1)
    invalid_candidate = route_valid & ((~active_route) | duplicate)
    accepted = active_route & (~duplicate) & (overlaps >= threshold)
    scored_relation = EdgeRelation(
        relation.source_indices,
        relation.target_indices,
        source_size=read_capacity,
        target_size=read_capacity,
        valid=accepted,
    )
    contract = _overlap_contract()
    graph = OverlapGraph(
        scored_relation,
        overlaps,
        candidates.source_reverse,
        candidates.target_reverse,
        reads.case_mask,
        contract,
        canonical_fingerprint(
            {
                "kind": "exact-overlap-graph",
                "candidates": candidates.candidate_id,
                "minimum_overlap": threshold,
                "alphabet": reads.alphabet.fingerprint,
            }
        ),
    )
    valid = ~jnp.any(invalid_candidate)
    status = jnp.where(
        valid,
        int(AssemblyStatus.SUCCESS),
        int(AssemblyStatus.INVALID_CANDIDATE),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        (
            jnp.sum(route_valid, dtype=jnp.int32),
            jnp.sum(accepted, dtype=jnp.int32),
            jnp.sum(
                active_route & (~duplicate) & (overlaps < threshold), dtype=jnp.int32
            ),
            jnp.sum(invalid_candidate, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return OverlapScoringResult(graph, valid, status, evidence, contract)


def certify_supplied_dag(
    relation: EdgeRelation,
    node_valid: ArrayLike,
    topological_order: ArrayLike,
    order_valid: ArrayLike,
    /,
    *,
    graph_id: str,
) -> DAGCertificate:
    """Certify a complete supplied order, rejecting repeated nodes and back edges."""
    if relation.source_size != relation.target_size:
        raise ValueError("DAG certification requires one shared node space.")
    node_capacity = relation.source_size
    active = jnp.asarray(node_valid, dtype=bool)
    order = jnp.asarray(topological_order, dtype=jnp.int32)
    supplied = jnp.asarray(order_valid, dtype=bool)
    if active.shape != (node_capacity,):
        raise ValueError("node_valid must match node capacity.")
    if order.shape != (node_capacity,) or supplied.shape != order.shape:
        raise ValueError("The supplied order and mask must match node capacity.")

    in_bounds = (order >= 0) & (order < node_capacity)
    safe_order = jnp.clip(order, 0, max(node_capacity - 1, 0))
    usable = supplied & in_bounds
    occurrence = jnp.sum(
        jax.nn.one_hot(safe_order, node_capacity, dtype=jnp.int32)
        * usable[:, None].astype(jnp.int32),
        axis=0,
        dtype=jnp.int32,
    )
    duplicate_count = jnp.sum(jnp.maximum(occurrence - 1, 0), dtype=jnp.int32)
    missing_count = jnp.sum(active & (occurrence == 0), dtype=jnp.int32)
    inactive_count = jnp.sum((occurrence > 0) & (~active), dtype=jnp.int32)
    range_count = jnp.sum(supplied & (~in_bounds), dtype=jnp.int32)

    positions = jnp.arange(node_capacity, dtype=jnp.int32)
    ranks = jnp.full((node_capacity,), node_capacity, dtype=jnp.int32)
    ranks = ranks.at[safe_order].min(jnp.where(usable, positions, node_capacity))
    edge_source = jnp.where(relation.valid, relation.source_indices, 0)
    edge_target = jnp.where(relation.valid, relation.target_indices, 0)
    endpoint_invalid = relation.valid & ((~active[edge_source]) | (~active[edge_target]))
    back_edge = relation.valid & (ranks[edge_source] >= ranks[edge_target])
    endpoint_invalid_count = jnp.sum(endpoint_invalid, dtype=jnp.int32)
    back_edge_count = jnp.sum(back_edge, dtype=jnp.int32)
    ordering_count = inactive_count + range_count + endpoint_invalid_count

    repeated = duplicate_count > 0
    bad_order = (missing_count > 0) | (ordering_count > 0)
    cyclic = back_edge_count > 0
    valid = (~repeated) & (~bad_order) & (~cyclic)
    status = jnp.where(
        repeated,
        int(AssemblyStatus.REPEATED_NODE),
        jnp.where(
            bad_order,
            int(AssemblyStatus.INVALID_DAG_ORDER),
            jnp.where(
                cyclic,
                int(AssemblyStatus.CYCLE),
                int(AssemblyStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        (
            jnp.sum(active, dtype=jnp.int32),
            jnp.sum(relation.valid, dtype=jnp.int32),
            duplicate_count,
            missing_count,
            ordering_count,
            back_edge_count,
        ),
        dtype=jnp.int32,
    )
    return DAGCertificate(
        order,
        supplied,
        ranks,
        valid,
        status,
        evidence,
        _dag_contract(),
        graph_id,
    )


def _path_checks(
    graph: OverlapGraph,
    certificate: DAGCertificate,
    paths: SuppliedAssemblyPaths,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    path_capacity, width = paths.node_indices.shape
    node_capacity = graph.relation.source_size
    prefix = jnp.cumprod(paths.node_valid.astype(jnp.int32), axis=1).astype(bool)
    prefix_valid = jnp.all(paths.node_valid == prefix, axis=1)
    node_count = jnp.sum(paths.node_valid, axis=1, dtype=jnp.int32)
    nonempty = node_count > 0
    in_bounds = (paths.node_indices >= 0) & (paths.node_indices < node_capacity)
    safe_nodes = jnp.clip(paths.node_indices, 0, max(node_capacity - 1, 0))
    node_membership = jnp.all(
        (~paths.node_valid) | (in_bounds & graph.node_valid[safe_nodes]), axis=1
    )
    same_node = safe_nodes[:, :, None] == safe_nodes[:, None, :]
    both_valid = paths.node_valid[:, :, None] & paths.node_valid[:, None, :]
    lower = jnp.arange(width)[:, None] > jnp.arange(width)[None, :]
    repeated = jnp.any(same_node & both_valid & lower[None, :, :], axis=(1, 2))

    if width > 1:
        left_node = safe_nodes[:, :-1]
        right_node = safe_nodes[:, 1:]
        left_reverse = paths.node_reverse[:, :-1]
        right_reverse = paths.node_reverse[:, 1:]
        transition_valid = paths.node_valid[:, :-1] & paths.node_valid[:, 1:]
        route_valid = graph.relation.valid
        matches = (
            (left_node[:, :, None] == graph.relation.source_indices[None, None, :])
            & (right_node[:, :, None] == graph.relation.target_indices[None, None, :])
            & (left_reverse[:, :, None] == graph.source_reverse[None, None, :])
            & (right_reverse[:, :, None] == graph.target_reverse[None, None, :])
            & route_valid[None, None, :]
        )
        edge_exists = jnp.any(matches, axis=2)
        transition_overlap = jnp.max(
            jnp.where(matches, graph.overlap_lengths[None, None, :], 0), axis=2
        )
        transitions_ok = jnp.all((~transition_valid) | edge_exists, axis=1)
        overlaps = jnp.concatenate(
            (jnp.zeros((path_capacity, 1), dtype=jnp.int32), transition_overlap),
            axis=1,
        )
    else:
        transitions_ok = jnp.ones((path_capacity,), dtype=bool)
        overlaps = jnp.zeros((path_capacity, 1), dtype=jnp.int32)

    active = paths.path_valid
    path_ok = (
        active
        & prefix_valid
        & nonempty
        & node_membership
        & (~repeated)
        & transitions_ok
        & certificate.valid
        & (certificate.graph_id == graph.graph_id)
    )
    return path_ok, repeated, transitions_ok, safe_nodes, overlaps, node_count


def _write_decoded_paths(
    reads: SequenceBatch,
    paths: SuppliedAssemblyPaths,
    safe_nodes: Array,
    overlaps: Array,
    path_lengths: Array,
    path_ok: Array,
    output_capacity: int,
    /,
) -> SequenceBatch:
    regular = reads.token_codes[safe_nodes]
    reversed_tokens = reverse_complement(reads).token_codes[safe_nodes]
    tokens = jnp.where(paths.node_reverse[:, :, None], reversed_tokens, regular)
    read_lengths = jnp.sum(reads.valid_mask, axis=1, dtype=jnp.int32)[safe_nodes]
    contributions = jnp.where(
        paths.node_valid,
        jnp.maximum(read_lengths - overlaps, 0),
        0,
    )
    starts = jnp.cumsum(contributions, axis=1) - contributions
    read_width = reads.token_codes.shape[1]
    pad_code = reads.alphabet.code(reads.alphabet.pad_symbol)

    def write_one(
        path_tokens: Array,
        valid_nodes: Array,
        node_lengths: Array,
        node_overlaps: Array,
        node_starts: Array,
        accepted: Array,
    ) -> Array:
        output = jnp.full((output_capacity,), pad_code, dtype=reads.token_codes.dtype)

        def body(flat_index: int, current: Array) -> Array:
            node_slot = flat_index // read_width
            position = flat_index % read_width
            overlap = node_overlaps[node_slot]
            write = (
                accepted
                & valid_nodes[node_slot]
                & (position >= overlap)
                & (position < node_lengths[node_slot])
            )
            destination = node_starts[node_slot] + position - overlap
            safe_destination = jnp.clip(destination, 0, output_capacity - 1)
            previous = current[safe_destination]
            value = jnp.where(write, path_tokens[node_slot, position], previous)
            return current.at[safe_destination].set(value)

        return jax.lax.fori_loop(
            0, paths.node_indices.shape[1] * read_width, body, output
        )

    output_tokens = jax.vmap(write_one)(
        tokens,
        paths.node_valid,
        read_lengths,
        overlaps,
        starts,
        path_ok,
    )
    output_mask = (
        jnp.arange(output_capacity, dtype=jnp.int32)[None, :] < path_lengths[:, None]
    ) & path_ok[:, None]
    return SequenceBatch(
        paths.path_ids,
        output_tokens,
        output_mask,
        path_ok,
        jnp.zeros_like(output_mask),
        reads.alphabet,
    )


def _decode(
    graph: OverlapGraph,
    certificate: DAGCertificate,
    reads: SequenceBatch,
    paths: SuppliedAssemblyPaths,
    output_capacity: int,
    /,
    *,
    require_unitigs: bool,
) -> AssemblyDecodeResult:
    capacity = int(output_capacity)
    if capacity < 1:
        raise ValueError("output_capacity must be positive.")
    if reads.token_codes.shape[0] != graph.relation.source_size:
        raise ValueError("Read and graph node capacities must match.")

    path_ok, repeated, transitions_ok, safe_nodes, overlaps, node_count = _path_checks(
        graph, certificate, paths
    )
    read_lengths = jnp.sum(reads.valid_mask, axis=1, dtype=jnp.int32)[safe_nodes]
    contributions = jnp.where(
        paths.node_valid,
        jnp.maximum(read_lengths - overlaps, 0),
        0,
    )
    path_lengths = jnp.sum(contributions, axis=1, dtype=jnp.int32)
    within_capacity = path_lengths <= capacity
    path_ok = path_ok & within_capacity

    unitig_failure = jnp.asarray(False)
    if require_unitigs:
        relation = graph.relation
        safe_source = jnp.where(relation.valid, relation.source_indices, 0)
        safe_target = jnp.where(relation.valid, relation.target_indices, 0)
        out_degree = jax.ops.segment_sum(
            relation.valid.astype(jnp.int32), safe_source, relation.source_size
        )
        in_degree = jax.ops.segment_sum(
            relation.valid.astype(jnp.int32), safe_target, relation.target_size
        )
        width = paths.node_indices.shape[1]
        slots = jnp.arange(width, dtype=jnp.int32)[None, :]
        internal = paths.node_valid & (slots > 0) & (slots < (node_count[:, None] - 1))
        internal_ok = jnp.all(
            (~internal) | ((in_degree[safe_nodes] == 1) & (out_degree[safe_nodes] == 1)),
            axis=1,
        )
        first = safe_nodes[:, 0]
        last_slot = jnp.maximum(node_count - 1, 0)
        last = jnp.take_along_axis(safe_nodes, last_slot[:, None], axis=1)[:, 0]
        first_boundary = (in_degree[first] != 1) | (out_degree[first] != 1)
        last_boundary = (in_degree[last] != 1) | (out_degree[last] != 1)
        occurrence = jnp.sum(
            jax.nn.one_hot(safe_nodes, relation.source_size, dtype=jnp.int32)
            * paths.node_valid[:, :, None].astype(jnp.int32)
            * paths.path_valid[:, None, None].astype(jnp.int32),
            axis=(0, 1),
            dtype=jnp.int32,
        )
        partition_ok = jnp.all(occurrence == graph.node_valid.astype(jnp.int32))
        unitig_path_ok = internal_ok & first_boundary & last_boundary
        unitig_failure = (~partition_ok) | jnp.any(paths.path_valid & (~unitig_path_ok))
        path_ok = path_ok & unitig_path_ok & (~unitig_failure)

    sequences = _write_decoded_paths(
        reads,
        paths,
        safe_nodes,
        overlaps,
        path_lengths,
        path_ok,
        capacity,
    )
    active = paths.path_valid
    any_repeat = jnp.any(active & repeated)
    any_capacity = jnp.any(active & (~within_capacity))
    any_invalid = jnp.any(active & (~transitions_ok)) | jnp.any(active & (~path_ok))
    overall_valid = jnp.all((~active) | path_ok)
    status = jnp.where(
        unitig_failure,
        int(AssemblyStatus.NON_UNITIG_PATH),
        jnp.where(
            any_repeat,
            int(AssemblyStatus.REPEATED_NODE),
            jnp.where(
                any_capacity,
                int(AssemblyStatus.CAPACITY_EXCEEDED),
                jnp.where(
                    any_invalid,
                    int(AssemblyStatus.INVALID_PATH),
                    jnp.where(
                        jnp.any(active),
                        int(AssemblyStatus.SUCCESS),
                        int(AssemblyStatus.EMPTY_INPUT),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    contract = _decode_contract(unitig=require_unitigs)
    evidence = jnp.asarray(
        (
            jnp.sum(active, dtype=jnp.int32),
            jnp.sum(path_ok, dtype=jnp.int32),
            jnp.sum(active & repeated, dtype=jnp.int32),
            jnp.sum(active & (~transitions_ok), dtype=jnp.int32),
            jnp.sum(active & (~within_capacity), dtype=jnp.int32),
            jnp.asarray(unitig_failure, dtype=jnp.int32),
        ),
        dtype=jnp.int32,
    )
    return AssemblyDecodeResult(
        sequences,
        path_lengths,
        overall_valid,
        status,
        evidence,
        contract,
    )


def decode_supplied_paths(
    graph: OverlapGraph,
    certificate: DAGCertificate,
    reads: SequenceBatch,
    paths: SuppliedAssemblyPaths,
    /,
    *,
    output_capacity: int,
) -> AssemblyDecodeResult:
    """Decode complete supplied DAG paths, rejecting repeats and capacity overflow."""
    return _decode(
        graph,
        certificate,
        reads,
        paths,
        output_capacity,
        require_unitigs=False,
    )


def decode_supplied_unitigs(
    graph: OverlapGraph,
    certificate: DAGCertificate,
    reads: SequenceBatch,
    unitigs: SuppliedAssemblyPaths,
    /,
    *,
    output_capacity: int,
) -> AssemblyDecodeResult:
    """Decode a supplied complete maximal non-branching partition of a DAG."""
    return _decode(
        graph,
        certificate,
        reads,
        unitigs,
        output_capacity,
        require_unitigs=True,
    )


__all__ = [
    "AssemblyDecodeResult",
    "AssemblyStatus",
    "DAGCertificate",
    "OverlapGraph",
    "OverlapScoringResult",
    "SuppliedAssemblyPaths",
    "SuppliedOverlapCandidates",
    "certify_supplied_dag",
    "decode_supplied_paths",
    "decode_supplied_unitigs",
    "score_supplied_overlaps",
]
