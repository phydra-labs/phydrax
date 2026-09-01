#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Structural-variant breakends, event topology, and breakpoint evidence."""

from __future__ import annotations

from enum import IntEnum, IntFlag
from math import isfinite

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import SequenceBatch


class BreakendOrientation(IntEnum):
    """Reference-side orientation retained at a breakend."""

    LEFT = -1
    RIGHT = 1


class EventLinkKind(IntEnum):
    """Meaning of an explicit non-mate event edge."""

    SAME_EVENT = 0
    ORDERED_CHAIN = 1
    ALTERNATIVE_RESOLUTION = 2


class StructuralVariantStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_EVIDENCE = 1
    INVALID_BREAKEND = 2
    INVALID_MATE = 3
    INVALID_EVENT_LINK = 4
    CAPACITY_EXCEEDED = 5
    CANDIDATE_SET_INCOMPLETE = 6


class CandidateLimitation(IntFlag):
    """Observable limitations of the candidate-generation search."""

    NONE = 0
    REGIONS_NOT_EXHAUSTIVE = 1
    INTERCONTIG_DISABLED = 2
    UNPAIRED_DROPPED = 4
    ASSEMBLY_DISABLED = 8
    CAPACITY_EXCEEDED = 16
    PRECOMPUTED_CANDIDATES = 32


class StructuralVariantCandidatePlan(StrictModule):
    """Static candidate-search envelope; it never implies exhaustive discovery."""

    maximum_breakends: int = eqx.field(static=True)
    intercontig_search: bool = eqx.field(static=True)
    retain_unpaired: bool = eqx.field(static=True)
    assembly_search: bool = eqx.field(static=True)
    regions_exhaustive: bool = eqx.field(static=True)
    precomputed_candidates: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_breakends: int,
        intercontig_search: bool = True,
        retain_unpaired: bool = True,
        assembly_search: bool = True,
        regions_exhaustive: bool = False,
        precomputed_candidates: bool = False,
    ):
        capacity = int(maximum_breakends)
        if capacity < 1:
            raise ValueError("maximum_breakends must be positive.")
        self.maximum_breakends = capacity
        self.intercontig_search = bool(intercontig_search)
        self.retain_unpaired = bool(retain_unpaired)
        self.assembly_search = bool(assembly_search)
        self.regions_exhaustive = bool(regions_exhaustive)
        self.precomputed_candidates = bool(precomputed_candidates)


class CandidateGenerationEvidence(StrictModule):
    """Candidate-search coverage and all known omissions."""

    generated_breakends: Array
    requested_breakends: Array
    dropped_breakends: Array
    limitation_mask: Array
    exhaustive: Array
    capacity_sufficient: Array


class BreakendGraph(StrictModule):
    """A breakend/event graph, distinct from a collection of small variants.

    Coordinates are zero-based interbase positions. ``mate_index == -1`` encodes
    an explicitly unpaired breakend. Imprecision is the inclusive interval
    ``[interval_start, interval_end]`` around each position. Event links are
    additional graph edges and do not replace mate relationships.
    """

    contig_index: Array
    position: Array
    orientation: Array
    mate_index: Array
    event_index: Array
    interval_start: Array
    interval_end: Array
    inserted_sequence: SequenceBatch
    event_links: Array
    event_link_kind: Array

    def __init__(
        self,
        contig_index: ArrayLike,
        position: ArrayLike,
        orientation: ArrayLike,
        mate_index: ArrayLike,
        event_index: ArrayLike,
        interval_start: ArrayLike,
        interval_end: ArrayLike,
        inserted_sequence: SequenceBatch,
        event_links: ArrayLike,
        event_link_kind: ArrayLike,
        /,
    ):
        contig = jnp.asarray(contig_index, dtype=jnp.int32)
        positions = jnp.asarray(position, dtype=jnp.int64)
        orientations = jnp.asarray(orientation, dtype=jnp.int8)
        mates = jnp.asarray(mate_index, dtype=jnp.int32)
        events = jnp.asarray(event_index, dtype=jnp.int32)
        starts = jnp.asarray(interval_start, dtype=jnp.int64)
        ends = jnp.asarray(interval_end, dtype=jnp.int64)
        links = jnp.asarray(event_links, dtype=jnp.int32)
        kinds = jnp.asarray(event_link_kind, dtype=jnp.int8)
        if contig.ndim != 1 or contig.shape[0] < 1:
            raise ValueError("Breakend fields must be non-empty vectors.")
        shape = contig.shape
        if any(
            value.shape != shape
            for value in (positions, orientations, mates, events, starts, ends)
        ):
            raise ValueError("Breakend fields must have identical shapes.")
        if not isinstance(inserted_sequence, SequenceBatch):
            raise TypeError("inserted_sequence must be a SequenceBatch.")
        if inserted_sequence.record_ids.shape != shape:
            raise ValueError("inserted_sequence must contain one record per breakend.")
        if not bool(jnp.all(inserted_sequence.case_mask)):
            raise ValueError(
                "Each breakend must have an active inserted-sequence record; "
                "use an active zero-length record when no insertion is present."
            )
        if links.ndim != 2 or links.shape[1:] != (2,):
            raise ValueError("event_links must have shape (link, 2).")
        if kinds.shape != links.shape[:1]:
            raise ValueError("event_link_kind must contain one kind per event link.")
        self.contig_index = contig
        self.position = positions
        self.orientation = orientations
        self.mate_index = mates
        self.event_index = events
        self.interval_start = starts
        self.interval_end = ends
        self.inserted_sequence = inserted_sequence
        self.event_links = links
        self.event_link_kind = kinds

    @property
    def reciprocal(self) -> Array:
        count = self.mate_index.shape[0]
        safe = jnp.clip(self.mate_index, 0, max(count - 1, 0))
        paired = self.mate_index >= 0
        return paired & (self.mate_index[safe] == jnp.arange(count, dtype=jnp.int32))

    @property
    def unpaired(self) -> Array:
        return self.mate_index < 0

    @property
    def imprecise(self) -> Array:
        return (self.interval_start != self.position) | (
            self.interval_end != self.position
        )


class BreakpointEvidence(StrictModule):
    """Independent support channels aligned one-to-one with breakends."""

    split_reads: Array
    spanning_pairs: Array
    assembled_reads: Array
    depth_log_ratio: Array
    mapping_quality: Array
    strand_balance: Array
    local_uncertainty: Array

    def __init__(
        self,
        split_reads: ArrayLike,
        spanning_pairs: ArrayLike,
        assembled_reads: ArrayLike,
        depth_log_ratio: ArrayLike,
        mapping_quality: ArrayLike,
        strand_balance: ArrayLike,
        local_uncertainty: ArrayLike,
        /,
    ):
        split = jnp.asarray(split_reads, dtype=jnp.float32)
        pair = jnp.asarray(spanning_pairs, dtype=jnp.float32)
        assembly = jnp.asarray(assembled_reads, dtype=jnp.float32)
        depth = jnp.asarray(depth_log_ratio, dtype=jnp.float32)
        quality = jnp.asarray(mapping_quality, dtype=jnp.float32)
        balance = jnp.asarray(strand_balance, dtype=jnp.float32)
        uncertainty = jnp.asarray(local_uncertainty, dtype=jnp.float32)
        if split.ndim != 1:
            raise ValueError("Breakpoint evidence must be one-dimensional.")
        if any(
            value.shape != split.shape
            for value in (pair, assembly, depth, quality, balance, uncertainty)
        ):
            raise ValueError("Breakpoint evidence channels must have identical shapes.")
        self.split_reads = split
        self.spanning_pairs = pair
        self.assembled_reads = assembly
        self.depth_log_ratio = depth
        self.mapping_quality = quality
        self.strand_balance = balance
        self.local_uncertainty = uncertainty


class BreakpointAggregationPlan(StrictModule):
    """Calibrated direct aggregation of heterogeneous breakpoint support."""

    channel_weights: Array
    prior_log_odds: Array
    minimum_read_support: Array

    def __init__(
        self,
        channel_weights: (
            ArrayLike | tuple[float, float, float, float, float, float]
        ) = (1.0, 0.7, 1.3, 0.5, 0.02, 0.4),
        /,
        *,
        prior_log_odds: float = -4.0,
        minimum_read_support: int = 1,
    ):
        weights = jnp.asarray(channel_weights, dtype=jnp.float32)
        prior = float(prior_log_odds)
        minimum = int(minimum_read_support)
        if weights.shape != (6,) or bool(jnp.any(~jnp.isfinite(weights))):
            raise ValueError("channel_weights must contain six finite values.")
        if not isfinite(prior):
            raise ValueError("prior_log_odds must be finite.")
        if minimum < 0:
            raise ValueError("minimum_read_support must be non-negative.")
        self.channel_weights = weights
        self.prior_log_odds = jnp.asarray(prior, dtype=jnp.float32)
        self.minimum_read_support = jnp.asarray(minimum, dtype=jnp.int32)


class BreakpointAggregationEvidence(StrictModule):
    read_support: Array
    finite_channels: Array
    minimum_support_met: Array
    channel_contributions: Array


class BreakpointAggregationResult(StrictModule):
    log_odds: Array
    probability: Array
    valid: Array
    status: Array
    evidence: BreakpointAggregationEvidence
    method_contract: BioinformaticsMethodContract


class StructuralVariantEvidence(StrictModule):
    reciprocal_breakends: Array
    unpaired_breakends: Array
    imprecise_breakends: Array
    linked_breakends: Array
    candidate_generation: CandidateGenerationEvidence
    breakpoint: BreakpointAggregationEvidence


class StructuralVariantResult(StrictModule):
    graph: BreakendGraph
    breakpoint_probability: Array
    valid: Array
    status: Array
    evidence: StructuralVariantEvidence
    method_contract: BioinformaticsMethodContract


def structural_variant_candidate_evidence(
    plan: StructuralVariantCandidatePlan,
    requested_breakends: int | Array,
    /,
) -> CandidateGenerationEvidence:
    """Preflight a fixed candidate set without truncating it."""
    if not isinstance(plan, StructuralVariantCandidatePlan):
        raise TypeError("plan must be a StructuralVariantCandidatePlan.")
    requested = jnp.asarray(requested_breakends, dtype=jnp.int32).reshape(())
    if bool(requested < 0):
        raise ValueError("requested_breakends must be non-negative.")
    capacity_ok = requested <= plan.maximum_breakends
    limitation = int(CandidateLimitation.NONE)
    if not plan.regions_exhaustive:
        limitation |= int(CandidateLimitation.REGIONS_NOT_EXHAUSTIVE)
    if not plan.intercontig_search:
        limitation |= int(CandidateLimitation.INTERCONTIG_DISABLED)
    if not plan.retain_unpaired:
        limitation |= int(CandidateLimitation.UNPAIRED_DROPPED)
    if not plan.assembly_search:
        limitation |= int(CandidateLimitation.ASSEMBLY_DISABLED)
    if plan.precomputed_candidates:
        limitation |= int(CandidateLimitation.PRECOMPUTED_CANDIDATES)
    limitation_mask = jnp.asarray(limitation, dtype=jnp.int32)
    limitation_mask = jnp.where(
        capacity_ok,
        limitation_mask,
        limitation_mask | int(CandidateLimitation.CAPACITY_EXCEEDED),
    )
    generated = jnp.where(capacity_ok, requested, 0)
    return CandidateGenerationEvidence(
        generated_breakends=generated,
        requested_breakends=requested,
        dropped_breakends=jnp.where(capacity_ok, 0, requested),
        limitation_mask=limitation_mask,
        exhaustive=jnp.asarray(plan.regions_exhaustive) & capacity_ok,
        capacity_sufficient=capacity_ok,
    )


def aggregate_breakpoint_evidence(
    evidence: BreakpointEvidence,
    plan: BreakpointAggregationPlan | None = None,
    /,
) -> BreakpointAggregationResult:
    """Aggregate breakpoint channels under an explicit approximate model."""
    if not isinstance(evidence, BreakpointEvidence):
        raise TypeError("evidence must be BreakpointEvidence.")
    selected = BreakpointAggregationPlan() if plan is None else plan
    if not isinstance(selected, BreakpointAggregationPlan):
        raise TypeError("plan must be BreakpointAggregationPlan or None.")
    read_support = (
        evidence.split_reads + evidence.spanning_pairs + evidence.assembled_reads
    )
    safe_uncertainty = jnp.maximum(evidence.local_uncertainty, jnp.finfo(jnp.float32).eps)
    channels = jnp.stack(
        (
            jnp.log1p(jnp.maximum(evidence.split_reads, 0.0)),
            jnp.log1p(jnp.maximum(evidence.spanning_pairs, 0.0)),
            jnp.log1p(jnp.maximum(evidence.assembled_reads, 0.0)),
            jnp.abs(evidence.depth_log_ratio) / safe_uncertainty,
            jnp.maximum(evidence.mapping_quality, 0.0),
            1.0 - 2.0 * jnp.abs(jnp.clip(evidence.strand_balance, 0.0, 1.0) - 0.5),
        ),
        axis=-1,
    )
    contributions = channels * selected.channel_weights
    log_odds = selected.prior_log_odds + jnp.sum(contributions, axis=-1)
    finite = jnp.all(jnp.isfinite(channels), axis=-1) & jnp.isfinite(read_support)
    minimum = read_support >= selected.minimum_read_support
    valid = finite & minimum
    status = jnp.where(
        finite,
        jnp.where(
            minimum,
            int(StructuralVariantStatus.SUCCESS),
            int(StructuralVariantStatus.CANDIDATE_SET_INCOMPLETE),
        ),
        int(StructuralVariantStatus.NONFINITE_EVIDENCE),
    ).astype(jnp.int32)
    aggregation_evidence = BreakpointAggregationEvidence(
        read_support, finite, minimum, contributions
    )
    contract = BioinformaticsMethodContract(
        "breakpoint-evidence-aggregation",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Conditional on supplied breakpoint candidates and calibrated "
            "independent evidence channels."
        ),
        truncation_statement="Candidate generation is external; omitted candidates cannot receive probability.",
        capacity_semantics="One output per supplied breakend; no truncation is performed.",
        assumptions=(
            "evidence channels are calibration-compatible",
            "read support is non-negative",
        ),
    )
    return BreakpointAggregationResult(
        log_odds, jnn.sigmoid(log_odds), valid, status, aggregation_evidence, contract
    )


def _structural_variant_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "structural-variant-event-graph-evaluation",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.GRAPH,
        conditioning_statement="Conditional on supplied oriented breakends, event links, and breakpoint evidence.",
        truncation_statement=(
            "Candidate-generation limitations are reported by "
            "CandidateGenerationEvidence and never hidden."
        ),
        capacity_semantics="Capacity excess returns CAPACITY_EXCEEDED; candidates are never silently truncated.",
        assumptions=(
            "mate indices are reciprocal when paired",
            "event links refer to supplied breakends",
        ),
        nondifferentiable_outputs=(
            "status",
            "topology",
            "candidate limitation mask",
        ),
    )


def evaluate_breakend_graph(
    graph: BreakendGraph,
    breakpoint_evidence: BreakpointEvidence,
    candidate_evidence: CandidateGenerationEvidence,
    aggregation_plan: BreakpointAggregationPlan | None = None,
    /,
) -> StructuralVariantResult:
    """Validate event topology and score supplied structural-variant candidates."""
    if not isinstance(graph, BreakendGraph):
        raise TypeError("graph must be a BreakendGraph.")
    if not isinstance(candidate_evidence, CandidateGenerationEvidence):
        raise TypeError("candidate_evidence must be CandidateGenerationEvidence.")
    count = graph.position.shape[0]
    if breakpoint_evidence.split_reads.shape != (count,):
        raise ValueError("breakpoint_evidence must align with graph breakends.")
    if not bool(candidate_evidence.capacity_sufficient):
        zeros = jnp.zeros((count,), dtype=jnp.float32)
        breakpoint = BreakpointAggregationEvidence(
            read_support=zeros,
            finite_channels=jnp.zeros((count,), dtype=bool),
            minimum_support_met=jnp.zeros((count,), dtype=bool),
            channel_contributions=jnp.zeros((count, 6), dtype=jnp.float32),
        )
        result_evidence = StructuralVariantEvidence(
            reciprocal_breakends=graph.reciprocal,
            unpaired_breakends=graph.unpaired,
            imprecise_breakends=graph.imprecise,
            linked_breakends=jnp.zeros((count,), dtype=bool),
            candidate_generation=candidate_evidence,
            breakpoint=breakpoint,
        )
        return StructuralVariantResult(
            graph,
            zeros,
            jnp.asarray(False),
            jnp.asarray(int(StructuralVariantStatus.CAPACITY_EXCEEDED), dtype=jnp.int32),
            result_evidence,
            _structural_variant_contract(),
        )
    aggregate = aggregate_breakpoint_evidence(breakpoint_evidence, aggregation_plan)
    orientation_valid = (graph.orientation == -1) | (graph.orientation == 1)
    interval_valid = (
        (graph.interval_start >= 0)
        & (graph.interval_start <= graph.position)
        & (graph.position <= graph.interval_end)
    )
    coordinate_valid = (
        (graph.contig_index >= 0)
        & (graph.position >= 0)
        & (graph.event_index >= 0)
        & interval_valid
        & orientation_valid
    )
    mate_in_range = (graph.mate_index == -1) | (
        (graph.mate_index >= 0) & (graph.mate_index < count)
    )
    safe_mate = jnp.clip(graph.mate_index, 0, count - 1)
    paired = graph.mate_index >= 0
    mate_valid = (
        mate_in_range
        & (~paired | (graph.mate_index[safe_mate] == jnp.arange(count, dtype=jnp.int32)))
        & (~paired | (graph.event_index[safe_mate] == graph.event_index))
    )
    link_in_range = jnp.all(
        (graph.event_links >= 0) & (graph.event_links < count), axis=-1
    )
    kind_valid = (graph.event_link_kind >= int(EventLinkKind.SAME_EVENT)) & (
        graph.event_link_kind <= int(EventLinkKind.ALTERNATIVE_RESOLUTION)
    )
    safe_links = jnp.clip(graph.event_links, 0, count - 1)
    same_event_semantics = (graph.event_link_kind != int(EventLinkKind.SAME_EVENT)) | (
        graph.event_index[safe_links[:, 0]] == graph.event_index[safe_links[:, 1]]
    )
    event_link_valid = link_in_range & kind_valid & same_event_semantics
    graph_valid = (
        jnp.all(coordinate_valid) & jnp.all(mate_valid) & jnp.all(event_link_valid)
    )
    complete = candidate_evidence.limitation_mask == int(CandidateLimitation.NONE)
    candidate_alignment = candidate_evidence.requested_breakends == count
    valid = (
        graph_valid
        & jnp.all(aggregate.valid)
        & candidate_evidence.capacity_sufficient
        & candidate_alignment
    )
    status = jnp.where(
        ~candidate_evidence.capacity_sufficient,
        int(StructuralVariantStatus.CAPACITY_EXCEEDED),
        jnp.where(
            ~jnp.all(coordinate_valid),
            int(StructuralVariantStatus.INVALID_BREAKEND),
            jnp.where(
                ~jnp.all(mate_valid),
                int(StructuralVariantStatus.INVALID_MATE),
                jnp.where(
                    ~jnp.all(event_link_valid),
                    int(StructuralVariantStatus.INVALID_EVENT_LINK),
                    jnp.where(
                        (~complete) | (~candidate_alignment),
                        int(StructuralVariantStatus.CANDIDATE_SET_INCOMPLETE),
                        jnp.max(aggregate.status),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    linked = jnp.zeros((count,), dtype=bool)
    linked = (
        linked.at[graph.event_links.reshape((-1,))].set(True)
        if graph.event_links.shape[0]
        else linked
    )
    result_evidence = StructuralVariantEvidence(
        reciprocal_breakends=graph.reciprocal,
        unpaired_breakends=graph.unpaired,
        imprecise_breakends=graph.imprecise,
        linked_breakends=linked,
        candidate_generation=candidate_evidence,
        breakpoint=aggregate.evidence,
    )
    contract = _structural_variant_contract()
    return StructuralVariantResult(
        graph, aggregate.probability, valid, status, result_evidence, contract
    )


__all__ = [
    "BreakendGraph",
    "BreakendOrientation",
    "BreakpointAggregationEvidence",
    "BreakpointAggregationPlan",
    "BreakpointAggregationResult",
    "BreakpointEvidence",
    "CandidateGenerationEvidence",
    "CandidateLimitation",
    "EventLinkKind",
    "StructuralVariantCandidatePlan",
    "StructuralVariantEvidence",
    "StructuralVariantResult",
    "StructuralVariantStatus",
    "aggregate_breakpoint_evidence",
    "evaluate_breakend_graph",
    "structural_variant_candidate_evidence",
]
