#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Total and allele-specific copy-number states and finite-state segmentation."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...pgm import (
    initialize_belief_propagation,
    MaxProductBeliefPropagation,
    MaxProductBeliefPropagationResult,
    potts_factor_graph,
    prepare_belief_propagation,
    run_belief_propagation,
    SumProductBeliefPropagation,
    SumProductBeliefPropagationResult,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class CopyNumberStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    CAPACITY_EXCEEDED = 2
    INFERENCE_FAILED = 3


class CopyNumberState(StrictModule):
    """Total and minor-allele copy number for one or more intervals."""

    total_copy: Array
    minor_copy: Array

    def __init__(self, total_copy: ArrayLike, minor_copy: ArrayLike, /):
        total = jnp.asarray(total_copy, dtype=jnp.int32)
        minor = jnp.asarray(minor_copy, dtype=jnp.int32)
        if total.shape != minor.shape:
            raise ValueError("total_copy and minor_copy must have identical shapes.")
        self.total_copy = total
        self.minor_copy = minor

    @property
    def major_copy(self) -> Array:
        return self.total_copy - self.minor_copy

    @property
    def loss_of_heterozygosity(self) -> Array:
        return (self.total_copy > 0) & (self.minor_copy == 0)

    @property
    def valid(self) -> Array:
        return (
            (self.total_copy >= 0)
            & (self.minor_copy >= 0)
            & (2 * self.minor_copy <= self.total_copy)
        )


class CopyNumberReferencePlan(StrictModule):
    """Baseline ploidy by contig with explicit pseudoautosomal overrides.

    ``contig_baseline_copy`` represents the biological sample, so haploid X/Y
    baselines are encoded directly rather than inferred from contig names.
    Pseudoautosomal intervals may override that contig baseline.
    """

    baseline_ploidy: Array
    contig_baseline_copy: Array
    par_contig_index: Array
    par_start: Array
    par_end: Array
    par_baseline_copy: Array

    def __init__(
        self,
        baseline_ploidy: float,
        contig_baseline_copy: ArrayLike,
        /,
        *,
        par_contig_index: ArrayLike | tuple[()] = (),
        par_start: ArrayLike | tuple[()] = (),
        par_end: ArrayLike | tuple[()] = (),
        par_baseline_copy: ArrayLike | tuple[()] = (),
    ):
        ploidy = float(baseline_ploidy)
        baseline = jnp.asarray(contig_baseline_copy, dtype=jnp.float32)
        par_contig = jnp.asarray(par_contig_index, dtype=jnp.int32)
        starts = jnp.asarray(par_start, dtype=jnp.int64)
        ends = jnp.asarray(par_end, dtype=jnp.int64)
        par_copy = jnp.asarray(par_baseline_copy, dtype=jnp.float32)
        if not isfinite(ploidy) or ploidy <= 0.0:
            raise ValueError("baseline_ploidy must be finite and positive.")
        if baseline.ndim != 1 or baseline.shape[0] < 1:
            raise ValueError("contig_baseline_copy must be a non-empty vector.")
        if any(value.ndim != 1 for value in (par_contig, starts, ends, par_copy)):
            raise ValueError("Pseudoautosomal fields must be vectors.")
        if not (par_contig.shape == starts.shape == ends.shape == par_copy.shape):
            raise ValueError("Pseudoautosomal fields must have identical shapes.")
        if bool(jnp.any(~jnp.isfinite(baseline))) or bool(jnp.any(baseline <= 0.0)):
            raise ValueError("Contig baseline copies must be finite and positive.")
        if bool(jnp.any(~jnp.isfinite(par_copy))) or bool(jnp.any(par_copy <= 0.0)):
            raise ValueError("PAR baseline copies must be finite and positive.")
        if par_contig.size and bool(
            jnp.any((par_contig < 0) | (par_contig >= baseline.shape[0]))
        ):
            raise ValueError("PAR contig index is outside contig_baseline_copy.")
        if starts.size and bool(jnp.any((starts < 0) | (ends <= starts))):
            raise ValueError("PAR intervals must be non-empty and non-negative.")
        self.baseline_ploidy = jnp.asarray(ploidy, dtype=jnp.float32)
        self.contig_baseline_copy = baseline
        self.par_contig_index = par_contig
        self.par_start = starts
        self.par_end = ends
        self.par_baseline_copy = par_copy

    def expected_copy(self, contig_index: ArrayLike, position: ArrayLike, /) -> Array:
        contig = jnp.asarray(contig_index, dtype=jnp.int32)
        coordinate = jnp.asarray(position, dtype=jnp.int64)
        if contig.shape != coordinate.shape:
            raise ValueError("contig_index and position must have identical shapes.")
        if bool(jnp.any((contig < 0) | (contig >= self.contig_baseline_copy.shape[0]))):
            raise ValueError("contig_index is outside contig_baseline_copy.")
        baseline = self.contig_baseline_copy[contig]
        if self.par_contig_index.shape[0] == 0:
            return baseline
        membership = (
            (contig[..., None] == self.par_contig_index)
            & (coordinate[..., None] >= self.par_start)
            & (coordinate[..., None] < self.par_end)
        )
        par_value = jnp.sum(jnp.where(membership, self.par_baseline_copy, 0.0), axis=-1)
        return jnp.where(jnp.any(membership, axis=-1), par_value, baseline)


class CopyNumberObservations(StrictModule):
    """Depth and B-allele-frequency summaries on genomic intervals."""

    contig_index: Array
    start: Array
    end: Array
    normalized_depth: Array
    depth_uncertainty: Array
    baf: Array
    baf_uncertainty: Array
    baf_valid: Array

    def __init__(
        self,
        contig_index: ArrayLike,
        start: ArrayLike,
        end: ArrayLike,
        normalized_depth: ArrayLike,
        depth_uncertainty: ArrayLike,
        baf: ArrayLike,
        baf_uncertainty: ArrayLike,
        baf_valid: ArrayLike,
        /,
    ):
        contig = jnp.asarray(contig_index, dtype=jnp.int32)
        starts = jnp.asarray(start, dtype=jnp.int64)
        ends = jnp.asarray(end, dtype=jnp.int64)
        depth = jnp.asarray(normalized_depth, dtype=jnp.float32)
        depth_sigma = jnp.asarray(depth_uncertainty, dtype=jnp.float32)
        baf_value = jnp.asarray(baf, dtype=jnp.float32)
        baf_sigma = jnp.asarray(baf_uncertainty, dtype=jnp.float32)
        baf_mask = jnp.asarray(baf_valid, dtype=bool)
        if contig.ndim != 1 or contig.shape[0] < 1:
            raise ValueError("Copy-number observations must be non-empty vectors.")
        if any(
            value.shape != contig.shape
            for value in (
                starts,
                ends,
                depth,
                depth_sigma,
                baf_value,
                baf_sigma,
                baf_mask,
            )
        ):
            raise ValueError("Copy-number observation fields must have identical shapes.")
        self.contig_index = contig
        self.start = starts
        self.end = ends
        self.normalized_depth = depth
        self.depth_uncertainty = depth_sigma
        self.baf = baf_value
        self.baf_uncertainty = baf_sigma
        self.baf_valid = baf_mask


class CopyNumberSegmentationPlan(StrictModule):
    """Finite allele-specific state space and chain transition model."""

    state_total_copy: Array
    state_minor_copy: Array
    maximum_bins: int = eqx.field(static=True)
    maximum_total_copy: int = eqx.field(static=True)
    maximum_states: int = eqx.field(static=True)
    transition_penalty: Array
    minimum_depth_uncertainty: Array
    minimum_baf_uncertainty: Array

    def __init__(
        self,
        *,
        maximum_bins: int,
        maximum_total_copy: int = 8,
        maximum_states: int = 64,
        transition_penalty: float = 4.0,
        minimum_depth_uncertainty: float = 0.03,
        minimum_baf_uncertainty: float = 0.01,
    ):
        bins = int(maximum_bins)
        max_total = int(maximum_total_copy)
        max_states = int(maximum_states)
        penalty = float(transition_penalty)
        depth_floor = float(minimum_depth_uncertainty)
        baf_floor = float(minimum_baf_uncertainty)
        if bins < 1 or max_total < 0 or max_states < 1:
            raise ValueError(
                "Copy-number capacities must be positive (maximum_total_copy may be zero)."
            )
        if not isfinite(penalty) or penalty < 0.0:
            raise ValueError("transition_penalty must be finite and non-negative.")
        if (
            not isfinite(depth_floor)
            or depth_floor <= 0.0
            or not isfinite(baf_floor)
            or baf_floor <= 0.0
        ):
            raise ValueError("Uncertainty floors must be finite and positive.")
        totals: list[int] = []
        minors: list[int] = []
        for total in range(max_total + 1):
            for minor in range(total // 2 + 1):
                totals.append(total)
                minors.append(minor)
        if len(totals) > max_states:
            raise ValueError(
                f"Allele-specific state space requires {len(totals)} states, exceeding maximum_states={max_states}."
            )
        self.state_total_copy = jnp.asarray(totals, dtype=jnp.int32)
        self.state_minor_copy = jnp.asarray(minors, dtype=jnp.int32)
        self.maximum_bins = bins
        self.maximum_total_copy = max_total
        self.maximum_states = max_states
        self.transition_penalty = jnp.asarray(penalty, dtype=jnp.float32)
        self.minimum_depth_uncertainty = jnp.asarray(depth_floor, dtype=jnp.float32)
        self.minimum_baf_uncertainty = jnp.asarray(baf_floor, dtype=jnp.float32)


class CopyNumberCandidateEvidence(StrictModule):
    """Explicit finite-state coverage; states above the bound are not generated."""

    candidate_state_count: Array
    maximum_total_copy: Array
    exhaustive_within_bound: Array
    unmodeled_above_bound: Array
    capacity_sufficient: Array
    interval_generation_performed: Array
    intervals_precomputed: Array
    interval_search_exhaustive: Array


class CopyNumberSegmentationEvidence(StrictModule):
    expected_baseline_copy: Array
    expected_depth_by_state: Array
    expected_baf_by_state: Array
    depth_log_likelihood: Array
    baf_log_likelihood: Array
    posterior_entropy: Array
    posterior_exact: Array
    map_optimal: Array
    finite_state_inference: Array
    candidates: CopyNumberCandidateEvidence


class CopyNumberSegmentationResult(StrictModule):
    state: CopyNumberState
    state_index: Array
    segment_index: Array
    posterior_probability: Array
    valid: Array
    status: Array
    evidence: CopyNumberSegmentationEvidence
    method_contract: BioinformaticsMethodContract


def _copy_number_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "allele-specific-copy-number-chain-inference",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.PARTITION,
        conditioning_statement=(
            "Exact finite-state chain inference conditional on the supplied "
            "depth/BAF Gaussian emission model and bounded copy-number state space."
        ),
        truncation_statement=(
            "Total-copy states above maximum_total_copy are unmodeled and reported "
            "explicitly; genomic candidates are supplied intervals."
        ),
        capacity_semantics=(
            "Bin and state capacities are preflighted; excess returns "
            "CAPACITY_EXCEEDED without truncation."
        ),
        assumptions=(
            "adjacent same-contig intervals form a Markov chain",
            "depth is baseline-normalized",
            "BAF is folded to the minor-allele fraction",
        ),
        nondifferentiable_outputs=("state", "segment_index", "status"),
    )


def _invalid_segmentation(
    observations: CopyNumberObservations,
    plan: CopyNumberSegmentationPlan,
    baseline: Array,
    status: CopyNumberStatus,
    /,
) -> CopyNumberSegmentationResult:
    count = observations.contig_index.shape[0]
    states = plan.state_total_copy.shape[0]
    candidate_evidence = CopyNumberCandidateEvidence(
        candidate_state_count=jnp.asarray(states, dtype=jnp.int32),
        maximum_total_copy=jnp.asarray(plan.maximum_total_copy, dtype=jnp.int32),
        exhaustive_within_bound=jnp.asarray(True),
        unmodeled_above_bound=jnp.asarray(True),
        capacity_sufficient=jnp.asarray(count <= plan.maximum_bins),
        interval_generation_performed=jnp.asarray(False),
        intervals_precomputed=jnp.asarray(True),
        interval_search_exhaustive=jnp.asarray(False),
    )
    evidence = CopyNumberSegmentationEvidence(
        expected_baseline_copy=baseline,
        expected_depth_by_state=jnp.zeros((count, states), dtype=jnp.float32),
        expected_baf_by_state=jnp.zeros((states,), dtype=jnp.float32),
        depth_log_likelihood=jnp.zeros((count, states), dtype=jnp.float32),
        baf_log_likelihood=jnp.zeros((count, states), dtype=jnp.float32),
        posterior_entropy=jnp.zeros((count,), dtype=jnp.float32),
        posterior_exact=jnp.asarray(False),
        map_optimal=jnp.asarray(False),
        finite_state_inference=jnp.asarray(True),
        candidates=candidate_evidence,
    )
    return CopyNumberSegmentationResult(
        state=CopyNumberState(
            jnp.zeros((count,), dtype=jnp.int32), jnp.zeros((count,), dtype=jnp.int32)
        ),
        state_index=jnp.zeros((count,), dtype=jnp.int32),
        segment_index=jnp.zeros((count,), dtype=jnp.int32),
        posterior_probability=jnp.zeros((count, states), dtype=jnp.float32),
        valid=jnp.asarray(False),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=evidence,
        method_contract=_copy_number_contract(),
    )


def segment_copy_number(
    observations: CopyNumberObservations,
    reference: CopyNumberReferencePlan,
    plan: CopyNumberSegmentationPlan,
    /,
) -> CopyNumberSegmentationResult:
    """Segment total/minor copy number with exact native finite-state chain BP."""
    if not isinstance(observations, CopyNumberObservations):
        raise TypeError("observations must be CopyNumberObservations.")
    if not isinstance(reference, CopyNumberReferencePlan):
        raise TypeError("reference must be CopyNumberReferencePlan.")
    if not isinstance(plan, CopyNumberSegmentationPlan):
        raise TypeError("plan must be CopyNumberSegmentationPlan.")
    count = observations.contig_index.shape[0]
    midpoint = observations.start + (observations.end - observations.start) // 2
    baseline = reference.expected_copy(observations.contig_index, midpoint)
    if count > plan.maximum_bins:
        return _invalid_segmentation(
            observations, plan, baseline, CopyNumberStatus.CAPACITY_EXCEEDED
        )
    finite = (
        jnp.all(jnp.isfinite(observations.normalized_depth))
        & jnp.all(jnp.isfinite(observations.depth_uncertainty))
        & jnp.all(
            (~observations.baf_valid)
            | (
                jnp.isfinite(observations.baf)
                & jnp.isfinite(observations.baf_uncertainty)
            )
        )
    )
    domain = (
        jnp.all(observations.start >= 0)
        & jnp.all(observations.end > observations.start)
        & jnp.all(observations.normalized_depth >= 0.0)
        & jnp.all(observations.depth_uncertainty > 0.0)
        & jnp.all(
            (~observations.baf_valid)
            | (
                (observations.baf >= 0.0)
                & (observations.baf <= 1.0)
                & (observations.baf_uncertainty > 0.0)
            )
        )
    )
    if not bool(finite & domain):
        return _invalid_segmentation(
            observations, plan, baseline, CopyNumberStatus.INVALID_INPUT
        )

    total = plan.state_total_copy.astype(jnp.float32)
    minor = plan.state_minor_copy.astype(jnp.float32)
    expected_depth = total[None, :] / baseline[:, None]
    depth_sigma = jnp.maximum(
        observations.depth_uncertainty, plan.minimum_depth_uncertainty
    )
    depth_residual = (
        observations.normalized_depth[:, None] - expected_depth
    ) / depth_sigma[:, None]
    depth_log_likelihood = -0.5 * depth_residual * depth_residual - jnp.log(
        depth_sigma[:, None]
    )
    expected_baf = jnp.where(total > 0.0, minor / jnp.maximum(total, 1.0), 0.5)
    folded_baf = jnp.minimum(observations.baf, 1.0 - observations.baf)
    baf_sigma = jnp.maximum(observations.baf_uncertainty, plan.minimum_baf_uncertainty)
    baf_residual = (folded_baf[:, None] - expected_baf[None, :]) / baf_sigma[:, None]
    baf_log_likelihood = jnp.where(
        observations.baf_valid[:, None],
        -0.5 * baf_residual * baf_residual - jnp.log(baf_sigma[:, None]),
        0.0,
    )
    unary = depth_log_likelihood + baf_log_likelihood
    same_contig = observations.contig_index[1:] == observations.contig_index[:-1]
    all_edges = jnp.stack(
        (
            jnp.arange(max(count - 1, 0), dtype=jnp.int32),
            jnp.arange(1, count, dtype=jnp.int32),
        ),
        axis=-1,
    )
    edges = all_edges[same_contig]
    state_count = total.shape[0]
    transition = -plan.transition_penalty * (
        1.0 - jnp.eye(state_count, dtype=unary.dtype)
    )
    pairwise = jnp.broadcast_to(transition, (edges.shape[0], state_count, state_count))
    graph = potts_factor_graph(unary, edges, pairwise, name="copy_number_state")

    sum_prepared = prepare_belief_propagation(
        graph,
        SumProductBeliefPropagation(maximum_steps=max(1, 2 * count + 1)),
        max_factor_configurations=plan.maximum_states * plan.maximum_states,
    )
    sum_result = run_belief_propagation(
        sum_prepared,
        initialize_belief_propagation(sum_prepared),
    )
    if not isinstance(sum_result, SumProductBeliefPropagationResult):
        raise RuntimeError("Sum-product inference returned the wrong result type.")
    max_prepared = prepare_belief_propagation(
        graph,
        MaxProductBeliefPropagation(maximum_steps=max(1, 2 * count + 1)),
        max_factor_configurations=plan.maximum_states * plan.maximum_states,
    )
    max_result = run_belief_propagation(
        max_prepared,
        initialize_belief_propagation(max_prepared),
    )
    if not isinstance(max_result, MaxProductBeliefPropagationResult):
        raise RuntimeError("Max-product inference returned the wrong result type.")
    posterior = jnp.exp(
        sum_result.variable_log_probabilities.values.reshape((count, state_count))
    )
    state_index = max_result.map_assignment
    state = CopyNumberState(
        plan.state_total_copy[state_index], plan.state_minor_copy[state_index]
    )
    changed = jnp.ones((count,), dtype=bool)
    if count > 1:
        changed = changed.at[1:].set(
            (state_index[1:] != state_index[:-1])
            | (observations.contig_index[1:] != observations.contig_index[:-1])
        )
    segment_index = jnp.cumsum(changed.astype(jnp.int32)) - 1
    entropy = -jnp.sum(
        jnp.where(posterior > 0.0, posterior * jnp.log(posterior), 0.0), axis=-1
    )
    inference_valid = (
        sum_result.valid
        & max_result.valid
        & sum_result.marginals_exact
        & max_result.map_available
    )
    candidate_evidence = CopyNumberCandidateEvidence(
        candidate_state_count=jnp.asarray(state_count, dtype=jnp.int32),
        maximum_total_copy=jnp.asarray(plan.maximum_total_copy, dtype=jnp.int32),
        exhaustive_within_bound=jnp.asarray(True),
        unmodeled_above_bound=jnp.asarray(True),
        capacity_sufficient=jnp.asarray(True),
        interval_generation_performed=jnp.asarray(False),
        intervals_precomputed=jnp.asarray(True),
        interval_search_exhaustive=jnp.asarray(False),
    )
    evidence = CopyNumberSegmentationEvidence(
        expected_baseline_copy=baseline,
        expected_depth_by_state=expected_depth,
        expected_baf_by_state=expected_baf,
        depth_log_likelihood=depth_log_likelihood,
        baf_log_likelihood=baf_log_likelihood,
        posterior_entropy=entropy,
        posterior_exact=jnp.asarray(sum_result.marginals_exact),
        map_optimal=max_result.optimal,
        finite_state_inference=jnp.asarray(True),
        candidates=candidate_evidence,
    )
    return CopyNumberSegmentationResult(
        state=state,
        state_index=state_index,
        segment_index=segment_index,
        posterior_probability=posterior,
        valid=inference_valid,
        status=jnp.where(
            inference_valid,
            int(CopyNumberStatus.SUCCESS),
            int(CopyNumberStatus.INFERENCE_FAILED),
        ).astype(jnp.int32),
        evidence=evidence,
        method_contract=_copy_number_contract(),
    )


__all__ = [
    "CopyNumberCandidateEvidence",
    "CopyNumberObservations",
    "CopyNumberReferencePlan",
    "CopyNumberSegmentationEvidence",
    "CopyNumberSegmentationPlan",
    "CopyNumberSegmentationResult",
    "CopyNumberState",
    "CopyNumberStatus",
    "segment_copy_number",
]
