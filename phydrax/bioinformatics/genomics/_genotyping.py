#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-state likelihood and posterior inference for germline small variants."""

from __future__ import annotations

import math
from enum import IntEnum
from itertools import combinations_with_replacement

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class GenotypingStatus(IntEnum):
    """Machine-readable state of bounded germline genotype inference."""

    OK = 0
    CAPACITY_EXCEEDED = 1
    INVALID_INPUT = 2
    NO_COVERAGE = 3
    CANDIDATE_OMITTED = 4
    INVALID_PRIOR = 5
    LOW_CONFIDENCE = 6


GENOTYPE_ENUMERATION_CONTRACT = BioinformaticsMethodContract(
    "bounded_genotype_state_enumeration",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.SET,
    conditioning_statement=(
        "Exact VCF Number=G colexicographic enumeration for the declared allele "
        "count and ploidy."
    ),
    truncation_statement="The genotype state set is never truncated.",
    capacity_semantics=(
        "The multiset coefficient is preflighted against max_genotypes; overflow "
        "returns an empty state mask and CAPACITY_EXCEEDED."
    ),
    assumptions=("Germline unordered genotype states",),
    nondifferentiable_outputs=("states", "state_mask", "status"),
    input_dtype="int32",
    output_dtype="int32",
)

GENOTYPE_LIKELIHOOD_CONTRACT = BioinformaticsMethodContract(
    "local_haplotype_genotype_likelihood",
    MethodKind.EXACT_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.PROBABILISTIC,
    conditioning_statement=(
        "Conditionally independent reads with a uniform latent chromosome copy "
        "within each unordered germline genotype."
    ),
    truncation_statement="Reads, alleles, and genotype states are never truncated.",
    capacity_semantics=(
        "Fixed input shapes bound reads, alleles, and states; candidate or state "
        "overflow is an observable failure."
    ),
    assumptions=(
        "Calibrated per-read local allele or haplotype log likelihoods",
        "Conditional independence of reads given genotype",
    ),
    nondifferentiable_outputs=("depth", "status"),
    input_dtype="float32",
    compute_dtype="float32",
    output_dtype="float32",
)

GENOTYPE_INFERENCE_CONTRACT = BioinformaticsMethodContract(
    "germline_small_variant_genotype_inference",
    MethodKind.EXACT_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Finite-state Bayes inference over the complete bounded genotype state set; "
        "hard calls are thresholded posterior decisions."
    ),
    truncation_statement="Posterior support is never pruned or truncated.",
    capacity_semantics=(
        "Inference requires a valid complete state space and complete candidate set; "
        "otherwise it returns an explicit no-call."
    ),
    assumptions=("Germline small variants", "A normalized prior over genotype states"),
    nondifferentiable_outputs=(
        "hard_call",
        "called",
        "no_call",
        "best_state_index",
        "status",
    ),
    input_dtype="float32",
    compute_dtype="float32",
    output_dtype="float32",
)


class GenotypeEnumerationEvidence(StrictModule):
    required_state_count: Array
    genotype_capacity: Array
    allele_count: Array
    ploidy: Array


class GenotypeStateSpace(StrictModule):
    """Fixed-capacity unordered genotype states in VCF Number=G order."""

    states: Array
    state_mask: Array
    state_count: Array
    valid: Array
    status: Array
    evidence: GenotypeEnumerationEvidence
    allele_count: int = eqx.field(static=True)
    ploidy: int = eqx.field(static=True)
    genotype_capacity: int = eqx.field(static=True)
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class LocalAlleleEvidence(StrictModule):
    """Bounded read evidence whose candidate columns may be full local haplotypes."""

    allele_log_likelihoods: Array
    read_mask: Array
    observed_alleles: Array
    base_quality: Array
    mapping_quality: Array
    omitted_read_mask: Array
    candidate_complete: Array
    valid: Array
    allele_count: int = eqx.field(static=True)

    def __init__(
        self,
        allele_log_likelihoods: ArrayLike,
        read_mask: ArrayLike,
        observed_alleles: ArrayLike,
        base_quality: ArrayLike,
        mapping_quality: ArrayLike,
        omitted_read_mask: ArrayLike,
        candidate_complete: ArrayLike,
        valid: ArrayLike,
        allele_count: int,
    ):
        likelihoods = jnp.asarray(allele_log_likelihoods, dtype=jnp.float32)
        mask = jnp.asarray(read_mask, dtype=bool)
        observed = jnp.asarray(observed_alleles, dtype=jnp.int32)
        base = jnp.asarray(base_quality, dtype=jnp.float32)
        mapping = jnp.asarray(mapping_quality, dtype=jnp.float32)
        omitted = jnp.asarray(omitted_read_mask, dtype=bool)
        complete = jnp.asarray(candidate_complete, dtype=bool)
        valid_ = jnp.asarray(valid, dtype=bool)
        if complete.ndim != 0 or valid_.ndim != 0:
            raise ValueError("candidate_complete and valid must be scalar.")
        if likelihoods.ndim != 2:
            raise ValueError("allele_log_likelihoods must have shape (reads, alleles).")
        if any(
            value.shape != (likelihoods.shape[0],)
            for value in (mask, observed, base, mapping, omitted)
        ):
            raise ValueError("Every read-level evidence array must share read capacity.")
        count = int(allele_count)
        if count < 1 or likelihoods.shape[1] != count:
            raise ValueError("allele_count must match the likelihood allele axis.")
        self.allele_log_likelihoods = likelihoods
        self.read_mask = jax.lax.stop_gradient(mask)
        self.observed_alleles = jax.lax.stop_gradient(observed)
        self.base_quality = jax.lax.stop_gradient(base)
        self.mapping_quality = jax.lax.stop_gradient(mapping)
        self.omitted_read_mask = jax.lax.stop_gradient(omitted)
        self.candidate_complete = jax.lax.stop_gradient(complete)
        self.valid = jax.lax.stop_gradient(valid_)
        self.allele_count = count


class GenotypeLikelihoodEvidence(StrictModule):
    depth: Array
    omitted_depth: Array
    finite_state_count: Array


class GenotypeLikelihoods(StrictModule):
    """Natural-log genotype likelihoods, separate from priors and posterior."""

    log_likelihoods: Array
    state_mask: Array
    valid: Array
    status: Array
    evidence: GenotypeLikelihoodEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class GenotypePriorEvidence(StrictModule):
    probability_sum: Array
    finite_state_count: Array


class GenotypePrior(StrictModule):
    """Normalized log prior over one bounded genotype state space."""

    log_probabilities: Array
    state_mask: Array
    valid: Array
    status: Array
    evidence: GenotypePriorEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class GenotypePosteriorEvidence(StrictModule):
    log_evidence: Array
    maximum_posterior: Array


class GenotypePosterior(StrictModule):
    """Normalized posterior and expected allele copy counts (dosage)."""

    log_probabilities: Array
    probabilities: Array
    dosage: Array
    valid: Array
    status: Array
    evidence: GenotypePosteriorEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class HardGenotypeCall(StrictModule):
    """A thresholded decision kept separate from continuous posterior output."""

    alleles: Array
    best_state_index: Array
    called: Array
    no_call: Array
    genotype_quality: Array
    depth: Array
    valid: Array
    status: Array
    evidence: GenotypeLikelihoodEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class GenotypeInferenceResult(StrictModule):
    likelihoods: GenotypeLikelihoods
    posterior: GenotypePosterior
    hard_call: HardGenotypeCall
    valid: Array
    status: Array
    evidence: GenotypeLikelihoodEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def enumerate_genotype_states(
    allele_count: int,
    ploidy: int,
    max_genotypes: int,
    /,
) -> GenotypeStateSpace:
    """Enumerate every unordered genotype, failing observably on capacity overflow."""
    alleles = int(allele_count)
    copies = int(ploidy)
    capacity = int(max_genotypes)
    if alleles < 1 or copies < 1 or capacity < 1:
        raise ValueError("allele_count, ploidy, and max_genotypes must be positive.")
    required = math.comb(alleles + copies - 1, copies)
    states = jnp.full((capacity, copies), -1, dtype=jnp.int32)
    mask = jnp.zeros((capacity,), dtype=bool)
    valid = required <= capacity
    if valid:
        ordered = sorted(
            combinations_with_replacement(range(alleles), copies),
            key=lambda genotype: tuple(reversed(genotype)),
        )
        states = states.at[:required].set(jnp.asarray(ordered, dtype=jnp.int32))
        mask = mask.at[:required].set(True)
    status = GenotypingStatus.OK if valid else GenotypingStatus.CAPACITY_EXCEEDED
    return GenotypeStateSpace(
        jax.lax.stop_gradient(states),
        jax.lax.stop_gradient(mask),
        jnp.asarray(required if valid else 0, dtype=jnp.int32),
        jnp.asarray(valid),
        jnp.asarray(int(status), dtype=jnp.int32),
        GenotypeEnumerationEvidence(
            jnp.asarray(required, dtype=jnp.int32),
            jnp.asarray(capacity, dtype=jnp.int32),
            jnp.asarray(alleles, dtype=jnp.int32),
            jnp.asarray(copies, dtype=jnp.int32),
        ),
        alleles,
        copies,
        capacity,
        GENOTYPE_ENUMERATION_CONTRACT,
    )


def local_haplotype_evidence(
    haplotype_log_likelihoods: ArrayLike,
    read_mask: ArrayLike,
    /,
    *,
    candidate_complete: ArrayLike = True,
    omitted_read_mask: ArrayLike | None = None,
) -> LocalAlleleEvidence:
    """Create local evidence when each candidate allele is a full haplotype."""
    likelihoods = jnp.asarray(haplotype_log_likelihoods, dtype=jnp.float32)
    reads = jnp.asarray(read_mask, dtype=bool)
    if likelihoods.ndim != 2 or reads.shape != (likelihoods.shape[0],):
        raise ValueError("Haplotype likelihoods must have shape (reads, candidates).")
    if likelihoods.shape[1] < 1:
        raise ValueError("At least one local haplotype candidate is required.")
    omitted = (
        jnp.zeros(reads.shape, dtype=bool)
        if omitted_read_mask is None
        else jnp.asarray(omitted_read_mask, dtype=bool)
    )
    if omitted.shape != reads.shape:
        raise ValueError("omitted_read_mask must match read capacity.")
    omitted = reads & omitted
    complete = jnp.asarray(candidate_complete, dtype=bool) & ~jnp.any(omitted)
    valid = jnp.all((~reads[:, None]) | jnp.isfinite(likelihoods))
    return LocalAlleleEvidence(
        jnp.where(reads[:, None] & ~omitted[:, None], likelihoods, 0.0),
        reads,
        -jnp.ones(reads.shape, dtype=jnp.int32),
        jnp.zeros(reads.shape, dtype=jnp.float32),
        jnp.zeros(reads.shape, dtype=jnp.float32),
        omitted,
        complete,
        valid,
        likelihoods.shape[1],
    )


def local_allele_evidence_from_calls(
    observed_alleles: ArrayLike,
    base_quality: ArrayLike,
    mapping_quality: ArrayLike,
    /,
    *,
    allele_count: int,
    read_mask: ArrayLike | None = None,
) -> LocalAlleleEvidence:
    """Calibrate allele-call and mapping Phred qualities into local likelihoods."""
    observed = jnp.asarray(observed_alleles, dtype=jnp.int32)
    base = jnp.asarray(base_quality, dtype=jnp.float32)
    mapping = jnp.asarray(mapping_quality, dtype=jnp.float32)
    if (
        observed.ndim != 1
        or base.shape != observed.shape
        or mapping.shape != observed.shape
    ):
        raise ValueError(
            "Observed alleles and qualities must be matching rank-one arrays."
        )
    count = int(allele_count)
    if count < 1:
        raise ValueError("allele_count must be positive.")
    mask = (
        jnp.ones(observed.shape, dtype=bool)
        if read_mask is None
        else jnp.asarray(read_mask, dtype=bool)
    )
    if mask.shape != observed.shape:
        raise ValueError("read_mask must match read capacity.")

    quality_valid = jnp.all(
        (~mask)
        | (jnp.isfinite(base) & jnp.isfinite(mapping) & (base >= 0.0) & (mapping >= 0.0))
    )
    omitted = mask & ((observed < 0) | (observed >= count))
    complete = ~jnp.any(omitted)
    safe_observed = jnp.clip(observed, 0, count - 1)
    safe_base = jnp.where(jnp.isfinite(base), jnp.maximum(base, 0.0), 0.0)
    safe_mapping = jnp.where(jnp.isfinite(mapping), jnp.maximum(mapping, 0.0), 0.0)
    base_error = jnp.power(10.0, -0.1 * safe_base)
    mapping_error = jnp.power(10.0, -0.1 * safe_mapping)
    error = 1.0 - (1.0 - base_error) * (1.0 - mapping_error)
    error = jnp.clip(error, 1.0e-7, 1.0 - 1.0e-7)
    alternatives = max(count - 1, 1)
    calls = jnp.arange(count, dtype=jnp.int32)[None, :]
    match = calls == safe_observed[:, None]
    probabilities = jnp.where(
        match,
        1.0 - error[:, None],
        error[:, None] / float(alternatives),
    )
    if count == 1:
        probabilities = jnp.ones_like(probabilities)
    log_likelihoods = jnp.where(
        (mask & ~omitted)[:, None],
        jnp.log(probabilities),
        0.0,
    )
    return LocalAlleleEvidence(
        log_likelihoods,
        mask,
        observed,
        base,
        mapping,
        omitted,
        complete,
        quality_valid,
        count,
    )


def _likelihood_result(
    values: Array,
    state_space: GenotypeStateSpace,
    *,
    depth: ArrayLike,
    omitted_depth: ArrayLike,
    valid: ArrayLike,
    status: ArrayLike,
) -> GenotypeLikelihoods:
    finite_count = jnp.sum(state_space.state_mask & jnp.isfinite(values), dtype=jnp.int32)
    return GenotypeLikelihoods(
        jnp.asarray(values, dtype=jnp.float32),
        state_space.state_mask,
        jnp.asarray(valid, dtype=bool),
        jnp.asarray(status, dtype=jnp.int32),
        GenotypeLikelihoodEvidence(
            jnp.asarray(depth, dtype=jnp.int32),
            jnp.asarray(omitted_depth, dtype=jnp.int32),
            finite_count,
        ),
        GENOTYPE_LIKELIHOOD_CONTRACT,
    )


def _external_likelihoods(
    values: ArrayLike,
    state_space: GenotypeStateSpace,
    *,
    scale: str,
    depth: int,
) -> GenotypeLikelihoods:
    supplied = jnp.asarray(values, dtype=jnp.float32)
    if supplied.ndim != 1:
        raise ValueError("Genotype likelihood values must be rank one.")
    required = math.comb(
        state_space.allele_count + state_space.ploidy - 1, state_space.ploidy
    )
    shape_valid = (
        supplied.shape[0] == required and required <= state_space.genotype_capacity
    )
    padded = jnp.full((state_space.genotype_capacity,), -jnp.inf, dtype=jnp.float32)
    if shape_valid:
        if scale == "gl":
            converted = supplied * jnp.asarray(math.log(10.0), dtype=jnp.float32)
        else:
            converted = -0.1 * jnp.asarray(math.log(10.0), dtype=jnp.float32) * supplied
        converted = converted - jnp.max(converted)
        padded = padded.at[:required].set(converted)
    valid = state_space.valid & jnp.asarray(shape_valid) & jnp.all(jnp.isfinite(supplied))
    status = jnp.where(
        valid,
        int(GenotypingStatus.OK),
        int(GenotypingStatus.INVALID_INPUT),
    )
    return _likelihood_result(
        padded,
        state_space,
        depth=depth,
        omitted_depth=0,
        valid=valid,
        status=status,
    )


def genotype_likelihoods_from_gl(
    gl: ArrayLike,
    state_space: GenotypeStateSpace,
    /,
    *,
    depth: int = 0,
) -> GenotypeLikelihoods:
    """Convert VCF GL (base-10 log likelihoods) to natural-log likelihoods."""
    return _external_likelihoods(gl, state_space, scale="gl", depth=int(depth))


def genotype_likelihoods_from_pl(
    pl: ArrayLike,
    state_space: GenotypeStateSpace,
    /,
    *,
    depth: int = 0,
) -> GenotypeLikelihoods:
    """Convert VCF PL (Phred-scaled likelihoods) to natural-log likelihoods."""
    return _external_likelihoods(pl, state_space, scale="pl", depth=int(depth))


def genotype_likelihoods_to_gl(likelihoods: GenotypeLikelihoods, /) -> Array:
    """Return base-10 GL values with NaN in unused bounded state slots."""
    values = likelihoods.log_likelihoods / jnp.asarray(math.log(10.0), dtype=jnp.float32)
    return jnp.where(likelihoods.state_mask, values, jnp.nan)


def genotype_likelihoods_to_pl(likelihoods: GenotypeLikelihoods, /) -> Array:
    """Return integer PL values with -1 in unused bounded state slots."""
    scaled = (
        -10.0
        * likelihoods.log_likelihoods
        / jnp.asarray(math.log(10.0), dtype=jnp.float32)
    )
    return jnp.where(
        likelihoods.state_mask,
        jnp.rint(scaled).astype(jnp.int32),
        -jnp.ones_like(scaled, dtype=jnp.int32),
    )


def genotype_likelihoods_from_reads(
    evidence: LocalAlleleEvidence,
    state_space: GenotypeStateSpace,
    /,
) -> GenotypeLikelihoods:
    """Marginalize each local read over chromosome copies for every genotype."""
    if evidence.allele_count != state_space.allele_count:
        raise ValueError("Evidence and genotype state space must share allele count.")
    safe_states = jnp.clip(state_space.states, 0, state_space.allele_count - 1)
    copy_log_likelihoods = jnp.take(evidence.allele_log_likelihoods, safe_states, axis=1)
    per_read = jsp.special.logsumexp(copy_log_likelihoods, axis=-1) - math.log(
        state_space.ploidy
    )
    per_read = jnp.where(evidence.read_mask[:, None], per_read, 0.0)
    values = compensated_sum(per_read, axis=0)
    state_maximum = jnp.max(
        jnp.where(state_space.state_mask, values, -jnp.inf),
        initial=-jnp.inf,
    )
    state_maximum = jnp.where(jnp.any(state_space.state_mask), state_maximum, 0.0)
    values = jnp.where(state_space.state_mask, values - state_maximum, -jnp.inf)
    depth = jnp.sum(evidence.read_mask, dtype=jnp.int32)
    omitted_depth = jnp.sum(evidence.omitted_read_mask, dtype=jnp.int32)
    valid = state_space.valid & evidence.valid & evidence.candidate_complete
    status = jnp.where(
        ~state_space.valid,
        int(GenotypingStatus.CAPACITY_EXCEEDED),
        jnp.where(
            ~evidence.valid,
            int(GenotypingStatus.INVALID_INPUT),
            jnp.where(
                ~evidence.candidate_complete,
                int(GenotypingStatus.CANDIDATE_OMITTED),
                jnp.where(
                    depth == 0,
                    int(GenotypingStatus.NO_COVERAGE),
                    int(GenotypingStatus.OK),
                ),
            ),
        ),
    )
    return _likelihood_result(
        values,
        state_space,
        depth=depth,
        omitted_depth=omitted_depth,
        valid=valid,
        status=status,
    )


def uniform_genotype_prior(state_space: GenotypeStateSpace, /) -> GenotypePrior:
    """Construct a uniform prior over every populated genotype state."""
    count = jnp.sum(state_space.state_mask, dtype=jnp.float32)
    log_probabilities = jnp.where(
        state_space.state_mask, -jnp.log(jnp.maximum(count, 1.0)), -jnp.inf
    )
    return GenotypePrior(
        log_probabilities,
        state_space.state_mask,
        state_space.valid,
        jnp.where(
            state_space.valid,
            int(GenotypingStatus.OK),
            int(GenotypingStatus.CAPACITY_EXCEEDED),
        ).astype(jnp.int32),
        GenotypePriorEvidence(
            jnp.where(state_space.valid, 1.0, 0.0),
            jnp.sum(state_space.state_mask, dtype=jnp.int32),
        ),
        GENOTYPE_INFERENCE_CONTRACT,
    )


def allele_frequency_genotype_prior(
    allele_frequencies: ArrayLike,
    state_space: GenotypeStateSpace,
    /,
) -> GenotypePrior:
    """Construct the random-mating multinomial prior for arbitrary ploidy."""
    frequencies = jnp.asarray(allele_frequencies, dtype=jnp.float32)
    if frequencies.shape != (state_space.allele_count,):
        raise ValueError("allele_frequencies must match state-space allele count.")
    finite_nonnegative = jnp.all(jnp.isfinite(frequencies) & (frequencies >= 0.0))
    total = compensated_sum(frequencies)
    normalized = frequencies / jnp.maximum(total, jnp.finfo(jnp.float32).tiny)
    safe_states = jnp.clip(state_space.states, 0, state_space.allele_count - 1)
    counts = jnp.sum(
        jax.nn.one_hot(safe_states, state_space.allele_count, dtype=jnp.float32),
        axis=1,
    )
    log_frequencies = jnp.where(normalized > 0.0, jnp.log(normalized), -jnp.inf)
    frequency_term = compensated_sum(
        counts * jnp.where(counts > 0.0, log_frequencies[None, :], 0.0),
        axis=1,
    )
    coefficient = jsp.special.gammaln(float(state_space.ploidy) + 1.0) - compensated_sum(
        jsp.special.gammaln(counts + 1.0), axis=1
    )
    raw = coefficient + frequency_term
    raw = jnp.where(state_space.state_mask, raw, -jnp.inf)
    valid = (
        state_space.valid
        & finite_nonnegative
        & jnp.isfinite(total)
        & (total > 0.0)
        & (jnp.abs(total - 1.0) <= 1.0e-5)
    )
    log_normalizer = jsp.special.logsumexp(raw)
    log_probabilities = jnp.where(valid, raw - log_normalizer, -jnp.inf)
    probability_sum = jnp.where(valid, compensated_sum(jnp.exp(log_probabilities)), 0.0)
    return GenotypePrior(
        log_probabilities,
        state_space.state_mask,
        valid,
        jnp.where(
            valid,
            int(GenotypingStatus.OK),
            int(GenotypingStatus.INVALID_PRIOR),
        ).astype(jnp.int32),
        GenotypePriorEvidence(
            probability_sum,
            jnp.sum(
                state_space.state_mask & jnp.isfinite(log_probabilities),
                dtype=jnp.int32,
            ),
        ),
        GENOTYPE_INFERENCE_CONTRACT,
    )


def infer_genotype(
    likelihoods: GenotypeLikelihoods,
    prior: GenotypePrior,
    state_space: GenotypeStateSpace,
    /,
    *,
    min_depth: int = 1,
    min_posterior: float = 0.9,
    max_genotype_quality: float = 99.0,
) -> GenotypeInferenceResult:
    """Combine likelihood and prior, then make an explicitly separable hard call."""
    if likelihoods.log_likelihoods.shape != state_space.state_mask.shape:
        raise ValueError("Likelihoods must match genotype capacity.")
    if prior.log_probabilities.shape != state_space.state_mask.shape:
        raise ValueError("Prior must match genotype capacity.")
    depth_threshold = int(min_depth)
    posterior_threshold = float(min_posterior)
    quality_cap = float(max_genotype_quality)
    if depth_threshold < 0 or not 0.0 <= posterior_threshold <= 1.0:
        raise ValueError("min_depth and min_posterior must be valid thresholds.")
    if not math.isfinite(quality_cap) or quality_cap < 0.0:
        raise ValueError("max_genotype_quality must be finite and non-negative.")

    combined = likelihoods.log_likelihoods + prior.log_probabilities
    combined = jnp.where(state_space.state_mask, combined, -jnp.inf)
    finite_support = jnp.any(jnp.isfinite(combined))
    log_evidence = jsp.special.logsumexp(combined)
    log_posterior = jnp.where(finite_support, combined - log_evidence, -jnp.inf)
    probabilities = jnp.where(
        state_space.state_mask & jnp.isfinite(log_posterior),
        jnp.exp(log_posterior),
        0.0,
    )
    posterior_valid = state_space.valid & likelihoods.valid & prior.valid & finite_support
    probabilities = jnp.where(posterior_valid, probabilities, 0.0)
    safe_states = jnp.clip(state_space.states, 0, state_space.allele_count - 1)
    allele_counts = jnp.sum(
        jax.nn.one_hot(safe_states, state_space.allele_count, dtype=jnp.float32),
        axis=1,
    )
    dosage = compensated_sum(probabilities[:, None] * allele_counts, axis=0)
    best_index = jnp.argmax(probabilities).astype(jnp.int32)
    maximum = probabilities[best_index]
    depth = likelihoods.evidence.depth
    enough_depth = depth >= depth_threshold
    called = posterior_valid & enough_depth & (maximum >= posterior_threshold)
    hard_alleles = jnp.where(called, state_space.states[best_index], -1)
    error_probability = jnp.maximum(1.0 - maximum, 1.0e-10)
    genotype_quality = jnp.where(
        called,
        jnp.minimum(-10.0 * jnp.log10(error_probability), quality_cap),
        0.0,
    )

    failure_status = jnp.where(
        ~likelihoods.valid,
        likelihoods.status,
        jnp.where(~prior.valid, prior.status, int(GenotypingStatus.INVALID_INPUT)),
    )
    decision_status = jnp.where(
        depth == 0,
        int(GenotypingStatus.NO_COVERAGE),
        jnp.where(
            enough_depth & (maximum >= posterior_threshold),
            int(GenotypingStatus.OK),
            int(GenotypingStatus.LOW_CONFIDENCE),
        ),
    )
    status = jnp.where(posterior_valid, decision_status, failure_status).astype(jnp.int32)
    posterior = GenotypePosterior(
        log_posterior,
        probabilities,
        dosage,
        posterior_valid,
        status,
        GenotypePosteriorEvidence(log_evidence, maximum),
        GENOTYPE_INFERENCE_CONTRACT,
    )
    hard_call = HardGenotypeCall(
        hard_alleles,
        best_index,
        called,
        ~called,
        genotype_quality,
        depth,
        posterior_valid,
        status,
        likelihoods.evidence,
        GENOTYPE_INFERENCE_CONTRACT,
    )
    return GenotypeInferenceResult(
        likelihoods,
        posterior,
        hard_call,
        posterior_valid,
        status,
        likelihoods.evidence,
        GENOTYPE_INFERENCE_CONTRACT,
    )


__all__ = [
    "GENOTYPE_ENUMERATION_CONTRACT",
    "GENOTYPE_INFERENCE_CONTRACT",
    "GENOTYPE_LIKELIHOOD_CONTRACT",
    "GenotypeEnumerationEvidence",
    "GenotypeInferenceResult",
    "GenotypeLikelihoodEvidence",
    "GenotypeLikelihoods",
    "GenotypePosterior",
    "GenotypePosteriorEvidence",
    "GenotypePrior",
    "GenotypePriorEvidence",
    "GenotypeStateSpace",
    "GenotypingStatus",
    "HardGenotypeCall",
    "LocalAlleleEvidence",
    "allele_frequency_genotype_prior",
    "enumerate_genotype_states",
    "genotype_likelihoods_from_gl",
    "genotype_likelihoods_from_pl",
    "genotype_likelihoods_from_reads",
    "genotype_likelihoods_to_gl",
    "genotype_likelihoods_to_pl",
    "infer_genotype",
    "local_allele_evidence_from_calls",
    "local_haplotype_evidence",
    "uniform_genotype_prior",
]
