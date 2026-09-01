#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Copy-aware matched tumor-normal somatic likelihoods."""

from __future__ import annotations

from enum import IntEnum, IntFlag
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class SomaticHypothesis(IntEnum):
    SOMATIC = 0
    GERMLINE = 1
    ARTIFACT = 2


class SomaticStatus(IntEnum):
    SUCCESS = 0
    NO_MATCHED_NORMAL = 1
    INVALID_COUNTS = 2
    INVALID_CONTEXT = 3
    CAPACITY_EXCEEDED = 4


class SomaticCandidateLimitation(IntFlag):
    """Limitations inherited from upstream candidate generation."""

    NONE = 0
    PRECOMPUTED_CANDIDATES = 1
    SEARCH_SPACE_NOT_EXHAUSTIVE = 2
    CAPACITY_EXCEEDED = 4


class TumorNormalAlleleCounts(StrictModule):
    """Alternate/total counts with an explicit matched-normal availability mask."""

    tumor_alt_count: Array
    tumor_total_count: Array
    normal_alt_count: Array
    normal_total_count: Array
    matched_normal: Array

    def __init__(
        self,
        tumor_alt_count: ArrayLike,
        tumor_total_count: ArrayLike,
        normal_alt_count: ArrayLike,
        normal_total_count: ArrayLike,
        matched_normal: ArrayLike,
        /,
    ):
        tumor_alt = jnp.asarray(tumor_alt_count, dtype=jnp.int32)
        tumor_total = jnp.asarray(tumor_total_count, dtype=jnp.int32)
        normal_alt = jnp.asarray(normal_alt_count, dtype=jnp.int32)
        normal_total = jnp.asarray(normal_total_count, dtype=jnp.int32)
        has_normal = jnp.asarray(matched_normal, dtype=bool)
        if tumor_alt.ndim != 1 or tumor_alt.shape[0] < 1:
            raise ValueError("Tumor-normal allele counts must be non-empty vectors.")
        if any(
            value.shape != tumor_alt.shape
            for value in (tumor_total, normal_alt, normal_total, has_normal)
        ):
            raise ValueError(
                "Tumor-normal allele-count fields must have identical shapes."
            )
        self.tumor_alt_count = tumor_alt
        self.tumor_total_count = tumor_total
        self.normal_alt_count = normal_alt
        self.normal_total_count = normal_total
        self.matched_normal = has_normal


class SomaticCopyContext(StrictModule):
    """Purity, contamination, allele-specific copy, and subclonality context.

    ``normal_tumor_contamination`` is the malignant-DNA fraction in the matched
    normal library. ``tumor_purity`` is the malignant-DNA fraction in the tumor
    library, so normal-cell contamination of tumor is exactly ``1 - purity``.
    """

    tumor_purity: Array
    normal_tumor_contamination: Array
    tumor_total_copy: Array
    normal_total_copy: Array
    mutated_copy: Array
    subclone_fraction: Array

    def __init__(
        self,
        tumor_purity: ArrayLike,
        normal_tumor_contamination: ArrayLike,
        tumor_total_copy: ArrayLike,
        normal_total_copy: ArrayLike,
        mutated_copy: ArrayLike,
        subclone_fraction: ArrayLike,
        /,
    ):
        purity = jnp.asarray(tumor_purity, dtype=jnp.float32)
        contamination = jnp.asarray(normal_tumor_contamination, dtype=jnp.float32)
        tumor_copy = jnp.asarray(tumor_total_copy, dtype=jnp.float32)
        normal_copy = jnp.asarray(normal_total_copy, dtype=jnp.float32)
        mutated = jnp.asarray(mutated_copy, dtype=jnp.float32)
        subclone = jnp.asarray(subclone_fraction, dtype=jnp.float32)
        shape = purity.shape
        if purity.ndim != 1:
            raise ValueError("Somatic copy-context fields must be vectors.")
        if any(
            value.shape != shape
            for value in (contamination, tumor_copy, normal_copy, mutated, subclone)
        ):
            raise ValueError("Somatic copy-context fields must have identical shapes.")
        self.tumor_purity = purity
        self.normal_tumor_contamination = contamination
        self.tumor_total_copy = tumor_copy
        self.normal_total_copy = normal_copy
        self.mutated_copy = mutated
        self.subclone_fraction = subclone


class SomaticPanelProvenance(StrictModule):
    """Panel-of-normals support without embedding files or third-party objects."""

    panel_id: str = eqx.field(static=True)
    sample_count: Array
    covered: Array
    artifact_alt_frequency: Array

    def __init__(
        self,
        panel_id: str,
        sample_count: int,
        covered: ArrayLike,
        artifact_alt_frequency: ArrayLike,
        /,
    ):
        identifier = str(panel_id)
        samples = int(sample_count)
        coverage = jnp.asarray(covered, dtype=bool)
        frequency = jnp.asarray(artifact_alt_frequency, dtype=jnp.float32)
        if not identifier:
            raise ValueError("panel_id must be non-empty.")
        if samples < 0:
            raise ValueError("sample_count must be non-negative.")
        if coverage.ndim != 1 or frequency.shape != coverage.shape:
            raise ValueError(
                "Panel coverage and artifact frequency must be aligned vectors."
            )
        if bool(jnp.any(~jnp.isfinite(frequency))) or bool(
            jnp.any((frequency < 0.0) | (frequency > 1.0))
        ):
            raise ValueError("artifact_alt_frequency must lie in [0, 1].")
        if samples == 0 and bool(jnp.any(coverage)):
            raise ValueError("A zero-sample panel cannot mark candidates covered.")
        self.panel_id = canonical_fingerprint(
            {
                "kind": "somatic-panel-of-normals",
                "source_id": identifier,
                "sample_count": samples,
            }
        )
        self.sample_count = jnp.asarray(samples, dtype=jnp.int32)
        self.covered = coverage
        self.artifact_alt_frequency = frequency


class SomaticLikelihoodPlan(StrictModule):
    """Bounded conditional binomial model for three biological hypotheses."""

    maximum_candidates: int = eqx.field(static=True)
    sequencing_error_rate: Array
    somatic_prior: Array
    germline_prior: Array
    artifact_prior: Array

    def __init__(
        self,
        *,
        maximum_candidates: int,
        sequencing_error_rate: float = 1e-3,
        somatic_prior: float = 1e-3,
        germline_prior: float = 1e-3,
        artifact_prior: float = 0.998,
    ):
        capacity = int(maximum_candidates)
        error = float(sequencing_error_rate)
        priors = (float(somatic_prior), float(germline_prior), float(artifact_prior))
        if capacity < 1:
            raise ValueError("maximum_candidates must be positive.")
        if not isfinite(error) or not 0.0 < error < 0.5:
            raise ValueError("sequencing_error_rate must lie in (0, 0.5).")
        if any((not isfinite(value)) or value <= 0.0 for value in priors):
            raise ValueError("Somatic hypothesis priors must be finite and positive.")
        if abs(sum(priors) - 1.0) > 1e-6:
            raise ValueError("Somatic hypothesis priors must sum to one.")
        self.maximum_candidates = capacity
        self.sequencing_error_rate = jnp.asarray(error, dtype=jnp.float32)
        self.somatic_prior = jnp.asarray(priors[0], dtype=jnp.float32)
        self.germline_prior = jnp.asarray(priors[1], dtype=jnp.float32)
        self.artifact_prior = jnp.asarray(priors[2], dtype=jnp.float32)


class SomaticCandidateEvidence(StrictModule):
    supplied_candidates: Array
    candidates_scored: Array
    candidate_generation_performed: Array
    search_space_exhaustive: Array
    limitation_mask: Array
    capacity_sufficient: Array


class SomaticLikelihoodEvidence(StrictModule):
    expected_tumor_alt_fraction: Array
    expected_normal_alt_fraction: Array
    normal_likelihood_used: Array
    copy_context_used: Array
    subclonal_context_used: Array
    panel_covered: Array
    panel_sample_count: Array
    candidate_generation: SomaticCandidateEvidence


class SomaticLikelihoodResult(StrictModule):
    log_likelihood: Array
    posterior_probability: Array
    somatic_probability: Array
    call: Array
    valid: Array
    status: Array
    evidence: SomaticLikelihoodEvidence
    panel_provenance: SomaticPanelProvenance
    method_contract: BioinformaticsMethodContract


def _binomial_log_likelihood(
    successes: Array, trials: Array, probability: Array, /
) -> Array:
    probability = jnp.clip(
        probability,
        jnp.finfo(probability.dtype).eps,
        1.0 - jnp.finfo(probability.dtype).eps,
    )
    successes_f = successes.astype(probability.dtype)
    trials_f = trials.astype(probability.dtype)
    return (
        jsp.special.gammaln(trials_f + 1.0)
        - jsp.special.gammaln(successes_f + 1.0)
        - jsp.special.gammaln(trials_f - successes_f + 1.0)
        + successes_f * jnp.log(probability)
        + (trials_f - successes_f) * jnp.log1p(-probability)
    )


def _observed_probability(
    biological_fraction: Array, error_rate: Array, /
) -> Array:
    return error_rate + (1.0 - 2.0 * error_rate) * biological_fraction


def _somatic_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "copy-aware-tumor-normal-somatic-likelihood",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Exact binomial likelihood conditional on supplied candidates, purity, "
            "malignant contamination, total/mutated copy, subclone fraction, and "
            "panel summaries."
        ),
        truncation_statement=(
            "Candidate generation is not performed; upstream candidate omissions "
            "remain unscored and are explicitly reported."
        ),
        capacity_semantics="Candidate capacity is preflighted; excess returns CAPACITY_EXCEEDED without truncation.",
        assumptions=(
            "counts are conditionally binomial",
            "germline hypothesis is balanced heterozygous",
            "panel frequency is an artifact probability proxy",
        ),
        nondifferentiable_outputs=("call", "status", "counts", "matched-normal mask"),
    )


def _candidate_evidence(count: int, capacity: int, /) -> SomaticCandidateEvidence:
    sufficient = count <= capacity
    limitation = int(
        SomaticCandidateLimitation.PRECOMPUTED_CANDIDATES
        | SomaticCandidateLimitation.SEARCH_SPACE_NOT_EXHAUSTIVE
    )
    if not sufficient:
        limitation |= int(SomaticCandidateLimitation.CAPACITY_EXCEEDED)
    return SomaticCandidateEvidence(
        supplied_candidates=jnp.asarray(count, dtype=jnp.int32),
        candidates_scored=jnp.asarray(count if sufficient else 0, dtype=jnp.int32),
        candidate_generation_performed=jnp.asarray(False),
        search_space_exhaustive=jnp.asarray(False),
        limitation_mask=jnp.asarray(limitation, dtype=jnp.int32),
        capacity_sufficient=jnp.asarray(sufficient),
    )


def _capacity_failure(
    count: int,
    panel: SomaticPanelProvenance,
    candidates: SomaticCandidateEvidence,
    /,
) -> SomaticLikelihoodResult:
    zeros = jnp.zeros((count, 3), dtype=jnp.float32)
    evidence = SomaticLikelihoodEvidence(
        expected_tumor_alt_fraction=zeros,
        expected_normal_alt_fraction=zeros,
        normal_likelihood_used=jnp.zeros((count,), dtype=bool),
        copy_context_used=jnp.asarray(False),
        subclonal_context_used=jnp.zeros((count,), dtype=bool),
        panel_covered=panel.covered,
        panel_sample_count=jnp.asarray(panel.sample_count, dtype=jnp.int32),
        candidate_generation=candidates,
    )
    return SomaticLikelihoodResult(
        log_likelihood=zeros,
        posterior_probability=zeros,
        somatic_probability=jnp.zeros((count,), dtype=jnp.float32),
        call=jnp.zeros((count,), dtype=jnp.int32),
        valid=jnp.zeros((count,), dtype=bool),
        status=jnp.full((count,), int(SomaticStatus.CAPACITY_EXCEEDED), dtype=jnp.int32),
        evidence=evidence,
        panel_provenance=panel,
        method_contract=_somatic_contract(),
    )


def somatic_likelihoods(
    counts: TumorNormalAlleleCounts,
    context: SomaticCopyContext,
    panel: SomaticPanelProvenance,
    plan: SomaticLikelihoodPlan,
    /,
) -> SomaticLikelihoodResult:
    """Evaluate somatic, balanced-germline, and artifact hypotheses."""
    if not isinstance(counts, TumorNormalAlleleCounts):
        raise TypeError("counts must be TumorNormalAlleleCounts.")
    if not isinstance(context, SomaticCopyContext):
        raise TypeError("context must be SomaticCopyContext.")
    if not isinstance(panel, SomaticPanelProvenance):
        raise TypeError("panel must be SomaticPanelProvenance.")
    if not isinstance(plan, SomaticLikelihoodPlan):
        raise TypeError("plan must be SomaticLikelihoodPlan.")
    count = counts.tumor_alt_count.shape[0]
    if context.tumor_purity.shape != (count,) or panel.covered.shape != (count,):
        raise ValueError("Counts, copy context, and panel provenance must align.")
    candidates = _candidate_evidence(count, plan.maximum_candidates)
    if count > plan.maximum_candidates:
        return _capacity_failure(count, panel, candidates)
    count_valid = (
        (counts.tumor_alt_count >= 0)
        & (counts.tumor_total_count >= counts.tumor_alt_count)
        & (counts.tumor_total_count > 0)
        & (
            ~counts.matched_normal
            | (
                (counts.normal_alt_count >= 0)
                & (counts.normal_total_count >= counts.normal_alt_count)
                & (counts.normal_total_count > 0)
            )
        )
    )
    finite_context = (
        jnp.isfinite(context.tumor_purity)
        & jnp.isfinite(context.normal_tumor_contamination)
        & jnp.isfinite(context.tumor_total_copy)
        & jnp.isfinite(context.normal_total_copy)
        & jnp.isfinite(context.mutated_copy)
        & jnp.isfinite(context.subclone_fraction)
    )
    context_valid = (
        finite_context
        & (context.tumor_purity >= 0.0)
        & (context.tumor_purity <= 1.0)
        & (context.normal_tumor_contamination >= 0.0)
        & (context.normal_tumor_contamination <= 1.0)
        & (context.tumor_total_copy > 0.0)
        & (context.normal_total_copy > 0.0)
        & (context.mutated_copy > 0.0)
        & (context.mutated_copy <= context.tumor_total_copy)
        & (context.subclone_fraction > 0.0)
        & (context.subclone_fraction <= 1.0)
    )
    per_candidate_valid = count_valid & context_valid & candidates.capacity_sufficient

    purity = context.tumor_purity
    contamination = context.normal_tumor_contamination
    tumor_denominator = (
        purity * context.tumor_total_copy + (1.0 - purity) * context.normal_total_copy
    )
    normal_denominator = (
        contamination * context.tumor_total_copy
        + (1.0 - contamination) * context.normal_total_copy
    )
    safe_tumor_denominator = jnp.where(
        jnp.isfinite(tumor_denominator) & (tumor_denominator > 0.0),
        tumor_denominator,
        1.0,
    )
    safe_normal_denominator = jnp.where(
        jnp.isfinite(normal_denominator) & (normal_denominator > 0.0),
        normal_denominator,
        1.0,
    )
    somatic_tumor_fraction = jnp.where(
        context_valid,
        jnp.clip(
            purity
            * context.subclone_fraction
            * context.mutated_copy
            / safe_tumor_denominator,
            0.0,
            1.0,
        ),
        0.0,
    )
    somatic_normal_fraction = jnp.where(
        context_valid,
        jnp.clip(
            contamination
            * context.subclone_fraction
            * context.mutated_copy
            / safe_normal_denominator,
            0.0,
            1.0,
        ),
        0.0,
    )
    germline_fraction = jnp.full_like(somatic_tumor_fraction, 0.5)
    panel_artifact = jnp.where(
        panel.covered,
        jnp.maximum(panel.artifact_alt_frequency, plan.sequencing_error_rate),
        plan.sequencing_error_rate,
    )
    tumor_probabilities = jnp.stack(
        (
            _observed_probability(somatic_tumor_fraction, plan.sequencing_error_rate),
            _observed_probability(germline_fraction, plan.sequencing_error_rate),
            panel_artifact,
        ),
        axis=-1,
    )
    normal_probabilities = jnp.stack(
        (
            _observed_probability(somatic_normal_fraction, plan.sequencing_error_rate),
            _observed_probability(germline_fraction, plan.sequencing_error_rate),
            panel_artifact,
        ),
        axis=-1,
    )
    safe_tumor_total = jnp.maximum(counts.tumor_total_count, 0)
    safe_tumor_alt = jnp.clip(counts.tumor_alt_count, 0, safe_tumor_total)
    safe_normal_total = jnp.maximum(counts.normal_total_count, 0)
    safe_normal_alt = jnp.clip(counts.normal_alt_count, 0, safe_normal_total)
    tumor_log_likelihood = _binomial_log_likelihood(
        safe_tumor_alt[:, None],
        safe_tumor_total[:, None],
        tumor_probabilities,
    )
    normal_log_likelihood = _binomial_log_likelihood(
        safe_normal_alt[:, None],
        safe_normal_total[:, None],
        normal_probabilities,
    )
    data_log_likelihood = tumor_log_likelihood + jnp.where(
        counts.matched_normal[:, None], normal_log_likelihood, 0.0
    )
    log_prior = jnp.log(
        jnp.stack((plan.somatic_prior, plan.germline_prior, plan.artifact_prior)).astype(
            data_log_likelihood.dtype
        )
    )
    joint = data_log_likelihood + log_prior
    posterior = jnp.exp(joint - jsp.special.logsumexp(joint, axis=-1, keepdims=True))
    call = jnp.argmax(posterior, axis=-1).astype(jnp.int32)
    status = jnp.where(
        ~candidates.capacity_sufficient,
        int(SomaticStatus.CAPACITY_EXCEEDED),
        jnp.where(
            ~count_valid,
            int(SomaticStatus.INVALID_COUNTS),
            jnp.where(
                ~context_valid,
                int(SomaticStatus.INVALID_CONTEXT),
                jnp.where(
                    counts.matched_normal,
                    int(SomaticStatus.SUCCESS),
                    int(SomaticStatus.NO_MATCHED_NORMAL),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = SomaticLikelihoodEvidence(
        expected_tumor_alt_fraction=tumor_probabilities,
        expected_normal_alt_fraction=normal_probabilities,
        normal_likelihood_used=counts.matched_normal,
        copy_context_used=jnp.asarray(True),
        subclonal_context_used=context.subclone_fraction < 1.0,
        panel_covered=panel.covered,
        panel_sample_count=jnp.asarray(panel.sample_count, dtype=jnp.int32),
        candidate_generation=candidates,
    )
    return SomaticLikelihoodResult(
        log_likelihood=data_log_likelihood,
        posterior_probability=posterior,
        somatic_probability=posterior[:, int(SomaticHypothesis.SOMATIC)],
        call=call,
        valid=per_candidate_valid,
        status=status,
        evidence=evidence,
        panel_provenance=panel,
        method_contract=_somatic_contract(),
    )


__all__ = [
    "SomaticCandidateEvidence",
    "SomaticCandidateLimitation",
    "SomaticCopyContext",
    "SomaticHypothesis",
    "SomaticLikelihoodEvidence",
    "SomaticLikelihoodPlan",
    "SomaticLikelihoodResult",
    "SomaticPanelProvenance",
    "SomaticStatus",
    "TumorNormalAlleleCounts",
    "somatic_likelihoods",
]
