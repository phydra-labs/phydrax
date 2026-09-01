#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...sparse import RowRelation
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExchangeabilityPlan,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


TRANSCRIPT_SUCCESS = 0
TRANSCRIPT_INVALID_COUNTS = 1
TRANSCRIPT_EMPTY_LIBRARY = 2
TRANSCRIPT_UNSUPPORTED_EQUIVALENCE_CLASS = 3
TRANSCRIPT_NONCONVERGED = 4
TRANSCRIPT_MISSING_EXCHANGEABILITY = 5
TRANSCRIPT_EMPTY_GROUP = 6


def transcript_status_name(status: int, /) -> str:
    """Return the stable name of a transcript-analysis status code."""
    names = (
        "success",
        "invalid_counts_or_lengths",
        "empty_library",
        "unsupported_equivalence_class",
        "nonconverged",
        "missing_exchangeability_plan",
        "empty_contrast_group",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown transcript-analysis status {code}.")
    return names[code]


def _abundance_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "transcript_equivalence_class_abundance",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.ITERATIVE_TOLERANCE,
        DifferentiationKind.UNROLLED,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Abundance is conditioned on fixed equivalence classes, transcript "
            "effective lengths, and a multinomial fragment-allocation model."
        ),
        truncation_statement=(
            "Iteration stops at tolerance or the declared maximum; nonconvergence "
            "is returned as status rather than accepted silently."
        ),
        capacity_semantics=(
            "Equivalence-class route width is explicit; invalid routes are masked "
            "and every positive-count class must have at least one route."
        ),
        assumptions=(
            "Fragments within a sample are exchangeable under the allocation model.",
            "Effective lengths are positive and fixed.",
        ),
        nondifferentiable_outputs=("iterations", "status", "valid"),
    )


def _usage_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "differential_transcript_usage_primitive",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Usage contrasts condition on transcript abundance estimates and the "
            "declared sample exchangeability structure."
        ),
        truncation_statement="No transcripts, genes, or samples are truncated.",
        capacity_semantics="Gene-to-transcript route width is explicit and masked.",
        assumptions=("Biological samples, not fragments, are replicate units.",),
        nondifferentiable_outputs=("status", "valid"),
    )


class TranscriptEquivalenceBatch(StrictModule):
    """Sample-by-class counts and fixed-width class-to-transcript compatibility."""

    counts: Array
    compatibility: RowRelation
    effective_lengths: Array

    def __init__(
        self,
        counts: ArrayLike,
        compatibility: RowRelation,
        effective_lengths: ArrayLike,
        /,
    ):
        values = jnp.asarray(counts)
        lengths = jnp.asarray(effective_lengths)
        if not isinstance(compatibility, RowRelation):
            raise TypeError("compatibility must be a RowRelation.")
        if values.ndim != 2:
            raise ValueError("Equivalence-class counts must have shape (sample, class).")
        if not jnp.issubdtype(values.dtype, jnp.integer):
            raise TypeError("Equivalence-class counts must have an integer dtype.")
        if compatibility.case_shape or compatibility.target_shape != (
            int(values.shape[1]),
        ):
            raise ValueError(
                "compatibility must have one target row per equivalence class."
            )
        if lengths.shape != (compatibility.source_size,):
            raise ValueError("effective_lengths must contain one value per transcript.")
        self.counts = values.astype(jnp.int32)
        self.compatibility = compatibility
        self.effective_lengths = lengths


class TranscriptAbundanceEvidence(StrictModule):
    """Convergence and compatibility evidence for every sample estimate."""

    finite_positive_lengths: Array
    nonnegative_counts: Array
    class_has_compatible_transcript: Array
    unsupported_positive_class: Array
    library_fragments: Array
    iterations: Array
    residual: Array
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    replicate_unit: str = eqx.field(static=True)


class TranscriptAbundanceResult(StrictModule):
    """Transcript proportions and allocated fragment expectations by sample."""

    relative_abundance: Array
    expected_counts: Array
    valid: Array
    status: Array
    evidence: TranscriptAbundanceEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


class DifferentialUsageEvidence(StrictModule):
    """Replicate and sparse feature-relation evidence for a usage contrast."""

    first_group_samples: Array
    second_group_samples: Array
    gene_has_transcript: Array
    exchangeability_declared: Array
    replicate_unit: str = eqx.field(static=True)


class DifferentialTranscriptUsageResult(StrictModule):
    """Within-gene transcript usages and equal-sample two-group differences."""

    usage: Array
    mean_difference: Array
    standard_error: Array
    route_valid: Array
    valid: Array
    status: Array
    evidence: DifferentialUsageEvidence
    exchangeability: ExchangeabilityPlan | None
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def estimate_transcript_abundance(
    batch: TranscriptEquivalenceBatch,
    /,
    *,
    tolerance: float = 1e-6,
    maximum_iterations: int = 256,
    method_contract: BioinformaticsMethodContract | None = None,
) -> TranscriptAbundanceResult:
    """Estimate transcript abundance by bounded EM over equivalence classes."""
    if not isinstance(batch, TranscriptEquivalenceBatch):
        raise TypeError("batch must be TranscriptEquivalenceBatch.")
    tol = float(tolerance)
    maximum = int(maximum_iterations)
    if not math.isfinite(tol) or tol <= 0.0 or maximum < 1:
        raise ValueError("tolerance must be positive and maximum_iterations nonzero.")

    counts = batch.counts
    relation = batch.compatibility
    lengths = batch.effective_lengths
    samples, classes = counts.shape
    transcripts = relation.source_size
    safe_indices = jnp.where(relation.valid, relation.source_indices, 0)
    route_valid = relation.valid
    class_has_route = jnp.any(route_valid, axis=-1)
    unsupported = jnp.any((counts > 0) & ~class_has_route[None, :], axis=1)
    nonnegative_counts = jnp.all(counts >= 0, axis=1)
    lengths_valid = jnp.all(jnp.isfinite(lengths) & (lengths > 0.0))
    library_fragments = jnp.sum(jnp.maximum(counts, 0), axis=1, dtype=jnp.int32)
    estimable = (
        nonnegative_counts & lengths_valid & (library_fragments > 0) & ~unsupported
    )
    initial = jnp.full(
        (samples, transcripts),
        1.0 / transcripts,
        dtype=jnp.result_type(lengths, jnp.float32),
    )

    def em_step(theta: Array) -> tuple[Array, Array, Array]:
        route_weight = theta[:, safe_indices]
        route_weight = jnp.where(route_valid[None, :, :], route_weight, 0.0)
        class_weight = jnp.sum(route_weight, axis=-1)
        allocation = (
            counts[:, :, None]
            * route_weight
            / jnp.where(class_weight[:, :, None] > 0.0, class_weight[:, :, None], 1.0)
        )
        flattened_indices = safe_indices.reshape((-1,))
        flattened_valid = route_valid.reshape((-1,))

        def scatter_sample(sample_allocation: Array) -> Array:
            material = jnp.where(
                flattened_valid,
                sample_allocation.reshape((-1,)),
                0.0,
            )
            return (
                jnp.zeros((transcripts,), dtype=material.dtype)
                .at[flattened_indices]
                .add(material)
            )

        expected = jax.vmap(scatter_sample)(allocation)
        rate = expected / lengths[None, :]
        next_theta = rate / jnp.where(
            jnp.sum(rate, axis=1, keepdims=True) > 0.0,
            jnp.sum(rate, axis=1, keepdims=True),
            1.0,
        )
        next_theta = jnp.where(estimable[:, None], next_theta, theta)
        residual = jnp.max(jnp.abs(next_theta - theta), axis=1)
        return next_theta, expected, residual

    def body(
        iteration: int,
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        theta, expected, residual, first_converged, converged = state
        next_theta, next_expected, next_residual = em_step(theta)
        newly_converged = estimable & ~converged & (next_residual <= tol)
        first_converged = jnp.where(
            newly_converged,
            jnp.asarray(iteration + 1, dtype=jnp.int32),
            first_converged,
        )
        update = estimable & ~converged
        theta = jnp.where(update[:, None], next_theta, theta)
        expected = jnp.where(update[:, None], next_expected, expected)
        residual = jnp.where(update, next_residual, residual)
        return theta, expected, residual, first_converged, converged | newly_converged

    initial_state = (
        initial,
        jnp.zeros_like(initial),
        jnp.full((samples,), jnp.inf, dtype=initial.dtype),
        jnp.zeros((samples,), dtype=jnp.int32),
        ~estimable,
    )
    theta, expected, residual, iterations, converged = jax.lax.fori_loop(
        0, maximum, body, initial_state
    )
    iterations = jnp.where(
        estimable & ~converged,
        jnp.asarray(maximum, dtype=jnp.int32),
        iterations,
    )
    valid = estimable & converged
    status = jnp.where(
        ~nonnegative_counts | ~lengths_valid,
        TRANSCRIPT_INVALID_COUNTS,
        jnp.where(
            library_fragments == 0,
            TRANSCRIPT_EMPTY_LIBRARY,
            jnp.where(
                unsupported,
                TRANSCRIPT_UNSUPPORTED_EQUIVALENCE_CLASS,
                jnp.where(converged, TRANSCRIPT_SUCCESS, TRANSCRIPT_NONCONVERGED),
            ),
        ),
    ).astype(jnp.int32)
    evidence = TranscriptAbundanceEvidence(
        lengths_valid,
        nonnegative_counts,
        class_has_route,
        unsupported,
        library_fragments,
        iterations,
        residual,
        tol,
        maximum,
        "sample",
    )
    return TranscriptAbundanceResult(
        theta,
        expected,
        valid,
        status,
        evidence,
        method_contract if method_contract is not None else _abundance_contract(),
        "iterative_model_estimate",
    )


def differential_transcript_usage(
    abundance: TranscriptAbundanceResult,
    gene_to_transcript: RowRelation,
    second_group: ArrayLike,
    /,
    *,
    exchangeability: ExchangeabilityPlan | None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> DifferentialTranscriptUsageResult:
    """Form donor/sample-replicated within-gene usage contrast primitives."""
    if not isinstance(abundance, TranscriptAbundanceResult):
        raise TypeError("abundance must be TranscriptAbundanceResult.")
    if not isinstance(gene_to_transcript, RowRelation):
        raise TypeError("gene_to_transcript must be a RowRelation.")
    theta = abundance.relative_abundance
    if gene_to_transcript.case_shape or gene_to_transcript.source_size != theta.shape[1]:
        raise ValueError("gene_to_transcript must route the estimated transcript space.")
    group = jnp.asarray(second_group, dtype=bool)
    if group.shape != (theta.shape[0],):
        raise ValueError("second_group must contain one indicator per sample.")

    route_valid = gene_to_transcript.valid
    safe = jnp.where(route_valid, gene_to_transcript.source_indices, 0)
    routed = jnp.where(route_valid[None, :, :], theta[:, safe], 0.0)
    denominator = jnp.sum(routed, axis=-1, keepdims=True)
    usage = jnp.where(denominator > 0.0, routed / denominator, 0.0)
    first = abundance.valid & ~group
    second = abundance.valid & group
    first_count = jnp.sum(first, dtype=jnp.int32)
    second_count = jnp.sum(second, dtype=jnp.int32)
    first_mean = jnp.sum(
        jnp.where(first[:, None, None], usage, 0.0), axis=0
    ) / jnp.maximum(first_count, 1)
    second_mean = jnp.sum(
        jnp.where(second[:, None, None], usage, 0.0), axis=0
    ) / jnp.maximum(second_count, 1)
    first_residual = jnp.where(first[:, None, None], usage - first_mean, 0.0)
    second_residual = jnp.where(second[:, None, None], usage - second_mean, 0.0)
    first_variance = jnp.sum(first_residual**2, axis=0) / jnp.maximum(first_count - 1, 1)
    second_variance = jnp.sum(second_residual**2, axis=0) / jnp.maximum(
        second_count - 1, 1
    )
    standard_error = jnp.sqrt(
        first_variance / jnp.maximum(first_count, 1)
        + second_variance / jnp.maximum(second_count, 1)
    )
    gene_has_transcript = jnp.any(route_valid, axis=-1)
    exchangeability_declared = jnp.asarray(exchangeability is not None)
    complete_groups = (first_count > 0) & (second_count > 0)
    valid = complete_groups & exchangeability_declared & jnp.all(gene_has_transcript)
    status = jnp.where(
        ~exchangeability_declared,
        TRANSCRIPT_MISSING_EXCHANGEABILITY,
        jnp.where(complete_groups, TRANSCRIPT_SUCCESS, TRANSCRIPT_EMPTY_GROUP),
    ).astype(jnp.int32)
    evidence = DifferentialUsageEvidence(
        first_count,
        second_count,
        gene_has_transcript,
        exchangeability_declared,
        "sample",
    )
    return DifferentialTranscriptUsageResult(
        usage,
        second_mean - first_mean,
        standard_error,
        route_valid,
        valid,
        status,
        evidence,
        exchangeability,
        method_contract if method_contract is not None else _usage_contract(),
        "model_based_estimate",
    )


__all__ = [
    "TRANSCRIPT_EMPTY_GROUP",
    "TRANSCRIPT_EMPTY_LIBRARY",
    "TRANSCRIPT_INVALID_COUNTS",
    "TRANSCRIPT_MISSING_EXCHANGEABILITY",
    "TRANSCRIPT_NONCONVERGED",
    "TRANSCRIPT_SUCCESS",
    "TRANSCRIPT_UNSUPPORTED_EQUIVALENCE_CLASS",
    "DifferentialTranscriptUsageResult",
    "DifferentialUsageEvidence",
    "TranscriptAbundanceEvidence",
    "TranscriptAbundanceResult",
    "TranscriptEquivalenceBatch",
    "differential_transcript_usage",
    "estimate_transcript_abundance",
    "transcript_status_name",
]
