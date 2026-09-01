#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._cohort import GenotypeCohort


class PopulationSummaryStatus(IntEnum):
    SUCCESS = 0
    NO_OBSERVATIONS = 1
    MONOMORPHIC = 2
    NON_DIPLOID = 3
    UNPOLARIZED = 4
    CAPACITY_EXCEEDED = 5
    INSUFFICIENT_OVERLAP = 6
    NONFINITE = 7


def _contract(
    method_name: str, output_kind: OutputKind, /, *, exact: bool
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL if exact else MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        output_kind,
        conditioning_statement=(
            "Conditioned on normalized biallelic genotype posterior probabilities, "
            "declared ploidy, and the observed-data mask."
        ),
        truncation_statement="No scientific output is truncated.",
        capacity_semantics="Any finite output capacity is preflighted and failure is reported.",
        assumptions=("Variants are biallelic.",),
        nondifferentiable_outputs=("status", "valid"),
    )


class AlleleCountResult(StrictModule):
    alternate_count: Array
    reference_count: Array
    allele_number: Array
    alternate_frequency: Array
    missing_samples: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(PopulationSummaryStatus.SUCCESS))


class HardyWeinbergResult(StrictModule):
    genotype_counts: Array
    expected_counts: Array
    statistic: Array
    p_value: Array
    diploid_samples: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(PopulationSummaryStatus.SUCCESS))


class SiteFrequencySpectrumResult(StrictModule):
    spectrum: Array
    per_variant_count_distribution: Array
    allele_number: Array
    included: Array
    valid: Array
    status: Array
    evidence: Array
    folded: bool = eqx.field(static=True)
    polarized: bool = eqx.field(static=True)
    maximum_allele_number: int = eqx.field(static=True)
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(PopulationSummaryStatus.SUCCESS))


class LinkageDisequilibriumResult(StrictModule):
    correlation: Array
    r_squared: Array
    covariance: Array
    overlapping_samples: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(PopulationSummaryStatus.SUCCESS))


class KinshipResult(StrictModule):
    kinship: Array
    relationship: Array
    informative_variants: Array
    variant_weights: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(PopulationSummaryStatus.SUCCESS))


def allele_counts(cohort: GenotypeCohort, /) -> AlleleCountResult:
    """Posterior expected allele counts respecting ploidy and missingness."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    observed = cohort.observed
    dosage = jnp.where(observed, cohort.dosage, 0.0)
    allele_number = jnp.sum(jnp.where(observed, cohort.ploidy, 0), axis=1)
    alternate = jnp.sum(dosage, axis=1)
    reference = allele_number.astype(alternate.dtype) - alternate
    valid = allele_number > 0
    frequency = jnp.where(valid, alternate / jnp.maximum(allele_number, 1), jnp.nan)
    finite = jnp.isfinite(alternate) & jnp.isfinite(reference)
    valid = valid & finite
    status = jnp.where(
        ~finite,
        int(PopulationSummaryStatus.NONFINITE),
        jnp.where(
            allele_number == 0,
            int(PopulationSummaryStatus.NO_OBSERVATIONS),
            int(PopulationSummaryStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    missing = jnp.sum(~observed, axis=1).astype(jnp.int32)
    evidence = jnp.stack(
        (allele_number.astype(alternate.dtype), missing.astype(alternate.dtype)), axis=-1
    )
    return AlleleCountResult(
        alternate,
        reference,
        allele_number.astype(jnp.int32),
        frequency,
        missing,
        valid,
        status,
        evidence,
        _contract("posterior-allele-counts", OutputKind.STRUCTURED, exact=True),
    )


def hardy_weinberg(cohort: GenotypeCohort, /) -> HardyWeinbergResult:
    """Likelihood-aware Pearson HWE test over observed diploid samples only."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    if cohort.maximum_ploidy < 2:
        variant_count = cohort.variant_count
        zeros = jnp.zeros((variant_count, 3), dtype=cohort.genotype_probabilities.dtype)
        status = jnp.full(
            (variant_count,), int(PopulationSummaryStatus.NON_DIPLOID), dtype=jnp.int32
        )
        return HardyWeinbergResult(
            zeros,
            zeros,
            jnp.full((variant_count,), jnp.nan),
            jnp.full((variant_count,), jnp.nan),
            jnp.zeros((variant_count,), dtype=jnp.int32),
            jnp.zeros((variant_count,), dtype=bool),
            status,
            zeros,
            _contract("hardy-weinberg-pearson", OutputKind.STRUCTURED, exact=False),
        )
    selected = cohort.observed & (cohort.ploidy == 2)
    genotype_counts = jnp.sum(
        jnp.where(selected[..., None], cohort.genotype_probabilities[..., :3], 0.0),
        axis=1,
    )
    diploid_samples = jnp.sum(selected, axis=1).astype(jnp.int32)
    alternate = genotype_counts[:, 1] + 2.0 * genotype_counts[:, 2]
    denominator = jnp.maximum(2 * diploid_samples, 1)
    q = alternate / denominator
    p = 1.0 - q
    n = diploid_samples.astype(genotype_counts.dtype)
    expected = jnp.stack((n * p * p, 2.0 * n * p * q, n * q * q), axis=-1)
    terms = jnp.where(expected > 0.0, (genotype_counts - expected) ** 2 / expected, 0.0)
    statistic = jnp.sum(terms, axis=-1)
    p_value = jsp.special.erfc(jnp.sqrt(jnp.maximum(statistic, 0.0) / 2.0))
    monomorphic = (alternate <= 0.0) | (alternate >= 2.0 * n)
    finite = jnp.isfinite(statistic) & jnp.isfinite(p_value)
    valid = (diploid_samples > 0) & ~monomorphic & finite
    status = jnp.where(
        diploid_samples == 0,
        int(PopulationSummaryStatus.NON_DIPLOID),
        jnp.where(
            monomorphic,
            int(PopulationSummaryStatus.MONOMORPHIC),
            jnp.where(
                finite,
                int(PopulationSummaryStatus.SUCCESS),
                int(PopulationSummaryStatus.NONFINITE),
            ),
        ),
    ).astype(jnp.int32)
    return HardyWeinbergResult(
        genotype_counts,
        expected,
        jnp.where(valid, statistic, jnp.nan),
        jnp.where(valid, p_value, jnp.nan),
        diploid_samples,
        valid,
        status,
        genotype_counts,
        _contract("hardy-weinberg-pearson", OutputKind.STRUCTURED, exact=False),
    )


def site_frequency_spectrum(
    cohort: GenotypeCohort,
    /,
    *,
    folded: bool = False,
    maximum_allele_number: int | None = None,
) -> SiteFrequencySpectrumResult:
    """Exact posterior SFS by bounded convolution of genotype dosage laws.

    Unfolded spectra require an explicitly known ancestral allele at every included
    site. Folded spectra do not use polarization. Missing samples contribute no
    chromosomes rather than their canonical uninformative posterior.
    """
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    if not isinstance(folded, bool):
        raise TypeError("folded must be a bool.")
    observed_ploidy = np.where(np.asarray(cohort.observed), np.asarray(cohort.ploidy), 0)
    required = int(observed_ploidy.sum(axis=1).max(initial=0))
    capacity = required if maximum_allele_number is None else int(maximum_allele_number)
    if capacity < 1:
        raise ValueError("maximum_allele_number must be positive.")
    variant_count = cohort.variant_count
    dtype = cohort.genotype_probabilities.dtype
    allele_number = jnp.sum(jnp.where(cohort.observed, cohort.ploidy, 0), axis=1).astype(
        jnp.int32
    )
    if required > capacity:
        status = jnp.asarray(
            int(PopulationSummaryStatus.CAPACITY_EXCEEDED), dtype=jnp.int32
        )
        return SiteFrequencySpectrumResult(
            jnp.zeros((capacity + 1,), dtype=dtype),
            jnp.zeros((variant_count, capacity + 1), dtype=dtype),
            allele_number,
            jnp.zeros((variant_count,), dtype=bool),
            jnp.asarray(False),
            status,
            jnp.asarray((required, capacity), dtype=jnp.int32),
            folded,
            not folded,
            capacity,
            _contract(
                "posterior-site-frequency-spectrum", OutputKind.STRUCTURED, exact=True
            ),
        )

    distributions: list[Array] = []
    states = cohort.maximum_ploidy + 1
    for variant in range(variant_count):
        distribution = jnp.zeros((capacity + 1,), dtype=dtype).at[0].set(1.0)
        for sample in range(cohort.sample_count):
            posterior = jnp.where(
                cohort.observed[variant, sample],
                cohort.genotype_probabilities[variant, sample],
                jnp.zeros((states,), dtype=dtype).at[0].set(1.0),
            )
            convolved = jnp.zeros_like(distribution)
            for dosage in range(min(states, capacity + 1)):
                convolved = convolved.at[dosage:].add(
                    distribution[: capacity + 1 - dosage] * posterior[dosage]
                )
            distribution = convolved
        total = allele_number[variant]
        if folded:
            transformed = jnp.zeros_like(distribution)
            for count in range(capacity + 1):
                minor = jnp.minimum(count, total - count)
                valid_count = (count <= total) & (minor >= 0)
                transformed = transformed.at[jnp.clip(minor, 0, capacity)].add(
                    jnp.where(valid_count, distribution[count], 0.0)
                )
            distribution = transformed
        else:
            transformed = jnp.zeros_like(distribution)
            for count in range(capacity + 1):
                derived = jnp.where(
                    cohort.ancestral_is_alternate[variant], total - count, count
                )
                valid_count = (count <= total) & (derived >= 0)
                transformed = transformed.at[jnp.clip(derived, 0, capacity)].add(
                    jnp.where(valid_count, distribution[count], 0.0)
                )
            distribution = transformed
        distributions.append(distribution)
    per_variant = jnp.stack(distributions, axis=0)
    has_data = allele_number > 0
    polarized = folded | cohort.polarization_known
    included = has_data & polarized
    spectrum = jnp.sum(jnp.where(included[:, None], per_variant, 0.0), axis=0)
    all_polarized = jnp.all(polarized | ~has_data)
    valid = jnp.any(included) & all_polarized
    status = jnp.where(
        ~all_polarized,
        int(PopulationSummaryStatus.UNPOLARIZED),
        jnp.where(
            jnp.any(included),
            int(PopulationSummaryStatus.SUCCESS),
            int(PopulationSummaryStatus.NO_OBSERVATIONS),
        ),
    ).astype(jnp.int32)
    return SiteFrequencySpectrumResult(
        spectrum,
        per_variant,
        allele_number,
        included,
        valid,
        status,
        jnp.asarray((required, capacity), dtype=jnp.int32),
        folded,
        bool(np.all(np.asarray(cohort.polarization_known))) if not folded else False,
        capacity,
        _contract("posterior-site-frequency-spectrum", OutputKind.STRUCTURED, exact=True),
    )


def linkage_disequilibrium(cohort: GenotypeCohort, /) -> LinkageDisequilibriumResult:
    """Pairwise posterior dosage correlation over jointly observed samples."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    dosage = cohort.dosage
    observed = cohort.observed
    weights = observed[:, None, :] & observed[None, :, :]
    overlap = jnp.sum(weights, axis=-1).astype(jnp.int32)
    denominator = jnp.maximum(overlap, 1).astype(dosage.dtype)
    left_mean = (
        jnp.sum(jnp.where(weights, dosage[:, None, :], 0.0), axis=-1) / denominator
    )
    right_mean = (
        jnp.sum(jnp.where(weights, dosage[None, :, :], 0.0), axis=-1) / denominator
    )
    centered_left = dosage[:, None, :] - left_mean[..., None]
    centered_right = dosage[None, :, :] - right_mean[..., None]
    covariance = (
        jnp.sum(jnp.where(weights, centered_left * centered_right, 0.0), axis=-1)
        / denominator
    )
    left_variance = (
        jnp.sum(jnp.where(weights, centered_left * centered_left, 0.0), axis=-1)
        / denominator
    )
    right_variance = (
        jnp.sum(jnp.where(weights, centered_right * centered_right, 0.0), axis=-1)
        / denominator
    )
    scale = jnp.sqrt(jnp.maximum(left_variance * right_variance, 0.0))
    valid = (overlap >= 2) & (scale > 0.0) & jnp.isfinite(covariance)
    correlation = jnp.where(valid, covariance / scale, jnp.nan)
    status = jnp.where(
        overlap < 2,
        int(PopulationSummaryStatus.INSUFFICIENT_OVERLAP),
        jnp.where(
            scale <= 0.0,
            int(PopulationSummaryStatus.MONOMORPHIC),
            jnp.where(
                jnp.isfinite(covariance),
                int(PopulationSummaryStatus.SUCCESS),
                int(PopulationSummaryStatus.NONFINITE),
            ),
        ),
    ).astype(jnp.int32)
    return LinkageDisequilibriumResult(
        correlation,
        correlation * correlation,
        jnp.where(valid, covariance, jnp.nan),
        overlap,
        valid,
        status,
        jnp.stack((overlap, valid.astype(jnp.int32)), axis=-1),
        _contract(
            "posterior-dosage-linkage-disequilibrium", OutputKind.STRUCTURED, exact=False
        ),
    )


def genomic_kinship(cohort: GenotypeCohort, /) -> KinshipResult:
    """Posterior-mean standardized genomic relationship and kinship matrices."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    counts = allele_counts(cohort)
    frequency = jnp.where(counts.valid, counts.alternate_frequency, 0.0)
    mean_ploidy = jnp.sum(
        jnp.where(cohort.observed, cohort.ploidy, 0), axis=1
    ) / jnp.maximum(jnp.sum(cohort.observed, axis=1), 1)
    expected = mean_ploidy[:, None] * frequency[:, None]
    variance = mean_ploidy * frequency * (1.0 - frequency)
    informative = counts.valid & (variance > 0.0)
    imputed_dosage = jnp.where(cohort.observed, cohort.dosage, expected)
    standardized = jnp.where(
        informative[:, None],
        (imputed_dosage - expected) / jnp.sqrt(jnp.maximum(variance[:, None], 1e-30)),
        0.0,
    )
    informative_count = jnp.sum(informative).astype(jnp.int32)
    relationship = (jnp.swapaxes(standardized, 0, 1) @ standardized) / jnp.maximum(
        informative_count, 1
    )
    relationship = 0.5 * (relationship + jnp.swapaxes(relationship, -1, -2))
    valid = (informative_count > 0) & jnp.all(jnp.isfinite(relationship))
    status = jnp.where(
        informative_count == 0,
        int(PopulationSummaryStatus.MONOMORPHIC),
        jnp.where(
            jnp.all(jnp.isfinite(relationship)),
            int(PopulationSummaryStatus.SUCCESS),
            int(PopulationSummaryStatus.NONFINITE),
        ),
    ).astype(jnp.int32)
    return KinshipResult(
        0.5 * relationship,
        relationship,
        informative_count,
        informative.astype(relationship.dtype),
        valid,
        status,
        jnp.asarray((informative_count, cohort.variant_count), dtype=jnp.int32),
        _contract("posterior-genomic-kinship", OutputKind.STRUCTURED, exact=False),
    )


__all__ = [
    "AlleleCountResult",
    "HardyWeinbergResult",
    "KinshipResult",
    "LinkageDisequilibriumResult",
    "PopulationSummaryStatus",
    "SiteFrequencySpectrumResult",
    "allele_counts",
    "genomic_kinship",
    "hardy_weinberg",
    "linkage_disequilibrium",
    "site_frequency_spectrum",
]
