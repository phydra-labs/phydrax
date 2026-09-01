#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class GenotypeCohort(StrictModule):
    """Biallelic genotype posteriors with explicit mixed ploidy and missingness.

    The final probability axis enumerates alternate-allele dosage from zero through
    ``maximum_ploidy``. Entries above a sample/site's declared ploidy are exactly
    zero. A missing observation has ``observed == False`` and a normalized uniform
    posterior over the admissible dosage states; it is consequently excluded from
    observed-data summaries while remaining usable by imputation methods.
    """

    genotype_probabilities: Array
    ploidy: Array
    observed: Array
    positions: Array
    chromosome_index: Array
    ancestral_is_alternate: Array
    polarization_known: Array
    sample_ids: tuple[str, ...] = eqx.field(static=True)
    chromosome_labels: tuple[str, ...] = eqx.field(static=True)
    reference_alleles: tuple[str, ...] = eqx.field(static=True)
    alternate_alleles: tuple[str, ...] = eqx.field(static=True)
    maximum_ploidy: int = eqx.field(static=True)
    cohort_id: str = eqx.field(static=True)

    def __init__(
        self,
        genotype_probabilities: ArrayLike,
        ploidy: ArrayLike,
        observed: ArrayLike,
        positions: ArrayLike,
        chromosome_index: ArrayLike,
        /,
        *,
        sample_ids: Sequence[str],
        chromosome_labels: Sequence[str],
        reference_alleles: Sequence[str] | None = None,
        alternate_alleles: Sequence[str] | None = None,
        ancestral_is_alternate: ArrayLike | None = None,
        polarization_known: ArrayLike | None = None,
        probability_tolerance: float = 1e-6,
    ):
        probabilities = jnp.asarray(genotype_probabilities)
        if probabilities.ndim != 3:
            raise ValueError(
                "genotype_probabilities must have variant, sample, and dosage axes."
            )
        if jnp.iscomplexobj(probabilities):
            raise TypeError("genotype probabilities must be real-valued.")
        if not jnp.issubdtype(probabilities.dtype, jnp.inexact):
            probabilities = probabilities.astype(float)
        variant_count, sample_count, dosage_count = probabilities.shape
        maximum_ploidy = int(dosage_count) - 1
        if maximum_ploidy < 1:
            raise ValueError("At least haploid genotype states are required.")

        ploidy_ = jnp.asarray(ploidy)
        if ploidy_.shape != (variant_count, sample_count):
            raise ValueError("ploidy must have shape (variants, samples).")
        if not jnp.issubdtype(ploidy_.dtype, jnp.integer):
            raise TypeError("ploidy must contain integers.")
        observed_ = jnp.asarray(observed, dtype=bool)
        if observed_.shape != (variant_count, sample_count):
            raise ValueError("observed must have shape (variants, samples).")

        positions_ = jnp.asarray(positions)
        chromosome_index_ = jnp.asarray(chromosome_index)
        if positions_.shape != (variant_count,):
            raise ValueError("positions must contain one coordinate per variant.")
        if chromosome_index_.shape != (variant_count,):
            raise ValueError("chromosome_index must contain one value per variant.")
        if not jnp.issubdtype(chromosome_index_.dtype, jnp.integer):
            raise TypeError("chromosome_index must contain integers.")
        if jnp.iscomplexobj(positions_):
            raise TypeError("positions must be real-valued.")
        if not jnp.issubdtype(positions_.dtype, jnp.inexact):
            positions_ = positions_.astype(float)

        samples = tuple(str(value) for value in sample_ids)
        chromosomes = tuple(str(value) for value in chromosome_labels)
        if (
            len(samples) != sample_count
            or any(not value for value in samples)
            or len(set(samples)) != len(samples)
        ):
            raise ValueError(
                "sample_ids must contain one unique non-empty ID per sample."
            )
        if not chromosomes or any(not value for value in chromosomes):
            raise ValueError("chromosome_labels must be non-empty strings.")
        if len(set(chromosomes)) != len(chromosomes):
            raise ValueError("chromosome_labels must be unique.")

        references = (
            tuple("N" for _ in range(variant_count))
            if reference_alleles is None
            else tuple(str(value) for value in reference_alleles)
        )
        alternates = (
            tuple("<ALT>" for _ in range(variant_count))
            if alternate_alleles is None
            else tuple(str(value) for value in alternate_alleles)
        )
        if len(references) != variant_count or len(alternates) != variant_count:
            raise ValueError("Allele labels must contain one value per variant.")
        if any(not value for value in references + alternates):
            raise ValueError("Allele labels must be non-empty.")
        if any(left == right for left, right in zip(references, alternates, strict=True)):
            raise ValueError(
                "Reference and alternate alleles must differ at every variant."
            )

        if ancestral_is_alternate is None:
            if polarization_known is not None:
                raise ValueError(
                    "polarization_known requires ancestral_is_alternate values."
                )
            polarized = jnp.zeros((variant_count,), dtype=bool)
            polarization_known_ = jnp.zeros((variant_count,), dtype=bool)
        else:
            polarized = jnp.asarray(ancestral_is_alternate, dtype=bool)
            polarization_known_ = (
                jnp.ones((variant_count,), dtype=bool)
                if polarization_known is None
                else jnp.asarray(polarization_known, dtype=bool)
            )
        if polarized.shape != (variant_count,) or polarization_known_.shape != (
            variant_count,
        ):
            raise ValueError(
                "Ancestral allele and polarization masks must have shape (variants,)."
            )

        host_ploidy = np.asarray(ploidy_)
        host_probabilities = np.asarray(probabilities)
        host_positions = np.asarray(positions_)
        host_chromosomes = np.asarray(chromosome_index_)
        host_observed = np.asarray(observed_)
        tolerance = float(probability_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("probability_tolerance must be finite and non-negative.")
        if np.any(host_ploidy < 1) or np.any(host_ploidy > maximum_ploidy):
            raise ValueError("Every ploidy must be between one and maximum_ploidy.")
        if not np.all(np.isfinite(host_probabilities)) or np.any(
            host_probabilities < -tolerance
        ):
            raise ValueError("Genotype probabilities must be finite and non-negative.")
        dosage = np.arange(dosage_count)[None, None, :]
        admissible = dosage <= host_ploidy[..., None]
        if np.any(np.where(admissible, 0.0, np.abs(host_probabilities)) > tolerance):
            raise ValueError("Probability above the declared ploidy must be zero.")
        sums = host_probabilities.sum(axis=-1)
        if not np.all(np.abs(sums - 1.0) <= tolerance):
            raise ValueError("Every genotype posterior must sum to one.")
        if not np.all(np.isfinite(host_positions)) or np.any(host_positions < 0.0):
            raise ValueError("Variant positions must be finite and non-negative.")
        if np.any(host_chromosomes < 0) or np.any(host_chromosomes >= len(chromosomes)):
            raise ValueError("chromosome_index refers outside chromosome_labels.")
        for chromosome in range(len(chromosomes)):
            selected = host_positions[host_chromosomes == chromosome]
            if selected.size and np.any(selected[1:] < selected[:-1]):
                raise ValueError(
                    "Positions must be nondecreasing within each chromosome."
                )

        # Missing values have no observed genotype evidence. Canonicalizing them to
        # a uniform admissible law prevents a caller from smuggling arbitrary values
        # into downstream imputation while preserving the declared ploidy.
        uniform = admissible / (host_ploidy[..., None] + 1)
        canonical = np.where(host_observed[..., None], host_probabilities, uniform)
        canonical = np.where(admissible, canonical, 0.0)

        self.genotype_probabilities = jnp.asarray(canonical, dtype=probabilities.dtype)
        self.ploidy = ploidy_.astype(jnp.int32)
        self.observed = observed_
        self.positions = positions_
        self.chromosome_index = chromosome_index_.astype(jnp.int32)
        self.ancestral_is_alternate = polarized
        self.polarization_known = polarization_known_
        self.sample_ids = samples
        self.chromosome_labels = chromosomes
        self.reference_alleles = references
        self.alternate_alleles = alternates
        self.maximum_ploidy = maximum_ploidy
        self.cohort_id = canonical_fingerprint(
            {
                "kind": "genotype-cohort",
                "sample_ids": list(samples),
                "chromosome_labels": list(chromosomes),
                "reference_alleles": list(references),
                "alternate_alleles": list(alternates),
                "maximum_ploidy": maximum_ploidy,
                "variant_count": variant_count,
                "sample_count": sample_count,
                "positions": host_positions.tolist(),
                "chromosome_index": host_chromosomes.tolist(),
                "ploidy": host_ploidy.tolist(),
                "observed": host_observed.tolist(),
                "genotype_probabilities": canonical.tolist(),
                "ancestral_is_alternate": np.asarray(polarized).tolist(),
                "polarization_known": np.asarray(polarization_known_).tolist(),
            }
        )

    @classmethod
    def from_calls(
        cls,
        calls: ArrayLike,
        ploidy: ArrayLike,
        positions: ArrayLike,
        chromosome_index: ArrayLike,
        /,
        *,
        sample_ids: Sequence[str],
        chromosome_labels: Sequence[str],
        reference_alleles: Sequence[str] | None = None,
        alternate_alleles: Sequence[str] | None = None,
        ancestral_is_alternate: ArrayLike | None = None,
        missing_value: int = -1,
        polarization_known: ArrayLike | None = None,
        maximum_ploidy: int | None = None,
    ) -> GenotypeCohort:
        """Construct degenerate posteriors from dosage calls and a missing sentinel."""
        calls_ = jnp.asarray(calls)
        ploidy_ = jnp.asarray(ploidy)
        if calls_.ndim != 2 or ploidy_.shape != calls_.shape:
            raise ValueError("calls and ploidy must share shape (variants, samples).")
        if not jnp.issubdtype(calls_.dtype, jnp.integer):
            raise TypeError("calls must contain integer alternate-allele dosages.")
        if not jnp.issubdtype(ploidy_.dtype, jnp.integer):
            raise TypeError("ploidy must contain integers.")
        resolved_maximum = (
            int(np.asarray(ploidy_).max())
            if maximum_ploidy is None
            else int(maximum_ploidy)
        )
        if resolved_maximum < 1:
            raise ValueError("maximum_ploidy must be positive.")
        host_calls = np.asarray(calls_)
        host_ploidy = np.asarray(ploidy_)
        observed = host_calls != int(missing_value)
        if np.any(observed & ((host_calls < 0) | (host_calls > host_ploidy))):
            raise ValueError("Observed dosage calls must lie between zero and ploidy.")
        probabilities = np.zeros(calls_.shape + (resolved_maximum + 1,), dtype=float)
        rows, columns = np.indices(calls_.shape)
        safe_calls = np.where(observed, host_calls, 0)
        probabilities[rows, columns, safe_calls] = 1.0
        admissible = (
            np.arange(resolved_maximum + 1)[None, None, :] <= host_ploidy[..., None]
        )
        uniform = admissible / (host_ploidy[..., None] + 1)
        probabilities = np.where(observed[..., None], probabilities, uniform)
        return cls(
            probabilities,
            ploidy_,
            observed,
            positions,
            chromosome_index,
            sample_ids=sample_ids,
            chromosome_labels=chromosome_labels,
            reference_alleles=reference_alleles,
            alternate_alleles=alternate_alleles,
            ancestral_is_alternate=ancestral_is_alternate,
            polarization_known=polarization_known,
        )

    @classmethod
    def from_log_likelihoods(
        cls,
        genotype_log_likelihoods: ArrayLike,
        ploidy: ArrayLike,
        observed: ArrayLike,
        positions: ArrayLike,
        chromosome_index: ArrayLike,
        /,
        *,
        sample_ids: Sequence[str],
        chromosome_labels: Sequence[str],
        reference_alleles: Sequence[str] | None = None,
        alternate_alleles: Sequence[str] | None = None,
        ancestral_is_alternate: ArrayLike | None = None,
        polarization_known: ArrayLike | None = None,
    ) -> GenotypeCohort:
        """Normalize real log genotype likelihoods over each admissible dosage set."""
        log_likelihood = jnp.asarray(genotype_log_likelihoods)
        ploidy_ = jnp.asarray(ploidy)
        observed_ = jnp.asarray(observed, dtype=bool)
        if log_likelihood.ndim != 3:
            raise ValueError(
                "genotype_log_likelihoods must have variant, sample, and dosage axes."
            )
        if ploidy_.shape != log_likelihood.shape[:2] or observed_.shape != ploidy_.shape:
            raise ValueError(
                "ploidy and observed must match the likelihood variant/sample axes."
            )
        if jnp.iscomplexobj(log_likelihood):
            raise TypeError("Genotype log likelihoods must be real-valued.")
        if not jnp.issubdtype(log_likelihood.dtype, jnp.inexact):
            log_likelihood = log_likelihood.astype(float)
        maximum_ploidy = int(log_likelihood.shape[-1]) - 1
        host = np.asarray(log_likelihood)
        host_ploidy = np.asarray(ploidy_)
        host_observed = np.asarray(observed_)
        if np.any(np.isnan(host)) or np.any(np.isposinf(host)):
            raise ValueError(
                "Genotype log likelihoods may contain finite values and -inf only."
            )
        admissible = (
            np.arange(maximum_ploidy + 1)[None, None, :] <= host_ploidy[..., None]
        )
        if np.any(host_observed & ~np.any(np.isfinite(host) & admissible, axis=-1)):
            raise ValueError(
                "Every observed genotype needs a finite admissible log likelihood."
            )
        masked = jnp.where(jnp.asarray(admissible), log_likelihood, -jnp.inf)
        log_normalizer = jsp.special.logsumexp(masked, axis=-1, keepdims=True)
        probabilities = jnp.exp(masked - log_normalizer)
        probabilities = jnp.where(
            observed_[..., None],
            probabilities,
            jnp.asarray(admissible, dtype=probabilities.dtype) / (ploidy_[..., None] + 1),
        )
        return cls(
            probabilities,
            ploidy_,
            observed_,
            positions,
            chromosome_index,
            sample_ids=sample_ids,
            chromosome_labels=chromosome_labels,
            reference_alleles=reference_alleles,
            alternate_alleles=alternate_alleles,
            ancestral_is_alternate=ancestral_is_alternate,
            polarization_known=polarization_known,
        )

    @property
    def variant_count(self) -> int:
        return int(self.genotype_probabilities.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.genotype_probabilities.shape[1])

    @property
    def dosage(self) -> Array:
        states = jnp.arange(
            self.maximum_ploidy + 1, dtype=self.genotype_probabilities.dtype
        )
        admissible = states[None, None, :] <= self.ploidy[..., None]
        probabilities = jnp.where(admissible, self.genotype_probabilities, 0.0)
        return jnp.sum(probabilities * states, axis=-1)

    @property
    def dosage_variance(self) -> Array:
        states = jnp.arange(
            self.maximum_ploidy + 1, dtype=self.genotype_probabilities.dtype
        )
        admissible = states[None, None, :] <= self.ploidy[..., None]
        probabilities = jnp.where(admissible, self.genotype_probabilities, 0.0)
        mean = self.dosage
        second = jnp.sum(probabilities * states * states, axis=-1)
        return jnp.maximum(second - mean * mean, 0.0)

    @property
    def hard_calls(self) -> Array:
        calls = jnp.argmax(self.genotype_probabilities, axis=-1).astype(jnp.int32)
        return jnp.where(self.observed, calls, -jnp.ones((), dtype=jnp.int32))

    def variants_on_chromosome(self, chromosome: int | str, /) -> Array:
        """Return a boolean variant mask for a chromosome index or label."""
        if isinstance(chromosome, str):
            if chromosome not in self.chromosome_labels:
                raise KeyError(chromosome)
            index = self.chromosome_labels.index(chromosome)
        else:
            index = int(chromosome)
            if index < 0 or index >= len(self.chromosome_labels):
                raise IndexError("chromosome index is outside chromosome_labels.")
        return self.chromosome_index == index


__all__ = ["GenotypeCohort"]
