#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._cohort import GenotypeCohort
from ._recombination import _finite_hmm, RecombinationMap


class ImputationStatus(IntEnum):
    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    PANEL_MISMATCH = 2
    DEGENERATE_LIKELIHOOD = 3
    NONFINITE = 4


class ReferenceHaplotypePanel(StrictModule):
    alleles: Array
    positions: Array
    chromosome_index: Array

    def __init__(
        self,
        alleles: ArrayLike,
        positions: ArrayLike,
        chromosome_index: ArrayLike,
        /,
    ):
        alleles_ = jnp.asarray(alleles)
        positions_ = jnp.asarray(positions)
        chromosome_ = jnp.asarray(chromosome_index)
        if alleles_.ndim != 2:
            raise ValueError("alleles must have shape (reference haplotypes, variants).")
        haplotype_count, variant_count = alleles_.shape
        if haplotype_count < 1 or variant_count < 1:
            raise ValueError(
                "Reference panels require at least one haplotype and variant."
            )
        if not jnp.issubdtype(alleles_.dtype, jnp.integer):
            raise TypeError("Reference haplotype alleles must be integer coded.")
        if positions_.shape != (variant_count,) or chromosome_.shape != (variant_count,):
            raise ValueError("Panel coordinates must contain one value per variant.")
        if not jnp.issubdtype(chromosome_.dtype, jnp.integer):
            raise TypeError("chromosome_index must contain integers.")
        host_alleles = np.asarray(alleles_)
        host_positions = np.asarray(positions_)
        host_chromosome = np.asarray(chromosome_)
        if np.any((host_alleles < 0) | (host_alleles > 1)):
            raise ValueError(
                "Reference panel alleles must be biallelic codes zero or one."
            )
        if not np.all(np.isfinite(host_positions)):
            raise ValueError("Panel positions must be finite.")
        for chromosome in np.unique(host_chromosome):
            selected = host_positions[host_chromosome == chromosome]
            if np.any(np.diff(selected) <= 0.0):
                raise ValueError("Panel positions must increase within a chromosome.")
        self.alleles = alleles_.astype(jnp.int32)
        self.positions = positions_
        self.chromosome_index = chromosome_.astype(jnp.int32)

    @property
    def haplotype_count(self) -> int:
        return int(self.alleles.shape[0])

    @property
    def variant_count(self) -> int:
        return int(self.alleles.shape[1])


class ImputationPlan(StrictModule):
    maximum_copying_states: int = eqx.field(static=True)
    mismatch_probability: float = eqx.field(static=True)
    recombination_scale: float = eqx.field(static=True)

    def __init__(
        self,
        maximum_copying_states: int,
        /,
        *,
        mismatch_probability: float = 1e-3,
        recombination_scale: float = 1.0,
    ):
        capacity = int(maximum_copying_states)
        mismatch = float(mismatch_probability)
        scale = float(recombination_scale)
        if capacity < 1:
            raise ValueError("maximum_copying_states must be positive.")
        if not np.isfinite(mismatch) or mismatch <= 0.0 or mismatch >= 1.0:
            raise ValueError(
                "mismatch_probability must lie strictly between zero and one."
            )
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("recombination_scale must be finite and non-negative.")
        self.maximum_copying_states = capacity
        self.mismatch_probability = mismatch
        self.recombination_scale = scale


class GenotypeImputationResult(StrictModule):
    genotype_probabilities: Array
    dosage: Array
    dosage_variance: Array
    information: Array
    copying_path: Array
    log_likelihood: Array
    required_copying_states: Array
    valid: Array
    status: Array
    evidence: Array
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(ImputationStatus.SUCCESS))


def _imputation_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "li-stephens-genotype-imputation",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.EXACT_AD,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Conditioned on a fixed biallelic reference haplotype panel, recombination "
            "map, target genotype likelihoods, ploidy, and observed-data mask."
        ),
        truncation_statement="Copying states are never pruned or truncated.",
        capacity_semantics=(
            "The full ordered haplotype copying-state count H^P is preflighted for "
            "every sample; any excess returns CAPACITY_EXCEEDED before recursion."
        ),
        assumptions=(
            "Li-Stephens first-order copying approximation.",
            "Reference and target variants have identical order and allele orientation.",
        ),
        nondifferentiable_outputs=("copying_path", "status", "valid"),
    )


def _copying_states(haplotype_count: int, ploidy: int, /) -> np.ndarray:
    return np.asarray(
        tuple(product(range(haplotype_count), repeat=ploidy)), dtype=np.int32
    )


def impute_genotypes(
    cohort: GenotypeCohort,
    panel: ReferenceHaplotypePanel,
    recombination_map: RecombinationMap,
    plan: ImputationPlan,
    /,
) -> GenotypeImputationResult:
    """Impute mixed-ploidy genotype posteriors with a bounded copying HMM.

    Ordered copying tuples retain the exact product transition law for every active
    chromosome copy. At sites of lower ploidy, inactive tuple coordinates contribute
    neither dosage nor emission evidence; this supports PAR/non-PAR and sex-chromosome
    ploidy changes without altering the fixed state capacity of a sample's chain.
    """
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    if not isinstance(panel, ReferenceHaplotypePanel):
        raise TypeError("panel must be a ReferenceHaplotypePanel.")
    if not isinstance(recombination_map, RecombinationMap):
        raise TypeError("recombination_map must be a RecombinationMap.")
    if not isinstance(plan, ImputationPlan):
        raise TypeError("plan must be an ImputationPlan.")
    aligned = (
        cohort.variant_count == panel.variant_count == recombination_map.variant_count
        and np.array_equal(np.asarray(cohort.positions), np.asarray(panel.positions))
        and np.array_equal(
            np.asarray(cohort.chromosome_index), np.asarray(panel.chromosome_index)
        )
        and np.array_equal(
            np.asarray(cohort.positions), np.asarray(recombination_map.positions)
        )
        and np.array_equal(
            np.asarray(cohort.chromosome_index),
            np.asarray(recombination_map.chromosome_index),
        )
    )
    sample_ploidy = np.asarray(cohort.ploidy).max(axis=0)
    required = np.asarray(
        [panel.haplotype_count ** int(value) for value in sample_ploidy], dtype=np.int32
    )
    shape = (
        cohort.variant_count,
        cohort.sample_count,
        cohort.maximum_ploidy + 1,
    )
    dtype = cohort.genotype_probabilities.dtype
    if not aligned:
        return GenotypeImputationResult(
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape[:2], dtype=dtype),
            jnp.zeros(shape[:2], dtype=dtype),
            jnp.zeros(shape[:2], dtype=dtype),
            jnp.full(
                (cohort.sample_count, cohort.variant_count, cohort.maximum_ploidy),
                -1,
                dtype=jnp.int32,
            ),
            jnp.full((cohort.sample_count,), -jnp.inf, dtype=dtype),
            jnp.asarray(required),
            jnp.zeros((cohort.sample_count,), dtype=bool),
            jnp.full(
                (cohort.sample_count,),
                int(ImputationStatus.PANEL_MISMATCH),
                dtype=jnp.int32,
            ),
            jnp.stack(
                (
                    jnp.asarray(required),
                    jnp.full(required.shape, plan.maximum_copying_states),
                ),
                axis=-1,
            ),
            _imputation_contract(),
        )
    if required.max(initial=0) > plan.maximum_copying_states:
        return GenotypeImputationResult(
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape[:2], dtype=dtype),
            jnp.zeros(shape[:2], dtype=dtype),
            jnp.zeros(shape[:2], dtype=dtype),
            jnp.full(
                (cohort.sample_count, cohort.variant_count, cohort.maximum_ploidy),
                -1,
                dtype=jnp.int32,
            ),
            jnp.full((cohort.sample_count,), -jnp.inf, dtype=dtype),
            jnp.asarray(required),
            jnp.zeros((cohort.sample_count,), dtype=bool),
            jnp.full(
                (cohort.sample_count,),
                int(ImputationStatus.CAPACITY_EXCEEDED),
                dtype=jnp.int32,
            ),
            jnp.stack(
                (
                    jnp.asarray(required),
                    jnp.full(required.shape, plan.maximum_copying_states),
                ),
                axis=-1,
            ),
            _imputation_contract(),
        )

    recombination = recombination_map.haldane_recombination_fraction(
        scale=plan.recombination_scale
    )
    recombination = jnp.where(
        recombination_map.chromosome_index[1:] != recombination_map.chromosome_index[:-1],
        1.0,
        recombination,
    )
    haplotypes = panel.alleles
    probability_outputs: list[Array] = []
    dosage_outputs: list[Array] = []
    variance_outputs: list[Array] = []
    information_outputs: list[Array] = []
    path_outputs: list[Array] = []
    log_likelihoods: list[Array] = []
    valid_outputs: list[Array] = []
    status_outputs: list[Array] = []
    for sample in range(cohort.sample_count):
        maximum_ploidy = int(sample_ploidy[sample])
        states_np = _copying_states(panel.haplotype_count, maximum_ploidy)
        states = jnp.asarray(states_np)
        state_count = int(states.shape[0])
        prior = jnp.full((state_count,), 1.0 / state_count, dtype=dtype)
        single = (1.0 - recombination[:, None, None]) * jnp.eye(
            panel.haplotype_count, dtype=dtype
        )[None, :, :] + recombination[:, None, None] / panel.haplotype_count
        transitions = jnp.ones(
            (cohort.variant_count - 1, state_count, state_count), dtype=dtype
        )
        for copy in range(maximum_ploidy):
            transitions = (
                transitions * single[:, states[:, copy, None], states[None, :, copy]]
            )

        state_dosage_rows: list[Array] = []
        for variant in range(cohort.variant_count):
            active_ploidy = cohort.ploidy[variant, sample]
            dosage = jnp.zeros((state_count,), dtype=jnp.int32)
            for copy in range(maximum_ploidy):
                dosage = dosage + jnp.where(
                    copy < active_ploidy,
                    haplotypes[states[:, copy], variant],
                    0,
                )
            state_dosage_rows.append(dosage)
        state_dosage = jnp.stack(state_dosage_rows, axis=0)
        target = cohort.genotype_probabilities[:, sample]
        gathered = jnp.take_along_axis(target, state_dosage, axis=-1)
        admissible_count = cohort.ploidy[:, sample].astype(dtype) + 1.0
        emission_probability = (
            1.0 - plan.mismatch_probability
        ) * gathered + plan.mismatch_probability / admissible_count[:, None]
        emission = jnp.where(
            cohort.observed[:, sample, None],
            jnp.log(jnp.maximum(emission_probability, 1e-300)),
            0.0,
        )
        posterior = _finite_hmm(emission, prior, transitions)
        genotype_probability = jnp.zeros(
            (cohort.variant_count, cohort.maximum_ploidy + 1), dtype=dtype
        )
        variant_indices = jnp.arange(cohort.variant_count)[:, None]
        genotype_probability = genotype_probability.at[variant_indices, state_dosage].add(
            posterior.probabilities
        )
        dosage_states = jnp.arange(cohort.maximum_ploidy + 1, dtype=dtype)
        mean = jnp.sum(genotype_probability * dosage_states, axis=-1)
        second = jnp.sum(genotype_probability * dosage_states * dosage_states, axis=-1)
        variance = jnp.maximum(second - mean * mean, 0.0)
        entropy = -jnp.sum(
            jnp.where(
                genotype_probability > 0.0,
                genotype_probability * jnp.log(genotype_probability),
                0.0,
            ),
            axis=-1,
        )
        information = jnp.clip(
            1.0 - entropy / jnp.log(cohort.ploidy[:, sample].astype(dtype) + 1.0),
            0.0,
            1.0,
        )
        copying_path = states[posterior.path]
        padded_path = (
            jnp.full((cohort.variant_count, cohort.maximum_ploidy), -1, dtype=jnp.int32)
            .at[:, :maximum_ploidy]
            .set(copying_path)
        )
        probability_outputs.append(genotype_probability)
        dosage_outputs.append(mean)
        variance_outputs.append(variance)
        information_outputs.append(information)
        path_outputs.append(padded_path)
        log_likelihoods.append(posterior.log_likelihood)
        valid_outputs.append(posterior.valid)
        status_outputs.append(
            jnp.where(
                posterior.valid,
                int(ImputationStatus.SUCCESS),
                int(ImputationStatus.DEGENERATE_LIKELIHOOD),
            ).astype(jnp.int32)
        )

    probabilities = jnp.stack(probability_outputs, axis=1)
    dosage = jnp.stack(dosage_outputs, axis=1)
    variance = jnp.stack(variance_outputs, axis=1)
    information = jnp.stack(information_outputs, axis=1)
    paths = jnp.stack(path_outputs, axis=0)
    log_likelihood = jnp.stack(log_likelihoods)
    valid = jnp.stack(valid_outputs)
    status = jnp.stack(status_outputs)
    return GenotypeImputationResult(
        probabilities,
        dosage,
        variance,
        information,
        paths,
        log_likelihood,
        jnp.asarray(required),
        valid,
        status,
        jnp.stack((jnp.asarray(required, dtype=dtype), log_likelihood), axis=-1),
        _imputation_contract(),
    )


__all__ = [
    "GenotypeImputationResult",
    "ImputationPlan",
    "ImputationStatus",
    "ReferenceHaplotypePanel",
    "impute_genotypes",
]
