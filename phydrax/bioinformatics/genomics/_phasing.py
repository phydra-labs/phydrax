#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded read- and pedigree-backed phasing for germline small variants."""

from __future__ import annotations

import math
from enum import IntEnum
from itertools import permutations

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


class PhasingStatus(IntEnum):
    """Machine-readable state of bounded read/pedigree-backed phasing."""

    OK = 0
    CAPACITY_EXCEEDED = 1
    INVALID_INPUT = 2
    CANDIDATE_OMITTED = 3
    NO_HETEROZYGOUS_SITES = 4
    INSUFFICIENT_EVIDENCE = 5
    MENDELIAN_INCONSISTENT = 6


PHASING_CONTRACT = BioinformaticsMethodContract(
    "bounded_read_pedigree_small_variant_phasing",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Exact max-product dynamic programming over all bounded copy permutations, "
        "using local read likelihoods and optional parental transmission constraints."
    ),
    truncation_statement="Reads, sites, copy permutations, and phase paths are never truncated.",
    capacity_semantics=(
        "The factorial copy-permutation requirement is preflighted; insufficient "
        "phase-state capacity returns CAPACITY_EXCEEDED without a partial path."
    ),
    assumptions=(
        "Germline small variants ordered on one contig",
        "Each spanning read originates from one uniformly selected chromosome copy",
        "For pedigree constraints, child copy zero is paternal and copy one maternal",
    ),
    nondifferentiable_outputs=(
        "phased_genotypes",
        "phase_mask",
        "phase_sets",
        "state_indices",
        "switch_supported",
        "mendelian_consistent",
        "status",
    ),
    input_dtype="float32",
    compute_dtype="float32",
    output_dtype="float32",
)


class ReadPhaseEvidence(StrictModule):
    """Per-read, per-site local allele likelihoods for one bounded interval."""

    allele_log_likelihoods: Array
    observation_mask: Array
    read_mask: Array
    omitted_observation_mask: Array
    candidate_complete: Array
    valid: Array
    allele_count: int = eqx.field(static=True)

    def __init__(
        self,
        allele_log_likelihoods: ArrayLike,
        observation_mask: ArrayLike,
        read_mask: ArrayLike,
        omitted_observation_mask: ArrayLike,
        candidate_complete: ArrayLike,
        valid: ArrayLike,
        allele_count: int,
    ):
        likelihoods = jnp.asarray(allele_log_likelihoods, dtype=jnp.float32)
        observations = jnp.asarray(observation_mask, dtype=bool)
        reads = jnp.asarray(read_mask, dtype=bool)
        omitted = jnp.asarray(omitted_observation_mask, dtype=bool)
        if likelihoods.ndim != 3:
            raise ValueError(
                "allele_log_likelihoods must have shape (reads, sites, alleles)."
            )
        if (
            observations.shape != likelihoods.shape[:2]
            or omitted.shape != observations.shape
        ):
            raise ValueError("Observation masks must match read and site capacities.")
        if reads.shape != (likelihoods.shape[0],):
            raise ValueError("read_mask must match read capacity.")
        count = int(allele_count)
        if count < 1 or likelihoods.shape[2] != count:
            raise ValueError("allele_count must match the likelihood allele axis.")
        self.allele_log_likelihoods = likelihoods
        self.observation_mask = jax.lax.stop_gradient(observations)
        self.read_mask = jax.lax.stop_gradient(reads)
        self.omitted_observation_mask = jax.lax.stop_gradient(omitted)
        self.candidate_complete = jax.lax.stop_gradient(
            jnp.asarray(candidate_complete, dtype=bool)
        )
        self.valid = jax.lax.stop_gradient(jnp.asarray(valid, dtype=bool))
        self.allele_count = count


class PedigreePhaseEvidence(StrictModule):
    """Bounded parental genotypes for diploid transmission-aware phasing."""

    paternal_genotypes: Array
    maternal_genotypes: Array
    paternal_mask: Array
    maternal_mask: Array

    def __init__(
        self,
        paternal_genotypes: ArrayLike,
        maternal_genotypes: ArrayLike,
        paternal_mask: ArrayLike | None = None,
        maternal_mask: ArrayLike | None = None,
    ):
        paternal = jnp.asarray(paternal_genotypes, dtype=jnp.int32)
        maternal = jnp.asarray(maternal_genotypes, dtype=jnp.int32)
        if (
            paternal.ndim != 2
            or maternal.ndim != 2
            or paternal.shape[0] != maternal.shape[0]
        ):
            raise ValueError("Parental genotypes must be rank two with matching sites.")
        paternal_valid = (
            paternal >= 0
            if paternal_mask is None
            else jnp.asarray(paternal_mask, dtype=bool)
        )
        maternal_valid = (
            maternal >= 0
            if maternal_mask is None
            else jnp.asarray(maternal_mask, dtype=bool)
        )
        if (
            paternal_valid.shape != paternal.shape
            or maternal_valid.shape != maternal.shape
        ):
            raise ValueError("Parental masks must match their genotype arrays.")
        self.paternal_genotypes = jax.lax.stop_gradient(paternal)
        self.maternal_genotypes = jax.lax.stop_gradient(maternal)
        self.paternal_mask = jax.lax.stop_gradient(paternal_valid)
        self.maternal_mask = jax.lax.stop_gradient(maternal_valid)


class PhasingEvidenceSummary(StrictModule):
    required_phase_state_count: Array
    phase_state_capacity: Array
    informative_link_count: Array
    phased_site_count: Array
    omitted_observation_count: Array
    mendelian_inconsistent_site_count: Array


class PhasingResult(StrictModule):
    """Phased alleles, phase blocks, switch evidence, and audit state."""

    phased_genotypes: Array
    phase_mask: Array
    phase_sets: Array
    state_indices: Array
    phase_quality: Array
    switch_log_odds: Array
    switch_supported: Array
    link_read_count: Array
    mendelian_consistent: Array
    valid: Array
    status: Array
    evidence: PhasingEvidenceSummary
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def read_phase_evidence_from_calls(
    observed_alleles: ArrayLike,
    base_quality: ArrayLike,
    mapping_quality: ArrayLike,
    observation_mask: ArrayLike,
    /,
    *,
    allele_count: int,
    read_mask: ArrayLike | None = None,
) -> ReadPhaseEvidence:
    """Calibrate bounded read-by-site allele calls into phase likelihoods."""
    observed = jnp.asarray(observed_alleles, dtype=jnp.int32)
    base = jnp.asarray(base_quality, dtype=jnp.float32)
    mapping = jnp.asarray(mapping_quality, dtype=jnp.float32)
    observations = jnp.asarray(observation_mask, dtype=bool)
    if (
        observed.ndim != 2
        or base.shape != observed.shape
        or mapping.shape != observed.shape
    ):
        raise ValueError("Allele calls and qualities must be matching rank-two arrays.")
    if observations.shape != observed.shape:
        raise ValueError("observation_mask must match allele calls.")
    count = int(allele_count)
    if count < 1:
        raise ValueError("allele_count must be positive.")
    reads = (
        jnp.ones((observed.shape[0],), dtype=bool)
        if read_mask is None
        else jnp.asarray(read_mask, dtype=bool)
    )
    if reads.shape != (observed.shape[0],):
        raise ValueError("read_mask must match read capacity.")
    active = observations & reads[:, None]
    quality_valid = jnp.all(
        (~active)
        | (jnp.isfinite(base) & jnp.isfinite(mapping) & (base >= 0.0) & (mapping >= 0.0))
    )
    omitted = active & ((observed < 0) | (observed >= count))
    safe_observed = jnp.clip(observed, 0, count - 1)
    safe_base = jnp.where(jnp.isfinite(base), jnp.maximum(base, 0.0), 0.0)
    safe_mapping = jnp.where(jnp.isfinite(mapping), jnp.maximum(mapping, 0.0), 0.0)
    base_error = jnp.power(10.0, -0.1 * safe_base)
    mapping_error = jnp.power(10.0, -0.1 * safe_mapping)
    error = jnp.clip(
        1.0 - (1.0 - base_error) * (1.0 - mapping_error),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    calls = jnp.arange(count, dtype=jnp.int32)[None, None, :]
    probabilities = jnp.where(
        calls == safe_observed[:, :, None],
        1.0 - error[:, :, None],
        error[:, :, None] / float(max(count - 1, 1)),
    )
    if count == 1:
        probabilities = jnp.ones_like(probabilities)
    likelihoods = jnp.where((active & ~omitted)[:, :, None], jnp.log(probabilities), 0.0)
    return ReadPhaseEvidence(
        likelihoods,
        observations,
        reads,
        omitted,
        ~jnp.any(omitted),
        quality_valid,
        count,
    )


def _failure_result(
    genotypes: Array,
    site_mask: Array,
    status: PhasingStatus,
    *,
    required_states: int,
    state_capacity: int,
    omitted_count: ArrayLike = 0,
    mendelian_consistent: Array | None = None,
) -> PhasingResult:
    site_count, ploidy = genotypes.shape
    boundaries = max(site_count - 1, 0)
    consistency = (
        jnp.ones((site_count,), dtype=bool)
        if mendelian_consistent is None
        else mendelian_consistent
    )
    return PhasingResult(
        jnp.where(site_mask[:, None], genotypes, -1),
        jnp.zeros((site_count,), dtype=bool),
        jnp.full((site_count,), -1, dtype=jnp.int32),
        jnp.zeros((site_count,), dtype=jnp.int32),
        jnp.zeros((site_count,), dtype=jnp.float32),
        jnp.zeros((boundaries,), dtype=jnp.float32),
        jnp.zeros((boundaries,), dtype=bool),
        jnp.zeros((boundaries,), dtype=jnp.int32),
        consistency,
        jnp.asarray(False),
        jnp.asarray(int(status), dtype=jnp.int32),
        PhasingEvidenceSummary(
            jnp.asarray(required_states, dtype=jnp.int32),
            jnp.asarray(state_capacity, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(omitted_count, dtype=jnp.int32),
            jnp.sum(~consistency, dtype=jnp.int32),
        ),
        PHASING_CONTRACT,
    )


def phase_small_variants(
    unphased_genotypes: ArrayLike,
    read_evidence: ReadPhaseEvidence,
    /,
    *,
    max_phase_states: int,
    site_mask: ArrayLike | None = None,
    pedigree_evidence: PedigreePhaseEvidence | None = None,
    min_switch_log_odds: float = math.log(10.0),
    max_phase_quality: float = 99.0,
) -> PhasingResult:
    """Exactly phase a bounded ordered germline interval by finite-state DP.

    A positive switch_log_odds means read evidence favors switching the canonical
    copy orientation at that boundary; a negative value favors retaining it.
    Phase sets contain only sites connected by decisive read evidence or a unique
    parental transmission assignment.
    """
    genotypes = jnp.asarray(unphased_genotypes, dtype=jnp.int32)
    if genotypes.ndim != 2 or genotypes.shape[0] < 1 or genotypes.shape[1] < 1:
        raise ValueError("unphased_genotypes must have shape (sites, positive ploidy).")
    sites, ploidy = genotypes.shape
    if read_evidence.allele_log_likelihoods.shape[1] != sites:
        raise ValueError("Read evidence and genotypes must share site capacity.")
    active_sites = (
        jnp.all(genotypes >= 0, axis=1)
        if site_mask is None
        else jnp.asarray(site_mask, dtype=bool)
    )
    if active_sites.shape != (sites,):
        raise ValueError("site_mask must match site capacity.")
    active_sites = active_sites & jnp.all(genotypes >= 0, axis=1)
    if pedigree_evidence is not None and (
        ploidy != 2
        or pedigree_evidence.paternal_genotypes.shape[0] != sites
        or pedigree_evidence.maternal_genotypes.shape[0] != sites
    ):
        return _failure_result(
            genotypes,
            active_sites,
            PhasingStatus.INVALID_INPUT,
            required_states=math.factorial(ploidy),
            state_capacity=int(max_phase_states),
        )
    candidate_complete = read_evidence.candidate_complete & jnp.all(
        (~active_sites[:, None]) | (genotypes < read_evidence.allele_count)
    )
    if pedigree_evidence is not None:
        candidate_complete = (
            candidate_complete
            & jnp.all(
                (~active_sites[:, None])
                | (~pedigree_evidence.paternal_mask)
                | (
                    (pedigree_evidence.paternal_genotypes >= 0)
                    & (pedigree_evidence.paternal_genotypes < read_evidence.allele_count)
                )
            )
            & jnp.all(
                (~active_sites[:, None])
                | (~pedigree_evidence.maternal_mask)
                | (
                    (pedigree_evidence.maternal_genotypes >= 0)
                    & (pedigree_evidence.maternal_genotypes < read_evidence.allele_count)
                )
            )
        )
    state_capacity = int(max_phase_states)
    required_states = math.factorial(ploidy)
    if state_capacity < 1:
        raise ValueError("max_phase_states must be positive.")
    omitted_count = jnp.sum(read_evidence.omitted_observation_mask, dtype=jnp.int32)
    if required_states > state_capacity:
        return _failure_result(
            genotypes,
            active_sites,
            PhasingStatus.CAPACITY_EXCEEDED,
            required_states=required_states,
            state_capacity=state_capacity,
            omitted_count=omitted_count,
        )

    copy_permutations = list(permutations(range(ploidy)))
    permutation_indices = (
        jnp.zeros((state_capacity, ploidy), dtype=jnp.int32)
        .at[:required_states]
        .set(jnp.asarray(copy_permutations, dtype=jnp.int32))
    )
    permutation_mask = jnp.arange(state_capacity) < required_states
    phase_states = jnp.take_along_axis(
        genotypes[:, None, :], permutation_indices[None, :, :], axis=2
    )
    unique_parts = []
    for state_index in range(state_capacity):
        if state_index == 0:
            unique = jnp.ones((sites,), dtype=bool)
        else:
            duplicate = jnp.any(
                jnp.all(
                    phase_states[:, state_index, None, :]
                    == phase_states[:, :state_index, :],
                    axis=-1,
                ),
                axis=1,
            )
            unique = ~duplicate
        unique_parts.append(unique & permutation_mask[state_index])
    unique_state_mask = jnp.stack(unique_parts, axis=1) & active_sites[:, None]

    heterozygous = active_sites & jnp.any(genotypes != genotypes[:, :1], axis=1)
    mendelian_consistent = jnp.ones((sites,), dtype=bool)
    pedigree_informative = jnp.zeros((sites,), dtype=bool)
    state_mask = unique_state_mask
    if pedigree_evidence is not None:
        paternal_allowed = jnp.any(
            (
                phase_states[:, :, 0, None]
                == pedigree_evidence.paternal_genotypes[:, None, :]
            )
            & pedigree_evidence.paternal_mask[:, None, :],
            axis=2,
        )
        maternal_allowed = jnp.any(
            (
                phase_states[:, :, 1, None]
                == pedigree_evidence.maternal_genotypes[:, None, :]
            )
            & pedigree_evidence.maternal_mask[:, None, :],
            axis=2,
        )
        transmission_mask = unique_state_mask & paternal_allowed & maternal_allowed
        mendelian_consistent = (~active_sites) | jnp.any(transmission_mask, axis=1)
        pedigree_informative = (
            heterozygous
            & mendelian_consistent
            & (jnp.sum(transmission_mask, axis=1) == 1)
        )
        state_mask = jnp.where(
            mendelian_consistent[:, None], transmission_mask, unique_state_mask
        )

    working_mask = jnp.where(
        active_sites[:, None],
        state_mask,
        jnp.arange(state_capacity)[None, :] == 0,
    )
    transition_parts = []
    link_counts = []
    for boundary in range(sites - 1):
        left_states = jnp.clip(phase_states[boundary], 0, read_evidence.allele_count - 1)
        right_states = jnp.clip(
            phase_states[boundary + 1], 0, read_evidence.allele_count - 1
        )
        left = jnp.take(
            read_evidence.allele_log_likelihoods[:, boundary, :],
            left_states,
            axis=1,
        )
        right = jnp.take(
            read_evidence.allele_log_likelihoods[:, boundary + 1, :],
            right_states,
            axis=1,
        )
        copy_log_likelihood = left[:, :, None, :] + right[:, None, :, :]
        read_log_likelihood = jsp.special.logsumexp(
            copy_log_likelihood, axis=-1
        ) - math.log(ploidy)
        covering = (
            read_evidence.read_mask
            & read_evidence.observation_mask[:, boundary]
            & read_evidence.observation_mask[:, boundary + 1]
        )
        transition = compensated_sum(
            jnp.where(covering[:, None, None], read_log_likelihood, 0.0),
            axis=0,
        )
        transition = jnp.where(
            working_mask[boundary, :, None] & working_mask[boundary + 1, None, :],
            transition,
            -jnp.inf,
        )
        transition_parts.append(transition)
        link_counts.append(jnp.sum(covering, dtype=jnp.int32))
    transitions = (
        jnp.stack(transition_parts, axis=0)
        if transition_parts
        else jnp.zeros((0, state_capacity, state_capacity), dtype=jnp.float32)
    )
    link_read_count = (
        jnp.stack(link_counts) if link_counts else jnp.zeros((0,), dtype=jnp.int32)
    )

    dynamic_scores = jnp.where(working_mask[0], 0.0, -jnp.inf)
    backpointers = []
    for boundary in range(sites - 1):
        path_scores = dynamic_scores[:, None] + transitions[boundary]
        backpointers.append(jnp.argmax(path_scores, axis=0).astype(jnp.int32))
        dynamic_scores = jnp.max(path_scores, axis=0)
    state_indices = jnp.zeros((sites,), dtype=jnp.int32)
    terminal_state = jnp.argmax(dynamic_scores).astype(jnp.int32)
    state_indices = state_indices.at[-1].set(terminal_state)
    for site_index in range(sites - 2, -1, -1):
        terminal_state = backpointers[site_index][terminal_state]
        state_indices = state_indices.at[site_index].set(terminal_state)

    switch_log_odds = jnp.zeros((max(sites - 1, 0),), dtype=jnp.float32)
    if ploidy == 2 and state_capacity >= 2 and sites > 1:
        switch_parts = []
        for boundary in range(sites - 1):
            same = jnp.logaddexp(transitions[boundary, 0, 0], transitions[boundary, 1, 1])
            switched = jnp.logaddexp(
                transitions[boundary, 0, 1], transitions[boundary, 1, 0]
            )
            informative = heterozygous[boundary] & heterozygous[boundary + 1]
            switch_parts.append(
                jnp.where(
                    informative & jnp.isfinite(same) & jnp.isfinite(switched),
                    switched - same,
                    0.0,
                )
            )
        switch_log_odds = jnp.stack(switch_parts)

    threshold = float(min_switch_log_odds)
    quality_cap = float(max_phase_quality)
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("min_switch_log_odds must be finite and non-negative.")
    if quality_cap < 0.0 or not math.isfinite(quality_cap):
        raise ValueError("max_phase_quality must be finite and non-negative.")
    decisive_read_link = (
        (link_read_count > 0)
        & (jnp.abs(switch_log_odds) >= threshold)
        & heterozygous[:-1]
        & heterozygous[1:]
    )
    pedigree_link = pedigree_informative[:-1] & pedigree_informative[1:]
    strong_link = decisive_read_link | pedigree_link
    previous_link = jnp.concatenate((jnp.asarray([False]), strong_link), axis=0)
    next_link = jnp.concatenate((strong_link, jnp.asarray([False])), axis=0)
    phase_mask = heterozygous & (previous_link | next_link | pedigree_informative)
    starts = phase_mask & ~previous_link
    site_indices = jnp.arange(sites, dtype=jnp.int32)
    phase_labels = jax.lax.associative_scan(
        jnp.maximum, jnp.where(starts, site_indices, -1)
    )
    phase_sets = jnp.where(phase_mask, phase_labels, -1)
    selected_states = phase_states[site_indices, state_indices]
    phased_genotypes = jnp.where(phase_mask[:, None], selected_states, genotypes)
    boundary_quality = jnp.minimum(
        10.0 * jnp.abs(switch_log_odds) / math.log(10.0), quality_cap
    )
    phase_quality = jnp.maximum(
        jnp.concatenate((jnp.zeros((1,), dtype=jnp.float32), boundary_quality)),
        jnp.concatenate((boundary_quality, jnp.zeros((1,), dtype=jnp.float32))),
    )
    phase_quality = jnp.where(
        pedigree_informative, quality_cap, jnp.where(phase_mask, phase_quality, 0.0)
    )
    switch_supported = decisive_read_link & (switch_log_odds > 0.0)

    all_mendelian = jnp.all(mendelian_consistent)
    valid = read_evidence.valid & candidate_complete & all_mendelian
    has_heterozygous = jnp.any(heterozygous)
    has_phase = jnp.any(phase_mask)
    status = jnp.where(
        ~read_evidence.valid,
        int(PhasingStatus.INVALID_INPUT),
        jnp.where(
            ~candidate_complete,
            int(PhasingStatus.CANDIDATE_OMITTED),
            jnp.where(
                ~all_mendelian,
                int(PhasingStatus.MENDELIAN_INCONSISTENT),
                jnp.where(
                    ~has_heterozygous,
                    int(PhasingStatus.NO_HETEROZYGOUS_SITES),
                    jnp.where(
                        has_phase,
                        int(PhasingStatus.OK),
                        int(PhasingStatus.INSUFFICIENT_EVIDENCE),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return PhasingResult(
        jax.lax.stop_gradient(phased_genotypes),
        jax.lax.stop_gradient(phase_mask),
        jax.lax.stop_gradient(phase_sets),
        jax.lax.stop_gradient(state_indices),
        phase_quality,
        switch_log_odds,
        jax.lax.stop_gradient(switch_supported),
        jax.lax.stop_gradient(link_read_count),
        jax.lax.stop_gradient(mendelian_consistent),
        valid,
        status,
        PhasingEvidenceSummary(
            jnp.asarray(required_states, dtype=jnp.int32),
            jnp.asarray(state_capacity, dtype=jnp.int32),
            jnp.sum(strong_link, dtype=jnp.int32),
            jnp.sum(phase_mask, dtype=jnp.int32),
            omitted_count,
            jnp.sum(~mendelian_consistent, dtype=jnp.int32),
        ),
        PHASING_CONTRACT,
    )


__all__ = [
    "PHASING_CONTRACT",
    "PedigreePhaseEvidence",
    "PhasingEvidenceSummary",
    "PhasingResult",
    "PhasingStatus",
    "ReadPhaseEvidence",
    "phase_small_variants",
    "read_phase_evidence_from_calls",
]
