#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...uq import FiniteStateSmootherResult
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


class RecombinationStatus(IntEnum):
    SUCCESS = 0
    DEGENERATE_LIKELIHOOD = 1
    NONFINITE = 2
    STATE_MISMATCH = 3


class RecombinationMap(StrictModule):
    positions: Array
    genetic_positions: Array
    chromosome_index: Array

    def __init__(
        self,
        positions: ArrayLike,
        genetic_positions: ArrayLike,
        /,
        *,
        chromosome_index: ArrayLike | None = None,
    ):
        positions_ = jnp.asarray(positions)
        genetic_ = jnp.asarray(genetic_positions)
        if positions_.ndim != 1 or genetic_.shape != positions_.shape:
            raise ValueError(
                "positions and genetic_positions must be equal-length vectors."
            )
        if jnp.iscomplexobj(positions_) or jnp.iscomplexobj(genetic_):
            raise TypeError("Recombination map coordinates must be real-valued.")
        if not jnp.issubdtype(positions_.dtype, jnp.inexact):
            positions_ = positions_.astype(float)
        if not jnp.issubdtype(genetic_.dtype, jnp.inexact):
            genetic_ = genetic_.astype(float)
        chromosome = (
            jnp.zeros(positions_.shape, dtype=jnp.int32)
            if chromosome_index is None
            else jnp.asarray(chromosome_index)
        )
        if chromosome.shape != positions_.shape or not jnp.issubdtype(
            chromosome.dtype, jnp.integer
        ):
            raise ValueError(
                "chromosome_index must be an integer vector matching positions."
            )
        host_position = np.asarray(positions_)
        host_genetic = np.asarray(genetic_)
        host_chromosome = np.asarray(chromosome)
        if not np.all(np.isfinite(host_position)) or not np.all(
            np.isfinite(host_genetic)
        ):
            raise ValueError("Recombination map coordinates must be finite.")
        if np.any(host_position < 0.0) or np.any(host_genetic < 0.0):
            raise ValueError("Recombination map coordinates must be non-negative.")
        for value in np.unique(host_chromosome):
            selected = host_chromosome == value
            if np.any(np.diff(host_position[selected]) <= 0.0):
                raise ValueError("Physical positions must increase within a chromosome.")
            if np.any(np.diff(host_genetic[selected]) < 0.0):
                raise ValueError(
                    "Genetic positions must not decrease within a chromosome."
                )
        self.positions = positions_
        self.genetic_positions = genetic_
        self.chromosome_index = chromosome.astype(jnp.int32)

    @property
    def variant_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def interval_distance(self) -> Array:
        same = self.chromosome_index[1:] == self.chromosome_index[:-1]
        distance = self.genetic_positions[1:] - self.genetic_positions[:-1]
        return jnp.where(same, distance, jnp.inf)

    def haldane_recombination_fraction(self, /, *, scale: float = 1.0) -> Array:
        """Haldane recombination fraction, with chromosome boundaries at one half."""
        resolved = float(scale)
        if not np.isfinite(resolved) or resolved < 0.0:
            raise ValueError("scale must be finite and non-negative.")
        distance = self.interval_distance * resolved
        return jnp.where(jnp.isinf(distance), 0.5, -0.5 * jnp.expm1(-2.0 * distance))


class RecombinationMosaicResult(StrictModule):
    state_probabilities: Array
    transition_probabilities: Array
    state_path: Array
    switch_probability: Array
    log_likelihood: Array
    valid: Array
    status: Array
    evidence: Array
    native_smoother: FiniteStateSmootherResult | None
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(RecombinationStatus.SUCCESS))


class LocalAncestryResult(StrictModule):
    ancestry_probabilities: Array
    ancestry_path: Array
    transition_probabilities: Array
    switch_probability: Array
    log_likelihood: Array
    valid: Array
    status: Array
    evidence: Array
    native_smoother: FiniteStateSmootherResult | None
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(RecombinationStatus.SUCCESS))


def _hmm_contract(method_name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.EXACT_AD,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Conditioned on the supplied finite-state emissions, ancestry/state prior, "
            "and recombination-map transition law."
        ),
        truncation_statement="All declared finite states are retained.",
        capacity_semantics="State capacity is the explicit final emission dimension.",
        assumptions=("First-order Markov mosaic along each chromosome.",),
        nondifferentiable_outputs=("state_path", "ancestry_path", "status", "valid"),
    )


class _FiniteHMMPosterior(StrictModule):
    probabilities: Array
    transition_probabilities: Array
    path: Array
    log_likelihood: Array
    valid: Array
    status: Array


def _normalize_log(values: Array, /) -> tuple[Array, Array]:
    normalizer = jsp.special.logsumexp(values, axis=-1)
    normalized = values - normalizer[..., None]
    return normalized, normalizer


def _finite_hmm(
    emission_log_likelihood: Array,
    initial_probabilities: Array,
    transition_matrices: Array,
    /,
) -> _FiniteHMMPosterior:
    """Exact scaled forward-backward and Viterbi recursion for batched chains."""
    emission = jnp.asarray(emission_log_likelihood)
    if emission.ndim < 2:
        raise ValueError("emission_log_likelihood needs site and state axes.")
    site_count, state_count = emission.shape[-2:]
    case_shape = emission.shape[:-2]
    initial = jnp.asarray(initial_probabilities, dtype=emission.dtype)
    if initial.shape != (state_count,):
        raise ValueError("initial_probabilities must contain one value per state.")
    transitions = jnp.asarray(transition_matrices, dtype=emission.dtype)
    if transitions.shape != (max(site_count - 1, 0), state_count, state_count):
        raise ValueError(
            "transition_matrices must have shape (sites - 1, states, states)."
        )
    host_initial = np.asarray(initial)
    host_transitions = np.asarray(transitions)
    host_emission = np.asarray(emission)
    if (
        np.any(host_initial < 0.0)
        or not np.isclose(host_initial.sum(), 1.0)
        or np.any(host_transitions < 0.0)
        or (host_transitions.size and not np.allclose(host_transitions.sum(axis=-1), 1.0))
    ):
        raise ValueError("Initial probabilities and transition rows must be stochastic.")
    if np.any(np.isnan(host_emission)) or np.any(np.isposinf(host_emission)):
        raise ValueError(
            "Emission log likelihoods may contain finite values and -inf only."
        )

    log_initial = jnp.where(initial > 0.0, jnp.log(initial), -jnp.inf)
    log_transition = jnp.where(transitions > 0.0, jnp.log(transitions), -jnp.inf)
    alpha, first_increment = _normalize_log(log_initial + emission[..., 0, :])
    alpha_history = [alpha]
    increments = [first_increment]
    viterbi = log_initial + emission[..., 0, :]
    pointers: list[Array] = []
    for site in range(1, site_count):
        joint = alpha[..., :, None] + log_transition[site - 1]
        predicted = jsp.special.logsumexp(joint, axis=-2)
        alpha, increment = _normalize_log(predicted + emission[..., site, :])
        alpha_history.append(alpha)
        increments.append(increment)
        viterbi_joint = viterbi[..., :, None] + log_transition[site - 1]
        pointers.append(jnp.argmax(viterbi_joint, axis=-2).astype(jnp.int32))
        viterbi = jnp.max(viterbi_joint, axis=-2) + emission[..., site, :]
    alpha_values = jnp.stack(alpha_history, axis=-2)
    log_likelihood = jnp.sum(jnp.stack(increments, axis=-1), axis=-1)

    beta = jnp.zeros(case_shape + (state_count,), dtype=emission.dtype)
    beta_history: list[Array] = [beta]
    for site in range(site_count - 2, -1, -1):
        beta = jsp.special.logsumexp(
            log_transition[site] + emission[..., site + 1, None, :] + beta[..., None, :],
            axis=-1,
        )
        beta_history.append(beta)
    beta_values = jnp.stack(beta_history[::-1], axis=-2)
    log_posterior, _ = _normalize_log(alpha_values + beta_values)
    posterior = jnp.exp(log_posterior)

    pairwise: list[Array] = []
    for site in range(site_count - 1):
        log_pair = (
            alpha_values[..., site, :, None]
            + log_transition[site]
            + emission[..., site + 1, None, :]
            + beta_values[..., site + 1, None, :]
        )
        normalizer = jsp.special.logsumexp(log_pair, axis=(-2, -1))
        pairwise.append(jnp.exp(log_pair - normalizer[..., None, None]))
    transition_posterior = (
        jnp.stack(pairwise, axis=-3)
        if pairwise
        else jnp.zeros(case_shape + (0, state_count, state_count), dtype=emission.dtype)
    )

    current = jnp.argmax(viterbi, axis=-1).astype(jnp.int32)
    path = jnp.zeros(case_shape + (site_count,), dtype=jnp.int32)
    path = path.at[..., -1].set(current)
    for site in range(site_count - 2, -1, -1):
        current = jnp.take_along_axis(pointers[site], current[..., None], axis=-1)[..., 0]
        path = path.at[..., site].set(current)

    finite = jnp.isfinite(log_likelihood)
    posterior_finite = jnp.all(jnp.isfinite(posterior), axis=(-2, -1))
    valid = finite & posterior_finite
    status = jnp.where(
        valid,
        int(RecombinationStatus.SUCCESS),
        jnp.where(
            finite,
            int(RecombinationStatus.NONFINITE),
            int(RecombinationStatus.DEGENERATE_LIKELIHOOD),
        ),
    ).astype(jnp.int32)
    return _FiniteHMMPosterior(
        posterior,
        transition_posterior,
        path,
        log_likelihood,
        valid,
        status,
    )


def _switch_probability(pairwise: Array, /) -> Array:
    state_count = int(pairwise.shape[-1])
    same = jnp.sum(pairwise * jnp.eye(state_count, dtype=pairwise.dtype), axis=(-2, -1))
    return jnp.clip(1.0 - same, 0.0, 1.0)


def infer_recombination_mosaic(
    emission_log_likelihood: ArrayLike,
    recombination_map: RecombinationMap,
    initial_probabilities: ArrayLike,
    /,
    *,
    recombination_scale: float = 1.0,
) -> RecombinationMosaicResult:
    """Infer an exact copying-state mosaic using Haldane interval transitions."""
    if not isinstance(recombination_map, RecombinationMap):
        raise TypeError("recombination_map must be a RecombinationMap.")
    emission = jnp.asarray(emission_log_likelihood)
    if emission.shape[-2] != recombination_map.variant_count:
        raise ValueError("Emission sites must match the recombination map.")
    state_count = int(emission.shape[-1])
    prior = jnp.asarray(initial_probabilities, dtype=emission.dtype)
    if prior.shape != (state_count,):
        raise ValueError("initial_probabilities must contain one value per state.")
    recombination = recombination_map.haldane_recombination_fraction(
        scale=recombination_scale
    )
    recombination = jnp.where(
        recombination_map.chromosome_index[1:] != recombination_map.chromosome_index[:-1],
        1.0,
        recombination,
    )
    eye = jnp.eye(state_count, dtype=emission.dtype)
    transitions = (1.0 - recombination[:, None, None]) * eye[None, :, :] + recombination[
        :, None, None
    ] * prior[None, None, :]
    posterior = _finite_hmm(emission, prior, transitions)
    switches = _switch_probability(posterior.transition_probabilities)
    evidence = jnp.stack((posterior.log_likelihood, jnp.sum(switches, axis=-1)), axis=-1)
    return RecombinationMosaicResult(
        posterior.probabilities,
        posterior.transition_probabilities,
        posterior.path,
        switches,
        posterior.log_likelihood,
        posterior.valid,
        posterior.status,
        evidence,
        None,
        _hmm_contract("recombination-copying-mosaic-hmm"),
    )


def infer_local_ancestry(
    emission_log_likelihood: ArrayLike,
    recombination_map: RecombinationMap,
    ancestry_proportions: ArrayLike,
    /,
    *,
    generations: float,
) -> LocalAncestryResult:
    """Infer local ancestry under a pulse-admixture finite-state HMM."""
    generation_count = float(generations)
    if not np.isfinite(generation_count) or generation_count <= 0.0:
        raise ValueError("generations must be finite and positive.")
    emission = jnp.asarray(emission_log_likelihood)
    proportions = jnp.asarray(ancestry_proportions, dtype=emission.dtype)
    if emission.shape[-2] != recombination_map.variant_count:
        raise ValueError("Emission sites must match the recombination map.")
    ancestry_count = int(emission.shape[-1])
    if proportions.shape != (ancestry_count,):
        raise ValueError("ancestry_proportions must contain one value per ancestry.")
    host_proportions = np.asarray(proportions)
    if np.any(host_proportions < 0.0) or not np.isclose(host_proportions.sum(), 1.0):
        raise ValueError("ancestry_proportions must be a probability vector.")
    distance = recombination_map.interval_distance
    switch = jnp.where(jnp.isinf(distance), 1.0, -jnp.expm1(-generation_count * distance))
    eye = jnp.eye(ancestry_count, dtype=emission.dtype)
    transitions = (1.0 - switch[:, None, None]) * eye[None, :, :] + switch[
        :, None, None
    ] * proportions[None, None, :]
    posterior = _finite_hmm(emission, proportions, transitions)
    switches = _switch_probability(posterior.transition_probabilities)
    evidence = jnp.stack((posterior.log_likelihood, jnp.sum(switches, axis=-1)), axis=-1)
    return LocalAncestryResult(
        posterior.probabilities,
        posterior.path,
        posterior.transition_probabilities,
        switches,
        posterior.log_likelihood,
        posterior.valid,
        posterior.status,
        evidence,
        None,
        _hmm_contract("pulse-admixture-local-ancestry-hmm"),
    )


def recombination_mosaic_from_finite_state(
    smoother: FiniteStateSmootherResult, /
) -> RecombinationMosaicResult:
    """Adapt a native finite-state UQ smoother without altering its probabilities."""
    if not isinstance(smoother, FiniteStateSmootherResult):
        raise TypeError("smoother must be a FiniteStateSmootherResult.")
    probabilities = smoother.smoothed_probabilities
    pairwise = smoother.transition_probabilities[..., 1:, :, :]
    path = jnp.argmax(probabilities, axis=-1).astype(jnp.int32)
    switches = _switch_probability(pairwise)
    log_likelihood = smoother.filter_result.cumulative_log_likelihood[..., -1]
    valid = smoother.successful
    status = jnp.where(
        valid, int(RecombinationStatus.SUCCESS), int(RecombinationStatus.NONFINITE)
    ).astype(jnp.int32)
    evidence = jnp.stack((log_likelihood, jnp.sum(switches, axis=-1)), axis=-1)
    return RecombinationMosaicResult(
        probabilities,
        pairwise,
        path,
        switches,
        log_likelihood,
        valid,
        status,
        evidence,
        smoother,
        _hmm_contract("native-finite-state-recombination-adapter"),
    )


__all__ = [
    "LocalAncestryResult",
    "RecombinationMap",
    "RecombinationMosaicResult",
    "RecombinationStatus",
    "infer_local_ancestry",
    "infer_recombination_mosaic",
    "recombination_mosaic_from_finite_state",
]
