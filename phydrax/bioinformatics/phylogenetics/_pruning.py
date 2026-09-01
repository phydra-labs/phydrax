#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import SequenceBatch
from ._rates import DiscreteRateMixture, unit_rate_mixture
from ._substitution import FiniteStateSubstitutionModel
from ._tree import TreeTopology


AscertainmentKind = Literal["none", "variable"]


class PruningStatus(IntEnum):
    SUCCESS = 0
    INVALID_TOPOLOGY = 1
    INVALID_BRANCH_LENGTH = 2
    INVALID_TIP_PARTIAL = 3
    INVALID_PATTERN_WEIGHT = 4
    INVALID_PARTITION = 5
    NONSTOCHASTIC_TRANSITION = 6
    IMPOSSIBLE_ASCERTAINMENT = 7
    NONFINITE_LIKELIHOOD = 8


class LikelihoodPartition(StrictModule):
    """One complete pattern subset with its model and site-rate law."""

    pattern_mask: Array
    model: FiniteStateSubstitutionModel
    rate_mixture: DiscreteRateMixture
    root_distribution: Array
    valid: Array
    ascertainment: AscertainmentKind = eqx.field(static=True)
    partition_name: str = eqx.field(static=True)

    def __init__(
        self,
        pattern_mask: ArrayLike,
        model: FiniteStateSubstitutionModel,
        /,
        *,
        rate_mixture: DiscreteRateMixture | None = None,
        root_distribution: ArrayLike | None = None,
        ascertainment: AscertainmentKind = "none",
        partition_name: str = "partition",
    ):
        if not isinstance(model, FiniteStateSubstitutionModel):
            raise TypeError("model must be a FiniteStateSubstitutionModel.")
        mixture = (
            unit_rate_mixture(dtype=model.rate_matrix.dtype)
            if rate_mixture is None
            else rate_mixture
        )
        if not isinstance(mixture, DiscreteRateMixture):
            raise TypeError("rate_mixture must be a DiscreteRateMixture.")
        mask = jnp.asarray(pattern_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("pattern_mask must be rank one.")
        root = (
            model.root_distribution
            if root_distribution is None
            else jnp.asarray(root_distribution, dtype=model.rate_matrix.dtype)
        )
        if root.shape != (model.state_count,):
            raise ValueError("root_distribution has the wrong state count.")
        if ascertainment not in ("none", "variable"):
            raise ValueError("ascertainment must be 'none' or 'variable'.")
        tolerance = 64.0 * jnp.finfo(root.dtype).eps
        root_valid = (
            jnp.all(jnp.isfinite(root))
            & jnp.all(root >= 0.0)
            & (jnp.abs(jnp.sum(root) - 1.0) <= tolerance)
        )
        self.pattern_mask = mask
        self.model = model
        self.rate_mixture = mixture
        self.root_distribution = root
        self.valid = model.valid & mixture.valid & root_valid
        self.ascertainment = ascertainment
        self.partition_name = str(partition_name)


class PruningEvidence(StrictModule):
    """Numerical and data-domain evidence for a pruning evaluation."""

    topology_valid: Array
    branch_lengths_valid: Array
    root_branch_zero: Array
    tip_partials_valid: Array
    pattern_weights_valid: Array
    partition_coverage: Array
    partitions_valid: Array
    transition_matrices_stochastic: Array
    ascertainment_probabilities: Array
    minimum_scaling_factor: Array
    finite_pattern_likelihoods: Array


class FelsensteinPruningResult(StrictModule):
    """Exact fixed-tree finite-state likelihood with scaled conditional vectors."""

    log_likelihood: Array
    pattern_log_likelihood: Array
    weighted_pattern_log_likelihood: Array
    partition_log_likelihood: Array
    partition_pattern_weight: Array
    valid: Array
    status: Array
    evidence: PruningEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class TipPartialEvidence(StrictModule):
    """Evidence for lowering encoded sequence observations to state sets."""

    canonical_state_count: Array
    active_tip_count: Array
    active_site_count: Array
    codes_in_range: Array
    complete_tip_sites: Array


class TipPartialEncodingResult(StrictModule):
    """Tip conditional vectors preserving ambiguity and missingness."""

    tip_partials: Array
    site_mask: Array
    valid: Array
    status: Array
    evidence: TipPartialEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _pruning_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "fixed_tree_felsenstein_pruning",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.SCALAR,
        conditioning_statement=(
            "Likelihood is conditioned on the supplied rooted topology, branch "
            "lengths, finite-state models, root laws, rate mixtures, and pattern weights."
        ),
        truncation_statement="No nodes, children, patterns, states, or rate categories are truncated.",
        capacity_semantics=(
            "Tree child capacity is preflighted during topology lowering; every supplied "
            "pattern belongs to exactly one declared partition."
        ),
        assumptions=(
            "Sites are conditionally independent within each declared partition.",
            "Tip state sets encode observation likelihoods up to a state-independent factor.",
        ),
        nondifferentiable_outputs=("valid", "status", "evidence"),
    )


def _tip_partial_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "sequence_tip_partial_lowering",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.ARRAY,
        conditioning_statement="State sets follow the static AlphabetPlan ambiguity table.",
        truncation_statement="No active records, sites, or alphabet symbols are truncated.",
        capacity_semantics="SequenceBatch record and site capacities are preserved exactly.",
        assumptions=("Canonical alphabet symbols define the substitution state order.",),
        nondifferentiable_outputs=("tip_partials", "site_mask", "valid", "status"),
    )


def tip_partials_from_sequence(batch: SequenceBatch, /) -> TipPartialEncodingResult:
    """Lower canonical, ambiguous, gap, and missing tokens to tip state sets.

    Canonical tokens become one-hot vectors. Declared ambiguity tokens expand to
    their canonical state sets. Gap, unknown, missing, mask, and padding tokens
    carry no state information and therefore become all-ones vectors.
    """

    if not isinstance(batch, SequenceBatch):
        raise TypeError("batch must be a SequenceBatch.")
    alphabet = batch.alphabet
    canonical = alphabet.canonical_symbols
    canonical_index = {symbol: index for index, symbol in enumerate(canonical)}
    ambiguity = alphabet.ambiguity_map
    table = np.ones((alphabet.size, len(canonical)), dtype=np.float32)
    for code, symbol in enumerate(alphabet.symbols):
        if symbol in canonical_index:
            table[code] = 0.0
            table[code, canonical_index[symbol]] = 1.0
        elif symbol in ambiguity:
            table[code] = 0.0
            for represented in ambiguity[symbol]:
                table[code, canonical_index[represented]] = 1.0
    lookup = jnp.asarray(table)
    codes = jnp.asarray(batch.token_codes)
    in_range = (codes >= 0) & (codes < alphabet.size)
    safe_codes = jnp.where(in_range, codes, 0)
    record_partials = lookup[safe_codes]
    active_records = jnp.asarray(batch.case_mask, dtype=bool)
    complete = jnp.all(
        jnp.where(active_records[:, None], batch.valid_mask & in_range, True), axis=0
    )
    site_mask = complete & jnp.any(active_records)
    tip_partials = jnp.swapaxes(record_partials, 0, 1)
    valid = jnp.all(jnp.where(active_records[:, None], in_range, True)) & jnp.any(
        active_records
    )
    evidence = TipPartialEvidence(
        canonical_state_count=jnp.asarray(len(canonical), dtype=jnp.int32),
        active_tip_count=jnp.sum(active_records, dtype=jnp.int32),
        active_site_count=jnp.sum(site_mask, dtype=jnp.int32),
        codes_in_range=jnp.all(jnp.where(active_records[:, None], in_range, True)),
        complete_tip_sites=complete,
    )
    return TipPartialEncodingResult(
        tip_partials=tip_partials,
        site_mask=site_mask,
        valid=valid,
        status=jnp.where(valid, 0, 1).astype(jnp.int32),
        evidence=evidence,
        method_contract=_tip_partial_contract(),
    )


def _category_pruning(
    topology: TreeTopology,
    tip_partials: Array,
    branch_lengths: Array,
    model: FiniteStateSubstitutionModel,
    root_distribution: Array,
    rate: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return log likelihood, normalized partials, scales, transition validity, min scale."""

    pattern_count = int(tip_partials.shape[0])
    state_count = model.state_count
    dtype = jnp.result_type(
        tip_partials.dtype, branch_lengths.dtype, model.rate_matrix.dtype
    )
    partials = (
        jnp.zeros((topology.node_count, pattern_count, state_count), dtype=dtype)
        .at[topology.tip_indices]
        .set(jnp.swapaxes(tip_partials.astype(dtype), 0, 1))
    )
    log_scales = jnp.zeros((topology.node_count, pattern_count), dtype=dtype)
    transition_valid = jnp.asarray(True)
    minimum_scale = jnp.asarray(jnp.inf, dtype=dtype)
    tolerance = jnp.maximum(
        jnp.asarray(1.0e-5, dtype=dtype),
        256.0 * jnp.finfo(dtype).eps,
    )

    def node_step(index, carry):
        node_partials, node_scales, matrices_valid, minimum = carry
        node = jax.lax.stop_gradient(topology.postorder[index])
        product = jnp.ones((pattern_count, state_count), dtype=dtype)
        inherited_scale = jnp.zeros((pattern_count,), dtype=dtype)
        has_child = jnp.asarray(False)

        def child_step(slot, child_carry):
            active = topology.child_mask[node, slot]

            def include_child(local_carry):
                running, scale_running, _, all_valid = local_carry
                child = jax.lax.stop_gradient(topology.child_indices[node, slot])
                duration = branch_lengths[child] * rate
                matrix = model.transition_matrix(duration).astype(dtype)
                contribution = node_partials[child] @ jnp.swapaxes(matrix, -1, -2)
                stochastic = (
                    jnp.all(jnp.isfinite(matrix))
                    & jnp.all(matrix >= -tolerance)
                    & jnp.all(jnp.abs(jnp.sum(matrix, axis=-1) - 1.0) <= tolerance)
                )
                return (
                    running * contribution,
                    scale_running + node_scales[child],
                    jnp.asarray(True),
                    all_valid & stochastic,
                )

            return jax.lax.cond(active, include_child, lambda value: value, child_carry)

        product, inherited_scale, has_child, local_valid = jax.lax.fori_loop(
            0,
            topology.child_capacity,
            child_step,
            (product, inherited_scale, has_child, jnp.asarray(True)),
        )
        scale = jnp.max(product, axis=-1)
        positive_scale = jnp.isfinite(scale) & (scale > 0.0)
        safe_scale = jnp.where(positive_scale, scale, 1.0)
        normalized = product / safe_scale[:, None]
        updated_partials = jnp.where(has_child, normalized, node_partials[node])
        updated_scales = jnp.where(
            has_child,
            inherited_scale + jnp.log(safe_scale),
            node_scales[node],
        )
        node_partials = node_partials.at[node].set(updated_partials)
        node_scales = node_scales.at[node].set(updated_scales)
        minimum = jnp.minimum(minimum, jnp.where(has_child, jnp.min(safe_scale), minimum))
        return node_partials, node_scales, matrices_valid & local_valid, minimum

    partials, log_scales, transition_valid, minimum_scale = jax.lax.fori_loop(
        0,
        topology.node_count,
        node_step,
        (partials, log_scales, transition_valid, minimum_scale),
    )
    root = jax.lax.stop_gradient(topology.root_index)
    root_probability = partials[root] @ root_distribution.astype(dtype)
    positive = jnp.isfinite(root_probability) & (root_probability > 0.0)
    safe_probability = jnp.where(positive, root_probability, 1.0)
    log_likelihood = jnp.where(
        positive, jnp.log(safe_probability) + log_scales[root], -jnp.inf
    )
    return log_likelihood, partials, log_scales, transition_valid, minimum_scale


def _mixture_pruning(
    topology: TreeTopology,
    tip_partials: Array,
    branch_lengths: Array,
    partition: LikelihoodPartition,
    /,
) -> tuple[Array, Array, Array]:
    category_logs, transition_valid, minimum_scale = jax.vmap(
        lambda rate: (lambda result: (result[0], result[3], result[4]))(
            _category_pruning(
                topology,
                tip_partials,
                branch_lengths,
                partition.model,
                partition.root_distribution,
                rate,
            )
        )
    )(partition.rate_mixture.rates)
    log_weights = jnp.where(
        partition.rate_mixture.weights > 0.0,
        jnp.log(
            jnp.maximum(
                partition.rate_mixture.weights, jnp.finfo(category_logs.dtype).tiny
            )
        ),
        -jnp.inf,
    )
    pattern_log_likelihood = jsp.special.logsumexp(
        category_logs + log_weights[:, None], axis=0
    )
    return (
        pattern_log_likelihood,
        jnp.all(transition_valid),
        jnp.min(minimum_scale),
    )


def _variable_ascertainment_probability(
    topology: TreeTopology,
    branch_lengths: Array,
    partition: LikelihoodPartition,
    /,
) -> tuple[Array, Array, Array]:
    state_count = partition.model.state_count
    identity = jnp.eye(state_count, dtype=partition.model.rate_matrix.dtype)
    constant_partials = jnp.broadcast_to(
        identity[:, None, :], (state_count, topology.tip_count, state_count)
    )
    constant_logs, transition_valid, minimum_scale = _mixture_pruning(
        topology, constant_partials, branch_lengths, partition
    )
    constant_probability = jnp.sum(jnp.exp(constant_logs))
    variable_probability = 1.0 - constant_probability
    return variable_probability, transition_valid, minimum_scale


def felsenstein_pruning(
    topology: TreeTopology,
    tip_partials: ArrayLike,
    branch_lengths: ArrayLike,
    partitions: tuple[LikelihoodPartition, ...],
    /,
    *,
    pattern_weights: ArrayLike | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> FelsensteinPruningResult:
    """Evaluate a scaled finite-state fixed-tree likelihood.

    ``tip_partials`` has shape ``(patterns, tips, states)``. Ambiguities are
    represented by multiple positive states and missing observations by an
    all-ones state vector. Partitions must cover every pattern exactly once.
    ``ascertainment='variable'`` applies the exact Lewis correction by summing
    all constant-state patterns under that partition's model and rate mixture.
    """

    if not isinstance(topology, TreeTopology):
        raise TypeError("topology must be a TreeTopology.")
    values = jnp.asarray(tip_partials)
    lengths = jnp.asarray(branch_lengths)
    if values.ndim != 3:
        raise ValueError("tip_partials must have shape (patterns, tips, states).")
    pattern_count, tip_count, state_count = values.shape
    if tip_count != topology.tip_count:
        raise ValueError("tip_partials tip axis does not match topology.tip_indices.")
    if lengths.shape != (topology.node_count,):
        raise ValueError("branch_lengths must have one entry per tree node.")
    if not partitions:
        raise ValueError("At least one likelihood partition is required.")
    for partition in partitions:
        if not isinstance(partition, LikelihoodPartition):
            raise TypeError("partitions must contain LikelihoodPartition values.")
        if partition.pattern_mask.shape != (pattern_count,):
            raise ValueError("Every partition mask must match the pattern count.")
        if partition.model.state_count != state_count:
            raise ValueError("Every partition model must match the tip state count.")

    weights = (
        jnp.ones((pattern_count,), dtype=values.dtype)
        if pattern_weights is None
        else jnp.asarray(pattern_weights, dtype=values.dtype)
    )
    if weights.shape != (pattern_count,):
        raise ValueError("pattern_weights must have one entry per pattern.")
    coverage_count = jnp.sum(
        jnp.stack([partition.pattern_mask for partition in partitions]), axis=0
    )
    partition_coverage = jnp.all(coverage_count == 1)
    tip_valid = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(values >= 0.0)
        & jnp.all(jnp.any(values > 0.0, axis=-1))
    )
    root = jax.lax.stop_gradient(topology.root_index)
    branch_finite_nonnegative = jnp.all(jnp.isfinite(lengths)) & jnp.all(lengths >= 0.0)
    root_branch_zero = lengths[root] == 0.0
    branch_valid = branch_finite_nonnegative & root_branch_zero
    weights_valid = jnp.all(jnp.isfinite(weights)) & jnp.all(weights >= 0.0)
    partitions_valid = jnp.all(jnp.stack([partition.valid for partition in partitions]))

    pattern_logs = jnp.zeros((pattern_count,), dtype=jnp.result_type(values, lengths))
    partition_logs = []
    partition_weights = []
    ascertainment_probabilities = []
    transition_valid = jnp.asarray(True)
    minimum_scale = jnp.asarray(jnp.inf, dtype=pattern_logs.dtype)
    ascertainment_valid = jnp.asarray(True)
    for partition in partitions:
        local_logs, local_transition_valid, local_minimum = _mixture_pruning(
            topology, values, lengths, partition
        )
        if partition.ascertainment == "variable":
            ascertainment_probability, correction_transition_valid, correction_minimum = (
                _variable_ascertainment_probability(topology, lengths, partition)
            )
            correction_valid = jnp.isfinite(ascertainment_probability) & (
                ascertainment_probability > 0.0
            )
            safe_ascertainment = jnp.where(
                correction_valid, ascertainment_probability, 1.0
            )
            local_logs = local_logs - jnp.log(safe_ascertainment)
            local_transition_valid = local_transition_valid & correction_transition_valid
            local_minimum = jnp.minimum(local_minimum, correction_minimum)
            ascertainment_valid = ascertainment_valid & correction_valid
        else:
            ascertainment_probability = jnp.asarray(1.0, dtype=pattern_logs.dtype)
        selected = partition.pattern_mask
        pattern_logs = jnp.where(selected, local_logs, pattern_logs)
        local_pattern_weights = jnp.where(selected, weights, 0.0)
        partition_logs.append(
            jnp.sum(
                jnp.where(
                    local_pattern_weights > 0.0,
                    local_pattern_weights * local_logs,
                    0.0,
                )
            )
        )
        partition_weights.append(jnp.sum(local_pattern_weights))
        ascertainment_probabilities.append(ascertainment_probability)
        transition_valid = transition_valid & local_transition_valid
        minimum_scale = jnp.minimum(minimum_scale, local_minimum)

    weighted_logs = jnp.where(weights > 0.0, weights * pattern_logs, 0.0)
    partition_log_array = jnp.stack(partition_logs)
    admissible_patterns = jnp.all(
        (~jnp.isnan(pattern_logs) & (pattern_logs != jnp.inf)) | (weights == 0.0)
    )
    finite_patterns = jnp.all(jnp.isfinite(pattern_logs) | (weights == 0.0))
    valid = (
        topology.valid
        & branch_valid
        & tip_valid
        & weights_valid
        & partition_coverage
        & partitions_valid
        & transition_valid
        & ascertainment_valid
        & admissible_patterns
    )
    status = jnp.where(
        ~topology.valid,
        int(PruningStatus.INVALID_TOPOLOGY),
        jnp.where(
            ~branch_valid,
            int(PruningStatus.INVALID_BRANCH_LENGTH),
            jnp.where(
                ~tip_valid,
                int(PruningStatus.INVALID_TIP_PARTIAL),
                jnp.where(
                    ~weights_valid,
                    int(PruningStatus.INVALID_PATTERN_WEIGHT),
                    jnp.where(
                        ~partition_coverage | ~partitions_valid,
                        int(PruningStatus.INVALID_PARTITION),
                        jnp.where(
                            ~transition_valid,
                            int(PruningStatus.NONSTOCHASTIC_TRANSITION),
                            jnp.where(
                                ~ascertainment_valid,
                                int(PruningStatus.IMPOSSIBLE_ASCERTAINMENT),
                                jnp.where(
                                    admissible_patterns,
                                    int(PruningStatus.SUCCESS),
                                    int(PruningStatus.NONFINITE_LIKELIHOOD),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = PruningEvidence(
        topology_valid=topology.valid,
        branch_lengths_valid=branch_finite_nonnegative,
        root_branch_zero=root_branch_zero,
        tip_partials_valid=tip_valid,
        pattern_weights_valid=weights_valid,
        partition_coverage=partition_coverage,
        partitions_valid=partitions_valid,
        transition_matrices_stochastic=transition_valid,
        ascertainment_probabilities=jnp.stack(ascertainment_probabilities),
        minimum_scaling_factor=minimum_scale,
        finite_pattern_likelihoods=jnp.isfinite(pattern_logs),
    )
    return FelsensteinPruningResult(
        log_likelihood=jnp.sum(weighted_logs),
        pattern_log_likelihood=pattern_logs,
        weighted_pattern_log_likelihood=weighted_logs,
        partition_log_likelihood=partition_log_array,
        partition_pattern_weight=jnp.stack(partition_weights),
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_pruning_contract()
        if method_contract is None
        else method_contract,
    )


pruning_log_likelihood = felsenstein_pruning


__all__ = [
    "AscertainmentKind",
    "FelsensteinPruningResult",
    "LikelihoodPartition",
    "PruningEvidence",
    "PruningStatus",
    "TipPartialEncodingResult",
    "TipPartialEvidence",
    "felsenstein_pruning",
    "pruning_log_likelihood",
    "tip_partials_from_sequence",
]
