#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._pruning import (
    _category_pruning,
    felsenstein_pruning,
    FelsensteinPruningResult,
    LikelihoodPartition,
)
from ._substitution import FiniteStateSubstitutionModel
from ._tree import TreeTopology


class AncestralMarginalEvidence(StrictModule):
    """Normalization evidence for exact node-state posteriors."""

    pruning_valid: Array
    finite: Array
    normalized: Array
    minimum_normalizer: Array
    partition_coverage: Array


class AncestralMarginalResult(StrictModule):
    """Exact per-pattern posterior state probabilities at every tree node."""

    marginals: Array
    valid: Array
    status: Array
    evidence: AncestralMarginalEvidence
    likelihood: FelsensteinPruningResult
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _ancestral_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "fixed_tree_ancestral_state_marginals",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            "Node-state probabilities are conditioned on the supplied fixed tree, "
            "finite-state model, rate mixture, root law, and observed tip state sets."
        ),
        truncation_statement="No nodes, states, patterns, or rate categories are truncated.",
        capacity_semantics="The validated topology child capacity is traversed in full.",
        assumptions=("The Felsenstein conditional-independence factorization holds.",),
        nondifferentiable_outputs=("valid", "status", "evidence"),
    )


def _normalize_rows(values: Array, /) -> tuple[Array, Array]:
    normalizer = jnp.sum(values, axis=-1, keepdims=True)
    positive = jnp.isfinite(normalizer) & (normalizer > 0.0)
    safe = jnp.where(positive, normalizer, 1.0)
    return values / safe, jnp.squeeze(normalizer, axis=-1)


def _category_marginals(
    topology: TreeTopology,
    tip_partials: Array,
    branch_lengths: Array,
    model: FiniteStateSubstitutionModel,
    root_distribution: Array,
    rate: Array,
    /,
) -> tuple[Array, Array, Array]:
    category_log_likelihood, upward, _, _, _ = _category_pruning(
        topology,
        tip_partials,
        branch_lengths,
        model,
        root_distribution,
        rate,
    )
    pattern_count = int(tip_partials.shape[0])
    state_count = model.state_count
    dtype = upward.dtype
    downward = jnp.zeros((topology.node_count, pattern_count, state_count), dtype=dtype)
    root = jax.lax.stop_gradient(topology.root_index)
    root_message = jnp.broadcast_to(
        root_distribution.astype(dtype), (pattern_count, state_count)
    )
    root_message, _ = _normalize_rows(root_message)
    downward = downward.at[root].set(root_message)

    def parent_step(order_index, messages):
        parent = jax.lax.stop_gradient(topology.preorder[order_index])

        def child_step(child_slot, local_messages):
            active_child = topology.child_mask[parent, child_slot]

            def include_child(messages):
                child = jax.lax.stop_gradient(topology.child_indices[parent, child_slot])
                context = messages[parent]

                def sibling_step(sibling_slot, running):
                    active_sibling = topology.child_mask[parent, sibling_slot] & (
                        sibling_slot != child_slot
                    )

                    def include_sibling(values):
                        sibling = jax.lax.stop_gradient(
                            topology.child_indices[parent, sibling_slot]
                        )
                        sibling_matrix = model.transition_matrix(
                            branch_lengths[sibling] * rate
                        ).astype(dtype)
                        contribution = upward[sibling] @ jnp.swapaxes(
                            sibling_matrix, -1, -2
                        )
                        return values * contribution

                    return jax.lax.cond(
                        active_sibling, include_sibling, lambda values: values, running
                    )

                context = jax.lax.fori_loop(
                    0, topology.child_capacity, sibling_step, context
                )
                context_scale = jnp.max(context, axis=-1, keepdims=True)
                context = context / jnp.where(context_scale > 0.0, context_scale, 1.0)
                child_matrix = model.transition_matrix(
                    branch_lengths[child] * rate
                ).astype(dtype)
                child_message = context @ child_matrix
                child_message, _ = _normalize_rows(child_message)
                return messages.at[child].set(child_message)

            return jax.lax.cond(
                active_child, include_child, lambda messages: messages, local_messages
            )

        return jax.lax.fori_loop(0, topology.child_capacity, child_step, messages)

    downward = jax.lax.fori_loop(0, topology.node_count, parent_step, downward)
    marginals, normalizers = _normalize_rows(upward * downward)
    return marginals, category_log_likelihood, normalizers


def ancestral_marginals(
    topology: TreeTopology,
    tip_partials: ArrayLike,
    branch_lengths: ArrayLike,
    partitions: tuple[LikelihoodPartition, ...],
    /,
    *,
    pattern_weights: ArrayLike | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> AncestralMarginalResult:
    """Run exact upward/downward finite-state inference on a fixed tree."""

    values = jnp.asarray(tip_partials)
    lengths = jnp.asarray(branch_lengths)
    likelihood = felsenstein_pruning(
        topology,
        values,
        lengths,
        partitions,
        pattern_weights=pattern_weights,
    )
    pattern_count, _, state_count = values.shape
    marginals = jnp.zeros(
        (pattern_count, topology.node_count, state_count),
        dtype=jnp.result_type(values, lengths),
    )
    minimum_normalizer = jnp.asarray(jnp.inf, dtype=marginals.dtype)

    for partition in partitions:
        category_marginals, category_logs, normalizers = jax.vmap(
            lambda rate: _category_marginals(
                topology,
                values,
                lengths,
                partition.model,
                partition.root_distribution,
                rate,
            )
        )(partition.rate_mixture.rates)
        log_weights = jnp.where(
            partition.rate_mixture.weights > 0.0,
            jnp.log(
                jnp.maximum(
                    partition.rate_mixture.weights,
                    jnp.finfo(category_logs.dtype).tiny,
                )
            ),
            -jnp.inf,
        )
        category_posterior = jax.nn.softmax(category_logs + log_weights[:, None], axis=0)
        mixed = jnp.sum(
            category_marginals * category_posterior[:, None, :, None],
            axis=0,
        )
        mixed = jnp.swapaxes(mixed, 0, 1)
        marginals = jnp.where(partition.pattern_mask[:, None, None], mixed, marginals)
        minimum_normalizer = jnp.minimum(
            minimum_normalizer,
            jnp.min(jnp.where(normalizers > 0.0, normalizers, jnp.inf)),
        )

    finite = jnp.all(jnp.isfinite(marginals))
    tolerance = 512.0 * jnp.finfo(marginals.dtype).eps
    normalized = jnp.all(jnp.abs(jnp.sum(marginals, axis=-1) - 1.0) <= tolerance)
    valid = likelihood.valid & finite & normalized
    status = jnp.where(valid, 0, jnp.where(~likelihood.valid, 1, 2)).astype(jnp.int32)
    evidence = AncestralMarginalEvidence(
        pruning_valid=likelihood.valid,
        finite=finite,
        normalized=normalized,
        minimum_normalizer=minimum_normalizer,
        partition_coverage=likelihood.evidence.partition_coverage,
    )
    return AncestralMarginalResult(
        marginals=marginals,
        valid=valid,
        status=status,
        evidence=evidence,
        likelihood=likelihood,
        method_contract=_ancestral_contract()
        if method_contract is None
        else method_contract,
    )


__all__ = [
    "AncestralMarginalEvidence",
    "AncestralMarginalResult",
    "ancestral_marginals",
]
