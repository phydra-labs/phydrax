#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._belief_propagation import SumProductBeliefPropagationResult
from ._model import (
    DiscreteFactorGraph,
    EnumeratedFactorGroup,
    factor_graph_contains,
    factor_graph_log_score,
    factor_group_cardinality_signature,
    factor_group_dense_tables,
    pack_assignments,
)
from ._types import ExactFactorGraphResult


ExactNormalizerResult: TypeAlias = (
    ExactFactorGraphResult | SumProductBeliefPropagationResult
)


class FactorGraphTrainingDiagnostics(StrictModule):
    """Finite positive/negative score evidence for one factor-graph objective."""

    objective: Array
    positive_mean_log_score: Array
    negative_mean_log_score: Array
    positive_finite_fraction: Array
    negative_finite_fraction: Array
    exact_normalizer: bool = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return (
            jnp.isfinite(self.objective)
            & (self.positive_finite_fraction == 1.0)
            & (self.negative_finite_fraction == 1.0)
        )


def _exact_log_normalizer(
    graph: DiscreteFactorGraph,
    result: ExactNormalizerResult,
    /,
) -> Array:
    if not isinstance(
        result,
        (ExactFactorGraphResult, SumProductBeliefPropagationResult),
    ):
        raise TypeError(
            "normalizer must be an exact enumeration or forest sum-product result."
        )
    if result.provenance.structure_id != graph.structure_id:
        raise ValueError("Exact normalizer belongs to another graph structure.")
    if isinstance(result, ExactFactorGraphResult):
        if not bool(result.provenance.exact):
            raise ValueError("ExactFactorGraphResult does not carry exact provenance.")
        expected = tuple(
            factor_group_dense_tables(graph, index)
            for index in range(len(graph.factor_groups))
        )
        valid = result.successful
    elif isinstance(result, SumProductBeliefPropagationResult):
        if not result.log_normalizer_exact:
            raise ValueError("Loopy Bethe normalizers cannot enter the exact likelihood.")
        expected = tuple(
            group.log_potentials
            if isinstance(group, EnumeratedFactorGroup)
            else factor_group_dense_tables(graph, index)
            for index, group in enumerate(graph.factor_groups)
        )
        valid = result.successful
    if len(expected) != len(result.factor_tables) or any(
        left.shape != right.shape for left, right in zip(expected, result.factor_tables)
    ):
        raise ValueError(
            "Exact normalizer numeric factor layout does not match the graph."
        )
    numeric_match = jnp.asarray(True)
    for current, normalized in zip(expected, result.factor_tables):
        numeric_match = numeric_match & jnp.array_equal(current, normalized)
    value = eqx.error_if(
        result.log_normalizer,
        ~numeric_match,
        "Exact normalizer numeric factors do not match the supplied graph.",
    )
    return jnp.where(valid, value, jnp.nan)


def exact_factor_graph_negative_log_likelihood(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    normalizer: ExactNormalizerResult,
    /,
) -> tuple[Array, FactorGraphTrainingDiagnostics]:
    """Return exact mean negative log likelihood from a certified normalizer."""
    states = pack_assignments(graph, assignments)
    if states.ndim == 1:
        states = states[None, :]
    scores = factor_graph_log_score(graph, states)
    log_normalizer = _exact_log_normalizer(graph, normalizer)
    objective = log_normalizer - jnp.mean(scores)
    diagnostics = FactorGraphTrainingDiagnostics(
        objective=objective,
        positive_mean_log_score=jnp.mean(scores),
        negative_mean_log_score=jnp.asarray(jnp.nan, dtype=scores.dtype),
        positive_finite_fraction=jnp.mean(jnp.isfinite(scores)),
        negative_finite_fraction=jnp.asarray(1.0, dtype=scores.dtype),
        exact_normalizer=True,
    )
    return objective, diagnostics


def contrastive_divergence_loss(
    graph: DiscreteFactorGraph,
    positive_assignments: ArrayLike,
    negative_assignments: ArrayLike,
    /,
    *,
    stop_sample_gradient: bool = True,
) -> tuple[Array, FactorGraphTrainingDiagnostics]:
    """Return the standard positive/negative phase score-difference objective."""
    positive = pack_assignments(graph, positive_assignments)
    negative = pack_assignments(graph, negative_assignments)
    if positive.ndim == 1:
        positive = positive[None, :]
    if negative.ndim == 1:
        negative = negative[None, :]
    if stop_sample_gradient:
        positive = jax.lax.stop_gradient(positive)
        negative = jax.lax.stop_gradient(negative)
    positive_scores = factor_graph_log_score(graph, positive)
    negative_scores = factor_graph_log_score(graph, negative)
    positive_mean = jnp.mean(positive_scores)
    negative_mean = jnp.mean(negative_scores)
    objective = -positive_mean + negative_mean
    diagnostics = FactorGraphTrainingDiagnostics(
        objective=objective,
        positive_mean_log_score=positive_mean,
        negative_mean_log_score=negative_mean,
        positive_finite_fraction=jnp.mean(jnp.isfinite(positive_scores)),
        negative_finite_fraction=jnp.mean(jnp.isfinite(negative_scores)),
        exact_normalizer=False,
    )
    return objective, diagnostics


def _configuration_indices(states: Array, signature: tuple[int, ...], /) -> Array:
    stride = prod(signature)
    strides: list[int] = []
    for cardinality in signature:
        stride //= cardinality
        strides.append(stride)
    return jnp.sum(states * jnp.asarray(strides, dtype=jnp.int32), axis=-1)


def factor_graph_moments(
    graph: DiscreteFactorGraph,
    assignments: ArrayLike,
    /,
) -> tuple[Array, ...]:
    """Return empirical factor-configuration probabilities for each factor group."""
    states = pack_assignments(graph, assignments)
    if states.ndim == 1:
        states = states[None, :]
    samples = states.reshape((-1, graph.num_variables))
    samples = eqx.error_if(
        samples,
        ~jnp.all(factor_graph_contains(graph, samples)),
        "assignments contain values outside graph support.",
    )
    count = samples.shape[0]
    outputs: list[Array] = []
    for group_index, scope in enumerate(graph.factor_scopes):
        signature = factor_group_cardinality_signature(graph, group_index)
        config_count = prod(signature)
        scope_states = samples[:, scope]
        factors: list[Array] = []
        for factor in range(scope.shape[0]):
            indices = _configuration_indices(scope_states[:, factor, :], signature)
            occurrences = jax.ops.segment_sum(
                jnp.ones((count,), dtype=jnp.float64),
                indices,
                num_segments=config_count,
            )
            factors.append((occurrences / max(count, 1)).reshape(signature))
        outputs.append(
            jnp.stack(factors)
            if factors
            else jnp.zeros((0,) + signature, dtype=jnp.float64)
        )
    return tuple(outputs)


__all__ = [
    "ExactNormalizerResult",
    "FactorGraphTrainingDiagnostics",
    "contrastive_divergence_loss",
    "exact_factor_graph_negative_log_likelihood",
    "factor_graph_moments",
]
