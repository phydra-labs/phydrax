#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class NestedSamplingDiagnostics(StrictModule):
    """Observable constrained-sampling, ordering, and lineage checks."""

    insertion_ranks: Array
    insertion_rank_pvalue: Array
    rolling_insertion_rank_pvalues: Array
    likelihood_monotonic: Array
    constraints_satisfied: Array
    initial_finite_fraction: Array
    inner_acceptance_rate: Array
    expansion_cap_fraction: Array
    shrinkage_cap_fraction: Array
    zero_movement_fraction: Array
    unique_lineage_count: Array
    effective_lineage_count: Array
    covariance_rank: Array
    covariance_condition: Array
    failures: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        insertion_ranks: Array,
        insertion_rank_pvalue: Array,
        rolling_insertion_rank_pvalues: Array,
        likelihood_monotonic: Array,
        constraints_satisfied: Array,
        initial_finite_fraction: Array,
        inner_acceptance_rate: Array,
        expansion_cap_fraction: Array,
        shrinkage_cap_fraction: Array,
        zero_movement_fraction: Array,
        unique_lineage_count: Array,
        effective_lineage_count: Array,
        covariance_rank: Array,
        covariance_condition: Array,
        failures: tuple[str, ...],
    ):
        self.insertion_ranks = jnp.asarray(insertion_ranks, dtype=jnp.int32)
        self.insertion_rank_pvalue = jnp.asarray(insertion_rank_pvalue)
        self.rolling_insertion_rank_pvalues = jnp.asarray(rolling_insertion_rank_pvalues)
        self.likelihood_monotonic = jnp.asarray(likelihood_monotonic, dtype=bool)
        self.constraints_satisfied = jnp.asarray(constraints_satisfied, dtype=bool)
        self.initial_finite_fraction = jnp.asarray(initial_finite_fraction)
        self.inner_acceptance_rate = jnp.asarray(inner_acceptance_rate)
        self.expansion_cap_fraction = jnp.asarray(expansion_cap_fraction)
        self.shrinkage_cap_fraction = jnp.asarray(shrinkage_cap_fraction)
        self.zero_movement_fraction = jnp.asarray(zero_movement_fraction)
        self.unique_lineage_count = jnp.asarray(unique_lineage_count, dtype=jnp.int32)
        self.effective_lineage_count = jnp.asarray(effective_lineage_count)
        self.covariance_rank = jnp.asarray(covariance_rank, dtype=jnp.int32)
        self.covariance_condition = jnp.asarray(covariance_condition)
        self.failures = tuple(failures)

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        """Return scalar diagnostics and exact failure labels."""
        return {
            "passed": self.passed,
            "failures": self.failures,
            "insertion_rank_pvalue": float(self.insertion_rank_pvalue),
            "likelihood_monotonic": bool(self.likelihood_monotonic),
            "constraints_satisfied": bool(self.constraints_satisfied),
            "initial_finite_fraction": float(self.initial_finite_fraction),
            "inner_acceptance_rate": float(self.inner_acceptance_rate),
            "expansion_cap_fraction": float(self.expansion_cap_fraction),
            "shrinkage_cap_fraction": float(self.shrinkage_cap_fraction),
            "zero_movement_fraction": float(self.zero_movement_fraction),
            "unique_lineage_count": int(self.unique_lineage_count),
            "effective_lineage_count": float(self.effective_lineage_count),
            "covariance_rank": int(self.covariance_rank),
            "covariance_condition": float(self.covariance_condition),
        }


def insertion_rank_pvalue(ranks: Array, num_live: int, /) -> Array:
    """Pearson cross-check for discrete-uniform nested insertion ranks."""
    count = int(num_live)
    if count < 2:
        raise ValueError("num_live must be at least two.")
    values = jnp.asarray(ranks, dtype=jnp.int32).reshape((-1,))
    if int(values.size) == 0:
        return jnp.asarray(jnp.nan)
    if bool(jnp.any((values < 0) | (values >= count))):
        raise ValueError("Insertion ranks must lie in [0, num_live).")
    observed = jnp.bincount(values, length=count)
    expected = jnp.asarray(values.size / count, dtype=float)
    statistic = jnp.sum((observed - expected) ** 2 / expected)
    return jsp.special.gammaincc(0.5 * (count - 1), 0.5 * statistic)


def rolling_insertion_rank_pvalues(
    ranks: Array,
    num_live: int,
    /,
    *,
    window: int | None = None,
) -> Array:
    """Evaluate non-overlapping insertion-rank windows for drift detection."""
    values = jnp.asarray(ranks, dtype=jnp.int32).reshape((-1,))
    width = max(4 * int(num_live), 100) if window is None else int(window)
    if width <= 0:
        raise ValueError("window must be positive.")
    if int(values.size) < width:
        return jnp.empty((0,), dtype=float)
    return jnp.stack(
        [
            insertion_rank_pvalue(values[start : start + width], num_live)
            for start in range(0, int(values.size) - width + 1, width)
        ]
    )


def build_nested_diagnostics(
    *,
    dead_log_likelihood: Array,
    dead_birth_log_likelihood: Array,
    insertion_ranks: Array,
    inner_accepted: Array,
    num_expansions: Array,
    num_shrink: Array,
    max_expansions: int,
    max_shrinkage: int,
    initial_log_likelihood: Array,
    sample_ids: Array,
    posterior_log_weights: Array,
    num_live: int,
    quadrature_valid: Array,
    final_live_positions: PyTree[Any],
) -> NestedSamplingDiagnostics:
    """Assemble post-run diagnostics without altering sampler output."""
    deaths = jnp.asarray(dead_log_likelihood)
    births = jnp.asarray(dead_birth_log_likelihood)
    ranks = jnp.asarray(insertion_ranks, dtype=jnp.int32).reshape((-1,))
    accepted = jnp.asarray(inner_accepted, dtype=bool)
    expansions = jnp.asarray(num_expansions)
    shrinkages = jnp.asarray(num_shrink)
    initial = jnp.asarray(initial_log_likelihood)
    ids = jnp.asarray(sample_ids, dtype=jnp.int32)
    log_weights = jnp.asarray(posterior_log_weights)

    monotonic = jnp.all(jnp.diff(deaths) >= 0.0)
    constraints = jnp.all(jnp.isnan(births) | (deaths > births))
    finite_fraction = jnp.mean(jnp.isfinite(initial).astype(float))
    acceptance_rate = jnp.mean(accepted.astype(float))
    zero_movement = jnp.mean((~jnp.any(accepted, axis=-1)).astype(float))
    expansion_cap = jnp.mean((expansions >= int(max_expansions)).astype(float))
    shrinkage_cap = jnp.mean((shrinkages >= int(max_shrinkage)).astype(float))
    rank_pvalue = insertion_rank_pvalue(ranks, num_live)
    rolling = rolling_insertion_rank_pvalues(ranks, num_live)

    lineage_count = int(num_live)
    lineage_mass = jnp.zeros((lineage_count,), dtype=log_weights.dtype)
    lineage_mass = lineage_mass.at[ids].add(jnp.exp(log_weights))
    active_lineages = lineage_mass > 0.0
    lineage_entropy = -jnp.sum(
        jnp.where(active_lineages, lineage_mass * jnp.log(lineage_mass), 0.0)
    )
    effective_lineages = jnp.exp(lineage_entropy)
    live_leaves = jax.tree_util.tree_leaves(final_live_positions)
    live_count = int(live_leaves[0].shape[0])
    live_matrix = jnp.concatenate(
        tuple(jnp.asarray(leaf).reshape((live_count, -1)) for leaf in live_leaves),
        axis=1,
    )
    centered = live_matrix - jnp.mean(live_matrix, axis=0, keepdims=True)
    covariance = (
        centered.T
        @ centered
        / jnp.asarray(
            live_count,
            dtype=live_matrix.dtype,
        )
    )
    eigenvalues = jnp.maximum(jnp.linalg.eigvalsh(covariance), 0.0)
    largest = jnp.max(eigenvalues)
    dimension = int(eigenvalues.size)
    tolerance = (
        jnp.finfo(live_matrix.dtype).eps
        * max(live_count, dimension)
        * jnp.maximum(largest, jnp.finfo(live_matrix.dtype).tiny)
    )
    covariance_rank = jnp.sum(eigenvalues > tolerance)
    smallest = jnp.min(jnp.where(eigenvalues > tolerance, eigenvalues, jnp.inf))
    covariance_condition = jnp.where(
        covariance_rank == dimension,
        largest / smallest,
        jnp.inf,
    )

    failures: list[str] = []
    if not bool(monotonic):
        failures.append("dead likelihoods are not monotonic")
    if not bool(constraints):
        failures.append("one or more particles violate their birth likelihood")
    if not bool(quadrature_valid):
        failures.append("evidence quadrature is invalid")
    if bool(zero_movement >= 1.0):
        failures.append("every replacement exhausted its inner chain without moving")
    if int(ranks.size) >= 4 * int(num_live) and bool(rank_pvalue < 1e-6):
        failures.append("insertion ranks reject constrained-prior uniformity")

    return NestedSamplingDiagnostics(
        insertion_ranks=ranks,
        insertion_rank_pvalue=rank_pvalue,
        rolling_insertion_rank_pvalues=rolling,
        likelihood_monotonic=monotonic,
        constraints_satisfied=constraints,
        initial_finite_fraction=finite_fraction,
        inner_acceptance_rate=acceptance_rate,
        expansion_cap_fraction=expansion_cap,
        shrinkage_cap_fraction=shrinkage_cap,
        zero_movement_fraction=zero_movement,
        unique_lineage_count=jnp.sum(active_lineages),
        effective_lineage_count=effective_lineages,
        covariance_rank=covariance_rank,
        covariance_condition=covariance_condition,
        failures=tuple(failures),
    )


__all__ = [
    "NestedSamplingDiagnostics",
    "build_nested_diagnostics",
    "insertion_rank_pvalue",
    "rolling_insertion_rank_pvalues",
]
