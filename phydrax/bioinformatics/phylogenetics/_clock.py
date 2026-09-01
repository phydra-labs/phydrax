#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal

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
    felsenstein_pruning,
    FelsensteinPruningResult,
    LikelihoodPartition,
)
from ._tree import TreeTopology


ClockKind = Literal["strict", "relaxed"]


class ClockStatus(IntEnum):
    SUCCESS = 0
    INVALID_TOPOLOGY = 1
    NONFINITE_TIME = 2
    NONPOSITIVE_DURATION = 3
    NONPOSITIVE_RATE = 4
    NONFINITE_BRANCH_LENGTH = 5
    LIKELIHOOD_FAILURE = 6


class ClockEvidence(StrictModule):
    """Temporal ordering and rate evidence for clock-derived branches."""

    topology_valid: Array
    times_finite: Array
    durations_positive: Array
    rates_finite: Array
    rates_positive: Array
    branch_lengths_finite: Array
    minimum_duration: Array
    minimum_rate: Array


class ClockEvaluation(StrictModule):
    """Strict- or relaxed-clock conversion from node times to branch lengths."""

    node_times: Array
    durations: Array
    branch_rates: Array
    branch_lengths: Array
    valid: Array
    status: Array
    evidence: ClockEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    clock_kind: ClockKind = eqx.field(static=True)


class ClockLikelihoodEvidence(StrictModule):
    """Joint validity of a clock conversion and its pruning likelihood."""

    clock_valid: Array
    likelihood_valid: Array


class ClockLikelihoodResult(StrictModule):
    """Fixed-topology likelihood evaluated under a supplied phylogenetic clock."""

    log_likelihood: Array
    clock: ClockEvaluation
    likelihood: FelsensteinPruningResult
    valid: Array
    status: Array
    evidence: ClockLikelihoodEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _clock_contract(kind: ClockKind, /) -> BioinformaticsMethodContract:
    rate_statement = (
        "one shared positive substitution rate"
        if kind == "strict"
        else "one supplied positive substitution rate per non-root branch"
    )
    return BioinformaticsMethodContract(
        f"{kind}_phylogenetic_clock_evaluation",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.ARRAY,
        conditioning_statement=(
            f"Branch lengths are exact products of parent-child time differences and {rate_statement}."
        ),
        truncation_statement="No nodes or branches are truncated.",
        capacity_semantics="The branch array has exactly one entry per topology node; the root entry is zero.",
        assumptions=("Every parent time is strictly older than each child time.",),
        nondifferentiable_outputs=("valid", "status", "evidence"),
    )


def _clock_likelihood_contract(kind: ClockKind, /) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        f"fixed_tree_{kind}_clock_felsenstein_likelihood",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.SCALAR,
        conditioning_statement=(
            "Likelihood is conditioned on the topology, node times, clock rates, "
            "partitions, observations, and pattern weights."
        ),
        truncation_statement="No nodes, branches, patterns, states, or rate categories are truncated.",
        capacity_semantics="Topology and partition capacities are evaluated in full.",
        assumptions=("Clock-derived branch lengths parameterize each partition CTMC.",),
        nondifferentiable_outputs=("valid", "status", "evidence"),
    )


def _clock_evaluation(
    topology: TreeTopology,
    node_times: ArrayLike,
    branch_rates: Array,
    kind: ClockKind,
    /,
    *,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ClockEvaluation:
    if not isinstance(topology, TreeTopology):
        raise TypeError("topology must be a TreeTopology.")
    times = jnp.asarray(node_times)
    if times.shape != (topology.node_count,):
        raise ValueError("node_times must have one value per topology node.")
    if branch_rates.shape != (topology.node_count,):
        raise ValueError("branch_rates must have one value per topology node.")
    parent = jax.lax.stop_gradient(topology.parent_indices)
    root = jax.lax.stop_gradient(topology.root_index)
    nonroot = jnp.arange(topology.node_count, dtype=jnp.int32) != root
    safe_parent = jnp.where(nonroot, parent, root)
    durations = times[safe_parent] - times
    durations = durations.at[root].set(0.0)
    effective_rates = branch_rates.at[root].set(0.0)
    branch_lengths = durations * effective_rates

    times_finite = jnp.all(jnp.isfinite(times))
    durations_positive = jnp.all(jnp.where(nonroot, durations > 0.0, True))
    rates_finite = jnp.all(jnp.where(nonroot, jnp.isfinite(effective_rates), True))
    rates_positive = jnp.all(jnp.where(nonroot, effective_rates > 0.0, True))
    branches_finite = jnp.all(jnp.isfinite(branch_lengths))
    valid = (
        topology.valid
        & times_finite
        & durations_positive
        & rates_finite
        & rates_positive
        & branches_finite
    )
    status = jnp.where(
        ~topology.valid,
        int(ClockStatus.INVALID_TOPOLOGY),
        jnp.where(
            ~times_finite,
            int(ClockStatus.NONFINITE_TIME),
            jnp.where(
                ~durations_positive,
                int(ClockStatus.NONPOSITIVE_DURATION),
                jnp.where(
                    ~rates_finite | ~rates_positive,
                    int(ClockStatus.NONPOSITIVE_RATE),
                    jnp.where(
                        branches_finite,
                        int(ClockStatus.SUCCESS),
                        int(ClockStatus.NONFINITE_BRANCH_LENGTH),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    minimum_duration = jnp.min(jnp.where(nonroot, durations, jnp.inf))
    minimum_rate = jnp.min(jnp.where(nonroot, effective_rates, jnp.inf))
    evidence = ClockEvidence(
        topology_valid=topology.valid,
        times_finite=times_finite,
        durations_positive=durations_positive,
        rates_finite=rates_finite,
        rates_positive=rates_positive,
        branch_lengths_finite=branches_finite,
        minimum_duration=minimum_duration,
        minimum_rate=minimum_rate,
    )
    return ClockEvaluation(
        node_times=times,
        durations=durations,
        branch_rates=effective_rates,
        branch_lengths=branch_lengths,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_clock_contract(kind)
        if method_contract is None
        else method_contract,
        clock_kind=kind,
    )


def strict_clock(
    topology: TreeTopology,
    node_times: ArrayLike,
    rate: ArrayLike,
    /,
    *,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ClockEvaluation:
    """Evaluate one shared-rate strict molecular clock."""

    rate_value = jnp.asarray(rate, dtype=jnp.asarray(node_times).dtype)
    if rate_value.shape != ():
        raise ValueError("rate must be scalar.")
    rates = jnp.broadcast_to(rate_value, (topology.node_count,))
    return _clock_evaluation(
        topology,
        node_times,
        rates,
        "strict",
        method_contract=method_contract,
    )


def relaxed_clock(
    topology: TreeTopology,
    node_times: ArrayLike,
    branch_rates: ArrayLike,
    /,
    *,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ClockEvaluation:
    """Evaluate supplied per-branch rates for an uncorrelated relaxed clock."""

    rates = jnp.asarray(branch_rates, dtype=jnp.asarray(node_times).dtype)
    return _clock_evaluation(
        topology,
        node_times,
        rates,
        "relaxed",
        method_contract=method_contract,
    )


def _clock_likelihood(
    clock: ClockEvaluation,
    topology: TreeTopology,
    tip_partials: ArrayLike,
    partitions: tuple[LikelihoodPartition, ...],
    kind: ClockKind,
    /,
    *,
    pattern_weights: ArrayLike | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ClockLikelihoodResult:
    likelihood = felsenstein_pruning(
        topology,
        tip_partials,
        clock.branch_lengths,
        partitions,
        pattern_weights=pattern_weights,
    )
    valid = clock.valid & likelihood.valid
    status = jnp.where(
        ~clock.valid,
        clock.status,
        jnp.where(
            likelihood.valid,
            int(ClockStatus.SUCCESS),
            int(ClockStatus.LIKELIHOOD_FAILURE),
        ),
    ).astype(jnp.int32)
    evidence = ClockLikelihoodEvidence(
        clock_valid=clock.valid,
        likelihood_valid=likelihood.valid,
    )
    return ClockLikelihoodResult(
        log_likelihood=likelihood.log_likelihood,
        clock=clock,
        likelihood=likelihood,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_clock_likelihood_contract(kind)
        if method_contract is None
        else method_contract,
    )


def strict_clock_likelihood(
    topology: TreeTopology,
    tip_partials: ArrayLike,
    node_times: ArrayLike,
    rate: ArrayLike,
    partitions: tuple[LikelihoodPartition, ...],
    /,
    *,
    pattern_weights: ArrayLike | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ClockLikelihoodResult:
    """Evaluate a fixed-tree likelihood under a strict molecular clock."""

    clock = strict_clock(topology, node_times, rate)
    return _clock_likelihood(
        clock,
        topology,
        tip_partials,
        partitions,
        "strict",
        pattern_weights=pattern_weights,
        method_contract=method_contract,
    )


def relaxed_clock_likelihood(
    topology: TreeTopology,
    tip_partials: ArrayLike,
    node_times: ArrayLike,
    branch_rates: ArrayLike,
    partitions: tuple[LikelihoodPartition, ...],
    /,
    *,
    pattern_weights: ArrayLike | None = None,
    method_contract: BioinformaticsMethodContract | None = None,
) -> ClockLikelihoodResult:
    """Evaluate a fixed-tree likelihood under supplied relaxed-clock rates."""

    clock = relaxed_clock(topology, node_times, branch_rates)
    return _clock_likelihood(
        clock,
        topology,
        tip_partials,
        partitions,
        "relaxed",
        pattern_weights=pattern_weights,
        method_contract=method_contract,
    )


evaluate_strict_clock = strict_clock
evaluate_relaxed_clock = relaxed_clock


__all__ = [
    "ClockEvaluation",
    "ClockEvidence",
    "ClockKind",
    "ClockLikelihoodEvidence",
    "ClockLikelihoodResult",
    "ClockStatus",
    "evaluate_relaxed_clock",
    "evaluate_strict_clock",
    "relaxed_clock",
    "relaxed_clock_likelihood",
    "strict_clock",
    "strict_clock_likelihood",
]
