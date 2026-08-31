#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ._gibbs import (
    _conditional_logits,
    gibbs_sweep,
    GibbsState,
    GibbsTransitionInfo,
    PreparedChromaticGibbs,
)
from ._model import factor_graph_log_score, IsingFactorGroup
from ._types import GibbsTransitionStatus


_RANDOM_SCAN_ADDRESS = SampleAddress(
    "factor-graph",
    "random-scan-gibbs",
    target="site",
    role="conditional-sample",
)
_TEMPERING_ADDRESS = SampleAddress(
    "factor-graph",
    "parallel-tempering",
    target="swap",
    role="acceptance",
)


class GibbsScanPolicy(StrictModule):
    """Explicit systematic, random-site, or randomized-color Gibbs schedule."""

    kind: Literal["systematic", "random-scan", "randomized-colors"] = eqx.field(
        static=True
    )
    updates_per_sweep: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["systematic", "random-scan", "randomized-colors"] = "systematic",
        /,
        *,
        updates_per_sweep: int | None = None,
    ):
        if kind not in ("systematic", "random-scan", "randomized-colors"):
            raise ValueError("Unknown Gibbs scan policy.")
        count = None if updates_per_sweep is None else int(updates_per_sweep)
        if count is not None and count < 1:
            raise ValueError("updates_per_sweep must be positive.")
        self.kind = kind
        self.updates_per_sweep = count
        self.policy_id = f"gibbs-scan:{kind}:{count}"


class JointDiscreteBlock(StrictModule):
    """Explicit dependent block whose exact conditional is enumerated under a cap."""

    variables: tuple[int, ...] = eqx.field(static=True)
    maximum_configurations: int = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(self, variables, /, *, maximum_configurations: int = 4096):
        selected = tuple(int(value) for value in variables)
        if not selected or len(set(selected)) != len(selected) or min(selected) < 0:
            raise ValueError("Joint block variables must be unique and non-negative.")
        maximum = int(maximum_configurations)
        if maximum < 1:
            raise ValueError("maximum_configurations must be positive.")
        self.variables = selected
        self.maximum_configurations = maximum
        self.block_id = "joint-block:" + ",".join(str(value) for value in selected)


class ParallelTemperingState(StrictModule):
    """Replica positions and inverse-temperature identity for one base graph."""

    positions: Array
    inverse_temperatures: Array
    base_log_score: Array
    step_index: Array


class ParallelTemperingInfo(StrictModule):
    accepted_swaps: Array
    attempted_swaps: Array
    state_change_fraction: Array


class ParallelTempering(StrictModule):
    """Alternating exact tempered Gibbs sweeps and neighboring replica exchanges."""

    inverse_temperatures: Array
    method_id: str = eqx.field(static=True)

    def __init__(self, inverse_temperatures: ArrayLike, /):
        values = jnp.asarray(inverse_temperatures, dtype=float).reshape((-1,))
        host = np.asarray(values)
        if values.size < 2 or np.any(~np.isfinite(host)) or np.any(host <= 0):
            raise ValueError(
                "inverse_temperatures must contain at least two positive values."
            )
        if np.any(np.diff(host) <= 0):
            raise ValueError("inverse_temperatures must be strictly increasing.")
        self.inverse_temperatures = values
        self.method_id = "parallel-tempering"


class ReducedGibbsResult(StrictModule):
    """Final chain state and fixed-memory online reduction."""

    state: GibbsState
    reduction: object
    root_key: Array
    num_sweeps: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


class AbstractChainReducer(StrictModule):
    """Fixed-structure online reduction of correlated chain states."""

    @abstractmethod
    def initialize(self, positions: Array, scores: Array, /):
        raise NotImplementedError

    @abstractmethod
    def update(self, carry, positions: Array, scores: Array, /):
        raise NotImplementedError

    @abstractmethod
    def finalize(self, carry, /):
        raise NotImplementedError


class MomentReducerState(StrictModule):
    count: Array
    first_sum: Array
    second_sum: Array


class MomentReducer(AbstractChainReducer):
    """Online first and second raw moments over chain states."""

    def initialize(self, positions: Array, scores: Array, /):
        del scores
        shape = positions.shape[1:]
        return MomentReducerState(
            count=jnp.asarray(0, dtype=jnp.int32),
            first_sum=jnp.zeros(shape, dtype=float),
            second_sum=jnp.zeros(shape, dtype=float),
        )

    def update(self, carry, positions: Array, scores: Array, /):
        del scores
        values = positions.astype(float)
        return MomentReducerState(
            count=carry.count + values.shape[0],
            first_sum=carry.first_sum + jnp.sum(values, axis=0),
            second_sum=carry.second_sum + jnp.sum(values**2, axis=0),
        )

    def finalize(self, carry, /):
        denominator = jnp.maximum(carry.count, 1)
        mean = carry.first_sum / denominator
        return {"mean": mean, "variance": carry.second_sum / denominator - mean**2}


class BestStateReducerState(StrictModule):
    position: Array
    score: Array


class BestStateReducer(AbstractChainReducer):
    """Online highest-score assignment with deterministic first-tie retention."""

    def initialize(self, positions: Array, scores: Array, /):
        return BestStateReducerState(
            position=jnp.zeros_like(positions[0]),
            score=jnp.asarray(-jnp.inf, dtype=scores.dtype),
        )

    def update(self, carry, positions: Array, scores: Array, /):
        index = jnp.argmax(scores)
        replace = scores[index] > carry.score
        return BestStateReducerState(
            position=jnp.where(replace, positions[index], carry.position),
            score=jnp.where(replace, scores[index], carry.score),
        )

    def finalize(self, carry, /):
        return carry


def _sample_site(
    prepared,
    positions,
    variable,
    key,
    sweep_index,
    event_index,
    clamped,
):
    chain_count = int(positions.shape[0])
    chain_indices = jnp.arange(chain_count, dtype=jnp.uint32)
    logits = jax.vmap(lambda position: _conditional_logits(prepared, position, variable))(
        positions
    )
    feasible = jnp.any(jnp.isfinite(logits), axis=-1)
    keys = jax.vmap(
        lambda chain: derive_key(
            key,
            _RANDOM_SCAN_ADDRESS,
            chain,
            sweep_index,
            event_index,
            variable,
        )
    )(chain_indices)
    selected = jax.vmap(lambda subkey, values: jr.categorical(subkey, values))(
        keys,
        jnp.where(feasible[:, None], logits, 0.0),
    ).astype(jnp.int32)
    fixed = clamped[:, variable]
    current = positions[:, variable]
    selected = jnp.where(feasible & ~fixed, selected, current)
    return positions.at[:, variable].set(selected), feasible | fixed, selected != current


def gibbs_sweep_with_policy(
    prepared: PreparedChromaticGibbs,
    state: GibbsState,
    key: Key[Array, ""],
    policy: GibbsScanPolicy,
    /,
    *,
    clamped: ArrayLike | None = None,
) -> tuple[GibbsState, GibbsTransitionInfo]:
    """Advance a systematic, random-site, or randomized-color exact Gibbs sweep."""
    if policy.kind == "systematic" and (
        clamped is None or jnp.asarray(clamped).ndim == 1
    ):
        return gibbs_sweep(prepared, state, key, clamped=clamped)
    masks = (
        jnp.zeros_like(state.positions, dtype=bool)
        if clamped is None
        else jnp.broadcast_to(jnp.asarray(clamped, dtype=bool), state.positions.shape)
    )
    if masks.shape != state.positions.shape:
        raise ValueError("clamped must broadcast to (chain, variable).")
    positions = state.positions
    valid = jnp.ones((state.num_chains,), dtype=bool)
    changed = jnp.zeros((state.num_chains,), dtype=jnp.int32)
    if policy.kind == "random-scan":
        count = (
            prepared.graph.num_variables
            if policy.updates_per_sweep is None
            else policy.updates_per_sweep
        )
        variables = jr.randint(
            derive_key(key, _RANDOM_SCAN_ADDRESS, state.sweep_index),
            (count,),
            0,
            prepared.graph.num_variables,
        )
        for index in range(count):
            branches = tuple(
                lambda current, variable=variable, event=index: _sample_site(
                    prepared,
                    current,
                    variable,
                    key,
                    state.sweep_index,
                    event,
                    masks,
                )
                for variable in range(prepared.graph.num_variables)
            )
            positions, site_valid, site_changed = jax.lax.switch(
                variables[index],
                branches,
                positions,
            )
            valid = valid & site_valid
            changed = changed + site_changed.astype(jnp.int32)
        attempted_updates = count
    else:

        def update_stage(current, stage, event_offset):
            stage_positions, stage_valid, stage_changed = current
            snapshot = stage_positions
            for local_index, variable in enumerate(stage):
                updated, site_valid, site_changed = _sample_site(
                    prepared,
                    snapshot,
                    variable,
                    key,
                    state.sweep_index,
                    event_offset + local_index,
                    masks,
                )
                stage_positions = stage_positions.at[:, variable].set(
                    updated[:, variable]
                )
                stage_valid = stage_valid & site_valid
                stage_changed = stage_changed + site_changed.astype(jnp.int32)
            return stage_positions, stage_valid, stage_changed

        if policy.kind == "systematic":
            for stage_index, stage in enumerate(prepared.stages):
                positions, valid, changed = update_stage(
                    (positions, valid, changed),
                    stage,
                    stage_index * prepared.graph.num_variables,
                )
        else:
            order = jr.permutation(
                derive_key(key, _RANDOM_SCAN_ADDRESS, state.sweep_index),
                jnp.arange(len(prepared.stages)),
            )
            for order_index in range(len(prepared.stages)):
                branches = tuple(
                    lambda current, stage=stage, event=order_index: update_stage(
                        current,
                        stage,
                        event * prepared.graph.num_variables,
                    )
                    for stage in prepared.stages
                )
                positions, valid, changed = jax.lax.switch(
                    order[order_index],
                    branches,
                    (positions, valid, changed),
                )
        attempted_updates = prepared.graph.num_variables
    scores = prepared.precision.accumulation(
        factor_graph_log_score(prepared.graph, positions)
    )
    valid = valid & jnp.isfinite(scores)
    next_state = GibbsState(
        positions,
        scores,
        valid=state.valid & valid,
        sweep_index=state.sweep_index + 1,
    )
    return next_state, GibbsTransitionInfo(
        status=jnp.where(
            valid,
            int(GibbsTransitionStatus.SUCCESS),
            int(GibbsTransitionStatus.INFEASIBLE_CONDITIONAL),
        ),
        valid=valid,
        invalid_conditional_count=(~valid).astype(jnp.int32),
        state_change_fraction=changed.astype(float) / max(attempted_updates, 1),
    )


def joint_block_sweep(
    prepared: PreparedChromaticGibbs,
    state: GibbsState,
    block: JointDiscreteBlock,
    key: Key[Array, ""],
    /,
) -> tuple[GibbsState, GibbsTransitionInfo]:
    """Sample one dependent block exactly by bounded conditional enumeration."""
    graph = prepared.graph
    if max(block.variables) >= graph.num_variables:
        raise ValueError("Joint block variable is outside the graph.")
    cards = tuple(int(graph.cardinalities[index]) for index in block.variables)
    count = int(np.prod(cards))
    if count > block.maximum_configurations:
        raise ValueError("Joint block conditional exceeds maximum_configurations.")
    configurations = jnp.asarray(tuple(np.ndindex(cards)), dtype=jnp.int32)
    chain_indices = jnp.arange(state.num_chains, dtype=jnp.uint32)

    def one(position, chain):
        candidates = jnp.broadcast_to(position, (count, graph.num_variables))
        candidates = candidates.at[:, jnp.asarray(block.variables)].set(configurations)
        scores = prepared.precision.accumulation(
            factor_graph_log_score(graph, candidates)
        )
        subkey = derive_key(
            key,
            _RANDOM_SCAN_ADDRESS,
            chain,
            state.sweep_index,
            *block.variables,
        )
        feasible = jnp.any(jnp.isfinite(scores))
        selected = jr.categorical(subkey, jnp.where(feasible, scores, 0.0))
        updated = jnp.where(feasible, candidates[selected], position)
        return updated, feasible, jnp.any(updated != position)

    positions, valid, changed = jax.vmap(one)(state.positions, chain_indices)
    scores = prepared.precision.accumulation(factor_graph_log_score(graph, positions))
    next_state = GibbsState(
        positions,
        scores,
        valid=state.valid & valid,
        sweep_index=state.sweep_index + 1,
    )
    return next_state, GibbsTransitionInfo(
        status=jnp.where(
            valid,
            int(GibbsTransitionStatus.SUCCESS),
            int(GibbsTransitionStatus.INFEASIBLE_CONDITIONAL),
        ),
        valid=valid,
        invalid_conditional_count=(~valid).astype(jnp.int32),
        state_change_fraction=changed.astype(float),
    )


def initialize_parallel_tempering(
    prepared: PreparedChromaticGibbs,
    positions: ArrayLike,
    method: ParallelTempering,
    /,
) -> ParallelTemperingState:
    states = jnp.asarray(positions, dtype=jnp.int32)
    expected = (int(method.inverse_temperatures.shape[0]), prepared.graph.num_variables)
    if states.shape != expected:
        raise ValueError(f"positions must have shape {expected}.")
    scores = prepared.precision.accumulation(
        factor_graph_log_score(prepared.graph, states)
    )
    if not bool(jnp.all(jnp.isfinite(scores))):
        raise ValueError("Every replica must start in finite graph support.")
    return ParallelTemperingState(
        positions=states,
        inverse_temperatures=method.inverse_temperatures,
        base_log_score=scores,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
    )


def parallel_tempering_step(
    prepared: PreparedChromaticGibbs,
    state: ParallelTemperingState,
    key: Key[Array, ""],
    /,
) -> tuple[ParallelTemperingState, ParallelTemperingInfo]:
    """Advance tempered replicas and apply alternating neighboring exchange moves."""
    positions = state.positions
    changed = jnp.zeros((positions.shape[0],), dtype=float)
    for replica in range(int(positions.shape[0])):
        single = GibbsState(
            positions[replica : replica + 1],
            state.base_log_score[replica : replica + 1],
        )
        # Scaling all factor scores yields the correct tempered scalar conditional.
        updated = single
        for stage in prepared.stages:
            for variable in stage:
                logits = _conditional_logits(prepared, updated.positions[0], variable)
                logits = state.inverse_temperatures[replica] * logits
                subkey = derive_key(
                    key,
                    _RANDOM_SCAN_ADDRESS,
                    replica,
                    state.step_index,
                    variable,
                )
                selected = jr.categorical(subkey, logits).astype(jnp.int32)
                changed = changed.at[replica].add(
                    (selected != updated.positions[0, variable]).astype(float)
                )
                updated = GibbsState(
                    updated.positions.at[0, variable].set(selected),
                    updated.log_score,
                )
        positions = positions.at[replica].set(updated.positions[0])
    scores = prepared.precision.accumulation(
        factor_graph_log_score(prepared.graph, positions)
    )
    parity = state.step_index % 2
    accepted = []
    attempted = []
    for left in range(int(positions.shape[0]) - 1):
        right = left + 1
        enabled = (left % 2) == parity
        log_ratio = (
            state.inverse_temperatures[left] - state.inverse_temperatures[right]
        ) * (scores[right] - scores[left])
        subkey = derive_key(key, _TEMPERING_ADDRESS, state.step_index, left)
        accept = enabled & (jnp.log(jr.uniform(subkey)) < jnp.minimum(log_ratio, 0.0))
        left_position, right_position = positions[left], positions[right]
        left_score, right_score = scores[left], scores[right]
        positions = positions.at[left].set(
            jnp.where(accept, right_position, left_position)
        )
        positions = positions.at[right].set(
            jnp.where(accept, left_position, right_position)
        )
        scores = scores.at[left].set(jnp.where(accept, right_score, left_score))
        scores = scores.at[right].set(jnp.where(accept, left_score, right_score))
        accepted.append(accept)
        attempted.append(enabled)
    return ParallelTemperingState(
        positions=positions,
        inverse_temperatures=state.inverse_temperatures,
        base_log_score=scores,
        step_index=state.step_index + 1,
    ), ParallelTemperingInfo(
        accepted_swaps=jnp.stack(accepted) if accepted else jnp.zeros((0,), dtype=bool),
        attempted_swaps=jnp.stack(attempted)
        if attempted
        else jnp.zeros((0,), dtype=bool),
        state_change_fraction=changed / max(prepared.graph.num_variables, 1),
    )


def reduce_gibbs_chain(
    prepared: PreparedChromaticGibbs,
    state: GibbsState,
    reducer: AbstractChainReducer,
    /,
    *,
    key: Key[Array, ""],
    num_sweeps: int,
    policy: GibbsScanPolicy | None = None,
) -> ReducedGibbsResult:
    """Run Gibbs transitions while retaining only a fixed-size online reduction."""
    sweeps = int(num_sweeps)
    if sweeps < 1:
        raise ValueError("num_sweeps must be positive.")
    selected = GibbsScanPolicy() if policy is None else policy
    if not isinstance(selected, GibbsScanPolicy):
        raise TypeError("policy must be GibbsScanPolicy or None.")
    reduction = reducer.initialize(state.positions, state.log_score)

    def step(carry, _):
        chain_state, reducer_state = carry
        updated, _info = gibbs_sweep_with_policy(
            prepared,
            chain_state,
            key,
            selected,
        )
        reduced = reducer.update(
            reducer_state,
            updated.positions,
            updated.log_score,
        )
        return (updated, reduced), None

    (final_state, final_reduction), _ = jax.lax.scan(
        step,
        (state, reduction),
        xs=None,
        length=sweeps,
    )
    return ReducedGibbsResult(
        state=final_state,
        reduction=reducer.finalize(final_reduction),
        root_key=key,
        num_sweeps=sweeps,
        policy_id=selected.policy_id,
    )


def wolff_cluster_step(
    prepared: PreparedChromaticGibbs,
    position: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    inverse_temperature: float = 1.0,
) -> Array:
    """Apply one exact eager Wolff update to a zero-field ferromagnetic Ising graph."""
    beta = float(inverse_temperature)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("inverse_temperature must be finite and positive.")
    graph = prepared.graph
    state = np.asarray(position, dtype=np.int32).copy()
    if state.shape != (graph.num_variables,):
        raise ValueError("position must have one state per graph variable.")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(graph.num_variables)]
    for group, scope in zip(graph.factor_groups, graph.factor_scopes):
        if not isinstance(group, IsingFactorGroup):
            raise TypeError("Wolff updates require IsingFactorGroup factors only.")
        arity = int(scope.shape[1])
        if arity == 1:
            if np.any(np.asarray(group.weights) != 0.0):
                raise ValueError("Wolff updates require zero unary fields.")
        elif arity == 2:
            weights = np.asarray(group.weights)
            if np.any(weights < 0.0):
                raise ValueError("Wolff updates require ferromagnetic couplings.")
            for row, weight in zip(np.asarray(scope), weights):
                left, right = int(row[0]), int(row[1])
                adjacency[left].append((right, float(weight)))
                adjacency[right].append((left, float(weight)))
        else:
            raise ValueError("Wolff updates support unary and pairwise Ising factors.")
    seed = int(jr.randint(key, (), 0, graph.num_variables))
    cluster = {seed}
    pending = deque([seed])
    counter = 0
    while pending:
        source = pending.popleft()
        for target, coupling in adjacency[source]:
            subkey = derive_key(key, _RANDOM_SCAN_ADDRESS, counter, source, target)
            counter += 1
            probability = 1.0 - np.exp(-2.0 * beta * coupling)
            if (
                state[target] == state[source]
                and target not in cluster
                and float(jr.uniform(subkey)) < probability
            ):
                cluster.add(target)
                pending.append(target)
    indices = np.asarray(sorted(cluster), dtype=np.int32)
    state[indices] = 1 - state[indices]
    return jnp.asarray(state)


__all__ = [
    "AbstractChainReducer",
    "BestStateReducer",
    "BestStateReducerState",
    "GibbsScanPolicy",
    "JointDiscreteBlock",
    "MomentReducer",
    "MomentReducerState",
    "ParallelTempering",
    "ParallelTemperingInfo",
    "ReducedGibbsResult",
    "ParallelTemperingState",
    "gibbs_sweep_with_policy",
    "initialize_parallel_tempering",
    "joint_block_sweep",
    "reduce_gibbs_chain",
    "parallel_tempering_step",
    "wolff_cluster_step",
]
