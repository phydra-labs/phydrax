#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._sampling import AbstractChainSampleResult, derive_key, SampleAddress
from .._strict import StrictModule
from ._model import (
    DiscreteFactorGraph,
    factor_graph_contains,
    factor_graph_log_score,
    factor_group_cardinality_signature,
    factor_group_dense_tables,
    pack_assignments,
)
from ._types import GibbsDiagnostics, GibbsTransitionStatus


_GIBBS_ADDRESS = SampleAddress(
    "factor-graph",
    "chromatic-gibbs",
    target="site",
    role="conditional-sample",
)


class ChromaticGibbs(StrictModule):
    """Deterministic strong-color schedule for exact scalar conditional updates."""

    colors: Array | None
    method_id: str = eqx.field(static=True)

    def __init__(self, colors: ArrayLike | None = None):
        if colors is None:
            resolved = None
        else:
            array = jnp.asarray(colors)
            if not jnp.issubdtype(array.dtype, jnp.integer):
                raise TypeError("Gibbs colors must be integers.")
            resolved = array.astype(jnp.int32).reshape((-1,))
        self.colors = resolved
        self.method_id = "chromatic-gibbs"


class GibbsSchedule(StrictModule):
    """Warmup, retained draws, and full sweeps between retained states."""

    warmup_sweeps: int = eqx.field(static=True)
    num_draws: int = eqx.field(static=True)
    sweeps_per_draw: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        warmup_sweeps: int = 0,
        num_draws: int,
        sweeps_per_draw: int = 1,
    ):
        warmup = int(warmup_sweeps)
        draws = int(num_draws)
        sweeps = int(sweeps_per_draw)
        if warmup < 0:
            raise ValueError("warmup_sweeps must be non-negative.")
        if draws < 1:
            raise ValueError("num_draws must be positive.")
        if sweeps < 1:
            raise ValueError("sweeps_per_draw must be positive.")
        self.warmup_sweeps = warmup
        self.num_draws = draws
        self.sweeps_per_draw = sweeps


class GibbsState(StrictModule):
    """Persistent chain positions, graph scores, validity, and logical sweep index."""

    positions: Array
    log_score: Array
    valid: Array
    sweep_index: Array

    def __init__(
        self,
        positions: ArrayLike,
        log_score: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        sweep_index: int | Array = 0,
    ):
        states = jnp.asarray(positions)
        if states.ndim != 2 or not jnp.issubdtype(states.dtype, jnp.integer):
            raise ValueError(
                "Gibbs positions must have shape (chain, variable) and integer dtype."
            )
        scores = jnp.asarray(log_score)
        if scores.shape != (int(states.shape[0]),) or jnp.iscomplexobj(scores):
            raise ValueError("Gibbs log_score must be one real value per chain.")
        validity = (
            jnp.isfinite(scores) if valid is None else jnp.asarray(valid, dtype=bool)
        )
        if validity.shape != scores.shape:
            raise ValueError("Gibbs validity must have one value per chain.")
        index = jnp.asarray(sweep_index, dtype=jnp.uint32)
        if index.shape != ():
            raise ValueError("sweep_index must be scalar.")
        self.positions = states.astype(jnp.int32)
        self.log_score = scores
        self.valid = validity
        self.sweep_index = index

    @property
    def num_chains(self) -> int:
        return int(self.positions.shape[0])


class GibbsTransitionInfo(StrictModule):
    """Per-chain support validity and movement from one full chromatic sweep."""

    status: Array
    valid: Array
    invalid_conditional_count: Array
    state_change_fraction: Array


class PreparedChromaticGibbs(StrictModule):
    """Topology-fixed strong-color Gibbs plan with refreshable factor tables."""

    graph: DiscreteFactorGraph
    method: ChromaticGibbs
    factor_tables: tuple[Array, ...]
    colors: Array
    stages: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    incidents: tuple[tuple[tuple[int, int, int], ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class GibbsSampleResult(AbstractChainSampleResult):
    """Persistent correlated Gibbs draws and complete support/movement evidence."""

    samples: Array
    log_score: Array
    transition_valid: Array
    invalid_conditional_count: Array
    state_change_fraction: Array
    final_state: GibbsState
    root_key: Array
    diagnostics: GibbsDiagnostics
    plan_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    warmup_sweeps: int = eqx.field(static=True)
    sweeps_per_draw: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        samples: ArrayLike,
        log_score: ArrayLike,
        transition_valid: ArrayLike,
        invalid_conditional_count: ArrayLike,
        state_change_fraction: ArrayLike,
        final_state: GibbsState,
        root_key: ArrayLike,
        diagnostics: GibbsDiagnostics,
        plan_id: str,
        method_id: str,
        warmup_sweeps: int,
        sweeps_per_draw: int,
    ):
        values = jnp.asarray(samples)
        scores = jnp.asarray(log_score)
        if values.ndim != 3:
            raise ValueError("Gibbs samples must have shape (chain, draw, variable).")
        chains, draws = int(values.shape[0]), int(values.shape[1])
        if scores.shape != (chains, draws):
            raise ValueError("Gibbs log scores must have shape (chain, draw).")
        transition_shape = (chains, draws, int(sweeps_per_draw))
        evidence = (
            transition_valid,
            invalid_conditional_count,
            state_change_fraction,
        )
        if any(jnp.asarray(value).shape != transition_shape for value in evidence):
            raise ValueError(
                f"Gibbs transition evidence must have shape {transition_shape}."
            )
        if final_state.num_chains != chains:
            raise ValueError("final_state chain count must match samples.")
        if not isinstance(diagnostics, GibbsDiagnostics):
            raise TypeError("diagnostics must be GibbsDiagnostics.")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be non-empty.")
        if not isinstance(method_id, str) or not method_id:
            raise ValueError("method_id must be non-empty.")
        self.samples = values.astype(jnp.int32)
        self.log_score = scores
        self.transition_valid = jnp.asarray(transition_valid, dtype=bool)
        self.invalid_conditional_count = jnp.asarray(
            invalid_conditional_count, dtype=jnp.int32
        )
        self.state_change_fraction = jnp.asarray(state_change_fraction)
        self.final_state = final_state
        self.root_key = jnp.asarray(root_key)
        self.diagnostics = diagnostics
        self.plan_id = plan_id
        self.method_id = method_id
        self.warmup_sweeps = int(warmup_sweeps)
        self.sweeps_per_draw = int(sweeps_per_draw)

    @property
    def num_chains(self) -> int:
        return int(self.samples.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.samples.shape[1])

    @property
    def chain_provenance(self) -> str:
        return f"markov:{self.method_id}:{self.plan_id}"


def _automatic_colors(graph: DiscreteFactorGraph, /) -> np.ndarray:
    conflicts: list[set[int]] = [set() for _ in range(graph.num_variables)]
    for scope in graph.factor_scopes:
        for row in np.asarray(scope, dtype=np.int32):
            for left_index, left in enumerate(row):
                for right in row[left_index + 1 :]:
                    conflicts[int(left)].add(int(right))
                    conflicts[int(right)].add(int(left))
    colors = np.full((graph.num_variables,), -1, dtype=np.int32)
    for variable in range(graph.num_variables):
        forbidden = {
            int(colors[neighbor])
            for neighbor in conflicts[variable]
            if colors[neighbor] >= 0
        }
        color = 0
        while color in forbidden:
            color += 1
        colors[variable] = color
    return colors


def _validate_colors(graph: DiscreteFactorGraph, colors: np.ndarray, /) -> None:
    if colors.shape != (graph.num_variables,):
        raise ValueError(f"colors must have shape ({graph.num_variables},).")
    if colors.size and np.any(colors < 0):
        raise ValueError("colors must be non-negative.")
    unique = np.unique(colors)
    if unique.size and not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("colors must be contiguous from zero.")
    for scope in graph.factor_scopes:
        for row in np.asarray(scope, dtype=np.int32):
            selected = colors[row]
            if len({int(value) for value in selected}) != len(selected):
                raise ValueError(
                    "Variables in one factor scope must have distinct Gibbs colors."
                )


def prepare_chromatic_gibbs(
    graph: DiscreteFactorGraph,
    method: ChromaticGibbs | None = None,
    /,
    *,
    max_factor_configurations: int = 65_536,
) -> PreparedChromaticGibbs:
    """Compile a deterministic validated strong-color Gibbs schedule."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    selected = ChromaticGibbs() if method is None else method
    if not isinstance(selected, ChromaticGibbs):
        raise TypeError("method must be ChromaticGibbs.")
    cap = int(max_factor_configurations)
    if cap < 1:
        raise ValueError("max_factor_configurations must be positive.")
    tables: list[Array] = []
    for group_index in range(len(graph.factor_groups)):
        signature = factor_group_cardinality_signature(graph, group_index)
        configurations = prod(signature)
        if configurations > cap:
            raise ValueError(
                f"Factor group {group_index} requires {configurations} configurations, "
                f"exceeding max_factor_configurations={cap}."
            )
        tables.append(factor_group_dense_tables(graph, group_index))

    colors_host = (
        _automatic_colors(graph)
        if selected.colors is None
        else np.asarray(selected.colors, dtype=np.int32)
    )
    _validate_colors(graph, colors_host)
    stages = tuple(
        tuple(int(value) for value in np.nonzero(colors_host == color)[0])
        for color in range(int(colors_host.max()) + 1 if colors_host.size else 0)
    )
    incident_lists: list[list[tuple[int, int, int]]] = [
        [] for _ in range(graph.num_variables)
    ]
    for group_index, scope in enumerate(graph.factor_scopes):
        for factor, row in enumerate(np.asarray(scope, dtype=np.int32)):
            for position, variable in enumerate(row):
                incident_lists[int(variable)].append((group_index, factor, position))
    incidents = tuple(tuple(values) for values in incident_lists)
    plan_id = canonical_fingerprint(
        {
            "kind": "chromatic-gibbs-plan",
            "structure_id": graph.structure_id,
            "colors": colors_host.tolist(),
            "max_factor_configurations": cap,
        }
    )
    return PreparedChromaticGibbs(
        graph=graph,
        method=selected,
        factor_tables=tuple(tables),
        colors=jnp.asarray(colors_host),
        stages=stages,
        incidents=incidents,
        plan_id=plan_id,
    )


def refresh_chromatic_gibbs(
    prepared: PreparedChromaticGibbs,
    graph: DiscreteFactorGraph,
    /,
) -> PreparedChromaticGibbs:
    """Refresh compatible numeric factor tables without recoloring."""
    if not isinstance(prepared, PreparedChromaticGibbs):
        raise TypeError("prepared must be PreparedChromaticGibbs.")
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    if graph.structure_id != prepared.graph.structure_id:
        raise ValueError("Refreshed graph structure does not match the Gibbs plan.")
    if graph.parameter_signature != prepared.graph.parameter_signature:
        raise ValueError("Refreshed graph parameter signature does not match the plan.")
    tables = tuple(
        factor_group_dense_tables(graph, index)
        for index in range(len(graph.factor_groups))
    )
    updated = eqx.tree_at(lambda value: value.graph, prepared, graph)
    return eqx.tree_at(lambda value: value.factor_tables, updated, tables)


def initialize_gibbs(
    prepared: PreparedChromaticGibbs,
    positions: ArrayLike,
    /,
) -> GibbsState:
    """Validate one or more finite-support chain initial positions."""
    if not isinstance(prepared, PreparedChromaticGibbs):
        raise TypeError("prepared must be PreparedChromaticGibbs.")
    states = pack_assignments(prepared.graph, positions)
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2:
        raise ValueError("Gibbs positions must have one leading chain axis.")
    contains = factor_graph_contains(prepared.graph, states)
    scores = jax.vmap(lambda value: factor_graph_log_score(prepared.graph, value))(states)
    valid = contains & jnp.isfinite(scores)
    if not bool(jnp.all(valid)):
        raise ValueError("Every initial Gibbs position must have finite graph support.")
    return GibbsState(states, scores, valid=valid)


def _conditional_logits(
    prepared: PreparedChromaticGibbs,
    position: Array,
    variable: int,
    /,
) -> Array:
    cardinality = int(np.asarray(prepared.graph.cardinalities)[variable])
    values: list[Array] = []
    for candidate in range(cardinality):
        score = jnp.asarray(0.0, dtype=position.dtype).astype(
            jnp.result_type(*[table.dtype for table in prepared.factor_tables])
        )
        for group_index, factor, scope_position in prepared.incidents[variable]:
            scope = prepared.graph.factor_scopes[group_index][factor]
            states = position[scope].at[scope_position].set(candidate)
            score = score + prepared.factor_tables[group_index][factor][tuple(states)]
        values.append(score)
    return jnp.stack(values)


def gibbs_sweep(
    prepared: PreparedChromaticGibbs,
    state: GibbsState,
    key: Key[Array, ""],
    /,
    *,
    clamped: ArrayLike | None = None,
) -> tuple[GibbsState, GibbsTransitionInfo]:
    """Advance all strong-color stages once from immutable per-stage snapshots."""
    if not isinstance(prepared, PreparedChromaticGibbs):
        raise TypeError("prepared must be PreparedChromaticGibbs.")
    if not isinstance(state, GibbsState):
        raise TypeError("state must be GibbsState.")
    if state.positions.shape[1:] != (prepared.graph.num_variables,):
        raise ValueError("Gibbs state variable axis does not match the plan.")
    clamp_mask = (
        jnp.zeros((prepared.graph.num_variables,), dtype=bool)
        if clamped is None
        else jnp.asarray(clamped, dtype=bool)
    )
    if clamp_mask.shape != (prepared.graph.num_variables,):
        raise ValueError("clamped must have one boolean per graph variable.")

    positions = state.positions
    invalid_count = jnp.zeros((state.num_chains,), dtype=jnp.int32)
    changed_count = jnp.zeros((state.num_chains,), dtype=jnp.int32)
    chain_indices = jnp.arange(state.num_chains, dtype=jnp.uint32)
    for stage_index, stage in enumerate(prepared.stages):
        snapshot = positions
        updates: list[tuple[int, Array, Array]] = []
        for variable in stage:
            logits = jax.vmap(
                lambda position, variable=variable: _conditional_logits(
                    prepared, position, variable
                )
            )(snapshot)
            feasible = jnp.any(jnp.isfinite(logits), axis=-1)
            site_keys = jax.vmap(
                lambda chain, stage_index=stage_index, variable=variable: derive_key(
                    key,
                    _GIBBS_ADDRESS,
                    chain,
                    state.sweep_index,
                    stage_index,
                    variable,
                )
            )(chain_indices)
            safe_logits = jnp.where(feasible[:, None], logits, 0.0)
            sampled = jax.vmap(lambda site_key, values: jr.categorical(site_key, values))(
                site_keys,
                safe_logits,
            ).astype(jnp.int32)
            fixed = clamp_mask[variable]
            current = snapshot[:, variable]
            sampled = jnp.where(feasible & ~fixed, sampled, current)
            updates.append((variable, sampled, feasible | fixed))
        for variable, sampled, feasible in updates:
            current = positions[:, variable]
            positions = positions.at[:, variable].set(sampled)
            invalid_count = invalid_count + (~feasible).astype(jnp.int32)
            changed_count = changed_count + (sampled != current).astype(jnp.int32)

    scores = jax.vmap(lambda value: factor_graph_log_score(prepared.graph, value))(
        positions
    )
    finite_score = jnp.isfinite(scores)
    valid = state.valid & (invalid_count == 0) & finite_score
    status = jnp.where(
        invalid_count > 0,
        int(GibbsTransitionStatus.INFEASIBLE_CONDITIONAL),
        jnp.where(
            ~finite_score,
            int(GibbsTransitionStatus.NONFINITE_SCORE),
            int(GibbsTransitionStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    denominator = max(prepared.graph.num_variables, 1)
    next_state = GibbsState(
        positions,
        scores,
        valid=valid,
        sweep_index=state.sweep_index + jnp.asarray(1, dtype=jnp.uint32),
    )
    return next_state, GibbsTransitionInfo(
        status=status,
        valid=valid,
        invalid_conditional_count=invalid_count,
        state_change_fraction=changed_count.astype(float) / denominator,
    )


def _mixing_diagnostics(
    samples: Array, invalid_count: Array, changed: Array
) -> GibbsDiagnostics:
    chains, draws = int(samples.shape[0]), int(samples.shape[1])
    if chains >= 2 and draws >= 4:
        from blackjax import diagnostics

        variance = jnp.var(samples.astype(float), axis=(0, 1))
        total = jnp.asarray(chains * draws, dtype=float)
        rhat = diagnostics.rhat(samples, chain_axis=0, sample_axis=1)
        bulk = diagnostics.ess_bulk(samples, chain_axis=0, sample_axis=1)
        tail = diagnostics.ess_tail(samples, chain_axis=0, sample_axis=1)
        rhat = jnp.where(variance == 0, 1.0, rhat)
        bulk = jnp.where(variance == 0, total, bulk)
        tail = jnp.where(variance == 0, total, tail)
        return GibbsDiagnostics(
            invalid_conditional_count=jnp.sum(invalid_count),
            mean_state_change_fraction=jnp.mean(changed),
            rhat=rhat,
            bulk_ess=bulk,
            tail_ess=tail,
            max_rhat=jnp.max(rhat),
            min_bulk_ess=jnp.min(bulk),
            min_tail_ess=jnp.min(tail),
            mixing_available=True,
        )
    unavailable = jnp.asarray(jnp.nan)
    return GibbsDiagnostics(
        invalid_conditional_count=jnp.sum(invalid_count),
        mean_state_change_fraction=jnp.mean(changed),
        rhat=None,
        bulk_ess=None,
        tail_ess=None,
        max_rhat=unavailable,
        min_bulk_ess=unavailable,
        min_tail_ess=unavailable,
        mixing_available=False,
    )


def sample_gibbs(
    prepared: PreparedChromaticGibbs,
    state: GibbsState,
    /,
    *,
    key: Key[Array, ""],
    schedule: GibbsSchedule,
    clamped: ArrayLike | None = None,
) -> GibbsSampleResult:
    """Warm persistent chains and retain chain-by-draw Gibbs states."""
    if not isinstance(prepared, PreparedChromaticGibbs):
        raise TypeError("prepared must be PreparedChromaticGibbs.")
    if not isinstance(state, GibbsState):
        raise TypeError("state must be GibbsState.")
    if not isinstance(schedule, GibbsSchedule):
        raise TypeError("schedule must be GibbsSchedule.")
    clamp_mask = (
        jnp.zeros((prepared.graph.num_variables,), dtype=bool)
        if clamped is None
        else jnp.asarray(clamped, dtype=bool)
    )
    if clamp_mask.shape != (prepared.graph.num_variables,):
        raise ValueError("clamped must have one boolean per graph variable.")

    def warmup_step(carry, _):
        next_state, _info = gibbs_sweep(
            prepared,
            carry,
            key,
            clamped=clamp_mask,
        )
        return next_state, None

    warmed, _ = jax.lax.scan(
        warmup_step,
        state,
        xs=None,
        length=schedule.warmup_sweeps,
    )

    def collect_draw(carry, _):
        def transition_step(inner, __):
            return gibbs_sweep(
                prepared,
                inner,
                key,
                clamped=clamp_mask,
            )

        next_state, infos = jax.lax.scan(
            transition_step,
            carry,
            xs=None,
            length=schedule.sweeps_per_draw,
        )
        output = (
            next_state.positions,
            next_state.log_score,
            infos.valid,
            infos.invalid_conditional_count,
            infos.state_change_fraction,
        )
        return next_state, output

    final_state, outputs = jax.lax.scan(
        collect_draw,
        warmed,
        xs=None,
        length=schedule.num_draws,
    )
    samples, scores, valid, invalid, changed = outputs
    samples = jnp.swapaxes(samples, 0, 1)
    scores = jnp.swapaxes(scores, 0, 1)
    valid = jnp.transpose(valid, (2, 0, 1))
    invalid = jnp.transpose(invalid, (2, 0, 1))
    changed = jnp.transpose(changed, (2, 0, 1))
    diagnostics = _mixing_diagnostics(samples, invalid, changed)
    return GibbsSampleResult(
        samples=samples,
        log_score=scores,
        transition_valid=valid,
        invalid_conditional_count=invalid,
        state_change_fraction=changed,
        final_state=final_state,
        root_key=key,
        diagnostics=diagnostics,
        plan_id=prepared.plan_id,
        method_id=prepared.method.method_id,
        warmup_sweeps=schedule.warmup_sweeps,
        sweeps_per_draw=schedule.sweeps_per_draw,
    )


__all__ = [
    "ChromaticGibbs",
    "GibbsSampleResult",
    "GibbsSchedule",
    "GibbsState",
    "GibbsTransitionInfo",
    "PreparedChromaticGibbs",
    "gibbs_sweep",
    "initialize_gibbs",
    "prepare_chromatic_gibbs",
    "refresh_chromatic_gibbs",
    "sample_gibbs",
]
