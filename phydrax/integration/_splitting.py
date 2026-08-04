#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import log
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

from .._strict import StrictModule
from ..stochastic._events import (
    evaluate_path_event,
    path_event_scores,
    PathEvent,
    PathEventResult,
)
from ..stochastic._trajectory import StochasticTrajectory


class AdaptiveSplittingStatus(IntEnum):
    """Terminal status for one adaptive multilevel splitting population."""

    CONVERGED = 0
    EXTINCTION = 1
    MAXIMUM_ROUNDS_REACHED = 2


class AdaptiveMultilevelSplittingPlan(StrictModule):
    """Population and stopping policy for adaptive multilevel splitting."""

    population_size: int = eqx.field(static=True)
    kill_count: int = eqx.field(static=True)
    max_rounds: int = eqx.field(static=True)
    target_level: float = eqx.field(static=True)

    def __init__(
        self,
        population_size: int,
        /,
        *,
        kill_count: int | None = None,
        max_rounds: int = 256,
        target_level: float = 0.0,
    ):
        population = int(population_size)
        if population < 2:
            raise ValueError("population_size must be at least two.")
        killed = max(1, population // 10) if kill_count is None else int(kill_count)
        if killed < 1 or killed >= population:
            raise ValueError("kill_count must lie in [1, population_size - 1].")
        rounds = int(max_rounds)
        if rounds < 1:
            raise ValueError("max_rounds must be positive.")
        level = float(target_level)
        if not np.isfinite(level):
            raise ValueError("target_level must be finite.")
        self.population_size = population
        self.kill_count = killed
        self.max_rounds = rounds
        self.target_level = level


class AdaptiveSplittingBranchRequest(StrictModule):
    """One solver-independent branching request at a crossed score level."""

    killed_indices: Array
    parent_indices: Array
    branch_indices: Array
    branch_keys: Array
    level: float = eqx.field(static=True)
    round_index: int = eqx.field(static=True)

    def __init__(
        self,
        killed_indices: Array,
        parent_indices: Array,
        branch_indices: Array,
        branch_keys: Array,
        /,
        *,
        level: float,
        round_index: int,
    ):
        killed = jnp.asarray(killed_indices, dtype=jnp.int32).reshape((-1,))
        parents = jnp.asarray(parent_indices, dtype=jnp.int32).reshape((-1,))
        branches = jnp.asarray(branch_indices, dtype=jnp.int32).reshape((-1,))
        keys = jnp.asarray(branch_keys)
        if killed.shape != parents.shape or killed.shape != branches.shape:
            raise ValueError("Killed, parent, and branch indices must have equal shapes.")
        if keys.shape[:1] != killed.shape:
            raise ValueError("branch_keys must contain one key per killed trajectory.")
        self.killed_indices = killed
        self.parent_indices = parents
        self.branch_indices = branches
        self.branch_keys = keys
        self.level = float(level)
        self.round_index = int(round_index)


InitialPathSampler: TypeAlias = Callable[[Key[Array, ""], int], StochasticTrajectory]
PathBranchSampler: TypeAlias = Callable[
    [StochasticTrajectory, AdaptiveSplittingBranchRequest],
    StochasticTrajectory,
]


class AdaptiveSplittingDiagnostics(StrictModule):
    """Levels, survival factors, and complete resampling genealogy."""

    levels: Array
    survival_probabilities: Array
    killed_counts: Array
    killed_masks: Array
    parent_indices: Array
    branch_indices: Array
    terminal_reached: Array
    initial_trajectory_ids: tuple[str, ...] = eqx.field(static=True)
    population_trajectory_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        levels: Array,
        survival_probabilities: Array,
        killed_counts: Array,
        killed_masks: Array,
        parent_indices: Array,
        branch_indices: Array,
        terminal_reached: Array,
        /,
        *,
        initial_trajectory_ids: tuple[str, ...],
        population_trajectory_ids: tuple[tuple[str, ...], ...],
    ):
        level_values = jnp.asarray(levels, dtype=float).reshape((-1,))
        survival = jnp.asarray(survival_probabilities, dtype=float).reshape((-1,))
        counts = jnp.asarray(killed_counts, dtype=jnp.int32).reshape((-1,))
        killed = jnp.asarray(killed_masks, dtype=bool)
        parents = jnp.asarray(parent_indices, dtype=jnp.int32)
        branches = jnp.asarray(branch_indices, dtype=jnp.int32)
        num_rounds = level_values.shape[0]
        if survival.shape != (num_rounds,) or counts.shape != (num_rounds,):
            raise ValueError("Round-level splitting diagnostics must have equal lengths.")
        if killed.ndim != 2 or killed.shape[0] != num_rounds:
            raise ValueError("killed_masks must have shape (round, population).")
        if parents.shape != killed.shape or branches.shape != killed.shape:
            raise ValueError("Genealogy arrays must share killed_masks shape.")
        terminal = jnp.asarray(terminal_reached, dtype=bool).reshape((-1,))
        if killed.shape[1:] != terminal.shape:
            raise ValueError("terminal_reached must have one entry per trajectory.")
        populations = tuple(tuple(ids) for ids in population_trajectory_ids)
        if len(populations) != num_rounds:
            raise ValueError(
                "population_trajectory_ids must contain one entry per round."
            )
        self.levels = level_values
        self.survival_probabilities = survival
        self.killed_counts = counts
        self.killed_masks = killed
        self.parent_indices = parents
        self.branch_indices = branches
        self.terminal_reached = terminal
        self.initial_trajectory_ids = tuple(initial_trajectory_ids)
        self.population_trajectory_ids = populations

    @property
    def num_rounds(self) -> int:
        return int(self.levels.shape[0])

    @property
    def terminal_success_count(self) -> Array:
        return jnp.sum(self.terminal_reached)

    @property
    def extinct(self) -> Array:
        return self.terminal_success_count == 0


class AdaptiveMultilevelSplittingResult(StrictModule):
    """One adaptive splitting probability estimate and its final path population."""

    probability: Array
    log_probability: Array
    status: Array
    population: StochasticTrajectory
    event_result: PathEventResult
    diagnostics: AdaptiveSplittingDiagnostics

    def __init__(
        self,
        probability: Array,
        log_probability: Array,
        status: Array,
        population: StochasticTrajectory,
        event_result: PathEventResult,
        diagnostics: AdaptiveSplittingDiagnostics,
        /,
    ):
        self.probability = jnp.asarray(probability, dtype=float).reshape(())
        self.log_probability = jnp.asarray(log_probability, dtype=float).reshape(())
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.population = population
        self.event_result = event_result
        self.diagnostics = diagnostics

    @property
    def successful(self) -> Array:
        return self.status == int(AdaptiveSplittingStatus.CONVERGED)

    @property
    def status_message(self) -> str:
        code = int(self.status)
        if code == int(AdaptiveSplittingStatus.CONVERGED):
            return "converged"
        if code == int(AdaptiveSplittingStatus.EXTINCTION):
            return "population became extinct at an adaptive level"
        if code == int(AdaptiveSplittingStatus.MAXIMUM_ROUNDS_REACHED):
            return "maximum adaptive splitting rounds reached"
        return f"unknown adaptive splitting status {code}"


class AdaptiveSplittingEnsembleResult(StrictModule):
    """Independent splitting replicates with an empirical estimator uncertainty."""

    probability: Array
    standard_error: Array
    replicate_probabilities: Array
    completed: Array
    statuses: Array
    results: tuple[AdaptiveMultilevelSplittingResult, ...]

    def __init__(
        self,
        results: tuple[AdaptiveMultilevelSplittingResult, ...],
        /,
    ):
        if not results:
            raise ValueError(
                "An adaptive splitting ensemble requires at least one result."
            )
        probabilities = jnp.stack(tuple(result.probability for result in results))
        statuses = jnp.stack(tuple(result.status for result in results))
        completed = statuses != int(AdaptiveSplittingStatus.MAXIMUM_ROUNDS_REACHED)
        count = int(jnp.sum(completed))
        if count == 0:
            estimate = jnp.asarray(jnp.nan)
            standard_error = jnp.asarray(jnp.nan)
        else:
            retained = probabilities[completed]
            estimate = jnp.mean(retained)
            standard_error = (
                jnp.std(retained, ddof=1) / jnp.sqrt(float(count))
                if count > 1
                else jnp.asarray(jnp.nan)
            )
        self.probability = estimate
        self.standard_error = standard_error
        self.replicate_probabilities = probabilities
        self.completed = completed
        self.statuses = statuses
        self.results = results

    @property
    def num_replicates(self) -> int:
        return len(self.results)

    @property
    def num_completed(self) -> Array:
        return jnp.sum(self.completed)

    @property
    def num_extinctions(self) -> Array:
        return jnp.sum(self.statuses == int(AdaptiveSplittingStatus.EXTINCTION))

    @property
    def extinction_fraction(self) -> Array:
        return self.num_extinctions / float(self.num_replicates)


def _validate_population(
    population: StochasticTrajectory,
    plan: AdaptiveMultilevelSplittingPlan,
    /,
) -> None:
    if not isinstance(population, StochasticTrajectory):
        raise TypeError("Path samplers must return a StochasticTrajectory.")
    if population.case_shape:
        raise ValueError("Adaptive splitting currently requires one physical case.")
    if population.realization_shape != (plan.population_size,):
        raise ValueError(
            "Adaptive splitting requires one realization axis of length population_size."
        )


def _population_signature(population: StochasticTrajectory, /) -> tuple:
    return (
        population.case_axes,
        population.case_shape,
        population.realization_axes,
        population.realization_shape,
        population.time_axis,
        population.state_axes,
        population.state_shape,
        population.discretization_id,
        population.basis_id,
        population.approximation_id,
    )


def _score_population(
    population: StochasticTrajectory,
    event: PathEvent,
    /,
) -> tuple[Array, Array]:
    score_paths = path_event_scores(population, event)
    score_paths = jnp.where(jnp.isfinite(score_paths), score_paths, -jnp.inf)
    return jnp.max(score_paths, axis=-1), score_paths


def _empty_round_matrix(population_size: int, /, *, dtype) -> Array:
    return jnp.empty((0, population_size), dtype=dtype)


def adaptive_multilevel_splitting(
    event: PathEvent,
    plan: AdaptiveMultilevelSplittingPlan,
    /,
    *,
    initial_sampler: InitialPathSampler,
    branch_sampler: PathBranchSampler,
    key: Key[Array, ""],
) -> AdaptiveMultilevelSplittingResult:
    """Estimate a path-event probability with adaptive multilevel splitting.

    ``initial_sampler`` invokes the ordinary path solver for the initial population.
    ``branch_sampler`` receives exact parent and first-level-crossing indices and must
    return the complete updated population, preserving every non-killed trajectory.
    This keeps time stepping, jump integration, SPDE discretization, and learned path
    models outside the estimator while retaining explicit resampling genealogy.

    Ties at an adaptive level are killed together. This is required for discrete path
    scores; it also makes extinction explicit instead of silently selecting arbitrary
    tied survivors.
    """

    if not isinstance(event, PathEvent):
        raise TypeError("event must implement the PathEvent contract.")
    if not isinstance(plan, AdaptiveMultilevelSplittingPlan):
        raise TypeError("plan must be an AdaptiveMultilevelSplittingPlan.")
    if not callable(initial_sampler) or not callable(branch_sampler):
        raise TypeError("initial_sampler and branch_sampler must be callable.")
    initial_key, branching_key = jr.split(key)
    population = initial_sampler(initial_key, plan.population_size)
    _validate_population(population, plan)
    signature = _population_signature(population)
    initial_ids = population.trajectory_ids
    levels: list[float] = []
    survival_probabilities: list[float] = []
    killed_counts: list[int] = []
    killed_masks: list[Array] = []
    parent_history: list[Array] = []
    branch_history: list[Array] = []
    population_ids: list[tuple[str, ...]] = []
    log_survival = 0.0
    status = AdaptiveSplittingStatus.MAXIMUM_ROUNDS_REACHED

    for round_index in range(plan.max_rounds):
        scores, score_paths = _score_population(population, event)
        host_scores = np.asarray(jax.device_get(scores))
        level = float(np.partition(host_scores, plan.kill_count - 1)[plan.kill_count - 1])
        if level >= plan.target_level:
            status = AdaptiveSplittingStatus.CONVERGED
            break
        killed_host = host_scores <= level
        num_killed = int(np.sum(killed_host))
        if num_killed >= plan.population_size:
            status = AdaptiveSplittingStatus.EXTINCTION
            break
        killed = jnp.asarray(killed_host)
        survivor_indices = jnp.asarray(np.flatnonzero(~killed_host), dtype=jnp.int32)
        killed_indices = jnp.asarray(np.flatnonzero(killed_host), dtype=jnp.int32)
        round_key = jr.fold_in(branching_key, round_index)
        parent_key, child_key = jr.split(round_key)
        parent_indices = jr.choice(
            parent_key,
            survivor_indices,
            shape=(num_killed,),
            replace=True,
        )
        parent_score_paths = score_paths[parent_indices]
        crossed = jnp.isfinite(parent_score_paths) & (parent_score_paths >= level)
        branch_indices = jnp.argmax(crossed, axis=-1).astype(jnp.int32)
        request = AdaptiveSplittingBranchRequest(
            killed_indices,
            parent_indices,
            branch_indices,
            jr.split(child_key, num_killed),
            level=level,
            round_index=round_index,
        )
        next_population = branch_sampler(population, request)
        _validate_population(next_population, plan)
        if _population_signature(next_population) != signature:
            raise ValueError("branch_sampler changed the path-population semantic axes.")
        survivor_host = np.flatnonzero(~killed_host)
        if not bool(
            jnp.allclose(
                next_population.times[survivor_host],
                population.times[survivor_host],
                rtol=0.0,
                atol=0.0,
            )
            & jnp.allclose(
                next_population.states[survivor_host],
                population.states[survivor_host],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            & jnp.all(
                next_population.valid[survivor_host] == population.valid[survivor_host]
            )
        ):
            raise ValueError("branch_sampler modified a non-killed trajectory.")
        survival = (plan.population_size - num_killed) / plan.population_size
        log_survival += log(survival)
        full_parents = jnp.arange(plan.population_size, dtype=jnp.int32)
        full_parents = full_parents.at[killed_indices].set(parent_indices)
        full_branches = jnp.full((plan.population_size,), -1, dtype=jnp.int32)
        full_branches = full_branches.at[killed_indices].set(branch_indices)
        levels.append(level)
        survival_probabilities.append(survival)
        killed_counts.append(num_killed)
        killed_masks.append(killed)
        parent_history.append(full_parents)
        branch_history.append(full_branches)
        population_ids.append(next_population.trajectory_ids)
        population = next_population
    else:
        scores, _ = _score_population(population, event)
        host_scores = np.asarray(jax.device_get(scores))
        terminal_order = float(
            np.partition(host_scores, plan.kill_count - 1)[plan.kill_count - 1]
        )
        if terminal_order >= plan.target_level:
            status = AdaptiveSplittingStatus.CONVERGED

    final_scores, _ = _score_population(population, event)
    terminal_reached = final_scores >= plan.target_level
    success_fraction = jnp.mean(terminal_reached)
    if status == AdaptiveSplittingStatus.CONVERGED:
        log_probability = jnp.asarray(log_survival) + jnp.log(success_fraction)
        probability = jnp.exp(log_probability)
    elif status == AdaptiveSplittingStatus.EXTINCTION:
        probability = jnp.asarray(0.0)
        log_probability = jnp.asarray(-jnp.inf)
    else:
        probability = jnp.asarray(jnp.nan)
        log_probability = jnp.asarray(jnp.nan)
    diagnostics = AdaptiveSplittingDiagnostics(
        jnp.asarray(levels, dtype=float),
        jnp.asarray(survival_probabilities, dtype=float),
        jnp.asarray(killed_counts, dtype=jnp.int32),
        (
            jnp.stack(killed_masks)
            if killed_masks
            else _empty_round_matrix(plan.population_size, dtype=bool)
        ),
        (
            jnp.stack(parent_history)
            if parent_history
            else _empty_round_matrix(plan.population_size, dtype=jnp.int32)
        ),
        (
            jnp.stack(branch_history)
            if branch_history
            else _empty_round_matrix(plan.population_size, dtype=jnp.int32)
        ),
        terminal_reached,
        initial_trajectory_ids=initial_ids,
        population_trajectory_ids=tuple(population_ids),
    )
    event_result = evaluate_path_event(population, event)
    return AdaptiveMultilevelSplittingResult(
        probability,
        log_probability,
        jnp.asarray(int(status), dtype=jnp.int32),
        population,
        event_result,
        diagnostics,
    )


def replicate_adaptive_multilevel_splitting(
    event: PathEvent,
    plan: AdaptiveMultilevelSplittingPlan,
    num_replicates: int,
    /,
    *,
    initial_sampler: InitialPathSampler,
    branch_sampler: PathBranchSampler,
    key: Key[Array, ""],
) -> AdaptiveSplittingEnsembleResult:
    """Run independent AMS populations and estimate uncertainty across replicates."""

    count = int(num_replicates)
    if count < 1:
        raise ValueError("num_replicates must be positive.")
    keys = jr.split(key, count)
    results = tuple(
        adaptive_multilevel_splitting(
            event,
            plan,
            initial_sampler=initial_sampler,
            branch_sampler=branch_sampler,
            key=replicate_key,
        )
        for replicate_key in keys
    )
    return AdaptiveSplittingEnsembleResult(results)


__all__ = [
    "adaptive_multilevel_splitting",
    "AdaptiveMultilevelSplittingPlan",
    "AdaptiveMultilevelSplittingResult",
    "AdaptiveSplittingEnsembleResult",
    "AdaptiveSplittingBranchRequest",
    "AdaptiveSplittingDiagnostics",
    "AdaptiveSplittingStatus",
    "InitialPathSampler",
    "PathBranchSampler",
    "replicate_adaptive_multilevel_splitting",
]
