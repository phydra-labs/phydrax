#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from evosax.algorithms import DifferentialEvolution as EvosaxDifferentialEvolution
from jaxtyping import Array, ArrayLike, Key

from .._sampling import (
    design_signature,
    DesignLike,
    LatinHypercubeDesign,
    materialize_design,
    resolve_design,
)
from .._strict import StrictModule


SearchStrategy = Literal["best1bin", "rand1bin"]


class DifferentialEvolutionSearch(StrictModule):
    """Validated bounded differential-evolution search configuration."""

    population_size: int = eqx.field(static=True)
    max_generations: int = eqx.field(static=True)
    strategy: SearchStrategy = eqx.field(static=True)
    differential_weight: float = eqx.field(static=True)
    crossover_rate: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    design: Any

    def __init__(
        self,
        population_size: int,
        max_generations: int,
        /,
        *,
        strategy: SearchStrategy = "best1bin",
        differential_weight: float = 0.8,
        crossover_rate: float = 0.9,
        relative_tolerance: float = 0.01,
        absolute_tolerance: float = 0.0,
        design: DesignLike = LatinHypercubeDesign(),
    ):
        population_size_ = int(population_size)
        generations_ = int(max_generations)
        weight_ = float(differential_weight)
        crossover_ = float(crossover_rate)
        relative_ = float(relative_tolerance)
        absolute_ = float(absolute_tolerance)
        if population_size_ < 4:
            raise ValueError(
                "population_size must be at least 4 for differential evolution."
            )
        if generations_ < 0:
            raise ValueError("max_generations must be non-negative.")
        if strategy not in ("best1bin", "rand1bin"):
            raise ValueError("strategy must be 'best1bin' or 'rand1bin'.")
        if not np.isfinite(weight_) or weight_ < 0.0 or weight_ >= 2.0:
            raise ValueError("differential_weight must be finite and lie in [0, 2).")
        if not np.isfinite(crossover_) or crossover_ < 0.0 or crossover_ > 1.0:
            raise ValueError("crossover_rate must be finite and lie in [0, 1].")
        if not np.isfinite(relative_) or relative_ < 0.0:
            raise ValueError("relative_tolerance must be finite and non-negative.")
        if not np.isfinite(absolute_) or absolute_ < 0.0:
            raise ValueError("absolute_tolerance must be finite and non-negative.")
        self.population_size = population_size_
        self.max_generations = generations_
        self.strategy = strategy
        self.differential_weight = weight_
        self.crossover_rate = crossover_
        self.relative_tolerance = relative_
        self.absolute_tolerance = absolute_
        self.design = resolve_design(design)


class _DifferentialEvolutionResult(StrictModule):
    population_vectors: Array
    population_objectives: Array
    best_vector: Array
    raw_objective: Array
    best_objective_history: Array
    lower_bounds: Array
    upper_bounds: Array
    key: Key[Array, ""]
    search: DifferentialEvolutionSearch
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        population_vectors: Array,
        population_objectives: Array,
        best_vector: Array,
        raw_objective: Array,
        best_objective_history: Array,
        lower_bounds: Array,
        upper_bounds: Array,
        key: Key[Array, ""],
        search: DifferentialEvolutionSearch,
        converged: bool,
        termination_reason: str,
        generations: int,
        objective_evaluations: int,
        invalid_evaluations: int,
        design_signature_: str,
    ):
        self.population_vectors = jnp.asarray(population_vectors)
        self.population_objectives = jnp.asarray(population_objectives)
        self.best_vector = jnp.asarray(best_vector)
        self.raw_objective = jnp.asarray(raw_objective).reshape(())
        self.best_objective_history = jnp.asarray(best_objective_history)
        self.lower_bounds = jnp.asarray(lower_bounds)
        self.upper_bounds = jnp.asarray(upper_bounds)
        self.key = key
        self.search = search
        self.converged = bool(converged)
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = int(objective_evaluations)
        self.invalid_evaluations = int(invalid_evaluations)
        self.design_signature = str(design_signature_)


def _reflect_unit_box(population: Array, /) -> Array:
    folded = jnp.mod(population, 2.0)
    return jnp.where(folded <= 1.0, folded, 2.0 - folded)


def _population_objectives(
    population: Array,
    objective: Callable[[Array], Array],
    lower_bounds: Array,
    upper_bounds: Array,
    /,
) -> Array:
    def evaluate(unit_vector):
        vector = lower_bounds + unit_vector * (upper_bounds - lower_bounds)
        return jnp.asarray(objective(vector)).reshape(())

    return jax.vmap(evaluate)(population)


def _fitness_converged(
    fitness: Array,
    /,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Array:
    return jnp.all(jnp.isfinite(fitness)) & (
        jnp.std(fitness)
        <= absolute_tolerance + relative_tolerance * jnp.abs(jnp.mean(fitness))
    )


@lru_cache(maxsize=128)
def _differential_evolution_algorithm(
    population_size: int,
    dimension: int,
    dtype_name: str,
    /,
) -> EvosaxDifferentialEvolution:
    return EvosaxDifferentialEvolution(
        population_size=population_size,
        solution=jnp.zeros((dimension,), dtype=np.dtype(dtype_name)),
    )


@eqx.filter_jit
def _run_differential_evolution(
    objective: Callable[[Array], Array],
    algorithm: Any,
    algorithm_params: Any,
    initial_population: Array,
    lower_bounds: Array,
    upper_bounds: Array,
    key: Key[Array, ""],
    *,
    max_generations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
):
    initial_raw_fitness = _population_objectives(
        initial_population,
        objective,
        lower_bounds,
        upper_bounds,
    )
    initial_finite = jnp.isfinite(initial_raw_fitness)
    initial_fitness = jnp.where(initial_finite, initial_raw_fitness, jnp.inf)
    key, init_key = jr.split(key)
    algorithm_state = algorithm.init(
        init_key,
        initial_population,
        initial_fitness,
        algorithm_params,
    )
    history = jnp.full(
        (max_generations + 1,),
        jnp.inf,
        dtype=initial_fitness.dtype,
    )
    history = history.at[0].set(jnp.min(initial_fitness))
    loop_state = (
        jnp.asarray(0, dtype=jnp.int32),
        algorithm_state,
        key,
        jnp.sum(~initial_finite, dtype=jnp.int32),
        history,
    )

    def condition(carry):
        generation, state, _key, _invalid, _history = carry
        return (
            (generation < max_generations)
            & jnp.any(jnp.isfinite(state.fitness))
            & ~_fitness_converged(
                state.fitness,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        )

    def generation_step(carry):
        generation, state, key_, invalid, history_ = carry
        key_, ask_key, tell_key = jr.split(key_, 3)
        population, state = algorithm.ask(ask_key, state, algorithm_params)
        population = _reflect_unit_box(population)
        raw_fitness = _population_objectives(
            population,
            objective,
            lower_bounds,
            upper_bounds,
        )
        finite = jnp.isfinite(raw_fitness)
        fitness = jnp.where(finite, raw_fitness, jnp.inf)
        state, _metrics = algorithm.tell(
            tell_key,
            population,
            fitness,
            state,
            algorithm_params,
        )
        next_generation = generation + 1
        history_ = history_.at[next_generation].set(jnp.min(state.fitness))
        return (
            next_generation,
            state,
            key_,
            invalid + jnp.sum(~finite, dtype=jnp.int32),
            history_,
        )

    generation, state, _key, invalid, history = jax.lax.while_loop(
        condition,
        generation_step,
        loop_state,
    )
    population = algorithm.get_population(state)
    best = algorithm.get_best_solution(state)
    converged = _fitness_converged(
        state.fitness,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    best_fitness = jnp.min(state.fitness)
    raw_objective = jnp.where(
        jnp.isfinite(best_fitness),
        best_fitness,
        initial_raw_fitness[0],
    )
    return (
        generation,
        population,
        state.fitness,
        best,
        raw_objective,
        history,
        invalid,
        converged,
    )


def _validated_search_vectors(
    initial_vector: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    initial_np = np.asarray(initial_vector)
    lower_np = np.asarray(lower_bounds)
    upper_np = np.asarray(upper_bounds)
    if initial_np.ndim != 1:
        raise ValueError("initial_vector must be one-dimensional.")
    if initial_np.size == 0:
        raise ValueError("Differential evolution requires at least one dimension.")
    if lower_np.shape != initial_np.shape or upper_np.shape != initial_np.shape:
        raise ValueError(
            "initial_vector, lower_bounds, and upper_bounds must have identical shapes."
        )
    dtype = np.result_type(initial_np.dtype, lower_np.dtype, upper_np.dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("Differential-evolution vectors and bounds must be real-valued.")
    initial_np = initial_np.astype(dtype, copy=False)
    lower_np = lower_np.astype(dtype, copy=False)
    upper_np = upper_np.astype(dtype, copy=False)
    if not np.all(np.isfinite(initial_np)):
        raise ValueError("initial_vector must be finite.")
    if not np.all(np.isfinite(lower_np)) or not np.all(np.isfinite(upper_np)):
        raise ValueError("Differential-evolution bounds must be finite.")
    if np.any(lower_np >= upper_np):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    if np.any((initial_np < lower_np) | (initial_np > upper_np)):
        raise ValueError("initial_vector lies outside the search bounds.")
    return (
        jnp.asarray(initial_np),
        jnp.asarray(lower_np),
        jnp.asarray(upper_np),
    )


def _bounded_differential_evolution(
    objective: Callable[[Array], Array],
    initial_vector: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: Key[Array, ""],
) -> _DifferentialEvolutionResult:
    if not callable(objective):
        raise TypeError("objective must be callable.")
    if not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be a DifferentialEvolutionSearch.")
    initial, lower, upper = _validated_search_vectors(
        initial_vector,
        lower_bounds,
        upper_bounds,
    )
    dimension = int(initial.shape[0])
    initial_unit = (initial - lower) / (upper - lower)
    design_key = jr.fold_in(key, 0)
    evolution_key = jr.fold_in(key, 1)
    population = jnp.asarray(
        materialize_design(
            search.design,
            count=search.population_size,
            dimension=dimension,
            key=design_key,
        ),
        dtype=initial.dtype,
    )
    population = population.at[0].set(initial_unit)

    algorithm = _differential_evolution_algorithm(
        search.population_size,
        dimension,
        str(initial.dtype),
    )
    algorithm_params = replace(
        algorithm.default_params,
        elitism=search.strategy == "best1bin",
        crossover_rate=search.crossover_rate,
        differential_weight=search.differential_weight,
    )
    (
        generations,
        unit_population,
        population_objectives,
        best_unit,
        raw_objective,
        history,
        invalid_evaluations,
        converged,
    ) = _run_differential_evolution(
        objective,
        algorithm,
        algorithm_params,
        population,
        lower,
        upper,
        evolution_key,
        max_generations=search.max_generations,
        relative_tolerance=search.relative_tolerance,
        absolute_tolerance=search.absolute_tolerance,
    )
    generations_ = int(generations)
    population_vectors = lower + unit_population * (upper - lower)
    best_vector = lower + best_unit * (upper - lower)
    finite_population = bool(jnp.any(jnp.isfinite(population_objectives)))
    converged_ = bool(converged)
    if not finite_population:
        termination_reason = "no_finite_candidates"
    elif converged_ and generations_ == 0:
        termination_reason = "initial_population_converged"
    elif converged_:
        termination_reason = "fitness_tolerance"
    else:
        termination_reason = "max_generations"

    return _DifferentialEvolutionResult(
        population_vectors=population_vectors,
        population_objectives=population_objectives,
        best_vector=best_vector,
        raw_objective=raw_objective,
        best_objective_history=history[: generations_ + 1],
        lower_bounds=lower,
        upper_bounds=upper,
        key=key,
        search=search,
        converged=converged_,
        termination_reason=termination_reason,
        generations=generations_,
        objective_evaluations=search.population_size * (generations_ + 1),
        invalid_evaluations=int(invalid_evaluations),
        design_signature_=design_signature(search.design),
    )


__all__ = ["DifferentialEvolutionSearch"]
