#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from evosax.algorithms import DifferentialEvolution
from jaxtyping import Array, ArrayLike, Key

from ..._sampling import (
    design_signature,
    DesignLike,
    LatinHypercubeDesign,
    materialize_design,
    resolve_design,
)
from ..._strict import StrictModule
from ._constraints import DesignConstraintSystem
from ._schema import DesignState, ParameterId


SearchStrategy = Literal["best1bin", "rand1bin"]
SearchBounds = Mapping[ParameterId, tuple[ArrayLike, ArrayLike]]


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


class DesignSearchResult(StrictModule):
    """Best design state and convergence evidence from a bounded global search."""

    state: DesignState
    residual: Array
    residual_norm: Array
    objective: Array
    population_vectors: Array
    population_objectives: Array
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
        state: DesignState,
        residual: Array,
        objective: Array,
        population_vectors: Array,
        population_objectives: Array,
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
        design_signature: str,
    ):
        residual_ = jnp.asarray(residual, dtype=float).reshape((-1,))
        self.state = state
        self.residual = residual_
        self.residual_norm = jnp.linalg.norm(residual_)
        self.objective = jnp.asarray(objective, dtype=float).reshape(())
        self.population_vectors = jnp.asarray(population_vectors, dtype=float)
        self.population_objectives = jnp.asarray(population_objectives, dtype=float)
        self.best_objective_history = jnp.asarray(best_objective_history, dtype=float)
        self.lower_bounds = jnp.asarray(lower_bounds, dtype=float)
        self.upper_bounds = jnp.asarray(upper_bounds, dtype=float)
        self.key = key
        self.search = search
        self.converged = bool(converged)
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = int(objective_evaluations)
        self.invalid_evaluations = int(invalid_evaluations)
        self.design_signature = str(design_signature)


def _parameter_bound(
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
    *,
    parameter_id: ParameterId,
    side: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float).reshape((-1,))
    if array.shape != shape:
        raise ValueError(
            f"{side} search bound for {parameter_id} must be scalar or have shape "
            f"{shape}, got {array.shape}."
        )
    return array.reshape((-1,))


def _resolve_search_bounds(
    system: DesignConstraintSystem,
    bounds: SearchBounds | None,
    initial_state: DesignState,
    /,
) -> tuple[Array, Array, Array]:
    if bounds is not None and not isinstance(bounds, Mapping):
        raise TypeError("bounds must be a mapping from ParameterId to (lower, upper).")
    overrides = {} if bounds is None else dict(bounds)
    schema = system.geometry.schema
    index_by_id = {spec.parameter_id: index for index, spec in enumerate(schema.specs)}
    slices_by_index = {
        index: slice_info
        for index, slice_info in zip(
            system.trainable_indices,
            system.slices,
            strict=True,
        )
    }
    lower = np.asarray(system.lower_bounds, dtype=float).copy()
    upper = np.asarray(system.upper_bounds, dtype=float).copy()

    for parameter_id, pair in overrides.items():
        if not isinstance(parameter_id, ParameterId):
            raise TypeError("Every search-bound key must be a ParameterId.")
        if parameter_id not in index_by_id:
            raise KeyError(f"Unknown geometry parameter {parameter_id}.")
        index = index_by_id[parameter_id]
        if index not in slices_by_index:
            raise ValueError(
                f"Search bounds were provided for non-trainable {parameter_id}."
            )
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                f"Search bounds for {parameter_id} must be a (lower, upper) tuple."
            )
        start, stop, shape = slices_by_index[index]
        lower_ = _parameter_bound(pair[0], shape, parameter_id=parameter_id, side="Lower")
        upper_ = _parameter_bound(pair[1], shape, parameter_id=parameter_id, side="Upper")
        if not np.all(np.isfinite(lower_)) or not np.all(np.isfinite(upper_)):
            raise ValueError(f"Search bounds for {parameter_id} must be finite.")
        if np.any(lower_ >= upper_):
            raise ValueError(
                f"Every lower search bound for {parameter_id} must be smaller "
                "than its upper bound."
            )
        physical_lower, physical_upper = schema.specs[index].bounds
        if physical_lower is not None and np.any(lower_ < physical_lower):
            raise ValueError(
                f"Search bounds for {parameter_id} extend below the physical lower "
                f"bound {physical_lower}."
            )
        if physical_upper is not None and np.any(upper_ > physical_upper):
            raise ValueError(
                f"Search bounds for {parameter_id} extend above the physical upper "
                f"bound {physical_upper}."
            )
        lower[start:stop] = lower_
        upper[start:stop] = upper_

    missing = []
    for index, (start, stop, _shape) in slices_by_index.items():
        if np.any(~np.isfinite(lower[start:stop])) or np.any(
            ~np.isfinite(upper[start:stop])
        ):
            missing.append(str(schema.specs[index].parameter_id))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Finite search bounds are required for: {names}.")
    if np.any(lower >= upper):
        raise ValueError("Every lower search bound must be smaller than its upper bound.")

    initial = np.asarray(system.pack(initial_state), dtype=float)
    outside = (initial < lower) | (initial > upper)
    if np.any(outside):
        raise ValueError("The initial design state lies outside the search bounds.")
    initial_unit = (initial - lower) / (upper - lower)
    return (
        jnp.asarray(lower, dtype=float),
        jnp.asarray(upper, dtype=float),
        jnp.asarray(initial_unit, dtype=float),
    )


def _reflect_unit_box(population: Array, /) -> Array:
    folded = jnp.mod(population, 2.0)
    return jnp.where(folded <= 1.0, folded, 2.0 - folded)


def _population_objectives(
    population: Array,
    system: DesignConstraintSystem,
    base_state: DesignState,
    lower_bounds: Array,
    upper_bounds: Array,
    /,
) -> Array:
    def objective(unit_vector):
        vector = lower_bounds + unit_vector * (upper_bounds - lower_bounds)
        residual = system.residual(system.unpack(vector, base_state=base_state))
        return jnp.sum(residual * residual)

    return jax.vmap(objective)(population)


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
) -> DifferentialEvolution:
    return DifferentialEvolution(
        population_size=population_size,
        solution=jnp.zeros((dimension,), dtype=np.dtype(dtype_name)),
    )


@eqx.filter_jit
def _run_search(
    system: DesignConstraintSystem,
    base_state: DesignState,
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
        system,
        base_state,
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
    history = jnp.full((max_generations + 1,), jnp.inf, dtype=initial_fitness.dtype)
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
            system,
            base_state,
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
    return generation, population, state.fitness, best, history, invalid, converged


def search_design_constraints(
    system: DesignConstraintSystem,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: Key[Array, ""],
    bounds: SearchBounds | None = None,
    initial_state: DesignState | None = None,
) -> DesignSearchResult:
    """Run bounded differential evolution over a compiled geometry design state."""
    if not isinstance(system, DesignConstraintSystem):
        raise TypeError("system must be a DesignConstraintSystem.")
    if not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be a DifferentialEvolutionSearch.")
    if system.geometry.field_certificate.validity_region != "all_space":
        raise NotImplementedError(
            "Global design search requires a geometry validity region of 'all_space'; "
            f"got {system.geometry.field_certificate.validity_region!r}."
        )
    state = system.geometry.state if initial_state is None else initial_state
    if not isinstance(state, DesignState):
        raise TypeError("initial_state must be a DesignState or None.")
    lower, upper, initial_unit = _resolve_search_bounds(system, bounds, state)
    dimension = int(lower.shape[0])
    design_key = jr.fold_in(key, 0)
    evolution_key = jr.fold_in(key, 1)
    population = materialize_design(
        search.design,
        count=search.population_size,
        dimension=dimension,
        key=design_key,
    )
    population = population.at[0].set(initial_unit)

    algorithm = _differential_evolution_algorithm(
        search.population_size,
        dimension,
        str(lower.dtype),
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
        history,
        invalid_evaluations,
        converged,
    ) = _run_search(
        system,
        state,
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
    best_state = system.unpack(best_vector, base_state=state)
    residual = system.residual(best_state)
    objective = jnp.sum(residual * residual)
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

    return DesignSearchResult(
        state=best_state,
        residual=residual,
        objective=objective,
        population_vectors=population_vectors,
        population_objectives=population_objectives,
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
        design_signature=design_signature(search.design),
    )


__all__ = [
    "DesignSearchResult",
    "DifferentialEvolutionSearch",
]
