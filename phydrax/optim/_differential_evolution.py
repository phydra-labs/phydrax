#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._fingerprint import canonical_fingerprint
from .._sampling import (
    design_signature,
    DesignLike,
    LatinHypercubeDesign,
    materialize_design,
    resolve_design,
)
from .._strict import StrictModule
from ._bounded_search import _BoundedVectorDomain
from ._finite import FiniteAxis


SearchStrategy = Literal["best1bin", "rand1bin"]
DifferentialEvolutionSelection = Literal["scalar", "pareto"]
DifferentialEvolutionValidityMode = Literal["guarded", "vectorized"]


class DifferentialEvolutionStatus(IntEnum):
    COMPLETE = 0
    NO_VALID_CANDIDATES = 1
    WORK_LIMIT = 2


class DifferentialEvolutionContinuous(StrictModule):
    lower: Array
    upper: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)

    def __init__(self, lower: ArrayLike, upper: ArrayLike, /):
        lower_, upper_ = np.broadcast_arrays(np.asarray(lower), np.asarray(upper))
        dtype = np.result_type(lower_.dtype, upper_.dtype, np.float32)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("Continuous bounds must be real floating values.")
        lower_, upper_ = lower_.astype(dtype), upper_.astype(dtype)
        if not np.all(np.isfinite(lower_)) or not np.all(np.isfinite(upper_)):
            raise ValueError("Continuous bounds must be finite.")
        if np.any(lower_ >= upper_):
            raise ValueError("Every continuous lower bound must be smaller than upper.")
        self.lower, self.upper = jnp.asarray(lower_), jnp.asarray(upper_)
        self.shape, self.size = lower_.shape, int(lower_.size)


class DifferentialEvolutionInteger(StrictModule):
    lower: Array
    upper: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)

    def __init__(self, lower: ArrayLike, upper: ArrayLike, /):
        lower_, upper_ = np.broadcast_arrays(np.asarray(lower), np.asarray(upper))
        if not np.issubdtype(lower_.dtype, np.integer) or not np.issubdtype(
            upper_.dtype, np.integer
        ):
            raise TypeError("Integer bounds must use integer dtypes.")
        if np.any(lower_ > upper_):
            raise ValueError("Every integer lower bound must not exceed its upper bound.")
        self.lower, self.upper = jnp.asarray(lower_), jnp.asarray(upper_)
        self.shape, self.size = lower_.shape, int(lower_.size)


class DifferentialEvolutionCategorical(StrictModule):
    axis: FiniteAxis

    def __init__(self, axis: FiniteAxis, /):
        if not isinstance(axis, FiniteAxis):
            raise TypeError("axis must be a FiniteAxis.")
        self.axis = axis


DifferentialEvolutionLeaf: TypeAlias = (
    DifferentialEvolutionContinuous
    | DifferentialEvolutionInteger
    | DifferentialEvolutionCategorical
)


def _is_de_leaf(value: Any, /) -> bool:
    return isinstance(
        value,
        (
            DifferentialEvolutionContinuous,
            DifferentialEvolutionInteger,
            DifferentialEvolutionCategorical,
        ),
    )


class DifferentialEvolutionSpace(StrictModule):
    """Static mixed continuous, integer, and categorical search space."""

    leaves: tuple[DifferentialEvolutionLeaf, ...]
    tree_definition: Any = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    integer_columns: tuple[int, ...] = eqx.field(static=True)
    categorical_columns: tuple[int, ...] = eqx.field(static=True)
    categorical_sizes: tuple[int, ...] = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(self, leaves: PyTree[DifferentialEvolutionLeaf], /):
        flat, definition = jax.tree_util.tree_flatten(leaves, is_leaf=_is_de_leaf)
        if not flat or any(not _is_de_leaf(value) for value in flat):
            raise TypeError("Every DifferentialEvolutionSpace leaf must be a DE leaf.")
        offsets = [0]
        integer_columns: list[int] = []
        categorical_columns: list[int] = []
        categorical_sizes: list[int] = []
        payload = []
        for leaf in flat:
            width = 1 if isinstance(leaf, DifferentialEvolutionCategorical) else leaf.size
            start = offsets[-1]
            offsets.append(start + width)
            if isinstance(leaf, DifferentialEvolutionInteger):
                integer_columns.extend(range(start, start + width))
                payload.append(("integer", leaf.shape, str(leaf.lower.dtype)))
            elif isinstance(leaf, DifferentialEvolutionCategorical):
                categorical_columns.append(start)
                categorical_sizes.append(leaf.axis.size)
                payload.append(("categorical", leaf.axis.size, leaf.axis.dtypes))
            else:
                payload.append(("continuous", leaf.shape, str(leaf.lower.dtype)))
        self.leaves = tuple(flat)
        self.tree_definition = definition
        self.offsets = tuple(offsets)
        self.dimension = offsets[-1]
        self.integer_columns = tuple(integer_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.categorical_sizes = tuple(categorical_sizes)
        self.space_id = canonical_fingerprint(
            {"kind": "differential-evolution-space", "payload": payload}
        )

    def decode(self, unit_vector: ArrayLike, /) -> PyTree[Array]:
        unit = jnp.asarray(unit_vector)
        if unit.shape[-1:] != (self.dimension,):
            raise ValueError(f"Encoded candidates must end in shape ({self.dimension},).")
        decoded = []
        for leaf, start, stop in zip(
            self.leaves, self.offsets[:-1], self.offsets[1:], strict=True
        ):
            values = unit[..., start:stop]
            if isinstance(leaf, DifferentialEvolutionContinuous):
                physical = leaf.lower.reshape((-1,)) + values * (
                    leaf.upper - leaf.lower
                ).reshape((-1,))
                decoded.append(physical.reshape(unit.shape[:-1] + leaf.shape))
            elif isinstance(leaf, DifferentialEvolutionInteger):
                span = (leaf.upper - leaf.lower).reshape((-1,))
                physical = leaf.lower.reshape((-1,)) + jnp.rint(values * span).astype(
                    leaf.lower.dtype
                )
                decoded.append(physical.reshape(unit.shape[:-1] + leaf.shape))
            else:
                index = jnp.rint(values[..., 0] * max(leaf.axis.size - 1, 1)).astype(
                    jnp.int32
                )
                decoded.append(leaf.axis._take_unchecked(index))
        return self.tree_definition.unflatten(decoded)


class DifferentialEvolutionSearch(StrictModule):
    population_size: int = eqx.field(static=True)
    max_generations: int = eqx.field(static=True)
    strategy: SearchStrategy = eqx.field(static=True)
    selection: DifferentialEvolutionSelection = eqx.field(static=True)
    objective_count: int = eqx.field(static=True)
    archive_capacity: int = eqx.field(static=True)
    validity_mode: DifferentialEvolutionValidityMode = eqx.field(static=True)
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
        selection: DifferentialEvolutionSelection = "scalar",
        objective_count: int = 1,
        archive_capacity: int | None = None,
        validity_mode: DifferentialEvolutionValidityMode = "guarded",
        differential_weight: float = 0.8,
        crossover_rate: float = 0.9,
        relative_tolerance: float = 0.01,
        absolute_tolerance: float = 0.0,
        design: DesignLike = LatinHypercubeDesign(),
    ):
        population, generations = int(population_size), int(max_generations)
        if population < 4:
            raise ValueError("population_size must be at least 4.")
        if generations < 0:
            raise ValueError("max_generations must be non-negative.")
        if strategy not in ("best1bin", "rand1bin"):
            raise ValueError("Unknown differential-evolution strategy.")
        if selection not in ("scalar", "pareto") or validity_mode not in (
            "guarded",
            "vectorized",
        ):
            raise ValueError("Unknown selection or validity mode.")
        objectives = int(objective_count)
        if objectives <= 0 or (selection == "scalar" and objectives != 1):
            raise ValueError(
                "Scalar selection requires one objective; Pareto count is positive."
            )
        capacity = population if archive_capacity is None else int(archive_capacity)
        if capacity <= 0:
            raise ValueError("archive_capacity must be positive.")
        weight, crossover = float(differential_weight), float(crossover_rate)
        relative, absolute = float(relative_tolerance), float(absolute_tolerance)
        if not np.isfinite(weight) or not 0.0 <= weight < 2.0:
            raise ValueError("differential_weight must lie in [0, 2).")
        if not np.isfinite(crossover) or not 0.0 <= crossover <= 1.0:
            raise ValueError("crossover_rate must lie in [0, 1].")
        if not np.isfinite(relative) or relative < 0.0:
            raise ValueError("relative_tolerance must be finite and non-negative.")
        if not np.isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_tolerance must be finite and non-negative.")
        self.population_size, self.max_generations = population, generations
        self.strategy, self.selection = strategy, selection
        self.objective_count, self.archive_capacity = objectives, capacity
        self.validity_mode = validity_mode
        self.differential_weight, self.crossover_rate = weight, crossover
        self.relative_tolerance, self.absolute_tolerance = relative, absolute
        self.design = resolve_design(design)


class DifferentialEvolutionResult(StrictModule):
    population: PyTree[Array]
    population_vectors: Array
    population_objectives: Array
    population_valid: Array
    nondominated_mask: Array
    crowding: Array
    archive: PyTree[Array]
    archive_vectors: Array
    archive_objectives: Array
    archive_valid: Array
    best: PyTree[Array]
    best_vector: Array
    raw_objective: Array
    best_objective_history: Array
    lower_bounds: Array
    upper_bounds: Array
    key: Key[Array, ""]
    search: DifferentialEvolutionSearch
    status: Array
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)
    space_id: str = eqx.field(static=True)


def _reflect_unit_box(population: Array, /) -> Array:
    folded = jnp.mod(population, 2.0)
    return jnp.where(folded <= 1.0, folded, 2.0 - folded)


def _dominance(objectives: Array, valid: Array, /) -> Array:
    left = objectives[:, None, :]
    right = objectives[None, :, :]
    return (
        valid[:, None]
        & valid[None, :]
        & jnp.all(left <= right, axis=-1)
        & jnp.any(left < right, axis=-1)
    )


def _ranks_and_crowding(objectives: Array, valid: Array, /) -> tuple[Array, Array]:
    count, objective_count = objectives.shape
    dominance = _dominance(objectives, valid)
    ranks = jnp.full((count,), count, dtype=jnp.int32)
    remaining = valid

    def assign(rank, state):
        ranks_, remaining_ = state
        dominated = jnp.any(dominance & remaining_[:, None], axis=0)
        front = remaining_ & ~dominated
        ranks_ = jnp.where(front, rank, ranks_)
        return ranks_, remaining_ & ~front

    ranks, _ = jax.lax.fori_loop(0, count, assign, (ranks, remaining))
    crowding = jnp.zeros((count,), dtype=objectives.dtype)
    for rank in range(count):
        front = valid & (ranks == rank)
        front_count = jnp.sum(front, dtype=jnp.int32)
        for objective in range(objective_count):
            values = objectives[:, objective]
            order = jnp.argsort(jnp.where(front, values, jnp.inf), stable=True)
            ordered_valid = front[order]
            ordered = values[order]
            position = jnp.arange(count)
            previous = ordered[jnp.maximum(position - 1, 0)]
            following = ordered[jnp.minimum(position + 1, count - 1)]
            finite_values = jnp.where(front, values, jnp.nan)
            span = jnp.nanmax(finite_values) - jnp.nanmin(finite_values)
            contribution = jnp.where(span > 0.0, (following - previous) / span, 0.0)
            boundary = (position == 0) | (position == front_count - 1)
            contribution = jnp.where(ordered_valid & boundary, jnp.inf, contribution)
            contribution = jnp.where(ordered_valid, contribution, 0.0)
            crowding = crowding.at[order].add(contribution)
    return ranks, crowding


def _select_union(vectors, objectives, valid, capacity):
    ranks, crowding = _ranks_and_crowding(objectives, valid)
    indices = jnp.arange(vectors.shape[0], dtype=jnp.int32)
    order = jnp.argsort(indices, stable=True)
    order = order[jnp.argsort(-crowding[order], stable=True)]
    order = order[jnp.argsort(ranks[order], stable=True)]
    selected = order[:capacity]
    return (
        vectors[selected],
        objectives[selected],
        valid[selected],
        ranks[selected],
        crowding[selected],
    )


def _evaluate_population(objective, validity, space, vectors, search):
    decoded = space.decode(vectors)
    if validity is None:
        valid = jnp.ones((vectors.shape[0],), dtype=bool)
    else:
        valid = jax.vmap(validity)(decoded)
    if search.validity_mode == "guarded":

        def one(arguments):
            candidate, accepted = arguments
            return jax.lax.cond(
                accepted,
                lambda value: jnp.asarray(objective(value)).reshape(
                    (search.objective_count,)
                ),
                lambda _value: jnp.full(
                    (search.objective_count,), jnp.nan, dtype=vectors.dtype
                ),
                candidate,
            )

        objectives = jax.lax.map(one, (decoded, valid))
    else:
        raw = jax.vmap(objective)(decoded)
        objectives = jnp.asarray(raw).reshape((vectors.shape[0], search.objective_count))
    finite = jnp.all(jnp.isfinite(objectives), axis=-1)
    valid = valid & finite
    return decoded, jnp.where(valid[:, None], objectives, jnp.inf), valid


def _round_integer_columns(space, vectors, key):
    if not space.integer_columns:
        return vectors
    columns = jnp.asarray(space.integer_columns, dtype=jnp.int32)
    values = vectors[:, columns]
    # Unit-lattice resolution is derived per integer leaf and kept static.
    spans = [
        jnp.ravel((leaf.upper - leaf.lower).astype(vectors.dtype))
        for leaf in space.leaves
        if isinstance(leaf, DifferentialEvolutionInteger)
    ]
    span = jnp.concatenate(spans)
    scaled = values * span
    base = jnp.floor(scaled)
    rounded = base + jr.bernoulli(key, scaled - base)
    unit = jnp.where(span > 0, rounded / span, 0.0)
    return vectors.at[:, columns].set(unit)


def _categorical_mutant(space, population, a, b, c, key):
    if not space.categorical_columns:
        return population[a]
    columns = jnp.asarray(space.categorical_columns, dtype=jnp.int32)
    choices = jr.randint(key, (population.shape[0], columns.size), 0, 3)
    parents = jnp.stack(
        (population[a][:, columns], population[b][:, columns], population[c][:, columns]),
        axis=-1,
    )
    selected = jnp.take_along_axis(parents, choices[..., None], axis=-1)[..., 0]
    equal_difference = population[b][:, columns] == population[c][:, columns]
    selected = jnp.where(equal_difference, population[a][:, columns], selected)
    return population[a].at[:, columns].set(selected)


@eqx.filter_jit
def _run_differential_evolution(
    objective, validity, space, search, initial_population, key
):
    population = initial_population
    _, objectives, valid = _evaluate_population(
        objective, validity, space, population, search
    )
    history = jnp.full((search.max_generations + 1,), jnp.inf, dtype=objectives.dtype)
    history = history.at[0].set(jnp.min(objectives[:, 0]))
    invalid = jnp.sum(~valid, dtype=jnp.int32)
    state = (
        jnp.asarray(0, dtype=jnp.int32),
        population,
        objectives,
        valid,
        key,
        invalid,
        history,
    )

    def condition(state_):
        generation, _, objectives_, valid_, _, _, _ = state_
        finite = jnp.where(valid_, objectives_[:, 0], jnp.nan)
        mean = jnp.nanmean(finite)
        spread = jnp.nanstd(finite)
        converged = (
            (search.selection == "scalar")
            & jnp.any(valid_)
            & jnp.isfinite(spread)
            & (
                spread
                <= search.absolute_tolerance + search.relative_tolerance * jnp.abs(mean)
            )
        )
        return (generation < search.max_generations) & jnp.any(valid_) & ~converged

    def step(state_):
        (
            generation,
            population_,
            objectives_,
            valid_,
            root_key,
            invalid_,
            history_,
        ) = state_
        generation_key = jr.fold_in(root_key, generation)
        mutation_key, rounding_key, categorical_key, crossover_key = jr.split(
            generation_key, 4
        )
        scores = jr.uniform(
            mutation_key, (search.population_size, search.population_size)
        )
        scores = jnp.where(jnp.eye(search.population_size, dtype=bool), jnp.inf, scores)
        donors = jnp.argsort(scores, axis=-1)[:, :3]
        a, b, c = donors[:, 0], donors[:, 1], donors[:, 2]
        if search.strategy == "best1bin":
            safe = jnp.where(valid_, objectives_[:, 0], jnp.inf)
            base = jnp.broadcast_to(population_[jnp.argmin(safe)], population_.shape)
        else:
            base = population_[a]
        mutant = _reflect_unit_box(
            base + search.differential_weight * (population_[b] - population_[c])
        )
        categorical = _categorical_mutant(space, population_, a, b, c, categorical_key)
        if space.categorical_columns:
            columns = jnp.asarray(space.categorical_columns, dtype=jnp.int32)
            mutant = mutant.at[:, columns].set(categorical[:, columns])
        mutant = _round_integer_columns(space, mutant, rounding_key)
        crossover_mask_key, forced_key = jr.split(crossover_key)
        crossover = jr.bernoulli(
            crossover_mask_key, search.crossover_rate, population_.shape
        )
        forced = jr.randint(forced_key, (search.population_size,), 0, space.dimension)
        crossover = crossover.at[jnp.arange(search.population_size), forced].set(True)
        trial = jnp.where(crossover, mutant, population_)
        _, trial_objectives, trial_valid = _evaluate_population(
            objective, validity, space, trial, search
        )
        if search.selection == "scalar":
            wins = trial_valid & (~valid_ | (trial_objectives[:, 0] <= objectives_[:, 0]))
            population_ = jnp.where(wins[:, None], trial, population_)
            objectives_ = jnp.where(wins[:, None], trial_objectives, objectives_)
            valid_ = valid_ | trial_valid
        else:
            population_, objectives_, valid_, _ranks, _crowding = _select_union(
                jnp.concatenate((population_, trial), axis=0),
                jnp.concatenate((objectives_, trial_objectives), axis=0),
                jnp.concatenate((valid_, trial_valid), axis=0),
                search.population_size,
            )
        next_generation = generation + jnp.asarray(1, dtype=generation.dtype)
        history_ = history_.at[next_generation].set(jnp.min(objectives_[:, 0]))
        return (
            next_generation,
            population_,
            objectives_,
            valid_,
            root_key,
            invalid_ + jnp.sum(~trial_valid, dtype=jnp.int32),
            history_,
        )

    generation, population, objectives, valid, key, invalid, history = jax.lax.while_loop(
        condition, step, state
    )
    return generation, population, objectives, valid, invalid, history


def search_differential_evolution(
    objective: Callable[[PyTree[Array]], Array],
    space: DifferentialEvolutionSpace,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: Key[Array, ""],
    validity: Callable[[PyTree[Array]], Array] | None = None,
    initial: ArrayLike | None = None,
) -> DifferentialEvolutionResult:
    """Search a static mixed space with scalar or fixed-objective Pareto selection."""
    if not callable(objective) or (validity is not None and not callable(validity)):
        raise TypeError("objective and optional validity must be callable.")
    if not isinstance(space, DifferentialEvolutionSpace) or not isinstance(
        search, DifferentialEvolutionSearch
    ):
        raise TypeError("space and search have incorrect types.")
    design_key, evolution_key = jr.split(key)
    population = jnp.asarray(
        materialize_design(
            search.design,
            count=search.population_size,
            dimension=space.dimension,
            key=design_key,
        ),
        dtype=float,
    )
    if initial is not None:
        initial_ = jnp.asarray(initial, dtype=population.dtype)
        if initial_.shape != (space.dimension,):
            raise ValueError("initial must have shape (space.dimension,).")
        population = population.at[0].set(initial_)
    population = _round_integer_columns(space, population, jr.fold_in(design_key, 1))
    if space.categorical_columns:
        columns = jnp.asarray(space.categorical_columns, dtype=jnp.int32)
        sizes = jnp.asarray(space.categorical_sizes, dtype=population.dtype)
        categorical = jnp.floor(population[:, columns] * sizes).astype(jnp.int32)
        categorical = jnp.minimum(categorical, sizes.astype(jnp.int32) - 1)
        denominator = jnp.maximum(sizes - 1.0, 1.0)
        population = population.at[:, columns].set(categorical / denominator)
    generation, vectors, objectives, valid, invalid, history = (
        _run_differential_evolution(
            objective, validity, space, search, population, evolution_key
        )
    )
    generations = int(np.asarray(generation))
    ranks, crowding = _ranks_and_crowding(objectives, valid)
    nondominated = valid & (ranks == 0)
    (
        archive_vectors,
        archive_objectives,
        archive_valid,
        archive_ranks,
        _archive_crowding,
    ) = _select_union(
        vectors,
        objectives,
        valid,
        min(search.archive_capacity, search.population_size),
    )
    archive_valid = archive_valid & (archive_ranks == 0)
    if search.archive_capacity > search.population_size:
        padding = search.archive_capacity - search.population_size
        archive_vectors = jnp.pad(archive_vectors, ((0, padding), (0, 0)))
        archive_objectives = jnp.pad(
            archive_objectives,
            ((0, padding), (0, 0)),
            constant_values=jnp.nan,
        )
        archive_valid = jnp.pad(archive_valid, ((0, padding),), constant_values=False)
    best_index = jnp.argmin(jnp.where(valid, objectives[:, 0], jnp.inf))
    any_valid = bool(np.asarray(jnp.any(valid)))
    best_vector = (
        vectors[best_index] if any_valid else jnp.full((space.dimension,), jnp.nan)
    )
    decoded = space.decode(vectors)
    best = space.decode(best_vector)
    archive = space.decode(archive_vectors)
    finite_scalar = jnp.where(valid, objectives[:, 0], jnp.nan)
    mean = jnp.nanmean(finite_scalar)
    spread = jnp.nanstd(finite_scalar)
    converged = bool(
        np.asarray(
            jnp.asarray(any_valid)
            & jnp.isfinite(spread)
            & (
                spread
                <= search.absolute_tolerance + search.relative_tolerance * jnp.abs(mean)
            )
        )
    )
    status = (
        DifferentialEvolutionStatus.NO_VALID_CANDIDATES
        if not any_valid
        else DifferentialEvolutionStatus.COMPLETE
        if converged
        else DifferentialEvolutionStatus.WORK_LIMIT
    )
    termination = (
        "no_valid_candidates"
        if not any_valid
        else "initial_population_converged"
        if converged and generations == 0
        else "fitness_tolerance"
        if converged
        else "max_generations"
    )
    raw = objectives[best_index, 0] if any_valid else jnp.asarray(jnp.nan)
    return DifferentialEvolutionResult(
        decoded,
        vectors,
        objectives[:, 0] if search.selection == "scalar" else objectives,
        valid,
        nondominated,
        crowding,
        archive,
        archive_vectors,
        archive_objectives[:, 0] if search.selection == "scalar" else archive_objectives,
        archive_valid,
        best,
        best_vector,
        raw,
        history[: generations + 1],
        jnp.zeros((space.dimension,)),
        jnp.ones((space.dimension,)),
        key,
        search,
        jnp.asarray(int(status), dtype=jnp.int32),
        converged,
        termination,
        generations,
        search.population_size * (generations + 1),
        int(np.asarray(invalid)),
        design_signature(search.design),
        space.space_id,
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
    validity: Callable[[Array], Array] | None = None,
) -> DifferentialEvolutionResult:
    domain = _BoundedVectorDomain(initial_vector, lower_bounds, upper_bounds)
    leaf = DifferentialEvolutionContinuous(domain.lower, domain.upper)
    space = DifferentialEvolutionSpace(leaf)
    result = search_differential_evolution(
        objective,
        space,
        search,
        key=key,
        validity=validity,
        initial=domain.to_unit(domain.initial),
    )
    # Continuous single-leaf decoding is the physical vector expected by legacy adapters.
    return eqx.tree_at(
        lambda value: (
            value.population,
            value.population_vectors,
            value.best,
            value.best_vector,
            value.lower_bounds,
            value.upper_bounds,
        ),
        result,
        (
            result.population,
            result.population,
            result.best,
            result.best,
            domain.lower,
            domain.upper,
        ),
    )


__all__ = [
    "DifferentialEvolutionCategorical",
    "DifferentialEvolutionContinuous",
    "DifferentialEvolutionResult",
    "DifferentialEvolutionSearch",
    "DifferentialEvolutionSelection",
    "DifferentialEvolutionSpace",
    "DifferentialEvolutionStatus",
    "DifferentialEvolutionValidityMode",
    "search_differential_evolution",
]
