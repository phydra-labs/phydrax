#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._strict import StrictModule
from ..optim import DifferentialEvolutionSearch
from ..optim._differential_evolution import _bounded_differential_evolution
from ..optim._pytree import _PyTreeVectorizer
from ._bayesian_optimization import (
    bayesian_optimize,
    BayesianOptimizationDomain,
    BayesianOptimizationPoint,
    BayesianOptimizationProblem,
    BayesianOptimizationResult,
    GaussianProcessBayesianOptimization,
)
from ._posterior import PosteriorProblem


PositionBounds = tuple[PyTree[ArrayLike], PyTree[ArrayLike]]


class _PosteriorObjective(StrictModule):
    problem: PosteriorProblem
    vectorizer: _PyTreeVectorizer

    def __init__(self, problem: PosteriorProblem, vectorizer: _PyTreeVectorizer, /):
        self.problem = problem
        self.vectorizer = vectorizer

    def __call__(self, vector: Array, /) -> Array:
        return self.problem.negative_log_density(self.vectorizer.unravel(vector))


class MAPSearchResult(StrictModule):
    """Posterior mode candidate and population evidence from differential evolution."""

    problem: PosteriorProblem
    position: PyTree[Array]
    parameters: PyTree[Array]
    objective: Array
    log_density: Array
    population_positions: PyTree[Array]
    population_objectives: Array
    best_objective_history: Array
    lower_bounds: PyTree[Array]
    upper_bounds: PyTree[Array]
    key: Key[Array, ""]
    search: DifferentialEvolutionSearch
    population_converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem,
        position,
        objective,
        population_positions,
        population_objectives,
        best_objective_history,
        lower_bounds,
        upper_bounds,
        key,
        search,
        population_converged,
        termination_reason,
        generations,
        objective_evaluations,
        invalid_evaluations,
        design_signature,
    ):
        position_ = jax.tree_util.tree_map(jnp.asarray, position)
        objective_ = jnp.asarray(objective, dtype=float).reshape(())
        self.problem = problem
        self.position = position_
        self.parameters = problem.parameter_space.constrain(position_)
        self.objective = objective_
        self.log_density = -objective_
        self.population_positions = jax.tree_util.tree_map(
            jnp.asarray, population_positions
        )
        self.population_objectives = jnp.asarray(population_objectives, dtype=float)
        self.best_objective_history = jnp.asarray(best_objective_history, dtype=float)
        self.lower_bounds = jax.tree_util.tree_map(jnp.asarray, lower_bounds)
        self.upper_bounds = jax.tree_util.tree_map(jnp.asarray, upper_bounds)
        self.key = key
        self.search = search
        self.population_converged = bool(population_converged)
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = int(objective_evaluations)
        self.invalid_evaluations = int(invalid_evaluations)
        self.design_signature = str(design_signature)

    @property
    def valid(self) -> bool:
        return (
            bool(jnp.isfinite(self.objective))
            and self.termination_reason != "no_finite_candidates"
        )


class BayesianOptimizationMAPResult(StrictModule):
    """Posterior mode with the complete canonical Bayesian-optimization evidence."""

    problem: PosteriorProblem
    position: PyTree[Array]
    parameters: PyTree[Array]
    objective: Array
    log_density: Array
    evaluated_positions: PyTree[Array]
    lower_bounds: PyTree[Array]
    upper_bounds: PyTree[Array]
    evidence: BayesianOptimizationResult
    valid: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)


def search_map(
    problem: PosteriorProblem,
    search: DifferentialEvolutionSearch | GaussianProcessBayesianOptimization,
    /,
    *,
    key: Key[Array, ""],
    position_bounds: PositionBounds,
    initial_position: PyTree[Array] | None = None,
) -> MAPSearchResult | BayesianOptimizationMAPResult:
    """Search a bounded posterior with DE or the canonical q=1 continuous BO plan."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if not isinstance(
        search, (DifferentialEvolutionSearch, GaussianProcessBayesianOptimization)
    ):
        raise TypeError(
            "search must be DifferentialEvolutionSearch or GaussianProcessBayesianOptimization."
        )
    if not isinstance(position_bounds, tuple) or len(position_bounds) != 2:
        raise TypeError("position_bounds must be a (lower, upper) PyTree tuple.")
    vectorizer = _PyTreeVectorizer(problem.initial_position)
    position = problem.initial_position if initial_position is None else initial_position
    initial_vector = vectorizer.ravel(position, name="initial_position")
    lower_vector = vectorizer.ravel_bound(position_bounds[0], side="Lower")
    upper_vector = vectorizer.ravel_bound(position_bounds[1], side="Upper")
    objective = _PosteriorObjective(problem, vectorizer)
    if isinstance(search, DifferentialEvolutionSearch):
        result = _bounded_differential_evolution(
            objective, initial_vector, lower_vector, upper_vector, search, key=key
        )
        return MAPSearchResult(
            problem=problem,
            position=vectorizer.unravel(result.best_vector),
            objective=result.raw_objective,
            population_positions=vectorizer.unravel_population(result.population_vectors),
            population_objectives=result.population_objectives,
            best_objective_history=result.best_objective_history,
            lower_bounds=vectorizer.unravel(result.lower_bounds),
            upper_bounds=vectorizer.unravel(result.upper_bounds),
            key=result.key,
            search=result.search,
            population_converged=result.converged,
            termination_reason=result.termination_reason,
            generations=result.generations,
            objective_evaluations=result.objective_evaluations,
            invalid_evaluations=result.invalid_evaluations,
            design_signature=result.design_signature,
        )
    if search.batch_size != 1:
        raise ValueError("search_map requires Bayesian optimization batch_size == 1.")
    domain = BayesianOptimizationDomain(
        initial_vector,
        lower_bounds=lower_vector,
        upper_bounds=upper_vector,
    )
    bo_problem = BayesianOptimizationProblem(
        lambda point: objective(_map_continuous(point)),
        domain,
    )
    result = bayesian_optimize(bo_problem, search, key)
    best_vector = jnp.asarray(result.best_point.continuous)
    objective_value = result.best_objective
    return BayesianOptimizationMAPResult(
        problem=problem,
        position=vectorizer.unravel(best_vector),
        parameters=problem.parameter_space.constrain(vectorizer.unravel(best_vector)),
        objective=objective_value,
        log_density=-objective_value,
        evaluated_positions=vectorizer.unravel_population(
            result.evaluated_continuous_unit * (upper_vector - lower_vector)
            + lower_vector
        ),
        lower_bounds=vectorizer.unravel(lower_vector),
        upper_bounds=vectorizer.unravel(upper_vector),
        evidence=result,
        valid=bool(jnp.isfinite(objective_value)),
        termination_reason=result.termination_reason,
        objective_evaluations=result.evaluation_count,
    )


def _map_continuous(point: BayesianOptimizationPoint, /) -> Array:
    if point.categorical is not None:
        raise ValueError("MAP Bayesian optimization does not admit categorical axes.")
    return jnp.asarray(point.continuous)


__all__ = ["BayesianOptimizationMAPResult", "MAPSearchResult", "search_map"]
