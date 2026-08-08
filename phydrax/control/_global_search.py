#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ..optim import DifferentialEvolutionSearch
from ..optim._differential_evolution import _bounded_differential_evolution
from ._multiple_shooting import _evaluate_held_control
from ._parameterization import (
    AbstractControlParameterization,
    PiecewiseConstantControlParameterization,
)
from ._problem import ControlProblem
from ._trajectory import ControlResult, ControlTrajectory


CoefficientBounds = tuple[ArrayLike, ArrayLike]


def _evaluate_control(
    problem: ControlProblem,
    parameterization: AbstractControlParameterization,
    coefficients: Array,
    solver_options: dict[str, Any],
    /,
) -> ControlResult:
    if isinstance(parameterization, PiecewiseConstantControlParameterization):
        return _evaluate_held_control(
            problem,
            parameterization,
            coefficients,
            solver_options=solver_options,
        )
    return problem.evaluate(parameterization, coefficients, **solver_options)


class _ControlObjective(StrictModule):
    problem: ControlProblem
    parameterization: AbstractControlParameterization
    solver_options: dict[str, Any]

    def __init__(
        self,
        problem: ControlProblem,
        parameterization: AbstractControlParameterization,
        solver_options: dict[str, Any],
        /,
    ):
        self.problem = problem
        self.parameterization = parameterization
        self.solver_options = solver_options

    def __call__(self, vector: Array, /) -> Array:
        coefficient_shape = (
            self.problem.case_shape + self.parameterization.parameter_shape
        )
        coefficients = jnp.reshape(vector, coefficient_shape)
        result = _evaluate_control(
            self.problem,
            self.parameterization,
            coefficients,
            self.solver_options,
        )
        objective = jnp.sum(result.sampled_loss.total)
        admissible = (
            jnp.all(result.successful)
            & jnp.all(result.feasibility.feasible)
            & jnp.isfinite(objective)
        )
        return jnp.where(admissible, objective, jnp.nan)


class ControlSearchResult(StrictModule):
    """Best control candidate found by a bounded differential-evolution search."""

    problem: ControlProblem
    parameterization: AbstractControlParameterization
    evaluation: ControlResult
    coefficients: Array
    objective: Array
    population_coefficients: Array
    population_objectives: Array
    best_objective_history: Array
    lower_bounds: Array
    upper_bounds: Array
    key: Key[Array, ""]
    search: DifferentialEvolutionSearch
    population_converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_candidates: int = eqx.field(static=True)
    valid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    control_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    rollout_method_id: str = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    parameter_shape: tuple[int, ...] = eqx.field(static=True)
    coefficient_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: ControlProblem,
        parameterization: AbstractControlParameterization,
        evaluation: ControlResult,
        coefficients: Array,
        objective: Array,
        population_coefficients: Array,
        population_objectives: Array,
        best_objective_history: Array,
        lower_bounds: Array,
        upper_bounds: Array,
        key: Key[Array, ""],
        search: DifferentialEvolutionSearch,
        population_converged: bool,
        termination_reason: str,
        generations: int,
        objective_evaluations: int,
        invalid_candidates: int,
        design_signature: str,
    ):
        invalid = int(invalid_candidates)
        evaluations = int(objective_evaluations)
        self.problem = problem
        self.parameterization = parameterization
        self.evaluation = evaluation
        self.coefficients = jnp.asarray(coefficients)
        self.objective = jnp.asarray(objective).reshape(())
        self.population_coefficients = jnp.asarray(population_coefficients)
        self.population_objectives = jnp.asarray(population_objectives)
        self.best_objective_history = jnp.asarray(best_objective_history)
        self.lower_bounds = jnp.asarray(lower_bounds)
        self.upper_bounds = jnp.asarray(upper_bounds)
        self.key = key
        self.search = search
        self.population_converged = bool(population_converged)
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = evaluations
        self.invalid_candidates = invalid
        self.valid_evaluations = evaluations - invalid
        self.design_signature = str(design_signature)
        self.problem_id = problem.problem_id
        self.dynamics_id = problem.dynamics.dynamics_id
        self.time_id = problem.time_grid.time_id
        self.parameterization_id = parameterization.parameterization_id
        self.approximation_id = parameterization.approximation_id
        self.result_id = (
            f"control-search:{problem.problem_id}:{parameterization.parameterization_id}"
        )
        self.method_id = "bounded-differential-evolution-control-search"
        self.rollout_method_id = evaluation.method_id
        self.control_id = evaluation.trajectory.control_id
        self.control_shape = parameterization.control_shape
        self.case_shape = problem.case_shape
        self.parameter_shape = parameterization.parameter_shape
        self.coefficient_shape = problem.case_shape + parameterization.parameter_shape

    @property
    def trajectory(self) -> ControlTrajectory:
        """Best-found trajectory, directly usable as a local-control seed."""
        return self.evaluation.trajectory

    @property
    def controls(self) -> Array:
        """Best-found sampled controls with explicit case and time axes."""
        return self.trajectory.controls

    @property
    def successful(self) -> Array:
        """Whether the best-found candidate is valid, finite, and feasible."""
        return (
            jnp.all(self.evaluation.successful)
            & jnp.all(self.evaluation.feasibility.feasible)
            & jnp.isfinite(self.objective)
        )


def _coefficient_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(
            f"{name} must have coefficient layout {shape}; got {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be real-valued floating-point coefficients.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _resolve_coefficients_and_bounds(
    coefficient_shape: tuple[int, ...],
    coefficient_bounds: CoefficientBounds,
    initial_coefficients: ArrayLike | None,
    /,
) -> tuple[Array, Array, Array]:
    if not isinstance(coefficient_bounds, tuple) or len(coefficient_bounds) != 2:
        raise TypeError("coefficient_bounds must be a (lower, upper) tuple.")
    shape = coefficient_shape
    lower = _coefficient_array(
        coefficient_bounds[0],
        shape,
        name="Lower coefficient bound",
    )
    upper = _coefficient_array(
        coefficient_bounds[1],
        shape,
        name="Upper coefficient bound",
    )
    if initial_coefficients is None:
        initial_array = None
        dtype = np.result_type(lower.dtype, upper.dtype)
    else:
        initial_array = _coefficient_array(
            initial_coefficients,
            shape,
            name="initial_coefficients",
        )
        dtype = np.result_type(lower.dtype, upper.dtype, initial_array.dtype)
    lower = lower.astype(dtype, copy=False)
    upper = upper.astype(dtype, copy=False)
    if np.any(lower >= upper):
        raise ValueError(
            "Every lower coefficient bound must be smaller than its upper bound."
        )

    if initial_array is None:
        initial = 0.5 * lower + 0.5 * upper
    else:
        initial = initial_array.astype(dtype, copy=False)
    if np.any((initial < lower) | (initial > upper)):
        raise ValueError("initial_coefficients lie outside coefficient_bounds.")
    return (
        jnp.asarray(initial),
        jnp.asarray(lower),
        jnp.asarray(upper),
    )


def search_control(
    problem: ControlProblem,
    parameterization: AbstractControlParameterization,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: Key[Array, ""],
    coefficient_bounds: CoefficientBounds,
    initial_coefficients: ArrayLike | None = None,
    **solver_options: Any,
) -> ControlSearchResult:
    """Search finite coefficient bounds for a good control candidate.

    The returned candidate is only the best one found by the configured finite
    search. Population convergence is diagnostic and is not an optimality claim.
    Supplied coefficients are validated as-is rather than clipped or repaired.
    If no initial coefficients are supplied, the exact bound midpoint is used.
    Losses over explicit problem cases are summed. Invalid or sampled-infeasible
    candidates are recorded as invalid evaluations instead of assigned a penalty.
    """
    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if not isinstance(parameterization, AbstractControlParameterization):
        raise TypeError(
            "parameterization must implement AbstractControlParameterization."
        )
    if not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be a DifferentialEvolutionSearch.")
    if parameterization.control_shape != problem.control_shape:
        raise ValueError(
            "Control parameterization control_shape does not match the problem."
        )

    coefficient_shape = problem.case_shape + parameterization.parameter_shape
    initial, lower, upper = _resolve_coefficients_and_bounds(
        coefficient_shape,
        coefficient_bounds,
        initial_coefficients,
    )
    objective = _ControlObjective(problem, parameterization, dict(solver_options))
    result = _bounded_differential_evolution(
        objective,
        jnp.reshape(initial, (-1,)),
        jnp.reshape(lower, (-1,)),
        jnp.reshape(upper, (-1,)),
        search,
        key=key,
    )
    coefficients = jnp.reshape(result.best_vector, coefficient_shape)
    evaluation = _evaluate_control(
        problem,
        parameterization,
        coefficients,
        dict(solver_options),
    )
    population_shape = (search.population_size,) + coefficient_shape

    return ControlSearchResult(
        problem=problem,
        parameterization=parameterization,
        evaluation=evaluation,
        coefficients=coefficients,
        objective=result.raw_objective,
        population_coefficients=jnp.reshape(
            result.population_vectors,
            population_shape,
        ),
        population_objectives=result.population_objectives,
        best_objective_history=result.best_objective_history,
        lower_bounds=jnp.reshape(result.lower_bounds, coefficient_shape),
        upper_bounds=jnp.reshape(result.upper_bounds, coefficient_shape),
        key=result.key,
        search=result.search,
        population_converged=result.converged,
        termination_reason=result.termination_reason,
        generations=result.generations,
        objective_evaluations=result.objective_evaluations,
        invalid_candidates=result.invalid_evaluations,
        design_signature=result.design_signature,
    )


__all__ = ["CoefficientBounds", "ControlSearchResult", "search_control"]
